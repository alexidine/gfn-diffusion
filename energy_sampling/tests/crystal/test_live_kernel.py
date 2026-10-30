"""noised_prior.live_kernel through noised_prior.settings and Modeller._live_kernel_draw, on a Modeller built without
__init__ and a hand-written kernel table. What each test would catch:

  * an unknown kernel, or a non-positive / non-finite tau or width cap, passing settings;
  * a draw leaving [-1, 1] on a bounded row, or a wrapped row not folded onto its period;
  * a dead row moved;
  * a row drawn with another row's kernel (the table is indexed by the loader's row indices).

CPU, no model or energy.

    python -m pytest -q tests/crystal/test_live_kernel.py
"""
import math
from types import SimpleNamespace

import pytest
import torch

from energy_sampling import noised_prior
from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast


def _args(**block):
    return SimpleNamespace(noised_prior=SimpleNamespace(**block))


@pytest.mark.parametrize('block', [dict(live_kernel='gauss'), dict(live_tau=0.0), dict(live_tau=math.nan),
                                   dict(live_width_cap=-1.0), dict(live_width_cap=True), dict(live_tau='1')])
def test_malformed_live_settings_are_refused(block):
    with pytest.raises(ValueError, match='live_'):
        noised_prior.settings(_args(**block))


def test_absent_live_keys_are_off():
    s = noised_prior.settings(_args())
    assert s['live_kernel'] is None and s['live_tau'] == 1.0


def _modeller(L, wrap, live):
    m = Modeller.__new__(Modeller)
    m.gfn_model = SimpleNamespace(live_idx=torch.tensor(live), ang_mask=torch.tensor(wrap))
    m._live_L = L
    return m


def test_draws_stay_in_the_box_wrap_folds_and_dead_rows_hold():
    torch.manual_seed(0)
    # rows 0..2 live (row 1 wrapped), row 3 dead; widths far past the box so every fold is exercised
    L = 5.0 * torch.eye(3).expand(2, 3, 3).clone()
    m = _modeller(L, wrap=[False, True, False, False], live=[0, 1, 2])
    x = torch.tensor([[0.9, 0.9, -0.9, 0.42]]).repeat(4000, 1)
    y = m._live_kernel_draw(x, torch.zeros(4000, dtype=torch.long))
    assert bool((y[:, :3].abs() <= 1.0).all())
    assert torch.equal(y[:, 3], x[:, 3])
    # the reflected bounded rows and the folded wrapped row each spread over the whole box
    for j in range(3):
        assert float(y[:, j].min()) < -0.9 and float(y[:, j].max()) > 0.9


def test_each_row_is_drawn_with_its_own_kernel():
    torch.manual_seed(0)
    L = torch.stack([torch.zeros(2, 2), 1e-2 * torch.eye(2)])
    m = _modeller(L, wrap=[False, False], live=[0, 1])
    x = torch.zeros(1000, 2)
    inds = torch.arange(1000) % 2
    y = m._live_kernel_draw(x, inds)
    assert torch.equal(y[inds == 0], x[inds == 0])
    assert 0.005 < float(y[inds == 1].std()) < 0.02
