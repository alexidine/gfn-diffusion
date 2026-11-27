"""fwd_loss_coeffs.z_level_alpha (gflownet_losses.z_level_target; config_invariants.z_level_alpha_is_read): the
condition-grouped emp_z regresses Z(c) onto the forward level plus alpha x the tracker's J_B(c) - J_F(c). What each test
would catch:

  * a shift at alpha 0, or one applied where the tracker does not trust both levels;
  * a shift of the wrong size or sign, or one that carries gradient;
  * the key accepted outside [0, 1], or where no condition-grouped emp_z reads it.

CPU, a stub tracker.

    python -m pytest -q tests/losses/test_z_level_alpha.py
"""
import copy
from types import SimpleNamespace

import pytest
import torch

from energy_sampling import config_invariants as ci
from energy_sampling.gflownet_losses import z_level_target

pytestmark = pytest.mark.fast


class Tracker:
    """lookup_delta: J_B(c) - J_F(c) = 10 + c, trusted for even c only."""

    def lookup_delta(self, condition_id):
        c = torch.as_tensor(condition_id).long()
        return (10.0 + c).double(), (c % 2 == 0)


def test_alpha_zero_or_no_tracker_leaves_the_target_alone():
    rows, cid = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0, 1, 2])
    for alpha, tr in ((0.0, Tracker()), (0.25, None), (None, Tracker())):
        got, shift = z_level_target(rows, tr, cid, alpha)
        assert got is rows and shift is None


def test_the_target_moves_alpha_of_the_gap_on_trusted_conditions_only():
    rows = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    cid = torch.tensor([0, 1, 2, 3])
    got, shift = z_level_target(rows, Tracker(), cid, 0.25)
    assert torch.allclose(shift, torch.tensor([2.5, 0.0, 3.0, 0.0]))
    assert torch.allclose(got.detach(), rows.detach() + shift)
    assert not shift.requires_grad and got.dtype == rows.dtype
    full, _ = z_level_target(rows, Tracker(), cid, 1.0)              # alpha 1: the backward level
    assert torch.allclose(full.detach()[0], torch.tensor(11.0))


def _cfg(alpha, emp_z=1.0, vg=1.0):
    stage = {'name': 'tb', 'loss_coeffs': {'fwd': {'z_level_alpha': alpha, 'emp_z': emp_z, 'vg_by_condition': vg}}}
    return {'protocol': 'p', 'protocols': {'p': {'stages': [stage]}}}, stage


@pytest.mark.parametrize('alpha', [0.0, 0.25, 1.0])
def test_alpha_in_range_on_a_grouped_emp_z_passes(monkeypatch, alpha):
    cfg, stage = _cfg(alpha)
    monkeypatch.setattr(ci, 'active_stages', lambda c: [stage])
    assert ci.z_level_alpha_is_read(cfg) == []


@pytest.mark.parametrize('alpha', [-0.1, 1.5])
def test_alpha_outside_zero_one_is_refused(monkeypatch, alpha):
    cfg, stage = _cfg(alpha)
    monkeypatch.setattr(ci, 'active_stages', lambda c: [stage])
    assert any('outside [0, 1]' in v.message if hasattr(v, 'message') else 'outside [0, 1]' in str(v)
               for v in ci.z_level_alpha_is_read(cfg))


@pytest.mark.parametrize('emp_z, vg', [(0.0, 1.0), (1.0, 0.0)])
def test_alpha_without_a_grouped_emp_z_is_a_dead_key(monkeypatch, emp_z, vg):
    cfg, stage = _cfg(0.25, emp_z=emp_z, vg=vg)
    monkeypatch.setattr(ci, 'active_stages', lambda c: [stage])
    assert any('dead key' in str(v) for v in ci.z_level_alpha_is_read(cfg))
