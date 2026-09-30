"""
CPU tests for `Nonthermal Fraction (equipartition)` -- train.py::equipartition_bar,
::row_dof_count and their wiring into Modeller.log_nonthermal_tail.

The bar is u*_i = Gamma(k_i/2, 1) quantile at 1 - p, plus W, with k_i the row's REAL
degree-of-freedom count: the carrier row's own width off `state_mask`, else the n_dof the
existing `Nonthermal Fraction` uses (gfn_model.live_dim, else data_ndim). The wiring half
drives the REAL Modeller methods bound onto a stub, as tests/protocol/
test_condition_fractions.py does, and checks that the existing family is untouched.
"""
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from scipy.stats import gamma

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from train import Modeller, equipartition_bar, row_dof_count  # noqa: E402

P, W = 1e-6, 10.0      # the canonical defaults
EQ = 'Nonthermal Fraction (equipartition)'
EQ_MEAN = 'Nonthermal Threshold (equipartition) Mean'


class _StubModeller:
    log_thermo_properties = Modeller.log_thermo_properties
    log_physical_properties = Modeller.log_physical_properties
    log_nonthermal_tail = Modeller.log_nonthermal_tail
    log_condition_fraction = Modeller.log_condition_fraction
    _reasonable_sample_mask = Modeller._reasonable_sample_mask
    _log_setting = Modeller._log_setting

    def __init__(self, floor, data_ndim, live_dim=None, **args):
        self._floor = floor
        self._settings_log_cache = {}
        self.gfn_model = None if live_dim is None else SimpleNamespace(live_dim=live_dim)
        self.energy_function = SimpleNamespace(data_ndim=data_ndim)
        base = dict(conditional_worst_quantile=0.25, nonthermal_entropy_per_dim=4.0,
                    nonthermal_cond_bar=0.1, reasonable_cond_bar=0.5,
                    nonthermal_equipartition_p=P, nonthermal_basin_window_kT=W)
        base.update(args)
        self.args = SimpleNamespace(**base)

    def _condition_energy_floor(self, condition_id):
        return self._floor


class _StubBatch:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def keys(self):
        return list(self.__dict__)

    def __getitem__(self, k):
        return self.__dict__[k]


def _arr(t):
    return t.cpu().detach().numpy()


def _val(t):
    return t.cpu().detach().item()


def _bar(k):
    return float(gamma.ppf(1.0 - P, k / 2.0)) + W


def _mask(widths, K):
    m = torch.zeros(len(widths), K, dtype=torch.bool)
    for i, w in enumerate(widths):
        m[i, :w] = True
    return m


# ----------------------------------------------------------------- the bar

def test_bar_matches_the_gamma_quantile():
    ks = torch.tensor([2, 12, 30, 49, 75, 12])
    bar = equipartition_bar(ks, P, W)
    # k = 2 is Exp(1): the quantile at 1 - p is -ln p, in closed form
    assert abs(bar[0].item() - (-math.log(P) + W)) < 1e-6, bar[0].item()
    for k, b in zip(ks.tolist(), bar.tolist()):
        # isf, an independent scipy path from the ppf the code calls
        assert abs(b - (gamma.isf(P, k / 2.0) + W)) < 1e-5, (k, b)
    assert bar[1].item() == bar[5].item(), 'same k, different bar'
    assert torch.all(bar[1:5].diff() > 0), 'the bar must grow with k'
    # no DoF holds no excess
    assert equipartition_bar(torch.tensor([0]), P, W).item() == W
    with pytest.raises(ValueError):
        equipartition_bar(torch.tensor([12]), 1.0, W)


def test_row_dof_count_reads_the_mask_or_falls_back():
    assert row_dof_count(None, 3, 10).tolist() == [10, 10, 10]
    assert row_dof_count(_StubBatch(), 2, 12).tolist() == [12, 12]
    m = _mask([2, 4, 6], 6)
    assert row_dof_count(_StubBatch(state_mask=m), 3, 6).tolist() == [2, 4, 6]
    # the carrier stores [1, K] per graph; a flat collation must still split per row
    assert row_dof_count(_StubBatch(state_mask=m.reshape(-1)), 3, 6).tolist() == [2, 4, 6]
    with pytest.raises(RuntimeError):
        row_dof_count(_StubBatch(state_mask=torch.ones(7, dtype=torch.bool)), 3, 6)


# ----------------------------------------------------------------- the wiring

def _run(m, energy, cid, sample_batch=None):
    metrics = {}
    m.log_nonthermal_tail(_arr, {'condition_id': cid}, torch.zeros(len(energy)),
                          -energy, metrics, sample_batch=sample_batch)
    return metrics


def test_rows_either_side_of_the_bar_are_classified():
    """Crystal-style batch at T = 1, so u = E - Emin exactly."""
    k, eps = 12, 1e-3
    b = _bar(k)
    energy = torch.tensor([b + eps, b - eps, b + 5.0, 0.0], dtype=torch.float64)
    cid = torch.tensor([1, 1, 2, 2])
    m = _StubModeller(floor=torch.zeros(4, dtype=torch.float64), data_ndim=k)
    metrics = _run(m, energy, cid)
    assert metrics[EQ] == 0.5, metrics[EQ]
    assert abs(metrics[EQ_MEAN] - b) < 1e-9
    for key in ('Cond Nonthermal (equipartition) Failing Frac',
                'Cond Nonthermal (equipartition) N', 'Cond Nonthermal (equipartition) Bar'):
        assert key in metrics, key
    assert metrics['Cond Nonthermal (equipartition) N'] == 2
    # the existing bar is 4 * 12 = 48, above every row but the b + 5 one
    assert metrics['Nonthermal Threshold'] == 48.0
    want = 0.25 if b + 5.0 > 48.0 else 0.0
    assert metrics['Nonthermal Fraction'] == want, metrics['Nonthermal Fraction']


def test_crystal_takes_live_dim():
    """No state_mask: every row gets gfn_model.live_dim, as the existing metric does."""
    m = _StubModeller(floor=torch.zeros(4), data_ndim=12, live_dim=10)
    metrics = _run(m, torch.tensor([1., 2., 3., 4.]), torch.tensor([0, 0, 1, 1]))
    assert abs(metrics[EQ_MEAN] - _bar(10)) < 1e-5, (metrics[EQ_MEAN], _bar(10))
    assert metrics['Nonthermal Threshold'] == 40.0


def test_conformer_carrier_takes_each_rows_own_width():
    """A width-6 carrier with rows owning 2, 4 and 6 columns. One excess sits above the
    k = 2 bar and below the k = 6 bar, so it is hot on the narrow row only."""
    K, widths = 6, [2, 4, 6, 2, 4, 6]
    u_mid = 0.5 * (_bar(2) + _bar(6))
    assert _bar(2) < u_mid < _bar(6)
    energy = torch.full((6,), u_mid, dtype=torch.float64)
    cid = torch.tensor([0, 1, 2, 0, 1, 2])
    batch = _StubBatch(state_mask=_mask(widths, K))
    m = _StubModeller(floor=torch.zeros(6, dtype=torch.float64), data_ndim=K, live_dim=K)
    metrics = _run(m, energy, cid, sample_batch=batch)
    hot = [u_mid > _bar(w) for w in widths]
    assert abs(metrics[EQ] - sum(hot) / len(hot)) < 1e-6, (metrics[EQ], hot)
    assert abs(metrics[EQ_MEAN] - sum(_bar(w) for w in widths) / len(widths)) < 1e-6
    # and the padded-width reading would have scored every row alike
    padded = _run(_StubModeller(floor=torch.zeros(6, dtype=torch.float64), data_ndim=K,
                                live_dim=K), energy, cid)
    assert padded[EQ] in (0.0, 1.0)
    assert padded[EQ] != metrics[EQ], 'the per-row width changed nothing'


def test_unreferenced_rows_are_dropped_with_their_widths():
    """The `seen` subset applies to k too: row 0 has no record and the narrowest width."""
    K = 6
    floor = torch.tensor([float('inf'), 0., 0.], dtype=torch.float64)
    energy = torch.tensor([100., 1., 1.], dtype=torch.float64)
    batch = _StubBatch(state_mask=_mask([2, 6, 6], K))
    m = _StubModeller(floor=floor, data_ndim=K, live_dim=K)
    metrics = _run(m, energy, torch.tensor([0, 1, 2]), sample_batch=batch)
    assert abs(metrics[EQ_MEAN] - _bar(6)) < 1e-6


def test_existing_family_is_unchanged():
    """Same inputs, equipartition on and off: every pre-existing key and value is identical,
    and the off switch removes exactly the new keys."""
    K = 6
    energy = torch.tensor([30., 1., 20., 5., 60., 2.], dtype=torch.float64)
    cid = torch.tensor([0, 0, 1, 1, 2, 2])
    floor = torch.zeros(6, dtype=torch.float64)
    batch = _StubBatch(state_mask=_mask([2, 4, 6, 2, 4, 6], K))

    on = _run(_StubModeller(floor=floor, data_ndim=K, live_dim=K), energy, cid, batch)
    off = _run(_StubModeller(floor=floor, data_ndim=K, live_dim=K,
                             nonthermal_equipartition_p=None), energy, cid, batch)
    no_batch = _run(_StubModeller(floor=floor, data_ndim=K, live_dim=K), energy, cid)

    new = {k for k in on if 'equipartition' in k}
    assert EQ in new and EQ_MEAN in new
    assert not any('equipartition' in k for k in off), sorted(off)
    assert set(on) - new == set(off), sorted(set(on) ^ set(off))
    for key, v in off.items():
        a, b, c = on[key], v, no_batch[key]
        if hasattr(a, 'tolist'):
            a, b, c = a.tolist(), b.tolist(), c.tolist()
        assert a == b == c, (key, a, b, c)
    assert off['Nonthermal Threshold'] == 24.0
    assert abs(off['Nonthermal Fraction'] - 2 / 6) < 1e-6


def test_log_thermo_properties_hands_the_batch_through():
    """The call site passes sample_batch, so a carrier batch reaches row_dof_count."""
    K = 6
    u_mid = 0.5 * (_bar(2) + _bar(6))
    e = torch.full((4,), -u_mid)          # reward-scale energies, bound, for the crystal block
    batch = _StubBatch(mol_energy=e, gfn_energy=e.clone(),
                       packing_coeff=torch.full((4,), 0.7),
                       reduction_en=torch.full((4,), 1e-3),
                       state_mask=_mask([2, 2, 6, 6], K))
    m = _StubModeller(floor=torch.full((4,), -2 * u_mid), data_ndim=K, live_dim=K)
    m.energy_function.energy_function = 'mol_energy'
    metrics = {}
    # log_r = -E / T at T = 1; excess over the floor is u_mid on every row
    m.log_thermo_properties(_arr, {'condition_id': torch.tensor([0, 0, 1, 1])},
                            torch.zeros(4), torch.zeros(4), -e, metrics, batch, _val)
    assert metrics[EQ] == 0.5, metrics[EQ]
