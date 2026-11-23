"""Dataset growth on the crystal store (train.py Modeller._store_grow, _growth_kicks_due), on a Modeller built without
__init__ around a real CrystalStore, a stub kick and a stub energy.

    python -m pytest -q tests/crystal/test_store_growth.py
"""
from types import SimpleNamespace

import pytest
import torch

from energy_sampling import crystal_store as CS
from energies.molecular_crystal import OOM_RECOVERY_KEEP_CACHE
from energy_sampling.buffer import ORIGIN_ROLLOUT
from energy_sampling.tests.crystal import store_stub as S

pytestmark = pytest.mark.fast


class Tracker:
    def __init__(self):
        self.calls = []

    def update_best_energy(self, cid, energy, energy_phys=None):
        self.calls.append((cid.clone(), energy.clone(), None if energy_phys is None else energy_phys.clone()))


def grown(monkeypatch, n=6, **kick):
    m = S.build()
    m.condition_log_z = Tracker()
    S.shift_kick(monkeypatch, **kick)
    fwd, batch = S.rollout(torch.arange(n) % 4)
    m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    return m


def test_kicks_equal_the_rollout_rows_and_each_is_a_linked_child(monkeypatch):
    m = grown(monkeypatch)
    st = m.row_store
    kids = st.stats()['store/children']
    assert m._growth_kicks_due() == 6
    m._store_grow()
    assert m._growth_kicks_due() == 0 and st.stats()['store/children'] == kids + 6
    new = torch.nonzero(st.origin[:st.high_water] == CS.ORIGIN_KICK).flatten()
    assert new.numel() == 6 and bool((st.kind[new] == CS.CHILD).all())
    assert bool((st.kind[st.parent[new].long()] == CS.ANCHOR).all())
    assert torch.equal(st.cid[st.parent[new].long()], st.cid[new])
    assert bool((st.birth_step[new] == 5).all()) and bool(torch.isnan(st.log_q[new]).all())
    st.check()


def test_no_rollout_no_growth(monkeypatch):
    m = S.build()
    m.condition_log_z = Tracker()
    calls = S.shift_kick(monkeypatch)
    m._store_grow()
    assert not calls and not m.energy_function.calls


def test_the_stored_row_is_the_scored_point_and_reward(monkeypatch):
    m = grown(monkeypatch)
    st = m.row_store
    m._store_grow()
    scored = m.energy_function.calls[-1]
    new = torch.nonzero(st.origin[:st.high_water] == CS.ORIGIN_KICK).flatten()
    assert torch.equal(st.x[new], scored.float())
    cid = st.cid[new].float()
    assert torch.allclose(st.log_r[new], -(scored ** 2).sum(-1) - cid / 10)
    assert torch.allclose(st.energy[new], (scored ** 2).sum(-1) + cid)


def test_a_kick_that_kept_its_anchor_spends_no_energy_call_and_stores_nothing(monkeypatch):
    m = grown(monkeypatch, keep=lambda c: torch.ones(len(c), dtype=torch.bool))
    st = m.row_store
    kids = st.stats()['store/children']
    m._store_grow()
    assert not m.energy_function.calls and st.stats()['store/children'] == kids
    assert m._growth_kicks_due() == 0 and m._store_growth['unchanged'] == 6


def test_the_window_reads_the_frozen_floor_not_the_falling_minimum(monkeypatch):
    m = grown(monkeypatch)
    st = m.row_store
    st.best_energy[:] = -1e6                                # a minimum far below the load floor
    m._store_grow()
    new = torch.nonzero(st.origin[:st.high_water] == CS.ORIGIN_KICK).flatten()
    assert not bool(((st.flags[new] & CS.UNDRAWN) != 0).any()) and m._store_growth['in_window'] == 6


def test_a_kick_above_the_window_is_stored_undrawn(monkeypatch):
    m = grown(monkeypatch)
    st = m.row_store
    m.energy_function.extra = 200.0                         # seed energy past floor 100 + 10 kT x T 2
    drawable = st.n_drawable(CS.CHILD)
    m._store_grow()
    new = torch.nonzero(st.origin[:st.high_water] == CS.ORIGIN_KICK).flatten()
    assert new.numel() == 6 and bool(((st.flags[new] & CS.UNDRAWN) != 0).all())
    assert st.n_drawable(CS.CHILD) == drawable and m._store_growth['in_window'] == 0
    assert float(st.best_energy[int(st.cid[new[0]])]) <= float(st.energy[new[0]])
    st.check()


def test_a_kick_outside_the_packing_window_is_stored_undrawn(monkeypatch):
    m = grown(monkeypatch)
    st = m.row_store
    m.energy_function.packing = 0.95
    m._store_grow()
    new = torch.nonzero(st.origin[:st.high_water] == CS.ORIGIN_KICK).flatten()
    assert bool(((st.flags[new] & CS.UNDRAWN) != 0).all())


def test_both_tracker_minima_take_the_scored_rows(monkeypatch):
    m = grown(monkeypatch)
    phys = {}

    def anchor_energy(batch, energy):
        phys['e'] = energy - 1.0
        return energy, phys['e']
    m._anchor_energy = anchor_energy
    m._store_grow()
    cid, energy, energy_phys = m.condition_log_z.calls[-1]
    assert cid.numel() == 6 and torch.allclose(energy_phys, energy - 1.0)
    new = torch.nonzero(m.row_store.origin[:m.row_store.high_water] == CS.ORIGIN_KICK).flatten()
    assert torch.allclose(energy, -m.row_store.log_r[new] * S.T)


def test_a_non_finite_kick_is_counted_and_never_reaches_a_minimum(monkeypatch):
    m = grown(monkeypatch)
    real = m.energy_function.log_reward

    def nan_first(*a, **kw):
        log_r, scored = real(*a, **kw)
        log_r = log_r.clone()
        log_r[0] = float('nan')
        return log_r, scored
    m.energy_function.log_reward = nan_first
    m._store_grow()
    assert m.row_store.nonfinite_rows == 1 and m._store_growth['scored'] == 5
    assert m.condition_log_z.calls[-1][0].numel() == 5
    assert bool(torch.isfinite(m.condition_log_z.calls[-1][1]).all())


def test_growth_keeps_chunked_recovery_without_the_end_of_call_flush(monkeypatch):
    m = grown(monkeypatch)
    seen = []
    score = m.energy_function.log_reward

    def spy(*a, internal_oom_recovery=None, **k):
        seen.append(internal_oom_recovery)
        return score(*a, internal_oom_recovery=internal_oom_recovery, **k)
    m.energy_function.log_reward = spy
    m._store_grow()
    assert seen == [OOM_RECOVERY_KEEP_CACHE]


def test_store_best_energy_median_reads_the_physical_minimum_over_visited_conditions():
    m = S.build()
    m._condition_group_axes = lambda: ([2], 1)
    st = m.row_store
    st.best_energy[:] = torch.tensor([1.0, 3.0, 10.0, float('inf')])
    assert m._store_metrics()['store/best_energy_median'] == pytest.approx(3.0)
    st.best_energy[:] = float('inf')
    assert 'store/best_energy_median' not in m._store_metrics()
