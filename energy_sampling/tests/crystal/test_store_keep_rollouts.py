"""buffers.store.keep_rollouts false (train.py Modeller._store_admit; config_invariants.row_store_problems): a training
rollout writes no store row, owes no growth kick, still feeds the eval's trailing pool and still lowers best_energy; eval
and bootstrap rows are written as before. On a Modeller built without __init__ around a real CrystalStore. CPU.

    python -m pytest -q tests/crystal/test_store_keep_rollouts.py
"""
import pytest
import torch

from energy_sampling import config_invariants as ci
from energy_sampling import crystal_store as CS
from energy_sampling.tests.crystal import store_stub as S
from energy_sampling.tests.crystal.test_eval_pool import crystals, modeller, rollout_rows

pytestmark = pytest.mark.fast


def unkept(n=8):
    m = modeller(n=n)
    m._store_cfg = dict(m._store_cfg, keep_rollouts=False)
    return m


def test_a_training_rollout_writes_no_row_but_still_owes_its_kicks_and_feeds_the_eval_pool():
    m = unkept()
    st = m.row_store
    before = st.stats()
    written = m._store_admit(rollout_rows(4), crystals(4), CS.ORIGIN_ROLLOUT)
    assert written.numel() == 0
    assert st.stats()['store/onpolicy'] == before['store/onpolicy'] == 0
    assert st.stats()['store/rows'] == before['store/rows']
    assert m._growth_kicks_due() == 4
    assert m._eval_pool.rows == 4
    st.check()


@pytest.mark.parametrize('keep', [True, False])
def test_growth_every_owes_kicks_on_the_first_of_every_n_training_rollouts(keep):
    m = modeller(n=64)
    m._store_cfg = dict(m._store_cfg, keep_rollouts=keep, growth_every=3)
    owed = []
    for _ in range(7):
        m._store_admit(rollout_rows(4), crystals(4), CS.ORIGIN_ROLLOUT)
        owed.append(m._growth_kicks_due())
    assert owed == [4, 4, 4, 8, 8, 8, 12]           # rollouts 1, 4 and 7 owe
    m._store_admit(rollout_rows(4), crystals(4), CS.ORIGIN_EVAL, eval_row=True, drawable=False)
    assert m._growth_kicks_due() == 12              # an eval rollout neither owes nor advances the cadence


def test_the_unwritten_rollout_still_lowers_best_energy():
    m = unkept()
    st = m.row_store
    low = st.best_energy.clone()
    batch = crystals(4)
    batch.mol_energy = torch.tensor([-1e3, 7.0, float('nan'), 7.0])
    m._store_admit(rollout_rows(4), batch, CS.ORIGIN_ROLLOUT)
    cid = rollout_rows(4)['condition_id'].tolist()
    want = low.clone()
    want[cid[0]] = -1e3
    for c in (cid[1], cid[3]):                       # the non-finite energy (row 2) leaves its condition alone
        want[c] = min(float(low[c]), 7.0)
    assert torch.equal(st.best_energy, want)
    st.check()


@pytest.mark.parametrize('origin', [CS.ORIGIN_EVAL, CS.ORIGIN_BOOTSTRAP])
def test_eval_and_bootstrap_rows_are_still_written(origin):
    m = unkept()
    written = m._store_admit(rollout_rows(4), crystals(4), origin, eval_row=True, drawable=False)
    assert written.numel() == 4 and m.row_store.stats()['store/onpolicy'] == 4
    assert m._eval_pool.rows == 0


def test_the_default_keeps_every_rollout_row():
    m = modeller(n=8)
    assert 'keep_rollouts' not in m._store_cfg
    written = m._store_admit(rollout_rows(4), crystals(4), CS.ORIGIN_ROLLOUT)
    assert written.numel() == 4 and m._growth_kicks_due() == 4


def _store_cfg(keep, replay, growth_every=1):
    return dict(ci.ROW_STORE_DEFAULTS, enabled=True, keep_rollouts=keep, growth_every=growth_every), [
        {'name': 'tb', 'fracs': {'fwd': 0.5, 'bwd': 0.5, 'replay': replay}, 'max_steps': 10}]


@pytest.mark.parametrize('keep, replay, refused', [(False, 0.5, True), (False, 0.0, False), (True, 0.5, False)])
def test_a_replay_frac_without_kept_rollouts_is_refused(monkeypatch, keep, replay, refused):
    store, stages = _store_cfg(keep, replay)
    monkeypatch.setattr(ci, 'active_stages', lambda cfg: stages)
    msgs = [m for sev, m in ci.row_store_problems({'buffers': {'store': store}}) if sev == ci.ERROR]
    assert any('keep_rollouts false' in m for m in msgs) == refused


def test_keep_rollouts_must_be_a_boolean(monkeypatch):
    store, stages = _store_cfg('no', 0.0)
    monkeypatch.setattr(ci, 'active_stages', lambda cfg: stages)
    msgs = [m for sev, m in ci.row_store_problems({'buffers': {'store': store}}) if sev == ci.ERROR]
    assert any('keep_rollouts must be true or false' in m for m in msgs)


@pytest.mark.parametrize('every, ok', [(1, True), (20, True), (0, False), (2.5, False), (True, False)])
def test_growth_every_must_be_a_whole_number_of_at_least_one(monkeypatch, every, ok):
    store, stages = _store_cfg(True, 0.5, every)
    monkeypatch.setattr(ci, 'active_stages', lambda cfg: stages)
    msgs = [m for sev, m in ci.row_store_problems({'buffers': {'store': store}}) if sev == ci.ERROR]
    assert (not any('growth_every' in m for m in msgs)) == ok
