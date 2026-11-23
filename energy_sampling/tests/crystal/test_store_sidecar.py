"""CrystalStore format 3 through the buffer sidecar (crystal_store.py state_dict / apply_state,
checkpointing.Checkpointer.buffer_state / restore_buffers / restore_row_store), on real stores with real template
batches. CPU, synthetic rows, no energy computed.

    python -m pytest -q tests/crystal/test_store_sidecar.py
"""
import io
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here)):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling import crystal_store as CS  # noqa: E402
from energy_sampling.checkpointing import Checkpointer  # noqa: E402
from energy_sampling.crystal_store import (  # noqa: E402
    ANCHOR, CHILD, ONPOLICY, ORIGIN_KICK, ORIGIN_ROLLOUT, CrystalStore, CrystalStoreError, stamp_diff)
from mxtaltools.dataset_utils.data_classes import MolData  # noqa: E402
from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402

pytestmark = pytest.mark.fast
DIM = 3
STAMP = {'prior': 'a', 'table': [1.0, float('nan')], 'window': (1, 2)}


def _molecule(i):
    scale = 1.0 + 0.1 * i
    pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1]], dtype=torch.float32) * scale
    return MolData(z=torch.tensor([6, 7, 8], dtype=torch.long), pos=pos, x=torch.zeros((3, 1), dtype=torch.float32),
                   identifier=f'm{i}', mol_volume=torch.tensor(55.0 * scale ** 3), mass=torch.tensor(42.0),
                   radius=torch.tensor(1.2 * scale), z_prime=torch.tensor(1, dtype=torch.long))


def _templates(n, first=0):
    b = collate_data_list([_molecule(first + i) for i in range(n)])
    b.add_graph_attr(torch.arange(first, first + n, dtype=torch.long) % 4, 'sg_ind')
    return b


def seeded(spec=((3, 30), (2, 20), (1, 10)), max_rows=400, H=20, seed=3, shift=0.0):
    cid, kind, parent = [], [], []
    for c, (na, nk) in enumerate(spec):
        first = len(cid)
        cid += [c] * (na + nk)
        kind += [ANCHOR] * na + [CHILD] * nk
        parent += [-1] * na + [first + j % na for j in range(nk)]
    cid, kind, parent = torch.tensor(cid), torch.tensor(kind), torch.tensor(parent)
    n = cid.numel()
    x = torch.linspace(-0.9, 0.9, n * DIM).reshape(n, DIM) + shift
    e = torch.arange(n, dtype=torch.float32) + 10.0
    st = CrystalStore(DIM, len(spec), max_rows, H, seed=seed)
    st.seed(x, cid, kind, -e / 2.5, e, parent, torch.zeros(n, dtype=torch.bool), _templates(len(spec)),
            torch.arange(len(spec)), torch.zeros(len(spec), dtype=torch.float64))
    return st


def grow(st, rounds=6, first_step=1):
    """Admissions of both kinds and trained rows, with no slot freed."""
    anchors = torch.nonzero(st.kind[:st.high_water] == ANCHOR).flatten()
    for i in range(rounds):
        step = first_step + i
        cids = torch.tensor([0, 1, 2, 1])
        st.add(torch.full((4, DIM), 0.1 * i), cids, ONPOLICY, ORIGIN_ROLLOUT, torch.full((4,), -20.0),
               torch.full((4,), 50.0 - i), step, log_q=torch.full((4,), -1.0 - i), resid=torch.full((4,), 0.5))
        parents = anchors[torch.tensor([0, 3, 5])]
        st.add(torch.full((3, DIM), -0.1 * i), st.cid[parents], CHILD, ORIGIN_KICK, torch.full((3,), -24.0),
               torch.full((3,), 60.0), step, parent=parents)
        rows = st.draw(CHILD, 6)
        st.touch(rows, st.epoch, torch.full((6,), -2.0 - i), torch.full((6,), 0.25), step)


def roundtrip(state):
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=False)


def saved_bytes(state):
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getbuffer().nbytes


def assert_identical(a, b):
    """Every column byte over the occupied rows, the tables, slots, free stack, index layout and generator."""
    assert a.high_water == b.high_water and a.n_free == b.n_free
    for name, _ in CS._COLUMNS:
        ta, tb = getattr(a, name)[:a.high_water], getattr(b, name)[:b.high_water]
        assert ta.contiguous().numpy().tobytes() == tb.contiguous().numpy().tobytes(), name
    for name in CS._TABLES:
        assert getattr(a, name).numpy().tobytes() == getattr(b, name).numpy().tobytes(), name
    assert (a._free[:a.n_free] == b._free[:b.n_free]).all()
    for name in ('count', 'start', 'cap'):
        assert (getattr(a.index, name) == getattr(b.index, name)).all(), name
    assert a.index.tail == b.index.tail and a.index.compactions == b.index.compactions
    pools = np.arange(len(a.index.count))
    assert (a.index.rows_of(pools) == b.index.rows_of(pools)).all()
    for sa, sb in zip(a.active, b.active):
        assert (sa.members() == sb.members()).all()
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert a.stats() == b.stats()
    a.check()
    b.check()


def assert_same_future(a, b):
    assert torch.equal(a.draw(CHILD, 100), b.draw(CHILD, 100))
    assert torch.equal(a.draw_anchors(50), b.draw_anchors(50))
    assert torch.equal(a.draw(ONPOLICY, 50), b.draw(ONPOLICY, 50))
    for st in (a, b):
        st.add(torch.zeros(2, DIM), [0, 1], ONPOLICY, ORIGIN_ROLLOUT, torch.full((2,), -20.0), torch.full((2,), 50.0),
               90, log_q=torch.full((2,), -1.0), resid=torch.full((2,), 0.5))
    assert_identical(a, b)
    assert a.thin(4) == b.thin(4)
    assert_identical(a, b)


# ---- compact while the seed is intact ------------------------------------------------------------------------


def test_compact_round_trip_of_a_grown_store_is_byte_identical():
    st = seeded()
    grow(st)
    state = st.state_dict(STAMP, compact=True)
    assert state['seed_digest'] is not None and state['templates'] is None
    n = st.n_seed_kept
    assert state['columns']['x'].shape[0] == st.high_water - n and state['seed_columns']['log_q'].shape[0] == n
    other = seeded()
    other.apply_state(roundtrip(state))
    assert_identical(st, other)
    assert other.seed_intact and other.templates is not None
    assert_same_future(st, other)


def test_compact_holds_only_the_mutable_columns_of_seed_rows():
    st = seeded(spec=((5, 2000),), max_rows=4000, H=50)
    compact, full = saved_bytes(st.state_dict(compact=True)), saved_bytes(st.state_dict())
    assert compact < 0.5 * full


def test_a_thinned_seed_rebuilds_identically():
    spec = ((3, 200), (2, 100))
    st = seeded(spec=spec, max_rows=120, H=10)
    assert st.n_seed_kept == 110 and st.seed_intact
    st.touch(st.draw(CHILD, 8), st.epoch, torch.full((8,), -3.0), torch.full((8,), 0.1), 4)
    other = seeded(spec=spec, max_rows=120, H=10)
    assert other._digest_differs(st.seed_digest()) is None
    other.apply_state(roundtrip(st.state_dict(STAMP, compact=True)))
    assert_identical(st, other)


def test_a_changed_seed_thin_is_refused():
    spec = ((3, 200), (2, 100))
    state = seeded(spec=spec, max_rows=120, H=10, seed=3).state_dict(STAMP, compact=True)
    with pytest.raises(CrystalStoreError, match='digest'):
        seeded(spec=spec, max_rows=120, H=10, seed=4).apply_state(state)


def test_float_noise_in_recomputed_seed_values_is_not_a_mismatch():
    """x, log_r and energy are recomputed at each launch on that launch's device."""
    st = seeded()
    grow(st)
    state = roundtrip(st.state_dict(STAMP, compact=True))
    other = seeded()
    n = other.n_seed_kept
    other.x[:n] *= 1 + 2e-7
    other.log_r[:n] *= 1 + 2e-7
    other.energy[:n] *= 1 - 2e-7
    other.best_energy.fill_(float('inf'))
    other.best_energy.scatter_reduce_(0, other.cid[:n].long(), other.energy[:n], reduce='amin', include_self=True)
    other._seed_digest = None
    assert other.state_mismatch(state, STAMP) is None
    other.apply_state(state)
    other.check()
    assert other.n_rows == st.n_rows and torch.equal(other.q_step[:n], st.q_step[:n])


def test_a_condition_without_rows_or_floor_digests_equal():
    def build():
        st = CrystalStore(DIM, 2, 100, 5, seed=1)
        n = 5
        st.seed(torch.linspace(0, 1, n * DIM).reshape(n, DIM), torch.zeros(n, dtype=torch.long),
                torch.tensor([ANCHOR] + [CHILD] * 4), torch.zeros(n), torch.ones(n), torch.tensor([-1, 0, 0, 0, 0]),
                torch.zeros(n, dtype=torch.bool), _templates(1), [0],
                torch.tensor([0.0, float('inf')], dtype=torch.float64))
        return st
    a, b = build(), build()
    assert b.state_mismatch(a.state_dict(STAMP, compact=True), STAMP) is None


@pytest.mark.parametrize('column', ['x', 'log_r', 'energy', 'cid', 'parent'])
def test_a_moved_seed_column_is_a_mismatch(column):
    st = seeded()
    state = st.state_dict(STAMP, compact=True)
    other = seeded()
    if column == 'cid':
        other.cid[0] = 1
    elif column == 'parent':            # another anchor of the same condition: no per-condition mean moves
        assert int(other.parent[3]) == 0
        other.parent[3] = 1
    else:
        getattr(other, column)[:10] += 0.5
    other._seed_digest = None
    assert 'seed' in other.state_mismatch(state, STAMP)


@pytest.mark.parametrize('leave', ['thin', 'remove', 'add_conditions'])
def test_leaving_the_seed_writes_every_column(leave):
    st = seeded()
    grow(st)
    if leave == 'thin':
        st.thin(5)
    elif leave == 'remove':
        st.remove(torch.nonzero(st.kind[:st.high_water] == ONPOLICY).flatten()[:2])
    else:
        st.add_conditions(1, _templates(1, first=3))
    state = st.state_dict(STAMP, compact=True)
    assert not st.seed_intact and state['seed_digest'] is None and state['templates'] is not None
    assert state['columns']['x'].shape[0] == st.high_water
    other = seeded()
    other.apply_state(roundtrip(state))
    assert_identical(st, other)
    assert not other.seed_intact
    assert_same_future(st, other)


def test_a_full_state_keeps_compacting_off_after_a_restore():
    st = seeded()
    grow(st)
    st.thin(5)
    other = seeded()
    other.apply_state(roundtrip(st.state_dict(STAMP, compact=True)))
    assert other.state_dict(STAMP, compact=True)['seed_digest'] is None


def test_a_restored_compact_store_writes_compact_again():
    st = seeded()
    grow(st)
    other = seeded()
    other.apply_state(roundtrip(st.state_dict(STAMP, compact=True)))
    grow(other, rounds=2, first_step=20)
    again = seeded()
    again.apply_state(roundtrip(other.state_dict(STAMP, compact=True)))
    assert_identical(other, again)


# ---- refusals ------------------------------------------------------------------------------------------------


def test_a_compact_state_refuses_other_seed_rows():
    st = seeded()
    state = st.state_dict(STAMP, compact=True)
    with pytest.raises(CrystalStoreError, match='digest'):
        seeded(shift=0.01).apply_state(state)


def test_a_compact_state_refuses_a_store_that_is_not_freshly_seeded():
    st = seeded()
    state = st.state_dict(STAMP, compact=True)
    unseeded = CrystalStore(DIM, 3, 400, 20)
    with pytest.raises(CrystalStoreError, match='freshly seeded'):
        unseeded.apply_state(state)
    used = seeded()
    grow(used, rounds=1)
    with pytest.raises(CrystalStoreError, match='freshly seeded'):
        used.apply_state(state)


def test_a_changed_row_limit_is_refused():
    st = seeded()
    assert 'max_rows' in seeded(max_rows=500).state_mismatch(st.state_dict(STAMP, compact=True), STAMP)


def test_stamp_compare_reads_nan_as_equal_and_names_the_key():
    twin = {'prior': 'a', 'table': [1.0, float('nan')], 'window': [1, 2]}
    assert stamp_diff(STAMP, twin) == []
    assert stamp_diff(STAMP, dict(twin, prior='b')) == ['prior']
    assert stamp_diff(None, STAMP) == ['<whole stamp>']


# ---- rewind --------------------------------------------------------------------------------------------------


def test_a_rewind_invalidates_only_later_log_q():
    st = seeded()
    kids = torch.nonzero(st.kind[:st.high_water] == CHILD).flatten()
    early, late = kids[:3], kids[3:6]
    st.touch(early, st.epoch, torch.full((3,), -2.0), torch.full((3,), 0.1), 10)
    st.touch(late, st.epoch, torch.full((3,), -3.0), torch.full((3,), 0.2), 20)
    on_old = st.add(torch.zeros(1, DIM), [0], ONPOLICY, ORIGIN_ROLLOUT, [-20.0], [50.0], 5, log_q=[-7.0], resid=[0.5])
    on_new = st.add(torch.zeros(1, DIM), [0], ONPOLICY, ORIGIN_ROLLOUT, [-20.0], [50.0], 18, log_q=[-8.0], resid=[0.5])
    st.touch(on_old, st.epoch, [-9.0], [0.3], 19)
    gen = st.generation
    assert st.invalidate_q_after(15) == 5 and st.generation == gen + 1

    assert (st.log_q[early] == -2.0).all() and (st.q_step[early] == 10).all() and torch.isfinite(st.resid[early]).all()
    assert torch.isnan(st.log_q[late]).all() and (st.q_step[late] == -1).all() and torch.isnan(st.resid[late]).all()
    assert float(st.log_q[on_old]) == -7.0 and int(st.q_step[on_old]) == 5 and bool(torch.isnan(st.resid[on_old]))
    assert float(st.log_q[on_new]) == -8.0 and int(st.q_step[on_new]) == 18 and bool(torch.isnan(st.resid[on_new]))
    st.check()
    gen = st.generation
    assert st.invalidate_q_after(15) == 0 and st.generation == gen


# ---- through the checkpointer --------------------------------------------------------------------------------


def _modeller(store=None, stamp=STAMP, run='run_a', step=7):
    return SimpleNamespace(problem_def={'p': 1}, problem_hash='abc', step_ind=step, run_name=run, row_store=store,
                           _store_stamp=lambda: stamp, buffer_device='cpu',
                           args=SimpleNamespace(checkpoint_read_only=False))


def _sidecar(src):
    return roundtrip(Checkpointer(src).buffer_state())


def test_sidecar_round_trip_through_the_checkpointer():
    st = seeded()
    grow(st)
    src = _modeller(st, step=6)
    state = _sidecar(src)
    assert state['run_name'] == 'run_a' and state['row_store']['seed_digest'] is not None
    dst = _modeller(step=6)
    ck = Checkpointer(dst)
    ck.restore_buffers(state, 'x_buffers.pt')
    assert dst.row_store is None and dst._row_store_state is not None
    fresh = seeded()
    assert ck.restore_row_store(fresh) is True and dst._row_store_state is None
    assert_identical(st, fresh)


def test_nothing_parked_returns_false():
    dst = _modeller()
    assert Checkpointer(dst).restore_row_store(seeded()) is False


@pytest.mark.parametrize('what', ['stamp', 'digest'])
def test_an_own_run_mismatch_raises(what):
    st = seeded()
    grow(st)
    state = _sidecar(_modeller(st))
    dst = _modeller(stamp=dict(STAMP, prior='b') if what == 'stamp' else STAMP)
    ck = Checkpointer(dst)
    ck.restore_buffers(state, 'x_buffers.pt')
    with pytest.raises(ValueError, match='stamp' if what == 'stamp' else 'digest'):
        ck.restore_row_store(seeded(shift=0.0 if what == 'stamp' else 0.01))


def test_another_runs_mismatch_keeps_the_seed(capsys):
    st = seeded()
    grow(st)
    state = _sidecar(_modeller(st, run='other'))
    dst = _modeller(stamp=dict(STAMP, prior='b'))
    ck = Checkpointer(dst)
    ck.restore_buffers(state, 'x_buffers.pt')
    fresh, reference = seeded(), seeded()
    assert ck.restore_row_store(fresh) is False
    assert 'ignored' in capsys.readouterr().out
    assert_identical(fresh, reference)


def test_an_own_run_problem_mismatch_raises_and_anothers_is_ignored():
    st = seeded()
    state = _sidecar(_modeller(st))
    own = _modeller()
    own.problem_def = {'p': 2}
    with pytest.raises(ValueError, match='different problem'):
        Checkpointer(own).restore_buffers(dict(state), 'x_buffers.pt')
    other = _modeller(run='run_b')
    other.problem_def = {'p': 2}
    Checkpointer(other).restore_buffers(dict(state), 'x_buffers.pt')
    assert getattr(other, '_row_store_state', None) is None


def test_a_resume_behind_the_sidecar_undoes_later_log_q_and_rekeys_the_generator():
    st = seeded()
    grow(st, rounds=4, first_step=10)      # log q written at steps 10..13
    state = _sidecar(_modeller(st, step=13))
    dst = _modeller(step=11)
    ck = Checkpointer(dst)
    ck.restore_buffers(state, 'x_buffers.pt')
    fresh = seeded()
    assert ck.restore_row_store(fresh) is True
    hw = fresh.high_water
    written = fresh.q_step[:hw] > 11
    assert not bool((written & torch.isfinite(fresh.resid[:hw])).any())
    assert fresh.rng.bit_generator.state != st.rng.bit_generator.state
    twin = np.random.default_rng([fresh.base_seed, 11])
    assert fresh.rng.bit_generator.state == twin.bit_generator.state
    fresh.check()


def test_a_resume_at_the_sidecar_step_keeps_the_generator():
    st = seeded()
    grow(st)
    state = _sidecar(_modeller(st, step=6))
    dst = _modeller(step=6)
    ck = Checkpointer(dst)
    ck.restore_buffers(state, 'x_buffers.pt')
    fresh = seeded()
    ck.restore_row_store(fresh)
    assert fresh.rng.bit_generator.state == st.rng.bit_generator.state


# ---- size ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('thinned', [False, True])
def test_bytes_written_follow_the_occupied_rows_not_the_capacity(thinned):
    def written(max_rows):
        st = seeded(max_rows=max_rows)
        grow(st)
        if thinned:
            st.remove(torch.nonzero(st.kind[:st.high_water] == ONPOLICY).flatten()[:2])
        return saved_bytes(Checkpointer(_modeller(st)).buffer_state())
    assert written(1_000_000) < written(400) + 4096
