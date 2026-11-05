"""Compact buffer sidecar entries (checkpointing.Checkpointer._store_state, .apply_compact_store;
train.py::Modeller.settle_seeded_store).

A store whose rows are still exactly the rows this launch's seed built is written WITHOUT its rows' graphs -- its
per-row records and the seed's row digest -- and a resume rebuilds the rows from the prior through the same seed,
then restores the records onto them. Contracts, on real CrystalBuffers:
- the compact entry carries no graphs and is a small fraction of the full one;
- a round trip restores every record onto the rebuilt rows, row for row;
- rows that are not the ones the records belong to are refused;
- a store whose rows changed after its seed, or that no seed built, is written in full;
- a restore before the seed parks the entry and settle_seeded_store applies it after the seed.

    python -m pytest -q tests/crystal/test_compact_store_sidecar.py
"""
import importlib.util
import io
import os
from types import SimpleNamespace

import pytest
import torch

from energy_sampling import noised_prior
from energy_sampling.checkpointing import Checkpointer
from energy_sampling.train import Modeller

_spec = importlib.util.spec_from_file_location(
    'buffer_columns_helpers', os.path.join(os.path.dirname(__file__), 'test_buffer_columns.py'))
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)

pytestmark = pytest.mark.fast
N = 6
RECORDS = ('ema_loss', 'select_counts', 'birth_step', 'birth_loss', 'birth_log_pf', 'is_val', 'origin', 'ema_logw',
           'ema_logw_sq', 'ema_log_z_emp')


def _seeded(seed=0):
    return _h._buffer(N, with_traj=False, seed=seed)


def _modeller(buf=None, seeded=True):
    m = SimpleNamespace(buffer_device=_h.CPU, problem_def={'p': 1}, problem_hash='h', step_ind=7, row_store=None,
                        args=SimpleNamespace(checkpoint_read_only=False))
    if buf is not None:
        m.prior_buffer = buf
        if seeded:
            m._seed_digests = {'prior_buffer': noised_prior.digest(buf.x)}
    ck = object.__new__(Checkpointer)
    ck.modeller = m
    m.checkpointer = ck
    return m, ck


def _nbytes(state):
    bio = io.BytesIO()
    torch.save(state, bio)
    return bio.tell()


def _train_a_little(buf):
    buf.select_counts[1] += 3
    buf.ema_loss[2] = 7.5
    buf.ema_logw[4] = -1.25


def test_a_seeded_store_is_written_without_its_graphs():
    buf = _seeded()
    _, ck = _modeller(buf)
    compact = ck._store_state('prior_buffer')
    assert compact['batch'] is None and compact['x'] is None and compact['compact_seed']
    assert _nbytes(compact) < 0.5 * _nbytes(buf.state_dict())


def test_a_round_trip_restores_every_record_onto_the_rebuilt_rows():
    buf = _seeded()
    _train_a_little(buf)
    _, ck = _modeller(buf)
    state = ck.buffer_state()
    rebuilt = _seeded()                       # the next launch's seed: the same rows, fresh records
    m2, ck2 = _modeller(rebuilt)
    ck2.restore_buffers(state, 'test')
    out = m2.prior_buffer
    assert torch.equal(out.x, buf.x) and out.batch.num_graphs == N
    for col in RECORDS:
        a, b = getattr(out, col), getattr(buf, col)
        assert torch.equal(torch.nan_to_num(a.float(), 99.0), torch.nan_to_num(b.float(), 99.0)), col


def test_rows_that_are_not_the_records_rows_are_refused():
    _, ck = _modeller(_seeded(seed=0))
    state = ck.buffer_state()
    _, ck2 = _modeller(_seeded(seed=1))       # another prior (or window, or seed): other rows
    with pytest.raises(ValueError, match='digest'):
        ck2.restore_buffers(state, 'test')


def test_a_store_whose_rows_changed_or_no_seed_built_is_written_in_full():
    buf = _seeded()
    _, ck = _modeller(buf)
    buf.add(_seeded(seed=2).batch)            # an admission after the seed
    assert ck._store_state('prior_buffer')['batch'] is not None
    _, ck2 = _modeller(_seeded(), seeded=False)
    assert ck2._store_state('prior_buffer')['batch'] is not None


def test_a_restore_before_the_seed_parks_and_the_seed_applies_it():
    buf = _seeded()
    _train_a_little(buf)
    _, ck = _modeller(buf)
    state = ck.buffer_state()
    m2, ck2 = _modeller()                     # init order: the sidecar is read before any buffer exists
    ck2.restore_buffers(state, 'test')
    assert not hasattr(m2, 'prior_buffer') and 'prior_buffer' in m2._compact_records
    m2.prior_buffer = _seeded()               # init_prior_buffer_seed
    Modeller.settle_seeded_store(m2, 'prior_buffer', seeded=True)
    assert torch.equal(m2.prior_buffer.select_counts, buf.select_counts)
    assert m2._seed_digests['prior_buffer'] == state['prior_buffer']['compact_seed']
    assert ck2._store_state('prior_buffer')['batch'] is None   # and it stays compact on the next write


def test_a_parked_entry_with_no_store_to_receive_it_raises():
    _, ck = _modeller(_seeded())
    state = ck.buffer_state()
    m2, ck2 = _modeller()
    ck2.restore_buffers(state, 'test')
    with pytest.raises(ValueError, match='built no prior_buffer'):
        Modeller.settle_seeded_store(m2, 'prior_buffer', seeded=True)
