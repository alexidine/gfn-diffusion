"""checkpointing.Checkpointer.save_buffers skips the eval-cadence write when no resident store changed since the last
one (_buffer_fingerprint): an MLE stage's static prior buffer is GBs per write and never moves.

    python -m pytest -q tests/infra/test_buffer_save_skip.py
"""
from types import SimpleNamespace

import pytest
import torch

import energy_sampling.checkpointing as ck
from energy_sampling.crystal_store import ANCHOR, CHILD, ONPOLICY, ORIGIN_ROLLOUT, CrystalStore
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

pytestmark = pytest.mark.fast


def _store(n=5):
    return SimpleNamespace(x=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3), y=torch.zeros(n),
                           ema_loss=torch.zeros(n), select_counts=torch.zeros(n), birth_step=torch.zeros(n),
                           birth_loss=torch.zeros(n), ema_logw=torch.zeros(n), traj=None)


def _ckpt(tmp_path, monkeypatch, **stores):
    writes = []
    monkeypatch.setattr(ck, 'atomic_save', lambda obj, path: (writes.append(path), open(path, 'wb').close()))
    c = object.__new__(ck.Checkpointer)
    c.modeller = SimpleNamespace(row_store=None, args=SimpleNamespace(checkpoint_read_only=False), **stores)
    c.buffers_path = lambda tag=None: str(tmp_path / (f'run_{tag}_buffers.pt' if tag else 'run_buffers.pt'))
    c.buffer_state = lambda: {}
    return c, writes


def test_an_unchanged_store_is_written_once(tmp_path, monkeypatch, capsys):
    c, writes = _ckpt(tmp_path, monkeypatch, prior_buffer=_store())
    c.save_buffers()
    c.save_buffers()
    c.save_buffers()
    assert len(writes) == 1 and 'buffers unchanged' in capsys.readouterr().out


def test_nan_columns_do_not_read_as_a_change(tmp_path, monkeypatch):
    """A seeded row's birth_log_pf is NaN by design, and NaN != NaN: the first fingerprint never matched."""
    s = _store()
    s.birth_loss[1] = float('nan')
    c, writes = _ckpt(tmp_path, monkeypatch, prior_buffer=s)
    c.save_buffers()
    c.save_buffers()
    assert len(writes) == 1
    s.birth_loss[2] = float('nan')    # one more NaN is a change
    c.save_buffers()
    assert len(writes) == 2


@pytest.mark.parametrize('change', ['row', 'draw', 'loss'])
def test_any_change_to_a_store_writes_again(tmp_path, monkeypatch, change):
    s = _store()
    c, writes = _ckpt(tmp_path, monkeypatch, prior_buffer=s)
    c.save_buffers()
    if change == 'row':
        s.x = torch.cat([s.x, torch.ones(1, 3)])
    elif change == 'draw':
        s.select_counts[2] += 1
    else:
        s.ema_loss[0] = 0.5
    c.save_buffers()
    assert len(writes) == 2


def test_tagged_writes_always_happen_and_a_missing_file_is_rewritten(tmp_path, monkeypatch):
    c, writes = _ckpt(tmp_path, monkeypatch, prior_buffer=_store())
    c.save_buffers()
    c.save_buffers('phase1_exit')
    c.save_buffers('phase1_exit')
    assert len(writes) == 3
    (tmp_path / 'run_buffers.pt').unlink()
    c.save_buffers()
    assert len(writes) == 4


def test_a_row_store_without_a_generation_always_writes(tmp_path, monkeypatch):
    c, writes = _ckpt(tmp_path, monkeypatch, prior_buffer=_store())
    c.modeller.row_store = object()
    c.save_buffers()
    c.save_buffers()
    assert len(writes) == 2


def _crystal_store():
    b = collate_data_list([MolData(z=torch.tensor([6, 7, 8], dtype=torch.long), pos=torch.eye(3, dtype=torch.float32),
                                   x=torch.zeros((3, 1), dtype=torch.float32), identifier='m0',
                                   mol_volume=torch.tensor(55.0), mass=torch.tensor(42.0), radius=torch.tensor(1.2),
                                   z_prime=torch.tensor(1, dtype=torch.long))])
    b.add_graph_attr(torch.zeros(1, dtype=torch.long), 'sg_ind')
    st = CrystalStore(3, 1, 200, 10, seed=1)
    kind = torch.tensor([ANCHOR] + [CHILD] * 30)
    parent = torch.tensor([-1] + [0] * 30)
    st.seed(torch.rand(31, 3), torch.zeros(31, dtype=torch.long), kind, torch.zeros(31), torch.ones(31), parent,
            torch.zeros(31, dtype=torch.bool), b, [0], torch.zeros(1, dtype=torch.float64))
    return st


def _add(st, durable=True):
    return st.add(torch.zeros(2, 3), [0, 0], ONPOLICY, ORIGIN_ROLLOUT, torch.zeros(2), torch.ones(2), 3,
                  log_q=torch.zeros(2), resid=torch.zeros(2), durable=durable)


def test_an_unchanged_crystal_store_is_written_once_and_draws_do_not_count(tmp_path, monkeypatch):
    st = _crystal_store()
    c, writes = _ckpt(tmp_path, monkeypatch)
    c.modeller.row_store = st
    c.save_buffers()
    st.draw(CHILD, 8)
    st.draw_anchors(8)
    c.save_buffers()
    assert len(writes) == 1


@pytest.mark.parametrize('change', ['add', 'touch', 'thin', 'rewind'])
def test_a_durable_crystal_store_change_writes_again(tmp_path, monkeypatch, change):
    st = _crystal_store()
    st.touch(st.draw(CHILD, 4), st.epoch, torch.zeros(4), torch.zeros(4), 5)
    c, writes = _ckpt(tmp_path, monkeypatch)
    c.modeller.row_store = st
    c.save_buffers()
    if change == 'add':
        _add(st)
    elif change == 'touch':
        st.touch(st.draw(CHILD, 4), st.epoch, torch.zeros(4), torch.zeros(4), 6)
    elif change == 'thin':
        st.thin(2)
    else:
        assert st.invalidate_q_after(1) > 0
    c.save_buffers()
    assert len(writes) == 2


def test_a_failed_crystal_store_write_is_retried(tmp_path, monkeypatch):
    st = _crystal_store()
    c, writes = _ckpt(tmp_path, monkeypatch)
    c.modeller.row_store = st
    c.save_buffers()
    _add(st)
    monkeypatch.setattr(ck, 'atomic_save', lambda obj, path: writes.append(path))    # leaves the old file in place
    c.save_buffers()
    assert len(writes) == 2
    c.save_buffers()
    assert len(writes) == 3


def test_a_non_durable_add_skips_the_rolling_write_and_a_tagged_write_carries_it(tmp_path, monkeypatch):
    st = _crystal_store()
    c, writes = _ckpt(tmp_path, monkeypatch)
    c.modeller.row_store = st
    c.save_buffers()
    _add(st, durable=False)
    c.save_buffers()
    assert len(writes) == 1
    c.save_buffers('phase1_exit')
    assert len(writes) == 2
