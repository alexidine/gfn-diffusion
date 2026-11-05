"""checkpointing.Checkpointer.save_buffers skips the eval-cadence write when no resident store changed since the last
one (_buffer_fingerprint): an MLE stage's static prior buffer is GBs per write and never moves.

    python -m pytest -q tests/infra/test_buffer_save_skip.py
"""
from types import SimpleNamespace

import pytest
import torch

import energy_sampling.checkpointing as ck

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


def test_a_live_row_store_always_writes(tmp_path, monkeypatch):
    c, writes = _ckpt(tmp_path, monkeypatch, prior_buffer=_store())
    c.modeller.row_store = object()
    c.save_buffers()
    c.save_buffers()
    assert len(writes) == 2
