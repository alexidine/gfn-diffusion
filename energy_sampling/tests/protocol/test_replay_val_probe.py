"""The held-out probe: Modeller._replay_val_size / _replay_val_stats.

replay/val_gap is a DIFFERENCE of two means taken inside one step, so what has
to be true of the probe is that it changes nothing the training step then reads.
Two ways it could: by leaving `_stash_live_branches` on (get_gfn_backward_loss
overwrites gfn._live_bwd under that flag, and fused_train_step's pooled-VarGrad
block reads it later in the SAME step), and by letting an OOM reach
handle_train_epoch_error, which would cut the global batch size off a diagnostic.

The size rule is the other half: the probe is clipped to the live batch_size as
well as to val_cap, and it refuses to publish at all below val_min, where the
mean is noise that would poison the EMA it is read on.
"""
from types import MethodType, SimpleNamespace

import pytest
import torch

from energy_sampling.train import Modeller


class _OOM(RuntimeError):
    pass


def _m(n_val=200, val_frac=0.1, val_cap=256, val_min=64, batch_size=400,
       losses=None, raise_oom=False, stash=True):
    cfg = SimpleNamespace(val_frac=val_frac, val_cap=val_cap, val_min=val_min)
    m = SimpleNamespace(
        args=SimpleNamespace(buffers=SimpleNamespace(replay_buffer=cfg)),
        batch_size=batch_size,
        replay_buffer=SimpleNamespace(is_val=torch.ones(n_val, dtype=torch.bool)),
        gfn_model=SimpleNamespace(_stash_live_branches=stash),
        seen={})
    m.mode_repeats = lambda mode: 1

    def _step(discretizer, repeats, report_losses, side_effects, val_rows):
        m.seen['val_rows'] = val_rows
        m.seen['side_effects'] = side_effects
        m.seen['stash_during'] = m.gfn_model._stash_live_branches
        if raise_oom:
            raise _OOM('CUDA out of memory. Tried to allocate 2 GiB')
        return None, {'losses': torch.as_tensor(losses)}

    m.replay_train_step = _step
    m._replay_val_frac = MethodType(Modeller._replay_val_frac, m)
    m._replay_val_size = MethodType(Modeller._replay_val_size, m)
    m._replay_val_stats = MethodType(Modeller._replay_val_stats, m)
    return m


# ------------------------------------------------------------------ the size

@pytest.mark.parametrize('kw,expected', [
    (dict(n_val=200), 200),                          # pool binds
    (dict(n_val=1000), 256),                         # val_cap binds
    (dict(n_val=1000, batch_size=100), 100),         # an OOM-cut batch binds
    (dict(n_val=63), 0),                             # under val_min: no reading
    (dict(n_val=200, val_frac=0.0), 0),              # split off
])
def test_the_probe_size(kw, expected):
    assert _m(**kw)._replay_val_size() == expected


def test_no_buffer_means_no_probe():
    m = _m()
    del m.replay_buffer
    assert m._replay_val_size() == 0


# --------------------------------------------------------------- the measure

def test_the_gap_is_val_minus_the_step_s_own_training_loss():
    m = _m(losses=[1.0, 3.0, 2.0, 2.0])
    st = m._replay_val_stats(discretizer=None, train_loss=1.5)
    assert st['val_loss'] == pytest.approx(2.0)
    assert st['val_gap'] == pytest.approx(0.5)
    assert st['val_n'] == 4.0
    assert st['val_skips'] == 0.0
    assert m.seen['side_effects'] is False, 'the probe must not write to the buffer'
    assert m.seen['val_rows'] == 200


def test_the_probe_does_not_leak_into_the_pooled_vargrad_term():
    m = _m(losses=[1.0], stash=True)
    m._replay_val_stats(discretizer=None, train_loss=0.0)
    assert m.seen['stash_during'] is False, 'gfn._live_bwd would carry the probe'
    assert m.gfn_model._stash_live_branches is True, 'and it must be restored'


def test_an_oom_skips_the_measurement_rather_than_cutting_the_batch():
    m = _m(raise_oom=True)
    assert m._replay_val_stats(discretizer=None, train_loss=0.0) == {'val_skips': 1.0}
    assert m.gfn_model._stash_live_branches is True


def test_a_non_oom_failure_still_raises():
    m = _m(losses=[1.0])

    def _boom(**kwargs):
        raise ValueError('something else entirely')

    m.replay_train_step = lambda *a, **k: _boom()
    with pytest.raises(ValueError, match='something else'):
        m._replay_val_stats(discretizer=None, train_loss=0.0)
    assert m.gfn_model._stash_live_branches is True


def test_nothing_is_published_while_the_pool_is_below_val_min():
    """Not even a skip: 'the pool has not filled' is a different statement from
    'the measurement failed', and replay_buffer_val_rows already says it."""
    assert _m(n_val=10)._replay_val_stats(discretizer=None, train_loss=0.0) == {}
