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
import math
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


# ---------------------------------------------------------------------------
# The nats spelling (val_gap_nats): median |resid|, val minus train.
# ---------------------------------------------------------------------------

def test_weighted_median_matches_the_plain_one_with_flat_weights():
    x = torch.tensor([5.0, 1.0, 3.0, 2.0, 4.0])
    assert Modeller._weighted_median(x) == pytest.approx(3.0)
    assert Modeller._weighted_median(x, torch.ones(5)) == pytest.approx(3.0)


def test_weighted_median_undoes_a_skewed_draw():
    """The training draw is prioritised toward high |resid|; the IS weights that
    make the MEAN unbiased for the uniform buffer must move the MEDIAN too, or
    the two sides of the gap are quantiles of different populations."""
    x = torch.tensor([1.0, 2.0, 3.0, 90.0, 95.0])       # the tail is over-drawn
    w = torch.tensor([8.0, 8.0, 8.0, 1.0, 1.0])          # ... and down-weighted
    assert Modeller._weighted_median(x) == pytest.approx(3.0)
    assert Modeller._weighted_median(x, w) == pytest.approx(2.0)


def test_weighted_median_falls_back_rather_than_dividing_by_zero():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert Modeller._weighted_median(x, torch.zeros(3)) == pytest.approx(2.0)
    assert math.isnan(Modeller._weighted_median(torch.empty(0)))


def test_weighted_median_tiles_weights_over_repeats():
    """One weight per ROW, K residuals per row -- the same tiling the loss
    reduction does. A mis-pairing here would silently weight the wrong rows."""
    x = torch.tensor([1.0, 1.0, 9.0, 9.0])               # 2 rows x 2 repeats
    w = torch.tensor([10.0, 1.0])                        # row 0 dominates
    assert Modeller._weighted_median(x, w) == pytest.approx(1.0)


def test_one_clash_row_moves_the_mean_gap_but_not_the_nats_gap():
    """The reason the nats spelling exists. A single |resid|=200 row is worth
    more than the whole measured gap in loss units, and nothing in a median."""
    clean = torch.tensor([40.0] * 200)
    dirty = torch.cat([torch.tensor([40.0] * 199), torch.tensor([200.0])])
    beta = 80.0

    def huber_mean(r):
        a = r.abs()
        return float(torch.where(a < beta, 0.5 * r ** 2, beta * (a - 0.5 * beta)).mean())

    assert huber_mean(dirty) - huber_mean(clean) > 40      # bigger than the real gap (45)
    med = Modeller._weighted_median
    assert med(dirty) - med(clean) == pytest.approx(0.0)   # the median does not notice


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs an accelerator')
def test_weighted_median_accepts_weights_from_another_device():
    """_replay_is_w is built host-side by the buffer's draw while `resid` lives on
    the accelerator. The CPU-only tests above cannot see that mismatch; it took
    down rr07_rr_n7_rat on its first live measurement at step 10."""
    x = torch.tensor([1.0, 2.0, 3.0, 90.0, 95.0], device='cuda')
    w = torch.tensor([8.0, 8.0, 8.0, 1.0, 1.0])          # CPU, deliberately
    assert Modeller._weighted_median(x, w) == pytest.approx(2.0)
