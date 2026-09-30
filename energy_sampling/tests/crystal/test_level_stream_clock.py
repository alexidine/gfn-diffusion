"""
The two level streams behind zmatch/delta_* decay on ONE clock.

delta(c) = bwd_level_ema(c) - fwd_level_ema(c) subtracts two EMAs fed at
different rates: bwd on every fused step, fwd only on rollout steps. Decayed once
per OWN visit, the fwd stream's lag in steps was fwd_rollout_every times the bwd
one, so whenever the levels drifted at rate v the gap read (tau_fwd - tau_bwd) * v
on top of the real one -- at every 5, half-life 200 and v = -6e-5 nats/step, about
-0.069 nats, enough to fold a 0.07 gap through zero. update_mode_level now decays
each stream by one factor per tick of a per-condition clock that ticks once per
step the condition reaches EITHER stream, so a common drift cancels.

These tests feed the tracker directly with noise-free levels, so every deviation
from the true gap is lag.

    pytest tests/crystal/test_level_stream_clock.py
"""
import os
import sys

import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling.buffer import ConditionLogZTracker  # noqa: E402

HALF_LIFE = 200.0     # the shipped condition_log_z.half_life_visits
SLOPE = -6e-5         # nats per step: a level falling 0.06 per 1k steps
GAP = 0.07            # true J_B - J_F, nats
DECAY = 0.5 ** (1.0 / HALF_LIFE)


def _feed(t, mode, cids, levels, step):
    """One batch: every row of condition c carries exactly levels[c]."""
    cids = torch.as_tensor(cids, dtype=torch.long)
    t.update_mode_level(mode, cids, levels[cids], step=step)


def test_common_drift_cancels_at_a_five_to_one_visit_ratio():
    """Single condition, bwd every step, fwd every 5th: the unconditional route at
    fwd_rollout_every 5. Read after every step past warm-up, as the reporter would
    on any step. What remains is the fwd stream's hold between rollouts, +-2 steps
    of drift (1.2e-4 nats, measured 1.21e-4); per-own-visit decay read up to
    0.064 off over the same window."""
    every, warm, n_steps = 5, 3000, 6000
    t = ConditionLogZTracker(library_size=1, half_life_visits=HALF_LIFE)
    fwd_rows, bwd_rows = torch.zeros(64, dtype=torch.long), torch.zeros(256, dtype=torch.long)
    errs, signed = [], []
    for step in range(n_steps):
        j_f = torch.tensor([SLOPE * step])
        if step % every == 0:          # the fused step runs fwd before bwd
            _feed(t, 'fwd', fwd_rows, j_f, step)
        _feed(t, 'bwd', bwd_rows, j_f + GAP, step)
        if step >= warm:
            errs.append(t.delta_stats()['mean'] - GAP)
            delta, mask = t.lookup_delta(torch.tensor([0]))
            assert bool(mask[0])
            signed.append(float(delta[0]))
    assert len(errs) == n_steps - warm
    assert max(abs(e) for e in errs) < 2e-4, max(abs(e) for e in errs)
    assert min(signed) > 0.0, 'the signed gap crossed zero under a common drift'
    # the fwd stream ticked the shared clock `every` times per update, bwd once
    assert int(t.level_clock[0]) == n_steps
    assert int(t.fwd_level_visits[0]) == n_steps // every
    assert int(t.bwd_level_visits[0]) == n_steps


def test_common_drift_cancels_on_a_sparsely_drawn_library():
    """Conditional shape: 16 conditions at different levels, bwd draws 8 rows per
    step (~40% of conditions visited), fwd draws 64 rows every 20th step. The
    per-condition visit ratio is ~8:1 and irregular. Per-own-visit decay read a
    mean 0.043 nats low over this window; the shared clock leaves the hold between
    rollouts and the irregular revisits (measured mean -6e-5, worst 7.4e-4)."""
    lib, every, warm, n_steps = 16, 20, 4000, 7000
    g = torch.Generator().manual_seed(0)
    offset = torch.linspace(-2.0, 2.0, lib)
    t = ConditionLogZTracker(library_size=lib, half_life_visits=HALF_LIFE)
    errs = []
    for step in range(n_steps):
        j_f = offset + SLOPE * step
        if step % every == 0:
            _feed(t, 'fwd', torch.randint(lib, (64,), generator=g), j_f, step)
        _feed(t, 'bwd', torch.randint(lib, (8,), generator=g), j_f + GAP, step)
        if step >= warm:
            d = t.delta_stats()
            assert d['n'] == lib
            errs.append(d['mean'] - GAP)
    mean_err = sum(errs) / len(errs)
    assert abs(mean_err) < 3e-4, mean_err
    assert max(abs(e) for e in errs) < 1.5e-3, max(abs(e) for e in errs)


def test_a_sparse_condition_keeps_its_half_life_in_its_own_steps():
    """The clock is per condition, not the global step: a condition revisited only
    every 50 steps must decay ONE factor per revisit (the reason update() keys
    decay to own visits), not 50."""
    t = ConditionLogZTracker(library_size=4, half_life_visits=HALF_LIFE)
    lv = torch.zeros(4)
    for step in (0, 50, 100):
        _feed(t, 'bwd', torch.zeros(10, dtype=torch.long), lv, step)
        _feed(t, 'fwd', torch.zeros(10, dtype=torch.long), lv, step)
    expect = 10.0 * (1 + DECAY + DECAY ** 2)
    for eff in (t.bwd_level_effective_count[0], t.fwd_level_effective_count[0]):
        assert float(eff) == pytest.approx(expect, rel=1e-6)
    assert int(t.level_clock[0]) == 3


def test_one_tick_per_step_and_a_same_step_refeed_does_not_decay():
    """Both streams on one step tick the clock once. A second fwd feed on the same
    step (a z-calibration rollout beside the fused one) only adds evidence."""
    t = ConditionLogZTracker(library_size=2, half_life_visits=HALF_LIFE)
    lv = torch.zeros(2)
    c = torch.zeros(10, dtype=torch.long)
    _feed(t, 'fwd', c, lv, 7)
    _feed(t, 'bwd', c, lv, 7)
    assert int(t.level_clock[0]) == 1 and int(t.level_clock[1]) == 0
    _feed(t, 'fwd', c, lv, 7)
    assert float(t.fwd_level_effective_count[0]) == pytest.approx(20.0)
    _feed(t, 'fwd', c, lv, 8)
    assert int(t.level_clock[0]) == 2
    assert float(t.fwd_level_effective_count[0]) == pytest.approx(20.0 * DECAY + 10.0)


def _drive(t, steps):
    for step in steps:
        lv = torch.tensor([SLOPE * step, 1.0 + SLOPE * step])
        if step % 5 == 0:
            _feed(t, 'fwd', torch.tensor([0, 1, 1]), lv, step)
        _feed(t, 'bwd', torch.tensor([0, 0, 1]), lv + GAP, step)


def test_the_clock_survives_a_checkpoint():
    """A resume mid-cadence continues bit-identically to the uninterrupted run."""
    ref = ConditionLogZTracker(library_size=2, half_life_visits=HALF_LIFE)
    _drive(ref, range(0, 400))
    a = ConditionLogZTracker(library_size=2, half_life_visits=HALF_LIFE)
    _drive(a, range(0, 213))
    b = ConditionLogZTracker.from_state_dict(a.state_dict(), current_step=213)
    _drive(b, range(213, 400))
    for name in ('fwd_level_ema', 'bwd_level_ema', 'fwd_level_effective_count',
                 'bwd_level_effective_count', 'level_clock', 'fwd_level_clock',
                 'bwd_level_clock', 'level_clock_step'):
        assert torch.equal(getattr(ref, name), getattr(b, name)), name


def test_a_checkpoint_without_the_clock_restarts_the_level_streams(capsys):
    """Streams accumulated under per-own-visit decay carry a fwd effective count
    ~fwd_rollout_every times the clock's steady state, which would keep the old lag
    for several half-lives. They restart empty instead, visit counts included, so
    the gap reads +inf (missing) until both streams are trusted again -- never a
    low reading off one or two batches. The rest of the tracker is untouched."""
    a = ConditionLogZTracker(library_size=2, min_visits=4, half_life_visits=HALF_LIFE)
    _drive(a, range(0, 50))
    a.update(torch.tensor([0, 1]), torch.tensor([1.0, 2.0]), step=49)
    assert a.delta_stats()['n'] == 2
    state = a.state_dict()
    for k in ('level_clock', 'level_clock_step', 'fwd_level_clock', 'bwd_level_clock'):
        del state[k]
    b = ConditionLogZTracker.from_state_dict(state, current_step=50)
    assert 'restart empty' in capsys.readouterr().out
    assert torch.isnan(b.fwd_level_ema).all() and torch.isnan(b.bwd_level_ema).all()
    assert int(b.fwd_level_visits.sum()) == 0 and int(b.bwd_level_visits.sum()) == 0
    assert torch.equal(b.ema_logw, a.ema_logw), 'restarted more than the level streams'
    assert b.delta_stats()['mean'] == float('inf')
    _drive(b, range(50, 60))       # 2 fwd writes: condition 0 not yet trusted on fwd
    assert b.delta_stats()['mean'] == float('inf')
    _drive(b, range(60, 70))       # 4 fwd writes: trusted on both
    assert b.delta_stats()['mean'] == pytest.approx(GAP, abs=5e-3)
