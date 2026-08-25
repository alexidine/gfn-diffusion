"""
The two trial guards added by owner decision 2026-08-24, from the toy workout:

  trial_settle_steps  the switch-splash window (D9): the instant burn-in ->
                      candidate rate jump produces a stochastic few-step
                      excursion that measures the JUMP, not the rate. Bar
                      verdicts inside the window are not judged; non-finite and
                      exceptions still fail; a genuinely fatal rate is convicted
                      the step the window closes.
  logz_detour_nats    the guiding-star guard: on these runs log Z rises
                      monotonically until saturation -- a dip-and-recover is
                      safe, a detour is not. A candidate whose batch-mean
                      log_Z_learned falls more than this below the ROOT's value
                      fails, catching the minority-branch (fwd trains Z)
                      destabilization the frac-weighted composite bar is blind
                      to under a dominant bwd frac.

`pytest tests/lr/test_lr_bracket_trial_guards.py -q`
"""

import pytest

from lr_bracket import LRBracket
from test_lr_bracket_driver import HOT, _armed, _trial

pytestmark = pytest.mark.fast


# ------------------------------------------------------------- settle -------

def test_settle_delays_conviction_to_the_first_judged_step():
    """The HOT rung's loss floor sits over the bar for the WHOLE trial, so with
    a settle window it must be convicted exactly when the window closes -- the
    exclusion may not become an acquittal."""
    m0, d0 = _armed(trial_settle_steps=0)
    d0.run_trial(_trial(HOT, label='hot0'))
    o0 = d0.bracket._results[-1]
    assert not o0.ok and o0.steps_to_failure <= 2

    m5, d5 = _armed(trial_settle_steps=5)
    d5.run_trial(_trial(HOT, label='hot5'))
    o5 = d5.bracket._results[-1]
    assert not o5.ok, 'a genuinely fatal rate must still be convicted'
    assert o5.steps_to_failure == 6, (
        f'conviction must land on the first judged step (settle+1), got '
        f'{o5.steps_to_failure}')


def test_nonfinite_fails_inside_the_settle_window():
    """The window excuses BAR verdicts only: a non-finite state cannot recover
    and unjudged steps would just deepen the wreck."""
    m, d = _armed(trial_settle_steps=8)

    original = m.train_step

    def nan_step(step_type):
        original(step_type)
        return float('nan')

    m.train_step = nan_step
    d.run_trial(_trial(0.05, label='nan'))
    o = d.bracket._results[-1]
    assert not o.ok and o.steps_to_failure == 1
    assert 'nonfinite' in o.reason


def test_settle_must_leave_judged_steps():
    with pytest.raises(ValueError, match='trial_settle_steps'):
        LRBracket(candidate_scales=(0.05, 0.1, 0.2, 0.4), burn_in_steps=10,
                  burn_in_scale=0.05, trial_steps=12, trial_settle_steps=12)


# ------------------------------------------------------------- log Z --------

def _with_log_z(value, **kw):
    m, d = _armed(**kw)
    m._last_stats = {'bwd': {'log_Z_learned': 10.0}}
    d.root_log_z = d._live_log_z()          # re-capture: take_root ran before
    assert d.root_log_z == 10.0
    m._last_stats = {'bwd': {'log_Z_learned': float(value)}}
    return m, d


def test_a_log_z_detour_fails_the_rung():
    m, d = _with_log_z(2.0, trial_settle_steps=0, logz_detour_nats=5.0)
    d.run_trial(_trial(0.05, label='detour'))
    o = d.bracket._results[-1]
    assert not o.ok and o.reason.startswith('logz_detour'), o.reason
    assert o.steps_to_failure == 1


def test_a_dip_inside_the_bar_survives():
    """The owner's actual observation: log Z slightly down then stable = safe."""
    m, d = _with_log_z(7.0, trial_settle_steps=0, logz_detour_nats=5.0)
    d.run_trial(_trial(0.05, label='dip'))
    assert d.bracket._results[-1].ok


def test_the_guard_is_off_where_z_does_not_train():
    """No log_Z_learned in the stats (the MLE warm start, the conformer route,
    every CPU harness): root_log_z is None and the guard must abstain."""
    m, d = _armed(trial_settle_steps=0, logz_detour_nats=5.0)
    assert d.root_log_z is None
    d.run_trial(_trial(0.05, label='noz'))
    assert d.bracket._results[-1].ok


def test_the_settle_window_excuses_the_log_z_splash_too():
    m, d = _with_log_z(2.0, trial_settle_steps=5, logz_detour_nats=5.0)
    d.run_trial(_trial(0.05, label='zsplash'))
    o = d.bracket._results[-1]
    assert not o.ok and o.steps_to_failure == 6, (o.reason, o.steps_to_failure)


# ------------------------------------------------- two-tier fire response ---

def _cruising(**kw):
    """A hot seat promoted into cruise, importing the driver-test helpers."""
    from test_lr_bracket_driver import _hot_seat, _to_cruise
    m = _hot_seat(**kw)
    _to_cruise(m)
    return m, m.lr_controller


def test_a_moderate_fire_cuts_in_place_without_a_rewind():
    """A finite excursion over the bar = the state is intact, just too hot:
    the rate halves, observe returns None (no fire_loss_spike), and the
    cooldown makes one incident one cut."""
    m, c = _cruising(hard_failure=dict(cruise_rederive=False),
                     fire_cut_factor=0.5, fire_cooldown_steps=50)
    before = c.scale
    fired = None
    for _ in range(120):
        m.step_ind += 1
        loss = m.train_step(m.train_key)
        fired = c.observe(m.train_key, loss, m.last_grad_norm_pre_clip)
        if c._moderate_fires:
            break
    assert c._moderate_fires == 1, 'the cold bar never fired on this fixture'
    assert fired is None, 'a moderate fire must not reach fire_loss_spike'
    assert c.scale == pytest.approx(before * 0.5)
    # inside the cooldown a second crossing is not a second cut
    scale_after_first = c.scale
    for _ in range(10):
        m.step_ind += 1
        c.observe(m.train_key, m.train_step(m.train_key), m.last_grad_norm_pre_clip)
    assert c._moderate_fires == 1
    assert c.scale == scale_after_first


def test_a_disaster_returns_diverged_and_the_rewind_cut_follows():
    """Non-finite = the disaster tier: observe hands it to the host loop
    (rewind), and on_divergence -- called post-restore -- cuts the rate so the
    restored weights are not re-entered at the rate that detonated them."""
    m, c = _cruising(fire_cut_factor=0.5)
    verdict = c.observe(m.train_key, float('nan'), 10.0)
    assert verdict == 'diverged'
    before = c.scale
    c.on_divergence()
    assert c._divergences == 1
    assert c.scale == pytest.approx(before * 0.5)


# --------------------------------------------- log Z drift compensation -----

class _WalkingStats:
    """_last_stats whose log_Z_learned walks with the trainer's step counter."""

    def __init__(self, m, start, per_step):
        self.m, self.start, self.per_step = m, start, per_step
        self.t0 = int(m.step_ind)

    def get(self, d, default=None):
        if d != 'bwd':
            return default or {}
        t = int(self.m.step_ind) - self.t0
        return {'log_Z_learned': self.start + self.per_step * t}


def test_drift_compensation_acquits_the_stage_walk():
    """The fj119r1o failure shape: Z walks down at the STAGE's own rate under
    every rung. With the root's drift extrapolated into the baseline, a rung
    that merely rides the walk must survive; frozen-baseline judgment convicted
    the whole ladder uniformly."""
    m, d = _armed(trial_settle_steps=0, logz_detour_nats=2.0)
    d.root_log_z, d.root_log_z_slope = 10.0, -0.1
    m._last_stats = _WalkingStats(m, 10.0, -0.1)     # exactly the root's drift
    d.run_trial(_trial(0.05, label='ride'))
    o = d.bracket._results[-1]
    assert o.ok, (o.reason, o.steps_to_failure)      # would fail ~step 20 uncompensated


def test_rate_driven_detour_is_still_convicted_over_the_drift():
    m, d = _armed(trial_settle_steps=0, logz_detour_nats=2.0)
    d.root_log_z, d.root_log_z_slope = 10.0, -0.1
    m._last_stats = _WalkingStats(m, 10.0, -0.4)     # 4x the stage drift
    d.run_trial(_trial(0.05, label='detour3x'))
    o = d.bracket._results[-1]
    assert not o.ok and 'drift-adjusted' in o.reason, o.reason
    # detour vs drifting baseline grows 0.3/step; crosses bar 2.0 just after
    # step 6 -> conviction in the single digits, not at the horizon
    assert o.steps_to_failure <= 8


def test_root_slope_is_estimated_from_the_ema_gap_and_never_positive():
    """take_root reads (raw - ema)/period: a walking-down Z shows raw below its
    lagging EMA; an upward walk is NOT compensated (the star is monotone-up,
    stalling a rise is not a detour)."""
    m, d = _armed()
    for s in range(1, 40):
        m.metric_tracker.update('bwd', {'log_Z_learned': 10.0}, m.step_ind + s)
    m.step_ind += 40
    m._last_stats = {'bwd': {'log_Z_learned': 7.5}}   # raw below EMA: downward walk
    assert d.take_root(m.step_ind) is None
    assert d.root_log_z_slope < 0
    m._last_stats = {'bwd': {'log_Z_learned': 12.5}}  # raw above EMA: upward walk
    assert d.take_root(m.step_ind) is None
    assert d.root_log_z_slope == 0.0
