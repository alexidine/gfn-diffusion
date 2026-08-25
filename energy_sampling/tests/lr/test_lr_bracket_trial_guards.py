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


def test_an_excursion_fire_rewinds_like_every_other_fire():
    """UNIFIED FIRES (owner 2026-08-25, superseding the two-tier design). The
    in-place cut assumed a finite excursion left the state 'intact, just too
    hot'; qm9c aug25 falsified it -- the excursion weights carry poison that
    cuts only slow (vg_lb 126 -> 2700 through two cuts), invisibly to
    tb_err_worst. An excursion now returns 'diverged' so the host loop rewinds
    to the rolling checkpoint; the seat itself does NOT cut (the reload would
    overwrite it -- on_divergence cuts once, from the restored scale), and the
    cooldown makes one incident one response."""
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
    assert fired == 'diverged', (
        'an excursion fire must reach the rewind path -- keeping the excursion '
        'weights is how qm9c aug25 accumulated 20x damage through two cuts')
    assert c.scale == pytest.approx(before), (
        'the fire seat cut the rate itself; the reload overwrites that, so the '
        'single cut belongs to on_divergence after the restore')
    # the post-restore response: one count, one cut
    c.on_divergence()
    assert c._divergences == 1
    assert c.scale == pytest.approx(before * 0.5)
    # inside the cooldown a second crossing is not a second response
    for _ in range(10):
        m.step_ind += 1
        verdict = c.observe(m.train_key, m.train_step(m.train_key),
                            m.last_grad_norm_pre_clip)
        assert verdict is None
    assert c._moderate_fires == 1


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


# --------------------------------------- evidence-scaled confirmation -------

def _decision_bracket(**kw):
    from lr_bracket import LRBracket
    cfg = dict(candidate_scales=(0.05, 0.2, 0.8), burn_in_steps=10,
               burn_in_scale=0.05, trial_steps=50, boundary_confirm_repeats=1,
               boundary_densify=False, trial_settle_steps=5)
    cfg.update(kw)
    return LRBracket(**cfg)


def _drive(b, verdicts):
    """verdicts: scale -> (ok, decisive). Returns the kinds of trials run."""
    from lr_bracket import CONFIRM
    b.begin_bracket(100, bias_correction=0.99)
    kinds = []
    for _ in range(50):
        tr = b.next_trial()
        if tr is None:
            break
        kinds.append(tr.kind)
        ok, decisive = verdicts.get(tr.scale, (True, False))
        if tr.kind == CONFIRM:
            ok, decisive = verdicts.get(('confirm', tr.scale), (ok, decisive))
        b.record(tr, ok=ok, reason=None if ok else 'loss_excursion_x',
                 steps_completed=10, steps_to_failure=None if ok else 10,
                 decisive=decisive)
    return kinds, b.select()


def test_a_decisive_failure_skips_its_confirmation():
    """The re-run exists for the marginal coin flip; a beyond-doubt detonation
    is its own confirmation and its 150 steps are saved."""
    b = _decision_bracket()
    kinds, verdict = _drive(b, {0.8: (False, True)})
    assert 'confirm' not in kinds, kinds
    assert verdict['status'] == 'bracketed'
    assert verdict['boundary_scale'] == 0.8
    assert verdict['boundary_confirmed'] == 1


def test_a_marginal_failure_still_confirms():
    b = _decision_bracket()
    kinds, verdict = _drive(b, {0.8: (False, False),
                                ('confirm', 0.8): (False, False)})
    assert kinds.count('confirm') == 1, kinds
    assert verdict['boundary_scale'] == 0.8


def test_bars_classify_excursion_magnitude():
    """Decisive threshold = hi + DECISIVE_X * (bar - hi): a graze over the bar
    is marginal; far past it is decisive; non-finite is decisive by kind."""
    from lr_bracket_probe import HardFailureBars
    bars = HardFailureBars(loss_excursion_k=3.0, root_window=10,
                           min_observations=5)
    assert bars.derive({'bwd': [0.0, 1.0, 0.5, 0.8, 0.2]}, []) is None
    bar, hi = bars.loss_bar['bwd'], bars.loss_hi['bwd']   # hi=1, span=1 -> bar=4
    assert (bar, hi) == (4.0, 1.0)
    threshold = hi + HardFailureBars.DECISIVE_X * (bar - hi)   # = 10

    assert bars.judge('bwd', 5.0, None) is not None
    assert bars.last_fire['decisive'] is False                 # graze
    assert bars.judge('bwd', threshold + 1, None) is not None
    assert bars.last_fire['decisive'] is True                  # far past
    assert bars.judge('bwd', float('nan'), None) is not None
    assert bars.last_fire['decisive'] is True                  # by kind


def test_logz_detour_decisiveness_scales_with_its_bar():
    m, d = _with_log_z(10.0 - 3.0, trial_settle_steps=0, logz_detour_nats=2.0)
    d.run_trial(_trial(0.05, label='marginalz'))
    o = d.bracket._results[-1]
    assert not o.ok and not o.decisive                         # 3 nats < 3x2
    m2, d2 = _with_log_z(10.0 - 7.0, trial_settle_steps=0, logz_detour_nats=2.0)
    d2.run_trial(_trial(0.05, label='decisivez'))
    o2 = d2.bracket._results[-1]
    assert not o2.ok and o2.decisive                           # 7 nats >= 6
