"""balance.kind: gated_ramp -- one sensor, two motions, hard rails
(docs/design/rarer_rollouts.md, the v1 replay/bwd controller).

The stage is the REAL shipped equilibration stage (configs/prod_sep02, active
protocol) parsed by the real protocol.Stage, with the controller patched in; the
tick is the real StageProtocol method bound onto a stub carrying only what it
reads. A hand-built stage would test the parser against my own idea of the
config's shape.

THE LOAD-BEARING TEST is test_active_modes_knows_the_kind. Stage.active_modes and
read_modes ENUMERATE balance kinds and fall through to self.balance['rules'] for
any other -- a KeyError on the first fused step, which is exactly how 'ratio'
first shipped (the comment in active_modes records it). A new kind that is not
in those tuples parses fine, loads fine, and dies at step 1.
"""
import copy
import pathlib
from types import MethodType, SimpleNamespace

import pytest
import yaml

from energy_sampling import config_invariants
from energy_sampling.protocol import Stage, StageProtocol

HERE = pathlib.Path(__file__).resolve().parent
SHIPPED = HERE.parent.parent / 'configs' / 'prod_sep02' / 'p02_mip_lr1.yaml'

BOUNDS = {'bwd': [0.25, 0.9], 'replay': [0.1, 0.75]}
BALANCE = {'kind': 'gated_ramp', 'ramp': 'replay', 'guard': 'bwd',
           'pinned': {'fwd': 0.0}, 'metric': 'bwd/relative_under_rise150',
           'bar': 1.0, 'up': 0.0017, 'down': 0.043, 'bounds': BOUNDS}


def _spec(**patch):
    cfg = yaml.safe_load(SHIPPED.read_text(encoding='utf-8'))
    for i, st in enumerate(config_invariants.active_stages(cfg)):
        if st.get('name') == 'equilibration':
            st = copy.deepcopy(st)
            st['fwd_rollout_every'] = 10
            st['flags']['z_calibration'] = False
            st['fracs'] = {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}
            st.pop('min_fracs', None)
            st['balance'] = copy.deepcopy(BALANCE)
            st.update(patch)
            return st, i
    raise AssertionError('no equilibration stage in the active protocol')


def _stage(**patch):
    spec, i = _spec(**patch)
    return Stage(spec, i)


class _Tracker:
    def __init__(self, **values):
        self.values = values

    def get(self, direction, name, default=None):
        return self.values.get(f'{direction}/{name}', default)


def _proto(stg, tracker, share=None):
    m = SimpleNamespace(fwd_frac=0.0, bwd_frac=0.5, replay_frac=0.5, step_ind=0,
                        metric_tracker=tracker)
    p = SimpleNamespace(m=m, stage=stg, tracker=tracker,
                        ctrl={'gates': {}, 'gr_share': share, 'gr_fired': 0.0,
                              'gr_best': None, 'gr_held': 0.0})
    p._resolve = MethodType(StageProtocol._resolve, p)
    p._gated_ramp_tick = MethodType(StageProtocol._gated_ramp_tick, p)
    return p


# ---------------------------------------------------------------- parsing

def test_parses_the_shipped_stage_with_the_controller():
    stg = _stage()
    b = stg.balance
    assert b['kind'] == 'gated_ramp' and b['ramp'] == 'replay' and b['guard'] == 'bwd'
    assert b['bar'] == 1.0 and b['up'] == 0.0017 and b['down'] == 0.043
    assert b['pinned'] == {'fwd': 0.0}
    assert set(b['metrics']) == {'bwd', 'replay'}, 'the split pair, for pinned/bounds parsing'


@pytest.mark.parametrize('patch, match', [
    ({'ramp': 'bwd'}, 'distinct'),
    ({'bar': -1.0}, 'must be >= 0'),
    ({'up': 0.0}, r'\(0, 1\]'),
    ({'down': 2.0}, r'\(0, 1\]'),
    ({'gain': 0.5}, 'unknown keys'),
    ({'metric': 'under_coverage'}, 'dir/name'),
])
def test_refuses_malformed_controllers(patch, match):
    spec, i = _spec()
    spec['balance'].update(patch)
    with pytest.raises(ValueError, match=match):
        Stage(spec, i)


def test_refuses_a_pin_that_disagrees_with_fracs():
    spec, i = _spec()
    spec['fracs']['fwd'] = 0.05           # pinned says 0.0
    with pytest.raises(ValueError, match='pinned'):
        Stage(spec, i)


# ------------------------------------------------------- active / read modes

def test_active_modes_knows_the_kind():
    """Must not raise, and must name BOTH split modes plus the pinned one.
    Against a build where 'gated_ramp' is missing from active_modes' kind tuple
    this raises KeyError('rules') -- the failure the fused step would hit."""
    stg = _stage()
    modes = stg.active_modes                   # a property; raises KeyError('rules') on the old tuple
    assert {'bwd', 'replay', 'fwd'} <= set(modes)
    # read_modes maps the balance's metric names back to the branches that
    # produce them; bwd produces the sensor, so it must never be dormant --
    # a dormant branch skips even its force-refresh and the sensor goes stale
    assert 'bwd' in stg.read_modes


# --------------------------------------------------------------------- tick

def test_holds_still_while_the_sensor_is_unwritten():
    p = _proto(_stage(), _Tracker())
    p._gated_ramp_tick(p.stage.balance)
    assert (p.m.replay_frac, p.m.bwd_frac) == (0.5, 0.5)
    assert p.ctrl['gr_share'] == 0.5 and p.ctrl['gr_fired'] == 0.0


def test_ramps_up_by_up_when_the_guard_is_quiet():
    p = _proto(_stage(), _Tracker(**{'bwd/relative_under_rise150': 0.2}))
    p._gated_ramp_tick(p.stage.balance)
    assert p.m.replay_frac == pytest.approx(0.5 + 0.0017)
    assert p.m.bwd_frac == pytest.approx(0.5 - 0.0017)
    assert p.ctrl['gr_fired'] == 0.0


def test_drops_by_down_and_reports_fired_when_the_guard_trips():
    p = _proto(_stage(), _Tracker(**{'bwd/relative_under_rise150': 2.5}))
    p._gated_ramp_tick(p.stage.balance)
    assert p.m.replay_frac == pytest.approx(0.5 - 0.043)
    assert p.m.bwd_frac == pytest.approx(0.5 + 0.043)
    assert p.ctrl['gr_fired'] == 1.0


def test_bar_is_a_strict_threshold():
    """Exactly AT the bar is 'not rising': the deadband is the bar itself."""
    p = _proto(_stage(), _Tracker(**{'bwd/relative_under_rise150': 1.0}))
    p._gated_ramp_tick(p.stage.balance)
    assert p.ctrl['gr_fired'] == 0.0 and p.m.replay_frac > 0.5


def test_rails_hold_at_both_ends():
    up = _proto(_stage(), _Tracker(**{'bwd/relative_under_rise150': 0.0}), share=0.749)
    up._gated_ramp_tick(up.stage.balance)
    assert up.m.replay_frac == pytest.approx(0.75), 'replay cap 0.75 (== 1 - bwd floor 0.25)'
    assert up.m.bwd_frac == pytest.approx(0.25)
    down = _proto(_stage(), _Tracker(**{'bwd/relative_under_rise150': 9.0}), share=0.12)
    down._gated_ramp_tick(down.stage.balance)
    assert down.m.replay_frac == pytest.approx(0.10), 'replay floor 0.1 (== 1 - bwd cap 0.9)'
    assert down.m.bwd_frac == pytest.approx(0.90)


def test_pinned_mode_is_reasserted_every_tick():
    p = _proto(_stage(), _Tracker())
    p.m.fwd_frac = 0.3                    # something drifted it
    p._gated_ramp_tick(p.stage.balance)
    assert p.m.fwd_frac == 0.0


def test_the_split_pair_is_conserved():
    p = _proto(_stage(), _Tracker(**{'bwd/relative_under_rise150': 3.0}))
    for _ in range(25):
        p._gated_ramp_tick(p.stage.balance)
    assert p.m.replay_frac + p.m.bwd_frac == pytest.approx(1.0)
    assert p.m.replay_frac == pytest.approx(0.10), '25 ticks x 0.043 from 0.5 is past the floor; must rail, not overshoot'


def _bar0(**patch):
    """A stage whose guard fires on ANY deterioration (bar 0), reading the
    Z-anchored channel -- the owner's 2026-09-07 call."""
    bal = dict(BALANCE, metric='bwd/under_coverage_rise150', bar=0.0)
    return _stage(balance=bal, **patch)


def test_bar_zero_is_accepted_and_parses_as_the_setpoint():
    """bar 0 is the SETPOINT spelling: the guard's job is to keep the metric's
    slope negative, so 'fire whenever it rose at all' is the honest bar and any
    positive value is a TOLERATED rate of deterioration. Refused as 'strictly
    positive' until 2026-09-07, when bar 1.0 was measured firing 0 times in 2000
    steps while the metric it watches doubled."""
    b = _bar0().balance
    assert b['bar'] == 0.0 and b['metric'] == 'bwd/under_coverage_rise150'


def test_bar_zero_ramps_only_while_the_guard_metric_is_FALLING():
    """The motion that bar 1.0 could not produce: a small positive rise -- well
    inside the old bar -- now backs the ramp off instead of being ignored."""
    rising = _proto(_bar0(), _Tracker(**{'bwd/under_coverage_rise150': 0.22}))
    rising._gated_ramp_tick(rising.stage.balance)
    assert rising.ctrl['gr_fired'] == 1.0
    assert rising.m.replay_frac == pytest.approx(0.5 - 0.043)
    assert rising.m.bwd_frac == pytest.approx(0.5 + 0.043)

    falling = _proto(_bar0(), _Tracker(**{'bwd/under_coverage_rise150': -0.1}))
    falling._gated_ramp_tick(falling.stage.balance)
    assert falling.ctrl['gr_fired'] == 0.0
    assert falling.m.replay_frac == pytest.approx(0.5 + 0.0017)


def test_bar_zero_holds_the_ramp_open_on_an_exactly_flat_sensor():
    """`>` not `>=`, so a converged run (slope 0) still ramps rather than
    deadlocking on the boundary."""
    p = _proto(_bar0(), _Tracker(**{'bwd/under_coverage_rise150': 0.0}))
    p._gated_ramp_tick(p.stage.balance)
    assert p.ctrl['gr_fired'] == 0.0 and p.m.replay_frac > 0.5


def test_negative_bar_is_still_refused():
    """It would demand the run keep IMPROVING at a rate merely to stay quiet."""
    with pytest.raises(ValueError, match='must be >= 0'):
        _stage(balance=dict(BALANCE, bar=-1.0))


# ---------------------------------------------------------------------------
# The ratchet: the ramp is released only at a new best LEVEL of ratchet_metric.
# ---------------------------------------------------------------------------

# Level and slope of the SAME quantity -- a ratchet on a different metric from
# the one the guard watches does not ratchet anything the guard cares about.
RATCHET = dict(BALANCE, metric='bwd/under_coverage_rise150', bar=0.0,
               ratchet_metric='bwd/under_coverage', ratchet_tol=0.25)


def _r(level, sensor=-1.0, share=0.5):
    """A tick with the slope sensor QUIET (-1, well under bar 0), so any motion
    away from the ramp is the ratchet's doing and not the slope gate's."""
    p = _proto(_stage(balance=dict(RATCHET)),
               _Tracker(**{'bwd/under_coverage_rise150': sensor,
                           'bwd/under_coverage': level}), share=share)
    p._gated_ramp_tick(p.stage.balance)
    return p


def test_ratchet_parses_and_defaults_to_absent():
    assert _stage(balance=dict(RATCHET)).balance['ratchet_metric'] == 'bwd/under_coverage'
    assert _stage().balance['ratchet_metric'] is None, 'absent = pure slope gate'


def test_ratchet_refuses_a_malformed_metric_or_negative_tol():
    with pytest.raises(ValueError, match='ratchet_metric'):
        _stage(balance=dict(RATCHET, ratchet_metric='relative_under'))
    with pytest.raises(ValueError, match='ratchet_tol'):
        _stage(balance=dict(RATCHET, ratchet_tol=-1.0))


def test_first_level_seen_is_a_new_best_and_releases_the_ramp():
    p = _r(level=5.0)
    assert p.ctrl['gr_best'] == pytest.approx(5.0)
    assert p.ctrl['gr_held'] == 0.0
    assert p.m.replay_frac == pytest.approx(0.5 + RATCHET['up'])


def test_a_level_above_the_best_vetoes_the_ramp_even_with_a_quiet_slope():
    """The failure the ratchet exists for: on rr07_rr_n7_uc0 the slope was
    negative on two thirds of ticks while the level went nowhere."""
    p = _proto(_stage(balance=dict(RATCHET)),
               _Tracker(**{'bwd/under_coverage_rise150': -1.0, 'bwd/under_coverage': 5.0}))
    p._gated_ramp_tick(p.stage.balance)          # best := 5.0, ramp released
    p.tracker.values['bwd/under_coverage'] = 6.0  # ... then the level worsens
    p._gated_ramp_tick(p.stage.balance)
    assert p.ctrl['gr_best'] == pytest.approx(5.0), 'best must not follow the level up'
    assert p.ctrl['gr_held'] == 1.0
    assert p.m.replay_frac < 0.5, 'weight went to the guard despite a negative slope'


def test_the_tolerance_band_still_counts_as_at_best():
    assert _r(level=5.0).ctrl['gr_held'] == 0.0
    p = _proto(_stage(balance=dict(RATCHET)),
               _Tracker(**{'bwd/under_coverage_rise150': -1.0, 'bwd/under_coverage': 5.0}))
    p._gated_ramp_tick(p.stage.balance)
    for lvl, held in ((5.20, 0.0), (5.30, 1.0)):      # tol 0.25
        p.tracker.values['bwd/under_coverage'] = lvl
        p._gated_ramp_tick(p.stage.balance)
        assert p.ctrl['gr_held'] == held, lvl


def test_a_new_best_re_releases_the_ramp_after_a_hold():
    p = _proto(_stage(balance=dict(RATCHET)),
               _Tracker(**{'bwd/under_coverage_rise150': -1.0, 'bwd/under_coverage': 5.0}))
    for lvl in (5.0, 9.0, 9.0):
        p.tracker.values['bwd/under_coverage'] = lvl
        p._gated_ramp_tick(p.stage.balance)
    held_share = p.ctrl['gr_share']
    assert p.ctrl['gr_held'] == 1.0 and held_share < 0.5
    p.tracker.values['bwd/under_coverage'] = 4.0      # earned it
    p._gated_ramp_tick(p.stage.balance)
    assert p.ctrl['gr_best'] == pytest.approx(4.0)
    assert p.ctrl['gr_held'] == 0.0
    assert p.ctrl['gr_share'] > held_share


def test_the_ratchet_holds_before_the_slope_sensor_has_a_window():
    """The rise needs 300 steps; the level does not. A stage must not spend its
    first 300 steps ramping into a deterioration just because v is unwritten."""
    p = _proto(_stage(balance=dict(RATCHET)), _Tracker(**{'bwd/under_coverage': 5.0}))
    p._gated_ramp_tick(p.stage.balance)           # best := 5.0, no v -> no motion
    assert p.m.replay_frac == pytest.approx(0.5)
    p.tracker.values['bwd/under_coverage'] = 8.0
    p._gated_ramp_tick(p.stage.balance)
    assert p.ctrl['gr_held'] == 1.0
    assert p.m.replay_frac == pytest.approx(0.5 - RATCHET['down'])


def test_gr_fired_reports_the_SLOPE_gate_only_not_the_ratchet():
    """Two different reasons to feed the guard; conflating them would make the
    run unreadable ('did it deteriorate, or was it merely not at its best?')."""
    p = _r(level=9.0, sensor=-1.0, share=0.5)     # first tick -> best, released
    p.tracker.values['bwd/under_coverage'] = 20.0
    p._gated_ramp_tick(p.stage.balance)
    assert p.ctrl['gr_held'] == 1.0 and p.ctrl['gr_fired'] == 0.0


# ---------------------------------------------------------------------------
# bootstrap_z:rollout -- the phase-2 entry Z seed (a different mechanism from
# the bare bootstrap_z, which regresses the head onto the tracker's ema_logw).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('arg', ['rollout', 'rollout:4000', 'train_conditioner', ''])
def test_bootstrap_z_accepts_its_forms(arg):
    action = f'bootstrap_z:{arg}' if arg else 'bootstrap_z'
    st = _stage(on_enter=['rebuild_prior_by_churn', action])
    assert ('bootstrap_z', arg) in st.on_enter


@pytest.mark.parametrize('arg', ['rollout:', 'rollout:many', 'rollouts', 'rollout:4000:2'])
def test_bootstrap_z_refuses_a_malformed_rollout_arg(arg):
    with pytest.raises(ValueError, match='bootstrap_z'):
        _stage(on_enter=[f'bootstrap_z:{arg}'])


# ---------------------------------------------------------------------------
# The Schmitt ratchet: trip high, release low. A single threshold lets an arm
# CYCLE -- excurse, drift back to the trip line, ramp, excurse again -- because
# touching the line is enough to be released. These pin the two thresholds
# apart and pin the degenerate cases back onto the old behaviour.
# ---------------------------------------------------------------------------

def _ratchet_run(levels, tol, release_tol, sensor=-1.0):
    """Feed a level series through the ratchet; return the held flags."""
    st = _stage(balance=dict(RATCHET, ratchet_tol=tol, ratchet_release_tol=release_tol))
    bal = st.balance
    ctrl, held = {}, []
    best = float('inf')
    for L in levels:
        if L < best:
            best = L
        hi, lo = float(bal['ratchet_tol']), float(bal['ratchet_release_tol'])
        tripped = bool(ctrl.get('gr_tripped', False))
        if L > best + hi:
            tripped = True
        elif L <= best + min(lo, hi):
            tripped = False
        ctrl['gr_tripped'] = tripped
        held.append(tripped)
    return held


def test_release_is_lower_than_trip_so_a_touch_does_not_clear_it():
    """Level dips to 10 (best), excurses to 10.6 (over trip 0.5), then settles at
    10.4 -- back under the TRIP line but not under the RELEASE line, so it stays
    held. Under one threshold it would have been released at 10.4 and could
    excurse again immediately: that is the cycle this exists to stop."""
    held = _ratchet_run([10.0, 10.6, 10.4, 10.4], tol=0.5, release_tol=0.2)
    assert held == [False, True, True, True]


def test_a_repaired_excursion_releases():
    held = _ratchet_run([10.0, 10.6, 10.4, 10.15], tol=0.5, release_tol=0.2)
    assert held == [False, True, True, False]


def test_equal_thresholds_reduce_to_the_old_single_threshold_behaviour():
    lv = [10.0, 10.6, 10.4, 10.15, 10.9]
    assert _ratchet_run(lv, tol=0.5, release_tol=0.5) == [L > 10.0 + 0.5 for L in lv]


def test_release_tol_defaults_to_the_trip_tol():
    st = _stage(balance=dict(RATCHET, ratchet_tol=0.25))
    assert st.balance['ratchet_release_tol'] == pytest.approx(0.25)


def test_release_above_trip_is_refused():
    with pytest.raises(ValueError, match='ratchet_release_tol'):
        _stage(balance=dict(RATCHET, ratchet_tol=0.2, ratchet_release_tol=0.5))


def test_a_negative_release_is_refused():
    with pytest.raises(ValueError, match='ratchet_release_tol'):
        _stage(balance=dict(RATCHET, ratchet_tol=0.5, ratchet_release_tol=-0.1))


# ---------------------------------------------------------------------------
# THE ENTRY COOLDOWN on the ratchet's WRITE.
#
# `gr_best` is a running MINIMUM, so a level read while the stage is still
# settling is one no later state can return to: the Schmitt trigger latches for
# the rest of the stage and the guard rails at its cap. Measured across three
# propanol runs, bwd/under_coverage entered at ~0.5 and rose to an operating
# ~4.4-5.1 over ~200-430 steps, so the first eval's level was ~10x too low.
#
# This REPLACES ratchet_z_bar, which gated the same write on
# |fwd/tb_resid_clipped|. That proxy could not see the lag that matters: the
# ratchet's level is a BWD quantity and the two branches' log Z reach their
# fixed points at different times (measured: fwd residual -0.079 with
# bwd/jensen_z_err at 42.5 and the level still at 0.49).
# ---------------------------------------------------------------------------

CD = dict(RATCHET, ratchet_cooldown_steps=250)


def _c(level, step, proto=None, share=0.5):
    """One tick at a given level and step-into-stage, slope sensor quiet."""
    p = proto or _proto(_stage(balance=dict(CD)),
                        _Tracker(**{'bwd/under_coverage_rise150': -1.0}), share=share)
    p.tracker.values['bwd/under_coverage'] = level
    p.m.step_ind = step
    p._gated_ramp_tick(p.stage.balance)
    return p


def test_cooldown_parses_and_defaults_to_zero():
    assert _stage(balance=dict(CD)).balance['ratchet_cooldown_steps'] == 250
    assert _stage(balance=dict(RATCHET)).balance['ratchet_cooldown_steps'] == 0
    with pytest.raises(ValueError, match='ratchet_cooldown_steps'):
        _stage(balance=dict(CD, ratchet_cooldown_steps=-1))


def test_ratchet_z_bar_is_retired_and_refused_not_ignored():
    """A dead gate key reads as an armed gate, so it must fail at parse."""
    with pytest.raises(ValueError, match='ratchet_z_bar is retired'):
        _stage(balance=dict(RATCHET, ratchet_z_bar=0.5))


def test_the_startup_transient_does_not_become_the_best():
    """THE BUG. The stage enters at a level of 0.5 that it never returns to;
    the honest operating level is 5.0. Pre-fix, best latched at 0.5 and the
    ramp was vetoed for the rest of the stage."""
    p = _c(level=0.5, step=0)
    assert p.ctrl['gr_cooling'] == 1.0
    assert p.ctrl['gr_best'] is None, 'a settling level must not be recorded'
    for st in (100, 200):
        _c(level=0.5, step=st, proto=p)
    assert p.ctrl['gr_best'] is None, 'still inside the cooldown'
    _c(level=5.0, step=250, proto=p)
    assert p.ctrl['gr_cooling'] == 0.0
    assert p.ctrl['gr_best'] == pytest.approx(5.0), 'captures the operating level'
    assert p.ctrl['gr_tripped'] is False


def test_the_cooldown_freezes_the_fracs_not_just_the_capture():
    """Freezing the split too is what makes the length uncritical: a cooldown
    that overruns costs nothing, so it can be set long."""
    p = _c(level=0.5, step=0, share=0.5)
    entry = p.m.replay_frac
    for st in (50, 100, 150, 200):
        _c(level=0.5, step=st, proto=p)
    assert p.m.replay_frac == pytest.approx(entry), 'no motion during the cooldown'
    _c(level=5.0, step=260, proto=p)
    assert p.m.replay_frac > entry, 'and it ramps once released'


def test_the_cooldown_is_anchored_on_the_first_tick_not_on_step_zero():
    """Stage entry is not step 0 on a resume -- stage_ctrl is fresh at every
    transition, so the window has to start from the first tick it sees."""
    p = _c(level=0.5, step=6510)                 # a resumed run's entry step
    assert p.ctrl['gr_cooling'] == 1.0 and p.ctrl['gr_best'] is None
    _c(level=0.5, step=6700, proto=p)
    assert p.ctrl['gr_cooling'] == 1.0, 'still inside 250 of the FIRST tick'
    _c(level=5.0, step=6761, proto=p)
    assert p.ctrl['gr_cooling'] == 0.0 and p.ctrl['gr_best'] == pytest.approx(5.0)


def test_zero_cooldown_is_the_pre_cooldown_behaviour():
    p = _c(level=2.0, step=0,
           proto=_proto(_stage(balance=dict(RATCHET)),
                        _Tracker(**{'bwd/under_coverage_rise150': -1.0})))
    assert p.ctrl['gr_best'] == pytest.approx(2.0)
