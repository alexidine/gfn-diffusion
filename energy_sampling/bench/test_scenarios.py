"""
Tests for the oracle and the scenario scoring.

These guard the MEASUREMENT, not the controller. Every regret number in the
battery is a ratio against `find_oracle`, so an oracle that is quietly not an
oracle would make the whole scoreboard wrong in a way that reads as a controller
result. Same for `recovered_at`: a bug there turns "never recovered" into a
finding about the controller instead of a finding about the scorer.

Deliberately small surfaces -- these run in seconds, unlike the battery itself.
"""

import math

import numpy as np
import pytest
import torch

from bench.oracle import Surface, find_oracle, median_trace
from bench.scenarios import RECOVERY_TOL, detectability, recovered_at

FAST = Surface('fast', 'mle', dict(dim=8, cond=20.0, noise=0.02, init_scale=2.0),
               steps=400, lr_grid=(1e-5, 1e0, 9))


# --------------------------------------------------------------------- oracle

def test_oracle_is_interior_and_beats_both_edges():
    o = find_oracle(FAST, seeds=(0, 1), refine=False)
    grid = sorted(o.curve)
    assert grid[0] < o.lr < grid[-1], 'winner must be interior to the bracket'
    assert o.curve[o.lr] < 0.5 * min(o.curve[grid[0]], o.curve[grid[-1]])
    assert math.isfinite(o.final) and o.final > 0


def test_oracle_refuses_a_bracket_whose_minimum_is_at_an_edge():
    """
    A minimum at an edge means the bracket was wrong, and the number is the best
    of a badly chosen set rather than an oracle. Every regret figure downstream
    would be silently deflated.
    """
    truncated = Surface('truncated', 'mle',
                        dict(dim=8, cond=20.0, noise=0.02, init_scale=2.0),
                        steps=400, lr_grid=(1e-6, 1e-4, 5))   # all far too cold
    with pytest.raises(ValueError, match='EDGE of the bracket'):
        find_oracle(truncated, seeds=(0,), refine=False)


def test_oracle_refuses_an_lr_insensitive_surface():
    """If the best rate does not beat the bracket edges, regret against it
    measures nothing and saying so is better than reporting 1.0x everywhere."""
    flat = Surface('flat', 'mle', dict(dim=4, cond=1.0, noise=0.0, init_scale=1.0),
                   steps=20, lr_grid=(1e-6, 1e-5, 5))
    with pytest.raises(ValueError):
        find_oracle(flat, seeds=(0,), refine=False)


def test_oracle_trace_is_the_median_across_seeds():
    o = find_oracle(FAST, seeds=(0, 1), refine=False)
    assert len(o.trace) == FAST.steps
    assert o.trace[0] > o.trace[-1], 'the oracle should make progress'
    assert o.healthy_at(0, 2.0) == pytest.approx(o.trace[0] * 2.0)
    assert o.healthy_at(10 ** 6, 1.0) == pytest.approx(o.trace[-1])


# ------------------------------------------------------------------- recovery

class _FakeRun:
    """Just enough of a BenchRun for the scorers."""

    def __init__(self, dists, calibrations=None, divergences=0, bounds=(0.01, 2000.0),
                 peaks=None):
        from types import SimpleNamespace
        self.history = [{'dist': d, 'peak_scale': (peaks[i] if peaks else 1.0),
                         'lr': 1e-3} for i, d in enumerate(dists)]
        self.calibrations = calibrations or []
        self.divergences = divergences
        self.args = SimpleNamespace(
            adaptive_lr=SimpleNamespace(bounds=bounds))


class _FakeOracle:
    def __init__(self, trace):
        self.trace = np.array(trace, dtype=float)


def test_recovered_at_is_zero_when_never_unhealthy():
    oracle = _FakeOracle([10.0, 5.0, 1.0, 0.5])
    run = _FakeRun([10.0, 5.0, 1.0, 0.5])
    assert recovered_at(run, oracle, tol=RECOVERY_TOL) == 0


def test_recovered_at_is_the_step_after_the_last_violation():
    oracle = _FakeOracle([10.0, 10.0, 10.0, 10.0, 10.0])
    # violates (>3x oracle) at indices 1 and 2, healthy from 3 onward
    run = _FakeRun([10.0, 100.0, 100.0, 10.0, 10.0])
    assert recovered_at(run, oracle, tol=3.0) == 3


def test_recovered_at_is_none_when_still_unhealthy_at_the_end():
    oracle = _FakeOracle([10.0, 10.0, 10.0])
    run = _FakeRun([10.0, 10.0, 1000.0])
    assert recovered_at(run, oracle, tol=3.0) is None


def test_recovery_is_measured_against_the_trace_not_the_final_value():
    """
    Early in training everything is far from the optimum. Scoring against the
    oracle's FINAL distance would mark the first steps of a perfectly healthy run
    as unrecovered, and every scenario would report a spurious recovery time.
    """
    oracle = _FakeOracle([100.0, 10.0, 1.0, 0.1])
    healthy = _FakeRun([100.0, 10.0, 1.0, 0.1])
    assert recovered_at(healthy, oracle, tol=1.5) == 0


def test_non_finite_distances_count_as_unhealthy():
    oracle = _FakeOracle([1.0, 1.0, 1.0])
    assert recovered_at(_FakeRun([1.0, math.inf, 1.0]), oracle, tol=3.0) == 2
    assert recovered_at(_FakeRun([1.0, 1.0, math.nan]), oracle, tol=3.0) is None


# -------------------------------------------------------------- detectability

def test_divergence_always_flags():
    assert detectability(_FakeRun([1.0], divergences=1))['flagged'] is True


def test_a_pinned_sensor_flags():
    cals = [{'status': 'above_range'} for _ in range(10)]
    d = detectability(_FakeRun([1.0], calibrations=cals))
    assert d['sat_frac'] == 1.0 and d['flagged'] is True


def test_a_healthy_run_does_not_flag():
    cals = [{'status': 'bracketed'} for _ in range(10)]
    d = detectability(_FakeRun([1.0] * 10, calibrations=cals))
    assert d['flagged'] is False


def test_peak_scale_parked_at_a_bound_flags():
    """peak_scale sitting at its floor is the signature of the stranded-cold
    state, and it is otherwise indistinguishable from slow progress."""
    d = detectability(_FakeRun([1.0] * 10, calibrations=[{'status': 'bracketed'}] * 4,
                               peaks=[0.01] * 10))
    assert d['bound_frac'] == 1.0 and d['flagged'] is True


def test_too_few_readings_does_not_flag_on_saturation_alone():
    """A run with 2 calibrations is not evidence of a pinned sensor."""
    cals = [{'status': 'above_range'} for _ in range(2)]
    assert detectability(_FakeRun([1.0], calibrations=cals))['flagged'] is False


# ------------------------------------------------------------------- rewind

"""
The divergence response is REWIND + peak cut (train.py fire_loss_spike ->
load_model_only -> on_divergence), and modelling only the cut is the difference
between "recovers in ~124 steps" and "never recovers, 1985 divergences". The
bench got this wrong once by omission, so it is pinned.
"""


def _blowup_run(steps_before=200, factor=1e4, **kw):
    from bench.harness import BenchRun
    run = BenchRun(game='mle', need_batch_sizer=False,
                   game_kwargs=dict(dim=8, cond=20.0, noise=0.01, lr=1e-2, seed=0),
                   args_overrides={'adaptive_lr.warmup_steps': 20,
                                   'ray_calibration.enabled': False,
                                   'lr_policy': 1e-2, 'min_lr': 1e-12},
                   **kw)
    run.run(steps_before, stop_on_divergence=False)
    st = run.m.lr_ctrl
    st['peak_scale'] = float(st.get('peak_scale', 1.0)) * factor
    run.m.lr_controller.step()
    return run


def test_divergence_rewinds_to_the_last_healthy_checkpoint():
    """
    A survivable blow-up with the sensor OFF: rewind plus the divergence ladder
    alone brings it back. Base lr 1e-2 against a stability limit of 2/cond = 0.1
    leaves 10x of headroom, so a 40x excursion needs 2 halvings -- inside the
    3-reload budget.
    """
    run = _blowup_run(factor=40.0)
    assert run._ckpt is not None, 'a healthy checkpoint should exist by step 200'

    run.run(300, stop_on_divergence=False)
    assert run.divergences > 0 and run.total_reloads > 0
    assert math.isfinite(run.game.distance_to_opt()), (
        'the rewind must restore finite parameters; without it check_spike fires '
        'on every subsequent step forever')
    assert run.aborted is None


def test_the_divergence_ladder_alone_can_only_undo_about_2_to_the_budget():
    """
    HOW BIG AN EXCURSION IS RECOVERABLE, and what does the recovering.

    `divergence_cut` 0.5 removes a factor of 2 per reload and the budget is
    max(3, 0.2 per 1k steps), so the bar+rewind path alone can walk back roughly
    2^budget of overshoot -- about 8x beyond the headroom. Past that the budget
    runs out and train.py raises FrozenTrainingState: the run is aborted, not
    left thrashing. A large enough excursion is unrecoverable BY DESIGN.

    This is why the sensor matters for recovery and not just for tuning: with
    calibration ON the servo cuts between divergences and does most of the work
    (the `mle` blowup_100x scenario needs only ONE divergence), while with it off
    the ladder has to do it all and runs out of budget.
    """
    off = _blowup_run(factor=1e4)
    off.run(400, stop_on_divergence=False)
    assert off.aborted is not None
    assert off.total_reloads > 3


def test_checkpoints_are_only_taken_from_healthy_states():
    """Saving a diverged state would leave the rewind nothing to go back to."""
    run = _blowup_run()
    before = run._ckpt['params'][0].clone()
    run.run(60, stop_on_divergence=False)
    assert torch.all(torch.isfinite(run._ckpt['params'][0]))


def test_no_rewind_target_still_cuts_the_peak():
    """
    train.py's NO REWIND TARGET branch is reachable in real runs -- 'best' is
    only written once an eval has improved -- and it must still take the half of
    the response that is available.
    """
    run = _blowup_run(checkpoint_every=0)          # never checkpoints
    assert run._ckpt is None
    peak_before = run.m.lr_ctrl['peak_scale']
    run.run(50, stop_on_divergence=False)
    assert run.divergences > 0
    assert run.m.lr_ctrl['peak_scale'] < peak_before, 'peak must still be cut'


def test_reload_budget_aborts_the_run():
    """
    Past max_reloads_per_1k_steps train.py raises FrozenTrainingState and the job
    dies. A run over budget is dead, not slow, and scoring it as though it kept
    training would flatter it.
    """
    run = _blowup_run(checkpoint_every=0, max_reloads_per_1k=0.0)
    run.run(500, stop_on_divergence=False)
    assert run.aborted is not None
    assert run.total_reloads > 3, 'floor of 3 reloads applies before the rate does'


def test_rewind_restores_non_parameter_state():
    """The buffer mean is not an optimizer parameter, and a restore that leaves
    it at its diverged value is not a restore."""
    from bench.harness import BenchRun
    run = BenchRun(game='equilibration', need_batch_sizer=False,
                   game_kwargs=dict(dim=4, a=4.0, w_rep=0.7, w_bwd=0.3,
                                    kappa=0.05, noise=0.05, lr=0.05, seed=0),
                   args_overrides={'adaptive_lr.warmup_steps': 20,
                                   'ray_calibration.enabled': False,
                                   'lr_fused': 0.05, 'min_lr': 1e-12})
    run.run(200, stop_on_divergence=False)
    assert 'mu' in run._ckpt['extra']
    saved_mu = run._ckpt['extra']['mu'].clone()
    run.game.mu = torch.full_like(run.game.mu, float('nan'))
    assert run._rewind() is True
    assert torch.equal(run.game.mu, saved_mu)


# ------------------------------------------------- climber / braker factorial

"""
Raising and lowering are separate jobs (F-020), so they are separate switches.
These pin the split itself -- if `ray`-as-climber ever started cutting, the
factorial would silently collapse back to the monolithic sensor and every
comparison drawn from it would be wrong.
"""


def _bench(args_overrides=None, **kw):
    from bench.harness import BenchRun
    overrides = {'adaptive_lr.warmup_steps': 0, 'ray_calibration.enabled': False}
    overrides.update(args_overrides or {})
    return BenchRun(game='mle', need_batch_sizer=False,
                    game_kwargs=dict(dim=4, cond=2.0, noise=0.0, lr=1e-3),
                    args_overrides=overrides, **kw)


def test_legacy_sensor_names_map_onto_the_factorial():
    from bench.harness import BenchRun
    for name, (climber, braker) in BenchRun._SENSOR_PAIRS.items():
        run = _bench(sensor=name)
        assert (run.climber, run.braker) == (climber, braker), name


def test_ray_as_climber_only_ignores_a_cut():
    """A reading below target would lower the rate; a climber must not act on it."""
    run = _bench(climber='ray', braker='none')
    assert run._probe_role_allows({'alpha_star': 32.0}) is True     # would raise
    assert run._probe_role_allows({'alpha_star': 1.0}) is False     # would cut


def test_ray_as_braker_only_ignores_a_raise():
    run = _bench(climber='none', braker='ray')
    assert run._probe_role_allows({'alpha_star': 32.0}) is False
    assert run._probe_role_allows({'alpha_star': 1.0}) is True


def test_an_unresolved_reading_is_acted_on_by_neither():
    run = _bench(climber='ray', braker='ray')
    assert run._probe_role_allows({'alpha_star': float('nan')}) is False


def test_the_probe_runs_if_either_role_asks_and_not_otherwise():
    assert 'ray' in (_bench(climber='ray', braker='plateau').climber,
                     _bench(climber='ray', braker='plateau').braker)
    plain = _bench(climber='ramp', braker='plateau')
    assert 'ray' not in (plain.climber, plain.braker), 'no probe cost when unused'


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError, match='climber in'):
        _bench(climber='magic')


# ------------------------------------------------------- loss-slope sensors

"""
`slope` and `slope_seek` are CANDIDATES -- train.py's LR_SENSOR_KINDS is
('ray', 'plateau', 'none') and neither exists there. These pin the semantics
that any real implementation would have to reproduce.
"""


def test_progress_rate_is_scale_free_and_signed():
    """Negative = improving. Scale-free, because it is compared across learning
    rates and across surfaces whose losses differ by orders of magnitude."""
    run = _bench(climber='slope_seek', braker='none')
    run._slope_window = [10.0, 9.0, 8.0, 7.0]          # falling
    falling = run._progress_rate()
    run._slope_window = [7.0, 8.0, 9.0, 10.0]          # rising
    rising = run._progress_rate()
    assert falling < 0 < rising

    # same shape, 1000x the magnitude -> same rate
    run._slope_window = [10000.0, 9000.0, 8000.0, 7000.0]
    assert run._progress_rate() == pytest.approx(falling, rel=1e-9)


def test_progress_rate_declines_to_answer_on_a_short_window():
    run = _bench(climber='slope_seek', braker='none')
    run._slope_window = [1.0, 2.0]
    assert run._progress_rate() is None


def test_slope_seek_reverses_when_the_rate_gets_worse():
    """The whole rule: improved -> keep going, worse -> turn around."""
    run = _bench(climber='slope_seek', braker='none')
    run.m.step_ind = run.SLOPE['window']
    run._seek_dir, run._seek_prev = 1, -0.5
    run._slope_window = [10.0, 9.0, 8.0, 7.0]           # rate ~ -0.14, worse than -0.5
    run._slope_seek_tick()
    assert run._seek_dir == -1, 'a worse rate must flip the direction'

    run.m.step_ind = 2 * run.SLOPE['window']
    run._seek_prev = 0.5
    run._slope_window = [10.0, 9.0, 8.0, 7.0]           # better than 0.5
    run._slope_seek_tick()
    assert run._seek_dir == -1, 'a better rate must keep the direction'


def test_slope_brake_needs_consecutive_stalled_windows():
    run = _bench(climber='none', braker='slope')
    run.m.lr_controller.step()
    run.m.step_ind = 500                                  # past warmup
    run.m.lr_controller.step()
    peak0 = run.m.lr_ctrl['peak_scale']

    for k in range(1, run.SLOPE['patience'] + 1):
        run.m.step_ind = 500 + k * run.SLOPE['window']
        run._slope_window = [1.0, 1.0, 1.0, 1.0]          # flat -> stalled
        run._slope_brake_tick()
    assert run.m.lr_ctrl['peak_scale'] < peak0, 'patience consecutive stalls must cut'


def test_slope_brake_does_not_fire_while_improving():
    run = _bench(climber='none', braker='slope')
    run.m.step_ind = 500
    run.m.lr_controller.step()
    peak0 = run.m.lr_ctrl['peak_scale']
    for k in range(1, 8):
        run.m.step_ind = 500 + k * run.SLOPE['window']
        run._slope_window = [10.0, 8.0, 6.0, 4.0]
        run._slope_brake_tick()
    assert run.m.lr_ctrl['peak_scale'] == pytest.approx(peak0)


def test_every_sensor_is_held_through_warmup():
    """The envelope is deliberately moving the rate during warmup, so no sensor
    may act on what it measures there -- the same rule LRController applies to
    on_calibration and on_plateau."""
    run = _bench(climber='slope_seek', braker='none',
                 args_overrides={'adaptive_lr.warmup_steps': 1000,
                                 'ray_calibration.enabled': False})
    run.m.step_ind = 0
    run.m.lr_controller.step()
    before = run.m.lr_ctrl['peak_scale']
    run.m.step_ind = 500
    assert run._scale_peak(4.0) == 0.0, 'a sensor must be refused during warmup'
    assert run.m.lr_ctrl['peak_scale'] == pytest.approx(before)


def test_scale_peak_respects_bounds_and_the_divergence_ceiling():
    run = _bench(climber='slope_seek', braker='none',
                 args_overrides={'adaptive_lr.warmup_steps': 0,
                                 'adaptive_lr.bounds': (0.5, 2.0),
                                 'ray_calibration.enabled': False})
    run.m.step_ind = 100
    run.m.lr_controller.step()
    run.m.step_ind = 101
    for _ in range(10):
        run._scale_peak(4.0)
    assert run.m.lr_ctrl['peak_scale'] == pytest.approx(2.0)
    for _ in range(10):
        run._scale_peak(0.25)
    assert run.m.lr_ctrl['peak_scale'] == pytest.approx(0.5)
