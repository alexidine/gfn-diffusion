"""
Batch sizer tests, against train.Modeller.increment_batch_size ITSELF (bound onto
a fake modeller -- see fake_modeller.attach_real_batch_sizer).

These are the slow ones: importing train.py costs ~11 s once, for wandb,
mxtaltools and PyG. Everything after that is virtual time.

The controller's correct answer is known in closed form here (clock.knee_bound),
which is precisely what a cluster run cannot provide.

=============================================================================
RE-DERIVED 2026-08-12 against the THROUGHPUT gate
=============================================================================
These were written against `batch_growth_max_step_regression` -- accept a jump iff
`t(f*B)/t(B) <= 1+tol`. That gate was retired once the objective was stated
explicitly: priority is (1) GPU occupancy, (2) optimizer-step throughput at the
grad-accum target. Since updates/sec = samples_per_sec / accum_target, step time
does not enter, and the shipping gate is a saturation detector on samples/sec
(`batch_growth_min_throughput_gain`).

Re-derived rather than patched. Tests run at TEST_GAIN 0.30, chosen because it puts
the planted knee on the SAME RUNG the old gate did (bound 7071 vs 6250, pin 7410
either way) -- so these still ask the original questions instead of being quietly
re-baselined onto whatever the new gate happens to do.

What changed in the re-derivation:
  * the knee is found identically (7410, pinned);
  * jitter tolerance carries over with slightly different counts, and the new
    gate's errors are ONE-SIDED UPWARD -- see that test;
  * the degenerate end INVERTED. Under the old gate a tolerance >= f-1 disabled the
    gate (everything passed, batch ran to max). Under the new one it FREEZES the
    batch at the floor, because a factor-f jump cannot deliver a (1+g) gain once
    g >= f-1. `knee_bound` returned +inf for that case at first, which is exactly
    backwards, and mk_dev's inherited "or every jump passes" warning shares the
    inversion.

STILL SKIPPED: the three F-013/F-017 RECORDED-FAILURE tests. They assert specific
blind spots of the retired criterion (pins ~47% low after a step in the cost curve;
the recheck cannot escape it; the answer depends on the starting batch). Whether the
throughput gate shares them is an open experiment -- do not adjust their numbers
until the old assertions pass again, that would overwrite two measured findings with
a tautology.

bench/test_usage_floor.py covers the new priority-1 (occupancy) behaviour.
"""

import pytest

from bench.old.clock import SyntheticGPU
from bench.old.harness import BenchRun

#: F-013/F-017 -- the three RECORDED-FAILURE tests below are still skipped
#: individually. They assert specific blind spots OF THE RETIRED CRITERION (pins
#: ~47% low after a step in the cost curve; the recheck cannot escape it; the answer
#: depends on the starting batch). Whether the throughput gate shares them is an open
#: experiment, not a number to adjust until the old assertions pass again.
RETIRED_CRITERION_FINDING = pytest.mark.skip(
    reason='F-013/F-017 measured the retired step-time-regression gate; re-run as an '
           'EXPERIMENT against batch_growth_min_throughput_gain, do not patch the numbers')


#: Strictness these tests are calibrated at, NOT the shipping default (0.05).
#: Chosen so the planted knee lands on the same rung the retired step-time gate put
#: it on (bound 7071 vs 6250, expected_pin 7410 either way) -- which keeps the
#: pre-existing expected values meaningful across the criterion change instead of
#: silently re-baselining them. The shipping 0.05 puts the bound at 72727, i.e. off
#: the top of any ladder these tests can walk, so it would test nothing here.
TEST_GAIN = 0.30


def batch_run(steps=2000, gpu=None, seed=0, **args_overrides):
    """A cheap run with the sensor off -- these tests are about the batch controller."""
    overrides = {'grow_batch_size': True, 'ray_calibration.enabled': False,
                 'max_batch_size': 200000, 'max_step_seconds': 0,
                 'batch_growth_min_throughput_gain': TEST_GAIN}
    overrides.update(args_overrides)
    # the SEED MUST REACH THE CLOCK. SyntheticGPU has its own RNG, so leaving it
    # at the default made every "replicate" replay one identical timing stream --
    # eight runs that agreed because they were the same run, which reads exactly
    # like a systematic effect.
    gpu_kwargs = dict(gpu or dict(t_fixed=2.0, sps_max=5000.0))
    gpu_kwargs.setdefault('seed', seed)
    return BenchRun(
        game='mle', game_kwargs=dict(dim=4, cond=2.0, noise=0.0, lr=1e-3),
        gpu_kwargs=gpu_kwargs, args_overrides=overrides, seed=seed,
    ).run(steps, stop_on_divergence=False)


# ------------------------------------------------------------------ the knee

def test_finds_the_planted_knee():
    """
    The THROUGHPUT gate accepts a jump iff sps(f*B) >= sps(B) * (1+g), so the
    largest batch a jump still pays from is

        B_max = sps_max * t_fixed * (f - 1 - g) / (f * g)

    With t_fixed 2.0 s, sps_max 5000/s, f 1.65 and g 0.30 that is 7071, and the
    ladder from 1000 puts the first rung above it at 7410. The controller must
    land there and PIN.

    (7410 rather than 1000*1.65^4 = 7412.06: the controller re-rounds at every
    rung, so the realised ladder drifts off the ideal geometric series.
    clock.expected_pin mirrors that rounding.)

    The retired step-time gate at tol 0.25 gave bound 6250 and the SAME pin, 7410 --
    which is why TEST_GAIN is 0.30: the rung is unchanged, so this test still asks
    the original question rather than a re-baselined one.
    """
    gpu = SyntheticGPU(t_fixed=2.0, sps_max=5000.0)
    bound = gpu.knee_bound(1.65, TEST_GAIN)
    expected = gpu.expected_pin(1000, 1.65, TEST_GAIN)
    assert bound == pytest.approx(7070.7, rel=1e-3)
    assert expected == 7410

    run = batch_run(steps=3000)
    assert run.m.batch_size == expected, (
        f'pinned {run.m.batch_size}, analytic first-rung-above-knee {expected}')
    assert run.m.batch_size_saturated_stage == 'naive', 'must pin, not keep walking'

    # the pin is systematically one growth factor HOT: a rung is only convicted
    # after the controller has moved past it, so prev_batch is itself above the bound
    assert bound < run.m.batch_size <= bound * 1.65


def test_pin_stays_within_one_rung_under_realistic_timing_jitter():
    """
    The knee decision is intrinsically MARGINAL at the boundary rung: the ladder is
    coarse (1.65x) next to the decision band, so at the decisive rung the measured
    gain is +29% against a +30% threshold -- a 1% margin. Averaging over 20 timings
    is what holds it together.

    RE-MEASURED against the throughput gate, 10 clock seeds, TEST_GAIN 0.30
    (the retired step-time gate at tol 0.25 in brackets):

        jitter 0.00 -> 7410 x10                        (7410 x10)
        jitter 0.05 -> 7410 x8,  12226 x2              (7410 x9,  12226 x1)
        jitter 0.10 -> 7410 x8,  12226 x2              (7410 x7,  12226 x2, 4491 x1)
        jitter 0.20 -> 7410 x6,  12226 x4              (7410 x8,  12226 x2)
        jitter 0.40 -> scattered 1000..12226           (scattered, same)

    THE FINDING CARRIES OVER: modal answer correct to ~20% jitter, worst case one
    rung, collapse at 0.40. Two differences worth noting. The new gate is slightly
    looser at 0.20 (6/10 vs 8/10), and its errors are ONE-SIDED UPWARD -- it never
    lands at 4491, whereas the old gate did. Under the current priority order
    (occupancy first) erring high is the safe direction, so this is a mild
    improvement in character even where it is worse in count.
    """
    for jitter in (0.05, 0.10, 0.20):
        pins = [batch_run(steps=3000,
                          gpu=dict(t_fixed=2.0, sps_max=5000.0, jitter=jitter, seed=100 + s),
                          seed=s).m.batch_size
                for s in range(10)]
        assert all(p in (4491, 7410, 12226) for p in pins), (jitter, pins)
        assert max(set(pins), key=pins.count) == 7410, (jitter, pins)


def test_gain_at_or_above_factor_minus_one_rejects_every_jump():
    """
    THE SEMANTICS INVERTED with the criterion, and the old config warning is now
    backwards.

    Under the retired step-time gate a tolerance >= f-1 meant every jump passed
    (the gate was disabled and the batch ran to max_batch_size). Under the
    throughput gate the same arithmetic means the opposite: a factor-f jump can
    never deliver a (1+g) throughput gain once g >= f-1, because throughput
    saturates at sps_max. So NO jump clears and the batch pins immediately at the
    floor -- the controller is not disabled, it is frozen.

    Both ends asserted, since they are easy to confuse:
        g = 0     -> nothing demanded -> +inf -> runs to max_batch_size
        g >= f-1  -> more demanded than possible -> 0 -> pins at the floor
    """
    gpu = SyntheticGPU(t_fixed=2.0, sps_max=5000.0)
    assert gpu.knee_bound(1.65, 0.0) == float('inf'), 'no gain demanded = never pins'
    assert gpu.knee_bound(1.65, 0.65) == 0.0, 'unreachable gain = pins immediately'

    frozen = batch_run(steps=3000, batch_growth_min_throughput_gain=0.65,
                       max_batch_size=50000)
    assert frozen.m.batch_size == 1000, (
        f'expected the floor, got {frozen.m.batch_size} -- an unreachable gain must '
        f'reject every jump, not disable the gate')

    unbounded = batch_run(steps=3000, batch_growth_min_throughput_gain=0.0,
                          max_batch_size=50000)
    assert unbounded.m.batch_size == 50000, 'a zero gain must never pin'


def test_flat_throughput_walks_down_only_to_the_floor():
    """
    In a regime with no knee (t_fixed = 0, so throughput is flat in batch) every
    jump fails and the periodic recheck walks the batch down a rung at a time
    forever. The floor -- the configured batch_size -- is what stops it, and it
    is load-bearing rather than defensive.
    """
    run = batch_run(steps=6000, gpu=dict(t_fixed=0.0, sps_max=5000.0),
                    batch_knee_recheck_steps=300)
    assert run.m.batch_size >= 1000, 'walked below the configured floor'


# ------------------------------------------------------------------- the OOM

def test_never_walks_back_into_the_oom_ceiling():
    """
    The size that OOM'd is the most informative reading the controller ever gets.
    Before it was remembered, the post-OOM baseline reset made every subsequent
    jump a blind re-entry -- prod0810 mipcas_elj ran that sawtooth
    (6113->10086->OOM->5043->8321->OOM->...) for the rest of the run.
    """
    # t_fixed 10 s puts the knee bound at 31250, well above the 4000 OOM ceiling,
    # so the walk is stopped by memory rather than by throughput -- which is the
    # case this test is about. (With t_fixed 0 there is no knee at all, the first
    # jump pins at 1650, and the ceiling is never reached.)
    run = batch_run(steps=4000, gpu=dict(t_fixed=10.0, sps_max=5000.0, oom_at=4000),
                    batch_knee_recheck_steps=0)
    assert run.oom_steps >= 1
    assert run.m.batch_size_oom_ceiling is not None
    assert run.m.batch_size < run.m.batch_size_oom_ceiling
    assert run.oom_steps <= 2, (
        f'{run.oom_steps} OOMs means the walk re-entered the ceiling (sawtooth)')


def test_growth_is_never_blind_after_an_oom():
    """
    handle_train_epoch_error clears _rung_throughput. The old code fell through
    to an unconditional multiply whenever the baseline was None, so after the
    first OOM every jump was taken with no measurement. A rung with no baseline
    must be MEASURED first and grown on the next interval.
    """
    run = batch_run(steps=1200, gpu=dict(t_fixed=0.0, sps_max=5000.0, oom_at=4000))
    m = run.m
    m.batch_size, m._rung_throughput = 2000, None
    m.batch_size_cooldown_until, m.batch_size_last_grow = -1, 0
    m.batch_size_oom_ceiling, m.batch_size_saturated_stage = None, None
    m.step_ind = 10_000
    m.increment_batch_size()
    assert m.batch_size == 2000, 'grew with no baseline'
    assert m._rung_throughput is not None, 'must record the rung it just measured'


def test_oom_leaves_exactly_one_poisoned_timing_entry():
    """
    handle_train_epoch_error CLEARS the timing deques (train.py:3662), and the
    outer loop then appends the step that just failed (train.py:1943) -- so the
    window restarts holding one entry that pairs the PRE-CUT batch size with the
    time it took to fail.

    That single entry overstates throughput for the new, smaller rung. It cannot
    move the controller on its own: the gate needs 10 timings before it computes
    a median, by which point 9 honest entries have joined it and the deque is
    dominated by the true rung. Pinned so that stays true if the window shrinks.
    """
    gpu = SyntheticGPU(t_fixed=2.0, sps_max=5000.0, oom_at=1000)
    run = BenchRun(game='mle', game_kwargs=dict(dim=4, cond=2.0, noise=0.0, lr=1e-3),
                   gpu_kwargs=dict(t_fixed=2.0, sps_max=5000.0, oom_at=1000),
                   args_overrides={'grow_batch_size': True, 'ray_calibration.enabled': False,
                                   'max_step_seconds': 0})
    # handle_train_epoch_error returns early at step_ind == 0 (an OOM on the very
    # first step is a config error, not a growth overshoot), so start past it
    run.m.step_ind = 5
    run.step()
    m = run.m
    assert run.oom_steps == 1
    assert len(m._recent_step_times) == 1 and len(m._recent_step_work) == 1
    assert m._recent_step_work[0] == 1000, 'work is charged at the attempted (pre-cut) batch'
    assert m._recent_step_times[0] == pytest.approx(gpu.true_step_time(1000) * 0.1)
    assert m.batch_size == 500, 'batch must have been cut'
    # the gate refuses to act on fewer than 10 timings, which is what contains this
    assert len(m._recent_step_times) < 10


# ------------------------------------------------------ wall-clock ceiling

def test_max_step_seconds_cuts_proportionally():
    """
    Step cost is close enough to linear in batch that the overshoot ratio
    estimates the target directly. A fixed /f ladder converges far too slowly in
    exactly the case that matters -- a 181 s step needs 4 cuts at
    oom_cooldown_steps apart, i.e. ~40 h of 181 s steps.
    """
    run = batch_run(steps=200, gpu=dict(t_fixed=1.0, sps_max=100.0),
                    max_step_seconds=10, fused_grad_accum_min_samples=0)
    # t(1000) = 11 s against a 10 s ceiling -> min(1000/1.65, 1000*(10/11)*0.9) = 606
    assert run.m.batch_size == 606
    assert run.m.batch_size_cooldown_until > 0, 'a cut must start a cooldown'


def test_accumulation_floor_blocks_the_wall_clock_ceiling():
    """
    A REAL INTERACTION IN THE SHIPPING CONFIG. mk_dev sets batch_size 1000 and
    fused_grad_accum_min_samples 1000. Below the accumulation floor a fused step
    is a micro-step, so cutting the batch buys proportionally more micro-steps
    for the same samples and time per OPTIMIZER UPDATE does not fall -- the code
    therefore refuses to cut below it.

    Consequence: in a fused stage at the base batch, max_step_seconds cannot cut
    at all. It only has authority over batch sizes the growth walk added on top.
    """
    run = batch_run(steps=200, gpu=dict(t_fixed=1.0, sps_max=100.0),
                    max_step_seconds=10, fused_grad_accum_min_samples=1000)
    assert run.m.batch_size == 1000, 'the accumulation floor should have blocked the cut'

    # above the floor it has authority again
    grown = batch_run(steps=200, gpu=dict(t_fixed=1.0, sps_max=100.0),
                      max_step_seconds=10, fused_grad_accum_min_samples=500)
    assert grown.m.batch_size < 1000


def test_max_step_seconds_is_checked_in_both_directions():
    """An oversized batch inherited across a transition (or restored from a
    checkpoint) has to be able to come back down; the knee walk alone only ever
    moves up from where it starts."""
    run = BenchRun(
        game='mle', game_kwargs=dict(dim=4, cond=2.0, noise=0.0, lr=1e-3),
        gpu_kwargs=dict(t_fixed=1.0, sps_max=100.0),
        args_overrides={'grow_batch_size': True, 'ray_calibration.enabled': False,
                        'max_step_seconds': 10, 'fused_grad_accum_min_samples': 0,
                        'max_batch_size': 200000},
    )
    run.m.batch_size = 50000            # inherited, absurdly slow
    run.run(200, stop_on_divergence=False)
    assert run.m.batch_size < 50000


# ------------------------------------------------------------------ recheck

# --------------------------------------------------- discrete cost models

def test_wave_quantisation_makes_throughput_sawtooth():
    """A partial wave costs a whole one, so throughput is NOT monotone in batch
    -- which is the property the smooth model lacks."""
    gpu = SyntheticGPU(t_fixed=2.0, sps_max=5000.0, tile=1024)
    sps = [gpu.throughput(b) for b in range(900, 4200, 100)]
    drops = sum(1 for i in range(1, len(sps)) if sps[i] < sps[i - 1])
    assert drops >= 3, 'tiling must make throughput non-monotone'

    smooth = SyntheticGPU(t_fixed=2.0, sps_max=5000.0)
    smooth_sps = [smooth.throughput(b) for b in range(900, 4200, 100)]
    assert all(smooth_sps[i] > smooth_sps[i - 1] for i in range(1, len(smooth_sps)))


def test_knee_bound_refuses_a_discrete_model():
    """The closed form is the smooth special case. Returning a number for a
    tiled clock would be quietly meaningless."""
    gpu = SyntheticGPU(t_fixed=2.0, sps_max=5000.0, tile=256)
    with pytest.raises(ValueError, match='SMOOTH model'):
        gpu.knee_bound(1.65, 0.30)
    assert gpu.expected_pin(1000, 1.65, 0.30) is not None


@pytest.mark.parametrize('gpu_kwargs,label', [
    (dict(tile=256), 'wave quantisation'),
    (dict(tile=256, recompile_s=40.0), 'quantisation + recompile stalls'),
    (dict(tile=256, regimes=[(4096, 0.8)]), 'quantisation + a kernel switch'),
])
def test_controller_tracks_whatever_the_cost_model_says(gpu_kwargs, label):
    """
    Mild discreteness does not break tracking: the controller lands where
    expected_pin (a walk against the actual cost model) says it should.

    Note the recompile stall does NOT corrupt the decision at the shipping
    growth interval: it is charged on the first step at a new size, and the gate
    medians the last 20 timings 50 steps later, by which point it has aged out.
    """
    kw = dict(t_fixed=2.0, sps_max=5000.0, **gpu_kwargs)
    predicted = SyntheticGPU(**kw).expected_pin(1000, 1.65, 0.30)
    pins = [batch_run(steps=4000, gpu=dict(kw, seed=100 + s), seed=s).m.batch_size
            for s in range(3)]
    assert all(p == predicted for p in pins), (label, predicted, pins)


@RETIRED_CRITERION_FINDING
def test_a_step_in_the_cost_curve_pins_the_batch_far_too_low():
    """
    THE CRITERION'S REAL LIMIT. The gate is a LOCAL two-point comparison, so a
    one-off step in the cost curve between two rungs is indistinguishable from
    saturation. With a kernel-switch efficiency drop at 2722, the walk trips
    there and pins at 1650 -- while throughput at 7410 is still more than double.

    The controller is not misbehaving; it is answering the question it was asked.
    The question is local, and the cost curve is not.
    """
    kw = dict(t_fixed=2.0, sps_max=5000.0, regimes=[(2722, 0.5)])
    gpu = SyntheticGPU(**kw)
    assert gpu.expected_pin(1000, 1.65, 0.30) == 1650

    run = batch_run(steps=4000, gpu=kw)
    assert run.m.batch_size == 1650
    assert gpu.throughput(7410) > 2 * gpu.throughput(1650), (
        'the setup is meant to leave a lot of throughput on the table')


@RETIRED_CRITERION_FINDING
def test_the_recheck_cannot_escape_a_premature_pin():
    """
    `batch_knee_recheck_steps` drops ONE rung and re-climbs, so it re-tests the
    same failing comparison and re-pins in the same place. It adapts to a knee
    that MOVED; it cannot discover that the knee was never there.
    """
    kw = dict(t_fixed=2.0, sps_max=5000.0, regimes=[(2722, 0.5)])
    run = batch_run(steps=20000, gpu=kw, batch_knee_recheck_steps=1500)
    assert run.m.batch_size == 1650, 'recheck escaped -- update this finding'


@RETIRED_CRITERION_FINDING
def test_the_gate_is_path_dependent_on_a_non_monotone_curve():
    """
    Same cost model, same config, different STARTING batch -> different pin.
    Measured across the ladder: start 1000 pins 1650 (40% of the best available
    throughput), start 2722 pins 4491 (67%), start 7410 pins 7410 (84%).

    So on a non-monotone curve `batch_size` is not merely a floor and a starting
    point -- it selects the answer. With a monotone step-time ratio (the smooth
    model) the walk is path-INdependent, which is why this never shows up there.
    """
    kw = dict(t_fixed=2.0, sps_max=5000.0, regimes=[(2722, 0.5)])
    pins = {}
    for start in (1000, 2722, 7410):
        run = BenchRun(
            game='mle', game_kwargs=dict(dim=4, cond=2.0, noise=0.0, lr=1e-3),
            gpu_kwargs=dict(kw, seed=0),
            args_overrides={'grow_batch_size': True, 'ray_calibration.enabled': False,
                            'max_batch_size': 200000, 'max_step_seconds': 0,
                            'batch_size': start},
        ).run(6000, stop_on_divergence=False)
        pins[start] = run.m.batch_size

    assert pins[1000] < pins[2722] < pins[7410], pins
    gpu = SyntheticGPU(**kw)
    assert gpu.throughput(pins[7410]) > 2 * gpu.throughput(pins[1000])

    # ...and on the SMOOTH model the same sweep agrees, so this is a property of
    # the cost curve rather than of the walk
    smooth = dict(t_fixed=2.0, sps_max=5000.0)
    smooth_pins = set()
    for start in (1000, 2722):
        run = BenchRun(
            game='mle', game_kwargs=dict(dim=4, cond=2.0, noise=0.0, lr=1e-3),
            gpu_kwargs=dict(smooth, seed=0),
            args_overrides={'grow_batch_size': True, 'ray_calibration.enabled': False,
                            'max_batch_size': 200000, 'max_step_seconds': 0,
                            'batch_size': start},
        ).run(6000, stop_on_divergence=False)
        smooth_pins.add(run.m.batch_size)
    assert smooth_pins == {7410}, smooth_pins


def test_knee_recheck_reclimbs_to_the_same_pin():
    """
    The knee moves WITHIN a stage as the fused composition drifts, so a pin
    decays: drop one rung and re-climb. A healthy re-climb must re-pin at the
    same place, not ratchet downward.
    """
    run = batch_run(steps=12000, batch_knee_recheck_steps=1500)
    assert run.m.batch_size == 7410, f'recheck ratcheted the pin to {run.m.batch_size}'
