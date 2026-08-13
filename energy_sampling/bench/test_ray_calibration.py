"""
Sensor tests. These need neither train.py nor a GPU and run in about a second.

The headline is test_due_latches_across_accumulation_cycles: that bug was found
by running 3000 real steps four times on the cluster. It is an integer-clock
coincidence and costs 20 ms to reproduce here.
"""

import math

import pytest
import torch

from bench.surfaces import MLEGame
from energy_sampling.ray_calibration import RayCalibration


# ---------------------------------------------------------------- construction

def test_alphas_must_include_zero():
    with pytest.raises(ValueError, match='must include 0.0'):
        RayCalibration([], alphas=(1.0, 2.0, 4.0))


def test_period_must_be_multiple_of_ten():
    """Metrics drain on a 10-step clock; a period that aliases hides calibrations."""
    with pytest.raises(ValueError, match='multiple of 10'):
        RayCalibration([], period=333)


# ------------------------------------------------------------------ the latch

@pytest.mark.parametrize('cycle', [1, 2, 3, 4, 7, 8])
def test_due_latches_across_accumulation_cycles(cycle):
    """
    `due` must stay latched from the moment a calibration falls due until one
    COMPLETES, so the count does not depend on a coincidence between the
    calibration period and the gradient-accumulation cycle.

    With an exact-modulo trigger, a boundary k*period is a stepping step only
    when the cycle DIVIDES the period. Measured on the cluster at period 500 over
    3000 steps: cycle 4 gave 6 calibrations, cycle 8 gave 3, cycle 3 gave 2, and
    cycle 7 gave ZERO -- peak_scale never moved while the config said adaptive.

    Latching removes the coincidence rather than making it less likely, so every
    cycle must give the same full count -- but note the count is only equal once
    the run is long enough for the LAST boundary to be satisfied. Satisfaction is
    delayed by up to one accumulation cycle (asserted below), so a window that
    ends exactly on a boundary loses the final reading to latency, not to the
    trigger. At cycle 7 the 6th calibration of a 3000-step window lands at step
    3003.
    """
    period, boundaries = 500, 6
    steps = period * boundaries + cycle          # room for the last one to land
    cal = RayCalibration([], period=period, enabled=True)
    completed, latency = [], []
    for step in range(1, steps + 1):
        if not cal.due(step):
            continue
        if step % cycle != 0:
            continue          # mid-accumulation: no optimizer step to rate
        # a calibration completed -- this is what measure() does on success,
        # keyed on the armed step so a delayed reading consumes its boundary
        cal._last_done = step // period
        completed.append(step)
        latency.append(step - (len(completed)) * period)

    assert len(completed) == boundaries, (
        f'accumulation cycle {cycle} produced {len(completed)} calibrations; '
        f'the latch must make this independent of the cycle')
    assert max(latency) < cycle, (
        f'satisfaction must lag its boundary by less than one accumulation '
        f'cycle, saw {max(latency)} at cycle {cycle}')


def test_due_fires_on_the_first_new_period_index():
    """
    The first call pins _last_done to whatever index it lands on, so the trigger
    fires when step_ind crosses into a STRICTLY GREATER index -- not at the first
    boundary in absolute terms. A run resuming mid-period therefore waits for the
    next boundary rather than calibrating immediately.
    """
    cal = RayCalibration([], period=100, enabled=True)
    assert cal.due(50) is False       # first sight: idx 0, pinned, no fire
    assert cal.due(99) is False
    assert cal.due(150) is True       # idx 1 > 0

    resumed = RayCalibration([], period=100, enabled=True)
    assert resumed.due(1250) is False  # first sight at idx 12
    assert resumed.due(1299) is False
    assert resumed.due(1300) is True   # idx 13


def test_disabled_never_due():
    cal = RayCalibration([], period=100, enabled=False)
    assert not any(cal.due(s) for s in range(1, 1000))


# ----------------------------------------------------------------- the reading

def _measure_once(game, lr, n_sub=16, probe_batch=4096,
                  alphas=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)):
    """Run one arm/step/measure cycle at a fixed LR and return (reading, alpha_true)."""
    cal = RayCalibration(game.policy_params, alphas=alphas, n_sub=n_sub,
                         period=10, enabled=True)
    cal._last_done = -1                       # force due
    assert cal.arm(10)
    before = game.policy_params[0].detach().clone()

    for g in game.optimizers['fwd'].param_groups:
        g['lr'] = lr
    game.train_step(game.draw(probe_batch))

    d = game.policy_params[0].detach() - before
    alpha_true = game.alpha_star_true(before, d)
    reading = cal.measure(lambda: game.draw(probe_batch), game.probe_loss)
    return reading, alpha_true


@pytest.mark.parametrize('lr,expect', [
    (3.0e-2, 'in_range'),
    (1.0e-4, 'above_range'),     # far too cold: alpha* past the grid edge
])
def test_bracket_contains_true_alpha_star(lr, expect):
    """
    The bracket must contain the analytically known alpha*. This is the property
    the whole controller rests on, and on a real problem it is unobservable.
    """
    game = MLEGame(dim=16, cond=8.0, noise=0.02, lr=lr, init_scale=1.0, seed=3)
    reading, alpha_true = _measure_once(game, lr)
    assert reading is not None
    lo, hi = reading['lo'], reading['hi']

    if expect == 'above_range':
        assert reading['status'] == 'above_range'
        assert alpha_true > lo, f'true alpha* {alpha_true:.3g} below the bound {lo}'
        return

    assert reading['status'] == 'bracketed', reading['status']
    assert lo <= alpha_true <= hi, (
        f'true alpha* {alpha_true:.4g} outside bracket [{lo}, {hi}]')
    # the geometric-mean point estimate is what the controller acts on
    assert lo <= reading['alpha_star'] <= hi


def test_measure_restores_params_bitwise():
    """
    Every parameter touched is restored to theta_after EXACTLY. A sensor that
    perturbs training is not a sensor.
    """
    game = MLEGame(dim=32, cond=10.0, noise=0.05, lr=1e-2, seed=5)
    cal = RayCalibration(game.policy_params, n_sub=4, period=10, enabled=True)
    cal._last_done = -1
    assert cal.arm(10)
    game.train_step(game.draw(512))
    after = game.policy_params[0].detach().clone()

    cal.measure(lambda: game.draw(512), game.probe_loss)
    assert torch.equal(game.policy_params[0].detach(), after), \
        'measure() must restore parameters bitwise, not approximately'


def test_pairing_cancels_per_batch_level():
    """
    L(2a) and L(0) are evaluated on the SAME sub-batch, so a per-batch constant
    -- which varies by hundreds of nats across conditions in the real run --
    cancels exactly in the difference.

    Injecting a huge random offset per sub-batch must not change the bracket.
    """
    torch.manual_seed(0)
    game = MLEGame(dim=16, cond=8.0, noise=0.02, lr=3e-2, init_scale=1.0, seed=3)
    clean, _ = _measure_once(game, 3e-2)

    torch.manual_seed(0)
    game2 = MLEGame(dim=16, cond=8.0, noise=0.02, lr=3e-2, init_scale=1.0, seed=3)
    offsets = iter([1e4 * (i - 8) for i in range(64)])
    base_draw, base_loss = game2.draw, game2.probe_loss
    pending = {}

    def draw(bs):
        b = base_draw(bs)
        pending[id(b)] = next(offsets)
        return b

    def loss(b):
        return base_loss(b) + pending.get(id(b), 0.0)

    game2.draw, game2.probe_loss = draw, loss
    shifted, _ = _measure_once(game2, 3e-2)

    assert clean['lo'] == shifted['lo'] and clean['hi'] == shifted['hi'], \
        'per-batch level did not cancel: pairing is broken'


def test_deferred_when_no_step_taken():
    """Mid-accumulation there is no step to rate -- deferred, not skipped."""
    game = MLEGame(dim=8, lr=1e-2, seed=1)
    cal = RayCalibration(game.policy_params, n_sub=4, period=10, enabled=True)
    cal._last_done = -1
    assert cal.arm(10)
    reading = cal.measure(lambda: game.draw(256), game.probe_loss)   # no optimizer step
    assert reading is None
    assert cal.n_deferred == 1 and cal.n_skipped == 0


def test_status_codes_are_explicit():
    """Positional encodings silently re-map historical logs when reordered."""
    assert RayCalibration._STATUS == {
        'unresolved': 0, 'bracketed': 1, 'above_range': 2,
        'below_range': 3, 'inconsistent': 4, 'warmup': 5}


def test_top_of_grid_is_a_bound_not_an_estimate():
    """
    With alphas up to 64 the largest testable alpha is 32, so a colder-than-grid
    run reports alpha_star = 32 as a BOUND. It must never be extrapolated -- the
    controller's response then saturates instead of running away.
    """
    game = MLEGame(dim=16, cond=8.0, noise=0.01, lr=1e-6, init_scale=1.0, seed=3)
    reading, alpha_true = _measure_once(game, 1e-6)
    assert reading['status'] == 'above_range'
    assert reading['alpha_star'] == 32.0
    assert alpha_true > 100 * 32.0, 'this setup should be far outside the grid'
