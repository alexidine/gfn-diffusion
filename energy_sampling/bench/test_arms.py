"""
Does each arm actually DO something, and the thing its name says?

The board is a leaderboard, so an arm that silently no-ops does not error -- it
posts a plausible row. That has already happened once here: `ray+ray` armed its
probe AFTER the optimizer step, so all 1900 readings returned None and the arm
scored bit-identical to `null (no sensor)` on every metric. Nothing failed. The
tell was two rows agreeing to eight significant figures.

So the rule these tests enforce: **every controller arm must be measurably
different from the null arm**, and the sensor arms must show their sensor firing.
"""
import math

import pytest

from bench.arms import (Fixed, Hyper, HyperSNR, HyperStep, Null, RampPlateau,
                        RayRay)
from bench.board import SEED_LR, make_game
from bench.runner import Run

STEPS = 600


def _run(arm_factory, seed=0, steps=STEPS, optimizer='adam'):
    arm = arm_factory(SEED_LR)
    game = make_game(optimizer=optimizer, seed=seed)
    return Run(game, arm, seed=seed, steps=steps, batch=64).run()


def _lrs(run):
    return [h['lr'] for h in run.trace
            if h['lr'] is not None and math.isfinite(h['lr'])]


# ------------------------------------------------------------------ fixed

def test_a_fixed_arm_holds_its_rate_exactly():
    """If a fixed arm's rate moves, it is not the reference it claims to be and
    every 'relative to the best fixed rate' reading is wrong."""
    run = Run(make_game('adam', 0), Fixed(1e-3), steps=300, batch=64).run()
    lrs = set(_lrs(run))
    assert len(lrs) == 1, f'fixed rate moved: {sorted(lrs)[:5]}'
    assert abs(next(iter(lrs)) - 1e-3) < 1e-12


def test_the_min_lr_floor_cannot_truncate_the_bottom_of_the_ladder():
    """A fixed arm below the production floor must still run AT its own rate --
    otherwise two ladder rungs silently become the same rate."""
    run = Run(make_game('adam', 0), Fixed(1e-5), steps=200, batch=64).run()
    assert abs(_lrs(run)[-1] - 1e-5) < 1e-12


# ------------------------------------------------------------------- null

def test_null_does_not_move_its_rate_after_warmup():
    """Null is the do-nothing control: past the warmup envelope its rate is
    constant, so any arm that matches it is also doing nothing."""
    run = _run(Null, steps=1400)
    warm = int(run.m.args.adaptive_lr.warmup_steps)
    after = set(_lrs(run)[warm + 50:])
    assert len(after) == 1, f'null moved the rate: {sorted(after)[:5]}'


# ------------------------------------------------------------------ hyper

def test_hyper_moves_the_rate_and_respects_its_own_gain_cap():
    """Per-step moves are bounded by exp(beta) BY CONSTRUCTION -- cos is a
    cosine. Checked past warmup, where the envelope is no longer ramping and
    the only thing moving the rate is the arm."""
    run = _run(Hyper, steps=1400)
    warm = int(run.m.args.adaptive_lr.warmup_steps)
    lrs = _lrs(run)[warm + 50:]
    assert len(set(lrs)) > 10, 'hyper did not move the rate'
    jumps = [abs(math.log(b) - math.log(a)) for a, b in zip(lrs, lrs[1:])]
    # float32 round-off in the rate, not slack in the cap: the observed excess
    # is ~2e-9 on a 0.02 bound. That the max SITS at the cap is itself the
    # finding -- cos reaches exactly 1.0, i.e. consecutive gradients perfectly
    # aligned, which is what a smooth descent looks like from inside.
    assert max(jumps) <= 0.02 * (1 + 1e-6), f'exceeded its gain cap: {max(jumps)}'


def test_hyper_climbs_from_a_cold_start():
    """MK requirement (1): a cold start should ramp up."""
    run = _run(Hyper, steps=1400)
    lrs = _lrs(run)
    assert lrs[-1] > 5 * lrs[0]


# -------------------------------------------------------------------- ray

def test_the_ray_probe_actually_resolves_readings():
    """
    THE REGRESSION TEST FOR THE SILENT NO-OP. Arming after the optimizer step
    made every reading None; the arm still ran, still posted a row, and was
    identical to null. Require real readings, not just armings.
    """
    run = _run(RayRay, steps=1400)
    r = run.arm.readings
    assert r['armed'] > 0, 'probe never armed'
    resolved = {k: v for k, v in r.items() if k not in ('armed', 'none')}
    assert sum(resolved.values()) > 0, (
        f'probe armed {r["armed"]}x and resolved NOTHING: {r}. It is not a '
        f'sensor arm, it is the null arm wearing a different name.')


def test_ray_is_distinguishable_from_null():
    """Two arms agreeing to 8 significant figures is how the no-op was found."""
    ray = _run(RayRay, steps=1400)
    null = _run(Null, steps=1400)
    assert _lrs(ray)[-1] != pytest.approx(_lrs(null)[-1], rel=1e-6)


# ------------------------------------------------- hyper step (the operand)

def test_hyper_step_is_identical_to_hyper_under_sgd():
    """
    THE FALSIFIABLE CHECK ON THE OPERAND FIX. `hyper step` correlates the
    gradient against the direction actually stepped in; `hyper` correlates it
    against the previous gradient. Under plain SGD (no momentum, `_mk_opt`) the
    step IS -lr*g, so the two are the same statistic and the arms must trace
    IDENTICAL rates.

    If they differ here, the fix is doing something other than changing the
    operand, and its large Adam improvement (4.87 -> 1.04 nats) is not evidence
    about Adam preconditioning.

    MUST RUN PAST WARMUP. This test used to run 900 steps against the config's
    `warmup_steps` of 1000, so `_scale_peak` was held on all 899 calls and it
    compared two identical warmup envelopes -- `peak_scale` was exactly 1.0 at
    the end of both. It passed with the hypergradient sign REVERSED, with
    `HyperStep.tick` a total no-op, and with the operand fix undone. It had no
    power at all, while its own name claimed it was the falsifiable check.

    The tolerance is float32 round-off, not exactness: the arm reconstructs the
    step as `after - theta_before` while `Hyper` uses `g` directly, and
    `-lr*g` accumulated into a parameter is not bitwise equal to `lr*g`. The
    measured clean difference is ~6e-7 in log space; the mutations above show up
    at 4.2 and 8.6, so a 1e-4 bar separates them by four orders of magnitude.
    """
    a = _run(Hyper, steps=2500, optimizer='sgd')
    b = _run(HyperStep, steps=2500, optimizer='sgd')
    la, lb = _lrs(a), _lrs(b)
    assert len(la) == len(lb)
    moved = max(abs(math.log(x) - math.log(la[0])) for x in la)
    assert moved > 0.5, (
        f'the rate barely moved over the run (max {moved:.3g} in log space), so '
        f'this comparison is between two arms that never acted -- exactly the '
        f'hold-through-warmup trap this test fell into before')
    worst = max(abs(math.log(x) - math.log(y)) for x, y in zip(la, lb))
    assert worst < 1e-4, (
        f'under SGD the two operands are the same vector, but the rates differ '
        f'by up to {worst:.3g} in log space -- the fix is not (only) changing '
        f'the operand')


def test_hyper_step_differs_from_hyper_under_adam():
    """...and under Adam they MUST differ, or the preconditioner is not being
    picked up and the arm is a no-op rename."""
    a = _run(Hyper, steps=1400, optimizer='adam')
    b = _run(HyperStep, steps=1400, optimizer='adam')
    assert _lrs(a)[-1] != pytest.approx(_lrs(b)[-1], rel=1e-6)


# --------------------------------------------------------- ramp + plateau

def test_ramp_climbs_when_nothing_stops_it():
    run = _run(RampPlateau, steps=1400)
    lrs = _lrs(run)
    assert lrs[-1] > lrs[0], 'the ramp never ramped'
