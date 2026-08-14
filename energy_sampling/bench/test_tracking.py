"""
WHAT THE FOUNDATION SURFACE MUST DEMONSTRATE, stated before it is used.

`TrackingGame` is the base construction for the whole controller bench, so its
properties are asserted here rather than assumed. Four requirements; each one
corresponds to a way the previous surface failed:

  1. A SHARP INTERIOR OPTIMUM.        The old equilibration surface had a 30x
                                      band of rates all within 0.4 nats, so every
                                      arm landed in it and tied.
  2. THE OPTIMUM MOVES WITH A KNOB.   Without this, an arm that lands in a good
                                      place by luck is indistinguishable from one
                                      that tracks. This is the only property that
                                      separates control from a lucky shape.
  3. IT HOLDS UNDER ADAM.             Production runs Adam on every optimizer.
                                      The old board was SGD on 11 of 12 cells.
  4. SEED NOISE MUCH SMALLER THAN     The old board's 5-seed noise was 0.054-0.065
     THE GAPS.                        nats against gaps of 0.01-0.06, and it
                                      reproduced its own top-5 ordering 3-7% of
                                      the time.
"""
import math

import numpy as np
import pytest

from bench.arms import Fixed
from bench.metrics import score_run
from bench.runner import Run
from bench.surfaces import TrackingGame

STEPS = 6000
LADDER = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def _score(lr, seed=0, **kw):
    """THROUGH `score_run`, which is what the board calls.

    An earlier version called `final_loss` directly and so silently used the
    DEFAULT 100-step window rather than the surface's own `score_window`. It
    measured 2.4 sigma of separation where the real scoring path gives ~15 --
    a test of a code path the bench does not take.
    """
    g = TrackingGame(lr=lr, seed=seed, **kw)
    r = Run(g, Fixed(lr), seed=seed, steps=STEPS, batch=64).run()
    return score_run(r)['final_loss']


def _curve(seeds=(0, 1, 2), **kw):
    return {lr: float(np.median([_score(lr, seed=s, **kw) for s in seeds]))
            for lr in LADDER}


def _nats(curve):
    best = min(curve.values())
    return {lr: math.log(v / best) for lr, v in curve.items()}


@pytest.fixture(scope='module')
def slow():
    return _curve(speed=1e-3)


@pytest.fixture(scope='module')
def fast():
    return _curve(speed=1e-2)


def test_there_is_a_sharp_interior_optimum(slow):
    """
    Wrong by 30x must COST something, and the best rate must not be a ladder
    edge -- a ladder reporting its own edge is reporting nothing.
    """
    n = _nats(slow)
    best = min(n, key=n.get)
    assert best not in (LADDER[0], LADDER[-1]), (
        f'best rate {best:g} is a ladder edge; the ladder does not bracket the '
        f'optimum, so this measures the ladder rather than the surface')
    assert max(n.values()) > 2.0, (
        f'the whole ladder spans {max(n.values()):.2f} nats -- being wrong by '
        f'1000x barely costs anything and no controller can show skill here')
    band = [lr for lr, v in n.items() if v <= math.log(2)]
    assert max(band) / min(band) <= 30.0, (
        f'the rates within 2x of best span {max(band) / min(band):.0f}x; arms '
        f'all land inside and tie')


def test_the_optimum_moves_with_the_target_speed(slow, fast):
    """
    THE PROPERTY THAT MAKES THIS A TEST OF CONTROL.

    Both errors scale with the rate: lag ~ speed/lr, jitter ~ lr*sigma. So the
    optimum goes as sqrt(speed*sigma) and a 10x faster target should want a
    materially hotter rate. Without this the board cannot tell an arm that
    TRACKS from an arm that happens to sit somewhere reasonable -- which is
    exactly the ambiguity that made every verdict on the previous surface
    unusable.
    """
    a = min(_nats(slow), key=_nats(slow).get)
    b = min(_nats(fast), key=_nats(fast).get)
    assert b > a, (
        f'a 10x faster target wanted rate {b:g} against {a:g} for the slow one '
        f'-- the optimum does not move, so tracking is untestable here')
    assert b / a >= 3.0, (
        f'the optimum moved only {b / a:.1f}x for a 10x change in target speed; '
        f'too little to separate tracking from luck against seed noise')


def test_it_holds_under_adam_which_is_what_production_runs():
    """
    Adam is the default here precisely because the previous board ran SGD on 11
    of 12 cells while `train.py:1647` builds Adam for all five optimizer keys.
    Under Adam the hypergradient statistic has no zero crossing, so a surface
    that only works under SGD hides the single largest controller failure.
    """
    n = _nats(_curve(speed=1e-3, optimizer='adam'))
    assert max(n.values()) > 2.0
    assert min(n, key=n.get) not in (LADDER[0], LADDER[-1])


def test_seed_noise_is_far_below_the_gaps_between_rungs():
    """
    The gap between neighbouring rungs must clear the seed noise by a wide
    margin, or a ranking is noise. This is the check the previous board failed:
    0.054-0.065 nats of noise against 0.01-0.06 nats of gap.
    """
    lrs = (3e-4, 1e-3, 3e-3)
    per = {lr: [_score(lr, seed=s, speed=1e-3) for s in range(5)] for lr in lrs}
    noise = max(float(np.std(np.log(v))) for v in per.values())
    med = [float(np.median(per[lr])) for lr in lrs]
    gaps = [abs(math.log(b / a)) for a, b in zip(med, med[1:])]
    assert noise > 0, 'zero seed noise means the seeds are not independent'
    assert min(gaps) / noise > 5.0, (
        f'smallest adjacent gap {min(gaps):.3f} nats against {noise:.3f} nats '
        f'of seed noise ({min(gaps) / noise:.1f} sigma) -- not separable')


def test_the_target_path_is_common_across_seeds():
    """Common random numbers for the target, exactly as for the gradient noise.
    A per-seed path is a second macroscopic noise source -- measured at 8x on the
    surface where it was first tried."""
    import torch
    a, b = TrackingGame(seed=0), TrackingGame(seed=11)
    for _ in range(50):
        a.advance(); b.advance()
    assert torch.equal(a.target, b.target)
    assert float(a.target.norm()) > 0, 'the target never moved'
