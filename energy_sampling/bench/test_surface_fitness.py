"""
IS THE SURFACE FIT TO RANK CONTROLLERS? The properties, as tests.

Four adversarial reviews found the equilibration surface was not doing what its
docstring said, and none of the failures were subtle once measured -- they were
simply never measured. These are the checks that should have existed before any
arm was ranked on it. They are slow by test standards (a few seconds each) and
worth it: every one of them corresponds to a defect that shipped.

The properties, and the defect each one pins:

  FLOOR WIDTH    the band of rates within 2x of the best must be NARROW. On the
                 shipped surface it was 30x -- every controller landed inside it
                 and tied, so the board was a divergence detector with a
                 leaderboard's formatting.
  STATIONARY     the outcome must not be systematically moving at the horizon.
                 Descending means arms are ranked on convergence SPEED and the
                 ranking moves with the budget; rising means the score depends on
                 where one realisation of the target path happened to end.
  BWD IS A PLAYER  a branch supplying 0.3% of the gradient is not one of three
                 competing optimisations. `cond_bwd` was never set, so the whole
                 opposed-spectra design was inert in every cell.
  SEED NOISE     must be small against the gap between adjacent rungs, or no
                 number of arms resolves anything. Measured on the shipped
                 surface: gaps ~0.1 nats against 0.068 nats of noise, and the
                 board reproduced its own top-5 ordering 3-7% of the time.
"""
import math

import numpy as np
import pytest

from bench.arms import Fixed
from bench.metrics import final_loss
from bench.runner import Run
from bench.surfaces import EquilibrationGame

#: THE REDESIGNED CELL. `cond_bwd` activates the opposed spectra that shipped
#: inert; `drift`/`drift_pull` make the target a stationary OU process, which is
#: what buys a narrow floor without an unsettled run.
FIT = dict(dim=8, a=2.0, b=1.0, w_rep=0.7, w_bwd=0.3, kappa=0.02, noise=0.1,
           init_scale=1.0, cond_rep=100.0, cond_bwd=100.0,
           drift=0.01, drift_pull=0.01)
STEPS = 6000
RUNGS = (3.2e-3, 5.6e-3, 1e-2, 1.8e-2)


def _end(lr, seed, **over):
    g = EquilibrationGame(lr=lr, optimizer='sgd', seed=seed, **{**FIT, **over})
    r = Run(g, Fixed(lr), seed=seed, steps=STEPS, batch=64).run()
    return math.inf if r.aborted else final_loss(r), g, r


@pytest.fixture(scope='module')
def rungs():
    return {lr: [_end(lr, s)[0] for s in range(3)] for lr in RUNGS}


def test_the_good_band_is_narrow_enough_to_separate_arms(rungs):
    """
    THE PROPERTY THE WHOLE BOARD RESTS ON. Every controller settles somewhere in
    the band of rates that work; if that band is wide and flat, they all score
    the same and the ranking is seed noise wearing a leaderboard.

    Measured on the shipped surface: rates from 0.001 to 0.03 -- a 30x span --
    all landed within 0.38 nats of each other.
    """
    med = {lr: float(np.median(v)) for lr, v in rungs.items()}
    best = min(med.values())
    band = [lr for lr, v in med.items() if math.log(v / best) <= math.log(2)]
    width = max(band) / min(band)
    assert width <= 6.0, (
        f'the rates within 2x of best span {width:.0f}x; arms all land in there '
        f'and tie, whatever the seed count')


def test_adjacent_rates_are_separable_against_seed_noise(rungs):
    """
    The gap between neighbouring rungs must clear the seed noise by a wide
    margin, or the board cannot tell a controller that found the optimum from
    one that landed a factor away.

    This is the quantitative form of the review finding that the battery
    reproduced its own top-5 ordering 3-7% of the time.
    """
    lrs = sorted(rungs)
    noise = max(float(np.std(np.log(rungs[lr]))) for lr in lrs)
    med = [float(np.median(rungs[lr])) for lr in lrs]
    gaps = [abs(math.log(b / a)) for a, b in zip(med, med[1:])]
    assert noise > 0, 'zero seed noise means the seeds are not independent'
    assert min(gaps) / noise > 4.0, (
        f'smallest gap between adjacent rates is {min(gaps):.3f} nats against '
        f'{noise:.3f} nats of seed noise ({min(gaps) / noise:.1f} sigma) -- '
        f'arms one rung apart are not distinguishable')


def test_the_outcome_is_stationary_at_the_horizon():
    """
    Neither descending nor degrading. Descending means the cell ranks arms by
    how fast they got hot and the answer changes with the budget; degrading
    means a non-stationary target, where the score depends on where one
    realisation of the path happened to end. A pure random-walk target scored
    1.7x WORSE in the last fifth than the fifth before, which is why the target
    is mean-reverting.
    """
    _, _, r = _end(1e-2, 0)
    el = [h['eloss'] for h in r.trace if h['eloss']]
    fifth = len(el) // 5
    tail, prev = np.median(el[-fifth:]), np.median(el[-2 * fifth:-fifth])
    ratio = float(prev / tail)
    assert 0.8 <= ratio <= 1.5, (
        f'last fifth vs the one before it is {ratio:.2f}x -- '
        + ('still descending, so this ranks convergence speed'
           if ratio > 1.5 else 'getting worse, so the target is not stationary'))


def test_all_three_branches_actually_pull():
    """
    `cond_bwd` defaulted to 1.0 and no cell ever set it, so `S_bwd` was all ones
    everywhere and the bwd branch supplied 0.3% of the policy gradient while
    pointing the OPPOSITE way to the docstring's claim. A 30-line design comment
    described a configuration nothing instantiated.

    Measured at a SETTLED state, not at init: `mu` starts equal to `theta`, so
    the bwd gradient is exactly zero there and any alignment claim evaluated at
    init is vacuous.
    """
    import torch
    _, g, _ = _end(1e-2, 0)
    n_theta, _ = g.draw(64)
    rep = torch.autograd.grad(g.w_rep * g._replay_loss(n_theta), [g.theta],
                              retain_graph=True)[0]
    bwd = torch.autograd.grad(g.w_bwd * g._bwd_loss(), [g.theta])[0]
    share = float(bwd.norm()) / (float(rep.norm()) + float(bwd.norm()))
    assert share > 0.10, (
        f'the bwd branch supplies {share:.1%} of the policy gradient -- it is a '
        f'rounding error, not one of three competing optimisations')


def test_the_drift_path_is_common_across_seeds():
    """
    The moving target must be COMMON random numbers, exactly like the gradient
    noise. A per-seed target path is a second macroscopic noise source: measured,
    it drove seed noise from 0.054 to 0.413 nats and swamped the sharper signal
    the drift was added to buy.
    """
    import torch
    a = EquilibrationGame(lr=1e-2, optimizer='sgd', seed=0, **FIT)
    b = EquilibrationGame(lr=1e-2, optimizer='sgd', seed=7, **FIT)
    for _ in range(50):
        a.advance(); b.advance()
    assert torch.equal(a.c, b.c), (
        'two seeds see different target paths, so a difference between arms run '
        'at different seeds is partly the target and not the arm')
    assert float(a.c.norm()) > 0, 'the drift never moved; the cell is static'


def test_the_closed_form_cliff_survives_the_drift():
    """
    The drift is an additive forcing, so it must not move the Jacobian -- the
    stability boundary is the one thing on this surface that is exact, and it is
    what `lr/cliff` is read against.
    """
    static = EquilibrationGame(lr=1e-2, optimizer='sgd', seed=0,
                               **{**FIT, 'drift': 0.0})
    moving = EquilibrationGame(lr=1e-2, optimizer='sgd', seed=0, **FIT)
    assert (moving.stability_lr(lr_level=0.1)
            == pytest.approx(static.stability_lr(lr_level=0.1), rel=1e-9)), (
        'the drift moved the stability boundary; it is supposed to be additive '
        'forcing, so either it is entering the dynamics or the closed form is '
        'no longer describing the game')
