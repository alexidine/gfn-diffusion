"""
Does the equilibration cell pose a control problem at all, and is the ground
truth real?

Written BEFORE ranking any arm on it, because the first look showed every fixed
rate from 0.05 to 1.5 settling to a noise floor with only the floor level
differing -- i.e. a surface where the learning rate barely matters and doing
nothing is competitive. A leaderboard on a surface like that produces an ordering
with no meaning.

Two things must hold for the cell to be worth running:
  1. the closed-form `stability_lr` must actually predict divergence
  2. the recorded rate must be the rate the game TRAINS with
"""
import math

import pytest

from bench.arms import Fixed
from bench.runner import Run
from bench.surfaces import EquilibrationGame

#: THE CONFIGURED CELL. `cond_rep` is what makes the branches genuinely compete:
#: with scalar curvatures the theta-problem is a single isotropic attractor and
#: the rate barely matters (measured: 13x over a 50x sweep, `null` competitive).
KW = dict(dim=8, a=2.0, b=1.0, w_rep=0.7, w_bwd=0.3, kappa=0.02, noise=0.1,
          init_scale=1.0, cond_rep=100.0)


def _pinned_boundary(g, run):
    """The applicable boundary: the Z head is PINNED at `lr_flow` and exempt from
    the servo, so the level's rate does not scale with the policy's. The
    both-scale default answers a different question and reads 2.15 where the real
    one reads 0.03."""
    pin = g.optimizers['fused'].param_groups[-1]['lr']
    return g.stability_lr(lr_level=pin)


def _game(lr, **over):
    return EquilibrationGame(lr=lr, optimizer='sgd', seed=0, **{**KW, **over})


def _run(lr, steps=1500, **over):
    g = _game(lr, **over)
    r = Run(g, Fixed(lr), seed=0, steps=steps, batch=64)
    r.run()
    return g, r


def test_the_recorded_rate_is_the_rate_the_game_trains_with():
    """
    `EquilibrationGame.train_key` is 'fused', not 'fwd'. The runner used to
    record `lr_of('fwd')` unconditionally -- a SPECTATOR optimizer here, since
    the five-key dict is always built but this game steps only 'fused'. Every
    learning-rate number on this surface would have been about an optimizer
    nothing trains with.
    """
    g, r = _run(0.3, steps=40)
    trained = g.optimizers[g.train_key].param_groups[0]['lr']
    assert r.trace[-1]['lr'] == pytest.approx(trained, rel=1e-9)


def test_the_stability_boundary_actually_predicts_divergence():
    """
    The cell's whole value is that `stability_lr` is exact. If a rate well above
    it still settles, the number is not describing this configuration and must
    not be drawn on a plot as ground truth.
    """
    g0 = _game(0.05)
    r0 = Run(g0, Fixed(0.05), seed=0, steps=30, batch=64)
    r0.run()
    bound = _pinned_boundary(g0, r0)
    assert 0 < bound < 10

    _, below = _run(bound * 0.5)
    gb, above = _run(bound * 2.0)

    d_below = below.trace[-1]['loss']
    d_above = above.trace[-1]['loss']
    assert d_below is not None and math.isfinite(d_below)
    assert (d_above is None or not math.isfinite(d_above)
            or d_above > 1e3 * abs(d_below)), (
        f'at 2x the stability boundary ({bound:.3g}) the system settled to '
        f'{d_above:.3g} against {d_below:.3g} below it -- `stability_lr` does '
        f'not describe this configuration, so it is not ground truth here')


def test_this_cell_is_a_CLIFF_problem_not_a_tuning_problem():
    """
    THE CELL-IS-WORTH-RUNNING TEST, and it is a different test than the MLE
    family needs.

    Measured: inside the stable band the settled distance moves only ~3.3x across
    a 10x rate sweep, while crossing the boundary is fatal. So the reward
    structure here is not "find the rate that minimises the loss" -- it is "get
    as close to the cliff as you can without going over", which is a genuinely
    different control problem and arguably the real one.

    The property that makes the cell worth running is therefore the SHARPNESS of
    the cliff, not the gradient of the outcome within the band: a small change in
    rate has to flip survival, or there is no edge to sit near.
    """
    g0 = _game(0.05)
    r0 = Run(g0, Fixed(0.05), seed=0, steps=30, batch=64)
    r0.run()
    b = _pinned_boundary(g0, r0)

    g_in, _ = _run(b * 0.7)
    g_out, _ = _run(b * 1.5)
    inside, outside = g_in.expected_loss(), g_out.expected_loss()
    assert math.isfinite(inside) and inside < 1.0, (
        f'0.7x the boundary already fails ({inside:.3g}); no usable band')
    assert (not math.isfinite(outside)) or outside > 1e3 * inside, (
        f'1.5x the boundary settles at {outside:.3g} against {inside:.3g} '
        f'inside -- the cliff is not sharp, so there is no edge to control to')


def test_noise_is_a_pure_gain_on_the_linear_game():
    """
    THE `noisy` CELL IS NOT A CELL, and the reason is STRUCTURAL rather than a
    metric defect -- which is a stronger statement than the first version of
    this test made, and it changes the fix.

    `bench.eqsuite`'s `noisy` cell returned a nats column identical to `base` to
    two decimals for eight arms. The game is linear and its fixed point is the
    origin, so the state is `deterministic(t) + noise * stochastic(t)`; once the
    transient decays the WHOLE trajectory is proportional to `noise`. Distance
    is quadratic in it -- hence exactly 100x for 10x noise, measured, for arms
    at different rates -- and the gradients scale too, so every scale-free
    controller is exactly blind to it. It is not that nats hides a real
    difference; there is no difference to hide.

    Consequence: no rescoring rescues the linear cell. Only breaking the scale
    invariance does, which is what `quartic` is for -- see the next test.
    """
    lo = [_run(lr, steps=3000)[0].expected_loss() for lr in (0.01, 0.003)]
    hi = [_run(lr, steps=3000, noise=1.0)[0].expected_loss()
          for lr in (0.01, 0.003)]
    ratios = [h / l for h, l in zip(hi, lo)]

    assert all(r > 10.0 for r in ratios), (
        f'10x noise moved the settled distance by {ratios} -- the knob is INERT '
        f'and the game, not the metric, is what needs fixing')
    assert abs(ratios[0] / ratios[1] - 1.0) < 0.05, (
        f'noise rescaled the two rates differently ({ratios}) -- then the game '
        f'is NOT scale-invariant and the cell carries information after all')


def test_the_quartic_is_live_and_STILL_cannot_break_the_noise_invariance():
    """
    THE ATTEMPTED REPAIR AND ITS FAILURE, recorded as a test so the idea is not
    tried again from scratch.

    The reasoning was sound: a quartic makes curvature grow with |theta|, a
    bigger noise ball sits in a stiffer region, and noise should stop being a
    common rescale. It does not work at any usable coefficient. theta starts at
    O(1) and settles at O(1e-3), so the quartic term spans six orders of
    magnitude between the two ends. Measured at the widest ball the game
    tolerates (noise=1.0): the settled distance moves 2.7e-7 at quartic=0.05,
    6.3e-5 at quartic=10, and NaN at quartic=50 -- it goes from unmeasurable to
    fatal without passing through useful. The `noisy` cell is DROPPED from
    `bench.eqsuite`.

    TWO ASSERTIONS, and the pair is the point. The quartic must be LIVE -- it
    changes the settled distance -- and must still FAIL to separate the rates.
    Asserting only the failure would pass just as well if the term were never
    added at all, which is the state this test would then be certifying.

    (An earlier version asserted the repair WORKED and was failing the whole
    time. It survived a mutation sweep because that sweep only checked each test
    failed with its bug reinstated and never checked it passed clean -- so
    "caught" meant nothing for a test that was red in both states.)
    """
    # the LARGEST coefficient that does not detonate, and the widest ball -- if
    # the term cannot matter here it cannot matter anywhere on this surface
    q = dict(quartic=10.0)
    # ON `distance_to_opt`, NOT `expected_loss`. The score ADDS the quartic term
    # itself, so it moves when the coefficient changes even if the term never
    # reaches an update -- measured: deleting the quartic from `_replay_loss`
    # left an expected_loss-based version of this assertion passing. The state
    # norm carries no quartic, so only the DYNAMICS can move it.
    plain = _run(0.01, steps=3000, noise=1.0)[0].distance_to_opt()
    quart = _run(0.01, steps=3000, noise=1.0, **q)[0].distance_to_opt()
    assert abs(quart / plain - 1.0) > 1e-6, (
        f'the quartic changed nothing at all ({quart:.10g} vs {plain:.10g}) -- '
        f'it is not reaching the dynamics, so the rest of this test would be '
        f'certifying an absent term rather than an insufficient one')

    lo = [_run(lr, steps=3000, **q)[0].expected_loss() for lr in (0.01, 0.003)]
    hi = [_run(lr, steps=3000, noise=1.0, **q)[0].expected_loss()
          for lr in (0.01, 0.003)]
    ratios = [h / l for h, l in zip(hi, lo)]
    assert abs(ratios[0] / ratios[1] - 1.0) < 0.05, (
        f'the quartic now DOES separate the rates under noise ({ratios}) -- the '
        f'`noisy` cell is repairable after all and eqsuite should carry it')


def test_the_buffer_rate_is_inert_at_the_battery_horizon():
    """
    THE `slow buffer` CELL IS NOT A CELL EITHER, for an unrelated reason: at the
    fixed point mu reaches theta whatever kappa is, so kappa shapes only the
    transient -- and over 6000 steps that has washed out. Measured 0.3%, roughly
    100x under the gaps between arms.

    The fix is therefore a SHORTER HORIZON, not a bigger knob, and this test
    pins both ends of that claim rather than only the null: kappa is worth ~13%
    at 3000 steps (comparable to the ray-vs-hyper gap, so scoreable) and ~0.3%
    at the battery's 6000 (not). A null on its own would be satisfied by a
    kappa that never did anything, which is a different and worse story.
    """
    def dist(steps, **over):
        return _run(0.01, steps=steps, **over)[0].expected_loss()

    early = dist(3000, kappa=0.002) / dist(3000)
    late = dist(6000, kappa=0.002) / dist(6000)

    assert abs(early - 1.0) > 0.05, (
        f'kappa moves nothing even at 3000 steps ({early:.4f}x) -- the buffer '
        f'pole is not reaching the dynamics at all, which is a game bug and not '
        f'a horizon problem')
    assert abs(late - 1.0) < 0.05, (
        f'kappa now moves the settled distance {late:.3f}x at the battery '
        f'horizon -- `slow buffer` is a live cell again and eqsuite should '
        f'score it')
    assert abs(late - 1.0) < abs(early - 1.0), (
        'the kappa effect is supposed to DECAY with horizon (transient-only); '
        f'it grew instead ({early:.4f} -> {late:.4f})')


def test_the_closed_form_tracks_the_conditioning():
    """
    `iteration_matrix` used to assume every coordinate identical and return one
    3x3. With opposed spectra that is false, and it kept reporting a cliff of
    2.15 while the real one had moved to ~0.03 -- ground truth silently no longer
    describing the game. It is now a max over per-coordinate blocks.
    """
    plain = EquilibrationGame(lr=0.05, optimizer='sgd', seed=0,
                              **{**KW, 'cond_rep': 1.0})
    hard = EquilibrationGame(lr=0.05, optimizer='sgd', seed=0, **KW)
    assert hard.stability_lr(lr_level=0.1) < 0.2 * plain.stability_lr(lr_level=0.1)
