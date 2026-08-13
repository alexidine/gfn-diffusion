"""
Tests on the SURFACES, not the controllers.

A synthetic problem is only worth anything if its planted properties are actually
planted. These assert the ground truth the controller tests are scored against --
if the equilibration game's stability boundary is not where stability_lr() says
it is, every conclusion drawn from it is void.
"""

import math

import numpy as np
import pytest
import torch

from bench.surfaces import EquilibrationGame, MLEGame, VarCondGame


# ------------------------------------------------------------------------ mle

def test_mle_optimum_is_the_origin():
    g = MLEGame(dim=16, cond=10.0, noise=0.0, lr=1e-2, init_scale=2.0, seed=0)
    assert g.distance_to_opt() > 1.0
    for _ in range(3000):
        g.train_step(g.draw(1024))
    assert g.distance_to_opt() < 1e-3


def test_mle_gradient_noise_scales_as_one_over_sqrt_batch():
    """Minibatch scaling has to be real, or batch size cannot interact with LR."""
    g = MLEGame(dim=64, noise=1.0, seed=0)
    for b, expect in [(16, 1 / 4), (256, 1 / 16), (4096, 1 / 64)]:
        draws = torch.stack([g.draw(b) for _ in range(400)])
        assert float(draws.std()) == pytest.approx(expect, rel=0.15)


def test_mle_curvature_falls_along_the_path_under_a_quartic():
    """
    The quartic term is what makes alpha* a MOVING quantity: local curvature is
    H + 12c*diag(theta^2), so it falls as the run converges and alpha* rises.
    Without this the surface is a bowl and periodic recalibration measures
    nothing.
    """
    g = MLEGame(dim=16, cond=4.0, quartic=0.05, noise=0.0, lr=5e-3,
                init_scale=3.0, seed=0)
    start = float(g.hessian_diag().mean())
    for _ in range(4000):
        g.train_step(g.draw(1024))
    end = float(g.hessian_diag().mean())
    assert end < 0.5 * start, f'curvature did not fall along the path: {start} -> {end}'


def test_mle_alpha_star_matches_a_brute_force_line_search():
    """alpha_star_true is the reference every sensor test is scored against, so
    it has to be right. Check it against a direct scan of the population loss."""
    g = MLEGame(dim=12, cond=6.0, quartic=0.0, noise=0.0, lr=2e-2, seed=1)
    before = g.theta.detach().clone()
    g.train_step(g.draw(4096))
    d = g.theta.detach() - before

    analytic = g.alpha_star_true(before, d)
    alphas = np.linspace(0.05, 4 * analytic, 4000)
    losses = [float(g._loss(torch.zeros(g.dim), theta=before + float(a) * d)) for a in alphas]
    brute = float(alphas[int(np.argmin(losses))])
    assert analytic == pytest.approx(brute, rel=2e-3)


# ------------------------------------------------------------------- var_cond

def test_var_cond_reaches_its_joint_optimum():
    g = VarCondGame(dim=8, n_cond=64, spread=20.0, noise=0.0, lr=5e-2,
                    lr_level=5e-2, seed=0)
    start = g.distance_to_opt()
    for _ in range(20000):
        g.train_step(g.draw(64))
    assert g.distance_to_opt() < 0.25 * start


def test_var_cond_batch_size_buys_condition_coverage_not_just_noise():
    """
    THE POINT OF THIS GAME. In the conditional route batch size is not primarily
    a noise knob -- it is condition coverage. A per-condition level that is only
    touched once every k steps is stale for k-1 of them, and theta is fitting
    against a target that has not caught up.

    Staleness must fall roughly as n_cond/batch.
    """
    stale = {}
    for b in [8, 32, 128]:
        g = VarCondGame(dim=8, n_cond=256, noise=0.0, lr=1e-2, seed=0)
        for _ in range(2000):
            g.train_step(g.draw(b))
        stale[b] = g.staleness()

    # A condition is drawn with probability b/N per step, so sightings are ~N/b
    # apart and the mean AGE scales the same way. The constant is below 1 because
    # the draw is WITHOUT replacement within a step, which suppresses the gap
    # variance -- most visibly at b=128, where every condition is seen almost
    # every other step (measured 1.03 against a with-replacement 2.0).
    assert stale[8] > stale[32] > stale[128]
    for b in (8, 32, 128):
        assert 0.4 <= stale[b] * b / 256 <= 1.0, (b, stale)
    assert stale[8] / stale[32] == pytest.approx(4.0, rel=0.25)


# -------------------------------------------------------------- equilibration

def test_equilibration_fixed_point_is_attracting_below_the_boundary():
    g = EquilibrationGame(dim=4, a=4.0, w_rep=0.7, w_bwd=0.3, kappa=0.05,
                          noise=0.0, init_scale=1.0, seed=0)
    lr = 0.5 * g.stability_lr()
    for grp in g.optimizers['fused'].param_groups:
        grp['lr'] = lr
    start = g.distance_to_opt()
    for _ in range(4000):
        g.train_step(g.draw(1024))
    assert g.distance_to_opt() < 0.1 * start


def test_equilibration_diverges_above_the_boundary():
    g = EquilibrationGame(dim=4, a=4.0, w_rep=0.7, w_bwd=0.3, kappa=0.05,
                          noise=0.0, init_scale=1.0, seed=0)
    lr = 1.15 * g.stability_lr()
    for grp in g.optimizers['fused'].param_groups:
        grp['lr'] = lr
    for _ in range(4000):
        g.train_step(g.draw(1024))
        if g.diverged():
            return
    pytest.fail('stability_lr did not bound the actual dynamics')


def test_stability_boundary_matches_the_closed_form():
    """
    Ignoring the (slow, weakly coupled) buffer pole, the policy/level pair has
    eigenvalues -c +/- i*sqrt(w_rep*b*a) with c = w_rep*b^2 + w_bwd, so

        |1 + lr*lambda|^2 = 1  ->  lr_crit = 2c / (c^2 + w_rep*b*a)

    Verified against the numerical spectral radius, which is what the tests use.
    """
    for a in [1.0, 2.0, 4.0, 8.0]:
        for w_rep in [0.3, 0.5, 0.7]:
            g = EquilibrationGame(a=a, b=1.0, w_rep=w_rep, w_bwd=1.0 - w_rep,
                                  kappa=1e-6)
            c = w_rep + (1.0 - w_rep)
            closed = 2 * c / (c ** 2 + w_rep * a)
            assert g.stability_lr() == pytest.approx(closed, rel=0.02), (a, w_rep)


def test_required_alpha_target_is_half_the_loop_gain_plus_a_half():
    """
    THE RESULT THIS GAME EXISTS FOR.

    A ray probe at frozen zeta and mu sees only the theta self-curvature c and
    reports alpha* = 1 at lr = 1/c. The rate the LOOP survives is
    2c/(c^2 + w_rep*b*a). Their ratio is the smallest alpha_target that keeps the
    servo inside the stability boundary:

        alpha_target_min = one_step_lr / stability_lr = (c^2 + a*b*w_rep) / (2c^2)

    which at c = 1 is exactly (1 + loop_gain)/2, with loop_gain = a*b*w_rep.

    Two consequences, neither visible to the sensor:
      * the required margin is a property of the POLICY/LEVEL COUPLING, not of
        anything measurable along a ray. alpha_target is therefore per-route by
        construction, exactly as the config comment says -- this is why.
      * loop gain rises with the replay weight, so a stage that shifts weight
        onto the replay branch RAISES the minimum safe alpha_target while the
        controller holds it fixed.
    """
    for a in [1.0, 2.0, 4.0, 8.0]:
        for w_rep in [0.3, 0.5, 0.7]:
            g = EquilibrationGame(a=a, b=1.0, w_rep=w_rep, w_bwd=1.0 - w_rep,
                                  kappa=1e-6)
            required = g.one_step_lr() / g.stability_lr()
            assert required == pytest.approx((1.0 + a * w_rep) / 2.0, rel=0.02)


def test_symmetric_coupling_would_be_unstable_at_every_lr():
    """
    Guards the sign choice. With a +/+ coupling -- the naive reading of "the
    level chases the policy" -- the 2x2 determinant is c - a*w_rep*b < 0 for any
    interesting a, the fixed point is a saddle, and no LR is stable. The bench
    would then report "diverged" for every controller setting and look like a
    controller bug.
    """
    g = EquilibrationGame(a=4.0, w_rep=0.7, w_bwd=0.3, kappa=0.05)
    M = g.iteration_matrix(0.1)
    M_sym = M.copy()
    M_sym[1, 0] = -M_sym[1, 0]                      # flip back to +/+
    assert np.max(np.abs(np.linalg.eigvals(M_sym))) > 1.0
    assert np.max(np.abs(np.linalg.eigvals(M))) < 1.0


def test_fast_buffer_churn_destroys_the_restoring_force():
    """
    A BUFFER THAT TRACKS THE POLICY IS NOT AN ANCHOR.

    The bwd branch pulls theta toward mu, so its restoring force is proportional
    to (theta - mu). As kappa -> 1 the buffer becomes a mirror of the current
    policy, that difference goes to zero, and the only damping on the
    policy/level spiral disappears. As kappa -> 0 the buffer is a stiff anchor to
    the past: heavily damped, but converging on the buffer's slow timescale
    rather than the policy's.

    So churn rate trades ANCHORING against CURRENCY, and the fast end is the
    dangerous one -- which is the mechanical version of the architecture's claim
    that the off-policy branch is what prevents collapse. (My first guess was the
    opposite: that a slow buffer would ring because its correction arrives late.
    It does not -- a late correction toward a nearly-fixed point is still a
    correction toward a nearly-fixed point.)
    """
    def crossings(kappa):
        g = EquilibrationGame(dim=1, a=4.0, w_rep=0.7, w_bwd=0.3, kappa=kappa,
                              noise=0.0, init_scale=1.0, seed=0)
        lr = 0.6 * g.stability_lr()
        for grp in g.optimizers['fused'].param_groups:
            grp['lr'] = lr
        signs = []
        for _ in range(3000):
            g.train_step(g.draw(1024))
            signs.append(math.copysign(1, float(g.theta.detach()[0])))
        return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])

    assert crossings(0.5) > 10 * crossings(0.002), 'fast churn should ring more'


def test_buffer_churn_barely_moves_the_stability_boundary():
    """
    A CLEAN NEGATIVE, and it is what makes the required-alpha_target result
    usable. Across the entire churn range -- from a buffer that barely updates to
    one that is a mirror of the policy -- the stability boundary moves by under
    10%:

        kappa 0.002 -> 0.5265      kappa 0.10 -> 0.5343
        kappa 0.020 -> 0.5279      kappa 0.50 -> 0.5669
                                   kappa 0.95 -> 0.5760

    So the LR ceiling is set by the POLICY/LEVEL LOOP GAIN, not by the buffer.
    Churn rate governs how the run oscillates and how fast it converges (see the
    test above) but not what rate it survives -- which means the loop-gain
    formula does not need a churn correction.
    """
    boundaries = {}
    for kappa in (0.002, 0.02, 0.1, 0.5, 0.95):
        g = EquilibrationGame(a=4.0, w_rep=0.7, w_bwd=0.3, kappa=kappa)
        boundaries[kappa] = g.stability_lr()

    lo, hi = min(boundaries.values()), max(boundaries.values())
    assert hi / lo < 1.15, boundaries
    # ...whereas the loop gain moves it by more than 3x over its usable range
    span = [EquilibrationGame(a=4.0, w_rep=w, w_bwd=1.0 - w, kappa=0.05).stability_lr()
            for w in (0.1, 0.9)]
    assert span[0] / span[1] > 3.0, span
