"""
!! MEASURED UNFIT TO RANK CONTROLLERS, 2026-08-14. DO NOT WIRE INTO A BOARD. !!

It runs cleanly on every arm -- that is not the problem. Audited with the same
five checks `bench/audit.py` applies, against `TrackingGame` on an identical
ladder: seed noise 1.683 nats against a 4.30-nat ladder (range/noise 2.6, the bar
is 10) where tracking is 0.017 and 286. Its within-2x floor is 1000x wide against
tracking's 10x. Under ADAM -- production's optimizer -- every rate from 3.2e-4 to
0.32 lands within 0.13 nats, and an arm board ties `fixed@0.3`, `fixed@0.1` and
`fixed@0.01` at 0.00 nats with a per-arm seed sd of 1.3-1.7.

Three causes, each measured, and each an instructive trap:

  * THE FLAT DIRECTION IS NOT FLAT (curvature 12-24 in a training Hessian whose
    smallest eigenvalue is 3.39) and it supplies 90-98% of the score as a
    PER-SEED CONSTANT that moves 3% over a 320x rate range. `score_floor` is left
    at 0.0, so that constant is never subtracted and it swamps the signal.
  * THE THIRD PLAYER CANNOT PULL. Both autograd cross-derivatives between the
    flow player and the rest come back None -- it is structurally decoupled.
  * `lr_flow = lr * 800`, the file's headline design choice, IS OVERWRITTEN every
    10 steps by `controller.py`'s flow pinning, so |zeta| is bit-identical at
    every policy rate.

Its `grad_clip` default is also now wrong: production ships a per-branch p=0.99
quantile guard (`grad_clip_guard.py`), and under SGD an always-binding clip keeps
alive rates that die 4/4 without it -- it REMOVES the cliff. The quantile guard
reproduces the no-clip result to four digits.

WHAT IT IS STILL GOOD FOR: the fidelity analysis below. Its four corrections to
`EquilibrationGame` all still hold against current code. Recommendation is to
harvest that into a design note and delete the class; it is kept for now only so
that analysis is not lost with it.

--------------------------------------------------------------------------------

THE FUSED EQUILIBRATION STAGE, rebuilt against the real one.

`EquilibrationGame` is kept as the CONTROL. A fidelity critique of it against
`configs/mk_dev.yaml:335-382` found it modelling the wrong mechanism in four
separate ways, each measured rather than argued. This is the corrected structure;
the point of keeping both is that every difference in a controller's behaviour
between them is attributable to a named mismatch.

WHAT THE OLD SURFACE GOT WRONG, AND WHAT THIS DOES INSTEAD

1. THE ANTI-PHASE SPIRAL DOES NOT EXIST. The old game's reason to exist was a
   policy/level antisymmetric coupling giving oscillation and a finite LR cliff.
   In the real stage `fwd` carries `freeze_policy: 1.0` (mk_dev.yaml:349) and
   `bwd`/`replay` carry `freeze_z: 1.0` (:209, :230), so policy and Z are
   PARAMETER-DISJOINT -- measured Jacobian off-diagonals exactly 0.0, and TB
   (gflownet_losses.py:805) enters both with the SAME sign. Here Z is genuinely
   decoupled, and the anti-phase that does exist comes from the place the real
   one comes from: NORMALISATION ACROSS TWO SUPPORTS. Raising log_pf on the
   buffer's support forces mass off it, so it falls on the on-policy support
   (gflownet_losses.py:840-849). That is modelled as a soft mass constraint, so
   the coupling gain is a consequence rather than an invented `a = 2`.

2. THE BUFFER ROLES WERE SWAPPED. Real REPLAY holds the policy's own past
   trajectories -- uniform intake, memoryless purge, residence tau = 50 steps,
   i.e. kappa 0.02, which is exactly the number the old game applied to the
   wrong player. Real BWD draws from a frozen-snapshot prior buffer with ~25,000
   step residence (mk_dev.yaml:400-409), churned only at eval cadence: an
   EXOGENOUS ANCHOR containing nothing the current policy produced. So here the
   fast self-referential branch is replay and the slow external one is bwd,
   which is the opposite of the old assignment.

3. THE TARGET RATCHETS, IT DOES NOT DIFFUSE. The old surface used an OU random
   walk, added to create ranking signal. The real non-stationarity is a
   STAIRCASE: the prior buffer churns only every eval period, gated on
   `energy < Emin(c) + ramp_floor` against a RATCHETING `Emin(c)`
   (train.py:4872-4873) -- monotone, one-way, capped per jump by
   `expire_max_frac`. A walk penalises a cold rate as drift/lr forever; a ratchet
   penalises it only until it catches up. The walk therefore biases the whole
   bench toward hotter arms.

4. THE MARGINAL DIRECTION IS THE REAL PATHOLOGY, and it is not an instability.
   Without the freezes the policy/Z pair is rank-1 PSD with a UNIT eigenvalue:
   `log_pf - log_Z` drifts freely. Training cannot see it -- the loss is exactly
   flat along it -- while the sampler it produces is wrong. Modelled here as a
   direction with zero curvature in every branch that the SCORE still penalises.
   It gives a genuine LR cost with no compensating benefit, invisible to any
   controller reading the training loss, which is a different and more realistic
   hazard than an oscillation.

ALSO, AND THEY MATTER MORE THAN ANY CELL:
  * ADAM BY DEFAULT. Production runs Adam on every optimizer (train.py:1647).
    Measured divergence-free ceiling on the old base cell: SGD 0.0294 (closed
    form exact), Adam >= 100 -- a factor of >= 3411. An SGD-only verdict is a
    recommendation for a configuration nobody ships.
  * A BINDING GRADIENT CLIP. `gradient_norm_clip: auto` resolves to 37.88
    against a median pre-clip grad norm of ~1.0e3, so it binds on essentially
    every step and train.py:2868-2870 says the consequence: "Adam is effectively
    running on normalized gradients". Measured on the bench, a binding clip moves
    the empirical stability boundary from 0.0319 to >= 3 -- it REMOVES the cliff.
    Every "sit at 0.6x the cliff" verdict is about a regime production never
    enters.

WHAT IS DELIBERATELY NOT MODELLED, so it is not mistaken for covered: the Huber
knee (curvature FALLS as residuals grow, opposite of a quartic), the prioritised
residual-dependent replay draw, and the three other live feedback loops (balance
servo, freshness servo, z_calibration_tick) that move this surface every 10 steps.
"""
import math

import numpy as np
import torch

from bench.surfaces import _Game, _full_optimizer_set


class FusedStageGame(_Game):
    """
    theta (policy) is trained by TWO off-policy branches with DIFFERENT SUPPORTS;
    zeta (Z) is trained on its own, disjoint, and fast.

        replay (fast, self-referential):  1/2 |theta - mu|^2 on S_rep
                                          mu <- (1-kappa) mu + kappa theta
        bwd    (slow, exogenous anchor):  1/2 |theta_A - anchor|^2 on S_bwd,
                                          over the ANCHOR BLOCK only
        flow   (decoupled):               1/2 |zeta - z_target|^2
        mass   (the real anti-phase):     w_mass/2 * (sum(theta_A) + sum(theta_B))^2

    `A` is the anchor block -- the support bwd constrains. `B` is the on-policy
    block only replay sees. The mass term couples them with the SAME sign, so
    pushing A up pushes B down: that is normalisation, and it is where a genuine
    anti-phase comes from once the invented one is removed.
    """

    name = 'fused_stage'
    train_key = 'fused'

    def __init__(self, dim=8, anchor_frac=0.5, w_rep=0.5, w_bwd=0.3,
                 w_mass=0.2, kappa=0.02, noise=0.1, lr=1e-3, optimizer='adam',
                 init_scale=1.0, seed=0, device='cpu', cond_rep=100.0,
                 cond_bwd=100.0, grad_clip=1.0, ratchet_period=250,
                 ratchet_step=0.05, flat_dim=1, flat_weight=1.0,
                 lr_flow=None, probe_scores='replay'):
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.device, self.dim = device, int(dim)
        self.w_rep, self.w_bwd, self.w_mass = float(w_rep), float(w_bwd), float(w_mass)
        self.kappa, self.noise = float(kappa), float(noise)
        self.probe_scores = probe_scores
        self._gen = g
        self._step = 0

        #: THE SUPPORT SPLIT, which is the structural claim MK made: "the
        #: backward training only samples buffer states" while the forward
        #: policy also holds whatever else it wants. bwd's support is a strict
        #: SUBSET, permanently -- so the coordinates outside it are constrained
        #: by replay alone and are softer in the combined problem. The old
        #: surface modelled this as opposed spectra, which is a different thing
        #: and was inert anyway because `cond_bwd` was never set.
        self.n_anchor = max(1, int(round(anchor_frac * self.dim)))

        self.S_rep = torch.logspace(0, math.log10(max(cond_rep, 1.0)),
                                    self.dim, dtype=torch.float64).float()
        self.S_bwd = torch.logspace(math.log10(max(cond_bwd, 1.0)), 0,
                                    self.n_anchor, dtype=torch.float64).float()

        #: THE FLAT DIRECTION: zero curvature in every training branch, and
        #: scored anyway. `log_pf - log_Z` drifts freely under TB; training is
        #: blind to it and the resulting sampler is still wrong. A hotter rate
        #: random-walks along it faster, so it is an LR cost with NO offsetting
        #: benefit and no controller reading the training loss can see it.
        self.flat_dim = int(flat_dim)
        self.flat_weight = float(flat_weight)
        if self.flat_dim > 0:
            f = torch.randn(self.dim, self.flat_dim, generator=g)
            self.flat, _ = torch.linalg.qr(f)          # orthonormal basis
        else:
            self.flat = None

        t0 = torch.randn(self.dim, generator=g) * init_scale
        self.theta = torch.nn.Parameter(t0.clone())
        self.zeta = torch.nn.Parameter(torch.zeros(1))
        self.mu = t0.clone()

        #: THE ANCHOR AND ITS RATCHET. Exogenous, static between refreshes, and
        #: when it moves it moves ONE WAY: `Emin(c)` only ever tightens, and
        #: `expire_max_frac` caps how much of the buffer can turn over at once.
        self.anchor = torch.randn(self.n_anchor, generator=g) * init_scale
        self.ratchet_period = int(ratchet_period)
        self.ratchet_step = float(ratchet_step)
        self._ratchets = 0

        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.clip_hits = self.clip_steps = 0

        #: 800x IN PRODUCTION (policy 1.25e-4, flow 0.1) and structurally
        #: unservoable -- `_LR_KEYS` omits lr_flow. The old surface built both at
        #: the same rate, which made its level a fast equal-weight player when
        #: the real one is a single scalar that equilibrates essentially
        #: instantly. Default here keeps the real ratio.
        flow_lr = float(lr * 800.0 if lr_flow is None else lr_flow)
        self.optimizers = _full_optimizer_set(
            optimizer, [self.theta], [self.zeta],
            {'policy': lr, 'flow': flow_lr})
        self.policy_params = [self.theta]

    # ---------------------------------------------------------------- data

    def draw(self, batch_size):
        b = max(1, int(batch_size))
        s = self.noise / math.sqrt(b)
        return (torch.randn(self.dim, generator=self._gen) * s,
                torch.randn(1, generator=self._gen) * s)

    def advance(self):
        """The anchor ratchets: discrete, periodic, MONOTONE."""
        self._step += 1
        if self.ratchet_period and self._step % self.ratchet_period == 0:
            self._ratchets += 1
            # one-way: the anchor only ever contracts toward the origin, the way
            # a ratcheting energy floor only ever tightens. A cold rate pays for
            # each jump until it catches up, and then stops paying -- unlike a
            # random walk, which charges it forever.
            self.anchor = self.anchor * (1.0 - self.ratchet_step)

    # ---------------------------------------------------------- objectives

    def _replay_loss(self, n_theta):
        d = self.theta - self.mu
        return 0.5 * (self.S_rep * d * d).sum() + (n_theta * self.theta).sum()

    def _bwd_loss(self):
        d = self.theta[:self.n_anchor] - self.anchor
        return 0.5 * (self.S_bwd * d * d).sum()

    def _mass_loss(self):
        """Normalisation across the two supports -- the REAL anti-phase.

        Same sign on both blocks, so raising one forces the other down. No
        invented gain: the coupling strength is `w_mass`, a loss weight, and the
        structure is a rank-1 PSD term exactly like the real policy/Z block."""
        return 0.5 * (self.theta.sum() ** 2)

    def _flow_loss(self, n_zeta):
        return 0.5 * (self.zeta ** 2).sum() + (n_zeta * self.zeta).sum()

    def train_step(self, batch):
        n_theta, n_zeta = batch
        opt = self.optimizers['fused']
        opt.zero_grad(set_to_none=True)

        theta_loss = (self.w_rep * self._replay_loss(n_theta)
                      + self.w_bwd * self._bwd_loss()
                      + self.w_mass * self._mass_loss())
        flow_loss = self._flow_loss(n_zeta)

        (g_theta,) = torch.autograd.grad(theta_loss, [self.theta],
                                         retain_graph=True)
        (g_zeta,) = torch.autograd.grad(flow_loss, [self.zeta])
        self.theta.grad, self.zeta.grad = g_theta, g_zeta

        if self.grad_clip is not None:
            self.clip_steps += 1
            n = float(self.theta.grad.norm())
            if n > self.grad_clip:
                self.clip_hits += 1
                self.theta.grad.mul_(self.grad_clip / n)

        theta_before = self.theta.detach().clone()
        opt.step()
        self.mu = (1.0 - self.kappa) * self.mu + self.kappa * theta_before
        return float(theta_loss.detach())

    # -------------------------------------------------------------- scoring

    def expected_loss(self):
        """
        Noise-free residuals PLUS the flat direction the training loss cannot
        see. The second part is the point: a controller can drive the training
        objective to its floor and still be losing here, which is what a free
        `log_pf - log_Z` drift does to a sampler.
        """
        with torch.no_grad():
            d = self.theta - self.mu
            rep = 0.5 * (self.S_rep * d * d).sum()
            db = self.theta[:self.n_anchor] - self.anchor
            bwd = 0.5 * (self.S_bwd * db * db).sum()
            mass = 0.5 * self.theta.sum() ** 2
            flat = 0.0
            if self.flat is not None:
                proj = self.flat.T @ self.theta
                flat = self.flat_weight * 0.5 * float((proj * proj).sum())
            return float(self.w_rep * rep + self.w_bwd * bwd
                         + self.w_mass * mass + 0.5 * float((self.zeta ** 2).sum())
                         + flat)

    def distance_to_opt(self):
        with torch.no_grad():
            return float(torch.cat([self.theta.detach().reshape(-1),
                                    self.zeta.detach().reshape(-1),
                                    self.mu.reshape(-1)]).norm())

    def probe_loss(self, batch):
        n_theta, _ = batch
        with torch.no_grad():
            if self.probe_scores == 'replay':
                return float(self._replay_loss(n_theta))
            return float(self.w_rep * self._replay_loss(n_theta)
                         + self.w_bwd * self._bwd_loss()
                         + self.w_mass * self._mass_loss())

    def grad_on(self, batch):
        n_theta, _ = batch
        loss = (self.w_rep * self._replay_loss(n_theta)
                + self.w_bwd * self._bwd_loss()
                + self.w_mass * self._mass_loss())
        (g,) = torch.autograd.grad(loss, [self.theta])
        return g.detach().reshape(-1)

    def extra_state(self):
        return {'mu': self.mu.clone(), 'anchor': self.anchor.clone()}

    def load_extra_state(self, state):
        self.mu = state['mu'].clone()
        self.anchor = state['anchor'].clone()

    # ---------------------------------------------------------------- truth

    def curvature(self):
        """Per-coordinate training curvature. The flat block is exactly 0."""
        c = self.w_rep * self.S_rep.clone()
        c[:self.n_anchor] = c[:self.n_anchor] + self.w_bwd * self.S_bwd
        return c + self.w_mass          # the mass term is rank-1, +w_mass on the diagonal

    def sgd_stability_lr(self):
        """
        Exact only for SGD and only for the training objective. Kept because a
        cell with exact ground truth is worth having -- but it does NOT describe
        the shipped configuration, which is Adam WITH A BINDING CLIP, and a
        binding clip was measured to move the empirical boundary by 94x.
        """
        return float(2.0 / self.curvature().max())
