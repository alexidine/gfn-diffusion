"""
The synthetic games -- three loss surfaces chosen for their CHARACTER, not their
realism.

A quadratic bowl would make this whole bench worthless. On a quadratic, alpha* is
constant (so periodic recalibration measures nothing), batch size is decoupled
from optimization (so the two controllers cannot interact), and there is no
multi-step instability for a one-step sensor to miss -- which is the single most
important failure mode in the real system. The games below each reproduce the
structure of one training stage:

  mle            single player, regression-like. Curvature DRIFTS along the path,
                 so alpha* is a moving quantity even here.

  var_cond       two players, cooperative: one shared potential, but a per-
                 condition nuisance parameter that only updates for conditions
                 the batch happened to draw. Batch size therefore controls
                 CONDITION COVERAGE, not just gradient noise.

  equilibration  three players, NOT cooperative -- no joint potential. A policy
                 chasing a level, a level chasing the policy (positive feedback,
                 the documented Z-dispersion mechanism), and a lagged buffer
                 supplying the only restoring force. Its stability boundary in LR
                 is far below its one-step optimum, so a ray probe targeting
                 alpha* = 1 diverges by construction.

That last property is the point of the whole file. `alpha_target` defaults to 4
rather than 1 because a ray probe at frozen theta cannot see the tr(H*Sigma) term
-- an argument currently supported by one cluster measurement (tuphwfkm). Here
the boundary is a spectral radius we can compute exactly, so the claim becomes
checkable in a second on a laptop.

GROUND TRUTH is the other reason these are synthetic. Every game knows its own
optimum, and `equilibration` knows the exact LR at which it goes unstable. On a
crystal there is no oracle, so "controller broken" and "problem hard" are not
distinguishable; here they are.

NOISE. Each player's per-batch gradient noise enters as a linear term g^T theta
with g ~ N(0, sigma^2 / B). That is exact minibatch scaling (variance ~ 1/B),
costs one randn, and is differentiable -- so autograd, the optimizers and
RayCalibration all see a genuine stochastic gradient rather than a simulated one.

OPTIMIZER. Plain SGD by default, because it is the only choice for which alpha*
has a closed form (see MLEGame.alpha_star_true) and the exactness assertions are
worth more than the realism. Pass optimizer='adam' for the dynamics tests.
"""

import math

import numpy as np
import torch


def _mk_opt(kind, groups, lr):
    if kind == 'sgd':
        return torch.optim.SGD(groups, lr=lr)
    if kind == 'adam':
        return torch.optim.Adam(groups, lr=lr)
    raise ValueError(f'unknown optimizer {kind!r}')


def _full_optimizer_set(kind, policy, level, lrs):
    """
    Build the SAME five-key optimizer dict train.py always builds
    (train.py:1619-1664), whatever the game actually trains with.

    This matters for faithfulness in both directions. LRController._apply_lrs
    iterates every key and branches on 'flow' and on the trailing group of
    'fused', and LRController.step() ends with an unconditional
    optimizers['fwd'] lookup -- safe in production precisely because all five are
    built unconditionally. A game that exposed only the optimizer it uses would
    both miss those branches and make that lookup look like a latent bug.

    fwd/bwd/replay are identical up to their LR, over the same parameter list,
    exactly as in train.py. 'fused' carries the level params as its LAST group.
    """
    opts = {}
    for mode in ('fwd', 'bwd', 'replay'):
        opts[mode] = _mk_opt(kind, list(policy), lrs.get(mode, lrs['policy']))
    opts['fused'] = _mk_opt(
        kind, [{'params': list(policy)}, {'params': list(level)}], lrs['policy'])
    opts['fused'].param_groups[-1]['lr'] = lrs['flow']
    opts['flow'] = _mk_opt(kind, list(level), lrs['flow'])
    return opts


class _Game:
    """Common surface protocol. The harness and the tests only use this."""

    name = 'base'

    #: which optimizer key this game's train_step actually drives. The full
    #: five-key dict is always present (train.py builds it unconditionally), so
    #: reporting must name the one that moves or it silently reads a spectator.
    train_key = 'fwd'

    #: optimizers dict, keyed as train.py keys them ('fwd'/'bwd'/'replay'/'fused'/'flow').
    #: LRController._apply_lrs iterates exactly this, including its special cases for
    #: 'flow' and for the trailing group of 'fused', so the keys are load-bearing.
    optimizers: dict

    #: parameters RayCalibration snapshots. POLICY ONLY -- decision D26 option b:
    #: the flow/level head is LR-pinned separately and held at its post-step value,
    #: so it contributes an identical constant to every ray evaluation.
    policy_params: list

    def draw(self, batch_size):
        raise NotImplementedError

    def train_step(self, batch):
        """Apply one update to every player. Returns the reported scalar loss."""
        raise NotImplementedError

    def probe_loss(self, batch):
        """What RayCalibration scores. May deliberately differ from what trains."""
        raise NotImplementedError

    def distance_to_opt(self):
        raise NotImplementedError

    def advance(self):
        """
        Move any non-stationary part of the surface by one step. Default no-op.

        Called once per step by the runner. Exists so a surface can pose a
        TRACKING problem rather than a convergence one -- every cell built so far
        converges, so its optimal rate only ever decays, and a controller whose
        natural shape is overshoot-then-decay matches that trajectory for free.
        """

    def grad_on(self, batch):
        """
        Flat gradient at the CURRENT parameters on `batch`, WITHOUT stepping and
        without touching `.grad`.

        Exists so a sensor can draw two independent estimates of the same
        gradient at the same point -- which is the only way to measure the
        VARIANCE side of the step-size tradeoff. Every cross-step statistic sees
        alignment only.
        """
        raise NotImplementedError

    def expected_loss(self):
        """
        The loss with the minibatch noise term set to ZERO -- what the loss would
        be at an infinite batch, i.e. the quality of the parameters themselves.

        SCORE ON THIS, NOT ON THE TRAINING LOSS. The training loss carries a
        `noise_vec . theta` term whose sign is random, so as theta -> 0 it stops
        measuring the parameters and becomes the noise draw: near the optimum the
        series crosses zero constantly and a median over a window ranks arms by
        where that random term happened to land. Measured, this made the best arm
        on the board an arm whose loss trace was oscillating across zero.

        The CONTROLLER still sees the noisy loss, as in production. Only the
        scoring uses this.
        """
        raise NotImplementedError

    def diverged(self):
        d = self.distance_to_opt()
        return (not math.isfinite(d)) or d > 1e6

    # ---- non-parameter state, for the rewind

    def extra_state(self):
        """
        State a checkpoint must carry that is NOT an optimizer parameter.

        The divergence response in train.py is a REWIND plus a peak cut
        (`fire_loss_spike` -> `checkpointer.load_model_only`), so a harness that
        only cuts the peak is testing half of it -- and on a surface where the
        blow-up drives parameters non-finite, the half it omits is the half that
        recovers. Anything a game carries outside its parameters (a buffer mean,
        a staleness counter) has to ride along or the restore is incomplete.
        """
        return {}

    def load_extra_state(self, state):
        pass


# ---------------------------------------------------------------------------
# 1. MLE -- single player, regression-like
# ---------------------------------------------------------------------------

class MLEGame(_Game):
    """
    L(theta) = 1/2 theta^T H theta + c * sum(theta^4) + g^T theta

    H is diagonal with log-spaced eigenvalues spanning `cond`, so the surface is
    ill-conditioned the way a real regression is. The quartic term is what makes
    this more than a bowl: local curvature is H + 12c*diag(theta^2), which FALLS
    as the run converges, so alpha* RISES along the path. A controller that
    calibrates once and holds is wrong here by construction, which is the whole
    argument for periodic recalibration.

    Optimum is the origin, exactly.
    """

    name = 'mle'

    def __init__(self, dim=32, cond=30.0, quartic=0.0, noise=0.5, lr=1e-2,
                 optimizer='sgd', init_scale=1.0, seed=0, device='cpu',
                 floor=0.0, drift=0.0):
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.device = device
        self.dim = dim
        self.quartic = float(quartic)
        self.noise = float(noise)
        # AN UNKNOWN, NONZERO LOSS FLOOR.
        # As written, this game's expected loss bottoms out at exactly 0, which
        # is the INTERPOLATION REGIME -- and interpolation is the assumption that
        # makes the Polyak step size and its descendants work at all. The real
        # MLE stage has no such property: its TB loss has an irreducible floor
        # set by the stochastic policy, that floor is strictly positive, and
        # nobody knows its value. A surface that quietly satisfies the assumption
        # would score those methods on a problem we do not have.
        #
        # A constant offset is the minimal honest model of it. It changes no
        # gradient, so it is invisible to every method that reads only
        # DIFFERENCES of the loss (ray, armijo, plateau) and visible to every
        # method that reads its LEVEL or its RELATIVE rate (sps, and the slope
        # sensors, which normalise by the window's own magnitude). Which of our
        # sensors are floor-invariant is exactly the question worth asking, and
        # this isolates it.
        self.floor = float(floor)
        self._gen = g

        # log-spaced spectrum in [1, cond]
        self.H = torch.logspace(0, math.log10(cond), dim, dtype=torch.float64).float()
        theta0 = torch.randn(dim, generator=g) * init_scale
        self.theta = torch.nn.Parameter(theta0.clone())
        # A MOVING OPTIMUM. `drift` is the per-step speed; the direction is a
        # fixed random unit vector, so the target recedes steadily and the run
        # NEVER converges -- the rate has to stay up to keep tracking rather
        # than decay to a noise floor.
        self.theta_opt = torch.zeros(dim)
        self.drift = float(drift)
        if self.drift:
            v = torch.randn(dim, generator=g)
            self._drift_vec = v / v.norm() * self.drift
        else:
            self._drift_vec = None
        # a Z head that exists but does not enter this stage's loss -- mk_dev's
        # unconditional flow_model is a LearnableScalar, and phase 1 does not
        # train it either. Present so the flow-pinning branch is still exercised.
        self.zeta = torch.nn.Parameter(torch.zeros(1))

        self.optimizers = _full_optimizer_set(
            optimizer, [self.theta], [self.zeta], {'policy': lr, 'flow': lr})
        self.policy_params = [self.theta]

    # ---- data

    def draw(self, batch_size):
        b = max(1, int(batch_size))
        # exact minibatch scaling: mean of b iid unit draws has variance 1/b
        g = torch.randn(self.dim, generator=self._gen) * (self.noise / math.sqrt(b))
        return g

    # ---- objective

    def advance(self):
        if self._drift_vec is not None:
            self.theta_opt = self.theta_opt + self._drift_vec

    def _loss(self, noise_vec, theta=None):
        t = (self.theta if theta is None else theta) - self.theta_opt
        quad = 0.5 * (self.H * t * t).sum()
        quart = self.quartic * (t ** 4).sum() if self.quartic else 0.0
        return quad + quart + (noise_vec * t).sum() + self.floor

    def train_step(self, batch):
        opt = self.optimizers['fwd']
        opt.zero_grad(set_to_none=True)
        loss = self._loss(batch)
        loss.backward()
        opt.step()
        return float(loss.detach())

    def lr(self):
        return self.optimizers['fwd'].param_groups[0]['lr']

    def probe_loss(self, batch):
        with torch.no_grad():
            return float(self._loss(batch))

    # ---- ground truth

    def hessian_diag(self, theta=None):
        t = (self.theta if theta is None else theta).detach() - self.theta_opt
        h = self.H.clone()
        if self.quartic:
            h = h + 12.0 * self.quartic * t * t
        return h

    def true_grad(self, theta=None):
        t = (self.theta if theta is None else theta).detach() - self.theta_opt
        g = self.H * t
        if self.quartic:
            g = g + 4.0 * self.quartic * t ** 3
        return g

    def alpha_star_true(self, theta_before, d):
        """
        Exact one-step optimum along the ray theta(alpha) = theta_before + alpha*d,
        for the POPULATION loss:

            alpha* = -(grad.d) / (d^T H_eff d)

        evaluated at theta_before. This is the quantity RayCalibration brackets,
        so it is the reference every sensor test is scored against.
        """
        g = self.true_grad(theta_before)
        h = self.hessian_diag(theta_before)
        num = -float((g * d).sum())
        den = float((d * h * d).sum())
        return num / den if den > 0 else float('nan')

    def distance_to_opt(self):
        return float((self.theta.detach() - self.theta_opt).norm())

    def expected_loss(self):
        with torch.no_grad():
            return float(self._loss(torch.zeros_like(self.theta)))

    def grad_on(self, batch):
        g, = torch.autograd.grad(self._loss(batch), self.theta)
        return g.detach().reshape(-1)


# ---------------------------------------------------------------------------
# 2. var_conditioning -- two players, cooperative, sparse per-condition levels
# ---------------------------------------------------------------------------

class VarCondGame(_Game):
    """
    L(theta, zeta) = E_c[ 1/2 (a_c . theta + zeta_c - t_c)^2 ] + ridge/2 * |zeta|^2

    Cooperative: both players descend ONE potential, so there is a joint optimum
    and no game-theoretic instability. What makes it non-trivial is that zeta is
    PER CONDITION and only the conditions a batch drew receive a gradient.

    That reproduces the thing batch size actually does in the conditional route,
    which is not noise reduction: it is condition coverage. At batch 32 with 256
    conditions, any given zeta_c updates once per 8 steps and is stale in
    between, so theta is fitting against a level that has not caught up. Larger
    batches shorten that lag. A bench where batch size only scaled 1/sqrt(B)
    noise would miss this entirely.

    The t_c are drawn with deliberately wide spread, because per-condition level
    dispersion of hundreds of nats is what breaks pooled metrics in the real run.

    zeta is the LAST group of the fused optimizer, so it takes LRController's
    flow-pinning branch exactly as the real Z head does.
    """

    name = 'var_cond'
    train_key = 'fused'

    def __init__(self, dim=16, n_cond=256, spread=50.0, ridge=1e-3, noise=0.5,
                 lr=1e-2, lr_level=1e-2, optimizer='sgd', seed=0, device='cpu'):
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.device = device
        self.dim, self.n_cond = dim, n_cond
        self.ridge, self.noise = float(ridge), float(noise)
        self._gen = g

        self.A = torch.randn(n_cond, dim, generator=g) / math.sqrt(dim)
        self.t = torch.randn(n_cond, generator=g) * spread

        self.theta = torch.nn.Parameter(torch.zeros(dim))
        self.zeta = torch.nn.Parameter(torch.zeros(n_cond))

        # theta first, zeta LAST: _apply_lrs pins the trailing fused group at lr_flow
        self.optimizers = _full_optimizer_set(
            optimizer, [self.theta], [self.zeta], {'policy': lr, 'flow': lr_level})
        self.policy_params = [self.theta]

        self._solve_optimum()
        self.cond_last_seen = np.zeros(n_cond, dtype=np.int64)
        self._step = 0

    def _solve_optimum(self):
        """
        Joint minimiser in closed form. Stationarity in zeta gives
        zeta_c = (t_c - a_c.theta) / (1 + ridge*n_cond/n_cond) -- with the ridge
        written per-condition this is zeta_c = (t_c - a_c.theta)/(1+ridge).
        Substituting into the theta equation leaves an ordinary least squares
        problem in theta alone.
        """
        r = 1.0 / (1.0 + self.ridge)
        # residual after zeta absorbs what it can: (1-r)(t - A theta)
        A, t = self.A.double(), self.t.double()
        w = (1.0 - r)
        # minimise w^2/2 |t - A theta|^2  ->  ordinary least squares
        sol = torch.linalg.lstsq(A, t).solution if w > 0 else torch.zeros(self.dim, dtype=torch.float64)
        self.theta_opt = sol.float()
        self.zeta_opt = (r * (t - A @ sol)).float()

    def draw(self, batch_size):
        # conditions per step scales with batch, capped by the library size
        n = max(1, min(self.n_cond, int(batch_size)))
        idx = torch.randperm(self.n_cond, generator=self._gen)[:n]
        g = torch.randn(self.dim, generator=self._gen) * (self.noise / math.sqrt(max(1, int(batch_size))))
        return idx, g

    def _loss(self, batch):
        idx, noise_vec = batch
        resid = (self.A[idx] @ self.theta) + self.zeta[idx] - self.t[idx]
        return 0.5 * (resid ** 2).mean() + 0.5 * self.ridge * (self.zeta[idx] ** 2).mean() \
            + (noise_vec * self.theta).sum()

    def train_step(self, batch):
        idx, _ = batch
        opt = self.optimizers['fused']
        opt.zero_grad(set_to_none=True)
        loss = self._loss(batch)
        loss.backward()
        opt.step()
        self._step += 1
        self.cond_last_seen[idx.numpy()] = self._step
        return float(loss.detach())

    def probe_loss(self, batch):
        with torch.no_grad():
            return float(self._loss(batch))

    # ---- ground truth

    def staleness(self):
        """Mean age, in steps, of the per-condition levels. The lag batch size buys down."""
        return float(self._step - self.cond_last_seen.mean())

    def extra_state(self):
        return {'cond_last_seen': self.cond_last_seen.copy(), 'step': self._step}

    def load_extra_state(self, state):
        self.cond_last_seen = state['cond_last_seen'].copy()
        self._step = state['step']

    def distance_to_opt(self):
        return float((self.theta.detach() - self.theta_opt).norm())


# ---------------------------------------------------------------------------
# 3. equilibration -- three players, non-cooperative, buffer lag
# ---------------------------------------------------------------------------

class EquilibrationGame(_Game):
    """
    No joint potential. Each player descends its OWN objective:

        replay (theta):  1/2 |b*theta - zeta|^2      -- policy chases the level
        bwd    (theta):  1/2 |theta - mu|^2          -- policy pulled to the buffer
        flow   (zeta):   1/2 |zeta + a*theta|^2      -- level moves ANTI-PHASE

    THE SIGN IS THE WHOLE DESIGN. The policy chases the level with +, the level
    responds to the policy with -, so the coupling is ANTISYMMETRIC and the pair
    is a SPIRAL rather than a race: perturb theta and the system circles the fixed
    point instead of running away from it. That matches the measured behaviour --
    Z's target moves anti-phase to the policy -- and it is what gives the system
    a finite LR stability boundary rather than an unconditional instability.

    (A symmetric +/+ coupling, the naive reading of "Z chases the policy", makes
    the 2x2 block's determinant c - a*w_rep*b, which is negative for any
    interesting a: the fixed point is then a saddle and NOTHING is stable at any
    LR. Worth stating because it is the obvious first guess and it is wrong.)

        buffer:          mu <- (1-kappa)*mu + kappa*theta

    The buffer supplies the extra restoring force and a slow third pole -- the
    architecture's own claim, that the off-policy buffer branch is what prevents
    collapse, made mechanical.

    WHY THIS IS THE IMPORTANT GAME. The iteration is linear in (theta, zeta, mu),
    so `stability_lr()` returns the exact LR at which the spectral radius reaches
    1. Meanwhile a ray probe at frozen zeta and mu sees only the theta-curvature
    w_rep*b^2 + w_bwd and reports alpha* = 1 at lr = 1/(w_rep*b^2 + w_bwd) -- a
    rate typically many times ABOVE the stability boundary. The probe is not
    wrong; it is answering a one-step question about a multi-step system. Every
    claim about alpha_target being 4 rather than 1 is testable against these two
    numbers.

    kappa is the buffer churn rate. Small kappa is a slow pole: the restoring
    force arrives late, and the system limit-cycles at a period set by kappa
    rather than by LR.
    """

    name = 'equilibration'
    train_key = 'fused'

    def __init__(self, dim=8, a=2.0, b=1.0, w_rep=0.7, w_bwd=0.3, kappa=0.02,
                 noise=0.1, lr=0.05, optimizer='sgd', init_scale=1.0,
                 probe_scores='replay', seed=0, device='cpu',
                 cond_rep=1.0, cond_bwd=1.0):
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.device = device
        self.dim = dim
        self.a, self.b = float(a), float(b)
        self.w_rep, self.w_bwd = float(w_rep), float(w_bwd)
        self.kappa = float(kappa)
        self.noise = float(noise)
        self.probe_scores = probe_scores
        self._gen = g

        # COMPETING BRANCHES NEED DIFFERENT GEOMETRY, or they do not compete.
        #
        # With scalar curvatures the theta-problem is w_rep*b^2 + w_bwd -- a
        # single ISOTROPIC attractor at a weighted midpoint, condition number 1.
        # Two isotropic pulls average; they never disagree about which direction
        # wants a small step, so no learning-rate headroom is lost and the usable
        # band is enormous. Measured on the scalar version: a 50x rate sweep moved
        # the settled distance 13x and `null` was competitive with every arm.
        #
        # `cond_rep` and `cond_bwd` give each branch its own log-spaced spectrum,
        # DELIBERATELY IN OPPOSITE ORDER, so the direction replay finds stiffest
        # is the one bwd finds softest. That is what "multiple competing
        # optimizations on a shared policy model" means mechanically, and it is
        # the part the scalar game left out.
        self.S_rep = torch.logspace(0, math.log10(max(cond_rep, 1.0)),
                                    dim, dtype=torch.float64).float()
        self.S_bwd = torch.logspace(math.log10(max(cond_bwd, 1.0)), 0,
                                    dim, dtype=torch.float64).float()
        t0 = torch.randn(dim, generator=g) * init_scale
        self.theta = torch.nn.Parameter(t0.clone())
        self.zeta = torch.nn.Parameter(self.a * t0.clone())   # level starts consistent
        self.mu = t0.clone()                                   # buffer starts at theta

        self.optimizers = _full_optimizer_set(
            optimizer, [self.theta], [self.zeta], {'policy': lr, 'flow': lr})
        self.policy_params = [self.theta]

    # ---- data

    def draw(self, batch_size):
        b = max(1, int(batch_size))
        s = self.noise / math.sqrt(b)
        return (torch.randn(self.dim, generator=self._gen) * s,
                torch.randn(self.dim, generator=self._gen) * s)

    # ---- objectives, one per player

    def _replay_loss(self, n_theta):
        r = self.b * self.theta - self.zeta.detach()
        return 0.5 * (self.S_rep * r * r).sum() + (n_theta * self.theta).sum()

    def _bwd_loss(self):
        d = self.theta - self.mu
        return 0.5 * (self.S_bwd * d * d).sum()

    def _flow_loss(self, n_zeta):
        r = self.zeta + self.a * self.theta.detach()      # anti-phase; see class docstring
        return 0.5 * (r ** 2).sum() + (n_zeta * self.zeta).sum()

    def train_step(self, batch):
        n_theta, n_zeta = batch
        opt = self.optimizers['fused']
        opt.zero_grad(set_to_none=True)

        theta_loss = self.w_rep * self._replay_loss(n_theta) + self.w_bwd * self._bwd_loss()
        flow_loss = self._flow_loss(n_zeta)

        # each player's gradient comes from its OWN objective -- this is what makes
        # the system a game rather than a joint descent, and it is exactly how
        # train.py wires per-group optimizers
        (g_theta,) = torch.autograd.grad(theta_loss, [self.theta], retain_graph=True)
        (g_zeta,) = torch.autograd.grad(flow_loss, [self.zeta])
        self.theta.grad, self.zeta.grad = g_theta, g_zeta

        theta_before = self.theta.detach().clone()
        opt.step()
        # buffer admits the sample that was just produced
        self.mu = (1.0 - self.kappa) * self.mu + self.kappa * theta_before
        return float(theta_loss.detach())

    def expected_loss(self):
        """
        The three players' objectives, summed, with every noise term zeroed.

        THERE IS NO JOINT POTENTIAL HERE -- that is the point of the game -- so
        this is NOT something anybody descends. It is a distance-to-equilibrium:
        each term is a squared residual, all three vanish exactly at the fixed
        point (theta = zeta = mu = 0) and are positive elsewhere. That makes it a
        legitimate SCORE even though it is not an objective, in the same way
        "how far is the system from settled" is well posed without any player
        optimising it.

        Scoring the noisy `train_step` return instead would repeat the MLE trap:
        near the fixed point the residuals vanish and what is left is the noise
        draw, sign and all, so arms get ranked on a coin flip.
        """
        with torch.no_grad():
            r = self.b * self.theta - self.zeta
            d = self.theta - self.mu
            rep = 0.5 * (self.S_rep * r * r).sum()
            bwd = 0.5 * (self.S_bwd * d * d).sum()
            flw = 0.5 * ((self.zeta + self.a * self.theta) ** 2).sum()
            return float(self.w_rep * rep + self.w_bwd * bwd + flw)

    def distance_to_opt(self):
        """Whole-state distance from the fixed point, all three players."""
        with torch.no_grad():
            return float(torch.cat([self.theta.detach().reshape(-1),
                                    self.zeta.detach().reshape(-1),
                                    self.mu.reshape(-1)]).norm())

    def probe_loss(self, batch):
        """
        DEFAULT: score the replay objective ONLY, while the step trained
        replay + bwd. That is not a simplification -- it is the real probe's
        documented behaviour (it draws from replay and scores with
        replay_loss_coeffs), and it means the sensor rates a loss nobody is
        wholly optimising. Set probe_scores='total' to switch the mismatch off
        and measure what it was worth.
        """
        n_theta, _ = batch
        with torch.no_grad():
            if self.probe_scores == 'replay':
                return float(self._replay_loss(n_theta))
            if self.probe_scores == 'total':
                return float(self.w_rep * self._replay_loss(n_theta)
                             + self.w_bwd * self._bwd_loss())
            if self.probe_scores == 'system':
                # ALSO score the level's own objective. zeta stays frozen, but
                # ||zeta + a*theta||^2 still MOVES when theta moves -- so this is
                # the only term in which the probe can see the cost its theta-step
                # imposes on the other player. Everything else it scores is about
                # theta alone, which is why the probe is blind to the loop that
                # F-012 showed sets the stability boundary.
                n_theta_, n_zeta = batch
                return float(self.w_rep * self._replay_loss(n_theta_)
                             + self.w_bwd * self._bwd_loss()
                             + self._flow_loss(n_zeta))
            raise ValueError(self.probe_scores)

    # ---- ground truth

    def iteration_matrix(self, lr, lr_level=None, coord=None):
        """
        The exact linear map on (theta, zeta, mu) for one SGD step at these rates.

        ONE 3x3 PER COORDINATE. The branches carry their own spectra `S_rep` and
        `S_bwd` (opposed, so the direction replay finds stiffest is the one bwd
        finds softest), so the dimensions are NO LONGER identical and a single
        3x3 does not cover the system. `coord=None` returns the stiffest
        coordinate's block, which is the one that sets stability.

        This used to say "per-dimension identical, so a 3x3 suffices" -- true
        only while both curvatures were scalar. With `cond_rep=100` the empirical
        cliff moved from 2.15 to ~0.03 while this function kept returning 2.15,
        i.e. the ground truth silently stopped describing the game.
        """
        e_t = float(lr)
        e_z = float(lr if lr_level is None else lr_level)
        if coord is None:
            # the stiffest theta-curvature is what binds
            c_all = self.w_rep * self.S_rep * self.b ** 2 + self.w_bwd * self.S_bwd
            coord = int(np.argmax(c_all.numpy()))
        s_rep = float(self.S_rep[coord])
        s_bwd = float(self.S_bwd[coord])
        c = self.w_rep * s_rep * self.b ** 2 + self.w_bwd * s_bwd
        return np.array([
            [1.0 - e_t * c, e_t * self.w_rep * s_rep * self.b, e_t * self.w_bwd * s_bwd],
            [-e_z * self.a, 1.0 - e_z, 0.0],               # anti-phase: note the sign
            [self.kappa, 0.0, 1.0 - self.kappa],
        ])

    def extra_state(self):
        return {'mu': self.mu.clone()}

    def load_extra_state(self, state):
        self.mu = state['mu'].clone()

    def spectral_radius(self, lr, lr_level=None):
        """Max over COORDINATES of the per-coordinate 3x3 spectral radius."""
        return max(
            float(np.max(np.abs(np.linalg.eigvals(
                self.iteration_matrix(lr, lr_level, coord=i)))))
            for i in range(self.dim))

    def stability_lr(self, lo=1e-6, hi=10.0, iters=200, lr_level=None):
        """
        Largest POLICY LR with spectral radius < 1. Bisection; 1e-6 relative.

        lr_level=None scales both players together (the symmetric case the closed
        form in test_games covers). Passing a number holds the level's rate fixed
        while the policy's is swept -- which is the real configuration, since the
        Z head is pinned at lr_flow and exempt from the servo.

        Note rho(0) = 1 exactly -- at zero LR nothing moves, so the map is the
        identity on (theta, zeta) and the buffer contributes 1-kappa. The
        bisection therefore starts from a small POSITIVE lo and returns 0.0 only
        when the system is unstable even there, which means the continuous-time
        fixed point is not attracting at all (see the +/+ note in the docstring).
        """
        if self.spectral_radius(lo, lr_level) >= 1.0:
            return 0.0
        for _ in range(iters):
            mid = math.sqrt(lo * hi)
            if self.spectral_radius(mid, lr_level) < 1.0:
                lo = mid
            else:
                hi = mid
        return lo

    def one_step_lr(self):
        """
        LR at which a frozen-target ray probe reads alpha* = 1 -- the rate the
        sensor would drive to if alpha_target were 1. Compare with stability_lr().
        """
        return 1.0 / (self.w_rep * self.b ** 2 + self.w_bwd)

    def distance_to_opt(self):
        return float(self.theta.detach().norm())


GAMES = {'mle': MLEGame, 'var_cond': VarCondGame, 'equilibration': EquilibrationGame}
