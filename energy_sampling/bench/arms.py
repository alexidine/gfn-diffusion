"""
THE CONTROLLERS UNDER TEST, and the fixed rates they have to beat.

Five kinds, and the last two exist to make the others interpretable:

  hyper           lr *= exp(beta * cos(g_t, g_{t-1}))
  ray             the shipping periodic ray calibration, both roles
  ramp+plateau    blind ramp up, ReduceLROnPlateau down
  fixed@X         a constant rate. THESE REPLACE THE ORACLE.
  null            servo live, no sensor -- the do-nothing control

WHY FIXED RATES ARE ARMS. The goal is stated as "at worst ~2x the best fixed
rate". The old stack computed that best rate up front and divided by it, which
put a selection step -- with an edge guard, a refinement pass, a time-vs-distance
choice and a feasibility predicate -- between the data and every number. All four
went wrong at least once. A ladder of fixed rates on the same leaderboard answers
the same question by subtraction, and there is nothing to select.

WHY NULL STAYS. It earned its place by revealing that four of five scenarios in
the old battery were pass-through. An arm that cannot beat doing nothing is not a
controller, and only `null` can say so.

(THE SECOND HALF OF THAT CLAIM IS RETRACTED: "`ramp+plateau` is worse than having
no sensor at all on a multi-player surface" does not reproduce. On the current
equilibration surface, 5 seeds, `ramp+plateau` finishes 0.03 nats behind the best
arm and `null` finishes 4.01 behind -- the ramp is ~4 nats BETTER, not worse. The
old reading came from the stale equilibration base that `eqboard` and `audit`
were still importing.)

ON THE NAME "hyper". This is NOT Baydin et al.'s published rule, and the old docs
called it that. Published is additive and unnormalised, `lr += beta * <g_t,
g_{t-1}>`; this is multiplicative on the cosine. They share a fixed point and
nothing else: beta changes from units of `lr/gradient^2` to dimensionless, and
the additive form's self-annealing (its step shrinks with ||g||^2 near a
stationary point) is gone. On real measured gradients (||g|| ~ 583, cos ~ 0.29)
the published rule at the paper's beta=1e-7 would multiply the LR by ~80 in a
single step. The cosine form is the defensible choice; the citation was not.

HYPER DOES HAVE A PRODUCTION COUNTERPART, and this file used to say it did not.
`LRController.on_hypergradient` (controller.py:237) ships; `train.py:3089` calls
it when a stage selects `lr_sensor: {kind: hyper}`; `protocol.py:121` lists 'hyper' beside 'ray' and
'plateau' as a configurable `lr_sensor` kind with its own beta/beta_down/every
schema. Its law is `peak_scale *= exp(beta * cos)` -- bit-identical to this
file's actuator -- and its operand is the REALISED DISPLACEMENT, i.e.
`HyperStep`'s, not `Hyper`'s.

That matters for how the boards read. `Hyper` correlates the current gradient
with the PREVIOUS GRADIENT, which is the right statistic only under SGD, and it
exists in no production path. The README headline that hyper "ends 89x too hot
and loses to four fixed rates" was measured on that arm; the operand production
actually uses scores 1.81x on the same cell. `HyperStep` now routes through the
shipping call so the production counters and status channel are exercised.
"""
import math

import torch


class Arm:
    """A controller. `tick` is called once per step, after the optimizer step."""

    needs_gradient = False

    def __init__(self, name):
        self.name = name

    def args_overrides(self):
        """Config the run should be built with. Sensors off unless asked for."""
        return {'lr_servo_managed': ('lr_policy', 'lr_fused'),
                'ray_calibration.enabled': False}

    def reset(self, run):
        pass

    def pre_step(self, run):
        """Before the optimizer step. Only the ray probe needs this, and it
        needs it absolutely: `RayCalibration.arm` snapshots the parameters to
        difference against, so arming after the step measures a zero delta."""
        pass

    def tick(self, run, loss, g_before, batch):
        pass

    def on_rewind(self, run):
        """
        The parameters just jumped backwards. Drop any cross-step state.

        Every arm here differences something against the PREVIOUS step -- a
        gradient, a realised displacement, a best-loss watermark. After a rewind
        that previous step belongs to a trajectory that no longer exists, so the
        next difference spans the discontinuity and reports a huge spurious
        move. For `hyper` that is a cosine against a jump of the size of the
        whole detonation, applied at the exact moment the run is least able to
        absorb a bad verdict.
        """
        pass

    # -- shared actuator ---------------------------------------------------
    @staticmethod
    def _scale_peak(run, factor):
        """
        Multiply `peak_scale`, respecting the SHIPPING bounds, ceiling and
        warmup hold. This is the actuator every arm shares; the update LAW is
        what differs between them, and that difference is the point rather than
        a flaw.
        """
        ctrl = run.m.lr_controller
        st = ctrl._state()
        # The SHIPPING warmup hold, taken from the controller rather than
        # reimplemented: `on_calibration` and `on_plateau` both gate on exactly
        # this (controller.py:186, :217). Acting during warmup would let an arm
        # move a rate the envelope is deliberately ramping.
        if ctrl._elapsed(st) < int(ctrl._cfg('warmup_steps', 1000)):
            return
        lo, hi = ctrl._peak_bounds()
        ceiling = ctrl._current_ceiling()
        if ceiling is not None:
            hi = min(hi, ceiling)
        st['peak_scale'] = max(lo, min(hi, float(st['peak_scale']) * float(factor)))
        st['envelope'] = ctrl._envelope(st)
        ctrl._apply_lrs(st)


class Fixed(Arm):
    """A constant rate. The bar every controller has to clear."""

    def __init__(self, lr):
        super().__init__(f'fixed@{lr:g}')
        self.lr = float(lr)

    def args_overrides(self):
        return {'lr_policy': self.lr, 'lr_fused': self.lr,
                'lr_servo_managed': (),      # nothing manages it
                'lr_warmup_ratio': 1,        # and no warmup envelope
                'ray_calibration.enabled': False}


class Null(Arm):
    """Servo live -- warmup, bounds, tripwire -- and no sensor at all."""

    def __init__(self, lr):
        super().__init__('null (no sensor)')
        self.lr = float(lr)

    def args_overrides(self):
        return {'lr_policy': self.lr, 'lr_fused': self.lr,
                'lr_servo_managed': ('lr_policy', 'lr_fused'),
                'ray_calibration.enabled': False}


class Hyper(Arm):
    """
    lr *= exp(beta * cos(g_t, g_{t-1})), clipped to the shipping bounds.

    Bounded and proportional: `cos` is a cosine, so the response is capped at
    `exp(+-beta)` per step no matter how wrong the rate is. That is what makes it
    safe under noise and also what makes it slow to make a large correction --
    the trade is intrinsic, not a tuning error.
    """

    needs_gradient = True

    def __init__(self, lr, beta=0.02, beta_down=None):
        super().__init__(f'hyper b={beta:g}' if beta_down in (None, beta)
                         else f'hyper {beta:g}/{beta_down:g}')
        self.lr = float(lr)
        self.beta = float(beta)
        self.beta_down = float(beta if beta_down is None else beta_down)

    def args_overrides(self):
        return {'lr_policy': self.lr, 'lr_fused': self.lr,
                'lr_servo_managed': ('lr_policy', 'lr_fused'),
                'ray_calibration.enabled': False}

    def reset(self, run):
        self._prev = None

    def on_rewind(self, run):
        self._prev = None

    def tick(self, run, loss, g_before, batch):
        if g_before is None:
            return
        prev, self._prev = self._prev, g_before.clone()
        if prev is None:
            return
        na, nb = float(g_before.norm()), float(prev.norm())
        if not (na > 0 and nb > 0):
            return
        cos = float(torch.dot(g_before, prev)) / (na * nb)
        if not math.isfinite(cos):
            return
        beta = self.beta if cos > 0 else self.beta_down
        self._scale_peak(run, math.exp(beta * cos))


class HyperStep(Hyper):
    """
    Hypergradient against THE DIRECTION ACTUALLY STEPPED IN, not the raw
    gradient. A bug fix, not a new heuristic.

    The identity is `dL/d(lr) = -<g_t, d_{t-1}>` where `d` is the update
    direction, which follows from `theta_t = theta_{t-1} - lr*d_{t-1}`. Under SGD
    `d = g`, so `cos(g_t, g_{t-1})` is correct and this arm reduces to `hyper`.
    Under ADAM `d = mhat/(sqrt(vhat)+eps)`, and on an ill-conditioned surface the
    preconditioner rescales coordinates by wildly different factors -- so `g` and
    `d` point in materially different directions and the plain arm has been
    correlating the wrong pair. Baydin et al. derive Adam-HD with exactly this
    correction.

    `d` is read as the REALISED parameter displacement rather than from optimizer
    internals, so this is optimizer-agnostic and cannot drift out of sync with
    what the optimizer actually did.
    """

    #: WHERE THIS ARM SETTLES, and the knob that moves it.
    #:
    #: The update is `exp(beta * (cos - target))`, so the fixed point is at
    #: `cos = target`. With target 0 that is `cos = 0`: the new gradient is
    #: orthogonal to the step just taken, which on a quadratic is EXACTLY the
    #: line-search optimum -- i.e. this arm targets alpha* = 1 in the ray probe's
    #: units, while the shipping servo targets alpha* = 4.
    #:
    #: RETRACTED: that cos ~ `1 - lr/lr_opt`, so target `1 - 1/A` parks the arm
    #: at `1/A` of the line-search optimum. That roughly holds under SGD. Under
    #: ADAM the lr->0 asymptote is ~0.5 rather than +1 and the curve is
    #: non-monotone, so target->lr is not invertible; any target >= 0.6 sits
    #: above the whole attainable range and the arm sinks to the peak floor.
    #: The `target=0.75` once recommended here returns a dead arm.
    #:
    #: Swept, target=0 gives the best worst case across optimizers -- a setpoint
    #: does not buy generality. No shipped board sets `target`.
    def __init__(self, lr, beta=0.02, target=0.0, period=1, beta_down=None):
        # ASYMMETRIC GAIN. `beta_down` is the gain when the statistic says TOO
        # HOT. The base class has always had it; this subclass did not pass it
        # through, so no arm on any board could ever use it.
        #
        # It is the lever for the second half of the stated goal. "At worst ~2x"
        # and "never 50x" are different requirements: the first wants a good
        # setpoint, the second wants the response to be FASTER DOWNWARD than
        # upward, so an excursion is corrected before it compounds while the
        # climb stays gentle enough not to chase noise.
        super().__init__(lr, beta=beta, beta_down=beta_down)
        self.target = float(target)
        self.period = int(period)
        base = (f'hyper b={beta:g}' if beta_down in (None, beta)
                else f'hyper b={beta:g}/{beta_down:g}')
        self.name = (f'{base} step' if not target and period == 1
                     else f'{base} t={target:g} p={period}')

    def reset(self, run):
        super().reset(run)
        self._theta_before = None
        self._last_step = None

    def on_rewind(self, run):
        super().on_rewind(run)
        # `_last_step` is the displacement of a step on the abandoned
        # trajectory; differencing across the rewind would read the detonation
        # itself as this step's update direction.
        self._theta_before = None
        self._last_step = None

    def pre_step(self, run):
        self._theta_before = torch.cat(
            [p.detach().reshape(-1).clone() for p in run.game.policy_params])

    def tick(self, run, loss, g_before, batch):
        # `_theta_before` is None only when `on_rewind` cleared it EARLIER IN
        # THIS STEP -- the divergence check runs between `pre_step` and `tick`.
        # Without this guard the rewind path raises on the subtraction.
        if self._theta_before is None:
            return
        after = torch.cat([p.detach().reshape(-1)
                           for p in run.game.policy_params])
        step, self._last_step = self._last_step, after - self._theta_before
        if g_before is None or step is None:
            return
        # d = -step/lr, and lr > 0, so cos(g_t, -step) has the sign we want
        d = -step
        na, nb = float(g_before.norm()), float(d.norm())
        if not (na > 0 and nb > 0):
            return
        cos = float(torch.dot(g_before, d)) / (na * nb)
        if not math.isfinite(cos):
            return
        # THE STATISTIC IS MEASURED EVERY STEP; ONLY THE ACTUATION IS GATED.
        # That is the honest way to separate "ray is slower" from "ray measures
        # something different" -- a period that also skipped the measurement
        # would confound the two, which is the whole point of the comparison.
        if self.period > 1 and run.m.step_ind % self.period:
            return
        err = cos - self.target
        if not self.target:
            # THROUGH THE SHIPPING SENSOR. `LRController.on_hypergradient` exists
            # (controller.py:237), train.py calls it on exactly this operand
            # (:3089) -- the realised displacement --
            # and protocol.py lists 'hyper' as a first-class lr_sensor kind.
            # Verified bit-identical to the local actuator on all six tracking
            # cells, so routing through it changes no number and buys the
            # production counters, the status channel and the first-fire
            # announce, none of which any bench cell used to exercise.
            run.m.lr_controller.on_hypergradient(err, self.beta, self.beta_down)
            return
        # A NONZERO TARGET IS A BENCH EXTENSION. The shipping sensor has no
        # setpoint, so it cannot carry this; using the local actuator keeps the
        # reported `hyper_cos` the raw cosine rather than a shifted error.
        beta = self.beta if err > 0 else self.beta_down
        self._scale_peak(run, math.exp(beta * err))


class HyperSNR(Arm):
    """
    Drive the rate off the GRADIENT SIGNAL-TO-NOISE RATIO, measured by splitting
    the batch -- the variance side of the tradeoff, which no cross-step statistic
    can see.

    THE MEASUREMENT. Two independent half-batch gradients at the SAME point:
        snr = cos(g_A, g_B)  ->  ||gbar||^2 / (||gbar||^2 + sigma^2)
    Two half-batch backward passes cost about what one full-batch pass costs, so
    this is nearly free, and -- unlike a loss probe -- it needs NO scalar
    objective, just two estimates of the same gradient. That is what should let
    it survive a multi-player game where "did the loss go down" is ill-posed.

    THE SETPOINT IS DERIVED, NOT CHOSEN. At equilibrium in a noise ball the ball
    radius is `r ~ lr*sigma` and the mean gradient is `||gbar|| ~ lambda*r`, so
        snr = (lr*lambda)^2 / ((lr*lambda)^2 + 1)
    i.e. `snr` is a monotone increasing function of `lr*lambda` -- exactly the
    quantity that sets stability. `lr*lambda = 1` gives **snr = 0.5**, so that is
    the setpoint, and `snr -> 1` means far too hot while `snr -> 0` means the
    gradient is buried in noise.

    WHY THIS IS NOT THE THRESHOLD-ON-A-NOISY-STATISTIC DEATH (four arms died of
    that): the response is PROPORTIONAL to the error, not an asymmetric switch,
    and the statistic is a PAIRED estimate -- both halves are drawn at the same
    point, so the common signal cancels out of the comparison rather than having
    to be averaged out.

    Derived for the SGD quadratic. Under Adam the preconditioner changes the
    effective lambda, so 0.5 is a hypothesis there, not a derivation.
    """

    needs_gradient = False

    def __init__(self, lr, beta=0.02, target=0.5, period=1):
        super().__init__(f'hyper snr t={target:g}')
        self.lr = float(lr)
        self.beta = float(beta)
        self.target = float(target)
        self.period = int(period)

    def args_overrides(self):
        return {'lr_policy': self.lr, 'lr_fused': self.lr,
                'lr_servo_managed': ('lr_policy', 'lr_fused'),
                'ray_calibration.enabled': False}

    def reset(self, run):
        self.snr_log = []
        #: COUNTED, because this arm's two failure modes are both SILENT.
        #: `_Game.grad_on` is a base-class stub that RAISES, so `hasattr` is
        #: True on every game and a bare `except` turned "this surface cannot
        #: support the sensor" into "the sensor chose not to act" -- measured, 0
        #: readings in 300 steps on EquilibrationGame with no error and no tell,
        #: which is indistinguishable from the null arm.
        self.unsupported = 0

    def tick(self, run, loss, g_before, batch):
        if self.period > 1 and run.m.step_ind % self.period:
            return
        game = run.game
        half = max(1, run.batch // 2)
        try:
            ga = game.grad_on(game.draw(half))
            gb = game.grad_on(game.draw(half))
        except NotImplementedError:
            self.unsupported += 1
            return
        except Exception:
            self.unsupported += 1
            return
        na, nb = float(ga.norm()), float(gb.norm())
        if not (na > 0 and nb > 0):
            return
        snr = float(torch.dot(ga, gb)) / (na * nb)
        if not math.isfinite(snr):
            return
        self.snr_log.append(snr)
        # snr RISES with lr, so an excess means too hot -> cool.
        self._scale_peak(run, math.exp(self.beta * (self.target - snr)))


class RayRay(Arm):
    """
    The shipping probe in both roles. Verdicts go through the REAL
    `LRController.on_calibration`, so this arm carries the production
    `ratio**eta` damping, the abstention policy and the recorded ceiling.
    """

    #: The probe pays for paired sub-batches; that IS its cost and its noise
    #: robustness. Larger than the train batch, as in production.
    PROBE_BATCH = 512

    def __init__(self, lr, period=100):
        # THE PERIOD IS IN THE NAME. Results are keyed by arm name, so two ray
        # arms at different cadences silently MERGED into one row -- the board
        # printed two identical 'ray+ray' lines and one config's numbers were
        # simply gone.
        super().__init__(f'ray+ray p={period}')
        self.lr = float(lr)
        self.period = int(period)

    def args_overrides(self):
        return {'lr_policy': self.lr, 'lr_fused': self.lr,
                'lr_servo_managed': ('lr_policy', 'lr_fused'),
                'ray_calibration.enabled': True,
                'ray_calibration.period': self.period}

    def reset(self, run):
        # Counted so a probe that never resolves cannot masquerade as a
        # controller that chose not to act. The first version of this arm armed
        # AFTER the optimizer step, so every one of 1900 readings came back None
        # and the arm scored bit-identical to `null` -- which is what caught it.
        self.readings = {'armed': 0, 'none': 0}
        self._armed = False

    def on_rewind(self, run):
        # The probe armed against parameters that have just been replaced;
        # measuring now would difference across the restore and read the rewind
        # as this step's curvature.
        self._armed = False

    def pre_step(self, run):
        self._armed = run.m.ray_cal.arm(run.m.step_ind)
        if self._armed:
            self.readings['armed'] += 1

    def tick(self, run, loss, g_before, batch):
        if not self._armed:
            return
        self._armed = False
        m = run.m
        reading = m.ray_cal.measure(
            lambda: run.game.draw(self.PROBE_BATCH), run.game.probe_loss)
        if reading is None:
            self.readings['none'] += 1
            return
        st = reading.get('status', '?')
        self.readings[st] = self.readings.get(st, 0) + 1
        m.lr_controller.on_calibration(reading)


class RampPlateau(Arm):
    """
    Blind ramp up, ReduceLROnPlateau down. No sensor to corrupt, and no way to
    know it has gone too far except by the loss getting worse.

    `PLATEAU` mirrors `protocol.py::_parse_lr_sensor`'s defaults and train.py's
    cadence: one check per 10 train steps.
    """

    PLATEAU = dict(factor=0.5, patience=30, cooldown=10, check_every=10,
                   ema_period=25.0)
    #: what the servo applies on an unresolved reading:
    #: (grid_top / alpha_target) ** eta_up = (32/4) ** 0.25
    RAMP_PER_FIRING = 1.6818

    def __init__(self, lr, period=100):
        super().__init__('ramp+plateau')
        self.lr = float(lr)
        self.period = int(period)

    def args_overrides(self):
        return {'lr_policy': self.lr, 'lr_fused': self.lr,
                'lr_servo_managed': ('lr_policy', 'lr_fused'),
                'ray_calibration.enabled': False}

    def reset(self, run):
        self._ema = None
        self._best = math.inf
        self._bad = 0
        self._cool = 0

    def on_rewind(self, run):
        # `_best` is a legitimate watermark and survives -- the run really did
        # reach that loss, and the rewind restored the state that did it. The
        # EMA does not: it is averaging the detonation, so the next comparison
        # against `_best` would count patience against a loss that has been
        # undone.
        self._ema = None
        self._bad = 0

    def tick(self, run, loss, g_before, batch):
        m = run.m
        if m.step_ind and m.step_ind % self.period == 0:
            self._scale_peak(run, self.RAMP_PER_FIRING)

        if loss is None or not math.isfinite(loss):
            return
        # EMA on the PER-STEP clock. The old harness used the 10-step cadence's
        # alpha but applied it every step, giving a 2.5-step horizon where 25 was
        # intended.
        a = 1.0 - math.exp(-1.0 / self.PLATEAU['ema_period'])
        self._ema = loss if self._ema is None else (1 - a) * self._ema + a * loss

        if m.step_ind % self.PLATEAU['check_every']:
            return
        if self._cool > 0:
            self._cool -= 1
            return
        if self._ema < self._best:
            self._best, self._bad = self._ema, 0
            return
        self._bad += 1
        if self._bad >= self.PLATEAU['patience']:
            m.lr_controller.on_plateau(True, self.PLATEAU['factor'])
            self._bad = 0
            self._cool = self.PLATEAU['cooldown']


def standard_set(seed_lr, ladder):
    """The arms for a cell: the three controllers, the fixed ladder, and null."""
    return ([Hyper(seed_lr), RayRay(seed_lr), RampPlateau(seed_lr),
             Null(seed_lr)] + [Fixed(x) for x in ladder])
