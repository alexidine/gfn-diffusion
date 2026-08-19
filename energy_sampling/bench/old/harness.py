"""
The driver loop. Mirrors train.py's step body ORDER EXACTLY, because the order is
where the bugs live.

Two of the four control-logic failures this bench exists to catch were ordering
or bookkeeping bugs, not policy bugs:

  * growing the batch BEFORE appending the timing paired a new batch with the old
    rung's measurements (append train.py:1943, growth train.py:1966-1967);
  * handle_train_epoch_error CLEARS the timing deques, and the outer loop then
    appends the OOM'd step anyway -- so the first entry after every OOM pairs the
    pre-cut batch size with the time it took to fail (clear train.py:3662, in
    handle_train_epoch_error at :3605, vs the append at :1943).

A harness that "does roughly what train.py does" would reproduce neither. So the
sequence below is transcribed from the step body at train.py:1917-1972 and from
train_step (def at :2225), with line references.

RE-CHECK THE LINE NUMBERS WHENEVER EITHER FILE MOVES -- this note used to say so
and was itself the thing that rotted. Every reference in bench/ was ~250-350
lines stale as of 2026-08-13 (the step body had been cited as 1668-1710,
train_step as 2018-2047) and they pointed at unrelated code, which is worse than
no reference: a wrong line number reads as a verified one. They are corrected
throughout, but nothing enforces them.

VIRTUAL TIME. Nothing sleeps. Step time is a number produced by SyntheticGPU and
appended to the same deques the real loop appends to -- which is all
increment_batch_size ever reads. A 5000-step run takes about a second.
"""

import copy
import math

import torch

from bench.old.clock import SyntheticGPU
from bench.fake_modeller import FakeModeller, FakeStage, attach_real_batch_sizer, make_args
from bench.surfaces import GAMES

from energy_sampling.controller import LRController
from energy_sampling.ray_calibration import RayCalibration


class BenchRun:
    """
    One synthetic training run: a game, a synthetic GPU, and the real controllers.

    game_kwargs / gpu_kwargs / args_overrides are passed straight through, so a
    test reads as a statement about a configuration rather than about plumbing.
    """

    #: Sensor arms. Every arm drives the SAME actuator (LRController) with the
    #: same warmup, bounds, ceiling, tripwire and rewind -- only the source of
    #: the verdict changes. That is what makes them comparable.
    #:
    #:   ray            the alpha probe (shipping behaviour)
    #:   plateau        watch the loss, cut when nothing improves for `patience`
    #:                  checks. train.py's other implemented sensor.
    #:   ramp           NO sensor: raise by a constant every period, brake only
    #:                  on the divergence tripwire. This is the null hypothesis --
    #:                  72-82% of real probe readings come back at a grid edge,
    #:                  where the servo already applies exactly this constant, so
    #:                  `ramp` is what the probe degenerates to when unresolved.
    #:   ramp_plateau   climb blindly, brake on evidence
    #:   none           nothing moves the rate (the floor of the comparison)
    SENSORS = ('ray', 'plateau', 'ramp', 'ramp_plateau', 'none')

    #: THE DECOMPOSITION. Raising and lowering the rate are separate jobs with
    #: separate evidence requirements, and F-020 measured that the best mechanism
    #: for each differs by surface -- so they are independent switches, not one
    #: monolithic "sensor".
    #:
    #:   climber  none  never raises
    #:            ramp  a constant raise per period, no evidence at all
    #:            ray   the alpha probe, but ONLY when it would raise
    #:
    #:   braker   none  the divergence tripwire only (always on, in both cases)
    #:            plateau  cut when the smoothed loss stops improving
    #:            ray   the alpha probe, but ONLY when it would cut
    #:
    #: The probe runs if EITHER role asks for it, so a split arm pays its cost
    #: once. The tripwire and the rewind are outside this entirely -- they are
    #: not a sensor, they are the last resort.
    #: `slope` and `slope_seek` do not exist in train.py -- LR_SENSOR_KINDS is
    #: ('ray', 'plateau', 'none'). They are candidates being evaluated here
    #: before anyone writes them into the trainer.
    #:
    #:   slope       brake on the OBSERVED PROGRESS RATE rather than on
    #:               best-ever tracking. Fires when the smoothed loss stops
    #:               falling, without plateau's ratchet (one lucky sample sets
    #:               plateau's best and then nothing can beat it).
    #:   slope_seek  a whole controller: perturb the rate, keep the direction
    #:               that improved the progress rate, flip when it did not.
    #:               Handles BOTH directions itself, so it is declared as a
    #:               climber and its braker slot is a safety net, not a partner.
    #:
    #: THE PUBLISHED SINGLE-LOSS METHODS. Everything above was invented in this
    #: repo. These are the standard answers to "adapt one learning rate against
    #: one loss", implemented at their PAPER DEFAULTS so the comparison is not a
    #: contest between a tuned local idea and an untuned foreign one. All five
    #: move the rate in both directions, so all five are climbers whose braker
    #: slot is a safety net rather than a partner.
    #:
    #:   armijo  stochastic line search (Vaswani et al. 2019). The direct
    #:           like-for-like against `ray`: same clock, same probe batch, one
    #:           FEWER forward pass, but tests SUFFICIENT DECREASE
    #:           (L(theta+d) <= L(theta) + c*g.d) instead of alpha*.
    #:   bb      Barzilai-Borwein. Two consecutive gradients give
    #:           alpha* = -g.s/(s.y) -- the SAME quantity the ray probe measures,
    #:           for ZERO extra forward passes. Fed to the controller as a
    #:           synthetic bracketed reading, so only the sensor differs.
    #:   hyper   hypergradient descent (Baydin et al. 2018): lr *= exp(beta *
    #:           cos angle between consecutive gradients). One dot product.
    #:   dog     DoG (Ivgi et al. 2023), the parameter-free family's
    #:           representative: lr = max||theta - theta_0|| / sqrt(sum ||g||^2).
    #:           No target, no L*, nothing to tune.
    #:   sps     stochastic Polyak (Loizou et al. 2021), lr = (L - L*)/(c||g||^2),
    #:           with L* estimated by a running minimum because the true floor is
    #:           unknown here. Included to MEASURE that substitution, not because
    #:           it is expected to survive it.
    CLIMBERS = ('none', 'ramp', 'ray', 'slope_seek',
                'armijo', 'bb', 'hyper', 'hyperx', 'hyperz', 'hypern',
                'hyperm', 'hyperg', 'hyperp', 'dog', 'sps')
    BRAKERS = ('none', 'plateau', 'slope', 'ray')

    #: The monolithic names, kept because they are what the earlier findings were
    #: measured under, expressed in the decomposition.
    _SENSOR_PAIRS = {
        'ray': ('ray', 'ray'),
        'plateau': ('none', 'plateau'),
        'ramp': ('ramp', 'none'),
        'ramp_plateau': ('ramp', 'plateau'),
        'none': ('none', 'none'),
    }

    def __init__(self, game='mle', game_kwargs=None, gpu_kwargs=None,
                 args_overrides=None, stage=None, probe_batch=256,
                 need_batch_sizer=True, seed=0, reading_filter=None,
                 checkpoint_every=50, max_reloads_per_1k=0.2, sensor=None,
                 climber=None, braker=None, standard=None, probe_enabled=True):
        torch.manual_seed(seed)
        # per-instance copy so a sweep can vary the averaging windows without
        # mutating the class default under every other arm in the same process
        self.STANDARD = {**BenchRun.STANDARD, **(standard or {})}
        unknown = set(standard or ()) - set(BenchRun.STANDARD)
        if unknown:
            raise KeyError(f'unknown STANDARD keys {sorted(unknown)}')
        if sensor is not None:
            if sensor not in self.SENSORS:
                raise ValueError(f'sensor must be one of {self.SENSORS}, got {sensor!r}')
            climber, braker = self._SENSOR_PAIRS[sensor]
        climber = climber or 'ray'
        braker = braker or 'ray'
        if climber not in self.CLIMBERS or braker not in self.BRAKERS:
            raise ValueError(f'climber in {self.CLIMBERS}, braker in {self.BRAKERS}; '
                             f'got {climber!r}/{braker!r}')
        self.climber, self.braker = climber, braker
        self.sensor = sensor or f'{climber}+{braker}'
        # reading_filter(reading) -> reading | None, applied between the sensor
        # and the controller. This is how a candidate CONTROL POLICY is A/B'd
        # without editing controller.py: returning None suppresses the move, so
        # e.g. `lambda r: None if r['status'] == 'above_range' else r` tests
        # "a saturated reading may not raise". Keeping the intervention outside
        # the shipping module means the baseline arm is always the real code.
        self.reading_filter = reading_filter

        self.game = GAMES[game](**(game_kwargs or {}))
        self.gpu = SyntheticGPU(**(gpu_kwargs or {}))
        self.args = make_args(**(args_overrides or {}))
        self.probe_batch = int(probe_batch)

        self.need_batch_sizer = bool(need_batch_sizer)
        if need_batch_sizer:
            attach_real_batch_sizer(FakeModeller)

        self.m = FakeModeller(self.args, self.game.optimizers,
                              stage=stage or FakeStage())

        # THE BLOCK MOVED UNDER adaptive_lr and its `enabled` flag is DELETED
        # (utils._RETIRED_KEYS, both spellings): production derives the switch
        # from the stages that declare `lr_sensor: {kind: ray}`, so a second flag
        # cannot disagree with them.
        #
        # This harness has no stages -- its arm IS the climber/braker pair, which
        # is what the probe gate in step() reads ("train.py:2304" there) -- so
        # there is nothing here to derive from, and the off-switch the batch
        # cells need becomes a HARNESS argument rather than a config key. It is
        # `enabled` under a name that cannot be mistaken for config, at the same
        # default (True) and the same call sites. bench/runner.py, which does
        # carry a stage, derives it the production way.
        rc = self.args.adaptive_lr.ray_calibration
        self.m.ray_cal = RayCalibration(
            self.game.policy_params, alphas=tuple(float(a) for a in rc.alphas),
            n_sub=rc.n_sub, period=rc.period, enabled=bool(probe_enabled))
        self.m.lr_controller = LRController(self.m)

        self.history = []
        self.calibrations = []
        self.divergences = 0
        self.oom_steps = 0
        self._last_probe_alpha_true = None

        # DIVERGENCE RESPONSE = REWIND + PEAK CUT, not peak cut alone.
        # train.py's fire_loss_spike (def at train.py:2109) reloads the running
        # checkpoint (`load_model_only`, train.py:2169) and only then calls
        # on_divergence.
        # A harness that cut the peak without restoring the weights was testing
        # half the response -- and on a surface where the blow-up drives
        # parameters non-finite, the omitted half is the half that recovers:
        # check_spike then fires on every subsequent step forever, which reads as
        # a spectacular controller failure and is actually a missing feature here.
        self.checkpoint_every = int(checkpoint_every)
        self.max_reloads_per_1k = float(max_reloads_per_1k)
        self.total_reloads = 0
        self.aborted = None
        self._ckpt = None
        self._loss_ema = None
        self._plateau = {}
        self._slope_window = []
        self._slope_stale = 0
        self._seek_prev = None
        self._seek_dir = 1          # start by climbing; the seed LR is always low
        self._bb_prev = None
        self._bb_window = []
        self._armijo_margins = []
        self._hyper_prev = None
        self._hyper_absmean = None
        self._hyper_v = 0.0
        self._hyper_gmean = None
        self._hyper_cbar = None
        self._dog_theta0 = None
        self._dog_r = None
        self._dog_gsq = 0.0
        self._sps_ema = None
        self._sps_floor = None

    # ------------------------------------------------------------ other arms

    #: train.py's plateau defaults (protocol.py::_parse_lr_sensor) and its
    #: cadence: "checks, not train steps -- one check per 10 train steps".
    PLATEAU = dict(factor=0.5, patience=30, cooldown=10, threshold=0.0,
                   check_every=10, ema_period=25.0)

    #: Loss-slope sensors. `window` matches the calibration period so every arm
    #: acts on the same clock -- F-021's whole point is that mismatched clocks,
    #: not weak sensors, are what breaks a pair.
    SLOPE = dict(window=50, patience=2, factor=0.5, seek_factor=1.68,
                 dead_zone=0.01, ema_period=25.0)

    def _plateau_tick(self, loss):
        """
        train.py's ReduceLROnPlateau, transcribed (`update_lr_plateau`,
        train.py:4186-4258).

        Tracks a best-ever per metric and cuts when nothing has improved for
        `patience` checks. Two details are load-bearing and both are copied
        rather than reinvented:

          * it reads the SMOOTHED value. Tracking a best-ever over raw samples
            ratchets the best down to the luckiest noise draw, after which
            nothing ever beats it and the sensor cuts a still-improving run --
            observed on lrs_blowup.
          * the improvement bar is ABSOLUTE, because these channels are
            unbounded below and a fractional bar has nothing to be relative to.
        """
        cfg = self.PLATEAU
        if loss is None or not math.isfinite(loss):
            return
        a = 1.0 - math.exp(-cfg['check_every'] / cfg['ema_period'])
        self._loss_ema = loss if self._loss_ema is None else \
            (1 - a) * self._loss_ema + a * loss
        if self.m.step_ind % cfg['check_every']:
            return

        gs = self._plateau
        val, prev = self._loss_ema, gs.get('best')
        improved = prev is None or val < prev - cfg['threshold']
        if prev is None or val < prev:
            gs['best'] = val

        if gs.get('cooldown', 0) > 0:
            gs['cooldown'] -= 1
            gs['stale'] = 0
        elif improved:
            gs['stale'] = 0
        else:
            gs['stale'] = gs.get('stale', 0) + 1

        fired = gs['stale'] >= cfg['patience']
        if fired:
            gs['stale'] = 0
            gs['cooldown'] = cfg['cooldown']
        self.m.lr_controller.on_plateau(fired, cfg['factor'])

    def _probe_role_allows(self, reading):
        """
        Does the role that asked for the probe want THIS reading?

        A reading raises when alpha* is above target and cuts when it is below,
        so splitting the probe into a climber and a braker is a sign test on that
        ratio -- nothing about the measurement changes, only whether it is acted
        on. That is what makes `ray`-to-climb + `plateau`-to-brake expressible
        without inventing a new sensor.
        """
        alpha = reading.get('alpha_star')
        if not (isinstance(alpha, float) and math.isfinite(alpha) and alpha > 0):
            return False
        target = float(getattr(getattr(getattr(self.args, 'adaptive_lr', None),
                                       'calibration', None), 'alpha_target', 4.0))
        return self.climber == 'ray' if alpha > target else self.braker == 'ray'

    # ---- loss-slope sensors (candidates; not in train.py's LR_SENSOR_KINDS)

    def _scale_peak(self, mult):
        """
        Multiply peak_scale, respecting bounds and the divergence ceiling.

        The actuator path a NEW sensor kind would have to use. LRController
        exposes on_calibration / on_plateau / on_divergence, each of which
        applies its own exponent, so a sensor that wants a plain symmetric
        multiplier has to do the clamping itself -- which is worth knowing before
        anyone adds one to train.py.
        """
        ctrl = self.m.lr_controller
        if ctrl.in_warmup():
            return 0.0          # every sensor is held through warmup
        st = ctrl._state()
        lo, hi = ctrl._peak_bounds()
        ceiling = ctrl._current_ceiling()
        if ceiling is not None:
            hi = min(hi, ceiling)
        before = float(st['peak_scale'])
        st['peak_scale'] = max(lo, min(hi, before * float(mult)))
        ctrl._apply_lrs(st)
        return st['peak_scale'] / before if before else 1.0

    def _progress_rate(self):
        """
        Scale-free rate of improvement over the last window of smoothed losses:
        the second half's mean minus the first half's, divided by the window's
        own magnitude. Negative = improving.

        Scale-free because it is compared ACROSS learning rates and across
        surfaces whose losses differ by orders of magnitude, and differenced over
        halves rather than fitted, because a least-squares slope on an EMA has
        badly understated standard errors (the same autocorrelation argument
        train.py makes for why a best-tracking test wants smoothed input and a
        slope test does not).
        """
        w = self._slope_window
        if len(w) < 4:
            return None
        half = len(w) // 2
        first = sum(w[:half]) / half
        second = sum(w[half:]) / (len(w) - half)
        scale = abs(sum(w) / len(w)) + 1e-12
        return (second - first) / scale

    def _slope_brake_tick(self):
        """Cut when the loss has stopped falling for `patience` consecutive windows."""
        cfg = self.SLOPE
        if self.m.step_ind == 0 or self.m.step_ind % cfg['window']:
            return
        rate = self._progress_rate()
        self._slope_window = []
        if rate is None:
            return
        if rate > -cfg['dead_zone']:
            self._slope_stale += 1
        else:
            self._slope_stale = 0
        if self._slope_stale >= cfg['patience']:
            self._slope_stale = 0
            self._scale_peak(cfg['factor'])

    def _slope_seek_tick(self):
        """
        Derivative-free hill-climb on the learning rate, using observed progress
        as the objective.

        Every window: move the rate one notch in the current direction, measure
        the progress rate, and compare with the previous window. Improved -> keep
        going the same way; worse -> turn around. That is the whole rule.

        WHY THIS SHAPE. F-021 found that a controller is stable when ONE
        measurement on ONE clock moves the rate in BOTH directions, and unstable
        when a fast climber is paired with a slow brake. This has that property
        by construction, and unlike the ray probe it optimises the thing actually
        wanted -- progress per step at the rate being run -- rather than a
        one-step surrogate that cannot see multi-step effects.
        """
        cfg = self.SLOPE
        if self.m.step_ind == 0 or self.m.step_ind % cfg['window']:
            return
        rate = self._progress_rate()
        self._slope_window = []
        if rate is None:
            return
        prev = self._seek_prev
        self._seek_prev = rate
        if prev is not None and rate > prev + cfg['dead_zone']:
            self._seek_dir *= -1        # got worse -> turn around
        step = cfg['seek_factor'] ** self._seek_dir
        self._scale_peak(step)

    def _ramp_tick(self):
        """
        The null hypothesis: raise by a constant every calibration period, with
        no sensor at all.

        Implemented by handing the REAL actuator a synthetic `above_range`
        reading, rather than by writing peak_scale directly, so this arm goes
        through the same warmup hold, bounds, ceiling and asymmetric update as
        every other arm. The multiplier it produces -- (grid_top/target)^eta_up
        -- is exactly what the shipping servo applies whenever the probe fails
        to resolve, which is most of the time.
        """
        period = self.args.adaptive_lr.ray_calibration.period
        if self.m.step_ind == 0 or self.m.step_ind % period:
            return
        top = max(a for a in self.args.adaptive_lr.ray_calibration.alphas
                  if 2 * a in set(self.args.adaptive_lr.ray_calibration.alphas))
        self.m.lr_controller.on_calibration(
            {'status': 'above_range', 'alpha_star': float(top), 'lo': float(top), 'hi': None})

    # ------------------------------------------- published single-loss methods

    #: Paper defaults, and where they are not stated, the most obvious choice.
    #: NONE of these were tuned on this bench -- the whole point of importing a
    #: published method is to see what it does before anyone fits it to us.
    #:
    #: `armijo_c` 0.1 and backtrack 0.5 are Vaswani et al.'s; their growth is
    #: gamma^(b/n) per step with gamma=2, which for our batch/dataset ratio is
    #: within rounding of 2^(1/50) applied once per calibration period.
    #: `hyper_beta` is the normalized-hypergradient setting from Baydin et al.'s
    #: Adam experiments. `dog_eps` is their r_epsilon coefficient. `sps_c` 0.5 is
    #: Loizou et al.'s and `sps_max` is their bound on the step.
    #: `armijo_window` and `bb_window` are THE AVERAGING KNOBS, and they are the
    #: structural difference between these arms and the ray probe. The probe
    #: averages its statistic over `n_sub` paired sub-batches and applies a
    #: significance test before acting, which is why it degrades gracefully under
    #: gradient noise where these two collapse. At window=1 each acts on a single
    #: noisy sample, which is the published form; raising it buys the same defence
    #: the probe has. Swept rather than chosen -- see `bench/averaging.py`.
    STANDARD = dict(
        armijo_c=0.1, armijo_backtrack=0.5, armijo_grow=2.0 ** (1.0 / 50.0),
        armijo_window=1,
        bb_window=20, bb_min_curv=1e-12,
        hyper_beta=0.02,
        # `hyperx` only. beta_down is the HOUSE RATIO, not a new constant:
        # MK_DEV_CALIBRATION ships eta_up 0.25 / eta_down 0.5, and the config
        # argues that 2:1 asymmetry is on principle (raising is licensed by a
        # one-step measurement that cannot see multi-step effects; lowering is
        # not). hyper as published has one gain for both directions, which is
        # what makes its brake rate-limited.
        #
        # TESTED AND NOT RECOMMENDED. The three hyper variants now tie at 0.0%
        # over budget (summary section 0), so the asymmetry buys nothing, and
        # under noise it is actively harmful: cos is symmetric about a small
        # positive mean, so a LARGER beta_down drives the rate DOWN and it cannot
        # climb back out -- `.02/.08` scores 50% at dim2048/noise2 with
        # cold_start at 100%, where the symmetric published rule scores 0%
        # (section 0b(a)). The argument above is about the COST of overshoot and
        # is silent on the DRIFT of a noisy statistic, which is the term that
        # decides it. Kept live because it is the mechanism the `hyper 2:1` arm
        # measures.
        #
        # THIS DEFAULT IS A TRAP FOR NEW ARMS: 0.04 against hyper_beta 0.02 means
        # a `hyperx` arm declared with NO standard-override runs 2:1, not the
        # published rule. Section 1's "`hyper` (published)" row was exactly that
        # mistake. Declare both betas explicitly, as crucible.ARMS now does.
        hyper_beta_down=0.04,
        # A CONSTANT CLIMB UNDER THE HYPERGRADIENT: lr *= exp(beta*cos + bias).
        #
        # hyper's gain is its own signal-to-noise ratio -- E[cos] =
        # ||gbar||^2 / (||gbar||^2 + tr Sigma) -- so under noise the statistic
        # attenuates toward 0 and exp(0)=1 is "do nothing". That is exactly the
        # graceful degradation that makes it safe, and it is also why it fails:
        # the identity is the wrong default when the rate is 35x too cold. Its
        # crucible failures are ONLY cold_start and hot_90pct, never drift or
        # regime change -- i.e. only where the rate has far to travel.
        #
        # The bias is what it degrades TO when the measurement says nothing,
        # which is the ray probe's architecture (abstain -> apply the constant)
        # written into a free sensor.
        #
        # ITS PROVENANCE IS HALF RIGHT. The claim was "0.0104/step is not chosen:
        # it is the blind ramp's own rate, ln(1.682)/50". The per-FIRING
        # multiplier is real -- on an unresolved reading the servo applies
        # (grid_top/alpha_target)^eta_up = (32/4)^0.25 = 1.682 (see `_ramp_tick`)
        # and ln(1.682) = 0.5199. The /50 is not the servo's clock: it is the
        # bench's own `ray_calibration.period` override (oracle.py:77), while
        # MK_DEV_RAYCAL ships 500. At the shipping period the rate is 0.00104, so
        # the constant is 10x its stated source and IS a chosen number.
        #
        # TESTED AND NOT RECOMMENDED. `hyper gated`, the arm this exists for, now
        # ties both unbiased variants at 0.0% (summary section 0). Default 0.0 --
        # an arm that wants it passes it explicitly.
        hyper_bias=0.0,
        # GATE THE BIAS BY THE SENSOR'S OWN CONFIDENCE: bias * (1 - |cos|).
        #
        # A CONSTANT bias is right where the measurement is worthless and wrong
        # where it is not. `hyper+ramp d8` reaches 4.8% over budget but its only
        # remaining failure is `mle q1e-2` (29%) -- quartic at noise 0.01, where
        # cos ~ 1, the climb needs no help, and the constant ramp is a headwind
        # the doubled brake has to fight.
        #
        # |cos| is not a proxy for confidence, it IS one: E[cos] =
        # ||gbar||^2/(||gbar||^2 + tr Sigma) is the signal-to-noise ratio. So
        # (1 - |cos|) is the fraction of the reading that is noise, and gating on
        # it makes the arm ramp exactly to the extent the measurement has nothing
        # to say -- the ray probe's abstain-and-apply-the-constant, with a
        # continuous gate instead of a significance threshold. Continuous
        # deliberately: a THRESHOLD on a noisy statistic is what killed four arms
        # this session.
        #
        # TESTED AND NOT RECOMMENDED, and the argument above is where it went
        # wrong. `mle q1e-2` is a cell the gate was DESIGNED against, so 29% ->
        # 0% there is a fit, not a result. On the corrected gradual
        # `regime_change` the gate is a headwind rather than a help (95% on that
        # column at cond=30, summary section 0a), and with the budget defect
        # fixed all three hyper variants tie at 0.0% -- so there is nothing left
        # for it to buy. The reasoning survives as reasoning; it just describes a
        # mechanism whose benefit was an artifact of a violent step change that
        # does not happen.
        hyper_bias_gate=False,
        # ORPHANS, KEPT AS A RECORD: the two paragraphs that used to sit here
        # documented `hyper_rho_target` (2 as the stability edge against 1 as the
        # one-step optimum, a 2x margin the shape of alpha_target's) and
        # `hyper_rho_ema` (10 steps for rho to power-iterate onto the dominant
        # eigenvalue). BOTH KEYS ARE GONE with the rho branch -- see the note at
        # the end of `_hyper_tick`. They read as live knobs for as long as they
        # sat above unrelated entries, which is the failure they were deleted for.
        # `hypern`: horizon for the running scale of |cos|. A SMOOTHING choice,
        # not a gain -- it sets how fast the normaliser tracks a change in the
        # noise level, and the response is bounded whatever it is.
        hyper_scale_ema=50.0,
        # `hyperm`: momentum on the log-LR velocity. mu 0.9 caps the steady-state
        # move at beta/(1-mu) = 10x the single-step rate, so the bound is implied
        # by mu rather than being a separate cap.
        hyper_mu=0.9,
        # `hyperg`: horizon for the running gradient-norm level the gate compares
        # against.
        hyper_gnorm_ema=100.0,
        # `hyperp`: PERSISTENT-gradient gain. tau sets how much noise the average
        # removes (variance down ~2*tau), which is what licenses k: gain up to
        # ~sqrt(2*tau) is free in signal-to-noise terms, so k=4 at tau=50 is well
        # inside the budget rather than a fitted number.
        hyper_persist_tau=50.0,
        hyper_persist_k=4.0,
        dog_eps=1e-6,
        sps_c=0.5, sps_max=8.0, sps_ema=25.0,
    )

    def _flat(self, which='param'):
        """Policy parameters or their gradients, flattened into one vector."""
        out = []
        for p in self.game.policy_params:
            v = p.detach() if which == 'param' else (
                p.grad.detach() if p.grad is not None else torch.zeros_like(p))
            out.append(v.reshape(-1).clone())
        return torch.cat(out) if out else torch.zeros(0)

    def _set_peak_to(self, target_lr):
        """
        Drive the actuator to an ABSOLUTE learning rate.

        Methods that compute a rate outright (dog, sps) rather than a correction
        still go through `_scale_peak`, so they inherit the real controller's
        warmup hold, peak bounds and recorded ceiling. A method that wrote the
        rate directly would be competing against the others with the safety rails
        removed, which would flatter it for the wrong reason.
        """
        cur = self.m.lr_of(self.game.train_key)
        if not (math.isfinite(target_lr) and target_lr > 0 and cur > 0):
            return
        self._scale_peak(target_lr / cur)

    def _armijo_tick(self, theta_before, g_before, d):
        """
        Stochastic line search: accept the step if it bought a decrease at least
        `c` times the decrease the gradient PREDICTED, else shrink the rate.

        Runs EVERY STEP, which is what the paper specifies and what its growth
        constant is sized for. Gating it on the ray probe's 50-step clock instead
        -- the first thing tried here -- leaves it climbing at 2^(1/50) per
        period rather than per step, i.e. ~28x slower than the blind ramp, and it
        then scores 7552 on a cold start: indistinguishable from having no
        climber at all. That was a clock error in this harness, not a property of
        line search, and it is worth recording because it is the same failure
        mode F-021 describes from the other direction.

        THIS IS THE EXPENSIVE ARM, which is worth being explicit about because
        the opposite was assumed here first. The ray probe looks costly per
        firing -- 8 alphas x 8 paired sub-batches -- but that is 8 batch-forwards
        ONCE PER PERIOD, and mk_dev's period is 500, so it amortises to under 2%.
        Armijo pays 2 probe-batch forwards EVERY step, ~50% at these sizes. It is
        roughly 30x the probe's cost in production and ~3x in this bench, where
        the period is shortened to 50 so a 2000-step run gets enough firings.

        So the cost ordering across these arms is: bb / hyper / dog / sps free,
        ray cheap, armijo expensive. Any verdict has to be read against that.

        Backtracking within the step -- the form the paper states -- is not
        available to us: the step has been taken and re-taking it would mean a
        second optimizer update, which no trainer does. The multiplicative
        accept/shrink form here is the standard practical variant.
        """
        cfg = self.STANDARD
        predicted = float(torch.dot(g_before, d))     # g.d, negative on a descent step
        if not (math.isfinite(predicted) and predicted < 0):
            return
        batch = self._draw_probe()
        after = self._probe_loss(batch)               # already at theta_before + d
        flat = torch.cat([p.detach().reshape(-1) for p in self.game.policy_params])
        with torch.no_grad():
            self._write_flat(theta_before)
            before = self._probe_loss(batch)
            self._write_flat(flat)
        if not (math.isfinite(after) and math.isfinite(before)):
            return
        # sufficient decrease: L(theta+d) <= L(theta) + c * g.d, i.e. the
        # achieved decrease clears `c` times the predicted one. Kept as a signed
        # MARGIN rather than a bool so it can be averaged -- averaging the
        # statistic is what the probe does, averaging the decision would not fix
        # anything, because a coin flip on the decision stays a coin flip.
        margin = (before - after) + cfg['armijo_c'] * predicted
        w = max(int(cfg['armijo_window']), 1)
        self._armijo_margins.append(margin)
        if len(self._armijo_margins) < w:
            return
        mean = sum(self._armijo_margins) / len(self._armijo_margins)
        self._armijo_margins = []
        # THE GROWTH IS COMPENSATED FOR THE WINDOW. Acting once per w steps also
        # divides the climb rate by w, and without this the sweep measures "acts
        # less often" rather than "acts on better evidence" -- observed directly:
        # at noise 0.01, where averaging cannot possibly help, window 10 scored
        # 33% over budget against window 1's 0%, purely from the slower climb.
        #
        # The backtrack is NOT compensated, because 0.5**w is 1e-15 at w=50 and
        # would slam the floor on one decision. That asymmetry is itself the
        # finding: armijo's response is not scale-free in its decision rate, so
        # raising the window necessarily symmetrises it (at w=50 the up move is
        # 1.014**50 = 2.0 against a 0.5 down move, i.e. exactly symmetric), and
        # the symmetrisation is part of what a window buys here rather than a
        # confound to be removed.
        up = cfg['armijo_grow'] ** w
        self._scale_peak(up if mean >= 0 else cfg['armijo_backtrack'])

    @torch.no_grad()
    def _write_flat(self, vec):
        i = 0
        for p in self.game.policy_params:
            n = p.numel()
            p.copy_(vec[i:i + n].view_as(p))
            i += n

    def _bb_tick(self, theta_before, g_before):
        """
        Barzilai-Borwein as an alpha* estimator, at zero measurement cost.

        With s = theta_t - theta_{t-1} and y = g_t - g_{t-1} ~ H s, the multiple
        of the step that a quadratic model says was optimal is

            alpha* = -g_{t-1}.s / (s.y)

        which is the SAME number the ray probe brackets, from gradients the
        trainer computed anyway. So this arm differs from `ray` in exactly one
        respect -- how alpha* was obtained -- and any gap between them is the
        value of the probe's PAIRING, not of the alpha* idea.

        The catch is that g_t and g_{t-1} are computed on different batches, so y
        carries the full gradient noise while the signal H*s shrinks with the step
        size. Hence the median over a window; hence, presumably, why the probe
        pays for common random numbers.
        """
        cfg = self.STANDARD
        prev = self._bb_prev
        cur_theta, cur_grad = self._flat('param'), self._flat('grad')
        self._bb_prev = (theta_before.clone(), g_before.clone())
        if prev is None:
            return
        prev_theta, prev_grad = prev
        s = theta_before - prev_theta      # the step that WAS taken, t-1 -> t
        y = g_before - prev_grad           # gradient change across it
        sy = float(torch.dot(s, y))
        gs = float(torch.dot(prev_grad, s))
        if not (math.isfinite(sy) and math.isfinite(gs)) or sy <= cfg['bb_min_curv']:
            return                          # non-positive curvature: no estimate
        alpha = -gs / sy
        if not (math.isfinite(alpha) and alpha > 0):
            return
        # ROLLING window, not one cleared each period: the averaging length is
        # the knob under test and tying it to the calibration period would pin it
        # to 50 and hide the effect.
        self._bb_window.append(alpha)
        keep = max(int(cfg['bb_window']), 1)
        if len(self._bb_window) > keep:
            self._bb_window = self._bb_window[-keep:]
        period = self.args.adaptive_lr.ray_calibration.period
        if self.m.step_ind == 0 or self.m.step_ind % period:
            return
        if len(self._bb_window) < min(4, keep):
            return
        w = sorted(self._bb_window)
        med = w[len(w) // 2]
        # handed over in the probe's own vocabulary, so the SAME actuator code
        # runs and the arms differ only in where the number came from
        self.m.lr_controller.on_calibration(
            {'status': 'bracketed', 'alpha_star': float(med),
             'lo': float(med), 'hi': float(med)})

    def _hyper_tick(self, g_before):
        """
        Hypergradient descent (Baydin et al. 2018), normalized form.

        dL/d(lr) = -g_t . g_{t-1}: consecutive gradients agreeing means the rate
        was too small, disagreeing means it overshot. Normalizing to a cosine
        makes the update scale-free, which is what lets one beta serve every
        surface. One dot product, no extra evaluations at all.
        """
        cfg = self.STANDARD
        prev = self._hyper_prev
        self._hyper_prev = g_before.clone()
        if prev is None:
            return
        na, nb = float(g_before.norm()), float(prev.norm())
        if not (na > 0 and nb > 0):
            return
        cos = float(torch.dot(g_before, prev)) / (na * nb)
        if not math.isfinite(cos):
            return
        if self.climber == 'hyperp':
            # AMPLIFY THE AVERAGED cos, NOT THE RAW ONE.
            #
            # `hyperm` accumulated raw beta*cos, so runs of same-sign noise
            # compounded -- amplification hit signal and noise alike (96% over
            # budget at dim2048 noise2). `hypern` divided the INSTANTANEOUS cos by
            # a running scale, with the same defect plus loss of the proportional
            # feedback. Smoothing FIRST separates them: an EMA cuts the noise
            # variance by ~2*tau while leaving E[cos] -- the signal -- untouched,
            # so gain on the smoothed value buys speed without buying noise.
            #
            # At low noise cbar ~ 1, the clip binds and this IS plain hyper. At
            # noise 2, cbar ~ 0.29 and k*cbar ~ 1.16 clips to 1, restoring the
            # full climb with no bias term and nothing unconditional. Once cbar
            # falls below 1/k the response is proportional again, so the
            # approach-to-optimum brake that normalising destroyed is preserved.
            a = 1.0 - math.exp(-1.0 / cfg['hyper_persist_tau'])
            self._hyper_cbar = cos if self._hyper_cbar is None else                 (1 - a) * self._hyper_cbar + a * cos
            u = max(-1.0, min(1.0, cfg['hyper_persist_k'] * self._hyper_cbar))
            beta = cfg['hyper_beta'] if u > 0 else cfg['hyper_beta_down']
            self._scale_peak(math.exp(beta * u))
            return
        if self.climber == 'hyperm':
            # RPROP-STYLE MOMENTUM: speed from PERSISTENCE, not from a constant.
            #
            # `hyperx`'s bias is unconditionally upward; normalising (hyperz,
            # hypern) destroys the proportional feedback -- cos falling as the
            # rate approaches correct IS the brake, and dividing it out makes the
            # rule climb straight through. Momentum leaves the per-step response
            # exactly as published and only lets AGREEMENT COMPOUND: a sustained
            # one-sided cos (init, or a regime change) accelerates, while in
            # steady state cos oscillates, nothing accumulates and the rate sits.
            #
            # The accumulation is DROPPED, not reversed, when new evidence
            # opposes it, so a spurious reset costs "stop accelerating" -- the
            # identity -- rather than an asymmetric cut. That is what separates
            # this from the four threshold-plus-asymmetric-response arms that
            # collapsed. Sign-symmetric, so nothing drifts under pure noise.
            step = (cfg['hyper_beta'] if cos > 0 else cfg['hyper_beta_down']) * cos
            if step * self._hyper_v < 0:
                self._hyper_v = step        # direction changed: drop the run-up
            else:
                self._hyper_v = cfg['hyper_mu'] * self._hyper_v + step
            self._scale_peak(math.exp(self._hyper_v))
            return
        if self.climber == 'hyperg':
            # GRADIENT-NORM GATED BIAS. The ambiguity `hyperx` cannot resolve is
            # that cos ~ 0 means either "noise drowned the signal" (ramp) or
            # "the rate is right" (do not). ||g|| separates them: a regime change
            # RAISES it, convergence lowers it. So bias only while the norm is
            # above its own recent level -- an off switch that is not the
            # divergence tripwire. Continuous and clipped, never a threshold.
            gn = float(g_before.norm())
            a = 1.0 - math.exp(-1.0 / cfg['hyper_gnorm_ema'])
            self._hyper_gmean = gn if self._hyper_gmean is None else                 (1 - a) * self._hyper_gmean + a * gn
            ratio = gn / max(self._hyper_gmean, 1e-12)
            gate = max(0.0, min(1.0, ratio - 1.0))
            beta = cfg['hyper_beta'] if cos > 0 else cfg['hyper_beta_down']
            self._scale_peak(math.exp(beta * cos + cfg['hyper_bias'] * gate))
            return
        if self.climber == 'hypern':
            # NORMALISE BY THE OBSERVED SCALE OF cos, not by its theoretical null.
            #
            # `hyperx`'s bias is unconditionally upward -- it survives only
            # because the tripwire bounds it, which is not a foundation. `hyperz`
            # divided by sqrt(2/(pi*d)), which at d=2048 is 0.0197, so typical
            # readings mapped to |z| ~ 51, tanh saturated every step and the rule
            # became bang-bang: 75% over budget, bimodal (median 0.48 with 75%
            # never converging). Normalising by a RUNNING MEAN of |cos| instead
            # puts a typical reading at ~1 by construction, so the response stays
            # proportional and saturates only on above-average agreement.
            #
            # Sign-symmetric, so there is nothing to drift on: under pure noise
            # cos is symmetric about 0 and the asymmetric beta makes the residual
            # drift DOWNWARD. Self-calibrating, so it needs no knowledge of d, of
            # the noise level, or of the attenuation -- and it recovers the climb
            # the bias was added for: at noise 2 a typical cos is 0.29, so u ~ 1
            # and the rate climbs at the full beta rather than 0.02*0.29.
            a = 1.0 - math.exp(-1.0 / cfg['hyper_scale_ema'])
            m = abs(cos)
            self._hyper_absmean = m if self._hyper_absmean is None else                 (1 - a) * self._hyper_absmean + a * m
            scale = max(self._hyper_absmean, 1e-6)
            u = max(-1.0, min(1.0, cos / scale))
            beta = cfg['hyper_beta'] if u > 0 else cfg['hyper_beta_down']
            self._scale_peak(math.exp(beta * u))
            return
        if self.climber == 'hyperz':
            # CONFIDENCE-SCALED, SIGN-SYMMETRIC. `cos` is not dimension-free:
            # between two independent vectors E|cos| ~ sqrt(2/(pi*d)), so the
            # response beta*cos shrinks like 1/sqrt(d) while a constant bias does
            # not -- which is why `hyperx`'s ramp bias drifts UP at every
            # realistic width (net +0.0047/step at d=32, +0.0103 at d=1e5, i.e.
            # it degenerates into a blind ramp).
            #
            # Dividing by the null level makes it a z-score: how many noise units
            # of agreement. tanh keeps it bounded (so it still degrades toward
            # doing nothing, never toward a confident wrong answer) and saturates
            # once the reading is unambiguous, so a CONFIDENT move gets full beta
            # in EITHER direction regardless of width or noise. No bias term, so
            # nothing to drift on: under pure noise z is symmetric about 0 and the
            # asymmetric beta makes the residual drift DOWNWARD, the safe way.
            d = max(int(g_before.numel()), 2)
            z = cos / math.sqrt(2.0 / (math.pi * d))
            beta = cfg['hyper_beta'] if z > 0 else cfg['hyper_beta_down']
            self._scale_peak(math.exp(beta * math.tanh(z)))
            return
        if self.climber != 'hyperx':
            self._scale_peak(math.exp(cfg['hyper_beta'] * cos))
            return

        # ---- hyperx: ASYMMETRIC GAIN ONLY ----------------------------------
        #
        # Being 2x cold costs ~2x the steps. Being 2x hot past the cliff costs a
        # divergence, a rewind, and a ceiling cleared only at a stage transition
        # -- a permanent penalty against a temporary one. So a symmetric gain is
        # wrong regardless of beta, and beta_down > beta_up follows. This is the
        # house rule already: MK_DEV_CALIBRATION ships eta_up 0.25 / eta_down 0.5.
        #
        # THE ARGUMENT IS ABOUT COSTS AND THE MEASUREMENT IS ABOUT DRIFT, which
        # is why it loses. Under noise cos is symmetric about a small positive
        # mean, so beta_down > beta_up biases E[log lr] DOWNWARD and the rate
        # cannot climb back out: `.02/.08` scores 50% at dim2048/noise2 with
        # cold_start at 100% where the published symmetric rule scores 0%
        # (summary section 0b(a)). With the budget defect fixed, symmetric, 2:1
        # and gated all tie at 0.0% (section 0). TESTED AND NOT RECOMMENDED --
        # the branch stays because `hyper 2:1` and `hyper gated` are the arms
        # that measure it, and a mechanism with a recorded verdict is worth more
        # than a deletion.
        #
        # Note the default: STANDARD ships hyper_beta_down 0.04 against
        # hyper_beta 0.02, so `hyperx` WITHOUT an explicit override is the 2:1
        # arm. Plain `hyper` (the branch above) is the symmetric published rule.
        beta = cfg['hyper_beta'] if cos > 0 else cfg['hyper_beta_down']
        bias = cfg['hyper_bias']
        if cfg['hyper_bias_gate']:
            bias *= (1.0 - min(abs(cos), 1.0))
        self._scale_peak(math.exp(beta * cos + bias))
        return

        # The unbounded rho branch that used to live here is DELETED, not
        # left below this return: it fired on coin flips (threshold at
        # rho > 1, the centre of that statistic's own noise; cold start
        # 0% -> 100% at noise 2), and an adversarial review found it as
        # ~55 lines of live but unreachable code with `hyper_s` and
        # `hyper_rho_ema` reading as active knobs. In a repo whose
        # documented failure mode is "inert flags fail silently", that is
        # the shape of the next bug.

    def _dog_tick(self, g_before):
        """
        DoG (Ivgi et al. 2023) -- the parameter-free family's representative.

            lr_t = max_i ||theta_i - theta_0|| / sqrt(sum_i ||g_i||^2)

        No target, no L*, no base rate: the numerator is how far the run has
        already travelled and the denominator is how much gradient it took. It is
        the only arm here with NOTHING to tune, which is the entire reason to
        care about it.

        Off-label warning: DoG is derived for SGD and we run Adam, whose update
        is not lr*g. Prodigy is the variant that fixes this; the point of running
        plain DoG is to see how much that mismatch actually costs.
        """
        cfg = self.STANDARD
        theta = self._flat('param')
        if self._dog_theta0 is None:
            self._dog_theta0 = theta.clone()
            self._dog_r = cfg['dog_eps'] * (1.0 + float(theta.norm()))
            self._dog_gsq = 0.0
        self._dog_r = max(self._dog_r, float((theta - self._dog_theta0).norm()))
        self._dog_gsq += float(g_before.pow(2).sum())
        if self._dog_gsq <= 0:
            return
        self._set_peak_to(self._dog_r / math.sqrt(self._dog_gsq))

    def _sps_tick(self, loss, g_before):
        """
        Stochastic Polyak step (Loizou et al. 2021): lr = (L - L*)/(c*||g||^2).

        L* IS NOT KNOWN HERE. The MLE surface has an irreducible noise floor and
        the others are worse, so the interpolation assumption L*=0 that the method
        is normally run under is simply false, and the step it produces is too
        large by (L_floor)/(c*||g||^2) forever.

        The substitution tested is the obvious one: estimate L* by the running
        minimum of the smoothed loss. It is biased LOW by construction (a running
        minimum of a noisy series sits under the mean it is drawn from), so the
        rate stays too hot, and the bias does not shrink as the run converges --
        it is the standing question this arm exists to answer quantitatively.
        """
        cfg = self.STANDARD
        if loss is None or not math.isfinite(loss):
            return
        a = 1.0 - math.exp(-1.0 / cfg['sps_ema'])
        self._sps_ema = loss if self._sps_ema is None else \
            (1 - a) * self._sps_ema + a * loss
        self._sps_floor = self._sps_ema if self._sps_floor is None \
            else min(self._sps_floor, self._sps_ema)
        gsq = float(g_before.pow(2).sum())
        if gsq <= 0:
            return
        gap = max(0.0, self._sps_ema - self._sps_floor)
        self._set_peak_to(min(gap / (cfg['sps_c'] * gsq), cfg['sps_max']))

    # -------------------------------------------------------------- checkpoint

    def _all_params(self):
        """Every optimizer-visible parameter, deduplicated by identity -- the
        games share `theta` across fwd/bwd/replay/fused exactly as train.py does."""
        seen, out = set(), []
        for opt in self.m.optimizers.values():
            for group in opt.param_groups:
                for p in group['params']:
                    if id(p) not in seen:
                        seen.add(id(p))
                        out.append(p)
        return out

    def _save_checkpoint(self):
        """The analogue of checkpointer.save('running'), on train.py's 50-step clock."""
        self._ckpt = {
            'params': [p.detach().clone() for p in self._all_params()],
            'opt': {k: copy.deepcopy(o.state_dict()) for k, o in self.m.optimizers.items()},
            'extra': self.game.extra_state(),
            'step': self.m.step_ind,
        }

    @torch.no_grad()
    def _rewind(self):
        """
        Restore the last healthy checkpoint. Returns False when there is none --
        train.py's `NO REWIND TARGET` branch, which is reachable in a real run
        because 'best' is only written once an eval has improved.
        """
        if self._ckpt is None:
            return False
        for p, saved in zip(self._all_params(), self._ckpt['params']):
            p.copy_(saved)
            if p.grad is not None:
                p.grad = None
        for key, state in self._ckpt['opt'].items():
            self.m.optimizers[key].load_state_dict(copy.deepcopy(state))
        self.game.load_extra_state(self._ckpt['extra'])
        return True

    def _reload_budget(self):
        """train.py's max_reloads_per_1k_steps, floor 3."""
        return max(3.0, self.max_reloads_per_1k * max(self.m.step_ind, 1) / 1000.0)

    # ------------------------------------------------------------------ probe

    def _draw_probe(self):
        return self.game.draw(self.probe_batch)

    def _probe_loss(self, batch):
        return self.game.probe_loss(batch)

    # ------------------------------------------------------------------- step

    def step(self):
        m = self.m
        game = self.game

        # train.py:1918 -- captured BEFORE the step: an OOM slashes batch_size in
        # handle_train_epoch_error, and the throughput denominator wants the size
        # that was actually attempted at that cost
        attempted = m.batch_size
        m._z_cal_rollouts = 0
        loss = grad_norm = None
        theta_before = self.game.policy_params[0].detach().clone()
        # the published methods all key off (theta, grad) BEFORE the step, over
        # every policy tensor rather than just the first
        flat_before = self._flat('param') if self.climber in (
            'armijo', 'bb', 'hyper', 'hyperx', 'hyperz', 'hypern',
                'hyperm', 'hyperg', 'hyperp', 'dog', 'sps') else None

        try:
            # the synthetic GPU is consulted first: an OOM has to happen where a
            # real one does, INSIDE the timed region and before the optimizer step
            t_step = self.gpu.step_time(attempted)

            # train.py:2304 -- the probe runs only where the stage asks for it.
            # In train.py that gate is `protocol.stage.lr_sensor`; here it is the
            # arm under test, which is the same decision made explicit.
            probe_armed = (m.ray_cal.arm(m.step_ind)
                           if 'ray' in (self.climber, self.braker) else False)

            batch = game.draw(attempted)
            loss = game.train_step(batch)
            grad_norm = float(sum(float(p.grad.pow(2).sum())
                                  for p in game.policy_params if p.grad is not None) ** 0.5)

            # train.py:2324 -- measure immediately after, then apply
            if probe_armed:
                reading = m.ray_cal.measure(self._draw_probe, self._probe_loss)
                if reading is not None:
                    d = self.game.policy_params[0].detach() - theta_before
                    self._record_calibration(reading, theta_before, d)
                    if self.reading_filter is not None:
                        reading = self.reading_filter(reading)
                    if reading is not None and self._probe_role_allows(reading):
                        m.lr_controller.on_calibration(reading)

        except (RuntimeError, ValueError) as e:
            # train.py:1932 -- the shared OOM recovery path. This CLEARS the timing
            # deques, and the append below then re-poisons the window with one
            # entry pairing the pre-cut batch against the time-to-failure.
            m.handle_train_epoch_error(e, 'fused')
            t_step = self.gpu.true_step_time(attempted) * 0.1  # failed early
            self.oom_steps += 1

        # the other roles, driving the same actuator
        if self.braker == 'plateau':
            self._plateau_tick(loss)
        if self.braker == 'slope' or self.climber == 'slope_seek':
            # both slope sensors read the same smoothed series
            if loss is not None and math.isfinite(loss):
                a = 1.0 - math.exp(-1.0 / self.SLOPE['ema_period'])
                self._loss_ema = loss if self._loss_ema is None else                     (1 - a) * self._loss_ema + a * loss
                self._slope_window.append(self._loss_ema)
        if self.braker == 'slope':
            self._slope_brake_tick()
        if self.climber == 'slope_seek':
            self._slope_seek_tick()
        if self.climber == 'ramp':
            self._ramp_tick()

        # the published methods. All read the gradient at theta_before, which is
        # still what .grad holds -- train_step steps the optimizer but does not
        # zero afterwards, exactly as train.py leaves it.
        if flat_before is not None and loss is not None:
            g_before = self._flat('grad')
            if self.climber == 'armijo':
                self._armijo_tick(flat_before, g_before,
                                  self._flat('param') - flat_before)
            elif self.climber == 'bb':
                self._bb_tick(flat_before, g_before)
            elif self.climber in ('hyper', 'hyperx', 'hyperz', 'hypern',
                                  'hyperm', 'hyperg', 'hyperp'):
                self._hyper_tick(g_before)
            elif self.climber == 'dog':
                self._dog_tick(g_before)
            elif self.climber == 'sps':
                self._sps_tick(loss, g_before)

        # train.py:2057 -- the one always-on tripwire. The response is
        # fire_loss_spike: REWIND to the running checkpoint, THEN cut the peak
        # (train.py:2109-2169), with an abort once the reload RATE is exceeded.
        if m.lr_controller.check_spike('fused', loss, grad_norm) == 'diverged':
            self.divergences += 1
            self.total_reloads += 1
            if self.total_reloads > self._reload_budget():
                self.aborted = f'reload budget exceeded at step {m.step_ind}'
            elif not self._rewind():
                # train.py's NO REWIND TARGET branch: cut the peak alone and say so
                m.lr_controller.on_divergence()
            else:
                m.lr_controller.on_divergence()

        # train.py:1943/1951 -- timing and WORK, outside the try
        m._recent_step_times.append(t_step)
        m._recent_step_work.append(attempted * (1 + m._z_cal_rollouts))

        # simulated wall clock, then the occupancy sensor -- the REAL one, called
        # every step exactly as train.py does, with its own wall-clock gate deciding
        # whether this step actually takes a reading. Only the NVML leaf is faked
        # (FakeModeller._read_gpu_util reads the synthetic GPU). This used to append
        # to _gpu_util directly on a %10 cadence, which reimplemented the sampling
        # policy and hid the fact that the shipped one could not fill its window.
        m.sim_time += t_step
        m._bench_gpu, m._bench_attempted = self.gpu, attempted
        if self.need_batch_sizer:      # else _sample_gpu_util was never bound (LR-only
            m._sample_gpu_util()       # runs skip the 11 s train.py import)

        # the controller scores the step it just timed (train.py gates
        # select_batch_size on grow_batch_size; the walk-era name
        # increment_batch_size is retired with the state-8 sizer). The
        # need_batch_sizer guard matters since grow defaults TRUE now: an
        # LR-only run never bound the method.
        if self.need_batch_sizer and m.args.grow_batch_size:
            m.select_batch_size()

        # train.py:1970-1971 -- LR schedule on the 10-step clock
        if m.step_ind % 10 == 0:
            m.lr_controller.step()

        self.history.append(dict(
            step=m.step_ind, lr=m.lr_of(game.train_key),
            batch=attempted, loss=loss, grad_norm=grad_norm,
            peak_scale=float(m.lr_ctrl.get('peak_scale', float('nan'))),
            dist=game.distance_to_opt(), step_time=t_step,
        ))
        m.step_ind += 1

        # checkpoint AFTER the step, on train.py's 50-step clock, and only from a
        # healthy state -- saving a diverged one would give the rewind nothing to
        # go back to, which is the whole point of the mechanism
        if (self.checkpoint_every and m.step_ind % self.checkpoint_every == 0
                and not game.diverged()):
            self._save_checkpoint()

    def _record_calibration(self, reading, theta_before, d):
        """Store the reading beside the TRUE alpha*, where the game knows it."""
        true_alpha = None
        if hasattr(self.game, 'alpha_star_true'):
            true_alpha = self.game.alpha_star_true(theta_before, d)
        self.calibrations.append(dict(
            step=self.m.step_ind, status=reading['status'],
            alpha_star=reading['alpha_star'], lo=reading['lo'], hi=reading['hi'],
            alpha_true=true_alpha,
            peak_scale=float(self.m.lr_ctrl.get('peak_scale', float('nan'))),
            lr=self.m.lr_of(self.game.train_key),
        ))

    # -------------------------------------------------------------------- run

    def run(self, steps, stop_on_divergence=True):
        for _ in range(int(steps)):
            self.step()
            if self.aborted:
                # train.py raises FrozenTrainingState here and the job dies. A
                # run past its reload budget is not a slow run, it is a dead one,
                # and scoring it as though it kept training would flatter it.
                break
            if stop_on_divergence and self.game.diverged():
                break
        return self

    # ---------------------------------------------------------------- summary

    def summary(self):
        h = self.history
        return dict(
            steps=len(h),
            final_lr=h[-1]['lr'] if h else None,
            final_batch=self.m.batch_size,
            final_peak=float(self.m.lr_ctrl.get('peak_scale', float('nan'))),
            final_dist=h[-1]['dist'] if h else None,
            diverged=self.game.diverged(),
            n_calibrations=len(self.calibrations),
            n_resolved=sum(1 for c in self.calibrations
                           if c['status'] in ('bracketed', 'above_range', 'below_range')),
            n_divergences=self.divergences,
            oom_steps=self.oom_steps,
            # the knee pin retired at state 8; the sizer's conclusion dict is
            # the successor state (None until the controller first runs)
            sizer=getattr(self.m, 'batch_sizer', None),
            oom_ceiling=self.m.batch_size_oom_ceiling,
        )

    def status_counts(self):
        out = {}
        for c in self.calibrations:
            out[c['status']] = out.get(c['status'], 0) + 1
        return out

    def lr_trace(self, every=1):
        return [(h['step'], h['lr']) for h in self.history[::every]]


def geomean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x) and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float('nan')
