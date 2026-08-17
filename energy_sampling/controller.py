import math


class LRController:
    """
    LR controller v8: a warmup envelope under a peak set by a PER-STAGE SENSOR,
    plus one coarse divergence bar.

        lr = base_lr x peak_scale x envelope(t)

    `envelope` is a fixed warmup ramp, restarted at every stage transition.
    `peak_scale` is moved by whichever sensor the CURRENT STAGE declares
    (protocol.py `lr_sensor`, parsed at Stage._parse_lr_sensor) and by nothing
    else:

        kind: ray      on_calibration(), at most once per
                       ray_calibration.period steps. Coherent only in a fused
                       stage that trains replay TB -- the probe draws from
                       replay and scores replay_loss_coeffs
        kind: hyper    on_hypergradient(), EVERY step, each move bounded by
                       exp(+-beta). Scores no loss, so it is coherent whatever
                       the stage trains
        kind: plateau  on_plateau(), on a ReduceLROnPlateau verdict. Cuts only
        kind: none     nothing moves it; the LRs sit at their resolved base

    Omitting the block entirely means no sensor -- silently -- so `auto` keys
    stay at adaptive_lr.seed_lr for the whole stage while the config reads as
    adaptive. on_divergence() also cuts peak_scale, whatever the sensor. All
    three sensors HOLD THROUGH WARMUP, for the reason written at
    on_calibration.

    WHAT V8 DELETED, AND WHY. v7 estimated alpha* online: a 3-point parabola per
    probe, then a windowed median, a censoring taxonomy, quorum fractions, a
    ramp/cruise state machine, a boost leg, and a relative damage tripwire. Every
    one of those existed to defend a statistic that could not carry the decision.
    alpha* was formed as a RATIO whose denominator, the second difference, has a
    per-probe sd/mean around 3.3 on this route -- so ~40% of single fits came back
    concave and were discarded as 'censored', the median bound to the probe span,
    the IQR collapsed, and the trigger's own quorum became unreachable because
    censored readings sat in its denominator but could never enter its numerator.
    Measured consequence: a 55x LR ramp with the trigger never firing, ending in
    non-finite gradients (lrdisc v1, 2026-08-10).

    v8 does not form that ratio. ray_calibration.py measures the sign of a paired
    loss difference, which answers "is alpha* above this alpha" directly, and
    brackets alpha* on a doubling grid. See that module for the identity.

    THE SETPOINT IS NOT 1. alpha* = 1 is the one-step optimum along a frozen ray,
    and it is ABOVE the rate a run survives: the step that maximises expected
    progress is |g|^2 / (g'Hg + tr(H*Sigma)), and a ray probe at fixed theta
    cannot see the tr(H*Sigma) term. Measured on this route (tuphwfkm): stable,
    improving training sat at alpha* 3.6-5.0, and an excursion to alpha* ~2.8
    visibly degraded it. `alpha_target` therefore defaults to 4 -- roughly a
    quarter of the one-step optimum -- and is a per-route quantity to re-measure,
    not a constant.

    THE UPDATE IS ASYMMETRIC, on principle rather than for tuning:

        peak_scale <- peak_scale x (alpha_hat / alpha_target) ^ eta

    with eta = eta_up when raising and eta_down when lowering. Raising is
    speculative -- it is licensed by a one-step measurement that cannot see
    multi-step effects -- while lowering is the safe direction, because
    undertraining is recovered and damage is not. So eta_up is small and eta_down
    is large. This is also what makes a per-interval overshoot bounded: the LR
    cannot move far up between calibrations, but can be halved in one.

    Because the alpha grid is log-spaced, the response is automatically
    proportional to log-distance from target and saturates at the grid edge -- a
    rate far below target moves fast, a rate near target moves slowly, and a
    reading outside the grid is treated as a BOUND and never extrapolated.

    CEILING. A divergence records the peak_scale that blew up, and it is
    permanent for the run (reset only at a stage transition, whose surface is
    different). It is INSTANCE state, never in lr_ctrl: the rewind that follows a
    divergence restores lr_ctrl from a healthy checkpoint and would otherwise
    erase the evidence.

    lr_ctrl state ver=8 invalidates v1-v7; a stale dict is DISCARDED and rebuilt,
    never reinterpreted.
    """

    CHANNELS = ('fwd', 'bwd', 'replay', 'fused')

    # Policy optimizer keys -> the args attribute holding their base LR. The flow
    # head is deliberately absent: it is pinned, not scheduled.
    _POLICY_BASE = {'fwd': 'lr_policy', 'bwd': 'lr_back', 'replay': 'lr_replay',
                    'fused': 'lr_fused'}

    # How far peak_scale must fall below its own high-water mark before the warmup
    # ramp freezes (see _maybe_freeze_envelope). PER SENSOR, because their firing
    # rates differ by ~500x and the same number cannot mean the same thing to both:
    #   ray      reads at most once per ray_calibration.period, so ANY downward move
    #            is already a considered verdict from a bracketed measurement, not a
    #            noisy sample. 0.0 = freeze on the first cut.
    #   plateau  only ever cuts, so its first cut carries the same weight.
    #   hyper    fires EVERY step against a cosine that swings either side of zero,
    #            so a single reading is noise. peak_scale is the integral of those
    #            readings, which is where persistence accumulates -- 5% off the
    #            high-water mark means it has been pulling down, not jittering.
    # A stage with no sensor never moves peak_scale, so it never freezes and the
    # ramp runs to hold with only on_divergence able to cut -- the no-controller
    # mode, which needs no special case here.
    _FREEZE_DROP_DEFAULT = {'ray': 0.0, 'plateau': 0.0, 'hyper': 0.05}

    _STATE_VER = 8

    def __init__(self, modeller):
        self.modeller = modeller
        self._report = {}
        self._ceiling = None          # peak_scale a divergence proved unusable
        self._divergences = 0
        self._calibrations = 0
        self._last = {}               # last calibration's decision, for the log
        self._plateau_last = {}       # last plateau decision, same purpose
        self._hyper_last = {}         # last hypergradient decision, same purpose
        self._hypergrads = 0
        self._plateau_cuts = 0
        self._restarts = 0
        self._check_bars()

    # ------------------------------------------------------------------ config

    def _cfg(self, name, default):
        return getattr(getattr(self.modeller.args, 'adaptive_lr', None), name, default)

    def _cal_cfg(self, name, default):
        node = getattr(getattr(self.modeller.args, 'adaptive_lr', None), 'calibration', None)
        return getattr(node, name, default) if node is not None else default

    def _managed_keys(self):
        """Config keys the controller owns -- those written `auto`, recorded by
        resolve_derived_config at load. Empty set = it reads and logs but
        actuates nothing, which is its own control arm."""
        return set(getattr(self.modeller.args, 'lr_servo_managed', ()) or ())

    def _check_bars(self):
        """Divergence bars are a sanity floor, not a calibration. Refuse a bar low
        enough to fire on ordinary training: a graduated divergence bar is the
        deleted cut tier coming back in through the config."""
        for name in ('divergence_loss_abs', 'divergence_grad_abs'):
            bar = self._cfg(name, 1.0e9)
            if bar is not None and float(bar) < 1.0e5:
                raise ValueError(
                    f'adaptive_lr.{name} = {bar:g} is below 1e5. This bar exists only to '
                    f'catch numerical explosion; anything that fires on ordinary training '
                    f'is a graduated cut tier, which v8 deleted on evidence.')

    def announce(self):
        cal = getattr(self.modeller, 'ray_cal', None)
        on = cal is not None and getattr(cal, 'enabled', False)
        print(f"lr_ctrl v8: calibration {'ON' if on else 'off'}"
              + (f" (period {cal.period}, n_sub {cal.n_sub}, alphas {list(cal.alphas)})" if on else '')
              + f" | target alpha* {self._cal_cfg('alpha_target', 4.0)}"
              + f" | eta up/down {self._cal_cfg('eta_up', 0.25)}/{self._cal_cfg('eta_down', 0.5)}"
              + f" | managed {','.join(sorted(self._managed_keys())) or 'NOTHING (control arm)'}")

    # -------------------------------------------------------------- divergence

    def check_spike(self, step_type, current_loss, grad_norm):
        """The one always-on tripwire. Returns 'diverged' or None.

        Non-finite readings, or a finite reading past an absolute ~1e9 bar. No
        cooldown, no latch: at this bar a second reading is a second explosion,
        and train.py's max_reloads cap is what stops a rewind loop."""
        checks = []
        if current_loss is not None and step_type in self.CHANNELS:
            checks.append((f'{step_type}_loss', float(current_loss),
                           self._cfg('divergence_loss_abs', 1.0e9)))
        if grad_norm is not None:
            checks.append(('grad_norm', float(grad_norm),
                           self._cfg('divergence_grad_abs', 1.0e9)))
        for channel, value, bar in checks:
            if math.isfinite(value) and (bar is None or value < float(bar)):
                continue
            self._divergences += 1
            print(f"lr_ctrl DIVERGENCE: {channel} = {value:.4g} "
                  f"(bar {float(bar) if bar is not None else float('nan'):.4g}) "
                  f"at step {self.modeller.step_ind} -- reload + peak cut")
            return 'diverged'
        return None

    def on_divergence(self, count: int = 1):
        """Cut the peak and record the ceiling. Called by train.py AFTER the
        rewind, so recompute the envelope rather than trusting the restored one."""
        cut = float(self._cfg('divergence_cut', 0.5)) ** max(int(count), 1)
        st = self._state()
        st['envelope'] = self._envelope(st)
        lo, hi = self._peak_bounds()
        st['peak_scale'] = max(lo, min(hi, float(st['peak_scale']) * cut))
        self._ceiling = st['peak_scale']
        print(f"lr_ctrl: peak_scale -> {st['peak_scale']:.4g} (ceiling recorded)")
        self._apply_lrs(st)

    def _current_ceiling(self):
        return self._ceiling

    # ------------------------------------------------------------- calibration

    def calibration_refusal(self) -> str | None:
        """Why a ray calibration's reading would be thrown away, decided WITHOUT
        measuring anything -- or None if it would be acted on.

        THE POINT OF A SEPARATE PREDICATE. The probe cannot be allowed to draw
        first and find out afterwards. `RayCalibration.measure` draws `n_sub`
        sub-batches from the replay buffer, and those draws consume RNG that
        nothing restores -- so a calibration whose reading is discarded still
        shifts every subsequent training step (findings.md F-039). Anything
        knowable in advance has to be knowable HERE, before a draw happens.

        `on_calibration` consults this too, so the two cannot drift: the gate
        that skips the probe and the gate that refuses the reading are the same
        function, not two copies of one rule.

        DECIDABLE IN ADVANCE (returned by this function):

          warmup   the envelope is deliberately below 1, so the step just taken
                   is a scheduled fraction of the operating step and alpha* rates
                   THAT. peak_scale is the multiplier on the un-suppressed rate,
                   so acting would inflate it by exactly the warmup factor and
                   hand that back the moment the envelope releases -- a jump of
                   lr_warmup_ratio, all at once.

        DECIDABLE IN ADVANCE, AND DELIBERATELY NOT REFUSED:

          an empty `lr_servo_managed` means peak_scale reaches no learning rate,
          but `_managed_keys` calls that "its own control arm" -- the controller
          reads and logs while actuating nothing, and there the reading IS the
          deliverable. "No LR moved" is not the same as "the reading was thrown
          away", and conflating them would delete a documented operating mode.

        NOT DECIDABLE IN ADVANCE, and named here so the gap is explicit rather
        than papered over. Each of these IS the measurement, so no predicate can
        anticipate it and the draws are genuinely spent to find out:

          unresolved       no paired test cleared its CI
          inconsistent     the tests contradict (lo >= hi)
          bad alpha_star   non-finite or non-positive
          no_batch /       the draw itself failed or returned too few
          too_few_subbatches
          clamped          peak_scale already at a `bounds`/ceiling edge, so the
                           multiplier is a no-op -- depends on alpha_star
        """
        st = self._state()
        if self._elapsed(st) < int(self._cfg('warmup_steps', 1000)):
            return 'warmup'
        return None

    def on_calibration(self, reading):
        """
        Apply one periodic ray calibration. `reading` is ray_calibration's dict.

        Acts only on a reading that resolved. 'unresolved' (no test cleared its
        CI) and 'inconsistent' (tests contradict) produce NO move -- that is the
        whole fallback policy, and it is deliberate that there is no other one.
        A calibration that cannot see the answer must not guess it.
        """
        st = self._state()
        status = reading.get('status')
        alpha = reading.get('alpha_star', float('nan'))
        self._calibrations += 1
        self._last = {'status': status, 'alpha_star': alpha, 'applied': 0.0}
        # Kept as the authoritative refusal even though the probe path now checks
        # it BEFORE drawing: this is the rule, and `calibration_refusal` is the
        # same function, so a caller that reaches here anyway still gets it.
        refusal = self.calibration_refusal()
        if refusal is not None:
            self._last['status'] = refusal
            return
        if status not in ('bracketed', 'above_range', 'below_range'):
            return
        if not (isinstance(alpha, float) and math.isfinite(alpha) and alpha > 0):
            return
        target = float(self._cal_cfg('alpha_target', 4.0))
        ratio = alpha / max(target, 1e-9)
        eta = float(self._cal_cfg('eta_up' if ratio > 1.0 else 'eta_down',
                                  0.25 if ratio > 1.0 else 0.5))
        mult = ratio ** eta
        lo, hi = self._peak_bounds()
        ceiling = self._current_ceiling()
        if ceiling is not None:
            hi = min(hi, ceiling)
        before = float(st['peak_scale'])
        st['peak_scale'] = max(lo, min(hi, before * mult))
        self._last['applied'] = st['peak_scale'] / before if before else 1.0
        st['envelope'] = self._envelope(st)
        self._apply_lrs(st)

    def on_plateau(self, fired: bool, factor: float):
        """
        Apply one ReduceLROnPlateau verdict. Held through warmup for the same
        reason the sensor declines to look: the envelope is deliberately moving
        the LR there, so a lack of progress is not evidence about the operating
        point.
        """
        st = self._state()
        self._plateau_last = {'fired': bool(fired), 'applied': 0.0, 'status': 'clean'}
        if self._elapsed(st) < int(self._cfg('warmup_steps', 1000)):
            self._plateau_last['status'] = 'warmup'
            return
        if not fired:
            return
        lo, hi = self._peak_bounds()
        ceiling = self._current_ceiling()
        if ceiling is not None:
            hi = min(hi, ceiling)
        before = float(st['peak_scale'])
        st['peak_scale'] = max(lo, min(hi, before * float(factor)))
        self._plateau_last['applied'] = st['peak_scale'] / before if before else 1.0
        self._plateau_last['status'] = 'cut'
        self._plateau_cuts += 1
        st['envelope'] = self._envelope(st)
        self._apply_lrs(st)
        print(f"lr_ctrl: plateau cut -> peak_scale {st['peak_scale']:.4g}")

    def on_hypergradient(self, cos: float, beta: float, beta_down: float = None,
                         cos_target: float = 0.0):
        """
        Apply one hypergradient verdict: `peak_scale *= exp(beta * cos)`.

        `cos` is the cosine between the CURRENT gradient and the direction the
        PREVIOUS step actually moved the policy in. The identity is
        `dL/d(lr) = -<g_t, d_{t-1}>`, so a positive cosine means the last step
        was too short and a negative one means it overshot.

        WHY THIS EXISTS ALONGSIDE `on_calibration`. The ray probe scores a LOSS,
        which requires that loss to be one the stage actually trains -- the
        precondition written at `protocol.py::_parse_lr_sensor` ("only coherent
        in a fused stage that trains replay TB ... anywhere else it rates a loss
        nobody is optimising"). Measured on run 7tjno8m6, whose `var_conditioning`
        stage pins replay to 0.0 and trains VarGrad on fwd/bwd: 35% of
        calibrations came back `inconsistent` and the t-statistic alternated sign
        at the +-99 clamp between consecutive readings, while the same code on
        `prod0810_mipcas_elj` -- an equilibration stage with replay live at
        0.05-0.6 -- scored 100 bracketed of 102 with zero inconsistent.

        This sensor reads the gradient and the realised displacement, both of
        which exist whatever the branch mixture is and whatever loss family the
        stage trains. It cannot be pointed at the wrong loss because it does not
        score a loss.

        BOUNDED BY CONSTRUCTION: `cos` is a cosine, so one step can move the peak
        by at most `exp(+-beta)` however wrong the rate is. That is what makes it
        safe to run every step, and also what makes it slow to make a large
        correction -- the trade is intrinsic, not a tuning error.

        NO SINGLE beta IS ROBUST ACROSS PROBLEM FAMILIES. Swept on the bench over
        12 cells (two optimizers x tracking and MLE surfaces), the best worst-case
        beta was 0.1 at 3.2x the best fixed rate, and the per-cell optimum spanned
        20x. beta is a BANDWIDTH: a stage whose optimum keeps moving wants a high
        one, a stage whose optimum is static wants a low one to reject noise. It
        is therefore per-stage config with no default that claims universality.
        """
        st = self._state()
        self._hyper_last = {'cos': float(cos), 'applied': 0.0, 'status': 'clean'}
        # Held through warmup for exactly the reason `on_calibration` and
        # `on_plateau` are: the envelope is deliberately ramping the rate, so a
        # cosine measured through that suppression is not evidence about the
        # operating point.
        # HYPER DOES NOT HOLD THROUGH WARMUP -- and it is the reason the ramp can
        # self-terminate. `_maybe_freeze_envelope` freezes the envelope once this
        # sensor has pulled peak_scale materially off its high-water mark, which
        # requires the sensor to be moving peak_scale DURING the ramp. Reinstating
        # the hold here would silently disable that freeze, since peak_scale would
        # sit at 1.0 for the whole warmup and could never fall.
        # ray and plateau still hold: both score a LOSS, which a deliberate ramp
        # really does distort. This one scores no loss.
        if not (isinstance(cos, float) and math.isfinite(cos)):
            self._hyper_last['status'] = 'nonfinite'
            return
        # STEER TO cos == cos_target, NOT to cos == 0. The fixed point of
        # peak_scale *= exp(beta*cos) is cos == 0, which is the one-step optimum --
        # the rate adaptive_lr.calibration's own comment says a run does not
        # survive, because a local probe cannot see the gradient-noise term. `ray`
        # carries that margin as alpha_target: 4.0 (it runs at a QUARTER of the
        # greedy optimum); this is hyper's equivalent, and 0.0 reproduces the old
        # behaviour exactly.
        err = float(cos) - float(cos_target)
        # up/down is decided by the ERROR, not by the raw cosine: with a positive
        # target, a small positive cos now means "hotter than intended" and must
        # take the down branch.
        b = float(beta if err > 0 or beta_down is None else beta_down)
        lo, hi = self._peak_bounds()
        ceiling = self._current_ceiling()
        if ceiling is not None:
            hi = min(hi, ceiling)
        before = float(st['peak_scale'])
        st['peak_scale'] = max(lo, min(hi, before * math.exp(b * err)))
        self._hyper_last['applied'] = st['peak_scale'] / before if before else 1.0
        self._hypergrads = getattr(self, '_hypergrads', 0) + 1
        # ANNOUNCE THE FIRST FIRE, once, the way the other two sensors announce
        # their actions. A per-step sensor must not print per step, but a sensor
        # that silently never runs is the failure this whole exercise was about:
        # the ray probe spent a live run rating a branch pinned to zero weight,
        # and nothing said so.
        if self._hypergrads == 1:
            print(f"lr_ctrl: hypergradient sensor live (beta {b:g}) -- first "
                  f"reading cos {cos:+.3f}, peak_scale {before:.4g} -> "
                  f"{st['peak_scale']:.4g}")
        st['envelope'] = self._envelope(st)
        self._apply_lrs(st)

    # -------------------------------------------------------------------- state

    def _fresh_state(self, phase):
        return {
            'ver': self._STATE_VER,
            'phase_seen': phase,
            'stage_start_step': int(self.modeller.step_ind),
            'peak_scale': 1.0,
            # per-stage, so each stage ramps and freezes on its own evidence
            'peak_high_water': 1.0,
            'envelope_frozen_at': None,
            'envelope': 1.0 / self.modeller.args.lr_warmup_ratio,
        }

    def _state(self):
        m = self.modeller
        st = getattr(m, 'lr_ctrl', None)
        if not isinstance(st, dict) or st.get('ver') != self._STATE_VER:
            if isinstance(st, dict) and st.get('ver') is not None:
                # DISCARD, never reinterpret. v7 and earlier carried disc_state,
                # disc_since, disc_ramp_base and a peak_scale accumulated by a
                # different rule; silently reusing any of it would let a deleted
                # state machine steer a controller that no longer has one.
                print(f"lr_ctrl: discarding stale state ver={st.get('ver')} "
                      f"(this controller is ver={self._STATE_VER}); rebuilding from seed")
            st = self._fresh_state(m.phase)
            m.lr_ctrl = st
        elif st.get('phase_seen') != m.phase:
            st['phase_seen'] = m.phase
        if st.get('stage_start_step', 0) > m.step_ind:
            st['stage_start_step'] = int(m.step_ind)
        st.setdefault('peak_scale', 1.0)
        st.setdefault('envelope', 1.0)
        return st

    def rearm_warmup(self):
        """Protocol.advance hook at every stage transition: restart the warmup
        clock (the optimizers were rebuilt onto a surface with different
        curvature) and forget the ceiling, whose evidence describes a surface
        that no longer exists.

        peak_scale is RESET to 1.0, so each stage re-discovers its own peak.
        Carrying it forward would impose the previous stage's verdict on a
        surface with different curvature, and measured peaks differ
        substantially between stages and runs.

        The reset is NOT free, and deliberately so: `ray` and `hyper` both
        climb, so a stage that inherited a high peak pays warmup plus a
        re-climb to get back to it. That cost is accepted in exchange for never
        carrying a stale verdict across a surface change.

        Returns the warmup length in TRAIN STEPS."""
        m = self.modeller
        st = self._state()
        st['phase_seen'] = m.phase
        self._ceiling = None
        st['peak_scale'] = 1.0
        st['restart_step'] = int(m.step_ind)
        st['stage_start_step'] = int(m.step_ind)
        st['envelope'] = self._envelope(st)
        self._apply_lrs(st)
        return int(self._cfg('warmup_steps', 1000))

    # ----------------------------------------------------------------- envelope

    def _elapsed(self, st):
        return max(0, int(self.modeller.step_ind) - int(st.get('stage_start_step', 0)))

    def _envelope(self, st):
        """Ramp -> hold, in [1/lr_warmup_ratio, 1.0]. No decay leg: the
        calibration rates the PRODUCT peak x envelope, so a deterministic
        multiplier on it is absorbed and only inflates peak against its bounds."""
        # A FROZEN envelope short-circuits the ramp -- see _maybe_freeze_envelope.
        frozen = st.get('envelope_frozen_at')
        if frozen is not None:
            return float(frozen)
        warmup_steps = max(1, int(self._cfg('warmup_steps', 1000)))
        elapsed = self._elapsed(st)
        if elapsed >= warmup_steps:
            return 1.0
        return (1.0 / self.modeller.args.lr_warmup_ratio) ** (1.0 - elapsed / warmup_steps)

    # ----------------------------------------------------------------- actuator

    def _peak_bounds(self):
        b = self._cfg('bounds', (0.01, 2000.0))
        return float(b[0]), float(b[1])

    def _apply_lrs(self, st):
        """lr = base x peak_scale x envelope, floored at min_lr -- EXCEPT the flow
        (Z head) groups, pinned flat at lr_flow, and except groups whose base LR
        was configured as an explicit float, which the controller does not own
        (peak_scale does not apply to them; the envelope still does)."""
        m = self.modeller
        a = m.args
        control_flow = self._cfg('control_flow_lr', False)
        managed = self._managed_keys()
        env = st['envelope']
        peak = float(st['peak_scale'])
        for key, opt in m.optimizers.items():
            n_groups = len(opt.param_groups)
            for gi, g in enumerate(opt.param_groups):
                is_flow_group = key == 'flow' or (key == 'fused' and gi == n_groups - 1)
                if is_flow_group and not control_flow:
                    g['lr'] = a.lr_flow
                    continue
                if key == 'fused':
                    base_key = 'lr_fused'
                elif key == 'flow':
                    base_key = 'lr_flow'
                else:
                    base_key = self._POLICY_BASE[key]
                base = getattr(a, base_key)
                scale = env * (peak if base_key in managed else 1.0)
                g['lr'] = max(a.min_lr, base * scale)

    # --------------------------------------------------------------------- tick

    def step(self):
        """One controller evaluation. Re-stamps the envelope and the LRs; it does
        NOT move peak_scale, except for a warm restart -- otherwise only
        on_calibration, on_hypergradient, on_plateau and on_divergence do."""
        st = self._state()
        self._maybe_restart(st)
        self._maybe_freeze_envelope(st)
        st['envelope'] = self._envelope(st)
        ceiling = self._current_ceiling()
        if ceiling is not None:
            st['peak_scale'] = min(float(st['peak_scale']), ceiling)
        self._apply_lrs(st)
        self._emit(st)
        return self.modeller.optimizers['fwd'].param_groups[0]['lr']

    def _freeze_drop(self):
        """The fall from high-water that freezes the ramp, or None to disable it.

        `adaptive_lr.envelope_freeze_drop` selects between the three modes:
          absent / 'auto'  per-SENSOR default (_FREEZE_DROP_DEFAULT) -- ray and
                           plateau freeze on their first cut, hyper waits for a
                           persistent 5% pull-down
          a float          that threshold whatever the sensor declares
          null / false     freeze OFF: ramp to hold, only on_divergence cuts
        """
        cfg = self._cfg('envelope_freeze_drop', 'auto')
        if cfg is None or cfg is False:
            return None
        if not isinstance(cfg, str):
            return float(cfg)
        if cfg != 'auto':
            raise ValueError(
                f"adaptive_lr.envelope_freeze_drop: expected 'auto', a float, or "
                f"null -- got {cfg!r}")
        stage = getattr(getattr(self.modeller, 'protocol', None), 'stage', None)
        kind = (getattr(stage, 'lr_sensor', None) or {}).get('kind')
        # kind None / 'none' -> no entry -> None -> freeze off, which is right:
        # nothing moves peak_scale on such a stage, so there is nothing to freeze on
        return self._FREEZE_DROP_DEFAULT.get(kind)

    def _maybe_freeze_envelope(self, st):
        """Stop the warmup ramp the moment the sensor is materially pulling AGAINST
        it, and hold the envelope there for the rest of the stage.

        WHY THE RAMP NEEDS AN OFF SWITCH AT ALL. `warmup_steps` is a step count, so
        the ramp ends on a constant rather than on evidence, and it re-arms at every
        stage transition -- which is exactly where the optimizers are fresh and the
        loss surface just changed. Whatever the rate is doing, the envelope keeps
        climbing to 1.0 on schedule. It also fights `on_divergence`: that cuts
        peak_scale, while the envelope goes on re-inflating the product underneath
        it.

        WHY THE TRIGGER IS THE ACTUATOR, NOT THE SENSOR. The obvious rule is "freeze
        when cos goes negative", but cos is noisy about zero -- measured swinging
        -0.2 to +0.4 through a live warmup -- so a single negative reading fires
        within a few steps and the ramp never happens at all. peak_scale is the
        integral of those readings, so noise averages out of it while a sustained
        too-hot verdict accumulates. Freezing on a fall from its own HIGH-WATER mark
        also means the test cannot be tripped by the climb itself.

        Recoverable by construction: after the freeze the sensor still owns
        peak_scale in both directions, with `bounds` (default 2000x) far above
        anything the ramp would have reached. Freezing early costs some warmup, not
        the operating point.

        Per stage: `_fresh_state` clears both fields, so each stage ramps and
        freezes on its own evidence.
        """
        if st.get('envelope_frozen_at') is not None:
            return
        if self._elapsed(st) >= int(self._cfg('warmup_steps', 1000)):
            return                       # ramp already finished; nothing to freeze
        peak = float(st['peak_scale'])
        hw = float(st.get('peak_high_water', peak))
        if peak >= hw:
            st['peak_high_water'] = peak
            return
        st['peak_high_water'] = hw
        drop = self._freeze_drop()
        if drop is None:                 # freeze disabled: ramp to hold regardless
            return
        # peak < hw is already established above, so drop 0.0 means "any fall"
        if peak >= hw * (1.0 - drop):
            return
        st['envelope_frozen_at'] = self._envelope(st)
        print(f"lr_ctrl: warmup ramp FROZEN at envelope {st['envelope_frozen_at']:.4g} "
              f"-- peak_scale {peak:.4g} is {100.0 * (1.0 - peak / hw):.1f}% off its "
              f"high-water {hw:.4g}, i.e. the sensor is pulling against the ramp")

    def _maybe_restart(self, st):
        """Warm restart (SGDR-style): put peak_scale back to 1.0 and let the
        plateau rule decay it again.

        The plateau rule is a pure ratchet, and "no improvement" cannot tell
        too-hot from too-cold -- descent rate is an inverted U in LR -- so an
        over-cut run keeps cutting toward the floor. A periodic reset bounds that
        without a servo that tries to hunt the peak, and cannot oscillate.

        ONE trigger: restart_after train steps since the last restart.
        adaptive_lr.restart_after: null disables it, which is the default.

        There is deliberately NO floor trigger. Restarting because peak_scale
        reached its floor multiplies the LR by 1/floor in a single step -- 100x
        at the old 0.01 bound -- and that detonated five of six qm9anchor_aug14
        arms from a healthy state. It also fired regardless of which sensor moved
        peak_scale, so a hypergradient run correctly tracking a descending
        optimum was reset by plateau-rule machinery it never used. peak_scale
        sitting at the floor means the FLOOR IS TOO HIGH; lower the bound.
        """
        every = self._cfg('restart_after', None)
        if every is None:
            return
        since = int(self.modeller.step_ind) - int(st.get('restart_step', 0))
        if since < int(every):
            return
        st['peak_scale'] = 1.0
        st['restart_step'] = int(self.modeller.step_ind)
        self._restarts += 1
        print(f"lr_ctrl: warm restart ({since} steps) -- peak_scale -> 1.0")

    _STATUS = {'unresolved': 0, 'bracketed': 1, 'above_range': 2,
               'below_range': 3, 'inconsistent': 4, 'warmup': 5}
    # 'warmup' deliberately shares code 5 with _STATUS so the two sensors' status
    # channels read on one scale
    _PLATEAU_STATUS = {'clean': 0, 'cut': 1, 'warmup': 5}
    _HYPER_STATUS = {'clean': 0, 'warmup': 1, 'nonfinite': 2}

    def in_warmup(self) -> bool:
        """Whether the LR envelope is still ramping. Public because a sensor may
        need to decline to SAMPLE during warmup, not merely to act."""
        st = self._state()
        return self._elapsed(st) < int(self._cfg('warmup_steps', 1000))

    def _emit(self, st):
        self._report = {
            'lr_ctrl/envelope': st['envelope'],
            'lr_ctrl/peak_scale': st['peak_scale'],
            'lr_ctrl/scale': st['envelope'] * st['peak_scale'],
            'lr_ctrl/warmup': float(self._elapsed(st) < int(self._cfg('warmup_steps', 1000))),
            'lr_ctrl/divergences': float(self._divergences),
            'lr_ctrl/calibrations': float(self._calibrations),
        }
        if self._last:
            # The ACTUATOR beside the sensor, always: a controller that is
            # holding and one that is satisfied are otherwise indistinguishable
            # from peak_scale alone, which is the failure this module has logged
            # three separate times.
            self._report['lr_ctrl/cal_applied'] = float(self._last.get('applied', 0.0))
            s = self._last.get('status')
            self._report['lr_ctrl/cal_status'] = float(self._STATUS.get(s, -1))
        # Same discipline for the plateau sensor: publish the ACTUATOR beside the
        # sensor, so a sensor that is never firing and one that is never running
        # can be told apart from the log alone.
        if self._plateau_last:
            self._report['lr_ctrl/plateau_applied'] = float(self._plateau_last.get('applied', 0.0))
            self._report['lr_ctrl/plateau_status'] = float(
                self._PLATEAU_STATUS.get(self._plateau_last.get('status'), -1))
            self._report['lr_ctrl/plateau_cuts'] = float(self._plateau_cuts)
        # Same contract as the block above: a sensor that is never FIRING and one
        # that is never RUNNING must be distinguishable from the log alone. This
        # sensor has no equivalent of `raycal/status` to fall back on, so without
        # a counter a misconfigured stage would look identical to a quiet one.
        hl = getattr(self, '_hyper_last', None)
        if hl:
            self._report['lr_ctrl/hyper_cos'] = float(hl.get('cos', 0.0))
            self._report['lr_ctrl/hyper_applied'] = float(hl.get('applied', 0.0))
            self._report['lr_ctrl/hyper_status'] = float(
                self._HYPER_STATUS.get(hl.get('status'), -1))
            self._report['lr_ctrl/hypergrads'] = float(getattr(self, '_hypergrads', 0))
        ceiling = self._current_ceiling()
        if ceiling is not None:
            self._report['lr_ctrl/peak_ceiling'] = ceiling

    def report(self):
        return dict(self._report)
