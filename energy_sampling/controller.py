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

    _STATE_VER = 8

    def __init__(self, modeller):
        self.modeller = modeller
        self._report = {}
        self._ceiling = None          # peak_scale a divergence proved unusable
        self._divergences = 0
        self._calibrations = 0
        self._last = {}               # last calibration's decision, for the log
        self._plateau_last = {}       # last plateau decision, same purpose
        self._hyper_win = None        # hypergradient readings since the last report() drain
        self._hypergrads = 0
        self._plateau_cuts = 0
        self._restarts = 0
        self._lr_capped_groups = 0    # param groups max_lr bound on the last _apply_lrs
        self._lr_floored_groups = 0   # ...and min_lr; a floor that binds is a clamp, not a default
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
        # max_lr vs min_lr CANNOT be resolved by clamp order -- whichever is
        # applied second wins, so one of the two bounds is silently defeated and
        # the run trains at a rate neither bound describes. Refuse it here, where
        # the constructor already refuses an incoherent divergence bar.
        cap = getattr(self.modeller.args, 'max_lr', None)
        if cap is not None:
            cap = float(cap)
            floor = float(getattr(self.modeller.args, 'min_lr', 0.0) or 0.0)
            if cap <= 0.0:
                raise ValueError(f'max_lr = {cap:g} must be positive, or null for no cap.')
            if cap < floor:
                raise ValueError(
                    f'max_lr = {cap:g} is below min_lr = {floor:g}. Clamping to both is '
                    f'not possible: whichever is applied second wins and the other bound '
                    f'is silently defeated. Raise max_lr or lower min_lr.')

    def announce(self):
        cal = getattr(self.modeller, 'ray_cal', None)
        on = cal is not None and getattr(cal, 'enabled', False)
        print(f"lr_ctrl v8: calibration {'ON' if on else 'off'}"
              + (f" (period {cal.period}, n_sub {cal.n_sub}, alphas {list(cal.alphas)})" if on else '')
              + f" | target alpha* {self._cal_cfg('alpha_target', 4.0)}"
              + f" | eta up/down {self._cal_cfg('eta_up', 0.25)}/{self._cal_cfg('eta_down', 0.5)}"
              + f" | managed {','.join(sorted(self._managed_keys())) or 'NOTHING (control arm)'}")

    # -------------------------------------------------------------- divergence

    #: Observations on a channel before its relative bar arms. Without it the
    #: first loss of a stage becomes the reference and the second can convict.
    _REL_MIN_OBS = 50
    #: Floor under the reference, so a loss that legitimately approaches zero
    #: cannot drag the bar to zero with it and convict ordinary noise.
    _REL_FLOOR = 1.0e-3

    def _rel_loss_bar(self, step_type, current_loss):
        """Update this channel's stage-scoped running minimum and return the
        relative divergence bar, or None while the rule is unset or unarmed.

        Reads the CURRENT loss into the minimum BEFORE returning a bar, so a
        genuinely new low can never convict itself."""
        mult = self._cfg('divergence_loss_rel', None)
        if mult is None or float(mult) <= 0:
            return None
        st = self._state()
        book = st.setdefault('rel_loss', {})
        seen = book.get(step_type)
        value = float(current_loss)
        if math.isfinite(value):
            if seen is None:
                book[step_type] = [value, 1]
            else:
                seen[0] = min(seen[0], value)
                seen[1] += 1
            seen = book[step_type]
        if seen is None or seen[1] < self._REL_MIN_OBS:
            return None
        return max(seen[0], self._REL_FLOOR) * float(mult)

    def check_spike(self, step_type, current_loss, grad_norm):
        """The one always-on tripwire. Returns 'diverged' or None.

        Non-finite readings, a finite reading past an absolute ~1e9 bar, or --
        where `divergence_loss_rel` is set -- a reading more than that multiple
        above the QUIETEST loss this stage has produced. No cooldown, no latch:
        at these bars a second reading is a second explosion, and train.py's
        `max_reloads_per_1k_steps` budget is what stops a rewind loop. Note it is
        a RATE, not a count -- a long run is not aborted for the same per-step
        behaviour as a short one.

        WHY A RELATIVE BAR EXISTS AT ALL. The absolute 1e9 is a backstop against
        numerical death, not a statement about training: a route whose loss lives
        at O(1) can go up a hundredfold -- destroying the run -- and never come
        near it. On a well-behaved stage that excursion IS the event worth
        rewinding from, and it is invisible to a bar six orders of magnitude
        above the operating point.

        THE REFERENCE IS THE STAGE'S OWN MINIMUM, and it is per stage for the
        same reason peak_scale is: stages differ in loss SCALE by orders of
        magnitude (an MLE stage and a VarGrad stage are not comparable), so a
        minimum carried across a transition would either never fire or fire
        immediately. `rearm_warmup` clears it.

        THREE GUARDS, because a ratio is easy to make trigger-happy:
          * ARMING. The bar is inert until `_REL_MIN_OBS` observations on the
            channel, so the first reading cannot become the reference and
            convict the second.
          * A FLOOR. The reference is `max(min_seen, _REL_FLOOR)`, so a loss that
            legitimately touches ~0 cannot make the bar ~0 with it.
          * PER CHANNEL. fwd, bwd and replay have different scales; a shared
            minimum would be the smallest of them and would convict the others.
        """
        checks = []
        if current_loss is not None and step_type in self.CHANNELS:
            checks.append((f'{step_type}_loss', float(current_loss),
                           self._cfg('divergence_loss_abs', 1.0e9)))
            rel = self._rel_loss_bar(step_type, current_loss)
            if rel is not None:
                checks.append((f'{step_type}_loss_rel', float(current_loss), rel))
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

        DECIDABLE IN ADVANCE (returned by this function): nothing, currently.
        This function is kept because it is the one place the rule lives and the
        probe path still asks before drawing; the warmup case moved below.

        DECIDABLE IN ADVANCE, AND DELIBERATELY NOT REFUSED:

          warmup   the envelope is deliberately below 1, so the step just taken
                   is a scheduled fraction of the operating step and alpha* rates
                   THAT. peak_scale is the multiplier on the un-suppressed rate,
                   so ACTING would inflate it by exactly the warmup factor and
                   hand that back the moment the envelope releases -- a jump of
                   lr_warmup_ratio, all at once. That argument is unchanged and
                   `on_calibration` still declines to actuate here. It is an
                   argument against acting, not against LOOKING: the reading is
                   what tells the ramp to stop, and a ramp nothing watches was
                   the worse failure. Freeze-only during warmup.

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
        # WARMUP IS NO LONGER A REFUSAL, and that is a deliberate reversal. It
        # used to be, because acting on a reading taken under a suppressed
        # envelope would inflate the peak by exactly the warmup factor. That
        # reason is intact and is why `on_calibration` still refuses to ACTUATE
        # during the ramp -- but it is an argument against acting, not against
        # LOOKING, and the ramp needs something watching it. At period 500
        # against a 1000-step warmup this sensor gets two readings, so it cannot
        # average and instead freezes the ramp on the first downward one.
        #
        # THE PROBE IS NOT FREE. `measure` draws n_sub sub-batches whose RNG
        # nothing restores, so arming here shifts every subsequent step
        # (findings F-039) and runs are not comparable with pre-change ones.
        # That cost is accepted: an unwatched ramp was the worse trade.
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
        # DURING THE RAMP THIS READING STOPS IT AND DOES NOTHING ELSE. Actuating
        # is still refused for the original reason -- the envelope is
        # deliberately below 1, so the multiplier would be applied to a rate that
        # is about to be raised by the ramp anyway -- but a resolved reading
        # BELOW target says the rate is already hotter than we steer to, and that
        # is exactly the "stop climbing" verdict the ramp needs.
        #
        # FIRST READING, NO AVERAGING, unlike hyper. With period 500 against a
        # 1000-step warmup there are two readings in the whole ramp, so there is
        # nothing to average over -- and the asymmetry licenses it: freezing
        # early costs some warmup, not the operating point.
        if self._elapsed(st) < int(self._cfg('warmup_steps', 1000)):
            self._last['status'] = 'warmup_ramp'
            if alpha < target:
                self._freeze_envelope(
                    st, f'ray alpha* {alpha:.3g} below target {target:g} on its '
                        f'first resolved reading of the ramp')
            return
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

    def _hyper_window(self):
        """The per-reporting-period accumulator `_emit` publishes and `report`
        drains. Lazily created, so `None` means UNAMBIGUOUSLY 'this sensor has
        not fired since the last report' -- which is the whole point; see _emit."""
        w = self._hyper_win
        if w is None:
            w = self._hyper_win = {'n': 0, 'cos_sum': 0.0, 'log_applied': 0.0,
                                   'nonfinite': 0, 'status': 'clean'}
        return w

    def _clip_saturated(self, st, clip_ratio):
        """Is the gradient clip firing so hard that `cos` has stopped being a
        learning-rate statistic? Updates the persistence EMA as a side effect.

        `clip_ratio` is pre-clip grad norm / the guard's bar for that branch,
        handed in by the caller because the guard's own counters are DRAINED at
        every report and reading them here would race the reporter.

        WHY THIS IS NOT A REFUSAL. `ray` answers an unusable reading with "no
        move" (on_calibration: "a calibration that cannot see the answer must not
        guess it"). That is right for `ray` and wrong here, because the state
        that makes cos unusable is itself unambiguous evidence about the rate:
        once the clip binds on essentially every step the update magnitude is set
        by the LR alone, decoupled from curvature, and a rate that does that is
        too high. So the correct response is to CUT, not to abstain.

        MEASURED, 2026-08-17, hyperslope_aug17: `gradclip/fused_fire_rate` is
        0.000 through every healthy window (lr8e5, hl28) and 1.000 through every
        window in which cos misreads (lr2e4, lr5e4) -- with lr2e4's pre-clip norm
        at 3.7e4 against a healthy 37. The separation is total, so the threshold
        does not need to be delicate.

        PERSISTENCE, NOT ONE READING. The guard targets a 1-p fire rate (0.01 at
        the shipped p=0.99), so single firings are the design and only a
        SUSTAINED rate means anything. The default bar is 0.5 -- fifty times the
        design rate -- and a full window must elapse before it may fire at all.
        """
        bar = self._cfg('hyper_clip_fire_rate_max', 0.5)
        if bar is None or clip_ratio is None:
            return False
        try:
            ratio = float(clip_ratio)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(ratio):
            ratio = float('inf')        # non-finite grad IS saturation, not a skip
        span = max(1, int(self._cfg('hyper_clip_window', 50)))
        alpha = 2.0 / (span + 1.0)
        fired = 1.0 if ratio >= 1.0 else 0.0
        prev = st.get('hyper_clip_ema')
        st['hyper_clip_ema'] = fired if prev is None else alpha * fired + (1.0 - alpha) * prev
        st['hyper_clip_n'] = int(st.get('hyper_clip_n', 0)) + 1
        # THE EMA IS NEVER RESET, and that is load-bearing. An earlier version
        # cleared it on each cut to rate-limit the braking, which also cleared
        # the SUPPRESSION -- so between cuts cos resumed integrating and simply
        # out-ran the brake. Measured on the first version of
        # test_sustained_clip_saturation_cuts_the_rate: cos +0.5 lifts peak_scale
        # by exp(beta*0.5) per firing, x3.49 over a 50-firing window, against a
        # single x0.5 cut -- a NET RISE of 1.75x per window while the clip was
        # pinned. Suppression has to be continuous; only the cut is rate-limited,
        # which _clip_saturation_cut does with its own counter.
        return st['hyper_clip_n'] >= span and st['hyper_clip_ema'] > float(bar)

    def _clip_saturation_cut(self, st, w):
        """Cut the peak because the clip is saturated, and re-arm the detector.

        TWO EFFECTS, ON DIFFERENT CLOCKS. Reaching here at all means cos is not
        integrated this firing -- that suppression is CONTINUOUS, for as long as
        the clip stays saturated, because a statistic measuring clip geometry
        should never move the rate. The CUT is rate-limited to one per window:
        halving on every firing would compound to 0.5**n and floor the rate
        inside a single window. So the response is "stop listening to cos, and
        halve once per window of sustained evidence" -- hard, but recoverable,
        and it cannot be out-run by cos the way a reset-on-cut version was."""
        cut = float(self._cfg('hyper_clip_cut', 0.5))
        span = max(1, int(self._cfg('hyper_clip_window', 50)))
        w['n'] += 1
        w['status'] = 'clip_saturated'
        since = int(st.get('hyper_clip_n', 0)) - int(st.get('hyper_clip_cut_at', -span))
        if since < span:
            return                      # suppressed, but not yet due another cut
        lo, hi = self._peak_bounds()
        ceiling = self._current_ceiling()
        if ceiling is not None:
            hi = min(hi, ceiling)
        before = float(st['peak_scale'])
        st['peak_scale'] = max(lo, min(hi, before * cut))
        st['hyper_clip_cut_at'] = int(st.get('hyper_clip_n', 0))
        w['clip_cuts'] = int(w.get('clip_cuts', 0)) + 1
        if before > 0 and st['peak_scale'] > 0:
            w['log_applied'] += math.log(st['peak_scale'] / before)
        self._clip_cuts = getattr(self, '_clip_cuts', 0) + 1
        if self._clip_cuts == 1:
            print(f"lr_ctrl: CLIP SATURATED -- the grad clip is firing on "
                  f"essentially every step, so cos is measuring clip geometry "
                  f"rather than curvature and the rate is too high on that "
                  f"evidence alone. peak_scale {before:.4g} -> "
                  f"{st['peak_scale']:.4g}. Further cuts are silent.")
        st['envelope'] = self._envelope(st)
        self._apply_lrs(st)

    def on_hypergradient(self, cos: float, beta: float, beta_down: float = None,
                         cos_target: float = 0.0, clip_ratio: float = None):
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
        w = self._hyper_window()
        # THE REGIME GATE RUNS FIRST, before anything reads cos -- including
        # during warmup, where the envelope is deliberately holding the rate
        # BELOW the operating point, so a saturated clip there is worse news
        # still. See _clip_saturated for why this CUTS rather than abstains.
        if self._clip_saturated(st, clip_ratio):
            self._clip_saturation_cut(st, w)
            return
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
            w['n'] += 1
            w['nonfinite'] += 1
            w['status'] = 'nonfinite'
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
        #
        # ASYMMETRIC BY DEFAULT, for the reason `ray` ships eta_up 0.25 against
        # eta_down 0.5: a raise is licensed only by a local reading that cannot
        # see multi-step damage, while a cut is the recoverable direction --
        # undertraining is recovered, a detonation is not. `beta_down` has been
        # plumbed through protocol.py since this sensor shipped and was set by
        # nothing, so in practice the gain was symmetric everywhere. Setting
        # hyper_down_gain to 1.0 restores that exactly.
        if beta_down is None:
            beta_down = float(beta) * float(self._cfg('hyper_down_gain', 2.0))
        b = float(beta if err > 0 else beta_down)
        # THE RAMP IS DETERMINISTIC; THIS SENSOR ONLY DECIDES WHEN IT ENDS.
        #
        # During warmup the envelope is deliberately holding the rate
        # lr_warmup_ratio below the operating point, so `err` is structurally
        # POSITIVE -- that is the suppression reflected back, not evidence that
        # the rate is too low. Actuating on it made peak_scale climb AGAINST the
        # ramp: measured on the bench, to the 2000x bound in ~76 steps at
        # beta 0.1, which destroys the one property the ramp exists to provide.
        #
        # So the reading feeds a SMOOTHED error instead of the actuator, and the
        # ramp ends when that average reaches the setpoint -- "ramp until cos
        # reaches the target, then stop". Smoothed because one reading is noise
        # (measured swinging -0.2 to +0.4 through a live warmup); the window is
        # the only knob. This is NOT a rare safety catch: cos at the operating
        # point sits near zero, so on a healthy run the ramp is EXPECTED to end
        # this way rather than by running out of steps.
        if self._ramping(st):
            span = max(1, int(self._cfg('warmup_freeze_cos_window', 25)))
            n = int(st.get('hyper_cos_n', 0)) + 1
            prev = st.get('hyper_cos_ema')
            a = 2.0 / (span + 1.0)
            st['hyper_cos_ema'] = err if prev is None else a * err + (1.0 - a) * prev
            st['hyper_cos_n'] = n
            # The statistic is still MEASURED and published; only the actuation
            # is withheld, so the sensor channel cannot go dark through a ramp.
            w['n'] += 1
            w['cos_sum'] += float(cos)
            w['status'] = 'warmup_ramp'
            # A FULL WINDOW BEFORE IT MAY FIRE, so one early reading cannot end
            # the ramp -- the failure the high-water rule below was built against.
            if n >= span and st['hyper_cos_ema'] <= 0.0:
                self._freeze_envelope(
                    st, f'smoothed err {st["hyper_cos_ema"]:+.3f} <= 0 over a '
                        f'{span}-step mean (cos target {float(cos_target):+.3g})')
            return
        lo, hi = self._peak_bounds()
        ceiling = self._current_ceiling()
        if ceiling is not None:
            hi = min(hi, ceiling)
        before = float(st['peak_scale'])
        # THE LEAK. Without it this is a pure integrator in log space -- pole
        # exactly on the unit circle, infinite DC gain -- so ANY constant bias in
        # the error produces unbounded exponential drift in the rate, and
        # zero-mean noise produces an unbounded random walk. The `bounds` clip is
        # saturation, not a restoring force, which is why the failure looks like
        # "railed at 0.01" rather than "drifted somewhere unhelpful".
        #
        #     log peak <- (1 - lam) * log peak + b * err
        #
        # moves the pole to 1 - lam. A sustained bias then buys a FINITE offset
        # b*err_bar/lam instead of a ramp, and the stationary spread under noise
        # is sigma*sqrt(b*tau_c/(2*lam)) instead of growing without limit. The
        # controller's total authority becomes exactly b*err_bar/lam, so lam is
        # chosen by inverting that: pick how far the rate may travel from seed
        # against the worst sustained bias, and solve. Measured on this route the
        # per-firing |err| runs 0.01-0.05, so lam 2e-3 at beta 0.05 bounds the
        # excursion near 3x.
        #
        # DEFAULT 0.0 = OFF = today's behaviour, bit for bit. It is off rather
        # than on because lam encodes a timescale, and the only route it has been
        # measured on is the QM9 conditional one; a default here would be a
        # universal claim the measurements do not support -- the same reason
        # `beta` has no default.
        lam = float(self._cfg('peak_leak', 0.0) or 0.0)
        if lam > 0.0 and before > 0.0:
            logp = (1.0 - lam) * math.log(before) + b * err
            st['peak_scale'] = max(lo, min(hi, math.exp(logp)))
        else:
            st['peak_scale'] = max(lo, min(hi, before * math.exp(b * err)))
        w['n'] += 1
        w['cos_sum'] += float(cos)
        # ACCUMULATED IN LOG SPACE, because the actuator is multiplicative: the
        # period's total move is the product of its per-firing multipliers, and
        # summing the ratios instead would report a period that halved then
        # doubled as having moved by 2.5x rather than 1.0.
        if before > 0 and st['peak_scale'] > 0:
            w['log_applied'] += math.log(st['peak_scale'] / before)
        w['status'] = 'clean'
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
            # hyper's warmup ramp-exit detector: an EMA of the cos ERROR and the
            # count behind it. Read with .get() everywhere, so a state restored
            # from before these existed simply starts the average fresh.
            'hyper_cos_ema': None,
            'hyper_cos_n': 0,
            # WHERE THE RAMP STARTS. At a cold start there is no previous rate to
            # continue from, so it is the configured 1/lr_warmup_ratio below seed;
            # at a stage transition `rearm_warmup` overwrites it with the rate the
            # OUTGOING stage was actually running.
            'ramp_from': 1.0 / self.modeller.args.lr_warmup_ratio,
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

        BUT THE RESET IS ONLY SAFE BECAUSE THE RAMP ABSORBS IT, so this method
        owns re-arming the ramp as much as it owns the reset, and the two are
        one transaction:

          `ramp_from` <- the scale the OUTGOING stage was actually running
                         (peak x envelope), so peak_scale -> 1.0 changes no
                         learning rate on the transition step itself. The ramp
                         then walks that back up to seed, and a stage that ran
                         hotter than seed gets no ramp (see `_ramp_from`).
          `envelope_frozen_at` <- None, or the outgoing stage's freeze latch
                         short-circuits `_envelope` and there is NO RAMP AT ALL
                         to absorb the reset. Measured, mmnxotsr 2026-08-20:
                         train_prior froze its ramp at envelope 0.1194 and the
                         latch survived every later transition, so phase 1 -> 2
                         landed as a bare 81x step in ONE optimizer step, on the
                         same step Adam's moments were rebuilt. This method used
                         to print "LR re-warming" and then not re-warm.
          `peak_high_water` / `hyper_cos_*` <- the freeze rules' own evidence,
                         which describes the outgoing surface.

        Returns the warmup length in TRAIN STEPS."""
        m = self.modeller
        st = self._state()
        st['phase_seen'] = m.phase
        self._ceiling = None
        # READ BEFORE RESETTING: this is the rate the outgoing stage settled on.
        outgoing = float(st.get('peak_scale', 1.0)) * float(st.get('envelope', 1.0))
        st['ramp_from'] = outgoing
        st['peak_scale'] = 1.0
        st['peak_high_water'] = 1.0
        st['envelope_frozen_at'] = None
        st['hyper_cos_ema'] = None
        st['hyper_cos_n'] = 0
        st['restart_step'] = int(m.step_ind)
        st['stage_start_step'] = int(m.step_ind)
        # the relative divergence bar's reference is per stage -- see
        # check_spike. Loss SCALE differs by orders of magnitude between stages,
        # so a minimum carried across would either never fire or fire at once.
        st['rel_loss'] = {}
        st['envelope'] = self._envelope(st)
        self._apply_lrs(st)
        warmup_steps = int(self._cfg('warmup_steps', 1000))
        start = self._ramp_from(st)
        if start >= 1.0:
            print(f"lr_ctrl: no warmup ramp -- the outgoing stage was running at "
                  f"scale {outgoing:.4g}, at or above seed, so peak_scale 1.0 "
                  f"applies immediately")
        else:
            print(f"lr_ctrl: warmup ramp re-armed -- envelope {start:.4g} -> 1.0 "
                  f"over {warmup_steps} train steps ({1.0 / start:.1f}x), "
                  f"continuing the outgoing stage's rate")
        return warmup_steps

    # ----------------------------------------------------------------- envelope

    def _elapsed(self, st):
        return max(0, int(self.modeller.step_ind) - int(st.get('stage_start_step', 0)))

    def _ramping(self, st) -> bool:
        """Is the warmup envelope still MOVING? Not the same as "inside the
        warmup step budget", and the difference is 800 steps of dead time.

        hyper is held to freeze-only while the ramp runs, because a cosine
        measured through a deliberately-suppressed rate says "too cold" whatever
        the operating point is. That argument expires the moment the ramp is
        FROZEN: the envelope is then a constant, the rate is no longer being
        walked, and a reading is ordinary evidence again.

        It expires the same way when there is NOTHING TO RAMP -- `ramp_from` at
        1.0, which is what a stage that ran at or above seed hands over. The
        envelope is a constant there too, so holding the sensor for the warmup
        budget would be the same dead time with no suppression to justify it.

        Keying the hold on `warmup_steps` alone kept the sensor mute for the rest
        of the budget after an early freeze. Measured, run
        newlogic_qm9cond_newlogic 2026-08-17: var_conditioning opened at step
        150, the ramp froze at 350 on a negative smoothed cos, and hyper then sat
        in freeze-only mode until 1150 -- 800 steps, 16% of the run, at a
        constant 1.98e-5 with cos reading -0.02 to -0.04 throughout and
        hypergrads stuck at 0.
        """
        return (st.get('envelope_frozen_at') is None
                and self._ramp_from(st) < 1.0
                and self._elapsed(st) < int(self._cfg('warmup_steps', 1000)))

    def _ramp_from(self, st):
        """The envelope value the current ramp starts at, in (0, 1].

        Clamped to 1.0 at the top because THE RAMP HAS NO DECAY LEG: a stage
        whose outgoing rate was at or above seed gets no ramp at all (the
        exponent below collapses to 1.0 everywhere), which is the cut to seed
        that `rearm_warmup` intends, applied at once rather than annealed."""
        v = st.get('ramp_from')
        if v is None:                     # state restored from before this key
            v = 1.0 / float(self.modeller.args.lr_warmup_ratio)
        return min(1.0, max(1e-12, float(v)))

    def _envelope(self, st):
        """Ramp -> hold, in [ramp_from, 1.0]. No decay leg: the calibration rates
        the PRODUCT peak x envelope, so a deterministic multiplier on it is
        absorbed and only inflates peak against its bounds.

        THE RAMP STARTS AT THE OUTGOING RATE, NOT AT A FIXED FRACTION OF SEED.
        A fixed 1/lr_warmup_ratio floor is anchored to `seed_lr`, and a stage
        that ran for thousands of steps has usually moved a long way from it --
        so "reset to a low rate and re-ramp" would reset to a rate far ABOVE the
        one the run had settled on. Measured, mmnxotsr 2026-08-20: train_prior
        exited at peak_scale 0.0113 x envelope 0.1194 = 1.7e-7, while
        seed_lr/lr_warmup_ratio is 1.25e-5 -- the "low" starting point was 74x
        HOTTER than the rate it was meant to be lower than. Starting at the
        outgoing scale makes the boundary continuous in LR by construction.

        The ceiling stays 1.0 (= seed): every stage gets one ramp back up to the
        seed rate and no further, with `bounds` remaining the sensor's own
        post-ramp exploration range."""
        # A FROZEN envelope short-circuits the ramp -- see _maybe_freeze_envelope.
        frozen = st.get('envelope_frozen_at')
        if frozen is not None:
            return float(frozen)
        warmup_steps = max(1, int(self._cfg('warmup_steps', 1000)))
        elapsed = self._elapsed(st)
        if elapsed >= warmup_steps:
            return 1.0
        return self._ramp_from(st) ** (1.0 - elapsed / warmup_steps)

    # ----------------------------------------------------------------- actuator

    def _peak_bounds(self):
        b = self._cfg('bounds', (0.01, 2000.0))
        return float(b[0]), float(b[1])

    def _max_lr(self):
        """The absolute ceiling on any rate this controller writes, or None.

        A RAIL, NOT A RANGE. `adaptive_lr.bounds` stays wide on purpose -- the
        controller is meant to find its own operating range, and narrowing bounds
        to express a safety limit would also delete the exploration. This is the
        separate thing: a hard number, in absolute learning-rate units, that no
        group may exceed however the servo got there.

        Measured 2026-08-17 (hyperslope_aug17, QM9 conditional, rate pinned per
        arm): 5e-6 through 8e-5 run 2000 steps clean, 2e-4 goes non-finite at
        step 1560, 5e-4 at step 560. The survivable range is about 16x wide,
        against `bounds` defaults spanning 200,000x -- four orders of magnitude
        more room than the run tolerates. Absent (the default) is no cap, which
        reproduces the behaviour of every config that predates the key."""
        cap = getattr(self.modeller.args, 'max_lr', None)
        return None if cap is None else float(cap)

    def _apply_lrs(self, st):
        """lr = base x peak_scale x envelope, capped at max_lr and floored at
        min_lr -- EXCEPT the flow (Z head) groups, pinned flat at lr_flow, and
        except groups whose base LR was configured as an explicit float, which
        the controller does not own (peak_scale does not apply to them; the
        envelope still does).

        THE CAP APPLIES TO EVERY GROUP THIS METHOD WRITES, including the two the
        servo does not own:

          the FLOW group, because it is the one rate with no other guard at all.
          peak_scale never reaches it (control_flow_lr is false), the envelope
          never reaches it, and a divergence cut cannot move it -- so before this
          cap there was no mechanism by which any controller could lower it. The
          conditional route's flow head is a real network, and this base config
          warns it diverges at the canonical lr_flow 0.1.

          EXPLICIT-FLOAT groups, because the rail is about what the optimizer
          actually receives, not about who chose it. For those the cap binds iff
          the written float itself exceeds it, since the envelope only reduces.

        The cap is the LAST transform before the floor, so it binds on the
        product rather than on any one factor, and `_check_bars` has already
        refused a cap below min_lr -- otherwise the two clamps would fight and
        whichever ran second would silently win."""
        m = self.modeller
        a = m.args
        control_flow = self._cfg('control_flow_lr', False)
        managed = self._managed_keys()
        env = st['envelope']
        peak = float(st['peak_scale'])
        cap = self._max_lr()
        capped = 0
        floored = 0
        for key, opt in m.optimizers.items():
            n_groups = len(opt.param_groups)
            for gi, g in enumerate(opt.param_groups):
                is_flow_group = key == 'flow' or (key == 'fused' and gi == n_groups - 1)
                if is_flow_group and not control_flow:
                    # pinned, so no min_lr floor here either -- unchanged
                    want = a.lr_flow
                    if cap is not None and want > cap:
                        want, capped = cap, capped + 1
                    g['lr'] = want
                    continue
                if key == 'fused':
                    base_key = 'lr_fused'
                elif key == 'flow':
                    base_key = 'lr_flow'
                else:
                    base_key = self._POLICY_BASE[key]
                base = getattr(a, base_key)
                scale = env * (peak if base_key in managed else 1.0)
                want = base * scale
                if cap is not None and want > cap:
                    want, capped = cap, capped + 1
                # A BINDING FLOOR IS AS INVISIBLE AS A BINDING CEILING, and on
                # this route it is the more likely of the two: measured quality
                # optimum on the conditional VarGrad route is ~2e-6 to 2e-5,
                # against a shipped min_lr of 1e-6 -- so the floor sits barely
                # below the good range and a controller asking to go lower is
                # silently refused. It also truncates the BOTTOM OF THE RAMP once
                # seed_lr is set sensibly: seed 5e-6 at lr_warmup_ratio 10 starts
                # at 5e-7, under the floor, so warmup would begin clamped.
                if want < a.min_lr:
                    floored += 1
                g['lr'] = max(a.min_lr, want)
        # A BINDING RAIL MUST BE VISIBLE. A clamped controller and a satisfied
        # one are otherwise indistinguishable from the rate alone, which is the
        # failure this module has now logged four separate times.
        self._lr_capped_groups = capped
        self._lr_floored_groups = floored

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

    def _freeze_enabled(self) -> bool:
        """Whether the warmup ramp may be frozen at all. `adaptive_lr.envelope_freeze`,
        default TRUE.

        A BOOLEAN, WHERE IT USED TO BE A THRESHOLD. `envelope_freeze_drop` named
        how far peak_scale had to fall from its high-water mark, per sensor,
        because the freeze read the actuator and the actuator carried the
        sensor's noise. It does not any more: no sensor moves peak_scale during a
        ramp, so the only thing that can is on_divergence, whose cut is
        unambiguous by construction. Measured before removing it -- every
        threshold from 0.0 to 0.4 returned the identical verdict, since
        divergence_cut 0.5 is a 50% fall against a largest default of 5%. Nothing
        noisy reaches this decision, so there is nothing left to threshold.
        Retired at state 7."""
        return bool(self._cfg('envelope_freeze', True))

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

        Per stage: `_fresh_state` at a cold start and `rearm_warmup` at every
        transition clear both fields, so each stage ramps and freezes on its own
        evidence.
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
        if not self._freeze_enabled():   # ramp to hold regardless
            return
        # peak < hw is established above, and the only thing that can have moved
        # it during a ramp is on_divergence -- so ANY fall here is a divergence,
        # which needs no threshold to be believed.
        self._freeze_envelope(
            st, f'peak_scale {peak:.4g} is {100.0 * (1.0 - peak / hw):.1f}% off its '
                f'high-water {hw:.4g}, i.e. the sensor is pulling against the ramp')

    def _freeze_envelope(self, st, reason: str):
        """Latch the warmup ramp at its current envelope, once, with a reason.

        THREE PATHS REACH THIS and they are deliberately different shapes, each
        matched to how often its sensor speaks:

          hyper   every step, so it can afford to average -- fires on a smoothed
                  error reaching the setpoint (on_hypergradient)
          ray     once per `period` (500) against a 1000-step ramp, so averaging
                  is not available: it fires on the FIRST downward reading
                  (on_calibration)
          divergence / plateau  moved peak_scale off its high-water mark
                  (_maybe_freeze_envelope, above)

        Idempotent, so a second trigger in the same stage is a no-op rather than
        re-latching at a lower envelope. `_fresh_state` (cold start) and
        `rearm_warmup` (every transition) clear the field, so each stage ramps
        and freezes on its own evidence.

        `envelope_freeze: false` still means FREEZE OFF, and it is honoured
        here rather than at each caller so the off switch cannot be bypassed by
        adding a path. Only its ON/OFF sense applies to the two new callers: its
        numeric value is a fall-from-high-water threshold, which is meaningless
        for a rule that reads a smoothed cos or a single alpha*."""
        if st.get('envelope_frozen_at') is not None:
            return
        if not self._freeze_enabled():
            return
        st['envelope_frozen_at'] = self._envelope(st)
        print(f"lr_ctrl: warmup ramp FROZEN at envelope "
              f"{st['envelope_frozen_at']:.4g} -- {reason}")

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
    # 'warmup_ramp' is the freeze-only warmup mode; 'clip_saturated' is the
    # regime gate having cut on clip evidence rather than on cos.
    _HYPER_STATUS = {'clean': 0, 'warmup': 1, 'nonfinite': 2,
                     'warmup_ramp': 1, 'clip_saturated': 3}

    def in_warmup(self) -> bool:
        """Whether the LR envelope is still ramping. Public because a sensor may
        need to decline to SAMPLE during warmup, not merely to act."""
        st = self._state()
        return self._ramping(st)

    def _emit(self, st):
        self._report = {
            'lr_ctrl/envelope': st['envelope'],
            'lr_ctrl/peak_scale': st['peak_scale'],
            'lr_ctrl/scale': st['envelope'] * st['peak_scale'],
            # THE RAMP, NOT THE STEP BUDGET. This used to read
            # elapsed < warmup_steps, which reports 1 for the full budget after
            # an early freeze -- so the metric claimed a ramp was running while
            # the envelope was a constant, and disagreed with in_warmup(), the
            # predicate the sensors actually gate on.
            'lr_ctrl/warmup': float(self._ramping(st)),
            'lr_ctrl/ramp_from': self._ramp_from(st),
            'lr_ctrl/divergences': float(self._divergences),
            'lr_ctrl/calibrations': float(self._calibrations),
        }
        # Published only when a cap is configured, so its ABSENCE means "no rail"
        # rather than "rail never bound" -- two states a constant 0.0 could not
        # tell apart. A non-zero value means the servo is asking for a rate the
        # rail is refusing, i.e. peak_scale no longer describes the live rate.
        if self._max_lr() is not None:
            self._report['lr_ctrl/lr_capped_groups'] = float(self._lr_capped_groups)
        # Always published: min_lr has no "off", so unlike the cap there is no
        # absent-means-no-rail state to preserve. Non-zero means the servo is
        # asking for a rate the floor is refusing, i.e. peak_scale has stopped
        # describing the live rate at the bottom as well as the top.
        self._report['lr_ctrl/lr_floored_groups'] = float(self._lr_floored_groups)
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
        #
        # ONE-SHOT, DRAINED BY report(). A reporting period in which the sensor
        # did not fire publishes NO cos/applied/status AT ALL, rather than
        # republishing the last live reading. Measured 2026-08-17 on the qm9
        # conditional route: after the fused branch began returning non-finite
        # gradients at step 902 -- which returns from train.py::step_loss BEFORE
        # the sensor block, so nothing fires -- run `liveservo` emitted the same
        # cos (-0.267178) for 327 consecutive rows with `hypergrads` frozen at
        # 379 and `peak_scale` frozen at 2.0929. A dead sensor read exactly like a
        # working one, and any statistic taken off the channel was then an average
        # over a repeated constant. `hypergrads` is deliberately still published
        # once the sensor has EVER fired, precisely so a flat counter beside an
        # absent cos reads as 'stopped' rather than 'never configured'.
        #
        # AND THE READING IS THE PERIOD, NOT THE LAST FIRING. peak_scale is the
        # integral of every firing in the period, but this channel used to carry
        # only the last one. With fused_grad_accum_min_samples above the batch
        # size the optimizer -- and so the sensor -- steps once per several
        # step_ind, giving ~5 firings per reported row, so the published cos was
        # one sample of five while the actuator moved on all five. hyper_cos is
        # now the MEAN over the period and hyper_applied the TOTAL multiplier, so
        # the sensor channel and the actuator channel describe the same steps.
        w = self._hyper_win
        if self._hypergrads or w is not None:
            self._report['lr_ctrl/hypergrads'] = float(self._hypergrads)
        if w is not None and w['n']:
            live = w['n'] - w['nonfinite']
            if live:
                self._report['lr_ctrl/hyper_cos'] = w['cos_sum'] / live
            self._report['lr_ctrl/hyper_applied'] = math.exp(w['log_applied'])
            self._report['lr_ctrl/hyper_status'] = float(
                self._HYPER_STATUS.get(w['status'], -1))
            # the denominator behind the two averages above, so a period built
            # from one firing and one built from ten are not read alike
            self._report['lr_ctrl/hyper_n'] = float(w['n'])
            # The brake, beside the sensor. A period in which the gate fired is
            # one where peak_scale moved on CLIP evidence and not on cos, so a
            # reader comparing hyper_cos against hyper_applied would otherwise
            # find them inconsistent with no way to see why.
            if w.get('clip_cuts'):
                self._report['lr_ctrl/hyper_clip_cuts'] = float(w['clip_cuts'])
            ema = self._state().get('hyper_clip_ema')
            if ema is not None:
                self._report['lr_ctrl/clip_fire_ema'] = float(ema)
        ceiling = self._current_ceiling()
        if ceiling is not None:
            self._report['lr_ctrl/peak_ceiling'] = ceiling

    def report(self):
        """The metrics for one reporting period. TAKING THE REPORT ENDS THE
        PERIOD: the hypergradient accumulator is drained here, so the next report
        describes what that period did or says nothing. Same one-shot discipline
        train.py uses for `_grad_nonfinite`. The caller must therefore be the
        reporter -- train.py::ten_step_reporting, which runs immediately after
        step_lr_schedule() so `_emit` has already published this period's
        reading. A second caller would silently eat readings."""
        out = dict(self._report)
        self._hyper_win = None
        return out
