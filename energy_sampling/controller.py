import math


class LRController:
    """
    LR controller v8: a warmup envelope under a peak set by PERIODIC RAY
    CALIBRATION, plus one coarse divergence bar. Nothing else.

        lr = base_lr x peak_scale x envelope(t)

    `envelope` is a fixed warmup ramp, restarted at every stage transition.
    `peak_scale` is moved only by on_calibration(), at most once per
    ray_calibration.period steps.

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
        # HOLD THROUGH WARMUP. The envelope is deliberately below 1 here, so the
        # step just measured is a scheduled fraction of the operating step and
        # alpha* rates THAT. peak_scale is defined as the multiplier on the
        # un-suppressed rate, so acting on this reading would inflate it by
        # exactly the warmup factor and hand that back as a real LR the moment
        # the envelope releases -- a jump of lr_warmup_ratio, all at once. The
        # sensor cannot anticipate a suppression it is measuring through.
        if self._elapsed(st) < int(self._cfg('warmup_steps', 1000)):
            self._last['status'] = 'warmup'
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

    # -------------------------------------------------------------------- state

    def _fresh_state(self, phase):
        return {
            'ver': self._STATE_VER,
            'phase_seen': phase,
            'stage_start_step': int(self.modeller.step_ind),
            'peak_scale': 1.0,
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

        peak_scale is deliberately CARRIED across the transition -- it is the
        accumulated estimate of the right LR for this problem, and re-deriving it
        from the seed each stage would spend thousands of steps re-climbing.

        Returns the warmup length in TRAIN STEPS."""
        m = self.modeller
        st = self._state()
        st['phase_seen'] = m.phase
        self._ceiling = None
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
        NOT move peak_scale -- only on_calibration and on_divergence do."""
        st = self._state()
        st['envelope'] = self._envelope(st)
        ceiling = self._current_ceiling()
        if ceiling is not None:
            st['peak_scale'] = min(float(st['peak_scale']), ceiling)
        self._apply_lrs(st)
        self._emit(st)
        return self.modeller.optimizers['fwd'].param_groups[0]['lr']

    _STATUS = {'unresolved': 0, 'bracketed': 1, 'above_range': 2,
               'below_range': 3, 'inconsistent': 4, 'warmup': 5}

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
        ceiling = self._current_ceiling()
        if ceiling is not None:
            self._report['lr_ctrl/peak_ceiling'] = ceiling

    def report(self):
        return dict(self._report)
