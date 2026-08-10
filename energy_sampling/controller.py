import math


class LRController:
    """
    Two-regime LR controller (v7): a per-stage warmup envelope under a peak the
    ALPHA-STAR SERVO owns, plus one coarse divergence bar that reloads and cuts.
    Nothing in between. See docs/to_do_rebuild.md A4/A4a/A5 and decisions.md D4.

    WHAT V7 DELETED, AND WHY. v6 had a third regime between those two -- the cut
    tier, its latch, its hot clock, the recovery ramp and the cut-factor AIMD.
    Every mechanism in it existed to contain a problem the tier itself created
    (module_lr_controller.md 9), it demonstrably could not arrest a live
    explosion (1219ddv9 degraded monotonically THROUGH a 100x cut; s706frkh's
    runaway ran at policy lr 1e-6), and both documented deadlock modes lived
    inside it. It is gone. The scheduled decay leg went with it: alpha* rates
    the PRODUCT peak x envelope(t), so a deterministic multiplier on that
    product is absorbed -- the servo just raises peak to compensate, leaving
    peak inflated against the units its own ceiling is expressed in.

    THE ENVELOPE is now ramp -> hold, forever:
      ramp: blind exponential 1/lr_warmup_ratio -> 1.0 over warmup_steps,
            restarted by rearm_warmup at every stage transition (rebuilt
            optimizers must not land at the operating LR on cold Adam moments).
            It survives the decay deletion on a different warrant: it runs
            BEFORE the servo has any alpha* to act on.
      hold: scale 1.0 thereafter.
    Durations are TRAIN STEPS and the schedule is a pure function of
    step_ind - stage_start_step, so the call cadence cannot change what a
    config value means.

    THE SERVO (adaptive_lr.servo) drives one multiplicative `peak_scale` on the
    policy groups:

        peak_scale <- peak_scale * clip(median(alpha*) / target, clip_lo, clip_hi)

    alpha* is step_probe.py's two-point reading: a dimensionless multiplier on
    the step actually taken. It was designed as a GROWTH signal -- "alpha* > 1
    is affirmative permission to climb", which a breach-only AIMD can never
    have -- and measurement withdrew that half. AS SHIPPED clip_hi is 1.0, so
    the multiplier is <= 1 always and peak_scale can only FALL: the servo is a
    one-sided brake and `seed_lr` is the operating LR, not a starting point.
    See THE SETPOINT below.

    WHICH GROUPS IT DRIVES is a per-key config choice, per the `auto` semantics
    in utils.resolve_derived_config: a key written `auto` is servo-managed
    (seeded at _SERVO_SEED_LR and then owned by the loop); a key written as a
    float is a FIXED peak the servo never touches. So `lr_fused: auto` with
    `lr_back: 2e-4` is a legal and meaningful configuration, and a config with
    no `auto` at all still logs alpha* while actuating nothing -- which is the
    servo's own A/B control arm.

    CEILING WITH FORGETTING. A divergence records ceiling = the peak_scale that
    was running, cut by divergence_cut; growth is damped by distance below it,
    so approach is asymptotic and re-breach is rare. The ceiling then relaxes
    upward with ceiling_halflife_steps -- an LR the surface refused at step 2k
    should not still bind at step 40k. The ceiling is INSTANCE state, never in
    lr_ctrl: the rewind that follows a divergence restores lr_ctrl from a
    healthy checkpoint and would otherwise erase the evidence that this LR just
    detonated (the djr13t0j sawtooth). peak_scale itself IS checkpointed, so a
    resume keeps the climb instead of re-deriving it from the seed, and it is
    clamped to the live ceiling on every read.

    THE SETPOINT: alpha* transfers as a SHAPE, not as a setpoint (lr_aug08,
    2026-08-08; module_lr_controller.md F6/F8).

      * the shape holds. lr x alpha* is constant to ~10% across 26x within one
        run, and pair D got 1.73 against a predicted 1.72 across two.
      * the setpoint does not. Following target 1.0 to its fixed point (3.2e-4,
        alpha_median 1.006 -- the loop tracks perfectly) cost 2.2 nats of
        bwd/tb_err and 2.8 of fwd against a hand-set 1.25e-4.
      * ⚠ and target 1.0 is a POSITIVE FEEDBACK LOOP. b_descend blew fwd/tb_err
        21 -> 35 with alpha_median pinned at ~1.0 throughout and the LR creeping
        3.1e-4 -> 4.5e-4 BECAUSE of the degradation: a flattening surface reads
        as more headroom. bwd/tb_err IMPROVED across the whole collapse, so
        neither branch metric saw it either, and no containment bar fired.

    clip_hi: 1.0 makes that loop impossible by construction rather than
    avoided by tuning. `target` then means "the hottest LR you will tolerate"
    and must sit WELL BELOW the alpha* actually observed at the operating LR --
    a target AT it would ratchet, since SE(windowed median) is ~9% and a
    one-sided clip can never give back what noise takes.

    Restoring the growth servo (clip_hi > 1) needs a guard that is not alpha*
    first: a local ray measurement is structurally blind to a sampling
    distribution collapsing. See decisions.md D32.

    DIVERGENCE (the only actuating bar left) is deliberately coarse: non-finite
    loss or gradient, or either past an absolute ~1e9 bar. Per D4, if we are
    only looking at hard blow-ups then almost any metric works, so this bar
    needs no calibration and must never be tuned into a graduated one. It fires
    BOTH actions together -- train.py reloads the checkpoint AND on_divergence
    cuts the peak -- because a reload without a cut re-enters the same state at
    the same LR and explodes again, while a cut without a reload keeps the
    damaged weights.

    The flow (Z head) LR stays PINNED at lr_flow, exempt from envelope and
    servo alike: alpha* is measured over POLICY parameters only (decision D26
    option b), so the servo has no sensor mandate over Z, and scaling it ran
    the Z head ~20x under design (ylmtpqjy). control_flow_lr: true restores
    uniform scaling for A/B.

    lr_ctrl state ver=7 invalidates v1-v6 checkpoint state.
    """

    CHANNELS = ('fwd', 'bwd', 'replay', 'fused')

    # Policy optimizer keys -> the args attribute holding their base LR. The
    # flow head is deliberately absent: it is pinned, not scheduled.
    _POLICY_BASE = {'fwd': 'lr_policy', 'bwd': 'lr_back', 'replay': 'lr_replay',
                    'fused': 'lr_fused'}

    def __init__(self, modeller):
        self.modeller = modeller
        self._report = {}
        # Ceiling evidence, deliberately INSTANCE state and NOT in lr_ctrl: the
        # rewind that follows a divergence restores lr_ctrl from the 'best'
        # checkpoint, which would erase the record that this LR detonated.
        self._ceiling = None          # peak_scale the servo must not exceed
        self._ceiling_step = None     # when it was recorded (drives forgetting)
        self._divergences = 0
        self._last_servo_step = None
        self._servo_hold_reason = ''
        # has the warmup ramp completed since the last flush? False forces the
        # probe's window to be emptied on the first post-ramp tick, so the servo
        # never acts on readings taken at the envelope-suppressed LR (F7)
        self._warm = False
        self._check_bars()

    # ------------------------------------------------------------------ config

    def _cfg(self, name, default):
        return getattr(getattr(self.modeller.args, 'adaptive_lr', None), name, default)

    def _servo_cfg(self, name, default):
        node = getattr(getattr(self.modeller.args, 'adaptive_lr', None), 'servo', None)
        return getattr(node, name, default) if node is not None else default

    @property
    def servo_enabled(self):
        return bool(self._servo_cfg('enabled', False)) and bool(self._managed_keys())

    def _managed_keys(self):
        """Config keys the servo owns -- the ones written `auto`, recorded by
        resolve_derived_config at load. Empty set = the servo reads and logs
        but actuates nothing, which is its own control arm."""
        return set(getattr(self.modeller.args, 'lr_servo_managed', ()) or ())

    def _check_bars(self):
        """Divergence bars are a sanity floor, not a calibration. Refuse a bar
        low enough to fire on ordinary training: a graduated divergence bar is
        the cut tier coming back in through the config."""
        for name in ('divergence_loss_abs', 'divergence_grad_abs'):
            bar = self._cfg(name, None)
            if bar is None:
                continue
            bar = float(bar)
            if bar < 1.0e5:
                raise ValueError(
                    f"adaptive_lr.{name} = {bar:g} is too low to be a divergence bar. "
                    f"This tier reloads the checkpoint and discards progress; it exists "
                    f"for non-finite values and ~1e9 blow-ups only. A bar tuned to fire "
                    f"on merely-hot training is the deleted cut tier by another name.")
        print(f"lr_ctrl v7: divergence bars loss={self._cfg('divergence_loss_abs', None)} "
              f"grad={self._cfg('divergence_grad_abs', None)}  |  servo "
              f"{'ON ' + ','.join(sorted(self._managed_keys())) if self.servo_enabled else 'off'}")

    # -------------------------------------------------------------- divergence

    def check_spike(self, step_type, current_loss, grad_norm):
        """The one remaining tripwire. Returns 'diverged' or None.

        Non-finite readings, or a finite reading past an absolute ~1e9 bar.
        No cooldown, no latch, no per-channel memory: at this bar a second
        reading is a second explosion, and train.py's max_reloads cap is what
        stops a rewind loop. Always on -- there is no flag that disables it."""
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
        """Cut the servo peak and record the ceiling. Called by train.py's
        fire_loss_spike AFTER the rewind, for both the tripwire and the
        terminal-policy path; `count` compounds the cut across repeated
        terminal rewinds so the ceiling actually descends across policy deaths
        instead of sawtoothing.

        Being instance state, the ceiling survives the rewind that precedes
        this call. If the servo is off this still runs -- the ceiling is then
        simply the record, and the LRs are whatever the config fixed them at."""
        cut = float(self._cfg('divergence_cut', 0.5)) ** max(int(count), 1)
        st = self._state()
        # recompute rather than trust the stored value: this is called from
        # fire_loss_spike, which runs AFTER a checkpoint restore has overwritten
        # lr_ctrl wholesale, so st['envelope'] here can be the saved checkpoint's
        # and not this step's. _apply_lrs below multiplies by it.
        st['envelope'] = self._envelope(st)
        lo, hi = self._peak_bounds()
        st['peak_scale'] = max(lo, min(hi, float(st['peak_scale']) * cut))
        self._ceiling = st['peak_scale']
        self._ceiling_step = int(self.modeller.step_ind)
        print(f"lr_ctrl: peak_scale -> {st['peak_scale']:.4g} (ceiling recorded)")
        self._apply_lrs(st)

    # ------------------------------------------------------------------ actuator

    def _peak_bounds(self):
        b = self._servo_cfg('bounds', (0.02, 20.0))
        return float(b[0]), float(b[1])

    def _apply_lrs(self, st):
        """lr = base x peak_scale x envelope, floored at min_lr -- EXCEPT the
        flow (Z head) groups, pinned flat at lr_flow, and except groups whose
        base LR was configured as an explicit float, which the servo does not
        own (peak_scale does not apply to them; the envelope still does)."""
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
                    base_key = 'lr_flow'   # only reachable under control_flow_lr
                else:
                    base_key = self._POLICY_BASE[key]
                base = getattr(a, base_key)
                scale = env * (peak if base_key in managed else 1.0)
                g['lr'] = max(a.min_lr, base * scale)

    # ------------------------------------------------------------------ state

    def _fresh_state(self, phase):
        return {
            'ver': 7,  # v7 = servo peak + coarse divergence -- invalidates v1-v6
            'phase_seen': phase,
            'stage_start_step': int(self.modeller.step_ind),
            'peak_scale': 1.0,
            'envelope': 1.0 / self.modeller.args.lr_warmup_ratio,
        }

    def _state(self):
        m = self.modeller
        st = getattr(m, 'lr_ctrl', None)
        if not isinstance(st, dict) or st.get('ver') != 7:
            st = self._fresh_state(m.phase)
            m.lr_ctrl = st
        elif st.get('phase_seen') != m.phase:
            # stage transitions run rearm_warmup, which owns the reset; this
            # branch only catches a rewind restoring state from another phase
            st['phase_seen'] = m.phase
        if st.get('stage_start_step', 0) > m.step_ind:
            # a restore stamped a start ahead of the live clock -- re-anchor
            st['stage_start_step'] = int(m.step_ind)
        st.setdefault('peak_scale', 1.0)
        st.setdefault('envelope', 1.0)
        return st

    def rearm_warmup(self):
        """Protocol.advance hook at EVERY stage transition: restart the warmup
        clock (the optimizers were just rebuilt onto a loss surface with
        different curvature) and forget the ceiling -- the old stage's
        divergence evidence describes a surface that no longer exists.

        peak_scale is deliberately CARRIED across the transition. It is the
        servo's accumulated estimate of the right LR for this problem, and
        re-deriving it from the seed at every stage would spend thousands of
        steps re-climbing ground alpha* has already covered. The ceiling reset
        is what lets it climb again if the new surface allows it.

        Returns the warmup length in TRAIN STEPS."""
        m = self.modeller
        st = self._state()
        st['phase_seen'] = m.phase
        self._ceiling = None
        self._ceiling_step = None
        self._last_servo_step = None
        # the incoming stage re-ramps, so the probe's buffered readings describe
        # both the old stage's loss surface AND a pre-ramp LR -- flushed here as
        # well as at the ramp's end, since a transition changes both at once
        self._warm = False
        probe = getattr(m, 'step_probe', None)
        if probe is not None and hasattr(probe, 'flush_window'):
            probe.flush_window()
        st['stage_start_step'] = int(m.step_ind)
        st['envelope'] = self._envelope(st)
        self._apply_lrs(st)
        return int(self._cfg('warmup_steps', 1000))

    # ------------------------------------------------------------------ envelope

    def _elapsed(self, st):
        """TRAIN STEPS since the current stage's schedule started."""
        return max(0, int(self.modeller.step_ind) - int(st.get('stage_start_step', 0)))

    def _envelope(self, st):
        """Ramp -> hold, in [1/lr_warmup_ratio, 1.0]. No decay leg (A4a)."""
        warmup_steps = max(1, int(self._cfg('warmup_steps', 1000)))
        elapsed = self._elapsed(st)
        if elapsed >= warmup_steps:
            return 1.0
        frac = elapsed / warmup_steps
        return (1.0 / self.modeller.args.lr_warmup_ratio) ** (1.0 - frac)

    # --------------------------------------------------------------------- servo

    def _current_ceiling(self):
        """The recorded ceiling, relaxed upward with its forgetting half-life.
        None = never breached, so growth is undamped."""
        if self._ceiling is None:
            return None
        half = float(self._servo_cfg('ceiling_halflife_steps', 20000.0) or 0.0)
        if half <= 0:
            return self._ceiling
        dt = max(0, int(self.modeller.step_ind) - int(self._ceiling_step or 0))
        return min(self._peak_bounds()[1], self._ceiling * (2.0 ** (dt / half)))

    def _advance_servo(self, st):
        """One servo tick. Holds (and says why) rather than guessing whenever
        the sensor is not reporting something it is entitled to act on."""
        self._servo_hold_reason = ''
        if not self.servo_enabled:
            self._servo_hold_reason = 'disabled'
            return
        probe = getattr(self.modeller, 'step_probe', None)
        if probe is None or not getattr(probe, 'enabled', False):
            # A servo whose sensor was never switched on is the exact failure
            # this codebase keeps rediscovering -- an unreadable sensor and a
            # satisfied controller produce identical silence. resolve_derived_
            # config refuses this combination at load; the guard is here because
            # the servo must not act on a sensor it cannot see either way.
            self._servo_hold_reason = 'no_probe'
            return
        # HOLD THROUGH WARMUP. The envelope is below 1 there, so alpha* rates a
        # deliberately shrunken step and reads high; acting on it would inflate
        # peak_scale by exactly the warmup factor and then hand that back as a
        # real LR the moment the ramp completes.
        if self._elapsed(st) < int(self._cfg('warmup_steps', 1000)):
            self._servo_hold_reason = 'warmup'
            self._warm = False
            return
        if not self._warm:
            # FIRST tick after the ramp. Holding the servo was not enough: the
            # probe kept BUFFERING through warmup, and its window is 500 train
            # steps deep at the defaults, so the median still describes the
            # envelope-suppressed LR. alpha* ~ 1/lr, so those readings are biased
            # high by exactly the warmup factor and the first tick climbs on
            # them. Measured overshoot before this flush: 34%, in the wrong
            # direction, on lr_aug08 b_descend.
            self._warm = True
            probe.flush_window()
            self._servo_hold_reason = 'cold'
            return
        period = max(1, int(self._servo_cfg('period', 200)))
        step = int(self.modeller.step_ind)
        if self._last_servo_step is not None and step - self._last_servo_step < period:
            return
        reading = probe.servo_reading()
        if reading is None:
            self._servo_hold_reason = 'cold'
            return
        median, n, bad_rate = reading
        if n < int(self._servo_cfg('min_readings', 10)):
            self._servo_hold_reason = 'few_readings'
            return
        # A rising flat/downward rate voids the sensor independently of what
        # the alpha* values say (to_do_rebuild A3a.3): a downward-opening fit
        # means the local quadratic model is simply wrong, and a flat one means
        # the probe is under-resolved. Either way the median is not a reading.
        if bad_rate > float(self._servo_cfg('max_bad_rate', 0.5)):
            self._servo_hold_reason = 'fit_invalid'
            return
        self._last_servo_step = step

        target = float(self._servo_cfg('target', 1.0))
        clip = self._servo_cfg('clip', (0.7, 1.5))
        clip_lo, clip_hi = float(clip[0]), float(clip[1])
        raw = min(max(median / max(target, 1e-9), clip_lo), clip_hi)
        ceiling = self._current_ceiling()
        if raw > 1.0 and ceiling is not None:
            # Asymptotic approach: growth is proportional to the fraction of
            # the gap to the ceiling still unspent, so the servo slows as it
            # returns to a level that has already blown up once instead of
            # walking straight back into it.
            room = max(0.0, 1.0 - float(st['peak_scale']) / max(ceiling, 1e-12))
            raw = 1.0 + (raw - 1.0) * room
        lo, hi = self._peak_bounds()
        if ceiling is not None:
            hi = min(hi, ceiling)
        st['peak_scale'] = max(lo, min(hi, float(st['peak_scale']) * raw))

    # ------------------------------------------------------------------ tick

    def step(self):
        """One controller evaluation (called every 10 train steps from
        step_lr_schedule; the envelope is keyed on step_ind, so the call
        cadence only sets how often LRs are re-stamped). Returns the applied
        fwd LR, mirroring the legacy path."""
        m = self.modeller
        st = self._state()
        st['envelope'] = self._envelope(st)
        self._advance_servo(st)
        ceiling = self._current_ceiling()
        if ceiling is not None:
            st['peak_scale'] = min(float(st['peak_scale']), ceiling)
        self._apply_lrs(st)
        self._emit(st)
        return m.optimizers['fwd'].param_groups[0]['lr']

    def _emit(self, st):
        self._report = {
            'lr_ctrl/envelope': st['envelope'],
            'lr_ctrl/peak_scale': st['peak_scale'],
            'lr_ctrl/scale': st['envelope'] * st['peak_scale'],
            'lr_ctrl/warmup': float(self._elapsed(st) < int(self._cfg('warmup_steps', 1000))),
            'lr_ctrl/divergences': self._divergences,
        }
        # Emit the ACTUATOR alongside the sensor, always. A servo that is
        # holding and a servo that is satisfied are indistinguishable from
        # peak_scale alone, which is the general failure this doc set records
        # three separate instances of.
        holds = ('', 'disabled', 'no_probe', 'warmup', 'cold', 'few_readings', 'fit_invalid')
        self._report['lr_ctrl/servo_hold'] = float(holds.index(self._servo_hold_reason)
                                                   if self._servo_hold_reason in holds else 0)
        ceiling = self._current_ceiling()
        if ceiling is not None:
            self._report['lr_ctrl/peak_ceiling'] = ceiling

    def report(self):
        return dict(self._report)
