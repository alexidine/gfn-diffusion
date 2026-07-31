import math


class LRController:
    """
    Fixed-peak LR controller (v6): a deterministic ramp -> hold -> decay
    schedule under a FIXED peak (scale never exceeds 1.0 = the configured
    base LRs), plus two-tier ABSOLUTE tripwires. Replaces the v5 adaptive
    (probe/coherence) controller wholesale -- see git history. The probe
    measured update AGREEMENT, which stayed honestly positive straight into
    detonations (aijrfwuy scale-1.94, s706frkh scale-1.46), and the climb it
    licensed manufactured the very breaches the balance layer then spent
    thousands of steps repairing. There is no climb any more: the configured
    LR is the peak, full stop.

    SCHEDULE (ALL durations in TRAIN STEPS -- a pure function of
    step_ind - stage_start_step, so the controller's own call cadence
    (currently every 10 steps) can never change what a config value means;
    restarted by rearm_warmup at every stage transition, so each stage runs
    its own ramp/hold/decay):
      ramp:   blind exponential 1/lr_warmup_ratio -> 1.0 over warmup_steps
              (rebuilt optimizers must not land at the full operating LR on
              cold Adam moments)
      hold:   scale 1.0 for hold_steps
      decay:  exponential toward decay_floor_scale with half-life
              decay_halflife_steps (0/null = hold at 1.0 forever). Decay buys
              back the late-run precision the LR jitter floor otherwise caps
              (weight jitter at the operating LR bounds attainable wass/r2).
    adaptive_lr.enabled: false pins scale flat at 1.0 -- no ramp, no decay.
    FIRE runs regardless of the flag.

    FIRE (two tiers, event-triggered, ABSOLUTE bars -- no medians, no
    relative baselines. The old floored-median bars failed two-sided in
    s706frkh: with the median riding the floor they fired on a
    clip-neutralized grad 745 -- the applied update was identical to any
    at-clip step -- and once the incident's own 1e4 norms raised the median
    they went blind exactly when the real excursion began):
      cut tier   (cut_loss_abs / cut_grad_abs): parameter thrash -- LR cut
                 only, no rewind. Training state is intact, just too hot.
      reset tier (reset_loss_abs / reset_grad_abs, or a non-finite reading):
                 true explosion (tb err at +1e4 scale) -- train.py routes
                 this to fire_loss_spike's rewind-to-best + cut. Stale best
                 weights against fresher live buffers re-synchronize quickly
                 and are an accepted cost here.
    Cuts multiply an INSTANCE-held _cut_factor -- deliberately NOT in
    lr_ctrl, which the rewind restores from a healthy checkpoint, erasing
    any evidence kept there that this LR already detonated (the djr13t0j
    sawtooth). The factor resets at stage transitions: optimizers are
    rebuilt onto a new loss surface and the old stage's fire evidence does
    not transfer.

    LATCH: a fire disarms its channel's CUT tier until the metric has
    recrossed BELOW the cut bar (still pure level logic -- no trends, no
    baselines). An excursion's decay tail re-tripping the absolute bar at
    every cooldown expiry is the same single event, and cutting again does
    not drain it faster (g8d8se26: the first cut did the work; four tail
    fires on a 17k->1k grad drain dug the factor from 0.25 to the 0.01
    floor). The reset tier and non-finite readings ignore the latch -- an
    excursion that ESCALATES past the reset bar is a new fact and must not
    be blind-windowed (which is also why the fix is a latch and not a
    longer cooldown: the cooldown suppresses ALL finite readings on the
    channel, reset tier included). If the metric NEVER recrosses -- a
    sustained simmer between the bars -- the cut tier stays disarmed, on
    purpose: repeat cuts demonstrably do not drain a sustained excursion
    (1219ddv9 degraded monotonically through a 100x cut; the s706frkh
    runaway ran at policy lr 1e-6), so the stuck state belongs to the
    reset tier and _terminal_policy_state. What a simmer must NOT do is
    let recovery re-ramp into it -- see the hot clock below.

    RECOVERY (recovery_target_frac > 0): a fire records the cut factor
    that was running when its episode started -- the measured ceiling.
    Once the system has been COLD (every monitored reading below its cut
    bar, no fires) for recovery_wait_steps, the factor re-ramps
    exponentially over recovery_ramp_steps toward recovery_target_frac x
    that ceiling, then cruises there (the schedule envelope stays
    authoritative on top, so ramp/decay behave normally). The wait clock
    runs from the last HOT reading, not the last fire: a latched channel
    simmering between the bars fires nothing, and recovery must not raise
    the LR back into a still-hot state whose cut tier the latch has
    disarmed -- if the metric never recrosses, recovery never starts. A
    hot reading during the ramp pauses it in place (anchor resets, clock
    pushes out) without needing a fire. Episode grouping uses the same
    clock: a fire after a full cold wait fired at a level recovery
    deliberately returned to -- new evidence, the ceiling re-records and
    the cruise target ratchets down (AIMD) instead of sawtoothing;
    fires inside an ongoing episode keep the original ceiling.
    Without recovery a cut is permanent for the stage: g8d8se26 spent 24k
    of its 27k steps pinned at the 0.01 floor with grads 30-50x under the
    bar, still slowly improving -- pure LR starvation.

    The flow (Z head) LR stays PINNED at lr_flow, exempt from scaling -- the
    schedule has no sensor mandate over Z, and scaling it ran the Z head
    ~20x under design (ylmtpqjy). control_flow_lr: true restores uniform
    scaling for A/B.
    """

    CHANNELS = ('fwd', 'bwd', 'replay', 'fused')

    def __init__(self, modeller):
        self.modeller = modeller
        self._report = {}
        # fire memory, deliberately INSTANCE state and NOT in lr_ctrl: the
        # rewind that follows a reset-tier fire restores lr_ctrl from the
        # 'best' checkpoint, which would erase any evidence kept there.
        self._cut_factor = 1.0
        self._fire_counts = {}
        self._channel_cooldown_until = {}
        # per-channel cut-tier latch + fire memory for the recovery ramp;
        # instance state for the same rewind-proofness reason as _cut_factor
        self._latched = set()
        self._last_fire_step = None
        self._last_hot_step = None   # last reading at/above any cut bar (or non-finite)
        self._pre_trigger_cold = True  # was there a >= recovery_wait cold gap before this tick's readings
        self._fire_cut_factor = None
        self._recovery_anchor = None

    @property
    def enabled(self):
        cfg = getattr(self.modeller.args, 'adaptive_lr', None)
        return cfg is not None and getattr(cfg, 'enabled', False)

    def _cfg(self, name, default):
        return getattr(getattr(self.modeller.args, 'adaptive_lr', None), name, default)

    # ------------------------------------------------------------- fire (spikes)

    def check_spike(self, step_type, current_loss, grad_norm):
        """Absolute two-tier tripwires feeding train.py's monitor_losses.
        Returns None, 'cut' (thrash: LR cut only), or 'reset' (true
        explosion: rewind + cut). Always on regardless of adaptive_lr.enabled.

        A finite fire arms fire_cooldown_steps on its channel; further finite
        fires on that channel are suppressed until it expires, so a sustained
        excursion (or the post-rewind re-sync transient) can't machine-gun
        cuts/rewinds at the check cadence (the s706frkh 18-fire loop ran at
        exactly the old cooldown period). Non-finite readings ignore the
        cooldown -- NaN weights are not going to re-sync on their own."""
        step = self.modeller.step_ind
        checks = []
        if current_loss is not None and step_type in self.CHANNELS:
            checks.append((f'{step_type}_loss', float(current_loss),
                           self._cfg('cut_loss_abs', 1.0e3),
                           self._cfg('reset_loss_abs', 1.0e4)))
        if grad_norm is not None:
            checks.append(('grad_norm', float(grad_norm),
                           self._cfg('cut_grad_abs', 3.0e3),
                           self._cfg('reset_grad_abs', 3.0e4)))

        cooldown = int(self._cfg('fire_cooldown_steps', 200))
        # episode grouping for on_explosion: was the system cold (all readings
        # below the cut bars) for a full recovery_wait before this tick? Uses
        # the PRE-tick hot clock, because the triggering reading is itself hot.
        wait = int(self._cfg('recovery_wait_steps', 5000))
        self._pre_trigger_cold = (self._last_hot_step is None
                                  or step - int(self._last_hot_step) >= wait)
        tier = None
        for channel, value, cut_bar, reset_bar in checks:
            if (math.isfinite(value) and cut_bar is not None
                    and value < float(cut_bar)):
                # healthy reading: the excursion has drained -- re-arm the latch
                self._latched.discard(channel)
                continue
            if not math.isfinite(value):
                this, bar = 'reset', float('nan')
            elif reset_bar is not None and value >= float(reset_bar):
                this, bar = 'reset', float(reset_bar)
            elif cut_bar is not None and value >= float(cut_bar):
                this, bar = 'cut', float(cut_bar)
            else:
                continue
            self._last_hot_step = step
            if math.isfinite(value) and step < self._channel_cooldown_until.get(channel, -1):
                continue
            if this == 'cut' and channel in self._latched:
                # decay tail of an already-cut excursion (still between the
                # bars): one event, one cut. Reset tier stays armed above.
                continue
            self._channel_cooldown_until[channel] = step + cooldown
            self._latched.add(channel)
            self._fire_counts[channel] = self._fire_counts.get(channel, 0) + 1
            print(f"lr_ctrl tripwire FIRED [{this}]: {channel} value {value:.4g} "
                  f">= bar {bar:.4g} (absolute), fire #{self._fire_counts[channel]} "
                  f"on this channel (step {step})")
            if this == 'reset':
                tier = 'reset'
            elif tier is None:
                tier = 'cut'
        return tier

    def reset_spike_monitors(self, names):
        """Stage-transition hook (protocol.StageProtocol.advance): arm a fire
        cooldown on every channel. A transition can cause transient
        turbulence anywhere (rebuilt optimizers, new loss definitions), and
        that turbulence must not eat a fire."""
        step = self.modeller.step_ind
        cooldown = int(self._cfg('fire_cooldown_steps', 200))
        for name in names:
            self._channel_cooldown_until[f'{name}_loss'] = step + cooldown
        self._channel_cooldown_until['grad_norm'] = step + cooldown

    def on_explosion(self, count: int = 1):
        """Cut hook, called for BOTH tiers (cut tier directly from
        monitor_losses; reset tier after fire_loss_spike's rewind). Multiplies
        the instance-held cut factor by cut_ratio**count -- count > 1 only
        from repeated TERMINAL rewinds (train.py terminal_reloads), which
        compound so the ceiling actually descends across policy deaths.
        Being instance state, the factor survives the rewind that typically
        precedes this call."""
        m = self.modeller
        ratio = float(self._cfg('cut_ratio', 0.5))
        floor = m.args.min_lr / m.args.lr_policy
        # fire memory for the recovery ramp: the cut factor RUNNING at the
        # START of a fire episode is the measured ceiling. An episode begins
        # with a hot reading after a full recovery_wait of cold ones
        # (_pre_trigger_cold, stamped by the check_spike call that triggered
        # this) -- fires inside an ongoing episode (draining tail past the
        # latch on another channel, sustained simmer escalating to reset,
        # NaN storm) happened at an already-cut level that proves nothing
        # about the ceiling, so they keep the recorded level. A new-episode
        # fire fired at a level recovery deliberately returned to -- new
        # evidence, re-record, and the cruise target ratchets down (AIMD).
        if self._pre_trigger_cold or self._fire_cut_factor is None:
            self._fire_cut_factor = self._cut_factor
        self._last_fire_step = int(m.step_ind)
        self._recovery_anchor = None
        self._cut_factor = max(self._cut_factor * ratio ** max(count, 1), floor)
        print(f"lr_ctrl: cut factor -> {self._cut_factor:.4g}")
        st = self._state()
        st['scale'] = self._schedule_scale(st) * self._cut_factor
        self._apply_lrs(st)

    # ------------------------------------------------------------------ actuator

    def _apply_lrs(self, st):
        """lr = configured base x scale per group, floored at min_lr -- EXCEPT
        the flow (Z head) groups, pinned flat at lr_flow. See class docstring."""
        m = self.modeller
        a = m.args
        control_flow = self._cfg('control_flow_lr', False)
        for key, opt in m.optimizers.items():
            n_groups = len(opt.param_groups)
            for gi, g in enumerate(opt.param_groups):
                if key == 'fused':
                    base = a.lr_flow if gi == n_groups - 1 else a.lr_fused
                else:
                    base = {'fwd': a.lr_policy, 'bwd': a.lr_back,
                            'replay': a.lr_replay, 'flow': a.lr_flow}[key]
                is_flow_group = key == 'flow' or (key == 'fused' and gi == n_groups - 1)
                if is_flow_group and not control_flow:
                    g['lr'] = base
                else:
                    g['lr'] = max(a.min_lr, base * st['scale'])

    # ------------------------------------------------------------------ state

    def _fresh_state(self, phase):
        return {
            'ver': 6,  # v6 = fixed-peak schedule -- invalidates v1-v5 state
            'phase_seen': phase,
            'stage_start_step': int(self.modeller.step_ind),
            'scale': 1.0 / self.modeller.args.lr_warmup_ratio,
        }

    def _state(self):
        m = self.modeller
        st = getattr(m, 'lr_ctrl', None)
        if not isinstance(st, dict) or st.get('ver') != 6:
            st = self._fresh_state(m.phase)
            m.lr_ctrl = st
        elif st.get('phase_seen') != m.phase:
            # stage transitions run rearm_warmup, which owns the reset; this
            # branch only catches a rewind restoring state from another phase
            st['phase_seen'] = m.phase
        if st.get('stage_start_step', 0) > m.step_ind:
            # a restore stamped a start ahead of the live clock -- re-anchor
            st['stage_start_step'] = int(m.step_ind)
        return st

    def rearm_warmup(self):
        """Protocol.advance hook at EVERY stage transition: restart the
        ramp/hold/decay clock (the optimizers were just rebuilt onto a loss
        surface with different curvature, so the first steps must not land at
        the full operating LR with empty Adam moments) and forgive
        accumulated cuts -- the old stage's fire evidence describes a surface
        that no longer exists. Returns the warmup length in TRAIN STEPS."""
        m = self.modeller
        st = self._state()
        st['phase_seen'] = m.phase
        self._cut_factor = 1.0
        self._latched.clear()
        self._last_fire_step = None
        self._last_hot_step = None
        self._pre_trigger_cold = True
        self._fire_cut_factor = None
        self._recovery_anchor = None
        if not self.enabled:
            return  # disabled: LRs sit flat at configured values, nothing to ramp
        st['stage_start_step'] = int(m.step_ind)
        st['scale'] = self._schedule_scale(st)
        self._apply_lrs(st)
        return int(self._cfg('warmup_steps', 1000))

    # ------------------------------------------------------------------ schedule

    def _elapsed(self, st):
        """TRAIN STEPS since the current stage's schedule started."""
        return max(0, int(self.modeller.step_ind) - int(st.get('stage_start_step', 0)))

    def _schedule_scale(self, st):
        """Pure function of train steps since stage start: ramp -> hold ->
        decay, in [~0, 1.0]. Never above 1.0 -- the configured LR is the peak."""
        if not self.enabled:
            return 1.0
        warmup_steps = max(1, int(self._cfg('warmup_steps', 1000)))
        elapsed = self._elapsed(st)
        if elapsed < warmup_steps:
            frac = elapsed / warmup_steps
            return (1.0 / self.modeller.args.lr_warmup_ratio) ** (1.0 - frac)
        half = self._cfg('decay_halflife_steps', 0) or 0
        past = elapsed - warmup_steps - int(self._cfg('hold_steps', 20000))
        if half <= 0 or past <= 0:
            return 1.0
        return max(float(self._cfg('decay_floor_scale', 0.05)),
                   0.5 ** (past / float(half)))

    # ---------------------------------------------------------------- recovery

    def _advance_recovery(self):
        """Re-ramp the cut factor after a quiet period (see class docstring).
        recovery_target_frac <= 0 (the default) disables recovery entirely --
        a cut then stays for the stage, the pre-recovery behavior."""
        frac = float(self._cfg('recovery_target_frac', 0.0))
        if frac <= 0 or self._last_fire_step is None or self._fire_cut_factor is None:
            return
        # A cut lands at cut_ratio x the pre-fire factor, and the recovery target
        # is frac x that SAME factor -- so frac <= cut_ratio means the cut always
        # lands at or below the target and the `>=` below returns immediately.
        # Recovery is then dead code for every FIRST fire in a stage, silently,
        # which is what shipped (both were 0.5). Warn once rather than let a
        # whole subsystem be disabled by an unremarkable-looking coefficient.
        ratio = float(self._cfg('cut_ratio', 0.5))
        if frac <= ratio and not getattr(self, '_recovery_inert_warned', False):
            self._recovery_inert_warned = True
            print(f"lr_ctrl WARNING: recovery_target_frac ({frac}) <= cut_ratio "
                  f"({ratio}) -- recovery can never raise the cut factor, so the "
                  f"recovery_wait/recovery_ramp/AIMD path is inert. Set "
                  f"recovery_target_frac > cut_ratio to enable it.")
        target = min(1.0, frac * self._fire_cut_factor)
        if self._cut_factor >= target:
            return
        step = int(self.modeller.step_ind)
        # the wait clock runs from the last HOT reading (any value at/above a
        # cut bar), not just the last fire: a latched channel simmering
        # between the bars fires nothing, and ramping the LR back up into a
        # still-hot state -- with its cut tier disarmed -- must not happen.
        # If the metric never recrosses, recovery simply never starts.
        last_event = int(self._last_fire_step)
        if self._last_hot_step is not None:
            last_event = max(last_event, int(self._last_hot_step))
        if step - last_event < int(self._cfg('recovery_wait_steps', 5000)):
            self._recovery_anchor = None
            return
        if self._recovery_anchor is None:
            self._recovery_anchor = (step, self._cut_factor)
        a_step, a_val = self._recovery_anchor
        ramp = max(1, int(self._cfg('recovery_ramp_steps', 1000)))
        prog = min(1.0, (step - a_step) / ramp)
        self._cut_factor = min(target, a_val * (target / a_val) ** prog)

    # ------------------------------------------------------------------ tick

    def step(self):
        """One controller evaluation (called every 10 train steps from
        step_lr_schedule; the schedule itself is keyed on step_ind, so the
        call cadence only sets how often LRs are re-stamped). Returns the
        applied fwd LR, mirroring the legacy path."""
        m = self.modeller
        st = self._state()
        self._advance_recovery()
        st['scale'] = self._schedule_scale(st) * self._cut_factor
        self._apply_lrs(st)
        in_warmup = self.enabled and self._elapsed(st) < int(self._cfg('warmup_steps', 1000))
        self._emit(st, warmup=in_warmup)
        return m.optimizers['fwd'].param_groups[0]['lr']

    def _emit(self, st, warmup):
        self._report = {
            'lr_ctrl/scale': st['scale'],
            'lr_ctrl/warmup': float(warmup),
            'lr_ctrl/cut_factor': self._cut_factor,
        }
        for channel, n in self._fire_counts.items():
            self._report[f'lr_ctrl/fires_{channel}'] = n

    def report(self):
        return dict(self._report)
