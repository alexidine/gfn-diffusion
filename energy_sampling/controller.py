import math

import numpy as np


class AdaptiveLRController:
    """
    Safety-net LR controller (v4): exponential warmup to a configured operating
    LR, then HOLD there and only ever CUT on broad, sustained reward-space
    damage. Optionally a very gentle health-gated climb explores headroom above
    the configured LR. Replaces the v1-v3 probe/cruise "find the edge by
    climbing" controller, whose complexity (~30 params) and untrustworthiness
    all lived in the optimum-SEARCH; this one never searches -- you supply the
    operating LR (lr_policy et al., now the operating point, not a ceiling) from
    domain knowledge, and the controller's only job is to keep the model from
    being scrambled by it.

    Why it survives this model's specific weirdness:
      - health = the Z-INVARIANT reward channels (scatter_err, slope_err over
        fwd/bwd/replay), immune to the log Z re-pricing / lag artifacts that
        fooled every level-sensitive signal;
      - reward-space, because the always-binding grad clip + Adam launder LR
        damage OUT of parameter space (bwd/tb reverses while parameter-space
        thrash keeps falling -- run 8k2y2bvm) so a parameter-noise probe is
        structurally blind to the real failure;
      - the cut fires on the FRACTION of channels breaching, not any single one:
        LR damage is GLOBAL (random parameter noise degrades every channel
        together) while mode addition is LOCAL (only channels touching the new
        mode degrade), so "most channels breaching together" cleanly separates
        damage from purposeful mode-growth churn -- the confounder is the
        discriminator. The fraction is taken over the channels that could
        actually vote (see _channels): a dormant or saturated channel is an
        abstention, not a NO, and counting it in the denominator is what
        silently disarmed the controller in tngticqq;
      - safety-net-only: it can never strand LR low the way the probe did
        (warmup lands at the configured operating point, and recovery climbs
        back to it after any cut), and it can never scramble fast (the optional
        headroom climb is additive, health-gated, and soft-ceilinged at the
        last cut).

    Phase-agnostic in FORM (scatter/slope mean the same thing in every phase --
    spread of the TB residual, Z-invariant -- so there is no phase-1 special
    case), but not in MEMBERSHIP: which branches are live is a phase/mode-frac
    fact, so the channel set is recomputed every tick rather than fixed at 6.
    A phase change re-seeds the channel bests and cools down (the loss levels
    jump) but keeps the running scale -- a phase change is not an LR event, so
    no re-warmup.

    Config knobs you actually tune: lr_policy/lr_back/lr_replay/lr_fused (the
    operating LR), breach_fraction (a fraction of the LIVE channels -- 0.66 is
    4/6 with all three branches training, 2/3 with only fwd+replay live; note
    it is a >= test on a ratio, so 0.67 does NOT mean 4/6, it means 5/6),
    channel_margin.
    Everything else has a sensible default. Enabled iff adaptive_lr.enabled;
    when disabled, step_lr_schedule runs the legacy warmup/anneal (revert).

    The flow (Z-head) LR is PINNED at lr_flow, exempt from scaling: the health
    is Z-shift-invariant so the controller has no sensor mandate over Z, and
    scaling it down with the policy ran the Z head ~20x under design (ylmtpqjy).
    Z-head explosions stay covered by the loss-spike monitors.
    """

    # scatter_err (residual std, mean-removed) and slope_err (|slope-1| from
    # centered covariances) are both Z-shift invariant by construction.
    CHANNEL_METRICS = ('scatter_err', 'slope_err')
    CHANNEL_DIRS = ('fwd', 'bwd', 'replay')

    # Metrics that saturate: past this value the channel is already maximally
    # damaged and carries no further information (slope_err = |slope-1|, so a
    # fully decorrelated fit sits at ~1.0 and cannot rise further). A channel
    # whose running-best is seeded at/near its ceiling can never satisfy
    # best x channel_margin, so it is silently unbreachable -- see _channels.
    # scatter_err is unbounded above and has no entry.
    CHANNEL_CEILING = {'slope_err': 1.0}

    def __init__(self, modeller):
        self.modeller = modeller
        self._report = {}

    @property
    def enabled(self):
        cfg = getattr(self.modeller.args, 'adaptive_lr', None)
        return cfg is not None and getattr(cfg, 'enabled', False)

    def _cfg(self, name, default):
        return getattr(getattr(self.modeller.args, 'adaptive_lr', None), name, default)

    # ------------------------------------------------------------------ health

    def _live_dirs(self):
        """Branches actually being trained right now, with a fresh reading."""
        m = self.modeller
        min_frac = self._cfg('channel_min_frac', 0.01)
        stale_steps = self._cfg('channel_stale_steps', 200)
        out = set()
        for d in self.CHANNEL_DIRS:
            if float(getattr(m, f'{d}_frac', 0.0) or 0.0) < min_frac:
                continue
            last_it = m.metric_tracker.last_it.get(d)
            if (last_it is None or m.step_ind is None
                    or (m.step_ind - last_it) > stale_steps):
                continue
            out.add(d)
        return out

    def _channels(self, st=None):
        """The RELEVANT Z-invariant health channels this tick: {name: value}
        over {fwd,bwd,replay} x {scatter_err, slope_err}, skipping any that
        can't carry evidence about LR damage. A channel is dropped when:

          - it hasn't reported (cold start, never-run branch);
          - its branch is dormant -- frac below channel_min_frac, or its
            metric-tracker reading is staler than channel_stale_steps. A
            dormant branch isn't being trained, so its EMA is a frozen echo of
            an older policy: it drifts monotonically to its own best and pins
            elevation at exactly 1.0 forever (tngticqq: bwd_frac 0.001 in
            phase 3 held bwd/scatter_err + bwd/slope_err at elev 1.000 for the
            whole run);
          - the channel is SATURATED -- its running-best already sits so high
            that best x channel_margin exceeds the metric's ceiling, so no
            reading can ever breach it (tngticqq: fwd/slope_err re-seeded at
            0.9466 on the phase 1->3 transition, i.e. already fully
            decorrelated, needing 1.23 from a metric capped near 1.0).

        This matters because breach_fraction is a fraction: an unbreachable
        channel is not a neutral abstention, it's a permanent NO vote that
        silently lowers the ceiling on breach_frac. In tngticqq three of the
        six were unbreachable, capping breach_frac at 0.5 against a 0.67
        threshold -- the controller could not fire regardless of the damage.
        Keeping the denominator to channels that could actually vote is what
        makes the global-vs-local discriminator mean what it claims."""
        m = self.modeller
        margin = self._cfg('channel_margin', 1.3)
        cb = (st or {}).get('chan_best', {})
        out = {}
        for d in self._live_dirs():
            for k in self.CHANNEL_METRICS:
                v = m.metric_tracker.get(d, k)
                if v is None or not math.isfinite(v):
                    continue
                name = f'{d}/{k}'
                ceiling = self.CHANNEL_CEILING.get(k)
                best = cb.get(name)
                if ceiling is not None and best is not None and best * margin > ceiling:
                    continue  # saturated: unbreachable, so it gets no vote
                out[name] = float(v)
        return out

    def _prune_bests(self, st):
        """Drop running-bests for branches that have gone dormant, so a branch
        that later reactivates re-seeds against its own fresh baseline instead
        of a stale best set by a policy that has since moved on. Saturated
        channels keep their best: it's what marks them unbreachable, and
        dropping it would make them re-seed and flicker back into the vote."""
        live = self._live_dirs()
        for name in [n for n in st['chan_best'] if n.split('/')[0] not in live]:
            del st['chan_best'][name]

    def _update_bests_and_breach(self, st):
        """Update each channel's running best (a MINIMUM -- lower scatter/slope
        error is better -- that relaxes UPWARD by best_drift/tick so a lucky low
        reading can't pin the bar unreachably), and return (breach_fraction,
        n_available). A channel breaches when its value exceeds its own best x
        channel_margin: a purely relative, per-channel, scale-free test. Only
        channels that could actually breach are counted (see _channels)."""
        margin = self._cfg('channel_margin', 1.3)
        drift = self._cfg('best_drift', 0.003)
        cb = st['chan_best']
        self._prune_bests(st)
        chans = self._channels(st)
        breaches = 0
        for name, val in chans.items():
            prev = cb.get(name)
            if prev is None:
                cb[name] = val  # seed; never breaches on its seeding tick
                continue
            cb[name] = min(val, prev * (1.0 + drift))
            if val > cb[name] * margin:
                breaches += 1
        n = len(chans)
        return (breaches / n if n else 0.0), n

    # ------------------------------------------------------------------ actuator

    def _apply_lrs(self, st):
        """lr = configured base x scale per group, floored at min_lr -- EXCEPT
        the flow (Z head) groups, pinned flat at lr_flow (the standalone 'flow'
        optimizer and the fused optimizer's trailing group; see
        init_schedulers_optimizers). Scaling them with the policy silently broke
        the deliberate policy/Z decoupling (Z head ~20x under design at a typical
        scale, ylmtpqjy) exactly when off-policy TB needs Z tracking the
        re-priced terminals. control_flow_lr: true restores uniform scaling for
        A/B."""
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

    def _cut(self, st, new_scale):
        """Multiplicative decrease. The cut level becomes the new CRUISE ceiling,
        arm a cooldown, and clear the streaks so post-cut transients don't
        immediately re-trigger.

        The ceiling is the post-cut scale, not the pre-cut one. Pre-cut was
        useless: below the configured LR the climb caps at 1.0 anyway, so a
        ceiling recording "where damage was" only ever restated the cap already in
        force, and the climb walked straight back to the scale that caused the cut
        (~250 ticks from a 0.5 cut). Cutting is evidence about THIS surface -- the
        level we cut TO is the level we now believe is safe, so that is what we sit
        at, and ceiling_relax decides if/how slowly we ever try higher again.
        """
        m = self.modeller
        floor = m.args.min_lr / m.args.lr_policy
        st['scale'] = float(max(new_scale, floor))
        st['ceiling'] = st['scale']  # cruise here; ceiling_relax may lift it later
        st['cooldown_until'] = st['tick'] + self._cfg('cooldown_ticks', 20)
        st['last_action_tick'] = st['tick']
        st['breach_streak'] = 0
        st['clean_streak'] = 0

    # ------------------------------------------------------------------ state

    def _fresh_state(self, phase):
        m = self.modeller
        return {
            'ver': 4,  # v4 = safety-net (warmup/hold/cut) -- invalidates v1-v3 probe/cruise state,
            # whose 'scale'/'best'/'eta_star' are in incompatible (thrash) semantics
            'phase_seen': phase,
            'scale': 1.0 / m.args.lr_warmup_ratio,  # warmup start; ramps to 1.0 (= configured LR)
            'warmup_done': False,
            'chan_best': {},  # channel name -> running-best (min, drifts up)
            'breach_streak': 0,
            'clean_streak': 0,
            'breach_frac': 0.0,
            'ceiling': None,  # soft last-cut ceiling for the headroom climb; None until a cut
            'tick': 0,
            'cooldown_until': 0,
            'last_action_tick': 0,
        }

    def _state(self):
        m = self.modeller
        st = getattr(m, 'lr_ctrl', None)
        if not isinstance(st, dict) or st.get('scale') is None or st.get('ver') != 4:
            st = self._fresh_state(m.phase)
            m.lr_ctrl = st
        elif st.get('phase_seen') != m.phase:
            # phase change: not an LR event, so keep the running scale (no
            # re-warmup), but the new phase's loss levels differ -- re-seed the
            # channel bests and cool down so the jump can't spurious-cut.
            st['phase_seen'] = m.phase
            st['chan_best'] = {}
            st['breach_streak'] = 0
            st['clean_streak'] = 0
            st['cooldown_until'] = st['tick'] + self._cfg('cooldown_ticks', 20)
            st['warmup_done'] = True
        return st

    def rearm_warmup(self):
        """Re-run the blind warmup ramp (1/lr_warmup_ratio -> 1.0 over
        warmup_ticks) from here, as if the run had just started.

        The generic phase-change branch above deliberately does NOT do this: a
        phase change is not an LR event, and re-warming on every transition
        would keep stranding the LR low. The prior-training -> TB changeover is
        the exception, and the only caller: it is the one boundary where the
        optimizers are rebuilt (PhaseController._refresh_optimization) onto a
        loss surface with a completely different curvature, so the first TB
        step would otherwise land at the full operating LR with empty Adam
        moments. Ramping in is what makes the changeover survivable without
        cross-fading the losses themselves.

        Must be called with m.phase already set to its post-transition value,
        so _state()'s phase-change branch (which forces warmup_done) runs first
        and this overrides it rather than the other way round."""
        m = self.modeller
        if not self.enabled:
            return  # legacy path owns the ramp via m.lr_warmup_finished
        st = self._state()
        st['warmup_done'] = False
        st['tick'] = 0
        st['scale'] = 1.0 / m.args.lr_warmup_ratio
        st['cooldown_until'] = 0
        st['last_action_tick'] = 0
        # a soft ceiling left by a phase-1 cut describes damage on the MLE
        # surface; the ramp is heading back to the operating point regardless,
        # so an inherited cap would only mis-describe the climb it gates
        st['ceiling'] = None
        self._apply_lrs(st)
        return int(self._cfg('warmup_ticks', 100))

    # ------------------------------------------------------------------ tick

    def step(self):
        """One controller tick (every 10 train steps from step_lr_schedule).
        Returns the applied fwd LR, mirroring the legacy path."""
        m = self.modeller
        st = self._state()
        st['tick'] += 1
        in_cooldown = st['tick'] < st['cooldown_until']

        warmup_ticks = int(self._cfg('warmup_ticks', 100))
        # the warmup TARGET is the cruise ceiling, not a blind 1.0. fire_loss_spike
        # restores lr_ctrl from the best checkpoint, which re-arms warmup -- so
        # without this cap a rewind re-ramps straight back through the ceiling the
        # cut just set and into the scale that caused it (fn6erccw: two terminal
        # rewinds set ceiling 0.397 then 0.199, and scale was back at 1.0 within
        # ~50 steps of each -- 5x its own ceiling). The climb branch below already
        # respects the ceiling but never runs while breach_frac keeps clean_streak
        # at 0, so warmup is the ONLY path back up after a rewind, and it has to
        # honour the cap. ceiling is None on a genuine phase change (_state clears
        # it: a cut on the previous surface doesn't describe this one), so a normal
        # phase-transition warmup still targets 1.0 exactly as before.
        ceil = st.get('ceiling')
        target = 1.0 if ceil is None else min(1.0, ceil)
        if not st.get('warmup_done'):
            # blind exponential ramp (1/warmup_ratio -> target); metrics are still
            # warming and the LR itself is moving, so no breach logic here. Bests
            # seed lazily at HOLD entry, off the near-operating-LR baseline.
            frac = min(1.0, st['tick'] / max(1, warmup_ticks))
            st['scale'] = min((1.0 / m.args.lr_warmup_ratio) ** (1.0 - frac), target)
            if st['tick'] >= warmup_ticks:
                st['warmup_done'] = True
                st['scale'] = target
            self._apply_lrs(st)
            self._emit(st, warmup=True, breach_frac=0.0, n_avail=0)
            return m.optimizers['fwd'].param_groups[0]['lr']

        # HOLD regime
        breach_frac, n_avail = self._update_bests_and_breach(st)
        st['breach_frac'] = breach_frac
        breaching = n_avail >= 2 and breach_frac >= self._cfg('breach_fraction', 0.67)
        st['breach_streak'] = st['breach_streak'] + 1 if breaching else 0
        clean = n_avail >= 2 and breach_frac == 0.0
        st['clean_streak'] = st['clean_streak'] + 1 if clean else 0

        if not in_cooldown:
            if st['breach_streak'] >= self._cfg('breach_persistence', 3):
                # CUT: broad, sustained degradation across the reward channels
                self._cut(st, st['scale'] * self._cfg('cut_ratio', 0.5))
            elif st['clean_streak'] >= self._cfg('climb_patience', 10):
                # gentle additive climb, health-gated, hard-capped by the last cut's
                # soft ceiling. climb_above_base only decides the ABSOLUTE cap
                # (your LR, or scale_max above it); the ceiling applies in BOTH
                # regimes, so after a cut we CRUISE at the cut level instead of
                # walking back into the scale that caused it. ceiling_relax is the
                # only way back up: 0.0 makes a cut permanent, 0.001/tick takes
                # ~693 clean ticks (~7k steps) to lift the ceiling 2x -- "after a
                # good long time, optionally climb slowly".
                cap = self._cfg('scale_max', 2.0) if self._cfg('climb_above_base', False) else 1.0
                ceil = st.get('ceiling')
                if ceil is not None:
                    ceil = min(cap, ceil * (1.0 + self._cfg('ceiling_relax', 0.001)))
                    st['ceiling'] = ceil
                    cap = min(cap, ceil)
                new_scale = min(st['scale'] + self._cfg('climb_increment', 0.02), cap)
                if new_scale > st['scale']:
                    st['scale'] = new_scale
                    st['clean_streak'] = 0  # one increment, then re-earn it
                    st['last_action_tick'] = st['tick']

        self._apply_lrs(st)
        self._emit(st, warmup=False, breach_frac=breach_frac, n_avail=n_avail)
        return m.optimizers['fwd'].param_groups[0]['lr']

    def _emit(self, st, warmup, breach_frac, n_avail):
        self._report = {
            'lr_ctrl/scale': st['scale'],
            'lr_ctrl/warmup': float(warmup),
            'lr_ctrl/breach_frac': breach_frac,
            'lr_ctrl/breach_streak': st['breach_streak'],
            'lr_ctrl/clean_streak': st['clean_streak'],
            'lr_ctrl/n_channels': n_avail,
            'lr_ctrl/ceiling': st['ceiling'] if st['ceiling'] is not None else float('nan'),
        }
        # per-channel elevation (value / running-best): whichever crosses
        # channel_margin is a breaching channel -- reads the cut's reasoning
        # straight off the run page.
        cb = st.get('chan_best', {})
        for name, val in self._channels(st).items():
            b = cb.get(name)
            if b:
                self._report[f'lr_ctrl/elev_{name}'] = val / b

    def report(self):
        return dict(self._report)

    def on_explosion(self, count: int = 1):
        """fire_loss_spike hook (replaces the legacy flat 0.75 cut when enabled).
        Runs AFTER the best-checkpoint rewind restored lr_ctrl from healthy
        times, so the cut applies to the pre-damage scale. Same multiplicative
        cut as a breach, plus a cooldown.

        count is the number of TERMINAL rewinds so far (1 for an ordinary loss
        spike), and compounds the cut to cut_ratio**count. The rewind restores
        this controller's scale from a checkpoint written while the run was still
        healthy, which necessarily erases the evidence that this very scale
        already killed the policy -- so a single flat cut is undone by the next
        clean-streak recovery and the run walks back into the same detonation
        (djr13t0j's lr_fwd sawtooth). count is the only memory that survives.
        """
        st = self._state()
        self._cut(st, st['scale'] * (self._cfg('cut_ratio', 0.5) ** max(count, 1)))
        self._apply_lrs(st)


class ModeBalanceController:
    """
    Phase-3 mode-frac balancer: nudges fwd_frac/bwd_frac/replay_frac toward
    whichever of {bwd, fwd, replay} the current under/over/Z-error coverage
    metrics call for, and anneals the controller's thresholds once all three
    stay within bounds for long enough. Holds a reference to its owning
    Modeller for the metrics it reads and the frac/streak state it mutates.
    """

    def __init__(self, modeller):
        self.modeller = modeller

    def step(self):
        under, over, zerr = self._get_controller_metrics()
        under, over, zerr = self._lookahead_controller_metrics(under, over, zerr)
        # what the controller actually thresholds on, exposed for logging --
        # the raw metrics alone can make state selection look impossible
        self.modeller.controller_projections = {'under': under, 'over': over, 'zerr': zerr}
        state = self._select_controller_state(under, over, zerr)
        self._nudge_mode_fracs(state)

    def _get_controller_metrics(self):
        m = self.modeller
        # RELATIVE under-coverage (batch-centered on its own z_jensen, see
        # quick_tb_stats): the Z-anchored version conflates the collective
        # level gap -- which backward training provably cannot close; only Z
        # can -- with the within-buffer spread backward actually owns, so a
        # lagging Z read as "backward failing" and starved every other mode.
        # The Z-anchored under_coverage stays reported as the absolute-merge
        # gauge; it just no longer steers allocation.
        under = m.metric_tracker.get('bwd', 'relative_under')
        over = m.metric_tracker.get('fwd', 'over_coverage')
        # zerr = |EMA'd signed batch mean of the beta-clipped fwd TB residual|
        # (quick_tb_stats' tb_resid_clipped, same metric_tracker path as
        # under/over): up to the constant beta scale this IS dL/dZ of the
        # Huber TB loss, so it reads ~0 exactly when Z sits at the loss's own
        # fixed point (the self-consistent beta-Winsorized mean of log w) and
        # keeps a persistent sign -- hence a large EMA -- while Z lags a
        # moving target. Bounded by beta, so fat/skewed log w tails can't
        # inflate it. Replaces tb_err (residual RMS), whose hard floor is
        # std(log w) at the current policy: with a fat-spread policy that
        # floor sat permanently above zerr_threshold, locking the controller
        # in the Z-repair state and starving bwd/replay -- the only losses
        # that can shrink the spread -- at min_mode_frac (the "98% of compute
        # on fwd Z-only passes" deadlock). The EMA runs on the SIGNED value
        # and abs() is taken at read time, so a mean-zero oscillating
        # gradient correctly cancels to "converged" rather than accumulating.
        zerr = m.metric_tracker.get('fwd', 'tb_resid_clipped')

        if zerr is None:
            zerr = float("inf")  # bootstrap toward fwd
        else:
            zerr = abs(zerr)
        if under is None:
            under = 1.0  # forward gets attention first
        if over is None:
            over = 0.0  # do not demand replay before fwd stats exist

        return under, over, zerr

    def _lookahead_controller_metrics(self, under, over, zerr):
        """
        Extrapolate each controller metric a few controller ticks into the
        future by EMA-smoothing its trend and projecting forward linearly, so
        _select_controller_state reacts to where a metric is heading rather
        than where it currently sits. Without this, mode-frac mass keeps
        moving at full speed right up to a threshold crossing and overshoots
        the optimal balance before the (lagging) raw metric ever reflects it.
        """
        m = self.modeller
        ctrl = m.args.controller
        lookahead = m.controller_lookahead
        return (
            self._log_ema_lookahead(lookahead['under'], under, ctrl),
            self._log_ema_lookahead(lookahead['over'], over, ctrl),
            self._log_ema_lookahead(lookahead['zerr'], zerr, ctrl),
        )

    @staticmethod
    def _log_ema_lookahead(state, value, ctrl):
        """
        under/over/zerr are already EMAs coming out of metric_tracker, so
        `value` is already a smoothed level - only its trend (the per-tick
        delta) needs its own EMA here. All three metrics are nonnegative nats
        living on a log scale (raw values run orders of magnitude above
        thresholds of order 1), so the trend is tracked in log space and the
        projection is multiplicative: a steep descent predicts "k-fold lower
        in `horizon` ticks", never "below zero". The linear projection this
        replaces could tunnel through the entire threshold band and flip the
        controller state while the raw metric was still far out of bounds.
        state: {'level', 'trend'} dict, mutated in place (persisted via
        MODELLER_STATE_DEFAULTS so it survives checkpoint reloads); 'level'
        holds the previous raw value, 'trend' is in log-nats per tick.
        Returns value * exp(horizon * trend), the predicted value
        `ctrl.lookahead_horizon` controller ticks ahead.
        """
        if not math.isfinite(value):
            return value
        if state['level'] is None:
            state['level'] = value
            return value

        eps = 1e-3  # keeps the log finite when a metric sits at its floor of 0
        log_delta = math.log(max(value, eps)) - math.log(max(state['level'], eps))
        trend_alpha = getattr(ctrl, 'lookahead_trend_alpha', 0.1)
        horizon = getattr(ctrl, 'lookahead_horizon', 5)
        state['trend'] = trend_alpha * log_delta + (1 - trend_alpha) * state['trend']
        state['level'] = value

        # cap the projection at e^±3 (~20x) of the current value: beyond that the
        # extrapolation is guessing, and the cap also defuses stale linear-unit
        # trends restored from pre-log-space checkpoints
        exponent = min(max(horizon * state['trend'], -3.0), 3.0)
        return value * math.exp(exponent)

    def _select_controller_state(self, under, over, zerr):
        """
        Priority order: Z convergence > undercoverage repair > replay repair >
        global tightening. Z is checked first, above even backward undercoverage
        repair -- under/over_coverage are themselves computed from TB residuals
        relative to log_Z (quick_tb_stats), so they're only meaningful once Z is
        reasonably calibrated; chasing coverage repairs against a miscalibrated
        Z means measuring with a broken ruler. This is the state/step-size
        selection logic most likely to change later, so it's kept isolated from
        the metric plumbing and frac math around it.
        """
        m = self.modeller
        ctrl = m.args.controller
        if zerr > ctrl.zerr_threshold:
            m.controller_anneal_streak = 0
            return "fwd"
        elif under > ctrl.under_threshold:
            m.controller_anneal_streak = 0
            return "bwd"
        elif over > ctrl.over_threshold:
            m.controller_anneal_streak = 0
            return "replay"
        else:
            # joint condition (all three metrics within threshold) satisfied this tick;
            # require it to hold for `anneal_patience` consecutive ticks before tightening
            # the margins, since a single tick is too susceptible to metric noise
            m.controller_anneal_streak += 1
            if m.controller_anneal_streak >= getattr(ctrl, 'anneal_patience', 1):
                self._anneal_controller_thresholds()
                m.controller_anneal_streak = 0
            return 'replay'

    def _anneal_controller_thresholds(self):
        ctrl = self.modeller.args.controller
        if ctrl.under_threshold > ctrl.min_threshold:
            ctrl.under_threshold *= ctrl.decay_rate
        if ctrl.over_threshold > ctrl.min_threshold:
            ctrl.over_threshold *= ctrl.decay_rate
        if ctrl.zerr_threshold > ctrl.zerr_min_threshold:
            ctrl.zerr_threshold *= ctrl.decay_rate

    def _nudge_mode_fracs(self, state):
        if state is not None:
            m = self.modeller
            ctrl = m.args.controller
            probs = np.array([m.fwd_frac, m.bwd_frac, m.replay_frac], dtype=float)
            probs /= probs.sum()

            idx = {"fwd": 0, "bwd": 1, "replay": 2}[state]

            m_floor = ctrl.min_mode_frac  # requires m_floor < 1/3
            free = 1.0 - 3.0 * m_floor  # total mass available above the floors

            # excess space: x_i = p_i - m_floor, with x_i >= 0 and sum(x) = free
            excess = np.clip(probs - m_floor, 0.0, None)
            s = excess.sum()
            excess = excess * (free / s) if s > 0.0 else np.full(3, free / 3.0)

            # EMA toward the one-hot on the boosted mode
            excess *= 1.0 - ctrl.beta
            excess[idx] += ctrl.beta * free

            m.fwd_frac, m.bwd_frac, m.replay_frac = m_floor + excess


class ForwardFirstController:
    """
    Optional replacement for phases 2/3 (config: forward_first.enabled), around
    one principle: THE FORWARD POLICY IS CALIBRATED ALWAYS, on a GROWING covered
    support S (KL(P_F || pi_S) ~ 0). Z is then an exact running ledger of
    annexed mass (log Z_S), approached from below at all times, and mode
    addition is a sequence of small local problems instead of one global chase
    -- rather than the standard protocol's Z lagging the buffer level by the
    policy's entire remaining error, against which backward TB spends a
    saturated gradient on a level it provably cannot move (E_mu[log P_F] is
    capped at -H(mu) by normalization).

    Phase 1 (MLE) still runs by default: it produces the prior model (the
    backward buffer source -- a static dataset is atomic, so backward would
    overfit to spikes) plus a broad-range, self-consistent policy init. The
    phase-1 exit routes here (PhaseController.phase1_to_forward_first). A prior
    loaded by path skips phase 1 (see maybe_begin). Forward TB collapses the
    policy almost immediately in stage A regardless, so the MLE coverage is
    transient and cannot moot the build-out.

    Two stages (state in modeller.forward_first_state, persisted via
    MODELLER_STATE_DEFAULTS; rides phase-3 plumbing throughout -- fused
    3-branch stepping, controller cadence, buffer management):

      A  build-out: fwd TB trains policy AND Z on-policy; replay retains;
         backward dormant at the frac floor (and its force-refresh skipped
         entirely, see bwd_dormant). Collapse expected -- calibration over
         coverage. Exit: fwd r2/scatter healthy for calib_patience ticks.
      B  TERMINAL lexicographic control (see _stage_b_tick): fwd policy frozen
         (Z-only), and
             Z > calibration-not-degrading > coverage-while-it-still-moves
         with one actuator each (fwd/replay/bwd). There is NO stage C: this rule
         asymptotes into what standard phase 3 was reaching for, and handing
         over would only INVERT the priority (phase 3 ranks coverage above
         calibration; this ranks calibration above coverage -- coverage is
         EARNED, never taken at the cost of the fit).

    Diagnostics: once differentiated, dZ/dt < 0 can only mean retention failure
    (Z is a monotone ledger of covered mass while calibration holds). Watch
    forward_first_stage, ff_state, fwd/r2, fwd/scatter_err, bwd/relative_under,
    and log_Z_learned (should ledger UP, ~log 2 per equal-mass mode annexed).
    NB mode adoption looks SUDDEN in the marginals/Total Var but is smooth
    underneath -- sampling visibility is exp(-residual) while the metrics are
    linear in nats; the distributional metrics (delta_overlap, mmd, nn_sep) show
    the true, smooth trend.
    """

    STAGES = {'A': 0, 'B': 1}

    def __init__(self, modeller):
        self.modeller = modeller

    @property
    def enabled(self):
        cfg = getattr(self.modeller.args, 'forward_first', None)
        return cfg is not None and getattr(cfg, 'enabled', False)

    @property
    def active(self):
        """Engaged and not yet handed over to the standard controller."""
        return self.enabled and self.modeller.forward_first_state.get('stage') in ('A', 'B')

    @property
    def bwd_dormant(self):
        """Stage A: backward is off and nothing reads its metrics until stage B,
        so fused_train_step may skip its force-refresh rollout entirely (the
        stats just start stale in B and populate once backward is admitted).
        False whenever the protocol is disabled, so standard runs keep their
        normal force-refresh behavior."""
        return self.enabled and self.modeller.forward_first_state.get('stage') == 'A'

    def _cfg(self, name, default):
        return getattr(getattr(self.modeller.args, 'forward_first', None), name, default)

    def _fwd_quality(self):
        m = self.modeller
        return m.metric_tracker.get('fwd', 'r2'), m.metric_tracker.get('fwd', 'scatter_err')

    def _boundary_elevation(self):
        """
        Box-containment elevation: fwd/box_violation over the LOWEST containment
        this run has achieved. Same best x margin idiom as _calib_elevation, and
        for the same reason -- box_violation's absolute scale carries no
        information (1219ddv9 ran 0.004 -> 19.8, a 4500x range) but its rise off
        the floor is the earliest warning in the panel: it moved 4.4x a full ~100
        steps before r2 collapsed and ~100 before the LR controller's first
        breach, while still three orders of magnitude below the eventual damage.

        Denominator floored: an early near-zero batch must not pin `best` so low
        that the ratio is hypersensitive for the rest of the run.

        Returns None when the channel hasn't reported -- runs predating train.py's
        _update_rolling plumbing, or a cold tracker -- so the branch no-ops rather
        than crashing.
        """
        m = self.modeller
        cur = m.metric_tracker.get('fwd', 'box_violation')
        best = m.metric_tracker.get_best('fwd', 'box_violation', mode='min')
        if cur is None or best is None:
            return None
        if not (math.isfinite(cur) and math.isfinite(best)):
            return None
        return cur / max(best, self._cfg('boundary_floor', 1e-3))

    def _set_fracs(self, bwd_frac):
        m = self.modeller
        floor = m.args.controller.min_mode_frac
        share = self._cfg('replay_share', 0.2)
        m.bwd_frac = max(bwd_frac, floor)
        m.replay_frac = share
        m.fwd_frac = max(1.0 - share - m.bwd_frac, floor)

    def engage(self):
        """
        Enter stage A -- shared by both entry points: the default phase-1-exit
        route (PhaseController.phase1_to_forward_first, after the MLE warm-start
        produced the prior model) and the step-0 loaded-prior route
        (maybe_begin). Assumes a prior_model EXISTS as the backward buffer
        source, so bwd draws come from the churned prior_buffer, not the static
        dataset (which is atomic -> backward would overfit to spikes). Rides
        phase-3 plumbing; set_loss_coeffs after the flips (before too, so a
        step-0 engagement doesn't _flip into empty schedules and suppress the
        initial full config parse).
        """
        m = self.modeller
        m.set_loss_coeffs()
        m.phase = 3
        m.bwd_sampling_mode = 'prior'  # churned prior_buffer, not the static prior_dataset
        m.phase_controller.forward_first_stage_a()
        m.set_loss_coeffs()
        self._set_fracs(m.args.controller.min_mode_frac)
        st = m.forward_first_state
        st['stage'] = 'A'
        st['streak'] = 0
        # phase is already 3 above, so this overrides the phase-change branch's
        # no-re-warmup rule for this one boundary (see rearm_warmup)
        warmup_ticks = m.lr_controller.rearm_warmup()
        print(f"forward-first: stage A engaged -- forward+replay build-out, backward dormant "
              f"(fwd {m.fwd_frac:.3f} / replay {m.replay_frac:.3f} / bwd {m.bwd_frac:.4f})")
        if warmup_ticks:
            print(f"forward-first: LR re-warming for the TB surface -- "
                  f"lr_policy/{m.args.lr_warmup_ratio:g} -> lr_policy over "
                  f"{warmup_ticks} ticks (~{warmup_ticks * 10} train steps)")

    def maybe_begin(self):
        """
        Step-0 entry, ONLY when a prior is loaded by path (prior_model_name set,
        so self.prior_model exists): that signals "skip phase 1", and the loaded
        prior both warm-starts the policy and feeds the buffer. Without a loaded
        prior this is a no-op -- phase-1 MLE runs normally and the phase-1 exit
        routes into forward-first (phase1_to_forward_first), producing the prior
        model + warm start as a byproduct exactly like the standard 1->3 route.
        Resumed runs (stage already set, or not fresh) are left alone.
        """
        m = self.modeller
        if not self.enabled:
            return
        st = m.forward_first_state
        if st.get('stage') is not None:
            return  # resumed mid-protocol
        if m.step_ind != 0 or m.phase != 1:
            return  # resumed standard-protocol run
        if not hasattr(m, 'prior_model'):
            return  # no loaded prior -> let phase 1 run; the exit route engages us
        # loaded prior by path: warm-start the policy from it (it "is literally
        # the starting point for forward training"), then engage.
        m.gfn_model.load_state_dict(m.prior_model.state_dict())
        m.ema_model.load_state_dict(m.prior_model.state_dict())
        print("forward-first: prior loaded by path -- skipping phase 1, warm-starting policy from it")
        self.engage()

    def step(self):
        """One tick, called from the phase-3 controller slot (every 10 steps)
        in place of ModeBalanceController while the protocol is active."""
        st = self.modeller.forward_first_state
        if st.get('stage') == 'A':
            self._stage_a_tick(st)
        elif st.get('stage') == 'B':
            self._stage_b_tick(st)

    def _stage_a_tick(self, st):
        """
        Stage A is stage B's rule MINUS the coverage level -- the two stages are
        one lexicographic controller, and the stage flag only decides (a) whether
        fwd trains the policy (A: yes, B: frozen) and (b) whether bwd is in the
        priority list (A: no -- collapse and calibrate BEFORE growing; that
        exclusion is what "stage A" means). So fwd:replay is managed dynamically
        here by the same Z > calibration priority, not held at a static split.

        NO churn control here: with the forward policy live, fwd is itself an
        always-fresh calibration source, so replay memorizing is fixed by simply
        WITHHOLDING ITS FRAC (the _replay_outcompeting gate below) and letting
        fwd take the throughput. That gate is load-bearing, not decorative:
        without it the rule amplifies overfit -- replay memorizes -> on-policy
        calibration degrades -> "calibration degrading" boosts REPLAY -> worse.
        Stage B can't use this lever (fwd is Z-only there, so withholding
        replay's frac would abandon calibration outright) and reaches for churn
        instead; see _churn_control.

        bwd is normally never boosted, so _nudge_mode_fracs' EMA decays it to
        min_mode_frac on its own (dormancy emerges rather than being forced) and
        bwd_dormant skips its force-refresh entirely. The ONE exception is the
        boundary servo below -- stage A's exclusion of bwd is about COVERAGE
        (don't grow support before calibrating), not about containment, and the
        two are separable.

        The boundary servo: 1219ddv9 died with mass in the box wall, and no LR
        response could touch it -- the controller cut 100x while every channel
        degraded monotonically, because escape is an objective pull, not a
        step-size instability. LR is not the actuator for it; bwd_frac is. The
        wall can only oppose further escape, while bwd is the only force pointing
        back AT the basin, so admitting bwd exactly when containment slips is the
        minimum intervention that addresses the actual failure.

        Deliberately ABOVE zerr in the chain: containment is causally upstream of
        calibration. In 1219ddv9 the boundary moved at 6800 while Z was still
        fine; by the time Z degrades you are calibrating against a distribution
        that is about to stop existing. Getting this order wrong costs the run.

        Bang-bang, not proportional, and that is enough because _nudge_mode_facs
        is ALREADY bidirectional: when the drift stops, 'bwd' stops firing, the
        EMA decays bwd back to min_mode_frac on its own, and the servo's output
        returns to zero. So this is NOT a ratchet -- no new one-way mechanism is
        introduced, and while fwd is healthy and off the wall the branch never
        fires and stage A behaves exactly as before. bwd_dormant needs no change:
        train.py's bwd_active (bwd_frac >= deactivate_threshold) admits the real
        backward branch on frac alone, and dormant only ever skipped the
        stats-refresh rollout, which is moot once bwd is genuinely training.
        """
        m = self.modeller
        ctrl = m.args.controller

        zerr = self._zerr()
        elev = self._calib_elevation(st)
        over = m.metric_tracker.get('fwd', 'over_coverage')
        boundary = self._boundary_elevation()

        if boundary is not None and boundary > self._cfg('boundary_margin', 3.0):
            state = 'bwd'  # containment first: nothing downstream survives escape
        elif zerr is not None and zerr > ctrl.zerr_threshold:
            state = 'fwd'  # Z is the ruler: calibrate it first, here as everywhere
        elif (((elev is not None and elev > self._cfg('calib_margin', 1.3))
               or (over is not None and math.isfinite(over) and over > ctrl.over_threshold))
              and not self._replay_outcompeting()):
            state = 'replay'  # on-policy calibration degrading / junk to drain
        else:
            # default, AND the overfit path: hand throughput to fwd -- it's the
            # live, always-fresh calibration source, and replay's frac decays
            state = 'fwd'
        st['ff_state'] = state
        self._nudge_mode_fracs(state)

        if m.step_ind < self._cfg('min_steps', 500):
            return  # tracker EMAs still warming; a lucky early window must not fire the gate
        r2, scatter = self._fwd_quality()
        ok = (r2 is not None and math.isfinite(r2) and r2 >= self._cfg('calib_r2_min', 0.95)
              and scatter is not None and math.isfinite(scatter)
              and scatter <= self._cfg('calib_scatter_max', 5.0))
        st['streak'] = st.get('streak', 0) + 1 if ok else 0
        if st['streak'] >= self._cfg('calib_patience', 20):
            st['stage'] = 'B'
            st['streak'] = 0
            m.phase_controller.forward_first_stage_b()
            # the reload point for ramp experiments, so it freezes its own buffer
            # sidecar. Unlike the phase snapshots this fires from the 10-step
            # controller tick rather than at eval, but the streak gate latches it
            # to once per run, so it's a one-off write
            m.checkpointer.save('ff_calibrated', with_buffers=True)
            print(f"forward-first: calibration gate passed (r2 {r2:.3f}, scatter {scatter:.2f}) "
                  f"-> stage B: forward policy frozen, backward ramp begins")

    # ------------------------------------------------------ stage B (terminal)

    def _zerr(self):
        """Z's own raw gradient: |EMA of the beta-clipped fwd TB residual| = dL/dZ
        up to the constant beta scale. ~0 exactly at the loss's fixed point
        regardless of Var(log w), so unlike tb_err (RMS, floored at std(log w))
        it can always converge -- no unreachable-threshold deadlock."""
        v = self.modeller.metric_tracker.get('fwd', 'tb_resid_clipped')
        return abs(v) if v is not None and math.isfinite(v) else None

    def _calib_elevation(self, st):
        """fwd scatter_err relative to its OWN running best (which drifts upward):
        the question is "is calibration DEGRADING", never "is it below an absolute
        bar". The calibration floor RISES with coverage -- b9ze0p5c floored at 0.81
        with one mode and 1.23 after adopting the second, permanently, because two
        modes are genuinely harder to represent. An absolute threshold would read
        that new legitimate floor as permanent failure, pin replay at priority
        forever, and block every subsequent mode: the protocol would add exactly
        one mode and deadlock. Elevation-vs-best makes a mode adoption a TRANSIENT
        spike that resolves as the best drifts up to the new floor. Never annealed,
        for the same reason."""
        m = self.modeller
        v = m.metric_tracker.get('fwd', 'scatter_err')
        if v is None or not math.isfinite(v):
            return None
        best = st.get('calib_best')
        if best is None:
            st['calib_best'] = v
            return 1.0
        st['calib_best'] = min(v, best * (1.0 + self._cfg('calib_best_drift', 0.003)))
        return v / max(st['calib_best'], 1e-8)

    def _replay_outcompeting(self):
        """Is replay MEMORIZING? The replay buffer stores OVERWEIGHTED
        (high-residual) trajectories, so it is a strictly HARDER set than fresh
        on-policy: replay scatter WORSE than fwd is the healthy default, and
        replay BEATING fwd can only mean it memorized the finite stored set.
        False (not None) when the comparison isn't trustworthy yet -- a
        near-empty buffer or a missing channel means "no evidence of overfit",
        which is the safe reading for both callers. Shared by stage A (which
        uses it to withhold replay's frac) and stage B (which uses it to raise
        churn); see _churn_control for why the fix differs per stage."""
        m = self.modeller
        if not hasattr(m, 'replay_buffer'):
            return False
        if len(m.replay_buffer) < self._cfg('churn_min_buffer', 1000):
            return False
        r = m.metric_tracker.get('replay', 'scatter_err')
        f = m.metric_tracker.get('fwd', 'scatter_err')
        if r is None or f is None or not (math.isfinite(r) and math.isfinite(f)):
            return False
        return r < f

    def _churn_control(self, st):
        """Replay-overfit guard for STAGE B ONLY, nested under the replay actuator
        (not a priority level of its own).

        Why the fix is churn here but the frac in stage A: stage B freezes the
        forward policy (fwd is Z-only), so replay is the SOLE calibration
        actuator -- withholding its throughput would abandon calibration
        outright, leaving no lever but the DATA. Raising churn fixes overfit at
        its cause (a moving target can't be memorized) and is guaranteed to work:
        higher churn drags replay's distribution toward on-policy, closing the
        gap, with the sole exception that the policy genuinely fits both --
        which is convergence, and reads identically healthy. In stage A the
        forward policy is live, so fwd is an alternative calibration source and
        simply moving throughput off replay suffices; no churn needed there."""
        m = self.modeller
        cfg = getattr(getattr(m.args, 'buffers', None), 'replay_buffer', None)
        if cfg is None or not hasattr(m, 'replay_buffer'):
            return
        if len(m.replay_buffer) < self._cfg('churn_min_buffer', 1000):
            return
        if self._replay_outcompeting():  # memorizing -> flood faster
            base = st.setdefault('churn_base', float(cfg.random_churn_rate))
            cfg.random_churn_rate = min(cfg.random_churn_rate * self._cfg('churn_up', 1.05),
                                        base * self._cfg('churn_max_mult', 20.0))
        else:  # healthy (replay is the harder set) -> relax back toward configured
            base = st.setdefault('churn_base', float(cfg.random_churn_rate))
            cfg.random_churn_rate = max(cfg.random_churn_rate * self._cfg('churn_down', 0.99), base)
        st['churn_rate'] = cfg.random_churn_rate

    def _floor_track(self, st, name, value, held_priority):
        """Is an annealed metric still RESPONDING while its own mode holds
        priority? Once it stalls (no new best for floor_patience ticks of
        priority) it has hit its achievable floor -- freeze its threshold.
        Without this, annealing walks the threshold below the floor and pins that
        priority forever: exactly the old tb_err deadlock (RMS floored at
        std(log w) -> unreachable -> controller stuck). The floors RISE with mode
        count (relative_under floors at ~0.7 x the scatter floor: b9ze0p5c 0.9 vs
        1.23), so a fixed min_threshold is only right for the current mode count --
        this self-tracks instead."""
        if value is None or not math.isfinite(value):
            return
        rec = st.setdefault('floors', {}).setdefault(
            name, {'best': value, 'stall': 0, 'floored': False})
        if value < rec['best'] * (1.0 - self._cfg('floor_improve_frac', 0.02)):
            rec['best'] = value  # still responding
            rec['stall'] = 0
            rec['floored'] = False
        elif value > rec['best'] * (1.0 + self._cfg('floor_reset_frac', 0.5)):
            # jumped well above its floor: NEW work appeared (a fresh mode
            # admitted, buffer churned, a phase of discovery) -- the old floor
            # verdict is stale, re-arm so this priority can act again
            rec['stall'] = 0
            rec['floored'] = False
        elif held_priority:
            rec['stall'] += 1
            if rec['stall'] >= self._cfg('floor_patience', 50):
                rec['floored'] = True

    def _anneal(self, st):
        """Tighten ONLY the outlier-tamping thresholds (under/over), and only
        while they're still responding. zerr is already at its small-batch noise
        floor by design (zerr_min_threshold == zerr_threshold), and the
        calibration gate is elevation-relative so it self-tracks its rising floor
        -- neither is annealable without manufacturing a deadlock."""
        ctrl = self.modeller.args.controller
        fl = st.get('floors', {})
        if not fl.get('under', {}).get('floored') and ctrl.under_threshold > ctrl.min_threshold:
            ctrl.under_threshold *= ctrl.decay_rate
        if not fl.get('over', {}).get('floored') and ctrl.over_threshold > ctrl.min_threshold:
            ctrl.over_threshold *= ctrl.decay_rate

    def _stage_b_tick(self, st):
        """
        TERMINAL lexicographic controller (there is no stage C: this rule
        asymptotes into what standard phase 3 was reaching for, so a handover
        would only INVERT the priority order -- phase 3 puts coverage above
        calibration, this puts calibration above coverage, and that inversion was
        the conceptual confusion the handover encoded).

            Z  >  calibration-not-degrading  >  coverage-while-it-still-moves

        One actuator each: fwd (Z-only) tamps log Z, replay tamps on-policy
        calibration (retention AND junk-draining are the same operation -- replay
        TB drives stored residuals to zero regardless of sign), bwd pries open
        support. Z is first because everything else is measured against it.
        Calibration outranks coverage because coverage is EARNED, never taken at
        the cost of the fit. When all three are satisfied, anneal (under/over
        only, floor-aware).
        """
        m = self.modeller
        ctrl = m.args.controller

        self._churn_control(st)

        zerr = self._zerr()
        elev = self._calib_elevation(st)
        over = m.metric_tracker.get('fwd', 'over_coverage')
        under = m.metric_tracker.get('bwd', 'relative_under')

        # a FLOORED metric yields priority: it has already proven it can't improve
        # while holding it, so pinning its mode there only starves the others
        # (freezing the anneal alone doesn't help -- the metric still sits above
        # its frozen threshold forever). _floor_track re-arms it the moment new
        # work appears, so this yields coverage only when coverage is genuinely done.
        fl = st.get('floors', {})
        under_live = not fl.get('under', {}).get('floored')
        over_live = not fl.get('over', {}).get('floored')

        if zerr is not None and zerr > ctrl.zerr_threshold:
            state = 'fwd'
            st['anneal_streak'] = 0
        elif ((elev is not None and elev > self._cfg('calib_margin', 1.3))
              or (over is not None and math.isfinite(over)
                  and over > ctrl.over_threshold and over_live)):
            state = 'replay'
            st['anneal_streak'] = 0
        elif (under is not None and math.isfinite(under)
              and under > ctrl.under_threshold and under_live):
            state = 'bwd'
            st['anneal_streak'] = 0
        else:
            # all three satisfied -- tighten the outlier thresholds (patience-gated
            # against metric noise, exactly as the old phase-3 anneal was)
            st['anneal_streak'] = st.get('anneal_streak', 0) + 1
            if st['anneal_streak'] >= getattr(ctrl, 'anneal_patience', 5):
                self._anneal(st)
                st['anneal_streak'] = 0
            state = 'replay'  # default idle actuator: keep the fit clean

        # floor tracking: only counts while that metric's own mode holds priority
        self._floor_track(st, 'under', under, state == 'bwd')
        self._floor_track(st, 'over', over, state == 'replay')

        st['ff_state'] = state
        self._nudge_mode_fracs(state)

    def _nudge_mode_fracs(self, state):
        """EMA nudge of the fracs toward a one-hot on the boosted mode, with the
        min_mode_frac floor -- identical mechanics to ModeBalanceController, so
        only the SELECTION rule above differs from standard phase 3."""
        m = self.modeller
        ctrl = m.args.controller
        probs = np.array([m.fwd_frac, m.bwd_frac, m.replay_frac], dtype=float)
        probs /= probs.sum()
        idx = {"fwd": 0, "bwd": 1, "replay": 2}[state]
        m_floor = ctrl.min_mode_frac
        free = 1.0 - 3.0 * m_floor
        excess = np.clip(probs - m_floor, 0.0, None)
        s = excess.sum()
        excess = excess * (free / s) if s > 0.0 else np.full(3, free / 3.0)
        excess *= 1.0 - ctrl.beta
        excess[idx] += ctrl.beta * free
        m.fwd_frac, m.bwd_frac, m.replay_frac = m_floor + excess
