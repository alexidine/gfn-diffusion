import math

import numpy as np
import torch

from energy_sampling.eval.utils import LossSpikeMonitor
from energy_sampling.utils import get_discretizer


class AdaptiveLRController:
    """
    Safety-net LR controller (v5): a single global multiplicative `scale`
    over the configured per-branch base LRs (lr_policy/lr_back/lr_replay/
    lr_fused), continuously adapted. No fixed "operating point" to hold at,
    no reward-channel breach-fraction bookkeeping (that whole apparatus --
    channel liveness/saturation/pruning, running-bests, warmup-then-HOLD --
    is gone; see git history on this class for why v1-v4 needed it and why
    this one doesn't). Two independent halves:

    ADAPT (soft, continuous, every tick): replay a small FIXED batch of
    stored trajectories -- drawn once from replay_buffer and held fixed --
    through the CURRENT policy under no_grad (`_score_probe`), and read off
    log_pf/log_pb. How far has the model's behaviour on these exact inputs
    moved since last tick (`_probe_mag`), and is that movement accumulating
    in a consistent direction (`_probe_coh` > 0 -> climb) or reversing tick
    to tick (`_probe_coh` < 0 -> ease off)? This is a direct, dimensionless
    read on step size vs local curvature -- the function-space analogue of
    "do successive updates agree or fight" -- and it survives what would
    blind a parameter-space version of the same idea: it's phase-agnostic
    (get_traj_replay doesn't care which branches are "live" right now), it
    never touches a parameter norm (so it's immune to reparameterization
    and, unlike an update-magnitude probe, immune to whatever the grad clip
    does upstream of it), and it catches diffusive scrambling (no consistent
    update direction to read a sign off of) that a purely directional
    parameter-space probe would read as neutral.

    FIRE (hard, event-triggered): the same rewind-to-best-checkpoint +
    ratcheting cut already used for a loss explosion (train.py's
    fire_loss_spike/on_explosion, unchanged), now armed by THREE independent
    detectors instead of one: a per-branch loss ceiling breach (`check_spike`
    -- folded in from the four standalone LossSpikeMonitor instances this
    replaces, same detection algorithm, same defaults), a pre-clip grad-norm
    spike (new: the FAST catch, firing in a single tick rather than waiting
    on a persistence streak or on the loss to follow -- a genuine leading
    indicator now that gradient_norm_clip is loose (100.0) and rarely binds,
    so the clip is no longer laundering it away before anyone sees it), and
    the pre-existing absolute terminal-state bounds check in train.py
    (_terminal_policy_state, untouched -- that one stays a hard kill switch,
    not a detector here). All three converge on the same fire_loss_spike()
    rewind; only the terminal channel compounds the cut via terminal_reloads
    -- an ordinary spike (loss or gradient) is treated as a recoverable event
    and gets a flat cut_ratio, same as before.

    Config knobs you actually tune (adaptive_lr.*): adapt_gain (how fast the
    scale climbs when steps agree / eases off when they fight -- also the
    per-tick cap, since it multiplies an EMA'd cosine already bounded in
    [-1, 1]), loss_tripwire_mult and grad_tripwire_mult (how many multiples
    of a channel's own rolling median counts as an explosion). Everything
    else below is a sensible default.

    The flow (Z head) LR is still PINNED at lr_flow, exempt from scaling --
    unchanged from v4 (ylmtpqjy): the ADAPT signal has no sensor mandate over
    Z, and scaling it with the policy ran the Z head ~20x under design.
    control_flow_lr: true restores uniform scaling for A/B.
    """

    CHANNELS = ('fwd', 'bwd', 'replay', 'fused')

    def __init__(self, modeller):
        self.modeller = modeller
        self._report = {}
        self._spike_loss = None
        self._spike_grad = None
        self._probe_src = None  # lazily-drawn fixed (traj, condition, mol_batch)
        # fire memory, deliberately INSTANCE state and NOT in lr_ctrl: the
        # rewind that follows every fire restores lr_ctrl from the 'best'
        # checkpoint, which erases any evidence kept there that this scale
        # already detonated (the djr13t0j sawtooth; same reasoning as
        # train.py's terminal_reloads). _fire_steps drives the repeat-fire
        # escalation in on_explosion; _fire_scales [(step, post-cut scale)]
        # drives the recent-fire climb ceiling; _fire_counts is per-channel
        # telemetry.
        self._fire_steps = []
        self._fire_scales = []
        self._fire_counts = {}

    @property
    def enabled(self):
        cfg = getattr(self.modeller.args, 'adaptive_lr', None)
        return cfg is not None and getattr(cfg, 'enabled', False)

    def _cfg(self, name, default):
        return getattr(getattr(self.modeller.args, 'adaptive_lr', None), name, default)

    # ------------------------------------------------------------- fire (spikes)

    def _spikes(self):
        """Lazy so a config that isn't fully populated at __init__ time (same
        laziness as v4's _cfg-per-tick pattern) doesn't matter. window/warmup/
        cooldown are the same shape LossSpikeMonitor always ran with here;
        only the ceiling multiple is exposed, per channel-family, as the two
        knobs users actually look at."""
        if self._spike_loss is None:
            self._spike_loss = {
                name: LossSpikeMonitor(window=200, warmup=250, cooldown=100,
                                       ceiling_factor=self._cfg('loss_tripwire_mult', 100.0),
                                       min_baseline=self._cfg('loss_tripwire_floor', 1.0),
                                       name=f'{name}_loss')
                for name in self.CHANNELS}
            # the grad channel's floor defaults to gradient_norm_clip: a
            # pre-clip norm below ~the clip produces the SAME applied update as
            # one exactly at it (the clip normalizes magnitude away), so
            # excursions in that range are already neutralized and firing a
            # rewind on them punishes a step that was never taken (lepiqh54:
            # 9 fires at 106-344 vs medians 10-39, all under 6x the bar once
            # floored at clip=100). The bar bottoms out at
            # grad_tripwire_mult x clip -- a genuine detonation's grads blow
            # orders beyond that.
            grad_floor = self._cfg('grad_tripwire_floor',
                                   float(getattr(self.modeller.args, 'gradient_norm_clip', 0.0)))
            self._spike_grad = LossSpikeMonitor(window=200, warmup=250, cooldown=100,
                                                ceiling_factor=self._cfg('grad_tripwire_mult', 6.0),
                                                min_baseline=grad_floor,
                                                name='grad_norm')
        return self._spike_loss, self._spike_grad

    def check_spike(self, step_type, current_loss, grad_norm):
        """Fast, event-triggered detector feeding train.py's fire_loss_spike().
        Ports LossSpikeMonitor's actual behaviour unchanged for loss (non-
        finite, or >= loss_tripwire_mult x this branch's own long-window
        median, floored -- see LossSpikeMonitor.min_baseline), and applies the
        identical algorithm to the pre-clip grad norm as a second, faster
        channel. Either firing routes to the SAME ordinary (non-terminal)
        rewind-and-cut as before -- see this class's docstring, FIRE
        paragraph. Every fire is attributed: channel + value + operative bar
        printed, per-channel counts logged as lr_ctrl/fires_* -- reverse-
        engineering the firing channel from scale discontinuities (hnh70s0g)
        is not a diagnosis workflow to repeat."""
        loss_monitors, grad_monitor = self._spikes()
        step = self.modeller.step_ind
        fired = False
        if current_loss is not None and step_type in loss_monitors:
            trig = loss_monitors[step_type].record(current_loss, step)
            if trig:
                self._note_fire(f'{step_type}_loss', trig)
                fired = True
        if grad_norm is not None and math.isfinite(grad_norm):
            trig = grad_monitor.record(grad_norm, step)
            if trig:
                self._note_fire('grad_norm', trig)
                fired = True
        return fired

    def _note_fire(self, channel: str, trig):
        """Attribution: say WHICH tripwire fired and against what bar."""
        self._fire_counts[channel] = self._fire_counts.get(channel, 0) + 1
        base = trig.long_baseline
        bar = 'non-finite' if trig.reason == 'non-finite' else (
            f'bar {base:.4g} (floored median) x factor' if base is not None else '?')
        print(f"lr_ctrl tripwire FIRED: {channel} [{trig.reason}] value {trig.value:.4g}, "
              f"{bar}, fire #{self._fire_counts[channel]} on this channel "
              f"(step {self.modeller.step_ind})")

    def reset_spike_monitors(self, names):
        """Stage-transition hook (protocol.StageProtocol.advance, part of the
        automatic optimization reset every transition runs). The named
        branches' loss windows describe the OUTGOING stage's loss scale -- a
        stale ceiling for the incoming stream -- so those reset. EVERY monitor
        (including grad_norm, and branches not named) still gets a cooldown: a
        stage transition can cause transient turbulence anywhere even when
        that branch's own loss definition didn't change."""
        loss_monitors, grad_monitor = self._spikes()
        step = self.modeller.step_ind
        for name in names:
            if name in loss_monitors:
                loss_monitors[name].reset()
        for mon in loss_monitors.values():
            mon.fire_cooldown(step)
        grad_monitor.fire_cooldown(step)

    # ------------------------------------------------------------ adapt (probe)

    def _draw_probe(self):
        """Draw and freeze a small batch of REPLAYED (not resampled)
        trajectories from replay_buffer. Mirrors draw_replay_sample's body
        exactly, parameterized on probe_size instead of the live (possibly
        grown) training batch_size, so the probe's cost stays constant across
        the run. Only replay_buffer carries stored states -- draw_bwd_sample
        regenerates its trajectory fresh every step regardless of sampling
        mode, so 'fixed inputs' is only available here -- which means the
        probe is unavailable until replay_buffer has samples (early phase 1,
        typically). That's a bootstrap gap, not a bug: ADAPT simply holds
        scale flat while polling, the same graceful-degradation idiom this
        codebase already uses for dormant branches elsewhere."""
        m = self.modeller
        if not (hasattr(m, 'replay_buffer') and len(m.replay_buffer) > 0):
            return None
        size = min(int(self._cfg('probe_size', 32)), len(m.replay_buffer))
        if size < 2:
            return None
        with torch.no_grad():
            mol_batch, traj, inds = next(m.replay_buffer.loader(
                batch_size=size, mode='graphs', repeats=1, return_inds=True,
                weighted=False, temperature=0.1, beta=1.0, return_traj=True))
            traj = traj.to(m.device)
            mol_batch = mol_batch.to(m.device)
            mol_batch, log_T_tensor, sg_inds, zps, condition, condition_id = \
                m.energy_function.condition_samples(mol_batch, repeats=1)
        return {'traj': traj.detach(), 'condition': condition, 'mol_batch': mol_batch}

    def _score_probe(self, probe):
        """log_pf/log_pb of the CURRENT policy against the frozen probe
        trajectories, under no_grad -- get_traj_replay reads states from
        `trajectory` rather than sampling them, so this never resamples and
        never touches the training computation graph."""
        m = self.modeller
        discretizer = get_discretizer(m.args.integrator)
        with torch.no_grad():
            states, log_pfs, log_pbs, log_flow = m.gfn_model.get_traj_replay(
                probe['traj'], discretizer, probe['condition'], probe['mol_batch'],
                return_gauss_params=False, freeze_policy=False)
            y = torch.cat([log_pfs.sum(-1), log_pbs.sum(-1)], dim=0).detach().cpu()
        return y

    def _probe_tick(self, st):
        """One ADAPT reading: draw the probe if needed, score it, and fold
        the resulting displacement into the running EMAs. Returns the
        coherence EMA to drive the climb/ease step, or None if no probe is
        available yet (ADAPT should hold flat) or this is the first reading
        (nothing to compare against yet)."""
        if self._probe_src is None:
            self._probe_src = self._draw_probe()
        if self._probe_src is None:
            return None
        y = self._score_probe(self._probe_src)
        if not torch.isfinite(y).all():
            # a non-finite probe score is itself spike material, but that's
            # check_spike's job (loss/grad channels already cover it); here
            # just decline to adapt off garbage and try a fresh probe next tick
            self._probe_src = None
            st['probe_y'] = None
            return None

        a = 1.0 / max(1, int(self._cfg('ema_horizon', 20)))
        y_prev = st.get('probe_y')
        st['probe_y'] = y
        if y_prev is None or y_prev.shape != y.shape:
            st['probe_d_ema'] = torch.zeros_like(y)
            st['probe_mag_ema'] = 0.0
            st['probe_coh_ema'] = 0.0
            return None  # need a prior reading to form a displacement

        d = y - y_prev
        d_ema = st.get('probe_d_ema')
        if d_ema is None or d_ema.shape != d.shape:
            d_ema = torch.zeros_like(d)
        d_ema_norm = d_ema.norm()
        coh = torch.dot(d, d_ema) / (d.norm() * d_ema_norm) if d_ema_norm > 1e-12 else torch.zeros(())
        coh = float(torch.nan_to_num(coh, nan=0.0, posinf=0.0, neginf=0.0))
        mag = float(d.pow(2).mean().sqrt())

        st['probe_d_ema'] = (1 - a) * d_ema + a * d
        st['probe_mag_ema'] = (1 - a) * st.get('probe_mag_ema', 0.0) + a * mag
        st['probe_coh_ema'] = (1 - a) * st.get('probe_coh_ema', 0.0) + a * coh
        return st['probe_coh_ema']

    def _invalidate_probe(self, st):
        """Force a fresh probe baseline: the next _probe_tick draws a new
        source (if the buffer's contents have moved on) and treats the next
        reading as a first sample rather than diffing against a baseline
        that no longer describes the current policy. Required after any
        weight-discontinuous event (a checkpoint rewind) and any point where
        the run's own semantics change under it (phase transition, forced
        re-warm) -- otherwise the next displacement compares two unrelated
        policies and both the magnitude and coherence signals are garbage."""
        self._probe_src = None
        st['probe_y'] = None
        st['probe_d_ema'] = None
        st['probe_mag_ema'] = 0.0
        st['probe_coh_ema'] = 0.0

    # ------------------------------------------------------------------ actuator

    def _apply_lrs(self, st):
        """lr = configured base x scale per group, floored at min_lr -- EXCEPT
        the flow (Z head) groups, pinned flat at lr_flow. Unchanged from v4;
        see class docstring."""
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

    def _cut(self, st, new_scale, cooldown_mult: int = 1):
        """Multiplicative decrease, arm a cooldown (scaled by cooldown_mult for
        repeat-fire escalation), and force a fresh probe baseline (see
        _invalidate_probe -- the scale just moved, so the next displacement
        reading must not be diffed against pre-cut history)."""
        m = self.modeller
        floor = m.args.min_lr / m.args.lr_policy
        st['scale'] = float(max(new_scale, floor))
        st['cooldown_until'] = st['tick'] + self._cfg('cooldown_ticks', 20) * max(1, cooldown_mult)
        self._invalidate_probe(st)

    # ------------------------------------------------------------------ state

    def _fresh_state(self, phase):
        m = self.modeller
        return {
            'ver': 5,  # v5 = global-scale + function-space ADAPT/FIRE -- invalidates v1-v4 state
            'phase_seen': phase,
            'scale': 1.0 / m.args.lr_warmup_ratio,  # warmup start; ramps to 1.0 (= configured LR)
            'warmup_done': False,
            'tick': 0,
            'cooldown_until': 0,
            'probe_y': None,
            'probe_d_ema': None,
            'probe_mag_ema': 0.0,
            'probe_coh_ema': 0.0,
        }

    def _state(self):
        m = self.modeller
        st = getattr(m, 'lr_ctrl', None)
        if not isinstance(st, dict) or st.get('scale') is None or st.get('ver') != 5:
            st = self._fresh_state(m.phase)
            m.lr_ctrl = st
        elif st.get('phase_seen') != m.phase:
            # phase change: not an LR event, so keep the running scale (no
            # re-warmup), but the loss/log-prob levels differ across the
            # boundary -- cool down and force a fresh probe baseline so the
            # jump can't spurious-cut or spurious-climb.
            st['phase_seen'] = m.phase
            st['cooldown_until'] = st['tick'] + self._cfg('cooldown_ticks', 20)
            st['warmup_done'] = True
            self._invalidate_probe(st)
        return st

    def rearm_warmup(self):
        """Re-run the blind warmup ramp (1/lr_warmup_ratio -> 1.0 over
        warmup_ticks) from here, as if the run had just started. Called by
        protocol.StageProtocol.advance as part of the automatic optimization
        reset at EVERY stage transition: the optimizers were just rebuilt onto
        a loss surface with different curvature, so the first steps must not
        land at the full operating LR with empty Adam moments. Also forces a
        fresh probe baseline -- the boundary changes what the displacement
        readings mean.

        Must be called with the stage (and thus m.phase) already switched to
        its post-transition value, so _state()'s stage-change branch (which
        forces warmup_done) runs first and this overrides it rather than the
        other way round."""
        m = self.modeller
        if not self.enabled:
            return  # disabled: LRs sit flat at configured values, nothing to ramp
        st = self._state()
        st['warmup_done'] = False
        st['tick'] = 0
        st['scale'] = 1.0 / m.args.lr_warmup_ratio
        st['cooldown_until'] = 0
        self._invalidate_probe(st)
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

        if not self.enabled:
            # ADAPT is off; FIRE (check_spike/on_explosion) is NOT gated by
            # this flag and keeps running regardless -- same always-on
            # rewind-on-explosion behaviour this class has always had, only
            # the climbing/warming half is a switch. Flat at the configured
            # base LRs, no warmup, no probe.
            st['scale'] = 1.0
            self._apply_lrs(st)
            self._emit(st, warmup=False)
            return m.optimizers['fwd'].param_groups[0]['lr']

        warmup_ticks = int(self._cfg('warmup_ticks', 100))
        if not st.get('warmup_done'):
            # blind exponential ramp (1/warmup_ratio -> 1.0). Keep the probe
            # warming in the background (cheap, and means ADAPT has a real
            # baseline the instant warmup ends) without acting on it yet --
            # the LR itself is moving, so a displacement reading here would
            # be measuring the ramp, not the model.
            self._probe_tick(st)
            frac = min(1.0, st['tick'] / max(1, warmup_ticks))
            st['scale'] = (1.0 / m.args.lr_warmup_ratio) ** (1.0 - frac)
            if st['tick'] >= warmup_ticks:
                st['warmup_done'] = True
                st['scale'] = 1.0
            self._apply_lrs(st)
            self._emit(st, warmup=True)
            return m.optimizers['fwd'].param_groups[0]['lr']

        # ADAPT regime: continuous climb/ease driven by function-space
        # displacement coherence. No PERMANENT ceiling -- but two
        # evidence-based restraints on the climb (aijrfwuy: coherence stayed
        # honestly positive all the way into a scale-1.94 detonation that took
        # 13 fires and an ~8k-step repair; coherence measures update
        # AGREEMENT, not safety margin, so the climb must not outrun the
        # tripwires that do measure damage):
        #   1. HEADROOM TAPER: above scale 1.0 the climb gain drops to
        #      climb_gain_above_base (easing keeps full gain) -- beyond the
        #      configured LR is exploration, and a ~4x slower approach gives
        #      the leading grad tripwire many cheap chances to fire NEAR the
        #      edge instead of deep past it.
        #   2. RECENT-FIRE CEILING: while any fire is inside
        #      fire_memory_steps, the climb cannot pass the most conservative
        #      post-cut level in memory ("the level we cut TO is the level we
        #      now believe"). Expires with the memory -- a cooling-off, not
        #      the v4 ratchet.
        coh = self._probe_tick(st)
        if not in_cooldown and coh is not None:
            gain = self._cfg('adapt_gain', 0.02)
            if coh > 0 and st['scale'] > 1.0:
                gain = self._cfg('climb_gain_above_base', 0.25 * gain)
            floor = m.args.min_lr / m.args.lr_policy
            new_scale = max(floor, st['scale'] * math.exp(gain * coh))
            if new_scale > st['scale']:
                horizon = self._cfg('fire_memory_steps', 2000)
                live = [s for (t, s) in self._fire_scales
                        if m.step_ind - t <= horizon]
                if live:
                    new_scale = min(new_scale, max(st['scale'], min(live)))
            st['scale'] = new_scale

        self._apply_lrs(st)
        self._emit(st, warmup=False)
        return m.optimizers['fwd'].param_groups[0]['lr']

    def _emit(self, st, warmup):
        self._report = {
            'lr_ctrl/scale': st['scale'],
            'lr_ctrl/warmup': float(warmup),
            'lr_ctrl/probe_coh_ema': st.get('probe_coh_ema', 0.0),
            'lr_ctrl/probe_mag_ema': st.get('probe_mag_ema', 0.0),
            'lr_ctrl/probe_available': float(self._probe_src is not None),
            'lr_ctrl/cooldown': float(st['tick'] < st['cooldown_until']),
        }
        for channel, n in self._fire_counts.items():
            self._report[f'lr_ctrl/fires_{channel}'] = n
        horizon = self._cfg('fire_memory_steps', 2000)
        live = [s for (t, s) in self._fire_scales
                if self.modeller.step_ind - t <= horizon]
        self._report['lr_ctrl/fire_ceiling'] = min(live) if live else float('nan')

    def report(self):
        return dict(self._report)

    def on_explosion(self, count: int = 1):
        """fire_loss_spike hook. Runs AFTER the best-checkpoint rewind
        restored lr_ctrl from healthy times, so the cut applies to the
        pre-damage scale. Same multiplicative cut as before, plus a cooldown
        and a fresh probe baseline (_cut).

        count is the number of TERMINAL rewinds so far (1 for an ordinary
        loss or grad-norm spike) and compounds the cut to cut_ratio**count.
        The rewind restores this controller's scale from a checkpoint written
        while the run was still healthy, which necessarily erases the
        evidence that this very scale already killed the policy -- so a
        single flat cut is undone by the next climb and the run walks back
        into the same detonation (djr13t0j's lr_fwd sawtooth, the original
        motivation for this ratchet).

        ORDINARY fires now escalate the same way, via the instance-held fire
        memory (_fire_steps, rewind-proof by construction -- see __init__):
        each additional fire within fire_memory_steps deepens the cut by one
        more cut_ratio factor and stretches the cooldown, so a repeatedly
        detonating run cools longer and retries lower instead of flat-cutting
        into the same wall every cooldown. The memory DECAYS -- fires older
        than the horizon drop out -- so this is a cooling-off period, not a
        permanent ceiling: once fires stop, ADAPT climbs freely again (a
        one-way ratchet's fixed point is a permanently strangled LR, the
        exact deadlock shape the threshold anneals also guard against).
        """
        st = self._state()
        step = self.modeller.step_ind
        horizon = self._cfg('fire_memory_steps', 2000)
        self._fire_steps = [s for s in self._fire_steps if step - s <= horizon]
        self._fire_scales = [(t, s) for (t, s) in self._fire_scales if step - t <= horizon]
        repeat = min(len(self._fire_steps), int(self._cfg('fire_escalation_cap', 4)))
        self._fire_steps.append(step)
        exponent = max(count, 1) + repeat
        self._cut(st, st['scale'] * (self._cfg('cut_ratio', 0.5) ** exponent),
                  cooldown_mult=1 + repeat)
        # the level we cut TO is the level we now believe safe: it becomes a
        # climb ceiling for as long as this fire stays in memory (see step())
        self._fire_scales.append((step, st['scale']))
        # a fire ENDS the blind warmup ramp. The ramp's mission -- approach the
        # configured LR safely on cold Adam moments -- is over the moment
        # something detonates below its target: the stability edge is now known
        # to sit under the ramp's destination, and the moments are warm. The
        # rewind restores lr_ctrl from a mid-warmup 'best', so without this the
        # ramp resumes immediately (it never consults the cooldown) and
        # bulldozes back to the edge at a ~150-step doubling time, ignoring
        # every escalated cool-off (7laa8lbl: four ramp->knee->fire cycles at
        # the same scale~0.11 knee). Ending warmup hands recovery to ADAPT,
        # which honors the cooldown and climbs at adapt_gain per tick only
        # while updates cohere -- the damped approach escalation intends.
        st['warmup_done'] = True
        if repeat:
            print(f"lr_ctrl: fire #{len(self._fire_steps)} within {horizon} steps -- "
                  f"escalating: cut {self._cfg('cut_ratio', 0.5) ** exponent:.3g}x, "
                  f"cooldown x{1 + repeat}")
        self._apply_lrs(st)

