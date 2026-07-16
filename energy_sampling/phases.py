"""
Phase-transition machinery for the staged training protocol.

Training moves through up to three phases:
  1. MLE/TBC warm-start on the dataset prior (unconditional by design)
  2. variance conditioning (conditional runs only): condition-grouped
     VarGrad tightens within-condition Var(log w) before Z ever trains
  3. on-policy equilibration: TB with a learned (bootstrapped) Z

PhaseController owns the discrete transitions between them -- phase1to2,
phase2to3, and the direct phase1to3 route -- plus the phase-2 fwd/bwd
balance tick. Every transition starts by checkpointing the untouched
end-state of the outgoing phase (_snapshot_pre_transition -- the reload
point for experimenting with transition behavior), then makes the same
three moves with route-specific values: flip the loss schedules (_flip),
refresh the optimization machinery (_refresh_optimization), and set the
mode fractions; the two routes out of phase 1 additionally freeze the
converged warm-start as THE prior (_snapshot_prior).

The controller is stateless: all training state (phase, loss schedules,
mode fracs, optimizers, monitors, checkpoints) lives on the Modeller, so
checkpoint save/load is untouched by this extraction.
"""

import os
from copy import deepcopy


class PhaseController:
    def __init__(self, modeller):
        self.m = modeller

    """shared building blocks"""

    def _flip(self, mode: str, key: str, value: float):
        """
        Re-point one loss-schedule entry at a new value from this step on:
        schedule[key] = [(0, current), (step_ind, value)]. The (0, current)
        start is inert -- evaluate_schedule only reads the final breakpoint
        once step_ind is reached, and steps never rewind -- it just keeps
        the entry a well-formed piecewise ramp.
        """
        m = self.m
        schedule = getattr(m, f'{mode}_loss_schedule')
        current = getattr(getattr(m.args, f'{mode}_loss_coeffs'), key, 0.0)
        schedule[key] = [(0, float(current)), (m.step_ind, float(value))]

    def _refresh_optimization(self, reset_monitors):
        """
        Shared 'refresh optimization machinery' tail of every transition.
        Reset the named loss-spike monitors -- their windows are full of the
        outgoing phase's loss scale, a stale ceiling for the incoming stream
        -- and put ALL monitors on cooldown across the boundary. Clear the
        best-checkpoint combo record (the outgoing phase's minima shouldn't
        gate the new phase's 'best' saves), rebuild optimizers/schedulers
        from scratch (fresh Adam moments for the new loss surface) with LR
        warmup re-armed, and make the flipped schedules take effect
        immediately via set_loss_coeffs().
        """
        m = self.m
        for name in reset_monitors:
            m.loss_monitors[name].reset()
        for monitor in m.loss_monitors.values():
            monitor.fire_cooldown(m.step_ind)
        m.combo_loss_record = []
        m.lr_warmup_finished = False
        m.init_schedulers_optimizers()
        m.set_loss_coeffs()

    def _snapshot_pre_transition(self, tag: str):
        """
        Checkpoint the converged end-state of the outgoing phase BEFORE the
        transition mutates anything (loss schedules, optimizers, Z bootstrap,
        phase index), so transition behavior can be re-run from this point
        under changed code/config without retraining the whole phase.
        Everything the exit gates read is already part of the checkpoint
        (mle_gate's latched 'flat', metric_tracker's tbc / logw_std_within
        EMAs, condition_log_z), so on reload the gate re-fires by itself at
        the next eval; stamping request_eval into the SAVED state (live value
        restored right after, so this run is unaffected) pulls that eval to
        the first post-resume step. The transition thus replays through the
        normal eval -> gate -> transition path -- with the fresh eval metrics
        it consumes (jensen_z, wass_debiased), which are never checkpointed --
        after a single training step of drift. The post-transition
        checkpoints ('prior'/'thermalized') are unchanged and remain the
        seamless-resume record.

        request_eval nominally belongs to the phase-1 MLE gate, but the train
        loop's eval trigger reads it unconditionally, so it serves the
        phase-2 exit snapshot equally well.
        """
        m = self.m
        prev = m.mle_gate.get('request_eval', False)
        m.mle_gate['request_eval'] = True
        # with_buffers freezes this snapshot's own buffer sidecar: the rolling
        # one is overwritten at the next eval, which would leave a replay from
        # here running against buffers from a phase this checkpoint predates
        m.checkpointer.save(tag, with_buffers=True)
        m.mle_gate['request_eval'] = prev

    def _snapshot_prior(self):
        """
        Leaving phase 1 (either route): freeze the converged MLE warm-start
        as THE prior. Checkpoint it, keep a frozen eval copy of the EMA
        model for backward sampling, switch backward draws onto it, and
        delete the phase-1 'best' checkpoint -- it converged, 'prior'
        supersedes it, and the best-gate restarts on the cleared combo
        record.
        """
        m = self.m
        m.hit_prior = True
        m.bwd_sampling_mode = 'prior'
        m.checkpointer.save('prior')
        m.prior_model = deepcopy(m.ema_model)
        m.prior_model.eval()
        # tolerate a missing 'best': when the transition is REPLAYED from a
        # reloaded 'phase1_exit' checkpoint, the original run already deleted
        # it here (or a renamed resume never wrote one)
        best_path = m.checkpointer.path_for('best')
        if os.path.exists(best_path):
            os.remove(best_path)

    """phase-2 balance tick"""

    def phase2_balance_step(self):
        """
        Phase-2 fwd/bwd mode-frac balancer: allocate rollout share toward
        whichever direction's WITHIN-condition log w spread is lagging, so the
        two VarGrad objectives co-descend rather than one winning the shared
        parameters. The two share the Boltzmann fixed point where both vanish
        -- no goal conflict, only optimization imbalance -- so proportional
        feedback on the observed per-direction spreads suffices, and the
        controller needn't know WHY one side is stronger: it compensates for
        the sum of all causes.

        Reads 'logw_std_within' (the condition-grouped std from _update_rolling)
        NOT quick_tb_stats' batch-wide 'logw_std' -- the latter is dominated by
        between-condition log Z(c) spread, which VarGrad can't touch, so
        balancing on it steered the controller by cross-condition Z structure
        rather than the objective (the fwd-good/bwd-bad asymmetry that read off
        the batch-wide metric was largely that contamination; the clean metrics
        show the two directions converging comparably). When a direction has no
        within-condition signal yet (all-singleton batch -> nan -> unset), hold
        the fracs rather than driving to the floor on missing data. Fracs move
        by EMA nudges (phase2_frac_alpha per 10-step tick) toward the
        spread-proportional target, floored both sides (phase2_frac_floor).
        Same 10-step cadence as the phase-3 controller.
        """
        m = self.m
        if not getattr(m.args, 'phase2_dynamic_frac', False):
            return
        s_f = m.metric_tracker.get('fwd', 'logw_std_within')
        s_b = m.metric_tracker.get('bwd', 'logw_std_within')
        if s_f is None or s_b is None:  # no within-condition signal yet -> hold, don't starve on missing data
            return
        floor = getattr(m.args, 'phase2_frac_floor', 0.01)
        target = s_f / max(s_f + s_b, 1e-8)  # share proportional to remaining spread
        target = min(max(target, floor), 1.0 - floor)
        alpha = getattr(m.args, 'phase2_frac_alpha', 0.05)
        m.fwd_frac = (1.0 - alpha) * m.fwd_frac + alpha * target
        m.bwd_frac = 1.0 - m.fwd_frac

    """transitions"""

    def phase1to2(self, metrics):
        m = self.m
        print("Hit initial KLD threshold. Starting variance conditioning.")
        self._snapshot_pre_transition('phase1_exit')
        m.phase = 2

        "adjust loss coefficients"
        # phase 2 = variance conditioning: make the current policy self-consistent
        # and well-behaved BEFORE Z ever trains -- deliberately NOT thermalization
        # toward the true target; the exit criterion is tightness of log w, not
        # correctness. fwd flips from phase-1 Z-only TB to grouped VarGrad
        # (vg_by_condition), which attacks the cross-terminal component of
        # Var(log w) -- the part phase-1 TBC structurally can't touch (TBC -> 0
        # makes log w a pure function of the terminal, so ALL remaining spread is
        # cross-terminal: empirically hundreds of nats of reward-vs-prior
        # mismatch). bwd swaps MLE for grouped VarGrad too: MLE's fixed point is
        # P_F = mu (the prior), and at P_F = mu, Var(log w) IS the reward-vs-prior
        # mismatch -- full-weight MLE would anchor the policy at exactly the point
        # the exit gate measures distance from, making the gate structurally
        # unreachable. Backward VarGrad has the SAME fixed point as the forward
        # one (zero variance is measure-independent; the off-policy bias lives
        # only in the group center, which nothing consumes) and doubles as the
        # mode-retention anchor: a dropped buffer terminal is an extreme positive
        # log w outlier, the variance's loudest gradient. TBC carries over
        # unchanged. Z stays completely untrained until the phase-3 bootstrap,
        # which the tightened w (via ema_logw) strictly improves.
        self._flip('fwd', 'tb', 0.0)  # phase-1 fwd Z-only TB off; Z untouched until phase 3
        self._flip('fwd', 'vg_lb', getattr(m.args, 'phase2_fwd_vg_lb', 1.0))  # grouped VarGrad ON
        if 'repeats' in m.fwd_loss_schedule:
            # forward VarGrad NEEDS repeats > 1: condition-grouped variance is
            # undefined on singleton groups, and at repeats=1 over a large
            # library every condition appears ~once, so the loss is ~zero
            # gradient (only accidental birthday collisions carry signal). K
            # stochastic forward rollouts per row give K distinct terminals =
            # the within-condition cross-terminal samples VarGrad needs. Costs
            # K x forward energy evals -- restored to 1 at phase 3, where fwd is
            # Z-only TB (no grouping) and the K-tiling would buy nothing.
            self._flip('fwd', 'repeats', getattr(m.args, 'phase2_fwd_repeats', 2.0))
        self._flip('fwd', 'freeze_policy', 0.0)  # the policy must move: VarGrad trains it on-policy
        self._flip('fwd', 'freeze_z', 1.0)  # belt and suspenders: nothing fwd touches Z this phase
        self._flip('bwd', 'mle',
                   getattr(m.args, 'phase2_bwd_mle', 0.0))  # MLE off (or tiny): its P_F=mu anchor fights the gate
        self._flip('bwd', 'vg_lb',
                   getattr(m.args, 'phase2_bwd_vg_lb',
                           1.0))  # grouped bwd VarGrad ON: retention + consistency at buffer terminals
        # NB deliberately NO fwd traj_grads/reward_grads here: forward pathwise
        # gradients have been severely pathological or a null element in every
        # prior test (unlike bwd, where traj_grads are make-or-break) -- fwd
        # trains through density evaluations only, and the balance controller
        # compensates for the resulting pressure asymmetry with throughput

        "refresh optimization machinery"
        self._refresh_optimization(reset_monitors=('fwd', 'bwd'))  # new fwd + bwd losses

        # phase-2 mode split. NB the pressure is structurally lopsided toward bwd
        # even at 0.5/0.5: bwd runs two losses (VarGrad + TBC) on anchored,
        # prebuilt-reward targets with K-tiled (repeats x) rows, while fwd runs
        # one loss on a landscape the bwd re-pricing keeps shifting, and the
        # Huber sign-caps its rare most-informative outliers -- empirically the
        # forward direction gets left behind (fwd logw_std/z_gap re-expanding
        # while bwd's contract). Raise phase2_fwd_frac to compensate.
        fwd_frac = getattr(m.args, 'phase2_fwd_frac', 0.5)
        m.fwd_frac = fwd_frac
        m.bwd_frac = 1.0 - fwd_frac
        m.replay_frac = 0.0

        "save checkpoint"
        self._snapshot_prior()

    def phase2to3(self, metrics):
        print("Variance conditioning complete. Starting on-policy equilibration.")
        self._snapshot_pre_transition('phase2_exit')
        # same phase-3 entry as phase1to3 (shared block), minus what phase1to2
        # already did at the 1->2 boundary: prior snapshot, 'prior' checkpoint,
        # bwd_sampling_mode switch, and deleting the phase-1 'best' checkpoint
        self._activate_phase3_losses(metrics)
        self.m.checkpointer.save('thermalized', with_buffers=True)

    def phase1to3(self, metrics):
        m = self.m
        print("Hit initial KLD threshold. Equilibration.")
        self._snapshot_pre_transition('phase1_exit')
        self._activate_phase3_losses(metrics, train_conditioner=m.uncond_prior_mode())
        "save checkpoint"
        self._snapshot_prior()

    def phase1_to_forward_first(self, metrics):
        """
        Phase-1 exit route for the forward-first protocol (default: no prior
        loaded by path). The MLE warm-start produced a broadly-covering,
        self-consistent policy and its EMA snapshot becomes the prior model --
        EXACTLY the standard 1->3 prior snapshot -- but instead of standard
        equilibration we hand off to the forward-first build-out (stages
        A/B/C). _snapshot_prior sets bwd_sampling_mode='prior' and freezes the
        prior model; engage() flips to the stage-A schedules and sets the
        fracs; then rebuild the optimizers for the new (TB) loss surface, same
        as every other phase transition. No Z bootstrap: stage A learns Z
        on-policy from the warm-started policy's own price.
        """
        m = self.m
        print("MLE warm-start complete. Starting forward-first build-out.")
        self._snapshot_pre_transition('phase1_exit')
        self._snapshot_prior()
        m.forward_first_controller.engage()
        self._refresh_optimization(reset_monitors=('bwd', 'fused'))

    def _activate_phase3_losses(self, metrics, train_conditioner: bool = False):
        """
        Shared phase-3 entry block for phase1to3 (direct route) and phase2to3
        (after variance conditioning): loss schedules, Z warm start, optimizer/
        monitor refresh, and mode fractions. Callers own the route-specific
        work (prior snapshot/checkpoints, bwd_sampling_mode, print).

        train_conditioner: threaded into bootstrap_log_z (see its docstring).
        Only the direct 1->3 route under uncond_prior_mode() sets it -- there
        the conditioner was never trained in phase 1, so the Z(c) fit must be
        allowed to shape it; on the 2->3 route it must stay frozen (phase 2
        gave it policy-relevant structure).
        """
        m = self.m

        "adjust loss coefficients"
        self._flip('bwd', 'mle', 0.0)  # turn off mle
        # tbc: pre-training aid off; optionally retained at a small supplementary
        # weight (tbc_supplementary) -- it's implied at the TB fixed point, so as
        # auxiliary pressure it can't fight convergence, and bwd rewards are
        # prebuilt so its K-tiling costs rollouts only, no extra energy
        tbc_supp = getattr(m.args, 'tbc_supplementary', 0.0)
        self._flip('bwd', 'tbc', tbc_supp)
        if 'repeats' in m.bwd_loss_schedule:  # per-mode repeats configs only; legacy global-repeats configs untouched
            # keep K-tiling only if supplementary tbc still consumes it
            self._flip('bwd', 'repeats', m.args.bwd_loss_coeffs.repeats if tbc_supp > 0 else 1.0)
        # phase-2 variance conditioning off, both directions (no-op on the
        # direct 1->3 route, where vg_lb never turned on)
        self._flip('fwd', 'vg_lb', 0.0)
        self._flip('bwd', 'vg_lb', 0.0)
        if 'repeats' in m.fwd_loss_schedule:
            # drop the phase-2 fwd VarGrad K-tiling: phase-3 fwd is Z-only TB
            # (no condition grouping), so tiling would only pay K x forward
            # energy for nothing. No-op on the direct 1->3 route (fwd repeats
            # was never raised).
            self._flip('fwd', 'repeats', 1.0)
        # turn on tb everywhere...
        # ...except fwd when emp_z owns on-policy Z training (the vargrad-Z
        # experiment): under freeze_policy, per-trajectory fwd TB trains nothing
        # BUT Z, and its gradient is exactly the stalled one emp_z exists to
        # replace -- keeping it on would just mix the stalling pull back into
        # the Z fixed point. emp_z: 0 in the config restores old behavior.
        fwd_tb_on = 0.0 if getattr(m.args.fwd_loss_coeffs, 'emp_z', 0) > 0 else 1.0
        self._flip('fwd', 'tb', fwd_tb_on)  # on-policy TB ACTIVATE (unless emp_z owns Z)
        self._flip('bwd', 'tb', 1.0)  # off-policy TB ACTIVATE
        self._flip('replay', 'tb', 1.0)  # replay TB ACTIVATE

        # on-policy log Z ONLY (train Z, freeze policy + conditioner)
        self._flip('fwd', 'freeze_policy', 1.0)
        self._flip('fwd', 'freeze_z', 0.0)
        # off-policy policy only... but detach Z: no off-policy log Z
        self._flip('bwd', 'freeze_policy', 0.0)
        self._flip('replay', 'freeze_policy', 0.0)
        self._flip('bwd', 'freeze_z', 1.0)
        self._flip('replay', 'freeze_z', 1.0)

        if (not m.gfn_model.full_flow) and (not m.gfn_model.conditional):
            empirical_Z = metrics['eval_fwd/jensen_z']
            m.gfn_model.flow_model.scalar.data.fill_(empirical_Z)  # warm start at the target value
            m.ema_model.flow_model.scalar.data.fill_(empirical_Z)
        else:
            # conditional analog of the unconditional .fill_() above -- a single
            # scalar can't warm-start a per-condition function, so instead fit
            # flow_model's Z(c) directly onto condition_log_z's ema_logw (built up
            # during the MLE warm-start, and tightened by phase-2 variance
            # conditioning when that route is taken) via a short, rollout-free
            # regression. See bootstrap_log_z's docstring for why ema_logw and
            # not ema_log_z_emp.
            m.bootstrap_log_z(train_conditioner=train_conditioner)

        "refresh optimization machinery"
        # fused stepping now runs through phase 2, so its monitor arrives here with a
        # window of VarGrad-scale losses -- stale ceiling for the phase-3 TB stream
        self._refresh_optimization(reset_monitors=('bwd', 'fused'))

        m.bwd_frac = 1.0
        m.fwd_frac = 0.0
        m.replay_frac = 0.0

        self.m.phase = 3

    """forward-first build-out protocol (controller.ForwardFirstController)"""

    def forward_first_stage_a(self):
        """
        Stage-A loss configuration for the forward-first protocol: ordinary
        forward training. fwd TB trains policy AND Z together on-policy,
        replay retains whatever forward has captured, backward is armed but
        dormant (its branch rides at the frac floor; ForwardFirstController
        owns the fracs). Mode collapse of the forward policy is EXPECTED and
        accepted -- the protocol's invariant is calibration-on-covered-
        support, not coverage; backward later grows the support (stage B).

        Deliberately mirrors _activate_phase3_losses' terminal coefficients
        EXCEPT fwd freeze_policy=0, so the whole protocol differs from
        standard phase 3 by exactly one coefficient and stage B/C restore it
        with a single flip. Z is NOT bootstrapped -- it learns inside forward
        TB from the (warm-started) policy's own on-policy price. bwd_sampling_
        mode is 'prior' (set by the caller, engage()): stage-B backward draws
        from the churned prior_buffer, fed by the prior model the phase-1 MLE
        warm-start produced (or a prior loaded by path), NOT the static
        prior_dataset -- an atomic dataset would let backward overfit to spikes
        instead of smooth coverage (the entropy-floor argument).
        """
        m = self.m
        fwd_tb_on = 0.0 if getattr(m.args.fwd_loss_coeffs, 'emp_z', 0) > 0 else 1.0
        self._flip('fwd', 'tb', fwd_tb_on)
        self._flip('fwd', 'freeze_policy', 0.0)  # THE stage-A difference: forward trains its own policy
        self._flip('fwd', 'freeze_z', 0.0)
        self._flip('fwd', 'vg_lb', 0.0)
        self._flip('fwd', 'vg_lme', 0.0)
        if 'repeats' in m.fwd_loss_schedule:
            self._flip('fwd', 'repeats', 1.0)

        tbc_supp = getattr(m.args, 'tbc_supplementary', 0.0)
        self._flip('bwd', 'tb', 1.0)
        self._flip('bwd', 'mle', 0.0)
        self._flip('bwd', 'tbc', tbc_supp)
        self._flip('bwd', 'vg_lb', 0.0)
        self._flip('bwd', 'freeze_policy', 0.0)
        self._flip('bwd', 'freeze_z', 1.0)
        if 'repeats' in m.bwd_loss_schedule:
            self._flip('bwd', 'repeats', m.args.bwd_loss_coeffs.repeats if tbc_supp > 0 else 1.0)

        self._flip('replay', 'tb', 1.0)
        self._flip('replay', 'freeze_policy', 0.0)
        self._flip('replay', 'freeze_z', 1.0)

    def forward_first_stage_b(self):
        """
        Stage-B flip: freeze the forward policy (it is too powerful at
        concentration to leave live while backward carves new support --
        on-policy TB would re-collapse annexed mass faster than backward can
        place it). Forward becomes Z-only, identical to standard phase 3;
        replay carries retention; backward ramps under
        ForwardFirstController's fit-quality gate.
        """
        self._flip('fwd', 'freeze_policy', 1.0)
        self.m.set_loss_coeffs()
