"""
The learning-rate controller: ONE multiplier over the managed policy optimizers,
moved only by the brute-force bracket (`lr_bracket.py`), plus the always-on hard
tripwire.

    lr = base_lr x scale

`scale` is piecewise constant and every value it takes is auditable: it is the
configured burn-in scale, a configured candidate rung under trial, or the rung
the bracket promoted. There is no envelope, no warmup ramp, no decay leg and no
continuous servo underneath it -- a hidden multiplier would mean the rate under
test is not the rate applied, which is precisely how a bracket comes to report a
boundary it never found.

WHAT THIS REPLACED, AND WHY. Controller v8 moved `peak_scale` from a per-stage
declared sensor: `ray` (a line-search optimum alpha*), `hyper` (a hypergradient
cosine) or `plateau`, under a warmup envelope with its own freeze rules, with a
pooled estimator and a periodic re-probe on top. The ray was killed on
2026-08-23 by its own acceptance test -- alpha* is defined as s*/lr, so the slope
of log(alpha*) against log(lr) MUST be -1, and measured 0.00 +- 0.2 across twelve
runs, two stages and 2.7 decades of rate. The sensor was uncorrelated with the
variable it steered. Everything built on it -- the pool, the sweep, the rung
ladder, the alpha target -- inherited that. A cruise that held one rate for 1,200
steps with zero moves was reported as the controller working; it is the null
output, and the null output is what a dead sensor produces.

The bracket cannot fail that way. Its only reading is "did this candidate
detonate", which is not a statistic.

WHAT SURVIVES FROM v8, and it is deliberately little:

  * the ACTUATOR -- `_apply_lrs`, the managed-key rule, the flow-head pin, the
    max_lr rail and the min_lr floor. That layer was never in question.
  * the HARD TRIPWIRE -- non-finite readings and absolute bars. Its bars are no
    longer the only ones: see `lr_bracket_probe.HardFailureBars`, which derives a
    bar from the root's own loss scale, because at 1e9 the shipped bars caught
    numerical death and nothing else.
  * `ray` and `hyper` as OPTIONAL, OFF-BY-DEFAULT DIAGNOSTICS. They no longer
    reach any learning rate, and no canonical config declares one. They are kept
    reachable so a future claim about either can be measured rather than argued
    about; their reporting is off unless a stage explicitly asks.

WHAT IS GONE: `plateau`, the pooled estimator (`lr_pool`), the rung ladder
(`lr_ramp`), the sweep (`lr_sweep`), the warmup envelope and its freeze rules,
the divergence LR cut and its ceiling, the alpha target, the warm restart.
"""

from __future__ import annotations

import math
from collections import deque

from energy_sampling.lr_bracket import BURN_IN, CRUISE, LRBracket
from energy_sampling.lr_bracket_probe import BracketDriver, HardFailureBars
from energy_sampling.utils import is_cuda_oom


class LRController:
    """Owns every learning rate the run writes, and hosts the bracket."""

    CHANNELS = ('fwd', 'bwd', 'replay', 'fused')

    #: Policy optimizer keys -> the args attribute holding their base LR. The
    #: flow head is deliberately absent: it is pinned, not scheduled.
    _POLICY_BASE = {'fwd': 'lr_policy', 'bwd': 'lr_back', 'replay': 'lr_replay',
                    'fused': 'lr_fused'}

    #: ver 10 invalidates every earlier lr_ctrl dict. A v8 state carries
    #: `peak_scale`, an envelope, a freeze latch and a ramp clock, none of which
    #: has a meaning here. DISCARD, never reinterpret.
    _STATE_VER = 10

    def __init__(self, modeller):
        self.modeller = modeller
        self._report = {}
        self._divergences = 0            # every rewind+cut response (the counting seat)
        self._moderate_fires = 0         # the excursion-tier subset of the above
        self._fire_cooldown_until = 0
        self._calibrations = 0
        self._hypergrads = 0
        self._lr_capped_groups = 0
        self._lr_floored_groups = 0
        self._skip_steps = 0          # promoted steps the host loop must skip
        self._last_scale_reason = 'init'

        cfg = self._cfg_node()
        # NO `lr_control` BLOCK AT ALL means no LR control was configured, and
        # the honest reading of that is "run the base rates", not "bracket".
        # Defaulting the mode to `bracket` here made such a config LOAD CLEANLY
        # -- every invariant abstains, because with no `auto` rate there is
        # nothing for them to judge -- and then raise on the absent candidate
        # grid when this constructor ran, i.e. a queued job dying at step 0 with
        # a message about a key the config never mentions.
        default_mode = 'bracket' if cfg is not None else 'fixed'
        hf = getattr(cfg, 'hard_failure', None)
        self.bars = HardFailureBars(
            loss_excursion_k=float(_get(hf, 'loss_excursion_k', 10.0)),
            grad_excursion_x=float(_get(hf, 'grad_excursion_x', 100.0)),
            loss_abs=_get(hf, 'loss_abs', 1.0e6),
            grad_abs=_get(hf, 'grad_abs', 1.0e6),
            root_window=int(_get(hf, 'root_window', 200)),
            min_observations=int(_get(hf, 'min_observations', 20)))
        # THE BARS ARE DRAWN TWICE, because they do two jobs at two different
        # rates. The root bars are fitted to burn-in -- a deliberately cold rate
        # -- and that is the right scale for judging a TRIAL, since every trial
        # restores that same root and is comparable to it. It is the wrong scale
        # for the LIVE TRIPWIRE, which then runs for the rest of the stage at the
        # promoted rate: a hotter rate moves the loss more for ordinary reasons,
        # so a bar fitted cold is crossed by healthy training and the response is
        # a rewind charged to max_reloads_per_1k_steps.
        self._cruise_rederive = bool(_get(hf, 'cruise_rederive', True))
        self._cruise_settle_steps = int(_get(hf, 'cruise_settle_steps', 200))
        if self._cruise_settle_steps < 0:
            raise ValueError(
                f'lr_control.hard_failure.cruise_settle_steps must be >= 0, got '
                f'{self._cruise_settle_steps}')
        self.bracket = LRBracket(
            mode=str(_get(cfg, 'mode', default_mode)),
            burn_in_steps=int(_get(cfg, 'burn_in_steps', 3000)),
            burn_in_scale=float(_get(cfg, 'burn_in_scale', 0.05)),
            min_root_bias_correction=float(_get(cfg, 'min_root_bias_correction', 0.9)),
            candidate_scales=_get(cfg, 'candidate_scales', ()) or (),
            trial_steps=int(_get(cfg, 'trial_steps', 150)),
            safety_rungs=int(_get(cfg, 'safety_rungs', 1)),
            repeat_every=int(_get(cfg, 'repeat_every', 0) or 0),
            boundary_confirm_repeats=int(_get(cfg, 'boundary_confirm_repeats', 1)),
            boundary_densify=bool(_get(cfg, 'boundary_densify', False)),
            # 1.0 when nothing is configured: the base rates, unmodified.
            fixed_scale=_get(cfg, 'fixed_scale', None if cfg is not None else 1.0),
            loss_abs=self.bars.loss_abs,
            grad_abs=self.bars.grad_abs,
            trial_settle_steps=int(_get(cfg, 'trial_settle_steps', 10)),
            logz_detour_nats=_get(cfg, 'logz_detour_nats', 2.0))
        self.driver = None            # a live BracketDriver, or None

        # THE ROOT'S LOSS SCALE, collected during burn-in. Kept on the INSTANCE
        # rather than in `lr_ctrl`: it is a within-process observation, and a
        # window restored from a checkpoint would describe a rate and a stage
        # this process never ran. A resume therefore refills it before the
        # bracket arms -- which is a bounded wait, because it fills at one
        # observation per step.
        self._loss_history = {c: deque(maxlen=self.bars.root_window)
                              for c in self.CHANNELS}
        self._grad_history = deque(maxlen=self.bars.root_window)
        #: The post-promotion redraw, or None when nothing is pending. It carries
        #: its own clock rather than counting from the promotion step, because
        #: `_open_bracket` returns a horizon the host loop then SKIPS -- so the
        #: step the run resumes at is not the step promotion happened at. The
        #: clock starts on the first cruise tick instead.
        self._cruise_bar = None
        #: Which rate the LIVE bars were fitted at: the cold burn-in one, or the
        #: promoted one. Published, because the two answer the same question with
        #: different tolerances and a reader of a rewind needs to know which.
        self._bars_redrawn = False

        self._check_schema()
        self._check_rails()
        self._check_bar_window()
        # MATERIALISE THE STATE NOW, so `stage_entry_step` is stamped at
        # construction rather than whenever something first happens to call
        # `_state()`. Burn-in length is an EXACT quantity -- "exactly
        # burn_in_steps at burn_in_scale" -- and a lazily stamped entry step
        # makes it depend on call order, which is how a run burns in for
        # burn_in_steps minus one and nothing says so. A checkpoint restore
        # overwrites this dict wholesale afterwards, which is correct: the
        # restored entry step is the one that stage really started at.
        self._state()

    # ------------------------------------------------------------------ config

    def _cfg_node(self):
        return getattr(self.modeller.args, 'lr_control', None)

    def _cfg(self, name, default=None):
        return _get(self._cfg_node(), name, default)

    def _managed_keys(self):
        """Config keys the bracket owns -- those written `auto`, recorded by
        resolve_derived_config at load. Empty set = the controller reads and logs
        but actuates nothing, which is its own documented control arm."""
        return set(getattr(self.modeller.args, 'lr_servo_managed', ()) or ())

    def managed_optimizer_keys(self):
        """The optimizers `scale` actually reaches. `lr_bracket_probe` asks,
        because a bias-correction refusal taken over an optimizer the bracket
        does not steer would be a refusal about an irrelevant fact."""
        managed = self._managed_keys()
        return {k for k, base in self._POLICY_BASE.items() if base in managed}

    def stepping_optimizer_keys(self):
        """The managed optimizers THIS STAGE ACTUALLY STEPS.

        MANAGED IS NOT THE SAME AS STEPPING, and conflating them welded the
        bracket shut on every canonical config. `mk_dev` writes all four policy
        rates `auto`, so the managed set is {fwd, bwd, replay, fused} -- but a
        stage runs ONE train_mode, so `train_prior` (bwd) steps only 'bwd' and
        `equilibration` (fused) steps only 'fused'. The other three optimizers
        exist, hold no state, and report an Adam step counter of None.

        The bias-correction check reads the WORST managed optimizer and maps a
        counter of None to 0.0 -- correctly, since an optimizer that has never
        stepped is the extreme case of an unequilibrated one. Taken over
        optimizers the stage never steps, that made the worst value 0.0 on every
        stage of every canonical config, so the bracket refused every time and
        reported it as caution. A mechanism that always declines and says it is
        being careful is precisely the failure this design exists to avoid, and
        no test caught it because the driver's fake trainer has exactly one
        managed optimizer and always steps it.

        Mirrors `step_loss`: a bwd/fwd/replay stage steps that branch then
        'flow'; a fused stage steps 'fused' alone (its param groups already
        carry the flow head at lr_flow). 'flow' is never managed.
        """
        stage = getattr(getattr(self.modeller, 'protocol', None), 'stage', None)
        mode = getattr(stage, 'train_mode', None)
        if mode is None:
            return self.managed_optimizer_keys()
        stepping = {'fused'} if mode == 'fused' else {mode}
        return self.managed_optimizer_keys() & stepping

    #: Everything `lr_control` may contain. CLOSED, and it has to be: the whole
    #: `adaptive_lr` block was retired into this one, so the natural mistake when
    #: migrating by hand is to carry a key across -- `warmup_steps`, `bounds`,
    #: `alpha_target`, `divergence_cut`. The retired-key gate cannot see those,
    #: because it matches on the OLD path, so without this they would land in a
    #: live block and be silently ignored: a config that reads as configuring a
    #: warmup and behaves as having none.
    _KEYS = frozenset({
        'mode', 'seed_lr', 'control_flow_lr',
        'burn_in_steps', 'burn_in_scale', 'min_root_bias_correction',
        'candidate_scales', 'trial_steps', 'safety_rungs', 'repeat_every',
        'boundary_confirm_repeats', 'boundary_densify', 'fixed_scale',
        'verbose', 'hard_failure', 'ray_calibration',
        'trial_settle_steps', 'logz_detour_nats',
        'fire_cut_factor', 'fire_cooldown_steps',
    })
    _HARD_FAILURE_KEYS = frozenset({
        'loss_excursion_k', 'grad_excursion_x', 'loss_abs', 'grad_abs',
        'root_window', 'min_observations',
        'cruise_rederive', 'cruise_settle_steps',
    })

    def _check_schema(self):
        """Refuse an unknown key under `lr_control`. See `_KEYS`."""
        for node, known, where in ((self._cfg_node(), self._KEYS, 'lr_control'),
                                   (getattr(self._cfg_node(), 'hard_failure', None),
                                    self._HARD_FAILURE_KEYS, 'lr_control.hard_failure')):
            if node is None:
                continue
            unknown = sorted(k for k in vars(node)
                             if not k.startswith('_') and k not in known)
            if unknown:
                raise ValueError(
                    f'{where}: unknown key(s) {unknown}. Expected a subset of '
                    f'{sorted(known)}. `adaptive_lr` was retired INTO this block, so '
                    f'a key carried across by hand lands here and would otherwise be '
                    f'ignored in silence -- the config would read as configuring '
                    f'something the run does not have.')

    def _check_rails(self):
        """max_lr vs min_lr CANNOT be resolved by clamp order -- whichever is
        applied second wins, so one bound is silently defeated and the run trains
        at a rate neither describes."""
        cap = getattr(self.modeller.args, 'max_lr', None)
        if cap is None:
            return
        cap = float(cap)
        floor = float(getattr(self.modeller.args, 'min_lr', 0.0) or 0.0)
        if cap <= 0.0:
            raise ValueError(f'max_lr = {cap:g} must be positive, or null for no cap.')
        if cap < floor:
            raise ValueError(
                f'max_lr = {cap:g} is below min_lr = {floor:g}. Clamping to both is not '
                f'possible: whichever is applied second wins and the other bound is '
                f'silently defeated. Raise max_lr or lower min_lr.')

    def announce(self):
        b = self.bracket
        managed = ','.join(sorted(self._managed_keys())) or 'NOTHING (control arm)'
        if b.mode == 'fixed':
            print(f'lr_ctrl (bracket v10): FIXED mode -- burn-in {b.burn_in_steps} steps '
                  f'at scale {b.burn_in_scale:g}, then scale {b.fixed_scale:g} held for '
                  f'the stage. No trials, no re-bracketing. Managed: {managed}')
            return
        print(f'lr_ctrl (bracket v10): burn-in {b.burn_in_steps} steps at scale '
              f'{b.burn_in_scale:g} (min root bias correction '
              f'{b.min_root_bias_correction:g}), then {len(b.candidate_scales)} fixed-LR '
              f'trials of {b.trial_steps} steps each over scales '
              f'{[round(s, 6) for s in b.candidate_scales]}; select {b.safety_rungs} '
              f'rung(s) below the lowest failure'
              + (f', confirmed x{b.boundary_confirm_repeats}'
                 if b.boundary_confirm_repeats else ', UNCONFIRMED (one failure decides)')
              + (', densified' if b.boundary_densify else '')
              + f'. Re-bracket: '
              + ('once per stage' if b.repeat_every <= 0
                 else f'every {b.repeat_every} promoted steps')
              + f'. Managed: {managed}')
        print(f'lr_ctrl: bracket cost ~'
              f'{len(b.candidate_scales) * b.trial_steps} discarded steps per cycle')

    # -------------------------------------------------------------- actuation

    def _state(self):
        m = self.modeller
        st = getattr(m, 'lr_ctrl', None)
        if not isinstance(st, dict) or st.get('ver') != self._STATE_VER:
            if isinstance(st, dict) and st.get('ver') is not None:
                print(f"lr_ctrl: discarding stale state ver={st.get('ver')} "
                      f'(this controller is ver={self._STATE_VER}); the bracket has no '
                      f'peak_scale, envelope or ramp clock to reinterpret it into.')
            st = {'ver': self._STATE_VER,
                  'scale': float(self.bracket.scale_now()),
                  'stage_entry_step': int(getattr(m, 'step_ind', 0) or 0),
                  'bracket': None}
            m.lr_ctrl = st
            self.bracket.load_state_dict(None)
        elif st.get('bracket') is not None and not getattr(self, '_bracket_loaded', False):
            restored = self.bracket.load_state_dict(st['bracket'])
            self._bracket_loaded = True
            if restored and getattr(self.bracket, 'resumed_mid_bracket', False):
                print('lr_ctrl: resumed inside a bracket cycle. The candidate states '
                      'lived in host memory and died with the process, so the cycle is '
                      're-armed from the resumed state rather than half-restored -- the '
                      'resumed state is already mature, so burn-in is not repeated '
                      'beyond refilling the loss window the bars are derived from.')
        st.setdefault('scale', float(self.bracket.scale_now()))
        return st

    def set_scale(self, scale, why=None):
        """Move the one multiplier, and say so. Every caller is the bracket, a
        stage transition, or (owner decision 2026-08-24) a FIRE RESPONSE -- the
        unified rewind-and-cut (owner 2026-08-25), which stands until the
        scheduled re-race re-measures the rate."""
        st = self._state()
        st['scale'] = float(scale)
        self._last_scale_reason = str(why or 'set')
        self._apply_lrs(st)
        return st['scale']

    @property
    def scale(self):
        return float(self._state()['scale'])

    def live_rates(self) -> dict:
        """What each managed optimizer's first group is ACTUALLY set to. The
        bracket logs this beside the candidate identifier, because 'the rate
        under test' is a claim about the optimizer, not about the config."""
        out = {}
        for key, opt in self.modeller.optimizers.items():
            if opt.param_groups:
                out[key] = float(opt.param_groups[0]['lr'])
        return out

    def _max_lr(self):
        cap = getattr(self.modeller.args, 'max_lr', None)
        return None if cap is None else float(cap)

    def _apply_lrs(self, st):
        """lr = base x scale, capped at max_lr and floored at min_lr -- EXCEPT
        the flow (Z head) groups, pinned flat at lr_flow, and except groups whose
        base LR was configured as an explicit float, which the bracket does not
        own and which therefore receive their float unmodified.

        THE CAP APPLIES TO EVERY GROUP THIS METHOD WRITES, the flow group
        included: nothing else can lower it -- the scale never reaches it and
        there is no divergence cut any more -- so without the cap there is no
        mechanism by which any controller could.
        """
        m = self.modeller
        a = m.args
        control_flow = bool(self._cfg('control_flow_lr', False))
        managed = self._managed_keys()
        scale = float(st['scale'])
        cap = self._max_lr()
        capped = floored = 0
        for key, opt in m.optimizers.items():
            n_groups = len(opt.param_groups)
            for gi, g in enumerate(opt.param_groups):
                is_flow_group = key == 'flow' or (key == 'fused' and gi == n_groups - 1)
                if is_flow_group and not control_flow:
                    want = a.lr_flow                # pinned; no min_lr floor here
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
                want = base * (scale if base_key in managed else 1.0)
                if cap is not None and want > cap:
                    want, capped = cap, capped + 1
                # A BINDING FLOOR IS AS INVISIBLE AS A BINDING CEILING, and it is
                # the more likely of the two here: a burn-in scale is deliberately
                # small, so a floor set anywhere near the operating range silently
                # turns the conservative rate into a hotter one -- and then the
                # root is not the rate the config named.
                if want < a.min_lr:
                    floored += 1
                g['lr'] = max(a.min_lr, want)
        self._lr_capped_groups = capped
        self._lr_floored_groups = floored

    def step(self):
        """Re-stamp every learning rate from the current scale. Called on the
        10-step reporting clock; it does NOT move the scale."""
        st = self._state()
        self._apply_lrs(st)
        self._emit(st)
        return self.modeller.optimizers['fwd'].param_groups[0]['lr']

    # -------------------------------------------------------------- tripwire

    def observe(self, step_type, loss, grad_norm):
        """One training step's reading, EVERY step. Returns 'diverged' or None.

        Two jobs, and they are the same measurement: collect the root's loss
        scale during burn-in, and judge the live step against the bars derived
        from it. Direct conditions only -- non-finite values, an absolute
        backstop, and an excursion measured against the root's own scale. There
        is no fitted trend, no loss ratio, no cosine and no composite score,
        because a graduated too-hot indicator is what v8 deleted on evidence and
        what this mechanism exists not to reintroduce.
        """
        if loss is not None and step_type in self.CHANNELS:
            v = float(loss)
            if math.isfinite(v):
                self._loss_history[step_type].append(v)
        if grad_norm is not None and math.isfinite(float(grad_norm)):
            self._grad_history.append(float(grad_norm))
        why = self.bars.judge(step_type, None if loss is None else float(loss),
                              None if grad_norm is None else float(grad_norm))
        if why is None:
            return None
        # EVERY FIRE REWINDS AND CUTS (owner decision 2026-08-25, superseding
        # the 2026-08-24 two-tier design). The two-tier split assumed a finite
        # excursion left the state "intact, just too hot" -- qm9c aug25
        # falsified that: cruising a hot promotion poisoned the fwd branch in
        # ~500 steps (vg_lb 126 -> 2700) while two in-place cuts only slowed
        # the accumulation, and the damage was invisible to tb_err_worst, so
        # nothing downstream would ever have flagged it. With bars refit at the
        # promoted rate the fire lands within tens of steps of excursion onset,
        # so the rolling checkpoint (<= ~50 steps old) is still clean: the
        # rewind costs almost nothing and caps the carried damage at zero.
        # The excursion tier keeps its own counter and cooldown; the CUT
        # happens in on_divergence, once, from the RESTORED checkpoint's scale
        # -- a cut applied here would be overwritten by the reload.
        step = int(getattr(self.modeller, 'step_ind', 0))
        if '_excursion_' in why:
            if step < int(self._fire_cooldown_until):
                return None       # one incident, one response -- not a machine gun
            cooldown = int(self._cfg('fire_cooldown_steps', 100))
            self._moderate_fires += 1
            self._fire_cooldown_until = step + cooldown
            print(f'lr_ctrl FIRE: {why} at step {step} (scale {self.scale:.4g}) '
                  f'-- rewinding to the rolling checkpoint; the post-restore cut '
                  f'follows (cooldown {cooldown}). Poisoning accumulates faster '
                  f'than cuts arrest it (qm9c aug25), so no fire keeps the '
                  f'excursion weights.')
            return 'diverged'
        # NOT counted here: every 'diverged' return leads into fire_loss_spike,
        # whose on_divergence() call is the single counting seat -- the
        # non-finite-gradient path reaches fire_loss_spike WITHOUT passing
        # through this method (current_loss is None there), so counting at
        # detection either doubles this channel or misses that path.
        print(f'lr_ctrl DISASTER: {why} at step {step} '
              f'(scale {self.scale:.4g}) -- reload')
        return 'diverged'

    #: Kept under its old name because `bench/` and the smoke harnesses call it;
    #: it is the same judgement as `observe` without the history side effect.
    #: UNLIKE observe it still counts at detection: harness callers have no
    #: rewind path, so no on_divergence() ever follows a hit here.
    def check_spike(self, step_type, current_loss, grad_norm):
        why = self.bars.judge(step_type,
                              None if current_loss is None else float(current_loss),
                              None if grad_norm is None else float(grad_norm))
        if why is None:
            return None
        self._divergences += 1
        return 'diverged'

    def on_divergence(self, count: int = 1):
        """Called after a rewind. THE COUNTING SEAT -- and since the owner's
        two-tier decision (2026-08-24) it also CUTS: a disaster rewind restores
        healthy weights, and re-entering them at the same rate is how the toy
        workout's death loops re-detonated every 2 steps. The cut stands until
        the scheduled re-race re-measures the rate. The rewind budget
        (max_reloads_per_1k_steps) remains the backstop for a loop the cuts
        cannot break.

        This is the ONE place `lr_ctrl/divergences` increments. `observe`
        counting at detection too used to double every event (observed as
        #2/#4/#6 across three rewinds on toy_wk_aug24), and this channel's
        absolute counts are what calibration work reasons from -- the
        hot-sensor sweep leaned on '0 in all 97 segments'. Counting here covers
        both real paths once each: bar-fired (observe -> fire_loss_spike) and
        non-finite gradient (monitor_losses -> fire_loss_spike, which never
        passes through observe because its current_loss is None). Known
        undercount: fire_loss_spike's UNRECOVERABLE branch raises before
        reaching this, so the final fatal event of an aborting run is not in
        the counter -- the abort message is its record.

        Unlike v8 the cut records no permanent ceiling and runs no recovery
        ramp: it is a flat factor, LOUD, and it expires at the next re-race.
        """
        self._divergences += max(int(count), 1)
        factor = float(self._cfg('fire_cut_factor', 0.5))
        new = self.set_scale(self.scale * factor, why='disaster_rewind_cut')
        print(f'lr_ctrl: fire #{self._divergences} -- rewound, and the rate is '
              f'CUT to scale {new:.4g} (x{factor:g}): re-entering restored weights '
              f'at the rate that just detonated them is a loop (toy_wk_aug24). '
              f'The scheduled re-race re-measures; if this repeats the reload '
              f'budget aborts the run.')

    # ---------------------------------------------------------------- the seat

    def tick(self) -> int:
        """One host-loop iteration. Returns the number of PROMOTED steps the
        caller's step clock must skip -- the winner's horizon, never the sum of
        all trial compute.

        This is the whole control law. Read it top to bottom: burn in for a fixed
        number of steps, take a root, run the grid, promote, hold. There is no
        settling gate, no quorum, no retry and no wait on a learned metric --
        every branch below either advances or returns.
        """
        m = self.modeller
        st = self._state()
        step = int(getattr(m, 'step_ind', 0))
        b = self.bracket
        skip = 0

        if (b.phase == CRUISE and not self.bars.loss_bar
                and self._bars_ready(full=True)
                and self._cruise_bar is None):
            # (a pending refit that SUSPENDED the bars leaves loss_bar empty on
            # purpose; `_cruise_bar` is what tells the two states apart)
            # A RESUME LANDS HERE WITH NO BARS AT ALL. `bars.derive` runs once,
            # inside `_open_bracket` -- and a run resumed mid-cruise never calls
            # it, because burn-in and the bracket are both behind it. The derived
            # excursion bar is the run's only tripwire that can fire on this
            # route, so without this the whole remaining stage trains on the
            # absolute backstops alone: exactly the 1e9 situation this design
            # replaced, arriving through the resume path instead of the config.
            #
            # `full=True` FOR THE SAME REASON THE OTHER TWO CALLERS USE IT. This
            # branch took the 20-observation minimum, so a resumed leg armed its
            # tripwire from a sliver: measured on localprod_lp02_resume the bar
            # came out at 2054 against the 7662 the same stage carried when
            # fitted cold, i.e. ~3.7x tighter, and a too-tight bar fires on
            # ordinary training and costs a rewind. A chained multi-day run
            # crosses this seat at every leg boundary, so it is the common case
            # here, not the rare one. The ~200 steps spent on the absolute
            # backstops meanwhile is the cost the burn-in and repeat branches
            # already accept.
            why = self.bars.derive(self._loss_history, self._grad_history)
            print('lr_ctrl: resumed into cruise -- hard-failure bars re-derived '
                  'from the post-resume window: '
                  + (self.bars.scale_note if why is None
                     else 'NOT DERIVED ({})'.format(why)))

        if b.phase == BURN_IN:
            elapsed = step - int(st.get('stage_entry_step', 0))
            # full=True: on a fresh stage burn-in runs thousands of steps, so
            # the window is full by construction and this costs nothing. The
            # case it changes is a RESUME landing with burn-in already elapsed
            # by step count -- there the window holds only what this process
            # has seen, and 20 observations is a sliver no other race would
            # accept (same reasoning as the repeat branch below).
            if b.burn_in_complete(elapsed) and self._bars_ready(full=True):
                skip = self._open_bracket(step)
        elif b.phase == CRUISE and b.repeat_due(step) and self._bars_ready(full=True):
            # A REPEAT TAKES THE CURRENT MATURE STATE AS THE NEW ROOT and runs
            # the same explicit grid again. No second burn-in: the run has been
            # training at a promoted rate, so the optimizers are at steady state
            # by construction and the bias-correction check passes trivially.
            #
            # `_bars_ready` GATES THIS THE SAME WAY IT GATES BURN-IN, and the
            # asymmetry was costing resumed runs their operating point. The loss
            # history is a deque on the controller: it dies with the process, so
            # a resume lands in CRUISE with an ALREADY-DUE clock and an EMPTY
            # window. The cycle opened at step ~10, could only refuse ("no
            # channel accumulated 20 finite loss observations"), and the refusal
            # stamps `promoted_at = step` -- which CONSUMES the cycle and pushes
            # the real race a full repeat_every into the future. qm9c aug25 rode
            # that through three resumes at the burn-in scale 0.05 while its own
            # ladder showed 0.8 surviving: 16x cold for ~28k steps, read as
            # "VarGrad has saturated". Waiting instead costs the ~200 steps the
            # window needs to refill and keeps the clock due.
            print(f'lr_ctrl: re-bracketing -- {step - int(b.promoted_at)} promoted steps '
                  f'since the last selection')
            skip = self._open_bracket(step)

        if b.phase == CRUISE:
            b.note_promoted_steps(1)
            if self._cruise_bar is not None:
                self._tick_cruise_bar(step)
        # RE-READ THE STATE. `lr_ctrl` is in `TrainerSnapshot.FIELDS`, so every
        # trial restore REPLACES `modeller.lr_ctrl` with a fresh dict -- which
        # leaves the `st` captured at the top of this method pointing at an
        # orphan. Writing the bracket's verdict into that orphan means a
        # checkpoint taken on this iteration records a bracket that never
        # happened, and a resume from it re-runs one that already has.
        st = self._state()
        st['bracket'] = b.state_dict()
        st['scale'] = float(st.get('scale', b.scale_now()))
        return int(skip)

    def _bars_ready(self, full: bool = False) -> bool:
        """Has the root window filled enough to derive a bar that can fire?

        A BOUNDED WAIT, not a settling gate: the window fills at exactly one
        observation per training step, so this resolves in at most
        `hard_failure.min_observations` steps (`root_window` steps with
        `full=True`) and cannot be held open by anything the rate is itself
        moving. It exists for the resume case, where burn-in is already
        complete by step count but this process has seen no losses yet -- and
        bracketing with no derived bar is bracketing with bars that cannot
        fire.

        `full` is the REPEAT branch's requirement: a repeat race is not racing
        against a 3000-step burn-in whose window filled long ago, it is racing
        against whatever this process has observed since it resumed -- and a
        span fitted to the minimum 20 observations is a materially noisier bar
        than the 200-entry window every other race derives from (audit
        2026-08-25). Waiting the extra ~180 steps is cheap; a bar drawn from a
        20-step sliver convicts or clears rungs on luck.
        """
        # FIXED MODE WAITS TOO (2026-08-25). The old blanket True existed so
        # fixed mode could not be held hostage by an underivable window -- but
        # on a resume with burn-in already elapsed it opened the stage with an
        # EMPTY window: derive failed, and the asserted rate ran ~450 steps on
        # the absolute backstops alone (qm9c fixed-0.4). Waiting is safe by
        # construction -- until the window fills the run holds the burn-in
        # scale -- and the genuinely-underivable case still falls through to
        # the "continues on the absolute backstops" path after the wait,
        # because the wait is on OBSERVATION COUNT, which fills at one per
        # step regardless of what the values are.
        best = max((len(v) for v in self._loss_history.values()), default=0)
        need = self.bars.root_window if full else self.bars.min_observations
        return best >= need

    def _keep_rate(self, cap=None):
        """The rate a refusal should fall back to.

        On the first cycle nothing has been measured, so it is the burn-in scale
        -- the safe answer, and labelled as one. On a REPEAT the run already has
        a rate a previous bracket measured and that has been training ever since,
        and dropping that to the burn-in scale would make a refused re-bracket
        COST the run its operating point for no evidential reason."""
        promoted = self.bracket.promoted_scale
        kept = self.bracket.burn_in_scale if promoted is None else float(promoted)
        # Capped at the rate in force when the caller's cycle opened: a fire cut
        # stands until a MEASUREMENT replaces it, and every _keep_rate caller is
        # by definition not measuring.
        if cap is not None:
            kept = min(kept, float(cap))
        return kept

    def _train_mode(self):
        stage = getattr(getattr(self.modeller, 'protocol', None), 'stage', None)
        return getattr(stage, 'train_mode', None) or 'fused'

    def _check_bar_window(self):
        """`min_observations` above `root_window` can never be satisfied.

        The window is a deque of maxlen `root_window`, so a larger requirement
        means `_bars_ready` returns False on every step for the life of the run:
        burn-in never ends, no bracket is ever armed, and NOTHING SAYS SO -- the
        run just trains at the burn-in scale forever, which is a plausible enough
        thing to see that nobody would look. That is an unbounded wait, which
        this design does not have anywhere else, so it is refused at construction
        rather than diagnosed after a wasted job."""
        if self.bracket.mode != 'bracket':
            return
        if self.bars.min_observations > self.bars.root_window:
            raise ValueError(
                'lr_control.hard_failure.min_observations ({}) is above '
                'root_window ({}). The observation window is a ring of '
                'root_window entries, so the requirement can never be met: '
                'burn-in would never end, no bracket would ever run, and the run '
                'would train at the burn-in scale for its whole life without '
                'saying so.'.format(self.bars.min_observations,
                                    self.bars.root_window))

    def _arm_cruise_bar(self, promoted_scale, suspend: bool):
        """Queue a refit of the bars at the rate the run is about to hold.

        NOT ARMED when the promoted rate IS the burn-in rate -- a refusal, or a
        grid whose bottom rung was selected. The root bars already describe that
        rate, so a refit would spend 400 steps to arrive at the same answer.

        `suspend` DECIDES WHAT GUARDS THE REFIT WINDOW, and the two callers want
        opposite things because they know different amounts about the rate:

          bracket mode (suspend=True)   the promoted rate survived a full trial
            horizon minutes ago. The cold bar is now the likelier source of a
            wrong answer than the rate is, so it comes down and the backstops
            stand for the window.
          fixed mode (suspend=False)    NOTHING tested this rate; it was asserted
            from outside. A bar that is too tight costs a rewind, an absent one
            costs the run, so the cold bar stays live until the refit replaces
            it.
        """
        if not self._cruise_rederive:
            return
        if promoted_scale is None:
            return
        if float(promoted_scale) == float(self.bracket.burn_in_scale):
            self._cruise_bar = None
            return
        # `prev_*`, not `root_*`: on a repeat_every re-bracket the bars standing
        # here were derived at the PREVIOUS PROMOTED rate, not at burn-in.
        self._cruise_bar = {'prev_loss': dict(self.bars.loss_bar),
                            'prev_grad': self.bars.grad_bar,
                            'suspended': bool(suspend)}
        self._bars_redrawn = False
        held = ('SUSPENDED (non-finite and the absolute backstops stand meanwhile)'
                if suspend else
                'HELD LIVE -- nothing tested this rate, so a bar that is too tight '
                'beats no bar at all')
        if suspend:
            self.bars.loss_bar = {}
            self.bars.grad_bar = None
        print(f'lr_ctrl: rate promoted to scale {float(promoted_scale):g} -- the '
              f'excursion bars fitted at the burn-in rate are {held}. They are '
              f'refitted from {self.bars.root_window} steps of ordinary training at '
              f'the new rate, after a {self._cruise_settle_steps}-step settle.')

    def _tick_cruise_bar(self, step: int):
        """Advance the post-promotion redraw: settle, clear, collect, refit.

        THE CLOCK IS THIS METHOD'S OWN CALL COUNT, not `step_ind`. `_open_bracket`
        returns the winning trial's horizon and the host loop SKIPS that many
        steps off its iterator, so `step_ind` jumps forward by up to a full trial
        the instant a rate is promoted. A clock written as `step + settle` would
        be in the past before the first cruise step ran, and the settle it exists
        to enforce would be skipped entirely. This method is called once per real
        cruise step, which is exactly the quantity being counted.

        THE SETTLE IS NOT A CONVERGENCE GATE. It is a fixed number of steps
        discarded because the rate has just changed by up to the width of the
        grid, and the steps immediately after that change are the transient, not
        the behaviour the bar is meant to describe. Nothing here waits on a
        learned metric.
        """
        cb = self._cruise_bar
        n = cb['ticks'] = cb.get('ticks', 0) + 1
        settle = self._cruise_settle_steps
        if n < settle:
            return
        if n == settle:
            # Drop the burn-in observations: they were taken at the cold rate,
            # and mixing them with post-promotion ones fits the bar to neither.
            for q in self._loss_history.values():
                q.clear()
            self._grad_history.clear()
            return
        filled = max((len(v) for v in self._loss_history.values()), default=0)
        # THE DEADLINE IS NOT OPTIONAL. A stage whose steps do not all land on
        # one channel may never fill a window, and without a deadline the refit
        # would stay pending for the rest of the stage -- leaving the tripwire
        # suspended, silently, which is strictly worse than the cold bar this
        # exists to replace.
        deadline = settle + 4 * self.bars.root_window
        if filled < self.bars.root_window and n < deadline:
            return
        self._redraw_bars_for_cruise(step, filled)
        self._cruise_bar = None

    def _redraw_bars_for_cruise(self, step: int, filled: int):
        """Refit the bars to the promoted rate, falling back to the stashed
        previous bars for anything the new window cannot replace.

        `HardFailureBars.derive` rebuilds its table from scratch, so a channel
        that fired before and not since would come back with no bar at all.
        Keeping the previous bar for such a channel is the conservative
        direction -- it was fitted at a colder rate, so it is tighter than a
        correct one, and a tight bar costs a rewind where an absent one costs
        the run.
        """
        cb = self._cruise_bar or {}
        old_loss = dict(cb.get('prev_loss') or {})
        old_grad = cb.get('prev_grad')
        why = self.bars.derive(self._loss_history, self._grad_history)
        if why is not None:
            self.bars.loss_bar, self.bars.grad_bar = old_loss, old_grad
            print(f'lr_ctrl: cruise bars NOT redrawn at step {step} ({why}) -- '
                  f'restoring the bars fitted at the previous rate, which are '
                  f'TIGHTER than this one warrants but are not absent.')
            return
        carried = [c for c in old_loss if c not in self.bars.loss_bar]
        for c in carried:
            self.bars.loss_bar[c] = old_loss[c]
        if self.bars.grad_bar is None and old_grad is not None:
            self.bars.grad_bar = old_grad
            carried.append('grad')
        self._bars_redrawn = True
        note = (f' Kept the previous bar for {carried} -- no post-promotion window.'
                if carried else '')
        short = (f' Window short ({filled}/{self.bars.root_window}) at the deadline.'
                 if filled < self.bars.root_window else '')
        print(f'lr_ctrl: hard-failure bars REDRAWN at step {step} from ordinary '
              f'training at the promoted scale {self.scale:.4g} -- '
              f'{self.bars.scale_note}.{note}{short}')

    def _cancel_pending_refit(self):
        """Put back whatever a pending post-promotion refit took down.

        `repeat_every` can re-open a bracket while a refit is still pending -- at
        the shipped 20000 against a ~400-step refit it cannot in practice, but
        the two are independent clocks and nothing enforces the ordering. Without
        this the stash is dropped: a refit that had SUSPENDED the bars leaves
        them empty, and if the re-opened bracket then refuses (its own window was
        just cleared) the stage runs on the absolute backstops alone for the rest
        of its life. Restoring first is free -- a successful derive overwrites
        them a moment later anyway.
        """
        cb = self._cruise_bar
        if cb is None:
            return
        if cb.get('suspended'):
            self.bars.loss_bar = dict(cb.get('prev_loss') or {})
            self.bars.grad_bar = cb.get('prev_grad')
        self._cruise_bar = None

    def _open_bracket(self, step: int) -> int:
        """Burn-in is over. Take the root and either bracket from it or say why
        not. Returns promoted steps to skip."""
        self._cancel_pending_refit()
        b = self.bracket
        # THE RATE IN FORCE WHEN THE RACE OPENED, and the ceiling on every
        # fallback below. `promoted_scale` alone is NOT that rate: after a fire
        # cut the run cruises at promoted x fire_cut_factor^k, and a refusal or
        # abort that fell back to the bare promotion would silently REVERT the
        # cut -- re-entering the rate whose bars just fired, through the seat
        # whose whole job is to be the safe answer (audit 2026-08-25).
        entry_scale = float(self.scale)
        if b.mode == 'fixed':
            # THE DERIVED BARS ARE STILL WORTH HAVING HERE, even though nothing
            # is being bracketed. Fixed mode chooses the rate from outside; it
            # does not make the rate safe, and without this the run's only
            # hard-failure guard for the whole stage is the absolute backstop --
            # which is what caught nothing on this route when the loss went from
            # about -25 to +318. A refusal is NOT fatal in this mode: there is no
            # measurement to corrupt, so it says so and carries on.
            why = self.bars.derive(self._loss_history, self._grad_history)
            if why is None:
                print(f'lr_ctrl: hard-failure bars derived from burn-in -- '
                      f'{self.bars.scale_note}')
            else:
                print(f'lr_ctrl: NO derived hard-failure bar ({why}) -- fixed mode '
                      f'continues on the absolute backstops alone.')
            self.set_scale(b.fixed_scale, why='fixed_mode')
            b.promote(b.fixed_scale, step)
            self._arm_cruise_bar(b.fixed_scale, suspend=False)
            print(f'lr_ctrl: burn-in complete at step {step}; FIXED mode -- holding '
                  f'scale {b.fixed_scale:g} for the stage. No trials run.')
            return 0

        if not self.stepping_optimizer_keys():
            # THE DOCUMENTED CONTROL ARM: every lr_* key is an explicit float, so
            # the scale reaches no optimizer. There is no rate under test, so
            # spending N x trial_steps discarding training to measure one would
            # be pure waste -- and the "boundary" it found would describe a
            # multiplier nothing applies.
            self.set_scale(b.refuse('no managed learning rate this stage steps -- '
                                    'the scale reaches no optimizer that takes a '
                                    'step here (the control arm)',
                                    scale=self._keep_rate(entry_scale), step=step),
                           why='control_arm')
            print('lr_ctrl: no learning rate this stage steps is bracket-managed, so '
                  'no trials are run. The controller reads and logs while actuating '
                  'nothing.')
            return 0
        # DON'T CALIBRATE A FINISHED STAGE (owner decision 2026-08-24). If the
        # stage's exit trigger is already armed, a transition executes at the
        # next eval and rebuilds the optimizers anyway -- trials would spend
        # N x trial_steps measuring a rate with no stage left to run on, and
        # the promoted rate's only product would be exit-window jitter (seen on
        # the toy: the ladder ran after MLE had fully converged). Same family
        # as the control-arm refusal: declining work, not steering a rate.
        if bool((getattr(self.modeller, 'stage_ctrl', None) or {}).get('exit_armed')):
            self.set_scale(b.refuse('the stage exit trigger is armed -- a transition '
                                    'is imminent, so there is no stage left to run a '
                                    'selected rate on',
                                    scale=self._keep_rate(entry_scale), step=step),
                           why='stage_finished')
            print('lr_ctrl: NOT bracketing -- the stage exit trigger is already armed; '
                  'holding the current rate until the transition re-brackets.')
            try:
                import wandb
                if wandb.run is not None:
                    wandb.run.summary[f'lr_bracket/refusal_step_{int(step)}'] = \
                        'exit trigger armed; stage finished'
            except Exception:
                pass
            return 0
        driver = BracketDriver(self.modeller, b, self.bars,
                               verbose=bool(self._cfg('verbose', True)))
        refusal = self.bars.derive(self._loss_history, self._grad_history)
        if refusal is None:
            # INSIDE THE GUARD. Taking the root is the single largest allocation
            # the mechanism makes -- it copies the model, every optimizer's
            # moments and all three buffers off the card in one go -- so it is
            # the most likely place for the bracket to OOM, and it sat outside
            # the try/except that the abort path documents as covering it.
            try:
                refusal = driver.take_root(step)
            except Exception as e:                   # noqa: BLE001 -- classified
                if is_cuda_oom(e):
                    self.modeller.handle_train_epoch_error(e, self._train_mode())
                refusal = ('the root snapshot could not be taken ({}: {})'
                           .format(type(e).__name__, e))
        if refusal is not None:
            driver.release()
            # THE PRINT REPORTS THE RATE ACTUALLY KEPT. `_keep_rate` falls back
            # to the burn-in scale only on the first cycle; on a repeat it holds
            # the previously promoted rate, and a message hardcoding "burn-in
            # scale" reported a 16x rate drop that never happened (qm9c aug25).
            kept = self._keep_rate(entry_scale)
            self.set_scale(b.refuse(refusal, scale=kept, step=step),
                           why='bracket_refused')
            label = ('the burn-in scale' if kept == b.burn_in_scale
                     else 'the previously promoted scale')
            print(f'lr_ctrl: REFUSING TO BRACKET -- {refusal}\n'
                  f'          Holding {label} {kept:g} until the next re-race. '
                  f'This is the safe answer, not a measured one.')
            # The refusal is race telemetry too -- keyed by STEP, not by cycle:
            # begin_bracket never ran, so a later successful repeat takes the
            # next cycle index and must not overwrite this record. Fire-and-
            # forget, same contract as _publish_race.
            try:
                import wandb
                if wandb.run is not None:
                    wandb.run.summary[f'lr_bracket/refusal_step_{int(step)}'] = str(refusal)
            except Exception:
                pass
            return 0
        print(f'lr_ctrl: hard-failure bars derived from the root -- {self.bars.scale_note}')
        self.driver = driver
        b.begin_bracket(step, driver.root_bias_correction()[0])
        # CAPTURED, NOT READ LATER. `_emit` runs on the 10-step reporting clock,
        # by which time the whole bracket has completed inside this one tick and
        # `self.driver` is None again -- so `driver.report()` was dead code and
        # `lr_bracket/held_mb` could never be published. Take it here.
        self._bracket_report = {}
        try:
            skip = driver.run(step)
            self._bracket_report = driver.report()
            # A PROMOTION IS DURABLE THE MOMENT IT EXISTS (owner concern,
            # 2026-08-25). The race concludes mid-tick and the next scheduled
            # save can be minutes out; a kill inside that window silently
            # discarded the promotion twice in one day -- and since every fire
            # now rewinds to the rolling checkpoint, an unpersisted promotion
            # would also make the first post-promotion fire restore PRE-race
            # weights under a POST-race rate. Model + optimizers only (~74 MB
            # on every route -- MLIP cost lives in energy calls, not
            # checkpoint bytes); the buffer sidecar keeps its eval-cadence
            # contract, whose staleness a resume already tolerates.
            try:
                self.modeller.checkpointer.save('running')
            except Exception as save_error:      # noqa: BLE001 -- best-effort
                print(f'lr_ctrl: post-promotion checkpoint not written '
                      f'({type(save_error).__name__}: {save_error})')
        except Exception as e:                       # noqa: BLE001 -- classified
            # AN ABORTED BRACKET MUST NOT ABORT THE RUN, and this seat is outside
            # the host loop's `try/except (RuntimeError, ValueError)` -- that one
            # wraps `train_step` alone, so anything raised in here would leave
            # the training loop entirely.
            #
            # AN OOM IS THE CASE THIS EXISTS FOR. It says nothing about the rate,
            # so it may not convict a candidate; but it is also not fatal --
            # trials run at the live batch size, and the bracket makes an OOM
            # MORE likely by holding several full trainer snapshots in host
            # memory while the card is already loaded. Hand it to the shared
            # recovery path (which cuts the batch and records the ceiling), put
            # the trainer back on the root, and hold the burn-in scale: the safe
            # answer, labelled as one rather than dressed up as a measurement.
            reason = f'{type(e).__name__}: {e}'
            try:
                if driver.root is not None:
                    driver.root.restore()
            except Exception as restore_error:       # noqa: BLE001
                reason += f' (and the root would not restore: {restore_error})'
            if is_cuda_oom(e):
                # THE STAGE'S OWN TRAIN MODE, not a label.
                # `handle_train_epoch_error` gates the batch cut and the OOM
                # ceiling on `step_type in TRAIN_MODES` -- eval OOMs have a
                # different memory profile and must not install a train-batch
                # ceiling. A trial step IS a train step of this stage's mode, so
                # passing a descriptive string like 'lr_bracket' silently skipped
                # both halves of the recovery and left the batch untouched for
                # the next attempt.
                #
                # AFTER THE RESTORE, NEVER BEFORE. batch_size, the OOM ceiling,
                # the cooldown and the sizer are all in TrainerSnapshot.FIELDS,
                # so a restore run after the handler wound every one of them
                # back to the root's values: the run re-entered cruise at the
                # exact batch that had just OOMed twice, with no ceiling and a
                # re-armed sizer (audit 2026-08-25). Restore first, then let
                # the recovery's cut and ceiling stand on the restored state.
                self.modeller.handle_train_epoch_error(e, self._train_mode())
                if getattr(driver, 'race_ooms', 0):
                    # the success path publishes this through driver.report();
                    # the abort path is precisely the one where it says the most
                    self._bracket_report = {'lr_bracket/race_ooms': driver.race_ooms}
            kept = self._keep_rate(entry_scale)
            self.set_scale(b.refuse(f'bracket aborted -- {reason}',
                                    scale=kept, step=step),
                           why='bracket_aborted')
            label = ('the burn-in scale' if kept == b.burn_in_scale
                     else 'the previously promoted scale')
            print(f'lr_ctrl: BRACKET ABORTED -- {reason}\n'
                  f'          Restored the root and holding {label} {kept:g}. '
                  f'No boundary was measured.')
            skip = 0
        finally:
            self.driver = None
            driver.release()
        # ARMED AFTER THE CYCLE, from whatever the bracket actually settled on --
        # a promotion, a refusal, or an abort all land in `promoted_scale`, and
        # only the first of the three is a rate the root bars do not describe.
        self._arm_cruise_bar(b.promoted_scale, suspend=True)
        return skip

    def on_stage_change(self):
        """Protocol.advance hook. The optimizers were rebuilt onto a surface with
        different curvature, so the promoted rate describes a surface that no
        longer exists: the stage re-enters burn-in and re-brackets.

        A GENUINE STAGE TRANSITION RUNS ANOTHER FIXED BURN-IN, and it has to --
        rebuilding the optimizers restarts Adam's step counter, and bracketing
        from a counter at t=10 measures 0.153 of the rate under test.

        Returns the burn-in length in train steps, for the caller's log.
        """
        m = self.modeller
        st = self._state()
        st['stage_entry_step'] = int(m.step_ind)
        b = self.bracket
        b.phase = BURN_IN
        b.promoted_scale = None
        b.promoted_at = None
        b.refusal = None
        for q in self._loss_history.values():
            q.clear()
        self._grad_history.clear()
        # THE BARS GO WITH THE WINDOW. Clearing the observations but keeping the
        # bars fitted to them left the outgoing stage's bars deciding rewinds
        # through the incoming stage's burn-in -- a different train_mode, a
        # different composite, and a loss scale that can differ by orders of
        # magnitude. Until the new root is taken the absolute backstops stand
        # alone, which is what a stage with no measured scale yet has earned.
        self.bars.loss_bar = {}
        self.bars.grad_bar = None
        self.bars.scale_note = ''
        # NOT `_cancel_pending_refit`: a stage change is exactly the case where
        # the outgoing bars must NOT be put back.
        self._cruise_bar = None
        self._bars_redrawn = False
        self.set_scale(b.burn_in_scale, why='stage_change')
        print(f'lr_ctrl: stage change -- burn-in restarted, {b.burn_in_steps} steps at '
              f'scale {b.burn_in_scale:g}. The optimizers were rebuilt, so Adam\'s step '
              f'counter is back at zero and nothing may be bracketed until it is not.')
        return b.burn_in_steps

    #: The name protocol.py and the smoke harnesses still call. Same transaction.
    rearm_warmup = on_stage_change

    def in_burn_in(self) -> bool:
        return self.bracket.phase == BURN_IN

    #: Retained spelling for the diagnostic sensors, which decline to sample
    #: while the rate is deliberately conservative.
    in_warmup = in_burn_in

    # ------------------------------------------------- diagnostics (off by default)

    def calibration_refusal(self):
        """Why a ray reading would be thrown away, decided without measuring --
        or None. The ray is a DIAGNOSTIC here: it reaches no learning rate, so
        the only refusal left is 'nothing asked for it'."""
        return None

    def on_calibration(self, reading):
        """Record one ray calibration. IT MOVES NOTHING.

        The sensor is retained as an optional, off-by-default diagnostic so a
        future claim about it can be measured rather than argued about. It is
        wired to no actuator: alpha* does not respond to the learning rate (the
        slope of log alpha* against log lr measured 0.00 +- 0.2 where -1 is
        required), so any rate it set would be a number with no relationship to
        the rate.
        """
        self._calibrations += 1
        self._last_ray = dict(reading or {})

    def on_hypergradient(self, cos, beta=None, beta_down=None, cos_target=0.0,
                         clip_ratio=None):
        """Record one hypergradient cosine. IT MOVES NOTHING -- same contract as
        `on_calibration`. `cos` is a stationarity statistic: it is negative at
        every stable rate once the iterate has equilibrated, so it has no fixed
        point to steer to."""
        if cos is None or not math.isfinite(float(cos)):
            return
        self._hypergrads += 1
        self._last_cos = float(cos)

    # ---------------------------------------------------------------- report

    def _emit(self, st):
        """The LR channel. Pared to what a reader needs to reconstruct the rate
        and the experiment that chose it -- and deliberately NOT carrying alpha*
        or cos, which would read as explanations for a selection neither entered.
        """
        self._report = {
            'lr_ctrl/scale': float(st['scale']),
            'lr_ctrl/divergences': float(self._divergences),
            'lr_ctrl/moderate_fires': float(self._moderate_fires),
        }
        self._report.update(self.bracket.report())
        self._report.update(self.bars.report())
        self._report['lr_bracket/bars_redrawn'] = float(bool(self._bars_redrawn))
        self._report.update(getattr(self, '_bracket_report', None) or {})
        # Published only when a cap is configured, so its ABSENCE means "no rail"
        # rather than "rail never bound" -- two states a constant 0.0 could not
        # tell apart.
        if self._max_lr() is not None:
            self._report['lr_ctrl/lr_capped_groups'] = float(self._lr_capped_groups)
        self._report['lr_ctrl/lr_floored_groups'] = float(self._lr_floored_groups)
        # THE DIAGNOSTIC SENSORS ARE SILENT UNLESS THEY RAN. A channel that
        # publishes a constant whether or not the sensor fired is how a dead
        # sensor reads exactly like a working one.
        if self._calibrations:
            self._report['lr_ctrl/calibrations'] = float(self._calibrations)
        if self._hypergrads:
            self._report['lr_ctrl/hypergrads'] = float(self._hypergrads)
            if getattr(self, '_last_cos', None) is not None:
                self._report['lr_ctrl/hyper_cos'] = float(self._last_cos)

    def report(self):
        return dict(self._report)


def _get(node, name, default=None):
    return default if node is None else getattr(node, name, default)
