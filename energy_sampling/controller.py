"""
The learning-rate controller: ONE multiplier over the managed policy optimizers,
plus the always-on hard tripwire.

    lr = base_lr x scale

    scale = schedule(steps since stage entry) x cut

THE SCHEDULE (`lr_control.mode: fixed`, the only mode) is `lr_control.schedule`,
or the stage's own `lr_schedule` where it declares one: an ordered list of
segments {to, steps, kind} (config_invariants.lr_schedule_problems). Segment 0
starts at its own `to`; each later segment moves from the previous `to` to its
own over `steps` (null = to the stage's max_steps) by its kind; past the last
segment the value holds (`lr_schedule_value`). The clock is anchored at the
stage's entry step, stamped in `stage_ctrl['lr_sched_entry']`, so a resume or a
rewind continues mid-schedule; the schedule itself is read from the config.

THE CUT. A divergence fire rewinds and multiplies `cut` (in `lr_ctrl`) by
`fire_cut_factor`; it carries through the rest of the stage. A stage transition
resets it to 1 and restarts the schedule. Nothing else moves the scale.

THE TRIPWIRE. `hard_failure_bars.HardFailureBars` fits per-branch loss and
gradient bars from the last `hard_failure.root_window` observations of the
schedule's first segment (the warm-up), and refits them `cruise_settle_steps`
after the last segment whose `to` differs from its predecessor's ends
(`cruise_rederive`).

`lr_control.mode: bracket` (the brute-force candidate-grid search) was deleted
2026-10-28 and is refused at construction. `ray` survives as an optional,
off-by-default diagnostic that reaches no learning rate.
"""

from __future__ import annotations

import math
from collections import deque

from energy_sampling.config_invariants import (LR_CONTROL_RETIRED_KEYS,
                                               LR_SCHEDULE_REPLACED_KEYS,
                                               legacy_lr_schedule,
                                               lr_schedule_problems,
                                               normalize_lr_schedule,
                                               replaced_lr_schedule_keys,
                                               retired_lr_control_keys)
from energy_sampling.hard_failure_bars import HardFailureBars

# ------------------------------------------------------------------- phases

BURN_IN = 'burn_in'      # inside the schedule's first segment (the warm-up)
CRUISE = 'cruise'        # past the first segment


# ---------------------------------------------------------------- schedule

def _interp(kind, a, b, p):
    """The value of a segment from `a` to `b` at progress p in [0, 1]."""
    if kind == 'hold':
        return b
    if kind == 'linear':
        return a + (b - a) * p
    if kind == 'cosine':
        return b + (a - b) * 0.5 * (1.0 + math.cos(math.pi * p))
    return a * (b / a) ** p          # geometric, exp


def lr_schedule_ends(segments, max_steps=None) -> list:
    """Each segment's end, in steps since stage entry. A `steps: null` segment
    ends at `max_steps` (never before its start), or never (inf) with no
    max_steps."""
    ends, t = [], 0.0
    for s in segments:
        n = s['steps']
        if n is None:
            t = math.inf if max_steps is None else max(t, float(max_steps))
        else:
            t += n
        ends.append(t)
    return ends


def lr_schedule_value(segments, elapsed, max_steps=None):
    """(multiplier, segment index) at `elapsed` steps since stage entry, for a
    normalized schedule (config_invariants.normalize_lr_schedule). Segment 0
    starts at its own `to`, each later segment at the previous `to`; a 0-step
    segment is a jump; past the last segment its `to` holds."""
    elapsed = max(0.0, float(elapsed))
    start, a = 0.0, segments[0]['to']
    for i, (s, end) in enumerate(zip(segments, lr_schedule_ends(segments, max_steps))):
        b = s['to']
        if elapsed < end:
            p = 0.0 if math.isinf(end) else (elapsed - start) / (end - start)
            return _interp(s['kind'], a, b, p), i
        start, a = end, b
    return segments[-1]['to'], len(segments) - 1


class LRController:
    """Owns every learning rate the run writes."""

    CHANNELS = ('fwd', 'bwd', 'replay', 'fused')

    #: Policy optimizer keys -> the args attribute holding their base LR. The
    #: flow head is deliberately absent: it is pinned, not scheduled.
    _POLICY_BASE = {'fwd': 'lr_policy', 'bwd': 'lr_back', 'replay': 'lr_replay',
                    'fused': 'lr_fused'}

    #: The checkpointed `lr_ctrl` dict layout, {'ver', 'scale', 'cut'}. A dict
    #: with another `ver` is discarded and replaced with a fresh one.
    _STATE_VER = 11

    def __init__(self, modeller):
        self.modeller = modeller
        self._report = {}
        self._divergences = 0            # every rewind+cut response (the counting seat)
        self._moderate_fires = 0         # the excursion-tier subset of the above
        self._fire_cooldown_until = 0
        self._calibrations = 0
        self._lr_capped_groups = 0
        self._lr_floored_groups = 0
        self._last_scale_reason = 'init'

        self._check_schema()
        cfg = self._cfg_node()
        hf = getattr(cfg, 'hard_failure', None)
        self.bars = HardFailureBars(
            loss_excursion_k=float(_get(hf, 'loss_excursion_k', 10.0)),
            grad_excursion_x=float(_get(hf, 'grad_excursion_x', 100.0)),
            loss_abs=_get(hf, 'loss_abs', 1.0e6),
            grad_abs=_get(hf, 'grad_abs', 1.0e6),
            root_window=int(_get(hf, 'root_window', 200)),
            min_observations=int(_get(hf, 'min_observations', 20)))
        self._cruise_rederive = bool(_get(hf, 'cruise_rederive', True))
        self._cruise_settle_steps = int(_get(hf, 'cruise_settle_steps', 200))
        if self._cruise_settle_steps < 0:
            raise ValueError(
                f'lr_control.hard_failure.cruise_settle_steps must be >= 0, got '
                f'{self._cruise_settle_steps}')

        self.mode = str(_get(cfg, 'mode', 'fixed'))
        if self.mode == 'bracket':
            raise ValueError(
                "lr_control.mode 'bracket' was deleted 2026-10-28 (the candidate "
                "grid, trials and boundary search are gone). Write `mode: fixed` "
                "with an explicit `schedule` of {to, steps, kind} segments.")
        if self.mode != 'fixed':
            raise ValueError(f"lr_control.mode must be 'fixed', got {self.mode!r}")
        self._run_schedule = self._resolve_run_schedule(cfg)

        # ------------------------------------------------------------- phase
        self.phase = BURN_IN
        self._promoted_steps = 0
        #: (stage name, segments, ends, refit end, max_steps), rebuilt when the
        #: stage name changes.
        self._sched = None
        #: Steps since stage entry at this process's previous tick (-1 right
        #: after a stage change; None before the first tick).
        self._last_elapsed = None
        #: schedule x cut as last applied by tick (None = apply on the next tick).
        self._last_sched = None
        self._segment = 0

        # THE OBSERVATION WINDOWS the bars are fitted from. On the instance, not
        # in `lr_ctrl`: a resume refills them before any bar is fitted, at one
        # observation per step.
        self._loss_history = {c: deque(maxlen=self.bars.root_window)
                              for c in self.CHANNELS}
        self._grad_history = deque(maxlen=self.bars.root_window)
        #: The pending refit after the last rate change, or None. It counts its
        #: own ticks.
        self._cruise_bar = None
        #: Whether the live bars were refitted at the held rate (published).
        self._bars_redrawn = False

        self._check_rails()
        # Builds `lr_ctrl` at construction. A checkpoint restore overwrites it
        # wholesale afterwards.
        self._state()

    # ------------------------------------------------------------------ config

    def _cfg_node(self):
        return getattr(self.modeller.args, 'lr_control', None)

    def _cfg(self, name, default=None):
        return _get(self._cfg_node(), name, default)

    @staticmethod
    def _resolve_run_schedule(cfg):
        """`lr_control.schedule`, normalized. On the conformer route a block with
        no `schedule` is read through config_invariants.legacy_lr_schedule from
        the keys it replaced (off that route _check_schema has refused them).
        With no block the run takes legacy_lr_schedule's defaults; nothing is
        managed without a block, so that scale reaches no rate."""
        if cfg is None:
            return legacy_lr_schedule()
        present = set(vars(cfg)) if hasattr(cfg, '__dict__') else set()
        legacy = [k for k in LR_SCHEDULE_REPLACED_KEYS if k in present]
        node = _get(cfg, 'schedule')
        if node is not None:
            if legacy:
                raise ValueError(f'lr_control carries both `schedule` and {legacy}, '
                                 f'which `schedule` replaced. Remove {legacy}.')
            problems = lr_schedule_problems(node, 'lr_control.schedule')
            if problems:
                raise ValueError('; '.join(problems))
            return normalize_lr_schedule(node)
        if not legacy:
            raise ValueError(
                "lr_control needs an explicit lr_control.schedule: a list of "
                "{to, steps, kind} segments, the multiplier on the auto rates over "
                "each stage.")
        fixed = _get(cfg, 'fixed_scale', None)
        if fixed is None or not float(fixed) > 0:
            raise ValueError("lr_control.mode 'fixed' needs a positive "
                             "lr_control.fixed_scale -- the rate is stated in the config.")
        burn_steps = int(_get(cfg, 'burn_in_steps', 3000))
        if burn_steps < 0:
            raise ValueError(f'lr_control.burn_in_steps must be >= 0, got {burn_steps}')
        burn = float(_get(cfg, 'burn_in_scale', 0.05))
        if not (burn > 0) or not math.isfinite(burn):
            raise ValueError(f'lr_control.burn_in_scale must be a positive finite '
                             f'multiplier, got {burn!r}. Burn-in must still train.')
        ramp = int(_get(cfg, 'promotion_ramp_steps', 1000) or 0)
        if ramp < 0:
            raise ValueError(f'lr_control.promotion_ramp_steps must be >= 0, got {ramp}')
        return legacy_lr_schedule(burn_steps, burn, ramp, float(fixed))

    def _schedule(self):
        """(segments, ends, refit end, max_steps) for the current stage: its
        `lr_schedule`, else the run-level schedule. The refit end is the end of
        the last segment whose `to` differs from the previous segment's, or None.
        A non-hold `steps: null` segment on a stage with no max_steps raises."""
        stage = getattr(getattr(self.modeller, 'protocol', None), 'stage', None)
        name = getattr(stage, 'name', None)
        if self._sched is not None and self._sched[0] == name:
            return self._sched[1:]
        segs = getattr(stage, 'lr_schedule', None) or self._run_schedule
        max_steps = getattr(stage, 'max_steps', None)
        if stage is not None and max_steps is None:
            problems = lr_schedule_problems(segs, 'lr_control.schedule',
                                            judge_open_ended=True)
            if problems:
                raise ValueError(f"stage '{name}': " + '; '.join(problems))
        ends = lr_schedule_ends(segs, max_steps)
        refit = None
        for i in range(1, len(segs)):
            if segs[i]['to'] != segs[i - 1]['to']:
                refit = ends[i]
        self._sched = (name, segs, ends, refit, max_steps)
        return self._sched[1:]

    def entry_scale(self) -> float:
        """The current stage's schedule value at its entry step (segment 0's `to`)."""
        return float(self._schedule()[0][0]['to'])

    def _sched_store(self) -> dict:
        """Where the schedule's entry stamp lives: `stage_ctrl` (reset at every
        transition, checkpointed with it), else the `lr_ctrl` dict."""
        sc = getattr(self.modeller, 'stage_ctrl', None)
        return sc if isinstance(sc, dict) else self._state()

    def _elapsed(self, step: int) -> int:
        """Steps since the stage's schedule entry. A store with no stamp is
        stamped at `step`."""
        store = self._sched_store()
        entry = store.get('lr_sched_entry')
        if entry is None:
            entry = store['lr_sched_entry'] = int(step)
            if step:
                print(f'lr_ctrl: no LR schedule entry stamp in the stage state -- '
                      f'the schedule starts at step {step}.')
        return max(0, int(step) - int(entry))

    @staticmethod
    def _describe(segs) -> str:
        return ', '.join(
            f"{s['kind']} {s['to']:g}"
            + (' to the stage cap' if s['steps'] is None else f" over {s['steps']}")
            for s in segs)

    def _managed_keys(self):
        """Config keys the scale reaches -- those written `auto`, recorded by
        resolve_derived_config at load. Empty set = the controller reads and
        logs but actuates nothing (the control arm)."""
        return set(getattr(self.modeller.args, 'lr_servo_managed', ()) or ())

    #: Everything `lr_control` may contain. CLOSED: an unknown key raises.
    _KEYS = frozenset({
        'mode', 'seed_lr', 'control_flow_lr', 'schedule',
        'verbose', 'hard_failure', 'ray_calibration',
        'fire_cut_factor', 'fire_cooldown_steps',
    })
    _HARD_FAILURE_KEYS = frozenset({
        'loss_excursion_k', 'grad_excursion_x', 'loss_abs', 'grad_abs',
        'root_window', 'min_observations',
        'cruise_rederive', 'cruise_settle_steps',
    })

    def _check_schema(self):
        """Refuse an unknown key under `lr_control`, the retired bracket keys
        (config_invariants.LR_CONTROL_RETIRED_KEYS) and the keys `schedule`
        replaced (LR_SCHEDULE_REPLACED_KEYS) off the conformer route. On the
        conformer route the bracket keys are ignored and the replaced keys are
        read when there is no `schedule` (_resolve_run_schedule)."""
        retired = retired_lr_control_keys(self.modeller.args)
        if retired:
            raise ValueError(
                f'lr_control: {retired} were deleted 2026-10-28 with '
                f"lr_control.mode 'bracket' and are read by nothing. Remove them.")
        replaced = replaced_lr_schedule_keys(self.modeller.args)
        if replaced:
            raise ValueError(
                f'lr_control: {replaced} were replaced 2026-10-28 by '
                f'lr_control.schedule. The old burn-in and ramp are `schedule: '
                f'[{{to: <burn_in_scale>, steps: <burn_in_steps>, kind: hold}}, '
                f'{{to: <fixed_scale>, steps: <promotion_ramp_steps>, kind: geometric}}]`.')
        for node, known, where in ((self._cfg_node(),
                                    self._KEYS | set(LR_CONTROL_RETIRED_KEYS)
                                    | set(LR_SCHEDULE_REPLACED_KEYS),
                                    'lr_control'),
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
                    f'ignored in silence.')

    def _check_rails(self):
        """max_lr vs min_lr CANNOT be resolved by clamp order -- whichever is
        applied second wins, so one bound is silently defeated."""
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
        managed = ','.join(sorted(self._managed_keys())) or 'NOTHING (control arm)'
        segs = self._schedule()[0]
        print(f'lr_ctrl: FIXED mode -- schedule {self._describe(segs)}, from each '
              f'stage entry (a stage lr_schedule replaces it). Managed: {managed}')

    # -------------------------------------------------------------- actuation

    def _state(self):
        m = self.modeller
        st = getattr(m, 'lr_ctrl', None)
        if not isinstance(st, dict) or st.get('ver') != self._STATE_VER:
            if isinstance(st, dict) and st.get('ver') is not None:
                print(f"lr_ctrl: discarding stale state ver={st.get('ver')} "
                      f'(this controller is ver={self._STATE_VER}).')
            st = {'ver': self._STATE_VER, 'scale': self.entry_scale(), 'cut': 1.0}
            m.lr_ctrl = st
        if st.get('scale') is None:
            st['scale'] = self.entry_scale()
        st.setdefault('cut', 1.0)
        return st

    def set_scale(self, scale, why=None):
        """Move the one multiplier. Callers: the schedule (tick), a stage
        transition, and the post-rewind cut."""
        st = self._state()
        st['scale'] = float(scale)
        self._last_scale_reason = str(why or 'set')
        self._apply_lrs(st)
        return st['scale']

    @property
    def scale(self):
        return float(self._state()['scale'])

    def live_rates(self) -> dict:
        """What each optimizer's first group is ACTUALLY set to."""
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
        base LR was configured as an explicit float, which receive their float
        unmodified.

        The cap applies to every group this method writes, the flow group
        included; the floor does not apply to the pinned flow groups.
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

        Appends the reading to the observation windows and judges it against
        the bars. An excursion verdict inside the fire cooldown returns None.
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
        # EVERY FIRE REWINDS (owner decision 2026-08-25). The CUT happens in
        # on_divergence, once, from the RESTORED checkpoint's scale -- a cut
        # applied here would be overwritten by the reload.
        step = int(getattr(self.modeller, 'step_ind', 0))
        if '_excursion_' in why:
            if step < int(self._fire_cooldown_until):
                return None       # one incident, one response
            cooldown = int(self._cfg('fire_cooldown_steps', 100))
            self._moderate_fires += 1
            self._fire_cooldown_until = step + cooldown
            print(f'lr_ctrl FIRE: {why} at step {step} (scale {self.scale:.4g}) '
                  f'-- rewinding to the rolling checkpoint; the post-restore cut '
                  f'follows (cooldown {cooldown}).')
            return 'diverged'
        # NOT counted here: on_divergence() is the single counting seat.
        print(f'lr_ctrl DISASTER: {why} at step {step} '
              f'(scale {self.scale:.4g}) -- reload')
        return 'diverged'

    #: The same judgement as `observe` without the window side effect, for
    #: `bench/` and the smoke harnesses. It counts at detection: those callers
    #: have no rewind path, so no on_divergence() follows a hit here.
    def check_spike(self, step_type, current_loss, grad_norm):
        why = self.bars.judge(step_type,
                              None if current_loss is None else float(current_loss),
                              None if grad_norm is None else float(grad_norm))
        if why is None:
            return None
        self._divergences += 1
        return 'diverged'

    def on_divergence(self, count: int = 1):
        """Called after a rewind. THE COUNTING SEAT for `lr_ctrl/divergences`,
        and it CUTS: `lr_ctrl['cut']`, the multiplier on the schedule for the
        rest of the stage, and the live scale are multiplied by
        `fire_cut_factor`. The rewind budget (max_reloads_per_1k_steps) is the
        backstop for a loop the cuts cannot break. fire_loss_spike's
        unrecoverable branch raises before reaching this, so the final fatal
        event of an aborting run is not counted."""
        self._divergences += max(int(count), 1)
        factor = float(self._cfg('fire_cut_factor', 0.5))
        st = self._state()
        st['cut'] = float(st.get('cut', 1.0)) * factor
        new = self.set_scale(self.scale * factor, why='disaster_rewind_cut')
        self._last_sched = None
        print(f'lr_ctrl: fire #{self._divergences} -- rewound, and the rate is '
              f'CUT to scale {new:.4g} (x{factor:g}). If this repeats the reload '
              f'budget aborts the run.')

    # ---------------------------------------------------------------- the seat

    def tick(self):
        """One host-loop iteration: set the scale to schedule x cut when that
        product moves, fit the bars when the first segment ends, queue their
        refit when the last rate-changing segment ends, and advance a pending
        refit. Every branch advances or returns."""
        m = self.modeller
        step = int(getattr(m, 'step_ind', 0) or 0)
        segs, ends, refit_end, max_steps = self._schedule()
        elapsed = self._elapsed(step)
        st = self._state()
        value, self._segment = lr_schedule_value(segs, elapsed, max_steps)
        want = value * float(st['cut'])
        # applied only when the product moves, so a scale set by another caller
        # stands until the schedule or the cut next moves
        if want != self._last_sched:
            self._last_sched = want
            if want != self.scale:
                self.set_scale(want, why=f'schedule_segment_{self._segment}')

        prev = self._last_elapsed
        if prev is None:
            prev = -1 if elapsed == 0 else elapsed
        self._last_elapsed = elapsed
        self.phase = BURN_IN if elapsed < ends[0] else CRUISE

        if prev < ends[0] <= elapsed:
            self._fit_warmup_bars(step)
        elif (self.phase == CRUISE and not self.bars.loss_bar
              and self._bars_ready(full=True) and self._cruise_bar is None):
            # NO BARS PAST THE WARM-UP: a process that started past the first
            # segment (a resume), or a first segment that ended with too short a
            # window. Fit them from this process's window once it holds
            # `root_window` observations.
            why = self.bars.derive(self._loss_history, self._grad_history)
            print('lr_ctrl: past the first schedule segment with no bars -- '
                  'hard-failure bars derived from the current window: '
                  + (self.bars.scale_note if why is None
                     else 'NOT DERIVED ({})'.format(why)))

        if refit_end is not None and prev < refit_end <= elapsed:
            self._arm_cruise_bar(step)

        if self.phase == CRUISE:
            self._promoted_steps += 1
            if self._cruise_bar is not None:
                self._tick_cruise_bar(step)

    def _fit_warmup_bars(self, step: int):
        """The first segment has ended: fit the bars from its window (its last
        `root_window` observations), or leave them to tick's fallback when no
        channel holds `min_observations`."""
        if not self._bars_ready(full=False):
            print(f'lr_ctrl: first schedule segment ended at step {step} with under '
                  f'{self.bars.min_observations} observations in the window -- the '
                  f'bars are fitted once it holds {self.bars.root_window}.')
            return
        why = self.bars.derive(self._loss_history, self._grad_history)
        if why is None:
            print(f'lr_ctrl: first schedule segment ended at step {step}; hard-failure '
                  f'bars derived from it -- {self.bars.scale_note}')
        else:
            print(f'lr_ctrl: NO derived hard-failure bar ({why}) -- the run '
                  f'continues on the absolute backstops alone.')

    def _bars_ready(self, full: bool = False) -> bool:
        """Has the observation window filled enough to fit a bar?

        A BOUNDED WAIT: the window fills at one observation per training step,
        so this resolves in at most `hard_failure.min_observations` steps
        (`root_window` with `full=True`) whatever the values are.
        """
        best = max((len(v) for v in self._loss_history.values()), default=0)
        need = self.bars.root_window if full else self.bars.min_observations
        return best >= need

    def _arm_cruise_bar(self, step: int):
        """Queue a refit of the bars at the rate the schedule ends its last
        rate change on. The bars fitted on the first segment stay live until the
        refit replaces them. Not armed with `cruise_rederive` off, nor while no
        bar has been fitted (tick's fallback fits them from a full window)."""
        if not self._cruise_rederive or not self.bars.loss_bar:
            return
        self._cruise_bar = {'prev_loss': dict(self.bars.loss_bar),
                            'prev_grad': self.bars.grad_bar}
        self._bars_redrawn = False
        print(f'lr_ctrl: the schedule\'s last rate change ended at step {step}, scale '
              f'{self.scale:g} -- the excursion bars fitted on the first segment stay '
              f'live; they are refitted from {self.bars.root_window} steps of ordinary '
              f'training at this rate, after a {self._cruise_settle_steps}-step settle.')

    def _tick_cruise_bar(self, step: int):
        """Advance the refit after the last rate change: settle, clear, collect,
        refit.

        The clock is this method's own call count. At `cruise_settle_steps`
        the windows are cleared; the bars are refitted once a channel's window
        holds `root_window` observations, or at `cruise_settle_steps + 4 x
        root_window` calls where none has filled by then.
        """
        cb = self._cruise_bar
        n = cb['ticks'] = cb.get('ticks', 0) + 1
        settle = self._cruise_settle_steps
        if n < settle:
            return
        if n == settle:
            for q in self._loss_history.values():
                q.clear()
            self._grad_history.clear()
            return
        filled = max((len(v) for v in self._loss_history.values()), default=0)
        deadline = settle + 4 * self.bars.root_window
        if filled < self.bars.root_window and n < deadline:
            return
        self._redraw_bars_for_cruise(step, filled)
        self._cruise_bar = None

    def _redraw_bars_for_cruise(self, step: int, filled: int):
        """Refit the bars to the held rate, keeping the previous bar for any
        channel (or the gradient) the new window cannot fit, and all of them
        when nothing can be fitted."""
        cb = self._cruise_bar or {}
        old_loss = dict(cb.get('prev_loss') or {})
        old_grad = cb.get('prev_grad')
        why = self.bars.derive(self._loss_history, self._grad_history)
        if why is not None:
            self.bars.loss_bar, self.bars.grad_bar = old_loss, old_grad
            print(f'lr_ctrl: cruise bars NOT redrawn at step {step} ({why}) -- '
                  f'restoring the bars fitted at the previous rate.')
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

    def on_stage_change(self):
        """Protocol.advance hook, called with the new stage current. The
        optimizers were rebuilt, so the schedule restarts: its entry is stamped
        at this step, the cut returns to 1, the windows, the bars and a pending
        refit are cleared, and the scale is set to the new stage's segment-0
        `to`.

        Returns the first segment's length in train steps (0 when it runs to
        the end of a stage with no cap), for the caller's log.
        """
        m = self.modeller
        self._sched = None
        self._sched_store()['lr_sched_entry'] = int(m.step_ind)
        st = self._state()
        st['cut'] = 1.0
        self.phase = BURN_IN
        self._last_elapsed = -1
        self._last_sched = None
        for q in self._loss_history.values():
            q.clear()
        self._grad_history.clear()
        # Until the new first segment ends only the non-finite and absolute
        # checks stand.
        self.bars.loss_bar = {}
        self.bars.grad_bar = None
        self.bars.scale_note = ''
        self._cruise_bar = None
        self._bars_redrawn = False
        segs, ends, _, _ = self._schedule()
        self.set_scale(segs[0]['to'], why='stage_change')
        print(f'lr_ctrl: stage change -- LR schedule restarted: {self._describe(segs)}.')
        return int(ends[0]) if math.isfinite(ends[0]) else 0

    #: The name protocol.py and the smoke harnesses still call. Same transaction.
    rearm_warmup = on_stage_change

    def in_burn_in(self) -> bool:
        return self.phase == BURN_IN

    #: Retained spelling for the diagnostic sensors, which decline to sample
    #: during burn-in.
    in_warmup = in_burn_in

    # ------------------------------------------------- diagnostics (off by default)

    def calibration_refusal(self):
        """Why a ray reading would be thrown away, decided without measuring --
        or None. The ray reaches no learning rate, so nothing is refused."""
        return None

    def on_calibration(self, reading):
        """Record one ray calibration. IT MOVES NOTHING."""
        self._calibrations += 1
        self._last_ray = dict(reading or {})

    # ---------------------------------------------------------------- report

    _PHASE_CODE = {BURN_IN: 0, CRUISE: 2}

    def _emit(self, st):
        self._report = {
            'lr_ctrl/scale': float(st['scale']),
            'lr_ctrl/cut': float(st.get('cut', 1.0)),
            'lr_ctrl/segment': float(self._segment),
            'lr_ctrl/divergences': float(self._divergences),
            'lr_ctrl/moderate_fires': float(self._moderate_fires),
            'lr_bracket/phase': float(self._PHASE_CODE.get(self.phase, -1)),
            'lr_bracket/promoted_steps': float(self._promoted_steps),
        }
        self._report.update(self.bars.report())
        self._report['lr_bracket/bars_redrawn'] = float(bool(self._bars_redrawn))
        # Published only when a cap is configured, so its ABSENCE means "no rail"
        # rather than "rail never bound".
        if self._max_lr() is not None:
            self._report['lr_ctrl/lr_capped_groups'] = float(self._lr_capped_groups)
        self._report['lr_ctrl/lr_floored_groups'] = float(self._lr_floored_groups)
        # The diagnostic sensor publishes only when it ran.
        if self._calibrations:
            self._report['lr_ctrl/calibrations'] = float(self._calibrations)

    def report(self):
        return dict(self._report)


def _get(node, name, default=None):
    return default if node is None else getattr(node, name, default)
