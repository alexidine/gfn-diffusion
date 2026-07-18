"""
The unified stage interface: one declarative, config-level description of the
training protocol, replacing the tangled phase (1/2/3) + forward-first (A/B)
frameworks (phases.PhaseController, controller.ModeBalanceController,
controller.ForwardFirstController) with a single engine.

A protocol is an ordered list of STAGES (config: protocol.stages). Each stage
declares:

  name              unique identifier; checkpoints store it as the run's position
  train_mode        'bwd' | 'fused' -- what train_logic returns every step
  bwd_sampling_mode 'dataset' | 'prior' -- where backward draws terminals from
  flags             explicit behavior switches read by train.py (update_log_z,
                    scramble_conditions, zgap_mol_sampling, buffers_active,
                    mle_gate) -- the replacements for the old `self.phase == N`
                    integer checks
  loss_coeffs       per-mode dicts of NON-DEFAULT coefficient overrides; the
                    base config's fwd/bwd/replay_loss_coeffs blocks are the
                    defaults, and a stage's live coeffs are a pure function of
                    (defaults, stage) -- no schedules, no mutation, and
                    checkpoints no longer carry behavior, only position
  fracs             mode fractions applied once at stage entry (omit to carry
                    the previous stage's fracs across)
  balance           the mode-frac controller for fused stages (see below)
  exit              trigger: an AND-list of {metric, above|below, patience}
                    terms; when it fires, the run advances to the next stage.
                    A stage with no exit is terminal.
  on_exit/on_enter  transition actions (snapshot:<tag>, snapshot_prior,
                    bootstrap_z) -- the route-specific physics; everything
                    generic (optimizer rebuild, monitor cooldown, LR re-warm)
                    happens automatically at EVERY transition
  skip_if           entry condition ('prior_loaded'): on a fresh run the stage
                    is skipped when the condition holds (e.g. the MLE warm-
                    start is redundant when a prior model was loaded by path)

Balance rules (kind: lexicographic) walk in order; the FIRST violated rule's
`boost` mode gets the frac nudge this tick (the same EMA-toward-one-hot nudge
ModeBalanceController and ForwardFirstController both used); `default_boost`
takes it when all rules are clean, and a clean streak of
controller.anneal_patience ticks tightens every annealed rule's threshold.
A rule is either absolute (`above: X`, annealable) or relative to its own
running best (`relative: best, margin: M` -- "is this metric DEGRADING",
never "is it below an absolute bar"; the calibration floor legitimately
rises as coverage grows, so absolute bars deadlock -- see b9ze0p5c). Rules
with `anneal` get floor tracking: a metric that stalls while its own mode
holds priority has hit its achievable floor, so it yields priority and stops
annealing until a large upward jump (new work) re-arms it -- without this,
annealing walks the threshold below the floor and pins that priority forever
(the old tb_err deadlock). kind: proportional instead splits two modes'
combined mass proportionally to a pair of lagging-spread metrics (the old
phase-2 balancer, generalized).

Mode dormancy is DERIVED, not flagged: a mode no rule (and no default_boost)
ever boosts has no way off the frac floor, so it is dormant -- the fused step
skips even its force-refresh rollout (the old bwd_dormant, generalized).

Exit triggers resolve metric names against the RUNNING metric_tracker values
('dir/name'), gate-published values ('gates/name', e.g. the MLE slope gate's
gates/mle_flat), or -- explicit opt-in only, since evals are rare -- the eval
metrics dict ('eval/name', e.g. eval/wass_debiased). Tick-resolvable terms
are checked every 10 steps; the moment they all reach patience, the next eval
is pulled forward (stage_ctrl['request_eval'], the generalization of the old
MLE-gate request_eval) and the transition executes inside evaluation() with
fresh eval metrics in hand -- which is also what makes a reloaded
pre-transition snapshot ('phase1_exit' etc., saved with request_eval stamped
True) replay its transition through the normal path on its first post-resume
eval.

All mutable engine state (rule bests / floors / streaks, live annealed
thresholds, gate latches, request_eval) lives in modeller.stage_ctrl, which
is checkpointed and reset at every stage transition. Live thresholds riding
in stage_ctrl also fixes a latent legacy bug: the old controllers annealed
args.controller.*_threshold in place, which was never checkpointed, so anneal
progress silently reset on every resume.
"""

import math
import os
from copy import deepcopy

import numpy as np

MODES = ('fwd', 'bwd', 'replay')
TRAIN_MODES = ('bwd', 'fused')
BWD_SAMPLING_MODES = ('dataset', 'prior')
STAGE_FLAGS = ('update_log_z', 'scramble_conditions', 'zgap_mol_sampling',
               'buffers_active', 'mle_gate')
ACTIONS = ('snapshot', 'snapshot_prior', 'bootstrap_z')
SKIP_CONDITIONS = ('prior_loaded',)
RULE_KEYS = {'metric', 'boost', 'above', 'below', 'relative', 'margin', 'drift', 'floor',
             'abs', 'if_missing', 'lookahead', 'anneal'}
TERM_KEYS = {'metric', 'above', 'below', 'abs', 'patience'}


def fresh_stage_ctrl():
    """The per-stage mutable engine state, reset at every transition and
    checkpointed via MODELLER_STATE_DEFAULTS['stage_ctrl']."""
    return {
        'gates': {},        # gate-published values, e.g. {'mle_flat': 1.0}
        'gate_state': {},   # gate internals, e.g. the MLE slope gate's window
        'rules': {},        # rule index -> {'best', 'thr', 'floor', 'look'}
        'exit': {},         # exit term index -> consecutive-pass streak
        'anneal_streak': 0,
        'boost': None,      # last chosen boost mode (logging)
        'exit_armed': False,
        'request_eval': False,
    }


class Stage:
    """Parsed + validated view of one config stage dict. Pure data; all the
    engine state lives in modeller.stage_ctrl."""

    def __init__(self, spec: dict, index: int):
        if not isinstance(spec, dict):
            raise TypeError(f"protocol.stages[{index}] must be a mapping, got {type(spec)}")
        unknown = set(spec) - {'name', 'train_mode', 'bwd_sampling_mode', 'flags',
                               'loss_coeffs', 'fracs', 'balance', 'exit',
                               'on_exit', 'on_enter', 'skip_if'}
        if unknown:
            raise ValueError(f"protocol.stages[{index}] has unknown keys {sorted(unknown)}")
        self.index = index
        self.name = spec.get('name')
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"protocol.stages[{index}] needs a string 'name'")
        self.train_mode = spec.get('train_mode', 'fused')
        if self.train_mode not in TRAIN_MODES:
            raise ValueError(f"stage '{self.name}': train_mode must be one of {TRAIN_MODES}")
        self.bwd_sampling_mode = spec.get('bwd_sampling_mode', 'prior')
        if self.bwd_sampling_mode not in BWD_SAMPLING_MODES:
            raise ValueError(f"stage '{self.name}': bwd_sampling_mode must be one of {BWD_SAMPLING_MODES}")

        self.flags = dict(spec.get('flags') or {})
        bad = set(self.flags) - set(STAGE_FLAGS)
        if bad:
            raise ValueError(f"stage '{self.name}': unknown flags {sorted(bad)} "
                             f"(known: {STAGE_FLAGS})")

        self.loss_coeffs = {m: dict(v) for m, v in (spec.get('loss_coeffs') or {}).items()}
        bad = set(self.loss_coeffs) - set(MODES)
        if bad:
            raise ValueError(f"stage '{self.name}': loss_coeffs for unknown modes {sorted(bad)}")

        self.fracs = dict(spec.get('fracs') or {})
        if self.fracs:
            bad = set(self.fracs) - set(MODES)
            if bad:
                raise ValueError(f"stage '{self.name}': fracs for unknown modes {sorted(bad)}")

        self.balance = self._parse_balance(spec.get('balance'))
        self.exit = self._parse_exit(spec.get('exit'))
        self.on_exit = self._parse_actions(spec.get('on_exit'), 'on_exit')
        self.on_enter = self._parse_actions(spec.get('on_enter'), 'on_enter')

        self.skip_if = spec.get('skip_if')
        if self.skip_if is not None and self.skip_if not in SKIP_CONDITIONS:
            raise ValueError(f"stage '{self.name}': skip_if must be one of {SKIP_CONDITIONS}")

    # ------------------------------------------------------------ sub-parsers

    def _parse_balance(self, node):
        if node is None:
            return None
        node = dict(node)
        kind = node.get('kind', 'lexicographic')
        if kind == 'lexicographic':
            rules = node.get('rules') or []
            for i, r in enumerate(rules):
                bad = set(r) - RULE_KEYS
                if bad:
                    raise ValueError(f"stage '{self.name}' rule {i}: unknown keys {sorted(bad)}")
                if r.get('boost') not in MODES:
                    raise ValueError(f"stage '{self.name}' rule {i}: boost must be one of {MODES}")
                forms = ('above' in r) + ('below' in r) + ('relative' in r)
                if forms != 1:
                    raise ValueError(f"stage '{self.name}' rule {i}: exactly one of "
                                     f"'above' (absolute), 'below' (floor, e.g. an r2 "
                                     f"calibration gate), or 'relative: best' is required")
                if 'relative' in r and r['relative'] != 'best':
                    raise ValueError(f"stage '{self.name}' rule {i}: only 'relative: best' is supported")
                if 'below' in r and r.get('anneal'):
                    raise ValueError(f"stage '{self.name}' rule {i}: 'below' rules don't anneal "
                                     f"(tightening would RAISE the bar -- set it where you mean it)")
                if r.get('if_missing', 'clean') not in ('clean', 'violated'):
                    raise ValueError(f"stage '{self.name}' rule {i}: if_missing must be clean|violated")
            default = node.get('default_boost')
            if isinstance(default, dict):
                # idle MIX: when all rules are clean, nudge toward this frac
                # split instead of a one-hot (e.g. {replay: 0.9, fwd: 0.1} --
                # keep the fit clean AND hold Z gently at its fixed point,
                # rather than letting fwd decay below the deactivate threshold
                # and only revisiting Z after zerr has drifted over its bar)
                bad = set(default) - set(MODES)
                if bad or not default:
                    raise ValueError(f"stage '{self.name}': default_boost mix has "
                                     f"unknown/empty modes {sorted(bad)}")
                if any(not isinstance(w, (int, float)) or w <= 0 for w in default.values()):
                    raise ValueError(f"stage '{self.name}': default_boost mix weights must be positive")
                total = float(sum(default.values()))
                node['default_boost'] = {m: w / total for m, w in default.items()}
            elif default not in MODES:
                raise ValueError(f"stage '{self.name}': balance.default_boost must be one of "
                                 f"{MODES} or a {{mode: weight}} mix")
            node['rules'] = [dict(r) for r in rules]
        elif kind == 'proportional':
            metrics = node.get('metrics') or {}
            if len(metrics) != 2 or set(metrics) - set(MODES):
                raise ValueError(f"stage '{self.name}': proportional balance needs a "
                                 f"'metrics' mapping of exactly two modes to metric names")
            node['metrics'] = dict(metrics)
        else:
            raise ValueError(f"stage '{self.name}': balance.kind must be lexicographic|proportional")
        node['kind'] = kind
        return node

    def _parse_exit(self, node):
        if node is None:
            return None
        terms = []
        for i, t in enumerate(node):
            bad = set(t) - TERM_KEYS
            if bad:
                raise ValueError(f"stage '{self.name}' exit term {i}: unknown keys {sorted(bad)}")
            if ('above' in t) == ('below' in t):
                raise ValueError(f"stage '{self.name}' exit term {i}: exactly one of "
                                 f"'above'/'below' is required")
            terms.append(dict(t))
        return terms

    def _parse_actions(self, node, where):
        actions = []
        for a in (node or []):
            name, _, arg = str(a).partition(':')
            name, arg = name.strip(), arg.strip()
            if name not in ACTIONS:
                raise ValueError(f"stage '{self.name}' {where}: unknown action '{a}' "
                                 f"(known: {ACTIONS})")
            actions.append((name, arg))
        return actions

    @property
    def active_modes(self):
        """Modes this stage's balance can ever boost. A mode outside this set
        can never rise off the frac floor, so the fused step may exclude it
        outright (the old 'replay joins in phase 3 only' check, derived). A
        stage without balance makes no claims."""
        if self.balance is None:
            return set(MODES)
        if self.balance['kind'] == 'proportional':
            return set(self.balance['metrics'])
        out = {r['boost'] for r in self.balance['rules']}
        default = self.balance['default_boost']
        out.update(default if isinstance(default, dict) else {default})
        return out

    @property
    def read_modes(self):
        """Modes whose rolling metric_tracker stats something in this stage
        actually READS -- a balance-rule metric, a proportional metric, or an
        exit term. The force-refresh rollout exists purely to keep a
        low-frac mode's stats fresh for those readers, so a mode nobody reads
        skips it entirely (the old bwd_dormant, generalized: stage A's
        boundary servo can BOOST bwd off fwd/box_violation without ever
        reading a bwd channel, and once boosted above the deactivate
        threshold it trains for real and produces stats anyway)."""
        if self.balance is None:
            return set(MODES)
        names = []
        if self.balance['kind'] == 'proportional':
            names += list(self.balance['metrics'].values())
        else:
            names += [r['metric'] for r in self.balance['rules']]
        for term in (self.exit or []):
            names.append(term['metric'])
        out = set()
        for name in names:
            direction = name.partition('/')[0]
            if direction in MODES:
                out.add(direction)
        return out


class StageProtocol:
    """The engine. Holds a reference to its owning Modeller (same pattern as
    the other controllers); all mutable state lives on the Modeller (stage,
    stage_ctrl, fracs), so checkpoint save/load needs nothing new beyond
    those two fields."""

    def __init__(self, modeller):
        self.m = modeller
        self._stages = None       # parsed lazily: args may still be assembling at __init__
        self._coeff_defaults = None

    # ------------------------------------------------------------------ parse

    @property
    def stages(self):
        if self._stages is None:
            node = getattr(self.m.args, 'protocol', None)
            specs = getattr(node, 'stages', None) if node is not None else None
            if not specs:
                raise ValueError("config needs a protocol.stages list (see configs/mk_dev.yaml)")
            self._stages = [Stage(s, i) for i, s in enumerate(specs)]
            names = [s.name for s in self._stages]
            if len(set(names)) != len(names):
                raise ValueError(f"protocol.stages names must be unique, got {names}")
        return self._stages

    @property
    def stage(self) -> Stage:
        name = getattr(self.m, 'stage', None)
        if name is None:
            return self.stages[0]
        for s in self.stages:
            if s.name == name:
                return s
        raise ValueError(
            f"checkpoint/run is at stage '{name}', which this config's protocol does not "
            f"define (stages: {[s.name for s in self.stages]}) -- rename the config stage "
            f"back, or map the checkpoint by editing its stored stage")

    @property
    def ctrl(self) -> dict:
        st = getattr(self.m, 'stage_ctrl', None)
        if not isinstance(st, dict):
            st = fresh_stage_ctrl()
            self.m.stage_ctrl = st
        return st

    def flag(self, name: str) -> bool:
        if name not in STAGE_FLAGS:
            raise KeyError(f"unknown stage flag '{name}'")
        return bool(self.stage.flags.get(name, False))

    def mode_boostable(self, mode: str) -> bool:
        """Can this stage's balance ever boost `mode`? False means it can never
        leave the frac floor, so the fused step excludes it from the loss
        outright (the old 'replay joins in phase 3 only' phase check, derived)."""
        return mode in self.stage.active_modes

    def mode_dormant(self, mode: str) -> bool:
        """True when nothing in this stage reads `mode`'s rolling stats, so the
        fused step may skip even its force-refresh rollout (the old
        bwd_dormant, generalized -- see Stage.read_modes)."""
        return mode not in self.stage.read_modes

    # ----------------------------------------------------------------- coeffs

    def coeffs(self, mode: str) -> dict:
        """Live loss coefficients for `mode`: the base config block (captured
        once, pristine) overlaid with the current stage's overrides. Unknown
        override keys are a config error, not a silent no-op."""
        if self._coeff_defaults is None:
            self._coeff_defaults = {
                m: {k: v for k, v in vars(getattr(self.m.args, f'{m}_loss_coeffs')).items()
                    if isinstance(v, (int, float))}
                for m in MODES}
        base = self._coeff_defaults[mode]
        overrides = self.stage.loss_coeffs.get(mode, {})
        unknown = set(overrides) - set(base)
        if unknown:
            raise ValueError(f"stage '{self.stage.name}' {mode} loss_coeffs override unknown "
                             f"keys {sorted(unknown)} -- add them to the base "
                             f"{mode}_loss_coeffs block first")
        return {**base, **overrides}

    # ---------------------------------------------------------------- metrics

    def _resolve(self, name: str, eval_metrics: dict = None):
        """'gates/x' -> gate-published values; 'eval/x' -> the eval metrics
        dict (only available inside maybe_advance); 'dir/x' -> the running
        metric_tracker EMA. Returns a finite float or None."""
        if name.startswith('gates/'):
            v = self.ctrl['gates'].get(name[len('gates/'):])
        elif name.startswith('eval/'):
            v = None if eval_metrics is None else eval_metrics.get(name[len('eval/'):])
        else:
            direction, _, metric = name.partition('/')
            if not metric:
                raise ValueError(f"metric name '{name}' must look like dir/name, gates/name or eval/name")
            v = self.m.metric_tracker.get(direction, metric)
        if v is None:
            return None
        v = float(v)
        return v if math.isfinite(v) else None

    def publish_gate(self, name: str, value: float):
        """Gate publishers (e.g. the MLE slope gate) drop their verdicts here;
        triggers and rules read them as 'gates/<name>'."""
        self.ctrl['gates'][name] = float(value)

    def gate_state(self, name: str) -> dict:
        """Scratch dict for a gate's internals (windows, latches), checkpointed
        with stage_ctrl and reset at transitions like everything else here."""
        return self.ctrl['gate_state'].setdefault(name, {})

    # ------------------------------------------------------------------- tick

    def tick(self):
        """10-step-cadence work: the balance nudge, then exit-trigger arming.
        Transitions themselves only execute inside evaluation() (maybe_advance),
        with fresh eval metrics in hand."""
        if self.stage.balance is not None:
            self._balance_tick()
        self._exit_tick()

    # ------------------------------------------------------------- exit logic

    def _term_passes(self, term: dict, eval_metrics: dict = None):
        v = self._resolve(term['metric'], eval_metrics)
        if v is None:
            return False
        if term.get('abs'):
            v = abs(v)
        return (v > term['above']) if 'above' in term else (v < term['below'])

    def _exit_tick(self):
        """Advance each tick-resolvable term's pass-streak; on the rising edge
        of 'all tick terms at patience', pull the next eval forward. eval/*
        terms can't be watched at tick cadence -- they are checked with fresh
        values inside maybe_advance, exactly like the old phase-1 gate latched
        on the MLE slope and left wass to the pulled-forward eval."""
        stage = self.stage
        if stage.exit is None:
            return
        streaks = self.ctrl['exit']
        armed = True
        for i, term in enumerate(stage.exit):
            if term['metric'].startswith('eval/'):
                continue
            streaks[i] = streaks.get(i, 0) + 1 if self._term_passes(term) else 0
            if streaks[i] < term.get('patience', 1):
                armed = False
        if armed and not self.ctrl['exit_armed']:
            self.ctrl['request_eval'] = True  # pull the eval that will run maybe_advance
        self.ctrl['exit_armed'] = armed

    def _exit_satisfied(self, eval_metrics: dict) -> bool:
        stage = self.stage
        if stage.exit is None:
            return False  # terminal stage
        for i, term in enumerate(stage.exit):
            if term['metric'].startswith('eval/'):
                if not self._term_passes(term, eval_metrics):
                    return False
            elif self.ctrl['exit'].get(i, 0) < term.get('patience', 1):
                return False
        return True

    def maybe_advance(self, eval_metrics: dict) -> bool:
        """Called from evaluation() after metrics are computed -- the one place
        transitions execute. Clears any pending pulled-forward eval request
        (this eval satisfies it, whoever set it: the trigger arming tick, or a
        reloaded pre-transition snapshot's stamped request_eval)."""
        self.ctrl['request_eval'] = False
        if not self._exit_satisfied(eval_metrics):
            return False
        self.advance(eval_metrics)
        return True

    # ------------------------------------------------------------ transitions

    def begin(self):
        """Fresh-run entry (step 0): pin the starting stage, then walk the
        skip chain -- a stage whose skip_if condition holds is skipped without
        its on_exit actions (nothing to snapshot: the work it represents
        never ran). Resumed runs are left wherever their checkpoint says."""
        m = self.m
        if getattr(m, 'stage', None) is None:
            m.stage = self.stages[0].name
        if m.step_ind != 0:
            return
        while self.stage.skip_if is not None and self.stage.index + 1 < len(self.stages):
            if self.stage.skip_if == 'prior_loaded':
                if not hasattr(m, 'prior_model'):
                    break
                # the loaded prior IS the product of the skipped warm-start, but
                # it is a sampling-only object that may be a different
                # architecture than the live model, so the policy is
                # deliberately NOT warm-started from it -- warm-start explicitly
                # via checkpoint_name + load_weights_only when the architectures
                # do match
                print(f"protocol: prior loaded by path -- skipping stage "
                      f"'{self.stage.name}' (policy weights untouched)")
            self.advance(None, run_exit_actions=False)

    def advance(self, eval_metrics, run_exit_actions: bool = True):
        """Execute the transition to the next stage: outgoing stage's on_exit
        actions (pre-transition snapshots first, while nothing has mutated),
        then the switch (stage name, fresh stage_ctrl, entry fracs, coeffs),
        then the AUTOMATIC optimization reset every boundary gets -- spike
        monitors reset + cooldown, combo record cleared, optimizers rebuilt
        (fresh Adam moments for the new loss surface), LR re-warmed with a
        fresh probe baseline -- and finally the incoming stage's on_enter
        actions (e.g. bootstrap_z, which wants the new coeffs active and the
        fresh eval metrics in hand)."""
        m = self.m
        old, new = self.stage, self.stages[self.stage.index + 1]

        if run_exit_actions:
            for name, arg in old.on_exit:
                self._run_action(name, arg, eval_metrics)

        print(f"protocol: stage '{old.name}' -> '{new.name}'")
        m.stage = new.name
        m.stage_ctrl = fresh_stage_ctrl()
        if new.fracs:
            total = float(sum(new.fracs.values()))
            for mode in MODES:
                setattr(m, f'{mode}_frac', float(new.fracs.get(mode, 0.0)) / total)

        # automatic optimization reset -- every transition, uniformly:
        # the outgoing stage's loss windows are a stale ceiling for the incoming
        # stream, its best-checkpoint minima shouldn't gate the new stage's
        # 'best' saves, its Adam moments describe the wrong loss surface, and
        # the LR controller's probe baseline describes the wrong policy
        # semantics. rearm_warmup must run with m.stage already switched (its
        # _state() phase-sync branch runs first and it overrides).
        m.lr_controller.reset_spike_monitors(m.lr_controller.CHANNELS)
        m.combo_loss_record = []
        m.init_schedulers_optimizers()
        m.set_loss_coeffs()
        warmup_ticks = m.lr_controller.rearm_warmup()
        if warmup_ticks:
            print(f"protocol: optimizers rebuilt, LR re-warming over {warmup_ticks} ticks "
                  f"(~{warmup_ticks * 10} train steps)")

        for name, arg in new.on_enter:
            self._run_action(name, arg, eval_metrics)
        print(f"protocol: stage '{new.name}' engaged (fwd {m.fwd_frac:.3f} / "
              f"bwd {m.bwd_frac:.4f} / replay {m.replay_frac:.3f}, "
              f"train_mode {new.train_mode}, bwd_sampling {new.bwd_sampling_mode})")

    def _run_action(self, name: str, arg: str, eval_metrics):
        if name == 'snapshot':
            self._snapshot(arg or 'stage_exit')
        elif name == 'snapshot_prior':
            self._snapshot_prior()
        elif name == 'bootstrap_z':
            self._bootstrap_z(eval_metrics, train_conditioner=(arg == 'train_conditioner'))

    def _snapshot(self, tag: str):
        """Pre-transition snapshot: the untouched end-state of the outgoing
        stage, the reload point for replaying transition behavior under changed
        code/config without retraining the stage. request_eval is stamped True
        into the SAVED state only (live value restored right after), so a
        resume from this snapshot pulls its eval to the first post-resume step
        and the exit trigger -- whose pass-streaks ride in the same saved
        stage_ctrl -- re-fires through the normal eval -> maybe_advance path.
        with_buffers freezes this snapshot's own buffer sidecar: the rolling
        one is overwritten at the next eval, which would leave a replay from
        here running against buffers from a stage this snapshot predates."""
        prev = self.ctrl.get('request_eval', False)
        self.ctrl['request_eval'] = True
        self.m.checkpointer.save(tag, with_buffers=True)
        self.ctrl['request_eval'] = prev

    def _snapshot_prior(self):
        """Freeze the converged warm-start as THE prior: checkpoint it, keep a
        frozen eval copy of the EMA model for backward sampling, and delete the
        outgoing stage's 'best' checkpoint -- it converged, 'prior' supersedes
        it, and the best-gate restarts on the cleared combo record. (Backward
        draws switch to the prior buffer via the next stage's declarative
        bwd_sampling_mode -- no flip here any more.)"""
        m = self.m
        m.checkpointer.save('prior')
        m.prior_model = deepcopy(m.ema_model)
        m.prior_model.eval()
        # tolerate a missing 'best': when the transition is REPLAYED from a
        # reloaded pre-transition snapshot, the original run already deleted it
        best_path = m.checkpointer.path_for('best')
        if os.path.exists(best_path):
            os.remove(best_path)

    def _bootstrap_z(self, eval_metrics, train_conditioner: bool = False):
        """Z warm start at a TB-stage entry. Unconditional single-scalar Z is
        filled directly from the eval's empirical estimate; the conditional
        analog fits flow_model's Z(c) onto condition_log_z's ema_logw via a
        short rollout-free regression (see bootstrap_log_z). train_conditioner
        ('bootstrap_z:train_conditioner') additionally lets the fit shape the
        conditioner -- declare it only when the preceding prior stage scrambled
        its conditions (scramble_conditions), so nothing ever trained the
        conditioner and this fit is the first thing that does; see
        bootstrap_log_z's docstring."""
        m = self.m
        if (not m.gfn_model.full_flow) and (not m.gfn_model.conditional):
            if eval_metrics is None or 'eval_fwd/jensen_z' not in eval_metrics:
                raise RuntimeError("bootstrap_z needs eval metrics (eval_fwd/jensen_z) -- "
                                   "it can only run from an eval-time transition")
            empirical_z = eval_metrics['eval_fwd/jensen_z']
            m.gfn_model.flow_model.scalar.data.fill_(empirical_z)
            m.ema_model.flow_model.scalar.data.fill_(empirical_z)
        else:
            m.bootstrap_log_z(train_conditioner=train_conditioner)

    # ---------------------------------------------------------------- balance

    def _rule_state(self, i):
        return self.ctrl['rules'].setdefault(i, {})

    def _rule_value(self, rule, rs):
        """Resolve + condition one rule's metric: abs, then the optional
        log-space lookahead projection (port of the old controller's
        _log_ema_lookahead, state per rule)."""
        v = self._resolve(rule['metric'])
        if v is None:
            return None
        if rule.get('abs'):
            v = abs(v)
        if rule.get('lookahead'):
            v = self._lookahead(rs.setdefault('look', {'level': None, 'trend': 0.0}), v)
        return v

    def _lookahead(self, state, value):
        """Log-space trend projection (verbatim port from
        ModeBalanceController._log_ema_lookahead): the metrics are nonnegative
        nats on a log scale, so the trend is multiplicative and can never
        project below zero; capped at e^±3 because beyond ~20x the
        extrapolation is guessing."""
        ctrl = self.m.args.controller
        if not math.isfinite(value):
            return value
        if state['level'] is None:
            state['level'] = value
            return value
        eps = 1e-3
        log_delta = math.log(max(value, eps)) - math.log(max(state['level'], eps))
        trend_alpha = getattr(ctrl, 'lookahead_trend_alpha', 0.1)
        horizon = getattr(ctrl, 'lookahead_horizon', 5)
        state['trend'] = trend_alpha * log_delta + (1 - trend_alpha) * state['trend']
        state['level'] = value
        exponent = min(max(horizon * state['trend'], -3.0), 3.0)
        return value * math.exp(exponent)

    def _rule_threshold(self, rule, rs):
        """The live threshold: for absolute rules the annealable value seeded
        from 'above'; for relative rules best x margin, where the running best
        (a MINIMUM) relaxes upward by 'drift'/tick so a lucky low reading can't
        pin the bar unreachably, floored by 'floor' so an early near-zero batch
        can't make the ratio hypersensitive forever."""
        if 'above' in rule:
            if 'thr' not in rs:
                rs['thr'] = float(rule['above'])
            return rs['thr']
        return None  # relative rules resolve inside _rule_violated

    def _rule_violated(self, rule, rs, value):
        if value is None:
            return rule.get('if_missing', 'clean') == 'violated'
        if rs.get('floor', {}).get('floored'):
            return False  # at its achievable floor: yields priority (see _floor_track)
        if 'below' in rule:
            # floor violation (e.g. fwd/r2 below 0.9): fixed bar, no anneal.
            # Bounded, saturating metrics like r2 make LOOSE gates here -- they
            # sit at ~1 through routine absorption churn (7laa8lbl: r2 held
            # 1.00 while scatter tripled during the mode annexation), so this
            # form doesn't steal priority from coverage work the way a tight
            # elevation margin does.
            return value < float(rule['below'])
        if 'above' in rule:
            return value > self._rule_threshold(rule, rs)
        # relative-to-own-running-best
        best = rs.get('best')
        if best is None:
            rs['best'] = value  # seed tick: elevation is 1.0, never violated
            return False
        drift = float(rule.get('drift', 0.003))
        rs['best'] = min(value, best * (1.0 + drift))
        elevation = value / max(rs['best'], float(rule.get('floor', 1e-8)))
        rs['elevation'] = elevation
        return elevation > float(rule.get('margin', 1.3))

    def _floor_track(self, rule, rs, value, held_priority):
        """A metric that stalls while its own mode holds priority has hit its
        achievable floor -- it yields priority and stops annealing; a jump of
        floor_reset_frac above the anchor means new work appeared and re-arms.

        'Stalls' is a WINDOWED NET-improvement test, deliberately not a
        per-dip best-chase: the stall clock runs while priority is held, and
        at floor_patience ticks the rule floors unless the metric improved by
        floor_improve_frac NET against the window's anchor (in which case the
        anchor re-bases and the clock restarts -- genuine descent never
        floors). The original form reset the clock on ANY 2% dip below a
        ratcheting all-time best, which a noisy metric defeats forever:
        over_coverage is an order-statistic RMS wiggling +-20% around its
        sampling floor, so it printed a fresh 'best' every few ticks and held
        replay priority for 6000+ steps against a hard-violated under rule
        (aijrfwuy, relative_under 5.7 -> 9.0 while bwd sat at the frac floor)
        -- the exact unreachable-threshold deadlock this tracker exists to
        break, re-entered through the noise loophole."""
        if value is None or not math.isfinite(value):
            return
        bal = self.stage.balance
        improve = bal.get('floor_improve_frac', 0.02)
        fl = rs.setdefault('floor', {'anchor': value, 'stall': 0, 'floored': False})
        if 'anchor' not in fl:  # state written by the pre-fix ('best'-keyed) tracker
            fl['anchor'] = fl.pop('best', value)
        if value > fl['anchor'] * (1.0 + bal.get('floor_reset_frac', 0.5)):
            fl['anchor'] = value  # new work appeared: re-base and re-arm
            fl['stall'] = 0
            fl['floored'] = False
        elif fl['floored']:
            # yielded: something else may still improve it -- a genuine net
            # descent below the anchor proves it responsive again
            if value < fl['anchor'] * (1.0 - improve):
                fl['anchor'] = value
                fl['stall'] = 0
                fl['floored'] = False
        elif held_priority:
            fl['stall'] += 1
            if fl['stall'] >= bal.get('floor_patience', 50):
                if value < fl['anchor'] * (1.0 - improve):
                    fl['anchor'] = value  # net descent over the window: still responding
                    fl['stall'] = 0
                else:
                    fl['floored'] = True

    def _anneal(self):
        """All rules clean for anneal_patience ticks: tighten every annealed,
        un-floored ABSOLUTE rule's live threshold by controller.decay_rate
        (per-rule 'rate' overrides), down to its 'min' -- a number, or a metric
        name resolved live (the dynamic floor that replaced the old
        replay-outcompeting guard: e.g. min: fwd/scatter_err says 'never
        require replay to beat fresh on-policy')."""
        ctrl = self.m.args.controller
        for i, rule in enumerate(self.stage.balance['rules']):
            spec = rule.get('anneal')
            if not spec or 'above' not in rule:
                continue
            rs = self._rule_state(i)
            if rs.get('floor', {}).get('floored'):
                continue
            rate = float(spec.get('rate', getattr(ctrl, 'decay_rate', 0.95)))
            floor = spec.get('min', 0.0)
            if isinstance(floor, str):
                floor = self._resolve(floor)
                if floor is None:
                    continue  # dynamic floor unavailable: hold rather than guess
            thr = self._rule_threshold(rule, rs)
            if thr > floor:
                rs['thr'] = max(thr * rate, float(floor))

    def _balance_tick(self):
        bal = self.stage.balance
        if bal['kind'] == 'proportional':
            self._proportional_tick(bal)
            return
        ctrl = self.m.args.controller

        chosen = None
        values = {}
        for i, rule in enumerate(bal['rules']):
            rs = self._rule_state(i)
            v = self._rule_value(rule, rs)
            values[i] = v
            # every rule is evaluated EVERY tick, even once a higher rule has
            # already won: a relative rule's running best must keep tracking
            # while it is outranked (the legacy controllers computed every
            # elevation unconditionally), or it would re-baseline at whatever
            # level the metric drifted to during the excursion and mask the
            # degradation the detector exists to catch
            if self._rule_violated(rule, rs, v) and chosen is None:
                chosen = rule['boost']
        if chosen is None:
            chosen = bal['default_boost']
            self.ctrl['anneal_streak'] += 1
            if self.ctrl['anneal_streak'] >= getattr(ctrl, 'anneal_patience', 5):
                self._anneal()
                self.ctrl['anneal_streak'] = 0
        else:
            self.ctrl['anneal_streak'] = 0

        # the boost's NAME for floor tracking and logging: a mix default counts
        # as its dominant mode (so e.g. the over rule keeps accumulating floor
        # stall through idle ticks exactly as it did under a one-hot replay
        # default), while the nudge below receives the full mix
        chosen_name = (max(chosen, key=chosen.get) if isinstance(chosen, dict) else chosen)

        # floor tracking counts only while that rule's own mode holds the boost
        for i, rule in enumerate(bal['rules']):
            if rule.get('anneal'):
                self._floor_track(rule, self._rule_state(i), values[i],
                                  held_priority=(chosen_name == rule['boost']))

        self.ctrl['boost'] = chosen_name
        self._nudge_mode_fracs(chosen)

    def _proportional_tick(self, bal):
        """Port of PhaseController.phase2_balance_step, generalized to any two
        modes: split their combined frac mass proportionally to each side's
        remaining spread metric, EMA-nudged, floored both sides; hold when
        either metric is missing rather than starving on absent data."""
        m = self.m
        (mode_a, metric_a), (mode_b, metric_b) = bal['metrics'].items()
        s_a, s_b = self._resolve(metric_a), self._resolve(metric_b)
        if s_a is None or s_b is None:
            return
        floor = float(bal.get('floor', 0.01))
        alpha = float(bal.get('alpha', 0.05))
        frac_a = getattr(m, f'{mode_a}_frac')
        frac_b = getattr(m, f'{mode_b}_frac')
        total = frac_a + frac_b
        if total <= 0:
            return
        target = s_a / max(s_a + s_b, 1e-8)
        target = min(max(target, floor), 1.0 - floor)
        share_a = (1.0 - alpha) * (frac_a / total) + alpha * target
        setattr(m, f'{mode_a}_frac', share_a * total)
        setattr(m, f'{mode_b}_frac', (1.0 - share_a) * total)
        self.ctrl['boost'] = mode_a if target > frac_a / total else mode_b

    def _nudge_mode_fracs(self, boost):
        """EMA nudge of the fracs toward a target split, with the
        min_mode_frac floor -- the mechanics both old controllers shared,
        generalized from a one-hot: `boost` is a mode name (one-hot target,
        every rule) or a normalized {mode: weight} mix (idle default)."""
        m = self.m
        ctrl = m.args.controller
        probs = np.array([m.fwd_frac, m.bwd_frac, m.replay_frac], dtype=float)
        probs /= probs.sum()
        weights = boost if isinstance(boost, dict) else {boost: 1.0}
        target = np.array([weights.get('fwd', 0.0), weights.get('bwd', 0.0),
                           weights.get('replay', 0.0)], dtype=float)
        m_floor = ctrl.min_mode_frac  # requires m_floor < 1/3
        free = 1.0 - 3.0 * m_floor
        excess = np.clip(probs - m_floor, 0.0, None)
        s = excess.sum()
        excess = excess * (free / s) if s > 0.0 else np.full(3, free / 3.0)
        excess = (1.0 - ctrl.beta) * excess + ctrl.beta * free * target
        m.fwd_frac, m.bwd_frac, m.replay_frac = m_floor + excess

    # ---------------------------------------------------------------- logging

    def report(self) -> dict:
        """Loggable (numeric-only) view of the engine: chosen boost, per-rule
        live thresholds and elevations, exit-term streaks. The stage itself is
        logged as metrics['phase'] (its 1-based index) for wandb continuity."""
        stage = self.stage
        out = {}
        boost = self.ctrl.get('boost')
        if boost is not None:
            out['protocol/boost'] = {'fwd': 0, 'bwd': 1, 'replay': 2}[boost]
        out['protocol/anneal_streak'] = self.ctrl.get('anneal_streak', 0)
        if stage.balance is not None and stage.balance['kind'] == 'lexicographic':
            for i, rule in enumerate(stage.balance['rules']):
                rs = self.ctrl['rules'].get(i, {})
                tag = rule['metric'].replace('/', '_')
                # every rule reports its bar: live annealed threshold for
                # 'above' rules, the static floor for 'below' rules, and the
                # margin (the line elev_* is judged against) for relative ones
                if 'thr' in rs:
                    out[f'protocol/thr_{tag}'] = rs['thr']
                elif 'below' in rule:
                    out[f'protocol/thr_{tag}'] = float(rule['below'])
                elif 'relative' in rule:
                    out[f'protocol/thr_{tag}'] = float(rule.get('margin', 1.3))
                if 'elevation' in rs:
                    out[f'protocol/elev_{tag}'] = rs['elevation']
                if rs.get('floor', {}).get('floored'):
                    out[f'protocol/floored_{tag}'] = 1.0
        if stage.exit:
            for i, term in enumerate(stage.exit):
                tag = term['metric'].replace('/', '_')
                out[f'protocol/exit_streak_{tag}'] = self.ctrl['exit'].get(i, 0)
        return out
