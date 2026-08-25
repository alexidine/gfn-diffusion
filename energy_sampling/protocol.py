"""
The unified stage interface: one declarative, config-level description of the
training protocol. It replaced three separate controllers -- a phase machine
(1/2/3), a forward-first A/B switcher, and a mode-balance nudger -- whose
interactions were the thing nobody could predict. All three are deleted; this
engine is the only thing that moves a run between regimes.

A protocol is an ordered list of STAGES (config: protocol.stages). Each stage
declares:

  name              unique identifier; checkpoints store it as the run's position
  train_mode        'bwd' | 'fused' -- what train_logic returns every step
  bwd_sampling_mode 'dataset' | 'prior' -- where backward draws terminals from
  flags             explicit behavior switches read by train.py (update_log_z,
                    scramble_conditions, weighted_condition_sampling,
                    buffers_active, mle_gate, weighted_bwd_sampling) -- the
                    replacements for the old `self.phase == N` integer checks.
                    STAGE_FLAGS below is the authoritative list
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
                    bootstrap_z, seed_prior_from_anchors,
                    reseed_prior_from_dataset, rebuild_prior_by_churn,
                    set_lr_flow:<float>; set_max_batch_size:<int>;
                    set_traj_checkpoint:<0|1>; ACTIONS below is the authoritative
                    list) -- the route-specific physics; everything generic
                    (optimizer rebuild, monitor cooldown, LR re-warm) happens
                    automatically at EVERY transition.
                    NB the first stage is never ENTERED via a transition, so its
                    on_enter does not fire -- put entry actions on the stage
                    being transitioned INTO
  skip_if           entry condition ('prior_loaded'): on a fresh run the stage
                    is skipped when the condition holds (e.g. the MLE warm-
                    start is redundant when a prior model was loaded by path
                    given by prior_model_name)

Balance rules (kind: lexicographic) walk in order; the FIRST violated rule's
`boost` (a mode name, or a {mode: weight} mix -- same form as default_boost)
gets the frac nudge this tick -- an EMA step of `controller.beta` toward the
boosted mix, so a rule that stays violated compounds while a rule that flickers
averages out; `default_boost`
takes it when all rules are clean, and a clean streak of
controller.anneal_patience ticks tightens every annealed rule's threshold.
A rule is either absolute (`above: X`, annealable) or relative to its own
running best (`relative: best, margin: M` -- "is this metric DEGRADING",
never "is it below an absolute bar"; the calibration floor legitimately
rises as coverage grows, so absolute bars deadlock).
kind: proportional instead splits two modes' combined mass proportionally to
a pair of lagging-spread metrics (the old phase-2 balancer, generalized).
kind: constraint treats the same two modes ASYMMETRICALLY -- one side's metric
is a bar that must hold, the other's is a best-effort objective -- and drives
the split with an INTEGRATOR rather than a static map (see _constraint_tick).
kind: ratio holds the RATIO of the two split modes' metrics at `setpoint`,
integrating in the logit of the numerator mode's share and pinning any third
mode's frac. It declares an exchange rate between two disjoint halves of one
residual field instead of a bar per side, so neither drive can clamp to zero
and go one-sided (see _ratio_tick).

A stage may also declare `buffer_servo`: a second, independent controller whose
actuator is the replay buffer's freshness rather than the loss weights (see
_buffer_servo_tick). It exists because branch weights cannot fix replay
OVERFITTING -- down-weighting a memorized buffer trains less on it but does not
make it less memorized -- so the two pathologies need two actuators.

The same clean-streak anneal event can also RAMP UP energy_config
coefficients (balance.anneal_coeffs: {bounding_coeff: {target: 10.0}, ...})
by controller.decay_rate (dividing, not multiplying -- the mirror image of
the rule-threshold tightening above). The base energy_config value is the
SOFT one -- it's what the energy function is constructed with, so it holds
for every stage from run start, including stages that name nothing here;
only once a stage's own anneal events fire does its live value climb off
that base toward its `target`. Since that only happens once every rule
above has gone quiet for anneal_patience ticks, it is lexicographically
LAST -- the boundary/reduction penalty only firms back up to full strength
once that stage's whole balance has converged, never in place of chasing an
active violation.

Frac floors are EXPLICIT per stage: min_fracs {mode: floor} (fallback
controller.min_mode_frac) and an optional per-stage deactivate_threshold. A
floor at or above the deactivate threshold means that branch is always
computed; below it, the mode can switch off entirely. THE RULE EXISTS
BECAUSE THAT WENT UNNOTICED ONCE: a run sat with bwd below the global
deactivate threshold for ~1700 steps of its terminal stage -- zero backward
gradients, nothing saying so -- because the floor/deactivate relationship was
implicit. Each stage now states which modes may go dark. Mode dormancy for
force-refresh purposes stays DERIVED: a mode no rule (and no default_boost)
ever boosts is dormant -- the fused step skips even its force-refresh
rollout (the old bwd_dormant, generalized).

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

`patience` COUNTS MEASUREMENTS, NOT CHECKS -- a term's streak only moves on a
tick where its metric was freshly written (see _advance_term). Every source
above persists its last value, so counting checks counted one sample many
times: `patience: 5` on a metric written every 500 steps was cleared by a
single clean write. A term is therefore denominated in its OWN metric's
cadence, and a patience of N on an eval/* metric costs N * eval_period train
steps where the same N on a tick metric costs N * 10. config_invariants'
`exit_patience_is_reachable` states that relation at load time.

All mutable engine state (rule bests / streaks, live annealed
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
STAGE_FLAGS = ('update_log_z', 'scramble_conditions', 'weighted_condition_sampling',
               'buffers_active', 'weighted_bwd_sampling', 'z_calibration')

# `mle_gate` is a BLOCK, not a flag. It was a flag on the stage while its three
# parameters sat at top level, which put the switch and the settings in different
# places -- and the parameters read as global while only one stage ever used
# them. Presence of the block is the switch; its contents are the parameters.
MLE_GATE_DEFAULTS = {
    'slope_t': 2.0,      # confidence multiplier on the descent rate's standard error
    'min_rate': 0.05,    # negligible descent rate, nats per 100 train steps
    'window': 300,       # train steps of gate samples the slope is regressed over
}
ACTIONS = ('snapshot', 'snapshot_prior', 'bootstrap_z', 'seed_prior_from_anchors',
           'reseed_prior_from_dataset', 'rebuild_prior_by_churn', 'set_lr_flow',
           'set_lr_policy', 'set_max_batch_size', 'set_traj_checkpoint')
SKIP_CONDITIONS = ('prior_loaded',)

# Per-stage LR sensor kinds -- see Stage._parse_lr_sensor for why this is
# declared rather than derived from the active loss coefficients.
# DIAGNOSTIC ONLY since the LR bracket took over actuation (controller.py).
# `plateau` is gone entirely -- it was a pure actuator with nothing to report.
# `ray` and `hyper` are retained, OFF unless a stage names one, and neither
# reaches a learning rate: alpha* was measured uncorrelated with the rate it
# steered, and cos is a stationarity statistic with no fixed point. No canonical
# config declares either.
LR_SENSOR_KINDS = ('ray', 'hyper', 'none')

#: `hot_lr_sensor.action`. ONE value, deliberately -- see Stage._parse_hot_lr_sensor.
HOT_LR_ACTIONS = ('report',)
#: `hot_lr_sensor.form`. `absolute` is required on a channel that crosses zero
#: (`bwd/mle` runs +9.75 to -33.74); `log` is the log ratio to the floor.
HOT_LR_FORMS = ('log', 'absolute')
RULE_KEYS = {'metric', 'boost', 'above', 'below', 'relative', 'margin', 'drift', 'floor',
             'abs', 'if_missing', 'lookahead', 'anneal'}
TERM_KEYS = {'metric', 'above', 'below', 'abs', 'patience'}


def fresh_stage_ctrl():
    """The per-stage mutable engine state, reset at every transition and
    checkpointed via MODELLER_STATE_DEFAULTS['stage_ctrl']."""
    return {
        'gates': {},        # gate-published values, e.g. {'mle_flat': 1.0}
        'gate_state': {},   # gate internals, e.g. the MLE slope gate's window
        'rules': {},        # rule index -> {'best', 'thr', 'look'}
        'coeffs': {},       # anneal_coeffs name -> {'val': live value}
        'exit': {},         # exit term index -> consecutive-pass streak
        # exit term index -> the write-stamp of the last value that streak
        # ACTUALLY JUDGED. Without it a streak cannot tell a new measurement
        # from the same one read again (see _advance_term).
        'exit_seen': {},
        # gate name -> step of its last publish, the gates/* half of the same
        # question. ctrl['gates'] alone is a stale read, exactly like the
        # metric tracker's.
        'gate_written': {},
        'anneal_streak': 0,
        'boost': None,      # last chosen boost mode (logging)
        'exit_armed': False,
        'request_eval': False,
        # proportional balance: one multiplicative scale over the WHOLE target
        # vector (so every target ratio is preserved exactly by construction),
        # plus its both-satisfied streak. See _proportional_tick.
        'prop_scale': 1.0,
        'prop_streak': 0,
        # constraint balance: the integrator state IS the actuator -- the
        # logit of the best-effort mode's share of the split pair. None = not
        # yet seeded (first tick seeds it from the stage's entry fracs, so the
        # controller starts exactly where a fixed-mix arm would sit).
        'cs_theta': None,
        # buffer freshness servo: log of the multiplicative churn/residence
        # boost. 0.0 = the configured buffer, i.e. inert.
        'bs_log_boost': 0.0,
    }


class Stage:
    """Parsed + validated view of one config stage dict. Pure data; all the
    engine state lives in modeller.stage_ctrl."""

    def __init__(self, spec: dict, index: int):
        if not isinstance(spec, dict):
            raise TypeError(f"protocol.stages[{index}] must be a mapping, got {type(spec)}")
        unknown = set(spec) - {'name', 'train_mode', 'bwd_sampling_mode', 'flags',
                               'loss_coeffs', 'fracs', 'min_fracs',
                               'deactivate_threshold', 'balance', 'buffer_servo',
                               'lr_sensor', 'exit', 'on_exit', 'on_enter', 'skip_if',
                               'mle_gate', 'hot_lr_sensor'}
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

        # explicit per-stage frac floors and branch-deactivation threshold.
        # min_fracs: {mode: floor} -- unspecified modes fall back to
        # controller.min_mode_frac. A floor AT OR ABOVE the deactivate
        # threshold keeps that mode's branch always computed (never skipped by
        # fused_train_step); a floor below it lets the mode go truly dormant.
        # Each stage states its intent explicitly -- nothing is derived.
        # Binds under EVERY balance kind: the lexicographic nudge reads it
        # directly (_nudge_mode_fracs) and the two integrator kinds fold it into
        # their `bounds` at parse (_parse_bounds), which is also where an
        # inconsistent pair of declarations fails.
        self.min_fracs = dict(spec.get('min_fracs') or {})
        bad = set(self.min_fracs) - set(MODES)
        if bad:
            raise ValueError(f"stage '{self.name}': min_fracs for unknown modes {sorted(bad)}")
        for mode, v in self.min_fracs.items():
            if not isinstance(v, (int, float)) or not 0.0 <= v < 1.0 / 3:
                raise ValueError(f"stage '{self.name}': min_fracs.{mode} must be in [0, 1/3), got {v}")
        if sum(self.min_fracs.values()) >= 1.0:
            raise ValueError(f"stage '{self.name}': min_fracs sum to >= 1")
        self.deactivate_threshold = spec.get('deactivate_threshold')
        if self.deactivate_threshold is not None:
            v = self.deactivate_threshold
            if not isinstance(v, (int, float)) or not 0.0 <= v < 1.0 / 3:
                raise ValueError(f"stage '{self.name}': deactivate_threshold must be in [0, 1/3), got {v}")
            self.deactivate_threshold = float(v)

        self.balance = self._parse_balance(spec.get('balance'))
        self.buffer_servo = self._parse_buffer_servo(spec.get('buffer_servo'))
        self.lr_sensor = self._parse_lr_sensor(spec.get('lr_sensor'))
        self.mle_gate = self._parse_mle_gate(spec.get('mle_gate'))
        self.hot_lr_sensor = self._parse_hot_lr_sensor(spec.get('hot_lr_sensor'))
        self.exit = self._parse_exit(spec.get('exit'))
        self.on_exit = self._parse_actions(spec.get('on_exit'), 'on_exit')
        self.on_enter = self._parse_actions(spec.get('on_enter'), 'on_enter')

        self.skip_if = spec.get('skip_if')
        if self.skip_if is not None and self.skip_if not in SKIP_CONDITIONS:
            raise ValueError(f"stage '{self.name}': skip_if must be one of {SKIP_CONDITIONS}")

    # ------------------------------------------------------------ sub-parsers

    def _normalize_boost(self, raw, where):
        """A boost value: a single mode name, or a {mode: weight} mix (positive
        weights, normalized to sum 1) -- shared by rule 'boost' and
        default_boost, so a rule can cap its target at a fixed mixed share
        instead of one-hotting a mode to 100%."""
        if isinstance(raw, dict):
            bad = set(raw) - set(MODES)
            if bad or not raw:
                raise ValueError(f"stage '{self.name}': {where} mix has unknown/empty modes {sorted(bad)}")
            if any(not isinstance(w, (int, float)) or w <= 0 for w in raw.values()):
                raise ValueError(f"stage '{self.name}': {where} mix weights must be positive")
            total = float(sum(raw.values()))
            return {m: w / total for m, w in raw.items()}
        if raw not in MODES:
            raise ValueError(f"stage '{self.name}': {where} must be one of {MODES} or a {{mode: weight}} mix")
        return raw

    def _parse_pinned(self, node, metrics):
        """Modes HELD at a fixed frac, outside a two-mode split (shared by
        kind: proportional and kind: constraint). Declaring them is mandatory
        rather than implied by absence from 'metrics': a split only rewrites
        its own two modes, so a third mode's pin was previously invisible in
        the config and readable only from the implementation. Spelled out here,
        it is also re-asserted every tick, so the pin is enforced rather than
        merely emergent."""
        pinned = dict(node.get('pinned') or {})
        bad = set(pinned) - set(MODES)
        if bad:
            raise ValueError(f"stage '{self.name}': pinned has unknown modes {sorted(bad)}")
        overlap = set(pinned) & set(metrics)
        if overlap:
            raise ValueError(f"stage '{self.name}': modes {sorted(overlap)} are both "
                             f"'pinned' and in 'metrics' -- a mode is either held or split")
        for mode, v in pinned.items():
            if not isinstance(v, (int, float)) or not 0.0 <= v < 1.0:
                raise ValueError(f"stage '{self.name}': pinned.{mode} must be in [0, 1), got {v}")
            entry = self.fracs.get(mode)
            if entry is not None and abs(float(entry) - float(v)) > 1e-9:
                raise ValueError(
                    f"stage '{self.name}': pinned.{mode} ({v}) disagrees with "
                    f"fracs.{mode} ({entry}) -- they are the same quantity")
        # nothing carrying entry mass may be left unmanaged: a mode that is
        # neither split nor pinned would silently sit wherever it landed
        unmanaged = {mode for mode, v in self.fracs.items()
                     if isinstance(v, (int, float)) and v > 0} - set(metrics) - set(pinned)
        if unmanaged:
            raise ValueError(
                f"stage '{self.name}': modes {sorted(unmanaged)} have nonzero entry fracs but "
                f"are neither in 'metrics' (split) nor 'pinned' (held) -- declare them")
        return pinned

    def _parse_bounds(self, node, metrics, pinned, kind):
        """ABSOLUTE [lo, hi] frac bounds on a two-mode split's own modes,
        shared by kind: constraint and kind: ratio. These are the load-bearing
        safety element, not a nicety: they are what makes a MISCALIBRATED bar
        or setpoint degrade to a fixed mix instead of to a collapse. An
        integrator whose drive never reaches zero walks to a bound and parks --
        so the bound is the mix the run actually gets, and it must be one that
        is safe to run forever. (replay_july26: bwd_frac 0.001 dropped EffDim
        5.99 -> 1.3 inside one eval window.)"""
        bounds = dict(node.get('bounds') or {})
        bad = set(bounds) - set(metrics)
        if bad:
            raise ValueError(f"stage '{self.name}': {kind} bounds for modes "
                             f"{sorted(bad)} that aren't in 'metrics'")
        if not bounds:
            raise ValueError(f"stage '{self.name}': {kind} balance needs 'bounds' on at "
                             f"least one split mode -- an unbounded integrator against a "
                             f"target it cannot reach walks to a degenerate mix")
        pair_mass = 1.0 - sum(float(v) for v in (pinned or {}).values())
        # min_fracs BINDS UNDER EVERY KIND. It used to be read only by
        # _nudge_mode_fracs (the lexicographic path), so a stage declaring both
        # min_fracs and an integrator balance had its floors silently ignored
        # by the integrator -- the floor was there, it just wasn't a floor.
        # Folding it in here rather than adding a check in each tick keeps ONE
        # live bound per mode (R3: three frac bounds in three files is already
        # the confusing part) and makes the two declarations either agree or
        # fail loudly at parse.
        for mode in metrics:
            floor = self.min_fracs.get(mode)
            if floor is None:
                continue
            if mode not in bounds:
                bounds[mode] = [float(floor), pair_mass]
            elif float(bounds[mode][0]) < float(floor):
                raise ValueError(
                    f"stage '{self.name}': bounds.{mode} lower bound {bounds[mode][0]} is "
                    f"below min_fracs.{mode} ({floor}). They are the same quantity -- a "
                    f"floor on this mode's frac -- so declaring them inconsistently means "
                    f"one of them is not doing what it says.")
        for mode, v in (pinned or {}).items():
            floor = self.min_fracs.get(mode)
            if floor is not None and float(v) < float(floor):
                raise ValueError(
                    f"stage '{self.name}': pinned.{mode} ({v}) is below min_fracs.{mode} "
                    f"({floor}) -- the pin would hold the mode under its own floor")
        for mode, lohi in bounds.items():
            if (not isinstance(lohi, (list, tuple)) or len(lohi) != 2
                    or not all(isinstance(v, (int, float)) for v in lohi)):
                raise ValueError(f"stage '{self.name}': bounds.{mode} must be [lo, hi], got {lohi}")
            lo, hi = float(lohi[0]), float(lohi[1])
            if not 0.0 < lo <= hi < pair_mass + 1e-9:
                raise ValueError(
                    f"stage '{self.name}': bounds.{mode} = [{lo}, {hi}] must satisfy "
                    f"0 < lo <= hi <= {pair_mass:.4f} (the split pair's total mass)")
            # a floor UNDER the deactivate threshold means the branch can
            # switch off entirely while the controller still believes it is
            # steering that mode -- the silent-dark-branch failure this
            # rule exists to make unrepresentable
            if self.deactivate_threshold is not None and lo < self.deactivate_threshold:
                raise ValueError(
                    f"stage '{self.name}': bounds.{mode} lower bound {lo} is below the "
                    f"stage's deactivate_threshold ({self.deactivate_threshold}) -- the "
                    f"branch would go dark while the controller still steers it")
            bounds[mode] = [lo, hi]
        return bounds

    #: Everything a `hot_lr_sensor` block may contain. CLOSED: an unknown key
    #: here would be a threshold the run silently does not apply.
    _HOT_KEYS = frozenset({'action', 'channel', 'form', 'rows', 'above',
                           'floor_percentile', 'row_steps'})

    def _parse_hot_lr_sensor(self, node):
        """The stage's "the LR is hot" drawdown sensor. REPORT-ONLY; moves nothing.

        Fills a measured gap: `lr_ctrl/divergences` was 0 in all 97 stage segments
        examined, including all 19 failures, so the hard tripwire does not fire on
        this failure mode at all. The sensor compares the channel's current level
        against the 10th percentile of a short trailing window and fires on a
        single row. It says "destabilising", never "the LR caused it".

        THRESHOLDS ARE PER-STAGE CALIBRATION DATA, like the grad-clip guard's
        quantile. They are not derivable at runtime and they do not transfer
        between stages -- the equilibration bar would fire on most healthy
        `var_conditioning` runs. A stage that wants a sensor states its own
        numbers; a stage that omits the block has none.

        The three validated sensors:

            hot_lr_sensor:                 # train_prior
              channel: bwd/mle
              form: absolute               # bwd/mle crosses zero
              rows: 31                     # 300 steps at a 10-step cadence
              above: 5.0

            hot_lr_sensor:                 # equilibration
              channel: fwd/scatter_err
              rows: 11                     # 100 steps
              above: 2.0

            hot_lr_sensor:                 # var_conditioning
              channel: fwd/vg_lb           # FORWARD branch: bwd/vg_lb has no
              rows: 11                     # clean band under any statistic tested
              above: 3.0

        `rows` is W, the window length the thresholds were fitted at, and it is
        the primary quantity rather than a step count: the statistic is over rows,
        and the lookback in steps follows as `(rows - 1) * row_steps`.
        """
        if node is None:
            return None
        if not isinstance(node, dict):
            raise TypeError(f"stage '{self.name}': hot_lr_sensor must be a mapping, "
                            f"got {type(node)}")
        bad = set(node) - self._HOT_KEYS
        if bad:
            raise ValueError(f"stage '{self.name}': unknown hot_lr_sensor keys "
                             f"{sorted(bad)} (known: {sorted(self._HOT_KEYS)})")

        out = {'action': node.get('action', 'report'),
               'form': node.get('form', 'log'),
               'row_steps': int(node.get('row_steps', 10)),
               'floor_percentile': float(node.get('floor_percentile', 10.0))}

        # ACTION IS A CLOSED VOCABULARY OF ONE. Report-only is not a default to
        # be overridden -- making this actuate is a code change with a review,
        # not a config edit. `ray_calibration.enabled` is why: a second switch
        # that could disagree with the thing it switched.
        if out['action'] not in HOT_LR_ACTIONS:
            raise ValueError(
                f"stage '{self.name}': hot_lr_sensor.action must be one of "
                f"{list(HOT_LR_ACTIONS)}, got {out['action']!r}. This sensor is "
                f"report-only: nothing downstream can move a learning rate from "
                f"it, so any other value would describe behaviour that does not "
                f"exist.")
        if out['form'] not in HOT_LR_FORMS:
            raise ValueError(
                f"stage '{self.name}': hot_lr_sensor.form must be one of "
                f"{list(HOT_LR_FORMS)}, got {out['form']!r}. Use 'absolute' on a "
                f"channel that crosses zero -- the log ratio is undefined there.")

        channel = node.get('channel')
        if not isinstance(channel, str) or channel.partition('/')[0] not in MODES:
            raise ValueError(
                f"stage '{self.name}': hot_lr_sensor.channel must be "
                f"'<mode>/<channel>' with mode in {MODES}, got {channel!r}. The "
                f"branch is part of the calibration: `bwd/vg_lb` has no clean band "
                f"under any statistic tested, so a config that generalises a "
                f"channel name across branches is a bug.")
        out['channel'] = channel

        out['rows'] = int(node.get('rows', 0))
        if out['rows'] < 3:
            raise ValueError(
                f"stage '{self.name}': hot_lr_sensor.rows must be >= 3, got "
                f"{out['rows']}. The floor is a percentile over the window, and "
                f"below three rows it degenerates to the minimum -- which has zero "
                f"breakdown, so one spuriously low row would inflate every "
                f"subsequent reading for a full lookback.")
        if out['row_steps'] < 1:
            raise ValueError(f"stage '{self.name}': hot_lr_sensor.row_steps must be "
                             f">= 1, got {out['row_steps']}")

        above = node.get('above')
        out['above'] = float('nan') if above is None else float(above)
        if not math.isfinite(out['above']) or out['above'] <= 0:
            raise ValueError(
                f"stage '{self.name}': hot_lr_sensor.above must be a positive "
                f"finite number, got {above!r}. A sensor with no threshold cannot "
                f"fire, and one that cannot fire reports a clean run "
                f"indistinguishably from one that is not running.")
        if not 0.0 <= out['floor_percentile'] < 50.0:
            raise ValueError(
                f"stage '{self.name}': hot_lr_sensor.floor_percentile must lie in "
                f"[0, 50), got {out['floor_percentile']}. It names the FLOOR of the "
                f"window; at or above the median it is not a floor.")
        return out

    def _parse_mle_gate(self, node):
        """The MLE descent gate's parameters, or None if this stage has no gate.

        PRESENCE IS THE SWITCH. This used to be `flags: {mle_gate: true}` with
        slope_t / min_rate / window living at config top level -- so the switch
        and the settings sat in different places, and the settings read as global
        while only the one MLE stage ever consulted them. A stage that publishes
        the gate now carries the numbers that shape it.

        `{}` is legal and means "gate on, defaults" -- an empty block still
        declares intent, which a missing one does not."""
        if node is None:
            return None
        if not isinstance(node, dict):
            raise TypeError(f"stage '{self.name}': mle_gate must be a mapping, "
                            f"got {type(node)}")
        bad = set(node) - set(MLE_GATE_DEFAULTS)
        if bad:
            raise ValueError(f"stage '{self.name}': unknown mle_gate keys "
                             f"{sorted(bad)} (known: {sorted(MLE_GATE_DEFAULTS)})")
        out = dict(MLE_GATE_DEFAULTS)
        out.update(node)
        for k in ('slope_t', 'min_rate'):
            if not (isinstance(out[k], (int, float)) and float(out[k]) > 0):
                raise ValueError(f"stage '{self.name}': mle_gate.{k} must be a "
                                 f"positive number, got {out[k]!r}")
        # The gate samples every 10 steps and the regression needs at least a few
        # points, so a window under 40 steps yields fewer than 4 and the slope is
        # noise fitted to noise.
        if not (isinstance(out['window'], int) and out['window'] >= 40):
            raise ValueError(f"stage '{self.name}': mle_gate.window must be an "
                             f"int >= 40 (it is sampled every 10 steps and the "
                             f"regression needs >= 4 points), got {out['window']!r}")
        return out

    def _parse_lr_sensor(self, node):
        """
        An OPTIONAL, OFF-BY-DEFAULT DIAGNOSTIC this stage may run. It reaches no
        learning rate.

        THIS BLOCK NO LONGER STEERS ANYTHING. Learning rates are set by the
        brute-force bracket (controller.py, lr_bracket.py), which is a run-level
        mechanism rather than a per-stage one: burn in, checkpoint, trial a fixed
        grid, promote a rung a safety margin below the lowest failure, hold. Both
        sensors below survive only as instruments, so a future claim about either
        can be measured rather than argued about, and no canonical config
        declares one.

          kind: ray       the alpha* ray calibration -- scores the fused
                          composite this stage's step descends, on batches
                          harvested from its own live steps (lr_larder.py).
                          RETIRED AS AN ACTUATOR 2026-08-23 by its own acceptance
                          test: alpha* is s*/lr, so the slope of log(alpha*)
                          against log(lr) must be -1, and it measured 0.00 +- 0.2
                          across twelve runs, two stages and 2.7 decades. It is
                          also not free -- about 4.8% of step time at period 500
                          on `train_prior` -- so declaring it is a deliberate
                          purchase. Optional `period` / `n_sub` override the
                          global values for this stage.
          kind: hyper     the hypergradient cosine between the current gradient
                          and the direction the previous step moved the policy.
                          Retired as an actuator for a different reason: cos is a
                          STATIONARITY statistic, negative at every stable rate
                          once the iterate has equilibrated, so it has no fixed
                          point to steer to. `beta` is still required, because a
                          recorded bandwidth that was never chosen is not a
                          record of anything.
          kind: none      this stage deliberately runs no diagnostic.

        `none` is spelled out rather than left to omission, so "no sensor" is a
        decision in the config and not an oversight. Omitting the block means the
        same thing -- and, since nothing here actuates, omission is now the
        correct default rather than a trap.
        """
        if node is None:
            return None
        if not isinstance(node, dict):
            raise TypeError(f"stage '{self.name}': lr_sensor must be a mapping, got {type(node)}")
        node = dict(node)
        kind = node.get('kind')
        if kind not in LR_SENSOR_KINDS:
            raise ValueError(f"stage '{self.name}': lr_sensor.kind must be one of "
                             f"{LR_SENSOR_KINDS}, got {kind!r}")
        if kind == 'hyper':
            # beta is REQUIRED. It is a bandwidth, not a safe constant: swept on
            # the bench across 12 cells the per-cell optimum spanned 20x and the
            # best worst-case setting was still 3.2x the best fixed rate. A
            # default here would be a universal claim the measurements do not
            # support.
            if 'beta' not in node:
                raise ValueError(f"stage '{self.name}': lr_sensor kind 'hyper' "
                                 f"requires an explicit 'beta' -- it is a "
                                 f"bandwidth, and no value is right for every "
                                 f"stage")
            bad = set(node) - {'kind', 'beta', 'beta_down', 'every', 'cos_target'}
            if bad:
                raise ValueError(f"stage '{self.name}': unknown lr_sensor keys "
                                 f"for kind 'hyper': {sorted(bad)}")
            # cos_target is hyper's analogue of calibration.alpha_target, and it
            # exists for the same reason. The update's fixed point is cos == this,
            # so 0.0 (the default) parks the rate at the ONE-STEP optimum -- which
            # adaptive_lr.calibration's own comment says "the rate a run survives
            # sits well BELOW", because a local probe cannot see the gradient-noise
            # term. `ray` carries that margin explicitly as alpha_target: 4.0;
            # hyper had none. A POSITIVE value holds the rate under the greedy
            # optimum by steering to "still mildly under-stepped".
            ct = node.get('cos_target', 0.0)
            if not isinstance(ct, (int, float)) or not (-1.0 < float(ct) < 1.0):
                raise ValueError(f"stage '{self.name}': lr_sensor.cos_target must "
                                 f"be a number in (-1, 1) -- it is compared against "
                                 f"a cosine -- got {ct!r}")
            node['cos_target'] = float(ct)
            for k in ('beta', 'beta_down'):
                if k in node and not (isinstance(node[k], (int, float))
                                      and float(node[k]) > 0):
                    raise ValueError(f"stage '{self.name}': lr_sensor.{k} must "
                                     f"be a positive number, got {node[k]!r}")
            node['beta'] = float(node['beta'])
            if node.get('beta_down') is not None:
                node['beta_down'] = float(node['beta_down'])
            node['every'] = int(node.get('every', 1))
            if node['every'] < 1:
                raise ValueError(f"stage '{self.name}': lr_sensor.every must be "
                                 f">= 1, got {node['every']}")
            return node
        if kind == 'none':
            bad = set(node) - {'kind'}
            if bad:
                raise ValueError(f"stage '{self.name}': lr_sensor kind 'none' takes no "
                                 f"other keys, got {sorted(bad)}")
            return {'kind': 'none'}
        if kind == 'ray':
            # `period` and `n_sub` MAY be overridden here, and the reason is a
            # measurement. The probe's absolute cost is n_sub x len(alphas)
            # forward passes over one batch; its OVERHEAD is that divided by the
            # stage's step cost, and stages differ by more than an order of
            # magnitude. Measured on elj/mipcas: ~24 training steps per
            # calibration on `train_prior` (a bwd/dataset step runs no rollout
            # and no energy call, median 0.158 s) = 4.8% at period 500, against
            # the 1.2% recorded for the same probe on the fused stage. One
            # global period cannot serve both. Absent = the global value.
            bad = set(node) - {'kind', 'period', 'n_sub'}
            if bad:
                raise ValueError(f"stage '{self.name}': lr_sensor kind 'ray' takes only "
                                 f"'period' and 'n_sub', got {sorted(bad)}")
            out = {'kind': 'ray'}
            if 'period' in node:
                period = int(node['period'])
                if period < 1 or period % 10:
                    # Same rule RayCalibration enforces: metrics drain on a
                    # 10-step clock, so a period that is not a multiple of it
                    # aliases and some calibrations never reach the log.
                    raise ValueError(f"stage '{self.name}': lr_sensor.period must be a "
                                     f"positive multiple of 10, got {node['period']!r}")
                out['period'] = period
            if 'n_sub' in node:
                n_sub = int(node['n_sub'])
                if n_sub < 2:
                    raise ValueError(f"stage '{self.name}': lr_sensor.n_sub must be >= 2 "
                                     f"-- they are the replicates every confidence "
                                     f"interval is built from -- got {node['n_sub']!r}")
                out['n_sub'] = n_sub
            return out

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
                r['boost'] = self._normalize_boost(r.get('boost'), f'rule {i} boost')
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
            # idle MIX: when all rules are clean (or, for a rule's own boost, when
            # that rule fires), nudge toward this frac split instead of a one-hot
            # (e.g. {replay: 0.9, fwd: 0.1} -- keep the fit clean AND hold Z gently
            # at its fixed point, rather than letting fwd decay below the
            # deactivate threshold and only revisiting Z after zerr has drifted
            # over its bar; a rule can use the same form to cap ITS boosted mode's
            # share instead of letting a persistent violation pull it to 100%)
            node['default_boost'] = self._normalize_boost(node.get('default_boost'), 'default_boost')
            node['rules'] = [dict(r) for r in rules]
            # coefficients (e.g. energy_config.bounding_coeff/reduction_coeff)
            # RAMPED UP from the base energy_config value (kept LOW there so
            # it's soft for every stage, from run start) toward 'target' once
            # THIS stage's balance runs clean -- the SAME anneal event as the
            # rule thresholds above, so strictly lower priority than
            # (lexicographically after) all of them: the boundary/reduction
            # penalty only firms up to full strength once this stage is fully
            # converged, never at the cost of an active violation. Validated
            # against actual energy_config attributes in
            # StageProtocol.energy_coeffs, since the set of numeric
            # coefficients is config-defined, not fixed here.
            anneal_coeffs = dict(node.get('anneal_coeffs') or {})
            for name, spec in anneal_coeffs.items():
                if not isinstance(spec, dict) or 'target' not in spec:
                    raise ValueError(f"stage '{self.name}': anneal_coeffs.{name} needs a 'target'")
                bad = set(spec) - {'target', 'rate'}
                if bad:
                    raise ValueError(f"stage '{self.name}': anneal_coeffs.{name} "
                                     f"unknown keys {sorted(bad)}")
            node['anneal_coeffs'] = anneal_coeffs
        elif kind == 'proportional':
            if node.get('anneal_coeffs'):
                raise ValueError(f"stage '{self.name}': anneal_coeffs needs kind: lexicographic "
                                 f"(it anneals off the lexicographic clean-streak event)")
            metrics = node.get('metrics') or {}
            if len(metrics) != 2 or set(metrics) - set(MODES):
                raise ValueError(f"stage '{self.name}': proportional balance needs a "
                                 f"'metrics' mapping of exactly two modes to metric names")
            node['metrics'] = dict(metrics)
            pinned = self._parse_pinned(node, metrics)
            node['pinned'] = pinned
            # optional anneal of the TARGET VECTOR: a single multiplicative
            # scale decaying toward min_scale, applied to every target. One
            # scale rather than per-target rates so the ratios between targets
            # (which encode the priority trade) are preserved exactly -- a
            # per-target anneal would silently rewrite that trade as the
            # targets hit their own floors at different times. Fires only when
            # BOTH sides have been at/under their live targets for `patience`
            # consecutive ticks, i.e. the same "nothing left to arbitrate"
            # event that would otherwise be a stage exit -- tighten and keep
            # going instead of handing over. min_scale bounds it: targets can
            # never anneal below min_scale x their configured value, which is
            # what keeps them from dropping under the metric's irreducible
            # floor and leaving that side with permanent drive.
            anneal = node.get('anneal')
            if anneal is not None:
                bad = set(anneal) - {'rate', 'patience', 'min_scale'}
                if bad:
                    raise ValueError(f"stage '{self.name}': proportional anneal unknown keys {sorted(bad)}")
                rate = float(anneal.get('rate', 0.98))
                if not 0.0 < rate < 1.0:
                    raise ValueError(f"stage '{self.name}': proportional anneal.rate must be in (0, 1), got {rate}")
                min_scale = float(anneal.get('min_scale', 0.25))
                if not 0.0 < min_scale <= 1.0:
                    raise ValueError(f"stage '{self.name}': proportional anneal.min_scale "
                                     f"must be in (0, 1], got {min_scale}")
                patience = int(anneal.get('patience', 20))
                if patience < 1:
                    raise ValueError(f"stage '{self.name}': proportional anneal.patience must be >= 1")
                anneal = {'rate': rate, 'patience': patience, 'min_scale': min_scale}
            node['anneal'] = anneal
            # per-mode SOFT reference levels subtracted before the split (see
            # _proportional_tick). Not gates: crossing one only zeroes that
            # side's contribution, it never switches control -- which is the
            # whole difference from a lexicographic threshold.
            targets = dict(node.get('targets') or {})
            bad = set(targets) - set(metrics)
            if bad:
                raise ValueError(f"stage '{self.name}': proportional targets for modes "
                                 f"{sorted(bad)} that aren't in 'metrics'")
            for mode, v in targets.items():
                if not isinstance(v, (int, float)) or v < 0:
                    raise ValueError(f"stage '{self.name}': proportional targets.{mode} "
                                     f"must be a non-negative number, got {v}")
            node['targets'] = targets
            # how a target converts a metric into a DRIVE (see _proportional_tick).
            #   absolute: s = max(metric - target, 0)      -- metric's own units
            #   relative: s = max(metric / target - 1, 0)  -- dimensionless
            # The split is a RATIO of the two drives, so with 'absolute' the two
            # metrics' units and dynamic ranges set the gain. Measured on the
            # fixed-mix dev run: bwd/relative_under settles 2.48-3.18 while
            # fwd/over_coverage settles 17.0-17.8, so even with both targets
            # placed inside their own bands (2.5 / 15.0) the healthy drives are
            # 0.44 vs 2.4 and replay takes 85% of the pair on scale alone.
            # 'relative' divides that out -- the same pair gives 0.176 vs 0.16
            # -- and is combined by TILTING the idle split rather than by
            # equalizing the drives (see _proportional_tick), so both sides
            # healthy reproduces default_boost exactly and only a side
            # degrading RELATIVE TO ITS OWN target moves the mix. Default stays
            # 'absolute' so configs written against the old semantics keep them.
            drive = node.get('drive', 'absolute')
            if drive not in ('absolute', 'relative'):
                raise ValueError(f"stage '{self.name}': proportional drive must be "
                                 f"absolute|relative, got {drive}")
            if drive == 'relative':
                missing = sorted(m for m in metrics if float(targets.get(m, 0.0)) <= 0.0)
                if missing:
                    raise ValueError(
                        f"stage '{self.name}': proportional drive 'relative' divides by the "
                        f"target, so modes {missing} need a strictly positive target")
            node['drive'] = drive
            # optional ABSOLUTE per-mode ceilings on the split's own two modes.
            # `floor` bounds each side's SHARE OF THE PAIR symmetrically, so it
            # cannot express "replay may never exceed 0.15" while leaving bwd
            # free to take the rest. Without a ceiling a structurally one-sided
            # drive walks the split to floor/(1-floor) -- which is what the
            # previous targets did (bwd's drive pinned at 0, replay's fixed
            # point at ~0.90 of the pair, converging toward mode collapse since
            # bwd share is the mode-retention dial). The ceiling bounds the
            # damage a miscalibrated target can do, rather than trusting the
            # calibration.
            caps = dict(node.get('max_fracs') or {})
            bad = set(caps) - set(metrics)
            if bad:
                raise ValueError(f"stage '{self.name}': proportional max_fracs for modes "
                                 f"{sorted(bad)} that aren't in 'metrics'")
            for mode, v in caps.items():
                if not isinstance(v, (int, float)) or not 0.0 < v <= 1.0:
                    raise ValueError(f"stage '{self.name}': proportional max_fracs.{mode} "
                                     f"must be in (0, 1], got {v}")
            if caps:
                # the pair has to be able to HOLD its own mass: pinned modes are
                # re-asserted every tick, so the split's two modes always carry
                # 1 - sum(pinned) between them and ceilings summing below that
                # are unsatisfiable (the clamp would silently not bind)
                pair_mass = 1.0 - sum(float(v) for v in pinned.values())
                ceiling = sum(float(caps.get(m, 1.0)) for m in metrics)
                if ceiling < pair_mass - 1e-9:
                    raise ValueError(
                        f"stage '{self.name}': proportional max_fracs sum to {ceiling:.3f} but the "
                        f"split's modes must hold {pair_mass:.3f} between them (1 - pinned)")
            node['max_fracs'] = caps
            # optional idle split, used when BOTH sides are at/under target
            # (see _proportional_tick). Must be a mix over the two split modes;
            # omitted means "fall back to the stage's entry fracs".
            idle = node.get('default_boost')
            if idle is not None:
                idle = self._normalize_boost(idle, 'default_boost')
                if not isinstance(idle, dict) or set(idle) - set(metrics):
                    raise ValueError(
                        f"stage '{self.name}': proportional default_boost must be a "
                        f"{{mode: weight}} mix over the two split modes {sorted(metrics)}")
            node['default_boost'] = idle
            alpha = float(node.get('alpha', 0.05))
            if not 0.0 < alpha <= 1.0:
                raise ValueError(f"stage '{self.name}': proportional alpha must be in (0, 1], got {alpha}")
            node['alpha'] = alpha
            node['floor'] = float(node.get('floor', 0.01))
        elif kind == 'constraint':
            if node.get('anneal_coeffs'):
                raise ValueError(f"stage '{self.name}': anneal_coeffs needs kind: lexicographic "
                                 f"(it anneals off the lexicographic clean-streak event)")
            metrics = node.get('metrics') or {}
            if len(metrics) != 2 or set(metrics) - set(MODES):
                raise ValueError(f"stage '{self.name}': constraint balance needs a "
                                 f"'metrics' mapping of exactly two modes to metric names")
            node['metrics'] = dict(metrics)
            node['pinned'] = self._parse_pinned(node, metrics)
            # WHICH side is the hard one. The asymmetry is the whole point (see
            # _constraint_tick): the constrained mode's bar has to be a level
            # you actually know, and the other mode's is allowed to be a guess.
            constrain = node.get('constrain')
            if constrain not in metrics:
                raise ValueError(f"stage '{self.name}': balance.constrain must name one of "
                                 f"the two split modes {sorted(metrics)}, got {constrain!r}")
            node['constrain'] = constrain
            # Both bars are STRICTLY POSITIVE because both drives are relative
            # (metric/bar - 1). Relative is not a style choice here: the two
            # metrics are RMS nats on different tails with different dynamic
            # ranges (bwd/relative_under_wcen settles ~2-3, fwd/over_coverage
            # ~17-18), so an absolute-difference drive would hand the split to
            # whichever metric happens to be numerically larger -- the exact
            # scale artifact the proportional controller's 'drive: relative'
            # was added to divide out.
            bars = dict(node.get('bars') or {})
            if set(bars) != set(metrics):
                raise ValueError(f"stage '{self.name}': constraint balance needs a 'bars' "
                                 f"entry for each of {sorted(metrics)}, got {sorted(bars)}")
            for mode, v in bars.items():
                if not isinstance(v, (int, float)) or v <= 0:
                    raise ValueError(f"stage '{self.name}': bars.{mode} must be strictly "
                                     f"positive (drives are relative), got {v}")
            node['bars'] = dict(bars)
            node['bounds'] = self._parse_bounds(node, metrics, node['pinned'], 'constraint')
            gain = float(node.get('gain', 0.02))
            if not 0.0 < gain <= 1.0:
                raise ValueError(f"stage '{self.name}': constraint gain must be in (0, 1], got {gain}")
            node['gain'] = gain
            priority = float(node.get('priority', 3.0))
            if priority < 1.0:
                raise ValueError(f"stage '{self.name}': constraint priority must be >= 1 -- it is "
                                 f"the gain MULTIPLE on the constrained side, and below 1 it "
                                 f"would make the best-effort objective the dominant one")
            node['priority'] = priority
            max_step = float(node.get('max_step', 0.03))
            if not 0.0 < max_step <= 1.0:
                raise ValueError(f"stage '{self.name}': constraint max_step must be in (0, 1], got {max_step}")
            node['max_step'] = max_step
        elif kind == 'ratio':
            if node.get('anneal_coeffs'):
                raise ValueError(f"stage '{self.name}': anneal_coeffs needs kind: lexicographic "
                                 f"(it anneals off the lexicographic clean-streak event)")
            bad = set(node) - {'kind', 'metrics', 'pinned', 'numerator', 'setpoint',
                               'gain', 'max_step', 'bounds', 'converge_floor'}
            if bad:
                raise ValueError(f"stage '{self.name}': ratio balance unknown keys {sorted(bad)}")
            metrics = node.get('metrics') or {}
            if len(metrics) != 2 or set(metrics) - set(MODES):
                raise ValueError(f"stage '{self.name}': ratio balance needs a "
                                 f"'metrics' mapping of exactly two modes to metric names")
            node['metrics'] = dict(metrics)
            node['pinned'] = self._parse_pinned(node, metrics)
            # WHICH metric is on top of rho. Named explicitly rather than taken
            # from mapping order, because 'setpoint' is meaningless without it
            # and a silently inverted rho is a sign flip on the whole loop.
            numerator = node.get('numerator')
            if numerator not in metrics:
                raise ValueError(f"stage '{self.name}': balance.numerator must name one of the "
                                 f"two split modes {sorted(metrics)}, got {numerator!r}")
            node['numerator'] = numerator
            setpoint = node.get('setpoint')
            if not isinstance(setpoint, (int, float)) or setpoint <= 0:
                raise ValueError(f"stage '{self.name}': ratio setpoint must be strictly "
                                 f"positive (it is a ratio of two positive metrics), got {setpoint!r}")
            node['setpoint'] = float(setpoint)
            node['bounds'] = self._parse_bounds(node, metrics, node['pinned'], 'ratio')
            gain = float(node.get('gain', 0.02))
            if not 0.0 < gain <= 1.0:
                raise ValueError(f"stage '{self.name}': ratio gain must be in (0, 1], got {gain}")
            node['gain'] = gain
            max_step = float(node.get('max_step', 0.05))
            if not 0.0 < max_step <= 1.0:
                raise ValueError(f"stage '{self.name}': ratio max_step must be in (0, 1], got {max_step}")
            node['max_step'] = max_step
            # convergence gate, in the metrics' own units (nats). A scale-free
            # setpoint keeps demanding the same RATIO whether the two halves
            # are 20 nats apart or 2, so once both are inside the convergence
            # scale the loop would be steering on noise. Null = never fade.
            floor = node.get('converge_floor')
            if floor is not None:
                floor = float(floor)
                if floor <= 0:
                    raise ValueError(f"stage '{self.name}': ratio converge_floor must be "
                                     f"strictly positive or null, got {floor}")
            node['converge_floor'] = floor
        else:
            raise ValueError(f"stage '{self.name}': balance.kind must be "
                             f"lexicographic|proportional|constraint|ratio")
        node['kind'] = kind
        return node

    def _parse_buffer_servo(self, node):
        """The replay-buffer freshness servo (see _buffer_servo_tick). Declared
        per stage because it is only meaningful where replay trains, and
        because its state resets at transitions like every other controller
        here. Absent = the buffer runs exactly at its configured knobs."""
        if node is None:
            return None
        bad = set(node) - {'numerator', 'denominator', 'bar', 'release', 'scale',
                           'gain', 'relax', 'max_boost', 'max_step'}
        if bad:
            raise ValueError(f"stage '{self.name}': buffer_servo unknown keys {sorted(bad)}")
        out = {'numerator': node.get('numerator', 'replay/scatter_err'),
               'denominator': node.get('denominator', 'fwd/scatter_err')}
        for key in ('numerator', 'denominator'):
            if not isinstance(out[key], str) or '/' not in out[key]:
                raise ValueError(f"stage '{self.name}': buffer_servo.{key} must be a "
                                 f"'dir/metric' name, got {out[key]!r}")
        bar = float(node.get('bar', 1.0))
        release = float(node.get('release', 1.5))
        if not 0.0 < bar <= release:
            raise ValueError(f"stage '{self.name}': buffer_servo needs 0 < bar <= release "
                             f"(got bar={bar}, release={release}) -- bar > release would make "
                             f"the tighten and release terms fire simultaneously and fight")
        out['bar'], out['release'] = bar, release
        # Deviation at which the servo runs at FULL rate. Without it the drive
        # is the raw deficit (bar - ratio), which is bounded above by `bar` and
        # in practice sits at ~0.03: measured live, replay/fwd scatter entered
        # at 0.964, so a raw-deficit servo would need ~23k steps to traverse
        # its boost range and is effectively inert exactly where it lives. The
        # inversion is a THRESHOLD phenomenon -- crossing below 1 at all is the
        # pathology, and depth below it is not proportionally meaningful -- so
        # the drive saturates at `scale` and the servo behaves like a
        # constant-rate ramp with a deadband, which is also why it cannot
        # chatter (the two ramp directions are separated by bar..release).
        scale = float(node.get('scale', 0.1))
        if scale <= 0.0:
            raise ValueError(f"stage '{self.name}': buffer_servo.scale must be > 0, got {scale}")
        out['scale'] = scale
        # Sized against the LOOP DELAY, which is what limits this servo: the
        # sensor is a metric_tracker EMA refreshed once per 10 replay steps
        # (~250 train steps of smoothing) sitting on top of a buffer that needs
        # ~mean_residence_steps to turn over. An integrator whose traverse time
        # is comparable to that delay will overshoot and hunt, so the default
        # puts a full-drive traverse of the boost range at ~1200 train steps,
        # several times the delay.
        gain = float(node.get('gain', 0.02))
        if not 0.0 < gain <= 1.0:
            raise ValueError(f"stage '{self.name}': buffer_servo.gain must be in (0, 1], got {gain}")
        out['gain'] = gain
        # relax < 1 releases SLOWER than it tightens. It must be > 0: a
        # one-way servo's fixed point is the maximum boost, which is the same
        # ratchet failure the LR controller's recovery ramp exists to avoid.
        relax = float(node.get('relax', 0.25))
        if not 0.0 < relax <= 1.0:
            raise ValueError(f"stage '{self.name}': buffer_servo.relax must be in (0, 1] -- 0 "
                             f"makes the servo one-way and its fixed point max_boost")
        out['relax'] = relax
        max_boost = float(node.get('max_boost', 12.0))
        if max_boost < 1.0:
            raise ValueError(f"stage '{self.name}': buffer_servo.max_boost must be >= 1, got {max_boost}")
        out['max_boost'] = max_boost
        max_step = float(node.get('max_step', 0.03))
        if not 0.0 < max_step <= 1.0:
            raise ValueError(f"stage '{self.name}': buffer_servo.max_step must be in (0, 1], got {max_step}")
        out['max_step'] = max_step
        return out

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
            if name == 'seed_prior_from_anchors' and arg:
                # 'N' or 'N:flush' -- validate at parse time so a config typo
                # fails at startup, not mid-transition
                parts = arg.split(':')
                if (not parts[0].isdigit()
                        or len(parts) > 2
                        or (len(parts) == 2 and parts[1] != 'flush')):
                    raise ValueError(f"stage '{self.name}' {where}: '{a}' -- expected "
                                     f"seed_prior_from_anchors:<int> or seed_prior_from_anchors:<int>:flush")
            if name == 'reseed_prior_from_dataset' and arg and arg != 'flush':
                raise ValueError(f"stage '{self.name}' {where}: '{a}' -- expected "
                                 f"reseed_prior_from_dataset or reseed_prior_from_dataset:flush")
            if name == 'rebuild_prior_by_churn' and arg and not arg.isdigit():
                raise ValueError(f"stage '{self.name}' {where}: '{a}' -- expected "
                                 f"rebuild_prior_by_churn or rebuild_prior_by_churn:<int>")
            if name in ('set_lr_flow', 'set_lr_policy'):
                try:
                    if float(arg) <= 0:
                        raise ValueError
                except (TypeError, ValueError):
                    raise ValueError(f"stage '{self.name}' {where}: '{a}' -- expected "
                                     f"{name}:<positive float>, e.g. {name}:1.0e-4")
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
        if self.balance['kind'] in ('proportional', 'constraint', 'ratio'):
            # the two split modes, PLUS any mode the stage pins at a nonzero
            # entry frac. A two-mode split only redistributes its two
            # modes' combined mass, so a third mode held fixed (e.g. fwd
            # pinned while bwd/replay co-converge) is never "boosted" yet is
            # very much active -- omitting it here made the fused step drop it
            # from the loss entirely.
            # 'ratio' belongs here for the same reason: like the other two it
            # is a two-mode split parsed into {'metrics', 'pinned'} and it has
            # no 'rules', so falling through to the lexicographic branch below
            # raises KeyError on the first fused step of the stage.
            return (set(self.balance['metrics'])
                    | set(self.balance.get('pinned') or {})
                    | {mode for mode, v in self.fracs.items()
                       if isinstance(v, (int, float)) and v > 0})
        out = set()
        for r in self.balance['rules']:
            b = r['boost']
            out.update(b if isinstance(b, dict) else {b})
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
        if self.balance['kind'] in ('proportional', 'constraint', 'ratio'):
            names += list(self.balance['metrics'].values())
        else:
            names += [r['metric'] for r in self.balance['rules']]
        for term in (self.exit or []):
            names.append(term['metric'])
        # the buffer servo reads two branch metrics of its own, and a branch it
        # reads must not be allowed to skip its force-refresh rollout
        if self.buffer_servo is not None:
            names += [self.buffer_servo['numerator'], self.buffer_servo['denominator']]
        # ...and so does the hot-LR sensor. This clause used to name the plateau
        # LR sensor, for the same reason, and it went dead when `plateau` left
        # LR_SENSOR_KINDS -- a gate on a retired key can never fire, so it reads
        # as "this is handled" while handling nothing. The sensor advances its
        # window ONLY on fresh writes, so a channel whose branch is dormant is
        # never written, the window never fills, and the sensor reports
        # NO_READING for the whole stage without ever claiming to be blind.
        # Naming it here is what keeps its branch's force-refresh rollout alive.
        # Note this is currently a no-op on mk_dev -- all four declared channels
        # already sit in read_modes via the balance rules -- which is exactly the
        # accident it exists to stop depending on: var_conditioning reads
        # {bwd, fwd} only, so a replay/* sensor there would be silently blind.
        if self.hot_lr_sensor is not None:
            names.append(self.hot_lr_sensor['channel'])
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
        self._energy_coeff_defaults = None
        # pristine replay-buffer knobs, captured the first time the buffer
        # servo runs and never written back. Instance state is correct here
        # BECAUSE args are re-parsed from the yaml at every launch while the
        # boost itself rides in stage_ctrl: base x checkpointed boost
        # reconstructs the live values exactly, with no risk of a boosted value
        # being mistaken for the base after a resume.
        self._rb_base = None

    # ------------------------------------------------------------------ parse

    @property
    def stages(self):
        if self._stages is None:
            # ONE resolution point, shared with the config validators. The config
            # holds every protocol under `protocols:` and names the live one in
            # `protocol:`; the validators must agree with the trainer about which
            # stages are live, or a check passes on a list the run never executes.
            from config_invariants import (active_protocol_name, active_stages,
                                           PROTOCOL_LIBRARY)
            args = self.m.args
            specs = active_stages(args)
            if not specs:
                name = active_protocol_name(args)
                have = getattr(args, PROTOCOL_LIBRARY, None)
                known = sorted(vars(have)) if have is not None else []
                raise ValueError(
                    f"no live protocol: `protocol` names {name!r} and `protocols` "
                    f"defines {known or 'nothing'}. Set `protocol:` to one of them "
                    f"(see configs/mk_dev.yaml).")
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
            # str is admitted alongside the numbers because not every entry in a
            # loss_coeffs block is a weight -- tb_z_source is 'learned'/'persistent'
            # and lives here so a stage can set it per branch. Filtering to numerics
            # dropped it from `base`, which then made the stage override read as an
            # unknown key. Callers that do arithmetic select the coefficients they
            # want by name; none iterate the block and multiply.
            self._coeff_defaults = {
                m: {k: v for k, v in vars(getattr(self.m.args, f'{m}_loss_coeffs')).items()
                    if isinstance(v, (int, float, str))}
                for m in MODES}
        base = self._coeff_defaults[mode]
        overrides = self.stage.loss_coeffs.get(mode, {})
        unknown = set(overrides) - set(base)
        if unknown:
            raise ValueError(f"stage '{self.stage.name}' {mode} loss_coeffs override unknown "
                             f"keys {sorted(unknown)} -- add them to the base "
                             f"{mode}_loss_coeffs block first")
        return {**base, **overrides}

    def energy_coeffs(self) -> dict:
        """Live values for whichever energy_config coefficients the CURRENT
        stage's balance.anneal_coeffs names (e.g. {bounding_coeff: 4.2}).
        The base energy_config value is the SOFT one -- it's what
        self.energy_function is constructed with, so it's already in effect
        for every stage from run start, including stages that name nothing
        here. Only once THIS stage's anneal events start firing (see
        _anneal) does the live value leave that base and ramp up toward the
        stage's own `target`. Coefficients no stage ever names are NOT
        returned here and so are left untouched by set_energy_coeffs -- e.g.
        lj_coeff, which train.py calibrates once at init from the prior's
        thermal_scaling_factor (see lj_coeff_silent_override) and this must
        never clobber back to its static config value."""
        if self._energy_coeff_defaults is None:
            self._energy_coeff_defaults = {
                k: v for k, v in vars(self.m.args.energy_config).items()
                if isinstance(v, (int, float))}
        bal = self.stage.balance
        if not bal or bal['kind'] != 'lexicographic':
            return {}
        out = {}
        for name, spec in bal['anneal_coeffs'].items():
            if name not in self._energy_coeff_defaults:
                raise ValueError(f"stage '{self.stage.name}': anneal_coeffs.{name} is not "
                                 f"a numeric energy_config key")
            rs = self.ctrl['coeffs'].setdefault(name, {})
            if 'val' not in rs:
                rs['val'] = self._energy_coeff_defaults[name]
            out[name] = rs['val']
        return out

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
        triggers and rules read them as 'gates/<name>'. The publish is stamped
        so a reader can tell a fresh verdict from the last one left lying
        around -- a gate that stops publishing leaves its value behind."""
        self.ctrl['gates'][name] = float(value)
        self.ctrl.setdefault('gate_written', {})[name] = int(getattr(self.m, 'step_ind', 0))

    def _write_step(self, name: str, eval_metrics: dict = None):
        """The step at which `name` last received a FRESH value, or None if it
        has not been written (in this process, for tracker/gate metrics).

        The companion to `_resolve`, and the thing the exit engine was missing:
        every source here PERSISTS its last value, so `_resolve` alone cannot
        distinguish "the metric is passing" from "the metric passed once, a
        long time ago, and nothing has written it since"."""
        if name.startswith('gates/'):
            return self.ctrl.get('gate_written', {}).get(name[len('gates/'):])
        if name.startswith('eval/'):
            # eval metrics are handed in for the duration of one maybe_advance
            # call and do not persist, so presence in the dict IS freshness;
            # the eval's own step is the stamp.
            if eval_metrics is None or name[len('eval/'):] not in eval_metrics:
                return None
            return int(getattr(self.m, 'step_ind', 0))
        direction, _, metric = name.partition('/')
        return self.m.metric_tracker.written_step(direction, metric)

    def gate_state(self, name: str) -> dict:
        """Scratch dict for a gate's internals (windows, latches), checkpointed
        with stage_ctrl and reset at transitions like everything else here."""
        return self.ctrl['gate_state'].setdefault(name, {})

    # ------------------------------------------------------------------- tick

    def tick(self):
        """10-step-cadence work: the balance nudge, the buffer servo, then
        exit-trigger arming. Transitions themselves only execute inside
        evaluation() (maybe_advance), with fresh eval metrics in hand."""
        if self.stage.balance is not None:
            self._balance_tick()
        self._buffer_servo_tick()
        self._exit_tick()

    # ------------------------------------------------------------- exit logic

    def _term_passes(self, term: dict, eval_metrics: dict = None):
        v = self._resolve(term['metric'], eval_metrics)
        if v is None:
            return False
        if term.get('abs'):
            v = abs(v)
        return (v > term['above']) if 'above' in term else (v < term['below'])

    def _advance_term(self, i: int, term: dict, eval_metrics: dict = None):
        """Advance ONE exit term's pass-streak, gated on the metric having been
        FRESHLY WRITTEN since this streak last judged it. Three outcomes, and
        keeping them distinct is the whole point:

            fresh value, passes  -> advance
            fresh value, fails   -> reset to 0
            NO fresh value       -> HOLD, neither advance nor reset

        PATIENCE COUNTS MEASUREMENTS, NOT TICKS. It used to count ticks, and
        every value source here (metric_tracker EMAs, gates/* publishes, eval
        metrics) persists its last value, so a term read faster than its metric
        is written counted ONE sample as N. Measured on the prod0810 phase-1
        block: a single `bwd/tbc` write at step 100 carried the streak to 20
        over 20 subsequent ticks with no further write, and `gates/mle_flat`
        published once cleared its `patience: 5` three ticks later. A patience
        of 5 on a metric written every 500 steps meant 50 steps of the same
        number, not five independent readings.

        Holding rather than resetting is deliberate, and it is the other half
        of the same bug. Resetting on a quiet tick would make patience > 1
        UNREACHABLE for any metric slower than the 10-step tick -- which is
        what `next_battery.md` 1.3 believed was already happening. Neither
        counting nor discounting a non-measurement is right; not judging it
        is."""
        streaks, seen = self.ctrl['exit'], self.ctrl.setdefault('exit_seen', {})
        stamp = self._write_step(term['metric'], eval_metrics)
        if stamp is None or stamp == seen.get(i):
            return                      # nothing new to judge
        seen[i] = stamp
        streaks[i] = streaks.get(i, 0) + 1 if self._term_passes(term, eval_metrics) else 0

    def _exit_tick(self):
        """Advance each tick-resolvable term's pass-streak; on the rising edge
        of 'all tick terms at patience', pull the next eval forward. eval/*
        terms can't be watched at tick cadence -- they are advanced with fresh
        values inside maybe_advance, exactly like the old phase-1 gate latched
        on the MLE slope and left wass to the pulled-forward eval.

        Arming stays a TICK-TERM question. An eval/* term cannot contribute to
        it without deadlocking the pull-forward against the very eval that
        would satisfy it."""
        stage = self.stage
        if stage.exit is None:
            return
        streaks = self.ctrl['exit']
        armed = True
        for i, term in enumerate(stage.exit):
            if term['metric'].startswith('eval/'):
                continue
            self._advance_term(i, term)
            if streaks.get(i, 0) < term.get('patience', 1):
                armed = False
        if armed and not self.ctrl['exit_armed']:
            self.ctrl['request_eval'] = True  # pull the eval that will run maybe_advance
        self.ctrl['exit_armed'] = armed

    def _exit_satisfied(self, eval_metrics: dict) -> bool:
        """Every term at its patience. UNIFORM over eval/* and tick terms now
        that both keep a real streak -- eval/* used to be tested once against
        the fresh metrics with `patience` silently DISCARDED, so a patience of
        5 on an eval metric fired on the first clean eval. The key was accepted
        by _parse_exit and then ignored, which is the config reading as one
        thing and meaning another."""
        stage = self.stage
        if stage.exit is None:
            return False  # terminal stage
        return all(self.ctrl['exit'].get(i, 0) >= term.get('patience', 1)
                   for i, term in enumerate(stage.exit))

    def maybe_advance(self, eval_metrics: dict) -> bool:
        """Called from evaluation() after metrics are computed -- the one place
        transitions execute. Advances the eval/* terms first (this is the only
        cadence at which those metrics exist, so one eval is one measurement),
        then tests the whole trigger. Clears any pending pulled-forward eval
        request (this eval satisfies it, whoever set it: the trigger arming
        tick, or a reloaded pre-transition snapshot's stamped request_eval)."""
        self.ctrl['request_eval'] = False
        for i, term in enumerate(self.stage.exit or []):
            if term['metric'].startswith('eval/'):
                self._advance_term(i, term, eval_metrics)
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
                print(f"protocol: prior model loaded (prior_model_name) "
                      f"-- skipping stage '{self.stage.name}' (policy weights untouched)")
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
        # 'best' saves, and its Adam moments describe the wrong loss surface.
        # `LRController.on_stage_change` must run with m.stage already switched.
        # It re-enters burn-in, which is not optional at a transition: rebuilding
        # the optimizers restarts Adam's step counter, and bracketing from a
        # counter at t=10 measures 0.153 of the rate under test.
        m.combo_loss_record = []
        # batch sizer: its conclusion and rung table describe the OUTGOING stage's
        # step-cost and occupancy profile -- carried over, they would answer the
        # incoming stage's question with the outgoing stage's measurements (same
        # rationale as the OOM path's reset). The incoming stage re-runs the ladder.
        m.batch_sizer = None
        m.batch_size_oom_ceiling = None  # the incoming stage has its own memory profile
        m.batch_size_oom_ceiling_at = None       # ...and its own expiry clock
        m.batch_size_oom_min = None              # ...and its own OOM history
        # runaway-guard latches: both are conclusions about the OUTGOING stage's
        # fixed per-step cost ("cutting the batch did not move step time", "the batch
        # is already at the accumulation target"). The incoming stage has a different
        # cost profile, so it gets to re-derive them -- and to say its piece once.
        m._runaway_last_cut = None
        m._runaway_unresponsive_stage = None
        m._accum_floor_warned_stage = None
        times = getattr(m, '_recent_step_times', None)
        if times is not None:
            times.clear()
            m._recent_step_work.clear()
        # THE LEVEL IS STAGE STATE TOO, not just the bookkeeping around it. It
        # used to carry over, so equilibration inherited train_prior's level --
        # and since selection only ever moves UP from the base, an MLE-sized
        # batch became a FLOOR for a stage whose steps cost ~100x more
        # (prod0810: 2722 cheap bwd/dataset steps -> 2722 fused steps at 181 s,
        # then grown further). Re-enter every stage at the configured base and
        # let the sizer re-derive any growth from in-stage measurements.
        if bool(getattr(m.args, 'grow_batch_size', True)):
            base_batch = min(int(m.args.batch_size), int(m.args.max_batch_size))
            if m.batch_size != base_batch:
                print(f"batch: stage change -- resetting {m.batch_size} -> {base_batch} "
                      f"(re-selecting under '{new.name}' step costs)")
                m.batch_size = base_batch
        m.batch_size_last_grow = m.step_ind  # full dwell of in-stage steps before the first grow
        m.init_schedulers_optimizers()
        m.set_loss_coeffs()
        burn_in_steps = m.lr_controller.on_stage_change()
        if burn_in_steps:
            print(f"protocol: optimizers rebuilt, LR burn-in over {burn_in_steps} train steps")
        # The adaptive clip bar is calibrated against a gradient distribution, and
        # a stage boundary is where that distribution moves -- new coeffs, new
        # branch fracs, fresh Adam moments, sometimes a different train_mode
        # entirely. The tracker's own rate limit would walk across a shift like
        # that eventually (ln(ratio)/eta steps), but a boundary is the one place
        # the change is KNOWN rather than inferred, so it recalibrates on the
        # signal instead of paying the walk. refresh() holds the outgoing bar live
        # while it re-measures -- the run is never unguarded across the
        # turbulence -- and prints the before/after ratio, so unlike a silent
        # reset this one reports its own size.
        m.grad_guard.refresh(reason=f"stage {old.name} -> {new.name}")

        for name, arg in new.on_enter:
            self._run_action(name, arg, eval_metrics)
        # freeze this stage's healthy turnover point AFTER on_enter's buffer
        # surgery has run, for fire_loss_spike to rewind to instead of reversing
        # a phase. modeller_state.stage is already the NEW stage here (unlike the
        # outgoing on_exit snapshot, which records the old one), so the rewind
        # can tell them apart. No buffers: the spike path keeps the live
        # (same-stage, strictly fresher) buffers, exactly as for a 'best' rewind.
        m.checkpointer.save('stage_start')
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
        elif name == 'seed_prior_from_anchors':
            parts = arg.split(':') if arg else []
            n_per_condition = int(parts[0]) if parts else 1
            flush = len(parts) > 1 and parts[1] == 'flush'
            self.m.seed_prior_from_condition_minima(n_per_condition, flush=flush)
        elif name == 'reseed_prior_from_dataset':
            self.m.reseed_prior_from_dataset(flush=(arg == 'flush'))
        elif name == 'rebuild_prior_by_churn':
            self.m.rebuild_prior_by_churn(int(arg) if arg else None)
        elif name == 'set_lr_flow':
            # The flow/Z group is exempt from the bracket's scale
            # (lr_control.control_flow_lr: false), so nothing else will move it
            # and nothing else will move it BACK -- this is the only lever on it.
            # It is a stage action because the right rate is a property of what
            # the flow head IS on this route: a LearnableScalar unconditionally
            # (1-D convex, 0.1 is jitter) versus a per-condition NETWORK on the
            # conditional one, where the same 0.1 diverges once emp_z trains it.
            # Set on args AND on both live groups, since a rebuild reads args
            # while an already-built optimizer does not.
            v = float(arg)
            m = self.m
            m.args.lr_flow = v
            if 'flow' in m.optimizers:
                for g in m.optimizers['flow'].param_groups:
                    g['lr'] = v
            if 'fused' in m.optimizers:      # the fused optimizer's TRAILING group is the flow one
                m.optimizers['fused'].param_groups[-1]['lr'] = v
            print(f"protocol: lr_flow -> {v:g} on entering '{self.stage.name}'")
        elif name == 'set_traj_checkpoint':
            # PER-STAGE, because the trade it makes is only worth paying where
            # memory is actually scarce. Gradient checkpointing over the rollout
            # recomputes every SDE sub-step in the backward pass: it buys a large
            # VRAM saving (~33x at T=100) at roughly a doubling of the rollout's
            # dispatch count, and at T=60 the rollout is dispatch-bound, so that
            # cost lands squarely on step time.
            #
            # train_prior makes no energy call (energy/frac_of_step = 0), so the
            # MLIP spike the checkpointing exists to survive is not present
            # there; equilibration scores every step through the MLIP and needs
            # it. One global flag has to satisfy the stage that needs it.
            #
            # NOTE the batch this frees or costs is NOT re-derived here -- the
            # sizer re-measures on its own, and it must, because the per-sample
            # memory changes by an order of magnitude across this switch. Clearing
            # the ladder is the point: rungs measured under the old regime say
            # nothing about the new one.
            v = str(arg).strip().lower() not in ('0', 'false', 'off', '')
            m = self.m
            m.args.traj_checkpoint = v
            for mdl in (getattr(m, 'gfn_model', None), getattr(m, 'ema_model', None)):
                if mdl is not None:
                    mdl.traj_checkpoint = v
            m.batch_sizer = None
            print(f"protocol: traj_checkpoint -> {v} on entering "
                  f"'{self.stage.name}'; ladder re-armed")
        elif name == 'set_max_batch_size':
            # PER-STAGE BATCH CEILING, because what a step COSTS is a property of
            # the stage, not of the run. On the MLIP routes train_prior makes no
            # energy call at all, so MLE can hold a batch orders of magnitude
            # larger than the stage that follows it -- and one global
            # max_batch_size has to be small enough for the expensive stage,
            # which starves the cheap one for the whole of phase 1.
            #
            # Sizing the global cap for the EXPENSIVE stage is what
            # p4_mace_mle did (max_batch_size 400 against an MLE that could hold
            # thousands), and it spent its entire phase 1 there.
            #
            # CLAMPS THE LIVE BATCH TOO, not just the ceiling. Lowering the cap
            # on entry is exactly the transition-OOM guard this exists for, and a
            # cap that let the current batch stay above it would guard nothing --
            # the first step of the new stage would run at the old size, which is
            # the allocation the cap was lowered to prevent. Raising it leaves the
            # batch where it is and lets the ladder climb normally.
            v = int(float(arg))
            m = self.m
            m.args.max_batch_size = v
            if m.batch_size > v:
                print(f"protocol: max_batch_size -> {v} on entering "
                      f"'{self.stage.name}'; clamping batch {m.batch_size} -> {v}")
                m.batch_size = v
            else:
                print(f"protocol: max_batch_size -> {v} on entering "
                      f"'{self.stage.name}' (batch {m.batch_size} already under it)")
            # the ladder's conclusions were reached under the OLD cap, at rungs
            # this one may not permit -- re-measure rather than carry them over
            m.batch_sizer = None
        elif name == 'set_lr_policy':
            # PER-STAGE BASE RATE for the managed policy groups. lr_control.seed_lr
            # is ONE number for every stage, but a bwd MLE stage and a fused VarGrad
            # stage on a per-condition target do not want the same rate, and the
            # bracket's candidate grid is a set of MULTIPLIERS on this base -- so a
            # base that is wrong by 10x moves the whole grid off the interesting
            # region and the bracket reports unbracketed_high or all_failed rather
            # than a boundary.
            # This sets the BASE only. The group stays bracket-managed, because
            # _managed_keys reads args.lr_servo_managed -- recorded by
            # resolve_derived_config at load -- and not the live value.
            # _apply_lrs recomputes base * scale on the next tick.
            v = float(arg)
            m = self.m
            for key in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
                setattr(m.args, key, v)
            print(f"protocol: lr_policy/back/replay/fused -> {v:g} on entering "
                  f"'{self.stage.name}' (base only; the bracket still owns the scale)")

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
            # ANCHOR ON THE FORWARD BRANCH, because the forward branch is what
            # log_Z's fixed point IS. In the TB stages this action opens, `fwd`
            # carries freeze_policy and `bwd`/`replay` carry freeze_z, so the
            # forward loss is the only thing that trains the scalar and TB drives
            # it to E[log w] under FORWARD samples. Anchoring anywhere else does
            # not change that fixed point, only how far the scalar has to travel
            # to reach it -- and it travels under a Huber whose gradient is capped
            # at fwd_loss_coeffs.beta, so a large opening offset descends at a rate
            # independent of its own size while every other branch's residual is
            # carried along with it.
            #
            # The three candidates:
            #   eval_fwd/jensen_z  E[log w] on forward eval samples: the fixed
            #                  point itself, measured raw -- one eval, no trim, no
            #                  EMA. Noisy exactly when the forward policy is bad,
            #                  which is accepted: a noisy reading of the right
            #                  quantity beats a clean reading of another one.
            #   ema_logw       the tracker's E[log w], rank-trimmed, 1/n-diluted
            #                  and EMA'd over an evidence half-life. NOT the same
            #                  estimand unless the same sampler feeds it: the
            #                  tracker is updated by whichever branches ran, so
            #                  after a bwd-only MLE phase 1 its ema_logw is the
            #                  ANCHOR level, and anchoring there hands phase 2 the
            #                  full anchor-to-policy gap as an opening transient.
            #                  Kept as the fallback for a transition with no eval
            #                  stream.
            #   ema_log_z_emp  logmeanexp -- (nearly) UNBIASED for log Z, and
            #                  unusable as an anchor: logsumexp has no 1/n
            #                  dilution, so one bad off-policy sample sends it to
            #                  billions of nats. Feeding it back measurably broke a
            #                  live run. See ConditionLogZTracker.lookup.
            empirical_z, src = None, None
            if eval_metrics is not None and 'eval_fwd/jensen_z' in eval_metrics:
                empirical_z = eval_metrics['eval_fwd/jensen_z']
                src = 'eval_fwd/jensen_z'
            tracked = None
            tracker = getattr(m, 'condition_log_z', None)
            if tracker is not None:
                # plain lists, not tensor ops: this module is deliberately torch-free.
                # `v == v` is the NaN test (an unvisited slot is NaN, not missing).
                seen = [v for c, v in zip(tracker.count.tolist(), tracker.ema_logw.tolist())
                        if c >= tracker.min_visits and v == v]
                if seen:
                    tracked = (sum(seen) / len(seen), len(seen))
            if empirical_z is None:
                # no eval stream (a transition that did not fire at an eval): take
                # the tracker rather than refuse to bootstrap at all
                if tracked is None:
                    raise RuntimeError(
                        "bootstrap_z needs either eval metrics (eval_fwd/jensen_z) or "
                        "a visited condition_log_z tracker -- it can only run from an "
                        "eval-time transition")
                empirical_z, src = tracked[0], f'condition_log_z.ema_logw ({tracked[1]} visited, no eval stream)'
            # both levels in the same line: their GAP is the handoff quality, and
            # it is the opening transient phase 2 has to absorb
            print(f"bootstrap_z: log_Z <- {empirical_z:.3f} from {src}"
                  + (f" (condition_log_z.ema_logw was {tracked[0]:.3f} over {tracked[1]} visited)"
                     if tracked is not None else " (tracker cold)"))
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
        """Log-space trend projection: the metrics are nonnegative
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

    def _anneal(self):
        """All rules clean for anneal_patience ticks: tighten every annealed
        ABSOLUTE rule's live threshold by controller.decay_rate (per-rule
        'rate' overrides), down to its 'min' -- a number, or a metric name
        resolved live (the dynamic floor that replaced the old
        replay-outcompeting guard: e.g. min: fwd/scatter_err says 'never
        require replay to beat fresh on-policy')."""
        ctrl = self.m.args.controller
        for i, rule in enumerate(self.stage.balance['rules']):
            spec = rule.get('anneal')
            if not spec or 'above' not in rule:
                continue
            rs = self._rule_state(i)
            rate = float(spec.get('rate', getattr(ctrl, 'decay_rate', 0.95)))
            floor = spec.get('min', 0.0)
            if isinstance(floor, str):
                floor = self._resolve(floor)
                if floor is None:
                    continue  # dynamic floor unavailable: hold rather than guess
            thr = self._rule_threshold(rule, rs)
            if thr > floor:
                rs['thr'] = max(thr * rate, float(floor))

        # energy_config coefficients (bounding_coeff, reduction_coeff, ...):
        # same clean-streak event, RAMPED UP from the base config value
        # (soft, in effect since run start) toward this stage's 'target'.
        # Since this only runs once every rule above has gone quiet for
        # anneal_patience ticks, it is strictly lower priority than all of
        # them -- the lexicographically last thing to move: the
        # boundary/reduction penalty only firms up to full strength once
        # THIS stage's balance is fully converged.
        if self._energy_coeff_defaults is None:  # normally seeded by energy_coeffs() first
            self._energy_coeff_defaults = {
                k: v for k, v in vars(self.m.args.energy_config).items()
                if isinstance(v, (int, float))}
        for name, spec in self.stage.balance['anneal_coeffs'].items():
            rate = float(spec.get('rate', getattr(ctrl, 'decay_rate', 0.95)))
            target = float(spec['target'])
            rs = self.ctrl['coeffs'].setdefault(name, {})
            cur = rs.get('val', self._energy_coeff_defaults[name])
            if cur < target:
                rs['val'] = min(cur / rate, target)

    def _balance_tick(self):
        bal = self.stage.balance
        if bal['kind'] == 'proportional':
            self._proportional_tick(bal)
            return
        if bal['kind'] == 'constraint':
            self._constraint_tick(bal)
            return
        if bal['kind'] == 'ratio':
            self._ratio_tick(bal)
            return
        ctrl = self.m.args.controller

        chosen = None
        for i, rule in enumerate(bal['rules']):
            rs = self._rule_state(i)
            v = self._rule_value(rule, rs)
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

        # the boost's NAME for logging: a mix default counts as its dominant
        # mode, while the nudge below receives the full mix
        chosen_name = (max(chosen, key=chosen.get) if isinstance(chosen, dict) else chosen)

        self.ctrl['boost'] = chosen_name
        self._nudge_mode_fracs(chosen)

    @staticmethod
    def _drive(value, live_target, kind):
        """Metric -> unmet-need drive, in whichever form the stage declared.
        Shared by _proportional_tick and report() so the logged prop_drive_* is
        the SAME number the split is computed from (they diverged silently once
        report() hardcoded the absolute form, which is how a welded-off drive
        got read as 'one thing to watch' instead of the cause)."""
        if kind == 'relative':
            return max(value / live_target - 1.0, 0.0) if live_target > 0.0 else 0.0
        return max(value - live_target, 0.0)

    def _proportional_tick(self, bal):
        """Split two modes' COMBINED frac mass in proportion to how far each
        side still is from its own target, EMA-nudged by alpha, floored both
        sides. Any third mode is untouched, so a stage can pin it (see
        active_modes).

            s_i    = max(metric_i - targets_i, 0)
            target = s_a / (s_a + s_b)
            share_a <- (1 - alpha) * share_a + alpha * target

        `targets` are SOFT reference levels, not gates. Subtracting them is
        what makes the equilibrium mean something: raw metrics rarely reach
        zero (fwd/over_coverage floors near its natural level, c8utdn8q), so
        an un-offset ratio equilibrates on those floors rather than on need,
        and the side with the larger floor wins permanently. Setting one
        target BELOW another expresses a priority -- the mode whose target is
        tighter keeps drawing share for longer.

        This is deliberately NOT a lexicographic threshold: crossing a target
        only zeroes that side's contribution, it never hands the whole batch
        to one mode, so there is no switching and therefore no limit cycle
        (which is what the rule-based controller did in replay_july26). When
        BOTH sides are at or under target there is nothing to arbitrate, so
        the current split is held rather than snapped to an arbitrary ratio.

        alpha sets the response time (~10 train steps per tick, so the time
        constant is ~10/alpha steps). It MUST be slower than the natural
        absorption cycle -- bwd dragging in buffer states degrades forward
        calibration for ~1-2k steps before recovering, and a controller fast
        enough to react to that fights the mechanism instead of the
        pathology. Default 0.05 (~200 steps) is far too fast for that; use
        ~0.002-0.005 unless the plant is known to be quicker.
        """
        m = self.m
        # re-assert held modes first, so the pin survives a resume (fracs come
        # back from the checkpoint, but config owns behavior) and holds even on
        # the early-return paths below
        for mode, value in (bal.get('pinned') or {}).items():
            setattr(m, f'{mode}_frac', float(value))
        (mode_a, metric_a), (mode_b, metric_b) = bal['metrics'].items()
        s_a, s_b = self._resolve(metric_a), self._resolve(metric_b)
        if s_a is None or s_b is None:
            return  # hold rather than starve on absent data
        targets = bal.get('targets') or {}
        # live targets = configured targets x the annealed scale (see the
        # 'anneal' block in _parse_balance for why it is ONE scale)
        scale = float(self.ctrl.get('prop_scale', 1.0))
        s_a = self._drive(s_a, float(targets.get(mode_a, 0.0)) * scale, bal.get('drive'))
        s_b = self._drive(s_b, float(targets.get(mode_b, 0.0)) * scale, bal.get('drive'))
        floor = float(bal.get('floor', 0.01))
        alpha = float(bal.get('alpha', 0.05))
        frac_a = getattr(m, f'{mode_a}_frac')
        frac_b = getattr(m, f'{mode_b}_frac')
        total = frac_a + frac_b
        if total <= 0:
            return
        # anneal bookkeeping, on the drives computed at the CURRENT scale: a
        # streak of both-satisfied ticks tightens the whole target vector for
        # the NEXT tick. Any drive at all resets the streak, so the anneal
        # only ever fires from a genuinely quiet stretch.
        spec = bal.get('anneal')
        if spec:
            if s_a + s_b <= 0.0:
                streak = int(self.ctrl.get('prop_streak', 0)) + 1
                if streak >= spec['patience']:
                    self.ctrl['prop_scale'] = max(scale * spec['rate'], spec['min_scale'])
                    streak = 0
                self.ctrl['prop_streak'] = streak
            else:
                self.ctrl['prop_streak'] = 0
        # the idle split: the allocation this stage wants when nothing is
        # wrong. `default_boost` declares it; absent that, the stage's own
        # entry fracs are the idle by construction.
        idle = bal.get('default_boost')
        src = idle if isinstance(idle, dict) else self.stage.fracs
        w_a, w_b = float(src.get(mode_a, 0.0)), float(src.get(mode_b, 0.0))
        if bal.get('drive') == 'relative':
            # TILT the idle allocation by each side's relative unmet need:
            #     share_a = w_a(1+s_a) / [w_a(1+s_a) + w_b(1+s_b)]
            # Equalizing the drives alone (the 'absolute' form below) targets
            # EQUAL RELATIVE DISTANCE FROM TARGET, which is not the same as a
            # good allocation: at the measured healthy point the drives are
            # 0.176/0.160, so that form lands on share_bwd 0.52 -- replay at
            # ~0.43 of the batch, inside the 0.29-0.40 range where every
            # ratcheted dev run rang. Anchoring on the idle mix instead makes
            # the healthy fixed point the known-good mix and the controller a
            # BOUNDED PERTURBATION around it. Both sides at target gives
            # exactly w, so the "nothing to arbitrate" case is the continuous
            # limit of this rule rather than a separate branch that the split
            # discontinuously snaps to.
            if w_a + w_b <= 0.0:
                self.ctrl['prop_target'] = frac_a / total  # nothing to aim at: hold
                return
            num_a, num_b = w_a * (1.0 + s_a), w_b * (1.0 + s_b)
            target = num_a / (num_a + num_b)
        elif s_a + s_b > 0.0:
            target = s_a / (s_a + s_b)
        else:
            # BOTH at or under target: nothing left to arbitrate. Relax toward
            # the idle split rather than freezing wherever the path happened to
            # end -- the split at the moment of satisfaction is an artifact of
            # the route taken, not a designed state, and if one side later
            # degrades the controller should be starting from a sane mix rather
            # than an extreme.
            if w_a + w_b <= 0.0:
                self.ctrl['prop_target'] = frac_a / total  # nothing to aim at: hold
                return
            target = w_a / (w_a + w_b)
        target = min(max(target, floor), 1.0 - floor)
        # ABSOLUTE ceilings, converted into bounds on mode_a's share of the pair
        # (mode_b's ceiling is a LOWER bound on mode_a's share). Applied to the
        # aim AND to the post-EMA share: clamping the target alone bounds the
        # fixed point but not a share that arrives out of bounds from a resumed
        # checkpoint. Infeasible bounds are validated away at parse time; the
        # lo <= hi guard is belt-and-braces against a pinned frac drifting.
        caps = bal.get('max_fracs') or {}
        lo, hi = 0.0, 1.0
        if caps:
            hi = min(hi, float(caps.get(mode_a, 1.0)) / total)
            lo = max(lo, 1.0 - float(caps.get(mode_b, 1.0)) / total)
            if lo <= hi:
                target = min(max(target, lo), hi)
        share_a = (1.0 - alpha) * (frac_a / total) + alpha * target
        if caps and lo <= hi:
            share_a = min(max(share_a, lo), hi)
        setattr(m, f'{mode_a}_frac', share_a * total)
        setattr(m, f'{mode_b}_frac', (1.0 - share_a) * total)
        self.ctrl['prop_target'] = target
        self.ctrl['boost'] = mode_a if target > frac_a / total else mode_b

    @staticmethod
    def _logit(p, eps=1e-6):
        p = min(max(float(p), eps), 1.0 - eps)
        return math.log(p / (1.0 - p))

    def _share_interval(self, bal, mode_num, mode_den, pair):
        """Turn ABSOLUTE frac bounds into an interval on mode_num's SHARE OF
        THE PAIR, then into theta limits. A ceiling on one mode is a floor on
        the other, so either bound constrains both; naming both simply
        intersects. Shared by the two integrator laws so the two cannot drift
        apart on the direction of that conversion."""
        s_lo, s_hi = 0.0, 1.0
        bounds = bal.get('bounds') or {}
        if mode_num in bounds:
            s_lo = max(s_lo, bounds[mode_num][0] / pair)
            s_hi = min(s_hi, bounds[mode_num][1] / pair)
        if mode_den in bounds:
            s_lo = max(s_lo, 1.0 - bounds[mode_den][1] / pair)
            s_hi = min(s_hi, 1.0 - bounds[mode_den][0] / pair)
        if s_lo > s_hi:  # only reachable if a pinned frac drifted off its declared value
            s_lo = s_hi = 0.5 * (s_lo + s_hi)
        return self._logit(s_lo), self._logit(s_hi)

    def _ratio_tick(self, bal):
        """Hold the RATIO of the two split modes' error metrics at a setpoint,
        by integrating in the logit of the numerator mode's share:

            e     = log(v_num / v_den) - log(setpoint)
            theta <- clip(theta + clip(gain * k * e, +-max_step), th_lo, th_hi)
            share_num = sigmoid(theta)

        WHAT IT ASSUMES, AND WHY THAT IS THE WHOLE DESIGN. Each mode owns the
        error it is the instrument for -- replay owns fwd/over_coverage (the
        policy over-weighting its own support, delta > 0), bwd owns
        bwd/relative_under_wcen (buffer modes the policy under-weights,
        delta < 0). By the blindness bound each branch is exponentially blind
        to the other's half of the residual field, so the split is an
        allocation between two DISJOINT halves of one error, and the only
        thing that has to be declared is their exchange rate. `setpoint` is
        that exchange rate and it is the only judgement in the loop: 1.0 says
        the two halves should be equally large, 4.0 says the under side should
        settle at a quarter of the over side.

        WHY A RATIO AND NOT TWO BARS. kind: proportional and kind: constraint
        both carry one target/bar PER SIDE, which is one more number than the
        problem has: only their exchange rate moves the equilibrium, and the
        redundant dimension is free to be set wrong. Worse, both convert a
        metric to a drive through max(v/t - 1, 0), so a side sitting under its
        target contributes NOTHING and the loop silently goes one-sided --
        which is the live defect on this route (bwd's target was read off a
        metric it has since been swapped away from, pinning that drive at 0).
        A signed log-ratio has no clamp: both sides always contribute, the
        error changes sign rather than vanishing, and 'inert' is not a state
        this loop can enter.

        WHY LOGS. The two metrics are positive and their observed oscillation
        has constant RELATIVE amplitude -- measured at ~2x peak-to-trough while the
        level itself fell 10x -- i.e. the noise is multiplicative. In logs that
        cycle is a symmetric additive perturbation the integrator averages to
        zero; in linear units the same cycle biases the mean.

        SIGN. Raising the numerator mode's share reduces its own metric and
        raises the other's, so rho falls as theta rises: e > 0 must RAISE
        theta for negative feedback. Getting this backwards is a runaway to a
        bound, which is why `numerator` is declared rather than inferred.

        RATE. theta moves at most gain*max_step per tick at 10 train steps per
        tick, so the time constant must exceed the ~1-2k-step absorption cycle
        (bwd dragging in buffer states degrades forward calibration before
        recovering, and a controller fast enough to react to that fights the
        mechanism instead of the pathology).

        CONVERGENCE GATE. A scale-free setpoint demands the same ratio whether
        the halves are 20 nats apart or 2. converge_floor fades the gain to
        zero as the larger metric approaches it, so the loop steers while
        there is a real imbalance and retires itself once both halves are
        inside the convergence scale -- where the objective it is a proxy for
        has gone flat anyway.
        """
        m = self.m
        for mode, value in (bal.get('pinned') or {}).items():
            setattr(m, f'{mode}_frac', float(value))
        mode_n = bal['numerator']
        mode_d = next(mode for mode in bal['metrics'] if mode != mode_n)
        v_n, v_d = self._resolve(bal['metrics'][mode_n]), self._resolve(bal['metrics'][mode_d])
        frac_n, frac_d = getattr(m, f'{mode_n}_frac'), getattr(m, f'{mode_d}_frac')
        pair = frac_n + frac_d
        if pair <= 0:
            return
        th_lo, th_hi = self._share_interval(bal, mode_n, mode_d, pair)

        theta = self.ctrl.get('rt_theta')
        if theta is None:
            # seed from where the stage actually entered, so tick 0 of the
            # controller is bit-identical to the matching fixed-mix arm
            theta = self._logit(frac_n / pair)
        theta = min(max(float(theta), th_lo), th_hi)

        # a non-positive reading is a degenerate metric, not a perfect one
        # (relative_under_wcen goes nan when no sample clears the reward ramp,
        # and 0 would send log to -inf): hold the actuator, do not guess
        if v_n is None or v_d is None or v_n <= 0.0 or v_d <= 0.0:
            # rt_hold, not rt_gain_scale = 0: the two reasons the loop stops
            # moving are 'converged past converge_floor' and 'the sensor is
            # not reporting', and collapsing them onto one zero is exactly the
            # ambiguity this controller exists to avoid. rt_gain_scale keeps
            # its last real value rather than being overwritten with a reading
            # that was never taken.
            self.ctrl['rt_theta'] = theta
            self.ctrl['rt_hold'] = 1.0
            return
        rho = v_n / v_d
        err = math.log(rho) - math.log(bal['setpoint'])
        floor = bal.get('converge_floor')
        k = 1.0 if not floor else min(max((max(v_n, v_d) - floor) / floor, 0.0), 1.0)
        step = bal['gain'] * k * err
        step = min(max(step, -bal['max_step']), bal['max_step'])
        # clamping theta IS the anti-windup: the integrator state is the
        # actuator, so a bounded theta cannot accumulate unrealizable demand
        theta = min(max(theta + step, th_lo), th_hi)

        share_n = 1.0 / (1.0 + math.exp(-theta))
        setattr(m, f'{mode_n}_frac', share_n * pair)
        setattr(m, f'{mode_d}_frac', (1.0 - share_n) * pair)
        self.ctrl['rt_theta'] = theta
        self.ctrl['rt_rho'] = rho
        self.ctrl['rt_err'] = err
        self.ctrl['rt_gain_scale'] = k
        self.ctrl['rt_hold'] = 0.0
        self.ctrl['rt_at_bound'] = (-1.0 if theta <= th_lo + 1e-9
                                    else 1.0 if theta >= th_hi - 1e-9 else 0.0)
        self.ctrl['boost'] = mode_n if step > 0 else mode_d

    def _constraint_tick(self, bal):
        """Split two modes' combined mass by INTEGRATING a one-sided
        constraint violation against a one-sided objective shortfall, in the
        logit of one side's share:

            d_c = max(metric_c / bar_c - 1, 0)     # constrained side
            d_r = max(metric_r / bar_r - 1, 0)     # best-effort side
            theta <- clip(theta + clip(gain * (d_r - priority * d_c),
                                       +-max_step),  theta_lo, theta_hi)
            share_r = sigmoid(theta)

        WHY AN INTEGRATOR, AND NOT THE PROPORTIONAL MAP. kind: proportional is
        a static function of the metrics: the split it lands on is set by where
        the two targets sit relative to each other, so a target that is merely
        a guess produces a confidently wrong equilibrium and nothing ever
        corrects it. That is a real cost here and not a hypothetical -- this
        stage's own config carried `targets.bwd: 3.0` annotated UNCALIBRATED
        for a metric it had already been swapped away from. An integrator moves
        the actuator whenever a drive is nonzero and holds it when both are
        zero, so the equilibrium is a property of the BARS, not of their ratio,
        and only bars that are actually reachable have to be right.

        WHY ASYMMETRIC. The two sides are not the same kind of quantity. The
        constrained one (bwd absorbing the buffer's modes) is a level we can
        state a priori -- relative_under_wcen ~ 2 means "no mode is badly
        under-weighted" -- while the best-effort one (over_coverage on fresh
        forward samples) has no known reachable level. `priority` (a gain
        MULTIPLE, not a switch) makes the constraint win contests without ever
        handing it the whole batch, so this is a soft lexicographic order:
        there is no discontinuity for the split to limit-cycle on, which is
        what the rule-based controller did in replay_july26.

        WHAT AN UNREACHABLE BAR DOES. If the best-effort bar is set below its
        metric's floor -- expected, since over_coverage's floor is unknown --
        its drive never reaches zero and theta walks up until either the
        constrained side pushes back or theta hits its bound. Both outcomes are
        the intended answer to "take as good as we can get provided the
        constraint holds", and both are legible in the log: prop of the run
        spent at a bound is reported as cs_at_bound. The bound is therefore
        load-bearing -- see the `bounds` validation for why it is required.

        RATE. theta moves at most gain*max_step per tick and ticks every 10
        train steps. Sized so a typical contest traverses the full bounded
        range in ~1000-2000 steps: slower than the ~200-step proportional
        default (which is far quicker than the plant), and slower than the
        1-2k-step absorption cycle in the direction that STARVES bwd, while
        `priority` lets the restoring direction move ~3x faster. That asymmetry
        matches the failure asymmetry: starving bwd collapses coverage fast but
        recovers fast, whereas starving replay/fwd degrades the policy slowly
        and recovers slowly.
        """
        m = self.m
        for mode, value in (bal.get('pinned') or {}).items():
            setattr(m, f'{mode}_frac', float(value))
        mode_c = bal['constrain']
        mode_r = next(mode for mode in bal['metrics'] if mode != mode_c)
        v_c, v_r = self._resolve(bal['metrics'][mode_c]), self._resolve(bal['metrics'][mode_r])
        frac_c, frac_r = getattr(m, f'{mode_c}_frac'), getattr(m, f'{mode_r}_frac')
        pair = frac_c + frac_r
        if pair <= 0:
            return

        th_lo, th_hi = self._share_interval(bal, mode_r, mode_c, pair)

        theta = self.ctrl.get('cs_theta')
        if theta is None:
            # seed from where the stage actually entered, so step 0 of the
            # controller is bit-identical to the matching fixed-mix arm
            theta = self._logit(frac_r / pair)
        theta = min(max(float(theta), th_lo), th_hi)

        if v_c is None or v_r is None:
            # absent/non-finite metric (relative_under_wcen goes nan when no
            # sample clears the reward ramp): hold the actuator, do not guess
            self.ctrl['cs_theta'] = theta
            return
        d_c = max(v_c / float(bal['bars'][mode_c]) - 1.0, 0.0)
        d_r = max(v_r / float(bal['bars'][mode_r]) - 1.0, 0.0)
        step = bal['gain'] * (d_r - bal['priority'] * d_c)
        step = min(max(step, -bal['max_step']), bal['max_step'])
        # clamping theta IS the anti-windup: the integrator state is the
        # actuator, so a bounded theta cannot accumulate unrealizable demand
        theta = min(max(theta + step, th_lo), th_hi)

        share_r = 1.0 / (1.0 + math.exp(-theta))
        setattr(m, f'{mode_r}_frac', share_r * pair)
        setattr(m, f'{mode_c}_frac', (1.0 - share_r) * pair)
        self.ctrl['cs_theta'] = theta
        self.ctrl['cs_drive'] = {mode_c: d_c, mode_r: d_r}
        self.ctrl['cs_at_bound'] = (-1.0 if theta <= th_lo + 1e-9
                                    else 1.0 if theta >= th_hi - 1e-9 else 0.0)
        self.ctrl['boost'] = mode_r if step > 0 else mode_c

    # ---------------------------------------------------------- buffer servo

    def _buffer_servo_tick(self):
        """Hold the replay buffer on the healthy side of the train/test
        crossover by moving its FRESHNESS, not its loss weight.

        SENSOR: ratio = replay/scatter_err over fwd/scatter_err. Replay draws
        are a |resid|-prioritized resample of stored forward rollouts, so a
        replay batch is by construction the HARD tail of the forward
        distribution and its residual spread should exceed fresh forward's --
        the observed healthy value is ~2x. The ratio crossing below 1 says the
        policy fits reused stored trajectories BETTER than the fresh draws they
        were selected from, which is memorization of the buffer's contents and
        nothing else. Both branches now carry policy gradient (fwd runs
        freeze_policy 0 in this route), so the two sides differ only in their
        sampler, which is what makes the ratio a clean generalization gap
        rather than a train-vs-heldout artifact.

        ACTUATOR: one multiplicative boost B applied as churn_rate x B and
        mean_residence_steps / B. In steady state (train.py manage_replay_buffer,
        Little's law) occupancy = churn_rate x mean_residence_steps and
        draws_per_row = batch_size / churn_rate, so this leaves OCCUPANCY
        exactly invariant and moves only reuse (1/B) and policy lag (1/B). One
        knob with one invariant is deliberate: churn_rate, mean_residence_steps
        and max_size are three handles on the same steady state, and moving
        them independently is how a buffer ends up in a corner nobody meant.
        toxic_min_draws rides 1/B too, because it is defined relative to the
        expected number of draws a row sees and would otherwise silently change
        meaning as B moves.

        WHY THIS IS A SECOND CONTROLLER AND NOT A BALANCE RULE. Loss weights
        cannot fix overfitting. Down-weighting replay trains less on a
        memorized buffer; it does not make the buffer less memorized, and it
        also gives up the residual-tail correction replay exists to provide.
        Freshness is the actuator that acts on the cause, so it gets its own
        loop -- and the two loops are near-orthogonal by construction (this one
        holds occupancy fixed and changes no frac; the balance controller
        changes fracs and touches no buffer knob).

        DEADBAND AND RELEASE. Tighten below `bar`, release above `release`,
        hold in between, with `relax` < 1 making release the slower direction.
        Each side's drive is the deviation normalized by `scale` and saturated
        at 1, so the rate is set by the CONFIGURED ramp speed rather than by
        how deep the excursion happens to be -- see the `scale` note in
        _parse_buffer_servo for why the raw deficit is unusable here.
        The deadband keeps the servo off during normal ratio jitter, and the
        release term is what keeps it from being a ratchet whose fixed point is
        max_boost -- the same one-way-anneal failure mode as
        controller-ratchet-marginal-breach. Cost of over-churning is real
        (admission work rises with B, and at high B replay degenerates into an
        on-policy duplicate of fwd), which is why release exists at all.
        """
        spec = self.stage.buffer_servo
        if spec is None:
            # a previous stage's boost must not survive into a stage that
            # declares no servo (stage_ctrl resets, so nothing else would undo it)
            if self._rb_base is not None:
                self._apply_buffer_boost(1.0)
            return
        num, den = self._resolve(spec['numerator']), self._resolve(spec['denominator'])
        if num is None or den is None or den <= 0.0:
            return  # cold start (replay has not trained yet): hold at the configured buffer
        ratio = num / den
        sc = spec['scale']
        drive = (min(max(spec['bar'] - ratio, 0.0) / sc, 1.0)
                 - spec['relax'] * min(max(ratio - spec['release'], 0.0) / sc, 1.0))
        step = min(max(spec['gain'] * drive, -spec['max_step']), spec['max_step'])
        log_boost = float(self.ctrl.get('bs_log_boost', 0.0)) + step
        log_boost = min(max(log_boost, 0.0), math.log(spec['max_boost']))
        self.ctrl['bs_log_boost'] = log_boost
        self.ctrl['bs_ratio'] = ratio
        self._apply_buffer_boost(math.exp(log_boost))

    def _apply_buffer_boost(self, boost):
        """Write the live replay-buffer knobs as base x boost. train.py reads
        these off args on every manage call, so the change takes effect on the
        next churn with no plumbing."""
        rb = self.m.args.buffers.replay_buffer
        # `toxic_min_draws` used to be scaled here too. It is a DELETED
        # retirement (utils._RETIRED_KEYS), so no config can set it: the base
        # captured 0.0 every time and the branch that scaled it was unreachable.
        if self._rb_base is None:
            self._rb_base = {'churn_rate': float(rb.churn_rate),
                             'mean_residence_steps': float(rb.mean_residence_steps)}
        base = self._rb_base
        rb.churn_rate = max(1, int(round(base['churn_rate'] * boost)))
        # floor at 2 steps: below that the hazard evicts essentially the whole
        # buffer every call and replay stops being a buffer at all
        rb.mean_residence_steps = max(2.0, base['mean_residence_steps'] / boost)

    def _nudge_mode_fracs(self, boost):
        """EMA nudge of the fracs toward a target split, with PER-MODE floors
        -- the stage's explicit min_fracs where given, controller.min_mode_frac
        otherwise. A floor at or above the stage's deactivate threshold keeps
        that branch always computed; below it, the mode can go truly dormant.
        `boost` is a mode name (one-hot target, every rule) or a normalized
        {mode: weight} mix (idle default)."""
        m = self.m
        ctrl = m.args.controller
        probs = np.array([m.fwd_frac, m.bwd_frac, m.replay_frac], dtype=float)
        probs /= probs.sum()
        weights = boost if isinstance(boost, dict) else {boost: 1.0}
        target = np.array([weights.get('fwd', 0.0), weights.get('bwd', 0.0),
                           weights.get('replay', 0.0)], dtype=float)
        floors = np.array([self.stage.min_fracs.get(mode, ctrl.min_mode_frac)
                           for mode in ('fwd', 'bwd', 'replay')], dtype=float)
        free = 1.0 - floors.sum()  # > 0: floors validated to sum below 1
        excess = np.clip(probs - floors, 0.0, None)
        s = excess.sum()
        excess = excess * (free / s) if s > 0.0 else np.full(3, free / 3.0)
        excess = (1.0 - ctrl.beta) * excess + ctrl.beta * free * target
        m.fwd_frac, m.bwd_frac, m.replay_frac = floors + excess

    # ---------------------------------------------------------------- logging

    def report(self) -> dict:
        """Loggable (numeric-only) view of the engine: chosen boost, per-rule
        live thresholds and elevations, exit-term streaks. The stage itself is
        logged as metrics['phase'] (its 1-based index) for wandb continuity."""
        stage = self.stage
        out = {}
        boost = self.ctrl.get('boost')
        kind = stage.balance['kind'] if stage.balance is not None else None
        if boost is not None and kind != 'ratio':
            # under kind: ratio this is just sign(rt_err) recoded as a 3-valued
            # categorical, i.e. strictly less information than rt_err itself.
            # It stays for the rule-based kinds, where WHICH mode won a contest
            # is not recoverable from any other series.
            out['protocol/boost'] = {'fwd': 0, 'bwd': 1, 'replay': 2}[boost]
        if stage.balance is not None and stage.balance['kind'] == 'lexicographic':
            # only _balance_tick's lexicographic path ever writes this, so
            # emitting it unconditionally publishes a constant 0 on every other
            # kind -- a flat series that looks like a reading
            out['protocol/anneal_streak'] = self.ctrl.get('anneal_streak', 0)
        if stage.balance is not None and stage.balance['kind'] == 'proportional':
            # the split the controller is steering toward (share of the two
            # modes' COMBINED mass going to the first), alongside each side's
            # target-offset drive -- so a stalled split can be read as either
            # 'both at target' (drives ~0) or 'alpha too slow'
            if 'prop_target' in self.ctrl:
                out['protocol/prop_target'] = self.ctrl['prop_target']
            targets = stage.balance.get('targets') or {}
            scale = float(self.ctrl.get('prop_scale', 1.0))
            if stage.balance.get('anneal'):
                out['protocol/prop_scale'] = scale
                out['protocol/prop_streak'] = int(self.ctrl.get('prop_streak', 0))
            for mode, metric in stage.balance['metrics'].items():
                live = float(targets.get(mode, 0.0)) * scale
                out[f'protocol/prop_bar_{mode}'] = live  # the live target, in the metric's units
                v = self._resolve(metric)
                if v is not None:
                    # drive in the stage's declared form -- dimensionless under
                    # 'relative', so the two modes' drives are directly
                    # comparable on one axis and the split is readable as
                    # drive_a / (drive_a + drive_b)
                    out[f'protocol/prop_drive_{mode}'] = self._drive(
                        v, live, stage.balance.get('drive'))
        if stage.balance is not None and stage.balance['kind'] == 'constraint':
            # theta is the integrator state AND the actuator, so it is the one
            # series that says what the controller is doing; cs_at_bound says
            # whether it is still steering (0) or has parked against a bound
            # (+-1), which is the difference between "converged" and
            # "the bar was unreachable and the bound is the answer"
            if self.ctrl.get('cs_theta') is not None:
                theta = float(self.ctrl['cs_theta'])
                out['protocol/cs_theta'] = theta
                out['protocol/cs_share'] = 1.0 / (1.0 + math.exp(-theta))
            if 'cs_at_bound' in self.ctrl:
                out['protocol/cs_at_bound'] = self.ctrl['cs_at_bound']
            for mode, d in (self.ctrl.get('cs_drive') or {}).items():
                # dimensionless (metric/bar - 1), so the two are directly
                # comparable and the contest is readable as d_r vs priority*d_c
                out[f'protocol/cs_drive_{mode}'] = d
            for mode, bar in stage.balance['bars'].items():
                out[f'protocol/cs_bar_{mode}'] = float(bar)
        if stage.balance is not None and stage.balance['kind'] == 'ratio':
            # rt_err is the whole loop in one series: SIGNED, never clamped, so
            # unlike a one-sided drive it cannot read the same when satisfied
            # and when welded off. Sign says which mode is being fed, magnitude
            # says by how many nats of log-ratio the split is off its setpoint,
            # and it crossing zero is the equilibrium.
            if self.ctrl.get('rt_theta') is not None:
                theta = float(self.ctrl['rt_theta'])
                out['protocol/rt_theta'] = theta
                out['protocol/rt_share'] = 1.0 / (1.0 + math.exp(-theta))
            for key in ('rt_rho', 'rt_err', 'rt_gain_scale', 'rt_at_bound', 'rt_hold'):
                if key in self.ctrl:
                    out[f'protocol/{key}'] = float(self.ctrl[key])
            out['protocol/rt_setpoint'] = float(stage.balance['setpoint'])
            # both metrics in their own units next to the ratio they form, so a
            # gain_scale that has faded to 0 can be read as 'converged' rather
            # than 'sensor died'
            for mode, metric in stage.balance['metrics'].items():
                v = self._resolve(metric)
                if v is not None:
                    out[f'protocol/rt_metric_{mode}'] = v
        if stage.buffer_servo is not None:
            out['protocol/bs_boost'] = math.exp(float(self.ctrl.get('bs_log_boost', 0.0)))
            if 'bs_ratio' in self.ctrl:
                out['protocol/bs_ratio'] = self.ctrl['bs_ratio']
            # The ACTUATOR, not just the sensor. Without this a servo that is
            # reading fine but has no authority looks identical to one that is
            # correctly holding -- the S2 drive-liveness failure shape again.
            if 'bs_log_boost' in self.ctrl:
                out['protocol/bs_log_boost'] = self.ctrl['bs_log_boost']
            rb = self.m.args.buffers.replay_buffer
            # the LIVE knobs, so the servo's effect is visible next to its
            # sensor rather than having to be recomputed from boost x base
            out['protocol/bs_churn_rate'] = float(rb.churn_rate)
            out['protocol/bs_residence'] = float(rb.mean_residence_steps)
        if stage.balance is not None and stage.balance['kind'] == 'lexicographic':
            for i, rule in enumerate(stage.balance['rules']):
                rs = self.ctrl['rules'].get(i, {})
                tag = rule['metric'].replace('/', '_')
                # every rule reports its bar IN THE METRIC'S OWN UNITS, so
                # thr_* overlays directly on the metric it gates and tracks
                # down with it: the live annealed threshold for 'above' rules,
                # the static floor for 'below' rules, and best x margin for
                # relative ones. (Relative rules used to report the raw
                # `margin` -- a dimensionless config constant that never
                # moves, so the plot showed a flat 1.3 line next to a metric
                # in nats. The running best is also exported so the bar and
                # the floor it rides on are both visible.)
                if 'thr' in rs:
                    out[f'protocol/thr_{tag}'] = rs['thr']
                elif 'below' in rule:
                    out[f'protocol/thr_{tag}'] = float(rule['below'])
                elif 'relative' in rule and rs.get('best') is not None:
                    out[f'protocol/thr_{tag}'] = rs['best'] * float(rule.get('margin', 1.3))
                    out[f'protocol/best_{tag}'] = rs['best']
                if 'elevation' in rs:
                    out[f'protocol/elev_{tag}'] = rs['elevation']
            for name in stage.balance['anneal_coeffs']:
                rs = self.ctrl['coeffs'].get(name)
                if rs and 'val' in rs:
                    out[f'protocol/coeff_{name}'] = rs['val']
        if stage.exit:
            for i, term in enumerate(stage.exit):
                tag = term['metric'].replace('/', '_')
                out[f'protocol/exit_streak_{tag}'] = self.ctrl['exit'].get(i, 0)
                # AGE, next to the streak, because the streak alone is
                # ambiguous in exactly the way that cost a battery. A flat zero
                # reads as "this condition never passes"; on the eval/* terms it
                # used to mean "this condition is never JUDGED here", and
                # `next_battery.md` 1.3 read the first from the second. Age
                # separates them: a streak stuck at 0 with age climbing is a
                # metric nobody is writing, a streak stuck at 0 with age at the
                # metric's own cadence is a bar that genuinely is not met.
                last = self.ctrl.get('exit_seen', {}).get(i)
                out[f'protocol/exit_age_{tag}'] = (
                    -1.0 if last is None else float(int(self.m.step_ind) - int(last)))
        return out
