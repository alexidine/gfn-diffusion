"""
Route detection and key resolution -- the file that tracks metric names and what
they MEAN on each route.

This is the package's one coupling point to the training code. Every metric-name
string literal belongs here, so a rename upstream is a one-file change. If you
find yourself typing `'fwd/'` in another module of this package, stop.

THE CENTRAL RULE (spec H2). A metric resolves to one of THREE states, and the
report must distinguish all three:

    LIVE      present, and meaningful on this route
    ABSENT    not logged by this run
    NA_ROUTE  logged, carrying numbers, and NOT meaningful on this route

Never collapse NA_ROUTE into ABSENT, and never render it as zero. On the
conditional VarGrad route the log Z and TB series exist and are populated;
reading them as one would on a TB run is wrong, and a hole in the report is far
safer than a number that invites the wrong reading.

CONFIG STRUCTURE, verified by inspection rather than assumed. wandb stores the
config twice and only one form is usable:

  * `protocol` -> {'value': "Namespace(stages=[{...}])"} -- a REPR STRING. Not
    parseable, and not to be eval'd.
  * flattened scalars -- `protocol_stages_0_name`,
    `protocol_stages_1_loss_coeffs_fwd_tb`, `bwd_loss_coeffs_vg_lb`, ... These
    are what this module reads.

The cloud API (`wandb.Api().run(...).config`) returns the same flattened dict, so
one reader serves both sources.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional

# Loss modes, in the order the trainer names them.
MODES = ('fwd', 'bwd', 'replay')


class Route(str, Enum):
    """How a run is trained. Distinguishing these is what makes a topline
    meaningful; a single global list of keys is wrong on at least one route."""

    TB_UNCONDITIONAL = 'tb_unconditional'
    VARGRAD_CONDITIONAL = 'vargrad_conditional'
    MLE_PRIOR = 'mle_prior'
    UNKNOWN = 'unknown'


class KeyState(str, Enum):
    LIVE = 'live'
    ABSENT = 'absent'
    NA_ROUTE = 'na_route'


# ---------------------------------------------------------------------------
# Toplines
# ---------------------------------------------------------------------------
# TB/unconditional: reading_runs.md §2's six, verbatim. Two of them do not exist
# under these names on real runs -- `bwd/under_coverage_wcen` is logged as
# `bwd/under_coverage`, and `log_Z_learned` is namespaced. They are kept as
# written so `resolve` reports them as RENAMES rather than holes; correcting the
# list here would hide the discrepancy from the doc.
TOPLINE_TB = (
    'fwd/tb_err_worst', 'bwd/tb_err_worst', 'replay/tb_err_worst',
    'fwd/over_coverage',
    'bwd/under_coverage_wcen',
    'fwd/tb_resid_clipped',
    'log_Z_learned',
    'zmatch/delta_worst',
)

# Conditional VarGrad. The TB topline does not transfer: its log Z and TB terms
# are the ones that do not track here. Four families, per the user's call:
#   1. the dispersion the VG objective actually minimises
#   2. the worst-case per-condition Z mismatch (upper tail, not pooled)
#   3. per-condition fractions -- Cond * Spread form ONLY (a thresholded
#      per-condition fraction is biased by n_c and does not compare across
#      streams)
#   4. held-out eval (R17): read `eval_test` BEFORE `eval_fwd`, because train
#      r2/tb_err/scatter_err can all improve on the same evaluation where the
#      held-out set blows up
# The per-parameter subtleties within these are deliberately NOT encoded -- they
# are judgment, and this package does not emit judgment.
TOPLINE_VARGRAD = (
    'eval_test/logw_std_within', 'eval_test/over_coverage', 'eval_test/jensen_z',
    'fwd/logw_std_within', 'bwd/logw_std_within',
    'fwd/vg_lb', 'bwd/vg_lb',
    'zmatch/delta_worst', 'zmatch/delta_mean',
    'bwd/condition_log_z_visited_frac',
)

TOPLINE_MLE = (
    'bwd/mle', 'bwd/tbc', 'gates/mle_flat', 'eval/wass_debiased',
    'bwd/logw_std_within',
)

TOPLINE = {
    Route.TB_UNCONDITIONAL: TOPLINE_TB,
    Route.VARGRAD_CONDITIONAL: TOPLINE_VARGRAD,
    Route.MLE_PRIOR: TOPLINE_MLE,
    Route.UNKNOWN: TOPLINE_TB,
}

# Series that EXIST and are not meaningful, per route. Deliberately narrow: it
# names exactly the families reported as not tracking on the VarGrad route (log Z
# level, and TB fit/residual). Over-marking here hides real data, which is the
# same failure as under-marking, in the other direction. Widen only on a stated
# reason, not on suspicion.
_NA_PATTERNS = {
    Route.VARGRAD_CONDITIONAL: (
        r'(^|/)log_Z_learned$',
        r'(^|/)tb_err(_worst)?$',
        r'(^|/)tb_resid(_clipped)?$',
    ),
}

# Read order for the report -- reading_runs.md §1, which is a floor for someone
# with no priors rather than an algorithm.
READ_ORDER = (
    ('STAGE', ('protocol/stage_index', 'gates/mle_flat')),
    ('FIT', ('fwd/tb_err_worst', 'bwd/tb_err_worst', 'replay/tb_err_worst',
             'fwd/scatter_err', 'bwd/scatter_err', 'replay/scatter_err')),
    ('PARTITION', ('bwd/log_Z_learned', 'fwd/jensen_z')),
    ('ALLOCATION', ('Fwd Frac', 'Bwd Frac', 'Replay Frac', 'lr_fused', 'lr_policy')),
    ('VARIANCE', ('fwd/logw_std_within', 'bwd/logw_std_within', 'fwd/vg_lb',
                  'bwd/vg_lb', 'fwd/z_gap', 'zmatch/delta_mean', 'zmatch/delta_worst')),
    ('COVERAGE', ('fwd/over_coverage', 'bwd/under_coverage', 'fwd/tb_resid_clipped')),
    ('HELD_OUT', ('eval_test/logw_std_within', 'eval_test/over_coverage',
                  'eval_test/jensen_z', 'eval_test/cond_tb_err')),
    ('SDE', ('fwd/step_var', 'fwd/terminal_var')),
    ('COST', ('gpu/util_recent', 'gpu/util_policy', 'batch_size', 'train_step_time')),
)

# EMA-derived: show the trend, suppress significance. No trend test is valid on a
# smoothed series. `tracker/logw_std_rms` and `tracker/z_bias_rms` are separately
# flagged low-trust -- carried, never ranked on.
EMA_PREFIXES = ('tracker/',)
LOW_TRUST = ('tracker/logw_std_rms', 'tracker/z_bias_rms')

# Positive series where runaway growth is the thing being watched for.
ESCAPE_KEYS = ('fwd/step_var', 'fwd/terminal_var', 'bwd/step_var')


# ---------------------------------------------------------------------------
# Config reading
# ---------------------------------------------------------------------------

def _value(config: dict, key: str):
    """One config entry. Local `config.yaml` wraps each value as
    {'value': ...}; the cloud API returns it bare. Handle both."""
    v = config.get(key)
    if isinstance(v, dict) and 'value' in v:
        return v['value']
    return v


def stage_names(config: dict) -> list[str]:
    """Declared protocol stages, in order, from the flattened keys."""
    pat = re.compile(r'^protocol_stages_(\d+)_name$')
    found = []
    for k in config:
        m = pat.match(k)
        if m:
            found.append((int(m.group(1)), _value(config, k)))
    return [n for _, n in sorted(found)]


def effective_loss_coeffs(config: dict, stage_index: int) -> dict[str, dict[str, Any]]:
    """Base `<mode>_loss_coeffs_*` overlaid with stage `stage_index`'s overrides.

    The overlay is the whole point: the base blocks are structural and a stage
    turns things on. Classifying a route from the base alone reads every run as
    whatever the defaults say, which on this config is 'no TB, no VarGrad'."""
    out = {m: {} for m in MODES}
    base = re.compile(r'^(%s)_loss_coeffs_(.+)$' % '|'.join(MODES))
    over = re.compile(r'^protocol_stages_%d_loss_coeffs_(%s)_(.+)$'
                      % (stage_index, '|'.join(MODES)))
    for k in config:
        m = base.match(k)
        if m:
            out[m.group(1)][m.group(2)] = _value(config, k)
    for k in config:  # second pass: overrides win regardless of dict order
        m = over.match(k)
        if m:
            out[m.group(1)][m.group(2)] = _value(config, k)
    return out


def stage_train_mode(config: dict, stage_index: int) -> Optional[str]:
    """A stage's `train_mode`: 'fwd', 'bwd', 'replay' or 'fused'."""
    return _value(config, f'protocol_stages_{stage_index}_train_mode')


def active_modes(config: dict, stage_index: int) -> tuple[str, ...]:
    """The loss branches a stage ACTUALLY runs.

    Without this filter, route detection reads coefficients belonging to modes
    the stage never evaluates. Concretely: `train_prior` is `train_mode: bwd`,
    and the canonical config's BASE `replay_loss_coeffs_tb` is 1.0 -- so a naive
    reading sees TB active and classifies the MLE warm-start stage as the TB
    route, which is the one route whose topline is least applicable to it.

    'fused' runs all three. An unknown or absent mode falls back to all three,
    which over-reports rather than under-reports: a spurious extra coefficient
    makes the route UNKNOWN or too general, while missing a real one silently
    picks the wrong topline."""
    mode = stage_train_mode(config, stage_index)
    if mode in MODES:
        return (mode,)
    return MODES


def is_conditional(config: dict) -> bool:
    return bool(_value(config, 'vector_conditioning')
                or _value(config, 'molecule_conditioning'))


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def detect_route(config: dict, stage_index: Optional[int] = None) -> Route:
    """Classify a run's training route from its effective loss coefficients.

    `stage_index` defaults to the LAST declared stage, which is the terminal one
    a long run spends nearly all its time in. Pass an index to classify a
    specific stage -- the route genuinely differs between them, which is why
    `current_stage` exists.

    Order matters. VarGrad is checked before TB because a stage can carry both
    (VarGrad as the policy loss, TB for Z), and the VarGrad leg is what makes the
    TB reporting not track. MLE is last: `train_prior` is the only stage that
    turns it on, and a run sitting there is genuinely on the prior route."""
    stages = stage_names(config)
    if stage_index is None:
        stage_index = max(len(stages) - 1, 0)
    coeffs = effective_loss_coeffs(config, stage_index)
    live = active_modes(config, stage_index)

    any_vg = any(_num(coeffs[m].get('vg_lb')) > 0 or _num(coeffs[m].get('vg_lme')) > 0
                 for m in live)
    any_tb = any(_num(coeffs[m].get('tb')) > 0 for m in live)
    any_mle = ('bwd' in live
               and (_num(coeffs['bwd'].get('mle')) > 0
                    or _num(coeffs['bwd'].get('tbc')) > 0))

    if any_vg:
        # Checked FIRST, and before TB. A VarGrad leg is what makes the log Z and
        # TB reporting stop tracking, and a stage can carry both (VarGrad as the
        # policy loss, TB for Z). Classifying on TB when VarGrad is live hands
        # back a topline whose central terms are the untrustworthy ones.
        return (Route.VARGRAD_CONDITIONAL if is_conditional(config)
                else Route.UNKNOWN)
    if any_mle and not any_tb:
        return Route.MLE_PRIOR
    if any_tb:
        return Route.TB_UNCONDITIONAL
    return Route.UNKNOWN


# The metric carrying protocol position, and its offset. Found by inspecting a
# live run rather than assumed -- the plausible-looking `protocol/stage_index`
# does not exist. `phase` is ONE-BASED: train.py's property returns
# `self.protocol.stage.index + 1`, so phase 2 is stages[1]. Getting this wrong
# silently reports every run as one stage behind.
STAGE_METRIC = 'phase'
STAGE_METRIC_OFFSET = 1


def current_stage(summary: dict, config: dict) -> Optional[str]:
    """The stage a run is in, by NAME, or None if it cannot be determined.

    Resolved from the run's own record, never defaulted. Falling back to the last
    declared stage would silently mislabel every run that died in phase 1 as
    having reached the terminal one -- and those are precisely the runs being
    read to find out why they stopped."""
    names = stage_names(config)
    if not names or STAGE_METRIC not in summary:
        return None
    try:
        i = int(summary[STAGE_METRIC]) - STAGE_METRIC_OFFSET
    except (TypeError, ValueError):
        return None
    return names[i] if 0 <= i < len(names) else None


def current_stage_index(summary: dict, config: dict) -> Optional[int]:
    """Zero-based stage index, for `detect_route`. None when unknown."""
    names = stage_names(config)
    if not names or STAGE_METRIC not in summary:
        return None
    try:
        i = int(summary[STAGE_METRIC]) - STAGE_METRIC_OFFSET
    except (TypeError, ValueError):
        return None
    return i if 0 <= i < len(names) else None


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Resolution:
    wanted: str
    state: KeyState
    resolved_to: Optional[str] = None   # set when a fuzzy match renamed it
    note: str = ''

    @property
    def key(self) -> Optional[str]:
        """The name to actually request from history, or None."""
        if self.state is KeyState.LIVE:
            return self.resolved_to or self.wanted
        return None

    def __str__(self):
        if self.state is KeyState.LIVE and self.resolved_to:
            return f'{self.wanted:34s} LIVE      -> {self.resolved_to}'
        return f'{self.wanted:34s} {self.state.value.upper():9s} {self.note}'


def _is_na_on_route(key: str, route: Route) -> bool:
    return any(re.search(p, key) for p in _NA_PATTERNS.get(route, ()))


def _namespace_candidates(wanted: str, available: set[str]) -> list[str]:
    """Keys whose tail matches an unnamespaced `wanted`, e.g. `log_Z_learned` ->
    ['bwd/log_Z_learned', 'fwd/log_Z_learned']."""
    if '/' in wanted:
        return []
    return sorted(k for k in available if k.rsplit('/', 1)[-1] == wanted)


def _fuzzy(wanted: str, available: set[str]) -> Optional[str]:
    """Best rename candidate for an absent key, or None.

    Two cheap structural rules first, because they cover the renames that
    actually occur here and a similarity score does not reliably prefer the
    right one:

      1. NAMESPACE ONLY -- bare `log_Z_learned` vs `bwd/log_Z_learned`. Match on
         the tail; require exactly one candidate, since `fwd/` and `bwd/` forms
         of the same tail are different quantities and picking one would be a
         guess.
      2. SUFFIX FAMILY -- `bwd/under_coverage_wcen` vs `bwd/under_coverage`. The
         `_wcen` / `_within` forms divide out a bias; the plain form is the same
         quantity without that correction, which is a rename worth reporting and
         a substitution worth making visible.

    Only then a similarity fallback, deliberately tight."""
    tail = wanted.rsplit('/', 1)[-1]
    if '/' not in wanted:
        cands = [k for k in available if k.rsplit('/', 1)[-1] == tail]
        if len(cands) == 1:
            return cands[0]
        return None

    ns = wanted.rsplit('/', 1)[0]
    for suffix in ('_wcen', '_within', '_worst', '_clipped'):
        if tail.endswith(suffix):
            cand = f'{ns}/{tail[:-len(suffix)]}'
            if cand in available:
                return cand

    close = difflib.get_close_matches(wanted, sorted(available), n=1, cutoff=0.9)
    return close[0] if close else None


def resolve(available: Iterable[str], wanted: Iterable[str],
            route: Route = Route.UNKNOWN) -> list[Resolution]:
    """Resolve wanted keys against what a run actually logged.

    ALWAYS call this before requesting history. `scan_history(keys=[...])`
    returns zero rows SILENTLY if any single requested key is absent -- no error,
    no warning -- so an unresolved key does not cost you that key, it costs you
    the entire pull. That is check number one in this package."""
    available = set(available)
    out = []
    for w in wanted:
        if _is_na_on_route(w, route):
            # Checked BEFORE presence: the defining property of this state is
            # that the key IS there. Testing presence first would file it as LIVE
            # and hand back a number that does not mean what it appears to.
            out.append(Resolution(
                w, KeyState.NA_ROUTE,
                note=f'exists but does not track on {route.value}'))
            continue
        if w in available:
            out.append(Resolution(w, KeyState.LIVE))
            continue
        alt = _fuzzy(w, available)
        if alt is not None:
            out.append(Resolution(w, KeyState.LIVE, resolved_to=alt))
            continue
        # AMBIGUOUS is reported as its own thing, not as 'not logged'. The
        # quantity IS there, under two or more namespaces that are DIFFERENT
        # QUANTITIES -- fwd and bwd log Z are not interchangeable. Picking one
        # would be a guess; calling it absent would send the reader looking for
        # a logging bug. Naming the candidates lets them ask for the one they
        # meant.
        cands = _namespace_candidates(w, available)
        if len(cands) > 1:
            out.append(Resolution(
                w, KeyState.ABSENT,
                note=f'ambiguous -- logged under {len(cands)} namespaces: '
                     f'{", ".join(cands)}. Name the one you mean.'))
        else:
            out.append(Resolution(w, KeyState.ABSENT, note='not logged by this run'))
    return out


def live_keys(resolutions: Iterable[Resolution]) -> list[str]:
    """The names safe to request from history."""
    return [r.key for r in resolutions if r.key]


def is_ema(key: str) -> bool:
    return key.startswith(EMA_PREFIXES)


# ---------------------------------------------------------------------------
# Config key literals
# ---------------------------------------------------------------------------
# Config names are literals for the same reason metric names are: a rename
# upstream must stay a one-file change. `%d` is a stage index, `%s` a mode or a
# metric tag.

CFG_STAGE = 'protocol_stages_%d_%s'          # stage-scoped key
CFG_STAGE_NAME = 'protocol_stages_%d_name'
CFG_STAGE_TRAIN_MODE = 'protocol_stages_%d_train_mode'
CFG_STAGE_EXIT_METRIC = 'protocol_stages_%d_exit_%d_metric'
CFG_STAGE_EXIT_ABOVE = 'protocol_stages_%d_exit_%d_above'
CFG_STAGE_EXIT_BELOW = 'protocol_stages_%d_exit_%d_below'
CFG_STAGE_BALANCE_METRIC = 'protocol_stages_%d_balance_metrics_%s'
CFG_STAGE_LR_SENSOR_METRIC = 'protocol_stages_%d_lr_sensor_metrics_%d'
CFG_STAGE_BUFFER_SERVO_NUM = 'protocol_stages_%d_buffer_servo_numerator'
CFG_STAGE_BUFFER_SERVO_DEN = 'protocol_stages_%d_buffer_servo_denominator'
# The servo's own bar and release, AS CONFIGURED BY THE RUN. `protocol.py`
# defaults the servo's pair to sensor A with bar/release 1.0/1.5, and most local
# runs declare no servo at all -- so the 0.368/0.60 pair below is the config
# generators' choice, not every run's, and calling it "the servo's hold band"
# without reading these is an assertion about a controller the run may not have.
CFG_STAGE_BUFFER_SERVO_BAR = 'protocol_stages_%d_buffer_servo_bar'
CFG_STAGE_BUFFER_SERVO_RELEASE = 'protocol_stages_%d_buffer_servo_release'
CFG_STAGE_DEACTIVATE = 'deactivate_threshold'          # stage-scoped tail
CFG_ANCHOR_GATE_CEILING_METRIC = 'buffers_anchor_buffer_health_gate_ceiling_metric'
CFG_ANCHOR_GATE_FLOOR_METRIC = 'buffers_anchor_buffer_health_gate_floor_metric'

# §4 confounds, all config-level.
CFG_EVAL_T = 'eval_T'
CFG_TRAIN_T = 'integrator_T'
CFG_CHECKPOINT_NAME = 'checkpoint_name'
CFG_CONTINUE_FROM_CHECKPOINT = 'continue_from_checkpoint'
CFG_PRIOR_PATH = 'prior_path'
CFG_ENERGY_FUNCTION = 'energy_function'
CFG_SEED = 'seed'
CFG_RUN_NAME = 'run_name'
CFG_TAG = 'tag'
CFG_EPOCHS = 'epochs'
CFG_WANDB_BLOB = '_wandb'                    # carries the git commit and argv

# Config keys that identify a run rather than configure it. A sweep table that
# lists these as "knobs that differ" is listing noise: every arm differs in its
# name, and comparing arms is the whole point.
CFG_IDENTITY = (CFG_RUN_NAME, CFG_TAG, CFG_WANDB_BLOB, CFG_EPOCHS,
                'Experiment', 'checkpoints_dir', 'device')

# The steps counter, and the run-position metric.
STEP_KEY = '_step'


def git_commit(config: dict, metadata: Optional[dict] = None) -> Optional[str]:
    """The commit the run actually executed, or None.

    §4's first confound is code version drift between arms, and the stamp is not
    where it looks like it should be: wandb buries it in the `_wandb` config
    blob under a per-machine hash key, and `wandb-metadata.json` carries a second
    copy. Both are read, blob first, because the blob survives the cloud API."""
    for src in (config, ):
        blob = _value(src, CFG_WANDB_BLOB)
        if isinstance(blob, dict):
            for entry in (blob.get('e') or {}).values():
                if isinstance(entry, dict):
                    commit = (entry.get('git') or {}).get('commit')
                    if commit:
                        return str(commit)
    if isinstance(metadata, dict):
        commit = (metadata.get('git') or {}).get('commit')
        if commit:
            return str(commit)
    return None


# ---------------------------------------------------------------------------
# Mechanism registry -- R2
# ---------------------------------------------------------------------------
# R2 is the highest-value check in the package: for every mechanism a config
# declares active, assert a nonzero activation trace. An inert mechanism does not
# give a null result -- it VOIDS the arm while looking like an answer, and it has
# repeatedly made whole batteries meaningless.
#
# EVERY ENTRY BELOW IS VERIFIED against the local run corpus (182 directories,
# 85 with a config and summary). An unverified declaration-to-trace pair is worse
# than a missing one: it manufactures findings on runs that are fine, and a
# check that cries wolf is switched off. Pairs that could not be verified are
# named in `docs/module_analysis.md` rather than guessed at here.
#
# Two traps this registry encodes, both of which produced a wrong reading during
# its construction:
#
#  * `protocol/bs_boost` is exp(log_boost), so it reads 1.0 -- NONZERO -- while
#    the servo is doing nothing. The actuator is `protocol/bs_log_boost`, which
#    is 0 when idle. A trace that is nonzero at rest is not a trace.
#  * `ray_calibration.enabled` and the stage's `lr_sensor.kind: ray` are
#    DIFFERENT declarations, and the trainer's own startup check warns when they
#    disagree. Both are registered; they answer different questions.


class Declare(str, Enum):
    """How a config key declares its mechanism active."""

    POSITIVE = 'positive'      # numeric and > 0
    TRUTHY = 'truthy'          # bool True, nonzero number, or non-empty string
    NOT_NULL = 'not_null'      # present and not None
    EQUALS = 'equals'          # string equality with `declares_value`


class Rule(str, Enum):
    """How a trace shows the mechanism fired."""

    NONZERO = 'nonzero'        # active on ticks where |v| exceeds the floor
    MOVES = 'moves'            # active on ticks where v differs from v[0]
    COUNTER = 'counter'        # monotone event counter; events = last - first


@dataclass(frozen=True)
class Mechanism:
    """One declared-active mechanism and the trace that proves it ran."""

    name: str
    scope: str                       # 'stage' or 'global'
    declared_by: str                 # config key ('stage' scope omits prefix)
    declares: Declare
    trace: tuple[str, ...]           # ANY of these firing counts as fired
    rule: Rule
    declares_value: str = ''         # for Declare.EQUALS
    threshold_key: str = ''          # config key supplying the activation floor
    note: str = ''


# 'stage'-scoped `declared_by` values are tails: the reader prefixes
# `protocol_stages_<current stage>_`.
MECHANISMS = (
    # --- allocation. R1 reads the allocation before the metric, and R2's
    # canonical case is "a frac below its deactivation threshold": the branch is
    # configured on, the controller has driven it under the floor, and it
    # contributes nothing while the config still claims it does.
    Mechanism('frac.fwd', 'stage', 'fracs_fwd', Declare.POSITIVE,
              ('Fwd Frac',), Rule.NONZERO, threshold_key=CFG_STAGE_DEACTIVATE,
              note='forward branch share; floor is the stage deactivate_threshold'),
    Mechanism('frac.bwd', 'stage', 'fracs_bwd', Declare.POSITIVE,
              ('Bwd Frac',), Rule.NONZERO, threshold_key=CFG_STAGE_DEACTIVATE),
    Mechanism('frac.replay', 'stage', 'fracs_replay', Declare.POSITIVE,
              ('Replay Frac',), Rule.NONZERO, threshold_key=CFG_STAGE_DEACTIVATE),

    # --- the balance controller. Which trace exists depends on the KIND, so the
    # kinds are separate mechanisms rather than one with a union of traces: a
    # union would report a ratio controller as fired because the proportional
    # keys are absent, which is the wrong answer twice over.
    Mechanism('balance.ratio', 'stage', 'balance_kind', Declare.EQUALS,
              ('protocol/rt_theta', 'protocol/rt_err'), Rule.MOVES,
              declares_value='ratio',
              note='rt_theta is the logit it actually steers'),
    Mechanism('balance.proportional', 'stage', 'balance_kind', Declare.EQUALS,
              ('protocol/prop_target', 'protocol/prop_drive_fwd',
               'protocol/prop_drive_bwd', 'protocol/prop_drive_replay'),
              Rule.MOVES, declares_value='proportional'),

    # --- the replay buffer residence servo.
    Mechanism('buffer_servo', 'stage', 'buffer_servo_gain', Declare.POSITIVE,
              ('protocol/bs_log_boost',), Rule.NONZERO,
              note='bs_log_boost is 0 at rest; bs_boost is exp() of it and '
                   'reads 1.0 while idle, so it cannot serve as the trace'),

    # --- LR sensors. Two independent declarations, deliberately both here.
    Mechanism('lr_sensor.ray', 'stage', 'lr_sensor_kind', Declare.EQUALS,
              ('lr_ctrl/calibrations',), Rule.COUNTER, declares_value='ray',
              note='per-stage opt-in, and as of project state 2 the ONLY switch: '
                   'the block-level `ray_calibration.enabled` flag is deleted and '
                   'the probe now arms iff some stage declares this kind'),
    Mechanism('lr_sensor.loss_slope', 'stage', 'lr_sensor_kind', Declare.EQUALS,
              ('lr_ctrl/slope_cuts',), Rule.COUNTER, declares_value='loss_slope'),
    # KEPT, though the key is retired in the trainer as of project state 2 --
    # same two-era reasoning as `adaptive_lr_enabled` below. `ray_calibration.
    # enabled` was a second declaration of what the stage already declares, and
    # deleting it from the config was right; deleting it from THIS registry would
    # be wrong, because every run recorded before state 2 carries it and reading
    # those runs is the entire job. A config key's retirement does not retire the
    # runs that used it.
    Mechanism('ray_calibration', 'global', 'ray_calibration_enabled',
              Declare.TRUTHY, ('lr_ctrl/calibrations',), Rule.COUNTER,
              note='PRE-STATE-2 declaring key, retired in the trainer: the stage '
                   'entry above is the switch now. On runs that carry it, enabled '
                   'with no stage opting in leaves the block inert (parameters '
                   'only) -- measured on 14 of 21 runs that declare it'),
    # TWO ENTRIES, ONE MECHANISM, because the declaring key changed and the two
    # eras are DISJOINT in the corpus: 53 runs carry `adaptive_lr_enabled`, 22
    # carry `adaptive_lr_seed_lr`, none carry both. Registering only the newer
    # key left R2 asserting nothing about the LR controller on the majority of
    # runs -- and saying so in language ('the config asserts nothing') that reads
    # as 'not configured'. An absent key makes no claim, so the era a run is not
    # from simply reports OFF.
    Mechanism('adaptive_lr', 'global', 'adaptive_lr_seed_lr', Declare.POSITIVE,
              ('lr_ctrl/scale',), Rule.MOVES),
    Mechanism('adaptive_lr.enabled', 'global', 'adaptive_lr_enabled',
              Declare.TRUTHY, ('lr_ctrl/scale',), Rule.MOVES,
              note='the earlier declaring key for the same controller'),
    # THE THIRD ERA, and its trace had to change WITH the mechanism rather than
    # inherit the two above. `lr_ctrl/scale` under the bracket is piecewise
    # constant BY DESIGN -- burn-in scale, then the promoted rung, and nothing in
    # between -- so `Rule.MOVES` on it would report a correctly-working bracket
    # as inert on any run whose selection happened to land on the burn-in scale,
    # and would report EVERY fixed-mode run as inert. `lr_bracket/phase` moves in
    # both modes (burn_in -> cruise) and moves only when the machine advances,
    # which is the thing being asserted.
    Mechanism('lr_control', 'global', 'lr_control_seed_lr', Declare.POSITIVE,
              ('lr_bracket/phase',), Rule.MOVES,
              note='the brute-force bracket (project state 10). A flat scale is '
                   'NOT evidence of inertness here -- that is what the mechanism '
                   'does between selections'),

    # --- Z calibration. Measured inert on 11 of 69 runs that enable it.
    Mechanism('z_calibration', 'global', 'z_calibration_enabled', Declare.TRUTHY,
              ('z_cal/steps', 'z_cal/p'), Rule.NONZERO),

    # --- the MLE slope gate. `gates/mle_flat` is published to the protocol but
    # never logged as a metric, so the streak counter is its ONLY trace.
    Mechanism('mle_gate', 'stage', 'flags_mle_gate', Declare.TRUTHY,
              ('protocol/exit_streak_gates_mle_flat',), Rule.NONZERO,
              note='measured inert on 10 of 14 runs that enable it'),

    # --- prioritised replay draw.
    Mechanism('replay_prioritise', 'global',
              'buffers_replay_buffer_prioritise_enabled', Declare.TRUTHY,
              ('replay/is_elig_frac',), Rule.NONZERO),

    # --- the two-point LR probe. REGISTERED PRECISELY BECAUSE IT IS RETIRED:
    # `step_probe.py` is gone from the tree, while the config generators still
    # emit the block, so any run launched now declares a sensor that cannot
    # log. That is R2's "a knob retired upstream" as a live instance rather
    # than a hypothetical, and this entry is what turns it into a finding
    # instead of a silence. The pair is verified on runs from when the module
    # existed: 33 of 36 declaring runs carry the trace, 3 declare and log
    # nothing.
    Mechanism('step_probe', 'global', 'step_probe_enabled', Declare.TRUTHY,
              ('lrprobe/alpha_n', 'lrprobe/alpha_star'), Rule.NONZERO,
              note='step_probe.py is deleted from the tree; a run declaring '
                   'this today cannot produce the trace'),
)

# Generated mechanisms. These are families, not fixed entries: the set depends on
# what the config declares, so they are templates the check expands.
#
# LOSS_COEFF_TRACE is the strongest liveness evidence in the run, because it is
# the coefficient the trainer is ACTUALLY holding, after the stage's overrides.
# It is a CHANGE-ONLY channel -- emitted at eval time and only when a value
# moved -- so it must be read from the SUMMARY (which holds the last value) and
# not from history, where it is a 1-2 point series that the local parser drops.
LOSS_COEFF_TRACE = 'loss_coeffs/%s_/%s'
LOSS_COEFF_IS_SUMMARY_ONLY = True

# One per stage exit condition. `tag` is the metric name with '/' -> '_'.
EXIT_STREAK_TRACE = 'protocol/exit_streak_%s'
# The live annealed bar for the same condition, in the metric's own units. R13:
# never ratchet a threshold below a floor you have not measured.
EXIT_THRESHOLD_TRACE = 'protocol/thr_%s'


def metric_tag(metric: str) -> str:
    """`bwd/tbc` -> `bwd_tbc`, the form the protocol uses in its own key names."""
    return metric.replace('/', '_')


# ---------------------------------------------------------------------------
# R11 -- replay overfitting
# ---------------------------------------------------------------------------
# TWO SENSORS, AND ONLY ONE OF THEM HAS A BAR WORTH FLAGGING ON.
#
# SENSOR A -- `replay/scatter_err / fwd/scatter_err`. The original, and what the
# build spec named. Replay draws are a |resid|-prioritised resample of stored
# forward rollouts, so a replay batch is by construction the hard tail of the
# forward distribution and its spread should exceed fresh forward's.
#
# It is REPORTED AND NOT FLAGGED, for a reason `module_metrics.md` states
# outright: a ratio below 1 is equally the signature of memorisation and of a
# coverage gap, "the statistic does not distinguish them, and reading it as
# either one alone is unwarranted". `module_modulators.md` adds that its
# "stated justification is stale and its thresholds are uncalibrated".
#
# Measured over the local corpus, that is not a theoretical worry: 45 of 60
# TB-route runs sit below 1, and 58 of 60 below the '~2 healthy' figure. A
# finding state that fires on three quarters of a corpus is not a finding, and a
# check that cries wolf gets switched off -- so the number is surfaced with its
# ambiguity named, and the reader decides.
R11_NUMERATOR = 'replay/scatter_err'
R11_DENOMINATOR = 'fwd/scatter_err'
# Kept as a stated REFERENCE printed beside the number, never as a bar.
R11_SCATTER_REFERENCE = 2.0
R11_SCATTER_BELOW = 1.0

# SENSOR B -- `replay/ema_loss_mean / replay/birth_loss_mean`, each resident
# row's current residual against the one it was admitted with. The preferred
# memorisation sensor, and the one this check FLAGS on, because its bar is
# DERIVED rather than calibrated: 0.368 is lambda*tau = 1, i.e. rows are being
# corrected exactly as fast as they are replaced. Below it, the buffer is being
# fitted faster than it turns over, which is memorisation and not a coverage gap.
#
# It behaves like a bar should: 0 of 44 runs below the derived bar, 4 below the
# release threshold. That is the difference a derived bar makes.
R11_MEMORISATION_NUMERATOR = 'replay/ema_loss_mean'
R11_MEMORISATION_DENOMINATOR = 'replay/birth_loss_mean'
R11_MEMORISATION_BAR = 0.368        # lambda*tau = 1, derived
R11_MEMORISATION_RELEASE = 0.60     # lambda*tau ~ 0.5
# TB ROUTE ONLY, per the spec. Everywhere else the check reports NA_ROUTE and
# stops. `MLE_PRIOR` was in this tuple and contradicted both the spec line and
# the comment above it -- the prior route trains no replay TB branch, so a ratio
# computed there is a ratio between a quantity the optimiser is minimising and
# one nothing is touching.
R11_ROUTES = (Route.TB_UNCONDITIONAL,)


# ---------------------------------------------------------------------------
# R14 -- censoring bounds
# ---------------------------------------------------------------------------
# A censored estimator reported AT its censoring bound is not a reading. These
# are the bounds this codebase imposes, with the series they clamp.
CENSORED = {
    # ray_calibration clamped its per-alpha t-statistics to +/-99 before logging.
    # That grid stopped being logged on 2026-08-23, so this entry is HISTORICAL:
    # it still reads runs recorded before the change and matches nothing in a
    # current one. Remove it once those runs stop mattering.
    'raycal/t_': 99.0,
}
# Config keys that name a clip a series can pin against.
CFG_CLIP_KEYS = ('gradient_norm_clip', 'model_gfn_clip')
