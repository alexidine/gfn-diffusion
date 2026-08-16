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
