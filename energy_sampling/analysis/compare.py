"""
Multi-arm comparison: the sweep table, and the aligned feature table.

TWO DELIVERABLES, IN THIS ORDER OF IMPORTANCE.

  1. `compare(runs)` -> a `Comparison` of plain dataclasses. This is the primary
     product: an agent or a script consumes it and synthesises. Nothing here
     returns text that a caller has to parse back.
  2. `format_comparison(...)` -> the same thing rendered for a terminal.

COMPARABILITY IS NOT A CAVEAT, IT IS THE GATE. A comparison across arms that are
not comparable is not a weaker result, it is not a result. So §4 is answered
FIRST and the answer travels WITH the data: `FeatureTable.blockers` is a field of
the table itself, not a separate section a reader can skip, and
`format_feature_table` emits the banner before a single number. §4 is NOT
reimplemented here -- `checks.check_confounds` owns it, and its shared helpers
(`_conf_get`, `_conf_show`, `_conf_equal`) are reused so this module and the
check cannot disagree about whether two config values are the same.

THREE STATES, PER CELL. A blank cell is the failure this package exists to
prevent, so there are none: every cell renders a word. ABSENT (this arm does not
log it), NA_ROUTE (it does log it, populated, and the numbers do not mean here
what they mean on a TB run), THIN (logged, too few ticks in the window for a
feature). None of them is a zero and none of them is a blank.

ARMS ON DIFFERENT ROUTES DO NOT SHARE A TOPLINE. With no explicit metric list the
table SPLITS into one block per route, each with its own arms and its own
`K.TOPLINE[route]`; there is no union row, because a union row asserts the two
columns are commensurable. Passing `metrics=` is the honest way to read one
series across a route boundary: every arm is still resolved against ITS OWN
route, so the cells that do not mean the same thing say NA_ROUTE rather than
lining up beside the ones that do.

NO METRIC-NAME OR CONFIG-KEY LITERALS. Metric sets come from `K.TOPLINE`, key
resolution from `K.resolve`, identity keys from `K.CFG_IDENTITY`. If you find
yourself typing `'fwd/'` here, stop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import numpy as np

from . import checks as C
from . import features as F
from . import keys as K


# ---------------------------------------------------------------------------
# Cell and knob states
# ---------------------------------------------------------------------------

class CellState(str, Enum):
    """What one arm can say about one metric.

    LIVE / ABSENT / NA_ROUTE are `K.KeyState`'s three, carried through unchanged
    -- this module never invents a fourth reading of a key. THIN and NO_SERIES
    are downstream of resolution: the key resolved LIVE and the window still
    could not produce a feature, which is a different fact from the key not
    being there and must not render as the same thing.
    """

    LIVE = 'live'
    ABSENT = 'absent'            # not logged by this arm (note carries why)
    NA_ROUTE = 'na_route'        # logged, populated, not meaningful on its route
    THIN = 'thin'                # logged; too few ticks in the window
    NO_SERIES = 'no_series'      # resolved LIVE, no numeric series or scalar


class KnobKind(str, Enum):
    """How a config knob differs across the arms."""

    VALUE = 'value'          # present everywhere, values disagree
    PRESENCE = 'presence'    # in some configs and not others (absent = default)
    BLOB_ONLY = 'blob_only'  # differs INSIDE a repr string with no flat child


# Rendered in place of a number. Never blank, and never zero.
_CELL_TOKEN = {
    CellState.ABSENT: 'ABSENT',
    CellState.NA_ROUTE: 'NA_ROUTE',
    CellState.THIN: 'THIN',
    CellState.NO_SERIES: 'NO_SERIES',
}

# The block label when the caller named the metrics instead of taking the
# per-route toplines. Not a route, deliberately: the arms in it can be on
# several, and each cell is still resolved against its own.
REQUESTED = 'requested'

# What a `Feature` can be read out as. The renderer and the API share this map so
# a column heading and a programmatic read cannot name different quantities.
STATS = {
    'last': lambda f: f.last,
    'delta': lambda f: f.delta,
    'slope/1k': lambda f: f.slope_per_1k,
    'sigma': lambda f: f.sigma,
    'n': lambda f: float(f.n),
}
DEFAULT_STATS = ('last', 'slope/1k')


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    """One run's identity and reading context, without the run attached.

    `label` is `checks.run_label` -- `name#run_id` -- because a DISPLAY NAME IS
    NOT UNIQUE here (nine names are shared by two or more runs in the local
    corpus). `short` is the column heading and collapses to the bare name only
    when it is unambiguous WITHIN THIS BATTERY, which is the only scope in which
    ambiguity can mislead a reader of this table.
    """

    label: str
    short: str
    run_id: str
    name: str
    route: K.Route
    stage_name: Optional[str]
    stage_index: Optional[int]
    last_step: float
    n_series: int


@dataclass(frozen=True)
class KnobRow:
    """One config knob that differs, decoded to arm names.

    `values` is arm label -> the RENDERED value, using `checks._conf_show`, so
    `<missing>` (the key is not in this config; the knob takes its default) and
    `<null>` (the key is there holding nothing) stay distinguishable. `raw` keeps
    the objects for a caller that wants to compute on them.
    """

    key: str
    kind: KnobKind
    values: dict            # arm label -> rendered str
    raw: dict               # arm label -> the object, or None when absent
    present: dict           # arm label -> bool
    n_distinct: int
    note: str = ''


@dataclass(frozen=True)
class SweepTable:
    """Which knobs actually differ across the arms.

    `K.CFG_IDENTITY` is excluded: every arm differs in its name, and listing that
    is listing the index the sweep is keyed by.

    `blobs` records the wandb config's OTHER copy of itself. Alongside the
    flattened scalars, wandb stores each config section as a top-level repr
    STRING (`Namespace(...)`, thousands of characters). Those strings differ
    whenever any of their children do, so left in, the batt0807 sweep reported
    eleven differing knobs of which four were unreadable duplicates of the other
    seven. A blob is detected STRUCTURALLY -- a string value with at least one
    `<key>_*` sibling -- so no config-key literal is needed and a renamed section
    keeps working.
    """

    arms: tuple
    rows: tuple
    n_knobs_compared: int
    blobs: tuple            # (blob key, differing flattened children) pairs

    @property
    def differs(self) -> bool:
        return bool(self.rows)


@dataclass(frozen=True)
class Cell:
    """One arm's answer for one metric."""

    arm: str
    metric: str
    state: CellState
    feature: Optional[F.Feature] = None
    resolved_to: Optional[str] = None   # set when the key was renamed
    n_ticks: int = 0
    note: str = ''

    def value(self, stat: str) -> Optional[float]:
        """The statistic, or None when this cell has no feature. None and 0.0
        are different answers and this never returns the second for the first."""
        if self.feature is None:
            return None
        return float(STATS[stat](self.feature))


@dataclass(frozen=True)
class MetricRow:
    metric: str
    cells: dict             # arm label -> Cell

    def state_counts(self) -> dict:
        out: dict = {}
        for c in self.cells.values():
            out[c.state] = out.get(c.state, 0) + 1
        return out

    @property
    def n_live(self) -> int:
        return sum(1 for c in self.cells.values() if c.state is CellState.LIVE)


@dataclass(frozen=True)
class MetricBlock:
    """One route's arms against one route's topline, or the caller's list.

    A block is the unit that is internally comparable. Two blocks are two
    different questions and the report never puts a number from one beside a
    number from the other.
    """

    label: str
    route: Optional[K.Route]
    arm_labels: tuple
    metrics: tuple
    rows: tuple
    source: str             # where the metric list came from


@dataclass(frozen=True)
class FeatureTable:
    """The aligned table: one metric per row, one arm per column.

    `blockers` is a FIELD OF THE TABLE and not a sibling section. That is the
    enforcement: a caller cannot hold the numbers without holding the §4
    findings that say whether the columns may be compared, and the renderer
    emits them before the first cell.
    """

    arms: tuple
    blocks: tuple
    blockers: tuple
    window: Optional[float]
    route_groups: tuple     # (Route, (arm label, ...)) pairs, in arm order

    @property
    def split_by_route(self) -> bool:
        return len(self.route_groups) > 1

    @property
    def comparable(self) -> bool:
        return not self.blockers


@dataclass(frozen=True)
class Comparison:
    """Everything `compare` produces. Confounds first, deliberately."""

    arms: tuple
    confounds: C.CheckResult
    sweep: SweepTable
    features: FeatureTable
    window: Optional[float]

    @property
    def blockers(self) -> tuple:
        """The CROSS-ARM §4 findings -- the ones that say these arms may not be
        read against each other. Per-run findings are real and are in
        `confounds.findings`; they qualify an arm, they do not void the
        comparison, and merging the two would make the gate fire on facts about
        a single run."""
        return self.features.blockers

    @property
    def comparable(self) -> bool:
        return not self.blockers

    def records(self, stats: Iterable[str] = tuple(STATS)) -> list:
        """One flat dict per (block, metric, arm) -- the shape a script or an
        agent tabulates.

        EVERY RECORD CARRIES `state`, `comparable` AND `n_blockers`. A flattened
        row is the form most likely to be pulled out of context and averaged, so
        the gate and the three states travel inside the row rather than in a
        header the consumer can drop. `state != 'live'` means the numeric fields
        are None -- never 0.0.
        """
        out = []
        n_block = len(self.blockers)
        by_label = {a.label: a for a in self.arms}
        for block in self.features.blocks:
            for row in block.rows:
                for label in block.arm_labels:
                    cell = row.cells[label]
                    arm = by_label[label]
                    rec = {
                        'block': block.label,
                        'block_route': block.route.value if block.route else '',
                        'metric': row.metric,
                        'arm': label,
                        'arm_short': arm.short,
                        'arm_route': arm.route.value,
                        'arm_stage': arm.stage_name,
                        'state': cell.state.value,
                        'resolved_to': cell.resolved_to,
                        'n_ticks': cell.n_ticks,
                        'note': cell.note,
                        'comparable': not n_block,
                        'n_blockers': n_block,
                    }
                    for stat in stats:
                        rec[stat] = cell.value(stat)
                    out.append(rec)
        return out

    def sweep_records(self) -> list:
        """One flat dict per (knob, arm). Same shape rule as `records`."""
        return [{'knob': row.key, 'how': row.kind.value, 'arm': arm.label,
                 'arm_short': arm.short, 'value': row.values[arm.label],
                 'present': row.present[arm.label], 'raw': row.raw[arm.label],
                 'note': row.note}
                for row in self.sweep.rows for arm in self.sweep.arms]


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

# Subject prefix `checks` uses for its cross-arm rows. Not a metric name and not
# a config key, so it does not belong in keys.py -- but it is spelled once, here,
# so a change to the check's subject naming fails at one site instead of
# silently emptying the blocker list.
_BATTERY_PREFIX = 'battery/'


def _labels(runs) -> tuple:
    """`(label, short)` per run, positionally.

    EVERY TABLE IS KEYED BY `label`, so it has to be unique or two arms share a
    column. `checks.run_label` gives `name#run_id` and that is unique for real
    runs, but the same run object passed twice is not, and a silent collision
    loses a whole arm -- so a repeat gets a visible `~N` rather than
    disappearing.

    `short` is the column heading: the bare display name, and only when it is
    unique in THIS battery. Nine names in the local corpus are shared by two or
    more runs and `mk_dev` alone by eleven.
    """
    labels, seen = [], set()
    for run in runs:
        lab = C.run_label(run)
        base, i = lab, 1
        while lab in seen:
            i += 1
            lab = f'{base}~{i}'
        seen.add(lab)
        labels.append(lab)

    names = [str(getattr(r, 'name', '') or '') for r in runs]
    counts: dict = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return tuple((lab, n if (n and counts[n] == 1) else lab)
                 for lab, n in zip(labels, names))


def arms(runs) -> tuple:
    """`Arm` per run, in the order given.

    The stage and route come from `checks.context`, which is the package's
    single resolution point: it refuses to infer a route from a stage the run
    never reached, and NA_ROUTE marking is driven entirely by that route."""
    runs = list(runs)
    out = []
    for run, (label, short) in zip(runs, _labels(runs)):
        ctx = C.context(run)
        out.append(Arm(
            label=label,
            short=short,
            run_id=str(getattr(run, 'run_id', '') or ''),
            name=str(getattr(run, 'name', '') or ''),
            route=ctx.route,
            stage_name=ctx.stage_name,
            stage_index=ctx.stage_index,
            last_step=C.as_float(getattr(run, 'last_step', 0.0), 0.0),
            n_series=len(getattr(run, 'history', {}) or {}),
        ))
    return tuple(out)


# ---------------------------------------------------------------------------
# The sweep table
# ---------------------------------------------------------------------------

# A python `repr()` of a structured object: a constructor name and parentheses.
# This is a test on the SHAPE OF A VALUE, not a config-key name, so it stays out
# of keys.py and survives any rename of the sections themselves.
_REPR_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_.]*\(.*\)$', re.S)


def _blob_keys(config: dict) -> set:
    """Config keys holding wandb's repr-string copy of a config SECTION.

    wandb writes every section twice -- once as flattened scalars and once as
    the section's `repr()` under the bare name -- and the repr differs whenever
    ANY of its children do. Left in, the batt0807 sweep reported eleven knobs of
    which four were the same seven differences again, each a few thousand
    unreadable characters wide.

    TWO CONDITIONS, and the second one is not optional. The key must have at
    least one flattened `<key>_*` sibling (so its content is recoverable) AND
    its value must look like a repr. On the sibling test alone,
    `z_calibration_sensor` -- a genuine string knob holding `pooled` / `rms`,
    with an unrelated `z_calibration_sensor_quantile` beside it -- was
    classified as a blob and reported with a note saying its difference was
    "inside the repr string", which was simply false. Measured across the local
    corpus that was the only false positive, and one is enough: a sweep table
    that misdescribes a swept knob is worse than one that omits it.
    """
    keys = set(config)
    out = set()
    for k in keys:
        v = K._value(config, k)
        if not isinstance(v, str) or not _REPR_RE.match(v):
            continue
        pre = k + '_'
        if any(o != k and o.startswith(pre) for o in keys):
            out.add(k)
    return out


def _knob_keys(config: dict) -> set:
    return {k for k in config if k not in K.CFG_IDENTITY}


def _differs(runs, key: str):
    """`(kind, seen)` for one key, or None when it does not differ.

    `seen` is POSITIONAL -- `[(present, value), ...]` in arm order -- and not a
    dict keyed by the run. The same `Run` object can legitimately appear twice in
    a battery (a self-comparison, a smoke test), and keying by identity silently
    merges those two columns into one.

    Uses `checks._conf_get` / `_conf_equal` so this table and the §4 check cannot
    disagree about what "the same value" means -- `_conf_equal` treats NaN as
    equal to NaN, without which a yaml `.nan` reads as a sweep dimension nobody
    swept."""
    seen = [C._conf_get(run.config or {}, key) for run in runs]
    p0, v0 = seen[0]
    if any(p != p0 for p, _ in seen[1:]):
        return KnobKind.PRESENCE, seen
    if any(not C._conf_equal(v, v0) for _, v in seen[1:]):
        return KnobKind.VALUE, seen
    return None


def _knob_row(key: str, kind: KnobKind, seen, arm_list, note: str = '') -> KnobRow:
    vals, rawv, pres = {}, {}, {}
    for arm, (present, value) in zip(arm_list, seen):
        vals[arm.label] = C._conf_show(present, value)
        rawv[arm.label] = value if present else None
        pres[arm.label] = present
    return KnobRow(key=key, kind=kind, values=vals, raw=rawv, present=pres,
                   n_distinct=len(set(vals.values())), note=note)


def sweep_table(runs, arm_list=None) -> SweepTable:
    """Which config knobs differ across the arms, decoded to arm names.

    A knob differing by PRESENCE counts. An absent key takes its default, and a
    default that differs from a sibling's explicit value is a swept dimension
    whether or not anyone meant to sweep it -- §4's "arms that differ by
    omission" is exactly this shape read from the other side.
    """
    runs = list(runs)
    arm_list = arm_list if arm_list is not None else arms(runs)
    keys: set = set()
    blobs: set = set()
    for run in runs:
        cfg = run.config or {}
        keys |= _knob_keys(cfg)
        blobs |= _blob_keys(cfg)

    rows, blob_rows = [], []
    flat_differ = set()
    for key in sorted(keys - blobs):
        got = _differs(runs, key)
        if got is None:
            continue
        flat_differ.add(key)
        kind, seen = got
        rows.append(_knob_row(key, kind, seen, arm_list))

    # The blobs, second, and only to say what the flattened keys already showed
    # -- OR, loudly, that they did not.
    for key in sorted(keys & blobs):
        got = _differs(runs, key)
        if got is None:
            continue
        pre = key + '_'
        children = sorted(k for k in flat_differ if k.startswith(pre))
        if children:
            blob_rows.append((key, tuple(children)))
            continue
        # A section whose repr differs while every flattened child agrees. The
        # difference is real and is not visible in any scalar key -- a nested
        # list, a None, an object wandb did not flatten. Silence here would hide
        # a swept dimension, so it becomes a row of its own.
        rows.append(_knob_row(
            key, KnobKind.BLOB_ONLY, got[1], arm_list,
            note='differs only inside the repr string -- no flattened child of '
                 'this section differs, so the swept value is not readable from '
                 'any scalar config key'))

    return SweepTable(arms=tuple(arm_list), rows=tuple(rows),
                      n_knobs_compared=len(keys - blobs),
                      blobs=tuple(blob_rows))


# ---------------------------------------------------------------------------
# The aligned feature table
# ---------------------------------------------------------------------------

def _window_or_all(window: Optional[float]) -> float:
    """`features.extract` needs a number. None means the whole history."""
    return float('inf') if window is None else float(window)


def _cell(run, arm: Arm, metric: str, window: Optional[float]) -> Cell:
    """One arm's answer for one metric, in three states plus two refusals.

    Resolution is per ARM, against that arm's own route. That is what lets a
    requested metric list span a route boundary honestly: the same row holds a
    number on the arm where it means something and NA_ROUTE on the arm where it
    does not, instead of two numbers that invite being subtracted.
    """
    res, = K.resolve(run.available_keys(), [metric], arm.route)
    if res.state is K.KeyState.NA_ROUTE:
        return Cell(arm.label, metric, CellState.NA_ROUTE, note=res.note)
    if res.state is K.KeyState.ABSENT:
        return Cell(arm.label, metric, CellState.ABSENT, note=res.note)

    key = res.key
    got = C.series(run, key)
    if got is None:
        return Cell(arm.label, metric, CellState.NO_SERIES,
                    resolved_to=res.resolved_to,
                    note=f'{key} resolved LIVE but holds no numeric series and '
                         f'no scalar summary value')
    s, v = np.asarray(got[0], float), np.asarray(got[1], float)
    w = _window_or_all(window)
    n_in = int(np.count_nonzero(s >= max(s[-1] - w, s[0]))) if len(s) else 0
    feat = F.extract(key, s, v, w,
                     is_ema=K.is_ema(key),
                     low_trust=key in K.LOW_TRUST,
                     watch_escape=key in K.ESCAPE_KEYS)
    if feat is None:
        # NOT absent. The series is there; the window does not hold enough of it
        # for a trend, and reporting that as a hole would send the reader after
        # a logging bug that is not there.
        return Cell(arm.label, metric, CellState.THIN,
                    resolved_to=res.resolved_to, n_ticks=n_in,
                    note=f'{n_in} tick(s) in the window; a feature needs 3')
    return Cell(arm.label, metric, CellState.LIVE, feature=feat,
                resolved_to=res.resolved_to, n_ticks=feat.n)


def _block(runs, arm_list, metrics, label, route, source,
           window) -> MetricBlock:
    rows = []
    for metric in metrics:
        cells = {a.label: _cell(r, a, metric, window)
                 for a, r in zip(arm_list, runs)}
        rows.append(MetricRow(metric=metric, cells=cells))
    return MetricBlock(label=label, route=route,
                       arm_labels=tuple(a.label for a in arm_list),
                       metrics=tuple(metrics), rows=tuple(rows), source=source)


def _route_groups(arm_list) -> tuple:
    groups: dict = {}
    for a in arm_list:
        groups.setdefault(a.route, []).append(a.label)
    return tuple((r, tuple(v)) for r, v in groups.items())


def feature_table(runs, *, metrics: Optional[Iterable[str]] = None,
                  window: Optional[float] = None,
                  blockers: Iterable = ()) -> FeatureTable:
    """The aligned table.

    `metrics=None` splits by ROUTE: one block per route, each holding only its
    own arms and its own `K.TOPLINE[route]`. Two arms on different routes do not
    have a comparable topline, and a union row would assert that they do.

    `metrics=[...]` is the deliberate cross-route read: ONE block, every arm a
    column, and each cell still resolved against its own arm's route -- so the
    cells that do not mean the same thing say NA_ROUTE.
    """
    runs = list(runs)
    arm_list = arms(runs)
    blockers = tuple(blockers)

    if metrics is not None:
        metrics = tuple(metrics)
        if not metrics:
            # Same rule as the empty battery: a table with no rows renders
            # identically to a table whose every row came back clean.
            raise ValueError(
                'metrics=[] asks for a table with no rows, which renders '
                'identically to a comparison in which nothing differed. Pass '
                'metrics=None for each arm\'s own topline.')
        blocks = (_block(runs, arm_list, metrics, REQUESTED, None,
                         'caller', window),)
    else:
        blocks = []
        for route, labels in _route_groups(arm_list):
            pairs = [(a, r) for a, r in zip(arm_list, runs) if a.label in labels]
            blocks.append(_block(
                [r for _, r in pairs], [a for a, _ in pairs],
                K.TOPLINE[route], route.value, route,
                f'K.TOPLINE[{route.value}]', window))
        blocks = tuple(blocks)

    return FeatureTable(arms=arm_list, blocks=blocks, blockers=blockers,
                        window=window, route_groups=_route_groups(arm_list))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# A per-run subject that nonetheless voids the comparison. See `_blockers`.
_CONFIG_SUBJECT_SUFFIX = '/config'


def _blockers(res: C.CheckResult) -> tuple:
    """The §4 findings that say these arms may not be read against each other.

    Cross-arm findings -- a `battery/` subject in one of `checks.FINDING_STATES`
    -- defined by the check and not by this module: picking a subset by hand
    would be this module deciding which of §4's confounds count, which is
    exactly the judgment the package does not emit.

    PLUS ONE PER-RUN SUBJECT: an arm whose CONFIG DID NOT LOAD. That is a fact
    about one run and it still voids the comparison, because every cross-arm
    subject reads it as `<missing>` and therefore agrees with itself. Two arms
    with unparseable configs produced a clean comparability bill, a "no knob
    differs" sweep, and a full aligned metric table putting their numbers in the
    same rows -- a confident comparison of two things nothing is known about.
    Nothing else per-run belongs here: an ordinary per-run finding qualifies an
    arm, it does not make the arms incommensurable.
    """
    return tuple(f for f in res.rows
                 if f.is_finding
                 and (f.subject.startswith(_BATTERY_PREFIX)
                      or f.subject.endswith(_CONFIG_SUBJECT_SUFFIX)))


def compare(runs, *, metrics: Optional[Iterable[str]] = None,
            window: Optional[float] = None) -> Comparison:
    """Compare arms: §4 first, then the sweep, then the aligned features.

    `runs` is an iterable of `pull.Run`. One run is allowed and answers
    honestly -- the §4 check says its cross-arm subjects were skipped, the sweep
    table has nothing to compare, and the feature table is one column.

    Raises ValueError on an empty battery. There is no comparison of nothing,
    and returning an empty `Comparison` would render as "compared, found no
    differences".
    """
    runs = list(runs)
    if not runs:
        raise ValueError(
            'no runs to compare. An empty comparison renders identically to '
            '"compared them, nothing differs", which is the opposite of what '
            'is true.')

    confounds = C.check_confounds(runs, window=window)
    confounds.run = ', '.join(C.run_label(r) for r in runs)
    confounds.header = f'{len(runs)} arm(s)'

    arm_list = arms(runs)
    sweep = sweep_table(runs, arm_list)
    feats = feature_table(runs, metrics=metrics, window=window,
                          blockers=_blockers(confounds))
    return Comparison(arms=arm_list, confounds=confounds, sweep=sweep,
                      features=feats, window=window)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_COL = 15
_NAME_COL = 34
_NAME_COL_MAX = 64
_VALUE_COL = 26


def _name_col(names: Iterable[str]) -> int:
    """Width for the row-label column, sized to the longest name.

    THE ROW LABEL IS THE ROW'S IDENTITY and a fixed width elides it. Measured on
    a real four-arm battery, a 34-column cap rendered
    `buffers_replay_buffer_prioritise_enabled` and
    `buffers_replay_buffer_prioritise_kappa` as the SAME truncated string, so the
    sweep table showed two rows that could not be told apart -- the failure the
    table exists to prevent, in the renderer."""
    longest = max((len(str(n)) for n in names), default=0)
    return max(_NAME_COL, min(longest + 2, _NAME_COL_MAX))


def _trunc(s: str, n: int) -> str:
    """Elide the MIDDLE, never the tail. These names are prefix-heavy
    (`protocol_stages_1_lr_sensor_*`, `buffers_replay_buffer_*`); cutting the
    tail deletes the only distinguishing part."""
    s = str(s)
    if len(s) <= n:
        return s
    if n < 8:
        return s[:n]
    head = (n - 1) // 2
    return s[:head] + '~' + s[len(s) - (n - head - 1):]


def _num(v: float) -> str:
    if v is None:
        return ''
    if not np.isfinite(v):
        return 'nan' if np.isnan(v) else ('inf' if v > 0 else '-inf')
    return f'{v:.4g}'


# Terminal caps. The structured `SweepTable` is always complete; only the render
# is trimmed, and it says how many it trimmed -- a truncation that does not state
# its own size is the report deciding what the reader may see.
_SWEEP_ROWS_SHOWN = 40
_BLOB_CHILDREN_SHOWN = 8


def format_sweep(sweep: SweepTable, *, width: int = _VALUE_COL,
                 limit: int = _SWEEP_ROWS_SHOWN) -> str:
    """The sweep table. Says "no knob differs" out loud rather than printing an
    empty table, because an empty table and a battery of identical arms render
    the same and mean different things."""
    head = (f'SWEEP  {len(sweep.rows)} knob(s) differ of '
            f'{sweep.n_knobs_compared} compared, across {len(sweep.arms)} arm(s)')
    if not sweep.rows:
        extra = ''
        if len(sweep.arms) < 2:
            extra = ' (one arm -- there is nothing to differ from)'
        return (head + '\n  NO KNOB DIFFERS outside the identity keys'
                + extra + _format_blobs(sweep))

    # BLOB_ONLY rows first. They are the ones whose swept value is not readable
    # from any scalar key, so they are the rows a cap must never eat.
    ordered = ([r for r in sweep.rows if r.kind is KnobKind.BLOB_ONLY]
               + [r for r in sweep.rows if r.kind is not KnobKind.BLOB_ONLY])
    shown = ordered[:limit] if limit else ordered

    cols = [a.short for a in sweep.arms]
    nw = _name_col(r.key for r in shown)
    lines = [head, '  ' + 'KNOB'.ljust(nw) + 'HOW'.ljust(10)
             + ''.join(_trunc(c, width - 1).ljust(width) for c in cols)]
    for row in shown:
        raw = [str(row.values[a.label]) for a in sweep.arms]
        cut = [_trunc(v, width - 1) for v in raw]
        # A SWEEP ROW EXISTS BECAUSE THESE VALUES DIFFER. If the cap renders two
        # differing values as the same string, the row now asserts the opposite
        # of the fact that put it in the table -- so the cap loses, and the row
        # prints one value per line at full width instead.
        collided = len(set(cut)) < len(set(raw))
        if not collided:
            lines.append('  ' + _trunc(row.key, nw - 1).ljust(nw)
                         + row.kind.value.ljust(10)
                         + ''.join(c.ljust(width) for c in cut))
        else:
            lines.append('  ' + _trunc(row.key, nw - 1).ljust(nw)
                         + row.kind.value.ljust(10)
                         + '(values too long to tabulate -- listed below)')
            for a, v in zip(sweep.arms, raw):
                lines.append(f'      {a.short}: {v}')
        if row.note:
            lines.append(f'      ! {row.note}')
    if len(shown) < len(sweep.rows):
        lines.append(f'  ... {len(sweep.rows) - len(shown)} further differing '
                     f'knob(s) not printed; all of them are in `sweep.rows`')
    return '\n'.join(lines) + _format_blobs(sweep)


def _format_blobs(sweep: SweepTable) -> str:
    if not sweep.blobs:
        return ''
    parts = ['\n  wandb also stores each config section as a repr STRING; these '
             'differ only because a flattened knob above does:']
    for key, children in sweep.blobs:
        head = ', '.join(children[:_BLOB_CHILDREN_SHOWN])
        more = len(children) - _BLOB_CHILDREN_SHOWN
        parts.append(f'    {key} -> {head}'
                     + (f'  (+{more} more)' if more > 0 else ''))
    return '\n'.join(parts)


def _cell_text(cell: Cell, stat: str) -> str:
    if cell.state is not CellState.LIVE:
        return _CELL_TOKEN[cell.state]
    v = cell.value(stat)
    mark = ''
    f = cell.feature
    if f is not None and stat != 'n':
        mark = '~' if f.ema_suppressed else ('*' if f.significant else '')
        if f.low_trust:
            mark += '!'
    return _num(v) + mark


def format_block(block: MetricBlock, arms_by_label: dict, *,
                 stats: Iterable[str] = DEFAULT_STATS) -> str:
    stats = tuple(stats)
    cols = [arms_by_label[l] for l in block.arm_labels]
    route = ('routes: ' + ', '.join(sorted({a.route.value for a in cols}))
             if block.route is None else f'route: {block.route.value}')
    head = (f'\n  [{block.label}]  {len(cols)} arm(s)  {route}  '
            f'metrics from {block.source}')
    nw = _name_col(r.metric for r in block.rows)
    lines = [head, '  ' + 'METRIC'.ljust(nw) + 'STAT'.ljust(10)
             + ''.join(_trunc(a.short, _COL - 1).ljust(_COL) for a in cols)]
    for row in block.rows:
        for i, stat in enumerate(stats):
            name = _trunc(row.metric, nw - 1) if i == 0 else ''
            lines.append(
                '  ' + name.ljust(nw) + stat.ljust(10)
                + ''.join(_cell_text(row.cells[a.label], stat).ljust(_COL)
                          for a in cols))
        renamed = {c.resolved_to for c in row.cells.values() if c.resolved_to}
        if renamed:
            lines.append(f'      renamed -> {", ".join(sorted(renamed))}')
        # WHY a cell is not a number. `log_Z_learned` is ABSENT because it is
        # logged under three namespaces that are different quantities -- a
        # reader shown a bare ABSENT goes hunting for a logging bug that is not
        # there. The note is `K.resolve`'s own and is reproduced, not
        # paraphrased.
        for note in _distinct_notes(row):
            lines.append(f'      note: {note}')
    return '\n'.join(lines)


def _distinct_notes(row: MetricRow) -> list:
    """Distinct explanations attached to this row's non-LIVE cells, in column
    order. Deduplicated because N arms failing the same way is one fact."""
    out = []
    for cell in row.cells.values():
        if cell.state is CellState.LIVE or not cell.note:
            continue
        if cell.note not in out:
            out.append(cell.note)
    return out


def format_feature_table(table: FeatureTable, *,
                         stats: Iterable[str] = DEFAULT_STATS) -> str:
    """The aligned table, ALWAYS behind its comparability banner.

    The banner is emitted by this function and not by the caller, so there is no
    code path that produces a bare metric table. A reader who sees numbers here
    has seen what §4 said about reading them.
    """
    parts = [_format_gate(table)]
    if table.split_by_route:
        # TWO DIFFERENT SITUATIONS, and saying the wrong one is a lie about what
        # the table below does. Split blocks put no row across a route boundary;
        # a requested list deliberately does, and every cell on it is still
        # resolved against its own arm's route, which is what NA_ROUTE is for.
        spanning = any(b.route is None for b in table.blocks)
        parts.append(
            ('  ARMS ARE ON DIFFERENT ROUTES, so they do not share a topline. '
             + ('The metrics below were named by the caller and DO span the '
                'boundary: each cell is resolved against its own arm\'s route, '
                'so a series that does not track there says NA_ROUTE.'
                if spanning else
                'One block per route below; no row spans two of them.'))
            + '\n'
            + '\n'.join(f'    {r.value}: {", ".join(labels)}'
                        for r, labels in table.route_groups))
    win = 'all' if table.window is None else f'trailing {table.window:g} steps'
    parts.append(f'FEATURES  window={win}')
    by_label = {a.label: a for a in table.arms}
    for block in table.blocks:
        parts.append(format_block(block, by_label, stats=stats))
    parts.append(
        '\n  legend: * significant trend   ~ EMA (significance suppressed)   '
        '! low-trust\n'
        '  ABSENT = this arm does not log it.  '
        'NA_ROUTE = it does, populated, and the number does not mean here what\n'
        '  it means on a TB run -- withheld, not missing, not zero.  '
        'THIN = logged, too few ticks in the window.')
    return '\n'.join(parts)


def _format_gate(table: FeatureTable) -> str:
    if not table.blockers:
        return ('COMPARABILITY  no cross-arm §4 finding. The per-run subjects '
                'still apply -- see the §4 block.')
    lines = [f'COMPARABILITY  {len(table.blockers)} cross-arm §4 finding(s). '
             f'A comparison across arms that are not comparable is not a '
             f'weaker result, it is not a result.']
    for f in table.blockers:
        lines.append(f'  {f.state.value.upper():11s} {f.subject:38s} {f.detail}')
    return '\n'.join(lines)


def format_comparison(cmp: Comparison, *, verbose: bool = False,
                      stats: Iterable[str] = DEFAULT_STATS) -> str:
    """§4, then the sweep, then the features. The order is the point."""
    return '\n\n'.join([
        C.format_result(cmp.confounds, verbose=verbose),
        format_sweep(cmp.sweep),
        format_feature_table(cmp.features, stats=stats),
    ])
