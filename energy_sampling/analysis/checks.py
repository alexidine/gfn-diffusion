"""
Assertions over a run: R2 liveness, R14 dead sensors, section-4 confounds, R11.

`reading_runs.md` section 3 splits its principles into mechanical and judgment.
The mechanical ones are checks, and they are here. The rest surface their inputs
in `features.py` and stop -- this module emits NO VERDICTS, and a report that
concludes "the run is healthy" has failed its spec.

WHAT A CHECK IS HERE. It reports STATE and NUMBERS and never a conclusion:
"fired, 1281 of 1316 ticks" is a check, "the controller is working" is not.
FIRED is not "working" and INERT is not "broken" -- a frac pinned at its declared
value fires on every tick, and a servo whose actuator correctly never left its
rest point is inert.

TWO FAILURE MODES A CHECK MUST NOT HAVE.

  * SILENT PASS. A check that could not run because its inputs were missing says
    so -- `CheckResult.not_run(reason)` -- and never returns an empty finding
    list, which renders identically to "looked, found nothing wrong". Swallowed
    diagnostics do not fail as silence; they fail as REASSURANCE.
  * COLLAPSING NA_ROUTE. On the conditional VarGrad route the log Z and TB series
    exist and carry numbers that must not be read as they would be on a TB run.
    A check whose subject is not meaningful on the route reports NA_ROUTE. Never
    ABSENT, never zero.

EVERY metric-name and config-key literal lives in `keys.py`, so a rename upstream
is a one-file change. If you find yourself typing `'fwd/'` here, stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

import numpy as np

from . import features as F
from . import keys as K



# ===========================================================================
# Shared types and series helpers
# ===========================================================================
# WHAT A CHECK IS HERE. `reading_runs.md` §3 splits its principles into mechanical
# and judgment. A mechanical one becomes a check; everything else surfaces its
# inputs and stops. So a check reports STATE and NUMBERS and never a conclusion:
# 'fired, 1281 of 1316 ticks' is a check, 'the controller is working' is not, and
# a tool that concludes 'the run is healthy' has failed its spec.
#
# TWO FAILURE MODES A CHECK MUST NOT HAVE.
#
#   * Silent pass. A check that could not run because its inputs were missing must
#     say so -- `ran=False` with a reason -- and never return an empty finding list
#     that reads identically to 'looked, found nothing wrong'. Swallowed
#     diagnostics do not fail as silence; they fail as REASSURANCE.
#   * Collapsing NA_ROUTE. On the conditional VarGrad route the log Z and TB series
#     exist and carry numbers that must not be read as they would be on a TB run.
#     A check whose subject is NA on the route reports NA_ROUTE. Never ABSENT,
#     never zero.

class State(str, Enum):
    """What a check found about one subject.

    FIRED/INERT/NO_TRACE are R2's three answers for a declared mechanism; OFF is
    a mechanism the config does not declare, carried in the table so the report
    shows what was considered rather than only what tripped.
    """

    FIRED = 'fired'                   # declared active, trace shows activity
    INERT = 'inert'                   # declared active, trace present, no activity
    NO_TRACE = 'no_trace'             # declared active, no trace logged at all
    OFF = 'off'                       # not declared active; nothing asserted
    UNDECLARED_ACTIVE = 'undeclared'  # trace active while the config says off
    NA_ROUTE = 'na_route'             # subject is not meaningful on this route
    OK = 'ok'                         # a non-R2 check's subject, nothing to report
    FLAG = 'flag'                     # a non-R2 check's subject, condition met
    UNREADABLE = 'unreadable'         # the input needed was missing or unusable


# The states that constitute a finding -- something a reader must look at.
FINDING_STATES = (State.INERT, State.NO_TRACE, State.UNDECLARED_ACTIVE,
                  State.FLAG, State.UNREADABLE)


@dataclass(frozen=True)
class Finding:
    """One observation, with the numbers that produced it.

    `numbers` is not optional in spirit: a finding without its inputs is an
    assertion, and this package does not make assertions. Whatever the check
    compared belongs in here so the reader can disagree with it.
    """

    check: str
    subject: str
    state: State
    detail: str = ''
    numbers: dict = field(default_factory=dict)

    @property
    def is_finding(self) -> bool:
        return self.state in FINDING_STATES

    def __str__(self):
        nums = '  '.join(f'{k}={_fmt(v)}' for k, v in self.numbers.items())
        return (f'  {self.state.value.upper():11s} {self.subject:44s} '
                f'{self.detail}{("   " + nums) if nums else ""}')


def _fmt(v):
    if isinstance(v, float):
        return f'{v:.6g}'
    return str(v)


@dataclass
class CheckResult:
    """One check's whole output.

    `rows` is everything examined; `findings` is the subset a reader must look
    at. Both are kept because a report showing only findings cannot be
    distinguished from a report of a check that did not run -- and the second is
    the failure this package exists to prevent.
    """

    check: str
    ran: bool = True
    reason: str = ''
    rows: list = field(default_factory=list)
    # WHICH RUN, and under WHAT READING. A battery renders one block per (check,
    # run), and without these the blocks are indistinguishable: four arms
    # produced twelve identically-titled blocks, so a finding could not be
    # attributed to an arm. The route belongs here for a second reason -- it
    # decides every NA_ROUTE marking in the report, and a reader cannot audit a
    # withheld metric without knowing which route's rules were applied.
    run: str = ''
    header: str = ''

    @property
    def findings(self) -> list:
        return [r for r in self.rows if r.is_finding]

    def add(self, finding: Finding) -> None:
        self.rows.append(finding)

    def label(self) -> str:
        return f'{self.check}  [{self.run}]' if self.run else self.check

    @classmethod
    def not_run(cls, check: str, reason: str) -> 'CheckResult':
        """A check that could not run. NOT an empty pass: the reason is carried
        into the report, because 'no findings' and 'never looked' render the
        same and mean opposite things."""
        return cls(check=check, ran=False, reason=reason)


# ---------------------------------------------------------------------------
# Series access
# ---------------------------------------------------------------------------

def series(run, key: str) -> Optional[tuple]:
    """`(steps, values)` for a key, or None.

    History first, then the SUMMARY as a single-point series. The summary
    fallback is load-bearing rather than a convenience: `loss_coeffs/*` is a
    change-only channel -- the trainer emits a setting only when a stage
    transition moved it -- so those series carry one or two points and the local
    datastore reader drops anything shorter than three. Read from history alone,
    the most direct evidence of which loss terms are live is invisible.
    """
    if key in run.history:
        s, v = run.history[key]
        if len(s):
            return np.asarray(s, float), np.asarray(v, float)
    v = run.summary.get(key)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
        return np.asarray([run.last_step], float), np.asarray([float(v)], float)
    return None


def trailing(s: np.ndarray, v: np.ndarray, window: Optional[float]):
    """The last `window` steps of a series, or all of it when window is None."""
    if window is None or not len(s):
        return s, v
    m = s >= max(s[-1] - window, s[0])
    return s[m], v[m]


def count_active(v: np.ndarray, rule: K.Rule, floor: float = 0.0) -> dict:
    """How many ticks a trace shows the mechanism doing something.

    Three rules, because three shapes of trace occur and reading one as another
    is how a live mechanism reads inert:

      NONZERO  the trace is 0 at rest. Ticks above `floor` are active ticks.
      MOVES    the trace has a nonzero resting VALUE (a servo's set point, a
               batch size). Only departure from where it started is activity.
      COUNTER  the trace is a monotone event count. The events are its RISE, not
               its level -- a counter sitting at 26 for the whole trailing window
               did 26 calibrations, and a counter sitting at 0 did none.
    """
    v = np.asarray(v, float)
    finite = v[np.isfinite(v)]
    if not len(finite):
        return dict(n_active=0, n_ticks=len(v), events=0.0, first=float('nan'),
                    last=float('nan'))
    if rule is K.Rule.COUNTER:
        rises = np.diff(finite) > 0
        return dict(n_active=int(rises.sum()), n_ticks=len(finite),
                    events=float(finite[-1] - finite[0]),
                    first=float(finite[0]), last=float(finite[-1]))
    if rule is K.Rule.MOVES:
        moved = np.abs(finite - finite[0]) > max(floor, 0.0)
        return dict(n_active=int(moved.sum()), n_ticks=len(finite),
                    events=float(np.abs(finite - finite[0]).max()),
                    first=float(finite[0]), last=float(finite[-1]))
    active = np.abs(finite) > floor
    return dict(n_active=int(active.sum()), n_ticks=len(finite),
                events=float(np.abs(finite).max()),
                first=float(finite[0]), last=float(finite[-1]))


def route_state(key: str, route) -> bool:
    """True when `key` exists on this route but does not track on it."""
    res, = K.resolve({key}, [key], route)
    return res.state is K.KeyState.NA_ROUTE


def stage_value(config: dict, stage_index: Optional[int], tail: str):
    """A stage-scoped config value, or None when the stage is unknown."""
    if stage_index is None:
        return None
    return K._value(config, K.CFG_STAGE % (stage_index, tail))


def declared(value, how: K.Declare, wanted: str = '') -> bool:
    """Whether a config value declares its mechanism active.

    `None` is never a declaration. That matters: several of these keys exist with
    a null value to hold a slot (`health_gate_floor_metric: None`), and treating
    presence as declaration would manufacture a finding on every run carrying the
    placeholder.
    """
    if value is None:
        return False
    if how is K.Declare.EQUALS:
        return str(value) == wanted
    if how is K.Declare.NOT_NULL:
        return True
    if how is K.Declare.POSITIVE:
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return bool(str(value))


@dataclass(frozen=True)
class Context:
    """The stage a run is IN, and the route that stage puts it on.

    Resolved once and passed to every check, because getting it per-check
    invites two checks disagreeing about which stage a run is in. Both fields are
    None when the run's record does not say -- never defaulted to the terminal
    stage, since a run that died in phase 1 is exactly the run being read to find
    out why it stopped.
    """

    stage_index: Optional[int]
    stage_name: Optional[str]
    route: K.Route
    stages: tuple


def context(run) -> Context:
    """The stage and route, or UNKNOWN.

    THE ROUTE IS NEVER INFERRED FROM A STAGE THE RUN DID NOT REACH.
    `keys.detect_route` defaults a None stage index to the LAST DECLARED stage,
    which is right when a caller means "classify the terminal stage" and wrong
    here: measured on the local corpus, 11 runs declare stages and log no
    `phase` at all, and defaulting classified every one of them from a terminal
    stage they almost certainly died before reaching. The damage is not a
    mislabelled row -- NA_ROUTE marking is driven entirely by this route, so a
    route inferred from the wrong stage silently switches NA_ROUTE OFF and hands
    back TB numbers on a VarGrad run, which is the one failure H2 is about.

    The single-stage case is exempt, and only that case: with one declared stage
    there is nothing to be wrong about.
    """
    stages = tuple(K.stage_names(run.config))
    idx = K.current_stage_index(run.summary, run.config)
    if idx is None and len(stages) == 1:
        idx = 0
    if idx is None:
        return Context(stage_index=None, stage_name=None,
                       route=K.Route.UNKNOWN, stages=stages)
    return Context(stage_index=idx,
                   stage_name=K.current_stage(run.summary, run.config) or (
                       stages[idx] if idx < len(stages) else None),
                   route=K.detect_route(run.config, idx),
                   stages=stages)


def run_label(run) -> str:
    """How a run is named in a report.

    A DISPLAY NAME IS NOT UNIQUE. Nine names are shared by two or more runs in
    the local corpus (`mk_dev` alone covers eleven), so labelling by name gave
    two arms of one battery identical subject strings -- and a duplicate row
    reading `duplicate/r3_kappa00~r3_kappa00`, which names one arm twice and
    tells the reader nothing. The id disambiguates and is always present.
    """
    name = str(getattr(run, 'name', '') or '')
    run_id = str(getattr(run, 'run_id', '') or '')
    if name and run_id and name != run_id:
        return f'{name}#{run_id}'
    return name or run_id or '?'


def context_header(run, ctx: Context, window: Optional[float] = None) -> str:
    """The one line that says what reading this is. Every check block carries it,
    because route, stage and window together decide what was withheld, what was
    counted and over how much of the run -- and none of the three was previously
    stated anywhere in the report."""
    return (f'route={ctx.route.value}  '
            f'stage={ctx.stage_name or "UNKNOWN"}'
            f'[{"?" if ctx.stage_index is None else ctx.stage_index}]'
            f' of {len(ctx.stages)}  '
            f'window={"all" if window is None else f"{window:g}"}  '
            f'last_step={run.last_step:.0f}')


def as_float(v, default=float('nan')) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


def format_result(res: CheckResult, *, verbose: bool = False) -> str:
    """One check, rendered. A not-run check renders LOUDLY -- it is reported
    before any finding, because a check that did not run is a bigger hole in the
    report than anything a check that did run could say."""
    ctx_line = f'\n  {res.header}' if res.header else ''
    if not res.ran:
        return (f'\n{res.label()}  DID NOT RUN{ctx_line}\n'
                f'  {res.reason}\n'
                f'  (this is not a pass -- nothing was asserted)')
    # NA_ROUTE ROWS ARE ALWAYS SHOWN, and counted separately in the heading.
    # They are not findings -- nothing is wrong -- but they are not silence
    # either: withheld at default verbosity, a conditional VarGrad run's R11
    # rendered BYTE-IDENTICALLY to a clean TB run, which is the collapse H2
    # forbids, arrived at through the report rather than through the state.
    na = [r for r in res.rows if r.state is State.NA_ROUTE]
    shown = res.rows if verbose else res.findings + [
        r for r in na if r not in res.findings]
    head = (f'\n{res.label()}  {len(res.findings)} finding(s)'
            f'{f", {len(na)} NA_ROUTE" if na else ""} '
            f'of {len(res.rows)} examined{ctx_line}')
    if not shown:
        return head + '\n  (nothing to report; run with -v for the full table)'
    return head + '\n' + '\n'.join(str(r) for r in shown)



# ===========================================================================
# R2 -- confirm the thing ever fired
# ===========================================================================
# `reading_runs.md` R2: a frac below its deactivation threshold, a gate that never
# tripped, a knob retired upstream, a servo silent because its sensor is
# structurally zero. An inert mechanism is the most common explanation for a null
# result, and it INVALIDATES THE ARM rather than answering it.
#
# Three families of subject, because a run declares its mechanisms in three
# different places and the weakest evidence is the one that is easiest to read:
#
#   1. `K.MECHANISMS` -- the verified registry: a declaring config key and the
#      trace that proves the mechanism ran.
#   2. the loss coefficients the trainer is ACTUALLY holding (`loss_coeffs/*`),
#      compared against the config's effective values for the current stage. This
#      is the strongest liveness evidence in the run: it is what the optimiser
#      saw, after the stage's overrides, not what the yaml asked for.
#   3. the current stage's exit conditions -- a gate cannot trip if its streak
#      never reached 1, and it cannot trip AT ALL if the metric it is defined on
#      is not logged.
#
# WHAT THIS CHECK REFUSES TO SAY. FIRED is not "working" and INERT is not
# "broken": a frac pinned at its declared value fires on every tick and a servo
# whose actuator correctly never left its rest point is inert. The row carries the
# declaration, the trace it read and the counts; what that means about the run is
# the reader's.
#
# FOUR TRAPS ENCODED HERE, each of which produced a wrong reading while this was
# being built and measured against the local corpus:
#
#   * A DECLARING KEY THAT IS ABSENT IS NOT A DECLARATION OF "OFF". Stage-scoped
#     keys live on the stage that uses them -- `flags_mle_gate` is on stage 0, and
#     a run reading in stage 1 finds no key while the streak counter, which spans
#     the whole run, still carries stage 0's activity. Config eras differ too
#     (`adaptive_lr.enabled` became `adaptive_lr.seed_lr`). Treating absent as
#     "the config says off" manufactured UNDECLARED_ACTIVE on 85 of 115 real runs.
#     So UNDECLARED_ACTIVE requires the key to be PRESENT and falsy: an absent key
#     makes no claim, and there is nothing for the trace to contradict.
#   * A COUNTER'S LEVEL IS THE EVIDENCE, NOT ITS RISE. `base.count_active` counts
#     rises, which are zero in any window that opens after the last event -- so a
#     run that calibrated 26 times reads INERT under a trailing window. R2 asks
#     whether the thing EVER fired, and for a monotone counter the last value
#     answers that on its own.
#   * A ONE-POINT SERIES CANNOT SHOW MOVEMENT. `base.series` falls back to the
#     summary, and a MOVES rule on a single point is always zero -- which renders
#     as INERT and reads as a dead controller. That is UNREADABLE, not inert.
#   * NA_ROUTE IS CHECKED BEFORE ANYTHING ELSE and stops the row. A trace that
#     does not track on this route cannot show inertness or activity, and a zero
#     printed there is worse than a hole.

_R2_CHECK = 'R2 liveness'

# `Mechanism.scope`'s two values. Not metric names and not config keys, so they
# do not belong in keys.py -- but they are still spelled once here rather than
# inline, so a third scope fails loudly at one site.
_R2_STAGE_SCOPE = 'stage'

# Below this, a MOVES trace has no baseline to have departed from.
_R2_MIN_MOVES_TICKS = 2

# Config-effective vs live-logged coefficients come from the same float through
# two serialisations (yaml -> config, python -> summary json), so anything
# outside this is a real disagreement and not a round-trip.
_R2_COEFF_RTOL = 1e-9
_R2_COEFF_ATOL = 1e-12


def _r2_window(window: Optional[float]) -> str:
    return 'all' if window is None else f'trailing {window:g} steps'


# ---------------------------------------------------------------------------
# Reading one trace
# ---------------------------------------------------------------------------

def _r2_read(run, key: str, rule: K.Rule, floor: float,
             window: Optional[float]) -> Optional[dict]:
    """`count_active` over the window, or None when the trace is not logged.

    None means ABSENT from history AND summary -- `base.series` covers both, and
    the summary half is not a nicety: several of these traces are change-only
    channels that the local datastore reader drops for being shorter than three
    points."""
    got = series(run, key)
    if got is None:
        return None
    s, v = trailing(got[0], got[1], window)
    out = dict(count_active(v, rule, floor))
    out['key'] = key
    return out


def _r2_fired(c: dict, rule: K.Rule) -> bool:
    """Whether a reading shows the mechanism ever did something.

    The counter branch deliberately disagrees with `n_active`: R2's question is
    whether the thing EVER fired, and a monotone event counter answers that with
    its LEVEL. Reading only the rises makes every completed calibration invisible
    the moment a trailing window is passed."""
    if rule is K.Rule.COUNTER:
        last = as_float(c.get('last'), 0.0)
        return c['n_active'] > 0 or last > 0
    return c['n_active'] > 0


def _r2_readable(c: dict, rule: K.Rule) -> bool:
    """Whether the reading can answer at all. A MOVES rule measures departure
    from the first tick, so one tick is not a null result -- it is no result."""
    return not (rule is K.Rule.MOVES and c['n_ticks'] < _R2_MIN_MOVES_TICKS)


def _r2_numbers(c: Optional[dict], floor: float,
                window: Optional[float]) -> dict:
    """The triple R2 owes the reader -- mechanism, fired?, n_steps_active --
    plus what it was measured against.

    `n_steps_active` counts LOGGED TICKS, not training steps: the trace is
    sampled at the reporting cadence and nothing here can recover the steps
    between two ticks. `n_ticks` is beside it so the two are read as a ratio."""
    out = {'n_steps_active': 0, 'n_ticks': 0, 'window': _r2_window(window)}
    if floor:
        out['floor'] = float(floor)
    if c is None:
        return out
    out['n_steps_active'] = int(c['n_active'])
    out['n_ticks'] = int(c['n_ticks'])
    # `magnitude` is rule-dependent by construction: max |v| for NONZERO, the
    # largest departure from the first tick for MOVES, the counter's rise for
    # COUNTER. Named vaguely on purpose -- calling it 'events' on a NONZERO
    # trace would be a lie.
    out['magnitude'] = float(c['events'])
    out['last'] = float(c['last'])
    return out


def _r2_best(readings: list, rule: K.Rule) -> dict:
    """The trace to report when a mechanism has several. ANY of them firing
    counts as fired, so a firing reading outranks a silent one; among equals the
    registry's own order wins, since it lists the actuator first."""
    for c in readings:
        if _r2_fired(c, rule):
            return c
    for c in readings:
        if _r2_readable(c, rule):
            return c
    return readings[0]


# ---------------------------------------------------------------------------
# Family 1 -- the mechanism registry
# ---------------------------------------------------------------------------

def _r2_declared_key(ctx: Context, m: K.Mechanism) -> Optional[str]:
    """The config key that declares `m`, or None when the stage is unknown."""
    if m.scope == _R2_STAGE_SCOPE:
        if ctx.stage_index is None:
            return None
        return K.CFG_STAGE % (ctx.stage_index, m.declared_by)
    return m.declared_by


def _r2_floor(config: dict, ctx: Context, m: K.Mechanism) -> float:
    """The activation floor, from `threshold_key`. Zero when the key is not in
    this config -- the conservative direction: a floor of zero can only turn an
    INERT row into a FIRED one, and a manufactured finding costs more than a
    missed one on a check whose value depends on being believed."""
    if not m.threshold_key:
        return 0.0
    raw = (stage_value(config, ctx.stage_index, m.threshold_key)
           if m.scope == _R2_STAGE_SCOPE else K._value(config, m.threshold_key))
    return as_float(raw, 0.0)


def _r2_mechanism(res: CheckResult, run, ctx: Context, m: K.Mechanism,
                  window: Optional[float]) -> None:
    config = run.config or {}
    key = _r2_declared_key(ctx, m)
    if key is None:
        return  # stage-scoped, stage unknown; the skipped row says so once

    # NA_ROUTE FIRST, and it stops the row. A trace that does not track on this
    # route can show neither inertness nor activity, and both readings would be
    # asserted off numbers that are real but do not mean what they look like.
    na = [t for t in m.trace if route_state(t, ctx.route)]
    if na:
        res.add(Finding(_R2_CHECK, m.name, State.NA_ROUTE,
                        f'trace does not track on {ctx.route.value}: '
                        f'{", ".join(na)}; liveness not asserted',
                        {'n_steps_active': 0, 'n_ticks': 0,
                         'window': _r2_window(window)}))
        return

    present = key in config
    value = K._value(config, key)
    dec = declared(value, m.declares, m.declares_value)
    floor = _r2_floor(config, ctx, m)
    readings = [c for c in (_r2_read(run, t, m.rule, floor, window)
                            for t in m.trace) if c is not None]
    shown = f'{key}={value!r}'

    if not readings:
        state = State.NO_TRACE if dec else State.OFF
        detail = (f'{shown}; no trace logged -- looked for '
                  f'{", ".join(m.trace)}')
        res.add(Finding(_R2_CHECK, m.name, state, detail,
                        _r2_numbers(None, floor, window)))
        return

    c = _r2_best(readings, m.rule)
    nums = _r2_numbers(c, floor, window)
    nums['n_traces_logged'] = len(readings)
    fired = _r2_fired(c, m.rule)

    if dec:
        if fired:
            state = State.FIRED
            detail = (f'{shown}; {c["key"]} active on '
                      f'{nums["n_steps_active"]} of {nums["n_ticks"]} tick(s)')
        elif not _r2_readable(c, m.rule):
            # NOT inert. A one-point series under a MOVES rule has no baseline,
            # so 'never moved' is not something this data can say.
            state = State.UNREADABLE
            detail = (f'{shown}; {c["key"]} has {c["n_ticks"]} point(s) and the '
                      f'rule is {m.rule.value} -- movement needs a baseline, so '
                      f'liveness is unread, not absent')
        else:
            state = State.INERT
            detail = (f'{shown}; {c["key"]} shows no activity over '
                      f'{_r2_window(window)}')
        if m.note and state in (State.INERT, State.NO_TRACE):
            detail += f' [{m.note}]'
    elif fired and present:
        # The config carries the knob and sets it off, and the trace disagrees.
        # §4: an arm that was not running the code it was written to test, which
        # voids its hypotheses outright.
        state = State.UNDECLARED_ACTIVE
        detail = (f'{shown} does not declare this active, but {c["key"]} shows '
                  f'activity')
    else:
        state = State.OFF
        detail = (f'{shown}; not declared active' if present else
                  f'{key} is not in this config -- the knob takes its default '
                  f'and the config asserts nothing')
    res.add(Finding(_R2_CHECK, m.name, state, detail, nums))


# ---------------------------------------------------------------------------
# Family 2 -- the loss coefficients actually being held
# ---------------------------------------------------------------------------

def _r2_live_coeff(run, key: str) -> Optional[float]:
    """The last logged value of one `loss_coeffs/` entry, or None.

    NOT WINDOWED, and that is `K.LOSS_COEFF_IS_SUMMARY_ONLY`'s doing: this is a
    change-only channel emitted at eval time and only when a stage transition
    moved the value, so the series is one or two points sitting wherever the
    last change happened. A trailing window would delete the evidence and the
    check would report the strongest liveness signal in the run as missing."""
    got = series(run, key)
    if got is None:
        return None
    v = got[1]
    return float(v[-1]) if len(v) else None


def _r2_loss_coeffs(res: CheckResult, run, ctx: Context) -> None:
    config = run.config or {}
    eff = K.effective_loss_coeffs(config, ctx.stage_index)
    # The mode filter is load-bearing, not tidiness: the canonical base sets
    # `replay_loss_coeffs_tb: 1.0`, so counting modes the stage never evaluates
    # reports the MLE warm-start as holding a live TB term.
    for mode in K.active_modes(config, ctx.stage_index):
        for name in sorted(eff.get(mode, {})):
            _r2_one_coeff(res, run, ctx, mode, name, eff[mode][name])


def _r2_one_coeff(res: CheckResult, run, ctx: Context, mode: str, name: str,
                  raw) -> None:
    key = K.LOSS_COEFF_TRACE % (mode, name)
    if route_state(key, ctx.route):
        res.add(Finding(_R2_CHECK, key, State.NA_ROUTE,
                        f'does not track on {ctx.route.value}',
                        {'n_steps_active': 0, 'n_ticks': 0}))
        return

    cfg = as_float(raw)
    live = _r2_live_coeff(run, key)
    nums = {'config': cfg, 'live': live if live is not None else float('nan'),
            'n_steps_active': int(live not in (None, 0.0)),
            'n_ticks': int(live is not None)}

    if np.isnan(cfg):
        # A coefficient whose configured value is not a number cannot be
        # compared, and comparing it as zero would invent an agreement.
        res.add(Finding(_R2_CHECK, key, State.UNREADABLE,
                        f'config value {raw!r} is not numeric', nums))
        return
    if live is None:
        state = State.NO_TRACE if cfg > 0 else State.OFF
        res.add(Finding(_R2_CHECK, key, state,
                        f'config sets {cfg:g}; the trainer never logged this '
                        f'coefficient, so what it held is unknown', nums))
        return
    if cfg > 0 and live == 0.0:
        res.add(Finding(_R2_CHECK, key, State.INERT,
                        f'config sets {cfg:g} and the trainer is holding 0 -- '
                        f'the term is not in the loss', nums))
        return
    if not np.isclose(cfg, live, rtol=_R2_COEFF_RTOL, atol=_R2_COEFF_ATOL):
        # Either direction. A knob retired upstream reads as config > live; a
        # stage override the reader did not know about reads as live > config.
        res.add(Finding(_R2_CHECK, key, State.FLAG,
                        f'config-effective {cfg:g} but the trainer is holding '
                        f'{live:g}', nums))
        return
    res.add(Finding(_R2_CHECK, key,
                    State.FIRED if live != 0.0 else State.OFF,
                    f'held at {live:g}', nums))


# ---------------------------------------------------------------------------
# Family 3 -- the current stage's exit conditions
# ---------------------------------------------------------------------------

def _r2_exits(res: CheckResult, run, ctx: Context,
              window: Optional[float]) -> None:
    config = run.config or {}
    j = 0
    while True:
        metric = K._value(config, K.CFG_STAGE_EXIT_METRIC % (ctx.stage_index, j))
        if metric is None:
            break
        _r2_exit(res, run, ctx, j, str(metric), window)
        j += 1


def _r2_exit(res: CheckResult, run, ctx: Context, j: int, metric: str,
             window: Optional[float]) -> None:
    resolution, = K.resolve(run.available_keys(), [metric], ctx.route)
    _r2_exit_metric(res, ctx, j, metric, resolution)

    subject = f'exit[{j}] {metric}'
    streak = K.EXIT_STREAK_TRACE % K.metric_tag(metric)
    if resolution.state is K.KeyState.NA_ROUTE:
        res.add(Finding(_R2_CHECK, subject, State.NA_ROUTE,
                        f'the condition is defined on {metric}, which does not '
                        f'track on {ctx.route.value}',
                        {'n_steps_active': 0, 'n_ticks': 0}))
        return

    c = _r2_read(run, streak, K.Rule.NONZERO, 0.0, window)
    nums = _r2_numbers(c, 0.0, window)
    nums['metric_state'] = resolution.state.value
    if c is None:
        res.add(Finding(_R2_CHECK, subject, State.NO_TRACE,
                        f'{streak} is not logged -- whether the condition ever '
                        f'held cannot be read', nums))
        return
    if _r2_fired(c, K.Rule.NONZERO):
        res.add(Finding(_R2_CHECK, subject, State.FIRED,
                        f'{streak} reached {nums["magnitude"]:g}', nums))
        return
    # SCOPED TO THE WINDOW, like the registry family already is. Unqualified,
    # this said "the condition never held once" -- a claim about the whole run --
    # on windowed rows whose own `window` field contradicted it, and measured on
    # a real run whose streak had reached 4 on 14 of 1001 ticks outside the
    # window. A windowed INERT means "not active lately"; only an unwindowed one
    # means "never".
    if window is None:
        said = (f'{streak} never reached 1 -- the condition did not hold on a '
                f'single evaluation in the whole run')
    else:
        said = (f'{streak} never reached 1 over {_r2_window(window)} -- the '
                f'condition did not hold in this window; earlier ticks were '
                f'not read')
    res.add(Finding(_R2_CHECK, subject, State.INERT, said, nums))


def _r2_exit_metric(res: CheckResult, ctx: Context, j: int, metric: str,
                    resolution) -> None:
    """Whether the metric the gate is DEFINED ON is logged at all.

    Reported as its own row rather than folded into the streak, because they are
    independent facts and a merged row lets either hide the other. What is
    reported is exactly what `K.resolve` says: the protocol resolves some of
    these internally against names wandb never sees (`gates/mle_flat` is
    published to the protocol and logged only as its streak), so 'not logged'
    here is a statement about the RUN RECORD, which is all this package reads."""
    subject = f'exit[{j}].metric {metric}'
    nums = {'metric_state': resolution.state.value,
            'resolved_to': resolution.key or ''}
    if resolution.state is K.KeyState.NA_ROUTE:
        res.add(Finding(_R2_CHECK, subject, State.NA_ROUTE, resolution.note,
                        nums))
        return
    if resolution.state is K.KeyState.LIVE:
        res.add(Finding(_R2_CHECK, subject, State.OK,
                        f'logged as {resolution.key}', nums))
        return
    res.add(Finding(_R2_CHECK, subject, State.NO_TRACE,
                    f'the gate is defined on {metric}, which is not in this '
                    f'run record -- {resolution.note}', nums))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def check_r2(run, *, ctx: Optional[Context] = None,
             window: Optional[float] = None) -> CheckResult:
    """Every mechanism the config declares active, against the trace that proves
    it ran.

    `window` trims every trace except the loss coefficients, and every row
    carries it, because a windowed INERT means 'not active lately' and an
    unwindowed one means 'never' -- the same word for two different claims.

    A run whose stage cannot be read still gets its global mechanisms, with a
    LOUD row naming what was skipped. Returning `not_run` there would throw away
    real answers; returning the globals silently would let a third of the check
    vanish into a table that looks complete.
    """
    ctx = ctx or context(run)
    if not (run.config or {}):
        return CheckResult.not_run(
            _R2_CHECK, 'no config -- nothing declares a mechanism, so there is '
                       'nothing to confirm fired. This is not "no mechanism is '
                       'inert".')

    res = CheckResult(check=_R2_CHECK)
    if ctx.stage_index is None:
        skipped = sum(1 for m in K.MECHANISMS if m.scope == _R2_STAGE_SCOPE)
        res.add(Finding(
            _R2_CHECK, 'stage', State.UNREADABLE,
            f'{K.STAGE_METRIC} is not in the summary, so the stage is unknown: '
            f'{skipped} stage-scoped mechanism(s), the loss coefficients and '
            f'the exit conditions were NOT examined',
            {'n_mechanisms': len(K.MECHANISMS), 'n_skipped': skipped}))

    for m in K.MECHANISMS:
        _r2_mechanism(res, run, ctx, m, window)
    if ctx.stage_index is not None:
        _r2_loss_coeffs(res, run, ctx)
        _r2_exits(res, run, ctx, window)
    return res



# ===========================================================================
# R14 -- a pinned metric is a dead sensor
# ===========================================================================
# `reading_runs.md` R14: "Zero spread, a value bound at its clip, a threshold
# annealed below its own noise floor, a censored estimator reported at its
# censoring bound. None of these are readings." Scope: the series a controller
# READS.
#
# THE SUBJECT LIST COMES FROM THE CONFIG. This codebase's configs name their own
# sensors -- the balance controller's `balance_metrics_<mode>`, the LR
# controller's `lr_sensor_metrics_<j>`, the stage's `exit_<j>_metric`, the buffer
# servo's numerator and denominator, the anchor gate's ceiling and floor. Reading
# them is what makes this check general rather than a hardcoded list that rots
# one rename after it is written.
#
# It is also what keeps the check honest in the other direction. A legitimately
# constant SET POINT is not a dead sensor: `protocol/rt_setpoint` never moves
# because it is config being echoed, not a reading, and a check built from "every
# flat series in the run" reports it every time. Only a series something reads is
# a subject here.
#
# WHAT IS REPORTED, AND WHAT IS NOT. Each row carries the numbers that produced
# it -- tick counts, extremes, the fraction pinned, the bar and the sigma it was
# compared against. What a pinned sensor means for the run is the reader's; this
# check says which condition holds and hands over the arithmetic.
#
# TRAPS THIS ENCODES.
#
#   * H5 -- `tracker/*` are EMA OUTPUTS. Smoothing manufactures autocorrelation
#     and lowers variance, so an EMA series can read as pinned when the filter is
#     what is flat. The row says so; it does not suppress the finding, because a
#     smoothed sensor a controller is steering on is still a sensor whose spread
#     the controller cannot see.
#   * SHORTNESS IS NOT A FINDING. A series with a handful of ticks has no
#     measurable spread, and calling that a dead sensor fires on every run read
#     early and on every config-only capture. Under `_R14_MIN_TICKS` the row says
#     how many ticks there were and asserts nothing about the sensor.
#   * CENSORING IS CHECKED BEFORE PINNING. A t-statistic clamped to +/-99 for
#     most of a run is zero-spread too, and reporting it as "flat" loses the one
#     fact that explains it: `ray_calibration` clamps before logging, so the
#     values above the bound were never in the record to begin with.
#   * ABSENT IS A FINDING, NA_ROUTE IS NOT. A controller reading a series the run
#     does not log is a dead sensor in the most literal sense. A series that is
#     not meaningful on this route is NOT that -- it is logged, populated, and
#     not this route's to read, and flagging it would send the reader hunting a
#     logging bug that is not there.
#
# R13 -- WHERE THE BAR COMES FROM. "Never ratchet a threshold below a floor you
# have not measured." The protocol publishes each live bar in the METRIC'S OWN
# UNITS at `K.EXIT_THRESHOLD_TRACE`, and this check finds the pairs by inverting
# `K.metric_tag` against the keys the run actually logged rather than by asking
# the config which conditions exist. That is deliberate: measured against
# `protocol.py`, the publisher of `protocol/thr_*` is the LEXICOGRAPHIC BALANCE
# controller's rules, not the stage exit block (which publishes only
# `protocol/exit_streak_*`). Deriving the pairs from the exit config alone finds
# nothing on a real run and reports it as a clean bill.
#
# The comparison span is the OVERLAP -- the metric restricted to the steps the
# bar actually covers. A bar that appears at step 9,570 compared against sigma
# from the whole run is compared against a regime it was never in, and on real
# data that flips the answer both ways.

_R14_CHECK = 'R14 dead sensor'

# Ticks at one single value, as a fraction, before the sensor is reported as
# pinned. Half is deliberately permissive: a controller input that spends more
# than half its ticks at exactly one number is not resolving the thing it
# steers on, whatever the other half does. Measured across the fixture corpus,
# no config-named sensor comes within an order of magnitude of this except the
# censored ray t-statistics, which are the case it exists for.
_R14_PIN_FRACTION = 0.5

# Below this many finite ticks, spread is not a measurement. The row still
# renders -- with the count -- because silence and "no spread" look identical
# in a report and mean opposite things.
_R14_MIN_TICKS = 8

# Enumeration ceiling for the `_<j>_` config families. A gap in the indices
# ends the family, per how the flattener writes them; the cap only stops a
# malformed config spinning.
_R14_MAX_INDEX = 64

# Config clip values are stored float32 in a captured history, so equality is
# a tolerance, not `==`.
_R14_CLIP_RTOL = 1e-5

# Robust-sigma scale factor: MAD -> sd for a normal. Carried alongside the std
# so a sigma inflated by one excursion is visible as such.
_R14_MAD_TO_SD = 1.4826

# Condition tags. First token of `detail`, so a row's condition is legible in
# the rendered table and assertable in a test without parsing prose.
_R14_TAG_ABSENT = 'NOT LOGGED'
# Distinct from NOT LOGGED because the reader's action is different: for a hole
# you go looking for a logging bug, for this you name the namespace you meant.
# Tagging the ambiguous case NOT LOGGED stated the opposite of what was true --
# the quantity IS logged, under several names that are different quantities.
_R14_TAG_AMBIGUOUS = 'AMBIGUOUS NAME'
# A bar cannot be compared with a floor that is not a measurement. Sigma of zero
# means the gated metric is itself pinned, and `bar < 0` is false for every bar,
# so killing the sensor turned an existing FLAG into OK -- the check got QUIETER
# as the run got worse.
_R14_TAG_NO_FLOOR = 'NO MEASURABLE FLOOR'
_R14_TAG_NAN = 'NO FINITE VALUES'
_R14_TAG_THIN = 'TOO FEW TICKS'
_R14_TAG_CENSORED = 'CENSORED'
_R14_TAG_CLIP = 'AT CONFIG CLIP'
_R14_TAG_FLAT = 'ZERO SPREAD'
_R14_TAG_PINNED = 'PINNED AT EXTREMUM'
_R14_TAG_BAR = 'BAR BELOW SIGMA'
_R14_TAG_UNPAIRED = 'BAR WITHOUT A METRIC'


def _r14_window_desc(window: Optional[float]) -> str:
    return 'all' if window is None else f'trailing {window:g} steps'


def _r14_str(v) -> Optional[str]:
    """A config entry that names a metric, or None. `None` holds a slot on
    several of these keys (`health_gate_floor_metric`), and a slot-holder is
    not a sensor."""
    return v if isinstance(v, str) and v else None


def _r14_subjects(run, ctx: Context) -> dict:
    """`metric -> [role, ...]` for every series a controller on this run reads.

    Roles are joined into one row per metric rather than emitted per role: two
    controllers reading the same series is one sensor, and two rows for it would
    double-count the same finding.
    """
    cfg = run.config or {}
    found: dict = {}

    def add(role: str, value) -> None:
        metric = _r14_str(value)
        if metric is not None:
            found.setdefault(metric, []).append(role)

    idx = ctx.stage_index
    if idx is not None:
        for mode in K.MODES:
            add(f'balance.{mode}',
                K._value(cfg, K.CFG_STAGE_BALANCE_METRIC % (idx, mode)))
        for j in range(_R14_MAX_INDEX):
            key = K.CFG_STAGE_LR_SENSOR_METRIC % (idx, j)
            if key not in cfg:
                break
            add(f'lr_sensor[{j}]', K._value(cfg, key))
        for j in range(_R14_MAX_INDEX):
            key = K.CFG_STAGE_EXIT_METRIC % (idx, j)
            if key not in cfg:
                break
            add(f'exit[{j}]', K._value(cfg, key))
        add('servo.num', K._value(cfg, K.CFG_STAGE_BUFFER_SERVO_NUM % idx))
        add('servo.den', K._value(cfg, K.CFG_STAGE_BUFFER_SERVO_DEN % idx))

    # Global, not stage-scoped.
    add('anchor.ceiling', K._value(cfg, K.CFG_ANCHOR_GATE_CEILING_METRIC))
    add('anchor.floor', K._value(cfg, K.CFG_ANCHOR_GATE_FLOOR_METRIC))

    # Censored estimators are subjects wherever they appear. `K.CENSORED` is a
    # registry of quantities this codebase CLAMPS BEFORE LOGGING, and nothing
    # gets clamped that nothing reads -- so the registry is a sensor list the
    # config does not have to repeat.
    prefixes = tuple(K.CENSORED)
    for key in sorted(run.available_keys()):
        if key.startswith(prefixes):
            add('censored', key)
    return found


def _r14_censor_bound(key: str) -> Optional[float]:
    """The magnitude `key` is clamped to before logging, or None."""
    for prefix, mag in K.CENSORED.items():
        if key.startswith(prefix):
            return float(mag)
    return None


def _r14_clip_values(config: dict) -> dict:
    """`config key -> clip magnitude` for the clips a series can pin against."""
    out = {}
    for name in K.CFG_CLIP_KEYS:
        try:
            v = float(K._value(config, name))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v != 0.0:
            out[name] = v
    return out


def _r14_sigma(s: np.ndarray, v: np.ndarray) -> tuple:
    """`(sigma, sigma_robust)` of the DETRENDED residual.

    Detrended because a metric still descending has a spread dominated by the
    descent, and a bar compared against that is compared against progress
    rather than against noise. Both scales are reported: the std is the stated
    comparison, and one excursion moves it a long way, so the MAD-derived scale
    travels beside it as the check on itself."""
    _, resid = F.theil_sen(s, v)
    sigma = float(np.std(resid))
    mad = float(np.median(np.abs(resid - np.median(resid))))
    return sigma, _R14_MAD_TO_SD * mad


def _r14_finite(got) -> tuple:
    s, v = got
    m = np.isfinite(v)
    return np.asarray(s, float)[m], np.asarray(v, float)[m]


# ---------------------------------------------------------------------------
# One sensor
# ---------------------------------------------------------------------------

def _r14_sensor(res: CheckResult, run, ctx: Context, window: Optional[float],
                metric: str, roles: list) -> None:
    subject = f'{"+".join(roles)}={metric}'
    # RESOLVED TWICE, deliberately. `K.resolve` tests the NA pattern BEFORE
    # presence -- correct there, since NA_ROUTE's defining property is that the
    # key IS present -- but asking it route-first here made a genuinely ABSENT
    # sensor on a VarGrad run report NA_ROUTE, i.e. "logged, populated, and not
    # this route's to read", about a series that is not logged at all. The row
    # asserted a falsehood and swallowed every dead-sensor condition with it.
    # Presence is route-blind; meaning is not.
    blind, = K.resolve(run.available_keys(), [metric], K.Route.UNKNOWN)
    resn, = K.resolve(run.available_keys(), [metric], ctx.route)

    if blind.state is K.KeyState.ABSENT:
        ambiguous = 'ambiguous' in blind.note
        res.add(Finding(_R14_CHECK, subject, State.FLAG,
                        f'{_R14_TAG_AMBIGUOUS if ambiguous else _R14_TAG_ABSENT}'
                        f' -- a controller reads this series and '
                        f'{"the name matches several logged keys" if ambiguous else "the run does not log it"}'
                        f': {blind.note}'))
        return

    if resn.state is K.KeyState.NA_ROUTE:
        # NOT a flag, and not ABSENT. The key is there -- established above,
        # route-blind -- and carries numbers; what is true is that this route is
        # not the one they mean.
        res.add(Finding(_R14_CHECK, subject, State.NA_ROUTE,
                        f'{resn.note} -- a controller reads it, and its spread '
                        f'is not this route\'s to interpret'))
        return
    resn = blind if resn.state is not K.KeyState.LIVE else resn

    key = resn.key
    notes = []
    if resn.resolved_to:
        notes.append(f'read as {resn.resolved_to}')
    if K.is_ema(key):
        notes.append('EMA output (H5) -- smoothing manufactures autocorrelation '
                     'and lowers variance, so flat here may be the filter')
    if key in K.LOW_TRUST:
        notes.append('low-trust: carried, never ranked on')

    def emit(state, tag, said, numbers=None):
        res.add(Finding(_R14_CHECK, subject, state,
                        ' | '.join([f'{tag} -- {said}' if tag else said] + notes),
                        numbers or {}))

    got = series(run, key)
    if got is None:
        emit(State.UNREADABLE, '',
             f'{key} resolved LIVE but holds no numeric series and no scalar '
             f'summary value')
        return

    s, v = _r14_finite(trailing(*got, window))
    n_raw = len(got[0])
    base = {'n_ticks': len(s), 'n_logged': n_raw,
            'window': _r14_window_desc(window)}

    if not len(s):
        emit(State.FLAG, _R14_TAG_NAN,
             f'{n_raw} logged tick(s), not one of them finite', base)
        return

    lo, hi = float(np.min(v)), float(np.max(v))
    frac_lo = float(np.mean(v == lo))
    frac_hi = float(np.mean(v == hi))
    numbers = dict(base, minimum=lo, maximum=hi, spread=hi - lo,
                   frac_at_min=frac_lo, frac_at_max=frac_hi, last=float(v[-1]))

    if len(s) < _R14_MIN_TICKS:
        # Deliberately NOT a dead-sensor finding. Say the count and stop.
        emit(State.UNREADABLE, _R14_TAG_THIN,
             f'{len(s)} finite tick(s) over {_r14_window_desc(window)}; '
             f'{_R14_MIN_TICKS} needed before spread is a measurement, so '
             f'nothing is asserted about this sensor', numbers)
        return

    # --- censoring first. A clamped estimator sitting at its bound is flat too,
    # and reporting the flatness loses the reason for it.
    bound = _r14_censor_bound(key)
    if bound is not None:
        frac = float(np.mean(np.abs(v) >= bound))
        numbers = dict(numbers, censor_bound=bound, frac_at_bound=frac)
        if frac > _R14_PIN_FRACTION:
            emit(State.FLAG, _R14_TAG_CENSORED,
                 f'at the +/-{bound:g} censoring bound on {frac:.1%} of ticks '
                 f'-- the values beyond it were clamped before logging and are '
                 f'not in the record', numbers)
            return

    # --- a value bound at its clip.
    for name, clip in _r14_clip_values(run.config or {}).items():
        frac = float(np.mean(np.isclose(np.abs(v), clip, rtol=_R14_CLIP_RTOL,
                                        atol=0.0)))
        if frac > _R14_PIN_FRACTION:
            emit(State.FLAG, _R14_TAG_CLIP,
                 f'at the {name}={clip:g} clip on {frac:.1%} of ticks',
                 dict(numbers, clip_key=name, clip=clip, frac_at_clip=frac))
            return

    # --- zero spread.
    if hi == lo:
        emit(State.FLAG, _R14_TAG_FLAT,
             f'constant at {lo:g} across all {len(s)} tick(s) of '
             f'{_r14_window_desc(window)}', numbers)
        return

    # --- pinned at an extremum.
    if max(frac_lo, frac_hi) > _R14_PIN_FRACTION:
        at_top = frac_hi >= frac_lo
        emit(State.FLAG, _R14_TAG_PINNED,
             f'sits at its {"maximum" if at_top else "minimum"} '
             f'{hi if at_top else lo:g} on {max(frac_lo, frac_hi):.1%} of ticks',
             numbers)
        return

    emit(State.OK, '', f'spread {hi - lo:.4g} over {len(s)} tick(s)', numbers)


# ---------------------------------------------------------------------------
# R13 -- the bar against the floor
# ---------------------------------------------------------------------------

def _r14_bars(run) -> list:
    """`(bar_key, [gated metric, ...])` for every live bar the protocol
    published.

    The gated metric is recovered by inverting `K.metric_tag` against the keys
    the run logged. Two different metrics can tag identically (`a/b_c` and
    `a_b/c`), so the candidates are counted and never chosen between."""
    avail = run.available_keys()
    prefix = K.EXIT_THRESHOLD_TRACE % ''
    out = []
    for bar in sorted(k for k in avail if k.startswith(prefix)):
        tag = bar[len(prefix):]
        out.append((bar, sorted(m for m in avail
                                if m != bar and K.metric_tag(m) == tag)))
    return out


def _r14_configured_bars(run, ctx: Context) -> list:
    """`(label, bar, metric)` for every STATICALLY CONFIGURED exit bar.

    The published `protocol/thr_*` series is the annealed bar, and only the
    lexicographic balance controller publishes it -- measured on the local
    corpus, 94 runs configure an exit bar and ONE publishes a thr_* series. So
    deriving R13's pairs from the published bar alone leaves the check dark on
    almost every run while still returning a full sensor table, which renders as
    'R14 examined this run'. The configured number is the bar the stage actually
    gates on, and it is in the config on every one of those runs.
    """
    cfg = run.config or {}
    idx = ctx.stage_index
    if idx is None:
        return []
    out = []
    for j in range(_R14_MAX_INDEX):
        key = K.CFG_STAGE_EXIT_METRIC % (idx, j)
        if key not in cfg:
            break
        metric = _r14_str(K._value(cfg, key))
        if metric is None:
            continue
        for sense, cfg_key in (('above', K.CFG_STAGE_EXIT_ABOVE),
                               ('below', K.CFG_STAGE_EXIT_BELOW)):
            bar = K._value(cfg, cfg_key % (idx, j))
            if bar is None:
                continue
            try:
                out.append((f'exit[{j}] {sense}', float(bar), metric))
            except (TypeError, ValueError):
                continue
    return out


def _r14_r13_configured(res: CheckResult, run, ctx: Context,
                        window: Optional[float],
                        label: str, bar: float, metric: str) -> None:
    """A configured bar against the measured noise floor of the metric it gates.

    R13: never ratchet a threshold below a floor you have not measured. A bar
    inside the metric's own scatter is crossed by noise, in either sense -- an
    `above` rule trips on a fluctuation, a `below` rule can never be reliably
    met -- so one comparison covers both and the sense is named in the row.
    """
    subject = f'{label} {metric} vs configured {bar:g}'
    resolved, = K.resolve(run.available_keys(), [metric], ctx.route)
    if resolved.state is K.KeyState.NA_ROUTE:
        res.add(Finding(_R14_CHECK, subject, State.NA_ROUTE, resolved.note))
        return
    key = resolved.key
    got = series(run, key) if key else None
    if got is None:
        res.add(Finding(_R14_CHECK, subject, State.FLAG,
                        f'{_R14_TAG_ABSENT} -- the stage gates on this metric '
                        f'and the run does not log it: {resolved.note}'))
        return

    s, v = _r14_finite(trailing(*got, window))
    numbers = {'bar': float(bar), 'n_ticks': len(s),
               'window': _r14_window_desc(window)}
    if len(s) < _R14_MIN_TICKS:
        res.add(Finding(_R14_CHECK, subject, State.UNREADABLE,
                        f'{_R14_TAG_THIN} -- {len(s)} tick(s); '
                        f'{_R14_MIN_TICKS} needed before a noise floor is a '
                        f'measurement', numbers))
        return

    sigma, sigma_robust = _r14_sigma(s, v)
    numbers = dict(numbers, sigma=sigma, sigma_robust=sigma_robust,
                   metric_median=float(np.median(v)),
                   ratio=(float(bar) / sigma if sigma else float('inf')))
    ema = ' | EMA output (H5): its sigma is the filter\'s, not the sensor\'s' \
        if K.is_ema(key) else ''
    if sigma <= 0:
        # Same trap as the published-bar path: `bar < 0` is false for every bar,
        # so a pinned gated metric would silently read as a healthy bar.
        res.add(Finding(_R14_CHECK, subject, State.FLAG,
                        f'{_R14_TAG_NO_FLOOR} -- the gated metric has zero '
                        f'detrended spread over {len(s)} tick(s), so there is '
                        f'no measured floor to compare {bar:g} against{ema}',
                        numbers))
        return
    if abs(bar) < sigma:
        res.add(Finding(_R14_CHECK, subject, State.FLAG,
                        f'{_R14_TAG_BAR} -- the configured bar is inside the '
                        f'metric\'s own detrended scatter{ema}', numbers))
    else:
        res.add(Finding(_R14_CHECK, subject, State.OK, f'bar above sigma{ema}',
                        numbers))


def _r14_r13(res: CheckResult, run, window: Optional[float],
             bar_key: str, gated: list) -> None:
    if len(gated) != 1:
        subject = f'{bar_key} vs ?'
        if not gated:
            said = ('the metric it gates is not logged under any name, so the '
                    'bar cannot be compared with the floor it rides on')
        else:
            said = (f'{len(gated)} logged keys tag identically '
                    f'({", ".join(gated)}); naming which one the bar gates '
                    f'would be a guess')
        res.add(Finding(_R14_CHECK, subject, State.UNREADABLE,
                        f'{_R14_TAG_UNPAIRED} -- {said}'))
        return

    metric = gated[0]
    subject = f'{bar_key} vs {metric}'
    bar_got, met_got = series(run, bar_key), series(run, metric)
    if bar_got is None or met_got is None:
        res.add(Finding(_R14_CHECK, subject, State.UNREADABLE,
                        f'{_R14_TAG_UNPAIRED} -- no numeric series for '
                        f'{bar_key if bar_got is None else metric}'))
        return

    bs, bv = _r14_finite(trailing(*bar_got, window))
    ms, mv = _r14_finite(met_got)
    if len(bs):
        # The metric restricted to the span the bar COVERS. A bar that switched
        # on late compared against sigma from the whole run is compared against
        # a regime it was never in.
        inside = (ms >= bs[0]) & (ms <= bs[-1])
        ms, mv = ms[inside], mv[inside]

    numbers = {'n_bar': len(bs), 'n_metric_in_span': len(ms),
               'window': _r14_window_desc(window)}
    if len(bs) < _R14_MIN_TICKS or len(ms) < _R14_MIN_TICKS:
        res.add(Finding(_R14_CHECK, subject, State.UNREADABLE,
                        f'{_R14_TAG_THIN} -- {len(bs)} bar tick(s) and '
                        f'{len(ms)} metric tick(s) inside the bar\'s span; '
                        f'{_R14_MIN_TICKS} of each needed before a noise floor '
                        f'is a measurement', numbers))
        return

    sigma, sigma_robust = _r14_sigma(ms, mv)
    bar_last = float(bv[-1])
    numbers = dict(numbers, bar_last=bar_last, bar_min=float(np.min(bv)),
                   bar_median=float(np.median(bv)), sigma=sigma,
                   sigma_robust=sigma_robust,
                   metric_median=float(np.median(mv)))
    ema = ' | EMA output (H5): its sigma is the filter\'s, not the sensor\'s' \
        if K.is_ema(metric) else ''

    # SIGMA OF ZERO IS NOT A FLOOR OF ZERO. `bar < 0` is false for every bar, so
    # a gated metric that is itself pinned turned this row from FLAG to OK --
    # killing a controller input made the check QUIETER, which is the swallowed
    # diagnostic failing as reassurance rather than as silence. A dead gated
    # metric is the more serious finding of the two, and it is stated as one.
    if sigma <= 0:
        res.add(Finding(_R14_CHECK, subject, State.FLAG,
                        f'{_R14_TAG_NO_FLOOR} -- the metric this bar gates has '
                        f'ZERO detrended spread over the {len(ms)} tick(s) the '
                        f'bar covers, so there is no measured floor to compare '
                        f'{bar_last:.4g} against. R13 is unanswerable here and '
                        f'the sensor itself is the finding{ema}', numbers))
        return
    # Named, not judged: the std and the MAD scale disagreeing is a fact about
    # the metric's tails that the reader needs in order to weigh the row.
    split = ('' if (bar_last < sigma) == (bar_last < sigma_robust)
             else f' | the two sigmas straddle the bar -- std {sigma:.4g} is '
                  f'excursion-weighted, robust {sigma_robust:.4g} is not')

    if bar_last < sigma:
        res.add(Finding(_R14_CHECK, subject, State.FLAG,
                        f'{_R14_TAG_BAR} -- live bar {bar_last:.4g} is below '
                        f'the detrended sigma {sigma:.4g} of the metric it '
                        f'gates, over the {len(ms)} tick(s) the bar '
                        f'covers{split}{ema}', numbers))
    else:
        res.add(Finding(_R14_CHECK, subject, State.OK,
                        f'bar {bar_last:.4g} vs detrended sigma '
                        f'{sigma:.4g}{split}{ema}', numbers))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def check_r14(run, *, ctx: Optional[Context] = None,
              window: Optional[float] = None) -> CheckResult:
    """Dead-sensor conditions over every series a controller on `run` reads.

    One row per sensor and one per published bar. A run whose config names no
    sensor and whose protocol published no bar gets `not_run` naming what was
    looked for -- an empty table would render as "checked, all fine", which is
    the failure this package exists to prevent.
    """
    ctx = ctx or context(run)
    subjects = _r14_subjects(run, ctx)
    bars = _r14_bars(run)
    configured = _r14_configured_bars(run, ctx)

    if not subjects and not bars and not configured:
        stage = ('the stage could not be determined from the run\'s record, so '
                 'no stage-scoped sensor was read'
                 if ctx.stage_index is None
                 else f'stage {ctx.stage_index} ({ctx.stage_name}) names none')
        return CheckResult.not_run(
            _R14_CHECK,
            f'no controller input to examine: {stage}, no anchor-gate metric, '
            f'no censored estimator logged, and no {K.EXIT_THRESHOLD_TRACE % "*"} '
            f'bar published. Nothing about any sensor was asserted.')

    res = CheckResult(check=_R14_CHECK)
    for metric, roles in subjects.items():
        _r14_sensor(res, run, ctx, window, metric, roles)
    for bar_key, gated in bars:
        _r14_r13(res, run, window, bar_key, gated)
    for label, bar, metric in configured:
        _r14_r13_configured(res, run, ctx, window, label, bar, metric)
    return res



# ===========================================================================
# S4 -- the confounds named routinely
# ===========================================================================
# `run_all` calls this FIRST. A comparison across arms that are not comparable is
# not a weaker result, it is not a result, so the arm table has to be settled
# before any metric is shown rather than caveated afterwards.
#
# Everything here is read from `config` and the `phase` series, so it answers on a
# run that logged almost nothing -- which is exactly the run whose comparability
# is in doubt.
#
# WHAT IS DELIBERATELY NOT HERE. §4 names ten confounds; three of them do not
# belong to this check:
#
#   * 'a knob that was retired or inert in that tree' is R2's subject. Two checks
#     detecting one condition can disagree about it, and then the reader has to
#     adjudicate the tool.
#   * 'another process on the GPU' is NOT READABLE from wandb output. The run
#     record carries this process's utilisation, not the machine's tenancy, and a
#     proxy built from it would manufacture findings. It stays something the
#     reader has to know about the box.
#   * 'the LR sitting in a different part of its cycle at read time' needs the LR
#     series and a cycle model -- that is `features.py`'s oscillation extraction,
#     not a config assertion.
#
# A SINGLE RUN STILL GETS CHECKED. Its cross-arm subjects are skipped and the
# result SAYS so, as a row: one arm is a fact about the battery, not an absence of
# findings about it.

_CONF_CHECK = '§4 confounds'

# A read taken within this many steps of a stage ENTRY is a read of the
# injection point rather than of the stage: at a transition the optimiser state
# is fresh, the LR ramp (`adaptive_lr.warmup_steps`) is still running, and log Z
# is still relevelling. Under this many steps, whatever the metrics say is the
# transient. It is a constant and not a fraction of the run because the
# transient's length is set by the ramp, not by how long the run went on for.
_CONF_MIN_STAGE_STEPS = 1000.0

_CONF_IDENTITY = frozenset(K.CFG_IDENTITY)

# What decides whether two arms started from the same place. T is here as well
# as in the per-run subject: two arms can each be self-consistent (T == eval_T)
# and still be incomparable to each other.
_CONF_START_KEYS = (K.CFG_PRIOR_PATH, K.CFG_CONTINUE_FROM_CHECKPOINT,
                    K.CFG_SEED, K.CFG_ENERGY_FUNCTION,
                    K.CFG_TRAIN_T, K.CFG_EVAL_T)

# A key absent from the config and a key present holding null are DIFFERENT and
# are rendered differently: the first is a config from another tree (the knob
# takes its default), the second is a knob explicitly set to nothing.
_CONF_MISSING = '<missing>'
_CONF_NULL = '<null>'

# Knob names spelled out in the sweep row. The exact count is in `numbers`, so
# the cap shortens the line without hiding the size of the sweep.
_CONF_SWEEP_NAMES_SHOWN = 24


# ---------------------------------------------------------------------------
# Reading one config entry
# ---------------------------------------------------------------------------

def _conf_get(config: dict, key: str) -> tuple:
    """`(present, value)`. Both halves are needed: `K._value` answers None for a
    key that is absent and for a key holding null, and those are different
    findings -- the first says this config came from a different tree."""
    return key in config, K._value(config, key)


def _conf_show(present: bool, value) -> str:
    if not present:
        return _CONF_MISSING
    return _CONF_NULL if value is None else str(value)


def _conf_equal(a, b) -> bool:
    """Config values compared for the sweep table.

    NaN equals NaN here. A yaml `.nan` reaching the default comparison makes
    that knob differ between every pair of arms, which reads as a sweep
    dimension nobody swept."""
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


def _conf_label(run) -> str:
    return run_label(run)


def _conf_normalise(runs) -> list:
    """One Run, a list of them, or any iterable of them -> a list.

    A `config` of None is coerced to {} HERE rather than guarded at each use.
    Every helper below already wrote `run.config or {}`, five times over, but
    `check_confounds` reaches `context(run)` first and that iterates the config,
    so a None config crashed before any of the guards ran."""
    if runs is None:
        return []
    out = [runs] if hasattr(runs, 'config') else list(runs)
    for r in out:
        if getattr(r, 'config', None) is None:
            r.config = {}
    return out


def _conf_knobs(config: dict) -> set:
    """Config keys that CONFIGURE the run. Identity keys are dropped: every arm
    differs in its name, and a sweep table that lists the name as a swept knob
    is listing the thing the sweep is indexed BY."""
    return {k for k in config if k not in _CONF_IDENTITY}


# ---------------------------------------------------------------------------
# The stage series
# ---------------------------------------------------------------------------

def _conf_stage_series(run):
    """`(steps, values)` for the stage metric, or None.

    A one-point series is refused. `base.series` falls back to the SUMMARY,
    which hands back a single point at `last_step`; read as a residence that
    says the run entered its stage this instant, and the barely-started flag
    then fires on a hole in the data instead of on a short stage."""
    got = series(run, K.STAGE_METRIC)
    if got is None:
        return None
    s, v = got
    m = np.isfinite(s) & np.isfinite(v)
    s, v = s[m], v[m]
    return (s, v) if len(s) >= 2 else None


def _conf_boundaries(s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Steps at which the stage metric CHANGED. Any change counts, in either
    direction -- a rewind that puts a run back into an earlier stage is a
    boundary for the reader in exactly the same way an advance is."""
    idx = np.nonzero(np.diff(v) != 0)[0]
    return s[idx + 1]


# ---------------------------------------------------------------------------
# Per-run subjects
# ---------------------------------------------------------------------------

def _conf_per_run(res: CheckResult, run, ctx: Optional[Context],
                  window: Optional[float]) -> None:
    cfg = run.config or {}
    label = _conf_label(run)

    # --- the config itself. The per-subject rows below already say UNREADABLE
    # one at a time, but a reader meeting six of those has to infer the cause;
    # and in a BATTERY an empty config is worse than unreadable, because it
    # renders as `<missing>` on one side of every cross-arm comparison. Said
    # once, as its own subject, so the cause is stated rather than reconstructed.
    if not cfg:
        res.add(Finding(_CONF_CHECK, f'{label}/config', State.UNREADABLE,
                        'no config -- nothing below it is a comparison. `pull` '
                        'returns an empty config when files/config.yaml is '
                        'absent or unparseable, and it raises only on empty '
                        'HISTORY, so a run can arrive fully parsed and '
                        'unconfigured'))

    # --- T. 'different problem, different T (T dominates; keep eval_T = train
    # T)'. A config fact, so it is answerable on a run that logged nothing.
    p_t, t = _conf_get(cfg, K.CFG_TRAIN_T)
    p_e, e = _conf_get(cfg, K.CFG_EVAL_T)
    ft, fe = as_float(t), as_float(e)
    subj = f'{label}/T'
    if np.isnan(ft) or np.isnan(fe):
        res.add(Finding(_CONF_CHECK, subj, State.UNREADABLE,
                        f'{K.CFG_TRAIN_T}={_conf_show(p_t, t)}  '
                        f'{K.CFG_EVAL_T}={_conf_show(p_e, e)}'))
    else:
        nums = {K.CFG_TRAIN_T: ft, K.CFG_EVAL_T: fe}
        if ft != fe:
            res.add(Finding(_CONF_CHECK, subj, State.FLAG,
                            f'{K.CFG_EVAL_T} is not {K.CFG_TRAIN_T} -- the run '
                            f'is evaluated on a different integrator than the '
                            f'one it trains on', nums))
        else:
            res.add(Finding(_CONF_CHECK, subj, State.OK, '', nums))

    # --- code version. Absent is a FLAG in its own right: without the stamp,
    # drift against a sibling cannot be ruled out, and §4 opens with drift.
    commit = K.git_commit(cfg)
    subj = f'{label}/code_version'
    if commit:
        res.add(Finding(_CONF_CHECK, subj, State.OK, commit))
    else:
        res.add(Finding(_CONF_CHECK, subj, State.FLAG,
                        'no commit stamp in the config -- version drift against '
                        'another arm cannot be ruled out'))

    # --- start condition, REPORTED not judged. Resuming is normal here; what
    # makes it a confound is a sibling that started somewhere else, and that
    # comparison is a battery subject.
    p_c, cont = _conf_get(cfg, K.CFG_CONTINUE_FROM_CHECKPOINT)
    p_n, ckpt = _conf_get(cfg, K.CFG_CHECKPOINT_NAME)
    subj = f'{label}/start_condition'
    if not p_c and not p_n:
        res.add(Finding(_CONF_CHECK, subj, State.UNREADABLE,
                        'neither start-condition key is in the config'))
    else:
        res.add(Finding(_CONF_CHECK, subj, State.OK,
                        f'{K.CFG_CONTINUE_FROM_CHECKPOINT}='
                        f'{_conf_show(p_c, cont)}  '
                        f'{K.CFG_CHECKPOINT_NAME}={_conf_show(p_n, ckpt)}'))

    _conf_stage_subjects(res, run, ctx, window, label)


def _conf_stage_subjects(res: CheckResult, run, ctx: Optional[Context],
                         window: Optional[float], label: str) -> None:
    subj_r, subj_b = f'{label}/stage_residence', f'{label}/stage_boundary'
    got = _conf_stage_series(run)
    if got is None:
        reason = (f'{K.STAGE_METRIC} has fewer than two logged points -- a '
                  f'stage entry cannot be located')
        res.add(Finding(_CONF_CHECK, subj_r, State.UNREADABLE, reason))
        res.add(Finding(_CONF_CHECK, subj_b, State.UNREADABLE, reason))
        return

    s, v = got
    bounds = _conf_boundaries(s, v)
    now = max(as_float(getattr(run, 'last_step', 0.0), 0.0), float(s[-1]))
    stage = (ctx.stage_name if ctx is not None and ctx.stage_name
             else f'{K.STAGE_METRIC}={int(v[-1])}')

    # No boundary in the history is NOT 'the run never transitioned'. Runs here
    # restart from a checkpoint with the step counter carried over, so the
    # history can begin mid-stage; the residence is then bounded below by the
    # span, and that span is also all there is to read.
    exact = bool(len(bounds))
    entered = float(bounds[-1]) if exact else float(s[0])
    resid = now - entered
    nums = {'steps_in_stage': resid, 'entered_at': entered, 'last_step': now,
            'n_boundaries': len(bounds)}
    if exact:
        where = f'{stage}, entered at {entered:.0f}'
        # The stage entry IS in the history, so the residence is the residence
        # and a short one means the metrics are still the transition's.
        flag = (f'{where}; under {_CONF_MIN_STAGE_STEPS:.0f} steps in the '
                f'stage, so the read is of the injection point')
    else:
        where = (f'{stage}, no stage change in the history -- a LOWER BOUND on '
                 f'residence, and the whole readable span')
        # A different sentence, because the residence is NOT known here: the
        # run may have been in this stage for a hundred thousand steps and be
        # readable for four hundred of them. Claiming an injection point would
        # be asserting something the data does not say.
        flag = (f'{where}; under {_CONF_MIN_STAGE_STEPS:.0f} steps of history '
                f'in this stage, whatever the true residence is')
    res.add(Finding(_CONF_CHECK, subj_r,
                    State.FLAG if resid < _CONF_MIN_STAGE_STEPS else State.OK,
                    flag if resid < _CONF_MIN_STAGE_STEPS else where, nums))

    last_b = float(bounds[-1]) if exact else float('nan')
    if window is None:
        # Not a flag. With no window the read is the whole history, so every
        # boundary is inside it by construction and flagging that would fire on
        # every multi-stage run while saying nothing about the read.
        res.add(Finding(_CONF_CHECK, subj_b, State.OK,
                        'no window given -- the read spans the whole history '
                        'and every stage boundary in it',
                        {'n_boundaries': len(bounds), 'last_boundary': last_b}))
        return
    inside = bounds[bounds > now - float(window)]
    if len(inside):
        res.add(Finding(_CONF_CHECK, subj_b, State.FLAG,
                        'the trailing window straddles a stage boundary -- '
                        'features over it mix two stages plus the transition '
                        'transient (fresh optimiser, LR ramp)',
                        {'n_in_window': len(inside),
                         'nearest': float(inside[-1]),
                         'window': float(window), 'last_step': now}))
    else:
        res.add(Finding(_CONF_CHECK, subj_b, State.OK, '',
                        {'n_in_window': 0, 'last_boundary': last_b,
                         'window': float(window)}))


# ---------------------------------------------------------------------------
# Battery subjects
# ---------------------------------------------------------------------------

def _conf_group(runs, fn) -> dict:
    """Arms grouped by a hashable reading of their config, insertion-ordered."""
    out = {}
    for run in runs:
        out.setdefault(fn(run), []).append(_conf_label(run))
    return out


def _conf_battery(res: CheckResult, runs: list) -> None:
    _conf_battery_position(res, runs)
    _conf_battery_commit(res, runs)
    _conf_battery_checkpoint(res, runs)
    _conf_battery_start(res, runs)
    _conf_battery_duplicates(res, runs)
    _conf_battery_sweep(res, runs)


def _conf_battery_position(res: CheckResult, runs: list) -> None:
    """Which stage each arm is in, and which route that puts it on.

    "These arms are in different stages" is the strongest not-comparable
    statement §4 has, and it was the one the check could not make: a battery of
    a run in `train_prior` and a run in `equilibration` produced findings about
    commits and checkpoints and nothing about the fact that the two were being
    asked different questions. The route is reported beside it because it is
    what decides, per arm, which metrics are withheld as NA_ROUTE -- two arms on
    different routes do not have a comparable topline at all.
    """
    for what, read in (('stage', lambda c: c.stage_name or 'UNKNOWN'),
                       ('route', lambda c: c.route.value)):
        groups = _conf_group(runs, lambda r, f=read: f(context(r)))
        detail = ' | '.join(f'{val}: {", ".join(a)}' for val, a in groups.items())
        nums = {'n_values': len(groups), 'n_arms': len(runs)}
        if len(groups) > 1:
            res.add(Finding(_CONF_CHECK, f'battery/{what}', State.FLAG,
                            f'arms are not in the same {what} -- {detail}', nums))
        else:
            res.add(Finding(_CONF_CHECK, f'battery/{what}', State.OK,
                            detail, nums))


def _conf_battery_commit(res: CheckResult, runs: list) -> None:
    groups = _conf_group(runs, lambda r: K.git_commit(r.config or {}))
    detail = ' | '.join(f'{c or "no stamp"}: {", ".join(a)}'
                        for c, a in groups.items())
    nums = {'n_commits': len(groups), 'n_arms': len(runs)}
    if len(groups) > 1:
        res.add(Finding(_CONF_CHECK, 'battery/code_version', State.FLAG,
                        f'arms are on different code -- {detail}', nums))
    else:
        res.add(Finding(_CONF_CHECK, 'battery/code_version', State.OK,
                        detail, nums))


def _conf_battery_checkpoint(res: CheckResult, runs: list) -> None:
    """`checkpoint_name` mixed null / non-null across a battery.

    The worked case: three arms running with `checkpoint_name: None` beside nine
    that carried an explicit checkpoint are two batches, not one battery of
    twelve -- the second nine started from a trained model and the three did
    not, so every metric between them is offset by that."""
    missing, null, named = [], [], []
    for run in runs:
        present, value = _conf_get(run.config or {}, K.CFG_CHECKPOINT_NAME)
        (missing if not present else null if value is None else named).append(
            _conf_label(run))
    nums = {'n_named': len(named), 'n_null': len(null),
            'n_missing': len(missing), 'n_arms': len(runs)}
    parts = [f'named: {", ".join(named)}' if named else '',
             f'null: {", ".join(null)}' if null else '',
             f'key missing: {", ".join(missing)}' if missing else '']
    detail = '  |  '.join(p for p in parts if p)
    if named and (null or missing):
        res.add(Finding(_CONF_CHECK, 'battery/checkpoint_name', State.FLAG,
                        f'arms started from different things -- {detail}', nums))
    else:
        res.add(Finding(_CONF_CHECK, 'battery/checkpoint_name', State.OK,
                        detail, nums))

    # THE VALUES, not only their null-ness. §4's second-named confound is
    # "checkpoint chaining, where arms silently resume FROM EACH OTHER rather
    # than a pinned start", and null-ness cannot see it: a battery in which every
    # arm carries a checkpoint passes the test above while one arm resumed from a
    # phase-1 exit and the rest resumed from that arm's own rolling checkpoint.
    # Measured on a real 16-arm battery, which is exactly that shape.
    if len(runs) > 1:
        groups = _conf_group(
            runs, lambda r: _conf_show(*_conf_get(r.config or {},
                                                  K.CFG_CHECKPOINT_NAME)))
        detail = ' | '.join(f'{val}: {", ".join(a)}' for val, a in groups.items())
        nums = {'n_values': len(groups), 'n_arms': len(runs)}
        state = State.FLAG if len(groups) > 1 else State.OK
        res.add(Finding(_CONF_CHECK, 'battery/checkpoint_source', state,
                        detail if state is State.OK else
                        f'arms resumed from different checkpoints -- {detail}',
                        nums))


def _conf_battery_start(res: CheckResult, runs: list) -> None:
    for key in _CONF_START_KEYS:
        groups = _conf_group(
            runs, lambda r, k=key: _conf_show(*_conf_get(r.config or {}, k)))
        detail = ' | '.join(f'{val}: {", ".join(a)}' for val, a in groups.items())
        nums = {'n_values': len(groups), 'n_arms': len(runs)}
        state = State.FLAG if len(groups) > 1 else State.OK
        res.add(Finding(_CONF_CHECK, f'battery/start/{key}', state,
                        detail, nums))


def _conf_battery_duplicates(res: CheckResult, runs: list) -> None:
    """Arms whose SHARED knobs all agree are the same arm written twice.

    An absent knob takes its default, so a pair that differs only in which keys
    are PRESENT is a pair of duplicates -- the sweep dimension the author
    thought they were varying is not in the config at all. The stricter case,
    where the two configs are equal outright, is the same finding and is named
    as such rather than being folded in silently."""
    dup = 0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            ca, cb = a.config or {}, b.config or {}
            ka, kb = _conf_knobs(ca), _conf_knobs(cb)
            shared = ka & kb
            # NO SHARED KNOBS MEANS NOTHING WAS COMPARED. `differ` is empty
            # either because every shared knob agreed or because there were
            # none, and those render as the same confident FLAG. `pull` returns
            # a run with `config == {}` whenever `files/config.yaml` is absent
            # or unparseable -- true of 96 of the 182 local run directories --
            # so the empty-vs-real pair was reported as "the same arm written
            # twice" on the strength of zero evidence.
            if not shared:
                continue
            differ = [k for k in shared
                      if not _conf_equal(K._value(ca, k), K._value(cb, k))]
            if differ:
                continue
            dup += 1
            only_a, only_b = sorted(ka - kb), sorted(kb - ka)
            omitted = only_a + only_b
            nums = {'n_shared': len(ka & kb), 'n_differing': 0,
                    'n_present_only_in_one': len(omitted)}
            if omitted:
                detail = ('same arm written two ways -- every shared knob '
                          'agrees and the rest are absent, taking their '
                          f'defaults: {", ".join(omitted[:_CONF_SWEEP_NAMES_SHOWN])}')
            else:
                detail = ('identical configs -- not one knob differs outside '
                          'the identity keys')
            res.add(Finding(_CONF_CHECK,
                            f'battery/duplicate/{_conf_label(a)}~{_conf_label(b)}',
                            State.FLAG, detail, nums))
    n_pairs = len(runs) * (len(runs) - 1) // 2
    res.add(Finding(_CONF_CHECK, 'battery/duplicates', State.OK,
                    f'{dup} duplicate pair(s) of {n_pairs} compared',
                    {'n_pairs': n_pairs, 'n_duplicate': dup}))


def _conf_battery_sweep(res: CheckResult, runs: list) -> None:
    """The sweep table: which knobs actually differ across the battery.

    Not a finding -- it is the table the reader needs in order to say which arm
    is which. A knob differing by PRESENCE counts: absent means the default, and
    a default that differs from a sibling's explicit value is a swept knob
    whether or not anyone meant to sweep it."""
    keys = set()
    for run in runs:
        keys |= _conf_knobs(run.config or {})
    by_presence, by_value = [], []
    for key in sorted(keys):
        seen = [_conf_get(run.config or {}, key) for run in runs]
        p0, v0 = seen[0]
        if any(p != p0 for p, _ in seen[1:]):
            by_presence.append(key)
        elif any(not _conf_equal(v, v0) for _, v in seen[1:]):
            by_value.append(key)
    names = by_value + by_presence
    shown = ', '.join(names[:_CONF_SWEEP_NAMES_SHOWN])
    more = len(names) - _CONF_SWEEP_NAMES_SHOWN
    res.add(Finding(_CONF_CHECK, 'battery/sweep', State.OK,
                    (shown + (f'  (+{more} more)' if more > 0 else ''))
                    or 'no knob differs across the battery',
                    {'n_knobs': len(names), 'by_value': len(by_value),
                     'by_presence': len(by_presence), 'n_arms': len(runs)}))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def check_confounds(runs, *, ctx: Optional[Context] = None,
                    window: Optional[float] = None) -> CheckResult:
    """§4's confounds, over one run or a battery.

    Battery subjects run first: 'these arms are not comparable' outranks
    anything true of one of them. A single run gets the per-run subjects and a
    row saying the cross-arm ones were skipped -- returning `not_run` there
    would throw away the T, code-version, start-condition and stage-residence
    answers, which are properties of the run and not of the battery.
    """
    runs = _conf_normalise(runs)
    if not runs:
        return CheckResult.not_run(
            _CONF_CHECK, 'no runs given -- nothing to read or to compare')
    res = CheckResult(check=_CONF_CHECK)
    if len(runs) > 1:
        _conf_battery(res, runs)
    else:
        res.add(Finding(_CONF_CHECK, 'battery', State.OK,
                        'cross-arm subjects skipped -- one arm, and a confound '
                        'between arms needs a sibling to be a confound of',
                        {'n_arms': 1}))
    for run in runs:
        rctx = ctx if (ctx is not None and len(runs) == 1) else context(run)
        _conf_per_run(res, run, rctx, window)
    return res



# ===========================================================================
# R11 -- replay memorisation: TWO sensors, and only one of them has a bar
# ===========================================================================
# `reading_runs.md` R11 names one statistic. `module_modulators.md` §3 names two,
# and says which of them to steer on. Both are reported here; exactly one flags.
#
# SENSOR B -- `K.R11_MEMORISATION_*`, each resident row's current residual
# against the one it was ADMITTED with. THE FLAGGING SUBJECT, because its bar is
# DERIVED and not calibrated: under exponential relaxation at rate lambda and
# exponential residence with mean tau the ratio is ~exp(-lambda*tau), so the
# lambda*tau = 1 boundary -- rows corrected exactly as fast as they are replaced
# -- lands at 1/e for every problem, T and buffer size (`module_buffers.md` B8).
# The span up to `K.R11_MEMORISATION_RELEASE` is the buffer servo's HOLD band:
# `protocol.py` `_buffer_servo_tick` tightens below the bar, releases above the
# release, and holds between. All three bands are named in the row.
#
# SENSOR A -- `K.R11_NUMERATOR / K.R11_DENOMINATOR`. REPORTED, NEVER FLAGGED, and
# its ambiguity is named IN the row: `module_metrics.md` records that a ratio
# below 1 is equally the signature of memorisation and of a coverage gap, that
# the statistic does not distinguish them, and that reading it as either one alone
# is unwarranted. `module_modulators.md` D1 adds that its stated justification is
# stale and its thresholds are uncalibrated. Measured over the local corpus,
# flagging it fired on roughly three quarters of the readable TB-route runs and
# nearly all of them sat under the `K.R11_SCATTER_REFERENCE` figure -- a state
# that fires on three quarters of a corpus is not a finding, and a check that
# cries wolf gets switched off. It is NOT deleted: it is the sensor the build spec
# named and several configs still steer on it, so a reader who comes looking for
# it must find it with its status attached. `K.R11_SCATTER_REFERENCE` is printed
# beside the number as a stated REFERENCE and is never compared against.
#
# AN ABSENT SENSOR B IS NOT A CLEAN SENSOR B. `buffer.absorption_stats` publishes
# NOTHING below its minimum valid-row count, and the whole family was only wired
# into the metric tracker on 2026-08-07 (`module_modulators.md` D7), so a quarter
# of the local TB corpus logs sensor A and not sensor B. That row is UNREADABLE
# and says which series it wanted -- reporting sensor A alone in its place would
# hand the reader the ambiguous number as though it were the answer.
#
# THE ROUTE GATE IS A PROPERTY OF THE ROUTE, NOT OF THE KEYS, and is unchanged.
# R11 is defined on `K.R11_ROUTES`; on the conditional VarGrad route the answer is
# NA_ROUTE -- the check RAN and its subject is not meaningful there. `keys.resolve`
# does NOT mark any of these four series as NA on any route (its NA patterns cover
# log Z and the TB residuals), so asking the key would return LIVE and hand back a
# number to be read as if it were on a TB run. The gate is therefore tested BEFORE
# presence: presence-first would report a VarGrad run that never ran a replay
# branch as `not_run`, which says "the data is missing" where what is true is "the
# question does not apply".
#
# THE RATIO IS POINTWISE, on both sensors. A ratio of medians is not the median of
# ratios, and one excursion in either series moves the former. Measured on a real
# five-stage run the two disagree by enough to move the answer across a band.
#
# WHAT IS DELIBERATELY NOT HERE. `replay/absorbed_frac` is `1 - ratio` by
# construction (`buffer.absorption_stats`) -- measured against sensor B over every
# local run that logs both, the two agree to floating-point exactness, so a row
# for it would double-count sensor B under a second name. `replay/absorption_n` is
# the sensor's valid-row count; the only reading the docs ground for it is the
# abstention floor, which is stated in the UNREADABLE row above and needs no
# threshold. Neither is given a bar, because the repo derives none.

_R11_CHECK = 'R11 replay memorisation'

# Below this many aligned ticks the "median" is whichever tick the eval cadence
# happened to land on rather than a level, so the sensor reports no number rather
# than that one. Also the guard that stops `base.series`' summary fallback -- a
# single point carried from the summary -- being reported as a ratio.
_R11_MIN_ALIGNED = 8

# First token of `detail`, so which sensor a row is and what standing it has are
# legible in the rendered table and assertable in a test without parsing prose.
_R11_TAG_BAR = 'DERIVED BAR'
_R11_TAG_REFERENCE = 'REFERENCE ONLY'


def _r11_window_desc(window: Optional[float]) -> str:
    return 'all' if window is None else f'trailing {window:g} steps'


def _r11_align(num, den):
    """`(steps, numerator, denominator, ok_mask)` on one step grid.

    THE MASK IS RETURNED RATHER THAN A COUNT so the caller can count drops
    INSIDE its window. Counted here, `n_dropped` spanned the whole history while
    the median and `n_aligned` beside it were windowed -- so a windowed row
    reported drops for ticks it had not read, always over-reporting, and a clean
    window could be made to look a third garbage.

    The denominator is interpolated onto the NUMERATOR's steps, restricted to the
    span the denominator actually covers: `np.interp` clamps outside its range
    without complaint, which would invent denominator values for steps it never
    saw and pair them with real numerator values.

    Ticks where either value is non-finite, or the denominator is not positive,
    are dropped and counted -- a zero denominator makes an infinite ratio, and
    infinities in a median are how one bad tick decides a band.
    """
    ns, nv = np.asarray(num[0], float), np.asarray(num[1], float)
    ds, dv = np.asarray(den[0], float), np.asarray(den[1], float)
    empty = np.zeros(0, float)
    if not len(ns) or not len(ds):
        return empty, empty, empty, np.zeros(0, bool)
    if np.any(np.diff(ns) < 0):
        # The NUMERATOR needs the same insurance the denominator gets below.
        # `base.trailing` windows on `s[-1]`, so an out-of-order last row makes
        # the window anchor on a step that is not the newest -- the row then
        # reports a window it did not apply.
        order = np.argsort(ns, kind='stable')
        ns, nv = ns[order], nv[order]
    if np.any(np.diff(ds) <= 0):
        # np.interp requires an increasing xp and returns garbage silently
        # otherwise. Cheap insurance against a merged or resumed history.
        order = np.argsort(ds, kind='stable')
        ds, dv = ds[order], dv[order]

    inside = (ns >= ds[0]) & (ns <= ds[-1])
    s, a = ns[inside], nv[inside]
    b = np.interp(s, ds, dv)
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    return s, a, b, ok


def _r11_read(run, num_key: str, den_key: str,
              window: Optional[float]) -> tuple:
    """`(numbers, reason)` for one sensor. `numbers` is None when no median
    could be formed, and `reason` then says why, naming the series it wanted.

    Never raises and never substitutes: a sensor that cannot be read hands back
    the reason so the caller can put it in a row, because "no number" and "a
    number near 1" are opposite readings and the second must never stand in for
    the first.
    """
    num, den = series(run, num_key), series(run, den_key)
    missing = [k for k, sv in ((num_key, num), (den_key, den)) if sv is None]
    if missing:
        return None, f'{" and ".join(missing)} not logged by this run'

    s_all, a_all, b_all, ok = _r11_align(num, den)
    # WINDOW FIRST, THEN DROP. Both counts then describe the same ticks.
    if len(s_all):
        keep, _ = trailing(s_all, s_all, window)
        in_window = (s_all >= keep[0]) if len(keep) else np.zeros(len(s_all), bool)
    else:
        in_window = np.zeros(0, bool)
    n_dropped = int((in_window & ~ok).sum())
    sel = in_window & ok
    ws = s_all[sel]
    w_num, w_den = a_all[sel], b_all[sel]
    with np.errstate(divide='ignore', invalid='ignore'):
        w_ratio = w_num / w_den
    if len(ws) < _R11_MIN_ALIGNED:
        return None, (f'{len(ws)} aligned point(s) over '
                      f'{_r11_window_desc(window)}; {_R11_MIN_ALIGNED} needed '
                      f'for a median ({num_key}: {len(num[0])} point(s), '
                      f'{den_key}: {len(den[0])} point(s), '
                      f'{int((in_window & ok).sum())} usable in the window, '
                      f'{n_dropped} dropped)')

    numbers = dict(median_ratio=float(np.median(w_ratio)),
                   num_median=float(np.median(w_num)),
                   den_median=float(np.median(w_den)),
                   n_aligned=len(ws),
                   window=_r11_window_desc(window))
    if n_dropped:
        numbers['n_dropped'] = n_dropped
    return numbers, ''


def _r11_servo(run, ctx: Context) -> dict:
    """What the run's OWN stage declares for its buffer servo, if anything.

    The derived 1/e bar is a property of the QUANTITY -- lambda*tau = 1 holds
    wherever the ratio is computed -- so it stays the flag. The release is not
    derived; it comes from the config generators. Naming [bar, release) "the
    buffer servo's hold band" without reading this asserted a controller most
    runs do not declare, and misnamed the band on the runs that declare one
    steering a different pair entirely."""
    idx = ctx.stage_index
    if idx is None:
        return {}
    cfg = run.config or {}
    out = {}
    for name, tmpl in (('bar', K.CFG_STAGE_BUFFER_SERVO_BAR),
                       ('release', K.CFG_STAGE_BUFFER_SERVO_RELEASE),
                       ('numerator', K.CFG_STAGE_BUFFER_SERVO_NUM),
                       ('denominator', K.CFG_STAGE_BUFFER_SERVO_DEN)):
        v = K._value(cfg, tmpl % idx)
        if v is not None:
            out[name] = v
    return out


def _r11_row_b(res: CheckResult, subject: str, numbers: Optional[dict],
               reason: str, servo: Optional[dict] = None,
               logged: bool = False) -> None:
    """The flagging row. Below the DERIVED bar is the finding; everything above
    it is named against whatever the run itself declares, or against nothing."""
    servo = servo or {}
    if numbers is None:
        # UNREADABLE, and the reason must not be overwritten by a story. An
        # absent sensor is the sensor abstaining or predating its plumbing; a
        # sensor that IS logged and whose every tick is unusable is a dead
        # sensor, which is the opposite reading. Asserting the benign one on
        # both is a swallowed diagnostic failing as reassurance.
        gloss = ('' if logged else
                 '. Absence here is the sensor abstaining or predating its '
                 'plumbing, not a reading about the buffer')
        res.add(Finding(_R11_CHECK, subject, State.UNREADABLE,
                        f'{_R11_TAG_BAR} -- the sensor with the derived bar '
                        f'could not be read: {reason}{gloss}'))
        return

    med = numbers['median_ratio']
    bar = K.R11_MEMORISATION_BAR
    numbers = dict(numbers, bar=float(bar))
    # The run's own release if it declares one, else the generators' figure,
    # labelled as what it is.
    own = as_float(servo.get('release'), float('nan'))
    if np.isfinite(own):
        release, rel_src = own, 'this stage\'s buffer_servo release'
        numbers['release'] = float(release)
        numbers['release_source'] = 'run'
    else:
        release, rel_src = K.R11_MEMORISATION_RELEASE, (
            'the release the config generators use; THIS RUN DECLARES NO '
            'buffer_servo on this stage')
        numbers['release'] = float(release)
        numbers['release_source'] = 'generator_default'
    pair = ''
    if servo.get('numerator') and servo['numerator'] != K.R11_MEMORISATION_NUMERATOR:
        pair = (f' | note: this stage\'s servo steers '
                f'{servo["numerator"]}/{servo.get("denominator")}, not this pair')

    if med < bar:
        state = State.FLAG
        where = (f'below the derived {bar:g} bar -- lambda*tau above 1, i.e. '
                 f'resident rows corrected faster than they are replaced')
    elif med < release:
        state = State.OK
        where = f'between the derived {bar:g} bar and {release:g} ({rel_src})'
    else:
        state = State.OK
        where = f'at or above {release:g} ({rel_src})'
    res.add(Finding(_R11_CHECK, subject, state,
                    f'{_R11_TAG_BAR} -- {med:.3g}, {where}{pair}', numbers))


def _r11_row_a(res: CheckResult, subject: str, numbers: Optional[dict],
               reason: str) -> None:
    """The reference row. IT NEVER FLAGS, on any value.

    Not even when it cannot be read: the reader's action on a below-1 scatter
    ratio is to go and read sensor B, which is already its own row, and a finding
    state here would put the ambiguous number back in the position of an answer.
    """
    if numbers is None:
        res.add(Finding(_R11_CHECK, subject, State.OK,
                        f'{_R11_TAG_REFERENCE} -- not computed: {reason}. '
                        f'Carried so a reader who came looking for this sensor '
                        f'finds its status rather than a hole'))
        return

    med = numbers['median_ratio']
    below, ref = K.R11_SCATTER_BELOW, K.R11_SCATTER_REFERENCE
    numbers = dict(numbers, below_ratio=float(below), ref_ratio=float(ref))
    side = 'below' if med < below else 'at or above'
    res.add(Finding(
        _R11_CHECK, subject, State.OK,
        f'{_R11_TAG_REFERENCE} -- {med:.3g}x, {side} {below:g}x, against a '
        f'stated reference of {ref:g}x (a reference, not a bar). Below {below:g}x '
        f'is equally the signature of memorisation and of a coverage gap and '
        f'this statistic does not distinguish them, so it is reported and not '
        f'flagged; the row above carries the bar', numbers))


def check_r11(run, *, ctx: Optional[Context] = None,
              window: Optional[float] = None) -> CheckResult:
    """Both replay-memorisation sensors, pointwise, median over the window.

    Two rows: the derived-bar sensor, which flags, and the scatter ratio, which
    is reported and never flags. Every path that produces neither row either says
    NA_ROUTE (the question does not apply here) or `not_run` naming which series
    were missing -- never an empty result, which renders identically to "checked,
    nothing wrong".
    """
    ctx = ctx or context(run)
    res = CheckResult(check=_R11_CHECK)
    routes = '/'.join(r.value for r in K.R11_ROUTES)
    subj_b = f'{K.R11_MEMORISATION_NUMERATOR} / {K.R11_MEMORISATION_DENOMINATOR}'
    subj_a = f'{K.R11_NUMERATOR} / {K.R11_DENOMINATOR}'

    if ctx.route is K.Route.UNKNOWN:
        # NOT NA_ROUTE. NA_ROUTE asserts the route is known and R11 does not
        # apply on it; an unclassified route leaves applicability UNDETERMINED,
        # and that is a hole in the report, which renders loudly, rather than a
        # table row, which does not.
        return CheckResult.not_run(
            _R11_CHECK,
            f'route not classified from the config, so it is unknown whether '
            f'R11 applies (it is defined on {routes}). Neither sensor was '
            f'computed and nothing about replay was asserted.')

    if ctx.route not in K.R11_ROUTES:
        # BOTH sensors get a row. One row for two withheld subjects would leave
        # the reader unable to tell which of them was withheld.
        for subj in (subj_b, subj_a):
            res.add(Finding(_R11_CHECK, subj, State.NA_ROUTE,
                            f'withheld -- R11 is defined on {routes}; this run '
                            f'is on {ctx.route.value}'))
        return res

    b_numbers, b_reason = _r11_read(run, K.R11_MEMORISATION_NUMERATOR,
                                    K.R11_MEMORISATION_DENOMINATOR, window)
    a_numbers, a_reason = _r11_read(run, K.R11_NUMERATOR, K.R11_DENOMINATOR,
                                    window)

    if b_numbers is None and a_numbers is None:
        # Only when NEITHER sensor produced a number. Refusing because the
        # reference sensor is missing would throw away the one row that carries
        # a bar, which is the whole point of the rework.
        return CheckResult.not_run(
            _R11_CHECK,
            f'neither sensor could be read on a {ctx.route.value} run. '
            f'Derived-bar sensor: {b_reason}. Reference sensor: {a_reason}. '
            f'Nothing about replay was asserted -- this is not "replay is fine".')

    # `logged` separates "the sensor is not there" from "the sensor is there and
    # every tick of it is unusable". The second is a dead sensor and must not be
    # told the benign story.
    b_logged = (series(run, K.R11_MEMORISATION_NUMERATOR) is not None
                and series(run, K.R11_MEMORISATION_DENOMINATOR) is not None)
    _r11_row_b(res, subj_b, b_numbers, b_reason,
               servo=_r11_servo(run, ctx), logged=b_logged)
    _r11_row_a(res, subj_a, a_numbers, a_reason)
    return res



# ---------------------------------------------------------------------------
# Running them all
# ---------------------------------------------------------------------------

def run_all(runs, *, window: Optional[float] = None) -> list:
    """Every check, over one run or a battery of them.

    Section 4 is the only check that needs more than one run, and it is the only
    one that must run FIRST: a comparison across arms that are not comparable is
    not a weaker result, it is not a result.
    """
    runs = [runs] if not isinstance(runs, (list, tuple)) else list(runs)

    # `window` reaches section 4 too. Its stage-boundary subject asks whether the
    # read window straddles a transition, which is unanswerable without one --
    # left unforwarded, that subject could never fire on the default path, and
    # its rows could not be read against the windowed rows printed beside them.
    battery = check_confounds(runs, window=window)
    battery.run = ', '.join(run_label(r) for r in runs)
    battery.header = f'{len(runs)} arm(s)'
    out = [battery]

    for run in runs:
        ctx = context(run)
        header = context_header(run, ctx, window)
        for check in (check_r2, check_r14, check_r11):
            res = check(run, ctx=ctx, window=window)
            # Stamped HERE and not inside each check: a block that cannot be
            # attributed to an arm is unreadable in a battery, and leaving it to
            # four separate call sites is four chances to forget.
            res.run, res.header = run_label(run), header
            out.append(res)
    return out


def format_report(results: Iterable[CheckResult], *, verbose: bool = False) -> str:
    """The checks, rendered, with the ones that DID NOT RUN first.

    Order is deliberate. A check that did not run is a hole in the report, and a
    hole placed after the findings reads as a footnote to a complete picture."""
    results = list(results)
    parts = [format_result(r, verbose=verbose) for r in results if not r.ran]
    parts += [format_result(r, verbose=verbose) for r in results if r.ran]
    return '\n'.join(parts)
