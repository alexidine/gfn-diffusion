"""Shared types and series helpers for the checks.

WHAT A CHECK IS HERE. `reading_runs.md` §3 splits its principles into mechanical
and judgment. A mechanical one becomes a check; everything else surfaces its
inputs and stops. So a check reports STATE and NUMBERS and never a conclusion:
'fired, 1281 of 1316 ticks' is a check, 'the controller is working' is not, and
a tool that concludes 'the run is healthy' has failed its spec.

TWO FAILURE MODES A CHECK MUST NOT HAVE.

  * Silent pass. A check that could not run because its inputs were missing must
    say so -- `ran=False` with a reason -- and never return an empty finding list
    that reads identically to 'looked, found nothing wrong'. Swallowed
    diagnostics do not fail as silence; they fail as REASSURANCE.
  * Collapsing NA_ROUTE. On the conditional VarGrad route the log Z and TB series
    exist and carry numbers that must not be read as they would be on a TB run.
    A check whose subject is NA on the route reports NA_ROUTE. Never ABSENT,
    never zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

import numpy as np

from .. import keys as K


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

    @property
    def findings(self) -> list:
        return [r for r in self.rows if r.is_finding]

    def add(self, finding: Finding) -> None:
        self.rows.append(finding)

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
    idx = K.current_stage_index(run.summary, run.config)
    return Context(stage_index=idx,
                   stage_name=K.current_stage(run.summary, run.config),
                   route=K.detect_route(run.config, idx),
                   stages=tuple(K.stage_names(run.config)))


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
    if not res.ran:
        return (f'\n{res.check}  DID NOT RUN\n'
                f'  {res.reason}\n'
                f'  (this is not a pass -- nothing was asserted)')
    shown = res.rows if verbose else res.findings
    head = (f'\n{res.check}  {len(res.findings)} finding(s) '
            f'of {len(res.rows)} examined')
    if not shown:
        return head + '\n  (nothing to report; run with -v for the full table)'
    return head + '\n' + '\n'.join(str(r) for r in shown)
