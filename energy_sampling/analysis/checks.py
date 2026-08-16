"""Assertions over a run: R2 liveness, R14 dead sensors, §4 confounds, R11.

`reading_runs.md` §3's principles split into mechanical and judgment. The
mechanical ones are here as checks. The rest surface their inputs in
`features.py` and stop -- this module emits no verdicts, and a report that
concludes 'the run is healthy' has failed its spec.

Each check returns a `CheckResult` carrying every subject it examined, with the
numbers behind each. A check that could not run says so; it never returns an
empty result that reads like a pass.
"""

from __future__ import annotations

from typing import Iterable, Optional

from . import keys as K
from ._parts.base import (CheckResult, Context, Finding, State, context,
                          format_result)
from ._parts.confounds import check_confounds
from ._parts.r2 import check_r2
from ._parts.r11 import check_r11
from ._parts.r14 import check_r14

__all__ = ['CheckResult', 'Context', 'Finding', 'State', 'context',
           'format_result', 'check_r2', 'check_r14', 'check_r11',
           'check_confounds', 'run_all', 'format_report']


def run_all(runs, *, window: Optional[float] = None) -> list:
    """Every check, over one run or a battery of them.

    §4 is the only check that needs more than one run, and it is the only one
    that must run FIRST: a comparison across arms that are not comparable is not
    a weaker result, it is not a result. The per-run checks follow.
    """
    runs = [runs] if not isinstance(runs, (list, tuple)) else list(runs)
    out = [check_confounds(runs)]
    for run in runs:
        ctx = context(run)
        for check in (check_r2, check_r14, check_r11):
            out.append(check(run, ctx=ctx, window=window))
    return out


def format_report(results: Iterable[CheckResult], *, verbose: bool = False) -> str:
    """The checks, rendered, with the ones that DID NOT RUN first.

    Order is deliberate. A check that did not run is a hole in the report, and a
    hole placed after the findings reads as a footnote to a complete picture."""
    results = list(results)
    parts = [format_result(r, verbose=verbose) for r in results if not r.ran]
    parts += [format_result(r, verbose=verbose) for r in results if r.ran]
    return '\n'.join(parts)
