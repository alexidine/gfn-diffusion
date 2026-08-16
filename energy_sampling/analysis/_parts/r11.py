"""R11 -- replay error below forward error.

`reading_runs.md` R11: replay is drawn from higher-residual trajectories by
construction, so its error sits ABOVE the forward error. The doc's stated
reference is ~2x, and below 1x is the band it names. This check reports the
number, the two series behind it, and WHICH BAND the number is in. What that
band implies about the run is the reader's to apply: R11 is a mechanism, and a
mechanism does not survive being compressed into a verdict.

THE ROUTE GATE IS A PROPERTY OF THE ROUTE, NOT OF THE KEYS. R11 is defined on
`K.R11_ROUTES`; on the conditional VarGrad route the answer is NA_ROUTE -- the
check RAN and its subject is not meaningful there. `keys.resolve` does NOT mark
either scatter_err series as NA on any route (its NA patterns cover log Z and the
TB residuals), so asking the key would return LIVE and hand back a number to be
read as if it were on a TB run. The gate is therefore tested BEFORE presence:
presence-first would report a VarGrad run that never ran a replay branch as
`not_run`, which says "the data is missing" where what is true is "the question
does not apply".

THE RATIO IS POINTWISE. A ratio of medians is not the median of ratios, and one
excursion in either series moves the former. Measured on a real five-stage run,
the two disagree by ~40% (pointwise 1.77 vs 2.48 over the same aligned ticks) --
enough to move the answer across a band.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import keys as K
from .base import (CheckResult, Context, Finding, State, context, series,
                   trailing)

_R11_CHECK = 'R11 replay vs forward error'

# Below this many aligned ticks the "median" is whichever tick the eval cadence
# happened to land on rather than a level, so the check refuses rather than
# reporting one. Also the guard that stops `base.series`' summary fallback --
# a single point carried from the summary -- being reported as a ratio.
_R11_MIN_ALIGNED = 8


def _r11_window_desc(window: Optional[float]) -> str:
    return 'all' if window is None else f'trailing {window:g} steps'


def _r11_align(num, den):
    """`(steps, numerator, denominator, n_dropped)` on one step grid.

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
        return empty, empty, empty, 0
    if np.any(np.diff(ds) <= 0):
        # np.interp requires an increasing xp and returns garbage silently
        # otherwise. Cheap insurance against a merged or resumed history.
        order = np.argsort(ds, kind='stable')
        ds, dv = ds[order], dv[order]

    inside = (ns >= ds[0]) & (ns <= ds[-1])
    s, a = ns[inside], nv[inside]
    b = np.interp(s, ds, dv)
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    return s[ok], a[ok], b[ok], int((~ok).sum())


def check_r11(run, *, ctx: Optional[Context] = None,
              window: Optional[float] = None) -> CheckResult:
    """`K.R11_NUMERATOR / K.R11_DENOMINATOR`, pointwise, median over the window.

    One subject, one row. Every path that cannot produce that row either says
    NA_ROUTE (the question does not apply here) or `not_run` with what it had --
    never an empty result, which renders identically to "checked, nothing wrong".
    """
    ctx = ctx or context(run)
    subject = f'{K.R11_NUMERATOR} / {K.R11_DENOMINATOR}'
    res = CheckResult(check=_R11_CHECK)
    routes = '/'.join(r.value for r in K.R11_ROUTES)

    if ctx.route is K.Route.UNKNOWN:
        # NOT NA_ROUTE. NA_ROUTE asserts the route is known and R11 does not
        # apply on it; an unclassified route leaves applicability UNDETERMINED,
        # and that is a hole in the report, which renders loudly, rather than a
        # table row, which does not.
        return CheckResult.not_run(
            _R11_CHECK,
            f'route not classified from the config, so it is unknown whether '
            f'R11 applies (it is defined on {routes}). The ratio was not '
            f'computed and nothing about replay was asserted.')

    if ctx.route not in K.R11_ROUTES:
        res.add(Finding(
            check=_R11_CHECK, subject=subject, state=State.NA_ROUTE,
            detail=(f'ratio withheld -- R11 is defined on {routes}; this run is '
                    f'on {ctx.route.value}')))
        return res

    num = series(run, K.R11_NUMERATOR)
    den = series(run, K.R11_DENOMINATOR)
    missing = [k for k, sv in ((K.R11_NUMERATOR, num), (K.R11_DENOMINATOR, den))
               if sv is None]
    if missing:
        return CheckResult.not_run(
            _R11_CHECK,
            f'ABSENT on a {ctx.route.value} run: {" and ".join(missing)}. '
            f'The ratio was not computed -- this is not "replay is fine".')

    s, a, b, n_dropped = _r11_align(num, den)
    ws, w_ratio = trailing(s, a / b, window)
    _, w_num = trailing(s, a, window)
    _, w_den = trailing(s, b, window)

    if len(ws) < _R11_MIN_ALIGNED:
        return CheckResult.not_run(
            _R11_CHECK,
            f'{len(ws)} aligned point(s) over {_r11_window_desc(window)}; '
            f'{_R11_MIN_ALIGNED} needed for a median. '
            f'{K.R11_NUMERATOR}: {len(num[0])} point(s); '
            f'{K.R11_DENOMINATOR}: {len(den[0])} point(s); '
            f'{len(s)} usable after alignment, {n_dropped} dropped.')

    med = float(np.median(w_ratio))
    numbers = dict(median_ratio=med,
                   num_median=float(np.median(w_num)),
                   den_median=float(np.median(w_den)),
                   n_aligned=len(ws),
                   window=_r11_window_desc(window),
                   ref_ratio=float(K.R11_HEALTHY_RATIO))
    if n_dropped:
        numbers['n_dropped'] = n_dropped

    # Bands only. Naming the band is mechanical; what the band means about the
    # run is R11's mechanism and belongs to the reader.
    if med < K.R11_OVERFIT_BELOW:
        state = State.FLAG
        detail = f'{med:.3g}x -- below the {K.R11_OVERFIT_BELOW:g}x band'
    elif med < K.R11_HEALTHY_RATIO:
        state = State.OK
        detail = (f'{med:.3g}x -- between {K.R11_OVERFIT_BELOW:g}x and the '
                  f'{K.R11_HEALTHY_RATIO:g}x reference')
    else:
        state = State.OK
        detail = f'{med:.3g}x -- at or above the {K.R11_HEALTHY_RATIO:g}x reference'

    res.add(Finding(check=_R11_CHECK, subject=subject, state=state,
                    detail=detail, numbers=numbers))
    return res
