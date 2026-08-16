"""R14 -- a pinned metric is a dead sensor.

`reading_runs.md` R14: "Zero spread, a value bound at its clip, a threshold
annealed below its own noise floor, a censored estimator reported at its
censoring bound. None of these are readings." Scope: the series a controller
READS.

THE SUBJECT LIST COMES FROM THE CONFIG. This codebase's configs name their own
sensors -- the balance controller's `balance_metrics_<mode>`, the LR
controller's `lr_sensor_metrics_<j>`, the stage's `exit_<j>_metric`, the buffer
servo's numerator and denominator, the anchor gate's ceiling and floor. Reading
them is what makes this check general rather than a hardcoded list that rots
one rename after it is written.

It is also what keeps the check honest in the other direction. A legitimately
constant SET POINT is not a dead sensor: `protocol/rt_setpoint` never moves
because it is config being echoed, not a reading, and a check built from "every
flat series in the run" reports it every time. Only a series something reads is
a subject here.

WHAT IS REPORTED, AND WHAT IS NOT. Each row carries the numbers that produced
it -- tick counts, extremes, the fraction pinned, the bar and the sigma it was
compared against. What a pinned sensor means for the run is the reader's; this
check says which condition holds and hands over the arithmetic.

TRAPS THIS ENCODES.

  * H5 -- `tracker/*` are EMA OUTPUTS. Smoothing manufactures autocorrelation
    and lowers variance, so an EMA series can read as pinned when the filter is
    what is flat. The row says so; it does not suppress the finding, because a
    smoothed sensor a controller is steering on is still a sensor whose spread
    the controller cannot see.
  * SHORTNESS IS NOT A FINDING. A series with a handful of ticks has no
    measurable spread, and calling that a dead sensor fires on every run read
    early and on every config-only capture. Under `_R14_MIN_TICKS` the row says
    how many ticks there were and asserts nothing about the sensor.
  * CENSORING IS CHECKED BEFORE PINNING. A t-statistic clamped to +/-99 for
    most of a run is zero-spread too, and reporting it as "flat" loses the one
    fact that explains it: `ray_calibration` clamps before logging, so the
    values above the bound were never in the record to begin with.
  * ABSENT IS A FINDING, NA_ROUTE IS NOT. A controller reading a series the run
    does not log is a dead sensor in the most literal sense. A series that is
    not meaningful on this route is NOT that -- it is logged, populated, and
    not this route's to read, and flagging it would send the reader hunting a
    logging bug that is not there.

R13 -- WHERE THE BAR COMES FROM. "Never ratchet a threshold below a floor you
have not measured." The protocol publishes each live bar in the METRIC'S OWN
UNITS at `K.EXIT_THRESHOLD_TRACE`, and this check finds the pairs by inverting
`K.metric_tag` against the keys the run actually logged rather than by asking
the config which conditions exist. That is deliberate: measured against
`protocol.py`, the publisher of `protocol/thr_*` is the LEXICOGRAPHIC BALANCE
controller's rules, not the stage exit block (which publishes only
`protocol/exit_streak_*`). Deriving the pairs from the exit config alone finds
nothing on a real run and reports it as a clean bill.

The comparison span is the OVERLAP -- the metric restricted to the steps the
bar actually covers. A bar that appears at step 9,570 compared against sigma
from the whole run is compared against a regime it was never in, and on real
data that flips the answer both ways.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import features as F
from .. import keys as K
from .base import (CheckResult, Context, Finding, State, context, series,
                   trailing)

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
    resn, = K.resolve(run.available_keys(), [metric], ctx.route)

    if resn.state is K.KeyState.NA_ROUTE:
        # NOT a flag, and not ABSENT. The key is there and carries numbers; what
        # is true is that this route is not the one they mean.
        res.add(Finding(_R14_CHECK, subject, State.NA_ROUTE,
                        f'{resn.note} -- a controller reads it, and its spread '
                        f'is not this route\'s to interpret'))
        return

    if resn.state is K.KeyState.ABSENT:
        res.add(Finding(_R14_CHECK, subject, State.FLAG,
                        f'{_R14_TAG_ABSENT} -- a controller reads this series '
                        f'and the run does not log it: {resn.note}'))
        return

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

    if not subjects and not bars:
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
    return res
