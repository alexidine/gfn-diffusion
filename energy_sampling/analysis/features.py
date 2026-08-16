"""
Feature extraction over raw scalar streams: trend, oscillation, runaway growth.

Ported from the sixth scratchpad copy of `wa.py`. The statistics are unchanged --
they were correct and their comments recorded real traps. What did NOT come
across is that file's `bellwether_verdict`, which printed conclusions like
"healthy climb" and "policy losing ground". This package does not emit verdicts.
It extracts the inputs a person reads and stops; a tool that concludes "the run
is healthy" is a failure of its spec, not a convenience.

The division of labour behind that: Python beats the model on raw data
processing -- extract trends, oscillations, noise levels -- while the reading
itself is context-dependent and jumps to whatever the phase and symptom
implicate. Encoding that jump is what produces confident wrong answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Trend significance bar, in units of |total change| / detrended sigma.
SIG_BAR = 2.0
# Amplitude-ratio bands for "is the oscillation growing", second half vs first.
GROWING, DAMPING = 1.3, 0.77


@dataclass(frozen=True)
class Oscillation:
    period: float          # in steps
    rms_amplitude: float
    amplitude_ratio: float  # rms(2nd half) / rms(1st half)

    @property
    def trend(self) -> str:
        if self.amplitude_ratio > GROWING:
            return 'growing'
        if self.amplitude_ratio < DAMPING:
            return 'damping'
        return 'steady'


@dataclass(frozen=True)
class Feature:
    """What was measured on one series over one window. No judgment attached."""

    key: str
    n: int
    last: float
    delta: float
    slope_per_1k: float
    sigma: float                       # detrended residual sd
    significant: bool                  # None-ish semantics: False when suppressed
    ema_suppressed: bool = False
    low_trust: bool = False
    oscillation: Optional[Oscillation] = None
    doubling_time: Optional[float] = None   # steps, when growing exponentially


def theil_sen(s: np.ndarray, v: np.ndarray, max_pairs: int = 4000):
    """Median-of-pairwise-slopes, plus the detrended residual.

    Robust to the heavy tails these series carry: a least-squares slope on a
    trace with one excursion in it reports the excursion, not the trend. The pair
    sample is seeded, so a report is reproducible."""
    n = len(s)
    if n < 3:
        return 0.0, np.zeros_like(v)
    rng = np.random.default_rng(0)
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    m = i != j
    ds = s[i[m]] - s[j[m]]
    ok = ds != 0
    slopes = (v[i[m]][ok] - v[j[m]][ok]) / ds[ok]
    slope = float(np.median(slopes)) if len(slopes) else 0.0
    detrended = v - slope * (s - s[0])
    return slope, detrended - np.median(detrended)


def oscillation(s: np.ndarray, v: np.ndarray) -> Optional[Oscillation]:
    """Dominant period and amplitude trend, via the detrended autocorrelation.

    Clean repeating oscillation here is a MECHANISM, not noise -- eval cadence,
    replay churn, buffer purge, the LR cycle all have periods, and matching a
    periodicity to a process with the same period is how the driver gets found.
    Reported without matching it to anything: naming the process is the read."""
    if len(s) < 40:
        return None
    dt = float(np.median(np.diff(s)))
    if dt <= 0:
        return None
    grid = np.arange(s[0], s[-1], dt)
    if len(grid) < 40:
        return None
    u = np.interp(grid, s, v)
    slope, _ = theil_sen(grid, u)
    d = u - slope * (grid - grid[0])
    d = d - d.mean()
    if np.allclose(d, 0):
        return None

    ac = np.correlate(d, d, 'full')[len(d) - 1:]
    if ac[0] == 0:
        return None
    ac = ac / ac[0]
    lo = int(np.argmax(ac < 0.3)) or 1      # first real dip
    if lo >= len(ac) - 2:
        return None
    seg = ac[lo:min(len(ac), lo + int(len(ac) * 0.8))]
    pk = int(np.argmax(seg)) + lo
    if ac[pk] < 0.25 or pk <= lo:
        return None
    half = len(d) // 2
    r1, r2 = float(np.std(d[:half])), float(np.std(d[half:]))
    return Oscillation(period=pk * dt,
                       rms_amplitude=float(np.std(d) * np.sqrt(2)),
                       amplitude_ratio=r2 / max(r1, 1e-12))


def doubling_time(s: np.ndarray, v: np.ndarray, tail: float = 600.0) -> Optional[float]:
    """Steps per doubling on the trailing segment, or None.

    Fits a log-slope on positive values only and demands significance, so an
    ordinary noisy climb does not read as an escape. Reported for series where
    runaway growth is the failure mode being watched."""
    m = (s >= s[-1] - tail) & (v > 0)
    if m.sum() < 8:
        return None
    slope, resid = theil_sen(s[m], np.log(v[m]))
    if slope <= 0:
        return None
    sig = abs(slope) * (s[m][-1] - s[m][0]) / max(float(np.std(resid)), 1e-12)
    if sig < 3:
        return None
    return float(np.log(2) / slope)


def extract(key: str, s: np.ndarray, v: np.ndarray, window: float,
            *, is_ema: bool = False, low_trust: bool = False,
            watch_escape: bool = False) -> Optional[Feature]:
    """Features for one series over the trailing `window` steps.

    EMA GUARD. `tracker/*` series are EMA outputs; the trend is shown and
    SIGNIFICANCE IS SUPPRESSED, because no trend test is valid on a smoothed
    series -- smoothing manufactures autocorrelation, which is exactly what a
    significance test reads as signal. Oscillation is suppressed for the same
    reason."""
    m = s >= max(s[-1] - window, s[0])
    s, v = s[m], v[m]
    if len(s) < 3:
        return None

    slope, resid = theil_sen(s, v)
    sigma = float(np.std(resid))
    sig = abs(slope) * (s[-1] - s[0]) / max(sigma, 1e-12)

    osc = None if is_ema else oscillation(s, v)
    if osc is not None and osc.rms_amplitude <= sigma:
        osc = None      # not distinguishable from the residual scatter

    return Feature(
        key=key, n=len(s), last=float(v[-1]), delta=float(v[-1] - v[0]),
        slope_per_1k=float(slope * 1000.0), sigma=sigma,
        significant=(False if is_ema else bool(sig > SIG_BAR)),
        ema_suppressed=is_ema, low_trust=low_trust, oscillation=osc,
        doubling_time=doubling_time(s, v) if watch_escape else None,
    )


def format_feature(f: Feature, width: int = 30) -> str:
    """One line per series. Marks: `*` significant trend, `~` EMA (significance
    suppressed), `!` low-trust (carried, never ranked on)."""
    mark = '~' if f.ema_suppressed else ('*' if f.significant else ' ')
    trust = '!' if f.low_trust else ' '
    line = (f'  {f.key:{width}s}{trust} last {f.last:10.4g}  d {f.delta:+10.3g}  '
            f'slope/1k {f.slope_per_1k:+10.3g}{mark} sigma {f.sigma:9.3g}')
    if f.oscillation is not None:
        o = f.oscillation
        line += f'  osc[T~{o.period:.0f} amp {o.rms_amplitude:.3g} {o.trend}]'
    if f.doubling_time is not None:
        line += f'  GROWING[x2 per {f.doubling_time:.0f} steps]'
    return line
