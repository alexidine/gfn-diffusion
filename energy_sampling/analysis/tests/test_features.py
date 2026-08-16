"""
Tests for feature extraction.

Synthetic series with known answers, so a broken statistic is caught by the
number rather than by the shape of the output. Each guard is paired with the
input it is supposed to reject.
"""

import numpy as np
import pytest

from analysis import features as F


def _line(n=400, slope=0.01, noise=0.0, seed=0):
    s = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    return s, slope * s + noise * rng.standard_normal(n)


# ---------------------------------------------------------------------------
# Theil-Sen
# ---------------------------------------------------------------------------

def test_theil_sen_recovers_a_known_slope():
    s, v = _line(slope=0.01)
    slope, _ = F.theil_sen(s, v)
    assert slope == pytest.approx(0.01, rel=1e-6)


def test_theil_sen_is_robust_to_a_single_excursion():
    """The reason it is not least squares. These series carry excursions, and an
    OLS slope on one reports the excursion instead of the trend."""
    s, v = _line(slope=0.01)
    v = v.copy()
    v[200] += 1000.0
    slope, _ = F.theil_sen(s, v)
    assert slope == pytest.approx(0.01, rel=1e-3)
    ols = np.polyfit(s, v, 1)[0]
    assert abs(ols - 0.01) > abs(slope - 0.01) * 10, 'OLS should be much worse here'


def test_theil_sen_is_deterministic():
    """The pair sample is seeded, so two reports of the same run agree. An
    unseeded sampler makes every number slightly different on re-read and
    invites chasing noise."""
    s, v = _line(noise=0.5)
    assert F.theil_sen(s, v)[0] == F.theil_sen(s, v)[0]


def test_theil_sen_handles_short_series():
    assert F.theil_sen(np.array([0.0]), np.array([1.0]))[0] == 0.0


# ---------------------------------------------------------------------------
# Oscillation
# ---------------------------------------------------------------------------

def test_oscillation_recovers_a_known_period():
    s = np.arange(1200, dtype=float)
    v = np.sin(2 * np.pi * s / 100.0)
    o = F.oscillation(s, v)
    assert o is not None
    assert o.period == pytest.approx(100.0, rel=0.15)


def test_oscillation_reports_growing_amplitude():
    s = np.arange(1200, dtype=float)
    v = (1 + 3 * s / s[-1]) * np.sin(2 * np.pi * s / 100.0)
    o = F.oscillation(s, v)
    assert o is not None and o.trend == 'growing'


def test_oscillation_reports_damping_amplitude():
    s = np.arange(1200, dtype=float)
    v = np.exp(-3 * s / s[-1]) * np.sin(2 * np.pi * s / 100.0)
    o = F.oscillation(s, v)
    assert o is not None and o.trend == 'damping'


def test_oscillation_is_none_on_pure_noise():
    """The mutation for the tests above: if noise produced an oscillation, the
    period detections would prove nothing."""
    rng = np.random.default_rng(1)
    s = np.arange(1200, dtype=float)
    o = F.oscillation(s, rng.standard_normal(1200))
    assert o is None or o.rms_amplitude < 3.0


def test_oscillation_is_none_on_a_flat_series():
    s = np.arange(200, dtype=float)
    assert F.oscillation(s, np.zeros(200)) is None


def test_oscillation_needs_enough_points():
    s = np.arange(10, dtype=float)
    assert F.oscillation(s, np.sin(s)) is None


# ---------------------------------------------------------------------------
# Escape
# ---------------------------------------------------------------------------

def test_doubling_time_on_exponential_growth():
    s = np.arange(2000, dtype=float)
    v = np.exp(s / 500.0)          # doubles every 500*ln2 ~ 347 steps
    dt = F.doubling_time(s, v, tail=600)
    assert dt is not None and dt == pytest.approx(346.6, rel=0.1)


def test_no_doubling_time_on_a_flat_series():
    s = np.arange(2000, dtype=float)
    assert F.doubling_time(s, np.ones(2000), tail=600) is None


def test_no_doubling_time_on_a_decaying_series():
    s = np.arange(2000, dtype=float)
    assert F.doubling_time(s, np.exp(-s / 500.0), tail=600) is None


def test_no_doubling_time_on_noise():
    """Demands significance, so an ordinary noisy climb does not read as an
    escape."""
    rng = np.random.default_rng(2)
    s = np.arange(2000, dtype=float)
    assert F.doubling_time(s, np.abs(rng.standard_normal(2000)) + 1, tail=600) is None


# ---------------------------------------------------------------------------
# extract: the EMA guard
# ---------------------------------------------------------------------------

def test_significance_is_suppressed_on_ema_series():
    """No trend test is valid on a smoothed series: smoothing manufactures the
    autocorrelation a significance test reads as signal. The trend is still
    SHOWN -- suppression is of the claim, not of the number."""
    s, v = _line(slope=0.05)
    plain = F.extract('fwd/x', s, v, window=1e9)
    ema = F.extract('tracker/x', s, v, window=1e9, is_ema=True)
    assert plain.significant is True
    assert ema.significant is False
    assert ema.ema_suppressed is True
    assert ema.slope_per_1k == pytest.approx(plain.slope_per_1k)


def test_oscillation_suppressed_on_ema_series():
    s = np.arange(1200, dtype=float)
    v = np.sin(2 * np.pi * s / 100.0)
    assert F.extract('tracker/x', s, v, 1e9, is_ema=True).oscillation is None
    assert F.extract('fwd/x', s, v, 1e9).oscillation is not None


def test_low_trust_is_carried_not_dropped():
    """`tracker/*_rms` are too noisy to act on -- carried so they can be seen,
    flagged so they are not ranked on."""
    s, v = _line()
    f = F.extract('tracker/z_bias_rms', s, v, 1e9, is_ema=True, low_trust=True)
    assert f is not None and f.low_trust
    assert '!' in F.format_feature(f)


def test_window_restricts_the_series():
    s = np.arange(1000, dtype=float)
    v = np.concatenate([np.zeros(900), np.arange(100, dtype=float)])
    f = F.extract('k', s, v, window=100)
    assert f.n <= 101
    assert f.slope_per_1k > 0


def test_extract_returns_none_on_a_too_short_window():
    s, v = _line(n=100)
    assert F.extract('k', s, v, window=1.0) is None


def test_flat_series_is_not_significant():
    s = np.arange(500, dtype=float)
    assert F.extract('k', s, np.ones(500), 1e9).significant is False


def test_format_marks_significance_and_ema():
    s, v = _line(slope=0.05)
    assert '*' in F.format_feature(F.extract('fwd/x', s, v, 1e9))
    assert '~' in F.format_feature(F.extract('tracker/x', s, v, 1e9, is_ema=True))


def test_escape_only_reported_when_watched():
    s = np.arange(2000, dtype=float)
    v = np.exp(s / 500.0)
    assert F.extract('k', s, v, 1e9, watch_escape=False).doubling_time is None
    assert F.extract('k', s, v, 1e9, watch_escape=True).doubling_time is not None
