"""
Do the off-target metrics measure what the requirement says?

The requirement is MK's: "should not get stuck too-hot or too-cold for long
periods". Two numbers, because one cannot express it -- the TOTAL time outside
the band, and the LONGEST unbroken excursion. A run that crosses the band 200
times and one that parks outside it for 200 steps have the same total and are
different failures.

Hand-built histories with a known answer, so a defect in the metric cannot hide
behind a plausible-looking bench result.
"""
import math

import pytest

from bench.scenarios import (ON_TARGET_BAND, longest_off_target,
                             time_off_target)


class _Run:
    """Minimal stand-in: the metrics read `history[i]['lr']` and nothing else."""

    def __init__(self, lrs):
        self.history = [{'lr': x} for x in lrs]


REF = 1e-3


def test_a_run_parked_on_target_scores_zero():
    r = _Run([REF] * 100)
    assert time_off_target(r, REF)['off'] == 0.0
    assert longest_off_target(r, REF) == 0.0


def test_the_band_edges_are_inside():
    """Exactly 2x and exactly 0.5x count as on target -- the band is inclusive,
    so a controller sitting on the stated tolerance is not scored as failing."""
    r = _Run([REF * ON_TARGET_BAND, REF / ON_TARGET_BAND] * 50)
    assert time_off_target(r, REF)['off'] == 0.0


def test_hot_and_cold_are_reported_apart():
    r = _Run([REF * 10] * 30 + [REF / 10] * 20 + [REF] * 50)
    got = time_off_target(r, REF)
    assert got['hot'] == pytest.approx(0.30)
    assert got['cold'] == pytest.approx(0.20)
    assert got['off'] == pytest.approx(0.50)


def test_total_and_longest_disagree_where_they_should():
    """
    THE CASE THE SECOND METRIC EXISTS FOR. Both runs spend 50 of 100 steps off
    target; one oscillates, one parks. Same total, 25x different excursion.
    """
    oscillating = _Run([REF, REF * 10] * 50)
    parked = _Run([REF] * 50 + [REF * 10] * 50)
    assert time_off_target(oscillating, REF)['off'] == pytest.approx(0.5)
    assert time_off_target(parked, REF)['off'] == pytest.approx(0.5)
    assert longest_off_target(oscillating, REF) == pytest.approx(0.01)
    assert longest_off_target(parked, REF) == pytest.approx(0.50)


def test_a_late_excursion_is_caught():
    """Ending badly is a failure even if most of the run was fine -- a
    trailing-window metric would miss this."""
    r = _Run([REF] * 90 + [REF * 50] * 10)
    assert time_off_target(r, REF)['hot'] == pytest.approx(0.10)
    assert longest_off_target(r, REF) == pytest.approx(0.10)


def test_non_finite_rates_are_dropped_not_counted():
    """A diverged run writes nan into the history; counting nan as 'off' would
    conflate divergence (already scored by the tripwire) with mis-setting."""
    r = _Run([REF] * 50 + [float('nan')] * 30 + [REF * 10] * 20)
    got = time_off_target(r, REF)
    assert got['n'] == 70
    assert got['hot'] == pytest.approx(20 / 70)


def test_an_empty_or_unusable_run_returns_none_rather_than_zero():
    """Zero would read as PERFECT. Nothing measured must not look like nothing
    wrong -- that is the reassurance-shaped failure this project keeps hitting."""
    assert time_off_target(_Run([]), REF) is None
    assert longest_off_target(_Run([]), REF) is None
    assert time_off_target(_Run([REF] * 10), 0.0) is None
    assert time_off_target(_Run([REF] * 10), math.nan) is None
