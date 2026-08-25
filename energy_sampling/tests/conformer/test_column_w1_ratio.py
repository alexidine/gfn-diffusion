"""The worst-marginal W1 ratio must READ 1 ON A PERFECT SAMPLER and large on a bad one.

A convergence statistic that cannot separate those two is worse than none: it would sit on
the dashboard looking like evidence. So the load-bearing tests here are the pair -- draw the
"sampler" from the target itself and require ~1, then broaden it and require a large multiple.
Testing only the second would pass for a metric that returns a big number unconditionally.
"""
import numpy as np
import pytest

from conformer_modeller import _column_w1_ratio

D, N_REF, N = 12, 8000, 512


def _ref(rng):
    """A reference with heterogeneous per-column scales, so a metric that secretly depends
    on the raw scale rather than the ratio cannot pass by accident."""
    scales = np.linspace(0.05, 0.8, D)
    return rng.normal(0.0, 1.0, size=(N_REF, D)) * scales


NOT_PERIODIC = np.zeros(D, dtype=bool)


def test_a_sampler_drawn_from_the_target_reads_about_one():
    """THE CALIBRATION. Rows drawn from the reference are, by construction, a perfect
    sampler, so every column must sit at its own noise floor."""
    rng = np.random.default_rng(0)
    ref = _ref(rng)
    smp = ref[rng.choice(N_REF, size=N, replace=False)]
    m = _column_w1_ratio(smp, ref, NOT_PERIODIC, cache={})
    assert m, 'metric declined to report on a valid input'
    assert 0.4 < m['w1r/median'] < 2.0, m['w1r/median']
    assert m['w1r/worst'] < 3.5, m['w1r/worst']


def test_a_broadened_sampler_reads_far_above_one():
    """THE DISCRIMINATION. 2x too wide on every column -- the failure mode in the latent
    figure -- must be unmistakable, not a marginal shift."""
    rng = np.random.default_rng(1)
    ref = _ref(rng)
    smp = ref[rng.choice(N_REF, size=N, replace=False)] * 2.0
    m = _column_w1_ratio(smp, ref, NOT_PERIODIC, cache={})
    assert m['w1r/worst'] > 5.0, m['w1r/worst']
    assert m['w1r/n_above_2x'] >= D // 2, m['w1r/n_above_2x']


def test_it_separates_the_two_cases_by_a_wide_margin():
    """The pair, compared directly: a metric can pass both tests above and still be nearly
    blind if its two answers are close."""
    rng = np.random.default_rng(2)
    ref = _ref(rng)
    good = ref[rng.choice(N_REF, size=N, replace=False)]
    bad = ref[rng.choice(N_REF, size=N, replace=False)] * 2.0
    g = _column_w1_ratio(good, ref, NOT_PERIODIC, cache={})['w1r/worst']
    b = _column_w1_ratio(bad, ref, NOT_PERIODIC, cache={})['w1r/worst']
    assert b > 4 * g, f'good {g:.2f} vs bad {b:.2f} -- not separated'


def test_the_ratio_is_scale_invariant():
    """The reason for a ratio rather than wass_debiased's subtraction: scaling the whole
    space must not move the answer. A difference would scale with it."""
    rng = np.random.default_rng(3)
    ref = _ref(rng)
    smp = ref[rng.choice(N_REF, size=N, replace=False)] * 1.5
    a = _column_w1_ratio(smp, ref, NOT_PERIODIC, cache={})['w1r/worst']
    b = _column_w1_ratio(smp * 1000.0, ref * 1000.0, NOT_PERIODIC, cache={})['w1r/worst']
    assert abs(a - b) < 0.02 * a, f'{a:.4f} vs {b:.4f} -- ratio is not scale free'


def test_it_declines_rather_than_guessing_when_the_reference_is_too_small():
    """No floor can be estimated from a reference barely bigger than the sample, and a
    fabricated one would be indistinguishable from a real reading on the dashboard."""
    rng = np.random.default_rng(4)
    ref = _ref(rng)[:N + 10]
    smp = ref[rng.choice(len(ref), size=N, replace=False)]
    assert _column_w1_ratio(smp, ref, NOT_PERIODIC, cache={}) == {}


def test_a_constant_reference_column_is_excluded_not_divided_by():
    """A column with no spread has no scale to divide by; including it would emit inf."""
    rng = np.random.default_rng(5)
    ref = _ref(rng)
    ref[:, 3] = 0.0
    smp = ref[rng.choice(N_REF, size=N, replace=False)]
    m = _column_w1_ratio(smp, ref, NOT_PERIODIC, cache={})
    assert np.isfinite(m['w1r/worst'])
    assert m['w1r/n_live'] == D - 1
    assert m['w1r/worst_col'] != 3
