"""The energy-marginal check must catch what `energy_vs_reference` cannot.

That metric is one-sided -- it asks whether training bought LOWER energies -- so a sampler
with the correct median and twice the spread scores frac_below_ref_median = 0.5, the same
value a perfect sampler scores. The whole reason this module exists is that 0.5 means two
different things there. So the load-bearing test is the WIDTH case: same median, wrong
distribution, and the new metric has to separate it while the old one cannot.
"""
import numpy as np
import pytest

from energies.conformer_eval_metrics import (energy_marginal_overlap,
                                             energy_vs_reference)

N_REF, N = 8000, 1024


def _ref(rng):
    """Bimodal and skewed, like a real curated energy set -- not a Gaussian, so a metric
    that implicitly assumes one cannot pass by luck."""
    a = rng.normal(24.0, 0.8, size=int(N_REF * 0.3))
    b = rng.normal(31.0, 1.6, size=N_REF - len(a))
    return np.concatenate([a, b])


def test_a_perfect_sampler_sits_at_its_floor():
    rng = np.random.default_rng(0)
    r = _ref(rng)
    s = r[rng.choice(N_REF, N, replace=False)]
    m = energy_marginal_overlap(s, r, temperature=1.0, cache={})
    assert m['E/emarg_w1_ratio'] < 4.0, m['E/emarg_w1_ratio']
    assert m['E/emarg_overlap_rel'] > 0.95, m['E/emarg_overlap_rel']
    assert m['E/emarg_w1_kT'] < 0.25, m['E/emarg_w1_kT']


def test_the_width_failure_that_energy_vs_reference_scores_as_converged():
    """THE ONE THAT MATTERS. Median preserved, spread doubled: the old metric reports the
    converged value and the new one must not."""
    rng = np.random.default_rng(1)
    r = _ref(rng)
    s = r[rng.choice(N_REF, N, replace=False)]
    wide = (s - np.median(s)) * 2.0 + np.median(s)

    old = energy_vs_reference(wide, r)
    assert abs(old['E/frac_below_ref_median'] - 0.5) < 0.06, (
        'premise broken: the old metric was supposed to be fooled here')

    new = energy_marginal_overlap(wide, r, temperature=1.0, cache={})
    assert new['E/emarg_w1_ratio'] > 10.0, new['E/emarg_w1_ratio']
    assert new['E/emarg_overlap_rel'] < 0.85, new['E/emarg_overlap_rel']


def test_a_shift_of_one_kT_reads_about_one_kT():
    """w1_kT is meant to be physically readable, not merely monotone."""
    rng = np.random.default_rng(2)
    r = _ref(rng)
    s = r[rng.choice(N_REF, N, replace=False)] + 1.0
    m = energy_marginal_overlap(s, r, temperature=1.0, cache={})
    assert 0.8 < m['E/emarg_w1_kT'] < 1.25, m['E/emarg_w1_kT']


def test_temperature_scales_the_effect_size_and_not_the_ratio():
    """w1_kT is in units of kT; the significance ratio is dimensionless and must not move."""
    rng = np.random.default_rng(3)
    r = _ref(rng)
    s = r[rng.choice(N_REF, N, replace=False)] + 1.0
    a = energy_marginal_overlap(s, r, temperature=1.0, cache={})
    b = energy_marginal_overlap(s, r, temperature=2.0, cache={})
    assert abs(b['E/emarg_w1_kT'] - a['E/emarg_w1_kT'] / 2.0) < 1e-9
    assert abs(b['E/emarg_w1_ratio'] - a['E/emarg_w1_ratio']) < 1e-9


def test_it_declines_when_the_reference_is_too_small_to_give_a_floor():
    rng = np.random.default_rng(4)
    r = _ref(rng)[:N + 10]
    s = r[rng.choice(len(r), N, replace=False)]
    assert energy_marginal_overlap(s, r, temperature=1.0, cache={}) == {}


def test_non_finite_samples_are_dropped_not_propagated():
    """A single inf would otherwise take the whole quantile curve with it."""
    rng = np.random.default_rng(5)
    r = _ref(rng)
    s = r[rng.choice(N_REF, N, replace=False)].copy()
    s[:5] = np.inf
    m = energy_marginal_overlap(s, r, temperature=1.0, cache={})
    assert np.isfinite(m['E/emarg_w1_kT'])
    assert m['E/emarg_n'] == N - 5
