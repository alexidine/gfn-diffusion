"""
Do the metrics measure what they claim, on traces whose answer is known by hand?

Written before the first battery run, on synthetic traces rather than bench
output, because the previous generation of metrics was debugged against real runs
and three defects survived that: a censored ratio reported as a slowdown, a band
degenerate at its own edge, and a perfect score mapped to "no data". A metric
tested only against plausible-looking data cannot be caught being plausible and
wrong.
"""
import math

import numpy as np

import pytest

from bench.metrics import (backslide, catastrophes, final_loss, lead_fraction,
                           lr_stability, smoothed_loss)


class _Arm:
    def __init__(self, name):
        self.name = name


class _Run:
    def __init__(self, losses=None, lrs=None, name='a', seed=0,
                 divergences=0, aborted=None):
        n = len(losses if losses is not None else lrs)
        losses = [1.0] * n if losses is None else losses
        lrs = [1e-3] * n if lrs is None else lrs
        self.trace = [{'step': i, 'loss': losses[i], 'lr': lrs[i]}
                      for i in range(n)]
        self.arm = _Arm(name)
        self.seed = seed
        self.divergences = divergences
        self.aborted = aborted


# ------------------------------------------------------------- final_loss

def test_final_loss_is_a_window_median_not_the_last_value():
    """One lucky last step must not win. The last value here is the best in the
    run and the median of the window is not."""
    r = _Run(losses=[10.0] * 150 + [5.0] * 99 + [0.001])
    assert final_loss(r, window=100) == pytest.approx(5.0)


def test_final_loss_ignores_a_heavy_tail_rather_than_being_set_by_it():
    r = _Run(losses=[1.0] * 100 + [1.0] * 99 + [1e12])
    assert final_loss(r, window=100) == pytest.approx(1.0)


def test_final_loss_of_an_all_dead_run_is_inf_not_zero():
    """inf ranks last. 0.0 would rank FIRST -- a dead run winning the board is
    the reassurance-shaped failure this project keeps hitting."""
    r = _Run(losses=[float('nan')] * 200)
    assert final_loss(r) == math.inf


# ---------------------------------------------------------- lead_fraction

def test_lead_fraction_sums_to_one():
    a = _Run(losses=[1.0] * 100, name='a')
    b = _Run(losses=[2.0] * 100, name='b')
    c = _Run(losses=[3.0] * 100, name='c')
    got = lead_fraction([a, b, c])
    assert sum(got.values()) == pytest.approx(1.0)
    assert got['a'] == pytest.approx(1.0)
    assert got['b'] == 0.0


def test_ties_are_split_not_given_to_the_first_arm():
    """Identical arms must tie at 0.5. Awarding by list order would silently
    rank whichever arm was declared first."""
    a = _Run(losses=[1.0] * 100, name='a')
    b = _Run(losses=[1.0] * 100, name='b')
    got = lead_fraction([a, b])
    assert got['a'] == pytest.approx(0.5)
    assert got['b'] == pytest.approx(0.5)


def test_an_arm_that_dies_early_cannot_inflate_the_others():
    """Steps where nobody has a usable loss leave the denominator."""
    a = _Run(losses=[1.0] * 50 + [float('nan')] * 50, name='a')
    b = _Run(losses=[2.0] * 50 + [float('nan')] * 50, name='b')
    got = lead_fraction([a, b])
    # 'a' leads every counted step; the nan tail is dropped, so it is 1.0 not 0.5
    assert got['a'] == pytest.approx(1.0)


def test_the_lead_can_change_hands():
    """Long enough that the EMA's ~25-step lag is negligible; at 100 steps the
    lag alone put the first arm at 0.67 and that is the smoothing, not a bias."""
    a = _Run(losses=[1.0] * 500 + [9.0] * 500, name='a')
    b = _Run(losses=[9.0] * 500 + [1.0] * 500, name='b')
    got = lead_fraction([a, b])
    assert got['a'] == pytest.approx(0.5, abs=0.05)
    assert got['b'] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------- lr_stability

def test_a_constant_rate_is_perfectly_stable():
    r = _Run(lrs=[1e-3] * 200)
    st = lr_stability(r)
    assert st['sd'] == pytest.approx(0.0)
    assert st['max_jump'] == pytest.approx(0.0)
    assert st['span'] == pytest.approx(0.0)


def test_stability_is_scale_free():
    """The same multiplicative behaviour at 1e-5 and at 1e-1 must score the
    same. In linear space it would not -- that is why this is log space."""
    lo = _Run(lrs=[1e-5 * (1.1 ** (i % 5)) for i in range(200)])
    hi = _Run(lrs=[1e-1 * (1.1 ** (i % 5)) for i in range(200)])
    a, b = lr_stability(lo), lr_stability(hi)
    assert a['sd'] == pytest.approx(b['sd'])
    assert a['max_jump'] == pytest.approx(b['max_jump'])


def test_max_jump_catches_one_wild_swing_that_sd_barely_notices():
    """The case the second number exists for: 199 quiet steps and one 100x
    lurch. sd stays small; max_jump is ln(100)."""
    r = _Run(lrs=[1e-3] * 100 + [1e-1] + [1e-3] * 99)
    st = lr_stability(r)
    assert st['max_jump'] == pytest.approx(math.log(100), rel=1e-6)
    assert st['sd'] < 0.5


def test_non_positive_rates_are_dropped_not_logged():
    r = _Run(lrs=[1e-3] * 100 + [0.0] * 10 + [1e-3] * 90)
    assert math.isfinite(lr_stability(r)['sd'])


# ------------------------------------------------------------- backslide

def test_a_monotone_descent_never_backslides():
    r = _Run(losses=[100.0 - i * 0.1 for i in range(500)])
    assert backslide(r) == pytest.approx(0.0)


def test_a_monotone_rise_backslides_always():
    """Not exactly 1.0: the EMA starts at the first sample and takes ~1 horizon
    to catch the trend, so the opening comparisons sit under the threshold.
    Measured 0.977 -- that is the smoothing warm-up, not a miss."""
    r = _Run(losses=[1.0 + i * 0.1 for i in range(500)])
    assert backslide(r) > 0.95


def test_pure_noise_does_not_read_as_backsliding():
    """
    THE REGRESSION TEST FOR THE DEFECT THIS METRIC WAS REBUILT AROUND. The first
    version returned `mean(diff > 0)` on the smoothed series and scored **0.501**
    here -- a flat loss reported as backsliding half the time, which would have
    ranked arms by batch noise. Both shapes below must read ~0.
    """
    alternating = [1.0 + (0.3 if i % 2 else -0.3) for i in range(500)]
    assert backslide(_Run(losses=alternating)) < 0.05

    rng = np.random.default_rng(0)
    noisy_flat = list(1.0 + rng.normal(0, 0.3, 500))
    assert backslide(_Run(losses=noisy_flat)) < 0.10


def test_a_real_rise_is_caught_through_heavy_noise():
    """...while still catching a genuine upward trend buried in the same noise."""
    rng = np.random.default_rng(0)
    rising = [1.0 + 0.02 * i + rng.normal(0, 0.3) for i in range(500)]
    assert backslide(_Run(losses=rising)) > 0.8


# ----------------------------------------------------------- catastrophes

def test_catastrophes_are_counts_not_rates():
    r = _Run(losses=[1.0] * 98 + [float('inf')] * 2, divergences=3,
             aborted='budget')
    got = catastrophes(r)
    assert got['divergences'] == 3
    assert got['aborted'] is True
    assert got['nonfinite_steps'] == 2


def test_a_clean_run_reports_zeros_not_none():
    got = catastrophes(_Run(losses=[1.0] * 100))
    assert got == {'divergences': 0, 'aborted': False, 'nonfinite_steps': 0}


# -------------------------------------------------------------- smoothing

def test_smoothed_loss_holds_through_a_nan_rather_than_resetting():
    s = smoothed_loss(_Run(losses=[1.0] * 50 + [float('nan')] * 5 + [1.0] * 50))
    assert all(x is not None for x in s[50:55])


def test_the_scoring_reads_eloss_and_not_the_training_loss():
    """
    THE WHOLE SCORING PREMISE, WHICH HAD NO TEST.

    `_series` substitutes `SCORE_KEY` ('eloss', the noise-free loss) for 'loss'
    whenever a trace carries it. That substitution is the reason the battery's
    rankings are not coin flips: near the fixed point the training loss is
    dominated by a noise term whose SIGN IS RANDOM, so a median over a window
    ranks arms by where that term happened to land.

    Measured: deleting the substitution from `_series` left 47/47 tests passing,
    and so did setting `SCORE_KEY = 'loss'`. Every test in this file built traces
    with no `eloss` key at all, so they all exercised the FALLBACK branch and
    none of them ever touched the branch the battery actually runs.
    """
    run = _Run(losses=[7.0] * 300)
    for h in run.trace:
        h['eloss'] = 0.5
    assert final_loss(run) == pytest.approx(0.5), (
        'final_loss read the noisy training loss, not the noise-free one -- the '
        'battery is ranking arms on the sign of a random term')


def test_one_stray_eloss_does_not_switch_the_whole_series():
    """
    `_series` switches on `any(v is not None ...)`, so a SINGLE non-None `eloss`
    flips the entire series over to a key that is absent everywhere else --
    measured, a trace with `eloss` on 1 step of 300 scored 0.5 off that one
    sample while `loss` said 7.0. A partially-populated trace should not be
    scored off the handful of steps that happen to carry the key.
    """
    run = _Run(losses=[7.0] * 300)
    run.trace[137]['eloss'] = 0.5
    got = final_loss(run)
    assert got == pytest.approx(7.0), (
        f'one stray eloss in 300 steps set the score to {got:.4g}; the run is '
        f'being scored off a single sample')
