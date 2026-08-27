"""Which checkpoint a divergence rewind restores.

2026-08-27, qm9c_t20: five consecutive fires each rewound to 'running' from
INSIDE the excursion (steps 21750-22350) because the healthy 'best' at step
18500 was 3250 steps old and the freshness bar is 10 x eval_period = 2500 at
this config's eval_period of 250. The LR was cut 16x across those fires and
changed nothing -- the weights were the problem, not the rate.

The fix separates two questions that shared one artifact: 'best' ranks QUALITY
(which a scalar does badly -- a sideways move can be a real improvement), while
the rewind only needs a SURVIVABLE state. 'last_ok' is the most recent sample
within a band of the best, so sideways drift qualifies without ranking it.
"""
import numpy as np


def _qualifies(record, tol=0.25):
    """The write-side rule from train.py: link 'running' -> 'last_ok' when the
    latest sample is within `tol` of the best, additive in the metric's own
    scale so it works for negative metrics too."""
    best = float(np.amin(record))
    return record[-1] <= best + tol * abs(best)


def test_the_band_admits_sideways_drift():
    """A sideways move must qualify -- that is the whole point of a band rather
    than an argmin. Ranking distributional quality by a scalar is what we are
    declining to do here."""
    assert _qualifies([17.0, 17.4, 16.9, 18.2])
    assert _qualifies([17.0, 20.0])          # +18%, inside a 25% band


def test_the_band_rejects_the_qm9c_t20_excursion():
    """The real numbers. best ~17, then the run drifted to 25.4 and exploded to
    62 -> 117. Every one of those must be refused, INCLUDING the 25.4 sample --
    it was already drifting, and admitting it is how a rewind lands inside the
    onset."""
    for bad in (25.4, 62.4, 98.8, 117.0):
        assert not _qualifies([17.0, bad]), f'{bad} should be outside the band'


def test_the_band_works_for_negative_metrics():
    """Phase-1 stages score on bwd/mle, which is ~-22. A multiplicative band
    would move the bar the WRONG WAY on a negative number (-22.9 * 1.25 is
    better, not worse), so the rule is additive in |best|."""
    assert _qualifies([-22.9, -21.0])        # mild worsening, inside
    assert not _qualifies([-22.9, -15.0])    # -22.9 + 0.25*22.9 = -17.2


def test_rewind_prefers_last_ok_above_everything_else():
    """Order is load-bearing. 'last_ok' must be consulted BEFORE 'best' and
    'running': it is the only target selected for being non-diverging rather
    than for ranking well, and 'running' is written UNCONDITIONALLY every 50
    steps so during an excursion it is guaranteed to carry it.

    NB the 2026-08-25 demotion of a STALE 'best' below 'running' is left
    intact -- it encodes its own real failure (a frozen near-init best
    re-tripping the bar, toy_wk_aug24). last_ok is what makes that question
    rare, not wrong."""
    src = open('train.py', encoding='utf-8').read()
    i = src.index('def _rewind_checkpoint_path')
    body = src[i:i + 6000]
    pos_ok = body.index("stage_and_step('last_ok')")
    pos_best = body.index("stage_and_step('best')")
    pos_run = body.index("stage_and_step('running')")
    assert pos_ok < pos_best < pos_run, (
        "rewind order regressed: last_ok must be consulted first")
