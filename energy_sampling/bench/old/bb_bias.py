"""
IS bb's NOISE FAILURE VARIANCE OR BIAS? -- because only one of them is fixable
by averaging.

`bb` estimates alpha* = -g.s/(s.y) from two consecutive gradients. That is a
RATIO whose denominator is an estimated curvature, and under gradient noise the
denominator can approach and cross zero, which gives the ratio tails heavy enough
that a sample mean barely concentrates (the Cauchy limit being the case where
averaging does nothing at all). A median concentrates where a mean does not, and
`bb` uses a median, so the variance side is not hopeless.

The bias side is not so kind. The step is built FROM the earlier gradient,
s = -lr * P * g_{t-1}, while y = g_t - g_{t-1} CONTAINS -g_{t-1}. Their noise is
therefore correlated, and

    E[s.y] = s'Hs + lr * tr(P Sigma)

so the denominator carries a systematic positive noise term and alpha is biased
DOWNWARD -- bb should run too cold as noise rises, and no window length removes
a bias.

This separates the two against GROUND TRUTH. `MLEGame.alpha_star_true` computes
alpha* exactly from the analytic Hessian, so for every bb estimate we can form
the ratio alpha_bb / alpha_true and report:

    median  -- the BIAS. 1.0 means unbiased. Averaging cannot move this.
    IQR/med -- the SPREAD. Averaging should shrink this like 1/sqrt(N).

If spread falls with the window while the median walks away from 1, windows are
the wrong fix and pairing (both gradients on ONE batch) is the right one.
"""
import math
import sys

import numpy as np
import torch

from bench.old.oracle import Surface
from bench.old.robustness import BASE
from bench.old.scenarios import SEED_LR

NOISES = (0.01, 0.1, 0.5, 2.0)
WINDOWS = (1, 10, 50, 200)


def collect(noise, window, seed=0, steps=1500):
    """Run bb and record (alpha_bb, alpha_true) at every estimate it makes."""
    surface = Surface('mle', 'mle', dict(BASE, noise=noise), steps=steps,
                      lr_grid=(1e-6, 1e-1, 12))
    run = surface.make(SEED_LR, seed=seed, servo=True, climber='bb',
                       braker='none', standard={'bb_window': window})
    pairs = []
    orig = run._bb_tick

    def spy(theta_before, g_before):
        prev = run._bb_prev
        before = len(run._bb_window)
        orig(theta_before, g_before)
        # a fresh estimate landed in the rolling window this step
        if prev is not None and len(run._bb_window) and (
                len(run._bb_window) != before or window == 1):
            alpha_bb = run._bb_window[-1]
            prev_theta, prev_grad = prev
            s = theta_before - prev_theta
            true = run.game.alpha_star_true(prev_theta, s)
            if true and math.isfinite(true) and true > 0 and alpha_bb > 0:
                pairs.append(alpha_bb / true)
    run._bb_tick = spy
    run.run(steps)
    return np.array(pairs, dtype=float)


def rolling_median(x, w):
    """What a window of `w` would have produced, applied to the SAME stream."""
    if w <= 1 or len(x) < w:
        return x
    return np.median(np.lib.stride_tricks.sliding_window_view(x, w), axis=1)


def main(seeds=3):
    print(f'{"=" * 78}\nbb ESTIMATE vs GROUND TRUTH  (ratio alpha_bb / alpha_true)'
          f'\n{"=" * 78}')
    print('  median 1.0 = unbiased. BIAS is what averaging cannot fix;')
    print('  spread (IQR/median) is what it can. Windows applied POST-HOC to one')
    print('  recorded stream, because the window changes which estimates are')
    print('  ACTED on, not what any individual estimate is -- measuring it by')
    print('  re-running conflates the estimator with the trajectory it steers.')
    print(f'\n  {"noise":>6} {"window":>7} {"n":>6} {"median":>8} {"spread":>8} '
          f'{"frac<0.5x":>10}')
    for noise in NOISES:
        raw = np.concatenate([collect(noise, 1, seed=s) for s in range(int(seeds))])
        if not len(raw):
            continue
        for w in WINDOWS:
            r = rolling_median(raw, w)
            q1, med, q3 = np.percentile(r, [25, 50, 75])
            spread = (q3 - q1) / med if med > 0 else math.inf
            print(f'  {noise:>6g} {w:>7} {len(r):>6} {med:>8.3f} '
                  f'{spread:>8.2f} {float((r < 0.5).mean()):>10.0%}')
        print()


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
