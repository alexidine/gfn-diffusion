"""
DOES AVERAGING BUY THE CHEAP SENSORS THEIR NOISE ROBUSTNESS BACK?

The guarantee board found that `bb` and `armijo` are clean at low gradient noise
and collapse at high noise (bb 0% -> 43% over budget, armijo 0% -> 88%), while
the ray probe degrades gracefully and a blind ramp is nearly immune. The
structural difference is that the probe averages its statistic over `n_sub`
paired sub-batches and applies a significance test before acting; these two act
on a single noisy sample.

If that is the whole story, giving them a comparable averaging window should
recover most of the gap, and the required window should GROW WITH THE NOISE. If
it is not, the window will not help and the weakness is in the estimator rather
than in its sample size -- which would be the more interesting answer, and is why
this is worth running rather than assuming.

Averaging is applied to the STATISTIC, not the decision:
  * bb      rolling median over the last N alpha estimates
  * armijo  mean of the signed sufficient-decrease MARGIN over N steps.
            Averaging the accept/reject BOOL would fix nothing -- a coin flip on
            the decision stays a coin flip however many you take.
"""
import math
import sys

import numpy as np

from bench.oracle import Surface, find_oracle
from bench.robustness import BASE, BUDGET, SCENARIOS
from bench.scenarios import SEED_LR, sc_blowup, steps_to_target

WINDOWS = (1, 10, 50, 200)
NOISES = (0.01, 0.5, 2.0)

#: `ramp`+`plateau` carries no window; it is the noise-immunity reference the
#: windowed arms are trying to reach.
REFERENCE = ('ramp', 'plateau')


def cell(noise, seeds, verbose=True):
    surface = Surface('mle', 'mle', dict(BASE, noise=noise), steps=2000,
                      lr_grid=(1e-6, 1e-1, 12))
    oracle = find_oracle(surface, seeds=(0, 1, 2), verbose=False)
    denom = steps_to_target(surface.run(oracle.lr, seed=0, servo=False), oracle)
    if not denom:
        print(f'  noise={noise:g}: oracle never reaches its own target')
        return []

    def over(climber, braker, standard=None):
        slow = []
        for sc in SCENARIOS:
            for seed in seeds:
                if sc == 'blowup_100x':
                    run = sc_blowup(surface, oracle, seed, climber=climber,
                                    braker=braker)
                else:
                    lr = SEED_LR if sc == 'cold_start' else oracle.hot_lr(0.9)
                    run = surface.run(lr, seed=seed, servo=True, climber=climber,
                                      braker=braker, standard=standard)
                t = steps_to_target(run, oracle)
                slow.append(math.inf if t is None else t / denom)
        arr = np.array(slow, dtype=float)
        live = arr[np.isfinite(arr)]
        return (float((arr > BUDGET).mean()),
                float(np.percentile(live, 90)) if len(live) else math.inf)

    if verbose:
        print(f'\n  noise={noise:<6g} oracle lr {oracle.lr:.3g}  '
              f'target at {denom} steps')
        print(f'    {"window":>7} {"bb %>2x":>9} {"bb p90":>8} '
              f'{"armijo %>2x":>12} {"armijo p90":>11}')
    rows = []
    for w in WINDOWS:
        bo, bp = over('bb', 'none', {'bb_window': w})
        ao, ap = over('armijo', 'none', {'armijo_window': w})
        rows.append(dict(noise=noise, window=w, bb=bo, armijo=ao))
        if verbose:
            print(f'    {w:>7} {bo:>8.0%} {bp:>8.2f} {ao:>11.0%} {ap:>11.2f}')
    ro, rp = over(*REFERENCE)
    if verbose:
        print(f'    {"ramp+pl":>7} {"":>8} {"":>8} {ro:>11.0%} {rp:>11.2f}'
              f'   <- windowless reference')
    return rows


def main(seeds=12):
    seeds = tuple(range(int(seeds)))
    print(f'{"=" * 74}\nAVERAGING SWEEP -- mle, {len(seeds)} seeds, '
          f'budget {BUDGET:g}x\n{"=" * 74}')
    rows = []
    for n in NOISES:
        rows.extend(cell(n, seeds))
    print(f'\n{"=" * 74}\nBEST WINDOW PER NOISE LEVEL\n{"=" * 74}')
    for n in NOISES:
        sub = [r for r in rows if r['noise'] == n]
        if not sub:
            continue
        b = min(sub, key=lambda r: r['bb'])
        a = min(sub, key=lambda r: r['armijo'])
        print(f'  noise={n:<6g} bb: window {b["window"]:>3} -> {b["bb"]:.0%}   '
              f'armijo: window {a["window"]:>3} -> {a["armijo"]:.0%}')
    return rows


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
