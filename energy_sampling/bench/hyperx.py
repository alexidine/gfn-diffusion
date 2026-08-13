"""
`hyperx` -- hypergradient with the two structural gaps closed -- against the
published rule and the two incumbents.

THE PREDICTION, COMMITTED BEFORE RUNNING (see the session's operating rule: a
change must move the frontier, not slide along it):

  1. `rho > 1` only fires while the run is actually unstable, which is the HOT
     and EXPLOSIVE scenarios. On a cold start rho < 1 throughout, so the branch
     never triggers and hyper's cold-start number -- the best on the whole board
     -- must come back ESSENTIALLY UNCHANGED. If it moves, the term is doing
     something unaccounted for and no improvement elsewhere should be believed.

  2. It should help on hot/explosive recovery, the axis it was derived for.

  3. It should be roughly NEUTRAL ON NOISE. The derivation says nothing about
     noise; the only noise claim is that rho's bias pulls toward 1, i.e. the
     brake under-reacts rather than firing spuriously. If the noise cells improve
     a lot, suspect a fit to the bench rather than a derivation.

Reported per scenario rather than pooled, because pooling is what let a
worst-case improvement hide a cold-start regression last time.
"""
import math
import sys

import numpy as np

from bench.oracle import Surface, find_oracle
from bench.robustness import BASE, BUDGET
from bench.scenarios import SEED_LR, sc_blowup, steps_to_target

ARMS = (('hyperx', 'none'), ('hyper', 'none'),
        ('ray', 'ray'), ('ramp', 'plateau'))

CELLS = (('baseline', dict(BASE)),
         ('noise=0.5', dict(BASE, noise=0.5)),
         ('noise=2', dict(BASE, noise=2.0)),
         ('quartic=0.01', dict(BASE, quartic=1e-2)))

SCENARIOS = ('cold_start', 'blowup_100x', 'hot_90pct')


def main(seeds=15):
    seeds = tuple(range(int(seeds)))
    print(f'{"=" * 86}\nhyperx vs hyper vs incumbents -- mle, {len(seeds)} seeds, '
          f'budget {BUDGET:g}x\n{"=" * 86}')
    print('  per-scenario median slowdown vs this cell\'s oracle, and % of runs '
          'over budget.\n  COLD START IS THE FALSIFIER: hyperx must match hyper '
          'there.\n')

    for label, kw in CELLS:
        surface = Surface('mle', 'mle', kw, steps=2000, lr_grid=(1e-6, 1e-1, 12))
        try:
            oracle = find_oracle(surface, seeds=(0, 1, 2), verbose=False)
        except ValueError as e:
            print(f'  {label}: no usable oracle ({e})')
            continue
        denom = steps_to_target(surface.run(oracle.lr, seed=0, servo=False), oracle)
        if not denom:
            print(f'  {label}: oracle never reaches its own target')
            continue
        print(f'  {label:<14} oracle lr {oracle.lr:.3g}  target at {denom} steps')
        print(f'    {"arm":<14} ' + ' '.join(f'{s:>22}' for s in SCENARIOS))
        for climber, braker in ARMS:
            cells = []
            for sc in SCENARIOS:
                slow = []
                for seed in seeds:
                    if sc == 'blowup_100x':
                        run = sc_blowup(surface, oracle, seed,
                                        climber=climber, braker=braker)
                    else:
                        lr = SEED_LR if sc == 'cold_start' else oracle.hot_lr(0.9)
                        run = surface.run(lr, seed=seed, servo=True,
                                          climber=climber, braker=braker)
                    t = steps_to_target(run, oracle)
                    slow.append(math.inf if t is None else t / denom)
                arr = np.array(slow, dtype=float)
                live = arr[np.isfinite(arr)]
                med = float(np.median(live)) if len(live) else math.inf
                over = float((arr > BUDGET).mean())
                cells.append(f'{med:>7.2f} ({over:>3.0%} over)' if math.isfinite(med)
                             else f'{"never":>7} ({over:>3.0%} over)')
            print(f'    {climber + "+" + braker:<14} ' +
                  ' '.join(f'{c:>22}' for c in cells))
        print()


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
