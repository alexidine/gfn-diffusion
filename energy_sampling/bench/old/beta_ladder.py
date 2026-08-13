"""
CAN HYPERGRADIENT BE RUN FASTER THAN beta = 0.02?

MK's question: hypergradient is ~free, so why not run it harder? The cadence
answer is no -- it needs consecutive gradients, so it already fires every step
and there is no "more often". The only knob is the GAIN, which caps the rate
change at `exp(beta)` per step: 2% at the published 0.02.

The bench has said no to raising it (0.02 -> 0.04 -> 0.08 scored monotonically
worse). The reason is a random walk: each step moves log-lr by `beta * cos`, so
the excursion after T steps goes as `beta * sigma(cos) * sqrt(T)`, and the
surface has an absorbing boundary above and none below. Raising beta scales the
excursion without improving the signal-to-noise of any single decision.

WHY THAT VERDICT IS SUSPECT ON THE REAL SYSTEM. It was measured where
`sigma(cos)` is much larger than the real system's. Comparing 400-step windows
to 400-step windows (`bench/cos_axis.py`, the whole-run IQR is NOT comparable --
it is dominated by within-run drift):

    bench, any cell, at a window whose median matches the real system:
        median 0.13-0.34   IQR 0.40-0.51
    real fused (elj nehzor sg14, 4 checkpoints):
        median 0.29        IQR 0.11

About 4x tighter -- so the same random-walk risk allows roughly 4x the gain,
i.e. beta ~ 0.08. And the qualitative difference is larger than the ratio: the
real reading has p25 = 0.24, so it essentially NEVER CHANGES SIGN, while the
bench at a comparable median straddles zero constantly. A drift that does not
flip sign is not a random walk at all.

WHAT THIS FILE CANNOT DO. No cell in the bench's surface family reproduces
(median 0.29, IQR 0.11): the quadratic bowl goes from aligned (median ~1, IQR
~0) straight to the noise ball (median ~ -0.15) without a stable intermediate,
because it has no persistent slowly-varying gradient component. The real fused
gradient has one (a shared Z/level direction). So this ladder tests beta where
the statistic is ~4x noisier than reality, which makes every result here a LOWER
BOUND on the safe gain -- if a beta survives this, it survives the real system.
That is stated rather than fixed, and the fix is a surface with a persistent
drift term.

Scored on both metrics, because they answer different questions:
  * `%over`      -- time to a target, the crucible's metric
  * `off-target` -- fraction of STEPS outside a 2x band around the reference
                    rate, split hot/cold, plus the longest unbroken excursion.
                    This is MK's requirement ("should not get stuck too-hot or
                    too-cold for long periods") measured directly.
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.old.crucible import (BUDGET, DEEP_FRAC, EQ, MLE, SCENARIOS, _mk,
                            _oracle_task, _sc_drift, _sc_mixture, _sc_regime)
from bench.old.scenarios import (SEED_LR, longest_off_target, steps_to_target,
                             time_off_target)

#: The published gain and four multiples of it. 0.08 is the estimate the cos
#: measurement supports; 0.15 and 0.3 are past it, to find where it breaks
#: rather than only confirming where it holds.
BETAS = (0.02, 0.05, 0.08, 0.15, 0.3)

CELLS = [
    # the workhorse bowl, and the two axes that punish a hot rate
    ('mle base', 'mle', dict(MLE), 2000, (1e-6, 1e-1, 12), {}),
    ('mle n0.5', 'mle', dict(MLE, noise=0.5), 2000, (1e-6, 1e-1, 12), {}),
    ('mle n2', 'mle', dict(MLE, noise=2.0), 2000, (1e-6, 1e-1, 12), {}),
    ('mle q0.1', 'mle', dict(MLE, quartic=1e-1), 2000, (1e-6, 1e-1, 12), {}),
    ('mle d2048 n2', 'mle', dict(MLE, dim=2048, noise=2.0), 2000,
     (1e-6, 1e-1, 12), {}),
    # the three-player surface: structurally the closest thing here to TB, and
    # the one where a too-fast climber has another player to destabilise
    ('eq base', 'equilibration', dict(EQ), 3000, (1e-4, 1e1, 12),
     {'lr_flow': 1.0}),
    ('eq n1', 'equilibration', dict(EQ, noise=1.0), 3000, (1e-4, 1e1, 12),
     {'lr_flow': 1.0}),
]


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _arm(item):
    cell, oracle, denom, reg, beta, seeds = item
    std = {'hyper_beta': beta, 'hyper_beta_down': beta}
    s = _mk(cell)
    slow, off, hot, cold, longest, div = [], [], [], [], [], 0
    for sc in SCENARIOS:
        for seed in seeds:
            if sc == 'drift_10x':
                run = _sc_drift(s, oracle, seed, 'hyperx', 'none', std)
            elif sc == 'regime_change':
                run = _sc_regime(s, oracle, seed, 'hyperx', 'none', std)
            elif sc == 'mixture_drift':
                run = _sc_mixture(s, oracle, seed, 'hyperx', 'none', std)
            elif sc == 'cold_start':
                run = s.run(SEED_LR, seed=seed, servo=True, climber='hyperx',
                            braker='none', standard=std)
            else:
                run = s.run(oracle.hot_lr(0.9), seed=seed, servo=True,
                            climber='hyperx', braker='none', standard=std)
            ref, ref_denom = ((reg[0], reg[1]) if sc == 'regime_change' and reg
                              else (oracle, denom))
            t = steps_to_target(run, ref, frac=DEEP_FRAC)
            slow.append(math.inf if t is None else t / ref_denom)
            o = time_off_target(run, ref.lr)
            if o is not None:
                off.append(o['off'])
                hot.append(o['hot'])
                cold.append(o['cold'])
                lo = longest_off_target(run, ref.lr)
                if lo is not None:
                    longest.append(lo)
            div += run.divergences
    arr = np.array(slow, dtype=float)
    return dict(cell=cell[0], beta=beta, n=len(arr),
                over=float((arr > BUDGET).mean()),
                off=float(np.mean(off)) if off else math.nan,
                hot=float(np.mean(hot)) if hot else math.nan,
                cold=float(np.mean(cold)) if cold else math.nan,
                longest=float(np.mean(longest)) if longest else math.nan,
                div=div / max(len(arr), 1))


def main(seeds=10, workers=None):
    seeds = tuple(range(int(seeds)))
    workers = int(workers or max(2, min(20, (os.cpu_count() or 4) - 4)))
    print(f'{"=" * 96}\nBETA LADDER -- can hypergradient be run faster?\n'
          f'{len(seeds)} seeds, betas {BETAS}, {workers} workers\n'
          f'NOTE: the bench statistic is ~4x noisier than the real fused '
          f'signal, so these are LOWER bounds\n{"=" * 96}\n')

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker) as pool:
        oracles = {}
        for label, got, why in pool.map(_oracle_task, CELLS):
            if got is None:
                print(f'  {label:<14} SKIPPED -- {why}')
            else:
                oracles[label] = got
        jobs = [(c, oracles[c[0]][0], oracles[c[0]][1], oracles[c[0]][3],
                 b, seeds)
                for c in CELLS if c[0] in oracles for b in BETAS]
        rows = list(pool.map(_arm, jobs))

    print()
    for c in CELLS:
        label = c[0]
        if label not in oracles:
            continue
        print(f'  {label}   (oracle lr {oracles[label][0].lr:.3g})')
        print(f'    {"beta":>6} {"%over":>7} {"off-target":>11} {"too hot":>9} '
              f'{"too cold":>9} {"longest":>9} {"div/run":>8}')
        for r in [r for r in rows if r['cell'] == label]:
            print(f'    {r["beta"]:>6.2f} {r["over"]:>6.0%} {r["off"]:>11.1%} '
                  f'{r["hot"]:>9.1%} {r["cold"]:>9.1%} {r["longest"]:>9.1%} '
                  f'{r["div"]:>8.2f}')
        print()

    print(f'{"=" * 96}\nACROSS ALL CELLS\n{"=" * 96}')
    print(f'    {"beta":>6} {"%over":>7} {"off-target":>11} {"too hot":>9} '
          f'{"too cold":>9} {"longest":>9} {"div/run":>8}')
    for b in BETAS:
        rs = [r for r in rows if r['beta'] == b]
        if not rs:
            continue
        n = sum(r['n'] for r in rs)
        print(f'    {b:>6.2f} '
              f'{sum(r["over"] * r["n"] for r in rs) / n:>6.0%} '
              f'{np.nanmean([r["off"] for r in rs]):>11.1%} '
              f'{np.nanmean([r["hot"] for r in rs]):>9.1%} '
              f'{np.nanmean([r["cold"] for r in rs]):>9.1%} '
              f'{np.nanmean([r["longest"] for r in rs]):>9.1%} '
              f'{np.mean([r["div"] for r in rs]):>8.2f}')
    print('\n  off-target = fraction of STEPS outside a 2x band around the '
          'reference rate.\n  longest = the longest unbroken excursion, as a '
          'fraction of the run.')
    return rows


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10,
         int(sys.argv[2]) if len(sys.argv) > 2 else None)
