"""
DOES THE RANKING SURVIVE ADAM?

Every cell in every bench battery takes `optimizer='sgd'` (the default at
`surfaces.py:165,301,433`); production runs `torch.optim.Adam` on every optimizer
(`train.py:1647,1659,1664`). No cell in `crucible.CELLS`, `HELDOUT`, `EQ_HARD`,
`beta_ladder.CELLS` or `requirements.CELLS` overrides it.

This is not a realism quibble, it is the derivation. Hypergradient descent comes
from differentiating the SGD update:

    theta_t = theta_{t-1} - lr * g_{t-1}      =>   d(theta_t)/d(lr) = -g_{t-1}
    dL/d(lr) = <grad L(theta_t), d(theta_t)/d(lr)> = -<g_t, g_{t-1}>

Under Adam the step direction is `mhat / (sqrt(vhat) + eps)`, not `g`, so the
correct hypergradient is `-<g_t, mhat/(sqrt(vhat)+eps)>`. Baydin et al. derive
Adam-HD with exactly that corrected direction; the arm measured here uses the SGD
form. Adam's preconditioner is also strongly anisotropic and slowly varying,
which is precisely the regime where the two directions disagree.

So: same arms, same cells, same seeds, `sgd` vs `adam`. If the ranking is stable
the SGD-only measurement was harmless. If it inverts, every battery in this
directory was run on the wrong dynamics.
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.old.crucible import (ARMS, BUDGET, DEEP_FRAC, MLE, SCENARIOS, _mk,
                            _oracle_task, _sc_drift, _sc_mixture, _sc_regime)
from bench.old.scenarios import (SEED_LR, longest_off_target, steps_to_target,
                             time_off_target)

BASE = [
    ('mle base', dict(MLE)),
    ('mle n0.5', dict(MLE, noise=0.5)),
    ('mle q0.1', dict(MLE, quartic=1e-1)),
]


def cells(opt):
    return [(f'{label} [{opt}]', 'mle', dict(kw, optimizer=opt), 2000,
             (1e-6, 1e-1, 12), {}) for label, kw in BASE]


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _arm(item):
    cell, oracle, denom, reg, arm, seeds = item
    name, climber, braker, std, extra = arm
    s = _mk(cell, extra)
    slow, off, longest = [], [], []
    for sc in SCENARIOS:
        for seed in seeds:
            if sc == 'drift_10x':
                run = _sc_drift(s, oracle, seed, climber, braker, std)
            elif sc == 'regime_change':
                run = _sc_regime(s, oracle, seed, climber, braker, std)
            elif sc == 'mixture_drift':
                run = _sc_mixture(s, oracle, seed, climber, braker, std)
            elif sc == 'cold_start':
                run = s.run(SEED_LR, seed=seed, servo=True, climber=climber,
                            braker=braker, standard=std)
            else:
                run = s.run(oracle.hot_lr(0.9), seed=seed, servo=True,
                            climber=climber, braker=braker, standard=std)
            ref, ref_denom = ((reg[0], reg[1]) if sc == 'regime_change' and reg
                              else (oracle, denom))
            t = steps_to_target(run, ref, frac=DEEP_FRAC)
            slow.append(math.inf if t is None else t / ref_denom)
            o = time_off_target(run, ref.lr)
            if o is not None:
                off.append(o['off'])
                lo = longest_off_target(run, ref.lr)
                if lo is not None:
                    longest.append(lo)
    arr = np.array(slow, dtype=float)
    return dict(cell=cell[0], arm=name, n=len(arr),
                over=float((arr > BUDGET).mean()),
                off=float(np.mean(off)) if off else math.nan,
                longest=float(np.mean(longest)) if longest else math.nan)


def main(seeds=6, workers=None):
    seeds = tuple(range(int(seeds)))
    workers = int(workers or max(2, min(20, (os.cpu_count() or 4) - 4)))
    allcells = cells('sgd') + cells('adam')
    print(f'{"=" * 92}\nSGD vs ADAM -- does the arm ranking survive the '
          f'optimizer production actually uses?\n{len(seeds)} seeds, '
          f'{workers} workers\n{"=" * 92}\n')

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker) as pool:
        oracles = {}
        for label, got, why in pool.map(_oracle_task, allcells):
            if got is None:
                print(f'  {label:<20} SKIPPED -- {why}')
            else:
                oracles[label] = got
        jobs = [(c, oracles[c[0]][0], oracles[c[0]][1], oracles[c[0]][3],
                 a, seeds)
                for c in allcells if c[0] in oracles for a in ARMS]
        rows = list(pool.map(_arm, jobs))

    print()
    for opt in ('sgd', 'adam'):
        sel = [r for r in rows if r['cell'].endswith(f'[{opt}]')]
        if not sel:
            continue
        print(f'  {opt.upper()}')
        print(f'    {"arm":<18} {"%over":>7} {"off-target":>11} {"longest":>9}')
        agg = {}
        for r in sel:
            a = agg.setdefault(r['arm'], {'o': 0.0, 'n': 0, 'f': [], 'l': []})
            a['o'] += r['over'] * r['n']
            a['n'] += r['n']
            a['f'].append(r['off'])
            a['l'].append(r['longest'])
        for name, a in sorted(agg.items(), key=lambda kv: kv[1]['o'] / max(kv[1]['n'], 1)):
            print(f'    {name:<18} {a["o"] / max(a["n"], 1):>6.0%} '
                  f'{np.nanmean(a["f"]):>11.1%} {np.nanmean(a["l"]):>9.1%}')
        print()
    return rows


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6,
         int(sys.argv[2]) if len(sys.argv) > 2 else None)
