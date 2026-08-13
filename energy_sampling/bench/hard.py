"""
THE HARD SURFACES: `var_cond` (two-player, stale per-condition levels) and
`equilibration` (three players, anti-phase coupling).

Reduced arm set on purpose. On `mle` the three hyper asymmetry rows tied at 0%
over budget, so that surface cannot separate them and running all three here
would buy nothing; 1:1 (the published rule) and 2:1 (the house ratio, which
improved the MEDIAN on mle) are enough to keep the question open.

WHY THIS RUN MATTERS MORE THAN THE mle ONE. Every hypergradient result so far is
from a surface with ONE loss and ONE batch stream, where the standing objection
to the method -- that g_t and g_{t-1} are gradients of DIFFERENT objective
realizations, so their disagreement can mean "overshot" or merely "this batch
disagrees with the last one" -- cannot manifest. `equilibration` has explicit
branch weights and three players; `var_cond` has per-condition levels that go
stale. If hypergradient's clean sweep survives here it means something. If it
does not, that is the objection landing, and it should be reported as such.

The `w_rep` cell is a STATIC mixture change, which is the weak form of the
objection: production moves the mixture DURING a run via the balance controller.
Treat a clean result there as necessary, not sufficient.
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.oracle import Surface, find_oracle
from bench.robustness import BUDGET, SCENARIOS
from bench.scenarios import SEED_LR, sc_blowup, steps_to_target

MIN_DROP = 100.0

ARMS = (
    ('hyper 1:1', 'hyperx', 'none', {'hyper_beta_down': 0.02}, {}),
    ('hyper 2:1', 'hyperx', 'none', {'hyper_beta_down': 0.04}, {}),
    ('ray+ray', 'ray', 'ray', None, {}),
    ('ramp+plateau', 'ramp', 'plateau', None, {}),
)

VC = dict(dim=16, n_cond=256, spread=50.0, noise=0.5)
EQ = dict(dim=4, a=1.0, b=1.0, w_rep=0.7, w_bwd=0.3, kappa=0.05,
          noise=0.3, init_scale=1.0)

#: (label, game, kwargs, steps, lr_grid, extra_args)
CELLS = [
    ('vc baseline', 'var_cond', VC, 1500, (1e-4, 1e1, 12), {'batch_size': 32}),
    ('vc noise=2', 'var_cond', dict(VC, noise=2.0), 1500, (1e-4, 1e1, 12),
     {'batch_size': 32}),
    ('vc n_cond=64', 'var_cond', dict(VC, n_cond=64), 1500, (1e-4, 1e1, 12),
     {'batch_size': 32}),
    ('eq baseline', 'equilibration', EQ, 3000, (1e-4, 1e1, 12), {'lr_flow': 1.0}),
    ('eq noise=1', 'equilibration', dict(EQ, noise=1.0), 3000, (1e-4, 1e1, 12),
     {'lr_flow': 1.0}),
    # the weak form of the changing-mixture objection: a different but STATIC
    # branch weighting. w_rep 0.3 inverts which player dominates theta's update.
    ('eq w_rep=0.3', 'equilibration', dict(EQ, w_rep=0.3, w_bwd=0.7), 3000,
     (1e-4, 1e1, 12), {'lr_flow': 1.0}),
]


def _mk(cell, extra=None):
    _, game, kw, steps, grid, base_extra = cell
    return Surface(game, game, dict(kw), steps=steps, lr_grid=grid,
                   extra_args={**base_extra, **(extra or {})})


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _oracle_task(cell):
    label = cell[0]
    surface = _mk(cell)
    try:
        oracle = find_oracle(surface, seeds=(0, 1, 2), verbose=False)
    except ValueError as e:
        return label, None, f'no usable oracle ({e})'
    drop = float(oracle.trace[0]) / max(float(oracle.trace[-1]), 1e-300)
    if drop < MIN_DROP:
        return label, None, (f'oracle converges only {drop:.3g}x in '
                             f'{surface.steps} steps; trace too flat to time against')
    denom = steps_to_target(surface.run(oracle.lr, seed=0, servo=False), oracle)
    if not denom:
        return label, None, 'oracle misses its own target'
    return label, (oracle, denom, drop), None


def _arm_task(item):
    cell, oracle, denom, arm, seeds = item
    label = cell[0]
    name, climber, braker, std, extra = arm
    s = _mk(cell, extra)
    allslow, per = [], []
    for sc in SCENARIOS:
        slow = []
        for seed in seeds:
            if sc == 'blowup_100x':
                run = sc_blowup(s, oracle, seed, climber=climber,
                                braker=braker, standard=std)
            else:
                lr = SEED_LR if sc == 'cold_start' else oracle.hot_lr(0.9)
                run = s.run(lr, seed=seed, servo=True, climber=climber,
                            braker=braker, standard=std)
            t = steps_to_target(run, oracle)
            slow.append(math.inf if t is None else t / denom)
        per.append(float((np.array(slow) > BUDGET).mean()))
        allslow.extend(slow)
    arr = np.array(allslow, dtype=float)
    live = arr[np.isfinite(arr)]
    return dict(cell=label, arm=name, n=len(arr),
                over=float((arr > BUDGET).mean()),
                med=float(np.median(live)) if len(live) else math.inf,
                p90=float(np.percentile(live, 90)) if len(live) else math.inf,
                per=per)


def main(seeds=25, workers=None):
    seeds = tuple(range(int(seeds)))
    workers = int(workers or max(2, min(20, (os.cpu_count() or 4) - 4)))
    print(f'{"=" * 80}\nHARD SURFACES -- {len(seeds)} seeds, budget {BUDGET:g}x, '
          f'{workers} workers\n{"=" * 80}\n')

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        oracles = {}
        for label, got, why in pool.map(_oracle_task, CELLS):
            if got is None:
                print(f'  {label:<14} SKIPPED -- {why}')
            else:
                oracles[label] = got
        print()
        jobs = [(c, *oracles[c[0]][:2], arm, seeds)
                for c in CELLS if c[0] in oracles for arm in ARMS]
        rows = list(pool.map(_arm_task, jobs))

    agg = {}
    for c in CELLS:
        label = c[0]
        if label not in oracles:
            continue
        oracle, denom, drop = oracles[label]
        print(f'  {label:<14} oracle lr {oracle.lr:.3g}  drop {drop:.3g}x  '
              f'target at {denom} of {c[3]} steps')
        print(f'    {"arm":<14} {"%over":>7} {"med":>7} {"p90":>7}   '
              f'per-scenario %over {list(SCENARIOS)}')
        for r in [r for r in rows if r['cell'] == label]:
            a = agg.setdefault(r['arm'], {'over': 0.0, 'n': 0, 'worst': 0.0,
                                          'cell': '-'})
            a['over'] += r['over'] * r['n']
            a['n'] += r['n']
            if r['over'] > a['worst']:
                a['worst'], a['cell'] = r['over'], label
            med = f'{r["med"]:.2f}' if math.isfinite(r['med']) else 'never'
            print(f'    {r["arm"]:<14} {r["over"]:>6.0%} {med:>7} {r["p90"]:>7.2f}'
                  f'   {[f"{p:.0%}" for p in r["per"]]}')
        print()

    print(f'{"=" * 80}\nACROSS EVERY MEASURABLE HARD CELL\n{"=" * 80}')
    print(f'  {"arm":<14} {"%over budget":>13} {"worst cell":>12}  where')
    for name, a in sorted(agg.items(),
                          key=lambda kv: (kv[1]['over'] / max(kv[1]['n'], 1),
                                          kv[1]['worst'])):
        print(f'  {name:<14} {a["over"] / max(a["n"], 1):>12.1%} '
              f'{a["worst"]:>11.0%}  {a["cell"]}')
    return rows


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25,
         int(sys.argv[2]) if len(sys.argv) > 2 else None)
