"""
THE BATTERY: hypergradient with asymmetric gain, against both incumbents, on
every hardness axis.

Arms are (label, climber, braker, standard-overrides, arg-overrides). The three
`hyper` rows differ ONLY in beta_down/beta_up, and the 1:1 row is the published
rule exactly -- so the asymmetry's effect is read off one column, not inferred.

The two `ray` variants are the two changes today's mechanism actually argues for,
rather than a sweep:

  n_sub 16   the probe survives noise because its significance test makes it
             ABSTAIN and fall back to the constant ramp. More paired sub-batches
             = a sharper test = abstention that tracks the noise instead of
             lagging it.
  eta_down 1.0  the same cost asymmetry the hyper rows test, applied to the
             probe's own actuator (shipped at 0.25/0.5, i.e. 2:1; this makes it
             4:1). Testing one principle on two mechanisms is worth more than
             testing two principles once each.

A CELL GUARD. `find_oracle` accepts a cell if a best LR EXISTS, which is not the
same as the run converging: at cond=3000 the oracle itself improves only 9x over
2000 steps and its distance trace is 5x flatter than baseline, so every arm fails
and the cell reports a controller problem that is really a run-length problem.
Cells whose oracle does not achieve MIN_DROP are reported and skipped.

PARALLELISM. Runs are independent, so this fans out over processes. Each worker
pins torch to ONE thread: these tensors are 32-dimensional and torch's intra-op
threading is pure overhead at that size, so the default would have every worker
fighting for cores to do nothing. Two phases -- oracles per cell, then (cell,
arm) -- because every arm in a cell needs that cell's oracle first.
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.oracle import Surface, find_oracle
from bench.robustness import BASE, BUDGET, SCENARIOS  # noqa
from bench.scenarios import SEED_LR, sc_blowup, steps_to_target

MIN_DROP = 100.0        # the oracle must actually converge for the cell to count

ARMS = (
    ('hyper 1:1', 'hyperx', 'none', {'hyper_beta_down': 0.02}, {}),
    ('hyper 2:1', 'hyperx', 'none', {'hyper_beta_down': 0.04}, {}),
    ('hyper 4:1', 'hyperx', 'none', {'hyper_beta_down': 0.08}, {}),
    ('ray+ray', 'ray', 'ray', None, {}),
    ('ray n_sub16', 'ray', 'ray', None, {'ray_calibration.n_sub': 16}),
    ('ray eta_dn1', 'ray', 'ray', None,
     {'adaptive_lr.calibration.eta_down': 1.0}),
    ('ramp+plateau', 'ramp', 'plateau', None, {}),
)

CELLS = [('baseline', dict(BASE))]
CELLS += [(f'noise={v:g}', dict(BASE, noise=v)) for v in (0.1, 0.5, 2.0)]
CELLS += [(f'cond={v:g}', dict(BASE, cond=v)) for v in (30.0, 3000.0)]
CELLS += [(f'quartic={v:g}', dict(BASE, quartic=v)) for v in (1e-4, 1e-2)]


def _mk(kw, extra=None):
    return Surface('mle', 'mle', dict(kw), steps=2000, lr_grid=(1e-6, 1e-1, 12),
                   extra_args=dict(extra or {}))


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _oracle_task(item):
    """Phase 1: one cell's oracle, its convergence check and its own time."""
    label, kw = item
    surface = _mk(kw)
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
    """Phase 2: one (cell, arm) -- all scenarios x all seeds."""
    label, kw, oracle, denom, arm, seeds = item
    name, climber, braker, std, extra = arm
    s = _mk(kw, extra)
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


def main(seeds=15, workers=None):
    seeds = tuple(range(int(seeds)))
    workers = int(workers or max(2, min(20, (os.cpu_count() or 4) - 4)))
    print(f'{"=" * 80}\nBATTERY -- mle, {len(seeds)} seeds, budget {BUDGET:g}x, '
          f'{workers} workers\n{"=" * 80}')
    print('  %over = runs slower than the budget vs THIS cell\'s oracle '
          '(never-converged counts).\n  hyper 1:1 IS the published rule.\n')

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker) as pool:
        oracles = {}
        for label, got, why in pool.map(_oracle_task, CELLS):
            if got is None:
                print(f'  {label:<14} SKIPPED -- {why}')
            else:
                oracles[label] = got
        print()

        jobs = [(label, kw, *oracles[label][:2], arm, seeds)
                for label, kw in CELLS if label in oracles for arm in ARMS]
        rows = list(pool.map(_arm_task, jobs))

    agg = {}
    for label, kw in CELLS:
        if label not in oracles:
            continue
        oracle, denom, drop = oracles[label]
        print(f'  {label:<14} oracle lr {oracle.lr:.3g}  drop {drop:.3g}x  '
              f'target at {denom} steps')
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

    print(f'{"=" * 80}\nACROSS EVERY MEASURABLE CELL\n{"=" * 80}')
    print(f'  {"arm":<14} {"%over budget":>13} {"worst cell":>12}  where')
    for name, a in sorted(agg.items(), key=lambda kv: (kv[1]['over'] / max(kv[1]['n'], 1),
                                                       kv[1]['worst'])):
        print(f'  {name:<14} {a["over"] / max(a["n"], 1):>12.1%} '
              f'{a["worst"]:>11.0%}  {a["cell"]}')
    return rows


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15,
         int(sys.argv[2]) if len(sys.argv) > 2 else None)
