"""
THE GUARANTEE BOARD: what is the WORST a controller does, anywhere.

This scores a different question from `scenarios.toolkit`. That board ranks arms
by how fast or how well they converge, which is the wrong objective for a
controller you intend to set once and stop thinking about. The objective here is
the one that actually matters for that:

    across every hardness setting, every perturbation, and every seed,
    what is the worst slowdown, and how often is the budget exceeded?

An arm with a better median and a fat tail LOSES to a duller arm with no tail.

WHY CENSORING STOPS BEING A PROBLEM. `steps_to_target` has no value for a run
that never converges, which made it a poor ranking metric -- the arms most worth
penalising simply vanished from the average. Under a worst-case budget a run that
never arrives is not a missing measurement, it is the largest possible violation,
and it scores as one. The defect and the objective cancel.

THE DENOMINATOR IS PER-CELL. Each hardness setting gets its own oracle, so a
slowdown always means "against the best fixed rate FOR THIS PROBLEM", never
against some other problem's. A cell where everything is slow is a hard cell, not
a controller failure, and this keeps the two apart.
"""
import math
import sys

import numpy as np

from bench.old.oracle import Surface, find_oracle
from bench.old.scenarios import (SEED_LR, SURFACES, sc_blowup, steps_to_target,
                             steps_behind)

#: The budget. Set from the operating requirement -- at worst a ~2x training
#: bill, never 50x -- not from anything measured here.
BUDGET = 2.0

#: HARDNESS AXES, swept ONE AT A TIME from the baseline rather than as a full
#: factorial. A full grid over three axes is 27 cells and most of them answer
#: nothing: what is wanted is which axis breaks which arm, and a one-at-a-time
#: sweep answers that at a third of the cost. Interactions are a later question
#: and this is the run that tells us whether any of them is worth the cell count.
BASE = dict(dim=32, cond=300.0, noise=0.01, init_scale=3.0, quartic=0.0)

AXES = {
    # gradient noise. The baseline sits at 0.01, which is FIFTY TIMES BELOW the
    # game's own default of 0.5 -- every MLE result to date was measured on an
    # almost noiseless surface, which is not a property the real problem has.
    'noise': (0.01, 0.1, 0.5, 2.0),
    # condition number, which sets the width of the usable band (cliff/oracle).
    # Narrower band = less room for a coarse actuator to sit in.
    'cond': (30.0, 300.0, 3000.0),
    # quartic term. THIS IS THE ONLY AXIS THAT MOVES THE TARGET: the effective
    # Hessian is H + 12*c*diag(theta^2), so curvature FALLS as the run converges
    # and the optimal rate RISES. A controller that records a ceiling once and
    # never releases it cannot follow that, which is the specific failure the
    # rest of the board cannot provoke.
    'quartic': (0.0, 1e-4, 1e-2),
}

ARMS = (('ray', 'ray'), ('ramp', 'plateau'), ('bb', 'none'),
        ('armijo', 'none'), ('hyper', 'none'), ('none', 'none'))

SCENARIOS = ('cold_start', 'blowup_100x', 'hot_90pct')


def cells():
    """Baseline plus one-at-a-time variations, deduplicated."""
    seen, out = set(), []
    for axis, values in AXES.items():
        for v in values:
            kw = dict(BASE)
            kw[axis] = v
            key = tuple(sorted(kw.items()))
            if key in seen:
                continue
            seen.add(key)
            label = 'baseline' if kw == BASE else f'{axis}={v:g}'
            out.append((label, kw))
    return out


def make_surface(kw, steps=2000):
    return Surface('mle', 'mle', dict(kw), steps=steps, lr_grid=(1e-6, 1e-1, 12))


def run_cell(label, kw, seeds, verbose=True):
    surface = make_surface(kw)
    try:
        oracle = find_oracle(surface, seeds=(0, 1, 2), verbose=False)
    except ValueError as e:
        # A cell whose oracle fails its own bracket test is a statement about the
        # CELL -- it is not LR-sensitive enough to score a controller against --
        # and reporting it beats quietly dropping it or scoring against a bad
        # reference.
        if verbose:
            print(f'\n  {label:<16} NO USABLE ORACLE: {e}')
        return None

    # the oracle's own time to target, as the denominator every slowdown uses
    oracle_run = surface.run(oracle.lr, seed=0, servo=False)
    denom = steps_to_target(oracle_run, oracle)
    if not denom:
        if verbose:
            print(f'\n  {label:<16} oracle never reaches its own target -- skipped')
        return None

    band = oracle.cliff / oracle.lr if oracle.cliff else float('nan')
    if verbose:
        print(f'\n  {label:<16} oracle lr {oracle.lr:.3g}  band {band:.2f}x  '
              f'oracle reaches target at {denom} steps')
        print(f'    {"arm":<16} {"worst":>8} {"p90":>8} {"median":>8} '
              f'{">budget":>8} {"never":>6}')

    rows = []
    for climber, braker in ARMS:
        slow = []
        for sc in SCENARIOS:
            for seed in seeds:
                if sc == 'blowup_100x':
                    run = sc_blowup(surface, oracle, seed,
                                    climber=climber, braker=braker)
                else:
                    lr = SEED_LR if sc == 'cold_start' else oracle.hot_lr(0.9)
                    run = surface.run(lr, seed=seed, servo=True,
                                      climber=climber, braker=braker)
                t = steps_to_target(run, oracle)
                # never arrived -> the largest violation, not a missing value
                slow.append(math.inf if t is None else t / denom)
        arr = np.array(slow, dtype=float)
        live = arr[np.isfinite(arr)]
        row = dict(
            cell=label, arm=f'{climber}+{braker}',
            worst=float(arr.max()),
            p90=float(np.percentile(live, 90)) if len(live) else math.inf,
            median=float(np.median(live)) if len(live) else math.inf,
            over=float((arr > BUDGET).mean()),
            never=int((~np.isfinite(arr)).sum()), n=len(arr))
        rows.append(row)
        if verbose:
            w = 'never' if not math.isfinite(row['worst']) else f'{row["worst"]:.2f}'
            print(f'    {row["arm"]:<16} {w:>8} {row["p90"]:>8.2f} '
                  f'{row["median"]:>8.2f} {row["over"]:>7.0%} {row["never"]:>6}')
    return rows


def main(seeds=20):
    seeds = tuple(range(int(seeds)))
    print(f'{"=" * 78}\nGUARANTEE BOARD -- mle, {len(seeds)} seeds, '
          f'budget {BUDGET:g}x\n{"=" * 78}')
    print('slowdown = steps to the oracle\'s mid-run distance, over the ORACLE\'S '
          'own\ntime to the same level, in THIS cell. never-converged counts as '
          'a violation.')

    allrows = []
    for label, kw in cells():
        r = run_cell(label, kw, seeds)
        if r:
            allrows.extend(r)

    print(f'\n{"=" * 78}\nWORST CASE ANYWHERE (the number the guarantee rests on)'
          f'\n{"=" * 78}')
    print(f'  {"arm":<16} {"worst":>8} {"over budget":>12} {"never":>7}  '
          f'worst cell')
    agg = {}
    for r in allrows:
        a = agg.setdefault(r['arm'], {'worst': 0.0, 'over': 0, 'n': 0,
                                      'never': 0, 'cell': ''})
        if r['worst'] > a['worst']:
            a['worst'], a['cell'] = r['worst'], r['cell']
        a['over'] += r['over'] * r['n']
        a['n'] += r['n']
        a['never'] += r['never']
    for arm, a in sorted(agg.items(), key=lambda kv: kv[1]['worst']):
        w = 'never' if not math.isfinite(a['worst']) else f'{a["worst"]:.2f}x'
        print(f'  {arm:<16} {w:>8} {a["over"] / max(a["n"], 1):>11.0%} '
              f'{a["never"]:>7}  {a["cell"]}')
    ok = [arm for arm, a in agg.items() if a['worst'] <= BUDGET]
    print(f'\n  within a {BUDGET:g}x budget EVERYWHERE: '
          f'{", ".join(ok) if ok else "NONE"}')
    return allrows


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
