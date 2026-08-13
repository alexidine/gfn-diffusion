"""
THE LEADERBOARD. Arms x seeds on one game, scored by `bench/metrics.py`.

    python -m bench.board            # adam, the default ladder
    python -m bench.board sgd        # the control

READ IT AS A LEADERBOARD, NOT A SCORE. There is no oracle, no budget and no
threshold: the fixed-rate arms ARE the reference, so "at worst ~2x the best fixed
rate" is read by comparing rows. Everything that went wrong in the previous
generation went wrong in the machinery that produced a single number -- a
selected reference rate, a censored ratio, a threshold, a feasibility predicate.
None of that exists here.

WHAT A ROW MEANS

    final       median loss over the last 100 steps, relative to the best row.
                1.00 is the winner; 2.30 is 2.3x the best final loss.
    lead        share of steps this arm's smoothed loss was the lowest of all
                arms on the SAME seed. Sums to ~1 down the column.
    lr sd       sd of log(lr). 0 for a fixed rate by construction.
    max jump    largest single-step move in log(lr). "wild swings".
    backslide   share of the run the loss was rising by more than its own noise
                explains.
    div / abort / nonfin
                catastrophe COUNTS, summed over seeds. Never averaged into
                anything -- the goal is a tail statement.

Seeds are paired: one seed is one noise stream, shared by every arm, so a
difference between rows is the arm and not the draw.
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.arms import Fixed, Hyper, Null, RampPlateau, RayRay
from bench.metrics import lead_fraction, score_run
from bench.runner import Run
from bench.surfaces import MLEGame

#: The rate every controller STARTS at -- deliberately cold, because MK's
#: requirement (1) is "all starts are relatively cold and expected to ramp".
SEED_LR = 1.25e-4

#: The fixed-rate ladder. Spans four decades so the best fixed rate is interior
#: to it: a ladder whose winner sits at an edge is telling you the ladder is
#: wrong, and that is visible here rather than hidden in a guard.
LADDER = (1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)

STEPS = 2000
BATCH = 64


def make_game(optimizer='adam', seed=0, **kw):
    """The simplest case: MLE-type. `cond` sets how ill-conditioned it is."""
    return MLEGame(dim=32, cond=300.0, noise=0.5, quartic=0.0,
                   init_scale=3.0, lr=SEED_LR, optimizer=optimizer,
                   seed=seed, **kw)


def build_arms():
    return [Hyper(SEED_LR), RayRay(SEED_LR), RampPlateau(SEED_LR),
            Null(SEED_LR)] + [Fixed(x) for x in LADDER]


def _one(item):
    """One (arm, seed). Returns the scored row plus the smoothed-loss trace the
    lead metric needs, so the pool sends back numbers rather than live objects."""
    arm_idx, seed, optimizer = item
    arm = build_arms()[arm_idx]
    game = make_game(optimizer=optimizer, seed=seed)
    # the arm's own rate is what the game was built with; override cleanly
    run = Run(game, arm, seed=seed, steps=STEPS, batch=BATCH)
    run.run()
    from bench.metrics import smoothed_loss
    return score_run(run), [x if x is not None and math.isfinite(x) else None
                            for x in smoothed_loss(run)]


def _init_worker():
    import torch
    torch.set_num_threads(1)


def main(seeds=8, optimizer='adam', workers=None):
    seeds = tuple(range(int(seeds)))
    arms = build_arms()
    workers = int(workers or max(2, min(16, (os.cpu_count() or 4) - 4)))
    print(f'{"=" * 104}\nLR BOARD -- MLE game, {optimizer}, {len(seeds)} seeds, '
          f'{len(arms)} arms, {STEPS} steps\n'
          f'seed lr {SEED_LR:g}; fixed ladder {LADDER}\n{"=" * 104}\n')

    jobs = [(i, s, optimizer) for i in range(len(arms)) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker) as pool:
        out = list(pool.map(_one, jobs))

    rows = {}
    smoothed = {}
    for (job, (row, sm)) in zip(jobs, out):
        rows.setdefault(row['arm'], []).append(row)
        smoothed.setdefault(job[1], {})[row['arm']] = sm

    # lead is computed PER SEED across arms, then averaged over seeds
    lead_acc = {a.name: [] for a in arms}
    for seed, by_arm in smoothed.items():
        got = _lead_from_smoothed(by_arm)
        for k, v in got.items():
            lead_acc[k].append(v)

    best_final = min(
        float(np.median([r['final_loss'] for r in rs]))
        for rs in rows.values()
        if any(math.isfinite(r['final_loss']) for r in rs))

    print(f'  {"arm":<18} {"final":>8} {"lead":>7} {"lr sd":>7} '
          f'{"max jump":>9} {"backslide":>10} {"div":>5} {"abort":>6} {"nonfin":>7}')
    table = []
    for a in arms:
        rs = rows.get(a.name, [])
        if not rs:
            continue
        fin = float(np.median([r['final_loss'] for r in rs]))
        table.append((fin, a.name, rs))
    for fin, name, rs in sorted(table):
        rel = fin / best_final if math.isfinite(fin) and best_final > 0 else math.inf
        lead = float(np.mean(lead_acc[name])) if lead_acc[name] else math.nan
        print(f'  {name:<18} {rel:>8.2f} {lead:>6.1%} '
              f'{np.nanmedian([r["lr_sd"] for r in rs]):>7.3f} '
              f'{np.nanmedian([r["lr_max_jump"] for r in rs]):>9.3f} '
              f'{np.nanmedian([r["backslide"] for r in rs]):>10.1%} '
              f'{sum(r["divergences"] for r in rs):>5} '
              f'{sum(bool(r["aborted"]) for r in rs):>6} '
              f'{sum(r["nonfinite_steps"] for r in rs):>7}')

    print(f'\n  final = median over seeds of the last-100-step median loss, '
          f'RELATIVE to the best arm.\n  lead  = share of steps with the lowest '
          f'smoothed loss, paired by seed.')
    return rows


def _lead_from_smoothed(by_arm):
    n = min(len(s) for s in by_arm.values())
    wins = {k: 0.0 for k in by_arm}
    counted = 0
    for i in range(n):
        vals = {k: s[i] for k, s in by_arm.items() if s[i] is not None}
        if not vals:
            continue
        counted += 1
        best = min(vals.values())
        leaders = [k for k, v in vals.items() if v <= best]
        for k in leaders:
            wins[k] += 1.0 / len(leaders)
    if not counted:
        return {k: float('nan') for k in by_arm}
    return {k: v / counted for k, v in wins.items()}


if __name__ == '__main__':
    opt = 'sgd' if 'sgd' in sys.argv else 'adam'
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8)
    main(n, opt)
