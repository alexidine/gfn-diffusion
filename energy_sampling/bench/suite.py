"""
THE SAME ARMS ACROSS HARDER SINGLE-PLAYER CELLS.

One cell told us `hyper step b=0.2` beats `ray+ray`. One cell is one cell. This
sweeps the two axes that should break a rate controller, before anything moves to
the multi-player game:

  NOISE      how much of the gradient is minibatch draw. Swept by the surface's
             own `noise`, and separately by shrinking the BATCH, which is the
             production knob and also raises the variance the sensor sees.
  SHIFTS     how much, and in which direction, the optimal rate moves during the
             run.

WHY A SHIFT UPWARD MATTERS. Every result so far is on a surface whose optimal
rate DECAYS (~30x, the noise ball forming). A controller that simply cools
monotonically looks perfect there and would be indistinguishable from a good one.
The `regime` cell softens the curvature 8x mid-run, so the optimal rate jumps
UP -- which a monotone cooler cannot follow, and which is the case MK's
requirement (2) actually describes.

    python -m bench.suite            # all cells, adam
    python -m bench.suite 5 sgd
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.arms import Fixed, HyperStep, Null, RampPlateau, RayRay
from bench.metrics import score_run
from bench.runner import Run
from bench.surfaces import MLEGame

SEED_LR = 1.25e-4
STEPS = 8000
LADDER = (1e-4, 1e-3, 3e-3, 1e-2, 1e-1)


def _mutate_soften(game):
    """Curvature /8 -> the optimal rate jumps ~8x UP, mid-run."""
    game.H = game.H / 8.0


#: (label, game kwargs, batch, mutate_fn or None)
CELLS = [
    ('base',        dict(cond=300.0, noise=0.5, quartic=0.0),  64, None),
    ('noise x10',   dict(cond=300.0, noise=5.0, quartic=0.0),  64, None),
    ('noise x40',   dict(cond=300.0, noise=20.0, quartic=0.0), 64, None),
    ('batch 8',     dict(cond=300.0, noise=0.5, quartic=0.0),   8, None),
    ('quartic .1',  dict(cond=300.0, noise=0.1, quartic=0.1),  64, None),
    ('illcond 3k',  dict(cond=3000.0, noise=0.5, quartic=0.0), 64, None),
    ('regime /8',   dict(cond=300.0, noise=0.5, quartic=0.0),  64, _mutate_soften),
]


def build_arms():
    return ([HyperStep(SEED_LR, beta=0.02), HyperStep(SEED_LR, beta=0.2),
             RayRay(SEED_LR), RampPlateau(SEED_LR), Null(SEED_LR)]
            + [Fixed(x) for x in LADDER])


def _one(item):
    ci, ai, seed, optimizer = item
    label, kw, batch, mutate = CELLS[ci]
    arm = build_arms()[ai]
    game = MLEGame(dim=32, init_scale=3.0, lr=SEED_LR, optimizer=optimizer,
                   seed=seed, **kw)
    run = Run(game, arm, seed=seed, steps=STEPS, batch=batch)
    at = STEPS // 2
    for i in range(STEPS):
        if mutate is not None and i == at:
            mutate(game)
        run.step()
        if run.aborted:
            break
    return ci, score_run(run)


def _init():
    import torch
    torch.set_num_threads(1)


def main(seeds=5, optimizer='adam', workers=None):
    seeds = tuple(range(int(seeds)))
    arms = build_arms()
    workers = int(workers or max(2, min(16, (os.cpu_count() or 4) - 4)))
    print(f'{"=" * 100}\nSUITE -- {len(CELLS)} cells x {len(arms)} arms x '
          f'{len(seeds)} seeds, {STEPS} steps, {optimizer}\n{"=" * 100}')

    jobs = [(ci, ai, s, optimizer)
            for ci in range(len(CELLS)) for ai in range(len(arms))
            for s in seeds]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        out = list(pool.map(_one, jobs))

    per = {}
    for ci, row in out:
        per.setdefault(ci, {}).setdefault(row['arm'], []).append(row)

    totals = {a.name: [] for a in arms}
    for ci, (label, kw, batch, mutate) in enumerate(CELLS):
        rows = per.get(ci, {})
        if not rows:
            continue
        best = min(float(np.median([r['final_loss'] for r in rs]))
                   for rs in rows.values()
                   if any(math.isfinite(r['final_loss']) for r in rs))
        print(f'\n  {label}   (noise {kw["noise"]:g}, cond {kw["cond"]:g}, '
              f'quartic {kw["quartic"]:g}, batch {batch}'
              + (', curvature /8 at midpoint)' if mutate else ')'))
        print(f'    {"arm":<20} {"nats":>7} {"final lr":>10} {"backslide":>10} '
              f'{"div":>5}')
        tbl = []
        for a in arms:
            rs = rows.get(a.name, [])
            if not rs:
                continue
            fin = float(np.median([r['final_loss'] for r in rs]))
            nats = (math.log(fin / best) if math.isfinite(fin) and fin > 0
                    and best > 0 else math.inf)
            totals[a.name].append(nats)
            tbl.append((nats, a.name, rs))
        for nats, name, rs in sorted(tbl):
            s = f'{nats:>7.2f}' if math.isfinite(nats) else f'{"never":>7}'
            print(f'    {name:<20} {s} '
                  f'{np.nanmedian([r["final_lr"] for r in rs]):>10.3g} '
                  f'{np.nanmedian([r["backslide"] for r in rs]):>10.1%} '
                  f'{sum(r["divergences"] for r in rs):>5}')

    print(f'\n{"=" * 100}\nACROSS CELLS -- mean nats behind that cell\'s best, '
          f'and the WORST cell\n{"=" * 100}')
    print(f'  {"arm":<20} {"mean nats":>10} {"worst cell":>11}')
    for name, v in sorted(totals.items(),
                          key=lambda kv: np.mean([x for x in kv[1]
                                                  if math.isfinite(x)] or [1e9])):
        fin = [x for x in v if math.isfinite(x)]
        m = np.mean(fin) if fin else math.inf
        w = max(v) if v else math.inf
        ws = f'{w:>11.2f}' if math.isfinite(w) else f'{"never":>11}'
        print(f'  {name:<20} {m:>10.2f} {ws}')
    return per


if __name__ == '__main__':
    opt = 'sgd' if 'sgd' in sys.argv else 'adam'
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 5)
    main(n, opt)
