"""
HOW MUCH OF HYPER'S WIN IS THE HEAD START?

MK's observation: hyper reaches a useful rate ~900 steps before ray, and that
head start compounds into every later metric. Arithmetic backs it up -- ray's
climb is capped by cadence x per-firing move, (32/4)^0.25 = 1.682 every 100
steps = 0.0052 log-units/step, against hyper's beta = 0.2/step. A 38x
difference in ramp rate, before any question of signal quality.

Two ways to take the head start away, run together:

  WARM START   both begin AT the best fixed rate (3e-3) instead of cold. No ramp
               to win. Whatever remains is TRACKING.
  RAY FASTER   ray at period 25 and 10 instead of 100, i.e. 4x and 10x its
               cadence. If the gap closes, the advantage was cadence. Note this
               is NOT free -- ray pays ~n_sub extra loss evaluations per firing,
               so period 10 is roughly 10x the probe cost, which is the actual
               engineering trade rather than a defect in the comparison.
"""
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.arms import Fixed, HyperStep, RayRay
from bench.metrics import score_run
from bench.runner import Run
from bench.surfaces import MLEGame

COLD, WARM = 1.25e-4, 3e-3
STEPS, SEEDS = 8000, 5


def arms(start):
    return [HyperStep(start, beta=0.2), HyperStep(start, beta=0.02),
            RayRay(start, period=100), RayRay(start, period=20),
            RayRay(start, period=10), Fixed(3e-3)]


def label(a, i):
    return {2: 'ray p=100', 3: 'ray p=20', 4: 'ray p=10'}.get(i, a.name)


def _one(item):
    which, ai, seed = item
    start = COLD if which == 'cold' else WARM
    a = arms(start)[ai]
    game = MLEGame(dim=32, cond=300.0, noise=0.5, quartic=0.0, init_scale=3.0,
                   lr=start, optimizer='adam', seed=seed)
    run = Run(game, a, seed=seed, steps=STEPS, batch=64).run()
    r = score_run(run)
    # when did the rate first get within 2x of the best fixed rate, and stay?
    lo, hi = 3e-3 / 2, 3e-3 * 2
    ok = [h['lr'] is not None and lo <= h['lr'] <= hi for h in run.trace]
    reach = next((i for i in range(len(ok) - 200)
                  if all(ok[i:i + 200])), None)
    r['reach'] = reach if reach is not None else math.inf
    return which, ai, r


def _init():
    import torch
    torch.set_num_threads(1)


if __name__ == '__main__':
    jobs = [(w, ai, s) for w in ('cold', 'warm')
            for ai in range(len(arms(COLD))) for s in range(SEEDS)]
    with ProcessPoolExecutor(max_workers=min(14, (os.cpu_count() or 4) - 2),
                             initializer=_init) as pool:
        out = list(pool.map(_one, jobs))

    for which in ('cold', 'warm'):
        rows = {}
        for w, ai, r in out:
            if w == which:
                rows.setdefault(ai, []).append(r)
        best = min(float(np.median([r['final_loss'] for r in rs]))
                   for rs in rows.values())
        start = COLD if which == 'cold' else WARM
        print(f'\n  {which.upper()} START (lr {start:g})'
              + ('  -- no ramp to win; what remains is TRACKING'
                 if which == 'warm' else ''))
        print(f'    {"arm":<20} {"nats":>7} {"reach step":>11} {"final lr":>10}')
        tbl = []
        for ai, rs in rows.items():
            fin = float(np.median([r['final_loss'] for r in rs]))
            nats = math.log(fin / best) if fin > 0 and best > 0 else math.inf
            rc = np.median([r['reach'] for r in rs])
            tbl.append((nats, label(arms(start)[ai], ai), rc, rs))
        for nats, name, rc, rs in sorted(tbl):
            rcs = f'{rc:>11.0f}' if math.isfinite(rc) else f'{"never":>11}'
            print(f'    {name:<20} {nats:>7.2f} {rcs} '
                  f'{np.nanmedian([r["final_lr"] for r in rs]):>10.3g}')
