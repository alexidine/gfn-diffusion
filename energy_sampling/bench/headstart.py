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

!! READ THIS BEFORE RANKING ANYTHING IN THIS FILE !!

THE RAY-VS-RAY AND RAY-VS-HYPER GAPS HERE ARE INSIDE THE SEED NOISE, at 5 seeds,
and it is not close. A probing arm draws through `run.game.draw`, so it consumes
extra randomness the non-probing arms never see and the paired-seed design does
NOT hold across that boundary (runner.py says so; this file compares across it
anyway, three times).

Measured directly -- same cell, same control law, arms differing ONLY in how much
RNG their probe consumes -- the spread from the draw alone is 1.28 nats cold and
1.21 warm. Against that null:

    largest ray-vs-ray gap    0.49 cold / 0.34 warm
    ray-vs-hyper gaps         0.95-1.44 cold / 0.49-0.83 warm

Every one of them fits inside. The null even reproduces a cadence ORDERING (p=20
best cold, p=100 best warm) out of nothing but the draw, so "increasing ray's
cadence does not close the head start" is not supported by this cell at 5 seeds
-- neither the claim nor its ordering.

This is specific to comparisons across the probing boundary. `board.py`'s gaps
(4.87 / 1.04 / 0.92 nats against a 0.169 per-seed sd) are unaffected. Fixing it
needs many more seeds, or arms matched on probe cost -- not a rerun.
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


def arms(start, warm=False):
    """
    THE WARM START HAS TO TURN THE WARMUP ENVELOPE OFF, and it did not.

    Starting an arm AT 3e-3 does not start it at 3e-3: the servo's envelope
    ramps from `1/lr_warmup_ratio` (10x down) over `warmup_steps`, so every
    controller began at 3.0e-4 and spent the first 1000 steps climbing the same
    deterministic curve. Measured: all five servo arms read lr@0 = 3.0000e-04,
    lr@100 = 3.7768e-04, lr@500 = 9.4868e-04, lr@999 = 2.9317e-03 -- BIT-
    IDENTICAL across five different controllers, because none of it is the
    controller. `reach` was therefore 700 for all five by construction, which is
    the envelope crossing the band and nothing else; with the envelope off it
    goes to 0 for four of them.

    So the file's premise ("both begin AT the best fixed rate. No ramp to win.
    Whatever remains is TRACKING") only holds with `lr_warmup_ratio = 1`.
    """
    a = [HyperStep(start, beta=0.2), HyperStep(start, beta=0.02),
         RayRay(start, period=100), RayRay(start, period=20),
         RayRay(start, period=10), Fixed(3e-3)]
    if warm:
        for arm in a:
            base = arm.args_overrides
            arm.args_overrides = (lambda b=base: {**b(), 'lr_warmup_ratio': 1})
    return a


def label(a, i):
    return {2: 'ray p=100', 3: 'ray p=20', 4: 'ray p=10'}.get(i, a.name)


def _one(item):
    which, ai, seed = item
    start = COLD if which == 'cold' else WARM
    a = arms(start, warm=(which != 'cold'))[ai]
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
