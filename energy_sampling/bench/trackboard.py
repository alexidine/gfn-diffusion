"""
THE CONTROLLERS ON THE TRACKING SURFACE.

One question, which is the reason this surface exists: WHEN THE RIGHT ANSWER
MOVES, DOES THE CONTROLLER MOVE WITH IT? Everything else here is supporting
detail.

`bench.ladder` measured where the optimum sits for each (optimizer, target
speed), so the true move is known:

    adam:  target 1e-4 -> best 1e-4 | 1e-3 -> 1e-3 | 1e-2 -> 0.03   (300x)
    sgd:   target 1e-4 -> best 0.01 | 1e-3 -> 0.1  | 1e-2 -> 0.3    (30x)

Every arm starts COLD at the same seed rate in every cell, so the rate an arm
ends at is a statement about adaptation and not about where it was placed. An arm
whose final rate barely moves across the speeds is not controlling -- it settled
somewhere and stayed, which on a single cell is indistinguishable from skill and
is exactly the ambiguity that made the previous board's verdicts unusable.

TWO NUMBERS PER ARM:
  nats      how much worse than the best arm in that cell (0 = best)
  tracked   final_lr(fastest target) / final_lr(slowest). Compare against the
            TRUE move in the same column. ~1 means it did not track at all.

    python -m bench.trackboard 5
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.arms import Fixed, HyperStep, Null, RampPlateau, RayRay
from bench.metrics import score_run
from bench.runner import Run
from bench.surfaces import TrackingGame

SEED_LR = 1e-4
STEPS = 6000
SPEEDS = (1e-4, 1e-3, 1e-2)
OPTS = ('adam', 'sgd')
#: spans both optimizers' optima, which sit ~100x apart on the same cell
LADDER = (1e-4, 1e-3, 1e-2, 1e-1, 3e-1)


def build_arms():
    return ([HyperStep(SEED_LR, beta=0.02), HyperStep(SEED_LR, beta=0.2),
             RayRay(SEED_LR, period=100), RayRay(SEED_LR, period=20),
             RampPlateau(SEED_LR), Null(SEED_LR)]
            + [Fixed(x) for x in LADDER])


def _one(item):
    opt, speed, ai, seed = item
    arm = build_arms()[ai]
    lr = arm.lr if isinstance(arm, Fixed) else SEED_LR
    g = TrackingGame(lr=lr, speed=speed, seed=seed, optimizer=opt)
    r = Run(g, arm, seed=seed, steps=STEPS, batch=64).run()
    row = score_run(r)
    row['aborted_run'] = bool(r.aborted)
    return opt, speed, row


def _init():
    import torch
    torch.set_num_threads(1)


def main(seeds=5, workers=None):
    names = [a.name for a in build_arms()]
    workers = int(workers or max(2, min(16, (os.cpu_count() or 4) - 4)))
    jobs = [(o, sp, ai, s) for o in OPTS for sp in SPEEDS
            for ai in range(len(names)) for s in range(int(seeds))]
    print(f'{"=" * 96}\nCONTROLLERS ON THE TRACKING SURFACE -- '
          f'{len(OPTS)}x{len(SPEEDS)} cells x {len(names)} arms x {seeds} seeds, '
          f'{STEPS} steps\nall arms start cold at {SEED_LR:g}\n{"=" * 96}')
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        out = list(pool.map(_one, jobs))

    per = {}
    for opt, sp, row in out:
        per.setdefault((opt, sp), {}).setdefault(row['arm'], []).append(row)

    lrs = {}
    for opt in OPTS:
        for sp in SPEEDS:
            rows = per[(opt, sp)]
            fin = {}
            for n in names:
                rs = rows[n]
                fin[n] = (math.inf if any(r['aborted_run'] for r in rs)
                          else float(np.median([r['final_loss'] for r in rs])))
                lrs[(opt, sp, n)] = float(np.nanmedian(
                    [r['final_lr'] for r in rs]))
            best = min(v for v in fin.values() if math.isfinite(v))
            print(f'\n  {opt}, target speed {sp:g}')
            print(f'    {"arm":<20} {"nats":>7} {"final lr":>10}')
            for n in sorted(names, key=lambda k: fin[k]):
                s = (f'{math.log(fin[n] / best):>7.2f}'
                     if math.isfinite(fin[n]) else f'{"died":>7}')
                print(f'    {n:<20} {s} {lrs[(opt, sp, n)]:>10.4g}')

    # ---- THE HEADLINE: did the chosen rate follow the optimum?
    print(f'\n{"=" * 96}\nDID THE RATE FOLLOW THE OPTIMUM?\n{"=" * 96}')
    print('  final_lr(target 1e-2) / final_lr(target 1e-4). ~1 = did not track.')
    for opt in OPTS:
        ref = {}
        for sp in SPEEDS:
            rows = per[(opt, sp)]
            f = {n: float(np.median([r['final_loss'] for r in rows[n]]))
                 for n in names if n.startswith('fixed@')}
            ref[sp] = min(f, key=f.get)
        true_move = (float(ref[SPEEDS[-1]].split('@')[1])
                     / float(ref[SPEEDS[0]].split('@')[1]))
        print(f'\n  --- {opt} ---   best FIXED rate moved '
              f'{ref[SPEEDS[0]]} -> {ref[SPEEDS[-1]]}  ({true_move:.0f}x)')
        print(f'    {"arm":<20} ' + ' '.join(f'{f"lr@{sp:g}":>10}' for sp in SPEEDS)
              + f' {"tracked":>9}')
        for n in names:
            if n.startswith('fixed@'):
                continue
            a, b = lrs[(opt, SPEEDS[0], n)], lrs[(opt, SPEEDS[-1], n)]
            r = b / a if a > 0 else float('nan')
            print(f'    {n:<20} '
                  + ' '.join(f'{lrs[(opt, sp, n)]:>10.4g}' for sp in SPEEDS)
                  + f' {r:>8.1f}x')
        print(f'    {"(true move)":<20} ' + ' ' * 33 + f' {true_move:>8.0f}x')
    return per


if __name__ == '__main__':
    main(next((int(a) for a in sys.argv[1:] if a.isdigit()), 5))
