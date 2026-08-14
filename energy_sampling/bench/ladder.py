"""
THE STATIC-RATE REFERENCE for `TrackingGame`. Run this before any controller.

Fixed rates are the arms on this bench -- there is no oracle -- so this table IS
the thing every controller is measured against. It also states, per cell, how
much of the ladder is worth reporting: a rate difference smaller than the seed
noise is not a result.

NO CLOSED FORM IS QUOTED HERE. The obvious guess, lr* ~ sqrt(speed*noise) from
balancing lag against jitter, does NOT predict the measured optima -- it is 32x
off for Adam at the slowest target and 10x off for SGD at the fastest, because
Adam normalises per coordinate and the lag/jitter balance is not the naive one.
The empirical table is the reference; a formula that does not reproduce it would
just be an oracle with the same defect the old bench was rebuilt to remove.

    python -m bench.ladder
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.arms import Fixed
from bench.metrics import score_run
from bench.runner import Run
from bench.surfaces import TrackingGame

#: WIDE ENOUGH TO BRACKET EVERY ROW. At (1e-4 .. 3e-1) two rows put
#: their optimum on a ladder edge -- adam at the bottom, sgd at the top --
#: and an edge optimum is the ladder reporting its own limit, not the
#: surface's answer. Adam and SGD want rates ~100x apart on the same cell,
#: so one ladder has to span both.
LADDER = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
SPEEDS = (1e-4, 1e-3, 1e-2)
STEPS = 6000
SEEDS = 5


def _one(item):
    speed, lr, seed, opt = item
    g = TrackingGame(lr=lr, speed=speed, seed=seed, optimizer=opt)
    r = Run(g, Fixed(lr), seed=seed, steps=STEPS, batch=64).run()
    return speed, opt, lr, score_run(r)['final_loss']


def _init():
    import torch
    torch.set_num_threads(1)


def main(seeds=SEEDS, workers=None):
    workers = int(workers or max(2, min(16, (os.cpu_count() or 4) - 4)))
    jobs = [(sp, lr, s, opt) for opt in ('adam', 'sgd') for sp in SPEEDS
            for lr in LADDER for s in range(seeds)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        out = list(pool.map(_one, jobs))

    by = {}
    for sp, opt, lr, v in out:
        by.setdefault((opt, sp), {}).setdefault(lr, []).append(v)

    print(f'{"=" * 92}\nSTATIC RATE LADDER -- TrackingGame, {STEPS} steps, '
          f'{seeds} seeds\nnats above the best rate for that row (0 = best)'
          f'\n{"=" * 92}')
    for opt in ('adam', 'sgd'):
        print(f'\n  --- {opt} ---')
        print(f'  {"target":>8} ' + ' '.join(f'{lr:>7g}' for lr in LADDER)
              + f' {"best":>8} {"seed sd":>8}')
        for sp in SPEEDS:
            d = by[(opt, sp)]
            med = {lr: float(np.median(v)) for lr, v in d.items()}
            best = min(med.values())
            nats = {lr: math.log(med[lr] / best) for lr in LADDER}
            blr = min(nats, key=nats.get)
            sd = float(np.std(np.log(d[blr])))
            print(f'  {sp:>8g} ' + ' '.join(f'{nats[lr]:>7.2f}' for lr in LADDER)
                  + f' {blr:>8g} {sd:>8.3f}')
        edges = [sp for sp in SPEEDS
                 if min({lr: float(np.median(by[(opt, sp)][lr]))
                         for lr in LADDER}.items(), key=lambda kv: kv[1])[0]
                 in (LADDER[0], LADDER[-1])]
        print(f'    optimum on a ladder EDGE (not bracketed): '
              f'{edges if edges else "none"}')
    return by


if __name__ == '__main__':
    main(next((int(a) for a in sys.argv[1:] if a.isdigit()), SEEDS))
