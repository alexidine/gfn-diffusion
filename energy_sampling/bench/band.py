"""
THE GOOD BAND: for each cell, which learning rates actually work.

Every reference this bench has used so far is derived -- a closed-form cliff, a
ratio to the best arm, a nats gap. All of them need a paragraph before a number
means anything. This measures the thing directly instead: sweep constant rates
on a cell and record which ones land within 2x of the best outcome that cell
allows. That is a band of learning rates, in learning-rate units, and a
controller either sits in it or does not.

It is deliberately NOT the cliff. The cliff is where the system goes unstable;
the band is where it does well, and the battery's main negative result is that
those are different -- riding the cliff scores badly. The band needs no theory
and applies equally to the Adam cell, where no closed form exists.

    python -m bench.band 3
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.arms import Fixed
from bench.eqsuite import BASE, CELLS, STEPS, _spec
from bench.runner import Run
from bench.surfaces import EquilibrationGame

#: Wide enough to bracket every cell's optimum from below and above -- checked,
#: not assumed: a ladder that does not contain the answer reports its own edge.
RATES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)

#: "within 2x of the best rate" -- the user's own bar, in nats.
BAND_NATS = math.log(2.0)


def _one(item):
    ci, ri, seed = item
    over, flow, _, opt = _spec(ci)
    lr = RATES[ri]
    g = EquilibrationGame(lr=lr, optimizer=opt, seed=seed, **{**BASE, **over})
    run = Run(g, Fixed(lr), seed=seed, steps=STEPS, batch=64)
    if flow is not None:
        run.m.args.lr_flow = flow
    run.run()
    v = math.inf if run.aborted else g.expected_loss()
    return ci, ri, (v if math.isfinite(v) and v > 0 else math.inf)


def _init():
    import torch
    torch.set_num_threads(1)


def main(seeds=3, workers=None):
    seeds = tuple(range(int(seeds)))
    workers = int(workers or max(2, min(16, (os.cpu_count() or 4) - 4)))
    jobs = [(ci, ri, s) for ci in range(len(CELLS))
            for ri in range(len(RATES)) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        out = list(pool.map(_one, jobs))

    grid = {}
    for ci, ri, v in out:
        grid.setdefault(ci, {}).setdefault(ri, []).append(v)

    print(f'{"=" * 92}\nTHE GOOD BAND -- constant rates within '
          f'2x of the best outcome each cell allows\n{"=" * 92}')
    print(f'  {"cell":<16} {"band low":>10} {"band high":>10} {"best":>9} '
          f'{"width":>7}   rates 1e-5 .. 1e-1')
    bands = {}
    for ci, (label, _) in enumerate(CELLS):
        vals = [float(np.median(grid[ci][ri])) for ri in range(len(RATES))]
        best = min(vals)
        nats = [math.log(v / best) if math.isfinite(v) else math.inf
                for v in vals]
        inb = [RATES[i] for i, n in enumerate(nats) if n <= BAND_NATS]
        bands[label] = (min(inb) if inb else None, max(inb) if inb else None,
                        RATES[int(np.argmin(vals))], vals, nats)
        spark = ''.join('#' if n <= BAND_NATS else
                        ('+' if n <= 2.3 else ('.' if math.isfinite(n) else 'X'))
                        for n in nats)
        lo, hi = (min(inb), max(inb)) if inb else (None, None)
        w = f'{hi / lo:.0f}x' if inb else '-'
        print(f'  {label:<16} {lo if lo else "-":>10} {hi if hi else "-":>10} '
              f'{RATES[int(np.argmin(vals))]:>9g} {w:>7}   {spark}')
    print('\n  # within 2x of best   + within 10x   . worse   X did not finish')
    print('  A band touching an END of the ladder is not bracketed -- treat it '
          'as a bound.')
    return bands


if __name__ == '__main__':
    main(next((int(a) for a in sys.argv[1:] if a.isdigit()), 3))
