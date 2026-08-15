"""
HYPER'S HYPERPARAMETERS: which setting is robust across a wide range of problems?

The goal is a setting you turn on once and stop thinking about, so the figure of
merit is WORST CASE, never a mean. Two questions, and they want different knobs:

  "at worst ~2x"   wants the right setpoint and enough gain to reach it
  "never 50x"      wants the response DOWNWARD to be faster than upward, so an
                   excursion is corrected before it compounds

`beta` is the climb gain; `beta_down` is the gain when the statistic says too
hot. Symmetric arms have beta_down = beta.

Eight cells: both optimizers x three target speeds, plus two that vary the NOISE
at fixed speed -- because the slow-target cell is where both sensors fail, and
the reason is that the noise-ball side of the tradeoff starts to dominate. If a
setting is only robust when noise is held constant, that is worth knowing before
it is recommended.

Every arm starts COLD at the same rate in every cell, so what is measured is
adaptation, not placement.

!! THE WORST-CELL COLUMN -- THIS FILE'S ENTIRE FIGURE OF MERIT -- IS NOT SCALE
   FREE, so the beta it recommends is partly a statement about STEPS. !!

The worst cell is set by the four MLE cells, and MLEGame's optimum is EXACTLY
zero: as a run converges the denominator of `log(final/best)` collapses toward 0
and the ratio grows without bound. So the gap those cells report keeps changing
with the horizon rather than settling, and the recommended beta -- and the
verdict on `beta_down` -- move with `STEPS=6000`. The tracking cells do not have
this problem (stationary surface, nonzero floor).

Two ways out, neither applied yet because either changes the published numbers:
give the MLE cells a `floor=` so the ratio has a scale (see `_Game.score_floor`),
or take the worst case over the tracking cells alone. Until then read the
per-cell columns, not the worst-case summary.

    python -m bench.hypersweep 5
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.arms import Fixed, HyperStep
from bench.metrics import score_run
from bench.runner import Run
from bench.surfaces import MLEGame, TrackingGame

SEED_LR = 1e-4
STEPS = 6000
LADDER = (1e-4, 1e-3, 1e-2, 1e-1, 3e-1)

#: (label, optimizer, kwargs). `game` picks the surface -- MLE cells are here
#: because a beta that is only robust WITHIN one surface family is not robust.
#: The MLE family is a CONVERGING problem (its optimum decays along the path) and
#: tracking is a STATIONARY one, so if they disagree about beta that is the most
#: important thing this sweep can report.
CELLS = [('adam slow',   'adam', dict(speed=1e-4)),
         ('adam med',    'adam', dict(speed=1e-3)),
         ('adam fast',   'adam', dict(speed=1e-2)),
         ('sgd slow',    'sgd',  dict(speed=1e-4)),
         ('sgd med',     'sgd',  dict(speed=1e-3)),
         ('sgd fast',    'sgd',  dict(speed=1e-2)),
         ('adam quiet',  'adam', dict(speed=1e-3, noise=0.01)),
         ('adam loud',   'adam', dict(speed=1e-3, noise=1.0)),
         ('mle base',    'adam', dict(game='mle', cond=300.0, noise=0.5)),
         ('mle noisy',   'adam', dict(game='mle', cond=300.0, noise=5.0)),
         ('mle illcond', 'adam', dict(game='mle', cond=3000.0, noise=0.5)),
         ('mle sgd',     'sgd',  dict(game='mle', cond=300.0, noise=0.5))]

#: (beta, beta_down). None = symmetric.
SETTINGS = [(0.005, None), (0.01, None), (0.02, None), (0.05, None),
            (0.1, None), (0.2, None), (0.4, None),
            (0.02, 0.1), (0.02, 0.4), (0.05, 0.2), (0.05, 0.4), (0.1, 0.4)]


def build_arms():
    return ([HyperStep(SEED_LR, beta=b, beta_down=d) for b, d in SETTINGS]
            + [Fixed(x) for x in LADDER])


def _one(item):
    ci, ai, seed = item
    _, opt, kw = CELLS[ci]
    arm = build_arms()[ai]
    lr = arm.lr if isinstance(arm, Fixed) else SEED_LR
    kw = dict(kw)
    if kw.pop('game', None) == 'mle':
        g = MLEGame(lr=lr, seed=seed, optimizer=opt, **kw)
    else:
        g = TrackingGame(lr=lr, seed=seed, optimizer=opt, **kw)
    r = Run(g, arm, seed=seed, steps=STEPS, batch=64).run()
    row = score_run(r)
    row['aborted_run'] = bool(r.aborted)
    return ci, row


def _init():
    import torch
    torch.set_num_threads(1)


def main(seeds=5, workers=None):
    names = [a.name for a in build_arms()]
    fixed = [n for n in names if n.startswith('fixed@')]
    workers = int(workers or max(2, min(16, (os.cpu_count() or 4) - 4)))
    jobs = [(ci, ai, s) for ci in range(len(CELLS))
            for ai in range(len(names)) for s in range(int(seeds))]
    print(f'{"=" * 104}\nHYPER HYPERPARAMETER SWEEP -- {len(CELLS)} cells x '
          f'{len(names)} arms x {seeds} seeds, {STEPS} steps\n{"=" * 104}')
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        out = list(pool.map(_one, jobs))

    per = {}
    for ci, row in out:
        per.setdefault(ci, {}).setdefault(row['arm'], []).append(row)

    med, ref = {}, {}
    for ci in range(len(CELLS)):
        rows = per[ci]
        for n in names:
            rs = rows[n]
            med[(ci, n)] = (math.inf if any(r['aborted_run'] for r in rs)
                            else float(np.median([r['final_loss'] for r in rs])))
        #: THE BAR IS THE BEST FIXED RATE IN THAT CELL, chosen with hindsight --
        #: the hardest available comparison, and the one the goal is phrased
        #: against.
        ref[ci] = min(med[(ci, f)] for f in fixed if math.isfinite(med[(ci, f)]))

    def rel(ci, n):
        v = med[(ci, n)]
        return math.inf if not math.isfinite(v) else math.log(v / ref[ci])

    print(f'\n  nats behind the best FIXED rate in each cell (0 = matched it)\n')
    print(f'  {"beta / down":<16} ' + ' '.join(f'{c[:9]:>9}' for c, _, _ in CELLS)
          + f' {"WORST":>7} {"as x":>7}')
    rows = []
    for n in names:
        if n.startswith('fixed@'):
            continue
        v = [rel(ci, n) for ci in range(len(CELLS))]
        rows.append((max(v), n, v))
    for w, n, v in sorted(rows):
        print(f'  {n.replace("hyper ", "").replace(" step", ""):<16} '
              + ' '.join((f'{x:>9.2f}' if math.isfinite(x) else f'{"died":>9}')
                         for x in v)
              + f' {w:>7.2f} {math.exp(w):>6.1f}x')

    print(f'\n  and the constants, same yardstick:')
    for f in fixed:
        v = [rel(ci, f) for ci in range(len(CELLS))]
        w = max(v)
        print(f'  {f:<16} ' + ' '.join((f'{x:>9.2f}' if math.isfinite(x)
                                        else f'{"died":>9}') for x in v)
              + f' {w:>7.2f} {math.exp(w):>6.1f}x')
    print('\n  WORST is the number the goal is stated against. A mean would hide '
          'exactly the\n  cell that makes a setting unusable.')
    return per


if __name__ == '__main__':
    main(next((int(a) for a in sys.argv[1:] if a.isdigit()), 5))
