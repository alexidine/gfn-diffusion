"""
DO THE BENCH'S GRADIENTS LOOK LIKE THE REAL SYSTEM'S?

`bench/calibrate_noise.py` records, on a real TB run, how much consecutive
gradients agree: `cos(g_t, g_{t-1})`. That number is only useful if the bench can
be read in the same units. This file measures the same statistic on the bench
surfaces and prints both next to each other.

REWRITTEN 2026-08-14. The previous version imported `MLE` and `_mk` from
`bench.crucible`, which the rebuild moved to `old/`, so the module had not
imported -- and therefore this check had not been runnable -- since 2026-08-13.
It also went through the old `Surface`/`find_oracle` machinery, which the rebuild
deliberately deleted; nothing here needs an oracle, so nothing here has one.

TWO NUMBERS, NOT ONE, and this is the point of the file. A median alone does not
locate the real system on this axis, because the same median means different
things at different widths. Two UNRELATED vectors in d dimensions already agree
by about

    chance = sqrt(2 / (pi * d))

At the bench's usual d=32 that is 0.141, so a median of 0.29 is twice chance and
a single reading is mostly luck. At the real policy's d = 6,163,969 chance is
0.00032, so the same 0.29 is ~900x chance and one reading is essentially exact.
Matching only the median would put the real system in a cell where its own
sensor is far noisier than it really is -- which flatters every arm that ignores
the signal and penalises every arm that measures it. So a cell matches on the
PAIR: median and width.

MEASURED AT A FIXED RATE, deliberately. The same cosine computed inside a live
servo is contaminated by the rate moving underneath it; the axis wants the
statistic at a settled operating point, computed the way `calibrate_noise.py`
computes it -- consecutive gradients of the branch that trains, before any
rescale.

REPORTED PER QUARTILE, because the statistic decays as a run converges and a
whole-run median pools regimes a real 400-step window does not.

    python -m bench.cos_axis            # bench cells against the real run
    python -m bench.cos_axis tune       # search dim x speed x noise for a match
"""
import json
import math
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from bench.surfaces import EquilibrationGame, MLEGame, TrackingGame
import bench.eqsuite as eqsuite

#: written by `bench/calibrate_noise.py`, at the repo root
REAL_JSON = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'bench_noise_calibration.json')


def chance(d):
    """|cos| between two independent vectors in d dimensions."""
    return math.sqrt(2.0 / (math.pi * d))


def _fmt(label, xs, d, extra=''):
    if len(xs) < 8:
        print(f'  {label:<34} (only {len(xs)} readings)')
        return None
    a = np.asarray(xs, dtype=float)
    c = chance(d)
    med = float(np.median(a))
    q1, q3 = float(np.percentile(a, 25)), float(np.percentile(a, 75))
    print(f'  {label:<34} d={d:>9,} chance={c:.5f}  med={med:>7.4f}  '
          f'[{q1:>7.4f},{q3:>7.4f}] w={q3 - q1:>6.3f}  x-chance={med / c:>7.1f}'
          f'  quartiles ' + ' '.join(
              f'{float(np.median(x)):>7.4f}' for x in np.array_split(a, 4)) + extra)
    return med, q1, q3


def real_reference():
    """Median/IQR per branch from the real run. Returns {} if never measured."""
    if not os.path.exists(REAL_JSON):
        print(f'  (no {os.path.basename(REAL_JSON)} -- run bench.calibrate_noise)')
        return {}
    recs = json.load(open(REAL_JSON))
    by = {}
    for r in recs:
        by.setdefault(r['step_type'], []).append(r)
    out = {}
    for k, rs in sorted(by.items()):
        gn = [r['gnorm'] for r in rs]
        sd = st.pstdev(gn) if len(gn) > 1 else 0.0
        # a gradient norm with no spread is a clip pinning every step, which
        # changes what the whole axis means -- say so rather than let it pass
        tag = (f'  |g| med={st.median(gn):.4g} sd={sd:.2e}'
               + ('   <- CLIP BINDING EVERY STEP' if sd < 1e-4 * max(st.median(gn), 1e-12)
                  else ''))
        got = _fmt(f'real {k} (n={len(rs)})', [r['cos'] for r in rs],
                   rs[0]['dim'], extra=tag)
        if got:
            out[k] = got
    return out


def series(game, lr, steps=3000, batch=64, burn=0.25):
    """cos between consecutive policy gradients at a FIXED rate."""
    out, prev = [], None
    for _ in range(steps):
        game.advance()
        game.train_step(game.draw(batch))
        g = torch.cat([p.grad.detach().reshape(-1)
                       for p in game.policy_params if p.grad is not None]).float()
        n = float(g.norm())
        if prev is not None and n > 0:
            pn = float(prev.norm())
            if pn > 0:
                out.append(float(torch.dot(g, prev)) / (n * pn))
        prev = g.clone()
    return out[int(burn * len(out)):]


#: (label, builder, rate). The rate is roughly what each cell wants, so the
#: statistic is read at a sensible operating point rather than an arbitrary one.
CELLS = [
    ('board MLE adam', lambda: MLEGame(dim=32, cond=300.0, noise=0.5, quartic=0.0,
                                       init_scale=3.0, lr=1e-3, optimizer='adam',
                                       seed=0), 1e-3),
    ('tracking adam sp=1e-3', lambda: TrackingGame(dim=32, speed=1e-3, noise=0.1,
                                                   lr=1e-3, optimizer='adam',
                                                   seed=0), 1e-3),
    ('tracking adam sp=1e-2', lambda: TrackingGame(dim=32, speed=1e-2, noise=0.1,
                                                   lr=3e-2, optimizer='adam',
                                                   seed=0), 3e-2),
    ('tracking adam d=256 sp=3e-3',
     lambda: TrackingGame(dim=256, speed=3e-3, noise=0.1, lr=3e-3,
                          optimizer='adam', seed=0), 3e-3),
    ('tracking sgd sp=1e-3', lambda: TrackingGame(dim=32, speed=1e-3, noise=0.1,
                                                  lr=1e-1, optimizer='sgd',
                                                  seed=0), 1e-1),
    ('equilibration sgd', lambda: EquilibrationGame(lr=1.8e-2, optimizer='sgd',
                                                    seed=0, **eqsuite.BASE), 1.8e-2),
]


def main():
    print('=' * 132)
    print('THE REAL RUN')
    print('=' * 132)
    real = real_reference()

    print()
    print('=' * 132)
    print('BENCH CELLS, at a fixed rate')
    print('=' * 132)
    for label, mk, lr in CELLS:
        g = mk()
        _fmt(label, series(g, lr), sum(p.numel() for p in g.policy_params))

    if real.get('fused'):
        med, q1, q3 = real['fused']
        print(f'\n  target to match: median {med:.3f}, width {q3 - q1:.3f}. '
              f'`python -m bench.cos_axis tune` searches for it.')


#: dim lowers the chance floor (and so the width); the speed/noise RATIO sets
#: the median. Both are needed -- see the module docstring.
TUNE = [(d, s, n) for d in (32, 256, 1024, 4096)
        for s in (3e-4, 1e-3, 3e-3, 1e-2) for n in (0.03, 0.1, 0.3, 1.0)]


def _init():
    import torch
    torch.set_num_threads(1)


def _tune_one(job):
    dim, speed, noise = job
    g = TrackingGame(dim=dim, speed=speed, noise=noise, lr=speed,
                     optimizer='adam', seed=0)
    a = np.asarray(series(g, speed, steps=2500))
    return (dim, speed, noise, float(np.median(a)),
            float(np.percentile(a, 25)), float(np.percentile(a, 75)))


def tune(workers=None):
    real = real_reference()
    if not real.get('fused'):
        print('no real `fused` reference to tune against.')
        return
    tmed, tq1, tq3 = real['fused']
    tw = tq3 - tq1
    workers = int(workers or max(2, min(12, (os.cpu_count() or 4) - 4)))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as p:
        out = list(p.map(_tune_one, TUNE))
    print()
    print('=' * 104)
    print(f'CLOSEST TRACKING CELLS  (target median {tmed:.3f}, width {tw:.3f})')
    print('=' * 104)
    print(f'  {"dim":>6} {"chance":>8} {"speed":>7} {"noise":>6} '
          f'{"median":>8} {"middle half":>19} {"width":>7} {"miss":>7}')
    scored = sorted((abs(m - tmed) + 0.5 * abs((c - b) - tw), d, s, n, m, b, c)
                    for d, s, n, m, b, c in out)
    for miss, d, s, n, m, b, c in scored[:12]:
        print(f'  {d:>6} {chance(d):>8.4f} {s:>7.2g} {n:>6.2g} {m:>8.3f} '
              f'[{b:>7.3f},{c:>7.3f}] {c - b:>7.3f} {miss:>7.3f}')


if __name__ == '__main__':
    if 'tune' in sys.argv:
        tune()
    else:
        main()
