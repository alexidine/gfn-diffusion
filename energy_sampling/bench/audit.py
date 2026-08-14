"""
IS THIS SURFACE FIT TO RANK CONTROLLERS? Six measurements, run before any arm.

Every equilibration verdict so far was produced on a surface nobody had audited,
and four reviews found the surface was not doing what its own docstring said.
The failures were not subtle once measured -- they were simply never measured.
This is the check that should have existed first.

A surface is fit for purpose iff:

  1. THE RATE MATTERS.        Across the rates that survive, the outcome must
                              move a lot. Measured on the current cell: 1.7x
                              over a 30x sweep. A controller cannot demonstrate
                              skill on a problem where every survivable rate is
                              equally good -- the board becomes a divergence
                              detector wearing a leaderboard's clothes.
  2. THE OPTIMUM IS INTERIOR. If the best rate is the top or bottom rung, the
                              ladder is reporting its own edge.
  3. IT IS SETTLED.           If the outcome is still falling at the horizon,
                              the cell ranks arms by how fast they got hot, and
                              the ranking flips when the budget changes.
  4. EVERY PLAYER PULLS.      A branch contributing 0.3% of the gradient is not
                              a player. This is what "three competing
                              optimisations" has to mean mechanically.
  5. SIGNAL BEATS SEED NOISE. The spread across rates must be large against the
                              spread across seeds at ONE rate. Measured on the
                              current cell: 5-seed noise 0.054-0.065 nats
                              against reported gaps of 0.01-0.06 -- a board that
                              reproduces its own top-5 ordering 3-7% of the time.
  6. THE CLIFF IS REAL.       The closed form must predict the empirical
                              boundary, or `lr/cliff` is decoration.

    python -m bench.audit                 # audit the shipped BASE
    python -m bench.audit --variants      # and the redesign candidates
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from bench.arms import Fixed
from bench.metrics import final_loss
from bench.runner import Run
from bench.surfaces import EquilibrationGame

RATES = (3e-4, 5.6e-4, 1e-3, 1.8e-3, 3.2e-3, 5.6e-3, 1e-2,
         1.8e-2, 3.2e-2, 5.6e-2)
STEPS = 6000
SEEDS = 6

#: what the battery ships today
BASE = dict(dim=8, a=2.0, b=1.0, w_rep=0.7, w_bwd=0.3, kappa=0.02, noise=0.1,
            init_scale=1.0, cond_rep=100.0)


def _one(item):
    kw, lr, seed, steps = item
    g = EquilibrationGame(lr=lr, optimizer='sgd', seed=seed, **kw)
    r = Run(g, Fixed(lr), seed=seed, steps=steps, batch=64)
    r.run()
    if r.aborted:
        return lr, seed, math.inf, math.inf, math.nan
    # THE BATTERY'S OWN ESTIMATOR, not a single endpoint. `expected_loss()` at
    # the last step is one draw and is far noisier than what the board scores;
    # measuring the surface with a different estimator than the board uses
    # reports a noise floor the board does not have.
    end = final_loss(r)
    # SETTLED? compare the last fifth of the run to the one before it. A cell
    # still descending at the horizon ranks arms on speed, not on placement.
    tail = [h['eloss'] for h in r.trace[-steps // 5:] if h['eloss']]
    prev = [h['eloss'] for h in r.trace[-2 * steps // 5:-steps // 5]
            if h['eloss']]
    ratio = (float(np.median(prev)) / float(np.median(tail))
             if tail and prev and np.median(tail) > 0 else math.nan)
    return lr, seed, (end if math.isfinite(end) and end > 0 else math.inf), \
        end, ratio


def _init():
    import torch as t
    t.set_num_threads(1)


def branch_shares(kw, lr, steps=3000):
    """
    What fraction of the policy gradient each branch supplies, and how aligned
    they are -- measured at a SETTLED state, not at initialisation.

    At init `mu` is set equal to `theta`, so the bwd gradient is exactly zero and
    the alignment is NaN. A claim about branch conflict evaluated there says
    nothing, which is how a docstring came to assert cos = +0.85 when the settled
    value is -0.92.
    """
    g = EquilibrationGame(lr=lr, optimizer='sgd', seed=0, **kw)
    Run(g, Fixed(lr), seed=0, steps=steps, batch=64).run()
    n_theta, _ = g.draw(64)
    rep = torch.autograd.grad(g.w_rep * g._replay_loss(n_theta), [g.theta],
                              retain_graph=True)[0]
    bwd = torch.autograd.grad(g.w_bwd * g._bwd_loss(), [g.theta])[0]
    a, b = float(rep.norm()), float(bwd.norm())
    cos = (float(torch.dot(rep, bwd)) / (a * b)) if a > 0 and b > 0 else math.nan
    return a, b, (b / (a + b) if a + b > 0 else math.nan), cos


def audit(kw, label, workers=None, steps=STEPS):
    workers = int(workers or max(2, min(16, (os.cpu_count() or 4) - 4)))
    jobs = [(kw, lr, s, steps) for lr in RATES for s in range(SEEDS)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        out = list(pool.map(_one, jobs))

    by = {}
    for lr, seed, score, end, ratio in out:
        by.setdefault(lr, []).append((score, ratio))

    med = {lr: float(np.median([x[0] for x in v])) for lr, v in by.items()}
    alive = {lr: v for lr, v in med.items() if math.isfinite(v)}
    if not alive:
        print(f'{label}: every rate died'); return None
    best_lr = min(alive, key=alive.get)
    rng = math.log(max(alive.values()) / min(alive.values()))

    # seed noise AT THE BEST RATE, in the same units as the range
    vs = [x[0] for x in by[best_lr] if math.isfinite(x[0])]
    noise = float(np.std(np.log(vs))) if len(vs) > 1 else math.nan

    g = EquilibrationGame(lr=0.01, optimizer='sgd', seed=0, **kw)
    r = Run(g, Fixed(0.01), seed=0, steps=30, batch=64); r.run()
    cliff = g.stability_lr(lr_level=g.optimizers['fused'].param_groups[-1]['lr'])
    rep_n, bwd_n, share, cos = branch_shares(kw, best_lr)
    settle = float(np.median([x[1] for x in by[best_lr]
                              if x[1] and math.isfinite(x[1])] or [math.nan]))
    interior = best_lr not in (RATES[0], RATES[-1])
    survivors = sorted(alive)

    print(f'\n=== {label} ===')
    print(f'  {"rate":<8} ' + ' '.join(f'{lr:>8.2g}' for lr in RATES))
    print(f'  {"nats":<10} ' + ' '.join(
        f'{math.log(med[lr] / min(alive.values())):>8.2f}'
        if math.isfinite(med[lr]) else f'{"died":>8}' for lr in RATES))
    # THE NUMBER THAT DECIDES WHETHER CONTROLLERS ARE SEPARABLE. The full
    # ladder spread is dominated by rates far too cold to converge, which no
    # controller ever visits. What matters is the FLOOR: the set of rates within
    # 2x of the best. Every arm lands there, so if the floor is wide and flat
    # the board cannot tell them apart no matter how many seeds it runs.
    floor = [lr for lr in alive
             if math.log(alive[lr] / min(alive.values())) <= math.log(2)]
    fw = max(floor) / min(floor) if floor else math.nan
    fspread = (math.log(max(alive[lr] for lr in floor)
                        / min(alive[lr] for lr in floor)) if floor else math.nan)
    print(f'  1 rate matters      full ladder {rng:>6.2f} nats ({math.exp(rng):>7.1f}x)'
          f'   {"OK" if rng > 2.3 else "WEAK"}')
    print(f'  1b FLAT FLOOR       {fw:>5.0f}x wide, spread {fspread:>5.2f} nats'
          f'    {"OK" if fw <= 10 else "WIDE -- arms land here and tie"}')
    print(f'  2 optimum interior  best {best_lr:<9g}'
          f'              {"OK" if interior else "NO -- ladder edge"}')
    # TWO-SIDED. ratio = prev/tail, so >1 is still descending and <1 is
    # getting WORSE -- and a one-sided `< 1.5` test passed both. A drifting
    # target that wanders away scores 0.58 here and was being called settled.
    drift_up = settle < 0.8
    print(f'  3 stationary        prev/tail {settle:>5.2f}'
          f'             {"OK" if 0.8 <= settle <= 1.5 else ("NO -- DEGRADING" if drift_up else "NO -- still descending")}')
    print(f'  4 every player      bwd share {share:>6.1%} cos {cos:>6.2f}'
          f'     {"OK" if share > 0.1 else "NO -- bwd is a rounding error"}')
    print(f'  5 signal vs seeds   range/noise {rng / noise if noise else float("nan"):>6.1f}'
          f'         {"OK" if noise and rng / noise > 10 else "WEAK"}'
          f'   (noise {noise:.3f} nats)')
    print(f'  6 cliff             closed form {cliff:<9.4g}'
          f'          top survivor {survivors[-1]:g}, '
          f'{"consistent" if survivors[-1] <= cliff <= (survivors[-1] * 10) else "CHECK"}')
    return dict(range=rng, best=best_lr, noise=noise, share=share, cos=cos,
                settle=settle, cliff=cliff, interior=interior)


if __name__ == '__main__':
    audit(BASE, 'SHIPPED BASE (what every equilibration verdict used)')
    if '--variants' in sys.argv:
        audit({**BASE, 'cond_bwd': 100.0},
              'V1  cond_bwd=100 -- ACTIVATE the opposed spectra (currently inert)')
        audit({**BASE, 'cond_rep': 1000.0, 'cond_bwd': 1000.0},
              'V2  both spectra x1000 -- harder conditioning')
        audit({**BASE, 'cond_bwd': 100.0, 'kappa': 0.005},
              'V3  opposed spectra + a genuinely stale buffer')
