"""
THE EQUILIBRATION LEADERBOARD -- three players, no joint potential.

    python -m bench.eqboard            # 5 seeds
    python -m bench.eqboard 10

WHAT IS DIFFERENT FROM THE MLE BOARD, and it changes how to read the table:

  * THIS IS A CLIFF PROBLEM, NOT A TUNING PROBLEM. Measured, inside the stable
    band the settled distance moves only ~3.3x across a 10x rate sweep, while
    crossing the boundary reaches 1e35 in ~200 steps. `lr/cliff` is therefore
    the most informative column -- but read it as a POSITION, not a score.

    THE OPTIMUM IS NOT AT THE CLIFF. This file used to say the reward was "get
    as close to the cliff as you can without going over", i.e. that lr/cliff 1.0
    was the target. `bench.eqsuite` refutes that across cells: `fixed@0.03` sits
    at 0.94-1.04x and finishes 0.34 / 0.52 / 3.91 nats back, then dies outright
    in two of the nine. The good zone is 0.3-0.65x. The edge is a constraint to
    respect, not a setpoint to seek -- the same shape-not-setpoint mistake
    already recorded for alpha*.
  * THERE IS EXACT GROUND TRUTH. `stability_lr` is the closed-form
    spectral-radius-1 rate: 0.0318 with the level pinned at `lr_flow`, which is
    the applicable variant (the Z head is exempt from the servo, so its rate
    does not scale with the policy's).

    THE "2.15 vs 0.032" JUSTIFICATION FOR PREFERRING IT IS RETRACTED. Measured,
    the both-scale variant now returns 0.0293 -- 0.92x the pinned one, agreeing
    to within 8%. Both numbers here were the pre-fix scalar values that
    `surfaces.py::one_step_lr` (:896-916) labels unsupported.
  * A ONE-STEP PROBE IS *NOT* 31x WRONG HERE, and this file's stated reason for
    existing was wrong by ~34x. Measured `one_step_lr` is 0.287, not 1.0.
    Worse, the ratio depends entirely on WHEN it is read: `cliff()` reads it
    after 30 steps, while the gradient is still transient, and gets 4.2-9.2x.
    At any settled operating point it is 0.46-0.91x, median 0.90 -- i.e. the
    one-step probe sits slightly BELOW the boundary, not over it. THE SIGN OF
    THE CLAIM FLIPS. `surfaces.py`'s own docstring already says this ("a median
    of 0.85x the cliff -- essentially AT the boundary, not 31x over it"); this
    header had not been updated to match.

    The runtime banner below still prints the 30-step reading, so a run of this
    board prints "Nx too hot" from a transient. Read it as a transient, or read
    the settled numbers above.
  * SGD, because that is where the closed form applies -- the game is linear.
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.arms import Fixed, HyperStep, Null, RampPlateau, RayRay
from bench.metrics import score_run
from bench.runner import Run
from bench.surfaces import EquilibrationGame

SEED_LR = 1e-4
STEPS = 6000

#: THE SURFACE COMES FROM `eqsuite`, IT IS NOT RESTATED HERE.
#:
#: This file used to carry its own copy, and the copy had gone stale by two
#: FIELDS THAT WERE SIMPLY ABSENT -- `cond_bwd` and `drift` -- which default to
#: off. `eqsuite`'s docstring records both as the repairs that made the surface
#: able to rank anything at all. Measured, the same ladder on each:
#:
#:     stale copy      bwd branch = 0.7% of the policy gradient, 2x band 32x
#:                     wide, seed sd 0.086 nats
#:     eqsuite.BASE    bwd = 19.1%, band 3x wide, seed sd 0.021 nats
#:
#: On the stale one this board's top eight arms spanned 0.00-0.30 nats against
#: a seed sd of 0.07-0.13 -- it resolved nothing. On the repaired one it does,
#: and the answer CHANGES: ray goes from 0.11 to 0.63 nats behind and ends 3-5x
#: too cold. Every equilibration verdict read off the old copy is about a
#: surface that was fixed and not re-run.
from bench.eqsuite import BASE as KW

#: 0.018 IS ON THE LADDER because it is the cell's actual optimum. Without it
#: the fixed-rate reference the controllers are scored against was itself ~0.28
#: nats off, and a ladder that does not contain the answer reports its own
#: spacing.
LADDER = (3e-4, 1e-3, 3e-3, 1e-2, 1.8e-2, 3e-2, 1e-1)


def make_game(lr, seed=0, **over):
    return EquilibrationGame(lr=lr, optimizer='sgd', seed=seed,
                             **{**KW, **over})


def build_arms():
    return ([HyperStep(SEED_LR, beta=0.02), HyperStep(SEED_LR, beta=0.1),
             HyperStep(SEED_LR, beta=0.2), RayRay(SEED_LR, period=100),
             RayRay(SEED_LR, period=20), RampPlateau(SEED_LR), Null(SEED_LR)]
            + [Fixed(x) for x in LADDER])


def _one(item):
    ai, seed = item
    arm = build_arms()[ai]
    lr = arm.lr if isinstance(arm, Fixed) else SEED_LR
    game = make_game(lr, seed=seed)
    run = Run(game, arm, seed=seed, steps=STEPS, batch=64).run()
    return score_run(run)


def _init():
    import torch
    torch.set_num_threads(1)


def cliff():
    g = make_game(0.01)
    r = Run(g, Fixed(0.01), seed=0, steps=30, batch=64)
    r.run()
    pin = g.optimizers['fused'].param_groups[-1]['lr']
    return g.stability_lr(lr_level=pin), g.one_step_lr(), pin


def main(seeds=5, workers=None):
    seeds = tuple(range(int(seeds)))
    arms = build_arms()
    workers = int(workers or max(2, min(16, (os.cpu_count() or 4) - 4)))
    c, one, pin = cliff()
    print(f'{"=" * 96}\nEQUILIBRATION BOARD -- {len(arms)} arms x {len(seeds)} '
          f'seeds, {STEPS} steps, SGD\n'
          f'closed-form cliff {c:.4f} (level pinned at {pin:g}) | a one-step '
          f'probe would say {one:.3g} -- {one / c:.0f}x too hot\n{"=" * 96}\n')

    jobs = [(i, s) for i in range(len(arms)) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        out = list(pool.map(_one, jobs))

    rows = {}
    for r in out:
        rows.setdefault(r['arm'], []).append(r)
    live = [float(np.median([r['final_loss'] for r in rs]))
            for rs in rows.values()
            if all(math.isfinite(r['final_loss']) for r in rs)]
    best = min(live) if live else math.nan

    print(f'  {"arm":<20} {"nats":>8} {"lr/cliff":>9} {"final lr":>10} '
          f'{"div":>5} {"nonfin":>7} {"backslide":>10}')
    tbl = []
    for a in arms:
        rs = rows.get(a.name, [])
        if not rs:
            continue
        fin = float(np.median([r['final_loss'] for r in rs]))
        nats = (math.log(fin / best) if math.isfinite(fin) and fin > 0
                and best > 0 else math.inf)
        tbl.append((nats, a.name, rs))
    for nats, name, rs in sorted(tbl):
        s = f'{nats:>8.2f}' if math.isfinite(nats) else f'{"never":>8}'
        endlr = np.nanmedian([r['final_lr'] for r in rs])
        print(f'  {name:<20} {s} {endlr / c:>9.2f} {endlr:>10.3g} '
              f'{sum(r["divergences"] for r in rs):>5} '
              f'{sum(r["nonfinite_steps"] for r in rs):>7} '
              f'{np.nanmedian([r["backslide"] for r in rs]):>10.1%}')
    print(f'\n  lr/cliff: 1.0 = riding the stability edge, 0.1 = leaving most of '
          f'the rate unused,\n            >1 = should be dead. nats is secondary '
          f'here -- inside the band the\n            outcome is nearly flat, so '
          f'read the cliff ratio and the divergences.')
    return rows


if __name__ == '__main__':
    main(next((int(a) for a in sys.argv[1:] if a.isdigit()), 5))
