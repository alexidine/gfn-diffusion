"""
THE SAFETY HALF: cells built to punish an over-hot rate.

The 7-cell suite produced ZERO divergences in 350 runs, so `hyper step b=0.2`'s
aggressive 20%/step gain has never actually been tested against a cliff -- and
"at worst ~2x the best fixed rate, NEVER 50x" is a statement about exactly that
tail. A confidence sweep nothing can fail does not test safety.

Four hazards, each a different way to be suddenly too hot:

  regime x8    curvature x8 mid-run. Whatever rate you had, it is now 8x too hot.
               The mirror of the suite's `regime /8`, and the harder direction:
               softening gives you slack, hardening takes it away.
  hot start    start at 100x the best fixed rate instead of cold. MK's
               requirement (5), "recover from very bad states -- a purposely too
               hot/fast LR ramp. We want a graceful and fast recovery."
  blowup       parameters scaled hard mid-run, so the loss genuinely explodes and
               the whole tripwire path runs. Not a faked loss value.
  sgd          the same base cell under SGD, where the cliff is REAL: measured,
               fixed@0.01/0.03/0.1 all diverge under SGD on this surface while
               being stable and near-optimal under Adam.

WHAT TO READ. `div` and `nonfin` are the point; `nats` is secondary and is
meaningless for a run that never recovered. An arm that posts a good `nats` with
a nonzero `div` got lucky, not good.

RESULTS, 2026-08-13 (5 seeds, 8000 steps):

  * THE SGD CLIFF IS REAL and lands where theory says. `fixed@0.01/0.1/1` all
    die (~3995 divergences, ~39.7k non-finite steps, 5/5 seeds, never recovers)
    against a stability limit of 2/lambda_max = 2/300 = 0.0067.
  * THE GAIN'S SAFETY BOUNDARY IS BETWEEN 0.2 AND 0.4. On `sgd cliff`,
    `hyper step` b=0.02/0.1/0.2 take ZERO divergences; b=0.4 trips the wire on
    5/5 seeds. The gain that won the 7-cell suite (0.2) is the largest one that
    is still clean, which is a convenient place for the boundary to sit and is
    the reason to stop there rather than at 0.4.
  * `ray+ray` IS NOT THE SAFE OPTION. It trips on 4/5 seeds of `sgd cliff`,
    where hyper step at 0.2 trips on none.
  * !! THE `sgd hot` CELL IS A PASS-THROUGH AND ITS BULLET BELOW IS WRONG. !!
    Measured on seeds 0/1/2: all seven arms produce exactly 821 trace rows, 4
    divergences, and abort with "unrecoverable: 4 rewinds at step 820". The lr
    series is element-wise IDENTICAL to `null`'s for every servo arm, and so is
    `eloss` for the four hyper arms and `ramp+plateau`. The shipped 5-seed table
    prints one value across nine rows (div 20, nonfin 0, nats "never", final lr
    0.00736).

    THE MECHANISM: the cell dies at step 820, INSIDE the 1000-step warmup hold,
    where `Arm._scale_peak` returns without acting. So no controller has been
    allowed to move the rate yet and the cell scores the RELOAD BUDGET, not the
    controllers. Removing either half separates them -- with `warmup_steps=0`,
    `hyper b=0.02` runs all 8000 steps (3 div, final 2.6e-3, eloss 6.7e-5) and
    its lr series stops matching the others; with the reload floor lifted, all
    four survive and separate. Either would make it a real cell.

    The bullet below describes an outcome nothing produces: no arm lands finite.
  * ~~EVERY CONTROLLER RECOVERS FROM A 15x-TOO-HOT START~~ (`sgd hot`, lr 0.1
    against a 0.0067 cliff): ~20 divergence events each, then a finite landing at
    1.4e-3 to 6e-3. Every fixed rate at or above the cliff never recovers. Note
    `null` recovers too -- the tripwire's peak cut is doing the recovering, so
    this scenario scores the SAFETY NET, and the controllers are separated only
    by where they land afterwards (b=0.2 best at 0.55 nats, null worst at 2.40).
  * `ramp+plateau` FROM A HOT START ends at lr=200 under Adam (15.67 nats). It
    ramps away from an already-bad rate and the plateau brake never catches it.

  * THE `blowup` CELL MEASURES A MISSING FEATURE, NOT THE ARMS. Every arm dies on
    5/5 seeds because this runner has no PARAMETER REWIND -- production's
    `fire_loss_spike` restores a checkpoint before cutting the rate, and once
    theta has been scaled to ~1e12 no learning rate can recover it. The ranking
    in that cell is noise (`fixed@1` "wins" it). Until a rewind is modelled, read
    it as a reminder of the gap.
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from bench.arms import Fixed, HyperStep, Null, RampPlateau, RayRay
from bench.metrics import score_run
from bench.runner import Run
from bench.surfaces import MLEGame

SEED_LR = 1.25e-4
#: 100x the best fixed rate on the base cell -- deliberately, badly too hot.
HOT_LR = 1e-1
STEPS = 8000
LADDER = (1e-3, 3e-3, 1e-2, 1e-1, 1.0)   # 1.0 is a guaranteed cliff under SGD
                                        # (stability limit is 2/lambda_max = 2/300 = 0.0067),
                                        # so a cell where NOTHING diverges is a cell with no cliff


def _harden(game):
    """Curvature x8: the rate you are running is now 8x too hot."""
    game.H = game.H * 8.0


@torch.no_grad()
def _blowup(game):
    """A real explosion: scale the parameters until the loss crosses the bar."""
    for _ in range(12):
        if not math.isfinite(float(game.expected_loss())) or \
                float(game.expected_loss()) >= 2e9:
            return
        game.theta.mul_(10.0)


#: (label, optimizer, start_lr, mutate, kwargs)
CELLS = [
    ('regime x8', 'adam', SEED_LR, _harden, dict(cond=300.0, noise=0.5)),
    ('hot start', 'adam', HOT_LR, None, dict(cond=300.0, noise=0.5)),
    ('blowup', 'adam', SEED_LR, _blowup, dict(cond=300.0, noise=0.5)),
    ('sgd cliff', 'sgd', SEED_LR, None, dict(cond=300.0, noise=0.5)),
    ('sgd hot', 'sgd', HOT_LR, None, dict(cond=300.0, noise=0.5)),
]


def build_arms(start_lr):
    return ([HyperStep(start_lr, beta=0.02), HyperStep(start_lr, beta=0.1),
             HyperStep(start_lr, beta=0.2), HyperStep(start_lr, beta=0.4),
             RayRay(start_lr), RampPlateau(start_lr), Null(start_lr)]
            + [Fixed(x) for x in LADDER])


def _one(item):
    ci, ai, seed = item
    label, opt, start, mutate, kw = CELLS[ci]
    arm = build_arms(start)[ai]
    game = MLEGame(dim=32, init_scale=3.0, lr=start, optimizer=opt,
                   seed=seed, **kw)
    run = Run(game, arm, seed=seed, steps=STEPS, batch=64)
    at = STEPS // 2
    for i in range(STEPS):
        if mutate is not None and i == at:
            mutate(game)
        run.step()
        if run.aborted:
            break
    return ci, score_run(run)


def _init():
    import torch as t
    t.set_num_threads(1)


def main(seeds=5, workers=None):
    seeds = tuple(range(int(seeds)))
    workers = int(workers or max(2, min(16, (os.cpu_count() or 4) - 4)))
    n_arms = len(build_arms(SEED_LR))
    print(f'{"=" * 100}\nHAZARDS -- {len(CELLS)} cells x {n_arms} arms x '
          f'{len(seeds)} seeds, {STEPS} steps\n{"=" * 100}')

    jobs = [(ci, ai, s) for ci in range(len(CELLS))
            for ai in range(n_arms) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        out = list(pool.map(_one, jobs))

    per = {}
    for ci, row in out:
        per.setdefault(ci, {}).setdefault(row['arm'], []).append(row)

    for ci, (label, opt, start, mutate, kw) in enumerate(CELLS):
        rows = per.get(ci, {})
        if not rows:
            continue
        live = [float(np.median([r['final_loss'] for r in rs]))
                for rs in rows.values()
                if any(math.isfinite(r['final_loss']) for r in rs)]
        best = min(live) if live else math.nan
        print(f'\n  {label}   ({opt}, start lr {start:g}'
              + (f', {mutate.__name__.strip("_")} at midpoint)' if mutate
                 else ')'))
        print(f'    {"arm":<20} {"div":>5} {"nonfin":>7} {"seeds hit":>10} '
              f'{"nats":>8} {"final lr":>10}')
        tbl = []
        for a in build_arms(start):
            rs = rows.get(a.name, [])
            if not rs:
                continue
            fin = float(np.median([r['final_loss'] for r in rs]))
            nats = (math.log(fin / best) if math.isfinite(fin) and fin > 0
                    and best and best > 0 else math.inf)
            hit = sum(1 for r in rs if r['divergences'] or r['nonfinite_steps'])
            tbl.append((sum(r['divergences'] for r in rs), nats, a.name, rs, hit))
        for div, nats, name, rs, hit in sorted(tbl, key=lambda t: (t[0], t[1])):
            s = f'{nats:>8.2f}' if math.isfinite(nats) else f'{"never":>8}'
            print(f'    {name:<20} {div:>5} '
                  f'{sum(r["nonfinite_steps"] for r in rs):>7} '
                  f'{hit:>6}/{len(rs):<3} {s} '
                  f'{np.nanmedian([r["final_lr"] for r in rs]):>10.3g}')
    return per


if __name__ == '__main__':
    main(next((int(a) for a in sys.argv[1:] if a.isdigit()), 5))
