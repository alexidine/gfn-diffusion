"""
THE EQUILIBRATION BATTERY -- the same arms across the axes that should move the
cliff, plus the hazards the MLE family already covers and this one did not.

`eqboard` is ONE configuration. Given that configuration took three bugs and a
wrong boundary variant to make valid, one cell is not evidence. This sweeps the
knobs that set the problem, each cell scored against its OWN closed-form cliff.

  conflict     `cond_rep`, how much the branches disagree per-coordinate. This
               is the least justified knob on the board -- 100 is a choice, not
               a measurement -- so it is swept hardest, including cond_rep=1
               (the isotropic control, where the rate should NOT matter and any
               arm separation would mean the board is measuring something else).
  coupling     `a`, the anti-phase gain. Prior work flagged a=4 as invented and
               a=b as the honest setting, so both ends are here.
  buffer       `kappa`, how fast mu tracks theta. Small kappa is a slow pole and
               a permanently stale buffer, which is the documented concern.
  weights      w_rep/w_bwd -- the branch mixture, which in production is a live
               controlled quantity rather than a constant.
  level rate   `lr_flow`. The Z head is pinned and exempt from the servo, and the
               cliff depends on it strongly: 2.15 vs 0.03 on the same game.
  hazards      hot start, a regime change mid-run, and repeated blow-ups. The
               MLE family has all three; equilibration had none, so every
               equilibration verdict so far was about a cold start on a
               stationary surface approached from below.
  optimizer    an Adam cell. Production runs Adam on every optimizer and this
               surface has been SGD-only, so an SGD-only verdict is a
               recommendation for a configuration nobody ships.

TWO AXES ARE STRUCTURALLY UNMEASURABLE HERE, both established by measurement
rather than argued, and BOTH USED TO BE CELLS:

  noise        DROPPED. The game is linear with its fixed point at the origin,
               so the state is `deterministic(t) + noise * stochastic(t)` and
               after the transient the whole trajectory is proportional to
               `noise`. Distance is quadratic in it (measured: exactly 100.0x
               for 10x noise, for every arm) and gradients scale with it too, so
               every scale-free controller -- a cosine, a ratio test, a plateau
               comparison, i.e. all of them -- is EXACTLY blind to it. Hyper's
               chosen rate came back bit-identical at 0.01984. No rescoring
               fixes this; only breaking the scale invariance would, and the
               attempt failed: `quartic=1` is still inert at equilibrium while
               `quartic>=100` detonates at init, because theta starts at O(1)
               and settles at O(1e-3) and no coefficient is negligible at one
               end and significant at the other. Noise robustness has to be
               read off the MLE family, which has three cells that do separate.
  slow buffer  REPAIRED, not dropped. kappa=0.002 moved the settled distance
               0.3% at 6000 steps (13% at 3000 -- the pole is transient-only and
               washes out). kappa=0.0005 moves it 57x and is the documented
               limit-cycle regime, so that is the cell.

`HyperCapped` (hyper's tracking + a ray-derived ceiling) WAS AN ARM HERE AND IS
DELETED. It was refuted by a PLACEBO -- an arm running the identical probe, with
identical RNG consumption, that then discards the cap. In `no conflict`, the cell
it was built for, hybrid and placebo both scored 1.03 nats: the cap contributed
exactly nothing, and the ~0.03 improvement over plain hyper came from the probe's
draws shifting the noise stream, not from any control. Two structural reasons,
either of which is fatal:
  * the ceiling was written once and never held, so the per-step hypergradient
    climbed back over it within 2-7 steps -- in force ~10-24% of the interval;
  * it could only bind when alpha* < target/slack = 2, and the only probe-grid
    values under 2 are 1.0 and 1.414, making it a TWO-VALUED SWITCH (x0.5 or
    x0.707) on a noisy statistic -- the failure mode `HyperSNR`'s docstring says
    killed four earlier arms. It cannot express "catastrophically hot" at all.
A sticky-ceiling variant reached 0.80 nats, still far from the 0.3 predicted, so
the target is unreachable at this slack rather than the implementation being off.

    python -m bench.eqsuite 5
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
# THE REPAIRED SURFACE. Every equilibration verdict before 2026-08-13 was run
# on a version of this that could not rank controllers, for two reasons found by
# audit rather than argument (`bench.audit`, `bench.test_surface_fitness`):
#
#   cond_bwd was never set, so `S_bwd` was all ones in every cell and the
#   "opposed spectra" design was INERT. The bwd branch supplied 0.7% of the
#   policy gradient while pointing the opposite way to its own docstring's
#   claim -- three competing optimisations was two plus a rounding error.
#   Setting it takes bwd to ~26% of the gradient.
#
#   drift/drift_pull make the target a stationary Ornstein-Uhlenbeck process
#   instead of a fixed point. This is the substantive change. On a problem that
#   DECAYS TO A STATIC POINT, a narrow band of good rates and a settled run are
#   mutually exclusive: once converged the outcome is the noise floor, which
#   depends on the rate only as ~sqrt(lr), so the band within 2x of best was 30x
#   WIDE and every arm landed inside it and tied; tighten the budget or drop the
#   noise and the band narrows but the run is still descending, so what gets
#   ranked is convergence speed and the answer moves with the budget. Six
#   configurations, same wall. With a moving target both errors persist -- too
#   cold lags by ~drift/lr, too hot sits in a noise ball ~lr*sigma -- giving a
#   sharp interior optimum on a stationary problem. Measured: the band goes 30x
#   -> 3x, seed noise 0.078 -> 0.021 nats, and adjacent rungs separate at 13-20
#   sigma where they used to manage 1.5. Roughly 10x the resolving power.
#
#   It is also the honest picture: equilibration's target never stops moving,
#   because the buffer refreshes and the level moves.
BASE = dict(dim=8, a=2.0, b=1.0, w_rep=0.7, w_bwd=0.3, kappa=0.02, noise=0.1,
            init_scale=1.0, cond_rep=100.0, cond_bwd=100.0,
            drift=0.01, drift_pull=0.01)

#: Repeated, because a single shock is FREE once the rewind works -- the restore
#: erases it and cold and hot rates end bit-identically. What the cell actually
#: measures is the peak cut each divergence leaves behind and how fast an arm
#: climbs back out of it. Three fits inside the reload budget; a fourth aborts.
SHOCKS = ((1500, 1e4), (3000, 1e4), (4500, 1e4))

#: label -> dict(over=game kwargs, flow=lr_flow, seed_lr=, optimizer=)
CELLS = [
    ('base',           {}),
    # the CONTROL for the redesign: same cell, target held still. If the
    # arms do not separate more here than on `base`, the drift bought
    # nothing and the whole change should be reverted.
    ('static target',  dict(over=dict(drift=0.0))),
    # BOTH spectra flat, or it is not the isotropic control it claims to be
    ('no conflict',    dict(over=dict(cond_rep=1.0, cond_bwd=1.0))),
    ('conflict x10',   dict(over=dict(cond_rep=1000.0, cond_bwd=1000.0))),
    ('coupling a=6',   dict(over=dict(a=6.0))),
    ('coupling a=b=1', dict(over=dict(a=1.0, b=1.0))),
    ('bwd-heavy',      dict(over=dict(w_rep=0.3, w_bwd=0.7))),
    ('slow buffer',    dict(over=dict(kappa=0.0005))),
    ('slow level',     dict(flow=0.01)),
    ('hot start',      dict(seed_lr=0.05)),
    ('regime shift',   dict(over=dict(schedule=((3000, {'cond_rep': 1000.0}),)))),
    ('blow-up',        dict(over=dict(shock=SHOCKS))),
    ('adam',           dict(optimizer='adam')),
]

LADDER = (1e-3, 3e-3, 1e-2, 3e-2)


def build_arms(seed_lr=SEED_LR):
    return ([HyperStep(seed_lr, beta=0.02), HyperStep(seed_lr, beta=0.2),
             RayRay(seed_lr, period=100), RayRay(seed_lr, period=20),
             RampPlateau(seed_lr), Null(seed_lr)]
            + [Fixed(x) for x in LADDER])


def _spec(ci):
    _, s = CELLS[ci]
    return (s.get('over', {}), s.get('flow'), float(s.get('seed_lr', SEED_LR)),
            s.get('optimizer', 'sgd'))


def _mk(ci, lr, seed):
    over, _, _, opt = _spec(ci)
    return EquilibrationGame(lr=lr, optimizer=opt, seed=seed,
                             **{**BASE, **over})


def _one(item):
    ci, ai, seed = item
    _, flow, seed_lr, _ = _spec(ci)
    arm = build_arms(seed_lr)[ai]
    lr = arm.lr if isinstance(arm, Fixed) else seed_lr
    game = _mk(ci, lr, seed)
    run = Run(game, arm, seed=seed, steps=STEPS, batch=64)
    if flow is not None:
        run.m.args.lr_flow = flow
    run.run()
    row = score_run(run)
    row['aborted_run'] = bool(run.aborted)
    row['reloads'] = run.reloads
    return ci, row


def _init():
    import torch
    torch.set_num_threads(1)


def cliff_of(ci):
    """
    The cell's own level-pinned closed-form boundary.

    Returns None where there is no exact one: Adam's preconditioner changes the
    map, so the spectral-radius argument does not describe it. Reporting the SGD
    number there would be ground truth that has quietly stopped describing the
    game -- the failure `iteration_matrix` already had once.
    """
    over, flow, _, opt = _spec(ci)
    if opt != 'sgd':
        return None, None
    g = _mk(ci, 0.01, 0)
    r = Run(g, Fixed(0.01), seed=0, steps=30, batch=64)
    if flow is not None:
        r.m.args.lr_flow = flow
    r.run()
    # THE CLIFF AS IT STANDS AT THE END OF THE RUN, because `final_lr` is a
    # median of the LAST 100 steps and the ratio has to divide like by like.
    #
    # This used to report the boundary at step 30 -- pre-shift -- and then reuse
    # it as the column divisor for a post-shift rate. On `regime shift` the two
    # differ by 9.96x, and it INVERTED the cell's verdict: the three arms that
    # died printed 0.94 / 0.68 / 0.31, which the legend reads as "inside the
    # band", when they were 9.4 / 6.8 / 3.1x OVER it. The surviving controllers
    # printed 0.01-0.06 ("leaving most of the rate unused") when they were at
    # 0.1-0.6x, i.e. exactly the good zone. This is the only cell that tests
    # tracking, and this column is the one its verdict is read from.
    for at, changes in getattr(g, 'schedule', ()):
        if int(at) <= STEPS and 'cond_rep' in changes:
            g.set_conflict(changes['cond_rep'])
    pin = g.optimizers['fused'].param_groups[-1]['lr']
    return g.stability_lr(lr_level=pin), pin


def main(seeds=5, workers=None):
    seeds = tuple(range(int(seeds)))
    names = [a.name for a in build_arms()]
    workers = int(workers or max(2, min(16, (os.cpu_count() or 4) - 4)))
    print(f'{"=" * 100}\nEQUILIBRATION SUITE -- {len(CELLS)} cells x '
          f'{len(names)} arms x {len(seeds)} seeds, {STEPS} steps\n{"=" * 100}')

    jobs = [(ci, ai, s) for ci in range(len(CELLS))
            for ai in range(len(names)) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        out = list(pool.map(_one, jobs))

    per = {}
    for ci, row in out:
        per.setdefault(ci, {}).setdefault(row['arm'], []).append(row)

    totals = {n: [] for n in names}
    divs = {n: 0 for n in names}
    aborts = {n: 0 for n in names}
    for ci, (label, spec) in enumerate(CELLS):
        rows = per.get(ci, {})
        if not rows:
            continue
        c, pin = cliff_of(ci)
        live = [float(np.median([r['final_loss'] for r in rs]))
                for rs in rows.values()
                if all(math.isfinite(r['final_loss']) for r in rs)]
        best = min(live) if live else math.nan
        head = (f'cliff {c:.4f} (level {pin:g})' if c else 'cliff n/a (adam)')
        print(f'\n  {label:<16} {head}  {spec}')
        print(f'    {"arm":<22} {"nats":>8} {"lr/cliff":>9} {"div":>5} '
              f'{"abort":>6}')
        tbl = []
        for n in names:
            rs = rows.get(n, [])
            if not rs:
                continue
            # ANY ABORTED SEED KILLS THE ARM IN THIS CELL. A median over the
            # surviving seeds is exactly the censoring trap: an arm that dies 1
            # run in 5 keeps a healthy median and outranks arms that finished
            # every time. The stated goal is a worst-case one -- an arm that
            # detonates occasionally is unusable however good its median is.
            #
            # This also repairs a regression the abort fix introduced: `live`
            # (below) requires ALL seeds finite while the score was a MEDIAN, so
            # an arm with 1-2 aborted seeds kept a finite median, was excluded
            # from `best`, and could print a NEGATIVE nats in a column whose
            # legend says 0 is the winner -- measured at -2.30.
            fin = (math.inf if any(r['aborted_run'] for r in rs)
                   else float(np.median([r['final_loss'] for r in rs])))
            nats = (math.log(fin / best) if math.isfinite(fin) and fin > 0
                    and best > 0 else math.inf)
            totals[n].append(nats)
            divs[n] += sum(r['divergences'] for r in rs)
            aborts[n] += sum(1 for r in rs if r['aborted_run'])
            tbl.append((nats, n, rs))
        for nats, name, rs in sorted(tbl):
            s = f'{nats:>8.2f}' if math.isfinite(nats) else f'{"never":>8}'
            lr = np.nanmedian([r['final_lr'] for r in rs])
            ratio = f'{lr / c:>9.2f}' if c else f'{lr:>9.2g}'
            print(f'    {name:<22} {s} {ratio} '
                  f'{sum(r["divergences"] for r in rs):>5} '
                  f'{sum(1 for r in rs if r["aborted_run"]):>6}')

    #: SORT BY WORST CASE, SURVIVORS FIRST. Sorting on mean nats ranked
    #: `fixed@0.01` top of this table while it was DEAD in a cell: the mean is
    #: over finite cells only, so censoring flatters exactly the arms that die.
    #: The stated goal ("at worst ~2x the best rate, never 50x") is a worst-case
    #: bar anyway, which no mean can answer.
    print(f'\n{"=" * 100}\nACROSS CELLS -- sorted by WORST cell, arms that died '
          f'last\n{"=" * 100}')
    print(f'  {"arm":<22} {"worst":>8} {"mean nats":>10} {"died in":>9} '
          f'{"div":>6} {"aborts":>7}')
    dead = {k: sum(1 for x in v if not math.isfinite(x))
            for k, v in totals.items()}
    for name, v in sorted(totals.items(),
                          key=lambda kv: (dead[kv[0]] > 0,
                                          max(kv[1]) if kv[1] else math.inf,
                                          np.mean([x for x in kv[1]
                                                   if math.isfinite(x)] or [1e9]))):
        fin = [x for x in v if math.isfinite(x)]
        m = np.mean(fin) if fin else math.inf
        w = max(v) if v else math.inf
        ws = f'{w:>8.2f}' if math.isfinite(w) else f'{"never":>8}'
        d = f'{dead[name]}/{len(v)}' if dead[name] else '-'
        print(f'  {name:<22} {ws} {m:>10.2f} {d:>9} {divs[name]:>6} '
              f'{aborts[name]:>7}')
    print('\n  mean nats is over SURVIVING cells only -- an arm that died has a '
          'flattering\n  mean and a meaningless one. Read `worst` and `died in` '
          'first.\n'
          '  In `blow-up` the fixed arms are IMMUNE BY CONSTRUCTION: the peak '
          'cut applies to\n  servo-managed groups and a fixed rate is not one, '
          'so they take the divergence\n  without the penalty every controller '
          'pays. Do not read that column as skill.')
    return per


if __name__ == '__main__':
    main(next((int(a) for a in sys.argv[1:] if a.isdigit()), 5))
