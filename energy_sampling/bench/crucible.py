"""
THE CRUCIBLE: cells built to break something, because the one-at-a-time sweep
could not.

Three reasons the previous battery failed to discriminate, each addressed here.

1. AXES WERE SWEPT ONE AT A TIME. Noise favours the blind arm (no sensor to
   corrupt); a moving target favours the measuring arm (a recorded ceiling cannot
   follow an optimum that rises). They select OPPOSITE winners, so an arm that
   fails only on the INTERACTION is invisible to a one-at-a-time sweep. The
   `noise x quartic` cells are the ones nothing should survive.

2. EVERY OVER-BUDGET RUN WAS A COLD START. Blow-up and hot recovery are handled
   by the divergence tripwire and rewind for EVERY arm, so those scenarios scored
   the safety net rather than the controller. Two new scenarios attack where the
   net does not reach: a slow drift (too gradual to trip anything) and a mid-run
   REGIME CHANGE (the surface's curvature jumps under a converged controller).

3. THE BUDGET WAS UNREACHABLE. This item used to read "the target was too shallow
   -- DEEP_FRAC pushes it to 60% of the oracle's run". That fix inverted the
   defect it was fixing. `steps_to_target` can only return a value in
   [window, steps], so the largest FINITE ratio is ~1/frac; at frac=0.60 the
   ceiling is 1.63, BELOW the 2.0 BUDGET, and every "% over budget" printed was
   identically "% never converged". DEEP_FRAC is back at 0.25 (ceiling 4.0),
   where 2.0 is a live threshold, and `_oracle_task` asserts it per cell.

`var_cond` IS NOT IN `CELLS`. Its oracle converges 8.7x at 1500 steps, 17.4x at
4000 and 40x at 8000 -- failing the 100x cell guard at every length tried -- so
it was dropped rather than lengthened again.
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.fake_modeller import MK_DEV_ADAPTIVE
from bench.harness import BenchRun
from bench.oracle import (OracleResult, Surface, final_distance, find_oracle,
                          median_trace)
from bench.scenarios import SEED_LR, steps_to_target

BUDGET = 2.0
MIN_DROP = 100.0
# THE BUDGET MUST BE REACHABLE. steps_to_target can only return a value in
# [window, steps], so the largest FINITE ratio is steps/denom ~ 1/frac. At
# frac=0.60 that ceiling is 1.63, BELOW the 2.0 budget -- so every "% over
# budget" was identically "% never converged" and BUDGET could have been set to
# anything in [1.64, inf) without changing one number. That is the same defect
# the summary's section 7 claims to have fixed, reintroduced worse. frac=0.25
# puts the ceiling at 4.0 and makes 2.0 a live threshold. _oracle_task asserts
# it per cell so this cannot regress silently again.
DEEP_FRAC = 0.25

# `bias` was introduced as "the blind ramp's own per-step rate, ln(1.682)/50 --
# what the servo applies on an unresolved reading" -- so `hyper+ramp` is "measure
# when the signal is there, ramp when it is not", the ray probe's architecture in
# a free sensor.
#
# HALF THAT PROVENANCE CHECKS OUT AND HALF DOES NOT. The per-FIRING multiplier is
# real: on an unresolved reading the servo applies (grid_top/alpha_target)^eta_up
# = (32/4)^0.25 = 1.682 (harness._ramp_tick), and ln(1.682) = 0.5199. The /50 is
# NOT the servo's clock. It is THIS BENCH's override -- oracle.py:77 sets
# ray_calibration.period to 50 so a 2000-step run gets enough firings -- while
# MK_DEV_RAYCAL ships period 500. The shipping per-step rate is ln(1.682)/500 =
# 0.00104, so this constant is 10x its stated source. Every result in
# docs/lr_control_summary.md was scored at 0.0104; read it as a chosen constant,
# not a derived one.
#
# The bias makes beta_down LOAD-BEARING for the first time: at noise 2 the brake
# attenuates to 0.04*0.29 = 0.0116 against a CONSTANT +0.0104, so it barely
# descends. `d8` doubles the brake to test whether that is what bites.
#
# TESTED, NOT RECOMMENDED. `hyper gated` -- the only arm that uses this -- now
# ties both unbiased hyper variants at 0.0% (summary section 0). The bias existed
# to survive an 8x-in-one-step regime change that does not happen.
RAMP_BIAS = 0.0104

ARMS = (
    # THE NULL ARM IS PERMANENT. Servo live -- warmup, bounds, tripwire, rewind --
    # and NO sensor. Any scenario column where this scores 0% is not testing a
    # controller: measured, `regime_change` and `hot_90pct` are pass-through, so
    # an arm that "fails" them is doing damage relative to standing still rather
    # than failing to track. Without this row that is invisible.
    ('NULL (no sensor)', 'none', 'none', None, {}),
    # THE HONEST DEFAULT: Baydin's rule, symmetric, nothing added. 0% at the
    # hardest noise cell; its whole deficit was regime_change, which is the
    # scenario that just got corrected.
    ('hyper sym', 'hyperx', 'none',
     {'hyper_beta': 0.02, 'hyper_beta_down': 0.02}, {}),
    ('hyper 2:1', 'hyperx', 'none',
     {'hyper_beta': 0.02, 'hyper_beta_down': 0.04}, {}),
    # the biased leader, carried to see whether a REALISTIC regime change still
    # justifies it -- both of its mechanisms exist to fix things that may not
    # happen at 1.5x over 5000 steps
    ('hyper gated', 'hyperx', 'none',
     {'hyper_beta_down': 0.08, 'hyper_bias': RAMP_BIAS,
      'hyper_bias_gate': True}, {}),
    ('ray+ray', 'ray', 'ray', None, {}),
    ('ramp+plateau', 'ramp', 'plateau', None, {}),
)

MLE = dict(dim=32, cond=300.0, noise=0.01, init_scale=3.0, quartic=0.0)
EQ = dict(dim=4, a=1.0, b=1.0, w_rep=0.7, w_bwd=0.3, kappa=0.05,
          noise=0.3, init_scale=1.0)
#: UNUSED since var_cond left CELLS -- no cell list in this file references it.
#: Flagged rather than deleted because a live-looking constant nothing reads is
#: the shape the "inert flags fail silently" bug takes here. bench/hard.py:43
#: has the same dict and DOES use it.
VC = dict(dim=16, n_cond=256, spread=50.0, noise=0.5)

#: (label, game, kwargs, steps, lr_grid, extra_args)
CELLS = [
    # THE INTERACTION CELLS -- the point of this run
    ('mle n2 q1e-2', 'mle', dict(MLE, noise=2.0, quartic=1e-2), 2000,
     (1e-6, 1e-1, 12), {}),
    ('mle n.5 q1e-2', 'mle', dict(MLE, noise=0.5, quartic=1e-2), 2000,
     (1e-6, 1e-1, 12), {}),
    ('mle n2 q1e-2 c1k', 'mle', dict(MLE, noise=2.0, quartic=1e-2, cond=1000.0),
     3000, (1e-6, 1e-1, 12), {}),
    # controls, so an interaction effect can be separated from its marginals
    ('mle n2', 'mle', dict(MLE, noise=2.0), 2000, (1e-6, 1e-1, 12), {}),
    ('mle q1e-2', 'mle', dict(MLE, quartic=1e-2), 2000, (1e-6, 1e-1, 12), {}),
    # the hard surfaces -- `equilibration` only. THERE IS NO var_cond CELL HERE
    # and this comment used to describe one, which is the failure mode to avoid:
    # var_cond converges ~n_cond/batch times slower than mle by construction --
    # each condition's level is only touched every 8 steps at n_cond 256 /
    # batch 32 -- so 1500 steps gave an 8.7x drop, 4000 gave 17x and 8000 gave
    # 40x, all failing the guard. The proposed repair (n_cond 128 at 12000 steps,
    # ~47 effective passes per condition, staleness still on) was never added.
    ('eq noisy', 'equilibration', dict(EQ, noise=1.0), 3000, (1e-4, 1e1, 12),
     {'lr_flow': 1.0}),
]

SCENARIOS = ('cold_start', 'drift_10x', 'regime_change', 'hot_90pct',
             'mixture_drift')

#: HELD-OUT CELLS -- none of them were looked at while choosing hyper's bias,
#: gate or brake. `hyper gated` reaching 0.2% on CELLS above is in-sample: the
#: gate was derived from `mle q1e-2` failing at 29%, which is in that set. A
#: constant selected on the same cells it is scored on is a fit, not a result,
#: so the number that counts is this one.
#:
#: Same scoring throughout -- deep target, all four scenarios, per-cell oracle.
HELDOUT = [
    ('h baseline', 'mle', dict(MLE), 2000, (1e-6, 1e-1, 12), {}),
    ('h noise=0.1', 'mle', dict(MLE, noise=0.1), 2000, (1e-6, 1e-1, 12), {}),
    ('h noise=0.5', 'mle', dict(MLE, noise=0.5), 2000, (1e-6, 1e-1, 12), {}),
    ('h cond=30', 'mle', dict(MLE, cond=30.0), 2000, (1e-6, 1e-1, 12), {}),
    ('h cond=1000', 'mle', dict(MLE, cond=1000.0), 3000, (1e-6, 1e-1, 12), {}),
    ('h quartic1e-4', 'mle', dict(MLE, quartic=1e-4), 2000, (1e-6, 1e-1, 12), {}),
    # genuinely new: a much bigger moving target than anything used for tuning
    ('h quartic=0.1', 'mle', dict(MLE, quartic=1e-1), 2000, (1e-6, 1e-1, 12), {}),
    ('h n.1 q1e-1', 'mle', dict(MLE, noise=0.1, quartic=1e-1), 2000,
     (1e-6, 1e-1, 12), {}),
    # equilibration at its OWN noise (0.3), not the 1.0 cell used above
    ('h eq base', 'equilibration', dict(EQ), 3000, (1e-4, 1e1, 12),
     {'lr_flow': 1.0}),
    ('h eq w_rep.3', 'equilibration', dict(EQ, w_rep=0.3, w_bwd=0.7), 3000,
     (1e-4, 1e1, 12), {'lr_flow': 1.0}),
    # DIMENSION. E|cos| between independent vectors ~ sqrt(2/(pi*d)), so a
    # constant bias term outruns the beta*cos response as d grows: net drift is
    # +0.0047/step at d=32 and +0.0103 at d=1e5. `hyper gated` should therefore
    # degrade toward a blind ramp with width while `hyper z`, which divides by
    # that same null level, should not. d=32 is the bench's usual width and the
    # most favourable this design will ever see.
    ('h dim=256', 'mle', dict(MLE, dim=256), 2000, (1e-6, 1e-1, 12), {}),
    ('h dim=2048', 'mle', dict(MLE, dim=2048), 2000, (1e-6, 1e-1, 12), {}),
    ('h dim2048 n2', 'mle', dict(MLE, dim=2048, noise=2.0), 2000,
     (1e-6, 1e-1, 12), {}),
]

#: EQUILIBRATION, PUSHED. 11 of the 13 HELDOUT cells are ONE surface (`mle`, a
#: convex quadratic with an optional quartic) swept along four parameters, and
#: the two equilibration cells score 0% for every arm -- so the battery has
#: essentially no evidence from a multi-player surface. That matters because a
#: quadratic bowl has none of the properties that make TB hard: no trajectory, no
#: stochastic policy, one loss, one batch stream, one global optimum.
#:
#: These push the 3-player game along the axes that should actually bite:
#: gradient noise, coupling strength (a=b, the justified setting -- raising BOTH
#: raises the loop gain without inventing an asymmetry), buffer churn kappa,
#: width, and the flow rate the level chases at.
EQ_HARD = [
    ('eq n1', 'equilibration', dict(EQ, noise=1.0), 3000, (1e-4, 1e1, 12),
     {'lr_flow': 1.0}),
    ('eq n3', 'equilibration', dict(EQ, noise=3.0), 3000, (1e-4, 1e1, 12),
     {'lr_flow': 1.0}),
    # loop gain: a=b=3 triples the policy/level coupling both ways at once
    ('eq ab3', 'equilibration', dict(EQ, a=3.0, b=3.0), 3000, (1e-4, 1e1, 12),
     {'lr_flow': 1.0}),
    ('eq ab3 n1', 'equilibration', dict(EQ, a=3.0, b=3.0, noise=1.0), 3000,
     (1e-4, 1e1, 12), {'lr_flow': 1.0}),
    # a nearly-frozen buffer: mu tracks theta 25x more slowly
    ('eq kappa.002', 'equilibration', dict(EQ, kappa=0.002), 3000,
     (1e-4, 1e1, 12), {'lr_flow': 1.0}),
    ('eq dim64', 'equilibration', dict(EQ, dim=64), 3000, (1e-4, 1e1, 12),
     {'lr_flow': 1.0}),
    # the level chasing 10x faster -- mk_dev runs lr_flow 0.1 against a policy
    # rate ~1.25e-4, so a FAST level is the production regime, not an extreme
    ('eq flow10', 'equilibration', dict(EQ), 3000, (1e-4, 1e1, 12),
     {'lr_flow': 10.0}),
    ('eq ab3 flow10', 'equilibration', dict(EQ, a=3.0, b=3.0), 3000,
     (1e-4, 1e1, 12), {'lr_flow': 10.0}),
]


def _mk(cell, extra=None):
    _, game, kw, steps, grid, base_extra = cell
    return Surface(game, game, dict(kw), steps=steps, lr_grid=grid,
                   extra_args={**base_extra, **(extra or {})})


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _sc_drift(surface, oracle, seed, climber, braker, std, factor=10.0):
    """
    Walk peak_scale up by `factor` GRADUALLY over the middle half of the run.

    The blow-up scenario injects 100x in one step, which trips the divergence
    tripwire immediately -- so it scores the rewind, not the sensor. A slow drift
    never trips anything, so only a sensor can catch it. This is the cold-start
    failure's mirror image and nothing in the battery tested it.
    """
    run = surface.make(oracle.lr, seed=seed, servo=True, climber=climber,
                       braker=braker, standard=std)
    n, start = surface.steps, surface.steps // 4
    run.run(start, stop_on_divergence=False)
    per = factor ** (1.0 / max(n // 2, 1))
    for _ in range(n // 2):
        st = run.m.lr_ctrl
        st['peak_scale'] = float(st.get('peak_scale', 1.0)) * per
        run.step()
        if run.aborted or run.game.diverged():
            break
    if not (run.aborted or run.game.diverged()):
        run.run(n - start - n // 2, stop_on_divergence=False)
    return run


def _sc_regime(surface, oracle, seed, climber, braker, std, cell=None):
    """
    Change the SURFACE under a settled controller: rebuild the game mid-run with
    a different curvature, keeping parameters and optimizer state.

    Every scenario so far perturbs the CONTROLLER and asks it to recover a rate it
    already had right. This asks the opposite -- the rate was right and the
    problem moved -- which is what a stage transition or a schedule change
    actually does, and what a recorded ceiling cannot follow.
    """
    run = surface.make(oracle.lr, seed=seed, servo=True, climber=climber,
                       braker=braker, standard=std)
    at = surface.steps // 4
    span = min(REGIME_OVER, surface.steps - at)
    run.run(at, stop_on_divergence=False)
    for _ in range(span):
        _regime_shift(run.game, 1.0 / span)
        run.step()
        if run.aborted or run.game.diverged():
            return run
    return run.run(surface.steps - at - span, stop_on_divergence=False)


#: How far the optimum moves during a regime change, and over how long.
#: MK 2026-08-12: outside PHASE TRANSITIONS, real regime changes are about 1.5x
#: in LR over ~5000 steps. The original scenario softened curvature 8x IN ONE
#: STEP, which is ~2 orders of magnitude more violent in rate-of-change, and
#: every arm that lost only on that column lost to a scenario that does not
#: happen. Curvature scales as 1/lr, so a 1.5x LR move is a 1.5x curvature move.
REGIME_FACTOR = 1.5
REGIME_OVER = 5000          # steps; clipped to the run length by the callers


def _sc_mixture(surface, oracle, seed, climber, braker, std):
    """
    THE BRANCH MIXTURE MOVES MID-RUN -- w_rep 0.7/0.3 walking to 0.3/0.7 over the
    middle half, which is what the balance controller does in production.

    This is the scenario MK's standing objection to hypergradient needs: g_t and
    g_{t-1} are then gradients of DIFFERENT OBJECTIVES, so cos < 0 can mean "the
    step overshot" OR merely "the mixture changed underneath me", and the rule
    cannot tell them apart. Nothing in the battery has tested it -- `mle` has one
    loss and one batch stream so it cannot arise there, and the static w_rep cell
    changes the mixture BEFORE the run rather than during it.

    No-ops on surfaces without branch weights, which is every non-equilibration
    cell; those score it identically to a plain run.
    """
    run = surface.make(oracle.lr, seed=seed, servo=True, climber=climber,
                       braker=braker, standard=std)
    g = run.game
    if not (hasattr(g, 'w_rep') and hasattr(g, 'w_bwd')):
        return run.run(surface.steps, stop_on_divergence=False)
    n, start = surface.steps, surface.steps // 4
    span = n // 2
    w0, w1 = float(g.w_rep), 1.0 - float(g.w_rep)
    run.run(start, stop_on_divergence=False)
    for i in range(span):
        f = (i + 1) / span
        g.w_rep = w0 + (w1 - w0) * f
        g.w_bwd = 1.0 - g.w_rep
        run.step()
        if run.aborted or g.diverged():
            return run
    return run.run(n - start - span, stop_on_divergence=False)


def _regime_shift(game, frac=1.0):
    """
    Apply `frac` of the total regime change. Called once per step over the ramp
    so the surface DRIFTS rather than jumping -- a step change is a different
    problem (it makes the sensor's lag the whole story) and is not the one the
    real system poses.
    """
    f = REGIME_FACTOR ** frac
    if hasattr(game, 'H'):
        game.H = game.H / f
    elif hasattr(game, 'b'):
        game.b = float(game.b) * f
    elif hasattr(game, 'spread'):
        game.spread = float(game.spread) * f


def _regime_oracle(cell, seeds=(0, 1, 2)):
    """
    A SEPARATE oracle for the regime-change scenario.

    Scoring a run whose curvature was softened mid-flight against an oracle that
    never experienced the change measures the handicap, not the controller: the
    softened problem simply converges slower, so every arm reads 100% over budget
    -- which is what the first crucible pass showed and why it is not a result.

    So the denominator here is the best FIXED rate for a run that undergoes the
    same shift. Then a slowdown means "against the best you could have done on
    this composite problem", which is the question.
    """
    s = _mk(cell)
    lo, hi, n = cell[4]
    curve, runs_by_lr = {}, {}
    at = cell[3] // 4
    for lr in np.geomspace(lo, hi, int(n)):
        runs = []
        for seed in seeds:
            r = s.make(float(lr), seed=seed, servo=False)
            r.run(at, stop_on_divergence=False)
            span = min(REGIME_OVER, cell[3] - at)
            for _ in range(span):
                _regime_shift(r.game, 1.0 / span)
                r.step()
            r.run(cell[3] - at - span, stop_on_divergence=False)
            runs.append(r)
        runs_by_lr[float(lr)] = runs
        curve[float(lr)] = final_distance(runs)
    live = {k: v for k, v in curve.items() if math.isfinite(v)}
    if not live:
        return None
    best = min(live, key=live.get)
    trace = median_trace(runs_by_lr[best])
    o = OracleResult(cell[1], best, curve, trace, live[best], seeds)
    denom = steps_to_target(runs_by_lr[best][0], o, frac=DEEP_FRAC)
    drop = float(trace[0]) / max(float(trace[-1]), 1e-300)
    if not denom or drop < MIN_DROP:
        return None
    return o, denom


def _time_oracle(base, cell, seeds=(0, 1, 2)):
    """
    Re-select the reference rate on the metric it is actually the denominator
    FOR: time to the deep target, not final distance.

    WIRED IN at `_oracle_task` (see the call below). This docstring said "NOT
    WIRED IN. Nothing calls this" for as long as it WAS called, which is the
    same defect it was written to warn about, pointing the other way: a fix
    documented as absent is as misleading as an absence documented as a fix.
    Consequence to keep in mind when reading old numbers -- every result printed
    before it was wired in used the distance-selected denominator.

    `find_oracle` minimises the last-50 median distance. The crucible scores
    TIME. On flat-above-optimum surfaces those pick different rates, and badly:
    on `eq base` the fastest fixed rate to the deep target reaches it in ~190
    steps against the distance-optimal rate's ~1823, so the denominator was 10.9x
    too large. A "2x budget" against a 10.9x-too-slow baseline is really a 0.18x
    budget -- which is why every arm scored 0% on both equilibration cells and
    the surface looked undiscriminating when it was only mis-normalised.

    Keeps `find_oracle`'s trace (so the TARGET LEVEL is unchanged and still comes
    from a rate that genuinely converges) and swaps only the rate that sets the
    clock.
    """
    best, best_t = None, None
    for lr in sorted(base.curve):
        ts = [steps_to_target(cell_surface.run(lr, seed=sd, servo=False), base,
                              frac=DEEP_FRAC)
              for sd, cell_surface in ((sd, _mk(cell)) for sd in seeds)]
        ts = [t for t in ts if t]
        if len(ts) < len(seeds):
            continue                      # must reach it on EVERY seed
        t = float(np.median(ts))
        if best_t is None or t < best_t:
            best, best_t = lr, t
    return best, best_t


def _cold_start_feasible(oracle_lr, denom):
    """
    Can ANY rate-limited climber pass `cold_start` on this cell, or does the
    budget forbid it?

    THE MIRROR IMAGE OF THE UNREACHABLE BUDGET. `_oracle_task` already refuses a
    cell whose budget is too LOOSE to fail (`BUDGET*denom >= steps`, where "over
    budget" degenerates into "never converged"). Nothing checked the other end,
    and wiring in `_time_oracle` moved three cells straight through it: the
    time-optimal rate is much faster than the distance-optimal one, so `denom`
    collapsed -- `h eq base` went to 50 steps of a 3000-step run -- and the
    cold-start budget went with it.

    Two independent walls, both from the SHIPPING controller, not the bench:

      * `peak_scale` is bounded by `adaptive_lr.bounds` (mk_dev ships
        [0.01, 2000]). A cell needing more than 2000x the seed rate is
        unreachable at any speed, by any arm. `h eq w_rep.3` needs 9840x.
      * hypergradient's climb is capped at `exp(hyper_beta)` per step, so
        closing a gap of R takes at least `ln(R)/hyper_beta` steps -- 408 for
        `h eq base`, against a 100-step budget.

    Checked against the run of 2026-08-13: every cell where an arm scored 100%
    of cold starts is one this rejects, and every cell it accepts was passed by
    all three hyper variants. A 100% column on a rejected cell is a property of
    the budget, and reporting it as a controller score inflated every arm --
    including a hypothetical perfect one -- by about 4.6 points.
    """
    need = float(oracle_lr) / SEED_LR
    cap = float(MK_DEV_ADAPTIVE['bounds'][1])
    if need > cap:
        return False, (f'cold_start UNREACHABLE: needs peak_scale {need:.0f} '
                       f'> the controller cap {cap:.0f}')
    beta = float(BenchRun.STANDARD['hyper_beta'])
    climb = math.log(need) / beta
    if climb > BUDGET * denom:
        return False, (f'cold_start UNREACHABLE: >= {climb:.0f} steps to climb '
                       f'{need:.0f}x at exp({beta}) / step, vs a '
                       f'{BUDGET * denom:.0f}-step budget')
    return True, ''


def _oracle_task(cell):
    label = cell[0]
    surface = _mk(cell)
    try:
        oracle = find_oracle(surface, seeds=(0, 1, 2), verbose=False)
    except ValueError as e:
        return label, None, f'no usable oracle ({e})'
    drop = float(oracle.trace[0]) / max(float(oracle.trace[-1]), 1e-300)
    if drop < MIN_DROP:
        return label, None, (f'oracle converges only {drop:.3g}x in '
                             f'{surface.steps} steps -- trace too flat to time')
    # SELECT THE REFERENCE RATE ON THE METRIC IT IS THE DENOMINATOR FOR.
    # `find_oracle` minimises final DISTANCE; this battery scores TIME. On
    # flat-above-optimum surfaces they pick different rates and the gap is large:
    # on `eq base` the fastest fixed rate reaches the deep target in ~190 steps
    # against the distance-optimal rate's ~1823, so the denominator was 10.9x too
    # big and a "2x budget" was really a 0.18x budget -- which is why every arm
    # scored 0% on both equilibration cells. `_time_oracle` was written to fix
    # this and then never called; that is why it is wired in here explicitly.
    fast_lr, denom = _time_oracle(oracle, cell)
    if not denom:
        return label, None, 'no fixed rate reaches the deep target on every seed'
    oracle = OracleResult(oracle.surface, fast_lr, oracle.curve, oracle.trace,
                          oracle.final, oracle.seeds)
    # THE BUDGET MUST BE REACHABLE, asserted per cell. steps_to_target returns at
    # most `steps`, so if BUDGET*denom >= steps then "over budget" is identically
    # "never converged" and BUDGET is inert -- the defect that invalidated three
    # earlier rankings. This assert is the thing that stops it recurring.
    if BUDGET * denom >= surface.steps:
        return label, None, (f'BUDGET {BUDGET} unreachable: {denom:.0f} x '
                             f'{BUDGET} >= {surface.steps} steps')
    reg = _regime_oracle(cell)
    return label, (oracle, denom, drop, reg,
                   _cold_start_feasible(oracle.lr, denom)), None


def _arm_task(item):
    cell, oracle, denom, reg, arm, seeds = item
    name, climber, braker, std, extra = arm
    s = _mk(cell, extra)
    per, allslow = [], []
    for sc in SCENARIOS:
        slow = []
        for seed in seeds:
            if sc == 'drift_10x':
                run = _sc_drift(s, oracle, seed, climber, braker, std)
            elif sc == 'regime_change':
                run = _sc_regime(s, oracle, seed, climber, braker, std)
            elif sc == 'mixture_drift':
                run = _sc_mixture(s, oracle, seed, climber, braker, std)
            elif sc == 'cold_start':
                run = s.run(SEED_LR, seed=seed, servo=True, climber=climber,
                            braker=braker, standard=std)
            else:
                run = s.run(oracle.hot_lr(0.9), seed=seed, servo=True,
                            climber=climber, braker=braker, standard=std)
            ref, ref_denom = ((reg[0], reg[1]) if sc == 'regime_change' and reg
                              else (oracle, denom))
            t = steps_to_target(run, ref, frac=DEEP_FRAC)
            slow.append(math.inf if t is None else t / ref_denom)
        per.append(float((np.array(slow) > BUDGET).mean()))
        allslow.extend(slow)
    arr = np.array(allslow, dtype=float)
    live = arr[np.isfinite(arr)]
    return dict(cell=cell[0], arm=name, n=len(arr),
                over=float((arr > BUDGET).mean()),
                med=float(np.median(live)) if len(live) else math.inf,
                p90=float(np.percentile(live, 90)) if len(live) else math.inf,
                per=per)


def main(seeds=20, workers=None, cells=None):
    cells = cells or CELLS
    seeds = tuple(range(int(seeds)))
    workers = int(workers or max(2, min(20, (os.cpu_count() or 4) - 4)))
    print(f'{"=" * 88}\nCRUCIBLE -- {len(seeds)} seeds, budget {BUDGET:g}x, '
          f'deep target at {DEEP_FRAC:.0%} of the oracle run, {workers} workers'
          f'\n{"=" * 88}\n')

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        oracles = {}
        for label, got, why in pool.map(_oracle_task, cells):
            if got is None:
                print(f'  {label:<18} SKIPPED -- {why}')
            else:
                oracles[label] = got
        print()
        jobs = [(c, oracles[c[0]][0], oracles[c[0]][1], oracles[c[0]][3],
                 arm, seeds)
                for c in cells if c[0] in oracles for arm in ARMS]
        rows = list(pool.map(_arm_task, jobs))

    agg = {}
    for c in cells:
        label = c[0]
        if label not in oracles:
            continue
        oracle, denom, drop, reg, (feasible, why) = oracles[label]
        print(f'  {label:<18} oracle lr {oracle.lr:.3g}  drop {drop:.3g}x  '
              f'deep target at {denom} of {c[3]} steps'
              + ('' if reg else '   [regime cell has NO oracle -- scored vs base]'))
        if not feasible:
            print(f'    >> {why}\n       That column is a property of the budget, '
                  f'not of any arm -- excluded from the second aggregate below.')
        print(f'    {"arm":<14} {"%over":>7} {"med":>7} {"p90":>7}   '
              f'{list(SCENARIOS)}')
        for r in [r for r in rows if r['cell'] == label]:
            a = agg.setdefault(r['arm'], {'over': 0.0, 'n': 0, 'worst': 0.0,
                                          'cell': '-', 'fover': 0.0, 'fn': 0})
            a['over'] += r['over'] * r['n']
            a['n'] += r['n']
            # ... and again over the scenarios an arm could actually have passed
            per_n = r['n'] / len(SCENARIOS)
            for sc, p in zip(SCENARIOS, r['per']):
                if sc == 'cold_start' and not feasible:
                    continue
                a['fover'] += p * per_n
                a['fn'] += per_n
            if r['over'] > a['worst']:
                a['worst'], a['cell'] = r['over'], label
            med = f'{r["med"]:.2f}' if math.isfinite(r['med']) else 'never'
            print(f'    {r["arm"]:<14} {r["over"]:>6.0%} {med:>7} {r["p90"]:>7.2f}'
                  f'   {[f"{p:.0%}" for p in r["per"]]}')
        print()

    print(f'{"=" * 88}\nACROSS EVERY MEASURABLE CELL\n{"=" * 88}')
    print(f'  {"arm":<14} {"%over budget":>13} {"passable only":>14} '
          f'{"worst cell":>12}  where')
    for name, a in sorted(agg.items(),
                          key=lambda kv: (kv[1]['over'] / max(kv[1]['n'], 1),
                                          kv[1]['worst'])):
        print(f'  {name:<14} {a["over"] / max(a["n"], 1):>12.1%} '
              f'{a["fover"] / max(a["fn"], 1):>13.1%} '
              f'{a["worst"]:>11.0%}  {a["cell"]}')
    # THE SECOND COLUMN IS THE ONE THAT SCORES CONTROLLERS. The first includes
    # cold starts the budget forbids anyone from passing, which adds the same
    # ~4.6 points to every arm and compresses the differences that matter.
    print('\n  "passable only" drops the cold_start column on cells where the '
          'budget makes it\n  unreachable for ANY arm (see the per-cell notes). '
          'Read that column, not the first.')
    return rows


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20,
         int(sys.argv[2]) if len(sys.argv) > 2 else None,
         EQ_HARD if 'eqhard' in sys.argv else
         (HELDOUT if 'heldout' in sys.argv else None))
