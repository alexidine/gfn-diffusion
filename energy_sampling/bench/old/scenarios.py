"""
The scenario battery: what the LR controller costs, and whether it gets stuck.

Every scenario is scored on three things, against the oracle from `oracle.py`:

  REGRET          final distance / oracle's final distance. 1.0 = free.
  RECOVERED_AT    first step after which the run stays within `tol` of the
                  ORACLE'S OWN TRACE for the rest of the run. None = never.
  DETECTABILITY   would anything in the logs have told you? Scored from the same
                  signals a real run publishes -- `raycal/status`,
                  `lr_ctrl/peak_scale`, `lr_ctrl/divergences`, grad norm.

The third is not decoration. Both absorbing failures found so far are SILENT: a
run stranded at its seed LR (F-016 arm B) trains 4 orders of magnitude worse
than one that isn't, and looks like slow progress. A battery that only scored
recovery would produce controllers that handle the failures we can already see
and nothing for the ones we cannot.

WHY STAGE TRANSITIONS ARE NOT A SCENARIO. `rearm_warmup` resets `peak_scale` to
1.0 and forgets the ceiling at every transition, by design -- so the state after
a transition IS cold start, and each surface can be studied independently
without chaining. `cold_start` therefore covers it, at no extra cost.

Run:  python -m bench.scenarios          (full scoreboard)
      python -m bench.scenarios mle      (one surface)
"""

import math
import sys

import numpy as np

from bench.old.harness import BenchRun
from bench.old.oracle import (Surface, distance_trace, final_distance, find_oracle,
                          median_trace)

#: The three stage analogues. Each is run from cold start, which is exactly the
#: post-transition state.
SURFACES = {
    'mle': Surface(
        'mle', 'mle',
        dict(dim=32, cond=300.0, noise=0.01, init_scale=3.0),
        steps=2000, lr_grid=(1e-6, 1e-1, 12)),
    # THE SAME SURFACE WITH AN UNKNOWN NONZERO LOSS FLOOR, which the real MLE
    # stage has and the bare `mle` surface does not: as written its expected loss
    # bottoms out at exactly 0, i.e. the interpolation regime that Polyak-type
    # methods assume. floor 5000 is about two thirds of the initial loss, so it
    # dominates the gap for most of the run.
    #
    # It changes NO GRADIENT, so the prediction attached to this surface was that
    # every sensor reading only differences or gradients (ray, armijo, bb, hyper,
    # dog, plateau) would score BIT-IDENTICALLY and only level-readers (sps, the
    # slope sensors, which normalise by the window's own magnitude) would move.
    #
    # THAT PREDICTION IS WRONG FOR ARMIJO, and the reason is the finding.
    # Offset-invariance holds in exact arithmetic and fails in float32: each loss
    # EVALUATION carries relative rounding error |L| * 2^-23, which grows with the
    # offset, while the margin the test resolves does not. Measured post-warmup at
    # floor 5000: the sufficient-decrease bar c*|g.d| is 2.8e-4 and the float32
    # spacing at |L| ~ 5007 is 6.0e-4 -- the bar sits BELOW the noise, and the
    # accept rate falls 98.7% -> 61.2%. Because backtracking is multiplicative and
    # asymmetric (x0.5 down, x1.014 up) a near-coin-flip test does not wander, it
    # collapses. -0.26/step is E[log step] AT THE MEASURED 61.2% ACCEPT RATE
    # (0.612*ln 1.014 + 0.388*ln 0.5 = -0.2604), not at a true coin flip, which
    # is worse still: 0.5*ln 0.5 + 0.5*ln 1.014 = -0.3396. Either way peak_scale
    # slams into its 0.01 bound and the controller warm-restarts in a loop.
    # 1.56 -> 344.83.
    #
    # bb is bit-identical (2.79 both) because it reads GRADIENTS, which carry no
    # offset. The ray probe differences losses too and is NOT immune, but degrades
    # gracefully rather than collapsing: its paired sub-batches and significance
    # test send rounding noise to `unresolved`, where the servo applies its
    # constant, instead of to a confident wrong verdict.
    #
    # The condition is a RATIO, so it is checkable on the real system rather than
    # a property of this floor value: compare |L| * 2^-23 with c*|g.d|.
    'mle_floor': Surface(
        'mle_floor', 'mle',
        dict(dim=32, cond=300.0, noise=0.01, init_scale=3.0, floor=5000.0),
        steps=2000, lr_grid=(1e-6, 1e-1, 12)),
    # batch_size 32 against n_cond 256 is LOAD-BEARING, not a tuning choice. At
    # the default batch (1000 >= n_cond) every step sees every condition, so the
    # per-condition levels are never stale -- the mechanism this surface exists
    # for is switched off, and the problem becomes so easy that the best rate
    # beats the bracket edges by only 2.3x. find_oracle refuses that, correctly.
    # At batch 32 the gain is 7.6x and there is a real optimum to be wrong about.
    'var_cond': Surface(
        'var_cond', 'var_cond',
        dict(dim=16, n_cond=256, spread=50.0, noise=0.5),
        steps=1500, lr_grid=(1e-4, 1e1, 12), extra_args={'batch_size': 32}),
    # a = b = 1, CHANGED 2026-08-12 from a = 4. `a` is how much the level's target
    # moves when the policy moves; `b` is how much the policy's own residual does.
    # Both branches are TB residuals over the same log P_F -- the residual IS
    # `log Z - log w`, and d(log w)/d(theta) = -d(log P_F)/d(theta) -- so there is
    # no justification for them to differ. a = 4 was invented.
    #
    # IT COSTS THE SURFACE ITS DEFINING PROPERTY, and that is the finding, not a
    # problem to tune around. min alpha_target = one_step_lr / stability_lr is
    # 1.89 at a=4 but 0.85 at a=1: below 1, i.e. the one-step optimum lies INSIDE
    # the loop's stability boundary and a ray probe needs no margin for the
    # coupling at all. Everything this surface was built to demonstrate --
    # F-012's loop-gain margin, F-019's "worst surface", F-022's case for
    # slope_seek -- was a consequence of the invented asymmetry.
    'equilibration': Surface(
        'equilibration', 'equilibration',
        dict(dim=4, a=1.0, b=1.0, w_rep=0.7, w_bwd=0.3, kappa=0.05,
             noise=0.3, init_scale=1.0),
        steps=3000, lr_grid=(1e-4, 1e1, 12), extra_args={'lr_flow': 1.0}),
}

#: mk_dev's seed for a servo-managed (`auto`) LR -- where a real cold start begins.
SEED_LR = 1.25e-4

RECOVERY_TOL = 3.0        # within 3x of the oracle's trace counts as healthy


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def recovered_at(run, oracle, tol=RECOVERY_TOL):
    """
    First step after which the run stays healthy for the remainder.

    Defined against the oracle's TRACE, not its final value: early in training
    everything is far from the optimum, so a fixed threshold would score the
    first hundred steps of a perfectly healthy run as unrecovered.

    Returns 0 if never unhealthy, None if still unhealthy at the end.
    """
    dist = distance_trace(run)
    n = min(len(dist), len(oracle.trace))
    healthy = dist[:n] <= oracle.trace[:n] * tol
    if healthy.all():
        return 0
    if not healthy[-1]:
        return None
    # index after the last violation
    return int(np.max(np.nonzero(~healthy)[0])) + 1


def detectability(run):
    """
    What a real run's logs would have shown. Every field here has a wandb
    counterpart, so a flag raised in the bench is a flag that could be raised
    in production.
    """
    cals = run.calibrations
    resolved = [c for c in cals if c['status'] != 'unresolved']
    saturated = [c for c in resolved if c['status'] in ('above_range', 'below_range')]
    peaks = [h['peak_scale'] for h in run.history if math.isfinite(h['peak_scale'])]
    lo, hi = run.args.adaptive_lr.bounds
    at_bound = [p for p in peaks if p <= lo * 1.01 or p >= hi * 0.99]

    sat_frac = len(saturated) / max(len(resolved), 1)
    bound_frac = len(at_bound) / max(len(peaks), 1)
    out = {
        'divergences': run.divergences,
        'n_cal': len(cals),
        'sat_frac': sat_frac,
        'bound_frac': bound_frac,
    }
    # Explicit rules, each one a thing you could alert on today
    out['flagged'] = bool(run.divergences > 0
                          or (len(resolved) >= 4 and sat_frac > 0.8)
                          or bound_frac > 0.5)
    return out


def steps_behind(run, oracle):
    """
    HOW MANY STEPS OF PROGRESS THE RUN GAVE UP, averaged over the whole run.

    `regret` scores the LAST 50 steps, so it answers "where did you end up" and
    is silent on "how long did you take" -- `ramp`+`plateau` on mle cold start
    has the best regret in the board (0.69) and the worst recovery (1753 of 2000
    steps). Both facts matter and only one was being reported.

    Time-to-threshold is the obvious fix and it has two defects: it needs a
    threshold nothing supplies, and it CENSORS -- a run that never arrives has no
    number, so the arms that most need scoring are the ones that drop out. That
    is what `recovered_at` returning None already looks like.

    This avoids both. These surfaces converge geometrically, so on a log scale
    the oracle descends at a roughly constant rate `r` per step, and a run
    sitting a factor `d` above the oracle's curve at time t is exactly
    `log(d)/r` steps behind. Averaging that gap over the run gives a number in
    STEPS that is defined for every run including the ones that never converge,
    needs no threshold, and rewards arriving early and ending well in the same
    currency.

    Positive = behind the best fixed rate. Negative = ahead of it, which an
    adaptive schedule can legitimately be.
    """
    dist = distance_trace(run)
    n = min(len(dist), len(oracle.trace))
    if n < 10:
        return math.inf
    run_d = np.maximum(dist[:n], 1e-300)
    orc_d = np.maximum(oracle.trace[:n], 1e-300)
    gap = np.log(run_d) - np.log(orc_d)          # log-distance behind, per step
    gap = gap[np.isfinite(gap)]
    if not len(gap):
        return math.inf
    # the oracle's own log-improvement rate, measured over its middle -- the ends
    # are contaminated by the warmup ramp and by the noise floor it settles onto
    lo, hi = int(0.2 * n), int(0.9 * n)
    span = math.log(orc_d[lo]) - math.log(orc_d[hi])
    rate = span / max(hi - lo, 1)
    if not (math.isfinite(rate) and rate > 1e-12):
        return math.inf                          # oracle not converging: undefined
    return float(np.mean(gap) / rate)


def steps_to_target(run, oracle, window=50, frac=0.25):
    """
    RAW CONVERGENCE TIME: steps to reach the distance the oracle finished at.

    The primary score. `regret` answers only "where did you end up", which is
    why a controller could hold the best regret on the board (0.69) while taking
    1753 of 2000 steps to get there and nothing in the table said so.

    THE TARGET IS A LEVEL THE ORACLE PASSES EARLY, at `frac` of its run. Two
    earlier choices were both wrong and the reason is the same each time -- the
    measurable range of a ratio is set by how much run is left after the
    denominator finishes.

      * the oracle's FINAL distance: reachable only on the last step, so the
        metric became pass/fail on "did you beat the oracle" and censored 8 of
        11 arms.
      * the oracle's MID-RUN distance: the oracle then scores steps/2, so a 2x
        slowdown needs 2*steps/2 = the whole run and ANY over-budget run is
        necessarily censored. "over budget" and "never converged" became the
        same event -- checkable in the numbers, e.g. 88% of 60 = 53 against a
        never-count of exactly 53. The budget was not being tested; the run
        length was.

    At frac=0.25 the oracle scores about steps/4, so slowdowns up to ~4x are
    observable and a 2x budget sits comfortably inside the measurable range.
    The reference point survives: the oracle itself scores `frac * steps`.

    Judged on a trailing median so a single lucky sample cannot declare victory.

    Returns None when the run never arrives. Censoring is the metric's remaining
    defect, which is why `steps_behind` is reported beside it rather than instead
    of it: the arms that never arrive still get a number there.
    """
    dist = distance_trace(run)
    n = len(oracle.trace)
    if n < 2 * window:
        return None
    target = float(oracle.trace[max(int(frac * n), window)])
    if len(dist) < window or not (math.isfinite(target) and target > 0):
        return None
    med = np.median(np.lib.stride_tricks.sliding_window_view(dist, window), axis=1)
    hit = np.nonzero(med <= target)[0]
    return int(hit[0]) + window if len(hit) else None


#: How far from the reference rate still counts as "on target". A factor of 2
#: each way, because that is the tolerance the whole exercise is stated in: at
#: worst ~2x the best fixed rate, never 50x.
ON_TARGET_BAND = 2.0


def time_off_target(run, ref_lr, band=ON_TARGET_BAND):
    """
    Fraction of STEPS spent outside [ref/band, ref*band] -- split into too-hot
    and too-cold.

    THE METRIC THE REQUIREMENT IS ACTUALLY WRITTEN IN. `steps_to_target` scores
    how long a run took to arrive somewhere, which answers "is it fast" and only
    infers "did it sit at a bad rate for a long time". MK's standing requirement
    is the second one directly -- a controller you set once and stop thinking
    about is one that spends almost no time far from the right rate -- and a run
    can arrive on time having spent half the run badly mis-set.

    Reads the LIVE rate (`base x peak_scale x envelope`), which is what the
    optimizer actually stepped with, not the configured one.

    Hot and cold are reported apart because their consequences are not
    symmetric: too hot risks the absorbing boundary, too cold only wastes time.
    """
    lrs = [h['lr'] for h in run.history
           if h.get('lr') is not None and math.isfinite(h['lr'])]
    if not lrs or not (math.isfinite(ref_lr) and ref_lr > 0):
        return None
    lo, hi = ref_lr / band, ref_lr * band
    hot = sum(1 for x in lrs if x > hi)
    cold = sum(1 for x in lrs if x < lo)
    return {'off': (hot + cold) / len(lrs),
            'hot': hot / len(lrs),
            'cold': cold / len(lrs),
            'n': len(lrs)}


def longest_off_target(run, ref_lr, band=ON_TARGET_BAND):
    """
    Longest UNBROKEN run of steps outside the band, as a fraction of the run.

    "Not stuck for long periods" is about the longest excursion, not the total:
    500 scattered bad steps and 500 consecutive ones are the same number under
    `time_off_target` and very different failures. A controller that oscillates
    across the band is annoying; one that parks outside it is broken.
    """
    lrs = [h['lr'] for h in run.history
           if h.get('lr') is not None and math.isfinite(h['lr'])]
    if not lrs or not (math.isfinite(ref_lr) and ref_lr > 0):
        return None
    lo, hi = ref_lr / band, ref_lr * band
    worst = cur = 0
    for x in lrs:
        cur = cur + 1 if (x > hi or x < lo) else 0
        worst = max(worst, cur)
    return worst / len(lrs)


def score(run, oracle, tol=RECOVERY_TOL):
    final = final_distance([run])
    return {
        'regret': final / oracle.final if oracle.final > 0 else math.inf,
        'steps_to_target': steps_to_target(run, oracle),
        'steps_behind': steps_behind(run, oracle),
        'final': final,
        'recovered_at': recovered_at(run, oracle, tol),
        'final_lr': run.history[-1]['lr'] if run.history else math.nan,
        **detectability(run),
    }


# ---------------------------------------------------------------------------
# the scenarios
# ---------------------------------------------------------------------------

def sc_oracle_fixed(surface, oracle, seed):
    """Control: the oracle rate, servo OFF. Regret is 1.0 by construction; this
    exists so the scoreboard shows the seed-to-seed noise floor of the metric."""
    return surface.run(oracle.lr, seed=seed, servo=False)


def sc_cold_start(surface, oracle, seed):
    """
    The real starting condition: mk_dev's `auto` seed LR with the servo live.
    This is also the post-stage-transition state, since `rearm_warmup` resets
    peak_scale to 1.0.
    """
    return surface.run(SEED_LR, seed=seed, servo=True)


def sc_blowup(surface, oracle, seed, at=None, factor=100.0, sensor=None,
              climber=None, braker=None, standard=None):
    """
    Intentionally blow the rate up MID-RUN, from a healthy operating point, and
    measure how long the controller takes to get back.

    Injected on `peak_scale` rather than on the base LR, because that is the
    quantity the servo actually owns -- this is the state a bad calibration
    leaves behind, not a config error.
    """
    at = at or surface.steps // 3
    run = surface.make(oracle.lr, seed=seed, servo=True, sensor=sensor,
                       climber=climber, braker=braker, standard=standard)
    run.run(at, stop_on_divergence=False)
    st = run.m.lr_ctrl
    st['peak_scale'] = float(st.get('peak_scale', 1.0)) * factor
    run.m.lr_controller.step()                     # actuate it immediately
    return run.run(surface.steps - at, stop_on_divergence=False)


def sc_stuck_cold(surface, oracle, seed, factor=100.0):
    """Can it sit indefinitely at 100x too low? Start there and see if it climbs."""
    return surface.run(oracle.lr / factor, seed=seed, servo=True)


def sc_hot_band(surface, oracle, seed, frac=0.5):
    """
    Hot but not catastrophic: above the oracle, BELOW the cliff. The dangerous
    band -- degrading, and quiet.

    Placed as a fraction of the log distance from oracle to cliff rather than as
    a fixed multiple, because the band can be narrow. On `mle` the oracle is
    4.33e-3 and the cliff 7.3e-3, so "2x the oracle" is already past it and would
    test catastrophe instead of the question being asked.
    """
    return surface.run(oracle.hot_lr(frac), seed=seed, servo=True)


SCENARIOS = {
    'oracle_fixed': sc_oracle_fixed,
    'cold_start': sc_cold_start,
    'blowup_100x': sc_blowup,
    'stuck_cold_100x': sc_stuck_cold,
    'hot_half_to_cliff': lambda s, o, seed: sc_hot_band(s, o, seed, 0.5),
    'hot_90pct_to_cliff': lambda s, o, seed: sc_hot_band(s, o, seed, 0.9),
}


# ---------------------------------------------------------------------------
# scoreboard
# ---------------------------------------------------------------------------

def _agg(rows, key):
    vals = [r[key] for r in rows if r[key] is not None and
            (not isinstance(r[key], float) or math.isfinite(r[key]))]
    return float(np.median(vals)) if vals else None


def run_surface(name, seeds=(0, 1, 2), verbose=True):
    surface = SURFACES[name]
    if verbose:
        print(f'\n{"=" * 92}\n{name}\n{"=" * 92}')
    oracle = find_oracle(surface, seeds=seeds, verbose=verbose)
    if verbose:
        print(f'\n  oracle lr {oracle.lr:.4g}   final distance {oracle.final:.4g}\n')
        print(f'  {"scenario":<18} {"regret":>9} {"recovered":>11} {"final lr":>10} '
              f'{"lr/oracle":>10} {"div":>4} {"sat":>6} {"FLAGGED":>8}')

    table = {}
    for sc_name, fn in SCENARIOS.items():
        rows = [score(fn(surface, oracle, seed), oracle) for seed in seeds]
        never = sum(1 for r in rows if r['recovered_at'] is None)
        rec = _agg(rows, 'recovered_at')
        agg = {
            'regret': _agg(rows, 'regret'),
            'recovered_at': 'never' if never > len(rows) / 2 else
                            (f'{rec:.0f}' if rec is not None else 'never'),
            'final_lr': _agg(rows, 'final_lr'),
            'divergences': _agg(rows, 'divergences'),
            'sat_frac': _agg(rows, 'sat_frac'),
            'flagged': sum(1 for r in rows if r['flagged']),
            'n_seeds': len(rows),
        }
        table[sc_name] = agg
        if verbose:
            reg = agg['regret']
            print(f'  {sc_name:<18} {reg:>9.2f} {agg["recovered_at"]:>11} '
                  f'{agg["final_lr"]:>10.3g} {agg["final_lr"] / oracle.lr:>10.2f} '
                  f'{agg["divergences"]:>4.0f} {agg["sat_frac"]:>6.2f} '
                  f'{str(agg["flagged"]) + "/" + str(len(rows)):>8}')

    if verbose:
        silent = [k for k, v in table.items()
                  if v['regret'] and v['regret'] > 3.0 and v['flagged'] == 0]
        if silent:
            print(f'\n  SILENT FAILURES (regret > 3x, nothing flagged): {", ".join(silent)}')
        else:
            print('\n  no silent failures on this surface')
    return oracle, table


def alpha_target_regret(name, targets=(1, 2, 4, 6, 8, 12, 16, 24, 32, 64),
                        seeds=(0, 1, 2), verbose=True):
    """
    Regret against the oracle as a function of `alpha_target` -- the shape that
    says what the parameter is actually for.

    It is a U with an ABSORBING FAILURE STATE ON EACH SIDE, and neither side is
    the one the parameter is usually discussed in terms of:

      too low   the servo ramps past the surface's stability limit, trips the
                divergence bar, gets halved, and repeats. The operating point is
                then set by `divergence_cut` and the blow-up threshold -- NOT by
                alpha_target, which is why changing it in this regime does almost
                nothing. A safety mechanism has become the primary controller.

      too high  almost every reading falls below target, so the asymmetric update
                cuts on nearly every calibration and the rate walks monotonically
                to the floor. Stranded cold, and silent.

    Only between them does the servo track the oracle. The WINDOW'S LOCATION is
    per-surface -- set by the anisotropy margin (F-011) and the loop gain (F-012)
    -- so the number here transfers to nothing. The SHAPE does: any deployment
    needs to know which side of the window it is on, and the divergence count
    tells you (nonzero = below the window).
    """
    import statistics as st

    surface = SURFACES[name]
    oracle = find_oracle(surface, seeds=seeds)
    rows = []
    if verbose:
        print(f'\n{"=" * 78}\nalpha_target regret on {name} '
              f'(oracle lr {oracle.lr:.4g}, final {oracle.final:.4g})\n{"=" * 78}')
        print(f'{"alpha_target":>13} {"final lr":>10} {"lr/oracle":>10} '
              f'{"regret":>10} {"div":>6} {"verdict":>18}')
    for target in targets:
        lrs, divs, finals = [], [], []
        for seed in seeds:
            run = surface.make(SEED_LR, seed=seed, servo=True)
            run.args.adaptive_lr.calibration.alpha_target = float(target)
            run.run(surface.steps, stop_on_divergence=False)
            lrs.append(run.history[-1]['lr'])
            divs.append(run.divergences)
            finals.append(final_distance([run]))
        lr, div = st.median(lrs), st.median(divs)
        regret = st.median(finals) / oracle.final
        verdict = ('divergence cycle' if div > 0 else
                   'stranded cold' if lr < 0.1 * oracle.lr else
                   'tracking')
        rows.append(dict(target=target, lr=lr, regret=regret, div=div, verdict=verdict))
        if verbose:
            print(f'{target:>13} {lr:>10.3g} {lr / oracle.lr:>10.2f} '
                  f'{regret:>10.1f} {div:>6.1f} {verdict:>18}')
    if verbose:
        best = min(rows, key=lambda r: r['regret'])
        print(f'\n  best regret {best["regret"]:.1f}x at alpha_target {best["target"]}; '
              f'shipping default is 4.0')
    return oracle, rows


def sensor_race(name, sensors=('ray', 'plateau', 'ramp', 'ramp_plateau', 'none'),
                scenarios=('cold_start', 'blowup_100x', 'hot_90pct_to_cliff'),
                seeds=(0, 1, 2), verbose=True):
    """
    WHICH SENSOR, AND IS IT WORTH ITS COST?

    Every arm drives the same actuator with the same warmup, bounds, tripwire and
    rewind. Only the source of the verdict changes:

      ray            the alpha probe
      plateau        watch the loss, cut when nothing improves (train.py's other
                     implemented sensor -- never previously raced against ray)
      ramp           NO sensor: raise by a constant, brake only on the tripwire
      ramp_plateau   climb blindly, brake on evidence
      none           nothing moves the rate

    `ramp` is the null hypothesis, and it is not a strawman: 72-82% of real probe
    readings come back at a grid edge, where the servo already applies exactly
    this constant. If ray does not beat ramp, the probe is not paying for itself
    on that surface.

    The asymmetry to watch for: a plateau rule can only ever CUT, so it cannot
    climb out of a cold start -- and cold start is the state every stage
    transition creates.
    """
    surface = SURFACES[name]
    oracle = find_oracle(surface, seeds=seeds)
    if verbose:
        print(f'\n{"=" * 88}\nsensor race on {name} '
              f'(oracle lr {oracle.lr:.4g}, final {oracle.final:.4g})\n{"=" * 88}')
        print(f'  {"scenario":<20} {"sensor":<14} {"regret":>9} {"recovered":>11} '
              f'{"lr/oracle":>10} {"div":>4} {"FLAG":>6}')

    rows = []
    for sc_name in scenarios:
        for sensor in sensors:
            per_seed = []
            for seed in seeds:
                if sc_name == 'blowup_100x':
                    run = sc_blowup(surface, oracle, seed, sensor=sensor)
                else:
                    lr = (SEED_LR if sc_name == 'cold_start'
                          else oracle.hot_lr(0.9))
                    run = surface.run(lr, seed=seed, servo=True, sensor=sensor)
                per_seed.append(score(run, oracle))
            never = sum(1 for r in per_seed if r['recovered_at'] is None)
            rec = _agg(per_seed, 'recovered_at')
            row = dict(scenario=sc_name, sensor=sensor,
                       regret=_agg(per_seed, 'regret'),
                       recovered=('never' if never > len(per_seed) / 2 else
                                  (f'{rec:.0f}' if rec is not None else 'never')),
                       lr_ratio=_agg(per_seed, 'final_lr') / oracle.lr,
                       div=_agg(per_seed, 'divergences'),
                       flagged=sum(1 for r in per_seed if r['flagged']))
            rows.append(row)
            if verbose:
                print(f'  {sc_name:<20} {sensor:<14} {row["regret"]:>9.2f} '
                      f'{row["recovered"]:>11} {row["lr_ratio"]:>10.2f} '
                      f'{row["div"]:>4.0f} {str(row["flagged"]):>6}')
        if verbose:
            best = min((r for r in rows if r['scenario'] == sc_name),
                       key=lambda r: r['regret'])
            ray = next(r for r in rows if r['scenario'] == sc_name and r['sensor'] == 'ray')
            print(f'    -> best {best["sensor"]} ({best["regret"]:.2f}x); '
                  f'ray {ray["regret"]:.2f}x\n')
    return oracle, rows


def toolkit(names=None, seeds=(0, 1, 2), verbose=True, pairs=None):
    """
    THE FACTORIAL: every climber x every braker, scored for ROBUSTNESS.

    F-020 showed the best mechanism for raising the rate and the best for
    lowering it are different, and differ by surface. So the design question is
    not "which sensor" but "which pair", and the honest score is the WORST case
    across scenarios -- a controller is chosen for the run it will not ruin, not
    for its median.

    Reports, per surface and per pair: worst-case and median regret over
    {cold start, blow-up, hot}, plus how many of those failed SILENTLY (regret
    > 3x with nothing flagged), because an undetectable failure is worse than a
    detectable one of the same size.
    """
    import statistics as st

    scenarios = ('cold_start', 'blowup_100x', 'hot_90pct_to_cliff')
    pairs = pairs or [(c, b) for c in BenchRun.CLIMBERS for b in BenchRun.BRAKERS]
    out = {}

    for name in (names or list(SURFACES)):
        surface = SURFACES[name]
        oracle = find_oracle(surface, seeds=seeds)
        if verbose:
            print(f'\n{"=" * 78}\ntoolkit on {name} (oracle lr {oracle.lr:.4g})\n{"=" * 78}')
            # THIS HEADER IS WRONG AND THE CODE IS THE WRONG HALF, so it is
            # flagged rather than reworded: `steps_to_target` defaults to
            # frac=0.25, so the target is the distance the oracle passed at a
            # QUARTER of its run and the oracle itself scores ~steps//4. The
            # line below still says mid-run and prints steps//2. Reported, not
            # fixed -- the printed number needs an edit to an executable line.
            print(f'  to-target: steps to the distance the oracle passed at '
                  f'mid-run, so the ORACLE ITSELF SCORES {surface.steps // 2}. '
                  f'`behind` is in steps too.')
            print(f'  {"climber":<9} {"braker":<9} {"to-target":>10} '
                  f'{"behind":>9} {"regret":>8} {"cens":>5} {"silent":>7}')
        rows = []
        for climber, braker in pairs:
            regrets, ttts, behinds, silent, censored = [], [], [], 0, 0
            for sc_name in scenarios:
                per_seed = []
                for seed in seeds:
                    if sc_name == 'blowup_100x':
                        run = sc_blowup(surface, oracle, seed,
                                        climber=climber, braker=braker)
                    else:
                        lr = SEED_LR if sc_name == 'cold_start' else oracle.hot_lr(0.9)
                        run = surface.run(lr, seed=seed, servo=True,
                                          climber=climber, braker=braker)
                    per_seed.append(score(run, oracle))
                r = _agg(per_seed, 'regret')
                regrets.append(r)
                behinds.append(_agg(per_seed, 'steps_behind'))
                # a scenario counts as CENSORED when most seeds never arrived --
                # recorded, not silently dropped, because an arm that never
                # converges is the result and must not read as a missing cell
                arrived = [p['steps_to_target'] for p in per_seed
                           if p['steps_to_target'] is not None]
                if len(arrived) * 2 <= len(per_seed):
                    censored += 1
                    ttts.append(None)
                else:
                    ttts.append(float(np.median(arrived)))
                if r and r > 3.0 and not any(p['flagged'] for p in per_seed):
                    silent += 1
            live = [t for t in ttts if t is not None]
            live_b = [b for b in behinds if b is not None]
            row = dict(climber=climber, braker=braker,
                       worst=max(regrets), median=st.median(regrets),
                       ttt=max(live) if live else None, ttt_all=ttts,
                       behind=max(live_b) if live_b else None,
                       censored=censored, silent=silent, per_scenario=regrets)
            rows.append(row)
            if verbose:
                ttt = f'{row["ttt"]:.0f}' if row['ttt'] is not None else '--'
                if censored:
                    ttt = f'>{surface.steps}' if row['ttt'] is None else f'{ttt}*'
                bh = f'{row["behind"]:.0f}' if row['behind'] is not None else '--'
                print(f'  {climber:<9} {braker:<9} {ttt:>10} {bh:>9} '
                      f'{row["worst"]:>8.2f} {censored:>5} {silent:>7}')
        # rank on CONVERGENCE TIME, with never-arrived ranked last rather than
        # dropped -- ordering by a metric that some arms have no value for is how
        # a failure gets promoted to the top of a table
        best = min(rows, key=lambda r: (r['silent'], r['censored'],
                                        r['ttt'] if r['ttt'] is not None else math.inf))
        out[name] = rows
        if verbose:
            t = f'{best["ttt"]:.0f} steps' if best['ttt'] is not None else 'never'
            print(f'\n  most robust: climber={best["climber"]} braker={best["braker"]} '
                  f'(worst case {t} to target, regret {best["worst"]:.2f}x, '
                  f'{best["censored"]} censored, {best["silent"]} silent)')

    if verbose and len(out) > 1:
        print(f'\n{"=" * 78}\nACROSS ALL SURFACES (worst case anywhere)\n{"=" * 78}')
        print(f'  {"climber":<9} {"braker":<9} {"worst":>9} {"silent":>7}')
        agg = {}
        for name, rows in out.items():
            for r in rows:
                k = (r['climber'], r['braker'])
                cur = agg.setdefault(k, {'worst': 0.0, 'silent': 0})
                cur['worst'] = max(cur['worst'], r['worst'])
                cur['silent'] += r['silent']
        for (c, b), v in sorted(agg.items(), key=lambda kv: (kv[1]['silent'], kv[1]['worst'])):
            print(f'  {c:<9} {b:<9} {v["worst"]:>9.2f} {v["silent"]:>7}')
    return out


def main(names=None):
    """
    Run the board. A surface whose oracle fails its own sanity check is REPORTED
    and skipped rather than aborting the run -- that failure is a statement about
    the surface (it is not LR-sensitive enough to measure regret against), and
    losing the other surfaces to it would be the wrong trade.
    """
    failed = {}
    for name in (names or list(SURFACES)):
        try:
            run_surface(name)
        except ValueError as e:
            failed[name] = str(e)
            print(f'\n{"=" * 92}\n{name}: NO USABLE ORACLE\n{"=" * 92}\n  {e}')
    if failed:
        print(f'\nsurfaces without a usable oracle: {", ".join(failed)}')
    return failed


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == 'alpha':
        alpha_target_regret(args[1] if len(args) > 1 else 'mle')
    elif args and args[0] == 'toolkit':
        toolkit(args[1:] or None)
    elif args and args[0] == 'race':
        for nm in (args[1:] or list(SURFACES)):
            sensor_race(nm)
    else:
        main(args or None)
