"""
MK'S FIVE REQUIREMENTS, EACH SCORED ON THE METRIC THAT ANSWERS IT.

Stated 2026-08-13, verbatim:

  1. all starts are relatively cold and expected to ramp until some stability
     bound (may be conservative)
  2. should respond dynamically to conditions -- though this evolution is
     generally slow, on the timescale of thousands of steps
  3. should do a big cut e.g. on exploding losses
  4. should not get stuck too-hot or too-cold for long periods
  5. should recover from very bad states -- hence a purposely too hot/fast LR
     ramp. We want to see a graceful and fast recovery.

WHY THIS FILE EXISTS RATHER THAN MORE CRUCIBLE CELLS. The crucible scores ONE
thing: time to reach a target, as a multiple of the best fixed rate's time. That
answers "is it fast". Three of the five above are not speed questions --
(1) accepts a conservative rate, (3) is about a discrete response, and (4) is
about DURATION at a bad rate, which a time-to-target metric only notices when it
happens to cost a deadline. A run can arrive on schedule having spent half of it
badly mis-set, and score perfectly.

So each requirement here gets its own scenario and its own number:

  R1  cold start      steps until the rate FIRST enters a 2x band, then the
                      fraction of the remaining run it stays there. A
                      conservative-but-stable arrival passes, which is what (1)
                      asks for and what `%over budget` refuses.
  R2  slow drift      off-target fraction while the optimum moves over
                      thousands of steps
  R3  explosion       parameters are genuinely blown up mid-run, so the loss
                      really does explode. Scores whether the rate is cut, how
                      far, and how long the run then spends off-target -- the
                      cut is the easy half, climbing back is the half that
                      distinguishes arms
  R4  steady state    started AT the reference rate and left alone: the longest
                      unbroken excursion outside the band. This is the "set it
                      and forget it" number
  R5  too-hot start   started ABOVE the divergence cliff, not at 90% of the way
                      to it. Scores recovery: does it come back to the band at
                      all, how fast, and how much damage on the way

R3 AND R5 ARE PARTLY TESTS OF THE SHARED SAFETY NET, and that is stated rather
than hidden: the divergence tripwire and rewind belong to every arm, so the
catch itself does not discriminate. What discriminates is what the rate does
AFTERWARDS -- the peak cut leaves it cold, and a bounded climber takes
`ln(gap)/beta` steps to come back. That is why both are scored on off-target
time rather than on "did it survive".
"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from bench.old.crucible import EQ, MLE, _mk, _oracle_task, _regime_shift
from bench.old.scenarios import (ON_TARGET_BAND, SEED_LR, longest_off_target,
                             time_off_target)

#: How far above the cliff R5 starts. The existing `hot_90pct` scenario starts
#: at 90% of the way from the oracle to the cliff -- still inside the stable
#: band, so it tests tracking rather than recovery. MK asked for a purposely
#: too-hot ramp, which means past the boundary.
TOO_HOT = 2.0

#: R3's blow-up: parameters scaled hard enough that the loss genuinely explodes.
#: A faked loss value would test the tripwire's arithmetic; scaling the
#: parameters tests the whole path, including whether the rewind has anything to
#: go back to.
#:
#: THE BAR IS ABSOLUTE, NOT RELATIVE. `check_spike` (controller.py:125) fires on
#: a raw 1e9 on the loss or the gradient norm. The first version of this
#: scenario scaled by 50, which on a converged quadratic multiplies a loss of
#: order 1 by ~2500 -- six orders short of the bar. Every arm scored
#: `cut_factor` 1.0 and `div` 0.0, which reads as "no cut was needed" and was
#: really "no explosion happened". `triggered` is reported for exactly that
#: reason: a scenario that fails to fire is not evidence about the response.
EXPLODE = 1.0e5

#: The servo ramps the rate up from ~0 over `warmup_steps` (50 in this bench),
#: so a run STARTED too hot passes through the target band on its way out. The
#: first version of R5 timed that transit and reported "recovered in 10 steps"
#: for every arm including the one with no sensor. Recovery is only counted
#: after the warmup, and only if the rate then STAYS put.
WARMUP_SKIP = 60
DWELL = 50

ARMS = (
    ('hyper 0.02 (published)', 'hyperx', 'none',
     {'hyper_beta': 0.02, 'hyper_beta_down': 0.02}),
    ('hyper 0.08', 'hyperx', 'none',
     {'hyper_beta': 0.08, 'hyper_beta_down': 0.08}),
    ('ray+ray (ships)', 'ray', 'ray', None),
    ('ramp+plateau', 'ramp', 'plateau', None),
    ('NULL (no sensor)', 'none', 'none', None),
)

CELLS = [
    ('mle base', 'mle', dict(MLE), 2000, (1e-6, 1e-1, 12), {}),
    ('mle n0.5', 'mle', dict(MLE, noise=0.5), 2000, (1e-6, 1e-1, 12), {}),
    ('mle q0.1', 'mle', dict(MLE, quartic=1e-1), 2000, (1e-6, 1e-1, 12), {}),
    ('eq base', 'equilibration', dict(EQ), 3000, (1e-4, 1e1, 12),
     {'lr_flow': 1.0}),
]


def _band(ref_lr):
    return ref_lr / ON_TARGET_BAND, ref_lr * ON_TARGET_BAND


def _first_in_band(run, ref_lr, after=0, dwell=DWELL):
    """
    First step at or after `after` where the rate is in the band AND STAYS for
    `dwell` steps.

    Both conditions are load-bearing. Without `after` this times the warmup
    envelope's climb rather than the controller. Without `dwell` it counts a
    single-step transit -- a rate sweeping past the band on its way somewhere
    worse scores the same as one that settled in it.
    """
    lo, hi = _band(ref_lr)
    xs = [h.get('lr') for h in run.history]
    ok = [x is not None and math.isfinite(x) and lo <= x <= hi for x in xs]
    for i in range(int(after), len(ok)):
        if not ok[i]:
            continue
        end = min(i + int(dwell), len(ok))
        if all(ok[i:end]) and end - i >= min(dwell, len(ok) - i):
            return i
    return None


def _held_after(run, ref_lr, start):
    """Fraction of the run AFTER `start` spent inside the band."""
    lo, hi = _band(ref_lr)
    tail = [h.get('lr') for h in run.history[start:]]
    tail = [x for x in tail if x is not None and math.isfinite(x)]
    if not tail:
        return 0.0
    return sum(1 for x in tail if lo <= x <= hi) / len(tail)


def _mk_run(surface, lr, seed, arm):
    _, climber, braker, std = arm
    return surface.make(lr, seed=seed, servo=True, climber=climber,
                        braker=braker, standard=std)


# --------------------------------------------------------------- scenarios

def r1_cold_start(surface, oracle, seed, arm):
    run = _mk_run(surface, SEED_LR, seed, arm)
    run.run(surface.steps, stop_on_divergence=False)
    at = _first_in_band(run, oracle.lr, after=WARMUP_SKIP)
    return run, {
        'arrived': at is not None,
        'steps_to_band': at if at is not None else math.inf,
        'held_after': _held_after(run, oracle.lr, at) if at is not None else 0.0,
    }


def r2_slow_drift(surface, oracle, seed, arm):
    """The optimum moves 1.5x over the middle of the run -- MK's stated rate of
    real regime change, thousands of steps rather than one."""
    run = _mk_run(surface, oracle.lr, seed, arm)
    n, start = surface.steps, surface.steps // 4
    span = n // 2
    run.run(start, stop_on_divergence=False)
    for _ in range(span):
        _regime_shift(run.game, 1.0 / span)
        run.step()
        if run.aborted or run.game.diverged():
            break
    if not (run.aborted or run.game.diverged()):
        run.run(n - start - span, stop_on_divergence=False)
    return run, {}


@torch.no_grad()
def _blow_up(run, bar=None, cap=14):
    """
    Scale the parameters until the loss ACTUALLY crosses the tripwire's bar.

    A fixed factor cannot do this: the bar is absolute (1e9) while the loss
    depends on the surface and on how converged the run is, so the same factor
    explodes one cell and does nothing on another. Measured: x50 moved a
    converged quadratic's loss by ~2500 and fired nothing; x1e5 still fell short
    of 1e9 and every arm scored `triggered` 0%.

    Uses `probe_loss`, which evaluates without stepping the optimizer, so
    sizing the explosion does not itself perturb the run. Returns the factor
    used and whether the bar was reached.
    """
    ctrl = run.m.lr_controller
    bar = float(bar if bar is not None
                else ctrl._cfg('divergence_loss_abs', 1.0e9))
    batch = run._draw_probe()
    factor = 1.0
    for _ in range(cap):
        loss = float(run._probe_loss(batch))
        if not math.isfinite(loss) or loss >= 2.0 * bar:
            return factor, True
        for p in run.game.policy_params:
            p.mul_(10.0)
        factor *= 10.0
    loss = float(run._probe_loss(batch))
    return factor, (not math.isfinite(loss)) or loss >= 2.0 * bar


def r3_exploding_loss(surface, oracle, seed, arm):
    """A real blow-up: the parameters are scaled, so the loss genuinely
    explodes and the whole spike path runs, rewind included."""
    run = _mk_run(surface, oracle.lr, seed, arm)
    at = surface.steps // 3
    run.run(at, stop_on_divergence=False)
    before = run.m.lr_of(run.game.train_key)
    div_before = run.divergences
    factor, reached = _blow_up(run)
    run.run(surface.steps - at, stop_on_divergence=False)
    after = min((h['lr'] for h in run.history[at:at + 200]
                 if h.get('lr') and math.isfinite(h['lr'])), default=math.nan)
    back = _first_in_band(run, oracle.lr, after=at)
    return run, {
        # DID THE EXPLOSION ACTUALLY EXPLODE? If this is not ~100%, nothing
        # below is evidence about the response to one. `reached` says the loss
        # crossed the bar; `triggered` says the controller noticed.
        'blew_up': reached,
        'triggered': run.divergences > div_before,
        'cut_factor': before / after if after and math.isfinite(after) and after > 0
                      else math.nan,
        'recovered': back is not None,
        'steps_to_recover': (back - at) if back is not None else math.inf,
    }


def r4_steady_state(surface, oracle, seed, arm):
    """
    Started right and left alone. Anything that moves is self-inflicted.

    THE NO-SENSOR ARM WINS THIS BY CONSTRUCTION -- it starts at the reference
    rate and cannot leave it, so it scores a perfect 0% and that is not a
    result. What this scenario is for is the opposite: catching an arm that
    WANDERS off a rate that was already right. Measured, `ray+ray` drifts 20.8%
    of the run cold with a 17.5% longest excursion, which no time-to-target
    metric would show.
    """
    run = _mk_run(surface, oracle.lr, seed, arm)
    run.run(surface.steps, stop_on_divergence=False)
    return run, {}


def r5_too_hot(surface, oracle, seed, arm):
    """Started ABOVE the cliff, not below it."""
    cliff = oracle.cliff or oracle.lr * 5.0
    run = _mk_run(surface, cliff * TOO_HOT, seed, arm)
    run.run(surface.steps, stop_on_divergence=False)
    at = _first_in_band(run, oracle.lr, after=WARMUP_SKIP)
    return run, {
        'recovered': at is not None,
        'steps_to_recover': at if at is not None else math.inf,
        'held_after': _held_after(run, oracle.lr, at) if at is not None else 0.0,
    }


REQUIREMENTS = (
    ('R1 cold start ramps up', r1_cold_start),
    ('R2 tracks a slow drift', r2_slow_drift),
    ('R3 cuts on an explosion', r3_exploding_loss),
    ('R4 steady state holds', r4_steady_state),
    ('R5 recovers from too hot', r5_too_hot),
)


def _task(item):
    cell, oracle, arm, seeds = item
    surface = _mk(cell)
    out = {}
    for name, fn in REQUIREMENTS:
        rows, extra = [], []
        for seed in seeds:
            run, ex = fn(surface, oracle, seed, arm)
            o = time_off_target(run, oracle.lr)
            rows.append({
                'off': o['off'] if o else math.nan,
                'hot': o['hot'] if o else math.nan,
                'cold': o['cold'] if o else math.nan,
                'longest': longest_off_target(run, oracle.lr) or math.nan,
                'div': run.divergences,
                'aborted': bool(run.aborted),
            })
            extra.append(ex)
        agg = {k: float(np.nanmean([r[k] for r in rows]))
               for k in ('off', 'hot', 'cold', 'longest', 'div')}
        agg['aborted'] = float(np.mean([r['aborted'] for r in rows]))
        for k in (extra[0] if extra else {}):
            vals = [e[k] for e in extra]
            if isinstance(vals[0], bool):
                agg[k] = float(np.mean(vals))
            else:
                fin = [v for v in vals if math.isfinite(v)]
                agg[k] = float(np.median(fin)) if fin else math.inf
        out[name] = agg
    return cell[0], arm[0], out


def _init_worker():
    torch.set_num_threads(1)


def main(seeds=10, workers=None):
    seeds = tuple(range(int(seeds)))
    workers = int(workers or max(2, min(20, (os.cpu_count() or 4) - 4)))
    print(f'{"=" * 104}\nREQUIREMENTS BATTERY -- MK\'s five cases, each on its '
          f'own metric\n{len(seeds)} seeds, {len(CELLS)} cells, '
          f'{len(ARMS)} arms, {workers} workers\n{"=" * 104}\n')

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker) as pool:
        oracles = {}
        for label, got, why in pool.map(_oracle_task, CELLS):
            if got is None:
                print(f'  {label:<12} SKIPPED -- {why}')
            else:
                oracles[label] = got[0]
        jobs = [(c, oracles[c[0]], a, seeds)
                for c in CELLS if c[0] in oracles for a in ARMS]
        rows = list(pool.map(_task, jobs))

    for name, _ in REQUIREMENTS:
        print(f'\n{"=" * 104}\n{name}\n{"=" * 104}')
        extra_keys = [k for k in rows[0][2][name]
                      if k not in ('off', 'hot', 'cold', 'longest', 'div',
                                   'aborted')]
        print(f'  {"cell":<10} {"arm":<24} {"off-target":>11} {"hot":>7} '
              f'{"cold":>7} {"longest":>8} {"div":>6}  '
              + '  '.join(f'{k:>16}' for k in extra_keys))
        for cell, arm, got in rows:
            g = got[name]
            tail = '  '.join(
                (f'{g[k]:>16.0f}' if math.isfinite(g[k]) and g[k] > 1.5
                 else f'{g[k]:>16.0%}') if isinstance(g[k], float) else
                f'{g[k]:>16}' for k in extra_keys)
            print(f'  {cell:<10} {arm:<24} {g["off"]:>11.1%} {g["hot"]:>7.1%} '
                  f'{g["cold"]:>7.1%} {g["longest"]:>8.1%} {g["div"]:>6.1f}  '
                  + tail)
    return rows


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10,
         int(sys.argv[2]) if len(sys.argv) > 2 else None)
