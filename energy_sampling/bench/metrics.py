"""
THE FIVE NUMBERS, and why each is the shape it is.

Every one is a pure function of a run's trace, so a metric can be corrected
without re-running anything. The old stack could not do that, which is why four
re-runs were spent on metric definitions.

  1. final_loss        trailing-window MEDIAN, not the last value
  2. lead_fraction     share of steps with the lowest smoothed loss, vs the
                       other arms on the SAME seed
  3. lr_stability      sd(log lr) and the worst single-step jump
  4. backslide         share of the run where the smoothed loss is RISING
  5. catastrophes      counted, never averaged

WHAT IS DELIBERATELY ABSENT: any metric that needs a reference rate. The old
topline was `steps_to_target(run)/steps_to_target(oracle)` against a `2x budget`,
and it failed three separate ways -- a ceiling below the threshold (so "over
budget" meant "never converged"), then a denominator at the metric's own floor,
then a reference rate 187x off on one surface family. Measured on the last full
battery, 92% of all "over budget" events were non-arrivals and only 3 of 900 runs
were finite-and-over. A censored ratio against a selected reference throws away
the data and then reports the censoring. `final_loss` and `lead_fraction` say the
same thing with no reference and no censoring.
"""
import math

import numpy as np

#: Trailing window for the final-loss estimate. A single last value is a
#: minibatch draw; the median of a window is not.
FINAL_WINDOW = 100
#: Smoothing horizon for lead and backslide. Both compare noisy series, and
#: without smoothing they measure batch noise.
EMA_PERIOD = 25.0


#: WHICH LOSS THE SCORING READS. `eloss` is the noise-free expected loss; `loss`
#: is the noisy training loss the controller acts on. Scoring on `loss` ranks
#: arms by a random sign near the optimum -- see `_Game.expected_loss`. Falls
#: back to `loss` for games that do not expose the clean one.
SCORE_KEY = 'eloss'


def _series(run, key):
    if key == 'loss':
        vals = [h.get(SCORE_KEY) for h in run.trace]
        # ALL, NOT ANY. On `any`, a single populated step switched the WHOLE
        # series to a key absent everywhere else -- measured, one stray `eloss`
        # at step 137 of 300 drove `final_loss` to inf, because the trailing
        # window it is computed over had none. A partially populated key means a
        # malformed trace, and the safe reading is the one that is complete.
        if vals and all(v is not None for v in vals):
            return vals
    return [h.get(key) for h in run.trace]


def _finite(xs):
    return [x for x in xs if x is not None and math.isfinite(x)]


def smoothed_loss(run, period=EMA_PERIOD):
    """EMA of the loss on the per-step clock. None where the loss was unusable."""
    a = 1.0 - math.exp(-1.0 / float(period))
    out, ema = [], None
    for x in _series(run, 'loss'):
        if x is not None and math.isfinite(x):
            ema = x if ema is None else (1 - a) * ema + a * x
        out.append(ema)
    return out


def final_loss(run, window=FINAL_WINDOW):
    """
    Median loss over the last `window` steps.

    Median, not mean: a diverging run's tail is heavy and a mean would be set by
    its worst single step. Window, not the last value: the last value is one
    minibatch draw, and an arm can win it by luck.
    """
    tail = _finite(_series(run, 'loss')[-int(window):])
    return float(np.median(tail)) if tail else math.inf


def lead_fraction(runs):
    """
    For each arm, the share of steps where its smoothed loss is the lowest.

    THE ONLY METRIC HERE THAT COMPARES ARMS DIRECTLY, and the reason no oracle is
    needed: `runs` are the same seed on the same game, so the noise stream is
    shared and a difference is the arm. Ties are split evenly rather than given
    to whichever arm happens to be first in the list -- with fixed-rate arms
    early in a run, exact ties are common and awarding them by list order would
    be a silent ranking bias.

    Returns {arm name: fraction}. Steps where no arm has a usable loss are
    dropped from the denominator, so this cannot be inflated by another arm
    dying early.
    """
    if not runs:
        return {}
    series = {r.arm.name: smoothed_loss(r) for r in runs}
    n = min(len(s) for s in series.values())
    wins = {k: 0.0 for k in series}
    counted = 0
    for i in range(n):
        vals = {k: s[i] for k, s in series.items()
                if s[i] is not None and math.isfinite(s[i])}
        if not vals:
            continue
        counted += 1
        best = min(vals.values())
        leaders = [k for k, v in vals.items() if v <= best]
        for k in leaders:
            wins[k] += 1.0 / len(leaders)
    if not counted:
        return {k: float('nan') for k in series}
    return {k: v / counted for k, v in wins.items()}


def lr_stability(run):
    """
    Dispersion and worst jump of the learning rate, IN LOG SPACE.

    Log space because the controllers act multiplicatively (`exp(beta*cos)`,
    `ratio**eta`, `factor=0.5`), so a move of the same size is the same event at
    any rate. `var(lr)` in linear space is set by whichever excursion happened to
    be largest and is not comparable across arms whose rates differ by decades.

    `sd` is the ordinary dispersion. `max_jump` is the largest single-step move,
    which is what "really wild swings" means operationally -- a fourth moment
    would answer the same question far less legibly and far more noisily.
    `span` is peak-to-trough, the excursion a run actually traversed.
    """
    lrs = [x for x in _series(run, 'lr')
           if x is not None and math.isfinite(x) and x > 0]
    if len(lrs) < 2:
        return {'sd': math.nan, 'max_jump': math.nan, 'span': math.nan}
    L = np.log(np.asarray(lrs, dtype=float))
    d = np.abs(np.diff(L))
    return {'sd': float(L.std()),
            'max_jump': float(d.max()),
            'span': float(L.max() - L.min())}


#: How far a rise must exceed the run's own high-frequency noise to count.
#: 3 sigma of a random walk over the horizon.
BACKSLIDE_SIGMA = 3.0


def backslide(run, period=EMA_PERIOD, horizon=None, k=BACKSLIDE_SIGMA):
    """
    Share of the run where the loss is HIGHER THAN IT WAS `horizon` steps ago by
    more than the run's own noise can explain.

    "Is training going backwards", directly. Two design points, both learned the
    hard way:

    NOT the sign of the step-to-step slope. Smoothing reduces the AMPLITUDE of
    noise but not its sign alternation, so `mean(diff > 0)` returns ~0.5 for any
    noise-dominated series no matter how hard it is smoothed -- measured, exactly
    0.501 on a flat alternating loss. That metric would have ranked arms by their
    batch noise, which is the failure `var(slope(loss))` was rejected for in the
    first place. It took a hand-built test to see it; it is invisible on real
    traces, where ~0.5 looks like a plausible number.

    NOT a bare comparison either. Over a horizon `h`, a pure random walk moves
    ~`sigma_1 * sqrt(h)`, so a threshold has to scale that way or a quiet run and
    a violent one are judged by the same absolute bar. `sigma_1` is estimated as
    the MEDIAN absolute one-step move of the smoothed series -- median, so a
    genuine trend does not inflate the noise estimate it is being tested against.

    A flat run scores ~0: not descending is not the same as going backwards, and
    `final_loss` is what says the run went nowhere.
    """
    s = [x for x in smoothed_loss(run, period) if x is not None]
    h = int(period if horizon is None else horizon)
    if len(s) < h + 2 or h < 1:
        return math.nan
    a = np.asarray(s, dtype=float)
    sigma1 = float(np.median(np.abs(np.diff(a))))
    tol = k * sigma1 * math.sqrt(h)
    rise = a[h:] - a[:-h]
    return float((rise > tol).mean())


def catastrophes(run):
    """
    Counted, never averaged.

    The goal is "at worst ~2x the best fixed rate, NEVER 50x" -- a statement
    about the tail. A mean hides exactly that: an arm that is excellent on 5
    seeds and detonates on the 6th has a fine mean and is unusable. Reported as
    raw counts so they can never be smoothed into a ranking.
    """
    losses = _series(run, 'loss')
    return {'divergences': int(run.divergences),
            'aborted': bool(run.aborted),
            'nonfinite_steps': sum(1 for x in losses
                                   if x is None or not math.isfinite(x))}


def final_lr(run, window=FINAL_WINDOW):
    """
    The rate the run ENDED at (median of the last `window` steps).

    Separate from `final_loss` because they answer different questions and can
    disagree loudly: measured, `hyper` ended ~8x above the best fixed rate and
    still matched its final loss, because the surface is flat above the optimum
    under Adam. "Got a good loss" is not "found the right rate", and only this
    column shows the second.
    """
    tail = [h.get('lr') for h in run.trace[-int(window):]]
    tail = [x for x in tail if x is not None and math.isfinite(x) and x > 0]
    return float(np.median(tail)) if tail else math.nan


def _final_loss_or_death(run, window=FINAL_WINDOW):
    """
    `final_loss`, except an ABORTED run scores infinite.

    AN ABORT MUST NOT BE SCORED AS A FINISH, and the trailing-window median makes
    that trap sharp rather than obvious. The rewind restores a healthy state
    after every divergence, so a run that detonates repeatedly and then exhausts
    its reload budget has a tail full of restored, healthy losses -- its median
    is excellent right up to the moment it dies. Measured on the `regime shift`
    cell: `ramp+plateau` aborted on 5 seeds out of 5 and came FIRST in the cell,
    and `fixed@0.01` aborted on 5 of 5 and came second. Both were being ranked on
    a run that never reached the end.

    Production treats the abort as terminal -- it raises `FrozenTrainingState` and
    releases the GPU -- so the honest score is the one for not finishing. This
    also makes `died in k/n` count aborts, which it otherwise never did: every
    aborted run had a finite loss, so the column read '-' while 30 runs had died.
    """
    return math.inf if run.aborted else final_loss(run, window)


def score_run(run):
    """Every per-run metric. `lead_fraction` is per-GROUP and added by the caller."""
    st = lr_stability(run)
    # THE WINDOW IS A PROPERTY OF THE SURFACE, not a constant. On a CONVERGING
    # surface a short window is right: the level is still moving, and a long
    # average would blend it with earlier, worse parameters. On a STATIONARY one
    # -- a tracking problem -- the quantity is fixed and a long average is simply
    # a better estimator of it. Measured on `TrackingGame`: seed noise 0.176 nats
    # at a 100-step window against 0.040 at 1000, taking adjacent rungs from 1.8
    # sigma apart (unusable) to 10.2 (clean). Same runs, same data, only the
    # estimator changed.
    window = getattr(run.game, 'score_window', FINAL_WINDOW)
    return {'arm': run.arm.name, 'seed': run.seed,
            'final_loss': _final_loss_or_death(run, window),
            'final_loss_at_abort': final_loss(run, window),
            'final_lr': final_lr(run),
            'lr_sd': st['sd'], 'lr_max_jump': st['max_jump'],
            'lr_span': st['span'],
            'backslide': backslide(run),
            **catastrophes(run)}
