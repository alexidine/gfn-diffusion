"""
Batch-sizer metrics. Every one a pure function of a trace, per `bench/metrics.py`.

WHAT IS DELIBERATELY ABSENT: any metric that divides by a selected reference. The
retired stack's topline divided by an oracle-selected best rate, and one of that
selection's three recorded failure modes was a GRID-EDGE WINNER. Here the only
cross-arm comparison is against the arms in the same cell (`bench/trackboard.py:109`),
and where a bound is unavoidable it is `min` over all arms -- the WORST -- so the
assertion reads "the controller must not be beaten by every constant batch on the
ladder", which no reasonable controller fails by accident.

THE OBJECTIVE IS A PARAMETER, NOT A CHOICE THIS MODULE GETS TO MAKE. `train.py`
maximises `samples_per_sec` and justifies it with an identity -- updates/sec =
samples/sec / accum_target -- that holds only while `accum_target > batch_size`
(`train.py:2467`, accumulation engages STRICTLY BELOW the target). mk_dev ships
`batch_size: 1000 == fused_grad_accum_min_samples: 1000`, so every reachable batch
sits at or above the target, where updates/sec is `1/t(B)` instead. On a saturating
cost curve the two have OPPOSITE argmaxes: samples/sec is maximised at the largest
rung, updates/sec at the smallest. So `objective=` is an explicit argument everywhere,
and any board that reports one without the other is reporting half a question.
"""

import math

#: Objectives, named. Both are pure functions of a trace.
SAMPLES_PER_SEC = 'samples_per_sec'
UPDATES_PER_SEC = 'updates_per_sec'


def _interior(trace):
    """
    Rows where the device was not extrapolating.

    `MeasuredDevice` holds its endpoints outside the measured span, so a row with
    `outside_range` carries a HELD value, not a measurement. Scoring it silently is
    how a curve gets invented past the last rung -- the never-measured knee, rebuilt
    inside the tool meant to detect it.
    """
    return [r for r in trace if not r['outside_range']]


def excluded_fraction(trace):
    """Share of rows dropped by `_interior`. Reported beside every score, never hidden."""
    return 0.0 if not trace else 1.0 - len(_interior(trace)) / len(trace)


def realised(trace, objective=SAMPLES_PER_SEC, observed=False):
    """
    The run's realised objective over its interior rows.

    `observed=False` scores the noise-free `true_t`, so an arm cannot win on its draw.
    `observed=True` scores `dt_observed`, which CHARGES SWITCHING COSTS -- recompiles
    land in `dt_observed` and not in `true_t`. Use it whenever the question is whether
    churn paid for itself; use the default whenever the question is placement.
    """
    rows = _interior(trace)
    if not rows:
        return math.nan
    key = 'dt_observed' if observed else 'true_t'
    secs = sum(r[key] for r in rows)
    if secs <= 0:
        return math.nan
    if objective == SAMPLES_PER_SEC:
        return sum(r['true_work'] for r in rows) / secs
    if objective == UPDATES_PER_SEC:
        return len(rows) / secs
    raise ValueError(f'unknown objective {objective!r}')


def time_weighted_occupancy(trace):
    """
    GROUND-TRUTH occupancy, integrated over wall clock -- what a scheduler would see.

    Time-weighted, not a mean of rows: `_gpu_util_mean` is an unweighted mean of point
    samples and that is one of the sensor's documented defects, not a model to copy.
    Eval seconds are included in the denominator at ZERO occupancy, which is exactly
    the blindness the in-process sampler has and the scheduler does not.
    """
    rows = _interior(trace)
    num = sum(r['true_util'] * r['true_t'] for r in rows)
    den = sum(r['true_t'] + r['eval_s'] for r in rows)
    return num / den if den > 0 else math.nan


def descent(trace):
    """
    The batch trajectory's shape. No constants anywhere -- these are counts.

    `n_distinct` is the statistic trap (b)'s structural assertion reads: on a
    STATIONARY device a converged controller returns the same answer at any horizon,
    so `n_distinct` must not depend on the horizon. A descent's does.

    `n_transitions` is reported beside it because the floor stops the DESCENT and not
    the CHURN: measured, with the floor intact under flat throughput the batch
    oscillates 1000 <-> 1650 permanently (57 transitions in 60k steps, n_distinct 2).
    Asserting only on `n_distinct` would call that converged.
    """
    b = [r['batch'] for r in trace]
    if not b:
        return dict(n_distinct=0, n_transitions=0, final=None,
                    terminal_at_floor=False, terminal_at_max=False, net_rungs=0)
    return dict(
        n_distinct=len(set(b)),
        n_transitions=sum(1 for x, y in zip(b, b[1:]) if x != y),
        final=b[-1],
        terminal_at_floor=bool(trace[-1]['at_floor']),
        terminal_at_max=bool(trace[-1]['at_max_batch']),
        net_rungs=b[-1] - b[0],
    )


def selection_edge(trace):
    """
    Is the arm's resting place a MEASUREMENT or a BOUND?

    `'high'` -- pinned at `max_batch_size` or the OOM ceiling: the arm grew as far as
                it was allowed, so its selection is a lower bound on what it wanted.
    `'low'`  -- pinned at the batch floor: symmetrically, an upper bound.
    `None`   -- interior, and only then is the selection an estimate.

    BOTH ENDS, deliberately. The candidate designs that carried edge vocabulary carried
    only the ceiling end; a monotone-decreasing objective rests at the LOWEST rung with
    nothing below it, which is the same defect mirrored. A ladder reporting its own
    edge is reporting nothing (`bench/test_tracking.py`), and an unresolved reading at
    a boundary presented as a verdict is what retracted the ray result.

    `'pinned'` -- A PINNED ARM HAS NO EDGE. `Fixed` and `Null` set
                `batch_size == max_batch_size`, so `at_max_batch` is True on every row
                and a naive reading labels every constant arm `'high'` -- meaningless,
                since the arm was never trying to grow. An arm whose batch never moved
                is `'pinned'`, and a board must not read that as a bound.
    """
    if not trace:
        return None
    if len({r['batch'] for r in trace}) == 1:
        return 'pinned'
    last = trace[-1]
    if last['at_max_batch'] or last['ceiling'] is not None and last['batch'] >= last['ceiling']:
        return 'high'
    if last['at_floor']:
        return 'low'
    return None


def cell_can_rank(scores, seed_spread=0.0):
    """
    Can this cell rank arms at all? `scores` is {arm name: realised objective}.

    Returns (True, '') or (False, reason). A cell whose between-arm range does not
    exceed its within-arm seed spread is a NULL CELL and must be DECLARED, never
    averaged into a board. The retired battery had 44 of 65 cell x scenario columns
    with identical scores across every arm INCLUDING the control -- 1300 runs wearing
    ~10 binary trials -- and nothing printed a spread.

    The implicit constant is 1.0, i.e. signal <= noise. That is a domain boundary, not
    a tuned tolerance: below it the cell is not measuring the arms. It is named here
    rather than buried because the brief requires any selected bar to be flagged, and
    a ratio of exactly 1 is the weakest possible such choice. `bench/audit.py` uses 10
    for the same job, and 10 IS a choice.

    `seed_spread=0` MEANS UNMEASURED, NEVER "ZERO NOISE". This function's first version
    compared `rng <= seed_spread` directly, so with the default it declared a cell
    rankable on any nonzero float difference -- and its own test caught it posting
    RANKABLE on a measured spread of 6.35e-13, which is a zero-spread column by any
    reading. That is precisely the defect the function exists to detect, reproduced
    inside the detector. So an unmeasured seed spread falls back to a RELATIVE floor:
    differences at the scale of float noise on the arms' own magnitude are not signal.
    """
    vals = [v for v in scores.values() if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return False, 'fewer than two finite arms'
    rng = max(vals) - min(vals)
    scale = max(abs(v) for v in vals)
    #: Relative floor used only when the seed spread is unmeasured. Set at ~1e-9 of the
    #: arms' own magnitude: comfortably above double-precision accumulation noise over
    #: a few 10^4 steps, and far below any difference a real controller produces.
    floor = max(seed_spread, _REL_NULL_EPS * scale)
    if rng <= floor:
        why = (f'NULL CELL: between-arm range {rng:.6g} <= {floor:.6g} '
               f'(seed spread {seed_spread:.6g}' +
               ('' if seed_spread > 0 else ', UNMEASURED -- relative floor applied') +
               ') -- this cell cannot rank arms')
        return False, why
    return True, ''


#: See `cell_can_rank`. A relative floor, applied only when the seed spread was not
#: measured. It is a numerical-precision boundary, not a tuned effect size.
_REL_NULL_EPS = 1e-9


def dominates(worse, better, objective=SAMPLES_PER_SEC):
    """
    Two-dimensional dominance: `worse` is worse on the objective AND not better on the
    constraint it exists to serve. Both comparisons are arm-vs-arm, in the same cell.

    THE POINT IS THAT THE TWO AXES ARE NEVER MADE COMMENSURABLE. There is no exchange
    rate between percent-occupancy and samples/sec anywhere in this function, which is
    what lets it convict an occupancy rule without knowing the cluster's threshold --
    the quantity Phase 4 has not delivered. `assert util_drop < X` would need X.
    """
    return (realised(worse, objective) < realised(better, objective)
            and time_weighted_occupancy(worse) <= time_weighted_occupancy(better))


def convicted_as_occupancy_rule(arm, null, objective=SAMPLES_PER_SEC):
    """
    The trap (a) verdict: dominated by null AND the growth was RETAINED.

    WHY DOMINANCE ALONE IS NOT ENOUGH -- measured, and it is the whole reason this
    function exists rather than a bare `dominates` call. On the umaperf0812 cell:

        arm             sps     occ%   final B   distinct
        null           57.70    52.0      100        1
        ship           57.14    51.6      100        2      <- explored once, RETURNED
        ship+occfloor  24.41    42.0      741        5      <- RETAINED the growth

    `dominates` convicts BOTH, because one exploratory probe up a declining curve costs
    ~1%. A detector that reddens for the shipping controller as well as the injected one
    distinguishes nothing. But the separator must not be a magnitude threshold ("worse
    by more than X%") -- X is exactly the kind of selected bar that produced the results
    this project retracted.

    The structural separator: the injected arm ENDED at the ladder top and never came
    back; the shipping controller ended where it started. `net_rungs > 0` is a sign test
    on the trajectory's endpoints, and zero is the natural boundary, not a tuned value.

    A DELIBERATE GAP, stated rather than closed: an arm that wastes a great deal and
    still returns to its start escapes this verdict. That is the cost-of-exploration
    question, which is a different one, and it is REPORTED as a number
    (`exploration_cost` below) rather than folded into a pass/fail.
    """
    return dominates(arm, null, objective) and descent(arm)['net_rungs'] > 0


def exploration_cost(arm, null, objective=SAMPLES_PER_SEC):
    """
    What an arm paid, relative to never moving, as a fraction. Reported, never asserted.

    Measured for the shipping controller on umaperf0812: 0.0097 -- one probe up a
    declining curve, then a correct pin back at the start. That is the price of
    learning the curve is declining, and it is not a defect.
    """
    base = realised(null, objective)
    if not (base and math.isfinite(base)):
        return math.nan
    return 1.0 - realised(arm, objective) / base
