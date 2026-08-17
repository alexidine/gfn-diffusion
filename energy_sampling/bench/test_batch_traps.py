"""
The two traps, DETECTED WHEN INJECTED -- not merely absent.

A design that does not contain a defect is not evidence. These cases inject each trap
into the REAL `train.Modeller` and require a FAILURE; the same cases run against the
unmutated controller and require a PASS. If an injection does not turn a case red, the
case is not protecting anything, which is the whole lesson of `bench/old`'s exclusion.

WHY NO CASE ASSERTS A PERCENTAGE. Phase 4's utilization proxy does not exist, so any
assertion of the form `util must not fall by more than X` needs a number nobody has.
Every verdict below is instead one of:

  * a DOMINANCE relation between two arms measured in the same cell, at the same seed,
    over the same horizon -- so the two objectives are never made commensurable and no
    exchange rate is needed;
  * an INVARIANCE -- a statistic that must not depend on a swept axis. Contains no
    constant at all, and is the property that made the tracking benchmark the one
    result that held up.

The two places a constant is unavoidable are named in-line, at their use.

GRID EDGES ARE REPORTED, NEVER SCORED AS VERDICTS. `selection_edge` and
`excluded_fraction` are checked in every case. An arm resting on `max_batch_size`, the
OOM ceiling or the batch floor has produced a BOUND; presenting one as an estimate is
what retracted the ray result.
"""

import math

import pytest

from bench.batch_arms import Fixed, NoFloor, Null, OccupancyFloor, Ship
from bench.batch_metrics import (SAMPLES_PER_SEC, UPDATES_PER_SEC, cell_can_rank,
                                 convicted_as_occupancy_rule, descent, dominates,
                                 excluded_fraction, exploration_cost, realised,
                                 selection_edge, time_weighted_occupancy)
from bench.batch_runner import BatchRun
from bench.gpu import (DECLINING, DECLINING_S_REGIMES, FLAT, RISING, SyntheticDevice,
                       umaperf0812)


@pytest.fixture
def patched():
    """Install an arm's injected defect and restore it afterwards, always."""
    import train
    undo = []

    def _install(arm):
        originals = arm.patch(train.Modeller)
        undo.append(originals)
        # rebind onto the fake, since attach_real_batch_sizer copied the ORIGINALS
        from bench.fake_modeller import FakeModeller
        for name in originals:
            setattr(FakeModeller, name, getattr(train.Modeller, name))
        return arm

    yield _install

    from bench.fake_modeller import FakeModeller
    for originals in reversed(undo):
        for name, fn in originals.items():
            setattr(train.Modeller, name, fn)
            setattr(FakeModeller, name, fn)


def _run(device, arm, steps=20000, **kw):
    return BatchRun(device, arm, steps=steps, **kw).run().trace


# =============================================================================
# TRAP (a) -- an occupancy rule, measured false, outranking the throughput gate
# =============================================================================

def _uma_cell():
    """
    The ONLY cell in this sandbox tied to a real measurement: umaperf0812's
    c_controller table, verbatim. U and S both fall in batch over a 7.4x range.
    """
    return umaperf0812(), [100, 165, 272, 449, 741]


def test_trap_a_cell_has_power_before_any_arm_runs():
    """
    THE PREMISE CHECK, run before the case that depends on it.

    Trap (a)'s condition is that occupancy does NOT reward growth while throughput
    PUNISHES it. Both halves are required: measured, a device with `util_shape=
    DECLINING` but the default timing model has throughput RISING across the ladder
    (2500 -> 4405), and an injected occupancy rule there is bit-identical to null --
    the cell has no power and would report a spurious PASS.

    Note the premise is an ENDPOINT relation, not monotonicity. `util_is_monotone_in`
    returns 0 on the measured cell because the real data wobbles (44 -> 49 at
    165 -> 272). A check written as `assert sign == -1` would reject the only measured
    cell in the repo, so the wobble is kept and the assertion is written around it.
    """
    dev, ladder = _uma_cell()
    assert dev.true_utilization(ladder[-1]) <= dev.true_utilization(ladder[0]), \
        'U rewards growth here -- this is not a trap (a) cell'
    assert dev.throughput(ladder[-1]) < dev.throughput(ladder[0]), \
        'S rewards growth here -- the injection would be RIGHT on priority 2'
    assert dev.util_is_monotone_in(ladder) == 0, \
        'the measured wobble 44->49 has been smoothed away; keep the real data'


def test_trap_a_is_detected_when_injected(patched):
    """
    The injected occupancy rule must be convicted by DOMINANCE against null.

    Not by a utilization threshold -- that number is Phase 4's undelivered deliverable.
    The verdict is: worse on the objective AND not better on the constraint the rule
    exists to serve. Both sides are realised numbers from the same cell and seed.
    """
    dev, ladder = _uma_cell()
    null = _run(dev, Null(batch=ladder[0]), steps=6000)
    inj = _run(dev, patched(OccupancyFloor(util_floor=60.0, batch=ladder[0],
                                           max_batch=ladder[-1])), steps=6000)

    # POWER PRECONDITION. An injection that never acted is the silent-no-op trap: it
    # posts a plausible row. UNRESOLVED, never PASS.
    assert [r['batch'] for r in inj] != [r['batch'] for r in null], \
        'UNRESOLVED: the injected rule never moved the batch -- cell has no power'
    assert any(r['util_reading'] is not None for r in inj), \
        'UNRESOLVED: the occupancy sensor produced no reading in this cell'

    assert convicted_as_occupancy_rule(inj, null), (
        f'trap (a) NOT detected: injected sps={realised(inj):.2f} '
        f'occ={time_weighted_occupancy(inj):.1f}% vs null sps={realised(null):.2f} '
        f'occ={time_weighted_occupancy(null):.1f}%')

    # The magnitude is a BOUND, not an estimate -- the rule grew as far as permitted.
    assert selection_edge(inj) == 'high'
    assert excluded_fraction(inj) == 0.0, 'device extrapolated; score is not interior'


def test_trap_a_detector_does_not_convict_the_shipping_controller():
    """
    THE FALSE-POSITIVE DIRECTION, on the cell where the trap actually fired.

    The shipping controller has no occupancy rule -- it was deleted -- so on
    umaperf0812 it must NOT be convicted by the same verdict that convicts the
    injection. A detector that reddens for the current code as well as the injected
    code distinguishes nothing, and the case would be protecting a number rather than
    a behaviour.
    """
    dev, ladder = _uma_cell()
    null = _run(dev, Null(batch=ladder[0]), steps=6000)
    ship = _run(dev, Ship(batch=ladder[0], max_batch=ladder[-1]), steps=6000)
    assert not convicted_as_occupancy_rule(ship, null), (
        f'the trap (a) detector convicts the SHIPPING controller '
        f'(exploration cost {exploration_cost(ship, null):.4f}, '
        f'net_rungs {descent(ship)["net_rungs"]}): '
        f'sps {realised(ship):.2f} vs {realised(null):.2f}, '
        f'occ {time_weighted_occupancy(ship):.1f}% vs '
        f'{time_weighted_occupancy(null):.1f}%')

    # ...but it IS dominated, by ~1%, and that is the cost of one probe up a declining
    # curve. Pinned as a number so the separation between "explored and returned" and
    # "grew and kept it" stays visible rather than becoming folklore.
    assert dominates(ship, null), \
        'the shipping controller no longer pays an exploration cost on this cell'
    assert descent(ship)['net_rungs'] == 0, \
        'the shipping controller retained a growth on a declining curve'
    # The magnitude is REPORTED, never bounded -- bounding it would be the selected bar
    # this whole file exists to avoid. Measured 2026-08-16: 0.0097.
    assert exploration_cost(ship, null) > 0


@pytest.mark.parametrize('shape', [RISING, FLAT])
def test_trap_a_case_does_not_fire_where_the_premise_is_false(patched, shape):
    """
    NEGATIVE CONTROL. Where occupancy genuinely rewards growth, the same injection must
    NOT be convicted -- and where it never acts, the case must say UNRESOLVED rather
    than PASS. A detector that fires everywhere detects nothing.
    """
    dev = SyntheticDevice(t_fixed=2.0, sps_max=5000.0, util_shape=shape)
    null = _run(dev, Null(batch=1000), steps=4000)
    inj = _run(dev, patched(OccupancyFloor(util_floor=60.0, batch=1000,
                                           max_batch=50000)), steps=4000)
    if [r['batch'] for r in inj] == [r['batch'] for r in null]:
        pytest.skip('UNRESOLVED: injection inert in this cell -- correctly not a PASS')
    assert not dominates(inj, null), \
        f'the trap (a) detector fired on a {shape} cell, where growth is not the defect'


# =============================================================================
# TRAP (b) -- a knee walk with no floor, under flat throughput
# =============================================================================

FLAT_DEVICE = dict(t_fixed=0.0, sps_max=5000.0, util_shape=FLAT)


@pytest.mark.parametrize('horizon', [20000, 40000, 60000])
def test_trap_b_structural_invariance_holds_with_the_floor(horizon):
    """
    B1, THE STRUCTURAL ASSERTION -- and it contains no constant at all.

    The device is STATIONARY, so a controller that has converged returns the same
    answer at any horizon. Doubling the horizon changes estimator precision, never the
    answer. `n_distinct` is therefore horizon-invariant for a converged controller and
    grows without bound for a descending one.

    This is the assertion that works at ZERO switching cost, where the objective is
    blind -- see the next test for why that case is not hypothetical.
    """
    tr = _run(SyntheticDevice(**FLAT_DEVICE), Ship(batch=1000, max_batch=50000),
              steps=horizon)
    assert descent(tr)['n_distinct'] == 2, (
        f'horizon {horizon}: n_distinct={descent(tr)["n_distinct"]} -- a converged '
        f'controller on a stationary device must not depend on the horizon')


@pytest.mark.parametrize('horizon', [20000, 60000])
def test_trap_b_is_detected_when_the_floor_is_removed(patched, horizon):
    """
    The injection must break the invariance the previous test establishes.

    Measured: with the floor intact `n_distinct` is 2 at every horizon; with it removed
    it is 13 at 20k and 19 at 60k -- a function of the horizon, which is exactly what
    "has not converged" means.
    """
    tr = _run(SyntheticDevice(**FLAT_DEVICE),
              patched(NoFloor(batch=1000, max_batch=50000)), steps=horizon)
    d = descent(tr)
    assert d['n_distinct'] > 2, \
        f'trap (b) NOT detected at horizon {horizon}: n_distinct={d["n_distinct"]}'
    assert d['final'] < 1000, \
        f'the walk did not descend below the configured batch: final={d["final"]}'
    # The terminal value is a BOUND: the walk stopped where the batch variable's own
    # domain stopped it, not where the controller decided.
    assert selection_edge(tr) in ('low', None)


def test_trap_b_objective_is_BLIND_at_zero_switching_cost():
    """
    THE MEASURED REASON B1 EXISTS, recorded as a test so it cannot be forgotten.

    Under exactly flat throughput with no switching cost, the descent to batch 1 costs
    EXACTLY NOTHING on samples/sec -- so an objective-based verdict cannot convict it.
    `train.py` says as much: "Flat throughput genuinely does argue for the smallest
    batch (same samples/sec, faster steps)... gradient quality is not something it can
    see."

    This test asserts the BLINDNESS, so that anyone who later replaces B1 with a
    throughput assertion discovers immediately that it cannot work. It is also the
    reason `cell_can_rank` declares this cell NULL for any objective-based score.
    """
    dev = SyntheticDevice(**FLAT_DEVICE)
    scores = {f'fixed@{b}': realised(_run(dev, Fixed(b), steps=4000))
              for b in (100, 1000, 7410)}
    spread = max(scores.values()) - min(scores.values())
    assert spread < 1e-6, (
        f'this cell is no longer flat in samples/sec (spread {spread:.3g}); B1 may be '
        f'redundant, but check before removing it')
    ok, why = cell_can_rank(scores, seed_spread=0.0)
    assert not ok and 'NULL CELL' in why, \
        'a zero-spread column must be DECLARED, never averaged into a board'


def test_trap_b_objective_DOES_convict_once_switching_is_charged(patched):
    """
    B2. Where the device charges for churn, the objective decides on its own -- and the
    reference is `min` over the Fixed arms in this cell, i.e. the WORST constant batch.

    "The controller must not be beaten by every constant batch on the ladder" is a
    statement no reasonable controller fails by accident, and its reference is
    regenerated inside the experiment rather than selected. Scored on OBSERVED seconds,
    so recompiles are charged to whoever incurs them.
    """
    dev_kw = dict(FLAT_DEVICE, recompile_s=30.0)
    ladder = [1000, 1650, 2722, 4491, 7410]
    fixed = {b: realised(_run(SyntheticDevice(**dev_kw), Fixed(b), steps=20000),
                         observed=True) for b in ladder}
    worst_fixed = min(fixed.values())

    floored = realised(_run(SyntheticDevice(**dev_kw),
                            Ship(batch=1000, max_batch=50000), steps=20000),
                       observed=True)
    descending = realised(_run(SyntheticDevice(**dev_kw),
                               patched(NoFloor(batch=1000, max_batch=50000)),
                               steps=20000), observed=True)

    # THE INJECTION IS CONVICTED. This is the assertion the case exists for.
    assert descending < floored, (
        f'removing the floor did not cost anything measurable even with switching '
        f'charged: descending={descending:.1f} floored={floored:.1f}')

    # ...AND SO IS THE SHIPPING CONTROLLER, which is a finding rather than a failure.
    #
    # Measured: floored=4927.3 against a WORST fixed arm of 4962.8. On a flat curve the
    # floor stops the DESCENT but not the CHURN -- the walk still climbs 1000 -> 1650,
    # is refused, drops back, and repeats (57 transitions in 60k steps). Every distinct
    # size it visits is charged a recompile, and it buys nothing, so it loses to the
    # WORST constant batch on the ladder.
    #
    # Asserted in the direction that is TRUE, so the fact is pinned rather than
    # discovered again later. If a replacement controller ever makes this pass in the
    # other direction, that is a real improvement and this assertion should be
    # inverted deliberately -- not deleted.
    assert floored < worst_fixed, (
        f'the shipping controller now BEATS the worst fixed batch on a flat curve '
        f'({floored:.1f} >= {worst_fixed:.1f}). That is an improvement over the '
        f'behaviour measured 2026-08-16; invert this assertion deliberately.')


# =============================================================================
# The same shape as the two named traps, found by the survey
# =============================================================================

def test_every_arm_is_distinguishable_from_null(patched):
    """
    The guard that has now fired twice in this repo. An arm that silently no-ops posts
    a plausible row rather than erroring, and the tell is two traces agreeing to many
    significant figures.
    """
    dev, ladder = _uma_cell()
    null = [r['batch'] for r in _run(dev, Null(batch=ladder[0]), steps=3000)]
    for arm in (Ship(batch=ladder[0], max_batch=ladder[-1]),
                patched(OccupancyFloor(util_floor=60.0, batch=ladder[0],
                                       max_batch=ladder[-1]))):
        traj = [r['batch'] for r in _run(dev, arm, steps=3000)]
        assert traj != null, f'arm {arm.name} is bit-identical to null'


def test_the_two_objectives_disagree_and_the_disagreement_is_visible():
    """
    THE LARGEST UNSTATED ASSUMPTION IN THE SHIPPING CONTROLLER, as a test.

    `train.py` maximises `samples_per_sec`, justified by "updates/sec = samples/sec /
    accum_target, so step time does not enter". That identity holds only while
    accumulation is engaged, and accumulation engages STRICTLY BELOW the target
    (`accumulating = accum_target > self.batch_size`). mk_dev ships `batch_size ==
    fused_grad_accum_min_samples == 1000`, so every reachable batch is at or above the
    target and the identity holds NOWHERE on the ladder.

    On a saturating cost curve the two objectives have OPPOSITE argmaxes. A sandbox
    that scored only one would report a controller as optimal while it maximised the
    wrong thing, so the objective is a parameter everywhere and this test pins the
    disagreement rather than letting it be rediscovered.
    """
    dev = SyntheticDevice(t_fixed=2.0, sps_max=5000.0, util_shape=RISING)
    ladder = [1000, 1650, 2722, 4491, 7410]
    sps = {b: realised(_run(dev, Fixed(b), steps=2000), SAMPLES_PER_SEC) for b in ladder}
    ups = {b: realised(_run(dev, Fixed(b), steps=2000), UPDATES_PER_SEC) for b in ladder}
    assert max(sps, key=sps.get) == max(ladder), 'samples/sec should favour the largest rung'
    assert max(ups, key=ups.get) == min(ladder), 'updates/sec should favour the smallest rung'
    assert max(sps, key=sps.get) != max(ups, key=ups.get), \
        'the two objectives agree here, so this cell cannot show the conflict'
