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

from bench.batch_arms import (DescentWalk, Fixed, Null, OccupancyFloor, Ship,
                              Sizer)
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

    Two shipping shapes to clear, because the replacement has two modes:

      * with no batch_util_target, the controller HOLDS -- its batch trace is
        bit-identical to Null's BY DESIGN (S3), so it cannot be convicted and it
        pays no exploration cost at all. The old controller paid ~1% here for one
        probe up the declining curve; that cost is gone and this pins it.
      * with a target set, the controller CALIBRATES: it walks the ladder, finds
        that no rung clears 60% (the measured cell tops out at 52%), returns to
        the argmax-occupancy rung -- the base -- and says INFEASIBLE. It moves,
        so it pays an exploration cost, but it must end where it started
        (net_rungs == 0) and must NOT be convicted: exploring and returning is
        the lawful behaviour the verdict's second clause exists to separate from
        growing and keeping it.
    """
    dev, ladder = _uma_cell()
    null = _run(dev, Null(batch=ladder[0]), steps=6000)

    ship = _run(dev, Ship(batch=ladder[0], max_batch=ladder[-1]), steps=6000)
    assert [r['batch'] for r in ship] == [r['batch'] for r in null], \
        'with no util target the replacement must HOLD -- something moved the batch'
    assert not convicted_as_occupancy_rule(ship, null)

    sizer = _run(dev, Sizer(util_target=0.60, batch=ladder[0],
                            max_batch=ladder[-1]), steps=6000)
    assert [r['batch'] for r in sizer] != [r['batch'] for r in null], \
        'UNRESOLVED: the sizer never calibrated -- this cell shows nothing'
    assert not convicted_as_occupancy_rule(sizer, null), (
        f'the trap (a) detector convicts the REPLACEMENT '
        f'(exploration cost {exploration_cost(sizer, null):.4f}, '
        f'net_rungs {descent(sizer)["net_rungs"]}): '
        f'sps {realised(sizer):.2f} vs {realised(null):.2f}, '
        f'occ {time_weighted_occupancy(sizer):.1f}% vs '
        f'{time_weighted_occupancy(null):.1f}%')
    assert descent(sizer)['net_rungs'] == 0, \
        'the sizer retained a growth on a cell where no rung clears the target'
    assert descent(sizer)['final'] == ladder[0], \
        'INFEASIBLE must resolve to the argmax-occupancy rung, the base here'
    # The magnitude is REPORTED, never bounded -- bounding it would be the selected
    # bar this whole file exists to avoid.
    assert exploration_cost(sizer, null) > 0


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
def test_trap_b_structural_invariance_holds(horizon):
    """
    B1, THE STRUCTURAL ASSERTION -- and it contains no constant at all.

    The device is STATIONARY, so a controller that has converged returns the same
    answer at any horizon. Doubling the horizon changes estimator precision, never the
    answer. `n_distinct` is therefore horizon-invariant for a converged controller and
    grows without bound for a descending one.

    Under the replacement the answer is n_distinct == 1 with ZERO transitions: there
    is no walk at all. The old controller measured 2 distinct sizes and 57 transitions
    here (the 1000 <-> 1650 churn the floor could not stop); both numbers going to
    their minimum is the improvement, pinned so a regression is visible.
    """
    tr = _run(SyntheticDevice(**FLAT_DEVICE), Ship(batch=1000, max_batch=50000),
              steps=horizon)
    d = descent(tr)
    assert d['n_distinct'] == 1, (
        f'horizon {horizon}: n_distinct={d["n_distinct"]} -- a hold controller on a '
        f'stationary device must not visit a second size')
    assert d['n_transitions'] == 0, (
        f'horizon {horizon}: {d["n_transitions"]} transitions -- the churn the old '
        f'controller paid (57 per 60k steps) has come back')


@pytest.mark.parametrize('horizon', [20000, 60000])
def test_trap_b_is_detected_when_a_floorless_walk_is_reintroduced(patched, horizon):
    """
    The injection must break the invariance the previous test establishes.

    The shipping controller no longer CONTAINS a walk -- trap (b) is prevented by
    construction -- so the detection case injects one: the retired knee recheck's
    downward step, minus the floor that saved it. The design keeps this case
    because the walk could be reintroduced, and this is the assertion that would
    catch it: n_distinct becomes a function of the horizon, which is exactly what
    "has not converged" means.
    """
    tr = _run(SyntheticDevice(**FLAT_DEVICE),
              patched(DescentWalk(batch=1000, max_batch=50000)), steps=horizon)
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

    holding = realised(_run(SyntheticDevice(**dev_kw),
                            Ship(batch=1000, max_batch=50000), steps=20000),
                       observed=True)
    descending = realised(_run(SyntheticDevice(**dev_kw),
                               patched(DescentWalk(batch=1000, max_batch=50000)),
                               steps=20000), observed=True)

    # THE INJECTION IS CONVICTED. This is the assertion the case exists for.
    assert descending < holding, (
        f'the injected walk did not cost anything measurable even with switching '
        f'charged: descending={descending:.1f} holding={holding:.1f}')

    # THE DELIBERATE INVERSION. The old controller LOST to the worst constant batch
    # here (measured 2026-08-16: 4927.3 vs 4962.8) because the floor stopped its
    # descent but not its churn -- 57 transitions in 60k steps, each distinct size
    # charged a recompile, buying nothing. The old assertion pinned that loss and
    # said to invert it deliberately if a replacement ever fixed it; the replacement
    # holds one size, pays one recompile, and so must no longer lose.
    assert holding >= worst_fixed, (
        f'the replacement loses to the worst fixed batch on a flat curve '
        f'({holding:.1f} < {worst_fixed:.1f}) -- churn is back')


# =============================================================================
# The same shape as the two named traps, found by the survey
# =============================================================================

def test_every_arm_is_distinguishable_from_null(patched):
    """
    The guard that has now fired twice in this repo. An arm that silently no-ops posts
    a plausible row rather than erroring, and the tell is two traces agreeing to many
    significant figures.

    `Ship` is deliberately NOT in this list any more: with no util target the
    replacement holds, so its batch trace equals Null's BY DESIGN (asserted as such
    in the false-positive case above). The arms that must prove they act are the
    calibrating sizer and the injected occupancy rule.
    """
    dev, ladder = _uma_cell()
    null = [r['batch'] for r in _run(dev, Null(batch=ladder[0]), steps=3000)]
    for arm in (Sizer(util_target=0.60, batch=ladder[0], max_batch=ladder[-1]),
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


# =============================================================================
# The replacement's positive behaviour: selection, minimality, and the S2 audit
# =============================================================================

def test_sizer_holds_the_smallest_rung_clearing_the_target():
    """
    Where occupancy genuinely rises with batch, the sizer must select the SMALLEST
    rung whose measured occupancy clears the target, and hold it: growth is bought
    for the constraint only, so any rung above the first clearing one is update
    rate spent on nothing.

    Structural assertions, not planted constants: the walk ascends monotonically,
    the resting rung clears the target while the rung below it does not, the rest
    is interior (a bound is not a selection), and the answer is horizon-invariant
    (B1 for the replacement: converged means the horizon changes nothing).
    """
    # percent, because that is the unit `true_utilization` reports and the unit
    # the assertions below read. The CONFIG key is a fraction (state 9), so the
    # arm takes target/100 -- the same conversion train.select_batch_size makes.
    target = 55.0
    finals = {}
    for horizon in (6000, 12000):
        run = BatchRun(SyntheticDevice(t_fixed=2.0, sps_max=5000.0,
                                       util_shape=RISING),
                       Sizer(util_target=target / 100.0, batch=1000,
                             max_batch=50000),
                       steps=horizon)
        tr = run.run().trace
        rungs = [b for i, b in enumerate(r['batch'] for r in tr)
                 if i == 0 or tr[i]['batch'] != tr[i - 1]['batch']]
        assert rungs == sorted(rungs), f'the walk moved downward: {rungs}'
        dev = run.device
        final = tr[-1]['batch']
        assert dev.true_utilization(final) >= target, (
            f'held rung {final} reads {dev.true_utilization(final):.1f}%, under the '
            f'{target:.0f}% target')
        below = max(b for b in rungs if b < final)
        assert dev.true_utilization(below) < target, (
            f'rung {below} below the selection already clears the target -- the '
            f'selection is not minimal')
        assert selection_edge(tr) is None, \
            'the selection rests on a bound, so it is not a selection'
        assert run.m.batch_sizer['reason'] == 'target_met'
        finals[horizon] = final
    assert len(set(finals.values())) == 1, (
        f'the selection depends on the horizon ({finals}) -- the controller has '
        f'not converged')


def test_growth_cap_bounds_the_overshoot_past_the_crossing():
    """
    The capped rung step exists to bound the selection's ABSOLUTE overshoot: the
    held rung may exceed the true occupancy crossing by at most batch_growth_cap
    samples. On this device the crossing is exactly computable
    (U(B) = 20 + 70*(x/(2+x)), x = B/5000, so U >= 55 iff B >= 10000), which is
    high enough on the ladder that pure-geometric rungs overshoot it by more than
    the cap -- so the uncapped run is the test's own power check: if IT lands
    within the cap too, the cell cannot show the cap doing anything.
    """
    target, crossing = 55.0, 10000
    dev_kw = dict(t_fixed=2.0, sps_max=5000.0, util_shape=RISING)
    dev = SyntheticDevice(**dev_kw)
    assert dev.true_utilization(crossing) >= target > dev.true_utilization(crossing - 1)

    # factor pinned at 1.65: the power check below (uncapped geometric rungs
    # overshoot the 10000 crossing by more than one cap) is a property of THIS
    # cell's rung geometry, and the shipping factor moved to 1.6 (2026-08-19),
    # where the uncapped ladder happens to land inside one cap of the crossing
    # target is a percent here (see the sibling test); the config key is a fraction
    capped = BatchRun(SyntheticDevice(**dev_kw),
                      Sizer(util_target=target / 100.0, batch=1000, max_batch=50000,
                            batch_growth_factor=1.65),
                      steps=6000).run()
    uncapped = BatchRun(SyntheticDevice(**dev_kw),
                        Sizer(util_target=target / 100.0, batch=1000, max_batch=50000,
                              batch_growth_factor=1.65, batch_growth_cap=0),
                        steps=6000).run()
    cap = int(capped.m.args.batch_growth_cap)
    sel_c, sel_u = capped.trace[-1]['batch'], uncapped.trace[-1]['batch']

    assert sel_u - crossing > cap, (
        f'UNRESOLVED: the uncapped ladder lands at {sel_u}, within one cap of the '
        f'crossing {crossing} -- this cell has no power to show the cap acting')
    assert crossing <= sel_c, f'{sel_c} does not clear the crossing {crossing}'
    assert sel_c - crossing <= cap, (
        f'capped selection {sel_c} overshoots the crossing {crossing} by more '
        f'than the cap {cap}')
    assert capped.m.batch_sizer['reason'] == 'target_met'


def test_sizer_says_infeasible_and_the_conclusion_is_readable():
    """
    On the measured MLIP cell no rung reaches 60%, and 'no batch works' must be a
    CONCLUSION the run carries (reason: infeasible), not a batch that happens to sit
    somewhere. The resting place is the argmax-occupancy rung -- the base, here.
    """
    dev, ladder = _uma_cell()
    # factor 1.65: the umaperf0812 table's rungs were measured on that spacing,
    # and the assertion below requires the walk to land on them exactly
    run = BatchRun(dev, Sizer(util_target=0.60, batch=ladder[0],
                              max_batch=ladder[-1],
                              batch_growth_factor=1.65), steps=6000)
    tr = run.run().trace
    s = run.m.batch_sizer
    assert s['reason'] == 'infeasible', s
    assert tr[-1]['batch'] == ladder[0]
    # every rung it climbed is in the table with a measured occupancy -- the account
    # a postmortem needs, kept as state rather than as stdout
    assert [r['batch'] for r in s['table']] == ladder
    assert all(r['util'] is not None for r in s['table'])


class _TransientlyBusyDevice(SyntheticDevice):
    """
    A device whose occupancy READS high at grown batches for a while, then stops --
    the shape that makes a calibration conclusion wrong AFTER it is reached. Real
    sources of the same shape: a transient co-tenant on a shared node, an init-time
    burst, a stage whose composition drifts. The S2 audit exists for exactly this.
    """

    def __init__(self, lie_reads=10, base_batch=1000, **kw):
        super().__init__(**kw)
        self.lie_reads = int(lie_reads)
        self.base_batch = int(base_batch)
        self.reads = 0

    def utilization(self, work):
        self.reads += 1
        if float(work) <= self.base_batch:
            return 30.0
        # a grown batch reads busy while the transient lasts, then reads WORSE than
        # the base -- so a lived policy window cannot help but disagree with the
        # calibration dwell, even if a lie sample or two lands inside the window
        return 90.0 if self.reads <= self.lie_reads else 25.0

    def true_utilization(self, work):
        return 30.0 if float(work) <= self.base_batch else 25.0


def test_s2_audit_stands_a_failed_growth_back_down():
    """
    S2: a growth kept on the strength of a calibration reading must survive a full
    policy window of lived occupancy, or stand down to the base rung. Without the
    audit this run would hold the grown batch forever on the strength of ten
    transient samples.
    """
    dev = _TransientlyBusyDevice(lie_reads=10, base_batch=1000,
                                 t_fixed=2.0, sps_max=5000.0, util_shape=FLAT)
    run = BatchRun(dev, Sizer(util_target=0.60, batch=1000, max_batch=50000),
                   steps=12000)
    m = run.m
    grew = False
    for _ in range(12000):
        run.step()
        s = m.batch_sizer or {}
        if s.get('phase') == 'hold' and s.get('selected', 0) > 1000:
            grew = True
        if grew and s.get('reason') == 'stood_down':
            break
    assert grew, ('UNRESOLVED: the transient reading never bought a growth, so '
                  'there is nothing for the audit to catch')
    s = m.batch_sizer
    assert s['reason'] == 'stood_down', (
        f'the audit never fired: still holding {m.batch_size} '
        f'(reason {s.get("reason")}) on the strength of a transient reading')
    assert m.batch_size == 1000, 'stand-down must return to the base rung'
