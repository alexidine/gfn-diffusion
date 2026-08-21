"""
Replay Racing: unit tests of the decision layer, and gate 6.1 of
`docs/design/lr_probe_protocol.md`.

Two halves, and the split matters. The first half pins the MECHANICS (the sign
test's exact critical values, that pairing cancels what it must, that the
vetoes veto). The second half is the GATE: statistical properties of the whole
policy, measured over thousands of simulated races, that the actuator has to
demonstrate before it is allowed to move a real learning rate.

Every constant here is COMPUTED, not transcribed. The one thing this project
has learned twice about controller batteries is that a number copied between
documents is a number nobody re-derives.

The gate includes its own negative control: the rejected v1 rule, run through
the identical harness, must FAIL. A suite that cannot see the disease the two
previous controllers actually had cannot certify their replacement.
"""

import math

import numpy as np
import pytest

from bench.race_sim import RaceSim, SimSpec, decide_v1_highest_competitive, run_v1
from lr_race import (ENTRY_ARMS, FINE_ARMS, INCUMBENT, ArmScores, RaceConfig,
                     RaceRecord, decide, decide_confirm, decide_fine,
                     decide_screen, min_favoring, rung_dex, shifted_bracket,
                     sign_sf)

CFG = RaceConfig()

# Acceptance bars, from design section 6.1. Named here so a failure message
# says which bar, and so changing one is a visible edit.
DRIFT_BOUND = 0.005        # dex per probe, TOST
OFFSET_BOUND = 0.15        # dex, |E[log10(lr/lr*)]|
SD_BOUND = 0.30            # dex
P_FAR_BOUND = 0.01         # P(|offset| > 0.6 dex)


# ---------------------------------------------------------------- mechanics

R = CFG.replicates          # the sample size of the test -- see lr_race rule 4
U = 10                      # held-out batches per replicate (precision, not n)


def _arm(mult, end_rows, mid_rows=None, died=False):
    """Build an ArmScores. `mid` defaults to half of `end`, which is what a
    constant-rate window produces and therefore passes the half-window check."""
    end = tuple(tuple(float(v) for v in r) for r in end_rows)
    mid = (tuple(tuple(float(v) for v in r) for r in mid_rows)
           if mid_rows is not None
           else tuple(tuple(v / 2.0 for v in r) for r in end))
    return ArmScores(multiplier=mult, mid=mid, end=end, died=died)


def _shared(rng, sd=1.0, reps=None):
    """The component every arm sees: batch difficulty. Must cancel in pairing."""
    return rng.normal(0, sd, size=(reps or R, U))


def test_sign_test_critical_values_are_exact():
    # Computed from the exact binomial, not transcribed from the design doc.
    assert sign_sf(10, 10) == pytest.approx(1 / 1024)
    assert sign_sf(9, 10) == pytest.approx(11 / 1024)
    assert min_favoring(10, 0.05 / 4) == 9      # entry-style, 4 challengers
    assert min_favoring(10, 0.05 / 2) == 9      # fine bracket, 2 challengers
    assert min_favoring(10, 0.05) == 9          # confirm, single contrast
    assert min_favoring(20, 0.05 / 4) == 16
    assert min_favoring(20, 0.05) == 15


def test_sign_test_reports_underpowered_rather_than_guessing():
    # With 4 units even a clean sweep is p = 1/16 = 0.0625 > 0.05: no count can
    # clear the level. The rule must SAY so, not silently accept 4-of-4.
    assert min_favoring(4, 0.05) is None
    assert min_favoring(5, 0.05) == 5           # 1/32 = 0.031 clears


def test_pairing_cancels_a_huge_shared_component():
    """A per-unit loss level hundreds of nats wide must not reach the verdict.

    This is the property the whole design rests on: unpaired scoring of this
    same contrast measured ~1/30th the t on the real system, with the wrong
    sign. If a shared offset can move the decision, the sensor is measuring
    batch difficulty.
    """
    rng = np.random.default_rng(0)
    shared = _shared(rng, 500.0)
    inc = _arm(1.0, shared + 1.0)
    chal = _arm(2.0, shared + 0.5)              # uniformly better by 0.5
    rec = RaceRecord(arms=(inc, chal), kind='fine')
    d = decide_fine(rec, CFG)
    assert d.moves and d.multiplier == 2.0


def test_decision_is_invariant_to_loss_scale():
    """Multiplying every loss by 1000 must not change the verdict.

    The sign test only reads signs, so this is true by construction -- which is
    exactly why it was chosen over anything denominated in nats. A margin in
    loss units would not survive this test, and that is the reason the design
    has no margin.
    """
    rng = np.random.default_rng(1)
    base = _shared(rng, 1.0)
    inc = _arm(1.0, base + 1.0)
    chal = _arm(2.0, base + 0.6)
    small = decide_fine(RaceRecord(arms=(inc, chal), kind='fine'), CFG)
    inc_big = _arm(1.0, (base + 1.0) * 1000.0)
    chal_big = _arm(2.0, (base + 0.6) * 1000.0)
    big = decide_fine(RaceRecord(arms=(inc_big, chal_big), kind='fine'), CFG)
    assert (small.action, small.multiplier) == (big.action, big.multiplier)


def test_half_window_check_rejects_a_sprinter():
    """An arm that wins by the end score alone, having given ground in the
    second half, is a transient and must not be adopted."""
    rng = np.random.default_rng(2)
    shared = _shared(rng, 1.0)
    inc = _arm(1.0, shared + 1.0)
    # Ahead at the end (0.9 < 1.0) but its whole lead was made by mid-window:
    # incumbent mid = 0.5, sprinter mid = 0.1, so the second half LOST ground.
    sprint = _arm(2.0, shared + 0.9, mid_rows=shared / 2 + 0.1)
    d = decide_fine(RaceRecord(arms=(inc, sprint), kind='fine'), CFG)
    assert not d.moves
    assert d.reason == 'no_significant_challenger'


def test_one_lucky_replicate_cannot_carry_a_win():
    """A huge win on one replicate against small losses on the rest is a
    training-path fluke, not a better rate.

    The sign test is immune to it by construction -- it reads the SIGN of each
    replicate, so a single enormous margin counts once and loses 9-to-1. A
    mean-based statistic would be dominated by the outlier, which is why this
    test exists at the level it does.
    """
    rng = np.random.default_rng(3)
    shared = _shared(rng, 0.5)
    inc = _arm(1.0, shared + 1.0)
    end = shared + 1.0
    end[0] -= 30.0                      # replicate 0 wins enormously
    end[1:] += 0.05                     # every other replicate loses slightly
    d = decide_fine(RaceRecord(arms=(inc, _arm(2.0, end)), kind='fine'), CFG)
    assert not d.moves


def test_symmetric_in_both_directions():
    """Mirroring the data must mirror the verdict, exactly.

    Every asymmetric response in this project's history rectified noise into a
    one-way ratchet: hyper_down_gain 2.0, ray's eta_up/eta_down, and an earlier
    revision of this very design that confirmed only raises.
    """
    rng = np.random.default_rng(4)
    shared = _shared(rng, 0.5)
    inc = _arm(1.0, shared + 1.0)
    up = decide_fine(RaceRecord(arms=(inc, _arm(2.0, shared + 0.5),
                                      _arm(0.5, shared + 1.5)), kind='fine'), CFG)
    dn = decide_fine(RaceRecord(arms=(inc, _arm(2.0, shared + 1.5),
                                      _arm(0.5, shared + 0.5)), kind='fine'), CFG)
    assert up.moves and dn.moves
    assert up.multiplier == 2.0 and dn.multiplier == 0.5


def test_tie_break_takes_the_nearest_clearing_arm():
    """Two clearing challengers -> the one closest to the incumbent.

    Taking the strongest instead selects outward-biased noise: the max of m
    correlated contrasts sits ~1 sd above the truth, and outward means further
    from the rate that is currently working.
    """
    rng = np.random.default_rng(5)
    shared = _shared(rng, 0.2)
    inc = _arm(1.0, shared + 1.0)
    near = _arm(2.0, shared + 0.4)
    far = _arm(4.0, shared + 0.2)               # bigger advantage, further away
    d = decide_fine(RaceRecord(arms=(inc, near, far), kind='fine'), CFG)
    assert d.moves and d.multiplier == 2.0


def test_died_arms_are_recorded_and_never_selected():
    rng = np.random.default_rng(6)
    shared = _shared(rng, 0.2)
    inc = _arm(1.0, shared + 1.0)
    dead = _arm(2.0, shared - 5.0, died=True)   # would have "won" enormously
    d = decide_fine(RaceRecord(arms=(inc, dead), kind='fine'), CFG)
    assert not d.moves
    assert d.detail['died'] == [2.0]


def test_isolation_failure_blocks_actuation():
    rng = np.random.default_rng(7)
    shared = _shared(rng, 0.2)
    rec = RaceRecord(arms=(_arm(1.0, shared + 1.0), _arm(2.0, shared + 0.4)),
                     kind='fine', isolation_ok=False)
    assert decide_fine(rec, CFG).action == 'invalid'


def test_positive_control_corrupted_restore_trips_the_duplicate_null():
    """Deliberately break the restore certificate and require a FAILURE.

    The house rule: a test that cannot fail when the bug is present is not
    evidence. On a deterministic route the same-order duplicate must reproduce
    bitwise; a nonzero spread means state leaked between arms, and a state leak
    is indistinguishable from a real effect from the outputs alone.
    """
    rng = np.random.default_rng(8)
    shared = _shared(rng, 0.2)
    arms = (_arm(1.0, shared + 1.0), _arm(2.0, shared + 0.4))
    clean = RaceRecord(arms=arms, kind='fine', expect_bitwise=True,
                       duplicate_spread=0.0)
    assert decide_fine(clean, CFG).moves
    leaked = RaceRecord(arms=arms, kind='fine', expect_bitwise=True,
                        duplicate_spread=1e-9)
    d = decide_fine(leaked, CFG)
    assert d.action == 'invalid' and d.reason == 'duplicate_not_bitwise'


def test_nondeterminism_floor_rivalling_the_effect_blocks_actuation():
    rng = np.random.default_rng(9)
    shared = _shared(rng, 0.2)
    arms = (_arm(1.0, shared + 1.0), _arm(2.0, shared + 0.4))
    # Effect is ~0.6; a floor of 0.5 rivals it, a floor of 0.01 does not.
    ok = RaceRecord(arms=arms, kind='fine', duplicate_spread=0.01)
    bad = RaceRecord(arms=arms, kind='fine', duplicate_spread=0.5)
    assert decide_fine(ok, CFG).moves
    assert decide_fine(bad, CFG).reason == 'duplicate_rivals_effect'


def test_screen_selects_without_testing_and_confirm_tests_without_selecting():
    rng = np.random.default_rng(10)
    shared = _shared(rng, 0.3, reps=1)
    arms = [_arm(1.0, shared + 1.0)]
    for m, adv in [(0.25, 0.0), (4.0, 0.5), (16.0, 0.2), (64.0, -0.1)]:
        arms.append(_arm(m, shared + 1.0 - adv))
    d = decide_screen(RaceRecord(arms=tuple(arms), kind='screen'), CFG)
    assert d.action == 'candidate' and d.multiplier == 4.0

    # The confirm race carries exactly one contrast, so no correction applies.
    shared3 = _shared(rng, 0.3)
    conf = RaceRecord(arms=(_arm(1.0, shared3 + 1.0), _arm(4.0, shared3 + 0.3)),
                      kind='confirm')
    assert decide_confirm(conf, CFG).moves


def test_screen_expansion_is_symmetric():
    """A bottom-edge winner must expand DOWN exactly as a top-edge winner
    expands up. An up-only expansion is a one-way ratchet in disguise."""
    rng = np.random.default_rng(11)
    shared = _shared(rng, 0.3, reps=1)

    def screen(best_mult):
        arms = [_arm(m, shared + 1.0 - (0.8 if m == best_mult else 0.0))
                for m in ENTRY_ARMS]
        return decide_screen(RaceRecord(arms=tuple(arms), kind='screen'), CFG)

    assert screen(max(ENTRY_ARMS)).action == 'expand_up'
    assert screen(min(ENTRY_ARMS)).action == 'expand_down'


def test_expansion_is_bounded():
    rng = np.random.default_rng(12)
    shared = _shared(rng, 0.3, reps=1)
    arms = tuple(_arm(m, shared + 1.0 - (0.8 if m == max(ENTRY_ARMS) else 0.0))
                 for m in ENTRY_ARMS)
    spent = RaceRecord(arms=arms, kind='screen',
                       expansions_used=CFG.max_expansions)
    # Out of expansions: take the edge arm as the candidate rather than looping.
    assert decide_screen(spent, CFG).action == 'candidate'


def test_shifted_bracket_preserves_spacing_and_direction():
    up = shifted_bracket(ENTRY_ARMS, 'up')
    dn = shifted_bracket(ENTRY_ARMS, 'down')
    span = max(ENTRY_ARMS) / min(ENTRY_ARMS)
    assert min(up) == pytest.approx(min(ENTRY_ARMS) * span)
    assert max(dn) == pytest.approx(max(ENTRY_ARMS) / span)
    assert rung_dex(up) == pytest.approx(rung_dex(ENTRY_ARMS))


def test_resolution_is_half_a_rung():
    """The honest precision of the scheme, stated as a test so it cannot drift
    out of the documentation unnoticed."""
    assert rung_dex(FINE_ARMS) == pytest.approx(math.log10(2.0))
    assert rung_dex(FINE_ARMS) / 2 == pytest.approx(0.1505, abs=1e-4)


# --------------------------------------------------------------- gate 6.1

CELLS = {
    'flat': SimSpec(curvature_hot=0.0, curvature_cold=0.0),
    'plateau': SimSpec(flat_halfwidth=0.3),
    'symmetric': SimSpec(),
    'asym_3': SimSpec(curvature_hot=3.0),
    'asym_10': SimSpec(curvature_hot=10.0),
    'skew': SimSpec(unit_skew=0.8),
    'skew_slope': SimSpec(skew_slope=1.5),
    'scale_drift': SimSpec(scale_drift=0.05),
    'hazard': SimSpec(hazard_max=0.3),
    'sprinter': SimSpec(transient_hot=0.3),
}

#: Cells with a well-defined optimum. The globally flat cell is excluded from
#: the OFFSET criteria on purpose and not by convenience: there, every rate is
#: equivalent by construction, so "distance from the optimum" names nothing.
#: What matters on a flat surface is the drift bound (asserted for every cell)
#: and, for the realistic version of the same question, the `plateau` cell --
#: flat near the optimum, curved outside -- which DOES carry the offset bars.
CURVED = [k for k in CELLS if k != 'flat']


@pytest.mark.parametrize('cell', list(CELLS))
def test_gate_drift_bound(cell):
    """TOST: the 95% CI on per-probe drift lies inside +-0.005 dex.

    A point null of "exactly zero drift" is untestable for a stochastic policy
    (the sampling sd of net drift over 10k probes is ~3 dex), which is why the
    bar is an interval.
    """
    r = RaceSim(CELLS[cell], seed=101).run(3000, entry_first=False)
    lo, hi = r['drift_ci']
    assert lo > -DRIFT_BOUND and hi < DRIFT_BOUND, (
        f'{cell}: drift {r["drift_per_probe"]:+.5f} CI [{lo:+.5f},{hi:+.5f}]')


@pytest.mark.parametrize('cell', CURVED)
def test_gate_stationary_behaviour(cell):
    r = RaceSim(CELLS[cell], seed=202).run(3000, entry_first=False)
    assert abs(r['mean_e']) < OFFSET_BOUND, f'{cell}: offset {r["mean_e"]:+.3f}'
    assert r['sd_e'] < SD_BOUND, f'{cell}: sd {r["sd_e"]:.3f}'
    assert r['p_far'] <= P_FAR_BOUND, f'{cell}: P(far) {r["p_far"]:.4f}'


def test_gate_centering_survives_asymmetric_curvature():
    """The bar the rejected margin failed.

    A margin denominated in loss units puts the hold band off-centre whenever
    the loss is steeper hot than cold, and the displacement grows with the
    margin -- the safeguard manufacturing the cold-running it was meant to
    prevent. The design carries no margin, and this is the evidence that it
    does not need one.
    """
    for ratio in (1.0, 3.0, 10.0):
        r = RaceSim(SimSpec(curvature_hot=ratio), seed=303).run(3000, entry_first=False)
        assert abs(r['mean_e']) < OFFSET_BOUND, f'C+/C- = {ratio}: {r["mean_e"]:+.3f}'


@pytest.mark.parametrize('cold_factor', [8.0, 800.0])
def test_gate_entry_event_escapes_a_mis_set_seed(cold_factor):
    """One entry event must do the bulk of the escape, in both directions.

    A HOLD is a legitimate outcome and is not counted as a failure: the screen
    is deliberately biased toward the incumbent (it scores exactly zero against
    itself while every challenger's score carries noise), so a miss defers to
    the next scheduled probe rather than making a wrong move. What must never
    happen is an entry event that leaves the rate FURTHER out than it found it.
    """
    for sign in (-1.0, 1.0):
        e0 = sign * -math.log10(cold_factor)
        moved = []
        for seed in range(16):
            sim = RaceSim(SimSpec(), seed=400 + seed)
            e_after, _, _ = sim.entry_event(e0)
            moved.append(abs(e0) - abs(e_after))
        assert min(moved) >= 0.0, (
            f'{cold_factor}x sign {sign}: an entry event moved the rate further '
            f'out (worst {min(moved):+.3f})')
        assert np.median(moved) > 0.6 * abs(e0), (
            f'{cold_factor}x sign {sign}: median closure {np.median(moved):.3f} '
            f'of {abs(e0):.3f}')


def _settle(e0, n_probes, seed, spec=None, entry_first=True):
    """Run the policy from `e0` and return the final log-error."""
    sim = RaceSim(spec or SimSpec(), seed=seed)
    e, first = e0, entry_first
    for _ in range(n_probes):
        e = (sim.entry_event(e) if first else sim.fine_probe(e))[0]
        first = False
    return e


@pytest.mark.parametrize('factor', [8.0, 800.0])
def test_gate_full_convergence_from_cold_and_hot(factor):
    """An entry event plus a few fine probes must land within one rung, from
    both directions, on the median seed."""
    for sign in (-1.0, 1.0):
        finals = [abs(_settle(sign * -math.log10(factor), 6, 500 + s))
                  for s in range(12)]
        assert np.median(finals) <= rung_dex(FINE_ARMS) * (1 + 1e-9), (
            f'{factor}x sign {sign}: median |e| {np.median(finals):.3f} '
            f'after 6 probes')


def test_gate_post_move_correction_is_bounded():
    """After a wrong move, the rule must not compound it, and must correct it
    once the error is large enough to be resolvable.

    NOTE THE HONEST BAR. At the simulator's SNR a ONE-rung error is barely
    detectable (measured: replicate-level SNR 0.23, so ~3% power per probe),
    and no amount of test design fixes that -- it is a property of the signal,
    not of the rule. What the rule must guarantee is that the error does not
    GROW, and that a two-rung error, where power exists (SNR 0.66), comes back.
    The one-rung residual is the scheme's honest resolution, and it is why
    section 8 of the design claims +-half-to-one rung rather than a factor.

    This is also the criterion that would catch a stale-evidence design: a rule
    can have zero net drift and still be slow to undo its own mistakes, because
    the evidence that justified the move outlives it. (That is why cross-probe
    pooling was deferred -- keyed relative to the incumbent, it cancelled
    exactly the evidence that would have corrected it.)
    """
    rung = rung_dex(FINE_ARMS)
    for sign in (-1.0, 1.0):
        one = [abs(_settle(sign * rung, 8, 600 + s, entry_first=False))
               for s in range(16)]
        assert np.median(one) <= rung * (1 + 1e-9), (
            f'sign {sign}: a one-rung error grew to {np.median(one):.3f}')
        two = [abs(_settle(sign * 2 * rung, 8, 700 + s, entry_first=False))
               for s in range(16)]
        assert np.median(two) <= rung * (1 + 1e-9), (
            f'sign {sign}: a two-rung error did not come back '
            f'({np.median(two):.3f})')


def test_gate_required_snr_is_reported_and_met():
    """The deliverable for the GPU phase: what the trainer must supply.

    The rule's power is a function of ONE number -- the replicate-level signal
    to noise ratio, `mean(replicate advantage) / sd(replicate advantage)` -- and
    that number is a property of the window length, the harvest and the surface,
    none of which this simulator knows. So the gate states the requirement, and
    the W-sweep on real data is what has to meet it.

    Pinned here so it cannot drift silently out of the design document.
    """
    from statistics import NormalDist
    from math import comb
    need = min_favoring(CFG.replicates, CFG.alpha / 2)
    assert need == 9 and CFG.replicates == 10

    def power(snr):
        p = NormalDist().cdf(snr)
        return sum(comb(CFG.replicates, k) * p ** k * (1 - p) ** (CFG.replicates - k)
                   for k in range(need, CFG.replicates + 1))

    # False-move rate per challenger with no signal at all.
    assert power(0.0) == pytest.approx(11 / 1024, rel=1e-6)
    # The requirement the W-sweep must hit: SNR >= 2 per contrast gives >= 95%
    # power, SNR 1.2 gives ~2/3, SNR 0.5 is not usable.
    assert power(2.0) > 0.95
    assert 0.6 < power(1.2) < 0.75
    assert power(0.5) < 0.2


def test_gate_broken_pairing_holds():
    """When the nondeterminism floor swamps the contrast, hold -- do not guess."""
    rng = np.random.default_rng(13)
    shared = _shared(rng, 0.2)
    arms = (_arm(1.0, shared + 1.0), _arm(2.0, shared + 0.4))
    rec = RaceRecord(arms=arms, kind='fine', duplicate_spread=10.0)
    d = decide_fine(rec, CFG)
    assert not d.moves and d.action == 'invalid'


def test_gate_the_rejected_v1_rule_must_fail():
    """The negative control. If this ever passes, the gate has gone blind.

    Note WHICH bar convicts it: v1 climbs to a hot equilibrium and then sits
    there, so its long-run DRIFT RATE is small and the drift bound alone would
    certify it. The stationary-offset bar is what sees the disease. That is
    exactly why 6.1 carries both.
    """
    v1 = run_v1(3000, seed=707)
    assert abs(v1['mean_e']) > OFFSET_BOUND, (
        f'v1 offset {v1["mean_e"]:+.3f} is inside the bar the shipping rule '
        f'must meet -- the control has stopped controlling')
    ours = RaceSim(SimSpec(), seed=707).run(3000, entry_first=False)
    assert abs(ours['mean_e']) < abs(v1['mean_e']) / 3.0


def test_gate_power_is_not_bought_by_never_moving():
    """A rule that never moves passes every drift test and fails the mission.

    So: on each noise cell, a genuinely mis-set rate must still be corrected.
    This is the counterweight to every conservatism bar above.
    """
    for cell in ('symmetric', 'skew', 'skew_slope', 'hazard', 'sprinter'):
        fixed = []
        for seed in range(8):
            sim = RaceSim(CELLS[cell], seed=800 + seed)
            e, _, _ = sim.entry_event(-math.log10(8.0))
            fixed.append(abs(e) < abs(math.log10(8.0)))
        assert sum(fixed) >= 6, f'{cell}: only {sum(fixed)}/8 seeds escaped 8x cold'
