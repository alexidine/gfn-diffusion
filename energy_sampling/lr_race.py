"""
Replay Racing -- the decision layer for discrete LR calibration.

Design: `docs/design/lr_probe_protocol.md` (rev d). This module is the whole of
section 3 and the scoring half of section 2.4, and NOTHING else: it is a pure
function of a probe record, with no torch, no trainer, and no state beyond what
the caller hands it. That is deliberate -- the two controllers this replaces
both failed in their DECISION ARITHMETIC, not in their sensors, so the
arithmetic is built where it can be hammered by simulation before a GPU is
involved (`bench/race_sim.py`, gate 6.1).

WHAT A RACE IS. At a calibration point the trainer snapshots itself, then runs
several short trials ("windows") at candidate LR multipliers over data harvested
verbatim from the preceding live steps. Every arm trains on the identical frozen
minibatch sequence, so the arms are exactly paired; every arm is scored on the
same held-out slice. The trainer then restores the snapshot and resumes at the
selected rate. This module decides what "selected" means.

THE FOUR RULES, each of which is a scar:

1. THE INCUMBENT IS THE NULL. A challenger must beat the CURRENT rate; it is
   never enough to be "not significantly worse than the best arm". That rule
   (v1 of this design) promotes on noise with probability ~1-alpha per probe and
   drifts +0.2 dex per probe -- the third appearance of the rectification family
   that railed `hyper` into its floor and holds `ray` ~15% hot of setpoint.

2. SYMMETRIC IN BOTH DIRECTIONS. Same test, same burden, same confirmation for a
   raise and for a cut. Every asymmetric response in this project's history --
   `hyper_down_gain 2.0`, ray's `eta_up 0.25 / eta_down 0.5`, and an earlier
   revision of THIS document that confirmed only raises -- rectified zero-mean
   noise into a one-way ratchet.

3. THE TEST IS AN EXACT BINOMIAL SIGN TEST. Not a t-test (gradient noise here is
   heavy-tailed by the same argument that justifies gradient clipping) and not a
   sign-flip permutation test (exact only under a symmetry assumption on the
   differences that nothing validates, and a symmetric-noise simulator cannot
   detect its violation). The sign test needs only independence and a zero null
   median. It is conservative; the moves it gates are rung-sized, so
   conservative is right.

4. THE INDEPENDENT UNIT IS THE REPLICATE, NOT THE HELD-OUT BATCH -- and this is
   the correction that matters most. Every arm in replicate j trains on the same
   sub-larder in the same order, so the luck of that path is a single offset
   shared by ALL of that replicate's held-out scores. Held-out batches within a
   replicate therefore carry one piece of evidence about the learning rate, not
   S pieces. Measured in the simulator: testing at the batch level ran a 4.2%
   false-move rate against a 1.07% nominal, because the batch scores are not
   independent draws. Batches still earn their cost -- averaging over them is
   what makes each replicate's score precise -- but the COUNT that enters the
   test is the number of replicates. (Same disease, one level up, as `ray`'s own
   t inflating ~1.7x on sub-batch overlap.)

NO EXTRAPOLATION AND NO BIAS DIVISOR. A move goes to a tested arm's rate,
exactly. Dividing a discrete winner by a calibration factor and re-snapping to
the lattice is incoherent -- a systematic 2x correction turns every 2x win into
a hold and freezes adaptation. The frozen-data probe's bias (direction unknown a
priori: short stochastic horizons bias cold, absent on-policy feedback biases
hot) is bounded by choosing the window length against on-policy forks, not by
per-move arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

#: Candidate multipliers for a routine in-stage race. Symmetric around the
#: incumbent in log space; one rung = log10(2) = 0.301 dex.
FINE_ARMS = (0.5, 1.0, 2.0)

#: Candidate multipliers for a stage-entry race. Wide and log-spaced (0.602 dex
#: rungs): the job here is escaping a badly mis-set seed, and probe SNR grows
#: with distance from the optimum, so a wide screen resolves a cold start that a
#: fine bracket would need many probes to walk to.
ENTRY_ARMS = (0.25, 1.0, 4.0, 16.0, 64.0)

#: The incumbent is always present as the 1x arm: every contrast is against it.
INCUMBENT = 1.0


@dataclass(frozen=True)
class RaceConfig:
    """Constants, not knobs.

    These are frozen by the simulation gate (bench/race_sim.py) and the W-sweep,
    not exposed per-route: a knob that can express a disqualified configuration
    (an asymmetric bracket, a one-sided confirmation) is a liability rather than
    flexibility. Only `enabled` and `window` are config surface.
    """
    #: Family-wise level for one race, one-sided. Split across challengers.
    alpha: float = 0.05
    #: Replicate windows per arm in a tested (non-screen) race. Each replicate
    #: trains on a DISJOINT sub-larder with its own preregistered order, so the
    #: spread across replicates resamples both data and order -- a same-larder
    #: re-run resamples only order and is ~circular with the selection that
    #: triggered it.
    #:
    #: THIS IS ALSO THE SAMPLE SIZE OF THE TEST (see rule 4), so it cannot be
    #: cut to save cost without losing the ability to decide anything: at r=3
    #: even a clean sweep is p = 1/8 = 0.125 and no verdict can reach any
    #: sensible level. 10 gives a 9-of-10 critical value at a realised 1.07%.
    replicates: int = 10
    #: Replicates per SCREEN arm. The screen only selects, so it needs no
    #: critical value -- but it does need to not MISS. Measured: at r=1 a screen
    #: on an 8x-hot incumbent returns `hold` 4.0% of the time, because the
    #: incumbent scores exactly 0 against itself (noise-free) while every
    #: challenger's score carries noise, so noise can only push a challenger
    #: BELOW the incumbent, never above. That asymmetry is conservative and
    #: therefore welcome; but a missed entry event costs a whole calibration,
    #: and r=2 drops the miss rate to 0.5% for five extra windows.
    screen_replicates: int = 2
    #: Max bracket expansions per entry event, each direction.
    max_expansions: int = 2
    #: On a nondeterministic route the duplicate-null spread must be small
    #: relative to the winning contrast, or the race cannot see its own signal.
    duplicate_ratio_max: float = 0.5

    #: Screen advantages within this FRACTION of the best count as tied, and a
    #: tie is broken toward the SMALLEST CHANGE from the incumbent.
    #:
    #: Measured, run race_L1_hot4x_W30 (elj, 4x hot seed, W=30): once the
    #: incumbent itself diverges, every surviving arm scores "did not diverge"
    #: and they land within 0.3% of each other (+5139.8 .. +5155.1 across a
    #: 256x span of rates). Taking the argmax of that is taking the argmax of
    #: noise, and it selected a 64x cut where a 4x cut was the evidence.
    #:
    #: Note this is NOT the rejected v1 rule in mirror image: v1 took the most
    #: AGGRESSIVE arm among those competitive with the best, which is a
    #: one-directional bias. Nearest-the-incumbent is direction-neutral -- it
    #: shrinks over-large moves upward and downward alike -- so it cannot
    #: rectify noise into drift.
    #: 2%, not 10%: the degenerate case measured had a 0.3% spread, so this
    #: catches it with an order of magnitude to spare, while a 10% band also
    #: collapsed GENUINE orderings and slowed convergence from an extreme
    #: mis-set seed past its gate (800x landed at 0.495 dex after 6 probes
    #: against a one-rung bar of 0.301).
    tie_fraction: float = 0.02


@dataclass(frozen=True)
class ArmScores:
    """Held-out losses for one arm of one race.

    `mid` and `end` are [replicate][unit] held-out losses at the half-window and
    end-of-window points. Losses, so LOWER IS BETTER. The half-window scores
    exist for the sign-consistency check: an arm that wins by sprinting and then
    diverging looks identical to a genuine winner at the end point alone.

    A screen arm has one replicate. `died` marks an arm whose window went
    nonfinite or exploded -- recorded, never silently dropped.
    """
    multiplier: float
    mid: tuple[tuple[float, ...], ...]
    end: tuple[tuple[float, ...], ...]
    died: bool = False

    def n_replicates(self) -> int:
        return len(self.end)

    def n_units(self) -> int:
        return len(self.end[0]) if self.end else 0


@dataclass(frozen=True)
class RaceRecord:
    """Everything one race measured. The complete record is logged whatever the
    verdict -- vetoes gate ACTUATION, never data. A verdict-only log would make
    every later re-analysis impossible, including the offline pooling that is
    the named upgrade path (appendix A of the design)."""
    arms: tuple[ArmScores, ...]
    #: 'fine' -- tested in-stage race. 'screen' -- entry, r=1, selection only,
    #: no test. 'confirm' -- entry, one pre-selected candidate vs the incumbent
    #: on reserved sub-larders, so no multiplicity correction.
    kind: str = 'fine'
    #: Trainer-side isolation assertions (RNG streams unchanged where they must
    #: be, arm scope not clipped by a rail, controllers frozen).
    isolation_ok: bool = True
    #: Same-order duplicate of the 1x arm. `None` = not run. On a deterministic
    #: route this must be exactly 0.0 (the restore certificate); on a GPU route
    #: it is the nondeterminism floor and is compared to the winning contrast.
    duplicate_spread: float | None = None
    #: True where the route is expected to reproduce bitwise (CPU / bench).
    expect_bitwise: bool = False
    #: How many expansions this entry event has already spent, each direction.
    expansions_used: int = 0
    note: str = ''

    def arm(self, multiplier: float) -> ArmScores | None:
        for a in self.arms:
            if a.multiplier == multiplier:
                return a
        return None

    def multipliers(self) -> tuple[float, ...]:
        return tuple(a.multiplier for a in self.arms)


@dataclass(frozen=True)
class Decision:
    """
    action:
      'hold'        -- no evidence to move; the default and the safe state.
      'move'        -- adopt `multiplier` (relative to the incumbent's rate).
      'candidate'   -- screen only: `multiplier` should go to a confirm race.
      'expand_up'   -- screen only: the best arm sits at the top edge; shift the
                       bracket up and re-screen within this same event.
      'expand_down' -- as above, downward. Symmetric by construction.
      'invalid'     -- the race could not be trusted; hold, and say why.
    """
    action: str
    multiplier: float = INCUMBENT
    reason: str = ''
    detail: dict = field(default_factory=dict)

    @property
    def moves(self) -> bool:
        return self.action == 'move'


# --------------------------------------------------------------- the sign test

def sign_sf(k: int, n: int) -> Fraction:
    """P(X >= k) for X ~ Binomial(n, 1/2), exactly.

    Exact rationals rather than a normal approximation: n here is 10-20, where
    the approximation is worst and where the whole point of the test is that its
    level is guaranteed rather than asymptotic.
    """
    if n <= 0:
        return Fraction(1)
    k = max(0, min(k, n + 1))
    return Fraction(sum(math.comb(n, i) for i in range(k, n + 1)), 2 ** n)


@lru_cache(maxsize=512)
def min_favoring(n: int, alpha: float) -> int | None:
    """Smallest count of favoring units that clears `alpha` one-sided on n units.

    Returns None when no count can -- i.e. even a clean sweep is not significant
    at this level, so the race is UNDERPOWERED and can never move. That is a
    real, reportable condition and not an error: the answer is more scoring
    units (which cost only a forward pass), never a weaker test.

    Computed, never transcribed. At the defaults: n=10 needs 9 at both
    alpha/4 = 0.0125 and alpha = 0.05; n=20 needs 16 and 15 respectively.
    """
    a = Fraction(alpha).limit_denominator(10 ** 9)
    best = None
    for k in range(n, 0, -1):
        if sign_sf(k, n) <= a:
            best = k
        else:
            break
    return best


# -------------------------------------------------------------- paired scoring

def _paired(challenger: ArmScores, incumbent: ArmScores, point: str = 'end'):
    """[replicate][unit] advantage of `challenger` over `incumbent`.

    Positive = challenger reached a LOWER held-out loss. Every component shared
    by the two arms -- the harvested batch's own difficulty, the condition's
    loss level, the run's overall scale -- cancels here. That cancellation IS
    the sensor: unpaired scoring of the same quantity was measured at ~1/30th
    the t on this system, with the wrong sign.
    """
    c = getattr(challenger, point)
    b = getattr(incumbent, point)
    n_rep = min(len(c), len(b))
    out = []
    for j in range(n_rep):
        n_unit = min(len(c[j]), len(b[j]))
        out.append(tuple(b[j][i] - c[j][i] for i in range(n_unit)))
    return tuple(out)


def _replicate_advantage(challenger: ArmScores, incumbent: ArmScores,
                         point: str = 'end') -> tuple[float, ...]:
    """One advantage per REPLICATE window, averaged over its held-out batches.

    This is the vector the test runs on. Averaging over batches sharpens each
    replicate's score; it does not create additional independent evidence,
    because every batch in a replicate is scored against the same trained
    parameters and therefore carries the same training-path luck.
    """
    return tuple(sum(r) / len(r) if r else 0.0
                 for r in _paired(challenger, incumbent, point))


def _unit_advantage(challenger: ArmScores, incumbent: ArmScores,
                    point: str = 'end') -> tuple[float, ...]:
    """One advantage per held-out batch, averaged over replicates.

    DIAGNOSTIC AND SELECTION ONLY -- never the sample for a test (see the
    module docstring, rule 4). Used for the screen's ranking, where nothing is
    being tested, and for logging an effect size.
    """
    rows = _paired(challenger, incumbent, point)
    if not rows:
        return ()
    n_unit = min(len(r) for r in rows)
    return tuple(sum(r[i] for r in rows) / len(rows) for i in range(n_unit))


def _half_window_ok(challenger: ArmScores, incumbent: ArmScores) -> bool:
    """The advantage must be positive in BOTH halves of the window, pooled.

    First half is the advantage at the mid-window score; second half is the
    increment from mid to end. An arm that leads at the end purely because it
    sprinted early and then gave ground is a transient, not a better rate -- and
    on this system a hot rate's transient is what precedes an excursion.

    POOLED ACROSS REPLICATES, not per replicate. Requiring it of every replicate
    separately is a conjunction of 2r noisy conditions: measured in the
    simulator, that version passed only 33.5% of the time on a signal the sign
    test resolved 87% of the time, i.e. the guard was rejecting the mission
    rather than the failure mode. Two pooled conditions keep the sprinter test
    (its second-half deficit is systematic) without spending the power.
    """
    mid = _replicate_advantage(challenger, incumbent, 'mid')
    end = _replicate_advantage(challenger, incumbent, 'end')
    if not mid or not end:
        return False
    m = sum(mid) / len(mid)
    e = sum(end) / len(end)
    return m > 0.0 and (e - m) > 0.0


def _evaluate(challenger: ArmScores, incumbent: ArmScores, alpha: float,
              require_all_replicates: bool = False) -> dict:
    """The contrast: an exact sign test over replicates, plus the half-window
    check. `require_all_replicates` exists for the rejected-rule control in
    bench/race_sim.py; the shipping path does not use it."""
    rep = _replicate_advantage(challenger, incumbent)
    # Ties carry no directional information; the exact conditional sign test
    # drops them and shrinks n rather than assigning them to either side.
    nz = [v for v in rep if v != 0.0]
    favor = sum(1 for v in nz if v > 0.0)
    need = min_favoring(len(nz), alpha)
    halves = _half_window_ok(challenger, incumbent)
    adv = _unit_advantage(challenger, incumbent)
    passed = (need is not None
              and favor >= need
              and halves
              and (all(v > 0.0 for v in rep) if require_all_replicates else True))
    return {
        'multiplier': challenger.multiplier,
        'n_replicates': len(rep),
        'n_nonzero': len(nz),
        'favoring': favor,
        'need': need,
        'underpowered': need is None,
        'replicate_adv': rep,
        'half_window_ok': halves,
        'mean_advantage': (sum(adv) / len(adv)) if adv else 0.0,
        'passed': passed,
    }


# ------------------------------------------------------------------ validity

def _validity(record: RaceRecord, cfg: RaceConfig,
              winning_effect: float | None = None) -> str | None:
    """None if the race may actuate, else the reason it may not."""
    if not record.isolation_ok:
        return 'isolation'
    inc = record.arm(INCUMBENT)
    if inc is None:
        return 'no_incumbent_arm'
    if inc.died:
        return 'incumbent_died'
    d = record.duplicate_spread
    if d is not None:
        if not math.isfinite(d):
            return 'duplicate_nonfinite'
        if record.expect_bitwise and d != 0.0:
            # A deterministic route that does not reproduce has a restore leak,
            # and a restore leak is indistinguishable from a real effect from
            # the outputs alone. This is the certificate for the whole fork.
            return 'duplicate_not_bitwise'
        if winning_effect is not None and abs(winning_effect) > 0.0:
            if d > cfg.duplicate_ratio_max * abs(winning_effect):
                return 'duplicate_rivals_effect'
    return None


# ---------------------------------------------------------------- the verdicts

def decide(record: RaceRecord, cfg: RaceConfig = RaceConfig()) -> Decision:
    """Dispatch on race kind. The only entry point the trainer needs."""
    if record.kind == 'screen':
        return decide_screen(record, cfg)
    if record.kind == 'confirm':
        return decide_confirm(record, cfg)
    return decide_fine(record, cfg)


def decide_fine(record: RaceRecord, cfg: RaceConfig = RaceConfig()) -> Decision:
    """A routine in-stage race: challengers tested against the incumbent.

    TIE-BREAK IS THE NEAREST CLEARING ARM, not the strongest. Picking the
    largest statistic over several correlated contrasts selects outward-biased
    noise (E[max] of m correlated draws sits ~1 sd above the truth), and
    outward means "further from the rate that is currently working". Nearest is
    the conservative direction and it is pre-declared, so it cannot be chosen
    after seeing the numbers.
    """
    inc = record.arm(INCUMBENT)
    bad = _validity(record, cfg)
    if bad is not None:
        return Decision('invalid', reason=bad)

    challengers = [a for a in record.arms
                   if a.multiplier != INCUMBENT and not a.died]
    died = [a.multiplier for a in record.arms if a.died]
    if not challengers:
        return Decision('hold', reason='no_live_challengers',
                        detail={'died': died})

    # Correct for the challengers actually compared, not for the bracket size:
    # an arm that died was never tested.
    m = len(challengers)
    level = cfg.alpha / m
    results = [_evaluate(c, inc, level) for c in challengers]
    winners = [r for r in results if r['passed']]

    if not winners:
        return Decision('hold', reason='no_significant_challenger',
                        detail={'results': results, 'died': died, 'level': level})

    winners.sort(key=lambda r: abs(math.log(r['multiplier'])))
    best = winners[0]
    effect = best['mean_advantage']
    bad = _validity(record, cfg, winning_effect=effect)
    if bad is not None:
        return Decision('invalid', reason=bad, detail={'results': results})
    return Decision('move', multiplier=best['multiplier'],
                    reason='significant_vs_incumbent',
                    detail={'results': results, 'died': died, 'level': level})


def decide_screen(record: RaceRecord, cfg: RaceConfig = RaceConfig()) -> Decision:
    """Entry race, phase 1: SELECT, do not test.

    The screen runs the wide arms once each and picks a single candidate; the
    evidence for moving comes later, from a confirm race on data the screen
    never touched. Splitting selection from testing is what makes an entry event
    able to jump 64x without inheriting the winner's curse of maximising over
    five noisy arms -- and it resolves a contradiction in the previous revision,
    where "nearest clearing challenger" and "top-rung winner rebrackets" gave
    different answers whenever several wide arms all beat a cold incumbent.

    Selection is on the END score only. The half-window check belongs to the
    confirm race, where it gates an actual move.
    """
    bad = _validity(record, cfg)
    if bad is not None:
        return Decision('invalid', reason=bad)

    live = [a for a in record.arms if not a.died]
    died = [a.multiplier for a in record.arms if a.died]
    if not live:
        return Decision('invalid', reason='all_arms_died', detail={'died': died})

    inc = record.arm(INCUMBENT)
    scored = []
    for a in live:
        if a.multiplier == INCUMBENT:
            scored.append((0.0, a.multiplier))
            continue
        adv = _unit_advantage(a, inc)
        scored.append((sum(adv) / len(adv) if adv else 0.0, a.multiplier))
    scored.sort(key=lambda t: (-t[0], abs(math.log(t[1]))))
    best_adv, best_mult = scored[0]

    # Collapse a near-tie to the smallest change. Ranking arms that differ only
    # by noise is how a screen turns "the incumbent is too hot" into an
    # arbitrary multiplier.
    if best_adv > 0.0:
        band = abs(best_adv) * float(cfg.tie_fraction)
        tied = [(adv, m) for adv, m in scored if adv >= best_adv - band]
        if len(tied) > 1:
            tied.sort(key=lambda t: abs(math.log(t[1])))
            best_mult = tied[0][1]
            best_adv = next(a for a, m in scored if m == best_mult)

    if best_mult == INCUMBENT:
        return Decision('hold', reason='screen_favours_incumbent',
                        detail={'scored': scored, 'died': died})

    # An edge winner is a BRACKETING verdict, not a resolution: the optimum may
    # lie beyond the arms tested. Expansion is symmetric -- a bottom-edge
    # winner shifts down exactly as a top-edge winner shifts up -- because an
    # up-only expansion is a one-way ratchet wearing a different hat.
    live_mults = sorted(a.multiplier for a in live)
    if record.expansions_used < cfg.max_expansions:
        if best_mult == live_mults[-1]:
            return Decision('expand_up', multiplier=best_mult,
                            reason='screen_winner_at_top_edge',
                            detail={'scored': scored, 'died': died})
        if best_mult == live_mults[0]:
            return Decision('expand_down', multiplier=best_mult,
                            reason='screen_winner_at_bottom_edge',
                            detail={'scored': scored, 'died': died})

    return Decision('candidate', multiplier=best_mult,
                    reason='screen_selected',
                    detail={'scored': scored, 'died': died,
                            'advantage': best_adv})


def decide_confirm(record: RaceRecord, cfg: RaceConfig = RaceConfig()) -> Decision:
    """Entry race, phase 2: ONE pre-selected candidate vs the incumbent, on
    reserved sub-larders the screen never used.

    No multiplicity correction -- there is exactly one contrast, chosen before
    this data existed. The independence from the screen is the whole point: it
    is what absorbs the winner's curse of the max over five screen arms, and it
    is why the confirm larder must be reserved at harvest rather than reused.
    """
    bad = _validity(record, cfg)
    if bad is not None:
        return Decision('invalid', reason=bad)

    inc = record.arm(INCUMBENT)
    cand = [a for a in record.arms if a.multiplier != INCUMBENT]
    if len(cand) != 1:
        return Decision('invalid', reason='confirm_needs_exactly_one_candidate',
                        detail={'arms': record.multipliers()})
    c = cand[0]
    if c.died:
        return Decision('hold', reason='candidate_died',
                        detail={'multiplier': c.multiplier})

    res = _evaluate(c, inc, cfg.alpha)
    if not res['passed']:
        return Decision('hold', reason='candidate_not_confirmed',
                        detail={'result': res})
    bad = _validity(record, cfg, winning_effect=res['mean_advantage'])
    if bad is not None:
        return Decision('invalid', reason=bad, detail={'result': res})
    return Decision('move', multiplier=c.multiplier,
                    reason='confirmed_vs_incumbent', detail={'result': res})


# ------------------------------------------------------------ bracket helpers

def shifted_bracket(arms: tuple[float, ...], direction: str) -> tuple[float, ...]:
    """Slide a screen bracket one full span up or down, keeping its spacing.

    The incumbent (1.0) is NOT carried along: after a shift the reference for
    the next screen is the previous screen's edge winner, and the arms are
    expressed relative to the ORIGINAL incumbent so that a candidate remains a
    multiplier the caller can apply directly.
    """
    if not arms:
        return arms
    span = max(arms) / min(arms)
    factor = span if direction == 'up' else 1.0 / span
    return tuple(a * factor for a in arms)


def rung_dex(arms: tuple[float, ...] = FINE_ARMS) -> float:
    """Spacing of a bracket in dex -- the resolution of any decision made on it.

    Stated because it is the honest precision of the whole scheme: a rule that
    may only move to a tested arm cannot resolve better than half this, whatever
    the sample size. Fine bracket: 0.301 dex, so +-0.15 dex (~+-41%).
    """
    ordered = sorted(arms)
    if len(ordered) < 2:
        return 0.0
    return math.log10(ordered[1] / ordered[0])
