"""
The brute-force LR bracket's decision layer.

WHAT THESE PIN, and each one is a way the bracket could return an answer while
having measured nothing:

  * A GRID THAT CANNOT FAIL, or BARS THAT CANNOT FIRE. Either one makes every
    rung a survivor, so no boundary is ever found, `unbracketed_high` is reported
    every cycle, and the mechanism returns the same number forever while every
    seam fires correctly. That is precisely how the retired controller passed for
    working, so both are refused at construction rather than warned about.
  * A CONFIRMATION THAT IS A REPLAY. Screen trials restore an identical RNG state
    so they are comparable; re-running the lowest failure under that SAME state
    reproduces the failure by construction. It would confirm nothing and read as
    a passing check.
  * A LOOP. Confirmation and densification are the only two things that can add
    work after the screen, and neither has a convergence criterion. An unbounded
    settling gate is what the previous controller used to stay quiet while its
    sensor could not resolve.
  * A SELECTION THAT IS NOT DETERMINISTIC IN THE ORDERING. Selection reads the
    configured rung order and hard survival, and nothing else -- no final loss,
    no interpolation.
  * A HORIZON NOBODY CHECKED. `trial_steps` is deliberately short and has no
    floor, so the bracket has to say when a failure landed late enough that the
    horizon is not to be trusted.
"""

import math

import pytest

from energy_sampling.lr_bracket import (ALL_FAILED, BRACKETED, BURN_IN,
                                        CAPPED_UNCONFIRMED, CONFIRM, CRUISE,
                                        DENSIFY, NO_ELIGIBLE, SCREEN,
                                        UNBRACKETED_HIGH, LRBracket)

pytestmark = pytest.mark.fast

GRID = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)


def _b(**kw):
    kw.setdefault('candidate_scales', GRID)
    kw.setdefault('burn_in_steps', 3000)
    kw.setdefault('burn_in_scale', 0.05)
    kw.setdefault('trial_steps', 150)
    return LRBracket(**kw)


def _run(b, failing=(), step=1000, steps_to_failure=None, fail_once=()):
    """Drive a whole cycle. `failing` is the set of scales that hard-fail every
    time; `fail_once` fail on their SCREEN and survive any confirmation, which is
    the non-reproducing case."""
    b.begin_bracket(step, bias_correction=0.99)
    seen = []
    guard = 0
    while True:
        guard += 1
        assert guard < 200, 'the trial plan did not terminate'
        t = b.next_trial()
        if t is None:
            break
        seen.append(t)
        fails = t.scale in failing or (t.scale in fail_once and t.kind == SCREEN)
        b.record(t, ok=not fails, reason='loss_excursion' if fails else None,
                 steps_completed=(steps_to_failure or 10) if fails else b.trial_steps,
                 steps_to_failure=(steps_to_failure or 10) if fails else None)
    return seen, b.select()


# ---------------------------------------------------------------- burn-in ---

def test_burn_in_lasts_exactly_the_configured_number_of_steps():
    b = _b(burn_in_steps=3000)
    assert not b.burn_in_complete(2999)
    assert b.burn_in_complete(3000)


def test_burn_in_cannot_wait_indefinitely():
    """STEP COUNT ONLY. The previous controller gated on a learned metric that
    the rate itself was moving -- z_cal/p fell 15.5 -> 0.135 and went back to
    0.85, never producing the run of readings the gate wanted, while bwd/loss ran
    183 -> 2344 and the controller could not act. A circular wait is not a wait.
    """
    b = _b(burn_in_steps=10)
    assert b.burn_in_complete(10)
    # ...and it does not consult anything else: no argument but the step count.
    assert b.burn_in_complete.__code__.co_argcount == 2


def test_burn_in_scale_must_train():
    """An inert burn-in reaches the root with the transients it exists to pass
    through still running, and leaves Adam's counter advancing over gradients the
    model never followed."""
    with pytest.raises(ValueError, match='must still train'):
        _b(burn_in_scale=0.0)


# ------------------------------------------------------------ the grid gate ---

def test_a_grid_too_narrow_to_fail_is_refused():
    """The bracket's only reading is "did this detonate", so the top rungs have
    to be EXPECTED to fail. A grid inside the safe region reports
    unbracketed_high every cycle and selects one rung below its own ceiling -- a
    number the config chose, dressed as a measurement."""
    with pytest.raises(ValueError, match='spans only'):
        _b(candidate_scales=(0.1, 0.15, 0.2))


def test_a_grid_too_short_for_the_margin_is_refused():
    with pytest.raises(ValueError, match='cannot support safety_rungs'):
        _b(candidate_scales=(0.1, 1.0), safety_rungs=1)


def test_an_unsorted_grid_is_refused():
    """Selection is defined by POSITION in the ordering -- "the lowest rung that
    failed", "one rung below" -- so an unsorted grid makes the margin meaningless
    while still producing an answer."""
    with pytest.raises(ValueError, match='STRICTLY ASCENDING'):
        _b(candidate_scales=(0.05, 0.4, 0.2, 1.6))


def test_a_burn_in_hotter_than_every_candidate_is_refused():
    with pytest.raises(ValueError, match='above the top candidate rung'):
        _b(burn_in_scale=2.0)


def test_bars_that_cannot_fire_are_refused():
    """MEASURED: a rate one rung too hot on this route took the loss from about
    -25 to +318. That is finite and eight orders of magnitude below a 1e9 bar, so
    under it the candidate completes, counts as a survivor and is eligible for
    selection. A bracket that cannot fail a candidate is not a bracket."""
    with pytest.raises(ValueError, match='catches numerical overflow'):
        _b(loss_abs=1.0e9)
    with pytest.raises(ValueError, match='catches numerical overflow'):
        _b(grad_abs=1.0e9)
    _b(loss_abs=1.0e6, grad_abs=1.0e6)          # the shipping pair is accepted


# ----------------------------------------------------------------- the screen ---

def test_every_candidate_is_trialled_once_and_survivors_run_the_full_horizon():
    b = _b(boundary_confirm_repeats=0)
    seen, _ = _run(b, failing=())
    assert [t.scale for t in seen] == list(GRID)
    assert all(t.kind == SCREEN for t in seen)
    assert all(o.steps_completed == b.trial_steps for o in b.results())


def test_no_early_winner_selection():
    """A candidate that has not hard-failed runs the whole horizon. Stopping
    early on a promising one would rank by loss, which is the estimator this
    mechanism exists not to have."""
    b = _b(boundary_confirm_repeats=0)
    _run(b, failing={1.6})
    for o in b.results():
        assert o.ok == (o.steps_completed == b.trial_steps)


# --------------------------------------------------------------- selection ---

def test_normal_case_selects_one_rung_below_the_confirmed_boundary():
    b = _b()
    _, v = _run(b, failing={0.8, 1.6})
    assert v['status'] == BRACKETED
    assert v['boundary_scale'] == 0.8
    assert v['boundary_confirmed'] is True
    assert v['scale'] == 0.4
    assert v['margin_rungs'] == 1


def test_safety_rungs_two_selects_two_below():
    b = _b(safety_rungs=2)
    _, v = _run(b, failing={0.8, 1.6})
    assert v['scale'] == 0.2 and v['margin_rungs'] == 2


def test_non_monotone_outcomes_are_treated_conservatively():
    """A survivor ABOVE a failure is ignored. alpha-style reasoning would read the
    survivor as evidence the boundary is higher; the conservative rule says a
    rate that detonated once bounds everything above it."""
    b = _b(boundary_confirm_repeats=0)
    _, v = _run(b, failing={0.2, 1.6})       # 0.4 and 0.8 survive ABOVE 0.2
    assert v['boundary_scale'] == 0.2
    assert v['scale'] == 0.1, 'a survivor above the boundary must not be selected'


def test_no_failure_at_all_reports_unbracketed_high_and_does_not_claim_a_boundary():
    b = _b()
    _, v = _run(b, failing=())
    assert v['status'] == UNBRACKETED_HIGH
    assert v['boundary_scale'] is None and v['boundary_confirmed'] is False
    assert v['scale'] == 0.8, 'one rung below the highest TESTED candidate'


def test_every_candidate_failing_falls_back_to_the_root_and_the_burn_in_scale():
    b = _b(boundary_confirm_repeats=0)
    _, v = _run(b, failing=set(GRID))
    assert v['status'] == ALL_FAILED
    assert v['restore'] == 'root' and v['scale'] == b.burn_in_scale


def test_a_boundary_at_the_bottom_rung_fails_safely_rather_than_guessing():
    """Nothing survives below the margin. Falling back to the root at the
    burn-in scale is the safe answer AND it is labelled as such -- reporting it
    as `bracketed` would claim a measurement that was never made."""
    b = _b(boundary_confirm_repeats=0)
    _, v = _run(b, failing={0.05})
    assert v['status'] == NO_ELIGIBLE
    assert v['restore'] == 'root' and v['scale'] == b.burn_in_scale


def test_selection_never_interpolates_between_rungs():
    b = _b()
    _, v = _run(b, failing={0.8, 1.6})
    assert v['scale'] in v['ordering']


def test_selection_is_deterministic_across_repeated_runs():
    verdicts = []
    for _ in range(5):
        b = _b()
        _, v = _run(b, failing={0.8, 1.6})
        verdicts.append((v['status'], v['scale'], v['boundary_scale']))
    assert len(set(verdicts)) == 1


# ------------------------------------------------------------ confirmation ---

def test_the_lowest_failure_is_confirmed_before_it_bounds_the_search():
    b = _b(boundary_confirm_repeats=1)
    seen, _ = _run(b, failing={0.8, 1.6})
    confirms = [t for t in seen if t.kind == CONFIRM]
    assert [t.scale for t in confirms] == [0.8], 'only the LOWEST failure is confirmed'


def test_a_confirmation_rerun_is_not_a_replay_of_the_original_trial():
    """THE SEED MUST DIFFER, AND ONLY THE SEED. Screen trials restore an
    identical RNG state so the candidates are comparable -- which makes a
    same-seed re-run a deterministic replay: it reproduces the failure by
    construction, confirms nothing, and reads as a passing check."""
    b = _b(boundary_confirm_repeats=2)
    seen, _ = _run(b, failing={0.8, 1.6}, step=4242)
    screen = next(t for t in seen if t.kind == SCREEN and t.scale == 0.8)
    confirms = [t for t in seen if t.kind == CONFIRM]

    assert screen.seed is None, 'a screen trial restores the captured RNG state'
    assert all(t.seed is not None for t in confirms)
    assert len({t.seed for t in confirms}) == len(confirms), 'repeats must differ'
    # ...and ONLY the seed differs: same rate, same horizon, same root.
    for t in confirms:
        assert t.scale == screen.scale
    assert b.trial_steps == 150


def test_the_confirmation_seed_is_derived_from_the_root_not_from_the_clock():
    """Reproducible without being a replay. Deriving it from wall-clock or
    leaning on GPU kernel nondeterminism would make the confirmation
    unreproducible, which is a different way of confirming nothing."""
    a, c = _b(), _b()
    a.begin_bracket(4242)
    c.begin_bracket(4242)
    assert a.confirm_seed(1) == c.confirm_seed(1)
    assert a.confirm_seed(1) != a.confirm_seed(2)
    d = _b()
    d.begin_bracket(9999)
    assert d.confirm_seed(1) != a.confirm_seed(1)


def test_a_failure_that_does_not_reproduce_does_not_bound_the_search():
    """One unlucky low rung would otherwise truncate the whole bracket far below
    the real boundary. The rung is recorded as a non-reproducing failure, the
    search continues upward -- and it is NOT a survivor either, because it did
    detonate once.

    THE SELECTION IS STILL CAPPED BY IT, and this assertion used to say the
    opposite. It expected 0.8 -- four rungs above a rung that had blown up --
    because the code conflated the BOUNDARY (where the search stops, which a
    non-reproducing failure must not set) with the CEILING (what may be run,
    where one detonation is enough). See
    `test_a_non_reproducing_failure_still_caps_the_selection`."""
    b = _b(boundary_confirm_repeats=1)
    _, v = _run(b, failing={1.6}, fail_once={0.2})
    assert v['non_reproducing'] == [0.2]
    assert v['boundary_scale'] == 1.6, 'the search must still reach the real boundary'
    assert 0.2 not in v['survivors']
    assert v['selection_ceiling'] == 0.2
    assert v['scale'] == 0.1, 'one rung below the lowest detonation'


def test_a_non_reproducing_failure_still_caps_the_selection():
    """FINDING C, and it inverted the conservative rule.

    A rung that detonates on its screen and survives its confirmation is
    correctly excluded from the BOUNDARY -- that is what confirming is for, and
    one unlucky trial must not truncate the search. But it must still cap what
    may be SELECTED, and it did not: with 0.2 detonating and nothing else
    failing, `select` reported `unbracketed_high` -- the claim that nothing
    failed -- and chose 0.8, four times the rate that had blown up."""
    b = _b(boundary_confirm_repeats=1)
    _, v = _run(b, failing=(), fail_once={0.2})

    assert v['non_reproducing'] == [0.2]
    assert v['scale'] < 0.2, (
        f"selected {v['scale']}, at or above a rung that detonated")
    assert v['selection_ceiling'] == 0.2
    # NOT `bracketed`: no failure reproduced, so no boundary was confirmed. NOT
    # `unbracketed_high` either: something did blow up. It is its own state.
    assert v['status'] == CAPPED_UNCONFIRMED
    assert v['boundary_scale'] is None


def test_a_non_reproducing_failure_still_lets_the_search_continue_upward():
    """The other half: capping the SELECTION must not also truncate the SEARCH.
    A higher rung that reproduces its failure is still the boundary."""
    b = _b(boundary_confirm_repeats=1)
    _, v = _run(b, failing={1.6}, fail_once={0.2})
    assert v['boundary_scale'] == 1.6, 'the search stopped at the unlucky rung'
    assert v['non_reproducing'] == [0.2]
    assert v['status'] == BRACKETED, 'a failure DID reproduce, so there is a boundary'
    # ...and the ceiling is still the lowest detonation, not the boundary.
    assert v['selection_ceiling'] == 0.2
    assert v['scale'] < 0.2, 'the selection is still capped by the detonation'


def test_confirmation_is_hard_capped_and_cannot_loop():
    b = _b(boundary_confirm_repeats=3)
    seen, v = _run(b, failing={0.8, 1.6})
    confirms = [t for t in seen if t.kind == CONFIRM]
    assert len(confirms) == 3
    assert b.next_trial() is None, 'the plan must be exhausted'
    assert v['boundary_scale'] == 0.8


def test_zero_confirm_repeats_lets_one_failure_decide():
    """Stated rather than defaulted: this is the configuration in which a single
    unlucky trial bounds the whole search."""
    b = _b(boundary_confirm_repeats=0)
    seen, v = _run(b, failing={0.8})
    assert not [t for t in seen if t.kind == CONFIRM]
    assert v['boundary_scale'] == 0.8


# ------------------------------------------------------------ densification ---

def test_densify_inserts_exactly_one_rung_between_boundary_and_survivor():
    b = _b(boundary_densify=True)
    seen, v = _run(b, failing={0.8, 1.6})
    dens = [t for t in seen if t.kind == DENSIFY]
    assert len(dens) == 1
    assert dens[0].scale == pytest.approx(math.sqrt(0.4 * 0.8))
    assert v['densified'] is True
    assert v['scale'] == pytest.approx(math.sqrt(0.4 * 0.8)), (
        'the inserted rung is now one rung below the boundary, so it is selectable')


def test_densify_never_densifies_the_interval_it_created():
    b = _b(boundary_densify=True)
    seen, _ = _run(b, failing={0.8, 1.6})
    assert len([t for t in seen if t.kind == DENSIFY]) == 1
    assert b.next_trial() is None


def test_a_densified_rung_that_fails_becomes_the_new_boundary():
    b = _b(boundary_densify=True)
    mid = math.sqrt(0.4 * 0.8)
    _, v = _run(b, failing={0.8, 1.6, mid})
    assert v['boundary_scale'] == pytest.approx(mid)
    assert v['scale'] == 0.4


def test_densify_off_by_default():
    b = _b()
    seen, _ = _run(b, failing={0.8, 1.6})
    assert not [t for t in seen if t.kind == DENSIFY]


# --------------------------------------------------------------- the horizon ---

def test_steps_to_failure_is_recorded_for_every_failing_candidate():
    b = _b(boundary_confirm_repeats=0)
    _run(b, failing={0.8, 1.6}, steps_to_failure=17)
    fails = [o for o in b.results() if not o.ok]
    assert fails and all(o.steps_to_failure == 17 for o in fails)
    assert all(o.steps_to_failure is None for o in b.results() if o.ok)


def test_a_late_failure_publishes_horizon_marginal():
    """The distribution is free -- it falls out of the bracket itself -- and it
    validates the horizon directly. NOT auto-extended: the owner changes the
    config, because an auto-extended horizon is a fitted quantity."""
    b = _b(boundary_confirm_repeats=0, trial_steps=150)
    _, v = _run(b, failing={1.6}, steps_to_failure=140)     # 93% of the horizon
    assert v['horizon_marginal'] is True
    assert v['scale'] == 0.8, 'it still selects; it just says the horizon is thin'

    b2 = _b(boundary_confirm_repeats=0, trial_steps=150)
    _, v2 = _run(b2, failing={1.6}, steps_to_failure=12)
    assert v2['horizon_marginal'] is False


def test_there_is_no_trial_steps_floor():
    """An earlier draft demanded trial_steps > 1/(1-beta2) = 1000. That is the
    warm-up time for Adam's moments FROM SCRATCH, and every trial restores a root
    whose moments are equilibrated -- changing the LR does not invalidate `v`,
    which estimates the gradient second moment rather than the step size."""
    b = _b(trial_steps=50)
    assert b.trial_steps == 50
    with pytest.raises(ValueError):
        _b(trial_steps=0)


# ---------------------------------------------------------------- the clock ---

def test_the_repeat_clock_counts_promoted_steps_only():
    """Charging discarded trial compute to the repeat clock would re-bracket
    sooner the more the run spent bracketing."""
    b = _b(repeat_every=10000)
    _run(b, failing={0.8, 1.6}, step=5000)
    b.promote(0.4, step=5150)
    assert not b.repeat_due(15149)
    assert b.repeat_due(15150)
    assert b.next_repeat_step() == 15150


def test_repeat_every_zero_means_once_per_stage():
    b = _b(repeat_every=0)
    b.promote(0.4, step=100)
    assert not b.repeat_due(10 ** 9)
    assert b.next_repeat_step() is None


# ----------------------------------------------------------------- fixed mode ---

def test_fixed_mode_bypasses_every_trial():
    b = LRBracket(mode='fixed', fixed_scale=0.2, burn_in_steps=500,
                  burn_in_scale=0.05)
    assert b.next_trial() is None
    b.promote(b.fixed_scale, step=500)
    assert b.phase == CRUISE and b.scale_now() == 0.2
    assert not b.repeat_due(10 ** 9), 'fixed mode never re-brackets'


def test_fixed_mode_still_burns_in():
    b = LRBracket(mode='fixed', fixed_scale=0.2, burn_in_steps=500,
                  burn_in_scale=0.05)
    assert b.phase == BURN_IN and b.scale_now() == 0.05
    assert not b.burn_in_complete(499) and b.burn_in_complete(500)


def test_fixed_mode_needs_an_explicit_scale():
    with pytest.raises(ValueError, match='needs a positive'):
        LRBracket(mode='fixed', fixed_scale=None)


def test_fixed_mode_does_not_have_to_satisfy_the_grid_rules():
    """A fixed-LR arm has no grid to validate, and demanding one would make the
    battery mode impossible to configure."""
    LRBracket(mode='fixed', fixed_scale=0.2, candidate_scales=())


# ------------------------------------------------------------------ refusal ---

def test_a_refusal_holds_the_burn_in_scale_and_says_so():
    b = _b()
    scale = b.refuse('root bias correction 0.31 below 0.9')
    assert scale == b.burn_in_scale
    assert b.phase == CRUISE
    assert 'bias correction' in b.refusal
    assert b.report()['lr_bracket/refused'] == 1.0


# ------------------------------------------------------------------- state ---

def test_a_resume_inside_a_bracket_re_arms_rather_than_half_restoring():
    """The candidate states live in host RAM and die with the process. Resuming
    into `bracket` would leave the machine expecting checkpoints it no longer
    holds, so the cycle restarts from the resumed state -- which is already
    mature -- and the driver says so."""
    b = _b()
    b.begin_bracket(1000, 0.99)
    state = b.state_dict()
    assert state['phase'] == 'bracket'
    fresh = _b()
    assert fresh.load_state_dict(state)
    assert fresh.phase == BURN_IN and fresh.resumed_mid_bracket is True


def test_stale_state_is_discarded_rather_than_reinterpreted():
    b = _b()
    assert not b.load_state_dict({'ver': 0, 'phase': 'cruise'})
    assert not b.load_state_dict(None)
    assert b.phase == BURN_IN


def test_the_report_carries_no_alpha_or_cosine():
    """Publishing either beside the selection would read as an explanation for a
    choice neither entered -- and alpha* is the statistic that was measured
    uncorrelated with the rate it steered."""
    b = _b()
    _run(b, failing={0.8, 1.6})
    keys = ' '.join(b.report())
    assert 'alpha' not in keys and 'cos' not in keys


def test_the_report_distinguishes_a_found_boundary_from_an_assumed_one():
    found = _b()
    _run(found, failing={0.8, 1.6})
    none = _b()
    _run(none, failing=())
    assert found.report()['lr_bracket/status'] != none.report()['lr_bracket/status']
    assert found.report()['lr_bracket/boundary_confirmed'] == 1.0
    assert none.report()['lr_bracket/boundary_confirmed'] == 0.0


# ------------------------------------------- slope-first selection ----------
# Owner 2026-08-25: stability is the CONSTRAINT (eligibility), the post-settle
# loss_drift is the OBJECTIVE. "Hottest survivor" was falsified twice in one
# day on var_conditioning (0.566 and 1.13 both survived their horizons and
# poisoned the run).

def _drift(v, se=0.05):
    return {'loss_drift': v, 'se': se, 't': (v / se if se else 0.0), 'n': 40}


def _run_with_drifts(b, drifts, failing=(), step=1000):
    """Like _run, attaching a loss_drift record per scale."""
    b.begin_bracket(step, bias_correction=0.99)
    while True:
        t = b.next_trial()
        if t is None:
            break
        fails = t.scale in failing
        b.record(t, ok=not fails, reason='loss_excursion' if fails else None,
                 steps_completed=10 if fails else b.trial_steps,
                 steps_to_failure=10 if fails else None,
                 loss_drift=drifts.get(t.scale))
    return b.select()


def test_selection_prefers_the_best_downward_slope():
    """Descending surface: hotter rungs descend faster, so slope-first
    recovers the old hottest-survivor answer exactly where it was correct."""
    b = _b(candidate_scales=(0.05, 0.2, 0.8), safety_rungs=0,
           boundary_confirm_repeats=0, boundary_densify=False)
    v = _run_with_drifts(b, {0.05: _drift(-1.0), 0.2: _drift(-3.0),
                             0.8: _drift(-5.0)})
    assert v['scale'] == 0.8
    assert v['selection_mode'] == 'loss_drift'
    assert v['loss_drift'] == -5.0


def test_selection_stays_cold_on_a_plateau():
    """No rung beats the coldest by 2 combined SEs: the no-signal case is a
    model pathology, not an LR question, and the answer is cold."""
    b = _b(candidate_scales=(0.05, 0.2, 0.8), safety_rungs=0,
           boundary_confirm_repeats=0, boundary_densify=False)
    v = _run_with_drifts(b, {0.05: _drift(0.0), 0.2: _drift(0.01),
                             0.8: _drift(-0.02)})
    assert v['scale'] == 0.05, (
        'a 0.02 drift edge inside a 0.14 noise bar promoted a hotter rung')
    assert v['selection_mode'] == 'loss_drift'


def test_selection_cannot_pick_an_upward_drifting_survivor():
    """The qm9c shape: the hottest rung survives its horizon while parking the
    loss above the root. Positive drift can never be argmin."""
    b = _b(candidate_scales=(0.05, 0.2, 0.8), safety_rungs=0,
           boundary_confirm_repeats=0, boundary_densify=False)
    v = _run_with_drifts(b, {0.05: _drift(-0.5), 0.2: _drift(-3.0),
                             0.8: _drift(+5.0)})
    assert v['scale'] == 0.2
    assert v['selection_mode'] == 'loss_drift'


def test_selection_falls_back_to_hottest_survivor_without_drift_data():
    """Short-horizon harnesses (and any misconfigured settle window) fit no
    drift; selecting on absent data would be worse than the legacy rule."""
    b = _b(candidate_scales=(0.05, 0.2, 0.8), safety_rungs=0,
           boundary_confirm_repeats=0, boundary_densify=False)
    v = _run_with_drifts(b, {})
    assert v['scale'] == 0.8
    assert v['selection_mode'] == 'survival_max'


def test_slope_selection_still_respects_the_detonation_ceiling():
    """Eligibility is unchanged: a detonated rung caps selection whatever the
    drifts say below it."""
    b = _b(candidate_scales=(0.05, 0.2, 0.8), safety_rungs=1,
           boundary_confirm_repeats=0, boundary_densify=False)
    v = _run_with_drifts(b, {0.05: _drift(-1.0), 0.2: _drift(-9.0),
                             0.8: _drift(-99.0)},
                         failing={0.8})
    # 0.8 detonated: its (absurdly good) drift is irrelevant because it is not
    # ELIGIBLE; slope selection runs over {0.05, 0.2} and picks 0.2
    assert v['scale'] == 0.2
    assert v['selection_mode'] == 'loss_drift'
