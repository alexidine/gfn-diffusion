"""
Tests for R2 -- confirm the thing ever fired.

MUTATION-TESTED THROUGHOUT. Every state the check can report gets two tests: a
REAL run that does not produce it for that subject, and a real run with the
condition re-introduced that does. A check that has never fired has not been
tested, and this repo has shipped tests that passed while blind more than once.

Mutations are edits to REAL captured runs (`fixtures.mutate`/`.pin` deep-copy,
so the module-scoped fixtures survive), never hand-built configs -- a hand-built
config agrees with whatever the author assumed.

NO METRIC OR CONFIG NAME IS SPELLED OUT HERE. Every key a test touches is taken
from `K.MECHANISMS` or a `K.CFG_*` template, so a rename upstream breaks these
tests loudly instead of leaving them testing nothing.

Real-run facts this file pins, because they are the evidence R2 was built
against and not accidents of the fixture set:

  * `tb_ramp` fired ray calibration 26 times; `vg_normal` declares the same
    global block and never calibrated. The FIRED/INERT pair, on real data.
  * `tb_ramp`'s buffer servo is live and its ACTUATOR never moved --
    `protocol/bs_log_boost` is 0 for every tick while `protocol/bs_boost`, which
    is exp() of it, reads 1.0. Registering the exp() form would have reported
    this servo as fired.
  * `tb_ramp`'s `Fwd Frac` is pinned at 0.05 BY DECLARATION. Pinned-by-
    declaration is not inert and must not produce a finding.
  * `mle_only`'s stage-0 exit conditions are the real gate cases: `bwd/tbc`
    fired, `gates/mle_flat` and `eval/wass_debiased` are gates whose metric the
    run never logged under that name.

Run: python -m pytest analysis/tests -q
"""

import re

import numpy as np
import pytest

from analysis import keys as K
from analysis.checks import State, check_r2
from analysis.tests import fixtures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mech(name):
    """One registry entry, or a loud failure. Every key these tests touch comes
    through here rather than being typed."""
    found = {m.name: m for m in K.MECHANISMS}
    assert name in found, f'{name!r} left K.MECHANISMS: {sorted(found)}'
    return found[name]


def _row(res, subject):
    """The single row for a subject, or a loud failure.

    Not a filter that shrugs on an empty list: a check that quietly stopped
    emitting a subject would turn every assertion about it into a vacuous pass.
    """
    hits = [r for r in res.rows if r.subject == subject]
    assert len(hits) == 1, (
        f'{subject!r}: expected exactly one row, got {len(hits)}. '
        f'rows={[r.subject for r in res.rows]}')
    return hits[0]


def _state(res, subject):
    return _row(res, subject).state


def _copy(run):
    """A deep copy with nothing changed -- the base for edits `mutate` cannot
    express, namely REMOVING a config key."""
    return fixtures.mutate(run)


def _stage_key(run, tail):
    return K.CFG_STAGE % (K.current_stage_index(run.summary, run.config), tail)


def _declared(run, mech):
    return K._value(run.config, _stage_key(run, mech.declared_by))


def _exit_subjects(res):
    return [r.subject for r in res.rows if r.subject.startswith('exit[')]


def _coeff_config_keys(run, mode, name):
    """Every config key feeding `effective_loss_coeffs[mode][name]`, found in
    the config rather than spelled out. Both the base block and the stage
    override have to be hit or the effective value does not move."""
    idx = K.current_stage_index(run.summary, run.config)
    stage_prefix = K.CFG_STAGE % (idx, '')
    keys = [k for k in run.config
            if k.endswith('_' + name) and mode in k
            and (k.startswith(mode) or k.startswith(stage_prefix))]
    assert keys, f'no config key resolves to {mode}/{name}'
    return keys


def _set_coeff_config(run, mode, name, value):
    return fixtures.mutate(
        run, config={k: value for k in _coeff_config_keys(run, mode, name)})


# ---------------------------------------------------------------------------
# The registry -- FIRED / INERT, on real data
# ---------------------------------------------------------------------------

def test_ray_calibration_fired_on_tb_ramp(tb_ramp):
    """The FIRED half of the pair. 26 calibrations, so the row is not a
    finding."""
    row = _row(check_r2(tb_ramp), 'ray_calibration')
    assert row.state is State.FIRED
    assert not row.is_finding
    assert row.numbers['last'] == 26.0
    assert row.numbers['n_ticks'] > 0


def test_ray_calibration_inert_on_vg_normal(vg_normal):
    """The INERT half, on a run that declares the identical global block.

    `lr_ctrl/calibrations` sits at 0 for the whole run and no `raycal/*` series
    exists at all -- the probe was enabled and never ran."""
    row = _row(check_r2(vg_normal), 'ray_calibration')
    assert row.state is State.INERT
    assert row.is_finding
    assert row.numbers['n_steps_active'] == 0
    assert row.numbers['n_ticks'] > 0
    assert not [k for k in vg_normal.available_keys() if k.startswith('raycal')]


def test_buffer_servo_inert_while_its_exp_sibling_reads_one(tb_ramp):
    """A servo that is live and whose ACTUATOR never moved.

    The second assertion is the reason the registry names `bs_log_boost`: the
    exp() sibling is 1.0 at rest, so a NONZERO rule on it reports this servo as
    fired on every tick. A trace that is nonzero at rest is not a trace."""
    mech = _mech('buffer_servo')
    trace = mech.trace[0]
    row = _row(check_r2(tb_ramp), mech.name)
    assert row.state is State.INERT
    assert row.numbers['n_steps_active'] == 0

    exp_sibling = trace.replace('_log_', '_')      # bs_log_boost -> bs_boost
    _, boost = tb_ramp.history[exp_sibling]
    assert np.all(boost == 1.0), 'the exp() sibling is no longer pinned at rest'
    assert np.count_nonzero(boost) == len(boost), (
        'the exp() sibling would have counted as active on every tick')


def test_fwd_frac_pinned_by_declaration_is_not_a_finding(tb_ramp):
    """R2's most dangerous false positive: a frac that never moves because the
    stage DECLARED it there. Pinned-by-declaration is not inert."""
    mech = _mech('frac.fwd')
    _, frac = tb_ramp.history[mech.trace[0]]
    assert frac.min() == frac.max(), 'fixture no longer has a pinned Fwd Frac'
    assert float(_declared(tb_ramp, mech)) == pytest.approx(float(frac[-1]))

    row = _row(check_r2(tb_ramp), mech.name)
    assert row.state is State.FIRED
    assert not row.is_finding
    assert row.numbers['n_steps_active'] == row.numbers['n_ticks']


def test_frac_below_its_deactivation_threshold_is_inert(tb_ramp):
    """The mutation for the row above, and R2's canonical case: the branch is
    configured on and the controller has driven it under the stage's
    deactivate_threshold, so it contributes nothing while the config still
    claims it does."""
    mech = _mech('frac.fwd')
    floor = float(K._value(tb_ramp.config, _stage_key(tb_ramp, mech.threshold_key)))
    assert floor > 0, 'the fixture no longer carries a deactivation floor'

    hurt = fixtures.pin(tb_ramp, mech.trace[0], floor / 2.0)
    row = _row(check_r2(hurt), mech.name)
    assert row.state is State.INERT
    assert row.is_finding
    assert row.numbers['n_steps_active'] == 0
    assert row.numbers['floor'] == floor
    # the declaration is untouched -- this is 'declared on, running off'
    assert float(_declared(hurt, mech)) > floor


# ---------------------------------------------------------------------------
# NO_TRACE
# ---------------------------------------------------------------------------

def test_replay_prioritise_has_no_trace_on_mle_only(mle_only):
    """Real NO_TRACE: prioritised draw is enabled globally and the stage the run
    is in never runs a replay branch, so the eligibility series was never
    logged. Not inert -- nothing was measured."""
    mech = _mech('replay_prioritise')
    row = _row(check_r2(mle_only), mech.name)
    assert row.state is State.NO_TRACE
    assert row.is_finding
    assert not any(t in mle_only.available_keys() for t in mech.trace)


def test_dropping_the_trace_turns_fired_into_no_trace(tb_ramp):
    """Mutation of `test_ray_calibration_fired_on_tb_ramp`: the same declared
    mechanism with its trace unlogged is NO_TRACE, never INERT -- 'the counter
    says zero' and 'there is no counter' are different findings."""
    mech = _mech('ray_calibration')
    blind = fixtures.mutate(tb_ramp, drop=mech.trace)
    row = _row(check_r2(blind), mech.name)
    assert row.state is State.NO_TRACE
    assert row.numbers['n_ticks'] == 0


# ---------------------------------------------------------------------------
# UNDECLARED_ACTIVE, and the absent-key rule that keeps it honest
# ---------------------------------------------------------------------------

def test_frac_declared_off_while_the_branch_draws_is_undeclared_active(tb_ramp):
    """§4's 'an arm that was not running the code it was written to test'.

    Modelled on a real local run whose stage sets `fracs.replay: 0` while
    `Replay Frac` sits at ~0.25 for every tick."""
    mech = _mech('frac.replay')
    assert _state(check_r2(tb_ramp), mech.name) is State.FIRED

    lying = fixtures.mutate(tb_ramp,
                            config={_stage_key(tb_ramp, mech.declared_by): 0})
    row = _row(check_r2(lying), mech.name)
    assert row.state is State.UNDECLARED_ACTIVE
    assert row.is_finding
    assert row.numbers['n_steps_active'] > 0


def test_absent_declaring_key_is_not_undeclared_active(tb_ramp, vg_normal):
    """The rule that stops R2 crying wolf.

    Stage-scoped keys live on the stage that uses them, and the traces span the
    whole run: `flags_mle_gate` is on stage 0 and both fixtures are reading
    stage 1, where the streak counter still carries stage 0's activity. An
    absent key makes no claim, so there is nothing for the trace to contradict.
    Treating absent as 'the config says off' manufactured this finding on most
    of the local corpus."""
    for run, name in ((vg_normal, 'mle_gate'), (tb_ramp, 'lr_sensor.ray')):
        mech = _mech(name)
        key = _stage_key(run, mech.declared_by)
        assert key not in run.config, f'{key} is now in the config'
        # the trace is live and moving -- this is not a vacuous pass
        assert any(t in run.available_keys() for t in mech.trace)
        assert _state(check_r2(run), mech.name) is State.OFF


def test_present_but_off_declaring_key_still_fires(vg_normal):
    """Mutation of the test above: the suppression is about PRESENCE, not about
    the mechanism. Write the key in as False and the contradiction is real
    again."""
    mech = _mech('mle_gate')
    key = _stage_key(vg_normal, mech.declared_by)
    lying = fixtures.mutate(vg_normal, config={key: False})
    row = _row(check_r2(lying), mech.name)
    assert row.state is State.UNDECLARED_ACTIVE
    assert row.numbers['n_steps_active'] > 0


def test_no_fixture_produces_an_undeclared_active_row(all_runs):
    """Corpus guard. Eight real runs, none of which is misconfigured, so any
    UNDECLARED_ACTIVE here is manufactured -- the failure mode that gets a check
    switched off."""
    for name, run in all_runs.items():
        res = check_r2(run)
        bad = [r.subject for r in res.rows
               if r.state is State.UNDECLARED_ACTIVE]
        assert not bad, f'{name}: manufactured undeclared-active on {bad}'


# ---------------------------------------------------------------------------
# UNREADABLE -- a trace that cannot answer
# ---------------------------------------------------------------------------

def test_one_point_moves_trace_is_unreadable_not_inert(tb_ramp):
    """`base.series` falls back to the summary, and a MOVES rule on a single
    point is zero by construction -- which renders as INERT and reads as a dead
    controller. Real runs hit this on their first eval tick."""
    mech = _mech('balance.ratio')
    assert _state(check_r2(tb_ramp), mech.name) is State.FIRED

    stub = {t: (tb_ramp.history[t][0][-1:], tb_ramp.history[t][1][-1:])
            for t in mech.trace if t in tb_ramp.history}
    assert stub, 'the balance traces left the fixture'
    thin = fixtures.mutate(tb_ramp, history=stub)
    row = _row(check_r2(thin), mech.name)
    assert row.state is State.UNREADABLE
    assert row.is_finding
    assert row.numbers['n_ticks'] == 1


def test_config_only_fixture_reports_unreadable_not_inert(ring_probe):
    """The same shape on real data: this fixture is captured for its config and
    carries a token history, so its balance traces come from the summary."""
    assert _state(check_r2(ring_probe), 'balance.proportional') is State.UNREADABLE


# ---------------------------------------------------------------------------
# A counter's LEVEL is the evidence
# ---------------------------------------------------------------------------

def test_counter_still_fired_under_a_window_that_holds_no_rise(tb_ramp):
    """R2 asks whether the thing EVER fired. `count_active` counts rises, which
    are zero in any window opening after the last event -- so a windowed read
    would report 26 completed calibrations as inert."""
    mech = _mech('ray_calibration')
    s, v = tb_ramp.history[mech.trace[0]]
    window = 200.0
    inside = v[s >= s[-1] - window]
    assert np.all(np.diff(inside) <= 0), 'the fixture now rises inside the window'

    row = _row(check_r2(tb_ramp, window=window), mech.name)
    assert row.state is State.FIRED
    assert row.numbers['n_steps_active'] == 0     # no rise in the window
    assert row.numbers['last'] > 0                # the level is the evidence


def test_a_counter_that_never_left_zero_is_inert_under_the_same_window(tb_ramp):
    """Mutation of the above -- the level rule must not make every counter
    fire."""
    mech = _mech('ray_calibration')
    dead = fixtures.pin(tb_ramp, mech.trace[0], 0.0)
    assert _state(check_r2(dead, window=200.0), mech.name) is State.INERT


# ---------------------------------------------------------------------------
# NA_ROUTE -- never inert, never zero
# ---------------------------------------------------------------------------

def test_na_route_trace_is_not_reported_as_inert(vg_normal, monkeypatch):
    """No registry trace is currently marked NA on any route, so the branch is
    exercised by adding one to the taxonomy for the length of this test. The
    property under test is the ORDER -- NA_ROUTE is decided before liveness, so
    a series that does not track on this route can never be read as inert."""
    mech = _mech('ray_calibration')
    assert _state(check_r2(vg_normal), mech.name) is State.INERT

    patched = dict(K._NA_PATTERNS)
    patched[K.Route.VARGRAD_CONDITIONAL] = (
        patched.get(K.Route.VARGRAD_CONDITIONAL, ())
        + (re.escape(mech.trace[0]) + '$',))
    monkeypatch.setattr(K, '_NA_PATTERNS', patched)

    row = _row(check_r2(vg_normal), mech.name)
    assert row.state is State.NA_ROUTE
    assert row.numbers['n_steps_active'] == 0     # reported, never as activity
    assert K.Route.VARGRAD_CONDITIONAL.value in row.detail


# ---------------------------------------------------------------------------
# Family 2 -- the loss coefficients the trainer is actually holding
# ---------------------------------------------------------------------------

def _coeff(run, mode, name):
    return K.LOSS_COEFF_TRACE % (mode, name)


def test_live_loss_coefficients_agree_with_the_config(mle_only):
    """The baseline the three mutations below are measured against: on a real
    run every coefficient the stage declares is being held at its configured
    value, so the family contributes no findings."""
    res = check_r2(mle_only)
    rows = [r for r in res.rows if r.subject.startswith('loss_coeffs')]
    assert rows, 'the loss-coefficient family emitted nothing'
    assert not [r for r in rows if r.is_finding]
    assert _state(res, _coeff(mle_only, 'bwd', 'mle')) is State.FIRED


def test_loss_coefficient_read_from_the_summary_not_history(mle_only):
    """`K.LOSS_COEFF_IS_SUMMARY_ONLY`: a change-only channel, emitted at eval
    time and only when a stage transition moved it, so the series is 1-2 points
    and the local datastore reader drops anything shorter than three. Read from
    history alone the whole family is invisible."""
    assert K.LOSS_COEFF_IS_SUMMARY_ONLY
    key = _coeff(mle_only, 'bwd', 'mle')
    assert key not in mle_only.history
    assert key in mle_only.summary


def test_coefficient_configured_on_and_held_at_zero_is_inert(mle_only):
    """'A knob retired upstream' -- the config asks for the term and the
    optimiser never saw it."""
    key = _coeff(mle_only, 'bwd', 'mle')
    hurt = fixtures.mutate(mle_only, summary={key: 0.0})
    row = _row(check_r2(hurt), key)
    assert row.state is State.INERT
    assert row.numbers['config'] > 0
    assert row.numbers['live'] == 0.0


def test_coefficient_disagreement_flags_in_both_directions(mle_only):
    """Either direction is a disagreement worth a number: config above live is a
    term that was turned down somewhere the reader did not look, live above
    config is one that was turned up."""
    on = _coeff(mle_only, 'bwd', 'mle')
    off = _coeff(mle_only, 'bwd', 'tb')
    assert _state(check_r2(mle_only), off) is State.OFF

    down = fixtures.mutate(mle_only, summary={on: 0.5})
    row = _row(check_r2(down), on)
    assert row.state is State.FLAG
    assert row.numbers['config'] == 1.0 and row.numbers['live'] == 0.5

    up = fixtures.mutate(mle_only, summary={off: 0.7})
    row = _row(check_r2(up), off)
    assert row.state is State.FLAG
    assert row.numbers['config'] == 0.0 and row.numbers['live'] == 0.7


def test_coefficient_with_no_live_trace_is_no_trace(mle_only):
    """Declared above zero and never logged: what the trainer held is unknown,
    which is not the same as zero."""
    key = _coeff(mle_only, 'bwd', 'mle')
    blind = fixtures.mutate(mle_only, drop=(key,))
    row = _row(check_r2(blind), key)
    assert row.state is State.NO_TRACE
    assert row.numbers['n_ticks'] == 0


def test_non_numeric_coefficient_config_is_unreadable(mle_only):
    """Comparing an unparseable config value as zero would invent an
    agreement."""
    key = _coeff(mle_only, 'bwd', 'mle')
    assert _state(check_r2(mle_only), key) is State.FIRED
    idx = K.current_stage_index(mle_only.summary, mle_only.config)
    junk = _set_coeff_config(mle_only, 'bwd', 'mle', 'off')
    assert K.effective_loss_coeffs(junk.config, idx)['bwd']['mle'] == 'off'
    assert _state(check_r2(junk), key) is State.UNREADABLE


def test_mode_filter_keeps_the_base_replay_tb_out_of_an_mle_stage(mle_only,
                                                                 tb_ramp):
    """The filter that stops the MLE warm-start being read as the TB route.

    The canonical base sets `replay_loss_coeffs_tb: 1.0`; the stage is
    `train_mode: bwd` and never evaluates a replay branch, so that 1.0 is not a
    live coefficient. `tb_ramp` is the fused counter-case, where it is."""
    idx = K.current_stage_index(mle_only.summary, mle_only.config)
    assert K.effective_loss_coeffs(mle_only.config, idx)['replay']['tb'] > 0
    assert K.active_modes(mle_only.config, idx) == ('bwd',)

    subjects = {r.subject for r in check_r2(mle_only).rows}
    assert _coeff(mle_only, 'replay', 'tb') not in subjects
    assert _coeff(mle_only, 'bwd', 'tb') in subjects

    fused = {r.subject for r in check_r2(tb_ramp).rows}
    assert _coeff(tb_ramp, 'replay', 'tb') in fused


# ---------------------------------------------------------------------------
# Family 3 -- the current stage's exit conditions
# ---------------------------------------------------------------------------

def _exit_metric(run, j):
    idx = K.current_stage_index(run.summary, run.config)
    return K._value(run.config, K.CFG_STAGE_EXIT_METRIC % (idx, j))


def _streak(run, j):
    return K.EXIT_STREAK_TRACE % K.metric_tag(_exit_metric(run, j))


def test_exit_condition_that_held_is_fired(mle_only):
    """Real FIRED: the `bwd/tbc` streak is above zero, so the condition has held
    on at least one evaluation."""
    subject = f'exit[2] {_exit_metric(mle_only, 2)}'
    row = _row(check_r2(mle_only), subject)
    assert row.state is State.FIRED
    assert row.numbers['n_steps_active'] > 0


def test_exit_condition_whose_streak_never_reached_one_is_inert(mle_only):
    """Real INERT, on the same run: the mle-flat gate's streak is zero for every
    tick -- the condition never held once, so the gate was never armed."""
    subject = f'exit[0] {_exit_metric(mle_only, 0)}'
    row = _row(check_r2(mle_only), subject)
    assert row.state is State.INERT
    assert row.is_finding
    assert row.numbers['n_steps_active'] == 0


def test_pinning_a_live_streak_to_zero_turns_fired_into_inert(mle_only):
    """Mutation of the FIRED case, so INERT is proved to be driven by the streak
    and not by which exit index it is."""
    subject = f'exit[2] {_exit_metric(mle_only, 2)}'
    dead = fixtures.pin(mle_only, _streak(mle_only, 2), 0.0)
    assert _state(check_r2(dead), subject) is State.INERT


def test_missing_streak_is_no_trace_not_inert(mle_only):
    """Whether the condition ever held cannot be read when the protocol never
    published the streak."""
    subject = f'exit[2] {_exit_metric(mle_only, 2)}'
    blind = fixtures.mutate(mle_only, drop=(_streak(mle_only, 2),))
    row = _row(check_r2(blind), subject)
    assert row.state is State.NO_TRACE
    assert row.numbers['n_ticks'] == 0


def test_gate_metric_not_in_the_run_record_is_its_own_finding(mle_only):
    """A gate whose metric is not logged. Two real cases, reported as resolution
    actually answers them rather than special-cased away:

      * `gates/mle_flat` is published to the protocol and never logged as a
        metric -- only its streak exists.
      * `eval/wass_debiased` is logged BARE, as `wass_debiased`, and `K.resolve`
        does not bridge that direction, so it reports ABSENT.

    The row is separate from the streak row because they are independent facts
    and a merged row lets either hide the other."""
    res = check_r2(mle_only)
    for j in (0, 1):
        metric = _exit_metric(mle_only, j)
        row = _row(res, f'exit[{j}].metric {metric}')
        assert row.state is State.NO_TRACE
        assert row.is_finding
        assert metric not in mle_only.available_keys()
    # the bare form IS in the record -- the finding is about resolution, not
    # about the run failing to measure anything
    assert 'wass_debiased' in {k.rsplit('/', 1)[-1]
                               for k in mle_only.available_keys()}


def test_gate_metric_that_resolves_is_reported_ok(mle_only):
    """The companion: the third condition's metric is logged under exactly the
    name the config names, and its row is not a finding."""
    metric = _exit_metric(mle_only, 2)
    row = _row(check_r2(mle_only), f'exit[2].metric {metric}')
    assert row.state is State.OK
    assert not row.is_finding
    assert row.numbers['resolved_to'] == metric


def test_dropping_a_resolving_gate_metric_makes_it_no_trace(mle_only):
    """Mutation of the row above. The streak row stays FIRED, which is the point
    of splitting them: the gate did trip, and its metric is nonetheless
    unreadable from the run record."""
    metric = _exit_metric(mle_only, 2)
    blind = fixtures.mutate(mle_only, drop=(metric,))
    res = check_r2(blind)
    assert _state(res, f'exit[2].metric {metric}') is State.NO_TRACE
    assert _state(res, f'exit[2] {metric}') is State.FIRED


def test_gate_on_a_metric_that_does_not_track_is_na_route(vg_normal, mle_only):
    """A gate defined on a series that exists and does not mean what it looks
    like on this route. Both rows say NA_ROUTE; neither says inert and neither
    prints a zero as if it were a reading."""
    assert not [r for r in check_r2(mle_only).rows if r.state is State.NA_ROUTE]

    idx = K.current_stage_index(vg_normal.summary, vg_normal.config)
    assert not _exit_subjects(check_r2(vg_normal)), 'the stage grew exits'
    na_metric = 'fwd/tb_err_worst'
    assert na_metric in K.TOPLINE_TB                    # not typed, cross-checked
    gated = fixtures.mutate(
        vg_normal, config={K.CFG_STAGE_EXIT_METRIC % (idx, 0): na_metric})

    res = check_r2(gated)
    assert _state(res, f'exit[0] {na_metric}') is State.NA_ROUTE
    assert _state(res, f'exit[0].metric {na_metric}') is State.NA_ROUTE


def test_exit_conditions_stop_at_the_first_absent_index(mle_only):
    """Three conditions on the stage, two rows each, and nothing past them."""
    subjects = _exit_subjects(check_r2(mle_only))
    assert len(subjects) == 6
    assert not any(s.startswith('exit[3]') for s in subjects)


# ---------------------------------------------------------------------------
# Entry point -- what happens when the check cannot run
# ---------------------------------------------------------------------------

def test_no_config_does_not_run(mle_only):
    """`not_run`, never an empty result: 'no findings' and 'never looked' render
    the same and mean opposite things."""
    empty = _copy(mle_only)
    empty.config.clear()
    res = check_r2(empty)
    assert res.ran is False
    assert res.reason
    assert not res.rows


def test_unknown_stage_is_loud_and_still_reads_the_global_mechanisms(mle_only):
    """A run whose stage cannot be read loses every stage-scoped subject. That
    is a hole in the report and it is stated as one -- while the global
    mechanisms, which do not need a stage, are still answered."""
    assert not [r for r in check_r2(mle_only).rows if r.subject == 'stage']

    lost = fixtures.mutate(mle_only, drop=(K.STAGE_METRIC,))
    res = check_r2(lost)
    row = _row(res, 'stage')
    assert row.state is State.UNREADABLE
    assert row.is_finding
    assert row.numbers['n_skipped'] > 0

    subjects = {r.subject for r in res.rows}
    stage_scoped = {m.name for m in K.MECHANISMS if m.scope == 'stage'}
    assert not (subjects & stage_scoped)
    assert {m.name for m in K.MECHANISMS if m.scope != 'stage'} <= subjects
    assert not [s for s in subjects if s.startswith('loss_coeffs')]
    assert not _exit_subjects(res)


# ---------------------------------------------------------------------------
# Shape invariants over the whole corpus
# ---------------------------------------------------------------------------

def test_every_mechanism_gets_exactly_one_row_on_every_run(all_runs):
    for name, run in all_runs.items():
        res = check_r2(run)
        assert res.ran, name
        for mech in K.MECHANISMS:
            _row(res, mech.name)      # raises with the subject list on a dup


def test_every_registry_row_carries_the_triple(all_runs):
    """mechanism -> fired? -> n_steps_active. A finding without its inputs is an
    assertion, and this package does not make assertions."""
    names = {m.name for m in K.MECHANISMS}
    for run_name, run in all_runs.items():
        for row in check_r2(run).rows:
            if row.subject not in names:
                continue
            assert 'n_steps_active' in row.numbers, f'{run_name}/{row.subject}'
            assert 'n_ticks' in row.numbers, f'{run_name}/{row.subject}'
            assert row.state in (State.FIRED, State.INERT, State.NO_TRACE,
                                 State.OFF, State.UNDECLARED_ACTIVE,
                                 State.NA_ROUTE, State.UNREADABLE)


def test_a_window_never_silently_changes_the_meaning_of_a_row(tb_ramp):
    """Every row names the window it was measured over, so a windowed INERT
    ('not active lately') is never read as an unwindowed one ('never')."""
    windowed = check_r2(tb_ramp, window=500.0)
    rows = [r for r in windowed.rows if 'window' in r.numbers]
    assert rows
    assert {r.numbers['window'] for r in rows} == {'trailing 500 steps'}
    assert {r.numbers['window'] for r in check_r2(tb_ramp).rows
            if 'window' in r.numbers} == {'all'}


def test_a_windowed_exit_inert_does_not_claim_the_whole_run(mle_only):
    """A windowed INERT means "not active lately"; only an unwindowed one means
    "never". This row used to append "the condition never held once" -- a claim
    about the whole run -- to rows whose own `window` field contradicted it, on
    a real run whose streak had reached 4 outside the window."""
    unwindowed = [r for r in check_r2(mle_only).rows
                  if r.subject.startswith('exit[') and '.metric' not in r.subject]
    assert unwindowed, 'fixture declares no exit condition'

    windowed = [r for r in check_r2(mle_only, window=200.0).rows
                if r.subject.startswith('exit[') and '.metric' not in r.subject]
    for row in windowed:
        if row.state is State.INERT:
            assert 'in this window' in row.detail
            assert 'never held once' not in row.detail
    for row in unwindowed:
        if row.state is State.INERT:
            assert 'whole run' in row.detail
