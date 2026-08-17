"""Tests for what the checks share: the context they are all resolved against,
and the report that renders them.

These cover the layer BELOW the individual checks, and every one of them exists
because the shared layer got something wrong that no per-check test could see --
a per-check test holds the context fixed, so a context that lies is invisible
from inside any one of them.

Run: python -m pytest analysis/tests -q
"""

import numpy as np
import pytest

from analysis import keys as K
from analysis.checks import (State, battery_labels, check_r11, context,
                             context_header, format_report, run_all,
                             run_label)
from analysis.tests import fixtures


# ---------------------------------------------------------------------------
# The route is never inferred from a stage the run did not reach
# ---------------------------------------------------------------------------

def test_an_unknown_stage_makes_the_route_unknown(mle_only):
    """`keys.detect_route` defaults a None stage index to the LAST DECLARED
    stage. That is right for a caller asking about the terminal stage and wrong
    here: it hands back a confident classification of a stage the run's own
    record does not say it reached.

    The damage is not a mislabelled row. NA_ROUTE marking is driven entirely by
    this route, so a route inferred from the wrong stage silently switches
    NA_ROUTE OFF."""
    blind = fixtures.mutate(mle_only)
    blind.summary.pop(K.STAGE_METRIC, None)
    blind.history.pop(K.STAGE_METRIC, None)

    ctx = context(blind)
    assert ctx.stage_index is None
    assert ctx.stage_name is None
    assert ctx.route is K.Route.UNKNOWN

    # the mutation half: the naive call still answers, and answers wrongly --
    # this is the behaviour `context` exists to not have
    assert K.detect_route(blind.config, None) is not K.Route.UNKNOWN


def test_a_known_stage_still_classifies(mle_only, vg_normal, tb_ramp):
    """The companion. Refusing to classify is only honest if it is not the
    answer every time -- a `context` that always said UNKNOWN would pass the
    test above while making the whole package useless."""
    assert context(mle_only).route is K.Route.MLE_PRIOR
    assert context(vg_normal).route is K.Route.VARGRAD_CONDITIONAL
    assert context(tb_ramp).route is K.Route.TB_UNCONDITIONAL


def test_a_single_stage_run_needs_no_phase_metric(tb_ramp):
    """With one declared stage there is nothing to be wrong about, so the
    refusal above must not fire there."""
    one = fixtures.mutate(tb_ramp)
    one.summary.pop(K.STAGE_METRIC, None)
    one.history.pop(K.STAGE_METRIC, None)
    for k in list(one.config):
        if k.startswith('protocol_stages_1'):
            one.config.pop(k)
    assert context(one).stage_index == 0
    assert context(one).route is not K.Route.UNKNOWN


def test_an_unknown_route_is_not_read_as_a_tb_run(mle_only):
    """The consequence, end to end. R11 is defined on the TB routes; asked about
    a run whose route could not be established, it must refuse rather than
    compute -- 'the question may not apply' is a hole, and a hole renders
    loudly."""
    blind = fixtures.mutate(mle_only)
    blind.summary.pop(K.STAGE_METRIC, None)
    blind.history.pop(K.STAGE_METRIC, None)
    res = check_r11(blind, ctx=context(blind))
    assert not res.ran
    assert 'not classified' in res.reason


# ---------------------------------------------------------------------------
# A finding can be attributed to an arm
# ---------------------------------------------------------------------------

def test_every_block_names_its_run_and_its_route(ring_probe, vg_normal):
    """A battery renders one block per (check, run). Without the run and the
    route on each, four arms produced twelve indistinguishable blocks -- and a
    reader could not tell which route's NA rules had been applied to any of
    them."""
    results = run_all([ring_probe, vg_normal], window=2000)
    per_run = [r for r in results if r.check != results[0].check]
    assert per_run
    for res in per_run:
        assert res.run, f'{res.check} does not name its run'
        assert 'route=' in res.header and 'window=' in res.header
    labels = {res.run for res in per_run}
    assert labels == {run_label(ring_probe), run_label(vg_normal)}


def test_runs_sharing_a_name_are_still_told_apart(ring_probe, ring_cal):
    a, b = fixtures.mutate(ring_probe), fixtures.mutate(ring_cal)
    a.name = b.name = 'mk_dev'
    labels = {res.run for res in run_all([a, b]) if res.run and ',' not in res.run}
    assert len(labels) == 2, labels


def test_the_window_reaches_the_battery_check(vg_blowup):
    """§4's stage-boundary subject asks whether the read window straddles a
    transition, which is unanswerable without one. Left unforwarded by
    `run_all`, that subject could never fire on the default path."""
    windowed = run_all(vg_blowup, window=2000)[0]
    unwindowed = run_all(vg_blowup)[0]
    boundary = [r for r in windowed.rows if r.subject.endswith('/stage_boundary')]
    assert boundary and 'no window given' not in boundary[0].detail
    unbounded = [r for r in unwindowed.rows if r.subject.endswith('/stage_boundary')]
    assert unbounded and 'no window given' in unbounded[0].detail


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_a_check_that_did_not_run_is_hoisted_above_the_findings(mle_only, tb_ramp):
    """A check that did not run is a bigger hole than anything a check that did
    run can say, and a hole placed after the findings reads as a footnote to a
    complete picture."""
    # R11's series dropped from a real TB run: in scope, no data -- which is a
    # refusal. `mle_only` is NOT a not-run case any more: it is NA_ROUTE, which
    # is an answer.
    #
    # ALL FOUR series, not just the scatter pair. R11 reads two sensors and
    # refuses only when NEITHER can be read -- dropping the scatter pair alone
    # leaves the derived-bar sensor, and the check correctly still runs.
    holed = fixtures.mutate(tb_ramp, drop=(
        K.R11_NUMERATOR, K.R11_DENOMINATOR,
        K.R11_MEMORISATION_NUMERATOR, K.R11_MEMORISATION_DENOMINATOR))
    results = run_all([mle_only, holed], window=4000)
    assert any(not r.ran for r in results), 'expected a not-run check'
    text = format_report(results)
    assert text.index('DID NOT RUN') < text.index('finding(s)')
    assert 'not a pass' in text


def test_no_check_returns_a_silent_pass_on_any_real_run(all_runs):
    """`ran=True` with zero rows renders identically to 'looked, found nothing'
    and means 'never looked'. It must not happen on real data."""
    for name, run in all_runs.items():
        for res in run_all(run, window=4000):
            assert res.ran is False or res.rows, f'{name}: {res.check} ran and said nothing'


def test_every_real_run_survives_every_check(all_runs):
    for name, run in all_runs.items():
        try:
            format_report(run_all(run, window=4000), verbose=True)
        except Exception as e:                       # pragma: no cover
            pytest.fail(f'{name}: {type(e).__name__}: {e}')


def test_context_header_states_an_unknown_stage_as_unknown(mle_only):
    blind = fixtures.mutate(mle_only)
    blind.summary.pop(K.STAGE_METRIC, None)
    blind.history.pop(K.STAGE_METRIC, None)
    header = context_header(blind, context(blind), 1000)
    assert 'UNKNOWN' in header and 'route=unknown' in header


def test_na_route_is_visible_without_verbose(vg_normal, tb_ramp):
    """Spec H2 / acceptance 4, at the REPORT level rather than the state level.

    NA_ROUTE is not a finding -- nothing is wrong -- but it is not silence
    either. Withheld at default verbosity, a conditional VarGrad run's R11
    rendered BYTE-IDENTICALLY to a clean TB run, which is the collapse H2
    forbids, reached through the renderer instead of through the check."""
    from analysis.checks import check_r11, format_result
    na = format_result(check_r11(vg_normal))
    live = format_result(check_r11(tb_ramp))
    assert na != live
    assert 'NA_ROUTE' in na and 'NA_ROUTE' not in live
    assert 'nothing to report' not in na


# ---------------------------------------------------------------------------
# Runs are named by something that MEANS something
# ---------------------------------------------------------------------------

def test_colliding_names_are_split_by_the_knob_that_differs(ring_probe, ring_cal):
    """`reading_runs.md` §7: refer to runs by NAME, TAG, or A DISTINGUISHING
    CONFIG FEATURE -- never by wandb id. An id is a hash; it carries nothing and
    makes the reader look every arm up.

    Two arms of a real cluster battery are both `prod0810_mipcas_elj`, so a
    collision is the normal case, not the edge case."""
    a, b = fixtures.mutate(ring_probe), fixtures.mutate(ring_cal)
    a.name = b.name = 'prod_elj'
    a.config[K.CFG_EVAL_T] = {'value': 10}
    b.config[K.CFG_EVAL_T] = {'value': 40}

    labels = battery_labels([a, b])
    assert set(labels.values()) == {'prod_elj[eval_T=10]', 'prod_elj[eval_T=40]'}
    for lab in labels.values():
        assert a.run_id not in lab and b.run_id not in lab


def test_a_unique_name_is_left_alone(ring_probe, vg_normal):
    labels = battery_labels([ring_probe, vg_normal])
    assert set(labels.values()) == {ring_probe.name, vg_normal.name}


def test_a_label_never_carries_a_config_repr_blob(ring_probe, ring_cal):
    """wandb stores each config section a SECOND time as a repr string
    (`adaptive_lr` -> "Namespace(warmup_steps=1000, ...)"). Those keys are the
    shortest and their values the longest, so ranking candidates on key length
    put a 200-character Namespace dump into every subject line."""
    a, b = fixtures.mutate(ring_probe), fixtures.mutate(ring_cal)
    a.name = b.name = 'prod_elj'
    a.config['blob'] = {'value': 'Namespace(' + 'x' * 200 + ')'}
    b.config['blob'] = {'value': 'Namespace(' + 'y' * 200 + ')'}
    a.config[K.CFG_EVAL_T] = {'value': 10}
    b.config[K.CFG_EVAL_T] = {'value': 40}
    for lab in battery_labels([a, b]).values():
        assert 'Namespace' not in lab
        assert len(lab) < 40, lab


def test_arms_that_differ_in_no_knob_fall_back_to_something_visible(ring_probe):
    """Two arms with the same name and the same config still need separating,
    and saying so with the id is the honest answer there: they differ in no
    knob."""
    a, b = fixtures.mutate(ring_probe), fixtures.mutate(ring_probe)
    a.name = b.name = 'twin'
    b.run_id = 'other'
    labels = set(battery_labels([a, b]).values())
    assert len(labels) == 2
