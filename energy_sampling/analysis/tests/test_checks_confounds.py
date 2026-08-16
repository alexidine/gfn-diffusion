"""
Tests for the §4 confound check.

MUTATION-TESTED THROUGHOUT. Every condition gets two tests: the real run does
NOT fire for that subject, and a real run with the condition re-introduced
DOES. A check that has never fired has not been tested, and this repo has
shipped tests that passed while blind more than once.

The mutations are edits to REAL captured runs (`fixtures.mutate` deep-copies,
so the module-scoped fixtures survive), never hand-built configs. A hand-built
config agrees with whatever the author assumed, which is the failure the whole
package exists to stop.

Real-run findings this file pins, because they are evidence and not accidents:

  * `ring_probe` and `ring_cal` are the SAME ARM written twice. Their configs
    (`configs/local_aug02/ring_probe{,_cal}.yaml`) differ in `run_name` and in
    nothing else, so the duplicates subject fires on real data.
  * `buildout` has `checkpoint_name: None` while `tb_resumed` carries an
    explicit one -- the spec's worked example of two batches that are not one
    battery.
  * `mle_only` carries a few hundred steps of history in its stage. It is the
    real barely-started case.

Run: python -m pytest analysis/tests -q
"""

import numpy as np
import pytest

from analysis import keys as K
from analysis.checks import State, check_confounds
from analysis.tests import fixtures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(res, subject):
    """The single row for a subject, or a loud failure.

    Deliberately not a filter that shrugs on an empty list: a check that quietly
    stops emitting a subject would otherwise turn every assertion about it into
    a vacuous pass."""
    hits = [r for r in res.rows if r.subject == subject]
    assert len(hits) == 1, (
        f'{subject!r}: expected exactly one row, got {len(hits)}. '
        f'rows={[r.subject for r in res.rows]}')
    return hits[0]


def _state(res, subject):
    return _row(res, subject).state


def _has(res, prefix):
    return [r for r in res.rows if r.subject.startswith(prefix)]


def _copy(run):
    """A deep copy with nothing changed -- the base for edits `mutate` cannot
    express, namely REMOVING a config key (its `drop` reaches history and
    summary only)."""
    return fixtures.mutate(run)


def _blob(run):
    import copy
    return copy.deepcopy(K._value(run.config, K.CFG_WANDB_BLOB))


def _set_commit(run, commit):
    blob = _blob(run)
    for entry in blob['e'].values():
        entry['git']['commit'] = commit
    return fixtures.mutate(run, config={K.CFG_WANDB_BLOB: blob})


def _strip_commit(run):
    blob = _blob(run)
    for entry in blob['e'].values():
        entry.pop('git', None)
    return fixtures.mutate(run, config={K.CFG_WANDB_BLOB: blob})


def _enter_stage_late(run, ticks=2):
    """Rewrite the stage series so the run transitioned `ticks` logged points
    before the end -- the §4 'runs barely started / phase-2 injection point'
    case, on a run that really did transition."""
    s, v = run.history[K.STAGE_METRIC]
    v2 = np.full_like(np.asarray(v, float), float(v[0]))
    v2[-ticks:] = float(v[0]) + 1
    return fixtures.mutate(run, history={K.STAGE_METRIC: (s, v2)},
                           summary={K.STAGE_METRIC: float(v2[-1])})


def _knobs(run):
    return set(run.config) - set(K.CFG_IDENTITY)


def _differing_knobs(a, b):
    """Independent re-derivation of the sweep set, so the test does not agree
    with the check by construction."""
    ka, kb = _knobs(a), _knobs(b)
    shared = {k for k in ka & kb
              if K._value(a.config, k) != K._value(b.config, k)}
    return shared | (ka ^ kb)


def _matched_to(a, b, omit):
    """`b` with every knob it shares with `a` set to `a`'s value, then `omit`
    removed: two arms that differ ONLY by omission."""
    shared = {k: K._value(a.config, k) for k in _knobs(a) & _knobs(b)}
    out = fixtures.mutate(b, config=shared)
    out.config.pop(omit)
    return out


# ---------------------------------------------------------------------------
# T vs eval_T
# ---------------------------------------------------------------------------

def test_T_agrees_on_every_captured_run(all_runs):
    """The unmutated corpus: nothing fires. Half of every mutation test."""
    for name, run in all_runs.items():
        res = check_confounds(run)
        assert _state(res, f'{name}/T') is State.OK, name


def test_T_mismatch_fires(tb_ramp):
    assert _state(check_confounds(tb_ramp), 'tb_ramp/T') is State.OK
    m = fixtures.mutate(tb_ramp, config={K.CFG_EVAL_T: 40})
    row = _row(check_confounds(m), 'tb_ramp/T')
    assert row.state is State.FLAG
    assert row.numbers[K.CFG_TRAIN_T] == 10 and row.numbers[K.CFG_EVAL_T] == 40


def test_T_set_to_null_is_unreadable_not_a_pass(tb_ramp):
    m = fixtures.mutate(tb_ramp, config={K.CFG_EVAL_T: None})
    row = _row(check_confounds(m), 'tb_ramp/T')
    assert row.state is State.UNREADABLE
    assert '<null>' in row.detail


def test_T_key_absent_is_reported_as_absent_not_as_null(tb_ramp):
    """A missing key and a key holding null are different findings: the first
    says this config came from a different tree."""
    m = _copy(tb_ramp)
    m.config.pop(K.CFG_EVAL_T)
    row = _row(check_confounds(m), 'tb_ramp/T')
    assert row.state is State.UNREADABLE
    assert '<missing>' in row.detail and '<null>' not in row.detail


# ---------------------------------------------------------------------------
# Code version stamp
# ---------------------------------------------------------------------------

def test_every_captured_run_carries_a_commit(all_runs):
    for name, run in all_runs.items():
        assert _state(check_confounds(run), f'{name}/code_version') is State.OK, name


def test_missing_commit_stamp_fires(tb_ramp):
    row = _row(check_confounds(_strip_commit(tb_ramp)), 'tb_ramp/code_version')
    assert row.state is State.FLAG


def test_commit_is_reported_verbatim(tb_ramp):
    assert _row(check_confounds(tb_ramp), 'tb_ramp/code_version').detail == \
        K.git_commit(tb_ramp.config)


# ---------------------------------------------------------------------------
# Start condition, per run
# ---------------------------------------------------------------------------

def test_start_condition_is_a_row_not_a_verdict(tb_resumed):
    """`tb_resumed` really did resume from an explicit checkpoint. That is
    reported, not flagged: what makes a resume a confound is a SIBLING that
    started somewhere else, which is the battery subject."""
    row = _row(check_confounds(tb_resumed), 'tb_resumed/start_condition')
    assert row.state is State.OK
    assert K._value(tb_resumed.config, K.CFG_CHECKPOINT_NAME) in row.detail


def test_start_condition_unreadable_when_both_keys_are_gone(tb_ramp):
    m = _copy(tb_ramp)
    m.config.pop(K.CFG_CONTINUE_FROM_CHECKPOINT)
    m.config.pop(K.CFG_CHECKPOINT_NAME)
    assert _state(check_confounds(m), 'tb_ramp/start_condition') is State.UNREADABLE


def test_start_condition_distinguishes_null_from_missing(vg_normal):
    """`vg_normal` holds `checkpoint_name: None`."""
    assert '<null>' in _row(check_confounds(vg_normal),
                            'vg_normal/start_condition').detail
    m = _copy(vg_normal)
    m.config.pop(K.CFG_CHECKPOINT_NAME)
    assert '<missing>' in _row(check_confounds(m),
                               'vg_normal/start_condition').detail


# ---------------------------------------------------------------------------
# Stage residence -- runs barely started
# ---------------------------------------------------------------------------

def test_residence_is_exact_when_the_transition_is_in_the_history(vg_normal):
    row = _row(check_confounds(vg_normal), 'vg_normal/stage_residence')
    assert row.state is State.OK
    assert row.numbers['n_boundaries'] == 1
    assert row.numbers['steps_in_stage'] == (row.numbers['last_step']
                                             - row.numbers['entered_at'])


def test_residence_is_a_lower_bound_when_no_transition_was_logged(tb_ramp):
    """`tb_ramp` restarted mid-stage: its history begins with `phase` already
    at its final value, so residence is bounded below by the span, not known."""
    row = _row(check_confounds(tb_ramp), 'tb_ramp/stage_residence')
    assert row.state is State.OK
    assert row.numbers['n_boundaries'] == 0
    assert 'LOWER BOUND' in row.detail


def test_barely_started_fires_on_a_real_run(mle_only):
    """`mle_only` resumed and then ran a few hundred steps. Real data, real
    fire -- the flag is not an artefact of the mutation helper."""
    row = _row(check_confounds(mle_only), 'mle_only/stage_residence')
    assert row.state is State.FLAG
    assert row.numbers['steps_in_stage'] < 1000


def test_barely_started_fires_after_a_late_transition(vg_blowup):
    assert _state(check_confounds(vg_blowup),
                  'vg_blowup/stage_residence') is State.OK
    row = _row(check_confounds(_enter_stage_late(vg_blowup)),
               'vg_blowup/stage_residence')
    assert row.state is State.FLAG
    assert row.numbers['n_boundaries'] == 1
    assert row.numbers['steps_in_stage'] < row.numbers['last_step']


def test_residence_unreadable_without_a_stage_series(tb_ramp):
    m = fixtures.mutate(tb_ramp, drop=(K.STAGE_METRIC,))
    assert _state(check_confounds(m), 'tb_ramp/stage_residence') is State.UNREADABLE
    assert _state(check_confounds(m), 'tb_ramp/stage_boundary') is State.UNREADABLE


def test_residence_refuses_the_summary_fallback(tb_ramp):
    """THE TRAP. `base.series` answers a missing history series from the
    SUMMARY, as one point at `last_step`. Read as a residence that says the run
    entered its stage this instant, and the barely-started flag fires on a hole
    in the data rather than on a short stage."""
    m = _copy(tb_ramp)
    m.history.pop(K.STAGE_METRIC)
    assert K.STAGE_METRIC in m.summary
    row = _row(check_confounds(m), 'tb_ramp/stage_residence')
    assert row.state is State.UNREADABLE
    assert 'steps_in_stage' not in row.numbers


# ---------------------------------------------------------------------------
# Stage boundary inside the read window
# ---------------------------------------------------------------------------

def test_boundary_without_a_window_is_reported_not_flagged(buildout):
    """With no window the read is the whole history, so every boundary is
    inside it by construction; flagging that fires on every multi-stage run
    while saying nothing about the read."""
    row = _row(check_confounds(buildout), 'buildout/stage_boundary')
    assert row.state is State.OK
    assert row.numbers['n_boundaries'] == 4


def test_boundary_outside_the_window_does_not_fire(vg_blowup):
    row = _row(check_confounds(vg_blowup, window=500),
               'vg_blowup/stage_boundary')
    assert row.state is State.OK
    assert row.numbers['n_in_window'] == 0


def test_boundary_inside_the_window_fires(vg_blowup):
    row = _row(check_confounds(vg_blowup, window=2000),
               'vg_blowup/stage_boundary')
    assert row.state is State.FLAG
    assert row.numbers['n_in_window'] == 1


def test_boundary_fires_on_a_mutated_late_transition(vg_blowup):
    m = _enter_stage_late(vg_blowup)
    assert _state(check_confounds(vg_blowup, window=500),
                  'vg_blowup/stage_boundary') is State.OK
    assert _state(check_confounds(m, window=500),
                  'vg_blowup/stage_boundary') is State.FLAG


# ---------------------------------------------------------------------------
# Battery: code version drift
# ---------------------------------------------------------------------------

def test_ring_arms_share_a_commit(ring_probe, ring_cal):
    row = _row(check_confounds([ring_probe, ring_cal]), 'battery/code_version')
    assert row.state is State.OK
    assert row.numbers['n_commits'] == 1


def test_commit_drift_fires_and_names_both_commits(ring_probe, ring_cal):
    other = 'f' * 40
    res = check_confounds([ring_probe, _set_commit(ring_cal, other)])
    row = _row(res, 'battery/code_version')
    assert row.state is State.FLAG
    assert row.numbers['n_commits'] == 2
    assert K.git_commit(ring_probe.config) in row.detail and other in row.detail


def test_commit_drift_fires_across_real_batteries(ring_probe, vg_normal):
    """Two arms from different weeks of this corpus. No mutation involved."""
    assert _state(check_confounds([ring_probe, vg_normal]),
                  'battery/code_version') is State.FLAG


def test_a_missing_stamp_counts_as_its_own_group(ring_probe, ring_cal):
    res = check_confounds([ring_probe, _strip_commit(ring_cal)])
    assert _state(res, 'battery/code_version') is State.FLAG


# ---------------------------------------------------------------------------
# Battery: checkpoint_name mixed null / non-null
# ---------------------------------------------------------------------------

def test_uniformly_named_checkpoints_do_not_fire(ring_probe, ring_cal):
    row = _row(check_confounds([ring_probe, ring_cal]),
               'battery/checkpoint_name')
    assert row.state is State.OK and row.numbers['n_named'] == 2


def test_uniformly_null_checkpoints_do_not_fire(vg_normal, vg_blowup):
    row = _row(check_confounds([vg_normal, vg_blowup]),
               'battery/checkpoint_name')
    assert row.state is State.OK and row.numbers['n_null'] == 2


def test_mixed_checkpoints_fire_on_real_arms(buildout, tb_resumed):
    """The spec's worked example: arms with `checkpoint_name: None` beside arms
    that carried an explicit checkpoint are two batches, not one battery."""
    row = _row(check_confounds([buildout, tb_resumed]),
               'battery/checkpoint_name')
    assert row.state is State.FLAG
    assert row.numbers['n_named'] == 1 and row.numbers['n_null'] == 1


def test_mixed_checkpoints_fire_after_mutation(ring_probe, ring_cal):
    m = fixtures.mutate(ring_cal, config={K.CFG_CHECKPOINT_NAME: None})
    assert _state(check_confounds([ring_probe, m]),
                  'battery/checkpoint_name') is State.FLAG


def test_an_absent_checkpoint_key_is_counted_apart_from_a_null_one(
        ring_probe, ring_cal):
    m = _copy(ring_cal)
    m.config.pop(K.CFG_CHECKPOINT_NAME)
    row = _row(check_confounds([ring_probe, m]), 'battery/checkpoint_name')
    assert row.state is State.FLAG
    assert row.numbers['n_missing'] == 1 and row.numbers['n_null'] == 0


# ---------------------------------------------------------------------------
# Battery: differing start conditions
# ---------------------------------------------------------------------------

_START = (K.CFG_PRIOR_PATH, K.CFG_CONTINUE_FROM_CHECKPOINT, K.CFG_SEED,
          K.CFG_ENERGY_FUNCTION, K.CFG_TRAIN_T, K.CFG_EVAL_T)


def test_start_conditions_agree_across_the_ring_arms(ring_probe, ring_cal):
    res = check_confounds([ring_probe, ring_cal])
    for key in _START:
        assert _state(res, f'battery/start/{key}') is State.OK, key


@pytest.mark.parametrize('key, value', [
    (K.CFG_PRIOR_PATH, 'somewhere_else.pt'),
    (K.CFG_CONTINUE_FROM_CHECKPOINT, True),
    (K.CFG_SEED, 999),
    (K.CFG_ENERGY_FUNCTION, 'latent_multiharmonic'),
    (K.CFG_TRAIN_T, 40),
    (K.CFG_EVAL_T, 40),
])
def test_each_start_condition_fires_when_one_arm_differs(
        ring_probe, ring_cal, key, value):
    res = check_confounds([ring_probe, fixtures.mutate(ring_cal,
                                                       config={key: value})])
    row = _row(res, f'battery/start/{key}')
    assert row.state is State.FLAG
    assert row.numbers['n_values'] == 2
    assert str(value) in row.detail


def test_differing_start_conditions_fire_on_real_arms(vg_normal, buildout):
    """Different problem and different T, unmutated."""
    res = check_confounds([vg_normal, buildout])
    assert _state(res, f'battery/start/{K.CFG_ENERGY_FUNCTION}') is State.FLAG
    assert _state(res, f'battery/start/{K.CFG_TRAIN_T}') is State.FLAG


# ---------------------------------------------------------------------------
# Battery: duplicates
# ---------------------------------------------------------------------------

def test_ring_arms_are_the_same_arm_written_twice(ring_probe, ring_cal):
    """REAL DATA. `ring_probe.yaml` and `ring_probe_cal.yaml` differ in
    `run_name` and nothing else, so the whole two-arm battery is one arm."""
    res = check_confounds([ring_probe, ring_cal])
    row = _row(res, 'battery/duplicate/ring_probe~ring_cal')
    assert row.state is State.FLAG
    assert row.numbers['n_differing'] == 0
    assert _row(res, 'battery/duplicates').numbers['n_duplicate'] == 1


def test_arms_that_really_differ_are_not_duplicates(vg_normal, vg_blowup):
    res = check_confounds([vg_normal, vg_blowup])
    assert _row(res, 'battery/duplicates').numbers['n_duplicate'] == 0
    assert not _has(res, 'battery/duplicate/')


def test_arms_differing_only_by_omission_fire(vg_normal, vg_blowup):
    """The absent knob takes its default, so a pair whose shared knobs all
    agree is a duplicate however many keys one of them is missing."""
    m = _matched_to(vg_normal, vg_blowup, omit=K.CFG_SEED)
    res = check_confounds([vg_normal, m])
    row = _row(res, 'battery/duplicate/vg_normal~vg_blowup')
    assert row.state is State.FLAG
    assert row.numbers['n_present_only_in_one'] == 1
    assert K.CFG_SEED in row.detail


def test_identity_keys_alone_do_not_make_two_arms_distinct(ring_probe, ring_cal):
    m = fixtures.mutate(ring_cal, config={K.CFG_RUN_NAME: 'a_different_name',
                                          K.CFG_TAG: 'another_tag',
                                          K.CFG_EPOCHS: 12345})
    assert _state(check_confounds([ring_probe, m]),
                  'battery/duplicate/ring_probe~ring_cal') is State.FLAG


def test_a_nan_knob_does_not_invent_a_sweep_dimension(ring_probe, ring_cal):
    """`nan != nan` under the default comparison, so one NaN-valued knob would
    otherwise make every arm differ from every other on it -- a sweep dimension
    nobody swept, and a duplicate pair that stops being reported."""
    nan = float('nan')
    a = fixtures.mutate(ring_probe, config={K.CFG_SEED: nan})
    b = fixtures.mutate(ring_cal, config={K.CFG_SEED: nan})
    res = check_confounds([a, b])
    assert _state(res, 'battery/duplicate/ring_probe~ring_cal') is State.FLAG
    assert _row(res, 'battery/sweep').numbers['n_knobs'] == 0


def test_one_real_knob_is_enough_to_stop_a_duplicate(ring_probe, ring_cal):
    m = fixtures.mutate(ring_cal, config={K.CFG_SEED: 999})
    res = check_confounds([ring_probe, m])
    assert not _has(res, 'battery/duplicate/')
    assert _row(res, 'battery/duplicates').numbers['n_duplicate'] == 0


# ---------------------------------------------------------------------------
# Battery: the sweep table
# ---------------------------------------------------------------------------

def test_sweep_counts_the_knobs_that_actually_differ(vg_normal, vg_blowup):
    row = _row(check_confounds([vg_normal, vg_blowup]), 'battery/sweep')
    assert row.state is State.OK
    assert row.numbers['n_knobs'] == len(_differing_knobs(vg_normal, vg_blowup))
    assert row.numbers['n_knobs'] > 0


def test_sweep_excludes_the_identity_keys(vg_normal, vg_blowup):
    """These two arms differ in `run_name` and `epochs`; a sweep table that
    lists them is listing the thing the sweep is indexed by."""
    for key in (K.CFG_RUN_NAME, K.CFG_EPOCHS):
        assert K._value(vg_normal.config, key) != K._value(vg_blowup.config, key)
    row = _row(check_confounds([vg_normal, vg_blowup]), 'battery/sweep')
    assert K.CFG_RUN_NAME not in row.detail and K.CFG_EPOCHS not in row.detail


def test_sweep_counts_a_knob_present_in_only_one_arm(buildout, tb_resumed):
    row = _row(check_confounds([buildout, tb_resumed]), 'battery/sweep')
    assert row.numbers['by_presence'] > 0
    assert row.numbers['n_knobs'] == (row.numbers['by_presence']
                                      + row.numbers['by_value'])


def test_duplicate_arms_sweep_nothing(ring_probe, ring_cal):
    row = _row(check_confounds([ring_probe, ring_cal]), 'battery/sweep')
    assert row.numbers['n_knobs'] == 0


# ---------------------------------------------------------------------------
# Shape of the result
# ---------------------------------------------------------------------------

def test_a_single_run_is_checked_not_skipped(tb_ramp):
    """`not_run` here would throw away the T, code-version, start-condition and
    residence answers, which are properties of the run and not of a battery."""
    res = check_confounds([tb_ramp])
    assert res.ran
    assert _state(res, 'battery') is State.OK
    for subject in ('T', 'code_version', 'start_condition', 'stage_residence',
                    'stage_boundary'):
        _row(res, f'tb_ramp/{subject}')


def test_a_bare_run_is_normalised(tb_ramp):
    bare, listed = check_confounds(tb_ramp), check_confounds([tb_ramp])
    assert ([(r.subject, r.state, r.detail) for r in bare.rows]
            == [(r.subject, r.state, r.detail) for r in listed.rows])


def test_no_runs_is_not_run_with_a_reason():
    res = check_confounds([])
    assert not res.ran and res.reason


def test_a_single_run_emits_no_cross_arm_subjects(tb_ramp):
    res = check_confounds(tb_ramp)
    assert not _has(res, 'battery/')


# ---------------------------------------------------------------------------
# Corpus invariants
# ---------------------------------------------------------------------------

def test_every_captured_run_reads_alone(all_runs):
    for name, run in all_runs.items():
        res = check_confounds(run)
        assert res.ran and len(res.rows) >= 6, name


def test_the_whole_corpus_reads_as_one_battery(all_runs):
    runs = list(all_runs.values())
    res = check_confounds(runs, window=1000)
    assert res.ran
    assert _row(res, 'battery/code_version').numbers['n_arms'] == len(runs)
    assert _row(res, 'battery/duplicates').numbers['n_pairs'] == \
        len(runs) * (len(runs) - 1) // 2
    for name in all_runs:
        _row(res, f'{name}/stage_residence')


def test_findings_carry_the_numbers_behind_them(all_runs):
    """A finding without its inputs is an assertion, and this package does not
    make assertions. The two states that legitimately have nothing to show are
    the ones whose whole content is that the input was missing."""
    res = check_confounds(list(all_runs.values()))
    for row in res.findings:
        assert row.numbers or row.state is State.UNREADABLE, row.subject
        assert row.detail, row.subject
