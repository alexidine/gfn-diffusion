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
  * `buildout` and `tb_resumed` ran on one commit while differing in many
    knobs -- the no-drift case that is not also a duplicate pair.
  * `buildout` has `checkpoint_name: None` while `tb_resumed` carries an
    explicit one -- the spec's worked example of two batches that are not one
    battery.
  * `mle_only` carries a few hundred steps of history in its stage. It is the
    real barely-started case, and it is the LOWER-BOUND one: it logged no
    transition, so its residence is not known, only bounded.

THE TWO THRESHOLDS ARE STRADDLED, NOT RESTATED. The short-stage bar and the
window edge are pinned by a pair of runs either side of each, rather than by a
number copied out of the module -- a copy keeps passing on whichever side of a
moved bar both cases happen to land.

Run: python -m pytest analysis/tests -q
"""

import numpy as np
import pytest

from analysis import keys as K
# Imported from the public module rather than from `_parts`, so this file keeps
# working once the part modules are concatenated into `analysis/checks.py`.
from analysis.checks import Context, State, check_confounds
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


def _stage(run):
    s, v = run.history[K.STAGE_METRIC]
    return np.asarray(s, float), np.asarray(v, float)


def _boundaries(run):
    """Steps at which the stage metric changed, re-derived from the raw series
    so the test does not take the check's word for where they are."""
    s, v = _stage(run)
    return s[np.nonzero(np.diff(v) != 0)[0] + 1]


def _grid(run):
    """The stage metric's logging interval -- the smallest step by which a
    straddle can move a boundary."""
    return float(np.diff(_stage(run)[0])[0])


def _enter_stage_late(run, ticks=2):
    """Rewrite the stage series so the run transitioned `ticks` logged points
    before the end -- the §4 'runs barely started / phase-2 injection point'
    case, on a run that really did transition."""
    s, v = _stage(run)
    v2 = np.full_like(v, float(v[0]))
    v2[-ticks:] = float(v[0]) + 1
    return fixtures.mutate(run, history={K.STAGE_METRIC: (s, v2)},
                           summary={K.STAGE_METRIC: float(v2[-1])})


def _transition_at(run, step):
    """A copy whose stage series changes value exactly at `step`, so the
    residence the check reads is `last_step - step`."""
    s, _ = _stage(run)
    v2 = np.where(s < float(step), 1.0, 2.0)
    return fixtures.mutate(run, history={K.STAGE_METRIC: (s, v2)},
                           summary={K.STAGE_METRIC: 2.0})


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


def test_T_mismatch_fires_in_the_other_direction(vg_normal):
    """Both ways round, because the subject is a comparison and not an order: a
    run evaluated on a SHORTER integrator than it trained on is the same
    confound, and a one-sided test would pass while blind to half of it."""
    assert _state(check_confounds(vg_normal), 'vg_normal/T') is State.OK
    m = fixtures.mutate(vg_normal, config={K.CFG_TRAIN_T: 10})
    row = _row(check_confounds(m), 'vg_normal/T')
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


def test_a_non_numeric_T_is_unreadable(tb_ramp):
    """A string where a number belongs is neither a mismatch nor a match, and
    reporting it as either would be a guess."""
    m = fixtures.mutate(tb_ramp, config={K.CFG_EVAL_T: 'auto'})
    row = _row(check_confounds(m), 'tb_ramp/T')
    assert row.state is State.UNREADABLE
    assert 'auto' in row.detail


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


def test_one_start_key_of_the_two_still_answers(tb_ramp):
    """Only losing BOTH makes the start condition unreadable -- otherwise the
    unreadable state would fire on every config that omits one knob."""
    m = _copy(tb_ramp)
    m.config.pop(K.CFG_CHECKPOINT_NAME)
    row = _row(check_confounds(m), 'tb_ramp/start_condition')
    assert row.state is State.OK
    assert '<missing>' in row.detail


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
    assert row.numbers['steps_in_stage'] == (row.numbers['last_step']
                                             - row.numbers['entered_at'])


def test_barely_started_fires_after_a_late_transition(vg_blowup):
    assert _state(check_confounds(vg_blowup),
                  'vg_blowup/stage_residence') is State.OK
    row = _row(check_confounds(_enter_stage_late(vg_blowup)),
               'vg_blowup/stage_residence')
    assert row.state is State.FLAG
    assert row.numbers['n_boundaries'] == 1
    assert row.numbers['steps_in_stage'] < row.numbers['last_step']


def test_the_short_stage_flag_says_which_of_the_two_things_it_knows(
        mle_only, vg_blowup):
    """Two sentences, and the difference is load-bearing.

    With the entry IN the history the residence is the residence, so a short one
    means the read is of the injection point. With no entry logged the true
    residence is unknown -- the run may have sat in that stage for a hundred
    thousand steps and be readable for four hundred of them -- and claiming an
    injection point would assert what the data does not say."""
    known = _row(check_confounds(_enter_stage_late(vg_blowup)),
                 'vg_blowup/stage_residence')
    assert known.numbers['n_boundaries'] == 1
    assert 'injection point' in known.detail

    bounded = _row(check_confounds(mle_only), 'mle_only/stage_residence')
    assert bounded.numbers['n_boundaries'] == 0
    assert 'injection point' not in bounded.detail
    assert 'LOWER BOUND' in bounded.detail


def test_the_short_stage_bar_is_straddled(vg_blowup):
    """The bar itself, pinned by one run either side of it. Restating the
    module's constant here would keep passing on whichever side of a moved bar
    both cases happened to land."""
    bar, grid = 1000.0, _grid(vg_blowup)
    now = vg_blowup.last_step
    at_bar = _row(check_confounds(_transition_at(vg_blowup, now - bar)),
                  'vg_blowup/stage_residence')
    under_bar = _row(check_confounds(_transition_at(vg_blowup,
                                                    now - bar + grid)),
                     'vg_blowup/stage_residence')
    assert at_bar.numbers['steps_in_stage'] == bar
    assert at_bar.state is State.OK
    assert under_bar.numbers['steps_in_stage'] == bar - grid
    assert under_bar.state is State.FLAG


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


def test_a_one_point_stage_history_is_refused(tb_ramp):
    """The same hole as the summary fallback, arriving through history."""
    m = fixtures.mutate(tb_ramp, history={K.STAGE_METRIC: ([100.0], [2.0])})
    row = _row(check_confounds(m), 'tb_ramp/stage_residence')
    assert row.state is State.UNREADABLE
    assert 'steps_in_stage' not in row.numbers


def test_an_all_nan_stage_series_is_refused(tb_ramp):
    """A logged-but-empty series is a hole, not a stage entered at step zero."""
    s, v = _stage(tb_ramp)
    m = fixtures.mutate(tb_ramp,
                        history={K.STAGE_METRIC: (s, np.full_like(v, np.nan))})
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
    assert row.numbers['n_boundaries'] == len(_boundaries(buildout))
    assert row.numbers['n_boundaries'] > 1


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
    assert row.numbers['nearest'] == float(_boundaries(vg_blowup)[-1])


def test_boundary_fires_on_a_mutated_late_transition(vg_blowup):
    m = _enter_stage_late(vg_blowup)
    assert _state(check_confounds(vg_blowup, window=500),
                  'vg_blowup/stage_boundary') is State.OK
    assert _state(check_confounds(m, window=500),
                  'vg_blowup/stage_boundary') is State.FLAG


def test_the_window_edge_is_straddled(vg_blowup):
    """A boundary exactly at the far edge of the window is OUTSIDE it. Pinned
    by a pair either side, for the same reason the residence bar is."""
    span = vg_blowup.last_step - float(_boundaries(vg_blowup)[-1])
    assert _state(check_confounds(vg_blowup, window=span),
                  'vg_blowup/stage_boundary') is State.OK
    assert _state(check_confounds(vg_blowup, window=span + _grid(vg_blowup)),
                  'vg_blowup/stage_boundary') is State.FLAG


def test_a_rewind_counts_as_a_boundary(tb_ramp):
    """Any change counts, in EITHER direction. A run put back into an earlier
    stage straddles the read exactly as an advance does, and a check that
    looked for an increase would pass over it in silence."""
    assert _state(check_confounds(tb_ramp, window=1000),
                  'tb_ramp/stage_boundary') is State.OK
    s, v = _stage(tb_ramp)
    v2 = v.copy()
    v2[-3:] = v[0] - 1
    m = fixtures.mutate(tb_ramp, history={K.STAGE_METRIC: (s, v2)})
    row = _row(check_confounds(m, window=1000), 'tb_ramp/stage_boundary')
    assert row.state is State.FLAG
    assert row.numbers['n_in_window'] == 1


def test_a_run_with_no_boundary_at_all_reports_the_window_it_used(tb_ramp):
    """`tb_ramp` logged no transition, so nothing can be inside the window --
    and the row still carries the window, or this cannot be told apart from a
    check that ignored it."""
    row = _row(check_confounds(tb_ramp, window=1000), 'tb_ramp/stage_boundary')
    assert row.state is State.OK
    assert row.numbers['n_in_window'] == 0
    assert row.numbers['window'] == 1000
    assert np.isnan(row.numbers['last_boundary'])


# ---------------------------------------------------------------------------
# Battery: code version drift
# ---------------------------------------------------------------------------

def test_ring_arms_share_a_commit(ring_probe, ring_cal):
    row = _row(check_confounds([ring_probe, ring_cal]), 'battery/code_version')
    assert row.state is State.OK
    assert row.numbers['n_commits'] == 1


def test_two_arms_that_really_differ_can_still_share_a_commit(buildout,
                                                              tb_resumed):
    """The no-drift case on arms that are NOT the same arm twice -- otherwise
    'does not fire' could be a property of the pair being duplicates."""
    res = check_confounds([buildout, tb_resumed])
    row = _row(res, 'battery/code_version')
    assert row.state is State.OK and row.numbers['n_commits'] == 1
    assert not _has(res, 'battery/duplicate/')


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
    """An unstamped arm beside a stamped one is drift that cannot be RULED
    OUT, which the battery reports as drift."""
    row = _row(check_confounds([ring_probe, _strip_commit(ring_cal)]),
               'battery/code_version')
    assert row.state is State.FLAG
    assert row.numbers['n_commits'] == 2


def test_a_battery_with_no_stamps_at_all_is_one_group(ring_probe, ring_cal):
    """Neither arm stamped is not drift BETWEEN them. The per-run rows are
    where an absent stamp is a finding, and they still fire -- so this is not
    the check going quiet."""
    res = check_confounds([_strip_commit(ring_probe), _strip_commit(ring_cal)])
    assert _state(res, 'battery/code_version') is State.OK
    assert _state(res, 'ring_probe/code_version') is State.FLAG
    assert _state(res, 'ring_cal/code_version') is State.FLAG


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


def test_uniformly_absent_checkpoint_key_does_not_fire(ring_probe, ring_cal):
    """No arm records the knob: nothing separates them, so there is nothing to
    flag. The state has to survive the key being gone from every arm, not only
    from none of them."""
    a, b = _copy(ring_probe), _copy(ring_cal)
    a.config.pop(K.CFG_CHECKPOINT_NAME)
    b.config.pop(K.CFG_CHECKPOINT_NAME)
    row = _row(check_confounds([a, b]), 'battery/checkpoint_name')
    assert row.state is State.OK
    assert row.numbers['n_missing'] == 2 and row.numbers['n_named'] == 0


def test_mixed_checkpoints_fire_on_real_arms(buildout, tb_resumed):
    """The spec's worked example: arms with `checkpoint_name: None` beside arms
    that carried an explicit checkpoint are two batches, not one battery."""
    row = _row(check_confounds([buildout, tb_resumed]),
               'battery/checkpoint_name')
    assert row.state is State.FLAG
    assert row.numbers['n_named'] == 1 and row.numbers['n_null'] == 1
    assert 'buildout' in row.detail and 'tb_resumed' in row.detail


def test_mixed_checkpoints_fire_after_mutation(ring_probe, ring_cal):
    m = fixtures.mutate(ring_cal, config={K.CFG_CHECKPOINT_NAME: None})
    assert _state(check_confounds([ring_probe, ring_cal]),
                  'battery/checkpoint_name') is State.OK
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


def test_every_start_key_gets_its_own_row(ring_probe, ring_cal):
    """A key that stopped being examined would make every assertion about it
    vacuous, and the row is the record that it was considered at all."""
    res = check_confounds([ring_probe, ring_cal])
    for key in _START:
        assert _row(res, f'battery/start/{key}').numbers['n_arms'] == 2


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


@pytest.mark.parametrize('key', _START)
def test_a_start_key_missing_from_one_arm_fires(ring_probe, ring_cal, key):
    """An absent knob takes its default, and a default beside a sibling's
    explicit value is a different start condition whether or not anyone chose
    it. Every start key, because one silently dropped from the loop would leave
    that confound unwatched."""
    m = _copy(ring_cal)
    m.config.pop(key)
    row = _row(check_confounds([ring_probe, m]), f'battery/start/{key}')
    assert row.state is State.FLAG
    assert '<missing>' in row.detail


def test_a_start_key_nulled_in_one_arm_reads_as_null_not_as_missing(
        ring_probe, ring_cal):
    m = fixtures.mutate(ring_cal, config={K.CFG_PRIOR_PATH: None})
    row = _row(check_confounds([ring_probe, m]),
               f'battery/start/{K.CFG_PRIOR_PATH}')
    assert row.state is State.FLAG
    assert '<null>' in row.detail and '<missing>' not in row.detail


def test_differing_start_conditions_fire_on_real_arms(vg_normal, buildout):
    """Different problem and different T, unmutated."""
    res = check_confounds([vg_normal, buildout])
    assert _state(res, f'battery/start/{K.CFG_ENERGY_FUNCTION}') is State.FLAG
    assert _state(res, f'battery/start/{K.CFG_TRAIN_T}') is State.FLAG


def test_T_can_agree_within_each_arm_and_differ_between_them(vg_normal,
                                                             buildout):
    """Why T is a battery subject as well as a per-run one: both arms are
    self-consistent and they are still not comparable to each other."""
    res = check_confounds([vg_normal, buildout])
    assert _state(res, 'vg_normal/T') is State.OK
    assert _state(res, 'buildout/T') is State.OK
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
    assert row.numbers['n_present_only_in_one'] == 0
    assert 'identical configs' in row.detail
    assert _row(res, 'battery/duplicates').numbers['n_duplicate'] == 1


def test_arms_that_really_differ_are_not_duplicates(vg_normal, vg_blowup):
    res = check_confounds([vg_normal, vg_blowup])
    assert _row(res, 'battery/duplicates').numbers['n_duplicate'] == 0
    assert not _has(res, 'battery/duplicate/')


def test_arms_differing_only_by_omission_fire(vg_normal, vg_blowup):
    """The absent knob takes its default, so a pair whose shared knobs all
    agree is a duplicate however many keys one of them is missing. The wording
    separates this from the identical-config case -- they are the same finding
    about two different mistakes."""
    m = _matched_to(vg_normal, vg_blowup, omit=K.CFG_SEED)
    res = check_confounds([vg_normal, m])
    row = _row(res, 'battery/duplicate/vg_normal~vg_blowup')
    assert row.state is State.FLAG
    assert row.numbers['n_present_only_in_one'] == 1
    assert K.CFG_SEED in row.detail
    assert 'defaults' in row.detail and 'identical configs' not in row.detail


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


def test_a_nan_beside_a_number_is_still_a_real_difference(ring_probe, ring_cal):
    """The other half of the NaN rule: nan-equals-nan must not become
    everything-equals-nan, which would hide a swept knob."""
    a = fixtures.mutate(ring_probe, config={K.CFG_SEED: float('nan')})
    b = fixtures.mutate(ring_cal, config={K.CFG_SEED: 1.0})
    res = check_confounds([a, b])
    assert not _has(res, 'battery/duplicate/')
    assert _row(res, 'battery/sweep').numbers['n_knobs'] == 1


def test_one_real_knob_is_enough_to_stop_a_duplicate(ring_probe, ring_cal):
    m = fixtures.mutate(ring_cal, config={K.CFG_SEED: 999})
    res = check_confounds([ring_probe, m])
    assert not _has(res, 'battery/duplicate/')
    assert _row(res, 'battery/duplicates').numbers['n_duplicate'] == 0


def test_duplicates_are_counted_over_pairs_not_over_arms(ring_probe, ring_cal):
    """Three arms, one duplicated pair among them: the count is over PAIRS, and
    a battery is not disqualified wholesale by one repeated arm."""
    third = fixtures.mutate(ring_cal, config={K.CFG_SEED: 7})
    third.name = 'ring_seeded'
    res = check_confounds([ring_probe, ring_cal, third])
    row = _row(res, 'battery/duplicates')
    assert row.numbers['n_pairs'] == 3 and row.numbers['n_duplicate'] == 1
    assert len(_has(res, 'battery/duplicate/')) == 1


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


def test_dropping_one_knob_from_one_arm_sweeps_it(ring_probe, ring_cal):
    """Mutation of the real duplicate pair: presence alone is a sweep
    dimension, because absent means the default."""
    assert _row(check_confounds([ring_probe, ring_cal]),
                'battery/sweep').numbers['n_knobs'] == 0
    m = _copy(ring_cal)
    m.config.pop(K.CFG_SEED)
    row = _row(check_confounds([ring_probe, m]), 'battery/sweep')
    assert row.numbers['by_presence'] == 1 and row.numbers['by_value'] == 0
    assert K.CFG_SEED in row.detail


def test_duplicate_arms_sweep_nothing(ring_probe, ring_cal):
    row = _row(check_confounds([ring_probe, ring_cal]), 'battery/sweep')
    assert row.numbers['n_knobs'] == 0
    assert 'no knob differs' in row.detail


def test_the_sweep_line_is_capped_but_the_count_is_not(all_runs):
    """The whole corpus sweeps hundreds of knobs. The LINE is shortened; the
    size of the sweep stays in the numbers, so the cap hides nothing."""
    row = _row(check_confounds(list(all_runs.values())), 'battery/sweep')
    shown = row.detail.split('  (+')[0].split(', ')
    assert row.numbers['n_knobs'] > len(shown)
    assert f'(+{row.numbers["n_knobs"] - len(shown)} more)' in row.detail


# ---------------------------------------------------------------------------
# Shape of the result
# ---------------------------------------------------------------------------

def test_a_single_run_is_checked_not_skipped(tb_ramp):
    """`not_run` here would throw away the T, code-version, start-condition and
    residence answers, which are properties of the run and not of a battery."""
    res = check_confounds([tb_ramp])
    assert res.ran
    row = _row(res, 'battery')
    assert row.state is State.OK and row.numbers['n_arms'] == 1
    assert 'skipped' in row.detail
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
    assert not res.rows


def test_none_is_not_run_with_a_reason():
    res = check_confounds(None)
    assert not res.ran and res.reason


def test_a_single_run_emits_no_cross_arm_subjects(tb_ramp):
    res = check_confounds(tb_ramp)
    assert not _has(res, 'battery/')


def test_the_label_falls_back_to_the_run_id(tb_ramp):
    """A run pulled without a name still has to be nameable, or its rows
    collide with every other unnamed arm's in a battery table."""
    m = _copy(tb_ramp)
    m.name = None
    m.run_id = 'abc12345'
    assert _state(check_confounds(m), 'abc12345/T') is State.OK


def test_a_supplied_context_names_the_stage(tb_ramp):
    """The stage name comes from the shared Context when one is given, so two
    checks in one report cannot disagree about which stage a run is in."""
    ctx = Context(stage_index=0, stage_name='a_named_stage',
                  route=K.Route.UNKNOWN, stages=('a', 'b'))
    row = _row(check_confounds(tb_ramp, ctx=ctx), 'tb_ramp/stage_residence')
    assert row.detail.startswith('a_named_stage')


def test_a_battery_resolves_each_arms_stage_for_itself(tb_ramp, vg_blowup):
    """One Context cannot describe several arms, so a battery ignores it rather
    than labelling every arm with the first one's stage."""
    ctx = Context(stage_index=0, stage_name='a_named_stage',
                  route=K.Route.UNKNOWN, stages=('a', 'b'))
    res = check_confounds([tb_ramp, vg_blowup], ctx=ctx)
    for name in ('tb_ramp', 'vg_blowup'):
        assert 'a_named_stage' not in _row(res, f'{name}/stage_residence').detail


def test_an_empty_config_is_unreadable_not_a_pass(tb_ramp):
    """A run whose config did not load answers NOTHING about comparability, and
    saying so is the loudest thing available -- silence here is the swallowed
    diagnostic that reads as reassurance."""
    m = _copy(tb_ramp)
    m.config.clear()
    res = check_confounds(m)
    assert res.ran
    assert _state(res, 'tb_ramp/T') is State.UNREADABLE
    assert _state(res, 'tb_ramp/start_condition') is State.UNREADABLE
    assert _state(res, 'tb_ramp/code_version') is State.FLAG
    # The stage subjects come from history, so they still answer.
    assert _state(res, 'tb_ramp/stage_residence') is State.OK


def test_an_empty_config_in_a_battery_does_not_crash(ring_probe, ring_cal):
    m = _copy(ring_cal)
    m.config.clear()
    res = check_confounds([ring_probe, m])
    assert res.ran and res.findings


def test_a_run_with_no_history_at_all_still_gets_its_config_subjects(tb_ramp):
    """The run whose comparability is in doubt is often the one that logged
    almost nothing, so the config subjects must not depend on a series."""
    m = _copy(tb_ramp)
    m.history.clear()
    m.summary.clear()
    res = check_confounds(m)
    assert _state(res, 'tb_ramp/T') is State.OK
    assert _state(res, 'tb_ramp/code_version') is State.OK
    assert _state(res, 'tb_ramp/stage_residence') is State.UNREADABLE


# ---------------------------------------------------------------------------
# Corpus invariants
# ---------------------------------------------------------------------------

_PER_RUN_SUBJECTS = ('T', 'code_version', 'start_condition', 'stage_residence',
                     'stage_boundary')


def test_every_captured_run_reads_alone(all_runs):
    for name, run in all_runs.items():
        res = check_confounds(run)
        assert res.ran, name
        for subject in _PER_RUN_SUBJECTS:
            _row(res, f'{name}/{subject}')


def test_the_whole_corpus_reads_as_one_battery(all_runs):
    runs = list(all_runs.values())
    res = check_confounds(runs, window=1000)
    assert res.ran
    assert _row(res, 'battery/code_version').numbers['n_arms'] == len(runs)
    assert _row(res, 'battery/duplicates').numbers['n_pairs'] == \
        len(runs) * (len(runs) - 1) // 2
    for name in all_runs:
        for subject in _PER_RUN_SUBJECTS:
            _row(res, f'{name}/{subject}')


def test_every_row_is_attributed_to_an_arm_or_to_the_battery(all_runs):
    """Subjects are namespaced by arm, or a battery table cannot say which arm
    a finding belongs to."""
    res = check_confounds(list(all_runs.values()), window=1000)
    for row in res.rows:
        head = row.subject.split('/')[0]
        assert head == 'battery' or head in all_runs, row.subject


def test_findings_carry_the_numbers_behind_them(all_runs):
    """A finding without its inputs is an assertion, and this package does not
    make assertions. The two states that legitimately have nothing to show are
    the ones whose whole content is that the input was missing."""
    res = check_confounds(list(all_runs.values()))
    for row in res.findings:
        assert row.numbers or row.state is State.UNREADABLE, row.subject
        assert row.detail, row.subject


def _every_result(all_runs):
    """Results spanning both halves of every subject -- the clean runs AND the
    mutations that make each condition fire. A vocabulary check run over the
    passing rows alone would never see the sentences that actually accuse."""
    runs = list(all_runs.values())
    one, vg = all_runs['tb_ramp'], all_runs['vg_blowup']
    out = [check_confounds(r) for r in runs]
    out.append(check_confounds(runs, window=1000))
    out.append(check_confounds([]))
    out.append(check_confounds(fixtures.mutate(one, config={K.CFG_EVAL_T: 40})))
    out.append(check_confounds(fixtures.mutate(one, config={K.CFG_EVAL_T: None})))
    out.append(check_confounds(_strip_commit(one)))
    out.append(check_confounds(fixtures.mutate(one, drop=(K.STAGE_METRIC,))))
    out.append(check_confounds(_enter_stage_late(vg), window=500))
    out.append(check_confounds([one, _set_commit(all_runs['tb_resumed'],
                                                 'f' * 40)], window=200))
    out.append(check_confounds([all_runs['ring_probe'], all_runs['ring_cal']]))
    out.append(check_confounds([all_runs['buildout'], all_runs['tb_resumed']]))
    return out


# Words that turn a reading into a conclusion. The ported wa.py had a
# `bellwether_verdict` printing "healthy climb" and "policy losing ground"; it
# was dropped deliberately, and this is the guard that keeps it from coming
# back through a detail string.
_VERDICT_WORDS = ('healthy', 'unhealthy', 'working', 'broken', 'is fine',
                  'looks good', 'suspicious', 'recommend', 'verdict',
                  'probably', 'likely', 'seems', 'appears to')


def test_no_row_reads_as_a_verdict(all_runs):
    for res in _every_result(all_runs):
        for row in res.rows:
            low = row.detail.lower()
            hits = [w for w in _VERDICT_WORDS if w in low]
            assert not hits, f'{row.subject}: {hits} in {row.detail!r}'
        low = (res.reason or '').lower()
        assert not [w for w in _VERDICT_WORDS if w in low], res.reason


def test_the_state_vocabulary_is_ok_flag_unreadable(all_runs):
    """§4 reads the config and the stage series, and both mean the same thing
    on every route -- so it never answers NA_ROUTE, and it never borrows R2's
    mechanism states, which would put one condition under two checks that can
    then disagree about it."""
    allowed = {State.OK, State.FLAG, State.UNREADABLE}
    for res in _every_result(all_runs):
        for row in res.rows:
            assert row.state in allowed, f'{row.subject}: {row.state}'


# ---------------------------------------------------------------------------
# Defects found by adversarial verification against the real corpus
# ---------------------------------------------------------------------------
# Each of the three below was a SILENT failure on real data: the check ran,
# reported `ran=True`, and said nothing about a condition that was present.

def test_arms_resuming_from_different_checkpoints_fire(ring_probe, ring_cal):
    """§4's second-named confound: 'checkpoint chaining, where arms silently
    resume FROM EACH OTHER rather than a pinned start'.

    Null-ness cannot see it. A real 16-arm battery had every arm carrying a
    checkpoint -- one from a phase-1 exit, the other fifteen from that arm's own
    rolling checkpoint -- and passed the mixed-null test while being exactly the
    thing §4 names."""
    a = fixtures.mutate(ring_probe, config={K.CFG_CHECKPOINT_NAME: 'armA_step12000.pt'})
    b = fixtures.mutate(ring_cal, config={K.CFG_CHECKPOINT_NAME: 'armA_running.pt'})
    res = check_confounds([a, b])
    assert _state(res, 'battery/checkpoint_source') is State.FLAG
    # the mixed-null subject is blind to this, which is why the other exists
    assert _state(res, 'battery/checkpoint_name') is State.OK


def test_arms_from_one_checkpoint_do_not_fire(ring_probe, ring_cal):
    """The companion half. Without it the test above passes on a check that
    flags every battery."""
    same = 'pinned_start.pt'
    res = check_confounds([fixtures.mutate(ring_probe, config={K.CFG_CHECKPOINT_NAME: same}),
                           fixtures.mutate(ring_cal, config={K.CFG_CHECKPOINT_NAME: same})])
    assert _state(res, 'battery/checkpoint_source') is State.OK


def test_a_run_with_no_config_is_not_a_duplicate_of_anything(ring_probe, ring_cal):
    """Zero shared knobs is zero evidence, not agreement.

    `pull` returns config=={} when files/config.yaml is absent or unparseable --
    true of most local run directories -- and the empty-vs-real pair was being
    reported as 'the same arm written two ways' on the strength of nothing."""
    blank = fixtures.mutate(ring_cal)
    blank.config.clear()
    res = check_confounds([ring_probe, blank])
    assert not [r for r in res.findings if 'duplicate/' in r.subject]
    assert _state(res, f'{blank.name}/config') is State.UNREADABLE


def test_real_duplicates_still_fire(ring_probe, ring_cal):
    """The companion: the fix above must not have switched the subject off.
    These two really are one arm written twice."""
    res = check_confounds([ring_probe, ring_cal])
    assert [r for r in res.findings if 'duplicate/' in r.subject]


def test_two_arms_sharing_a_display_name_stay_distinguishable(ring_probe, ring_cal):
    """Display names are not unique -- nine are shared by two or more runs in
    the local corpus. Labelling by name alone gave two arms identical subject
    strings and a row reading `duplicate/mk_dev~mk_dev`, which names one arm
    twice and tells the reader nothing."""
    a, b = fixtures.mutate(ring_probe), fixtures.mutate(ring_cal)
    a.name = b.name = 'mk_dev'
    res = check_confounds([a, b])
    subjects = [r.subject for r in res.rows]
    assert len(subjects) == len(set(subjects)), 'subject strings collided'
    dup = [s for s in subjects if 'duplicate/' in s]
    assert dup and a.run_id in dup[0] and b.run_id in dup[0]
