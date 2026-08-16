"""
Tests for the R14 dead-sensor check.

MUTATION-TESTED THROUGHOUT. Every condition gets two tests: the real run does
NOT fire for that subject, and a real run with the condition re-introduced
DOES. A check that has never fired has not been tested, and this repo has
shipped tests that passed while blind more than once.

The mutations are edits to REAL captured runs (`fixtures.mutate` deep-copies,
so the module-scoped fixtures survive), never hand-built histories. A hand-built
series agrees with whatever the author assumed, which is the failure the whole
package exists to stop.

NOTHING HERE HARDCODES A METRIC NAME. Every subject is discovered the way the
check discovers it -- from the run's own config and from the keys it logged --
so a test that stops exercising a condition because the fixture changed shape
fails on the discovery assert rather than passing vacuously.

Real-run findings this file pins, because they are evidence and not accidents.
Each was confirmed against the raw run, not only against the curated fixture:

  * `mle_only`'s stage names two exit metrics the run never logs. `gates/mle_flat`
    is published to the protocol and not to wandb; `eval/wass_debiased` is
    logged unnamespaced. Most of the local corpus carries the same pair.
  * `tb_ramp`'s anchor gate names an UNNAMESPACED metric that matches five real
    keys. The check names all five and picks none.
  * `tb_ramp`'s ray probe sat at its censoring bound for most of the run on one
    alpha and not on the others -- a real censored estimator, both directions,
    inside one run.
  * `vg_normal` / `vg_blowup` name the same unnamespaced gate metric, and on the
    VarGrad route it is NA_ROUTE rather than a flag.
  * `buildout` is the only fixture whose protocol published live bars.

Run: python -m pytest analysis/tests -q
"""

import numpy as np
import pytest

from analysis import keys as K
from analysis.checks import State, check_r14, context
from analysis.features import theil_sen
from analysis.tests import fixtures


# ---------------------------------------------------------------------------
# Helpers -- discovery, never hardcoding
# ---------------------------------------------------------------------------

def _sensor_rows(res):
    """Rows about a sensor. R13 pairs are the other row family."""
    return [r for r in res.rows if ' vs ' not in r.subject]


def _bar_rows(res):
    return [r for r in res.rows if ' vs ' in r.subject]


def _row(res, metric):
    """The single row for `metric`, or a loud failure.

    Deliberately not a filter that shrugs on an empty list: a check that quietly
    stopped emitting a subject would otherwise turn every assertion about it
    into a vacuous pass."""
    hits = [r for r in _sensor_rows(res) if r.subject.endswith(f'={metric}')]
    assert len(hits) == 1, (
        f'{metric!r}: expected exactly one sensor row, got {len(hits)}. '
        f'rows={[r.subject for r in res.rows]}')
    return hits[0]


def _bar_row(res, bar_key):
    hits = [r for r in _bar_rows(res) if r.subject.startswith(f'{bar_key} vs ')]
    assert len(hits) == 1, (
        f'{bar_key!r}: expected exactly one bar row, got {len(hits)}. '
        f'rows={[r.subject for r in res.rows]}')
    return hits[0]


def _tag(row):
    """The condition tag -- the first token of `detail`, before the em-dash."""
    return row.detail.split(' -- ')[0].split(' | ')[0]


def _stage_metric(run, template, *args):
    idx = context(run).stage_index
    assert idx is not None, f'{run.name}: no stage index, fixture changed shape'
    return K._value(run.config, template % ((idx,) + args))


def _censored_keys(run):
    """Series this codebase clamps before logging, as the check finds them."""
    return sorted(k for k in run.available_keys()
                  if k.startswith(tuple(K.CENSORED)))


def _bar_keys(run):
    prefix = K.EXIT_THRESHOLD_TRACE % ''
    return sorted(k for k in run.available_keys() if k.startswith(prefix))


def _hist(run, key):
    s, v = run.history[key]
    return np.asarray(s, float), np.asarray(v, float)


# ---------------------------------------------------------------------------
# The check runs, on every real run, and never returns an empty pass
# ---------------------------------------------------------------------------

def test_runs_on_every_real_run_without_crashing(all_runs):
    for name, run in all_runs.items():
        res = check_r14(run)
        assert res.check, name


def test_a_result_is_never_silently_empty(all_runs):
    """`ran` with no rows renders identically to 'looked, found nothing', which
    is the failure mode this package exists to prevent."""
    for name, run in all_runs.items():
        res = check_r14(run)
        if res.ran:
            assert res.rows, f'{name}: ran with an empty table'
        else:
            assert res.reason, f'{name}: did not run and did not say why'


def test_no_subjects_at_all_is_not_run_not_a_clean_bill(mle_only):
    """A run whose config names no sensor must say NOTHING WAS ASSERTED."""
    real = check_r14(mle_only)
    assert real.ran and real.rows

    blanked = {K.CFG_ANCHOR_GATE_CEILING_METRIC: None}
    idx = context(mle_only).stage_index
    for j in range(8):
        key = K.CFG_STAGE_EXIT_METRIC % (idx, j)
        if key in mle_only.config:
            blanked[key] = None
    assert len(blanked) > 1, 'fixture no longer names any sensor to blank'

    res = check_r14(fixtures.mutate(mle_only, config=blanked))
    assert not res.ran
    assert 'asserted' in res.reason


# ---------------------------------------------------------------------------
# Scope: only what a controller READS is a subject
# ---------------------------------------------------------------------------

def test_subjects_are_named_by_the_config_not_found_by_flatness(tb_ramp):
    """A constant SET POINT is config being echoed, not a reading. The subject
    list comes from the sensor-naming config keys, so a flat series nothing
    reads never becomes a row."""
    res = check_r14(tb_ramp)
    named = {v for v in (K._value(tb_ramp.config, k) for k in tb_ramp.config)
             if isinstance(v, str)}
    for row in _sensor_rows(res):
        metric = row.subject.split('=', 1)[1]
        assert metric in named or metric.startswith(tuple(K.CENSORED)), \
            f'{metric} is a subject but nothing in the config names it'


def test_flat_series_nobody_reads_are_not_subjects(tb_ramp):
    """The complement of the above, and the one that would catch a rewrite into
    'every flat series in the run'."""
    subjects = {r.subject.split('=', 1)[1] for r in _sensor_rows(check_r14(tb_ramp))}
    flat = {k for k, (s, v) in tb_ramp.history.items()
            if len(v) > 50 and np.nanmin(v) == np.nanmax(v)}
    assert flat, 'fixture carries no flat series -- the test proves nothing'
    assert flat - subjects, 'every flat series became a subject'


# ---------------------------------------------------------------------------
# Condition 1 -- the controller reads a series the run does not log
# ---------------------------------------------------------------------------

def test_real_exit_metric_that_is_never_logged_fires(mle_only):
    """`gates/mle_flat` is published to the protocol and never to wandb. The
    stage's third exit metric IS logged, and must not fire -- both directions
    inside one real run."""
    res = check_r14(mle_only)
    absent = _stage_metric(mle_only, K.CFG_STAGE_EXIT_METRIC, 0)
    present = _stage_metric(mle_only, K.CFG_STAGE_EXIT_METRIC, 2)
    assert absent and present and absent != present

    row = _row(res, absent)
    assert row.state is State.FLAG and row.is_finding
    assert _tag(row) == 'NOT LOGGED'
    assert 'not logged by this run' in row.detail

    assert _row(res, present).state is State.OK


def test_dropping_a_live_sensor_series_makes_it_fire(tb_ramp):
    """The mutation direction of the same condition, on a sensor that is LIVE
    on the real run. Both the `_wcen` form and the plain form go, or the
    rename resolver substitutes one for the other and the hole closes."""
    metric = _stage_metric(tb_ramp, K.CFG_STAGE_BALANCE_METRIC, 'bwd')
    assert metric, 'fixture no longer names a bwd balance metric'
    assert _row(check_r14(tb_ramp), metric).state is State.OK

    family = [k for k in tb_ramp.available_keys()
              if k.rsplit('/', 1)[-1].startswith(metric.rsplit('/', 1)[-1][:14])]
    row = _row(check_r14(fixtures.mutate(tb_ramp, drop=family)), metric)
    assert row.state is State.FLAG and _tag(row) == 'NOT LOGGED'


def test_dropping_only_the_suffixed_form_reports_a_RENAME_not_a_hole(tb_ramp):
    """The `_wcen` form divides out a bias the plain form does not. Substituting
    one for the other is a real substitution and the row has to say so."""
    metric = _stage_metric(tb_ramp, K.CFG_STAGE_BALANCE_METRIC, 'bwd')
    mutated = fixtures.mutate(tb_ramp, drop=[metric])
    row = _row(check_r14(mutated), metric)
    assert row.state is State.OK
    assert 'read as ' in row.detail and metric not in row.detail.split('read as ')[1]


def test_a_sensor_that_resolves_but_carries_no_number_is_unreadable(tb_ramp):
    """`resolve` matches against summary keys too, and a summary entry can be a
    STRING. Reporting that as OK would be reporting a sensor as live on the
    strength of its name."""
    metric = _stage_metric(tb_ramp, K.CFG_STAGE_BALANCE_METRIC, 'bwd')
    gone = fixtures.mutate(tb_ramp, drop=[metric])
    named = fixtures.mutate(gone, summary={metric: 'not a number'})

    row = _row(check_r14(named), metric)
    assert row.state is State.UNREADABLE
    assert 'no numeric series' in row.detail


def test_unnamespaced_gate_metric_names_every_candidate_and_picks_none(tb_ramp):
    """`tb_resid_clipped` is logged under several namespaces that are DIFFERENT
    QUANTITIES. Choosing one would be a guess."""
    metric = K._value(tb_ramp.config, K.CFG_ANCHOR_GATE_CEILING_METRIC)
    assert metric and '/' not in metric, 'fixture gate metric is now namespaced'
    row = _row(check_r14(tb_ramp), metric)
    assert row.state is State.FLAG
    assert _tag(row) == 'AMBIGUOUS NAME' and 'ambiguous' in row.detail
    for cand in (k for k in tb_ramp.available_keys()
                 if k.rsplit('/', 1)[-1] == metric):
        assert cand in row.detail


# ---------------------------------------------------------------------------
# NA_ROUTE is never a flag, and never ABSENT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['vg_normal', 'vg_blowup'])
def test_an_ambiguous_gate_metric_is_ambiguous_on_every_route(all_runs, name):
    """PRESENCE IS ROUTE-BLIND; MEANING IS NOT.

    `tb_resid_clipped` is not logged under that name at all -- five DIFFERENT
    quantities carry it as a tail. Resolving it route-first on a VarGrad run
    returned NA_ROUTE, whose meaning is "logged, populated, and not this route's
    to read", about a key that does not exist. The row asserted a falsehood and
    swallowed every dead-sensor condition behind it."""
    run = all_runs[name]
    assert context(run).route is K.Route.VARGRAD_CONDITIONAL
    metric = K._value(run.config, K.CFG_ANCHOR_GATE_CEILING_METRIC)
    assert metric not in run.available_keys(), 'fixture now logs it directly'
    row = _row(check_r14(run), metric)
    assert row.state is State.FLAG
    assert _tag(row) == 'AMBIGUOUS NAME'


def test_a_sensor_that_is_na_on_this_route_is_na_and_not_a_flag(vg_normal):
    """The genuine NA_ROUTE case, which needs a sensor that IS logged and whose
    quantity does not track here. No real VarGrad run points a controller at
    one, so the config is mutated to do it -- the mutation is in the CONFIG, so
    the series itself is untouched real data."""
    na_metric = K.TOPLINE_TB[0]                      # fwd/tb_err_worst
    assert na_metric in vg_normal.available_keys()
    m = fixtures.mutate(vg_normal, config={
        K.CFG_STAGE_BALANCE_METRIC % (context(vg_normal).stage_index, K.MODES[0]):
            na_metric})
    row = _row(check_r14(m), na_metric)
    assert row.state is State.NA_ROUTE
    assert not row.is_finding
    assert row.numbers == {}, 'NA_ROUTE must not carry a number to be read'


def test_the_same_sensor_on_a_tb_run_is_measured(tb_ramp):
    """The pair that proves NA_ROUTE is about the ROUTE and not about the key.
    Without it the test above passes on a check that calls everything NA."""
    na_metric = K.TOPLINE_TB[0]
    m = fixtures.mutate(tb_ramp, config={
        K.CFG_STAGE_BALANCE_METRIC % (context(tb_ramp).stage_index, K.MODES[0]):
            na_metric})
    row = _row(check_r14(m), na_metric)
    assert row.state is not State.NA_ROUTE
    assert row.numbers, 'a measured sensor must carry its numbers'


# ---------------------------------------------------------------------------
# Condition 2 -- all-NaN
# ---------------------------------------------------------------------------

def test_nan_series_fires(tb_ramp):
    metric = _stage_metric(tb_ramp, K.CFG_STAGE_BALANCE_METRIC, 'bwd')
    assert _row(check_r14(tb_ramp), metric).state is State.OK

    s, v = _hist(tb_ramp, metric)
    mutated = fixtures.mutate(tb_ramp, history={metric: (s, np.full_like(v, np.nan))},
                              summary={metric: float('nan')})
    row = _row(check_r14(mutated), metric)
    assert row.state is State.FLAG and _tag(row) == 'NO FINITE VALUES'
    assert row.numbers['n_ticks'] == 0 and row.numbers['n_logged'] == len(v)


# ---------------------------------------------------------------------------
# Condition 3 -- zero variance over the window
# ---------------------------------------------------------------------------

def test_pinned_sensor_fires_and_the_real_one_does_not(tb_ramp):
    """`fixtures.pin` is the canonical R14 mutation: a real sensor replaced by
    a constant on its own step grid."""
    metric = _stage_metric(tb_ramp, K.CFG_STAGE_BALANCE_METRIC, 'bwd')
    real = _row(check_r14(tb_ramp), metric)
    assert real.state is State.OK and real.numbers['spread'] > 0

    row = _row(check_r14(fixtures.pin(tb_ramp, metric)), metric)
    assert row.state is State.FLAG and row.is_finding
    assert _tag(row) == 'ZERO SPREAD'
    assert row.numbers['spread'] == 0
    assert row.numbers['n_ticks'] == real.numbers['n_ticks']


def test_every_config_named_sensor_can_be_killed_and_is_noticed(tb_ramp):
    """Pin them ONE AT A TIME. A check that only ever fires for the first
    subject in its table would pass every test above."""
    res = check_r14(tb_ramp)
    live = [r.subject.split('=', 1)[1] for r in _sensor_rows(res)
            if r.state is State.OK and 'censored' not in r.subject]
    assert len(live) >= 4, f'fixture no longer carries live sensors: {live}'
    for metric in live:
        row = _row(check_r14(fixtures.pin(tb_ramp, metric)), metric)
        assert row.state is State.FLAG, f'{metric} pinned and not noticed'
        assert _tag(row) == 'ZERO SPREAD', metric


def test_a_short_series_is_not_reported_as_a_dead_sensor(ring_probe):
    """Do not flag a series for being short -- say how many ticks there were.
    `ring_probe` carries one point per sensor, which is the real case."""
    metric = _stage_metric(ring_probe, K.CFG_STAGE_BALANCE_METRIC, 'bwd')
    row = _row(check_r14(ring_probe), metric)
    assert row.state is State.UNREADABLE
    assert row.state is not State.FLAG
    assert _tag(row) == 'TOO FEW TICKS'
    assert row.numbers['n_ticks'] == 1
    assert 'nothing is asserted' in row.detail


# ---------------------------------------------------------------------------
# Condition 4 -- pinned at an extremum for more than the threshold fraction
# ---------------------------------------------------------------------------

def test_sensor_held_at_its_maximum_fires_while_still_moving(tb_ramp):
    """Not zero spread: the series still moves on a minority of ticks, and a
    check that only caught constants would miss a clipped actuator."""
    metric = _stage_metric(tb_ramp, K.CFG_STAGE_BALANCE_METRIC, 'bwd')
    s, v = _hist(tb_ramp, metric)
    held = v.copy()
    held[: int(len(held) * 0.8)] = float(np.nanmax(v))

    row = _row(check_r14(fixtures.mutate(tb_ramp, history={metric: (s, held)})),
               metric)
    assert row.state is State.FLAG and _tag(row) == 'PINNED AT EXTREMUM'
    assert row.numbers['spread'] > 0, 'this is the not-constant case'
    assert row.numbers['frac_at_max'] > 0.5


def test_sensor_pinned_at_a_config_clip_names_the_clip(tb_ramp):
    """'a value bound at its clip', in its literal form: the clip is a config
    number, and the row has to name which knob it is."""
    clip_key = K.CFG_CLIP_KEYS[0]
    clip = float(K._value(tb_ramp.config, clip_key))
    metric = _stage_metric(tb_ramp, K.CFG_STAGE_BALANCE_METRIC, 'bwd')

    row = _row(check_r14(fixtures.pin(tb_ramp, metric, clip)), metric)
    assert row.state is State.FLAG
    assert _tag(row) == 'AT CONFIG CLIP', 'clip must outrank plain flatness'
    assert row.numbers['clip_key'] == clip_key
    assert row.numbers['frac_at_clip'] == 1.0


# ---------------------------------------------------------------------------
# Condition 5 -- a censored estimator at its censoring bound
# ---------------------------------------------------------------------------

def test_real_ray_probe_at_its_bound_fires_and_its_siblings_do_not(tb_ramp):
    """One run, both directions: `ray_calibration` clamps every t-statistic to
    the same bound, and only some of them sit on it."""
    keys = _censored_keys(tb_ramp)
    assert keys, 'fixture no longer carries a censored estimator'
    res = check_r14(tb_ramp)
    fired = [k for k in keys if _tag(_row(res, k)) == 'CENSORED']
    clean = [k for k in keys if _row(res, k).state is State.OK]
    assert fired, 'no real censored series sits at its bound'
    assert clean, 'every censored series fires -- the threshold is doing nothing'
    row = _row(res, fired[0])
    assert row.state is State.FLAG
    assert row.numbers['frac_at_bound'] > 0.5
    assert row.numbers['censor_bound'] == K.CENSORED[
        next(p for p in K.CENSORED if fired[0].startswith(p))]


def test_a_clean_censored_series_driven_to_the_bound_fires(tb_ramp):
    keys = _censored_keys(tb_ramp)
    res = check_r14(tb_ramp)
    clean = [k for k in keys if _row(res, k).state is State.OK]
    assert clean
    key = clean[0]
    bound = K.CENSORED[next(p for p in K.CENSORED if key.startswith(p))]

    row = _row(check_r14(fixtures.pin(tb_ramp, key, bound)), key)
    assert row.state is State.FLAG and _tag(row) == 'CENSORED'


def test_censoring_outranks_flatness(tb_ramp):
    """A series held at its bound is zero-spread too. Reporting the flatness
    loses the fact that explains it: the values above the bound were clamped
    before logging and were never in the record."""
    keys = _censored_keys(tb_ramp)
    key = max(keys, key=lambda k: _row(check_r14(tb_ramp), k)
              .numbers.get('frac_at_bound', 0.0))
    bound = K.CENSORED[next(p for p in K.CENSORED if key.startswith(p))]

    row = _row(check_r14(fixtures.pin(tb_ramp, key, bound)), key)
    assert _tag(row) == 'CENSORED'
    assert _tag(row) != 'ZERO SPREAD'
    assert row.numbers['spread'] == 0, 'it IS flat; it is reported as censored'


# ---------------------------------------------------------------------------
# H5 -- an EMA sensor still fires, and the row says it is an EMA
# ---------------------------------------------------------------------------

def test_ema_sensor_is_marked_as_one(tb_ramp):
    """Smoothing manufactures autocorrelation and lowers variance, so an EMA
    series can read as pinned. The row says so; it does not suppress."""
    ema = K.LOW_TRUST[0]
    assert K.is_ema(ema) and ema in tb_ramp.history
    idx = context(tb_ramp).stage_index
    named = fixtures.mutate(
        tb_ramp, config={K.CFG_STAGE_BALANCE_METRIC % (idx, 'fwd'): ema})

    row = _row(check_r14(named), ema)
    assert 'EMA output' in row.detail
    assert 'low-trust' in row.detail


def test_a_pinned_ema_sensor_still_fires(tb_ramp):
    ema = K.LOW_TRUST[0]
    idx = context(tb_ramp).stage_index
    named = fixtures.mutate(
        tb_ramp, config={K.CFG_STAGE_BALANCE_METRIC % (idx, 'fwd'): ema})
    row = _row(check_r14(fixtures.pin(named, ema)), ema)
    assert row.state is State.FLAG and _tag(row) == 'ZERO SPREAD'
    assert 'EMA output' in row.detail


# ---------------------------------------------------------------------------
# R13 -- a threshold annealed below the metric's own measured noise floor
# ---------------------------------------------------------------------------

def test_only_buildout_publishes_bars(all_runs):
    """Pinned because it is the reason R13's mutation tests live on one fixture,
    and because a capture that silently stopped carrying `protocol/thr_*` would
    make every R13 test below vacuous."""
    with_bars = {n for n, r in all_runs.items() if _bar_keys(r)}
    assert with_bars == {'buildout'}, with_bars


def test_real_bars_are_compared_against_the_span_they_cover(buildout):
    res = check_r14(buildout)
    rows = _bar_rows(res)
    assert len(rows) == len(_bar_keys(buildout))
    for row in rows:
        assert row.numbers['n_metric_in_span'] > 0
        assert 'sigma' in row.numbers and 'bar_last' in row.numbers


def test_a_bar_pushed_below_the_metric_sigma_fires(buildout):
    """Both directions on one real pair: the shipped bar sits above the metric's
    detrended sigma, and the same bar annealed under it fires."""
    res = check_r14(buildout)
    above = [r for r in _bar_rows(res) if r.state is State.OK]
    assert above, 'no real bar sits above its metric sigma'
    row = above[0]
    bar_key, sigma = row.subject.split(' vs ')[0], row.numbers['sigma']
    assert sigma > 0

    fired = _bar_row(check_r14(fixtures.pin(buildout, bar_key, sigma * 0.5)),
                     bar_key)
    assert fired.state is State.FLAG and fired.is_finding
    assert _tag(fired) == 'BAR BELOW SIGMA'
    assert fired.numbers['bar_last'] < fired.numbers['sigma']


def test_a_bar_raised_above_the_metric_sigma_stops_firing(buildout):
    """The converse mutation, on a bar that fires on the REAL run -- so the
    firing side is not an artefact of the mutation itself."""
    res = check_r14(buildout)
    below = [r for r in _bar_rows(res) if _tag(r) == 'BAR BELOW SIGMA']
    assert below, 'no real bar sits below its metric sigma'
    row = below[0]
    bar_key, sigma = row.subject.split(' vs ')[0], row.numbers['sigma']

    lifted = _bar_row(check_r14(fixtures.pin(buildout, bar_key, sigma * 2.0)),
                      bar_key)
    assert lifted.state is State.OK


def test_a_bar_whose_metric_is_not_logged_is_unreadable_not_a_pass(buildout):
    res = check_r14(buildout)
    row = _bar_rows(res)[0]
    bar_key, metric = row.subject.split(' vs ')

    orphan = check_r14(fixtures.mutate(buildout, drop=[metric]))
    hits = [r for r in _bar_rows(orphan) if r.subject.startswith(f'{bar_key} vs')]
    assert len(hits) == 1
    assert hits[0].state is State.UNREADABLE
    assert _tag(hits[0]) == 'BAR WITHOUT A METRIC'


def test_a_bar_matching_two_logged_metrics_names_both_and_picks_neither(buildout):
    """`a/b_c` and `a_b/c` tag identically. Choosing between them would be the
    same guess `keys.resolve` refuses to make for an unnamespaced name."""
    row = _bar_rows(check_r14(buildout))[0]
    bar_key, metric = row.subject.split(' vs ')
    tag = K.metric_tag(metric)
    cut = tag.index('_', len(metric.split('/')[0]) + 1)
    twin = f'{tag[:cut]}/{tag[cut + 1:]}'
    assert K.metric_tag(twin) == tag and twin != metric

    res = check_r14(fixtures.mutate(buildout,
                                    history={twin: _hist(buildout, metric)}))
    hits = [r for r in _bar_rows(res) if r.subject.startswith(f'{bar_key} vs')]
    assert len(hits) == 1 and hits[0].state is State.UNREADABLE
    assert _tag(hits[0]) == 'BAR WITHOUT A METRIC'
    assert metric in hits[0].detail and twin in hits[0].detail


def test_a_bar_with_too_few_ticks_is_unreadable_not_a_pass(buildout):
    row = _bar_rows(check_r14(buildout))[0]
    bar_key = row.subject.split(' vs ')[0]
    s, v = _hist(buildout, bar_key)

    res = check_r14(fixtures.mutate(buildout,
                                    history={bar_key: (s[:3], v[:3])}))
    hit = _bar_row(res, bar_key)
    assert hit.state is State.UNREADABLE and _tag(hit) == 'TOO FEW TICKS'
    assert hit.numbers['n_bar'] == 3


def test_r13_sigma_is_measured_over_the_span_the_bar_covers(buildout):
    """A bar that switched on late, compared against sigma from the whole run,
    is compared against a regime it was never in. On real data the restriction
    flips the answer, so it is pinned rather than left as a comment."""
    res = check_r14(buildout)
    flipped = []
    for row in _bar_rows(res):
        metric = row.subject.split(' vs ')[1]
        s, v = _hist(buildout, metric)
        whole = float(np.std(theil_sen(s, v)[1]))
        assert row.numbers['n_metric_in_span'] <= len(s)
        bar = row.numbers['bar_last']
        if (bar < row.numbers['sigma']) != (bar < whole):
            flipped.append((row.subject, row.state))
    assert flipped, ('no bar where restricting to its own span changes the '
                     'answer -- this fixture can no longer prove the point')


def test_r13_carries_a_robust_sigma_beside_the_std(buildout):
    """The std is the stated comparison and one excursion moves it a long way,
    so the scale that does not travels beside it. The reader has to be able to
    disagree with the row."""
    for row in _bar_rows(check_r14(buildout)):
        assert row.numbers['sigma_robust'] >= 0
        if (row.numbers['bar_last'] < row.numbers['sigma']) != \
                (row.numbers['bar_last'] < row.numbers['sigma_robust']):
            assert 'straddle' in row.detail


# ---------------------------------------------------------------------------
# The trailing window
# ---------------------------------------------------------------------------

def test_the_window_decides_what_pinned_means(tb_ramp):
    """A sensor that died recently is live over the whole history and dead over
    the trailing window. Reading it without a window would miss it."""
    metric = _stage_metric(tb_ramp, K.CFG_STAGE_BALANCE_METRIC, 'bwd')
    s, v = _hist(tb_ramp, metric)
    window = 2000.0
    tail = s >= s[-1] - window
    assert 8 < tail.sum() < len(s), 'window no longer splits this series'

    dead = v.copy()
    dead[tail] = float(v[-1])
    mutated = fixtures.mutate(tb_ramp, history={metric: (s, dead)})

    assert _row(check_r14(mutated), metric).state is State.OK
    row = _row(check_r14(mutated, window=window), metric)
    assert row.state is State.FLAG and _tag(row) == 'ZERO SPREAD'
    assert row.numbers['n_ticks'] == int(tail.sum())
    assert 'trailing' in row.numbers['window']


def test_a_dead_gated_metric_does_not_make_r13_quieter(buildout):
    """KILLING A SENSOR MUST NOT TURN A FLAG INTO OK.

    `bar < sigma` is false for every bar when sigma is 0, so a gated metric that
    is itself pinned used to read as a healthy bar. That is the swallowed
    diagnostic failing as REASSURANCE rather than as silence, and it is the
    worst shape a check can have: the run gets worse and the report gets
    calmer."""
    bars = [r for r in check_r14(buildout).rows if ' vs ' in r.subject]
    assert bars, 'fixture no longer publishes a bar'
    live = next(r for r in bars if r.state is State.OK)
    prefix, metric = live.subject.split(' vs ')

    killed = fixtures.pin(buildout, metric)
    row = next(r for r in check_r14(killed).rows
               if r.subject.startswith(prefix))
    assert row.state is State.FLAG
    assert _tag(row) == 'NO MEASURABLE FLOOR'
    assert row.is_finding
