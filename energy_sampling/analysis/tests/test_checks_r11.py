"""
Tests for R11 -- `replay/scatter_err` over `fwd/scatter_err`.

MUTATION-TESTED THROUGHOUT. Every condition gets two tests: the real run that
does NOT show it, and a real run with it re-introduced, which must FIRE. A check
that has never fired has not been tested, and this repo has shipped tests that
passed while blind more than once.

THE ROUTE GATE IS THE SUBTLE PART, AND IT IS NOT A PROPERTY OF THE KEYS.
`keys.resolve` marks NEITHER scatter_err series NA on ANY route -- asserted
below, so this file fails loudly if that ever changes -- which means an
implementation that asked the KEY whether R11 applied would hand back a ratio on
a conditional VarGrad run and invite it to be read as a TB number. Two mutations
pin the gate to the ROUTE instead:

  * a VarGrad run GIVEN a full `replay/scatter_err` series stays NA_ROUTE, so
    presence cannot buy a number;
  * the exact series that produce a ratio on `tb_ramp` produce NA_ROUTE when
    they are read against a VarGrad run's config, so the config alone decides.

NA_ROUTE is not `not_run` and not ABSENT: the check RAN, and its subject does
not apply here. An UNKNOWN route is the opposite -- applicability is
undetermined, which is a hole in the report rather than a table row.

Real-run facts this file leans on, re-derived from the fixture wherever a
literal would rot:

  * `tb_ramp` logs both series on ONE 1315-point step grid and sits between the
    1x and 2x bands.
  * `buildout` logs them on DIFFERENT grids (637 vs 653 points) and is the real
    evidence that a pointwise median and a ratio of medians are not the same
    number -- they land in different bands on it.
  * `mle_only` logs neither series: the real `not_run` case.
  * `vg_normal` logs `fwd/scatter_err` and never logged a replay branch.

Run: python -m pytest analysis/tests -q
"""

import copy
import re

import numpy as np
import pytest

from analysis import keys as K
from analysis.checks import State, check_r11, format_result
from analysis.tests import fixtures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _only_row(res):
    """R11's single row, or a loud failure.

    Not a filter that shrugs on an empty list: a check that quietly stopped
    emitting its row would turn every assertion below into a vacuous pass."""
    assert res.ran, f'check did not run: {res.reason}'
    assert len(res.rows) == 1, f'expected one row, got {len(res.rows)}'
    return res.rows[0]


def _copy(run):
    """A deep copy with nothing changed -- the base for edits `mutate` cannot
    express, namely REMOVING config keys (its `drop` reaches history and
    summary only)."""
    return fixtures.mutate(run)


def _num(run):
    return run.history[K.R11_NUMERATOR]


def _den(run):
    return run.history[K.R11_DENOMINATOR]


def _scaled(run, key, factor):
    """One series multiplied through, steps untouched. The ratio must follow it;
    a check that reports the same band either way is reading a constant."""
    s, v = run.history[key]
    return fixtures.mutate(run, history={key: (s, np.asarray(v, float) * factor)})


def _read_against(config_run, series_run):
    """`series_run`'s REAL history and summary read against `config_run`'s REAL
    config. Nothing is invented -- both halves are captured runs -- and it is the
    only way to hold the data fixed while the route changes."""
    out = copy.deepcopy(config_run)
    out.history = copy.deepcopy(series_run.history)
    out.summary = copy.deepcopy(series_run.summary)
    return out


def _min_aligned(run):
    """The module's own minimum, read out of the refusal it writes.

    Asked rather than imported: the constant is private to the part module, and
    a test that hardcodes 8 goes quietly wrong the day the guard moves."""
    s, v = _num(run)
    res = check_r11(fixtures.mutate(run, history={K.R11_NUMERATOR: (s[:2], v[:2])}))
    assert not res.ran
    m = re.search(r'(\d+) needed', res.reason)
    assert m, f'no minimum stated in the refusal: {res.reason}'
    return int(m.group(1))


# Words that would make a row a verdict rather than a reading. 'fine' is
# deliberately absent: its only occurrence in this check is the refusal's own
# denial ('this is not "replay is fine"'), and banning a disclaimer is not the
# point of the rule.
_VERDICT_WORDS = ('healthy', 'unhealthy', 'overfit', 'is working', 'is broken',
                  'suggests', 'indicates', 'therefore', 'you should',
                  'looks good', 'looks bad', 'converged')


# ---------------------------------------------------------------------------
# The route gate
# ---------------------------------------------------------------------------

def test_neither_series_is_marked_na_by_key_resolution():
    """THE PREMISE OF THE GATE. `keys.resolve` has no NA pattern covering either
    scatter_err series on any route -- its patterns cover log Z and the TB
    residuals -- so an implementation that asked the key whether R11 applied
    would be told LIVE on a VarGrad run and would hand back a number."""
    for route in K.Route:
        for key in (K.R11_NUMERATOR, K.R11_DENOMINATOR):
            res, = K.resolve({key}, [key], route)
            assert res.state is not K.KeyState.NA_ROUTE, (route, key)


def test_vargrad_run_is_na_route_with_no_number(vg_normal):
    row = _only_row(check_r11(vg_normal))
    assert row.state is State.NA_ROUTE
    assert 'median_ratio' not in row.numbers
    assert K.Route.VARGRAD_CONDITIONAL.value in row.detail


def test_the_vargrad_sibling_arm_agrees(vg_blowup):
    assert _only_row(check_r11(vg_blowup)).state is State.NA_ROUTE


def test_na_route_survives_a_full_replay_series(vg_normal):
    """THE MUTATION THAT SEPARATES ROUTE FROM KEY. `vg_normal` never ran a
    replay branch, so a presence-first implementation reaches NA_ROUTE by
    accident. Give it a populated `replay/scatter_err` -- built from its own
    forward series, so the numbers are its own -- and both keys resolve LIVE.
    A check gated on the key hands back a ratio here; this one must not."""
    ds, dv = _den(vg_normal)
    m = fixtures.mutate(vg_normal,
                        history={K.R11_NUMERATOR: (ds, np.asarray(dv, float) * 3.0)},
                        summary={K.R11_NUMERATOR: float(dv[-1]) * 3.0})
    for key in (K.R11_NUMERATOR, K.R11_DENOMINATOR):
        res, = K.resolve(set(m.history), [key], K.Route.VARGRAD_CONDITIONAL)
        assert res.state is K.KeyState.LIVE
    row = _only_row(check_r11(m))
    assert row.state is State.NA_ROUTE
    assert 'median_ratio' not in row.numbers


def test_the_same_series_read_on_a_vargrad_config_go_na_route(tb_ramp, vg_normal):
    """The data is held FIXED and only the route moves: `tb_ramp`'s two real
    series report a ratio against its own config and NA_ROUTE against
    `vg_normal`'s. Nothing about the keys changed between the two calls."""
    assert _only_row(check_r11(tb_ramp)).numbers['median_ratio'] > 0
    row = _only_row(check_r11(_read_against(vg_normal, tb_ramp)))
    assert row.state is State.NA_ROUTE
    assert not row.numbers


def test_na_route_is_not_absent_and_not_zero(vg_normal):
    """The three states must stay three. A VarGrad run whose replay series is
    genuinely missing still reports NA_ROUTE, never the `not_run` that a missing
    series earns on a TB route."""
    assert K.R11_NUMERATOR not in vg_normal.history
    res = check_r11(vg_normal)
    assert res.ran and not res.reason
    row = _only_row(res)
    assert row.state is not State.UNREADABLE
    assert row.numbers.get('median_ratio') is None


def test_na_route_stays_in_the_table_and_renders_as_itself(vg_normal):
    """It is not a finding -- nothing is wrong -- but it must be visible as its
    own state, because a report that shows it as nothing shows a VarGrad run
    exactly as it shows a clean TB run."""
    res = check_r11(vg_normal)
    assert res.rows and not res.findings
    assert 'NA_ROUTE' in format_result(res, verbose=True)


def test_unknown_route_is_not_run_rather_than_na_route(tb_ramp):
    """A route that could not be classified leaves applicability UNDETERMINED.
    NA_ROUTE would assert the question does not apply; `not_run` says nobody
    knows, which is a hole in the report and renders loudly."""
    m = _copy(tb_ramp)
    m.config.clear()          # a run whose config did not come through the pull
    res = check_r11(m)
    assert not res.ran
    assert res.reason and K.Route.UNKNOWN.value in res.reason
    assert not res.rows


def test_the_mle_route_is_in_scope_not_na(mle_only):
    """`mle_only` is on the MLE/prior route, which R11 IS defined on. It refuses
    for want of the series, not for want of applicability -- and those are
    different reports."""
    res = check_r11(mle_only)
    assert not res.ran
    assert 'na_route' not in res.reason.lower()
    assert K.Route.MLE_PRIOR.value in res.reason


# ---------------------------------------------------------------------------
# The ratio
# ---------------------------------------------------------------------------

def test_tb_ramp_reports_the_real_ratio(tb_ramp):
    """The unmutated TB run: a live row, the doc's ~2x neighbourhood, and every
    input behind it. Half of every mutation test below."""
    row = _only_row(check_r11(tb_ramp))
    assert row.state is State.OK
    s, v = _num(tb_ramp)
    ds, dv = _den(tb_ramp)
    assert row.numbers['n_aligned'] == len(s) == len(ds)
    assert row.numbers['median_ratio'] == pytest.approx(
        float(np.median(np.asarray(v, float) / np.asarray(dv, float))))
    assert 1.0 < row.numbers['median_ratio'] < K.R11_HEALTHY_RATIO
    assert row.numbers['ref_ratio'] == K.R11_HEALTHY_RATIO
    assert 'n_dropped' not in row.numbers      # nothing was thrown away


def test_dividing_the_replay_error_down_fires(tb_ramp):
    """R11's condition, re-introduced: replay error BELOW forward error."""
    assert _only_row(check_r11(tb_ramp)).state is State.OK
    row = _only_row(check_r11(_scaled(tb_ramp, K.R11_NUMERATOR, 0.2)))
    assert row.state is State.FLAG
    assert row.numbers['median_ratio'] < K.R11_OVERFIT_BELOW
    assert f'{K.R11_OVERFIT_BELOW:g}x' in row.detail


def test_multiplying_the_replay_error_up_does_not_fire(tb_ramp):
    """The other half, so the FLAG above is not a check that always fires."""
    row = _only_row(check_r11(_scaled(tb_ramp, K.R11_NUMERATOR, 5.0)))
    assert row.state is State.OK
    assert row.numbers['median_ratio'] > K.R11_HEALTHY_RATIO


def test_raising_the_forward_error_fires_too(tb_ramp):
    """The same condition driven from the DENOMINATOR. A check reading only the
    numerator would pass this while the ratio it claims to report collapsed."""
    row = _only_row(check_r11(_scaled(tb_ramp, K.R11_DENOMINATOR, 5.0)))
    assert row.state is State.FLAG
    assert row.numbers['median_ratio'] < K.R11_OVERFIT_BELOW


def test_the_band_is_exclusive_at_one(tb_ramp):
    """Equal errors are the boundary, not the condition: R11 names BELOW 1x."""
    ns, _ = _num(tb_ramp)
    _, dv = _den(tb_ramp)
    m = fixtures.mutate(tb_ramp,
                        history={K.R11_NUMERATOR: (ns, np.asarray(dv, float))})
    row = _only_row(check_r11(m))
    assert row.numbers['median_ratio'] == pytest.approx(K.R11_OVERFIT_BELOW)
    assert row.state is State.OK


def test_the_window_restricts_what_is_read(tb_ramp):
    ns, _ = _num(tb_ramp)
    expected = int((ns >= max(ns[-1] - 2000, ns[0])).sum())
    row = _only_row(check_r11(tb_ramp, window=2000))
    assert row.numbers['n_aligned'] == expected < len(ns)
    assert '2000' in row.numbers['window']


# ---------------------------------------------------------------------------
# Pointwise, not a ratio of medians
# ---------------------------------------------------------------------------

def test_the_two_disagree_on_a_real_run(buildout):
    """REAL DATA. `buildout`'s series are on different grids and its pointwise
    median and ratio-of-medians land in DIFFERENT BANDS -- one below the 2x
    reference, one at or above it. Whichever the check reports, it is choosing."""
    row = _only_row(check_r11(buildout))
    of_medians = row.numbers['num_median'] / row.numbers['den_median']
    assert row.numbers['median_ratio'] < K.R11_HEALTHY_RATIO <= of_medians
    s, _ = _num(buildout)
    assert row.numbers['n_aligned'] == len(s)          # interpolated onto these
    assert 'between' in row.detail                     # the POINTWISE band


def test_the_band_follows_the_pointwise_median(tb_ramp):
    """Constructed on `tb_ramp`'s real step grid so the two answers straddle the
    1x band: pointwise 0.5x (FLAG) against a ratio-of-medians of 10x (which
    would read as at-or-above the reference). One excursion pattern in either
    series moves the ratio of medians; the median of ratios is what R11 means."""
    ns, _ = _num(tb_ramp)
    n = len(ns)
    a = np.tile([1.0, 100.0, 1000.0], n // 3 + 1)[:n]
    b = np.tile([2.0, 1000.0, 10.0], n // 3 + 1)[:n]
    assert np.median(a) / np.median(b) > K.R11_HEALTHY_RATIO
    assert np.median(a / b) < K.R11_OVERFIT_BELOW
    row = _only_row(check_r11(fixtures.mutate(
        tb_ramp, history={K.R11_NUMERATOR: (ns, a), K.R11_DENOMINATOR: (ns, b)})))
    assert row.state is State.FLAG
    assert row.numbers['median_ratio'] == pytest.approx(float(np.median(a / b)))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def test_the_numerator_is_clipped_to_the_denominators_span(tb_ramp):
    """`np.interp` CLAMPS outside its range without complaint, which would
    invent forward-error values for steps the forward series never saw and pair
    them with real replay values. Cutting the denominator short must cost those
    ticks, not fabricate them."""
    ns, _ = _num(tb_ramp)
    ds, dv = _den(tb_ramp)
    cut = 600
    m = fixtures.mutate(tb_ramp, history={K.R11_DENOMINATOR: (ds[:cut], dv[:cut])})
    expected = int(((ns >= ds[0]) & (ns <= ds[cut - 1])).sum())
    row = _only_row(check_r11(m))
    assert expected < len(ns)
    assert row.numbers['n_aligned'] == expected


def test_a_zero_denominator_tick_is_dropped_and_counted(tb_ramp):
    """A zero forward error makes an infinite ratio, and one infinity in a
    median is how a single tick decides a band."""
    ds, dv = _den(tb_ramp)
    dv = np.asarray(dv, float).copy()
    dv[:100] = 0.0
    row = _only_row(check_r11(fixtures.mutate(
        tb_ramp, history={K.R11_DENOMINATOR: (ds, dv)})))
    assert row.numbers['n_dropped'] == 100
    assert row.numbers['n_aligned'] == len(ds) - 100
    assert np.isfinite(row.numbers['median_ratio'])


def test_non_finite_ticks_are_dropped_and_counted(tb_ramp):
    ns, nv = _num(tb_ramp)
    nv = np.asarray(nv, float).copy()
    nv[:50] = np.nan
    row = _only_row(check_r11(fixtures.mutate(
        tb_ramp, history={K.R11_NUMERATOR: (ns, nv)})))
    assert row.numbers['n_dropped'] == 50
    assert np.isfinite(row.numbers['median_ratio'])


# ---------------------------------------------------------------------------
# Fail loudly
# ---------------------------------------------------------------------------

def test_a_missing_replay_series_is_not_run_and_names_it(tb_ramp):
    res = check_r11(fixtures.mutate(tb_ramp, drop=(K.R11_NUMERATOR,)))
    assert not res.ran and not res.rows
    assert K.R11_NUMERATOR in res.reason
    assert res.reason.count(K.R11_DENOMINATOR) == 0


def test_a_missing_forward_series_is_not_run_and_names_it(tb_ramp):
    res = check_r11(fixtures.mutate(tb_ramp, drop=(K.R11_DENOMINATOR,)))
    assert not res.ran and not res.rows
    assert K.R11_DENOMINATOR in res.reason


def test_both_series_missing_on_a_real_run_names_both(mle_only):
    """`mle_only` died in phase 1 and logged neither series. The real case."""
    res = check_r11(mle_only)
    assert not res.ran
    assert K.R11_NUMERATOR in res.reason and K.R11_DENOMINATOR in res.reason


def test_too_few_aligned_points_says_how_many_it_had(tb_ramp):
    """A window narrower than the eval cadence leaves one tick, and 'the median
    of one point' is whichever tick the cadence landed on."""
    res = check_r11(tb_ramp, window=5)
    assert not res.ran
    assert '1 aligned point' in res.reason
    assert str(len(_num(tb_ramp)[0])) in res.reason   # what it did have


def test_disjoint_step_ranges_are_not_run_with_nothing_aligned(tb_ramp):
    """Two series that never overlap in step give zero aligned ticks. Reporting
    that as a clean pass would be the worst reading in the file."""
    ns, nv = _num(tb_ramp)
    res = check_r11(fixtures.mutate(
        tb_ramp, history={K.R11_NUMERATOR: (np.asarray(ns, float) + 1e6, nv)}))
    assert not res.ran
    assert '0 aligned point' in res.reason


def test_the_minimum_is_exact(tb_ramp):
    """One point either side of the guard, with the guard read out of the
    module's own refusal rather than copied from it."""
    need = _min_aligned(tb_ramp)
    s, v = _num(tb_ramp)
    assert not check_r11(fixtures.mutate(
        tb_ramp, history={K.R11_NUMERATOR: (s[:need - 1], v[:need - 1])})).ran
    row = _only_row(check_r11(fixtures.mutate(
        tb_ramp, history={K.R11_NUMERATOR: (s[:need], v[:need])})))
    assert row.numbers['n_aligned'] == need


def test_the_summary_fallback_is_not_reported_as_a_ratio(tb_ramp):
    """THE TRAP. `base.series` answers a missing history series from the SUMMARY
    as ONE point at `last_step`. Two of those make a ratio of two final values
    that renders exactly like a median over the run."""
    m = _copy(tb_ramp)
    m.history.pop(K.R11_NUMERATOR)
    m.history.pop(K.R11_DENOMINATOR)
    assert K.R11_NUMERATOR in m.summary and K.R11_DENOMINATOR in m.summary
    res = check_r11(m)
    assert not res.ran
    assert '1 aligned point' in res.reason


# ---------------------------------------------------------------------------
# Corpus invariants
# ---------------------------------------------------------------------------

def test_every_captured_run_reads_without_a_traceback(all_runs):
    for name, run in all_runs.items():
        res = check_r11(run)
        assert res.ran or res.reason, name
        assert len(res.rows) == (1 if res.ran else 0), name


def test_every_captured_run_reads_the_same_with_a_window(all_runs):
    """The window is the only optional argument and it must not change which
    branch a run takes -- only how much of it is read."""
    for name, run in all_runs.items():
        bare, windowed = check_r11(run), check_r11(run, window=100000)
        assert bare.ran == windowed.ran, name
        if bare.ran:
            assert bare.rows[0].state is windowed.rows[0].state, name


def test_no_row_or_reason_reads_as_a_verdict(all_runs):
    """R11 is a mechanism. The check names the band; what the band implies is
    the reader's, and a conclusion here would be the failure the whole package
    exists to stop."""
    texts = []
    for run in all_runs.values():
        for res in (check_r11(run), check_r11(run, window=5),
                    check_r11(_scaled(run, K.R11_NUMERATOR, 0.2))
                    if K.R11_NUMERATOR in run.history else check_r11(run)):
            texts.append(res.reason)
            texts += [r.detail for r in res.rows]
    blob = ' '.join(texts).lower()
    assert blob.strip()
    for word in _VERDICT_WORDS:
        assert word not in blob, word


def test_every_live_row_carries_the_numbers_behind_it(all_runs):
    """A finding without its inputs is an assertion. NA_ROUTE is the one row
    with nothing to show, because its whole content is that there is nothing to
    compute."""
    for name, run in all_runs.items():
        res = check_r11(run)
        if not res.ran:
            continue
        row = res.rows[0]
        if row.state is State.NA_ROUTE:
            continue
        assert row.detail, name
        for field in ('median_ratio', 'num_median', 'den_median', 'n_aligned',
                      'window', 'ref_ratio'):
            assert field in row.numbers, (name, field)
