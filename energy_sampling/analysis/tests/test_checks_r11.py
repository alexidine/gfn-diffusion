"""
Tests for R11 -- the two replay-memorisation sensors.

TWO SENSORS, AND THE WHOLE POINT OF THIS FILE IS THAT ONLY ONE FLAGS.

  * SENSOR B, `K.R11_MEMORISATION_*`, carries a DERIVED bar (lambda*tau = 1) and
    is the flagging subject.
  * SENSOR A, `K.R11_NUMERATOR / K.R11_DENOMINATOR`, is reported and NEVER
    flags, on any value. Flagging it fired on roughly three quarters of the local
    TB-route corpus while the user states most of those runs are not memorising,
    and `module_metrics.md` says the statistic cannot distinguish memorisation
    from a coverage gap. `test_sensor_a_never_flags_on_any_real_run` and
    `test_sensor_a_never_flags_however_far_the_ratio_is_driven` are the
    regression guards; if either starts failing, the old check has come back.

MUTATION-TESTED THROUGHOUT. Every condition gets two tests: the real run that
does NOT show it, and a real run with it re-introduced, which must FIRE. A check
that has never fired has not been tested, and this repo has shipped tests that
passed while blind more than once.

THE ROUTE GATE IS THE SUBTLE PART, AND IT IS NOT A PROPERTY OF THE KEYS.
`keys.resolve` marks NONE of the four series NA on ANY route -- asserted below,
so this file fails loudly if that ever changes -- which means an implementation
that asked the KEY whether R11 applied would hand back a ratio on a conditional
VarGrad run and invite it to be read as a TB number. Two mutations pin the gate
to the ROUTE instead:

  * a VarGrad run GIVEN a full `replay/scatter_err` series stays NA_ROUTE, so
    presence cannot buy a number;
  * the exact series that produce a ratio on `tb_ramp` produce NA_ROUTE when
    they are read against a VarGrad run's config, so the config alone decides.

NA_ROUTE is not `not_run` and not ABSENT: the check RAN, and its subject does
not apply here. An UNKNOWN route is the opposite -- applicability is
undetermined, which is a hole in the report rather than a table row.

Real-run facts this file leans on, re-derived from the fixture wherever a
literal would rot:

  * `tb_ramp` logs BOTH sensors. Sensor B's two series share one 1316-point step
    grid (they are published in one dict); sensor A's share a different
    1315-point one.
  * `buildout` logs sensor A only, on DIFFERENT grids (637 vs 653 points), and is
    the real evidence that a pointwise median and a ratio of medians are not the
    same number -- they land in different bands on it. It is also the real
    B-absent-A-present case, which is a quarter of the local TB corpus.
  * `mle_only` logs neither: the MLE route, so NA_ROUTE before presence matters.
  * `vg_normal` logs sensor B and never logged a replay TB branch.

Run: python -m pytest analysis/tests -q
"""

import copy
import re

import numpy as np
import pytest

from analysis import keys as K
from analysis.checks import State, check_r11, format_result
from analysis.tests import fixtures

# The four series, as (numerator, denominator) pairs, in the order the check
# emits their rows.
_B = (K.R11_MEMORISATION_NUMERATOR, K.R11_MEMORISATION_DENOMINATOR)
_A = (K.R11_NUMERATOR, K.R11_DENOMINATOR)
_ALL = _B + _A


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows(res):
    """R11's two rows, `(sensor_b, sensor_a)`, or a loud failure.

    Not a filter that shrugs on an empty list: a check that quietly stopped
    emitting a row would turn every assertion below into a vacuous pass."""
    assert res.ran, f'check did not run: {res.reason}'
    assert len(res.rows) == 2, f'expected two rows, got {len(res.rows)}'
    b, a = res.rows
    assert b.subject == f'{_B[0]} / {_B[1]}', b.subject
    assert a.subject == f'{_A[0]} / {_A[1]}', a.subject
    return b, a


def _row_b(res):
    return _rows(res)[0]


def _row_a(res):
    return _rows(res)[1]


def _tag(row):
    """The first token of `detail` -- which sensor the row is and what standing
    it has. Asserted as a literal, like R14's condition tags, so a row's identity
    is legible in the rendered table and testable without parsing prose."""
    return row.detail.split(' -- ', 1)[0]


def _copy(run):
    """A deep copy with nothing changed -- the base for edits `mutate` cannot
    express, namely REMOVING config keys (its `drop` reaches history and
    summary only)."""
    return fixtures.mutate(run)


def _scaled(run, key, factor):
    """One series multiplied through, steps untouched. The ratio must follow it;
    a check that reports the same band either way is reading a constant."""
    s, v = run.history[key]
    return fixtures.mutate(run, history={key: (s, np.asarray(v, float) * factor)})


def _flat_b(run, ratio):
    """Sensor B pinned at an exact `ratio`, on `tb_ramp`'s real step grid.

    A CONSTANT numerator over a constant denominator of 1.0, rather than
    `denominator * ratio`, so the band edges can be tested at EXACTLY the bar
    without a float round-trip deciding which side of it the test lands on."""
    s, _ = run.history[_B[0]]
    return fixtures.mutate(run, history={
        _B[0]: (s, np.full(len(s), float(ratio))),
        _B[1]: (s, np.ones(len(s)))})


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

    Asked rather than imported: the constant is private to `checks`, and a test
    that hardcodes 8 goes quietly wrong the day the guard moves. Read off SENSOR
    B's row, which is where the truncation below lands; sensor A stays whole, so
    the check still runs and the reason is a row detail rather than a refusal."""
    s, v = run.history[_B[0]]
    row = _row_b(check_r11(fixtures.mutate(run, history={_B[0]: (s[:2], v[:2])})))
    assert row.state is State.UNREADABLE
    m = re.search(r'(\d+) needed', row.detail)
    assert m, f'no minimum stated in the row: {row.detail}'
    return int(m.group(1))


# Words that would make a row a verdict rather than a reading. 'fine' is
# deliberately absent: its only occurrence in this check is the refusal's own
# denial ('this is not "replay is fine"'), and banning a disclaimer is not the
# point of the rule.
_VERDICT_WORDS = ('healthy', 'unhealthy', 'overfit', 'is working', 'is broken',
                  'suggests', 'indicates', 'therefore', 'you should',
                  'looks good', 'looks bad', 'converged')


# ---------------------------------------------------------------------------
# The route gate -- unchanged behaviour, and it must stay unchanged
# ---------------------------------------------------------------------------

def test_no_sensor_series_is_marked_na_by_key_resolution():
    """THE PREMISE OF THE GATE. `keys.resolve` has no NA pattern covering any of
    the four series on any route -- its patterns cover log Z and the TB
    residuals -- so an implementation that asked the key whether R11 applied
    would be told LIVE on a VarGrad run and would hand back a number."""
    for route in K.Route:
        for key in _ALL:
            res, = K.resolve({key}, [key], route)
            assert res.state is not K.KeyState.NA_ROUTE, (route, key)


def test_vargrad_run_withholds_both_sensors_with_no_numbers(vg_normal):
    b, a = _rows(check_r11(vg_normal))
    for row in (b, a):
        assert row.state is State.NA_ROUTE
        assert not row.numbers
        assert K.Route.VARGRAD_CONDITIONAL.value in row.detail


def test_the_vargrad_sibling_arm_agrees(vg_blowup):
    assert all(r.state is State.NA_ROUTE for r in check_r11(vg_blowup).rows)


def test_na_route_survives_full_series_for_both_sensors(vg_normal):
    """THE MUTATION THAT SEPARATES ROUTE FROM KEY. `vg_normal` already logs
    sensor B and never ran a replay TB branch, so a presence-first
    implementation reaches NA_ROUTE by accident on sensor A. Give it a populated
    `replay/scatter_err` -- built from its own forward series, so the numbers are
    its own -- and all four keys resolve LIVE. A check gated on the key hands
    back two ratios here; this one must hand back none."""
    ds, dv = vg_normal.history[_A[1]]
    m = fixtures.mutate(vg_normal,
                        history={_A[0]: (ds, np.asarray(dv, float) * 3.0)},
                        summary={_A[0]: float(dv[-1]) * 3.0})
    for key in _ALL:
        res, = K.resolve(set(m.history), [key], K.Route.VARGRAD_CONDITIONAL)
        assert res.state is K.KeyState.LIVE, key
    for row in _rows(check_r11(m)):
        assert row.state is State.NA_ROUTE
        assert 'median_ratio' not in row.numbers


def test_the_same_series_read_on_a_vargrad_config_go_na_route(tb_ramp, vg_normal):
    """The data is held FIXED and only the route moves: `tb_ramp`'s four real
    series report two ratios against its own config and NA_ROUTE against
    `vg_normal`'s. Nothing about the keys changed between the two calls."""
    for row in _rows(check_r11(tb_ramp)):
        assert row.numbers['median_ratio'] > 0
    for row in _rows(check_r11(_read_against(vg_normal, tb_ramp))):
        assert row.state is State.NA_ROUTE
        assert not row.numbers


def test_na_route_is_not_absent_and_not_zero(vg_normal):
    """The three states must stay three. A VarGrad run whose sensor A series is
    genuinely missing still reports NA_ROUTE, never the `not_run` that a missing
    series earns on a TB route."""
    assert _A[0] not in vg_normal.history
    res = check_r11(vg_normal)
    assert res.ran and not res.reason
    for row in res.rows:
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


def test_the_mle_route_is_na_not_in_scope(mle_only):
    """R11 is defined on the TB route ONLY, per the spec.

    `K.R11_ROUTES` used to include the prior route while the comment above it and
    the spec both said TB only -- and the check PRINTED that contradiction to the
    reader. The prior route trains no replay TB branch, so a ratio there compares
    a quantity the optimiser is minimising against one nothing is touching."""
    res = check_r11(mle_only)
    assert res.ran, 'NA_ROUTE is an answer, not a refusal'
    assert K.R11_ROUTES == (K.Route.TB_UNCONDITIONAL,)
    for row in _rows(res):
        assert row.state is State.NA_ROUTE
        assert not row.is_finding


# ---------------------------------------------------------------------------
# Sensor B -- the derived bar, and the only thing that flags
# ---------------------------------------------------------------------------

def test_the_bar_and_release_are_the_derived_pair():
    """The bar is lambda*tau = 1, i.e. 1/e, and it is DERIVED -- that is the
    entire reason this sensor and not the scatter ratio carries the flag. If the
    constant ever drifts off 1/e it has stopped being derived and this check has
    lost its warrant."""
    assert K.R11_MEMORISATION_BAR == pytest.approx(1.0 / np.e, abs=5e-4)
    assert 0.0 < K.R11_MEMORISATION_BAR < K.R11_MEMORISATION_RELEASE


def test_tb_ramp_reports_the_real_memorisation_ratio(tb_ramp):
    """The unmutated TB run: a live row above the release, and every input
    behind it. Half of every mutation test below."""
    row = _row_b(check_r11(tb_ramp))
    assert row.state is State.OK
    assert not row.is_finding
    s, v = tb_ramp.history[_B[0]]
    _, dv = tb_ramp.history[_B[1]]
    assert row.numbers['n_aligned'] == len(s)
    assert row.numbers['median_ratio'] == pytest.approx(
        float(np.median(np.asarray(v, float) / np.asarray(dv, float))))
    assert row.numbers['median_ratio'] >= K.R11_MEMORISATION_RELEASE
    assert row.numbers['bar'] == K.R11_MEMORISATION_BAR
    assert row.numbers['release'] == K.R11_MEMORISATION_RELEASE
    assert 'n_dropped' not in row.numbers        # nothing was thrown away


def test_driving_the_resident_residual_down_fires(tb_ramp):
    """R11's condition, re-introduced on the sensor that has a bar: resident
    rows corrected far below the residual they were admitted with."""
    assert _row_b(check_r11(tb_ramp)).state is State.OK
    row = _row_b(check_r11(_scaled(tb_ramp, _B[0], 0.3)))
    assert row.state is State.FLAG
    assert row.is_finding
    assert row.numbers['median_ratio'] < K.R11_MEMORISATION_BAR
    assert f'{K.R11_MEMORISATION_BAR:g}' in row.detail


def test_raising_the_intake_residual_fires_too(tb_ramp):
    """The same condition driven from the DENOMINATOR. A check reading only the
    numerator would pass this while the ratio it claims to report collapsed."""
    row = _row_b(check_r11(_scaled(tb_ramp, _B[1], 5.0)))
    assert row.state is State.FLAG
    assert row.numbers['median_ratio'] < K.R11_MEMORISATION_BAR


def test_leaving_the_ratio_high_does_not_fire(tb_ramp):
    """The other half, so the FLAG above is not a check that always fires."""
    row = _row_b(check_r11(_scaled(tb_ramp, _B[0], 1.1)))
    assert row.state is State.OK
    assert row.numbers['median_ratio'] > K.R11_MEMORISATION_RELEASE


def test_the_bar_is_exclusive(tb_ramp):
    """Sitting exactly ON the derived bar is the boundary, not the condition:
    lambda*tau = 1 exactly is rows corrected exactly as fast as they are
    replaced, which is where the servo starts tightening, not past it."""
    row = _row_b(check_r11(_flat_b(tb_ramp, K.R11_MEMORISATION_BAR)))
    assert row.numbers['median_ratio'] == K.R11_MEMORISATION_BAR
    assert row.state is State.OK


def test_one_ulp_under_the_bar_fires(tb_ramp):
    """The companion to the boundary test, and the one that proves the
    comparison is `<` against the bar rather than against something near it."""
    row = _row_b(check_r11(_flat_b(tb_ramp,
                                   np.nextafter(K.R11_MEMORISATION_BAR, 0.0))))
    assert row.state is State.FLAG


def test_the_hold_band_is_reported_and_is_not_a_finding(tb_ramp):
    """Between the bar and the release is the buffer servo's HOLD band
    (`protocol.py` `_buffer_servo_tick` tightens below the bar and releases above
    the release). It is named in the row and it is not a finding -- measured, it
    is where a small minority of the local TB corpus sits."""
    mid = 0.5 * (K.R11_MEMORISATION_BAR + K.R11_MEMORISATION_RELEASE)
    row = _row_b(check_r11(_flat_b(tb_ramp, mid)))
    assert row.state is State.OK
    assert not row.is_finding
    assert f'{K.R11_MEMORISATION_BAR:g}' in row.detail
    assert f'{K.R11_MEMORISATION_RELEASE:g}' in row.detail


def test_the_three_bands_are_three_different_details(tb_ramp):
    """A band that renders identically to another band is not reported."""
    details = {round(r, 3): _row_b(check_r11(_flat_b(tb_ramp, r))).detail
               for r in (0.2, 0.5, 0.9)}
    assert len(set(details.values())) == 3, details


def test_the_flag_follows_the_pointwise_median_not_the_ratio_of_medians(tb_ramp):
    """Constructed on `tb_ramp`'s real step grid so the two answers straddle the
    bar: pointwise below it (FLAG) against a ratio-of-medians well above the
    release. One excursion pattern in either series moves the ratio of medians;
    the median of ratios is what R11 means."""
    s, _ = tb_ramp.history[_B[0]]
    n = len(s)
    a = np.tile([0.05, 1.0, 10.0], n // 3 + 1)[:n]
    b = np.tile([1.0, 10.0, 1.0], n // 3 + 1)[:n]
    assert np.median(a) / np.median(b) > K.R11_MEMORISATION_RELEASE
    assert np.median(a / b) < K.R11_MEMORISATION_BAR
    row = _row_b(check_r11(fixtures.mutate(
        tb_ramp, history={_B[0]: (s, a), _B[1]: (s, b)})))
    assert row.state is State.FLAG
    assert row.numbers['median_ratio'] == pytest.approx(float(np.median(a / b)))


# ---------------------------------------------------------------------------
# Sensor A -- reported, ambiguity named, NEVER flagged
# ---------------------------------------------------------------------------

def test_sensor_a_reports_the_real_ratio_with_the_reference_beside_it(tb_ramp):
    row = _row_a(check_r11(tb_ramp))
    assert row.state is State.OK
    s, v = tb_ramp.history[_A[0]]
    _, dv = tb_ramp.history[_A[1]]
    assert row.numbers['n_aligned'] == len(s)
    assert row.numbers['median_ratio'] == pytest.approx(
        float(np.median(np.asarray(v, float) / np.asarray(dv, float))))
    assert row.numbers['ref_ratio'] == K.R11_SCATTER_REFERENCE
    assert row.numbers['below_ratio'] == K.R11_SCATTER_BELOW


def test_sensor_a_never_flags_however_far_the_ratio_is_driven(tb_ramp):
    """THE REGRESSION GUARD FOR THE WHOLE REWORK.

    Driving `replay/scatter_err` to a fifth of forward's is exactly the mutation
    the OLD check flagged on, and it is the state 44 of 59 readable TB-route runs
    in the local corpus are already in. It must be reported and must not be a
    finding, in either direction and at any magnitude."""
    for factor in (0.01, 0.2, 0.5, 1.0, 5.0, 100.0):
        row = _row_a(check_r11(_scaled(tb_ramp, _A[0], factor)))
        assert row.state is State.OK, factor
        assert not row.is_finding, factor
        assert 'median_ratio' in row.numbers, factor
    driven = _row_a(check_r11(_scaled(tb_ramp, _A[0], 0.2)))
    assert driven.numbers['median_ratio'] < K.R11_SCATTER_BELOW
    assert 'below' in driven.detail


def test_sensor_a_names_its_own_ambiguity_in_the_row(tb_ramp):
    """`module_metrics.md`: a ratio below 1 is equally the signature of
    memorisation and of a coverage gap, and the statistic does not distinguish
    them. That has to be IN the row -- a number printed without it is the number
    the old check flagged on, one rendering step removed."""
    detail = _row_a(check_r11(_scaled(tb_ramp, _A[0], 0.2))).detail.lower()
    assert 'memoris' in detail
    assert 'coverage gap' in detail
    assert 'not distinguish' in detail


def test_the_scatter_reference_is_printed_as_a_reference_and_never_compared(tb_ramp):
    """The ~2x figure is a stated reference, not a bar. Two runs on opposite
    sides of it must reach the SAME state, or it is a bar under another name."""
    below = _row_a(check_r11(_scaled(tb_ramp, _A[0], 0.5)))
    above = _row_a(check_r11(_scaled(tb_ramp, _A[0], 5.0)))
    assert below.numbers['median_ratio'] < K.R11_SCATTER_REFERENCE
    assert above.numbers['median_ratio'] > K.R11_SCATTER_REFERENCE
    assert below.state is above.state is State.OK
    assert f'{K.R11_SCATTER_REFERENCE:g}x' in below.detail
    assert 'reference, not a bar' in below.detail


def test_the_side_word_splits_at_one_and_not_at_the_reference(tb_ramp):
    """'>1, certainly' is as much as this statistic supports; the ~2x figure is
    a stated reference and NOTHING splits on it -- not the state, and not the
    word the row prints either. `tb_ramp` sits between the two, so a side word
    computed against the reference reads 'below' here and one computed against
    1x reads 'at or above'. The fixture separates them with no construction."""
    row = _row_a(check_r11(tb_ramp))
    assert (K.R11_SCATTER_BELOW < row.numbers['median_ratio']
            < K.R11_SCATTER_REFERENCE)
    assert f'at or above {K.R11_SCATTER_BELOW:g}x' in row.detail


def test_sensor_a_and_sensor_b_disagree_on_a_real_run(tb_ramp):
    """They are different quantities and they are not redundant: on the
    unmutated fixture sensor A sits between 1x and its reference while sensor B
    sits above its release. Two rows, two numbers, and the reader is told which
    one carries the bar."""
    b, a = _rows(check_r11(tb_ramp))
    assert b.numbers['median_ratio'] != a.numbers['median_ratio']
    assert b.numbers['n_aligned'] != a.numbers['n_aligned']   # different grids


def test_the_two_sensor_a_medians_disagree_on_a_real_run(buildout):
    """REAL DATA, and the reason the median is pointwise. `buildout`'s series are
    on different grids and its pointwise median and ratio-of-medians land in
    DIFFERENT BANDS relative to the stated reference. Whichever the check
    reports, it is choosing."""
    row = _row_a(check_r11(buildout))
    of_medians = row.numbers['num_median'] / row.numbers['den_median']
    assert row.numbers['median_ratio'] < K.R11_SCATTER_REFERENCE <= of_medians
    s, _ = buildout.history[_A[0]]
    assert row.numbers['n_aligned'] == len(s)        # interpolated onto these


# ---------------------------------------------------------------------------
# One sensor missing
# ---------------------------------------------------------------------------

def test_sensor_b_absent_is_unreadable_and_sensor_a_is_still_reported(buildout):
    """THE REAL CASE, not a mutation: `buildout` logs sensor A and not sensor B,
    which is a quarter of the local TB corpus (the family was only wired into the
    metric tracker partway through it). The row that carries the bar says it
    could not be read and names the series; the reference row still reports."""
    b, a = _rows(check_r11(buildout))
    assert b.state is State.UNREADABLE and b.is_finding
    for key in _B:
        assert key in b.detail
    assert a.state is State.OK
    assert a.numbers['median_ratio'] > 0


def test_sensor_b_absent_is_not_reported_as_a_reading(tb_ramp):
    """The mutation half of the above, from a run that HAS sensor B. An absent
    memorisation sensor must never be rendered as a number, and must never be
    quietly replaced by sensor A's -- that substitution is how the ambiguous
    statistic ends up in the position of the answer."""
    m = fixtures.mutate(tb_ramp, drop=_B)
    b, a = _rows(check_r11(m))
    assert b.state is State.UNREADABLE
    assert 'median_ratio' not in b.numbers
    assert a.numbers['median_ratio'] == pytest.approx(
        _row_a(check_r11(tb_ramp)).numbers['median_ratio'])


def test_dropping_only_the_memorisation_numerator_names_only_it(tb_ramp):
    b = _row_b(check_r11(fixtures.mutate(tb_ramp, drop=(_B[0],))))
    assert b.state is State.UNREADABLE
    assert _B[0] in b.detail
    assert b.detail.count(_B[1]) == 0


def test_sensor_a_absent_still_leaves_the_bar_row_intact(tb_ramp):
    """The other direction, which does NOT occur in the local corpus and must
    still be right. Refusing here would throw away the one row that carries a
    bar; flagging the missing reference sensor would put a finding on the sensor
    that is explicitly not allowed to produce one."""
    b, a = _rows(check_r11(fixtures.mutate(tb_ramp, drop=_A)))
    assert b.state is State.OK
    assert b.numbers['median_ratio'] > 0
    assert a.state is State.OK and not a.is_finding
    for key in _A:
        assert key in a.detail


# ---------------------------------------------------------------------------
# Alignment -- applied to both sensors
# ---------------------------------------------------------------------------

def test_the_numerator_is_clipped_to_the_denominators_span(tb_ramp):
    """`np.interp` CLAMPS outside its range without complaint, which would
    invent denominator values for steps the denominator never saw and pair them
    with real numerator values. Cutting the denominator short must cost those
    ticks, not fabricate them."""
    ns, _ = tb_ramp.history[_B[0]]
    ds, dv = tb_ramp.history[_B[1]]
    cut = 600
    m = fixtures.mutate(tb_ramp, history={_B[1]: (ds[:cut], dv[:cut])})
    expected = int(((ns >= ds[0]) & (ns <= ds[cut - 1])).sum())
    assert expected < len(ns)
    assert _row_b(check_r11(m)).numbers['n_aligned'] == expected


def test_a_zero_denominator_tick_is_dropped_and_counted(tb_ramp):
    """A zero intake residual makes an infinite ratio, and one infinity in a
    median is how a single tick decides a band."""
    ds, dv = tb_ramp.history[_B[1]]
    dv = np.asarray(dv, float).copy()
    dv[:100] = 0.0
    row = _row_b(check_r11(fixtures.mutate(tb_ramp, history={_B[1]: (ds, dv)})))
    assert row.numbers['n_dropped'] == 100
    assert row.numbers['n_aligned'] == len(ds) - 100
    assert np.isfinite(row.numbers['median_ratio'])


def test_non_finite_ticks_are_dropped_and_counted(tb_ramp):
    ns, nv = tb_ramp.history[_B[0]]
    nv = np.asarray(nv, float).copy()
    nv[:50] = np.nan
    row = _row_b(check_r11(fixtures.mutate(tb_ramp, history={_B[0]: (ns, nv)})))
    assert row.numbers['n_dropped'] == 50
    assert np.isfinite(row.numbers['median_ratio'])


def test_sensor_a_alignment_still_drops_a_zero_denominator(tb_ramp):
    """The same guard, on the reference sensor. It reports rather than flags, so
    a fabricated infinity there is a wrong NUMBER instead of a wrong state --
    which is quieter and no less wrong."""
    ds, dv = tb_ramp.history[_A[1]]
    dv = np.asarray(dv, float).copy()
    dv[:100] = 0.0
    row = _row_a(check_r11(fixtures.mutate(tb_ramp, history={_A[1]: (ds, dv)})))
    assert row.numbers['n_dropped'] == 100
    assert np.isfinite(row.numbers['median_ratio'])


def test_the_window_restricts_what_is_read(tb_ramp):
    ns, _ = tb_ramp.history[_B[0]]
    expected = int((ns >= max(ns[-1] - 2000, ns[0])).sum())
    b, a = _rows(check_r11(tb_ramp, window=2000))
    assert b.numbers['n_aligned'] == expected < len(ns)
    assert '2000' in b.numbers['window'] and '2000' in a.numbers['window']


# ---------------------------------------------------------------------------
# Fail loudly
# ---------------------------------------------------------------------------

def test_both_sensors_missing_on_an_in_scope_run_names_all_four(tb_ramp):
    """On a route R11 IS defined on, absent series are a HOLE, not a quiet pass.
    Dropped from a real TB run rather than taken from a run that never logged
    them, so the route stays in scope and only the data is missing."""
    res = check_r11(fixtures.mutate(tb_ramp, drop=_ALL))
    assert not res.ran and not res.rows
    for key in _ALL:
        assert key in res.reason, key


def test_too_few_aligned_points_says_how_many_it_had(tb_ramp):
    """A window narrower than the eval cadence leaves one tick per sensor, and
    'the median of one point' is whichever tick the cadence landed on."""
    res = check_r11(tb_ramp, window=5)
    assert not res.ran
    assert '1 aligned point' in res.reason
    assert str(len(tb_ramp.history[_B[0]][0])) in res.reason   # what it did have
    assert str(len(tb_ramp.history[_A[0]][0])) in res.reason


def test_disjoint_step_ranges_are_not_run_with_nothing_aligned(tb_ramp):
    """Two series that never overlap in step give zero aligned ticks. Reporting
    that as a clean pass would be the worst reading in the file."""
    m = _copy(tb_ramp)
    for num, den in (_B, _A):
        s, v = m.history[num]
        m.history[num] = (np.asarray(s, float) + 1e6, np.asarray(v, float))
    res = check_r11(m)
    assert not res.ran
    assert res.reason.count('0 aligned point') == 2


def test_the_minimum_is_exact(tb_ramp):
    """One point either side of the guard, with the guard read out of the
    module's own row rather than copied from it."""
    need = _min_aligned(tb_ramp)
    s, v = tb_ramp.history[_B[0]]
    assert _row_b(check_r11(fixtures.mutate(
        tb_ramp, history={_B[0]: (s[:need - 1], v[:need - 1])}))
    ).state is State.UNREADABLE
    row = _row_b(check_r11(fixtures.mutate(
        tb_ramp, history={_B[0]: (s[:need], v[:need])})))
    assert row.numbers['n_aligned'] == need


def test_the_summary_fallback_is_not_reported_as_a_ratio(tb_ramp):
    """THE TRAP. `base.series` answers a missing history series from the SUMMARY
    as ONE point at `last_step`. Two of those make a ratio of two final values
    that renders exactly like a median over the run."""
    m = _copy(tb_ramp)
    for key in _ALL:
        m.history.pop(key)
        assert key in m.summary, key
    res = check_r11(m)
    assert not res.ran
    assert res.reason.count('1 aligned point') == 2


# ---------------------------------------------------------------------------
# Corpus invariants
# ---------------------------------------------------------------------------

def test_every_captured_run_reads_without_a_traceback(all_runs):
    for name, run in all_runs.items():
        res = check_r11(run)
        assert res.ran or res.reason, name
        assert len(res.rows) == (2 if res.ran else 0), name


def test_every_in_scope_row_is_tagged_with_which_sensor_it_is(all_runs):
    """Two rows with the same subject shape are two numbers a reader has to tell
    apart, and the state column does not say which of them carries the bar. The
    tag does, on every path that produces a number or refuses to."""
    for name, run in all_runs.items():
        res = check_r11(run)
        if not res.ran or res.rows[0].state is State.NA_ROUTE:
            continue
        assert _tag(res.rows[0]) == 'DERIVED BAR', name
        assert _tag(res.rows[1]) == 'REFERENCE ONLY', name


def test_every_captured_run_reads_the_same_with_a_window(all_runs):
    """The window is the only optional argument and it must not change which
    branch a run takes -- only how much of it is read."""
    for name, run in all_runs.items():
        bare, windowed = check_r11(run), check_r11(run, window=100000)
        assert bare.ran == windowed.ran, name
        if bare.ran:
            assert [r.state for r in bare.rows] == \
                   [r.state for r in windowed.rows], name


def test_sensor_b_does_not_fire_on_any_unmutated_run(all_runs):
    """The other half of every FLAG test above, over the whole fixture corpus. A
    state that fires on a corpus of runs the user says are not memorising is not
    a finding, and that is exactly what the sensor this replaced was doing."""
    fired = [n for n, run in all_runs.items()
             if any(r.state is State.FLAG for r in check_r11(run).rows)]
    assert not fired, fired


def test_sensor_a_never_flags_on_any_real_run(all_runs):
    """THE MEASURED REASON THIS CHECK WAS REWORKED, as an invariant. Sensor A is
    below 1 on most of the local TB corpus; a finding state on it fires on three
    quarters of the runs and a check that cries wolf gets switched off."""
    for name, run in all_runs.items():
        res = check_r11(run)
        if not res.ran:
            continue
        a = res.rows[1]
        assert a.state in (State.OK, State.NA_ROUTE), (name, a.state)
        assert not a.is_finding, name


def test_no_row_or_reason_reads_as_a_verdict(all_runs):
    """R11 is a mechanism. The check names the band; what the band implies is
    the reader's, and a conclusion here would be the failure the whole package
    exists to stop."""
    texts = []
    for run in all_runs.values():
        results = [check_r11(run), check_r11(run, window=5)]
        for key in _ALL:
            if key in run.history:
                results.append(check_r11(_scaled(run, key, 0.2)))
        for res in results:
            texts.append(res.reason)
            texts += [r.detail for r in res.rows]
    blob = ' '.join(texts).lower()
    assert blob.strip()
    for word in _VERDICT_WORDS:
        assert word not in blob, word


def test_every_computed_row_carries_the_numbers_behind_it(all_runs):
    """A finding without its inputs is an assertion. NA_ROUTE and an unreadable
    sensor are the rows with nothing to show, because their whole content is
    that there is nothing to compute."""
    common = ('median_ratio', 'num_median', 'den_median', 'n_aligned', 'window')
    for name, run in all_runs.items():
        res = check_r11(run)
        if not res.ran:
            continue
        for row, extra in zip(res.rows, (('bar', 'release'),
                                         ('below_ratio', 'ref_ratio'))):
            assert row.detail, name
            if row.state is State.NA_ROUTE or 'median_ratio' not in row.numbers:
                continue
            for field in common + extra:
                assert field in row.numbers, (name, row.subject, field)


# ---------------------------------------------------------------------------
# Defects found by adversarial verification against the real corpus
# ---------------------------------------------------------------------------

def _row_b(res):
    return next(r for r in res.rows
                if K.R11_MEMORISATION_NUMERATOR in r.subject)


def test_the_release_is_attributed_to_whoever_actually_set_it(tb_ramp):
    """The derived BAR is a property of the quantity, so it travels. The RELEASE
    is not derived -- it comes from the config generators -- and most runs
    declare no `buffer_servo` at all. Naming [bar, release) "the buffer servo's
    hold band" unread asserted a controller the run may not have."""
    with_servo = _row_b(check_r11(tb_ramp))
    assert with_servo.numbers['release_source'] == 'run'

    stripped = fixtures.mutate(tb_ramp)
    for k in [k for k in stripped.config if 'buffer_servo' in k]:
        stripped.config.pop(k)
    without = _row_b(check_r11(stripped))
    assert without.numbers['release_source'] == 'generator_default'
    assert 'DECLARES NO' in without.detail


def test_a_logged_but_dead_sensor_is_not_told_it_is_merely_absent(tb_ramp):
    """A sensor logged on every tick whose values are all NaN is a DEAD SENSOR.
    The row used to append "the sensor abstaining or predating its plumbing" to
    every unreadable path, so it told the reader to stop looking at exactly the
    case worth looking at -- a swallowed diagnostic failing as reassurance."""
    s, v = tb_ramp.history[K.R11_MEMORISATION_NUMERATOR]
    dead = fixtures.mutate(tb_ramp, history={
        K.R11_MEMORISATION_NUMERATOR: (s, np.full_like(v, np.nan))})
    row = _row_b(check_r11(dead))
    assert row.state is State.UNREADABLE
    assert 'abstaining' not in row.detail

    # the companion: genuine absence DOES get the benign reading, or the test
    # above would pass on a check that had simply deleted the sentence
    gone = fixtures.mutate(tb_ramp, drop=(K.R11_MEMORISATION_NUMERATOR,))
    assert 'abstaining' in _row_b(check_r11(gone)).detail


def test_dropped_ticks_are_counted_inside_the_window(tb_ramp):
    """`n_dropped` spanned the whole history while the median and `n_aligned`
    beside it were windowed, so a windowed row reported drops for ticks it had
    not read -- a clean window could be made to look a third garbage."""
    s, v = tb_ramp.history[K.R11_MEMORISATION_DENOMINATOR]
    spoiled = v.copy()
    spoiled[:100] = 0.0                      # early ticks only
    m = fixtures.mutate(tb_ramp,
                        history={K.R11_MEMORISATION_DENOMINATOR: (s, spoiled)})
    assert _row_b(check_r11(m)).numbers.get('n_dropped', 0) == 100
    late = _row_b(check_r11(m, window=2000.0)).numbers
    assert late.get('n_dropped', 0) == 0, 'window excludes the spoiled ticks'
