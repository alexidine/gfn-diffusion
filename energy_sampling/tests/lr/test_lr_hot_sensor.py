"""
The per-stage "the LR is hot" drawdown sensor.

The thresholds are calibration data from a 97-segment corpus sweep and are not
re-derived here. What is tested is the seven mutations an implementer could
plausibly make, each of which leaves a sensor that still runs and still reports:

  1. detonation replay      -- not armed, wrong channel, threshold too high
  2. slow-drift             -- an expanding minimum instead of a trailing window
  3. downward glitch        -- a strict `min` floor instead of a percentile
  4. EMA-shaped one-off     -- the threshold lowered "because it looks insensitive"
  5. stale feed             -- polling `get` instead of tracking `written_step`
  6. sign crossing          -- the log form applied to a channel that crosses zero
  7. stage entry            -- a floor carried across a regime change

Several of these fail SILENTLY in the quiet direction, which is why the
assertions are two-sided wherever a mutation would merely make the sensor less
sensitive: a test that only checks "it stays silent here" passes on a sensor that
is silent everywhere. Each such test also asserts that the mutation it guards
against WOULD have fired.
"""

import math

import pytest

from energy_sampling.lr_hot_sensor import (CALIBRATED_PERIOD, HotSensor,
                                           percentile)

ROW = 10

#: The three validated sensors, as the config declares them.
TRAIN_PRIOR = {'channel': 'bwd/mle', 'form': 'absolute', 'rows': 31, 'above': 5.0}
EQUILIBRATION = {'channel': 'fwd/scatter_err', 'rows': 11, 'above': 2.0}
VAR_COND = {'channel': 'fwd/vg_lb', 'rows': 11, 'above': 3.0}

ALPHA = 1.0 - math.exp(-ROW / CALIBRATED_PERIOD)


def _sensor(spec=None, **over):
    s = dict(spec or EQUILIBRATION)
    s.update(over)
    return HotSensor(s, 'equilibration')


def _feed(sensor, values, start=1000, step=ROW):
    out = []
    for i, v in enumerate(values):
        at = start + i * step
        out.append(sensor.observe(at, v, at))
    return out


def _ema(raw, period=CALIBRATED_PERIOD, dt=ROW):
    """The logged series for a raw per-row sequence, as MetricTracker makes it."""
    a = 1.0 - math.exp(-dt / period)
    out, e = [], None
    for x in raw:
        e = x if e is None else (1 - a) * e + a * x
        out.append(e)
    return out


# ------------------------------------------------------------ the statistic ---

def test_the_percentile_convention_tolerates_exactly_one_outlier_at_W11():
    """"Tolerates one outlier per window" IS the interpolation convention: at
    W=11 and q=10 the position is 0.10 * 10 = 1.0, exactly the second-smallest
    row. A different convention moves the breakdown point silently."""
    assert percentile(list(range(11)), 10) == 1.0        # 2nd smallest of 11
    assert percentile(list(range(31)), 10) == 3.0        # 4th smallest of 31


def test_the_sensor_is_not_armed_until_the_window_is_full():
    s = _sensor()
    out = _feed(s, [1.0] * (s.rows - 1))
    assert all(v['drawdown'] is None for v in out), 'a short window produced a reading'
    assert all(not v['fired'] for v in out)


def test_a_no_reading_is_never_zero():
    """A hole that reads as "no drawdown" is indistinguishable from health."""
    s = _sensor()
    _feed(s, [1.0] * 3)
    assert math.isnan(s.report()['hot/drawdown'])
    assert s.report()['hot/drawdown'] != 0.0


# --------------------------------------------------------- 1. detonation ------

def test_a_detonation_fires_on_the_row_it_happens():
    """Kills: sensor never armed, wrong channel, threshold too high."""
    s = _sensor(TRAIN_PRIOR)                       # absolute form, bar 5.0
    descending = [7.0 - 0.02 * i for i in range(40)]
    out = _feed(s, descending + [descending[-1] + 100.0])
    assert not any(v['fired'] for v in out[:-1]), 'fired during a normal descent'
    assert out[-1]['fired'], 'the detonation row did not fire'
    assert out[-1]['drawdown'] > 50.0


# --------------------------------------------------------- 2. slow drift ------

def test_a_slow_drift_does_not_fire_and_an_expanding_minimum_would():
    """THE BUG THAT SANK THE FIRST VERSION OF THIS IDEA. A healthy run rising
    e^2.5 over 5000 steps is not destabilising; only a floor that remembers the
    whole stage calls it one."""
    s = _sensor()
    n = 5000 // ROW
    rising = [math.exp(2.5 * i / n) for i in range(n)] + [math.exp(2.5)] * 20
    out = _feed(s, rising)
    peak = max(v['drawdown'] for v in out if v['drawdown'] is not None)
    assert not any(v['fired'] for v in out), (
        f'the trailing window fired on a slow drift (peak {peak:.3f})')

    expanding = max(math.log(v / min(rising[:i + 1])) for i, v in enumerate(rising))
    assert expanding > s.above, (
        'an expanding-minimum floor would not fire either, so this test '
        'discriminates nothing')


# ------------------------------------------------------ 3. downward glitch ----

def test_one_low_row_does_not_inflate_the_reading_and_a_min_floor_would():
    """Kills a strict `min` floor. `min` has zero breakdown: one spurious low row
    raises every subsequent reading 1:1 for a full lookback."""
    s = _sensor()
    series = [1.0] * 12 + [0.1] + [1.0] * 12          # one row at a tenth
    out = _feed(s, series)
    assert not any(v['fired'] for v in out), 'a single low row false-fired'

    worst = max(math.log(v / min(series[max(0, i - 10):i + 1]))
                for i, v in enumerate(series) if i >= 10)
    assert worst > s.above, 'a min floor would not fire either; nothing is discriminated'


def test_two_low_rows_do_exceed_the_windows_tolerance():
    """The breakdown point is ONE per 11-row window, asserted rather than hoped:
    if a channel ever shows paired glitches the floor percentile must move."""
    s = _sensor()
    out = _feed(s, [1.0] * 12 + [0.1, 0.1] + [1.0] * 12)
    assert any(v['fired'] for v in out), (
        'two low rows did not breach the tolerance -- the floor is more forgiving '
        'than the calibration assumes')


# ---------------------------------------------------- 4. EMA-shaped one-off ---

def test_a_one_off_raw_batch_at_30x_does_not_fire_at_the_calibrated_bar():
    """THE REASON THERE IS NO DWELL. A bad raw batch does not arrive as one row --
    the EMA smears it into a ~10-row bump, so a "3 consecutive rows" confirmation
    would only DELAY the same false fire while costing a real detection. Headroom
    is the defence instead, which is why the bar is 2.0 and not 1.2."""
    s = _sensor()
    out = _feed(s, _ema([1.0] * 20 + [30.0] + [1.0] * 40))
    peak = max(v['drawdown'] for v in out if v['drawdown'] is not None)

    assert peak == pytest.approx(math.log(1 + ALPHA * 29), rel=0.05), (
        'the bump is not the analytic log(1 + alpha*(M-1)) shape')
    assert not any(v['fired'] for v in out), f'a one-off 30x batch fired (peak {peak:.3f})'
    assert peak > 1.2, (
        'the bump does not even clear 1.2, so this test would pass on a sensor '
        'with no headroom and proves nothing about the 2.0 bar')


def test_the_tolerated_one_off_multiple_is_the_documented_68x():
    """The noise margin is inherited from the tracker, not intrinsic: the
    tolerated M is 1 + (e^T - 1)/alpha, and it scales with 1/alpha."""
    assert 1 + (math.exp(2.0) - 1) / ALPHA == pytest.approx(68, rel=0.05)


# --------------------------------------------------------- 5. stale feed ------

def test_a_stale_feed_does_not_advance_the_window():
    """Kills `get`-based polling. `MetricTracker.get` returns the same sample
    forever once written; protocol exit streaks once counted one stale sample as
    N independent passes."""
    s = _sensor()
    _feed(s, [1.0] * 11, start=1000)
    assert s.armed
    before = list(s.window)
    frozen = s.last_written

    # THE STEPS MUST STAY INSIDE THE LOOKBACK. Drift past it and the staleness
    # horizon suppresses the reading for its own reason, which masks this bug --
    # measured: the first version of this test passed with the freshness check
    # deleted, because it had walked 1000 steps past a 100-step horizon.
    out = [s.observe(frozen + (i + 1) * ROW, 99.0, frozen) for i in range(5)]
    assert frozen + 5 * ROW - s.last_written <= s.lookback, 'fixture drifted stale'

    assert list(s.window) == before, 'a stale read extended the window'
    assert not any(v['fired'] for v in out), 'a stale read fired'
    assert all(v['drawdown'] == pytest.approx(0.0) for v in out)


def test_the_stale_guard_is_what_stops_it_not_the_staleness_horizon():
    """The mutation guard for the test above: accepting the stale value really
    would fire, so that assertion cannot pass for an unrelated reason."""
    s = _sensor()
    _feed(s, [1.0] * 11, start=1000)
    window_if_accepted = list(s.window)[1:] + [99.0]
    floor = percentile(window_if_accepted, s.floor_percentile)
    assert math.log(99.0 / floor) > s.above


def test_staleness_past_the_lookback_reports_no_reading():
    s = _sensor()
    _feed(s, [1.0] * 11, start=1000)
    last = s.last_written
    inside = s.observe(last + s.lookback, 1.0, last)
    beyond = s.observe(last + s.lookback + ROW, 1.0, last)
    assert inside['drawdown'] is not None, 'a gap inside the lookback lost the reading'
    assert beyond['drawdown'] is None, (
        'the channel has been silent longer than its window spans, so there is no '
        'current level to compare against -- NO_READING, not 0')


# ------------------------------------------------------- 6. sign crossing -----

def test_the_absolute_form_survives_a_channel_that_crosses_zero():
    """`bwd/mle` runs +9.75 to -33.74. The log ratio is undefined there, so the
    absolute form is not a stylistic choice."""
    s = _sensor(TRAIN_PRIOR)
    out = _feed(s, [5.0 - 0.8 * i for i in range(40)])     # +5 down through -25
    assert all(v['drawdown'] is None or math.isfinite(v['drawdown']) for v in out)
    assert not any(v['fired'] for v in out), 'an improving MLE fired'

    rising = _feed(s, [-25.0 + 2.0 * i for i in range(20)], start=5000)
    assert any(v['fired'] for v in rising), 'a 40-nat rise through zero did not fire'


def test_the_log_form_holds_rather_than_fires_on_a_non_positive_value():
    """Undefined is not a verdict."""
    s = _sensor()
    out = _feed(s, [1.0] * 11 + [0.0])
    assert out[-1]['drawdown'] is None
    assert not out[-1]['fired']


# --------------------------------------------------------- 7. stage entry -----

def test_a_reset_empties_the_window_so_a_floor_cannot_cross_a_regime():
    """A window carrying `train_prior` rows into `equilibration` has a floor from
    another regime -- and `fwd/scatter_err` is not even written during
    `train_prior`, which is bwd-only. Reset also covers resume and every bracket
    trial restore."""
    s = _sensor()
    _feed(s, [1.0] * 11)                    # a LOW-level regime
    assert s.armed

    s.reset()
    assert not s.armed and s.window == [] and s.last_written is None
    # ...and the new stage sits at a much higher level, which is the dangerous
    # direction: a floor remembered from the old stage would read the new
    # stage's ordinary level as a 100x drawdown.
    out = _feed(s, [100.0] * 11, start=9000)
    assert out[-1]['drawdown'] == pytest.approx(0.0), (
        'the new stage was judged against the old stage floor')
    assert not any(v['fired'] for v in out)


def test_without_the_reset_the_old_floor_would_convict_the_new_stage():
    """The mutation guard for the test above."""
    s = _sensor()
    _feed(s, [1.0] * 11)
    out = _feed(s, [100.0] * 11, start=9000)   # no reset
    assert any(v['fired'] for v in out), (
        'a 100x regime change did not fire even without a reset, so the test '
        'above discriminates nothing')


# ------------------------------------------------------------ the contract ----

def test_the_tracker_period_is_asserted_not_assumed():
    """The noise tolerance is inherited from the EMA and scales with 1/alpha, so
    a changed period silently rescales every threshold."""
    HotSensor(EQUILIBRATION, 'equilibration', tracker_period=CALIBRATED_PERIOD)
    with pytest.raises(ValueError, match='calibrated against'):
        HotSensor(EQUILIBRATION, 'equilibration', tracker_period=50)


def test_the_only_permitted_action_is_report():
    from energy_sampling.lr_hot_sensor import ACTIONS
    assert ACTIONS == ('report',)


def test_the_report_distinguishes_never_ran_from_ran_and_clean():
    s = _sensor()
    assert math.isnan(s.report()['hot/drawdown'])       # never ran
    _feed(s, [1.0] * 11)
    r = s.report()
    assert r['hot/drawdown'] == pytest.approx(0.0)      # ran, nothing to report
    assert r['hot/fired'] == 0.0 and r['hot/fires_total'] == 0.0


def test_fires_total_accumulates():
    s = _sensor()
    _feed(s, [1.0] * 11 + [30.0, 30.0])
    assert s.report()['hot/fires_total'] >= 1.0


def test_all_three_shipped_sensors_construct_and_arm():
    for spec in (TRAIN_PRIOR, EQUILIBRATION, VAR_COND):
        s = HotSensor(spec, 'x', tracker_period=CALIBRATED_PERIOD)
        base = -20.0 if spec.get('form') == 'absolute' else 1.0
        out = _feed(s, [base] * spec['rows'])
        assert out[-1]['drawdown'] is not None, f"{spec['channel']} never armed"
        assert s.lookback == (spec['rows'] - 1) * ROW
