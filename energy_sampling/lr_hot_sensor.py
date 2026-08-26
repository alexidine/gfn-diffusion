"""
Per-stage "the LR is hot" drawdown sensor -- the decision half. Pure: no torch,
no trainer. `train.Modeller.observe_hot_lr` feeds it.

WHAT GAP THIS FILLS. `lr_ctrl/divergences` was 0 in every one of 97 stage
segments examined, INCLUDING all 19 failures: the hard tripwire does not fire on
this failure mode in any training mode. `grad_norm_pre_clip` is not a substitute
-- during the worst of the a100_stab_aug16_f1 blow-up it FELL to ~860 while
fwd/scatter_err sat at 3.5e6, because the clip absorbed it.

The sensor says "this run is destabilising". It does NOT attribute cause: on
equilibration four healthy runs reached the same peak LR as the blow-ups, so a
causal reading of this signal alone would be wrong.

## The statistic

One channel, one trailing window, one threshold:

    push the fresh value into the window          # W most recent FRESH rows
    if the window is not full -> NO_READING
    floor    = 10th percentile of the window
    drawdown = log(value / floor)                 # log form
    drawdown = value - floor                      # absolute form (bwd/mle)
    fired    = drawdown > threshold

SINGLE-ROW FIRE. No dwell, no confirmation count, no multi-channel vote. That is
not an omission -- see WHY NO DWELL below, which is the counterintuitive part.

## Why a level comparison and not a trend fit

The logged channels are EMAs, with increment autocorrelation 0.92-0.99. A fitted
slope treats rows as independent evidence and overstates its confidence by a long
way; a drawdown compares two levels of one series and assumes nothing. It is also
shape-agnostic, which a jump test is not: it fires on the instantaneous
detonation (train_prior's step-110 discontinuity) AND on a gradual runaway (a
20-nat rise over 400 steps that a single-row jump test misses entirely).

## Why a trailing window and not the stage minimum

An expanding minimum is dragged down by the whole stage history and does not
separate at all -- healthy 0.81 against failing 0.55 on equilibration, i.e. the
wrong way round. Longer trailing windows are strictly worse too (separation 6.3x
at 100 steps, ~4.5x at 2000): healthy runs decline, so an old floor sits below
the current level and an ordinary pause reads as drawdown.

## Why the 10th percentile and not min

`min` has zero breakdown. One spuriously low reading inflates every subsequent
reading 1:1 for a full lookback, so a single bad row at threshold size
false-fires on its own. The 10th percentile absorbs exactly one outlier per
11-row window, at a separation cost of about 0.01x. At W=11 it tolerates ONE; if
a channel ever shows paired glitches, move to the 25th percentile and accept
6.6x/4.9x margins.

## WHY NO DWELL

A one-off bad raw batch does not arrive as one row. The EMA smears it into a
~10-row bump, so at 30x the current level it already produces three consecutive
rows over a 1.2 bar: a "3 consecutive rows" confirmation does not filter it, it
only DELAYS the same false fire -- while costing a real detection, since one
blow-up crosses for a single row. The correct defence is threshold headroom,
which is why the equilibration bar is 2.0 rather than 1.2. That moves the
crossover for a one-off batch from 25x to 68x at a cost of <= 40 steps latency.

## THE NOISE MARGIN IS INHERITED FROM THE TRACKER, NOT INTRINSIC

A one-off raw batch at M times the current level reads
`log(1 + alpha * (M - 1))`, so the tolerated M is `1 + (e^T - 1) / alpha`, about
68x at the calibrated constants -- and it scales with 1/alpha.

    alpha = 1 - exp(-cadence / tracker_period) = 0.0952 at 10 / 100

IF `MetricTracker`'s period OR the logging cadence CHANGES, THIS SENSOR'S NOISE
TOLERANCE CHANGES SILENTLY AND PROPORTIONALLY. Period 100 -> 50, or cadence
10 -> 20, halves it. `HotSensor` therefore asserts the tracker period it was
calibrated against rather than trusting a comment; see `CALIBRATED_PERIOD`.

## Read contract -- each item has a failure behind it

  1. READ THE TRACKER'S SMOOTHED VALUE, NEVER `_last_stats`. Every threshold is
     fitted to the logged channel, which is the EMA. The raw pre-EMA value is
     ~10.5x more responsive at this cadence, so on raw input every threshold is
     wrong in the unsafe direction and the noise analysis above is void. This
     deliberately diverges from `update_mle_gate`, which reads raw -- that gate
     fits a slope and needs independent samples; a level comparison does not.
  2. ADVANCE ONLY ON FRESH WRITES. `MetricTracker.get` is a stale read by
     construction; the caller passes `written_step` and the window moves only
     when it changes. Precedent: protocol exit streaks once counted one stale
     sample as N independent passes. A branch that is not running emits nothing,
     and its last value must not keep feeding the window.
  3. A GAP IS A GAP. A missed tick simply does not update the window -- the floor
     is still a floor over what arrived. But once the channel has gone unwritten
     for longer than the lookback, the reading is NO_READING, never 0. A hole
     that reads as "no drawdown" is indistinguishable from health.
     ON THE LIVE ROUTES THIS PATH SHOULD NEVER RUN, which is the point of
     keeping it. Every declared channel is written at least every 10 steps: a
     fused stage force-refreshes each non-dormant branch on `refresh_every`, and
     a branch that trains logs on its own 10-step count. The one escape is
     dormancy -- and `Stage.read_modes` now names the sensor's channel, so a
     declared branch cannot be dormant. The guard is therefore a backstop for a
     future config shape, not a case the current ones reach; if `hot/drawdown`
     ever goes NaN mid-stage on mk_dev, that is a finding about the run, not
     about the sensor.
  4. RESET AT STAGE ENTRY, ON RESUME, AND AT A BRACKET TRIAL RESTORE. A window
     carrying `train_prior` rows into `equilibration` has a floor from another
     regime -- and `fwd/scatter_err` is not even written during `train_prior`,
     which is bwd-only.
  5. ARM THE MOMENT THE WINDOW FILLS. No settling delay: four of seven
     equilibration blow-ups fired 150-190 steps after stage entry, so a 400-step
     settle loses them, and the entry period is the SAFEST on healthy runs
     (max reading 0.14 in the first 400 steps against up to 0.49 after).

## Report-only, structurally

`action` accepts exactly one value. Nothing in this module or its caller can move
a learning rate, so turning this into an actuator is a code change with a review
rather than a config edit. The retired `ray_calibration.enabled` flag is why: a
second switch that could disagree with the thing it switched.
"""

from __future__ import annotations

import math

#: Permitted responses. 'report' moves nothing; 'fire' hands the verdict to the
#: unified fire response (train.observe_hot_lr) -- the reviewed actuation the
#: module docstring's report-only doctrine required. The DECISION stays here;
#: the RESPONSE stays outside, so this module still cannot move a rate.
ACTIONS = ('report', 'fire')

#: How the drawdown is formed. `absolute` is required on a channel that crosses
#: zero -- `bwd/mle` runs +9.75 to -33.74, where the log ratio is undefined.
FORMS = ('log', 'absolute')

#: The `MetricTracker` period the thresholds were calibrated against. The
#: sensor's noise tolerance is inherited from the EMA and scales with 1/alpha, so
#: a changed period silently rescales every threshold. Asserted, not commented.
CALIBRATED_PERIOD = 100

#: Reported when the sensor cannot form a reading: window not yet full, or the
#: channel unwritten for longer than its lookback. NEVER 0 -- a hole that reads
#: as "no drawdown" is indistinguishable from health.
NO_READING = None


def percentile(values, q):
    """Linear-interpolated percentile over an unsorted sequence.

    Written out rather than taken from numpy so the interpolation convention is
    part of this file: at W=11 and q=10 the position is 0.10 * 10 = 1.0, i.e.
    EXACTLY the second-smallest row. That is what "tolerates one outlier per
    window" means, and it would change silently under a different convention.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (float(q) / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


class HotSensor:
    """One stage's drawdown sensor. `spec` is the parsed `hot_lr_sensor` block."""

    def __init__(self, spec, stage_name='', tracker_period=None):
        self.stage = str(stage_name)
        self.action = spec.get('action', 'report')
        self.channel = spec['channel']
        self.form = spec.get('form', 'log')
        self.rows = int(spec['rows'])
        self.above = float(spec['above'])
        self.floor_percentile = float(spec.get('floor_percentile', 10.0))
        self.row_steps = int(spec.get('row_steps', 10))
        #: The window spans (rows - 1) gaps, so 11 rows at a 10-step cadence is
        #: 100 steps. This is also the staleness horizon: unwritten for longer
        #: than this and the reading is NO_READING.
        self.lookback = (self.rows - 1) * self.row_steps

        if tracker_period is not None and int(tracker_period) != CALIBRATED_PERIOD:
            raise ValueError(
                f"stage '{self.stage}': hot_lr_sensor thresholds were calibrated "
                f"against MetricTracker(period={CALIBRATED_PERIOD}) and this run "
                f"builds it with period={int(tracker_period)}. The sensor's noise "
                f"tolerance is inherited from the EMA and scales with 1/alpha "
                f"(alpha = 1 - exp(-cadence/period)), so every threshold here is "
                f"silently rescaled by that change. Re-fit them or restore the "
                f"period; do not adjust the bar by hand.")

        self.window = []          # the W most recent FRESH values
        self.last_written = None  # step of the most recent fresh write
        self.fires = 0
        self.last = {}

    # ------------------------------------------------------------------ state

    def reset(self):
        """Drop the window. Called at stage entry, on resume, and at every
        bracket trial restore -- each makes the trailing rows describe a regime
        the run is no longer in."""
        self.window = []
        self.last_written = None

    @property
    def armed(self):
        return len(self.window) >= self.rows

    # ---------------------------------------------------------------- reading

    def observe(self, step, value, written_step):
        """One reporting tick. Returns the verdict dict, always.

        A sensor that publishes nothing while quiet cannot be told from one that
        is not running, so every tick produces a row -- with `drawdown` None when
        no reading can be formed.
        """
        fresh = (written_step is not None and value is not None
                 and math.isfinite(value)
                 and (self.last_written is None or written_step > self.last_written))
        if fresh:
            self.window.append(float(value))
            if len(self.window) > self.rows:
                del self.window[0]
            self.last_written = int(written_step)

        drawdown = self._drawdown()
        # STALE PAST THE LOOKBACK IS NO_READING. Inside the lookback a missed
        # tick just does not move the window -- the floor is still a floor over
        # what arrived -- but once the channel has been silent longer than the
        # window itself spans, there is no current level to compare against.
        if (self.last_written is not None
                and step - self.last_written > self.lookback):
            drawdown = NO_READING

        fired = drawdown is not None and drawdown > self.above
        if fired:
            self.fires += 1
        self.last = {'step': int(step), 'drawdown': drawdown, 'fired': fired,
                     'fresh': bool(fresh), 'rows': len(self.window),
                     'floor': self._floor(), 'value': self.window[-1] if self.window else None}
        return self.last

    def _floor(self):
        if not self.armed:
            return None
        return percentile(self.window, self.floor_percentile)

    def _drawdown(self):
        if not self.armed:
            return NO_READING
        floor = percentile(self.window, self.floor_percentile)
        value = self.window[-1]
        if self.form == 'absolute':
            return value - floor
        if floor is None or floor <= 0 or value <= 0:
            # THE LOG RATIO IS UNDEFINED HERE, and guessing a direction on an
            # undefined statistic turns an instrument failure into a training
            # verdict. Hold.
            return NO_READING
        return math.log(value / floor)

    # ----------------------------------------------------------------- report

    def report(self) -> dict:
        """Emitted DIRECTLY into the metrics dict by the caller, never through
        `metric_tracker` -- that would EMA the sensor's own output, which is
        already a statistic over an EMA."""
        d = self.last.get('drawdown')
        return {
            'hot/drawdown': float('nan') if d is None else float(d),
            'hot/fired': 1.0 if self.last.get('fired') else 0.0,
            'hot/fires_total': float(self.fires),
        }
