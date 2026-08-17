"""
The synthetic device: `t(B)`, `U(B)`, OOM, recompiles, and eval blocks.

WHY NOT `bench/old/clock.py`. That model is reusable in its timing half and
STRUCTURALLY UNABLE to pose the question Phase 6 exists to answer. Its occupancy is

    busy(B) = t_fixed*(1 - host_frac) + B/sps_max
    util(B) = busy(B) / t(B)

which is monotone rising in B and saturating at 100%, for every parameter setting.
So it can express the belief that batch is the occupancy lever, and it cannot express
the measurement that refuted it. The deleted `gpu_util_floor` was convicted on
umaperf0812's `c_controller`, where occupancy FELL as batch grew:

    batch      100    165    272    449    741
    util %      52     44     49     42      -
    samples/s  57.7   46.5   45.1   28.2   24.3

A sandbox whose device cannot produce that table cannot detect trap (a) when it is
injected -- it would report that the occupancy rule works, because on its device the
rule's premise is true by construction. That is the same shape as the failure being
guarded against: a rule that is right on the surface it was tested on.

WHAT IS PLANTED, AND WHAT THAT COSTS. Every number here is chosen, not measured.
That is legitimate for a sandbox -- the question is whether a CONTROL LAW behaves,
and the controller is problem-blind (it reads a timing series and nothing else) --
but it is NOT legitimate to read a batch size, a knee location or a threshold off
this file. The device transfers MECHANISM. The parameters that would make it a model
of the A100 are exactly what `docs/design/phase6_measurement_request.md` is for, and
until they land every cell here must be swept rather than pinned, so a verdict that
depends on one setting is visible as such.

THE PREVIOUS GENERATION'S KNEE WAS NEVER MEASURED. prod0810 walked 2722 -> 12226 on
three accounting bugs, and `bench/old/clock.py`'s tests plant a knee at 7410 and then
confirm the controller finds 7410. That is a legitimate regression test and it is not
evidence about hardware. Nothing in this file may be quoted as evidence about hardware.
"""

import math
import random

#: Occupancy shapes, named. Each is a claim about a ROUTE, and the whole point of
#: having more than one is that the two known routes disagree in SIGN.
RISING = 'rising'        # analytic energy: batch amortises a host-serial cost
FLAT = 'flat'            # batch does not move occupancy either way
DECLINING = 'declining'  # MLIP: the energy call already saturates; batch buys host work

#: THE OCCUPANCY SHAPE IS NOT ENOUGH ON ITS OWN, and this is the trap the cell design
#: most easily falls into. `util_shape=DECLINING` gives a falling U(B), but the TIMING
#: model `t(B) = t_fixed + B/sps_max` gives dS/dB > 0 for EVERY parameter setting -- so
#: a cell built from the shape alone has occupancy falling while throughput RISES, and
#: an injected occupancy rule that grows the batch is then *right on priority 2*.
#: Measured: at `t_fixed=0.02, sps_max=6000, util_shape=DECLINING` the injected arm is
#: BIT-IDENTICAL to null, because throughput climbs 2500 -> 4405 across the ladder and
#: the throughput gate never objects.
#:
#: Trap (a)'s condition is that U and S fall TOGETHER. Falling S needs an efficiency
#: that DEGRADES with size, which is what `regimes` is for. These multipliers produce
#: S = 2727/2357/1728/1139/706 and U = 76.0/69.6/61.7/53.0/44.5 at B = 100..741, i.e.
#: `util_is_monotone_in(...) == -1` and throughput monotone falling.
DECLINING_S_REGIMES = [(150, 0.55), (250, 0.33), (420, 0.20), (700, 0.12)]


class SyntheticDevice:
    """
    A deterministic step-time and occupancy model with jitter.

    TIMING (unchanged in shape from `bench/old/clock.py`, which was right about this):

        t(B) = t_fixed + effective_work(B) / sps(B)

    so samples/sec rises with batch and saturates. `tile`, `regimes` and
    `recompile_s` put the discreteness back; all default off, so the smooth case
    stays exact and a test written without them is visibly testing the easy one.

    OCCUPANCY (the part that is new):

        util(B) = clamp(util_floor + util_span * shape(B), 0, 100)

    where `shape` is one of RISING / FLAT / DECLINING. Occupancy is a SEPARATE
    parameter of the device, not a function of the timing model. That separation is
    the whole correction: on the real MLIP route occupancy and throughput moved in
    OPPOSITE directions, and a device that derives one from the other cannot
    represent it.

      RISING     shape(B) = (B/sps_max) / t(B)          -- batch amortises t_fixed
      FLAT       shape(B) = 0.5                          -- constant in B
      DECLINING  shape(B) = 1 / (1 + B/util_half)        -- batch buys host work

    `util_half` is the batch at which the declining shape has fallen halfway. It is
    PLANTED. Its real value is `U(B)` on the MLIP route, which is arm B of the
    measurement request and does not exist yet.
    """

    def __init__(self, t_fixed=2.0, sps_max=5000.0, oom_at=None, jitter=0.0,
                 tile=None, recompile_s=0.0, regimes=None, seed=0,
                 util_shape=RISING, util_floor=20.0, util_span=70.0,
                 util_half=400.0, util_jitter=0.0,
                 eval_period=None, eval_seconds=0.0, dynamo_cache_limit=None):
        self.t_fixed = float(t_fixed)
        self.sps_max = float(sps_max)
        self.oom_at = None if oom_at is None else int(oom_at)
        self.jitter = float(jitter)
        self.tile = None if tile in (None, 0) else int(tile)
        self.recompile_s = float(recompile_s)
        self.regimes = sorted(regimes or [], key=lambda r: r[0])
        self.util_shape = util_shape
        self.util_floor = float(util_floor)
        self.util_span = float(util_span)
        self.util_half = float(util_half)
        self.util_jitter = float(util_jitter)
        self.eval_period = None if eval_period in (None, 0) else int(eval_period)
        self.eval_seconds = float(eval_seconds)
        self.dynamo_cache_limit = (None if dynamo_cache_limit in (None, 0)
                                   else int(dynamo_cache_limit))
        self._rng = random.Random(seed)
        self._urng = random.Random(seed + 9973)
        self._seen_sizes = set()
        self.n_oom = 0
        self.n_steps = 0
        self.n_recompiles = 0

    # ------------------------------------------------------------------ timing

    def _effective_work(self, work):
        """Wave quantisation: a partial wave costs a whole one."""
        if self.tile is None:
            return float(work)
        return float(math.ceil(work / self.tile) * self.tile)

    def _sps(self, work):
        sps = self.sps_max
        for threshold, mult in self.regimes:
            if work >= threshold:
                sps = self.sps_max * mult
        return sps

    def true_step_time(self, work):
        """Noise-free, recompile-free step time for `work` samples."""
        return self.t_fixed + self._effective_work(work) / self._sps(work)

    def step_time(self, work):
        """
        One observed step time. Raises a CUDA-OOM-shaped error past the ceiling.

        The message is matched by `utils.is_cuda_oom`, so the REAL
        `handle_train_epoch_error` accepts it -- the bench must not carry its own OOM
        classifier, or it stops testing the one that ships.
        """
        self.n_steps += 1
        if self.oom_at is not None and work >= self.oom_at:
            self.n_oom += 1
            raise RuntimeError(
                f"CUDA out of memory. Tried to allocate {work / 1000:.2f} GiB "
                f"(synthetic; ceiling {self.oom_at})")
        t = self.true_step_time(work)
        size = int(work)
        if self.recompile_s > 0 and size not in self._seen_sizes:
            # first sight of this shape: one recompile + its own CUDA graph. Lands on
            # the FIRST step of a new rung -- i.e. the first measurement a growth gate
            # takes of it. Whether that biases the gate is a real question and this is
            # what lets it be asked.
            t += self.recompile_s
            self.n_recompiles += 1
        self._seen_sizes.add(size)
        if self.jitter > 0:
            t *= math.exp(self._rng.gauss(0.0, self.jitter))
        return t

    def throughput(self, work):
        """Samples per second at this batch, noise-free. Sawtooth when tiled."""
        return float(work) / self.true_step_time(work)

    # -------------------------------------------------------------- occupancy

    def true_utilization(self, work):
        """
        Noise-free occupancy percent at this batch. THE GROUND TRUTH -- what the
        scheduler integrates. The in-process sensor sees a sampled, eval-blind
        version of this; keeping the two separate is what makes eval blindness
        expressible rather than assumed away.
        """
        w = self._effective_work(work)
        if self.util_shape == FLAT:
            shape = 0.5
        elif self.util_shape == DECLINING:
            shape = 1.0 / (1.0 + w / self.util_half)
        elif self.util_shape == RISING:
            t = self.true_step_time(work)
            shape = (w / self._sps(work)) / t if t > 0 else 0.0
        else:
            raise ValueError(
                f'unknown util_shape {self.util_shape!r} -- expected one of '
                f'{RISING!r}, {FLAT!r}, {DECLINING!r}. A device whose occupancy '
                f'shape is a typo would silently fall back to one of the routes '
                f'and the cell would test the wrong one.')
        return max(0.0, min(100.0, self.util_floor + self.util_span * shape))

    def utilization(self, work):
        """One observed occupancy reading, with jitter. What `_read_gpu_util` returns."""
        u = self.true_utilization(work)
        if self.util_jitter > 0:
            u += self._urng.gauss(0.0, self.util_jitter)
        return max(0.0, min(100.0, u))

    # ------------------------------------------------------------ eval blocks

    def eval_block(self, step_ind):
        """
        Wall-clock seconds of GPU-IDLE eval on this step, contributing ZERO samples.

        Returns 0.0 unless an eval lands here. `eval_period=None` disables it, which is
        what every throughput benchmark in the registry does (`eval_period: 100000000`).

        WITHOUT THIS THE SANDBOX CANNOT POSE PHASE 6'S CENTRAL SENSOR PREMISE. The
        in-process sampler sits in the TRAINING portion of the loop body; eval, figure
        logging and archiving run later in the same iteration, so a 300 s eval
        contributes no occupancy samples while the scheduler counts every second of it
        (`docs/design/phase6_measurement_request.md` §2 (ii)). The consequence --
        `gpu/util_policy` OVERSTATES what the scheduler sees -- is the dangerous
        direction, and it is the one a controller must not be calibrated blind to.

        `bench/old/harness.py` advances its clock by train-step time only, so the
        retired harness had exactly this hole and could not have found it.
        """
        if self.eval_period is None or step_ind <= 0:
            return 0.0
        return self.eval_seconds if step_ind % int(self.eval_period) == 0 else 0.0

    def compile_cache_limit(self):
        """
        `torch._dynamo` cache size limit, or None for unlimited.

        Past N distinct shapes dynamo stops recompiling and falls back to EAGER -- a
        cliff, not the per-shape linear charge modelled in `step_time`. This is the
        mechanism that makes batch CHURN expensive rather than merely slow, and nothing
        in the repo models it. Note `_seen_sizes` is never cleared, so in this device
        churn is free after its first lap; a controller that oscillates between two
        rungs forever pays nothing here and would pay on real hardware.

        Returned rather than applied: the cliff's location is a torch config value, not
        a measurement, and no cell should silently assume one.
        """
        return self.dynamo_cache_limit

    # ------------------------------------------------------------- self-check

    def util_is_monotone_in(self, batches):
        """
        The SIGN of dU/dB over a ladder: +1 rising, -1 declining, 0 neither.

        Exists because a cell's whole meaning depends on this sign and it is easy to
        set `util_shape` and then choose parameters that flatten it -- at which point
        the cell silently tests nothing. `bench/audit.py`'s check 1 ("the rate
        matters") is the same idea one axis over.
        """
        us = [self.true_utilization(b) for b in batches]
        d = [b - a for a, b in zip(us, us[1:])]
        if all(x > 1e-9 for x in d):
            return 1
        if all(x < -1e-9 for x in d):
            return -1
        return 0


class MeasuredDevice(SyntheticDevice):
    """
    A device defined by an OBSERVED TABLE, interpolated -- not by a fitted formula.

    WHY THIS EXISTS AS A SEPARATE CLASS. The first attempt encoded umaperf0812 by
    choosing `t_fixed`, `sps_max` and four `regimes` multipliers to approximate five
    observed points. Checked at runtime, it failed the cell's own premise: throughput
    ROSE from batch 100 to 165 (54.5 -> 60.7) where the observation FELL
    (57.7 -> 46.5), because below the first regime threshold the `t_fixed`
    amortisation still dominates. The cell would have been reported as "U and S both
    decline" while its device delivered no such thing.

    Five points do not determine a four-parameter cost model, and a fit that misses
    the SIGN of the thing being tested is worse than no cell. `bench/README.md` lists
    "constants transcribed between documents rather than computed from data" as a
    cause of death; fitting a curve to a table and then quoting the curve is the same
    error with an extra step. So the table is the device.

    Between anchors: log-log linear on step time, linear on utilization.

    OUTSIDE the anchors, step time extrapolates PROPORTIONALLY (`t = t_end * B/B_end`),
    which holds THROUGHPUT flat at the endpoint value. That is the "no claim" choice
    and it is not the obvious one: holding the step time itself -- the first
    implementation here -- makes throughput grow LINEARLY outside the span, so this
    device reported 65.6 samples/s at batch 2000 against a top measured value of 24.3.
    A device that invents throughput past the last rung is the never-measured knee
    rebuilt inside the tool meant to detect it. Utilization is held flat, for the same
    reason and with no better option.

    `outside_range` reports it either way. Extrapolation is a held value, never a trend.
    """

    def __init__(self, batches, step_times, utils=None, **kw):
        if not (len(batches) == len(step_times) >= 2):
            raise ValueError('need >= 2 anchors and matching lengths')
        if utils is not None and len(utils) != len(batches):
            raise ValueError('utils must match batches in length')
        order = sorted(range(len(batches)), key=lambda i: batches[i])
        self.anchor_b = [float(batches[i]) for i in order]
        self.anchor_t = [float(step_times[i]) for i in order]
        self.anchor_u = None if utils is None else [
            (None if utils[i] is None else float(utils[i])) for i in order]
        super().__init__(**kw)

    def outside_range(self, work):
        """True if `work` is outside the measured span -- i.e. the reading is held."""
        return not (self.anchor_b[0] <= float(work) <= self.anchor_b[-1])

    def _interp(self, work, ys, log_y, proportional_outside=False):
        b = float(work)
        xs = self.anchor_b
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if b <= pts[0][0]:
            return pts[0][1] * (b / pts[0][0]) if proportional_outside else pts[0][1]
        if b >= pts[-1][0]:
            return pts[-1][1] * (b / pts[-1][0]) if proportional_outside else pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= b <= x1:
                f = (math.log(b) - math.log(x0)) / (math.log(x1) - math.log(x0))
                if log_y:
                    return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))
                return y0 + f * (y1 - y0)
        return pts[-1][1]

    def true_step_time(self, work):
        return self._interp(work, self.anchor_t, log_y=True,
                            proportional_outside=True)

    def true_utilization(self, work):
        if self.anchor_u is None:
            return super().true_utilization(work)
        return max(0.0, min(100.0, self._interp(work, self.anchor_u, log_y=False)))


#: umaperf0812 `c_controller`, VERBATIM. The only cell tied to a real measurement.
#:
#:     batch      100    165    272    449    741
#:     util %      52     44     49     42      -
#:     samples/s  57.7   46.5   45.1   28.2   24.3
#:
#: Step times are derived as B/sps, which is arithmetic on the observation rather than
#: a model of it. Occupancy at 741 was not recorded and is left None -- the
#: interpolator holds the last measured value and `outside_range` flags anything past
#: 741, so the cell cannot silently answer questions about batches nobody measured.
#:
#: THE OBSERVED OCCUPANCY IS NOT MONOTONE (44 -> 49 at 165 -> 272). That is kept.
#: Smoothing it away would remove the one place the real data says the relationship
#: is noisy, and a controller that needs U(B) to be monotone should fail here.
#:
#: !! THE OCCUPANCY HALF OF THIS TABLE IS CONTRADICTED BY AN INDEPENDENT SAMPLER. !!
#: Measured 2026-08-16 (F-038): wandb's own system monitor -- a separate thread, ~14 s
#: cadence, therefore NOT eval-blind -- reports, over the same batch sequence on the
#: same run:
#:
#:     batch          100    165    272    449    741
#:     ours (recent) 55.2   50.5   46.9   47.1   54.2
#:     wandb sys    100.0   69.7   71.3   89.4   86.6      <- minimum at 165, RISES at top
#:     samples/s     61.1   64.6   43.4   36.1   26.1
#:
#: So THROUGHPUT falling in batch is corroborated and OCCUPANCY declining in batch is
#: NOT. The in-process numbers rest on 2-5 ten-step rows per rung with a 900 s window
#: smearing across rungs ~11 minutes apart, i.e. the readings are not independent of
#: each other.
#:
#: The cell is KEPT AS IS, deliberately, and this is not stubbornness:
#:   * its purpose is a device where U does NOT reward growth while S punishes it, and
#:     BOTH samplers agree U does not reward growth -- wandb's U at the top rung (86.6)
#:     is below its bottom rung (100.0) too;
#:   * the trap-(a) verdict never reads U's LEVEL, only its direction against S's;
#:   * and a sandbox cell is a shape to design against, not a calibration.
#: But nothing here may be quoted as "occupancy declines with batch on UMA". That claim
#: is now one thin dev-box reading against one concurrent contradiction.
#:
#: Scope, and it is narrow: ONE arm, ONE route (UMA), ONE stage, no repeat, no measured
#: floor, and on host BB2 -- THE DEV BOX, not the cluster. This is a SHAPE to design
#: against, never a calibration.
UMAPERF0812_B = [100, 165, 272, 449, 741]
UMAPERF0812_SPS = [57.7, 46.5, 45.1, 28.2, 24.3]
UMAPERF0812_UTIL = [52.0, 44.0, 49.0, 42.0, None]


def umaperf0812(**kw):
    """The measured MLIP cell: occupancy and throughput both falling in batch."""
    return MeasuredDevice(
        batches=UMAPERF0812_B,
        step_times=[b / s for b, s in zip(UMAPERF0812_B, UMAPERF0812_SPS)],
        utils=UMAPERF0812_UTIL, **kw)
