"""
Synthetic GPU: a step-time model with a PLANTED knee and a PLANTED OOM ceiling.

WHY THIS EXISTS. increment_batch_size is a control law over a timing series. Its
only inputs are `_recent_step_times`, `_recent_step_work`, the OOM exceptions it
catches, and its own config -- it never touches the energy. So the honest way to
test it is to hand it a timing series whose correct answer is known in closed
form, which is exactly what a real run cannot provide: on the cluster the knee is
whatever the GPU does, and the controller's answer can only be compared against
itself. (The "knee at 10k+" that justified prod0810's growth ladder was never
measured.)

THE MODEL.

    t(B) = t_fixed + B / sps_max

Below B ~ sps_max*t_fixed the fixed overhead dominates and throughput RISES with
batch; above it, throughput asymptotes to sps_max and step time grows linearly.
That is the saturation shape a real accelerator has, and it is the regime the
step-time-regression gate was written for.

THE ANALYTIC ANSWER. The controller accepts a growth jump iff

    t(f*B) / t(B) <= 1 + tol

Substituting the model and solving for B (see knee_bound below) gives

    B_max = sps_max * t_fixed * tol / (f - 1 - tol)

as the largest batch from which a jump still pays. Every rung at or below B_max
should be taken; the first rung above it should fail and pin. That is a
decidable assertion, not a judgement call.

DISCRETENESS. Real step time is NOT smooth in B, and the smooth model above is
the easy case. Three separate effects put steps in it, all optional and all off
by default so the closed form stays exact when you want it:

  `tile`        Wave quantisation. GPU work dispatches in full waves of thread
                blocks, so a batch costs `ceil(B/tile)*tile` regardless of how
                much of the last wave it uses. Throughput is therefore SAWTOOTH
                in B, not monotone -- a rung landing just past a wave boundary
                pays for a whole wave it does not use.

  `recompile_s` torch.compile treats every distinct batch size as a new shape:
                one recompile + its own CUDA graph, ~30-60 s each on this
                codebase. Charged once per size ever seen, which is exactly the
                step after a growth jump -- i.e. the first measurement of the
                new rung.

  `regimes`     Discrete efficiency changes, as cuBLAS/cuDNN switch kernels past
                certain sizes. A step in `sps_max`, not in `t_fixed`.

With any of these on, `knee_bound`'s closed form no longer applies and
`expected_pin` walks the ladder against the actual cost model instead. That is
the honest ordering: the closed form is a special case of the walk, not the
other way round.
"""

import math
import random


class SyntheticGPU:
    """
    Deterministic step-time oracle with jitter.

    t_fixed     -- per-step overhead that does not scale with batch (s)
    sps_max     -- asymptotic throughput, samples/s
    oom_at      -- batch at or above which a step raises a CUDA-OOM-shaped error
    jitter      -- lognormal multiplicative noise, sigma in log space. The
                   controller medians the last 20 timings, so jitter is what
                   makes that median load-bearing rather than decorative.
    tile        -- wave quantisation: work costs ceil(B/tile)*tile. Makes
                   throughput SAWTOOTH in batch.
    recompile_s -- one-off cost the first time each distinct size is seen
                   (torch.compile's per-shape recompile + CUDA graph).
    regimes     -- [(batch_threshold, sps multiplier)], a discrete efficiency
                   change past a size, as cuBLAS/cuDNN switches kernels.

    The last three default to off, which keeps the model smooth and knee_bound's
    closed form exact. Turn them on to test the hard case -- and note that a test
    written without them is testing the easy one.
    """

    def __init__(self, t_fixed=2.0, sps_max=5000.0, oom_at=None, jitter=0.0,
                 tile=None, recompile_s=0.0, regimes=None, seed=0, host_frac=0.0):
        self.t_fixed = float(t_fixed)
        self.sps_max = float(sps_max)
        # fraction of t_fixed during which the GPU is IDLE (host-side graph
        # construction, collation, transfers). Sets the utilization floor at small
        # batch -- see `utilization`.
        self.host_frac = float(host_frac)
        self.oom_at = None if oom_at is None else int(oom_at)
        self.jitter = float(jitter)
        self.tile = None if tile in (None, 0) else int(tile)
        self.recompile_s = float(recompile_s)
        # [(batch_threshold, sps multiplier)], applied for the largest threshold <= B
        self.regimes = sorted(regimes or [], key=lambda r: r[0])
        self._rng = random.Random(seed)
        self._seen_sizes = set()
        self.n_oom = 0
        self.n_steps = 0
        self.n_recompiles = 0

    @property
    def is_smooth(self):
        return self.tile is None and not self.regimes

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
        """One observed step time. Raises a CUDA-OOM-shaped error past the ceiling.

        The message is matched by utils.is_cuda_oom, so the REAL
        handle_train_epoch_error accepts it -- the bench must not need its own
        OOM classifier, or it stops testing the one that ships.
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
            # first sight of this shape: one recompile + its own CUDA graph. This
            # lands on the FIRST step of a new rung, i.e. the first measurement
            # the growth gate would take of it.
            t += self.recompile_s
            self.n_recompiles += 1
        self._seen_sizes.add(size)
        if self.jitter > 0:
            t *= math.exp(self._rng.gauss(0.0, self.jitter))
        return t

    def throughput(self, work):
        """Samples per second at this batch, noise-free. Sawtooth when tiled."""
        return float(work) / self.true_step_time(work)

    def utilization(self, work):
        """
        GPU utilization percent at this batch.

        THE MODEL. Split the per-step fixed cost into a host-serial part (GPU idle:
        graph construction, PyG collation, host<->device transfers) and a part that
        keeps the GPU busy but does not scale with batch:

            busy(B) = t_fixed*(1 - host_frac) + B/sps_max
            util(B) = busy(B) / t(B)

        so utilization RISES with batch and saturates -- the batch-proportional work
        amortises a fixed serial cost. At B -> 0 it tends to (1 - host_frac), which is
        why host_frac sets the floor rather than utilization collapsing to zero.

        WHY THIS SHAPE. prod0810's uma arm measured 48% at batch 250, 54-62% at 1650
        and 75% at 2722-4491 -- monotone, saturating, and floored well above zero.
        A model where the GPU idles for ALL of t_fixed predicts ~14% at batch 250 and
        is plainly wrong. host_frac is the one parameter that fixes that, and it is
        exactly the quantity 'more host CPUs' is supposed to reduce.

        This is a SHAPE, not a calibration: like everything else in the bench it
        transfers mechanism, never numbers.
        """
        t = self.true_step_time(work)
        busy = self.t_fixed * (1.0 - self.host_frac) + self._effective_work(work) / self._sps(work)
        return 100.0 * min(1.0, busy / t)

    # ------------------------------------------------------- analytic answers

    def knee_bound(self, growth_factor, min_gain):
        """
        Largest batch from which a growth jump still clears the THROUGHPUT gate.

        The gate accepts a jump iff samples/sec improves by at least `min_gain`,
        because the objective is optimizer-step throughput at a fixed grad-accum
        target -- updates/sec = samples_per_sec / accum_target, so step time does
        not enter. (It previously bounded step-time REGRESSION, i.e. it optimised
        loop-iterations/hour; that gate rejected jumps buying +15% samples/sec for a
        43% slower step, which under this objective is a 15% win.)

        Derivation, with tau = t_fixed, s = sps_max, sps(B) = B*s/(s*tau + B):

            sps(f*B) / sps(B) >= 1 + g
            f * (s*tau + B) / (s*tau + f*B) >= 1 + g
            f*s*tau + f*B  >= (1+g)*s*tau + (1+g)*f*B
            s*tau*(f - 1 - g) >= f*B*g
            B <= s*tau*(f - 1 - g) / (f*g)

        NOTE HOW MUCH LARGER THIS IS. At f=1.65 the regression form with tol=0.25
        gave B_max = 0.625*s*tau; this gives 7.27*s*tau -- about 12x further up the
        ladder. That is not a bug, it is what the two objectives disagree about: a
        rung that costs step time but still buys samples/sec is worth taking here
        and was not before. OOM and max_batch_size become the binding constraints
        much more often as a result.

        THE TWO DEGENERATE ENDS ARE OPPOSITE, and they invert from the retired gate.
        Under step-time regression a BIGGER tol was more permissive; under a
        throughput gain a bigger g is STRICTER, because it demands a larger
        improvement. So:

            g <= 0        no improvement required -> every jump clears -> +inf
            f - 1 - g <= 0  the demanded gain exceeds what a factor-f jump can ever
                          deliver (throughput saturates at sps_max) -> NO jump ever
                          clears -> 0, i.e. it pins immediately at the floor.

        Returning +inf for the second case (as this did at first) is exactly
        backwards and reads as "the gate is disabled" when the gate is in fact
        rejecting everything. mk_dev's inherited warning that a too-large value means
        "every jump passes" carries the same inversion.

        SMOOTH MODEL ONLY. With tiling or regime switches there is no single
        bound -- the accept/reject test can flip back and forth along the ladder
        -- so this refuses rather than returning a number that quietly means
        nothing. Use expected_pin, which walks the real cost model.
        """
        if not self.is_smooth:
            raise ValueError(
                'knee_bound is the closed form for the SMOOTH model; this clock has '
                f'tile={self.tile}, regimes={self.regimes}. Use expected_pin(), which '
                'walks the ladder against the actual cost model.')
        f, g = float(growth_factor), float(min_gain)
        if g <= 0:
            return float('inf')          # nothing demanded -> every jump clears
        if f - 1.0 - g <= 0:
            return 0.0                   # more demanded than possible -> none clears
        return self.sps_max * self.t_fixed * (f - 1.0 - g) / (f * g)

    def expected_pin(self, base_batch, growth_factor, min_gain, max_batch=None):
        """
        The rung the controller should pin at, by WALKING THE LADDER against this
        clock's actual cost model. Works for smooth and discrete models alike;
        the closed form in knee_bound is the smooth special case of it.

        The walk mirrors the controller exactly:
          * rungs are `int(round(b * f))`, re-rounded at every step, so they
            drift off the ideal geometric series (1000*1.65^4 is 7412.06 but the
            rounded ladder reaches 7410);
          * the gate scores the jump FROM a rung, accepting iff
            `sps(next)/sps(b) >= 1+min_gain`, and pins at the rung it was standing
            on when a jump first failed.

        That last point is why the pin is systematically one growth factor hot: a
        rung is only convicted after the controller has moved past it, so the
        `prev_batch` it falls back to is itself already above the true bound.
        That is a property of the control law, not of this model.

        Returns None if no jump ever fails (the gate cannot pin).
        """
        f, g = float(growth_factor), float(min_gain)
        b = int(base_batch)
        for _ in range(200):                       # ladders are ~log_f(max/base) long
            nxt = max(b + 1, int(round(b * f)))
            if max_batch is not None and nxt >= max_batch:
                return int(max_batch)
            if self.throughput(nxt) < self.throughput(b) * (1.0 + g):
                return b                           # the jump from b failed -> pin b
            b = nxt
        return None
