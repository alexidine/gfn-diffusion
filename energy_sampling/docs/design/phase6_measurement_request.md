# Phase 6 measurement request — first cluster submission

Argument, in the `docs/PROTOCOL.md` sense: it records *why* this measurement is cut
this way, and is revised when the reasoning changes.

**This request does not design a controller.** Phase 6 is gated on Phase 4's
utilization proxy, which does not exist. What follows is the evidence Phase 6 needs
in hand *before* anyone writes a control law — specifically, the evidence that would
have made the previous generation's two failures visible as data rather than as a
production incident.

It **extends** [`benchmarks/registry.yaml`](../../benchmarks/registry.yaml) — 12
benchmarks, 6 suites — with four entries and one suite. It does not restate a field
that lives there, and it does not touch the schema.

---

## 0. Do the cheap thing first

Phase 4 admits three cases for the proxy. **Case 1 — an admin or the cluster docs
supply the actual statistic and window — costs one email and is worth between a third
and most of the budget below** (§5): it collapses three of the five arms outright, and
if the real window is shorter than 7200 s it shrinks every remaining job with it, since
that window is what sets the duration of all of them. Ask before submitting.

The specific questions, in the order they cut cost:

1. What statistic is averaged — `utilization.gpu` (NVML duty cycle), SM occupancy,
   memory-bandwidth utilization, or something per-node?
2. Over what window, and is it a **mean** or a **minimum-over-window**? A mean of
   55 % and a floor that no 15-minute block may fall below 40 % are different
   constraints and a controller built for one violates the other.
3. Is it evaluated continuously, or once at a checkpoint after N hours?
4. What is the threshold, and does it vary by partition or QoS?
5. Is the cancellation reason recorded anywhere a user can read — `sacct`, a mail,
   the job's stderr?

If all five are answered, this becomes a much smaller request: only §8's trap work
survives, and the sensor cross-calibration (§6) drops to a single confirmation arm.
**Write down which case applies either way** — a proxy adopted under case 3 and later
remembered as case 1 is how a margin quietly becomes a law.

---

## 1. The three quantities Phase 6 needs

Nothing else. A controller that has these can be designed; one that lacks any of
them is guessing, which is what the last one did.

| # | Quantity | Why Phase 6 cannot start without it |
|---|---|---|
| **Q1** | `U(B)` — occupancy as a function of batch, on **both** an analytic-energy route and an MLIP route, each with a measured floor | Priority 1 is "satisfy the utilization requirement". A controller that actuates batch against occupancy needs to know whether batch moves occupancy **at all**. This is trap (a) as a measurement instead of an assumption. |
| **Q2** | `t(B)` and `S(B)` — step time and samples/sec over the **reachable** batch range, bracketed on both sides, with a measured floor | Priority 2 is "maximize throughput subject to it". Whether the curve is FLAT, KNEED or NON-MONOTONE decides whether a walk is even the right instrument. This is trap (b), and the never-measured knee. |
| **Q3** | The proxy — how `gpu/util_policy`, `gpu/util_recent` and external `nvidia-smi` relate to each other and to cluster-visible outcome | Phase 4's stated deliverable, and the thing Phase 6 is gated on. |

Q1 and Q2 come off the same jobs — a batch ladder is simultaneously an occupancy
ladder and a throughput ladder. Q3 rides along on every job at near-zero cost.

---

## 2. The sensor, mechanically — and why two of the three readings are one reading

This is the single most important thing to get right before submitting, because it
changes what "they disagree" is capable of meaning.

**`gpu/util_recent` and `gpu/util_policy` are not two sensors. They are two window
lengths over one sample deque, filled by one function.** `_gpu_util_mean(900)` and
`_gpu_util_mean(7200)` both read `self._gpu_util`, appended to by
`_sample_gpu_util()` ([train.py:325-410](../../train.py#L325)). They cannot disagree
because of sensor error, calibration, or instrument choice. **Their difference is a
trend estimator and nothing else.**

Three consequences follow immediately, and all three are traps in their own right:

- On any run **shorter than 900 s the two are numerically identical** — every sample
  falls in both windows. On a run under 7200 s they differ only by whatever samples
  predate the 900 s mark. So "policy agrees with recent" on a short run is
  *guaranteed by construction* and carries **zero** information. An analyst who reads
  it as corroboration has been reassured by an identity. This is the
  swallowed-diagnostic shape this project has paid for before.
- **`nvidia-smi` sampled from outside the process is the only genuinely independent
  reading available.** It is therefore not optional on the first submission.
- Because the quantity is a time-average over a trailing window, it **cannot be
  reconstructed after the fact**. Any external check must be *concurrent* or it is
  not a check.

### The four mechanisms that can make the in-process series wrong

Each is invisible without the external sampler, and each has a *direction*.

**(i) Phase bias.** `_sample_gpu_util()` is called at exactly one point in the step
body — [train.py:2184](../../train.py#L2184), after the step is timed, before
`increment_batch_size`, before `ten_step_reporting`. The 60 s gate throttles *how
often* it fires, never *where in the step* it fires. NVML `utilization.gpu` is itself
a short-window duty cycle (~1 s), so a reading taken at a systematically host-heavy
or GPU-heavy instant is systematically biased. At 200 s/step the gate never blocks
and the sensor fires **once per step at the identical phase** — the worst case.
*Direction: unknown, which is why it must be measured.*

**(ii) Eval blindness — the dangerous one.** The sampler sits in the training portion
of the loop body. The eval block, figure logging, `checkpointer.link`, and the
periodic archive all execute *later in the same iteration*
([train.py:2196-2270](../../train.py#L2196)). **A 300 s eval contributes zero
samples.** This project's own profiling puts a large share of eval in CPU-only work
(numpy pairwise metrics, kaleido PNG re-render, scipy-KDE violins). So the in-process
series omits the run's **least-occupied minutes while the scheduler counts them**.

*Direction: `gpu/util_policy` **overstates** what the scheduler sees.* That is the
dangerous direction — the metric reads safe while the job sits closer to
cancellation than it looks.

The magnitude is computable from series that are already logged: at `eval_period`
250 with an eval costing `eval_step_time`, the unsampled share is
`eval_step_time / (inter_eval_time + eval_step_time)`. **It must be measured, not
assumed** — and measured on a run shape that actually evaluates, which brings up the
next point.

**(iii) The benchmark shape hides (ii) entirely.** Every throughput benchmark in the
registry sets `eval_period: 100000000` — eval off, by design, because eval is not the
work being timed. **A proxy calibrated on those arms does not transfer to production
runs, which are the runs that get cancelled.** The submission therefore needs one arm
at production eval cadence whose only job is to expose this gap.

**(iv) Source ambiguity.** `_read_gpu_util()` tries `torch.cuda.utilization()` and
falls back to `gpu_guard.gpu_memory()[3]` (shells out to `nvidia-smi`) — and
**nothing records which one answered** ([train.py:374-394](../../train.py#L374)). In
a container without pynvml the entire series is nvidia-smi; with it, NVML. If NVML
raises transiently the series silently *mixes two instruments under one metric name*.
The known precedent: the sensor was already found completely inert once, on a missing
optional dependency, with one line in a log nobody read.

### And the mean itself

`_gpu_util_mean` is an **unweighted arithmetic mean of point samples**, with a hard
floor of 5 samples ([train.py:396-410](../../train.py#L396)) below which it returns
`None`. It is not time-weighted, so a stretch of long steps contributes fewer samples
than its share of wall clock. And **the sample count backing any row is never
logged** — a `gpu/util_policy` row is a number whether it rests on 5 samples or 120.

> **A defect this exposes in the existing registry.**
> `a100-batch-scaling-elj` declares `gpu/util_policy` as primary and
> `min_wallclock_s: 7200`, which `_validate_work` checks. But its budget is
> `epochs_formula: resume_step + warmup_steps + measure_steps` = 500 steps. At an
> ELJ fused step of ~1 s that job ends in **~8 minutes**, and
> `_gpu_util_mean(7200)` will happily return the mean of its ~8 samples **labelled
> as the 7200 s policy average**. The validator's guard is a *declaration*, not a
> measurement: nothing checks that the step budget delivers the wall clock. This is
> the same shape as both named traps — a rule that is declared and not enforced.
> §4 and §9 fix it, and the fix requires the sample-count instrumentation below.

---

## 3. Instrumentation gaps — these are code changes, and they block interpretation

Named explicitly because new instrumentation is a code change and must never be
assumed. **The first three are cheap, are in `train.py` (which this session owns),
and should land before the submission, not after** — without them a disagreement
between the in-process series and the external sampler is *uninterpretable*, and 90
GPU-hours produce an ambiguity instead of an answer.

| Gap | What | Cost | Blocks |
|---|---|---|---|
| **G1** | `gpu/util_source` — which sensor answered, per report (`0` nvml / `1` smi / `2` mixed-in-window) | one line | Every row of the disagreement table. Without it, "in-process ≠ external" cannot be told from "in-process *is* nvidia-smi and something else is wrong". |
| **G2** | `gpu/util_n_recent`, `gpu/util_n_policy` — sample count backing each window | two lines | Whether a disagreement is real or a 5-sample coin flip. Also the honest enforcement of `min_wallclock_s` (§2 box). |
| **G3** | `gpu/util_sampled_wallclock_frac` — share of the window's wall clock lying within ±`period/2` of a sample | ~5 lines | The **direct** measure of eval blindness (ii). This one number decides whether `gpu/util_policy` overstates, and it is the difference between case 2 and case 3. |
| **G4** | `batch/oom_min` is logged at [train.py:878](../../train.py#L878) and is **absent from the registry metric catalogue** | registry only, no code | A benchmark cannot legally name it today. Registry gap, not an instrument gap. |
| **G5** | Controller decision series — rung throughput, the ratio tested, the rung rejected — currently exist only as prints | moderate | Needed **only** for the shadow-mode arm (§8). wandb uploads no console log for a run left in state `crashed`, so a scancelled job loses its entire account of itself — which is exactly the job you most need to read. |
| **G6** | UMA has no phase split (`AL_mace_utils.drain_mace_phase_timing` has no UMA counterpart) | `mxtaltools/`, not owned here | Already recorded in `benchmarks.md` §10. **Not blocking for Phase 6** — it blocks Phase 5.0. Listed so it is not rediscovered. |

**G1–G3 are the prerequisite.** If only one can land, make it **G3**.

---

## 4. What must be logged, and at what cadence

### 4.1 Already logged — no code change

Everything in the registry's `metrics` catalogue. The load-bearing subset:

| Group | Literals | Cadence |
|---|---|---|
| Cost | `train_step_time`, `samples_per_sec`, `step_time_max10`, `Batch Size` | 10-step report |
| Occupancy | `gpu/util_recent` (900 s), `gpu/util_policy` (7200 s), `vram/peak_reserved_mb`, `vram/cached_mb` | 10-step report, over 60 s samples |
| Energy | `energy/frac_of_step`, `energy/frac_outside_step`, `energy/ms_per_sample`, `energy/calls` | 10-step report |
| Eval | `eval_step_time`, `eval_sampling_time`, `eval_figs_time`, `inter_eval_time` | per eval |
| Catastrophe | `batch/oom_events`, `batch/ceiling_expiries`, `batch/oom_ceiling`, `gradnorm/nonfinite_steps` | 10-step report, **counts** |

> **Read the occupancy rows correctly.** `gpu/util_*` is *emitted* on the 10-step
> grid but *sampled* on a 60 s grid. At 1 s/step, six consecutive rows carry the same
> underlying mean. Treating rows as independent samples understates the spread by
> roughly √6. G2 makes this checkable instead of a thing you have to remember.

### 4.2 External, per job — not wandb

Launched by the job script, **before** `train.py` and killed after it, writing beside
the run directory. This is a *job-level* observation, not a run-level one.

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \
           --format=csv,nounits -l 10 > util_smi_${SLURM_JOB_ID}.csv &
SMI_PID=$!
```

- **Cadence 10 s.** Over a 7200 s window that is ~720 samples against the in-process
  ~120. The margin is what lets it adjudicate rather than merely add a third opinion.
- It costs no GPU work and it **runs during eval**, which is the entire point.
- `clocks_throttle_reasons.active` and `power.draw` are included because they
  separate "the GPU is idle" from "the GPU is throttled" — two states that produce
  the same utilization number and call for opposite responses.
- **It must span the whole job**, including startup, eval, and the final checkpoint
  write, so the denominator matches the scheduler's.

Also captured per job, into a small sidecar the run can be joined on:

| Field | Source | Why |
|---|---|---|
| `SLURM_JOB_ID`, `SLURM_NODELIST`, `hostname` | env | §5's node-confound question is unanswerable without it |
| `sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList,Reason,Comment` | post-hoc, in the epilogue | The only place a cancellation reason may appear |
| `scontrol show job $SLURM_JOB_ID` (pre-flight) | job script | QoS/partition, in case the policy varies by them |
| `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` | job script | The rung ladder is meaningless without the card's memory |
| `compile_policy` resolved value, printed by `maybe_compile_policy` | run log | Whether inductor actually engaged. A ladder where it engaged on some rungs and not others is not a ladder. |

**On a cancelled job, the epilogue capture is the whole record.** Write `sacct` and
the smi CSV to disk in a `trap`/epilogue that survives `scancel`, because wandb
uploads no console log for a run left in state `crashed`.

---

## 5. Repeats, and how they are laid out

The method is fixed by the registry and is not up for redesign: `repeat_launch`
(separate process launches, never re-timing inside one process), run statistic =
**median** of the primary metric over the measurement window, dispersion =
**relative span** `(max−min)/|median|`, minimum 3.

| Arm | Repeats | Layout | Reasoning |
|---|---:|---|---|
| Ladder rungs, general | **3** | separate jobs | The registry's declared minimum. A span from 3 points is what was observed; it can only widen honestly. |
| Ladder **anchor** rungs (lowest and highest) | **5** | separate jobs | The whole ladder rests on these two floors. A 3-repeat span at an anchor that turns out large invalidates every 3-repeat span in between, and re-running the ladder costs more than two extra jobs. |
| Sensor cross-calibration | **3** per cadence | separate jobs | Comparing two cadences, not detecting a small effect. |
| Production-shape (eval on) | **3** | separate jobs | The quantity (G3, unsampled wall-clock share) is nearly deterministic given the eval cadence; repeats guard the launch, not the statistic. |
| Node confound | **3 + 3** | 3 pinned to one node, 3 forced across distinct nodes | Two floors, and their ratio answers `benchmarks.md` §10's open question directly. |

**Seed policy: `fixed` throughout.** These are cost benchmarks — the question is
"what does this operating point cost", not "is this operating point better", so the
draw must not enter. The registry records `seed_policy` per benchmark and every
existing entry declares `fixed`; these follow.

**The node question is a real confound, not a formality.** A floor measured across
launches that all happened to land on one node understates a floor across launches
that did not, and every downstream comparison inherits that. If the cluster does not
pin nodes, `SLURM_NODELIST` must be recorded per repeat and a floor built from
single-node repeats **labelled as a within-node floor**.

### Budget

Sized from the constraint that dominates everything: **`gpu/util_policy` is
meaningless below 7200 s**, so every occupancy job is ≥2 h regardless of how fast the
measurement window completes.

| Arm | Jobs | Hours each | GPU-h |
|---|---:|---:|---:|
| Pre-flight (rung discovery + step-time sizing) | ~8 | 0.25 | **2** |
| A · ELJ ladder, 5 rungs (2 anchors × 5, 3 × 3) | 19 | 2 | **38** |
| B · UMA ladder, 4 rungs (2 anchors × 5, 2 × 3) | 16 | 2 | **32** |
| C · Sensor cross-calibration, 2 cadences × 3 | 6 | 2 | **12** |
| D · Production shape, eval on | 3 | 4 | **12** |
| E · Node confound, 3 + 3 | 6 | 2 | **12** |
| | **58** | | **≈ 108 GPU-h** |

≈ 4.5 days on one A100; ~1 day at 4-way concurrency.

**~95 % of that wall clock exists only to fill the 7200 s occupancy window** — the
throughput measurements themselves complete in minutes. That is the price of not
knowing the scheduler's statistic, and it is exactly why §0 comes first. If an admin
answers, arms C, D and E collapse to one confirmation job and the budget falls to
~74 GPU-h; if the window turns out shorter than 7200 s, every job shrinks with it.

### If only part of this can be run

In priority order, because a partial submission should still answer something whole
rather than answer everything partially:

1. **Arm B (UMA ladder) — 32 GPU-h.** The only arm that tests trap (a)'s premise on
   the route where it actually fired. If exactly one arm runs, run this one.
2. **Arm D (production shape) — 12 GPU-h.** Delivers G3 and therefore the *direction*
   of the proxy's error. A conservative proxy and an overstating proxy call for
   opposite responses, and this is the arm that tells them apart.
3. **Arm A (ELJ ladder) — 38 GPU-h.** Trap (b) and the never-measured knee, on the
   route where batch is *believed* to be the occupancy lever.
4. **Arm C, then E.** Refinements: C says whether a cadence change would fix a
   disagreement, E says how much to trust every floor above.

Arms B and D together are **44 GPU-h** and answer both traps' *premises* — which is
the difference between Phase 6 starting from evidence and starting from a belief.

---

## 6. Concurrency: yes, and the reason is mechanical

**All three readings must be captured on the same job, over the same wall clock.**
Not on comparable jobs, not sequentially.

1. **The quantity is a trailing time-average.** It cannot be reconstructed after the
   fact from anything else the run logged. A non-concurrent nvidia-smi run measures a
   *different two hours*.
2. **Two of the three are the same series** (§2). Without the external sampler there
   is no independent reading at all, and the two in-process readings will agree by
   construction on exactly the short runs where agreement is most tempting to
   believe.
3. **The failure modes are per-interval, not per-run.** Eval blindness and phase bias
   corrupt *specific intervals* of the series. Detecting them requires aligning
   in-process samples with external samples **on a common clock**, then asking what
   the GPU was doing during the intervals the in-process sensor did not sample. A
   run-level average of each cannot do this; the difference of two run-level averages
   is one number with several explanations.
4. **Cancellation is an event with a timestamp.** Relating a cancellation to the
   trace requires both series to be timestamped against the same wall clock as
   `sacct`'s `Elapsed`.

**One design point earns its own job.** Arm C runs the *shipped* sensor at
`gpu_util_sample_period_s: 60` and a second arm at `10`, everything else identical.
This turns an otherwise ambiguous disagreement into a decidable one:

- 10 s in-process agrees with smi, 60 s does not → **the cadence is too coarse**.
  Fix: lower `gpu_util_sample_period_s`. Cheap, and no proxy margin needed.
- Both in-process cadences disagree with smi *in the same direction* → **phase bias
  or eval blindness**, i.e. structural. A cadence change will not fix it, and the
  proxy needs a correction or a margin.

Without that arm, a single disagreement admits both explanations and the submission
has to be repeated.

---

## 7. The disagreement decision table

`P` = `gpu/util_policy` (7200 s, in-process) · `R` = `gpu/util_recent` (900 s,
in-process) · `S` = external nvidia-smi, time-weighted over the matching window.

Every row names the **second explanation**, because a table whose conclusions are
unique-by-assertion is how a proxy becomes a law.

| # | Pattern | Conclude | Second explanation to rule out | Do |
|---|---|---|---|---|
| 1 | `P ≈ R ≈ S`, job **survives**, ≥7200 s elapsed | The proxy tracks the GPU **in this regime**. Adopt `gpu/util_policy` under **case 2**. | The run never entered the regime where the mechanisms bite — no eval, uniform step time. Check G3 ≈ 1. | Record the case **and its regime**: route, batch, T, step time, eval cadence. It does *not* license the proxy at 200 s/step if measured at 2 s/step. |
| 2 | `P ≈ R ≈ S`, job **cancelled anyway** | The proxy tracks the GPU but **not the policy**. The scheduler's statistic is not mean `utilization.gpu`. | The cancellation was unrelated — preemption, walltime, node failure, OOM-kill. | `sacct` `Reason`/`State` first. If genuinely a utilization cancellation, **case 2 is refused**; go to case 3 and state the margin. This is the highest-information outcome in the table. |
| 3 | `S > P` and `S > R` (in-process reads **low**) | The in-process sampler misses busy time — phase bias sampling at a host-heavy instant. | NVML vs nvidia-smi instrument difference — **this is what G1 exists to exclude**. | The proxy is **conservative**: safe to adopt, but it costs throughput. Quantify the margin in samples/sec so the price of the ignorance is known. |
| 4 | `P > S` and `R > S` (in-process reads **high**) | **The predicted eval-blindness direction.** The in-process series omits low-occupancy stretches the scheduler counts. | Long non-eval stalls also outside the sample points — checkpoint writes, buffer churn, allocator growth. Distinguish via G3 against `eval_step_time / (inter_eval_time + eval_step_time)`: if they match, it is eval. | **`gpu/util_policy` is not admissible as the proxy unmodified.** Either the external sampler becomes the proxy, or the in-process reading is discounted by the measured unsampled share. Case 3 with a stated, measured margin. |
| 5 | `P > R` (long window above short) | **Utilization is falling.** Not a sensor disagreement at all — one series, two windows. | A stage transition or a batch change inside the window. Read against `Batch Size` and the stage marker. | This is prod0810's uma-arm shape (hourly means 75/62/54/49/48/48). A run that starts busy and settles idle **passes an early check and fails a later one** — the controller must eventually watch the trend, not the level. Record it; do not act on it yet. |
| 6 | `R > P` (short window above long) | **Utilization is rising** — warm-up, a transition, or a batch change. Nothing about the sensor. | The 7200 s window is not yet full and still carries the launch. | Discard the **first 7200 s** of every job for proxy purposes. `P` is not meaningful until its window has filled once. |
| 7 | `P` absent, `R` present | The job is too young: fewer than 5 samples in 7200 s while ≥5 in 900 s is impossible, so this means the run is short. | — (mechanically unambiguous) | **Not a fault.** No policy claim is available from this job. This is the registry's `min_wallclock_s` rule doing its job. |
| 8 | `R` absent, `P` present | Fewer than 5 samples in the trailing 900 s ⇒ steps longer than ~180 s, **or the loop stalled** — a long eval, an OOM recovery, a checkpoint write. | Genuinely slow steps on an MLIP route, which is expected, not pathological. Separate via `train_step_time`. | **Absence is the signal.** Every `R`-absent interval is an interval the sensor could not see — and those are precisely the intervals that bias `P`. Read `S` over exactly that interval. |
| 9 | `P` and `R` both absent, `S` present | The in-process sensor is **off** — both NVML and nvidia-smi failed inside the process, printed once, then silence. | The job never reached 5 samples at all (very short, or crashed early). | The external sampler is the **only** evidence for this job. This row alone justifies making it mandatory. Fix the sensor before re-running. |
| 10 | `S` absent, `P`/`R` present | The side-sampler died or was never launched — container, `srun` wrapper, or a `trap` that did not fire. | The CSV was written somewhere the epilogue did not collect. | The job still contributes to Q1/Q2 (`t(B)`, `S(B)`) but **not** to Q3. **State the exclusion**; do not quietly average it in. |
| 11 | All three disagree pairwise, no pattern | Sampling noise dominates. | A genuinely non-stationary workload — the mean is not the right summary. Check the smi time series for structure. | **G2 decides it.** If `S` (~720 samples) is stable while `P`/`R` scatter, the in-process cadence is too coarse — lower `gpu_util_sample_period_s`. Arm C answers this directly. |
| 12 | All three agree and are **low**, job survives | The threshold is lower than believed, **or the policy is not enforced on this partition/QoS**. | One job is not evidence about a policy. | `sacct` across **all** jobs in the submission before concluding anything. A single surviving low-utilization job licenses nothing. |
| 13 | `U(B)` **flat or declining across every rung**, jobs cancelled at every rung | **Trap (a)'s premise measured false in the only setting that matters** — occupancy is not addressable by batch on this route. | The rungs did not span enough range; the OOM ceiling was hit before any effect could appear. Check the ladder brackets a real maximum (§8). | **The strongest possible result.** It removes batch-vs-occupancy from Phase 6's design space entirely and redirects it to the levers that make the GPU busier per unit wall time. |

---

## 8. Making the two traps detectable — before a controller exists

### Trap (a): an occupancy rule measured false that still outranked the throughput gate

The rule grew batch 100 → 741 on a UMA arm while utilization went 52 → 42 % and
samples/sec fell 58 %. It was priority 1, so it overrode the throughput gate, which
would have refused every one of those jumps. It has been **deleted**, not repaired —
`gpu_util_floor` is a retired key.

**What makes it detectable this time is that `U(B)` is measured as data, on a ladder
with the controller off, before any rule is written.** Three requirements:

1. **The ladder must exist on the MLIP route.** The registry has
   `a100-batch-scaling-elj` and **no MLIP counterpart**. The trap fired on a UMA arm.
   `benchmarks.md` §4 already says batch is the occupancy lever on toy/ELJ and is not
   on MLIP, and that "both statements are true and neither generalises". Today
   nothing would catch it. `a100-batch-scaling-uma` (§9) is the single most important
   entry in this request.
2. **Flat must be distinguishable from noisy.** A rung's occupancy mean needs a
   measured floor like any other primary metric — `gpu/util_policy` is declared
   primary on both ladders, so `_validate_floor` requires the floor to cover it. With
   ~120 in-process samples and ~720 external samples per 2 h rung, and the previously
   observed sd of ~6.5 points, per-rung means are tight enough to resolve the ~10
   point effect that was actually observed. **If the measured span turns out to
   exceed the rung-to-rung differences, the correct finding is "occupancy is not
   resolvable in batch on this route", which is itself a complete answer to trap (a).**
3. **The finding must be recorded as evidence with its scope line**, not as a
   controller rule. `findings.md`, graded, with T, route, stage, batch range.

### Trap (b): a knee walk with no floor that descends forever under flat throughput

Under the shipped throughput gate (`batch_growth_min_throughput_gain: 0.05`) with
flat throughput, every jump fails, the recheck drops one rung and re-climbs into the
same failing comparison, and the batch ratchets down forever — 1000 → 606 → 367 → …
`_batch_floor()` exists precisely to stop this and is documented as load-bearing.

**But the floor has never been validated against a real `t(B)`, and the floor equals
`args.batch_size`, which is also the starting point** — which is the same quantity
the gate is known to be path-dependent on (1000 pins at 1650 ≈ 40 % of best; 2722
pins at 4491 ≈ 67 %; 7410 pins at 7410 ≈ 84 % on a synthetic non-monotone curve).

**What makes it detectable is measuring the *shape* of `t(B)` and `S(B)` directly,
with no walk in the loop.** Requirements:

1. **The rung ladder must be measured, not assumed.** The existing ladder
   `[1000, 1650, 2722, 4491, 7410]` is, by `benchmarks.md` §10's own admission,
   copied from observed cluster batch sizes — "a starting grid, not a measured
   range". **Replace it with a pre-flight**: bisect to the OOM ceiling at fixed `T`,
   measure step time at two anchors, then place ≥5 geometric rungs across
   [`batch_size`, ceiling]. This is what stops the never-measured knee recurring —
   the last generation's 10k+ knee was an artifact of three accounting bugs, not a
   property of the hardware, and nothing in the current ladder would have caught that.
2. **A knee claim requires bracketing rungs on both sides.** If `samples_per_sec` is
   still rising at the top rung, the finding is **"no knee below the OOM ceiling"** —
   *not* "the knee is at the top rung". This is the clause that forbids inferring a
   knee from a monotone climb that merely ran out of memory.
3. **The ladder must probe above a candidate optimum, not only below.** The recheck's
   known structural failure is that it only ever retests *downward*, so it cannot
   escape a bad pin. A ladder that spans the full reachable range is immune to this
   by construction — and it is the only way to learn whether the real `t(B)` is
   non-monotone enough for path-dependence to matter, which is currently **unknown**.
4. **Classify the curve explicitly** as FLAT / KNEED / NON-MONOTONE against the
   measured floor, and record which. Each implies a different Phase 6 instrument:
   FLAT says a walk is the wrong tool and the floor is doing all the work; KNEED says
   a walk is appropriate and where it should stop; NON-MONOTONE says a two-point
   local gate is unsafe at any starting batch.

### Shadow mode — optional, and only if G5 lands

One arm with `grow_batch_size: true` at production settings, logging every decision
the controller makes *and the rung measurements behind it*, so an alternative Phase 6
controller can be scored offline in `bench/` against a **real** trace instead of a
synthetic cost model. This is the natural bridge from a measured `t(B)` to a
controller that was never live-tested on the cluster.

**Not part of the first submission.** It needs G5, it is not a cost benchmark (the
batch moves by design, so it violates the fixed work quantity), and it must not be
mixed into the ladder jobs. Recorded here so it is not rediscovered as a new idea.

---

## 9. Registry extension

Four new benchmarks, one new suite, one catalogue fix. Written to fit the existing
schema — no new `work.kind`, no new blocks, no schema version bump.

### 9.0 Catalogue and existing-entry fixes

1. **Add `batch/oom_min`** to `metrics.catastrophe` (G4). It is logged at
   [train.py:878](../../train.py#L878) and no benchmark can legally name it today.
2. **Amend `a100-batch-scaling-elj`** (§2 box): `batch_rungs` becomes the *measured*
   ladder from the pre-flight rather than the copied one, and `epochs_formula` is
   sized so the step budget actually delivers `min_wallclock_s`. Add the liveness
   assertion below to every occupancy benchmark:

   > `gpu/util_policy` is backed by at least `0.8 * min_wallclock_s /
   > gpu_util_sample_period_s` samples — i.e. the window genuinely filled, rather
   > than a partial window printing a number.

   **This assertion depends on G2.** Until `gpu/util_n_policy` exists it cannot be
   checked, which is the concrete reason G2 is a prerequisite and not a nice-to-have.
3. `noise_floor.repeats` on `a100-batch-scaling-elj` rises 3 → 5, matching §5's
   anchor-rung reasoning.

### 9.1 New entries

```yaml
  - id: a100-batch-scaling-uma
    title: A100 batch and occupancy scaling, UMA (the MLIP counterpart)
    status: specified
    workload:
      energy_function: uma
      conditioning: unconditional
      domain: molecule
      space_groups: [2]
      z_primes: [1]
      T: 10
      dataset: 'mipcas_sg2_zp1 prior + UMA weights (mlip_path)'
      dataset_in_repo: false
    training_mode:
      stage: equilibration
      train_mode: fused
      bwd_sampling_mode: prior
      branches: [fwd, bwd, replay]
    hardware:
      class: a100
      local_adequate: false
      a100_required: true
      reason: >-
        the question is what the cluster scheduler judges on the route where the
        energy call dominates. The occupancy floor that had to be deleted was
        measured false on a UMA arm, and no benchmark in this registry would have
        caught it -- the only ladder is on an analytic energy, where batch IS
        believed to be the occupancy lever.
    work:
      kind: fixed_steps_per_rung
      batch_size: null
      batch_rungs: null          # MEASURED by the pre-flight, never copied
      pin_batch: true
      resume_step: null
      warmup_steps: 25
      measure_steps: 200
      epochs_formula: >-
        resume_step + warmup_steps + ceil(min_wallclock_s / measured_step_time)
        rounded up to a multiple of 10, per rung, one launch each
      wallclock_cap_s: 28800
      min_wallclock_s: 7200
    overrides:
      energy_function: uma
      integrator: {T: 10}
      eval_T: 10
      eval_period: 100000000
    metrics:
      primary: [gpu/util_policy, samples_per_sec, train_step_time, energy/frac_of_step]
      secondary: [gpu/util_recent, energy/ms_per_sample, energy/calls,
                  vram/peak_reserved_mb, vram/cached_mb, step_time_max10]
      catastrophes: [batch/oom_events, batch/ceiling_expiries, batch/oom_ceiling,
                     gradnorm/nonfinite_steps]
      unusable:
        energy/mace_host_frac: 'the phase split is instrumented on the MACE route only'
    liveness:
      - 'each rung ran at its declared batch for the whole window (Batch Size flat)'
      - 'the run lasted at least min_wallclock_s, so the gpu/util_policy window filled'
      - 'gpu/util_policy is backed by >= 0.8 * min_wallclock_s / gpu_util_sample_period_s samples'
      - 'fwd/*, bwd/* and replay/* all present and changing'
      - 'energy/calls > 0'
      - 'the neighbour-list fast path is TAKEN (a silent torch_cluster fallback reads as a slow GPU)'
      - 'compile_policy resolved the same way on every rung'
      - 'the external nvidia-smi CSV covers the whole job, startup and eval included'
    noise_floor:
      method: repeat_launch
      repeats: 3
      seed_policy: fixed
      run_statistic: median
      dispersion: relative_span
      measured: null
    correctness:
      reference: control_comparison
      detail: >-
        UMA is not bit-reproducible on GPU; same-path repeat spread is measured on the
        same batch inside the same test and cross-path spread must not exceed it by
        more than a small factor. torch.equal is refused.
      gate: 'mxtaltools/tests/test_uma_gpu_real_batches.py'
      exactness: floor
    comparison:
      valid_against: [a100-batch-scaling-uma]
      rule: within_suite_rungs
    purpose:
      - 'the occupancy-versus-batch curve on the route where the deleted floor was measured false'
      - 'batch is NOT believed to be the occupancy lever here; that belief is what this benchmark tests'
```

```yaml
  - id: a100-util-sensor-crosscal
    title: Utilization sensor cross-calibration, in-process versus nvidia-smi
    status: specified
    workload:
      energy_function: elj
      conditioning: unconditional
      domain: molecule
      space_groups: [2]
      z_primes: [1]
      T: 10
      dataset: 'mipcas_sg2_zp1_elj_prior_dataset.pt'
      dataset_in_repo: false
    training_mode:
      stage: equilibration
      train_mode: fused
      bwd_sampling_mode: prior
      branches: [fwd, bwd, replay]
    hardware:
      class: a100
      local_adequate: false
      a100_required: true
      reason: >-
        the local box has no scheduler, so it cannot answer what the cluster judges;
        and the disagreement being characterised is between a sensor sampled inside
        the training loop and one sampled outside the process on cluster hardware
    work:
      kind: fixed_steps
      batch_size: 1000
      pin_batch: true
      resume_step: null
      warmup_steps: 100
      measure_steps: 400
      epochs_formula: >-
        resume_step + warmup_steps + ceil(min_wallclock_s / measured_step_time)
        rounded up to a multiple of 10
      wallclock_cap_s: 14400
      min_wallclock_s: 7200
    overrides:
      energy_function: elj
      integrator: {T: 10}
      eval_T: 10
      batch_size: 1000
      max_batch_size: 1000
      eval_period: 100000000
    metrics:
      primary: [gpu/util_policy, samples_per_sec]
      secondary: [gpu/util_recent, train_step_time, vram/peak_reserved_mb]
      catastrophes: [batch/oom_events, gradnorm/nonfinite_steps]
      unusable: {}
    liveness:
      - 'two arms ran, identical but for gpu_util_sample_period_s (60 shipped, 10 probe)'
      - 'the external nvidia-smi CSV covers the whole job at 10 s cadence'
      - 'gpu/util_policy is backed by >= 0.8 * min_wallclock_s / gpu_util_sample_period_s samples'
      - 'the sensor SOURCE is recorded for the whole run (gpu/util_source constant)'
      - 'fwd/*, bwd/* and replay/* all present and changing'
    noise_floor:
      method: repeat_launch
      repeats: 3
      seed_policy: fixed
      run_statistic: median
      dispersion: relative_span
      measured: null
    correctness:
      reference: paired_control
      detail: >-
        the two cadences are the paired control for each other: identical work, one
        knob apart, so a difference between them is the SAMPLING and not the workload
      gate: null
      exactness: floor
    comparison:
      valid_against: [a100-util-sensor-crosscal]
      rule: same_hardware_class_paired_seed
    purpose:
      - 'gpu/util_recent and gpu/util_policy are two windows over ONE deque; nvidia-smi is the only independent reading'
      - 'separates "the in-process cadence is too coarse" from "the in-process sampler is structurally biased"'
```

```yaml
  - id: a100-util-production-shape
    title: Utilization on a production-shaped run, with eval ON
    status: specified
    workload:
      energy_function: elj
      conditioning: unconditional
      domain: molecule
      space_groups: [2]
      z_primes: [1]
      T: 10
      dataset: 'mipcas_sg2_zp1_elj_prior_dataset.pt'
      dataset_in_repo: false
    training_mode:
      stage: equilibration
      train_mode: fused
      bwd_sampling_mode: prior
      branches: [fwd, bwd, replay]
    hardware:
      class: a100
      local_adequate: false
      a100_required: true
      reason: >-
        the proxy has to be calibrated on the run shape that actually gets cancelled.
        Every throughput benchmark disables eval by design, and the in-process
        occupancy sampler takes NO samples during an eval block, so a proxy validated
        on an eval-free arm does not transfer to a production run.
    work:
      kind: fixed_steps
      batch_size: 1000
      pin_batch: true
      resume_step: null
      warmup_steps: 100
      measure_steps: 400
      epochs_formula: >-
        resume_step + warmup_steps + ceil(min_wallclock_s / measured_step_time)
        rounded up to a multiple of 10
      wallclock_cap_s: 21600
      min_wallclock_s: 14400
    overrides:
      energy_function: elj
      integrator: {T: 10}
      eval_T: 10
      batch_size: 1000
      max_batch_size: 1000
      eval_period: 250
      figs_period: 500
      eval_num_samples: 10000
    metrics:
      primary: [gpu/util_policy, eval_step_time, inter_eval_time]
      secondary: [gpu/util_recent, eval_sampling_time, eval_figs_time,
                  train_step_time, energy/frac_outside_step]
      catastrophes: [batch/oom_events, gradnorm/nonfinite_steps]
      unusable:
        samples_per_sec: 'counts training samples only; eval sampling is not in the denominator'
        energy/ms_per_sample: 'pools training and eval energy calls'
    liveness:
      - 'at least 8 evaluations fired inside the gpu/util_policy window'
      - 'figs_period is a multiple of eval_period, so figures actually logged'
      - 'the external nvidia-smi CSV covers the whole job, eval blocks included'
      - 'gpu/util_policy is backed by >= 0.8 * min_wallclock_s / gpu_util_sample_period_s samples'
      - 'fwd/*, bwd/* and replay/* all present and changing'
    noise_floor:
      method: repeat_launch
      repeats: 3
      seed_policy: fixed
      run_statistic: median
      dispersion: relative_span
      measured: null
    correctness:
      reference: paired_control
      detail: >-
        a100-util-sensor-crosscal at the same batch and T with eval disabled is the
        control; the delta is what eval does to the OCCUPANCY READING, which is a
        different question from what eval costs (elj-eval-cost owns that)
      gate: null
      exactness: floor
    comparison:
      valid_against: [a100-util-production-shape, a100-util-sensor-crosscal]
      rule: same_hardware_class_paired_seed
    purpose:
      - 'measures the share of wall clock the in-process occupancy sensor never samples'
      - 'the single number that decides whether gpu/util_policy OVERSTATES what the scheduler sees'
```

```yaml
  - id: a100-floor-node-confound
    title: Repeat-launch floor, within one node versus across nodes
    status: specified
    workload:
      energy_function: elj
      conditioning: unconditional
      domain: molecule
      space_groups: [2]
      z_primes: [1]
      T: 10
      dataset: 'mipcas_sg2_zp1_elj_prior_dataset.pt'
      dataset_in_repo: false
    training_mode:
      stage: equilibration
      train_mode: fused
      bwd_sampling_mode: prior
      branches: [fwd, bwd, replay]
    hardware:
      class: a100
      local_adequate: false
      a100_required: true
      reason: >-
        the question is whether cluster repeat spread is dominated by WHICH NODE the
        job landed on. A floor built from launches that happened to share a node is
        too tight, and every comparison in the a100 suites inherits it.
    work:
      kind: fixed_steps
      batch_size: 1000
      pin_batch: true
      resume_step: null
      warmup_steps: 100
      measure_steps: 400
      epochs_formula: >-
        resume_step + warmup_steps + ceil(min_wallclock_s / measured_step_time)
        rounded up to a multiple of 10
      wallclock_cap_s: 14400
      min_wallclock_s: 7200
    overrides:
      energy_function: elj
      integrator: {T: 10}
      eval_T: 10
      batch_size: 1000
      max_batch_size: 1000
      eval_period: 100000000
    metrics:
      primary: [gpu/util_policy, samples_per_sec, train_step_time]
      secondary: [gpu/util_recent, vram/peak_reserved_mb, step_time_max10]
      catastrophes: [batch/oom_events, gradnorm/nonfinite_steps]
      unusable: {}
    liveness:
      - 'SLURM_NODELIST recorded for every repeat'
      - 'the pinned-node group ran 3 launches on ONE node; the spread group ran 3 on THREE distinct nodes'
      - 'the run lasted at least min_wallclock_s'
      - 'compile_policy resolved the same way on every repeat'
      - 'fwd/*, bwd/* and replay/* all present and changing'
    noise_floor:
      method: repeat_launch
      repeats: 3
      seed_policy: fixed
      run_statistic: median
      dispersion: relative_span
      measured: null
    correctness:
      reference: paired_control
      detail: >-
        the two groups are identical work; the only difference is node assignment, so
        the ratio of their spans is the node contribution to every other floor here
      gate: null
      exactness: floor
    comparison:
      valid_against: [a100-floor-node-confound]
      rule: same_hardware_class_paired_seed
    purpose:
      - 'benchmarks.md §10 lists this as unmeasured, and it sets the required repeat count for every other a100 benchmark'
```

### 9.2 New suite

```yaml
  phase4-utilisation:
    description: >-
      The utilisation proxy and the occupancy-versus-batch question, on the hardware
      that cancels jobs. Everything Phase 6 is gated on. Absolute numbers here do not
      transfer to the dev box, which has no scheduler.
    benchmarks:
      - a100-batch-scaling-elj
      - a100-batch-scaling-uma
      - a100-util-sensor-crosscal
      - a100-util-production-shape
      - a100-floor-node-confound
```

`a100-batch-scaling-uma` also joins `a100-throughput` and `mlip-cost`. All four new
entries are named by a suite, satisfying the orphan rule.

### 9.3 Run through the real validator — three pass, one is refused on purpose

Claiming YAML validates is worth nothing unless the validator said so. Spliced into
a copy of the live registry and passed to `registry.validate()`:

```
a100-batch-scaling-uma           REJECTED: kind is fixed_steps_per_rung but batch_rungs is empty
a100-util-sensor-crosscal        PASS
a100-util-production-shape       PASS
a100-floor-node-confound         PASS
```

`resolved_overrides` confirms all four inherit the controller-off defaults —
`grow_batch_size: false`, `auto_batch_throughput_opt: false`, `max_step_seconds: 0`,
`batch_knee_recheck_steps: 0`. **The batch controller is off on every arm**, which is
what makes `t(B)` and `U(B)` data about the hardware rather than data about a
controller.

**The refusal is correct and is left in place.** `a100-batch-scaling-uma` carries
`batch_rungs: null` because its ladder is the pre-flight's *output*, and the registry
refuses to hold an unmeasured range. The consequence is that the entry **cannot be
committed to `registry.yaml` until the probe has run** — which is the desired
failure mode, since a placeholder ladder is exactly the thing that would get quietly
adopted. The entry lives in this document until it has real rungs.

> **A small gap this exposes, and the minimal fix.** The validator enforces that a
> ladder is *present*; it cannot tell a measured ladder from a copied one. That is
> how `a100-batch-scaling-elj` carries `[1000, 1650, 2722, 4491, 7410]` — copied
> from observed cluster batch sizes, by `benchmarks.md` §10's own admission — and
> passes cleanly.
>
> Proposed, in the registry's own idiom: a `work.batch_rungs_source` field on rung
> benchmarks, taking `measured` (with the probe run id and date) or `copied` (with
> what it was copied from), mirroring how `noise_floor.measured: null` already means
> "not yet measured" rather than "fine". One field, no schema version bump, and it
> makes the provenance of a ladder as visible as the provenance of a floor.

**Other validator notes.** All four declare `class: a100` with `local_adequate:
false` and a non-empty `reason`; none joins `local-dev`. All are `train_mode: fused`
with `measure_steps` a multiple of 10. All name `gpu/util_policy` primary and declare
`min_wallclock_s >= 7200`. `exactness` is `floor` throughout — `exact` is reserved
for the closed-form toy. The one `control_comparison` names its gate. The rung
benchmark sets no `batch_size`/`max_batch_size` override; the three pinned benchmarks
set both to the same value, since they are independent hard stops.

---

## 10. What this request does **not** settle

Listed rather than filled with a plausible number, per `benchmarks.md` §10's
discipline.

- **Whether the scheduler's statistic is a mean at all.** If it is a
  minimum-over-window or a per-node aggregate, every arm here still measures
  something real, but the proxy question is only partly answered. §0 is the cheap
  route out.
- **MACE.** No MACE ladder is proposed. UMA is the route the trap fired on and the
  route with the larger observed energy share; MACE's occupancy profile is a separate
  question and there is still no local MACE cost measurement on record at all.
- **Conditional routes.** The conditioning overhead is `elj-fused-cond`'s job. Whether
  conditioning changes the *occupancy* profile is not asked here.
- **Whether `t(B)` is non-monotone enough for path-dependence to matter.** The ladder
  will show the curve's shape; whether a *walk* on that shape is path-dependent is a
  `bench/` question, and it belongs to Phase 6 proper.
- **Compile's per-rung recompile stall.** The ladder pins one batch per job, so each
  job pays exactly one compile. That is the right design for a cost measurement and
  it means this request says nothing about what a *growing* batch costs in recompiles
  — which is a real Phase 6 input, and is deferred to the shadow-mode arm.
- **Anything about the dev box.** Absolute numbers never transfer between hardware
  classes, and `compile_policy: auto` resolves *off* on Windows, so a local step time
  under-represents the A100 by an unknown factor.

---

## 11. Acceptance

This request is discharged when:

- [ ] The proxy's **case (1/2/3) is written down**, with the regime it was measured
      in — route, batch, `T`, step time, eval cadence — and, if case 3, the margin
      **and the margin's measured cost in samples/sec**
- [ ] `noise_floor.measured` is recorded for every benchmark in `phase4-utilisation`,
      from at least the declared repeat launches, covering every primary metric
- [ ] `U(B)` is recorded on **both** an analytic-energy and an MLIP route, each
      classified against its measured floor as RISING / FLAT / DECLINING
- [ ] `t(B)`/`S(B)` are recorded over a **measured** reachable range, classified as
      FLAT / KNEED / NON-MONOTONE, with a knee claim made **only** where bracketing
      rungs exist on both sides
- [ ] The node-confound ratio is recorded, and every other floor in the a100 suites is
      labelled within-node or cross-node accordingly
- [ ] Every baseline is a graded finding in `findings.md` with its scope line, and
      catastrophe counts are reported alongside — never folded in
- [ ] Incomplete repeats are excluded **and the exclusion is stated**

No verdicts. This request produces evidence; the controller is Phase 6's problem and
it does not begin until the boxes above are checked.
