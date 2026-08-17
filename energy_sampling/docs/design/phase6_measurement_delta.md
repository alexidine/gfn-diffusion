# Phase 6 measurement request — delta

Argument, in the `docs/PROTOCOL.md` sense. A **delta against**
[`phase6_measurement_request.md`](phase6_measurement_request.md), not a replacement:
that document's purpose, arms, budget and priority order stand.

**Why a delta exists at all.** The request was written before the controller was
designed and says so in its own second line. So it specifies the evidence Phase 6 needs
to *start*, and could not specify what a particular control law *consumes*. Now that
[`phase6_batch_sizer.md`](phase6_batch_sizer.md) exists, the gap is visible. Finding it
before the submission rather than after is the whole point of running the design first.

Every entry states: the quantity · whether the request produces it · the delta · its
GPU-hour cost against the request's own 58-job / ~108 GPU-h budget · its priority.
**Several entries are explicitly worth zero new GPU hours**, and say so.

---

## Summary

| # | gap | covered? | GPU-h |
|---|---|---|---:|
| Δ1 | `t(B)` measured **within one process** at several `B` | no | +7 |
| Δ2 | the sampling distribution of the controller's own k-step median | no | 0 |
| Δ3 | marginal recompile cost per distinct batch | deferred by §10 | 0 (rides Δ1) |
| Δ4 | `t(B)` drift **within** a stage | no | 0 (rides A/B) |
| Δ5 | `B_max` repeatability and OOM hysteresis | no | +2 |
| Δ6 | one rung **below** the grad-accum crossover | no | 0 by displacement |
| Δ7 | proxy window vs controller decision timescale | partly | 0 |
| Δ8 | the objective itself | no, by design | **not funded** |
| Δ9 | eval memory profile at the train-discovered ceiling | no | +0.5 |
| Δ10 | registry/API defects the delta trips over | — | 0, blocking |

Revised total **≈117 GPU-h (+8%)**. The single most consequential entry, Δ7, costs
nothing and is settled by arithmetic already in the repo.

---

## Δ1 — `t(B)` measured within ONE process, at several `B`

**Quantity.** Step time and throughput at each ladder rung, measured **back to back
inside one process**, with `torch.compile` recompiling at each shape.

**Covered? No — and it is a different quantity from what the request produces.** Every
registry arm sets `pin_batch: true` and measures **one `B` per job**. That is exactly
right for a *cost benchmark*: it isolates the operating point. But a startup-calibrating
controller never gets that. It visits several `B` in one process, in sequence, paying a
recompile at each and inheriting whatever allocator state the previous rung left. The
cross-job ladder measures `t(B) | fresh process`; the controller consumes
`t(B) | process that has already run B', B''`. Nothing establishes these are the same
number.

**Delta.** A **calibration probe**, not a registry benchmark — `registry.py` is right to
refuse it, since the batch moves by design and that violates the fixed-work quantity.
One launch per route that walks the measured ladder up and back down with a fixed dwell,
logging per-rung `t` and `sps` plus the recompile counters from Δ3. Its output is a
**ratio against arm A/B's own dispersion**, regenerated in the same experiment — not an
absolute number.

**Cost.** UMA 2 jobs × 2 h = 4; ELJ 3 jobs × 1 h = 3. **+7 GPU-h.**

**Priority.** With its route's ladder (W-UMA with B, W-ELJ with A).

---

## Δ2 — how many steps a `t(B)` reading needs

**Quantity.** The sampling distribution of a **k-step median** of `train_step_time` at
fixed `B`, for k in {10, 20, 50, 100}, within one job.

**Covered? No.** The registry measures the **cross-launch repeat span** — the right
statistic for a benchmark, and not the one a controller acts on. `train.py` medians the
last 20 timings and decides on that. Nothing anywhere measures whether 20 is enough, or
what that median's own dispersion is.

This is directly load-bearing: it instantiates `RESOLVE_EPS` and `DWELL_STEPS`. Unset,
every rung is provisional and the controller can survey but never reject — safe, and
slower than it needs to be.

**Delta.** **Zero new jobs.** One instrument line:

> **G7.** Log the controller's own estimator as a series — `batch/med_step_s` and
> `batch/sps_rung`, the 20-step median and the rung throughput, at the 10-step report.

Then every already-funded ladder hour becomes a measurement of the statistic the
controller actually reads. Without it the submission produces a curve no in-process
controller can be shown to reproduce — the same argument the request makes for G3.

**Cost. 0 GPU-h.** **Priority: 0 (blocking), with G1–G3.**

---

## Δ3 — marginal recompile cost per distinct batch

**Covered?** The request **defers** it (§10, "deferred to the shadow-mode arm").

**Why it can no longer be deferred.** Any design that probes several `B` in one process
pays this cost on every rung transition, and it is the term that decides whether
calibration is affordable. The sandbox already shows it is not negligible: with
recompiles charged, the shipping controller's churn makes it **lose to the worst
constant batch on the ladder** (4927.3 vs 4962.8 — measured, `bench/test_batch_traps.py`).

**Delta.** Rides entirely on Δ1's probe. One instrument line:

> **G8.** A per-step cost sidecar plus `torch._dynamo` recompile counters, so a rung's
> first step is separable from its steady state.

Also record `torch._dynamo.config.cache_size_limit`. Past that limit dynamo stops
recompiling and falls back to **eager** — a cliff, not a linear charge, and the
mechanism that makes batch churn expensive rather than merely slow. Nothing in the repo
models it.

**Cost. 0 GPU-h** beyond Δ1. **Priority: with Δ1.**

---

## Δ4 — `t(B)` drift within a stage

**Quantity.** At a single fixed `B` inside one stage: `sps_rung` over ≥4 disjoint equal
blocks across the job, reported as a trend against the within-window dispersion from Δ2.

**Covered? No.** The request's §7 row 5 records *falling utilization* as prod0810's
shape — that is **U** drift, not **t** drift. Meanwhile `batch_knee_recheck_steps` exists
precisely because "the knee moves WITHIN a stage as the fused composition drifts": a
belief with no measurement behind it, driving a mechanism that ships.

**Delta. Zero new jobs — it is already funded.** Every ladder arm runs ≥7200 s at a
fixed `B` inside one stage, so each job *contains* this measurement. The delta is an
analysis deliverable and an acceptance box:

> Split each ladder job's measurement window into ≥4 disjoint equal blocks; report the
> trend of `sps_rung` across blocks against the between-block spread. Classify
> DRIFTING / STATIONARY per route. A route classified STATIONARY makes `RUNG_TTL_STEPS`
> unnecessary; DRIFTING falsifies calibrate-then-hold as an instrument.

**Two caveats that must travel with it.** (i) All arms set `resume_step: null` — fresh
starts, i.e. the *first* 7200 s of a stage, where the transient is worst. (ii)
`defaults.overrides` disables `z_calibration`, so the ladder's work numerator is
`attempted_batch` alone while production's is `attempted_batch · (1 + _z_cal_rollouts)`.
The ladder's `samples_per_sec` is therefore **not** the quantity the controller reads.
State it; do not spend hours fixing it.

**Cost. 0 GPU-h.** Across-*transition* drift is **not fundable** — `benchmarks.md`
excludes measuring across a transition by rule, and no arm sweeps stage.

---

## Δ5 — `B_max` repeatability and OOM hysteresis

**Quantity.** (a) `B_max` by bisection, **repeated across launches**, spread reported in
rungs; (b) whether a size that ran once runs again after the batch moved away and back
within one process; (c) `vram/peak_reserved_mb` at every rung.

**Covered? No.** The request treats OOM only as a catastrophe *count*. The only thing
touching `B_max` is the pre-flight, specified in one sentence and budgeted at ~2 GPU-h
with **no entry, no repeats, no floor, no liveness, no reproducibility statement**.

`B_max` is the top of the candidate ladder in every design. And the entire
ceiling-**expiry** mechanism is a *bet* that an OOM is transient fragmentation cleared by
its own recovery — a bet with zero measurement behind it anywhere in the repo.

**Delta.** Promote the pre-flight from a throwaway to the submission's first recorded
artifact: **3 bisections per route** in separate launches; **1 hysteresis probe per
route**; per-rung VRAM recorded.

**If `B_max` varies launch to launch by more than one rung, every design's ladder top is
a random variable** and the ceiling logic is operating on noise. That is a first-order
finding for ~1 GPU-h.

**Cost. +2 GPU-h** (pre-flight 8 jobs/2 h → 16 jobs/4 h). **Priority: 0 — it precedes
everything**, because `registry.py` refuses `batch_rungs: null` and arms A and B cannot
declare a ladder without it.

---

## Δ6 — the ladder BELOW the grad-accum crossover · **PROMOTED to priority 1**

> **Re-prioritized 2026-08-16** when the objective was decided as *steps/sec at a
> threshold effective batch size*. Under that objective the throughput optimum is
> `B = fused_grad_accum_min_samples` exactly, so **priority 2 needs no search at all** —
> and the only region where `B` is still a free variable is **below** `A`, which is
> exactly where the declared ladder has no rungs. This entry was "last, but free"; it is
> now the only part of the ladder priority 2 reads.
>
> What it must answer: is `samples_per_sec` still *rising* at `A`, or already flat? If
> flat, any `B` in the flat region is equally good and the smallest wins on memory
> headroom. If rising, `B = A` is confirmed as the optimum by measurement rather than by
> the algebra alone.
>
> The rungs **above** `A` do not become worthless — they are what priority 1 needs, since
> growth above `A` is now exclusively occupancy's business. But they answer a constraint
> question, not a throughput one, and should be labelled that way.

**Quantity.** `t(B)` and `updates_per_sec(B)` at one rung **below**
`fused_grad_accum_min_samples`.

**Covered? No, and this is the gap with the largest consequence for the objective.**
Accumulation engages **strictly below** the target. mk_dev ships `batch_size: 1000`
equal to `fused_grad_accum_min_samples: 1000`, so **every rung on the declared ladder is
at or above the crossover** — and the identity that justifies maximizing
`samples_per_sec` (asserted at four separate sites) holds **nowhere on it**. On a
saturating curve the two candidate objectives have **opposite argmaxes**.

**Delta.** Place **one rung below the crossover** on arm A. `pin_batch` with
`batch_size < fused_grad_accum_min_samples` is legal — the relevant invariant is
BASELINE, not ERROR, so it warns rather than blocks. This is the only way to observe
`updates_per_sec(B)` on both sides of its own crossover.

**Cost. 0** if the pre-flight places one of its ≥5 rungs below the crossover instead of
adding a sixth. **Recommend displacing.** (+6 GPU-h if added.)

**Priority.** Last, but free.

---

## Δ7 — the proxy's window versus the controller's decision timescale

**This is the most consequential entry, and it costs nothing.**

**Part (a) is not a measurement question at all**, and saying so is worth more than an
arm. `_gpu_util_mean` is an unweighted trailing mean over a window that is **never
cleared** at a batch change or a stage transition. So after moving to `B₂` the reading is
`(n₂·U(B₂) + n₁·U(B₁))/(n₁+n₂)`, and at constant cadence the mixture is linear in dwell.

> **Reading `U(B₂)` to within a fraction ε of the step requires dwelling `(1−ε)·W` at
> `B₂`.** At ε = 0.1 and `W` = 7200 s that is **1.8 h per rung**; a 5-rung survey is
> ≥9 h of dwell before priority 1 is evaluable, per stage.

Compounding it (`F-036`): the number of growth decisions inside one policy window is
**48–288×** on ELJ and **0.5–0.8×** on MLIP — a ~600× span that crosses 1 — while
`gpu/util_recent` is simultaneously **absent** on MLIP (3–4 samples against a 5-sample
floor).

**Part (b) is a measurement question and is already fully funded.** The mandatory
external `nvidia-smi` CSV at 10 s cadence is a full-rate record; every shorter window is
recoverable from it offline. The delta is one word — do it at **several** window lengths:

> Report the between-window spread of the external occupancy series at 300 / 900 / 1800 /
> 7200 s, per route, per rung, block-bootstrapped with block length ≥ the eval period.
> State the shortest window at which the spread is smaller than the smallest rung-to-rung
> `ΔU` the ladder resolves.

**Design consequence — the reason this is first among the free items.** Priority 1
**cannot be closed-loop on batch at controller timescales.** It must be **feed-forward**:
a `U(B)` relation fitted offline from arms A and B, evaluated at candidate `B`, with the
in-run reading demoted to a slow stage-granularity audit. That changes what the ladder is
*for* — it stops being calibration input for a live constraint and becomes the **training
set for a feed-forward one** — and it should be settled before the submission, not after.

**Cost. 0 GPU-h.** And §0's email answers half of it for free.

---

## Δ8 — the objective

**Quantity.** Loss reduction per wall-clock second as a function of `B`.

**Covered? No, and by design** — the request's §1 says "nothing else", and `seed_policy:
fixed` is set precisely because these are cost benchmarks.

**Explicitly NOT funded here.** A quality measurement needs varying seeds, long horizons
and eval on; it is comparable in size to the entire 108 GPU-h submission and does not
belong in a cost request. Two free things instead:

1. Δ6's sub-crossover rung gives the *shape* of `updates_per_sec(B)` on both sides of its
   argmax — which is what decides whether the two candidate objectives point in opposite
   directions on this hardware.
2. A documentation fix, zero cost: record that "samples/sec **is** opt-step throughput,
   so step time does not enter" is **regime-scoped to `B < fused_grad_accum_min_samples`**,
   and that mk_dev places the entire reachable walk outside that regime. It is asserted
   at four sites as if unconditional.

**Priority.** Out of scope — state it in §10 rather than leaving it implicit.

---

## Δ9 — eval memory profile at the train-discovered ceiling

**Covered? No.** `batch_size` is a single global knob — eval sampling, anchor refresh and
the loaders all read it. The OOM ceiling is installed only by train steps (correctly), but
the number the controller selects **also resizes eval**. Every ladder arm disables eval,
so no arm ever evaluates at a ladder rung; arm D is the only eval-on arm and sits at a
single batch.

**Delta.** One eval pass at the candidate `B_max` per route inside Δ5's pre-flight, plus:

> **G9.** Segment `vram/peak_reserved_mb` by train vs eval — reset the peak counter
> around the eval block so the eval peak is separable.

**Cost. +0.5 GPU-h.** **Priority: with Δ5.**

---

## Δ11 — arm C is largely already funded, by wandb's own system monitor · **−9 GPU-h**

**Discovered 2026-08-16 (F-038), after the rest of this delta was written.**

`system.gpu.0.gpu` is logged by wandb on **every run ever recorded**, from a **separate
thread** at ~14 s cadence — so it samples *during eval*, which the in-process sampler
structurally cannot. That is exactly the role §4.2 specifies for the external
`nvidia-smi` sidecar: an out-of-process, concurrent, higher-cadence sampler of the same
NVML counter, joined on the same clock.

It is an independent **sampler**, not an independent **instrument** — same counter — so
it controls for cadence, phase and eval blindness, and for nothing about the counter's
semantics. That is precisely what arm C was for.

**What this collapses.** Arm C (sensor cross-calibration, 2 cadences × 3 launches,
12 GPU-h) asks whether a disagreement between in-process and out-of-process readings is a
*cadence* problem or a *structural* one. **That is answerable retroactively, at zero
cost, on runs that already exist** — and answering it retroactively is strictly better,
because it covers many routes, batches and stages rather than one pinned operating point.
Reduce arm C to **one confirmation launch (2 GPU-h)** whose only job is to verify that the
system stream behaves the same on an A100 as on the dev box.

**What it does NOT replace, and the sidecar is still required for all four:**

1. **Resolution on long runs.** wandb decimates history: measured `n = 10` system samples
   in a 7200 s window on a 48-hour run. The sidecar's 10 s CSV does not decimate.
2. **Throttle reasons** — `clocks_throttle_reasons.active` separates "idle" from
   "throttled", two states with the same utilization number and opposite responses.
3. **Per-process attribution** — `--query-compute-apps` is what makes a co-tenanted
   interval *excludable* rather than silently averaged in.
4. **MIG detection** — on a MIG-partitioned A100 the in-process sensor is off from the
   first sample, and nothing else reports it.

**Also: it makes the §7 disagreement table checkable today.** Rows 3, 4, 5, 8 and 11 are
all in-process-versus-external comparisons, and the external series exists retroactively.
Running that table over the existing corpus before submitting would tell us which rows
actually occur.

**Cost. −9 GPU-h** (arm C: 12 → 2, plus ~1 h of analysis). **Priority:** do the
retroactive analysis **before** the submission — it may change what arms are needed.

> **The trap this closes, and it is the reason the entry is here at all.** Comparing our
> trailing-7200 s MEAN against wandb's whole-run MEDIAN gives differences of ±49 points.
> Comparing both as means over the same trailing window gives a median difference of
> **+1.1**. The first comparison is a statistic mismatch wearing a sensor disagreement,
> and it is exactly the shape §7 exists to prevent. **Any use of the system stream must
> state its window and its statistic**, or it will manufacture the disagreement it was
> brought in to adjudicate.

---

## Δ10 — registry and API defects this delta trips over

All 0 GPU-h, all blocking.

1. **`epochs_for()` ignores `epochs_formula`.** It hard-codes `resume_step +
   warmup_steps + measure_steps`; the validator only checks the formula *string mentions*
   `resume_step`. The request's §9 fix replaces the formula with
   `... + ceil(min_wallclock_s / measured_step_time)` on three entries — and `epochs_for()`
   returns **500** for all three against a declared intent of ~7300 at 1 s/step. **A 14.6×
   discrepancy, silently.** Either `epochs_for` learns the formula, or the entries carry a
   `measure_steps` computed by the pre-flight.
2. **`noise_floor.repeats` is one integer per benchmark**, so "3 general rungs / 5 at
   anchors" is inexpressible — and `_validate_floor` requires `measured.repeats >=
   noise_floor.repeats`, so raising 3 → 5 makes a floor from the three 3-repeat middle
   rungs **unrecordable**. Pick one: a per-rung repeat field, split anchors into separate
   entries, or keep 3 and record the anchors' extra repeats separately.
3. **The schema is open** — unknown keys validate silently (verified). So the proposed
   `work.batch_rungs_source` field is **decoration** until `registry.py` learns it. That
   cuts both ways: it is cheap to add and it enforces nothing.
4. **`defaults.overrides` disables `z_calibration`**, so the ladder's `samples_per_sec`
   uses a different work numerator from production's. Record the scope line — the missing
   z_cal term is one of the three accounting bugs that walked prod0810 to batch 12226.

---

## Revised budget

| arm | jobs | h each | GPU-h | Δ |
|---|---:|---:|---:|---|
| Pre-flight, **extended** (Δ5, Δ9) | 16 | 0.25 | **4** | +2 |
| B · UMA ladder | 16 | 2 | 32 | — |
| **W-UMA** · within-process ladder (Δ1, Δ3) | 2 | 2 | **4** | +4 |
| D · Production shape, eval on | 3 | 4 | 12 | — |
| A · ELJ ladder | 19 | 2 | 38 | — |
| **W-ELJ** · within-process ladder (Δ1, Δ3) | 3 | 1 | **3** | +3 |
| C · Sensor cross-calibration — **collapsed by Δ11** | ~~6~~ 1 | 2 | **2** | **−10** |
| E · Node confound | 6 | 2 | 12 | — |
| | **66** | | **≈107** | **−1 (−1%)** |

**Net: the delta pays for itself.** The nine entries that consume GPU hours add ~9;
Δ11 removes ~10 by noticing that arm C's question is already answered retroactively on
runs that exist. The submission gets *more* answers for *slightly fewer* hours than the
original request.

**Priority order** — the request's own order is preserved; new items attach rather than
compete.

```
0.  G7, G8, G9 (0 GPU-h) + extended pre-flight (4 GPU-h)      BLOCKING
1.  B (32) + W-UMA (4)                                = 36
2.  D (12)
3.  A (38) + W-ELJ (3)                                = 41
4.  C (12)
5.  E (12)
--  Δ4 and Δ7 are analysis riders on whatever runs;     0 GPU-h
opt Δ6 sub-crossover rung (0 by displacement)
```

If only B runs: **B + W-UMA + extended pre-flight = 40 GPU-h**, answering trap (a)'s
premise *and* whether a within-process ladder is affordable on the route that matters.
B and D together: **52 GPU-h** — the request's own "44 GPU-h" line plus the controller's
instrument.

---

## What this changes about the submission

1. **The pre-flight stops being a throwaway** and becomes priority 0 with repeats. If
   `B_max` is not reproducible across launches, three design documents are building on
   sand.
2. **Three instrument lines join G1–G3 as prerequisites** — G7, G8, G9. All in `train.py`,
   all cheap. **G7 alone converts every already-funded ladder hour into a measurement of
   the statistic the controller actually reads.**
3. **One genuinely new arm, 7 GPU-h**, and it must not become a registry benchmark.
4. **Two acceptance boxes, both free:** σ_within/σ_across recorded per route; each ladder
   job classified DRIFTING / STATIONARY.
5. **The biggest consequence is a design consequence, not a budget one.** Δ7 is settled by
   arithmetic already in the repo, before any job runs: priority 1 is not
   closed-loop-controllable on batch at calibration timescales and must be feed-forward.
6. **§0's email is still worth more than this whole delta.** If the real window is shorter
   than 7200 s, or the statistic is a minimum-over-blocks, or it is evaluated once at a
   checkpoint, Δ7's conclusion moves and arms C, D and E collapse. Ask before submitting.

## What this delta does not fund

- **The objective** (Δ8) — a quality measurement, not a cost one.
- **Drift across a stage transition** — excluded by rule; no arm sweeps stage. Report it
  opportunistically from arm D if G4b lands.
- **Eager vs compiled benefit** — a real hole, but not what a controller consumes; only
  the *marginal* per-shape cost is, and Δ1 delivers that.
