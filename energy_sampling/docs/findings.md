# Findings

Append-only evidence ledger. Entries are **never edited** — a later entry
supersedes an earlier one by naming it. Format and grades: [`PROTOCOL.md`](PROTOCOL.md).

Newest first.

---

## F-041 · The gradient-geometry diagnostic is fatal under `torch.compile`, and no local run can ever see it · `MECHANISM`

*2026-08-16. First cluster job of `a100_stab_aug16` (arm `f3_mipcas_uma`), A100
80 GB, UMA, batch 250, T=10, `unconditional_tb`, seed 12345.*

**Scope:** any run with `compile_policy` resolving ON **and**
`grad_geometry.enabled: true` — i.e. every cluster run spawned from `mk_dev` as
it currently stands, on both routes, as soon as it reaches a fused stage.
Confirmed on one job; the mechanism is not statistical.

**What happened.** The run cleared MLE, transitioned at ~step 250, rebuilt the
prior buffer by churn, engaged `equilibration` — and died at **step 320**, the
first fused step on which the diagnostic armed (`every: 50`):

```
RuntimeError: This backward function was compiled with non-empty donated buffers
which requires create_graph=False and retain_graph=False.
```

**Mechanism.** `_log_fused_gradient_geometry` takes one
`torch.autograd.grad(..., retain_graph=True)` per active branch to read branch
gradients without disturbing the real backward. AOTAutograd's *donated buffer*
optimisation frees-and-reuses intermediates on the assumption that each compiled
backward runs exactly once, and hard-raises on any re-entry. The exception
propagates out of `fused_train_step` through `handle_train_epoch_error` (which
re-raises anything that is not an OOM) and ends the job. A **diagnostic killed
the run it was observing**, having produced no diagnosis.

**Why every local shakeout passed.** `compile_policy: auto` resolves to
Linux+CUDA only — inductor has no CUDA on native Windows — so the dev box runs
eager, no AOTAutograd, no donated buffers, and this code path is *structurally
unreachable* there. The same day's four local shakeout runs all crossed the same
transition with the same diagnostic armed and none could have caught it. This is
the concrete case for the standing rule that a local pass is not evidence about
the cluster.

**Fixed, in two places.**

1. `maybe_compile_policy` sets `torch._functorch.config.donated_buffer = False`
   whenever it enables compile. It must be there and not at the diagnostic: the
   choice is baked in when AOTAutograd traces, so setting it after first forward
   is too late. Cost is some activation-memory reuse.
2. The probe now catches `RuntimeError`, **disables itself for the rest of the
   run, says so loudly, and logs `fused_grad/disabled: 1.0`** — because a metric
   that merely stops appearing reads as "the diagnostic looked and saw nothing
   wrong". Training is untouched; the probe only reads gradients.

Both halves are mutation-proven in `test_fused_grad_geometry.py`: removing
either makes its test fail (verified 2026-08-16), and the guard test has a
negative control asserting real geometry is still reported when nothing raises.

**Unverified locally, by construction.** Neither fix can be exercised on this
dev box, for exactly the reason the bug survived to the cluster: the tests pin
the *source* (that `donated_buffer = False` is set at the compile site) and the
*fallback* (that a raising probe disables instead of propagating). Whether
donated-buffer-off is sufficient in this torch build is established by the next
cluster run reaching step 320+ with `fused_grad/*` present and no
`fused_grad/disabled`.

**Two riders from the same log, both first-of-their-kind.**

- **`compile_policy: auto` engages on the cluster** — "trunk (…) compiled
  (default mode, lazy on first forward)". First direct confirmation; it had been
  an open question.
- **The loose MLE gate `{1.0, 0.5, 100}` fires on the A100**, at ~step 250 on
  the unconditional UMA route, vindicating the same-day reversal away from a
  stricter gate that had failed to fire in 3186 local steps.

Also: the UMA init prior scan peaked at **26.7 GB** reserved at batch 250 — fine
against 80 GB, fatal against the dev box's 16 GB, and worth carrying into any
batch-ceiling estimate.

---

## F-040 · F-039 fixed: the probe now decides before it draws, and the tier-C divergence goes to zero · `MECHANISM`

*2026-08-16. Fixes F-039 (below), verified with the harness that found it.*

> **THIS CHANGES THE NUMBERS FOR EVERY RAY-ARMED CONFIG, AND THAT IS THE POINT.**
> Any run whose protocol declares `lr_sensor: {kind: ray}` on a stage now takes
> a different trajectory from the same seed than it did before this commit,
> because the probe no longer consumes RNG inside warmup. Consequences, plainly:
>
> - **Any comparison that spans this fix is confounded.** An arm launched before
>   it and an arm launched after it differ by more than whatever was under test.
>   Re-run the baseline arm rather than reusing a stored one.
> - **Any run in flight right now is on the OLD behaviour.** Do not read a
>   resumed run as continuous across a restart that crosses this change.
> - **Checkpoints are unaffected** — no state layout changed — but a resume
>   after the fix will not reproduce the pre-fix continuation.
>
> No `project_state_version` bump: no config key changed meaning. A config
> written before this reads exactly as it did.

**Scope:** as F-039 — `latent_gaussian` sg 1 Z'=1, T=10, batch 1000,
`unconditional_tb`, 600 steps, seed 12345, RTX 5080, `--deterministic strict`.

**The fix is a predicate, not an early return.** `LRController
.calibration_refusal()` answers "would this reading be thrown away regardless of
what it measures", from state alone. `_ray_probe_armed` consults it BEFORE
arming; `on_calibration` consults the same function, so the gate that skips the
probe and the gate that refuses the reading cannot drift apart.
`RayCalibration.refuse` then consumes the period exactly as a completed
calibration does — without that, the latch would stay pending through warmup and
the first probe would fire on the first step after it instead of the next period
boundary, which would have moved the applied path.

**Every refusal path, classified.** This is the part that matters, because
"warmup" is only the one this run happened to hit:

| path | decidable before drawing? | gated |
|---|---|---|
| warmup envelope still ramping | yes | **yes** |
| `lr_servo_managed` empty | yes | **no, deliberately** |
| `unresolved` (no test cleared its CI) | no — it IS the measurement | — |
| `inconsistent` (lo ≥ hi) | no — same | — |
| `alpha_star` non-finite or ≤ 0 | no — same | — |
| `no_batch` / `too_few_subbatches` | no — the draw is what fails | — |
| clamped at a `bounds`/ceiling edge | no — depends on `alpha_star` | — |

The second row is the one worth arguing. An empty `lr_servo_managed` means
`peak_scale` reaches no learning rate, which looks like a guaranteed discard —
but `_managed_keys` calls that state *"its own control arm"*: the controller
reads and logs while actuating nothing, and there the reading IS the
deliverable. **"No LR moved" is not "the reading was thrown away."** Gating it
would delete a documented operating mode.

**Verified on the comparison that exposed it** — `configs/mk_dev.yaml` at
`7625d09` vs current, 600 steps, same seed:

| | before the fix | after |
|---|---:|---:|
| step records differing (600 × loss, LR, fused sub-losses) | 2562 | **0** |
| first divergent step | **501** | none |
| shared logged metrics differing | 1111 | **3** |
| reference-only keys | 0 | 0 |

**And re-introducing the bug fails it again.** With the gate deleted from
`train.py` and everything else held fixed, the same comparison returns to 2562
differing step records with the first divergence at **step 501** — the original
number, to the step. A test that cannot fail is not evidence, and this one can.

**The 3 residual differences are NOT reclassified away.** All three are
`probe/device_alloc_delta` — the delta of `torch.cuda.memory_stats()
['num_device_alloc']`, i.e. a count of `cudaMalloc` calls — at steps 390, 400
and 430. Step 430 coincides exactly with a `fused_grad/*` report, a
`grad_geometry` diagnostic the pre-consolidation config does not declare at all;
390 and 400 sit just past the stage transition, where the candidate's allocator
holds tensors the baseline never allocates (the hypergradient's cached
displacement). They are resource counters, not numerics, and every number the
model trains on is bit-identical. They are left in the comparison rather than
added to the wall-clock exclusion list, because that list is calibrated by the
NULL test and the null never flagged them — excluding a key at the moment it is
the last thing between a result and "identical" is how a comparison stops
meaning anything.

**Audit of the same shape elsewhere — reported, not chased.**

*Residual instances, inherently post-hoc:* `measure`'s partial draws (a `None`
draw after ≥1 success discards what was already drawn — cannot be pre-decided);
`arm`'s clone followed by `deferred_no_step` (cost only, no RNG, and the module
documents the trade).

*Checked and clean, so nobody re-checks them:* **`hyper`** reads `p.grad` and a
stored displacement and computes one dot product — no draws, no RNG, and its
warmup discard is free. **`plateau`** calls `in_warmup()` first and reads only
`metric_tracker` EMAs. `grad_geometry` is cadence-gated before any work and
reuses the fused step's graph. `z_calibration_tick` is decide-then-act, returning
on `excess <= 0` before touching RNG. `_per_step_probe` and
`_verify_dead_latent_rows` are opt-in and deterministic. The fused force-refresh
does sample a branch whose gradient is discarded, but its rolling stats are the
deliverable — adjacent, not an instance.

*A broader shape, worth its own work:* three diagnostics consume the GLOBAL RNG,
so their mere presence shifts training — `log_dist_stats` (`train.py:4504,4506`,
two `randperm` for the anchor split, in a function that seeds its
`sliced_wasserstein` projections explicitly and these not),
`eval/evaluations.py:299` (a funnel-figure subsample), and the held-out
`eval_test` rollout (`train.py:4268`, `side_effects=False`, feeding no gate,
controller or loss). **Consequence: two runs differing only in `eval_period`,
`figs_period`, or whether held-out eval is configured are not step-for-step
comparable.** That is why `tierc_smoke` pins all three.

---

## F-039 · The ray probe SAMPLES during warmup, so arming a sensor that applies nothing still changes every step of the run · `MECHANISM`

*2026-08-16. Found by the tier-C smoke harness (`tierc_smoke.py`) on its first
real use — the Phase-1 consolidation comparison. Verified against the shipping
code, not inferred from the traces alone.*

**Scope:** `latent_gaussian`, sg 1, Z'=1, T=10, batch 1000, protocol
`unconditional_tb` (train_prior MLE → equilibration fused), 600 steps, seed
12345, RTX 5080, torch 2.8.0+cu128, `--deterministic strict`.
`adaptive_lr.warmup_steps: 1000`, `ray_calibration.period: 500`.

`configs/mk_dev.yaml` at `7625d09` (pre-consolidation, migrated, `auto` rates
pinned to `seed_lr`) against the current file, both run under current code.

**Steps 0–500 are bit-identical.** 14,313 deterministic values, including the
phase-1→2 stage transition, which fires at **step 381 in both**. From **step
501** every subsequent loss differs.

The cause is at step 500 and it is one thing: the current config declares
`lr_sensor: {kind: ray}` on `equilibration` and the baseline declares no sensor,
so only the current config fires a ray calibration. What that calibration did:

| | ref (no sensor) | cand (ray) |
|---|---|---|
| `lr_ctrl/calibrations` @500 | 0 | **1** |
| `lr_ctrl/cal_status` | — | **5 = `warmup`** |
| `lr_ctrl/cal_applied` | — | **0.0** |
| `lr_ctrl/peak_scale` | 1.0 | 1.0 |
| `lr.fused` @599 | 2.0272626216986626e-05 | **identical** |
| loss @599 | 0.078703 | 0.048191 |

**The learning rate never moved.** Every LR is bit-identical across all 600
steps in both arms, and `peak_scale` is exactly 1.0 throughout — the controller
read the calibration and threw it away, by design, because the envelope is still
ramping (`cal_status: warmup`).

**The run diverged anyway, because the probe SAMPLED before being refused.**
`RayCalibration.measure` draws `n_sub: 8` fresh sub-batches through
`_draw_probe_batch` (replay buffer) and scores each at 8 alphas — 64 loss
evaluations. It restores every parameter it touches bitwise and writes no state,
which is what its contract promises. It does not, and cannot, restore the RNG
those eight draws consumed. So the training rollouts from step 501 onward read a
shifted random stream.

**The mechanism to prevent this exists, is documented, and this sensor does not
use it.** `LRController.in_warmup()` carries the docstring *"Public because a
sensor may need to decline to SAMPLE during warmup, not merely to act."* The
**plateau** sensor calls it and returns `{}` (`train.py:4861`). The **ray** path
does not: `_ray_probe_armed` checks the stage's sensor kind and then
`RayCalibration.arm`, which checks only `due(step_ind)`. There is no warmup
check anywhere on that path.

**Consequences, in order of how much they cost:**

1. **A sensor-on/sensor-off A/B is confounded from the first probe**, 500 steps
   before the sensor is permitted to act. The arms differ by an RNG shift, not
   only by the thing under test. This is a different defect from F-025 (where a
   saturated reading trained the D33 arms 8× apart in LR) — here the LR is
   provably identical and the run still diverges.
2. **The probe pays full price for a discarded reading**: 64 loss evaluations
   plus a full parameter clone per probe, for every probe inside the first 1000
   steps of each stage.
3. **`warmup_steps` is restarted at every stage transition**
   (`controller.py:11`, `rearm_warmup` on the `Protocol.advance` hook), so the
   dead zone is not once per run. In this run the transition at 381 pushed the
   sensor's first permitted action to step 1381 — no sensor of any kind actuates
   anywhere in a 1200-step window, which is why the divergence above is purely
   the sampling side effect.

**Not a consolidation defect.** Tier A/B over the same pair reports 8 changed
values, and all 8 are this feature: the two `lr_sensor` blocks, the four
`lr_servo_managed` flags that follow from them, and a protocol *name*. No loss
coefficient, batch size, clip, schedule or buffer setting moved. The LR-sensor
addition is a deliberate change and `change_history.md` records it as one. What
this finding adds is that its runtime footprint starts far earlier, and by a
different route, than "the sensor moves the LR" describes.

**The instrument.** Same-config spread on this target is **exactly zero** — 0 of
243 values at 30 steps (seeds 12345 and 777) and 0 of 14,313 at 600 steps — so
the comparison is exact and needs no tolerance. Two conditions are required for
that and both are measured, not chosen: `torch.use_deterministic_algorithms`
(without it 7 of 243 values differ, all grouped reductions at float32 rounding),
and excluding wall-clock keys (21 of 522). The harness detects a **1e-6**
relative change to a single loss coefficient.

---

## F-038 · wandb's system monitor is a free retroactive cross-calibration, and it does NOT corroborate "occupancy declines with batch" · `OBSERVED`

*2026-08-16. `system.gpu.0.gpu` from wandb's own system stream, which samples in a
SEPARATE THREAD (~14 s cadence) and therefore keeps sampling during eval, unlike
`_sample_gpu_util`.*

**Scope:** 18 runs carrying both readings; the within-run table is ONE run,
`umaperf0812_c_controller`, on host **BB2 — the dev box, not the cluster** — 0.94 h,
batch 100→741 in one stage. n is small throughout. Not replicated.

**1. Compared like with like, the two sensors mostly agree.** Both as a MEAN over the
same trailing window (`min(7200 s, runtime)`): median difference **+1.1 points**, 12 of
18 within 10 points. An earlier comparison of our trailing mean against wandb's
whole-run median showed ±49 and was an artifact of comparing different statistics over
different windows — the confound, not the sensors.

**2. Two of the six outliers are not evidence.** `prod0810_mipcas_elj` (−34.4) and
`prod0810_nehzor_uma` (+32.6) rest on **n = 10** system samples in a 7200 s window,
because wandb decimates history on 48-hour runs. Long cluster runs lose system-metric
resolution, which is the one thing the external CSV still has to supply.

**3. The surviving disagreements are the `umaperf0812` family — the runs whose table
deleted `gpu_util_floor`.** On `c_controller`, ours 50.6 vs wandb 82.0 over the same
window. Within the run, against batch:

| batch | ours (`util_recent`) | wandb sys gpu | samples/s |
|---:|---:|---:|---:|
| 100 | 55.2 | 100.0 *(n=2)* | 61.1 |
| 165 | 50.5 | 69.7 *(n=11)* | 64.6 |
| 272 | 46.9 | 71.3 *(n=18)* | 43.4 |
| 449 | 47.1 | 89.4 *(n=37)* | 36.1 |
| 741 | 54.2 | 86.6 *(n=20)* | 26.1 |

**Throughput falling in batch is corroborated. Occupancy declining in batch is NOT.**
The independent sampler's minimum is at batch 165, and it *rises* at the top rung. Our
sensor's decline rests on 2–5 ten-step rows per rung with a 900 s window smearing across
rungs ~11 min apart, so those readings are not independent of each other.

**What this does and does not change.** The deletion of `gpu_util_floor` **stands**: the
rule grew batch 7.4× while samples/sec fell 58 %, and the pre-registered joint clause —
no occupancy rule unless `U(B)` and `S(B)` move in the *same* direction — refuses it on
the throughput half alone. What is weakened is the *stated mechanism* ("utilization is
flat-to-declining in batch on this route"), which was carried as a fact and is now
one thin, dev-box, in-process-sensor reading contradicted by a concurrent independent one.

**4. The practical consequence: arm C is largely answerable for free, retroactively.**
wandb's system monitor is an out-of-process, concurrent, ~14 s-cadence sampler joined on
the same clock, present on **every run ever logged**. That is precisely the role the
external `nvidia-smi` sidecar was specified for. It is an independent **sampler**, not an
independent **instrument** — same NVML counter — so it controls for cadence, phase and
eval blindness and for nothing about the counter's semantics. What it does *not* supply,
and what still needs the sidecar: throttle reasons, per-process attribution for
co-tenancy, MIG detection, and resolution on long runs (see 2).

Feeds [`design/phase6_measurement_delta.md`](design/phase6_measurement_delta.md) Δ7 and
the arm C budget.

---

## F-037 · Trap (b) does not descend to 1, the floor does not stop the churn, and the shipping controller loses to every fixed batch once recompiles are charged · `MECHANISM`

*2026-08-16. Driven against the REAL `train.Modeller.increment_batch_size` in
`bench/`. Deterministic — no seeds — so every number reproduces exactly.*

**Scope, and it is a hard limit:** a *synthetic* device. `t(B) = t_fixed + B/sps_max`,
`sps_max = 5000`, mk_dev config (`batch_growth_factor 1.65`,
`batch_growth_min_throughput_gain 0.05`, `batch_knee_recheck_steps 2000`), one
`equilibration` stage, `train_mode fused`, no OOM. These are properties of the **control
law given a cost model**, not of any hardware. Nothing here may be quoted as a batch size.

**1. The descent terminates at the closed-form knee, not at 1.** With `_batch_floor`
removed, the walk runs down to the knee the cost model implies and stops:

| `t_fixed` | terminal batch | descent? |
|---:|---:|---|
| 0.0 | **1** | to the domain floor |
| 0.001 | 50 | yes (knee_bound 36) |
| 0.01 | 366 | yes (knee_bound 364) |
| 0.1 | 4491 | **none** |

Reaching batch 1 requires `t_fixed` **exactly** zero — a measure-zero, physically
impossible point, and the only point at which "descends forever" is reachable. The real
defect is *"descends to a level the configuration never chose"*, which is what actually
cost prod0810 (a whole stage at 0.825× its configured batch). `bench/old`'s only floor
test sits exactly at that impossible point.

**2. The floor stops the descent, not the churn.** With the floor intact under flat
throughput the batch oscillates `1000 ↔ 1650` **permanently** — 57 transitions in 60k
steps, `n_distinct = 2`. A horizon-invariance check on `n_distinct` alone calls that
converged, so `n_transitions` must be reported beside it.

**3. At zero switching cost the OBJECTIVE IS BLIND to the descent.** Measured regret of
the floorless arm against the floored one: **±0.0000** at both 20k and 60k steps. This
refutes the natural assumption that a throughput objective convicts trap (b);
`train.py:417-419` already says why ("flat throughput genuinely does argue for the
smallest batch… gradient quality is not something it can see"). Detection therefore needs
a **structural invariance** — on a stationary device a converged controller's
`n_distinct` must not depend on the horizon: measured **2/2/2** with the floor, **13 → 19**
without it, at 20k/40k/60k.

**4. Once recompiles are charged the SHIPPING controller loses to the worst fixed batch.**
At `recompile_s = 30`, flat curve, 20k steps: shipping **4927.3** samples/s against a
*worst* fixed arm of **4962.8**. The churn in (2) buys nothing and pays a recompile per
distinct shape. Pinned as an assertion in the true direction in
`bench/test_batch_traps.py` so it is not rediscovered.

**5. Trap (a) needs dominance PLUS a retained growth; dominance alone convicts the
shipping controller.** On the umaperf0812 table driven as a device, 6000 steps:

| arm | sps | occ % | final `B` | distinct |
|---|---:|---:|---:|---:|
| `null` | 57.70 | 52.0 | 100 | 1 |
| `ship` | 57.14 | 51.6 | 100 | 2 |
| `ship+occfloor` | 24.41 | 42.0 | 741 | 5 |

The shipping controller is dominated by null — worse on both axes — because one
exploratory probe up a declining curve costs **0.97%**. That is the correct cost of
learning the curve declines, not a defect. The structural separator, which needs no
magnitude threshold: the injected arm **retained** its growth (`net_rungs 641`), the
shipping controller ended where it started (`net_rungs 0`). Negative controls on RISING
and FLAT occupancy report UNRESOLVED, not PASS.

This was found by the false-positive check, not by the detection check — a detector that
reddens for current and injected code alike distinguishes nothing.

**6. The two candidate objectives have opposite argmaxes.** `samples_per_sec` is maximised
at the largest rung, `updates_per_sec` at the smallest, on the same saturating curve.
Accumulation engages **strictly below** `fused_grad_accum_min_samples` (`train.py:2467`),
and mk_dev ships `batch_size == fused_grad_accum_min_samples == 1000`, so **every
reachable batch sits where the identity justifying `samples_per_sec` does not hold.** That
identity is asserted at four sites as if unconditional.

Feeds [`design/phase6_batch_sizer.md`](design/phase6_batch_sizer.md) and
[`design/phase6_measurement_delta.md`](design/phase6_measurement_delta.md) Δ6, Δ8.

---

## F-036 · The utilization proxy's window is 48–288× slower than the batch actuator on ELJ, and inverts on MLIP · `MECHANISM`

*2026-08-16. Derived from the shipped sensor code and the shipped growth cadence;
no run required. Input to Phase 6's control law.*

**Scope:** `gpu_util_policy_window_s: 7200`, `gpu_util_sample_period_s: 60`,
`batch_growth_interval: 50` (`configs/mk_dev.yaml`, transcribed in
`bench/fake_modeller.MK_DEV_BATCH`); sampler cadence quantised to step boundaries
per `train.py::_sample_gpu_util`'s once-per-step gate; 5-sample floor per
`_gpu_util_mean`. Step times: 1.4 s measured on the dev box, 181–262 s measured on
prod0810's MLIP arms; the two A100 ELJ rows are estimates and are labelled as such.

How many growth decisions fire inside **one** `gpu/util_policy` window:

| route / step time | growth interval | `util_recent` | `util_policy` | policy ÷ interval |
|---|---:|---:|---:|---:|
| ELJ dev box, 1.4 s *(measured)* | 70 s | 14 smp | 119 smp | **103×** |
| ELJ A100 batch 1000, ~0.5 s *(est)* | 25 s | 15 smp | 120 smp | **288×** |
| ELJ A100 batch 7410, ~3.0 s *(est)* | 150 s | 15 smp | 120 smp | **48×** |
| MLIP prod0810, 181 s *(measured)* | 9050 s | **4 smp — ABSENT** | 39 smp | **0.8×** |
| MLIP prod0810, 262 s *(measured)* | 13100 s | **3 smp — ABSENT** | 27 smp | **0.5×** |

**The ratio spans ~600× across routes, and it crosses 1.** On ELJ the controller
takes 48–288 actions before its priority-1 sensor reflects the first one, so
`gpu/util_policy` **cannot be a closed-loop input there at any gain** — the dead
time exceeds the actuation period by two orders of magnitude. On MLIP the ratio
inverts, but `gpu/util_recent` is simultaneously **absent** (3–4 samples against
the 5-sample floor), so the only surviving reading is the slow one and the growth
interval is 2.5–3.6 **hours**.

**Consequence for Phase 6:** priority 1 cannot be a servo on the live reading.
It has to be **feed-forward** — a predicate over a calibrated `U(B)` — with the
live proxy demoted to a slow monitor that can *invalidate* the calibration but
never steer step-by-step. This is independent of what the cluster's statistic
turns out to be: it follows from the window length alone, so it does not wait on
the Phase 4 data.

**Also:** no single control cadence is correct for both routes, so a cadence is a
per-route **parameter**, not a constant. `docs/design/phase6_measurement_request.md`
§2 records the sensor mechanics that produce these numbers but does not draw the
control-timescale consequence.

---

## F-035 · `bench/old`'s pytest exclusion parked 58 live tests; the load-bearing half was the LR half, not the batch half · `MECHANISM`

*2026-08-16. Settled by mutation, not by reading. Changed `pytest.ini`.*

**Scope:** `bench/old/`, project venv, dev box. Whole directory:
**111 passed / 3 skipped in 65 s** — nothing in it is broken or stale.

Two mutations, each a one-line monkeypatch applied via a pytest plugin:

| mutation | `bench/old` | collected suite |
|---|---:|---:|
| `train.Modeller._batch_floor → 1` (re-introduces trap (b)) | **3 RED** | **3 RED** |
| `LRController.on_calibration → no-op` | **12 RED** | **1 RED** |

**The batch half was the less interesting answer.** Under the floor mutation
`bench/old` reddens `test_flat_throughput_walks_down_only_to_the_floor`,
`test_gain_at_or_above_factor_minus_one_rejects_every_jump` and
`test_A4_pin_does_not_rebuild_the_oom_sawtooth` — but the **already-collected**
`bench/test_oom_ceiling_expiry.py` reddens too, and 2 of its 3 are genuine
detections (`batch stuck at 303`; the descent `1000 → 606`). The third is only a
scaffolding precondition (`assert m._batch_floor() == 1000`) noticing its own
setup moved. So the floor was **not** unprotected, and
`docs/design/phase6_measurement_request.md` §8's "the controller's entire
regression protection is parked" overstates it for the batch half.

**The LR half is where the coverage actually was.** `bench/old/test_lr_controller.py`
drives the **shipping v8** `LRController`. Neutering `on_calibration` reddens 12
named behaviours there — warmup hold, asymmetric update, peak bounds, the
permanent divergence ceiling, the servo cut from a hot start, saturated-sensor
open loop, unresolved/inconsistent producing no move, the unmanaged-key control
arm — against **one** red in the entire collected `bench/`, and that one is
`test_arms.py::test_ray_is_distinguishable_from_null`, which reports that *an arm
went inert*, not which behaviour broke.

**Fix:** `bench/old` dropped from `norecursedirs`; new `bench/old/conftest.py`
carries `collect_ignore` for the three files that genuinely are retired
(`test_scenarios.py`, `test_off_target.py`, `test_crucible_feasibility.py` — each
self-tests the apparatus the 2026-08-13 review condemned). Whole-suite collection
**873 → 931**; `pytest -m fast` stays green (719 passed / 3 skipped / 209
deselected, 103 s).

**A test failing under a mutation is not automatically detection.** Read the
failing line: a broken precondition and a caught bug look identical in the summary.

---

## F-034 · The ranking holds on a second surface family, and the blind ramp is worse than no sensor there · `REPLICATED`

*2026-08-13. Second battery for §0's table; same arms, same scoring, an unrelated
surface family.*

**Scope:** `bench/crucible.py EQ_HARD` — 8 three-player `equilibration` cells swept
along noise, coupling (a=b), buffer churn κ, width and flow rate; 3000 steps, 5
scenarios, 20 seeds. 2 cells refused (best fixed rate at the edge of the searched
range). Compare against the 13 held-out cells, 11 of which are one quadratic bowl.

| arm | % over | passable only | vs. NULL |
|---|---|---|---|
| `hyper` symmetric | 23.7% | **8.4%** | −4.0 |
| `hyper` 2:1 / gated | 24.2% | 9.0% | −3.4 |
| `ray+ray` | 24.7% | 9.6% | −2.8 |
| NULL (no sensor) | 27.0% | 12.4% | — |
| `ramp+plateau` | 29.3% | 15.2% | **+2.8** |

**The order is identical to the held-out battery**, which is the first evidence
that §0's ranking is a property of the arms rather than of the quadratic bowl.

**`ramp+plateau` is worse than standing still on a multi-player surface.** On
`eq kappa.002` — a nearly frozen buffer (κ=0.002), and the one cell here that
genuinely discriminates — it fails 45% of `drift_10x`, 45% of `regime_change` and
85% of `mixture_drift`, against `hyper sym` at 5% overall and NULL at 25%. A
sensor that cannot measure and ramps anyway is a liability once a second player
is moving.

**But the sensors are worth much less here**: 4 points over the control, against
16.5 points on the held-out battery. Most of these cells are saturated rather
than discriminating.

**A saturated column the guard does not catch.** On `eq n3` every arm fails 100%
of `hot_90pct`, NULL included — the same signature as F-032's cold starts, and
the same arithmetic with the sign flipped: the deep target is at 56 of 3000
steps, so the budget is 112 steps, and cooling from `hot_lr(0.9)` at the maximum
`exp(−0.02)` per step does not fit. Until `_cold_start_feasible` grows a hot
counterpart, read `eq n3`'s 40% as a budget property.

---

## F-033 · `cos(g_t, g_{t-1})` measures the LEARNING RATE, not the gradient noise · `REPLICATED`

*2026-08-13. Retires the "calibrate the bench's noise axis" action in
`lr_control_summary.md` §0(e) — not by completing it, by showing the axis it
wanted to calibrate against is not the axis `cos` responds to.*

**Scope:** bench `mle` surface (convex quadratic + optional quartic), dim 32
unless stated, 2000 steps, cos measured at a FIXED rate with the servo off, one
seed per cell, oracle rate from seeds 0–2. Real numbers: T=10, elj nehzor sg14
Z'=1, 400 steps, **n=1 run per regime**. Measured by `bench/cos_axis.py` and
`bench/calibrate_noise.py`.

**The two sweeps, same surface, same statistic.** Median cos at each cell's own
optimal rate, against median cos at multiples of it:

| noise (at optimal rate) | 0.01 | 0.1 | 0.5 | 2 | 5 |
|---|---|---|---|---|---|
| d=32 | 0.998 | 0.841 | 0.175 | **−0.011** | −0.061 |
| d=2048 | 0.996 | 0.709 | −0.030 | −0.127 | −0.136 |

| rate / optimal | 0.125× | 0.25× | 0.5× | 1× | 2× | 4× |
|---|---|---|---|---|---|---|
| noise 2 | 0.995 | 0.964 | 0.648 | −0.011 | **−1.000** | −1.000 |
| noise 0.5 | 1.000 | 0.998 | 0.968 | 0.175 | **−1.000** | −1.000 |

A 200× noise sweep moves cos about as far as a 16× rate sweep, and **the zero
crossing sits at ~1× the optimal rate at both noise levels** — which is what
hypergradient is for, and it means a cos value does not identify a noise level.

**At 2× optimal cos is −1.0000 in all four quartiles of the run** — exact
period-2 oscillation, the classic signature above 2/λ. So the statistic
saturates hard on the hot side and is an unambiguous too-hot detector, while on
the cold side it is compressed into 0.96–1.00 across a 4× span.

**The anchor number in the summary does not reproduce.** "0.29 at noise 2" is
−0.011 when measured at the oracle rate; 0.29 corresponds to roughly 0.6–0.8×
optimal. The figure had no code behind it — it appeared as prose in three files
and `cos_axis.py` is the first implementation.

**Real gradients, per branch:**

| regime | stage | branch | median | IQR | ×null |
|---|---|---|---|---|---|
| eq mid-descent (step ~11000) | equilibration | `fused` | 0.2901 | 0.240–0.350 | 903 |
| eq from phase1_exit | equilibration | `fused` | 0.2871 | — | 893 |
| eq mid-run (step ~10500) | equilibration | `fused` | 0.2889 | — | 899 |
| **MLE fresh (step 0)** | `train_prior` | `bwd` | **0.3441** | **−0.649–0.771** | 1071 |
| MLE half way (step 5000) | `train_prior` | `bwd` | −0.1205 | −0.379–0.324 | −375 |
| MLE converged (step 10000) | `train_prior` | `bwd` | −0.0448 | −0.277–0.275 | −140 |
| eq mid-run (same run, same window) | equilibration | `bwd` | 0.0150 | — | 47 |

**The MLE trajectory runs slightly HOT and equilibration runs COLD.** Read as a
rate statistic, `bwd` goes +0.344 → −0.121 → −0.045 across one MLE run — at or
above its optimal rate from the middle onwards — while `fused` sits at +0.29
throughout. They do not want the same rate, and today they get the same one. At
step 5000 the gradient norm is flat (51.13 → 51.31 over 400 steps), so that model
is at its noise floor and the negative reading is the expected bouncing
signature; it also suggests the back half of phase 1 may be buying little.

**The converged measurement reproduces −0.0448 to four decimals** against the
file the broken `mle_fresh` regime wrote this morning — same 399 samples, same
steps 10001–10399, same ‖g‖ 50.96. That is not consistency with the diagnosis,
it is proof of it: those two regimes were always one measurement.

The `fused` medians — 0.2871, 0.2889, 0.2901 — are three separate runs at three
checkpoints and agree to within 0.003, which is the only replication in this
table.

**CORRECTED, same day, before this entry was relied on:** a fourth `fused`
window exists — the post-fix `eq_phase1exit` re-run, over the *identical* step
range 10642–11039 as the 0.2871 reading — and it reads **0.3037**. Four windows
give a median of 0.2895 and a full spread of **0.0166, not 0.003**. The
replication stands (four windows within 2% of each other, against a null of
0.0003); its tightness was overstated ~5x. The two readings over the same steps
differ because the checkpoint itself was rewritten between them by the
clobbering bug, so they are not the same model.

Null |cos| for independent vectors is `sqrt(2/πd)` = 0.00032 at the policy's
6,163,969 params, against 0.141 at the bench's d=32.

**Two consequences the median alone hides.**

1. **No bench cell reproduces the real fused distribution.** Real fused is 903×
   the null with an IQR 0.11 wide; the best bench cell is 56× (d2048, noise
   0.01) and every cell's IQR is ≥ 0.3. Matching on the median would place the
   real system in a cell where one reading is near-worthless (2.1× chance at
   d=32) when the real reading is essentially noiseless. **§0(e)'s worry that
   real training is an order of magnitude noisier than anything tested is
   refuted for the fused branch** — it is cleaner, not noisier.
2. **Phase 1 emits no `fused` steps at all.** `train_prior` is `bwd`-only, and
   its cos has a fused-like median with a **13× wider IQR** (1.42 against 0.11).
   A hypergradient controller running through phase 1 is running on a different
   statistic than the one measured in equilibration, not a noisier version of
   the same one.
3. **`bwd` is unusable everywhere except fresh.** 0.0150 mid-run and −0.0448 at
   convergence, against a null of 0.00032 — nominally significant and far too
   small to control on, while `fused` in the SAME window reads 0.2889. The
   branch, not the noise, is what separates them.

**Reading of the real number:** fused cos 0.29 says the production rate sits
somewhat BELOW its optimum — mildly cold — at whatever the noise is. It is not
evidence about noise.

---

## F-032 · Three of thirteen crucible cells forbid a cold start, by the shipping controller's own bounds · `MECHANISM`

*2026-08-13. Corrects the headline table in `lr_control_summary.md` §0, which
was inflated by ~4.6 points for every arm including a hypothetical perfect one.*

`peak_scale` is bounded by `adaptive_lr.bounds`, which `mk_dev.yaml:48` ships as
`[0.01, 2000.0]` and `bench/fake_modeller.py:47` mirrors; the seed rate is
`1.25e-4`. Hypergradient's climb is `exp(hyper_beta·cos)` with `cos ≤ 1`, so
closing a gap of R takes at least `ln(R)/hyper_beta` steps. Two independent
walls, both properties of the budget rather than of any arm:

| cell | optimal rate | peak_scale needed | min climb steps | budget (2×denom) | wall |
|---|---|---|---|---|---|
| `h eq w_rep.3` | 1.23 | **9840** > 2000 | — | 238 | cap |
| `h eq base` | 0.433 | **3464** > 2000 | — | 100 | cap |
| `h cond=30` | 0.0351 | 281 (ok) | **282** | 256 | deadline |
| `h baseline` | 0.00433 | 35 | 177 | 1052 | — passes |

**The predicate reproduces the measured column on all 13 cells with no
disagreement** (`bench/test_crucible_feasibility.py`, 20 seeds each): every cell
it rejects scored 100% of cold starts for *every* arm, every cell it accepts was
passed by all three hyper variants at 0%. Nothing lands in between, which is the
tell that the column was structural rather than a controller property.

**Where it came from:** wiring in `_time_oracle` (correct in itself — the
denominator must be selected on the metric it denominates) collapsed the
equilibration denominators, `h eq base` to 50 steps of a 3000-step run, and the
cold-start budget went with it. This is the mirror image of the unreachable
budget in §7: `_oracle_task` refused cells whose budget was too LOOSE to fail
and had no check at the other end. It does now, and `crucible.main` prints a
second `passable only` aggregate excluding those columns.

**The generalisable part:** hypergradient's cold-start recovery has a hard floor
of `ln(lr*/lr_seed)/hyper_beta` steps — 177 for a 35× gap at β=0.02, 408 for a
3464× gap — and a ceiling of 2000× the seed rate that no amount of time fixes.
Raising β moves the first and not the second.

---

## F-031 · `mol2cluster` FIXED; its energy cost is a rare heavy tail, not a broad bias; it is the only site · `MECHANISM`

*2026-08-13. Supersedes **F-030 item 2** on two counts: the defect is now FIXED (F-030 chose
not to), and F-030's severity framing was wrong in a way that mattered for how it was tested.
All numbers CPU-measured on the real 10000-row sg 9 Z'=2 elj prior.*

**1. FIXED.** The trailing `[1]` is gone; the `repeat_interleave` now aligns the per-graph
`T_fc` stack with the flattened centroids as intended. Because
`repeat_interleave(Zp, 0)` yields `[g0, g0, g1, g1, ...]`, `[1]` always resolved to **graph
0** — so one crystal's cell metric set `zp_buffer` for the whole batch.

**2. SEVERITY CORRECTED — the geometric shortfall is not the energy error.** F-030 reasoned
from buffer geometry to "biases the lattice energy". The buffer statistics were right
(median +0.13 Å, 5% of crystals short by more than the entire nominal cutoff, worst 20.3 Å),
but the energy consequence measured through `analyze(['elj'])` over 4 random 100-crystal
batches is **rare and concentrated, not a broad bias**:

| statistic | \|Δ elj\| (kJ/mol) |
|---|---|
| median | 0.0001 — numerically nil |
| crystals > 1.0 | **3 of 400** |
| worst single crystal | **126.1** (13% of that structure's \|elj\| ≈ 970) |
| one of the four batches | **no affected crystal at all** |

A generous supercell absorbs most of the shortfall. What survives is a rare, large,
draw-dependent corruption — which is still worth fixing, because 126 kJ/mol on one sample
makes a bad structure look good and get preferentially replayed, and because the error
depends on batch-mates, so a rerun over a reshuffled prior scores the same structure
differently.

**3. SCOPE — `mol2cluster` was the only instance.** Swept `mxtaltools/` for the class (a
per-graph geometric stack indexed by a constant). No `repeat_interleave(...)[N]` siblings.
The remaining `stack[0]` hits are all legitimate: `autoencoder_models.py` applies one
rotation to everything *on purpose* (equivariance checks); `crystal_reduction.py` compares
graph 0 against graph 1 *by design*; `featurization_utils.py:540,568` reads subunit 0 of a
Z'>1 **rebuild** whose subunits share one cell and one space group by construction (the
adjacent `any(nonstandard_symmetry)` shows the author knew); `ase_interface.py:146` is the
single-crystal branch. Grep found candidates; only reading them could classify them.

**4. TESTED by a metamorphic property, not a golden value** —
`test_batch_invariance.py`: *a crystal alone == in a batch == in a shuffled batch*. Any
collapsed-stack bug violates it by construction. Correct code is **exactly** invariant
(max \|Δ\| = 0 across composition and order, Z'=1 control and both Z'=2 priors, including a
worst-case pool with 29.5 Å of shortfall on tap). 9/9.

**The first version of this test was blind and passed.** It drew 12 crystals at random and
used a scale-relative `1e-3 × 970 ≈ 1.0` kJ/mol tolerance — sensible for an O(1000) quantity,
and larger than the effect on 397 of 400 crystals. Two changes made it real: an **absolute
1e-2** tolerance (100× above the 1e-4 two-call noise floor, 5× below the faintest observed
signal) and an **adversarial pool** — minimum-buffer crystal pinned at position 0, widest
crystals behind it — because a random draw was a coin flip. The negative control re-patches
`mol2cluster` itself (not the shared `fractional_transform`, which would fail for unrelated
reasons) and fires at **110.8 kJ/mol vs 0.01 tolerance**.

**5. The four existing Z'>1 priors do NOT need regenerating.** All Z'=2, all carrying baked
`mace` energies: `acridine_sg14_zp2_mace`, `acridine_sg9_zp2_mace`, `deadrow10k_sg14_zp2`,
`deadrow10k_sg9_zp2`. Two independent reasons:

- **Training never reads the baked energies.** `train.py:1660` re-analyzes the whole prior
  unconditionally (`if True:`), so every run since the fix already scores correctly.
- **Prior *contents* are energy-independent.** `generate_sg_prior.py:130` selects on
  `compute_cell_reduction_penalty < 0.01`, which reads only `cell_lengths`, `cell_angles`,
  `sg` — no cluster build — and the energy pass runs *after* selection (lines 147–169). So
  the bug could not have biased which structures are in a prior. `thermal_scaling_factor` is
  1 (neutral) on both acridine files and absent on both deadrow files.

Residual exposure is limited to **offline readers of a baked `mace` field**, which no
training path uses. Untested corollary worth noting rather than claiming: the `deadrow10k_*`
files are named `_elj` but carry `mace` — a naming inconsistency, not a correctness issue.

---

## F-030 · Two pre-existing Z'>1 defects in mxtaltools, silent by construction · `MECHANISM`

*2026-08-12/13. Adversarial audit of positional slices at Z'>1, prompted by D33's first
Z'=2 run. Both established from index arithmetic AND confirmed empirically on CPU with a
real CSD molecule at sg 9. Neither is caused by D33 — D33's Z'=2 work is what reached them.*

**1. FIXED — the rotation-magnitude clamp protected only the LAST aunit.**
`crystal_ops.py::latent_to_cell_params`. The layout is
`[3 lengths | 3 angles | 3*Zp centroids | 3*Zp orientations]` with BOTH aunit blocks
flattened, so aunit `ind`'s rotation magnitude sits at `6 + 3*Zp + 3*ind + 2`. The code
used `5 + 6*(1 + ind)` = `11 + 6*ind`:

| Zp | clamped | correct |
|---|---|---|
| 1 | [11] | [11] — identical, correct by coincidence |
| 2 | [11, 17] | **[14, 17]** — 11 is `centroid[1][z]` |
| 3 | [11, 17, 23] | **[17, 20, 23]** |

MEASURED at Zp=2, sg 9: latent row 14 driven to −1.0 gave `|rotvec|` **exactly 0.0** for
aunit 0 — the r=0 singularity the line exists to prevent, and the one
`compute_jacobian`'s `log(sin(r/2))` clamps at ~37 nats. Row 17 (the last aunit) correctly
floored at 0.0314. **Silent by construction**: the wrong index `5 + 6*Zp` is exactly
`width − 1`, so it can never IndexError.

Fixed, and the fix is **byte-identical at Zp=1**, so no current work changes. Covered by
`test_dead_latent_rows.py::test_rotation_magnitude_clamp_covers_every_aunit` (floors every
aunit at Zp=1/2/3).

**2. NOT FIXED, deliberately — `mol2cluster` computes Z'>1 padding in the wrong cell.**
`crystal_building.py:176-178`:
`fractional_transform(frac_centroids, self.T_fc.repeat_interleave(max_z_prime, dim=0)[1])`
— the trailing `[1]` reduces the `(n*Zp, 3, 3)` stack to ONE matrix, and
`fractional_transform_torch` dispatches `(n,3)+(3,3)` to a per-point transform. So **graph
0's cell metric is applied to every crystal in the batch** when computing `zp_buffer`, the
extra supercell cutoff for Z'>1 clusters. Under-padding drops neighbours from the lattice
energy. `max_z_prime > 1` branch only. (The `# [n, Zp, Zp, 3]` comment on the next line is
also wrong — `.norm(dim=-1)` leaves `[n, Zp, Zp]`.)

Left unfixed on purpose: unlike the clamp this is NOT a no-op for anything — it changes
computed lattice energies for every Z'>1 run — so it wants a physical Z'=2 validation
before being touched. Fix is to drop the `[1]` and let the per-graph stack broadcast.

**3. NOT FIXED — every conformer batch-growth knob is inert, with two latent crashes
behind it.** `train_conformer.py:458` borrows `Modeller.increment_batch_size`, which reads
its knobs off TOP-LEVEL `self.args`; `conformer_dev.yaml` nests them under `training:`. So
`grow_batch_size` is visible and true while `batch_growth_interval` etc. are None →
`wait=0` → it grows EVERY step at the default 2.0 rather than 1.65 every 50. Measured:
**batch 2048 by step 4**, against a config whose own comment says "CPU: no OOM headroom to
climb into". `__init__` bridges exactly one key (`max_batch_size`), which is the only
reason nothing crashes. Exposing the rest reaches missing attributes:
`knee_on=True` → `AttributeError: '_recent_step_work'` at step 50;
`max_step_seconds` → `AttributeError: 'NoneType' has no attribute 'stage'` at step 9.
Untouched because `train_conformer.py` is under active heavy edit and
`conformer_dev.yaml` is user-owned. Note `deadrow_aug12`'s prescribed acceptance test for
the conformer arm ("if it reaches step 1 the delegation works") cannot see this — the
first crash is at step 50.

**Verified CLEAN in the same sweep:** `compute_zp_order_penalty` is safe by construction —
swept all 230 space groups and at Zp=2 NO space group has a dead row at index >= 6, so
`raw_latents[:, 6:6+3*k]` never reads a pinned row. But **68 space groups have a dead
centroid row at Zp=1**, so if `free_centroid_rows` ever stops returning `()` above Zp=1,
this penalty would compare a pinned aunit 0 against a free aunit 1. `compute_jacobian`'s
`[:, -3*max_z_prime:]` correctly selects 12-17 at width 18. `TorsionGFN`'s delegation is
COMPLETE — all 13 attributes `_finalize_dim_partition` assigns are present and sane, fwd
and bwd roll, and a pass-through dead row genuinely pins. `aunit2ucell` is now safe only by
ROUTING: every Zp>1 caller goes through `split_to_zp1_batch()` first, so it never sees the
flattened 6-wide centroid that produced the 7-vector-vs-4x4 break in F-028.

---

## F-029 · Four reported metrics are diluted by `live_dim/dim`, and `wass_debiased` is a GATE · `MECHANISM`

*2026-08-12. Adversarial audit of every full-width per-dimension consumer, verified by
reading the code and running each claim on CPU. Sites confirmed independently afterwards.*

**Scope:** mechanism, so it generalises across space group. Magnitudes are per crystal
system. Applies wherever `hold_dead_latent_rows` resolves a non-empty set.

Dead rows sit at exactly 0.0 in `latent_params()`, so anything that AVERAGES over the
dimension axis includes rows contributing a hard zero:

| metric | site | reads |
|---|---|---|
| `Total Var` | `train.py:3772` | 83% monoclinic 10-15, 75% ortho / sg 3-5 / sg 1, **67%** sg 6-9 |
| `Total Mean` | `train.py:3773` | same factor, biased toward zero |
| `Mean Cell KLD`, `Mean Latent KLD` | `eval/evaluations.py:620` | same (a constant dim gives KLD exactly 0.0) |
| `wass_debiased` | `eval/evaluations.py:1636` | scales `~sqrt(live/dim)`; **measured 0.933** vs 0.913 predicted |

`Max Cell/Latent KLD` are CLEAN — a zero never wins a max.

**`wass_debiased` is the one with teeth: it is a live GATE**, not just a readout — one of
`mk_dev`'s three stage-1 exit terms (`below: 0.015`). The `theta = randn(D, n_proj)` unit
directions are normalised over FULL width, so only `||theta_live||` of each direction does
work, and `raw − null` does not cancel it (both terms scale). A threshold tuned on
triclinic sits ~7% loose for monoclinic, ~13% for orthorhombic. Fix is to build `theta` on
live dims, or scale by `sqrt(dim/live_dim)`.

**NONE of these jumped at D33 — they are not regressions.** `latent_params()` already
clobbered dead rows to 0.0 beforehand; that WAS the defect (see D33 "The defect"). So they
have always read low, and D33 only makes it systematic rather than incidental. Correcting
them is therefore a **RULER CHANGE**: it would move numbers that have been read for weeks
and break comparability across older runs. Left unfixed pending an explicit call.

**Tied rows compound it.** For trigonal/tetragonal/hexagonal/cubic, `a=b` (and `c` for
cubic) is a dependent direction that D33 deliberately does NOT treat as dead
(`dead_latent_rows.py:41`, `:260` — it needs reparameterisation, not row deletion). So
`live_dim` over-counts by 1-2 there, and the denominators in every item above, plus
`u_star` and `step_var`, are correspondingly too large.

**FIXED outright (not a ruler change):** `compute_1d_kld` raised
`ValueError: operands could not be broadcast together with shapes (201,) (2,)` when
EXACTLY ONE input was near-constant — the constant side short-circuits to a 3-point grid,
the spread side returns `n_kde_points+2`, and the function integrates one against the
other's grid. Unreachable from dead rows (both sides constant, symmetric, correctly 0.0)
but reachable from a collapsed sampler dim against a broad buffer dim, and it would take
out `buffer_kld` and the whole eval metrics block. Now returns **NaN**, which is the
honest answer — a constant-vs-spread pair has infinite KL, and 0.0 would read as
"these distributions agree".

**Verified CLEAN, with the check that earned it:** `Effective Dimension` (a 10-dim cloud
embedded in 12 with two rows pinned gives d_eff 4.311906 vs 4.311907 bare — the spurious
singular values land at ~1e-7 and drop out); `step_var`/`terminal_var`/`u_star` (already
live-dim); the Gaussian diagnostics (`_mean_over_live` applied at all three return sites,
and `Mean F Rho` is doubly correct since `dplr_zero_mask` also zeroes it); buffer distances
(`max|cdist_full − cdist_live| = 4.8e-07` — Euclidean distance is additive and a dim
holding the same constant in both rows contributes zero, so `dup_cutoff` / `d_cut` /
`distance_threshold` are UNSHIFTED); `box_violation`; the DPLR path (`u_hat` normalises
over the RANK axis, not the dim axis, so dead dims cannot dilute it).

**Also found, low severity:** `substitute_prior` (`utils.py:863`) and
`calibrate_prior_noise` (`utils.py:985`) normalise `rand_dir` over full width, so ~9-13%
of the configured noise magnitude lands in dead rows and is discarded by `_pin_dead`. The
latter self-corrects (it measures reward response); the former takes its range from config
and is quietly short. **Cosmetic width bugs** (F-009b siblings): `mean_flow_step_sizes`
and `visualize_latent_trajs` hardcode 12 latent labels, so at Z'=2 the axis is mislabelled
and six dims are never drawn; `log_buffer_kld`'s length guard means the entire KLD family
**silently vanishes** at Z'=2 rather than mislabelling — right shape, mute failure.

---

## F-028 · The dead-row runtime probe was silently absent at Z'>1, on the layout with the least other coverage · `MECHANISM`

*2026-08-12. Found by the first Z'=2 training run ever attempted with rows held
(`configs/zp2_smoke.yaml`, sg 9, Z'=2, elj, 300 steps). Reproduced and fixed on CPU;
regression test added.*

**Scope:** any `is_crystal` run with `max_z_prime > 1`. Verified against four real priors
(sg 9 Z'=2, sg 14 Z'=2, sg 14 Z'=1, sg 1 Z'=1). Mechanism, so it generalises.

`_verify_dead_latent_rows` — the ONE runtime check that the tabulated dead rows still
match what the crystal build actually discards, and the main reason to trust D33 — called
`pose_aunit()` + `build_unit_cell()` on its probe batch before probing. Those calls are
**dead weight**: the probe only drives `latent_to_cell_params -> latent_params`, which
needs cell parameters and nothing built from them.

They also **raise at Z'>1**. `aunit2ucell` assumes a 3-wide centroid, but at Z'=2
`aunit_centroid` is stored FLATTENED as `(n, 6)`; appending the affine 1 yields a
7-vector against a 4x4 operator:
`einsum(): subscript j has size 7 for operand 1 which does not broadcast with previously
seen size 4`. The surrounding `except Exception` then swallowed it and printed
*"the tabulated rows for SG9 are UNVERIFIED this run"*.

**So every Z'>1 run had no runtime verification of the dead-row table** — the layout with
the least unit-test and zero end-to-end coverage — and it announced this only as a startup
WARNING, which is exactly the kind of line nobody reads. The probe itself was never wrong:
called directly on the same batch it returns `(3, 5)` correctly, and the Z'=2 subsample is
well-formed (aunit attrs are flattened per graph, so `size(0)` is `n_graphs` and
`subsample_new_batch` slices correctly).

**Fixes.** (1) Dropped the two unnecessary calls; the probe now runs at Z'=2 and Z'=1
alike. (2) `test_dead_latent_rows.py::test_probe_actually_RUNS_on_real_priors_including_zprime_2`
asserts the probe COMPLETES and prints its confirmation — not merely that the table is
right, which was the only property previously tested. (3) The outcome is now logged as
`dead_rows/probe_verified` (1 verified / 0 unverified / absent if not applicable), so an
unverified run is queryable after the fact rather than depending on a scrolled log.

**The generalisable lesson: a swallowed diagnostic is worse than no diagnostic**, because
it reports as reassurance-shaped silence. Two properties need testing separately — that a
check gives the right answer, and that it RAN. Only the first was covered here.

*Same class as the earlier vacuity guard in `test_latent_gaussian.py` (a comparison loop
that found nothing would have passed by default) — third instance of this pattern in this
work.*

---

## F-027 · The rows-live floor is NOT a finite-T effect, and it is not a problem to solve · `OBSERVED`

*2026-08-12. `configs/gauss_aug12/par2_t25`, e_sg1_off at trajectory T=25, SOLO on the
card, 3840 steps, pinned LR 1e-5. Compared against the same arm at T=10 (F-026).
Supersedes two claims in F-024 and F-026.*

**Scope:** analytic energy, sg 1, Z'=1, one seed, T ∈ {10, 25} only.

**SUPERSEDES F-024/F-026's finite-T claim, which is REFUTED.** Both entries argued the
rows-live floor was a finite-T representational limit — a short gaussian chain unable to
reproduce a flat box marginal — predicting the floor shrinks as T grows. It does not.
At 2.5x the trajectory length and 2.5x the steps, `eval_fwd/logw_std` plateaus at ~1.5
across steps 1050-3750, the same 1.16-2.24 band T=10 occupied, and the log Z scatter is
still ±0.5 at step 3750.

**SUPERSEDES F-026's "variance grows with training".** It is STATIONARY, not growing.
T=25's longer window shows ~1.5 throughout; re-reading T=10 (1.16, 1.34, 2.03, 1.73,
1.62, 1.50, 1.48, 1.31, 2.24) it oscillates rather than trends, and d_sg4_off's
1.16 → 4.59 → 3.02 came back down. Short-window noise read as a trend.

**What stands, and is enough.** The rows-live arm reaches its analytic log Z and
STRADDLES it — T=25 excursions of +0.150, +0.838, +0.306 as well as negative, matching
d_sg4_off crossing to +0.111 at T=10. So `Δ = n_dead·log(2 + √(π/k))` (F-023) is
confirmed from both sides; what the live arm lacks is stability, not accuracy.

**Why this line of investigation is CLOSED.** Two separate things were being conflated:
- correction ON: `logw_std` 0.07-0.15, err ±0.002, loss 0.003. That is the ordinary
  noise / finite-convergence floor. Nothing to explain.
- correction OFF: the RETIRED method. Its floor is the documented cost D33 removes.

Characterising the rows-live dynamics further (an LR sweep was drafted, on the better
hypothesis that a flat-direction random walk scales with step size rather than chain
length) would explain the behaviour of a configuration we have deliberately stopped
running. It cannot change what we do next, so it fails the write gate and was dropped
rather than deferred. The harmonic-pin alternative is already answered analytically in
D33 (`0.5·log(2πT/c)` per row, walk bounded to σ=√(T/c), zero only if the row is held).

---

## F-026 · Dead rows are a FLAT DIRECTION: the rows-live arm reaches the right log Z and then walks off it · `OBSERVED`

*2026-08-12. `configs/gauss_aug12/par4b` run-set 2: sg 4 and sg 1, held vs live, 4 arms,
1 seed, stopped early at ~1500-1650 steps once the question was answered. LR PINNED at
1e-5 (no calibration, no sensor, flat envelope), stage 1 uncapped from `gates/mle_flat`
so it exits at step 150. `latent_gaussian`, T=1, width 0.1, mode 0.5, k=1, traj T=10.*

**Scope:** analytic energy, Z'=1, `periodic_centroids` false, ONE seed, T=10 only.

**The rows-live arms DO reach their analytic log Z.** `eval_fwd/emp_z` minus the closed
form, by step:

| step | d_sg4_off | `logw_std` | e_sg1_off | `logw_std` |
|---|---|---|---|---|
| 150 | −2.211 | 1.33 | −2.375 | 1.34 |
| 300 | −0.151 | 1.16 | **−0.006** | 1.16 |
| 450 | **+0.076** | 1.34 | −0.074 | 1.34 |
| 600 | **+0.111** | 1.49 | −0.182 | 2.03 |
| 1050 | −1.000 | 4.29 | −0.518 | 1.50 |
| 1500 | −0.480 | 3.02 | −0.525 | 2.24 |

So `Δ = n_dead·log(2 + √(π/k))` (F-023) is confirmed **in training**, not only by CPU
importance sampling — d_sg4_off crosses slightly *above* its target, which rules out a
one-sided approach artifact. Not a lucky crossing: both arms rise monotonically to the
value over 300-600 steps and then leave it.

**MECHANISM — the cost is failure to HOLD, not a wrong answer.** Departure tracks
`logw_std` growth (1.16 → 2.2-4.6) while the held arms move the other way (0.86 → 0.13,
err pinned at ±0.004 over the same span). Nothing anchors a dead row: every marginal
matching the flat soft-walled target is equally good and the gradient there is noise, so
the policy random-walks in that subspace and the log-weight variance grows with training.
This is the same phenomenon already recorded for `step_var`
(`project_step_var_is_the_flat_direction`), reached independently here.

**Held arms, all four space groups, ±0.002 nats** — and two of them by a code path
nothing else in either battery exercises:

| arm | dead rows | mechanism | err |
|---|---|---|---|
| c_sg19_on | (3,4,5) | `enforce_crystal_system` | +0.0004 |
| d_sg4_on | (3,5,**7**) | + `canonicalize_free_axes` | **−0.0003** |
| e_sg1_on | (**6,7,8**) | free axes ONLY | **+0.0019** |
| b_sg14_on | (3,5) | `enforce_crystal_system` | +0.0010 |

sg 4 and sg 1 land on sg 19's value (−12.4528, same `n_live`) through
`canonicalize_free_axes` rather than `enforce_crystal_system`. **First end-to-end
evidence for the free-axis half of D33**; `deadrow_aug12` states it cannot reach that
case at all, because a physical prior must be a real crystal.

**RETRACTED from F-024:** per-row excess `logw` variance was quoted there as 0.66-0.82
and near-constant. With four pairs it reads 0.66, 0.92, 1.71, 5.75. It is not a stable
quantity while the off arms are walking — of course it isn't, since the walk is what
generates it. Do not use those numbers.

**Still untested:** whether the floor shrinks with larger `T` (F-024's
finite-T-representational claim); and `a_sg2`, the empty-dead-set control, which was cut
because `test_dead_latent_rows` already proves that case BITWISE on CPU and a GPU A/B
would be strictly weaker evidence.

---

## F-024 · A live dead row traps the run in MLE, because `mle_flat` can never flatten on a zero-variance direction · `OBSERVED`

*2026-08-12. `configs/gauss_aug12/par4`, wave 1: sg 14 and sg 19, rows held vs live,
4 arms, 1 seed, 4000-step budget, 4-way co-tenant. `latent_gaussian`, T=1, width 0.1,
mode 0.5, k=1, trajectory T=10, LR servo ON (this is what F-025 then removes).*

**Scope:** analytic energy, Z'=1, `periodic_centroids` false, ONE seed. The mechanism
generalises (it is a property of MLE on a degenerate direction); the step counts do not.

Held-rows arms land on the closed form and stay there: `eval_fwd/emp_z` err **+0.0007**
(sg 14) and **+0.0012** (sg 19), reached at *stage-2 step 0* and held for ~2900 steps.
Rows-live arms did **not** confirm their value in budget: −0.455 and −1.148. So D33's
held-rows claim is confirmed end-to-end; **the rows-live half is not**, and this entry
is why.

**Stage allocation, which is the headline:**

| arm | total | stage 1 ends | stage-2 steps | min `bwd/mle` |
|---|---|---|---|---|
| c_sg19_on | 3440 | 450 | 2990 | −7.75 |
| c_sg19_off | 4000 | **3000** | **1000** | **−20.82** |
| b_sg14_on | 3330 | 450 | 2880 | −8.54 |
| b_sg14_off | 3410 | 600 | 2810 | −14.09 |

**MECHANISM.** Every prior row carries the dead rows at exactly 0.0, so MLE buys
*unbounded* likelihood by collapsing a delta onto a zero-variance direction: 13.1 nats
over 3 dims (4.4/dim) for sg 19, 5.6 over 2 (2.8/dim) for sg 14. The per-dim gain
differs because it grows as `log(1/σ)` with time — which is exactly why
`gates/mle_flat` **can never flatten**. Exit terms are ANDed (`should_exit` returns
False on any failing term), so that one un-flattenable term trapped c_sg19_off in
stage 1 for 75% of its budget while `bwd/tbc` sat at 0.04 against a 2.0 bar. **The
defect under test is what starved its own measurement.** 3 dead dims stalled the gate;
2 did not.

**The floor, on the fair clock.** Correcting for stage allocation does not remove it:
held arms reach |err| < 0.05 at stage-2 step **0**; b_sg14_off takes 300 stage-2 steps
and then drifts back out (−1.28, −0.74, −0.45); c_sg19_off never arrives in 1000.
`eval_fwd/logw_std` 0.069–0.074 held vs 1.28–1.41 live (~18×). Excess **variance** per
dead row 0.66–0.82 nats², bracketing the 0.5 expected if the policy leaves a roughly
gaussian marginal where the target is the near-flat soft-walled box. Provisional: the
sg-14 figure moved 0.67 → 0.82 between two reads and three arms were still running.

**NOT irreducible in principle.** A live dead row's target is a legitimate
distribution a GFN could match, driving TB to zero. The floor is a **finite-T
representational limit** — a 10-step gaussian chain from a delta cannot reproduce a
flat box marginal. Prediction: the floor shrinks with larger `T`. **Untested.**

---

## F-025 · The ray probe's saturation reading made the D33 A/B train its two arms 8× apart in LR · `OBSERVED`

*2026-08-12, same 4 arms as F-024. Stage-2 samples only.*

**Scope:** as F-024. One seed. A confound report, not a controller result.

| arm | scale swing | ends at | `cal_status` histogram |
|---|---|---|---|
| c_sg19_on | 16.0× | 0.0625 (floor) | `below_range` ×195 |
| c_sg19_off | 7.8× | 0.5000 | `below_range` ×1 |
| b_sg14_on | 11.3× | 0.0884 | `below_range` ×184 |
| b_sg14_off | 7.8× | 0.2726 | `below_range` ×100, `bracketed` ×92 |

`below_range` pinned is the **saturation** reading already on record, not "too hot":
once a held arm's loss reaches ~0.003 the probe cannot bracket α\* and falls to a
bound, and the controller keeps cutting on it — 16× down. The rows-live arms, still far
from converged, keep a high peak. **Consequence: the two arms of a pair were not
trained at the same learning rate**, so "rows-live converges more slowly" is partly
"rows-live ran at 8× the LR", and this data cannot apportion the two.

Costs nothing for the held arms (already exact from stage-2 step 0), but it means any
convergence-rate comparison from `par4` is confounded. `par4b` pins the LR at 1e-5 —
where the held arms' own controller settled and held err ~0.001 for 3000 steps, the
only LR *measured* stable here — with `ray_calibration` off, `lr_sensor: {kind: none}`
per stage, and a flat envelope (the warmup ramp **restarts at every stage transition**,
so leaving it on would place a ramp exactly where stage 2 begins).

Incidental: this toy is an unusually good bench for the saturation defect, since the
correct LR behaviour is knowable in closed form.

---

## F-023 · A held-out dead row costs `log(2 + √(π/k))` of fictitious volume, not `log 2` · `MECHANISM`

*2026-08-12. `test_latent_gaussian.py`, CPU, 51/51 checks. `latent_gaussian`
energy, T=1, width 0.1, mode 0.5, 20000 IS draws per cell. No training.*

**Scope:** analytic energy (`latent_gaussian`), Z'=1, `periodic_centroids: false`,
sg 2/14/19/4/1. Derived and then measured, hence `MECHANISM`; the derivation is
what generalises, the specific numbers are at k=1 unless a row says otherwise.

The target is `E(x) = ½Σ((x−c)/w)² + k·Σrelu(|x|−1)²` with `c` = mode on live
rows and 0 on dead rows, so

    rows HELD:  log Z = (n_live/2)·log(2πT) + n_live·log w
    rows LIVE:  log Z = ⟨above⟩ + n_dead·log(2 + √(π/k))

**The second term is what D33 removes,** and it is NOT `n_dead·log 2`. A
live-but-dead row is invisible to the gaussian (the crystal build clobbers it)
but `bounding_energy` reads `raw_latents`, so its marginal is
`exp(−k·relu(|x|−1)²)`, whose normaliser is `2` (the flat box) plus
`2∫₀^∞exp(−ku²)du = √(π/k)` of leakage past the soft wall. At k=1 the leak is
nearly as large as the box itself: 3.77 vs 2.

Measured, IS with an analytic proposal against the real `MolecularCrystal` object:

| sg | dead | n_live | HELD | analytic | LIVE | analytic |
|---|---|---|---|---|---|---|
| 2 | () | 12 | −16.6038 | −16.6038 | −16.6038 | −16.6038 |
| 14 | (3,5) | 10 | −13.8365 | −13.8365 | −11.1812 | −11.1810 |
| 19 | (3,4,5) | 9 | −12.4528 | −12.4528 | −8.4644 | −8.4696 |
| 4 | (3,5,7) | 9 | −12.4528 | −12.4528 | −8.4686 | −8.4696 |
| 1 | (6,7,8) | 9 | −12.4528 | −12.4528 | −8.4653 | −8.4696 |

HELD arms carry `Var(log w)` 4e-12 to 6e-12 — the proposal *is* the target, which
is only possible if the dead rows contribute exactly zero and neither reduction
nor jacobian is present. LIVE arms agree to ≤0.005 nats at `Var(log w)` ≈ 0.3.

**`bounding_coeff` is a second, independent dial**, reachable by no space group:

| k | measured | soft-wall | `n_dead·log 2` model | model error |
|---|---|---|---|---|
| 0.5 | −7.9432 | −7.9362 | −10.3734 | **+2.4302** |
| 2.0 | −8.9047 | −8.9138 | −10.3734 | **+1.4687** |
| 10.0 | −9.6226 | −9.6322 | −10.3734 | **+0.7508** |

**Refutes the `n_dead·log 2` prediction** made earlier in the same session, which
is wrong by +0.63/row at k=1 and diverges further as the wall softens. The
soft-wall form holds to ≤0.01 across a 20× range of k.

**sg 4 and sg 1 land on sg 19's number by a different code path.** All three have
3 dead rows, but sg 19's are clobbered by `enforce_crystal_system` inside
`latent_to_cell_params`, sg 1's by `canonicalize_free_axes` inside
`latent_params()`, and sg 4 mixes the two. Agreement to ≤0.005 nats is the first
evidence the free-axis half of D33 is right; before this it was unit-tested only
(deadrow_aug12 cannot reach it — a physical prior must be a real crystal).

Mechanism checked directly, not merely inferred: perturbing only the dead rows
moves the gaussian term by **exactly 0.00e+00** and the total by exactly
`k·Σrelu(|x|−1)²` (2.180000 measured vs 2.180000 predicted, 4 space groups).

*Feeds `decisions.md` D33 and `configs/gauss_aug12/`.*

---

## F-022 · A progress-seeking controller is the first thing that works on the 3-player surface, and no single pair wins everywhere · `REPLICATED`

*2026-08-12. `bench/scenarios.py::toolkit`, 9 pairs × 3 scenarios × 3 surfaces ×
**3 seeds**, worst-case regret. Adds two candidates that do NOT exist in
train.py (`LR_SENSOR_KINDS` is `ray`/`plateau`/`none`):*

- **`slope`** — brake on the observed rate of improvement rather than best-ever.
  No ratchet (one lucky sample sets `plateau`'s best and nothing beats it after),
  and it reacts in 2 windows rather than 30 checks.
- **`slope_seek`** — a whole controller: perturb the rate, keep the direction
  that improved the progress rate, flip when it did not. A derivative-free
  hill-climb on log-LR whose objective is *progress per step at the rate being
  run* — not a one-step surrogate. Built to have F-021's stability property: one
  measurement, one clock, both directions.

**`slope_seek` is the first arm to work on `equilibration`** — the surface where
every previous arm was bad and none recovered:

| pair | cold start | blow-up | hot | worst |
|---|---|---|---|---|
| `ray`+`ray` *(shipping)* | 6.37 | 6.01 | 7.22 | 7.22 |
| `slope_seek`+`ray` | **0.71** | **0.71** | 6.23 | 6.23 |
| `slope_seek`+`plateau` | **0.65** | 1.20 | 5.87 | 5.87 |
| `slope_seek`+`none` | 1.57 | 1.08 | 5.43 | **5.43** |

It beats the shipping pair on **every** scenario there, and regret below 1.0 means
it beat the best fixed rate outright.

**CONFOUND, found and controlled after the fact — the "~9×" headline was
inflated.** On `equilibration` the probe scores the **replay term only**
(`probe_scores='replay'`, reproducing defect #1: it rates a loss nobody is wholly
optimising), while `slope_seek` reads the loss `train_step` returns, which is the
**θ-total** `w_rep·replay + w_bwd·bwd`. The two arms were therefore watching
different objectives, and part of the gap was observation, not algorithm.
Re-run with both seeing the θ-total, 3 seeds:

| arm | probe scores | cold start | blow-up |
|---|---|---|---|
| `ray`+`ray` | replay only *(shipping)* | 6.37 | 6.01 |
| `ray`+`ray` | θ-total | **2.73** | **3.02** |
| `slope_seek`+`ray` | θ-total | 0.71 | 1.13 |
| `slope_seek`+`plateau` | θ-total | 0.65 | 1.20 |

So the objective mismatch costs **~2.2×** on its own, and `slope_seek`'s genuine
advantage is **~3.8× (cold) / ~2.7× (blow-up)**, not 9×.

**That reorders the recommendation: fix what the probe scores BEFORE adding a
controller.** It is a change to an existing sensor rather than a new kind, and it
recovers roughly half the available gain. This also puts a number on defect #1 of
[[project-ray-probe-sensor-defects]], which was previously argued but never
costed.

> ### 🔴 RETRACTED, same day: the flow/Z half of this entry
>
> The "+ flow/Z" result below is **an artifact of a parameter I chose** and does
> not hold. The game gave the level's objective a sensitivity to θ of `a = 4`
> while the replay branch's was `b = 1` — a 4× asymmetry with no justification.
> In the real system both branches are TB residuals over the same `log P_F`
> (the residual *is* `log Z − log w`), so there is no reason for their
> θ-sensitivities to differ. Sweeping the ratio:
>
> | a/b | regret + bwd | regret + flow | benefit of scoring flow |
> |---|---|---|---|
> | **1.0** | 1.26 | 2.02 | **0.63× — actively harmful** |
> | 2.0 | 0.93 | 0.92 | 1.01× — no effect |
> | 4.0 | 2.73 | 0.75 | 3.63× |
> | 8.0 | 9.54 | 0.73 | 13.1× |
>
> At the honest setting `a = b` scoring the flow term is **1.6× worse** than not.
> The benefit exists only above ~4× asymmetry.
>
> Two further reasons it was never as advertised, both found before the sweep:
> **(i)** ζ's residual is identical in every arm (0.027) and the policy LR cannot
> move it, so the "protect Z" story was empty — the arms differ 15× in ‖θ‖ and 0%
> in ‖ζ‖; **(ii)** under a whole-system metric `‖(θ,ζ,μ)‖` all arms score 0.93–0.97,
> i.e. indistinguishable, because ζ dominates the norm and is LR-independent.
>
> **What survives, re-measured across the same sweep:** scoring the loss that
> actually *trains the parameters being perturbed* (replay + bwd) is positive at
> **every** ratio, but modest at the honest one:
>
> | a/b | replay only | + bwd | benefit |
> |---|---|---|---|
> | **1.0** | 1.58 | 1.26 | **1.25×** |
> | 2.0 | 1.49 | 0.93 | 1.60× |
> | 4.0 | 6.37 | 2.73 | 2.34× |
>
> Direction robust, magnitude not. **What does not survive:** scoring a loss that
> trains *other* parameters. Do not build the flow-term change.
>
> ### 🔴 Wider consequence: `a/b` also set the surface's difficulty
>
> The `replay only` column above is **1.58 at a/b = 1 and 6.37 at a/b = 4**. So
> "`equilibration` is where the controller performs worst" — F-019's headline,
> F-021's 7.22 worst case, and the premise for `slope_seek` in this entry — were
> all measured at a/b = 4 and shrink by ~4× at a/b = 1. At the honest setting the
> shipping controller is *fine* on this surface.
>
> **`a/b` is now the single most load-bearing unknown in the bench**, and unlike
> most of them it is empirically determinable on the real system: compare the
> θ-sensitivity of the fwd/Z branch's TB residual with the replay branch's, on one
> stored batch. Until that number exists, every `equilibration` magnitude in
> F-019/F-021/F-022 should be read as conditional on a/b = 4, which nothing
> justifies.

**WHAT THE PROBE SCORES IS WORTH MORE THAN WHICH CONTROLLER RUNS IT.** Three-point
comparison on `equilibration`, `ray`+`ray` throughout, 3 seeds:

| probe scores | cold start | blow-up | settled lr / oracle |
|---|---|---|---|
| replay only *(shipping)* | 6.37 | 6.01 | 42.9 |
| + bwd term (the policy total) | 2.73 | 3.02 | 10.7 |
| **+ flow/Z term (whole system)** | **0.75** | **1.54** | 0.06 |

Widening from replay to the policy total is worth **2.3×**; adding the level's own
objective is worth a further **3.6×** — **8.5× end to end**, and below 1.0 means
beating the best fixed rate.

**This largely dissolves the case for a new controller.** `slope_seek`+`ray` scored
0.71 / 1.54 on the same two scenarios; `ray`+`ray` scoring the whole system gets
0.75 / 1.54. The probe was not the wrong mechanism — it was pointed at the wrong
loss. The remaining difference is inside the range this bench can resolve.

**Why the flow term specifically.** `ζ` is frozen during a probe, but the level's
objective `½‖ζ + aθ‖²` still *moves when θ moves*. It is therefore the only term
in which the probe can see the cost a bigger θ-step imposes on the other player —
i.e. the loop that F-012 showed sets the stability boundary. Every other term it
scores is about θ alone, which is exactly why it parked 43× above the oracle.

**Mapping to `train.py`.** `_draw_probe_batch` returns
`(draw_replay_sample(k), self.args.replay_loss_coeffs, 'replay', k)` — replay
draws, replay coefficients. The change is a mixture plus a coefficient swap, and
the backward-draw path already exists as a fallback for when replay is empty.
Two things to get right: mix at the **training proportions** (the branch fracs the
balance controller moves), or the probe scores a third objective nobody optimises;
and the fwd/Z branch normally needs on-policy rollouts, which are not stored —
whether re-scoring stored replay trajectories under fwd coefficients is a faithful
substitute is **not** something this bench can settle.

**Probe scope, for the record.** On `equilibration` the probe perturbs `θ` only
(`policy_params = [theta]`, decision D26b), with `ζ` and `μ` frozen throughout,
while the step it rates moved `θ` by the combined replay+bwd gradient *and* moved
`ζ` by its own objective. The frozen-`ζ` assumption is exactly the one F-012's
one-step-vs-loop-stability gap rests on.

**But it is fragile on the simple surfaces, and no pair wins everywhere:**

| pair | mle | var_cond | equilibration | global worst | silent |
|---|---|---|---|---|---|
| `ray`+`ray` | **1.23** | **3.28** | 7.22 | **7.22** | 2 |
| `slope_seek`+`plateau` | 1.85 | 8.04 | 5.87 | 8.04 | 2 |
| `ramp`+`ray` | 1.40 | 2.27 | 36.3 | 36.3 | **0** |
| `ramp`+`slope` | 1.20 | 41.8 | 71.4 | 71.4 | 0 |
| `slope_seek`+`ray` | 347 | 8.67 | 6.23 | 347 | 1 |
| `ray`+`slope` | 4.55 | 2.77 | **232584** | 232584 | 1 |
| `slope_seek`+`slope` | 8691 | 8.68 | 2811 | 8691 | 5 |

`ray`+`ray` and `slope_seek`+`plateau` are a **dead heat globally** (7.22 vs
8.04) with **inverted profiles** — the shipping pair is better on the two easy
surfaces, the seeker is better on the hard one.

**F-021's mechanism is confirmed on entirely new arms: every catastrophic entry
is a MIXED-CRITERION pair.** `ray`+`slope` (232584), `slope_seek`+`slope` (8691),
`slope_seek`+`ray` (347). Two sensors that disagree about what "too fast" means
fight each other; the safe pairs are same-criterion (`ray`+`ray`,
`slope_seek`+`none`) or a deliberately slow climber with any brake (`ramp`+`*`).

**Design reading — the elegant combination is two mechanisms, chosen per stage,
never mixed:**

1. Keep `ray`+`ray` as the default. It has the best global worst case and is
   strongest on regression-like and cooperative stages.
2. Add `slope_seek` as a sensor kind and select it on the fused 3-player stage,
   where it is ~9× better and where the current controller is worst.
3. Never pair two different criteria. That rule alone removes the three worst
   entries in the table.

`lr_sensor` already expresses per-stage selection, so (2) is a new kind plus a
config line, not a rewrite.

**Scope.** Bench surfaces, 3 seeds. `slope_seek`'s window, step factor and dead
zone were set once by argument and never tuned — the numbers would move under
tuning, the ordering on `equilibration` is what to take. It has not been run
against a real trajectory-based loss.

---

## F-021 · Splitting the probe between climbing and braking is much WORSE than using it for both; the shipping pair has the best worst case in the whole factorial · `REPLICATED`

*2026-08-12. `bench/scenarios.py::toolkit`. Full 3 climbers × 3 brakers × 3
scenarios × 3 surfaces × **3 seeds**. Scored on WORST-CASE regret, because a
controller is chosen for the run it will not ruin. Supersedes F-020's design
suggestion.*

F-020 observed that on `equilibration` the probe climbed best while `plateau`
braked best, and inferred that `ray`-to-climb + `plateau`-to-brake was worth
adding. **Running it says the opposite.**

**Worst-case regret anywhere, all nine pairs:**

| climber | braker | worst | silent failures |
|---|---|---|---|
| **ray** | **ray** | **7.22** | 2 |
| ramp | ray | 36.3 | 0 |
| ramp | plateau | 79.8 | 0 |
| ramp | none | 81.6 | 0 |
| none | ray | 8691 | 1 |
| none | plateau | 8691 | 5 |
| none | none | 8691 | 6 |
| ray | none | **89713** | 1 |
| ray | plateau | **113116** | 1 |

**The split arms are the two worst entries in the table.** `ray`+`plateau`
reaches 113116× on `equilibration`'s blow-up — five orders of magnitude worse
than `ray`+`ray` on the same scenario (6.01).

**Mechanism: the brake must run on the climber's clock and criterion.** The probe
raises whenever it reads "too cold", every calibration period. `plateau` needs
`patience` × `check_every` = 300 steps of no improvement before it cuts once by
half. Pair them and the climber outruns the brake — after a blow-up the rate
keeps rising while the brake is still counting. Neither component is at fault;
the *mismatch* is. `ray`+`ray` is stable precisely because the same measurement,
on the same clock, moves the rate in both directions, so it cannot outrun itself.

`ramp`+`ray` survives for the same reason in reverse: `ramp` is a slow, bounded
climber, so the probe's brake keeps up.

**So the shipping configuration is the right one**, and by a wide margin on worst
case — 7.22 against 36.3 for the next best, and it is the best arm on
`equilibration` (7.22 vs 36.3) and within 2× of the best on the other two.

**Its one real weakness is detection, not control.** It is the only good pair
that fails silently: 2 of its 9 surface-scenarios exceed 3× regret with nothing
flagged (`var_cond` cold start 3.28, `equilibration` hot 7.22). Both are moderate
misses, both invisible.

**Reading for the design question.** The elegant answer is not another tool. It
is: keep `ray` for both roles, do not split it, and add a *signal* rather than a
mechanism — one that asks whether progress is commensurate with the rate being
run, which is the one thing none of the nine pairs can currently tell you.

*Method note: `toolkit`'s printed "most robust" line sorts on silent-failure
count before worst-case regret, which flatters `ramp`+`ray`. The table above is
sorted on worst case, which is the criterion the finding uses.*

---

## F-020 · Climbing and braking are separate jobs; the probe is the only thing that climbs on the 3-player game, and a blind ramp beats it on the 1-player one · `REPLICATED`

*2026-08-12. `bench/scenarios.py::sensor_race`. Five sensor arms × 3 scenarios ×
3 surfaces × **3 seeds**. Every arm drives the SAME actuator — same warmup,
bounds, ceiling, tripwire, rewind — so only the verdict source differs.*

Arms: `ray` (alpha probe), `plateau` (train.py's ReduceLROnPlateau, transcribed
from train.py:4030 including its EMA input and absolute bar), `ramp` (no sensor,
a constant raise per period), `ramp_plateau`, `none`.

`ramp` is not a strawman: 72–82% of real probe readings come back at a grid edge,
and there the servo applies exactly this constant.

**Regret vs each surface's own oracle (lower is better):**

| surface | scenario | ray | plateau | ramp | ramp_plateau | none |
|---|---|---|---|---|---|---|
| `mle` | cold start | 1.23 | 8691 | 5.92 | **0.69** | 8691 |
| | blow-up | 0.90 | 399 | 0.07 | **0.07** | 399 |
| | hot | 1.08 | – | – | **0.11** | – |
| `var_cond` | cold start | 3.28 | 8.65 | **2.41** | **2.41** | 8.65 |
| | blow-up | **1.01** | 6.91 | 36.3 | 36.3 | 15.8 |
| | hot | **0.61** | – | – | – | – |
| `equilibration` | cold start | **6.37** | 3605 | 27.7 | 19.7 | 1499 |
| | blow-up | 6.01 | **3.68** | 81.6 | 79.8 | 14.1 |
| | hot | 7.22 | **2.02** | – | – | – |

**1. A loss-watcher cannot climb, and on a cold start that makes it worthless.**
`plateau` scores **bit-identically to `none`** on `mle` cold start (8690.54 both)
— from a seed 35× too low the loss is improving, so it never fires. Same on
`equilibration` (3605 vs 1499, both catastrophic). It only ever cuts, and cold
start is the state every stage transition creates.

**2. On the single-player surface the probe is not paying for itself.**
`ramp_plateau` beats `ray` on all three scenarios (0.69 vs 1.23, 0.07 vs 0.90,
0.11 vs 1.08). Climb blindly, brake on the loss. The probe's information is not
worth its cost there — which is consistent with it spending most of its time
saturated and applying that same constant anyway.

**3. On the 3-player game the probe is the only thing that climbs — and that
justifies the design choice.** `ray` 6.37 against `ramp` 27.7 and `plateau` 3605.
Nothing else gets near. This is the case the probe was deliberately deployed on,
and the reasoning holds up.

**4. But being least-bad is not being good.** On `equilibration` the best arm in
the entire matrix is 6.37× worse than the best fixed rate, `ray` parks 43× above
the ideal rate, and **no arm recovers within the run**. The surface where the
probe is most needed is the one where the whole controller performs worst.

**5. A blind ramp is dangerous exactly where the brake is weak.** `ramp` on
`var_cond`'s blow-up reaches 250× the oracle with 4 divergences (regret 36) —
worse than doing nothing (15.8). "Climb blind, brake on evidence" is only safe
when the evidence arrives in time.

**Design reading.** Climbing and braking want different mechanisms, and the
right pairing is per stage — which is what `lr_sensor` already expresses, so the
change is a configuration one, not a rewrite. The probe's climb is load-bearing
on the fused 3-player stage; its brake is beaten by `plateau` on both braking
scenarios there (3.68 vs 6.01, 2.02 vs 7.22). A stage running `ray` to climb and
a plateau rule to brake is not currently expressible and looks worth adding.

**Scope.** Bench surfaces, not crystals; `plateau` here watches a single
training loss, whereas the real one tracks a configured metric list. The ordering
between arms is what to take, not the magnitudes.

---

## F-019 · The direction of the servo's error is surface-dependent, and where the stability ceiling sits far above the optimum it parks near the ceiling · `REPLICATED`

*2026-08-12. `bench/scenarios.py` full board, 3 surfaces × 6 scenarios × **3 seeds**,
each scored against its own oracle. Post-rewind (see F-018's method correction).*

| surface | oracle lr | cold_start regret | settled lr / oracle | blowup recovery |
|---|---|---|---|---|
| `mle` (anisotropic, cliff) | 4.33e-3 | **1.23** | 0.33 | 123 steps |
| `var_cond` (bowl, stale levels) | 0.152 | **3.28** | 0.13 | 556 steps |
| `equilibration` (shallow, anti-phase loop) | 1.37e-3 | **6.37** | **42.9** | never |

**The sign of the error flips between surfaces.** On `mle` and `var_cond` the
servo settles *below* the oracle (0.33×, 0.13×); on `equilibration` it settles
**42.9× above** it. No single `alpha_target` can be right for all three, which is
F-011/F-012's per-route claim arriving from a completely different direction —
regret rather than mechanism.

**Where the stability ceiling is far above the optimum, the servo parks near the
ceiling.** `equilibration` is stable up to lr ≈ 0.36 but trains best at 1.37e-3 —
a 260× gap. The servo settles at 0.059: comfortably stable, and 43× too fast. A
one-step ray criterion answers "what rate does not blow up", and on a shallow
surface that is a very weak constraint. It never recovers within the run.

**Starting hot is not symmetric with starting cold.** On `var_cond` the hot arms
beat the oracle (regret 0.70, 0.61) while cold start never recovers; on
`equilibration` the *cold* arm is the best of the six (regret 0.72). The servo
descends onto a good rate far more reliably than it climbs to one — which is
`eta_up` 0.25 against `eta_down` 0.5 doing exactly what it was designed to do,
with a cost that had not been measured.

**Two silent failures.** `var_cond` cold_start (regret 3.28, never recovered) and
`equilibration` hot_90pct_to_cliff (regret 7.22) raise **no flag at all** — no
divergence, sensor not pinned, `peak_scale` off its bounds. Both look exactly
like slow training. That is the detectability gap the third scoring axis exists
to find, and it is the argument for adding an oracle-free health signal rather
than another LR heuristic.

---

## F-018 · Against an oracle LR the servo is nearly free at the shipping target; the danger is `alpha_target` too HIGH, and `divergence_cut` is the consequential knob · `MECHANISM` + `REPLICATED`

*2026-08-12. `bench/`, `mle` surface (dim 32, cond 300, SGD), cold start from
mk_dev's `auto` seed 1.25e-4, 2000 steps, **3 seeds**, median reported.*

> **Method correction, same day.** The first version of this entry reported
> regret 124–372 at `alpha_target` ≤ 4 and concluded the shipping default sat in
> a pathological regime. Those numbers were an artifact: `BenchRun` modelled the
> divergence response as a **peak cut alone**, when `train.py`'s `fire_loss_spike`
> does a **rewind to the running checkpoint and then** the cut
> (`load_model_only`, train.py:2023). On a surface where a blow-up drives
> parameters non-finite, the omitted half is the half that recovers — `check_spike`
> then fires forever and the run can never come back. The harness now
> checkpoints on train.py's 50-step clock, rewinds on divergence, and aborts past
> `max_reloads_per_1k_steps`. Every number below is post-fix. The conclusion
> reverses: at low `alpha_target` the controller is fine.

`bench/oracle.py` now brute-forces the best FIXED learning rate a surface admits,
which turns every controller result into a regret ratio and gives "recovered to
healthy" a definition (within `tol` of the oracle's own trace, per step). The
sweep is checked, not trusted: `find_oracle` refuses a bracket whose minimum sits
at an edge, and refuses a surface where the best rate does not beat the edges.

**The oracle sits next to the cliff.** On `mle`, best fixed lr **4.329e-3** with
final distance 6.777e-4 — just under the SGD stability limit `2/λ_max` = 6.67e-3.
One ladder rung above it, the run diverges. So the best available rate is
immediately adjacent to the boundary, which is what makes this control problem
hard rather than merely fiddly.

**The servo is nearly free at the shipping target, and the failure is one-sided:**

| `alpha_target` | lr/oracle | regret | divergences | regime |
|---|---|---|---|---|
| 1 | 1.10 | 1.2 | 1 | bar fires once, recovers |
| 2 | 0.65 | **0.6** | 1 | bar fires once, recovers |
| **4 (shipping)** | 0.33 | **1.2** | 1 | bar fires once, recovers |
| 6 | 0.50 | 2.0 | 1 | bar fires once, recovers |
| 8 | 0.60 | 2.7 | 0 | tracking |
| 12 | 0.98 | 7.5 | 0 | tracking |
| 16 | 1.10 | 25.9 | 0 | tracking |
| 24 | 0.48 | **1765** | 0 | stranded cold |
| 32 | 0.03 | **8691** | 0 | stranded cold |
| 64 | 0.00 | **13027** | 0 | stranded cold |

At `alpha_target` ≤ 8 the controller lands within **0.6–2.7×** of the best fixed
rate — regret 0.6 means it *beat* the best constant, which a schedule legitimately
can. This is the first direct evidence the servo is worth having.

**The ramp-to-the-bar is not pathological, it is edge-finding.** At low targets
the servo climbs until it crosses the stability limit, trips the bar **once**,
rewinds to the last healthy checkpoint, cuts the peak, and settles near the
oracle. One divergence, recovered. The mechanism only looked broken when the
harness modelled the cut without the rewind.

**The dangerous direction is `alpha_target` too HIGH.** Past ~24 nearly every
reading falls below target, the asymmetric update cuts on almost every
calibration, and the rate walks to the floor — regret 1765–13027, and **zero
divergences**, so the loudest signal the system has stays silent. Same absorbing
cold state F-016 arm B reached by a different route.

**`divergence_cut` is the consequential knob, not `alpha_target`.** Where the bar
fires, halving it costs 18–59× in final distance; a 4× change in `alpha_target`
costs ~4%:

| `alpha_target` | `divergence_cut` | settled lr | regret | div |
|---|---|---|---|---|
| 1 | 0.50 | 4.76e-3 | 1.18 | 1 |
| 1 | 0.25 | 2.38e-3 | **69.5** | 1 |
| 4 | 0.50 | 1.41e-3 | 1.23 | 1 |
| 4 | 0.25 | 2.83e-3 | **22.3** | 1 |
| 16 | 0.50 | 4.76e-3 | 25.9 | 0 |
| 16 | 0.25 | 4.76e-3 | 25.9 | 0 |

At target 16 the bar never fires and `divergence_cut` becomes irrelevant —
identical to three significant figures. So when the bar is part of the loop, the
recovery cut sets the operating point, and the knob that gets tuned is not the
one in charge.

**What transfers and what does not.** The window's *location* is per-surface,
set by the anisotropy margin (F-011) and the loop gain (F-012) — 8 is a fact
about `cond=300`, not a recommendation. The regret *magnitudes* are inflated by
this surface's cliff: near the stability boundary final distance is very steep in
LR, so 3× in rate costs 100×+ in distance, and a flatter problem would show the
same shape with smaller numbers. What transfers is the **shape** — a U with an
absorbing state on each side — and the indicator of which side you are on.

**Usable diagnostic, already logged — and it is the opposite of what the
uncorrected version said.** A *few* divergences followed by recovery is the
controller finding the edge and is not by itself a problem; what to alert on is
`lr_ctrl/peak_scale` walking toward its floor **with no divergences**, which is
the absorbing cold state and is otherwise indistinguishable from slow progress.
`raycal/status` pinned at a bound supports the same read. Both are already in
wandb.

**Recovery from an injected blow-up works.** Multiplying `peak_scale` by 100
mid-run: `mle` recovers in ~124 steps to regret 0.79–0.84 with one reload;
`equilibration` recovers on 1 of 2 seeds (step 2004 of 3000) with three. So the
answer to "if it never recovers we have a problem" is: it does recover, on these
surfaces, *provided the rewind target exists* — and `train.py`'s NO REWIND TARGET
branch is reachable in a real run, since `best` is only written once an eval has
improved.

---

## F-017 · The batch knee test is a local comparison, so a step in the cost curve pins it permanently and its answer depends on where it started · `MECHANISM`

*2026-08-12. `bench/`, `SyntheticGPU` with discreteness enabled. `t_fixed` 2.0 s,
`sps_max` 5000/s, `f` 1.65, `tol` 0.25. Supersedes F-013's smooth-model scope.*

Real step time is not smooth in batch, so the clock now models three effects,
all opt-in: wave quantisation (`tile`), `torch.compile`'s per-shape recompile
(`recompile_s`), and kernel-regime switches (`regimes`).

**Tracking is fine.** Under mild discreteness the controller lands exactly where
a walk against the actual cost model says it should, 6/6 seeds:

| cost model | predicted | pins |
|---|---|---|
| smooth | 7410 | 7410 ×6 |
| wave quantisation, tile 256 | 7410 | 7410 ×6 |
| + recompile stall 40 s | 7410 | 7410 ×6 |
| + kernel switch @4096 ×0.8 | 7410 | 7410 ×6 |
| + jitter 0.10 | 7410 | 7410 ×4, 4491 ×2 |

The recompile stall notably does **not** corrupt the decision at the shipping
growth interval: it is charged on the first step at a new size, and the gate
medians the last 20 timings 50 steps later, by which point it has aged out.

**The criterion is what fails.** The gate accepts a jump iff
`t(f·B)/t(B) ≤ 1+tol` — a purely local two-point comparison. A one-off step in
the cost curve between two rungs is therefore indistinguishable from saturation.
With a kernel-switch efficiency drop at 2722 the walk trips there and pins at
**1650**, where throughput is 708 samples/s against 1493 available at 7410 —
**47% of what the hardware could do**, permanently.

**The recheck cannot escape it.** `batch_knee_recheck_steps` drops one rung and
re-climbs, so it re-tests the same failing comparison and re-pins in the same
place; 20 000 steps with rechecks every 1500 left it at 1650. The recheck adapts
to a knee that *moved*. It cannot discover that the knee was never there.

**Hence the gate is path-dependent.** Same cost model, same config, only the
starting batch differs:

| start | pin | samples/s | % of best on the ladder |
|---|---|---|---|
| 1000 | 1650 | 708 | 40% |
| 1650 | 1650 | 708 | 40% |
| 2722 | 4491 | 1183 | 67% |
| 4491 | 4491 | 1183 | 67% |
| 7410 | 7410 | 1493 | 84% |

On the smooth model the same sweep converges to 7410 from every start, so this
is a property of the *cost curve's* monotonicity, not of the walk. On a
non-monotone curve, `batch_size` in the config does not merely floor the walk —
**it selects the answer.**

Not established: whether this route's real `t(B)` is non-monotone enough to
matter. The regime drops used here (×0.5, ×0.6) are constructions, not
measurements. What the bench settles is that the criterion has no defence if it
is. The obvious mitigation — have the recheck occasionally probe a rung *above*
the pin rather than only below — is untested.

---

## F-016 · Bounding the cumulative open-loop ramp removes every divergence AND trains better · `REPLICATED`

*2026-08-12. `bench/`, `mle` game, `dim` 32, `cond` 300, SGD, seed LR 5e-5,
calibration period 50, 3000 steps, **5 seeds**. Candidate fix for F-011.*

F-011 leaves the controller with no good option: raising on a saturated reading
runs to the bar, and refusing to raise strands a cold run at its seed. A third
policy works — allow saturated raises, but bound the **cumulative** open-loop
excursion since the last reading that actually landed inside the grid. A
`bracketed` or `below_range` reading is real feedback and resets the budget.

| arm | crossed `lr·λ_max` = 2 | divergences | median final dist |
|---|---|---|---|
| A shipping (saturated raises freely) | **5/5** | **10** | 4.6e-3 |
| B saturated may not raise | 0/5 | 0 | 5.04 *(no progress)* |
| D bounded, cap 8× | 0/5 | 0 | 0.59 *(undertrained)* |
| **E bounded, cap 64×** | **0/5** | **0** | **6.7e-4** |

Arm E is strictly better than shipping on all three axes, and by a margin far
larger than the seed spread (arm E peak `lr·λ_max` ranged 1.48–1.92; arm A was
identically 2.72 every seed, because an open-loop ramp is deterministic — seeds
differ only in the noise realisation).

**The transferable claim is the principle, not the number.** A cap of 64× is a
fact about this surface's seed-to-optimum ratio; the generalisable part is that
an unresolved sensor should be allowed a *bounded* speculative excursion rather
than an unbounded one, because the bound is what converts "the sensor cannot see
the constraint" from an unlimited licence into a finite one. Cap too tight (8×)
and it becomes arm B.

**Implementation** is a few lines in `on_calibration`: carry a
`saturated_budget` on `lr_ctrl`, multiply it by the applied ratio when
`status == 'above_range'`, reset it to 1.0 on any other resolved status, and
skip the update when the next raise would take it past the cap. It needs a
cluster arm before shipping — this is a bench result on one synthetic family.

---

## F-015 · A converged run's servo LR is uninformative · `MECHANISM`

*2026-08-12. `bench/`, `mle` and `equilibration` games, CPU.*

Once a run reaches its noise floor the gradient is pure noise, so the step
direction `d` is uncorrelated with the population gradient and
`alpha* = −(g·d)/(dᵀHd)` becomes a ratio of noise to noise. It goes **negative**
— measured directly against ground truth on `mle` at `dist ≈ 9e-5`:
`alpha_true` read −0.021, −0.104, −0.444, −0.603, −1.014 across successive
calibrations while the sensor reported `below_range` and the servo cut on every
one.

Consequence for reading logs: **`lr_ctrl/peak_scale` late in a converged run
carries no information about the right rate.** On `equilibration`, targets 1.0
and 8.0 ended within a factor of 2 of each other despite an 8× difference in
what they were asking for. What `alpha_target` actually controls is how far the
rate climbs *while there is still signal*.

The bench's own experiments therefore report the **maximum** excursion, not the
endpoint. The same caveat applies to any cluster run being read for "where did
the servo settle".

---

## F-014 · Three inert or mis-documented knobs in the shipping LR/batch surface · `MECHANISM`

*2026-08-12. Verified against `controller.py`, `train.py` and `configs/mk_dev.yaml`
by `bench/test_lr_controller.py` and `bench/test_batch_sizer.py`.*

**`control_flow_lr: true` grants the envelope, not `peak_scale`.** mk_dev's
comment says "the flow/Z groups also take envelope and peak_scale". They take
the envelope. `_apply_lrs` gates peak on `base_key in managed`, and
`lr_servo_managed` holds only keys written `auto`; `lr_flow` is an explicit
float (0.1) in every shipping config, so `peak_scale` can never reach it. The
knob is a warmup-envelope switch unless `lr_flow: auto` — which also changes
what it starts from.

**`warmup_steps: 0` does not disable warmup.** `_envelope` takes
`max(1, warmup_steps)` and `elapsed` is 0 on the evaluation that CREATES the
state, so the first tick still comes out at `1/lr_warmup_ratio`. One step at a
tenth of the intended rate.

**Warmup is measured from state creation, not from step 0.** `_fresh_state`
stamps `stage_start_step` with the current `step_ind`. A v7 state discarded at
step 8000 therefore buys a **full 1000-step warmup re-ramp from 8000**, not an
immediate release. Nothing in the logs distinguishes "held" from "live" except
`lr_ctrl/warmup`.

**`max_step_seconds` cannot cut a fused stage at the base batch.** mk_dev sets
`batch_size: 1000` and `fused_grad_accum_min_samples: 1000`; the ceiling refuses
to cut below the accumulation floor (correctly — below it a fused step is a
micro-step and time per *optimizer update* does not fall). So the 10 s ceiling
has authority only over batch sizes the growth walk added on top. Confirmed both
ways: with `fused_grad_accum_min_samples: 0` the same run cuts 1000 → 606.

---

## F-013 · The batch growth gate pins one rung above its own bound, and tolerates ~20% step-time jitter · `MECHANISM` + `REPLICATED`

*2026-08-12. `bench/`, `SyntheticGPU` `t(B) = t_fixed + B/sps_max`, `t_fixed` 2.0 s,
`sps_max` 5000/s, `f` 1.65, `tol` 0.25. 10 clock seeds per jitter level.*

**The gate's acceptance bound is closed-form.** Accept iff
`t(fB)/t(B) ≤ 1+tol`, so

```
B_max = sps_max · t_fixed · tol / (f − 1 − tol) = 6250
```

**The pin is systematically one growth factor hot** — `MECHANISM`. The gate
scores the jump *from* a rung, so a rung is only convicted after the controller
has moved past it, and the `prev_batch` it falls back to is itself the first
rung above the bound. Measured pin 7410 against a bound of 6250; the ladder's
`int(round(·))` per rung puts it at 7410 rather than the ideal 7412.

**The decisive comparison is marginal.** At that rung the true step-time ratio is
**1.2766 against a 1.25 threshold — a 2.1% margin**, decided from 20 timings.
The ladder (1.65×) is coarse next to the tolerance band (1.25×), so the boundary
decision is intrinsically delicate even though the pin is not.

**Jitter tolerance** — `REPLICATED`, 10 clock seeds × 5 levels:

| lognormal σ | pins |
|---|---|
| 0.00 | 7410 ×10 |
| 0.05 | 7410 ×9, 12226 ×1 |
| 0.10 | 7410 ×7, 12226 ×2, 4491 ×1 |
| 0.20 | 7410 ×8, 12226 ×2 |
| 0.40 | scattered 1000–12226, **including the floor** |

Modal answer correct through σ = 0.20, worst case one rung either way. At σ =
0.40 the walk can collapse to the configured floor — the flat-throughput
pathology triggered by *noise* rather than by flat throughput. Scope: smooth
step-time model; real accelerators have steps in `t(B)` that this does not have,
so this bounds the best case.

*(Method note: an earlier version of this measurement reported a systematic
one-rung overshoot. `SyntheticGPU` has its own RNG and was not receiving the
seed, so ten "replicates" replayed one identical timing stream. The table above
is after that fix.)*

---

## F-012 · The minimum safe `alpha_target` is set by the policy/level loop gain, which the sensor cannot see · `MECHANISM`

*2026-08-12. `bench/` `equilibration` game — three coupled players with an exact
linear iteration matrix. Derived, then verified numerically over 12 (a, w_rep)
combinations by `bench/test_games.py`.*

For a policy chasing a level, a level responding anti-phase, and a buffer, the
policy's self-curvature is `c = w_rep·b² + w_bwd` and the pair's eigenvalues are
`−c ± i·√(w_rep·b·a)`. So:

- a frozen-target ray probe reads `alpha* = 1` at `lr = 1/c`;
- the **loop** survives only to `lr = 2c/(c² + a·b·w_rep)`.

Their ratio is the smallest target that keeps the servo inside the boundary:

```
alpha_target_min = (c² + a·b·w_rep) / (2c²)   →   (1 + loop_gain)/2  at c = 1
```

Verified to 2% across `a ∈ {1,2,4,8} × w_rep ∈ {0.3,0.5,0.7}`.

**Closed-loop confirmation.** Running the real servo at `a=4, w_rep=0.7`
(derived minimum **2.79**) and recording the maximum excursion each target
reached, as a fraction of the stability boundary:

| `alpha_target` | max `lr` | max `lr`/stability | |
|---|---|---|---|
| 1.0 | 0.617 | **1.72** | outside |
| 1.5 | 0.581 | **1.62** | outside |
| 2.0 | 0.436 | **1.22** | outside |
| 3.0 | 0.221 | 0.62 | inside |
| 4.0 | 0.168 | 0.47 | inside |
| 8.0 | 0.077 | 0.22 | inside |

The measured crossover sits between 2.0 and 3.0, bracketing the derived 2.79.
The closed form and the servo agree, and the shipping `alpha_target: 4.0` is
inside the boundary for this loop gain while `1.0` is not — which is the
quantitative version of the argument the config comment makes from one cluster
observation.

**Loop gain rises with the replay branch weight**, and the balance controller
moves that weight during a run while `alpha_target` is a fixed config value:

| `w_rep` | loop gain | `stability_lr` | min `alpha_target` |
|---|---|---|---|
| 0.1 | 0.40 | 1.4310 | 0.70 |
| 0.3 | 1.20 | 0.9187 | 1.09 |
| 0.5 | 2.00 | 0.6738 | 1.48 |
| 0.7 | 2.80 | 0.5302 | 1.89 |
| 0.9 | 3.60 | 0.4360 | 2.29 |

**A 3.28× swing in the LR ceiling from the branch mix alone, with no sensor
able to observe it.** This is the mechanism behind `alpha_target` being a
per-route quantity: it is not a property of the optimizer, it is a property of
the coupling.

**Buffer churn is not the driver** — a clean negative. Across the whole range
(`kappa` 0.002 → 0.95) the boundary moves from 0.5265 to 0.5760, under 10%, so
the formula needs no churn correction. Churn governs how the run oscillates and
how fast it converges, not what rate it survives.

Scope caveat: the constraint binds only when the level moves at least as fast as
the policy (`lr_flow` ≥ ~0.5 in game units). mk_dev runs `lr_flow` 0.1 against a
policy rate near 1.25e-4 — an ~800× ratio — which is that regime. A *slow* level
is not the dangerous case: under anti-phase coupling a fast level amplifies the
oscillation and a sluggish one averages it out.

Sign note: a symmetric `+/+` coupling — the naive reading of "Z chases the
policy" — makes the 2×2 determinant `c − a·w_rep·b < 0`, a saddle, unstable at
every LR. The anti-phase sign is what makes the system have a boundary at all,
and it matches [[project-offpolicy-z-antiphase-verdict]].

**Not established here: the cost of the probe's objective mismatch.** Scoring the
replay objective alone vs the objective that actually trained moved the parked
rate from 0.084 to 0.071 and `final dist` from 1.9e-4 to 1.7e-4 — nothing. But
this game's two objectives *share their dominant term*, whereas the real defect
is that `var_conditioning`'s TB probe and VarGrad trained loss share **no** terms
at all. This is a weak test of that, and should not be read as clearing it.

---

## F-011 · A ray probe is structurally blind to the directions that limit the LR · `MECHANISM`

*2026-08-12. `bench/` `mle` game, `dim` 32, `cond` 300, SGD, calibration period
50, `alphas` [0..64]. Derived, and measured against ground truth.*

The probe rates the step it just took, so it measures curvature **along `d` and
nothing else**: `λ_eff = dᵀHd / dᵀd`. SGD stability is set by `λ_max` over *all*
directions. On an anisotropic surface the stiff modes converge first, `d`
migrates into the soft subspace, and `λ_eff` falls while `λ_max` does not move.

`λ_max = 300`, so the stability limit is `lr = 2/λ_max = 6.7e-3`:

| step | lr | `lr·λ_max` | `λ_eff(d)` | `alpha*` | status |
|---|---|---|---|---|---|
| 0 | 5.0e-6 | 0.002 | 210.5 | – | – |
| 100 | 1.4e-4 | 0.042 | 164.5 | 32 | `above_range` |
| 200 | 4.0e-4 | 0.120 | 56.2 | 32 | `above_range` |
| 300 | 1.1e-3 | 0.339 | 14.9 | 32 | `above_range` |
| 400 | 3.2e-3 | 0.960 | 3.8 | 32 | `above_range` |
| 450 | 5.4e-3 | **1.615** | 3.1 | 32 | `above_range` |
| 500 | 9.1e-3 | **2.715** | 2.7 | 32 | `above_range` |

The sensor read `above_range` for the entire climb, including the part past the
stability limit, and the run tripped the absolute divergence bar twice (loss
~1.9e9 and ~5.6e9). **Only that bar stopped it** — a 1800× open-loop ramp, the
same shape as lrdisc v1's 55×.

**It is not a grid artifact.** Widening the alpha grid to 2048 (so readings land
in range) does *not* fix it — that arm reached `lr·λ_max = 2.96` and tripped the
bar three times. Better resolution on the wrong quantity. `λ_eff` genuinely *is*
low; the probe's answer is correct and the question is wrong.

| arm | peak `lr·λ_max` | divergences | final dist |
|---|---|---|---|
| A shipping grid, saturated raises | 2.72 | 2 | 6.6e-4 |
| B saturated may not raise | 0.02 | 0 | **5.04** (no progress) |
| C wider grid, to 2048 | 2.96 | 3 | 1.2e-2 |

Arm B — refusing to raise on an out-of-grid reading — prevents the explosion but
leaves the run at its seed LR, because *every* reading is out of grid when the
seed is far too cold. Neither candidate is a fix.

This is the general form of "a ray probe at fixed θ cannot see the `tr(HΣ)`
term": the required margin is the **anisotropy ratio `λ_max/λ_eff(d)`**, a
property of the problem's conditioning. Together with F-012 it explains why
`alpha_target` is per-route and why a single constant cannot be right — the two
margins multiply.

Implication for a fix: any brake has to see directions the step does not occupy
— e.g. probing along a random direction as well as along `d`, or an explicit
curvature/grad-norm bound. Not yet tested.

---

## F-010b · Free-axis invariance holds on the ENERGY; the RDF reading of it is cutoff-sensitive · `MECHANISM`

*2026-08-12. `test_new_new_csd.pt`, 40 Z'=1 structures per space group, all `well_defined`.
Corrects an overstated claim in F-010, D33 and `configs/deadrow_aug12/make.py`, each of
which said the RDF was "exactly" unchanged on the strength of **6** mini-dataset structures.*

`canonicalize_free_axes` is **energy-invariant**: ≤1.2e-06 relative on 40 sg-1 and 40 sg-4
structures, the same float32 supercell-rebuild noise seen everywhere else, with the
centroid genuinely moving (0.49) rather than the call being a no-op. That is the property
the SDE change depends on, since the energy is what the GFN targets.

**The RDF is not a reliable witness to it.** 39/40 sg-4 structures gave exactly 0; one gave
0.054. That single structure resolves as an RDF-side counting effect, not a structural
change:

| `rdf_cutoff` | `supercell_size` | max abs Δrdf | rel abs Δelj |
|---|---|---|---|
| 6 | 5 | 0.0541 | 1.12e-06 |
| 8 | 5 | **0.0000** | 4.03e-07 |
| 10 | 7 | 0.0325 | 3.05e-07 |
| 6 | 7 | 0.0541 | 1.12e-06 |

A structural difference cannot vanish at cutoff 8 and return at 10. A pair sitting near the
cutoff radius crosses it under the shift, so it is counted on one side only; moving the
cutoff moves which pairs straddle it. `supercell_size` has no effect, which rules out a
cluster-size artifact, and the energy is invariant at every setting.

**Reading rule: assert gauge invariance on the ENERGY, not the RDF.** An RDF comparison at a
fixed cutoff carries a boundary-counting term that can reach ~0.05 on a single structure
even when the structures are physically identical. This is a companion to
[[feedback_wass_not_interpretive_on_crystals]] — another case of a structural metric being
less interpretive here than it looks.

---

## F-010 · The dead-row reduction is exact, and the log-weight variance it removes is measurable · `MECHANISM`

*2026-08-12. `test_dead_latent_rows_deep.py`. CPU, no training except where stated.*

Deep pass over the D33 SDE change, after the mechanical suite was already green.

**The reduction is BITWISE exact, not approximate.** `gauss_logprob` restricted to live
dims equals an independent live-only reimplementation to **0.00e+00** for dead sets
(3,5), (0,), (6,8) and (3,4,5,6,8), while differing from the full-width value by
0.29–1.02 nats. That separates "excluded the dims" from "excluded the dims and also
perturbed the surviving arithmetic".

**The additive log-weight variance is real and it is removed.** `Var(log P_F − log P_B)`
over 256 trajectories, T=10, falls monotonically with `n_dead`:

| dead rows | 0 | 2 | 3 |
|---|---|---|---|
| `Var(log w)` | 6.30 | 5.33 | 4.77 |

This is the quantitative form of the argument that decided D33 against pinning: dims
carrying no information contribute pure estimator variance, and holding them deletes
that term rather than shrinking it.

**Coverage.** 13 config settings × 5 dead sets = **65 models**, all satisfying the
three-way partition, trajectory constancy, finite log-probs, and `None == ()` bitwise.
Degenerate sets behave: `live_dim == 0` gives log-probs of exactly 0, not NaN.
T ∈ {1, 2, 60} fine. CPU/CUDA agree to ~1e-6 when scoring an identical trajectory
(sampling cannot be compared — separate RNG streams). Free-axis canonicaliser is
idempotent, and **energy-invariant to ≤1.2e-06 relative** — see F-010b for the RDF, whose
"exactly unchanged" reading did not survive a larger sample.

**The reduction is UNBIASED for log Z, measured against a closed form.** The GFN
importance-sampling estimator `logsumexp(log R + log P_B − log P_F) − log N` is
consistent for the true constant regardless of policy quality, so it settles bias
without any training. On an **untrained** policy, 20480 samples, 3 seeds, against a 2-d
isotropic target whose log Z = 1.8379 exactly:

| seed | A (dim 2, none dead) | B (dim 4, 2 dead) | \|A−B\| |
|---|---|---|---|
| 1 | +1.8389 | +1.8357 | 0.0031 |
| 2 | +1.8308 | +1.8435 | 0.0126 |
| 3 | +1.8377 | +1.8502 | 0.0126 |

B recovers the analytic constant to **0.012 nats**; A and B agree to **0.009**.

**A trained comparison nearly misled, and the resolution is worth keeping.** To 2500
steps A reached err −0.022 while B sat at −0.213, and the gap GREW with budget
(0.052 → 0.166 → 0.191) — the signature of a bias. Taken to longer budgets it collapses:

| steps | A err | A tb | B err | B tb | \|A−B\| |
|---|---|---|---|---|---|
| 2500 | −0.0223 | 0.0624 | −0.2130 | 0.2300 | 0.1907 |
| 6000 | −0.0175 | 0.0278 | −0.0084 | 0.0228 | 0.0091 |
| 12000 | −0.0103 | 0.0145 | −0.0050 | 0.0089 | **0.0053** |

By 12000 steps B is *closer* to the analytic constant than A and has the lower TB loss.
B was slower, not biased. The mechanism: `fwd_propagate` draws `batch × dim` noise per
step, so B consumes twice A's RNG stream and follows a different sample path from the
same seed — the two are effectively different random inits, and one seed each cannot
separate a bias from luck. **A growing gap is not evidence of bias. Prefer the IS
estimator for any log-Z-correctness question; reserve trained comparisons for rate.**

**Where the discrepancy actually lived.** Measuring the IS estimate and the learned flow
scalar side by side across budgets separates them completely (30720 samples per cell):

| steps | A IS err | B IS err | \|IS_A−IS_B\| | A **flow** err | B **flow** err |
|---|---|---|---|---|---|
| 0 | +0.0086 | +0.0023 | 0.0063 | −1.838 | −1.838 |
| 300 | +0.0062 | +0.0182 | 0.0120 | −0.394 | −0.487 |
| 1500 | +0.0038 | −0.0083 | 0.0121 | −0.204 | −0.278 |

**The SDE's normalizing constant is correct from step 0 in both models.** Only the learned
flow scalar lags, and it lags for both. So the trained-comparison gap was entirely a
flow-scalar convergence artifact and never a property of the sampler.

The control C (dim 4, nothing dead) has an IMPROPER target — two unconstrained dims mean
its Z is infinite — and its learned log Z duly runs away (4.42 → 10.64 → 11.55 with
budget). Holding the rows is what makes the problem well-posed at all.

**Two PRE-EXISTING limitations confirmed, not caused here:** DPLR + float64 raises
(a Float `V` meets a Double noise in `fwd_propagate`, identical with `dead=None`), and
the states buffer is always float32 because `init_traj_tensors` allocates at the default
dtype.

---

## F-009b · The `do_periodic_angles=False` angular mask was hardcoded to width 12 · `MECHANISM`

*2026-08-12. `gfn.py:295`, present unchanged at HEAD before this work.*

The non-crystal branch of `get_periodic_dimensions` built `angs = [False] * 12` rather
than `[False] * self.dim`. `lin_idx` derives from that mask, and
`expand_state_for_policy` selects on `lin_idx`, so at `dim > 12` the policy would have
been fed **only the first 12 dims of a wider state** — silently, with no shape error.
Reachable from config: a toy with `z_primes: [2]` has `data_ndim` 18. Unreached only
because every toy config to date sits at exactly 12.

Found by the width check inside `_finalize_dim_partition`, which exists for the dead-row
work and had no other purpose. Fixed to `self.dim`; verified the policy input width now
equals `dim` at 6, 12 and 18.

**A validation assert added for one feature caught a latent bug in another.** Worth
remembering when weighing whether a structural check is worth its lines.

---

## F-009 · The reduced-cell rewrite orphaned the monoclinic+ angle latents, and prior/energy disagree on them · `MECHANISM`

*2026-08-11. `mono_reduction_penalty` ([sym_utils.py:110](../../../mxtaltools/mxtaltools/common/sym_utils.py)), `enforce_crystal_system` ([geometry_utils.py:1310](../../../mxtaltools/mxtaltools/common/geometry_utils.py)), `instantiate_crystals` ([molecular_crystal.py:225](../energies/molecular_crystal.py:225)).*

The crystal-system pin exists — `(α − π/2)²`, `(γ − π/2)²` in
`mono_reduction_penalty` (ortho: all three; tetra/hex: `(a−b)²`), reaching the GFN
as `reduction_energy`. **In the GFN path it cannot fire.** `instantiate_crystals`
uses the default `skip=False`, so `enforce_crystal_system` sets those angles to
exactly π/2 *before* `analyze` evaluates `reduction_en` — the pin's argument is set
to its own target. It is live only in the crystal-search optimiser, which passes
`skip_enforce_crystal_system=True` ([crystal_opt_utils.py:635](../../../mxtaltools/mxtaltools/crystal_search/crystal_opt_utils.py)).

Each angle latent forced to −0.9/−0.3/+0.3/+0.9 through the real path,
`mean reduction_en`:

| latent | sg 2 (triclinic) | sg 14 (monoclinic) |
|---|---|---|
| `[3]` α | 15676 / 4935 / 137 / 0.74 — **live** | **0 / 0 / 0 / 0 — dead** |
| `[4]` β | 16931 / 6895 / 1039 / 1.14 — live | 0.563 / 0.078 / 0 / 0.009 — live |
| `[5]` γ | 10537 / 3635 / 208 / 0.46 — **live** | **0 / 0 / 0 / 0 — dead** |

Δ`zp1_cell_parameters` is also exactly 0 for sg-14 α/γ. β survives via the
reduced-cell inequality `cos β ∈ [−a/c, 0]`.

**The harm is not merely wasted capacity.** `latent_params()` round-trips the
clobbered angles to 0.0, so **every prior/buffer row has `latent[3] = latent[5] =
0.0` exactly** — measured on the stored sg-14 prior, std `0.0e+00` across n=535 and
n=3582. The energy is meanwhile flat over the whole box there. So `bwd` (starting
from buffer states) trains P_F toward a delta at 0 while `fwd` gets no gradient
signal at all: 2 of 12 dims trained against inconsistent targets.

**Provenance.** `8dea6b56` — "Niggli sampling ripped out in favor of general latent
space and an energy based penalty". The old design sampled *inside* the reduced
domain. The rewrite moved reduction into a penalty, which works for triclinic but is
preempted for monoclinic+ by a clobber that predates it (`33905974`).

**Scope.** Triclinic is `enforce_crystal_system`'s "anything goes" branch, so
**sg 1 and sg 2 are unaffected** — 63 sg-2 and 9 sg-1 configs are clean, including
`mk_dev`, `aug02` and `tw_july31` (all sg 2). Affected live configs: `acridine/`,
`uncond_14_1_test{,2}` (36 further sg-14 configs are under `configs/old/`). Dead
angle latents of 12: monoclinic **2**, orthorhombic/tetragonal **3**.

**`CONJECTURE`:** this should show as a floor on the fwd/bwd gap and on `tb_err` in
monoclinic+ runs that no training removes. The driver is measured; the consequence
is not.

---

## F-008 · Free aunit axes are the shared +1 eigenspace of G, and no discrete fold can reduce them · `MECHANISM`

*2026-08-11. `CONTINUOUS_DIMS` ([space_group_info.py:15217](../../../mxtaltools/mxtaltools/constants/space_group_info.py)).*

Moving the aunit centroid by δ displaces each symmetry copy by `R_k δ`. That is a
rigid translation of the whole crystal — hence exactly energy-preserving — iff
`R_k δ = δ` for every rotation part in G. So the free subspace is
`∩_k ker(R_k − I)`, and its dimension is `3 − rank[R_k − I]`.

Derived from `SYM_OPS` alone, this **agrees with cctbx `structure_seminvariants`
on 230/230 space groups**. 68 of 230 have ≥1 free dim, 42 of those have a real
aunit box. **All 230 have axis-aligned (principal) continuous shifts**, so gauge
fixing is deleting a fractional axis — no change of basis anywhere.

The orbit is continuous, so the N_E coset fold has nothing to compare against and
cannot canonicalise it. Signature, sg 4 (P21, free axis y), n=6: post-fold x
reduces to 0.240 and z to 0.378, **y still spans to 0.879**.

At Z'>1 the free translation is **one global** shift shared by all Z' molecules —
delete it from one centroid block only. Per-molecule zeroing would destroy the
physical relative offsets along that axis.

`sg_periodic_centroid_axes` ([aunit_periodicity.py](../models/aunit_periodicity.py))
under-reports: 16 of the 42 have a free axis with `auv < 1` (C2 y, Cmc21 z, I4 z,
P4cc z, …), flat and wrapping at period `auv_d` but currently walled.

---

## F-007 · The N_E fold reproduces the P21/c fundamental domain, and the domain depends on molecular chirality · `REPLICATED`

*2026-08-11. `get_fd_params` ([crystal_ops.py:843](../../../mxtaltools/mxtaltools/dataset_utils/data_class_methods/crystal_ops.py)).*

**Scope:** `mini_new_csd.pt`, Z'=1, n as tabled — a different dataset from the
7/3 session's `test_new_new_csd.pt` (n=100), which is the replicate. Geometry
only; no training.

Post-fold max fractional centroid, achiral / chiral gate:

| sg | n | index | aunit | post-fold max x,y,z |
|---|---|---|---|---|
| 14 P21/c | 41 | 8 | [1, .25, 1] | **.250 / .247 / .499** (both) |
| 2 P-1 | 10 | 8 | [.5, 1, 1] | .247 / .423 / .327 (both) |
| 15 C2/c | 9 | 4 | [.5, .5, .5] | .211 / .246 / .466 (both) |
| 19 P212121 | 12 | 16 | [.5, .5, 1] | .249/.216/**.241** vs .249/.216/**.406** |
| 61 Pbca | 5 | 8 | [.5, .5, .5] | .185 / .234 / .227 (both) |

sg 14 lands on `[0.25, 0.25, 0.5]`, volume 0.03125 = aunit/8 exactly, confirming
the 7/3 lexicographic tie-break result. Small n elsewhere — extents are lower
bounds, not saturated.

**The FD box is not derivable by per-axis division of the seminvariant moduli.**
That gives sg 14 `[0.5, 0.125, 0.5]` — correct volume, wrong shape, because the
(0,½,0) generator is already consumed by G's own screw fold while x and z absorb a
coupled extra factor.

**132 SGs have an improper element in N_E, so their FD is chirality-dependent** —
per-molecule, not per-space-group. `is_chiral` is unplumbed; the `None` default
resolves through `(~True) | (det>0)` → int64, collapsing to "all chiral" by
two's-complement accident.

Re-ran the G-box Monte Carlo: **the same 12 SGs still fail** (Pnnn, Pban, Pmmn,
Ccce, Fddd, P4/n, P42/n, P4/nnc, P4/ncc, P42/nbc, P42/nmc, I41/amd — 0.00 single-image
coverage). 123 of 230 have real box data.

---

## F-006 · The memorisation setpoint is derived, so it transfers · `MECHANISM`

*2026-08-07. `absorption_stats` ([buffer.py:863](../buffer.py:863)).*

`ratio = mean(ema_loss)/mean(birth_loss)`. Under exponential relaxation at rate
λ and exponential residence with mean τ, `ratio ≈ exp(−λτ)`, so the `λτ = 1`
boundary is **`ratio = 1/e = 0.368`** exactly. Nothing in the setpoint was
measured, so it transfers across problem, `T` and buffer size — the property
every previous buffer threshold lacked.

No survivorship bias, and that is a dividend of the uniform hazard: under a
residual-independent hazard, resident `birth_loss` is an unbiased sample of
admits. This would **not** hold under floor/stalled eviction.

Discriminates on 33 historical runs: λτ > 1.0 on four arms (BASE32K 1.54,
local_aug02 1.44, neat_dev 1.10), 0.5–1.0 on five, < 0.5 on the rest.

**A 1-D Wasserstein between intake and resident loss histograms matches the
mean-shift statistic to three decimals on every arm** — the distributions differ
by a translation, so the histogram machinery buys nothing. Do not re-propose it.

---

## F-005 · The prioritised draw is unbiased at every κ, and the variance payoff is not real · `MECHANISM`

*2026-08-07. `prioritised_weights` ([buffer.py:915](../buffer.py:915)).*

`p ∝ δ₊^κ` with `w = (1/n_elig)/p` gives `E_p[w·f] = E_uniform[f]` **at every
κ**. Unbiasedness is exact by construction; only variance changes with κ. So any
difference a κ ladder measures is estimator variance and nothing else.

**The Cauchy–Schwarz prediction that variance is minimised at κ=1 is wrong.**
Measured over 300 draws of 1000 rows, ESS/n runs 1.00 / 0.85 / 0.65 / 0.34 at
κ = 0 / 0.5 / 1 / 2 and batch sd moves the wrong way (0.38 → 2.23). The optimal
draw for a *self-normalised* estimator is `p ∝ |f − μ|`, not `p ∝ |f|`; δ is
tightly clustered about its own mean, so prioritising by δ over-samples where the
integrand is least informative.

**Correctness is established, payoff is not** — the κ ladder is diagnostic, not
confirmatory.

`floor_frac` is a relative floor on survivors (fraction of median `δ₊`), so the
weight range is bounded by `(median/floor)^κ`. Measured against a live buffer:

| `floor_frac` | ESS/n | max(w)/mean(w) |
|---|---|---|
| 0.01 | 0.11 | 73 |
| 0.15 | 0.50 | 5.3 |
| **0.25** | **0.63** | **3.3** |
| 0.50 | 0.80 | 1.9 |

**0.25 is the knee and is the default.** The shipped 0.01 gave `is_ess_frac`
0.02–0.06 live — a 1000-row batch doing the work of ~20–60 rows.

---

## F-004 · At κ=0 the IS estimator must read `is_ess_frac` exactly 1 · `MECHANISM`

*2026-08-07, found by the degenerate cell of the κ ladder.*

At κ=0 the draw and the weights are both uniform, so `is_ess_frac` is **exactly
1** and `is_w_max_ratio` **exactly 1**. Anything else means the draw is
mis-wired, not that the estimator is noisy.

This is a standing invariant, and it caught a live defect no unit test could:
`beta` is a **uniform fraction, not a temperature** — `_sample_indices` splits
the batch as `n_uniform = int(batch_size · beta)`, so a supplied `p` was silently
ignored while the weights `w ∝ 1/δ₊^κ` were still applied, targeting a measure
`∝ 1/δ^κ`, the exact inverse of the design. It read 0.40.

**A unit test of the estimator cannot catch a mis-wired draw.** Always put a
degenerate cell in a ladder.

Related class: a checkpointed per-row field with no reader is indistinguishable
from a live one when reading the schema. `update_logw_stats` was checkpointed,
resized on grow/purge, and called by nothing for months.

---

## F-003 · Uniform intake trades the forward tail for typical-population fit · `REPLICATED`

*2026-08-08. `local_aug09`, five isolation arms plus two full-length runs.*

**Scope:** T=10, mipcas ELJ, naive stage, 3600 steps. Seed floors quoted.

Turning on the B7b package moves buffer hardness because of **admission, not
eviction**. `birth_loss` is snapshotted once at admission and never updated, so
it is a pure admission statistic: **23.73 → 10.86**. Rows now enter with less
than half the residual they used to.

Verdict at 3600 steps (v7 = κ 1 / β 10, final window vs `a_frz`):

| | `a_frz` | v7 | gap | seed floor |
|---|---|---|---|---|
| **`bwd/tb_err`** | 15.14 | **14.64** | **−0.50** | 0.04 ✅ |
| `fwd/tb_err` | 18.72 | 23.12 | +4.40 | 0.52 ❌ |
| `EffDim` | 5.80 | 5.90 | +0.11 | 0.10 — |

`bwd` draws the prior buffer, a fixed diverse population; `fwd` is fresh
on-policy rollouts. The new construction fits the typical population better and
leaves the forward tail uncorrected — exactly what hard-tail-skimming admission
was buying. **The fwd gap is stable, not closing** (per-window 3.12, 3.97, 5.61,
5.09, 4.27, 4.09, 4.39).

Isolation arms, each killing a candidate mechanism:

| arm | κ | `beta` | `is_ess` | `w_max` | `fwd/tb_err` | rules out |
|---|---|---|---|---|---|---|
| **v4** | 1 | **10** | 0.363 | 7.3 | **27.29** | — best |
| v0 | 1 | 1e6 | 0.393 | 6.7 | 33.87 | — |
| v3 | 1 | 1e6, `max_size` 4000 | 0.396 | 6.7 | 33.84 | displacement purge (≡ v0) |
| v6 | **2** | 10 | **0.073** | **58.4** | 36.29 | κ-sharpening |
| v5 | **0** | 1e6 | **1.000** | 1.0 | 38.60 | IS-weight variance |

**The admission gap cannot be bought back through the draw.** The variance bound
bites before κ = 2, so κ ≈ 1 is the practical ceiling. De-huberising costs ~6.6
nats and is not a route either; independently replicates the `local_aug07` β
ladder.

**Read `replay/tb_err` and `replay_buffer_mean_loss` together.** The former
rising 16.9 → 23.5 while the latter falls to 5.75 is the draw **working** — a
κ=1 draw skimming the hard tail of a softening buffer. Read alone, it looks
broken.

**Watch `is_elig_frac`.** It drifted 0.74 → 0.33 over 1500 steps locally. At 0
the prioritised branch has nothing to draw.

---

## F-002 · `fwd/tb_err` cannot be read off point samples · `MECHANISM`

*2026-08-10, extracted from the pair-A analysis.*

Per-eval scatter on `fwd/tb_err` is **±1 nat**, comparable to the effect sizes
being chased. Sampling at 0/25/50/75/100% indices produced a spurious late
upturn and a spurious dead heat, both of which vanished under **binned medians**
over 400-step windows.

**Read trajectory metrics as binned medians. Never as point samples.**

→ *Pending promotion to `module_metrics.md` in the migration; this applies to
every future reading, not only to F-001.*

---

## F-001 · Unfreezing the policy on `fwd` improves `bwd/tb_err` · `REPLICATED`

*2026-08-08. Feeds `decisions.md` D30.*

**Scope:** T=10, mipcas ELJ, naive stage, 3600 steps from a shared post-transient
resume @2650, both arms verified to start at that step. Pairs A + D; pair B is
the seed replicate. **T=25 not measured.**

`bwd/tb_err`, final window. Seed floor **0.04** (frz) / **0.10** (unf):

| | lr 1.25e-4 | lr 2.15e-4 | LR effect |
|---|---|---|---|
| **frozen** | 15.14 | 16.06 | +0.92 (worse) |
| **unfrozen** | 14.07 | 14.63 | +0.56 (worse) |
| freeze effect | **−1.07** | **−1.43** | |

Effects are 10–35× the seed floor. `EffDim` is flat at ~5.8 in all four cells,
so the gain is not bought with coverage.

**It is not an LR effect.** The substitution test fails in the wrong direction:
if unfreezing were simply more LR, `frz@2.15e-4` should land on `unf@1.25e-4`;
it lands 1.99 nats worse. Raising LR hurts both rows, and the freeze benefit
*grows* at higher LR. Corroborated by `step_norm` (0.06496 frz vs 0.06360 unf —
a 2% difference).

**Supersedes** an 800-step n=1 reading that frozen *degrades* (21.94 → 23.54).
That did not reproduce at 4.5× the length. Frozen is **slower, not degrading** —
a materially weaker claim than the one `synthesis.md` §1 is in tension with.

**Blocked:** all 26 rb0808 arms ran `freeze_policy: 1.0`, so every replay, `beta`
and Z result in that battery was measured inside the slower regime. T=25 needs
resubmission.
