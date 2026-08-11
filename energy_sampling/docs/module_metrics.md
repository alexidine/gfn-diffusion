# Module: target metrics (`utils.py::quick_tb_stats`, `MetricTracker`, `sample_metrics.py`)

Pass 1 (audit + rationalize). Verified against the working tree, 2026-08-03.
Unconditional route.

---

## 1. What it is

Three layers:

1. **[`quick_tb_stats`](utils.py:1280)** — computes ~20 scalars per batch from
   `(log_pf, log_pb, log_Z, log_r)`. This is where every control metric is
   defined.
2. **[`MetricTracker`](utils.py:1206)** — a step-aware EMA (`period=100`) keyed
   by `(direction, name)`, where direction ∈ {`fwd`, `bwd`, `replay`}. Also
   tracks running min/max of the EMA.
3. **Eval-time metrics** — `eval/wass_debiased` and friends, computed on a full
   eval batch rather than a training batch.

Paper voice: *training is monitored by the trajectory-balance residual
decomposed into a level component (the signed, Huber-clipped mean, which is
dL/dZ) and a spread component (the RMS about zero), plus reward-ramp-weighted
coverage statistics that separate under-sampled high-reward regions from
over-weighted low-reward ones.*

## 2. The organising idea

The single best thing in this module is the level/spread decomposition, and it
should be stated first in any writeup:

> E[r²] = mean(r)² + Var(r)

`z_grad_worst` (level) is the part **Z training can fix**; the excess of
`tb_err_worst` over it is the part **only policy training can fix**. That is
directly the actuator question — which knob can act on this error — and it is
what makes the metric family a control surface rather than a dashboard.

The second good idea: **every control metric is an EMA-safe per-sample mean,
never a ratio.** The r² family this replaced could not be EMA'd (a ratio of two
EMAs is not the EMA of the ratio) and was unreachable at 2–3 samples/condition.

## 3. Three classes, not two

*(corrected 2026-08-05 — the original two-way split was wrong.)*

~590 metric keys are emitted. They fall into **three** classes, and only the
third is deletable:

1. **Load-bearing** — a code path reads it and changes behaviour. Twelve keys.
2. **Diagnostic** — no code consumer, but *read by a human on purpose*. These
   look identical to dead code from a call-graph audit and are not dead. Their
   consumer is a wandb panel and an eye.
3. **Dead** — neither. The MMD family is the clear case.

"No code consumer" is therefore not evidence of anything. The same mistake was
made about the loss term bank, where the answer was also "kept deliberately."
When a call-graph sweep says zero callers, the question to ask is *who reads
this*, not *can we delete it*.

The twelve that drive behaviour:

| Metric | Consumer | Effect |
|---|---|---|
| `gates/mle_flat` | phase-1 exit | stage transition |
| `eval/wass_debiased` | phase-1 exit | stage transition |
| `bwd/tbc` | phase-1 exit | stage transition |
| `bwd/relative_under_wcen` | `naive` balance (denominator of ρ) | bwd loss weight |
| `fwd/over_coverage` | `naive` balance (numerator of ρ) | replay loss weight |
| `fwd/scatter_err` | replay admit cap | admission sharpness |
| `fwd/r2` (EMA) | anchor health gate | pauses anchor admission |
| `fwd/tb_resid_clipped` | anchor health gate | pauses anchor admission |
| `logw_std` | terminal detector | rewind + LR cut |
| `box_violation` | terminal detector | rewind + LR cut |
| branch loss | LR tripwires | cut / reset |
| `grad_norm_pre_clip` | LR tripwires | cut / reset |

Everything else is class 2 or 3. That is not a criticism — diagnostics are how
the batteries get read — but it means the *control* surface is small enough to
reason about completely, and it should be presented that way.

### 3a. Three families added 2026-08-07 — and one of them is class 1

*(added 2026-08-08. All three are absent unless their feature is configured, so
they do not widen the default surface.)*

| Family | Class | Keys | Read by |
|---|---|---|---|
| **memorisation** | **1 on servo arms**, 2 otherwise | `replay/resid_vs_intake`, `lambda_tau`, `absorbed_frac`, `ema_loss_mean`, `birth_loss_mean`, `absorption_n` | `buffer_servo` via `_resolve` — see T7 |
| **prioritised-IS health** | 2 (diagnostic, but the *only* thing that can go wrong) | `replay/is_ess_frac`, `is_w_max_ratio`, `is_elig_frac` | a human; the estimator is unbiased at every κ, so variance is all there is to watch |
| **step probe** | **1 since 2026-08-08** (was 2 — sensor only) | `lrprobe/alpha_median`, `alpha_iqr`, `alpha_n`, `fit_*_rate`, `bad_rate_window`, `second_diff_rel`, `step_norm`, `loss_delta_rel` | the **LR servo**, via `StepProbe.servo_reading()`; `module_lr_controller.md` F0/F5 |

**`replay/ema_loss_mean` and `birth_loss_mean` are the two keys that promote the
memorisation family to class 1**, because they are what a configured
`buffer_servo` resolves. The derived bar is `resid_vs_intake = 1/e = 0.368`
(`module_buffers.md` B8), so unlike every other threshold in this table it needs no
calibration.

Controller-state series also grew, all class 2 and all self-describing:
`protocol/rt_*` (`ratio`), `protocol/cs_*` (`constraint`), `protocol/bs_*` (buffer
servo). Two reporting rules were applied at the same time and are worth stating as
rules, because both were violated first:

- **A series is emitted only by the kind that writes it.** `protocol/anneal_streak`
  used to be emitted unconditionally, publishing a constant 0 on every non-
  lexicographic kind — *a flat series that looks like a reading*. `protocol/boost`
  is likewise suppressed under `ratio`, where it is just `sign(rt_err)` recoded as a
  3-valued categorical, i.e. strictly less information than `rt_err` itself.
- **Every controller emits its actuator, not only its sensor.** `protocol/bs_log_boost`
  exists because a servo reading fine but with no authority looked identical to one
  correctly holding.

### 3b. The LR families, 2026-08-08 — and the promotion that came with them

The step-probe family **moved from class 2 to class 1** when `controller.py` v7
started reading it. That is the single most consequential thing in this document
right now: a metric a controller reads is one whose *absence* is a silent
failure, and the probe is absent unless a `step_probe` block is declared.
`resolve_derived_config` therefore **hard-fails** a config that writes `auto` on
an `lr_*` key without one — the servo cannot be configured without its sensor.

New family, all written by `LRController.report()`:

| Key | Meaning | Why it is not optional |
|---|---|---|
| `lr_ctrl/peak_scale` | the servo's **actuator** | the configured LR is no longer what ran. Read this first on any arm |
| `lr_ctrl/peak_ceiling` | post-divergence ceiling, relaxing | present only after a divergence |
| `lr_ctrl/envelope` | warmup multiplier | separates "still ramping" from "servo moved it" |
| `lr_ctrl/servo_hold` | **why** the loop is not moving: 0 acting, 1 disabled, 2 no_probe, 3 warmup, 4 cold, 5 few_readings, 6 fit_invalid | the actuator/sensor rule above, applied to the third controller |
| `lr_ctrl/divergences` | count | replaces v6's per-channel `fires_*` |

`lrprobe/fit_beyond_rate` is new and reads oddly at first: a **high** value is
not a fault, it is "the LR is below the probe's resolving range" (§A3b). The
signature of a working servo is `fit_beyond_rate → fit_ok_rate` as it climbs.
`bad_rate_window` deliberately excludes `beyond` so it matches the servo's own
validity gate exactly — a logged gate that disagrees with the live one
misreports why the loop held, which is the same class of defect as the two
reporting rules above.

### 3c. The non-thermal tail family, 2026-08-10 — class 2, and deliberately so

The high-energy tail was previously only inferable — from `Mean Sample Energy`,
`Reasonable Sample Fraction`, an effective temperature read off a histogram. The
family at [`train.py:3396`](../train.py:3396) states it directly. All class 2:
nothing reads them, and nothing should until the bar has been watched on a real
battery.

The quantity is the **reduced excess energy**, per sample:

```
u = (E - Emin(c)) / T   ==   log R*(c) - log R
```

in nats. Both condition- and temperature-reduced, so unlike a raw energy — and
unlike `r2` (T0) — it **pools legitimately** across a mixed-condition, mixed-T
eval batch. Emin(c) is `condition_log_z`'s running record, which
`fwd_eval_sampling` has already updated with the same batch, so `u >= 0` holds
by construction and a uniformly bad batch cannot flatter itself.

| Key | Meaning |
|---|---|
| `Nonthermal Fraction` | **headline** — frac(`u > u*`) |
| `Nonthermal Threshold` | `u*` in nats; emitted only when it moves (S3's rule, applied at birth) |
| `Excess Energy Nats P50/P90/P99/Max/Mean` | the tail's **shape**, so the bar is not the only view |
| `Excess Energy Nats` | histogram of `log10(1+u)` |
| `Excess Energy Referenced Fraction` | share of the batch that had an Emin(c) to score against |

**Where the bar comes from is the whole point.** Under any Boltzmann target,
the cost of sitting `u` nats up is `e^-u` and the only thing that can pay it is
the number of states up there:

```
P(u > u*) = int_{u>u*} g(E) e^-E/T dE / Z  <=  (V_acc / V_ref) e^-u*
```

so `log P <= S - u*` with `S = log(V_acc/V_ref)` the **entropy budget**. The
latents live in a bounded box, so `S` is finite and extensive in the latent
dimension — hence `u* = data_ndim * nonthermal_entropy_per_dim`, one knob that
transfers across problem and T by construction. The default `s = 4` reads two
ways at `data_ndim` 12: as a pure budget it grants the reference region as
little as `e^-4` ~ 1.8% of each axis; or, taking `dup_cutoff` (5% of an axis) as
the reference size, `S = 36` and the other 12 nats are rarity margin (~1 draw in
1.6e5). Measured on synthetic thermal batches (harmonic well, `data_ndim` 12,
both fixed T=2.5 and the mk_dev log-T sweep, n=2000): `max(u)` 16.8–19.7 against
a bar at 48. A known 5% contamination is recovered as 0.055.

Two biases, **both toward under-reporting**, which is the right direction for a
metric whose claim is "obviously": early in a run Emin(c) is only as deep as
what has been seen; and non-finite energies are patched to 0 upstream
([`molecular_crystal.py:238`](../energies/molecular_crystal.py:238)), so a
numerically blown-up sample reads as mildly excited rather than as tail. Neither
can manufacture a tail that is not there.

Read `P99` before `Nonthermal Fraction`: the fraction is one bar on a
distribution, and a tail growing *under* the bar is the same event seen earlier.

## 4. Unconditional degeneracies

With one condition, a large part of the family collapses:

| Name | Unconditional value |
|---|---|
| `cond_tb_err` | == `tb_err` |
| `tb_err_worst` | == `tb_err` |
| `z_grad_worst` | == `\|tb_resid_clipped\|` |
| `logw_std_within` | **omitted entirely** |
| `relative_under` | pooled (unchanged) |
| `relative_under_wcen` | differs from `relative_under` only because the ramp weights are non-uniform |
| `conditional_worst_quantile: 0.25` | **inert** |

This is deliberate and good: the docstring's stated goal is that "the same metric
names carry the same meaning on conditional and unconditional runs, so protocol
rules need no per-problem rewriting." It works. But it should be stated in any
unconditional writeup that `*_worst` is not a worst-case anything here — it is
the pooled value under a different name, and a rule that reads it is reading
`tb_err`.

## 5. Findings

**T0 — `r2` is not comparable across branches. Its denominator is batch
diversity.** *(measured 2026-08-08, four arms, reproducible)*

[`utils.py:1418`](../utils.py:1418) computes

```
x = log_pb + log_r ;  y = log_pf + log_Z ;  resid = y - x
r2 = 1 - sum(resid^2) / sum((y - ybar)^2)
```

so the normaliser is the spread of `y` **within that branch's own batch**. Two
branches with identical residuals report different `r2` whenever their sample
diversity differs — and on this route it differs by a lot. Recover the
denominator as `sigma_y = tb_err / sqrt(1 - r2)`:

| branch | `sigma_y` | `tb_err` | `r2` |
|---|---|---|---|
| fwd | 10.2 – 11.9 | 18.6 – 19.9 | −1.5 to −2.8 |
| **replay** | **20.2 – 21.0** | 15.7 – 15.8 | **+0.39 to +0.44** |
| bwd | 12.1 – 13.5 | 14.9 – 15.8 | −0.21 to −0.71 |

Replay batches carry ~4× the *variance* of fwd batches — unsurprising once
stated, since a buffer accumulating rows over `mean_residence_steps: 50` from
many policy states is a more diverse population than one on-policy batch.
Consequence: `replay/r2 = +0.39` against `fwd/r2 = −2.72` is **mostly the
denominator**. Standardise fwd's residual on replay's `sigma_y` and it scores
**+0.04**; the honest gap is ~0.35, not 3.11.

This is not a new defect — §3's control-metric family was introduced precisely
because "the conditional r2 family this replaced could not be EMA'd", and the
`quick_tb_stats` docstring already says `tb_err` has "no group-mean denominator
to collapse". What is new is the measured size of the effect, and that it is
large enough to invert a qualitative reading. **Cross-branch comparisons go on
`tb_err`.** `r2` stays useful within a branch over time, where the denominator
drifts slowly.

Corollary for **P3** (`replay/scatter_err ÷ fwd/scatter_err`): a ratio below 1 is
equally the signature of memorisation — which is what P3 exists to detect — and
of a coverage gap. The statistic does not distinguish them, and reading it as
either one alone is unwarranted.

**T1 — `sample_metrics.py` is ~65% class-3 dead code.** *(confirmed; deletion approved)*

Of 309 lines, only `wasserstein` has external callers (6). Zero external callers
for `linear_mmd2`, `poly_mmd2`, `mix_rbf_mmd2`, `mix_rbf_mmd2_and_ratio`,
`_mmd2`, `_mmd2_and_ratio`, `_mmd2_and_variance`, `compute_distances`, and
`compute_distribution_distances` — including the top-level entry point.

Unlike the loss term bank and the class-2 diagnostics, this one is genuinely
dead: nothing computes it, so nothing reads it either.

**T2 — the bwd side of the phase-2 balance controller was inert.** *(✅ **dissolved 2026-08-06** by `kind: ratio` — `module_protocol.md` P7. Kept because it governs how every result produced before that date must be read.)*

`targets: { bwd: 3.0 }` was read off a run measuring the **old** `relative_under`
(band 2.48–3.18). The metric has since been swapped to `relative_under_wcen`,
which drops the batch-composition offset and therefore reads **lower by an
unmeasured amount**. Under `drive: relative`, a target above the metric's
operating range pins the bwd drive at 0 — which under the tilt makes the split
exactly `default_boost`.

So as configured, **the phase-2 controller was one-sided**: replay's drive live,
bwd's pinned, and the split sitting at the fixed 0.75/0.25 idle ratio. Any battery
run on that config measures a one-sided controller, not a two-metric balance —
which is how it should still be read.

**Resolution was not the measurement it looked like.** The obvious action was to
read the settled band and reset the target. What shipped instead removes the
target: `kind: ratio` divides the two metrics and holds their ratio at one
setpoint, so there is no per-side level to calibrate and no `max(·, 0)` for a
satisfied side to fall through. Both metrics stay exactly as they are — only what
the controller does with them changed.

**T3 — non-finite readings hold the EMA, and nothing counts staleness.** *(confirmed)*

[`MetricTracker.update`](utils.py:1223) skips non-finite values, so the EMA
holds its last value. Combined with `under_coverage = nan` meaning "no sample
cleared the ramp floor" — a deliberate choice over a fake 0 — a stage where
nothing qualifies keeps reporting the last real value **indefinitely**, and a
controller reading it cannot distinguish "stable" from "no data since step
4000."

The nan-not-zero decision is right; the missing half is a per-key staleness
counter (steps since last finite update) so consumers can abstain. Cheap to add,
and it would have made several past diagnoses faster.

**T4 — `tb_resid_clipped`'s meaning depends on a global invariant nothing
enforces.** *(confirmed, latent)*

It is dL/dZ only up to the constant `beta` scale, so the docstring states the
Huber beta "must be held FIXED across the whole protocol for this to mean one
thing." mk_dev holds 10.0 in all three coefficient blocks, so it is currently
true — but `beta` is a per-mode, per-stage-overridable coefficient like any
other. A stage override would silently rescale the metric that the anchor health
gate keys on. One assertion at `set_loss_coeffs` would close it.

**T5 — the step-aware EMA handles dormancy correctly, and that is load-bearing.** *(confirmed, positive)*

`alpha = 1 − exp(−dt/period)` with `dt` measured **per direction**. A branch
returning after a long gap gets `alpha ≈ 1`, so its EMA is fully replaced rather
than blended with stale pre-gap data. This is why `period=100` does not
cross-contaminate across stage transitions for dormant modes. Non-obvious, worth
keeping, worth stating.

**T6 — `relative_under`'s re-centring rests on a real theoretical argument.** *(derived, positive)*

The collective level gap (learned Z vs buffer-implied Z) is *not* something
backward training can act on: `E_μ[log P_F]` is capped at `−H(μ)` by
normalisation, so the whole cloud cannot translate. A Z-anchored `under_coverage`
therefore reads "everything is under-covered" whenever Z lags, starving the
controller's other modes. Re-centring on the batch's own empirical normaliser
isolates the spread component that *is* the policy's to fix. This is the
best-derived metric in the module and the argument should appear in the paper.

**T7 — there are TWO publication paths, and only one of them is readable by a
controller.** *(found the hard way 2026-08-07; ✅ the instance is fixed, the class
is not guarded)*

A metric can reach the run two ways, and they are not equivalent:

| Path | Reaches | Read by |
|---|---|---|
| the report dict (`metrics.update(...)` in the eval/report path) | wandb | a human |
| `metric_tracker.update(direction, {...}, step)` | the tracker **and** wandb | `StageProtocol._resolve`, i.e. **every controller** |

`absorption_stats()` was published on the **first** path only. Every controller
sensor is resolved via `metric_tracker.get(direction, metric)`, so the tracker never
held `replay/ema_loss_mean`, the `buffer_servo` resolved `None`, and it took its
cold-start early return on every single tick — **while emitting no actuator series
at all**, which is indistinguishable in the logs from a servo correctly holding.
A deliberately induced overfit ran to `lambda_tau` 1.43 with the actuator inert.

Fixed by publishing from `replay_train_step` ([train.py:3116](train.py:3116)).
**But nothing prevents a recurrence**: the two paths look equally correct at the
call site, and "is this metric controller-visible?" is answerable only by grepping
for `metric_tracker.update`. The cheap guard is a load-time check that every
metric named by a stage's `balance`/`exit`/`buffer_servo` is one the tracker can
actually produce — the same preflight shape as `module_buffers.md` S2. See S5.

This is the metrics-module face of a pattern with three instances now
(`module_modulators.md` D7): **an unreadable sensor and a satisfied controller are
the same silence.**

## 6. Warrants

| Choice | Warrant | Evidence |
|---|---|---|
| Level/spread decomposition as the actuator map | **derived** — E[r²] = mean² + Var | docstring |
| EMA-safe per-sample means, never ratios | **derived + measured** — the r² family was un-EMA-able and unreachable at low samples/condition | replaced family |
| `relative_under` re-centred on batch normaliser | **derived** — `E_μ[log P_F] ≤ −H(μ)` | docstring, and the bwd-stationarity result |
| `relative_under_wcen` (weighted centre) | **derived** — scoring and centring must run over the same population | docstring at [line 1393](utils.py:1393) |
| `under_coverage = nan` when nothing qualifies | **derived** — a fake 0 is worse than no reading | docstring |
| `tb_resid_clipped` over `tb_resid` / `tb_err` | **derived** — bounded by beta, so fat tails can't inflate it; a lagging Z shows as persistent sign | docstring at [line 1301](utils.py:1301) |
| `MetricTracker period=100` | **derived** — the tracker's job is a *faithful* smoothed picture of the live policy: if the policy shifts fast, this must shift fast. Filtering belongs to each consumer, not to the sensor (user, 2026-08-05) | separation of concerns |
| ~~Balance targets `bwd: 3.0`, `replay: 18.0`~~ | **retired 2026-08-06** — `kind: ratio` replaces both with one `setpoint` on their ratio, so neither level is calibrated any more | T2 |
| Balance `setpoint` on `over_coverage ÷ relative_under_wcen` | **measured, provisional** — observed band 5.4–6.2 across four ctrl_aug03 arms | `module_protocol.md` P7 |
| `health_gate_r2: 0.9`, `health_gate_zerr: 0.5` | **arbitrary** — and since 2026-08-03 the *metric names* are config-driven too (`health_gate_floor_metric` / `_ceiling_metric`), so swapping the ruler is an A/B rather than a code change | `module_buffers.md`, anchor health gate |
| `worst_quantile: 0.25` | **arbitrary**, and inert here | — |
| `resid_vs_intake` bar at `1/e` | **derived** — `ratio ≈ exp(−λτ)` puts the `λτ = 1` boundary exactly there, so it transfers across problem, T and buffer size | `module_buffers.md` B8 |
| Shipping `is_ess_frac` / `is_w_max_ratio` / `is_elig_frac` at all | **derived** — a self-normalised IS estimator is unbiased by construction, so the weight *tail* is the only failure channel and it needs its own instrument | `module_buffers.md` B7 |
| Suppressing a series on kinds that never write it | **derived** — an unconditional emit publishes a constant 0, which reads as a measurement | §3a |
| Non-thermal bar at `data_ndim * s` rather than an absolute energy | **derived** — the entropy budget of a bounded latent box is extensive in its dimension, so this is the one form that transfers across problem, T and Z' | §3c |
| `nonthermal_entropy_per_dim: 4.0` | **arbitrary, conservative** — two readings both land near it (§3c), and synthetic thermal batches top out at `u` ~ 19 against a bar at 48 | §3c |
| Reducing by T and by Emin(c) before pooling | **derived** — `u` is dimensionless and per-condition-referenced, so pooling carries none of T0's denominator hazard | §3c |

## 7. Failure signatures

| Symptom | First check | Cause |
|---|---|---|
| Controller not responding to a metric | `protocol/rt_err` and `rt_gain_scale` | under `ratio`: `rt_err`≈0 = at setpoint, `rt_gain_scale` 0 = converged or metric absent. Under the legacy kinds: target outside the operating range → drive pinned at 0 (T2) |
| Metric flat for thousands of steps | is it finite? | non-finite readings held the EMA (T3) |
| `under_coverage` reads great, coverage is bad | `under_coverage` nan-ness | no sample cleared the ramp floor |
| `tb_resid_clipped` jumps at a stage boundary | that stage's `beta` | T4 |
| `*_worst` == pooled metric | `condition_library_size` | expected unconditionally (§4) |
| `logw_std` rising while VarGrad works | `logw_std_within` | between-condition component dominating — conditional only |
| A controller's series never appear at all | is the sensor in `metric_tracker`, or only the report dict? | T7 — resolving `None` and cold-starting every tick. **Absence of the series is the only tell**; it looks exactly like correct holding |
| `is_ess_frac` not exactly 1.000 at κ=0 | compare to `is_elig_frac` | the draw is not coming from `p` — [`findings.md`](findings.md) `F-004` |
| `probe/alpha_median` wandering, or `fit_*_rate` rising | `alpha_iqr`, `fit_ok` | the probe's parabola fit is failing; the α reading is void regardless of its value |
| `Nonthermal Fraction` 0 while samples plainly look bad | `Excess Energy Nats P99`/`Max` | a tail growing under the bar — the fraction is one threshold, the quantiles are the shape (§3c) |
| The whole non-thermal family absent | `condition_id` in `fwd_stats`; `nonthermal_entropy_per_dim` | no-ops silently pre-bootstrap (no tracker, no anchors) and when the knob is 0/null |
| `Excess Energy Referenced Fraction` well below 1 | `condition_log_z` coverage | those conditions have no Emin(c) yet, so the tail is measured on a subset |

## 8. Simplification candidates

**S1 — delete the MMD family** (T1). ✅ **Approved.** ~200 lines, zero callers,
and class 3 (not a diagnostic anyone reads).

**S2 — staleness counter on `MetricTracker`** (T3). **Approved in principle,
but nothing would read it.** Adding an unconsumed counter is speculative, so
pair it with its consumer or defer: the natural consumer is a controller drive
that **abstains** rather than reading a held EMA — which is the same fix as
drive-liveness reporting (`decisions.md` S2). Do them together or not at all.

**S3 — do NOT assert `beta` uniformity.** *(revised — user: "there may yet be
utility in variable beta")* The problem is not that beta varies; it is that
`tb_resid_clipped` silently changes meaning when it does, while gating anchor
admission. Two fixes that preserve the option:

- **emit `beta` alongside it**, so any reading is interpretable after the fact; or
- **report `tb_resid_clipped / beta`** — dimensionless, bounded to [−1, 1], and
  comparable across beta values by construction.

The second is stronger (it makes the metric *portable*, not just annotated) but
changes an established scale, so it wants a deliberate cutover rather than a
silent redefinition.

**S5 — preflight every controller-named metric against the tracker** (T7). *(new
2026-08-08)* Walk each stage's `balance` / `exit` / `buffer_servo` metric names at
config load and fail on any the tracker cannot produce. This is the same preflight
shape as `module_buffers.md` S2 (which was motivated by a battery dying at the
phase transition), and it closes the only failure mode a controller cannot report
about itself. Cheap, and it converts a silent no-op into a load-time error.

**S4 — ❌ REJECTED: do not suppress the degenerate aliases.** *(user)* The
unified wandb interface is the point — one panel reads `tb_err_worst` on both
conditional and unconditional runs, which is exactly the design intent stated in
`quick_tb_stats`'s docstring. Suppression would break that.

The confusion is real, though, and the fix is **annotation, not suppression**:
emit something that says the family has degenerated on this run (e.g.
`condition_library_size`, or an explicit `metrics/degenerate` flag), so a reader
sees `tb_err_worst == tb_err` is expected rather than having to remember it.

## 9. Open questions

1–2. **Withdrawn, and since dissolved.** Both asked for specific settling-band
values. Settling bands are intensely contextual (problem, T, W, stage) and do not
transfer, so treating one as a recorded fact is the failure mode in
[[findings-must-generalize]]. The general version — *target placement decides
whether a controller side is inert, a guard, or an allocator* — was resolved by
removing the target: `decisions.md` D8, `module_protocol.md` P7. A **ratio** of
two metrics in the same units is the one form of this that has a chance of
transferring, which is why the live controller uses it.

3. **Answered.** `period: 100` is not a lowpass choice — the tracker's job is a
faithful smoothed picture of the live policy, so if the policy shifts fast the
tracker should too. Consumers that need filtering do their own (the controller's
`alpha`, the servo's deadband). So the ~20× separation between tracker and
controller is **deliberate separation of concerns**, not an accident: fast
faithful sensing, per-consumer filtering. Recorded as a warrant.

4. *(new)* Class-2 metrics — the ones with no code consumer that are read by
eye — are not distinguishable from class 3 by any audit. Is a naming or
namespacing convention worth it (`diag/*`), so a future sweep does not propose
deleting them again?

---

*Warrant classes: **derived** · **measured** · **inherited** · **arbitrary** · **contested**.*
