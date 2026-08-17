# The next battery

Argument, in the `docs/PROTOCOL.md` sense: what to test after infrastructure
stabilization, and why those arms rather than others. Revised when the reasoning
changes, not appended to.

**The short version.** The two batteries currently on the cluster cannot answer the
questions they were built to ask, and the reason is not subtle: on the conditional
route four separate control mechanisms are railed or inert; on the unconditional
route the divergence detector cannot fire and the
phase-1→2 handoff is what produced the "spike". Most of what looks like a research
question right now is a defect. **§1 is a fix list, and it comes before any arm
list.** Spending a battery before it is done buys another unreadable battery.

---

## 0. Why the existing batteries are unreadable

### 0.1 The noise floor, measured — and it is enormous

`REPLICATED` — 6 replicate pairs. Scope: T=10, elj/QM9 anchors, SG2, Z'=1,
condition library 5850, `var_conditioning`, **seed 12345 on every run in the
battery** (so there are no designed seed replicates; these are accidental ones).

`qm9anchor_aug14` generations A and B are the *same six arms at the same config
and the same seed*, launched twice. That makes a true replicate pair per arm. The
B/A ratio over the overlapping stage span:

| metric | replicate spread | usable? |
|---|---|---|
| `fwd/logw_std_within` | **×0.30 – ×16.7** | no |
| `fwd/over_coverage` | ×0.28 – ×17.4 | no |
| `fwd/scatter_err` | ×0.29 – ×16.0 | no |
| `eval_test/cond_tb_err` | ×0.66 – ×2.33 | marginal |
| `eval_test/tb_err_worst` | −8 % / +21 % | yes |
| `fwd/tb_err_worst` | **±8 %** | yes — the only tight one |

**Two identical runs differ by 16.7x on `fwd/logw_std_within`.** The entire
cross-arm spread on that family (31 vs 443, 14x) sits *inside* the replicate noise.
Any finding read off the variance/coverage family in this battery is unsupported.

The positive control matters too: within one launch the three hyper-beta arms are
**bit-identical through phase 1** (523/523 exact ticks), diverging only at the stage
boundary where the sensor first acts. The plumbing is correct. The ×17 spread is
generated entirely inside `var_conditioning`.

Independently, on held-out `eval_test/cond_tb_err` the late-run step-to-step scatter
is **2.1–4.3 units on a level of ~28**, at `test_eval_num_samples: 2000`. **At one
seed, no effect below ~5 units is resolvable** — which is most loss-term ablations.

### 0.2 The configs changed mid-battery, twice

`MECHANISM` — read from the committed configs and each run's recorded config.
There are **three generations**, not one:

| gen | launched | `grow_batch_size` | `max_batch_size` | `checkpoint_name` | `vg_detach_center` |
|---|---|---|---|---|---|
| A | 08-14 15:40–18:15 | false | 1000 | phase1_exit | absent |
| B | 08-14 23:16 – 08-15 01:02 | false | 1000 | phase1_exit | absent |
| C/D | 08-15 02:35+ | **true** | **20000** | **null** (cold) | present |

Commit `5a5f17e` (08-14 22:12 EDT) is the A/B → C/D boundary. The same arm name
exists on both sides: `b020_sym` is `bjtn80fy` *and* `58zbv7zw`. **No cross-generation
comparison is valid.** Gen A/B were cancelled by the scheduler for low utilization,
which is what the batch-growth change was made to fix — and it incidentally changed
the VarGrad group size by 4–7x (§2.1), so a cross-generation read is also a repeats
comparison.

Phase-1 exit is not even reproducible across launches for the `bwd80` family: gen A
exits at 6030, gen B at 7850 (+30 %), gen C/D at 5970 — same config, same seed.

### 0.3 The unconditional battery is not code-matched or resource-matched

`OBSERVED`. Four different commits across six `prod0810` arms. `304f209` predates
`2e5f9bb` ("uma arm survival, traj_checkpoint + cpus, energy timing"), so
`9inim617` and `4r351oqm` log **no `energy/*` and no `gpu/*` series at all**. CPU
allocation differs too — `cpus_per_task` 1 on two arms, 8 on the rest, on a
host-bound MLIP where `energy/frac_of_step` is 0.80. That is a first-order
throughput variable confounded with the commit.

Also: `ur4bodzn` and `ma7mitbk` **did not crash**. Both hit the 48 h SLURM wall
exactly (172,800 s window). Their `crashed` state in wandb is a SIGTERM artefact.
Only the three `mipcas_uma` arms failed for cause.

---

## 1. Defects to fix before spending arms

Each of these is a mechanism that is measurably not doing its job. None is a
research question. Ordered by how much they distort a battery.

### 1.1 Conditional — four railed or inert controls

| # | mechanism | measured state | grade |
|---|---|---|---|
| a | **Z-level tether (`level_gap`)** | `level_gap_coeff_rms` reads **exactly the clamp (10.0) on 54–93 % of ticks**; the true per-condition gap is ~710 nats, 71x the clamp. It is a **constant-magnitude, sign-only force applied permanently**, carrying no information about the gap it should close, and it never relaxed in 14,000 steps | `MECHANISM` |
| b | **Balance controller** | `kind: proportional`, `drive: relative`, targets fwd 1.0 / bwd 1.0 — but `fwd/logw_std_within` operates at 40–190, so the fwd drive is 10–46x the bwd drive. Every arm runs at **~0.9 fwd / 0.1 bwd**, not the configured 0.5/0.5, and `Bwd Frac` sits exactly on the 0.1 floor for a quarter to a third of ticks on two arms | `MECHANISM` |
| c | **Anchor buffer** | `anchor_admitted_last_n` median **0.0 for the whole stage, every arm**. The health gate is `tb_resid_clipped < 0.5`, and `frac(\|fwd/tb_resid_clipped\| < 0.5)` is 0.000–0.032 — **the gate is shut ~100 % of the time** | `REPLICATED` |
| d | **Hypergradient LR loop** | `peak_scale *= exp(beta·cos)` every step: a **pure integrator on log(peak_scale) with no restoring term**. The measured `cos` is statistically indistinguishable from zero on **4 of 6 arms** (first- and second-half means have opposite signs), so the LR random-walks. `b005_sym` integrated to `lr_ctrl/scale` = 0.00400, **exactly the lower bound**, and sat at 250x below seed LR for 466 ticks. Two gen-A runs railed the *other* way and detonated to `logw_std_within` 2.3e6 | `MECHANISM` + `OBSERVED` |

(a) is the "Z level tether" suspect, and it is confirmed as a defect — though note
it is at the clamp on *every* arm equally, so it explains the *level* of the stall,
not its *shape*.

**Not on this list: the `var_conditioning` exit bar.** `fwd/logw_std_within < 6.0`
was never met once in 4,348 ticks across 6 arms (measured min 17.1, metric σ 9.9),
and the `naive` stage was never entered by any run. **That is intended —
`var_conditioning` is terminal by design.** The config defect is the opposite of
what it looks like: a vestigial `exit:` block, and a `naive` stage behind it, that
were never meant to fire. They should be deleted, because while they sit there
every reader — the analysis package's R14 check, and this document one draft ago —
reads a live gate that is missing its bar. **Removing them also removes the tidy
explanation for §2.2, which is the more interesting result.**

`config_invariants.exit_bar_is_within_measured_range` now says this at config
load: it carries the measured floor (17.1, σ 9.9) and reports the `< 6.0` bar as
one no run came near, naming all three readings — the bar wants raising, the
block is vestigial and wants deleting, or the run means to reach a regime the
measurement never saw. It reports on `mk_dev.yaml` the moment `protocol:` is
switched to `conditional_vargrad`, which still declares it.

**BASELINE, never ERROR** — deliberately, and the reason is this section: 17.1
was measured with five controls railed, so the configs written to unrail them
are exactly the ones that should be free to aim under it. A check built on
evidence must not block the experiment that supersedes the evidence.

(d) comes with a directly useful measurement: pooling the three >13k arms and
binning held-out `cond_tb_err` by concurrent `lr_fused` gives an **interior
optimum** — worse above *and* below a band of roughly **2e-6 to 2e-5**. `b010_sym`
shows it within-arm: LR falls to 2.1e-6, rises back to 4.6e-6, falls to 1.0e-6, and
held-out quality tracks it. The servo, not the model, may be the ceiling.

Also worth fixing while in there: `prior_buffer_anchor_fraction` is **1.0000** on
every arm and the buffer grows to 204k rows from a frozen ~31.3k anchor set — the
backward branch trains on a static set replicated ~6.5x. That is a memorisation
configuration, and R11 cannot catch it because replay is pinned to 0.

### 1.2 Unconditional — the divergence detector cannot fire

`MECHANISM`. `adaptive_lr.divergence_loss_abs` and `divergence_grad_abs` are both
**1e9** — infinite bars. `lr_ctrl/divergences` is **0** on `ur4bodzn` for all 2,253
ticks *and* on all three `mipcas_uma` arms **through a six-order-of-magnitude loss
blowup**, while the LR warmup ramp climbs 0.107 → 1.00 straight into it. There is
no fast LR cut path on this configuration; `step_probe` is also off, so the servo's
fine sensor does not exist either.

`OBSERVED`, and it corrects an earlier reading: the three `mipcas_uma` arms stopping
at the same step is **a deterministic step-keyed failure, not a wall-clock cut** —
three nodes, three commits, two different QOS budgets, and `4r351oqm` used **3 % of
a 168 h allocation**. Sequence, measured on all three: 150–200 steps after phase-2
entry `grad_norm_pre_clip` explodes (1,173 → 1.7e4), then **~50 steps later** the
batch collapses through the OOM ladder to 125. **The gradient blowup precedes the
OOM** — the batch collapse is a consequence, not a cause.

### 1.3 Both routes — the effective batch, and two dead gates

- **`train.py:2478`** — `accum_target = fused_grad_accum_min_samples if step_type
  == 'fused' else 0`. The effective-batch floor is **fused-only**; a non-fused stage
  takes an unfloored step at whatever the batch has collapsed to.
- **`config_invariants.effective_batch_meets_baseline`** reads the *configured*
  `batch_size`, so every run passes at 1000 while training far below it.
- **Exit-trigger `patience` counted checks, not measurements.** `FIXED`, and the
  mechanism is **the opposite of what this section first recorded** — the streak
  over-counted, it never reset. Corrected by running the real prod0810 exit block
  through the engine (2026-08-16):
  - `protocol/exit_streak_eval_wass_debiased` = 0 at every one of 1,567 ticks is a
    **logging artefact**, not a dead gate. `_exit_tick` skipped `eval/*` terms
    outright, so that series meant "never *judged* here", and was read as "never
    passes". The bar itself was live, checked against fresh metrics inside
    `maybe_advance`.
  - The real defect ran the other way. Every source an exit term reads **persists
    its last value** (tracker EMAs, `gates/*`), so a term checked every 10 steps
    counted one sample many times: measured, a single `bwd/tbc` write at step 100
    carried the streak to **20** over 20 quiet ticks, and one `gates/mle_flat`
    publish cleared its `patience: 5` three ticks later. A patience of 5 on a
    500-step metric meant *50 steps of the same number*.
  - Second defect: `patience` on an `eval/*` term was accepted by `_parse_exit` and
    then **silently discarded**, so it fired on the first clean eval.
  - Fix: `protocol._advance_term` gates the streak on a **fresh write** — fresh pass
    advances, fresh fail resets, no fresh write *holds*. Holding matters: resetting
    on a quiet tick is the fix this section's original reading implies, and it would
    make `patience > 1` genuinely unreachable for any metric slower than the tick.
    `protocol/exit_age_*` now logs staleness beside each streak so the two readings
    can never be confused again.
  - Standing conclusion **unchanged**: of three declared phase-1 exit conditions one
    is a no-op (`bwd/tbc < 2`, satisfied continuously from ~step 100) and only
    `gates/mle_flat` actually gates — but because `mle_flat` is the only *hard* one,
    not because `wass` was dead.
- **`eval_fwd/{logw_std_within, cond_tb_err, tb_err_worst, z_grad_worst}` are
  written twice per eval with different `worst_quantile`**, and `train.py:4275`
  silently overwrites `train.py:4384` through dict-update order. Latent today; a
  landmine.
- **Ambiguous controller metric names.** `health_gate_ceiling_metric` and
  `anchor.ceiling` both resolve a bare `tb_resid_clipped` that is logged under five
  namespaces differing by a factor of 78. Unauditable.
- **Ray probe censoring**: `raycal/t_16` is clamped at ±99 on **50 %** of ticks and
  `t_32` on **83 %**; `alpha_star` is grid-quantised to five values. Against
  `alpha_target: 4.0` the servo dithers `peak_scale` over 0.707–1.297 every 500
  steps — a standing ±30 % LR modulation that never settles.

### 1.4 The OOM cycle is fine; its operating level is not

**Not a re-proposal of the declined OOM-pin fix.** The sawtooth runs as designed —
regrowths ≈ cuts at roughly the expected cadence on every run. What differs is the
*level*:

| run | steps | cycle band (trailing half) | `gpu/util_policy` |
|---|---|---|---|
| `nehzor_elj` control | 36,550 | 4,160 – 6,864 | 94 % |
| `mipcas_elj` control | 138,610 | 1,650 – 4,491 | — |
| `nehzor_uma` | 22,520 | 1,000 – 4,674 | 51–57 % |
| acridine ×3, MACE | 2,450–3,280 | **69 – 188** | **32–35 %** |

**Mostly forced, and mostly deliberate** — `configs/prod0810/make.py` answers this
and I had it as open. `internal_oom_recovery: false` means the whole batch goes
through the MLIP in **one call**, on purpose, so that the energy batch equals the
rollout batch. The consequence is intended: **the MLIP's memory ceiling now sets the
training batch** instead of being hidden by the energy function's own sub-batching.
`traj_checkpoint: true` on MLIP arms is what makes that survivable, at a 1.7x step
cost. So 69–188 is largely the real ceiling at T=60, not a controller pathology.

And the utilization is not a lever: `gpu_util_floor` was **retired** because its
premise was measured false — on `umaperf0812/c_controller` every floor-driven growth
took utilization 52→42 % and samples/sec 57.7→24.3. **Occupancy does not rise with
batch once the MLIP dominates the step.** So 32–34 % cannot be fixed by growing the
batch, and the cluster's 60 % cancellation policy is a standing hazard on MLIP arms
regardless of batch sizing.

What remains genuinely open is narrower and worth stating precisely: **is training
at an effective batch of 69–188 acceptable at all**, and if not, the only levers are
T (memory scales with it), the number of crystals per energy call, and accepting
sub-batching again. That is a battery question (§3.5 B4), not a sizer question —
Phase 6 optimizes throughput and has nothing to say about whether the resulting
gradient is good enough.

---

## 2. Conditional

### 2.1 What the data actually says about the three named suspects

**Z level tether — SUPPORTED, as a defect.** See §1.1(a). But "the Z level loss" is
three distinct mechanisms and ablating them as one knob is unreadable:

| term | leg | what it does | in `var_conditioning` |
|---|---|---|---|
| `z_level` | fwd | Z-only regression of `log_Z(c)` onto live per-condition mean `log w`, detached | **0.0 (off)** |
| `emp_z` | fwd | trains `Z(c)` toward **this batch's** empirical per-condition estimate | **1.0** |
| `level_gap` | bwd | detached-EMA-delta controller pulling `J_B(c)` onto `J_F(c)`, clamped at 10 nats | **1.0, at the clamp** |

`emp_z_persistent` regresses onto the tracker's *accumulated* estimate instead of
the batch's — a purpose-built lower-variance substitute, and the cleanest single-arm
discriminator for "the Z target is too noisy".

**Repeats and huber beta are empirical gradient-quality and variance questions.**
They have to be measured; no amount of reading the loss form answers them. What the
mechanism work below buys is *knowing what the arm actually varies* and *what to
instrument* — it does not substitute for the experiment, and it is not a reason to
demote either axis.

*Repeats.* `condition_grouped_empirical_z` pools **every row sharing a condition in
the batch**, not just the `repeats` tile, so

    group_size ≈ rows / n_distinct_conditions,   n_distinct capped by the library

Measured `fwd/vg_group_size_mean`: **2.37–2.67 in gen A/B** (batch pinned 1000, i.e.
the `repeats: 2` tiling and nothing more) and **10.4–17.3 in gen C/D**. `vg_live_frac
= 1.000` everywhere, so no rows are wasted — the variance question is about group
*size*, not about waste. The operational consequence is that **the arm knob and the
delivered quantity are different things**: setting `repeats` does not set group
size, because batch size and condition concentration also move it. So set `repeats`
and `condition_block_m` as you like, but **declare the target group size and verify
it against `fwd/vg_group_size_mean`**, or the arm label will not describe the arm.
Note `prior_buffer.condition_block_m: 1` in every current arm (mk_dev ships 2), so
the backward condition-blocked draw is off and bwd groups form by chance collision.

Tested and **not** supported: the group-size ramp as the cause of the wobble. Gen
A/B (group pinned ~2.4) and gen C/D (group ramping) overlap completely in held-out
roughness and improving-step fraction. Also not refuted — the comparison has no
power. So the variance question is untouched, not settled.

*Huber beta.* `bwd/tb_resid_clipped` sits at **exactly −10 (sym) and −77 to −80
(bwd80)**, with `frac(|·| < 0.5) = 0.000` on every arm: the backward residual is
**saturated at −beta on essentially every row, all the time**. So the axis is real
but it is not the axis it was named for — `bwd80` is a **×8 gain multiplier on the
whole backward gradient**, not a robustness-knee shift, and it is applied to the
10 % of the batch the balance controller leaves to the backward branch (§1.1b).
That is worth testing on its own terms; it is also worth testing a beta small
enough to actually put the knee inside the residual distribution, which no arm has
yet done. The saturation measurement tells you the current arms bracket only one
side of the question.

**And the axis the battery was actually built for is invisible.** At the only
matched age all seven arms cover, the three `sym` arms — hyper beta 0.05, 0.10,
0.20 — read held-out `cond_tb_err` **28.53 / 28.68 / 28.74**, a 0.7 % spread against
a ×0.66–2.33 noise floor.

### 2.2 The shape of the trajectory

`REPLICATED`. Nearly all improvement is in the first ~5k steps (a 25x reduction),
and it is over by ~13k: in the final third the held-out metric is **flat inside its
own noise** on 2 of 3 long arms. Meanwhile everything on the forward branch gets
*worse* from the moment the stage is entered — `fwd/tb_err_worst`,
`fwd/logw_std_within`, `fwd/z_gap` and `zmatch/delta_worst` all take their minimum
at the first tick of `var_conditioning`, peak 800–950 steps later (= the end of the
1,000-step LR warmup), and never return.

And the two quality families diverge. On `b005_sym`, held-out TB fit is dead flat
(slope +0.13/1k, r = +0.14) while `Reasonable Sample Fraction` (r = +0.94) and
`Mean Sample Energy` (r = −0.92) keep moving linearly. **The distribution is still
in motion; the fit has stopped.**

So the question is not "why is improvement non-monotonic". It is **"why does the
fit stop while the samples keep improving"**, and it is genuinely open. The tidy
answer — the stage never exits, so the run is stuck on an exhausted objective —
is not available: `var_conditioning` is terminal by design and is *supposed* to run
indefinitely. Something else stops the fit while the distribution is demonstrably
still moving (r = ±0.92–0.94 on sample energy and reasonable fraction, against
r = +0.14 on held-out TB fit).

**This is the most interesting unexplained result in either battery**, and it is
what the conditional arms should be pointed at. Standing candidates, none tested:
the tether pinning the level (§1.1a); the backward branch getting 10 % of the batch
(§1.1b); the LR sitting outside the 2e-6–2e-5 band where quality responds (§1.1d);
the backward branch training on a frozen anchor set replicated ~6.5x; or the
VarGrad objective genuinely having a fit ceiling that sample quality does not share
— which would be a real result rather than a defect.

### 2.3 Proposed arms

Pinned across all arms: `grow_batch_size: false`, `max_batch_size = batch_size`, one
warm-start checkpoint, and a **declared seed policy with at least two seeds**.

**Step 0 — do §1.1 first.** Four railed controls is not a battery, it is a bug
queue. Arms spent before (a), (b) and (d) are fixed will re-measure the rails.
Delete the vestigial `exit:` block and `naive` stage at the same time.

**Step 1 — cheap reload probes, not a battery.** Each runs in 45–90 minutes off an
existing checkpoint and kills or confirms a hypothesis that would otherwise cost
arms:

| probe | change | reads out |
|---|---|---|
| LR band | pin `lr_fused` at 5e-7 / 4e-6 / 2e-5, `lr_sensor` removed | is the servo the ceiling? (§1.1d) |
| tether | `level_gap` 1.0 vs 0.0 | is it fighting, or doing its job? (§1.1a) |
| balance | `targets.fwd` inside the metric's real 40–190 range | does `Bwd Frac` leave the floor? (§1.1b) |
| condition draw | `weighted_condition_sampling: false` | is batch concentration upstream or downstream of the variance blow-up? |

**Step 2 — the battery.** Three axes, all genuinely open, all about gradient
quality and variance:

- **Z level (4 arms)** — control / `level_gap 0` / `emp_z 0 + emp_z_persistent 1` /
  both off. Separates "the tether hurts" from "the tether's *target* is noisy".
- **Group size (3 arms)** — target ∈ {2, ~6, ~12} at *fixed* batch, reached by
  `repeats` and `condition_block_m` together, **verified against
  `fwd/vg_group_size_mean`**. This is the VarGrad estimator-variance axis.
- **Huber beta (3 arms)** — the current `sym`/`bwd80` pair reads a ×8 gain change
  under full saturation, so add a third arm with beta small enough to put the knee
  inside the residual distribution. That brackets the question on both sides
  instead of one.

**Do not cross them.** Run one, then the next on the winner. Target the 6k–13k
window, where arms actually differ.

**How the noise floor changes the design, without vetoing anything.** §0.1 is a
constraint on *how* to run these, not a reason to skip them. Three consequences:
read the axes on `fwd/tb_err_worst` (±8 % replicate spread) and
`eval_test/tb_err_worst` (−8/+21 %), **not** on the `logw_std_within` /
`over_coverage` / `scatter_err` family (×0.3–×16.7, unusable); budget **two seeds
minimum**, since a one-seed arm cannot clear its own floor; and prefer three arms
at two seeds over six arms at one. If an axis's effect turns out to be smaller than
~5 units on the held-out metric, that is itself the answer — but it has to be
measured, not inferred from the loss form.

### 2.4 What is healthy, and should be left alone

`REPLICATED` — held-out generalization is the best thing in the battery. `eval_test
/ eval_fwd` is **1.04–1.17** on every arm and every metric; the pooled sample-quality
gap is 4–23 % relative, always in the same direction. **The conditioner generalizes.**
Do not spend arms here.

One instrumentation caveat that looks like a generalization signal and is not:
`eval_fwd/scatter_err` spikes to 690 while `eval_test/scatter_err` reads 43 at the
same tick, but the *quantile*-based keys show no such gap. The train stream pools
10,000 samples over ~4,470 conditions and the held-out stream 2,000 over ~570, so
the RMS family hits the tail 8x more often on the train side. **Do not read
`eval_fwd` minus `eval_test` on the RMS family.** Relatedly, `Cond Reasonable Worst`
reads **exactly 0.000 on every arm, both streams, every tick** — n_c ≈ 2.2 makes it
a dead sensor.

---

## 3. Unconditional — the paper refresh

### 3.1 What actually happened to `nehzor_uma`

`OBSERVED`, single run, no seed replicate. Scope: nehzor SG14 Z'=1, uma, T=60,
unconditional, seed 12345.

**The "big-ish spike" is the phase-1 → phase-2 handoff at step 15,670**, not a
mid-training event. Five things change at once at that boundary: `train_mode` goes
bwd→fused, the LR envelope is **cut 9.8x and warmup re-armed for 1,000 steps**, the
batch **resets to 1,000**, `on_enter` fires `rebuild_prior_by_churn` +
`bootstrap_z`, and `train_step_time` goes 1.86 s → 99–107 s for four windows.

`bootstrap_z` lifts `bwd/log_Z_learned` to +11.40 while the forward policy's own
estimate sits at −7.2 — **an 18.6-nat gap between the bootstrapped Z and what the
policy supports**. As the LR ramps back to full (peak at step 16,700), every
Z-anchored metric peaks 20–180 steps later: `log_Z` dives 11.82 nats,
`fwd/jensen_z` swings 29.4 nats, `zmatch/delta_worst` goes 3.51x. Three of those
never recover to their pre-spike values in the remaining 5,510 steps.

It is **not** a tripwire event (`divergences` = 0), **not** OOM (batch never below
1,000), **not** non-finite, and **not** an energy failure (`ms_per_sample` flat).
It is a Z excursion at a stage handoff. `CONJECTURE` for the mechanism: bootstrap
set Z above what the policy could support, and the TB gradient dragged it back down
through the policy — the log-Z dive *is* the under-coverage spike.

**The `mipcas_uma` detonation looks like the same event at higher amplitude** —
identical shape, transition → grad blowup → Z dive → coverage blowup — except that
nothing could cut the LR (§1.2) and it did not recover.

### 3.2 Why UMA is slow now — it is structural, and it is not a regression

`MECHANISM`. `energy/frac_of_step = 0.797` and the identity
`energy_seconds ÷ ms_per_sample` = 1,858 ≈ `Batch Size` 1,853: **the MLIP is called
on the full batch every training step.** The 2026-05/06 UMA runs cannot have been
doing that — 3,000 samples × 8.55 ms is 25.6 s against an observed 4.22 s/step. Their
logged `Fwd to Bwd Ratio` of 0.010–0.028 says they scored the MLIP on **~30–85
crystals per step** and trained backward off a **static 5,694-row buffer whose
energies were precomputed** (`Buffer Length` is literally constant for all 88,220
steps).

| arm | s/step | batch | samples/s | `ms_per_sample` | GPU util |
|---|---|---|---|---|---|
| `nehzor_uma` | **19.63** | 1,853 | **92** | 8.55 | 51–57 % |
| `nehzor_elj` | 5.32 | 6,864 | 1,254 | 0.177 | 94 % |
| old `nehzor_uma` (T=100) | 4.22 | 3,000 | ~711 | — | — |

So: **~7.7x fewer samples/s, ~250x more MLIP calls/s.** "UMA got slower" is correct
per step, and the cause is that the protocol now *uses* the MLIP rather than mostly
avoiding it. That is a design change, not a bug — but it means the old runs are not
a throughput baseline, and any plan that assumes old-run rates is wrong by ~8x.

Per-step, `nehzor_uma` is actually the **steepest descent in the battery** (−7.8 %/1k
on `fwd/tb_err_worst` vs −1.0 % for `nehzor_elj`); per *hour* it is only ~2x, and it
is descending from a level 5x higher. It is nowhere near equilibrium.

The uncomfortable comparison: at ~22,000 steps the 2026-05 run reached **−95.4
kJ/mol and 0.959 reasonable fraction**; this one is at **−40.0 and 0.804**. Confound:
**T=100 then, T=60 now**, and the old runs predate the periodic-scoring fix, so
their backward scoring at wrapped dims carried a large fictitious residual. Both
caveats are real and neither obviously explains a 2.4x sample-quality gap at matched
step count. `fwd/z_gap` is **32.1 on uma against 8.9 / 5.4 on the elj arms** — a
32-nat Z/policy gap is not "slow", it is a policy that has not found the target.

### 3.3 The 4 × 7-day runs

Three preconditions, and only one was on the original list:

1. **§1.2 fixed** — a run that cannot cut its LR through a six-order-of-magnitude
   blowup should not be given 7 days.
2. **The handoff de-risked** — §3.1 is the single event that damaged this run and
   killed the `mipcas_uma` arms. The 500-step reload probe (re-enter equilibration
   with `bootstrap_z` removed, or Z seeded to the policy's `jensen_z`) costs ~2.7 h
   and is the highest-value pre-flight in this document.
3. **Phase 6 landed** — §1.4.

Then: **one short arm per backend as pre-flight**, ~2 h each, whose only job is to
show the batch band and effective batch are sane and the handoff is clean. And
**compare at matched samples, not matched steps** — `analysis --matched` matches
step span, which is not the same thing once batch varies 20x.

A fourth item, cheap and worth doing: `nehzor_uma`'s phase 1 ran **15,670 steps
against `nehzor_elj`'s 7,440** on the same molecule, and exited with
`mle_gate_rate` −0.152 against elj's −0.046 — declared "flat" while still descending
3.3x faster. Combined with §1.3's finding that only one of three phase-1 exit
conditions actually gates, the warm start handed to phase 2 may simply be
undercooked.

### 3.4 Acridines are not a tack-on

`OBSERVED` — 3 runs, T=60, acridine SG9/SG14, Z'=1 and Z'=2, MACE, 2,450–3,280 steps.

All three crashed **still in `train_prior`**, cycling at batch 69–188, with 5–8 OOM
events and GPU utilization 32–35 %. A 2-day run at that rate produces nothing.
Note `sg14_zp1` behaves the same as the Z'=2 arms, so this is not obviously a Z'>1
problem — "limited Z'>1 experience" is probably the wrong diagnosis to reach for
first. They need their own shakeout before they are worth a production slot.

### 3.5 The unconditional battery — arms

**The allocation principle: run the battery on ELJ, and spend MLIP arms only on
MLIP-specific questions.** An ELJ step is 3.7–6.6x cheaper (2.96–5.32 s against
19.63 s), and `nehzor_elj` and `mipcas_elj` are the two healthiest runs in the whole
set — stable batch, 94 % utilization, log Z rising monotonically through the same
stage boundary that wrecked the UMA arms. Exactly three questions are genuinely
MLIP-specific: the handoff, the divergence bars, and the memory-ceiling batch.
Everything else — grad clip, LR aggressiveness, DPLR, `t_scale`, T, auxiliary
losses, buffer churn — is not, and paying UMA rates to answer it is waste.

Throughput used for costing, from the measured phase-2 rates: `nehzor_uma` 188
steps/h, `mipcas_uma` ~440, `nehzor_elj` 693, `mipcas_elj` 1,141.

#### Tier A — gates the production runs. 10 runs, 89 GPU-h (~3.7 days), protects ~35.

**A1–A5 · The handoff** (UMA, reloaded from `ur4bodzn`'s `phase1_exit`, ~2,000 steps
past re-entry so the dive and its recovery both land; ~10.6 h/arm).

| arm | change | reads out |
|---|---|---|
| A1 | control | does the 11.8-nat dive reproduce off a reload? |
| A2 | LR warmup **not** re-armed — hold the phase-1 LR across the boundary | the excursion peaks 20–180 steps after the ramp hits 1.0; does removing the ramp remove the peak? |
| A3 | batch **not** reset to 1,000 at the boundary | is the batch reset compounding it? |
| A4 | z-calibration burst suppressed at entry | 110 z-cal steps land inside the first 10-step window and take `train_step_time` 1.86 → 107 s |
| A5 | `gradient_norm_clip` as a **fixed normalizer** instead of `auto` | the old well-behaved runs clipped at a fixed 0.1, binding every step; `grad_norm_pre_clip` hit 4,973 in the excursion against a phase-1 median of 75 |

A correction to an earlier draft of these arms: **`bootstrap_z` already seeds from
`eval_fwd/jensen_z`** on the unconditional route — `protocol.py:1480` fills
`flow_model.scalar` with it directly. So "seed Z to the policy instead of the
buffer" is not an available arm; it is what already happens. The 18.6-nat gap opened
in the ~390 steps *after* the bootstrap, not at it, which points the arms at the
ramp (A2) rather than at the seed. Worth an **A6** if capacity allows: **no
`bootstrap_z` at all**, letting Z find its own level under the new objective.

**A7–A10 · The divergence bars** (the detonation is a gift — it reproduces at the
same step ±150 across three nodes, three commits and two QOS budgets, so it is a
test fixture). Reloaded from `mipcas_uma`'s `phase1_exit` at 8,740, run to ~11,400;
~6 h/arm.

| arm | change | reads out |
|---|---|---|
| A7 | bars at 1e9 (control) | confirm the detonation reproduces on a reload |
| A8 | `divergence_grad_abs` at a bar drawn from healthy phase-2 `grad_norm_pre_clip` (e.g. p99.9 on `nehzor_elj`) | does the cut fire, and does it fire in the ~50-step window *before* the OOM collapse? |
| A9 | a tighter bar | how much margin is there between firing early and firing spuriously? |
| A10 | **the chosen bar on healthy `nehzor_elj`, 5,000 steps** (ELJ, ~7 h) | **the true-negative test** — zero spurious cuts, or the bar is unusable |

A10 is not optional. A detector validated only on a positive is a detector with an
unmeasured false-positive rate, and this one can cut the learning rate of a 7-day
run.

#### Tier B — informs the production config. ELJ, 24 h/arm.

**B1–B4 · Effective batch at fixed wall clock.** Batch pinned (growth off) at 1k /
4k / 16k, plus one arm at 1k with the accumulation floor **extended past fused
steps**. Compared at equal *wall clock*, not equal steps — which is the question a
7-day run actually poses, and one the sizer cannot answer: Phase 6 maximizes
opt-step throughput and has nothing to say about whether the resulting gradient is
good enough. B4 tests whether extending the fused-only floor (§1.3) actually buys
quality back at low batch, which is what decides whether the MLIP arms' 69–188
regime is survivable (§1.4).

**B5–B7 · Grad clipping.** `auto` (control, 454.4 plus the p99 adaptive guard) /
fixed normalizer that binds every step / `auto` with the adaptive guard tightened.
The old runs used the fixed normalizer and were well-behaved; the new ones let a
22x larger gradient through. Pick the winner here at ELJ rates, then carry it into
A5 to test the interaction that actually matters.

**B8–B10 · LR aggressiveness.** **Prerequisite, not an arm: fix the ray probe
first.** `raycal/t_16` is censored at ±99 on 50 % of ticks and `t_32` on 83 %, and
`alpha_star` is quantised to five grid values — so the servo dithers `peak_scale`
over 0.707–1.297 every 500 steps and any `alpha_star` above ~8 is read off an
unresolved bracket. Extend the ladder and log uncensored `t` values, then run:
`alpha_target` 4 (control) / 8 / **servo off at a well-chosen fixed LR**. The third
arm is the important one — on the conditional route the servo walked the LR out of
the band where quality responded, and whether a servo beats a good constant is an
open question on both routes.

#### Tier C — the science. ELJ, and can run alongside production.

**C1–C2 · DPLR.** `dplr_rank: 6` (control) vs `dplr_rank: 0`, which the config
documents as **"exactly original diagonal SDE"** — an exact control, not an
approximate one. DPLR has been on in every production run and has never had an off
arm. This is the strongest of the open-ended axes and it is two arms.

**C3–C5 · T.** 40 / 60 / 100 at fixed wall clock. **`eval_T` is not an independent
knob** — it must equal `integrator.T` and this is enforced at config load, because
drift and variance are learned at one `dt` and P_B's reference bridge is written in
accumulated-variance time, so a different `dt` integrates a different SDE. So the
"T/eval_T speed–quality–variance trade-off" collapses to a T-only axis, which is
cleaner than it looked. T is also the memory lever behind §1.4, and T=100 is what
the pre-2026-06 runs used, so this axis doubles as re-establishing comparability.

**C6–C8 · `t_scale` scheduling.** `t_scale_ratio` / `_power` with
`_preserve_budget: true`, currently off. **Checkpoint-reconfigurable**
(`checkpointing.py:508`), so these are three branches off one warm start rather than
three runs — the cheapest axis in the document, and it removes warm-start variance
from the comparison for free.

**Deferred, with the specifics recorded so they are actionable later.** Auxiliary
losses: on the unconditional route `vg_by_condition` is meaningless (every row is
one group, so the loss reads identically zero), but fwd `vg_lb` with `repeats: 2`
gives genuine cross-terminal VarGrad, bwd `tbc` needs `repeats > 1`, and `subtb`
needs `model.full_flow: true`. Buffer temperature and churn: real, but the replay
buffer's three size knobs are three handles on **one** steady state
(`occupancy = churn_rate × mean_residence_steps`), so an arm that moves them
independently changes occupancy and reuse together and cannot be read.

#### What a 7-day run actually buys, and why Tier A pays for itself

At the measured phase-2 rate a 7-day UMA run is roughly **10 h of phase 1 plus
~30,000 phase-2 steps**. `nehzor_uma` was still 32 nats from a converged `z_gap` at
22,500 steps against the ELJ arms' 5–9, so 30,000 is not obviously enough even
without incident. And the handoff cost it ~1,300 steps of excursion plus a recovery
that had not finished 5,500 steps later — **on the order of a day of the seven,
spent undoing a transition.** Tier A costs 89 GPU-h (~3.7 days) and is aimed
squarely at that.

Two further production settings that are not arms but should be decided:
**warm-start all four arms from a validated `phase1_exit`** (`prod0810` currently
leaves `checkpoint_name` null on some arms, which spends ~10 h of each run
re-deriving a prior and makes the four runs non-comparable), and **fix T across all
four plus the acridines** — since `eval_T` is pinned to it, a T that differs by arm
means the arms are not on one ruler.

---

## 4. Sequencing and totals

Anchoring values, read off the live `prod0810` runs: **T = 60, eval_T = 60,
`dplr_rank` = 6, `t_scale` = 0.05 with `t_scale_ratio` null, eval/figs period
500/1000, `max_step_seconds` 300 on MLIP arms and 60 on ELJ.** The conditional route
runs T = 10.

### 4.1 Measured throughput — everything below is costed from these

Not estimates. Phase-2 medians from the runs named in §0.3 and §3.2, and wall-clock
from the conditional battery's own `_runtime`.

| route | config | s/step | steps/h |
|---|---|---|---|
| conditional, elj, T=10 | batch 1,000 | 0.31–0.51 | **7,100–11,500** |
| conditional, elj, T=10 | batch 20,000 | 4.48–6.52 | **550–800** |
| conditional, elj, T=10 | batch 4,000 | *not measured* | ~2,600 (interpolated) |
| uncond, nehzor, **uma**, T=60 | phase 2 | 19.63 | **188** |
| uncond, mipcas, **uma**, T=60 | phase 2 | 4.18 | **440** |
| uncond, nehzor, elj, T=60 | phase 2 | 5.32 | **693** |
| uncond, mipcas, elj, T=60 | — | 2.96 | **1,141** |
| uncond, uma, phase 1 | — | — | **1,438** |

The conditional route is 20x cheaper per step at batch 1,000 than at 20,000, and
that single fact decides most of the budget below — see stage 1.

### 4.2 The runs

| stage | runs | what | route / reload point | steps | batch | h/run | GPU-h | gates? |
|---|---:|---|---|---:|---:|---:|---:|:---:|
| **0** | — | fix §1.1, §1.2, §1.3; delete the vestigial `exit:`/`naive`; fix the ray probe censoring | code only | — | — | — | **0** | **yes** |
| **1** | 2 | **utilization pre-check** — cheapest pinned batch that clears the cluster's 60 % bar | conditional, batch 4,000 and 8,000 | — | 4k / 8k | 2.0 | **4** | no |
| **2** | 7 | conditional reload probes: 1 control + 3 LR band + tether + balance + condition draw | conditional, from a `var_conditioned`-age ckpt | 2,000 | 20k | 3.3 | **23** | no |
| **3a** | 6 | **A1–A6 handoff** | `ur4bodzn` phase1_exit @15,670, **uma** | 2,000 | live | 10.6 | **64** | **yes** |
| **3b** | 3 | **A7–A9 divergence bars** | `mipcas_uma` phase1_exit @8,740, **uma** | 2,660 | live | 6.0 | **18** | **yes** |
| **3c** | 1 | **A10 false-positive test** | `nehzor_elj`, healthy | 5,000 | live | 7.2 | **7** | **yes** |
| **4a** | 4 | **B1–B4 effective batch** at fixed wall clock (1k / 4k / 16k / 1k+floor) | uncond elj | — | pinned | 24 | **96** | no |
| **4b** | 3 | **B5–B7 grad clip** (`auto` / fixed normalizer / tightened guard) | uncond elj | — | live | 24 | **72** | no |
| **4c** | 3 | **B8–B10 LR** (`alpha_target` 4 / 8 / servo off at fixed LR) | uncond elj, *after* the probe fix | — | live | 24 | **72** | no |
| **5a** | 2 | **acridine shakeout** — does it hold a sane batch and leave `train_prior`? | acridine, mace | — | live | 12 | **24** | gates 5c |
| **5b** | 4 | **production**, warm-started, fixed T | mipcas/nehzor × elj/uma | — | live | 168 | **672** | — |
| **5c** | 3 | **acridine production** | acridine sg9_zp2 / sg14_zp2 / sg14_zp1 | — | live | 48 | **144** | — |
| **6a** | 2 | **C1–C2 DPLR** (`dplr_rank` 6 vs 0) | uncond elj | — | live | 24 | **48** | no |
| **6b** | 3 | **C3–C5 T** (40 / 60 / 100) at fixed wall clock | uncond elj | — | live | 24 | **72** | no |
| **6c** | 3 | **C6–C8 `t_scale`** — branches off one warm start | uncond elj | — | live | 24 | **72** | no |
| **6d** | 20 | **conditional battery** — 10 arms × 2 seeds, Z level (4) → group size (3) → huber beta (3), run in sequence | conditional, warm from `phase1_exit` | 8,000 | 20k | 13.3 | **267** | no |

### 4.3 Totals, and the number that matters

| bundle | GPU-h | GPU-days |
|---|---:|---:|
| **gates only** (stages 0–3) | 116 | **~5** |
| production (stage 5) | 840 | ~35 |
| gates + production | 956 | **~40** |
| Tier B (stage 4) | 240 | ~10 |
| everything (stages 0–6) | 1,655 | **~69** |

**The headline: the gating work costs ~5 GPU-days on top of a ~35-day production
plan — about 13 %** — and it is aimed at a handoff that cost `nehzor_uma` on the
order of a day of its seven and killed all three `mipcas_uma` arms outright. That
is the easiest trade in this document.

**Stage 1 is the highest-leverage 4 GPU-hours here.** Gen A/B at batch 1,000 were
cancelled by the scheduler for low utilization; gen C/D at 20,000 survived 34.5 h;
**nothing in between has been measured.** If batch 4,000 clears the 60 % bar, stages
2 and 6d get ~4.5x cheaper — 23 → 5 GPU-h and 267 → 62 GPU-h, a saving of ~9
GPU-days for two 2-hour runs. The 2-hour duration is not padding: `gpu_util_policy`
averages over a 7,200 s window, so a shorter run cannot fill the number the cluster
actually judges.

**What I would cut, in order.** 4c (LR) first — it needs the ray-probe fix landed to
be readable at all, so it is the axis most likely to be wasted. Then 6b (T) and 6c
(`t_scale`), which are genuine science but change nothing about the production runs
already in flight. I would not cut 4a or 4b: both inform the production config, both
are cheap ELJ time, and 4a (B4 specifically) is the only thing that tells you whether
the MLIP arms' 69–188 batch regime is survivable at all.

**What must not be run concurrently with itself.** 6d's three axes are sequential by
design (§2.3) — run Z level, then group size on its winner, then huber beta. The 20
runs are a total, not a fan-out.

**One reversal from an earlier draft, and the reason.** I had put T and `t_scale` on
the analytic toy first. For `t_scale` that still holds — it is checkpoint-reconfigurable
and `latent_gaussian` has a closed-form log Z to score a schedule against. For **T it
does not**: `eval_T` is pinned equal to `integrator.T` at config load, so the
"evaluate cheaply at a different T" half of that trade-off does not exist. What
remains is T as a cost, memory and comparability lever, and those only mean anything
on the real problem. T moves to ELJ.

---

## 5. Battery hygiene

Adopted because §0 cost both current batteries their interpretability:

1. **Freeze arm configs before launch; never regenerate mid-battery.** If arms must
   change, the battery gets a new tag.
2. **Stamp provenance into the run** — a content hash of the resolved config in the
   name or tag would have made §0.2 visible in the run list instead of requiring a
   `git diff` against launch timestamps. Same for the commit: §0.3 is four commits
   across six arms.
3. **Budget one replicate pair per battery, deliberately.** §0.1's noise floor
   exists only because a scheduler failure accidentally produced one. Two seeds on
   three arms beats one seed on six.
4. **Pin the batch in any battery whose metric depends on it** — for the conditional
   VarGrad route that is every metric, via §2.1.
5. **Read the actuator series before the outcome series.** `level_gap_coeff_rms`,
   `prop_drive_*`, `lr_ctrl/scale`, `anchor_admitted_last_n`, `vg_group_size_mean`,
   `exit_streak_*` — most of the defects in §1 are visible in one of these and in
   none of the topline metrics.
6. **Delete config that is not meant to fire.** The `var_conditioning` `exit:`
   block and the `naive` stage behind it were never intended to trigger, and while
   they sit in the file every reader — including the analysis package's R14 check —
   spends effort on a gate that is not one. A stage that is terminal by design
   should say so.
7. **A bar inside its own metric's noise is not a bar**, and a bar checked more
   often than its metric is written can never be met. The phase-1
   `wass_debiased` gate (checked 50x more often than written) would be caught by
   asserting check-cadence against write-cadence at config load.
