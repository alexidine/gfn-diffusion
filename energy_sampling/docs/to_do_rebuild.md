# To-do: rebuild

Unconditional route. Started 2026-08-05; batch-design part merged in 2026-08-06.
**Decisions ratified 2026-08-06** — see [`decisions.md`](decisions.md) Part 3 for
the ruling set and Part 1 for what is still open.

**A. LR controller** · **B. Batch construction** · **C+ partly written** — per
`decisions.md` D1(b) this is the *whole* plan, so the remaining open areas get a
part each. **The balance controller is done and is not a part here**: it was
settled and built in its own session (2026-08-06) and is written up where it
lives, as `module_protocol.md` **P7** (`kind: ratio`), with the rulings in
`decisions.md` D8/D11. Still to write: the anchor health gate, the smoke config,
and memory retirement.

## How this is meant to be used

The plan has three phases, and the doc is ordered to serve them:

1. **Decide** — work through each area, make the redesign calls. Parts A and B.
2. **Revert + baseline** — back to something close to the best run, re-run a
   baseline.
3. **Implement + test** — build and measure the new work.

**§0 comes before all three.** Every measurement in it runs read-only against
runs already on disk, several of them re-read the baseline, and each one gates a
decision in A or B. Doing them before the revert costs nothing and stops the
decide phase from being made on assumptions.

---

## 0. Read-only first — the gate battery

**What this is for** *(added 2026-08-06 — the table alone didn't say).* Six
numbers that are **already sitting in runs on disk**, each of which the design
below would otherwise have to *assume*. Nothing here is an experiment; it is
reading scalars that were logged months ago. In plain terms:

| # | The question it answers |
|---|---|
| 1 | Does the new LR sensor see the cliff coming, or only after? Sets `α_target` and the growth rate |
| 2 | ~~Is our learned `log Z` at the truth?~~ **Retired as a question** — Z-currency is an assumed invariant (§B8), enforced by `z_calibration` at the same 0.5 bar. All that remains is confirming the trace never breached it |
| 3 | How spread out are the residuals? The variance gain from prioritised sampling is `1 + CV²`, computable **before building it** — if it's ~1.2 the entire §B5 build is not worth doing |
| 4 | Is replay memorising *right now*? (`P3`, never read) |
| 5 | How fast does a buffer row get corrected? Sets `τ` (§B7a) instead of hand-picking it |
| 6 | Is the new sensor's reading even *stable*? Voids it independently of everything else |

Rows 3 and 6 are kill-switches: either one coming back wrong deletes a large
chunk of work below. Row 2 invalidates the others if it comes back wrong, so it
goes first.

No new compute. All of it against existing runs.

| Measurement | Reads | Gates | If it fails |
|---|---|---|---|
| **`α*` probe across aug02** | read-only probe, existing ladder | ~~the entire LR servo~~ → **`α_target` and the growth rate** (§A6, demoted to instrumentation by D3) | the servo still gets built; what changes is that it needs a separate hard ceiling, because α\* has no lead time on the cliff |
| ~~`log_Z_learned` vs `z_emp` vs `z_gap`~~ → **confirm `fwd/tb_resid_clipped < 0.5` held** | one trace | nothing — **demoted from experiment to invariant check** (§B8, user 2026-08-06). Z-currency is *assumed*, and `z_calibration` enforces the same 0.5 bar | if the invariant did **not** hold for a stretch, readings from that stretch are off-origin and `δ₊` splits in the wrong place there |
| ✅ **`1 + CV²(\|δ\|)` — MEASURED 2026-08-06** | 22 local buffer snapshots | the prioritised-sampling build | **replay ≈ 1.5–2.0 → build. prior ≈ 1.06–1.13 → drop it on bwd.** See §0a |
| **weighted replay err ÷ fwd err** | existing metrics + draw weights | the memorisation servo, and `τ` (§B7) | this is `P3`, still unread, and it is the master population diagnostic |
| **λ = `log(birth_loss/ema_loss)/(step − birth_step)`** | three stored fields | whether `λτ > 1` (§B7) | one division from `replay_buffer_live_delta_mean`, already logged |
| **windowed-median stability of `α*`, per-probe dispersion, downward-opening fit rate** | ships with the probe, free | voids the `α*` sensor independently of lead time (§A3a.2, §A3a.3) | — |

---

## 0a. Result: `1 + CV²(|δ|)`, measured 2026-08-06

Read off `ema_loss` (the per-row EMA of `|resid|`) and `birth_loss` (`|resid|` at
admission) in 22 buffer sidecars on disk — `ctrl_aug03`, `size_aug03`,
`local_aug02`, `BASE*`, and a toy run. **measured.**

| Buffer | `CV` | `1 + CV²` | Verdict |
|---|---|---|---|
| **replay** | 0.63 – 1.05 | **1.4 – 2.1** (median ~1.7) | above the 1.2 drop bar, far below 5 — **real but modest** |
| **prior** | 0.23 – 0.36 | **1.05 – 1.13** | **below the drop bar** |

Robust to the proxy: `ema_loss` and `birth_loss` disagree on individual arms but
both land replay at ~1.4–2.0. Two caveats, both pushing the same way — `ema_loss`
is smoothed, so it understates instantaneous spread, and `‖g‖` is omitted (§B5's
known omission), which would *raise* the bound if it correlates with `|δ|`. So
treat ~1.5–2 as a floor.

### What this decides

**Replay: build it.** ~1.7× is right where §B9 guessed (~2×) and inside the
literature's 1.2–2× band. Worth doing, and it confirms §B9's honesty about where
the payoff lives: **not in step count**. The arm that matters is §B10's LR
ceiling, not the κ ladder's convergence rate.

**Prior/bwd: §B5a is dead, and D28 is why it stays dead.** The prior buffer's
residuals are strikingly homogeneous — `std ≈ 4.2` against `mean ≈ 15–18`, i.e.
`CV² ≈ 0.06`, so prioritisation offers **~6%** variance reduction there. §0
anticipated this case and said "drop the sampling work, keep only §B5a's ordering
argument" — but **D28 dissolved the ordering argument**: within-tail ordering is
lost to *Huber*, and the fix is to run quadratic, not to prioritise. Uniform draw
+ quadratic already gives `Φ ∝ δ`.

> So on `bwd`, prioritisation has **no variance case (6%) and no ordering case
> (already linear once quadratic)**. Nothing is left to justify it. §B5a comes
> out of the plan; §C step 4 and §B10's bwd-priority arm go with it.

This is the §0 battery doing exactly its job: one afternoon reading files on
disk deleted a build step, an implementation stage, and an experimental arm.

---

## 0d. Checkpoint reuse across T — loadable is NOT interchangeable

*(established 2026-08-08; the constraint every future battery is sized by.)*

`integrator.T` is **not** in `get_problem_definition` — identity is
`energy_function`, `energy_config`, `prior_path`, `space_groups`, `z_primes`,
`mol_cond`, `temp_cond`, `vec_cond`. So a T=10 checkpoint is hash-compatible
with a T=60 run and `find_shared_prior` will happily match it.

> **It will still be a bad warm start** *(user, 2026-08-08)*. The policy learned
> per-step transition distributions for a specific step count, and `log P_F` /
> `log P_B` decompose over a different number of factors at a different T. The
> checkpoint loads; the knowledge does not transfer. A T-mismatched warm start
> gives the trunk a head start and nothing more — **phase 1 has to genuinely
> re-converge, not exit in ~600 steps.**

So per-T sources are mandatory, and the warm-start saving (1–12 h/run) is only
realised **within** a T.

### The correct warm-start pattern

```yaml
checkpoint_name: <same-T *_phase1_exit.pt or *_best.pt>
load_weights_only: true
continue_from_checkpoint: false
reuse_prior: false          # MUST stay false
```

**Never `reuse_prior: true` or `prior_model_name`** — those trigger
`skip_if: prior_loaded`, which skips phase 1 **and leaves the policy random**
(the prior model is sampling-only). `configs/aug02/generate_configs.py` carries
the same warning.

### ⚠ The slug differs between LOCAL and CLUSTER — `prior_path` is in the identity

`get_problem_definition` stores `prior_path` as a **raw string**, so the same
problem hashes differently depending on where the dataset lives:

```
local    D:\crystal_datasets\...\mipcas_sg2_zp1_elj_prior_dataset.pt   ->  573c92
cluster  /scratch/mk8347/data/...\mipcas_sg2_zp1_elj_prior_dataset.pt  ->  2df5a5
```

`prior_path` is the **only** differing identity field. `load_weights_only` calls
`assert_problem_match`, which **hard-raises** — so a config that hardcodes the
local slug dies in the first seconds on the cluster.

**Rule for any cluster generator: use the cluster `prior_path` verbatim, GLOB
for the checkpoint rather than hardcoding a slug, and assert the file exists
before submitting.** The table below lists run prefixes, which are
filesystem-independent; the slug suffix must be resolved on the machine.

### Per-T sources (prefixes; cluster slug is `…-T2.5-2df5a5`)

| T | prefix | `tb_err` | `r2` | EffDim | verdict |
|---|---|---|---|---|---|
| 10 | `tw_july31_tw_T10_lr64` | — | — | 9.37 | usable |
| 25 | **`aug02_a2_T25_lr2_tight`** | **3.89** | **0.95** | 5.85 | **best available** |
| 60 | **`aug02_a2_T60_lr2_tight`** | **4.82** | **0.99** | 5.91 | **excellent** |
| 100 | `uncond_july28_prop_T100` | 28.31 | 0.43 | 5.98 | poor fit, healthy diversity |

**⚠ High T has no good warm start, and historically collapses.** The
best-*fitting* T=100 checkpoint (`stab_july21c_elj_h512x6_T100_lr1.6e-5`,
`tb_err` 8.93, `r2` 0.87) has **EffDim 2.21 — collapsed**. Both T=80 checkpoints
are collapsed too (2.40, 2.95). So a T=100 baseline must either budget real
phase-1 time or start from a poorly-fit prior, and mode collapse is the expected
failure mode there rather than a surprise.

**Consequence for short experiment arms:** they should all share ONE T so they
can share one checkpoint. **T=25 is the right choice** — the best checkpoint by a
wide margin, the regime aug02 characterised, and 3400 steps/h means a 4 h arm is
~13.6k steps (an order of magnitude more than the 1500-step local arms that
resolved nothing).

## 0b. Local shakedown plan — 2026-08-07

One day, local, T=10 for speed. Ordered so the cheap kill-gates fire before the
expensive runs.

### Run 0 — null regression *(~20 min, do it first)*

**No `step_probe` block, no `prioritise` block.** Today's changes touched
`manage_replay_buffer`, `get_gfn_backward_loss`, `replay_train_step`,
`draw_replay_sample` and `train_step`; all are gated, but *gated* is a claim
until it is run once. Confirm the metric surface and early loss trace match a
pre-change run. **This is the cheapest possible insurance against spending a day
chasing a bug that was introduced today rather than discovered.**

### Run 1 — probe only, and read one number *(~30 min)*

`step_probe.enabled: true`, nothing else new. **Check `probe/second_diff_rel`
inside the first few hundred steps.** Sitting near `1e-6` means the fit is
resolving float32 rounding rather than curvature, α\* is noise wearing a
plausible number, and **all of Part A stops until the probe is reformulated**.
No amount of windowing repairs it. Everything else in Part A is downstream of
this one reading, so it goes first.

Then `probe/fit_downward_rate` and `probe/alpha_iqr` vs `alpha_median` — per
§A3a.2 a wide per-probe IQR with a stable windowed median is *acceptable*; only
a wandering median kills it.

### Run 2 — phase 1 ONCE, then branch *(the big time saver)*

Every replay arm needs a phase-2 stage, and phase 1 (`train_prior` MLE) is the
expensive part. Run it once and snapshot: `on_exit: ['snapshot:phase1_exit', …]`
already writes it, and `checkpoint_name` resumes from it in preference to
`running`. **Every arm below branches off that one checkpoint.** Without this the
day is mostly re-running MLE.

### Run 3 — replay wiring sanity, 2 numbers *(~20 min from the snapshot)*

`prioritise.enabled: true`. `update_logw_stats` was **dead code until today**, so
before trusting any replay result:

| Read | Expect | If not |
|---|---|---|
| `replay/is_elig_frac` | **~0.5** | D29's invariant says `E_Q[δ] ≈ 0`, so about half the buffer should have `δ > 0`. A value near 0 or 1 means the sign convention or the tiling in the `logw` refresh is wrong, and **every priority is garbage** |
| `replay/is_ess_frac` | 0.5 – 0.8 at κ=1 | collapsing toward 0 = a few rows own the batch; lower κ |

Metric absent entirely ⇒ `current_log_z()` returned None (conditional flow head)
and the draw silently fell back to uniform.

### Run 4 — the deliberate explosion **is** the §A6 experiment

Worth naming, because it upgrades a diagnostic into the crux test. §A6 asks
whether α\* degrades *before* the cliff or whether the cliff is a
gradient-outlier phenomenon α\* cannot see (the aug02 arm died at pre-clip grad
**587× the clip**). I said the aug02 retro-probe was infeasible — `Δ` for a
single step cannot be reconstructed from checkpoints thousands of steps apart.
**A deliberate local blow-up with the probe running answers the same question
directly, cheaply, and today.**

Log the pre-clip gradient distribution alongside. Two outcomes, both useful:

- α\* falls smoothly approaching the blow-up → the servo has lead time, §A4 is
  worth building.
- α\* healthy right up to death → the cliff is an outlier phenomenon, and the fix
  is bounded per-row influence (§B5), not a better LR sensor. **That is positive
  evidence for the replay work**, and §B10's LR-ceiling arm becomes the one that
  matters.

### ⚠ Trap: do not de-huberize `fwd`

B5b's "beta inactive on any prioritised branch" applies to **replay and bwd**.
`fwd` trains Z, and its `beta` *defines* the Z fixed point that D29's invariant
is stated against (`z_calibration` with `unclipped: false` targets exactly that
winsorised point). Change `fwd`'s beta and the invariant moves underneath the
`fwd/tb_resid_clipped < 0.5` check you are running.

### ⚠ Gap: the memorisation precept is NOT built

The "purposeful overfit and recovery" test has no sensor yet. `weighted_replay_err
≥ fwd_err` with its derived null of 1 (D22/§B7a) is **designed, not implemented**.
What exists is the old `replay/scatter_err ÷ fwd/scatter_err` at a hand-calibrated
~2×, whose calibration §B7 shows moves with the admission scheme — which uniform
intake just changed. So that ratio is **not readable** on a prioritised run.
Either build the weighted statistic first, or treat the overfit test as
observational (watch `is_ess_frac` and the raw ratio's *trend*, not its level).

### New metrics, complete list

```
probe/alpha_star  alpha_median  alpha_iqr  alpha_n
probe/curvature   step_norm     second_diff_rel
probe/fit_ok_rate fit_flat_rate fit_downward_rate  nostep_rate
replay/is_ess_frac  is_w_max_ratio  is_elig_frac
```

### Attribution note

`kind: ratio` (D8), the prioritised replay, and the probe all landed together.
The `over_coverage` / `relative_under_wcen` co-convergence check is therefore
also a test of the **new controller** — if it misbehaves, that is at least two
candidate causes. Consider one arm on the old controller kind to separate them.

---

## 0c. The `batt0807` local battery — arms, budgets, kill-gates

Generator: [`configs/local_aug07/make.py`](../configs/local_aug07/make.py).
Every arm has its own `run_name` under tag `batt0807`, so no arm can clobber the
`dev_mk_dev_*` set or another arm. T=10, batch 2831, ~4 it/s measured, so
**~250 steps/minute** — that is the budget currency below.

Launch (PYTHONPATH is required — `train.py` imports both packages):

```bash
PYTHONPATH="C:/Users/mikem/Projects/mxt_gfn/mxtaltools;C:/Users/mikem/Projects/mxt_gfn/gfn_diffusion" \
  "C:/Users/mikem/venvs/csd_mxt_gfn/Scripts/python.exe" train.py --config configs/local_aug07/<arm>.yaml
```

| # | Arm | Steps | ~Time | What it decides |
|---|---|---|---|---|
| 0 | `r0_null` | 600 | 2.5 m | **Null regression.** Everything new OFF. The five call sites touched on 08-06/07 are gated; this is the once that proves it |
| 1 | `p1` | to phase-1 exit | ? | **Fresh phase 1 + first probe read.** Produces the snapshot every arm below resumes from |
| 2 | `r2_wiring` | 1500 | 6 m | **Replay wiring sanity.** `is_elig_frac ≈ 0.5` |
| 3 | `r3_kappa00` / `r3_kappa10` | 3000 | 12 m | **κ ladder.** Φ is invariant by construction, so any difference IS estimator variance |
| 4 | `r4_overfit` / `r4_overfit_servo` | 3000 | 12 m | **Deliberate memorisation + servo recovery.** Intake starved to `churn_rate: 8`, residence 400 |
| 5 | `r5_nopolicy_replayoff` | 3000 | 12 m | **`P8` arm (i).** fwd carries policy grads, replay off — the closest thing to standard on-policy TB, which is the bar |
| 6 | `r6_bwdbeta_b10` / `_b60` | 2500 | 10 m | **bwd beta ladder.** B5b says de-huberizing is what restores within-tail ordering |

### Kill-gates, in the order they fire

1. **`probe/second_diff_rel` at ~1e-6** (run 1, first minutes) → the probe is
   resolving float32 rounding, not curvature. **All of Part A stops.**
2. **`replay/is_elig_frac` far from 0.5** (run 2) → the newly-live
   `update_logw_stats` has the wrong sign or tiling, and **every priority is
   garbage**. Nothing downstream is readable.
3. **`replay/is_ess_frac` → 0** (run 3) → a few rows own each batch; lower κ.
4. **`probe/fit_downward_rate` rising** → the local quadratic model is wrong and
   α\* is noise regardless of what its median does.

### Results so far — 2026-08-07

**`r0_null` ✅ PASSED.** 601 steps, zero errors, "Finished Training!". The five
call sites touched on 08-06/07 are genuinely inert without their config blocks.

**Three bugs found, all fixed.** This is what the battery is for:

| Bug | Whose | Effect if unfixed |
|---|---|---|
| Probe passed `mode_repeats('replay')` while scoring with `bwd_loss_coeffs` | mine | `AssertionError: tbc needs repeats > 1` on **every phase-1 probe**. A coeff bank is only valid at its own branch's K |
| `active_modes` / `read_modes` iterate `balance['rules']`, which `kind: ratio` does not populate | the new D8 controller | `KeyError: 'rules'` on the **first fused step of the naive stage** — would have killed every phase-2 run in the battery |
| `r5` set `fracs.fwd = 0.4` against an inherited `pinned.fwd = 0.2` | mine | arm dies at config parse |

The `kind: ratio` one was caught by running, the `r5` one by the config
regression the local-run recipe mandates (743 stages re-parsed and both fixed
properties exercised; all 10 `ratio` stages now pass).

### ✅ Kill-gate 1 CLEARED — the probe measures curvature, not rounding

`lrprobe/second_diff_rel` runs **5e-4 to 1e-2**, roughly **1000× above the 1e-6
float32 floor**. Part A survives its cheapest possible death.

**⚠ The first reading I published from this was WRONG — corrected 2026-08-07.**

I reported `alpha_median` 0.42–0.50 and concluded the route runs at the `2/λ`
edge of stability, taking steps ~2× the Newton step. **That reading was taken at
step ~790, ~200 steps into phase 2, deep in the log Z transient.** Across p1's
full phase 2 it keeps falling and settles somewhere else entirely:

| | p2 start | 25% | 50% | 75% | end |
|---|---|---|---|---|---|
| `fwd/log_Z_learned` | 0.91 | 20.7 | 20.5 | 20.45 | 20.58 |
| `fwd/tb_resid_clipped` | −6.24 | 0.13 | 0.21 | −0.06 | −0.15 |
| **`lrprobe/alpha_median`** | **10.19** | **5.14** | **2.42** | **1.73** | **1.68** |

`log Z` lands by ~25% of phase 2 and `tb_resid_clipped` is inside D29's ±0.5 band
from there on — but **α\* is still moving at 75%**, settling near **~1.7**.

So the route **undershoots by ~1.7×**, it does not run hot. That *restores*
§A1's undershoot premise rather than inverting it, and it means `α_target = 1.0`
would tell the servo to **raise** LR modestly, not halve it.

**The methodological lesson is the transferable part** (user, 2026-08-07):
anything whose reading depends on the residual distribution or on log Z having
settled must be measured **after** the transient, not during it. That applies to
more than α\*: `is_elig_frac` tracked log Z 0.27 → 0.54 over the same window
because `δ₊` is not shift-invariant (§B8). A `P2_CKPT` resume point at p1 step
2650 now anchors every such arm; only the baseline-convergence arms still start
from `phase1_exit`.

*(Namespace note:*(Namespace note: metrics renamed `probe/` → `lrprobe/`; `train.py:463-471`
already owned `probe/*` for the memory/timing probe.)*

### ✅ Kill-gate 2 CLEARED — and it validates §B8's argument

`r2_wiring`: **`replay/is_elig_frac` 0.27 → 0.535**, against a predicted ~0.5.
The whole reconstruction chain works — `δ = log Z − ema_logw` off the newly-live
`update_logw_stats`, with the right sign and tiling.

**The way it got there is the interesting part.** It *started* at 0.27 and climbed
to 0.54 while `fwd/log_Z_learned` climbed 0.9 → 10.8. That is §B8's shift
argument observed directly: `δ₊ = max(δ, 0)` is **not shift-invariant**, so the
log Z level literally sets how much of the buffer has nonzero priority. The
argument was derived; this is the measurement.

### ⚠ ESS collapse — found, diagnosed, fixed

`replay/is_ess_frac` came back at **0.02–0.06** against a synthetic prediction of
0.65, with `is_w_max_ratio` at 46–217. A 1000-row batch was doing the work of
20–60 rows.

Cause: a row *just barely* above zero residual draws with `p ≈ 0` and therefore
carries `w ≈ 1/p`. The shipped relative floor of 1% of the median `δ₊` was far
too permissive. Swept against r2's own live buffer:

| `floor_frac` | ESS/n | max(w)/mean(w) |
|---|---|---|
| 0.01 (shipped) | **0.11** | 73 |
| 0.15 | 0.50 | 5.3 |
| **0.25 (now default)** | **0.63** | **3.3** |
| 0.50 | 0.80 | 1.9 |

0.25 is the knee — it recovers the predicted ESS while keeping the
prioritisation. **This is why `is_ess_frac` shipped as a metric**; the estimator
cannot go wrong in the mean, so the weight tail was the only thing left to watch,
and it was the thing that was wrong.

### ❌ My Huber-suppression hypothesis was BACKWARDS

I predicted de-huberizing would *raise* `second_diff_rel` (Huber flattening the
surface the probe measures). Measured, `r2_wiring` (β = 1e6) came in at
**8.7e-8 — one hundred times WORSE than p1's 8.5e-6 at β = 10**, and below
float32's ~1e-7 resolution. `fit_flat_rate` climbed to 0.63 and `fit_ok_rate`
fell to 0.25: the probe was rejecting three quarters of its own readings.

**The driver is loss MAGNITUDE, not surface shape.** A quadratic TB loss on
~25-nat residuals is ~625; float32 resolves that to ~6e-5, which is the same
order as the second difference. Huber's cap was *keeping the loss small enough to
measure*. So the probe and de-huberizing are in direct tension — and B5b wants
de-huberizing everywhere prioritisation runs.

**Fix: widen the probe arc.** `span` evaluates at `α ∈ {0, s/2, s}`; the second
difference scales as `s²`, so `span=2` buys 4× at zero cost. Verified: α\* is
bit-identical across `span ∈ {1,2,4}` on an exact quadratic while
`second_diff_rel` scales ×4 and ×16 exactly. Default is now 2.

*(This is the §0 battery's whole purpose working: a derived hypothesis, measured,
refuted, and replaced by the real mechanism inside one arm.)*

### ⚠ Budgets were 4× too long — recalibrated

Phase 1 runs **2.67 it/s**; phase 2 settles at **1.0–1.2 it/s** (fused = three
branches per step, plus eval spikes). Every budget had been sized off the
phase-1 rate. All arms rebudgeted; `p1` was stopped at 2606/4001 once it had
delivered its checkpoint and readings, rather than spend 35 more minutes.

Paired at batch 1000: ~0.85 it/s each, **~1.7 it/s aggregate** vs 1.0–1.2 solo,
GPU util 28% → 72%, **no OOMs**. Pairing is worth ~1.5×, not 2×.

### ✅ r7 — the probe is VALIDATED, and `α_target = 1.0` looks wrong

Post-transient (resumed from `P2_CKPT`, span=2), 800 steps each, ~76 valid
readings per arm:

| arm | `lr_fused` | predicted α\* | **measured `alpha_median`** | `alpha_iqr` |
|---|---|---|---|---|
| `r7_lr_half` | 6.25e-5 | 3.4 | **3.13** (2.58 – 4.47) | 1.38 |
| `r7_lr_double` | 2.5e-4 | 0.84 | **0.97** (0.95 – 1.10) | 0.14 |

**The ratio test.** If `α* ∝ 1/lr` exactly, `α*(half)/α*(double)` must equal
`lr_double/lr_half = 4.0`. Measured **3.24** — within 20% across a 4× LR range,
with the two arms on genuinely different trajectories (so `H` and `g` are not
the same at both ends, and 20% is about as good as this test can be).

**Every §A3a concern is now closed, not deferred:**

| Concern | Before | Post-transient + span=2 |
|---|---|---|
| A3a.3 bad fits | `fit_downward` 0.25–0.47, `fit_flat` 0.63 | **0.00 and 0.00**, `fit_ok = 1.00` |
| precision | 8.7e-8 (below float32) | **0.008 – 0.12**, 3–4 orders above the floor |
| A3a.2 dispersion | IQR ≈ median | IQR/median **15%** (double arm), 44% (half) |

The two fixes that got there were the arc widening (span², §A4b) and the
post-transient anchor. Neither was in the original design.

**Cross-check:** interpolating to p1's `lr_fused = 1.25e-4` predicts α\* between
1.57 and 1.94; p1 measured **1.68**. The sensor is self-consistent across three
independent LRs.

#### ⚠ But `fwd/tb_err` moves the WRONG way at α\* = 1

| arm | α\* | `fwd/tb_err` over the 800 steps |
|---|---|---|
| `r7_lr_half` | 3.13 (undershooting 3×) | 21.5 → **20.6** ✅ improving |
| `r7_lr_double` | 0.97 (**at the setpoint**) | 21.5 → **23.4** ❌ degrading |

`α_target = 1.0` is defined as the locally optimal step on the probe batch — and
the arm sitting exactly there is the one whose objective gets *worse*, while the
arm undershooting 3× improves. **A locally greedy step size is not the best step
size for stochastic convergence**, which is unsurprising in the abstract and
decisive here: it means §A4's setpoint cannot be taken from the geometry alone.

This is precisely the question D27 deferred to measurement, and the measurement
now exists. Caveat honestly: 800 steps is short and `tb_err` is noisy, so treat
the *direction* as the result, not the magnitude. The follow-up is an α\*-vs-
convergence sweep at fixed wall clock — the sensor is trustworthy enough to
build one now.

### ⚠⚠ The draw was never prioritised — `beta` is a uniform FRACTION

**The most consequential bug of the shakedown.** `_sample_indices`' `beta` is the
*fraction of the batch drawn uniformly*, not a temperature:

```python
n_uniform  = int(batch_size * beta)     # beta=1.0  ->  ALL of it
n_weighted = batch_size - n_uniform     #           ->  none
```

`draw_replay_sample` inherited `beta=1.0` from the legacy uniform call, so a
supplied `p` was **silently ignored**. Worse than a no-op: the IS weights
`w ∝ 1/δ₊^κ` were still applied to the loss, and a *uniform* draw carrying `1/p`
weights targets a measure `∝ 1/δ^κ` — **the inverse of the design**,
up-weighting the lowest-residual rows.

**The tell was an identity that should have been impossible.** At κ=0 the draw is
uniform and the weights are uniform, so `is_ess_frac` must be *exactly* 1. It
read **0.40 — equal to `is_elig_frac`'s 0.39**, because the uniform draw kept
pulling ineligible `w = 0` rows. Post-fix κ=0 reads exactly 1.000 with
`is_w_max_ratio` 1.000.

**Why the offline tests missed it:** they called `prioritised_weights` directly
and never went through the loader. A unit test of the estimator cannot catch a
mis-wired *draw*. The κ=0 cell — the one with a known exact answer — is what
exposed it, which is an argument for always including a degenerate cell in a
ladder.

### ⚠ One-sided draw + without-replacement = crash

`ValueError: Fewer non-zero entries in p than size`, killing the κ=0 arm at step
119. `δ₊ ≤ 0` rows get `p = 0` by design (§B5), and `is_elig_frac` drifted
0.74 → 0.33, so the eligible pool fell below the 1000-row batch.

**Fixed at the principle, not with a guard: a supplied `p` is drawn WITH
replacement**, because importance-sampling correctness assumes iid draws from the
design measure. Without-replacement was both theoretically wrong and the
proximate crash. Reproduced directly — at 1500 eligible it succeeds, at 900 and
300 it raises.

**Watch `is_elig_frac`.** Its drift 0.74 → 0.33 over 1500 steps means the buffer
is trending toward mostly-negative δ. That is the §B8 shift mechanism running the
other way, and if it approaches 0 the prioritised branch has nothing to draw.

### r4 — the overfit worked, the sensor worked, the SERVO never fired

**Half a success.** The deliberate intake starve (`churn_rate: 3`,
`mean_residence_steps: 400`) drove `replay/lambda_tau` to a peak of **1.43**,
comfortably past the `λτ = 1` boundary, with `resid_vs_intake` down to 0.24
against the derived 0.368 bar. **The memorisation sensor detects induced
memorisation** — that half of D22 is validated, and the B7a rebudget (churn 8→3
to hold `λτ ∝ B/rate` against the batch change) was necessary to get there.

**But `protocol/bs_log_boost` and `bs_ratio` were never emitted at all.**

Cause: `StageProtocol._resolve` reads every servo sensor through
`metric_tracker.get(direction, metric)`. `absorption_stats()` was being published
only into the wandb metrics dict at report time, so the tracker never held
`replay/ema_loss_mean`. The servo resolved `None` and took its cold-start early
return on every tick — indistinguishable, in the logs, from a servo correctly
deciding to hold.

Fixed by publishing the sensor into the tracker from `replay_train_step`.
Deadband verified against the observed trajectory: at ratio 0.24 the drive is
**+0.85**, stepping `log_boost` +0.043/tick toward more churn.

> **The general lesson, and it is the third instance today:** a control loop
> whose sensor is unreadable is silent in exactly the same way as a control loop
> that is satisfied. This is `S2`'s drive-liveness argument again — the register
> already says every one-sided controller should report how long it has been at
> zero, and this servo would have announced itself immediately if it did.

### 🔴 THE SESSION'S RESULT: `freeze_policy` is the cost, not replay

`r9` isolates the one variable. Common window, steps 2650–3450:

| arm | fwd | replay | `fwd/tb_err` | `fwd/r2` | `over_coverage` |
|---|---|---|---|---|---|
| `r5` | policy grads | **OFF** | 21.54 → **20.97** ✅ | −3.26 → **−2.54** | 20.85 → **20.27** |
| **`r9`** | **policy grads** | **ON** | 21.63 → **21.14** ✅ | −3.30 → **−2.47** | 20.94 → **20.50** |
| `r6_b10` | **FROZEN** | ON | 21.94 → **23.54** ❌ | −3.46 → **−5.15** | 21.27 → **22.88** |

**Two conclusions, and the second is not the one Part B was built for.**

1. **`freeze_policy: 1.0` on fwd is what costs convergence.** Same replay
   configuration, same batch, same `beta`, same κ — unfreeze fwd and the sign
   flips from degrading to improving on all three metrics.
2. **Replay is neutral, not harmful.** With fwd unfrozen, replay-on (`r9`) and
   replay-off (`r5`) are tied (21.14 vs 20.97; `r2` −2.47 vs −2.54). So `P8`
   arm (i) does *not* say "delete replay" — it says replay neither helps nor
   hurts once the forward branch trains the policy.

**This contradicts `synthesis.md` §1's thesis directly.** That section states the
architecture as *"the policy is trained entirely off-policy; the on-policy
branch trains only Z — because Z is the ruler every off-policy residual is
measured against."* `freeze_policy: 1.0` is its implementation, and it is the
single knob that separates improving arms from degrading ones here.

**Independent corroboration, found earlier and not connected until now:**
`nys7cfrt` — the strongest baseline candidate on the cluster and one of the best
runs on record — **has no `freeze_policy`** (D2 flagged it as "the thesis does
not describe this run"). The best historical run was not using the thesis either.

**Honest limits.** 800 steps, one seed, one problem (mipcas ELJ, T=10, batch
1000/2831). `tb_err` is a forward-branch metric, and unfreezing fwd trains the
policy *on* that branch — so some of the gain could be the metric moving toward
its own trainer rather than genuine improvement. **`fwd/r2` and `over_coverage`
move the same way**, which argues against pure metric-gaming, but the clean test
is a held-out coverage measure and a longer horizon.

### Superseded: "replay-on degrades, replay-off improves"

| arm | `fwd/tb_err` | `fwd/r2` | `over_coverage` |
|---|---|---|---|
| **`r5`** — replay OFF, fwd trains policy | 21.54 → **20.06** ✅ | −3.26 → **−2.22** ✅ | 20.85 → **19.34** ✅ |
| κ ladder / β ladder — replay ON | 21.9 → 23.5–26.4 ❌ | −3.4 → −5.1 to −6.9 ❌ | 21.3 → 22.9–25.8 ❌ |

**Batch size is DEFINITIVELY ruled out** — `r8` (batch 2831, replay on) degrades
21.48 → 23.77, essentially identical to `r6_b10` (batch 1000, replay on) at
21.94 → 23.54, while `r5` improves in the same window.

**The split holds across six arms.** Replay ON degrades regardless of batch
(1000 / 2831), `beta` (10 / 60 / 1e6), or κ (0 / 1). Replay OFF improves. Every
knob inside Part B was varied and none of them changed the sign.

This is `P8` arm (i) — *"is a corrector necessary at all?"* — answering in the
direction that would remove most of Part B. **It is not yet attributable**: `r5`
changes two things, replay off *and* `freeze_policy` removed from fwd. `r9`
moves only one (replay stays on and prioritised; fwd gets policy grads back):

- **r9 improves** → the culprit was `freeze_policy`, and replay is exonerated
- **r9 degrades** → the culprit is the replay branch itself

Note how much rides on this. `synthesis.md` §1's thesis is *"the policy is
trained entirely off-policy; the on-policy branch trains only Z"* — and
`freeze_policy: 1.0` is what implements it. If r9 improves, the thesis is what is
costing convergence, not replay.

### r4 servo — reading now, but with no authority

Post-fix, `protocol/bs_ratio` appears with 200 readings spanning 0.27–0.56,
dipping below the 0.368 bar as intended. But `λτ` ends at 0.719 vs the no-servo
arm's 0.652 — **no evidence of successful control**, because the loop is far too
weak for the perturbation: `gain 0.05 × ~8 ticks` over 2000 steps is ~1.4× churn
recovery against a **27× intake starve** (`churn_rate` 80 → 3). Reaching
`max_boost: 8` at 0.043/tick needs ~48 ticks ≈ 12k steps.

So the servo is *validated as wired* and *unproven as effective*. A real test
needs either a milder perturbation or a much higher gain. `protocol/bs_log_boost`
is now reported so the actuator is visible at all — previously only the sensor
was, which is the same S2 blind spot in a third guise.

### Two traps these configs are written around

- **`epochs` is an ABSOLUTE ceiling on a resumed run** (`trange(init_step,
  epochs+1)`), not a budget. An arm resuming at 12000 with `epochs: 3000` runs
  **zero steps** and verifies nothing. Every resumed arm sets
  `epochs = P1_STEPS + budget`, asserted at generation.
- **`continue_from_checkpoint: true` + `checkpoint_name: null` resolves to
  `{tag}_{run_name}_..._running.pt`.** Since `run_name` is unique per arm that
  file never exists, so the arm does not chain — it **silently retrains phase 1**,
  which is invisible in the results and costs the day. `assert_pinned_resume`
  now enforces all four fields.

### Deliberately NOT in the battery

- **`fwd` beta is untouched in every arm.** De-huberizing is for replay and bwd
  (B5b); `fwd` trains Z and its beta *defines* the fixed point D29's invariant is
  stated against.
- **Stage 2 of the LR controller.** No servo, no line search — the probe is
  instrumentation only until its dispersion is measured (D27).
- **The bwd prioritised draw.** Dropped 2026-08-06 (§B5a): prior-buffer
  `1 + CV² ≈ 1.06`, and the ordering argument it rested on dissolved.

---

# Part A — LR controller

Answers the design decision left open at
[`module_lr_controller.md`](module_lr_controller.md) §9 ("if adaptivity is
wanted, it belongs on the peak — decide it deliberately") and takes a position
on what was `questions.md` §C *"Radical simplicity or an adaptive scheme for
LR?"*, whose answer there was **neither**. This argues for adaptivity, and §A6
confronts the aug02 evidence against it.

**Settled 2026-08-06** (`decisions.md` D3/D4): build the servo. §A6 is now
instrumentation rather than a gate — see §A8's status block for what that
changes.

> ## The point, up front
>
> *(added 2026-08-06 — user: "kind of a long story to get to the point, which is
> the new sensor.")*
>
> **§A3 is the proposal. Everything before it is the case for bothering, and
> everything after it is routine control engineering.**
>
> Today's sensor (`cut_grad_abs`) measures **gradient magnitude**. Under Adam
> with tight clipping the step actually taken is ~`lr` per coordinate *largely
> independent of gradient norm* — so the sensor is decoupled from the thing being
> controlled, and no recalibration fixes that (§A2).
>
> The replacement measures the controlled variable directly: **take the step the
> optimizer actually took, and ask whether it was the right size** — by
> re-evaluating the loss at 0×, ½×, and 1× that step on a held-out batch and
> fitting a parabola. The answer, `α*`, is *dimensionless with a setpoint at 1*,
> which is why it can transfer across `T`, energy function, and clip setting when
> an absolute gradient bar cannot.
>
> Read §A3 → §A3a.2 → §A3a.3 and you have the whole idea. §A1/§A2 are the
> motivation; §A4–§A8 are the loop, the deletions, and the fallback.

## A1. The controller has two costs and only sees one

| Cost | Shape | Currently sensed |
|---|---|---|
| **Undershoot** — steps burned at an LR below the envelope optimum | linear-ish in wall clock; no failure signature at all | **no** |
| **Detonation** — loss/grad runaway | thresholded, catastrophic, recoverable by rewind | yes (reset tier) |
| **Flat-direction diffusion** — LR-proportional random walk in `step_var` | linear in LR, *no threshold*, degrades solution without raising loss | no |

The controller's only actuation is downward. It is structurally an undershoot
machine, and the record bears that out: **every documented waste event in this
module's history is an undershoot event.** g8d8se26 spent 24k of 27k steps
pinned at the `0.01` floor with grads 30–50× *under* the bar, still improving.
replay_july26's gentle arms ran at policy LR 1e-6. tw_july31 arms sat frozen at
`cut_factor` exactly 0.50 for the rest of the run ([[live-vs-set-lr]]). Not one
of those was a safety event; all three were the safety machinery misfiring into
starvation.

Against that, the payoff for getting LR *right* is the largest measured lever on
this route (`decisions.md` §2c ranks LR first, 2.4×). So the asymmetry the current
design assumes — overshoot is expensive, undershoot is free — is backwards in
the observed record. **Undershoot is the empirically dominant cost.**

### Tolerance (design input, user 2026-08-05)

**Target band: stable, and within ~5× of optimal.** Not "extract the last 2×."
Stability is the requirement; leaving *tons* of LR on the table is the failure.
`strictly optimal` is itself an unsettled design question and is explicitly not
the target.

This band is load-bearing for everything below — it decides which of the §A3a
considerations matter and which are noise. In particular a 2–4× conservative
bias sits **inside** tolerance, so it is a preference, not a defect.

The flat-direction cost ([[step-var-is-the-flat-direction]],
[[buildout-variance-flat-direction]], [[flat-direction-limit-cycle-phase2]]) is
real and is genuinely thresholdless, but it is a *second-order* term next to
both of the above. It belongs in the design as a separate, slow input — not as
a reason to bias the whole servo down. *(Warrant: derived from the waste-event
audit in `module_lr_controller.md` §9; the ranking of the three costs against
each other is **contested** — no battery has priced diffusion against wall
clock.)*

## A2. The current sensor cannot be fixed by recalibration

`cut_grad_abs` measures **gradient magnitude**. Under Adam with the tight
clipping that won at T=10 and T=60 ([[tw-july31-battery-verdict]]), the step
actually taken is ~`lr` per coordinate largely independent of grad norm. The
sensor is decoupled from the controlled variable.

Consequences already in the record, all explained by that one fact:

- Bars moved **3.3×** between ty4xdlzo and the current tree with no principle
  separating them ([`audit_since_ty4xdlzo.md`](audit_since_ty4xdlzo.md) L37).
- tw_july31 arm 14 detonated at pre-clip 154,299 — **between** its cut bar
  (38,640) and reset bar (386,400). Cut-only, latched, never recrossed, parked
  forever.
- Every *relative* variant fails two-sided: in s706frkh the floored median fired
  on a clip-neutralised grad of 745, then the incident's own 1e4 norms lifted
  the median and it went blind exactly when the real excursion began.

Relative-to-history is not repairable by a better estimator alone, because the
excursion contaminates the baseline. (A hot-frozen quantile — update the
baseline *only* on cold readings — does repair both s706frkh modes and is the
cheap fallback in §A8. It still measures the wrong quantity.)

## A3. Proposed sensor: two-point step probe

Measure the thing being controlled: **was the step actually taken the right
size?**

Let `Δ = θ_after − θ_before` — the real optimizer step, momentum and clipping
included. On a fixed batch evaluate

```
L(θ_before)        L(θ_before + Δ/2)        L(θ_before + Δ)
```

**What `α` is** *(this was missing and the section was unreadable without it)*.
Define the ray

```
θ(α) = θ_before + α · Δ
```

so `α` is a **dimensionless multiplier on the step the optimizer actually took**:
`α = 0` is "didn't move", `α = 1` is "the step you took", `α = 2` would be "twice
that step, same direction". The three evaluations above are exactly
`α ∈ {0, ½, 1}`. Fit a parabola through those three `(α, L)` points and let `α*`
be its argmin — *what multiple of the step you took would have been optimal.*
`α` is not confined to [0, 1]; `α* > 1` is the undershoot case and is the whole
point of the sensor.

Closed form, with `L₀ = L(0)`, `L½ = L(½)`, `L₁ = L(1)`:

```
α* = (3L₀ + L₁ − 4L½) / (4·(L₀ + L₁ − 2L½))
```

- `α* ≈ 1` — step size correct
- `α* < 1` — overshot by `1/α*`
- `α* > 1` — undershooting; **affirmative permission to grow**

The denominator is `4a` where `a = 2(L₀ + L₁ − 2L½)` is the fitted curvature, so
**`a ≤ 0` is the degenerate case** — see §A3a.3.

Two design points carry the weight:

**(a) Held-out probe batch, identical data at all three α.** Same-batch probing
is biased high — the step reduces loss on its own batch more than on the
population, so it systematically licenses too-large steps. A disjoint batch
removes that bias; using the *same* disjoint batch at all three α keeps the
comparison noise-free, so we get unbiasedness without reintroducing the
cross-batch trajectory/condition noise that makes ordinary loss comparison
useless here. Rotate the probe batch slowly so the draw averages out.

**(b) `α*` is a setpoint, not a threshold.** It has a physical anchor at 1.0 —
see §A3a.1, the exact target is open. This is the argument for the rewrite: it
converts *"where do we set the bar?"* — a question that has never transferred
across T, problem, or clip setting — into a servo error against a
**dimensionless** target. Note the earlier claim of *no calibrated constant* was
overstated; `α_target` is one. It is a better one (transferable, theoretically
anchored) than an absolute grad bar, which is the whole claim.

**Cost.** Three forward passes over *stored* trajectories per probe: states and
energies are already in the batch, so no resampling and no energy calls. Plus
one parameter-sized buffer for `θ_before`. At 512×4 this is negligible; at probe
cadence ~20 steps it is a few percent of wall clock.

**What it sees:** curvature overshoot, and — as a free side effect — the
stale-Adam-at-transition spike ([[stale-adam-transition-ejection]]), which is
currently invisible because `Δ` includes momentum.

**What it does not see:** flat-direction diffusion (§A1, row 3). `α*` can sit at
1.0 while `step_var` walks. If that cost is to be controlled it needs its own
slow input, kept as a distinct term — not blended into `α*`.

## A3a. Considerations on the probe — open, not decided

Raised against the §A3 sensor 2026-08-05. **None of these is a design decision
yet.** Each is scored against the §A1 tolerance band, which is what determines
whether it matters.

**1. The α\* = 1 setpoint is ~2–4× conservative.** *(derived)* For a locally
quadratic loss with step `Δ = lr·d`:

```
α* = (g·Δ)/(ΔᵀHΔ) = (1/lr) · (g·d)/(dᵀHd)
```

so `α* = 1` ⟺ `lr = 1/λ`, the Newton step. Stability is at `2/λ`, so **α\* = 0.5
is the edge of stability** and the catapult regime (`2/λ → 4/λ`) is
**α\* ∈ [0.25, 0.5]**. A servo parked at α\* = 1 runs 2–4× under.

*Verdict against tolerance: inside the band.* This resolves the "is optimal 2–4×
below hot?" question — for this sensor the answer is inverted, α\* = 1 sits 2–4×
**below** the aggressive regime, not above it. Given the stated preference,
**α_target = 1.0 is a reasonable conservative default** and 0.5 is the
aggressive end of a one-knob sweep. The setpoint is not load-bearing and should
not absorb more design time.

**2. Dispersion is the kill switch, not the median.** *(open — measurable)* A
fixed-batch "correct step" only exists if batches agree on one. Report the
spread of α\* across probes. With interpolation violated (moving Z target,
churning buffer) wide spread is a live outcome.

*Scored against tolerance:* the bar is weaker than first stated. We need the
**windowed median reproducible to within ~2×**, not tight per-probe agreement.
Wide per-probe IQR with a stable windowed median is acceptable. Kill only if the
windowed median itself wanders.

**3. Three points cannot detect a bad fit.** *(open — measurable, free)* The
parabola opens downward — making α\* a *maximum*, meaningless, though it still
returns a number — exactly when

```
L½ > (L₀ + L₁)/2          i.e.  a = 2(L₀ + L₁ − 2L½) < 0
```

the midpoint sitting above the chord. *(Corrected 2026-08-06: an earlier draft
said "L(½) exceeds both endpoints," which is a strictly stronger condition and
would miss most degenerate fits.)* Note `a → 0` is the other failure — a
near-flat fit sends α\* to ±∞, so the guard is on `|a|`, not just its sign. Log
the curvature sign and the rate of degenerate fits. A high rate means the local
quadratic model is wrong and every α\* reading is noise. Ship this diagnostic
with the read-only probe; it costs nothing.

**4. TB loss vs the true objective — RULED, not open.** The concern was that a
step can cut TB loss on a fixed batch by moving log Z (shrinking residuals
uniformly, improving the sampler not at all), so α\* could be satisfied by
Z-fitting; signature would be α\* healthy while `EffDim` / `z_gap` degrade.

**User ruling (2026-08-05): TB loss is fine for this type of stability work.**
Correct, and the scope is the reason: the probe's job is *step-size safety*,
which needs a metric locally sensitive to the step, not one aligned with
`KL(Q‖P)`. Objective alignment is owned by other sensors ([[wandb-run-diagnostics]]:
EffDim, `z_gap`). Recorded so it is not re-litigated. The narrow residue —
`lr_flow` is pinned separately, so α\* rates a composite step including a head
the servo does not control — is a scoping note for whether the servo drives
`lr_fused` alone.

**5. ❌ RETRACTED 2026-08-06.** *(user objection, and it is correct.)*

The claim was: with a ~5× tolerance the unconditional route may need no servo at
all, because aug02 put the optimum at 4e-4 with the cliff before 8e-4, so a fixed
1e-4 sits 4× under — inside tolerance. The servo's case was therefore framed as
**transfer, not extraction**.

**Why that is wrong.** It assumes "the unconditional route" is one problem with
one optimum that aug02 measured. It is not:

> *"We have a very weak idea of the correct cruising LR and it's anyway highly
> problem dependent, and the problem (energy function, space group, conditions,
> models, rollouts) are constantly shifting."*

aug02 measured **one point** in that family — one energy, one `T`, one `W`, one
clip. Reading it as "the unconditional optimum" is exactly the failure mode in
[[findings-must-generalize]]: promoting one battery's number to a general fact.
There is no stable *here* to extract from, so the transfer/extraction distinction
collapses — **every run is a transfer.**

This *strengthens* the case for the servo rather than weakening it, and it is
retroactive support for D3. The fallback in §A8 remains a fallback, not a
destination.

### The scoping statement that replaces it *(user, 2026-08-06)*

> *"The main point of the new LR controller implementation is that we have a
> plausibly much better sensor. Everything else should be kind of
> straightforward."*

**The sensor is the deliverable.** §A4's loop, the ceiling, the growth rate are
ordinary control engineering once α\* is trustworthy. Design attention belongs on
§A3 and §A3a.2/.3 — is the reading *stable* and is the fit *valid* — not on the
servo wrapped around it.

## A4. Proposed loop

```
lr = peak × envelope(t)          # envelope: ramp → hold. NO decay leg -- see A4a

peak ← peak × clip(median α* over window, 0.9, 1.1)
```

Growth is *licensed by evidence*, not blind. This is what answers §A1's undershoot
problem and what a breach-only AIMD cannot do: a blind scheme only ever learns
it went too far by going too far, so it must creep. `α* > 1` is affirmative
permission, so the servo can climb fast while the surface says climb, and slow
only as `α*` approaches 1.

Three supporting pieces:

- **Ceiling with forgetting.** A breach records `ceiling = peak`; growth rate
  scales with distance below it, so approach is asymptotic and re-breach is
  rare. The ceiling relaxes upward with a half-life ~ stage length: an LR the
  surface refused at step 2k should not bind at step 40k.
- **Growth timescale is derived, not picked.** To traverse a plausible 100× LR
  range (~6.6 doublings) inside the first ~20% of a 40k-step stage needs ~1200
  steps per doubling. Slower than that and the servo cannot pay off its own
  undershoot cost within the run. *(derived; the 100× and 20% are **arbitrary**
  and should be stated as the assumptions they are.)*
- **Containment unchanged.** Reset tier + rewind + `max_reloads` + the frozen
  detector stay exactly as they are. They answer to real external events and
  they work. A breach cuts `peak` and records the ceiling; no separate cooldown
  constant is needed on the growth path, because growth is already proportional
  to distance from a ceiling that just moved.

## A4a. The envelope loses its decay leg

*(added 2026-08-06 from `decisions.md` D7. **derived**.)*

**There is no step budget.** The run gets whatever ~7 days on an A100 buys,
typically under 100k steps, and the goal is to train to convergence rather than
to a schedule. So `hold_steps: 20000` + half-life 25000 presuppose a horizon that
does not exist — at 40k steps they finish near 0.59×, at 100k near 0.11×, and
neither number was chosen.

**Worse, under the servo the decay leg is inert.** α\* rates the *product*
`peak × envelope(t)`, because `Δ` is the step actually taken. Any deterministic
multiplier on that product is absorbed: as `envelope` decays, steps shrink, α\*
rises above 1, and the servo raises `peak` to compensate. The live LR ends up
wherever α\* puts it regardless — and `peak` is left inflated relative to its
own meaning, which matters because §A4's ceiling is expressed in units of `peak`.

**Ruling: keep ramp and hold, drop decay.** Ramp survives on a different warrant
— it is warmup, and it runs before the servo has any α\* data to act on.

**The one job decay could still do** is the flat-direction diffusion cost (§A1
row 3): LR-proportional, thresholdless, and structurally invisible to α\*. That
is a real cost, but it is second-order, no battery has priced it against wall
clock (**contested**), and §A3 already says it belongs as a *separate slow
input*. If decay returns, it returns on that warrant with its own sensor — not as
a schedule inherited from a step budget that was never set.

**Knock-on:** `P2` ("decay is barely engaged where it is live") is retired rather
than answered — there is nothing to calibrate. Same for §A4's "~1200 steps per
doubling," whose 100× and 20%-of-40k inputs were already flagged **arbitrary**
and now have no horizon to be a fraction of. Size the growth rate off the
*measured* time to traverse the LR range in the §A6 probe instead.

## A4b. Build order — **ready to build, in two stages**

*(added 2026-08-06.)*

### First: §A6's retro-probe is not what it says it is

§A6 describes running the probe "read-only across the existing aug02 ladder…
read-only instrumentation on a battery that already ran. No new risk." **That is
not achievable as stated.** `Δ` is *one optimizer step's* worth of movement, and
checkpoints are thousands of steps apart — so `θ_before` / `θ_after` for a single
step cannot be reconstructed from saved checkpoints at all.

It is *feasible*, just not free: checkpoints do carry optimizer state
([checkpointing.py:376](checkpointing.py:376), restored by
`load_full(load_opt_state=True)`), so you can load an aug02 checkpoint, take live
steps, and probe those. But that is re-running training segments on the cluster,
not reading logged scalars.

**So the live read is cheaper *and* better data than the retro-probe.** Which is
the user's instinct 2026-08-06, and it is right.

### Stage 1 — sensor + logging, zero actuation ✅ **BUILT 2026-08-06**

[`step_probe.py`](../step_probe.py) + hooks in `train.py`. **Disabled unless a
`step_probe` block is declared**, so it is inert for every existing config:

```yaml
step_probe:
  enabled: true
  cadence: 20   # probe every N steps
  window: 25    # readings in the median/IQR window
```

**Probe batch: a fresh replay draw every probe** *(revised 2026-08-06 on user
objection; the first cut froze one batch for 200 probes = 4000 steps, which is
absurdly stale).* Two corrections behind that:

- The invariant that matters is **identical data across the three α within one
  probe**, which is what keeps the second difference free of
  trajectory/condition noise. *Identical across probes* was never required, and
  buys nothing — a buffer draw costs no energy calls, so there was no cost to
  amortise. Re-drawing per probe also averages the particular draw out for free.
- **Replay is the on-policy distribution.** Replay rows are fed by the fwd branch
  (§B2 — "it inherits Q's blindness, by intake"), so a replay draw *is* on-policy
  rollouts with stored energies. That is the right place to read a step-size
  sensor: highest loss variance in the system, i.e. the worst case for
  stability rather than the most forgiving one. Falls back to a backward draw
  only in phase 1 or when the buffer is empty.

Logs `probe/alpha_star`, `alpha_median`, `alpha_iqr`, `alpha_n`, `curvature`,
`step_norm`, `second_diff_rel`, and the five status rates
(`fit_ok/fit_flat/fit_downward/nostep/aborted`).

Verified: the closed form is exact on synthetic quadratics (rel err 1e-13 in
float64); parameters are restored **bitwise** after every probe, including one
that raises part-way through (the restore is in `finally`, and the abort is
tallied into `aborted` so a probe that OOMs every time is distinguishable in
the log from one that never ran); the frozen
batch is drawn read-only (`loader` mutates nothing, and the probe deliberately
skips `update_losses`, so a probe draw is not a training visit and cannot
perturb churn, priority, or residence); `update_log_z=False` gates the only
tracker mutation.

**One risk the design did not anticipate, now instrumented.** The fit turns on
the *second difference* `L₀ + L₁ − 2L½`, and float32 carries ~7 digits — so a TB
loss of ~1e3 resolves second differences no finer than ~1e-4 absolute. If the
step is small, the probe measures **rounding, not curvature**, and α\* is noise
that still looks like a number. `probe/second_diff_rel` is that quantity divided
by the loss scale: readings near the 1e-6 floor mean the probe is
precision-limited, and it voids α\* exactly as surely as a downward fit does.
**Watch it from the first run** — it is the cheapest way this whole approach
could be dead, and no amount of windowing repairs it.

Costs one parameter-sized buffer and three forward passes over *stored*
trajectories per probe (no resampling, no energy calls).

Log every probe: `α*`, the fitted curvature `a`, the windowed median, per-probe
dispersion, and the degenerate-fit rate (§A3a.3).

**Log the gradient distribution alongside it** — pre-clip norm and per-row max.
This is the discriminator for §A6's favourite failure branch: the aug02 arm died
at pre-clip grad **587× the clip**. If the cliff is a rare-large-batch *outlier*
event rather than a curvature event, a probe on a *held-out typical* batch stays
healthy right up to death **by construction**. Logging both together says which
regime kills a run — and if it is the outlier regime, the fix is bounded per-row
influence (§B5, κ=1), not a better LR sensor. Part A and Part B answer the same
question from two sides here.

Stage 1 also *produces the two constants D7 orphaned* — see below.

### Stage 2 — the servo ✅ **BUILT 2026-08-08** (`controller.py` v7)

§A4's loop, shipped. Two constants were **unparameterized** because D7 removed
the horizon they were defined against; both are now set from stage 1's data:

| Constant | Was | Shipped as |
|---|---|---|
| growth rate | "~1200 steps/doubling," from a 100× range over 20% of a 40k stage — all three inputs **arbitrary**, and there is no 40k stage | `clip: [0.8, 1.25]` every `period: 200` steps = a doubling per ~620 steps at full drive. The **clip** is the measured part: F0 puts SE(median) at ~9% over a 25-probe window, so §A4's ±10% would have spent the loop's authority on sampling noise |
| ceiling forgetting half-life | "~ stage length" | `ceiling_halflife_steps: 20000`. Still **arbitrary** — α\*'s autocorrelation (lag-1 ≈0.5 at cadence 20) sizes the *median window*, not the ceiling's memory, and nothing measures the latter yet |

Three things the build added that §A4 did not specify:

- **`auto` vs a float is the servo's scope.** A key written `auto` is
  servo-managed; a float is a fixed peak. So the servo's own A/B control arm —
  probe reading and logging, actuating nothing — is a config with no `auto` in
  it, which is what `lr_aug08`'s `a_fixed` is.
- **The servo holds through warmup, and says so.** The envelope is below 1 there,
  so α\* rates a deliberately shrunken step and reads high; acting on it would
  inflate `peak` by exactly the warmup factor. `lr_ctrl/servo_hold` encodes
  *which* of six reasons is holding the loop, because an unreadable sensor and a
  satisfied controller are otherwise indistinguishable.
- **`peak_scale` is checkpointed; the ceiling is not.** A resume should keep the
  climb rather than re-derive it from the seed, but a rewind must not erase the
  evidence that this LR just detonated. Clamping the restored peak to the
  instance-held ceiling gets both.

**⚠ The build's first measurement was that the sensor is one-sided at low LR.**
See §A3b — without that fix the loop was inert below ~1e-5 and every servo arm
would have sat at its seed.

### The one decision blocking stage 1

**Which loss does the probe evaluate, and which parameter group does the servo
drive?** Flagged in §A3a.4 as a scoping note and never resolved. In a fused stage
there are three branches at different weights, and `lr_flow` is pinned
separately — so `α*` currently rates a *composite* step including a head the
servo would not control. Three options:

- **(a)** Probe the combined fused loss; servo drives `lr_fused` only. Simplest;
  `α*` is contaminated by the flow head's movement.
- **(b)** Probe the combined loss but compute `Δ` over the policy parameters
  only, holding the flow head fixed during the probe. Clean attribution, one
  extra masked copy.
- **(c)** Probe each branch separately and servo on the weighted combination.
  Most information, most machinery, and the branches disagree by construction.

**Recommendation: (b).** It makes `α*` a statement about the parameters the servo
actually moves, which is the whole premise of the sensor. (a) is the fallback if
the masked copy proves awkward.

## A3b. The sensor is ONE-SIDED at low LR — found 2026-08-08, on the smoke arm

*(measured. This is the defect that would have made the whole servo inert.)*

At `lr ≈ 1.4e-6` the probe returned **`downward` on 100% of fits**: `curvature`
consistently negative (~−0.004), `α*` always `nan`, `bad_rate_window` pinned at
1.0. `second_diff_rel` was ~1.1e-5 — three orders **above** the 1e-6 float32
floor, so this is not the precision failure §A3a anticipated. The curvature was
being measured, and it was genuinely negative.

**Why, and why it is not a bug in the surface.** The probe samples
`α ∈ {0, span/2, span}` along the taken step. At a small enough `lr` that arc is
short enough to sit inside a locally *concave* stretch — negative-curvature
directions are ordinary in a deep net — and never reaches the basin whose
positive curvature the fit assumes. The magnitude tracks the step size exactly as
it should: F0's `batt0807` runs at ~1.25e-4 read `second_diff_rel` ≈ 3.6e-2, and
`(1.25e-4 / 1.4e-6)² ≈ 8000×` against the ~3300× actually seen.

**The fix is a sign, and it was already being logged.** `loss_delta_rel` splits
the two explanations, which is what `step_probe.py`'s own docstring said it was
for — but the fit collapsed them anyway:

| second difference | `l1` vs `l0` | Means | Status |
|---|---|---|---|
| < 0 | `l1 < l0` | loss falls monotonically and is *accelerating* down — the minimum is beyond `α = span`. **"Your step is too small."** | **`beyond`**, α\* = span (a lower bound) |
| < 0 | `l1 ≥ l0` | concave *and* the step raised loss. No basin, no descent — the local model really is wrong | `downward`, α\* = nan |

`beyond` counts as a **usable** reading, not a bad one. It is one-sided — α\* is
only bounded below — but it is correct, and it is the reading that dominates
exactly where a deliberately-low seed puts the probe. Counting it as bad is what
made the loop unable to license the growth it was seeded low in order to perform.

**Consequence for reading the log:** `fit_beyond_rate → fit_ok_rate` as the servo
climbs is the signature of a working loop. A `beyond_rate` stuck at 1.0 means the
LR is still below the probe's resolving range; an `ok_rate` near 1.0 means the
probe is bracketing the basin and α\* is a two-sided measurement.

**Consequence for §A6's objection:** it sharpens rather than answers it. α\* at
low LR does not say *how far* to grow, only *that* growth is licensed — so the
"affirmative permission" argument for evidence-led growth survives, but its
quantitative half does not apply until the probe is in the `ok` regime.

## A4c. Two ways to spend `α*` — servo, or line search

*(added 2026-08-06.)*

**First, the thing that was actually asked, and its answer: ❌ no.** Can one of
the three probe evaluations double as a "free backprop" — run with grad enabled
so its gradient feeds a training step? Mechanically yes: `α = 1` is `θ_after`,
the probe restores there, so a gradient taken at that point stays valid. But the
saving does not survive contact with the arithmetic. A backward pass is ~2× a
forward, so **the forward already paid for is the cheap third** of a
forward+backward. The saving is ~⅓ of one branch-step once per `cadence`,
against a fused step costing ~3 branch-steps plus a rollout and energy calls —
**under 1% of training compute**, in exchange for the probe contributing
gradients to the training path, which is precisely the property that makes stage
1 incapable of destabilising a run. Not worth it. *(Recorded so it is not
re-proposed; the cost lever here is cadence.)*

**Second, a different idea that came out of misreading the first one, and which
does stand on its own.** The probe evaluates three step lengths and then throws
the answer away — restoring to `α = 1` when it has just computed, for free, that
`α*·Δ` was the better step.

Using that discarded answer is not a cost optimisation — it is a different
control architecture:

| | **(a) Servo** (§A4, as written) | **(b) Line search** |
|---|---|---|
| Acts on | windowed **median** `α*` | this probe's **single** `α*` |
| Adjusts | `peak`, affecting *every* step | the step itself, on probe steps only |
| Speed | slow, many probes to move | immediate |
| Noise exposure | low — median is the robust statistic | **high — §A3a.2 explicitly says only the windowed median is trustworthy, per-probe dispersion may be wide** |
| Precedent | ordinary LR scheduling | Vaswani et al. 2019, stochastic Armijo (§A7) |

**The objection to (b) is the dispersion one, and it is the same objection §A3a.2
already raises.** Acting on a single-batch `α*` is acting on exactly the reading
the doc says not to trust. There is also an Adam interaction: scaling the taken
step by `α*` applies a per-step LR multiplier the optimizer does not know about,
so its moment estimates describe a trajectory that was not followed. Neither is
fatal; both are real.

**(c) The version worth building — one-sided backtracking.** Intervene only when
the step was *bad*:

```
if α* < backtrack_floor:      # e.g. 0.5 -- overshot by 2x or worse
    θ ← θ_before + α*·Δ       # take the better step, free
else:
    θ ← θ_after               # normal step, probe was just a sensor
```

This gets the safety benefit of (b) without acting on noise in the common case:
near `α* ≈ 1` it does nothing at all, so ordinary dispersion is ignored by
construction, and it fires only on readings far enough from 1 that noise cannot
plausibly explain them. It composes with (a) rather than replacing it — (c)
catches individual bad steps, (a) moves the operating point.

**Decide after stage 1**, on the measured dispersion: if per-probe `α*` is tight,
(b) is available and simplest; if wide, (c) is the safe way to use the same
information; either way (a) still wants the median.

## A5. What gets deleted ✅ **DONE 2026-08-08**

The middle layer, per `decisions.md` D4 and
`module_lr_controller.md` S1/S2: **cut tier as an actuator, latch, hot clock,
recovery ramp, cut-factor AIMD.** ~200 lines. Both documented deadlock modes
([[lr-tripwire-deadlock]]; F1 arithmetic inertness) live entirely inside it and
went with it. `controller.py` is 449 → 386 lines and the deleted concepts —
`_cut_factor`, `_latched`, `_last_hot_step`, `_pre_trigger_cold`,
`_recovery_anchor`, `_channel_cooldown_until`, `reset_spike_monitors` — have no
successor.

**The "logged warning only" cut tier was NOT kept, and that is a deliberate
departure from this section as written.** Keeping it means keeping a second,
graduated bar that has to be calibrated — the exact thing D4's *"if we are only
looking at hard blow-ups we can use almost any metric"* retires. The diagnostic
it was for ("this arm ran hot") is already served without it: `grad_norm_pre_clip`
is logged every step, and `lr_ctrl/peak_scale` now records what the LR actually
did, which is strictly more informative than a fire count. `_check_bars` instead
**refuses** a divergence bar below 1e5, so a config cannot quietly reintroduce a
graduated tier under the surviving key's name.

### What explicitly SURVIVES the deletion *(user 2026-08-06)*

> *"We should also keep checkpoint reloads / LR reductions on legitimate loss
> explosions — NaN/Inf gradients or values > 1e9 or something."*

**Both actions, together, on one coarse trigger.** This is not a partial walk-back
of D4 — it is the half D4 always kept, stated explicitly because the deletion
list reads like it takes everything:

| | |
|---|---|
| **Trigger** | non-finite gradient or loss, **or** either exceeding an absolute bar (~1e9). Deliberately coarse — per D4, *"if we are only looking at hard blow-ups, we can use almost any metric,"* so this bar needs no calibration and must never be tuned into a graduated one |
| **Action** | reload the checkpoint **and** cut `peak`, recording the ceiling (§A4) |

The pairing is the point. A reload without an LR cut re-enters the same state at
the same LR and explodes again; an LR cut without a reload keeps the damaged
weights. The middle layer's failure was never that it did both — it was that it
did them on a *graduated* trigger, with a latch and a recovery ramp, in a regime
the servo now covers continuously and at zero false-positive cost.

So the division of labour is exactly two regimes, as D4 sets out: **α\* < 1 →
smooth `peak` nudge, nothing discarded**; **explosion → reload + cut, progress
since the last checkpoint discarded.** Nothing in between, and the second one
keeps its existing machinery (reset tier, rewind, `max_reloads`, frozen
detector).

## A6. The objection this must survive

`questions.md` §C answered **no adaptive scheme** (now superseded by D3), on aug02: gains are
steep up to a ceiling and **death** above it, so *"within-run upward probing is
not available — overshoot isn't recoverable, and aggressive clipping doesn't
make it so (the 8e-4 arm died at clip 135 with pre-clip grad 587× the clip)."*

That objection is correct **about blind probing**, which is the only kind then
on the table: a blind climber learns the cliff is there by going over it. The
proposal's claim is narrower and different — that `α*` degrades *before* the
cliff, giving lead time a blind scheme cannot have.

**That claim is not established and it is the crux.** It is also cheap to
falsify:

> **Decisive experiment.** Run the probe **read-only** (compute and log `α*`,
> actuate nothing) across the existing aug02 LR ladder — 4e-4 healthy, the
> cliff before 8e-4 ([[aug02-battery]]). Plot `α*` against set LR.
>
> - `α*` falls smoothly toward 1 approaching the cliff → the servo has lead
>   time; build it.
> - `α*` sits at ~1 right up to the cliff → **the proposal is dead**, the cliff
>   is not a curvature phenomenon, and the ladder-from-`stage_start` is the
>   right answer. Keep §A5's deletions and §A8's fallback and stop there.
>
> Read-only instrumentation on a battery that already ran. No new risk, no new
> arms, and it settles a question that has been open across four batteries.
>
> **Log at the same time** (§A3a.2, §A3a.3, free): windowed-median stability of
> α\*, per-probe dispersion, and the rate of downward-opening parabola fits.
> Either of the latter two firing voids the sensor independently of lead time.

Do this before writing any actuation.

**The bar this has to clear is set by tolerance, not precision.** With a ~5×
band the servo does not need to track the optimum — it needs to (i) not sit
*dramatically* low and (ii) stop short of the cliff. So "α\* degrades before the
cliff" needs only enough lead time to halt growth within a factor of ~2, which
is a materially weaker requirement than the §A3 framing implies.

One live outcome worth naming: the aug02 arm died at pre-clip grad **587× the
clip**. If that cliff is a rare-large-batch instability rather than sharpness
growth, typical-batch α\* stays healthy right up to death and the probe sees
nothing. That is the favorite failure branch, not a formality.

**Note the cross-link to Part B.** That same 587× figure is a gradient-*outlier*
death, which is exactly what bounded per-row influence (§B5, κ=1) prevents. If
the probe sees nothing because the cliff is an outlier phenomenon rather than a
curvature one, that is not only a verdict against the servo — it is positive
evidence for the prioritised estimator, and §B10's LR-ceiling arm becomes the
one that matters.

## A7. Literature warrant, and its limits

Asked directly whether the optimum sits 2–4× below the "hot" threshold. The
literature mostly says **no — the margin is small** — but with a caveat that
matters more than any of it.

- **Edge of Stability** (Cohen et al., ICLR 2021): full-batch GD rises *to* the
  2/λ_max threshold and hovers, training fine with non-monotone per-step loss.
  Local loss increases are the normal regime, not a warning.
- **Adam at EoS** (Cohen et al. 2022): same on preconditioned sharpness;
  empirical threshold ~38/lr at β1 = 0.9.
- **Catapult** (Lewkowycz et al. 2020): lazy < 2/λ, catapult 2/λ→~4/λ, then
  divergent — and the catapult phase *often generalises best*. The optimum can
  sit **above** where single steps start increasing loss.
- **Against:** Smith's LR range test and the `lr_find` heuristic both say back
  off 3–10×. But from **divergence**, not from non-monotonicity — different
  thresholds, separated by roughly the catapult window. Once that is controlled
  for, the two literatures broadly agree.
- **Closest precedent for §A3:** Vaswani et al., *Painless Stochastic Gradient*
  (NeurIPS 2019) — SGD with stochastic Armijo line search, robust in practice.
  **Its guarantees lean on an interpolation assumption that TB with a moving Z
  target badly violates.** Precedent, not proof.

**The limit.** None of this studies a non-stationary target with a replay
buffer. Our own record already contradicts the standard picture in one place:
16× LR moves the *entry level*, not the slope ([[invariant-convergence-rate]]).
Treat every number above as directional. *(Warrant: **inherited** throughout.)*

## A8. Fallback if §A6 kills it, or if the probe is not worth building

Keep grad-norm bars, but estimate them from a **quantile updated only on cold
readings** — frozen while hot. That repairs both s706frkh failure modes for
free and costs nothing. Strictly better than absolute bars; strictly worse than
`α*`, because it still measures gradient magnitude rather than step size.

Worth wiring alongside it: the orphaned `uw_global`/`uw_max` instrumentation
noted in `decisions.md` §2c. It and `α*` are the two candidate ceiling-free
parameterisations; `α*` is the stronger one only because it has a setpoint.

**Status 2026-08-06: BUILD.** *(user, `decisions.md` D3 — supersedes the
2026-08-05 "design deferred" status.)* Proceed with §A3/§A4 as proposed.

Three consequences:

- **§A6 demotes from kill-gate to instrumentation.** Run the probe read-only to
  set `α_target` and validate the §A3a.2/§A3a.3 diagnostics — not to decide
  whether to build. The honest residue stands: if α\* *is* flat right up to the
  cliff, the servo cannot prevent it, so the probe now scopes whether a separate
  hard ceiling is needed rather than whether the servo happens.
- **§A5's deletion is resolved *by* this decision, not independently**
  (`decisions.md` D4). The middle layer occupied a third regime — "cut hard and
  latch, but don't reload" — which has no distinct response left once the servo
  cuts continuously on a better sensor. Slightly hot → α\* nudges `peak` down at
  ≈ zero false-positive cost; diverged → coarse bar → reload. Two regimes, two
  owners, nothing in between. This is also why the reset bar needs no precision:
  *"if we are only looking at hard blow-ups, we can use almost any metric."*
- **§A4's envelope loses its decay leg** — see §A4a.

---

# Part B — Batch construction

Sharpens `synthesis.md` §4a from an argument into a derivation, and reframes
three open questions (`decisions.md` §2c: replay's structural place, `beta: 10.0`,
replay freshness) as one design problem with one answer.

**The reframe.** The replay buffer's purpose is not "tame the tails of the
forward residual." It is **push `fwd/tb_err` down as fast as possible**. That
makes the sampling distribution a *design measure* over a regression problem,
which can be derived rather than tuned.

**The bar** *(user 2026-08-06)*: **standard on-policy TB.** That is what the new
batch construction and loss weighting have to beat, and it is the comparison every
arm in §B10 should be read against — not against the current tuned mixture, which
is neither a clean baseline nor what anyone else runs.

Scope note from the same conversation: **§B5a (the same estimator on `bwd`) is an
option to try, not a commitment**, and **§B7 population management is the richest
part** of this — with the hazard purge singled out as the piece that is already
right and elegantly simple, so leave it alone.

Full derivation of the estimator: `prioritised_tb_estimator` (artifact,
2026-08-05).

## B1. δ is a log-ratio

Let `Q(τ) = P_F(τ)` and `P(τ) = R(x_τ)·P_B(τ|x_τ)/Z`. Both are proper
distributions over trajectories. Then the TB residual is exactly the pointwise
log-ratio:

```
δ(τ) = log Q(τ) − log P(τ)
logw = log Z − δ                     (the code's `resid` is δ)
```

Two identities follow, true at every θ, for any valid `P_B` — learnable or not,
since they are statements about `P_B` being a normalised conditional over paths
given `x`, not about whether its parameters move:

```
E_Q[e^{−δ}] = Z_true / Z_learned            E_P[e^{+δ}] = Z_learned / Z_true
```

Three things follow:

**(a) A free, exact Z diagnostic.** `log E_Q[e^{−δ}] = log Z_true − log Z_learned`
on any fresh forward batch, no ground truth required. This is `z_emp −
log_Z_learned`, already computable. The backward branch gives the same number
with the opposite sign (approximately — the prior buffer only approximates `P`),
so the two branches cross-check each other.

**(b) The Jensen gap is a KL.**

```
z_jensen = E_Q[logw] = log Z − KL(Q‖P)
z_emp    = log Z                          (exact)
z_gap    = KL(Q‖P)
```

`z_gap` is not a Z-estimation diagnostic; **it is the reverse KL from sampler to
target** — i.e. the actual objective. And `E_Q[δ] = log Z_learned − z_jensen`,
which at a correct `log Z_learned` equals `KL(Q‖P) ≠ 0`.

**(c) Each branch is exponentially blind to half the residual field.** By Markov
on the identities:

```
Q(δ < −m) ≤ e^{−m}          P(δ > +m) ≤ e^{−m}
```

An on-policy batch essentially cannot contain a strongly negative residual. This
upgrades §4a's "the two blind spots are complementary" from an argument to a
theorem with a rate.

## B2. Scope: which half this covers

| | δ > 0 (policy over-weights) | δ < 0 (policy under-covers) |
|---|---|---|
| Visible to | on-policy / replay | backward / prior / anchor |
| Objective | `fwd/tb_err` | coverage metrics |
| Covered here | **yes** | no |

Corollary worth stating flatly: **no amount of `tb_err` reduction certifies
coverage.** `E_Q[δ²]` weights the coverage half by a measure that is
exponentially small there. That is why every coverage mechanism had to be bolted
on separately rather than falling out of the loss — a scoping fact, not a defect.

Consequence for the replay buffer specifically: it is fed by on-policy rollouts,
so it *inherits Q's blindness* and cannot contain strongly-negative-δ rows. It is
structurally a δ>0 instrument. Not by choice — by intake.

## B3. Why a buffer, stated as a rate

A region with on-policy mass `q` and residual `δ` contributes `q·δ²` to `tb_err`
but is *visited* with probability `q`. Under Huber each visit delivers gradient
≈ `beta`, so closing the gap takes `~δ/(η·beta·q)` **steps**. Time-to-converge
blows up as `1/q` while the loss contribution shrinks only as `q` — so the tail
is simultaneously a real fraction of `tb_err` and the slowest thing in the system.

Store-and-replay converts the visit rate from `O(q)` to `O(1)`. **The buffer is a
1/q rate multiplier on the slowest part of the residual field**, not a variance
trick. This is the original motivation and it survives intact; what follows
generalises it from "the tail" to every residual level.

## B4. The force spectrum Φ — the design principle

What the optimiser consumes each step is `ĝ = (1/B) Σ w_i·ℓ'(δ_i)·g_i`. Bin by
residual level and ask how much total force arrives from level δ:

```
Φ(δ) = m(δ) · w(δ) · ℓ'(δ)
       ↑        ↑       ↑
     draw    weight   loss shape
```

**The expected gradient depends only on the product.** Draw a level twice as
often and push half as hard → identical expected gradient. Three dials, one
function. They are *not* equivalent in variance: rare-and-hard is noisy,
frequent-and-soft is quiet.

So the design splits in two, and the split is the whole point:

- **What should Φ be?** — sets the fixed point. Bias lives here.
- **How is Φ realised?** — sets the noise. Cannot move the fixed point.

**Diagnosis of the current system: `beta` is chosen for variance reasons but acts
on Φ; buffer composition is chosen for Φ reasons but acts on m.** Each knob is
aimed at one job and lands mostly on the other. That cross-coupling is why both
have felt untunable.

Worked example — 1000 rows, 999 at δ=1, one at δ=30:

| scheme | p(tail row) | its push when drawn | its share of force |
|---|---|---|---|
| uniform, quadratic | 0.001 | 30 | 0.030 |
| uniform, Huber β=3 | 0.001 | 3 | **0.003** |
| p∝δ, IS-corrected, quadratic | 0.029 | **1.03** | 0.030 |

Huber and prioritisation both cap the per-sample push. Huber does it by cutting
the tail's share of the force 10× (Φ changed, log Z recalibrates to a winsorised
mean). Prioritisation does it by drawing 29× more often at 1/29 the weight (Φ
identical, unbiased). **Bounded per-sample push is a variance goal, so meet it
with the variance dial.**

## B5. Proposed replay design

Target measure `μ* ∝ Q_now(τ)·δ₊·‖g‖`. Buffer rows sit at unknown density
`μ_buf`, but if `log P_F` is stored **at admission**, the bridge back to current
on-policy is one float per row:

```
Δ_i = log P_F_now(τ_i) − log P_F_admit(τ_i)        Q_now/Q_admit ≈ exp(Δ_i)
```

Then the draw and weight are:

```
draw    p_i ∝ exp(clamp(Δ_i, ±Δ_max)) · δ₊,i^κ           δ₊ = max(δ, 0)
weight  w_i ∝ 1 / δ₊,i^κ                                  (self-normalised)
loss    quadratic on this branch
```

The drift cancels out of the weight and lives entirely in the draw. Properties:

- **κ is the Φ principle as one number.** It moves shape between draw and weight
  and **provably does not change Φ**: at every κ the expected gradient is the
  square-loss on-policy gradient `E_Q[δ·g]`. κ=0 is uniform draw with full
  magnitude spread; κ=1 makes every drawn row push equally hard. A κ sweep is
  therefore a *pure variance ladder* — any difference observed is estimator
  variance, nothing else. Variance is exactly minimised at κ=1 by
  Cauchy–Schwarz (`S_κ·S_{2−κ} ≥ S₁²`), and the ladder's whole profile is
  computable from the residual histogram before running anything.
- **One-sided by design.** `δ₊`, not `|δ|`. A row the policy has abandoned has
  small `log P_F` and therefore strongly *negative* δ — under `|δ|` priority it
  would be re-promoted into a batch serving an objective that does not want it.
  Negative-δ rows are mode-retention signal and belong to the backward
  instrument (§B5a).
- **`Δ_max` does two jobs with one constant.** It clamps the draw's dynamic range
  *and* is the eviction threshold: a row whose drift exceeds it has left the
  policy's support, which is precisely when it should be routed out. The clamp
  and the eviction bar cannot disagree because they are the same number.
- **Known omission: `‖g‖`.** Per-sample gradient norms are expensive, and rows
  deep in the tail have both large `|δ|` and large `‖g‖`, so `δ₊` alone
  under-corrects. Deferred — the variance of estimating `‖g‖` likely exceeds the
  gain. Revisit only if the κ ladder shows the design measure is worth
  sharpening.

**Priority must be refreshed, and it is nearly free.** `log R` is fixed per
stored trajectory; `log P_F` **and `log P_B`** need recomputation — one forward
pass, no reward call. *(Corrected 2026-08-06: this said `log P_B` was fixed too.
It is not — `learn_pb: true` is live, so it moves for a fixed `τ`. See §B5c.)* Stale priority is not a bias (IS corrects for whatever `p`
was actually used) but it is a worse design measure. Refresh on draw at minimum.

**This is the structural advantage over PER.** Prioritised Experience Replay's
central practical defect is stale priorities — TD targets cannot be cheaply
recomputed, which is why it needs `α<1` and an annealed correction. TB residuals
are exactly refreshable, so that whole class of compromise does not apply here.

## ~~B5a. The same estimator on the backward branch~~ ❌ **DROPPED 2026-08-06**

Killed by the conjunction of two results, neither of which existed when it was
written:

- **§0a (measured):** the prior buffer's `1 + CV²` is **1.06–1.13**. Prioritised
  sampling can buy ~6% there. Its residuals are homogeneous — `std ≈ 4.2` against
  `mean ≈ 15–18` — so there is no tail to over-sample.
- **§B5b / D28 (derived):** its stated benefit, restoring within-tail ordering,
  is not something prioritisation delivers. The ordering is lost to **Huber**,
  and it comes back by running quadratic. Uniform draw + quadratic already gives
  `Φ ∝ δ`.

No variance case and no ordering case. §C step 4 and §B10's bwd-priority arm are
removed with it.

**What survives, and it is the useful half:** the observation that a winsorised
`bwd` branch gives *a mode 80 nats out identical drive to one 10 nats out*. That
is real. The fix is `beta` inactive on bwd (D6 gives permission), not a
prioritised draw.

<details><summary>Original section, kept for the P_B / terminal-averaging
argument, which stands on its own</summary>

*(added 2026-08-06 — to try, in addition to the replay work above)*

The estimator is not replay-specific. It needs a pool larger than the batch, a
computable per-item residual, and a known draw probability. **`bwd` has all
three** — it draws from the prior buffer through the same `_loss_weights` path —
so implementing it for replay gets most of the way to implementing it here.

**Only the sign changes.** Replay serves the over-weighted half and prioritises
`δ₊`. `bwd` is the coverage instrument and prioritises `δ₋ = max(−δ, 0)`.

**Why it may matter more here than on replay.** Measured `logw_std ≈ 21` against
`beta: 10.0` puts the knee at roughly *half* a standard deviation of the residual
distribution, so most of the backward tail is winsorised and **a mode 80 nats out
receives identical drive to one 10 nats out**. Prioritised draw restores the 8:1
ratio — 8× the draw frequency, each contributing the same bounded `⟨|δ|⟩` — with
per-row influence still bounded. Note this is a **correctness** effect, not a
variance effect: no importance-sampling result predicts its size, because it is
specific to the winsorised loss.

Three consequences that come free:

- **No adaptive bar, so no ratchet.** A batch-quantile knee preserves tail
  ordering but compares each residual against a distribution-derived bar which
  then feeds back. `p_i = |δ_i| / Σ|δ_j|` is a normalisation, not a threshold —
  nothing is compared to anything. Price this against
  [[fwd-calibration-floor-creeps]] / [[controller-ratchet-marginal-breach]] and
  it comes out ahead of the quantile knee on the one axis those memories care
  about.
- **The Jensen leak closes.** `clip` is convex, so per-sample clipping gives
  `E[clip(r)] ≥ clip(E[r])` and high-variance terminals leak absorption drive in
  proportion to their spread. With no clip the estimator is linear in `δ` and
  there is no convexity to leak through.
- **Saturated `bwd` TB stops silently becoming MLE×`beta`.** Whether MLE is
  wanted on this branch is a real question; it stops being something that happens
  by accident whenever the buffer sits more than half an SD under-covered.

**One prerequisite is mandatory here, unlike on replay.** `learn_pb: true` is
live, so `δ` contains `−log P_B` with a learnable term. Single-path `|δ|` then
partly prioritises a *path* residual that the `P_B` update is about to remove on
its own — a moving target. **Priority must be on the terminal-averaged `δ̄(x)`
over the K repeats**, which composes with `tbc` (already owning the intra-terminal
spread) and with `log_pf_estimate`'s existing IWAE machinery.

**Extra confound to measure on this branch.** Since `∂L/∂ log P_B = −ℓ'(δ)`, a
learnable `P_B` can shrink `δ` by making the observed path more likely rather than
by the sampler improving. At convergence that is the *good* outcome — the
variance-minimising `P_B` for fixed `P_F` is its exact reverse, at which
`Var_τ(δ|x) = 0` — but en route it means a falling `tb_err` may be `P_B`
converging. Discriminate with `E_x[Var_τ(δ|x)]` (path noise, `P_B`'s job) against
`Var_x(δ̄(x))` (terminal mismatch, `P_F`'s job); `bwd/logw_std_within` and the
`tbc` loss value already carry both.

</details>

## B5b. IS correction and Huber do not mix — go quadratic on any prioritised branch

*(added 2026-08-06; **rewritten the same day** — the first version framed this as
a corrected-vs-uncorrected decision, which was wrong. User: "I had also thought
that beta would be much larger (or just quadratic) on the new weighting scheme."
That is the answer, and it dissolves the decision.)*

Two quantities, both functions of δ:

```
Φ(δ)            = m(δ) · w(δ) · ℓ'(δ)        total force arriving from level δ
per-row push    =        w(δ) · |ℓ'(δ)|      what ONE drawn row contributes
```

| draw `m` | weight `w` | loss `ℓ'` | `Φ(δ)` | per-row push |
|---|---|---|---|---|
| uniform | 1 | `δ` | `δ` | `δ` — unbounded |
| uniform | 1 | `clip(δ,β)` | `β` — **flat, ordering lost** | `β` |
| `δ^κ` | `δ^−κ` | **`δ`** | **`δ` — linear** | **`δ^(1−κ)` → constant at κ=1** |
| `δ^κ` | `δ^−κ` | `clip(δ,β)` | `β` — flat | **`β·δ^−κ` — DECREASING in δ** |
| `δ` | 1 | `clip(δ,β)` | `βδ` — linear | `β` |

**Row 3 at κ=1 is simply correct**, and it is what §B5 already specifies for
replay ("quadratic on this branch"). Linear Φ, ordering intact, every drawn row
pushing equally hard, unbiased, provably fixed Φ across the κ ladder. There is
no trade-off to make and no branch asymmetry needed in the draw.

**Row 4 is the trap, and it is the real finding.** Combining the IS correction
with a Huber loss at κ=1 makes per-row push `∝ β/δ` — **a mode 80 nats out
pushes eight times *less* than one 10 nats out.** That is worse than either
mechanism alone: Huber flattens Φ, and the IS weight then divides on top of an
already-flattened `ℓ'`. The two corrections stack in the same direction because
both shrink the tail.

> **Rule: any branch running a prioritised draw must run quadratic, or a `beta`
> large enough to be inactive over the residual range.** `logw_std ≈ 21`, so
> "inactive" means `beta` well above ~60, not 10.

This is the cleanest form of §B5a's ordering argument: it was never an argument
for dropping the IS correction, it was an argument against **winsorising a
branch whose draw is already prioritised**. And it sharpens D6 — branch-
asymmetric `beta` remains available, but the asymmetry it wants is *`beta`
inactive wherever prioritisation is on*, not two different knees.

**Consequence for §B10:** the κ ladder keeps its clean null (Φ fixed at every κ,
so any difference is estimator variance alone) **provided the branch is
quadratic**. Run the β×κ arm's `beta: 10` cell and it is measuring row 4 — the
decreasing-push pathology — not a `beta` effect.

<details><summary>Superseded first version — kept because the algebra is still
the reason, and because the corrected/uncorrected split it proposed is a real
alternative (row 5) that is simply worse than row 3</summary>

Row 5 (uncorrected draw `p ∝ δ`, weight 1, Huber) also gives linear Φ with
bounded push, so it *works* — but it is biased with respect to the objective,
loses the κ ladder's clean null, and buys nothing row 3 does not already have.
The original framing presented rows 3-with-Huber and 5 as the only options,
having mis-tabulated row 3's per-row push as `δ` (true at κ=0, not at κ=1).

</details>

## B5c. What is stored, what is recomputed, and which "weight" is which

*(added 2026-08-06. §B5 uses "weight" for two unrelated corrections and never
says where either one lands, which is a doc defect, not a reader's error.)*

**Three quantities, three jobs.**

| | Definition | Job | Recomputed? |
|---|---|---|---|
| **`δ`** | `log P_F − log P_B − log R + log Z` — the TB residual, all terms **current** | **priority** `p ∝ δ₊^κ` and the prioritisation weight `w ∝ 1/δ₊^κ` | **every draw** |
| **`Δ`** | `log P_F_now(τ) − log P_F_admit(τ)` — **`P_F` only** | (i) correct the draw for policy movement since admission; (ii) evict on loss of support | every draw |
| **`log R`** | reward of the stored terminal | enters `δ` | **never** — energies are fixed per stored trajectory |

**The two "weights" are different objects and they live in different places:**

- `w ∝ 1/δ₊^κ` corrects the **prioritised draw** — it undoes `p ∝ δ₊^κ` so the
  estimator stays unbiased. This is the κ ladder.
- `exp(Δ)` corrects **staleness** — that the row was drawn under an older policy.
  It goes in the **draw**, not the weight, which is why §B5 says "the drift
  cancels out of the weight."

**Why `Δ` uses `P_F` alone and not `log P_F − log P_B`.** Self-normalised
importance sampling over a buffer targets the measure `p_i · w_i · μ_buf`. We
want that `∝ Q_now`. Under uniform admission (D5) `μ_buf ∝ Q_admit`, so the
required factor is

```
Q_now / Q_admit  =  P_F_now(τ) / P_F_admit(τ)
```

because **`Q(τ) = P_F(τ)`** — the trajectory density of the sampler *is* the
forward policy. `P_B` is the backward decomposition that defines the *target*;
it never generated anything, so it has no place in a ratio between two sampler
densities. Putting it in would mis-correct by however far the backward network
happened to move, which says nothing about which trajectories the policy now
produces. Same reason `Δ_max` doubles as the eviction bar: "has the policy
stopped producing this row" is a `P_F` question.

### ⚠ The error this exposes

§B5 states: *"`log R` and `log P_B` are fixed per stored trajectory; only
`log P_F` needs recomputation."*

**`log P_B` is not fixed.** `learn_pb: true` is live
([mk_dev.yaml:135](../configs/mk_dev.yaml:135)), so `log P_B(τ|x)` moves as its
network trains, for a completely fixed `τ`. Only `log R` is genuinely frozen.
§B5a already knows this ("`δ` contains `−log P_B` with a learnable term") and
§B5 contradicts it.

So the correction is to `δ`'s refresh, not to `Δ`: **both** `log P_F` and
`log P_B` must be recomputed each draw. That is very likely free — they are
produced by the same trajectory-scoring pass — so §B5's "one forward pass, no
reward call" cost claim survives; it is the *description* that is wrong. **Verify
against the scoring path before relying on it.**

## B6. What this does to `beta`

`beta` is doing two jobs. Prioritised-IS retires one completely and the other not
at all:

- **Retired: per-sample magnitude variance.** One δ=100 row dominating 300 rows;
  and the Adam side effect where a rare spike inflates the second-moment EMA and
  produces a step-size dead zone for `~1/(1−β₂)` steps. After correction,
  per-sample magnitudes are flat by construction.
- **Not retired: loop gain.** δ is not a fixed regression target — it is a
  self-consistency condition coupling `log Z` and `log P_F` along the whole
  trajectory through one network. It is a fixed-point iteration whose loop gain
  is proportional to the applied force. Prioritised-IS preserves Φ exactly and
  therefore preserves the loop gain exactly. Oscillation risk is unchanged.

So **do not retire `beta` wholesale.** Relocate the loop-gain job to a control
that does not bias the level:

- Gradient-norm clipping is a *positive rescale of the whole gradient*, and
  positive rescaling does not move the zeros of a vector field — the fixed point
  is unchanged. Huber changes `ℓ'` itself, differentially by residual level,
  which is exactly how it shifts `log Z`. Same damping job, no level cost.
  **It is not a substitute for bounded influence**, though: a scalar on the
  *summed* gradient still lets one extreme row own the direction. Bounded
  influence can only be imposed per-row, pre-aggregation — which is what
  prioritisation does.
- **Minimal-change version, do this first:** keep Huber on the replay branch with
  `freeze_z` on, run the forward branch quadratic (or much larger `beta`) with Z
  live. If Z never sees the winsorised residual, winsorisation cannot bias the
  level. Config change, not a redesign.

**The A/B that says which job `beta` is currently doing:** raise it and watch
`grad_norm` against stability. Rises but stays stable → variance work,
prioritisation replaces it. Oscillates at unchanged `grad_norm` → loop-gain work,
keep the damping but move it to the global control. Cheaper once prioritisation
is in, because magnitudes are already flattened before `beta` is touched.

**Saturation and priority are the same knob.** If Φ is declared to saturate, `p ∝
|clip(δ,±β)|` is flat past β, so within-tail ordering is gone — *necessarily*,
because a saturating objective is the claim that those rows are equally
important. Which resolves the branch tension: fwd/replay saturate because
near-singular rewards make magnitude uninformative (a δ=3000 and a δ=300 clash
are the *same* policy error differing only in how steeply the potential rises);
bwd stays linear because a curated buffer's deep residual is a real uncovered
mode. **A branch-asymmetric `beta` with a reason** — not "outliers differ across
branches" but "the branches hold different beliefs about whether magnitude
carries information."

## B7. Population management: what survives

Under the current draw, composition **is** the design measure — and more
completely than first stated. **The replay draw is uniform**:
`draw_replay_sample` passes `weighted=False` ([train.py:3038](train.py:3038)),
so `_loss_weights` is never consulted on the replay path at all. Every bit of
replay's prioritisation happens at admission, and every eviction decision is a
direct edit to Φ with no correction anywhere. That is why population management
has been vexing.

(`_loss_weights` *is* live on the `bwd` path, gated by `weighted_bwd_sampling` at
`temperature: 0.5`, `beta: 0.9` — 10% of the bwd batch weighted, 90% uniform.
There its min-max-then-softmax gives a contrast ratio of `exp(1/T)` ≈ 7.4 at
T=0.5 *regardless of whether the residual range is 2 nats or 200*, and a single
outlier compresses everyone else toward uniform. §B5a replaces the priority it
reads; this normalisation should go with it.)

With prioritised-IS, a low-residual row is drawn rarely and contributes
proportionally when drawn. **Keeping it costs memory, not gradient budget.**

| Question | Before | After |
|---|---|---|
| Weighting | eviction policy | `p_i`, derived |
| Support | age / hazard | measured drift `Δ_i` vs `Δ_max` |
| Capacity | `max_size`, churn | memory + refresh compute only |
| Routing | — | δ<0 rows → backward instrument |

**Admission is the thing that has to change.** Every selective step multiplies
into the buffer's density — `μ_buf ∝ Q_admit · p_admit · p_survive` — and the
weight must divide by all of them, not just the draw. The current `|resid|`
softmax with a health-modulated cap ([train.py:4734](train.py:4734)) both
double-counts the residual (it enters Φ twice, once stale) and breaks the drift
correction (which assumes admission was unbiased w.r.t. Q). **Admit uniformly
from the sane pool** — or keep selection and record `p_admit` to divide out. Five
underived constants (`admit_cap_max/min/h0`, `admit_temperature`) retire either
way.

**Purge is already close to right.** The hazard is uniform-random, so
`p_survive` is unbiased and drops out. `stalled` is already the learnability
criterion. `floor` stays — and note a row driven to δ≈0 *by repeated replay* is
δ≈0 at that exact trajectory, which is memorisation; the low value is a symptom,
not just dead weight.

**The population dynamic, and the one number that governs it.** A resident row's
residual relaxes at rate λ (from being trained on) and is removed at hazard rate
1/τ. Everything follows from **λτ**:

- `λτ ≪ 1` — buffer ≈ on-policy residual distribution. A delay line; composition
  is Q; the drift correction is exact.
- `λτ ≫ 1` — rows are corrected before they leave. The buffer fills with δ≈0
  rows that have memorised, and since a corrected row has `p ≈ 0` under the
  prioritised draw it stops being sampled and just occupies a slot until the
  hazard finds it. `N_eff/N ~ 1/(λτ)`. You pay memory and refresh on a mostly
  dead population.

So **τ should be set from measured λ**, not hand-picked — and λ is one division
from `replay_buffer_live_delta_mean`.

### B7a. Servo it, don't set it *(user 2026-08-06)*

> *"It seems to me this could be dynamically measured and controlled mid-run?
> This is exactly where the anti-overfitting servo comes in as well.
> Anti-overfitting means increasing input rate and/or the total buffer size."*

Yes — and this is what completes D22. The pieces are all present: **sensor** =
the precept below (`weighted_replay_err ≥ fwd_err`, null derived, no calibration),
**state** = `λτ` measured online, **actuator** = intake. No hand-set constant
anywhere in the loop, which is the property the old `bar`/`release` servo lacked.

**One caveat on measuring λ, and it is the familiar one.** τ changes the
population composition, and the population is what λ is measured over — longer τ
means more corrected rows means a lower apparent λ. That is a loop with its own
measurement inside it, the same shape as [[controller-ratchet-marginal-breach]]
and [[fwd-calibration-floor-creeps]]. **Measure λ on a birth cohort** — rows
admitted within the last `k` steps — which is τ-independent by construction.

**But one of the two actuators is not a lever.** Under a binding capacity and
uniform draw, Little's law gives `N = rate × τ`, and a row's draw frequency is
`B/N`, so `λ ∝ B/N`. Then

```
λτ  ∝  (B/N) · (N/rate)  =  B / rate
```

**`N` cancels.** Buffer size does not appear in the memorisation product at all —
it buys diversity and costs memory, but it cannot move `λτ`. The only lever is
the ratio of **batch size to intake rate**, so anti-overfitting is *raise intake*
(or shrink the batch), not *grow the buffer*.

Two qualifications. Under the prioritised draw a corrected row has `p ≈ 0` and
stops being drawn, so `N → N_eff` and the mechanism is partly **self-limiting** —
prioritisation is itself anti-memorisation, which is a further argument for it.
And the cancellation assumes the cap binds; if occupancy sits below `max_size`,
τ is set by the hazard instead and the algebra changes. Occupancy currently
equilibrates at *exactly* `max_size` on both buffers, so the binding case is the
live one. *(**derived**; the `λ ∝ B/N` step assumes correction-per-draw is
roughly τ-independent, which is the part to check against the birth-cohort
measurement.)*

**The buffer servo folds in here.** *(user, `decisions.md` D22 — this is where
`E5` lands.)* Replay overfitting is bad on **any** route, so the answer is not to
mark the servo conditional-only but to make its enforcement automatic: the
precept below has a *derived* null, so it needs no hand-set `bar`/`release` pair
and no recalibration when the branch roles or the admission scheme change. That
is strictly better than the servo it replaces, and it is the reason `E5`'s stale
`freeze_policy 0` justification stops mattering rather than needing a fix.

**The memorisation sensor needs fixing to survive this.** `replay/scatter_err ÷
fwd/scatter_err` at "healthy ~2×" was calibrated under one particular admission
scheme; change the cap or T and the baseline moves with nobody recomputing it.
Under uniform admission an unweighted comparison is trivially 1. Two repairs,
either works:

- **Weighted:** compute the IS-weighted residual statistic on the replay batch.
  It estimates the same population quantity `fwd` does, so the null is exactly 1
  — a hypothesis test rather than a hand-calibrated ratio.
- **Predicted:** the unweighted null is `E[δ^{1+κ}]/(E[δ^κ]·E[δ])`, which is
  `1 + CV²` at κ=1, computable from `fwd`'s own level/spread decomposition.

**The precept, formalised:** `weighted_replay_err ≥ fwd_err`, equality as the
floor. The *gap* is `≈ (improvement rate) × τ`, so it reads three ways at once —
stable positive gap = healthy and measures progress per residence time; gap → 0 =
converged *or* memorising; gap < 0 = memorisation, unambiguous.

## B7b. Population management — the settled strategy

*(consolidated 2026-08-07. Supersedes the scattered statements in §B5/§B7; this
is the version to build from.)*

**The governing principle, and it decides four of the five questions below:**

> `μ_buf ∝ Q_admit · p_admit · p_survive`, and the estimator's weight divides by
> the draw only. So **any admission or purge rule that depends on `δ` re-enters
> the force spectrum uncorrected** — the residual gets counted twice, once
> stale. Rules that depend on anything *else* are support restrictions, and
> those divide out trivially.
>
> **Gate on energy, age, or randomness. Never on residual.**

### 1. Intake — uniform, and yes including negative `δ`

Admit uniformly from the sane pool, negative residuals included. Two reasons,
and the second is the one that makes it non-obvious:

- Filtering on `δ` is exactly the double-counting above.
- **A negative-`δ` row is not dead, it is dormant.** `δ` moves as the policy
  moves, so today's negative row is tomorrow's positive one. Meanwhile it draws
  with `p ∝ δ₊^κ = 0` and therefore costs **memory, not gradient budget**
  (§B7). Filtering it out also censors rather than reweights — a row admitted at
  `δ<0` that would have gone positive is unrecoverable, and no weight can undo
  that.

Under D29's invariant `E_Q[δ] ≈ 0`, so expect roughly half of fresh intake to be
dormant on arrival. That is the design working, not waste.

### 2. Purge — uniform-random hazard only

Keep the hazard exactly as it is: uniform-random, so `p_survive` is unbiased and
drops out of the weight. This is the piece that was already right.

**Drop the residual-based purge criteria** (`floor` / `stalled`). Two reasons:

- They bias `p_survive` by `δ`, which is the same defect as residual-based
  admission.
- A row driven to `δ ≈ 0` **by repeated replay** is `δ ≈ 0` *at that exact
  trajectory* — which is memorisation, and the low value is the **evidence**.
  Purging it destroys the signal the §B7 precept is trying to read.

Under the prioritised draw a `δ ≈ 0` row is drawn with `p ≈ 0` anyway, so the
floor was never buying gradient budget — only memory, which the hazard already
reclaims.

**Drift eviction (`Δ > Δ_max`) stays** and is benign: it removes rows whose
`Q_now ≈ 0`, i.e. whose IS weight would be ~0. Truncating a region of vanishing
measure is not a bias.

### 3. Gates — energy yes, residual no

`admit_reward_min` and friends are **support restrictions**: they define the
"sane pool" and restrict the estimator to it, which is the intent. Within the
retained region the correction is exact. This is the one place a hard filter is
correct, precisely because it does not depend on `δ`.

### 4. Buffer size — not a control knob, and *not* the overfitting lever

§B7a's algebra: under a binding cap with uniform draw, `N = rate × τ` and draw
frequency is `B/N`, so

```
λτ  ∝  (B/N) · (N/rate)  =  B / rate          N cancels
```

**Buffer size cannot move the memorisation product.** It buys diversity and
costs memory and refresh compute — so set it as large as the memory budget
allows and stop treating it as a dial.

**The overfitting servo actuates INTAKE RATE, not size:**

| | |
|---|---|
| **Sensor** | `weighted_replay_err ÷ fwd_err`, null **exactly 1** by construction under uniform admission (D22). Gap → 0 or negative = memorising |
| **Actuator** | raise intake rate (`churn_rate`), or lower batch size — the two terms in `B/rate` |
| **Not an actuator** | `max_size` |

*(Caveat carried from §B7a: the cancellation assumes the cap binds — it does,
occupancy sits at exactly `max_size` on both buffers — and uniform draw. Under
the prioritised draw `N → N_eff`, which makes prioritisation partly
self-limiting against memorisation.)*

### 5. Freshness — high intake, and `Δ` is the *measurement*, not the mechanism

"Any positive residual" is *nearly* the right instinct but conflates three
things. `δ = log Z − log R − log P_B + log P_F` rises with `log P_F`, so an
abandoned row does have low `δ` — but so does a **well-fit high-reward** row.
`δ` cannot tell "policy left it" from "we already learned it".

`Δ = log P_F_now − log P_F_admit` isolates the first. Its jobs, in order:

1. **The IS bridge** — this is the one `δ₊` cannot do. The draw needs
   `exp(Δ)` to convert `Q_admit → Q_now`; drop it and you are estimating the
   *old* policy's expectation.
2. **The eviction bar**, sharing the same constant.

**But there is a much simpler path, and today's results point at it.** If `τ` is
short, `Q_admit ≈ Q_now`, `exp(Δ) ≈ 1`, and **the entire `log_pf_admit`
machinery becomes unnecessary** — no per-row storage, no refresh, no drift
correction. Freshness is then guaranteed by intake rate alone.

That is the same lever as the overfitting servo in §4 above. Both pressures push
the same way:

> **High intake / short residence buys memorisation resistance AND removes the
> need for the drift correction.** Its cost is energy calls. `Δ` becomes a
> *diagnostic* — measure it to confirm it is negligible — rather than a
> mechanism to build.

**Recommended build order:** ship uniform intake + hazard purge + prioritised
draw with `exp(Δ)` omitted, log `Δ` as telemetry, and only build the drift
correction if the measured `Δ` distribution turns out to be wide. That is §C
step 1 demoted from machinery to instrumentation.

## B7c. Build status and what the build learned — 2026-08-07

**Shipped, off by default.** Absent a `buffers.replay_buffer.prioritise` block
nothing below runs, so every existing config is unaffected.

```yaml
buffers:
  replay_buffer:
    prioritise:
      enabled: true
      kappa: 1.0
```

Turning it on switches **all three** pieces together, because they are one
design: prioritised draw, uniform intake, hazard-only purge.

| Piece | Where |
|---|---|
| `p ∝ δ₊^κ`, row weights `w = (1/n_elig)/p` | `CrystalBuffer.prioritised_weights` |
| self-normalised weighting at the final reduction | `get_gfn_backward_loss(sample_weights=…)` |
| uniform intake (constant admission score) | `manage_replay_buffer` |
| hazard-only purge (`floor`/`stalled` disabled) | `manage_replay_buffer` |
| `ema_logw` refresh | `replay_train_step` |

**`δ` is reconstructed, not stored.** `δ = log Z − log w`, and `ema_logw` already
carried a per-row EMA of `log w` — checkpointed, resized on grow/purge, and
**never called by anything**. Wiring it up gives the signed residual with no new
field, which `ema_loss` (`|resid|`) cannot supply.

### What testing changed

**Zero-priority rows are excluded, not floored.** The first cut mixed a uniform
floor into `p` so every row stayed drawable. Measured `max(w) = 10⁴` — a row
drawn at probability ~0 carries weight ~∞ and single-handedly owns a
self-normalised batch. Now `δ₊ = 0` rows get `p = 0` outright, and the estimator
targets the uniform mean over the **positive half** — which is what the replay
branch is for (§B2). A relative floor (1% of the median `δ₊`) bounds the weight
range among the survivors.

**Unbiasedness verified, exactly.** Relative error `2e-16` at κ ∈ {0, 0.5, 1, 2}
on a synthetic population with `logw_std = 21`. Φ is invariant; only the draw
changes. That is the κ ladder's whole claim and it holds.

### ⚠ The variance claim did NOT reproduce, and the reason matters

§B5 says variance is *minimised* at κ=1 by Cauchy–Schwarz. Measured over 300
draws of 1000 rows:

| κ | ESS/n | max w | batch sd |
|---|---|---|---|
| 0 | 1.00 | 1.0 | **0.38** |
| 0.5 | 0.85 | 10 | 0.42 |
| 1.0 | 0.65 | 117 | 0.78 |
| 2.0 | 0.34 | 21026 | 2.23 |

Variance goes the **wrong way**. The reason is not a bug: **the optimal draw for
a self-normalised estimator is `p ∝ |f − μ|`, not `p ∝ |f|`.** For estimating a
*mean of δ*, δ is tightly clustered about its own mean, so prioritising by δ
over-samples where the integrand is *least* informative. Prioritisation only pays
when the integrand is concentrated where `p` is — a gradient dominated by tail
rows, which is the case §B9's `1 + CV²(|δ|·‖g‖)` bound describes and which this
synthetic test does **not** exercise.

So: **correctness is established, the payoff is not.** That is consistent with
§0a (replay `1 + CV² ≈ 1.7`, real but modest) and with §B9's warning that the
payoff here is more likely the LR ceiling than the step count. It also makes the
κ ladder's first job diagnostic rather than confirmatory.

**Three metrics ship for exactly this reason** — `replay/is_ess_frac`,
`replay/is_w_max_ratio`, `replay/is_elig_frac`. The estimator cannot go wrong in
the mean, so the only thing to watch is the weight tail. **If `is_ess_frac`
collapses, lower κ; the ladder is the instrument, not a foregone conclusion.**

## B7d. The memorisation sensor — built 2026-08-07

**D22 said fold the servo into the redesign and make enforcement automatic.
Done, and it needed no change to `protocol.py` at all** — only a sensor the
existing `_buffer_servo_tick` can read.

### Why not the IS statistic I proposed

`weighted_replay_err ÷ fwd_err` with a null of exactly 1 (D22) **stopped being
valid when the one-sided draw shipped.** `p ∝ δ₊^κ` draws only from the positive
half of the buffer, so the weighted statistic estimates the uniform mean over
*that half*, while `fwd_err` covers the whole on-policy distribution. The null is
no longer 1 and there is no clean way to recover it. Withdrawn.

### The sensor

Per-row, compare the residual a row carries **now** against the one it was
**admitted with** — both already stored, no new field:

```
ratio       = mean(ema_loss) / mean(birth_loss)      in (0, 1]
absorbed    = 1 - ratio
lambda_tau  = -ln(ratio)
```

`ratio = 1` is a pure delay line: composition equals intake, nothing fitted.
Falling toward 0 means residents have been corrected **at their own
trajectories** while the intake distribution has not moved — memorisation by
definition.

**The setpoint is derived, not calibrated.** Under exponential relaxation at rate
`λ` and exponential residence with mean `τ`, `ratio ≈ exp(−λτ)`, so §B7a's
boundary `λτ = 1` lands at **`ratio = 1/e = 0.368`**. It transfers across
problem, `T` and buffer size because nothing in it was measured.

**No survivorship bias — and that is a dividend of B7b.** `birth_loss` exists
only for resident rows, so it is the intake distribution *of survivors*. Under
the **uniform-random hazard** survivors are an unbiased sample of admits, so it
equals the intake distribution. This would **not** hold under the old
floor/stalled eviction, which conditioned survival on the residual.

Undrawn rows have `ema_loss == birth_loss` exactly and contribute `ratio` 1.
Correct: a row nothing trained on cannot have been memorised.

### The servo — config only

The existing servo forms `numerator/denominator` and tightens when the ratio is
**low**, which is already the right sign here. So:

```yaml
buffer_servo:
  numerator:   replay/ema_loss_mean
  denominator: replay/birth_loss_mean
  bar:     0.368     # lambda*tau = 1, derived
  release: 0.60      # lambda*tau ~ 0.5
  scale:   0.15
  gain:    0.05
  relax:   0.5
  max_step: 0.05
  max_boost: 8.0
```

Actuator unchanged: one boost `B` as `churn_rate × B` and
`mean_residence_steps / B`, which holds occupancy invariant and moves only reuse
and policy lag. That is **intake rate**, which §B7a proves is the only lever on
`λτ` — buffer size cancels out.

### Validation on 33 historical runs

The characteristic discriminates rather than pinning:

| `λτ` | arms | action |
|---|---|---|
| > 1.0 | BASE32K (1.54), local_aug02_abort_test / fix_validation (1.44), neat_dev (1.10) | **tighten** |
| 0.5 – 1.0 | ctrl_aug03_fx_static, local_aug02_ring_probe ×2, size_aug03_sz_lo_f20, subtb_l95 | hold |
| < 0.5 | the rest, incl. every `ctrl_aug03` servo arm (λτ ≈ 0.02) | release |

**Two negative results worth keeping.** A 1-D Wasserstein between the intake and
resident loss histograms matched the mean-shift statistic to three decimals on
every arm — the distributions differ by a translation, so the histogram
machinery buys nothing. And pre-schema buffers lacking `birth_loss` return `{}`,
so the servo holds at cold start rather than acting on garbage.

## B8. Z-currency is an INVARIANT, not a measurement

**Ruling, user 2026-08-06:** *"We should assume at all times here that
`log_Z_learned` is correct to current policies, i.e. `fwd/tb_resid_clipped <
0.5`, period, full stop."*

This is a premise, and it retires §0's Z row as an *experiment* — the bar is
already enforced by the `z_calibration` servo, whose `sensor: pooled` is
`|EMA fwd/tb_resid_clipped|` against `threshold: 0.5`
([mk_dev.yaml:56-58](../configs/mk_dev.yaml:56)). So the §0 row demotes from
"decide which of two explanations holds" to "**confirm the invariant held for
the whole run**", which is one trace, not an analysis.

**What the invariant actually says, stated precisely.** `tb_resid_clipped` is
`dL/dZ` up to `beta`, so pinning it to ~0 puts `log_Z_learned` at **TB's own
fixed point under the current policy**. With `unclipped: false` that is the
*winsorised* mean of `log w`; unwinsorised it is `z_jensen`. Either way it sits
**below `log Z_true` by the Jensen gap `KL(Q‖P)`** (§B1b) — and that is correct
behaviour, not drift. The gap *is* the objective, and it closes as training
succeeds. The old §B8 framing treated the `z_bias ≈ −4` reading as an error to
diagnose; under the invariant it is simply the objective being reported.

### Why this matters for the draw — it makes `δ₊` well-founded

`δ = log Z_learned − log w`, so pinning `log Z_learned` to TB's fixed point makes

```
E_Q[δ] ≈ 0
```

under the **current on-policy measure**. The one-sided priority `δ₊ = max(δ, 0)`
is therefore a split at the on-policy *mean*, not at an arbitrary origin. That is
exactly §B2's δ>0 / δ<0 partition, and it is what the earlier "one-sided means
not shift-invariant" worry was really asking for: the invariant supplies the
origin, so the worry is discharged by enforcement rather than by measurement.

**Two consequences worth keeping.**

- **`δ₊` already suppresses abandoned rows.** A row the policy has moved off has
  a fallen `log P_F` and therefore a *negative* `δ`, so `δ₊` gives it priority
  zero automatically. This overlaps with what the drift term `Δ` was introduced
  for — `Δ`'s remaining distinct jobs are the IS bridge for rows *still* in
  support, and the eviction bar. Worth knowing before building both.
- **The invariant is a `fwd`-branch statement.** `E[δ] ≈ 0` holds under
  `Q_now`; buffer rows were drawn from `Q_admit`, so over the *buffer* the mean
  is offset by exactly the accumulated drift. That offset is a staleness signal
  in its own right, and a cheaper one to read than per-row `Δ`.

<details><summary>Superseded framing — the two-explanation table</summary>

**Why this is in Part B at all** *(user 2026-08-06: "not sure what B8 is doing
here, seems unrelated" — fair, the stated reason was weak).* The original
justification was "every `beta` conclusion depends on it," which is true but
indirect. The direct reason is sharper:

> **The prioritised draw is one-sided, and one-sided means not shift-invariant.**
> `p_i ∝ δ₊^κ` with `δ₊ = max(δ, 0)`. If `log Z` is biased low by `z_gap`, every
> `δ` shifts uniformly upward — and a uniform shift through a `max(·, 0)`
> **changes which rows have nonzero priority at all.** Rows that should be the
> backward instrument's business get promoted into the replay draw, and the §B2
> split between the two halves of the residual field stops being where the design
> says it is.

So the Z level is not background context for Part B — it sets the **origin of the
coordinate the draw is defined on**. A quadratic loss on `|δ|` would not care; a
one-sided prioritised draw does. Read it before trusting any `δ₊` histogram.

*(Everything below is unchanged and still holds for the `beta` question too.)*

Three quantities already logged, no new code.

`z_level_loss` drives `E[logw − log_Z] → 0`, i.e. drives `log Z` to `z_jensen` —
which is **below the truth by exactly `z_gap = KL(Q‖P)`**. Its docstring records
the per-condition `z_bias` histogram sitting near −4 and treats that as the error
to be fixed, but `−KL(Q‖P)` is the *correct* reading when `log Z` is at the true
value. Two explanations produce identical readings:

| | `log_Z_learned` vs `z_emp` | `z_bias` |
|---|---|---|
| log Z correct, `z_level_loss` harmful | ≈ equal | ≈ `−z_gap` |
| log Z high by `z_gap`, `z_level_loss` working | high by `z_gap` | ≈ `−z_gap` |

Read all three on any existing post-fix run and they separate immediately.
Related: if the correct residual mean is `KL(Q‖P)` rather than 0, the `zerr`
convergence threshold is measuring sampler quality, not Z convergence.

</details>

## B9. Prior art and expected benefit

The technique is standard; the application to TB residuals is not.

- **RL:** Prioritised Experience Replay is the direct ancestor — priority ∝
  |TD error|^α with IS correction. ~2× median speedup on Atari.
- **SGD variance reduction:** [Katharopoulos & Fleuret, ICML 2018](https://arxiv.org/abs/1803.00942)
  derive the same optimal design and report up to an order of magnitude
  train-loss reduction at fixed wall-clock but only **5–17% relative test-error
  improvement** — and, instructively, they build an estimator of the achieved
  variance reduction **to decide when the method is worth switching on at all.**
  That gate is §0's `1 + CV²` row.
- **GFlowNets:** replay is well-studied but the field prioritises on **reward**,
  not residual ([Shen et al. ICML 2023](https://arxiv.org/pdf/2305.07170);
  [Vemgal et al.](https://arxiv.org/pdf/2307.07674)), with a known rich-get-richer
  coverage pathology. Residual-priority with IS correction is a different object —
  variance reduction on a fixed objective, unbiased by construction. No prior art
  found for it in GFlowNets specifically.

**Expected benefit, honestly.** The bound is `1 + CV²(|δ|·‖g‖)` and it is
measurable in advance; with `logw_std ≈ 21` it is plausibly ~2×. Translation into
steps-to-target is lossy — the literature's range is 1.2–2×, sometimes nothing.
**The payoff for this route is more likely the LR ceiling than the step count**
(§B10), and §B5a's ordering fix is not a variance effect at all.

## B10. Test plan

| Arm | Varies | Reads | Decides |
|---|---|---|---|
| **κ ladder** | κ ∈ {0, 0.5, 1} | `tb_err` vs step, ESS, grad_norm | Whether prioritisation buys anything **at provably fixed Φ**. The predicted profile is `S_κ·S_{2−κ} − S₁²`, computable before running — so this validates the framework rather than searching |
| **bwd priority** | `δ₋` draw on/off at fixed `beta` | `EffDim`, coverage, `bwd/logw_std_within` | Whether restoring within-tail ordering (§B5a) helps. **Not a variance test** — the only arm that changes an 80-nat and a 10-nat mode receiving identical drive |
| **LR ceiling** | κ ∈ {0,1} × LR across the aug02 cliff | where each arm dies | Whether bounded per-row influence moves the cliff. The 8e-4 arm died at pre-clip grad **587× the clip** — a gradient-outlier death, exactly what κ=1 prevents. LR is the largest measured lever, so this is where the payoff would be |
| **β × κ** | `beta` ∈ {10, ∞} × κ ∈ {0,1} | stability, `log_Z_learned` vs `z_emp` | Which of `beta`'s two jobs is load-bearing (§B6). Run per-branch — §B6 argues fwd and bwd should not land on the same answer |
| **P8 (i)** | replay off, bwd only | `over_coverage` | Whether a corrector is necessary at all — unchanged from `synthesis.md` §4a |
| **P8 (ii)** | in-batch reweighting, no buffer | `tb_err` vs step | Whether the *buffer* is necessary or only the prioritisation. §B3 predicts it loses by ~`1/q`, and §B5's "no pool to over-sample from" says it structurally cannot match |

## B11. What this reframes

| Item | Status |
|---|---|
| `synthesis.md` §4a | **sharpened, not overturned** — §B1–B3 derive it and give the rate |
| `decisions.md` `P8` "replay: structural or trick?" | reframed — both arms stand, arm (ii)'s prediction is now quantitative |
| `decisions.md` `P5` "`beta: 10.0`" | **answered in part** — the ladder is still wanted, but the *method* question resolves, and §B6 splits `beta`'s two jobs |
| `decisions.md` "right replay freshness?" (closed, D13) | **answered** — freshness is loss-of-support; measure `Δ`, don't proxy with age |
| `decisions.md` `R4` | replay's eviction philosophy restated: freshness → measured support |
| `synthesis.md` §3 "Freshness as a distinct actuator" | partially superseded — the Little's-law occupancy invariant holds, but the *reason* for churn changes from anti-memorisation to support maintenance |
| `synthesis.md` §1 (the thesis) | unchanged and strengthened |

**Not addressed:** intake. Prioritised-IS reweights what is in the buffer; it
cannot weight a region no row has occupied. Discovery remains the prior / anchor
/ search problem, and this makes that boundary sharper rather than moving it.

---

# C. Order of work

Phase-aligned. Steps within a phase are independent unless noted.

**Phase 1 — decide (now).** Run all of §0. It is read-only, it is all against
runs on disk, and five of the six rows gate a decision below. Then close out the
remaining design calls in A and B.

**Phase 2 — revert + baseline.** Baseline is **`nys7cfrt`** pending a config diff
against the working tree (`decisions.md` D2) — chosen over the historical-best
lr8x run (`ty4xdlzo`) because several recent runs land at nearly the same place
and the tiebreak is diff distance, not score. `ty4xdlzo` remains the trusted
*performance* reference.

Three things belong here rather than later, because all are the class of defect
that silently voids a battery:

- **§A5** — delete the LR middle layer. Independent of everything; both
  documented deadlock modes go with it.
- **`E4`** (cut-factor floor reads `lr_policy` in a fused stage) and **`S2`**
  (drive-liveness counters). An arm whose LR is silently pinned produces a
  confident wrong result, and the κ ladder is the cleanest experiment available —
  worth protecting.
- **Pre-fill the replay buffer at stage entry**, same pattern as the prior
  buffer's pre-fill. *(user 2026-08-06)* It fills fast in practice, so this is
  not about wait time — it removes `module_training_modes.md` M2 outright.
  Today, while the buffer is empty, `weights['bwd'] += replay_frac`
  ([train.py:2089](train.py:2089)) silently runs bwd at 0.8 instead of the
  configured 0.6, so **the controller's first balance ticks read metrics from a
  mixture the config does not describe.** That corrupts entry conditions for
  every arm including the baseline, which is why it belongs here rather than in
  phase 3. If pre-fill is not done, log the fold-in at minimum.

**Phase 3 — implement + test.**

1. **Store `log P_F` at admission.** One float per row, same pattern as
   `birth_loss` / `birth_step`. Ship as telemetry first and watch `Δ`
   distributions.
2. **Turn on a computed draw for replay** — `p_i ∝ δ₊^κ` with self-normalised
   `w_i ∝ 1/δ₊^κ` in the replay TB term. Note this is *adding* a weighted draw,
   not replacing one: `draw_replay_sample` currently passes `weighted=False`, so
   replay draws uniformly today. Start κ=1, drift out of the draw, quadratic on
   the replay branch, `freeze_z` on. Report ESS every step.
3. ✅ **DONE 2026-08-10. Uniform admission** (§B7), retiring `admit_cap_*` /
   `admit_temperature`. Paired with 2 (already shipped) as required — the
   draw was in place before admission stopped selecting. See `decisions.md`
   D5, `module_buffers.md` B0.
4. **The same draw on `bwd`** (§B5a), priority `δ₋`, on terminal-averaged
   `δ̄(x)`. Shares all of step 2's machinery. After replay only because replay is
   the simpler validation, not because it is the more valuable one.
5. **κ ladder** (§B10) — the first arm to run, and the only one with a provably
   clean null.
6. **Drift into the draw**, gated on ESS from step 2 behaving. Then **eviction on
   drift** sharing `Δ_max`, and **τ from measured λ**.
7. **`beta` relocation** per §B6, informed by §0's Z reading — per-branch.
8. **The peak servo** (§A4), containment untouched, envelope **without the decay
   leg** (§A4a). Committed by D3 regardless of §A6's outcome — what §A6 decides
   is `α_target`, the growth rate, and whether a separate hard ceiling is needed
   alongside α\* (if the probe shows no lead time on the cliff, §A8's hot-frozen
   quantile becomes that ceiling rather than a replacement for the servo).

---

*Warrant classes: **derived** (follows from the math) · **measured** (A/B'd, run
cited) · **inherited** (came from elsewhere, never re-examined here) ·
**arbitrary** (someone picked a number) · **contested** (conflicting evidence).*
