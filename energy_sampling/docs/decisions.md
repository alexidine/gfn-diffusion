# Decisions and open items

**Replaces `questions.md` and `register.md`** — consolidated 2026-08-06, and both
**deleted** the same day after a coverage check (all 30 questions and every
register ID map into this file). They had held one content set under two indexes
and drifted into contradiction. See the Appendix for every conflict found.

Three parts:

- **Part 1 — the docket.** Everything still needing *your* call. **3 open** (D2,
  D31, D25) after the 2026-08-06 decisions session, the controller session, and
  the 2026-08-08 v7 build (was 19). **D27 closed — stage 2 is built**; what
  survives it is the narrower **D31**, the servo's setpoint.
- **Part 2 — the register.** Work, measurements, and experiments, keyed by the
  IDs the module docs reference (`E4`, `P5`, `R4`, `S6`…).
- **Part 3 — closed**, including the full 2026-08-06 ruling set with
  consequences, and the **2026-08-08 documentation sync pass** (what in the module
  docs had gone stale against the 2026-08-07 build, and where it now lives).

Companion: [`to_do_rebuild.md`](to_do_rebuild.md) holds the design *argument*
(Parts A, B, and now C+) and the sequenced plan (§0, §C). This file is what is
open and who closes it; that one is why and in what order.

---

# Part 1 — The docket

## D33. Dead latent rows — flow the reduced object, do not pin ✅ DECIDED 2026-08-11

**Ruling: the SDE flows only the physically-real latent rows.** Rows that
`enforce_crystal_system` overwrites with a constant are held at their canonical value
and excluded from the diffusion. Evidence: `findings.md` F-009; argument:
[`design/fundamental_domain.md`](design/fundamental_domain.md) §3.

**Implementation: entirely inside `models/gfn.py`. BUILT 2026-08-11.** States stay full
width, so no consumer's index semantics move — `compute_zp_order_penalty` reads
`raw_latents[:, 6:6+3k]`, `compute_jacobian` slices the orientation block, and four
call sites pass `latent_params()` (already full width) rather than an SDE output. An
energy-boundary scatter/gather was considered and rejected for touching all of that.

`self.dim` is deliberately **unchanged**, so the `dim != 6+6·max_z_prime` guard
(`gfn.py:283`) stays meaningful and the drift head keeps its indexing. What changes:

- `get_periodic_dimensions` builds a **three-way partition** `ang_idx ⊎ lin_idx ⊎
  dead_idx = range(dim)`, asserted at construction. A dead dim leaves whichever block
  it was in — nothing may assume which, since 26 space groups have a centroid axis that
  is both free and `auv == 1` (F-008).
- `expanded_dim = lin_dim + 2·ang_dim` from the **final** sets: a dead *angular* dim
  removes 2 policy-input slots, a dead *linear* dim removes 1. Never `dim − n_dead`.
- `_pin_dead` writes the canonical value after each propagation and at both trajectory
  endpoints. Drift and noise are **not** masked — pinning discards them, so a misplaced
  mask cannot leave a dead dim moving. That was chosen over masking precisely because
  the failure mode is silent.
- `gauss_logprob` and the DPLR Woodbury path restrict to live dims via `_live_only`,
  **after** the angular wrap (`ang_idx` indexes full width). Leaving dead dims in the
  reductions would add a constant offset to the TB residual that `log Z` absorbs.
- `get_dplr_cov` zeroes ρ on `dplr_zero_mask = original_angular | dead`.
- Both helpers return the **same objects** when `dead_idx` is empty, which is what makes
  the no-dead-rows path bitwise identical rather than merely equal.

**One diagnostic was mis-scaled and is fixed:** `u_star = s_per_dim · data_ndim`
(`train.py`) is nats per *degree of freedom*, so it now scales by `live_dim` — was 17%
too lenient for monoclinic, 25% for orthorhombic. Identical for triclinic and toys.
`Effective Dimension` needs no change: it reads `sample_batch.latent_params()`, which
round-trips dead rows to 0.0 both before and after, so it is unaffected.

**Tests:** `test_dead_latent_rows.py`, 18 groups, all passing; the pre-existing
`test_periodic_scoring.py` passes unchanged. The load-bearing ones are the bitwise
pre-change comparison (recorded numbers inlined), measured trajectory constancy, the
density checked against an independent `MultivariateNormal`, and exact log-prob
invariance to the held constant.

**Found by adversarial review after the suite was green** — the tests missed all of
these, which is the argument for running the review at all:

1. **`TorsionGFN` construction broke outright.** It overrides
   `get_periodic_dimensions`, so changing that signature raised `TypeError` before step
   0 for *every* conformer run; a signature-only patch would then have `AttributeError`ed
   in `_pin_dead`, because the override hand-maintained six layout attributes and the base
   class had grown seven more. Fixed by extracting `_finalize_dim_partition` and having
   the override delegate, so the subclass now shares the partition assertions instead of
   a parallel copy. The suite had no subclass coverage at all.
2. **Stale checkpoints silently disabled the fix.** `_gfn_config_from` takes every
   architectural key from the file, so a pre-change monoclinic resume rebuilt with the
   rows LIVE while the startup probe printed reassurance. Now refused loudly
   (`_assert_dead_rows_match`); triclinic resumes still load, since they resolve to `()`.
3. **`dead_latent_values` paired against the *sorted* rows**, so `rows=(5,3),
   values=(a,b)` assigned them reversed. Now paired with the caller's ordering.
4. **`step_var` / `terminal_var` / the Gaussian diagnostics averaged over dead dims**,
   which would have made them drop by exactly `live_dim/dim` at this change and read as a
   coverage regression that never happened. All now average over live dims.
5. Minor: out-of-range space group raised `KeyError`; empty `space_groups` reached an
   `IndexError`.

Review coverage was **partial** — 20 of 42 verifier agents died on a spend limit, so
their findings were never adjudicated. Triaged by hand afterwards; the unresolved
remainder are diagnostic/cosmetic (per-dim figure labels hardcoding 12 latent names,
`Total Var` dividing by `data_ndim`, `get_traj_replay` not pinning — inert, since dead
values reach neither the policy input nor the log-probs).

**Deep pass, 2026-08-12** (`test_dead_latent_rows_deep.py`, evidence in F-010):
the reduction is **bitwise exact** against an independent live-only reimplementation;
`Var(log w)` falls 6.30 → 5.33 → 4.77 as `n_dead` goes 0 → 2 → 3, which is the
quantitative form of the argument that decided this entry against pinning; 65
config × dead-set models all hold the invariants; `live_dim == 0` gives log-probs of
exactly 0 rather than NaN; CPU/CUDA agree to ~1e-6 on an identical trajectory. That pass
also found **F-009b** — a pre-existing latent bug where the non-crystal angular mask was
hardcoded to width 12, which would have fed a `data_ndim`-18 toy only its first 12 dims.
Caught by the width assert added for this work, which had no other purpose.

**Free aunit axes now land too, at Z'=1** (`canonicalize_free_axes`, called from
`latent_params()`): pinned to the aunit box centre, which is latent 0 — the same constant
the angles take, so `_dead_values` stays all-zeros. Measured energy- and RDF-invariant on
physical structures (energy-invariant to <=1.2e-06 relative on 40 structures per SG) and
idempotent. The RDF is NOT a reliable witness to this -- see F-010b -- so assert gauge
invariance on the energy. **Z'>1 is
gated off**: the free translation there is one global shift, so fixing it needs a common
offset, and the leftover units then leave the box where re-wrapping a single unit by
`auv_d` is a symmetry only when `auv_d == 1`. The canonicaliser and the dead-row table
had to land TOGETHER — canonicalising without holding the rows would pin every buffer row
at the constant while the policy flowed the dim freely, i.e. this defect inverted.

**Certified against a closed form, 2026-08-12** (`test_latent_gaussian.py`, evidence in
F-023). Physical energies have never converged perfectly, so they cannot certify a
log Z; the new `latent_gaussian` energy can. It is the combination that did not
previously exist — `is_crystal` TRUE (crystal layout, dead rows, periodic angles) with
`latent_energy` TRUE (analytic reward on the latent) — so it is the first target that
exercises dead rows *and* has an exact answer. The old toys are `is_crystal` FALSE and
cannot reach the mechanism at all.

Two terms are **structurally zero** for any latent-scored problem, not config knobs:

- **`reduction_energy`.** On P-1 the reduced region is a thin set with no
  zero-reduction ball wide enough for a gaussian (best of 4000 draws: 0 at the centre,
  0.105 at the edge of a ±0.15 ball), so leaving it on would contaminate the target by
  ~1 nat at 1.5σ.
- **`jacobian_energy`.** The jacobian is a change of measure from box-latent to
  physical coordinates, so that a target defined by a *physical* energy is sampled
  correctly in physical space. A latent-space analytic target is defined *in* the latent
  space: there is no measure to correct, and applying one would make the target
  `gaussian × |J|`, which has no closed form in box coordinates. It would also break the
  dead-row test itself — rows 3/4/5 are cell angles, which enter `cell_volume`, so with
  the jacobian on those rows are not flat even when the energy ignores them and the
  rows-live arm's fictitious volume stops being analytic.

Being structural rather than a knob means no config can switch either back on by
accident. `reward_range` needs no such treatment: `set_reward_clip` has no callers, so
`energy_clip` is always `None` and `log_rescale_positive` is unreachable.

Battery: `configs/gauss_aug12/` — 10 arms, 5 space groups × rows held/live, each with a
predicted log Z. It also closes two gaps `deadrow_aug12` states it cannot: orthorhombic
(no physical prior on disk) and the free-axis path (a physical prior must be a real
crystal, but nothing builds a cell here, so any space group can be synthesised).
`periodic_centroids` is pinned FALSE on every arm — with centroid wrapping on, a free
axis has period 2 and its fictitious volume would be `log 2` rather than
`log(2 + √(π/k))`, invalidating the arms D/E predictions for a reason unrelated to D33.

**The defect.** `latent_params()` round-trips the clobbered angles to 0.0, so every
prior/buffer row carries `latent[3] = latent[5] = 0.0` exactly (std `0.0e+00`,
n=535 and n=3582) while the energy is flat across the whole box there. `bwd` trains
P_F toward a delta at 0; `fwd` gets no gradient. 2 of 12 dims trained against
inconsistent targets. Introduced by `8dea6b56`, which moved reduction from
Niggli *sampling* into an energy penalty that `enforce_crystal_system` preempts for
monoclinic and above.

**Why not pin the raw latent** (`E += k·z²`), which was the first proposal and will
look attractive again:

| | verdict |
|---|---|
| target correctness | **fine** — factorizes exactly, y-marginal is physical, `log Z_pin = log Z_phys + log(π/k)` analytic |
| prior consistency | **not fixable** — buffer rows are an exact delta, the pinned target has finite width. k→∞ is required for agreement and is capped by the terminal SDE resolution floor |
| free-energy cost | **additive** — target and policy both factorize, so `Var(log w) = Var_y + Var_z`. Pure ESS tax, zero information. Grows with k, since log-density scale is `−log σ` |

So the pin reduces the disagreement from *delta vs uniform-over-box* to *delta vs
narrow-Gaussian*. Mitigation, not repair. Flowing the reduced object removes it
outright: buffer rows and the target marginal become the same object.

**Scope and the cost we accepted.** `lin_dim` shrinks by `n_dead`, so
`expanded_dim = lin_dim + 2·ang_dim` shrinks and `StateEncoding`'s first layer changes
shape: **monoclinic+ checkpoints are orphaned.** That is the same class of break
`periodic_centroids` already causes (it moves dims between the two blocks, so
`expanded_dim` is already space-group dependent), which also means cross-SG warm
starting was never available for crystal runs and is not being given up here.
`n_dead = 0` for triclinic, so **sg 1 and sg 2 are an exact no-op** — `lin_dim`
unchanged, checkpoints still load, results bit-identical. No existing result is
invalidated, since every live battery is sg 2.

**Canonical dead value.** Measured 0.0 for every reachable case: π/2 maps to latent
0.0 exactly, verified on sg 4/14/15 (rows 3,5) and sg 19/61 (rows 3,4,5), min==max==0
across all rows. Hexagonal is the sole exception in principle (γ = 2π/3), so the
implementation asserts incoming terminals carry the pinned value rather than assuming
zero.

**Crystal problems only.** Resolution goes through `resolve_dead_rows(sg, is_crystal)`
(`models/dead_latent_rows.py`), which returns `()` for toy energies. Toys carry
`space_groups: [1]` as a placeholder; gating on the SG alone would pass today (P1 is
triclinic, no dead angles) and then break silently when free axes join the table — P1
has all three centroid axes free, so an ungated resolver would freeze 3 of a toy's 12
dims that its energy genuinely depends on. Same gate as `do_periodic_angles` and
`_resolve_periodic_centroid_axes`, for the same reason.

**Deliberately deferred.** (a) A genuinely multi-SG conditional model needs the max
width plus per-sample handling; there, masking the TB log-probs is the sound route
and pinning is the pragmatic one, with a tax that varies by `n_dead` and so biases
the mixture toward low-symmetry groups. (b) Rhombohedral, and the `a=b` averaging in
tetragonal/hexagonal/cubic, are **diagonal** degenerate directions — they need
reparameterisation, not row deletion, and are not covered. (c) Free aunit axes
(F-008) are the same class of defect but need a *new* projection first, since nothing
canonicalises them today.

---

## D30. `freeze_policy` on fwd — the thesis may be what costs convergence ⚠

**Measured 2026-08-07** (`to_do_rebuild.md` §0c). A three-way comparison at
matched steps, batch, `beta` and κ:

| fwd | replay | `fwd/tb_err` |
|---|---|---|
| policy grads | OFF | 21.54 → **20.97** ✅ |
| policy grads | ON | 21.63 → **21.14** ✅ |
| **FROZEN** | ON | 21.94 → **23.54** ❌ |

`freeze_policy: 1.0` is the single knob separating improving arms from degrading
ones. **Replay is neutral** — with fwd unfrozen, replay-on and replay-off are
tied — so `P8` arm (i) resolves as *"replay neither helps nor hurts here"*, not
*"delete replay"*.

This bears directly on **`synthesis.md` §1**, whose thesis is *"the policy is
trained entirely off-policy; the on-policy branch trains only Z."*
`freeze_policy` implements that thesis. Corroboration: **`nys7cfrt`**, the
strongest baseline candidate (D2), **has no `freeze_policy`**.

**Not yet a ruling — it needs your call and more evidence.** 800 steps, one seed,
one problem. And `tb_err` is a forward-branch metric while the intervention
trains the policy on that branch, so some gain could be the metric moving toward
its own trainer; `fwd/r2` and `over_coverage` moving the same way argues against
that, but the clean test is held-out coverage over a longer horizon.

**If it survives replication, the consequences are large:** §1's architecture
statement, `P8`'s framing, and the reason Part B exists at all are all downstream
of it.

### 🔴 rb0808 could not have answered this — found and fixed 2026-08-08

**The battery built to settle D30 never varied `freeze_policy`.** All 26 arms
carried `freeze_policy: 1.0`, because `mk_dev`'s naive stage ships it and the two
cells meant to be unfrozen were written by *omission* — leaving the key out,
on the assumption that absent meant off. Absent means **inherited**.

| arm as written | intended cell | what it actually was |
|---|---|---|
| `d30_unf_lr2` | unfrozen @ 2e-4 | `d30_frz_lr2` with a different `run_name`. **Nothing else differed.** |
| `d30_frz_lr4` | frozen @ 4e-4 | `base_T25` with a different `epochs` |
| `rep_ctrl_ratio` | "restore mk_dev balance" | **no delta at all** — nothing had removed it |

So the 2×2 was a 1×2 LR ladder, ~48 GPU-h went to duplicates, and the
discriminator every other area assumes an answer to would have come back silent.
The same root cause hit the controller arm, found by the guard written for the
first one.

**Fixed in [`configs/rb0808/make.py`](../configs/rb0808/make.py)** — an explicit
`unfreeze()`, arms repointed, and `assert_distinct()`, which hashes every emitted
config minus `run_name`/`epochs` and hard-fails on a collision. The battery stays
at 26 arms and 632 GPU-h:

| cell | arm | index |
|---|---|---|
| frz @ 4e-4 | `base_T25` | 1 |
| unf @ 4e-4 | `d30_unf_lr4` | 6 |
| frz @ 2e-4 | `d30_frz_lr2` | 4 |
| unf @ 2e-4 | `d30_unf_lr2` | 5 |

**`rep_ctrl_ratio` → `rep_fixed_fracs`** (index 23), balance controller *off*.
The intended contrast survives with the roles swapped: every other arm inherits
`kind: ratio`, so `base_T25` was always the controller arm and the missing cell
was the fixed-mix control.

**⚠ The contamination is wider than D30.** Every replay, `beta` and Z arm in the
battery ran frozen — which D30's own measurement calls the *degrading* regime,
and which is already the stated reason the aug07 `beta` ladder may be void. Any
rb0808 result read as "X helps / X hurts" was measured inside it. **Needs
resubmission**; index 1 (`base_T25`) was already running under the old config and
is unaffected, since frozen @ 4e-4 is a cell the corrected design still wants.

**Being answered locally instead of waiting.** `configs/local_aug08/` pair A runs
the frozen/unfrozen contrast at T=10 on the laptop — same question, one hour,
4.5× the length of the 800-step measurement above. Pair B is its seed replicate,
so the gap comes with a detection threshold rather than as an n=1 number.

### Pair A result, 2026-08-08 — **the degradation does not replicate**

3,600 steps at T=10 from a shared post-transient resume (step 2650), both arms
verified to start at exactly that step. `fwd/tb_err`, **median per 400-step
window** — this instrument matters, see the methodological note below:

| window | 2650 | 3050 | 3450 | 3850 | 4250 | 4650 | 5050 | 5450 | 5850 |
|---|---|---|---|---|---|---|---|---|---|
| `a_frz` | 21.16 | 21.79 | 22.07 | 21.27 | 19.44 | 19.44 | 19.87 | 19.21 | **18.76** |
| `a_unf` | 20.91 | 19.78 | 19.28 | 19.66 | 18.52 | 18.82 | 17.99 | 17.96 | **17.73** |

**Neither arm degrades**, and the 800-step headline above — frozen going
21.94 → 23.54 — does not reproduce at 4.5× the length. But **`a_unf` is below
`a_frz` in all nine windows**, by roughly 1–2.6 nats, and improves more overall
(−3.18 vs −2.40). Frozen also *rises* for its first three windows before turning
over; unfrozen falls immediately. `EffDim` is flat at ~5.80 in both, so this is
not bought with coverage.

So D30's **direction survives and its mechanism does not**: unfreezing helps,
consistently — but frozen is *slower*, not *degrading*, which is a materially
weaker claim than the one `synthesis.md` §1 is in tension with.

> **⚠ Methodological note — this entry was wrong once, and the error is
> instructive.** A first pass sampled `fwd/tb_err` at the 0/25/50/75/100%
> indices and reported a late "upturn" in `a_unf` (17.14 → 18.25) and a dead
> heat at the end. Both were artefacts: single samples, against a within-window
> scatter of ±1 nat. Binned medians show a monotone lead with no crossover.
> **Never read a trajectory off point samples on this metric** — the per-eval
> scatter is comparable to the effects being chased.

**One caution stands.** The arms differ by **0.25 nats in the first window**
despite identical initial conditions, and the sustained gap is ~1 nat, so the
effect is only ~4× the immediate divergence. Pair B (seed replicate) is what
turns that into a real threshold. And T=10 is not T=25 — *T dominates outcomes*
is well established here, so this constrains what the cluster should expect
rather than substituting for it.

### The 2×2 completes, 2026-08-08 — **unfreezing is NOT an LR effect**

Pair D added the LR leg at 1.72× base (2.15e-4, the probe's own estimate of
`a_frz`'s optimum). With pair A that is the freeze × LR 2×2 at T=10 — the cell
rb0808 could not run. Final-window (5850+) medians:

**`bwd/tb_err`** — the resolving metric, seed floor **0.04 (frz) / 0.10 (unf)**:

| | lr 1.25e-4 | lr 2.15e-4 | LR effect |
|---|---|---|---|
| **frozen** | 15.14 | 16.06 | **+0.92 (worse)** |
| **unfrozen** | 14.07 | 14.63 | **+0.56 (worse)** |
| freeze effect | **−1.07** | **−1.43** | |

Three readings, all against a noise floor 10–35× smaller than the effects:

1. **Raising LR hurts both rows.** 2.15e-4 is past the optimum, not short of it.
2. **The freeze benefit survives and grows** at the higher LR (−1.07 → −1.43).
3. **The substitution test fails outright.** If unfreezing were just more LR,
   `frz@2.15e-4` should land on `unf@1.25e-4`. It lands at 16.06 against 14.07 —
   a 1.99 nat gap, ~20–50× the seed floor, and in the *wrong direction*.

`fwd/tb_err` agrees but resolves poorly (frozen seed floor 0.52 swallows its
−0.17 LR effect); `EffDim` is flat at ~5.8 in all four cells, every gap inside
its own 0.08–0.10 floor. **So: unfreezing helps, raising LR hurts, and they are
not the same lever.** The gradient-weight / effective-step reading of D30 is
refuted on this route — as the `step_norm` evidence already suggested (0.06496
frozen vs 0.06360 unfrozen, a 2% difference).

### ⚠ And a warning for §A4's servo, which is the more valuable result

The **α\* ∝ 1/lr null PASSES** — raising LR 1.72× should divide `alpha_median` by
1.72, and it does: `a_unf` 2.65 → 1.53 is a ratio of **1.73** against a predicted
1.72. (The frozen leg gives 1.47, ~2× its own 0.16 seed floor — acceptable, and
far better than the 3.24-vs-4.0 the 800-step local reading produced, which was
transient-contaminated.) So the probe's scaling law is validated locally, and
this is the analogue of rb0808's `hold_T25_lr1` exact-answer cell.

**But following α\* made both arms worse.** The probe read ~1.7 at base LR,
meaning "your step was 1.7× too small" — and taking exactly that step degraded
`bwd/tb_err` in both rows. A servo targeting `α* = 1.0` on the median would have
driven LR up ~1.7× and lost ~0.6–0.9 nats.

The reason is not that the sensor is wrong; it is that **α\* is a local property
of one ray on one frozen batch, and raising `lr` changes the whole trajectory** —
Adam's second-moment state, the noise scale, and the batch-to-batch step
direction. The local optimum along the taken step is not the best global LR.
That is §A6's objection, now measured rather than argued.

**Consequence for D27 / stage 2.** `α_target = 1.0` is the wrong setpoint on this
route. Either the setpoint sits deliberately above 1 (accept a standing
undershoot), or α\* is used as a **ceiling / safety sensor** — "α\* falling below
1 means back off" — rather than as a growth signal. The second is what D3's
ruling already anticipated: *"the servo still gets built; what changes is that it
needs a separate hard ceiling."* This measurement says the growth half is the
part that does not work, and it should be built as a one-sided backtracker.

**Consequence for the local battery:** pair C was drafted unfrozen on the premise
that frozen is a degrading regime whose results may be void. That premise is
gone, so pair C now runs **frozen** — matching `mk_dev` as shipped and the
corrected rb0808 `z_track` / `zcal_off` arms, and reading against `a_frz`. Pair D
was a `beta` ladder resting on the same premise; it is repointed to **pair A at
2× length**, asking whether the ~1 nat lead widens, holds, or closes.

**And the D29 scare resolves.** `|fwd/tb_resid_clipped| < 0.5` breaches *only* in
the first ~500 steps after an unfreeze — `a_unf` 12.2% then **0.0%**, and the
same front-loaded profile in `r9` (20.4% → 0.0%) and `r5` (18.4% → 0.0%), against
0.0% throughout in `a_frz`. `z_calibration` displaces and recovers within one
window. **Start any δ₊ / `is_elig_frac` window ≥500 steps after an unfreeze** and
the readings are on-origin; there is no standing violation to design around.



## D2. Baseline — **diffed 2026-08-06; the answer split**

`nys7cfrt` = `aug02_a2_T60_lr4_tight`, 45,370 steps, `state=failed`.
`ty4xdlzo` = `postfix_july30_postfix_lr8x`, 46,240 steps, `state=crashed`.

**On code, `nys7cfrt` wins decisively.** It ran on commit **`9796fbb` — current
HEAD**. The only code delta is the uncommitted working tree: 5 files,
+559/−89, almost all `protocol.py`. `ty4xdlzo` ran on `433cbd5`, four commits
back, for a delta of 8 files, +1319/−217.

**On configuration it is a poor baseline for what Part B measures.** 29 keys
differ; four of them are not nuances:

| Key | `nys7cfrt` | `mk_dev` | Why it matters |
|---|---|---|---|
| `integrator_T` / `eval_T` | **60** | 10 | a different problem scale, and T dominates outcomes ([[stab-july21-eval-T-artifact]]) |
| `fwd.freeze_policy` | **absent** | `1.0` | nys7cfrt's fwd branch **trained the policy**. `synthesis.md` §1's thesis — *policy trained entirely off-policy, fwd trains only Z* — **does not describe this run** |
| `fracs.replay` / `max_fracs.replay` | **0.048 / 0.2** | 0.2 / 0.45 | replay is nearly *off*. That is close to `P8` arm (i), not a baseline against which a replay redesign can show a gain |
| `balance.metrics.bwd` | **`bwd/relative_under`** | `relative_under_wcen` | nys7cfrt used the metric target 3.0 was actually calibrated for. It is the run where the controller was *not* mis-targeted — which is `P1`'s whole problem |

Also absent from `nys7cfrt`: the entire **`z_calibration`** subsystem (14 keys),
`fwd_loss_coeffs.z_level`, `bwd_loss_coeffs.level_gap`,
`energy_config.log_temperature_range`, `anchor_buffer.noise_log_range`.

**Two things worth noticing.** `nys7cfrt` ran `adaptive_lr.cut_ratio: 1.0` — a
cut multiplies LR by 1, so **the middle layer was already inert as an actuator**
— and `decay_halflife_steps: 0`, **no decay**. That is essentially the post-D4 /
post-D7 LR configuration, already run, and it produced a top result. Independent
support for both rulings.

**Open — your call.** The "easier to diff" criterion picks `nys7cfrt` on code and
argues against it on config. Three ways to go:

- **(a)** Baseline = a **fresh run of the current `mk_dev`** at the intended `T`.
  Costs one run; gives a baseline that actually shares the architecture Part B
  modifies. `nys7cfrt` becomes the reference for "what HEAD produced at T=60 with
  replay near-off."
- **(b)** Baseline = `nys7cfrt` as-is, and accept that Part B's arms are measured
  against a near-`replay-off` configuration.
- **(c)** Re-run `nys7cfrt`'s config on the current tree at T=60, changing only
  `freeze_policy` and the replay fracs — an explicit bridge run.

**Recommendation: (a).** The replay redesign is the point, and (b) measures it
against a run where replay was 4.8% of the mix.

## ~~D8. Balance controller shape~~ — **CLOSED 2026-08-06, see Part 3**

Resolved in the deferred session: `kind: ratio`, implemented in
[`_ratio_tick`](../protocol.py:1467). The material below is kept because it is
the argument the answer came out of — the taxonomy in particular is still how to
read the other two kinds.

Under `drive: relative`, `s = max(v/target − 1, 0)`. Where a target sits relative
to its metric's operating range determines the controller's entire character:

| Target placement | Behaviour |
|---|---|
| Above the operating range | drive pinned at 0 — that side is **inert** |
| At the top of the range | drive 0 normally, arms on excursion → a **guard** |
| Inside the range | continuously positive → an **allocator** |

Same code, three controllers. Nothing in the config or the metrics distinguishes
them, and nothing reports which regime a side is in.

**The asymmetry argument.** The collapse hazard is fast and one-way — `bwd 0.001
→ EffDim 5.99 → 1.3 in one window` — and lost modes are not spontaneously
re-discovered. A one-way door is a floor that must hold, not a quantity to trade
against replay. That is `kind: constraint` (one side a bar, one best-effort),
implemented and never run on this route.

**The counter-evidence is weaker than it looks.** replay_july26's limit cycling
is real, but "fixed mix beats controlled" is **confounded** — every gentle arm
was LR-dead (`cut_factor` 0.02, pinned at `min_lr` from ~16k), so the damage
mechanism was transient → tripwire → latch, not mis-allocation. And sensitivity
≠ time-variation: that the mix strongly moves the landscape is established; that
the *optimal* mix moves over training is not.

> **No target values recorded here, deliberately.** Settling bands are intensely
> contextual (problem, `T`, `W`, stage) and do not transfer; treating one run's
> band as a general fact is the failure mode in [[findings-must-generalize]].

**Resolved:** `P1` is **dissolved, not answered** — `kind: ratio` has no target
to calibrate, so the band never needed reading. It stays open only for the two
older kinds.

## ~~D11. Should `naive` declare an `anneal` block?~~ ✅ CLOSED — **no**

Answered by D8. An anneal exists to walk two hand-set targets toward a fixed
point nobody could name a priori; `kind: ratio` has one setpoint that *is* the
judgement, so there is nothing to anneal toward. It would also reintroduce the
one-way ratchet whose terminal state is permanent marginal breach
([[controller-ratchet-marginal-breach]]). The setpoint moves when a battery says
it should, not on a clean-streak timer.

## ~~D26. LR probe scope~~ ✅ CLOSED — **(b)**, and stage 1 is built

`Δ` over policy parameters only; the flow head is held at its post-step value
for all three evaluations, so it contributes an identical constant to `L₀/L½/L₁`
and drops out of the fit. [`step_probe.py`](../step_probe.py), disabled unless a
`step_probe` config block is declared. See `to_do_rebuild.md` §A4b for the config
surface, what is logged, and the precision risk that turned up during the build.

<details><summary>original options</summary>

**Which loss does the probe evaluate, and which parameter group does the servo
drive?** `§A3a.4` flagged this and never resolved it: in a fused stage `α*` rates
a *composite* step including the flow head, which is LR-pinned separately and
which the servo would not control.

- **(a)** combined fused loss, servo drives `lr_fused` — simplest, `α*`
  contaminated by the flow head
- **(b)** combined loss, `Δ` over policy parameters only — clean attribution,
  one masked copy **(recommended)**
- **(c)** per-branch probes, servo on the weighted combination — most machinery,
  branches disagree by construction

</details>

## ~~D28. IS correction on `bwd` — `λ = κ` or `λ = 0`?~~ ✅ CLOSED — not a decision

**Withdrawn 2026-08-06, same day it was raised.** User: *"I had also thought
that beta would be much larger (or just quadratic) on the new weighting
scheme."* Correct, and it dissolves the question.

I mis-tabulated the per-row push for the IS-corrected draw as unbounded. It is
`w·|ℓ'| ∝ δ^(1−κ)`, which is **constant at κ=1** — so corrected draw + quadratic
gives linear Φ *and* bounded per-row push, unbiased, with the κ ladder's clean
null intact. No λ split, no branch asymmetry in the draw.

**What survives is a sharper rule** (`to_do_rebuild.md` §B5b): IS correction and
Huber actively conflict. At κ=1 the combination makes per-row push `∝ β/δ`, so a
mode 80 nats out pushes *eight times less* than one 10 nats out — both
mechanisms shrink the tail and they stack. **Any branch running a prioritised
draw must run quadratic, or `beta` large enough to be inactive** (well above ~60
at `logw_std ≈ 21`, not 10).

<details><summary>original framing</summary>

*(raised 2026-08-06 while scoping the replay build; see `to_do_rebuild.md`
§B5b, which is new and which corrects §B5/§B5a.)*

§B5 and §B5a read as one estimator with a sign flip. They are not. The κ ladder
sets draw `∝ δ^κ` and weight `∝ δ^−κ`, so their product is constant and **Φ is
invariant for any loss shape** — which is the ladder's whole virtue, but which
also means **the corrected draw cannot fix the "80 nats and 10 nats get
identical drive" problem** that §B5a exists to fix. That problem is a statement
about Φ.

Splitting the exponents (draw `δ^κ`, weight `δ^−λ`) makes it a decision:

- **`λ = κ`** — unbiased, pure variance reduction, clean null. Fits **replay**.
- **`λ = 0`** — uncorrected; Φ becomes linear in δ *while per-row push stays
  bounded at `beta`*. Fits **bwd**, and it is the only setting that does what
  §B5a claims.

**Recommendation: `λ = κ` on replay, `λ = 0` on bwd** — the same split as D6's
branch-asymmetric `beta`, for the same stated reason.

**Knock-on:** §B10's κ-ladder null ("any difference is estimator variance
alone") holds only at `λ = κ`. The bwd arm at `λ = 0` changes Φ and must not be
read as a variance result.

</details>

**Live knock-on for §B10:** the β×κ arm's `beta: 10` cell measures the
decreasing-push pathology, not a `beta` effect. Design that arm as
`beta ∈ {60, ∞} × κ ∈ {0,1}`, or read the `beta: 10` cell as a negative control.

## D29. Z-currency is an assumed invariant ✅ CLOSED — premise, not a measurement

**User 2026-08-06:** *"We should assume at all times here that `log_Z_learned`
is correct to current policies, i.e. `fwd/tb_resid_clipped < 0.5`, period, full
stop."*

`z_calibration` already enforces that exact bar (`sensor: pooled` =
`|EMA fwd/tb_resid_clipped|`, `threshold: 0.5`). So §0's Z row demotes from an
experiment to a one-trace **invariant check**, and §B8 is rewritten around the
premise.

**What it buys the replay design.** `δ = log Z_learned − log w`, so pinning
`log Z_learned` to TB's fixed point gives `E_Q[δ] ≈ 0` under the current
on-policy measure — which means `δ₊ = max(δ, 0)` splits at the on-policy *mean*
rather than an arbitrary origin. The "one-sided priority is not shift-invariant"
concern is discharged **by enforcement rather than by measurement**.

Two things fall out, both recorded in §B8: `δ₊` already gives priority zero to
rows the policy has abandoned (their `log P_F` fell, so their `δ` is negative),
which **overlaps with what the drift term `Δ` was introduced for**; and since the
invariant is a `fwd`-branch statement, the buffer's mean `δ` offset from zero is
itself a staleness signal, cheaper to read than per-row `Δ`.

## D31. The servo's setpoint ⚠ OPEN

**What the loop is.** `step_probe.py` moves the policy along the step the
optimizer just took and evaluates the loss at 3 points on that ray. `α*` = the
multiple of that step which would have minimised the loss. `α* = 1` means "the
step landed exactly at the minimum along its own direction". Every 200 steps:

    peak_scale  *=  clip( median(α*) / target , clip_lo , clip_hi )

So `target` **is** the setpoint — the value α\* is driven to. That part of your
mental model is right. `α* = 1` was assumed universal because it has a physical
meaning, not because anything measured it.

**What was measured** (lr_aug08, 7 arms, T=10, 5400 steps, shared resume):

| # | Finding | Confidence |
|---|---|---|
| 1 | The loop tracks. Seeded 12.5× low it climbed 26× and held α\* at 1.006 vs target 1.0; `peak_scale` 32 against a bound of 200, so a fixed point not a bound | **high** — 2 arms, both directions |
| 2 | `α* ∝ 1/lr`. `lr × α*` = 3.07e-4 ± 10% over 8 uncensored rungs | **high** — holds within one run and across two |
| 3 | `target: 1.0` lands at 3.2e-4 ≈ 2.6× too hot; costs 2.2 nats bwd, 2.8 fwd vs hand-set | **high** — direct A/B, everything else held |
| 4 | 🔴 `target: 1.0` **runs away** (see the mechanism below) | **medium** — n=1, but the mechanism is understood and predicts the trace |
| 5 | `target` calibrated (1.87 = α\* observed at a good LR) lands 1.14e-4 / 1.41e-4 from 11× below and 3.2× above; the from-above arm **matches hand-tuned** | **high** — two-sided convergence, and the hand-tuned value sits between them |
| 6 | Approaching from **below** costs 2.2 nats fwd at the *same* final LR | **medium** — n=1 each side |

**The runaway mechanism (#4), stated as a loop:**

1. policy degrades → loss surface along the step flattens
2. flatter surface → the parabola's minimum moves further out → α\* **rises**
3. α\* > target → servo **raises** `lr`
4. higher `lr` → more degradation → back to 1

Measured: `fwd/tb_err` 21 → 35 while `lr` crept 3.1e-4 → 4.5e-4 and α\* sat at
0.92–1.14 throughout.

**Why α\* did not report it — and this is the part worth keeping.** A closed loop
drives its own error to zero. **α\* ≈ target does not mean "the LR is right"; it
means "the servo is working".** The sensor is uninformative *precisely because*
it is the controlled variable. Any diagnostic read off a servo's own error signal
has this problem.

**`clip_hi: 1.0` kills step 3 structurally** — the multiplier can never exceed 1,
so `peak_scale` only falls and the loop cannot close.

### The decision

| posture | config | finds the LR? | can run away? |
|---|---|---|---|
| **(a) guard** *(shipped)* | seed = your LR, target 1.0, clip [0.8, **1.0**] | no | no |
| **(b) descending brake** | seed **above**, target **calibrated**, clip [0.8, **1.0**] | **yes**, from above | **no** |
| **(c) two-sided** | seed = your LR, target calibrated, clip [0.8, 1.25] | holds it | **yes** |

(b) is the only one that both finds the LR and forecloses the runaway. Its cost
is a hot transient descending (`fwd/tb_err` 22.5 → 25.9 → 18.0). I shipped (a)
because it is a no-op guard and the (b)/(c) choice is a risk call, not a
technical one.

**`target` is route-specific** — α\* transfers as a shape, not a value. Procedure:
run once at a hand-set LR, take `median(alpha_median)` over the second half.
`configs/lr_aug08/paird.py::measure_target` implements it.

⚠ `span` and `target` are coupled: a censored window reports `span`, so growth
there runs at `span/target`, not the clip. Keep `span` ≈ 2× target.


## D32. Detecting a slow collapse ⚠ OPEN — mostly ANSWERED by the balance controller

**Original claim (mine):** nothing detects a slow forward collapse, so a new
sensor on `fwd/logw_std` is needed. **User's challenge:** the balance controller
should already see it. **Checked against b_descend — the challenge is right, the
proposed mechanism is not, and the difference is the finding.**

| bin | `fwd/over_coverage` | `bwd/relative_under_wcen` | `rt_rho` | `rt_err` | `rt_theta` |
|---|---|---|---|---|---|
| 4 | 20.3 | 3.391 | 6.01 | 0.184 | **0.2513** ← at bound |
| 7 | 20.6 | 3.320 | 6.22 | 0.218 | **0.2513** |
| 9 | 29.0 | 3.241 | 9.15 | 0.604 | **0.2513** |
| 11 | **34.5** | **3.180** | **11.3** | **0.815** | **0.2513** |

1. **`bwd/relative_under_wcen` does NOT deteriorate.** 3.391 → 3.180 — it
   *improves* slightly. The mechanism you guessed is not the one that fires.
2. **The controller sees it anyway, through the numerator.**
   `rho = over_coverage / relative_under_wcen` went 6.0 → 11.3 and `rt_err`
   0.18 → 0.82. Its error signal more than quadrupled.
3. **But `rt_theta` was PINNED at its bound (0.2513) from bin 4** — before the
   collapse began. The controller watched its error quadruple with **zero
   remaining authority**.

**So no new sensor is needed. The detector already exists and is already logged:**

    rt_err rising  WHILE  rt_at_bound != 0     ->  the mix controller has lost
                                                   authority and the error is
                                                   still growing

`a_fixed` is the negative control: `rt_theta` also pins at 0.2513 (from bin 7),
but `rt_err` *falls* 0.17 → 0.02. **Pinning is not the pathology — pinning with a
rising error is.** Both terms are needed; either alone false-positives.

**What this does NOT cover, and why the LR sensor could not.** `fwd/tb_err` and
`fwd/over_coverage` are the same quantity to within 0.6 nats here, so the mix
controller's rho is really "how bad is fwd, scaled by how bad is bwd". It detects
the collapse; it cannot attribute it to the LR. And α\* — the thing that *should*
have flagged a hot LR — was pinned at target **by the servo holding it there**
(D31). A closed loop drives its own error to zero, so the controlled variable is
the one signal guaranteed not to report the fault.

**Revised item, much smaller:** emit/alarm on `rt_err` rising with
`rt_at_bound != 0`. No new metric, no calibration — both terms are already in
`protocol/rt_*`. The `fwd/logw_std` bar I proposed is **withdrawn**: it fires at
the same eval as everything else (nine forward metrics fire simultaneously —
they are all functions of the same batch's residuals), so it buys nothing the
above does not.

⚠ n=1 collapsed arm.


## ~~D27. Stage 2 — rebuild the servo **after** empirical tests~~ ✅ CLOSED — built 2026-08-08

All three blocking inputs were measured on 2026-08-08 (the table below), the
kill-gate cleared, and the servo shipped the same day. What remains open is
**D31** — narrower than D27, and about a number rather than about whether to
build. Original entry:

*(user 2026-08-06: "flag for us to rebuild the servo after empirical tests.")*
Stage 1 is instrumentation and ships alone. Stage 2 is deliberately blocked on a
run's worth of `α*` data, because three of its design inputs are measurements
rather than choices:

| Input | Settled by |
|---|---|
| growth rate | observed time for `α*` to traverse its range |
| ceiling forgetting half-life | `α*` autocorrelation |
| **servo vs. line search vs. one-sided backtracking** (§A4c) | measured per-probe dispersion — tight favours the line search, wide forces the servo-on-median or the backtracking variant |

Also pending from stage 1: whether `probe/second_diff_rel` sits clear of the
float32 floor. If it does not, the sensor is precision-limited and stage 2 does
not happen in this form at all.

### Measured 2026-08-08 — the kill-gate clears and the §A4c fork resolves

All three inputs were already on disk. Read across the 16 `batt0807` runs
(1,855 `α*` readings, 2,113 `second_diff_rel` readings), no new compute:

| Input | Measured | Consequence |
|---|---|---|
| **precision** | `second_diff_rel` median **3.6e-2** against the 1e-6 floor; **0.28%** of probes below it | ✅ **not precision-limited.** Stage 2 survives in this form — four and a half orders of margin |
| **per-probe dispersion** | within-run relative IQR **0.5–1.0** (pooled 1.45) | **wide, not tight** → §A4c resolves *against* the line search, toward **servo-on-median** or one-sided backtracking |
| **autocorrelation** | lag-1 **≈ 0.5** at cadence 20 | correlation time ~20–30 steps; sets the ceiling forgetting half-life |

**A consequence worth its own line: §A4's `clip(median, 0.9, 1.1)` is sized at
about one standard error of the quantity it clips.** With a within-window IQR
near 0.6 over ~25 probes, the standard error of the windowed median is ~9% of the
median. A ±10% clip therefore spends most of the servo's per-tick authority
chasing sampling noise. Either widen the clip, widen the window, or accept that
the loop moves on noise.

> **Caveat, and it is not small.** Every `batt0807` run is 3.5k–4.6k steps, and
> §0c measured `alpha_median` *still moving at 75% of phase 2* — so these
> readings are substantially inside the log-Z transient. The kill-gate verdict is
> robust to that (nothing about float32 margin depends on convergence); the
> dispersion number is a likely **over**-estimate. Re-read on `local_aug08`'s
> longer arms and on rb0808 before treating the ±10% conclusion as settled.

## D25. Experiment ladders — *"I will come back with firm decisions later on"*

The §B10 arms (κ ladder, bwd priority, LR ceiling, β×κ, `P8` i/ii) are specified
but not scheduled. Held pending your call.

*(D15, D22 and D24 closed 2026-08-06 — see Part 3. D3's "transfer not extraction"
justification was **retracted** 2026-08-06; the ruling stands on stronger
grounds — see Part 3.)*

---

# Part 2 — Register

Everything that does *not* need your call.

## 2a. Just needs doing

| # | Where | What | Fix |
|---|---|---|---|
| **N1** | [`to_do_rebuild.md`](to_do_rebuild.md) | **Extend the plan to Parts C+** per D1(b) — ~~balance controller (C)~~ ✅ **done: settled and shipped, written up as `module_protocol.md` P7 rather than as a plan section, since it is built rather than proposed**; ~~`naive` anneal~~ ✅ **closed (D11)**; still open: anchor health gate, smoke config, memory retirement. | the plan covered 2 of 9 open areas and read finished |
| ~~**N2**~~ | LR envelope | ✅ **DONE 2026-08-08.** `decay_halflife_steps` and `decay_floor_scale` are deleted keys; the envelope is ramp → hold, forever. | *derived:* α\* rates the **product** `peak × envelope(t)`, so a deterministic multiplier on it is absorbed — the servo just raises `peak` to compensate, leaving `peak` inflated against the units its own ceiling is expressed in |
| **N3** | anchor health gate | ⚠ **HALF DONE 2026-08-08, and the other half is a question.** The bar keys are renamed to `health_gate_floor` / `health_gate_ceiling` — named after their *role*, so swapping the ruler can no longer leave a bar called `health_gate_r2` holding a number that has nothing to do with r2. **The metric swap was NOT taken.** | D9 rules `tb_err_worst`, but its own rationale is why that stalls: no bar transfers. The incumbent `tb_resid_clipped @ 0.5` is **derived** — it is the D29 Z-currency invariant, the same bar `z_calibration` actively holds — while `tb_err_worst` is an unbounded RMS reading 18–21 when perfectly healthy, and nobody can yet state its bar. **Your call:** swap and pick a bar off a battery, or leave the derived one |
| **N4** | reporting | **Invariant: eval `beta` == traj `beta`, per trajectory type** (D16). Branches need not agree with each other; a within-type mismatch is the bug. | assert it rather than asserting uniformity. Retires the "emit beta / divide by beta" fork — neither is needed once the invariant holds and reporting is per-branch |
| **N5** | smoke config | ELJ / real crystals, unconditional, **fixed step counts** (D10). Plus: **shrink every phase-boundary buffer fill** — tens of thousands of energy calls for prior-buffer filling is not acceptable in a test config. | end-to-end functionality check only. `S3`'s preflight is separate and ships regardless |
| **E2** | [gflownet_losses.py:283](../gflownet_losses.py:283), [:525](../gflownet_losses.py:525) | `emp_z > 0` with both VarGrad coefficients at 0 leaves `log_Z` (fwd) / `log_Z_emp` (bwd) unbound → `NameError` mid-run. | config-load guard; **fold into `S6`** rather than patching twice |
| ~~**E4**~~ | `controller.py` | ✅ **DISSOLVED 2026-08-08.** The cut factor and its floor are deleted with the middle layer, and the floor arithmetic did **not** survive into the servo: `servo.bounds` is an explicit `[lo, hi]` on `peak_scale`, group-independent, so no LR appears in it at all. | the defect was that a floor derived from one group's base LR governed a different group's |
| ~~**S3**~~ | config load | ✅ **DONE 2026-08-08.** `utils.preflight_config` walks a `_RETIRED_KEYS` table at load and hard-fails with the replacement named, plus asserts `eval_T == integrator.T`. Wired into both `get_train_args` and `train_conformer`. | aug02 lost all 16 arms' entire phase 1 to a retired-key guard that lived inside `manage_replay_buffer`, which first runs at the phase-1 → 2 transition (`module_buffers.md` B4) |
| **T1** | `sample_metrics.py` | **Delete the MMD family** + `compute_distances` + `compute_distribution_distances`. | ~200 lines, zero external callers, only `wasserstein` is used |
| ~~**E6**~~ | `configs/mk_dev.yaml` | ✅ **closed — not a defect** (D15). The flag stays: `scramble_conditions` is intent, `scramble_applicable()` is the structural guard, and [train.py:2892](../train.py:2892) already ANDs them. Only the config comment's claim was false, and it is fixed | — |
| **M2** | [train.py:2089](../train.py:2089) | **Pre-fill the replay buffer at stage entry**, same pattern as the prior buffer's. While empty, `weights['bwd'] += replay_frac` runs bwd at 0.8 instead of the configured 0.6, so the controller's first balance ticks read a mixture the config does not describe. | corrupts entry conditions for every arm including the baseline → **§C Phase 2**. Supersedes `R7` |
| **S6** | `gflownet_losses.py` | **Functionalise** the two loss assemblers (your preference over collapsing). | extract shared sub-methods, keep both entry points; gives `E2` a home |
| **S2** | controller + protocol | Report drive liveness: ticks-since-last-nonzero-drive, per side. | ⏸ **descoped 2026-08-06** — the live balance loop no longer has a clampable drive (D8), so this is now maintenance on `proportional`/`constraint` plus the LR loop, not a route-critical fix |
| **S4** | `MetricTracker` | Staleness counter. | approved in principle, but nothing would read it today; its natural consumer is a drive that *abstains* on stale data — the same change as `S2`. **Do both or neither** |
| **S7** | `protocol.py` | Derive `_rb_base` from config each tick; drop the cache. | servo state split across instance cache / `stage_ctrl` / live `args`, correct today only for a non-local reason |
| **S8** | `buffer.py` | Rename/split `min_size`. | a per-cycle chunk bound in one place, a sampling count in another. Not a minimum size anywhere |
| **S9** | `quick_tb_stats` | Do **not** suppress degenerate aliases — **annotate**. | emit a `metrics/degenerate` flag so a reader knows the aliasing is expected |
| **S10/S12** | batch sizing | Log the batch-knee decision as a metric, not a print. | comparing arms currently means reading stdout |
| **S11** | balance controller | Gate the first balance ticks on metric freshness. | after a transition the controller acts on the previous stage's EMAs for ~100 steps. **Worth more under `kind: ratio`**: an integrator writes that error into its own state rather than washing it out |
| **N6** | LR instrumentation | **Wire the orphaned `uw_global` / `uw_max`.** | a few lines. It and `α*` are the two candidate *ceiling-free* LR parameterisations; `α*` is stronger only because it has a setpoint. Worth logging alongside the §A6 probe so both are on the same runs — see `to_do_rebuild.md` §A8 |
| **R2** | fused mode | Rename **declined**; state the fact at the definition site instead: every active branch runs every step, step cost is independent of the fracs. | a comment, not a rename |

**Rationale to write down** (no code change):

| # | Item | The correction |
|---|---|---|
| **R1** | Combined loss is `mean` over active terms, not sum. | Effective weight is `coeff_i / n_active_terms` — **turning on a term dilutes every other one**. Live in phase 1. Coefficients are *relative weights*; comparing one across configs with different term counts is invalid |
| **R3** | Three frac bounds live in three files. | `min_fracs` (lexicographic nudge), `balance.floor` (proportional only), `deactivate_threshold` (branch skip). **Not a defect** — the threshold is a binary declaration of deactivatability, set intentionally — but state the relationship in one place. **Partly closed 2026-08-06**: under `kind: ratio` the live bound is `bounds`, and the parser now *enforces* `bounds.lo ≥ deactivate_threshold` instead of leaving the relationship implicit |
| **R4** | Three eviction philosophies. | prior evicts on relative rank (**diversity**); replay on age (**freshness**); anchor never (**memory**). §B7 revises replay's: freshness → *measured support* (`Δ` vs `Δ_max`), not age |
| **R6** | Replay intake continues when `fwd` is deactivated as a trainer. | `manage_replay_buffer` fires on `fwd_ran`, which includes force-refresh-only runs. Almost certainly desirable; undocumented |
| ~~**R7**~~ | | **Promoted to a defect — see `M2`** |

## 2b. One measurement away — read-only, existing runs

| # | Claim | Why it matters |
|---|---|---|
| ~~**P1**~~ | ✅ **Dissolved 2026-08-06 (D8).** `kind: ratio` has no per-side target, so the band was never the thing that needed reading. | Replaced by a different measurement: the **plant gain** `d log ρ/dθ`, which the controller's own `(rt_theta, rt_rho)` trace yields for free. Measured provisionally at ≈0.15 (ctrl_aug03, confounded, 2000 steps) — if that holds, setpoints well inside the band are unreachable and the loop parks |
| **P3** | Replay memorisation may or may not be occurring on this route. | Sensor already computed: `replay/scatter_err ÷ fwd/scatter_err`. **= §0 row 4**, and the input to D22 |
| **P6** | Single-cut arms in postfix_july30 / tw_july31 / press_july29 had **stage-permanent** LR cuts. | **Re-read them by live LR** (D21) — one more scalar per run. Those ladders are otherwise confounded by an uncontrolled factor |
| **P9** | Buffer absorption stalls by **force balance against the fwd branch**, not insufficient drive. | Saturated bwd TB runs at `beta × bwd_frac` ≈ 3–5 vs the 0.5 at which phase-1 `mle` converges trivially — gain is not the deficit. Discriminating signature: fast-then-parked with fwd calibration degraded by the parked amount, vs a uniform slow crawl. If it parks, asymmetric `beta_bwd` is the lever — now available per D6 |
| — | `over_coverage`'s sensitivity to replay's share | aug02 and ctrl_aug03 varied it. Corroborates or weakens §4a with no new compute |

~~`P2`~~ — **retired by D7.** There is no step budget to set `hold_steps` and the
half-life from; see `N2`.

The remaining five rows of the read-only gate battery live in
[`to_do_rebuild.md`](to_do_rebuild.md) §0.

## 2c. Needs an experiment

| # | Question | Design |
|---|---|---|
| **P8** | Does the replay buffer earn a structural place, or is it a fine-tuning trick? | **Committed**, and sharpened by your 2026-08-06 framing: replay must **re-earn its place** against a plain fwd/bwd balance, with the optimistic expectation that if the redesign works it is *strictly* better — in which case `fwd` reduces to log Z calibration only. Arm (i): delete replay, keep bwd only. Arm (ii): in-batch `\|resid\|` reweighting, no buffer — if it matches, the whole replay subsystem collapses to a sampling weight |
| **P5** | `beta: 10.0` was never laddered. Shape settled; **value and branch-symmetry** open. | The knee sits at ~½ the residual SD (`logw_std ~21` vs `beta 10`). One ladder, **bwd-only**, sized to the tail (~60, not 30). Plus one free move: **average rollouts before the clip** (L8c) |
| **P7** | Is `fwd` pinned at 0.2 right? | ✅ **Answered analytically 2026-08-06 — under `freeze_policy: 1` any value above `deactivate_threshold` is equivalent.** Z is Adam-invariant to a uniform scale from its sole source, *and* the config's candidate-pool justification is void: fracs are loss weights, every active branch runs the full batch, so `fwd` yields ~2831 candidates against churn 80 regardless of `fwd_frac`. The frac is a **binary gate**, not a dial. Becomes a live question again only if `freeze_policy: 0` returns, and then `fwd` joins the δ>0 group rather than staying pinned |
| — | Does *learning* log Z buy anything over *tracking* it? | Biggest available swing at faster joint convergence, and **costs one config key** (`tb_z_source: persistent`). If tracking wins, much of the Z machinery becomes dead weight |
| — | Does `repeats: 2` beat 1 and 4, and does `dreg` do anything? | Neither tested recently; same arm answers both |
| — | Is `batch_growth_min_gain: 0.15` meaningful? | Possibly inside the noise — 14% reverted vs 16% kept, off a median of 20 step times |

The prioritised-estimator arms (κ ladder, bwd priority, LR ceiling, β×κ) are in
[`to_do_rebuild.md`](to_do_rebuild.md) §B10.

**Ranking of levers by measured effect:** **LR** (2.4× at matched wall clock,
with a cliff) → **sampling distributions** (untested architecturally) → **the
objective** (TB variants, Huber) → **the mix** (most machinery, weakest evidence).

## 2d. Ratified order of work (D20)

Derived from the 2026-08-06 answers, superseding the old top-three.

1. **§0 read-only battery.** Still first, still free — but its role changed: with
   D3 committing to the servo, §A6 is **instrumentation** (set `α_target`,
   validate the fit diagnostics), not a build/don't-build gate.
2. **Phase-2 protections** — `§A5` (delete the middle layer, D4), `E4`, `S2`,
   `M2` pre-fill. These are the class of defect that silently voids a battery.
3. **Baseline** on `nys7cfrt` once D2's diff confirms it.
4. **Part B implementation** — §C Phase 3 as written.

~~**Parallel track:** the D8 controller session~~ — ✅ **done 2026-08-06.**
`kind: ratio` is implemented and unit-tested; what remains is the one arm that
measures its plant gain, which rides along on any `naive` run.

## 2e. Cross-cutting

Three control loops can weld themselves off silently, sharing one shape: a
one-sided drive whose zero is indistinguishable from "satisfied."

- **LR recovery** when `recovery_target_frac ≤ cut_ratio` — voided three shipped
  batteries (`P6`). **Goes away with D4.**
- ~~**The bwd balance drive**~~ when its target sits above the metric's range —
  ✅ **closed 2026-08-06 (D8).** Fixed by removing the clamp, not by reporting
  around it: `kind: ratio`'s error is signed, so zero means *at setpoint* and
  nothing else. The two legacy kinds keep the defect.
- ~~**The buffer servo**~~ — `E5` **closed into §B7** (D22). Its replacement's null is *derived* (`weighted_replay_err ≥ fwd_err`, null exactly 1), so there is no `bar` to sit outside a range.

`S2` generalises to all three, a few lines per loop.

---

# Part 3 — Closed

## 2026-08-08 — LR controller v7, and the config TODO sweep

**One session, driven by the `# todo` markers in `mk_dev.yaml`.** Two of them
were the whole LR rebuild; the rest were deletions and comment debt. What
shipped:

### The controller

| | v6 | v7 |
|---|---|---|
| Envelope | ramp → hold → **decay** | ramp → hold. Decay deleted (N2/D7) |
| Peak | a config constant | **servo state** when the key is written `auto` |
| Hot | cut tier → latch → recovery ramp → AIMD | **deleted** (D4/§A5) |
| Diverged | reset tier → rewind + cut | unchanged, renamed `divergence_*`, bar at ~1e9 |
| `auto` on `lr_*` | anchor × 25/T | **servo-managed**, seeded at `seed_lr` |

`auto` changing meaning is the load-bearing part. The old rule promoted one
battery's number to a law; D3 (revisited) already recorded why that fails —
**every run is a transfer** — and there was no mechanism to replace it with until
the probe existed. A float still means a fixed peak, so `lr_fused: auto` beside
`lr_back: 3e-4` is legal and means what it looks like.

Deleted outright, with their code: `reuse_prior` (prior auto-discovery),
`_terminal_policy_state` + `terminal_logw_std` / `terminal_box_violation`, the
five variable-integrator keys and the four alternative discretizers,
`anchor_buffer.mcmc` + `_metropolis_reheat` + `_reheat_geometry`, and
`adaptive_lr.{enabled, hold_steps, decay_*, cut_*, reset_*, fire_cooldown_steps,
recovery_*}`. All 21 are in `preflight_config`'s `_RETIRED_KEYS` with their
replacement named.

### 🔴 Found by the smoke arm: the servo would have been INERT, silently

At `lr ≈ 1.4e-6` the probe returned **`downward` on 100% of fits** — `α*` always
`nan`, `bad_rate` pinned at 1.0, and the loop unable to actuate at all. The seed
was 1e-5 precisely so the servo would climb out on evidence, and the evidence was
structurally unavailable at that LR.

**The two signs of `loss_delta_rel` mean opposite things and v6's fit collapsed
them.** A concave fit that is *descending* (`l1 < l0`) is not a broken model — it
is a window too short to contain the basin, i.e. exactly *"your step is too
small"*. `step_probe.py` now splits `beyond` (concave + descending → `α* = span`,
a lower bound, and a **usable** reading) from `downward` (concave + ascending →
still void). `fit_beyond_rate` ships alongside `fit_ok_rate`; *beyond → ok as the
servo climbs* is the signature of a working loop.

Worth recording as a class: **the smoke arm was 150 steps and it caught a defect
that would have made a 4-arm, 6-GPU-hour battery measure nothing.** Every arm
would have sat at its seed and reported a clean, stable, entirely fictional
result.

### Two latent defects found on the way

- **`mk_dev.yaml` could not load.** `pinned.fwd` was 0.2 against `fracs.fwd`
  0.05, and the `kind: ratio` balance declared no `bounds` — both hard
  `ValueError`s at `Stage` parse. Fixed, and `configs/lr_aug08` now regenerates
  from it so the parse is exercised by the generator.
- **`load_yaml` opened configs in the LOCALE's encoding** (cp1252 on Windows), so
  any non-ASCII byte in a config — a µ, a degree sign, an accented path — raised
  `UnicodeDecodeError` on one machine and loaded fine on another. Pinned to
  utf-8.

### `min_fracs` now binds under every balance kind

It was read only by `_nudge_mode_fracs`, the *lexicographic* path — so a stage
declaring both `min_fracs` and a `ratio` balance had its floors silently ignored
by the integrator. The floor was there; it just was not a floor. `_parse_bounds`
now folds it into `bounds` and hard-fails on an inconsistent pair, which keeps
**one** live bound per mode (R3's complaint) rather than adding a fourth.

### `z_calibration.mode: replay` — built, exercised, and **measured harmful**

A third mode: the same Huber TB Z gradient over **stored** trajectories, so no
rollout and no energy call. It raises rather than mis-calibrate when `prioritise`
is off, since scored admission skims the residual tail (B0a: `birth_loss` 23.7 vs
10.9).

**`lr_aug08` v1_zcalrep ran it — 400 steps at lr 1.25e-4, differing from the
rollout control in this key alone:**

| | rollout | replay |
|---|---|---|
| `bwd/tb_err` | 16.02 → **15.04** | 15.71 → **21.74** |
| `fwd/tb_err` | 22.39 → **17.91** | 21.65 → **23.90** |
| `fwd/tb_resid_clipped` | −0.04 | **−1.97** (D29 bar is ±0.5) |

**Rollout improves both branches; replay degrades both and breaches the very
invariant it exists to hold.** The code is correct — the *estimator target* is
wrong. Uniform intake removes the admission bias but not the **lag**: the buffer
is a ~τ-step-delayed sample of the forward measure, and under a moving policy
that delay is itself an offset. Z pinned to a stale fixed point mis-centres the
fused forward TB loss, which reads the same `log_Z` — which is how a Z-only
sidecar ends up dragging the policy.

Caveats: one 400-step arm, and `threshold` was lowered to 0.2 so the loop would
fire at all (~9 Z steps per train step, an aggressive dose). The direction and the
breach are not in doubt; the magnitude at a gentler dose is untested.

**Stays in the tree, documented as harmful, not recommended.** The lag is a real
obstacle rather than a bug, and a fix would have to correct for it rather than
tune around it.

### `configs/lr_aug08` — the battery that reads D31

Four arms off one post-transient resume, 5400 steps each, varying only what owns
the LR: `a_fixed` (1.25e-4 pinned, the reference — must reproduce local_aug08's
15.14 or the rewrite changed training and the battery is void), `a_climb` (servo
from 12.5× below), `b_climbB` (its seed replicate — the *converged LR* is the
quantity that has to be stable), `b_descend` (servo from 3.2× above, testing the
direction pair D's measurement supports). Buffers are held at the **old**
configuration deliberately, so the LR axis is not confounded with B7b.

## 2026-08-08 — documentation sync pass

Not decisions: a **propagation pass** over the module docs, which had fallen
behind the 2026-08-07 build (`to_do_rebuild.md` §0b/§0c/§B5/§B7b–d) and the
2026-08-08 commit `880e2cb`. This is the failure mode the doc set exists to
prevent — rulings recorded where they were made and never carried into the module
they govern — so what moved is listed here rather than only in the modules.

| Where | What was stale | Now |
|---|---|---|
| `module_buffers.md` | replay described as one scheme; the prioritised draw, memorisation sensor and B7b intake/purge changes were absent | Two-regime framing throughout, plus **B7** (prioritised draw, `floor_frac` re-measured to 0.25), **B8** (memorisation sensor, derived `1/e` bar), **B9** (the two wiring bugs), **B10** (`update_logw_stats` had no caller) |
| `module_modulators.md` | D1 "sensor rationale stale" read as an open recalibration; D2 "not configured on this route" | D1 **superseded** by a second sensor with a *derived* bar; D2 revised (now on 6 configs incl. rb0808 arm 20); **D7** the servo was resolving `None` every tick; **D8** wired but authority unproven; S1 **unblocked** |
| `module_losses.md` | no `sample_weights`, no `path_grad_last_k` | **L10** (self-normalised IS at the final reduction, and the ⚠ Huber incompatibility), **L11** (truncated forward path gradient + `reward_grad_clip` as a *separate* destabiliser) |
| `module_training_modes.md` | "the replay draw is uniform" stated flatly | Now marked regime-dependent, with the default (majority) case named; **M7** the mis-wired draw was sign-inverting, **M8** the step probe's batch |
| `module_protocol.md` | **`kind: constraint` was documented nowhere in the doc set** despite being implemented and selectable | **P8** added (asymmetric integrator, `priority` as a gain multiple, why an unreachable bar is designed-for); P4 revised |
| `module_metrics.md` | three new metric families unlisted; no statement of the tracker-vs-report distinction | **§3a** (the families, and which key is class 1), **T7** (two publication paths, only one controller-readable), **S5** (preflight controller-named metrics) |
| `synthesis.md` | §1's thesis stated as settled; doc map linked two files deleted 2026-08-06 | **§1a** records the `freeze_policy` contradiction and what survives it; doc map corrected and a finding-ID collision note added |

**One general rule came out of it, with three instances now** (`module_modulators.md`
D7, `module_metrics.md` T7, and the older drive-liveness case): **an unreadable
sensor and a satisfied controller produce identical silence.** Two cheap guards
follow — a metric a controller reads must go through `metric_tracker.update`, and
every controller must emit its *actuator* alongside its sensor.

**Also corrected, then done:** `D5`'s row below says uniform admission "retires
`admit_temperature`". As of 2026-08-08 it did not yet -- B7b shipped opt-in, so
`admit_temperature` was live on `mk_dev` and on 22 of 26 rb0808 arms, and the
four cap/temperature keys were read with no default (deleting them raised).
**As of 2026-08-10, D5 is finished**: admission is unconditionally uniform and
the four keys are retired (`utils.py` `_RETIRED_KEYS`) -- a config setting any
of them now fails at load. `mk_dev.yaml` still sets `admit_temperature` and
friends as of this note; those lines need deleting by hand (mk_dev is
user-owned). See `module_buffers.md` B0.

## 2026-08-06 — decisions session

| # | Ruling | Consequence |
|---|---|---|
| **D8** | **`kind: ratio`** — one setpoint (the exchange rate between the two halves of the residual field), signed log error, integrated in the logit of the numerator mode's share, bounded, with a `converge_floor` that retires the loop. Implemented [`_ratio_tick`](../protocol.py:1467) | → `module_protocol.md` P5. Three of my own arguments were **withdrawn** en route and are recorded there: the "guard not allocator" recommendation (built on the LR-dead replay_july26 arms and a 2000-step screen — the user's own D8 entry had already flagged the confound); the self-scaling bar against `fwd/tb_err` (structurally ~6× the coverage metric, so it pins the bwd drive at 0 — the same inert-side failure by a new route); and the leak toward an idle mix (correct for a guard, wrong for an allocator, where it biases the equilibrium off the balance condition) |
| **D11** | **No anneal block** | Falls out of D8 — one setpoint *is* the judgement, so there is nothing to anneal toward, and an anneal would reimport the marginal-breach ratchet |
| `P7` | **`fwd` pin answered analytically** — binary gate above `deactivate_threshold`, not a dial | The config comment's candidate-pool justification is **void**: fracs are loss weights and every active branch runs the full batch |
| — | **J = `bwd/jensen_z − fwd/jensen_z`** is `KL(Q‖P) + KL(P̂‖Q)` exactly, `log Z` cancels | The objective is two logged scalars. But θ acts on its **rate**, and it carries a run-varying limit cycle, so it is a **scoring statistic, not a control input** — which is what makes D8 a setpoint-holding loop rather than an optimiser. Caveats: `P̂` is the buffer not `P`; `weighted_bwd_beta` tilts `E_P̂`; pooled |
| — | `best` checkpoint selects on **cycle phase** | [train.py:1587](../train.py:1587) takes a running min over a 50-step-sampled oscillating series, so it picks a trough — a reset-tier rewind then restores a state that regresses toward the cycle mean, and `best` is biased low by ~the half-amplitude so it is not comparable between arms. Trailing median instead of running min. **New item, not yet actioned** |
| **D1** | **(b) — the plan is the whole plan.** Fold the missing areas in as Parts C+ | → `N1`. Sharpened by your framing that replay must re-earn its place: if it doesn't, most of Part B is moot and the balance controller becomes the *whole* mix question |
| **D3** | **Build the servo** — proceed with §A3/§A4 as proposed | §A6's probe demotes from kill-gate to instrumentation. Honest residue: if α\* *is* flat up to the cliff, the servo can't prevent it, so the probe still scopes whether a separate hard ceiling is needed |
| **D4** | **Delete the LR middle layer** — resolved *by* D3, not independently | Your reframe was right and it dissolves the question. Three regimes, two owners: *slightly hot* → α\* < 1 → smooth `peak` nudge, false-positive cost ≈ 0; *diverged* → coarse bar → reload. The middle layer occupied a third regime — "cut hard and latch, but don't reload" — which has **no distinct response left** once the servo cuts continuously on a better sensor. So "should we always reload when a little hot?" answers **no**: a little hot now has a response that costs nothing, so it never needs escalating. And your own point — *"if we are only looking at hard blow-ups we can use almost any metric"* — retires §A2's entire calibration problem (the 3.3× bar drift; tw_july31 arm 14 dying in the gap between cut bar and reset bar) |
| **D5** | **Uniform admission** — simpler for a start | ✅ **DONE 2026-08-10.** → `to_do_rebuild.md` §C Phase-3 step 3. Retired `admit_cap_max` / `admit_cap_min` / `admit_cap_health_h0` / `admit_temperature` (`utils.py` `_RETIRED_KEYS`); admission is unconditionally uniform (`train.py` `manage_replay_buffer`). `mk_dev.yaml` still needs the keys deleted by hand |
| **D6** | **Branch-asymmetric `beta` is always available**, at minimum as an option. Consistency in *reporting* is the real constraint | → `N4`. Unblocks §B6, the β×κ arm, and `P9`'s `beta_bwd` lever |
| **D7** | **No step budget.** ~7 days on an A100, typically <100k steps. Train to convergence, don't strangle it | → `N2`, and `P2` retired. A scheduled decay presupposes a horizon that does not exist |
| **D9** | **`fwd/tb_err_worst`** — the simplest, most direct health metric | → `N3`. Bar needs recalibrating; also moves the gate off the metric D6 would make non-portable |
| **D10** | **ELJ / real crystals, unconditional, fixed step counts.** End-to-end functionality check. Every phase-boundary buffer fill shrunk to be fast | → `N5` |
| **D12** | **Dissolved.** The constants are the *replay* buffer's ([train.py:4726](../train.py:4726), `rb_cfg`), not the anchor buffer's — `questions.md` misattributed them and I carried the error forward. D5 retires them outright, so there is nothing to derive | Anchor *acquisition* genuinely defers to conditional training, consistent with `P4` |
| **D13** | **Not deliberate**, and superseded | §B7's `τ` from measured `λ` replaces the `mean_lifetime` / `mean_residence_steps` pair |
| **D14** | **`loss_clip: -1`** | Effectively unused today, and more so under D4 |
| **D16** | **Eval `beta` must line up with traj `beta`, per trajectory type. Branches need not agree** | → `N4`. A cleaner answer than either option I offered: assert the per-type invariant instead of emitting or dividing by a global `beta` |
| **D17** | MMD family — **delete** (already ruled in `module_metrics.md` §8 S1) | → `T1` |
| **D18** | Collapse `bwd`/`replay` samplers — **rejected** (already ruled in `module_training_modes.md` §8 S2: *"they do different things"*) | no action |
| **D19** | Three frac bounds — **not a defect** (already ruled: *"intended to be set intentionally"*) | → `R3` as documentation only |
| **D20** | Priority order | → §2d |
| **D21** | **Re-read** the confounded batteries by live LR — *"just one more scalar record to check per run"* | → `P6` |
| **D15** | **Keep the flag; the conditional behaviour is what it is for.** Not a deletion | The two layers already exist and already compose: `scramble_conditions` is the *intent* flag, `scramble_applicable()` is the *structural* guard, and [train.py:2892](../train.py:2892) ANDs them. So the flag is correct and inert here by design. The only false thing was the config comment claiming "unconditional prior by construction" — **fixed** in `mk_dev.yaml`. `E6` closes as *not a defect* |
| **D22** | **Fold the buffer servo into the replay redesign**, under population management, and have it **enforced automatically** rather than by hand-set thresholds. **Not conditional-only** — replay overfitting is bad on any route | Overrides my recommendation, correctly: §B7 already carries the mechanism, and it is *better* than the servo it replaces. Under D5's uniform admission the weighted statistic has a **derived null of exactly 1**, so the precept `weighted_replay_err ≥ fwd_err` becomes a hypothesis test rather than a calibrated 2×. `E5` closes into §B7; nothing to re-derive |
| **D24** | **Delete `questions.md` and `register.md`** | done, after verifying coverage: all 30 questions and every register ID map into this file. One item — the orphaned `uw_global`/`uw_max` instrumentation — existed only in `questions.md` and was carried across as `N6` before deletion |
| **D3** *(revisited)* | Ruling unchanged — **build the servo** — but its stated justification is **retracted**. §A3a.5 argued the unconditional route needs no servo because aug02 measured the optimum at 4e-4, making a fixed 1e-4 "inside tolerance." | Your objection is correct and it defeats that argument: aug02 measured **one point** — one energy, one `T`, one `W`, one clip — and the problem shifts constantly (energy function, space group, conditions, models, rollouts). Promoting one battery's number to "the unconditional optimum" is the [[findings-must-generalize]] failure. There is no stable *here*, so the transfer-vs-extraction framing collapses: **every run is a transfer.** The ruling now rests on stronger ground, and the scope narrows — *"the main point is that we have a plausibly much better sensor; everything else should be kind of straightforward"* |
| **D23** | **Extract, then retire.** Anything principled / general / qualitative / mechanistic moves into the docs; contextual, specific, or hypothetical content does not. Retired memories go to `memory/old/` rather than being deleted | separate work item — ~40 mechanism memories the module docs now supersede. `adaptive-lr-controller` is the confirmed-stale exemplar |
| `R2` | rename `*_frac` → `*_weight` — **declined**: the value is a fraction *of* the total weight | the fact still needs stating where the fracs are set |
| `R7`/`M2` | controller's entry mixture ≠ its configured one — **ruled a defect** | pre-fill; → §C Phase 2 |
| — | "replace `_loss_weights` in the replay draw" (§C step 2) | **corrected** — `draw_replay_sample` passes `weighted=False`, so replay draws **uniformly** today. Step 2 *adds* a computed draw |
| — | `questions.md` + `register.md` disagree | **consolidated into this file** |

## 2026-08-05

| # | What | Resolution |
|---|---|---|
| — | "would pushing `log Z` up lubricate buffer absorption?" | **answered — the sign is inverted** (L9). An offset `c` on `log Z` *is* an MLE-on-buffer term of weight `−2c`, since the fwd branch's score is zero-mean on-policy. Up = negative-weight MLE = unlearning the buffer |
| — | `beta`'s replacement candidates (log compression / grad-norm clipping) | **both eliminated** (L8d) — `soft_clip` is redescending, so the deepest rows get the *least* drive; a norm clip bounds step magnitude rather than per-row influence. The Huber *shape* is correct |
| — | was `level_gap` tried and found wanting on TB? | **no — never run on a TB branch at all.** Sole nonzero setting is a VarGrad stage in the archived 07-29 preset |
| — | is eviction what stalls buffer absorption on the prior route? | **ruled out** — the prior buffer represents a *distribution*; rows are exchangeable draws and churn is a feature. May still bind on *replay*, whose rows are discovered rather than drawn |

## 2026-08-03

| # | What | Resolution |
|---|---|---|
| `E1` | `vg_lme` unconditional normaliser | **fixed** — `math.log(log_ratio.shape[0])` |
| `E3` | forward-only finiteness assert | **removed** — `step_loss` already guards the gradient norm for both branches |
| `S-dead(a)` | `CrystalBuffer.purge()` | **removed** |
| `S-dead(b)` | `normed_smoothness_loss`, `soft_saturate` | **removed** |
| `E7` | L7's silent no-Z-trainer combination | **warning added** — config-level, so it reports rather than raises |
| — | anchor health gate hardcoded metrics | **now config-driven**; the `tb_err_worst` upgrade became a config A/B, chosen 2026-08-06 (D9) |
| `P4` | anchor buffer's role | **resolved** — anti-forgetting (conditional) + shoulder discovery via thermal widening (unconditional). Its constants stay untested until the conditional route is live |
| `S-menu` | delete the 9 unreachable loss terms | **rejected** — the menu is deliberate |

---

# Appendix — conflicts found during consolidation

| Item | `questions.md` said | `register.md` said | Resolved as |
|---|---|---|---|
| **`beta` uniformity** | assert it | **do not** assert | **register** — confirmed twice over (`R5`, `module_metrics.md` §8 S3), and D16 replaced the whole fork with a per-trajectory-type invariant |
| **`MetricTracker` staleness** | flat "**Yes**" | "do both or neither" (pair with `S2`) | **register** — `S4` keeps the pairing condition |
| **`S5` sampler collapse** | not listed | "collapse into one sampler" | **rejected** — already ruled in `module_training_modes.md` §8 S2 |
| **`*_frac` rename** | "**A: Yes.** Cheapest correction of the most-misread thing" | **DECLINED 2026-08-06** | **register.** `questions.md` was never updated — a live example of the drift this file exists to stop |
| **Priority order** | none stated | (1) controller shape, (2) `E4`+`S2`, (3) `R2` | **re-derived from the 2026-08-06 answers** → §2d |
| **Anchor cap constants** | "the anchor buffer's cap constants" | same | ❌ **both wrong.** They are `rb_cfg` — the **replay** buffer's ([train.py:4726](../train.py:4726)). I carried the error into D12 before catching it; D5 retires them |

## Rulings the register never picked up

Five items were live in the register but had **already been decided** in a module
doc. The register was simply not updated when the ruling was made.

| Item | Ruling, and where |
|---|---|
| MMD family (`T1`) | ✅ **delete** — `module_metrics.md` §8 S1 |
| Collapse `bwd`/`replay` (`S5`) | ❌ **rejected** — `module_training_modes.md` §8 S2 |
| Three frac bounds (`R3`) | ❌ **not a defect** — `module_training_modes.md` §8 S3 |
| `scramble_conditions` (`E6`) | ✅ **accepted** — `module_training_modes.md` §8 S4 |
| Assert `beta` uniformity | ❌ **no** — ruled twice, `R5` and `module_metrics.md` §8 S3 |

**Numbering collision:** `module_training_modes.md` §8 uses module-local `S1`–`S4`
that collide with the global `S1` (LR middle layer) and `S2` (drive-liveness).
Its `S1` is the global `R2`; its `S2` is `S5`.

---

*Warrant classes: **derived** (follows from the math) · **measured** (A/B'd, run
cited) · **inherited** (came from elsewhere, never re-examined here) ·
**arbitrary** (someone picked a number) · **contested** (conflicting evidence).*
