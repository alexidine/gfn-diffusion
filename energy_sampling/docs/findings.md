# Findings

Append-only evidence ledger. Entries are **never edited** — a later entry
supersedes an earlier one by naming it. Format and grades: [`PROTOCOL.md`](PROTOCOL.md).

Newest first.

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
