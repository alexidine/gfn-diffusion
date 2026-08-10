# Findings

Append-only evidence ledger. Entries are **never edited** — a later entry
supersedes an earlier one by naming it. Format and grades: [`PROTOCOL.md`](PROTOCOL.md).

Newest first.

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
