# The conditional VarGrad objective

Argument. Why the conditional route's objective is shaped the way it is, which of
its choices are forced and which are free, and where the shipped configuration
departs from the construction.

Math is plain text, as in `module_losses.md`. Notation: `u_i = log w_i` for row
`i`, `c` a condition, `g` a group size, `d_i = centre - u_i`, `psi = dL/dd`.

**Grades** per `PROTOCOL.md`. `MECHANISM` here means derived, or verified against
code in the tree. Anything below it carries a scope line. A claim sourced from a
run reading I have not personally reproduced is marked `CONJECTURE` and listed in
§7 rather than promoted.

---

## 1. The axiom

**A1 (estimand).** For every condition `c`, the terminal marginal must satisfy
`P_F(x|c) ∝ R(x|c)`, judged on held-out `c`. Nothing else is the objective.

**A2 (admissibility).** A design choice is justified only by what it does to the
influence function *evaluated at the residuals the run actually produces* — never
by the name on the loss term.

A1 is the goal; A2 is the rule for what counts as an answer. Every layer below is
A1 asking a question and A2 deciding what settles it.

Three independent derivations were run from different starting axioms
(estimand-first, gradient-first, operating-point-first). Where they converged the
reading is forced; where they diverged, the divergence localises a genuinely free
choice that the shipped config made silently. Both facts are recorded below.

---

## 2. The construction

### L0 — Estimand and the trajectory surrogate

`P_F(x|c)` is not evaluable without marginalising over trajectories, so a
trajectory-level surrogate is **forced**. With

```
log w(tau|c) = log R(x|c) + log P_B(tau|x,c) - log P_F(tau|c)
```

factor `log w = A(x) + B(tau)` where `A = log R(x|c) - log P_F(x|c)` and
`E[B|x] = -k(x) = -KL(P_F(.|x,c) || P_B(.|x,c))`. Then exactly

```
Var(log w) = Var_x[A - k]  +  E_x[Var(B|x)]
```

A1 wants `Var_x(A) = 0`; the estimator minimises the whole right-hand side.
`MECHANISM`.

**Free choice: is `P_B` learned?** Shipped `learn_pb: true`, `pb_var_range: 6`
([qm9_cond.yaml:140](../configs/shakeout_aug16/qm9_cond.yaml:140)).

The objection — that a learned `P_B` de-identifies A1, because the objective can
be reduced by shaping `k(x)` instead of `A` — does not survive pricing both
terms. The joint argmin over `P_B` at fixed `P_F` is `P_B = P_F(.|x,c)`, which
sends `k -> 0` *and* `Var(B|x) -> 0` simultaneously; at that point
`log w = log R(x) - log P_F(x)` exactly and **the trajectory objective becomes
the marginal objective**. `pb_var_range` is a penalty ramp, not a slack budget.
`MECHANISM`. Shipped setting is correct.

### L1 — What a group is

**Forced.** The estimand is per-`c`, so the group key is `condition_id` and only
`condition_id`. Grouping by the repeats tile is a different object — on `bwd` the
tile shares the *terminal*, so the reward cancels in the contrast and the result
is TBC in disguise. Grouping across conditions estimates no condition's
normaliser. `MECHANISM`.

**Free.** Group size, which is **emergent**: `rows / n_conditions`, not a knob.

`vg_group_size_mean` is **row-weighted**, i.e. size-biased `E[n^2]/E[n]`, not the
mean group size ([gflownet_losses.py:929](../gflownet_losses.py:929)). Verified:
group sizes `{2, 8}` report `6.80` against a true mean of `5.00`. The docstring
says the weighting is deliberate and gives the right reason — estimator variance
depends on the group the average *row* sits in — but the number is then read as
"the group size" in battery design, which it is not. `MECHANISM`.

Consequence: whenever the true mean exceeds 2, groups of 3+ are common, and the
`g = 2` identities below (where `detach_center` is inert and the Huber's
tail-skew leftover vanishes) do **not** apply.

### L2 — The centre

Two choices hide here. One is free; one is settled by algebra and is currently
documented backwards.

**Free: mean vs median vs logmeanexp.** Shipped: group mean (`vg_lb`,
`lme=False`). Mean-centred absolute deviation is a valid, location-invariant
dispersion functional, zero iff the group is degenerate; at `g = 2` it is exactly
the Gini mean difference. The mean is also the Jensen lower bound on `log Z(c)`,
which is what `emp_z` regresses onto, so a median centre would silently redefine
`Z(c)`. Shipped is fine. `MECHANISM`.

**Settled: the detach flag.** With `d_i = centre - u_i`, `centre = (1/g) sum_j u_j`,
`psi = clip(., ±beta)`:

```
un-detached:  dL/dtheta = sum_j [ psi_bar - psi(d_j) ] * grad u_j     coefficients sum to 0 EXACTLY
detached:     dL/dtheta = sum_j [         - psi(d_j) ] * grad u_j     coefficients sum to -g*psi_bar
```

Measured directly on the shipped code, one group of 5 with a skewed tail:

| `vg_detach_center` | per-row coefficients | sum |
|---|---|---|
| `0` (shipped) | `-4, -4, -4, -4, +16` | `+0.0e+00` — exactly centred |
| `1` | `-10, -10, -10, -10, +10` | `-3.0e+01` — carries a common mode |

**The un-detached form is the exactly-centred one.** The detached form is what
leaves a residual force along the batch-mean score — an MLE-on-buffer direction
whose weight is set by tail skew rather than by any coefficient. `MECHANISM`.

The docstring at [gflownet_losses.py:964](../gflownet_losses.py:964) attributes
the leftover to the opposite form. The repo's own test states the correct closed
form ([test_vg_detach_center.py:143](../test_vg_detach_center.py:143):
`grad_undetached_j = psi(d_j) - (1/K) sum_i psi(d_i)`), so the code and its test
agree and only the prose is wrong. The shipped setting is correct; the stated
reason for preferring the detached form is inverted.

### L3 — The functional, and the regime

[gflownet_losses.py:1035](../gflownet_losses.py:1035):
`vg_loss = beta * F.smooth_l1_loss(vg_center, log_ratio, beta=beta)`. The leading
`beta` cancels PyTorch's `1/beta`, giving `0.5 d^2` inside the knee and
`beta|d| - beta^2/2` outside — so `psi(d) = clip(d, ±beta)`, a unit-curvature
quadratic core of half-width `beta = 10` nats. `MECHANISM`.

`logw_std_within` is `centered_w.pow(2).mean().sqrt()`
([utils.py:1968](../utils.py:1968)), centred on `z_jensen_ref` — the same
per-condition group mean the loss centres on, over the same batch. **It is the
RMS of exactly the loss's own residual**, with no Bessel correction, so it reads
`sigma * sqrt(1 - 1/n_bar)` with `n_bar` the unweighted mean group size.
`MECHANISM`. Two consequences:

- `logw_std_within` vs `beta` *is* the saturation diagnostic, already logged. No
  new instrumentation is needed to establish the regime.
- The metric drifts with group size, which is emergent, and it is simultaneously
  a balance-controller input and an exit-gate metric.

`next_battery.md:97` records `fwd/logw_std_within` operating at **40–190** on the
live conditional battery. Against a knee at 10 that is a spread-to-knee ratio of
4–19. Distribution-free, `E[d^2 * 1{|d| > beta}] >= sigma^2 - beta^2`, so at
`sigma >= 40` **at least 94% of the second moment lies outside the knee**.
`MECHANISM` for the inequality; the 40–190 figure is `REPLICATED` per
`next_battery` §2 (scope: conditional QM9 route, `var_conditioning`, long arms).

This is not "a quadratic with a robust tail". **It is an L1 objective with a
small quadratic core**, and that changes what it controls:

```
above the knee   L ~ beta * sum_i |d_i| + const
```

Two configurations with the same `sum |d|` cost the same and can differ
arbitrarily in `sum d^2`: `n` rows at `|d| = D` and one row at `|d| = nD` both
cost `beta*n*D`, but the second carries `n` times the second moment. **The
objective's level set spans configurations whose RMS differs by up to `sqrt(N)`.**
`MECHANISM`.

So: the objective controls a Winsorised **first** moment of the within-condition
residual and leaves the **second** moment free to within a factor of order
`sqrt(N)`. Note this also means the marginal cost per nat in the tail is a
constant `beta > 0` — the tail is not *cheap*, it is *unselected*. Bounded
influence buys indifference, not preference; no "the optimiser is paid to put
mass in the tail" story is required or supported.

**What is forced:** the loss family, given A2. **What is free:** `beta`, which is
not a robustness knob but a declaration of *which moment you are matching*.
Shipped `beta = 10` with no derivation found anywhere in the tree; other branches
ship 80. `arbitrary`.

### L4 — The optimiser

Adam, fused stage. Its update is `m / (sqrt(v) + eps)`, so scaling a single
active loss term by `lambda` scales `m` by `lambda` and `v` by `lambda^2`, and
the two cancel. Therefore `beta` is **not** a gain and **not** a stability knob.
It survives only as (i) the knee location, i.e. which moment is matched, and
(ii) the mixing ratio between branches sitting at different saturation.
`MECHANISM`.

This retires two prescriptions in one stroke: "raise `beta` for stability", and
"a constant-magnitude force cannot define an equilibrium, so the transition
latches". Both are gradient-*magnitude* arguments aimed at a
magnitude-normalising optimiser.

It also promoted the mean-over-terms divisor to a real allocation knob — but
that divisor is **gone as of 2026-08-26**: terms now combine by SUM over the
active set, so a term's effective weight is its own `coeff` and nothing else.
Before the change the effective weight was `coeff / n_active`, which is how a
zero-gradient sidecar could set the policy's step size by being counted. Any
number quoted from a pre-change run on a two-term branch is at half the
coefficient its config states. `MECHANISM` (`module_losses.md` L1).

### L5 — The level (Z)

**Forced.** Condition-grouped VarGrad annihilates the level by centring, and
`tb: 0` on every branch of this stage, so `log_Z_learned` enters **no** policy
loss. On this route Z is a **readout**, not a training signal for the sampler.
`MECHANISM`.

That is the correct frame for the whole Z-sidecar family: `emp_z`,
`emp_z_persistent` and `z_level` train the flow head and must not reach the
sampler. Enforced at the source since 2026-08-17 —
`condition_grouped_empirical_z`, `vg_lb` and `vg_lme` return a detached estimate,
so the regression target cannot push back. Verified on the real model: sampler
gradient exactly 0 from every sidecar, against 1.05e6 from the pre-fix code.
`MECHANISM`.

### L6 — Allocation

[protocol.py:1720](../protocol.py:1720) and 1817: `drive: relative` gives
`s = max(v/T - 1, 0)`. With `T = 1.0` this makes `1 + s = v` identically, and
with `default_boost 0.5/0.5` the aim collapses to

```
share_fwd = v_fwd / (v_fwd + v_bwd),   clamped to [0.1, 0.9]
```

**The batch split is literally the ratio of the two branches' residual RMS.**
`MECHANISM`. With `targets: {fwd: 1.0, bwd: 1.0}` against a metric operating at
40–190, the offset is inoperative — and the controller's own docstring
([protocol.py:1734](../protocol.py:1734)) says the subtraction "is what makes the
equilibrium mean something… an un-offset ratio equilibrates on those floors
rather than on need, and the side with the larger floor wins permanently".

Two further defects, independent of the rail:

- The sensor is the RMS family — by L3, precisely the coordinate the objective
  leaves free. **Allocation is steered by the tail.** `MECHANISM`.
- `alpha: 0.01` at ~10 steps/tick is a ~1000-step constant; the same docstring
  requires it to be slower than the 1–2k-step absorption cycle and recommends
  0.002–0.005. `MECHANISM`.

The stage's exit bars (`fwd/logw_std_within < 6.0`, `bwd < 3.0`) sit 7–30x below
the operating level on a stage that is **terminal by design**. Dead claims.

---

## 3. Where the shipped objective departs

| Departure | Class |
|---|---|
| Docstring inverts `detach_center` | **Doc defect** — shipped setting correct, prose backwards |
| `vg_group_size_mean` read as the mean group size | **Reporting defect** — it is `E[n^2]/E[n]` |
| `logw_std_within` un-Bessel'd with emergent `g` | **Instrumentation defect** — a controller input and an exit gate |
| balance `targets: 1.0/1.0` | **Defect** — the code's own docstring names this failure |
| balance sensor = RMS family | **Defect** — steers on the coordinate L3 leaves free |
| `alpha: 0.01` | **Defect** — 5x faster than its own docstring allows |
| exit bars 6.0 / 3.0 on a terminal stage | **Defect (cosmetic)** — delete |
| `beta` shared with `ConditionLogZTracker.clip_beta` ([train.py:1156](../train.py:1156)) and `quick_tb_stats` (2918/2949/4379) | **Defect** — the knee and the ruler move together, blocking any `beta` experiment |
| anchor gate: ruler `tb_resid_clipped`, bar 0.5 inherited from `z_calibration`, which is off on this route | **Defect** — a bar does not survive a ruler swap |
| `beta = 10` against sigma 40–190 | **Deliberate trade, undeclared** — robust first-moment matching, but nothing reports the matched moment |
| `learn_pb: true` | **Correct** — objection refuted by pricing (L0) |
| un-detached centre; mean centre; condition grouping on both branches; `tb: 0`; Z as readout | **Correct — all five fall out of A1 and A2** |

**The headline: the objective is roughly right.** Five of its core choices follow
from the axiom unchanged. What is wrong is almost entirely *what reads it* — the
reported fit metrics are second moments or functions of them, and the second
moment is exactly what a bounded influence function declines to control. The run
is steered, gated and judged by rulers measuring a coordinate the objective
deliberately leaves free.

---

## 4. The plateau

The standing result — held-out fit flat while sample quality improves — compared
a **train-stream** quality metric against a **held-out-stream** fit metric. Those
differ by roughly 20x in per-tick precision (a bounded fraction over ~10^4
samples versus a 0.75-quantile of per-condition RMS over ~2x10^3 samples across
~570 conditions). Matching streams is a prerequisite to calling it a mechanism.
`CONJECTURE` pending §7.

What would remain to explain is on the train clock: the second moment degrading
while the quantile/fraction family is flat or improving. Ranked candidates, each
with a free offline discriminator:

1. **The LR floor.** If `lr_ctrl/scale` sat at its lower bound for the final
   third, every "flat" verdict on those arms is confounded with an optimiser far
   below the band where quality responds. *Discriminator: per-arm co-timing of
   the rail with the breakpoint.*
2. **Moment mismatch (L3).** The loss controls `E|d|`; the metric reads
   `sqrt(E[d^2])`; the level set is `sqrt(N)` wide. *Discriminator, already
   logged and never read: plot `fwd/vg_lb` (the loss's own value,
   [gflownet_losses.py:349](../gflownet_losses.py:349)) against
   `fwd/logw_std_within`. If `vg_lb` falls while `logw_std_within` rises, the fit
   family is simply the wrong ruler and no new code is needed.*
3. **Batch-growth artifact.** No Bessel correction means the metric reads
   `sigma * sqrt(1 - 1/n_bar)`; `n_bar` rises as the batch grows at roughly
   constant `vg_n_groups`, giving a mechanical rise. *Discriminator: regress
   `logw_std_within` on `Batch Size` and `vg_n_groups`.* Must be run before (2)
   is believed.

Three objective-level mechanisms proposed during derivation —
saturation-shuts-the-gate, a stiffness latch, and positive feedback in the
balance controller — were each refuted in cross-examination. None should be
pursued.

---

## 5. Permanently undecidable here

Stated so it stops being paid for. Scope: conditional QM9 route,
`var_conditioning`, one seed, current eval cadence.

- Any A/B whose signature lands in the `logw_std_within` / `over_coverage` /
  `scatter_err` family. Replicate spread x0.3–x16.7; a true 3x effect is
  invisible.
- `tb_err_worst` at ±8% resolves only differences above ~25% at one seed.
- Anything on the held-out clock below ~8% per 10k steps.
- Ranking arms on final fit value — fit is inside its own noise in the final
  third.
- `beta = 10` vs `80` on the backward branch: bwd's spread is below the smaller
  knee already, so this is a structural null.

Buy resolution once (eval samples, eval cadence) rather than buying arms.

---

## 6. Sequencing

**Upstream of everything.** Nothing about the objective can be read while the LR
floor is unresolved; it is also candidate 1 for the plateau.

**Free, offline, no training change:** the three discriminators in §4, plus a
`b005_sym_formb` vs `b005_sym` comparison, which now reads as *the size of the
common-mode force the detached form adds* rather than removes.

**Zero behaviour change:** fix the `detach_center` docstring; report
`vg_n_groups` beside `vg_group_size_mean` or rename the latter
`vg_group_size_row_mean`; add a Bessel-corrected series as a **second** key
without redefining the live one; split `beta` into a per-branch loss knee and a
pinned `resid_clip` feeding the tracker and `quick_tb_stats`.

**Waits for the LR battery:** balance controller (targets, sensor, `alpha`,
delete exit bars); anchor health gate (repoint at the tracker's per-condition
gap in nats, or delete).

**Probably never:** changing `beta`, the centre, the group-size policy, or the
grouping. Those are the four things a redesign would touch, and the construction
says all four are correct or free-and-fine as shipped. Reopen `beta` only if a
residue survives §4 — and then as a declared choice of which moment to match,
with the ruler split already landed.

---

## 7. Verification status

Separated because §2–§6 mix things checked in the tree with things read off runs.

**Verified by direct inspection or execution this session:** the `detach_center`
sign (executed on shipped code); `vg_group_size_mean` size-bias (executed);
`logw_std_within` centring and missing Bessel term; `beta` shared with
`clip_beta` and `quick_tb_stats`; `learn_pb`/`pb_var_range`; the balance block,
`alpha`, exit bars and anchor-gate settings; the L3 saturation algebra; the
Z-sidecar confinement (sampler gradient exactly 0, against 1.05e6 pre-fix).

**Not independently verified — treat as `CONJECTURE` until read off a run:** all
per-arm series values (`resid_skew`, `ess_frac`, `lr_ctrl/scale` rail timing,
`z_bias^2 / tb_err^2`, the stream-matched plateau table), and the claim that the
true mean group size is ~3.9, whose arithmetic assumes a batch size that does not
match the live config (`batch_size: 500`, growth off).

**Open contradiction.** `_condition_flow` returns
`self.flow_model(condition_embedding.detach())`
([models/gfn.py:755](../models/gfn.py:755), used at 984/1120/1195), which implies
a Z-only term cannot reach the conditioner. A direct measurement on the real
model during this session showed `emp_z` producing 1.29e2 of gradient on
`conditions_embedding_model` with the sampler at exactly 0. Both cannot be right.
The invariant that matters — no Z sidecar reaches the sampler — holds either way,
but the claim that `emp_z` delivers *nothing* to `get_policy_params` depends on
this and is unresolved.
