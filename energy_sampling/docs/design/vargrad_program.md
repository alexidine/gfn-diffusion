# Conditional VarGrad: why it stalls, what to add, and what to test

2026-08-27. Argument. Scope: the conditional crystal route (`var_conditioning`),
qm9c 5265-condition library, ELJ, T=20. Evidence is one seed unless stated.

Grades per `PROTOCOL.md`. Companions, not repeated here:
`vargrad_objective.md` (the objective's construction and its instrumentation
defects) and `vargrad_convergence_theory.md` (estimator noise, emergent group
size, Huber skew rectification).

---

## 1. The object

Write the trajectory log-weight

```
u(tau) = log R + log P_B(tau|x) - log P_F(tau) = log Z* - log( P_F(tau) / P*(tau) )
```

with `P* = R*P_B / Z*` the target trajectory distribution and `Z* = int R dx`.

Each branch groups its own rows by `condition_id`, centres on that group's mean,
and applies a Huberised deviation
([gflownet_losses.py:1116](../gflownet_losses.py:1116)):

```
c_g = mean_j u_j ,  d_i = c_g - u_i ,  loss_i = beta * huber_beta(d_i)
psi(d) = clip(d, +-beta)      (the leading beta cancels PyTorch's 1/beta)
```

Shipped: `beta = 10` on every branch, `tb = 0` everywhere, group size `m = 2` on
both branches, `level_gap = 0` on the `qm9c_t20_*` family. Branch terms are
**summed** since 2026-08-26. Two level statistics matter:

```
J_F(c) = E_{P_F}[u]        J_B(c) = E_{buffer x P_B}[u]        Delta = J_B - J_F
```

---

## 2. Why the current approach fails

### 2.1 The offset between branches is unpenalised

`dL/du_j = psi_bar - psi(d_j)`, and these **sum to exactly zero** over a group,
at any group size, under the shipped un-detached centre (at `m = 2` the detached
form coincides, since `psi` is odd). So a uniform shift of `u` across a group —
equivalently, rescaling the policy's mass on that group's support — has exactly
zero directional derivative. `MECHANISM`.

⚠ **Flat direction, not null space.** That is a statement in `u`-space. Parameter
motion is `<grad L, g_bar> = (1/m) sum_j w_j S_j` with `S_j` the group's NTK Gram
row sums, which vanishes only if that Gram has constant row sums — a property of
the network, not the loss. At `m = 2` it is
`clip((u_1-u_2)/2, +-beta) * (||grad u_1||^2 - ||grad u_2||^2)/2`. **Unmeasured
(T1).** A flat direction under Adam with a shared trunk is the direction that
moves *most* freely. The offset is **unpenalised, not fixed**.

### 2.2 But it is not meaningless, and the two branch levels are different objects

```
J_F = log Z* - KL(P_F || P*)                        E_{P_F}[e^u] = Z*  exactly
J_B = E_q[log R] + H(q) + KL(q*P_B || P_F)          q = the buffer draw
```

The forward identity holds for **any full-support policy**, and a Gaussian-kernel
diffusion is full-support — so there is no restricted partition function and no
second fixed point. `log Z*` is a *ceiling* on `J_F`, and it is inactive while
`KL` is tens of nats.

The backward identity is the load-bearing one: **`Delta` minus a buffer-only
constant is a KL divergence with a known zero.** It measures how far `P_F` has
moved off the buffer. It is the best mode-drop instrument on this route, not a
gauge artefact to be ignored. `MECHANISM`.

### 2.3 The only coupling is overlap, and it is self-suppressing

`u` is one function of `tau`, so wherever the two supports overlap the branches'
fixed points must agree; the residual freedom is proportional to their
**non-overlap**, `Delta ~= (1-p)(u_bar_out - c_f)`. The restoring force from the
forward branch on the abandoned region is weighted by that region's own policy
mass `m_B`, and clipped at `beta` — so it is `~ m_B * beta` with `m_B ~ e^-62`.
**The gradient that would fix the deficit is proportional to the deficit's own
smallness.** Self-suppressing, not self-correcting.

### 2.4 Mode dropping is invisible to the estimator, not to the objective

`J_F` can sit a fixed number of nats below `log Z*` at vanishing measured
variance if `u` carries a rare heavy right tail — mass `p`, height `M`, with
`p*M^2 -> 0` but `p*e^M = O(1)`. A dropped mode is exactly that. It stays hidden
because the Huber clips that row's influence to `+-beta = 10` however far out it
is, `m = 2` groups almost never contain it, and its sampling probability is
exponentially small. Unlike a fixed-point story this **responds to `beta` and to
group size**, so it is testable. `MECHANISM`.

### 2.5 The forward term is already two-thirds a level push

Regressing each row's gradient coefficient on its own within-branch deviation
(1.0 = intact VarGrad, 0 = pure level force) gives, in the shipped separate-term
configuration: **forward 0.363, backward 1.000**. The backward branch is doing
clean quadratic work; the forward branch — at `sigma_f ~ 30` against `beta = 10` —
has already lost about two thirds of its within-condition signal to the clip. Any
`beta` argument has to start from there, not from a nominal quadratic. `MECHANISM`
(Monte Carlo on the shipped estimator at the measured operating point).

### 2.6 What `Delta` is made of, and who can move each piece

With `q` the buffer terminal law, `pi = R/Z*`, `p_F^T` the forward terminal
marginal:

```
Delta = M + B + D + E
  M = E_q[log pi - log p_F^T]        terminal MASS deficit on the buffer, ANY SIGN   <- THE TARGET
  B = E_q KL(P_B(.|x) || P_F(.|x))   >= 0   bwd-typical trajectory KL
  D = KL(p_F^T || pi)                >= 0   forward terminal error
  E = E_{p_F} KL(P_F(.|x) || P_B(.|x)) >= 0 fwd-typical trajectory KL
```

`MECHANISM`. Two consequences settle questions this document previously left open.

**`Delta = 0` is the correct setpoint.** At `P_F = P*` the two KLs in `M` cancel
and `B = D = E = 0`. So a tether that drives `Delta -> 0` is **unbiased** — it is
not a proxy that happens to correlate with coverage. `Z` never appears anywhere in
`Delta`, so the flow head has `dDelta/dtheta_flow = 0` exactly.

**Only the forward drift and conditioner can touch `M` and `D`. The `P_B` heads
can only touch `B` and `E`.** That is the escape, stated exactly: a level force
that is not gated will be absorbed into the trajectory-KL terms and close `Delta`
without moving one gram of terminal mass. Budget for that route, exact for
diagonal Gaussians with `r = sigma_f/sigma_b` the per-step std ratio:

```
max Delta closable by the P_B variance head = 114 * (r - 1/r)^2 nats
  r = 1.24 -> 21.5      r = 1.41 -> 57      r = 1.50 -> 79
```

⚠ This corrects an earlier figure in this document. "A 24% widening erases all 64
nats" used the entropy derivative alone and dropped the `log P_F` response; the
true value at 24% is **21.5 nats**, and the forward kernel must be **≥43% wider**
than the bridge before `P_B` can eat the whole gap. The two routes are also
**coupled**: inflating `sigma_f` is what makes the `P_B` escape affordable. The
forward route's net sign reduces to `sign(sigma_f^2/sigma_b^2 - rho_B)` minus a
strictly positive reward term — undetermined from anything currently logged, and
that reward term is why the forward route self-limits while the backward one does
not (`P_B` never touches a terminal, so `log R` never opposes it).

### 2.6b Why widening is free: centring deletes the price

The owner's framing — easier to widen a kernel than to move the drift — is right,
and the reason is sharper than "cheap". For a uniform forward log-sigma bias `s`,

```
u = a - Q*e^{-2s} - N_f*s
```

The normaliser `-N_f*s` is the entropy price of widening. **It is row-constant, so
condition-grouped centring annihilates it exactly**, while the quadratic term's
within-group variance falls as `e^{-4s}`:

```
d(Var_g u)/ds = 4[ Cov_g(a,Q) - Var_g(Q) ] < 0  generically
```

**Widening is a free descent direction on both branches, with no opposing term.**
VarGrad does not merely fail to punish inflation; it deletes the term that would
pay for it. `MECHANISM`. This also removes the motive H6 attributes to the
excursions: the policy does not need a level reason to inflate — inflation is
simply downhill. And it is already well under way, with the forward log-variance
head running ~+1.9 to +2.8 above base (7-16x the neutral per-step variance) and
+3.2 of tanh headroom left.

### 2.7 What the training actually shows

Compressed to the claims each number supports. Runs `helr662f` (complete,
20000-28600) and `lj5nyun2`; ⚠ all four `qm9c_t20_*` runs branch from one
step-20000 checkpoint, so n=1 more than it looks.

**The level-free / level-carrying split is total.** Over 8600 steps at fwd/bwd
50:50, `bwd/vg_lb` fell 40% (1.94 -> 1.17), `bwd/relative_under` 1.39 -> 1.08,
`bwd/logw_std_within` 1.97 -> 1.53 — while `bwd/under_coverage` went 61.5 -> 62.8
and `zmatch/delta_mean` 64.36 -> 64.41, a random walk (drift below its own
scatter). `bwd/over_coverage` is **identically zero**: not one buffer row in 8600
steps sat on the over-covered side, so the under-weighting is uniform across the
anchor set — exactly the component a centred estimator annihilates.

**The decisive statistic is a decoupling, not a flat line.** `bwd/jensen_z` does
have a real +0.79-nat trend (monotone over eight blocks, 2.94x its scatter). What
is absent is its coupling to the backward loss: innovation Spearman
**rho = -0.002 (p = 0.96)**, against the forward branch's **-0.887** on the same
samples. Positive control across 27 local runs: the same correlation is
**+0.423 with `level_gap: 1` and +0.022 with it off (Mann-Whitney p = 0.0099)**,
while forward coupling is identical in both cohorts — as a backward-only term
requires. `REPLICATED`.

**Support improved; density did not.** `bwd_outside_fwd_final` 0.42 -> 0.17, i.e.
the forward manifold came to enclose most of the buffer, while the 62-nat density
deficit did not move. The policy can *reach* the anchors and gives them `e^-62`
too little mass. Meanwhile `delta_overlap` (Crooks) sits at 0.003-0.005 and spikes
to 0.11-0.15 **only during the forward excursions** — broadening is the one thing
that has ever opened the channel.

**Nothing about the samples moved.** `emarg_overlap` 0.117 -> 0.117, `w1_kT`
39.2 -> 40.2, Excess Energy P50 46.8 -> 47.3, Reasonable Fraction 0.31 -> 0.33,
Effective Dimension 5.41 -> 5.68. Eight thousand six hundred steps.

**The forward "blow-ups" are an LR ladder.** The divergence counter fired five
times (21600, 21700, 23780, 25160, 27410); `lr_fused` 3.13e-6 -> 1.95e-7, then
parked because `mode: fixed` cannot re-bracket. The excursions are *inflation*
(`over_coverage` 22 -> 46, `step_var` +50%, EffDim up), not narrowing. Do not read
the flat tail after step 24000 as convergence.

**`J_F` is not a level readout.** `J_F = emp_z - z_gap`, and the level term is
flat: reconstructed forward `emp_z` 45.72 +- 1.05, trend -0.016 nats/kstep, while
`jensen_z` swings 10.07 and `z_gap` 11.62 oppositely. Held-out:
`r(vg_lb, eval_test/jensen_z) = -0.849` but `r(vg_lb, eval_test emp_z) = +0.004`.
The co-movement is same-batch arithmetic — r = -0.97 at lag 0, |r| <= 0.04 at
every other lag, slope matching the mechanical `-1/(2*beta) = -0.100`. Read
`zmatch/fwd_level`, not `fwd/jensen_z`: over the run the pooled series fell 1.93
nats while the composition-free per-condition mean **rose** 0.40, because
`weighted_condition_sampling` draws the batch proportional to `Var(log w)` — the
very statistic `vg_lb` reads.

**The one counter-example, and it is the reason not to give up.** `lj5nyun2`,
steps 16770-18510: `vg_lb` 173 -> 51.5 and `jensen_z` -9.9 -> +11.9, of which
**`z_emp` rose 5.21 nats**. The level *can* move. It is the only sustained
improvement in the local record. `OBSERVED`, n=1.

⚠ **Instrument bug, a class not an instance.** [train.py:3785](../train.py:3785)
does an unguarded `stats.update` from `loss_dict` after `quick_tb_stats`, so
`fwd/emp_z` is the emp_z **Huber loss**, not the estimator; `z_gap = emp_z -
jensen_z` fails on fwd by ~15 nats and holds bit-exactly on bwd and all eval
streams. Reconstruct the forward estimator as `jensen_z + z_gap`. Any loss-term
name colliding with a `quick_tb_stats` key is silently overwritten.

⚠ **Three quantities get called "the gap".** 62 = `bwd/under_coverage` (anchors
vs the learned flow head); 64 = `zmatch/delta_mean` (backward level vs forward
level — the one this argument is about); 38.97 = the tracker, which sits 24.8 nats
above the flow head because `tb = 0` looks the persistent Z up and discards it.

---

## 3. Proposals

### 3.1 The form to build: three terms, three knees

```
L = w_f * VG_within(fwd)  +  w_b * VG_within(bwd)  +  w_z * Level(J_F, J_B)
```

with **independent `beta`, group size and coefficient per term**. This is the
owner's decomposition and it is the right one, because the law of total variance
says a single pooled per-condition group computes exactly

```
Var_pooled = lambda*V_f + (1-lambda)*V_b + lambda*(1-lambda)*(J_F - J_B)^2
```

— the same three pieces, with all three knobs fused into one. Writing them
separately buys the pooled *estimand* (one shared centre forces
`P_F(B)/P_F(A) = Z_B/Z_A`, the constraint §2.1 shows the two-group objective
omits) without the collateral in §3.2. `MECHANISM`.

**Why the separate form is necessary, not merely tidier.** The three residual
scales are `sigma_b ~ 1.5-2`, `sigma_f ~ 28-30`, `Delta ~ 64`. One knee cannot
serve them: keeping 95% of backward rows graded needs `beta >~ 50`, which takes
the forward branch from 61% clipped to ~22% — near-L2 mean-matching on the branch
whose documented failure mode is Huber-basin escape. The bracket is empty at
`lambda = 0.5`, and not by accident: `lambda*Delta = 31.5` and `sigma_f = 30` are
the same number here.

**Design choices inside `Level`, with the trade named.**

| choice | recommendation | why |
|---|---|---|
| source of `J_F`, `J_B` | **in-batch, LEAVE-ONE-OUT — no tracker, no half-life** | The term is *linear* in `gap`, so an unbiased `gap` gives an unbiased force and the sign being wrong 17-41% of the time near the setpoint is variance, not bias. The real hazard is that an in-batch `Delta_hat` shares rows with the `grad u_j` it multiplies, so `E[Delta_hat * grad u_j] != Delta * E[grad u_j]` — the covariance contributes a spurious within-variance gradient of order `1/n`. **Estimating `Delta_hat^(-j)` from the condition's other rows removes that term exactly**, keeps the coefficient on-policy and current, and retires the freshness-cadence question for this term entirely. Cost is `(sum - u_j)/(n-1)`. ⚠ It requires both branches to hold the same condition in the same batch, which today is true for only ~19% — so **aligned draws (T9) are a precondition for the tether, not just for pooling.** |
| functional form | **Huber in `(J_F - J_B)` with `beta_z` above the gap** | proportional inside the knee, bounded outside. A clamp on the *gain* (what `level_gap` does today) is sign-only bang-bang: it sets a rate, never an equilibrium. |
| sided-ness | **flag, default OFF** | `level_gap` deliberately omits the forward half ([:530](../gflownet_losses.py:530)), so `J_F` is a target and never a follower. Two-sided directs where the transferred mass comes from instead of leaving normalisation to decide it; against that, the forward half is an unlearning force on the policy's own samples and this run already fails by inflation. Owner is amenable, wants it optional. |
| `P_B` gradient | **flag, default OFF — do not gate it by default** | §2.6 says the `P_B` heads can close the gap through `B + E` with zero mass moved, but gating is the wrong response and an earlier draft of this table overstated it. Detaching the explicit `log_pb` term removes only a **zero-mean score** (variance reduction, no change of fixed point); the channel that actually carries the escape is the *reparameterised* path, and closing that needs `traj_grads: 0`, which degrades `P_B` to variance-with-no-drift. Owner (2026-08-27): a learned `P_B` is a large capacity term and should not be given up. **Detect instead of gate** — §3.5's `Mean B Var` read is quantitative and free. Keep the flag for a diagnostic arm. |
| bias correction | **second-order; do not design around it** | an explicit `(Jhat_F - Jhat_B)^2` on top of two full within-terms does double-count `V_f/n_f + V_b/n_b`, but priced at the operating point it is 15.4% contamination in value and a 25% effective forward-VarGrad surcharge at `n = (2,2)`, 8.3%/12.5% at (4,4). Tracker-fed means take it to ~1 nat^2. ⚠ Pooling avoids it not by having a better estimator but by removing the ability to choose a bad `rho` — and the whole identity is quadratic, so it does not hold at `beta = 10` anyway. |
| group sizes | independent per branch | already true (`repeats` fwd, `condition_block_m` bwd). Note the shelf is floor-limited, not noise-limited, so raising `m` is not where the value is. |

⚠ **Two similarly-named terms, and only one of them is this. The one with `z` in
its name is the one that cannot touch the policy.**

| key | branch | acts on | `log w` | what it is |
|---|---|---|---|---|
| `z_level` | forward only | the flow head **only** | detached ([:1004](../gflownet_losses.py:1004)) | `mean_c (log_Z(c) - mean_i u_i)^2`. A Z-fitting sidecar; zero sampler gradient. |
| `level_gap` | backward only | the **policy** | live | `clamp(delta, +-10) * u`, `delta` from the tracker. The only policy-side level force on this route. |

`level_gap` is the term this section is about. The delta from it to the
recommended form is: raise the clamp into a knee, add the forward half, and
stop-gradient the `P_B` channel. Roughly a dozen lines. (If the naming keeps
costing re-reads, `z_level -> z_fit` and `level_gap -> bwd_level_force` would say
what each does; that is an owner call, not a change to make in passing.)

### 3.2 Single pooled group — rejected **at `beta = 10`**, viable above `Delta/2`

Putting all sources in one VarGrad group is the same estimand, but the single knee
has a closed-form failure. A branch keeps its within-condition signal only while
its cluster's offset from the pooled centre stays inside the knee:

```
backward survives iff  lambda*Delta     <~ beta   ->  lambda <~ beta/Delta     = 0.156
forward  survives iff  (1-lambda)*Delta <~ beta   ->  lambda >~ 1 - beta/Delta = 0.844
```

**Disjoint at `beta = 10`: no composition preserves both.** Measured by regressing
each row's coefficient on its own within-branch deviation (1.0 = intact VarGrad,
0 = pure level push): pooled backward slope falls 0.994 -> **0.000** for any
`lambda >= 0.25`, forward from 0.363 to 0.15. The backward branch is the one
currently doing quadratic work, and pooling deletes it. `MECHANISM`.

The two windows overlap iff **`beta >= Delta/2 ~ 32`**. So pooling is not dead in
principle — it is dead at the shipped knee. The sequencing that follows is: raise
`beta` first as its own un-pooled arm (free on the backward side, whose residuals
are 1.5 nats), and only then pool.

Two further costs stand at any `beta`. Without aligned draws only **19% of rows**
land in a mixed group (fwd and bwd share ~212 of ~1393 conditions per step), so
81% of the batch sees no change and the arm is uninterpretable. And pooling
**welds five knobs into one**: `lambda` simultaneously sets the forward loss
weight, the backward loss weight, the bridge coefficient `rho = lambda(1-lambda)`,
the estimator bias `V_f/n_f`, and the compute split. Today they are decoupled —
`fused_train_step` draws a full batch on each branch and the fracs are pure loss
weights. It is also quantised: `lambda = n_f/(n_f+n_b)` lands on {1/3, 2/5, 1/2,
2/3}, not a dial, and pooling *today's* row counts would sit at 2/3. Finally it
would shift fwd `emp_z`'s target ~32 nats onto the mixture centre, the exact
construction the backward path asserts against.

**That weld is the argument for §3.1.** The three-term form is the pooled estimand
with the five knobs unwelded.

### 3.3 `level_gap: 1` at the shipped clamp — the free upper bound

Zero code. It delivers the *identical* per-row force pooling would (`dL/du = +10`
on backward rows) at four times the duty cycle, with both shape terms intact. So
it bounds §3.2 from above, and a null result there falsifies pooling without
writing anything. Note the whole neighbouring `qm9k_*` battery already runs
`level_gap: 1` at `clamp: 10` — i.e. that family measured bang-bang, never
proportional control.

### 3.4 Small `bwd mle` — the blunt baseline

The backward branch already supports it ([:603](../gflownet_losses.py:603)).
MLE-on-buffer is the unambiguous "raise the whole set" force with a gain set in
config rather than by a saturated clamp; the 2026-07 phase-2 postmortem already
prescribed 0.1-0.25 as the re-anchor. Hard to get wrong, and it does not
distinguish coverage from path mismatch.

### 3.5 Don't pin the variance — **detect** it, for free

The obvious counter-measure is to freeze the forward log-variance head and set
`learn_pb: false` so the only route left is the drift. **It does not close the
escape**, and it is not needed.

It does not close it because there are **seven** absorbing groups, not two: the
output layer carries **no bias** and the trunk is shared between the mean and
log-variance heads and with the backward policy, so freezing the log-variance
weight slice leaves `exp(logvar)` free to ride the drift gradient. It also omits
the DPLR correlation block (84 outputs per step, 1680 per trajectory, reshaping
the density at *fixed* marginals) and the 16-d conditioner (collapsing it widens
the effective per-condition marginal without touching any variance output).

It is not needed because **the escape is already instrumented and nobody reads
it**. From §2.6's budget, at `r^2 = 2` the sensitivity is `-57` nats per unit of
backward additive log-variance — so **`Mean B Var` rising by +1.1 accounts for the
entire 64-nat gap.** That series is already logged. And `stepkl_sum`, retired
2026-08-23 on the grounds that "nothing read the reductions", *is* the budget of
the degenerate route; it should come back.

So the recommended shape is: run the level arms with the variance channels free,
and put `Mean B Var`, `bwd Mean B Var`, `fwd/step_var` and the restored
`stepkl_sum` on the panel beside `zmatch/delta_mean`. If `Delta` falls and
`Mean B Var` rises ~1, the term bought path-matching and no coverage. Freezing
becomes worth doing only if that read is ambiguous.

⚠ Independent counter-argument for keeping variance free regardless:
`delta_overlap` only ever rises during inflation excursions, so widening may be a
necessary *intermediate* rather than an escape, and pinning may foreclose the only
path there is.

### 3.6 The branch where none of this is the answer

If the deficit is expressivity — a T=20 diffusion cannot represent the required
`P_F` — then every level term saturates and closes nothing, and the move is a
from-scratch run at larger T or width, not another term. `T`, depth and width
cannot be tested by warm start: the drift and variance heads are calibrated for
`dt = 1/T`. A related possibility is that this is an **initialisation** problem:
the warm start is an already-collapsed checkpoint, and phase-1 MLE exists to place
mass. Re-placing mass and *then* running VarGrad with a level term may beat asking
a collapsed policy to climb 62 nats by gradient descent.

---

## 4. Hypotheses

| # | claim | grade | falsifier |
|---|---|---|---|
| H1 | The offset is unpenalised; that is the primary defect | `MECHANISM` in u-space, `CONJECTURE` in θ-space | T1 shows strongly unequal NTK Gram row sums with a consistent sign |
| H2 | Much of `Delta` is `log P_B` path asymmetry, not missing terminal mass | `CONJECTURE` | T2's three-component split attributes it to `log R`/`log P_F` |
| H3 | The backward descent is bought by `P_B`, not `P_F` | `CONJECTURE` | `learn_pb: false` leaves `bwd/vg_lb`'s descent rate unchanged |
| H4 | The backward descent is memorisation of a frozen anchor set | `OBSERVED` conditions (prior churn skipped, `anchor_admitted_last_n` 0 throughout), mechanism untested | backward VG on held-out anchors descends at the same rate |
| H5 | The `logw_std_within` shelf is a collapse floor, not an expressivity floor | `CONJECTURE`, competes with the standing expressivity reading | a between-condition dispersion measure that is healthy at the shelf |
| H6 | The excursions are the level force acting through the only open channel (inflation), terminated by the divergence guard | `CONJECTURE` | `delta_overlap` flat through an excursion |

**Retired by measurement:** that the gap "closes slowly" (it is a random walk,
though `bwd/jensen_z` itself has a real trend); that `fwd/jensen_z` reports the
forward level (it reports the Jensen gap); that the sign of the
`vg_lb`/`jensen_z` co-movement is diagnostic (same-batch arithmetic, five
confounds, all pushing toward a reassuring reading); that a restricted `log Z_A`
fixed point exists; that `vg_detach_center` could serve as an absorption
switch. It is inert wherever the residuals are quadratic — `psi_bar = 0`
identically there, at **any** group size, which is stronger than the docstring's
`m = 2` oddness argument — and it only wakes up under saturation. ⚠ If §3.1 or a
pooled arm is built, keep it **off**: with realistic spread the pooled `psi_bar` is
−1.42 at `lambda = 1/2` (14% of `beta`, zero-crossing at 0.652), so detaching
injects an uncontrolled net level force on top of the one being designed.

---

## 5. Tests, in order

**T1 — `||grad_theta u||` per row vs `u` inside forward groups.** Instrumentation
only. Decides §2.1's θ-space step, which everything else rests on. If the two
correlate with a consistent sign, the level has been drifting under its own
gradient and several readings above need reinterpreting.

**T2 — mean `log R`, `log P_B`, `log P_F` per branch. SHIPPED 2026-08-28** as
`{fwd,bwd}/logr_mean`, `logpb_mean`, `logpf_mean` on both branches. The
decisive pair is `bwd/logpb_mean` rising while `bwd/logpf_mean` holds — that is
the P_B-widening route closing the gap with no terminal mass moved. Three
scalars. Makes
`Delta = KL(q*P_B || P_F)` exact and splits it into coverage versus path
mismatch. **Every proposal in §3 is premature until this runs** — if the gap is
mostly `log P_B`, they are all aimed at the wrong quantity.

**T3 — `level_gap` ladder from the step-20000 checkpoint, three arms, ~90 min
each.** `0.0` (baseline) / `1.0` at `clamp: 10` (saturated; the §3.2 upper bound)
/ `0.05` at `clamp: 200` (restored-proportional, comparable typical force but it
scales with the gap). Read `zmatch/delta_mean` slope first, `bwd/under_coverage`
second, `eval_test/*` against `eval_fwd/*` third.
*Kill:* `delta_mean` flat in **both** live arms — the level is not force-limited
and no bridge of any form closes it; go to §3.6.
*Kill differently:* `delta_mean` descends while `under_coverage` stays frozen and
T2 attributes the motion to `log P_B` — the degenerate solution; fix `learn_pb`
and stop-gradient the `P_B` channel.
*Proceed:* the proportional arm moves what the saturated arm cannot — knee-above-
the-gap is established and §3.1 is worth building.

**T4 — the discriminator that must be on the panel for any level arm, and it is
free.** `zmatch/delta_mean` must fall **while `fwd/logw_std_within` does not rise
and `Mean B Var` does not**. The second criterion is quantitative: from §2.6,
**+1.1 on `Mean B Var` accounts for the whole 64 nats**, so a level arm that closes
the gap while that series moves by ~1 bought `B + E`, not `M` — path matching, zero
coverage. Restore `stepkl_sum` in the same change; it is the budget of exactly that
route and was retired for being unread.

**T5 — overlap fraction `p`,** the row-level fraction of backward rows inside the
forward group's spread. Nothing logs it, it drifts silently from ~10% to ~85% as
the batch sizer grows the batch, and it sizes the residual coupling in §2.3.

**T8 — `beta` raised on both branches, un-pooled, as its own arm.** Free on the
backward side (residuals 1.5 nats, so 10 vs 80 changes nothing there) and it is
the single-variable version of the change §3.2 requires before pooling is even
coherent. Read `fwd/vg_lb` against `fwd/logw_std_within`: if the loss falls while
the RMS rises, the fit family is the wrong ruler. Precedent for `beta: 80` exists
in this config's `unconditional_tb` equilibration stage.

**T9 — aligned draws, if §3.1 or a pooled arm is built.** Draw one condition set
per step and have both branches serve it. Priced at identical compute:
`C = 800, n_f = 4, n_b = 2` gives 4800 rows, today's exact budget and energy cost,
800 groups of 6, 100% mixed — against today's 1982 groups of mean 2.4 at 19%
mixed. Or `n_f = n_b = 2`: 3200 rows, groups of 4, and it **cuts step time ~34%**.
Per-step condition coverage falls 1393 -> 800 of 5265, which costs 132 steps
rather than 76 to warm the tracker past `min_visits: 20`. ⚠ Backward rows are
**~1.9x cheaper, not free** — energy is only 34.7% of the step and both branches
run a full T=20 trajectory through both heads.
**Ship `vg_mixed_frac` beside `vg_live_frac` in the same change** — the fraction of
rows whose group holds both sources. Without it an alignment that quietly degrades
back to 19% reads as a perfectly healthy pooled run. This repo has shipped that
class of silent no-op more than once.

**T6 — a frozen reference set.** No expectation under `P_F` can detect `P_F`'s own
support shrinking. Snapshot a set `S` at stage entry and score
`mean_S(u) - [mean_S(log R) + log N]` plus the p95 of `u` on `S` each eval.
`refresh_anchor_buffer_surprise` is 90% of this but is fed from the forward eval
batch, so it inherits exactly the blindness being tested for.

**T7 — re-read the existing `beta` ladder for the level channel.** Free. The
2026-08-26 battery ran fwd `beta` 10/40/200 and read only level-free channels;
§2.4 says the tail is where the level information is, so re-read
`under_coverage`, `zmatch/delta_mean` and `delta_overlap` on the stored history.

**Not worth running:** anything whose signature lands in the
`logw_std_within`/`over_coverage`/`scatter_err` family at one seed; further
grouping or noise-reduction knobs (the shelf is floor-limited); `bwd beta` (its
residuals are already inside the smaller knee).

---

## 6. What is built and running (2026-08-28)

- **`level_gap_pf_only`** (new `bwd_loss_coeffs` key, **default on**). The tether
  multiplies `-log_pf` alone instead of the whole log-weight. Verified by autograd
  on the shipped expression: the forward force is **bit-identical** to the old form
  while the explicit `P_B` coefficient goes to exactly zero. So it is a pure
  variance reduction, not a redirection — the `P_B` escape rides the
  reparameterised path and survives this change.
- **T2 component split** on both branches, as above.
### Results, three arms, 2026-08-28

All from the shared `_step20000f50` warm start, `level_gap: 0.05` at
`level_gap_clamp: 200`, read at step ~20160. `OBSERVED`, one seed, ~200 steps.

| channel | entry | arm 1 tether | arm 3 +path_grad |
|---|---|---|---|
| `bwd/level_gap_coeff_rms` | 65.2 | 69.5 | 69.8 |
| `bwd/jensen_z` | 76.2 | **70.4** | **70.3** |
| `zmatch/delta_mean` | 64.4 | 72.9 | 73.6 |
| `fwd/logr_mean` | -2.0 | **-34.9** | **-38.1** |
| `fwd/jensen_z` | 11.7 | -33.9 | -37.3 |
| `fwd/vg_lb` | 63.8 | 325 | 340 |
| `fwd/over_coverage` | 25.7 | 80.5 | 89.1 |
| `bwd/logpf_mean` | 283 | 300.7 | 301.4 |
| `fwd/logpf_mean` | 280.5 | 299.3 | 300.4 |
| `bwd/vg_lb` | 1.94 | 1.58 | 1.62 |

**1. The proportional knee works, and it is what unlocked the level.**
`level_gap_coeff_rms` reads 65-70, the true gap, instead of railing at the clamp.
`bwd/jensen_z` falls 6 nats in 200 steps, against zero net motion over 8600 steps
with the tether off. The mechanism in §6 is real and the clamp was the blocker.

**2. But the lift is GLOBAL, not selective, and that is the new result.**
`bwd/logpf_mean` +17.7 and `fwd/logpf_mean` +18.8 move in lockstep. The tether is
backward-only, yet it raises `log P_F` by the same amount on the forward branch's
own samples. Through a shared trunk the policy cannot express "raise the buffer
only"; it raises the density everywhere. So `Delta` WIDENS (64 -> 73) even while
`J_B` comes down, and `fwd/logr_mean` falls 35 nats — the reward pays for the
lift. ⚠ The parametric-coupling objection was graded weak in the 2026-08-27
adversarial pass ("~10 nats of level on the buffer costs ~0.4 sigma of drift
perturbation"). On this evidence it dominates. **This is the finding that should
drive the next design.**

**3. `path_grad_last_k: 1` alone is inert.** Arm 3 matches arm 1 to within noise
on every channel (`bwd/jensen_z` 70.34 vs 70.35). Expected in hindsight: with
`reward_grads: 0` the reward is still computed under `no_grad`, so `grad log R = 0`
whatever the path does.

**4. The reward gradient is NON-FINITE, and `reward_grad_clip` cannot fix it.**
Arm 2 (`reward_grads: 1.0`, `reward_grad_clip: 10.0`) aborted at step 20049 on
**50 consecutive non-finite gradients, from the first step**. The hook is
`g.clamp(-c, c)`, which bounds a large gradient and passes a NaN straight
through; for ELJ a clashed sample gives an infinite energy and a NaN derivative.
So `local_aug02/make.py`'s two-cause diagnosis is incomplete — there is a third
failure mode ahead of both, and it is the one that fires. A fix has to sanitise
the reward gradient at the source (`nan_to_num` before the clamp) AND report the
sanitised fraction, or the drop rate becomes an invisible silent filter.

**5. The P_F-only form needs its own gain calibration and has none.** Launched
first at `coeff 0.05`, it detonated in 110 steps (`fwd/vg_lb` 53 -> 343,
`zmatch/delta_mean` 64 -> 86). Cause: `u` carries `grad log P_B - grad log P_F`,
two densities of the SAME path whose gradients largely cancel; dropping `log P_B`
removes that cancellation, so the same coefficient delivers a far larger force.
An autograd check with independent parameters cannot show this. Re-price before
re-running.

- **`configs/qm9c_lgp.yaml`** — arm 1, RUN (see above). `level_gap: 0.05` at
  `level_gap_clamp: 200` (proportional: knee above the 64-nat gap), everything else
  identical to `qm9c_t20_vghalf`. 4001 steps from the shared `_step20000f50`
  warm start. First read: `level_gap_coeff_rms` must sit near the true gap rather
  than railed at the clamp.
- **`configs/qm9c_lgp_fpg.yaml`** — arm 2, RUN, aborted on non-finite gradients. Arm 1 plus
  `path_grad_last_k: 1`, `reward_grads: 1.0`, `reward_grad_clip: 10.0` on the
  **global** `fwd_loss_coeffs` (stage overrides reject those keys).
  ⚠ `reward_grads` is a **separate key from `traj_grads`** — the forward reward is
  computed under `no_grad` on its own switch, so `path_grad_last_k` alone does not
  restore the reward gradient. Both are needed.
  Prior art: `configs/local_aug02/make.py` already diagnosed the historical
  "very destabilizing" result as two independent causes — the BPTT Jacobian over
  all T (fixed by truncation) and near-singular LJ reward gradients on clashes
  (fixed by source clipping) — and designed `fpg_k1_rg1_clip10` / `_noclip` /
  `_rg0` to separate them. **Those arms were never run**; no results appear in
  `findings.md` or `change_history.md`.

### The gain ladder: the tether is closed, negatively (2026-08-28)

Five runs from the shared `_step20000f50` warm start, `level_gap` swept over three
orders of magnitude against a matched `level_gap: 0` control. `OBSERVED`, one seed
per rung.

| gain | `delta_mean` slope /1k | differential slope /1k (settled) | verdict |
|---|---|---|---|
| 0 (control, 8600 steps) | ~0, random walk | n/a | gap static |
| 0.005 (3750 steps) | +22.9 | **-0.39** | gap grows |
| 0.05 (1100 steps) | +15.6 | **-0.13** | gap grows |
| 1.0 (400 steps) | +hundreds | +3.7% of a huge lift | runaway |

**Every nonzero gain makes the gap GROW; the minimum of |delta slope| is at zero.**
There is no window between "does nothing" and "destructive" -- the useful and the
destructive regimes do not overlap, because the term has no component along the
direction that would help.

**Mechanism, one line:** the tether raises `log P_F` fastest exactly where the
policy already samples, so it drags `J_F` down faster than `J_B` and the gap
widens. The force is ANTI-correlated with what is needed, not merely uncorrelated.

**The settled differential is zero at both usable gains** (-0.13, -0.39 nats/1k),
so the selective component is absent, not small. At that rate 62 nats takes
millions of steps.

⚠ **Reading lesson, and it cost an hour.** The differential slope is NOT readable
early: at 100 steps rung 1 read -8.4 and rung 2 read +22.7, and both converged to
~0. The derived "decision channel" was worse than useless at short horizons -- it
would have sent the ladder the wrong way. What WAS readable at ~10 steps, and
correct in all five runs, is the raw qualitative fact: **`bwd/logpf_mean` and
`fwd/logpf_mean` both climbing hundreds of nats in parallel.** Judge a level term
on that, not on a slope.

### ⚠ RETRACTION: every result above 2026-08-28 14:00 used a BIASED start

`var_conditioning` on this route begins at **step 16760**
(`..._stage_start.pt`). Every arm in this document before the pooled ladder --
the four `level_gap` gains, the `qm9c_t20_*` family, and the "8600 steps with no
level motion" control -- resumed from `_step20000*.pt`, which is **3240 steps
INTO the stage under test**. They measured the stage's late behaviour and
attributed it to the stage.

**The headline claim above is wrong as stated.** The level gap is not static:

| | step 16760 (phase-1 exit) | step 20000 | over 8600 further steps |
|---|---|---|---|
| `zmatch/delta_mean` | **79.1** | **64.4** | 64.4, random walk |
| `fwd/logw_std_within` | 47.4 | 17.9 | 28-31 |
| `bwd/logw_std_within` | 10.1 | 2.0 | 1.5-2.0 |
| `fwd/jensen_z` | -17.4 | 13.1 | ~11 |

**The stage closes ~15 nats of gap in its first 3240 steps and then stalls.**
Everything this document says about the level being unreachable was measured
after the stall. The structural results (zero-sum coefficients, the M/B/D/E
decomposition, reachability) are unaffected -- they do not depend on the
operating point -- but every NUMBER quoted as "the operating point" is the wrong
one, and the beta windows derived from `Delta = 63` should be re-derived at 79.

Corollary that matters for design: a baseline from the phase-1 exit is NOT flat,
so any arm started there needs a matched control before its improvement can be
attributed. `configs/qm9c_p1_ctl.yaml` is that control.

### The pooled term, built and under test (supersedes T7's rejection)

T7 rejected pooling on the grounds that at `beta = 10` and `Delta = 64` the
pooled residuals all saturate, deleting both shape terms. That reasoning holds
for a shared branch knee and a 50:50 mix. The form now running is different in
three ways and the rejection does not carry over:

- the pooled term has **its own knee** (`pooled_beta`, 40) so it does not inherit
  a branch's;
- the buffer share is **subsampled to a configured `pooled_ratio`** rather than
  falling out of the batch sizes;
- the backward draw is **aligned** to the forward batch's conditions, taking the
  mixed-group fraction from a measured 0.19 to 0.67-1.0. Unaligned, ~81% of the
  term was a silent duplicate of each branch's own VarGrad.

Implementation: `gflownet_losses.pooled_condition_vargrad`, consumed in
`train.fused_train_step` outside the frac-weighted branch sum. Verified on the
shipped function: per-group coefficient sum 2e-8, correct signs (descent raises
`log P_F` on buffer rows, lowers it on forward rows), ratio control exact. The
measured coherent drift is **+1.61 fwd / -1.61 bwd against 0.000 from both branch
VarGrad terms** -- it is the only term in the objective with a net level force.

⚠ Running it POOLED-ONLY (both branch `vg_lb` at 0) is the owner's call and is
the clean form: the pooled term already contains `(1-lam)V_f + lam V_b`, so the
branch terms double-count. The cost is that `lam` then sets the two within-weights
AND the bridge gain simultaneously, and the branch `vg_lb` metric keys vanish
when their coefficients hit 0 -- hence `pooled/{within_f,within_b,bridge,
bridge_frac}`, which decompose the term using each group's LOCAL lambda.

## 6. Reading corrections that change what the numbers mean

- The `qm9c_t20_vghalf` arm halves fwd `vg_lb` and `emp_z` to 0.5 with bwd at 1.0.
  Terms **sum**, so the effective policy-loss allocation is **1 fwd : 2 bwd**, not
  50:50. It is a balance arm wearing a coefficient arm's label.
- The 50:50 is a **pin**: `max_fracs` and `default_boost` are both 0.5 on both
  branches, so the proportional controller has no headroom and its asymmetric
  targets cannot express themselves.
- Prior churn is **skipped** (no `prior_model`), so the buffer is 100% frozen
  anchors and `anchor_admitted_last_n` reads 0 at every report. The branch that is
  improving is fitting a set that never changes.
- `mk_dev.yaml:863-874` documents `level_gap: 0` blowing up in exactly the way the
  analysed runs blew up; the key below it now reads `0.0`. Either the comment is
  stale or the value is a mistake — both cannot stand.
