# Module: losses (`gflownet_losses.py`)

> **Status: MODULE SNAPSHOT.** The verification dates below are historical.
> Use this document for explanation and navigation; verify material claims
> against current code, canonical config, and focused tests. See
> [`EPISTEMIC_PROTOCOL.md`](EPISTEMIC_PROTOCOL.md).

Pass 1 (audit + rationalize). Verified against the working tree, 2026-08-03;
**revised 2026-08-08** for the importance-weighted backward reduction (L10) and
the truncated forward path gradient (L11), both added 2026-08-07.
**Unconditional route only** (`configs/mk_dev.yaml`, mode `unconditional_molecule`:
every `*_conditioning` flag false, so `gfn.conditional == False` and
`condition_library_size == 1`).

---

## 1. What it is

Two entry points — `get_gfn_forward_loss` and `get_gfn_backward_loss` — that
assemble a per-row loss from a menu of GFlowNet objectives (TB, SubTB, DB,
VarGrad, terminal MLE, TBC, and several Z-regression sidecars). Each is switched
on by a coefficient in the stage's `*_loss_coeffs` block. Both share one
skeleton: sample or replay a trajectory, compute `log_pf` / `log_pb` / `log_r` /
`log_Z`, apply the `freeze_policy` / `freeze_z` detach contract at the source,
accumulate active terms, soft-clip, and average.

Paper voice: *a single trajectory-balance objective is trained on three sampling
distributions — on-policy forward rollouts, backward rollouts from a prior
buffer, and prioritized replay of stored forward trajectories — with the flow
(partition-function) head and the policy each receiving gradient from exactly
one of them.*

## 2. Contract

**Inputs** — `loss_coeffs` (a namespace rebuilt per stage), a trajectory source,
`log_reward_fn`, optionally a `ConditionLogZTracker` and `condition_id`.
**Outputs** — a scalar loss, plus a `loss_dict` of detached diagnostics when
`report_losses`.
**Invariants**

- `freeze_policy` / `freeze_z` detach **once, at the source** ([lines 209–216](gflownet_losses.py:209)),
  so the freeze holds no matter which downstream terms are active. This replaced
  a per-term `tb_z` flag and is the single most important structural property in
  the file — it makes branch roles composable.
- `log_flow[:, -1] = log_r` — terminal flow is the reward by definition.
- The Z-residual monitor is fed **only** from the forward call site: Z is
  trained and judged on-policy; `bwd`/`replay` are off-policy and it is the
  policy's job to fix those.
- Backward VarGrad's group centre may never feed a Z regression (asserted at
  [line 455](gflownet_losses.py:455)) — the off-policy mean of `log w` is a
  biased log Z.
- **The final reduction is a plain mean unless the caller supplies
  `sample_weights`** ([line 567](gflownet_losses.py:567)), in which case it is
  self-normalised importance-weighted. Both assemblers otherwise reduce
  identically. See L10.

## 3. What is actually live

*(§3 revised 2026-08-08: the backward assembler grew an optional weighted
reduction, and the forward rollout an optional truncated path gradient. Both are
inert by default — see L10, L11.)*

This is the headline. The menu has **12 distinct terms** across the two
assemblers (`tb`, `vg_lb`, `vg_lme`, `emp_z`, `emp_z_persistent`, `z_level`,
`db`, `subtb`, `level_gap`, `pf_boost`, `mle`, `tbc`). **The unconditional route
uses three** — `mle` and `tbc` in phase 1, `tb` in phase 2 — with `tb`
instantiated on three branches.

| Stage | Branch | Live terms | Trains |
|---|---|---|---|
| `train_prior` | bwd (dataset) | `mle` 1.0, `tbc` 1.0, `repeats` 2 | policy (`freeze_z: 1`) |
| `naive` | fwd | `tb` 1.0, `freeze_policy: 1` | **Z only** |
| `naive` | bwd (prior buffer) | `tb` 1.0, `freeze_z: 1` | **policy only** |
| `naive` | replay (stored fwd trajs) | `tb` 1.0, `freeze_z: 1` | **policy only** |

Zero on this route: `vg_lb`, `vg_lme`, `vg_by_condition` (fwd), `db`, `subtb`,
`emp_z`, `emp_z_persistent`, `z_level`, `pf_boost`, `level_gap`. They are live
only on conditional presets (`configs/mode_presets.yaml`).

`level_gap` in particular has **never run alongside a TB backward branch**. Added
2026-07-29 (`0bc198d`); the sole nonzero setting in the tree is the
`var_conditioning` stage of the archived 07-29 preset, whose bwd branch is
`vg_lb`. That is structural rather than accidental — VarGrad's mean-subtraction
annihilates Z-level information, so `level_gap` is the only absorption channel
available there, and TB has no equivalent need (L9). Any recollection that it was
tried and lost on TB is a misfiling.

So the phase-2 objective is **TB three times**, differing only in freeze flags
and sampler. That is the paper's core claim and it is much simpler than the file
suggests. The stated rationale is sound and worth quoting nearly verbatim:
on-policy TB is mode-seeking because it only ever sees the policy's own support,
so the policy term is taken off the on-policy branch entirely; a
`|resid|`-prioritized replay of those same rollouts upweights both residual
tails instead.

## 4. Findings

**L1 — the combined loss is the MEAN over active terms, not the sum.** *(confirmed)*

[Line 319](gflownet_losses.py:319): `torch.stack(losses).mean(dim=0)` where
`losses` is `[n_terms, N]`. The effective weight of term *i* is therefore
`coeff_i / n_active_terms`, and **turning on a term silently dilutes every other
one**.

Live consequence: `train_prior` runs `mle: 1.0` and `tbc: 1.0`, so each acts at
0.5. It is not a bug — normalising by term count keeps total gradient magnitude
stable across configs with different menus, which matters a lot given how sharp
the LR ceiling is (see `module_lr_controller.md`). But it must be stated,
because it means **loss coefficients are relative weights, not absolute ones**,
and comparing a coefficient across two configs with different active-term counts
is invalid. Worth an explicit note in any config that sets them.

**L2 — `vg_lme`'s unconditional branch normalises by the wrong count.** *(confirmed, latent)*

[Line 955](gflownet_losses.py:955):

```python
log_Z = torch.logsumexp(log_ratio, dim=0, keepdim=True) - math.log(repeats)
```

The group here is the **whole batch** (N rows), so log-mean-exp requires
`- log(N)`. Subtracting `log(repeats)` leaves the estimate high by
`log(N / repeats) = log(B)` nats — roughly +5.7 at a 300-row batch with
`repeats: 1`. The conditional branch immediately above is correct (its group
*is* `repeats`), which is the signature of a copy-paste.

Because `log_Z` here carries gradient and the shift is constant, the residual in
`vg_loss` is offset uniformly: the term pushes every `log_ratio` up, i.e. pushes
`log_pf` **down everywhere**, and since P_F is normalised the only way to comply
is to move mass off-support. That is precisely the uphill mechanism
[`z_level_loss`'s own docstring](gflownet_losses.py:823) warns about.

`vg_lb`'s unconditional branch (a plain mean) is correct — the defect is
specific to `vg_lme`.

**✅ FIXED 2026-08-03** — normaliser changed to `math.log(log_ratio.shape[0])`,
with the reasoning recorded inline. It was latent (`vg_lme: 0` everywhere) but
the two are documented as interchangeable, so a config that switched would have
got a silently biased centre.

**L3 — two unbound-variable landmines.** *(confirmed)*

- Forward [line 271](gflownet_losses.py:271): `emp_z > 0` with `vg_lb == 0`,
  `vg_lme == 0`, and `vg_by_condition` false → `log_Z` was never bound → `NameError`.
- Backward [line 521](gflownet_losses.py:521): same shape → `log_Z_emp` unbound.

The forward condition-grouped branch documents the constraint ("emp_z here does
NOT require a VG loss to be active, unlike the repeats-grouped branch below"), so
it is known — just unenforced. It surfaces as a mid-run crash rather than a
config-load error. Both are one-line guards.

**L4 — the finiteness assert was forward-only. ✅ REMOVED 2026-08-03.**

`assert combined_losses.isfinite().all()` had no backward counterpart, so a
non-finite **forward** loss killed the process while a non-finite **backward**
loss was contained.

*Analysis corrected during the fix*: the assert was not the protection.
[`step_loss`](train.py:2560) already clips and checks the **gradient** norm,
returns without stepping the optimizer on a non-finite reading, and feeds a
consecutive-streak counter to `_frozen_training_state` — and that guard covers
both branches. So weights were never poisoned by a NaN loss; the assert was a
redundant, less graceful duplicate that crashed instead of letting the working
path handle it. Removing it makes both branches behave identically and routes
forward NaNs to the containment built for them.

**L5 — dead code. ✅ REMOVED 2026-08-03.** `normed_smoothness_loss` and
`soft_saturate` had zero call sites anywhere in the tree.

**L6 — `loss_clip: 1.0E+9` is inert as configured.** *(agreed; config is user-owned)*
`soft_clip` engages only above the cutoff, so at 1e9 it never fires; but the
value is `!= -1`, so the clip path is still taken and a `stack`+`soft_clip` is
computed every step for nothing. Set `-1` (the documented off switch) or a real
cutoff. **Left for the user** — `mk_dev.yaml` is user-owned.

**L7 — TB-with-persistent-Z can leave the flow head with no trainer.**
*(by design; ✅ WARNING ADDED 2026-08-03)*

`get_tb_loss` substitutes a **detached** target where `target_mask` is true, so
those rows give the flow model no TB gradient; `emp_z_persistent` is the intended
sidecar. A config with `tb_z_source: persistent`, `emp_z_persistent: 0`, and no
other Z-bearing term trains Z with nothing.

User's call: this is a config-level mistake, not a code path — so it now
*reports* rather than raises. [`_warn_if_z_untrained`](train.py:504) runs from
`set_loss_coeffs` (i.e. at every stage transition) and prints when no mode can
move `log_Z`, accounting for `freeze_z`, the persistent-target substitution, and
the `emp_z` / `emp_z_persistent` / `z_level` / `db` / `subtb` sidecars.

**L8 — `beta` is not an outlier guard; on the backward branch it is a semantic
switch.** *(derived; (b) measured)* — 2026-08-05

`beta * smooth_l1(r, 0, beta=beta)` gives gradient `r` inside the knee and
`beta*sign(r)` outside. Four consequences, which together answer §8.1.

**(a) Saturated backward TB is exactly MLE, scaled by `beta`.** If every bwd
residual sits beyond the knee on the same side — `max_i r_i < -beta`, i.e. the
buffer is under-weighted by more than its spread, the common case — the
coefficient is the constant `-beta` for every row, and reward, `log_Z` and
`log_pb` all drop out. What remains is `beta ×` the `bound`-estimator gradient
([line 659](gflownet_losses.py:659)) that `terminal_mle` produces. TB has
collapsed to likelihood.

Consequence worth stating, since it combines with L1: phase 1 runs `mle` at
effective weight 0.5 (two terms, 100%-bwd batch), while a saturated absorption
stage runs `tb` at `beta × bwd_frac` ≈ 3–5. **The absorption drive is already
~10× the phase-1 MLE that converges trivially**, so slow buffer absorption is
not a gain deficit and more gain is the wrong first instrument.

**(b) The knee is aggressive, not mild.** Measured on this problem `logw_std ~21`
against `beta: 10` (see the winsorisation note in §6) — the knee sits at roughly
**half a standard deviation** of the residual distribution. A fixed absolute knee
therefore flattens the whole under-covered tail to one drive level: a row 80 nats
out and one 10 nats out receive identical gradient.

Conceptually `beta` is an *outlier-robustness* parameter. That is sound on the
forward branch, where a wild on-policy residual is often a reward spike. On the
backward branch it inverts: those trajectories come from a curated buffer, so a
huge residual is not noise but a known-good region that is badly uncovered — the
most valuable gradient in the batch. This argues for a **branch-asymmetric knee**,
not a different global value. `beta` is already read per-branch from
`loss_coeffs` ([line 209](gflownet_losses.py:209),
[line 420](gflownet_losses.py:420)).

**(c) Per-row clipping leaks drive on high-variance terminals.** `clip(r) =
max(r, -beta)` is convex in `r`, so `E[clip(r)] >= clip(E[r])` — both negative,
so clipping each rollout then averaging gives a *weaker* drive than clipping the
averaged residual, with the shortfall growing in the intra-terminal `log w`
spread. At fixed K, averaging **before** the clip recovers it. Natural form: a
terminal-level TB residual built on the IWAE estimate already in
`log_pf_estimate`, Huber'd once — which also makes a clipped row mean "this
terminal is genuinely far off" rather than "this rollout was unlucky", and
composes cleanly with `tbc`, which already owns the intra-terminal spread.

**(d) What cannot replace it.** The design object is the influence function
`psi(r) = dLoss/dr`:

| | `psi(r)` | priority | outlier |
|---|---|---|---|
| quadratic | `r` | perfect | hijacks direction |
| Huber | `clip(r, ±beta)` | destroyed past knee | bounded |
| `soft_clip` | `→ ~1/r` | **inverted** | bounded |

`clip_grad_norm_` is **not** a substitute — it is a uniform scalar on the
*summed* gradient, so one extreme row still owns the direction and the clip
merely makes the hijacked step smaller. Bounded influence can only be imposed
per-row, pre-aggregation. `soft_clip` is worse than Huber here: applied to loss
*values*, `dL_clipped/dL = 1/(1+L-cutoff)`, so it is **redescending** — the
deepest, most-uncovered rows get the *least* drive. The wanted shape is
monotone-nondecreasing and bounded, which *is* Huber. The form is right; the
fixed absolute value is the defect.

**L9 — a constant offset on `log Z` is exactly an MLE-on-buffer weight.**
*(derived)* — 2026-08-05

Offset `log_Z` by `c` and expand: `(r+c)^2 = r^2 + 2c·r + c^2`. The only
θ-dependent piece of the cross term is `2c·log_pf`, so the whole effect is
`2c · E_batch[∇ log P_F]`. The **forward branch contributes nothing** — on-policy
score-function samples give `E_{P_F}[∇ log P_F] = ∇∫P_F = 0`, which holds while
`traj_grads: 0` (the live setting; with fwd path-grad it would not). The
**backward branch is the entire effect**, and `E_buffer[∇ log P_F]` is the
MLE-on-buffer direction.

So a Z offset of `c` **is** an MLE-on-buffer term of weight `-2c`. Pushing
`log Z` *up* is negative-weight MLE — it unlearns the buffer; the absorbing sign
is down. A variance argument does not rescue the up direction either: the
injected noise has covariance `c^2·F` (Fisher), which is symmetric in the sign,
so `c<0` gets identical variance *plus* the absorption term.

Practical reading: `log Z` is a coverage meter (an ELBO), not a mass reservoir —
P_F is normalised per step, so there is no mass budget for Z to loosen. Its
naturally fitted value is already the absorption-maximising one. `level_gap` is
the sign-correct, self-retiring form of the same force (`gap == -2c`), and unlike
a constant offset — which relocates the TB fixed point to `r = -c` — it vanishes
when the levels match.

**L10 — the backward assembler takes optional self-normalised importance
weights, and they must not be combined with an active Huber knee.**
*(added 2026-08-07; inert unless `sample_weights` is passed)*

[Line 567](gflownet_losses.py:567). `sample_weights` is applied at the **final
reduction**, so it covers every active term at once, and it is **self-normalised**,
so turning prioritisation on does not change the overall loss scale — and
therefore does not move the LR the run is tuned at. That placement is the whole
design: weighting individual terms would make the effective weight depend on how
many terms are active (L1), and un-normalised weighting would silently rescale the
gradient.

Its one consumer is the prioritised replay draw (`module_buffers.md` B7), which
supplies `w = (1/n_elig)/p` so the estimator is unbiased for the uniform-buffer
average at every κ.

Two mechanical points worth keeping:

- **K repeats are tiled, and mis-pairing is asserted rather than broadcast.** If
  `w.numel()` does not match the loss count, the code derives the tile factor and
  **asserts** it divides exactly, then `repeat_interleave`s. Silent
  mis-pairing here would attach each row's weight to a different row's loss.
- **⚠ It does not compose with Huber.** Per `to_do_rebuild.md` §B5b, an active
  Huber knee makes per-row push `~ β/δ`, so under `w ∝ 1/δ₊^κ` the **deepest rows
  push least** — the two mechanisms fight, and the composition is worse than
  either alone. **Set `beta` inactive on any prioritised branch.** This is a
  config-level constraint that nothing enforces; it is recorded at the call site
  and here, and it is what rb0808 arm 13 `beta_bq` varies.

**L11 — the forward path gradient is now truncatable, not all-or-nothing.**
*(added 2026-08-07; `path_grad_last_k: 0` reproduces the old behaviour bitwise)*

`get_traj_fwd(..., path_grad_last_k=k)` keeps the reparameterised state gradient
alive for the **last k steps only**, detaching everything before. The motivation is
that full-T forward path gradient is BPTT through T reparameterised SDE steps — a
product of T Jacobians — and is measured to be badly destabilising (the standing
`traj_grads: fwd 0` setting). Truncation bounds that product while keeping the part
that carries the reward signal, since `x_T` and hence `log R` are set by the final
steps.

The subtlety that makes it work: detaching the first `T−k` steps leaves
`current_state` a non-grad leaf at step `T−k`, but `next_state = f(current_state,
θ)` still carries gradient through the drift and variance heads — so the last `k`
steps' **policy params** do get a genuine pathwise gradient.
`initial_state.requires_grad_` is therefore still keyed on `detach_traj` alone and
correctly does not change.

**Paired with it, and separate:** `reward_grad_clip` per-sample-clips the
**reward's own** gradient path via a tensor hook ([line 194](gflownet_losses.py:194)).
These address two *different* destabilisers — the BPTT Jacobian product versus a
near-singular `d log R/d x_T` at atom clash — and the global grad clip cannot
separate them, because by the time gradients are summed at the parameters the
reward path is indistinguishable from the density path. Both default to off; the
only configs exercising them are `local_aug02/fpg_*`. **Neither has been A/B'd on
the current HEAD.**

## 5. Warrants

| Choice | Warrant | Evidence |
|---|---|---|
| TB as the sole phase-2 objective | **derived** — the fixed point is the target distribution | standard |
| Policy trained off-policy only; Z on-policy only | **derived + measured** — on-policy TB is mode-seeking | design rationale in the stage comment; the whole `naive` protocol |
| `traj_grads`: fwd 0, bwd 1, replay 0 | **measured** | fwd path-grad destabilises, bwd path-grad essential |
| `mle` + `tbc` with `repeats: 2` in phase 1 | **weak / contested** — user 2026-08-03: "still kindof vibes. Var(w) is *maybe* better conditioned at the handoff with TBC, but it's not a slam dunk." Kept because it converges trivially and seems rational | the earlier TBC phase-1 note overstates this |
| `tbc` requires `repeats > 1` | **derived** — the residual is defined over K same-terminal rollouts | asserted at [line 713](gflownet_losses.py:713) |
| Huber (`smooth_l1`) over MSE | **derived** (2026-08-05) — bounded influence can only be imposed per-row, pre-aggregation; the monotone-bounded shape is right and the alternatives are worse | L8(d) |
| `beta: 10.0` as the value, shared across branches | **measured — too narrow, and wrong-signed on bwd** — sits at ~½ the residual SD; makes saturated bwd TB exactly MLE×beta; applies outlier robustness to the one branch whose outliers are the targets. Also carries the winsorisation cost (read `tb_resid_clipped`, not the raw residual) | L8(a)(b), §6 |
| `dreg: True` for the exact MLE estimator | **inherited, untested** — user: "haven't ever noticed this making a difference but also haven't tested in detail" | IWAE literature |
| Mean-over-terms combination | **arbitrary**, defensible post hoc | see L1 |
| `reward_grad_clip` default 0 | **derived** — LJ energy gradients are near-singular at clash, and the global clip cannot separate the reward path from the density path | docstring at [line 194](gflownet_losses.py:194) |
| IS weights at the **final** reduction, self-normalised | **derived** — covers every active term at once, and keeps the loss scale (hence the tuned LR) unchanged when prioritisation is switched on | L10 |
| IS weighting requires a **quadratic** branch loss | **derived** — under an active Huber knee per-row push is `~β/δ`, so the deepest rows would push least | `to_do_rebuild.md` §B5b; rb0808 arm 13 |
| Truncated rather than full forward path gradient | **derived + measured** — full-T BPTT is a product of T Jacobians and destabilises; `x_T` carries the reward signal, so the last k steps are the part worth keeping | L11 |

`beta: 10.0` is the one load-bearing constant with no calibration behind it. It
sets where every TB-family term switches from quadratic to linear, and the
`z_level_loss` docstring argues that the linear regime is exactly where Adam
turns a persistent sign-consistent gradient into a full step regardless of
magnitude. Worth a ladder — and L8 now says what a ladder should vary: the
backward branch alone, sized to the residual tail rather than to a round number
(`logw_std ~21` puts three SD at ~60, not 30).

## 6. Failure signatures

| Symptom | First metric | Cause |
|---|---|---|
| Z stuck high, policy inflating | `z_bias` histogram off-centre | Z target above the achievable level; see the uphill mechanism |
| tb_err plateaus, residual sign-consistent | `tb_resid_clipped` vs raw | Huber linear regime — magnitude information is being discarded |
| `NameError` mid-run on a fresh config | traceback names `log_Z` / `log_Z_emp` | L3 |
| Process dies with `AssertionError`, no rewind | traceback in the fwd loss | L4 |
| Phase-1 loss halves when a term is added | any branch loss | L1 dilution, not an improvement |

## 7. Simplification candidates

**S1 — ❌ REJECTED (user, 2026-08-03).** Proposal was to delete or relocate the
nine unreachable terms. **The menu is deliberate**: the other losses are kept so
new experiments don't require re-implementing them. Nine-of-twelve-unused is
therefore a *feature of the design*, not dead weight, and the right framing in
any writeup is "a term bank with a small live subset per route" rather than
"unreachable code." The L2/L3 defects are to be fixed in place, not deleted
around.

**S2 — collapse or functionalise the two assemblers.** *(user: prefers
functionalise, "not religious about it")* `get_gfn_forward_loss` and
`get_gfn_backward_loss` share the freeze contract, tracker update, term
accumulation, clip, and reporting; they differ in trajectory source and legal
terms. Extracting the shared sub-methods keeps two named entry points while
removing the duplication — and it makes S1's term-bank structure explicit.

**S3 — state the combination rule.** *(agreed)* Whichever of sum/mean you keep,
put it in the config comment where the coefficients are set, because it silently
rescales every one of them.

## 8. Open questions

1. **Method answered 2026-08-05 (L8); the value still needs a ladder.** The
   alternatives previously floated do *not* survive: log-compression
   (`soft_clip`) is redescending, and grad-norm clipping bounds step magnitude
   rather than per-row influence, so neither replaces the knee. The Huber
   *shape* is correct. What remains open is (i) the value, which is measurably
   too narrow at ~½ the residual SD, and (ii) whether the knee should be
   **branch-asymmetric** — the argument in L8(b) says the backward branch wants
   a far wider one, and the cost is paid in forward calibration as the
   equilibrium displacement grows. A batch-quantile knee would preserve
   within-tail ordering but is scale-adaptive, hence ratchet-shaped; price that
   against the calibration-floor creep before adopting it.
2. Is `soft_clip`-over-terms wanted at all now that only one term is active per
   branch in phase 2? It is a no-op there by construction.
3. **Open.** `repeats: 2` has not been tested recently — 2 works, higher cuts
   into batch size as expected, but its effect on MLE / TBC / VarGrad / TB has
   not been separated. Same for `dreg`.

---

*Warrant classes: **derived** · **measured** · **inherited** · **arbitrary** · **contested**.*
