# Scope: the training workflow as it runs today, and where the manuscript no longer describes it

Written 2026-09-07 on branch `rarer-rollouts-v0`. Part 1 is the workflow in point
form, including today's edits. Part 2 is the audited list of deltas from
`Crystal_Generator (7).pdf` (26 pp.): 61 entries, each verified against a paper page
and a code line by an independent pass (the full verified text with suggested
manuscript wording per entry is in the audit output; this file keeps the one-line
version of each).

---

## Part 1 — the workflow now

### Phase 1 — backward warm start (run once per system; the `pt100_*` runs)
- `train_prior` stage, `train_mode: bwd`, `bwd_sampling_mode: dataset` — terminals drawn
  from the fixed prior dataset, trajectories sampled backward under P_B.
- Trains P_F (and P_B's corrections) by **MLE only** (`bwd: {mle: 1.0, tbc: 0.0}`).
  **log Z receives no gradient in phase 1**: log w = log R + log p_B − log p_F is only
  accumulated, detached, into the tracker's `ema_logw`. log Z is set at equilibration
  entry — by `bootstrap_z` (mean log w over a forward eval batch) where that action is
  declared, and otherwise it opens at the checkpoint value and is snapped by the first
  `z_level_fill`. The paper's "warm start log Z with Z-only TB gradients" is not what runs.
- Exits on `gates/progress_done`: the per-latent-column 1-D Wasserstein distance to the
  prior (normalised by the prior-vs-prior floor), trailing medians of the median and
  worst columns under bars 5 / 10 — a distributional-fit gate, not the paper's ε_MLE.
  Writes `*_phase1_exit.pt` (weights, buffers, optimizer, stage state). Every production
  arm resumes from that file with `load_weights_only: false`.
- **Today:** `snapshot_prior` is removed from the exit actions under the anchor-only
  prior; no prior *model* is written or needed.

### Phase 2 — equilibration (one fused stage; the whole production run)
- **Every step, a fused policy step** = `w_bwd · L_bwd + w_replay · L_replay`:
  - bwd branch: full batch drawn from the **prior buffer**, backward trajectories under
    P_B, Huber-TB with β = 80, log Z frozen (`freeze_z`) → trains P_F and P_B only.
  - replay branch: full batch drawn from the **replay buffer** by a prioritised-IS draw
    (p ∝ |δ|, self-normalised weights that undo it), the stored trajectory re-scored
    under the current P_F/P_B, Huber-TB with β = 80, log Z frozen.
  - `fracs` are LOSS WEIGHTS; every branch runs a full batch every step.
- **Every N steps (`fwd_rollout_every`, today's edit), a forward rollout**:
  - `batch_size` trajectories sampled from P_F, energies evaluated (the only MLIP call in
    a training step), the whole batch admitted to the replay buffer (store-all).
  - log Z **snapped** to the batch's winsorized-Huber TB root (`z_level_fill`, β = 10,
    se-gated: the batch must resolve the gap it claims). Between rollouts Z is frozen.
  - **No gradient from the forward branch** (`fwd_active = False`): the rollout exists
    for the buffer and the Z pin. N = 1 recovers the previous design where fwd ran every
    step as a Z-only branch at weight 0.05.
- **Loss-weight controller (`balance.kind: gated_ramp`, today's edit):** one sensor,
  two motions, hard rails. Replay share ramps up 0.0017/tick while
  `bwd/under_coverage_rise150` ≤ 1 nat (the 150-step period-matched difference of bwd
  under-coverage — the forgetting sensor), and drops 0.043/tick while it exceeds 1 nat.
  Ticks every 10 steps. Rails: bwd ∈ [0.5, 0.9] on MLIP systems, [0.25, 0.9] on ELJ.
  fwd pinned at 0.
- **Learning rate:** `mode: fixed`, one `fixed_scale` per arm (= `burn_in_scale`, so a
  rewind can never land on a burn-in-rate checkpoint), seed LR 1.25e-4 × scale.
  Hard-failure bars (loss excursion 10× the burn-in root, absolute 1e6) rewind to the
  rolling checkpoint; `fire_cut_factor: 1.0` — a fire never moves the rate; a budget of
  `max(3, step/1000)` rewinds then abort. The hot-LR drawdown sensor reports only.
- **Z machinery:** `z_calibration` servo OFF (its rollout mode would call the MLIP on
  every skipped step); `z_level_fill` is the only thing that moves log Z.
- **Eval every `eval_period`:** eval rollouts (IS / log-mean-exp log Z estimators,
  energies, coverage), prior-buffer churn, one replay admission, figures.

### Buffers
- **Anchor buffer** — the thinned prior dataset (≤ 200k rows). **Frozen**: no admission,
  no thinning, no surprise refresh; draws are uniform (`replay_beta: 1.0`). This is the
  paper's "noised buffer" source: rows are isotropically noised (`noise_log_range`) and
  re-scored when drawn into the prior buffer.
- **Prior buffer** — bwd's sampling distribution. Seeded from the prior dataset;
  refilled by churn (`mean_lifetime` is a RATE divisor per eval call, not a lifetime)
  **from noised anchors** (`source: anchors`, today's edit), each admitted only if
  E < E_min(c) + 100 kJ/mol. Previously refilled by a learned prior *model*; that model
  is no longer built, loaded or resumed. Rows exit by the energy gate (E ≥ E_min + 100)
  or, on overflow, by low residual among often-drawn rows — no age term.
- **Replay buffer** — recent forward samples. Every forward batch admitted; random
  (hazard) eviction with `mean_residence_steps` **in steps** (today's clock fix — it was
  per manage call) and a hard age cap at 5τ; prioritised-IS draw as above; memorisation
  sensor `replay/resid_vs_intake` (current vs admission residual, derived bar 1/e)
  drives the freshness servo (churn/residence boost).
- Reuse per stored row = N (store-all); staleness = τ; fill event = N/τ of the buffer.

### Energy, reward, temperature
- Energy functions: ELJ (raw lattice sum × the dataset's `thermal_scaling_factor` →
  kJ/mol; mipcas 0.3636, nehzor 0.1556), UMA, MACE. The coefficient rides on the batch
  and every energy read is refused unless stamped (`assert_lj_coeff_stamped`).
- T = 2.5; reward = −E/T; **soft clip** armed at startup from the prior's own
  distribution: E > E_min + `reward_range`·T (250·T) is log-compressed, not truncated.
  (The clip's caller was added 2026-09-02; before that `reward_range` was inert.)
- E_tot = clip₁.₁(clip₁(E_phys + 10·E_ρ + PV) + 10·T·E_bound + 10·T·E_reduce) + E_Jacob.
  Bounding and reduction are pre-multiplied by T (temperature-independent in reward
  space); the Jacobian is never compressed.

### Model
- Latent: 12 dims rescaled to [−1, 1] (log lengths on [0.075, 3]/[0.1, 4], angles on
  [0.2π, 0.8π], aunit-fraction centroids on [0, 1], θ ∈ [0, π/2], φ ∈ [−π, π],
  r ∈ [0, 2π]); the cell is **projected** onto the crystal system at construction.
- Three-way partition at construction: **angular** (φ, r, and every centroid axis whose
  asymmetric unit spans the whole cell — P-1: b, c; P21/c: a, c), **linear**, and
  **dead** (angle rows the projection overwrites: none for P-1, α and γ for P21/c; plus
  free-translation axes gauge-fixed to the box centre in polar groups). Dead dims are
  pinned every step and excluded from every log-prob, so log Z is over live dims only:
  NEHZOR generates in 10 dims, MIPCAS in 12.
- P_F: SDE over T = 100 uniform steps, mean + tanh-gated log-variance per dim, plus a
  **diagonal-plus-low-rank covariance** (rank 6, correlated fraction ρ ≤ 0.5, marginal
  variance preserved; no low-rank part on angular or dead dims; Woodbury density).
- P_B: Brownian bridge to the source with learned bounded corrections
  (`pb_drift_range` 0.4, `pb_var_range` 6); on periodic dims the **exact reversal of the
  wrapped bridge** (mixture over 5 arrival windings, each image-summed over 7); all
  densities are functions of the wrapped trajectory, so stored trajectories replay exactly.
- Networks 4 × 512 residual MLPs, layer norm, 64-harmonic time encoding, no EMA; one
  fused Adam for the policies with the Z scalar as its own param group (lr 0.1).

---

## Part 2 — deltas from the manuscript

Legend — **NEW**: in code, absent from the paper. **CHANGED**: paper describes an older
version. **REMOVED**: paper describes it, code does not do it. **MISDESCRIBED**: code
unchanged since the paper, paper wrong about it. ⚠ marks items that are regressions or
silent drifts in the *method*, not just in the text.

### A. The deltas you listed — all confirmed, with the verified detail
1. **DPLR covariance — NEW** (gfn.py:584-614). Paper S1/S4: strictly diagonal forward
   kernel. Code: C = diag(d) + VVᵀ, rank 6, ρ ∈ [0, 0.5) redistributes the per-dim
   budget without changing the marginal; zero on wrapped and dead dims; P_B stays
   diagonal. Shipped in all 28 prod_sep02/rr_sep07 arms.
2. **Redundant-dimension removal — NEW** (dead_latent_rows.py:60-135; gfn.py:708-730).
   Paper: "twelve continuous dimensions", (−1,1)¹² everywhere. Code: projected angle
   rows + free-translation axes are dead, pinned, excluded from the log-probs and hence
   from log Z (P21/c = 10 dims). Consequence: p.23's "constraints applied via quadratic
   penalties" is now **MISDESCRIBED** — equality constraints are projections; only the
   inequality (Niggli / monoclinic β) constraints are penalties.
3. **Periodic-boundary scoring — CHANGED, and partly code-catching-up-to-paper**
   (gfn.py:1290-1381, 3bec208). P_F: the paper's S6 nearest-image kernel is now
   implemented verbatim (the paper-era code did not — `docs/periodic_scoring_fix.html`).
   P_B: the bridge drift is evaluated at the wrapped representative, and with
   `pb_exact_reversal` the angular backward kernel is the exact reversal of the wrapped
   reference bridge (a winding mixture), so S6's "single Gaussian, windings negligible"
   no longer describes P_B. Sampler and scorer agree by construction; replay is exact.
4. **Periodic asymmetric-unit centroid dims — NEW** (aunit_periodicity.py:53-67;
   gfn.py:335-347). Paper: only φ and r are periodic; centroids are walled. Code: any
   centroid axis with L_au = 1 is a circle (P-1: v, w; P21/c: u, w) — sin/cos embedding,
   wrapped kernels, no DPLR part. The (−1,1)¹² bounding wall is therefore inert on 4 of
   12 dims (it always was on φ, r).
5. **Backward-thermalisation phase — REMOVED** (protocol `prod_eq`: two stages). Paper:
   three phases, "smoothly transition" into backward-TB, Fig. 8. Code: `train_prior`
   (MLE, `tbc: 0`) → `equilibration` (fused); backward TB on the noised buffer runs
   concurrently with replay and rollouts from fused step 1. The transition is one hard
   boundary (fresh Adam, LR burn-in, prior buffer rebuilt through the admission gate).
6. **On-policy calibration split → forward = Z only, replay = TB on the policy —
   CHANGED + NEW.** Paper p.14: "Phase 3 mixes forward and backward TB training ...
   log Z only from the on-policy forward TB loss." Code: on prod_sep02 the fwd branch is
   Z-only (`freeze_policy`, weight 0.05) and the policy is trained only by bwd + a
   **replay branch the paper never mentions** (stored trajectories re-scored exactly,
   prioritised-IS draw). On rr_sep07 the fwd branch trains *nothing* — log Z is set by
   the closed-form fill (see B.7).
7. **Frac/ratio controller — REMOVED, then replaced.** Paper p.14 + S4: fwd:bwd *step
   ratio* driven by the eq. 13 slope/intercept fit (M > 1.2 halve, M < 0.8 +5%). Code:
   no step ratio exists; fracs are loss weights in one fused step; slope/intercept
   survive as eval diagnostics only. prod_sep02: fixed 0.05/0.475/0.475. rr_sep07:
   `gated_ramp` on `bwd/under_coverage_rise150` (B.8). S4 should be deleted.
8. **Prior model for bwd — CHANGED twice.** Paper S2: noised *prior states*. prod_sep02
   default: the churn budget is drawn from a frozen copy of the phase-1 policy
   (`snapshot_prior`) with anchors as backfill — an interim mechanism the paper never
   described. rr_sep07 (`source: anchors`): back to noised anchors, closest to the paper.
9. **Automated LR controller — CHANGED.** Paper: ramp to 5e-4, decay to 1e-5, log Z lr
   1.0, train "until loss instability". Code: constant per-system rate chosen by a
   brute-force bracket; log Z scalar at lr 0.1 inside the fused Adam (and on rr_sep07 it
   never steps at all); excursion bars rewind to the last checkpoint at the *unchanged*
   rate; > 1 rewind per 1000 steps aborts.

### B. Deltas you did not list — method changes the paper must state
1. **Replay buffer — NEW** (train.py:8093+; buffer.py:1299). A third training branch:
   uniform admission of forward batches with full trajectories, memoryless residence in
   steps (+ 5τ cap), |δ|-prioritised draw with self-normalised IS weights, exact
   re-scoring. "replay" does not appear in the paper.
2. **Forward rollouts on 1 in N steps — NEW** (protocol.py:256-275; train.py:3806-3830).
   Paper p.13: every step is a rollout + energy call ("energy evaluation is a key cost").
3. ⚠ **Importance-weight-prioritised anchor selection — REMOVED** (implemented,
   disabled: `refresh_every_n_evals: 0`, `replay_beta: 1.0`, assertion-guarded in
   make.py). Paper S2 describes it in three sentences. Anchors are drawn uniformly.
4. ⚠ **Online noise magnitude is a fixed constant, not the calibrated (d_low, d_char)**
   (train.py:552, 7798, 7918 use `anchor_buffer.noise_log_range = [-2.5, -1.5]`; the
   dataset's calibrated `log_noise_range` is used only by the offline noised set at load).
   The constant's top (0.032) is ~half the calibrated d_char (0.056-0.076). Paper §4.3
   step 2 / S2 claim calibration. **A regression in the method.**
5. **Admission gate on noised rows — NEW**: E < E_min(c) + 100 kJ/mol (~40 kT), with a
   record-breaker bypass. **Noised-buffer exit rule — CHANGED**: energy-gate expiry
   (primary) + low-residual overflow drop; "older ... exiting preferentially" was never
   true (no age term). **Size — CHANGED**: rebuilt at equilibration entry to 62.5k
   admission-gated rows (capacity 250k), refreshed inside every eval; not "200k
   pre-generated before training".
6. **Phase-1 log Z warm start — CHANGED**: no TB loss in phase 1; log Z bootstrapped at
   entry from a forward eval batch's mean log w (Part 1). **ε_MLE — CHANGED**: exit is
   the W1 progress gate.
7. **log Z by closed-form fill — CHANGED (rr_sep07)**: Z is the winsorized-Huber root of
   the rollout batch's log w (β = 10, applied when |gap| > 0.5 nat and > 3 se), held
   between rollouts; no optimizer. On prod_sep02 additionally a **z_calibration sidecar
   — NEW**: sensor-gated extra Z-only Adam steps on *fresh* rollouts (up to 100/step,
   real energy cost), which the paper does not describe.
8. **`gated_ramp` controller — NEW (rr_sep07)**: replay share +0.0017/tick, −0.043/tick
   when the 150-step rise of bwd under-coverage exceeds 1 nat; bwd floor 0.5 (MLIP) /
   0.25 (ELJ); fwd pinned 0. **Replay memorisation sensor + freshness servo — NEW**
   (`resid_vs_intake`, bar 1/e; churn × B, residence / B at fixed occupancy).
9. **S1 hyperparameters — CHANGED**: 512 hidden (not 1024); **no EMA** (`ema_decay:
   null` everywhere; paper says 0.95); one fused Adam (not separate fwd/bwd/Z
   optimizers); Huber β = 80 for bwd/replay, 10 for fwd (paper: 10); `pb_drift_range`
   0.4 (paper: 0.2); batch 1000 grown to 4000 on occupancy (ELJ) / fixed 1000-1600
   (MLIP), not 50 → 3000; budget 7-day single leg from a phase-1 checkpoint (not 4 days).
10. **Eval convergence diagnostics — NEW**: IS log Z (log-mean-exp), Jensen bound, Kish
    ESS, |log Z_IS − log Z_θ| as the primary statistic; r² is against the diagonal, not
    Pearson. The S7 parity fit survives as a figure.
11. **MACE + acridine — NEW, conditional on shipping** (4 short-wall arms, no 7-day arm;
    fresh MACE lattice energies sit 11836.127 kJ/mol below the stored prior values —
    analysis must not mix them).
12. **`freeze_pb` — NEW, unshipped** (other tab's uncommitted work): full snapshot freeze
    of P_B + its trunk; no production arm declares it; no manuscript change unless adopted.

### C. Reward / energy — the SM's equations vs the code
1. **Soft clip — CHANGED**: R_range 250 (paper 100); dead from 2026-07-12 to 2026-09-02;
   at 250 it is never reached on-policy (a guard). **Structure — MISDESCRIBED**: nested
   clips, not "separate" — inner on E_phys + 10 E_ρ (+ PV), outer on that plus the walls
   at 1.1 E_max, Jacobian outside both.
2. **Bounding and reduction × T — CHANGED** (aff3897, post-paper): temperature-
   independent in reward space; S16 as written softens them with T. **Density coefficient
   10 — MISDESCRIBED**: absent from S16 but was in the paper-era configs. **Density
   penalty — CHANGED**: C¹ linear tail below u = 1 (de5bf0c); the "2" on the
   high-density term never existed in any commit.
3. ⚠ **PV term — NEW, silent**: 1 atm × V/(Z·Z') inside E_phys on every route
   (~0.02 kJ/mol, negligible; makes the ensemble isobaric). Add to S6 or delete.
4. **UMA lattice energy — CHANGED / undescribed**: E_crystal/(Z·Z') − E_gas with the gas
   leg evaluated once per molecule (halves MLIP cost); **periodic neighbour-graph fix
   (F-047)** — the paper's UMA models, priors and the LJ/UMA rescaling ratio were all
   produced on the defective graph (~1 kT non-directional noise, 7% of rows off > 1 kT).
5. **LJ→UMA scaling — MISDESCRIBED**: ratio of lowest-decile *tail means* (0.364 /
   0.156), never a ratio of minima (the docstring says why). **Softened LJ —
   MISDESCRIBED**: k = 2.5/σ_ij (not 2.5), giving ~107 ε at r = 0; 10 Å cutoff, ε = 1.
6. **Latent ranges — MISDESCRIBED / dangling**: "(detailed in SM)" points nowhere; the
   coded ranges are in Part 1 → Model.

### D. What the audit says stands as written
- The learned P_B corrections (`learn_pb`) match the paper's description.
- S6's nearest-image *forward* kernel — now implemented exactly.
- Backward-TB gradients kept off log Z (`freeze_z`) — still true.
- The parity-plot regression as a diagnostic (S7 figure) — still produced.

### E. Owner decisions the text depends on
- Which arms the paper reports: prod_sep02 (Z-only fwd at 0.05 + sidecar + prior model)
  or rr_sep07 (fill-only Z, 1-in-N rollouts, anchor-only, gated ramp). A.6-A.8, B.2,
  B.7-B.8 read differently under each.
- Whether to restore the calibrated online noise range (B.4) and the IW-prioritised
  anchor draw (B.3), or to write the paper to the frozen-uniform design.
- PV term: keep and state, or delete (C.3). Acridine/MACE: in or out (B.11).
