# Scope: the training workflow as it runs today, and where the manuscript no longer describes it

Written 2026-09-07 on branch `rarer-rollouts-v0`. Part 1 is the workflow in point
form, including today's edits. Part 2 is the audited list of deltas from
`Crystal_Generator (7).pdf` (26 pp.), each verified against a paper page and a
code line. Items marked *(audit pending)* are being confirmed as this is written.

---

## Part 1 — the workflow now

### Phase 1 — backward warm start (run once per system; the `pt100_*` runs)
- `train_prior` stage, `train_mode: bwd`, `bwd_sampling_mode: dataset` — terminals drawn
  from the fixed prior dataset, trajectories sampled backward under P_B.
- Trains P_F by MLE onto the prior's support and warm-starts log Z from TB gradients on
  those backward trajectories (Z-only).
- Exits on its convergence criterion and writes `*_phase1_exit.pt` (weights, buffers,
  optimizer, stage state). Every production arm resumes from that file with
  `load_weights_only: false` — a full continuation, not a weights-only restart.
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
  no thinning, no surprise refresh; draws are unweighted. This is the paper's "noised
  buffer" source: rows are isotropically noised (`noise_log_range`) and re-scored when
  drawn into the prior buffer.
- **Prior buffer** — bwd's sampling distribution. Seeded from the prior dataset;
  refilled by churn (`mean_lifetime` per eval call) **from noised anchors**
  (`source: anchors`, today's edit). Previously refilled by a learned prior *model*;
  that model is no longer built, loaded or resumed. bwd draws 90% uniform / 10%
  loss-weighted (`weighted_bwd_beta: 0.9`).
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
- Domain penalties in `generator_energy`: bounding (box manifold), density, reduction.

### Model *(audit pending on the details)*
- SDE policy P_F with drift and variance heads over T = 100 steps; learned bounded
  correction to the analytic backward kernel (`learn_pb: true`); exact-reversal P_B on
  periodic angular dimensions (`pb_exact_reversal`); low-rank policy covariance (DPLR);
  redundant/flat latent dimensions removed; standardized box manifold.

---

## Part 2 — deltas from the manuscript

*(populated from the verified audit; each entry: paper page and quote → code file:line →
delta type → suggested wording)*
