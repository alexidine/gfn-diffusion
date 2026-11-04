# Converging the conditional sampler across a huge condition space: a fresh look (2026-11-04)

*Status: proposal and analysis, not policy. Written by the review hub overnight at the owner's request
("attack afresh converging a TB-ish model on a huge conditioning space"). Source: workflow wf_c1c46233-a70, which ran
four evidence readers (run history, code, the 2026-11-02 reviews, literature), five independent designers, two
refuters per design and a completeness critic. Every derivation here was checked only on CPU toys. A toy checks a
mechanism; it is never evidence about crystals. Nothing was launched. Raw reports: hub scratchpad
`conv_afresh/out/` (synthesis.md, critic.md, ground_*.md, routes.json).*

## 1. Topline

- **The owner's framing.** "Pooled VarGrad or learned Z(c) is necessary for off-policy training" is true only if
  stored rows must receive an energy-shaped (π-shaped) force. With an exact-density policy they need not.
- **Split the work by currency.**
  - **Absorption:** stored rows (energy already paid) absorb the buffer through conditional MLE. MLE needs no level
    and absorbs from a single source.
  - **Correction:** energy calls go only to fresh on-policy draws, one per visited condition, trained by
    score-function reverse KL with a detached amortised baseline.
  - **Nothing accumulates per condition**, so the visit-rate wall does not apply.
- **Why one draw per condition is enough.** On a fresh draw, E_q[∇log q] = 0, so any per-condition constant
  subtracted from the residual is a baseline:
  - an error in it costs variance, never bias, as long as it is not clipped (Huber knee, §2);
  - so TB with a learned head and VarGrad's group centre differ only in variance;
  - the toy (`conv_afresh/math_lens/headk1.py`) says K = 1 with a head baseline beats grouped K = 2–4 at equal calls
    when the head's error is under about 10 nats. That held at a toy between-condition spread τ = 3; the real τ has
    never been measured.
- **Where a level error does harm: on a stored row.** There, a level error e = L − log Z(c) adds +e·∇KL(μ_c‖q_c)
  to each descent step:
  - a level that is too high pushes mass off the buffer;
  - one that is too low is an extra MLE pull;
  - this fits the p2lvl collapses (memory `project_p2lvl_nov01_battery`; review `zc/zc_map.md` §4d).
- **Rule:** a level never multiplies a stored row, and never multiplies a stale replay row without an importance
  correction (§5).
- **What the learned head can honestly be.** A head fed by fresh draws estimates the forward Jensen level
  J_F(c) = log Z(c) − KL(q_c‖π_c), not log Z(c). It is good as a baseline and as the owner's sidecar monitor. Its low
  bias is a safety property if it is ever used as a floor level.
- **Honest prior.**
  - No energy-trained arm has beaten continued MLE on training-molecule excess beyond about 20k steps.
  - TB did keep the smaller held-out Jensen gap on qm9full: 3.20 against 4.45 nats
    (`project_qm9full_sep30_training`). The one recorded TB advantage is a transfer advantage.
  - No arm has yet trained the NF policy on-policy with no level touching stored or replay rows. p2lvl's forward
    branch was Z-only.
  - So the decisive question is narrow: do on-policy energy calls buy anything over MLE, judged on held-out
    molecules?

## 2. Recommended route

Losses: λ_M·L_MLE + λ_R·L_R on the flow and encoder, plus a separate head loss.

1. **L_MLE on stored rows.**
   - Definition: −log q(x|c), uniform over rows; this is the existing bwd `mle`.
   - What it fits: the buffer law (search-visit weighting), not π. Its job is coverage and transfer.
   - It stays on as rehearsal. In toys, reverse KL alone was mode-seeking (KL(π‖q) rose from 7.5 to 11.9 nats) or
     diverged.
   - Anneal λ_M to a small floor on a step-count schedule, not a gate, so the anneal terminates. Until the floor is
     reached, the fixed point is a per-condition compromise between reverse KL to π and forward KL to the buffer law
     (toy `conv_afresh/math_lens/fixedpoint.py`).
2. **L_R on fresh draws only.**
   - Draws: one per visited condition, drawn without gradient (as `NFPolicy.get_traj_fwd` already does).
   - Coefficient: a = sg[log q − log R + b(c)], with b(c) = sg[head(c)].
   - Knee: the default fwd β = 10 binds on roughly 40–45% of rows at a within-condition sd(log w) of about 13 nats.
     Past the knee the baseline enters the gradient (toy: 0.55 of the exact gradient at b = J_F, 0.015 at J_F + 30;
     wiki `trajectory-balance.md`). Proposed: β ≈ 40 on the fresh line with global gradient-norm clipping; β = 10 as
     the fallback arm. Report the knee fraction.
3. **Head: squared regression of head(c) onto sg[log w] on fresh rows** (fwd `z_level`).
   - Use squared loss, not Huber: the Huber root sits 2–4 nats high when log w is left-skewed.
   - The head reads only the detached invariant pooled embedding plus the sg/T/Z′ columns.
   - **Code fact (checked):** fwd `freeze_z` detaches `log_Z_learned` before `z_level` reads it
     (`gflownet_losses.py:437-439`, `:573-576`), so `freeze_z 1` + `z_level` leaves the head untrained at zero.
     Train it with `freeze_z 0` and a wide knee, or with the stage `fwd_z_sidecar`. Set its learning rate, since
     `lr_flow` is pinned outside the LR controller. Gate the policy term on mean(head − fresh log w) ≈ 0 after a
     warm-up, with a step cap.
4. **Batches.**
   - Stored rows: uniform over rows.
   - Fresh draws: conditions drawn iid over the library, K = 1, B_f ≈ 1,000 conditions per rollout. Every rollout
     costs B_f energy calls.
   - At 3.32M molecules each molecule is visited about once per 3,000 rollouts. Whether energy training transfers to
     never-visited molecules is experiment 4.
5. **Space-group weights.** Never take a softmax over a J_F-type head: it weights groups by Z_g·e^(−KL_g). Space-group
   weights come from per-group IS on a fixed panel (`eval/nf_fit.py`).

## 3. Knowing what conditional MLE learned (zero energy calls)

**Information ladder** (new, about 60 lines in `eval/nf_fit.py`). For each row x of condition c:
- I_own = log q(x|c) − log q̄(x), where q̄ is the model's own mixture over training conditions;
- I_nn is the same with c replaced by the nearest other training molecule;
- I_spec = I_own − I_nn.

How the ladder is computed and read:
- **Estimator:** the own and the 8 nearest conditions exactly, plus 256 uniform draws from the complement.
- **Bound:** on training rows I_own ≤ log N exactly (13.4 nats at 681k molecules, 15.0 at 3.32M). Held-out rows lack
  an own term, so their estimate is biased up; measure that bias on training rows with the own term removed.
- **Reading:**
  - **memorised rows:** I_own near log N on training rows, below 0 on held-out molecules and held-out basins, and a
    held-out NLL that rises after its minimum;
  - **learned the marginal:** I_own ≈ 0 everywhere and no scramble gain;
  - **nearest-neighbour lookup:** I_spec > 0 on training rows, ≈ 0 on held-out molecules;
  - **learned the conditional (of the buffer law):** 0 < I_own < log N, held-out I_own > 0, I_spec on held-out
    molecules close to training I_spec at matched neighbour distance, and mass on unseen basins.
- **Partners:** the existing `scramble`, `swap`, `amortise` and `c2st` probes. Their baselines (space-group-only and
  nearest-molecule models) have never been calibrated.
- **HPD rank** (Hermans et al. 2022) is used only as an overconfidence detector. Over all conditions it reads
  calibrated for a model that ignores the condition.
- **Row-count ablation:** k ∈ {1, 2, all} rows per molecule at matched passes. If held-out-basin I_own rises with k
  but held-out-molecule I_spec does not, rows feed within-molecule structure but not the condition map. That is the
  direct answer to "does it learn more than the three points".

## 4. Experiments, in order (proposals for approval; nothing launched)

| # | question | energy calls | compute | decision it feeds |
|---|---|---|---|---|
| 0 | Which rows did each candidate checkpoint train on (raw minima vs noised children), and is its head trained? | 0 | config read | which ladder/HPD readings are valid; which checkpoint can start experiment 3 |
| 1 | Did conditional MLE learn the conditional, nearest-neighbour lookup, the rows, or the marginal? | 0 (about 10k for an optional held-out J_F check) | about 2M density evaluations, under 1 GPU-hour, local | conditional: proceed; lookup: embedding noise, plus energy visits to most molecules; memorised: fresh noise and an earlier exit; marginal: fix the encoder first |
| 2 | How large is KL(q‖π) per condition at the MLE exit? | about 65k (64 conditions × K = 1024), after the paused mle28 run | under 1 GPU-hour, local | whether IS readouts can report log Ẑ or only ESS and k̂; how large a floor margin can be |
| 3 | Do on-policy energy calls beat continued MLE on held-out molecules? | about 6M per energy arm (halved at cadence 2) | a few GPU-hours per arm | adopt the route, or stop spending training calls and put them into relax-and-absorb (§5) |
| 4 | Does energy training transfer to molecules it never visited? | about 100k per readout | cluster | what fraction of pool v1 the energy step must visit |
| 5 | Does fresh noise per draw delay the memorisation turn? | 0 | a few GPU-hours, local | whether to reopen live noising (parked 2026-11-01) |

Caption: costs are designer and critic estimates, UNVERIFIED. The replicate spread on this recipe is 0.3–0.4 nats
of held-out NLL (`project_cap_size_nov01_battery`), which sets the noise floor for any comparison.

**Experiment 3 must be built carefully.** The critic found four confounds in the first draft:
- **Branch-weight dilution.** `fused_train_step` normalises branch weights by their total (`train.py:6052-6054`).
  Adding a fwd branch shrinks MLE's weight, and only on rollout steps. Hold the effective MLE weight fixed across
  arms, or add an arm A′ (MLE at B's per-step share).
- **Anchors-only start.** The proposed checkpoint (cap k2, `train_kinds: [anchor]`) trained on raw minima. Continued
  MLE there shrinks below thermal width, so arm A degrades by construction. Start from a checkpoint trained on noised
  children, or state the confound.
- **Threshold inside the noise.** Run a paired design (same checkpoint, same condition and row RNG streams) with two
  seeds. Pre-register the effect, e.g. ≥ 2 kT median excess on held-out molecules. Report at equal steps and equal
  wall clock, with energy calls listed separately.
- **Group size against coverage.** At equal calls the grouped-VarGrad arm (K = 4) visits a quarter as many molecules.
  Report distinct conditions visited, and add K = 2 if calls allow.

Arms:
- **A:** continued MLE.
- **B:** MLE plus K = 1 reverse KL with the head baseline.
- **B-frozen:** B with the encoder frozen after the MLE exit. tbhead lost conditioning through the shared trunk
  (xcond 0.89 → 0.38), so make xcond a stop criterion.
- **D:** MLE plus condition-grouped VarGrad on own draws.

Expect B ≈ A on training molecules (the `qf30_tbl` precedent); judge on held-out molecules.

## 5. Other routes, and when to switch

**Added by the critic; no designer considered them:**
- **Importance-corrected replay** (reuses energy calls with no level).
  - Store each draw's birth log q_old. Then E_q_old[(q/q_old)(log q − log R + b)∇log q] = ∇KL(q‖π) for any
    detached b.
  - The cost is variance from the ratio (truncate it, PPO-style), not bias.
  - It is the only route here that gets more than one gradient per energy call. UNVERIFIED that replay can store
    birth log q today.
- **Relax-and-absorb** (expert iteration).
  - Model draws relax in 43–52 steps against 77–93 from random starts (`project_qm9full_sep30_training`).
  - So calls can relax model draws on new pool v1 molecules, and the resulting minima become MLE rows. MLE stays
    level-free.
  - This merges "fresh draws" and "more searched rows" into one loop, and is the natural way to give pool v1 (SMILES
    only today) its rows. It is the fallback if B ≈ A.
- **Geometric λ-path entry** (`contingency_quiver.md` A1).
  - The target π_λ ∝ q_MLE^(1−λ)·π^λ starts at zero gap, removing the sd(log w) ≈ 13-nat variance at entry.
  - A global λ is stepped by a high quantile of the per-condition Var(log w), which needs no per-condition state.
  - The switch if B stalls on variance.

**Runner-ups from the panel:**
- **Floor-level stored-row TB, as a finisher.**
  - Level L(c) = sg[min(head(c), fresh log w(c))]. In expectation L ≤ J_F ≤ log Z, so it never pushes mass off the
    buffer; the min costs about 5 nats of extra MLE weight at σ = 13.
  - It is inert while the stored-row gap exceeds the knee.
  - Switch when experiment 3 favours on-policy calls and the batch-median gap falls below β/2, with a step cap.
  - Never use a level mixed with J_B: it harms stored rows iff λ·Δ > KL(q‖π) (toy `learned_z/sweepE.py`).
- **Condition-grouped VarGrad, K = 4, own draws.** Switch if the head's error against J_F exceeds about 10 nats.
  Expect the old VarGrad shelf and LR ceiling to return.
- **Pathwise reverse KL** through `ConditionalLatentFlow.rsample` (currently refused in `NFPolicy`). Switch if
  score-function variance stalls B. It needs forces.
- **MLE only, then relaxation and per-molecule IS at inference.**

**Dropped:**
- **Pooled VarGrad with stored partners.** In expectation it is stored-row TB at level J_F, plus a fresh-row baseline
  near J_B (toy `mathcrit/checks.py`).
- **Per-condition trackers in the loss.**
- **Condition-count curricula** (`contingency_quiver.md` A3 stays in reserve).
- **Self-normalised IS on stored children.** Their proposal density is not computable (`build_noised_prior.py`), and
  reweighting stored rows gave ESS 1e-5 to 8e-4 on MIPCAS.
- **Energy-weighted MLE.** It overfit: held-out gain 0.01 nats against 0.56 for uniform weighting.

## 6. Still uncertain

- **Neither key quantity has been measured.** KL(q‖π) per condition has never been measured on a crystal checkpoint.
  The on-policy NF policy gradient has never run at large condition counts.
- **The head's error floor.** Real heads sat 6 to 15 nats above J_F (`qf30_tbl`, `tbhead`): tolerable as a baseline,
  not as a level.
- **Cross-molecule smoothness of c → q has no proof.** The AR §1 smoke gave a held-out transfer ratio of about 0.
  Experiment 1 measures it.
- **Experiment 0 is open.** Sources disagree on what rows run zo2jwtz4 trained on, and its config was not found under
  `configs/`.
- **Pool v1 has no searched rows.** "Stored rows are free" holds only for the about 681k rcoct22i1 molecules. At 3.32M
  molecules the search is the dominant energy cost, of order 10⁹ calls (UNVERIFIED arithmetic). Experiment 3 is
  really "calls on fresh draws against calls on rows", which is why relax-and-absorb matters.
