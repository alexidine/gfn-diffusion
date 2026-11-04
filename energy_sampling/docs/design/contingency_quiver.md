# Contingency quiver: ideas held in reserve for the Day-landscape paper

*Status: proposals and analyses, not policy. Collected 2026-11-03 from the review hub's discussions with the owner,
who asked that every idea be kept in case it is needed. Each entry says what the idea is, when to deploy it, where
it stands, and the cheapest test that would decide it. Nothing here is scheduled. Where an entry rests on analysis
rather than a run, it says so.*

The paper's crux, in the owner's words: conditional MLE must absorb the stored buffer, and that must transfer to
held-out molecules. Most entries below are alternative routes to that, or protections against it failing. The
readouts that judge all of them are in `eval/nf_fit.py`: level-free NLL on fixed row sets, per-row NLL, the gain
over a scrambled-condition model, per-condition KL, ESS and Pareto k̂.

## A. Buffer absorption and phase 2

**A1. Geometric path from the MLE fit to the target.**
- *Idea:* anneal along π_λ ∝ q_MLE^(1−λ)·π^λ. At λ = 0 the target is the MLE model, so the stored-row level gap is
  zero by construction. Choose each step Δλ so that Var_q[Δλ·(log π − log q_MLE)] stays at or below the VarGrad noise
  floor. That is adaptive tempering, the step rule SMC uses to hold its effective sample size.
- *Status:* the machinery exists as λ annealing (`cfg:energy_config.lambda_mix` with `cfg:energy_config.prior_flow_path`;
  wiki `lambda-annealing.md`). It was built and tested for unconditional MLE, then set aside when conditional MLE
  appeared to work.
- *Conditional complication:* a global λ steps at the pace of the worst condition. A per-condition λ needs
  per-condition state, which hits the same visit-rate wall as per-condition levels. And the frozen MLE model must be
  evaluated on every sample.
- *Deploy when:* conditional MLE fits and transfers, but phase 2 cannot close the remaining stored-row gap. The likely
  form is a global λ whose step is set by a high quantile of the pooled per-condition variance.
- *Caveat:* if q_MLE has memorised training rows, the path starts from a memorised distribution.

**A2. Temperature annealing (owner, 2026-11-03), with analysis.**
- *Idea:* after MLE, raise T until J_B − J_F falls below the VarGrad noise, then anneal T down.
- *Analysis (not run):* J_B − J_F = (E_q[E] − E_μ[E])/T + (CE(μ, q) − H(q)), with μ the stored rows. Only the energy
  term shrinks with T. The density-mismatch term and Var_q(log q) do not depend on T. So heating helps only to the
  extent the gap is energy-dominated. At high T the target also stops favouring the low-energy stored rows, so
  annealing down has to re-absorb them step by step.
- *Deploy when:* the measured gap decomposes mostly into the energy term. A1 is the safer variant otherwise.

**A3. Grow the condition count (owner, 2026-11-03).**
- *Idea:* converge on a small set of conditions, where convergence is known to work (a few parallel TB problems with
  trustworthy per-condition levels), then add conditions in stages. Old conditions keep a fixed share of every batch
  (rehearsal).
- *For:* phase 2 stays local; new molecules start warm if amortisation works; the level head learns from
  well-converged conditions; seed searches are only needed for the molecules being added.
- *Against:* tight convergence on few conditions is the memorisation regime (inn_cap26 small-fraction arms; the
  1,105-molecule C1/C2 runs overfit within ~1,000 steps), so the start may need unlearning. Without amortisation,
  growth only reorders the work.
- *Precedent:* curriculum learning and rehearsal-based continual learning in general. No growing-condition curriculum
  for GFlowNets or Boltzmann generators is known to us.
- *Cheapest test:*
  1. Converge on about 100 molecules, then read KL, ESS and stored-row NLL on unseen molecules before training on
     them (zero-shot transfer).
  2. Count steps for new molecules warm-started against trained from scratch.
  3. Grow against joint training at equal compute, judged on held-out molecules.
- *Prior composition:* per-condition buffers stay bounded, old rows kept for rehearsal; new molecules need seeds
  before entry; the per-space-group molecule-unconditional prior gives a new molecule support on day one.

**A4. Pooled VarGrad with mixed groups.** Absorption needs on-policy and stored rows in the same group; a
single-source group gives no pressure to move mass onto the buffer. The hazard is the size of the gap between the
two (30-55 nats recorded), which scales an MLE-like push. Drive the gap down in MLE first and watch it as the
stability signal.

**A5. The level as a detached sidecar.** Keep a learned level alive as a monitor even when no loss uses it. It costs
nothing and gives a per-condition convergence readout. For small mismatch, KL ≈ ½ Var(log w), so the same readout
bounds how far learned space-group weights can be trusted.

## B. Generalisation across molecules

**B1. Split conditioner, one-way.** An invariant trunk on distances and elements gives per-node 0e features and a
pooled invariant s, which alone feeds log Z (detached). A point head, a set encoder over the 0e node features plus
standard-frame coordinates, gives p for the policy. 0e may feed p; p never feeds 0e. *Status:* `SplitAtomEncoder` in
`9c2f59d8` implements the invariant-trunk-to-level part.

**B2. Condition noise.** Add noise to the condition embedding at the spacing of neighbouring training molecules, so
the decoder cannot separate conditions closer than that and must share samples across them. Identity lookup costs
only about log N_molecules nats, so capacity limits alone cannot prevent it; smoothness can.

**B3. Freeze the conditioner after MLE,** early-stopped on held-out molecules, so phase 2 can only overfit the
distribution side. That side is measurable with fixed probe rows.

**B4. Frame and symmetry handling on the target side.**
- Draw one of the four proper sign-flip frames at random on each training pass: genuine augmentation.
- Average over frames at inference, keeping the density exact as a mixture.
- Symmetrise over the molecule's point group the same way.
- *Deploy when:* held-out NLL rises with low frame margin, as measured on Day molecules: 5-29% change presentation
  under 1e-5 to 0.01 Å noise.

**B5. Prefix physics features (idea to test).** The AR conditioner may read any function of the rows already drawn
and keep a triangular Jacobian. Candidates: the packing coefficient after the cell rows; symmetry-image centroid and
ellipsoid-contact features after the pose rows; space-group symmetry operations as tokens, which share meaning across
groups (a 2₁ screw is a 2₁ screw everywhere). *Cheapest test:* the packing coefficient alone, read through per-row
held-out NLL on the cell rows across space groups.

**B6. Encoder pretraining and channels.**
- Pretrain a Mo3ENet-style encoder on a large conformer set (side track; training route repaired in MXtalTools
  `4acf8918`).
- Optionally add cross-product channels (1e, 0o) and a readout-level 2e.
- Review the Gaussian overlap loss against a Sinkhorn/optimal-transport loss for larger molecules.

**B7. Spline regularisation.** Tied knot derivatives bound in-bin density (`cfg:model.nf.tied_derivatives`). Add an
explicit bin floor set as a fraction of the stiffest harmonic mode's width at the training temperature. In the
harmonic approximation that width lower-bounds every conditional width, so the model cannot be sharper than its
target.

## C. Space-group weights and evaluation

**C1. Hybrid space-group sampling rule:** p(g | m) = (1 − ε)·Ẑ_g / ΣẐ + ε·π_CSD(g). Learned per-molecule slice
weights, with a molecule-agnostic CSD frequency floor so a badly estimated Ẑ cannot drop a group. Report uniform, CSD
frequencies and learned as a benchmark ablation on coverage per energy call.

**C2. Amortised Z(c) at no extra energy cost.**
- Train the level head by regression onto per-condition estimates built from the log w that forward rollouts already
  compute. The mean estimates log Z − KL; logmeanexp is closer to log Z but biased low at small K.
- Validate against per-condition importance sampling on held-out molecules, a one-off cost reported separately.
- A head trained by TB instead carries a bias ∝ e^(−KL_g) per space group, which vanishes as the sampler converges.

**C3. Two-stage evaluation sampler (fallback if C2 is not accurate enough):** draw a pilot of K samples per space
group, estimate each Ẑ_g by importance sampling with a defensive mixture, then allocate the remaining budget in
proportion. It costs energy calls at evaluation, which counts against the efficiency metric.

## D. Added 2026-11-04 from `conditional_convergence_afresh.md`

Each idea is derived there and checked only on CPU toys.

- **D1. Level tolerance rule.** A level ℓ = J_F + a harms stored rows iff a > KL(q‖π). A mixed level λJ_B + (1−λ)J_F
  harms iff λΔ > KL.
- **D2. Floor level** min(head(c), fresh log w(c)). In expectation it never pushes mass off stored rows: a finisher.
- **D3. Mixed VarGrad identity.** One fresh plus one stored row in mixed VarGrad is, in expectation, stored-row TB at
  level J_F, with the fresh row's baseline sitting near J_B.
- **D4. Information ladder** I_own / I_nn / I_spec. On training rows I_own ≤ log N exactly. It is the readout of what
  conditional MLE learned.
- **D5. ΔJ_F between two checkpoints = −ΔKL(q‖π)** at a fixed condition. A transfer readout that needs no Z.
- **D6. Importance-corrected replay.** Store birth log q_old, weight by a truncated q/q_old. Level-free reuse of
  energy calls.
- **D7. Relax-and-absorb.** Relax model draws on new molecules and feed the minima back as MLE rows.
- **D8. Empirical-Bayes shrinkage of the head** toward group means, and contrastive Δlog Z (evidence networks) as
  monitors. Not evaluated.

## Sources

Review lane reports from 2026-11-02 (hub scratchpad `review/`): `zc/zc_map.md`, `ar/critic_conditioner.md`,
`ar/critic_process.md`, `sg/sg_free_energy_theory.md`, `prior/day_test.md`, `critic.md`. Wiki: `lambda-annealing.md`,
`vargrad.md`, `trajectory-balance.md`, `z-calibration.md`. Code: `eval/nf_fit.py`, `models/nf_policy.py`,
`models/spline_flow.py`.
