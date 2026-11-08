# Conditional crystal sampler: a critical review of the learning problem (2026-11-08)

*Status: proposal and analysis, not policy. Requested by the review hub on the owner's behalf: a reset on conditional
overfitting and generalisation. Produced by workflow `wf_d6e0ad3d-b17`: five independent reviewers (learning problem,
MLE transfer, stage-2 losses, datasets and evidence, unseen-condition transfer), one adversarial refuter per reviewer,
then a completeness critic, all Opus. Agents' scripts and pulled W&B histories:
`C:/Users/mikem/AppData/Local/Temp/claude/2a693c8d9c6b4b73/d34125ff-6d3b-4cb8-8074-fb722e99e452/scratchpad/review_nov08/`
(session scratch, not durable). Only findings that survived refutation are stated here, in their corrected form.
Re-checked by hand at commit `ae89b4b3`:*
- *the condition-draw gate (`train.py:3455-3465`);*
- *the canonical route (`configs/mk_dev.yaml:226`, `:638`, `:976-1002`);*
- *the battery encoder (`configs/norb_mle_nov05/make.py:4,28` against `models/nf_policy.py:37,321`);*
- *the held-out file builder (`D:/crystal_datasets/conditional/norbornane_nov04/priors/scripts/heldout_test_file.py`);*
- *the W&B numbers in Table 3, pulled independently.*

*Toys are mechanism checks, never crystal evidence.*

## 1. Topline

1. **Ill-posed: the battery's held-out score is not a score for the Boltzmann target.**
   - **What it is.** `eval_test/mle` is the uniform mean of −log q at the held-out molecules' raw search minima within
     10 kT. There are about 50 per molecule, a number set by the search quota, not by the landscape.
   - **Why it fails as a Boltzmann score.** It is minimised by equal basin weights and has no lower bound as q
     sharpens onto the minima.
   - **Size of the effect.** A harmonic (Laplace) estimate on the training basin set says the true π_c would lose a
     median of about 2.8 nats per row on basin weights alone. The Boltzmann effective basin count is about 4 of 50.
   - **Consequence.** Every ranking, "best point" and "overfits by 20k" reading in `norb_mle_nov05` describes coverage
     of found minima, not approach to π_c.
   - **Same-molecule contrast.** On the same held-out molecules, the reverse readout `eval_test/jensen_z` keeps
     improving to about 11k steps and then holds, while the NLL bottoms out at about 6k (Table 3).
2. **Gap: most stage-1 overfitting is memorisation within a molecule, not identification of the molecule.**
   - **Size.** At the best step, about 1.4 of the 2.3-nat held-out-molecule gap already appears on unseen basins of
     training molecules. At 20k it is about 25 of 32 nats.
   - **Same failure in the control.** The molecule-blind arm overfits through the same channel.
   - **Consequence for the owner's framing.** The buffer-state overfitting the owner expects in stage 2 is already
     under way in stage 1.
   - **What slows it.** Kick width is the one battery lever that clearly slows it (single seeds). `cond_bottleneck`
     and `cond_noise` cannot touch it, and they do not even bound molecule identification, because atom tokens and
     prefix features bypass the bottleneck.
3. **Error and gap: the NF stage 2 as built cannot deliver "thermalises in the limit", and cannot reach unsearched
   molecules.**
   - **The only policy gradient.** In the conditional NF stage (`conditional_vargrad.var_conditioning`, used only
     through battery overrides; mk_dev runs `kind: sde` with `protocol: unconditional_tb`), the policy's only gradient
     is pooled VarGrad. It acts on 2 stored minima and 2 stale, unweighted replay rows per condition.
   - **No on-policy force.** The forward branch trains Z only (`freeze_policy: 1.0`).
   - **Molecules without rows get nothing.** A condition gets policy gradient only if it holds ≥ 2 prior-buffer rows
     (`train.py:3455-3465`, `mk_dev.yaml:976`). The 3.68M-molecule pool, and any molecule without searched rows,
     never receives an energy gradient in this stage.
   - **Basins can drop silently.** Every live loss has zero-loss fixed points that drop basins which are neither
     stored nor sampled.
   - **Fixed point moves off π.** Any non-zero MLE floor or level error on stored rows moves the fixed point off π.
4. **Gap: "generalise to an unseen molecule" has no operational definition.**
   - **What is missing.** There is no per-molecule metric and no reference points: no molecule-blind energy-trained
     control, no non-learned Laplace mixture, no leave-in oracle. There is also no named deliverable: zero-shot CSP
     cost to the best structure, against Boltzmann quality.
   - **What the current numbers cannot carry.** q-draw IS has ESS 0 on held-out molecules, so absolute reverse KL is
     unidentified. The pooled train-versus-held-out Jensen gap is confounded by level.
   - **What is level-free.** Per-molecule paired ΔJ_F between arms with the same reward, and relative `jensen_z` across
     arms on fixed molecules.
   - **Effective sample size.** Every held-out claim rests on 11 constitutions, and that set is also used to select
     the checkpoint.
5. **Confirmed sound: there is real, if modest, conditional transfer, and the held-out splits are clean.**
   - **Conditioning helps.** `cond_plain` beats its matched `blind_plain` by 1.09 nats on held-out minima, and
     `jensen_z` agrees: best 13.0 against 3.0, single seed.
   - **The split is clean.** It is constitution-disjoint, with no sibling or parent leakage, scalers fitted on
     training data only, and readouts in eval mode.
   - **Seed noise on smoothed curves is small.** About 0.07 nats, with one degree of freedom.

**Verdict on the owner's framing.**

| Owner's claim | Verdict | Basis |
|---|---|---|
| Generalisation is nonzero but poor; testing is not thorough | **Nonzero: supported.** **Poor: not yet judgeable.** **Not thorough: true, more than stated.** | Nonzero: topline 5. Poor needs a ceiling (leave-in oracle) and a non-learned reference (Laplace mixture). Not thorough: the metric is improper, effective n is 11, and the space-group axis is untested. |
| Stage 1 is "just fitting" | **Partly misattributed.** | The dominant overfitting channel is within the molecule (topline 2). "Except tiny models" is untested: cap128 and cap64 were still improving when the run ended at 20k. Stage 1 also fixes which basin *locations* an unseen molecule's q covers, and reverse-KL stage 2 cannot restore a basin that stage 1 dropped. |
| Stage 2 thermalises; overfitting only via buffer states | **Monitorable: yes. "Only buffer states": wrong. "Perfect in the limit": not supported as built.** | Monitorable: the stored-row against fresh-draw pair exists, and it already fires in stage 1. Wrong because of mode-dropping fixed points. Not supported because of topline 3. |
| Unseen conditions could be a huge problem | **Agreed, and sharper.** | Energy calls on visited molecules are still N examples on the condition axis. The live stage cannot train unsearched molecules at all. No usable evidence says energy training transfers better than MLE (§6). |

## 2. The learning problem (question 1)

**What the stored rows sample.**
- **Weights.** p_buf(x|c) is close to uniform over the about 50 quota-limited minima found within 10 kT. The
  stop rule is `quota 50, quota_band_kT 10.0` in the basin `meta.json`.
- **Shape.** It is convolved with an isotropic metric kick: log-uniform radius, 0.01–0.42 Å in base, so mostly below
  thermal width. It is then truncated by the energy window.
- **Distance from π_c.** It is far from π_c in basin weights (effective count about 4 of 50) and in basin shape.
- **Support.** Whether π_c is absolutely continuous with respect to the buffer is unknown. The Good-Turing missing
  *visit* mass is a median of 0.35, but the Boltzmann mass on unfound basins has not been measured.

**What reward-free MLE can identify from 3–100 rows.**
- **It can identify:** roughly the map from molecule to the *locations* of a quota-limited sample of low minima.
- **It cannot identify:** basin weights, which are uniform by construction; thermal shapes, which the builder sets;
  or log Z_c.
- **These need energy calls or the stored Hessians.** Every minimum carries `basin_E` and `basin_H`, and stage 1
  uses neither.

**Is the held-out metric measuring what we think?** No.

| Quantity | What it averages | Proper for π_c? | Comparable across arms? |
|---|---|---|---|
| `eval_test/mle` (held-out molecules) | −log q at 1,685 raw minima of 35 held-out molecules (11 constitutions), uniform over minima | No. It charges π_c a KL(u‖w) of about 2.8 nats median and has no peaking bound. | Yes. Identical rows in all 24 arms. |
| `eval_bwd/mle_heldout_rows` (held-out crystals) | −log q at 4,096 noised children of held-out basins of training molecules | No. Its entropy depends on the kick kernel (+3.6 and +7.2 nats for the wider kicks). | Only across arms sharing base's prior file, window and seed. |
| `eval_test/jensen_z` | Pooled mean of log R − log q over forward draws on held-out molecules | Reverse-KL-like, but mode-seeking: dropping all but the top basin costs only about 0.8 nats median. | Relative values on fixed molecules are comparable; absolute values are not, because ESS is 0. |
| `eval_test/logw_std_within` | Within-condition sd of log w | No. Within runs it tracks the unreasonable-draw fraction at r ≈ −0.86 to −0.95. | Measures junk suppression only. |

Caption: what each logged held-out readout measures, from the readout code (`train.py:10187-10213`, `utils.py:2190-2202`)
and the dataset builders. The KL(u‖w) figure is a Laplace estimate over the 117 training molecules' 8,696 minima. It
omits the chart reference term `log_ref`, which spans about 3 nats over rows. About 1% of Hessians are near-singular or
non-PSD; with a 5th-percentile eigenvalue floor the 90% quantile is about 4 nats, and unfloored it is about 20. The
median holds with or without the Hessian term.

**A well-posed definition, proposed.** "Generalises to an unseen molecule c" should mean a per-molecule quantity
judged against references. It should not be the pooled NLL of buffer rows.
- **Thermal quality.** Per-molecule KL in both directions, against a non-learned Laplace mixture and a molecule-blind
  control. Until log Z_c is identified, use per-molecule paired ΔJ_F between arms (= −ΔKL for the NF, at the same
  reward).
- **CSP cost.** Energy calls to reach the floor structure from model draws plus relaxation, against random starts
  and the molecule-blind model. This is the owner's "total cost to the best structure".
- **These can disagree.** Which one is the paper's deliverable is an owner decision (§8, check 0).

## 3. Training procedure (questions 2 and 3)

**Stage 1 (conditional MLE).**
- **Draws are row-uniform.** Per-molecule gradient weight varies about 4× with the surviving-child count (1,147–4,613
  rows). This is a mild bias and low priority.
- **No exit on held-out fit.** The conditional_vargrad exit gate tests training-marginal fit (w1r), with a bound of
  `max_steps` 50000. A future end-to-end run would enter stage 2 blind to held-out overfitting. Batteries choose the
  stage-2 source by hand today, so this gap is latent, not live.
- **Selection uses the reported set.** Best points are chosen on the same 11-constitution set that is reported, with
  no validation split. Archives every 2,500 steps miss most best steps.
- **The memorisation channel (topline 2).**
  - **Where the gap sits.** Training children reach about −15.5 nats. Fresh thermal draws around the same basins rise
    from their best, near step 8k, to about +5.5 (2 seeds). Unseen basins of training molecules carry most of the
    held-out-molecule gap.
  - **The stopping signal.** The turning point of `eval_bwd/mle_kernel` is the within-molecule stopping signal.
  - **What is not yet separated.** Whether the channel memorises the 100 fixed children per minimum, or the minimum
    locations themselves (§8, check 4).

**Stage 2 (energy).** The table lists what each live term does to each row source in the NF conditional stage as run
in `p2lvl_nov01` (battery overrides of `mk_dev.yaml:976-1002`).

| Term | Stored rows | Fresh draws | Replay rows | Fixed point |
|---|---|---|---|---|
| pooled VarGrad (`pooled_vg`, β 40, replay source) | Lift, through a zero-sum coefficient against replay | none (fwd `freeze_policy: 1`) | Push-down. In expectation over stale rows this is ascent on KL(q_old‖q). `birth_log_pf` is stored but unused. | π restricted to any basin union that contains every stored row (zero loss, zero variance) |
| `emp_z` Z sidecar (fwd) | none | Fits the head to the group's log w | none | Head ≈ the forward Jensen level J_F(c), not log Z(c) |
| MLE rehearsal (when on) | Forward-KL pull toward the buffer law | none | none | A compromise between π and p_buf unless its weight anneals to exactly 0 |

Caption: from `gflownet_losses.py`, `train.py` and the wiki pages `vargrad`, `on-policy-and-off-policy-training` and
`z-calibration`, checked by the stage-2 reviewer and its refuter.

Findings:
- **Error (design level): nothing in the live NF stage is an on-policy force.**
  - **The mechanism.** The policy moves only by pooled VarGrad against replay. Replay goes stale within tens of steps:
    `replay/policy_drift_ess_frac` was 0.18–0.65 at step 10 and below 0.05 by step 500 in all four p2lvl arms. With
    stale rows, the pooled push-down repels the policy from its own recent samples.
  - **What the evidence shows.** p2lvl collapsed in 3 of 4 arms. The mechanism is not identified: the surviving arm,
    tbhead, was equally stale, so the choice of centre separates collapse from survival.
  - **The surviving arm still lost ground.** No arm closed the stored-versus-replay gap: `pooled/pooled_gap_signed_avg`
    went from 55 to 64–99 nats. tbhead's `eval_fwd/jensen_z` fell from 7.0 to −7.3.
  - **What fixes it.** Importance-corrected replay is implementable now, because replay rows carry birth log q. The
    2026-11-04 note's "UNVERIFIED that replay can store birth log q" resolves to true.
- **Error (owner claim): within-condition failure is not limited to buffer states.**
  - **The fixed point.** For on-policy TB or VarGrad with a detached level, and for pooled VarGrad, π restricted to any
    basin union that contains every stored row is a zero-loss stationary point.
  - **Why nothing sees it.** A basin with no stored row and negligible q mass is invisible to the loss, to fresh-draw
    Var(log w), to ESS, to J_F and to δ = J_B − J_F.
  - **What can detect it.** Only reference rows outside q's support: an independent search or a thermal set. This is
    mode-dropping, which is a different failure from buffer-state overfitting.
  - **Owed fix.** The mk_dev comment near `:1036-1045` says δ "is the only logged quantity that sees mass MISSING from
    a basin". That holds only for basins with stored rows. The owner should qualify it.
- **Error (design note): any non-zero MLE floor or level error on stored rows moves the fixed point off π.**
  - **Conflict.** The 2026-11-04 note's "anneal λ_M to a small floor" conflicts with "perfect convergence" unless the
    floor reaches exactly 0.
- **Gap: matched-MLE controls are not matched.**
  - **Why.** A fused step normalises branch weights (`train.py:6347-6349`; the 2026-11-04 note's `:6052` is stale) and
    feeds one Adam optimiser. When energy gradients are 10–100× larger (cross-run norms of 5e3–5e4 against 2e1–4e2),
    Adam's second moment silences the MLE direction whatever its frac. Clipping is not the cause.
  - **Measurement gap.** Per-term pre-mix gradient norms are not logged.
- **Gap: Z-head traps.**
  - A Z-head term on a `freeze_z` branch trains nothing. It is reported only when no other branch trains Z
    (`train.py:2567-2593`).
  - `emp_z` without a VarGrad flavour crashes at step 1.
  - Neither combination is live. An invariant refusing them would close the trap.

## 4. Architecture

- **Gap (battery attribution): the battery base runs the xyz `AtomEncoder`, not the split encoder its docstring
  states.** `atom_tokens: True` resolves to the `xyz` encoder (`models/nf_policy.py:321`), against
  `configs/norb_mle_nov05/make.py:4`. Three consequences:
  - `atom_chirality` is inert in the 23 non-split arms;
  - `pooled_mol` (`split_pooled`) changes the encoder family and drops cross-attention together, so "pooled molecule:
    no effect" cannot be attributed to either change;
  - `rotate_frames` was not in the battery.
- **Gap: `cond_bottleneck` and `cond_noise` squeeze only the pooled molecule vectors.**
  - **What bypasses them.** Atom tokens under cross-attention, and prefix molecule features.
  - **Capacity.** At bottleneck 16 and noise 0.5, the squeezed path alone carries more than ln 117 nats.
  - **Consequence.** The knob can neither bound molecule identification nor address the dominant within-molecule
    channel. Whether it helps transfer by smoothing is untested.
  - **No other regulariser.** The conditioner and encoder have no dropout.
- **Gap: the typed-row anomaly.** `rows_typed` is 0.5–0.7 nats worse than plain rows across the pre-overfit window
  (single seed). That is large for a feature change. Nobody has read the typed path of the row describers
  (`models/nf_policy.py:272-286`). A defect there could also affect the prefix base.
- **Unresolved: capacity.** cap128_l2 and cap64_l2 were still improving at 20k, so their "best" values are
  truncations. Whether a small model avoids the overfitting has not been tested at convergence.

## 5. Datasets

| Set | Molecules | Constitutions | Rows | Row kind |
|---|---|---|---|---|
| Training (P-1) | 117 | 43 (13 singletons) | 447,677 (1,147–4,613 per molecule) | Noised children of minima within 10 kT, K = 100 metric kicks |
| Held-out molecules (`norbornane_p1_test_w10.pt`) | 35 | 11 (sizes 1–6) | 1,685 (25–52 per molecule) | Raw minima within 10 kT |
| Held-out crystals of training molecules | 117 | 43 | 4,096 drawn from 23,442 in base | Noised children of held-out basins |
| `mol_50` / `mol_25` subsets | 60 / 36 | 35 / 27 | — | As training |

Caption: counts from the build logs and basin-set identifiers, with constitution taken as the first block of the InChIKey
(the split key), counted by the dataset and learning-problem reviewers and their refuters. The held-out-crystal set is
rebuilt for each arm's prior file, window and seed.

- **Gap: the effective n is about 11 constitutions.** Stereoisomers of one constitution are near-duplicate
  conditions. The data-size curve should be read in constitutions: 27 → 35 → 43 kept, about 2 nats per constitution
  doubling, over 3 single-seed points.
- **Gap: the search quota sets the per-molecule minima count.** It also truncates the held-out set. "Minima per
  molecule" as a data lever, and any completeness claim, depend on that quota.
- **Gap: the P2_1/c search is unused.** It holds 16,772 minima on the same molecules. It is available as an untouched
  replication or test set, and as the first cross-space-group transfer test.
- **Confirmed sound: the split is leak-free by constitution.** Held-out molecules receive no gradient, buffer or level
  writes. Held-out-molecule rows are identical across arms. The held-out builder uses T 2.5 and density coefficient
  10, matching `configs/norb_mle_nov05/base.yaml`, with the floor over each held-out molecule's own rows.

## 6. Empirical results, graded

| Claim | Seeds | n | Selection | Confounds | Grade |
|---|---|---|---|---|---|
| Molecule count is the biggest lever ("0.7 nats per doubling, no levelling") | 1 per subset | 11 test constitutions; 3 data points | Best-of-81 on the test set | Doublings counted in molecules, not constitutions; one subset draw each; LR phase (small, about 0.08) | Direction robust: the penalty is visible at matched step 3000 inside the constant-LR hold. Slope about 2 nats per constitution doubling. "No levelling" unsupported. |
| Minima per molecule matter (`minima_50` −0.76; `window_15` −1.75 best arm) | 1 | 11 | As above | `window_15` adds +36% minima *and* +50% wider children; its proper pair is `base_parentwin` (gain about 0.22–0.27) | Halving hurts: robust direction. `window_15`'s cause is unattributed. "Molecules beat minima" (about 0.2) is unresolved. |
| Longer kicks help (−1.48 / −1.58 / −1.66) | 1 | 11 | As above | Pair with `base_parentwin`, not `base`; best-point gains of 0.10–0.30 | Best-value gain is plausible but small. The robust effect is **slower overfitting**: eval_test/mle at step 8000 is −0.82 / −1.10 / −1.52, and the 20k test-minus-anchor gap is 31.9 / 27.0 / 22.8 nats. |
| Conditioning transfers (blind −0.16) | 1 | 11 | As above | Pair `blind_plain` with `cond_plain`, not with base | **Robust:** a 1.09-nat gain, which `jensen_z` agrees with. The most solid transfer evidence there is. |
| The two base seeds agree within 0.03 | 2 | — | Best-of-81 | Raw points have one-sided spikes up to 0.5 | Right but uninformative. Smoothed matched-step spread is about 0.07 nats, with 1 degree of freedom. Constitution-sampling variance is unmeasured. |
| Prefix rows help; typed rows hurt | 1 | 11 | As above | — | Prefix over plain about 0.3: plausible. Typed +0.5 to +0.7: anomalous; read the code. |
| "No effect" knobs (LR, weight decay, floor, bins, pooled_mol, tf_l2) | 1 | 11 | As above | `pooled_mol` is a two-factor arm | Differences of 0.1–0.25 are unresolved, not shown null. |
| Every arm overfits badly by 20k; final values meaningless | 2 (base) | 11 | — | Metric-specific | True of coverage of held-out minima only. `jensen_z` peaks near 11k in base and both seeds hold about 9.5–10 at 20k. |
| p2lvl phase-2 arms collapse | 2 (pooledvg), 1 each otherwise | 1 source checkpoint | — | rc_oct22 raw-minima prior, not the norbornane noised-child recipe; the continued-MLE comparator was itself degrading; all four W&B states 'crashed' | Collapse is confirmed in 3 of 4 arms. Mechanism unidentified. Does not transfer to the current recipe. |
| TB keeps a smaller held-out Jensen gap (3.20 vs 4.45, qm9full) | 1 per arm | — | — | SDE policy with learned P_B; `reward_range` None against 250 across arms; the cmle_lead offset is present at step 50 | No usable evidence about the NF either way. |
| No energy arm beats continued MLE beyond 20k | — | — | — | No NF energy arm has run that long without collapse | Absence of evidence. |

Caption: each claim in the battery summary handed to this review, plus the older stage-2 observations, graded by the
dataset and evidence reviewer, its refuter and the critic. "n" counts held-out constitutions. The base-seed noise floor
for paired arm comparisons on 5-eval-smoothed curves over steps 4–8k is about 0.07 nats.

**Table 3. Spot check pulled independently for this note** (W&B `mkilgour/GFN Energy`, tag `norb_mle_nov05`).

| Arm (W&B id) | best eval_test/mle (nats) @ step | 5-eval-smoothed best (nats) | eval_test/mle at 8k / 20k (nats) | eval_test/jensen_z at 6.25k / 11.5k / 20k (nats) |
|---|---|---|---|---|
| base (2ycn46cg) | −1.38 @ 6250 | −1.29 | −0.64 / 13.88 | 4.66 / 13.64 / 9.45 |
| base_s2 (6ax9oeqv) | −1.35 @ 5750 | −1.30 | −0.51 / 13.68 | 4.94 / 12.20 / 9.83 |
| cond_plain (clxf4tql) | −1.25 @ 5000 | −1.14 | −0.03 / 14.20 | 5.53 / 12.18 / 7.66 |
| blind_plain (s8bzg8d7) | −0.16 @ 4750 | −0.14 | 2.10 / 10.72 | −0.87 / 1.34 / 0.73 |
| base_parentwin, 0.42 Å (5epooalr) | −1.48 @ 5500 | −1.37 | −0.82 / 13.39 | 5.82 / 12.48 / 8.24 |
| kick133_parentwin, 1.33 Å (0ivnzbzj) | −1.66 @ 6500 | −1.61 | −1.52 / 5.39 | 3.28 / 13.41 / 10.67 |
| mol_50 (pzt4owie) | −0.57 @ 3750 | −0.47 | 2.96 / 22.73 | −0.39 / 3.83 / 1.91 |
| mol_25 (cs1y7eqj) | −0.04 @ 3250 | −0.02 | 6.27 / 31.91 | 0.20 / −1.34 / −4.31 |

Caption: 81 evals per run; each cell is one seed except the two base rows. Lower eval_test/mle is better; higher
jensen_z is better, and only differences between arms are meaningful. Compare kick arms with base_parentwin, blind
with cond_plain, and subsets with the two base seeds. The best points reproduce the battery summary exactly.

## 7. Gates and controls (question 5)

- **Unbounded wait: the λ-anneal δ gate has no cap of its own.**
  - Its relative bests never reset, and δ stays missing until the quorum fills.
  - It is bounded only by the run budget (100,000 epochs), and it prints no stall.
  - It is inert while `lambda_mix` starts at 1.0. It needs a per-gate cap and a printed stall before any λ-path
    launch.
- **Unbounded wait: the replay warm-up.**
  - It can keep every step a rollout if entry rows fail admission. In p2lvl it released within 100 steps.
  - It is bounded only by the run budget, with no stall print.
- **Gap: collapse is invisible to the tripwire.** The hard-failure tripwire did not fire in any p2lvl collapse. The
  signals that moved (`eval_fwd/jensen_z`, `replay/logpf_mean`, `bwd/logpf_mean`) feed no stage rule.
- **Missing controls:**
  - a leave-in oracle (a ceiling for "poor");
  - a non-learned Laplace mixture;
  - a nearest-molecule baseline;
  - a molecule-blind *energy-trained* control, which separates generic cooling from conditional transfer;
  - a constitution-clustered standard error;
  - a validation split disjoint from the reported test molecules.

## 8. Cheapest decisive checks, ranked

| # | Check | Cost | Decision it feeds |
|---|---|---|---|
| 0 | Owner names the deliverable for unseen molecules: zero-shot CSP cost to the floor structure, or per-molecule Boltzmann quality (paired ΔJ_F / KL) | 0 | Which metric every later check reports; whether energy calls must visit pool molecules |
| 1 | Re-read the 24 arms as 5-eval-smoothed curves at matched steps (4–8k) with correct pairs: kicks against base_parentwin, blind against cond_plain, data size in constitutions. Plot NLL, jensen_z, reasonable fraction and the `eval_bwd/mle_kernel` turning point on one axis. | 0 calls; about 15 CPU min on pulled histories | Corrected battery table; the named stage-2 checkpoint criterion (coverage-best near 5–6k by default); the within-molecule stopping signal |
| 2 | Zero-call harmonic readout on the 35 test molecules from `basin_E`/`basin_H`, in the trainer chart with `log_ref` and a soft-mode eigenvalue floor. (a) Score Boltzmann-weighted and uniform Laplace mixtures on `eval_test/mle`, confirming impropriety. (b) Score archives (5k, 7.5k, 10k, 20k of base, base_parentwin, kick133_parentwin, cap128_l2, cond_plain, blind_plain) on draws from both mixtures. | 0 calls; 30–60 CPU min plus about 50 lines | Whether `eval_test/mle` is kept only as a coverage guard; whether kick width enters the recipe; location error against weight error |
| 3 | Dump per-row −log q for the 1,685 test rows from 2–3 archives per decision arm, and bootstrap over the 11 constitutions. Reserve a validation split, or adopt P2_1/c as the replication set. | 0 calls; 10–30 CPU min | The standard error any transfer claim carries; the smallest effect a battery can resolve; removes test-set selection |
| 4 | Score fresh same-law kicks (new seed) of training anchors on base archives, against the stored children | 0 calls; CPU minutes | Larger pre-built K (more children) against a wider or thermal-width pre-built kernel as the next row-design rebuild |
| 5 | Per-block decomposition of held-out −log q (cell, centroid, orientation) for test against training molecules, plus a code read of the typed-row path | 0 calls; CPU minutes plus a read | Whether encoder ablations or a row-path fix come first |
| 6 | Back-test a report-only stage rule (jensen_z drop below its running best, replay/bwd log q) on the four p2lvl histories; confirm each run's last step and crash state | 0 compute; minutes | Whether a rewind or stop rule can bound stage-2 damage before the next energy run |
| 7 | IS log Z from a floored Laplace proposal against the harmonic sum on about 10 training molecules, reporting ESS | about 20k eLJ calls; under an hour | Whether the harmonic yardstick can stand in for π_c; how much π_c mass lies on unfound basins (more search, or within-molecule generalisation) |
| 8 | 200-step smoke from a norbornane MLE archive with fused MLE plus energy, logging per-term pre-mix gradient norms | about 100k eLJ calls; under 0.5 GPU-h | Whether per-term gradient normalisation must precede any "MLE + energy against MLE" comparison |
| 9 | First NF stage 2 on norbornane, short and at production shape. Start from a fixed-step (about 5k) noised-child archive. Arms: replay as run; replay with truncated IS weights from `birth_log_pf`; fresh-paired (`pooled_source: fwd`). 2 seeds. | about 3k steps × 3 arms × 2 seeds; 3–6 GPU-h; 1–2M calls per arm | Whether the p2lvl collapse is a property of raw-minimum rows or of stale replay; whether stage 2 keeps a replay seat, and with IS weights |
| 10 | Second seeds for base_parentwin and kick133_parentwin; cap128_l2 run to convergence; window 15 kT with `window_by: parent` | 4–5 arms × 20k; 4–8 GPU-h; 0 calls | Which data lever and which capacity enter the stage-1 recipe |
| 11 | Transfer experiment, only after 0, 2, 3 and 9. Arms: A continued MLE; B plus on-policy reverse KL on training molecules; B with the encoder frozen; C blind plus energy; D visiting held-out molecules. Readouts: per-molecule paired ΔJ_F with shared noise, the harmonic readout, CSP calls-to-floor, constitution bootstrap. Thresholds set from measured norbornane-NF variance. | about 10M calls; 5–7 arms × a few GPU-h | Whether energy training transfers conditionally to unseen molecules, and whether calls must visit the pool |

Caption: costs are reviewer and critic estimates, UNVERIFIED. Checks 0–6 need no energy calls and no GPU training.
Check 11 is the question the paper turns on. It needs on-policy policy gradient that the live NF stage lacks, so it
cannot run on the current var_conditioning stage as configured.

## 9. Not settled by this review

- **The target measure in the FD chart.** No reviewer checked that the reward, the Laplace basin weights and q are
  densities with respect to one reference measure. The `log_ref` term (about 3 nats over rows) and the chart Jacobian
  must enter every harmonic yardstick (checks 2 and 7).
- **Literature figures are UNVERIFIED.** The transfer reviewer cited Transferable Boltzmann Generators, Prose,
  Adjoint Sampling, and Atanackovic and Bengio on GFlowNet generalisation; none was checked against the papers. None
  of them supports a conclusion here.
- **The 2026-11-04 note's rejection of energy-weighted MLE.** It cites only the T = 10 point. The same single-seed
  probe found T = 100 weighting better than uniform (held-out gain 0.76 against 0.56). The rejection should read
  "unresolved".
- **Knowledge-base drift owed** by whoever next touches these pages (not edited here):
  - `docs/wiki/vargrad.md` needs a full-support qualifier on its fixed point;
  - `docs/wiki/on-policy-and-off-policy-training.md` "mass transfer" holds in u-space only, and off-policy rows come
    from q_old;
  - the stale `train.py:6052` citation in `conditional_convergence_afresh.md`;
  - the owner-edited mk_dev δ comment (§3).
