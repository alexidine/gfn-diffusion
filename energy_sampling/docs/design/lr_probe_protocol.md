# Discrete LR calibration and control — protocol v2 (rev d)

Status: design spec, 2026-08-21, rev d. **Probe scaffolding and simulator
are approved for implementation behind `lr_probe.enabled: false`; actuation
remains gated on sections 6.1-6.5.** Rev c was the simplification pass after
the adversarial review (W ~ 10 with replication; pooling, hyper detector,
margin, permuted null, escalation ladder, and instability ceiling moved to
appendix A; numeric bars everywhere). Rev d fixes four review points on rev
c: the probe's bias direction is UNKNOWN (rev c's "reads hot" was backwards
w.r.t. Wu et al.) and the T_route divisor is deleted; the entry search is
restructured as screen-then-confirm (rev c's two entry rules contradicted
each other); the unit test is an exact binomial sign test (sign-flip
permutation is exact only under a symmetry assumption nobody validated); and
the validation/cost anchors are tightened (W-sweep includes the default W;
the pinned-rate reference must be a controlled fork with the LR override ON
and live rates verified; the cost gate counts always-on harvesting).
Companion documents: `docs/hypergradient_review.md`,
`docs/lr_control_session_2026-08-20.md`.

---

## 0. Decisions on one screen

- **Piecewise-constant LR.** The rate changes only through a probe or the
  emergency rollback path. No continuous actuation anywhere.
- **The probe is a short-window, replicated, frozen-data fork** — a
  multi-arm generalization of `ray`: real optimizer steps (Adam, momentum,
  clip), on any stage with a harvest (including phase 1), on the stage's
  actual loss composition, at roughly ray's cost. W ~ 10 steps per window.
- **Small-W frozen probes have an unknown, route- and regime-dependent
  bias.** Two known components point opposite ways: short *stochastic*
  horizons bias COLD (Wu et al., ICLR 2018 — orders of magnitude on noisy
  quadratics), while training on frozen data with no on-policy feedback
  biases HOT. The net direction is not derivable; it is **measured against
  on-policy forks** (6.3). There is **no numeric bias correction**: if the
  surrogate's winner is not within one rung of the on-policy winner, W grows
  until it is, and a route where no W satisfies both 6.3 and the cost gate
  is unsupported (holds its seeded rate) until v2.1.
- **Incumbent-as-null selection.** A discrete selector: move only on a
  significant, replicated, sign-consistent win over the current rate;
  symmetric up and down; inconclusive holds. No margins, no pooling, no
  extrapolation. Centering under asymmetric curvature is *demonstrated* in
  simulation (7.1), not engineered.
- **`hyper` and `ray` actuate nothing.** Ray's actuation path is removed
  (with the benchmark-registry disable fix); the entry probe takes over its
  one safety job (rate-setting after stage entry). Emergencies are the
  existing `fire_loss_spike` rollback plus the section-6 tripwire work.
- **Opt-in, conformer-first**; canonical crystal config untouched; every
  conformer result names its internal-coordinate tier. Config surface:
  `{enabled, window}`.

## 1. Evidence base and scope

- Within-stage, the optimum is stationary **where measured** — phase-2 ray
  readings on crystal routes (variogram flat 500 -> 40k steps, mean pinned
  +-7%). This licenses "no dumb clock", not universal stationarity; the
  composition trigger (4.2) covers the drift the user expects in phase 2.
- The non-stationarity that matters is between stages and across composition
  changes — exactly what no existing sensor measures (ray has never fired
  outside phase 2).
- Pairing is the sensor (~30x in t measured); unpaired probing produced 35%
  self-contradictory readings. The frozen larder gives exact pairing.
- Asymmetric response to noisy readings rectifies into drift (railed `hyper`
  1/800 down; holds ray ~+15% hot). Every rule here is symmetric, and 7.1
  tests the whole policy for drift with a numeric bound.
- Bias directions on record: *deterministic one-step* probes read hot (why
  ray runs at 1/4 of its one-step optimum), *stochastic multi-step* unrolls
  read cold (Wu et al., ICLR 2018). This probe is neither exactly — real
  stochastic steps, frozen data, tiny horizon — so its bias direction is
  unknown until measured (6.2/6.3). Rev d accepts small W and bounds the
  bias empirically rather than paying ~40x window cost to argue about it.
- The disease: chronic 5-10x (once 800x) cold-running on routes that train
  fine at a hand-pinned rate.

## 2. The probe

### 2.1 Harvest

During normal training keep a rolling verbatim record of the trailing live
batches per active branch (fwd, bwd, replay: trajectories, rewards, latents,
conditions). The trajectories already leave every live step in
`loss_dict['flow_states']` (`gflownet_losses.py:342` fwd, `:619` bwd), so
harvesting = forcing `report_losses=True` over the trailing window and
recording fields already in scope — **zero extra rollouts or energy calls**.

Larder per branch: `H = r*W` disjoint training batches + `S` held-out
scoring batches + a reserved entry-confirmation set (defaults r=3, W=10,
S in 10-20 -> ~40-60 batches), kept as an always-on ring buffer of the
trailing batches (triggers are not predictable, so harvesting cannot be
on-demand). Rows are **cloned first, then the clone is moved to host
memory** — the originals are live tensors and PyG `.to`/`.cpu` mutate in
place, and a detached clone left on the GPU would hold VRAM. `mol_batch`
included on molecule-conditioned routes. At this size the larder is
trivially small; the always-on clone/transfer cost is part of the 6.5
measurement, not assumed free.

### 2.2 The trial step, and what it evaluates

A new function, not a gated `train_step`: score the next frozen minibatch of
each branch under that branch's coefficient bank **through the
stored-trajectory evaluator** (`get_gfn_backward_loss(trajectories=...)` ->
`get_traj_replay`), compose with the pre-fork frac weights, backward, clip at
the **pinned** guard bar, step the stage's live optimizers. Trials run at the
**live warmup envelope** (not 1 — the envelope can latch below 1, and arms
must bracket the rate the run is actually at).

Isolation contract (the ray `_probe_loss` contract, extended): loss calls go
directly to the evaluators with `update_log_z=False` **and**
`mode_level_stream=None`; never through the branch step functions; no buffer
reads/writes, no `manage_replay_buffer`, no z-cal cache stash, no
`metric_tracker` writes, no EMA update, no wandb at host steps, no entries
into the batch sizer's timing deques, and the trial step must not route
through `step_loss` (its non-finite bookkeeping belongs to the parent).
`condition_log_z` is read-only inherited pre-fork state — correct, and
stated. Guard pin = **do not call `grad_guard.observe()` during trials; keep
calling `threshold()`** — a functioning clip at the frozen bar.

Two evaluator caveats, both asserted at probe entry:

- **fwd bank**: `z_level`, condition-grouped `emp_z`, `reward_grads`,
  `traj_grads`/`path_grad_last_k` have no counterpart in the stored-trajectory
  evaluator. All are 0 in the canonical conformer fwd bank; the probe asserts
  they are 0 on any probed stage (escape hatch if a route turns one on: a
  `trajectories` path in `get_gfn_forward_loss`).
- **bwd bank**: replaying a stored trajectory drops the reparameterised P_B
  path gradient. The frozen bwd trial therefore optimizes the
  **`traj_grads=0` variant** of the bwd loss — live on the canonical route,
  which sets `traj_grads: 1.0` with a learned P_B. The probe proceeds (the
  deviation is part of what 7.2 measures) but the assertion logs it; a route
  where 7.2 shows it matters either sets bwd `traj_grads: 0` for probing or
  waits for the noise-replay evaluator (appendix A). The 8.1 unit test
  asserts log_pf/log_pb **value** agreement only — it establishes nothing
  about gradients.

### 2.3 Fork, arms, replicates, restore

In-RAM snapshot before the first window; restore between windows and at the
end: model params (bitwise clone-and-finally-restore); **every optimizer that
steps this stage** ({bwd, flow} or {fused}), moments deep-copied
(`bench/runner.py::_clone_opt_state` — `state_dict()` returns live views);
guard state; `lr_ctrl` dict deep-copied plus the controller's instance
counters and the non-finite bookkeeping (`last_grad_norm_pre_clip`, streaks,
`_nonfinite_pending`); all four RNG streams, with the streams a trial should
not consume **asserted unchanged** (CUDA assert only when
`scramble_conditions` is off; when on, the scramble permutation is pinned per
probe so arms stay paired); `_hyper_prev_step` dropped on restore. The probe
owns **no checkpoint writes** (the tag namespace has no probe slot, `best` is
a hardlink to `running`, and the rewind path reads `best`/`stage_start`).

**Windows.** Every arm runs **r = 3 replicate windows** of W steps, replicate
k training on disjoint sub-larder k with preregistered order — identical
sub-larders and orders across arms, so arms are exactly paired and the
between-replicate spread is an honest variance estimate (it resamples both
data and order, which a same-larder re-run does not). Plus **one same-order
duplicate of the 1x arm**: on CPU/bench routes asserted bitwise (the restore
certificate); on GPU routes it measures the nondeterminism floor `S_nd`
(scatter-atomic reductions are not bitwise even on the conformer route).

- **Fine probe** (in-stage): {0.5, 1, 2}x, each arm replicated r=3, plus
  the duplicate -> 10 windows. Tested per section 3.
- **Entry probe** (stage entry / cold suspicion): a two-phase
  **screen-then-confirm** — rev c's two entry rules (nearest-clearing
  challenger vs top-rung-rebracket) contradicted each other whenever
  several wide arms all beat a cold incumbent; this replaces both.
  1. *Screen*: run the wide arms {0.25, 1, 4, 16, 64}x once each (r=1 —
     the screen selects, it does not test). If the best stable screen arm
     sits at either edge, shift the bracket that direction and re-screen
     (symmetric; max 2 expansions, reaching >= 1000x either way — covers
     the 800x case).
  2. *Select* the best stable screen arm as the single candidate.
  3. *Confirm*: run {incumbent, candidate} with full r=3 replication on the
     **reserved confirmation sub-larders never touched by the screen** —
     fresh, independent data, a single contrast (no multiplicity), the
     section-3 test. Move only if it passes. This is what absorbs the
     winner's curse of selecting a maximum over five noisy screen arms.

At W=10 a fine probe is ~10 windows x 10 steps x ~0.5 live-step-cost =
**~50 step-equivalents** on cheap-energy routes (~5-10 on energy-dominated
routes) — ray-class cost. An entry probe is ~11-21 windows (screen +
expansions + confirmation), still under ~110 step-equivalents. Restore-and-resume, never adopt: the winning arm's
weights are discarded; training resumes from the snapshot at the selected
rate.

**Insertion point.** The probe runs nested in one host-loop iteration, but
**after** the step-timing window closes and the sizer/throughput deques are
appended (after `train.py:~2551`), not beside `z_calibration_tick` — that
site is deliberately inside the timing window, and a probe there would feed
its wall time into the batch sizer and throughput meters.

### 2.4 Scoring

- Unit = held-out harvested batch; rows grouped by `condition_id` at scoring
  time where present (a singleton group is its own unit; on the shipped
  replay config this degenerates to the batch — fine).
- Score per contrast: paired per-unit loss difference vs the incumbent arm
  on the common held-out slice, per replicate, at matched step counts.
- **The independent unit is the REPLICATE, not the held-out batch.**
  Corrected in the build (2026-08-21) after the simulator measured a **4.2%
  false-move rate against a 1.07% nominal** when the test ran at batch
  level. Every arm in replicate `j` trains on the same sub-larder in the
  same order, so that path's luck is a single offset shared by all of that
  replicate's held-out scores: the batches carry ONE piece of evidence about
  the rate, not S. Batches still earn their keep — averaging over them makes
  each replicate's score precise — but the COUNT that enters the test is the
  number of replicates. This is `ray`'s sub-batch-overlap disease one level
  up.
- The test is an **exact binomial sign test over replicate advantages**:
  count the replicates favoring the challenger against the exact binomial
  null at the corrected, one-sided level. Distribution-free for the median;
  a sign-flip permutation test is exact only under a symmetry assumption
  nothing here validates (and a symmetric-noise simulator cannot detect its
  violation). At the defaults (r=10, m=2 challengers, alpha/2 = 0.025):
  **>= 9 of 10 replicates**, realised false-move rate 1.07% per challenger.
  Computed, never transcribed — `bench/test_lr_race.py` derives every
  critical value from `math.comb`.
- `replicates` is therefore **the sample size, not a cost knob**: at r=3
  even a clean sweep is p = 1/8 and no verdict can reach any level, so the
  rule would be structurally unable to decide anything. Screens use r=2
  (selection only, no critical value) — measured, r=1 misses an 8x-hot
  incumbent 4.0% of the time versus 0.5% at r=2.
- Half-window check: the advantage must be positive in both halves of the
  window, **pooled across replicates**. Per-replicate is a conjunction of 2r
  noisy conditions and was measured rejecting 2 of every 3 genuine wins
  (33.5% pass on a signal the sign test resolved 87% of the time) — a guard
  that rejects the mission instead of the failure mode.
- Any nonfinite/exploding window disqualifies that arm for this probe.
- Any nonfinite/exploding window disqualifies that arm for this probe,
  logged as `died k/n`. Vetoes gate **actuation, never data**: every probe's
  full per-arm, per-unit, per-replicate record is logged regardless of
  outcome (this is what makes offline pooling possible later).

Loss values need be comparable only within a probe.

## 3. The decision rule

**Estimand, stated once:** the probe estimates
`x*(theta_t, larder_t; W) = argmin over the arm lattice of E[held-out loss
after W frozen steps at rate x]` — a W-step frozen-surrogate quantity,
conditional on the current parameters and harvest. Its offset from the
live-optimal rate is unknown in sign and size a priori; 6.2/6.3 measure it
and require it to sit **within one rung** at the shipped W. No numeric
correction is derived from it (see rule 4).

`decide(probe_record) -> {hold | move(arm) | rebracket | hold-invalid}` — a
pure function, no trainer dependency.

1. **Validity**: RNG assertions clean; duplicate-null clean (bitwise on CPU;
   on GPU, `S_nd < 0.5x` the winning contrast); half-window checks pass.
   Otherwise hold, reason logged.
2. **Incumbent is the null.** A challenger moves the rate only if it clears
   the exact binomial sign test of 2.4 at the **Bonferroni-corrected,
   one-sided** level (fine probe: m=2; entry confirmation: a single
   pre-selected contrast, uncorrected) **and** favors the challenger in all
   r replicates. Symmetric for raises and cuts.
3. **Tie-break — fine probes only**: among multiple clearing challengers,
   the arm **nearest the incumbent** that clears (pre-declared; never
   max-|t|, which selects outward-biased noise). Conservatism is right for
   routine in-stage moves. Entry probes never face this choice: the screen
   selects one candidate and only that candidate is tested (2.3).
4. **Moves are discrete**: to the winning arm's rate, exactly. No fits, no
   extrapolation, and **no bias divisor** — dividing a discrete winning arm
   by a correction factor and re-snapping to the lattice is incoherent (a
   systematic 2x correction turns every 2x win into a hold and freezes
   adaptation). If the surrogate is offset by more than a rung, the fix is
   a larger W or an unsupported route (6.2/6.3), never per-move arithmetic.
5. **Cold acceleration without a ladder**: two consecutive confirmed
   same-direction moves -> the next probe is an entry probe. No escalation
   state, no step-size schedule.
6. **Inconclusive means hold. Always.**

Resolution is honest: half the lattice spacing, i.e. **+-0.15 dex (~+-40%)**.
That is the deliverable — a coarse, unbiased-after-calibration anchor on
every stage. Finer resolution is a v2.1 question (appendix A).

## 4. Triggers and cadence

- **Phase entry**: gentle warmup ramp (existing `rearm_warmup` semantics;
  note it resets `peak_scale` to 1.0, so nothing survives the boundary by
  design). The ramp window doubles as the harvest window. `protocol.advance`
  only **arms** a pending entry probe; the step loop **fires** it when
  `controller._ramping(st)` first reads false (covers ramp-complete and
  ramp-frozen alike — there is no ramp-completion event to hook). Entry
  probes use the entry bracket.
- **Composition trigger**: fire a probe when the normalized frac/coeff
  vector has moved L1 >= 0.2 since the last probe (branch dormancy flips and
  coefficient ramps are special cases of the same distance).
- **Geometric clock**: stage-relative ~0.5k, 1.5k, 3.5k, 7.5k, 15k, 30k —
  dense across the fast-equilibration window, sparse after. At rev-c probe
  cost this cadence is ~1% overhead; still subject to the section-9 gate.
- Between probes the LR is exactly constant. Nothing else actuates.

## 5. Emergency path

Unchanged in architecture: extend `fire_loss_spike` +
`_rewind_checkpoint_path` (rewind + rate budget + abort, battle-tested).
Trial-arm failures never count against the parent's reload budget.

Tripwire workstream (independent, parallel; item 1 gates test 7.3c):

1. Add a per-branch **unwinsorized tb_err statistic** to the spike check.
   Corrected rationale: the returned loss is Huber-**linearized**, not
   bounded — past the knee it grows in proportion to tb_err — so a relative
   bar on it is in principle capable; the absolute 1e9 bars are simply ~7
   orders too high. Run the `mvwsu5d5` retrospective (would
   `divergence_loss_rel` at 100x have tripped?) **without prejudging it**;
   tb_err remains the cleaner signal either way (it survives the fused
   weighted sum).
2. Run the spike check every step (currently 1-in-10).
3. Move the clip-saturation detector out of the `hyper`-only path.
4. Arm the relative bar across transitions (inherit a widened reference;
   today it is inert for the first 500 steps of every stage).

## 6. Validation gates (numeric, all pre-actuation)

**6.1 Decision-function simulation (bench/, no GPU).** The simulator models
the **measurement, not the summary**: three variance components (unit,
order, larder), heavy-tailed AND **skewed** unit differences (t_5 plus
asymmetric mixtures — a symmetric-only noise model cannot detect the
symmetry-assumption failures that disqualified the sign-flip test),
loss-scale drift across probes, side-specific curvature with C+/C- in
{1, 3, 10}, censoring at a declared rate — with the test statistic computed
exactly as the shipped code computes it. Assertions:

- Drift: TOST — the 95% CI on per-probe drift lies within **+-0.005
  dex/probe** on a flat optimum (point-null "zero drift" is untestable).
- Stationary behavior: |E[offset]| <= 0.15 dex, SD <= 0.30 dex,
  P(|offset| > 0.6 dex) <= 1%.
- Centering under asymmetry: the offset bound holds at C+/C- = 10 (this is
  what replaced the margin).
- Escapes: 8x cold or hot corrected within <= 2 probes; 800x cold within
  one entry event (screen + <= 2 expansions + confirmation).
- Post-move hysteresis: after a forced wrong move, median time-to-correct
  <= 2 probes.
- Broken pairing (null trips) -> holds.
- **The v1 rule ("highest statistically competitive"), in the same harness,
  must FAIL the drift bound** — the re-introduce-the-bug discipline.
- Positive controls: under a skewed null the shipped sign test's realized
  level must stay at nominal (and the sign-flip variant, run as a control,
  should show the violation that got it rejected); a deliberately corrupted
  optimizer-moment restore must trip the duplicate null.

**6.2a THE NUMBER THE SWEEP MUST DELIVER (from the built simulator).** All
of the rule's power reduces to one quantity the simulator cannot know:

    SNR = mean(replicate advantage) / sd(replicate advantage)   for one rung

Measured power at r=10, 9-of-10: SNR 0.0 -> 1.1% (the false-move rate),
0.5 -> 14%, 0.8 -> 34%, 1.2 -> 68%, 2.0 -> 98%. **A contrast the probe must
resolve needs SNR >= ~1.2.** Signal grows with the window length while the
replicate-level noise does not, so W is the lever. Print the table with
`python -m bench.race_sim`; the W-sweep's job is to find the smallest W that
clears it on the real surface, and to report the measured SNR per route.

**6.2 W-sweep.** Run the full probe at W in **{5, 10, 30, 100}** (the
default W must be in the sweep), >= 3 preregistered replicate sets, at a
**clean pinned-rate reference**, and **report the measured one-rung SNR at
each W** alongside the selected rate. The current
`conformer_ring_mle_fixedlr.yaml` is NOT that reference: it resumes a
step-25k checkpoint with `override_learning_rates: false`, so the
checkpoint's optimizer LRs — not the configured 1e-4 — are what actually
runs. Build the reference as a controlled fork of the warm-start checkpoint
with `override_learning_rates: true`, **verify the logged live rates**
(live-vs-set is a known trap in this trainer), and name the
internal-coordinate tier (`full`). Read off the offset
`selected rate / pinned rate` as a function of W. Ship rule: the shipped W
is the smallest W at which the offset is within one rung and stable across
replicate sets, **and** 6.3 passes. No divisor is installed at any W (3.4).
If no W satisfies this inside the cost gate, the route is unsupported —
probes hold — until v2.1.

**6.3 Surrogate rank fidelity.** At >= 3 representative checkpoints
(phase 1, early phase 2, late phase 2), frozen probe vs a small on-policy
forked search (real forks, multiple seeds). Bar: the frozen probe's winner
is **within one rung** of the on-policy winner at every checkpoint.

**6.4 Recovery tests.** Seeded 6x cold -> corrected by the entry event.
Seeded 4x hot -> corrected through probes. Seeded 12x hot -> corrected
through tripwire + rollback (the acceptance test of 5.1).

**6.5 Cost gate.** Measured per route at first integration; amortized
overhead <= 2%, counting **everything**: trial windows, scoring passes, AND
the always-on harvest (forcing `report_losses=True` in the ring-buffer
window, per-batch clones, host transfer) — not only the probe events.
Predicted: fine probe ~50 step-equivalents on elj-class routes (7 probes
~ 350/40k ~ 0.9%), ~5-10 on energy-dominated routes; the harvest overhead
is the unmeasured term the gate exists to catch.

**6.6 Battery discipline.** Null arm always; sort by worst cell; report
died-in-k/n; no means over survivors; conformer results name the
internal-coordinate tier.

## 6.7 Build status (2026-08-21)

**Workstream 3 (decision layer + simulator) is BUILT and its gate PASSES.**
`lr_race.py` (pure logic, no torch) + `bench/race_sim.py` (measurement-level
simulator) + `bench/test_lr_race.py` (46 tests, fast tier, ~14 s). Report:
`python -m bench.race_sim`.

Gate 6.1 results, 3000 simulated probes per cell: drift CI inside +-0.005
dex/probe on every cell including pure-noise, plateau, skewed, LR-dependent
skew, scale drift, hazard and sprinter; stationary offset |E| <= 0.05 dex and
sd <= 0.25 on every curved cell; centering holds at C+/C- = 10 (offset -0.020),
which is the evidence the rejected margin is not needed. Escapes: 8x cold and
8x hot corrected by one entry event, 800x cold to +0.107 in one event. The
rejected v1 rule, in the same harness, parks at **+0.55 dex** and fails the
offset bar — and note it PASSES the drift bar, which is why 6.1 needs both.

Three defects the simulator caught in the design as specified, all fixed:
batch-level testing ran 4.2% false moves against 1.07% nominal (the unit is
the replicate); the per-replicate half-window conjunction rejected 2 of 3
genuine wins; an r=1 screen misses 4% of hot entries.

Honest resolution, measured: at the simulator's SNR the rule resolves ~2-3
rung errors decisively and one-rung errors barely (3% per probe), so it
settles with a **~1 rung residual** and rarely moves once settled. That is
consistent with the design's premise — within a stage the optimum is
stationary and a mostly-holding rule is correct — but it means the entry/wide
race, not the fine race, is what actually sets the rate.

## 7. Implementation plan — five workstreams

1. **Harvest + stored-trajectory scoring**: `report_losses=True` over the
   trailing window; larder assembly (clone, don't move); the two bank
   assertions of 2.2; the log_pf/log_pb value-equivalence unit test
   (fwd/bwd roundtrip tests already exist in `test_periodic_scoring.py` —
   extend, don't rewrite).
2. **Trial step + fork/restore**: the isolation contract, guard pin, the 2.3
   snapshot manifest, RNG assertions, arm-scope assertion via the existing
   `_lr_capped_groups`/`_lr_floored_groups` counters (an arm they touch is
   disqualified).
3. **`decide()` + simulation suite in bench/** (pure logic, parallel to 1-2;
   gate 6.1 passes before integration).
4. **Runner integration + triggers**: insertion after the timing window
   (2.3); armed-entry-probe plumbing (4); composition distance; geometric
   clock; ray actuation removal + registry disable fix (ray's
   ramp-freeze-on-cold-reading safety job is taken over by the entry probe).
5. **Validation battery** (6.2-6.6) on the conformer route; tripwire
   workstream (5) proceeds in parallel.

Config: `lr_probe: {enabled: false, window: 10}`. Everything else is a
module constant (brackets, r=3, S, thresholds, trigger distances, clock),
frozen by 6.1/6.2 — a knob that can express a disqualified configuration is
a liability, not flexibility.

## 8. Deliberate limitations

- **The probe is a surrogate**: real multi-step optimization on a frozen
  empirical distribution, blind to on-policy feedback, with a bias whose
  direction and size are route- and regime-dependent and unknown until
  measured (6.2/6.3). The bias is bounded to within a rung by choosing W —
  never corrected arithmetically per move.
- **Resolution is +-half a rung (~40%)**. Good enough to kill 5-10x
  cold-running with a decade of margin; not a precision instrument. Raw
  probe records are logged completely, so precision upgrades (offline
  pooling) need no new data.
- The frozen bwd loss omits one gradient channel on routes with a learned
  P_B (2.2); 6.2/6.3 price it, and appendix A names the fix if it matters.
- Stages that cannot be paired even on stored data are not probed; they
  hold their seeded rate.

---

## Appendix A — v2.1 candidates (each returns only with its own 6.1 demonstration)

- **Pooled-evidence actuation.** Offline pooling of the logged records
  (keyed by ABSOLUTE log-LR, standardized effects, random-effects pool) may
  later justify moves finer than a rung. Deferred because the rev-b design
  was incoherent twice (incumbent-relative key = hysteresis; raw-nats IVW =
  scale-incoherent) and its precision serves the band where holding is
  already correct.
- **The hyper change detector.** Requires four sensor fixes first (publish
  the actuated quantity; un-dilute the mean of clip-saturated firings; key
  the operand by branch; clear on rewind) plus baseline re-arm on every LR
  change. Composition trigger + clock bound detection latency meanwhile.
- **Margin / deadband.** If 6.1 shows the plain rule mis-centers under
  asymmetric curvature: a deadband symmetric in log-LR (move only when the
  confidence-bounded implied error exceeds half a rung) — never a fixed
  nats margin (centers cold), never kappa x single-draw null scale.
- **Permuted-order null.** Superseded by replication across disjoint
  sub-larders; could return as a scheduled diagnostic.
- **Instability ceiling.** Deleted: within-probe disqualification plus
  `fire_loss_spike` cover safety; the ceiling was a cost optimization with
  one-way-ratchet risk.
- **Noise-replay bwd evaluator** (harvest per-step eps/u_lift, re-derive
  through `get_traj_bwd`): restores the P_B path gradient to frozen trials;
  a real new evaluator, built only if 6.2/6.3 show the omission matters.
- **Ray as a periodic audit.** Its measurement path is sound; re-admit only
  with a cost budget (its probe is ~40 full-batch forwards per calibration —
  route-dependent, not universally "1.2%").
