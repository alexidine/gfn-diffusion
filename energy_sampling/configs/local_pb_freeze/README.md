# Should P_B be frozen after phase 1, and what is P_B doing in phase 2?

Analysis 2026-09-07, branch `conditional`. Files beside this note: `analyze_cluster.py`
(sections e: `e1_*`, `e2_*`, `e3_*`, `e4_*` PNG/CSV, read off wandb `mkilgour/GFN Energy`
tags `pt100` and `p02`), `probe_pb.py` (P_B scored on stored replay paths), `make.py` and
the three `pbab_*.yaml` arms (section g, wandb tag `pbab`), `ab_curves.png` and
`ab_summary.csv` (section g).

Code that came with it (working tree):

- `Modeller.set_pb_freeze(mode)` (`train.py`) with `GFN.freeze_backward_policy` /
  `_pb_net` (`models/gfn.py`): `None` trainable, `'head'` the cheap (and wrong) freeze,
  `'full'` P_B on a snapshot of `t_model`+`s_model`+`backward_policy`, one object shared
  by the train and EMA models. Reached three ways: the top-level config key
  `freeze_backward_policy` (from step 0 of the process), and the stage actions
  `freeze_pb[:full|head]` / `unfreeze_pb` (`protocol.py`), which is the intended use:
  `on_enter: [rebuild_prior_by_churn, freeze_pb]` on the stage after the MLE warm-start.
- The snapshot is PERSISTED: `checkpointing.py` writes it under `pb_frozen` and every
  load path restores it through `set_pb_freeze('full', source_state=...)`, so a resumed
  leg scores P_B on the same function instead of re-snapshotting a trunk that has
  drifted under P_F. A trainable checkpoint (`pb_frozen: None`) lifts a freeze on
  reload, so a rewind to a pre-freeze checkpoint does not keep a later snapshot.
  `set_pb_freeze('full')` on a model that already carries a restored snapshot keeps it.
  Proof: `make_resume.py` (`pbab_resume_a` freezes at the boundary, `pbab_resume_b`
  resumes it; the snapshot tensors are compared bitwise, section 6c).
- A per-submodel split of the per-branch gradient norms inside the existing
  `grad_geometry` diagnostic (`fused_grad/{branch}_norm_{submodel}`).
- Tests: `tests/models/test_pb_freeze.py`, `tests/protocol/test_pb_freeze_action.py`.

## 0. The short answer

RECOMMENDATION and the numbers behind it are in section 7. Everything before it is the
argument and the measurements.

## 1. What P_B actually is (read off the code)

Forward kernel (`_forward_kernel`, `fwd_propagate`, `fwd_gauss_logprob`):

    x_{i+1} = x_i + dt * mu_theta(x_i, t_i) + sqrt(dt) * C_theta(x_i, t_i)^{1/2} eps

with a free drift `mu` and a DPLR covariance `C = diag(d) + V V^T` (rank 6), both from
`forward_policy` on `s_model(x_i)` and `t_model(t_i)`. Angular dims wrap.

Reference backward kernel, the thing the learned correction modifies (`_eval_pb_logprob`,
`_bwd_step`, `var_drift_coeff`, `var_bridge_step`): with constant noise rate,
`c_i = dt / t_{i+1}` and

    mean_B(x_i | x_{i+1}) = x_{i+1} * (1 - c_i * kappa)          kappa = 1 -> x_{i+1} * t_i / t_{i+1}
    var_B (x_i | x_{i+1}) = t_scale * exp(delta) * dt * t_i / t_{i+1}

At `kappa = 1, delta = 0` this is exactly the Brownian bridge of the reference process
`x_t = sqrt(t_scale) W_t` pinned at `x_0 = 0` (the "OU-type" pull toward the origin is the
bridge contraction `1/t`, not a stationary OU). The first backward step (i = 0) is
deterministic to the origin and scores 0. On angular dims `pb_exact_reversal` makes it the
exact reversal of the wrapped bridge (a mixture over arrival lifts), with `kappa` applied to
the contraction of each component.

The learned part (`fwd_get_back_correction` / `get_bwd_correction`): `backward_policy` is a
4x512 MLP head on `s_model(x_{i+1})` and `t_model(t_{i+1})` -- the SAME trunk P_F uses --
returning per-coordinate

    kappa = 1 + 0.4 * tanh(dmean / 0.4)   in (0.6, 1.4)      pb_drift_range 0.4
    delta = 6   * tanh(dvar  / 6)         in (-6, 6)         pb_var_range 6  -> variance x e^{+-6}

So P_B's freedom over the fixed bridge is: a state- and time-dependent rescaling of the
contraction rate toward the origin by up to +-40%, and a state- and time-dependent
rescaling of the diagonal variance by up to 400x either way. It CANNOT move the backward
mean off the ray from `x_{i+1}` to the origin, cannot couple coordinates (P_F has rank-6
coupling, P_B is diagonal), and cannot change the pin. `learn_pb` is construction-time
only; there was no runtime freeze before this note.

Optimizer (`train.py` `get_policy_params`): `t_model`, `s_model`, `forward_policy`,
`backward_policy` are four groups of one fused Adam, all at the same controlled LR. Note
the consequence for "freeze P_B by zeroing its group": that freezes the HEAD only. P_B's
inputs come through `s_model`/`t_model`, which P_F keeps training, so P_B keeps moving,
and P_B's loss terms keep pushing the trunk through the frozen head. Section 6 measures
the size of that leak; the `full` freeze mode evaluates P_B on a snapshot of all three.

## 2. First principles

### (a) TB's solution set with P_B fixed vs learned

TB asks `log Z + log P_F(tau) = log R(x_T) + log P_B(tau | x_T)` for every trajectory.
Summing over paths into each terminal gives the terminal-marginal condition
`P_F^T(x) = R(x)/Z`; the per-path form additionally pins the PATH law.

**Fixed P_B.** The joint `Q(tau) = P_B(tau | x_T) p*(x_T)` is a single, fully specified
distribution over trajectories, and it is Markov (P_B is a Markov kernel, `p*` is a
terminal density). TB is satisfied iff `P_F = Q` path by path, so the forward kernel is
UNIQUE: `P_F*(x_{i+1} | x_i) = Q(x_{i+1} | x_i)`. That kernel is generally NOT in the
Gaussian-DPLR family P_F is parameterised in: `Q(x_{i+1}|x_i) ∝ P_B(x_i|x_{i+1}) m_{i+1}(x_{i+1})`
where `m_{i+1}` is the bridge-smoothed target marginal at time i+1, and the conditional is
Gaussian only to the extent that `log m_{i+1}` is quadratic over the width of the step
kernel. What training then finds is the TB-objective projection of `Q` onto the family;
the terminal marginal is matched only up to that projection error. With P_B fixed there
is nothing else to trade against: the residual left over is the representability gap.

**Learned P_B.** The solution set becomes `{(P_F, P_B): P_F(tau) = P_B(tau|x_T) p*(x_T)}`
for every admissible P_B in the kappa/delta box: one P_F for every P_B, a family
parameterised by P_B. All members have the same terminal marginal `p*` and differ in the
path law. TB itself has NO preference among them -- the objective is exactly zero on
every member -- so the member reached is set by the initialisation (phase 1) and by the
dynamics of joint descent, not by the loss. The reason to want the freedom is purely
representational: the member picked should be one whose required forward kernel lies
inside the Gaussian-DPLR family, and a learned P_B can bend the path law (per-coordinate
contraction rate and per-coordinate variance, i.e. the shape of the annealing path from
`p*` to the origin) to make that so. A learned P_B also adds a second, T-independent role
that TB does not price but variance does: as the posterior `P_F(tau | x_T)`, it minimises
`Var[log P_F(tau) - log P_B(tau|x_T)]` within a terminal, which is what makes the
per-trajectory TB residual approximately a per-terminal quantity and the IS log Z tight.
That is exactly what phase 1 trains (section 3).

**Uniqueness in one line.** Fixed P_B: TB has a unique forward solution, outside the
model class in general. Learned P_B: a manifold of exact solutions indexed by P_B, with
the parameterisation (not the loss) deciding which is reachable.

### (b) What a flexible P_B buys at T = 100

The short-rollout argument is about the per-step kernel width. The reference step
variance is `t_scale * dt = t_scale / T`: per-coordinate step sd 0.071 at T = 10 and 0.022
at T = 100 (t_scale 0.05), against a terminal spread of `sqrt(t_scale)` = 0.22. One step
covers 32% of the terminal spread at T = 10 and 10% at T = 100. The non-Gaussianity of the
required forward conditional `Q(x_{i+1}|x_i)` is set by how far `log m_{i+1}` departs from
quadratic across that width. With a learnable per-step Gaussian (mean AND covariance --
DPLR here), the linear and quadratic terms are absorbed exactly; the leading error is
third-order, `O((sigma_step^3 * d^3 log m)^2)` per step, `sigma_step^6 ∝ T^-3` per step and
`∝ T^-2` summed over T steps. Even the cruder bound that counts the quadratic term as
unrepresentable (rank-6 + diagonal cannot hold an arbitrary 12x12 curvature) goes as
`T * sigma_step^4 ∝ T^-1`. Either way the representability purchase of a flexible P_B
falls by at least an order of magnitude from T = 10 to T = 100, and it was at T = 10 (and
below) that the "converge with much shorter rollouts" experience was formed.

Where it does NOT fall: hard walls. Near a steric clash the curvature of the (clipped,
`reward_range` 250) energy is far larger than `1/sigma_step^2` at any T in use, so the
required conditional there is non-Gaussian at every T. But P_B's correction cannot fix
that either -- it rescales the contraction toward the origin and the diagonal variance,
which is a change to the annealing path, not a wall-shaped kernel. The wall problem is
P_F's (and the Huber knee's), at any T, frozen or not.

What I could NOT quantify: the size of the phase-1 P_B departure at T = 100. Section 4
measures it at T = 10 on the local rig (25 nats per path above the pure bridge). No T = 100
phase-1 checkpoint is on this machine (the pt100 exits live on the cluster), so whether
that number is 25 or 2.5 at T = 100 is open. The scaling argument says it should be much
smaller in the part that matters for representability; the posterior-variance role is
T-independent.

### (c) The replay degeneracy

Replay scores a STORED path `tau_j` with residual
`r_j = log P_F(tau_j) - log P_B(tau_j|x_j) - log R(x_j) + log Z`. Its gradient on P_B's
parameters `phi` is `-r_j * d log P_B(tau_j|x_j)/d phi` (Huber-winsorised at beta 80): every
stored path with `r_j > 0` is made MORE probable under P_B, every path with `r_j < 0` less
probable, whether or not P_F has moved. This is a plain regression gradient because the
path is fixed; it is not the score term that averages to zero on bwd (the level_gap
comment in `get_gfn_backward_loss` is right for a coefficient-weighted `log P_B` under
`tau ~ P_B`; it does not apply to a squared residual on a fixed path). There is no
detach on P_B in `_replay_step`, and replay's `freeze_policy` is 0.

Can it lower the replay residual without P_F improving as a sampler? Capacity-wise, yes,
by a wide margin: `delta` alone allows +-6 nats per coordinate per step, i.e. hundreds of
nats per path at T = 10, and `kappa`/`delta` are functions of `(x_{i+1}, t)` through the
2.2 M-parameter head plus the shared trunk, so per-path corrections are representable. The
box is not the constraint. What constrains it is the bwd branch, which every step pulls
P_B toward being the posterior of P_F on FRESH prior-drawn paths (fixed-terminal Huber TB
through the reparameterised backward rollout); a P_B that "explains" stored paths is a
worse posterior on fresh ones, and both branches carry the same weight (0.475) on the same
network. So the direction is taken to the extent that replay's push on high-|r| rows
(prioritised draw, kappa 1, symmetric) outweighs bwd's restoring pull on the same regions
of state space. The measurable signature, if it were taken, is `replay/logpb_mean` rising
ABOVE `fwd/logpb_mean` (stored paths becoming more probable under P_B than fresh ones).

Measured on every p02 arm (section 5, `e2_*.csv`): `replay/logpb - fwd/logpb` is NEGATIVE
everywhere (-4 to -91 nats), and `replay/logpf - fwd/logpf` is more negative still (-8 to
-112). Stored paths are LESS probable than fresh ones under both policies, more so under
P_F -- the staleness signature of a policy moving away from its own ~50-step-old samples,
not P_B explaining the buffer. So on the production battery the degenerate direction is
present in the gradient but is not the direction the parameters take.

Connection to the sensor: `replay/resid_vs_intake` is `mean(ema_loss)/mean(birth_loss)`
over resident rows. A row corrected by P_B moving and a row corrected by P_F moving read
identically. The sensor therefore bounds the SUM of both routes; the per-branch
per-submodel gradient norms in section 6 are what split them.

### (d) Needed, or merely tolerated, at T = 100?

Freezing at the phase-1 exit fixes the path law to phase-1's P_B, which was trained as
the posterior of the phase-1 P_F -- a P_F fit by MLE to the PRIOR dataset's terminals. In
phase 2 P_F moves from that toward the Boltzmann target; the posterior of the moved P_F is
a different kernel, so a frozen P_B is a posterior of the wrong policy. What that costs,
by (a) and (b): (i) representability -- at T = 100, small by the scaling argument, and
only if the required forward kernel under phase-1's P_B leaves the DPLR family, which the
learned P_B could not have helped with at a wall anyway; (ii) variance -- `logw_std_within`
grows as P_F departs from the policy P_B is the posterior of, and with it the TB gradient
noise and the IS log Z spread. What it buys: TB's solution becomes unique; the replay
degeneracy of (c) closes; the P_B share of the fused gradient (10-40% of P_F's, section 5)
is removed from the update; and P_F converges onto a stationary target instead of chasing
a P_B that is itself chasing P_F.

So the a-priori reading is: at T = 100 the forward policy TOLERATES a learned P_B (it is
paying a moving target and a gradient share for a representability margin it no longer
needs much of) and NEEDS it only through the variance role -- which is the thing a frozen
posterior loses gradually as P_F moves. That is a rate question, not a structural one, and
the A/B in section 6 measures the rate at T = 10 where every effect is larger than it
would be at T = 100.

## 3. What phase 1 trains, on what data (e, protocol)

`prod_eq` / `unconditional_tb` stage `train_prior`: `train_mode: bwd`,
`bwd_sampling_mode: dataset`, loss `mle: 1.0`, `repeats: 1`, `traj_grads: 1`. Per step,
`draw_bwd_sample` takes prior-DATASET terminals (unweighted), `get_traj_bwd` rolls P_B
backward from them (reparameterised, no detach), and `terminal_mle` with `repeats 1` uses
the `bound` estimator: loss `= -(log P_F(tau) - log P_B(tau|x))` averaged over
`tau ~ P_B(.|x)`, i.e. the negative ELBO on `log p_F(x)`. Its gradient reaches P_F
explicitly (teacher-forced `log P_F` on P_B's paths -- raise P_F on prior terminals along
P_B's paths) and P_B through the reparameterised path (tighten the bound: make P_B the
posterior `P_F(tau|x)`). No reward, no Z, no replay. Exit `gates/progress_done` (or, on
`prod_eq`, immediately), `snapshot:phase1_exit`, `snapshot_prior`.

So at the phase-1 exit P_B is the variational posterior of a P_F that is itself a density
model of the prior dataset. Measured departure from the pure bridge on the local T = 10 rig
(`probe_pb.py`, 1000 stored replay paths, 12-d): median `|kappa - 1|` per step 0.17-0.31
against the 0.4 cap, median `|delta|` 0.44-1.9 (largest on the last two steps), and
`log P_B(learned) - log P_B(bridge)` = +24.9 nats per path (sd 11.7). Phase 1 does a lot
to P_B; it is not near the identity.

## 4. Phase 1 -> phase 2 on the cluster (e, measured)

`e1_phase1_pb_gradnorm.png` / `.csv` (pt100 sources, all `train_prior` only at T = 100):
the P_B head's share of the MLE gradient falls through phase 1 on every family, from a
ratio `gradnorm/backward_policy : gradnorm/forward_policy` of 0.8-0.97 in the first 10%
to 0.21-0.39 in the last 10% (mip 0.78 -> 0.29, neh 0.93 -> 0.39, mipu 0.87 -> 0.28, nehu
0.97 -> 0.26, acr 0.45 -> 0.21). P_B's gradient is the ELBO-tightening term; it decays as
the bound tightens while P_F's stays up. P_B is still moving at exit on every family.

`e4_boundary_pb_ratio.png` (phase-1 tail -> phase-2 head, per family) and
`e2_phase2_pb_gradnorm.png` / `.csv` (p02 equilibration, fixed fracs 0.05/0.475/0.475,
fused loss so bwd + replay mixed):

| family | scale | P_B/P_F ratio first 20% | last 20% | gradnorm/backward_policy last 20% | replay-fwd logpb | memo |
|---|---|---|---|---|---|---|
| mip ELJ | 1.0 | 0.36 | 0.39 | 17.5 | -4.4 | 0.475 |
| neh ELJ | 0.5 | 0.39 | 0.38 | 19.0 | -16.3 | 0.546 |
| mipu UMA | 0.031 | 0.49 | 0.15 | 196 | -50 | 0.832 |
| mipu UMA | 0.062 | 0.22 | 0.11 | 220 | -18 | 0.750 |
| mipu UMA | 0.125 | 0.70 | 0.16 | 260 | -51 | 0.722 |
| mipu UMA | 0.25 | 0.55 | 0.24 | 369 | -91 | 0.727 |
| nehu UMA | 0.016 | 0.58 | 0.16 | 180 | +3.4 | 0.919 |
| nehu UMA | 0.031 | 0.23 | 0.13 | 203 | -13 | 0.858 |
| nehu UMA | 0.062 | 0.20 | 0.12 | 256 | -11 | 0.823 |
| nehu UMA | 0.125 | 0.18 | 0.17 | 236 | -21 | 0.751 |
| nehu UMA | 0.25 | 0.21 | 0.21 | 263 | -37 | 0.713 |
| acr MACE | 0.0125 | 0.26 | 0.23 | 57 | -5.4 | 0.943 |
| acr MACE | 0.05 | 0.21 | 0.17 | 74 | -8.3 | 0.850 |
| acr MACE | 0.1 | 0.22 | 0.19 | 81 | -12 | 0.801 |

Three readings:

1. **The transition re-fits P_B.** On the UMA arms the P_B share spikes to 0.5-1.0 in the
   first ~1000 steps of equilibration and then decays to 0.11-0.24; on acr it spikes to
   0.55 and settles at ~0.2; on the ELJ arms it steps from 0.29/0.39 (MLE) to a flat
   0.36-0.39 and stays there for 48k steps (mip). The loss changed (ELBO -> Huber TB
   against a reward, with fresh Adam), so P_B is re-solved for the new objective right at
   the boundary. This is where a frozen P_B would differ most from a trainable one.
2. **P_B keeps moving in phase 2 at a steady share.** After the spike, the head's
   gradient is 10-25% of P_F's on UMA/MACE and ~38% on ELJ, flat to slowly falling, for the
   whole run. Under Adam the step size is ~LR regardless of norm, so "how much P_B moves"
   is LR x steps whatever the ratio says; what the ratio says is that the P_B gradient is
   a persistent component of the update, not a transient.
3. **LR dependence.** The absolute `gradnorm/backward_policy` rises with scale within a
   family (mipu 196 -> 369 over 8x LR; nehu 180 -> 263 over 16x), i.e. a hotter run keeps
   a larger residual gradient on P_B, but the P_B/P_F ratio is not monotone in LR
   (`e3_pb_share_vs_lr.png`). Within a family the steady-state SHARE is LR-independent to
   within noise; the level of both gradients rises together with LR.

The bwd-branch `logpb` gap (`bwd/logpb - fwd/logpb`, `.csv`) is small everywhere (-13 to
+4), so P_B scores its own fresh backward paths and P_F's fresh forward paths alike; the
big negative replay gap is buffer staleness, per (c).

## 5. bwd vs replay as the source of P_B's phase-2 gradient (f)

Measured on the trainable local arm (`pbab_train`, `f_branch_split.png`, medians over
equilibration, raw per-branch norms before frac weighting so the two branches are directly
comparable):

| submodel | bwd branch | replay branch | replay / (bwd + replay) |
|---|---|---|---|
| backward_policy | 594 | 442 | 0.43 |
| forward_policy | 4031 | 3419 | 0.46 |
| s_model (shared trunk) | 3376 | 2956 | 0.47 |

Replay supplies ~43% of the gradient norm on the P_B head, and P_B's share of each
branch's own gradient is the same on both (head/forward_policy: bwd 0.15, replay 0.13).
So replay is not disproportionately a P_B objective, and bwd is not disproportionately
P_F's; the two branches move the same three networks in the same proportions. What
differs is DIRECTION: the whole-model cosine between the bwd and replay gradients is
-0.45 to -0.60 throughout, on the trainable AND on the fully frozen arm. The two branches
are fighting, and since the frozen arm has no P_B gradient at all, the fight is P_F-side
(fresh prior-terminal paths vs stale high-residual stored paths), not a P_B artefact.

## 6. The local A/B (g)

`make.py` -> `pbab_train` (P_B trainable), `pbab_frozen` (P_B on a snapshot of trunk +
head), `pbab_headfrozen` (requires_grad off on the head only). lp02 rig: ELJ mipcas sg2,
T = 10, batch 400, fixed fracs 0.05/0.475/0.475, replay beta 80, fixed scale 0.125
(lr 1.56e-5), warm start from the dev_elj_p2_cruise phase-1 exit, seed 12345, 3200 steps
(equilibration from ~step 200), grad_geometry every 20. `ab_curves.png`, `ab_summary.csv`
(medians over the last 25% of equilibration). Check that the freeze took:
`gradnorm/backward_policy` is 0.000 on the frozen arm, 271 on the trainable one.

Three logged pairs are numerically the same channel on this unconditional route and are
read as one: `fwd/tb_err` = `fwd/tb_err_worst`, `fwd/logw_std_within` = `fwd/scatter_err`,
`bwd/tb_err` = `bwd/under_coverage`.

| channel (last 25%) | trainable | full freeze | read |
|---|---|---|---|
| fwd/tb_err | 51.8 | 52.5 | same within either arm's own swing (sd 1.2 / 0.9) |
| fwd/over_coverage | 50.8 | 51.5 | same |
| fwd/logw_std_within | 48.8 | 49.5 | same: NO variance penalty from the frozen posterior in 3000 steps |
| bwd/under_coverage | 33.5 | 33.2 | same, frozen marginally lower |
| bwd/logw_std_within | 12.26 | 12.28 | identical, both drifting up |
| replay/tb_err | 89.0 | 91.0 | trainable LOWER: the residual P_B absorbs on stored paths |
| replay/resid_vs_intake | 0.959 | 0.961 | identical: the sensor is P_F-dominated |
| fwd/jensen_z (mean log w) | -7.30 | -7.31 | identical |
| log Z learned | 9.82 | 10.29 | frozen climbs faster and higher; both still rising |
| fwd/logr_mean (train batch) | 3.80 | 4.42 | frozen higher reward on its own samples |
| Mean Sample Energy (eval, 500) | -10.5 | -6.6 | too noisy to read: swings -22..+10 within one arm |
| gradnorm/forward_policy | 1739 | 1940 | frozen ~10% larger P_F gradient |
| cos(bwd, replay) | -0.48 | -0.46 | the branch conflict is present in both |

**The decisive difference is not in any level; it is in the DYNAMICS.** The trainable arm
runs a clean limit cycle of period 600 steps (ACF peak r = 0.65) in
`replay/logpb - fwd/logpb` (swing -8.9 .. +7.1 nats, sd 4.0), in
`replay/logpf - fwd/logpf` (-13 .. +5.6, sd 4.3) and in `fwd/step_var` (r = 0.65), and the
same period is imprinted on `fwd/tb_err`. On the fully frozen arm the P_B gap is flat
(sd 0.18, range -0.7 .. +0.3), the P_F gap is a smooth drift to -5, `step_var` decays
monotonically, and no channel has a periodicity above r = 0.2. The 600-step period
matches no configured cadence (eval 200, mean row age 46, churn 80, buffer 3975 rows,
turnover ~50 steps): it is an emergent P_B-P_F-buffer loop, and it is gone when P_B is
held fixed. Read: with P_B trainable, replay alternately makes stored paths MORE probable
under P_B than fresh ones (the (c) direction, gap +7) and bwd pulls it back past zero
(gap -9), and P_F's own noise level and forward fit ride that cycle. The sign of the gap
on p02 (always negative, section 4) is the cluster arms sitting on one side of the same
cycle at 10x the buffer size; the oscillation's amplitude there is unmeasured.

**Probe on stored paths** (`probe_pb.py`, `probe_pb_out.txt`, 1000 of `pbab_train`'s final
replay paths, scored by each checkpoint's LIVE modules):

| checkpoint | learned - bridge, nats/path | log P_B vs phase-1 exit, same paths |
|---|---|---|
| phase-1 exit | +24.1 (sd 13.2) | -- |
| pbab_train final | +28.6 (sd 12.2) | +4.6 mean, sd 4.2, median abs 3.9 |
| pbab_frozen final (live trunk, frozen head) | +24.2 (sd 13.0) | +0.14 mean, sd 1.4, median abs 0.6 |

The trainable arm moved P_B by ~4.6 nats per path in 3000 steps at lr 1.56e-5, mostly by
raising `|kappa - 1|` and `|delta|` on the last three steps (the ones nearest the terminal).
The third row is the size of the "head-only" leak on the fully frozen arm's trunk: the
trunk that trained under P_F alone moved P_B's function by 0.14 +- 1.4 nats. That is with
NO P_B-side gradient reaching the trunk; the head-frozen arm, where P_B's loss terms still
push the trunk through the fixed head, is the direct measurement (section 6b).

### 6b. Head-only freeze

`pbab_headfrozen`: `requires_grad` off on `backward_policy` only, the "zero that param
group" freeze. `gradnorm/backward_policy` reads 0.000 as on the full freeze, so by that
check it looks identical. It is not.

| channel (last 25%) | trainable | head freeze | full freeze |
|---|---|---|---|
| fwd/tb_err | 51.8 | **53.6** | 52.5 |
| fwd/over_coverage | 50.8 | **52.4** | 51.5 |
| fwd/logw_std_within | 48.8 | **50.4** | 49.5 |
| bwd/under_coverage | 33.5 | **33.9** | 33.2 |
| fwd/jensen_z | -7.30 | **-7.81** | -7.31 |
| fwd/logr_mean | 3.80 | **2.46** | 4.42 |
| replay/tb_err | 89.0 | 89.4 | 91.0 |
| replay/logpb - fwd/logpb | -0.8 (cycling +-8) | **+2.4 (range +1.9 .. +6.1)** | -0.1 (flat) |
| P_B moved on stored paths vs phase-1 exit | +4.6 (sd 4.2) | **+2.2 (sd 4.5)** | +0.14 (sd 1.4) |

Worst arm on every forward channel, and the only arm whose `replay/logpb - fwd/logpb`
stays POSITIVE for the whole run: stored paths are more probable under its P_B than the
fresh forward samples are. That is the (c) direction, taken. Mechanism: with the head
fixed, replay's P_B gradient still reaches `s_model`/`t_model` through it, so P_B moves
(2.2 nats per path, half the trainable arm's motion, entirely through the trunk), but the
head can no longer be re-fit by bwd against that motion, so the trunk absorbs the stored
paths and P_F pays -- the replay residual is as low as the trainable arm's (89.4 vs 89.0)
while the forward fit is worse than either. The 600-step cycle is damped (ACF r 0.27 at
~1700 steps, gap sd 1.1) because the restoring half of the loop (the head) is gone.

So the cheap freeze is not a weaker version of the full one; it is the worst of the three
options. Any freeze of P_B in this architecture has to include the trunk it reads.

### 6c. Resume proof

`make_resume.py`. `pbab_resume_a`: no config key; the equilibration stage's
`on_enter: [rebuild_prior_by_churn, freeze_pb]` takes the snapshot at the boundary; 320
steps. Read off its checkpoints on CPU:

| checkpoint | step | `pb_frozen` |
|---|---|---|
| phase-1 exit (source) | -- | no key (pre-snapshot checkpoint, restore is a no-op) |
| stage_start | 0 | 48 tensors |
| running | 300 | 48 tensors |
| final | 320 | 48 tensors |

The snapshot is bitwise identical across stage_start and final, and bitwise identical to
the phase-1 exit's `t_model`+`s_model`+`backward_policy`, while the live `s_model` had
drifted by up to 8e-4 per weight under P_F and the live head had not moved (0.0). So the
checkpoint carries the phase-1 function, not the drifted trunk.

`pbab_resume_b` (full reload of A's `running`, no freeze key in its config) printed
`freeze_backward_policy=full: ... (checkpoint)` on load, i.e. the restore path, not a
re-snapshot. It was stopped by hand ~20 steps in on GPU-load grounds before writing its
own final checkpoint, so the last link (B's saved snapshot == A's) is covered by the
unit test `test_snapshot_state_round_trip_restores_the_same_pb` rather than by the run.

## 7. Recommendation

**Freeze P_B after phase 1 -- the FULL freeze (P_B on a snapshot of trunk + head), not a
zeroed param group, and if it must be a cheap change, freeze it in replay only.**

On the evidence:

- Nothing P_F needs was lost. Forward fit, coverage, log w spread, Jensen Z, the
  memorisation sensor: all identical to the trainable arm over 3000 steps at T = 10,
  where every P_B effect is larger than at T = 100 ((b): representability value falls at
  least as 1/T; the posterior-variance role showed NO measurable cost here).
- Something was gained: the 600-step limit cycle in the buffer gaps, `step_var` and
  `fwd/tb_err` vanishes, P_F converges smoothly, learned log Z climbs faster, and the
  trainable arm's only "win" (`replay/tb_err` 2 nats lower) is exactly the residual that
  P_B absorbs on stored paths -- the (c) degeneracy, not a sampler improvement.
- The boundary re-fit that the cluster traces show (P_B share spiking to 0.5-1.0 for
  ~1000 steps after the transition) turned out not to be needed: the arm frozen at the
  phase-1 exit, before any re-fit, matched the trainable one.
- The cheap freeze is not a freeze, and it is WORSE than both alternatives (6b): with
  the head fixed, replay's P_B gradient still moves P_B through the shared trunk and bwd
  can no longer correct it, so the degenerate direction is taken and P_F pays on every
  forward channel. `freeze_backward_policy: full` is what the frozen arm ran.

Ordering of the three options as asked:
1. **Freeze P_B after phase 1 (full)** -- recommended; the arm did it and lost nothing.
2. **Freeze in replay only** (detach P_B in `_replay_step`) -- the minimal change that
   closes the (c) degeneracy and should kill the cycle's driver (replay is the branch that
   pushes the gap positive); leaves bwd's reparameterised posterior-fit alive. Not run here;
   a one-line arm off `make.py` if the full freeze is judged too blunt for T = 100.
3. **Keep trainable** -- not supported: it buys a limit cycle and a degenerate replay
   residual for no measured forward gain.

If a refit is ever wanted (section 2d's variance argument at long horizons), do it with the
phase-1 ELBO (reward-free, P_F detached) on a trigger from `fwd/logw_std_within`, not by
riding the fused TB loss.

## 8. What could not be determined

- **T = 100.** Everything measured locally is at T = 10 on ELJ mipcas. The direction of
  every argument says the frozen arm's parity gets easier at T = 100 (smaller step kernel,
  10x more steps), but the size of the phase-1 P_B departure at T = 100 and the amplitude
  of the trainable cycle on the cluster arms are unmeasured; no T = 100 phase-1 checkpoint
  is on this machine.
- **Long horizon.** 3000 steps. Whether a frozen posterior's variance cost appears at
  30k-50k steps (mip runs 48k) is open; `fwd/logw_std_within` was flat over the window.
- **Forward energies.** The eval `Mean Sample Energy` at 500 samples swings +-15 kJ/mol
  within one arm and cannot rank them; the train-batch `fwd/logr_mean` favours the frozen
  arm by 0.6 nats, which is within its own swing.
- **One seed, one system, one LR.** The cycle's absence on the frozen arm is a clean
  qualitative difference (sd 0.18 vs 4.0 on the gap), the level comparisons are not.
- **UMA / MACE.** The cluster P_B shares are 2-3x smaller than ELJ's and the boundary
  spike is larger; the A/B was ELJ only.
- **The 600-step period's mechanism.** It matches no configured cadence; it is P_B-linked
  (gone when frozen) but not further decomposed.
