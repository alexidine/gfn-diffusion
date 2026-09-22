# Loss composition

*Drift: **M** (mixed). Verified against commit `e17167e`, 2026-09-20. Sources at the end.*

A fused training step assembles one scalar loss out of three branches: forward, backward and replay. This page is about that assembly, from the branch draws up through the per-branch coefficient blocks to the weighted sum, and what each branch's residual can and cannot see. The controllers that move the weights during a stage are on [balance-controllers](balance-controllers.md); the VarGrad objective is on [vargrad](vargrad.md); the trajectory-balance residual and its fixed point are on [trajectory-balance](trajectory-balance.md); the coupling between the rollout that fills the replay buffer and the branch that drains it is on [on-policy-and-off-policy-training](on-policy-and-off-policy-training.md).

## The three branches

Each branch is a separate draw, a separate pass and a separate per-row loss, fired by `train.py::Modeller.fused_train_step`.

The **forward** branch rolls out its own trajectories. `train.py::Modeller.fwd_train_step` draws $B$ rows from `mol_dataset`, tiles each `repeats` times, samples forward under $P_F$ and calls the energy function on the terminals; it is the only branch that calls the energy function inside a training step. Its loss is built by `gflownet_losses.py::get_gfn_forward_loss`.

The **backward** branch draws terminal states, not trajectories: from the prior buffer under `cfg:stage.bwd_sampling_mode` `prior`, or from the fixed prior dataset under `dataset` (`train.py::Modeller.draw_bwd_sample`). A trajectory is then sampled backward from each terminal under $P_B$, and the reward is the stored one.

The **replay** branch draws stored forward trajectories from the replay buffer (`train.py::Modeller.draw_replay_sample`) and re-scores them under the current $P_F$ and $P_B$: the path is fixed, the densities move. Backward and replay share one builder, `gflownet_losses.py::get_gfn_backward_loss`, whose `trajectories` argument is what distinguishes a replayed path from a freshly sampled backward one.

Every branch that runs, runs a full batch. `cfg:stage.fracs` are coefficients on losses, not sample fractions, so a branch at a small nonzero weight still draws, scores and backpropagates $B$ rows.

## The fused sum

Weights are the live fracs, zeroed for any branch below the deactivation bar:

$$
L = \sum_{k \in \mathcal{A}} \frac{w_k}{\sum_{j \in \mathcal{A}} w_j}\, L_k
\;+\; \texttt{pooled\_vg}\cdot \overline{\ell_{\text{pooled}}}
\;+\; \mathbb{1}[\text{sidecar}]\, L_{\text{fwd}} ,
$$

with $\mathcal{A}$ the set of branches whose frac is at or above `cfg:stage.deactivate_threshold` (falling back to `cfg:controller.deactivate_threshold`). A branch below the bar is skipped entirely rather than down-weighted. The renormalisation is over the survivors, so the three fracs summing to 1 does not mean the weights entering the sum do. Two terms sit outside the renormalised mix: the cross-branch pooled VarGrad term, whose coefficient and Huber knee are its own, and the forward Z sidecar, which enters at weight 1 on a rollout that is not a Z pin and is refused if the forward branch also carries a frac.

Two further paths change the set. Every `cfg:controller.refresh_every` steps a branch that some rule or exit term reads (`protocol.py::StageProtocol.mode_dormant` is false for it) is evaluated regardless of its frac; that evaluation is detached and enters at weight 0, so it refreshes the branch's rolling statistics and contributes no gradient. And when the replay buffer is empty or unavailable, replay's share is added to the backward weight.

## Inside a branch

`protocol.py::StageProtocol.coeffs` overlays a stage's `loss_coeffs` overrides onto the base `cfg:fwd_loss_coeffs`, `cfg:bwd_loss_coeffs` and `cfg:replay_loss_coeffs` blocks, and `train.py::Modeller.set_loss_coeffs` rebuilds the three namespaces on every call, so a stage transition takes effect the moment it runs.

Within a builder, every coefficient greater than zero appends one per-row term to a list, and `gflownet_losses.py::combine_branch_terms` stacks the list and **sums** it after an elementwise `gflownet_losses.py::soft_clip` at `loss_clip`. Under the sum, a term's contribution is `coeff * term`, independent of how many other terms are active. The reduction was a mean until 2026-08-26, under which the effective weight of every term was `coeff / n_active`. A branch whose coefficients are all zero yields an explicit zero row vector rather than an empty stack.

### The coefficient inventory

The table lists each key of the three coefficient blocks against the branches that read it, the term or effect it multiplies, and the symbol implementing it.

| key | branch(es) | term or effect it multiplies | symbol |
|---|---|---|---|
| `tb` | fwd, bwd, replay | Huber trajectory-balance residual | `gflownet_losses.py::get_tb_loss` |
| `vg_lb` | fwd, bwd, replay | VarGrad, group mean centring | `gflownet_losses.py::vg_lb`, `::condition_grouped_empirical_z` |
| `vg_lme` | fwd, bwd, replay | VarGrad, group log-mean-exp centring; exclusive with `vg_lb` | `gflownet_losses.py::vg_lme` |
| `emp_z` | fwd, bwd, replay | regression of $\log Z$ onto this batch's empirical per-group estimate | `gflownet_losses.py::emp_Z` |
| `emp_z_persistent` | fwd, bwd, replay | regression of $\log Z$ onto the tracker's persistent per-condition target | `gflownet_losses.py::get_gfn_forward_loss` |
| `z_level` | fwd | Z-only per-condition level regression, $\log w$ detached | `gflownet_losses.py::z_level_loss` |
| `db`; `subtb` (`subtb_lambda`) | fwd, bwd, replay | detailed-balance and sub-trajectory-balance residuals; both need `model.full_flow` | `gflownet_losses.py::get_db_loss`, `::get_subtb_loss` |
| `mle` | bwd, replay | maximum likelihood on stored terminals | `gflownet_losses.py::terminal_mle` |
| `tbc` | bwd | reward-free VarGrad over K backward rollouts from one terminal | `gflownet_losses.py::get_tbc_loss` |
| `pf_boost` | bwd, replay | $P_F$ retention term | `gflownet_losses.py::get_pf_retention_loss` |
| `level_gap` (`level_gap_clamp`, `level_gap_pf_only`) | bwd, replay | detached EMA level delta times $-\log P_F$, or times $\log w$ | `gflownet_losses.py::get_gfn_backward_loss` |
| `pooled_vg` (`pooled_beta`, `pooled_ratio`, `pooled_source`, `pooled_bridge_only`, `pooled_thin_by_condition`) | declared on fwd, spans two branches | one condition group over both branches' live rows, added outside the mix | `gflownet_losses.py::pooled_condition_vargrad` |
| `beta`; `loss_clip` | fwd, bwd, replay | Huber knee inside every TB-style term above; elementwise ceiling on each term before the sum | `gflownet_losses.py::get_tb_loss`, `::soft_clip` |
| `freeze_policy`; `freeze_z` | fwd, bwd, replay | detaches $\log P_F$, $\log P_B$ and the conditioner-to-flow path at source; detaches $\log Z$ and the flow | `gflownet_losses.py::get_gfn_forward_loss` |
| `traj_grads` | fwd, bwd, replay | keeps the reparameterised trajectory path live | `models/gfn.py::GFN.get_traj_fwd` |
| `reward_grads` (`reward_grad_clip`, `reward_grad_gate`, `reward_grad_force_clip`, `path_grad_last_k`, `path_grad_scale`) | fwd; replay under `resample_last_k` | lets $d\log R/dx_T$ reach the policy, plain or reshaped | `gflownet_losses.py::get_gfn_forward_loss` |
| `stored_force_k` (`stored_force_mode`, `force_chunk_rows`); `resample_last_k` | replay | a zero-valued surrogate adding the stored terminal force to $\log R$; a live re-sample of the last k steps with the reward re-scored. Mutually exclusive | `gflownet_losses.py::get_gfn_backward_loss` |
| `tb_z_source` | fwd, bwd, replay | which level the TB term uses: `learned`, `persistent`, or `batch_root` on fwd | `gflownet_losses.py::batch_root_z` |
| `repeats` | fwd, bwd, replay | K-tiling of the draw; a cross-terminal group on fwd, a shared-terminal group on bwd | `train.py::Modeller.mode_repeats` |
| `condition_block_m` | bwd, replay | makes the draw condition-blocked, M distinct same-condition terminals per block | `buffer.py::CrystalBuffer._sample_indices` |
| `vg_by_condition`, `vg_detach_center` | fwd, bwd, replay | group by `condition_id` rather than by K-tile; treat the group centre as constant | `gflownet_losses.py::condition_grouped_empirical_z` |
| `exploration_std` | fwd | additive per-step log-std on the sampling kernel only | `train.py::Modeller._fwd_exploration_std` |

Rules over these blocks live in `config_invariants.py`: `vargrad_needs_groups` refuses a VarGrad coefficient with no group to centre over; `runs_grouped_vargrad` is the single predicate for whether grouped VarGrad is running, over the keys `BRANCH_VARGRAD_COEFFS` and `CROSS_BRANCH_VARGRAD_COEFF`; `batch_root_forward_is_well_formed` confines `batch_root` to the forward branch; `replay_seat_problems` and `condition_draw_problems` are re-run against the resolved config for every stage by `set_loss_coeffs`. `train.py::Modeller._warn_if_z_untrained` prints a warning naming the stage when the live coefficients leave no trainer on the flow head.

## What each branch's residual is blind to

Write $\delta = \log Q(\tau) - \log P(\tau)$ with $Q = P_F(\tau)$ the policy's trajectory law and $P = R\,P_B/Z$ the target's. Both are proper distributions, so $\mathbb{E}_Q[e^{-\delta}] = 1$ and $\mathbb{E}_P[e^{+\delta}] = 1$ under the true $Z$. Markov's inequality applied to each gives

$$
Q(\delta < -m) \le e^{-m}, \qquad P(\delta > +m) \le e^{-m}.
$$

An on-policy draw is therefore exponentially unlikely to contain a strongly negative residual, and a target-side draw exponentially unlikely to contain a strongly positive one. The forward branch draws from $Q$, so its residual sees over-weighting and is blind at rate $e^{-m}$ to under-coverage. The backward branch draws terminals from the prior buffer or prior dataset and trajectories from $P_B$, the target-side measure, so its blindness runs the other way. The replay branch draws stored forward rollouts, so it inherits $Q$'s blindness at intake: a region the policy never visited cannot have been admitted. $\mathbb{E}_Q[\delta^2]$ weights the under-coverage half by a measure that is exponentially small there, so a reduction in the forward branch's residual carries no information about coverage.

## Held-out quantities

In the equilibration stage of the canonical config the forward block is `fwd: { tb: 1.0, freeze_policy: 1.0 }` and `cfg:stage.fracs` pins `fwd` at 0, below that stage's `deactivate_threshold` of 0.01. Two mechanisms then hold the forward residual out of training. `freeze_policy` detaches $\log P_F$ and $\log P_B$ at source inside `get_gfn_forward_loss`, so no term built from them reaches the policy parameters whatever else is active. And `fwd_active` is false, so `fused_train_step` detaches the whole forward loss before the sum and gives it weight 0. The policy is trained by the backward and replay branches only, and `fwd/tb_err`, the batch RMS residual computed by `utils.py::quick_tb_stats`, is a measurement on rows against which no policy gradient was taken, while `bwd/tb_err` and `replay/tb_err` are measurements on the branches that took them.

`train.py::Modeller._fwd_gates` separates `fwd_ran` from `fwd_active`: under `cfg:stage.fwd_rollout_every` the rollout is paid for on the steps it fires, for the replay admission and the $\log Z$ pin, and carries no weight on any of them. On such a step $\log Z$ is in no live graph, so `train.py::Modeller.z_level_fill` runs before the other two branches and their residuals are computed against the pinned level.

## Retired and dead keys

`detach_pb` was carried in the `bwd_loss_coeffs` and `replay_loss_coeffs` blocks of the canonical config, commented as detaching $\log P_B$ on that branch's rows, and no module read it: it appeared in no `.py` file a training step executes. On 2026-09-20 both lines were deleted from `configs/mk_dev.yaml` and `bwd_loss_coeffs.detach_pb` and `replay_loss_coeffs.detach_pb` were added to `utils.py::_RETIRED_KEYS`, so a config still carrying either one is now refused at load rather than silently ignored. The live control over $P_B$ is `cfg:freeze_backward_policy` and the stage actions that set it.

`fwd_loss_coeffs.terminal_force` is refused at `set_loss_coeffs` with a message stating its retirement on 2026-09-15 and naming `replay_loss_coeffs.stored_force_k` as where forces are scored now. `condition_block_m` moved out of the buffer blocks into the two coefficient blocks on 2026-08-17, and the old `buffers.*.condition_block_m` paths fail at load; `condition_log_z.{fwd,bwd,replay}_tb_z_source` moved into the coefficient blocks as `tb_z_source`.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:stage.fracs`, `cfg:stage.deactivate_threshold`, `cfg:stage.loss_coeffs`, `cfg:stage.fwd_rollout_every`, `cfg:stage.fwd_z_sidecar`, `cfg:stage.bwd_sampling_mode`, `cfg:controller.deactivate_threshold`, `cfg:controller.refresh_every`, `cfg:freeze_backward_policy`, and every key of `cfg:fwd_loss_coeffs`, `cfg:bwd_loss_coeffs`, `cfg:replay_loss_coeffs` named in the inventory above.

Code: `train.py::Modeller.fused_train_step`, `.fwd_train_step`, `.bwd_train_step`, `.replay_train_step`, `.set_loss_coeffs`, `._warn_if_z_untrained`, `._fwd_gates`, `.mode_repeats`, `.draw_bwd_sample`, `.draw_replay_sample`, `.z_level_fill`, `._stash_z_fill_logw`; `gflownet_losses.py::get_gfn_forward_loss`, `::get_gfn_backward_loss`, `::combine_branch_terms`, `::soft_clip`, `::get_tb_loss`, `::pooled_condition_vargrad`, `::condition_grouped_empirical_z`; `protocol.py::StageProtocol.coeffs`, `.base_coeffs`, `.mode_dormant`, `.mode_boostable`, `protocol.py::POOLED_SOURCES`; `config_invariants.py::vargrad_needs_groups`, `::runs_grouped_vargrad`, `::batch_root_forward_is_well_formed`, `::replay_seat_problems`, `::condition_draw_problems`; `utils.py::quick_tb_stats`.

## Could be tooling

The coefficient inventory is generatable. Both branch builders assemble their term list by coefficient-gated appends of the form `if loss_coeffs.<key> > 0: losses.append(<term> * loss_coeffs.<key>)`, which an AST walk matches directly: the attribute on `loss_coeffs` gives the key, the call inside the append gives the term, the enclosing function gives the branch, with `getattr(loss_coeffs, ...)` reads taking the key from the literal second argument. Run against the union of keys in the three canonical blocks, that walk yields two difference sets: keys the config carries that no builder reads, and keys a builder reads that the canonical config does not carry. A second pass over `config_invariants.py` would name the keys each load-time rule mentions, leaving a coefficient governed by no rule visible as such.

## Sources

The code above, read at the stamped commit, and the three coefficient blocks and the `unconditional_tb` stage list of the canonical config. Repo: docs/design/training_workflow_vs_manuscript.md Part 1, docs/to_do_rebuild.md Part B (the log-ratio identities and the blindness bound). Memory files located the code and were not used as evidence: feedback_fracs_are_loss_weights_on_full_batches, project_loss_terms_are_averaged_not_summed, project_fused_branch_roles_and_replay_overfit, project_detach_pb_is_a_dead_key, project_batch_design_force_spectrum, project_condition_block_m_is_a_loss_coeff.
