# MLE phase

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

The first stage of the canonical protocol is `train_prior`, a maximum-likelihood warm start: terminals come from a fixed dataset, a trajectory is sampled backward from each under $P_B$, and the forward log-density of that trajectory is maximised. No reward enters the loss, no trajectory-balance residual is formed, and the learned normaliser receives no gradient. This page covers the objective, the structure of its descent, the exit gate, the exit actions, and what a resume from the snapshot re-enters. Left out and named: the TB residual and its level decomposition, [trajectory-balance](trajectory-balance.md); the stage engine, [stage-protocol-engine](stage-protocol-engine.md); the buffer the next stage draws from, [prior-buffer](prior-buffer.md); what a checkpoint holds, [checkpoints-and-resume](checkpoints-and-resume.md).

## The objective

In the canonical config the stage carries `train_mode: bwd`, `bwd_sampling_mode: dataset` and the single coefficient block `bwd: { mle: 1.0, tbc: 0.0, repeats: 1.0 }`. `train.py::Modeller.bwd_train_step` draws terminals through `::Modeller.draw_bwd_sample` -- from `prior_dataset` under `dataset` -- and `gflownet_losses.py::get_gfn_backward_loss` appends one term, `gflownet_losses.py::terminal_mle` of `log_pf` and `log_pb` alone, scaled by `cfg:bwd_loss_coeffs.mle`. `train.py::Modeller.mode_repeats` reads `cfg:bwd_loss_coeffs.repeats`; at 1 the `bound` estimator runs and the per-terminal loss is

$$\ell(x) \;=\; -\,\mathbb{E}_{\tau \sim P_B(\cdot \mid x)}\!\left[\log P_F(\tau) - \log P_B(\tau \mid x)\right],$$

a sample average over the $K$ tiles. By Jensen this is an upper bound on $-\log P_F^\top(x)$, the negative forward terminal log-marginal: the maximised quantity is a lower bound on the log-likelihood the policy assigns to the stored terminal. At `repeats > 1` the `exact` estimator replaces the average with the DReG form: detached softmax responsibilities over the tiles' $\log w = \log P_F - \log P_B$, against the importance-weighted bound `gflownet_losses.py::log_pf_estimate` reduces.

**No log Z gradient.** Neither $\log R$ nor $\log Z$ appears in `terminal_mle`'s arguments, so the term is structurally independent of the flow head; `cfg:bwd_loss_coeffs.freeze_z: 1.0` detaches `log_Z_learned` in `get_gfn_backward_loss` in addition. `train.py::Modeller._warn_if_z_untrained` prints, once per stage, `no mode trains the flow (Z) head ... log_Z will not move for this stage`. The stage's `flags.update_log_z: true` feeds this batch's $\log w$ into the per-condition tracker through `gflownet_losses.py::update_and_lookup_condition_log_z`; that accumulation is detached bookkeeping, not a gradient path.

**What it fits.** The loss is an expectation over terminals drawn from the dataset distribution $\mu$ and paths drawn from $P_B$, so its stationary point is $P_F = \mu P_B$ as path measures, with $\mu$ the empirical distribution of the prior dataset and not the target $\pi$. In the notation of [trajectory-balance](trajectory-balance.md), this drives $B := D_{\mathrm{KL}}(\mu P_B \| P_F)$ toward zero and leaves $C := D_{\mathrm{KL}}(\mu \| \pi)$ untouched; at $B = 0$ the backward level $J_B$ equals $\mathrm{ELBO}(\mu) = \log Z^\star - C$, the floor below which no forward policy makes the next stage's backward residual mean-zero. The scalar is left where the checkpoint put it: in the unconditional protocol the successor stage declares `on_enter: [ 'rebuild_prior_by_churn', 'bootstrap_z:rollout:4000' ]` and `train.py::Modeller.bootstrap_z_by_rollout` sets $\log Z$ from a large forward rollout through the winsorised-Huber root, and where no such action is declared $\log Z$ opens the next stage at the checkpoint's value and moves first at a `z_level_fill`.

## Two legs of the descent

The forward policy is Gaussian per step. `models/gfn.py::GFN.split_params` produces a mean and a log-variance per dimension, the latter through `logvar = tanh(logvar_i / log_var_range) * log_var_range` plus a constant base and a `var_clip` clamp, under `cfg:model.learned_variance` and `cfg:model.log_var_range`; `::GFN.fwd_gauss_logprob` scores the increment under $\mathcal{N}(\Delta t\,\mu_\theta,\ \Delta t\,C)$. Write $\lambda = \log\sigma^2$ for one dimension of one step and $u = z/\sigma$ for its standardised residual, $z$ being the increment minus the drift. The diagonal contribution to $\log P_F$ is $-\tfrac12(u^2 + \lambda)$, so the loss gradient splits into two channels of different form:

$$\frac{\partial \ell}{\partial \mu} \;=\; -\frac{z}{\sigma^{2}}, \qquad \frac{\partial \ell}{\partial \lambda} \;=\; \tfrac12\left(1 - u^{2}\right).$$

The mean channel's magnitude is $|u|/\sigma$: unbounded in the residual and in the inverse of the current scale. The variance channel's is a function of the standardised residual only, bounded on one side by $\tfrac12$ whatever the fit, and further multiplied by $\operatorname{sech}^2$ of the head's pre-activation over `log_var_range` by the tanh gate, which vanishes as that pre-activation leaves the range. Two consequences follow from the objective alone. Under a rate that starts small and rises -- in the canonical config `cfg:lr_control.burn_in_steps` at `cfg:lr_control.burn_in_scale`, then the configured rate -- the variance channel's per-step motion is bounded by the rate times an $O(1)$ gradient, while the mean channel's is proportional to the residual it is reducing. And the sign of the variance channel is set by the mean channel: while $\mathbb{E}[u^2] > 1$ the gradient demands a larger $\sigma$, so a descent direction that sharpens the variance exists only once the mean fit has brought $\mathbb{E}[u^2]$ under 1. Sharpening carries the larger part of the attainable objective value: $\log P_F$ contains $-\tfrac12\sum\lambda$ over $T$ steps and every live dimension, so a uniform factor $e^{-a}$ on every $\sigma$ is worth $a\,T\,D_{\text{live}}$ nats.

The two channels are distinct parameters of the same networks, and a checkpoint carries both. A warm-started policy enters with its variance head wherever that checkpoint left it, and the remaining descent is whatever the loaded parameters do not already satisfy; `cfg:load_weights_only` takes the weights alone and starts optimizers, buffers and step count fresh.

## The exit gate

The stage's exit block is one term: `{ metric: gates/progress_done, above: 0.5, patience: 1 }`. The channel is published by `train.py::Modeller.progress_metrics` through `progress_metrics.py::progress_gate` under `cfg:progress_gate`, whose canonical spec is `mode: level`, `level_window: 2500`, `min_history: 2000`, and two metrics: `w1r/median` at bar 5.0 and `w1r/worst` at bar 10.0.

`progress_metrics.py::_column_w1_ratio` computes, per latent column, the 1-D Wasserstein distance between the eval sample and the prior dataset's own latents (`progress_metrics.py::_column_w1`, in quantile form, minimised over rigid rotations on periodic columns), divided by a floor measured by scoring disjoint draws of the reference against itself. `w1r/median` is the median of that ratio over columns with a non-degenerate floor, `w1r/worst` its maximum. In level mode the gate takes, per metric, the median of the values recorded within `level_window` steps of the latest one, requires at least three such points and at least six history points, and publishes 1.0 only when every metric's smoothed level is at or under its bar; a metric without enough history makes the gate return 0.0 rather than passing, as does a step below `min_history`, and `progress_metrics` announces an abstention once per stage when it cannot run at all.

What the bar reads is a set of one-dimensional marginals: every column is scored independently and both headline statistics are order statistics over those columns, so a sampler whose per-column distributions match the reference reads at the measured floor whatever its joint structure. `train.py::Modeller.progress_metrics` records, beside the energy-marginal block it publishes next to `w1r`, that a column-shuffled sampler with perfect marginals scores at the floor on `w1r` and fails on the energy marginal, a joint function of every coordinate.

A slope bar on the loss itself exists but is not declared here: `mle_gate` is a block whose presence is the switch (`protocol.py::Stage._parse_mle_gate`), and where declared `train.py::Modeller.update_mle_gate` fits `bwd/mle` over `mle_gate.window` train steps and publishes `gates/mle_flat` when the one-sided bound `rate + slope_t * se` falls under `mle_gate.min_rate`. The config records beside `progress_gate` that `bwd/mle` carries an unknown additive constant $\mathbb{H}(p_{\text{data}})$, so no value of it has an absolute meaning.

## Exit actions, and what the next stage inherits

`on_exit: [ 'snapshot:phase1_exit', 'snapshot_prior' ]`. `protocol.py::StageProtocol._snapshot` saves a checkpoint tagged `phase1_exit` with `with_buffers=True`, stamping `request_eval` True into the saved state only. `protocol.py::StageProtocol._snapshot_prior` saves a `prior` checkpoint, sets `prior_model` to a frozen `deepcopy` of the EMA model in eval mode, and deletes the stage's `best` checkpoint, which was kept against `bwd/mle`; it raises where `internal_prior` is set, since backward sampling there never reads `prior_model`.

`protocol.py::StageProtocol.advance` then switches the stage name, installs a fresh `stage_ctrl` and the successor's entry fracs, clears `combo_loss_record`, drops the batch sizer and its OOM ceilings, resets the batch to the configured base under `grow_batch_size`, rebuilds the optimizers, re-resolves the coefficient blocks, re-enters LR burn-in, refreshes the gradient-clip bar, and runs `on_enter`. Policy weights, buffers and the step count carry across; Adam moments, stage control state and the batch selection do not.

## Resuming from the snapshot

The `phase1_exit` snapshot is written by an `on_exit` action of `train_prior`, so the stage name it records is `train_prior`. `checkpointing.py`'s `MODELLER_STATE_DEFAULTS` carries `stage`, `stage_ctrl` (exit-term pass streaks, gate windows, published gate values) and `_progress_history`, the gate's own evidence. `protocol.py::StageProtocol.begin` returns before the `skip_if` chain when `step_ind != 0`, so a resume is left where its checkpoint says and re-enters `train_prior` rather than skipping it; the stamped `request_eval` pulls an eval to the first post-resume step and the exit re-fires through the ordinary `evaluation` to `maybe_advance` path. `skip_if: prior_loaded` skips the stage only on a fresh run and only when a `prior_model` is loaded by name, and the skipped path runs no `on_exit` actions.

*Open hypothesis, owner, not measured: that MLE fixes which regions the flows reach and TB only reweights within them.*

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:protocols.<name>.stages[train_prior].{train_mode, bwd_sampling_mode, skip_if, flags.update_log_z, flags.scramble_conditions, hot_lr_sensor, loss_coeffs.bwd, exit, on_exit}`; `cfg:bwd_loss_coeffs.{mle, tbc, repeats, freeze_z, traj_grads, beta, tb_z_source}`; `cfg:progress_gate.{mode, level_window, min_history, metrics}`; `cfg:model.{learned_variance, log_var_range, t_scale, learn_pb, pb_var_range, pb_drift_range}`; `cfg:lr_control.{burn_in_steps, burn_in_scale}`; `cfg:checkpoint_name`, `cfg:load_weights_only`, `cfg:prior_model_name`, `cfg:eval_period`.

Code: `gflownet_losses.py::terminal_mle`, `::log_pf_estimate`, `::get_gfn_backward_loss`, `::combine_branch_terms`, `::update_and_lookup_condition_log_z`; `train.py::Modeller.bwd_train_step`, `.draw_bwd_sample`, `.mode_repeats`, `._bwd_retention_priority`, `._warn_if_z_untrained`, `.bootstrap_z_by_rollout`, `.progress_metrics`, `.update_mle_gate`; `progress_metrics.py::_column_w1`, `::_column_w1_ratio`, `::progress_gate`; `protocol.py::StageProtocol.begin`, `.advance`, `._run_action`, `._snapshot`, `._snapshot_prior`, `protocol.py::Stage._parse_mle_gate`; `models/gfn.py::GFN.split_params`, `::GFN.fwd_get_logvars`, `::GFN.fwd_gauss_logprob`, `::GFN.get_bwd_correction`; `utils.py::gaussian_params`; `checkpointing.py::MODELLER_STATE_DEFAULTS`.

## Could be tooling

Both of the gate properties described above are checkable without a run. Scoring a column-wise shuffle of a reference draw through `_column_w1_ratio` yields, per route, the value a joint-wrong, marginal-right sampler reads. And the floor cache is keyed on $n$, so a gate whose `w1r/n_eval` moves between evals is scored against a different floor.

## Sources

Repo at the stamped commit: the code listed above, and the `train_prior` stage blocks and `bwd_loss_coeffs` of `configs/mk_dev.yaml`. `docs/design/training_workflow_vs_manuscript.md` Part 1 (phase 1). That note records `snapshot_prior` as removed from the exit actions under the anchor-only prior; the canonical config at this commit carries it on both `train_prior` stages. Memory files located the code and were not used as evidence: project_phase1_mle_two_leg_descent, project_phase1_exit_gate_fires_on_marginals, project_phase1_exit_resume_and_mle_gate, project_tb_surface_lr_ceiling, project_mle_shapes_tb_reweights_hypothesis.
