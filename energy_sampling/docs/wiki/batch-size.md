# Batch size

*Drift: **M** (mixed). Verified against commit `e17167e`, 2026-09-19. Sources at the end.*

The batch size $B$, `cfg:batch_size`, is the width of every draw a training step makes: the forward rollout, the backward draw, the replay draw, the rows admitted to the replay buffer, the eval sampler's starting size. It is not divided between branches. The branch weights `cfg:stage.fracs` are coefficients on losses, not sample fractions, so a branch active at weight 0.05 still rolls out, scores and backpropagates a full $B$ rows ([loss-composition](loss-composition.md)). This page is about what $B$ sets, what moves it at runtime, and what a change in it does to step time, gradient noise and the log-$Z$ estimate.

Two names are needed. The *live* batch is `Modeller.batch_size`, run state the OOM path, the wall-clock guard and the sizer may move, reported as `Batch Size`. The *configured* batch is `cfg:batch_size`, restored at every stage entry (`protocol.py::StageProtocol.advance`) and used as the sizer's domain floor (`train.py::Modeller._batch_floor`).

## Where $B$ is consumed

| consumer | what $B$ does there | symbol |
|---|---|---|
| forward rollout, condition draw | draws $B$ `mol_dataset` rows, each tiled `repeats` times: $BR$ trajectories | `train.py::Modeller._rollout_mol_batch` |
| forward branch, energy call | scores all $BR$ structures in one call, unless `cfg:energy_config.internal_oom_recovery` chunks it | `train.py::Modeller.fwd_train_step` |
| backward branch | draws $B$ rows from the prior buffer or prior dataset | `train.py::Modeller.draw_bwd_sample` |
| replay branch | draws $B$ rows from the replay buffer | `train.py::Modeller.draw_replay_sample` |
| aligned per-condition draw | caps conditions at $C = \lfloor B / \max(X, Y)\rfloor$, so branch batches $CX$, $CY$ shrink with a cut | `train.py::Modeller._choose_draw_conditions` |
| replay admission | `cfg:buffers.replay_buffer.churn_rate` at 0 means the live $B$: $B$ attempts per manage call | `train.py::Modeller.manage_replay_buffer` |
| replay occupancy | $O = B\tau/N$ by Little's law ([rollout-cadence-and-dose](rollout-cadence-and-dose.md)) | `train.py::Modeller.manage_replay_buffer` |
| occupancy trigger | the bar reads buffer length against the *live* $B$, while churn was sized at stage entry | `train.py::Modeller._rollout_trigger_reading` |
| log-$Z$ fill | the closed-form root's standard error carries $1/\sqrt{B}$ | `gflownet_losses.py::winsorized_z_root` |
| dose | $B$ enters the memorisation dose $D$ inversely | [rollout-cadence-and-dose](rollout-cadence-and-dose.md) |
| gradient accumulation | each iteration contributes $B$ toward `cfg:fused_grad_accum_min_samples` | `train.py::Modeller.train_step` |
| held-out replay probe | rows are $\min(\texttt{val\_cap}, B, n_{\text{val}})$, so a cut shrinks the probe | `train.py::Modeller._replay_val_size` |
| eval draws | start at the live $B$ and only ever fall, independently of training; the learned cap is never cleared, since `reset_eval_draw_size` has no call site in the package | `train.py::Modeller.eval_draw_size` |
| Z bootstrap by rollout | falls back to $B$ when `cfg:eval_num_samples` is unset | `train.py::Modeller.bootstrap_z_by_rollout` |
| prior-buffer churn | does **not** read $B$: paced by `cfg:buffers.prior_buffer.churn_batch_ref`, with $B$ its default | `train.py::Modeller.manage_prior_buffer` |

## Step time

Step time is close to affine in the batch, $t(B) \approx a + bB$, and which term dominates decides whether $B$ is a lever. The fixed term $a$ is Python dispatch and kernel launches: the policy rollout issues one small module call per layer per integration step, so it is dispatch-bound rather than FLOP-bound at production width, and $a$ is larger where `cfg:compile_policy` resolves to eager (native Windows) than on a compiled Linux node. The scaled term $bB$ is dominated by the energy call whenever the reward is a machine-learned potential, since the forward branch scores $BR$ structures every rollout step. On an ELJ route much of the step is fixed cost and growing $B$ buys samples nearly free; on a UMA or MACE route the step is energy-bound and $B$ costs close to linearly ([mlip-energy-routes](mlip-energy-routes.md)).

GPU occupancy is a ratio, not work: $\text{util} \approx bB/(a + bB)$ to first order, so occupancy rises with $B$ only where $a$ is a real share of the step, and a reading can rise while samples per second falls. The sensors and their windows belong to [gpu-occupancy](gpu-occupancy.md).

## What $B$ buys statistically

Every branch loss is a mean over rows, so $B$ sets the variance of the gradient estimate and nothing about its expectation: per-step gradient noise falls as $1/\sqrt{B}$ while updates per unit wall clock fall as $1/t(B)$. Where $t$ is dominated by $a$, doubling $B$ cuts gradient noise by a factor of $\sqrt{2}$ almost for free; where $t$ is dominated by $bB$, it trades update rate one-for-one against a $\sqrt{2}$ noise reduction.

The log-$Z$ side is sharper, because the estimator is explicit. `gflownet_losses.py::winsorized_z_root` returns the level at which this batch's Huber TB loss has zero $Z$-gradient, with the M-estimator standard error

$$
\mathrm{se} = \frac{\mathrm{rms}\big(\mathrm{clip}(r, \pm\beta)\big)}{\sqrt{B}\, f_{\text{unclipped}}},
$$

$r$ being the per-row residual and $f_{\text{unclipped}}$ the fraction inside the Huber knee, which is also the loss's curvature in $z$. Precision of the pinned level improves as $\sqrt{B}$, and the `cfg:z_calibration.fill_se` gate reads this number directly: a batch whose rows are all saturated has $f_{\text{unclipped}} = 0$, $\mathrm{se} = \infty$, and licenses no fill at any $B$. Since $Z$ is pinned to a batch statistic, a batch that moves moves the pinned level with it.

## Gradient accumulation

`cfg:fused_grad_accum_min_samples`, written $A$, is a floor on the *effective* update size and engages only below itself. In `train.py::Modeller.train_step`, accumulation is on when $A > B$; each host iteration then adds $B$ to `fused_accum_count`, scales its loss by $B/A$, and an optimizer step is taken once the count reaches $A$. At $B \ge A$ the step is plain and unscaled, never up-weighted by $B/A > 1$. Two consequences. Optimizer updates per second are $\text{samples per second} / \max(A, B)$, so for $B \le A$ the update rate is maximised at $B = A$ exactly and a cut below $A$ leaves it unchanged. And an OOM wipes both the gradients and the count (`train.py::Modeller.handle_train_epoch_error`), discarding the partial window rather than descending it. The reported target is `batch/accum_target`; `config_invariants.py::effective_batch_meets_baseline` warns when $\max(A, B)$ falls below `MIN_EFFECTIVE_BATCH`.

## The batch sizer

`train.py::Modeller.select_batch_size` runs every step, and only when `cfg:grow_batch_size` is true; with growth off it never runs and an OOM cut stands until a resume. Its throughput objective is a constant, not a search: $B = A$. Growth above the base is driven by GPU occupancy alone, and runs only when `cfg:batch_util_target` is set (a fraction of the card, converted to percent at the one read site). Occupancy evidence only vetoes candidates and never orders the batch; a growth justified by predicted occupancy is audited one policy window later against the base rung's reading and stands down if the occupancy did not arrive; and where there is no target, no sensor or no reading, nothing is grown and nothing is removed.

The calibration is feed-forward, once per stage. The walk climbs from the base rung by `cfg:batch_growth_factor` while the increment stays under `cfg:batch_growth_cap` and linearly at the cap, dwelling at least `cfg:batch_growth_interval` steps per rung and reading raw occupancy samples taken during that rung, not a trailing window straddling rungs. It holds the smallest measured rung clearing the target; if none clears, it holds the argmax rung and reports `infeasible`, naming the binding bound. A rung that never gathers enough samples over enough wall clock is starved, and the walk concludes. A `target_met` verdict is re-measured every `cfg:batch_sizer_retest_steps`, stepping one rung down first so the re-walk leaves a two-row table and arms the audit; other verdicts are not retested here.

Every other mechanism bounds the domain and never selects. `cfg:max_batch_size` is a hard ceiling. An OOM records the failing size, cuts by `cfg:oom_batch_shrink_factor`, holds flat for `cfg:oom_cooldown_steps`, and installs a per-stage ceiling expiring after `cfg:batch_oom_ceiling_retest_steps` quiet steps; eval OOMs cut the eval draw alone ([compute-guards](compute-guards.md)). `cfg:max_step_seconds` is a runaway guard: it cuts proportionally, refuses to cut into the accumulation regime, and stands down for the stage if a cut did not move step time. A stage transition clears the conclusion, the ceiling and the timing windows and restores the base. A resume clamps the restored size to this config's ceiling, and with growth off restores the configured value verbatim and drops the conclusion (`checkpointing.py::Checkpointer.reconcile_batch_size`).

Retired keys fail loudly at load (`utils.py::_RETIRED_KEYS`): the occupancy actuator `gpu_util_floor`, and the throughput walk `auto_batch_throughput_opt` with `batch_growth_min_throughput_gain`, `batch_knee_recheck_steps`, `batch_growth_slow_interval`, `batch_growth_max_step_regression`, `batch_growth_min_gain`.

## Calibrations

- **[calibration, a100_stab_aug16 U tier, 2026-08-18]** mipcas ELJ, two eval-free rungs: $t \approx 0.35 + 1.9\times10^{-4} B$ seconds, so the fixed term is about 65% of the step at $B = 1000$ and 20% at $B = 7410$.
- **[calibration, a100_stab_aug16 U tier, 2026-08-18]** `gpu/util_policy` minus an out-of-process reading, same window and statistic: $-5$ to $-8$ points at $B = 1000$, $+38$ to $+42$ at $B = 7410$.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:batch_size`, `cfg:grow_batch_size`, `cfg:max_batch_size`, `cfg:fused_grad_accum_min_samples`, `cfg:batch_util_target`, `cfg:batch_growth_factor`, `cfg:batch_growth_cap`, `cfg:batch_growth_interval`, `cfg:batch_sizer_retest_steps`, `cfg:batch_oom_ceiling_retest_steps`, `cfg:oom_batch_shrink_factor`, `cfg:oom_cooldown_steps`, `cfg:max_step_seconds`, `cfg:gpu_util_sample_period_s`, `cfg:gpu_util_window_s`, `cfg:gpu_util_policy_window_s`, `cfg:compile_policy`, `cfg:eval_num_samples`, `cfg:buffers.replay_buffer.churn_rate`, `cfg:buffers.replay_buffer.val_cap`, `cfg:buffers.prior_buffer.churn_batch_ref`, `cfg:energy_config.internal_oom_recovery`, `cfg:z_calibration.fill_se`, `cfg:stage.condition_draw`, `cfg:stage.fwd_rollout_triggers.occupancy_min_batches`.

Code: `train.py::Modeller.select_batch_size`, `._batch_floor`, `._conclude_batch_calibration`, `.handle_train_epoch_error`, `.eval_draw_size`, `.reset_eval_draw_size`, `.train_step`, `.fused_train_step`, `._rollout_mol_batch`, `.fwd_train_step`, `.draw_bwd_sample`, `.draw_replay_sample`, `._choose_draw_conditions`, `.manage_replay_buffer`, `.manage_prior_buffer`, `._replay_val_size`, `._rollout_trigger_reading`, `._stash_z_fill_logw`, `.z_level_fill`, `.bootstrap_z_by_rollout`; `gflownet_losses.py::winsorized_z_root`; `protocol.py::StageProtocol.advance`; `checkpointing.py::Checkpointer.reconcile_batch_size`; `config_invariants.py::util_target_actuable`, `.batch_ceiling_above_floor`, `.effective_batch_meets_baseline`; `utils.py::_RETIRED_KEYS`.

## Could be tooling

The consumer table is the obvious candidate. An AST walk would collect every attribute read of `batch_size` on a `Modeller` or an args namespace, resolve the enclosing function, and classify the use: passed as a `batch_size=` keyword to a buffer or dataset `loader` (a draw), compared against another quantity (a bound), divided into something (a cap or a rate), or accumulated (the gradient window). That gives the first two columns mechanically, and any consumer absent from this page appears as a row with no prose. A second check belongs at config-generation time: print the predicted step time and effective update size for a candidate $B$ from a fitted $a$ and $b$, so a battery's batch choice is stated rather than discovered.

## Sources

The code above, read at the stamped commit, and the batch block of the canonical config. Repo: docs/design/phase6_batch_sizer.md, docs/design/phase6_handoff.md, docs/design/width_and_length_scaling.md. Memory files located the code and were not used as evidence: project_phase6_batch_sizer_design, project_batch_sizer_rewrite_aug11, project_batch_sizer_retest_forces_the_z_cycle, project_util_policy_overstates_and_batch_is_not_the_lever, project_policy_rollout_is_dispatch_bound, project_real_runs_are_energy_bound, feedback_fracs_are_loss_weights_on_full_batches.
