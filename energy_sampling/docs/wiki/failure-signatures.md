# Failure signatures

*Drift: **C** (code-bound). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

A run that ends, stalls or degrades leaves two kinds of trace: text on stdout, and numeric series in wandb. This page maps the mechanism to the trace, so a trace in hand can be read back to the mechanism. The machinery behind each event is elsewhere: the start paths and the sidecar in [checkpoints-and-resume](checkpoints-and-resume.md), the OOM and step-time guards in [compute-guards](compute-guards.md), the batch the cut moves in [batch-size](batch-size.md), the retired-key inventory and the invariant rules in [config-validation](config-validation.md), the stage engine in [stage-protocol-engine](stage-protocol-engine.md), the fire/rewind path and its bars in [learning-rate](learning-rate.md), and the submission side that reads a job's stdout in [cluster-operations](cluster-operations.md).

`train.py::Modeller.train` opens a `wandb.init` context and every initialisation step, including `init_gfn`, runs *inside* it; everything up to the constructor call in the module's `__main__` block runs outside it. A failure's position relative to that line decides whether a wandb run exists at all.

The mapping, mechanism to signature:

| mechanism | what is emitted | site |
|---|---|---|
| config load fails: a retired key, or `eval_T` disagreeing with `integrator.T` | a `ValueError` traceback, beginning `"retired config keys found:"` and naming the migration command, or the `eval_T` text; no wandb run | `utils.py::get_train_args`, `utils.py::preflight_config`, `utils.py::_RETIRED_KEYS` |
| a violated config invariant | `"config invariants:"` and one indented line per violation; the run proceeds | `utils.py::_report_config_invariants`, `config_invariants.py::check` |
| GPU pre-flight refuses the card | the describe text plus `"Refusing to launch. Wait for the other run, ..."`, raised as `SystemExit`; no wandb run | `gpu_guard.py::require_free_gpu`, `gpu_guard.py::GPUBusy` |
| `cfg:epochs` already at or below the restored `step_ind` | zero loop iterations, then `final` written with buffers and `"Finished Training!"`; the wandb context exits normally, so the run state is finished | `train.py::Modeller.train`, `checkpointing.py::Checkpointer.save` |
| resume finds no buffer sidecar | `"No buffer sidecar found for {checkpoint_path} - buffers will initialize fresh"`; the run continues with empty buffers | `checkpointing.py::Checkpointer.load_buffers_for`, `.sidecar_candidates` |
| named checkpoint trained on another problem | a `ValueError` whose text is `"{config_key} checkpoint {path} was trained on a different problem than the current config solves"` then one `key: stored=...  current=...` line per differing field; raised inside the wandb context, so a run exists and ends crashed | `checkpointing.py::Checkpointer.assert_problem_match`, `.problem_mismatch_report` |
| auto-resume candidate mismatched | `"Checkpoint {path} exists but its stored problem definition doesn't match the current config - starting fresh instead."` plus the same field report; no raise | `checkpointing.py::Checkpointer.find_matching` |
| hard-failure bar or excursion tier fires | `lr_ctrl FIRE:` (excursion) or `lr_ctrl DISASTER:` (bar), then `"Divergence response: rewind #N + peak cut"`, then `lr_ctrl: fire #N -- rewound, and the rate is CUT to scale ...` | `controller.py::LRController.observe`, `train.py::Modeller.fire_loss_spike`, `controller.py::LRController.on_divergence` |
| a fire with no rewind target on disk | `"lr_ctrl WARNING: divergence at step ... but NO rewind target exists"`, and only the rate cut happens | `train.py::Modeller.fire_loss_spike` |
| rewind budget exhausted, the budget being `max(3, cfg:max_reloads_per_1k_steps * step_ind / 1000)` and disabled at a rate of 0 | `"UNRECOVERABLE at step ...: N rewinds (budget ... at .../1k steps) and the run keeps re-detonating"`, `wandb.run.summary['unrecoverable_abort']` set, `FrozenTrainingState` raised | `train.py::Modeller.fire_loss_spike` |
| a warm start whose restored weights give a non-finite gradient at step 0 | the non-finite-gradient print at step 0, then the no-rewind-target warning, since no `best` exists yet under this run name: only the rate is cut, and `gradnorm/nonfinite_steps` climbs from the first step | `train.py::Modeller.monitor_losses`, `.fire_loss_spike` |
| non-finite gradient | `"non-finite gradient at {step} (streak N) -> rewind + peak cut"`, printed on streak 1, 11, 21 | `train.py::Modeller.monitor_losses` |
| non-finite gradients past `cfg:nonfinite_abort_streak` | `"UNRECOVERABLE at step ...: N consecutive non-finite gradients (since step ...)"`, naming the stale `last_grad_norm_pre_clip`; `FrozenTrainingState` | `train.py::Modeller.step_loss` |
| CUDA OOM inside a train or eval step | `"Caught error during '{step_type}' step: ..."`, `"OOMED!"`, then on a train step `"Reducing batch size to N"` | `train.py::Modeller.handle_train_epoch_error` |
| OOM after the cut reaches 1 | `RuntimeError("Cascading OOM Failure")` | `train.py::Modeller.handle_train_epoch_error` |
| OOM during a stage transition's actions | the exception leaves `evaluation()` uncaught: no `Caught error` line, no cut, an ordinary traceback | `protocol.py::StageProtocol.advance`, `train.py::Modeller.evaluation` |
| MLIP forward raises | `"UMA error (attempt A of R)"` and the error text per retry; NaN is substituted for that batch's rows | `uma_utils.safe_predict_uma`, `uma_utils._crashed_energy` |
| three consecutive MLIP crashes | `RuntimeError("UMA failed N times in a row; refusing to keep substituting NaN, ...")` | `uma_utils.safe_predict_uma`, `uma_utils.MAX_CONSECUTIVE_CRASHES` |
| crashed rows reaching an eval | `"eval: n/m non-finite log-weights excluded from pooled Z estimates (check energy/uma_crash_rows)"` | `train.py::Modeller.fwd_eval_sampling` |
| gas-phase reference comes back non-finite | `RuntimeError` beginning `"gas-phase reference for mol_id(s) ... came back non-finite"` | `energies/molecular_crystal.py::MolecularCrystal.attach_gas_phase_reference` |
| a `stop` action on a stage's `on_exit` | `"protocol: stage '...' exit requested STOP -- no next stage; ..."`, then `"protocol stop honored at step N -- ending the run"`, then the normal `final` save; an ordinary transition instead prints `"protocol: stage 'A' -> 'B'"` | `protocol.py::StageProtocol._run_action`, `.advance`, `train.py::Modeller.train` |
| wandb publish fails | one `"wandb.log failed at step ..."` line on the first of a consecutive run, then silence; the total rides `run/log_failures`, and at `Modeller._MAX_LOG_FAILURES` consecutive failures a `RuntimeError` beginning `"wandb.log has failed N times in a row"` ends the run | `train.py::Modeller._log_metrics` |

## What the numeric series carry

Prints do not survive a run that is hard-killed: wandb uploads no console log for a run left in state `crashed`. The counters below ride the ten-step reporting clock into history and do survive, so several rows above are also readable as series after the fact.

The LR channel is `lr_ctrl/scale`, `lr_ctrl/divergences` and `lr_ctrl/moderate_fires` (`controller.py::LRController._emit`). `divergences` increments in `on_divergence` alone, reached by both the bar path and the non-finite-gradient path; the `UNRECOVERABLE` branch raises before it, so an aborting run's final fatal event is not in the counter. `moderate_fires` counts the excursion tier separately.

The batch channel is `Batch Size` beside `batch/oom_events`, `batch/oom_ceiling` and `batch/ceiling_expiries` (`train.py::Modeller.ten_step_reporting`). `oom_events` counts every allocation failure including eval ones; `oom_ceiling` moves only on a train-step OOM and reads 0 when no ceiling stands.

Non-finite steps accumulate in `gradnorm/nonfinite_steps`, MLIP crashes in `energy/uma_crash_calls` and `energy/uma_crash_rows`, drained per window by `uma_utils.drain_uma_phase_timing` and reported always, including as zero. The pooled eval dict carries `nonfinite_rows` and `nonfinite_frac` in the same way.

## Stalls, which emit nothing by themselves

A stage whose exit term names a metric nothing writes never advances and prints no message. `protocol.py::StageProtocol._advance_term` holds a streak when `_write_step` returns no new stamp, neither advancing nor resetting it, so the visible state is `protocol/exit_streak_<tag>` pinned at 0. The companion age separates the two zeros: `protocol/exit_age_<tag>` is the step distance to the last write of that metric, so it climbs without bound once writes stop, oscillates at the metric's own cadence when a bar is not met, and reads `-1` for a metric never written in this process.

The step bar is a bare `trange(init_step, self.args.epochs + 1)` carrying no loss, metric or stage (`train.py::Modeller.train`). It advances once per host iteration whatever happened inside it: identically through a rewind, through a run whose gradients are all non-finite, and through a stage that never exits. Every other line is event-driven, so between events a healthy run and a stalled one emit the same bar.

The occupancy series `gpu/util_recent` and `gpu/util_policy`, whose sensor is in [gpu-occupancy](gpu-occupancy.md), are means over `cfg:gpu_util_window_s` and `cfg:gpu_util_policy_window_s` of samples from a thread started before the energy function loads (`train.py::Modeller._start_gpu_util_thread`), so they include initialisation. The number a scheduler judges is an out-of-process one, wandb's `system.gpu.0.gpu` or an `nvidia-smi` sidecar, sampled on a wall clock that knows nothing about the loop. The scheduler-side reading is in [cluster-operations](cluster-operations.md).

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:epochs`, `cfg:checkpoint_name`, `cfg:continue_from_checkpoint`, `cfg:eval_period`, `cfg:rewind_tolerance`, `cfg:max_reloads_per_1k_steps`, `cfg:nonfinite_abort_streak`, `cfg:anomaly_detection`, `cfg:oom_batch_shrink_factor`, `cfg:oom_cooldown_steps`, `cfg:batch_oom_ceiling_retest_steps`, `cfg:cuda_memory_fraction`, `cfg:gpu_util_window_s`, `cfg:gpu_util_policy_window_s`, `cfg:lr_control.fire_cooldown_steps`, `cfg:lr_control.fire_cut_factor`, `cfg:stage.exit`, `cfg:stage.on_exit`.

Code: `train.py::Modeller.train`, `.init_gfn`, `.monitor_losses`, `.step_loss`, `.fire_loss_spike`, `.handle_train_epoch_error`, `.evaluation`, `.fwd_eval_sampling`, `.ten_step_reporting`, `._log_metrics`, `._start_gpu_util_thread`; `train.py::FrozenTrainingState`; `checkpointing.py::Checkpointer.find_matching`, `.assert_problem_match`, `.problem_mismatch_report`, `.load_buffers_for`, `.sidecar_candidates`, `.restore_buffers`, `.save`; `controller.py::LRController.observe`, `.on_divergence`, `._emit`; `protocol.py::StageProtocol.advance`, `.maybe_advance`, `._advance_term`, `._exit_tick`, `._write_step`, `._run_action`, `.report`; `utils.py::get_train_args`, `preflight_config`, `_RETIRED_KEYS`, `_report_config_invariants`; `gpu_guard.py::require_free_gpu`, `GPUBusy`; `energies/molecular_crystal.py::MolecularCrystal.attach_gas_phase_reference`, `.drain_energy_timing`; `mxtaltools/mlip_interfaces/uma_utils.py::safe_predict_uma`, `_crashed_energy`, `drain_uma_phase_timing`, `MAX_CONSECUTIVE_CRASHES`.

## Could be tooling

The table is a manifest, and it drifts the moment a message is edited. An AST walk could collect every `print` and `raise` reachable from `Modeller.train`, keep each format string's literal prefix, and emit the first two columns mechanically: a message with no row here appears as a row with no prose, and a quoted phrase matching no literal fails. The reader's half is a log scanner that matches those prefixes against a job's stdout, places each hit on the step axis beside `Batch Size`, `lr_ctrl/divergences`, `gradnorm/nonfinite_steps` and `energy/uma_crash_rows`, and prints the terminal state and last step. The stalls are what a scanner adds most to, since they are defined by the absence of a line: an exit streak at 0 with an age exceeding a multiple of `cfg:eval_period`, a `gradnorm/nonfinite_steps` slope of one per step, a last history row far behind the last checkpoint.

## Sources

The code above, read at the stamped commit. Nine memory files located the code and were not used as evidence: project_epochs_is_absolute_and_resumes_can_start_past_it, project_best_pt_without_phase1_exit_is_fatal, project_lj_stamp_lost_on_buffer_restore, project_lr_fire_attribution, project_transition_oom_is_fatal, project_uma_crash_returns_zeros_in_production, feedback_live_log_looks_like_a_dead_job, project_low_util_kill_is_node_contention_gpu_busy_constant, project_local_paths_pass_local_preflight.
