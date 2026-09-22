# Run reading

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

A run publishes one flat stream of named scalars to wandb, from two clocks: a ten-step report (`train.py::Modeller.ten_step_reporting`) and an eval (`::Modeller.log_metrics`, reached from `.evaluation`). This page defines those namespaces, the EMA behind three of them, the held-out quantities, the relations the code states between keys, the in-repo package that reads a run, and the figure budget. The coverage family is defined on [under-coverage-metric-family](under-coverage-metric-family.md), `z_fill/` and `z_cal/` on [z-calibration](z-calibration.md), `batch/` on [batch-size](batch-size.md), `gpu/` on [gpu-occupancy](gpu-occupancy.md).

## The namespaces

The table below sets each published namespace against the quantities it holds and the writer that emits it.

| namespace | what it holds (examples) | writer |
|---|---|---|
| `fwd/`, `bwd/`, `replay/` | per-branch EMA of `utils.py::quick_tb_stats` and that branch's loss-dict scalars: `loss`, `log_Z_learned`, `step_var` | `Modeller._update_rolling` into `utils.py::MetricTracker.update` |
| `eval_fwd/`, `eval_bwd/`, `eval_test/` | the same `quick_tb_stats` key set on an eval batch: train conditions, the backward stream, held-out conditions | `Modeller._eval_conditional_stats` |
| `gates/` | values published for exit terms: `delta_worst`, `progress_done`, `mle_flat` | `protocol.py::StageProtocol.publish_gate` |
| `protocol/` | stage-engine state: boost, per-rule bars and elevations, exit streaks and ages | `::StageProtocol.report` |
| `lr_ctrl/` | `scale`, `divergences`, `moderate_fires`, capped and floored group counts | `controller.py::LRController.report` |
| `z_fill/`, `z_cal/` | the fill's root, gap and refusals; the calibration tick's counts, losses, seconds | `Modeller`'s z-calibration paths |
| `rollout/` | `rate`, `every_eff`, `steps_since`, one `trigger_<reason>` count per reason | `ten_step_reporting` |
| `gpu/`, `vram/` | `util_recent`, `util_policy`; live, reserved and peak megabytes | `ten_step_reporting`, `Modeller.vram_metrics` |
| `batch/` | `accum_target`, `med_step_s`, `sps_rung`, OOM counters, the sizer's phase | `ten_step_reporting` |
| `tracker/` | cross-visit reductions over the condition library: `tb_err_rms`, `z_grad_worst`, `z_bias_var` | `buffer.py::ConditionLogZTracker` |
| `zmatch/` | `delta_worst`, `delta_n_trusted`, `fwd_level` | `::ConditionLogZTracker.delta_stats`, `.pooled_levels` |
| `Cond <name> *` | per-condition breakdown of a per-sample 0/1 indicator: `Failing Frac`, `Worst`, `Bar` | `Modeller.log_condition_fraction` over `utils.py::per_condition_fraction` |

`Cond` is a space-separated prefix, not a slash namespace, and takes an optional namespace in front of it (`eval_test/Cond Reasonable Worst`); its `Bar` key is unprefixed, so both streams score against one series. The same two writers also emit `pooled/`, `energy/`, `gradnorm/`, `gradclip/`, `raycal/`, `cond_draw/`, `run/`, and the settings blocks `energy_func/` and `loss_coeffs/`.

There is no `eval_gap/` namespace: `Modeller.log_test_metrics` records that it was exactly `eval_fwd` minus `eval_test` on six keys already logged, and was removed; the only surviving use of the token is `z_fill/eval_gap`, an unrelated key. Two further absences are stated in code: a stage's position is the top-level `phase`, one-based, and `protocol/stage_index` does not exist; and `bwd/under_coverage_wcen` is logged as `bwd/under_coverage`.

## The tracker and its EMA

`train.py::Modeller.__init__` constructs `MetricTracker(period=100)`. `MetricTracker.update` is keyed by `(direction, name)`, takes `dt` as the step gap since that direction's last update, and forms `alpha = 1 - exp(-dt/period)`. A non-finite value is skipped, leaving the value and the `written_at` stamp untouched; `.written_step` reports that stamp. The stamp is not in `state_dict`, so after a resume every key reads as never freshly written. `.rebase` resets the per-direction clock to the restored step. `snapshot(changed_only=True)`, what `ten_step_reporting` publishes, clears the changed set on read.

Only `fwd`, `bwd` and `replay` reach the tracker. `protocol.py::StageProtocol._resolve` reads a `dir/name` reference off the tracker, so only those three resolve to a value; it also accepts `gates/name`, and `eval/name` inside `maybe_advance`, where the eval metrics dict is available. The `tracker/` family is published and not resolvable.

The `quick_tb_stats` docstring states the criterion for its control family: its members are per-sample means, never ratios. `buffer.py` states the algebra a ratio fails, $\mathbb{E}[A/B] \ne \mathbb{E}[A]/\mathbb{E}[B]$. Against that criterion the stream divides three ways. Per-sample means: `tb_resid`, `tb_resid_clipped`, `jensen_z`. Square roots of per-sample means, not themselves means because the root is concave: `tb_err`, `scatter_err`, `cond_tb_err`, `logw_std`, the coverage ladder. Ratios and quantiles: `r2`, `slope_err` and `intercept_err` (a covariance over a variance), `ess_frac`, the `replay/` absorption family, and the across-condition quantiles `tb_err_worst`, `z_grad_worst`.

`Modeller._per_step_probe`, armed by `cfg:per_step_probe_steps`, writes raw per-step values to an `.npz` beside the checkpoints with no EMA and no tracker.

## Held-out quantities

`eval_test/` is the eval protocol re-run against the conditions in `cfg:test_molecules_path`, at `cfg:test_eval_num_samples` samples, through `Modeller.log_test_metrics`. The sampling call passes `side_effects=False`, so those conditions reach neither `condition_log_z`, the anchor buffer, nor prior-buffer churn, and nothing in the block feeds a gate, a controller or a loss. With `test_molecules_path` null every `eval_test/` key is absent. The block also publishes `eval_test/Reasonable Sample Fraction` and its `Cond Reasonable` family; the non-thermal family is not, being scored against a per-condition minimum `side_effects=False` never writes.

`replay/val_*` is the replay buffer's own split. `train.py::_val_flags` marks each admission batch Bernoulli(`cfg:buffers.replay_buffer.val_frac`), once at admission, and `Modeller._replay_val_stats` re-runs the replay loss on a capped subset of those rows (`cfg:buffers.replay_buffer.val_cap`, also clipped to the live batch size), unweighted and with the Z tracker read-only, scoring stored trajectories through the prebuilt path so no energy call is made. It writes `val_loss`; `val_gap`, mean Huber loss val minus train, in nats squared; `val_gap_nats`, median absolute residual val minus train in nats, the train side using the importance-weighted median; `val_gap_nats_se`, 1.858 times the val side's MAD over $\sqrt{n}$; `val_n`; and `val_skips`, which a CUDA OOM in the probe sets to 1.0.

## Relations the code states

- $\mathbb{E}[r^2] = \mathrm{mean}(r)^2 + \mathrm{Var}(r)$: the level and spread halves of one square, `z_grad_worst` being the level half and the excess of `tb_err_worst` over it the spread half.
- With `condition_id=None`, `cond_tb_err` and `tb_err_worst` reduce exactly to `tb_err`, `z_grad_worst` to $|$`tb_resid_clipped`$|$, and `logw_std_within` is omitted outright.
- With the reward ramp unconfigured, `under_coverage` equals `under_coverage_uniform` and `relative_under_wcen` equals `relative_under`; `ramp_ess_frac` is omitted rather than set to 1.0.
- `rollout/rate` and `rollout/every_eff` are reciprocals of one number, both cumulative since stage entry: energy calls per training step, and steps per rollout.
- `replay/absorbed_frac` $= 1 -$ `replay/resid_vs_intake`; `replay/lambda_tau` $= -\log($`resid_vs_intake`$)$.
- `updates_per_sec` $=$ `samples_per_sec` $/$ `batch/accum_target`; `z_cal/frac_of_step` and `energy/frac_of_step` share the report window's seconds as denominator.
- `tracker/tb_err_worst` reduces across the condition library over visits, `fwd/tb_err_worst` within one batch: they share a suffix and the quantile `cfg:conditional_worst_quantile`, not a population.

Settings are emitted on change only, not as series: `Modeller._log_setting` for a bar, the `dump_numeric` helper inside `log_metrics` for the energy-function constants and the three loss-coefficient blocks, both against `Modeller._settings_log_cache`.

## The analysis package

`analysis/` reads a run from outside the training process; `python -m analysis <spec>` resolves `newest`, a run directory, a run id, a display name or a tag. `analysis/pull.py::pull` tries the local `.wandb` datastore first and the cloud API second, caches pickled runs under the system temp directory, and raises `EmptyPull` rather than returning an empty result. Its docstrings mark four local-parse hazards (an empty `item.key` with the name in `nested_key`, rows without `_step`, a partially written final record, a zero-byte file on a just-restarted run) and one cloud hazard: `scan_history(keys=...)` returns only rows containing every requested key, so the package streams all columns and filters client-side. Local directories are ordered by the launch timestamp in the name, not by mtime.

`analysis/keys.py` is the package's coupling point to the training code and holds every metric-name and config-key literal. Its `Route` enum has four members (`TB_UNCONDITIONAL`, `VARGRAD_CONDITIONAL`, `MLE_PRIOR`, `UNKNOWN`), detected from the base `<mode>_loss_coeffs_*` blocks overlaid with the stage's overrides and filtered to the branches that stage's `train_mode` runs. Its `KeyState` enum has three (`LIVE`, `ABSENT`, and `NA_ROUTE` for a key logged and populated that does not track on this route). `resolve` tests `NA_ROUTE` before presence, reports a fuzzy match as a rename, and reports a name available under several namespaces as ambiguous. `EMA_PREFIXES` is `('tracker/',)`, for which trend significance is suppressed.

`analysis/features.py` provides a Theil-Sen slope, a detrended-ACF oscillation reading and a doubling time. `analysis/checks.py` holds `check_r2` (declared mechanisms against the trace that proves each ran), `check_r14` (dead-sensor conditions), `check_confounds` and `check_r11`. Its docstring names two failure modes the checks exclude: a silent pass, for which a check returns `CheckResult.not_run(reason)` rather than an empty finding list, and collapsing `NA_ROUTE` into `ABSENT`.

## Figure budget

`eval/evaluations.py` names three compaction mechanisms, in its own order of preference: `_hist_bar` replaces a raw per-sample histogram with a pre-binned `go.Bar`; `_thin_idx` thins an overplotted scatter to `SCATTER_MAX_POINTS` = 4000, summary statistics being computed on the full sample first; `adjust_fig_filesize` rasterises anything above `FIG_SIZE_LIMIT_MB` = 0.25 except the `KEEP_INTERACTIVE` names, `TB Parity Plot` and `Lattice Latents Distribution`. `FUNNEL_MAX_POINTS` = 2000 thins the density funnel at its call site, that figure being built inside mxtaltools and so never passing the compaction. `CONDITION_SCATTER_FIGS` is `False`, gating the splom and calibration scatters out of `condition_tracker_figs`. Figures ride `cfg:figs_period`, a multiple of `cfg:eval_period`, and every block runs inside `fig_guard`, which prints and continues.

## Where a conformer run logs

`train.py::Modeller.train` calls `wandb.init(project="GFN Energy")`. `conformer_modeller.py::ConformerModeller` subclasses `Modeller`, makes no `wandb` call of its own and defines no project hook, so its runs land in that project. The older standalone loop `train_conformer.py` sets `WANDB_PROJECT = "GFN Conformers"` in its own `wandb.init`; `analysis/pull.py` carries both as `DEFAULT_PROJECT` and `CONFORMER_PROJECT`.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:figs_period`, `cfg:eval_period`, `cfg:test_molecules_path`, `cfg:test_eval_num_samples`, `cfg:conditional_worst_quantile`, `cfg:per_step_probe_steps`, `cfg:reasonable_cond_bar`, `cfg:buffers.replay_buffer.val_frac`, `cfg:buffers.replay_buffer.val_cap`.

Code: `train.py::Modeller.ten_step_reporting`, `.log_metrics`, `.evaluation`, `._update_rolling`, `._per_step_probe`, `._eval_conditional_stats`, `.log_test_metrics`, `.log_condition_fraction`, `._log_setting`, `._replay_val_stats`, `.vram_metrics`, `.train`; `train.py::_val_flags`; `utils.py::MetricTracker.update`, `.written_step`, `.snapshot`, `.rebase`, `::quick_tb_stats`, `::per_condition_fraction`; `buffer.py::ConditionLogZTracker.delta_stats`, `.pooled_levels`; `protocol.py::StageProtocol.report`, `._resolve`, `.publish_gate`; `controller.py::LRController.report`; `eval/evaluations.py::_hist_bar`, `::_thin_idx`, `::adjust_fig_filesize`, `::condition_tracker_figs`, `::fig_guard`; `analysis/keys.py::resolve`; `analysis/pull.py::pull`; `analysis/checks.py::check_r2`, `::check_r14`, `::check_confounds`, `::check_r11`; `analysis/features.py::theil_sen`; `conformer_modeller.py::ConformerModeller`; `train_conformer.py`.

## Could be tooling

The key registry is generatable: every metric name reaches wandb through a bounded set of writers, and an AST walk over them yields each key's namespace, writer and clock, cross-checkable against the literals in `analysis/keys.py`.

## Sources

The code above, read at the stamped commit, and `configs/mk_dev.yaml`. Repo: `docs/module_metrics.md`, `docs/reading_runs.md`. Memory files located the code and were not used as evidence: reference_analysis_package, reference_wandb_run_diagnostics, reference_local_wandb_reading, project_metric_audit_duplicates, project_r2_ratio_metrics_dont_ema.
