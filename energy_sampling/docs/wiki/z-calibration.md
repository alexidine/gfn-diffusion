# Z calibration

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

The learned log Z is a parameter, or conditionally a head, that TB's gradient moves like any other. This page is about every *other* mechanism that places or moves it: the closed-form estimators, the level fill, the Z-only servo, and the stage-entry bootstrap. The TB residual and the identities these estimators estimate are on [trajectory-balance](trajectory-balance.md).

The object written is `flow_model.scalar` and its EMA twin. `models/gfn.py::GFN.init_flow_model` builds it as `models/architectures.py::LearnableScalar` on an unconditional run, and on a conditional run over a condition set of size one (`scalar_flow`, derived from that set); otherwise the conditional head is an `mxtaltools.models.modules.components.scalarMLP` over the condition embedding, and `full_flow` makes it a `scalarMLP` over the state and time embeddings.

## The estimators

Four different estimands are computed from a batch of per-trajectory log weights $\log w_i = \log p_B + \log r - \log p_F$.

**The Huber fixed point.** `gflownet_losses.py::winsorized_z_root(logw, beta)` returns `(root, se, frac_unclipped)`. The root is the $z$ at which

$$\frac{dL}{dz} = \operatorname{mean}_i \operatorname{clip}(z - \log w_i, \pm\beta) = 0,$$

found by bisection over a fixed iteration count inside the bracket $[\min \log w - \beta, \max \log w + \beta]$. It is a location estimator winsorized at $\beta$; as $\beta \to \infty$ it becomes the plain mean of $\log w$. Its standard error is the M-estimator sandwich

$$se = \frac{\operatorname{rms}(\operatorname{clip}(\text{resid}, \pm\beta))}{\sqrt{B}\,f_{\text{unclipped}}},$$

whose denominator's $f_{\text{unclipped}}$ is $d^2L/dz^2$, the loss's curvature in $z$, bounded above by 1. $se$ is $+\infty$ when no row is unclipped, and a degenerate input raises rather than returning a nan root. `utils.py::quick_tb_stats` publishes the three returns as `huber_z`, `huber_z_se` and `huber_z_frac_unclipped`, and `gflownet_losses.py::batch_root_z` wraps the root as a detached level under `tb_z_source: 'batch_root'`.

**The Jensen mean.** $\mathbb{E}[\log w]$, reported as `jensen_z` by `utils.py::quick_tb_stats` and held per condition as `buffer.py::ConditionLogZTracker.ema_logw` (rank-trimmed by `cfg:condition_log_z.trim_frac` and EMA'd over `cfg:condition_log_z.half_life_visits`).

**Log-mean-exp.** `utils.py::logmeanexp` of $\log w$, reported as `emp_z` and held per condition as `ema_log_z_emp`. `ConditionLogZTracker.lookup` returns `ema_logw`, not `ema_log_z_emp`.

**The tracker's residual EMAs.** `ConditionLogZTracker.update_z_residual` keeps two per-condition streams of $\log w - \log Z_{\text{learned}}$: `z_grad_ema`, clipped at `clip_beta`, and `z_bias_ema`, unclipped. `clip_beta` comes once from the base forward coefficients, not from the live stage.

Pooled beside these, `quick_tb_stats` reports `tb_resid_clipped`, the signed batch mean of the clipped residual, which is $dL/dz$ up to the $\beta$ scale.

## The level fill

`train.py::Modeller._stash_z_fill_logw` keeps the forward branch's own $\log w$ vector for one step. It arms only where `train.py::Modeller._z_fill_head_is_fillable` passes: the head must not be `full_flow` and must carry a `.scalar` attribute (the test is on the head, not on `gfn_model.conditional`), and either the forward branch carries `freeze_policy` or it takes `tb_z_source: 'batch_root'` under a rollout cadence. The second clause is inert at this commit; nothing sets `batch_root`.

`train.py::Modeller.z_level_fill` consumes that stash once, computes `winsorized_z_root` at `cfg:fwd_loss_coeffs.beta`, and reports `z_fill/gap`, `z_fill/se`, `z_fill/frac_unclipped` and `z_fill/n`; a `ValueError` from the root counts `z_fill/bad_batch`. `cfg:z_calibration.fill_threshold` at or below 0 returns first, and `cfg:z_calibration.fill_cooldown_steps` train steps since the last applied fill gate both modes ahead of the mode branch.

**`fill_mode: snap`** is a gated overwrite. The gap must exceed `cfg:z_calibration.fill_threshold` and also `cfg:z_calibration.fill_se` standard errors, which covers the $se = +\infty$ case; a block counts `z_fill/blocked_by_se`. Adam's flow moments are always dropped.

**`fill_mode: absorb`** is a one-dimensional Kalman filter with state $(Z, P)$. With $q$ = `cfg:z_calibration.fill_process_var` and $\Delta$ steps since the last applied fill,

$$P_{\text{pred}} = P + q\Delta, \qquad K = \frac{P_{\text{pred}}}{P_{\text{pred}} + se^2}, \qquad \log Z \mathrel{+}= K(\text{root} - \log Z), \qquad P \leftarrow \frac{P_{\text{pred}}\, se^2}{P_{\text{pred}} + se^2}.$$

The first measurement is taken whole and leaves $P = se^2$. Only a non-finite $se$ blocks. Adam's flow moments are dropped when $|dz|$ exceeds `cfg:z_calibration.fill_moment_reset`. `z_fill/K`, `z_fill/P` and `z_fill/dz` are reported. Under `absorb`, `fill_threshold` acts as an enable switch, `fill_se` never gates as a bar, and `fill_cooldown_steps` ships at 0.

Either mode increments `z_fill/fired` and prints a line naming mode, source, old and new level, root, gap, $se$ and unclipped fraction.

A fill moves the numeraire, so it re-signs every residual at once. It shifts the tracker's two level EMAs at the fill site, exactly for `z_bias_ema` and clamped for `z_grad_ema`, and drops Adam's flow moments as above. It cannot shift `fwd/tb_resid_clipped` in the metric tracker, which keeps reporting the pre-fill level for its own time constant.

**Where it runs.** `train.py::Modeller.fused_train_step` stashes after the forward rollout and, when the forward loss is neither active nor a `fwd_z_sidecar`, fills there, before the backward and replay losses are built; the post-step call in `train.py::Modeller.train` then finds the stash consumed. With a live forward loss, log Z is already in the forward graph and the post-step call acts instead. Either site precedes `z_calibration_tick`.

`train.py::Modeller._eval_z_fill` routes the eval rollout's $\log w$ through the same actuator under `cfg:z_calibration.fill_from_eval` (`off`, `report`, `fill`), refusing on an unfillable head, under `cfg:temperature_conditioning`, on `cfg:eval_T` differing from `cfg:integrator.T`, or on `cfg:ema_decay` set, each counted as `z_fill/eval_refused_{head,temperature,grid,ema}`.

## The Z-only calibration servo

`train.py::Modeller.z_calibration_tick` takes extra Z-only optimizer steps on top of the training step, frequency-modulated and never size-modulated. It returns immediately unless `cfg:stage.flags.z_calibration` is true, and also mid-accumulation (`fused_accum_count`), in a `scramble_conditions` stage, when no owning optimizer (`fused` on a fused step, otherwise `flow`) exists, on a non-finite sensor, when the sensor sits at or below `threshold`, and when `mode: regression` has no cached embeddings.

Sensors, all compared against `cfg:z_calibration.threshold`: `grad_rms` is `ConditionLogZTracker.rms_z_grad`, `rms` is `rms_z_bias`, `worst` is `worst_z_bias` at `cfg:z_calibration.sensor_quantile`, and anything else falls through to `pooled`, $|$EMA `fwd/tb_resid_clipped`$|$. The first three read the tracker, returning 0.0 while it is cold. Steps per train step are

$$n \sim \min\!\left(\text{gain}\cdot\left(\frac{\text{sensor}}{\text{threshold}} - 1\right),\ \text{max\_steps\_per\_step}\right),$$

Bernoulli on the fractional part, and in `rollout` or `replay` mode the loop breaks once that step's own fresh reading falls under `threshold` $\times$ `cfg:z_calibration.grace` (`z_cal/early_out`).

Three step bodies exist. `train.py::Modeller._z_rollout_step` runs a fresh forward rollout with `freeze_policy` forced to 1 and `z_level`, `z_var`, `emp_z`, `emp_z_persistent` zeroed, verifies once that no gradient reached outside `flow_model` and raises otherwise, and admits its rows to the replay buffer under `ORIGIN_ROLLOUT`. `train.py::Modeller._z_replay_step` takes the same gradient over a read-only replay draw; it returns without acting on an empty buffer and raises unless `cfg:buffers.replay_buffer.prioritise.enabled` is on, the message naming scored admission as what would otherwise set the level. `train.py::Modeller._z_calibration_step` is weighted least squares of the head onto `ConditionLogZTracker.calibration_targets` over the embeddings cached by `train.py::Modeller._stash_z_cal_cache`, with `cfg:z_calibration.holdout_modulus` conditions excluded and reported as `z_cal/holdout_rms` against `z_cal/train_rms`. That cache fills only on a stage declaring the flag and under `mode: regression`, and bails when `_z_cal_embedding` is None, which is the unconditional case, so `regression` is unreachable there.

Each step body processes a full `cfg:batch_size`, and `rollout` mode calls the energy function per step taken. The loop sits inside the step timing window, is timed into `z_cal/seconds`, and `_z_cal_rollouts` enters the batch sizer's per-step work as `attempted_batch * (1 + _z_cal_rollouts)`.

The `rms` sensor's docstring in `z_calibration_tick` calls its reading "UNCLIPPED level dispersion"; `rms_z_bias` computes an RMS over conditions of the signed level error itself, and the canonical config's comment on `cfg:z_calibration.sensor` states that all four sensors read on a single-condition run.

## bootstrap_z

`bootstrap_z` is a stage `on_enter` action with three spellings (`protocol.py::ACTIONS`). `bootstrap_z:rollout[:n]` calls `train.py::Modeller.bootstrap_z_by_rollout`: one forward eval-sampling draw of `n` or `cfg:eval_num_samples` samples with `side_effects=False`, $\log w$ built from it, the absorber re-opened (`P := None`) and the cooldown cleared so the fill takes the whole gap, routed through `z_level_fill` with `source='boot'`, and the same rollout admitted to the replay buffer under `ORIGIN_BOOTSTRAP`. Fewer than two finite rows leaves log Z at its checkpoint value and says so.

Plain `bootstrap_z` and `bootstrap_z:train_conditioner` reach `protocol.py::StageProtocol._bootstrap_z`, which does not use the Huber root. On a non-`full_flow`, non-conditional model it fills the scalar from `eval_fwd/jensen_z` when eval metrics carry it, otherwise from the mean of `ConditionLogZTracker.ema_logw` over conditions past `min_visits`; with neither it raises, naming both sources. Otherwise it calls `train.py::Modeller.bootstrap_log_z`.

## The conditional case

Conditionally the level is a field: `flow_model` is a `scalarMLP` over the detached condition embedding, so there is no `.scalar` to write and `_z_fill_head_is_fillable` returns False. The fill and `bootstrap_z_by_rollout` are refused on such a head, and an eval fill counts `z_fill/eval_refused_head`. The exception is a condition set of size one, where `scalar_flow` gives back a `LearnableScalar` and the fill is available again.

What places Z(c) instead: `buffer.py::ConditionLogZTracker` keeps `ema_logw`, `ema_log_z_emp`, `count`, `effective_count` and the two residual EMAs per integer `condition_id`, and is checkpointed under `condition_log_z`. `train.py::Modeller.bootstrap_log_z` fits the head onto `ema_logw`, weighted by `trust_c`, through a fresh local Adam with a holdout and a per-condition coverage gate, and never onto `ema_log_z_emp`. In the loss, `cfg:*_loss_coeffs.emp_z` regresses the head through `gflownet_losses.py::emp_Z` onto each condition group's estimate from `gflownet_losses.py::condition_grouped_empirical_z`, Jensen-mean or log-mean-exp following the active VarGrad branch, singleton groups masked out; `emp_z_persistent` regresses onto the tracker instead; and `gflownet_losses.py::z_level_loss` is the condition-grouped squared level error against detached $\log w$. On the conditional stage that forward loss enters via `cfg:stage.fwd_z_sidecar` at weight 1, outside the frac mix.

## The stage flag and the cadence refusals

`z_calibration` is a stage flag (`protocol.py::STAGE_FLAGS`). `protocol.py::Stage` refuses `fwd_rollout_every > 0` beside `flags.z_calibration: true` at load. `config_invariants.py::fwd_rollout_cadence_is_well_formed` repeats that refusal on the audit path and adds a second: with any active stage running `fwd_rollout_every > 0` and not declaring `fwd_z_sidecar`, `cfg:z_calibration.fill_threshold` must be positive ([rollout-cadence-and-dose](rollout-cadence-and-dose.md)).

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:z_calibration.{mode, sensor, sensor_quantile, threshold, grace, gain, max_steps_per_step, fill_mode, fill_process_var, fill_moment_reset, fill_from_eval, fill_threshold, fill_se, fill_cooldown_steps, min_visits, freshness_half_life_steps, se2_floor, holdout_modulus}`, `cfg:stage.flags.z_calibration`, `cfg:stage.on_enter`, `cfg:stage.fwd_rollout_every`, `cfg:stage.z_pin_rollout_every`, `cfg:stage.fwd_z_sidecar`, `cfg:condition_log_z.{min_visits, half_life_visits, trim_frac}`, `cfg:*_loss_coeffs.{beta, freeze_policy, emp_z, emp_z_persistent, tb_z_source}`, `cfg:buffers.replay_buffer.prioritise.enabled`, `cfg:eval_num_samples`, `cfg:eval_T`, `cfg:ema_decay`, `cfg:temperature_conditioning`.

Code: `gflownet_losses.py::winsorized_z_root`, `batch_root_z`, `emp_Z`, `z_level_loss`, `condition_grouped_empirical_z`; `utils.py::quick_tb_stats`, `utils.py::logmeanexp`; `train.py::Modeller.z_level_fill`, `._stash_z_fill_logw`, `._z_fill_head_is_fillable`, `._eval_z_fill`, `.z_calibration_tick`, `._z_rollout_step`, `._z_replay_step`, `._z_calibration_step`, `._stash_z_cal_cache`, `.bootstrap_z_by_rollout`, `.bootstrap_log_z`, `.fused_train_step`, `.train`; `buffer.py::ConditionLogZTracker.update_z_residual`, `.rms_z_grad`, `.rms_z_bias`, `.worst_z_bias`, `.calibration_targets`, `.lookup`; `config_invariants.py::fwd_rollout_cadence_is_well_formed`; `protocol.py::Stage`, `protocol.py::StageProtocol._bootstrap_z`, `protocol.py::STAGE_FLAGS`, `ACTIONS`; `models/gfn.py::GFN.init_flow_model`; `models/architectures.py::LearnableScalar`.

## Could be tooling

Two of these checks are config-time. `_bootstrap_z` raises when a stage entering it has neither an eval stream nor a visited tracker, which is decidable before submission from the protocol's skip chain and the seed's `load_weights_only`. The inert-key surface is decidable the same way: under `fill_mode: absorb` three fill keys do not gate, and with `flags.z_calibration` false on every stage the servo's keys and the regression sub-block are unread, so a generator could print which keys a config consumes. Third, `fill_process_var` has a closed-form estimator, $q = (\operatorname{var}(\Delta\text{root}) - 2\,\overline{se^2})/N$, whose inputs `z_fill/root` and `z_fill/se` are already logged.

## Sources

Repo: docs/design/z_calibration_capabilities.md, docs/design/z_fill_process_variance.md, the code above at the stamped commit, and the `z_calibration`, `condition_log_z` and protocol blocks of configs/mk_dev.yaml. Six memory files (project_z_calibration_interspersed, project_z_calibration_regression_mode_is_dead_code, project_zcal_cost_multiplier_at_transitions, project_bootstrap_z_anchors_on_tracker_ema_logw, project_z_convergence_gradient_signal, project_skip_if_kills_bootstrap_z) located the code and were not used as evidence.
