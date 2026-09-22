# Conditional route

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

A conditional run samples from a family of targets indexed by a condition $c$. The policy takes $c$ through an embedding, the normaliser becomes $\log Z(c)$, each draw selects which conditions it visits, and each pooled statistic either pools over conditions or groups within them. Left out and named: the VarGrad objective and its condition grouping, [vargrad](vargrad.md); the cross-branch pooled term and the rollout-to-replay coupling, [on-policy-and-off-policy-training](on-policy-and-off-policy-training.md); the parametric prior densities, [prior-density-models](prior-density-models.md); condition sets, anchors and the encoder, [molecule-conditions-and-anchors](molecule-conditions-and-anchors.md) and [molecule-encoder](molecule-encoder.md).

## The condition channel into the policy

`train.py::Modeller._build_gfn_config` sets `conditional` to the disjunction of the six conditioning flags (`cfg:temperature_conditioning`, `cfg:molecule_conditioning`, `cfg:sg_conditioning`, `cfg:zp_conditioning`, `cfg:vector_conditioning`, `cfg:embedding_conditioning`) and `conditions_type` to `molecule` under the second, `vector` otherwise. `utils.py::get_problem_definition` stamps `vec_cond` always and `emb_cond` when on, and feeds the problem hash, so a vector-conditional problem and an unconditional one over the same prior file are distinct identities.

`models/gfn.py::GFN.init_conditioner` builds a `scalarMLP` from `cfg:vector_conditioning_dim` to `cfg:model.condition_embedding_dim` under `vector` and a `VectorMoleculeGraphModel` under `molecule`; `GFN.get_condition_embedding` is the single entry point for both.

The embedding is computed once per trajectory and passed down to three consumers. The trunk call `s_model(expanded_state, condition_embedding)` inside `GFN._forward_kernel` and `GFN._pb_net` is upstream of both policies. `GFN._condition_flow` reads it for the flow head as `self.flow_model(condition_embedding.detach())`, so a Z-side gradient reaches the flow head's own parameters and no others, `conditions_embedding_model` sitting in the policy parameter groups. And `freeze_policy` detaches the embedding at source in `GFN.get_traj_fwd`, `.get_traj_bwd` and the rollout path.

### The scrambled phase-1 prior

`GFN._maybe_scramble_condition_embedding` runs the conditioner on the true pairing, then permutes the rows of its detached output in tiles of `scramble_condition_tiles` at the conditioner-to-trunk seam, so the trunk is trained against a mismatched embedding while the per-sample reward, condition and `condition_id` tensors stay correctly paired. It fires when the stage declares `scramble_conditions` (one of `protocol.py::STAGE_FLAGS`) and `Modeller.scramble_applicable` holds, which requires a conditional model on `conditions_type == 'vector'`. The tile size is the caller's repeats grouping, so same-terminal rollouts keep sharing one scrambled condition.

## Condition identity

`Modeller.init_identifiers` collects `identifier` strings across `mol_dataset`, `prior_dataset` and `test_mol_dataset` and assigns each a dense integer `mol_id`, attached to every resident batch. `energies/molecular_crystal.py::MolecularCrystal.condition_samples` combines `mol_id` with dense-local space-group and $Z'$ indices in mixed radix to give `condition_id`, samples once per `repeats`-sized group, and attaches `conditions` and `condition_id` to the batch, so both ride into any buffer it enters. Identity runs through the identifier string, never the condition vector, so conditions sharing an identifier share one tracker slot; `data_processing/generate_toy_prior.py::expand_condition_manifold` gives every manifold draw its own, `{base}_NNNN`, configured anchors keeping their names.

## Which conditions each draw visits

**Forward rollout.** `Modeller._rollout_condition_mode` reads `cfg:condition_log_z.rollout_condition_draw`: `iid` (uniform without replacement within a rollout), `cycle` (a shuffled pass over `mol_dataset`, reshuffled when spent, not checkpointed) or `under_drawn` ($p \propto (1+n)^{-\texttt{power}}$ on the condition's trainable replay rows $n$). A stage declaring `weighted_condition_sampling` instead draws by `Modeller.weighted_condition_sampling`, the tracker's per-condition fit-error measure; `config_invariants.py::condition_draw_problems` refuses that flag beside a non-`iid` draw.

**Forward eval.** `Modeller.fwd_eval_sampling` takes `next(dataset.loader(bsz, mode='graphs'))` on both the train and held-out pass, an unweighted draw over the dataset's rows. Uniformity over conditions then follows the number of rows each condition holds in the file; the eval path applies no condition weighting of its own.

**Backward.** Under `cfg:stage.bwd_sampling_mode` `prior`, `Modeller.draw_bwd_sample` draws terminal states from the prior buffer, whose occupancy is shaped by admission, loss-based churn and anchor top-ups. Two modifiers sit on it: `condition_block_m`, armed only while `train.py::_runs_grouped_vargrad` holds for the branch, blocks the draw by condition; the stage flag `weighted_bwd_sampling` sets `weighted=True` at `cfg:buffers.prior_buffer.weighted_bwd_beta`, the uniform fraction of the batch, the rest tilted by each row's `ema_loss` retention priority. A blocked draw bypasses the weighting.

**The aligned draw.** `cfg:stage.condition_draw` is `{conditions, replay_rows, prior_rows, pick}`, and `Modeller._choose_draw_conditions` runs once per fused step, after the rollout and before either buffer is drawn. Eligible conditions hold at least `replay_rows` trainable replay rows and `prior_rows` prior rows (`buffer.py::CrystalBuffer.condition_row_counts`); the count taken is `conditions`, capped at (and at 0 equal to) `batch_size // max(X, Y)`, selected by `pick` uniformly or through `Modeller._weighted_draw_conditions`. The ids go to both `draw_bwd_sample` and `draw_replay_sample`; an empty return skips replay and folds its share into backward. It raises on a route with more than one space-group and $Z'$ combination, since every draw re-runs `condition_samples`.

## The per-condition normaliser

`GFN.init_flow_model` gives a conditional run without `full_flow` a `scalarMLP` over the condition embedding, so $\log Z$ is a field over $c$. The constructor argument `scalar_flow` forces the `LearnableScalar` branch instead; `conformer_modeller.py` derives it from the condition set, true when the set has exactly one member. `Modeller.z_level_fill` writes `flow_model.scalar.data`, which a `scalarMLP` does not have, so `Modeller._z_fill_head_is_fillable` tests the head for that attribute rather than `conditional`.

Beside the head sits `buffer.py::ConditionLogZTracker`, a persistent per-condition EMA of the empirical $\log Z$ keyed by `condition_id` over flat tensors sized to the condition library, whose `effective_count` decays at `cfg:condition_log_z.half_life_visits` in own-visits; `min_visits` is the floor before an estimate is used. Three coefficients write to the head: `emp_z` (`gflownet_losses.py::emp_Z`) onto this batch's grouped empirical estimate, `emp_z_persistent` onto the tracker's target under that trust mask, and `z_level` (`gflownet_losses.py::z_level_loss`) with $\log w$ detached.

### The three Z settings the rule judges

`config_invariants.py::conditional_z_settings_are_conditional` fires when any of `embedding_conditioning`, `molecule_conditioning` or `vector_conditioning` is true, and emits `BASELINE` violations on three keys. The table sets the unconditional setting of each key beside the setting the rule requires under a conditional run.

| key | unconditional | conditional |
|---|---|---|
| the stage flag `z_calibration` | declared | not declared on any stage |
| `cfg:{fwd,bwd,replay}_loss_coeffs.tb_z_source` | `learned` | `persistent`, per branch the stage trains |
| `cfg:condition_log_z.half_life_visits` | code default 200 | an explicit value below 28 is flagged |

`Modeller.tb_z_source` reads the key off the branch's coefficient block and falls back to `learned`. The rule also names `cfg:condition_log_z.{fwd,bwd,replay}_tb_z_source`, which nothing reads. Under `persistent`, `gflownet_losses.py::get_gfn_forward_loss` passes the tracker's per-condition target and mask into `get_tb_loss` in place of the learned scalar. The stage-scoped clauses read the selected protocol's stage list, a global override being unable to reach a stage declaration.

## Held-out conditions

`cfg:test_molecules_path` loads a second condition file into `test_mol_dataset` through `Modeller._load_condition_file`, the unwrapper `cfg:molecules_path` uses; its identifiers enter the registry, which is what lets `condition_id` resolve. `Modeller.log_test_metrics` runs `fwd_eval_sampling` on it with `side_effects=False`, at `cfg:test_eval_num_samples` rows falling back to `cfg:eval_num_samples`, publishing under `eval_test/`; `side_effects=False` suppresses the tracker's best-energy update, the anchor screen and admit, and the eval-timing writes. There is no `eval_gap/` family: the method's comment records it as removed, being a difference of two already logged streams.

## Per-condition statistics

Given the optional `condition_id`, `utils.py::quick_tb_stats` centres `relative_under` on each sample's own condition's group mean of $\log w$ rather than on the batch-pooled `z_jensen`; `jensen_z` and `z_gap` stay pooled. The per-condition keys are `cond_tb_err`, the unweighted mean over conditions of the within-condition RMS residual, `tb_err_worst`, its `worst_quantile` upper tail, and `z_grad_worst`, the same tail over the clipped signed per-condition mean; with `condition_id=None` the batch is one group and the three degrade to `tb_err` and $|$`tb_resid_clipped`$|$. `logw_std_within`, the pooled within-condition standard deviation of $\log w$, is written only where some condition holds two of that branch's rows. The docstring states the family is per-sample means rather than ratios and records that it replaced a conditional $r^2$ family which could not be EMA'd; of those names only `r2_worst` survives, in `buffer.py`'s note that `fwd/r2_worst` is retired.

`utils.py::per_condition_fraction` turns a per-sample indicator into the `Cond <name> *` family published by `Modeller.log_condition_fraction`: `Failing Frac`, `Worst`, `Spread`, `Frac`, `N` and `Bar`, against `cfg:reasonable_cond_bar` or `cfg:nonthermal_cond_bar`. `Spread` is the between-condition standard deviation with the binomial term subtracted; the docstring states `Failing Frac` is biased by samples per condition. The quantile everywhere is `cfg:conditional_worst_quantile`, passed at the one eval-time call site (`Modeller._eval_conditional_stats`) the three streams share; `ConditionLogZTracker.worst_tb_err`, `.worst_z_bias` and `.delta_stats` take it too, the last reducing the per-condition level gap at $1-q$ and returning $+\infty$ until `min_trusted_frac` of the library is trusted on both streams (`zmatch/delta_worst`).

## The replay seat in `var_conditioning`

In the canonical config's `conditional_vargrad` protocol, `var_conditioning` puts replay in the forward branch's seat. `cfg:stage.fracs` are `{fwd: 0, bwd: 0.5, replay: 0.5}` and `cfg:fwd_loss_coeffs.pooled_source` is `replay`, so `gflownet_losses.py::pooled_condition_vargrad` pairs the replay branch's live rows with the backward branch's while neither carries a term of its own. The forward branch runs on `cfg:stage.fwd_rollout_every` steps under `cfg:stage.fwd_z_sidecar`, with `freeze_policy` and `emp_z` on, so it enters the sum at weight 1 outside the frac mix and reaches $Z(c)$ only; `cfg:stage.replay_warmup_rows` rolls out every step until the replay buffer holds that many trainable rows. `condition_draw` supplies the only groups the two policy-training branches have, `condition_block_m` being 0 on both.

`config_invariants.py::replay_seat_problems` re-runs at every `Modeller.set_loss_coeffs`: `pooled_source` `replay` requires the pooled coefficient on, a replay branch that can carry weight, `cfg:buffers.replay_buffer.prioritise.enabled` false and a declared `condition_draw`; `fwd_z_sidecar` requires a Z-only forward branch with `emp_z` on and a forward frac floor below the deactivation bar.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:{vector,molecule,embedding,temperature,sg,zp}_conditioning` and `cfg:{vector,embedding}_conditioning_dim`; `cfg:molecules_path`, `cfg:test_molecules_path`, `cfg:test_eval_num_samples`, `cfg:eval_num_samples`; `cfg:model.{condition_embedding_dim, cond_hidden_dim, cond_layers}`; `cfg:condition_log_z.{min_visits, half_life_visits, rollout_condition_draw, rollout_under_drawn_power, weighted_condition_sampling*}`; `cfg:conditional_worst_quantile`, `cfg:reasonable_cond_bar`, `cfg:nonthermal_cond_bar`; `cfg:buffers.prior_buffer.weighted_bwd_beta`, `cfg:buffers.replay_buffer.prioritise.enabled`; `cfg:stage.{condition_draw, fwd_z_sidecar, replay_warmup_rows, fwd_rollout_every, bwd_sampling_mode}` and the stage flags `scramble_conditions`, `weighted_bwd_sampling`, `weighted_condition_sampling`, `z_calibration`; `cfg:{fwd,bwd,replay}_loss_coeffs.{tb_z_source, emp_z, emp_z_persistent, z_level, condition_block_m, freeze_policy, pooled_vg, pooled_source}`; `cfg:lr_flow`.

Code: `models/gfn.py::GFN.init_conditioner`, `.init_flow_model`, `.get_condition_embedding`, `._condition_flow`, `._maybe_scramble_condition_embedding`; `train.py::Modeller._build_gfn_config`, `.scramble_applicable`, `.init_identifiers`, `._rollout_condition_mode`, `._choose_draw_conditions`, `.draw_bwd_sample`, `.fwd_eval_sampling`, `._eval_conditional_stats`, `.log_test_metrics`, `.log_condition_fraction`, `.tb_z_source`, `._z_fill_head_is_fillable`; `buffer.py::ConditionLogZTracker.delta_stats`, `.worst_tb_err`, `.worst_z_bias`, `CrystalBuffer.condition_row_counts`; `utils.py::quick_tb_stats`, `::per_condition_fraction`, `::get_problem_definition`; `gflownet_losses.py::emp_Z`, `::z_level_loss`, `::pooled_condition_vargrad`; `config_invariants.py::conditional_z_settings_are_conditional`, `::replay_seat_problems`, `::condition_draw_problems`; `energies/molecular_crystal.py::MolecularCrystal.condition_samples`; `data_processing/generate_toy_prior.py::expand_condition_manifold`.

## Could be tooling

The forward eval draws rows unweighted, so uniformity over conditions coincides with every identifier appearing equally often in the conditions file. A load-time pass over each dataset's `batch.identifier` would publish the rows-per-condition histogram and its range at init, turning a property of file generation into a logged series.

## Sources

The code above at the stamped commit, and the `condition_log_z` block, the conditional globals and the `conditional_vargrad` protocol of the canonical config. Memory files located the code and are not evidence: project_unconditional_prior_by_design, project_condition_sampling_bias_policy, project_conditional_route_z_settings, project_conditional_generalization_check, feedback_holdout_eval_catches_what_train_metrics_hide, project_per_condition_fraction_family, project_r2_ratio_metrics_dont_ema, project_r2_worst_quantile_and_weighted_bwd, project_relative_under_pooled_conditional_fix, project_manifold_unique_identifiers, project_vec_cond_in_problem_identity, project_conditional_replay_seat_redesign.
