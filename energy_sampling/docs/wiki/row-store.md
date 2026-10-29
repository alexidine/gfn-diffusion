# Row store

*Drift: **C** (code). Verified against commit `75e2d97` plus the uncommitted working-tree physical-energy records and `is_exponent`, 2026-10-28. Sources at the end.*

The row store is one flat table of terminal states that serves both off-policy branches of a run under `model.kind nf` when `cfg:buffers.store.enabled` is true. The backward branch draws its rows uniformly from the store's data rows, the replay branch draws every drawable row by a residual priority with importance weights, and forward, evaluation and bootstrap rollouts are offered to it for admission. The table and its arithmetic are `row_store.py::RowStore`; the trainer's side is the `_store_*` methods of `train.py::Modeller`. With the key absent or false, every site below takes the path it takes without the store, and the stores it replaces are on [prior-buffer](prior-buffer.md) and [replay-buffer](replay-buffer.md). The design document is `docs/design/unified_store.md`.

## Rows

A row is a chart state $x$ (the state the NF policy scores: the trainer latent, or the FD chart row under `cfg:energy_config.crystal_coordinates` `fd`), a condition id, the stored $\log R$, and the row's physical energy $E$. The columns are plain tensors of one length $n$, rebuilt by `index_select` and `cat`:

| column | dtype | content |
|---|---|---|
| `cid` | int64 | condition id |
| `x` | float32 [n, dim] | chart state |
| `log_r` | float32 | $\log R$ at admission, in the run's referenced currency |
| `energy` | float32 | physical energy $E$ per molecule, unreferenced: `generator_energy`'s `ens_dict['mol_energy']` |
| `log_q` | float32 | $\log q(x \mid c)$ at the last refresh or touch; NaN before the first |
| `q_step` | int64 | training step of that write; -1 before the first |
| `tag` | int8 | anchor, child, onpolicy or generated (`TAG_NAMES`) |
| `protected` | bool | never evicted |
| `held_out` | bool | never drawn |
| `born_eval` | int32 | the store's eval counter at admission |

Two per-condition tables sit beside them: `cond_table`, the condition vector of each condition id (NaN until first seen), and `best_energy`, the lowest finite $E$ among that condition's non-held-out seed rows and its records (plus infinity until then).

$E$ is `MolecularCrystal.generator_energy`'s `mol_energy`: the configured energy key divided by $Z'$ under every key but uma and mace (eLJ, LJ, qLJ, silu) and read as-is under those two, times `lj_rescale` when that is set on an LJ-type key, and passed through `screen_implausible_energy` under `cfg:energy_config.implausible_energy_below` (a flagged row reads $|\text{below}|$). It carries no chart Jacobian, bounding, reduction, density or pressure term, no clip and no energy reference, so it is not $-T \log R$; the two differ row by row within a condition by the Jacobian and the penalty terms. `RowStore.set_conditions` fills `cond_table` and raises when a condition id arrives with a vector that differs from the stored one by more than $10^{-6}$.

The rows are ordered: $[0, n_{\text{mle}})$ are data rows not held out, $[n_{\text{mle}}, n_{\text{seed}})$ are held-out seed rows, and $[n_{\text{seed}}, n)$ are admitted rows. Seed rows are protected and never demoted, and compaction keeps order, so the layout holds for the store's life. `RowStore.admit` refuses the data tags, so data rows enter only through `RowStore.seed`.

## Seeding

`train.py::Modeller.init_row_store` runs in `Modeller.train` after `init_anchor_buffer_seed` and before the buffer-currency check. It raises on every ERROR of `config_invariants.py::row_store_problems` and when the prior is not a pre-noised build (`prior_is_prenoised`). Seeding (`Modeller._store_seed`) reads two batches of the pre-noised prior (the build and its load are on [prior-buffer](prior-buffer.md)): `prior_dataset`'s rows, which are the training split's rows of `cfg:noised_prior.train_kinds` inside both load windows, tagged from their `noised_kind`; and `prior_heldout_batch`, the held-out anchor groups, flagged held out. For each row $x$ is `Modeller._batch_latents`, $\log R$ is `Modeller._stored_reward_terms` (the stored energy terms composed at the run's temperature, after the energy reference is installed), $E$ is the same call's `ens_dict['mol_energy']` (`with_mol_energy`; no energy is computed), and the condition id and vector come from `condition_samples` on the row's own space group and $Z'$. Clean anchors outside `train_kinds` are not seeded. A seed row's $\log R$ carries no latent bounding term, while a rollout row's does: `_stored_reward_terms` composes the reward without `raw_latents`. `best_energy` is set from the finite $E$ of the non-held-out seed rows only, and the seeding line prints its minimum over conditions. The store is refreshed once after seeding.

## The level and the residual

Every draw, admission, refresh and monitor reads a level table $L(c)$ built at the call (`Modeller._store_level_table`); no row stores a level. $L(c)$ is the replay branch's TB level: the head (`flow_model` on `get_condition_embedding` of the stored condition vector, or the unconditional scalar), replaced under `cfg:replay_loss_coeffs.tb_z_source` `persistent` by `gflownet_losses.py::tb_z_per_traj` over the tracker's `lookup` and `untrusted_z_for`, which is the per-row $Z$ the replay TB loss uses. Conditions with no stored vector read NaN. The residual of row $i$ is

$$
r_i = \log R_i - \log q_i - L(c_i),
$$

and since $\log P_B = 0$ on the NF policy, the TB residual the loss computes is $-r_i$. The drawable set $E$ is the rows not held out with finite $r$, $n_E = |E|$.

## The two draws

The backward draw (`Modeller.draw_bwd_sample` returns `Modeller._store_draw` `'mle'` first) takes `batch_size` indices i.i.d. uniform over $[0, n_{\text{mle}})$ (`RowStore.draw_mle`), whatever the stage's backward loss terms are. Priority is not read.

The replay draw (`Modeller.draw_replay_sample` after its held-out branch; `RowStore.draw_tb`) works on $E$. With $s_i = |r_i|$, $m$ the median of $s$ over $E$, floor $f = \max(\text{floor\_frac} \cdot m, 10^{-12})$ and $s'_i = \max(s_i, f)$:

$$
P_i = \frac{s'^{\kappa}_i}{\sum_{E} s'^{\kappa}}, \qquad
q_i = (1-u)\,P_i + \frac{u}{n_E}, \qquad
w_i = \left(\frac{1/n_E}{q_i}\right)^{a},
$$

with $\kappa$ `cfg:buffers.store.kappa`, $u$ `cfg:buffers.store.uniform_frac` and $a$ `cfg:buffers.store.is_exponent` in $[0, 1]$ (code default 1, as in the canonical config; the canonical config sets $u$ to 0.5 against a code default of 0.1). `batch_size` indices are drawn i.i.d. from $q$ (`torch.multinomial` with replacement, computed in float64); $a$ changes the weights, not the draw. At $a = 1$ the weights satisfy $\sum_E q_i w_i f_i = \frac{1}{n_E}\sum_E f_i$ for any $f$; the loss applies them self-normalised at its final reduction (`gflownet_losses.py::get_gfn_backward_loss`), which makes it a consistent estimator of the uniform-over-$E$ branch loss, Huber included. At $a < 1$ the self-normalised loss estimates the branch loss under the measure proportional to $q_i^{1-a}$ over $E$: $a = 0$ is the loss under the draw measure $q$ itself, unweighted. $w \le (1/u)^a$ for $u > 0$, the floor bounds the ratio of priorities by $(\max s / f)^\kappa$, and $\kappa = 0$ is the uniform draw. The draw writes, over the exponentiated weights, `replay/is_ess_frac` $= (\sum w)^2 / (B \sum w^2)$, `replay/is_w_max_ratio` $= \max w / \bar w$ and `replay/is_elig_frac` $= n_E$ over the non-held-out rows, the keys the prioritised replay buffer writes. `torch.multinomial` limits $n_E$ to $2^{24}$ rows.

Both draws return the seven-tuple of the buffer draws with no molecule batch, the stored $\log R$, the condition vector from `cond_table`, and for replay the one-step trajectory $(0, x)$. The policy reads `mol_batch` only under `atom_tokens`, which the store refuses. `inds` is the pair (row indices, store version). After the loss, `Modeller._store_touch` writes the drawn rows' `log_pf` as their $\log q$ (`RowStore.touch`, which raises on a version other than the store's); the backward branch skips the write on a step with scrambled conditions.

## Admission

`Modeller.manage_replay_buffer` hands a rollout to `Modeller._store_admit` after its harvest and `replay_in_play` checks, so every site that admits to the replay buffer admits to the store instead: the training rollout, the z-calibration rollout, the `bootstrap_z:rollout` action and, under `cfg:buffers.replay_buffer.admit_from_eval`, the evaluation rollout, which is offered whole, not capped at a churn budget. A Z-pin rollout is not admitted. `_store_admit` reads the condition ids, the terminal states `flow_states[:, -1]`, `log_r` and `log_pf`, and $E$ from the analysed rollout batch's `mol_energy` attribute (the ens_dict keys `analyze_crystal_batch` copies onto the batch); it raises when the batch carries no `mol_energy` or one of another length; it writes the batch's condition vectors with `set_conditions` before it builds the level table, so a condition the store has not held gets a level. Every admitted row is tagged onpolicy.

For candidate $i$, `RowStore.admit` forms $r_i$ and a finiteness mask (finite $\log R$, $\log q$ and $L$), and drops a finite row whose unreferenced $\log R$ (`MolecularCrystal.unreferenced_log_r`) is below `cfg:buffers.store.admit_reward_min` (counted as `rejected_floor`). A candidate is a record when it is finite, passes the floor and has finite $E$, its $E$ equals the minimum $M_c$ of $E$ over this call's such candidates of its condition, $M_c$ is strictly below `best_energy[c]`, and it has the lowest index among ties; there is at most one record per condition per call. A candidate is admitted when it is finite and $|r_i| \ge$ `cfg:buffers.store.bar` or it is a record. Records are protected, and `best_energy[c]` becomes $M_c$. The record currency is the physical energy $E$, not the reward $-T \log R$: a row with lower $\log R$ and lower $E$ than the condition's best is a record, and one with higher $\log R$ and higher $E$ is not. Admission by $|r|$, the draw priority and the eviction weights still read $\log R$. The floor stays on the unreferenced $\log R$ and sits inside the record test, so a row under it is never a record whatever its $E$; a non-finite $E$ (possible under uma and mace) is never a record.

## Demotion and eviction

After the candidates are appended, `RowStore._demote` ranks each condition's protected non-data rows by $E$, lowest first, ties by position, and removes protection from ranks at or past `cfg:buffers.store.records_kept` (`store/demoted`), so the record with the highest $E$ goes first. Records strictly decrease in $E$ within a condition, so the rows demoted are its oldest records. Data rows are never demoted. With `records_kept` null nothing is demoted.

`RowStore._evict` then evicts from the pool of unprotected, non-held-out rows, the call's own admissions included. With $n^{\text{pool}}_c$ the pool rows of condition $c$, the per-condition excess is $\max(0, n^{\text{pool}}_c - \texttt{cap\_per\_condition})$ and the global excess is $\max(0, \sum_c n^{\text{pool}}_c - \sum_c \text{excess}_c - \texttt{max\_rows})$. Each pool row gets the key $-\log(|r_j| + \epsilon) + G_j$ with $G_j \sim \text{Gumbel}(0, 1)$ and $\epsilon$ `cfg:buffers.store.eps`; a non-finite $r$ counts as $|r| = 0$. Per condition the excess rows with the highest keys are evicted, then the global excess with the highest keys among the survivors. For one eviction this is $P(\text{evict } j) = \pi_j / \sum \pi$ with $\pi_j = 1/(|r_j| + \epsilon)$; both passes reuse one set of keys. The columns are compacted with `index_select`, and the store version increments when a row that existed before the call moved.

After every call the unprotected rows number at most `cap_per_condition` per condition and `max_rows` in total, and the protected rows at most the seed rows plus `records_kept` per condition. A condition holding only protected rows evicts nothing. No step of admission waits on anything.

## Refresh

`Modeller._store_refresh` recomputes every row's $\log q$ under the policy in eval mode, `cfg:buffers.store.refresh_chunk` rows per call of the policy's `log_density`, and increments the store's eval counter (`RowStore.refresh`). A CUDA out-of-memory error in a call halves the chunk and redoes the call; the halved size is kept as `RowStore.chunk_cap` for the rest of the run and is not saved, and an out-of-memory error at one row is raised. It runs once at init, at every evaluation directly after `protocol.maybe_advance` (so before the evaluation rollout is admitted), and after the divergence rewind reloads weights. It prints one line: rows, seconds, non-finite $\log q$ count, median $|r|$ over $E$, and the window's admitted and evicted counts.

## Branch availability and the occupancy trigger

`Modeller._replay_rows` returns $n_E$ when the store is on, the replay buffer's length when one exists, and None otherwise. The fused step's replay branch runs when the stage can boost replay and `_replay_rows` is positive, so with the store on replay draws seed rows from the first fused step; at $n_E = 0$ its share folds into the backward branch as it does for an empty replay buffer. The `occupancy_min_batches` rollout trigger (`Modeller._rollout_trigger_reading`) reads `_replay_rows` over the live batch; the seed rows alone hold it at or above $n_{\text{mle}}/B$. A None reading never fires. The prior and anchor buffers are still built and published under their own keys and are not drawn; the replay buffer is never built.

## Persistence

`checkpointing.py::Checkpointer.buffer_state` writes `RowStore.state_dict` under `row_store` in the buffer sidecar, stamped with `Modeller._store_stamp`: the noised-prior identity and load settings, the prior file's name, size and manifest record, the chart and `data_ndim`, the condition library size and identifier registry, the temperature, the energy configuration less its execution keys, the armed energy clip, the energy-reference mode and table, and the store's `row_store.FORMAT_VERSION` (`store_format`, 2 since the `energy` column). A sidecar written by an earlier format has no `store_format` entry, so its stamp differs and the store reseeds; `RowStore.from_state_dict` raises on any other format version. `Checkpointer.restore_buffers` only parks the state on the modeller; `init_row_store` loads it when its stamp equals the run's, else prints that it is ignored and seeds afresh. Either way the store is refreshed before the first step. Settings come from the config, never the state. The sidecar is written before the evaluation's refresh, so it holds the previous evaluation's $\log q$. The divergence rewind keeps the live store.

## Monitoring

`Modeller.log_buffer_stats` publishes `RowStore.stats` against the live level at each evaluation and drains the admission window:

| key | quantity | unit |
|---|---|---|
| `store/rows`, `store/rows_{anchor,child,onpolicy,generated}`, `store/rows_heldout`, `store/rows_protected`, `store/rows_records` | rows by tag, held out, protected, protected non-data | rows |
| `store/cond_rows_{min,median,max}` | rows per condition over conditions with a non-held-out row | rows |
| `store/conds`, `store/conds_at_cap` | conditions holding rows; those whose unprotected rows reach `cap_per_condition` | conditions |
| `store/admitted`, `store/admitted_records`, `store/rejected_floor` | window admissions, records, floor refusals | rows |
| `store/evicted_cap`, `store/evicted_global`, `store/demoted` | window evictions by pass, demotions | rows |
| `store/admitted_resid_med`, `store/evicted_resid_med` | median $\lvert r\rvert$ at admission, at eviction; NaN when empty | nats |
| `store/evicted_age_{med,max}` | age of evicted rows | evaluations |
| `store/protected_above_bar`, `store/protected_above_bar_conds`, `store/protected_above_bar_cond_max` | protected non-held-out rows with $\lvert r\rvert \ge$ bar: total, conditions with any, the most in one condition | rows, conditions, rows |
| `store/resid_med`, `store/resid_p90` | $\lvert r\rvert$ over $E$ | nats |
| `store/refresh_seconds`, `store/refresh_nonfinite` | last refresh wall time; rows with non-finite refreshed $\log q$ | s, rows |
| `store/best_energy_min`, `store/best_energy_med` | `best_energy` over conditions with a finite one: lowest, lower median | energy per molecule, run units, unreferenced |
| `store/seed_best_energy_min` | lowest $E$ over the finite non-held-out seed rows | same |
| `store/record_conds` | conditions whose `best_energy` is below their seed rows' lowest $E$ (or that hold no finite seed row) | conditions |
| `store/record_gain_med` | lower median over those conditions with a finite seed minimum of (seed minimum $-$ `best_energy`) | same energy unit |
| `store/records_energy_min`, `store/records_energy_med` | $E$ over protected non-data rows | same energy unit |

The medians are lower medians. `torch.quantile` in `resid_p90` limits $E$ to 16M rows.

## Refusals

`config_invariants.py::row_store_is_well_formed` reports, and `init_row_store` raises on, an enabled store with: `model.kind` other than nf; the conformer route; `cfg:temperature_conditioning`; `cfg:energy_config.prior_flow_path`; `model.nf.atom_tokens`; `cfg:buffers.replay_buffer.prioritise.enabled`; `cfg:buffers.replay_buffer.val_frac` above 0; `cfg:buffers.anchor_buffer.frozen` not true; harvested rows in `cfg:noised_prior.train_kinds`; `cfg:condition_log_z.rollout_condition_draw` `under_drawn`. Per active stage: `cfg:stage.condition_draw`, `cfg:stage.replay_warmup_rows` above 0, replay `mle`, forward `pooled_vg`, backward or replay `vg_lb` or `vg_lme`, `weighted_bwd_sampling`, a non-empty `cfg:stage.coeff_schedule` or `balance.anneal_coeffs`, the `drift_std_max`, `ess_min` or `val_gap_max` bars of `cfg:stage.fwd_rollout_triggers` above 0 (their readings are replay-buffer statistics the store does not write), `lr_sensor` kind ray, and `flags.z_calibration` with `cfg:z_calibration.mode` replay (`Modeller._z_replay_step` reads the replay buffer, which the store never fills, and returns without a step). It also refuses unknown keys under `buffers.store` and malformed values, and reports `records_kept` null as a BASELINE. An absent key reads `config_invariants.py::ROW_STORE_DEFAULTS`.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:buffers.store.{enabled, bar, admit_reward_min, cap_per_condition, max_rows, eps, records_kept, kappa, floor_frac, uniform_frac, is_exponent, refresh_chunk}`, `cfg:energy_config.implausible_energy_below`, `cfg:replay_loss_coeffs.tb_z_source`, `cfg:buffers.replay_buffer.admit_from_eval`, `cfg:noised_prior.train_kinds`, `cfg:stage.fwd_rollout_triggers`, `cfg:batch_size`.

Code: `row_store.py::RowStore.seed`, `.set_conditions`, `.resid`, `.n_drawable`, `.draw_probs`, `.draw_tb`, `.draw_mle`, `.touch`, `.refresh`, `.admit`, `._demote`, `._evict`, `.stats`, `.state_dict`, `.from_state_dict`; `train.py::Modeller.store_settings`, `._stored_reward_terms`, `._store_stamp`, `.init_row_store`, `._store_seed`, `._store_level_table`, `._store_refresh`, `._store_draw`, `._store_touch`, `._store_admit`, `._replay_rows`, `.draw_bwd_sample`, `.bwd_train_step`, `.draw_replay_sample`, `.replay_train_step`, `.fused_train_step`, `._rollout_trigger_reading`, `.manage_replay_buffer`, `.evaluation`, `.log_buffer_stats`; `checkpointing.py::Checkpointer.buffer_state`, `.restore_buffers`; `config_invariants.py::row_store_settings`, `::row_store_problems`, `::row_store_is_well_formed`; `energies/molecular_crystal.py::MolecularCrystal.generator_energy`.

## Could be tooling

The record test, the protected-row bound and the eviction weights are functions of the stored columns alone. A script loading a `_buffers.pt` sidecar's `row_store` state could recompute $r$ against a supplied level, list each condition's protected rows and records with their $E$ and $\log R$, and tabulate occupancy per condition against `cap_per_condition`, without building a policy.

## Sources

Repo: `docs/design/unified_store.md` (the design), `energy_sampling/row_store.py`, `energy_sampling/train.py`, `energy_sampling/checkpointing.py`, `energy_sampling/config_invariants.py`, `energy_sampling/gflownet_losses.py`, `energy_sampling/energies/molecular_crystal.py` and the `buffers.store` block of the canonical config, read at the stamped commit with the working-tree changes. Tests: `tests/crystal/test_row_store.py`, `tests/crystal/test_row_store_trainer.py`, `tests/crystal/test_row_store_sidecar.py`, `tests/config/test_row_store_invariants.py`.
