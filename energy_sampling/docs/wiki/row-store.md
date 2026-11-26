# Row store

*Drift: **C** (code). Verified against commit `6dc6aa6a` plus the change that wires `crystal_store.py` into the trainer, 2026-11-23. Sources at the end.*

The row store is one table of scored crystal rows that serves both phases of a run under `model.kind nf` when `cfg:buffers.store.enabled` is true: phase 1 draws anchors from it and trains on fresh kicks of them, phase 2 draws stored children and on-policy rows for trajectory balance, and every row scored in the run is written to it. The table and its index are `crystal_store.py::CrystalStore`; the trainer's side is the `_store_*` methods of `train.py::Modeller`, with `Modeller.init_row_store`, `Modeller._kick_anchors` and `Modeller._kick_fresh`. With the key absent or false, every site below takes the path it takes without the store, and the stores it replaces are on [prior-buffer](prior-buffer.md) and [replay-buffer](replay-buffer.md).

## Rows

A row is a chart state $x$ on the FD chart (`cfg:energy_config.crystal_coordinates` `fd`), a condition id, and plain per-row columns, preallocated to the row limit on the CPU and written by slot, so an add costs O(rows added):

| column | dtype | content |
|---|---|---|
| `x` | float32 [n, d] | chart state |
| `cid` | int32 | condition id: (molecule, space group, $Z'$) |
| `kind` | int8 | `ANCHOR`, `CHILD`, `ONPOLICY`, or `FREE` for a freed slot |
| `flags` | uint8 | bit `UNDRAWN`: in no draw pool; bit `EVAL`: written by an eval or bootstrap rollout |
| `origin` | int8 | `ORIGIN_ROLLOUT`, `ORIGIN_EVAL`, `ORIGIN_BOOTSTRAP`, `ORIGIN_SEED`, `ORIGIN_KICK` |
| `parent` | int32 | anchor slot of a child, -1 otherwise |
| `log_r` | float32 | $\log R$ at write, in the run's currency (after the reward clip and energy reference) |
| `energy` | float32 | physical energy per molecule, unreferenced: `generator_energy`'s `mol_energy` |
| `log_q` | float32 | $\log q$ at the last training draw or at on-policy admission; NaN = none |
| `log_q_birth` | float32 | $\log q$ at on-policy admission, written once; NaN on anchors and children |
| `q_step` | int32 | step of the `log_q` write; -1 = none |
| `resid` | float32 | signed $\log P_F - \log P_B - \log R + \log Z$ at that write; NaN = none |
| `draws` | int32 | times trained on, by occurrence |
| `birth_step` | int32 | step the row was written; 0 for seed rows |

Per condition the store holds one molecule template (`CrystalStore.templates`, cut to `train.py::Modeller.STORE_TEMPLATE_KEYS` and never oriented), the space group, radius and radius of gyration read from it, `best_energy` (the lowest `mol_energy` seen from any source, which only falls) and `window_floor` (the load-time floor of `noised_prior.py::select_rows`, written once). `CrystalStore.check` verifies the invariants listed in the module docstring.

## Load

`train.py::Modeller._load_noised_prior` runs its identity check, sampled re-score and selection as on [prior-buffer](prior-buffer.md), then on the store route returns through `Modeller._store_take_rows`:

1. The rows kept are `select_rows`' `buffer` mask with a finite $\log R$: every scored training row inside the packing and energy windows.
2. A kept child whose parent anchor has no chart point is dropped and counted (`live_kernel/physical_rows_without_parent`). A kept child's parent that is not an anchor of the same condition and training split raises.
3. A kept child's parent anchor outside the windows is stored with `UNDRAWN`, so the link holds and the anchor is never kicked.
4. Each condition's template is its first stored anchor. A stored row whose atoms differ from the template's, or lie more than `Modeller.STORE_FRAME_TOL` Angstrom from them, raises (`Modeller._store_frame_check`).
5. Three frozen graph panels are cut for the readouts, each capped at $4 \times$ `cfg:eval_num_samples` rows and deterministic in the file and `cfg:seed`: `prior_dataset` (in-window training children, read as `eval_bwd/mle`), `prior_heldout_batch` (held-out children, `select_rows`' `heldout_children`, read as `eval_bwd/mle_heldout_rows`) and `prior_anchor_batch` (in-window training anchors, read as `eval_bwd/mle_anchors`). `eval_bwd/mle` is therefore a frozen panel of children and not the MLE draw, which is fresh, unfiltered kicks of anchors. `prior_dataset` is built with `buffer.py::CrystalBuffer`'s `orient=False`, as are the two conditions datasets on this route.

The returned energies are $-T \log R$ of the kept rows, so the reward clip is measured over every scored row. The file batch stays held in `Modeller._store_load` until `Modeller.init_row_store`.

`Modeller._store_after_identifiers` then gives each stored row its condition id and adds one template per training condition with no stored row, from that molecule's `cfg:molecules_path` row; the load prints their count. `Modeller.init_row_store` runs after the clip and energy reference are armed: it reads each row's final $\log R$ and `mol_energy` (`Modeller._stored_reward_terms`), sizes the store from `config_invariants.py::store_row_limit`, seeds and checks it, warms `condition_log_z`'s per-condition floors from $-T \log R$ of every row given (`Modeller._store_warm_tracker`), applies a sidecar entry when one is parked (`checkpointing.py::Checkpointer.restore_row_store`), folds every live row of a restored store into the same floors, keeping the tracker's discovery counters, and releases the file batch. With the store off, a parked entry of this run raises, since the run's next sidecar would drop its grown rows, and another run's entry is cleared with a printed line. `init_prior_buffer_seed` and `init_anchor_buffer_seed` return early on this route.

`Modeller._stored_reward_terms` runs twice per launch over the file's rows: once when the pre-noised prior is loaded (seed energies and $\log R$ for the windows) and once here, after the clip and reference are armed. Given `prior_path`, each pass goes through a sidecar beside the prior file (`noised_prior.cache_path` kind `rewards`), keyed on the file, the rows, their condition ids, the temperature, the energy function's settings, its clip and its reference, the FD chart digest, and a digest of the reward code's source (`Modeller._reward_code_digest`). A cached result is used only after a fresh random draw of `Modeller.REWARD_CACHE_CHECK_ROWS` rows, the stratified rows of `noised_prior.check_rows` and the 32 highest-energy rows re-score to it; otherwise every row is recomputed and the sidecar rewritten. A sidecar that cannot be written is reported and the run continues (`noised_prior.save_cache`). Nothing removes old sidecars: each distinct setting leaves about 3 × 8 bytes per row, and old `*.rewards-*.pt` files may be deleted freely. `Modeller._stored_reward_terms_fresh` drops `Modeller.ROW_LIST_KEYS` from each chunk before its device move, since torch_geometric walks a Python list element by element and the reward reads neither list.

Headroom $H$ (`Modeller._store_headroom`) is the largest single add: the largest of twice a rollout ($B \times$ forward repeats), `cfg:eval_num_samples`, and any `bootstrap_z:offset` draw size. The load refuses a row limit below anchors $+ 2H$. Over the limit `CrystalStore.thin` frees rows 50:50 between children and on-policy rows, uniformly within each; anchors are never thinned and children never fall below $H$.

## Phase 1

`draw_bwd_sample` calls `Modeller._store_draw('mle', repeats)` in a `bwd_sampling_mode` `'dataset'` stage. It draws $B$ anchors (`cfg:buffers.store.mle_draw`: `anchor` uniform over anchors, `condition` condition-first) and kicks each with `physical_kick.py::kick` ([prior-buffer](prior-buffer.md), the physical kernel). The kick uses the run's own `torch.Generator`, seeded from `cfg:seed` plus the launch step, and neither reads nor advances the global generators. A row whose kick returned the anchor's own point is replaced by a kick of another anchor, at most `Modeller.MLE_KICK_ROUNDS` times, then dropped and counted (`live_kernel/mle_kept_dropped`); an empty draw raises. So the batch holds no anchor point. The draw carries no reward, writes no store column (`inds` is None), and raises in a stage whose backward loss reads $\log R$.

Under `cfg:temperature_conditioning` each kicked row is conditioned on its own effective temperature (`Modeller._kick_labels_temperature`, `Modeller._kick_temperature`). `physical_kick.py::kick` returns each row's accepted step length $\rho$ (Angstrom, `rho_out`), and the row's temperature is

$$T_{\mathrm{eff}} = \frac{\kappa_{\mathrm{atom}}\, n_{\mathrm{atoms}}\, \rho^2}{d},$$

with $\kappa_{\mathrm{atom}}$ = `cfg:noised_prior.kick_kappa_per_atom`, $n_{\mathrm{atoms}}$ the molecule's atom count and $d$ the live coordinates of the row's space group. It is the harmonic estimate: a kick of length $\rho$ costs about $\tfrac12 \kappa \rho^2$ and a Boltzmann sample has excess $d\,kT/2$; the per-atom scaling of $\kappa$ is a working assumption fitted on the pool4 test prior's children (kick lengths 0.03 to 0.1 A), not a derivation. The row passes it to `MolecularCrystal.condition_samples` as `temperature`. Every other condition draw on the store route (evaluations, rollouts, readouts) takes `cfg:energy_config.temperature`: `train.py` sets `MolecularCrystal.untagged_temperature` after the policy's scalers are fitted, and the scaler fit itself (`Modeller.fit_policy_data_scalers`) draws temperatures log-uniformly over `cfg:energy_config.log_temperature_range` (log10 of kT).

## Phase 2

| branch | draw | write |
|---|---|---|
| bwd | uniform condition, uniform drawable child | `log_q`, `resid`, `q_step`, `draws` |
| replay | uniform condition, then an on-policy row by `cfg:buffers.store.replay_weight` | the same |
| fwd | a rollout of $B$ molecules, each with a uniform space group and $Z'$, on the condition's template | every scored row, on-policy |
| growth | one kick per rollout row written | one child per kick |

*Each branch's source of rows and what it writes back to the store.*

`Modeller._store_draw` builds the batch from `CrystalStore.mol_batch`, so a stored row is scored and trained against its condition's template. The replay path is the row's terminal only: the policy is a flow, and `traj` is a two-state stack of zeros and the terminal. Replay draws no rows while the pool of drawable on-policy rows is empty, and `fused_train_step` then folds replay's share into bwd.

`crystal_store.py::REPLAY_WEIGHTS` maps a rule name to a function `rule(store, rows, step)` that returns one weight per row, or None for uniform within the condition. The registered rules are `uniform`, `crystal_store.py::duty` and `crystal_store.py::visits`. A rule in `crystal_store.py::UNCORRECTED_RULES` goes through `CrystalStore.draw_accepted`; any other rule but `uniform` goes through `CrystalStore.draw_weighted`, whose correction $1 / (n_c\, p)$ is handed to the loss as the sample weight.

`CrystalStore.draw_accepted` takes weights in $[0, 1]$ and has `draw_weighted`'s law without reading every row of a drawn condition: each slot proposes a uniform row of its condition and keeps it with probability $w$, for at most `crystal_store.py::ACCEPT_ROUNDS` rounds, and a slot still open takes `draw_weighted`'s scan of its condition. The cap bounds the loop whatever the weights are.

`duty` and `visits` are in that tuple, so each changes which rows are drawn and the replay loss stays unweighted (`Modeller._store_replay_rows` returns None as the weight). With $r = \log R - \log q - L$, which is `-resid` because $\log P_B = 0$ on this route, a row's draw weight within its condition is

$$w = \begin{cases} 1 & r > 0 \\ \max\!\big(e^{-\max(\log q_\text{birth} - \log q,\ 0)/\tau},\ 0.05\big) & \text{otherwise} \end{cases}$$

with $\tau$ = `crystal_store.py::DUTY_TAU` and the floor `crystal_store.py::DUTY_FLOOR`. It reads the row's stored columns, rewritten on each training draw, and makes no $\log q$ call. A NaN `resid` (after a rewind) is an unknown $r$ and takes weight 1. `config_invariants.py::row_store_problems` refuses `duty` with `rotate_frames` or `mirror_frames`, since $\log q$ then carries a different random frame per write.

`visits` weights a row $w = \max(1 / (1 + d), w_{\min})$, with $d$ its `draws` column (training occurrences) and $w_{\min}$ = `crystal_store.py::VISITS_FLOOR`. It reads no $\log q$, so it is not refused under random frames. A divergence rewind and a sidecar resume reset `log_q` and `resid` written after the restored step (`CrystalStore.invalidate_q_after`) but not `draws`, so under `visits` a row drawn in the discarded window keeps its lower weight.

The headroom `Modeller._store_headroom` (the largest single add) is twice a rollout of `cfg:max_batch_size` rows when `cfg:grow_batch_size` is true, since the sizer can raise the batch after the store is built, and of the configured batch otherwise. A resume takes this launch's headroom: `CrystalStore.apply_state` matches `dim` and `max_rows` only and checks I11 under the new headroom, so the batch settings may change between legs.

`Modeller._store_admit` writes each rollout row with $\log q$, residual and `q_step`; `CrystalStore.add` drops and counts rows with a non-finite $x$, $\log R$ or energy (`store/nonfinite_rows`) and stores a row with a finite reward and non-finite $\log q$ as `UNDRAWN`. `manage_replay_buffer` calls it before its `replay_in_play` check, so every stage's rollouts are kept. Under `cfg:buffers.store.keep_rollouts` false (default true) a training rollout (`ORIGIN_ROLLOUT`) writes no row: `Modeller._store_admit` lowers `best_energy` over its finite rows, feeds the eval pool, owes growth kicks for its finite rows on the same cadence as a kept rollout, and returns an empty index. Eval and bootstrap rows are written either way.

| rows | stored | drawable |
|---|---|---|
| training rollout | yes | yes |
| eval rows drawn fresh in a stage with `buffers_active`: the rollout pool's shortfall, or every row when the pool is empty or refused | yes, `EVAL` | yes |
| eval scored in any other stage, the one that ends MLE included | yes, `EVAL` | no |
| `bootstrap_z:offset` at stage entry | yes, `EVAL`, `ORIGIN_BOOTSTRAP` | no |

*How each kind of on-policy rollout enters the store. `evaluation` reads `buffers_active` before `protocol.maybe_advance`, so rows are judged by the stage that scored them.*

In a `buffers_active` stage the eval's on-policy statistics read `Modeller._eval_pool`, a `crystal_store.py::RolloutPool` of the trailing $N$ = `cfg:eval_num_samples` rows of training rollouts. `Modeller._store_admit` feeds it through `Modeller._eval_pool_add`, from `ORIGIN_ROLLOUT` rows only. It is held on the CPU and trimmed at row level when read, so the set is always $N$ rows and its age varies with the batch size and `cfg:stage.fwd_rollout_every`. `Modeller._eval_forward_rows` reads it, `Modeller._pool_eval_stats` shapes the rows as an eval's (each row's `log_flow` is the level it was scored with, the Gaussian diagnostics the NF policy's constant zeros) and `Modeller._fwd_eval_summary` computes the pooled summary, the same function a fresh draw goes through. Pool rows are not written to the store again and make no floor or tracker update: the rollout that scored them did both.

A shortfall (the start of a stage, or after a rewind or resume, since the pool is not checkpointed) is drawn fresh by `Modeller.fwd_eval_sampling`, so an eval never waits on the pool. Those rows take a fresh eval's treatment, floor updates included, and are stored as drawable `EVAL` rows. The pool is cleared on a rewind (`Modeller._store_rewind`) and at every stage transition. `Modeller._eval_pool_refusal` makes the eval draw every row fresh, printing the reason once, under `exploration_std` above 0, `rotate_frames` or `mirror_frames`, a fit-error rollout condition draw (`Modeller._rollout_draw_is_weighted`), a `drift_force` provider, or `z_calibration.fill_from_eval` on a head it can fill. The held-out panel, backward and density readouts, cross-condition readouts and figures run as they do without the pool, and a stage without `buffers_active` draws its eval fresh.

The held-out molecules are read through a fixed panel (`Modeller._store_heldout_panel`): $\min(\lfloor n / (2 n_{sg}) \rfloor, \text{test molecules})$ molecules, at least one, for $n$ = `cfg:test_eval_num_samples`, chosen once from `cfg:seed`, each scored twice in every space group of the run. Those rows are not stored.

## Growth

`Modeller._store_grow` runs after each training step. For each kick owed (`Modeller._growth_kicks_due`: one per on-policy row the last training rollouts wrote) it draws a uniform condition and a uniform anchor of it, kicks it, makes one energy call and stores the result as a child linked to that anchor. A kick that kept its anchor spends no energy call and stores nothing. Only every `cfg:buffers.store.growth_every`-th training rollout owes its kicks (`Modeller._owe_growth`; default 1, every rollout; the count is not checkpointed, so a resume owes on its first rollout). At the default the scoring budget is split evenly between rollout rows and kicks, at $2BR/N$ scores per step for rollout period $N$ = `cfg:stage.fwd_rollout_every`; in general it is $BR/N$ rollout scores and $BR/(N g)$ kicks per step for `growth_every` $g$.

A kick outside the packing window, or whose seed energy (`MolecularCrystal.seed_energy_from`) is above the frozen `window_floor` by more than `cfg:noised_prior.window_kT` $\times T$, is stored `UNDRAWN`. The comparison does not read `best_energy`, so a falling minimum does not re-filter rows. Both `condition_log_z` minima take the scored rows through `Modeller._anchor_energy`.

## Rewind and resume

The divergence rewind calls `Modeller._store_rewind`, which runs `CrystalStore.invalidate_q_after` at the checkpoint's step: a row's `log_q` written after it returns to its birth value, or to NaN. It also clears the eval's rollout pool. No per-eval refresh of stored $\log q$ exists on this route.

## Monitoring

`Modeller._store_metrics`, through `log_buffer_stats`, publishes at each evaluation:

| key | quantity | unit |
|---|---|---|
| `store/rows`, `store/anchors`, `store/children`, `store/onpolicy`, `store/undrawn` | rows by kind; rows in no draw pool | rows |
| `store/drawable_anchors`, `store/drawable_children`, `store/drawable_onpolicy` | rows in each draw pool | rows |
| `store/thinned_total`, `store/nonfinite_rows` | rows freed by thinning; rows dropped at add | rows |
| `store/epoch`, `store/generation`, `store/index_compactions`, `store/conditions` | store counters | count |
| `store/kicks_owed`, `store/kicks_drawn`, `store/kicks_scored`, `store/kicks_unchanged` | growth tally | rows |
| `store/kick_in_window_frac` | share of scored kicks inside the fixed windows | fraction |
| `store/rows_out_of_box` | on-policy rows written with a coordinate outside $[-1, 1]$ | rows |
| `store/conditions_improved` | conditions whose lowest physical `mol_energy` stored from any source is below their seed minimum (`Modeller._store_seed_best`, taken when the store is seeded, before a restore) by more than `Modeller.STORE_IMPROVED_RTOL` of it | conditions |
| `store/bwd_draw_kicked_frac` | share of the bwd rows drawn since the last eval that are kicked children (`ORIGIN_KICK`); NaN when none were drawn | fraction |
| `store/replay_draw_age_mean`, `store/replay_draw_reuse_mean` | mean, over the replay rows drawn since the last eval, of steps since birth (0 for a row born after the current step, which a rewind keeps) and of the `draws` column before the draw | steps; occurrences |
| `store/onpolicy_draws_mean`, `store/onpolicy_drawn_frac` | mean `draws` of the drawable on-policy rows; share drawn at least once | occurrences; fraction |
| `store/onpolicy_cond_frac` | share of conditions with any drawable row that hold a drawable on-policy row (the conditions replay draws from) | fraction |
| `store/onpolicy_per_cond_median` | drawable on-policy rows per condition, over the conditions that hold one | rows |
| `store/replay_w_mean`; under a rule of `crystal_store.RULE_FLOORS` also `store/replay_w_one_frac`, `store/replay_w_floor_frac` | the replay rule's weight over the drawable on-policy rows (absent under `uniform`); share at 1 and at the rule's floor | weight; fraction |
| `sg<g>/store_share` | share of live rows in space group g, on a run over several groups | fraction |
| `eval/pool_rows`, `eval/pool_fresh_rows` | this eval's rows read from the rollout pool; rows it drew fresh | rows |
| `eval/pool_oldest_age_steps` | steps since the oldest rollout read by this eval; NaN when it read none | steps |

*The store's readouts. The counters from `store/rows` to `store/kick_in_window_frac` run over the whole run; the `*_draw_*` keys cover the training draws since the last eval (`Modeller._tally_store_draw`, in `Modeller._store_draw` before rows are tiled by `repeats`, counting the rows of a step that then fails; `Modeller._store_draw_readouts` clears them at each read); the on-policy, weight and `conditions_improved` keys read the store as it stands (`Modeller._store_population_readouts`, O(rows)).*

Under `cfg:smoke`, `Modeller._store_check_currency` re-scores up to `Modeller.STORE_CURRENCY_ROWS` rows of each origin present through `x` and the template at every eval, and raises when $\log R$ differs by more than `Modeller.STORE_CURRENCY_TOL` nats or `mol_energy` by more than that times $T$ plus `Modeller.STORE_CURRENCY_REL` times $|E|$. The energy bar judges only rows at or below the armed energy clip (`energy_clip` of the energy function): above it the reward reads the energy log-compressed, and an MLIP's float32 energy of an overlapping cell moves with the batch it is scored in (MACE-CSP on acridine, 2026-11-25: a row at 9e5 kJ/mol by about 2e2, its $\log R$ by 1e-4 nats), so such rows are judged by $\log R$ alone. On the `uma` route both bars are `Modeller.STORE_CURRENCY_UMA` times wider: it runs TF32, and the same crystal re-scored in another batch moved by 0.05 kJ/mol at 482 kJ/mol ($\log R$ 0.02 nats; UMA-s 1.1, acridine, 2026-11-25). Its message names the row furthest past its bar, with both energies and the row's packing coefficient.

## Refusals

`config_invariants.py::row_store_is_well_formed` reports, and `init_row_store` raises on, an enabled store with: `model.kind` other than nf; the conformer route; `cfg:temperature_conditioning` with a stage that reads rewards (`config_invariants.py::_stage_reads_reward`: a forward or replay share, a forward sidecar or rollout cadence, or a backward reward term), or without `cfg:noised_prior.kick_kappa_per_atom`, or without a `cfg:energy_config.log_temperature_range` [lo, hi] with lo < hi; `cfg:noised_prior.kick_kappa_per_atom` without `cfg:temperature_conditioning`; `cfg:energy_config.prior_flow_path`; `cfg:buffers.replay_buffer.prioritise.enabled`; `cfg:buffers.replay_buffer.val_frac` above 0; `cfg:buffers.anchor_buffer.frozen` or `cfg:buffers.anchor_buffer.seed_floors_only` not true; `cfg:noised_prior.live_kernel` other than `'physical'`; `cfg:noised_prior.train_kinds` other than `['anchor']`; `cfg:energy_config.crystal_coordinates` other than `fd`; `cfg:buffers.store.ram_gb` unset; a malformed `ram_frac` or `mle_draw`; a `replay_weight` outside `crystal_store.py::REPLAY_WEIGHTS` (`config_invariants.py::replay_weight_rules`); `replay_weight` `duty` with `rotate_frames` or `mirror_frames`; a key of `config_invariants.py::ROW_STORE_RETIRED_KEYS`; `cfg:buffers.store.keep_rollouts` not a boolean, or false with a stage whose replay frac is above 0 (replay would draw only the eval's stored rows); `cfg:buffers.store.growth_every` not an integer of at least 1; `cfg:condition_log_z.rollout_condition_draw` `under_drawn`; `cfg:condition_log_z.rollout_group_draw` other than `uniform`; `cfg:buffers.replay_buffer.admit_from_eval` false; `model.nf.cond_noise` above 0 with a bwd or replay `tb` term; a stage without `max_steps`; `cfg:epochs` below the sum of the stage caps plus `config_invariants.py::ROW_STORE_STAGE_MARGIN` per stage. Per active stage: `cfg:stage.z_pin_rollout_every` above 0, an action of `config_invariants.py::ROW_STORE_REFUSED_ACTIONS`, any `bootstrap_z` form but `bootstrap_z:offset` (`config_invariants.py::row_store_refuses_action`: the others fit the $Z(c)$ head on std-oriented `cfg:molecules_path` rows instead of the store's templates), `cfg:stage.condition_draw`, `cfg:stage.replay_warmup_rows` above 0, replay `mle`, forward `pooled_vg`, backward or replay `vg_lb` or `vg_lme`, `weighted_bwd_sampling`, a non-empty `cfg:stage.coeff_schedule` or `balance.anneal_coeffs`, the `drift_std_max`, `ess_min` or `val_gap_max` bars of `cfg:stage.fwd_rollout_triggers` above 0, `lr_sensor` kind ray, and `flags.z_calibration` with `cfg:z_calibration.mode` replay. An enabled block refuses unknown keys; a block whose `cfg:buffers.store.enabled` is not true is not inspected. `inn_modeller.py::InnModeller.check_buffer_options` refuses an enabled store.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:buffers.store.{enabled, ram_gb, ram_frac, mle_draw, replay_weight, keep_rollouts, growth_every}`, `cfg:noised_prior.{live_kernel, train_kinds, window_kT, window_by, packing_window, live_log_range, live_redraws}`, `cfg:buffers.anchor_buffer.seed_floors_only`, `cfg:stage.fwd_rollout_every`, `cfg:eval_num_samples`, `cfg:test_eval_num_samples`, `cfg:batch_size`.

Code: `crystal_store.py::CrystalStore.seed`, `.add`, `.thin`, `.draw`, `.draw_anchors`, `.draw_weighted`, `.draw_accepted`, `.mol_batch`, `.touch`, `.invalidate_q_after`, `.stats`, `.check`, `.state_dict`, `.apply_state`; `crystal_store.py::RolloutPool`; `train.py::Modeller.store_settings`, `._store_take_rows`, `._store_frame_check`, `._store_after_identifiers`, `._store_headroom`, `.init_row_store`, `._store_warm_tracker`, `._store_rewind`, `._kick_anchors`, `._kick_fresh`, `._kick_labels_temperature`, `._kick_temperature`, `._store_replay_rows`, `._store_draw`, `._store_touch`, `._store_admit`, `._replay_rows`, `._store_condition_batch`, `._store_heldout_panel`, `._growth_kicks_due`, `._owe_growth`, `._store_grow`, `._store_check_currency`, `._store_metrics`, `._eval_pool_add`, `._eval_pool_refusal`, `._pool_eval_stats`, `._eval_forward_rows`, `._fwd_eval_summary`; `config_invariants.py::row_store_settings`, `::store_row_limit`, `::row_store_problems`, `::row_store_is_well_formed`.

## Could be tooling

The store's columns are enough to tabulate rows per condition, kind and origin, and the share of undrawn kicks, from a `_buffers.pt` sidecar's `row_store` state without building a policy.

## Sources

Repo: `energy_sampling/crystal_store.py`, `energy_sampling/train.py`, `energy_sampling/noised_prior.py`, `energy_sampling/physical_kick.py`, `energy_sampling/config_invariants.py`, `energy_sampling/gflownet_losses.py` and the `buffers.store` block of the canonical config, read at the stamped commit with the change that wires the store in. Tests: `tests/crystal/test_crystal_store.py`, `tests/crystal/test_row_store_trainer.py`, `tests/crystal/test_store_tb.py`, `tests/crystal/test_store_growth.py`, `tests/crystal/test_eval_pool.py`, `tests/crystal/test_store_keep_rollouts.py`, `tests/config/test_row_store_invariants.py`.
