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
5. Three frozen graph panels are cut for the readouts, each capped at $4 \times$ `cfg:eval_num_samples` rows and deterministic in the file and `cfg:seed`: `prior_dataset` (in-window training children), `prior_heldout_batch` (held-out children, `select_rows`' `heldout_children`) and `prior_anchor_batch` (in-window training anchors). `prior_dataset` is built with `buffer.py::CrystalBuffer`'s `orient=False`, as are the two conditions datasets on this route.

The returned energies are $-T \log R$ of the kept rows, so the reward clip is measured over every scored row. The file batch stays held in `Modeller._store_load` until `Modeller.init_row_store`.

`Modeller._store_after_identifiers` then gives each stored row its condition id and adds one template per training condition with no stored row, from that molecule's `cfg:molecules_path` row; the load prints their count. `Modeller.init_row_store` runs after the clip and energy reference are armed: it reads each row's final $\log R$ and `mol_energy` (`Modeller._stored_reward_terms`), sizes the store from `config_invariants.py::store_row_limit`, seeds and checks it, warms `condition_log_z`'s per-condition floors from $-T \log R$ of every row given (`Modeller._store_warm_tracker`), applies a sidecar state when one is parked (`Modeller._store_restore_state`, which raises on another run's stamp) and releases the file batch. `init_prior_buffer_seed` and `init_anchor_buffer_seed` return early on this route.

Headroom $H$ (`Modeller._store_headroom`) is the largest single add: the largest of twice a rollout ($B \times$ forward repeats), `cfg:eval_num_samples`, and any `bootstrap_z:offset` or `bootstrap_z:rollout` draw size. The load refuses a row limit below anchors $+ 2H$. Over the limit `CrystalStore.thin` frees rows 50:50 between children and on-policy rows, uniformly within each; anchors are never thinned and children never fall below $H$.

## Phase 1

`draw_bwd_sample` calls `Modeller._store_draw('mle', repeats)` in a `bwd_sampling_mode` `'dataset'` stage. It draws $B$ anchors (`cfg:buffers.store.mle_draw`: `anchor` uniform over anchors, `condition` condition-first) and kicks each with `physical_kick.py::kick` ([prior-buffer](prior-buffer.md), the physical kernel). The kick uses the run's own `torch.Generator`, seeded from `cfg:seed` plus the launch step, and neither reads nor advances the global generators. A row whose kick returned the anchor's own point is replaced by a kick of another anchor, at most `Modeller.MLE_KICK_ROUNDS` times, then dropped and counted (`live_kernel/mle_kept_dropped`); an empty draw raises. So the batch holds no anchor point. The draw carries no reward, writes no store column (`inds` is None), and raises in a stage whose backward loss reads $\log R$.

## Phase 2

| branch | draw | write |
|---|---|---|
| bwd | uniform condition, uniform drawable child | `log_q`, `resid`, `q_step`, `draws` |
| replay | uniform condition, then an on-policy row by `cfg:buffers.store.replay_weight` | the same |
| fwd | a rollout of $B$ molecules, each with a uniform space group and $Z'$, on the condition's template | every scored row, on-policy |
| growth | one kick per rollout row written | one child per kick |

*Each branch's source of rows and what it writes back to the store.*

`Modeller._store_draw` builds the batch from `CrystalStore.mol_batch`, so a stored row is scored and trained against its condition's template. The replay path is the row's terminal only: the policy is a flow, and `traj` is a two-state stack of zeros and the terminal. Replay draws no rows while the pool of drawable on-policy rows is empty, and `fused_train_step` then folds replay's share into bwd.

`crystal_store.py::REPLAY_WEIGHTS` maps a rule name to a function `rule(store, rows, step)` that returns one weight per row, or None for uniform within the condition. The registered rule is `uniform`. A rule other than it goes through `CrystalStore.draw_weighted`, whose correction $1 / (n_c\, p)$ is handed to the loss as the sample weight.

`Modeller._store_admit` writes each rollout row with $\log q$, residual and `q_step`; `CrystalStore.add` drops and counts rows with a non-finite $x$, $\log R$ or energy (`store/nonfinite_rows`) and stores a row with a finite reward and non-finite $\log q$ as `UNDRAWN`. `manage_replay_buffer` calls it before its `replay_in_play` check, so every stage's rollouts are kept.

| rows | stored | drawable |
|---|---|---|
| training rollout | yes | yes |
| eval scored in a stage with `buffers_active` | yes, `EVAL` | yes |
| eval scored in any other stage, the one that ends MLE included | yes, `EVAL` | no |
| `bootstrap_z:offset` at stage entry | yes, `EVAL`, `ORIGIN_BOOTSTRAP` | no |

*How each kind of on-policy rollout enters the store. `evaluation` reads `buffers_active` before `protocol.maybe_advance`, so rows are judged by the stage that scored them.*

The held-out molecules are read through a fixed panel (`Modeller._store_heldout_panel`): $\min(\lfloor n / (2 n_{sg}) \rfloor, \text{test molecules})$ molecules, at least one, for $n$ = `cfg:test_eval_num_samples`, chosen once from `cfg:seed`, each scored twice in every space group of the run. Those rows are not stored.

## Growth

`Modeller._store_grow` runs after each training step. For each kick owed (`Modeller._growth_kicks_due`: one per on-policy row the last training rollouts wrote) it draws a uniform condition and a uniform anchor of it, kicks it, makes one energy call and stores the result as a child linked to that anchor. A kick that kept its anchor spends no energy call and stores nothing. So the scoring budget is split evenly between rollout rows and kicks, at $2BR/N$ scores per step for rollout period $N$ = `cfg:stage.fwd_rollout_every`.

A kick outside the packing window, or whose seed energy (`MolecularCrystal.seed_energy_from`) is above the frozen `window_floor` by more than `cfg:noised_prior.window_kT` $\times T$, is stored `UNDRAWN`. The comparison does not read `best_energy`, so a falling minimum does not re-filter rows. Both `condition_log_z` minima take the scored rows through `Modeller._anchor_energy`.

## Rewind and resume

The divergence rewind calls `Modeller._store_rewind`, which runs `CrystalStore.invalidate_q_after` at the checkpoint's step: a row's `log_q` written after it returns to its birth value, or to NaN. No per-eval refresh of stored $\log q$ exists on this route.

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
| `sg<g>/store_share` | share of live rows in space group g, on a run over several groups | fraction |

*The store's readouts; each is a counter over the run so far unless named a share.*

Under `cfg:smoke`, `Modeller._store_check_currency` re-scores up to `Modeller.STORE_CURRENCY_ROWS` rows of each origin present through `x` and the template at every eval, and raises when $\log R$ differs by more than `Modeller.STORE_CURRENCY_TOL` nats or `mol_energy` by more than that times $T$.

## Refusals

`config_invariants.py::row_store_is_well_formed` reports, and `init_row_store` raises on, an enabled store with: `model.kind` other than nf; the conformer route; `cfg:temperature_conditioning`; `cfg:energy_config.prior_flow_path`; `cfg:buffers.replay_buffer.prioritise.enabled`; `cfg:buffers.replay_buffer.val_frac` above 0; `cfg:buffers.anchor_buffer.frozen` or `cfg:buffers.anchor_buffer.seed_floors_only` not true; `cfg:noised_prior.live_kernel` other than `'physical'`; `cfg:noised_prior.train_kinds` other than `['anchor']`; `cfg:energy_config.crystal_coordinates` other than `fd`; `cfg:buffers.store.ram_gb` unset; a malformed `ram_frac` or `mle_draw`; a `replay_weight` outside `crystal_store.py::REPLAY_WEIGHTS` (`config_invariants.py::replay_weight_rules`); a key of `config_invariants.py::ROW_STORE_RETIRED_KEYS`; `cfg:condition_log_z.rollout_condition_draw` `under_drawn`; `cfg:condition_log_z.rollout_group_draw` other than `uniform`; `cfg:buffers.replay_buffer.admit_from_eval` false; `cfg:grow_batch_size` true; `model.nf.cond_noise` above 0 with a bwd or replay `tb` term; a stage without `max_steps`; `cfg:epochs` below the sum of the stage caps plus `config_invariants.py::ROW_STORE_STAGE_MARGIN` per stage. Per active stage: `cfg:stage.z_pin_rollout_every` above 0, an action of `config_invariants.py::ROW_STORE_REFUSED_ACTIONS`, `cfg:stage.condition_draw`, `cfg:stage.replay_warmup_rows` above 0, replay `mle`, forward `pooled_vg`, backward or replay `vg_lb` or `vg_lme`, `weighted_bwd_sampling`, a non-empty `cfg:stage.coeff_schedule` or `balance.anneal_coeffs`, the `drift_std_max`, `ess_min` or `val_gap_max` bars of `cfg:stage.fwd_rollout_triggers` above 0, `lr_sensor` kind ray, and `flags.z_calibration` with `cfg:z_calibration.mode` replay. An enabled block refuses unknown keys; a block whose `cfg:buffers.store.enabled` is not true is not inspected. `inn_modeller.py::InnModeller.check_buffer_options` refuses an enabled store.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:buffers.store.{enabled, ram_gb, ram_frac, mle_draw, replay_weight}`, `cfg:noised_prior.{live_kernel, train_kinds, window_kT, window_by, packing_window, live_log_range, live_redraws}`, `cfg:buffers.anchor_buffer.seed_floors_only`, `cfg:stage.fwd_rollout_every`, `cfg:eval_num_samples`, `cfg:test_eval_num_samples`, `cfg:batch_size`.

Code: `crystal_store.py::CrystalStore.seed`, `.add`, `.thin`, `.draw`, `.draw_anchors`, `.draw_weighted`, `.mol_batch`, `.touch`, `.invalidate_q_after`, `.stats`, `.check`, `.state_dict`, `.apply_state`; `train.py::Modeller.store_settings`, `._store_take_rows`, `._store_frame_check`, `._store_after_identifiers`, `._store_headroom`, `.init_row_store`, `._store_warm_tracker`, `._store_restore_state`, `._store_rewind`, `._kick_anchors`, `._kick_fresh`, `._store_replay_rows`, `._store_draw`, `._store_touch`, `._store_admit`, `._replay_rows`, `._store_condition_batch`, `._store_heldout_panel`, `._growth_kicks_due`, `._store_grow`, `._store_check_currency`, `._store_metrics`; `config_invariants.py::row_store_settings`, `::store_row_limit`, `::row_store_problems`, `::row_store_is_well_formed`.

## Could be tooling

The store's columns are enough to tabulate rows per condition, kind and origin, and the share of undrawn kicks, from a `_buffers.pt` sidecar's `row_store` state without building a policy.

## Sources

Repo: `energy_sampling/crystal_store.py`, `energy_sampling/train.py`, `energy_sampling/noised_prior.py`, `energy_sampling/physical_kick.py`, `energy_sampling/config_invariants.py`, `energy_sampling/gflownet_losses.py` and the `buffers.store` block of the canonical config, read at the stamped commit with the change that wires the store in. Tests: `tests/crystal/test_crystal_store.py`, `tests/crystal/test_row_store_trainer.py`, `tests/crystal/test_store_tb.py`, `tests/crystal/test_store_growth.py`, `tests/config/test_row_store_invariants.py`.
