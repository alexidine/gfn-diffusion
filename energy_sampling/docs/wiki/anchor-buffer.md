# Anchor buffer

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

The *anchor buffer* is a permanent archive of low-energy states, one row per stored structure, held apart from the two training stores and never drawn from during a train step. Its rows are jittered and rescored to fill the prior buffer, and their energies are read by the code gating that fill. The store is `buffer.py::AnchorBuffer`, a subclass of `buffer.py::CrystalBuffer` adding three per-row columns, `reward`, `energy` and `original_surprise`, plus two mutators of its own, `admit` and `thin`. The graph variant is `buffer.py::ConformerAnchorBuffer`. What happens to a drawn anchor after it leaves here is [prior-buffer](prior-buffer.md); the geometry of the jitter is [prior-buffer-row-geometry](prior-buffer-row-geometry.md); when a sidecar's rows are current is [checkpoints-and-resume](checkpoints-and-resume.md).

`AnchorBuffer.energy` holds $E_{\text{anchor}}$, the row's training total evaluated at $\lambda = 1$, named `ANCHOR_ENERGY_CURRENCY` (`'e_anchor'`) and recorded on disk as `energy_currency`. Without a `prior_flow`, $E_{\text{anchor}}$ *is* the site's own energy; with one it is read off the scored batch as `physical_energy + bounding_energy * bounding_coeff` by `train.py::Modeller._anchor_energy_phys`, and `train.py::Modeller._anchor_energy` returns the pair $(E_{\text{anchor}}, \texttt{energy\_phys})$ every anchor site uses. `reward` and `energy` are separate per-row columns: $\text{reward} = -E/T$, and $T$ is not persisted per graph. `original_surprise` is written once at admission and never updated.

## Seeding, by problem domain

`train.py::Modeller.init_anchor_buffer_seed` runs at init and returns immediately if an anchor buffer exists, so a restored one is never clobbered. `cfg:buffers.anchor_buffer.seed_source` takes three forms: `'generated'` leaves the buffer unbuilt, to be bootstrapped later from the first screened batch; `'prior_dataset'` seeds from `self.prior_dataset`, the dataset already loaded for backward training; anything else is a path, `torch.load`ed through the `equalized_prior`/`prior` keys, stripped of lazy space-group caches, rescored, and mapped into the identifier registry (an absent identifier raises). `configs/problems.yaml` sets the key per problem: the molecule entries and `latent_gaussian` take `prior_dataset`; the toy entries take an explicit path to their *conditions* `.pt`. That file's header documents the split.

Seed rows carry no rollout-based surprise measurement, so `original_surprise` is NaN. `thin`'s hard-cap sort fills NaN with $+\infty$ and evicts the lowest first, so seed rows rank last. The seed pass also calls `update_best_energy` on `buffer.py::ConditionLogZTracker` when that object exists: the tracker's minima are separate state from the buffer's rows, and the constructor does not touch them.

## Where membership changes

Six sites; all post-seed mutation funnels through two primitives.

| site | call |
|---|---|
| `train.py::Modeller.init_anchor_buffer_seed` | constructor |
| `train.py::Modeller.screen_and_admit_anchors`, lazy-bootstrap branch | constructor |
| `train.py::Modeller.screen_and_admit_anchors` | `admit`, then overflow `thin` |
| `train.py::Modeller.top_up_prior_from_anchors`, record-breaker block | `admit`, then overflow `thin` |
| `train.py::Modeller.evaluation`, anchor block | cadence `thin` |
| `checkpointing.py::Checkpointer.restore_buffers` | `AnchorBuffer.from_state_dict` |

`cfg:buffers.anchor_buffer.frozen` is checked inside `AnchorBuffer.admit` and `AnchorBuffer.thin`, at the primitive rather than at the call sites, and both return without acting when it is set. The two constructors do not route through either, so a freeze does not reach seeding. The cadence `thin` in `evaluation` is gated on the buffer existing, not on the `buffers_active` stage flag, and it drops rows against a per-condition minimum that only falls, so it can evict while the buffer is far under `max_size`. Its cadence and the surprise sweep's are `cfg:buffers.anchor_buffer.thin_every_n_evals` and `.refresh_every_n_evals`; `0` or absent disables either, and the eval counter they divide is not checkpointed.

`AnchorBuffer.admit` processes candidates best-energy-first against a reference set seeded with the current rows. A candidate within `cfg:buffers.anchor_buffer.dup_cutoff` in latent $L_2$ of the nearest *same-condition* reference point displaces it only if strictly lower in energy, otherwise it is dropped; a candidate farther than that takes a new slot. The displacement path purges the losing slot and adds the winner one for one, so `len(buffer)` (logged as `anchor_buffer_length`) does not move when membership churns that way. The `admit_range` parameter is dead: both call sites pass `None`. `AnchorBuffer.thin` drops rows whose `energy` exceeds `per_condition_min_energy[cid] + energy_window`, then, if still over `max_size`, protects each condition's lowest-energy row and evicts by lowest `original_surprise`. A row admitted by the record-breaker block carries its parent anchor's `original_surprise`, not a freshly measured one.

## Restore, and re-asserting the config's policy

`AnchorBuffer.from_state_dict` restores `reward`, `energy`, `energy_currency` (absent means `None`), `original_surprise` and the mean-energy snapshot, and the inherited `CrystalBuffer.from_state_dict` restores `ema_loss` verbatim. `ema_loss` on this class is a replay priority, not a training loss: it is written only by `train.py::Modeller.refresh_anchor_buffer_surprise` and read by `CrystalBuffer._loss_weights` at draw time. `train.py::Modeller.apply_anchor_buffer_policy` runs after every construction and after the sidecar restore. It sets `frozen` from the config (the flag is not serialized); raises `NotImplementedError` on `cfg:buffers.anchor_buffer.online_loss_flow: true`; and, with no priority writer enabled, overwrites `ema_loss` and `birth_loss` with all-NaN, printing that the draw is uniform when the vector was not already all-NaN. "No priority writer" is `cfg:buffers.anchor_buffer.refresh_every_n_evals` at `0` or absent, the sweep in `refresh_anchor_buffer_surprise` being the only writer. All-NaN is the one state `_loss_weights` treats as uniform: a partially-NaN vector is filled at the `0.90` quantile of the rest.

The same hook is where the currency guard fires. With a `prior_flow`, a restored store whose `energy_currency` is not `'e_anchor'` raises `buffer.py::BufferCurrencyError`: its rows hold $\lambda$-mixed totals at each row's admission-time $\lambda$, while `thin` and `admit` compare them against a $\lambda = 1$ minimum. Both constructors stamp, so only a sidecar restore can trip it.

## The physical-energy minimum

`ConditionLogZTracker` keeps two per-condition running minima: `best_energy`, the mixture being sampled, and `best_energy_phys`, the same per-sample total at $\lambda = 1$. `update_best_energy` scatter-reduces both with `amin`, drops any row non-finite in *either* leg so the two visited sets are identical by construction, and raises when `energy_phys` is omitted on a run with a `prior_flow`. Five sites call it: `gflownet_losses.py::update_condition_best_energy`, and in `train.py` the forward eval-sampling pass, `top_up_prior_from_anchors`, `seed_prior_from_condition_minima` and `init_anchor_buffer_seed`. `best_energy_phys` is what every anchor decision reads: `thin`'s energy window, the plausibility window `cfg:buffers.anchor_buffer.screen_energy_window`, and the record-breaker test.

## Noise on the way out, and the tile sidecar

`train.py::Modeller._noise_and_condition` is the single seam through which a drawn anchor is jittered and then conditioned, in that order. Its default branch applies `log_noise_latent_parameters(*cfg:buffers.anchor_buffer.noise_log_range)`, a log10-uniform isotropic latent kick. Both anchor-sourced prior fills pass through it: `top_up_prior_from_anchors`, drawing priority-weighted with a random floor `cfg:buffers.anchor_buffer.replay_beta`, and `seed_prior_from_condition_minima`, tiling each condition's lowest-energy row (`AnchorBuffer.best_per_condition_indices`).

Under `cfg:buffers.anchor_buffer.tile: 'shaped'` the isotropic kick is replaced by a per-anchor Gaussian read from a sidecar at `cfg:buffers.anchor_buffer.shape_path`, and `noise_log_range` is then unread. The sidecar is bound by `train.py::Modeller._load_anchor_tile`, called at the end of `apply_anchor_buffer_policy` so every construction and every restore passes through it. Its checks are fatal, with no fallback to the isotropic draw: `shape_path` is set, the file exists, `format_version` is `1`, `n` and `dim` match the anchor set, and `anchor_x_sha1` equals the SHA-1 of the live `AnchorBuffer.x` over contiguous float32 bytes. The tile is positional, so a call site supplying no `anchor_inds` raises. The shape of the draw itself is [prior-buffer-row-geometry](prior-buffer-row-geometry.md).

## The `lj_coeff` stamp

Independently of the anchor currency, every crystal row carries an `lj_coeff` stamp, and `CrystalBuffer.state_dict` writes `format_version` (`BUFFER_FORMAT_VERSION`, `2`) and the store's uniform stamp beside the rows. `CrystalBuffer._refuse_unknown_currency`, run at the end of `from_state_dict`, refuses a crystal store whose rows carry no stamp, and a format-version-2 dict whose recorded `lj_coeff` disagrees with its rows. Its message names `migrate_buffer_sidecar.py` as the one sanctioned re-stamp, over `buffer.py::migrate_legacy_lj_coeff`.

## Retired

`cfg:buffers.anchor_buffer.mcmc` and the local Metropolis reheat it configured were deleted. The key sits in the retired table in `utils.py` and the drop list in `config_state.py`, so a config carrying it is refused at load, with a message saying the reheat is in git history and that anchors seed from `prior_dataset`.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:buffers.anchor_buffer.frozen`, `.online_loss_flow`, `.thin_every_n_evals`, `.refresh_every_n_evals`, `.seed_source`, `.max_size`, `.dup_cutoff`, `.screen_energy_window`, `.thin_energy_window`, `.surprise_cutoff`, `.confirm_cutoff`, `.confirm_k`, `.health_gate_floor_metric`, `.health_gate_floor`, `.health_gate_ceiling_metric`, `.health_gate_ceiling`, `.replay_beta`, `.noise_log_range`, `.tile`, `.shape_path`, `.tile_temperature`, `.tile_width_cap`, `.topup_admit_record_breakers`.

Code: `buffer.py::AnchorBuffer` (`.state_dict`, `.from_state_dict`, `.purge_by_index`, `.admit`, `.thin`, `.best_per_condition_indices`); `buffer.py::ConformerAnchorBuffer`; `buffer.py::CrystalBuffer.state_dict`, `.from_state_dict`, `._refuse_unknown_currency`, `._loss_weights`; `buffer.py::migrate_legacy_lj_coeff`, `BUFFER_FORMAT_VERSION`, `ANCHOR_ENERGY_CURRENCY`, `BufferCurrencyError`; `buffer.py::ConditionLogZTracker.update_best_energy`; `train.py::Modeller.init_anchor_buffer_seed`, `.apply_anchor_buffer_policy`, `._load_anchor_tile`, `._noise_and_condition`, `._shaped_anchor_tile`, `._anchor_energy`, `._anchor_energy_phys`, `.screen_and_admit_anchors`, `.top_up_prior_from_anchors`, `.seed_prior_from_condition_minima`, `.refresh_anchor_buffer_surprise`, `.evaluation`; `gflownet_losses.py::update_condition_best_energy`; `checkpointing.py::Checkpointer.restore_buffers`.

## Could be tooling

Both checks here are mechanical and belong before a run is launched, not at its first eval. Given a buffer sidecar: print the anchor store's row count, `energy_currency`, `format_version`, stored `lj_coeff`, the NaN fraction of `ema_loss` and of `original_surprise`, and the per-condition row counts. That is everything `apply_anchor_buffer_policy` would raise or blank on. Given a tile sidecar beside it: run `_load_anchor_tile`'s checks offline.

## Sources

The code above, read at the stamped commit; the `buffers.anchor_buffer` block of `configs/mk_dev.yaml`; and the `anchor seed_source` header and per-problem blocks of `configs/problems.yaml`. Six memory files (project_anchor_buffer_freeze_mechanics, project_anchor_seed_mcmc_reheat, project_anchor_seeding_policy_by_domain, project_phys_energy_guard_without_mechanism, project_lj_stamp_lost_on_buffer_restore, project_thermal_tile_prototype) located the code and were not used as evidence.
