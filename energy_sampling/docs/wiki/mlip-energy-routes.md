# MLIP energy routes

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

Two of the `cfg:energy_function` values name a machine-learned interatomic potential as the crystal energy: `uma` (fairchem UMA, loaded through `pretrained_mlip.load_predict_unit`) and `mace` (a MACE checkpoint loaded by `AL_mace_utils.py::load_mace_model`). This page covers how a batch of sampled crystals becomes a number on those two routes: the dispatch, the graph each backend is handed, the lattice energy's definition, and how the call fails. The reward built on top of that number is [reward-construction](reward-construction.md); the force field behind `elj` is [crystal-force-fields](crystal-force-fields.md); the occupancy sensors the timers feed are [gpu-occupancy](gpu-occupancy.md).

## Dispatch

`MolecularCrystal.__init__` builds one predictor at construction and holds it for the run: `init_uma_crystal_predictor` on `uma`, `load_mace_model(self.mlip_path, self.device, torch.float32)` on `mace`, and `self.predictor = None` on every other energy function. The weights path is the top-level `cfg:mlip_path`, read on these two routes only.

The request list is `MolecularCrystal.computes`, `['reduction_en']` plus the energy function's own name. `energies/molecular_crystal.py::MolecularCrystal.analyze_crystal_batch` passes it to `crystal_analysis.py::MolCrystalAnalysis.analyze` with `cutoff=10`, `supercell_size=10`, `std_orientation=False` and `predictor=self.predictor`, merged over `cfg:energy_config.analyze_kwargs`. `uma` and `mace` are `False` in `COMPUTES_REQUIRE_CLUSTER` and `True` in `COMPUTES_REQUIRE_UNIT_CELL`, so an MLIP route takes `analyze`'s `mol2ucell` branch and builds the home unit cell only, not the exploded supercell. `_init_computes` binds `uma` to `compute_lattice_uma` and `mace` to `compute_lattice_mace`. `analyze` consumes `std_orientation` as its own parameter and does not forward it into `compute`, so the orientation reaching the model is the one `mol2ucell` applied, while each compute function runs at its own signature default: `True` on `compute_lattice_uma` and `compute_lattice_mace`, `False` on the two gas-phase legs.

## The UMA route

The crystal leg reaches `uma_utils.py::compute_crystal_uma_on_mxt_batch`. That function first runs a density guard: while any `packing_coeff` exceeds `max_cp=2.0` it adds 2 to that row's `cell_lengths` and re-runs `box_analysis`. It then builds the fairchem batch in one shot with `batch_to_fairchem_batch` -- a single `AtomicData` for the whole batch, the asymmetric unit's atomic numbers tiled `sym_mult` times to match `unit_cell_pos`. `USE_VECTORISED_ATOMICDATA`, read from `MXT_VECTORISED_ATOMICDATA` and on unless the variable is `0`, selects that build; with it off the per-crystal list path runs instead. `verify_fairchem_batch_equivalence` builds both and raises on any shape, dtype or value difference over a fixed field list; its callers are tests, not the energy path.

The neighbour list is external by default. `crystal_inference_settings` sets `external_graph_gen` from `USE_UMA_EXTERNAL_GRAPH` (`MXT_UMA_EXTERNAL_GRAPH`, default on), and the call site reads the predictor's own setting through `_predictor_wants_external_graph` rather than the module flag. When it is on, `attach_external_graph` fills `edge_index`, `cell_offsets` and `nedges` from `pbc_neighbours.py::batched_pbc_neighbour_list` at the radius read off the backbone, remapping index order and offset ownership to fairchem's convention and sorting edges by graph. When it is off, the model runs its own `otf_graph` inside the forward. The rest of the settings object is `tf32=True`, `merge_mole=False`, `internal_graph_gen_version=2`, and `activation_checkpointing`, `compile` and `edge_chunk_size` from the three `cfg:energy_config.mlip_*` keys. `_build_uma_crystal_predictor` overrides `direct_forces`, `regress_forces` and `regress_stress` to `False` and pops the `omc_forces` and `omc_stress` tasks. The three `mlip_*` keys are read on the `uma` branch of `MolecularCrystal.__init__` only, and setting any of them off its default prints a one-line notice there.

`drain_uma_phase_timing` reports per-call seconds under `guard`, `build`, `forward` and `ext_graph`, plus the crash counters as zero when nothing crashed. A `graph` phase, nested inside `forward` and excluded from the total, is reported only when `MXT_UMA_GRAPH_TIMER` (default off) installed the wrapper on fairchem's two `generate_graph` bindings and it fired.

## The MACE route

`compute_lattice_mace` reaches `AL_mace_utils.py::compute_crystal_mace_on_mxt_batch`, which selects a builder from three module flags and the periodicity of the call. `gpu_batch` is `USE_GPU_MACE_BATCH and USE_BATCHED_MACE_NEIGHBOURS and pbc`; on that branch `batch_to_mace_input_dict` builds the collated MACE input dict directly on device, with `batched_pbc_neighbour_list` supplying the edges. Otherwise the builder is `batch_to_mace_atomicdata_hoisted`, or `batch_to_mace_atomicdata` when `USE_HOISTED_MACE_ATOMICDATA` is off, and the per-crystal objects go through PyG's `Collater` and a device transfer. All three flags are on unless their variable is `0`.

The split is keyed on `pbc`. `batch_to_mace_input_dict` raises `ValueError` when `pbc` is false, stating that the non-periodic leg stays on the reference builder, and the batched neighbour list inside `batch_to_mace_atomicdata_hoisted` is likewise scoped to `pbc=True`; the non-periodic leg therefore runs MACE's host-side `get_neighborhood` per graph on numpy arrays. That branch rewrites the `cell` array it was handed in place, to `max_positions * 5 * cutoff` along each axis, and the builder uses the cell it returns. Whichever builder ran, the caller then overwrites `positions` with `unit_cell_pos` passed through `T_cf` and back through `T_fc`, and recomputes `shifts` from `unit_shifts` through `T_fc`, so those two fields carry gradient while `cell` is detached. `drain_mace_phase_timing` reports `build`, `collate`, `xfer`, `forward` and the nested `neighbours`, and gives `energy/mace_flag_gpu_batch` as a fraction of calls that executed the branch rather than a flag value.

## The gas-phase reference and the lattice energy

Both routes define the same quantity. `compute_lattice_uma` returns `(uma_pot / (sym_mult * z_prime) - uma_gas_pot) * 96.485`, and `compute_lattice_mace` the same with its own attributes: the periodic energy per molecule minus an isolated-molecule reference, converted from eV to kJ/mol. Each leg is skipped when its attribute is already on the batch.

`compute_lattice_gas_phase_uma` and `compute_lattice_gas_phase_mace` build that reference by cloning the batch, restricting to `aux_ind == 0` where a cluster is present, calling `reset_sg_info(sg_ind=1)` and `box_analysis` to put the molecule in a P1 cell, and scoring at `pbc=False` with `force_rebuild=True`. At `z_prime > 1` the batch is split by `split_to_zp1_batch`, scored per conformer and scattered back with `reduce='mean'`.

`MolecularCrystal.attach_gas_phase_reference` hosts that value. Keyed on `mol_id`, it calls the same `compute_lattice_gas_phase_*` on the first crystal seen for each unseen molecule, caches the scalar in `self._gas_pot_cache` (not checkpointed), and writes it onto the batch as `uma_gas_pot` or `mace_gas_pot` so the leg is skipped; a batch with no `mol_id` is left alone, and a non-finite value raises rather than being cached. `host_gas_phase_reference` defaults to `True` in the constructor and is absent from `configs/mk_dev.yaml`. `gas_reference_audit` recomputes the leg for a few live rows and reports the drift in kJ/mol.

## What is scored, and when

`MolecularCrystal.energy` routes through `batched_analyze_crystal_batch`. With `cfg:energy_config.internal_oom_recovery` false -- its value in `configs/mk_dev.yaml` -- the whole batch is analyzed in one call and an OOM propagates to the caller. With it true, the batch is chunked from `self.batch_size`, seeded at 1000 on `uma`/`mace` and 10000 otherwise; a successful chunk grows the size by 1% until the first OOM, an OOM classified by `is_cuda_oom` shrinks it to 65% and prints `OOM in energy evaluation: dropping chunk size to ...`, and a size of 1 that still OOMs asserts. `train.py`'s prior re-analysis passes `internal_oom_recovery=True` explicitly.

Only the forward branch calls the MLIP. `get_loss_reward` scores the terminal states of a fresh rollout, so a rollout step scores the full forward batch in one call; the backward and replay branches read a stored per-graph attribute named after the energy function through `prebuilt_sample_to_reward`. Off rollout steps there is no MLIP call.

## Grad state

fairchem's `_run_inference` selects `torch.no_grad()` only when `direct_forces` is true, and this predictor sets it false, so the UMA forward leaves grad enabled unless an outer context disables it. `analyze_crystal_batch` wraps the `analyze` call in `torch.set_grad_enabled(keep_grads)`, and `attach_gas_phase_reference` runs before that block and detaches its own result. `get_loss_reward` uses `torch.no_grad()` when `loss_coeffs.reward_grads == 0` and `torch.enable_grad()` otherwise. `prebuilt_sample_to_reward` and `gas_reference_audit` carry `@torch.no_grad()` decorators. `train.py`'s prior re-analysis wraps its `batched_analyze_crystal_batch` call in an explicit `torch.no_grad()`; `keep_grads=False` detaches the output only.

## Failure and non-determinism

`safe_predict_uma` synchronises on both sides of `predictor.predict`, re-raises anything `is_cuda_oom` classifies as OOM, and on any other `RuntimeError` prints `UMA error (attempt ...)`, retries once after a guarded `synchronize`/`empty_cache`, then counts a crash and returns a crashed flag. The caller substitutes `_crashed_energy`, a tensor of NaN of width `num_graphs` -- not zeros. A success resets the streak; at `MAX_CONSECUTIVE_CRASHES` (3) the substitution stops and a `RuntimeError` is raised chained to the last error. `safe_predict_mace` has the same synchronise-and-classify shape but no retry and no streak bound; its caller substitutes NaN inline. Both drains report their crash call and row counts including as zero. `MolecularCrystal._TOLERATES_NONFINITE` is `('uma', 'mace')`, so `_assert_finite_energy` does not raise on these routes. `tests/crystal/test_mlip_crash_containment.py` covers the substitution and its downstream containment on both routes.

`crystal_inference_settings` sets `tf32=True` against fairchem's `False` default, so the same UMA path run twice does not reproduce itself and the GPU gates assert a control comparison rather than equality. `docs/mlip_validation.md` attributes most of the measured MACE gap to the fractional position round-trip above rather than to precision.

- **[calibration, mlip_validation.md GPU gates, 2026-08-30]** UMA, local RTX 5080: production (`tf32=True`) against a stock fairchem ASE workflow, 1.55-1.64e-2 eV; the same comparison at `tf32=False`, 2.3-3.8e-5 eV, with a same-stack rerun control of 4.6-7.6e-5 eV.
- **[calibration, test_uma_gpu_real_batches.py, 2026-08-30]** same-path rerun spread 4.79e-3 eV, vectorised against list 4.15e-3 eV, on an energy scale of about 837 eV; at `tf32=False` both fall to about 5e-5 eV.

## The ground-truth gates

`docs/design/dependency_validation_protocol.md` names three levels: L0 compares two of our own routes, L1 our production path against the dependency's own documented workflow, L2 that workflow against published numbers. The test files are in the `mxtaltools` repo. L1 is `tests/test_uma_vs_stock_fairchem.py`, whose checkpoint comes from the `--uma-checkpoint` option declared in `tests/conftest.py` or, failing that, from `UMA_CHECKPOINT`, and `tests/test_mace_vs_stock.py`, which reads `MACE_CHECKPOINT`; both skip without one. The L2 check inside `test_uma_vs_stock_fairchem.py` requires `UMA_CHECKPOINT` specifically and skips when the stored reference names a different checkpoint. `tests/test_pbc_neighbours.py` checks the production neighbour list against matscipy and needs no checkpoint. L0 is `test_uma_atomicdata_vectorisation.py`, `test_uma_external_graph.py`, `test_mace_atomicdata_vectorisation.py`, `test_uma_gpu_real_batches.py` and `test_mace_gpu_real_batches.py`. A GPU-touching test requests the session-scoped `gpu` fixture, which skips unless `gpu_preflight()` passes: no CUDA device, a refusal from `gpu_guard.describe()`, or free VRAM under `MIN_FREE_MB` (6000). `MXT_SKIP_GPU_PREFLIGHT` bypasses it.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:energy_function`, `cfg:mlip_path`, `cfg:energy_config.internal_oom_recovery`, `cfg:energy_config.analyze_kwargs`, `cfg:energy_config.mlip_compile`, `cfg:energy_config.mlip_edge_chunk_size`, `cfg:energy_config.mlip_activation_checkpointing`, `cfg:energy_config.temperature`.

Environment switches, all defaulting to the shipping path: `MXT_VECTORISED_ATOMICDATA`, `MXT_UMA_EXTERNAL_GRAPH`, `MXT_UMA_GRAPH_TIMER` (default off), `MXT_GPU_MACE_BATCH`, `MXT_BATCHED_MACE_NEIGHBOURS`, `MXT_HOISTED_MACE_ATOMICDATA`, `MXT_SKIP_GPU_PREFLIGHT`.

Code, `gfn_diffusion/energy_sampling`: `energies/molecular_crystal.py::MolecularCrystal.__init__`, `.analyze_crystal_batch`, `.batched_analyze_crystal_batch`, `.energy`, `.prebuilt_sample_to_reward`, `.attach_gas_phase_reference`, `.gas_reference_audit`, `._TOLERATES_NONFINITE`; `gflownet_losses.py::get_loss_reward`.

Code, `mxtaltools`: `crystal_analysis.py::MolCrystalAnalysis.analyze`, `.compute_lattice_uma`, `.compute_lattice_gas_phase_uma`, `.compute_lattice_mace`, `.compute_lattice_gas_phase_mace`; `uma_utils.py::init_uma_crystal_predictor`, `crystal_inference_settings`, `compute_crystal_uma_on_mxt_batch`, `batch_to_fairchem_batch`, `attach_external_graph`, `safe_predict_uma`, `_crashed_energy`; `AL_mace_utils.py::load_mace_model`, `compute_crystal_mace_on_mxt_batch`, `batch_to_mace_input_dict`, `batch_to_mace_atomicdata_hoisted`, `safe_predict_mace`; `pbc_neighbours.py::batched_pbc_neighbour_list`.

## Could be tooling

Two checks are mechanical. The first is an execution audit at construction: resolve each module flag and the predictor's `inference_settings` and print the builder that will run for `pbc=True` and for `pbc=False`. The drains report executed fractions after the fact (`energy/mace_flag_gpu_batch`, `energy/uma_flag_external_graph`); the same resolution before the first energy call would name a flag whose branch is unreachable. The second belongs at config-generation time: `mlip_compile` true with `mlip_edge_chunk_size` null, and a chunk size below the last run's `energy/uma_edges_max`, are states the code accepts and that fall back to eager silently.

## Sources

The code above, read at the stamped commit, and the energy block of the canonical config. Repo: `docs/mlip_validation.md`, `docs/design/dependency_validation_protocol.md`, `docs/design/benchmarks.md`, `docs/findings.md` (F-053 to F-057). Memory files on the UMA and MACE routes located the code and were not used as evidence.
