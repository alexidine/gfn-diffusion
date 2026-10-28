# Codebase map

*Drift: **C** (code-bound). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

The training code is one flat Python package, `gfn_diffusion/energy_sampling/`, with seven code subdirectories (`models/`, `energies/`, `analysis/`, `eval/`, `bench/`, `benchmarks/`, `data_processing/`; all but `data_processing/` carry an `__init__.py`), a test tree and a config tree. It is launched from its own directory, imports the sibling repository `mxtaltools/` for crystal physics, and is not installed. Counted now with `wc -l` in the working tree, its 46 top-level `.py` files hold 49,234 lines, `train.py` 11,452 of them. The instruction file that covers it is `gfn_diffusion/AGENTS.md`, one level up; no nested one exists here.

## The modules

Every module below sits at the package root unless its path says otherwise, and the table gives what each file holds, not how the pieces interact. Checkpoint contents are on [checkpoints-and-resume](checkpoints-and-resume.md), the validators on [config-validation](config-validation.md), the failure modes on [dev-traps](dev-traps.md).

| module | what it holds |
|---|---|
| `train.py` | `Modeller`: construction, the step loop, both eval hubs, buffer management, logging |
| `conformer_modeller.py`, `train_conformer.py` | `ConformerModeller(Modeller)`, the conformer track on the same machinery, and the earlier stripped loop without the protocol or the buffer controllers |
| `protocol.py` | `Stage`, `StageProtocol`, `fresh_stage_ctrl`, the `ACTIONS` and `STAGE_FLAGS` vocabularies |
| `buffer.py` | `CrystalBuffer`, `AnchorBuffer`, `ConditionLogZTracker`, the `ConformerGraphHooks` mixin and its two subclasses `ConformerBuffer` and `ConformerAnchorBuffer`, the compact conformer row store `ConformerCompactRows` with its per-condition graph table `ConformerStatics`, `BufferCurrencyError`, `BufferColumnError`, `CompactRowsError` |
| `gflownet_losses.py` | `get_gfn_forward_loss` and `get_gfn_backward_loss`, the TB, VarGrad, DB, SubTB and MLE terms, the log Z estimators (`batch_root_z`, `winsorized_z_root`, `emp_Z`) |
| `controller.py` | `LRController`: one multiplier over the managed policy optimizers, moved by the per-stage segment schedule (`lr_control.schedule`, a stage's `lr_schedule`) and the fire cut, plus the always-on hard tripwire |
| `hard_failure_bars.py`, `lr_hot_sensor.py`, `lr_larder.py`, `ray_calibration.py`, `grad_clip_guard.py` | the tripwire's fitted loss and gradient bars, the drawdown sensor, the per-branch ring of replay-scoreable batches, the ray sensor, and `GradClipGuard`'s per-branch quantile bar |
| `checkpointing.py` | `Checkpointer`, `MODELLER_STATE_DEFAULTS`, `CHECKPOINT_TAGS`, `BUFFER_SUFFIX` |
| `config_snapshot.py` | the resolved config, parsed protocol and per-stage coefficients, diffed |
| `config_state.py`, `config_invariants.py` | the `project_state_version` history with its migrations; cross-key relations as pure functions of a raw config dict, at severities `ERROR` and `BASELINE` |
| `utils.py` | argument parsing and derived-value resolution, problem identity, `atomic_save`, `set_seed`, and `featurize_dataset` and `embed_dataset`, which call MXtalTools' MLIP and encoder loaders |
| `paths.py` | `ROOT`, `ARTIFACTS`, `artifact()` |
| `gpu_guard.py` | `require_free_gpu`, the pre-flight occupancy and VRAM check |
| `progress_metrics.py`, `profiling.py`, `tierc_smoke.py` | the per-column marginal fit and progress exit gate; CUDA-synchronised region timing; the run-level acceptance tier |
| `build_*.py`, `prep_*`, `merge_*`, `filter_*`, `extract_*`, `calibrate_*`, `bench_propanol_reference.py`, `migrate_buffer_sidecar.py` | dataset, condition, anchor and benchmark construction, with the propanol benchmark's reference draw; the sidecar migration |
| `models/` | `gfn.py::GFN` (the SDE policy, variance schedule and flow head), `architectures.py` (the MLP, encoding, flow and Langevin blocks), the conformer and ragged specialisations, the encoder stack, and the latent-geometry tables `dead_latent_rows.py` and `aunit_periodicity.py` |
| `energies/` | `base_set.py` (the energy protocol), `molecular_crystal.py::MolecularCrystal`, thirteen conformer modules, three modules on the conformer internal-coordinate prior (`prior_smoke.py`, `prior_baselines.py`, `prior_diagnostics.py`), three prior-density modules (`prior_flow.py`, `prior_knn.py`, `density_calibration.py`), five toy targets |
| `analysis/` | the run-reading package: `keys.py` (every metric-name literal and route detection), `pull.py` (local `.wandb` and cloud histories), `features.py`, `checks.py`, `compare.py`, `figures.py`, `cli.py` |
| `eval/` | `evaluations.py`, `traj_reporting.py` and the cross-condition metric set `xcond.py`, which `train.py` imports for in-run evaluation and figures; the basin modules (`basin_map.py`, `basin_map_utils.py`, `basin_analysis.py`) and `offline_figs.py`, which nothing outside `eval/` imports; the offline programs `conformer_model_eval.py` (a trained conformer GFN scored per condition), `conformer_logz_check.py` (per-condition log Z by importance sampling) and `campaign_compack.py` (COMPACK identity calibration for a crystal-search campaign) |
| `bench/` | the CPU LR-controller sandbox, rebuilt 2026-08-13, driving the shipping controller and `RayCalibration` against synthetic surfaces; the previous generation is `bench/old/`, whose conftest `collect_ignore`s all five test files it holds |
| `benchmarks/` | `registry.py`, which loads and validates the benchmark specifications in `registry.yaml` and runs nothing, and its test module `test_registry.py` |
| `data_processing/` | prior generation, capped Metropolis walkers (`capped_mc.py`), collation, promotion, normaliser symmetrisation (`symmetrize_prior.py`) and MLIP rescoring |

### `train.py` by method family

`Modeller` carries 180 methods, counted now, in these families: construction (`init_gfn`, `init_energy_function`, `init_schedulers_optimizers`, `init_prior_dataset`); the loop (`train`, `train_step`, `fused_train_step`, `fwd_train_step`, `bwd_train_step`, `replay_train_step`, `step_loss`); draws (`draw_bwd_sample`, `draw_replay_sample`); Z handling (`bootstrap_log_z`, `z_level_fill`, `z_calibration_tick`); buffers (`manage_prior_buffer`, `manage_replay_buffer`, `screen_and_admit_anchors`); batch and device control (`select_batch_size`, `handle_train_epoch_error`); eval (`evaluation`, `fwd_eval_sampling`, `bwd_eval_sampling`); logging (`log_metrics`, `ten_step_reporting`).

## Entry points

`AGENTS.md` names four crystal commands, run from `energy_sampling/`.

- `python -u train.py --config configs/mk_dev.yaml` launches crystal training. `__main__` preflights the config through `utils.py::get_train_args`, then calls `gpu_guard.py::require_free_gpu` before `Modeller()` touches CUDA, raising `SystemExit` on `GPUBusy`.
- `python -m config_snapshot configs/mk_dev.yaml --check` validates the fused config with no GPU and no data drive. The parser errors with `--check cannot be combined with comparison or snapshot-output options`.
- `python -m pytest -q tests/crystal/test_mxtaltools_crystal_boundary.py` exercises the live CPU/synthetic GFN-to-MXtalTools ELJ boundary.
- `python -m pytest -m fast -q` is the broader CPU development lane.

The conformer launch is not among them. `conformer_modeller.py`'s own `__main__` gives `python -u conformer_modeller.py --config configs/conformer_mk.yaml`; it mirrors train.py's entrypoint and sets `torch.set_default_dtype(torch.float32)` before the config is read. `python -m config_invariants`, `python -m analysis`, `python -m bench.board` and `python -m tierc_smoke` are among the other module entry points.

## The MXtalTools boundary

GFN imports `mxtaltools`, and no module of the `mxtaltools` package imports GFN. One tracked script in the sibling repository's config tree, `configs/crystal_searches/acr_proposals_sep27/make_seeds.py`, imports `energy_sampling.eval.nikos_comparison.summarize_search` inside a function. `MolCrystalData.analyze(...)` is the shared analysis and energy dispatch surface, with ELJ, MACE and UMA as selectable backends.

For the ELJ route the consumed chain, as `AGENTS.md` names it, runs `energies/molecular_crystal.py::MolecularCrystal.analyze_crystal_batch`, `MolCrystalData.latent_to_cell_params` (`crystal_ops.py`), `MolCrystalData.analyze(['reduction_en', 'elj'])` (`crystal_analysis.py`), `mol2cluster` (`crystal_building.py`), `construct_radial_graph` (`crystal_analysis.py`), then eLJ analysis, all under `mxtaltools/dataset_utils/data_class_methods/`. `MolecularCrystal.__init__` builds that `computes` list as `['reduction_en']` plus the configured `energy_function`, except for `latent_knn`, which is scored from the latent vector and appended to nothing. MXtalTools' on-device PBC neighbour list, `mxtaltools/mlip_interfaces/pbc_neighbours.py::batched_pbc_neighbour_list`, is not on this route. Its callers are the MACE input builders in `mxtaltools/mlip_interfaces/AL_mace_utils.py` and `mxtaltools/mlip_interfaces/uma_utils.py::attach_external_graph`, which the UMA interface runs when the predictor's `external_graph_gen` setting is true ([mlip-energy-routes](mlip-energy-routes.md)). `tests/crystal/test_mxtaltools_crystal_boundary.py` exercises the live CPU/synthetic boundary.

## Where generated artefacts go

Two conventions are in the tree. `paths.py` anchors `ARTIFACTS` to its own file location, and `artifact(name)` resolves a *bare* output name into `energy_sampling/artifacts/`, passing through any absolute path or any path with a directory component. Four producers instead hold a `results/` directory beside themselves: `bench/`, `energies/`, `models/`, `data_processing/`. Checkpoints go where `cfg:checkpoints_dir` names, and `wandb/` beside the run.

## Configs

`configs/` holds 3,062 YAML files, counted now in the working tree. `configs/mk_dev.yaml` is the canonical crystal config and the spawn point for runs; `configs/conformer_dev.yaml` is the conformer equivalent and, as `AGENTS.md` records, is not a stable schema contract. `configs/problems.yaml` is a registry of problem-intrinsic settings. `configs/generate.py` derives an arm from the canonical config and records that file's hash, and per-battery `make.py` generators sit beside it ([battery-generation](battery-generation.md)). The rest are dated battery directories and single-run YAMLs.

## Tests

`pytest.ini` sets `testpaths = .`, so the default run recurses into `tests/`, `bench/`, `benchmarks/` and `analysis/tests/`, and `pythonpath = . ..` keeps both the bare and the `energy_sampling.x` import spellings resolvable. `norecursedirs` excludes `.claude`, `.git`, `__pycache__`, `wandb`, `checkpoints`, `configs`, `SCRATCH` and `eval/paper1_results`. `bench/old` is not excluded there; its own conftest's `collect_ignore` makes the inclusion inert.

Two markers exist, and `conftest.py::pytest_collection_modifyitems` assigns one to every test: a module whose *source* matches `import torch` or `from torch` at any indentation is `slow`, the rest `fast`. An explicit `pytestmark` wins, and `pytest -m fast` is the opt-in lane. The same conftest wraps `pytest_pyfunc_call`, so a test that appended a failing entry to a module-level `_RESULTS` or `_R` log, or returned a false verdict, fails rather than passing.

`tests/` is split into `config/`, `conformer/`, `crystal/`, `infra/`, `losses/`, `lr/`, `models/`, `protocol/` and `wiki/`, holds one test module at its own root, and carries no `__init__.py`. Counting now in the working tree, 200 `test_*.py` files sit under `tests/`, `bench/`, `benchmarks/` and `analysis/tests/`: 171, 18, 1 and 10.

## Docs and this knowledge base

`docs/README.md` is headed a legacy router. Its head names this knowledge base as the active documentation and states that `docs/EPISTEMIC_PROTOCOL.md` and `docs/PROTOCOL.md` are retired as operating procedure, that `docs/findings.md` is a frozen evidence ledger and `docs/module_*.md` are snapshots, and that `docs/design/` holds derivation notes, plans and handoffs; the original router, with its status table per document family, follows that head. `bench/README.md` names `on_calibration`, `on_plateau` and `on_hypergradient` as the controller hooks the bench drives; `controller.py` defines `on_calibration` and neither of the other two. This knowledge base is `docs/wiki/`, built by `mkdocs.yml` at the package root, which sets `docs_dir: docs/wiki` and `site_dir: ../../.mkdocs_site` and enables the Mermaid and MathJax extensions ([writing-protocol](writing-protocol.md), [index](index.md)).

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:checkpoints_dir`, `cfg:protocol`, `cfg:energy_function`, `cfg:project_state_version`.

Code: `train.py::Modeller`; `conformer_modeller.py::ConformerModeller`; `protocol.py::StageProtocol`; `buffer.py::CrystalBuffer`, `AnchorBuffer`, `ConformerBuffer`, `ConditionLogZTracker`, `BufferCurrencyError`; `gflownet_losses.py::get_gfn_forward_loss`, `get_gfn_backward_loss`; `controller.py::LRController`; `checkpointing.py::Checkpointer`; `utils.py::get_train_args`, `get_problem_definition`; `paths.py::ARTIFACTS`, `artifact`; `gpu_guard.py::require_free_gpu`, `GPUBusy`; `models/gfn.py::GFN`; `energies/molecular_crystal.py::MolecularCrystal.analyze_crystal_batch`; `mxtaltools/mlip_interfaces/pbc_neighbours.py::batched_pbc_neighbour_list`; `mxtaltools/mlip_interfaces/uma_utils.py::attach_external_graph`; `conftest.py::pytest_collection_modifyitems`, `pytest_pyfunc_call`.

## Could be tooling

The table above drifts the moment a module moves, and most of it is derivable: a script walking the package could emit, per module, its line count, its top-level classes and functions, the first docstring line, and which of `torch` and `mxtaltools` it imports, then diff that against the installed page, so a renamed module fails a check rather than going undocumented. An importer census over consumed `mxtaltools` symbols, as (module, symbol) pairs with the test pinning each, would do the same for the boundary.

## Sources

The working tree, uncommitted and untracked files included, read with `ls` and `wc -l`; the module docstrings above; `gfn_diffusion/AGENTS.md`; `conftest.py`, `pytest.ini`, `docs/README.md`, `bench/README.md` and `mkdocs.yml`. Six memory files (project_train_py_refactor_plan, project_mxtaltools_refactor_plan, project_generated_artifacts_live_beside_producer, project_bench_controller_sandbox, reference_full_test_suite_timing, reference_analysis_package) located the code and were not used as evidence.
