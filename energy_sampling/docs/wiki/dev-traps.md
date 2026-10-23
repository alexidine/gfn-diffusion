# Developer traps

*Drift: **C** (code-bound). Verified against commit `78de930`, 2026-10-04. Sources at the end.*

A *silent wrong result* is an outcome that differs from the intended one while nothing raises, nothing warns, and the run or the test reports success. This page collects the mechanisms in this codebase and its dependencies that produce one, each with the code that exhibits it and the shape of a check that catches it. Config keys accepted but never read belong to [config-validation](config-validation.md); GPU admission and OOM to [compute-guards](compute-guards.md); launch lines and submission to [cluster-operations](cluster-operations.md).

## PyG device moves mutate the receiver

`torch_geometric/data/data.py::BaseData.to` is `self.apply(lambda x: x.to(device=...))`, and `storage.py::BaseStorage.apply` assigns back into its own mapping and returns `self`. So `to`, `cpu`, `cuda`, `pin_memory`, `contiguous` and `detach` on a `Data` or `Batch` are in-place edits of the receiver, returning that same object; of the members built on `apply`, `clone` alone copies first, explicitly `copy.copy(self).apply(lambda x: x.clone())`.

So `b = a.to('cuda')` leaves `a` on CUDA and `b is a`, and where the receiver is a caller's object, the caller's object moves: `buffer.py::CrystalBuffer.__init__` and `.add` both call `.to` on `self._as_batch(data)`, which is the caller's own object when a batch and not a list is passed, and `MXtalBase.append_batch` opens with `other = other.to(device)`. A check: assert `moved is not original`, or read the source object's `.device` after a move meant to copy.

## Batch fields are classified by their first dimension

`MXtalBase` rebuilds PyG's slice and increment bookkeeping from tensor shapes, not from a schema. `MXtalBase.rebuild_simple_slice_inc_` sorts each tensor in the store four ways: `ndim == 0` or `size(0) == 1` is shared metadata and is skipped, `size(0) == n` is node-level, `size(0) == g` is graph-level, anything else is shared metadata and is skipped. `MXtalBase.subsample_new_batch` runs the same test against `old_num_nodes` and `old_num_graphs` to decide whether a field is gathered by `node_src`, gathered by `idx`, or copied through, and `MXtalBase.append_batch` decides between concatenating and keeping self's value in a different order: `ndim == 0` first, then `size(0)` against both batches' node counts, then against both batches' graph counts, and only then the remainder, so a length-1 tensor of a single-node or single-graph batch is concatenated when the other batch's tensor matches that batch's own count.

Since 2026-10-20 (MXtalTools, uncommitted at the time of writing) the length decides only where it can: where a tensor's length is ambiguous (length 1, or a batch with as many atoms as rows), `subsample_new_batch`, `rebuild_simple_slice_inc_` (its `levels=` argument) and `append_batch` take the level the batch recorded in `_slice_dict` (`MXtalBase._recorded_level`), and fall back to the length rule only where no usable entry exists; ordinary batches behave as before (`tests/test_batch_levels.py` in MXtalTools). A caller that rebuilds slices without passing the levels it knows, such as the compact conformer store's `subsample_new_batch` in `buffer.py`, still gets the length rule on a one-row result.

Before 2026-10-23 two kinds of field were therefore not carried: a per-degree-of-freedom array whose length was neither the node nor the graph count was copied verbatim onto the subsampled batch, describing the rows it was built for rather than those that survived; and every increment written was zero, so an attribute holding absolute atom indices was gathered without being re-offset (`append_batch` did not offset `edge_index`). Since 2026-10-23 (MXtalTools, uncommitted at the time of writing) every stored value except `ptr` and `batch` gets one of five levels, atom, unit, edge, row or batch, from the first rule that applies: the name registry `data_classes.py::KEY_LEVELS` (extended by `data_classes.py::declare_level`), then collation's naming contract (a tensor key holding 'index' or 'face' is per edge and its values are re-offset as atom indices, as collation offsets them), then the slice entry, then the length (`MXtalBase._level_plan`). Edge and unit values are sliced by the counts their entry records, and a value the helpers cannot place raises `data_classes.py::BatchLevelError`. The level contract is MXtalTools' `tests/characterization/README.md` and `tests/test_batch_unit_levels.py`. What is still carried whole: a tensor with no usable entry and no registration whose length is neither the atom nor the row count (a table set on the batch, a per-unit value attached without `add_unit_attr`), and a per-DoF array whose total coincides with the atom or row count is still read per atom or per row. A field must therefore name atom indices with 'index' or 'face' and nothing else with those words; the conformer condition's tree references are graph-relative deltas under names without them ([conformer-training](conformer-training.md)), and are carried unchanged. A check: round-trip a batch with distinct node, graph and per-DoF lengths, comparing each field against the permutation applied by hand and index-valued fields against re-offset expectations, not comparing shapes.

## One module, two module objects

`pytest.ini` sets `pythonpath = . ..`, and the launch lines put both `mxtaltools` and `gfn_diffusion` on `PYTHONPATH`, so both spellings of every module here are importable. `import checkpointing` and `from energy_sampling.checkpointing import Checkpointer` build two distinct module objects with two distinct class objects; `bench/calibrate_noise.py` records the identity check evaluating false. Both spellings are live: `checkpointing.py` imports `from energy_sampling.buffer import ...`, `tests/infra/test_gpu_guard.py` does `import gpu_guard`.

A monkeypatch, spy or counter installed on one object is invisible to code that reached the other; the unpatched path runs and the patch records nothing. `bench/calibrate_noise.py` patches `type(m.checkpointer)`, the live instance's own class, which is one object under either name. A check: assert the patch fired, or resolve the target through an object the code under test holds.

## Module scope runs at collection

pytest imports every collected module before running any test, so a module-scope statement in one file has executed by the time an unrelated file's test runs. `torch.set_default_dtype` and writes into `os.environ` are process-global, so either at module scope changes the whole session and produces a file that passes alone and fails beside a module it never references. `tests/crystal/test_dead_latent_rows.py` records that exposure for its bitwise `torch.equal` comparisons and answers it with an autouse fixture, `_pinned_default_dtype`, that saves, sets and restores. A check: a session-scoped autouse fixture asserting the default dtype and named environment keys are unchanged at each test boundary.

## Reachability is not a text match

A search for `import <mod>` does not find `from pkg.mod import X`. A search for a config key by its leaf also matches live keys that share the leaf, and a search by its dotted path does not find a `getattr(cfg, 'leaf', default)` whose receiver is a local bound to the block. `tests/config/test_no_gating_on_retired_keys.py` records both and reconstructs dotted paths from the AST instead, resolving a `getattr` receiver back to its assignment in the same function. `tests/crystal/test_anchor_currency.py::_modeller_calls` aims the same instrument at call sites: it parses `Modeller`'s source and returns every call to a named method, and the tests built on it assert the number of sites and one argument at each (a keyword for one method, the first positional argument for the other), so a new site is a failure rather than silence.

## Staging is per file, not per hunk

`git add <file>` stages that file's whole difference, not the hunks under review, so a commit made after editing two things in one file carries both. What ships is the index, which `git diff --cached` prints and the working tree does not.

## A tolerance or a pool can be blind

A scale-relative tolerance on a large quantity can exceed the defect. `tests/crystal/test_batch_invariance.py::_tol` records the measurement: an absolute 1e-2 kJ/mol sits about 100x above the two-call reproducibility floor of ~1e-4 and about 5x below the smallest error the defect produced, 0.048, while a scale-relative 1e-3 on an O(1000) quantity is ~1.0 and is blind to all but the extreme tail. Where quantities are not expected to agree bitwise, `tests/crystal/test_dead_latent_rows_deep.py` judges a gap against the run's own convergence error and asserts a control's separation, so agreement cannot pass vacuously.

A randomly drawn pool can omit the case under test. `tests/crystal/test_batch_invariance.py::_adversarial_pool` records four random 100-crystal draws on one prior, one containing no affected crystal and another a 126 kJ/mol error, and selects deterministically from the structure of the defect instead. The paired check is a sensitivity assertion that the input exhibits the effect when the mechanism is disabled.

## The global RNG is reset by model construction

`mxtaltools/models/modules/components.py::scalarMLP.__init__` and `vectorMLP.__init__` call `torch.manual_seed(seed)`, so constructing a GFN puts torch's global RNG in a fixed state after `utils.py::set_seed` has run. The paths affected are on [checkpoints-and-resume](checkpoints-and-resume.md).

## A docstring is not a wiring assertion

Prose describing a mechanism is not evidence that the mechanism is reached, and nothing fails when the two diverge. `pytest.ini` carries the recorded case: a note stating that `bench/old` parked live regression coverage for the shipping LR controller, against a `bench/old/conftest.py` whose `collect_ignore` names all five `test_*.py` files there, so the inclusion collects nothing and zero collected tests is a green run. The mechanical check is the AST form above, an assertion that a named mechanism is called from an expected number of sites.

## A check that did not run reports as a pass

Three patterns turn an unrun or failed check into a green result. A suite reporting through a local `check(name, ok, detail)` that appends to a module list and prints: the list is read by the file's own `main()`, so under pytest a body full of FAIL entries passes. A test body ending `return ok`: pytest discards the value and warns, and the test passes; `tests/losses/test_vg_detach_center.py` is written that way. A collection error: the module never runs and the runner reports success, which is how `conftest.py` records six invariant tests going months without running. `conftest.py::pytest_pyfunc_call` closes the first two by wrapping the call phase, reading `_RESULTS` or `_R` for entries appended during the test and raising on any false one, and capturing the body's return value and raising when it is falsy and not `None`. The third is not closed there.

## The interpreter and the path

The project interpreter is `C:\Users\mikem\venvs\csd_mxt_gfn\Scripts\python.exe`, named by `tests/infra/test_gpu_guard.py::VENV` and by the local run scripts. There are no editable installs: `configs/cond_workup/generate_configs.py::PYTHONPATH` names `C:\Users\mikem\Projects\mxt_gfn\mxtaltools` and `C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion`, the second being the parent of `energy_sampling/` that the dotted imports require; under pytest `pythonpath = . ..` supplies both spellings. `gpu_guard.py::require_free_gpu` returns immediately, with a printed reason from `_skip_reason`, when `GFN_GPU_GUARD` is falsy, when `CUDA_VISIBLE_DEVICES` hides all GPUs, or when any of `SCHEDULER_ENV` is set; `GFN_ALLOW_GPU_SHARING` downgrades a block to a warning. Both go through `_env_true` and `_env_false`, which take `1/true/on/yes/y` and `0/false/off/no/n` case-insensitively. `TRAIN_ENTRYPOINTS` is `('train.py', 'train_conformer.py', 'conformer_modeller.py')`, so a test session holding the card is not a tenant.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:checkpoint_read_only`, `cfg:seed`. Environment: `GFN_GPU_GUARD`, `GFN_ALLOW_GPU_SHARING`, `GFN_COTENANTS`, `CUDA_VISIBLE_DEVICES`, `PYTHONPATH`, `WANDB_MODE`.

Code, this repo: `conftest.py::pytest_pyfunc_call`, `._check_log`, `._module_imports_torch`; `pytest.ini`; `gpu_guard.py::require_free_gpu`, `._skip_reason`, `._env_true`, `._env_false`, `DISABLE_ENV`, `OVERRIDE_ENV`, `SCHEDULER_ENV`, `TRAIN_ENTRYPOINTS`; `buffer.py::CrystalBuffer.__init__`, `.add`, `._as_batch`; `utils.py::set_seed`; `configs/cond_workup/generate_configs.py::PYTHONPATH`; `tests/crystal/test_batch_invariance.py::_tol`, `._adversarial_pool`; `tests/crystal/test_dead_latent_rows.py::_pinned_default_dtype`; `tests/crystal/test_anchor_currency.py::_modeller_calls`; `tests/config/test_no_gating_on_retired_keys.py::_retired_table`; `tests/infra/test_gpu_guard.py::VENV`; `tests/losses/test_vg_detach_center.py`; `bench/calibrate_noise.py::run`; `bench/old/conftest.py::collect_ignore`.

Code, mxtaltools: `mxtaltools/dataset_utils/data_classes.py::MXtalBase.rebuild_simple_slice_inc_`, `.subsample_new_batch`, `.append_batch`; `mxtaltools/models/modules/components.py::scalarMLP.__init__`, `vectorMLP.__init__`.

Code, torch_geometric: `torch_geometric/data/data.py::BaseData.to`, `.apply`, `.clone`; `torch_geometric/data/storage.py::BaseStorage.apply`.

## Could be tooling

Most items here have a mechanical form. An AST pass over `tests/` can list every module-scope call to `torch.set_default_dtype`, assignment into `os.environ` and `torch.manual_seed`, each a session-wide write executed at collection, and every test whose body ends in `return`. A session-scoped autouse fixture can assert the default dtype and named environment keys are unchanged at each test boundary. An import audit can walk `sys.modules` after collection and report any module present under two names. A field-kind audit can push a batch whose node, graph and per-DoF lengths are pairwise distinct through `subsample_new_batch` and `append_batch` and report every field that came through unchanged.

## Sources

The code above, read at the stamped commit, in this repo, in the sibling `mxtaltools` repo, and in `torch_geometric` as installed in the project venv. Twelve memory files located the code and were not used as evidence (project_pyg_inplace_device_moves, project_pyg_batch_ops_size0_classification, reference_dual_import_module_identity, feedback_module_scope_globals_leak_across_collection, feedback_dotted_import_grep_misses_reachability, feedback_git_add_ships_the_whole_file_diff, feedback_test_tolerance_and_pool_can_be_blind, feedback_swallowed_diagnostics_fail_as_reassurance, project_buffer_estate_review_aug31, reference_local_run_recipe, reference_conformer_run_recipe, feedback_crystal_code_fails_silently).
