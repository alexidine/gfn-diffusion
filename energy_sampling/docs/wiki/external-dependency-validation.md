# External dependency validation

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

Some numbers this project depends on come from a third-party computation the repo does not call in stock form: an upstream package ships a model and a documented evaluation workflow, and the repo rebuilds the input path around that model for speed. `docs/design/dependency_validation_protocol.md` states the procedure for establishing that the rebuilt path returns what the upstream one would, for any such reimplementation. This page restates that note and sets it beside the tests that implement it. The MLIP routes it was written from are [mlip-energy-routes](mlip-energy-routes.md); the pair-sum backends, which have no upstream package behind them, are [crystal-force-fields](crystal-force-fields.md). The GPU refusal the tests share with the trainer is [compute-guards](compute-guards.md).

## The reference

The reference is the upstream package's own result on identical inputs: its documented workflow, built from the checkpoint path rather than from an already-loaded model object, at the package's defaults, with no overrides and no post-load surgery. A stock side handed our loaded object excludes the load, itself part of what was changed. Its input is built from primitive state, positions, cell and elements, not through our converter, so a converter defect cannot appear on both sides and cancel.

## The three levels

The note classifies a gate by what its two sides share, not by who wrote the code.

- **L0, internal parity.** Two of our own routes sharing their upstream derivation. Rules out drift between routes that fork below the shared part.
- **L1, stock-workflow ground truth.** Our production path against the dependency's own documented workflow. Rules out our construction, conventions and setup being wrong.
- **L2, published reference.** That workflow against numbers the dependency's authors published. Rules out our install or checkpoint differing from theirs.

A test of our code against an upstream helper is L1 for whatever it covers, even when it sits beside the parity tests.

## Internal parity under a shared defect

The argument is a derivation. Let both routes be a composition $f = g \circ h$, with $h$ the shared input preparation and $g$ the part that forks, so the routes are $g_1 \circ h$ and $g_2 \circ h$. A parity assertion compares $g_1(h(x))$ with $g_2(h(x))$. If $h$ is wrong, returning $h'(x) \neq h(x)$, both sides evaluate at the same wrong argument, the difference $g_1(h'(x)) - g_2(h'(x))$ is unchanged in form, and the assertion is silent. The count of such assertions does not change this: the blindness follows from the shared factor.

Internal parity rules out only that our two paths disagree. The fork an L1 gate opens has its own upstream in turn; the note places that shared layer in the write-up's scope section, closed by a check that does not use the shared assumption.

## What is held fixed, and what may differ

Inputs and device are common to both sides, and anything that would desynchronise them gets an assertion or a skip rather than an assumption: a guard that mutates the input, an element table the model does not cover, a batch-size ceiling. Where the shipping path sets a precision flag the upstream default does not, that becomes two gates: one leaves the production setting in and sizes its bar for it, a second matches precision so the remainder is attributable to code alone.

A deliberate difference is allowed, and the note names making the two sides identical as an anti-pattern. Where our path is unwrapped and the upstream path wraps, the note's move is to argue the difference is immaterial, a lattice-vector translation being a symmetry of a crystal, and let the test cross it, rather than to wrap by hand and delete what the test probes.

## Tolerance and the noise floor

The note states that a tolerance is not hard-coded. The control is measured in the same test on the same data, by running one side twice and taking that spread as the stack's own run-to-run variance; the cross-stack delta must sit within a small multiple of it. The note places the bar between that noise floor and the smallest defect worth catching, and has the write-up name both numbers.

The note requires two further elements: a negative control, perturbing the input by less than a real defect would and required to be rejected; and a guard that the two sides are not bitwise equal, since two independent GPU stacks never are and equality means the fixture compares something against itself.

A residual inside the bar is attributed by intervention, feeding the stock side the same perturbation and watching the gap move, rather than tolerated, with cause and mechanism reported separately.

## What invalidates a gate

A change on either side of the boundary. Upstream: a version bump, particularly to the entry points the fairness argument reasons about, and a change of checkpoint, since the result is per-checkpoint. Ours: removing or changing any step the residual was attributed to, and any change to the settings object fixing precision or task heads. Extending a gate outside its stated regime invalidates it too, since an input outside that regime measures the upstream cap rather than our code.

## How the gate is wired as tests

The files live in the `mxtaltools` repo. L1 is `tests/test_uma_vs_stock_fairchem.py` and `tests/test_mace_vs_stock.py`; the L2 mechanism is `test_matches_published_reference_energies` inside the first. `tests/test_pbc_neighbours.py` is external for the graph alone, comparing edge sets against `mace.data.get_neighborhood` with no model on CPU; `tests/test_unit_cell_layout.py` is the shared-layer check, CPU-only and model-free, comparing intra-block distance matrices rather than element labels.

Each element is its own test. The UMA file carries `test_crystal_energy_matches_stock_ase_calculator` at production precision, `test_crystal_energy_matches_stock_at_matched_precision`, `test_energies_are_not_accidentally_identical`, the negative control `test_a_deliberately_broken_graph_is_caught`, and three CPU preconditions, `test_the_density_guard_did_not_fire`, `test_ase_cell_convention_round_trips` and `test_the_fixture_is_actually_unwrapped`. The density guard the first of those asserts against is the `max_cp` loop in `uma_utils.py::compute_crystal_uma_on_mxt_batch`, which grows `cell_lengths` while any `packing_coeff` exceeds it. The MACE file carries one CPU precondition, `test_ase_cell_convention_round_trips` on a model-free fixture, one L1 gate `test_crystal_energy_matches_stock_mace_calculator`, the attribution test `test_the_residual_is_the_fractional_round_trip_not_our_code`, `test_energies_are_not_accidentally_identical`, and the negative control `test_a_deliberately_broken_cell_is_caught`.

Where the note and the test files differ, plainly. The note says a tolerance is never hard-coded and the control is measured in the same test. The UMA production-precision gate asserts against the module constant `MAX_REWARD_UNIT_DELTA_KJ`, fixed in the file, and the UMA negative control asserts against the same constant. The UMA matched-precision gate computes `bar = max(control * 4.0, 1e-6 * scale)` from a control measured in that test, and the MACE L1 gate computes `bar = max(control * 20.0, 1e-5 * max(scale, 1.0))` the same way. The MACE negative control computes `bar = max(1e-5 * max(scale, 1.0), 1e-4)` with no control term. The MACE file has no matched-precision gate. No reference file for L2 is committed in either repo.

Checkpoints reach the tests from outside the repo: `tests/conftest.py::pytest_addoption` declares `--uma-checkpoint`; the UMA file's `checkpoint_path` fixture reads that option first and `UMA_CHECKPOINT` second, the MACE file reads `MACE_CHECKPOINT`, and the L2 test reads `UMA_REFERENCE_JSON` as a module-level `skipif` marker, so a run without reference data never opens a CUDA context; the L2 body reads `UMA_CHECKPOINT` from the environment directly rather than through the fixture. Absent any of these the tests skip: they do not fail and do not pass. A GPU-touching test requests the session-scoped `gpu` fixture, which skips unless `tests/conftest.py::gpu_preflight` passes: a CUDA device must be present, then `MXT_SKIP_GPU_PREFLIGHT` set to a truthy value returns a pass before the remaining checks, otherwise `gpu_guard.py::describe` must not refuse where that module imports, and free VRAM must be above `MIN_FREE_MB`. The note records that coverage loss shows as a rising pass count beside a rising skip count.

## The worked instance

`docs/mlip_validation.md` is the method's first worked instance, status `ACTIVE`, scoped to `mxtaltools/mlip_interfaces/`: the UMA (`uma_utils.py`) and MACE (`AL_mace_utils.py`) energy routes, as production calls them. It follows the template's section order and is accompanied by graded entries in `docs/findings.md` and a router line in `docs/README.md`.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

None; a gate is configured from its invocation. Pytest option: `--uma-checkpoint`. Environment: `UMA_CHECKPOINT`, `MACE_CHECKPOINT`, `UMA_REFERENCE_JSON`, `MXT_SKIP_GPU_PREFLIGHT`.

Code, `mxtaltools`: `tests/conftest.py::pytest_addoption`, `::gpu_preflight`, `::gpu`, `::MIN_FREE_MB`, `::SKIP_ENV`; `tests/test_uma_vs_stock_fairchem.py::checkpoint_path`, `::_decompose`, `::MAX_REWARD_UNIT_DELTA_KJ`; `tests/test_mace_vs_stock.py::MODEL_ENV`, `::MAX_GRAPHS`, `::any_crystals`; `tests/test_pbc_neighbours.py::test_edge_set_matches_matscipy`; `tests/test_unit_cell_layout.py::test_an_atom_major_layout_would_be_caught`; `mxtaltools/mlip_interfaces/uma_utils.py::compute_crystal_uma_on_mxt_batch`; plus the tests named above. Code, `gfn_diffusion/energy_sampling`: `gpu_guard.py::describe`.

## Sources

`docs/design/dependency_validation_protocol.md` and `docs/mlip_validation.md` in `gfn_diffusion/energy_sampling`, read at the stamped commit, with `docs/README.md` and `docs/findings.md`, and the test files above in `mxtaltools`. Three memory files (project_uma_level1_ground_truth_gate, project_uma_nondeterminism_and_tf32, reference_mlip_gpu_gate_checkpoints) located the files and are not evidence.
