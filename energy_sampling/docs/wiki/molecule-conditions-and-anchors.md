# Molecule conditions and anchors

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

A *condition* on the crystal route is a molecule: the policy is handed a fixed description of it and asked for a packing. That description is read from files on disk, not from a live encoder. This page covers which keys name those files, what a file must contain, how the QM9-derived sets are built, and which key enters problem identity. The encoder that produced the embeddings is [molecule-encoder](molecule-encoder.md); what training does with a condition is [conditional-route](conditional-route.md); the runtime anchor store is [anchor-buffer](anchor-buffer.md).

## The four keys

Four config keys name the files a run reads; the table lists, for each, the trainer method that loads it and what it becomes.

| key | read by | what it names |
|---|---|---|
| `cfg:molecules_path` | `train.py::Modeller.init_mol_dataset` | the condition set trained over, loaded into `mol_dataset` |
| `cfg:prior_path` | `train.py::Modeller.init_prior_dataset` | the structures backward training draws from, `prior_dataset` |
| `cfg:test_molecules_path` | `train.py::Modeller.init_mol_dataset` | held-out conditions, `test_mol_dataset`; `null` leaves it `None` and every `eval_test` metric absent |
| `cfg:buffers.anchor_buffer.seed_source` | `train.py::Modeller.init_anchor_buffer_seed` | `generated`, `prior_dataset`, or a path loaded the same way |

The envelopes differ. `train.py::Modeller._load_condition_file` unwraps `{'prior': batch}` and is shared by the two condition keys; `init_prior_dataset` reads the `'equalized_prior'` entry; a seed-source path accepts either. The three dataset keys each become a `buffer.py::CrystalBuffer` over one resident collated batch; a seed-source batch is re-scored and handed to the anchor buffer instead. On mk_dev the first two keys name the same file, one molecule, unconditional.

A prior file may also carry `thermal_scaling_factor`: `init_prior_dataset` tests for the key's presence and lets it replace `lj_coeff` for the whole run, raising on a non-unit value beside any `cfg:energy_function` other than `elj`, since the coefficient is applied inside `compute_eLJ_energy` and reaches no other route. That function also wipes `smiles`, keeps `identifier`, calls `buffer.py::strip_lazy_sg_caches`, and runs `train.py::Modeller._verify_dead_latent_rows`.

## What a condition file holds

The full QM9 file is a list of `MolData`, one molecule per row, each carrying `.smiles`. Everything the trainer reads is crystal-shaped: a collated `MolCrystalData` batch of structures carrying cell parameters, `identifier`, `sg_ind`, `z_prime`, `aunit_handedness`, and on the embedding route `embedding`.

Identity resolves through `identifier`. `train.py::Modeller.init_identifiers` collects every `identifier` across the three datasets, sorts them, mints a dense integer `mol_id`, attaches it to all three resident batches, and calls `energies/molecular_crystal.py::MolecularCrystal.set_n_molecules`. Backward training draws from `prior_dataset`, a separately loaded file, and the registry spans both; that the two files carry matching `identifier` fields is assumed, not checked by fingerprint or geometry. `MolecularCrystal.condition_samples` forms `condition_id` as a mixed-radix combination of `mol_id` with the dense-local space group and Z' indices, the library sized `n_molecules * n_sg * n_zp`. A batch with no `mol_id` collapses onto molecule index 0 rather than raising; a side-loaded anchor seed whose identifiers are absent from the registry does raise.

At intake every path re-poses the molecule, `orient_molecule(mode='std')` being called at rollout and at buffer admission while `MolecularCrystal.analyze_crystal_batch` scores with `std_orientation=False`, so a stored molecule has to be a fixed point of that function already; the frame itself is [asymmetric-unit-reduction](asymmetric-unit-reduction.md).

## The QM9-derived sets

Two QM9 sources exist. `models/encoder_probe.py::load_qm9` reads the full dataset at `QM9_FULL`, which its docstring states as 133,728 molecules with every SMILES distinct; it deduplicates on `smiles` and raises rather than returning fewer than asked. The `qm9c100k_chunk*.pt` files under `QM9_DIR` are crystal anchors: the same docstring records about 5,850 unique molecules across 30 chunks, roughly 4% of QM9, at about 8.2 crystal rows per molecule, and 0.0% of molecules carrying a chiral tag in either source.

`build_qm9_conditions.py` turns a QM9 SG2 crystal dataset into a conditions/prior pair: `normalize_schema` fills the `z_prime` and `aunit_handedness` fields those files leave unset, `select_molecules` keeps whole molecules replica-richest first so a rung is exactly a set of condition ids, `drop_pathological` drops structures whose baked `lj_pot` exceeds `--lj-max`, `reconstruct_parameterization` derives centroid, orientation and handedness from the real coordinates (the CIF-derived files store `pos` at real coordinates and aunit parameters that do not describe it), and `pin_to_trainer_frame` makes the file a fixed point of `orient_molecule(mode='std')`. Molecules whose principal-axis basis is left-handed are mapped by a reflection: they are stored as their mirror image, their `aunit_orientation` cannot be repaired, and they carry `crystal_valid = 0`, counted in the file's `n_crystals_valid` / `n_crystals_total`. A stage with `bwd_sampling_mode: dataset` trains on stored cell parameters, and the script prints that restriction.

## The frozen conditioner and the ladder

Molecule conditioning ships as pre-embedded vectors. `build_qm9_conditions.py::embed` and `build_anchor_conditions.py::embed` run a frozen Mo3ENet autoencoder once, offline, and bake `embedding` of shape `[n, 3, bottleneck]` onto each row, equivariant rather than scalarized. Mo3ENet's vocabulary is `{1, 6, 7, 8, 9}`, and both scripts exit on a molecule set carrying any other atom type.

Wiring is three places. `train.py::Modeller.get_conditioning_dim` adds `cfg:embedding_conditioning_dim` to the conditioner's input width while `conditions_type` stays `'vector'`, so the embedding rides the same `mxtaltools/models/modules/components.py::scalarMLP` conditioner a toy `c` does (`models/gfn.py::GFN.init_conditioner`). `MolecularCrystal.condition_samples` reshapes the stored embedding to `[num_graphs, -1]`, raising when the batch has none or the flattened width disagrees with the configured dim. `utils.py::get_problem_definition` writes `emb_cond` and `emb_cond_dim` into problem identity only when the flag is on, so a run that never sets it keeps the identity it had before the key existed.

The ladder as it exists in configs is `configs/qm9_aug11/`, arms `qm9_v1`, `qm9_v2` and `qm9_v8`, each with `embedding_conditioning: true`, `embedding_conditioning_dim: 192`, `energy_config.temperature: 6.9`, and `test_molecules_path` null on the first two and a held-out conditions file on `qm9_v8`. `configs/qm9_aug11/make.py` records the arms as built `--valid-only`, the ladder topping out at 8 molecules, and the mirrored remainder in the held-out file.

## The CSD latent leg and the ELJ anchors

`extract_csd_latents.py` harvests the molecule-agnostic latent distribution for one space group and Z' from the featurized CSD, storing the twelve rows of its `LATENT_NAMES` plus light provenance, and not the crystals. Latents are dimensionless, `latent_transform` dividing cell lengths by molecular radius and using fractional centroids. A latents-only file is not a `prior_path`: that key needs a collated batch with identifiers, since `init_prior_dataset` re-analyzes energies and `init_identifiers` reads the identifier column.

The anchor leg builds structures instead. `prep_qm9_anchor_mols.py` samples molecules from the CSD-free QM9 file, pins them to the `orient_molecule(mode='std')` fixed point and drops those that move; a crystal search optimises them under the house ELJ recipe; `merge_anchor_chunks.py` checks chunk disjointness; `filter_anchors.py` drops `elj > 0` and packing above `--max-packing` (default 0.85), `--min-packing` being off by default; `build_anchor_conditions.py` attaches embeddings and writes the file pair. Embeddings are computed once on the canonical molecules and broadcast to anchors by identifier, so every anchor of a molecule shares a bit-identical condition vector, and the prior carries them too. The conditions file takes one carrier per molecule, the lowest-energy anchor; the prior takes the full set. No `thermal_scaling_factor` is written.

## Splitting on the structural equivalence class

`build_anchor_conditions.py` splits at the SMILES level, not the identifier level: `--holdout-n` distinct SMILES keys are drawn under `--holdout-seed`, every molecule sharing a key moves together, and an assertion checks no SMILES spans the split. The prior is rebuilt from training molecules only; the held-out file carries `"split": "holdout"` and no `equalized_prior` companion, held-out conditions being forward-sampled and never trained on.

The probe path splits one level coarser. `models/encoder_probe.py::parent_skeleton` strips stereochemistry and returns the canonical SMILES; `encoder_probe.main` groups rows by skeleton, shuffles the groups, emits each group's members contiguously, advances the held-out boundary to the next group edge, and asserts no group spans the split. Its docstring records 67.9% of held-out rows carrying a same-skeleton sibling in training at `n_train = 4000` under a row-wise split, rising 14.0% to 50.5% to 67.9% with `n`. `load_qm9_stereo` enumerates those stereoisomers, raw QM9 carrying none.

## Prior identity

`cfg:prior_path` is a field of `utils.py::get_problem_definition`, beside schema version, energy function, the energy config minus the non-identity list, space groups, Z primes and the conditioning flags. `cfg:molecules_path` and `cfg:test_molecules_path` are not. The field is the filename, so two runs differing only in the prior file's name have different `utils.py::problem_hash` values and a checkpoint from one is refused by the other on an explicitly named file ([checkpoints-and-resume](checkpoints-and-resume.md)), while the condition set may be repointed freely: `configs/force_sep18d/make.py` asserts the cluster `prior_path` and swaps `molecules_path` for a local file. `checkpointing.py::Checkpointer.WARM_START_EXEMPTIBLE` is the single exception, `prior_path` alone, usable only through `cfg:warm_start_ignore_problem_keys` on the weights-only path, and it prints what it exempted.

The auto-discovery that used to consume this is gone. `init_prior_dataset` loads a frozen prior model only when `cfg:prior_model_name` names one; the `reuse_prior` key and the fallback chain through `Checkpointer.find_shared_prior` are deleted from the trainer, with the deletion recorded in place, and `find_shared_prior` itself remains in `checkpointing.py` with no caller. A named prior model is checked with `Checkpointer.assert_problem_match` under `ignore_keys=('mol_cond', 'temp_cond', 'vec_cond')`, the prior being a sampling-only object rebuilt from its own stored `gfn_config` and sampled at its stored `train_T`.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:prior_path`, `cfg:molecules_path`, `cfg:test_molecules_path`, `cfg:prior_model_name`, `cfg:warm_start_ignore_problem_keys`, `cfg:molecule_conditioning`, `cfg:embedding_conditioning`, `cfg:embedding_conditioning_dim`, `cfg:vector_conditioning`, `cfg:vector_conditioning_dim`, `cfg:temperature_conditioning`, `cfg:sg_conditioning`, `cfg:zp_conditioning`, `cfg:space_groups`, `cfg:z_primes`, `cfg:energy_function`, `cfg:energy_config.temperature`, `cfg:buffers.anchor_buffer.seed_source`, `cfg:test_eval_num_samples`, `cfg:rollout_condition_draw`, `cfg:stage.condition_draw.conditions`.

Code: `train.py::Modeller.init_mol_dataset`, `._load_condition_file`, `.init_prior_dataset`, `.init_identifiers`, `.init_anchor_buffer_seed`, `.get_conditioning_dim`, `._verify_dead_latent_rows`; `utils.py::get_problem_definition`, `problem_hash`; `checkpointing.py::Checkpointer.WARM_START_EXEMPTIBLE`, `._warm_start_ignore_keys`, `.assert_problem_match`, `.find_shared_prior`; `energies/molecular_crystal.py::MolecularCrystal.condition_samples`, `.set_n_molecules`, `.analyze_crystal_batch`; `models/gfn.py::GFN.init_conditioner`; `mxtaltools/models/modules/components.py::scalarMLP`; `buffer.py::CrystalBuffer`, `strip_lazy_sg_caches`; `build_qm9_conditions.py::normalize_schema`, `select_molecules`, `drop_pathological`, `reconstruct_parameterization`, `pin_to_trainer_frame`, `embed`; `build_anchor_conditions.py::embed`; `extract_csd_latents.py`, `prep_qm9_anchor_mols.py`, `filter_anchors.py`, `merge_anchor_chunks.py`; `models/encoder_probe.py::load_qm9`, `load_qm9_stereo`, `parent_skeleton`.

## Could be tooling

A dataset preflight, given a config, would load the three files, print each one's stored metadata (`n_molecules`, `n_crystals_valid`/`n_crystals_total`, `embedding_dim`, `frame`, `holdout_seed`, presence of `thermal_scaling_factor`), and assert what the trainer assumes rather than verifies: `prior_path`'s identifiers a subset of `molecules_path`'s, the test file's intersecting neither, `embedding` present at exactly `cfg:embedding_conditioning_dim` when the flag is on, no two conditions coincident in embedding, and a second `orient_molecule(mode='std')` moving no atom beyond a tolerance. A second tool belongs at config generation: report which `problem_def` fields differ between two configs, and whether the difference is `prior_path` alone.

## Sources

The code above, read at the stamped commit, and the data-path keys of the canonical config with their comments. Memory files located the code and were not used as evidence: project_qm9_full_dataset_not_anchor_chunks, project_qm9_molecule_conditioner, project_csd_latent_prior_qm9_anchors, project_split_on_structural_equivalence_class, project_prior_path_in_problem_def_blocks_battery_reuse, project_qm9anchor_aug14_two_waves. Two of them describe `reuse_prior` and `find_shared_prior` as live trainer paths, which they no longer are.
