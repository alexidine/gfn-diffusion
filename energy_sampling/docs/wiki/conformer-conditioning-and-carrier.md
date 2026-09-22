# Conformer conditioning and the carrier

*Drift: **C** (code-bound). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

A conformer run samples internal coordinates for one molecule in the unconditional case and for a set of molecules in the conditional one. Two mechanisms carry the set. The *carrier* is a fixed-width state layout holding molecules of different coordinate counts in one dense tensor, so the trajectory, the losses and the buffers stay rectangular. The *condition channel* tells the policy, the flow head and the energy which molecule each row is. The stack is mid-refactor.

Left out and named: the chart and the internal-coordinate measure are [conformer-chart-and-internal-coordinates](conformer-chart-and-internal-coordinates.md); the graph encoder is [molecule-encoder](molecule-encoder.md); the losses and stages are [conformer-training](conformer-training.md); condition-grouped objectives and per-condition log Z are [conditional-route](conditional-route.md).

## The carrier layout

A molecule's state is `cat[r, theta, phi]` over the columns surviving at the run's level, so its width `k` and which columns wrap both vary. `energies/conformer_carrier.py::CarrierLayout` reads each member energy's per-column `_free_block` code (0 r, 1 theta, 2 phi) and sets the three block widths to the per-block maxima over members. A member's column `j` goes to `offsets[block(j)]` plus its rank within that block; `K` is the block-width sum, and the member's other columns are pads.

Two properties follow from the block ordering. The periodic mask is column-constant, since every carrier column in the phi block wraps for every member, so the single `angular_mask` the GFN is constructed with remains correct for every member. And when all members share their block counts, `.is_identity` is true, column `j` maps to `j`, and the layout changes nothing.

`.to_carrier` places a `[n, k]` member state into `[n, K]` with pads at exactly zero and `.from_carrier` is an `index_select` back, so it is differentiable; `.valid`, `.pad_cols` and `.col_map` express the same map as masks and index tables. A member carrying a transverse column (`_free_block == 3`) raises at layout construction: there is no transverse block.

## Handling of the pad columns

- `models/conformer_gfn.py::ConformerGFN._pin_dead` writes pads to zero through `torch.where`, so a NaN in a pad becomes zero rather than propagating. `._bind_state_mask` raises when the GFN is flagged `_carrier` and the batch carries no `state_mask`, and the prior-jitter path in `conformer_modeller.py` masks its noise the same way.
- `ConformerGFN.gauss_logprob` and `._pb_logprob` apply the per-row `state_mask` to the per-dimension terms before the sum over dimensions; the exact backward mixture is recomputed in `._pb_mixture_ang_terms`, a replica of the shared-file method without its final sum. `gauss_logprob` raises when dead latent rows and a carrier state are both present.
- `energies/multi_conformer.py::MultiConformerTorsions._member_rows` refuses a row whose pad columns are not exactly zero, and refuses a batch whose `state_mask` disagrees with the layout, before slicing.
- On a carrier, `MultiConformerTorsions` is a dispatcher rather than a chart, every inherited chart method named in `multi_conformer.py::_CHART_METHODS`, among them `potential_energy`, `build_positions` and `prior_log_prob`, raises.

## The policy pools the ragged columns; the energy dispatches per row

`MultiConformerTorsions` subclasses `ConformerTorsions` and holds one member chart per identifier, the first being the reference member. `.energy` groups the rows by identifier (`._groups`), slices each group through `._member_rows`, calls that member's own `energy`, and reassembles with `._regroup`, a gather rather than an in-place scatter so gradients survive the `keep_grads=True` path. `.prebuilt_sample_to_reward` groups baked energies the same way and adds each row's own `log_jacobian_const` and `log_chart_jacobian`, recomputing the per-member `_log_jac` per group where the constant is `None`. A carrier state with `mol_batch` absent raises in `energy`; with one chart the call falls through to the parent.

The rollout stays dense `[B, K]`. `models/ragged_set_policy.py::RaggedConditionalSetPolicy` is the only object seeing ragged data: it gathers the valid columns flat with `flat_idx`, pools them with `segment_softmax` and `segment_sum` over `dof_batch`, and scatters back to `[B, 2K]` contiguous blocks with pad outputs zero, so `GFN.split_params` is unchanged. Per-column identity is the batch's `dof_static` concatenated with the learned `set_policy.py::DoFCorrelator` over the embeddings of the atoms a column spans, and the pooled molecular embedding joins `rho`'s context. Both index vectors are bound once per trajectory in `ConformerGFN.bind_molecular_conditioning`, called from the three `get_traj_*` entry points, which also shifts `dof_atoms` by the batch's `ptr` offsets, those indices being in local tree numbering.

## The condition channel

`energies/conformer_torsions.py::ConformerTorsions.condition_samples` returns `(mol_batch, log_T_tensor, condition, condition_id)` and attaches `conditions` and `condition_id` to the batch. The condition vector is log-temperature when `cfg:temperature_conditioning` is on, followed by the per-graph `embedding` when `cfg:embedding_conditioning` is on; with neither, one zero column. A batch missing `embedding` under that flag raises, and the width is checked against `cfg:embedding_conditioning_dim`. Conditions are drawn per group of `repeats` and broadcast within it. `condition_id` is `mol_id * (n_sg * n_zp)`, which with `n_sg = n_zp = 1` equals `mol_id`.

The identifier namespace has three levels. `identifier` is a Python list on the batch and the key `train.py::Modeller.init_identifiers` mints the registry from; `mol_id` is that registry's integer, a graph attribute and the only form surviving a buffer, since buffers keep tensors; `condition_id` sizes the per-condition log Z table through `ConformerTorsions.set_n_molecules`, which writes `condition_library_size`. `ConformerModeller.init_identifiers` hands the registry to `MultiConformerTorsions.bind_identifier_registry`, which lets `._row_identifiers` resolve a buffered row from `mol_id` alone; without it that row raises rather than falling back to the reference member.

Embeddings are baked offline. `models/encoder_cache.py::embed` runs the frozen encoder on the 2D graph, returning per-atom `h` and pooled `g`; the cache is stamped with the checkpoint's sha256 and refuses to load against another. `build_conformer_conditions.py` writes `embedding`, `atom_embedding`, `dof_atoms` and `dof_mask` onto each condition graph, the last two padded to a file-wide row count `R`. With `--carrier` it re-expresses every graph through `carrier_pad_condition`: the reconstruction columns `ctree_{r,th,ph}_col` are remapped, `n_torsions` becomes `K`, `state_mask` and `dof_static` are written, and `dof_atoms` and `dof_mask` are written in carrier form there instead. `ctree_state_col` is not remapped, since it indexes rotatable axes rather than state columns. `check_carrier_convention` then asserts the padded graph rebuilds the member chart's own geometry.

## `scalar_flow`

`conformer_modeller.py::ConformerModeller._build_gfn_config` sets `scalar_flow` true, from the conditions file rather than a config key, when the run is conditional and the set read off `cfg:molecules_path` has exactly one distinct identifier. `models/gfn.py::GFN.init_flow_model` then builds a `LearnableScalar` in place of the conditional `scalarMLP` over the condition embedding. `train.py::Modeller.z_level_fill` writes `flow_model.scalar.data`, which a `scalarMLP` does not have; `bootstrap_z_by_rollout` reads that scalar either side of a fill, and the stage-entry bootstrap in `protocol.py` writes it on both the live and the EMA model.

## Determinism of the encoder input, and the atom-order contract

The encoder reads atom types, bonds and parity from the 2D graph and nothing else, so its output is fixed for the rollout and is cached rather than recomputed. The tree is built with `spec_from_graph(..., use_geometry=False)`. Linearity flags are separate: `ConformerTorsions.linearity_source` is `'mmff_typed'` when the force field is `mmff`, and `angle_is_linear` and `torsion_frame_is_linear` then come from MMFF's typed equilibrium angle at 179.99 degrees, a function of the graph. With any other force field the source is `'measured'` and the flags are read off the reference conformer, and the state width then depends on an embedded geometry.

The two paths order atoms differently: `models/graph_encodings.py::graph_from_smiles` takes RDKit's own atom order after `Chem.AddHs`, which appends the hydrogens after the heavy atoms, while the conformer path uses the tree's placement order. `embed` permutes the per-atom embeddings by `spec.perm` once, at build time, and when `z_tree` is passed it asserts `z_enc[perm] == spec.z` and raises on mismatch; `build_conformer_conditions.py` passes both.

## Built but not reached

`models/ragged_gfn.py::RaggedConformerGFN` is imported only by `tests/models/test_ragged_gfn.py`. `RaggedSetPolicy.wants_ragged_state` is declared and read nowhere, while `wants_raw_state`, `wants_molecular_conditioning` and `wants_carrier` are read. `policy_kind: set` refuses to start from a checkpoint, since its keys live on the modeller rather than in `gfn_config`; it refuses `dplr_rank > 0`; and on a carrier it refuses to build without `embedding_conditioning`. On a carrier, `ConformerModeller.log_physical_properties` returns after a one-time notice, so the per-molecule energy, geometry, DoF-class, ring and coverage metrics are absent.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:molecules_path`, `cfg:embedding_conditioning`, `cfg:embedding_conditioning_dim`, `cfg:temperature_conditioning`, `cfg:model.policy_kind`, `cfg:model.set_policy_hidden`, `cfg:model.set_policy_layers`, `cfg:model.set_policy_corr_dim`, `cfg:model.dplr_rank`, `cfg:energy_config.level`, `cfg:energy_config.force_field`.

Code: `energies/conformer_carrier.py::CarrierLayout`, `::carrier_pad_condition`, `::check_carrier_convention`; `energies/multi_conformer.py::MultiConformerTorsions`; `energies/conformer_torsions.py::ConformerTorsions.condition_samples`, `.linearity_source`; `models/conformer_gfn.py::ConformerGFN`; `models/ragged_set_policy.py::RaggedConditionalSetPolicy`; `models/set_policy.py::DoFCorrelator`; `models/encoder_cache.py::embed`; `models/gfn.py::GFN.init_flow_model`; `conformer_modeller.py::ConformerModeller._build_gfn_config`, `._install_set_policy`, `._draw_carrier_prior`; `train.py::Modeller.init_identifiers`; `build_conformer_conditions.py`.

## Could be tooling

Most of what precedes a mixed-k run is a printable manifest. Given a conditions file: the layout `K` and its block widths, each member's `k` and pad count; which of `state_mask`, `dof_static`, `dof_atoms`, `dof_mask`, `embedding` and `atom_embedding` are present and at what widths; the encoder fingerprint against the checkpoint a config names. Beside it, rebuild the layout from the config's `energy_config` and assert it matches the file's, which is the disagreement `_member_rows` otherwise finds one batch into training.

## Sources

The code above, read at the stamped commit, plus `docs/design/ragged_multi_molecule_state.md` and `docs/design/conformer_conditional_stack.md`. Three memory files (project_conformer_conditional_build_state, project_conformer_stack_is_single_molecule_below_the_policy, project_encoder_and_conformer_atom_orders_differ) located the code and were not used as evidence.

Where the design notes and the code differ. `conformer_conditional_stack.md` section 5 records the n-body correlator heads as docstrings only; `DoFCorrelator` is built and supplies `f_j`. The same section records no `log Z(c)` head; `conformer_gfn.py`'s docstring states that `init_flow_model`'s conditional `scalarMLP` over the pooled embedding is that head, and no separate `Z_MLP` exists. Section 6 lists the chart as not graph-determined; `linearity_source` takes the MMFF-typed route whenever the force field is `mmff`. The same section lists `log |dq/dx|` as missing from the reward; `log_chart_jacobian` is added in `prebuilt_sample_to_reward`. The gaps `ragged_multi_molecule_state.md` still lists match the code.
