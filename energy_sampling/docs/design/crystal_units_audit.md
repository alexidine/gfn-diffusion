# Z'>1 and set-of-units audit: crystal building, analysis and the GFN code that uses them

**Status: AUDIT, 2026-10-22. Observations about the code at the tree named below; no authority over code.** Asked for by the owner before any Z'>1 or set-of-units work ("It rests on intricate assumptions about the shapes of input tensors ... would have to be done very carefully"). Drafted by four read-only readers and a synthesizer, then checked claim by claim by a separate verifier; its corrections are applied inline and its additions are in section 7. Probes and outputs: `artifacts/fd_design_oct15/units_audit_oct22/` (verifier probes in `verify/`).

**Since the audit (MXtalTools working tree, uncommitted):** stages 0-3 of section 5 were built on 2026-10-22/23 (pins, refusals, the atom -> unit map `unit_ind`, the batch helpers' unit and edge levels), stage 5 on 2026-10-23 (the per-unit all-atom cluster reach: #16 fixed, #11's summed radius no longer reaches the builder), and stage 4 on 2026-10-23 (one unit-cell element layout: #15 fixed). What each stage flipped and re-pinned is logged in MXtalTools' `tests/characterization/README.md`; the findings below describe the 2026-10-22 tree.

Scope: the MXtalTools working tree on 2026-10-22 (HEAD 1a0c138b plus large uncommitted changes; `crystal_io`, `crystal_steps`, `crystal_store`, `crystal_chem` and `crystal_recipes` are untracked) and the gfn_diffusion `conditional` working tree. MXtalTools paths are relative to `MXtalTools/mxtaltools/`, and `dc/` means `dataset_utils/data_class_methods/`. GFN paths are relative to `gfn_diffusion/energy_sampling/`.

Two cases are tested throughout:
- **(i)** Z'>1 of one compound, with each unit having its own conformer.
- **(ii)** units of different compounds.

Each outcome is marked with one of these labels:
- **SILENT**: wrong value, no error.
- **CRASH**: an exception is raised.
- **REFUSED**: the row gets a status code.
- **COST**: right answer, more compute.
- **FINE**: no problem.

"Probe" means a CPU scratch script under `artifacts/fd_design_oct15/units_audit_oct22/{datamodel,builders,analysis,gfn}/`.

## 1. Topline

For one compound with its own conformer in each unit, MXtalTools already builds and scores Z'>1 crystals correctly: eLJ matches a float64 brute-force lattice sum with the same pair count. The GFN trainer does not. It orients the whole multi-unit graph as one body and then builds with `std_orientation=False`, which is correct only for the identical, coincident copies in today's Z'=2 priors. On per-unit conformers or on rows from the new ingestion chain it is off by up to about 2000 raw eLJ units, with no error.

Units of different compounds break wherever units are found by arithmetic (`num_atoms // z_prime`) or elements are copied from unit 0's layout. The results are:
- wrong energies, with no error, whenever the atom total divides by Z';
- wrong elements on the MLIP routes;
- compounds swapping slots during canonicalisation.

Separately, the batch helpers carry any uneven-length per-unit or per-bond value whole, without slicing it and without an error.

## 2. How a crystal is carried today

```
one crystal row b   (one PyG graph; batch[N] / ptr[B+1] index its atoms)
├─ per atom [N]     pos[N,3]  z[N]  x[N]  = | unit 0 (n atoms) | unit 1 (n atoms) | ... |   n = num_atoms // z_prime
│                   mol_ind[N]: unit of each atom. Ingestion writes it; the builder never reads it.
├─ per row [B]      num_atoms (all units)  z_prime (long)  sym_mult (operator count)  sg_ind  T_fc/T_cf
│                   cell_lengths/angles  symmetry_operators (list of B arrays)  identifier (one string)
│                   radius = Σ_u r_u   mass = Σ_u m_u   mol_volume = Σ_u v_u   smiles 'A|B'
├─ per unit,        aunit_centroid[B,3·Zmax]  aunit_orientation[B,3·Zmax]  aunit_handedness[B,Zmax]
│  padded to the    slot k ↔ atom block k by POSITION only; slots k ≥ z_prime are pads (value depends on writer)
│  batch width      ingestion only: unit_radius/mass/volume[B,Zmax] (NaN pad), aunit_op[B,Zmax] (−1 pad)
└─ per batch        max_z_prime (python int) = the builder's dispatch key

latent / full_cell_parameters [B, 6+6·Zmax] = | lengths 3 | angles 3 | centroids 3·Zmax | orientations 3·Zmax |
```

Build path when `max_z_prime > 1` (`mol2cluster`, dc/crystal_building.py:426-472):

```
row (Z' units) ─split_to_zp1_batch─▶ Z' unit-graphs (n atoms each; slot k; radius = ROW SUM)
  ─pose_aunit (reads slot 0 of each unit-graph)─▶ build_unit_cell ─▶ build_cluster(cutoff + zp_buffer)
  ─join_zp1_cluster_batch─▶ cluster [crystal][unit][translation][image][atom]
                            aux_ind 0 on each unit's own copy; mol_ind numbered batch-wide
unit cell layout: [unit][image][atom]   (at Z'=1 this is the same as [image][atom])
radial graph:     intramolecular = same mol_ind among aux_ind-0 atoms; contacts between units → intermolecular
elj:              sum over all units' inside atoms; GFN divides by z_prime
```

GFN path:

```
file rows ─orient_molecule('std') on the WHOLE graph─▶ mol_batch
  ─condition_samples: draws sg and Z', writes z_prime, atoms untouched─▶
  ─init_blank_crystal_batch: handedness := +1 for every slot; latent → slots; canonicalize_zp_aunits moves atom blocks─▶
  ─analyze(['reduction_en','elj'], cutoff 10, std_orientation=False)─▶ reward elj/z_prime + pressure + Jacobians
```

The builder internally already works with units as graphs: split, then build each unit, then rejoin. Every unit has its own conformer and its own pose. The one-compound assumptions live in:
- how units are found (equal blocks);
- how elements are laid out (copied from unit 0);
- per-row quantities that are sums over units;
- canonical ordering;
- the GFN frame handling.

## 3. Assumption map

### A. Data model and batch helpers (`dataset_utils/data_classes.py`, dc/crystal_ops.py, `dataset_utils/utils.py`)

**1. Units are Z' equal contiguous atom blocks. Nothing authoritative maps atoms to units.**
- **Where:** dc/crystal_building.py:76-78, 97; dc/crystal_ops.py:186-198 (`_unit_block_moves`); dc/crystal_io.py:1210, 1240; dc/crystal_steps.py:1561, 1723; crystal_building/utils.py:410-415, 491; analysis/exp6_ff.py:342-351; analysis/crystal_rdf.py:114-116, 273-295.
- **Relied on by:** the split, canonical block moves, pose parameterisation, the rebuild check, exp6 typing and the RDF environment labels. None of them reads `mol_ind`.
- **(i):** FINE.
- **(ii):** **SILENT** when the total divides by Z':
  - ethanol (9) + methane (5) split 7+7 gave eLJ −68.7 (std_orientation False; −70.4 with True) against −58.9 when the true units are posed one by one, at cutoff 10. (Corrected 2026-10-22 by the stage-0 characterisation work: the +2607.8 first quoted here re-divided the wrongly posed 7/7 unit cell into 9/5 blocks and was not the true crystal.);
  - 6+8 atoms split 7/7 with status ok in `from_arrays`;
  - 23+21 atoms split 22/22 and gave a plausible eLJ of −1241.
- **(ii) otherwise:** CRASH. The only guard is a bare `assert` at dc/crystal_building.py:83; under `python -O` it splits 12/12 out of 25 atoms and fails later with an IndexError.

**2. `z_prime` is a label that nothing checks against the atoms.**
- A one-molecule row labelled Z'=2 is cut in half and still scored: eLJ 259.95, 200.86 and +2968 in three probes. **SILENT**, or CRASH when the atom count is odd.
- Writers that change `z_prime` without touching atoms:
  - GFN `condition_samples` (energies/molecular_crystal.py:1683);
  - `buffer._as_batch` (buffer.py:617);
  - `collate_data_list(max_z_prime=1)` (utils.py:132-143).

**3. The batch helpers know only two sliced levels: per atom and per row.**
- **Where:** `_length_level` (data_classes.py:40-59), `_attribute_levels` (443-484), `_recorded_level` (486-512) and `rebuild_simple_slice_inc_` (514-589) classify values by length. Anything else is carried whole by `subsample_new_batch` (632-779) and `append_batch` (781-980), and `*_index` keys are not re-offset on subsample.
- **Today:** FINE, because the only per-unit values are padded per row.
- **Worse than carried whole (verifier):** `_length_level` reads dim 0, but keys containing `index` or `face` are concatenated on the LAST dim (`__cat_dim__`, data_classes.py:222). On a 2-graph batch a `[2,E]` index tensor is classified as per graph and its rows are gathered: `subsample_new_batch([1])` turned `tree_index [2,17]` into `[1,17]`, a silent wrong shape. On a 2-node batch it would be read as per node; only with 3 or more graphs is it carried whole.
- **Any uneven-length per-unit or per-bond value: SILENT.**
  - Selecting one of 3 rows returned all 5 per-unit values.
  - Selecting one of 4 molecules returned all 24 conformer-tree bonds, with indices up to 27, on a 7-atom batch.
  - `append_batch` kept 5 of 10 per-unit values; with identical `edge_index` in both inputs it kept 6 edges for 12 atoms.
- **What already works:**
  - `batch_to_list` and PyG's own collate split a value correctly when it has a `unit_ptr` slice entry.
  - `crystal_io.select_rows` (499-584) and `featurize_cifs.concat_batches._level` (689-701) refuse such values loudly.
  - `crystal_store` saves and reloads them correctly.

**4. One batch-wide `max_z_prime` and one padded width.**
- `append_batch` across widths: CRASH.
- `collate_data_list` with a smaller `max_z_prime` truncates the slots and keeps `z_prime=2`, which fails later with an IndexError at the split. It cannot widen a batch (AssertionError).
- `batch_to_list` gives every element the batch's `max_z_prime`.
- `buffer._as_batch` (buffer.py:612-617) truncates the slots and clips `z_prime`:
  - A Z'=2 row stored at width 1 becomes one rigid two-molecule body: **SILENT**. eLJ went from 949.97 to 341.78 in one probe, and from −1327.5 to −1676.8 on acridine.
  - A narrower batch going into a wider store: CRASH.
- `init_mol_dataset` (train.py:3939-3955) never compares the file's Z' with `z_primes`.

**5. Pad slots are recognised only through `z_prime`, and their values depend on which code wrote them.**

| Writer | centroid pad | orientation pad | handedness pad |
|---|---|---|---|
| constructor (data_classes.py:1405-1413) | 0 | 1 | 1 |
| `collate_data_list`, `from_arrays`, `parameterize_pose` | 0 | 0 | 0 |
| canonicalisation (crystal_building/utils.py:1449-1456) | 0.5 | π/2 | left as is |
| GFN blank batch (molecular_crystal.py:1515-1519) | 1 | 1 | 1 |

- Example: the same Z'=1 crystal holds orientation `[.3,.2,1, 1,1,1]` on its own and `[…,0,0,0]` after being collated next to a Z'=2 row.
- Code that reads pads: `zp_buffer` (#13), the GFN rotational Jacobian (#33), the GFN bound term over all raw dimensions (molecular_crystal.py:841-851) and the dead-row masks (#36).

**6. The row's `radius`, `mass` and `mol_volume` are sums over units.**
- **Writers:**
  - the `set_mol_attrs` list path (crystal_ops.py:410-417);
  - crystal_io.py:1266-1271;
  - crystal_steps.py:998-1006;
  - zp_doubling.py:165-175;
  - the old featurizer (featurization_utils.py:526-528). It sums over the original units while `pos` holds copies of unit 0, so its stored radius need not match `pos`.
- **Readers:**
  - `box_analysis` (crystal_ops.py:470-505) is correct because of the sum (packing 0.8106 computed both ways). It becomes **SILENT** wrong by a factor of 1/Z' if any writer stores a single-unit value on a Z'>1 row.
  - The latent length scale `radius / z_prime^(2/3)` (crystal_ops.py:843, 909) equals Z'^(1/3)·r for one compound, which is consistent with cell volume. For (ii) it is a number that depends on the mix: 2.573 Å for A (1.981 Å) + B (2.102 Å).
  - The split gives every unit the sum (#11): COST.

Table 1. Molecular radius (Å) under each convention in use, one hand-built structure per row (data-model probe 3). Only the Z'=1 row agrees across conventions.

| crystal | stored row radius (sum over units) | `radius_calculation()` on the whole asymmetric unit | latent scale = radius / Z'^(2/3) | radius each unit-graph gets from the split | true per-unit radii |
|---|---|---|---|---|---|
| Z'=1, A | 1.981 | 1.981 | 1.981 | 1.981 | 1.981 |
| Z'=2, A + A2 (one compound, per-unit conformers) | 3.964 | 4.919 | 2.497 | 3.964 | 1.981, 1.983 |
| Z'=2, A + B (two compounds) | 4.084 | 5.173 | 2.573 | 4.084 (by reading) | 1.981, 2.102 |

**7. The list constructor merges molecules by the type of each key** (`set_mol_attrs` list path, crystal_ops.py:393-417).
- **Rules:** tensors are concatenated on dim 0, 0-dim values are summed, strings are joined with `|`, and `identifier` is molecule 0's. Keys come from molecule 0: a later molecule missing one raises KeyError, and extra keys are dropped. No unit map is written.
- **FINE:** `pos` keeps each unit's own conformer.
- **CRASH:** `edge_index` comes out `[4,6]`, with the second molecule's indices not offset, so collation fails.
- **SILENT, by reading:**
  - `[1,F]` row tensors become `[Z',F]` and are out of line with the rows after collation;
  - 0-dim per-molecule ids are summed;
  - for (ii), every identifier after the first is lost.
- **CRASH:** separately, `z_prime` is shape `[]` when given as an int and `[1]` by default (crystal_ops.py:362-368), and collating the two fails.
- **SILENT, then CRASH (verifier):** molecules taken out of a collated batch, as dataset rows are, carry `num_atoms`, `radius`, `mass` and `mol_volume` shaped `[1]`, and the list constructor concatenates them instead of summing. One Z'=2 crystal then stores `num_atoms [12,12]`, `radius [2.245,2.245]` and `packing_coeff [0.3958,0.3958]`, half the true 0.7916, written silently by `box_analysis`; `analyze` then raises (AssertionError at crystal_building.py:83 alone, a size mismatch when collated). Fresh `MolData` with 0-dim scalars sum correctly. `crystal_search/coordinator.py:263-266` works around exactly this.

**8. `mol_ind` has two meanings.**
- In rows, it is the row-local unit index. `from_arrays`, `find_molecules` and the old featurizer write it; the list constructor and the GFN blank batch do not. The sg 9 Z'=2 priors carry all zeros.
- In clusters, it is the molecule's index in the cluster, numbered across the whole batch after the join (crystal_building.py:202-209; crystal_building/utils.py:467).
- FINE today, but the natural carrier for an atom→unit map already has a second meaning.

### B. Ingestion (dc/crystal_steps.py, dc/crystal_io.py, the older featurizer)

**9. `find_molecules`** (crystal_steps.py:167-376; `_find_row` 1012-1082).
- Z' is the number of bonded components.
- Units 1 and up are re-indexed onto unit 0's atom order (1053-1071), and each keeps its own coordinates.
- Units of unequal size, or whose graphs do not match, get the status `multi_component` (1050-1052, 1059).
- `_write_unit_properties` (969-1005) is the only routine that already works through an atom→unit map.
- **(i):** FINE. **(ii):** REFUSED.

**10. `parameterize_pose`, `check_rebuild` and the older featurizer.**
- `parameterize_pose` (378-516; equal blocks at 1561) and `check_rebuild` (518-616; layout [unit][operator][atom] at 1722-1726): (i) FINE, the rebuild deviates by 2.5e-6 Å. (ii) cannot be reached, because #9 refuses first.
- The older featurizer (featurize_cif_chunks.py:130-185; featurization_utils.py:513-575, 642-692) refuses cocrystals. `overwrite_conformer_to_zp` copies unit 0's `pos`/`z`/`x` into every unit, so per-unit conformers are discarded.

### C. Builder (dc/crystal_building.py, crystal_building/utils.py)

**11. `split_to_zp1_batch`** (30-119) is the builder's only per-unit view.
- Each unit gets n atoms (#1) and slot columns `3k:3k+3` (109-117), and inherits the row sums of radius, mass and volume (98-100).
- Other row attributes (`identifier`, `unit_*`, `_slice_dict`) keep the parent's length.
- **(i):** geometry FINE, since each unit uses its own conformer.
- **COST:** the same crystal written as Z'=2 builds 16,848 cluster atoms per unit against 4,056 at Z'=1 (4.2×). eLJ per molecule is 419.538 against 419.541, and the pair counts are identical.
- **(ii):** see #1.

**12. The build dispatches on the batch's `max_z_prime`, never on each row's `z_prime`** (426, 496).
- Row Z'=2 in a `max_z_prime=1` batch:
  - `mol2ucell` (the MLIP route) poses it as one rigid body: **SILENT** (mechanism confirmed by reading; the probe printed no number).
  - `mol2cluster` stops at `assert False` in `_index_mols`: CRASH. Under `-O` it is **SILENT**: both units get the same `mol_ind`, and the contacts between them drop out.
- All rows Z'=1 in a `max_z_prime=2` batch: CRASH in the join (227) for `elj` and `rdf`. Any `z_primes: [1,2]` run would hit this on a batch, or a `batched_analyze_crystal_batch` chunk (molecular_crystal.py:1376-1476), that holds only Z'=1 rows.
- A Z'=2 row clipped to Z'=1 by #4, in a mixed batch: **SILENT** one rigid body, eLJ −196.6, packing 0.164.

**13. `zp_buffer`** (431-466) is the largest distance between a crystal's centroid slots, pads included, and it is added to the cutoff.
- For real slots the geometry is right (the cluster pairs matched brute force).
- Pads cost compute: a Z'=1 row in a Z'=2 batch gets a 4.41 Å buffer and 1984 cluster atoms instead of 1024.
- Through #16 that row's score then depends on its batch-mates: a pyridone Z'=1 row has 4993 intermolecular edges alone and 4995 inside a `max_z_prime=2` batch, exactly the two pairs a widened selection recovers (eLJ −200.4142 against −200.4179, cutoff 10; `verify/v1_out.txt`).

**14. `pose_aunit` → `get_aunit_positions` reads slot 0 only** (crystal_building.py:230-275; utils.py:1352-1408).
- On split unit-graphs this is FINE for (i) and (ii): intra-unit distances hold to 1.5e-6 Å.
- On an unsplit Z'>1 graph it moves the whole asymmetric unit as one body:
  - `compute_standard_cell` (crystal_ops.py:3213): **SILENT** for anything that reads `std_batch.pos`. The output `pos` is not the posed units, and in 2 of 3 handedness cases both units come out mirrored. A rebuild with `std_orientation=True` still reproduces the crystal.
  - `get_fd_params` (1756): CRASH in `aunit2ucell`'s einsum.
- Other helpers that read slot 0 only raise or change tensor shape at Z'>1: `zp1_cell_parameters`, `zp1_std_cell_parameters`, `standardize_aunit_orientation`, `noise_cell_parameters`, `do_embedding`, `sample_reasonable_random_parameters`, `_transform_aunit_params`, `get_fd_params`, and `fold_to_fd`, which writes `[n,3]` into an `[n,6]` field (`zp1_std_cell_parameters` crystal_ops.py:1350, `zp1_cell_parameters` 1984, `standardize_aunit_orientation` 2115, `noise_cell_parameters` 1242, `do_embedding` 2772, `sample_reasonable_random_parameters` 994, the FD helpers 1619-1820).

**15. At Z'>1 the unit cell is laid out unit by unit, `[unit][image][atom]`** (`join_zp1_ucell_batch` 134-157), not image by image as `[image][whole asymmetric unit]`.
- Code that tiles the asymmetric unit's `z` `sym_mult` times assumes the image-by-image layout:
  - mlip_interfaces/uma_utils.py:832-853 (its docstring states the image-major layout), 914-915, 1099;
  - mlip_interfaces/AL_mace_utils.py:166, 714-715, 940-941;
  - common/ase_interface.py:165;
  - the spglib labels at crystal_ops.py:3144;
  - zp_doubling.py:187-188.
- **(i):** FINE only because every unit carries unit 0's atom order: the new chain re-indexes it, and the old featurizer copies it.
- **(ii): SILENT wrong elements.** 4 of 36 unit-cell rows were wrong for ethanol + dimethyl ether in P-1, and 8 of 96 for pyridone + fluorobenzene in sg 14.
- No existing check catches it: the MLIP size checks (uma_utils.py:919, AL_mace_utils.py:718) pass, and the list-vs-vectorised equivalence test tiles the same way on both sides. spglib happened to raise.
- `tests/test_unit_cell_layout.py` pins the image-by-image layout using Z'=1 crystals only.

**16. The cluster reach test uses a heavy-atom radius against all-atom centroids** (radius: dc/mol_methods.py:173-203, heavy atoms about the heavy-atom centroid; test: crystal_building/utils.py:102-117, 181, 659, with a 0.1 Å margin).
- **SILENT and already live at Z'=1, but small.** Phenacetin's hydrogens reach up to 0.90 Å beyond that radius. At cutoff 6, one H…H pair at 5.81 Å was missed (654 of 655 found).
- At the live cutoff of 10 Å (molecular_crystal.py:745), measured by the verifier against a selection widened by 4 Å: phenacetin P-1 misses 1 pair (10021 of 10022 edges, |ΔeLJ| 8e-4 raw), pyridone P2₁/c misses 2 (|ΔeLJ| 3.7e-3), phenacetin P2₁/c misses none at 10 Å but 4 at 6 Å (|ΔeLJ| 0.088). Live, real and tiny at 10 Å.
- At Z'>1 the summed radius (#11) hides the problem, so switching to a per-unit radius would expose it.

**17. `ucell2cluster` and the cluster join.**
- `ucell2cluster` (utils.py:43-155) copies `z` and `x` per unit atom (459-460): FINE for both cases.
- `_index_mols`' Z'>1 branch (366-389) runs only under `-O`, and `_molwise_indexing` (474) is never called.
- `join_zp1_cluster_batch` (159-227): FINE for both cases.

**18. `reparameterize_unit_cell`** → `parameterize_crystal_batch` / `identify_canonical_asymmetric_unit` / `extract_aunit_orientation` (crystal_ops.py:1531-1617; utils.py:840-1124).
- Works per split unit, and each unit picks its own image independently.
- **(i):** exact round trip: the pose returns to 4 decimals, `pos` within 9.5e-7 Å, the cell within 1.9e-5 Å.
- **(ii):** FINE with equal atom counts; otherwise #1 applies.

**19. Canonical unit order sorts by the norm of the fractional centroid, so it treats every unit as exchangeable.**
- **Where:** `canonicalize_zp_aunits` (crystal_ops.py:3018-3056), `_canonical_unit_order` (101-126), `_unit_block_moves` (129-253), `canonicalize_aunit_order` (utils.py:1411; body 1436-1458). It is reached from `latent_params` and from GFN `instantiate_crystals`.
- **(i):** FINE. Blocks swap exactly, and `unit_radius` reorders from [10,11] to [11,10].
- **(ii): SILENT.**
  - Equal atom counts: the compounds trade slots, while `identifier` and `smiles` ('A|B6') are not reordered.
  - Unequal counts with a total that divides: the order is scrambled; block 0 became B minus its first atom.

**20. `canonicalize_free_axes`** (crystal_ops.py:2955-3016) returns early for the **whole batch** if any row is Z'>1 (line 3000, as its docstring says).
- Z'=1 rows in a mixed batch lose their free-axis gauge fix: **SILENT**.
- Dormant on mk_dev, which has no Z'>1 rows.

**21. Other files.**
- `crystal_building/crystal_latent_transforms.py` has no live importer.
- `zp_doubling.py` is exact for one compound; for two compounds it has the element-layout problem of #15.

### D. Analysis and energies (dc/crystal_analysis.py, analysis/vdw_analysis.py, models/functions/radial_graph.py, analysis/exp6_ff.py, analysis/crystal_rdf.py)

**22. Intramolecular exclusion is `mol_ind` equality among inside atoms** (radial_graph.py:318-324). Contacts between units of the same asymmetric unit move to the intermolecular list.
- FINE for both cases:
  - butanediol at Z'=2 with per-unit conformers gave −201.3020 against −201.3019 brute force, with 7026 edges each;
  - ethanol + dimethyl ether agreed up to the cutoff-boundary convention: the 7 extra edges are pairs at exactly 10.000 Å (an atom and its own lattice image, since the probe cell length equals the cutoff), which the radial graph includes and the brute force excluded; the one-compound control shows the same artifact.
- It assumes no covalent bond between units and no unit on a special position.

**23. `elj` and the other pair energies return the sum over all units' inside atoms** (crystal_analysis.py:420-484; vdw_analysis.py:624-681), so they scale with Z'. The same crystal written as Z'=1 and as Z'=2 gave a ratio of 2.00000.
- GFN divides by `z_prime` (molecular_crystal.py:869-870). That is per molecule for (i) and per average unit for (ii).
- **Trap:** `cluster_exp6_energy` already divides by 2·z_prime (exp6_ff.py:616). A consumer that divides by `z_prime` as it does for `elj` would get half the value at Z'=2. GFN refuses exp6 today (molecular_crystal.py:934).

**24. RDF** (`compute_rdf` 1917-1975; crystal_rdf.py:23-158). The histogram is divided by `z_prime`.
- Modes None, `elementwise` and `envwise` agree between the Z'=1 and Z'=2 descriptions of the same crystal to 1e-7.
- **`atomwise` (114-116): SILENT change of channel meaning** for (i).
  - Labels are `arange(num_atoms[0])` repeated over the cluster's storage order.
  - At Z'=2 that gives 528 channels instead of 136, and image labels alternate with molecule parity.
  - The channels become comparable only after folding the labels mod n.
  - Using `num_atoms[0]` is also wrong for any batch that mixes compounds.
- `envwise` (273-295) splits units by #1. It detects bonds by distance under 1.8 Å, so sampled conformers with close contacts could change the labels.

**25. exp6 atom typing** (342-351, 472-522) splits units by equal blocks and finds bonds from distances on the stored, unposed `pos`. For (ii) with unequal counts the units are mis-split. In the probe it raised ("1 H atoms have no bonded atom"), but it would be silent if the wrong fragments happened to be typable.

**26. MLIP lattice energy is `E_cell/(sym_mult·z_prime) − mean_k E_gas,k`** (1058, 1216).
- That is per average unit, and it assumes `sym_mult·z_prime` molecules per cell (no special positions).
- The gas leg works per unit, so #1 applies.
- When `self` is a cluster it leaves `x` and `ptr` stale; GFN never sends MLIP keys through clusters.
- Element assignment: see #15.

**27. Dispatch and the other computes.**
- `analyze` dispatch (734-944) inherits the crash in #12.
- `reduction_en` (1977-2018) and `_nan_undefined_crystals` (946-968): FINE.
- `ellipsoid`/`ellipsoid_emb` (dc/ellipsoid_ops.py:148-151) and the GNN periodize step (radial_graph.py:303 → models/graph_models/graph_neural_network.py:681-721) treat a cluster as consecutive copies of the whole asymmetric unit. That is wrong at Z'>1 even for (i), but neither is on the eLJ route.

**28. ELJ and exp6 have no intramolecular term.**
- This is FINE while conformers are fixed. Once conformers are sampled jointly with the packing, a distorted conformer costs nothing on these routes.
- The per-call MLIP gas leg subtracts each sample's own intramolecular energy, which removes conformational strain from the reward.
- These are design decisions (Q3), not bugs.

### E. GFN code that uses the crystal machinery (energy_sampling/)

**29. Shared frame. SILENT for (i).**
- The trainer orients the whole multi-unit graph once with `orient_molecule('std')` (buffer.py:622-630; train.py:732, 2565, 6521, 10279, 11301). It then builds with `std_orientation=False` (molecular_crystal.py:745-747), so every unit is posed in the frame of the whole graph.
- That is correct only for identical copies sitting at the origin (the sg 9 Z'=2 priors).
- `init_prior_dataset` (train.py:3778) and `init_anchor_buffer_seed` (10936-10943) rescore the stored `pos` before any orientation, which requires the file to be unchanged by `orient('std')`. Nothing checks this, and the Z'=2 priors are not: orienting moves their atoms by up to 9.23 Å.
- A builder probe confirms it independently: `orient('std')` followed by a `std_orientation=False` build at Z'=2 gave −288.98 against a correct 209.59.

Table 2. Trainer route (whole-graph `orient_molecule('std')`, then `instantiate_crystals`, then build with `std_orientation=False`) against the MXtalTools per-unit route (`std_orientation=True`), on Z'=2 rows. Handedness is +1 for every unit in both routes. The p2 column uses 6 acridine rows from `deadrow10k_sg9_zp2_elj.pt`; the p6 column uses 2 synthetic Z'=2 crystals built from CSD molecules with no point symmetry. eLJ is raw (lj_coeff 1); acridine eLJ is about −1300. Compare each case with the control in row A. The nearest-atom column has a float32 floor of about 0.02-0.03 Å (torch.cdist's matrix path): row A's per-atom deviation is 9.3e-6 Å, so case B's 0.031 Å is at the floor and its real signal is ΔeLJ 2.5 (p2) and 3.34 Å (p6).

| case (how `pos` is laid out) | max nearest-atom deviation (Å), p2 / p6 | max \|ΔeLJ\| (raw eLJ units), p2 |
|---|---|---|
| A: identical copies at the origin (control) | 0.022 / 0.008 | 0.0009 |
| B: unit 1 a different conformer | 0.031 / 3.34 | 2.5 |
| E: same conformer, unit 1 rigidly turned 40° | 1.49 / 2.61 | 408 |
| C: posed copies (ingestion-chain layout) | 3.14 / – | 1957 |
| D: posed copies, distinct conformers | 3.08 / – | 2177 |

**30. Every unit is built at handedness +1.** The build never reads `aunit_handedness`: `get_aunit_positions` applies handedness only with `std_orientation=True` (crystal_building/utils.py:1390-1394), and GFN builds with `std_orientation=False` (molecular_crystal.py:747; train.py:3662-3666 says so). The effective +1 comes from `orient_molecule('std')` with `target_handedness=None` on the whole graph (mol_methods.py:346-349, 370; buffer.py:630; train.py:732, 2565, 6521, 10279, 11301); the +1 written at molecular_crystal.py:1515 has no effect on the build.
- **SILENT** for any unit with handedness −1. The repository's own measurement (train.py:3674-3678; 200 rows per sign pattern, 2026-09-30) puts it at a median 0.25-0.46 and at most 3.4 raw eLJ units on acridine Z'=2.
- `_refuse_mirrored_rows` (train.py:242-263, 3659-3711) only prints for Z'>1 rows.

**31. Z' is drawn per condition, independently of the atoms** (`condition_samples`, molecular_crystal.py:1639-1644, 1683). The callers that pass no Z' are train.py:2578, 6523, 7421, 7660 and 7857.
- Dormant on mk_dev (`z_primes: [1]`, line 1272).
- With `[1,2]`, the backward and replay draws re-draw the Z' of stored rows: **SILENT**. In the probe, Z' went 2→1 on 2 of 8 rows, log R went 664.3→1311.3, and `condition_id` was re-keyed.
- A one-molecule row drawn at Z'=2 then hits #2.
- The same re-draw also changes the space group on multi-sg runs, and `config_invariants.py:1555-1561` warns about it only for `condition_draw` stages. Dormant: mk_dev has `space_groups: [2]`.

**32. One dense width `6+6·Zmax` per run.**
- **Where:** `data_ndim` (molecular_crystal.py:166-172, 351); `GFN.get_periodic_dimensions` (models/gfn.py:269-365, check at 329-334); the policy output blocks; the buffers' `x`, `traj` and `force_legs` (buffer.py:295-391, 632-659); stores built at `max(z_primes)` (train.py:660-674).
- **(i):** fine for poses. Conformer degrees of freedom would fit for one compound as `[B, Zmax, d]` with a mask.
- **(ii):** needs a variable-length layout.

**33. Reward terms.**
- `attr / z_prime` (870); pressure `V/(sym_mult·z')` (883); density penalty on the summed volume (866); fractional Jacobian `−z'·T·log(V/sym_mult)` (1204).
- The rotational Jacobian sums over all slots with no Z' mask (1202-1203). Pad slots are constant after canonicalisation, which by arithmetic (not run) gives about 0.25·T per pad slot.
- **(i):** FINE.
- **(ii):** the reward is per average unit at one temperature. That is a convention for you to choose (Q3).

**34. The ordering penalty enforces one centroid-norm order over all slots** (`compute_zp_order_penalty`, 1221-1260, together with the canonicalisation in `instantiate_crystals`, 505-519).
- **(i) with fixed, differing per-unit conformers: SILENT.** Slot k is conformer k of the molecule file, so forward draws can only place conformer 0 nearer the origin. Canonicalised buffer rows contain the other order, so part of the support is lost and the targets are inconsistent. This is from reading the code; the probe confirmed that the blocks move.
- Exchangeability is correct once the conformer is sampled together with its unit.
- **(ii):** the same problem across compounds.

**35. Condition identity and conditioning are per crystal.**
- **Where:** `mol_id` comes from the crystal-level `identifier`, and `condition_id = mol_id·n_sg·n_zp + sg_local·n_zp + zp_local` (367-433, 1692-1706; train.py:3957-3998). There is one Z' scalar per crystal (1651) and one embedding per crystal (1674-1681), and the condition GNN runs over the whole crystal graph (models/gfn.py:181-227, 1286-1301).
- **(i):** the embedding is NaN on coincident copies, and on posed units it lands 1.25-2.67× (relative distance) away from the single-molecule embedding.
- **(ii): SILENT.** A cocrystal row takes molecule 0's identifier, so it shares a `mol_id` with pure A.

**36. Dead rows** (train.py:3128-3216; models/dead_latent_rows.py:71-95).
- `free_centroid_rows` covers only the first block and returns nothing when `max_z_prime > 1`.
- Deadness is set per run, not per row, so the pad slots of rows with fewer units are not held dead and rollouts move them freely.

**37. The gas-phase reference cache is keyed by `mol_id`** (550-634) and computed on the first row seen. It is used on the MLIP routes only.
- **(i):** every later row carries an offset.
- The cache becomes invalid once conformers are sampled, at Z'=1 too.

**38. The list path in `prebuilt_sample_to_reward`** (1308-1340) collates without passing `max_z_prime`. By reading, a list of only Z'=1 rows would crash at the reshape at line 1249. Not probed.

**39. Data-processing scripts.**
- Z'=1 only: capped_mc.py:183, 210; fd_canonical_prior.py:236-240, 255; symmetrize_prior.py:218-222; build_qm9_full_prior.py:285; pool_anchors.py:74, 78; pool_assemble.py:61-68; generate_sg_prior.py:96-102.
- Build Z'>1 by duplicating one molecule into coincident copies: process_target.py:41-56; eval/nikos_comparison/double_zp1_map.py:34.

**Tests that pin today's shapes.** These would need updating:
- test_dead_latent_rows.py:55-64, 680-737, 760-785;
- test_batch_invariance.py:108-112, 246-301;
- test_mirrored_rows.py:44-57;
- test_periodic_scoring.py:29, 43;
- test_buffer_columns.py:48;
- test_csp_replay.py:29-184;
- test_mxtaltools_crystal_boundary.py:55, 87 (asserts `z_primes == [1]`);
- test_lj_coeff_carried.py:115-123;
- test_energy_reference_reward.py:126-132;
- test_dead_latent_rows_deep.py.

**Live today on mk_dev (Z'=1, sg 2):** only #16.

**Dormant until `z_primes` or `space_groups` has more than one entry:** #12 (the all-Z'=1 crash), #13 (batch dependence), #20 and #31.

## 4. Data-model options for a set of units

Within one compound, all units have the same atom count and atom order, and they will have the same internal-coordinate count. So everything per unit can stay padded per row `[B, Zmax, ·]` for (i), including jointly sampled conformers. Variable-length per-unit storage is needed only for (ii).

**(a) One graph per crystal plus an explicit unit level. Recommended.**
- **Storage:**
  - The rows stay as they are, so `num_graphs` still counts crystals.
  - Add a per-atom unit index (or a per-row unit pointer) as the only definition of a unit, under a name other than the cluster's `mol_ind`.
  - Per-unit values go into tensors of uneven length per row with a slice entry; the policy keeps a padded pose view with an explicit unit mask.
- **Builder:** the split gathers atoms by the map instead of `num_atoms // z_prime` (about one function). Everything downstream already works per unit. The unit-by-unit layout becomes a documented contract, and one helper supplies elements for it.
- **Analysis:** only the equal-block sites change (exp6, RDF, utils.py:491).
- **Batch helpers:** add a unit level and an edge level to about ten functions; PyG's own collate and split already handle a `unit_ptr` entry.
- **GFN:** stored files, buffers and the latent width are unchanged.

**(b) Units as their own stored graphs, grouped into crystals.** This is the split's output made permanent.
- **Builder:** cheapest; the split disappears.
- **Everywhere else:** every per-crystal quantity (cell, `box_analysis`, the `elj` sum, `reduction_en`, RDF normalisation, the MLIP unit cell, reward, log Z, buffer rows) becomes a reduction over a variable number of graphs. `num_graphs` stops meaning "samples" throughout the trainer and every stored file.
- **Batch helpers:** still need a crystal level, because subsampling a crystal means taking a variable number of graphs. This is the same amount of work as (a), turned around.
- **Fit:** suits a set policy over units, but the crystal route stays dense under the 2026-09-10 decision.

**(c) Today's model plus guards.** Equal blocks only, and (ii) is refused.
- Smallest change, and it covers (i) entirely.
- `num_atoms // z_prime` remains the definition of a unit, so (ii) would later need (a) anyway.

## 5. Change plan, tests first

Stages 0-5 are MXtalTools-only. Each one is bit-identical on today's inputs except stage 5, whose small energy change is documented. None of them changes GFN behaviour at `z_primes: [1]`. If only (i) is wanted, stop after stage 6.

**Stage 0: pin today's behaviour (no code change).**
- **Fixtures:**
  - F1: Z'=1, 3-4 molecules without point symmetry (phenacetin among them), sg 2, 14 and 9, `std_orientation` True and False, cutoffs 6 and 10.
  - F2: the acridine sg 9 Z'=2 prior rows through GFN `instantiate_crystals` + `analyze`.
  - F3: one compound at Z'=2 with per-unit conformers from the ingestion chain, through the MXtalTools route.
  - F4: Z'=1 and Z'=2 rows in one batch with `max_z_prime=2`.
- **Pinned bit for bit on CPU (`torch.equal`):**
  - `latent_params` in both directions;
  - the output of `canonicalize_zp_aunits` (slots and atom order);
  - `unit_cell_pos` and `unit_cell_batch`;
  - cluster `pos`, `z`, `aux_ind` and `mol_ind`;
  - the intermolecular edge count;
  - `elj`, `reduction_en`, `packing_coeff` and `density`;
  - every RDF mode;
  - the MLIP input assembly (elements, positions, cell) without a model.
- **Independent check:** turn the float64 brute-force lattice sum from the analysis probe into a test helper and compare pair sets within the cutoff. It fails today on phenacetin at 6 Å (#16), so mark it as an expected failure that must start passing (strict xfail).
- **Strict xfails for known wrong behaviour:**
  - the unequal-unit split (ethanol + methane);
  - two-compound element assignment;
  - a one-molecule row relabelled Z'=2 must be refused;
  - a Z'=2 row going into a width-1 store must be refused;
  - the all-Z'=1 batch at `max_z_prime=2` must not crash;
  - the GFN trainer route on F3 must match the per-unit reference.
- **Fix the layout test's scope:** extend `tests/test_unit_cell_layout.py` to pin the unit-by-unit layout at Z'>1.

**Stage 1: refuse instead of guessing.**
- **Changes:**
  - The bare asserts (crystal_building.py:83, `_index_mols`) become exceptions that survive `-O`.
  - The builder dispatches on the rows' `z_prime`, and a row wider than the batch width raises.
  - `buffer._as_batch` and `collate_data_list` raise instead of truncating.
  - `init_mol_dataset` checks the file's Z' against `z_primes`.
  - `zp_buffer` ignores pad slots.
- **Contract:** valid inputs are unchanged; invalid inputs raise.
- **Proof:** F1-F3 bit-identical; the F4 Z'=1 row becomes bit-identical to scoring it alone; the refusal xfails start passing.

**Stage 2: make the atom→unit map authoritative.**
- **Changes:**
  - Every constructor writes the map: the list constructor, `from_arrays`, `find_molecules`, the featurizer and the GFN blank batch.
  - Construction checks that `z_prime` equals the number of units in the map.
  - Every `num_atoms // z_prime` site listed in #1 reads the map; equal blocks stay asserted for now.
- **Key name:** chosen on purpose. Keys containing `index` or `face` are concatenated on the last dim and offset by the node count at collation, keys containing `batch` are offset by max+1 (data_classes.py:219-233), and `_unit_block_moves` renumbers any key containing `index` or `face` as atom indices (crystal_ops.py:215). A map named `unit_index` would be corrupted silently; `mol_ind` survives today only because `ind` matches neither rule.
- Also fix the list constructor's concatenation of `[1]`-shaped row scalars (#7).
- **Contract:** a unit is what the map says, and `z_prime` can no longer disagree with the atoms.
- **Proof:** bit-identical, because the map equals the equal-block arithmetic for equal units; the relabel xfail starts passing.
- **Files:** rebuild the sg 9 Z'=2 priors from the ingestion chain rather than migrating them, since they also fail #29 and #30.

**Stage 3: give the batch helpers a unit level and an edge level.**
- **Changes:**
  - A value is per atom, per unit (sliced by the unit pointer), per edge (offset by atoms), per row, or refused. Nothing is silently carried whole.
  - Fix the list constructor's edge offsets, and make it refuse merges it cannot do.
  - Fix how dim-0 levelling treats tensors concatenated on the last dim (#3).
- **Proof:** the subsample and append cases of #3 pass; the store saves and reloads correctly; F1-F4 bit-identical.

**Stage 4: one element layout.**
- **Change:** one helper returns the element of every `unit_cell_pos` row through the same index the builder used. uma_utils, AL_mace_utils, ase_interface, the spglib labels and zp_doubling all use it.
- **Proof:** the two-compound element xfail starts passing; the Z'=1 and one-compound Z'=2 MLIP input assembly is bit-identical.

**Stage 5: per-unit reach for clusters.**
- **Change:** cell selection and the flag test use each unit's all-atom reach about its all-atom centroid. This fixes #16 and removes about half of the 4.2× Z'>1 cost: with `zp_buffer` kept, phenacetin P2₁/c at Z'=2 still builds 16,328 cluster atoms per unit (2.0× the Z'=1 4,056), and dropping `zp_buffer` loses 268 of 7,668 pairs per molecule. The rest needs nearest-image unit placement or sub-clusters centred on the asymmetric unit. The row's `radius`, which sets the latent scale, is left alone (Q2).
- **Contract:** every pair within the cutoff is found.
- **Proof:** the brute-force pair-set xfail starts passing. Re-pin F1 in this one change and report the energy differences per crystal.

**Stage 6: GFN for one compound.** This completes (i) with fixed conformers.
- **Changes:**
  - Orient per unit (split, orient, rejoin), or move to `std_orientation=True`, at the whole-graph sites listed in #29.
  - Carry each unit's handedness from the row, coupled to the orientation: orient each unit with its own handedness, or build with `std_orientation=True` (#30); carrying the slot value alone changes nothing.
  - Take Z' from the row on backward, replay and eval draws.
  - Turn the ordering penalty off, or restrict it to units with identical conformers.
  - Mask pad slots in the rotational Jacobian and in the dead-row masks, per row.
- **Proof:**
  - The F3 and Table 2 cases B-E match the per-unit reference, judged by order-matched per-atom or float64 deviations (not the float32 nearest-atom distance); a test covers the re-draw.
  - Then a short real run on a one-compound Z'=2 prior at production-shaped parameters, read before any battery (AGENTS verification rule 5). The prior must come from the ingestion chain (CSD, per-unit conformers): the crystal search can only make identical-copy Z'>1 rows (section 7), the one layout today's route already builds correctly.

**Before stage 7:** a frame convention for units with fewer than 3 heavy atoms or a linear shape (water, ammonia, monatomic ions, CO2, acetonitrile; section 7), and the fundamental-domain gauge for a set of units (crystal_fd refuses Z'>1).

**Stage 7: different compounds.**
- **Changes:**
  - A compound id per unit, with SMILES and identifier per unit.
  - Canonical order and the ordering penalty apply within a compound only.
  - `find_molecules` accepts `multi_component`.
  - Units of uneven size wherever the map is used.
  - Condition identity is the composition.
  - Atomwise RDF labels are per compound, and exp6 typing works per unit.
- **Proof:** ethanol + methane and pyridone + fluorobenzene match brute force; A + B6 never trade slots.

**Stage 8: conformer degrees of freedom per unit.** The owner said on 2026-10-22 that conformations will be sampled together with the packing, in internal coordinates, and that internal-coordinate assignment at ingestion is wanted, at least optionally. This happens with the conformer refactor and is outside this audit. Use `[B, Zmax, d]` padding for one compound and the stage-3 unit level for (ii). It needs Q3 answered first.

## 6. Questions for you

**Q1. Which data model?** I recommend (a): keep one graph per crystal, add the atom→unit map now, and add uneven-length per-unit storage only when multi-compound work starts.
- (b), units as stored graphs: cheapest for the builder, but it changes what `num_graphs` means across the trainer and every stored file.
- (c), guards only: smallest, but `num_atoms // z_prime` stays the definition of a unit.
- If you don't answer: stages 0-2 are the same under (a) and (c), so I'd go ahead with those.

**Q2. What should the row-level `radius` mean?** I recommend keeping it as the sum over units for the latent scale. For one compound, `radius/Z'^(2/3)` = Z'^(1/3)·r, which is consistent with cell volume; Z'=1 is unaffected, and no prior or checkpoint changes. The builder would use per-unit reach separately (stage 5).
- The alternative is to redefine it (mean or largest unit radius). That changes every Z'>1 latent. It is cheap only now, because no live run is Z'>1.
- If you don't answer: keep it.

**Q3. What is the reward per, at Z'>1, and should there be an intramolecular term?** This is needed before multi-compound training or joint conformer sampling, but not before stages 0-6.
- Today the reward uses `elj / z_prime`, i.e. per molecule, or per average unit for a cocrystal, at one temperature for every Z'.
- I recommend:
  - keep per molecule for one compound;
  - choose per formula unit when cocrystal work starts;
  - add a per-unit intramolecular energy to ELJ/exp6 before conformers are sampled;
  - use a fixed gas reference per compound for MLIP rewards instead of the per-call gas leg.

## 7. Additions from verification

- **The crystal search makes only identical-conformer Z'>1 rows.** It clones one molecule through the list constructor (crystal_search/coordinator.py:269; crystal_search/utils.py:445-529), and crystal_opt_utils.py:1200 asserts Z'=1 for the centroid-repulsion penalty. So a search-made prior cannot test the stage-6 fix.
- **crystal_fd refuses Z'>1 throughout** (canon.py status `z_prime_gt1`; charts, checks and dataset refuse or raise), so Z'>1 rows cannot be canonicalised or charted, and `data_processing/fd_canonical_prior.py` cannot make a Z'>1 prior. The fundamental-domain gauge for a set of units is an open design item. crystal_fd's `CrystalDescription` already carries per-unit poses with a `[B, Zmax]` `unit_mask` (description.py:79-137), a precedent for option (a)'s padded pose view.
- **A unit's frame needs at least 3 non-collinear heavy atoms.** `parameterize_pose` refuses such rows as `degenerate_axes` (crystal_steps.py:1564-1566); `align_mol_batch_to_standard_axes` uses heavy-atom principal axes (crystal_building/utils.py:1165); the radius is a heavy-atom radius, 0 for one-heavy-atom units (mol_methods.py:190-197). Water, ammonia, methanol, monatomic counter-ions and linear units have no orientation or handedness in this chart; a per-unit reach of 0 would make #16 worse. Needed before stage 7.
- **`collate_data_list` mutates its inputs when it widens them** (`d.aunit_centroid = F.pad(...)`, `d.max_z_prime = ...`; dataset_utils/utils.py:119-122): a Z'=1 element collated once beside a Z'=2 element stays widened in the caller's list, so pad values (#5) depend on collation history.
- **A GFN comment overstates the gauge.** molecular_crystal.py:512-517 says unit relabelling leaves the reward invariant; that holds for identical copies only. With per-unit conformers the canonicalisation moves atom blocks correctly for the build, but the raw-latent ordering penalty (1221-1260) and the stored trajectory's terminal state then disagree with the canonicalised x. Correct the comment in stage 6.
- **Smallest fix for the all-Z'=1 crash (#12):** `join_zp1_cluster_batch` asserts when no row has Z'>1 (crystal_building.py:195, 227); splitting a Z'=1 row is the identity, so the join can return the cluster unchanged. Dispatching on each row's `z_prime` is needed only for a Z'>1 row in a batch whose `max_z_prime` is 1.
- **Reward pad term (#33), re-derived:** the pad rotation vector (π/2, π/2, π/2) has r = 2.72 and θ = 0.955, so −2 log sin(r/2) − log sin θ = 0.045 + 0.203 = 0.248 T per pad slot: a fixed offset per Z' class that re-weights Z'=1 against Z'=2 rows by about exp(0.25) in a mixed-Z' run.
- **Out of scope but checked:** `crystal_building/image_pairs.py` raises for Z' ≠ 1 (64-65), so it is guarded.

**Verifier's verdict on the plan:** sound in order and scope (pin, refuse, unit map, batch levels, element layout, cluster reach, GFN for one compound, different compounds last); stages 0-2 can start as written, with the amendments applied above.
