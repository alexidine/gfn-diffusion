# Fundamental-domain crystals in the MXtalTools crystal object

**Status: PROPOSAL, 2026-10-20. Not adopted; no authority over code.** Companion to `fd_tooling.md` (the FD design and
decision record). Written in answer to the owner's nine points (B1–B9 below) from a read-only survey of MXtalTools and
GFN, with every "exists" claim re-checked against the code. `crystal_fd` is uncommitted, so file:line references will
drift; paths are under `MXtalTools/mxtaltools/` unless they say otherwise.

## Where things stand

| point | exists today | missing |
|---|---|---|
| B1 `is_fd` flag + canonicalise mixed batches | `ops.reduce_cell` (sg 1–15) then `domains.fold` (all groups, Z′ = 1); `domains.in_domain` / `in_asu`; a mixed-batch canonicaliser with per-row fallback only in a git-ignored build script (`artifacts/fd_design_oct15/canonical_data/csd_dataset/build_shards.py:121`) | a packaged canonicaliser with per-row status; one combined predicate; anything on `MolCrystalData` |
| B2 per-step functions on a batch | every step exists on `CrystalDescription`: import (`io_mxt`), setting (`settings`), sheet (`domains.to_sheet`), cell (`ops.reduce_cell`), pose (`fold`, `unfold`, `gauge_free_axes`), images, charts, measure, build (`dataset.to_mol_crystal_data`) | none is a method of the batch; both conversions go through Python lists |
| B3 crystal DoF → FD coordinates | `charts.CrystalChart` (forward, inverse, log-Jacobian), one space group per instance, sg 1–5 and 10–15; `PolytopeARChart` (pose, any group); `LatentV1Chart` (exact wrapper of today's latent) | a fixed-width transform for mixed space groups; sg 6–9 (no SO(3)/K chart); every cell family above monoclinic |
| B4 plotting | `plot_batch_cell_params` (`dataset_utils/data_class_methods/crystal_ops.py:2275`): real / latent / standard spaces, fixed 6 + 6Z′ panels, latent mode mutates the batch | any FD view |
| B5 where latent params are used | the latent family (`latent_to_cell_params` :480, `latent_transform` / inverse, `latent_params`, `latent_distmat`, the noise kicks, a disabled uniform sampler); bridges `LatentV1Chart`, `images(within='latent_v1_box')`, `measure.log_multiplicity` | FD counterparts of distance, noise, sampler and parameter read/write; a rule for picking one of a crystal's k trainer-box images |
| B6 ingestion (CIF, MolCrystalData, arrays) | MolCrystalData → FD: `io_mxt.from_mol_crystal_data` (per row, validating); arrays with pose parameters: `CrystalDescription.from_crystal_params` / `from_poses`; FD → MolCrystalData: `dataset.to_mol_crystal_data` (returns a list); a CCDC-free CIF reader used only by tests | a batched `MolCrystalData` constructor; raw xyz (no pose parameters) → FD; CIF → FD; a packed molecule type (molecules are a list of `MoleculeRef`) |
| B7 clip / reject non-FD samples | partial: valid but non-canonical samples are already fixed exactly (reduce, fold, `to_sheet`); membership tests and `LatentV1Chart`'s clamp mask exist | a packaged clip-or-reject step for genuinely out-of-support samples, with per-row masks |
| B8 uniform sampler over FD | `charts.from_unit` (uniform in chart coordinates); `so3.random_rotations` (Haar); measure-correct samplers only in the tests (`tests/fd/_helpers.py` `physical_cells`, `chart_draws`) | a packaged sampler; a monoclinic cell sampler |
| B9 periodicity, seams | every facet's gluing in all 230 groups (`GroupTable.facets`: translation, self-twisted, pair-twisted, free axis); circle detection in the charts; `fold` / `unfold` | public periodicity metadata for mixed batches; a seam-crossing helper; a step policy (periodise vs re-standardise); about six independent wrap/snap one-offs, none reading the gluing table |

Also present: `get_fd_params` / `fold_to_fd` in `crystal_ops.py:1472, :1615`, an earlier single-space-group fold with
no live callers, to retire once the new path lands (`docs/wiki/asymmetric-unit-reduction.md` names them).

## Principles

- **Additive.** The functions `fd_tooling.md` lists as bit-identical (the latent maps, `latent_params`,
  `canonicalize_free_axes`, `canonicalize_zp_aunits`, the builder, `analyze`, …) are not edited. Every FD feature is a
  new function or method, and no live default changes.
- **Always batched.** New APIs take a collated `MolCrystalData` batch or a `CrystalDescription`, never lists. Mixed
  space groups are grouped internally (as `fold` does); mixed Z′, unsupported groups and bad rows get a per-row
  status instead of a whole-batch raise (today `reduce_cell` raises for the whole batch on any sg ≥ 16, on a
  monoclinic α or γ off 90°, or when its search leaves a row unreduced).
- **No hidden choices.** Required arguments with no default: `cell_measure` wherever a density, Jacobian or
  sampler appears (O4: answered (a) on 2026-10-15; the owner asked on 2026-10-20 that options be compared, not
  presupposed); the orientation chart (O3 deferred; `hemisphere`, `euler_zyz` or `axis_angle` for the 12-slot layout,
  since `quaternion` has 4 coordinates); the length box; the seam policy; the out-of-support policy.
- **Two kinds of "not FD".** A valid crystal that is not canonical is fixed exactly (reduce, fold, gauge,
  `to_sheet`), never clipped. Only a sample outside the support (no real cell, lengths outside the box, Z′ > 1, …) is
  clipped or rejected, and every such row is masked.
- **Measure.** The pose fold preserves Haar × V d³t. Re-standardising the cell is exact as a crystal identity but
  carries a Jacobian unless the cell measure is basis-invariant. Sampling cells in a box and then reducing them is
  biased under any measure (each lattice is weighted by how many of its bases fall in the box): sample on FD ∩ box
  directly.
- **Every loop is bounded.** Rejection and retry take a required `max_rounds` and return a `filled` mask (or raise
  with the count). Existing defect: `sample_random_reduced_crystal_parameters` loops `while not valid.all()` with no
  bound (`crystal_ops.py:449`).
- **Canonical is not duplicate-free.** Special metrics, rows within rounding of a face, orthorhombic axis orders (until
  stage 2) and frame flips of near-symmetric molecules leave one crystal with more than one canonical form.
  `is_fd`, distances and deduplication carry `special_metric` and `near_face`, and never claim identity.
- **Handedness stays behind one hook.** `apply_sheet_policy(desc, mols, policy)`; `canonicalize` never reads
  `reference_handedness`. Mirroring a whole Sohncke crystal gives its enantiomorph, a different structure, and in the
  22 groups of the 11 enantiomorphic pairs (P3₁/P3₂, P4₁/P4₃, P6₁/P6₅, …) a different space-group number. The owner
  chose it for the canonical CSD set on 2026-10-21 (`fd_tooling.md` D2); it is an explicit `off_sheet` choice, not an
  importer default.
- **Imports.** The object imports `crystal_fd` inside its methods: `tables.py` hash-checks the FD table at import and
  can raise `TableMismatchError`, which every `MolCrystalData` import should not inherit.
- **dtype and device.** Group arithmetic is float64, batches and energies float32; tolerances follow the dtype.
  `reduce_cell` (monoclinic setting shifts) and `io_mxt` have numpy, CPU-only parts; no FD test touches CUDA yet.

## Per point

### B1 `is_fd` and canonicalising mixed batches
**Built 2026-10-20** (`crystal_fd/canon.py`, `tests/fd/test_canon.py`; README "API in brief"); the code is the
reference, and it departs from the sketch below in: statuses `cell_off_group` (sg ≥ 16 cells the group does not
preserve) and `sheet_failed`; a separate `near_tol`; `moved_tol` by dtype; no pre-filter in `fd_status`. Not built
yet: the `MolCrystalData` methods (B2).
- `crystal_fd/canon.py`: `canonicalize(desc, *, sheet, tol, face_tol) -> Canonicalization`, promoted from
  `build_shards.canonicalise`. Partitions rows first (non-finite, unreal cell, Z′ > 1, monoclinic angle off 90°),
  then `reduce_cell` on sg ≤ 15, `fold` on Z′ = 1, optional `to_sheet`; catches a per-chunk `RuntimeError`. Status
  per row: `canonical`, `special_metric`, `pose_only` (sg ≥ 16: pose folded, conventional cell kept), `off_sheet`,
  `mono_angle_off`, `z_prime_gt1`, `unreal_cell`, `non_finite`, `reduce_failed`. Returns the reduction and fold
  records, `near_face` and `moved`.
- `fd_status(desc) -> FDStatus` (`is_fd`, `decided`, `cell_reduced`, `pose_folded`, `on_sheet`, `special_metric`,
  `near_face`, `zp1`): a computed predicate, cheap pre-filter first (`tri_niggli_conditions` / `mono_conditions`,
  domain slack), full check only for rows that pass.
- `is_fd` is the computed predicate. If a stored graph flag is wanted for speed, only the `fd_*` methods write it and
  any other write leaves it stale (documented). `set_cell_parameters` is not edited: it runs on every GFN reward and
  `LatentV1Chart` borrows it (gate 8 bit-identity).
- Tests: per-row statuses on a mixed batch (sg 1, 2, 14, 19, Z′ = 2, an unreal cell, a monoclinic row with α off 90°)
  in one call; `fd_status` agrees with `canonicalize(desc).desc == desc`.

### B2 per-step methods on the batch
- Mixin `MolCrystalFDOps` (`dataset_utils/data_class_methods/crystal_fd_ops.py`), in place like `set_cell_parameters`:
  `fd_describe(*, std_orientation, frame='body'|'literal')` (the build convention is an argument: body-mode rows
  already carry h in `pos`, literal rows are consistent only with `std_orientation=True`), `fd_assign(desc)` (wrap
  centroids, check the builder clip band, write parameters through `set_cell_parameters`, drop cached geometry),
  `fd_reduce_cell`, `fd_fold`, `fd_gauge`, `fd_to_sheet(policy)`, `fd_canonicalize`, `fd_images(...) -> batch`.
  Rows a step cannot handle are left bit-identical and reported.
- Tests: each method against its `crystal_fd` function on the same rows; built unit cells before and after equal
  (needs a `MoleculeBatch`-aware `checks.same_crystal`; today it takes one `MoleculeRef` and equal cells).

### B3 the FD parameter transform
- `crystal_fd/fd_params.py`: `FDSpec(length_box, orientation)` (both required); `fd_transform(desc, spec,
  canonical='require'|'canonicalize') -> FDParams` with `y [B,12]` and per-slot `live`, `kind` (interval, circle,
  fixed), `period`, plus `logabsdet [B]`; `fd_inverse_transform(y, sg, spec)`. Slots: cell 0–5 per family
  (triclinic six; monoclinic four plus two fixed), centroid 6–8 by axis (free axes fixed at the gauge), orientation
  9–11. This fixes cut A of O1, which is still open.
- sg 6–9: opt-in double cover over SO(3), reported as a multiplicity log 2. sg ≥ 16: `NotImplementedError` naming the
  family until its cell chart exists; `fd_pose_transform` (pose only) for every group, gated so far for sg 1, 2, 14,
  19, 92 only.
- Tests: round trips per family (gate 3 pattern); `logabsdet` against autograd (gate 5); mixed-sg batches equal the
  per-group charts slot by slot.

### B4 plotting
- New `plot_batch_fd_params(space='fd_unit'|'fd_chart'|'cell', *, spec, ref, aux, quantiles, ...)`;
  `plot_batch_cell_params` stays as GFN calls it (`eval/evaluations.py:333`). Never mutates the batch.
- `fd_unit`, the space-group-agnostic view: each coordinate as its position in the domain, u = (y − lo)/(hi − lo)
  in [0, 1] from the chart's autoregressive bounds; circles marked; fixed slots dropped; generic titles (cell u1–u6,
  t_x/t_y/t_z, R u1–u3). sg ≥ 16 rows have no cell slots there; the `cell` view (reduced a, b, c, α, β, γ, volume
  per molecule, packing) covers every row.
- Also in scope later: `plot_batch_staircase` (latent mode mutates the batch) and the fixed-layout plots in
  `reporting/figures.py`.

### B5 FD counterparts of the latent family
- `fd_to_cell_params(y, spec, on_outside)` (with a mask, never a silent clamp), `fd_params(spec)` (non-mutating),
  `crystal_fd.metrics.fd_distance(a, b, spec)` (chart distance; minimum image on circle slots; SO(3) geodesic for
  orientation; inf across space groups; optional images of near-face rows), `fd_noise(desc, scale, spec, *, seams)`
  (small rotation kicks as `rotvec_to_matrix(scale · randn) @ R`, not `random_rotations`, which is Haar).
- `to_latent_v1(desc, radius, *, choose='first'|'random'|'all')`: `representable` = an in-box image exists **and** it
  is not moved by the live clamp (D7's refusal rule); `choose='random'` refused where `k_box` is not constant or box
  coverage is below 1.
- Open: whether the CSD z-score "standard" basis (`zp1_std_cell_parameters`, `standardize_*` / `destandardize_*`,
  feeding noise, embeddings and the "standard" plot space) needs an FD counterpart.

### B6 ingestion
**User-facing shape, agreed 2026-10-20 (owner: "On the I/O I think I basically agree"), modelled on ASE:**
- **Simple level.** One reader entry point, `MolCrystalData.read(cifs, format=..., **format_options)`, and
  `MolCrystalData.from_arrays(cell, xyz, z, sg=..., **attrs)`: a bare batch with the minimal attributes plus whatever
  the caller attaches; no molecule finding, no canonicalising. Cheap enough for training-time use.
- **Full level.** The featurizer's work as explicit, separately callable steps on that batch: find molecules and Z′,
  sanitize and protonate, parameterize the pose, check the unit-cell rebuild, canonicalize to the FD form, attach
  node- and molecule-level features. A convenience call (e.g. `read(..., full=True)`) runs the standard chain; it is
  not the definition of the object.
- **Differences from ASE.** Batched, not lists of single objects; a bad structure gets a status and a message, not
  an exception (as `canon.canonicalize` does). Choices (handedness representation and near-face nudge, both decided
  2026-10-21; dedup and protonation rules, open) stay explicit step arguments, so a step can be re-run on a stored batch without
  re-parsing CIFs.

The detailed contract (today's pipeline and its defects, the reader, the steps, the store, consumers, migration and
the owner questions) is `crystal_ingestion.md` (2026-10-20). The sketch below predates both and is kept for its
inventory of existing pieces.
- `MoleculeBatch` (packed `pos`, `z`, `ptr`, hashes: the layout `dataset.pack_molecules` already writes) replaces
  lists of `MoleculeRef` in new APIs.
- `MolCrystalData.from_tensors(pos, z, ptr, sg, cell_lengths, cell_angles, aunit_centroid, aunit_orientation,
  aunit_handedness, z_prime, ...)`: builds the collated batch directly, following GFN's `init_blank_crystal_batch`
  (`energies/molecular_crystal.py:1478`); computes radius, mass and molecular volume before `box_analysis`, which
  needs them; must match `collate_data_list` attribute for attribute.
- `crystal_fd.convert.to_mol_crystal_batch(desc, mols, *, mode, clip)`; `dataset.to_mol_crystal_data` becomes a shim
  until lists retire.
- Dropped (owner answer 3): a direct CIF / raw-xyz → FD path (`parameterize`, `from_unit_cell`, `from_cif`). Raw
  xyz and CIFs go through a `MolCrystalData` batch and the existing pose recovery
  (`reparameterize_unit_cell`), then the importer and `canonicalize`.
- `io_mxt` stays the validating importer; taking collated batches without per-row loops means vectorising
  `_row_fields`, `_row_checks` and `_batched` (`io_mxt.py:442`), not just opening a path.

### B7 clip or reject
- `crystal_fd/support.py`: `to_fd(desc, *, spec, out_of_support='clip'|'reject', sheet_policy) -> (desc,
  FDProjection)` with per-row `reduced`, `folded`, `sheet_moved`, `clipped [B,6]`, `rejected`, `reason`.
  `out_of_support` is required. Exact fixes first; then, for lengths outside the box, clip in the cell chart's
  autoregressive order (log c, then r₂, then r₁ given the clipped prefix), which always lands in FD ∩ box; monoclinic
  α or γ off 90° projected (the crystal-system constraint) or rejected; non-finite or unreal cells rejected.
- `chart_to_fd(y, sg, spec, *, on_outside)` for raw sampler output in chart coordinates (circle slots wrapped exactly
  first); `resample_rejected(draw_fn, n, *, max_rounds)`.
- Clipped rows are point masses on faces: always masked out of densities.

### B8 uniform sampling
- `crystal_fd/sampling.py`: `sample_fd(sg, n, *, length_box, cell_measure, mode='reference'|'chart', orientation,
  generator, max_rounds) -> (desc, log_density, filled)`.
  - Pose, every group: t uniform in the unit cell and R Haar, then fold. Exact, no rejection (the gate 2 pattern,
    `test_gate2_measure.physical_canonical`).
  - Cell, sg 1–2: port `physical_cells` with its analytic acceptance.
  - Cell, monoclinic, under `latent_v1_lebesgue`: either β uniform on [π/2, arccos(−m_max)] accepted with probability
    sin β · 1[cos β ≥ −m(a, c)], or s uniform given (a, c) accepted with probability m(a, c)/m_max — one of the two,
    not their product; verified by monoclinic gate 2 histograms before use. `metric_invariant` stays unimplemented
    for monoclinic (D5 withdrawn).
  - `mode='chart'`: uniform in chart coordinates, the analogue of the latent's uniform box.
  - sg ≥ 16 refused, naming the family.
- Add `max_rounds` to `sample_random_reduced_crystal_parameters` without changing its distribution.

### B9 periodicity and seams
- `crystal_fd/seams.py`: `seam_table(sg)` (tensorised facets: normal, offset, element, shift, kind);
  `periodic_slots(sg, spec)` for the B3 layout; `wrap_circles(y, circle, period)`; `restandardize(desc, *,
  cell_measure)` (fold and reduce only rows that left the domain; returns the elements used and `log_jac`: 0 for the
  pose, nonzero for the cell unless the measure is invariant); `so3.hemisphere_glue` (the equator gluing, today only
  in a docstring).
- Step helpers take a required `seams='wrap'|'restandardize'|'unfolded'`:
  - `wrap` (periodise inside steps) is exact only across translation seams and orientation circles; twisted seams
    and cell faces stay walls, so it is combined with `restandardize` for those rows. Kernel densities need an image
    sum.
  - `restandardize` (re-standardise when a trajectory leaves the zone) is exact for every gluing kind; it costs a fold
    per exit and a reduction per cell exit, jumps in chart coordinates, and needs the cell `log_jac`.
  - `unfolded` keeps the kernel smooth but canonicalises at every evaluation.
- Consolidation is opt-in. The one-offs (`wrap_centroid`, `canonicalize_aunit`, `_snap_edge_sliver`,
  `normalizer_images`, `dataset.wrap_centroids`, …) differ in seam representative, tolerance and gradient behaviour
  (`wrap_centroid` is a straight-through clamp the live search relies on), and `fd_tooling.md` keeps
  `normalizer_images` unchanged; unifying them changes live search and GFN behaviour, so it needs an owner decision
  and the wiki update in the same change.
- `latent_v1_periodic_slots(sg)` (public `LatentV1Chart._dims`) is an FD-side descriptor only. It disagrees with
  GFN's policy angle mask on r and on the sg 2 x row; a GFN boundary test should compare the two, changing neither.
- A second cell canonicaliser exists: `crystal_search.standardize.standardize_cells` / `choose_basis` (sg 1–15), with
  the input-basis tie-break defect `fd_tooling.md` records, called by `build_qm9_full_prior.py` and
  `eval/cond_panel/csp.py`. Proposal: an opt-in delegation to `reduce_cell`.

## Order
1. B1 `canonicalize` + `fd_status` (S).
2. B6 part 1: `MoleculeBatch`, `MolCrystalData.from_tensors`, `to_mol_crystal_batch` (M).
3. B2 mixin (M).
4. B8 pose sampler for every group and the triclinic cell sampler; `max_rounds` on the old sampler (S–M).
5. B7 `to_fd`, `chart_to_fd`, `resample_rejected` (M).
6. B3 `fd_transform` for sg 1–5, 10–15; opt-in double cover for 6–9 (M).
7. B4 `plot_batch_fd_params` (M).
8. B9 part 1: `seams.py` and the required seam argument (M).
9. B5 distance, noise, `to_latent_v1` (M).
10. B8 monoclinic cell sampler and monoclinic gate 2 (M).
11. B6 part 2: `apply_sheet_policy`; the featurizer rewrite emits batches; vectorise the importer for on-the-fly use (M).
12. B9 part 2: opt-in consolidation of the one-offs; retire `get_fd_params` / `fold_to_fd` with the wiki update (S–M).

Later, outside these nine points: cell charts for sg ≥ 16 (orthorhombic is stage 2), Z′ > 1 (stage 4).

Performance: `fold` has gate 7 budgets; `reduce_cell` has none beyond a `triclinic_basis` floor. Measured 2026-10-20
(2 CPU threads, shared machine; `artifacts/fd_design_oct15/walkthrough/check/step6/speed/`): reduce + fold about
1,100 rows/s for P-1 from random bases and about 28,000 rows/s for P2₁/c. Add `reduce_cell`, sampler and
`restandardize` budgets to `bench_fd.py` before any per-step use.

## Owner answers, 2026-10-20
1. **Where the API lives:** methods mixed into `MolCrystalData`; not every function has to live there, so the core
   stays as `crystal_fd` functions on `CrystalDescription`.
2. **`is_fd`:** a computed check only. No stored flag.
3. **CIF and array ingestion:** the owner is agnostic and prefers reusing existing machinery. So the route is the
   one that built the CSD set: instantiate a `MolCrystalData` batch (featurizer, or `from_tensors` for arrays, with
   `reparameterize_unit_cell` / `parameterize_crystal_batch` recovering the pose from coordinates), then standardise
   it (`io_mxt` import, then `canonicalize`). No separate CIF → FD path; B6's `parameterize`, `from_unit_cell` and
   `from_cif` are dropped. The featurizer rewrite only needs to emit a batch. Caveat: the importer reads rows one by
   one (the full CSD import ran at about 125 rows/s with all checks), fine for dataset builds, to be vectorised before
   any on-the-fly use.

Implementation defaults (reversible, not owner decisions):
- **Mixed-group layout:** a fixed 12-column tensor with per-row masks of live and periodic columns, as the latent
  does. The column meanings assume O1's default, the domain cut in the centroid (cut A).
- **Leaving the domain (owner, 2026-10-20: "it is context dependent so we actually want to be careful about
  assuming here"):** `seams` and `out_of_support` are required arguments with no default on every helper that moves
  or projects crystals, so each call site states its choice. Every call returns a mask of the rows it wrapped,
  re-standardised, clipped or rejected, so clipped rows can be kept out of densities.

## Drift found in existing documents
- `crystal_fd/README.md`: "nothing outside this package and `tests/fd/` consumes it" and "no prior migration" are
  stale (`data_processing/fd_canonical_prior.py`); "no GFN adapter" still holds.
- `fd_tooling.md` puts the free-axis projection and K in stage 2; `fold` already does both for all 230 groups. It
  lists `images(within='g_plus_asu')`, which the code does not accept.
- `docs/wiki/asymmetric-unit-reduction.md` names `get_fd_params` and `fold_to_fd`.
