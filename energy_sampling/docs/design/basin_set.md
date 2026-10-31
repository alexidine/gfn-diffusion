# Basin sets: search output, prior input

*Plan, 2026-10-30. Status: the direction is the owner's (Decisions below); every section after it elaborates that
direction and is a proposal, checked against both repositories on 2026-10-30 by two reviewers and to be checked again
as each piece lands. Not policy.*

*Built (2026-10-30): the guard that refuses a non-latent chart in `build_noised_prior.py` (`refuse_unscored_chart`);
the writer and reader `mxtaltools/crystal_search/basin_set.py` (`write_basin_set`, `read_basin_set`, `live_mask`;
committed locally in MXtalTools); `build_noised_prior.py --basins` (anchors only or with iso children, basin tags on
every row, off-chart basins dropped and counted); `noised_prior.live_kernel: 'basin'` on both charts with a rolling
bank (`train.py::Modeller._basin_kernel_table`, `._basin_fresh`, `._basin_kernel_draw`; the latent chart through
`energies/molecular_crystal.py::latent_rows_from_params`, composed of crystal_fd's canonicaliser, trainer-box images
and `LatentV1Chart.latents`); the kernel shared by trainer and builder (`noised_prior.py::basin_factors`,
`::basin_draw`) with the shape `cfg:noised_prior.live_shape` (`'hessian'` or `'iso'`, the same expected excess);
children drawn by the basin kernel (`build_noised_prior.py --child_kernel basin`, `basin_kick`), windowed by their
basin (`noised_prior.py::window_energy`); readouts on kernel draws (`train.py::Modeller._kernel_readouts`). Not
built: the
`log_mass_laplace` column, the conditions file, the molecule coordinate match, the flat-molecule twin map (section
2), and `FORMAT_VERSION` 2 (the basin tags are optional row attributes of version 1).*

## Why

The owner's workflow (2026-10-30), in two layers:

1. **Support discovery.** Classical crystal search is the only engine of mode discovery and must cover the modes. The
   streaming search (MXtalTools `crystal_search/stream_search.py` with the `minima` block) takes states or nothing as
   input and returns putative basin minima with the local harmonic curvature of the physical energy; most optimisation
   states are discarded. Transfer to another energy function is re-polish, re-score and extra starts. With the output
   format committed, one prior constructor always takes it; in the Laplace case the search procedure is, in effect,
   prior construction for a given temperature and physical energy.
2. **Tools.** Anchors, thermal ellipsoids and local noising are tools used downstream in training and analysis.

A read-only audit of both repositories (2026-10-30) found layer 1 built but unconnected to layer 2, and layer 2
crowded: seven noise kernels, four notions of a Hessian, about eight incompatible notions of "the same basin", three
stores. This plan connects the layers through one format, one writer, one constructor and one draw function, and
retires the duplicates once the new path works.

## Decisions (owner, 2026-10-30)

- The audit's proposal is the path: one basin-set format written by the search, one constructor that reads it, the
  duplicates retired afterwards ("I agree very much with this proposal").
- Search of record: the streaming search. MLIP targets "will fold into the streaming approach in the near future";
  until then the batch loop's minima hook (`run_search.py::_search_loop`) stays.
- Rows per basin: equal, with the Laplace log-mass stored as a column and not used for allocation.
- Harvested rollout states are not a constructor input ("I don't plan to use it").
- The sheet choice "should be flagged very explicitly in the system design. E.g., for flat (or effectively flat)
  molecules vs chiral, we choose whether to search both sheets".
- Energy currency: "as long as it's consistent within a problem it's fine. eLJ is just for prototyping anyway."
- MXtalTools' local and origin masters are merged (not rebased) and pushed.
- MLE draws need no scoring: "in MLE, with iso or hessian noise, we don't have to re-score the samples."
- Hessians are precomputed by a standalone step, not by the trainer at load ("It should be pre-computed with a
  standalone script").
- The owner reviews the training protocol eyes-on before production.

This page supersedes these lines of `prior_dataset_build.md` (Decisions, 2026-10-27): "Hessian-shaped tiles are set
aside until their tooling exists"; "No in-run noising ... for stage 1" (section 4 draws during MLE); `iso` as the noise
kernel (it becomes the fallback for a basin without a Hessian); "Thin very close duplicates" (identity is the
search's); growth "by harvested states" and the harvest-retention rule. Its other decisions stand.

## 1. The basin set

A basin set is a `crystal_fd.dataset/1` folder (MXtalTools `crystal_fd/dataset.py`), not a new format: one row per
distinct basin, each a Z' = 1 `CrystalDescription` with its reference molecule (`MoleculeRef`, kept with its
`reference_handedness`), identifier, file hashes and code digest. The format is Z' = 1 only today.

**One writer, two callers.** A converter beside `minima.py` turns a search's output rows into a basin-set folder: the
rows `minima.minimum_rows` selects (library minima: `minimum_id >= 0`, not a recapture). The streaming search calls it
at the end of a run and on a stop request; anyone can run it on a partial `<run>.pt` or on the batch loop's output
(which writes no per-minimum file), which is how MLIP basin sets are produced until MLIP folds into the stream.
`save_dataset` never overwrites a folder, so there is no per-group rewrite.

**Header (the format's `provenance`).**
- `energy`: target; an energy-model id formed by one function used on both sides (today
  `coordinator.py::energy_model_id` gives `'<target>:<file>:<bytes>:<sha1 of the first and last MiB>'` for an MLIP and
  bare `'elj'` for eLJ, and GFN's `noised_prior.identity` records only the MLIP file name; the writer adds the lj scale
  for eLJ and the two sides converge on one id); units; per asymmetric unit (eLJ) or per molecule (MLIP); every
  `StageEnergy` analyze setting (cutoff, supercell size, repulsion, margin, surface padding, the envelope,
  `std_orientation`).
- `search`: config path and hash; the search `kT` with units; the stop rule and each condition's stop record;
  cumulative effort (energy calls by stage, starts, seeds) across resumes; the seed source when the run is a transfer.
  The search `kT` is more than a band unit: the polish and BFGS tolerances, `merge_d2`, `capture_d2` and the glued
  merge window are all in units of it, so two sets searched at different `kT` resolve basins differently.
- `identity`: `merge_d2` and `capture_d2` with the `kT` they are measured in; the glued-FD merge (`FD_MERGE_A` = 0.05
  Angstrom within `FD_MERGE_KT` = 0.01 E/kT, Z' = 1, module constants of `minima.py`); the key (identifier, space group,
  Z', handedness signs in the identity description).
- `sheets`: section 2.
- Commits and dirty flags of MXtalTools (and GFN when the writer runs from there), host, time.

**Columns, one entry per basin.** `basin_id`; `p` [12] float64, the identity-description parameters `id_params` (a, b,
c in Angstrom; alpha, beta, gamma in radians; the fractional centre; the rotation vector in radians); `E`, the physical
energy at the hard cutoff in header units, raw (no temperature, no lj scale; the record currency, never -T log R);
`E_env`, the enveloped energy the Hessian is taken of; `H` [12, 12], optional, the Hessian of `E_env` at `p` in header
units per coordinate unit squared, i.e. the search's `id_hessian` (E/kT, float32) times the search `kT`; `live_mask`
[12]; `min_eig`; `hits` and `seed_hits`; `polish_status` (the search's `POLISH_STATUS` codes).

- `status` keeps its crystal_fd meaning, the canonicalisation status (`canon.STATUS`), which `load_dataset` and
  `FDCoordinates` read. The writer re-canonicalises each row and records it; it does not assume the search's identity
  description is canonical (`identity_description` discards `fd_fold_aunit`'s refold status).
- The writer checks that the stored description's `crystal_params()` round-trips `p` (rotation matrix to rotation
  vector can land on the antipode near |r| = pi) and refuses a row that does not.
- `live_mask` comes from one MXtalTools helper used by both the writer and the constructor: free centroid axes from
  the crystal_fd tables, fixed angles from the lattice system as `clean_cell_parameters` enforces them. Not a fourth
  notion beside `models/dead_latent_rows.py`, `CONTINUOUS_DIMS` and the FD chart's dimensions.
- Today the writer has only converged minima to write: `newton_polish` stops at a positive-semidefinite Hessian and
  `finish_minima` gives a `minimum_id` only to those. The owner wants low-energy saddles not covered by another
  ellipsoid kept (2026-10-30); that needs a producer: `finish_minima` keeping converged stationary points with a
  negative mode, flagged by `min_eig`.

Identity at reduced-cell faces: the glued-FD merge in `minima.py` merges a Z' = 1 seam pair at polish time, but it lives
in uncommitted `minima.py` with the untracked `crystal_fd/metric.py` (2026-10-30), capture still uses the ellipsoid
alone, and Z' > 1 is never glued.

## 2. Sheets

Owner (2026-10-30): the default is "FD with both sheets"; the sheet question matters "mostly to comparisons between
latent and FD parameterizations, or specific studies of symmetric molecules"; in Sohncke groups "the two sheets are
identical for our purposes".

- In every group with an improper element the FD chart, fixed to handedness +1, covers both sheets' crystals once:
  its centroid domain is the fundamental domain of the handedness-preserving part of the group (crystal_fd
  `fd_tables_v1.json`, `gamma_plus`), so a description on -1 is moved onto +1 by an improper element and kept. This
  is the default and needs no setting.
- Sohncke groups: a -1 row is stored as its mirror crystal (`io_mxt` `off_sheet='enantiomorph'`, owner 2026-10-21;
  energies are mirror-invariant). The writer uses this mode.
- The latent chart is a comparison chart with stated limits: in the groups whose trainer box has coverage below 1
  (sg 14, 8-9 and others, `fd_tables_v1.json` `trainer_box.coverage`) it reaches half of a chiral molecule's
  crystals. MIPCAS (sg 2) is covered fully by both charts.
- A flat or effectively flat molecule's crystal has two FD points (the pose and its mirror-equivalent through the
  molecule's own symmetry). A prior that puts each crystal at one of them leaves the other half of the domain without
  data. The owner's fixes (2026-10-30): duplicate each row at its twin, or use only half the chart. Both need the twin
  map from the molecule's point symmetry, which is not built (crystal_fd leaves molecular point symmetry out, owner
  2026-10-26). Until it exists, a flat molecule (NEHZOR) on the FD chart carries this gap.

## 3. The constructor

One constructor: `data_processing/build_noised_prior.py`, extended with `--basins PATH[,PATH...]` in place of
`--source`. It shares the existing steps (scoring, seed energy, held-out tagging, collation, manifest, atomic write)
and writes the pre-noised format, bumped to `FORMAT_VERSION` 2 with the new tags in `noised_prior.TAGS`.
`_load_noised_prior` stays the only reader of the manifest, tags and splits (the conditions loader and a file-valued
`anchor_buffer.seed_source` read the rows without them). With `--basins`, `--harvest`, `--anchors` and `--log_hi auto`
are refused; `--images` applies on the latent chart.

1. **Validate.** For each set: the header's target, energy-model id, cutoff, supercell size, repulsion and margin equal
   the run's; `std_orientation` and the envelope are excluded (the search builds with `std_orientation=True` and an
   envelope, the trainer with neither; step 3's re-score covers both). Each molecule matches the run's by identifier,
   atomic numbers and aligned coordinates within a stated tolerance (`molecule_hash` is exact to 1e-6 Angstrom and
   cannot match float32 conditions files). The sheet policy fits the chart. On an energy mismatch the refusal names the
   transfer route: a seeded streaming search under the run's energy, whose seeds can be a basin-set folder
   (`FDDataset.to_mol_crystal_data(mode='body')`; the stream does not read a folder today).
2. **Convert currency** with one function exported from `energies/molecular_crystal.py`, the code `MolecularCrystal`
   itself calls (lj scale, `lj_rescale`, per asymmetric unit or per molecule), applied to `E`, `E_env` and `H`; the
   factor goes in the manifest.
3. **Anchors.** Build each basin's crystal from its stored description; re-describe it into the run's chart (FD: one
   canonical point; latent: one of its valid trainer-box images, drawn uniformly, per the owner rule of 2026-09-28,
   noise first and then the normaliser images); score it with the run's analysis and compare with the converted `E`,
   refusing past a tolerance on more than a set share (frame drift, code drift, and the envelope, which leaves the
   minimum slightly non-stationary under the hard cutoff). Rows with no chart point are dropped and counted, never
   clipped. Each anchor row carries `basin_id`, `basin_p`, `basin_H` and `basin_live`: draws are centred at `basin_p`,
   never at the row's chart point (which may be another image or setting).
4. **Windows.** Every scored basin is stored. The energy and packing windows stay load-time settings
   (`noised_prior.select_rows` on seed energy, `noised_prior.window_kT`); the build reports its own view with the same
   seed-energy floor and adds no second floor.
5. **Children** (optional, for stage-2 backward data): K per basin from the draw function of section 4, scored.
6. **Hold out** whole basins per condition, never a condition's lowest.
7. **Record** per basin `log_mass_laplace` = -E/kT - 1/2 log det(H_live / (2 pi kT)) + log mu(p), mu the reference
   measure's density in the 12 parameters of `p`: crystal_fd `measure` with a named `cell_measure`
   (`latent_v1_lebesgue`) and the Jacobian from that measure's coordinates (log lengths, free angles) to `p`. For
   analysis and reweighting, not for allocating rows.
8. **Write** the prior, a conditions file (one carrier per condition, so `molecules_path` need not be the prior), and
   a manifest naming every input set (path, hash, header) and the counts.

Never: search, polish, invent or deduplicate basins; clip a row into a box.

A build of anchors only is trained with `noised_prior.train_kinds` including `'anchor'` (mk_dev's `['child']` would
leave the loader nothing).

## 4. One draw function, for builds and for MLE

One function (proposed `noised_prior.basin_draw(p, H, live, tau, caps)`) draws around a basin, used both by the
constructor's children and by the MLE draw:

- x = p + L z in the 12 parameters, L L^T = tau kT (H restricted to `live`)^-1 with |lambda| per mode (a saddle keeps
  its unstable directions) and per-block width caps (Angstrom, radian, fraction);
- build the crystal and re-describe it into the run's chart (FD: canonicalise; latent: canonicalise, then a uniform
  trainer-box image). Faces, origin shifts, the rotation-vector seam and the sheet are folded; nothing is reflected or
  clamped;
- a draw with no chart point is redrawn at most a set number of times, then dropped for the batch and counted per
  basin; a basin whose miss share passes a bar is reported; a basin whose centre has no chart point is excluded at
  build;
- a basin without `H` uses the same function with an isotropic per-block stiffness (widths stated in Angstrom,
  radian and fraction) in place of `H`.

The MLE loss evaluates no energy (`train.py::Modeller.draw_bwd_sample`, dataset branch), so a draw needs no energy
call; it does need a crystal build and a re-description per row per batch, whose cost step 6 measures. In the trainer
this is a new `noised_prior.live_kernel` value (proposed `'basin'`), reading the anchor rows' `basin_*` tags. The
2026-10-30 prototype values `'hessian'` (the Hessian of -log R at stored rows, taken by the trainer at load, reflected
at faces) and its `'iso'` refuse loudly once `'basin'` lands.

Working assumption (scope: stage-1 MLE; revisit near |r| = pi and the orientation poles): MLE fits the draw
distribution, the Laplace Gaussian in the 12 parameters pushed into the chart, so the configuration measure mu enters
the MLE target only through its variation across an ellipsoid, assumed small. The trainer's TB reward carries mu in
full. The owner's concern (2026-10-30): "I'm inclined to use physical hessians though I see that in the FD chart that
may distort the volume by not respecting certain jacobian terms".

## 5. What a chart must provide

- A canonicaliser with a status: any built crystal to (chart point, ok, status). FD: exists
  (`energies/fd_coordinates.py::FDCoordinates.from_params` over crystal_fd `canon.canonicalize`), for one space group
  and Z' = 1 per run. Latent: composed from existing parts, `canon.canonicalize` (+1 sheet), trainer-box images
  (`domains.images` restricted to the trainer box), `LatentV1Chart.latents` (unclipped), ok = inside [-1, 1] on live
  rows, returned in the same form; not a new canonicaliser.
- A differentiable decode and the reference density, for the trainer's reward. Both exist.
- Coverage: every basin and nearly every draw has a chart point. FD: its length box is narrower than the search's cell
  bounds, so it is widened or misses are reported with their mass. Latent: half of space group 14 (section 2).

## 6. Order of work

| step | where | who | proof | page updated in the same change |
|---|---|---|---|---|
| 1. Merge MXtalTools origin into local master and push (details below) | MXtalTools | the owner, or the session that owns the uncommitted search edits | `tests/fd`, `tests/test_search_minima.py`, `tests/test_stream_search.py`, `test_coordinator.py` on CPU | - |
| 2. Refuse the broken FD build path | GFN | done | `tests/crystal/test_noised_prior.py` | - |
| 3. Basin-set writer (section 1) and the live-mask helper | MXtalTools | the search's owner | a test converting run rows to a folder and reading it back (`p`, `H`, status, header) | `crystal-search-method.md` |
| 4. Constructor reads basin sets (section 3) | GFN | | a basin anchor rebuilt and scored matches header `E` at lj scale != 1; identity, molecule and sheet refusals; no held-out basin in training | `prior-buffer.md` (A pre-noised prior) |
| 5. One draw function and `live_kernel: 'basin'` (section 4); latent composed canonicaliser | GFN | | draws stay on chart points; the fold and image rule; the redraw bound; matched shapes on FD and latent | `mle-phase.md`; `asymmetric-unit-reduction.md` (images) |
| 6. MIPCAS end to end | both | | build B from the MIPCAS stream basin set (anchors, `train_kinds: ['anchor']`, `live_kernel: 'basin'`) against A, the current prior; run names and config diff (`prior_path`, `train_kinds`, `live_kernel`); `eval_bwd/mle` on B's held-out basins under both, and `density_readouts`; anchor-buffer seeding as in A | - |
| 7. Retire, crystal route only | GFN | | proposed after the owner's protocol review | the pages of what is retired |

Step 1 in detail: the uncommitted `stream_search.py`, `minima.py` and `tests/test_stream_search.py` edits and the
untracked `crystal_fd/metric.py` are committed together (the glued merge imports it), with GFN's working-tree edit of
`eval/fd_analysis/fd_distance.py`, which imports it too. The same hop-kick-limit change exists twice (local 1a0c138b,
origin 2670f412) and origin f0f2a579 edits `run_search.py` and `coordinator.py`, so conflicts fall in the batch loop
kept for MLIP. Pushing publishes the work-in-progress COMPACK commit 191c1b83 and behaviour changes (898895f6 rotation
vector fold, 55e390ce eLJ gathers, 072fa3ac Z' > 1 building); a campaign resubmitted across the push is pinned or held.

Step 7 in detail: the shaped tile (`_shaped_anchor_tile`, `_load_anchor_tile`) and `shape_path` are refused on the
crystal route only, gated in `config_invariants` (the conformer route accepts `iso` and `thermal`, never `shaped`, and
lists `shape_path` among its data-path keys; `InnModeller` accepts `shaped` and loses it in the same change).
`live_kernel` `'hessian'` and `'iso'` are refused. Before any legacy builder module is retired, `image_centroids` and
`ROWS_PER_SOURCE` (from `symmetrize_prior.py`), `keep_molecules` (`molecule_subset.py`) and `apply_overrides`
(`inn_modeller.py`) are re-homed, since the constructor imports them; retirement covers unconditional priors until
conditional basin sets exist. No method or config key the conformer route reaches is removed.

## Open

- Conditional support at scale. The streaming search runs many conditions in one job, each with its own starts,
  cascade reference and a quota (`quota` minima within `quota_band_kT` of its lowest; the owner's aim, 2026-10-30, is
  10-100 distinct low minima with Hessians per condition, not exhaustive coverage), conditions admitted as the slots
  need them ([crystal-search-method](../wiki/crystal-search-method.md)). Open: the cost at qm9rc's size, the quota and band
  defaults, and one shared `kT` across molecules.
- Z' > 1 in the dataset format.
- Saddles: their producer (section 1) and their weight under equal rows per basin.
- Whether the measure's curvature belongs in the draw near the rotation-vector and orientation poles (section 4).
