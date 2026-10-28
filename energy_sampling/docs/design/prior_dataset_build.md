# A pre-noised prior dataset, built once

*Plan, 2026-10-27. Status: direction agreed by the owner (the Decisions below); everything else is a proposal,
reviewed once against the code (2026-10-27) and to be checked again as it lands. Not policy.*

*Built (2026-10-28), in the crystal-data format the owner chose over flat tensors for now: the builder
(`data_processing/build_noised_prior.py`, `--harvest` merge-back), `noised_prior.py`, train.py's loader
(`Modeller._load_noised_prior`: identity, sampled entry check, filters; FD chart references), the static prior buffer
(`Modeller._prior_is_static`), the density readouts (`Modeller.density_readouts`) and the harvest writer
(`Modeller.harvest_scored_rows`, cfg:harvest). Not built: the flat-tensor format (sections 1 and 6's per-row fields
beyond the tags), carriers stored as rows (seed_min still analyses the conditions files at start-up), a vectorised
near-duplicate pass, and the INN loader.*

## Why

Stage 1 (likelihood) and the stage-2 backward data are built today by noising dataset rows inside the run. On the
INN's buffer route (`inn_modeller.py::InnModeller.init_inn_buffers`, `::pbuf_churn`) that means a start-up fill, a
churn at every eval, a `gauss` kick and an admission gate `E < Emin(c) + ramp_floor` whose floor the run's own draws
lower. Found against it (2026-10-27):

- The full-set buffer held 4 rows per molecule against about 9.8 anchors per molecule: a smaller and noisier target
  than the anchor set itself.
- A molecule whose floor drops by more than about 8 kT loses every row at the next refill, with nothing replacing it,
  so the training set depends on what the model happened to draw.
- In-run redrawing did not prevent overfitting: the inn_cap26 1% and 10% arms memorised each fresh set within its
  2,500-step window; the full-set arms held.
- Every noised row's energy call is spent inside one run, then discarded.

## Decisions (owner, 2026-10-27)

- One large pre-generated, pre-noised dataset. No in-run noising, churn or admission gate for stage 1 or for the
  stage-2 backward data.
- Noise kernel `iso` (log-uniform radius, unit direction). The `gauss` tile is retired; Hessian-shaped tiles are set
  aside until their tooling exists.
- Filters: N kT above each condition's minimum; packing coefficient roughly 0.55-0.95.
- Energy calls are paid once and saved. The dataset is tied to the energy definition and is not re-scored at run
  entry.
- Thin very close duplicates. Per-molecule landscape diversity is a separate prior-construction question, parked.
- Save every state a run scores, compactly, per condition, in a form that instantiates on crystal batches.
- Overfitting is answered by rebuilding (more noise samples, more molecules), not by in-run machinery.
- The dataset may grow between builds by new conditional minima and by harvested states.
- The energy window stays in kT at the run's temperature for now; that it moves with T is known and deferred.
- Harvest retention: low-error rows are dropped. The writer keeps what the replay's error filter keeps, plus states
  below their condition's minimum.

## Shape

### 1. Two tables: molecules once, rows flat

Today a row is a full collated crystal graph, about 1.97 kB on qm9rc (2,207,012,889 bytes over 1,120,750 rows), the
molecule's atoms and embedding copied into every row. The new store keeps each molecule once and each row as a few
numbers.

- **Molecule table**, one entry per molecule: the identifier, a hash of the molecule's atoms in its standard frame
  (the latent is defined relative to the molecule's radius and frame, MXtalTools `crystal_ops.py` `latent_params`, so
  the key pins the exact molecule, not only its SMILES), the atom arrays, radius, mass, volume, and the conditioning
  embedding. Training and held-out molecules both, flagged. Source: the conditions files' carriers.
- **Row table**, flat tensors, about 90 bytes per Z' = 1 row. A condition is (molecule, space group, Z'), as
  `molecular_crystal.py::condition_ids_of` defines it; a run's condition id is a run-local index and is never stored.

| field | type | meaning |
|---|---|---|
| `latent` | fp32 [N, 6 + 6Z'] | gauge-fixed `latent_params` on the latent chart (the FD chart is a later, separate column) |
| `mol` | int32 | index into the molecule table |
| `sg_ind`, `z_prime` | uint8 | passed to `condition_samples`, never resampled |
| `e_raw` | fp32 | the energy attribute exactly as analysis returns it (`.elj` per asymmetric unit, or the MLIP's), with the stamped `lj_coeff` divided out; before the Z' division, `lj_rescale` and the implausible-energy screen, which `generator_energy` applies at load |
| `reduction_en`, `packing_coeff`, `cell_volume`, `sym_mult` | fp32 | geometry terms `generator_energy` reads |
| `log_jac` | fp32 | the chart Jacobian in nats, `-jacobian_energy / T_build` |
| `kind` | uint8 | anchor, child, carrier, harvested |
| `split` | uint8 | train, held-out anchor group, held-out molecule |
| `parent` | int32 | the anchor row a child was kicked from (-1 otherwise) |
| `disp` | fp16 | the realised displacement from the parent in the gauge-fixed chart (the kick itself acts on the gauge-free latent and is then clipped and cleaned, so the kick radius alone is not it) |
| `source` | int32 | build or run that produced the row |

- Bounding energy is zero on every stored row (no raw latents), as it is for stored rows today.
- Shards by molecule range, plus a manifest: counts, the identity (section 2), the `lj_coeff` the build stamped,
  build arguments, both commits.

### 2. Identity: physics in the file, reward at load

`generator_energy` composes the reward from stored terms and reward settings. What is baked into stored values, and
so forms the identity the loader refuses on, read from the code (2026-10-27 review):

- `energy_function` and the MLIP weights (`mlip_path`);
- the analysis settings as resolved code defaults, since mk_dev's `analyze_kwargs` is empty: `cutoff` and
  `supercell_size` (both 10), `std_orientation`, `noise`, `repulsion`, `lj_envelope`, `margin` (changes
  `reduction_en`);
- the environment flag `MXT_LEGACY_TRICLINIC_WALLS` (changes `reduction_en`);
- `crystal_coordinates`;
- `mlip_max_packing_coeff`, which makes a guarded cell store a placeholder, not an energy.

Applied at load to the stored terms, so changing them costs no energy call: `lj_coeff` (the loader writes
`e_raw * lj_coeff` back to the energy attribute and stamps the batch, which `assert_lj_coeff_stamped` requires;
`prebuilt_sample_to_reward` does not do this itself), `lj_rescale`, the implausible-energy screen, `density_coeff`,
`pressure`, `reduction_coeff`, `bounding_coeff`, temperature, `reward_range`, `physical_energy_clip`,
`energy_reference`. `prior_scan_cache`, whose identity includes the whole `energy_config`, temperature included, is
not reused for this.

The MXtalTools commit is recorded, not refused on (any commit change would refuse every file). Drift is caught by the
entry check instead: 512 stored rows, stratified by shard, space group, Z' and row kind, rebuilt from latent +
molecule and scored; each component compared separately (`e_raw`, `reduction_en`, `packing_coeff`, `cell_volume`,
`log_jac` within 1e-4 relative plus 1e-2 kJ/mol; `sym_mult`, `z_prime` exactly), gated on the worst row, with any row
off by more than 1 kJ/mol failing the run. 512 energy calls, not a full re-score. (A median test, as
`check_buffer_reward_currency` uses, would miss one bad shard.)

### 3. One minimum per condition

One definition, used both as the reward reference (`energy_reference: seed_min`) and as the floor of the energy
window: the minimum over the condition's training rows (anchors, children, carriers) of `seed_energy_from` (molecule
energy + density + pressure terms, `molecular_crystal.py`), taken over rows inside the packing window with
`reduction_en <= 0`. The loader computes it from the stored terms under the run's reward settings: arithmetic, no
energy call. The carriers in today's conditions files, which `_seed_min_reference_table` analyses at start-up, are
stored as rows of kind carrier, held-out molecules' rows with split held-out molecule, so seed_min needs no analysis.

Guards: a child more than a set margin below its parent is flagged in the build report (one spurious low child would
set its condition's floor and empty its window); the loader reports, and refuses past a set share, conditions left
with no rows after filtering.

### 4. Build (offline, on the cluster)

1. **Anchors:** the existing prior rows (relaxed endpoints). On rc_oct22 the screened RDF dedupe and the refit to
   the trainer chart are already done (`build_rc_prior.py`, which exists only in the main checkout and must be
   committed before it is built on).
2. **Children:** K per anchor, `iso` kicks at a log-uniform radius in [lo, hi] (mk_dev's anchor-noise range,
   [-2.5, -1.5], as the default) through the MXtalTools primitive `log_noise_latent_parameters` (clip, decode, hard
   clean), the latent then re-read gauge-fixed. Scored in large batches. Every finite child is stored, whatever its
   energy: the energy and packing filters are load-time settings. Measured on qm9rc children at this range
   (`buf_iso_smoke_oct26`): excess over Emin(c) P10/P50/P90 0.55/3.80/8.52 kT at T 6.9, below the thermal shell's
   median of about 5.7 kT at 12 dimensions; 99.6% under a 14.5 kT gate.
3. **Held-out anchor groups:** about 5% of each training molecule's anchors, with all their children, held out
   whole (the INN's crystal-grouped split, `--val_group_ignore_rows`, generalised). A child sits within 0.03 latent
   units of its parent and ~K training siblings, so held-out children of training anchors would only measure how well
   the anchor is fitted; whole held-out anchors measure memorisation within a molecule.
4. **Thinning:** near-duplicates among anchors and harvested rows, within a condition, at a cutoff measured on the
   data (`AnchorBuffer.admit`'s `dup_cutoff` 0.05 is larger than the largest kick, 10^-1.5, and its per-candidate
   loop does not scale to millions of rows: a vectorised pass). Children are not thinned: they sit near their parent
   by design.

### 5. Load

- Filters, all at load: the energy window above the condition's minimum; the packing window; molecule fraction; row
  kinds; split.
- Stage 1 trains on children only by default. Anchors go to evaluation (`eval_bwd/mle` keeps its meaning: the clean
  anchors) and to the stage-2 backward set. Harvested rows are excluded from stage 1 by default (the INN already
  refuses training the likelihood term on the flow's own draws, `--grow_stored` with `--w_mle`).
- Emin(c) is the loader's and stays fixed for the run; a running minimum of the run's own draws is reported
  separately.
- Weighting: uniform over rows (natural counts) by default. qm9rc has 5-12 anchors per molecule (median 10; the top
  10% of molecules hold 10.3% of rows), so no molecule dominates at the first build. The loader logs the spread of
  rows across conditions after filtering; per-condition balancing is a sampler option, parked.
- Draws hand `condition_samples` the stored `sg_ind` and `z_prime` (today's backward draw passes neither,
  `train.py::Modeller.draw_bwd_sample`, so both are resampled: harmless at one of each, wrong otherwise).

### 6. Saving every scored state

A writer appends rows (the row-table fields plus step and source kind) where scored batches come back, not inside
`analyze_crystal_batch` (which would also harvest the entry checks): the forward scoring sites in `train.py`
(`return_exp=True`), evaluation, and the INN's scoring sites. The latent is read with
`latent_params(gauge_fix_free_axes=True)` on the returned batch (the policy's raw state is pre-clamp and not
gauge-fixed). Replay admissions are a subset of these rows and are not written twice. Shards are named by run, job
id, process id and first step, written atomically, with a shard manifest, since arms share a job and a reused run
name overwrites.

Rate: the conditional protocol scores about 140 rows per step (2,000 per 20 steps plus 10,000 per 250 steps); kept
unfiltered, a long run could write up to ~180 M rows (~18 GB) for a 2,700-minute arm. Retention (owner): low-error
rows are dropped; the writer keeps the rows the replay's error filter keeps, plus states below their condition's
minimum.

A rebuild merges harvested rows in: rows enter the row table as kind harvested; a state below its condition's minimum
becomes a new anchor and gets children. The build report gives each condition's change in kept rows.

### 7. What is replaced, and in what order

On the crystal route only. `rebuild_prior_by_churn` is an `on_enter` action of `unconditional_tb` (the primary
route) and `conditional_vargrad`, and of `configs/conformer_mk.yaml` (out of scope, pinned by
`tests/config/test_conformer_canonical_contract.py`); `manage_prior_buffer` runs on every eval rollout;
`conditional_vargrad`'s `condition_draw` needs at least 2 prior-buffer rows per condition. So:

1. The loader serves the draws the backward step reads: the per-condition aligned draw (`rows_per_condition`),
   condition blocks (`condition_block_m`), and a per-row mutable `ema_loss` for the loss-weighted draw.
2. The owner edits mk_dev's `on_enter` lists.
3. The crystal route stops calling `top_up_prior_from_anchors`, `rebuild_prior_by_churn`,
   `seed_prior_from_condition_minima` and the expiry in `manage_prior_buffer`. The protocol `ACTIONS` entries stay
   while the conformer route uses them.
4. The INN's `gauss` tile and `pbuf_*` go once no queued job reads them: the stage-2 batteries' `common.args` pass
   the buffer route, and both inn_cap27 sbatch scripts check `def pbuf_ref_energy` and call
   `calibrate_gauss_sigma` at start-up.

## Sizing (qm9rc)

| quantity | value |
|---|---|
| training molecules / anchors | 114,817 / 1,120,750 (9.76 per molecule) |
| children at K = 10 | about 11.2 M rows |
| store | about 1.1 GB of rows (90 B) + about 0.23 GB molecule table |
| energy calls | about 11.2 M, once |
| scoring time | about 15-55 GPU-minutes at the 0.07-0.25 ms/sample logged by local qm9rc runs (observation; wandb runs of 2026-10-26); decode, clean, collate and RDF thinning, not scoring, set the build time |

*Counts from the rc_oct22 qm9rc build metadata; K is a proposal; bytes at 90 per row.*

## First deliverable

1. A builder from one existing prior file: molecule table, anchors, K `iso` children, held-out anchor groups,
   manifest. MIPCAS first (the one-system tuning rule), then the qm9rc 1% subset.
2. A loader that feeds `InnModeller.init_data_rows` from tensors: it needs only latent, log R, condition id and the
   condition vector. `train.py`'s backward draw consumes collated graph batches (`draw_bwd_sample`), which would
   inflate 12 M rows back to about 25 GB, so its tensor path comes with the NF port, through the same loader.
3. Tests: a stored row rebuilt from latent + molecule scores its stored terms, at `lj_coeff` != 1; the tensor-path
   log R equals the rebuilt-batch log R (`prebuilt_sample_to_reward` recomputes the Jacobian rather than reading
   `log_jac`); the identity refusal; each filter; no held-out row reaches a training draw.
4. Wiki pages updated in the same change: the prior-buffer and anchor-buffer pages, the stage-protocol page, the MLE
   phase.

## Verification

The first runs read two held-out gaps:

- **Held-out molecules** (`test_molecules_path`, `eval_test/mle`): generalisation across molecules, which more
  molecules fix.
- **Held-out anchor groups of training molecules:** memorisation within a molecule, which more children or anchors
  fix.

## Open questions (owner)

1. The packing window's exact bounds.
2. K and the radius range for the first build.
3. The FD chart: a second latent column, or a separate build.

Deferred (owner, 2026-10-27): the energy window in kT changes the training set whenever T changes (the same
children are 2.76x further out in kT at mk_dev's T 2.5 than at 6.9; a 10 kT window keeps under half at 2.5 against
90% at 6.9) and is undefined under temperature conditioning.

## Noted for the training recipe (item 4 of the owner's list, not this plan)

A fixed backward set biases the stage-2 stored level until the model matches the target: draws (excess P50 about 31
kT) and stored rows (about 4 kT) barely overlap, and nothing now moves the stored set toward the draws. The recipe
should name its level mode and track the stored-minus-fresh level gap.

## Not in scope

The conformer route; per-molecule landscape diversity; Hessian-shaped kernels.
