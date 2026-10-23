# CSD ingestion with the step chain (`csd_v1`) and the canonical FD rebuild against `csd_fd_v1b`

*Battery record. Closed 2026-10-21. Code: uncommitted working trees over GFN `78b7dd60` and MXtalTools `1a0c138b`; the
run identity digest `a5293db0ccdc…` (in the run folder's `recipe.json`) hashes every MXtalTools package file and the
library versions. Not a finding.*

## Design

The question: where the new MXtalTools ingestion chain and the old CIF featurizer both accept a CSD crystal, does the
canonical fundamental-domain (FD) form built from the new chain equal the one in `csd_fd_v1b` (built from
`new_prot_csd.pt`, the old featurizer's output)? And which crystals does each side hold that the other does not?

The chain is the `csd_v1` recipe (`crystal_recipes.py` `CSD_V1`), run per CIF in this order: `read` (float64, Z ≤ 100),
`find_molecules` (file bonds, `special_tol` 0.1), `sanitize` (CCDC chemistry, hydrogens regenerated, a protonation
change flagged, not refused), `parameterize_pose` (measured handedness), `check_rebuild` (max 0.05 Å, mean 0.01 Å),
`check_validity` (≤ 100 heavy atoms, heavy nearest-neighbour 0.9 to 3.0 Å, occupancy, cell constraints 1e-4, minimum
contact 0.5 Å, heavy-atom composition against the moiety formula), `features` (Morgan radius 2, 2048 bits, Gasteiger
charges). No `find_symmetry` or `to_standard_setting` step: neither is built. The FD product is `canonicalize` on the
stored batch with `setting='reexpress'`, `sheet=1`, `off_sheet='mirror_reference'`, `near_face='flag'`,
`zp_gt1='refuse'`; it holds Z′ = 1 rows only.

Two arms on the same code identity:

| arm | CIFs | selection | workers | output |
|---|---|---|---|---|
| subset | 50,000 | seeded sample (seed 20261021) of `D:/crystal_datasets/CSD_dump` | 16 | collated store, FD product of its Z′ = 1 rows |
| full | 873,578 | every CIF in `CSD_dump` | 20 (5 chunks rerun on 5) | 1,748 chunk stores of 500 CIFs, merged FD product |

Reference: `D:/crystal_datasets/csd_fd_v1b` (339,097 rows), compared on shared CSD refcodes. Two FD rows are called
equal when space group, the `h` sign and the reference handedness agree and the cell lengths (relative), cell angles
(rad), fractional centroid `t` (modulo 1) and rotation `R` entries each differ by at most 1e-4. A shared row that is not
equal is counted as near-face if either side carries a `near_face` bit. All runs were local (CPU), early on 21 Oct.

## Data

- Subset store: `D:/crystal_datasets/ingest_csd_v1_sub50k/csd_v1-1b8fd7793a9f/collated` (1.6 GB).
- Full run: `D:/crystal_datasets/ingest_csd_v1/csd_v1-1b8fd7793a9f/` (13 GB): `chunks/` (1,748 stores, each with a JSON
  sidecar holding status counts, timings and digests; five `.failed.json` files from the MemoryError chunks, rerun),
  `recipe.json`, `claims/`, `logs/`. Not collated into one store.
- Full FD product: `D:/crystal_datasets/csd_fd_v2` (355,637 rows, 869 MB; `crystal_fd.dataset` format, `manifest.csv`
  gives every input row's code).
- CSD provenance side table, built the same night from the same refcode list: `D:/crystal_datasets/csd_provenance_v1`
  (873,578 refcodes, all found).
- Scripts and outputs: `energy_sampling/artifacts/fd_design_oct15/ingestion_survey/production/` (git-ignored):
  `paths_sub50k.txt`, `paths_all.txt`, `verify_sub50k.py` → `verify_sub50k.json` and `fd_sub50k/`,
  `diff_rows_sub50k.py` → `diff_rows_sub50k.json`, `post_full.py` → `post_full.json` and `post_full.log`,
  `diff_rows_full.py` → `diff_rows_full.json`, worker logs `logs_sub50k/`, `logs_full/`.

## Measurements

**Table 1.** Where each CIF stopped, by row status. Share of all CIFs in each arm (%). Statuses below 0.5% in the full
run are pooled as "other" (17 statuses; the largest are `heavy_atoms_above_max` 1,954, `chemistry_failed` 1,414,
`hydrogens_failed` 1,149, `component_mismatch` 1,018). The subset is one draw of 50,000, so a share of p% carries a
sampling error of about √(p(100−p)/50,000)%, 0.22% at p = 47.

| status | full: CIFs (count) | full: share (%) | subset: share (%) |
|---|---|---|---|
| `ok` | 411,315 | 47.08 | 47.30 |
| `multi_component` | 197,279 | 22.58 | 22.65 |
| `special_position` | 189,569 | 21.70 | 21.44 |
| `features_failed` | 46,057 | 5.27 | 5.27 |
| `polymer` | 10,357 | 1.19 | 1.19 |
| `composition_mismatch` | 5,881 | 0.67 | 0.62 |
| `unlisted_bond` | 4,437 | 0.51 | 0.47 |
| other | 8,683 | 0.99 | 1.05 |

**Table 2.** Accepted (`ok`) rows by Z′, full run (subset in brackets). Z′ = 1 is 86.5% of accepted rows in both arms.

| Z′ | 1 | 2 | 3 | 4 | 5 to 32 |
|---|---|---|---|---|---|
| accepted rows (count) | 355,641 (20,435) | 49,223 (2,817) | 2,725 (166) | 3,095 (188) | 631 (43) |

**Table 3.** The FD product of the full run, by outcome (`canon.STATUS`). Rows: Z′ = 1 accepted rows. `pose_only` is
space group 16 and above (pose folded, cell kept as given); `special_metric` is a reduced cell with more than one lattice
automorphism (the pose depends on the input basis). Of the 355,641 Z′ = 1 rows, 4 are not in the product: 3 refused as
`setting_metric_mismatch`, 1 failed `mono_angle_off`. The 55,674 accepted Z′ > 1 rows are refused (`z_prime_gt1`).

| outcome | rows (count) | share of product (%) |
|---|---|---|
| `canonical` | 268,813 | 75.59 |
| `pose_only` | 86,474 | 24.32 |
| `special_metric` | 350 | 0.10 |

**Table 4.** The FD product against `csd_fd_v1b` on shared refcodes. Counts of refcodes. Subset counts are over the
19,488 `csd_fd_v1b` rows whose refcode is in the 50,000 sample.

| arm | shared | equal | different, not near a face | different, near a face | only in the new product | only in `csd_fd_v1b` |
|---|---|---|---|---|---|---|
| subset | 19,250 | 19,223 (99.86%) | 8 | 19 | 1,185 | 238 |
| full | 334,377 | 333,771 (99.82%) | 360 | 246 | 21,260 | 4,720 |

**Table 5.** Largest difference over the equal rows of Table 4 (all equal rows, so the maximum, not a typical value).
Heavy-atom deviation of the reference molecules was computed on the subset only.

| arm | cell lengths, relative | cell angles (rad) | `t` (fractional) | `R` entries | reference molecule heavy atoms (Å) |
|---|---|---|---|---|---|
| subset | 2.2e-7 | 1.9e-7 | 1.9e-6 | 9.9e-5 | 5.9e-4 |
| full | 3.1e-7 | 4.7e-7 | 2.3e-6 | 1.0e-4 | not computed |

**Table 6.** Are the "different, not near a face" rows the same crystal? Each side's float64 unit cell is built from its
own reference molecule and pose; the number is the largest distance from any atom to the nearest same-element atom of
the other cell under periodic images, both ways (Å). Subset: all 8 rows. Full: the first 200 of the 360 by refcode.

| arm | rows checked | median, heavy atoms (Å) | largest, heavy atoms (Å) | rows above 0.01 Å |
|---|---|---|---|---|
| subset | 8 | 1.3e-3 | 4.6e-3 (VURXUE) | 0 |
| full | 200 | 9.7e-4 | 9.3e-3 (EWIBEU01) excluding DIMETH10 | 1 (DIMETH10, 1.56) |

DIMETH10 (CH₂I₂, C2/c, five atoms, a molecular mirror) failed the cell-overlay test above. The overlay compares the two cells in
one frame, so it assumes both rows share a cell basis; sorted heavy-atom distance lists, which do not, built separately
on each side both match the CIF. In the full spot check the rows are 76 in P2₁/c, 34 in P-1, 29 in P2₁2₁2₁, 12 in
C2/c and 12 in Pbca, the rest scattered.

**Table 7.** Refcodes on one side only, by the code the other side gives them (full run). Counts of refcodes.

| side | code on the other side | refcodes (count) |
|---|---|---|
| only in the new product (21,260) | absent from `new_prot_csd.pt` | 21,260 |
| only in `csd_fd_v1b` (4,720) | `composition_mismatch` | 3,914 |
| | `multi_component` | 413 |
| | `unlisted_bond` | 225 |
| | other (8 codes) | 168 |

**Table 8.** Row flags kept on the row (not refusals), full run. Counts over all 873,578 rows; a row that never reached
the step that sets the flag carries false. A sample of 10 chunk stores of the full run (5,000 rows, 2,398 accepted): 644 accepted rows carry
`charges_nonfinite`, 635 of them contain an element outside H, B, C, N, O, F, Si, P, S, Cl, Br, I, and in those rows a
median 100% of atoms have a non-finite charge (minimum 81%).

| flag | meaning | rows (count) | share of all CIFs (%) |
|---|---|---|---|
| `protonation_changed` | regenerated hydrogens differ in count from the deposited ones on some heavy atom | 51,425 | 5.89 |
| `charge_imbalance` | some unit's formal charges do not sum to zero | 109,715 | 12.56 |
| `charges_nonfinite` | some atom's Gasteiger charge is not finite (kept as NaN) | 110,487 | 12.65 |

`features_failed` messages in the same 10 chunks (243 rows): RDKit cannot rebuild the CCDC chemistry, at least
134 with a `KekulizeException` and most of the rest an `AtomValenceException` on O, B, C or Sn.

**Table 9.** Cost. Full run: 1,748 chunks of 500 CIFs; wall time 68 min (02:54 to 04:02 local, 20 workers, then 5
reruns); summed chunk time 70,818 s (81 ms per CIF); merging, canonicalizing and comparing the chunk stores 721 s.
Subset: 4.5 min on 16 workers; `canonicalize` on its collated store 25 s.

Ten accepted Z′ > 1 rows have two units whose heavy atoms overlay within 1e-4 Å RMSD after a fit: BAZGOY05, BAZGOY06,
IYIWIY, JISLUT01, PYRZOL04, PYRZOL05, QQQEJV07, ZZZUXA02, ZZZUXA03, ZZZVCO03 (2 of them, IYIWIY and PYRZOL04, in the
subset). The 3,212 other subset Z′ > 1 rows have a median smallest unit-unit RMSD of 0.40 Å.

## Trends

- Status shares in the subset are within 0.3 percentage points of the full run for every status in Table 1.
- On shared refcodes, 99.8% of rows are equal at 1e-4, with equal-row differences at or below 1e-4 in `R` and 3e-6 in
  every other field. The non-equal rows are 0.18% of shared rows; every one checked is the same crystal (Table 6), and
  41% of them carry a near-face bit.
- The new product's extra 21,260 refcodes are all absent from the old featurizer's output; 83% of the refcodes it lacks
  are `composition_mismatch` rows.
- `charges_nonfinite` and `charge_imbalance` each touch about 12.6% of all CIFs; in the 10-chunk sample the non-finite
  charges sit almost entirely on rows with an element outside the organic set, where the whole row is non-finite.

## Not covered

- Z′ > 1 rows were not canonicalized (refused by design); the ten near-identical-unit rows were not examined for missed
  symmetry.
- 160 of the 360 full-run differing rows were not spot checked, nor were the 246 near-face rows.
- No standard-setting or symmetry-finding step ran; rows keep their deposited setting until `canonicalize`
  re-expresses them.
- Hydrogens: regenerated positions were not compared with the deposited ones beyond the count flag.
- Chemistry against a CCDC-free route was not compared.
- The chunk stores were not collated into one store; the full-run status counts come from loading each chunk.
- Charges on `ok` rows were not compared with the old featurizer's.
