# Condition-set sources against the Day-1000 FIT+DMA landscapes

*Dataset record. Closed 2026-11-02. Code at commit `2f17b75d` plus the uncommitted
`data_processing/condition_set_stats.py`. Not a finding.*

This folder's README is written for test batteries. This record applies it to a designed dataset comparison: the
"arms" are data sources, and nothing was trained.

## Design

The question: how do the molecules and crystals available for training a molecule- and space-group-conditioned
generator compare with the Day-1000 FIT+DMA landscapes (Taylor, Butler and Day, 2025), which the generator is to be
benchmarked on, and how many Day molecules do those sources already contain?

Sources, each read in its native stored quantities (nothing re-scored, relaxed or re-embedded):

| key | source | what is read |
|---|---|---|
| day | `D:/crystal_datasets/day1000_fitdma/FIT-DMA_Landscapes/` | every CSV of each of the 1,017 refcode dirs, union of structure ids; molecule = asymmetric unit of the first CIF block, bonds by RDKit `rdDetermineBonds` |
| csd | `D:/crystal_datasets/csd_fd_v5` | 394,934 Z′ = 1 CSD crystals (all elements); molecule key from the stored RDKit SMILES |
| qm9 | `D:/crystal_datasets/qm9_dataset.pt` | 133,728 molecules, DFT geometries, no crystals |
| qm9rc, csdrc, zincrc, gdbrc, pubrc | `D:/crystal_datasets/conditional/priors/rc_oct22/rcoct22i1_<family>_*` | the five built rc_oct22 families: conditions and held-out conditions (molecules), prior and held-out prior (`equalized_prior` crystals; eLJ, sg 2, Z′ = 1) |
| single | MIPCAS, NEHZOR and acridine priors named in `condition_set_stats.py::SINGLE` | stored energy and density only |

A further source, **pool**, is the proposed Day-like training pool: the CSD H/C/N/O/F components, ZINC20 2D
tranche A and PubChem CID-SMILES molecules that pass the Day-like filter of Table 2. Day's own molecules are removed
by SMILES and by InChIKey skeleton. The pool is molecules only, with no conformer and no crystal. The key list is
`D:/crystal_datasets/paper_review_nov02/pool/daylike_pool_v0.csv.gz`: 155,602 census keys, which give 152,462
distinct keys once the stats script canonicalises them again.

### What the rc_oct22 families are

The five rc families are not samples of their named sources. Each is the 4 to 9 heavy-atom window that
`data_processing/smiles_pool.py` cuts from its source (`--lo 4 --hi 9`, parents up to `--max-parent` 40). A
source molecule inside the window enters as itself. A larger one is cut at every acyclic single bond whose sides
both hold more than one atom, and every connected union of pieces inside the window enters, hydrogen-capped, up to
32 per parent. `data_processing/smiles_mixture.py` then draws the selection `new_oct22`:

- sources in the order CSD, ZINC, GDB-11, PubChem, with quotas "all, all, all, 1,000,000";
- every QM9 molecule excluded;
- a molecule found in several pools is drawn once, under the first, so the family name is the pool a molecule was
  drawn from first;
- 5% held out by a salted hash of the SMILES.

qm9rc is the QM9 molecules themselves, keeping the qm9full split. Each molecule gets one ETKDG conformer with
random torsions (`data_processing/random_conformers.py`), a random-start eLJ search in P-1 (5 seeds), and the
`build_rc_prior.py` reductions. The pre-density-bound build is the one on D:.

**Table 0.** Composition of the built rc families, measured by joining each family's condition molecules (train and
held out) to the selection file. *Fragment* = no input component identical to the molecule (`n_direct` = 0), so
it exists only as a capped piece of a larger compound; the parent size is the smallest parent's heavy-atom
count. *Chunks* = search chunks built into the prior over chunks searched.

| family | source cut from | molecules (count) | fragments (%) | smallest parent, median (heavy atoms) | chunks built / searched |
|---|---|---|---|---|---|
| qm9rc | QM9 molecules as such | 121,469 | 0 | – | 66 / 66 |
| csdrc | CSD H/C/N/O/F components | 52,645 | 98.1 | 21 | 28 / 28 |
| zincrc | ZINC20 2D tranche A | 177,256 | 68.3 | 11 | 92 / 92 |
| gdbrc | GDB-11, sizes 1–11 | 192,004 | 64.7 | 11 | 100 / 477 |
| pubrc | PubChem, 1,000,000 drawn | 175,678 | 86.5 | 15 | 100 / 284 |

Molecule key: canonical stereo-free SMILES (`condition_set_stats.py::canon`). Per-source molecule descriptors come
from RDKit on that SMILES; geometry figures come from each source's stored 3D coordinates. Sub-sampling: bond,
angle and planarity figures use 20,000 molecules per source drawn with seed 0 (rc: 20,000 train molecules per
family, 100,000 pooled; Day: all 1,017 refcodes); the RDF panel uses 40 Day molecules per size stratum and 80
held-out molecules per rc family, at most 10 crystals each. All runs were local, CPU only.

## Data

- Script: `energy_sampling/data_processing/condition_set_stats.py` (stages `extract`, `rdf`, `plot`).
- Feature cache: `D:/crystal_datasets/paper_review_nov02/data/` (per-source `*_molecules.csv.gz`, `*_crystals.npz`,
  `*_geometry.npz`, `rdf_*.npz`, `single_systems.npz`).
- Figures and summary numbers: `D:/crystal_datasets/paper_review_nov02/fig/` (21 PNGs) and
  `D:/crystal_datasets/paper_review_nov02/stats.json` (every number in the tables below, plus the captions under
  `figures`).
- Command, from `energy_sampling/`, csd_mxt_gfn venv, MXtalTools on `PYTHONPATH`:
  `CUDA_VISIBLE_DEVICES=-1 python -m data_processing.condition_set_stats --steps plot --fig D:/crystal_datasets/paper_review_nov02/fig --stats D:/crystal_datasets/paper_review_nov02/stats.json`
  (`--steps extract,rdf,plot --refresh` rebuilds the cache; about one hour).

## Reading the Day deposit

- 661 achiral refcode dirs hold `<ref>.csv`. 356 chiral dirs hold `_racsg`, `_sohncke` and `_all` files. In 118
  of the chiral dirs, `_all.csv` holds only the Sohncke rows; their racemic rows are in `_racsg`/`_racemic`. The
  structure set of a refcode is therefore the union of ids over every CSV in its dir: 4,686,991 structures. Reading
  `_all.csv` (or `<ref>.csv`) alone gives 4,102,703.
- No id carries two different (space group, density, energy) triples across the CSVs of one dir (checked over all
  1,017 dirs).
- CSV density agrees with the density recomputed from the CIF block (cell volume, operator count, atom masses) to a
  relative 8.5 × 10⁻⁵ on 120 structures (3 per refcode for 40 random refcodes).
- Structures within 8 kJ/mol of their own molecule's minimum: 174,802 over the union. The paper reports 166,395.
  Other readings: restricted to the 26 standard space groups, 174,783; `_all.csv`/`<ref>.csv` alone, 165,731;
  racemic and Sohncke files each against their own minimum, 187,017. None reproduces 166,395.
- 1,472 rows sit in space groups outside the 26 (114, 45, 122, 110, 92, 41). One refcode's global minimum is one
  of them.
- Bond perception gives 989 distinct constitutions. For `lovwoi` it returns a fragmented radical, and the script
  drops that key, leaving 988 molecules over 1,016 keyed refcodes. For the 736 Day refcodes also in `csd_fd_v5`,
  the perceived SMILES and the CSD's own SMILES agree on 705. Of the 31 that differ, 11 have the same InChI
  skeleton; most of the rest are charge-separated against neutral spellings (N-oxides, furoxans, diazonium
  olates).

## Measurements

**Table 1.** Molecule size per source: 5th percentile, median and 95th percentile of each quantity over distinct
molecules (rc families: train split; Day: 988 distinct constitutions). The last column is the share of molecules
with at most 9 heavy atoms, the ceiling of every generated set. Compare each row with Day.

| source | molecules (count) | heavy atoms p05 / median / p95 (count) | MW p05 / median / p95 (g/mol) | ≤ 9 heavy atoms (%) |
|---|---|---|---|---|
| Day-1000 | 988 | 7 / 12 / 16 | 93 / 170 / 207 | 18.5 |
| CSD `csd_fd_v5` | 327,294 | 14 / 26 / 54 | 210 / 399 / 900 | 0.7 |
| QM9 | 133,641 | 8 / 9 / 9 | 109 / 125 / 130 | 100 |
| proposed pool | 152,462 | 8 / 12 / 16 | 107 / 170 / 220 | 15.1 |
| rc qm9rc | 114,817 | 8 / 9 / 9 | 110 / 125 / 130 | 100 |
| rc csdrc | 49,272 | 6 / 9 / 9 | 88 / 122 / 132 | 100 |
| rc zincrc | 168,114 | 7 / 9 / 9 | 100 / 126 / 134 | 100 |
| rc gdbrc | 182,451 | 8 / 9 / 9 | 112 / 127 / 134 | 100 |
| rc pubrc | 166,899 | 7 / 9 / 9 | 99 / 125 / 135 | 100 |

**Table 2.** Share of distinct molecules (%) meeting each Day selection criterion. "Rigid, Day-like" adds the
absence of a CH₃, OH, NH₂ or SH rotor group to RDKit's zero rotatable bonds. It admits 986 of the 988 Day molecules;
`kobxoo` and `xughud` each carry an NH₂. Z′ = 1 and the 26-group restriction are crystal-level criteria and are
not included.

| source | C/H/N/O/F only (%) | MW < 230 (%) | 0 rotatable bonds, RDKit (%) | all three (%) | all three + no rotor group (%) |
|---|---|---|---|---|---|
| Day-1000 | 100 | 100 | 99.9 | 99.9 | 99.8 |
| CSD `csd_fd_v5` | 37.8 | 7.6 | 10.6 | 1.4 | 0.5 |
| QM9 | 100 | 100 | 45.9 | 45.9 | 13.5 |
| proposed pool | 100 | 100 | 100 | 100 | 100 (by construction) |
| rc qm9rc | 100 | 100 | 41.4 | 41.4 | 12.1 |
| rc csdrc | 100 | 100 | 22.7 | 22.7 | 4.8 |
| rc zincrc | 100 | 100 | 13.5 | 13.5 | 2.6 |
| rc gdbrc | 100 | 100 | 24.9 | 24.9 | 6.2 |
| rc pubrc | 100 | 100 | 17.2 | 17.2 | 3.4 |

**Table 3.** Selected functional groups and planarity: share of distinct molecules (%) with at least one SMARTS
match, and the share whose stored conformer has a heavy-atom rms out-of-plane deviation below 0.05 Å. Planarity
uses the geometry samples (Day 1,017 refcodes; CSD and QM9 20,000 each; rc 100,000 pooled train).

| source | aromatic ring (%) | CH₃ (%) | O–H (%) | 3-membered ring (%) | carbonyl (%) | planar conformer (%) |
|---|---|---|---|---|---|---|
| Day-1000 | 45.2 | 0.0 | 0.0 | 12.6 | 44.7 | 29.1 |
| CSD `csd_fd_v5` | 84.6 | 72.7 | 17.0 | 6.5 | 42.5 | 2.1 |
| QM9 | 17.8 | 59.9 | 32.4 | 41.2 | 34.8 | 12.1 |
| rc qm9rc | 13.3 | 61.1 | 33.4 | 44.3 | 36.9 | – |
| rc csdrc | 8.8 | 62.9 | 30.9 | 6.5 | 44.1 | – |
| rc zincrc | 8.9 | 67.9 | 33.3 | 12.5 | 28.4 | – |
| rc gdbrc | 5.2 | 52.1 | 33.3 | 20.2 | 20.0 | – |
| rc pubrc | 3.3 | 65.6 | 32.5 | 14.2 | 23.9 | – |
| rc, five families pooled | – | – | – | – | – | 1.7 |

**Table 4.** Crystal-level comparison. Densities in g/cm³. *Groups* counts the space groups present. For the rc
priors, "packing < 0.55" is the share of rows with a packing coefficient below 0.55 and the share of molecules with
at least one such row; Day and CSD store no packing coefficient in the files read.

| source | crystal rows (count) | crystals per molecule, median (count) | groups (count) | median density (g/cm³) | packing < 0.55, rows / molecules (%) |
|---|---|---|---|---|---|
| Day-1000, all structures | 4,686,991 | 3,635 | 26 (+6) | 1.137 | – |
| Day-1000, within 8 kJ/mol | 174,802 | 107 | 26 (+ non-standard, 0.01% of rows) | – | – |
| Day-1000, global minimum per refcode | 1,017 | 1 | 23 (+1 non-standard) | 1.326 | – |
| CSD `csd_fd_v5` (Z′ = 1) | 394,934 | 1 | all observed | 1.373 | – |
| rc qm9rc train / held out | 1,120,750 / 64,925 | 10 | 1 (sg 2) | 1.321 / 1.313 | 5.1 / 39.8 |
| rc csdrc train / held out | 482,130 / 32,958 | 10 | 1 | 1.289 / 1.266 | 4.8 / 38.4 |
| rc zincrc train / held out | 1,647,385 / 89,605 | 10 | 1 | 1.264 / 1.265 | 5.1 / 39.7 |
| rc gdbrc train / held out | 1,789,638 / 93,728 | 10 | 1 | 1.318 / 1.319 | 4.9 / 38.9 |
| rc pubrc train / held out | 1,635,957 / 85,991 | 10 | 1 | 1.293 / 1.296 | 4.9 / 38.5 |

(Packing shares are the train split; the held-out shares agree to within 1 percentage point.)

**Table 5.** Space-group shares (% of rows) for the groups with the largest share in any Day row. Day global minima
fall in 23 of the 26 groups. rc priors are 100% sg 2.

| group | Day all structures (%) | Day within 8 kJ/mol (%) | Day global minima (%) | CSD Z′ = 1 (%) |
|---|---|---|---|---|
| 14 (P2₁/c) | 12.0 | 31.4 | 46.4 | 42.5 |
| 2 (P-1) | 1.1 | 4.4 | 11.0 | 19.6 |
| 19 (P2₁2₁2₁) | 2.1 | 5.0 | 9.0 | 13.0 |
| 61 (Pbca) | 4.9 | 7.1 | 8.4 | 4.9 |
| 4 (P2₁) | 1.1 | 2.8 | 5.2 | 6.9 |
| 15 (C2/c) | 14.7 | 14.9 | 4.1 | 4.5 |
| 60 (Pbcn) | 7.6 | 3.6 | 0.3 | 0.4 |
| outside Day's 26 | 0.03 | 0.01 | 0.1 (1 refcode) | 0.9 |

**Table 6.** Day molecules present in each training or held-out source, matched by canonical stereo-free SMILES. A
lower bound: charge-separated Day perceptions miss their neutral spellings. The last column restricts to the 187
keyed Day refcodes with at most 9 heavy atoms (188 with `lovwoi`), the only ones a ≤ 9-heavy-atom source can hold.

| source (split) | distinct molecules in source (count) | Day molecules hit (count) | Day refcodes hit (count) | Day refcodes hit, ≤ 9 heavy (count) |
|---|---|---|---|---|
| CSD `csd_fd_v5`, by SMILES | 327,294 | 744 | 766 | 130 |
| CSD `csd_fd_v5`, by exact refcode | 394,934 refcodes | 718 | 736 | 117 |
| QM9 | 133,641 | 94 | 96 | 96 |
| rc qm9rc train / held out | 114,817 / 6,652 | 25 / 10 | 25 / 10 | 25 / 10 |
| rc csdrc train / held out | 49,272 / 3,373 | 46 / 7 | 47 / 7 | 47 / 7 |
| rc zincrc, gdbrc, pubrc (both splits) | – | 0 | 0 | 0 |
| rc_oct22, five families pooled, train | 681,553 | 71 | 72 | 72 |
| rc_oct22, five families pooled, held out | 37,499 | 17 | 17 | 17 |

**Table 7.** Train against held-out, per rc family: two-sample Kolmogorov–Smirnov statistic D (unitless).
Molecules: one value per condition molecule. Crystals: every prior row, about 10 per molecule, so rows are not
independent and the effective sample is closer to the molecule count. At the molecule counts, the 1% critical D
for independent samples is about 0.021 (qm9rc), 0.029 (csdrc) and 0.017 to 0.018 (zincrc, gdbrc, pubrc).

| family | held-out molecules (count) | heavy atoms D | crystal density D | eLJ per crystal D |
|---|---|---|---|---|
| qm9rc | 6,652 | 0.060 | 0.027 | 0.015 |
| csdrc | 3,373 | 0.104 | 0.064 | 0.025 |
| zincrc | 9,142 | 0.010 | 0.005 | 0.008 |
| gdbrc | 9,553 | 0.005 | 0.004 | 0.006 |
| pubrc | 8,779 | 0.009 | 0.009 | 0.007 |

The rc held-out side is not an iid draw. `build_rc_prior.py::molecule_registry` holds a molecule out when any of
its spellings is held out in either selection file. That moves molecules listed in both selections onto the
held-out side: 563 of the 6,652 qm9rc held-out molecules and 721 of the 3,373 csdrc ones. Their mean heavy-atom count
is 7.45 (qm9rc) and 6.91 (csdrc), against 8.82 and 8.25 for the train side.

**Table 8.** Within-molecule structural spread: median pairwise envwise-RDF distance (unitless; mean per-channel
earth mover's distance, 10 Å cutoff, 100 bins) among up to 10 crystals per sampled molecule. Day crystals are
rebuilt in P1 from Day's own rigid conformer (`eval/day_bench` route); rc crystals are as stored.

| set | molecules (count) | pairs (count) | median pair distance (unitless) |
|---|---|---|---|
| Day, 10 lowest-energy, ≤ 9 heavy | 39 | 1,755 | 0.208 |
| Day, 10 lowest-energy, > 9 heavy | 40 | 1,800 | 0.399 |
| Day, 10 random within 8 kJ/mol, ≤ 9 heavy | 39 | 1,738 | 0.257 |
| rc held-out, per family (5 rows) | 80 each | 3,448 to 3,468 | 0.554 to 0.586 |

## Figures

Captions as they stand for the paper draft. The PNGs are in `D:/crystal_datasets/paper_review_nov02/fig/`. They
compare Day with the source datasets (CSD, QM9) and the proposed pool only. The earlier set, which also plotted the
rc_oct22 families and the single-system priors (former fig13 and fig15), is archived unchanged in
`D:/crystal_datasets/paper_review_nov02/fig_with_rc_2026-11-02/` with its `stats.json`; `--with-rc` regenerates
it. Tables 0, 4, 6, 7 and 8 above keep the rc numbers.
Histograms are density-normalised unless stated, and N in each legend is the number of molecules or rows behind
that curve.

- **fig00, sources.** One row per source: what it is, molecule and crystal counts, geometry, native energy and
  split.
- **fig01a/b/c, molecule size.** Heavy atoms, all atoms including H, and molecular weight per distinct molecule.
  Left: Day-1000 (bold), CSD `csd_fd_v5`, QM9 and the proposed pool. Right: Day against the pool split by the
  source each molecule was found in (a molecule in several sources counts in each). Values beyond the axis are
  clipped into the last bin (the CSD spike at 39 heavy atoms).
- **fig02a, element presence.** Share of each source's distinct molecules containing each element. Day is
  C/H/N/O/F by construction. `csd_fd_v5` is not organic-only: about 40% of its rows carry B, Si or a metal. The
  40,244 rows without an RDKit SMILES (mostly metal-bearing) are absent from every molecule-level CSD panel.
- **fig02b, heteroatom fraction.** (N + O + F) / heavy atoms per molecule.
- **fig03, functional groups.** Share of distinct molecules with at least one match per SMARTS pattern
  (`FG_SMARTS` in the script).
- **fig04a, rotatable bonds.** RDKit strict count per molecule. Day and the pool sit at 0.
- **fig04b, Day criteria.** Table 2 as a heat map.
- **fig04c, planarity.** Heavy-atom rms deviation from the best plane of each conformer; below 0.05 Å counts as
  planar. Day: one conformer per refcode (1,017). Pool: 4,849 of a seeded 5,000-molecule sample, embedded with
  ETKDGv3 and optimised with MMFF94 for this figure (the pool stores no geometry). The CSD spike at 1 Å is clipped
  large molecules.
- **fig05, bond lengths.** Per element pair, from 3D (bond if d < 1.15 × sum of covalent radii; no bond orders).
  Day: the rigid conformer in the first CIF block; CSD: deposited heavy atoms with regenerated H; QM9: DFT; pool:
  the fig04c sample. Values outside each panel's range are clipped into the end bins.
- **fig06, bond angles.** By central element, same molecules as fig05.
- **fig07, QM9 molecular energy.** U0 (Hartree) of 20,000 sampled QM9 molecules and its scaling with heavy atoms.
  No other source stores a molecular energy.
- **fig08, energy against density.** Day: FIT+DMA lattice energy (kJ/mol) of every structure up to 40 kJ/mol
  above its molecule's minimum, absolute and relative. Right: density only, for Day (all structures and global
  minima), all CSD Z′ = 1 crystals, and the CSD crystals of Day-like molecules outside Day (1,322 rows).
- **fig09, space groups and Z′.** Share of rows per space group, ordered by frequency over all Day structures, for
  Day (all, within 8 kJ/mol, global minima), all CSD Z′ = 1 crystals, and CSD Day-like crystals outside Day.
  Bottom: why CSD entries are not in `csd_fd_v5` (Z′ > 1 refused 60,877; special position 189,936;
  multi-component 197,295; other 30,536; of 873,578 ingested).
- **fig10, within-molecule diversity.** Left: crystals stored per molecule (Day, CSD). Middle: per-molecule
  standard deviation of Day crystal density (a scalar spread, not a structural distance). Right: Day's
  within-molecule RDF distances (Table 8, Day rows).
- **fig11, corner matrix.** One crystal per molecule (Day global minimum; CSD first entry, all of CSD and the
  Day-like subset outside Day): heavy atoms, MW, N + O count, rotatable bonds and density. At most 2,500 points per
  panel, integers jittered by ±0.3. Marginals clip values beyond the axis into the end bin.
- **fig12, energy correlates.** Day global-minimum FIT+DMA energy against heavy atoms (Spearman ρ = −0.64, 1,016
  refcodes), and Day relative energy against density.
- **fig14, overlap with Day.** Table 6 without the rc rows. The pool row is 0 by construction.
- **fig16, Day against the pool.** See "Biases of the pool and of Day" below.

## Biases of the pool and of Day

Measured by `condition_set_stats.py --steps shift` (fig16). "Pool" is the proposed pool of 152,462 molecules
described under Design. "In CSD", "in ZINC20-A" and "in PubChem" are its molecules found in that source (1,794,
45,566 and 135,153; a molecule can be in several).

**Table 9.** Two-sample classifier test: 5-fold cross-validated ROC AUC (mean ± standard deviation over folds) of a
class-balanced gradient-boosted classifier asked to tell Day's 988 molecules from up to 5,000 sampled molecules of
each set. Features: RDKit descriptors (heavy atoms, atoms, MW, rings, C/H/N/O/F counts and the fig03 group counts),
or count-Morgan fingerprints (radius 2, 2,048 bits). 0.5 means indistinguishable. The null row separates two random
pool samples and gives the noise floor.

| comparison | AUC, descriptors | AUC, fingerprints |
|---|---|---|
| null: pool vs pool | 0.49 ± 0.02 | 0.51 ± 0.01 |
| Day vs pool (all) | 0.85 ± 0.02 | 0.87 ± 0.02 |
| Day vs pool, in CSD | 0.79 ± 0.01 | 0.72 ± 0.02 |
| Day vs pool, in ZINC20-A | 0.90 ± 0.01 | 0.90 ± 0.01 |
| Day vs pool, in PubChem | 0.85 ± 0.02 | 0.87 ± 0.01 |

**Table 10.** Descriptors with the largest standardised mean difference between Day and the pool (Day minus pool,
in pooled-standard-deviation units; |SMD| < 0.2 is conventionally small). Shares are % of molecules with at least
one match.

| descriptor | SMD (Day − pool) | Day | pool |
|---|---|---|---|
| N atoms (count) | −0.46 | – | – |
| nitrile (% with one) | −0.45 | 0 | 11.1 |
| O atoms (count) | +0.42 | – | – |
| non-amide amine (%) | −0.37 | 4 | 20 |
| imine C=N (%) | −0.37 | 2 | 14 |
| carbonyl C=O (%) | +0.33 | 45 | 32 |
| 3-membered ring (%) | +0.22 | 13 | 9 |
| rings (count) | +0.22 | – | – |

**Table 11.** Nearest-neighbour count-Morgan Tanimoto, median per query set. Day queries: all 988. Pool queries:
1,000 sampled. "Rest of" excludes the query's own entry.

| query → reference | median NN Tanimoto |
|---|---|
| Day → pool | 0.70 |
| pool sample → rest of pool | 0.67 |
| Day → rest of Day | 0.50 |
| pool sample → Day | 0.34 |

Observations:

- A classifier separates Day from every pool source well above the noise floor. The smallest separation is from
  the pool's CSD part, the Day-like CSD molecules outside Day, at 0.72 to 0.79.
- Day carries no nitrile (0 of 988). 283 of the 1,794 Day-like CSD molecules outside Day (15.8%) carry one, and
  14 Day molecules carry an alkyne. Like CH₃ and OH (Trends), nitrile is absent from Day but common in its parent
  population.
- Coverage. A Day molecule's nearest pool neighbour is at least as close as a pool molecule's nearest other pool
  molecule (medians 0.70 and 0.67). Most of the pool is far from Day (median 0.34), and Day molecules are far
  from each other (0.50).

## Trends

- Size. Day's heavy-atom median (12) lies above the ceiling (9) of every generated source. Only 18.5% of Day
  molecules are within that ceiling. The CSD Z′ = 1 set has its heavy-atom median at 26.
- Rigidity. Day carries no CH₃ or O–H group in any molecule. This holds in the raw coordinates as well: none of
  Day's 1,509 oxygen atoms has a hydrogen within 1.15 Å in the asymmetric unit of its first CIF block, while 400
  nitrogens do. The CSD's own SMILES shows an OH for 15 Day refcodes, all of them charge-separated N-oxide-type
  structures where Day places the hydrogen elsewhere. Against its parent population: the CSD holds 7,901
  neutral, single-component C/H/N/O/F molecules with MW < 230 and zero RDKit rotatable bonds. 2,773 carry no CH₃,
  OH or NH₂ group, 3,787 carry CH₃, 1,991 OH and 710 NH₂ (groups overlap). Day's 988 molecules match 979 of the
  2,773 and 2 of the 710 NH₂ molecules, and none of the CH₃ or OH ones.
- Space groups among Day-like CSD crystals. The 1,322 Z′ = 1 crystals in `csd_fd_v5` of the 923 Day-like
  molecules outside Day fall 47.6% in P2₁/c, 19.0% in P2₁2₁2₁ and 7.1% in P-1 (fig09); median density 1.364 g/cm³.
  Day's global minima fall 46.4%, 9.0% and 11.0%, median density 1.326 g/cm³. The generated sources carry CH₃ in 52 to 68% of
  molecules and O–H in 31 to 34%. Under the rigid, Day-like criterion the rc families range from 2.6% (zincrc) to
  12.1% (qm9rc).
- Planarity. 29% of Day conformers are planar. QM9 has 12% and the rc conformers 1.7%. In the rc_oct22 selections,
  154 Day refcodes have their molecule selected. 89 reach the built files and 65 do not. 61 of the 65 have a planar
  heavy skeleton, against 16 of the 89. The builder leaves out molecules whose standard frame rests on a near-zero
  decision (`build_rc_prior.py::frame_margins`, default `--frame-margin` 1e-4). Its docstring names near-planar
  heavy skeletons as the main case. The script's margin, recomputed on Day's own CIF geometries, is below 1e-4 for
  521 of the 1,017 refcodes: 280 by the axis-overlap term (191 of them heavy-planar) and 363 by a tie between the
  farthest and second-farthest heavy atoms. Day geometries are symmetric, while the rc conformers are not, so these
  counts describe Day's geometry, not the rc build.
- Space groups. Day's within-8 kJ/mol window and its global minima concentrate in P2₁/c, as the CSD does. The rc
  priors are entirely P-1, the group of 11.0% of Day global minima.
- Density. Medians agree within 0.06 g/cm³ between Day global minima (1.33), CSD (1.37) and rc train priors (1.26
  to 1.32). The rc priors add a low-density lobe absent from Day's minima.
- Within-molecule spread. Stored rc crystals of one molecule are about twice as far apart in RDF distance (median
  0.55 to 0.59) as Day's ten lowest-energy structures (0.21 at ≤ 9 heavy atoms, 0.40 above).
- Energy scaling. Day's global-minimum energy falls with molecular size (ρ = −0.64), as rc's lowest eLJ does
  (ρ = −0.37).
- Leaks. 71 Day molecules are training conditions in rc_oct22, all from the qm9rc and csdrc families. 17 more are
  held-out conditions. `csd_fd_v5` holds the experimental crystal of 736 Day refcodes.

## Not covered

- The bias tests use RDKit descriptors and Morgan fingerprints only. The generator's own molecule encoder
  (Mo3ENet) embedding was not tested; it needs a conformer per molecule.
- The pool's geometry sample (ETKDG + MMFF94) is for these figures only; it is not the conformer route of any
  build.
- No energy is compared across sources. Day is FIT+DMA, rc is eLJ, the single systems are scaled eLJ or MACE, and
  no calibration between them was attempted.
- Day molecule keys come from bond perception, not the deposit. Charge-separated spellings make the overlap counts
  lower bounds. An InChI-skeleton match was run only for the 736 refcodes shared with `csd_fd_v5`.
- Which builder filter removed each of the 65 missing Day molecules was not determined per molecule. The margins
  on Day geometry above are not the margins of the rc conformers.
- The 166,395 figure of the Day paper was not reproduced.
- The rc_oct22 files read are the ones on D:, built at `668bf149` before the density bound. Cluster copies were not
  read.
- `blind_test_dataset.pkl`, the older CSD featurizations and the cap-thinned priors were not read.
- RDF distances use 40 to 80 molecules per set. The ≤ 9 / > 9 heavy-atom Day contrast confounds size with
  chemistry.
