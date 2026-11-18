# FD metric against COMPACK on Day landscapes (2026-11-18)

- **Status:** frozen record of one run (observation). It has no authority over code.
- **Plan:** `docs/design/fd_metric_calibration_plan.md` (simplified version).
- **Code:** GFN `eval/fd_analysis/day_calibration.py`, with MXtalTools `crystal_fd/metric.py` at `073b821d` (M1 + M2).
- **Output:** `D:/crystal_datasets/fd_analysis_oct26/day_calibration_nov18/` (summary.md, log.txt, pairs.pt, cif/,
  COMPACK caches).
- **Command:** `CUDA_VISIBLE_DEVICES=-1 python -m energy_sampling.eval.fd_analysis.day_calibration --out
  day_calibration_nov18 --n_mol 20` (seed 0). It took about 15 min on CPU, at about 1 s per COMPACK pair.

## Inputs

**Molecules.** 20 of Day's FIT+DMA landscapes, drawn at random (seeded) among molecules with no proper point symmetry.
13 landscapes were skipped because a C_n (n = 2..6) about a principal axis maps the molecule onto itself within
0.15 Å. 2 of the 20 are near-planar, so their distances take the molecular-mirror twin.

**Sets.** Each molecule in up to 3 of its landscape groups, one per crystal family, each group holding at least 4
low-energy structures (≤ 30 lowest within 10 kJ/mol). That gave 34 (molecule, group) sets:
- 20 monoclinic (P2₁/c);
- 4 triclinic (P-1);
- 10 orthorhombic (P2₁2₁2₁, Pbca).

No tetragonal, trigonal or hexagonal group had 4 or more low-energy structures for these molecules.

**Pairs.** 12 per set, 408 in all:
- 6 pairs of the set's lowest structure against rigid perturbations of itself, at amplitudes
  s = 0.005, 0.015, 0.04, 0.1, 0.25 and 0.6. Each moves the centroid by s Å, the orientation by s / r_g rad, the free
  lengths by s / 10 relative and the free angles by s / 10 rad.
- 6 random pairs of distinct low-energy structures.

**Scoring.** COMPACK on P1 unit-cell CIFs rebuilt from the fitted poses (pose RMSD ≤ 0.01 Å), on the repaired route:
20-molecule shells, CCDC defaults, and a match is 20/20. Against that, the glued FD distance (`GluedFDMetric`,
margin 1 Å, with the twin for planar molecules).

## Results

*Table 1. The glued FD distance against COMPACK, per crystal family, on 408 pairs (34 sets, 20 molecules).*
- "Scale" is FD / COMPACK RMSD on matched pairs; Spearman is on matched pairs; AUC is over all pairs of the family.
- The reference is the recalibration of 2026-11-17 on MIPCAS (P-1) and NEHZOR (P2₁/c): scale 1.17 / 1.30, every
  pair below 0.05 Å a match.

| family | pairs (count) | matches (count) | d < 0.05 Å: matched / all (count) | d at P(match) 0.9 / 0.5 / 0.1 (Å) | scale P10 / P50 / P90 (ratio) | Spearman, matched | AUC |
|---|---|---|---|---|---|---|---|
| triclinic | 48 | 19 | 8 / 8 | 0.331 / 0.478 / 0.692 | 0.94 / 1.06 / 1.13 | 0.993 | 0.995 |
| monoclinic | 240 | 100 | 40 / 40 | 0.442 / 0.565 / 0.724 | 0.98 / 1.08 / 1.18 | 0.994 | 0.999 |
| orthorhombic | 120 | 50 | 20 / 20 | 0.498 / 0.582 / 0.680 | 1.01 / 1.12 / 1.19 | 0.993 | 1.000 |
| all | 408 | 169 | 68 / 68 | 0.419 / 0.537 / 0.688 | 0.98 / 1.08 / 1.18 | 0.994 | 0.999 |

*Table 2. COMPACK outcome by glued FD distance, all families pooled (same 408 pairs).*

| glued FD bin (Å) | pairs (count) | matched (count) | P(match) | median COMPACK RMSD, matched (Å) |
|---|---|---|---|---|
| [0, 0.01) | 32 | 32 | 1.00 | 0.007 |
| [0.01, 0.05) | 36 | 36 | 1.00 | 0.021 |
| [0.05, 0.1) | 33 | 33 | 1.00 | 0.057 |
| [0.1, 0.2) | 32 | 32 | 1.00 | 0.140 |
| [0.2, 0.3) | 2 | 2 | 1.00 | 0.225 |
| [0.3, 0.5) | 35 | 33 | 0.94 | 0.351 |
| [0.5, 0.75) | 2 | 1 | 0.50 | 0.591 |
| [0.75, 1.0) | 29 | 0 | 0.00 | – |
| [1.0, 1.5) | 14 | 0 | 0.00 | – |
| [1.5, 3.0) | 25 | 0 | 0.00 | – |
| [3.0, ∞) | 168 | 0 | 0.00 | – |

## Reading

- **The plan's pass criteria hold in the three families sampled.** Every pair below 0.05 Å was a COMPACK match
  (68/68), and so was every pair below 0.3 Å (135/135). The median scale, 1.08, lies within the plan's band around
  the 1.17–1.30 measured on MIPCAS and NEHZOR.
- **The FD distance runs close to COMPACK's RMSD** (median ratio 1.06–1.12 per family) and ranks matched pairs the
  same way (Spearman 0.993–0.994). P(match) falls from 0.9 to 0.1 between about 0.4 and 0.7 Å.

**Limits of this run:**
- **Families:** no tetragonal, trigonal or hexagonal groups.
- **Molecules:** only molecules without proper point symmetry.
- **Pairs:** the matched pairs are synthetic rigid perturbations, not relaxation duplicates. The random pairs are
  almost all far apart (≥ 0.75 Å), so the AUC mostly measures the easy separation.
- **Transition region:** 0.2–0.75 Å holds only 39 pairs, because of the amplitude spacing.
