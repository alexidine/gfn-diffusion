# Broadening the FD metric's calibration: plan

- **Status:** PROPOSAL, 2026-11-18. Owner go to plan it on the same date. Nothing is built or run under this plan.
  It has no authority over code, config or tests.
- **Goal:** make the glued FD distance (MXtalTools `crystal_fd/metric.py::GluedFDMetric`, MXT `073b821d`) a result
  that stands on its own, for the paper and its SI. That means calibrating it against COMPACK on populations a referee
  would accept, and positioning it against the established alternatives.
- **Starting point** (`docs/design/fd_si_draft.md` §3, §7.3; recalibration of 2026-11-17):
  - It is invariant under every re-description, to ≤ 2e-7 Å in all 26 Day groups.
  - It is symmetric and exact locally. Since M1 and M2 it is also exact across cell faces.
  - It reads 1.17× / 1.30× COMPACK RMSD, with Spearman 0.997 / 0.981 and AUC 0.933 / 0.994. Every pair below 0.3 Å
    was a COMPACK match.
  - **All of this rests on two molecules (MIPCAS in P-1, NEHZOR in P2₁/c), on pairs drawn on decision boundaries.**

## 1. What a referee will ask, and what answers it

| question | today | what answers it |
|---|---|---|
| Does it hold beyond two molecules and two groups? | no evidence | many molecules × the Day groups, every crystal family (§3, design D) |
| Does it hold for symmetric molecules? | only the planar mirror is quotiented | a point-symmetry quotient (§2, step 0), then symmetric molecules in the panel |
| What happens on an unselected population? | pairs drawn on boundaries only | randomly drawn pairs alongside the stratified ones (designs R and N) |
| Where is the identity cut valid? | 0.05 Å on two molecules | P(match \| d), and every pair below the cut a match, per family and per symmetry class |
| Why not COMPACK, RDF, AMD/PDD or StructureMatcher? | not compared | the same pairs scored by each, with run time per pair (§4) |
| Is the scale stable? | 1.17 vs 1.30 | the scale per family and per molecule class, with a spread |

## 2. Work before any COMPACK run

- **Step 0. Molecular point symmetry in the metric** (MXtalTools; owner decision D2).
  - Today a molecule with a proper point group P⁺ sits at |P⁺| points per crystal (SI draft §4.3), and the metric
    sees them as distinct.
  - The quotient is cheap. The rotation term becomes a minimum over R_B·p for p in P⁺, the body-frame rotations that
    map the molecule onto itself; the cell and centroid terms are unchanged.
  - P⁺ would come from the molecule's coordinates, as rotations that permute like atoms within a tolerance, like
    `molecule_info`'s mirror fit.
  - Tests: re-descriptions by p read 0, and a C₂ or C₃ molecule's copies collapse to one point.
  - Without this step, the symmetric part of the panel can only be reported as "σ copies apart".
- **Step 1. Finish M2's leftovers.** Rerun the full nearest-neighbour identity scan (`identity_calibration`, GPU,
  ~20 min) and update the SI draft's §3.2 and §7. This keeps the draft consistent before new numbers arrive.
- **Step 2. A pair builder for Day landscapes** (GFN `eval/fd_analysis/`, new). It does three things:
  - reads Day CIF blocks, with the parser and pose fit that attack A's oracle already proved: pose RMSD ≤ 2e-14 Å,
    operators equal to `SYM_OPS`;
  - writes repaired CIFs for COMPACK through `common.compack_pair_repaired`, the route already used;
  - draws pairs by the designs of §3.

## 3. Panel and pair designs

**Panel.** Drawn from Day's 1,017 FIT+DMA landscapes:
- 36 molecules, split into three symmetry classes of 12: C₁; planar or C_s; and a proper axis (C₂, C₂ᵥ, C₃, …).
- Each molecule in 4 of its landscape groups, one per family where it has one: triclinic (P-1), monoclinic
  (P2₁/c or C2/c), orthorhombic (P2₁2₁2₁ or Pbca), and one of tetragonal or trigonal (88, 148, …).
- MIPCAS and NEHZOR stay in as anchors to the existing calibration.

That gives about 144 (molecule, group) sets. The classes would be balanced by drawing molecules at random within each
class (seeded) after the class is assigned by the point-group detector of step 0.

**Pair designs.** Each (molecule, group) set gets about 40 pairs:

| design | pairs per set | how drawn | what it measures |
|---|---|---|---|
| P: perturbed | 16 | one low-energy minimum against itself perturbed rigidly, in centroid, orientation and cell strain, at amplitudes log-spaced so the FD distance spans 0.005–1.5 Å | P(match \| d) and scale on a continuum, in every group |
| N: nearest distinct | 8 | each of 8 low-energy minima against its FD-nearest other minimum in the landscape | hard negatives; the gap the identity cut must sit in |
| R: random | 12 | random pairs of minima within 10 kJ/mol | AUC and false-match rate on an unselected population |
| X: re-described | 4 | a minimum against a scrambled re-description of itself (attack A's scrambler) | controls: COMPACK must match 20/20, FD ≈ 0 |

Total: 144 × 40 ≈ 5,800 pairs. At the measured 1.5 s per pair, that is about 2.4 h on one worker, or about 40 min on 4.

A second population, if the owner wants relaxation-noise duplicates from our own search (owner decision D3): the
`pool8sg_nov15` minima libraries (7 groups, cluster), with the same N and R designs plus pairs below the library's
merge cut. These are the pairs the 0.05 Å cut actually decides in production.

## 4. Comparators

Every pair is scored by each of these, with wall time per pair on the same machine:

- **FD:** glued FD distance, plain and with the twin or point-symmetry quotient.
- **RDF distance:** the campaign metric, `crystal_rdf.compute_rdf_distance`.
- **AMD/PDD** (Widdowson & Kurlin): the `average-minimum-distance` package on PyPI. Installing it is owner decision D1.
- **pymatgen `StructureMatcher`**, if it is installed in an environment we use. To be checked; it is not proposed as a
  new install.
- **COMPACK:** the reference, 20-molecule shells, CCDC defaults, the repaired route.

## 5. What gets reported

Each item has its acceptance reading (all [measured], into `docs/results/` as a frozen battery record):

1. **P(match | d)**, a logistic in log d, per family and per symmetry class. Pass: P ≥ 0.95 below 0.05 Å in every
   stratum, and no COMPACK non-match below 0.05 Å anywhere.
2. **Scale** (FD / COMPACK RMSD on matched pairs), P10 / P50 / P90 per stratum. Pass: P50 within 0.9–1.5 in every
   stratum, so that one sentence covers it.
3. **Rank correlation** with COMPACK RMSD on matched pairs (Spearman, Pearson), per stratum.
4. **Separation:** AUC and recall at 2% false matches, on design R (unselected) and on design N (hard), per stratum
   and pooled, for each comparator.
5. **Controls:** design X matches 20/20 by COMPACK and reads FD < 1e-6 Å. A failed control stops the report.
6. **Cost:** wall time per pair for FD (CPU and GPU), RDF, AMD/PDD and COMPACK.

## 6. Stages, cost and stopping rules

| stage | what | cost | stop if |
|---|---|---|---|
| 0 | point-symmetry quotient in the metric, with tests | ~half a day of code; tests ~1 min | the detector disagrees with a reference point group (RDKit or pymatgen) on the panel |
| 1 | identity scan rerun, SI update | ~20 min GPU | – |
| 2 | pair builder; a smoke of 3 molecules × 2 groups × 40 pairs (240 pairs) | ~10 min COMPACK | any X control fails; COMPACK fails on more than 2% |
| 3 | the full panel, ~5,800 pairs | ~40 min on 4 workers (sized on the smoke) | – |
| 4 | comparators on the same pairs | AMD/PDD minutes; RDF ~1 h CPU | – |
| 5 | report into `docs/results/`, SI §3 updated | – | – |

Each run is announced with its cost before it starts (owner rule, 2026-11-03).

## 7. Owner decisions

- **D1. Install the `average-minimum-distance` package** for the AMD/PDD comparison? **Recommended: yes.** It is the
  comparator a referee will name first. Without it, §4 is COMPACK and RDF only.
- **D2. Build the point-symmetry quotient (step 0) before calibrating?** **Recommended: yes.** Otherwise symmetric
  molecules can only be reported as "σ copies apart", and the metric can't be claimed as an identity test for them.
- **D3. Add the `pool8sg` search minima** (cluster, needs a transfer) as a second population? **Recommended: yes,
  after the Day panel.** It is the production regime of the 0.05 Å cut, but the Day panel answers the referee
  questions on its own.

## 8. What this plan does not do

- No change to the FD chart, canonicalisation or measure.
- No cross-space-group comparison: distances are within one molecule and one group, as the metric is defined.
- Z′ > 1 is out of scope, as everywhere in the FD tooling.
