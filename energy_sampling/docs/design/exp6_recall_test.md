# Exp-6 recall test: plan (draft for owner approval, 2026-11-10)

Status: **proposal, not approved, nothing run.** It runs the test that `contingency_quiver.md` F3 states; its
answer feeds F1 (cheap-energy search, then MLIP relaxation of the kept picks) and the cheap-energy windowing of
noised children. The result note goes to `docs/research/exp6_recall_<date>.md`.

**Question.** After a rigid search with exp-6 (MXtalTools `analyze` key `exp6`: FIT parameters with MMFF charges,
`form='exp6'`, one package), MLIP relaxation of the exp-6 minima. Does that recover every MLIP basin within a few
kJ/mol of the MLIP minimum? Or does exp-6 systematically miss one motif: dense packing, directional H-bonds, or
π-stacking? eLJ runs alongside as the control, since eLJ is what pool8 used.

## Panel (rigid, the experimental conformer held fixed, one space group per molecule)

| molecule | motif under test | anchor (CSD) | space group searched | in Day's deposit |
|---|---|---|---|---|
| imidazole | N-H···N chain | IMAZOL06 (P2₁/c, Z′ 1, 103 K) | P2₁/c | yes |
| 2-pyridone | lactam N-H···O=C helix | PYRIDO13 (P2₁2₁2₁, Z′ 1, 123 K) | P2₁2₁2₁ | yes |
| uracil | N-H···O sheets with stacking | URACIL (P2₁/a, Z′ 1) | P2₁/c | yes |
| benzoic acid | carboxylic acid R²₂(8) dimer | BENZAC04 (P2₁/c, Z′ 1, 130 K, ordered) | P2₁/c | no |
| acetic acid | acid catemer (dimer vs chain) | ACETAC07 (Pna2₁, Z′ 1, 40 K) | Pna2₁ | no |
| naphthalene | herringbone, edge-to-face | NAPHTA04 (P2₁/a, Z′ 0.5) | P2₁ (the Z′ 1 subgroup) | yes |
| acridine | π-stacking with C-H···N | ACRDIN11 (P2₁/n, Z′ 1, 100 K) | P2₁/c | yes |
| norbornane | dense aliphatic, no polar group | none: a CSD name search found no ordered entry | P-1 | no |

Caption: the eight panel molecules, each with the motif it is chosen to stress. The CSD entries were read on
2026-11-10. Norbornane gets recall readouts only, unless the owner knows of a reference structure. None of the panel
is among Day's 16 MACE evaluation refcodes. The Z′ > 1 forms of acridine and 2-pyridone are out of scope.

## Arms (the same 2,000 random starts per molecule, same seeds, in every arm)

- **R, the reference:** each start relaxed by the MLIP only.
- **X, exp-6:** each start relaxed by exp-6. The distinct minima, at most 300, lowest exp-6 energy first, are then
  each relaxed by the MLIP.
- **L, eLJ:** as X, with eLJ.
- **D, Day-seeded (optional, free):** for the five Day molecules, Day's FIT+DMA minima within 8 kJ/mol in the
  searched space group, relaxed by the MLIP. This tests whether distributed multipoles recover what point charges
  miss, at no search cost.
- **Anchors:** each experimental structure relaxed by the MLIP, by exp-6 and by eLJ.

The basin set is the union of every arm's MLIP minima, deduplicated at the FD identity cut of 0.05 Å, so the
reference is scored against it too.

**Equal budget:** the cheap arms spend at most 300 MLIP relaxations per molecule and R spends 2,000. Arms are
compared on a common axis of MLIP relaxations spent: recall from the top K picks against recall from R's first K
starts. K = 32 is the pool8 band.

**Termination:** if more than one new in-window basin appears in R's last 500 starts, that molecule extends once to
4,000 starts, then stops and is reported as unsaturated.

## Readouts, per molecule (tables with labels and captions)

1. **Recall.** The basins within 5 and within 10 kJ/mol of the lowest MLIP basin that each arm contains, as counts,
   with the missed basins listed by MLIP rank. Recall is also given against K, the number of MLIP relaxations spent.
2. **Experimental form.** Whether each arm finds the anchor's MLIP basin. If so, its MLIP rank and its ΔE above the
   lowest basin, and its exp-6 and eLJ rank before the MLIP relaxation.
3. **Funnel.** How many cheap picks land in each MLIP basin: the in-window basins that get no pick, the share of
   picks taken by the most-visited basin, and the picks that land outside the window (wasted calls).
4. **Motifs** of the in-window basins, compared between missed and recalled basins, which is what answers
   "systematically":
   - H-bond contacts per molecule (H···N/O < 2.3 Å);
   - acid dimer versus catemer;
   - face-to-face versus edge-to-face aromatic contacts (ring centroids < 4.2 Å, plane angle < 20° versus > 60°);
   - packing coefficient.

## Stages and cost

| stage | work | where | cost |
|---|---|---|---|
| A, pre-flight | Build conformers from CSD. Round-trip the anchors through the builder and FD, including naphthalene in P2₁. Relax the anchors under all three energies. Check the MLIP noise floor (relax 50 minima from perturbed starts; is the FD spread well under 0.05 Å?). Measure the MLIP rate. | local CPU, GPU at ≤ 40 % | about 2 h wall; ≤ 1 GPU-h |
| B, pilot | Imidazole and acetic acid, all arms. Go/no-go on whether 2,000 starts saturate. | cluster | about ¼ of C |
| C, panel | The remaining six molecules. | cluster | see below |

Totals for B and C (estimated; MLIP rates measured on acridine at batch 96, so smaller molecules should run faster):
- **Cheap relaxations:** 8 × 2,000 × 2 = 32,000. eLJ runs at 0.75 relaxations/s/core, about 6 core-h. Exp-6 is
  unmeasured; assuming at most 5× slower, about 30 core-h.
- **MLIP relaxations:** about 21,000, or about 38,000 if every molecule extends.
  - MACE-OFF24m (732 per A100-h): about 30 A100-h, at most 52.
  - UMA-s 1.1 (about 75 per A100-h): about 280 A100-h, at most 510.
  - eSEN-S: rate unmeasured; stage A measures it.

## What the result decides

- **Exp-6 recalls the in-window basins and every anchor, on every motif:** F1 stands. The route is an exp-6 search
  with banded picks and MLIP relaxation of the picks only. Exp-6 windowing of the children stands too.
- **Exp-6 misses a motif:** the route is limited to molecules without that motif, or topped up with MLIP random
  starts, or the MLIP budget moves to F2.
- **The eLJ arm:** measures what the pool8 P-1 prior misses, which bears on rebuilding it.
- The recall threshold is the owner's call. The note reports counts, not a verdict.

**Not in scope:** flexible molecules, Z′ > 1, more than one space group per molecule (P-1 for all eight is the
natural extension), any training.
