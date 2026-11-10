# Exp-6 recall test: plan (draft for owner approval, 2026-11-10)

Status: **proposal. Stage A done (2026-11-10). The owner chose eSEN-S as the reference on 2026-11-10; stage A then
found that eSEN-S compresses aromatic crystals (below), so the reference is back with the owner. Stages B and C are
not approved.** It runs the test that `contingency_quiver.md` F3 states; its
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

## Stage A observations (2026-11-10, local GPU at 40%)

Code: `eval/exp6_recall/` (`panel.py`, `make_configs.py`, `run_capped.py`, `basins.py`, `stage_a.py`). Data:
`D:/crystal_datasets/exp6_recall_nov10/` (`panel/`, `stageA_anchors/stage_a.json`). Anchor changes from the panel
table: BENZAC12 (BENZAC04 has no coordinates) and ACRDIN04 (ACRDIN11 has R = 20.6%). All seven anchors convert to
Z' = 1 in the searched group with residual ≤ 0.001 Å.

Two defects found and fixed on the way. First, the MLIP polish let a monoclinic cell's α and γ drift, so every eSEN-S
minimum in P2₁/c failed the cell-box check: MXtalTools `minima.py::StageEnergy.held`, see the
[crystal-search-method](../wiki/crystal-search-method.md) wiki page. Second, exp-6's float32 energy repeats only to
5.5×10⁻⁵ kT, so its polish stalled at the default tolerance of 10⁻⁴ kT; these configs set 10⁻³ kT.

| molecule | eLJ: FD anchor→end (Å) | exp-6: FD (Å) / ΔE (kJ/mol) | eSEN-S: FD (Å) / ΔE (kJ/mol) |
|---|---|---|---|
| imidazole | 1.86 | 0.32 / −3.8 | 0.28 / −0.8 |
| 2-pyridone | 3.25 | 3.28 / −2.6 | 0.84 / −15.9 |
| uracil | 1.22 | 1.12 / −8.2 | 0.48 / −15.9 |
| benzoic acid | 2.78 | 0.77 / −7.6 | 0.64 / −31.5 |
| acetic acid | 1.65 | 0.99 / −3.8 | 0.39 / −19.9 |
| naphthalene | 1.08 | 0.37 / −0.9 | 2.25 / −85.5 (stalled) |
| acridine | 1.39 | 0.25 / −1.6 (stalled) | 1.54 / −48.8 |

Caption: each energy relaxing the experimental structure. Rigid conformer, H normalised, one relaxation per cell. FD
is the symmetry-aware identity distance from the anchor to the polished end; 0.05 Å means the same structure. ΔE is
the end's energy minus the anchor's, per molecule. Compare the columns: how far each energy pulls the experimental
form.

| molecule | cell scale | eSEN-S, our route | eSEN-S, stock fairchem | UMA-s 1.1, stock fairchem |
|---|---|---|---|---|
| naphthalene | 0.97 / 0.94 / 0.91 | −21.4 / −38.1 / −40.9 | −21.4 / −38.1 / −40.9 | −7.3 / +29.1 / +98.6 |
| acridine | 0.97 / 0.94 / 0.91 | −22.2 / −35.6 / −16.9 | −22.2 / −35.6 / −16.9 | −6.5 / +18.0 / +104.9 |
| imidazole | 0.97 / 0.94 / 0.91 | +12.5 / +25.6 / +64.2 | +12.5 / +25.6 / +64.2 | +9.8 / +26.0 / +63.5 |

Caption: periodic energy per molecule (kJ/mol) relative to the experimental cell, with the cell lengths scaled
uniformly at fixed fractional centres and orientations. "Stock" is fairchem's `FAIRChemCalculator` (task `omc`) on
the same unit-cell atoms. Compare eSEN-S with UMA-s 1.1: eSEN-S's energy keeps falling down to about 25% volume
compression for the two aromatics, not for imidazole. Our route matches stock to 0.01 kJ/mol, so this is the
checkpoint's behaviour, not our code's.

Reading: exp-6 holds four of the seven experimental forms near their own geometry (FD ≤ 0.4 Å). It loses 2-pyridone
(3.3 Å) and moves uracil and acetic acid about 1 Å. eLJ moves every one by more than 1 Å. eSEN-S's minima for
naphthalene and acridine are compressed artefacts. eSEN-S's absolute lattice energies are 1.3–3.3× experimental
sublimation enthalpies, which is likely its isolated-molecule reference; that is constant for a rigid conformer and
harmless to rankings.
