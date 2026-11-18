# Exp-6 recall test: plan (draft for owner approval, 2026-11-10)

Status: **proposal. Reference MLIP: UMA-s 1.1 (owner, 2026-11-10, after stage A found that eSEN-S compresses
aromatic crystals), with the parity corrections below (owner, 2026-11-10). Stage A is done under them. Stages B and C
are not approved.** It runs the test that `contingency_quiver.md` F3 states; its
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

### Under UMA-s 1.1 (2026-11-10, after the owner's choice)

| molecule | UMA-s: FD anchor→end (Å) | ΔE relax (kJ/mol) | polish status |
|---|---|---|---|
| imidazole | 0.45 | −0.9 | converged |
| 2-pyridone | 0.46 | −9.1 | stalled |
| uracil | 0.91 | −14.2 | converged |
| benzoic acid | 3.62 | −25.0 | converged (closest H···O 1.28 Å) |
| acetic acid | 0.49 | +1.8 | converged |
| naphthalene | 2.38 | −22.2 | converged |
| acridine | 1.39 | −10.5 | converged |

Caption: UMA-s 1.1 relaxing each experimental structure, after the polar-axis fix (MXtalTools: `StageEnergy.held`
also holds a polar group's free centre coordinate, and `fgh` zeroes held rows, so minima in P2₁ and Pna2₁ no longer
stall on Hessian noise). FD is symmetry-aware (Å). Before the fix, exp-6's anchor minimum relaxed under UMA-s into the
UMA-s anchor basin (FD ≤ 0.06 Å) for all six molecules where exp-6 converged, and eLJ's did not for imidazole and
benzoic acid. That table was not rerun after the fix.

**Parity (resolved for this test, owner 2026-11-10).** UMA-s and eSEN-S are rotation- but not
reflection-equivariant: isolated rigid acetic acid and its mirror image differ by 7.5 kJ/mol under UMA-s, and a
2-pyridone P2₁2₁2₁ crystal and its mirror image by 3.9 kJ/mol per molecule. MXtalTools search keys (97061eb7):
`mlip_gas_reference: mirror_mean` on every UMA-s run, a constant mirror-mean isolated-molecule reference, so a
crystal's energy no longer depends on the description; `mlip_mirror_average: true` in the two chiral groups (P2₁,
P2₁2₁2₁), which averages the periodic energy over the crystal and its mirror image at 2× cost for those molecules.
In groups with an improper element no averaging is needed: the crystal is its own mirror image, so the part of the
error that changes sign under a mirror vanishes exactly. The part that doesn't is invisible to this check.

| molecule | UMA-s: FD anchor→end (Å) | ΔE relax (kJ/mol) | exp-6 end → UMA-s: FD (Å) / ΔE (kJ/mol) | eLJ end → UMA-s: FD (Å) / ΔE (kJ/mol) |
|---|---|---|---|---|
| imidazole | 0.45 | −0.9 | 0.024 / 0.00 | 0.546 / +0.28 |
| 2-pyridone | 0.85 | −8.5 | 0.057 / 0.00 | 0.127 / 0.00 |
| uracil | 0.91 | −14.2 | 0.022 / 0.00 | 0.017 / 0.00 |
| benzoic acid | 3.65 | −25.0 | 0.020 / 0.00 | 7.710 / +1.32 |
| acetic acid | 0.49 | −5.7 | 0.014 / +0.01 | 0.015 / 0.00 |
| naphthalene | 2.37 | −22.2 | 0.024 / 0.00 | 0.036 / 0.00 |
| acridine | 1.41 | −10.5 | 0.026 / 0.00 (from exp-6's stalled end) | 0.049 / 0.00 |

Caption: UMA-s 1.1 with both parity keys and both polish fixes. Second column: UMA-s relaxing the experimental
structure (all seven converged). Last two columns: UMA-s relaxing the minimum that exp-6 or eLJ reached from the
experimental structure, as FD (Å) and energy relative to UMA-s's own minimum from the experimental structure; under
0.05 Å is the same basin. One relaxation per cell. Reading: from the experimental structure, exp-6 always hands UMA-s
a start in UMA-s's experimental basin; eLJ fails for imidazole and benzoic acid. This is the F1 question at one start
per molecule, not the test. Open on UMA-s itself: benzoic acid (3.65 Å, closest H···O 1.28 Å) and naphthalene
(2.37 Å) move far from experiment under the rigid-molecule UMA-s surface.

**Mirror test (observation, 2026-11-18; `eval/exp6_recall/mirror_test.py`, outputs
`D:/crystal_datasets/exp6_recall_nov10/mirror_test_nov18/`).** Is the part of the error that changes sign under a
mirror a constant offset in a chiral group? Structures: the 60 lowest minima of a 600-start exp-6 search per molecule.
R = raw UMA-s 1.1 lattice energy (mirror-mean gas reference), σx = the crystal inverted through the origin (same
group). Construction check: 2·(averaged energy) − R(x) reproduces R(σx) to ≤ 0.011 kJ/mol.

| molecule (group) | R(σx) − R(x), mean ± sd [range] (kJ/mol), 60 structures | spread of R(x) (kJ/mol) | rank correlation R(x) vs R(σx) | same lowest structure | paired raw-UMA-s minima: FD median / max (Å), pairs > 0.05 Å | minimum-energy difference, mean ± sd [range] (kJ/mol) |
|---|---|---|---|---|---|---|
| naphthalene (P2₁) | 0.00 ± 0.08 [−0.30, 0.29] | 16.8 | 0.999 | yes | 0.019 / 0.76, 2 of 20 | 0.02 ± 0.27 [−0.51, 0.51] |
| 2-pyridone (P2₁2₁2₁) | −0.34 ± 1.88 [−5.34, 4.51] | 36.3 | 0.976 | no | 0.21 / 2.57, 19 of 20 | −0.06 ± 4.38 [−8.48, 9.98] |
| acetic acid (P2₁) | 7.97 ± 2.28 [2.23, 12.07] | 56.6 | 0.983 | no | 0.13 / 1.75, 19 of 19 | 8.15 ± 3.97 [2.84, 20.05] |

Caption: raw UMA-s single points on 60 structures per molecule, and raw UMA-s relaxations of x and of σx from the 20
lowest (the mirror of σx's minimum compared with x's minimum; FD = symmetry-aware FD, metric M2, ~1.2× COMPACK RMSD;
< 0.05 Å = same basin). One search per molecule. Reading: for the non-polar aromatic the error is negligible; for the
two hydrogen-bonding molecules it is not an offset: it varies by several kJ/mol between structures, changes which
structure is lowest, and moves the relaxed minima (some into a different basin, up to 2.6 Å). Raw UMA-s in a chiral
group is therefore not usable for polar molecules without the mirror average. The part of the error that does not
change sign under a mirror remains unmeasured.

**MACE-OFF24 against UMA-s 1.1 (observation, 2026-11-18; `eval/exp6_recall/mace_vs_uma.py`, outputs
`D:/crystal_datasets/exp6_recall_nov10/mace_vs_uma_nov18/`).** Agreement, not correctness: neither is a reference.

| structures compared | n | rank correlation (Spearman) | sd of the difference in relative energy (kJ/mol) | max abs difference (kJ/mol) | same lowest | UMA-s top 10 in MACE top 10 |
|---|---|---|---|---|---|---|
| imidazole: UMA-s basin minima within 15 kJ/mol, MACE single points | 164 | 0.61 | 8.1 | 20.4 | no (UMA-s's lowest is MACE's 2nd) | 9 |
| imidazole: the 30 lowest of those, relaxed under MACE-OFF24 | 30 | 0.69 | 4.1 | 10.5 | yes | 6 |
| 2-pyridone (P2₁2₁2₁): 60 exp-6 minima, vs mirror-averaged UMA-s | 60 | 0.40 | 12.2 | 29.4 | no | 1 |
| naphthalene (P2₁): same | 60 | 0.21 | 7.3 | 17.9 | no | 2 |
| acetic acid (P2₁): same | 60 | 0.65 | 16.8 | 42.4 | no | 5 |
| for scale: raw vs mirror-averaged UMA-s, acetic acid | 60 | 0.995 | 1.1 | 2.9 | no | 9 |

Caption: relative energy = each model's lattice energy minus its own lowest in the set; sd of (MACE − UMA-s) relative
energies after removing their mean. One search per molecule. The chiral-group rows are at exp-6 minima, off-minimum for
both models. Under MACE-OFF24 the 30 imidazole UMA-s minima all move (FD median 0.39 Å, max 1.80 Å) and merge into 19
minima; both models put the experimental structure lowest after relaxation. Reading: the two models disagree by
several kJ/mol on relative energies, about ten times the mirror error of UMA-s; the choice between them needs a
higher-level reference. Separately, UMA-s absolute lattice energies are inflated by a per-molecule constant
(naphthalene −151, acetic acid −154 kJ/mol at the experimental structures against reference values near −82 and −73):
the gas leg uses the `omc` task without periodic boundaries, and OMC25 holds crystals only. For a rigid molecule the
constant cancels in every within-molecule comparison.
