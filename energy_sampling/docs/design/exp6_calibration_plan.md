# Exp-6 calibration cycle: fit exp-6 to the MLIP, then test it as a search surrogate (plan, 2026-11-24)

Status: **plan for owner review. The scripts are built and smoke-tested end to end on tiny sizes (2026-11-25: labels
reproduce the searches' own minimum energies to < 0.001 kJ/mol; the fit path reproduces the labelled FIT energies;
the arms' starts are identical). A reduced-size prototype ran 2026-11-25 (16 fit molecules × 40 exp-6 starts, 4 × 8
MLIP starts; panel of 3 × 16 starts, 8 picks; outputs and `compare.json` in
`D:/crystal_datasets/exp6_calibration_proto_nov25`); no full-size run has started.** The cycle's decisions (refit first, trajectory data with thinning, energies + forces weighted to the low
end and to the order of minima, held-out molecules, no Day benchmark structures, same starts in both arms, an
optional loop) are the owner's, relayed 2026-11-24. The sizes, loss weights, thresholds and where it runs are
proposals.

**Question.** The paper's headline cost is total MLIP GPU time. If an exp-6 field tuned toward the MLIP finds the
MLIP's low-energy basins when its minima are handed to the MLIP for a single point or a relaxation, the MLIP search
can be replaced by a cheap search plus MLIP endpoints. Does it, and at what MLIP cost per basin found?

**MLIP.** MACE-CSP-25.2 large + D3(BJ, PBE), the current leaning (not yet a decision). Its atom-overlap holes
(10 of 150 search starts fell into one, `docs/results/mace-csp-transfer-2026-11-23.md`) are kept out of the fit
data by a contact filter (shortest intermolecular contact below 1.2 Å). The R arm's own minima stay clean because the
polish's minimum checks reject the collapsed rows; that held on the 131 minima of the transfer check.

## Molecules

| set | source | count | conformer | space group |
|---|---|---|---|---|
| fit | pool v1 `train` split, 5–16 heavy atoms, neutral | 200 (15% held back inside the fit for validation) | RDKit ETKDG seed 0 + MMFF94, rigid | alternating P2₁/c and P-1 |
| comparison panel | pool v1 `heldout_scaffold` split, same filters | 24 | same | same |

Caption: the molecule sets (`eval/exp6_calibration/common.py::selection`, seeded). No panel molecule shares a scaffold
with a fit molecule. None of Day's 989 benchmark molecules is in pool v1 (InChIKey check, 2026-11-24), so no Day
structure enters the fit.

## Steps

1. **Exp-6 trajectories** (`collect.py exp6`). Per fit molecule, an exp-6 (FIT + MMFF) search, 200 starts, every
   evaluated state recorded (the stream's `trajectories`).
2. **MLIP trajectories** (`collect.py mlip`). Every fifth fit molecule (40), a MACE-CSP search, 25 starts, recorded the
   same way: the MLIP's own basins and descent paths.
3. **Frames and labels** (`collect.py frames`, `label`). Per search, states at steps 0, 1, 2, 4, 8, … and each stage's
   last step, then at most 10 per energy band above the search's lowest state (0–2–5–10–20–40–80 kJ/mol), so the
   near-minimum frames do not dominate; plus its minima within 20 kJ/mol (at most 30). Each frame is labelled with the
   MLIP energy and gradient in the 12 cell parameters (the coordinates the search moves), the FIT exp-6 energy and
   gradient, and the shortest intermolecular contact. All evaluations go through `minima.StageEnergy`, the search's
   own path.
4. **Fit** (`fit.py`). The FIT set's per-type (A, B, C) for C, H on C, H on N, H on O, N, O and F, plus one scale on
   the MMFF charges: 22 numbers, started at FIT. Loss within each molecule (a molecule's constant offset, its gas leg,
   cancels): squared energy residual and normalised gradient residual, both weighted by
   max(exp(−(E − E_lowest)/10 kJ/mol), 0.02); a logistic penalty on every pair of MLIP minima more than 0.5 kJ/mol
   apart that exp-6 puts in the wrong order; a weak prior toward FIT. Starting weights: gradient 0.1, order 1, prior
   0.01, chosen on the validation molecules. The fit refuses to start unless its path reproduces the labelled FIT
   energies, and stops if no gradient reaches any parameter group. Output: a parameter file (`exp6_params/1`) the
   search reads directly (MXtalTools 3b8b7ef2).
5. **Comparison** (`compare.py`). Per panel molecule, the same 200 starts (same seed) in three arms. Every arm runs
   at the same slot count, because the stream draws a condition's starts in seeded blocks of min(slots, N); the
   report checks the shared starts from the searches' first recorded states.

   | arm | search | MLIP work |
   |---|---|---|
   | R | MACE-CSP | every start relaxed and polished |
   | T | tuned exp-6 | its distinct minima within 15 kJ/mol, at most 64, lowest first: one MLIP single point each, then each relaxed and polished |
   | U | FIT exp-6 (control: what the tuning bought) | same as T |

   Caption: the comparison's arms. Reference: every MLIP minimum of R, T and U, grouped into basins (symmetry-aware FD
   < 0.05 Å). Readouts per molecule: the in-window basins (5 and 10 kJ/mol) each arm finds; the same for the first K
   picks against R's first K starts (K = 8, 16, 32, 64); MLIP evaluations spent and per in-window basin found; the
   single-point rank of the pick that relaxes into the lowest basin.
6. **Loop (optional).** Molecules where T misses basins that R finds: their R trajectories join the fit data, refit,
   rerun T on the panel only.

## Cost

| work | size | rate measured (local GPU at 40%, shared) | estimate |
|---|---|---|---|
| exp-6 searches, fit set | 200 × 200 starts | 0.12–0.21 s per start (3 searches, FIT, 2026-11-18) | 2 GPU-h |
| MLIP searches, fit subset | 40 × 25 starts | 11–33 s per start, median 20 (10 searches, MACE-CSP L + D3, 2026-11-23) | 6 GPU-h |
| labels | ≈ 24,000 frames | ≈ 0.1 s per MLIP energy + gradient (same searches: 50–150 ms per evaluation) | 1 GPU-h |
| fit | 1,500 steps | 0.2 s per step at 2 molecules × 4 frames (smoke, 2026-11-25); scales with molecules × frames per step | ≤ 1 GPU-h |
| comparison, R | 24 × 200 starts | 20 s per start | 27 GPU-h |
| comparison, T and U | 24 × 2 × (200 exp-6 starts + ≤ 64 MLIP relaxations) | relaxation from a minimum assumed half a random start, ≈ 10 s | 9 GPU-h |

Caption: GPU time on the local card under the 40% memory share, from the search effort files named. The MLIP rate
includes out-of-memory retries at that cap. Whole-card or cluster rates are unmeasured; MACE-CSP and torch-dftd are
not installed on the cluster yet. Total about 45 local GPU-hours, of which the comparison's R arm is 60%.

## Owner decisions open

- **Pass rule.** Proposed: tuned exp-6 + MLIP relaxation finds at least 90% of the in-window (10 kJ/mol) basins R
  finds, pooled over the panel, at no more than a third of R's MLIP evaluations. Single-point endpoints are reported,
  not judged.
- **Where it runs.** Locally (about two days at the 40% share), or the comparison's R arm on the cluster after the
  MLIP and D3 are installed there.
- **Sizes.** 200 fit molecules and 24 panel molecules at 200 starts are proposals; the comparison's cost scales with
  panel size × starts.

## Not covered

Z′ > 1; flexible conformers (the rigid ETKDG conformer only); space groups beyond P2₁/c and P-1; whether the tuned
field transfers to molecules outside pool v1's chemistry.
