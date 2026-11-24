# Energy-fidelity program: is MACE-CSP + D3 safe on top, and does cheap seeding recover its basins? (draft 2, 2026-11-19)

Status: **not adopted.** The owner rejected this program as too expensive on 2026-11-23; the two smaller checks run
instead are recorded in `docs/results/mace-csp-transfer-2026-11-23.md`. Kept as history. Draft 1 was reviewed the same day by four critics
(a student who would run it, a PI, a physical chemist, a JCIM referee); this draft is the revision. Successor to the
local work in `exp6_recall_test.md`, whose panel conformers carry a hydrogen-placement defect (fixed in GFN 00028573).
Results go to `docs/results/` as a frozen record and a note under `docs/research/`.

## What changed from draft 1, and why

- **D1 is now one-sided.** MACE-CSP-25.2 + D3(BJ, PBE) and UMA-s both derive from PBE + D3, though with different
  damping (OMC25 uses zero damping, `IVDW = 11`), so their agreement (1.1 kJ/mol median on 16 polymorph pairs) does not
  show that either is right, and disagreement could not say which is wrong. Since they agree, the choice rests on cost (about 4× per relaxation, 8× in Sohncke groups
  where UMA-s is mirror-averaged), exact mirror symmetry, and whether MACE-CSP fails where it was not trained. The
  question becomes: **is MACE-CSP + D3 safe wherever GFN would use it?** It is the default; failures against an
  independent reference overturn it.
- **Independent references replace model-vs-model agreement:** the experimental structures (free), and a small number
  of periodic PBE-D3 single points chosen where the models disagree or a hole is flagged.
- **UMA-s random searches are cut** (about 210 A100-h, 85% of draft 1). UMA-s re-relaxes MACE-CSP's low basins instead.
- **Holes on out-of-selection molecules are the headline**, on molecules drawn from the GFN training pool, not CSD
  polymorph pairs; the hole criterion is fixed per molecule in advance.
- **One primary endpoint per decision, thresholds written down before anything runs** (proposed below; the owner sets
  the numbers).
- **Cut:** UMA-m (same training data, not a reference), the room-temperature-form readout (0 K rigid lattice energy
  against a free-energy observation), per-functional-class splits (about 2 molecules per class), Part P beyond triage.

## Decisions and their primary endpoints

| id | decision | primary endpoint | default |
|---|---|---|---|
| D1 | top-fidelity energy | holes and anchor failures of MACE-CSP + D3 on out-of-selection pool molecules, against UMA-s and the experimental structure | MACE-CSP + D3 |
| D2 | F1 (exp-6 search, MLIP relaxation of its picks) is the method | recall of the top energy's basins within 5 kJ/mol at K = 32 MLIP relaxations, paired against MLIP random starts at equal K | open |
| D3 | ask the MACE-CSP authors for a fine-tune | holes or disagreements concentrated in out-of-selection molecules; the failing structures are the data request | no |

## Step 0: prerequisites (local, hours)

1. **Rebuild conformers and anchors with corrected hydrogens** (panel, probes, polymorph pairs) into new folders.
2. **D3 cutoff:** 15 Å shifts the D3 lattice term by 0.2–1.1 kJ/mol depending on density, the size of the agreement
   being measured. Use 30 Å, or 15 Å with a tail correction, after measuring both on the anchors.
3. **Damping check (done 2026-11-23):** OMC25's dispersion is D3 with zero damping, not Becke–Johnson, so MACE-CSP
   needs zero-damped D3 to target the same functional as UMA.
4. **Training overlap:** list which program molecules are among MACE-CSP's 1,617 training molecules (ask the authors
   for the list); report inside/outside and seen/unseen separately.
5. **Long-range electrostatics flag:** per crystal, the MMFF-charge Ewald energy minus the same sum truncated at 9 Å
   (the reach of both models). Crystals where it changes a polymorph difference by more than 1 kJ/mol are flagged.
6. **Noise floor:** relaxation repeatability from perturbed restarts on 3 molecules; differences below it are noise,
   and the FD identity cut (0.05 Å) is checked to sit above it.

## Step 1: pilot (cluster, about 8 A100-h): run first

Four CH₃-bearing molecules from the training pool, each with a Z′ = 1 CSD crystal (one with an amide or ester
torsion), plus the probe set's acetic acid.
- MACE-CSP + D3 search of 1,000 random starts in the anchor's space group; the anchor relaxed.
- Cell-compression scan of the anchor (it caught eSEN-S in stage A).
- MACE-CSP's basins within 10 kJ/mol (at most 40) re-relaxed under UMA-s.
- **Hole criterion, fixed per molecule in advance:** a minimum that is (a) lower than the relaxed anchor by more than
  10 kJ/mol and (b) has an intermolecular contact shorter than the anchor's shortest contact of the same kind by more
  than 0.15 Å, or a packing coefficient outside the anchor's ±0.05; or a MACE-CSP basin that relaxes under UMA-s to a
  structure more than 10 kJ/mol above where MACE-CSP put it relative to the anchor.
- **Abort rule:** holes in ≥ 2 of the 5 molecules stop Steps 2–3; the failing structures go to the authors (D3), and
  D1 reverts to UMA-s.

## Step 2: safety and basin correspondence (cluster, about 35 A100-h)

16 molecules drawn by a stated stratified random rule from the training pool's CSD-anchored molecules: 8 outside the
MACE-CSP selection (CH₃, torsions), 8 inside; one seed each, plus a second seed on 2 molecules.
- As the pilot, per molecule.
- **Readouts:** holes per molecule (k/8 outside, k/8 inside, exact binomial intervals: 0/8 bounds the failure rate
  below about 31% at 95%, so this detects gross failure, it does not certify); the anchor's basin found, its rank and
  ΔE above the lowest under MACE-CSP and under UMA-s; the relaxed anchor's FD from the experimental structure and its
  cell-volume change; Good-Turing missing mass of each search; the cross-relaxation table (landed on a found basin,
  ended below MACE-CSP's lowest, merges).
- **Reference single points (negotiable; the authors may run them):** 15–20 periodic PBE-D3(BJ) single points set up
  like MACE-CSP's training data, on the structures where MACE-CSP and UMA-s disagree most and on every flagged hole.
  This settles which model copies PBE-D3 where they diverge, and whether a hole belongs to PBE-D3 or to a model.

## Step 3: exp-6 seeding recall (cluster, about 10 A100-h + 30 core-h)

The same 16 molecules, against MACE-CSP + D3.
- An exp-6 search of 2,000 starts; its distinct minima, lowest first, up to K = 64, relaxed under MACE-CSP; the same
  with eLJ as the pool8 control.
- Reference basin set: Step 2's search plus all relaxations, at FD identity 0.05 Å.
- **Primary endpoint:** recall within 5 kJ/mol at K = 32, exp-6 seeding against MACE-CSP random starts at equal K,
  paired across molecules (sign count with a Wilson interval). Secondary: the anchor found, and at which K (k/16 with
  an exact interval); cost in GPU-h and core-h with hardware named; recall against K to 64.
- A molecule whose reference has missing mass above 0.1 after its one extension is reported unsaturated and left out
  of the recall denominator; recall elsewhere is a lower bound.
- Motif tables of missed basins are descriptive only.

## Part P (triage, already mostly paid for)

The polymorph pairs (156 built) are used only to rank pairs by MACE-CSP vs UMA-s disagreement and send the worst to
the reference single points. Rigid relaxation from X-ray conformers puts unequal conformer strain into each form, so
agreement percentages are not reported as accuracy. Pairs whose two relaxed forms lie within 0.05 Å FD are dropped as
duplicates.

## Proposed go/no-go criteria (numbers for the owner to set before Step 1)

- **D1:** keep MACE-CSP + D3 unless, on the out-of-selection molecules, it shows holes in ≥ 2/8 where UMA-s shows
  none, or its anchor sits more than 5 kJ/mol above its lowest basin in at least 2 more molecules than under UMA-s, or
  the reference single points side with UMA-s on most of the disagreements. Model agreement never changes D1.
- **D2:** F1 is the method if median recall within 5 kJ/mol at K = 32 is ≥ 80% with no molecule below 50%, the anchor
  basin is found in ≥ 14/16, and total cost to 90% recall is ≤ ⅓ of MLIP random starts. Otherwise F1 is topped up
  with random MLIP starts or limited to the motifs where it passes.
- **D3:** ask for a fine-tune if and only if holes or reference-confirmed errors concentrate in out-of-selection
  molecules; the request is those structures.
- **Sequencing:** D1 is settled (Step 2) before Step 3's recall is final; Step 3's exp-6 searches can run alongside.

## Known limits, stated rather than tested

- **Rigid molecules.** The X-ray or DFT conformer is held fixed; cooperative H-bond lengthening and torsional
  relaxation are absent. A final all-atom relaxation of the low basins is the remedy but is not in MXtalTools' search
  route today.
- **PBE-D3 is the ceiling:** both models copy it, with its delocalisation error in short H-bonds and no many-body
  dispersion. Hybrid or MBD points on a few H-bonded structures would measure that; out of scope unless negotiated.
- **Lattice energy, not free energy:** vibrational differences between polymorphs (0.5–2 kJ/mol) are not included.
- **No absolute or cross-molecule lattice energies:** both models' isolated-molecule leg is out of domain. X23
  sublimation comparisons are therefore not made.
- **Reproducibility:** MACE-CSP-25.2 weights were obtained privately; permission to name and deposit them, or a public
  release, is needed before submission.
- **Scope:** rigid, Z′ = 1, one space group per molecule.

## Superseded

Draft 1 of this page. In `exp6_recall_test.md`, everything computed on the pre-00028573 conformers: the mirror test,
MACE-OFF24 vs UMA-s on the chiral molecules, the anchor tables (including the benzoic acid anomaly and the UMA-s
1.28 Å H···O contact) and the probe anchors, until rerun.
