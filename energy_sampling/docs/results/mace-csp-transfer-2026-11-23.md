# MACE-CSP-25.2 on molecules outside its training rules (2026-11-23)

*Battery record. Closed 2026-11-23. Code at commit `a538b210` (GFN), `7f10473b` (MXtalTools). Not a finding.*

## Design

**Question.** Two checks on MACE-CSP-25.2 large + D3 for the training pool, most of which falls outside the model's
selection rules (no CH3, at most one OH/NH2/COOH rotor):

1. **Across molecules.** Is its lattice energy as predictable from the molecular graph as UMA-s 1.1's?
2. **Within a molecule.** On pool molecules outside the rules, do its relative crystal energies match UMA-m 1.1 (`omc`)?

**Check 1.** The 2,000 random CSD Z' = 1 crystals of `eval/energy_baseline/baseline_fit.py`, experimental geometry,
restricted to H, C, N, O, F. MACE-CSP uses D3(BJ, PBE, 15 Å). Predictor: gradient-boosted trees on Morgan count
fingerprints and descriptors, five folds grouped by molecule. Criterion fixed before the run, on the bootstrap 95%
interval of RMSE(MACE-CSP) / RMSE(UMA-s): pass if the upper end is at most 1.10, fail if the lower end exceeds 1.10.

**Check 2.** A seeded draw from pool v1 of 20 molecules outside the rules and 5 inside, each one rigid RDKit conformer.
Per molecule: a MACE-CSP L + D3(BJ) search in P2_1/c and up to 5 distinct minima from the lowest to about +15 kJ/mol.
Single points with MACE-CSP + D3 (zero damping, the OMC25 scheme, 30 Å) and UMA-m. With d = E(MACE-CSP) − E(UMA-m) and
e = d − median(d) within a molecule, criterion fixed before the run: pass if RMS(e) ≤ 4 kJ/mol and min(e) ≥ −10 kJ/mol.

**Departures from the plan.**
- The owner cut the panel at the 10 molecules whose searches had finished (9 outside, 1 inside), before any UMA-m
  energy was read. The cut follows the panel's seeded order.
- The first three molecules ran 50 starts, the other seven 25.
- 21 of the 50 UMA-m single points exceeded the 40% GPU share and ran on the CPU. Three rows repeated on the CPU matched
  their GPU values to 0.0002 kJ/mol.

## Data

`D:/crystal_datasets/mlip_transfer_nov23/`: `smoothness/` (`smoothness.json`, `lattice_energies.npz`), `within_molecule/`
(`panel.json`, `structures.json`, `scores.json`, `within_molecule.json`), and the logs. `oom_diag/` is the instrumented
rerun of molecule out02, whose outputs were reused.

## Measurements

*Table 1. Out-of-fold prediction of lattice energy from the molecular graph, on 1,150 CSD crystals (790 with CH3) that
every model scores with a negative lattice energy; kJ/mol per molecule. RMSE ratio MACE-CSP / UMA-s: 0.911 (bootstrap 95%
interval 0.874–0.946), against the limit of 1.10.*

| model | lattice energy SD (kJ/mol) | R² | RMSE (kJ/mol) | MAE (kJ/mol) | RMSE with CH3 (kJ/mol) | RMSE no CH3 (kJ/mol) |
|---|---|---|---|---|---|---|
| MACE-CSP L + D3(BJ) | 33.8 | 0.724 | 17.73 | 12.67 | 16.82 | 19.58 |
| UMA-s 1.1 | 36.0 | 0.708 | 19.46 | 13.85 | 18.22 | 21.94 |
| MACE-OFF24 | 31.9 | 0.685 | 17.94 | 13.04 | 16.97 | 19.90 |

*Table 2. MACE-CSP L + D3(zero) against UMA-m on crystal energies relative within each molecule, 5 structures per
molecule. Interval: exact 95% on the fraction passing. The subgroups overlap.*

| molecules | evaluated (count) | passed (count) | 95% interval (%) | median RMS (kJ/mol) | worst RMS (kJ/mol) |
|---|---|---|---|---|---|
| outside the rules | 9 | 9 | 66–100 | 0.90 | 2.65 |
| outside, with CH3 | 8 | 8 | 63–100 | 1.25 | 2.65 |
| outside, one RDKit torsion | 7 | 7 | 59–100 | 0.63 | 2.32 |
| inside the rules (control) | 1 | 1 | 2–100 | 2.23 | 2.23 |

*Table 3. Per molecule, the same comparison. "Lowest e" is the most negative residual; "UMA-m span" is the highest minus
lowest UMA-m energy of the 5 structures.*

| molecule | SMILES | heavy atoms (count) | RMS(e) (kJ/mol) | lowest e (kJ/mol) | UMA-m span (kJ/mol) |
|---|---|---|---|---|---|
| out00 | `Cc1ccc(O)c(-c2cc(C)nn2C)n1` | 15 | 1.59 | −3.14 | 15.4 |
| out01 | `CN/N=c1/cc[nH]c2cccnc12` | 13 | 2.32 | −0.17 | 10.0 |
| out02 | `Cc1ccc(N2CCC(C)(C)C2)nn1` | 14 | 0.63 | −0.43 | 16.4 |
| out03 | `Cc1cc2ncnn2cc1F` | 11 | 1.72 | −1.79 | 12.5 |
| out04 | `OC1([C@]23CCC[C@H]2CCC3)CC1` | 12 | 0.38 | −0.62 | 15.5 |
| out05 | `C[C@@H]1[C@@H]2N[C@@H](C)C[C@H](O)[C@H]2CC[C@H]1C` | 14 | 2.65 | −2.51 | 12.7 |
| out06 | `C[C@@H]1CC[C@@H]([C@@]2(O)C[C@@H]3CO[C@H]2C3)C1` | 14 | 0.53 | −0.89 | 14.4 |
| out07 | `CO[C@@H]1C[C@@H](O)[C@]12CCCC21CCC1` | 14 | 0.28 | −0.23 | 14.3 |
| out08 | `CN1CCC=C1C=O` | 8 | 0.90 | −0.61 | 13.3 |
| in00 | `C1=Cc2cc3c(cc2=C1)N=c1ccccc1=3` | 16 | 2.23 | −4.07 | 13.7 |

Over all 50 structures, RMS(e) is 1.57 kJ/mol and the largest |e| is 5.06 kJ/mol (out05). The two models put the same
structure lowest in all 10 molecules.

*Table 4. Search rows that ended below the lowest stored minimum, in the three 50-start searches, measured before the
panel was cut. "Contact" is the shortest distance between atoms of different molecules.*

| molecule | starts (count) | rows below the lowest minimum (count) | their contact (Å) | their energy below the lowest minimum (kJ/mol) | stored minima (count) | shortest contact in stored minima (Å) |
|---|---|---|---|---|---|---|
| out00 | 50 | 6 | 0.00–0.61 | 8.2e3 – 9.5e7 | 40 | 1.79 |
| out01 | 50 | 0 | none | none | 48 | 1.90 |
| out02 | 50 | 4 | 0.13–0.40 | 1.3e5 – 1.6e9 | 43 | 2.00 |

## Trends

- Check 1's RMSE ratio lies below the limit across its whole bootstrap interval.
- Check 2's residuals are of similar size for the molecules outside the rules and the one inside.

## Not covered

- UMA-m is a model, not DFT, and shares the PBE + D3 family with MACE-CSP. Agreement can include an error they share.
- Structures are MACE-CSP minima. A structure UMA-m rates low and MACE-CSP rates high would not be in the set.
- Each molecule is one rigid conformer, so torsional and methyl-rotor energy differences were not compared.
- Check 1 compares how learnable the lattice energies are. It does not measure whether the gas leg is correct, and the
  predictor's RMSE of about 18 kJ/mol limits its sensitivity to small per-molecule jitter.
- 10 of the 25 planned molecules were run, with one inside-rules control.
- The overlap rows of Table 4 were not counted for UMA-s, UMA-m or MACE-OFF24, and not for the seven 25-start searches.
