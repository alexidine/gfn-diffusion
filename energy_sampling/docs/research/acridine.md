# Acridine (space group 14, Z'=1)

*Research note. Not repository policy and not a description of the code.*

Acridine is the system trained against a MACE energy. This note collects what has been measured on its landscape, its energy models and the trained sampler's distribution. Entries are dated. Numbers were carried over on 2026-10-05 from the session records of the date shown and were not re-run for this note; where a stored report exists its path is given and it is the fuller source.

Energies are lattice energies in kJ/mol per molecule unless a line says otherwise. "Old model" is `acr_112025_mh1_stagetwo.model`; "new model" is `acr_newmodel.model`; both are under `D:/crystal_datasets/`.

## Owner decisions

- **One universal conformer (owner, 2026-09-30).** The gas-phase minimum under the new model, used in priors, searches, analyses and training. "We do need a new 'universal' conformer. Let's gas-phase optimize one under newmodel ... Then make sure we use it everywhere (priors, searches, analyses, training)."
  - File: `D:/crystal_datasets/acridine/acr_newmodel_conformer.pt`; build scripts in `acr_newmodel_conformer_build/` beside it (both present 2026-10-05). The intended cluster copy is `/scratch/mk8347/data/crystal_datasets/acridine/acr_newmodel_conformer.pt`.
  - Atom order is that of `opt_acridine_conformer.pt` (N at index 0), not `acridine_conformer.pt` (N at index 13), because every search state and seed file uses the N-first order.
  - Geometry: BFGS in float64 to 1e-4 eV/A, reached identically from both older conformers.

## The conformers (2026-09-30)

| Conformer | Energy above the universal conformer, new model (kJ/mol) | Where it was in use on 2026-09-30 |
|---|---|---|
| universal (`acr_newmodel_conformer.pt`) | 0 | new work from this date |
| opt, relaxed under the old model (`opt_acridine_conformer.pt`) | 0.06 (RMSD 0.007 A) | crystal searches |
| old compressed (`acridine_conformer.pt`; C-N 1.300 A, C-C 1.376 A) | 76.9 | training priors `acridine_sg14_zp1_mace_prior_dataset{,_w3}.pt`, GFN runs, analysis references |

*Gas-phase energy of each stored acridine geometry relative to the universal conformer, one evaluation each. The comparison is between rows: the two relaxed geometries are the same molecule to 0.06 kJ/mol, the compressed one is strained by tens of kJ/mol.*

Training reads its molecule from the prior file, so a run follows whichever conformer its prior embeds. The gas leg of the lattice energy is computed at run time and is not stored.

## Old model against new model (2026-09-30)

Measured on CPU through the MXtalTools route in float32 and through an independent ASE `MACECalculator` in float64; the two agree to about 1e-5 eV on gas and crystal legs for both models, and a non-periodic gas evaluation equals a 30 A box. The two model files carry identical atomic reference energies (H, C, N), scale and shift, and a 6 A cutoff, so a gas-phase difference between them is the model and not reference bookkeeping. The scripts were in a session scratchpad and were not kept.

| Quantity | Old model | New model | New minus old |
|---|---|---|---|
| Gas energy, old compressed conformer (kJ/mol) | | | -35.96 |
| Gas energy, each model's own relaxed conformer (kJ/mol) | | | -30.0 |
| Strain of the compressed conformer (kJ/mol) | 83 | 77 | |
| Crystal energy per molecule, 64 prior rows (kJ/mol) | | | +6.5 (sd 2.0) |
| Lattice energy, same 64 rows (kJ/mol) | | | +42.5 (sd 2.0) |
| Lattice energy of ACRDIN04 (kJ/mol) | -71.6 | -32.9 | +38.7 |
| Lattice energy of ACRDIN08, Z'=3 (kJ/mol) | -29.8 | -29.2 | +0.6 |

*The two MACE models on the same acridine structures. Blank cells were not recorded as separate values. Reference: the experimental sublimation enthalpy is 91 to 95 kJ/mol (NIST), which puts the lattice energy near -95 kJ/mol; the old model's ACRDIN04 value is the closer of the two.*

- The new model binds the crystal about half as strongly as the old one. It is not a constant offset: over the 64 prior rows the two lattice energies correlate at Pearson 0.78 and Spearman 0.68, the slope of new on old is 0.65, and 15% of pairs whose old-model gap exceeds 2 kJ/mol change order.
- On 160 prior rows the new-minus-old crystal energy per molecule tracks the packing coefficient (Spearman +0.74, +93 kJ/mol per unit of packing coefficient, residual sd 1.6 kJ/mol). Energy against packing coefficient correlates at -0.58 under the old model and -0.16 under the new. Across the prior's packing-coefficient range of 0.62 to 0.73 the new model favours the looser packings by about 10 kJ/mol, about 4 nats at kT 2.5 kJ/mol.
- Rigid expansion of a cell reaches the gas value by a scale of 2 under both models. The new model has a repulsive bump of +1 to +2 kJ/mol at scale 1.35 to 1.5 and a flat or expanding minimum near scale 1.05 on prior rows.
- For a rigid conformer the gas leg is one constant, so on its own it moves only log Z.
- ACRDIN08 sits 42 kJ/mol above ACRDIN04 under the old model and 3.7 above under the new. Not explained.

## The dominant mode of the trained sampler (2026-09-28)

Run `p23_acr_ft` at step 274,600: one family held about 93% of the draws inside the energy window and the two experimental polymorphs about 0.01%. A sixteen-agent review (eight investigations, each with a skeptic) looked for the cause. **Full source:** `D:/crystal_datasets/acridine/reviews/acr_blob_2026-09-28/SUMMARY.md` (present 2026-10-05), with a `digests/` folder beside it holding each claim and its skeptic verdict. Section 9 of that file lists next steps and decisions put to the owner; section 10 holds the follow-up below.

What that review reported:

- **What the family is.** A lattice family of translation pi-stack columns 3.9 to 4.0 A apart with a c-glide herringbone, cell about 4.0 x 12.4 x 18.9 A, labelled TT-G in the review. It is the crystal search's dominant family at -60 kJ/mol and below and contains the search's global minimum (BF22). It is not the long-axis family that dominated the earlier p12 runs.
- **Ranking depends on the energy model.** On the MACE route the sampler trains against, the family's minima lie 0.1 to 2.6 kJ/mol below ACRDIN04 (-73.35) and 1.3 to 3.7 below ACRDIN12 (-72.24). eSEN-S, the other MACE model (`acridine_mh1_stagetwo`) and eLJ put the polymorphs below the family. No DFT-D calculation has arbitrated.
- **Conformer.** The sampler runs and the analysis references used the old compressed conformer and the search used the relaxed one. The roughly 12 kJ/mol gap between sampler and search energies of the same packing is that conformer difference under one MACE file, not two MACE versions.
- **Size of the target's preference.** The target favours the family over each polymorph basin by about 5 nats or more: 0.4 to 1.2 from the energy, about 2 from the width of each crystal's basin (soft sliding and libration, present under every potential tried), and 2 to 3.5 from the number of members in the family. Choices of measure moved this by at most 0.3 nats each.
- **Sampler shortfall on top.** The sampler under-populates the polymorph basins by a further 2 to 10 nats and reaches 1 of the 48 symmetry-equivalent poses of ACRDIN04.
- **How it developed.** The earlier p20 run held the family at 43 to 45% of draws, the share of its anchors, until about step 236,000 and reached 91% by about step 256,000 through replay, with no config change. It also stopped populating the face holding ACRDIN04's basin (BF28 to BF34). The mechanism was inferred and was not tested by ablation.
- **Defects in the analysis found on the way.** `load_gfn` ignored the checkpoint's `pb_frozen` entry; basin censuses at radii 0.05 to 0.147 undercount thermal basins by 4 to 10 times; the reduction-zero filter drops ACRDIN12 copies that sit on a wall; `std_acridine_polymorphs.pt` is not the as-deposited structures (volumes 4 to 5% low).

Follow-up the same evening, testing the owner's reading that the late collapse is correct thermalisation between basins:

- In the basins the sampler kept, the difference between its log density and the log reward is flat across thermal members, so thermalisation is right there.
- The basins it abandoned (both polymorphs, BF29, BF34) are under-populated by about 1 to 5 nats at the level of the basin, by counts that need no importance sampling. Per-member importance-sampling values there are lower bounds only.
- The normaliser symmetry is approximately learned: the spread of log density across pose copies is at most 0.45 inside the family and 0.5 to 1.3 outside. Earlier figures of 15 to 67 nats came from importance sampling that had not converged. The asymmetry between the C2 and screw orientations grew between steps 195,000 and 274,600.
- Stored backward data supervise only the floors of basins, and isotropic widening of the prior cannot reach the thermal cloud, which is anisotropic along y.
- The prior build removed 26% of parent structures as out of domain (14 of the 19 ACRDIN04 floor parents among them) instead of rewriting them into the domain.

## Atom-wise against environment-wise RDF on the same draws (2026-09-28)

Draws: `p23_acr_ft_running_10k`, step 274,600, 9,805 rows inside a 15 kT window. The same draws were given RDFs twice: atom-wise (276 channels) and environment-wise (91 channels). One run, not checked by a second reader.

| Basin | Draws, atom-wise (count) | Draws, environment-wise (count) |
|---|---|---|
| B1 | 6,762 | 9,357 |
| B2 | 2,323 | 126 |
| B3 | 165 | not recorded |
| B4 | 131 | not recorded |
| B5 | 109 | not recorded |
| unassigned | 315 | not recorded |

*Basin sizes from the same 9,805 draws under the two RDF schemes. Basin labels are assigned separately per scheme, so B2 atom-wise and B2 environment-wise are not the same set of draws.*

- The atom-wise second basin is far from the first under its own metric (median distance from a B2 draw to the nearest B1 draw 0.711, against 0.189 within B2). Under the environment-wise metric the same pairs are at 0.137 and 0.115, and the atom-wise B2 draws are interleaved through the environment-wise family.
- The split is not an atom-labelling effect. Relabelling a draw by acridine's C2 permutation (180 degrees about the N to C9 axis, 20 of 23 atoms move) leaves the environment-wise distance at about 3e-7 and moves the draw about 0.4 atom-wise, but relabelled B2 draws remain 0.72 from B1 and 0.19 from B2. The balance of the two labellings inside each basin was not measured. The sampler's second fractional-centroid marginal is strongly asymmetric, where the C2 symmetry would give 50:50.
- For identity the environment-wise metric did better on this system. COMPACK overlays with a 20-molecule shell on each polymorph's 10 nearest draws per metric found two 20 of 20 matches for ACRDIN12; they were environment-wise ranks 1 and 2 (distances 0.113 and 0.163) and atom-wise ranks 66 and 1 (0.581 and 0.216). One true match, at 0.163, lies beyond the 0.147 cut that had been calibrated as "never the same structure".
- Both metrics agree on the polymorphs. ACRDIN04 was not generated (best of the 18 nearest draws matched 13 of 20 molecules). ACRDIN12 was generated twice, both strained (-57.7 and -37.7 kJ/mol against -72.2, RMSD 0.26 to 0.38 A). Every other neighbour is a different packing (2 to 9 of 20). Where the polymorphs appear inside the family on a UMAP, that is the layout and not membership.

Artefacts (present 2026-10-05): `D:/crystal_datasets/gfn_results/p23_acr_ft_running_10k_atomwise.pt` and `p23_acr_ft_envwise_layout_atomwise_basins.png`. The analysis scripts were in a session scratchpad and were not kept.

## Open

- A DFT-D ranking of the dominant family against ACRDIN04 and ACRDIN12.
- Whether the new model's weaker and looser binding is acceptable for the target, given the experimental sublimation enthalpy.
- Rebuilding the training priors on the universal conformer.
- The ablation that would test how replay carried the family from 45% to 91%.
- Not yet folded into this note: the Z'=2 search strategy, the Z'=1 search campaign of 2026-09-30, and the older landscape maps (space group 14 motif maps, the p12 dominant mode, the packing atlas).
