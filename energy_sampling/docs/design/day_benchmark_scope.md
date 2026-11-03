# Day-landscape benchmark: scope, design decisions and staged plan

Status: **proposal, partly decided.** Drafted 2026-11-02. The owner answered D1–D7 on 2026-11-03; those answers are in
"Owner decisions, 2026-11-03" below and override §2 wherever the two differ. It controls nothing executable. It scopes the benchmark
half of the Day-landscape paper: a survey of existing crystal generators against the Day group's FIT+DMA reference
landscapes (Taylor, Butler & Day, Faraday Discuss. 256 (2025) 434-458; deposit DOI 10.5258/SOTON/D3094).

Relation to other documents:
- `generator_benchmark_protocol.md` (2026-10-26) is the per-molecule protocol: draw, substitute, score raw, optimise,
  rescore. It stays the protocol. This document adds what a paper-scale survey needs on top of it: which molecules and
  modes, which energies in which role, identity, metrics, space-group and Z fairness, the roster, and the staged plan.
- Owner decisions already recorded there are carried here as **decided**, with their dates. Everything else is a
  recommendation.
- `energy_sampling/eval/day_bench/` is the development code (committed `f156358c`, research code, not a contract).

Evidence tags: **RUN** = measured this session or in a dated development run named beside it; **READ** = read from
code, file or paper; **U** = estimate or unverified lead.

## Owner decisions, 2026-11-03

Owner's framing: the benchmark is in workflow development. It is de-prioritised behind generator convergence, because
it counts for nothing if the generator does not converge.

| decision | owner's answer (2026-11-03) | what changes in this document |
|---|---|---|
| D1a molecules | Pilot on a small subset of molecules, exp-6 tier only, for now. Return to the MLIP question once the MLIP we ultimately train on is chosen. | The pilot is a small exp-6 subset. Molecule tiers that depend on the MLIP, such as the 16 MACE-validated refcodes, are deferred. Stage E is deferred. |
| D1b window and modes | Deferred, as D1a, until the MLIP decision. | The consensus window stays a recommendation. The pilot uses it as a working assumption. |
| D2 conformer | **Hold**: keep thinking. For now, substitute early, or code the option set. | Build the substitution placement as a switchable option (early, end, own molecule). Early is the working default for development. The 2026-10-26 "decided" entry in D2 is reopened. |
| D3 energies and relaxation | Broadly agreed with the table. The severe tier should be simpler, for example pure van der Waals overlap: the owner does not want to defend exp-6 in print. The final MLIP comes later. Keep the optimisation simple and uniform, consistent with standards in the field. | Severe = geometric van der Waals overlap only, with no exp-6 role. Optimiser settings are to follow field practice (see D3 note below). |
| D4 identity | Agreed. The FD distance is a curio and an advertisement, not the settlement metric. | No change. |
| D5 metrics | The output is the two-dimensional object C(E, N): coverage as a joint function of sample size N and optimisation effort E. Otherwise reasonable. | The primary result is the surface C(E, N). Curves and topline numbers are read off it. |
| D6 groups and Z | Ours follows Z(c), or bare CSD statistics as the worst case. Methods that need a group also follow CSD statistics; uniform over groups would be "really brutal". P1 generators as recommended. | Group-needing baselines are allocated by CSD space-group frequencies, not equally over the 26. |
| D7 roster | Cover specifically **all crystal generators published in 2026**. Add a standard classical search, for example Genarris or FastCSP; readers will expect one. | Roster widened: every 2026 generator is listed and each gets a run, or a stated reason why it cannot. A classical search arm is added. |

**D3 note: field practice for the optimiser** (READ, from the papers via the 2026-10-19 scouting report).
- Packora relaxes with UMA through ASE BFGS at fmax 0.02 eV/Å (about 1.9 kJ/mol/Å), at most 500 steps.
- MolCrystalFlow and CG-OMatG relax in three stages, each capped at 1,000 steps: rigid-body BFGS for 100 steps, then
  cell plus atoms, then atoms only.
- The development optimiser stops at 0.01 kJ/mol/Å, about 200 times tighter than fmax 0.02 eV/Å.

Proposed for the pilot: one rigid-body optimiser with a force tolerance of fmax 0.02 eV/Å and a 500-step cap, taken
from the field. Depth snapshots (0, 10, 30, 100, 300, 500) supply the E axis. This is a proposal, not decided.

## Tooling state, 2026-11-03

What exists in `eval/day_bench/`, written to the decisions above. Development code, not a contract.
- `day_landscape.py`:
  - reads the union of a molecule's CSVs, so chiral directories are complete, and joins the Delta-ML correction;
  - writes every structure within 10 kJ/mol under either ranking as its primitive P1 cell, with `reference.csv` and
    Day's conformer spec.
  - The reference then passes through `protocol_refine` like any sampler, which closes task B1.
- `protocol_refine.py`: records the van der Waals overlap at every depth (`overlap`: smallest intermolecular heavy-atom
  distance over the Bondi radii sum) and takes `--f_tol`.
- `protocol_score.py`: reads the reference from that refine output, with both Day energies and the molecules per
  primitive cell.
- `protocol_report.py`:
  - the C(E, N) surface (table and heatmap), and coverage against cumulative calls, each draw charged min(steps, depth);
  - modes by complete linkage, in the window under both Day rankings;
  - severe = overlap below `--clash` (default 0.75), with no exp-6 role;
  - reachable-mode reach, dropped modes, the effective number of modes hit, and a ledger.
  - This closes task B5. COMPACK failures were already recorded without a verdict (`compack_endpoints.compare`).
- `protocol_random.py`: `csd:N` allocates starts over the 26 groups by CSD frequency (`allocate`, `csd_statistics.json`).
  Starts build in all 26 groups (checked 2026-11-03).
- MXtalTools `8f4c29ad`: `canonicalize_aunit` applies a half-cell origin shift where a group's operators and box use
  different origin choices. P4₂/n, one of the 26, refused 18 of 46 Day rows before the fix and none after (task B2).
- Tests: `tests/crystal/test_day_bench_protocol.py`, 11 CPU tests.
- Smoke, imidazole, 2026-11-03: Day reference of 250 minima, then 100 MolCrystalFlow draws refined to 1,000 steps,
  scored and reported (`artifacts/generator_scouting_oct19/pilot_nov03/`). This is a test of the pipeline, not a
  result.
  - 247 of the 250 reference minima rebuilt. 3 rebuilds missed by about 2 Å, probably the frame flip on symmetric
    molecules; this is open.
  - The 247 minima form 206 modes, 62 of them in the window.
  - No Day minimum falls under the clash cut: the smallest raw overlap is 0.87.
- Not built:
  - the conformer-substitution option set (early substitution only, as now);
  - environments for SALA (its install was refused by the session's permission classifier; the commands are with the
    owner), Packora, CrystAF and Genarris/FastCSP.
- Downloaded 2026-11-03, outside the repo:
  - Clari-H weights (Hugging Face cache);
  - SALA code and weights (`external_generators/SALA`, `sala_ckpt`; checksum matches);
  - Packora's `csd_clari.csv` and OXtal's `train_ccdc_ids.txt` (`external_generators/leak_lists/`);
  - PyXtal 1.1.5 (`C:/Users/mikem/venvs/pyxtal`).

## Topline (as drafted 2026-11-02; superseded where the owner decisions above differ)

The recommended benchmark, in one paragraph per axis:

1. **Molecules.** Imidazole for development. The 16 molecules of Day's MACE re-optimisation for the pilot and the
   MLIP tier (owner-agreed 2026-10-26). A stratified set of about 100 for the headline. All 1,017 refcodes only for the
   cheap arms (ours and random search). Every Day molecule is unseen by our model if the data thread's leak policy
   (pool proposal D2a) is adopted, so no internal Day split is needed.
2. **Modes.** Day minima within 8 kJ/mol of the global minimum under **both** of Day's rankings (FIT+DMA and the
   Delta-ML DFT correction). Each is re-relaxed with the benchmark relaxer and merged by structural identity. The mode
   set is fixed by Day's energies, never by our relaxer, so it does not change when the relaxer changes.
3. **Energies, in three separate roles.** Day's energies rank and weight the modes. A relaxer moves samples into
   basins: exp-6 FIT with MMFF point charges now, Day's MACE model for the final tier (owner, 2026-10-26). An exp-6
   threshold flags severe samples.
4. **Relaxation.** Rigid molecules with Day's conformer, in P1 at Z' = Z, with the same FIRE schedule for every method,
   the reference included. Cost is one ledger of energy-and-gradient calls.
5. **Identity.** Description-free: RDF distance as the screen, COMPACK settling the ambiguous band, both calibrated per
   molecule. A mode is found when one legit, non-severe sample's relaxed structure matches its relaxed representative.
6. **Metrics.** Coverage against draws at each optimisation depth, and against energy calls. The topline numbers are:
   raw coverage, raw clash and legit rates, calls to half coverage, and reach. Beside them sit dropped-mode and
   concentration diagnostics.
7. **Space groups and Z.** Draws are never allocated by the experimental space group or Z. Our model allocates by its
   learned slice weights. Methods that need a group get Day's own rule, equal over the 26. P1 generators get a CSD
   prior over Z. Coverage is reported over all modes and over the modes each method can represent.
8. **Roster.** Core: quasi-random search over the 26 groups, Clari, MolCrystalFlow, Packora-L, SALA, and ours. Day's
   own search is a cost reference point, not a curve: the deposit carries no occupancy information (§1).

## 1. What the deposit and the local files hold (checked 2026-11-02)

**Table 1.** Inventory of the Day material on this machine. Per-refcode figures read the union of ids over every CSV in
the refcode's directory (chiral directories hold `_all`, `_racsg` and `_sohncke` files). Scripts are in this session's
scratchpad (`daystats.py`, `daytrial.py`, `dml.py`, `zprim.py`).

| item | content | evidence |
|---|---|---|
| FIT+DMA landscapes, `D:/crystal_datasets/day1000_fitdma/FIT-DMA_Landscapes/` | 1,017 refcode directories. Columns: id, space group, density, FIT+DMA energy, `minimization_step`, `trial_number`, `minimization_time`, metadata (always empty). Window ≈ 30 kJ/mol above each global minimum. 3,635 structures per refcode (median; 10–90%: 1,205–8,685). | RUN |
| Delta-ML corrections, `Downloads/Delta-ML_Corr_Landscapes.zip` | 1,016 refcodes, `corrected.csv` (structure, zprime, energy, committee uncertainty, corrected energy in kJ/mol). Committee uncertainty 1.1 kJ/mol median per refcode. For 10% of refcodes, fewer than 18% of their FIT+DMA minima within 8 kJ/mol carry a correction. | RUN |
| MACE model, `Downloads/MACE_total_energy_model.zip` | `rigid_csp_mlp_iteration4_rattled_mlp_opt_pbed3_small_swa.model` | READ |
| MACE re-optimised landscapes, `Downloads/MACE_Reopt_Landscapes.zip` (1.36 GB) | 16 refcodes (atcdeo, birtur, bixkup, bixlok, buvgac, dalbic, debfoi, dobyoj, jasgit, kogfer, qekqig, tavtuh, tunwuv, udexuz, wacyab, wacyef). Each has `structures.csv` and `structures.zip`, inside a nested `Reopt_Structures.zip`. Ids end `-OPT-1`, energies are total MACE energies, and `trial_number` is 0. | RUN. This corrects an earlier note that it held energies only. |

The session brief says the MACE and Delta-ML archives were not yet downloaded. They have been in `C:/Users/mikem/Downloads/` since 2026-10-20.

Measured facts that shape the decisions:

- **Mode counts are heavy-tailed.** Minima within 8 kJ/mol under FIT+DMA number 107 per refcode (median; 10–90%:
  23–347). Within 8 kJ/mol under both rankings: 60.5 (5–250). Over all refcodes the union gives 174,802 rows within
  8 kJ/mol, not the paper's 166,395 (data defects note, item 12). RUN.
- **Day's own search cannot be turned into a cost curve from the deposit.** The deposited `trial_number` is the
  quasi-random trial index of the kept, lowest-energy duplicate. For minima within 8 kJ/mol it is uniform over
  [0, max trial in its group]: the fraction with index at most f × max is 0.007, 0.096, 0.497 and 0.894 at f = 0.01,
  0.1, 0.5 and 0.9 (median over refcodes). A uniform index carries no discovery-time or occupancy information, so the
  only bound it gives is the trivial straight line. RUN.
- **Day's per-group trial counts.** The maximum trial index per group is about 17,000 (median), so the 10,000
  successful minimisations per group took about 1.7 attempts each. RUN; the attempt reading is U.
- **Day's compute cost.** Reading `minimization_time` as CPU seconds per minimisation gives 9 s per kept structure
  (median). That is about 650 CPU-hours per molecule for 260,000 minimisations. U: whether the field covers all three
  stages, and whether kept structures are typical, are unchecked.
- **Molecules per primitive cell.** Of the ≤ 8 kJ/mol FIT+DMA rows, 0.5% have 1 molecule per primitive cell, 15.9%
  have 2, 1.0% have 3, 65.3% have 4, 1.2% have 6 and 16.2% have 8. 19 rows lie outside the 26 groups. A sampler
  restricted to Z ∈ {2, 4} can represent 81% of rows; adding 8 reaches 98%. Clari's suggested sweep (4, 2, 1) reaches
  82%. RUN.

## 2. Decisions

Each decision gives options, a recommendation, its consequences, and how it would be defended in print.

### D1. Reference set

**D1a. Which molecules.**

| option | molecules | consequence |
|---|---|---|
| all 1,007 molecules (1,017 refcodes) | every landscape | Strongest generality claim. Data-trained generators at about 1,000 samples per molecule, 3 seeds and depth curves cost around 10⁹ relaxation calls per method (U), so this is not affordable for them under an MLIP. |
| **tiered (recommended)** | dev: imidazole. Pilot and MLIP tier: the 16 MACE re-optimised refcodes. Headline: about 100 stratified refcodes containing the 16. Supplementary: all 1,017 for ours and random search only, exp-6 tier. | The 16 allow the MLIP tier to be validated against Day's own MACE energies (agreed 2026-10-26). The stratified 100 carries the leak contrast. The supplementary table shows our model's generalisation at no cost to the baselines. |
| the 16 only | MACE set | Cheapest and cleanest MLIP story. Too few for leak strata or size trends. |

Strata for the about 100 (counts from the scouting report, Day ESI Table S3):
- all 18 Day families held out from every OXtal-line model;
- up to 10 of the 19 held out from the Clari line only;
- all 16 polymorphic molecules;
- 10 whose experimental form ranks worse than 35th;
- the rest spread over heavy atoms (≤ 8, 9–12, 13–17) and hydrogen-bond donors (0, ≥ 1).

Exclusions and flags:
- Six experimental structures Day never matched: BUVGAC, DNNEPH10, IHEPUG02, MEGNES, QIBCEK and XUGHUD01. They are
  excluded from experimental-form metrics only. Their landscapes remain valid, and buvgac is one of the 16.
- The 20 landscapes spanning more than 60 kJ/mol (`geyfof` reaches +2,955 kJ/mol): inspect before use.
- `acrdin.csv` repeats 4,258 rows: deduplicate.

**Held-out split for our model.** Under the data thread's D2a, every Day molecule is excluded from our training pool by
SMILES and by InChIKey skeleton. All of Day is then held out and no internal split is needed. If any run trained on
rc_oct22 is shown, its 71 Day training conditions are removed or flagged in every table (data defects note, item 1).
For data-trained generators, each (model, molecule) pair carries a leakage flag (§D7).

**Defence.** The landscapes are chosen by a published, external criterion (Day's selection), not by us. Molecule
strata are fixed before any method is run. The leak contrast compares each generator with itself across strata.

**D1b. Which structures, and how they become modes.**

| option | window and weights | consequence |
|---|---|---|
| FIT+DMA ≤ 7, 8 or 10 kJ/mol | Day's deposited ranking | Simple. FIT+DMA misranks by a few kJ/mol against DFT, so some modes in the window are not low under better energies. |
| Delta-ML only | Day's best ranking | Corrections are missing for part of some landscapes (§1), and the committee uncertainty, 1.1 kJ/mol median, is comparable to the energy spacing. |
| **consensus (recommended)** | ≤ 8 kJ/mol under both FIT+DMA and Delta-ML, unweighted for the headline. Boltzmann-weighted at 300 K on the Delta-ML energy beside it. | Matches the owner's fairness principle: count modes that are low under every credible energy, so a generator is not punished for missing a mode one energy alone calls low. 8 kJ/mol is the Day paper's own window (97.8% of observed forms lie within it). Refcodes lacking corrections fall back to FIT+DMA alone and are flagged. |
| window under our relaxer's energy | exp-6 or MACE | Rejected. The mode set would change with the relaxer, and exp-6 shows phantom low minima absent from Day (imidazole, 2026-10-22). |

From windowed minima to modes:
1. Build Day's minima within 10 kJ/mol under FIT+DMA, so that the consensus window and its edge are both covered.
2. Re-relax each with the benchmark relaxer and protocol (D3).
3. Merge endpoints by identity (D4), with complete linkage. A mode takes the lowest Day energy among its members.

The merge is necessary. Day deduplicated only pairs within 1.0 kJ/mol and 0.05 g/cm³, and on imidazole 51 of 158 Day
minima had another within the RDF cut after exp-6 relaxation (RUN, 2026-10-23).

**Completeness.** Day's own completeness cannot be estimated from the deposit: there are no hit counts, and the trial
index is uninformative (§1). Recommended: a **closed reference** for the headline. A method's settled minimum that
matches no Day mode is reported as an **extra**, with its relaxer energy and space group. An extra is either a Day miss
or a relaxer artefact, and is counted neither for nor against the method. This departs from the scouting report's
open-reference recommendation (2026-10-19): an open reference ranked by our relaxer would reintroduce the
relaxer-dependence D1b removes.

**Defence.** The modes and their weights come from Day's energies alone, fixed before any sampling. Our relaxer only
brings samples and modes to a common geometry.

### D2. Reference conformer

**Decided by the owner, 2026-10-26:**
- Every method gets Day's own rigid conformer (B3LYP-D3(BJ)/6-311G**, the molecule every Day landscape was computed
  with).
- It is substituted first, before any energy call, by a Kabsch fit over graph automorphisms that keeps each molecule's
  centroid and orientation.
- Substitution is lenient up to a broken bond graph, which counts as a failed draw. The fit RMSD is recorded per
  sampler but not thresholded.
- Rigid-input samplers (MolCrystalFlow, SALA, ours) receive this conformer as input.

Options as considered:

| option | consequence |
|---|---|
| **Day's conformer, substituted first (decided)** | One molecule for every method and the reference. Intramolecular energy drops out as a confound. Poor molecular geometry surfaces as clashes after substitution. |
| a DFT gas minimum we compute | Same in effect as Day's, but not the reference's own molecule. No gain. |
| each method's own molecule | Flexible all-atom generators keep their geometry. Requires flexible relaxation with an intramolecular energy, and RDF identity stops being exact. Owner rejected this for the final tier (2026-10-26). |
| substitute at the end, after flexible refinement | The owner called it "might actually be the fairest" (2026-10-26). It requires flexible refinement, and substitution itself may need relaxation. Held as a check, not the default. |

Still owed: the **substitution-placement check** promised on 2026-10-26. On about 100 samples per molecule, compare
flexible-refine-then-substitute with substitute-then-rigid-refine. Where both paths land within the identity tolerance,
placement does not matter for that molecule. Where they do not, the molecule is too flexible for a constant conformer
and is flagged. Run it on imidazole and two of the 16.

Two additions:
- **Our model is conditioned on Day's conformer too.** The proposed pool trains on MMFF geometries (pool proposal D6).
  That train-to-test geometry shift penalises us, not the baselines. Report it.
- **SMILES-input generators** (Clari, Packora) need a SMILES. Use the SMILES perceived from Day's conformer, so the
  graph matches the molecule substituted later. Bond perception disagrees with the CSD SMILES for 31 refcodes and fails
  on `lovwoi` (data defects note, item 13); flag those.

**Defence.** The reference landscapes are rigid-molecule landscapes of this exact conformer. Giving every method that
molecule is the only setting in which "found the reference mode" is well defined.

### D3. Energy function and relaxation protocol

**Three roles, kept separate.**

| role | development tier | final tier | basis |
|---|---|---|---|
| ranks and weights the modes | Day FIT+DMA and Delta-ML (fixed) | the same | D1b |
| relaxes samples and modes to a common geometry | exp-6 FIT with MMFF point charges, rigid (MXtalTools `exp6`) | **Day's MACE model** (PBE+D3), rigid; owner, 2026-10-26 | the final tier is "doubly consistent" with Day (owner) |
| flags severe raw samples | exp-6 lattice energy above a threshold | exp-6, unchanged | owner, 2026-10-26 |

- **Delta-ML cannot relax samples.** It corrects FIT+DMA energies, and we cannot compute FIT+DMA (no DMACRYS). It
  serves only to rank the reference.
- **UMA rigid as a sensitivity tier** (recommended, on the 16 only, if cluster time allows). Clari, Packora, CrystAF,
  PackFlow and CG-OMatG rank or post-train with UMA, so a UMA-relaxed check answers the objection that the relaxer was
  chosen against them. The scouting report's case for this still stands. The local "UMA" checkpoint is OMC25 eSEN-S,
  not UMA, and must not be labelled UMA (memory lead, 2026-10-28; recheck before use).
- **MACE-tier validation gate.** For the 16, relax Day's FIT+DMA minima rigidly under Day's MACE and compare with
  `MACE_Reopt_Landscapes`. Day's re-optimisation was flexible (`-OPT-1` structures), so energies will not agree
  exactly. Gate on the rank correlation within the window, with a threshold set after a first look, plus COMPACK
  identity of each re-optimised structure with our rigid endpoint. MACE numbers count only after this gate passes
  (agreed 2026-10-26).

**Relaxation geometry.**

| option | consequence |
|---|---|
| **rigid, P1 at Z' = Z (primitive cell), for every method, the reference included (recommended)** | No symmetry detection, so no tolerance judgement and no failure mode that differs between methods. A space-group-native method gains nothing from its labels. A Day minimum that is a saddle in P1 may drift; relaxed-to-relaxed identity absorbs that, and the number of Day modes that leave their own tolerance is reported as a reference-side diagnostic. Measured control: perturbed IMAZOL03 relaxed in P1 returned 30/30 to the CSD structure, exactly P2₁/c, +0.15 kJ/mol against the space-group relaxation (RUN, 2026-10-22). |
| space-group relaxation after symmetry detection (spglib tolerance ladder) | Fewer degrees of freedom and fewer calls. But P1 generators fail detection at different rates, and space-group-native methods (ours, SALA, random search) skip the step, an information asymmetry in their favour. |
| P1 for P1 generators, space group for space-group methods | Two protocols in one table, with costs not comparable. Rejected. |

Our sampler loses its cheap in-group relaxation under the recommendation. That costs us, which is the defensible
direction. Space-group relaxation of our own samples can be shown as a labelled extra curve.

**Optimiser and budget.** `converge.fire` on the generalised forces:
- a row stops at largest force ≤ 0.01 kJ/mol per Å or per radian;
- step cap 0.1 in length-like coordinates; at most 1,000 steps;
- snapshots at depths 0 (raw), 10, 30, 100, 300 and 1,000. These are the protocol's starting depths; the owner may
  revise them after a first look.

The relaxer is the same for every method. Development observation: 363 of 3,120 imidazole samples were unconverged at
1,000 steps (MolCrystalFlow at Z = 8: 201 of 500; RUN, 2026-10-23). The cap therefore bounds cost, and unconverged
endpoints are scored as they stand.

**How cost is counted.** One ledger row per (method, molecule, seed):
- **Relaxation calls.** One energy-and-gradient evaluation of one crystal is one call. A row's cost is the steps it
  actually took, since converged rows stop. Depth-k cost is Σ min(steps, k). Exp-6 and MACE calls are never mixed on
  one axis; each tier carries its GPU-seconds per call as the conversion.
- **Sampling.** GPU-seconds and device, beside the curves rather than on them, because network cost is not an energy
  call.
- **Filters.** Legit and clash checks cost no calls, but a rejected sample still counts as a draw.
- **Training.** For energy-trained samplers, training calls are reported as an amortised cost per molecule. For
  data-trained models, the training data and compute are reported instead (owner's home-field answer, 2026-10-26).

**Defence.**
- Every method's samples, and the reference minima, pass through one relaxer, one schedule, one stopping rule and one
  call count.
- The relaxer's energy decides only which basin a structure falls into, never which basins count or how much.

### D4. Matching and identity

Generators may place one crystal in different space groups and settings, so identity must not read the description.

**Table 2.** Candidate identity measures. Calibrations are single-molecule development measurements, so read them as
indications. AUC is against COMPACK matches on the pairs named.

| measure | description-free? | cost | calibration on record | weakness |
|---|---|---|---|---|
| RDF earth-mover distance, `envwise`, 10 Å | yes | milliseconds per pair, batched | imidazole, 350 pairs over all depths: tolerance 0.086 [0.078–0.097] at P = 0.9 for a 15/20 match and 0.136 [0.126–0.147] for 10/20 (RUN, 2026-10-26). AUC 0.998 on 672 relaxed pairs (2026-10-22). Other molecules: 0.05–0.13 (wiki `structure-comparison-metrics`). | The tolerance is per molecule. The 10/20 cut overstated MolCrystalFlow's coverage (53% against 28% after COMPACK settlement; 2026-10-23). |
| COMPACK (`ccdc` PackingSimilarity) | yes | about 0.2 s per pair for 20-molecule shells; about 42 s for P1 30-molecule shells, with timeouts | Day's own duplicate rule is 30/30 at 30% and 30°. The repo's rule is 20/20. | Slow. A failure returns (0, 0), indistinguishable by value from a zero-molecule match. |
| glued FD metric (`mxtaltools.crystal_fd.metric`) | **no**: defined within one space group at Z' = 1 | fast | MIPCAS / NEHZOR: AUC 0.952 / 0.992, against 0.901 / 0.940 for RDF on the same pairs (memory lead, 2026-10-28) | Comparing a P1 generator's structure needs symmetry detection first; structures in different groups need a common description. |
| OXtalign | yes, and reads clusters | about 8 ms per pair (its README) | none on our pairs; claims 99% / 88% agreement with COMPACK | Not installed. Its method document was LLM-drafted and it has no preprint. |

**Recommended:**
- **Screen.** RDF distance between a sample's depth-k structure and each mode's relaxed representative.
- **Settle.** COMPACK on the band between the 15/20 and 10/20 tolerances, at most the 3 nearest band structures per
  (mode, method), stopping at the first match (`compack_band.py`). A match propagates to the method's structures within
  RDF 0.03 of the settled one.
- **Calibrate per molecule.** About 350 COMPACK 20-shell pairs spanning RDF distances once per landscape, reused for
  every method. Development speed: about 76 s on 12 workers (RUN, 2026-10-26).
- **Headline level.** A 15/20 match (P = 0.9) settled by COMPACK, with 10/20 as a sensitivity. Relaxed-to-relaxed
  comparison under one relaxer removes the reason for a loose tolerance: the shift of a minimum between energies (for
  example deposited IMAZOL03 against its exp-6 relaxation, RDF 0.146) no longer enters.
- **Mode found at depth k.** At least one legit, non-severe sample's depth-k structure lies within tolerance of the
  mode's representative. Raw samples (k = 0) are scored by the same rule and read as native quality. On imidazole, raw
  samples sit at a median RDF 0.231 from their own endpoints, so raw coverage near zero is expected for every method
  (RUN, 2026-10-22).
- **FD metric.** A within-group cross-check once the Search/prior/FD thread's port covers the 26 groups. Not the
  headline, because it is not description-free.
- **Experimental forms.** A side column for comparison with generator papers: 8/15 at RMSD15 < 2 Å, and 30/30 at
  RMSD30 < 1 Å (the blind-test standard).

**Defence.**
- Identity reads atoms and distances only, so a crystal is found in whatever description a method emits.
- The threshold is set per molecule against the field's standard overlay tool, never tuned per method.

### D5. Metrics

**Curves**, per method and molecule, averaged over random orders of the draws (exact rarefaction from per-mode hit
counts):
- C_k(N): unweighted coverage of the consensus modes against draws N, one curve per depth k;
- C(E): coverage against relaxation calls E, the upper envelope over (N, k);
- the same two curves Boltzmann-weighted (300 K, Delta-ML).

Across molecules: mean of per-molecule coverage, each molecule weighted equally, with a bootstrap over molecules.

**Topline numbers**, each tied to a failure mode the owner listed (2026-10-26):

| number | read from | flags |
|---|---|---|
| raw coverage at 1,000 draws | C_0(1000) | collapse; small probability spread over many modes |
| raw legit rate and raw clash rate | depth 0 | poor raw samples |
| E₅₀: calls to 50% unweighted coverage | C(E) | random-like search; slow clean-up |
| reach: coverage at the largest effort (1,000 draws × 1,000 steps) | C(E) | modes missed for good (a plateau below 1) |
| reachable reach: reach over the modes the method can represent | D6 | separates Z or group choices from sampling failures |

**Diagnostics:**
- **Dropped modes.** A mode is dropped by method m if some other method or the random search finds it, but m finds it
  in 0 of 3 × 1,000 draws at full depth. State it as "found in fewer than 3/N" at 95% confidence.
- **Concentration.** The effective number of modes in the method's hit distribution, 1/Σp², against the reference's
  Boltzmann effective number; also the share of hits on the method's top mode. On imidazole at 300 K the reference
  effective number was 39 (RUN, 2026-10-26).
- **Effective temperature**, from the slope of log occupancy against mode energy. It is descriptive only for
  data-trained models, which are not trained to be thermal (owner, 2026-10-26).
- **Clash rate**, geometric and independent of energy. Proposed: any intermolecular heavy-atom pair closer than 0.75 ×
  the sum of Bondi radii, a knob. It sits beside the exp-6 severe rate, whose proposed first line is a lattice energy
  above 0 kJ/mol per molecule. For scale, MolCrystalFlow raw imidazole samples have a median closest intermolecular
  heavy-atom contact of 1.46 Å (RUN, 2026-10-22).
- **Extras** (D1b): count and energy distribution of settled minima matching no Day mode.

**Energy-call ledger.** One row per (method, molecule, seed) with:
- draws;
- sampling GPU-seconds and device;
- filter rejects;
- relaxation calls by depth;
- wall time;
- training calls (energy-trained samplers) or training-data provenance (data-trained models).

**Our model's local density as a guide (optional arm).** Our sampler can rank its raw draws by model ball mass estimated
from existing draws (the settled local-density readout), then relax first the highest-density distinct draws. This is
shown as an extra arm, "ours + density triage", never in place of the default. A sample-only analogue is offered to
every method: deduplicate raw draws by RDF before relaxing. That separates the gain due to our density from the gain
any method gets from not relaxing near-duplicates.

**Statistics.** At least 3 independent sampling runs per method (agreed 2026-10-26). Development coverage numbers
(97/100 against 999 samples, one seed) were for building the test, never for comparing methods (owner, 2026-10-23).

**Defence.**
- The axes are draws and energy calls, which every method shares.
- The headline counts modes, not Boltzmann mass under an energy the generators never saw. Mass is shown beside it.

### D6. Space-group and Z fairness

**Rule for every method: no draw is ever allocated by the experimental space group or the experimental Z.**

| method class | allocation (recommended) | reported |
|---|---|---|
| ours | Draws across groups follow the model's learned slice weights, a declared convention used to allocate draws at evaluation (owner, 2026-11-02). Never per-molecule knowledge. | Headline under learned allocation. Sensitivity: equal allocation over the groups it covers. |
| needs a group (SALA, quasi-random search, PyXtal, CellNet) | Equal over Day's 26 groups, Day's own rule. Z' = 1. | Headline. A CSD-frequency allocation as sensitivity. |
| P1 generators (Clari, Packora, MolCrystalFlow) | Their own Z choice where they make one (Packora's CSD Z prior). Otherwise Z ∈ {2, 4, 8} by the frequency of primitive-cell Z among CSD Z' = 1 structures in the 26 groups (to be counted in the local CSD), never the experimental Z. | Coverage over all modes and over modes whose primitive Z is in the method's set. MolCrystalFlow was trained only at Z = 2 and 4, so its Z = 8 samples are an extrapolation; flag them. |

Consequences:
- Identity is description-free and relaxation is P1 (D3, D4), so a structure a P1 generator emits in any group counts
  if it matches a mode.
- Modes exist only in Day's 26 groups. Structures found elsewhere are extras (D1b).
- If our model covers fewer than 26 groups, or its cell box excludes some low Day structures, those modes leave its
  reachable set. The current FD box excludes 4.9% of Day's low-energy structures; the Search/prior/FD thread's proposed
  box excludes none. The reachable figure is reported beside the headline, never in place of it.

**Defence.** Every allocation rule is fixed before sampling and uses no knowledge of the molecule's experimental crystal.
Day's equal allocation is the reference's own rule.

### D7. Baseline roster

**Table 3.** Candidate methods: availability on 2026-11-02, licence, what each needs to run on a Day molecule, and the
recommendation. Licences and costs come from the 2026-10-19 scouting report (READ there from primary pages) unless the
row says otherwise. "Have" means installed on this machine.

| method | availability | licence (code / weights) | input on a Day molecule | cost per 1,000 samples | recommendation |
|---|---|---|---|---|---|
| Day's GLEE + DMACRYS search | not runnable here (no DMACRYS) | – | – | 260,000 minimisations per molecule; about 650 CPU-h (U, §1) | **cost reference point** at (260,000 minimisations, 100% of its own reference). Plotted as a marker on the minimisations axis, not a curve. |
| quasi-random search (ours, `random_search.starts`) | have | ours | Day's conformer, 26 groups equal | relaxation only | **core**: the cost-curve proxy for Day's search under our relaxer. Needs the 26-group allocation (currently 2 groups in development). |
| PyXtal random | `pip`, not installed | MIT | conformer and group | relaxation only | optional: a random baseline readers recognise |
| Clari-M | have (`C:/Users/mikem/venvs/clari_cpu`, `clari-med.ckpt` 813 MB) | MIT / CC BY-NC 4.0 | SMILES, Z | 289 s on L40S; runs on CPU | **core** now |
| Clari-H | weights not downloaded (3.07 GB, Hugging Face, not gated) | MIT / CC BY-NC 4.0 | SMILES, Z | about 15 s (L, H100) to minutes | **core** once the download is approved; Clari's strongest public model |
| MolCrystalFlow | have (`C:/Users/mikem/venvs/molcrystalflow`, torch 2.7.1 cu128; checkpoints omc25-mcf, thurlemann23) | MIT / CC BY 4.0 | rigid conformer, Z | about 22 s (B200) | **core**. The `--axis_flip` enantiomer limit matters for non-planar molecules (U). |
| Packora-L | not installed. CUDA-only, torch 2.10 with CUDA 13.0, Python 3.10.12; licensed CSD Python API needed even to sample. | PolyForm NC 1.0.0 / CC BY-NC 4.0 | SMILES (+ RDKit template, stereo labels), Z optional | about 150 s (H100, U) | **core** if it builds on the cluster with a CSD licence; otherwise local GPU, if its CUDA 13 build runs on the RTX 5080 (U) |
| SALA | code on GitHub `WENDI-CAI/SALA` (MIT); weights on Hugging Face `cWendi/SALA`, about 1.95 GB, no weight licence stated (web search 2026-11-02; repository not opened, U) | MIT / unstated | asymmetric-unit graph, space group and Z' required | unknown | **core**, as the only space-group-aware data-trained generator, with equal 26-group allocation (D6). Weight licence to be confirmed. |
| OXtal | public | MIT / CC BY-NC 4.0 | SMILES or SDF; emits a 30-molecule heavy-atom cluster with no cell | 0.4–2 GPU-h | defer to an appendix: needs a lattice extractor and hydrogen placement, which are our own code and a source of failures attributable to us |
| CrystAF | public; Linux, CUDA 12.8; no SMILES entry point | MIT / CC BY-NC 4.0 | Clari crystal object (a SMILES-to-crystal adapter through Clari's own builder is our code) | about 400 s (L40S) | **in** (2026 generator, owner 2026-11-03), on the cluster |
| CG-OMatG, PackFlow, OrgFlow | no code found (scouting report; web search 2026-11-02 for PackFlow and OrgFlow) | – | – | – | **listed** (2026 generators): availability rechecked at each stage; the table states "no public code" while that holds |
| CellNet | not found by web search (2026-11-02); per the 2026-10-28 tables, a lattice flow followed by a PyXtal packing search, needing Hall setting and Z′ | – | – | 1–21 min per cell on about 48 cores | **listed** (2026 generator); run if code is found, with groups allocated by CSD statistics (D6) |
| SymMC-Flow and the PXRD-conditioned rigid flow (F. Cai, ChemRxiv) | diagnostic papers; generation reported at 0% | – | – | – | listed as 2026 work; no run (they report no working generator) |
| EVA-Flow | generates one conformer inside a given crystal; no lattice or packing | – | – | – | listed as out of scope (not a crystal generator) |
| FastCSP (Genarris 3 + UMA relaxation) | open source per its paper, code under `facebookresearch/fairchem` (web search 2026-11-03; repository and licence not opened, U) | U | conformer; Genarris draws over groups and Z | U; the paper's relaxations cost 138–172 V100 GPU-s each | **classical arm, recommended** (owner 2026-11-03: Genarris or FastCSP). Run its Genarris 3 generation stage as the sampler, then our shared relaxation and ledger, so only the proposal distribution differs. |
| Genarris 3 alone | believed open source (U) | U | conformer, Z | U | the same arm as FastCSP's generation stage; choose one after checking the code |
| ours (energy-trained, molecule- and group-conditioned) | in development | ours | Day's conformer; groups by learned weights | inference only; training calls amortised | **core**, entering when trained on the Day-excluded pool with the frame-rule fix in (dependencies below) |

**Leakage per data-trained model.** Clari, Packora and CrystAF share a split that keeps every CSD crystal deposited by
1 May 2025. Only Day families among OXtal's targets (18) or in the CSD Teaching Subset (19) are held out from that line.
OXtal holds out only its 18. Each (model, molecule) pair is flagged from the published split files, and the leak
contrast of the scouting report (§5 there) is run on the stratified set. The split files are Packora's
`csd_clari.csv` (13.0 MB) and OXtal's `train_ccdc_ids.txt` (4.2 MB); downloading them awaits approval (scouting
report, decision 2).

**Defence.** Every generator with public code and weights that emits a periodic cell from a molecule is in. Those left
out are named, with the reason. Every method runs at its authors' recommended settings, apart from the shared
allocation rules of D6.

## 3. Staged to-do list

Effort is agent-days of building and checking, excluding compute wall time. Compute estimates are U unless marked.
Basis: development throughput of exp-6 rigid relaxation on imidazole was about 100 calls/s per CPU process with 2
threads (25 samples × 100 steps in 24 s), and about 1,000 relaxations in 476 s on the GPU (RUN, 2026-10-22/26). A
median Day molecule has about twice imidazole's atoms, and cells up to 8 molecules.

### Stage A: decisions and approvals (no compute)

| # | item | owner / dependency |
|---|---|---|
| A1 | Answers to D1–D7 | owner |
| A2 | Download approvals: Clari-H weights (3.07 GB); Packora checkpoint and split file; OXtal training-ID list; SALA checkpoint (1.95 GB) and its licence | owner |
| A3 | Leak policy D2(a): no Day molecule in our pool | Dataset analysis thread (paper-review session "Dataset analysis and condition-set composition"); owner decision |
| A4 | Frame-rule fix for planar and symmetric molecules (pool D4 route (a)). Our model's Day numbers wait on it: 521 of 1,017 refcodes sit on an ambiguous frame. It does not affect the baselines. | Dataset analysis thread; MXtalTools change |
| A5 | Cell-length box for our model's support (lo 2.5 Å, hi Z·2(r + 3 Å)) | Search/prior/FD thread (paper-review session "Search, prior construction, ellipsoids and FD"); owner decision |

### Stage B: reference and tooling (local, light compute)

| # | item | effort | compute | depends on |
|---|---|---|---|---|
| B1 | `day_landscape`: union of ids over every CSV per refcode (it reads one stem CSV today); join Delta-ML; consensus window; mode merge by complete linkage | 0.5 d | – | D1 |
| B2 | Fix MXtalTools `canonicalize_aunit` refusing P4₂/n (group 86, one of the 26). Two imidazole minima were set aside for it on 2026-10-22. Include a CPU test. | 0.5–1 d | – | MXtalTools change, separate commit |
| B3 | Quasi-random search over the 26 groups, equal allocation, labelled output | 0.5 d | – | D6 |
| B4 | Z prior: frequency of primitive-cell Z among Z' = 1 CSD structures in the 26 groups (local CSD API) | 0.5 d | minutes | D6 |
| B5 | Geometric clash metric; energy-call ledger and reachable-set columns in `protocol_report`; COMPACK (0, 0) failure distinguished from a real result | 1 d | – | D3, D5 |
| B6 | Focused CPU tests: Day block ingest across all 26 groups, substitution keeps the packing, identity and settlement, ledger arithmetic | 1 d | minutes | B1–B5 |
| B7 | Reference build for imidazole and the 16: about 212 minima within 10 kJ/mol each (median), relaxed in P1 to convergence | 0.5 d | about 17 × 2×10⁵ calls; 1–3 GPU-h | B1, B2 |
| B8 | Per-molecule COMPACK calibration for the same 17 | 0.5 d | about 2 CPU-h | B7 |
| B9 | Substitution-placement check on imidazole and 2 of the 16 (owed since 2026-10-26) | 1 d | about 1 GPU-h | B7 |

### Stage C: pilot, exp-6 tier (owner reads the curves; go or no-go to scale)

| # | item | effort | compute | depends on |
|---|---|---|---|---|
| C1 | 17 molecules × {quasi-random (26 groups), Clari-M, MolCrystalFlow} × 3 seeds × 1,000 draws, depths to 1,000 | 1 d of orchestration | 153 units of about 5×10⁵ calls: about 150–500 CPU-process-hours, or about 30–100 GPU-h. Cluster, or about a week locally at the 40% VRAM share. | Stage B |
| C2 | Report: curves, topline, ledger, dropped and extra modes, coverage map; frozen record under `docs/results/` | 1 d | CPU | C1 |

### Stage D: complete the roster on the pilot set

| # | item | effort | compute | depends on |
|---|---|---|---|---|
| D1 | Clari-H (venv exists) | 0.5 d | small | A2 |
| D2 | Packora-L environment, cluster or local | 1–2 d | about 150 s per 1,000 samples (U) | A2; CSD licence on the cluster (U) |
| D3 | SALA environment and adapter (asymmetric unit → P1, labelled) | 1–2 d | unknown | A2 |
| D4 | Optional: PyXtal random | 0.5 d | relaxation only | – |
| D5 | Leak flags per (model, molecule) from the split files | 0.5 d | – | A2 |
| D6 | CrystAF environment (Linux, CUDA 12.8) and a SMILES-to-Clari-crystal adapter | 1–2 d | about 400 s per 1,000 samples (L40S) | Clari-M checkpoint (have) |
| D7 | Classical arm: FastCSP's Genarris 3 generation stage, or Genarris 3 alone, as the sampler; our shared relaxation | 1–2 d | U | code and licence check |
| D8 | Availability recheck for CG-OMatG, PackFlow, OrgFlow and CellNet; run whichever has released code | 0.5 d per release | – | – |

### Stage E: MLIP tier (cluster): DEFERRED by the owner (2026-11-03) until the training MLIP is chosen

| # | item | effort | compute | depends on |
|---|---|---|---|---|
| E1 | Day's MACE model through the MXtalTools MACE route, rigid, in P1 up to 8 molecules per cell; cost per call measured | 1–2 d | small | – |
| E2 | Validation gate on the 16 against `MACE_Reopt_Landscapes`: rank correlation and COMPACK identity | 1 d | about 16 × 200 relaxations | E1 |
| E3 | Reference rebuild and pilot rerun on the 16 under MACE; depth cap of 300 if the cost per call demands it | 1 d | if a MACE call costs 30–100 exp-6 calls (U): about 2×10³–2×10⁴ GPU-h at full depth, which the cap and fewer seeds cut | E2, Stage D |
| E4 | Optional: UMA rigid sensitivity on a subset of the 16 | 1 d | cluster | E1 |

### Stage F: scale and our entry

| # | item | effort | compute | depends on |
|---|---|---|---|---|
| F1 | Stratified 100: reference build and calibration; all core methods, exp-6 tier, 3 seeds | 2 d | about 6× Stage C | Stage C go |
| F2 | Our model on all 1,017 refcodes, plus quasi-random, exp-6 tier, as the supplementary generalisation table | 1 d | our sampling plus relaxation | A3–A5, the trained model (AR architecture and conditioning session) |
| F3 | Ours on the pilot set and the stratified 100, both tiers; triage arm (D5) | 1 d | as Stage C per method | F2 |
| F4 | Paper figures and frozen records under `docs/results/` | 2 d | CPU | all |

### Dependencies on other sessions, in one place

- **Dataset analysis and condition-set composition.**
  - The leak policy decides whether all of Day is held out from our model.
  - The frame-rule fix gates our model's Day numbers.
  - The Day molecule keys decide the SMILES fed to Clari and Packora.
  - The held-out definitions decide what our generalisation table claims.
- **Search, prior construction, ellipsoids and FD.**
  - The cell box sets our model's reachable set.
  - The FD metric port to the 26 groups gives a within-group identity cross-check.
  - The Day ground-truth test of our search under exp-6 tells whether exp-6 re-relaxation keeps Day minima in their
    basins. That is the main validity question for the development tier, so its result should be read before Stage C
    is reported.
- **AR architecture and conditioning.**
  - The space-group head and its slice weights are our allocation rule.
  - The harness's "Day set C" needs Day minima in the trainer's chart, which is blocked on the frame rule.

## 4. Still open after 2026-11-03

- **MLIP choice.** It gates D1 tiers, the D1b window, and Stage E. It is an owner decision, taken with the choice of
  training MLIP.
- **Conformer placement (D2), on hold.** The option set is to be coded. Early substitution is the development default.
- **Pilot molecule subset.** Recommended: imidazole plus a few molecules chosen to span size and hydrogen-bond donors,
  with at least one polymorphic molecule. It is small enough to run locally.
- **Optimiser settings** from field practice (D3 note): fmax 0.02 eV/Å, 500-step cap. Proposed, not decided.
- **Severe-overlap rule.** Proposed: any intermolecular heavy-atom pair closer than 0.75 × the sum of Bondi radii, a
  knob. Exp-6 drops out of the severe role.
- **CSD space-group statistics** for group-needing baselines, and for ours as the worst case (D6). Count them from the
  local CSD at Z' = 1, together with the Z prior (task B4).
- **Downloads in A2**, plus the CrystAF and FastCSP/Genarris code, each on the owner's approval.

## Sources

- Taylor, Butler & Day, Faraday Discuss. 256 (2025) 434-458, DOI 10.1039/D4FD00105B, and its ESI. Deposit DOI
  10.5258/SOTON/D3094, local copies as in Table 1.
- `energy_sampling/docs/design/generator_benchmark_protocol.md` (protocol and owner decisions of 2026-10-26).
- `energy_sampling/artifacts/generator_scouting_oct19/generator_scouting_report.md`. It is git-ignored; it holds the
  roster, the leakage tables and the development runs under `protocol_oct26/` and `fair_oct23/`.
- Private artifact "CSP and Boltzmann Generator Tables" (2026-10-28), Table 1.
- Shared notes of the paper-review sessions (2026-11-02): `data_defects`, `day_like_pool_proposal`,
  `fd_length_box_proposal`, `ar_fit_harness`.
- Wiki: `structure-comparison-metrics`, `landscape-structure`, `crystal-force-fields`, `mlip-energy-routes`.
- Web search 2026-11-03 for FastCSP code (FastCSP, arXiv:2508.02641; code link `facebookresearch/fairchem`).
- Web search 2026-11-02 for SALA, PackFlow, OrgFlow and CellNet code. Abstract-level results only; repositories not
  opened.
