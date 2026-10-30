# Generator benchmark protocol

Status: **proposal**, drafted and revised 2026-10-26. The protocol is the owner's outline (2026-10-26) below; owner
decisions are dated, everything else is labelled a proposal, working assumption or knob. It controls nothing
executable; the development implementation is research code under `energy_sampling/eval/day_bench/` (committed 2026-10-29 as it stood; not a contract).

**Question.** How much of a molecule's low-energy crystal landscape does a structure sampler find, and at what cost?
One protocol for every sampler: generative models trained on data (Clari, MolCrystalFlow, ...), random search, and our
energy-trained samplers.

**Fairness principle (owner, 2026-10-26).** The samplers are not trained on our energy reference, so the benchmark
punishes serious clashes and missing basins, not subtle misrankings.

## Protocol

Per molecule, identical for every sampler:

1. **Draw samples.** N raw samples per sampler (1,000 in development), from at least 3 independent sampling runs.
2. **Rigid substitution** (a design choice): Day's own conformer replaces each molecule of each raw sample, which keeps
   its packing (molecule positions and orientations).
   1. **Score raw, as-is:**
      1. *coverage* of the reference by RDF: which reference modes have a sample within the identity tolerance;
      2. *legit molecules*: the sample's bond graph is intact (checked at substitution, which needs it);
      3. *severe*: very high energy under exp-6.
   2. **Optimize N steps**, then rescore with the same three.
3. **Analysis:** coverage against draws, at each optimization depth; effort = draws x steps, so one family of curves
   (one per depth, raw first) carries both axes, and the topline numbers are read off it.

Knobs, with their starting settings:

| knob | starting setting | basis |
|---|---|---|
| identity tolerance | an RDF cut at the 90% point of a logistic fit of COMPACK matches against RDF distance, with a match at 15/20 or at 10/20 molecules (both to be looked at); for small sets, COMPACK RMSD over 15 or 10 molecules directly | owner, 2026-10-26: "a knob for now"; COMPACK is standard though not loved for this work |
| severe | exp-6 lattice energy far above the reference minima: "really high, not unreasonably tight such that you would disagree with the whole packing mode under a different potential"; *proposed* first line: above zero per molecule (net repulsive; imidazole's minima sit near -75 kJ/mol) | owner, 2026-10-26: exp-6 is van der Waals plus hydrogen bonds and some electrostatics, so a structure it punishes badly is probably bad |
| optimization depths | 0 (raw), 10, 30, 100, 300, 1,000 steps of the shared rigid-body optimizer | owner, 2026-10-26: pick a set, look, then add or remove |

## Notes per step

**Reference.** The molecule's minima from the Day group's rigid-molecule CSP landscapes (Taylor, Butler & Day 2025,
deposit 10.5258/SOTON/D3094) within 10 kJ/mol of the lowest, re-relaxed under the benchmark energy. *Proposed:* minima
that are one structure under the identity tolerance count as one mode; mode weights for Boltzmann-weighted coverage
are Boltzmann factors at 300 K on the benchmark energy's own ranking, with Day's DFT-corrected ranking as a
cross-check.

**1. Sampling.** Cell size: the sampler's own choice where it makes one; otherwise the cell sizes the reference's
minima need (molecules per primitive cell), with coverage also reported over the minima those cell sizes can
represent. Sampling compute (GPU time) is reported beside the curves, not on them.

**2. Substitution.** **Decisions (owner, 2026-10-26):** Day's own conformer, the molecule the reference landscape was
computed with ("if we use Day's energy model it's doubly consistent. We can always change our minds later");
substitute first, for now; lenient up to a broken bond graph, which is a failed sample counted against the budget, the
fit RMSD recorded but not thresholded. Rigid-input samplers (MolCrystalFlow) receive the same conformer as input.
Considered: substituting at the end, after flexible optimization of the sample as generated (owner: "might actually
be the fairest", but the substitution may itself need relaxation). *Proposed check:* per molecule, about 100 samples
through both paths; if their endpoints agree within the identity tolerance the placement is immaterial for that
molecule, and a molecule that fails is not rigid enough for a constant conformer.

**2.1 Identity.** Owner, 2026-10-26: reconsider the identity rule, not Day's 30/30 at 30% for everything. One minimum
sits at slightly different geometries under different credible energies, and a sample at another energy's version of a
minimum should count as finding it (fairness principle). Observed in development: the deposited IMAZOL03 structure lies
at RDF 0.146 from its exp-6 relaxation, beyond the 0.093 cut calibrated against COMPACK 30/30 on exp-6 endpoints, so a
raw sample identical to the experimental form would have failed that cut. Observed, imidazole development run
(2026-10-26, `protocol_compack`, 350 pairs over all depths, 25 per 0.025-wide RDF bin on [0, 0.35]): a match at 15/20
gives tolerance 0.086 (bootstrap 5-95% 0.078-0.097), at 10/20 gives 0.136 (0.126-0.147). The COMPACK calibration is done once per
molecule landscape and reused for every sampler (a few hundred 20-molecule comparisons spanning RDF distances; about
half an hour on 12 CPU workers at the development speeds). Other candidates for the knob: the thermal
radius at 300 K (the RDF spread that vibration about the minimum produces, from Hessian-drawn displacements).

**2.2 Optimization.** One rigid-body optimizer for every sampler; each step is one energy-and-gradient call, so a set
at depth k costs draws x k calls. The optimizer is FIRE on the generalised forces (`converge.fire`), a row stopping
once its largest force is at most 0.01 kJ/mol per Angstrom or per radian, with a step cap of 0.1 in its length-like
coordinates. *Observed* (25 raw MolCrystalFlow samples, exp-6, development): median energy per molecule +1,376 raw,
-59 kJ/mol after 100 steps and -72 after 300 with that cap, against +29 and -48 with FIRE's polishing cap of 0.02.
Energies: *development:* FIT exp-6 with MMFF point charges (MXtalTools `exp6`
compute); its numbers are not claims. *Final:* **Day's MACE model** (`MACE_total_energy`, PBE+D3, the model Day used
for geometry optimisation; owner, 2026-10-26); *check:* for the 16 molecules Day re-optimised with it
(`MACE_Reopt_Landscapes`, energies only), our relaxation of Day's FIT+DMA structures should reproduce their energies
and rankings. The severe score stays exp-6 in both (owner, above).

**3. Analysis.** Coverage is counted over the reference modes within a window, unweighted, and Boltzmann-weighted
beside it. Topline numbers (*proposed*), each tied to a failure mode the owner listed (2026-10-26):

| number | read from | failure mode it flags |
|---|---|---|
| raw coverage at 1,000 draws | the depth-0 curve | small but nonzero probability on many modes; collapse |
| raw legit and severe rates | depth 0 | poor raw samples |
| effort to half coverage | the best curve at each effort | random-like; slow cleanup (shallow slope) |
| reach: coverage at the largest effort | the deepest curves | levels off below 1: modes missed for good |

Beside them: the map (the reference's modes as a UMAP terrain of RDF distances, each sampler's structures on it, modes
marked found or missed); missed modes stated as "found in fewer than 3/N of samples" (95%). Diagnostics for samplers
that claim a Boltzmann target (ours), descriptive only for data-trained ones (owner, 2026-10-26: expecting them to
respect an energy distribution is unfair unless the energy is very good): effective temperature (slope of log
occupancy against mode energy), effective number of modes (1 / sum of squared occupancies; imidazole at 300 K: 39), and
where P(x) exists the spread of log P(x) + E(x)/kT over raw samples. For energy-trained samplers, energy calls spent in
training are reported beside the curves as an amortised cost.

## Development implementation (untracked, `energy_sampling/eval/day_bench/`)

- `day_landscape.py`: Day's FIT+DMA structures into the pipeline (cspy CIF blocks, Z' = 1 in their space group).
- `p1_ingest.py`, `p1_relax.py`: a sample as a P1 cell of Z rigid conformers; a fixed two-stage first relaxation.
- `converge.py`: FIRE to a force threshold, energy calls counted.
- `random_search.py`: the random-search baseline.
- `coverage.py`: RDF distances, identity (`settled_identity`), coverage-against-calls curves.
- `compack_band.py`: COMPACK on an RDF band, capped per mode.
- `coverage_map.py`: the map.

The protocol itself (2026-10-26):
- `day_landscape.load_reference`: the constant conformer, `day:<stem CIF>:<structure id>` for Day's own.
- `protocol_refine.py`: substitution, the legit score and fit RMSD, then `converge.fire` from raw with snapshots at the
  depths; each depth's structure written with a molecule label per atom, so overlapping molecules still split. A
  sample's molecules come from its own labels, else its bond graph, else consecutive atom blocks rejoined along their
  own bonds (overlapping molecules join in the bond graph, which is a clash for the severe score, not a broken molecule).
- `protocol_random.py`: random-search raw samples (`random_search.starts`, constant conformer), written labelled:
  random cells can be smaller than the molecule, so a plain CIF of one cannot be split back into molecules.
- `protocol_score.py`: RDF distances from every structure at every depth to every reference minimum.
- `protocol_compack.py`: the identity knob's COMPACK calibration (20-molecule shells, logistic fit per match level).
- `protocol_report.py`: modes, the found matrix (legit, not severe, within tolerance), the curves against draws and
  effort, the topline table.

Gaps against the protocol: the substitution check, repeat sampling runs, the diagnostics, the map on protocol outputs,
the MACE route and its check.

## Open

- Settings of the three knobs after a first look.
- Training-data leakage: which benchmark molecules each data-trained model saw (see the 2026-10-19 scouting report,
  `energy_sampling/artifacts/generator_scouting_oct19/generator_scouting_report.md`, not in version control).
