# Typed intermolecular force field for molecular crystals — implementation and benchmarks

**Date:** 2026-08-25 · **Status:** exploratory; nothing landed in either repo · **Compute:** CPU only

This records what was built, why it was built that way, and what was measured. It is not a
recommendation and it is not a conclusion — several quantities below are unmeasured or
measured at low precision, and those are marked.

Origin: *"we have an intramolecular FF we trust; how hard is a molecular crystal FF?"* →
*"something more sophisticated than ELJ, at ELJ cost"* → *"more physical fidelity —
H-bonds, per-pair energies, maybe simple electrostatics."*

---

## 1. Implementation

### 1.1 Parameter source and functional form

CCDC/Mercury **speculator exp-6** set: `A·exp(−B·r) − C/r⁶`, 75 potentials, 274 type-pair
assignments (27 of them H-bond pairs), 23 atom types with SMARTS definitions.

```
C:\Users\mikem\CCDC\ccdc-software\mercury\templater\parameter_files\
    speculator_potential_6_exp.txt     # potentials + pair assignments
    speculator_atom_types_6_exp.txt    # 115 SMARTS typing rules
```

Sanity anchor: C···C → A = 226164.7, B = 3.4376, C = 2418.07, giving r_min 3.961 Å and
ε 0.3501 kJ/mol (0.084 kcal/mol — the expected physical value, confirming kJ/mol units).

**Provenance is undetermined.** No citation header anywhere in the install. Two pieces of
evidence suggest an original fit rather than a redistribution of published Williams
parameters: (a) the B values are unit-free (Å⁻¹) yet carry 6 significant figures with no
rounding — 4.00699, 3.43764, 3.75924 — where a literature table would show 3.60, 3.74;
(b) heteronuclear B values do not follow any combining rule from the homonuclear ones
(C···C 3.43764, H···H 4.00699 → arithmetic mean 3.7223, geometric 3.7113, but C···H is
4.04782, outside both), so every pair was fitted independently.

The implementation is parameter-agnostic — `spec_ff.py` reads a table — so an open
replacement (published Williams W99/W84, GAFF/OPLS LJ, or a fit of our own) is a table swap.

### 1.2 Evaluation form

Two forms were evaluated:

- **native exp-6**, spliced **linearly below the inner maximum** to remove the Buckingham
  catastrophe. The splice radius is a per-pair root-find done once at setup (0.34–0.87 Å,
  far inside any physical contact).
- **ELJ form** — the existing `4ε[(σ/r)¹²−(σ/r)⁶]` with matched exponential core below σ —
  with (σ, ε) obtained by **matching the exp-6 minimum**: σ = r_min/2^(1/6), ε = −E(r_min).

The ELJ form was used for the reported benchmarks so the comparison against production ELJ
differs only in parameters, not functional form. `k_factor` left at the production 2.5.

Buffered 14-7 (`buffered_147`) was considered and set aside: it is singularity-free and
already RDKit-verified, but its attractive branch is r⁻⁷ against dispersion's r⁻⁶, recovering
76% of a 12-6 tail at 6 Å and 44% at 12 Å. Acceptable for intramolecular pairs, not for a
lattice sum over many shells.

### 1.3 Atom typing

`spec_ff.py::type_atoms` implements the SMARTS rules as graph logic on a distance-derived
bond graph — no RDKit at runtime, hence no atom-ordering problem, and fully batched.

**Typing runs on the asymmetric unit, not the cluster.** `build_radial_graph` constructs
`edge_index` with `convolve_inds=inside_inds`, so intramolecular bonds exist **only** for the
asymmetric unit. Measured on a real cluster: 90,400 atoms, 25 inside, and all 90,375
periodic-image atoms carry zero intramolecular edges. Typing on that graph gives every image
atom `h_cnt = 0`, `hv_deg = 0` and drops it to a generic element type.

Propagation is by riding along in `x`: `_instantiate_cluster` already does
`cluster_batch.x = crystal_batch.x[cluster_node2aunit_node]`, so packing the type index as a
second column of `x` replicates it exactly like `z`, with no change to mxtaltools.
(`get_intermolecular_dists_dict` already branches on `x.ndim == 2` and reads charges from
column 0.) A dedicated attribute plus one line in `_instantiate_cluster` would be the
cleaner form if this is landed.

Two rule details that were got wrong first time and matter:

- `[OX1]` is **total** connectivity 1, not heavy-degree 1. Counting heavy neighbours only
  makes every hydroxyl O read as terminal, which marks its carbon acyl and promotes every
  alcohol to an acid. `ACID_OXYGEN + ACID_HYDROGEN` has no H-bond table entry, so alcohol
  H-bonds then fall through to plain `H_O`.
- `AMIDE_NITROGEN` is a misnomer for `[NX3H1]` — any N with 3 connections and 1 H,
  carbonyl or not. Likewise `PRIMARY_AMIDE_NITROGEN` = `[NX3H2]`, and
  `ACCEPTOR_NITROGEN` = `[NX1]`/`[NX2]` by total connectivity.

### 1.4 Electrostatics

Shifted-force Coulomb, `q_i q_j (1/r − 1/r_c + (r−r_c)/r_c²)`, zero in value and slope at the
cutoff. Charges are the stored Gasteiger values in `x[:,0]`.

Gasteiger under-polarises, and the vdW parameters were fitted with some charge model of
their own, so double-counting is possible. MMFF bond-charge-increment charges are available
free from the RDKit properties object `ff_from_mmff` already builds and were not tried.

### 1.5 Cost

Typing runs on the asymmetric unit before cluster construction, so it never pays the
Z × n_ucells multiplier.

| batch | GPU (RTX 5080) | CPU (24 threads) |
|---|---|---|
| 512 mols / 15k atoms | 0.78 ms | 7.6 ms |
| 4096 mols / 123k atoms | 1.63 ms | 51.4 ms |
| 20000 mols / 600k atoms | 7.20 ms | 228.9 ms |

Energy evaluation is the same edge list ELJ already builds; the added cost is two table
gathers and, with electrostatics, one extra term per edge.

---

## 2. Benchmark design and reasoning

**Reference.** UMA (`esen_s.pt`) lattice energy =
`(uma_pot/(sym_mult·z_prime) − uma_gas_pot) × 96.485`, CPU, ~0.3–0.5 s/crystal.

Validated for internal consistency by isotropic cell expansion — the lattice energy must go
to zero for isolated molecules, and does: 0.3 / −0.2 / −1.5 kJ/mol at ×2 and flat to ×4. So
the gas leg is consistent with the crystal leg. **UMA's absolute accuracy for molecular
crystals is not validated** — no comparison against experimental sublimation enthalpies was
run, and `esen_s` is among the smallest UMA models.

**Structures.** 62 organic Z'=1 well-defined CSD crystals (14–154 atoms) from
`mini_new_prot_csd.pt`, latent-noised via `latent_params()` → Gaussian → `latent_to_cell_params()`,
which perturbs cell lengths, cell angles, aunit centroids **and orientations**. 12 per
molecule at σ ∈ {0, 0.002, 0.005, 0.01} → 744 structures. Zero-noise round-trip verified at
1.9e-6 Å on cell lengths.

**Physical-envelope filter.** Heavy-atom minimum contact ratio `r/(R_i+R_j)` discriminates
clashes; the all-atom ratio does not, because real H-bonds legitimately reach 0.65. The 62
experimental structures set a floor at **0.805** (median 0.961). Fraction of generated
structures below that floor:

| latent σ | below floor |
|---|---|
| 0.002 | 0.5% |
| 0.005 | 2.2% |
| 0.010 | 8.6% |
| 0.020 | **32.3%** |

σ = 0.02 is not a local perturbation. Benchmarks use σ ≤ 0.01 filtered at the floor →
**712 structures**.

**Why calibrated comparison.** ELJ has ε = 1 and is in arbitrary units, so it cannot appear
on a kJ/mol axis without a scale. All variants are therefore compared after **one global
calibration** — a single slope + intercept fit once over all 712 — which is the analogue of
what `lj_rescale` already does. Per-panel or per-subset fitting was tried and rejected: it
is degenerate under a heavy-tailed potential (a single 121,431 kJ/mol outlier drove a slope
to ×0.005 and collapsed every point onto a vertical line).

**Why rank statistics are weak here.** Within-molecule UMA spread at σ ≤ 0.01 is
**5.8 kJ/mol** (median std) while every model's within-molecule residual is 6.0–7.5 kJ/mol,
so signal-to-noise ≈ 1. Confirmed by stratifying molecules on their own signal: mean
Spearman rises 0.57 → 0.79 as per-molecule UMA std goes 3.0 → 11.0. **Within-molecule ρ
differences of ±0.03 between variants are not measurable on this design.** For context,
polymorph pairs are typically separated by ~2 kJ/mol.

---

## 3. Benchmark results

### 3.1 The ladder

712 structures, one global calibration, ELJ form throughout, `k_factor` 2.5.

| variant | MAE (kJ/mol) | Pearson | within-mol ρ | calib slope | bias no-HB | bias HB |
|---|---|---|---|---|---|---|
| ELJ (production) | 24.857 | 0.449 | 0.621 | 0.057 | −13.1 | +20.9 |
| Williams, element-only | 23.313 | 0.483 | 0.591 | 0.220 | −13.3 | +21.1 |
| Williams, full typing | 14.359 | 0.868 | 0.656 | 0.406 | −4.9 | +7.8 |
| Williams, full typing + electrostatics | **12.634** | **0.896** | 0.624 | 0.362 | −1.5 | +2.3 |

Element-only typing is worth little (23.3 vs 24.9). The gain is in the **specialised pair
terms** — full typing takes MAE 23.3 → 14.4 and Pearson 0.483 → 0.868. Electrostatics adds
14.4 → 12.6 and 0.868 → 0.896.

Bias on H-bonding structures goes +20.9 → +2.3 kJ/mol, and both populations end
near-unbiased (−1.5 and +2.3), i.e. the model is not trading one against the other.

Within-molecule ρ is unchanged within the noise floor established in §2, as expected.

### 3.2 Scale vs offset

The calibration slope of 0.36 is a comparison of **absolute** energies across different
molecules. Separating that from the response to geometry change, using the geometric mean of
both OLS directions (the appropriate estimator when both variables carry error):

| | absolute (across molecules) | deltas (within molecule) |
|---|---|---|
| Williams full + elec | 0.40 [0.362, 0.451] | **0.65** [0.463, 0.910] |
| ELJ (production) | 0.13 [0.057, 0.281] | 0.28 [0.215, 0.367] |

Residual per-molecule offset at the within-molecule scale: Williams **std 23.7 kJ/mol**
(range −57 to +87), ELJ std 60.9 (range −159 to +226).

So the absolute discrepancy decomposes into a per-molecule offset plus a response-scale
error of roughly 1.5×, not the ~2.8× the absolute slope suggests. Spot checks show the
absolute ratio is molecule-dependent rather than constant — FACRIK04 0.77× (FF underbinds),
FAXZEK 1.98×, BINQAQ 1.89×.

**Precision caveat:** the delta brackets are wide because within-molecule S/N ≈ 1. Read 0.65
as "somewhere around 0.5–0.9", not a measured constant. Tightening it would need larger
perturbations, which leave the physical envelope.

### 3.3 Acceptance battery

Aimed at the recurring failure mode — a model applied to part of its domain returning a
smooth plausible energy. Every check is a count or a cross-implementation comparison, never
"is the energy reasonable".

| check | result |
|---|---|
| Per-atom typing vs real RDKit SMARTS, 62 molecules | **PASS** 2707/2711 = 99.85% |
| Every periodic image carries the aunit type vector | **PASS** 520 images × 30 atoms |
| Every intermolecular edge resolves to a real potential | **PASS** 0 of 13,356 with A = 0 |
| Homonuclear pairs present in the table | **PASS** |
| Re-introducing hv_deg-only `termO` is detected | **PASS** |
| Re-introducing cluster-typing is detected | **PASS** |
| Translation invariance | **PASS** |
| Batch invariance (alone vs in a batch) | **PASS** |

The 4 residual typing disagreements are aromatic 3-connected NH, where RDKit's aromatic `n`
perception differs from the graph rule.

The battery found two typing bugs (§1.3) in code that had already been reviewed and
benchmarked.

### 3.4 Cutoff behaviour

| cutoff | vdW only | vdW + Coulomb |
|---|---|---|
| 8 Å | −254.44 | −293.73 |
| 10 Å | −266.71 | −307.73 |
| 12 Å | −272.16 | −313.96 |
| 14 Å | −262.49 | −298.33 |
| 16 Å | −225.28 | −270.66 |
| 18 Å | −187.81 | −229.24 |

**Not converged at 10 Å** — ~18 kJ/mol still moving between 8 and 12 Å. Beyond ~12 Å the
scan is not measuring physics: intermolecular edge count saturates at exactly 31,031 while
the cluster grows 36k → 48k atoms, and max neighbours per atom falls 31 → 19 → 16. That is
neighbour-list truncation. At the production cutoff of 10 Å the cap is not binding
(13,252 edges).

Note the direction: more tail makes the FF *more* negative, so this cannot explain the
overbinding — it works against it.

### 3.5 H-site shift ablation

Williams-family potentials are sometimes defined with H interaction sites shifted along the
X–H bond toward the heavy atom. Tested as a candidate explanation for the scale discrepancy:

| H shift | calib slope | Pearson |
|---|---|---|
| 0.00 Å | 0.360 | 0.915 |
| 0.10 Å | 0.377 | 0.920 |
| 0.20 Å | 0.390 | 0.912 |

Slope moves 0.360 → 0.390 across a 0.2 Å shift. **This does not explain the discrepancy.**

---

## 4. What is not measured

- UMA's absolute lattice energies against experiment (benzene ≈ −50, naphthalene ≈ −72,
  anthracene ≈ −100 kJ/mol would be the obvious set). The definition is validated; the values
  are not.
- Whether `esen_s` is an appropriate reference for molecular crystals at all. Separately,
  `mace_utils.py:19` loads `mace-mpa-0-medium.model`, an inorganic materials foundation model,
  on molecular crystals.
- Cutoff convergence beyond 12 Å, blocked by neighbour-list truncation.
- MMFF charges in place of Gasteiger.
- Directional H-bonding. An isotropic well reproduces H-bond distance and energy but is blind
  to D–H···A linearity and acceptor lone-pair direction. The donor/acceptor index already
  exists in `old_new_hydrogen_bond_analysis`, which builds it and discards the geometry.
  Given §2, this would need evaluating on absolute energy, not ranking.
- Reconciliation with the parallel reference run, which reported 18.221/0.739 and
  15.537/0.798 against 14.359/0.868 and 12.634/0.896 here. Discriminator: whether that
  implementation types `DMANTL23` (mannitol) as six `ALCOHOL_*` (correct) or six `ACID_*`
  (the `[OX1]` bug).
- Parameter provenance and licensing (§1.1).

---

## 5. Errors found during the work

Recorded because four separate bugs shared one failure mode: **a model applied to only part
of its domain, returning a smooth, finite, physically plausible energy.** Energy-level
assertions caught none of them; counts and cross-implementation comparisons caught all.

**(1) Homonuclear pairs silently dropped.** Parsing pair names by `k.split("_")` gives
`['C','C']` for `C_C`; the element lookup then yields a one-element list and a
`if len(zs)<2: continue` guard skipped it. H···H, C···C, N···N, O···O, S···S, Cl···Cl were
all absent — the two most common contacts in an organic crystal among them. Present in
`compare2.py` and independently in `ffbank.py`. The broken potential appeared to agree with
UMA *without fitting* (bias −11.5 kJ/mol) because missing attraction cancelled against real
overbinding.

**(2) Three variables changed at once.** An early "fitted LJ vs ELJ" comparison moved
per-pair ε, per-pair σ, `k_factor` and H-bond types together. An ablation with a proper
control showed per-pair ε was a no-op (0.829 → 0.832) and `k_factor` was the entire effect.
A later run repeated the mistake by changing typing and functional form together.

**(3) `k_factor` optimum of ~7.** Measured on a set where 32% of structures were outside the
physical envelope; a stiffer wall is rewarded for matching UMA on clashed geometry. On the
filtered set k = 7 makes ELJ worse (Pearson 0.449 → 0.332, MAE 35.8 → 41.6).

**(4) Periodic copies never typed** (§1.3). Suppressed specialised pair terms almost
entirely — H-bond-tagged edges went 0 → 75,170 on correction.

**(5) `[OX1]` as heavy-degree** (§1.3). Typed every alcohol as an acid, which routed alcohol
H-bonds to a table entry that does not exist.

Also worth noting, independent of the force field: **stored CSD crystals require
`std_orientation=True`.** `pose_aunit(std_orientation=True)` reproduces stored `pos` to
0.000 Å; `False` lands 5–7 Å away and produces 1.6–2.7 Å vdW overlaps on experimental
structures, `lj` to 1.7e10, and UMA lattice energies of +650 to +1330 kJ/mol. The GFN's
`analyze_kwargs` uses `False`, which is correct for generated poses and wrong for stored ones.
Recorded separately at
`~/.claude/projects/.../memory/project_stored_csd_needs_std_orientation.md`.

**Dataset note:** `mini_new_new_csd.pt` and `mini_reduced_CSD_dataset.pt` contain **0%
hydrogen**. `mini_new_prot_csd.pt` is 44.5%, the conditional prior 30.8%, `eval_qm9_sg2` 49.1%.
On a heavy-atom-only dataset every H parameter silently never fires.

**Hydrogen positions are already normalised** — C–H 1.089–1.093 Å, N–H 1.014–1.015,
O–H 0.964–0.993, i.e. neutron/standard values rather than raw X-ray (~0.95 / ~0.85). No X–H
normalisation step is needed.

---

## 6. Paths

### 6.1 Scratchpad — **session-scoped, will not survive**

```
C:\Users\mikem\AppData\Local\Temp\claude\C--Users-mikem-Projects-mxt-gfn-gfn-diffusion-energy-sampling\f875f960-a680-4ea5-a26b-1471b1aeccdd\scratchpad\
```

| file | contents |
|---|---|
| `spec_ff.py` | The force field: parses both CCDC files, `type_atoms()` graph typing. 525/529 pairs filled. |
| `run_fixed.py` | Benchmark driver with aunit typing. Reproduces stored ELJ **bit-exactly** from seed 23, so it extends without re-running UMA. |
| `final.py` / `final.npz` | UMA reference, 744 structures, 402 s CPU, with `hmin` for the envelope filter. Keyed to seed 23 + molecule order. |
| `battery.py`, `battery2.py`, `battery3.py` | Acceptance battery (§3.3), invariances, cutoff scan, H-site ablation. |
| `calib.py`, `geom.py` | Envelope calibration (§2). |
| `plot_final.py`, `parity_eljvsfitted.png` | Parity figure. |
| `type_bench.py`, `real_test2.py` | Typing cost and accuracy. |
| `b147.py`, `exp6fit.py` | Buffered 14-7 analysis; exp-6→LJ tail relation `C_LJ/C_exp6 = 2(1−6/α)`. |

**Superseded, contains bug (1):** `ffbank.py` and everything downstream —
`compare*.py`, `gather.py`, `within*.py`, `hb*.py`, `bigrun.py`, `ablate*.py`, `ksweep.py`,
and figures `parity.png`, `parity_big.png`, `parity_hb.png`, `parity_local.png`,
`parity_robust.png`, `parity_final.png`, `ksweep.png`, `parity_nofit.png`.

### 6.2 Repo files referenced

```
mxtaltools/mxtaltools/analysis/vdw_analysis.py
    :37   electrostatic_analysis           # Yukawa screen, retains 7.6% at 2.8 A
    :52   buckingham_energy                # exp-6 skeleton, placeholder A/B/C
    :103  old_new_hydrogen_bond_analysis   # donor/acceptor index, geometry discarded
    :190  get_intermolecular_dists_dict    # branches on x.ndim, reads charges from col 0
    :277  exponential_edgewise_lj_energy   # ELJ
    :355  compute_lj_edgewise              # sigma = radii sum, eps hardcoded `4 * 1 *`
mxtaltools/mxtaltools/models/functions/radial_graph.py
    :218  edge_index built with convolve_inds=inside_inds   # why cluster typing fails
mxtaltools/mxtaltools/crystal_building/utils.py
    :325  _instantiate_cluster             # replicates x via cluster_node2aunit_node
mxtaltools/mxtaltools/conformers/energy.py
    :445  ff_from_mmff                     # MMFF params incl. charges
    :693  buffered_147
mxtaltools/mxtaltools/dataset_utils/data_class_methods/crystal_ops.py
    :266  latent_to_cell_params / :312 latent_params        # the noising path
mxtaltools/mxtaltools/dataset_utils/mol_building.py
    :348  get_partial_charges              # Gasteiger
gfn_diffusion/energy_sampling/energies/molecular_crystal.py
    :488  lj_rescale                       # existing one-scalar calibration
```

### 6.3 Run recipe

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="C:/Users/mikem/Projects/mxt_gfn/mxtaltools;C:/Users/mikem/Projects/mxt_gfn/gfn_diffusion" "C:/Users/mikem/venvs/csd_mxt_gfn/Scripts/python.exe" <script>.py
```

`GFN_GPU_GUARD=0` set in-script.
