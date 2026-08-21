# Nikos' acridine structures vs. our brute-force landscapes

Do the landmarks on our landscapes — low-energy minima, dense basins — correspond
to the structures Nikos sent?

Four scripts, run in order. Each writes into `out_dir` (see `config.yaml`) and
reads only what the previous one wrote.

```bash
python -m energy_sampling.eval.nikos_comparison.ingest
python -m energy_sampling.eval.nikos_comparison.levels --write-cifs
python -m energy_sampling.eval.nikos_comparison.compare
python -m energy_sampling.eval.nikos_comparison.controls
python -m energy_sampling.eval.nikos_comparison.modes
python -m energy_sampling.eval.nikos_comparison.identify
```

`controls.py` is **not optional**. Neither an energy nor an RDF distance from
`compare.py` means anything without it — see *Read nothing without the controls*
below. Add `--device cpu` to any of them; only L2 needs a GPU.

The answer lands in `nikos_comparison.csv`: one row per structure of his, with
where it sits on our energy surface, its nearest sample in each of our
landscapes, the RDF distance to it, and the COMPACK RMSD that confirms or denies
the match.

## What each stage does

**`ingest.py`** — his CIFs into our `MolCrystalData` format, through the same
ccdc-backed helpers that built our own crystal datasets.

**`levels.py`** — puts his structures on our energy surface at three levels:

| | | |
|---|---|---|
| **L0** | as-given | his atoms, his cell |
| **L1** | reprojected | *our* reference conformer, rigid, at his pose and cell |
| **L2** | relaxed | L1 relaxed on our MACE surface (cell + pose; molecule rigid) |

L1 is a bridge, not an extra step. Our landscapes are rigid-body searches over
one fixed acridine conformer, so a structure has to be expressed in those terms
before it can be relaxed in them or compared against them. The **L0→L1 RDF
distance is a guard, not a result**: it measures how much of his structure
survived the swap to our molecule, and a structure with a large gap did not come
through — its L1/L2 numbers then describe something else.

L2 is the actual basin-correspondence test. If his structure relaxes into one of
our minima, that *is* the correspondence, whether or not the unrelaxed structures
looked alike.

An all-atom relaxation (L3) is expected from a colleague. `--write-cifs` exports
the L1 unit cells for it.

**`compare.py`** — matches every level against every landscape in RDF space, then
confirms the top neighbours with COMPACK (ccdc `PackingSimilarity`, 20-molecule
shell). RDF distance is cheap and ranks candidates; COMPACK is what makes "the
same structure" a defensible claim.

**`modes.py`** — cuts each landscape into distinct structural modes and places his
structures on them. **`identify.py`** — COMPACKs his structures directly against
the known polymorphs, which settles "the experimental form, or another low-energy
state?" without routing the question through our landscape.

## Structural modes, and why not paper1's clustering

`paper1_results/utils.py::clustering` drives `mean_shift_density` from a KDE over
the distance matrix — a basin *is* a density maximum there. That is right for GFN
samples, drawn in proportion to exp(−E/kT). Our brute-force priors are not that:
`collate_prior.py` ran `greedy_bottom_up_anchors2`, which deletes any sample
within a thermal radius of one already kept. **Local density in a thinned set
measures the thinning radius, not the free energy** — running the paper1 path here
would recover our own preprocessing and present it as landscape structure.

So modes are cut on shape alone: average-linkage agglomerative on the RDF distance
matrix at a distance threshold. Two consequences:

- **Mode size is not a population.** `n_members` is retained diversity. Modes are
  ranked by energy; do not turn size into a weight.
- **The threshold is swept, not assumed**, and validated two ways: the known
  polymorphs a prior was built to find must land in different modes, and any
  structure of his that COMPACK-confirmed at a full 20/20 shell must land in the
  same mode as its confirmed partner (`modes._validate` raises if not).

**The resolution limit is measured.** Distinct known polymorphs are 0.156–0.526
apart in RDF-EMD (closest pair ACRDIN08/ACRDIN05 at 0.156; the two sg14-Z′2 forms
0.178). That bounds how coarse any mode cut can be and still separate real forms.

**A landscape only supports this if it contains its own known forms.** It is the
precondition, and the two priors differ sharply:

Verdicts below are COMPACK over the top-50 RDF neighbours of each form, against
`std_opt_` so the conformer matches. RDF distance ranks; COMPACK decides.

| prior | its target forms | RDF nn | COMPACK | verdict |
|---|---|---|---|---|
| sg14_zp1 | ACRDIN04 / ACRDIN12 | 0.069 / 0.106 | **20/20**, 0.170 / 0.283 Å | both found |
| sg9_zp2 | ACRDIN05 / ACRIDIN_VIII | 0.060 / 0.087 | **20/20**, 0.200 / 0.192 Å | both found |
| sg14_zp2 | ACRDIN07 / ACRDIN06 | 0.147 / 0.139 | 10/20, 11/20 | **neither found** |

So sg9_zp2 cuts into 145 interpretable modes, while sg14_zp2 degenerates to 5255
(the coarsest cut that still separates its targets), and his sg14 structures
cannot be meaningfully placed. That is a limitation of **sg14_zp2 specifically** —
not of sg14 in general, since sg14_zp1 contains both its forms at 20/20 — and not
of his structures.

## Read nothing without the controls

`controls.py` measures both scales on the seven known polymorphs, whose answer we
already know. Both scales mislead without it, and did:

**Energy — the two polymorph files differ by CONFORMER, not relaxation.** Their
cells and poses are bit-identical (`cell_lengths`, `cell_angles`,
`aunit_centroid`, `aunit_orientation` all differ by exactly 0.0); only the atoms
move. `std_acridine_polymorphs.pt` carries the old `acridine_conformer.pt`
(aromatic C–C **1.3668 Å**) and scores **+11 to +53** kJ/mol.
`std_opt_acridine_polymorphs.pt` carries `opt_acridine_conformer.pt` (C–C
**1.4027 Å**) and scores **−55 to −60**. Every prior, and our own L1
reprojection, carry the opt conformer — so **`std_opt_` is the only like-for-like
reference**, and that ~70 kJ/mol is a conformer effect with no relaxation in it.
"opt" in the filename means *optimized conformer*, not relaxed crystal.

Read like-for-like — L1 (opt conformer, his cell, unrelaxed) against `std_opt_`
(opt conformer, experimental cell, unrelaxed) — his structures sit at median
**−7.57**, i.e. **20–50 kJ/mol above** the experimental forms. Neither file
contains a relaxed polymorph, so producing that reference means relaxing them
ourselves alongside L2.

**Distance — the thinning cutoff is not an identity criterion.**
`10**log_noise_range[1]` (0.056–0.076) is the thermal radius `collate_prior.py`
used to thin samples drawn *from* the prior. Applied to a structure ingested from
outside it is far too strict: **5 of our own 6 targeted known polymorphs fall
outside the cutoff of the prior built to find them** (0.0592–0.1430, only ACRDIN05
inside). So "0/80 of his structures within the cutoff" is not the finding it
looks like. What "present in our landscape" actually spans in RDF distance is
that 0.059–0.143 range, and COMPACK is the arbiter above it.

## Things that will bite

**Scope: only the SG/Z′ combinations we actually searched.** `config.yaml` sets
`restrict_to_searched: true`, and `searched` lists the covered combinations and
which prior covers each. Of his 80 stage-3 `Unique` structures, **25** qualify —
21 at sg14-Z′2 (P2₁/c and P2₁/n) and 4 at sg9-Z′2 (Cc). The other 55 span ten
combinations we never searched, where a non-match would be absence of evidence
rather than a result; `levels.py` names what it drops. Pass `--all-space-groups`
to keep them anyway.

The mapping lives in config rather than in code so it cannot drift from the
prior list: adding a prior means adding its `searched` row. `compare.py` asserts
that a run claiming to be restricted really is.

sg19-Z′3 exists only as raw chunks in `prior_chunks/` and was never collated, so
his Z′=3 structures have no matched landscape and are outside scope.

**Cross-Z′ comparison is legitimate here.** RDF is divided by `z_prime` and MACE
energy by `(sym_mult * z_prime)`, so both are per-molecule and compare across Z′.
That is what lets his Z′=2/3 structures be compared against our Z′=1 landscape —
and a Z′=2 structure that is really a Z′=1 structure in a doubled cell is exactly
the kind of thing this should catch. `elj`/`lj` are per-cell and are **not**
usable this way.

**His `score` is not one quantity.** The `Cc` and `Cc_` folders carry values three
orders of magnitude apart (~1e5 vs ~5–8). It is kept only as provenance; every
comparable number is recomputed on our MACE model.

**His CIFs declare `_symmetry_Int_Tables_number 1`** regardless of the actual
group, and carry no symmetry-operator loop. ccdc reads the H-M symbol instead and
gets the group right — but that is a property of the reader, not of the file, so
`ingest.py` cross-checks the ingested space group against the containing folder
and raises if they disagree.

**`P21n` is setting 2 of space group 14.** It ingests as `sg_ind` 14 with
`nonstandard_symmetry` set. Real-space comparison (RDF, COMPACK) is
setting-agnostic; `levels.py` standardizes via spglib before reprojection because
L1 rebuilds from `sg_ind` using *standard* symmetry operators.

**A COMPACK result of `n_matched == 0` is a failed comparison, not a perfect
match.** The similarity engine returns `0, 0` when it throws. The table treats
those as missing, not as RMSD 0.

**Freshly computed MACE energies are not on the scale of the stored ones.**
Recomputing the `mace` of a sample from our own prior file gives −11898.93 kJ/mol
where the file says −62.80. The gap is a per-molecule constant — 11836.127
kJ/mol, std 0.003 over 18 samples spanning Z′=1/2 and space groups 9 and 14 —
which is the signature of the atomic E₀ sum being counted in the crystal leg and
omitted from the gas-phase leg of `compute_lattice_mace`. It cancels exactly in
any *difference* of two freshly computed energies and does not cancel against a
stored value. `compare.calibrate_energy_offset` measures it every run and asserts
it is constant rather than hardcoding it. **This is a live trap for anything else
that reads a stored `mace`** — including `eval/paper1_results/analysis.py`, which
mixes stored prior energies with freshly scored samples.

**`protonation_state` must be `'protonated'`.** The default `'deprotonated'` runs
`Chem.RemoveAllHs`, turning acridine into C₁₃N. Nothing downstream announces it:
the envwise RDF just builds 36 channels instead of 91 and every comparison fails
or, had the counts agreed, would have silently compared different descriptors.
`ingest.py` asserts his composition against the reference conformer.

## The planar-mirror label flip

`ingest.py` deliberately does not use `featurize_cif_chunks.process_chunk`.

`crystal_rebuild_checks` rejects a rebuild whose re-derived `aunit_handedness` or
`aunit_orientation` differs from the one it was given, and it tests that *before*
it tests any geometry. Acridine is planar, so reflection through the molecular
plane maps the molecule onto itself and the sign of its inertial frame is
ambiguous. Measured on his 80 stage-3 structures: the round trip flips the label
for **57 of 80**, while **all 80** rebuild their unit cell to ≤ 2×10⁻⁴ Å and pass
`validate_cell_params`. The flip is a relabelling of an exact rebuild.

So acceptance in `ingest.py` is **geometric** — every structure must rebuild its
own unit cell, and the residual is recorded per structure rather than assumed.
The flip is recorded as `mirror_flip`.

It is benign for everything downstream: RDF, MACE energy, COMPACK and rigid-body
relaxation all start from the stored parameters, which rebuild correctly. It is
**not** benign for comparison in cell-parameter or latent space, where one
physical structure can carry either sign — which is why `process_target.py` and
`collate_prior.py` both force `aunit_handedness.abs()`, as `build_l1` does here.

This is a local workaround. Whether `crystal_rebuild_checks` should judge
geometry before labels for planar molecules is a question for MXtalTools, and
changing it would affect dataset construction repo-wide.
