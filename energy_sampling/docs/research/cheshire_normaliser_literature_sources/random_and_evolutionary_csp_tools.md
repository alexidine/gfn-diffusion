# Random-search, evolutionary and swarm CSP tools: trial-structure generators and how they treat normaliser (Cheshire) redundancy

Evidence labels used throughout:
- **[CODE]**: read in source code during this session. PyXtal 1.1.5 is installed locally at `C:\Users\mikem\venvs\pyxtal\Lib\site-packages\pyxtal`, with the GitHub master read via raw URLs. pymatgen 2026.10.16 is installed locally at `C:\Users\mikem\venvs\clari_cpu\Lib\site-packages\pymatgen\core\structure_matcher.py`.
- **[PAPER]**: read in the paper or official documentation during this session.
- **[SECONDARY]**: taken from a search-engine summary or third-party description, and not checked against the primary text.
- **[PRIOR]**: bibliographic or technical facts from background knowledge that were not re-fetched this session. Treat these as leads to verify.

Scope note: classical systematic CSP programs and ML generative models are excluded, because other researchers cover them. GLEE (Day/Pulido) falls into the excluded classical/quasi-random group. The "threshold" algorithm belongs to the Day group's own landscape tooling and was not researched here.

---

## Q1. How does each program choose the space group and Wyckoff positions for molecular trial structures?

### Takeaway
Every tool surveyed works the same way. It picks a space group from a user list, from all 230 groups, or from those compatible with Z and the molecular point group. It then places one molecule (Z′ = 1, sometimes more) in a general position, or in a special position whose site symmetry the molecule can carry, and generates the remaining molecules by applying the space-group operations. Selection is by group number and Wyckoff letter. No tool samples from a quotient of the parameter space.

### Cited Findings
- **PyXtal** (Fredericks, Parrish, Sayre, Zhu, *Comput. Phys. Commun.* 261, 107810 (2021), DOI 10.1016/j.cpc.2020.107810; arXiv 1911.11123) [PAPER/docs]:
  - The user gives the space group, stoichiometry and volume factor.
  - PyXtal checks the stoichiometry against every combination of Wyckoff positions. It "starts with the largest available WP, which is the general position", and uses smaller positions only when fewer atoms remain.
  - A random generating point is projected into the chosen WP. If the images collide, it is merged into a special position. — [PyXtal "How PyXtal Works"](https://pyxtal.readthedocs.io/en/latest/Algorithm.html)
  - For molecules, "the Wyckoff position compatibility is also determined by orientation and shape." — [same](https://pyxtal.readthedocs.io/en/latest/Algorithm.html)
  - The PyXtal paper notes that most earlier molecular generators handled only general positions with Z′ = 1, and names Genarris 2.0 as the exception. — [arXiv 1911.11123](https://ar5iv.labs.arxiv.org/html/1911.11123)
- **PyXtal settings** [CODE]: random generation records `self.standard_setting = wp.is_standard_setting()`. The molecular branch calls `block_crystal(..., use_hall=use_hall)`, so it can generate in non-standard Hall settings. — [pyxtal/__init__.py](https://raw.githubusercontent.com/MaterSim/PyXtal/master/pyxtal/__init__.py)
- **HTOCSP** (Zhu & Hattori, *Digital Discovery* 4, 120–134 (2025), DOI 10.1039/D4DD00264D) [PAPER]:
  - Built on PyXtal. One or several molecules go on general positions.
  - The space groups are a user list, restricted "based on the experimentally determined space group and Z′". Fractional Z′ is handled by moving to a subgroup, or by a special Wyckoff position.
  - "530 concise space group settings according to Hall" are supported. — [arXiv 2408.08843v2](https://arxiv.org/html/2408.08843v2)
- **Genarris 1.0** (Li, Curtis, ..., Marom, *J. Chem. Phys.* 148, 241701 (2018), DOI 10.1063/1.5014038 [PRIOR DOI]) [SECONDARY]: chooses "compatible space groups" whose general-position multiplicity matches the number of molecules per cell. Special positions are optional. — [arXiv 1803.02145](https://arxiv.org/pdf/1803.02145)
- **Genarris 2.0** (Tom, Rose, Bier, O'Brien, Vázquez-Mayagoitia, Marom, *Comput. Phys. Commun.* 250, 107170 (2020) [PRIOR DOI 10.1016/j.cpc.2020.107170]) [SECONDARY]: generates over all 230 space groups, on general and special Wyckoff positions. The molecule must be semi-rigid. — [arXiv 1909.10629](https://ar5iv.labs.arxiv.org/html/1909.10629); [Marom group Genarris page](https://www.noamarom.com/software/genarris/)
- **Genarris 3.0** (Yang, Tom, Wui, Moussa, Marom, *J. Chem. Theory Comput.* 21 (2025), DOI 10.1021/acs.jctc.5c01080) [PAPER]:
  - Finds "all space groups compatible with the requested number of molecules per unit cell" and the molecular point group, special positions included.
  - Z′ = 1 only.
  - Generates 4000 raw structures per compatible space group.
  - The first molecule is placed at random, and the others follow from the symmetry operations. — [PMC12613321](https://pmc.ncbi.nlm.nih.gov/articles/PMC12613321/)
- **USPEX molecular mode** (Zhu, Oganov, Glass, Stokes, *Acta Cryst.* B68, 215–226 (2012), DOI 10.1107/S0108768112017466) [PAPER, full text extracted]:
  - "we randomly pick one of 230 space groups, and set up a Bravais cell ... with random cell parameters consistent with the space group."
  - Then one molecule "is randomly placed on a general Wyckoff position, and is multiplied by space group operations. If two or more symmetry-related molecules are found close to each other, we merge them into one molecule that sits on a special Wyckoff position and has averaged coordinates of the molecular center and averaged orientational vectors (or random, when averaging gives zero)."
  - Only the initial population and 20% of each new generation come from this random symmetric generator. The remainder comes from variation operators: heredity, permutation, coordinate mutation and rotational mutation. — [arXiv 1204.4756](https://arxiv.org/abs/1204.4756)
- **CALYPSO** (Wang, Lv, Zhu, Ma, *Comput. Phys. Commun.* 183, 2063 (2012) [PRIOR DOI 10.1016/j.cpc.2012.05.008]; PSO method arXiv 1008.3601) [SECONDARY]:
  - Uses "symmetry constraints" in random generation: a random space group is chosen, then lattice and coordinates are drawn within that group's constraints.
  - Supports fixed rigid molecules.
  - Its molecular site-symmetry handling was not confirmed. — [CALYPSO docs](https://iccms-calypso.github.io/CALYPSO-Fortran/_calypso.html); [arXiv 2407.21337 summary](https://arxiv.org/abs/2407.21337)
- **AIRSS buildcell** (Pickard & Needs, *J. Phys.: Condens. Matter* 23, 053201 (2011) [PRIOR DOI 10.1088/0953-8984/23/5/053201]) [PAPER/docs, partial]:
  - The `SYMMOPS` directive sets the number of symmetry operations (crystal values 1, 2, 3, 4, 6, 8, 12, 16, 24, 48; ranges are allowed). — [autoplex BuildcellOptions](https://autoatml.github.io/autoplex/reference/autoplex.settings.BuildcellOptions.html)
  - `ADJGEN=0` "enforces the maximum possible use of symmetry related general positions", and larger values allow special positions.
  - `SYSTEM` can fix the crystal system. — [AIRSS buildcell manual](https://airss-docs.github.io/zh/buildcell-manual)
  - Because the directive is "number of operations" and not a space-group number, AIRSS samples groups by their order.
- **RandSpg** (Avery & Zurek, *Comput. Phys. Commun.* 213, 208 (2017) [PRIOR DOI 10.1016/j.cpc.2016.12.005]) [SECONDARY]: enumerates "every possible combination of Wyckoff positions for a given space group and composition", picks one at random, and places atoms at random coordinates subject to minimum distances. It is atomistic and interfaced with XtalOpt. — [GitHub xtalopt/randSpg](https://github.com/xtalopt/randSpg); [paper PDF](https://xtalopt.github.io/files/2017-RandSpg-avery-zurek.pdf)
- **CrySPY** (Yamashita et al., *Sci. Technol. Adv. Mater.: Methods* 1, 87 (2021) [PRIOR DOI 10.1080/27660400.2021.1943171]) [SECONDARY]: since v0.9.0, `struc_mode = mol` delegates random molecular-crystal generation to PyXtal. Its search options are random search, EA, Bayesian optimisation (COMBO) and LAQA. — [CrySPY mol tutorial](https://tomoki-yamashita.github.io/CrySPY_doc/tutorial/mol); [CrySPY paper record](https://mdr.nims.go.jp/datasets/f6ae9615-b564-4c7f-95aa-367a9e442507.md)
- **Rosetta lattice docking** (Park, Zhou, Baek, Baker, DiMaio, *J. Chem. Theory Comput.* 2021 [PRIOR DOI 10.1021/acs.jctc.0c01184]) [SECONDARY]:
  - The space group is assigned at random at the start of each run.
  - Monte Carlo with minimisation acts on the ASU internal coordinates, the rigid-body orientation and the lattice.
  - Rosetta symmetry machinery was extended to mirror operations. — [Baker lab PDF](https://www.bakerlab.org/wp-content/uploads/2021/02/Park_etal_JCTC2021_Small_mol_force_field_optimization.pdf); [PMC8218654](https://pmc.ncbi.nlm.nih.gov/articles/PMC8218654)
- **Zhang, Choudhury, Chernatynskiy (2024, arXiv 2407.21337)**: a "symmetry-oriented" CSP for rigid bodies (code `sgrcsp`), tested on inorganic polyanions and benchmarked against USPEX. The abstract does not say how it handles Wyckoff positions or orientations. — [arXiv 2407.21337](https://arxiv.org/abs/2407.21337)

### Inferences
- In every generator covered, the search space is a disjoint union over (space group, Wyckoff assignment). Each component is parameterised by the full free coordinates of the representative molecule: x, y, z ranging over [0, 1)³ or the WP's free subspace, plus orientation and lattice. No tool restricts the representative's position to an asymmetric unit of the space group, let alone of its Euclidean normaliser.
- PyXtal's Hall-setting support (530 settings) widens the representation space instead of narrowing it.

### Gaps
- MAGUS, GOFEE and XtalOpt's molecular mode were not checked. GOFEE is atomistic and not a molecular-crystal generator, as far as I know.
- Bayesian-optimisation CSP other than CrySPY (for example "CSPBO") was not found or checked.
- The 7th CSP blind-test descriptions of participant methods (Hunnisett et al., *Acta Cryst.* B 2024) were not read.
- The CALYPSO molecular procedure was not seen in primary text.
- The AIRSS buildcell source (how a group with N operations is drawn, and how molecular units are placed on sites) was not read.
- AFLOW's Wyckoff-based prototype generator was not checked.

---

## Q2. How are lattice parameters drawn, and is Niggli or another reduction enforced?

### Takeaway
Most tools draw cell parameters within the crystal-system constraints and a target volume, and then apply rejection bounds: angles between 30° and 150°, and length ratios. Reduction comes afterwards and is optional:
- AIRSS Niggli-reduces by default.
- USPEX transforms to "maximally orthogonal" shapes.
- PyXtal and HTOCSP have an `optimize_lattice` step that moves angles toward 90°.

None of these samples directly in a reduced-cell domain, so equivalent cells are generated with non-uniform multiplicity.

### Cited Findings
- **PyXtal lattice generation** [CODE] (`lattice.py: generate_cellpara`, local v1.1.5):
  - Triclinic cells are derived from `random_shear_matrix(width=0.2)` plus a random 3-vector scaled to the volume.
  - Monoclinic β comes from `gaussian_random_variable(minangle, maxangle)` with `minangle=π/6`.
  - Draws are rejected unless every angle lies in (30°, 150°), plus `minvec=1.2` and `max_ratio=10`.
  - No Niggli call appears in generation.
  - The docs say "For all crystals, the conventional cell choice is used to avoid ambiguity," and that angles must not be below 30° or above 150°. — [PyXtal Algorithm docs](https://pyxtal.readthedocs.io/en/latest/Algorithm.html); [lattice.py](https://raw.githubusercontent.com/MaterSim/PyXtal/master/pyxtal/lattice.py)
- **PyXtal post-hoc cell fixing** [CODE]: `pyxtal.optimize_lattice(iterations=5, force=False, standard=False)` "Optimize the lattice if the cell has bad inclination angles". It uses `Lattice.optimize_once()` and, for monoclinic groups, can try to recover the standard setting. — [pyxtal/__init__.py](https://raw.githubusercontent.com/MaterSim/PyXtal/master/pyxtal/__init__.py)
- **HTOCSP** [PAPER]:
  - Volumes come from molecular volume × multiplicity, with factors 0.7–1.3.
  - Lengths are bounded by molecular shape, and angles lie in 30°–150°.
  - After optimisation, "the angles are transformed to approach 90° whenever possible."
  - No Niggli reduction is mentioned. — [arXiv 2408.08843v2](https://arxiv.org/html/2408.08843v2)
- **AIRSS** [docs]:
  - `COMPACT`: "Force a Niggli reduction of the unit cell". It is on by default unless the cell is fixed.
  - `NOCOMPACT` disables it.
  - Sample output: "Compacting the unit cell using a Niggli reduction."
  - `TARGVOL` sets the volume. — [AIRSS buildcell manual](https://airss-docs.github.io/zh/buildcell-manual)
- **USPEX molecular** [PAPER]: "All produced unit cell shapes are checked and, if necessary, transformed to maximally orthogonal shapes [ref 23]." — [arXiv 1204.4756](https://arxiv.org/abs/1204.4756)
- **Genarris 2.0/3.0** [PAPER/SECONDARY]:
  - The target volume comes from a machine-learned model (PyMoVE in 2.0). The 3.0 paper says the cell volume is drawn "within a normal distribution around a target value". With Rigid Press it is scaled by 1.5.
  - The Genarris 3.0 paper describes no Niggli or other reduction. — [PMC12613321](https://pmc.ncbi.nlm.nih.gov/articles/PMC12613321/); [Genarris 2.0](https://ar5iv.labs.arxiv.org/html/1909.10629)
- **Genarris 3.0 Rigid Press** [PAPER]:
  - The state vector holds the independent lattice vectors (set by crystal system), the ASU centre of geometry, and Euler angles.
  - The objective is volume plus a regularised hard-sphere penalty, minimised with SciPy BFGS.
  - "Space group symmetries are preserved by reconstructing the full crystal coordinates from the optimized state vector." — [PMC12613321](https://pmc.ncbi.nlm.nih.gov/articles/PMC12613321/)

### Inferences
- Because the triclinic draw is a random shear matrix with an angle window, many reduced cells are reachable through several non-reduced representatives. PyXtal therefore over-weights some lattices relative to a Niggli-domain measure. This is inferred from the code and not measured.
- AIRSS's Niggli step is a canonicalisation applied after generation. It does not define a sampling domain.

### Gaps
- USPEX's reference [23] (the "maximally orthogonal" transform) was not pinned down, though it is likely Oganov & Glass 2006 or Lyakhov et al. 2010 [PRIOR].
- CALYPSO's and Rosetta's lattice priors were not checked.
- The Genarris source code was not read.

---

## Q3. How are molecular orientations sampled? Is the sampling uniform on SO(3), and how are the molecular point group and site symmetry handled?

### Takeaway
PyXtal handles site-symmetry compatibility carefully, but only as a discrete set of constrained orientations for special positions. Its free (3-DOF) rotation sampling is not Haar-uniform: a normalised uniform-cube axis is combined with a uniform angle. PyXtal removes duplicates among the discrete special-position orientations using the molecular point group. No program found reduces the continuous orientation space to a fundamental zone of SO(3)/(molecular point group). USPEX and AIRSS simply say "random orientation" or "full rotation".

### Cited Findings
- **PyXtal orientation compatibility** [docs]:
  - The algorithm finds the molecular point group with pymatgen's `PointGroupAnalyzer` and maps molecular symmetry axes onto the WP's site-symmetry axes.
  - It then "Remove[s] redundant orientations" and keeps only orientations under which the molecule is symmetric under the site symmetry.
  - Each orientation is stored with 0, 1 (rotation about an axis) or 3 DOF, and "an allowed orientation is randomly chosen from the list".
  - In the general position, any orientation is allowed. — [PyXtal Algorithm docs](https://pyxtal.readthedocs.io/en/latest/Algorithm.html)
- **PyXtal redundancy removal** [CODE] (`molecule.py`, `get_orientations_in_wp`):
  - The code comments are "Remove redundancy for the first constraints" and "Remove redundancy from orientations".
  - A candidate orientation is dropped when the relative operation, conjugated by the orientation, is a valid operation of the molecular point group (`self.pga.is_valid_op(old_op)`). — [molecule.py](https://raw.githubusercontent.com/MaterSim/PyXtal/master/pyxtal/molecule.py)
- **PyXtal free-rotation sampler** [CODE] (`Orientation` in `molecule.py`, `aa2matrix` in `operations.py`):
  - In the 3-DOF case the axis is `random_state.random(3) - 0.5`, normalised, and the angle is uniform on [0, 2π).
  - `aa2matrix(random=True)` uses `np.random.sample(3)` for the axis (first octant), normalised, and `np.random.random()*2π` for the angle.
  - `change_orientation` may also apply a random 90°/180°/270° flip about x, y or z with probability 0.5.
  - In the 1-DOF case the angle is uniform about the constrained axis.
  - **Neither draw is Haar measure on SO(3)**: uniform rotation angle with a uniform axis over-weights small rotations, and the cube-normalised axis is anisotropic.
  - No `scipy.spatial.transform.Rotation.random` is used. — [operations.py](https://raw.githubusercontent.com/MaterSim/PyXtal/master/pyxtal/operations.py); [molecule.py](https://raw.githubusercontent.com/MaterSim/PyXtal/master/pyxtal/molecule.py)
- **PyXtal chirality** [docs]: "By default, PyXtal does not generate crystals with mirror copies of chiral molecules", though inversion can be allowed. — [PyXtal Algorithm docs](https://pyxtal.readthedocs.io/en/latest/Algorithm.html)
- **Genarris 3.0** [PAPER]: molecules on special positions are aligned with the site symmetry. Orientation is parameterised by Euler angles in Rigid Press. The paper does not state the sampling measure. — [PMC12613321](https://pmc.ncbi.nlm.nih.gov/articles/PMC12613321/); [FastCSP arXiv 2508.02641](https://arxiv.org/pdf/2508.02641) [SECONDARY]
- **USPEX molecular** [PAPER]:
  - The initial structures use "random positions and orientations of molecular units". Molecules merged onto special positions take averaged orientation vectors.
  - Rotational mutation rotates selected molecules "by random angles".
  - "For rigid molecules, there are only 3 varibles [sic] to define the orientation." — [arXiv 1204.4756](https://arxiv.org/abs/1204.4756)
- **AIRSS** [docs]: `ANGAMP` is the "Amplitude of rotation for units (or molecules)" in degrees. "A negative value implies full rotation" (default -1). `FLIP` will "Randomly mirror the structural units". The distribution is not documented. — [AIRSS buildcell manual](https://airss-docs.github.io/zh/buildcell-manual)

### Inferences
- PyXtal's "valid orientations" mechanism is the closest existing analogue to "site-symmetry-compatible orientation". Removing redundancy via the molecular point group is a discrete version of a quotient SO(3)/G_mol. The continuous 3-DOF component is still sampled over all of SO(3), so it covers each orientation |G_mol| times (proper part), and does so non-uniformly.
- No tool uses the terms "fundamental zone" or "orientation fundamental zone", which come from texture analysis. CSP generators speak of "valid orientations", "allowed orientations" or "orientations compatible with site symmetry".

### Gaps
- The orientation samplers of AIRSS, CALYPSO, Genarris and Rosetta were not code-verified.
- I did not check whether any tool uses quaternions with Haar measure (for example Shoemake's method).

---

## Q4. Do any of these tools remove origin/normaliser (Cheshire-group) equivalences a priori, or only after generation? What terminology do they use?

### Takeaway
None of the random, evolutionary or swarm molecular CSP tools surveyed removes Euclidean-normaliser redundancy a priori. "Cheshire" appears in none of the sources read. The only explicit normaliser machinery found is PyXtal's post-hoc `get_alternatives()`. It enumerates the "alternative settings" (equivalent Wyckoff sets) of a generated structure from a tabulated database. Zhu's group says these settings are computed from the Euclidean and affine normalisers, and uses them for symmetry-relation and phase-transition analysis, not to restrict search. Every other tool relies on post-hoc duplicate detection.

### Cited Findings
- **PyXtal `Group.get_alternatives()`** [CODE]:
  - "Get the alternative settings as a dictionary". It loads `database/wyckoff_sets.json` and returns `wyc_sets[str(self.number)]`, which has keys including `"No."` and `"Transformed WP"`.
  - `Group.get_orders()` uses the `'Transformed WP'` permutations.
  - The `wyckoff_sets.json` file was **absent** from the local PyXtal 1.1.5 install, though the code references it. It is presumably in the GitHub master or a packaging omission. Unverified. — [symmetry.py](https://raw.githubusercontent.com/MaterSim/PyXtal/master/pyxtal/symmetry.py)
- **PyXtal `pyxtal.get_alternatives(include_self, same_letters, ref_lat, d_tol, f_tol)`** [CODE]:
  - "Get alternative structure representations". It loops over each non-identity Wyckoff-set transformation ("skip the first setting since it is identity").
  - `_get_alternative` applies the inverse affine operation (`get_inverse(SymmOp.from_xyz_str(...))`) to the lattice and the sites.
  - It is used by `supergroup.py` (`strucs = struc.get_alternatives()`), which means it serves group–subgroup path searching, not random generation. — [pyxtal/__init__.py](https://raw.githubusercontent.com/MaterSim/PyXtal/master/pyxtal/__init__.py); local `supergroup.py:353`
- **Zhu group terminology** [SECONDARY, search snippet of the paper]: "All different Wyckoff settings can be computed by the Euclidean and the affine normalizers of a given space group." This is illustrated with FCC Cu on 4a (0, 0, 0) versus 4b (½, ½, ½). The paper adds: "This algorithm has been implemented in the open source code PyXtal." — Zhu, Kang, Parrish, *MRS Commun.* 12, 686 (2022), DOI 10.1557/s43579-022-00268-4, [arXiv 2208.10655](https://arxiv.org/pdf/2208.10655). The abstract page itself does not show this text, so it should be checked in the full PDF.
- **PyXtal "euclidean" naming caveat** [CODE]: `Wyckoff_position.set_euclidean()`, `get_euclidean_generator` and `get_euclidean_ops` concern Cartesian conversion of operations for hexagonal groups ("For the hexagonal groups, need to consider the euclidean conversion"). They have nothing to do with normalisers. A grep for "normalizer", "Cheshire" or "affine norm" in local PyXtal 1.1.5 finds no hits. — [symmetry.py](https://raw.githubusercontent.com/MaterSim/PyXtal/master/pyxtal/symmetry.py)
- **PyXtal duplicate check** [CODE]: `pyxtal.is_duplicate(ref_strucs)` "check if the structure is exactly the same". It works through lattice/site matching with tolerance thresholds (1e-3); `check_mapping` matches Wyckoff indices. — [pyxtal/__init__.py](https://raw.githubusercontent.com/MaterSim/PyXtal/master/pyxtal/__init__.py)
- **HTOCSP** [PAPER]: no normaliser, origin-shift or Wyckoff-set handling is described. StructureMatcher is used to compare against a target. — [arXiv 2408.08843v2](https://arxiv.org/html/2408.08843v2)
- **Genarris 3.0** [PAPER]: normaliser, origin-shift, Cheshire, enantiomorph and chirality handling are not discussed. Duplicates are removed within each space group with StructureMatcher after generation. — [PMC12613321](https://pmc.ncbi.nlm.nih.gov/articles/PMC12613321/)
- **USPEX molecular** [PAPER]: "At the end of each generation, all structures in the generation are compared using their fingerprints and all non-identical structures are ranked." Intramolecular distances are excluded from the fingerprint, which is post-hoc only. — [arXiv 1204.4756](https://arxiv.org/abs/1204.4756)
- **AIRSS, RandSpg, CALYPSO, CrySPY, Rosetta**: no a-priori normaliser reduction was described in any source read. [PAPER/SECONDARY as cited above]

### Inferences
- Origin and normaliser redundancy is real in all these generators. In PyXtal, for example, the general-position representative is drawn over the whole cell, so each crystal is reachable from |N_E(G) : G| × (number of molecules in the cell) equivalent representatives, before counting lattice-basis redundancy. They all absorb this through post-hoc deduplication: StructureMatcher, fingerprints, or clustering.
- "Alternative settings", "Wyckoff sets", "equivalent Wyckoff sets" and "Euclidean/affine normalizer" are the terms used in the PyXtal/Zhu lineage, following Bilbao's WYCKSETS. "Cheshire group" (Hirshfeld 1968) appears not to be used in this CSP-software literature.
- The requester's construction would be new relative to this tool set: reduced cell × ASU of the normaliser (chirality-preserving for chiral molecules) × orientation fundamental zone. The nearest ingredients are PyXtal's discrete Wyckoff-set enumeration and its point-group orientation redundancy removal, and neither is applied to the sampling domain.

### Gaps
- The `wyckoff_sets.json` content and provenance were not inspected, because the file was missing locally. Whether it encodes Euclidean or affine normaliser cosets, or Bilbao WYCKSETS exactly, is unconfirmed.
- The full text of Zhu, Kang, Parrish (2022) was not read.
- No negative-search evidence was gathered for MAGUS, XtalOpt-mol, GOFEE or AFLOW.

---

## Q5. Which duplicate-detection and similarity metrics are used after generation, and how do they handle origin shifts, settings and enantiomers?

### Takeaway
Post-hoc comparison absorbs the redundancy. StructureMatcher, XtalComp and COMPACK reach invariance by searching over lattice bases and translations, plus reflections in XtalComp's case. USPEX/Oganov fingerprints and Widdowson–Kurlin AMD/PDD are isometry invariants by construction: no search, no fundamental domain. The Kurlin programme explicitly presents complete continuous isometry invariants as the principled alternative to reduced-cell or fundamental-domain canonicalisation.

### Cited Findings
- **pymatgen StructureMatcher** [CODE, pymatgen 2026.10.16]:
  - The class docstring sets out the algorithm: optionally reduce to primitive; "Reduce to s1 and s2 to Niggli Cells"; find all lattice vectors of s2 within ltol; "For s1, translate an atom in the smallest set to the origin"; for each valid lattice, translate each atom of s2's smallest set to the origin and compare fractional sites; return the RMS on an averaged lattice.
  - The code loops over translations (`# loop over all translations`), and `_cart_dists` fits the mean translation.
  - Lattice candidates come from `Lattice.find_all_mappings`. It accepts any integer basis change with non-zero determinant, and `_get_lattices` keeps those with |det| matching the supercell size, so det = −1 (improper) mappings are allowed. — local `pymatgen/core/structure_matcher.py`, `lattice.py`
- **StructureMatcher usage**:
  - Genarris 3.0 uses "0.5 fractional length tolerance, 0.5 site tolerance, and 10° angle tolerance", per space group. — [PMC12613321](https://pmc.ncbi.nlm.nih.gov/articles/PMC12613321/)
  - HTOCSP ignores H, maps molecules one-to-one, and uses ltol 0.3, stol < 0.3 Å and angle < 5°. It notes COMPACK gave "nearly identical results". — [arXiv 2408.08843v2](https://arxiv.org/html/2408.08843v2)
- **XtalComp** (Lonie & Zurek, *Comput. Phys. Commun.* 183, 690 (2012) [PRIOR DOI 10.1016/j.cpc.2011.11.007]) [docs]:
  - "It begins by performing a Niggli reduction on the inputs, standardizing the translation vectors and orientations."
  - A transform search then identifies "candidate sets of rotations, reflections, and translations", "solving the problems of enantiomorphs and rotationally degenerate lattices".
  - A cell-expansion technique removes periodic-boundary issues. — [XtalComp GitHub](https://github.com/allisonvacanti/XtalComp)
  - XtalComp is the duplicate checker used by XtalOpt [PRIOR].
- **USPEX / Oganov fingerprint** [PRIOR, not re-fetched]: Oganov & Valle, *J. Chem. Phys.* 130, 104504 (2009), DOI 10.1063/1.3079326. It uses pair-distance-distribution fingerprints with a cosine distance, so it is invariant to origin, basis and handedness by construction. Lyakhov, Oganov, Stokes, Zhu, *Comput. Phys. Commun.* 184, 1172 (2013), DOI 10.1016/j.cpc.2012.12.009, extends this to niching and degeneracy removal. The molecular adaptation drops intramolecular distances [PAPER: arXiv 1204.4756].
- **CALYPSO** [SECONDARY]: uses a "bond characterization matrix" to eliminate similar structures. — [CALYPSO docs](https://iccms-calypso.github.io/CALYPSO-Fortran/_calypso.html)
- **Genarris 2.0/3.0 and GAtor clustering** [SECONDARY/PAPER]:
  - Genarris 2.0 uses affinity propagation on a relative coordinate descriptor (RCD) or radial symmetry function (RSF). — [Genarris 2.0](https://ar5iv.labs.arxiv.org/html/1909.10629)
  - GAtor (Curtis et al., *J. Chem. Theory Comput.* 14, 2246 (2018) [PRIOR DOI 10.1021/acs.jctc.7b01152]) uses ML clustering of the population with cluster-based fitness (niching). — [arXiv 1802.08602](https://arxiv.org/pdf/1802.08602)
- **AMD/PDD** (Widdowson & Kurlin, "Resolving the data ambiguity for periodic crystals", NeurIPS 2022; AMD: Widdowson, Mosca, Pulido, Cooper, Kurlin, *MATCH Commun. Math. Comput. Chem.* 87, 529 (2022) [PRIOR]):
  - Pointwise Distance Distributions are isometry invariants that are complete for periodic sets in general position. They come with an exactly computable metric (Earth Mover's Distance) in near-linear time, "without false negatives".
  - Lipschitz continuity is motivated by experimental noise. — [NeurIPS 2022 paper](https://proceedings.nips.cc/paper_files/paper/2022/file/9c256fa1965318b7fcb9ed104c265540-Paper-Conference.pdf); [arXiv 2108.04798](https://arxiv.org/abs/2108.04798v2); [near-duplicates paper](https://www.kurlin.org/projects/periodic-geometry/near-duplicates-materials-databases.pdf); [CSD maps / Crystal Isometry Principle](https://www.kurlin.org/projects/periodic-geometry/continuous-maps-CSD-extended.pdf); [average-minimum-distance package](https://pypi.org/project/average-minimum-distance/)
- **COMPACK** (Chisholm & Motherwell, *J. Appl. Cryst.* 38, 228 (2005) [PRIOR DOI 10.1107/S0021889804027074]) [PRIOR]: compares clusters of molecules (default 15) by interatomic geometry. It is independent of cell setting and origin by construction.

### Inferences
- StructureMatcher's acceptance of improper basis changes with |det| = 1, combined with comparison in fractional coordinates on an averaged lattice, implies that by default it identifies a structure with its mirror image. This is inferred from the code, not tested. For chiral-molecule CSP, a chirality-preserving domain such as the requester's would therefore be stricter than common post-hoc deduplication.
- Two philosophies are explicit in this literature: transform-search matchers (StructureMatcher, XtalComp) and complete isometry invariants (Kurlin's AMD/PDD, implicitly the Oganov fingerprints). A normaliser fundamental domain is a third option: canonicalise before sampling. None of the generator tools in this survey adopts it.

### Gaps
- Whether StructureMatcher really matches enantiomorphs was not tested empirically, and the pymatgen issue tracker was not checked.
- PXRD-based similarity (PyXtal XRD, HTOCSP's similarity score) and SOAP/REMatch were not examined in detail.
- Whether Kurlin's papers explicitly mention normalisers or Cheshire groups was not checked in full text.
- The XtalComp paper's default treatment of enantiomorphs was not confirmed: the README implies it is application-dependent.
