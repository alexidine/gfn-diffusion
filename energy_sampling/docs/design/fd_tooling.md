# Fundamental-domain tooling for MXtalTools, and how GFN uses it

- **Status:** PROPOSAL, 2026-10-15. Not adopted. It has no authority over code, config or tests. Owner decisions O1–O6 in §6 are open. The owner answered D1–D7 of the 2026-10-16 inspection on 2026-10-16; §6 records the answers, their scope and what was built under them (D1 answers O7).
- **Proposed path:** `energy_sampling/docs/design/fd_tooling.md`.
- **Scope:**
  - MXtalTools tooling for molecular crystals: folding, parameterising, building, round trips, and translation to standard formats. Z′ = 1 comes first and Z′ > 1 last.
  - The GFN adapter and prior intake that use it.
  - No change to the live trainer path is proposed.
- **Relation to `docs/design/fundamental_domain.md`:**
  - This note answers that note's §2 gaps: no domain table, a per-molecule domain, no bijectivity test, and plumbing.
  - It contradicts two of that note's claims (§2.10, item 10).
  - Its §3 (delete flat directions, do not pin them) still stands, and §3.4 here uses it for free axes.
- **Sources:**
  - Three design reports dated 2026-10-15, cited here as [math], [api] and [verif].
  - Those reports build on four read-only reports compiled the same day: R1 (chart per space group), R2 (normaliser images), R3 (cell reduction) and R4 (documents).
  - Everything ran on CPU with synthetic molecules, read-only on both repositories, unless a line says otherwise.
- **Evidence location:**
  - The scripts and their outputs were copied out of the session scratchpad to `energy_sampling/artifacts/fd_design_oct15/fdtool/` (git-ignored, local only; written `$FD` below). The four background reports and the problem statement sit beside it (`fd_reads.md`, `fd_framed.md`, `fd_tooling_reports.md`):
    - [math]: `t1`–`t10`, plus `fdlib.py`, `poly.py` and `cct.py`;
    - [api]: `api_gamma_plus.py` and `api_metric_check.py`;
    - [verif]: `vs/vs_*.py`.
  - That directory is in neither repository's history.
  - The gates in §4 re-implement every load-bearing measurement. Until they exist, the numbers below are observations.
- **Line numbers:** taken from the MXtalTools working tree on 2026-10-15, which has uncommitted changes. Paths are relative to the MXtalTools root unless marked GFN; GFN paths are relative to `energy_sampling/`.
- **Labels:**
  - [measured]: computed by the scripts above.
  - [derived]: an argument [math] gives in outline. It is consistent with the measurements but has not been independently reviewed.
  - [Inf.]: inferred; neither fully derived nor measured.
  - [code]: read at the cited line.

## 1. Topline

On the trainer's handedness +1 sheet, a Z′ = 1 pose in any of the 230 space groups has the exact fundamental domain ASU(Γ⁺) × SO(3), where Γ⁺ is the proper part of the Euclidean normaliser. This domain has no walls and needs neither a chirality token nor a per-molecule domain; only sg 6–9 need a correction, a 2-fold cut in SO(3) [measured: orbit counts in all 230 groups; eLJ invariance in 7 groups here and 11 in R2]. Since 2026-10-16 the centroid part is proved, not only measured: `mxtaltools/crystal_fd/certify.py` shows in exact rational arithmetic, for all 230 groups, that ASU(Γ⁺) tiles the non-free coordinates under Γ⁺ (under Γ⁺/K in sg 6–9, whose pose domain still relies on the K cut that gate 1 checks by sampling) [code: certificate in §2.2]. Still unproved: cctbx's cut-tree choices on boundaries, the fold's float tolerance band, the cell part (built and tested for sg 1–15 only, monoclinic since 2026-10-16), Z′ > 1, and special metrics (flagged, not resolved). The proposed build is an additive MXtalTools subpackage that is cctbx-free at runtime and driven by one generated per-group table. It is accepted through ten gates and built in six stages, from sg 1 and 2 (what the MIPCAS flow test needs) to Z′ > 1, and a GFN adapter maps the flow's domain into today's trainer latent with an exact log-det, so the trainer, reward and builder do not change.

## 2. Mathematics

### 2.1 Objects

**The crystal.** Take a Z′ = 1 crystal of a rigid molecule M in space group G, in the standard setting (`SYM_OPS[sg]`, `mxtaltools/constants/space_group_info.py:252`). It is a point (c, t, R):
- c is the cell;
- t is the fractional centroid of the asymmetric-unit molecule;
- R is a proper rotation of the stored coordinates.

The trainer builds handedness +1 only (GFN `energies/molecular_crystal.py:1515`, `:747`) [code, via R1].

**The normaliser tables.** N_E(G) is the Euclidean normaliser of G.
- Its coset representatives over G are `NORMALIZER_OPS` (`space_group_info.py:15510`).
- Its continuous part, the free axes, is `CONTINUOUS_DIMS` (`:15279`).

**The groups used throughout.**
- **Γ⁺** = {γ ∈ N_E(G) : det W_γ = +1}, counted modulo ℤ³ and modulo shifts along free axes. Because G ⊂ N_E(G), Γ⁺ is exactly the set {g∘n : g ∈ G, n ∈ N_E(G), det(W_g W_n) = +1} that the hypothesis names.
- **P⁺** = {W_γ} is the rotation group of Γ⁺.
- **L⁺** ⊇ ℤ³ is the translation lattice of Γ⁺.
- **G⁺** is the proper part of G.
- **K** is the subgroup of Γ⁺ that acts trivially on the centroid once free axes are projected out. Note that [api] uses "K" for P⁺; this note does not.

### 2.2 Result

**The hypothesis FD = ASU(Γ⁺) × SO(3) holds:**
- as stated in all 162 groups without free axes;
- in 64 of the 68 free-axis groups, once Γ⁺ is projected onto the non-free coordinates;
- in sg 6–9 (Pm, Pc, Cm, Cc) only with a correction.

**The sg 6–9 correction.** There the free axes are x and z. The 2-fold about b fixes every point of the remaining y line, so K = C2(b) and it has to be cut in SO(3): FD = ASU(Γ⁺/K) × SO(3)/K. K is trivial in the other 226 groups [measured: `$FD/t8_sweep.log`].

**Why it holds [derived].**
1. **Which composites are equivalences.** Suppose two parameter points describe the same crystal up to a proper rigid motion φ.
   - Then φ preserves the lattice and G, so φ ∈ N_E(G).
   - The new asymmetric-unit molecule is g·φ·x for some g ∈ G.
   - That molecule is M at a proper rotation exactly when det(W_g W_φ) = +1.
2. **Why the domain is a product.**
   - An element with W ≠ I moves every R, because left multiplication by Q ≠ I has no fixed point on SO(3).
   - An element with W = I is a nonzero translation, so it moves every t.
   - So Γ⁺ acts freely on ℝ³ × SO(3), and a generic t has a trivial stabiliser.
   - Fixing t to an asymmetric unit of Γ⁺ therefore leaves R free.
   - This fails only where projecting out free axes leaves elements that fix t: that set is K.
3. **Consequences.**
   - The pose quotient is a closed smooth manifold: no mirror walls and no cone points.
   - Every face of the pose domain is a seam. The only real walls come from the cell domain (§2.6) and from the modelling box.

**Measured over all 230 groups.**
- **Closure.** The closure of SYM_OPS ∪ NORMALIZER_OPS equals a brute-force N_E in 230 of 230 groups. The brute force takes all W ∈ {−1,0,1}^{3×3} that preserve a generic metric, with w on a 1/24 grid [math].
  - Γ⁺ built from the MXtalTools tables equals Γ⁺ built from cctbx alone in 230 of 230 [verif].
  - cctbx's expansion of Γ⁺ adds no element in 162 of 162 groups, so every Γ⁺ is a Sohncke group [api].
- **Group types and sizes.**
  - Γ⁺ falls into 16 Sohncke types, the most common being P422 (38 groups), P222 (36), P622 (20) and I432 (19) [math]. These occur in 29 distinct settings [api].
  - |Γ⁺| reaches 384 in sg 196, 216 and 219 [measured: `$FD/t8_sweep.log`].
- **Volume.** vol(ASU(Γ⁺)) × |Γ⁺| = 1.000000 in all 162 groups without free axes [math].
- **Orbit counts.**
  - The centroid alone gave exactly one Γ⁺ image per point in all 162 groups: 3,000 points per group [math], 300 per group [api] and 10⁴ per group [verif].
  - With orientations, both cuts below gave exactly one orbit point in 1,000 of 1,000 samples in all 230 groups. Selected groups were also run with 4,000 samples [math].
  - At exact rational face, edge and special points, using cctbx's tie-break cuts, 3,622 points in 12 groups each had exactly one image [verif].
- **Proof of the centroid part [code: `mxtaltools/crystal_fd/certify.py`, gate `tests/fd/test_gate0_tiling_certificate.py`; run 2026-10-16].** From the table alone, in exact rationals, for each group with d non-free axes, P = ASU(Γ⁺) on those axes and n = |Γ⁺/K| representatives, the certificate checks:
  - the representatives are a group modulo ℤ^d: identity first, |det W| = 1, no two equal and every product one of them (all n² ≤ 384² compositions);
  - vol(P) × n = 1;
  - facet coverage: on every facet of P, the zero-volume contacts of dimension d − 1 with images k·P + s have measures summing exactly to the facet's and pairwise overlaps of measure zero (areas for d = 3, lengths for d = 2, one touching image per endpoint for d = 1; 1,178 facets in all);
  - no examined image overlaps P in positive volume;
  - the dropped free-axis cuts are height-1 walls.
  - The module docstring derives from these that the images of P cover the non-free coordinates with disjoint interiors, without assuming the enumerated lattice-shift range is complete: a missing neighbour fails the coverage check. All 230 groups pass in about 2 s. Mutation controls pass the volume and overlap checks and are rejected by the new ones: a duplicated element, a translation moved 1/24 off the group (it passed those checks in 265 of 300 cases over sg 2, 14, 92 and 198), a wrong domain examined at one shift only, and a wall of the wrong height.
  - This turns the centroid orbit counts above into a proof. The sampled counts remain the evidence for the orientation rule and, in sg 6–9, the K cut. The certificate takes the table's Γ⁺ as given; gate 0 re-derives it by brute force (W ∈ {−1,0,1}^{3×3}, w on a 1/24 grid).
- **NORMALIZER_OPS.**
  - Every entry normalises G and is an isometry of a generic metric of its family, so Γ⁺ never moves the cell [math, api].
  - The empty entries for sg 229 and 230 are correct: there, N_E = G.
- **Sohncke groups.** 43 of the 65 Sohncke groups have improper N_E elements. Their images are enantiomorph crystals, which the +1 sheet correctly drops. The other 22 Sohncke groups are the 11 enantiomorphic pairs [math].

*Table 1. Γ⁺ and its centroid domain for the groups in use, plus one example per family. Counts are taken mod ℤ³ in the conventional cell, and mod free-axis shifts where free axes exist.*
- **Shape codes:**
  - C p: a circle of period p, from a translation gluing;
  - I w: an interval of width w whose ends are glued with a twist;
  - pin: a free axis.
  - Correction (2026-10-16): sg 9's y row was given as I½. The built table glues both y facets by a translation (y ↦ y + ½; the 2-fold of K acts trivially on y), so it is C½ [code: `mxtaltools/crystal_fd/data/fd_tables_v1.json`, facets of sg 9].
- **Trainer box columns.** These describe `RAW_ASYM_UNITS[sg]` (`constants/asymmetric_units.py`) at handedness +1. Coverage is the fraction of a chiral molecule's crystals the box reaches, from 20,000 random crystals per group. Images is the number of descriptions per reached crystal.
- **Reference:** a fundamental domain would show coverage 1.00 with 1 image [math, api, verif].

| sg | free axes | \|G\| (operations per cell) | \|Γ⁺\| (proper operations per cell) | P⁺ | centroid domain (fractional) | trainer box coverage (fraction) | trainer box images (count per crystal) |
|---|---|---|---|---|---|---|---|
| 2 P-1 | – | 2 | 8 | C1 | x C½, y C½, z C½ | 1.00 | 4 |
| 4 P2₁ | y | 2 | 8 | C2 | x C½, y pin, z I¼ | 1.00 | 4 |
| 9 Cc | x, z | 4 | 4, with K = C2(b) | C2 | x pin, y C½, z pin | 0.50 | 2 |
| 14 P2₁/c | – | 4 | 16 | C2 | x C½, y C½, z I¼ | 0.50 | 8 |
| 15 C2/c | – | 8 | 16 | C2 | as sg 14 | 1.00 | 2 |
| 19 P2₁2₁2₁ | – | 4 | 32 | D2 | x I¼, y I¼, z C½ | 1.00 | 8 |
| 61 Pbca | – | 8 | 32 | D2 | as sg 19 | 1.00 | 4 |
| 88 I4₁/a | – | 16 | 32 | D4 | oblique polytope | 0.50 | 4 |
| 92 P4₁2₁2 | – | 8 | 32 | D4 | oblique polytope | 1.00 | 4 |
| 148 R-3 | – | 18 | 36 | D3 | oblique polytope | no box | – |
| 198 P2₁3 | – | 12 | 48 | O | oblique polytope | no box | – |
| 205 Pa-3 | – | 24 | 24 | T | oblique polytope | no box | – |

**Two exact ways to cut (both measured by orbit count in all 230 groups [math]).**
- **Translation cut (A):** ASU(Γ⁺) × SO(3), with the K correction in sg 6–9. The orientation is the whole of SO(3).
- **Orientation cut (B):** (ℝ³/L⁺) × FZ(P⁺).
  - Every centroid row is a circle in an L⁺ basis. That basis is sheared in some tetragonal groups, for example (½,½,0) in sg 91, 92, 95 and 96.
  - FZ(P⁺) is a fundamental zone of P⁺ acting by left multiplication on SO(3):
    - **C_n:** Euler α̃ ∈ [0, 2π/n) about the principal axis, a circle.
    - **D_n:** the same, plus β̃ ≤ π/2. The β̃ = π/2 face is glued to itself with a twist.
    - **T and O:** the Dirichlet zone ω(R) ≤ ω(QR) for all Q ∈ P⁺, a convex polytope in Rodrigues coordinates.
  - **Coverage of the test.**
    - The sweep's classifier labels D2 as "T" (`$FD/t4_so3.py:32`), so the 59 D2 groups were counted with the Dirichlet zone. The Euler box for D2 is **untested**.
    - Euler boxes were tested for the 15 C1 and C2 groups and the 120 D3, D4 and D6 groups.
    - The Haar density of the Rodrigues zone for T and O is unchecked.
  - In sg 1 and 2 the two cuts coincide. §6 O1 is the owner's choice between them.

### 2.3 Where the polytopes come from

1. **cctbx.** `sgtbx.space_group_info(group=Γ⁺).direct_space_asu()` (`cctbx/sgtbx/__init__.py:161-168`; cctbx 2025.11 is importable in `csd_mxt_gfn`) takes Γ⁺ written in our basis. It returns the ITA reference asymmetric unit in our coordinates:
   - 5–7 cut planes with integer normals and rational offsets;
   - inclusion flags for tie-breaks on faces.

   Example, sg 14: 0 ≤ x ≤ ½, 0 ≤ y ≤ ½, 0 ≤ z ≤ ¼ [math].
2. **Dirichlet domain in the invariant metric** M = mean WᵀW. This is about 40 lines (`$FD/poly.py::dirichlet`) and needs no cctbx. It passed in all 230 groups. It has more facets than cctbx's, and they are oblique: 8 for sg 14, 14 for sg 19, 15–24 for cubic and hexagonal groups [math].
3. **Autoregressive bounds.** Fourier–Motzkin elimination of the inequality list, in a fixed order, gives x ∈ [l₁,u₁], y ∈ [l₂(x),u₂(x)], z ∈ [l₃(x,y),u₃(x,y)].
   - 100% of 200,000 AR samples fell inside P.
   - E[∏(uᵢ−lᵢ)] matched vol(P) within 2 standard errors in sg 2, 14, 19, 88, 92, 148, 152, 198 and 205 [math, `$FD/t10_fm.py`].

**Shape by family [verif].**
- The centroid domain is an axis box in all 74 triclinic, monoclinic and orthorhombic groups: the 44 without free axes, plus 30 free-axis groups after projection.
- The other 156 groups (tetragonal, trigonal, hexagonal and cubic) have polytopes with 4–8 oblique cuts.

### 2.4 Face gluings

*Table 2. How each face of the cctbx ASU(Γ⁺) is glued.*
- **Method:** 60 points were stepped 1e-6 outside each facet. The table gives the element that carries them back inside [measured, `$FD/t3_faces.py`].
- **Kinds:**
  - translation (W = I): the coordinate is a circle;
  - self, twisted: the face is glued to itself through a 2-fold lying in it, so it is an interval end.

| sg | facet (fractional) | gluing element: action on t (fractional); on R | kind |
|---|---|---|---|
| 2 | x = 0 ↔ ½, y = 0 ↔ ½, z = 0 ↔ ½ | t + ½e_d; R unchanged | translation, period ½ |
| 14 | x = 0 ↔ ½, y = 0 ↔ ½ | t + ½e_d; R unchanged | translation, period ½ |
| 14 | z = 0 | (½−x, y, −z); R ↦ C2_b R (2-fold on the line x = ¼, z = 0) | self, twisted |
| 14 | z = ¼ | (½−x, y, ½−z); R ↦ C2_b R (2-fold on the line x = ¼, z = ¼) | self, twisted |

**Other groups [measured].**
- **sg 19, 61:** z is a translation circle; the four x and y faces are self-glued by 2-folds.
- **sg 88, 92, 96, 142:** one translation pair along [1,±1,0], which becomes a circle only after a shear. The rest are 2-fold self-gluings. One facet is glued piecewise by two different 2-folds.
- **sg 148:** 120° twist pairs and 2-fold self-faces. One face is glued piecewise across three 2-folds.
- **sg 152:** all five faces are 2-fold self-gluings.
- **sg 198, 205:** a 3-fold twist pair and 2-fold self-faces. Two faces are piecewise.

No pose face in any group is a wall [derived, §2.2].

**What sg 2 and sg 14 look like under each cut.**
- **sg 2:** both cuts give T³(½) × SO(3).
- **sg 14:**
  - cut A gives S¹(½)_x × S¹(½)_y × [0,¼]_z × SO(3);
  - cut B gives S¹(½)³ × SO(3)/C2_b, which has no twisted seam. In Euler angles whose first axis is b, α̃ ∈ [0, π) is a circle, and only cos β has interval ends; those ends are gimbal poles of measure zero.

### 2.5 Chirality and coverage

**Coverage [derived].** A crystal has a +1 description exactly when it contains a copy of M at a proper rotation.
- **Non-Sohncke G:** every crystal holds both hands, so all of M's crystals are covered.
- **Sohncke G:** a crystal holds one hand. The crystals the +1 sheet misses are crystals of the mirror-image molecule, which are a different system.
- ASU(Γ⁺) therefore reaches every crystal of M with no chirality token, in every group.

**The owner's doubled boxes are ASU(G⁺).**
- In sg 8, 9, 11, 12, 14 and 88, each doubled box holds exactly one G⁺ image per crystal [measured: 20,000 samples].
- Those boxes still hold [Γ⁺:G⁺] exact normaliser images: 2, 2, 8, 4, 8 and 4 respectively.
- In sg 109 and 110 the doubled box holds 4 G⁺ images, so it is not minimal.
- ASU(Γ⁺) is the further quotient of these boxes.

**Molecular symmetry is not folded (the owner's position).**
- An exactly achiral molecule keeps an exact 2× multiplicity [Inf.; not measured].
- Near-achiral near-images stay as separate modes. In sg 2 they sit 0.46 raw eLJ apart at a mirror rms of 0.042 Å [R2].

### 2.6 Composition with the cell

**Order of operations [derived]:**
1. Canonicalise the cell: pick (P, s) with det P = +1 that keeps SYM_OPS up to an origin shift s.
2. Re-express the pose in the new basis: t′ = P⁻¹(t − s) and R′ = UᵀR, where BP = U·B(c′).
3. Fold the pose by Γ⁺.

**Why the result is a product.**
- A generic cell's stabiliser has as its linear parts exactly the generic-metric isometries, and they act on the pose through Γ⁺.
- ASU(Γ⁺) is fixed in fractional coordinates.
- Hence D_cell × ASU(Γ⁺) × SO(3) is a fundamental domain up to measure zero.
- Cut B's zone is also independent of the cell: P⁺'s axes are fixed in MXtalTools' frame (a ∥ x, b in the xy plane) for every metric of a family [measured on generic cells].

*Table 3. Cell canonicalisation composed with the pose fold [math, `$FD/t5_cell.py`].*
- **Method:** 200 random crystals per group. A second description of each applies random proper setting changes, with their origin shifts. The canonicaliser picks the basis at which `cell_reduction_penalty` (`common/sym_utils.py:394`, margin 0) is exactly 0.
- **Reference:** 0 is exact. Float64 noise is about 1e-15 for fractional coordinates.

| sg | proper setting changes searched (count) | zero-penalty bases per reduced cell (count) | max \|Δ\| between the two canonical forms (Å or °) | max atom error after undoing the Γ⁺ isometry (fractional) |
|---|---|---|---|---|
| 2 | 3,480 | 1 | 3.2e-11 | 9.3e-16 |
| 14 | 36 | 2 (identity and the 2-fold about b) | 5.0e-13 | 9.2e-16 |

In sg 14 the second zero-penalty basis is the cell's own stabiliser, and the pose fold absorbs it. That is the mechanism the product argument relies on.

**Obstructions and gaps.**
- **Special metrics** on cell faces have larger stabilisers. This is measure zero.
- **Origin shifts:** every setting change must carry its origin shift.
- **Improper basis changes** are equivalences only when composed with an improper element of G. In Sohncke groups they give the enantiomorph.
- **Existing cell domains:**
  - Only triclinic (Niggli, β/γ-obtuse convention, `sym_utils.py:174-209`) and monoclinic (three classes, `sym_utils.py:214-267`) have them.
  - `standardize.choose_basis` returns no candidate for higher systems (`crystal_search/standardize.py:241`) [code].
  - 38 of the 59 orthorhombic groups carry 2, 3 or 6 exact copies of every lattice today [R3].
  - Tetragonal and hexagonal groups have no cell redundancy [R3]. Trigonal (P and R on hexagonal axes) and cubic groups have none either: checked 2026-10-21 by four independent enumerations of setting-preserving basis changes (`artifacts/fd_design_oct15/higher_sg_oct21/`). From sg 75 up every such change preserves the metric, so it is a Γ element and moves only the pose; in orthorhombic groups the cell redundancy is the axis order (`crystal_fd/axis_order.py`).
- **Defect: the triclinic canonicaliser at the β = 90° face.**
  - Within about 1e-7 rad of that face it is not invariant: 6 of 100 cells canonicalise to the face partner (α ↦ 180° − α, a change of up to 50°). At 1e-4 rad the count is 0 of 100. Monoclinic cells are invariant even at 1e-7 rad [verif].
  - **Cause [code]:** the tie-break key is the distance of N from the identity, i.e. from the input basis (`standardize.py:255`), and ties are counted within `PENALTY_ZERO = 1e-10` (`:50`).
  - Away from faces, cells return to their original form to within 6.5e-10 Å or degrees in sg 2, 14, 15 and 4 [verif].
- **Interval coordinates for the cell [R3; Inf.].** R3 checked these maps numerically against the zero-penalty set. The triclinic Jacobian is now checked by gate 5 (stage 0). The monoclinic map is built (`CellChart('monoclinic', length_box, sg)`, 2026-10-16) and its Jacobian is derived and verified [code: `mxtaltools/crystal_fd/charts.py` docstring; `tests/fd/test_monoclinic_cell.py` against autograd and central differences].
  - **Triclinic:**
    - Coordinates: log c; r2 = b/c; r1 = a/b; s_β, s_γ ∈ [−1, 0]; v ∈ [0, 1], with s_α = Λ + (1−Λ)v.
    - Jacobian relative to the chart measure d(log a) d(log b) d(log c) dα dβ dγ: (1−Λ) r1 r2 / (8 sin α sin β sin γ) [code: `mxtaltools/crystal_fd/charts.py`, `CellChart.map`; gate 5 against autograd and central differences]. The r1² r2² given here before 2026-10-16 was the angle rows' factor alone, (1−Λ) r1² r2² / (8 sin α sin β sin γ); the length rows contribute 1/(r1 r2).
    - Faces: β and γ = 90° are self-glued with a twist; the pairwise faces are glued by c ↦ c ± b; the order faces are mirrors (real walls).
  - **Monoclinic:**
    - Coordinate: s = sin(0.3π x₄)/m ∈ [0, 1], with Jacobian m / sin β relative to d(log a) d(log b) d(log c) dβ (α = γ = 90° fixed) [code; derived and verified 2026-10-16].
    - All faces are mirrors.

### 2.7 Measure

**The physical measure** is μ_cell(dc) × ∏ᵢ [V d³tᵢ × Haar(dRᵢ)].

**What the trainer uses today [code: GFN `energies/molecular_crystal.py:1202-1206`].**
- Cell rows: chart Lebesgue measure (d log L dθ), with no cell term.
- Centroid: (V/sym_mult)^{Z′}.
- Orientation: sin²(r/2) sin θ.

**The invariant alternative [derived].**
- Any φ(V) dG, where G is the Gram matrix, is invariant under basis change.
- d⁹h with frame rotations quotiented gives φ = 1/V.
- Relative to chart Lebesgue, the density of φ(V) dG is 8(abc)⁴ ∏sin θ · φ(V).
- Chart Lebesgue is not invariant. Restricting it to a different reduced domain therefore reweights lattices rather than just relabelling them. This is owner decision O4.

**The Jacobian each chart map carries:**
- **Centroid.**
  - Flow to ASU: Σ log(uᵢ − lᵢ), plus the spline slopes.
  - ASU to trainer latent: Σ log(2/u_d), a constant.
  - Γ⁺ folds: 0, since |det W| = 1 and left multiplication preserves Haar measure.
- **Orientation.** Euler (α, cos β, γ) to latent rows 9–11: log|det J| = −log(4 sin²(r/2) sin θ) − log(π³/4) [measured, 2.3e-9 nats].
- **Exact images.** log Z_chart = log Z_FD + log k. This holds only in the hard-wall limit: soft walls add leaked mass and 4.8–22 effective copies per triclinic lattice at reduction coefficient 10 [R3].

*Table 4. Measure and Jacobian checks.*
- **Sources:**
  - [math] `$FD/t7_measure.py`: 400,000 Haar samples;
  - [verif] `$FD/vs/vs_gate25_measure.py`: 10⁵ draws per side, and 10⁶ points for volumes.
- **Reference:** under a correct measure, p-values are uniform on (0, 1), invariance ratios are exactly 1, and a constant ratio means the two measures agree up to normalisation.

| check | quantity (unit) | result |
|---|---|---|
| Haar in the trainer's hemisphere rotvec chart | KS p for r, cos θ, φ against sin²(r/2) sin θ / 2π² | 0.63, 0.68, 0.23 [math]; 0.04, 0.76, 0.13 with χ² p = 0.35 [verif]; dropping sin θ gives p = 0 |
| Haar in Euler ZYZ | KS p for cos β, α, γ against uniform | 0.74, 0.09, 0.72 |
| Euler → latent rows 9–11 | max \|log\|det J\| − formula\| (nats), 289 points | 2.3e-9 |
| dG under G ↦ PᵀGP | \|det\| (ratio), 200 unimodular P | 1.000000000000 |
| chart Lebesgue under a basis change | \|det\| (ratio), 100 small P | 4.4e-6 to 0.22 (not invariant); matches the (abc)⁴∏sin θ ratio to 7.5e-8 nats |
| d⁹h against dG/V | ratio, 100 cells | 0.125000 in every cell |
| Fourier–Motzkin chart log-det | max \|closed form − central difference\| (nats) | ≤ 9.5e-10; equal to autograd |
| Fourier–Motzkin chart volume, sg 88, 92, 227 | z-score of ∫J against 1/\|Γ⁺:ℤ³\| | −0.3, −0.3, −0.8 |
| canonicalised physical draws against the chart's own density | KS and χ² p over 8 seeds (96 p-values), tested for uniformity | p = 0.45; a face pushed out by 0.01 gives p ≤ 1e-26 and a volume z of +54 |

### 2.8 Z′ > 1

*2026-10-23: [zprime_fd.md](zprime_fd.md) (a proposal, independently orbit-checked in all 230 groups at Z′ = 2) finds that this section holds only for rigid units on the +1 sheet. Once each unit's conformer is sampled in internal coordinates its hand lives in the conformer, so the residual group of unit 2 is G, not G⁺ (ASU(G⁺) leaves 2 images per unit in non-Sohncke groups), and the swap rule below is a continuous key, which the owner's answer of 2026-10-22 rules out. Read that note before building on this section.*

**The equivalence group.**
- It is not diagonal Γ⁺ alone. It contains (g₁n, g₂n) with n ∈ N_E and each gᵢ ∈ G chosen so the unit stays proper.
- It also contains independent lattice shifts per unit, the common free-axis shift, and the swap S₂ [derived].

**The fundamental domain for Z′ = 2 [derived, measured below].**
- **Unit 1:** in FD(Γ⁺), with its free coordinates pinned. This uses up the joint shift, and K cuts R₁.
- **Unit 2:** in ASU(G⁺) × SO(3).
- **Swap:** keep h(R₁) < h(R₂), where h(R) = |a·R e_z| and a is P⁺'s principal axis.
  - For cyclic P⁺ this is one AR bound.
  - For dihedral P⁺ it is a union of two intervals.
  - No invariant was derived for T or O.

*Table 5. Z′ = 2 orbit counts [math, `$FD/t6_zp2.py`]. 1,500 random pairs per group, with each pair's full orbit enumerated. Reference: a fundamental domain gives exactly one orbit point for every pair.*

| sg | \|Γ⁺\| (per cell) | \|G⁺\| (per cell) | \|K\| | unit-1 centroid volume (fraction of the reduced cell) | unit-2 centroid volume (fraction of cell) | pairs with exactly one orbit point (count of 1,500) |
|---|---|---|---|---|---|---|
| 2 | 8 | 1 | 1 | 1/8 | 1 | 1,500 |
| 4 | 8 | 2 | 1 | 1/8 (2-D) | 1/2 | 1,500 |
| 9 | 4 | 2 | 2 | 1/2 (1-D) | 1/2 | 1,500 |
| 14 | 16 | 2 | 1 | 1/16 | 1/2 | 1,500 |
| 19 | 32 | 4 | 1 | 1/32 | 1/4 | 1,500 |

**Controls, sg 14, 600 pairs each.**
- Without the swap rule, every pair (600 of 600) has 2 orbit points: the exact Z′! multiplicity.
- With unit 2 restricted to ASU(Γ⁺), 533 pairs have 0 orbit points and 67 have 1. So unit 2's residual group is G⁺, not Γ⁺.

### 2.9 Energy evidence that the Γ⁺ images are physical

*Table 6. eLJ of descriptions predicted to be the same crystal.*
- **Route:** the trainer's own (`MolecularCrystal`, `analyze(['reduction_en','elj'])`, `std_orientation=False`, CPU).
- **Molecules:** synthetic 5- or 6-atom chiral molecules and a 3-atom planar one.
- **Scale and resolution:** |eLJ| is 6–60 raw units per crystal. The resolution is set by float32 build noise (about 1e-5 raw) and by the gap between the trainer route and a direct `analyze` call (≤ 5.1e-4 raw).
- **Controls:** the same pose read through a wrong (improper) gluing.

| source | space groups | samples | max \|ΔeLJ\| for predicted equivalences (raw eLJ units) | wrong-gluing control (raw eLJ units) |
|---|---|---|---|---|
| R1 face identities (problem statement Table 3.4) | 2, 14 | 8 crystals per identity | 1.8e-4 chiral; 3.6e-4 planar | medians 7.2–20.6 chiral, 3.3–6.2 planar |
| R2 image check | 11 groups | 37 det +1 images | relative ≤ 1.7e-4 | median 3.2–49 |
| [verif] canonicalised against the source | 2, 14, 15, 19, 61 | 16 rows per group | 2.1e-4 (relative ≤ 4.4e-5) | median 2.4–54; minimum 0.135 |
| [verif] canonicalised, wrapped to [0,1)³ before building | 92, 88 | 16 rows per group | 8.6e-5 (sg 92); 1.7e-3 (sg 88, relative 3.4e-4) | — |
| [verif] canonicalised, not wrapped | 92 | 16 rows | 29.9 (fails) | — |

The unwrapped failure comes from the builder: it **clips** fractional centroids to [0, 1−1e-4] instead of wrapping them (`dataset_utils/data_class_methods/crystal_ops.py:114-116`, called at `crystal_building/utils.py:996`) [code]. A canonical point with a negative coordinate is therefore built as a different crystal.

### 2.10 Disagreements, resolved

1. **Free-axis groups.**
   - [verif] stated the domain as "projected ASU × SO(3)"; [api] sketched the same.
   - [math] measured that sg 6–9 need K = C2(b) cut in SO(3), because the uncorrected form double-counts by a factor of 2.
   - [verif]'s free-axis check counted centroids only, so it does not contradict [math]. **Resolution: [math]'s form.**
2. **Notation.** [api]'s "K (rotation parts)" is P⁺ here; "K" here is [math]'s kernel on the projected centroid.
3. **Which cut to use.**
   - [api] recommended cut B wherever P⁺ is not T or O.
   - [verif] prototyped cut A (Fourier–Motzkin polytopes).
   - [math] measured both.
   - **Resolution:** owner decision O1, with A recommended (§6).
4. **Cut B in T and O groups.** [api] said cut A is required there; [math] counted a Rodrigues-zone cut B successfully in all 230 groups. **Resolution:** cut B exists everywhere for counting, but has no measured chart for T and O, because the Rodrigues Haar density is unchecked.
5. **vol × |Γ⁺|.** [api]'s 4,000-point estimates for sg 132 (0.947) and sg 202 (1.08) were estimator noise: [math] computed 1.000000 in all 162 groups. Gate 0 makes this exact.
6. **Orientation log-det.** [api] inferred the formula "up to a constant"; [math] measured the constant as −log(π³/4), to 2.3e-9 nats.
7. **Energy evidence.** [math] computed no energies itself; [verif]'s gate-4 prototype and R2 supply the evidence (Table 6).
8. **Callers of the existing fold code.**
   - [verif] said `get_fd_params`, `fold_to_fd` and `normalizer_images` have no callers.
   - [code, checked by grep]:
     - `get_fd_params` is called only by `fold_to_fd`, and `fold_to_fd` has no callers.
     - `normalizer_images` is imported by GFN `data_processing/symmetrize_prior.py:349`, in a check, and by `tests/test_normalizer_images.py`.
   - **Resolution: [api]** — keep `normalizer_images`.
9. **Shared constants and the live builder.**
   - [verif] proposed rationalising the SYM_OPS translations, which are stored to 8 decimals (`space_group_info.py:40-43`). [api] requires the SYM_OPS values to stay bit-identical.
   - **Resolution:** the new table carries exact rationals and SYM_OPS is not touched. Gate 0 checks that the two agree to 1e-8. Literal checks on the live builder path keep a floor of about 3e-7 Å in the 52 trigonal and hexagonal groups.
   - The same principle applies to the builder clip: the tooling wraps every centroid before building, and the live builder is not changed.
10. **The older design note (`fundamental_domain.md`) and wiki.**
    - **§2's "39 easy groups" criterion** (normaliser = pure translations) is the wrong cut on the +1 sheet. At fixed handedness, Γ⁺ carries 2-folds (sg 14 has P⁺ = C2), yet axis boxes exist in all 74 triclinic, monoclinic and orthorhombic groups. Of the 39, 30 have box domains and 9 (91, 92, 95, 96, 124, 128, 131, 135, 142) have polytopes [verif].
    - **"Outside this set … the FD stops being a product box"** is contradicted: the domain is a product in all 230 groups (§2.2).
    - **"The domain is per-molecule"** does not hold on the +1 sheet when molecular symmetry is not folded (§2.5).
    - **F-007's sg 14 domain** (volume 1/32) equals 1/|N_E|, which counts both hands; the +1 domain is 1/16.
    - **The wiki asymmetric-unit-reduction** says "the candidate pose count is one plus the list length". That overcounts the index in 60 groups. Closure confirms R2's deduplicated counts over R1's (sg 5: 4; sg 9: 2; sg 42 and 43: 2; sg 88: 4) [math].
11. **The largest Γ⁺.** [api] and [verif] assumed |Γ⁺| ≤ 192; it is 384 in sg 196, 216 and 219 [measured].
12. **Nonstandard settings.** [api] proposed a one-way *import* and never writing a nonstandard setting; [verif] proposed a one-way *export* and refusing imports. **Resolution:** owner decision O7, answered 2026-10-16 by D1: one-way import (§6).
13. **Box counts.** "Boxes in 44 of 162 groups" [api] and "74 box groups" [verif] are the same fact with different denominators (with and without the free-axis groups).
14. **The ASU reference generator.** `constants/generate_asymmetric_unit_reference.py:26` says the stored and cctbx boxes "agree on all such groups". They do not: 13 differ [api]. Nine differ in origin (48, 70, 85, 126, 130, 133, 137, 141, 142) and four in extent (50, 59, 68, 86). Twelve of these do not tile; sg 142 tiles anyway [R1].
15. **The "15 nat wrong" sg 2 x-shift images** (`symmetrize_prior.py:26-29`) are not a conflict.
    - Those images need the inversion fold to re-enter the trainer box (x ∈ [0, ½]), so read at +1 they are wrong.
    - The x = 0 ↔ ½ translation gluing itself is exact: 1.2e-4 raw [R1].

## 3. API and package layout

### 3.1 Rules

- **Additive.** No function on the live trainer path changes (the list is in §3.11).
- **Independent.** MXtalTools never imports GFN, and runtime never imports cctbx.
- **One table.** Every per-group fact comes from one generated, hash-stamped table, never from per-group code branches.
- **Matrices, not rotation vectors.** Rotations are stored as matrices; a rotation vector is a chart. `rotmat2rotvec` replaces rotations near π or near the identity with π about (1,1,1) (`crystal_building/zp_doubling.py:44-46`, `crystal_search/standardize.py:20-21`) [api].
- **Dtypes and devices.** Group arithmetic is float64; building and energies are float32. Tables are cached per device on first use, as `asym_unit_lut` is (`crystal_ops.py:916`).

### 3.2 The table

`group_table(sg, device=None) -> GroupTable`. The table is generated offline by `generate_fd_tables.py`, which absorbs `generate_normalizers.py` and `generate_asymmetric_unit_reference.py`. It is stored as `data/fd_tables_v1.json`, with exact rationals as strings and integer matrices.

**Per group, it holds:**
- **G:** operations index-aligned with SYM_OPS, as exact rationals.
- **N:** the deduplicated N_E representatives.
- **Γ elements:** W, w and det for each, plus the Γ⁺ indices, the L⁺ basis, P⁺ (name and principal frame), K, and the free axes.
- **The G-ASU:** cctbx's polytope, with its origin and tie-break cuts.
- **The translation cut:** the Γ⁺-ASU cuts, tie-break cuts, the gluing element and kind for each facet, and an AR order with its Fourier–Motzkin bounds.
- **The orientation cut:** the L⁺ basis and the Euler ranges; null for T and O groups. Generated only if O1 = (b).
- **The cell class.**
- **The trainer box** as stored in ASYM_UNITS, with its coverage and k_box.
- **Validation results:** closure, the orbit count, and vol × |Γ⁺|.

**`MANIFEST.json`** records:
- the schema version, the generator path and commit, the cctbx version and the date;
- sha256 hashes of canonical serialisations of SYM_OPS, NORMALIZER_OPS, CONTINUOUS_DIMS and ASYM_UNITS.

The loader recomputes those hashes at import and raises on any mismatch. Consumers such as prior files and flow artifacts store `table_hash`.

**The table digest [code: `mxtaltools/crystal_fd/tables.py`, `exact.text_sha256`; gate 0].** It is taken over LF-normalised bytes (CRLF read as LF), so a CRLF checkout keeps its digest. `data/.gitattributes` (`*.json -text`) stops git converting the JSON and must be committed with it.

### 3.3 Data model

**`CrystalDescription`** is a dataclass of batch-first tensors, not a PyG `Data`. Its fields:
- `sg` [B], long;
- `cell` [B,6]: a, b, c in Å; α, β, γ in radians;
- `z_prime` [B] and `unit_mask` [B, Z′max];
- per unit: `t` [B,Z′,3] fractional, `R` [B,Z′,3,3] proper, and `h` [B,Z′] = ±1.

The pose is P = R·diag(h,1,1) acting on body coordinates. `zp_doubling.py:12-14` and `canonicalize_aunit` (`crystal_opt_utils.py:69-135`) already use this convention.

**`MoleculeRef`** (one per unit) holds body coordinates X_ref, the species, and a hash of X_ref.
- X_ref is centred on the heavy-atom centroid, as the builder centres it (`crystal_building/utils.py:1036-1041`).
- For GFN rows, X_ref is the stored `pos` after admission orientation (GFN `buffer.py:623`), built with `std_orientation=False`. No principal-axis frame is recomputed.
- For a CSD row re-described under D2 (§6), X_ref is the mirror diag(−1,1,1)·X of the aligned stored `pos`, and the row carries `reference_handedness` = −1 (+1 on every other row) [code: `crystal_fd/io_mxt.py` `ImportInfo.reference_handedness`; `crystal_fd/dataset.py`]. Any re-alignment of X_ref must pass that flag as `target_handedness`. GFN's `orient_molecule(mode='std')` without it maps X_ref back to X, and the build is a different, clashing crystal: on 562 sampled mirrored rows the eLJ is off by a median 993 raw without the flag and 2.4e-4 raw with it [measured 2026-10-17, `artifacts/fd_design_oct15/canonical_data/reverify_importer/`]. Consumers keep the saved X_ref; re-deriving it from `pos` is unsafe, since a molecule with coplanar heavy atoms can come back in a reflected frame [measured: QQQFVD03].

**The setting** is always SYM_OPS[sg]. Nonstandard operator lists exist only at the import boundary (`io_mxt.py` with `settings.py`), which re-expresses them into SYM_OPS[sg] (D1, §6).

### 3.4 Operations

**Group action.**
- `apply(desc, W, w)`: P ← Q·P with Q = T_fc W T_cf, then a polar split into (R, h). This generalises `_transform_aunit_params` (`crystal_ops.py:1427`, validated 800/800), using `proper_rotvecs` (`zp_doubling.py:39`).
- `compose(...)` and `inverse(...)`.

**Folding.**
- `fold(desc, section, tol=1e-9) -> FoldResult`. The result holds [code: `mxtaltools/crystal_fd/domains.py`]:
  - `desc`;
  - `element` [B], an index into Γ⁺;
  - `lattice` [B,3], int;
  - `free_shift` [B,3], the continuous shift that moved free coordinates to the gauge;
  - `slack` [B];
  - `tie` [B]: 0 interior; 1 on a face, resolved by the cut trees; 2 several candidates passed and the orientation rule chose (a stabiliser, or K); 3 the band fallback: no candidate passed the cut trees within `tol` of a facet, edge or vertex, and the orientation rule chose among the candidates within `tol` of the largest slack. A `tie = 3` row is a valid image of the input within about `tol` of ASU(Γ⁺), not a defect.
- `unfold(desc, element, lattice, free_shift)` is the exact inverse.
- `in_domain(desc, section, tol) -> (inside, slack)`: inside means the row is its own fold, so every orbit has exactly one such row (except where a slack lies within rounding of exactly ±`tol`). The rule-by-rule membership this bullet first proposed is `in_asu`, kept for validating a domain; within the `tol` band it can count an orbit 0 or 2 times.
- **How the fold works:**
  - It applies all |Γ⁺| affine maps (at most 384) in vectorised form, adds lattice shifts, and tests membership by slack.
  - cctbx's tie-break cuts are consulted only when |slack| <= tol.
  - It never rebuilds a crystal; `get_fd_params` rebuilds every candidate (`crystal_ops.py:1560-1566`).
  - It is piecewise isometric, so it is differentiable inside each piece.

**Images and handedness.**
- `images(desc, group='gamma_plus'|'gamma', within=None|'g_asu'|'latent_v1_box') -> (desc, source, element)` (`'g_plus_asu'` was planned and is not built: it raises). This regenerates any ×k prior layout for any group.
- `to_sheet(desc, +1) -> (desc, ok)`. It returns ok = False for a chiral molecule's crystal in a Sohncke group whose description needs the mirror molecule.
  - The CSD importer no longer leaves such rows off the sheet. Under D2 (§6) it re-describes a Z′ = 1 Sohncke row with X′ = diag(−1,1,1)·X at h = +1 and marks it `ImportInfo.mirrored` (`from_mol_crystal_data(mirror_reference=True)`, the default; built 2026-10-16). Z′ > 1 rows are refused (D3).

**Cell.**
- `reduce_cell(desc, tol=1e-10, angle_tol=1e-6, face_tol=FACE_TOL) -> CellReduction` and `change_basis(desc, N, s)` [code: `mxtaltools/crystal_fd/ops.py`; sg 1–15]. `CellReduction` holds `desc, N, s, n_tied, violation, n_auto`.
- The tie-break is input-independent: ties are broken by the candidates' own Gram entries, never by the input basis (§2.6).
- `n_auto` counts the proper automorphisms of the reduced metric (1 for a generic lattice; monoclinic: within the setting class, modulo the 2-fold about b). Rows with `n_auto > 1` (special metrics, for example β = γ = 90°; every face of the monoclinic domain) are flagged, not resolved: the reduced cell is unique but the canonical pose depends on the input basis. Equality is tested within `face_tol` (default `FACE_TOL` = 1e-5 on the tie-break key, a choice the owner confirmed on 2026-10-16, §6 D4), not the search's 1e-10, so that a cell its source reported on a face is still flagged after float32 storage (the CSD datasets store cells in float32; a 90° angle is 4.4e-8 rad off) [code; tests/fd/test_monoclinic_cell.py, test_gate1_orbits.py, test_io_mxt.py].

**Free axes.**
- `gauge_free_axes(desc)` follows the rule of `canonicalize_free_axes` (`crystal_ops.py:2763`).
- The projection onto the non-free coordinates, and K, come from the table [Inf.; built: `fold` uses both in every group].

### 3.5 Charts

**The chart protocol.**
- `dims`: per coordinate, its kind (interval, circle, fixed or embedded), its period, and its position in the AR order.
- `forward(y, ctx) -> (desc, logabsdet)`.
- `inverse(desc) -> (y, logabsdet)`.
- `bounds(y_prefix, i) -> (lo, hi)`.

**The charts** (names as built in stage 0, `mxtaltools/crystal_fd/charts.py` [code]; unbuilt charts are marked):
- **`PolytopeARChart(sg)`** (centroid, cut A): Fourier–Motzkin bounds come from the table; log|det| = Σ log(hiᵢ − loᵢ). A box is the special case with constant bounds. Every pair of faces glued by a pure translation is a circle of period 1 (`seams`); in the 9 tetragonal groups whose seam runs along (½, ±½, 0) (88, 91, 92, 95, 96, 98, 122, 141, 142) the coordinates are a sheared basis (x+y, x−y, z) with its Jacobian (built 2026-10-21; `ar_coordinates`, `stages_in`).
- **Torus chart in an L⁺ basis** (centroid, cut B only): not built.
- **`OrientationChart(name)`**, over the whole of SO(3) (cut A), each with its Haar log-density; `name` is required:
  - `'hemisphere'`: the trainer's (θ, φ, r), with Haar density sin²(r/2) sin θ / 2π²;
  - `'euler_zyz'`: (α, u = cos β, γ), Haar uniform in (α, u, γ);
  - `'quaternion'` (w ≥ 0) and `'axis_angle'` (the π ball).
- **Euler zone of P⁺** (cut B only; α of period 2π/n, u ∈ [0, 1] for dihedral P⁺): not built.
- **`CellChart(family, length_box)`:**
  - built: R3's triclinic Λ map (`family='triclinic'`);
  - built: the three monoclinic classes (`family='monoclinic'`, `sg` names the class; 2026-10-16);
  - built 2026-10-21, one coordinate per free cell parameter (owner answer of 2026-10-21, §6): `'orthorhombic'` (log a, log b, log c; `sg` names the axis-order class of `crystal_fd/axis_order.py`, whose constraints are the AR bounds; Pbca is the shortest-first wedge with its two faces glued by the 3-cycle, interval ends under O1 (a)), `'tetragonal'` and `'hexagonal'` (log a, log c; trigonal P and R on hexagonal axes are `'hexagonal'`), `'cubic'` (log a). log|det| is 0 relative to the free log lengths; `charts.cell_family(sg)` names a group's family.
- **`CrystalChart(sg, cell, centroid, orientation)`:** their product on the +1 sheet, every group; `CrystalChart.for_length_box(sg, (lo, hi), orientation)` takes an explicit length box, `for_latent_v1(sg, radius, orientation)` the trainer's (used as a box only above sg 15). sg 6–9 only with `orientation_cover='double'`: the chart covers both K images (multiplicity 2, log 2), a density on it must be K-invariant relative to the reference measure, and `k_image` gives the other image for augmentation.
- **`LatentV1Chart(sg, radius, z_prime)`:** a read-only, exact wrapper of `latent_to_cell_params`, `latent_transform` and `inv_latent_transform` (`crystal_ops.py:480`, `:617`, `:695`), clamp included. It returns a `clamped` mask instead of clamping silently. Above sg 15 its forward, inverse and log-Jacobian refuse: the trainer latent there (averaged length rows, no axis order) is to be replaced in the trainer, not mirrored (owner, 2026-10-21).

### 3.6 Measure

- **`log_reference_density(desc, chart, cell_measure)`** returns the log-density of Haar ⊗ (V^{Z′} d³t per unit) ⊗ the cell measure, relative to the chart's Lebesgue measure.
  - `cell_measure` is required and has no default: `'latent_v1_lebesgue'` or `'metric_invariant'` (O4).
  - `'latent_v1_lebesgue'` is flat in each free log length (and each free angle) in every family, which from tetragonal up is not the measure the trainer's averaged latent induces (owner answer of 2026-10-21, §6).
  - `'metric_invariant'` is built for sg 1–2 only and raises elsewhere [code: `crystal_fd/measure.py::log_reference_density`]. It stays unimplemented for monoclinic groups: D5 was withdrawn on 2026-10-16 and the owner keeps `'latent_v1_lebesgue'` (§6).
  - **Contract:** for `LatentV1Chart` it equals −jacobian_energy/T from GFN `molecular_crystal.py:1202-1206`, up to a constant (gate 8).
- **`log_multiplicity(sg, src, dst)`** returns log k between two domains, for comparing log Z. It is valid in the hard-wall limit only. Domains include `'double_cover'` (sg 6–9: log 2); `'latent_v1_box'` refuses the orthorhombic groups with more than one axis order, whose trainer image count depends on the cell.

### 3.7 Build and parameterise

- **Building, stages 0–3.** `to_mol_crystal_data(desc)` wraps t to [0,1)³ and hands the result to the live builder (`aunit2ucell`, `crystal_building/utils.py:958`). Energies therefore go through exactly the trainer's path.
- **A closed-form builder.** `build_unit_cell` and `build_cluster` from `crystal_building/image_pairs.py` would come later. That file is **untracked** in MXtalTools (git status `??`), so nothing here depends on it until the owner commits it.
- **`parameterize(cell, sg, unit_cell_atoms, species, mol_ref) -> desc`.**
  - It fits each pose by Kabsch against X_ref, with atom correspondence; det gives h.
  - It replaces principal-axis extraction (`crystal_building/utils.py:683-737`), whose sign fix fails for mirror planes and symmetric tops (wiki asymmetric-unit-reduction).

### 3.8 Formats

- **To and from `MolCrystalData`.**
- **`to_cif(desc, path, mode='asymmetric_unit'|'unit_cell')`:** the asymmetric unit through `common/cif_io.py`, the unit cell through ASE's writer.
- **`from_cif(path)`:**
  - It uses the construction reader, which keeps the file's own operators (`dataset_utils/construction/cif/symmetry.py:3-11`). 19.2% of CSD entries carry a different operator set (`cif_io.py:138`).
  - A file whose operators equal SYM_OPS[sg] as a set is parameterised directly.
  - **Nonstandard settings:** O7 is answered by D1 (§6): one-way import into the standard setting; export is not chosen.
    - Built for `MolCrystalData` rows (`io_mxt.from_mol_crystal_data(nonstandard='reexpress')`, the default), not yet for CIF files.
    - The import does not use spglib. `settings.find_setting` derives (P, p) from the stored operators by a bounded search and accepts it only after an exact check that the re-expressed group equals SYM_OPS[sg] (`setting_matches`); `settings.re_express` applies it [code].
    - Centred P1 settings (det P = 1/2) are handled; the cell halves and Z′ stays 1.
    - A row the search cannot resolve is refused as `unresolved_setting`; a re-expressed row whose cell the group's rotations do not preserve is refused as `setting_metric_mismatch` (6 CSD rows).
- **`to_ase` and `from_ase`.** pymatgen is an optional import; it is absent from `csd_mxt_gfn`.

### 3.9 Checks shipped with the package

- `same_crystal(a, b) -> max atom deviation (Å)`, generalising `standardize.py:498-519`.
- `orbit_check(sg, section, n)`.
- `round_trip(desc, path)`.

### 3.10 Package layout

The name `crystal_fd` is provisional. A module is written in the stage that first needs it.

```
mxtaltools/crystal_fd/
  tables.py        stage 0  loader, manifest/hash check, GroupTable
  description.py   stage 0  CrystalDescription, MoleculeRef
  so3.py           stage 0  split_o3, Euler/rotvec conversions, Haar densities
  ops.py           stage 0  apply/compose/inverse; triclinic and monoclinic cell reduction by their own basis searches (triclinic_basis, monoclinic_basis; standardize.choose_basis untouched)
  domains.py       stage 0  fold/unfold/in_domain/images/to_sheet
  charts.py        stage 0  PolytopeARChart, OrientationChart, CellChart(triclinic, monoclinic), CrystalChart, LatentV1Chart
  certify.py       stage 0  exact tiling certificate of the centroid part (§2.2)
  measure.py       stage 0  log_reference_density, log_multiplicity
  checks.py        stage 0  same_crystal, orbit_check, round_trip
  io_mxt.py        stage 0  MolCrystalData import: nonstandard settings re-expressed (§6 D1), mirrored reference for off-sheet Z' = 1 Sohncke rows (§6 D2), Z' = 1 unless allow_zp_gt1 (§6 D3 keeps Z' > 1 refused), refusal reasons (built 2026-10-16)
  settings.py      stage 0  find_setting, setting_matches, re_express: the change of setting (P, p) into SYM_OPS[sg] (§6 D1; built 2026-10-16)
  dataset.py       stage 0  canonical datasets on disk: save_dataset, load_dataset, FDDataset.to_mol_crystal_data, code_digest (§6 D6; built 2026-10-17)
  formats.py       stage 1  CIF/ASE (not built); the O7 direction (§6 D1, import) is built for MolCrystalData in io_mxt.py
  build.py         later    closed-form builder, Kabsch parameterize
  generate_fd_tables.py     offline, cctbx
  data/fd_tables_v1.json, data/MANIFEST.json
tests/fd/                   conftest + gates 0-7 (section 4)
```

**What happens to the existing pieces:**
- **`_transform_aunit_params`** (`crystal_ops.py:1427`) becomes `ops.apply`, acting on matrices.
- **`get_fd_params` and `fold_to_fd`** (`:1472`, `:1615`) are superseded by `fold`. Today they:
  - rebuild every candidate;
  - discard `h_folded`;
  - judge det W alone instead of the composite's det (`:1568-1571`);
  - fold within G's box;
  - leave `.pos` stale.

  They are retired after `fold` lands; they have no external callers (§2.10, item 8).
- **Kept unchanged, and wrapped:** the latent maps (`:480-755`), which `LatentV1Chart` wraps under a bit-identity test.
- **Kept unchanged:**
  - `canonicalize_free_axes` and `canonicalize_zp_aunits` (`:2763`, `:2826`);
  - `compute_standard_cell` (`:2856`), import only;
  - RAW_ASYM_UNITS and ASYM_UNITS, SYM_OPS, NORMALIZER_OPS and CONTINUOUS_DIMS, which become hash-stamped generator inputs;
  - the `sym_utils` penalties (`:174-267`, `:394`);
  - the live builder (`aunit2ucell`, `get_aunit_positions`, `parameterize_crystal_batch`, `identify_canonical_asymmetric_unit`);
  - `normalizer_images` (`standardize.py:354`), kept as a wrapper because GFN calls it.
- **`validate_asym_units.py`** becomes tests.
- **Retired:** `crystal_latent_transforms.py`, the unimported 2025 Niggli transforms.
- **GFN:** `symmetrize_prior.py` (sg 2 only), `pool_anchors.py::Chart` (sg 2 and 14 hand-coded) and `pool_assemble.py::periodic_dims` are superseded for new priors. Existing files are untouched.

### 3.11 GFN consumption

**Adapter (with stages 0–1).** A new GFN file (provisional name `energies/chart_adapter.py`).
- **The flow** lives on the domain's chart: bounded splines on intervals with AR bounds, circular splines on circles, and a uniform base.
- **`forward(y, mol) -> (x, log|∂x/∂y|, representable)`** maps y → desc → fold into the trainer box at +1 → x through `LatentV1Chart`.
- **The target over y** is log R(x(y)) + log|∂x/∂y|.
  - R is the trainer's reward, scored through `analyze_crystal_batch` (GFN `energies/molecular_crystal.py:739`).
  - Inside the domain intersected with the clamp box, the trainer's bounding term and `reduction_en` (margin 0) are exactly 0, so this is the hard-wall target [Inf.; gate 8 pins it].
  - log|∂x/∂y| is closed form: §2.7 for centroid and orientation, the `CellChart` Jacobian with κ = ½ ln 40 (`crystal_ops.py:678-685`) for the cell, and 0 for the fold.
- **`representable = False` when:**
  1. the domain point has no Γ⁺ image in the trainer box, which happens in sg 8, 9, 11, 12, 14, 88, 109 and 110 for any molecule that is not exactly achiral, and (found 2026-10-21) in 70, 85, 126, 130, 133 and 137, whose boxes cover only part of the domain (handling: O2);
  2. θ < π/400 or r < π/100 (the −0.99 floors, `crystal_ops.py:546-558`). These cover Haar masses of 3.1e-5 and about 8e-7 [Inf., arithmetic checked];
  3. the cell lies outside the clamp box. The flow's domain is defined as the fundamental domain intersected with that box, through `CellChart(length_box)`.
- **log Z:** log Z_trainer (hard walls) = log Z_FD + log k_box. For sg 2, k_box = 4.

**Prior intake (stages 0–1).** A read-only migration that writes new files beside the old ones:
1. Rebuild each row exactly as the trainer would.
2. Canonicalise the cell, then the pose. Record the domain coordinates, the element used, and any tie flags.
3. **Collapse check:** every image of one source row in the ×4 and ×8 files must map to one domain point, to ≤ 1e-6. Each symmetrised file's canonical set must equal its canonicalised unsymmetrised source.
4. Rescore through the trainer path, to gate-4 tolerance.
5. Flag rows that sat on box faces (52.7% of MIPCAS anchors, from memory, not rechecked) and rows that canonicalisation moved into the domain. Flag them; do not drop them.
6. Write 1/k of the old row count, with provenance: source sha1, `table_hash`, and the MXtalTools commit.

**As built for sg 2 (§6 D6, 2026-10-16/17):** GFN `data_processing/fd_canonical_prior.py` wrote the four sg 2 priors as `<stem>_fd1` files beside their sources (§6, D6).
- They are training data for a model in the fundamental domain, not a drop-in prior for today's trainer: every output row sits in one quarter of the trainer's sg 2 centroid box (k_box = 4), where the sources spread about 25% into each quarter [measured]. A model fitted in the trainer latent would see no data in the other three quarters.
- Stored energies are carried from the source, not recomputed; step 4 ran as a sampled check (1,000 rows per file), not a rescoring.
- Rows the trainer's latent clamp would move are refused with counted reasons (§6 D7).

**Must stay bit-identical:**
- **MXtalTools:**
  - `latent_to_cell_params`, `inv_latent_transform`, `latent_transform` and `latent_params`;
  - `canonicalize_free_axes` and `canonicalize_zp_aunits`;
  - the values of ASYM_UNITS, SYM_OPS, CONTINUOUS_DIMS and NORMALIZER_OPS;
  - `aunit2ucell`, `get_aunit_positions`, `parameterize_crystal_batch`, `mol2cluster`, `analyze` and the `sym_utils` penalties;
  - the signature of `normalizer_images`.
- **GFN:** `aunit_periodicity`, `dead_latent_rows`, `molecular_crystal`, `prior_flow`, `mk_dev.yaml`, the buffer and prior formats, `symmetrize_prior` and `pool_assemble`.

**Not proposed here.** A native domain chart inside the GFN trainer would mean:
- a new config key;
- table circle dimensions replacing the `auv == 1` rule (`models/aunit_periodicity.py:53-67`);
- incompatible checkpoints.

It is a separate owner decision after the gates pass.

## 4. Verification suite (acceptance gates)

**Where the tests live.**
- The gates live in `MXtalTools/tests/fd/`, with their own conftest, following the pattern of `tests/cif/` and `tests/golden/`.
- **Markers:**
  - existing: `slow` (`pyproject.toml:101-106`, alongside `gpu`, `mlip` and `ccdc`) and the golden tests' conftest;
  - new: `cctbx`, which skips when cctbx is not importable.
- **Fixtures:** the conftest supplies a synthetic chiral molecule, a planar one, random cells per family, and a float64 literal-atom oracle.
- **GFN-boundary versions** go in GFN `tests/crystal/`.

**Gate 0. Tables** (all 230 groups; golden; seconds).
- **Group checks:**
  - Γ⁺ is closed, every element has det +1, and Γ⁺ contains G⁺ and ℤ³;
  - Γ⁺ equals brute-force N_E⁺, and under the `cctbx` marker it equals cctbx's;
  - K is nontrivial exactly in sg 6–9.
- **Domain checks:**
  - vol × |Γ⁺| = 1 exactly, by rational vertex enumeration;
  - every facet carries a gluing element and its tie-break cuts;
  - built in stage 0: the exact tiling certificate of §2.2, with its mutation controls (`tests/fd/test_gate0_tiling_certificate.py`).
- **Consistency checks:**
  - the table's rationals round to SYM_OPS within 1e-8;
  - the manifest hashes match;
  - the table digest, over LF-normalised bytes (§3.2), is pinned under the golden regeneration policy.

**Gate 1. Exactly one representative per orbit.**
- **Pose, generic points.**
  - Run with orientations, not centroids alone, so that sg 6–9 are exercised.
  - 10⁴ points per group against all |Γ⁺| images. Zero failures are allowed outside a 1e-6 face slab.
  - At that sample size, a defect of measure ≥ 1e-3 is missed with probability ≤ 4.5e-5.
  - Measured runtime is about 20 s for 162 groups.
- **Pose, special points.** Exact rational points on faces, edges and special positions pass when the canonical form of γx is identical for every γ ∈ Γ⁺.
- **Cut B zones**, if built: include the D2 Euler box, which is untested so far.
- **Cell.** canonicalise(scramble(c)) = canonicalise(c) to 1e-9 Å or degrees.
  - Tested in the interior and at ±1e-7, 1e-4 and 1e-2 rad from every face.
  - This **fails today** for triclinic cells (§2.6).
- **No molecule argument.** The canonicaliser takes none. For a planar molecule, the improper images re-expressed through the molecular mirror stay distinct domain points.
- **Mutation control:** a face pushed out by 0.01 must fail.

**Gate 2. Measure.**
- **(a) Volume.** ∫J du = vol(domain): exact for boxes; 10⁶ Monte Carlo points for polytopes (relative standard error 0.05%).
- **(b) Histogram test.** Independent physical draws are canonicalised, charted and compared with draws from the chart's own density J.
  - The physical draws are uniform t, scipy Haar R and a reference cell sampler.
  - Two-sample KS plus χ², 10⁵ per side. The critical D at α = 1e-4 is about 0.010.
  - Use a family-wise threshold.
- **(c) Orientation.** Every orientation chart, each with its measure term.
- **(d) Cell.** Rejection sampling in coordinates where the O4 measure is Lebesgue (Gram entries for dG; (log L, θ) for chart Lebesgue), restricted to the zero-penalty set.
- **(e) Adapter.** The symmetrised density q_FD(canon(x))/k_box(x) integrates to 1 over the trainer box.

**Gate 3. Round trips and the literal oracle.**
- **Chart → description → chart:** ≤ 1e-12 in float64 and ≤ 1e-6 latent units in float32.
  - Tested in the interior and at 1e-9 to 1e-3 from faces.
  - Also across circle seams, and near r = 0, π and 2π and θ or β at 0 and at the equator.
- **Description → canonical → atoms, against the float64 oracle:**
  - ≤ 1e-10 Å in groups whose translations are exact binary fractions;
  - ≤ 1e-6 Å in trigonal and hexagonal groups through the live builder (the SYM_OPS floor);
  - ≤ 1e-3 Å on the float32 builder path, using the pattern of `tests/test_centroid_wrap.py`;
  - an improper-element control must exceed 0.1 Å.
- **Kabsch parameterise → build**, including symmetric tops.
- **Replaces an old check.** `normalizer_reduction/test_torch_transform.py` validated copies of the functions (its lines 1-8), not the live code; this gate replaces its 800/800 check.

**Gate 4. Energy invariance.**
- MXtalTools runs through `analyze(['elj'])`; the GFN boundary test runs through `MolecularCrystal`.
- **Tolerance:** relative ≤ 1e-3 and absolute ≤ 5e-3 raw. Controls must sit at ≥ 0.1 raw, at least 100× the tolerance.
- **Wrap first:** every build wraps the centroid to [0,1)³.
- **Re-check sg 88** with more rows: its measured 3.4e-4 relative error is near the tolerance.

**Gate 5. Jacobians.**
- Closed form against autograd and against central differences, ≤ 1e-8 in float64.
  - 500 interior points, plus points near the bounds and near chart singularities.
- Assert the lower-triangular structure, so that the derivatives of the conditional bounds stay off the diagonal.
- **Covers:** the R3 cell maps, the Euler-to-rotvec map and the full adapter.

**Gate 6. Formats** (slow).
- CIF write → read → parameterise → canonicalise returns the same chart point.
- spglib identifies the group at symprec 1e-3 for float32 and 1e-5 for float64; an ASE `Atoms` export is checked the same way.
- **The O7 direction** (answered by D1: import; export not chosen):
  - import: a P2₁/n file imports as the same crystal as its standard description. For `MolCrystalData` rows this is built and gated in `tests/fd/test_io_settings.py` (the standard setting of all 230 groups; each of the 85 operator-set classes found in the CSD file; origins off every grid; unreduced bases). A verifier sweep resolved all 530 cctbx Hall settings exactly, plain, with origin shifts and with unreduced bases [measured 2026-10-17; a script, not a gate]. The CIF path is not built.
  - export (not chosen): P2₁/n, I2/a and a non-reduced cell are identified correctly and literally equal after the inverse basis change.
- Whatever O7 leaves unbuilt stays refused, as `tests/test_cell_standardize.py:236` checks today.

**Gate 7. Budgets** (`tests/bench_fd.py`, like `bench_image_pairs.py`).
- **Pose canonicaliser:** ≥ 1e5 rows/s at |Γ⁺| ≤ 32 and ≥ 5e4 rows/s at 192. The rate at 384 is unmeasured.
- **Cell canonicaliser:** ≤ 15 min per 1.2e6 rows.
- **Chart forward + inverse + log-det:** ≤ 4 ms per 1,000 rows on CPU, the same as today's latent transform. The mk_dev batch is 1,000 rows (`mk_dev.yaml:354`).

*Table 7. Baselines for gate 7 [verif].*
- **Conditions:** CPU, 4 threads, shared with a running training job, so rates are indicative.
- **Prototype:** the pose canonicaliser is the closed-form one in `$FD/vs/`, run on 10⁶ rows.
- **Comparison:** the prototype against the existing functions it would replace.

| operation | sg 2 (\|Γ⁺\| = 8) (rows/s) | sg 14 (16) (rows/s) | sg 61 (32) (rows/s) | sg 227 (192) (rows/s) |
|---|---|---|---|---|
| closed-form pose canonicaliser, float32 | 5.2e6 | 3.2e6 | 1.7e5 | 7.9e4 |
| closed-form pose canonicaliser, float64 | 1.5e6 | 1.1e6 | 1.5e5 | — |
| existing `get_fd_params` (rebuilds every candidate) | — | 133 | — | — |
| existing `choose_basis` (cell) | 270 | 184 | — | — |

Today's `latent_transform` plus its inverse runs at 2.5e5 rows/s (sg 14 rows).

**Gate 8. Trainer bit-identity and measure contract.**
- `LatentV1Chart` reproduces the latent maps bit for bit, clamp included, and `tests/golden/test_latent_map.py` stays green.
- Its reference density equals −jacobian_energy/T plus a constant.
- `CellChart`'s image equals the margin-0 zero set of the `sym_utils` penalties.

**Gate 9. A short real run** (AGENTS.md verification rule 5). The adapter sits in the flow's training loop. Before writing tests that pin its behaviour, run it at production-shaped parameters in the AR flow on MIPCAS, then pin what the run showed.

## 5. Staged plan

Sizes are guesses and include tests.

**Stage 0: generic machinery, sg 1 and sg 2.** This is what the MIPCAS flow test needs.
- **Deliverables:**
  - the generator and checked-in tables for all 230 groups;
  - the description model;
  - the generic pose fold, unfold, membership and images;
  - the box centroid chart;
  - the Euler chart, with the measured map to the trainer's rotvec rows;
  - the triclinic `CellChart` and the fix to the triclinic tie-break;
  - `LatentV1Chart` and the measure functions;
  - the sg 2 adapter;
  - the read-only MIPCAS prior migration.
- **Gates:**
  - gates 0 and 1 (pose part) for all 230 groups, since the pose machinery is generic and cheap;
  - gates 1 (cell), 2, 3, 4, 5 and 8 for sg 1 and 2;
  - gate 7;
  - gate 9 once the flow uses the adapter.
- **Blockers:** the triclinic tie-break fix; O4 (the default keeps the stage unblocked); O3.
- **Size:** about 1.5–2k lines.

**Stage 1: sg 14 and the monoclinic cell.**
- **Deliverables:**
  - the monoclinic `CellChart` for its three classes;
  - migration and the collapse check for the ×8 sg 14 priors (acridine, NEHZOR);
  - the adapter route for domain points outside the trainer box, per O2;
  - the format layer (CIF and ASE). The O7 direction is D1 (import), already built for `MolCrystalData` rows (§6).
- **Gates:** 1–6 and 8 for sg 14 and 15.
- **Needs:** O1, O2 (O7 is answered by D1).
- **Size:** about 0.8k lines.

**Stage 2: the other 70 box groups** (monoclinic 3–15 and orthorhombic 16–74).
- **Deliverables:**
  - an orthorhombic permutation-class cell canonicaliser (built 2026-10-21) and charts; for Pbca, polar coordinates with a cone point are one option, a wedge whose two faces are glued by the 3-cycle (interval ends, as O1 (a) treats rotation-glued faces) is the other;
  - the free-axis projection with K, with the gauge consistent with `canonicalize_free_axes`.
- **Gate 4 coverage:** one group per Γ⁺ type, plus every group with a prior on disk (4, 9, 19, 29, 33, 61, 62).
- **Size:** about 1k lines.

**Stage 3: the 156 polytope groups.**
- **Deliverables:**
  - Fourier–Motzkin charts, already through gates 1, 2 and 5 in prototype for sg 88, 92 and 227;
  - the wrap-before-build contract enforced in `to_mol_crystal_data`;
  - trigonal and cubic cell domains: checked 2026-10-21, no setting redundancy of the cell (only the exact-metric snap, built).
- **Limits:** corrected 2026-10-21: the trainer-box image count is defined in 216 groups. The 107 groups with no box entry fall back to the whole cell (multiplicity |Γ⁺|, 12 to 384), and of the 12 boxes listed as not tiling, 6 (48, 50, 59, 68, 86, 141) have full coverage with a constant image count; the other 6, with 88, 109 and 110, are the partial-coverage groups (O2). The real blockers are the cell chart, its Jacobian, the trainer's averaged length rows and its length clamp.

**Stage 4: Z′ > 1** (sg 9 and sg 14 at Z′ = 2).
- **Deliverables:** the §2.8 construction, the global free-axis gauge, and the `zp_doubling` round trip.
- **Gates:** gate 1 on the combined orbits.
- **Needs:** O6.

**Stage 5: the rest.**
- export to nonstandard settings, the O7 direction D1 did not choose;
- special metrics on cell faces;
- refusing molecules on special positions;
- an opt-in fold through molecular point symmetry (never on by default);
- vectorising the cell canonicaliser;
- retiring `get_fd_params` and `fold_to_fd`.

## 6. Owner decisions

**Positions already taken as settled (2026-10-15 discussion):**
- Work on the +1 sheet.
- No fold through molecular point symmetry by default.
- Cover the other-chirality-equivalent region with one continuous chart, not a token. ASU(Γ⁺) does this automatically.
- Hard walls only where real walls exist; circles where faces are glued by a pure translation.
- A canonicaliser is acceptable when the domain is defined with respect to the unit cell by a table.

**Owner answers of 2026-10-16 to D1–D7.** D1–D7 are the decisions put in the inspection report (`artifacts/fd_design_oct15/inspection/report.md` §6). In this note, "D1"–"D7" with a reference to §6 name these decisions, not the dihedral point groups D2, D3 and D4 used elsewhere. Each entry gives the answer, its scope, and what was built under it. Build and verification evidence: `artifacts/fd_design_oct15/canonical_data/report.md`.
- **D1 (= O7). Nonstandard settings:** the CSD importer re-expresses them into the standard setting, one way.
  - Scope: import only; nothing writes a nonstandard setting.
  - Built: `crystal_fd/settings.py` and `io_mxt.from_mol_crystal_data(nonstandard='reexpress')` (2026-10-16; search widened 2026-10-17).
  - On `new_prot_csd.pt`, 64,024 of the 64,294 rows filed in other settings now import; 264 stay refused under D3 and 6 as `setting_metric_mismatch`. Every re-expressed row matches its raw CIF within 3.44e-3 Å (all 64,024 checked) [measured].
- **D2. Z′ = 1 Sohncke rows off the +1 sheet** are re-described with their own enantiomer as the reference molecule, X′ = diag(−1,1,1)·X at h = +1, so the stored molecule of such a row is the mirror of the aligned `pos`.
  - Built: `io_mxt` `mirror_reference=True`, `ImportInfo.mirrored` and `reference_handedness`; 39,105 CSD rows, 49.5% of imported Sohncke rows.
  - Consumers must pass `reference_handedness` as `target_handedness` (§3.3). GFN's frame step does not yet.
  - **Superseded 2026-10-21 by the owner for the canonical CSD set:** such rows are stored as their mirror-image crystal (enantiomorph): reference X, h = +1, no `reference_handedness` flag; a column records `enantiomorph` and the source space group. In the 11 enantiomorphic pairs (P4₁/P4₃, P3₁/P3₂, …) the space group changes to the partner (1,668 of the 41,162 rows in `csd_fd_v2`). The owner's rule that goes with it: none of the energies may be chirality-sensitive, and a crystal and its mirror image scoring differently is a defect. Measured 2026-10-21 (`artifacts/fd_design_oct15/mirror_check_oct21/`): eLJ and the three MACE models are mirror-invariant to float32 noise; the UMA checkpoints (`esen_s.pt`, `uma-s-1p1`, `uma-m-1p1`) are not, by a median of about 0.2 and up to 2 kJ/mol per molecule on lattice energies. Built 2026-10-21: `crystal_fd/enantiomorph.py` (inversion through the origin, written in the standard setting: t → −t − p with p ≠ 0 only in sg 80, 98, 210; R → R·diag(1,−1,−1); the partner group in the 11 pairs) and `io_mxt.from_mol_crystal_data(off_sheet='enantiomorph')`; the wrapper takes `off_sheet='enantiomorph'`. The D2 mode stays available and is still the importer's default.
- **D3. Z′ > 1 stays refused,** for the FD work.
  - A featurizer update that keeps per-unit conformers is owed soon; its scoping is in `canonical_data/featurizer/`.
  - CSD: 2,350 Z′ > 1 input rows, 2,347 refused as `z_prime_gt1` and 3 under a refusal code checked earlier.
- **D4. `FACE_TOL` stays 1e-5** (§3.4).
- **D5. Withdrawn.** The owner keeps the `latent_v1_lebesgue` cell measure, so monoclinic `metric_invariant` stays unimplemented (§3.6). This follows the owner's answer to O4 (2026-10-15: (a)).
- **D6. The sg 2 priors and the CSD become canonical one-row-per-crystal datasets.** This is the critical path to modelling.
  - sg 2 priors: `D:/crystal_datasets/conditional/priors/<stem>_fd1.pt`, with `_fd1_provenance.pt` and `_fd1_manifest.json`, for niggli_v2, pooled_oct05, qm9full_prior and qm9full_test_prior (§3.11, "As built"). One row per crystal at a 1e-6 chart tolerance.
  - CSD: `D:/crystal_datasets/csd_fd_v1b/` (339,097 rows, schema `crystal_fd.dataset/1`). One row per imported Z′ = 1 CSD entry, not per crystal: repeated entries and orthorhombic axis orders are flagged, not merged.
  - Superseded 2026-10-21 by `D:/crystal_datasets/csd_fd_v3/` (355,637 rows): built from the new ingestion chain over the whole CSD dump (`crystal_ingestion.md` §4), with near-face rows nudged and off-sheet Sohncke rows stored as their enantiomorph (D2 above). Still one row per entry.
  - Superseded again the same day by `D:/crystal_datasets/csd_fd_v4/` (the same 355,637 rows): the cell step now covers sg 16–230 (orthorhombic axis order, exact-metric cells above), so every row is canonical (355,246) or a special metric (391); sg 1–15 rows are bit-identical to v3; 15,002 orthorhombic rows changed axis order, and the 1,986 v3 pairs of one refcode family that matched only after an allowed axis order now match in stored order (one same-order pair split across Pbca's seam; `post_v4.json`).
  - Superseded 2026-10-22 by `D:/crystal_datasets/csd_fd_v5/` (394,934 rows), built the same way from a full re-ingest (`D:/crystal_datasets/ingest_csd_v1_r2`) that recovers rows the features step used to refuse and reads rhombohedral-axes CIFs (`crystal_ingestion.md` §4): the 355,637 v4 rows are all kept (bit-identical except 18 Pc/Cc rows whose R differs by at most 5e-16, rounding that depends on the batch), and 39,297 rows are new, 39,236 of them flagged `features_fallback` (`post_v5.json`).
- **D7. qm9full rows that the trainer's latent box clamps are refused with a counted reason in the rebuilt data,** for now. The owner wants to revisit how clamped faces are treated later.
  - Built: 6,223 rows refused in qm9full_prior and 330 in qm9full_test_prior. The qm9full conditions files are not covered.

**Owner answers of 2026-10-21 for space groups 16–230.** Asked after the higher-group survey (`artifacts/fd_design_oct15/higher_sg_oct21/`).
- **The parameterization is designed for efficiency, not for today's trainer.** "We will change the trainer so it makes more physical sense. Averaging is probably dumb." So from tetragonal up the cell coordinates are one per free cell parameter (log a and log c; log a alone in cubic; R groups on hexagonal axes as hexagonal), and the reference measure is flat in each free log length, as in every lower family. The trainer's averaged length rows are to be replaced in the trainer, not mirrored here.
- **Molecules on special positions (Z′ < 1) are skipped.** Their CIFs keep their ingestion status.
- **Multi-component crystals (cocrystals, salts, solvates) are skipped for now.** They need the Z′ > 1 representation rebuilt as a set of units.

**Owner answers of 2026-10-22 on Z′ > 1.** Asked after the Z′ > 1 survey (`artifacts/fd_design_oct15/zprime_oct21/`).
- **Conformations will be sampled together with the packing,** in internal coordinates, so each independent molecule keeps its own conformation, and Z′ > 1 and multi-component crystals share one "set of units" machinery. Nothing may order units by a continuous key (it would be a wall in conformer space).
- **Re-describing a crystal in a higher-symmetry group touches special positions, where "all the rules change": be careful.** `find_symmetry` results are recorded, not applied, until special positions are designed.
- **Internal-coordinate assignment during ingestion, at least optionally,** is wanted for joint crystal and conformer sampling.

**O1. Which quotient and which cut define the canonical domain?** This is problem-statement Q2 and Q3. **Recommended: (a).**
- **(a) Full Γ⁺ quotient, cut in the centroid:** ASU(Γ⁺) × SO(3), with K cut in SO(3) for sg 6–9.
  - One SO(3) chart for every group.
  - The per-group data is an inequality list plus gluings.
  - Twisted gluings become interval ends in the centroid, for example sg 14's z ∈ [0, ¼].
  - It has the strongest evidence: gates 1, 2 and 5 were prototyped, including on polytopes.
- **(b) Full Γ⁺ quotient, cut in SO(3)** where P⁺ is cyclic or dihedral, with cut A for the 36 T and O groups.
  - Every centroid row is a circle (in a sheared basis in some tetragonal groups), and sg 14 has no twisted seam.
  - Dihedral groups keep a twisted face at β = π/2.
  - It forces O3 = Euler.
  - The D2 Euler box (59 groups) is untested, and it means two code paths.
- **(c) Quotient by G⁺ only** (your doubled boxes, generalised): no normaliser fold.
  - Every face is a seam.
  - The [Γ⁺:G⁺] exact images stay as multiplicity (8 in sg 14 and in sg 2), so the flow learns k copies of every mode, and log k offsets log Z.
- **Default if unanswered:** (a). Stage 0 does not depend on this answer: in sg 1 and 2, (a) and (b) coincide.

**O2. In the eight half-coverage groups (8, 9, 11, 12, 14, 88, 109, 110), how does the adapter handle domain points that have no image in the trainer box?** This is needed from stage 1, for sg 14. **Recommended: (a).**
- **(a) Leave the trainer box as it is.** Score those points by building from the description through `prebuilt_sample_to_reward` (GFN `energies/molecular_crystal.py:1308`), and report the flow's mass outside the box.
  - No trainer change.
  - The GFN trainer keeps its coverage gap, so in these groups the flow and GFN sample different domains whenever a molecule is not exactly achiral.
  - This scoring path's measure terms (`:1183-1206`) need gate 5 [Inf.].
- **(b) Double the trainer box to ASU(G⁺)** (your y-doubling).
  - The trainer then reaches every crystal.
  - The trainer latent changes in 8 groups: sg 14 and sg 9 checkpoints and priors become incompatible.
  - The doubled boxes for sg 109 and 110 still hold 4 G⁺ images.
- **(c) Restrict the flow to the half the trainer can represent.**
  - No new scoring path.
  - Exact only for exactly achiral molecules: NEHZOR and acridine (planar to about 0.02 Å) lose their near-image crystals.
- **Default if unanswered:** (a).

**O3. Which SO(3) chart does the flow use?** This is problem-statement Q4. **Recommendation withdrawn; see the owner's answer below.**
- **(a) Euler ZYZ (α, cos β, γ):** circle × interval × circle.
  - Haar measure is uniform, and the gimbal lines have measure zero.
  - The adapter to the trainer's rotvec rows is measured, to 2.3e-9 nats.
  - O1(b) requires this choice.
- **(b) The trainer's rotvec hemisphere (θ, φ, r):** no orientation adapter.
  - The flow has to learn the density sin²(r/2) sin θ.
  - The equator is a twisted interval end, and the pole and r = 0 are singular.
  - The −0.99 floors block exact round trips.
- **Default if unanswered:** (a). The SDE trainer keeps rotvec either way.
- **Owner answer 2026-10-15: deferred, and not (a).** The owner judged Euler angles a poor parameterisation for this model and is open to a native SO(3) token, which is where the rotation vector with its Jacobian was heading. Until then the package stores rotation matrices and offers rotvec, quaternion, Euler and axis-angle conversions with their Haar densities, and sets no default chart.

**O4. Which base measure on the cell?** This is problem-statement Q5b. **Recommended: (a).**
- **(a) Chart Lebesgue on today's latent (d log L dθ),** which is what the trainer targets now.
  - The adapter targets exactly the trainer's hard-wall target, so equal-energy-call comparisons stay like for like.
  - It is a modelling choice, not an invariant measure: relative to dG it weights lattices by 8(abc)⁴∏sin θ.
- **(b) The invariant measure dG/V** (equal to d⁹h with frame rotations quotiented).
  - The target changes for every run that adopts it, and `compute_jacobian` needs a new term.
  - log Z is then not comparable with existing runs.
- **Default if unanswered:** (a). `cell_measure` is a required argument, so (b) stays a one-argument switch.
- **Owner answer 2026-10-15: (a).** Today's measure is kept because it is defensible and covers all known crystals; no better cell measure is known. `cell_measure` stays an explicit required argument.

**O5. Where are the tables generated?** **Recommended: (a).**
- **(a) With cctbx, offline,** into checked-in JSON plus manifest hashes; runtime stays cctbx-free.
  - This matches `generate_normalizers.py` today.
  - A `cctbx` test marker re-derives the tables.
- **(b) With cctbx at runtime.**
  - No generated file.
  - cctbx becomes a hard dependency for every MXtalTools user and cluster environment. Cluster availability is unchecked.
- **(c) Without cctbx:** Dirichlet domains computed in the package. These passed in all 230 groups, but they have 8–24 oblique facets instead of 5–7 axis-aligned ones, and sg 14 is no longer a box.
- **Default if unanswered:** (a).

**O6. At Z′ = 2, quotient the unit swap or keep labels?** This is problem-statement Q6 and gates stage 4 only. **Recommended: (a).**
- **(a) Keep labels.**
  - An exact multiplicity of 2, which offsets log Z by log 2.
  - No ordering surface; the flow carries both labellings.
- **(b) Order units by h(R₁) < h(R₂).**
  - Halves the domain.
  - One AR bound for cyclic P⁺ (sg 9 and 14, the groups in use); a union of two intervals for dihedral P⁺.
  - Measured only for point groups C1, C2 and D2; no invariant exists yet for T or O.
- **Default if unanswered:** (a).

**O7. "Nonstandard cells, at least one-way": which way?** **Recommended: (a).**
- **Answered 2026-10-16 by D1: (b), one-way import.** As built it differs from (b) as written below: it does not use spglib, and det P = 1/2 is handled (§3.8).
- **(a) Export.**
  - Description → CIF or ASE in a named nonstandard setting (P2₁/n, I2/a, a non-reduced cell) through an explicit (P, p), verified by spglib and by literal atom equality.
  - Imports of nonstandard files stay refused.
- **(b) Import.**
  - Nonstandard file → standard description through spglib, for |det P| = 1 only.
  - This is what parameterising the 19.2% of CSD entries filed in other settings needs.
  - Nothing ever writes a nonstandard setting.
- **(c) Both.** Roughly the union of the work in (a) and (b).
- **Default if unanswered:** (a), the literal reading of the request.

## 7. Open risks and gaps

**Evidence.**
- **Narrow and partial.**
  - eLJ invariance is measured in 7 groups here and 11 in R2, all on synthetic molecules of 3–6 atoms.
  - Real rows have been canonicalised for the sg 2 priors and the Z′ = 1 CSD (inspection 2026-10-16; builds under §6 D6). The sg 14 search priors (NEHZOR, acridine) have not been.
  - Monte Carlo counts are evidence, not proof. The derivations are outlines, not independently reviewed.
- **Not durable.** Every number above comes from scratchpad scripts outside the repositories. Gates 0–5 are the durable form.
- **Cell composition.**
  - Tested only for sg 2 and 14 [math] and for sg 2, 14, 15 and 4 [verif].
  - The product claim for the full domain rests on the stabiliser argument (§2.6).
  - Only the β = 90° triclinic face was probed for the tie defect; the γ = 90°, pairwise and three-vector faces were not.
- **Untested pieces:**
  - the D2 Euler box;
  - the Rodrigues zone's Haar density for T and O;
  - Z′ = 2 for P⁺ other than C1, C2 and D2;
  - Z′ > 2;
  - the extra multiplicity for achiral molecules;
  - the cctbx plane-group ASUs for free-axis groups (only Dirichlet domains were used there);
  - cctbx's tie-break cuts, used only through cctbx's own `is_inside` and not yet parsed into a decision tree;
  - the Fourier–Motzkin bounds for free-axis groups.

**Known defects the plan has to work around** [code, measured]:
- the builder clip (`crystal_ops.py:114-116`), handled by wrapping before every build;
- the triclinic tie-break (`standardize.py:255`), a stage-0 fix;
- the 8-decimal SYM_OPS translations (`space_group_info.py:40-43`), handled by the exact table and the 1e-6 Å floor on the live builder path.

**Trainer coverage.**
- 107 groups have no box and fall back to [1,1,1] (`asymmetric_units.py:172-175`), and nothing guards against using them.
- 12 boxes do not tile.
- 8 valid boxes cover half of a chiral molecule's crystals.

The adapter cannot serve those groups until the owner fixes the boxes. Each fix changes the trainer latent.

**Soft walls.**
- log Z_trainer = log Z_FD + log k holds only with hard walls.
- At reduction coefficient 10, the soft walls give 4.8–22 effective copies per triclinic lattice and leak 21.9% of a flat row's mass [R1, R3].
- Comparing the flow with the soft-walled GFN therefore depends on the open wall-stiffness decision in the exact-density plan (from memory, not rechecked).

**Budget at |Γ⁺| = 384** (sg 196, 216, 219): unmeasured.

**Repository state.**
- `crystal_building/image_pairs.py` is untracked. The MXtalTools working tree has uncommitted changes (101 status entries on 2026-10-15), so the cited line numbers will drift.

**Trainer changes this tooling motivates, not proposed here.** Each is an owner decision:
- wrap the sg 2 x-row with period ½;
- treat the θ equator as a seam;
- remove the floor on the r row (it causes a 0.79 raw eLJ error, R1);
- double the sg 14 y-box (= O2(b));
- add a guard against groups with no box;
- turn the builder clip into a wrap;
- repair the ASYM_UNITS origins and extents.

**Documents to update when the code lands:**
- the wiki pages asymmetric-unit-reduction (the "39 space groups" section and the "one plus the list length" count), cell-fundamental-domains, and latent-dimension-structure (periodic rows), each in the same change;
- `fundamental_domain.md` §1–2, to be marked superseded in part, citing §2.10 item 10;
- the claim in `generate_asymmetric_unit_reference.py:26`.

**Checks made while merging this note** [code, measured]:
- callers of `normalizer_images`, `get_fd_params` and `fold_to_fd`, by grep over both trees (excluding worktrees);
- the clip at `crystal_ops.py:114-116` and its call at `crystal_building/utils.py:996`;
- the tie-break at `standardize.py:241-260` and `PENALTY_ZERO` at `:50`;
- the 8-decimal note at `space_group_info.py:40-43`;
- the RAW_ASYM_UNITS entries for sg 2 (½, 1, 1) and sg 14 (1, ¼, 1);
- the cited function definitions in `crystal_ops.py`, `standardize.py` and `utils.py`;
- the GFN measure terms at `molecular_crystal.py:1202-1206`;
- the test markers at `pyproject.toml:101-106`;
- the nonstandard-setting refusal test at `tests/test_cell_standardize.py:236`;
- from `$FD/t8_sweep.log` and `$FD/t4_so3.py`: the maximum |Γ⁺| of 384, K nontrivial only in sg 6–9, both cuts at one orbit point in all 230 groups, 68 free-axis groups, 43 of 65 Sohncke groups with improper normaliser elements, and the D2-labelled-as-T classifier quirk.