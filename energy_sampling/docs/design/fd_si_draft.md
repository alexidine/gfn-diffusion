# The fundamental-domain (FD) chart for Z′ = 1 rigid-molecule crystals: Supporting Information draft

- **Status:** DRAFT, 2026-11-02, documentation only. It is not adopted and has no authority over code, config or
  tests. Nothing here is committed without the owner's yes.
- **Verified against:** MXtalTools `master` at `adb86928` (`mxtaltools/crystal_fd/` clean) and GFN `conditional`
  at `2f17b75d`. One file differs in the GFN working tree: `energies/fd_coordinates.py` carries an uncommitted
  length-box change from another session (§1.6, §5.3).
- **Purpose:** an explanation of the FD chart that a referee can check, and the basis for SI text on the chart, the
  reduction, the metric, chirality handling, and what a generative model gains from the reduced basis.

**How each statement is labelled.**

| label | meaning | what backs it |
|---|---|---|
| **[proven]** | a proof, or an exact (rational-arithmetic) certificate | the proof is cited, or given here |
| **[tested]** | a sampled or floating-point test passes; this is evidence, not proof | the test is named |
| **[derived]** | follows from the definitions by the steps shown here | the steps |
| **[convention]** | a choice; another choice would also be valid | who chose it, and when |
| **[measured]** | an observation on data; evidence, not policy | inputs and counts |
| **[open]** | not established | what is missing |

Code paths are relative to the MXtalTools root (`mxtaltools/crystal_fd/...`) or to GFN's `energy_sampling/`.

---

## 0. The construction on one page

A Z′ = 1 crystal of a rigid molecule X in space group G is described by its cell, a fractional centroid t, a proper
rotation R and a handedness h = ±1. The molecule sits at

$$x \;=\; T\,t \;+\; R\,\mathrm{diag}(h,1,1)\,X,$$

where T is the fractional-to-Cartesian matrix of the cell and X is centred on its heavy-atom centroid
(`description.py::CrystalDescription`). Many descriptions give the same crystal. The FD keeps exactly one of them:

$$\boxed{\;\mathrm{FD}(G) \;=\; D_{\rm cell}(G)\;\times\;\mathrm{ASU}(\Gamma^+_G/K_G)\;\times\;SO(3)/K_G,\qquad h=+1\;}$$

Every factor is built from objects tabulated in *International Tables for Crystallography* (ITA) Vol. A, plus a few
rules:

| symbol | what it is | where a referee finds it | how it is built here |
|---|---|---|---|
| G | the space group's operators in the ITA standard setting | ITA Vol. A space-group tables (the default setting of each number) | `SYM_OPS[sg]`, which equals cctbx's `space_group_info(number=sg)` in all 230 groups (§2.1) |
| S(G) | the *setting group*: integer basis changes of det +1 that map G's operator set to itself up to an origin shift | derived from G | per lattice family (§1.2) |
| D_cell(G) | the reduced-cell domain of S(G) on the free cell parameters of G's lattice family | Niggli (triclinic); a Γ₀(2) domain (monoclinic); axis-order rules (orthorhombic); the family metric above that | §1.2 |
| Γ_G | the Euclidean normaliser N_E(G), modulo lattice translations and continuous shifts along free (polar) axes | ITA Vol. A (2016), §3.5 (Koch, Fischer & Müller, Euclidean and affine normalisers) | brute force in exact rationals, checked against cctbx (§1.3) |
| Γ⁺_G | the proper (det +1) part of Γ_G | from Γ_G | §1.3 |
| K_G | the elements of Γ⁺_G that act trivially on the centroid once the free axes are projected out; nontrivial only in sg 6–9 | from Γ⁺_G | §1.4 |
| ASU(Γ⁺/K) | an asymmetric unit of Γ⁺_G/K_G on the non-free centroid coordinates, with ITA's tie-break rules on its faces | in the 162 non-polar groups Γ⁺_G is itself a Sohncke space-group type (cctbx's expansion of Γ⁺ adds no element, 162 of 162: `fd_tooling.md` §2.2), so this is that type's ITA asymmetric unit written in G's basis (Table 2) | cctbx `direct_space_asu()` of Γ⁺ (§1.3) |

The rules are:

1. **Cell.** Reduce the cell under S(G) into D_cell(G).
2. **Sheet.** Map h = −1 to h = +1 through an improper element of G, where G has one (§4).
3. **Pose.** Fold (t, R) under Γ⁺ into ASU(Γ⁺/K) × SO(3). Elements act on poses as
   (t, R) ↦ (W t + w, Q_W R), with Q_W = T W T⁻¹. Free axes are fixed to a gauge value. In sg 6–9, R is cut to the
   half of SO(3) with R₀₀ ≥ 0.
4. **Faces.** On faces, ITA's tie-break cuts choose the centroid. At special positions of Γ⁺, the lexicographically
   largest Q R chooses the orientation.

Sections 1–5 justify each factor and rule. Section 6 lists what a referee would flag.

---

## 1. The FD for one space group

### 1.1 What counts as "the same crystal"

These operations change a description but not the crystal [derived]. Each one preserves the set of atom positions,
up to a rigid motion of space.

- **(a) Lattice translations.** t ↦ t + n, with n ∈ ℤ³.
- **(b) Elements of G.** t ↦ W t + w and R ↦ Q_W R, for (W, w) ∈ G. These map the crystal onto itself.
- **(c) Elements of N_E(G).** Same action. A normaliser element maps G's symmetry set onto itself, so it re-describes
  the crystal in the same setting, rather than mapping it onto itself. (b) is the special case G ⊂ N_E(G).
- **(d) Basis changes.** A basis change N in S(G) acts on the cell (Gram matrix G ↦ NᵀGN), on the centroid and on
  the frame.

Two notes on this list:

- A det −1 basis change gives the same cell parameters as its negative, which has det +1. Basis changes are therefore
  restricted to det +1 [derived]. This is done in the code: `ops._neighbourhood` is det +1, and `settings.find_setting`
  requires det P > 0.
- An improper element of N_E(G) maps h to −h. Its handling is the sheet rule (§4).

The FD is a set that meets every orbit of these operations exactly once, up to a set of measure zero.

### 1.2 The cell part: D_cell(G), one domain per setting group

*Table 1. The cell domain by lattice family, as the code defines it.* "Cell coords" counts the chart's free cell
coordinates, which equals the family's free cell parameters. Day's groups are in brackets. Proof status is the
strongest evidence that exists today.

| family (Day sg) | setting group S(G) | D_cell(G): reduced domain | chart coordinates (count) | proof status |
|---|---|---|---|---|
| triclinic (1, 2) | all of GL(3, ℤ), det +1 (the operators {E} and {E, −1} survive every basis change) | Niggli main conditions in a **shifted sign convention**: a ≤ b ≤ c; \|ξ\| ≤ B, \|η\| ≤ A, \|ζ\| ≤ A; η ≤ 0 and ζ ≤ 0 (β and γ obtuse, α free); ξ + η + ζ ≥ −(A + B) (that is, \|a+b+c\| ≥ c). Here A = a², B = b², ξ = 2b·c, η = 2a·c, ζ = 2a·b | log c, b/c, a/b, s_β, s_γ, v (6) | **argued + [tested]**: classical Niggli / Křivý–Gruber theory carries over through the det +1 bijection (a, −b, −c), which maps an all-acute Niggli cell into this domain and keeps P1/P-1 [derived, wiki `cell-fundamental-domains.md`]. Uniqueness on faces rests on a Gram-entry tie-break that is tested (gate 1, off, near and on every face) but not proven |
| monoclinic, b-unique, cell choice 1 (4, 5, 7, 9, 13, 14, 15) | changes in the ac plane, a′ = pa + qc and c′ = ra + sc, with b′ = (ps − qr) b. Classes (`MONO_CLASS`): GLIDE (7, 13, 14; r even), CENTRED (5, 8, 9, 12, 15; q even), FREE (3, 4, 6, 10, 11; any) | α = γ = 90° and −m ≤ cos β ≤ 0, where m = min(a/c, c/2a) for GLIDE, min(c/a, a/2c) for CENTRED, and a/2c for FREE, which also requires a ≤ c | log c, log a, log b, s ∈ [0, 1] with cos β = −s·m (4) | **derivation sketch + [tested]**: with τ = (a/c)e^{iβ} the GLIDE changes act as Γ₀(2) on the upper half plane, whose fundamental domain gives the three walls (wiki `cell-fundamental-domains.md`). Per-class face identifications are not written out as a proof. `test_monoclinic_cell.py` reaches one canonical cell and pose from every allowed basis, in the interior and at 1e-7 to 1e-2 on both sides of every face |
| orthorhombic (18, 19, 20, 29, 33, 43, 56, 60, 61) | the axis permutations G allows (signed, det +1, with an origin shift); sign flips leave the cell unchanged | the cell snapped to the exact orthorhombic metric, then the lexicographically smallest (a, b, c) among the allowed orders. Classes: all six orders (11 groups; Day 19): a ≤ b ≤ c. a ↔ b (26 groups; Day 18, 20, 43, 56): a ≤ b. Pbca (61), cyclic: a ≤ b and a ≤ c. One order (21 groups; Day 29, 33, 60): no condition | log a, log b, log c (3) | **[proven] (trivial)**: sorting picks one element of a finite orbit. The classes are derived from the tables at first use and checked against the 2026-10-21 survey (`axis_order.py`), otherwise a `RuntimeError` is raised. `test_higher_sg_cell.py` returns every allowed order of a random crystal to one row within 1e-12 |
| tetragonal, trigonal (on hexagonal axes), hexagonal (76, 86, 88, 96, 145, 148, 154, 169) | every setting change that preserves G is an isometry of the generic metric, i.e. a linear part of Γ | the cell snapped to the exact family metric; no further condition | log a, log c (2) | **[tested]**: survey of integer changes with entries up to 2 (`canon.py` docstring); `test_higher_sg_cell.py` returns every Γ image of a random crystal to one row |
| cubic (none in Day) | same | the snap only | log a (1) | as above |

Notes:

- **Why the cell can be chosen before the pose** [derived, with exceptions]. Once the cell is in D_cell(G), the
  remaining freedom is a basis change N ∈ S(G) that leaves the reduced cell unchanged. That is, N is an automorphism
  of the metric (NᵀGN = G) inside S(G).
  - For a *generic* metric of the family, those automorphisms are exactly the linear parts of Γ⁺. They act on the
    pose alone, and the fold absorbs them. In the monoclinic case, a cell's two bases (N and N times the 2-fold about
    b) differ by an element of Γ⁺ (README "What it does", item 1).
  - The exceptions are **special metrics**: lattices with extra automorphisms, such as a triclinic cell with
    β = γ = 90° exactly, a monoclinic cell on a domain face, or an orthorhombic a = b on a mirror face. They are
    flagged (`n_auto > 1`, status `special_metric`) and not resolved. On them, the canonical pose can depend on the
    input basis [tested: `test_monoclinic_cell.py`, `test_higher_sg_cell.py`; README "Known limits"]. They have
    measure zero; in Day's data they are 86 of 29,947 structures (§2.3) [measured].
- **The angles are snapped, not sampled.** In every family above triclinic, the fixed angles (90°, or 120° for γ in
  the hexagonal family) and the tied lengths (b = a, c = a) are set exactly. They have no coordinate and no measure
  factor (`charts.py` docstring).

### 1.3 The position part: the centroid in ASU(Γ⁺/K)

The normaliser does two things that the space group alone does not. It moves the origin among equivalent points, and
it re-describes the crystal by an affine map that preserves G's symmetry set. Its proper part Γ⁺ moves (t, R) without
changing h. Only Γ⁺ is used on the +1 sheet. The improper part acts on the sheet (§4).

**Construction of the table** (`tables.py`, `data/fd_tables_v1.json`). The table is generated offline with cctbx and
checked by gate 0 in exact rationals. Gate 0 checks:

- closure;
- Γ⁺ against a brute-force N_E over integer matrices that preserve a generic metric;
- K;
- the ASU volumes;
- every facet gluing;
- the AR bounds.

Under the `cctbx` marker it also checks that Γ, both ASUs and every tie-break tree equal cctbx's. These checks were
re-run for this draft: `test_gamma_equals_cctbx`, `test_asus_equal_cctbx` and `test_gamma_equals_brute_force`, all 230
groups, 690 passed [tested].

**Theorem (centroid tiling)** [proven, given two premises]. `certify.py` checks the following in exact rational
arithmetic, for every group:

- **(G)** The representatives of Γ⁺/K form a group modulo ℤ^d. This covers all n² products.
- **(i)** vol(P) · |Γ⁺/K| = 1.
- **(iii)** Every facet of P is covered exactly, without overlap, by its face-to-face contacts with the images of P.
- **(W)** Free axes are height-1 walls.

From these, the module docstring proves that P = ASU(Γ⁺/K) tiles the non-free centroid space. The steps are:

- *Covering:* local finiteness, plus facet coverage across codimension-1 faces, plus connectedness of the complement
  of the codimension-2 skeleton.
- *Disjoint interiors:* the unit cell's volume equals Σ vol(k·P) = 1, and the covering multiplicity is ≥ 1.

The certificate passes in all 230 groups. This was re-run for the 26 Day groups for this draft: 26 of 26 pass, in
0.9 s.

The two premises the certificate takes as given:

- the table's element list is Γ⁺/K of N_E(G) [tested exhaustively: gate 0, brute force, all 230 groups];
- the cut trees on faces keep exactly one point of each boundary orbit [tested, not proven: equality with cctbx;
  gate 1 on exact special points and on points 3e-10 to 5e-8 off them].

*Table 2. The pose part of the FD for Day's 26 space groups.*

- d is the number of non-free centroid coordinates.
- |Γ⁺| is counted modulo the lattice and free shifts.
- "Γ⁺ as a space-group type" is cctbx's identification of the group generated by the table's Γ⁺ elements, computed
  for this draft. It has not been cross-checked against the normaliser tables of ITA Vol. A §3.5.
- Facet gluing kinds count the 5–7 facets of ASU(Γ⁺) (`GroupTable.facets`). *Translation*: glued to the opposite
  facet by a pure translation. *Self-twisted*: glued to itself by a rotation. *Pair-twisted*: glued to another facet
  by a rotation. *Free*: a free-axis wall (gauged).
- "Circle rows" are the coordinates GFN wraps (from the 2026-11-02 FD audit, Table 1); φ is the orientation azimuth.
- Compare |Γ⁺/K| with the "FD points per crystal" column of Table 6.

| sg (symbol) | free axes | d (coords) | \|Γ⁺\| (elements) | \|K\| (elements) | \|Γ⁺/K\| (elements) | Γ⁺ as a space-group type (ITA No., basis) | facets: translation / self-twisted / pair-twisted / free (count) | circle rows |
|---|---|---|---|---|---|---|---|---|
| 1 (P1) | x, y, z | 0 | 1 | 1 | 1 | 1 P1 | 0 / 0 / 0 / 6 | φ |
| 2 (P-1) | – | 3 | 8 | 1 | 8 | 1 P1 (2a, 2b, 2c) | 6 / 0 / 0 / 0 | t_x, t_y, t_z, φ |
| 4 (P2₁) | y | 2 | 8 | 1 | 8 | 3 P2 (2a, b, 2c) | 2 / 2 / 0 / 2 | t_x, φ |
| 5 (C2) | y | 2 | 8 | 1 | 8 | 3 P2 (2a, b, 2c) | 2 / 2 / 0 / 2 | t_x, φ |
| 7 (Pc) | x, z | 1 | 4 | 2 | 2 | 3 P2 (a, 2b, c) | 2 / 0 / 0 / 4 | t_y, φ |
| 9 (Cc) | x, z | 1 | 4 | 2 | 2 | 3 P2 (a, 2b, c) | 2 / 0 / 0 / 4 | t_y, φ |
| 13 (P2/c) | – | 3 | 16 | 1 | 16 | 3 P2 (2a, 2b, 2c) | 4 / 2 / 0 / 0 | t_x, t_y, φ |
| 14 (P2₁/c) | – | 3 | 16 | 1 | 16 | 3 P2 (2a, 2b, 2c) | 4 / 2 / 0 / 0 | t_x, t_y, φ |
| 15 (C2/c) | – | 3 | 16 | 1 | 16 | 3 P2 (2a, 2b, 2c) | 4 / 2 / 0 / 0 | t_x, t_y, φ |
| 18 (P2₁2₁2) | – | 3 | 32 | 1 | 32 | 16 P222 (2a, 2b, 2c) | 2 / 4 / 0 / 0 | t_z, φ |
| 19 (P2₁2₁2₁) | – | 3 | 32 | 1 | 32 | 16 P222 (2a, 2b, 2c) | 2 / 4 / 0 / 0 | t_z, φ |
| 20 (C222₁) | – | 3 | 32 | 1 | 32 | 16 P222 (2a, 2b, 2c) | 2 / 4 / 0 / 0 | t_z, φ |
| 29 (Pca2₁) | z | 2 | 16 | 1 | 16 | 16 P222 (2a, 2b, c) | 0 / 4 / 0 / 2 | φ |
| 33 (Pna2₁) | z | 2 | 16 | 1 | 16 | 16 P222 (2a, 2b, c) | 0 / 4 / 0 / 2 | φ |
| 43 (Fdd2) | z | 2 | 16 | 1 | 16 | 16 P222 (2a, 2b, c) | 0 / 4 / 0 / 2 | φ |
| 56 (Pccn) | – | 3 | 32 | 1 | 32 | 16 P222 (2a, 2b, 2c) | 2 / 4 / 0 / 0 | t_z, φ |
| 60 (Pbcn) | – | 3 | 32 | 1 | 32 | 16 P222 (2a, 2b, 2c) | 2 / 4 / 0 / 0 | t_z, φ |
| 61 (Pbca) | – | 3 | 32 | 1 | 32 | 16 P222 (2a, 2b, 2c) | 2 / 4 / 0 / 0 | t_z, φ |
| 76 (P4₁) | z | 2 | 16 | 1 | 16 | 89 P422 (a+b, −a+b, c) | 0 / 3 / 0 / 2 | φ |
| 86 (P4₂/n) | – | 3 | 32 | 1 | 32 | 89 P422 (a+b, −a+b+½, 2c) | 0 / 2 / 4 / 0 | φ |
| 88 (I4₁/a) | – | 3 | 32 | 1 | 32 | 93 P4₂22 (sheared) | 2 / 4 / 0 / 0 | t_{x−y}, φ |
| 96 (P4₃2₁2) | – | 3 | 32 | 1 | 32 | 93 P4₂22 (a+b, −a+b, 2c) | 2 / 4 / 0 / 0 | t_{x+y}, φ |
| 145 (P3₂) | z | 2 | 36 | 1 | 36 | 177 P622 (2a+b, −a+b, c) | 0 / 3 / 0 / 2 | φ |
| 148 (R-3, hex. axes) | – | 3 | 36 | 1 | 36 | 155 R32:H (−a, −b, 2c) | 0 / 2 / 5 / 0 | φ |
| 154 (P3₂21) | – | 3 | 24 | 1 | 24 | 181 P6₄22 (a, b, 2c+⅙) | 0 / 5 / 0 / 0 | φ |
| 169 (P6₁) | z | 2 | 12 | 1 | 12 | 177 P622 | 0 / 3 / 0 / 2 | φ |

Read this way, the position part is a function of tabulated objects: **ASU(Γ⁺) is the ITA asymmetric unit of the
Sohncke group Γ⁺, expressed in G's basis, with the polar axes gauged out.** Two details apply only to some groups:

- In the nine groups whose translation seam runs along x ± y (sg 88, 91, 92, 95, 96, 98, 122, 141 and 142), the
  chart's centroid coordinates are the sheared s = M t, with |det M| = 2. ASU(Γ⁺) is then a box in s
  (`charts.PolytopeARChart`).
- The **gauge** on free axes is a [convention] set in code: half the trainer box, the value
  `MolCrystalOps.canonicalize_free_axes` sets (`domains.py` docstring).

### 1.4 The orientation part: SO(3), halved in sg 6–9

**Why the orientation needs no cut in most groups** [proven, `certify.py` docstring, steps 1–3]. An element g ≠ id of
Γ⁺ acts on poses by (x, R) ↦ (ḡx, Q_g R).

- If Q_g = I, then W = I, so g is a nonzero translation. A nonzero translation fixes no centroid.
- So no element except the identity fixes a pose: the action on poses is free.
- With K trivial, the centroid tiling of §1.3 lifts to the poses: P × SO(3) is a fundamental domain, up to ∂P × SO(3).
- Over a centroid on ∂P with a nontrivial stabiliser, the stabiliser acts freely on SO(3). The fold's strict
  lexicographic order on Q_g R then keeps exactly one of the candidates.

**Why sg 6–9 need a cut** [proven, elementary]. In Pm, Pc, Cm and Cc the free axes are x and z, and the one centroid
coordinate is t_y. Γ⁺ contains the 2-fold rotation about b, k = (W_K, w_K) with W_K = diag(−1, 1, −1). After
projecting out x and z it fixes every t_y, so k̄ = id, but Q_K = diag(−1, 1, −1) ≠ I (Cartesian y is b in MXtalTools'
frame). Every pose (t, R) therefore has a second description (t, Q_K R) over the same centroid. K = {1, k} must be
quotiented in SO(3).

The cut R₀₀ ≥ 0 keeps one of each pair, because (Q_K R)₀₀ = −R₀₀. It is a half-space in the matrix entry R₀₀, its
Haar mass is ½, and the face R₀₀ = 0 is glued to itself by K.

- In the hemisphere chart (θ, φ, r), with r signed in (−π, π], the cut is |r| ≤ ρ(θ, φ), where
  ρ = arccos(−u_x²/(1 − u_x²)) for u_x² < ½ and ρ = π above, with u_x = sin θ cos φ
  (`charts.KCutHemisphereChart`). [tested: Haar mass ½ by quadrature to 1e-6 and by Monte Carlo; agreement with the
  fold on 3,794 adversarial rows; `test_k_cut_chart.py`.]
- **Uniqueness, not a double cover, is an owner requirement** [convention, owner 2026-10-26]. The double cover
  (`orientation_cover='double'`, multiplicity 2) remains available as an option.

The orientation chart used by GFN is the hemisphere chart [convention, owner 2026-10-26; `fd_coordinates.ORIENTATION`].
It is the rotation by angle r about the axis u = (sin θ cos φ, sin θ sin φ, cos θ), with θ ∈ [0, π/2]. Its Haar
log-density is log(sin²(r/2) sin θ) − log(2π²) (`so3.py`).

### 1.5 Why each crystal appears exactly once

**Statement.** Fix G at Z′ = 1, a molecule with no proper point symmetry, and a cell that is not a special metric.
Then the map *description → canonical description* is constant on each orbit of §1.1, and its image FD(G) holds
exactly one description per orbit. This holds except on a set of measure zero: the faces of D_cell, the faces of
ASU(Γ⁺/K), the K-cut face, and the fold's float tolerance band (|slack| ≲ 1e-9 in fractional units).

**Status, factor by factor:**

| part | status | evidence |
|---|---|---|
| centroid, ASU(Γ⁺/K) tiles | [proven] | exact certificate, 230/230; proof in `certify.py` |
| orientation, product with SO(3) | [proven] | free-action argument, `certify.py` steps 1–3 |
| sg 6–9 K cut | [proven] (elementary) + [tested] | §1.4 |
| boundary points: one per orbit | [tested] | cctbx tie-break trees (gate 0), gate 1 special and near-special points |
| triclinic cell domain | [argued] + [tested] | Niggli theory through the (a, −b, −c) map; gate 1 face tests |
| monoclinic cell domain | [derivation sketch] + [tested] | Γ₀(2) argument; `test_monoclinic_cell.py` |
| orthorhombic and higher cell | [proven] trivially / [tested] | sorting a finite orbit; `test_higher_sg_cell.py` |
| cell × pose product | [derived] for generic metrics | §1.2 note; special metrics are flagged, not resolved |
| chart = FD (a uniform chart draw is its own canonical form) | [tested] | `test_higher_sg_charts.py`: rerun on 11 Day groups, 33/33 (FD audit); stress sweep of 230 groups, 2026-10-26 |
| invariance under re-description, 26 Day groups | [tested] | FD audit: 40,000 random crystals per group, re-described by a random Γ⁺ element and lattice shift, max \|Δs\| 2.1e-12 |

**What "exactly once" does not cover** [derived]. The FD quotients the crystal's descriptions, not the molecule's own
symmetry. A molecule whose proper point group P⁺_mol has order σ appears σ times per crystal, at R and R·p for
p ∈ P⁺_mol. For achiral molecules the count also depends on the space-group type (§4.3). This choice is a
[convention]: owner, 2026-10-26, "molecular point symmetry is ignored" (README, owner decisions).

### 1.6 Seams and walls

Every face of the FD is one of three kinds:

- **Glued seam.** The face is identified with another face, or with itself, by a group element. Across it, the
  quotient continues. A crystal just across a glued seam is close to one just inside, but its chart point is far away.
- **Mirror wall.** The face is fixed pointwise by an element, so the quotient genuinely ends there. On such a face the
  lattice has an extra automorphism, i.e. a special metric.
- **Support wall.** The length box truncates the support. It is not part of the quotient.

*Table 3. The faces of the FD, what each is in the quotient, and what GFN's flow sees.* GFN wraps exactly the circle
coordinates (`FDCoordinates.wrap_mask`); every other coordinate is a bounded interval in [−1, 1]. Sources:
`charts.py`, README "API in brief", the 2026-11-02 FD audit §2.

| face | in the quotient | in GFN's chart | glued by the metric of §3? |
|---|---|---|---|
| centroid facets glued by a pure translation | glued seam (circle) | **wrapped** (period 2 in s) | yes (lattice shifts and Γ⁺) |
| centroid facets glued by a rotation (self- or pair-twisted, Table 2) | glued seam; the orientation jumps by Q_g across it | **wall**: a basin cut by it becomes two chart regions | yes (Γ⁺) |
| free-axis ends | gauge, not a face | no coordinate | n/a |
| orientation azimuth φ = 0 ≡ 2π | glued (circle) | **wrapped** | yes (geodesic angle) |
| hemisphere equator θ = π/2, glued by (φ, r) ↦ (φ + π, −r) | glued seam | wall | yes |
| sg 6–9 cut face R₀₀ = 0, glued to itself by K | glued seam | wall | yes (Γ⁺ includes K) |
| triclinic and monoclinic cell faces | glued by a basis change in S(G), or a mirror wall where the lattice on the face has an extra automorphism. Every monoclinic face is a mirror of the setting class, so a cell on it is a special metric | wall | yes: `GluedFDMetric` minimises over near-minimal bases (sg 1–15) |
| orthorhombic a = b (and b = c in six-order groups) | mirror wall (special metric) | wall | **no**: axis-order faces are not glued |
| Pbca seams a = b and a = c, glued by the 3-cycle | glued seam; cone point a = b = c | wall | **no** |
| cell-length box | support wall (truncation) | wall | n/a |

**The length box is under revision** [open].

- The committed adapter (`energies/fd_coordinates.py` at `2f17b75d`) bounds cell lengths with the old trainer
  latent's per-axis box (`CrystalChart.for_latent_v1`).
- That box excludes 1,464 of Day's 29,947 low-energy structures (4.9%) and 42 of 1,017 global minima [measured,
  FD audit Table 5].
- A replacement rule, one box for all three axes, [2.5 Å, 2Z(r + 3 Å)], with Z the number of molecules in the
  conventional cell, holds every Day and CSD structure in the same audit [measured, search/prior session note of
  2026-11-02]. It is pending owner sign-off.
- An uncommitted implementation of that rule is in the working tree.

The FD itself (cell domain × pose) has no support truncation. Only the box truncates.

---

## 2. Reduction: from any description to its FD point

### 2.1 The standard settings

The charts use G's operators in the **ITA standard setting**, as cctbx returns it for each space-group number
[convention, design note `fd_tooling.md` §2.1]:

- monoclinic: b-unique, cell choice 1;
- origin choice 2 where ITA lists two (for example P4₂/n:2, I4₁/a:2);
- rhombohedral groups on hexagonal axes (R-3:H).

`SYM_OPS[sg]` equals cctbx's `space_group_info(number=sg)` operator set in all 230 groups [tested:
`test_gamma_equals_cctbx`, re-run for this draft].

**Day's CIFs use exactly these operators** [measured, this draft]. Every one of the 5,275,537 structure blocks in
Day's FIT+DMA landscape CIFs carries the operator set `SYM_OPS[sg]` of its space group (sg read from the block id):

- `settings.find_setting` returns P = I and p = 0 for all 32 distinct (sg, operator set) pairs;
- these are the 26 standard Day groups, plus 41, 45, 92, 110, 114 and 122, which hold 1,472 blocks between them.

Day's cells are **not reduced**:

- among the 18,891 monoclinic low-energy structures of the audit's selection (at most 30 per molecule within
  10 kJ/mol), 9,342 have an acute β, which is outside wall W1 [measured];
- 2,067 of 2,502 triclinic structures have a > b or b > c [measured].

So for Day, reduction is the cell step plus the pose fold. No change of setting is needed.

### 2.2 The pipeline

`canon.canonicalize(desc)` runs these steps on any batch, which may mix space groups. It never raises for a single
row; each row gets a status:

1. **Setting** (`io_mxt` and `settings.find_setting`; only for rows not already in `SYM_OPS[sg]`). This finds
   (P, p) with x_stored = P x_std + p and re-expresses the row as cell PᵀGP, t′ = P⁻¹(t − p), R′ = Q R.
   - A row is refused if the re-expressed cell is not preserved by G's rotations within 1e-6.
   - This direction (one-way import into the standard setting) is a [convention]: owner D1, 2026-10-16.
   - [tested: all 530 cctbx Hall settings resolve exactly, plain, with origin shifts on a 1/24 grid, and with
     unreduced bases (verifier sweep 2026-10-17, a script, not a gate); `test_io_settings.py` covers the standard
     setting of all 230 groups and the 85 operator-set classes of the CSD file.]
2. **Cell step.** sg 1–15: `ops.reduce_cell` into D_cell, applied jointly to cell, centroid and frame through
   `ops.change_basis`. sg ≥ 16: snap to the exact metric (the mean of WᵀGW over G's rotations); in sg 16–74, then the
   canonical axis order with its origin shift.
3. **Sheet.** `domains.to_sheet`: h = −1 → +1 through an improper element of G (§4).
4. **Fold.** `domains.fold`:
   - apply every Γ⁺ element with the lattice shifts that can reach the ASU;
   - keep the candidates inside the ASU (the slack test, with cctbx's tie-break trees within tol = 1e-9);
   - choose among them by the lexicographically largest Q R, then the largest centroid;
   - set free axes to the gauge.

   The choice is a function of the orbit, not of the input [tested: gate 1].
5. **Chart.** `CrystalChart.inverse`. In GFN, `FDCoordinates.from_params` also maps through u = (y − lo)/(hi − lo) and
   s = 2u − 1, and returns NaN with a status for rows outside the box.

The statuses that pass are `canonical`, `special_metric` (`n_auto` > 1) and `off_sheet` (a Sohncke row left at
h = −1). `fd_status(desc).is_fd` is the computed membership test: a row is in the FD when it equals its own decided
canonical form within tol.

### 2.3 What can still go wrong, and how often

- **Special metrics.** Status `special_metric`: the canonical pose depends on the input basis.
  - Day: 86 of 29,947 low-energy structures [measured, FD audit].
  - A float32 cell on a monoclinic face (β = 90°) is flagged too. The flag over-marks rather than under-marks
    (README "Known limits").
- **Rounding at faces.** A crystal within float32 rounding of a cell or centroid face can canonicalise to either side.
  - `near_face` flags these rows (tolerance 1e-5).
  - `canonicalize(nudge=δ)` moves them δ inside [convention, owner 2026-10-21; off by default]. A moved row is another
    crystal and must be rescored.
  - Observation (CSD, 2026-10-17): about 8% of pairs of CSD determinations of one crystal differ by more than 1 Å in
    canonical form [measured]. **Equal canonical forms are not a duplicate test**; §3 is the identity test.
- **The tolerance band.** Within ~1e-9 of a face, plane tests within a tolerance are not equivariant (README
  "tolerance band"). This has measure zero.

---

## 3. The metric: distance between two crystals on the FD

### 3.1 Definition

`metric.FDMetric` (MXtalTools; GFN's `eval/fd_analysis/fd_distance.py` imports it). Both crystals are first put in
canonical form. Then

$$d(A,B) \;=\; \min_{g=(W,w)\in\Gamma^+,\; n\in\{-1,0,1\}^3}\;\sqrt{D_{\rm cell}^2 + D_{\rm cent}(g,n)^2 + D_{\rm rot}(g)^2}\quad[\text{Å}],$$

with

$$D_{\rm cell}^2=\Delta a^2+\Delta b^2+\Delta c^2+(\Delta\alpha\sqrt{bc})^2+(\Delta\beta\sqrt{ac})^2+(\Delta\gamma\sqrt{ab})^2,$$

$$D_{\rm cent}=\big|\bar T\,(u-n)\big|,\quad u = t_A-(Wt_B+w)\ \text{wrapped to }[-\tfrac12,\tfrac12)^3,\quad \bar T=\tfrac12(T_A+T_B),$$

$$D_{\rm rot}= r_g\,\theta\big(R_A,\;Q_W R_B\big).$$

- θ is the geodesic angle on SO(3).
- r_g is the molecule's radius of gyration about the heavy-atom centroid.
- Lengths and angles in D_cell are scaled by the means of A's and B's.

Each term is an approximate Cartesian displacement of one molecule, so d is a one-molecule RMSD estimate [derived;
`metric.py` docstring]. The weights are a [convention], set in code without a recorded owner decision.

**What the minimisation glues** [derived from the definition]:

- Γ⁺ and the lattice shifts glue every pose seam of Table 3: translation and twisted centroid seams, the equator and
  the K cut.
- `GluedFDMetric` also minimises over B's near-minimal bases within `margin` = 1 Å of a + b + c. For sg 1–2 these are
  the 3,480 det +1 matrices with entries in {−1, 0, 1}. For sg 3–15 they are setting-class matrices with entries in
  {−2, …, 2}, each with its origin shift. This glues the cell faces of sg 1–15.
  - The entry-2 range is needed. With entries ±1 only, five NEHZOR (P2₁/c) pairs that COMPACK matches at 0.02–0.09 Å
    measured 2.5–3.5 Å; their cells lie on the face |c cos β| = a, glued by c ↦ c + 2a [measured, 2026-10-28].
- **Twin.** For a near-planar molecule, `twin_canonical` also compares B with the image of B under the mirror through
  the molecular plane. This is the only molecular symmetry the metric quotients.

**Not glued, and not quotiented** [derived]:

- orthorhombic axis-order faces and Pbca's seams (sg ≥ 16 gets the identity basis only);
- proper molecular point symmetry (C₂, C₃, … axes);
- crystals related by the enantiomorph map (Γ⁺ only).

**Status** [open]: d is the quotient distance only when the nearest images of the two crystals lie inside the
expanded set. The class docstring states this as a working assumption to be checked empirically. Farther apart, d is
an upper bound. The triangle inequality is not proven.

### 3.2 Calibration against COMPACK

*Table 4. Glued FD distance with the twin, against COMPACK (20-molecule shells, CCDC default tolerances; a match is
20/20).*

- Sources: `D:/crystal_datasets/fd_analysis_oct26/compack_glued_{mipcas,nehzor}_v2/summary.md`, written 2026-10-28.
- The summaries do not record the code commit. The v2 NEHZOR run brings 5 more COMPACK-matched pairs into the
  < 0.05 Å bin than v1, which is consistent with the entry-2 fix [measured].
- **The pairs were drawn on the RDF and unglued-FD decision boundaries.** The AUCs describe hard pairs, not a random
  population.
- Compare the AUC column with the RDF-distance AUC on the same pairs.

| molecule (sg) | pairs (count) | COMPACK matches (count) | AUC, glued FD + twin | AUC, RDF distance | d at P(match) = 0.9 / 0.5 / 0.1 (Å) | glued FD / COMPACK RMSD on matches, P10 / P50 / P90 (ratio) | Spearman vs COMPACK RMSD on matches |
|---|---|---|---|---|---|---|---|
| MIPCAS (P-1) | 720 | 410 | 0.952 | 0.901 | 0.81 / 1.11 / 1.52 | 2.14 / 2.58 / 2.92 | 0.995 |
| NEHZOR (P2₁/c) | 960 | 485 | 0.992 | 0.940 | 0.63 / 1.10 / 1.91 | 1.88 / 2.46 / 4.00 | 0.959 |

What the calibration supports [measured]:

- **Identity cut 0.05 Å.** Every pair below it was a COMPACK match: 97/97 for MIPCAS and 122/122 for NEHZOR. The
  matched COMPACK RMSD there has a median of 0.003–0.005 Å.
  - This is the cut `crystal_search/minima.py` uses to merge minima of *one* energy model.
- **Scale.** d ≈ 2.5 × COMPACK RMSD on matched pairs.
- **The twin is necessary for these two molecules.** Without it, the glued AUC is 0.80 for MIPCAS and 0.94 for
  NEHZOR.
- **It is not a benchmark matcher.** Matching relaxed samples to Day's FIT+DMA minima needs a tolerance of
  0.5–1.5 Å, where P(match) falls from 0.9 to 0.1. COMPACK, or an equivalent, remains the matcher there.
- **Coverage is thin** [open]: two molecules in two groups (P-1, P2₁/c). Nothing in sg ≥ 16, where axis-order faces
  are unglued. No molecule with proper point symmetry. No randomly drawn pairs.

---

## 4. Chirality

### 4.1 Conventions

- **Handedness h** multiplies the molecule's body x axis: the molecule is R·diag(h, 1, 1)·X.
- **The reference molecule X** is the molecule aligned to its principal axes at handedness +1
  (`align_mol_batch_to_standard_axes`). That alignment maps a molecule and its mirror image to the same point set. A
  stored h is therefore relative to the aligned frame, and a consumer must keep a dataset's saved molecule and its
  `reference_handedness` rather than re-deriving them [tested, README "Imports"].
- **The FD lives on the h = +1 sheet** [convention, design: `fd_tooling.md`; README "What it does"].

### 4.2 Non-Sohncke, Sohncke and enantiomorphic-pair groups

*Table 5. How each space-group type treats h = −1 descriptions, with Day's groups.*

- Sohncke = no improper element in G.
- "Γ improper" = the Euclidean normaliser has an improper element.
- [derived] from the definitions; [tested] by `test_enantiomorph.py` on all 65 Sohncke groups.

| type (Day groups) | an h = −1 description is | what the code does | mirror-image crystal of the same reference molecule lies in |
|---|---|---|---|
| non-Sohncke (2, 7, 9, 13, 14, 15, 29, 33, 43, 56, 60, 61, 86, 88, 148) | the same crystal, through an improper element of G | `to_sheet` maps it to h = +1 | the same group (it is the same crystal set) |
| Sohncke, Γ improper (43 groups; Day 1, 4, 5, 18, 19, 20) | a crystal of the mirror molecule X* | import mode D2 (default): re-describe with X* as reference at h = +1. Mode `enantiomorph` (owner 2026-10-21, used for the canonical CSD set): replace it by its mirror-image crystal with reference X | the **same** group: the map is Γ's improper element (−I, −p) |
| enantiomorphic pair (22 groups, 11 pairs; Day 76, 96, 145, 154, 169) | as above | as above | the **partner** group (78, 92, 144, 152, 170) |

The enantiomorph map is inversion through the origin, written in the standard setting [tested, `test_enantiomorph.py`]:

$$t'=(-t-p)\bmod 1,\qquad R'=R\,\mathrm{diag}(1,-1,-1),\qquad h'=-h,\qquad sg'=\begin{cases}sg & \Gamma\text{ has an improper element}\\ \text{ITA partner} & \text{otherwise}\end{cases}$$

The origin shift p is nonzero only in sg 80, 98 and 210. Applying the map twice gives the identity.

**Energies must be mirror-invariant** [convention, owner rule 2026-10-21: "a crystal and its mirror image scoring
differently is a defect"]. Measured on 2026-10-21:

- eLJ and the three MACE models are mirror-invariant to float32 noise;
- the UMA checkpoints are not: a median of about 0.2 kJ/mol per molecule and up to 2 kJ/mol on lattice energies
  [measured, `fd_tooling.md` §6 D2].

### 4.3 Molecular point-group multiplicity

*Table 6. FD points per physical crystal on the h = +1 sheet, by molecule chirality and space-group type.*

- [derived] in the SG free-energy theory note (§3), and recounted independently [measured] for P-1, P2₁/c, C2/c,
  P2₁2₁2₁, Pbca and P4₁ with C₁, C_s and C_2v molecules (refutation report, Table R1).
- Also consistent with `test_zprime_orbits.py`, whose mirror-only stand-in conformer gives f = 2, or 1 in the 22
  pair groups.
- |P⁺_mol| is the order of the molecule's proper point group; |P_mol| = 2|P⁺_mol| for an achiral molecule.
- Compare the rows: they differ by factors of 2 between space-group types.

| molecule | space-group type (Day examples) | the mirror-image crystal is | FD points per crystal (count) |
|---|---|---|---|
| chiral, enantiopure | Sohncke, not a pair (1, 4, 5, 18, 19, 20) | a crystal of the other enantiomer: not in the system | \|P⁺_mol\| |
| chiral, enantiopure | pair member (76, 96, 145, 154, 169) | the other enantiomer, in the partner group | \|P⁺_mol\| |
| chiral, racemic compound | non-Sohncke (2, 7, 9, 13–15, 29, 33, 43, 56, 60, 61, 86, 88, 148) | itself | \|P⁺_mol\| |
| achiral | non-Sohncke | itself | 2\|P⁺_mol\| = \|P_mol\| |
| achiral | Sohncke, not a pair | a **distinct** crystal in the **same** chart | \|P⁺_mol\|, plus another \|P⁺_mol\| for the mirror crystal |
| achiral | pair member | a distinct crystal in the **partner** group | \|P⁺_mol\| |

Consequences [derived, refutation report §2.1]:

- **Within one space group** the factor is a constant per molecule. It does not bias the shape of a density on the
  chart.
- **Across space groups** it differs by type. For an achiral molecule, counting per crystal weights Sohncke groups ×2
  and pair groups ×4 relative to non-Sohncke groups (pair groups ×4 when the partner is not sampled). Counting per
  Day structure (one weight per mirror pair) gives ×1 and ×2.
  - Which counting rule the paper uses is an owner decision [open].
  - ln 2 ≈ 0.69 nats, about 1.7 kJ/mol at kT = 2.5 kJ/mol.
- **Day's pair groups** [measured, refutation report T15]. Day searched one member of each pair. The partner groups
  are absent from the standard 26, although Day's CIFs hold 52 blocks of sg 92 (§2.1). Which enantiomer Day used is
  not stated [open]. 118 molecules have Sohncke-only landscapes.

---

## 5. What a generative model gains from the reduced basis, and what it does not

### 5.1 Gains

- **Uniqueness: no multiplicity correction, no reduction walls** [proven / tested, §1.5]. One chart point per crystal
  (times |P⁺_mol|, §4.3). A density on the chart is therefore a density on crystals.
  - The old trainer latent held k_box = 1–36 images of each crystal and covered only half of the crystals in some
    groups (theory note Table S1; e.g. P2₁/c: 8 images, coverage ½).
  - It relied on a soft reduction penalty whose effective copy count is not an integer near a wall (wiki
    `cell-fundamental-domains.md`).
  - On the FD chart the reduction penalty is zero by construction (`fd_coordinates.py` docstring).
- **Bounded, autoregressive coordinates** [derived; `charts.py`].
  - Column order: cell, then centroid, then orientation. Each coordinate's interval depends only on the coordinates
    before it: Fourier–Motzkin bounds of ASU(Γ⁺), the Niggli and monoclinic walls in the cell, ρ(θ, φ) for the K cut.
  - u ∈ [0, 1]^d therefore maps one-to-one onto the FD, and s = 2u − 1 ∈ [−1, 1]^d.
  - There are no dead rows: free centroid axes and fixed angles are not coordinates.
  - Day's charts have d = 7 to 12 coordinates (P4₁ 7, P-1 12).
- **Circles where the quotient is a circle** [derived; tested: `test_circle_ends_are_one_crystal`, translation-seam
  test over all 230 groups]. The wrap rows are exactly the orientation azimuth and the translation-glued centroid
  coordinates; s = −1 and s = +1 are then the same crystal. P-1 is the only Day group whose whole centroid block is a
  torus.
- **An explicit reference measure with closed-form Jacobians** [derived; tested: gate 5 against autograd and central
  differences; FD audit's 256 χ² tests on 26 Day groups found no pose-measure error]. The measure is

  $$d\mu_{\rm ref}=\mathrm{Haar}(dR)\times V\,d^kt\times\mu_{\rm cell}(dc),$$

  and the log-reward and its log-density on s are

  $$\log R(s) = -\frac{E(s)}{T}+\log\rho_{\rm ref}(s),\qquad \log\rho_{\rm ref}(s)=\log\rho_{\rm ref}(y)+\sum_i\log(hi_i-lo_i)-d\log 2.$$

  With this reward, a sampler on s targets the Boltzmann density relative to μ_ref, each crystal once
  (`fd_coordinates.py` docstring; `measure.log_reference_density`).
  - V d^k t is the Cartesian volume element of the centroid, with free axes integrated over one period.
  - Haar is normalised. On the K cut its mass is ½.
- **Identity and deduplication.** Canonical form plus the glued metric give a minima-merging test at 0.05 Å, with no
  learned or RDF component (§3.2).
- **Interpretable reduced cells.** Triclinic cells are Niggli-like (shortest vectors, obtuse β and γ). Monoclinic
  cells have β ∈ [90°, 135°] or [90°, 120°], depending on class. Orthorhombic axes are ordered. A model's cell block
  therefore never spends capacity on equivalent bases.

### 5.2 What the chart does not fix

- **The cell measure is a convention** [convention, owner 2026-10-15, O4 = (a): `'latent_v1_lebesgue'`].
  - It is flat in each free log length and free angle. Within the FD it is a choice, not an invariant.
  - The basis-invariant alternative is dG/V. Relative to the chosen measure its density is
    8(abc)⁴ sin α sin β sin γ / V [derived from dG = |∂G/∂(log a, …, γ)|; `measure.log_cell_density`]. It is built for
    triclinic groups only.
  - The choice moves relative basin weights within one molecule. The median range across a molecule's low-energy
    structures is 0.22 nats in P-1 and 1.90 nats in P2₁/c (P90 0.49 and 3.11) [measured, FD audit Table 2].
  - The sampler's target is therefore *Boltzmann × a declared cell prior*, and should be stated so in print.
- **No measure across space groups** [open; convention for any slice weight].
  - The number of cell coordinates changes with the family (6, 4, 3, 2, 1). Each chart is a measure-zero slice of
    the P1 description space with a different codimension (theory note §1.3).
  - A ratio of chart integrals between families is a convention. With flat measures, changing radians to degrees
    changes Z(P-1)/Z(P2₁/c) by (180/π)².
  - The physical polymorph free energy has no per-space-group constant (theory note §1.2). Slice weights between
    space groups are a declared convention, not a property of the chart.
- **Support truncation** [open, §1.6]. The length box sets the support, so it changes log Z per (molecule, space
  group). It is under revision.
- **Molecular point symmetry** [convention, §4.3]. A symmetric molecule's density has |P⁺_mol| copies of every mode.
- **Seams split basins** [open]. A basin straddling a twisted seam, the equator, the K cut or a cell face occupies two
  chart regions that the flow sees as walls.
  - Not measured at Z′ = 1.
  - For Z′ = 2, 4–16% of CSD configurations lie within 0.1 Å of a twisted facet (`zprime_fd.md` Table 1) [measured].
- **Special metrics** [open, §2.3]. These have measure zero, but they are where real crystals often sit, e.g. a
  float32 β of exactly 90°.
- **Scope.** Z′ = 1 only. GFN's trainer refuses FD with more than one space group
  (`MolecularCrystal.__init__`) [tested by reading, FD audit §1].

### 5.3 The support box, stated for the SI

**Under revision.** Today's committed chart is FD(G) ∩ {cell lengths in the trainer latent's per-axis box at the
molecule radius}, which excludes 4.9% of Day's low-energy structures. A uniform box [2.5 Å, 2Z(r + 3 Å)] that
excludes none is pending owner sign-off. The SI should give whichever box is adopted, with its coverage on Day and on
the CSD.

---

## 6. Gaps a referee would flag

1. **The triclinic cell domain has no written proof** in the shifted sign convention with its Gram-entry tie-break.
   Fix: cite Křivý–Gruber with the explicit (a, −b, −c) bijection and the face rules, or switch to standard Niggli
   conditions.
2. **The monoclinic Γ₀(2) domain per setting class** has a derivation sketch only. Each class's walls and face
   identifications need writing out (GLIDE, CENTRED, and FREE with a ≤ c).
3. **The product theorem** (cell × pose) with its measure-zero exceptions (special metrics, faces, tolerance band) is
   argued in §1.2 but not stated as a theorem. Special metrics are flagged, not resolved.
4. **The boundary tie-breaks** (cctbx cut trees; one point per boundary orbit) are tested, not proven. The
   certificate covers interiors and facet coverage only.
5. **Γ⁺ as ITA types (Table 2)** is a cctbx identification made for this draft. It is not cross-checked against the
   normaliser tables of ITA Vol. A §3.5.
6. **No cross-space-group measure.** Relative SG weights rest on a declared slice convention (§5.2) and on a
   counting rule for mirror pairs (§4.3); both are owner decisions [open].
7. **The cell measure** is a declared prior, not an invariant. It moves basin weights by ~2 nats in P2₁/c (§5.2).
8. **The support box** is under revision (§5.3).
9. **The metric:**
   - it is calibrated on two molecules in two triclinic/monoclinic groups, on boundary-drawn pairs;
   - axis-order faces are unglued;
   - proper molecular point symmetry is not quotiented;
   - it has no direct unit test (`GluedFDMetric` is exercised only through `tests/test_search_minima.py`);
   - quotient exactness and the triangle inequality are working assumptions.
10. **Seam mass is unmeasured at Z′ = 1.** How much low-energy mass sits near twisted seams, where the flow sees a
    wall, is not known.
11. **Day's enantiomer and pair-group coverage** is unverified: which enantiomer was searched, and whether pair
    groups count once (§4.3).
12. **Modelling evidence is thin.** There are four single-SG phase-1 FD runs with one seed each, no phase-2 FD run,
    and no multi-SG FD run (FD audit §6).

---

## Reproduction

From the MXtalTools root, CPU only (`CUDA_VISIBLE_DEVICES=-1`):

```bash
python -m mxtaltools.crystal_fd.certify
```

```bash
python -m pytest tests/fd -q -m "not slow"
```

```bash
python -m pytest tests/fd/test_gate0_tables.py -q -k "gamma_equals_cctbx or asus_equal_cctbx or gamma_equals_brute_force"
```

These are, in order: the exact tiling certificate for all 230 groups (seconds); the fast lane, which took 22 min in
the FD audit; and the cctbx equality checks, which need cctbx.

## Runs made for this draft (2026-11-02)

- `certify --sgs <26 Day groups>`: 26/26 pass, 0.9 s.
- `test_gate0_tables.py -k "gamma_equals_cctbx or asus_equal_cctbx or gamma_equals_brute_force"`: 690 passed, 82 s.
- Day CIF operator check: 5,275,537 blocks, 32 (sg, operator set) pairs, all with `find_setting` (P, p) = (I, 0).
- Day monoclinic and triclinic cells against the reduced domain, from the FD audit's `day_box_check.json`: 9,342 of
  18,891 monoclinic structures with acute β; 2,067 of 2,502 triclinic structures not in length order.
- Γ⁺ space-group type and facet-gluing kinds per Day group: cctbx `space_group_info(group=Γ⁺)` and
  `GroupTable.facets`.
- GFN FD tests: **not run.** The `csd_mxt_gfn` venv does not import MXtalTools without a path setting, and
  `fd_coordinates.py` was being edited by another session. The FD audit ran them at `2f17b75d`: 55 passed.

## Sources

- MXtalTools `mxtaltools/crystal_fd/`: `README.md`; the docstrings of `certify.py`, `domains.py`, `canon.py`,
  `charts.py`, `measure.py`, `metric.py` and `enantiomorph.py`; `tests/fd/`.
- GFN `energies/fd_coordinates.py` (committed version at `2f17b75d`); `eval/fd_analysis/fd_distance.py`.
- Wiki `cell-fundamental-domains.md` (triclinic and monoclinic walls; stamped 2026-10-04, before the FD charts).
- Design notes `fd_tooling.md` (owner decisions O4, D1, D2) and `zprime_fd.md`.
- Review reports of 2026-11-02, outside the repository (session scratchpad `.../d132fb12-.../scratchpad/review/`):
  - `fd/audit.md`: proof status per Day group, support box, cell-measure spread, metric calibration;
  - `sg/sg_free_energy_theory.md` §1.3 and §3: chart dimensions, multiplicities;
  - `sg/refutation.md` Table R1 and §2.1: an independent recount of the multiplicities.
- Search/prior session note `fd_length_box_proposal_2026-11-02.md` (shared session notes): the candidate length box.
