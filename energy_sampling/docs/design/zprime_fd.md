# Z′ > 1 as a set of units: equivalence group, fundamental domain, measure and charts

- **Status:** PROPOSAL, 2026-10-23. Owner answers so far (§10): Q1 (a), the fixed ASU(G) placement, "at least for now"; Q4 deferred far down the line; Q3 (handedness labels) open, working proposal (d); Q2 follows Q3. A design note; it has no authority over code, config or tests, and it changes nothing in `fd_tooling.md` until the owner adopts it (§11 lists what it would change). **Built 2026-10-23 (MXtalTools working tree, uncommitted):** stage Z0 (gate G1 as `tests/fd/test_zprime_orbits.py`) and stage Z2 (`crystal_fd.domains.fold_unit` / `UnitFold` / `unit_section`, and gate G3 as `crystal_fd.certify.g_asu_domain` / `g_asu_certificate`, all 230 groups); additions only, the existing `tests/fd` outcomes identical before and after (4,052 tests); reviewed independently (`artifacts/fd_design_oct15/zprime_build_oct23/`). Z1, Z3-Z7 not built; labelled units wait on Q3.
- **Verification:** an independent orbit counter written from this note's definitions (`$Z/verify/`) reproduces every count in §3.1 and §8 G1, every orbit size and torus count, and the table census; its four corrections to the first draft (the seam cost in §5 and Q1, the sg 4 tie result in §3.9, the census's 'other' row in §2.3, the pad-slot value in §4) are applied.
- **Inputs:**
  - three designs written on 2026-10-23: *anchored* (`$Z/anchored_labelled/`), *symmetric set* (`$Z/symmetric_set/`) and *efficiency* (`$Z/efficiency/`);
  - two reviews of them. One review chose the anchored design and the other the efficiency design. Every error either review found is fixed here (§12).
- **Evidence:**
  - `$Z` = `energy_sampling/artifacts/fd_design_oct15/zprime_fd_oct23/`;
  - `$ZP` = `energy_sampling/artifacts/fd_design_oct15/zprime_oct21/theory/`;
  - new checks for this draft are in `$Z/synthesis/synth_checks.py`, with its `synth_*.json` outputs.
  - Everything ran on CPU and read-only on both repositories. The CSD rows were read from `D:/crystal_datasets/ingest_csd_v1_r2` (read only).
- **Labels:**
  - [derived]: an argument given here. It is consistent with the measurements but has not been independently reviewed.
  - [measured]: computed by the named script, with the number given.
  - [assumed]: neither derived nor measured.
  - [code]: read at the cited place.
- **Tier:** every statement about conformers is for the `full` internal-coordinate tier, and holds at `dihedral` and `flex` too. At `torsion` the parity is frozen and §3.7 applies.

**Notation.**
- **Groups:**
  - G: the space group, in the standard setting;
  - Γ: N_E(G), counted modulo ℤ³ and free-axis shifts, improper elements included;
  - Γ⁺: the det +1 part of Γ;
  - K: the table's kernel, C2(b) in sg 6–9;
  - |G| and |Γ|: element counts per conventional cell.
- **Units:**
  - n_c: the number of units of compound c;
  - a: the anchor compound, and n_a its number of units.
- **Mirror:**
  - σ: negate every dihedral (and every transverse v) row of a conformer;
  - D = diag(1, 1, −1): the reflection through the conformer's seed plane.
- **Multiplicities:**
  - f: the anchor's improper copy (§3.1);
  - a_c: the number of chart copies of one conformer of compound c (§4).

## 1. Topline

Once each unit's conformation is sampled in internal coordinates, the unit's hand lives in its conformer, so the equivalence group is one common normaliser element, an independent element of G for each unit (an improper one flips that unit's conformer) and relabellings within compounds. Recommended: one anchor unit, chosen by a discrete compound rule, in the shipped Z′ = 1 domain ASU(Γ⁺) × SO(3) × Q, and every other unit in the shipped ASU(G) × SO(3) × Q, with labels kept and no unit ordered by any key; the remaining f · ∏_c n_c! copies (f = 2 in 208 groups, 1 in the other 22) are an exact multiplicity, of which an exchangeable model absorbs all but f · n_a [measured: exactly f images per labelling in all 230 groups at Z′ = 2, 19 groups at Z′ = 3 and three compositions; the absorption is derived]. The strongest alternative, for exact-density samplers only, puts the other units on an anchor-relative lattice torus and averages their G images out, which removes every seam they have at the cost of |G/T_prim|^(Z′−1) times more copies; the choice is the owner's (Q1).

## 2. The equivalence group

### 2.1 A unit, and the mirror in its conformer

**The unit.** Unit k carries:
- a compound c(k);
- a fractional heavy-atom centroid t_k;
- a proper rotation R_k, the rotation that takes the conformer's seed frame to the lab;
- internal coordinates q_k.

Its atoms are B t_k + R_k X̃(q_k), where X̃ is X(q) centred on its heavy-atom centroid.

**The seed frame [code].** `conformers/geometry.py::place_seed_second` and `::place_seed_third` put the root at the origin, slot 1 on +x and slot 2 in the xy-plane.

**The mirror [code, via wiki conformer-chart-and-internal-coordinates].** φ → −φ on every row is the global reflection: X(σq) = D X(q). This is checked by `mxtaltools/conformers/tests/test_dummy_frame.py::test_negating_every_phi_and_v_is_the_mirror_image`. Hence X̃(σq) = D X̃(q).

**How an isometry acts on one unit [derived].** An affine isometry (W, w) acts as

  (t, R, q) ↦ (W t + w, M_W R D^s, σ^s q), with s = [det W = −1] and M_W = B W B⁻¹.

This reproduces the isometry applied to the unit's atoms exactly: B(Wt + w) + M_W R D^s X̃(σ^s q) = M_W(Bt + R X̃(q)) + Bw. Every pose stays proper, so a per-unit handedness token disappears.

### 2.2 One compound

**The group [derived].** Fix the canonical cell; the Z′ = 1 cell step acts on all units alike. Two labelled descriptions are the same crystal (or mirror images, which the owner's rule identifies) exactly when they are related by

  H = { (n; g_1 … g_Z′; π) : n ∈ N_E(G), g_k ∈ G, π ∈ S_Z′ }, acting by u′_{π(k)} = g_k n u_k.

- n includes improper elements, lattice translations and the common free-axis shift.
- Each g_k includes its own lattice shifts.

**Why [derived].**
- A proper motion relating two descriptions preserves the lattice and G, so it lies in N_E.
- Each unit of the new description is a G-image of a unit of the old one. With the hand in q, an improper g_k is admitted (§2.1).

**Generic orbit size, modulo per-unit lattice shifts and the free shift:** |Γ| · |G|^(Z′−1) · Z′!.

**Measured [`$Z/efficiency/zp_design_orbits.json`].**
- Orbit sizes, all equal to the formula:
  - Z′ = 2: 4, 64, 64, 64, 256 and 512 in sg 1, 2, 4, 9, 14 and 19;
  - Z′ = 3: 384 and 3,072 in sg 2 and 14.
- Every sampled crystal matched: 3 to 20 per case.
- A literal atom oracle agrees on every image checked (up to 280 per case) within 6.1e-15 Å.

**Structure [derived; from the symmetric-set design].**
- G^Z′ is a normal subgroup of H, because G is normal in N_E.
- H / G^Z′ = D × S, where D = N_E/G acts on all units at once and S = S_Z′.
  - D has a finite part of order k_D = |Γ|/|G| and the free-shift torus.
- Measured k_D [`$Z/symmetric_set/zp_D_translations.json`]: 8 in sg 2, 14, 29, 33 and 61; 16 in sg 19; 2 in sg 9.
  - The mean over the 48,078 Z′ = 2 CSD rows of the 12 commonest groups is 8.03.

**Fixed points [derived].** An element (π, d) of S × D has a fixed point exactly when the order of d divides every cycle length of π.
- At Z′ = 2, (swap, d) fixes exactly the configurations u₂ = d̃ u₁ with d̃² ∈ G: crystals whose true group is ⟨G, d̃⟩ at Z′ = 1, i.e. **exact pseudo-symmetry**.
- (π, e) fixes only collisions (overlapping units).
- An improper g fixes a unit only where σq = q (an achiral conformer).

All of these sets have measure zero, so the orbit size above holds almost everywhere.
- Measured at u₂ = d̃ u₁: the orbit has half its generic size, for all 7 involution classes in each of sg 2, 14 and 19 (rigid model) and sg 2, 14 and 4 (reflection-closed model) (`$Z/symmetric_set/zp_sym_orbits.json`). With the conformer-carried hand, sg 19 has 15 involution classes, and all 15 halve the orbit (independent verification, `$Z/verify/`).

### 2.3 Several compounds

**The group [derived].** The same H, with π restricted to ∏_c S_{n_c}: units of different compounds are never exchanged. D does not depend on the compounds.

**The compound key.** The compound key is the stereo-free graph (canonical stereo-free SMILES with H).
- Two units that are enantiomers of each other are therefore one compound, since their hands live in q.
- Tautomers and protomers have different graphs, so they are different compounds.

**CSD census [measured, `$Z/anchored_labelled/csd_seams.json` part1, 49,223 Z′ = 2 rows].**

| unit pair | rows (fraction) |
|---|---|
| no '@' in either unit | 46.7% |
| identical stereo SMILES | 38.4% |
| mutual enantiomers | 9.3% (4,602 rows; 538 of them in Sohncke groups) |
| two compounds | 5.0% (2,453 rows) |
| other | 0.6% (275 rows) |

*The four named classes sum to 99.4%; the 275 'other' rows were left out of the first draft (independent verification, `$Z/verify/`).*

### 2.4 Consequences

1. **Each non-anchor unit's residual group is G, not the G⁺ of `fd_tooling.md` §2.8 [derived; measured].** A G⁺ residual leaves 2 images per labelling in every non-Sohncke group tested, against 1 for ASU(G). That is 4 against 2 with the shipped anchor, in sg 2, 7, 9, 14, 15, 29, 33, 61, 88, 148 and 205 (`$Z/anchored_labelled/counts_z2a.json`, `counts_z2b.json`).
2. **The improper copy and the mirror rule [derived].**
   - In the 165 non-Sohncke groups, an improper n equals an improper g₀ ∈ G times a proper element. So the improper copy of a whole crystal is the same crystal, whatever the energy.
   - The mirror rule adds something only in the 43 Sohncke groups with improper N_E. There, the copy is the mirror crystal, an equivalence only for mirror-invariant energies: eLJ and MACE, not UMA (`fd_tooling.md` §6 D2).
3. **The 22 enantiomorphic-pair groups have no improper normaliser element** [measured, `$Z/anchored_labelled/counts_kernel.json`]. Their mirror image lies in the partner group.
4. **The full-Γ kernel K_Γ** (elements of Γ that act trivially on the projected centroid) is nontrivial in 53 groups [measured, `counts_kernel.json`]:
   - sg 6–9: the proper C2(b), as at Z′ = 1;
   - 49 free-axis groups: an improper element (the mirror normal to the polar axis; in P1, the inversion).
5. **Z′ = 1 changes too, once conformers are sampled jointly.** The shipped ASU(Γ⁺) × SO(3) × Q holds 2 descriptions per crystal in the 208 groups whose Γ has an improper element [derived].
   - Measured (`$Z/anchored_labelled/z1_check.json`, 800 crystals per group): 2 in sg 1, 2, 4, 9, 14, 19, 29, 33, 61, 148 and 198; 1 in sg 76, 92 and 169.
   - The rigid +1-sheet statement of `fd_tooling.md` §2.2 is unaffected.

## 3. The recommended representation

### 3.1 What is cut and what is kept

**Cut exactly:**
- the cell, by the Z′ = 1 cell step;
- for the anchor: Γ⁺ (with K in sg 6–9) and the common free-axis shift;
- for every other unit: G, including its improper elements and lattice shifts.

**Kept as an exact multiplicity:** m = f · ∏_c n_c! · ∏_k a_{c(k)}, where:
- f = 2 in the 208 groups whose Γ has an improper element: the anchor's improper Γ coset, which applies σ to every unit;
- f = 1 in the 22 pair groups;
- a_c is the conformer chart's own copy count (§4). It is not a choice made here: owner constraint 6 forbids folding molecular symmetry.

**Measured [`$Z/synthesis/synth_z2_all{1,2,3}.json`, `synth_z3_{a,b,c}.json`, `synth_co_a.json`].**
- **Method:** the domain gets exactly f images per labelling. Each count enumerates all of Γ, all of G per unit with lattice shifts, and every compound-preserving labelling, using the enumerator in `$Z/anchored_labelled/counts.py`. Units carry a σ-odd stand-in for the conformer. `counts.py` cuts K with a Dirichlet zone and tests only the top-level planes; the independent check below uses the shipped rule (`crystal_fd.domains.in_asu`) and gets the same numbers.
- **Independent check [measured, `$Z/verify/zpv_run.py`, `zpv_extra.py`, `zpv_tables.py`]:** an orbit counter written from this note's definitions, with literal chain molecules built from their own internal coordinates (X(σq) = D X(q) to 0.0 Å), the shipped `domains.in_asu` for the anchor and a `Region` from `g_asu` for the other units: exactly f per labelling in all 230 groups at Z′ = 2 (46,000 crystals), 19 groups at Z′ = 3, the compositions B·A, B·A₂ and A₂·W, and Z′ = 1; every deliberate error (others in ASU(Γ⁺), the §2.8 G⁺ residual, K dropped, improper Γ removed) caught, except "others in ASU(Γ⁺)" in sg 205, where ASU(Γ⁺) equals ASU(G) and it is not an error; 309 in-domain images agree with literal atoms within 4.5e-15 Å.
- **Z′ = 2:** all 230 groups (6,400 crystals). f = 1 exactly in the 22 pair groups (76, 78, 91, 92, 95, 96, 144, 145, 151–154, 169–172, 178–181, 212, 213).
- **Z′ = 3, 19 groups:** 12 descriptions per crystal (6 labellings × 2) in sg 1, 2, 4, 5, 7, 9, 14, 15, 19, 29, 33, 61, 88, 148, 198 and 205; 6 in sg 76, 92 and 169.
- **Compositions:** B·A, B·A₂ and A₂·W (W never the anchor) in sg 1, 2, 4, 9, 14, 19, 33 and 76, 150 crystals each. These give f, 2f and 2f descriptions per crystal, over 1, 2 and 2 labellings respectively. Only A₂·W's labellings are anchor copies (n_a = 2); B·A₂'s are absorbed by exchangeability.

### 3.2 Coordinates per unit

These are the same for Z′ > 1 of one compound and for co-crystals, salts and solvates.

- **q_k:** the conformer chart's internal coordinates, in the compound's canonical tree order. There is no handedness token.
- **R_k:** the seed-frame rotation. It is smooth in q wherever the seed angle is not 0 or π [derived].
  - Heavy-atom principal axes, used by today's crystal path (`align_mol_batch_to_standard_axes`), jump where moments cross, and their handedness flips. They must not be used once q is sampled [derived].
- **t_k:** the fractional heavy-atom centroid of X(q_k) [assumed: the builder, canonicaliser and measure must use the same centroid]. It satisfies ρ(σq) = D ρ(q).
- **Compound id and slot label:** discrete.

### 3.3 The anchor

**What the anchor is.** The anchor is a reference unit for bookkeeping, with no physical role (owner, 2026-10-23: "I can't think of a geometric or physical reason"). Some reference is unavoidable: the crystal's global gauge (origin and setting among normaliser-equivalent descriptions) must be fixed once, and an exchange-symmetric way to fix it either leaves the whole normaliser quotient as multiplicity (k_D · ∏ n_c!, 16 copies for one compound at Z′ = 2 in sg 14) or, if it cuts it, seams every exactly pseudo-symmetric crystal (§3.5) [derived]. The design keeps every choice of reference as an exact copy (n_a, §3.5), so nothing physical depends on which unit it is; the analogy is the root atom of a conformer tree. Within one compound the reference is the first unit in stored order, and the sampler must be indifferent to it (equivariant, or symmetrised over the n_a choices). Only in co-crystals with a singleton compound is the reference distinguished by chemistry.

**Choosing the anchor compound [derived; a discrete rule, no continuous key].**
- A compound is eligible if it has at least 3 non-collinear heavy atoms.
- Among eligible compounds, take the one with the fewest units, then the most heavy atoms, then the smallest canonical stereo-free SMILES.
- If no compound is eligible, the row is refused (Q4).

**The anchor unit** is any unit of compound a. The domain treats slot 1 as the anchor. Data rows use the first unit of compound a in stored order; labels carry no meaning (§3.5).

**The anchor's domain.** The shipped Z′ = 1 domain:
- ASU(Γ⁺) on the non-free coordinates, with the free coordinates pinned at the table gauge (`domains.gauge_free_axes`);
- R in SO(3), with the table's K cut in sg 6–9;
- q free.

*Table: n_a, the number of anchor copies, by composition [derived from the rule above]. Rows with frameless units (Cl⁻, water) are refused until Q4 is answered; their n_a applies once they are admitted.*

| composition | n_a (count) |
|---|---|
| A·B, A₂·B, A·B₂ | 1 |
| salt A·Cl⁻ | 1 |
| A₂·B₂ (ties on units) | 2 |
| A₂·H₂O (water is not eligible) | 2 |
| one compound | Z′ |

**Pair groups.** Store the lower-numbered member of each of the 11 pairs, applying the enantiomorph (§7) to rows of the higher member. The sampler is never conditioned on a higher member [derived; valid under the mirror rule only].

### 3.4 Other units, and how units relate

**The domain of unit k ≠ anchor** is the shipped cctbx ASU(G) (`group_table(sg).g_asu`, all 230 groups) × SO(3) × Q_{c(k)}. It is a fixed product domain.
- vol(ASU(G)) × |G| = 1 exactly in all 230 groups [measured, table rationals, `$Z/synthesis/synth_vol_b.json`].
- ASU(G) is an axis box in all 74 groups sg 1–74, and in 59 of sg 75–230 [measured, same script].

**Facet gluings of ASU(G):**
- translation: a circle;
- proper rotation: R → M R;
- improper element: (R, q) → (M R D, σq). This flips one unit's conformer hand.

Every facet is a seam and none is a wall, because G acts freely almost everywhere [derived].
- Measured examples (`$Z/anchored_labelled/csd_seams.json`): in sg 2 the x-faces are glued by the inversion and y, z are circles; in sg 14 the y-faces are glued by improper elements and x, z are circles.

**How units relate.** Units relate only through the anchor: a common Γ⁺ element and free shift, then an independent G-fold of each unit. Coordinates are absolute.

**What the policy reads.** It should read the geometry of the partial crystal in the anchor's frame: the poses of all G-images of the other units, relative to R₁ and t₁, within a cutoff. These features are invariant to which image of a unit was stored [derived].
- The single stored image's δ_k = R₁ᵀ B (t_k − t₁) is **not** invariant here: when the anchor crosses a facet glued by an element outside G, unit k is refolded by some g_k ∈ G.
- δ_k is invariant only in the torus alternative (§3.8), where it was measured (≤ 1.3e-15).

### 3.5 Permutations of exchangeable units: no continuous key

**Keep the labels [derived].**
- At an exact pseudo-symmetric configuration x*, some relabelling s ≠ e fixes x* (§2.2).
- For any fundamental domain F of the relabellings and any s-invariant neighbourhood U of x*, F ∩ U and s(F ∩ U) split U.
- So x* lies on the boundary of every F. **Every exact cut of the relabellings puts a seam through every exactly pseudo-symmetric crystal.**
  - This includes the §2.8 rule h(R₁) < h(R₂), ordering by fold, and conformer keys.
  - The §2.8 key gap is ≤ 1.1e-16 at such configurations [measured, `$Z/symmetric_set/zp_sym_orbits.json`].
- With labels kept, the fixed set lies in the interior: the two labelled copies merge there, with no seam.

**Real crystals sit there [measured].** From `$ZP/zp_csd_scan.json` (49,223 Z′ = 2 rows) and `$ZP/zp_csd_keys.json`:
- the pseudo-symmetry residual is below 0.25 Å in 3.76% of rows and below 1 Å in 15.96%;
- the §2.8 h-key gap is below 0.01 in 19.8% of 24,090 monoclinic rows, against 2.0% for independent rotations;
- Γ⁺-folded centroids lie within 0.25 Å of each other in 7.5% of sg 14 rows (null 0.1%).

**Z′ = 3 [derived].** A pseudo-symmetric pair whose third unit breaks the symmetry is not a fixed point, so topology does not force a seam there. Every key that is invariant under the pseudo-operation still ties on it, the §2.8 key among them.

**Exchangeable non-anchor units [derived].** Non-anchor units of one compound permute as coordinate blocks with identical domains.
- A joint sampler that is permutation-equivariant and starts from an exchangeable law keeps its density exchangeable, because equivariant Markov kernels preserve invariance.
- It therefore absorbs the (n_a − 1)! · ∏_{c≠a} n_c! copies by construction.
- An autoregressive sampler must symmetrise over them explicitly.

**What a sampler must still find:** f · n_a copies. That is 4 for one compound at Z′ = 2 (2 in the pair groups) and 2 for A·B.
- An exact-density sampler can symmetrise over them, p_sym(x) = (f n_a)⁻¹ Σ_s p(s·x), where s ranges over the re-anchorings and the anchor's improper copy. This costs f · n_a density evaluations and canonicalisations per density, and needs no Jacobian.
  - KL(p_sym ‖ π) ≤ KL(p ‖ π) for any H-invariant π, by convexity [derived].
- A path-space sampler cannot shed copies by a final uniform relabelling. Trajectory balance with a uniform backward policy forces the forward marginal to cover every copy; shedding them needs a learned backward over the f · n_a choices [derived].

**Where a seam-free discrete key exists, use it:**
- compound identity (the anchor and the compound order);
- the space-group label within an enantiomorphic pair.

The stereo parity is **not** seam-free: it is a sign in q and a hypersurface at planarity (Q3).

**Required data handling [derived].**
- MLE and prior data must be augmented over the m labelled forms (or sampled uniformly from them). Otherwise the data density is a cut density that the sampler does not target.
- The trainer's raw-latent unit-ordering penalty is a continuous-key soft wall and must go (`crystal_units_audit.md` stage 6).

### 3.6 Co-crystals, salts, solvates

One machinery covers them: compound ids, π restricted within compounds, the anchor rule of §3.3, and ASU(G) for every other unit.
- The composition is the condition identity.
- The reward basis per formula unit is audit Q3, outside this note.
- Units without a frame are refused for now (Q4):
  - monatomic ions have no R;
  - linear units have S² in place of SO(3);
  - water H atoms are often absent.

### 3.7 Rigid limit (torsion tier, or fixed conformers)

With the parity frozen, each unit carries a discrete hand h. The domain is §2.8's without its swap rule [derived; measured for rigid units in `fd_tooling.md` Table 5]:
- the anchor at h = +1 in ASU(Γ⁺) (f = 1; in the 11 enantiomorphic pairs exact only when both members of the pair are sampled, as Q3(b) says);
- in non-Sohncke groups, units k ≥ 2 at h = +1 in ASU(G⁺). These are new tables; cctbx's asymmetric unit of G⁺ was prototyped in `$ZP/zp_checks.py::gplus_region` [code].
- in Sohncke groups, h_k is a free per-unit discrete coordinate (kryptoracemates, mixed hands).

### 3.8 Strongest alternative (owner choice Q1): anchor-relative torus, images averaged out

This alternative keeps everything in §3.1–3.6 except the domain of units k ≠ anchor.

**Coordinates:**
- Δt_k = t_k − t₁ mod L_prim, the primitive lattice, so the centring translations drop out;
- ΔR_k = R₁ᵀ R_k;
- q_k.

**Multiplicity.** Each unit's G/T_prim images, including the σ-flipped ones, stay as multiplicity: m_A = f · |G/T_prim|^(Z′−1) · ∏ n_c! · ∏ a_c.
- Measured on the conventional torus [`$Z/efficiency/zp_design_orbits.json`]: 8 in sg 2, 16 in sg 14, 16 in sg 19, 32 in sg 9 (K kept; 16 with K cut).
- [derived] On the primitive torus, sg 9 (K kept) and sg 15 both drop to 16.

**Seams.** None for units k ≠ anchor: the torus has no boundary. δ_k = R₁ᵀ B Δt_k (minimum image) and ΔR_k are invariant under every proper anchor gluing [measured ≤ 1.3e-15].

**Requirements:**
- an exact-density sampler, symmetrised over the m_A images;
- conditioners that see earlier units only through image-invariant quantities. The per-unit image sums then factorise, at f · n_a · (1 + (Z′−1)|G/T_prim|) evaluations: 20 for sg 14 at Z′ = 2 [derived].
- Without symmetrisation, a sampler must learn every copy. The SDE GFN is in this position.

### 3.9 Rejected

- **Voronoi placement** (each unit at its G-image nearest the anchor). It is exact (1 per labelling in 19 groups, `counts_z2a/b.json`) but puts seams through real mass (§5), and its polytope changes with t₁ and the cell.
- **The symmetric form with D × S kept** (no anchor). It leaves k_D · ∏ n_c! copies (16 for sg 14 at Z′ = 2), and it is efficient only for a policy equivariant under the whole group, which does not exist yet.
- **Any S-invariant cut of D.** By the same boundary argument it seams through every pseudo-symmetric configuration [derived; measured as 100% ties against 0% for generic configurations in 5 of the 6 runs of `zp_sym_orbits.json`; in sg 4 with the conformer-carried hand, generic configurations tie in all 7 classes too, so there the cut fails everywhere, not only at pseudo-symmetry (independent verification, `$Z/verify/`)].
- **The §2.8 swap rule**, for the reasons in §3.5.

## 4. Measure and log Z accounting

**Reference measure [derived]:**

  μ = μ_cell(dc) × [V dᵏt₁ · Haar(dR₁) · μ_Q(dq₁)] × ∏_{k≠1} [V d³t_k · Haar(dR_k) · μ_Q(dq_k)]

- k is the number of non-free axes, with the anchor's free coordinates at the gauge.
- μ_cell is the owner's O4 (a) choice.
- μ_Q is the conformer chart's measure, fixed by the conformer refactor [assumed]. Its Cartesian-induced Jacobian depends only on bond lengths and angles.

**The gauge slice carries Jacobian 1 [derived].** Changing (t₁^free, t_k^free) → (u = t₁^free − gauge, t_k^free − u) has Jacobian 1. The common-shift orbit has fractional length 1 per free axis, which gives `crystal_fd/measure.py`'s Z′ = 1 convention, V dᵏt, on the slice.

**H preserves μ [derived]:**
- |det W| = 1;
- R ↦ M R D equals (−M) R (−D), a left and a right multiplication by proper rotations, so Haar is preserved;
- σ negates coordinates and leaves the bond-length and angle Jacobian unchanged. Transverse rows depend on |(u, v)|;
- permutations preserve product measures.

**Domain volumes (fractional):**
- the anchor centroid has 1/|Γ⁺/K|, proved by `certify.py` (gate 0);
- ASU(G) has 1/|G| [measured: table rationals, all 230 groups, `synth_vol_b.json`].

**log Z [derived; m measured where stated]:**

  Z_dom = ∫_dom e^(−E/T) dμ = m · Z_cryst,  m = f · ∏_c n_c! · ∏_k a_{c(k)},

exact almost everywhere: the exceptions are exact pseudo-symmetry, achiral conformers and collisions.

- **a_c** = |Aut(G_c)| relabelled chart points per generic conformer [derived from the chart being reduced neither by reflection nor by permutations of identical atoms (wiki conformer-chart-and-internal-coordinates); unmeasured]. Examples: 6 per methyl (S₃), 2 per phenyl ring flip. It is constant per compound, so it matters only when compounds or Z′ classes are compared.
  - Data augmentation over Aut belongs to the conformer route.
- **Training within one (space group, composition) class needs no m:** the target on the domain is e^(−E/T)/Z_dom.
- **Comparing classes or parameterisations:** log Z_cryst = log Z_dom − log f − Σ_c log n_c! − Σ_k log a_{c(k)}.
  - For one compound at Z′ = 2 in sg 14 the offset is log 4 + 2 log a. For A·B it is log 2 + log a_A + log a_B.
  - Z′ = 1 with joint conformers has m₁ = f · a_c, so a run on today's domain carries log 2 in 208 groups.
  - A sampler that chooses a Z′ class or composition must weight it by Z_dom/m.
- **The audit's pad-slot reward term** (0.2474 T per pad slot; `crystal_units_audit.md` §7 rounds it to 0.248) is a spurious per-class offset and must go: pad slots carry no measure.
- **The trainer's centroid factor** (V/sym_mult)^Z′ differs from this convention by a constant per (space group, Z′) [derived]. The constant is not measured; reconcile it when the trainer changes.
- **Alternative A:** m_A replaces m (§3.8). The torus coordinates have Jacobian 1: dΔt by translation, dΔR by Haar left-invariance.

## 5. Seams and efficiency

*Table 1. Fraction of CSD Z′ = 2 configurations within 0.1 Å of each kind of seam.*
- **Data:** rows of the ingest store whose operators equal `SYM_OPS[sg]`, both labellings pooled. The anchor is folded into ASU(Γ⁺).
- **Null:** the non-anchor unit's centroid replaced by a uniform random point, with the same anchor and cell.
- **Source:** `$Z/anchored_labelled/csd_seams.py` → `csd_seams.json` and `joint_seams.py` → `joint_seams.json`.
- **How to read it:**
  - Column 3 is what the recommended design pays at the other unit's own facets.
  - Column 5 is what it pays when the anchor's move forces a refold.
  - Column 6 is what alternative A pays in fractional coordinates; A pays 0 in anchor-frame coordinates.
  - The anchor's own seams are common to every design.

| sg | rows (count) | other unit at an ASU(G) facet: observed / null (fraction) | other unit at a Voronoi face: observed / null (fraction) | anchor at a facet glued outside G, refolding the other unit in R: observed = null (fraction) | anchor at a twisted facet: observed = null (fraction) |
|---|---|---|---|---|---|
| 2 P-1 | 15,178 | 0.041 / 0.042 | 0.335 / 0.058 | 0.097 | 0.000 |
| 14 P2₁/c | 8,482 | 0.074 / 0.065 | 0.286 / 0.062 | 0.125 | 0.065 |
| 4 P2₁ | 7,918 | 0.037 / 0.029 | 0.299 / 0.061 | 0.117 | 0.081 |
| 19 P2₁2₁2₁ | 3,356 | 0.091 / 0.067 | 0.252 / 0.061 | 0.179 | 0.164 |

**Reading Table 1 [measured unless marked].**
- **The ASU(G) facets carry mass at about the null rate.** The CSD prefers the Voronoi faces 4.1- to 5.8-fold over the null.
  - The cause is pseudo-operations whose square is a nontrivial element of G. They put the partner exactly on the bisector with an anchor image [derived].
  - The efficiency design's independent sample (2,842 rows from 89 chunks; `$Z/efficiency/zp_seams_csd_n90.log`) agrees: Voronoi 31.4% against a null of 6.0% within 0.1 Å, and a 0.25 Å perturbation crosses it in 17.3% of rows against 3.7%. In that sample, rows with a pseudo-translation RMS below 0.5 Å have a median Voronoi seam distance of 0.06 Å.
- **The cost of R.** The recommended design pays both the column-3 and the column-5 terms. Counting either seam
  (`joint_seams.json` `table_joint`, 0.1 Å), 12.5%, 18.4%, 14.2% and 23.9% of configurations in sg 2, 14, 4 and 19, against
  nulls of 13.6%, 18.2%, 14.6% and 23.1%: at the null rate, with no excess from the CSD's own geometry (independent
  verification, `$Z/verify/`). Counted the same way, the Voronoi option is 32.9–37.0%.
  - At these joint seams, the anchor's move refolds the other unit by a G element.
  - In sg 2 every anchor facet is a half-translation. Those facets are invisible at Z′ = 1 (circles) but move the other unit's absolute coordinates.
- **Alternative A** removes the column-3 and column-5 terms. Writing Δt in the anchor frame also removes column 6 for units k ≠ anchor. The anchor's own seams remain in every design.

*Table 2. Copies of each crystal at one compound, Z′ = 2, by representation.*
- **Source:** [derived] from §3 and §4; Aut copies are excluded.
- **Measured checks:** the domain counts for R (`synth_z2_*.json`), A on the conventional torus (`zp_design_orbits.json`) and the symmetric form (`zp_sym_orbits.json`: sg 14 16, sg 19 32).
- **How to read it:** "left to learn" counts the copies that remain after a permutation-equivariant joint sampler; "evaluations" is the cost per density with exact symmetrisation.

| representation | sg 2: in domain / left to learn / evaluations (count) | sg 14 (count) | sg 19 (count) |
|---|---|---|---|
| R (recommended) | 4 / 4 / 4 | 4 / 4 / 4 | 4 / 4 / 4 |
| A (torus, primitive) | 8 / 8 / 12 | 16 / 16 / 20 | 16 / 16 / 20 |
| symmetric D × S (rejected) | 16 / 8 / – | 16 / 8 / – | 32 / 16 / – |

**Efficiency [derived; cost per energy call not measured].**
- **R.** It needs no symmetrisation to be exact, and it works with the SDE GFN. Its costs are the joint seams (Table 1, column 5) and the σ-glued facets, which make a non-local jump in q whenever the conformer is far from achiral.
- **A.** It has no seams for the other units, but only an exact-density sampler can afford it.
- Gate G9 measures both.

## 6. Interaction with per-unit internal coordinates

1. **No cut reads q.** σ enters only as:
   - a gluing on the improper facets of ASU(G) (R);
   - the anchor's multiplicity f;
   - A's image multiplicity.

   No key orders units. Each unit's conformer block is independent of the others' [derived].
2. **The body frame is the seed frame** and the centroid is that of X(q). Both are smooth in q. The rigid path's per-unit h is dropped, and the body reflection is D = diag(1, 1, −1), not crystal_fd's x-reflection diag(h, 1, 1) [code: seed plane xy].
3. **Mirror conformers are common.** In 44.5% of Z′ = 2 rows, unit 2 is closer to unit 1's mirror image than to unit 1; the median heavy-atom RMSD between units is 0.19 Å, and 90% are below 0.76 Å [measured, `$ZP/zp_csd_scan.json`].
   - With the hand in q these pairs need no label.
   - In non-Sohncke groups the improper G image lets a unit be represented in its same-hand form.
4. **Today's PCA handedness is a frame artefact, not a chemical label.** It differs between the two units in 51.9% of no-stereo rows and 38.6% of identical-stereo rows [measured, `csd_seams.json` part1].
5. **The stereo lock (`stereo_coeff` > 0; 300 in `configs/conformer_mk.yaml`) is not σ-invariant.** It pins absolute parity per unit, so a per-unit improper image of the same crystal would score differently [derived]. With the lock on, the representation becomes the rigid one of §3.7. This is Q3.
6. **An intramolecular energy is required before joint sampling** (audit Q3). It must be mirror-invariant and invariant under atom relabelling; MMFF is both.
   - It also decides whether a stereocentre can invert. The representation stays exact either way, because both parities are in Q and σ is an equivalence.
   - If the term empties the planar region, each parity is a separate basin of Q. A tagged anchor's f copies then sit in mirror basins that a local sampler cannot move between. That is where Q3 (c)'s parity cut costs nothing.
7. **Storage.** One compound uses padded per-unit q `[B, Zmax, d]`; several compounds use the audit's stage-3 unit level (`crystal_units_audit.md` §4).
8. **Conformer code is consumed, not changed.** The FD work reads the conformer builder and charts (the refactor owns them).

## 7. Canonicaliser and charts

**Input.** An ingested row in the standard setting (D1), with:
- the atom-to-unit map `unit_ind` (audit stage 2, built);
- per-unit SMILES;
- positions.

Units on special positions are refused, as they are today.

**Steps.**
1. **Per unit.**
   - Compound key: the canonical stereo-free SMILES.
   - q_k: from the compound's canonical tree, with no handedness token.
   - R_k and t_k: from the seed frame and the heavy-atom centroid.
   - Check that rebuilding from (t_k, R_k, q_k) reproduces the unit's atoms.
   - Units with fewer than 3 non-collinear heavy atoms are refused as anchors (Q4).
2. **Cell step.** crystal_fd's Z′ = 1 cell step, with its basis change (N, s) applied to every unit: t → N⁻¹(t − s), R → Uᵀ R.
3. **Pair groups.** If sg is the higher-numbered member of a pair, apply the enantiomorph to every unit and take the partner group:
   - t → −t − p;
   - R → −R D;
   - q → σq.

   This is `enantiomorph.py`'s map, rewritten for the seed-plane reflection [derived from §2.1].
4. **Anchor.** Pick compound a by the rule in §3.3; the anchor is its first unit in stored order.
5. **Anchor fold.** `crystal_fd.domains.fold` on the anchor as a Z′ = 1 row: Γ⁺, the K cut, the free gauge and the cctbx tie trees. It returns the element γ = (W, w), the lattice shift L and the free shift v.
6. **Move the others.** Apply γ and v to every other unit: t → W t + w + L + v, R → M_W R, q unchanged (γ is proper).
7. **Fold the others.** Fold each other unit into ASU(G) with the table's tie cuts, reusing `domains._fold_block` with a section built from G and `g_asu`. An improper g gives R → M R D and q → σq.
8. **Output.**
   - The description, with `unit_mask`, compound ids and slot labels.
   - The records (γ, L, v; per unit g_k, L_k) and a status. `z_prime_gt1` is lifted only for rows with per-unit q.
   - A pseudo-symmetry flag: fewer than f · n_a distinct anchored forms within tolerance, with the best supergroup element recorded. It is never applied (owner 2026-10-22).
9. **`labellings(desc)`.** The m labelled forms:
   - each re-anchoring within compound a, then steps 5–7 again;
   - every order of the other units within compounds;
   - the anchor's improper copy where f = 2.

   It is used to augment training data. At large Z′ it samples the forms rather than enumerating them.
10. **Equality and dedupe by orbit matching.** For each of Y's f · n_a anchored forms, match Y's other units to X's with a Hungarian assignment within each compound on per-unit distances (t after the fold, R, q); X equals Y when one match is within tolerance. This costs f · n_a canonicalisations, which is feasible up to the CSD's Z′ = 32. The lexicographic minimum over the m forms is a usable hash for data only, never a sampler domain.

**Charts.**
- **Cell:** `CellChart`, shipped for every family.
- **Anchor centroid:** `PolytopeARChart(sg)` (shipped).
- **Anchor orientation:** the deferred SO(3) chart (O3). In sg 6–9 it is the shipped double cover with K-invariance (`orientation_cover='double'`).
- **Other centroids:** a new AR chart of ASU(G), with Fourier–Motzkin bounds from `g_asu`, a circle on each translation-glued facet pair, and σ-twisted facets listed as seams. It is a box in sg 1–74.
- **Other orientations:** the same SO(3) chart.
- **Conformers:** the conformer refactor's full-tier chart.
- **log-det:** the sum of the pieces.
- **Alternative A** adds a torus chart (Δt mod L_prim, ΔR) and a symmetrisation helper.

## 8. Verification gates

Each gate states the number it must produce. G1–G3 and G5 need no conformer code; they use a σ-odd stand-in, as `counts.py` does.

- **G1. Orbit counts.** Exactly f images per labelling in the domain.
  - Z′ = 2: all 230 groups. f = 2 in 208 groups, and 1 in the 22 pair groups.
  - Z′ = 3: sg 1, 2, 4, 5, 7, 9, 14, 15, 19, 29, 33, 61, 88, 148, 198 and 205 give 12 descriptions per crystal; sg 76, 92 and 169 give 6. The list includes the sg 6–9 family (7, 9) and free-axis groups (1, 4, 5, 7, 9, 29, 33, 76, 169).
  - Compositions B·A, B·A₂ and A₂·W: f, 2f and 2f descriptions per crystal.
  - Prototyped on synthetic crystals [measured, §3.1]. The gate re-runs them through the library code, not the harness.
- **G2. Literal atom oracle.**
  - Float64 atoms of every enumerated image, built from q with the conformer builder, equal the reference crystal (or its mirror) within 1e-9 Å.
  - X(σq) = D X(q) to float precision.
  - The efficiency prototype reached 6.1e-15 Å with rigid stand-ins.
- **G3. Certificate for ASU(G).** `certify.py` extended to `g_asu` in exact rationals: the group check, vol × |G| = 1, and facet coverage, in all 230 groups. The table volumes already give exactly 1 [measured].
- **G4. The canonicaliser is a function of the orbit.** All |Γ| · |G|^(Z′−1) · ∏ n_c! images of a crystal map into the same set of m labelled forms. Not yet run: it needs stage Z3's code. The first run should cover Z′ = 2 in sg 2, 9, 14 and 19 and Z′ = 3 in sg 14.
- **G5. Round trip.** Canonicalise, then unfold, reproduces the atoms within 1e-9 Å on synthetic rows and within 1e-3 Å on CSD rows.
- **G6. Measure.**
  - Haar is invariant under R → M R D: KS p uniform.
  - μ_Q is invariant under σ.
  - The Monte Carlo ratio Z_dom/Z_cryst equals m on a toy H-invariant target.
- **G7. Energy.** eLJ and MACE agree, to float32 build noise, for descriptions related by:
  - per-unit G, with σ realised through the conformer builder;
  - Γ⁺;
  - relabelling;
  - the f copy.

  Run it through the MXtalTools per-unit route (audit stages 0–5, built), at Z′ = 2 with per-unit conformers. The control is the same image with σ omitted, which must differ at the scale of `fd_tooling.md` Table 6.
  - UMA is expected to fail only on the f copy, and only in the 43 Sohncke groups with improper N_E.
- **G8. CSD Z′ = 2.**
  - Canonicalise the rows from the ingestion chain.
  - Count distinct canonical forms per refcode family.
  - Report the fraction of rows flagged pseudo-symmetric, and the ASU(G) seam fraction against Table 1.
- **G9. A short real run at production-shaped parameters** (AGENTS verification rule 5), on a one-compound Z′ = 2 prior from the ingestion chain, after audit stage 6. It compares, in convergence per energy call:
  - R against A on an exact-density sampler;
  - an exchangeable against a non-exchangeable non-anchor model;
  - the f · n_a copies symmetrised against left to learn;
  - an L⁺-torus anchor with P⁺ kept as multiplicity (no anchor seams; an O1-type variant).

## 9. Staged build plan in crystal_fd

Each step lands with its gate. Nothing changes the Z′ = 1 outputs (the `csd_fd_v5` rows stay bit-identical).

- **Z0. Tests first.** Port the counting harness into `tests/fd` as G1, run on synthetic σ-odd stand-ins. No library change.
- **Z1. σ and q in the data model.**
  - `CrystalDescription` gains optional per-unit `q` `[B, Zmax, d]`, a per-compound σ mask (the rows that negate) and compound ids.
  - `ops.apply` gains the improper per-unit action (t, R, q) → (Wt + w, M R D, σq).
  - Gate: G2.
- **Z2. The ASU(G) fold for one unit.** `g_asu` with its tie cuts, with σ on improper elements. Gates: G3, and exactly one image per unit.
- **Z3. The anchored canonicaliser, one compound.** It lifts `z_prime_gt1` for rows with q. Gates: G1 through the code, G4 and G5.
- **Z4. Labellings, equality and multiplicity.**
  - `labellings`, orbit-matching equality and the pseudo-symmetry flag.
  - `measure.log_multiplicity` and `images_per_crystal` for Z′ > 1.
  - Gate: G6.
- **Z5. Charts.** The ASU(G) AR chart and the product chart with log-det, consuming the conformer chart read-only. Gate: log-det against autograd.
- **Z6. Several compounds.** Compound ids, the anchor rule, and π within compounds. Gate: G1's compositions.
- **Z7. CSD Z′ > 1 rows** with per-unit internal coordinates from the ingestion chain. Gates: G5, G7 and G8; output a Z′ > 1 canonical dataset.
- **Only if Q1 = A:** Z5a, the torus chart and symmetrisation helper. Gate: p_sym invariant under all m_A actions, and importance-sampled log Z_dom − log m_A recovering a toy log Z.
- **Outside crystal_fd:** audit stages 6–8 in GFN, then G9.

## 10. Questions for the owner

**Q1. How are units other than the anchor placed?** Recommended: (a).
- **(a) Absolute ASU(G) cut.** The shipped `g_asu` gives an exact domain for any sampler, the SDE GFN included.
  - Costs: the joint seams of Table 1 columns 3 and 5 (12.5–23.9% of configurations within 0.1 Å, at the null rate of 13.6–23.1%), and σ-glued facets that flip one unit's conformer.
- **(b) Anchor-relative primitive torus with the G images averaged out (§3.8).** No seams for those units, and the contact motif stays one blob.
  - Requires an exact-density sampler and image-invariant conditioners.
  - Copies grow as |G/T_prim|^(Z′−1).
  - It cannot serve the SDE GFN without that sampler learning every copy.
- **(c) Nearest image to the anchor (Voronoi).** Exact, but 25–34% of real configurations sit within 0.1 Å of its seams (null 6%).
- **Default if unanswered:** (a). Build (a); run G9's R-against-A comparison once an exact-density sampler consumes Z′ > 1.
- **Owner, 2026-10-23:** (a), "Agreed this is the one at least for now."

**Q2. The anchor's improper copy (f = 2 in 208 groups): keep it, or cut it?** Recommended: (a).
- **(a) Keep it as an exact multiplicity**, and symmetrise over it in exact-density samplers (2 evaluations).
  - No new tables.
  - A log 2 offset, and the same factor at Z′ = 1 with joint conformers.
- **(b) Cut it with ASU of the full Γ** (new cctbx tables).
  - Exact by a centroid cut in 159 groups.
  - In the 49 polar groups whose improper kernel element fixes the projected centroid, the cut must go in (R₁, q₁), a seam that flips every unit's conformer.
  - Every improper facet of the anchor domain flips every unit's hand.
- **Default if unanswered:** (a).

**Q3. The stereo lock and stereocentres in the crystal route.** Recommended: (a).
- **(a) Lock off (`stereo_coeff` 0) and no parity cut.** Every compound's hand lives in q, and every σ-equivalence is exact.
  - One code path.
  - Mixed-hand units and kryptoracemates (538 Sohncke Z′ = 2 rows with mutually enantiomeric units) are represented.
  - Whether a stereocentre can invert is left to the intramolecular energy.
- **(b) Lock on (as in `conformer_mk.yaml`, 300).** The representation becomes the rigid one of §3.7:
  - ASU(G⁺) per unit (new tables), and f = 1;
  - both members of each enantiomorphic pair sampled;
  - kryptoracemates need a per-unit reference parity label.
- **(c) A parity cut instead of the lock.** Fix the sign of one designated stereocentre per unit, a σ-half that keeps the diastereomers. Apply it where an intramolecular term empties the planar region.
  - This removes f for tagged anchors and the σ-glued facets for tagged units.
  - The cut is a wall in q at planarity.
- **(d) Split by the kind of chirality (proposed 2026-10-23, after the owner's direction below).**
  - **Configurational chirality** (stereocentres, E/Z bonds, slow atropisomers; the '@'-carrying units, about half of CSD Z′ = 2 rows, §2.3): a discrete per-unit label from the molecular graph, kept by the stereo lock and conditioned on. Racemates, kryptoracemates and homochiral sets become explicit conditions. Fixing the anchor's label cuts f to 1; a labelled non-anchor unit folds by G⁺ (ASU(G⁺) tables, as in §3.7), an improper element swapping its label.
  - **Conformational chirality of achiral graphs** (no stereocentre; 46.7% of rows, e.g. a twisted biphenyl): no label, because any label is a continuous key on q with a wall at the achiral conformations; the hand stays in q (ASU(G) with σ gluings), and f = 2 stays an exact multiplicity.
- **Default if unanswered:** (a).
- **Owner, 2026-10-23:** "I think we have to attach handedness to each individual conformer, and then probably condition on it downstream"; for conformer sampling the owner had planned chirality as a condition, not a free learnable parameter. On (d)'s split ("did you mean the configurational label only, or also a label on conformational hand?"): "Good point to consider", "I'm not 100% sure." **Paused by the owner (2026-10-23, "Can we pause, or sidestep?").** Parked default: (d), configurational labels only. Nothing built or planned through Z3 depends on the answer: the built path (hand in q, ASU(G) with the mirror) is exact for every molecule, and labelled units are an additive extension (ASU(G⁺) tables, a label-reading fold). Q2 follows from the answer: under (d), f is cut for labelled anchors and kept for achiral ones.
- **Points to carry into the decision [derived; from the 2026-10-23 discussion].**
  - Fold (position, hand) together: an improper element moves a unit and flips its hand. Points and achiral units fold into ASU(G); a chiral unit with a free hand into ASU(G) with the mirror; a chiral unit with a fixed hand into ASU(G⁺), the extended unit (twice ASU(G) in the 165 non-Sohncke groups). ASU(G) × {both hands} and ASU(G⁺) × {+} cover the same crystals.
  - Z′ = 1 with every molecule conditioned at +1 is the existing +1-sheet construction, ASU(Γ⁺) × SO(3), f = 1.
  - Z′ > 1 in non-Sohncke groups: each unit's hand can be swapped independently by its improper image, so per-unit labels are not properties of the crystal; the canonical form writes every unit as +, and a chirality condition always reads "all +" there (mixed labels would count each crystal several times). A compact A+···B− motif is written as A+ with B's + image, possibly far away.
  - Z′ > 1 in Sohncke groups: hands are properties of the crystal up to a global flip; the condition is the hand pattern modulo that flip (homochiral against kryptoracemate, 538 CSD Z′ = 2 rows).
  - A per-conformer hand for flexible achiral molecules: any sign that σ negates must, for most molecule types, also vanish at chiral conformations (Ruch's homochirality problem; Mislow and Weinberg's "chiral zeros"), and its zero set is a seam in q. Today's principal-axes handedness is a frame artefact (it disagrees between the two units in 51.9% of no-stereo Z′ = 2 rows, §6). If such labels are wanted, first measure the CSD and QM9 conformer mass near a candidate sign's zero set.

**Q4. Units without a full heavy-atom frame (water, monatomic ions, linear units).** Recommended: (a).
- **(a) Refuse such rows with a status** until reduced pose spaces are designed. Hydrates and halide salts wait.
- **(b) Admit water with H atoms as a full-frame non-anchor unit.**
  - SO(3) orientation, C2 not folded.
  - Rows with missing water H atoms are still refused.
- **(c) Design point and S² pose spaces now**, enlarging stage Z6.
- **Default if unanswered:** (a).
- **Owner, 2026-10-23:** "these can wait until way, way, down the line." Rows with frameless units stay refused.

## 11. What this supersedes in `fd_tooling.md` §2.8

- **"each gᵢ ∈ G chosen so the unit stays proper"** assumed a fixed hand. With conformers in q, every g ∈ G applies: improper ones apply σ.
- **"Unit 2: in ASU(G⁺) × SO(3)"** becomes ASU(G) × SO(3) × Q. ASU(G⁺) leaves 2 images per non-anchor unit in non-Sohncke groups (§2.4).
- **The swap rule h(R₁) < h(R₂) is dropped.** It is a continuous key, and it seams every pseudo-symmetric configuration (§3.5). Owner decision O6 is settled by the owner's answer of 2026-10-22: (a), keep labels.
- **Unit 1 stays** in FD(Γ⁺) with its free coordinates pinned and K cut. It is now chosen by the discrete compound rule (§3.3).
- **Table 5 stays valid** for rigid units on the +1 sheet (§3.7, the torsion tier). It does not describe jointly sampled conformers.
- **New consequence:** with joint conformers the Z′ = 1 domain of §2.2 is a double cover in 208 groups (§2.4 item 5).
- **§7's open gap** "Z′ = 2 for P⁺ other than C1, C2 and D2; Z′ > 2" is closed for counting on synthetic crystals by §3.1 (all 230 groups at Z′ = 2; 19 at Z′ = 3). It stays open for charts and energies.

## 12. Errors in the inputs, as fixed here

- **Anchored design:**
  - It called the stereo label a "seam-free" key. The label is a sign in q with a hypersurface at planarity, and "label = reference" also drops diastereomers (Q3 (c) fixes both).
  - It left a_c out of log Z (§4).
  - It bounded m ≤ 24 at Z′ ≤ 4. The bound is 48 with f = 2.
  - Its claim "n_a = 1 for solvates" fails for A₂·H₂O, where n_a = 2.
  - It assumed rather than derived the pinned free-axis measure (§4 derives it).
- **Symmetric-set design:**
  - "D × S_{m−1} leaves m copies" orders the non-anchor units, a continuous key at Z′ ≥ 3; labels are kept instead.
  - The admissible-subgroup list is a necessary condition, not a construction.
  - Its discrete-anchor kernel cut in the 49 polar groups is σ-glued (Q2 (b)).
- **Efficiency design:**
  - Its fallback ASU(N_E) anchor is not a fundamental domain in the 49 polar groups.
  - It called ASU(G⁺) an exact fallback; ASU(G) is the exact one (§3.4).
  - Its A·A·B count in sg 9 (64) was measured with K cut, while its recommendation keeps K (128 on the conventional torus).
  - Its M for sg 9 and 15 used the conventional torus; on the primitive torus both are 16.
  - "0 seams" for the torus holds only in anchor-frame coordinates; fractional Δt jumps by W at the anchor's twisted facets (Table 1, last column).
  - It omitted a_c.
  - It attributed its 13% of differing units to tautomers; they are mostly enantiomers and stereo variants (§2.3).
  - "Never average f with UMA" applies only in the 43 Sohncke groups.
  - Its cost formula counted Z′! relabellings where exchangeability leaves n_a.
- **Both reviews:** the anchor-frame feature δ_k of a single stored image is invariant only on the torus, not under the ASU(G) placement (§3.4).
