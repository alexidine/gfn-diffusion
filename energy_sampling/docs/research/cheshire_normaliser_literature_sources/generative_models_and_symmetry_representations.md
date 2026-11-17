# Generative models for crystals: representation and quotienting of crystal-symmetry redundancy

Scope note: all findings below come from fetched primary sources (arXiv HTML/abstract pages, PMLR pages, or PDF text extracted locally) unless marked otherwise. The fetch tool summarises pages with a small model, so short quotations are as reported by that summariser; the substance was checked against multiple passages where possible. Several arXiv IDs are from 2026 (e.g. 2602.x, 2606.x, 2607.x, 2609.x); some pages showed inconsistent dates (header vs arXiv stamp), noted where seen.

## Q1. How do inorganic crystal generative models represent crystals and handle cell, origin, permutation and rotation redundancy?

### Takeaway
Inorganic models split into two families. (a) Full-cell "joint diffusion/flow" models (DiffCSP, FlowMM, MatterGen, ADiT) work in P1 on a lattice plus fractional coordinates. They handle rotation by invariant lattice features or symmetric/polar-decomposed lattices, permutation with GNNs/sets, and origin by periodic-translation invariance on the torus. Lattice-basis choice is handled by **canonicalising to a Niggli-reduced cell**, not by invariance. (b) Symmetry-aware models (CrystalFormer, SymmCD, WyckoffDiff, Wyckoff Transformer, space-group-equivariant diffusion) represent the asymmetric unit / Wyckoff orbits plus the space group, in the conventional cell. Only the Wyckoff-token line (Kazeev, Hippalgaonkar groups) explicitly treats **Euclidean-normalizer non-uniqueness**, and it does so by augmentation and enumeration, not by quotienting.

### Cited Findings
**DiffCSP** (Jiao, Huang, Lin, Han, Chen, Lu, Liu; arXiv:2309.04475; NeurIPS 2023 [venue from prior knowledge; the fetched page shows only arXiv v2, revised 7 Mar 2024])
- The paper defines three symmetries: permutation invariance (handled by a GNN backbone), O(3) invariance ("any rotation/reflection of L keeps the distribution unchanged"; it uses the Gram matrix LᵀL in messages and left-multiplies L for equivariant lattice noise), and **periodic translation invariance** (translation of F, wrapped back into the cell). A Fourier transform of relative fractional coordinates gives that invariance — [DiffCSP HTML](https://arxiv.org/html/2309.04475)
- The fractional domain is called "a quotient space" ℝ^{3×N}/ℤ^{3×N}, realised as [0,1)^{3×N} via w(F) = F − ⌊F⌋. "Fundamental domain" is not used — [DiffCSP HTML](https://arxiv.org/html/2309.04475)
- Lattice-basis choice: footnote 1 reads "we apply the Niggli reduction on the primitive cell as a canonical scale representation of the lattice vectors." Other basis choices are not discussed — [DiffCSP HTML](https://arxiv.org/html/2309.04475)

**FlowMM** (Miller, Chen, Sriram, Wood; arXiv:2406.04713; ICML 2024)
- The abstract says the method generalises Riemannian flow matching "to suit the symmetries inherent to crystals: translation, rotation, permutation, and periodic boundary conditions" — [FlowMM abstract](https://arxiv.org/abs/2406.04713)
- Lattice: it uses six lattice parameters (lengths and angles), which "carry all non-orientation information about the unit cell" (rotation invariance by representation). It acknowledges "freedom to choose an alternative unit cell and corresponding fractional coordinates" and resolves this by taking "the unique one determined by Niggli reduction". That is why angles are restricted to 60°–120° — [FlowMM HTML](https://arxiv.org/html/2406.04713)
- Translation: the base distribution is uniform on the torus. The average torus translation is removed from the tangent-space conditional vector-field targets, so the marginal path "ends up translation invariant". The target concentrates "on an equivalence class of" translated fractional coordinates. The unit cell is called "the repeating fundamental volume element"; "quotient" and "fundamental domain" are not used — [FlowMM HTML](https://arxiv.org/html/2406.04713)

**MatterGen** (Zeni et al.; arXiv:2312.03687; Nature 2025, doi 10.1038/s41586-025-08628-5)
- The public model card says: "We use the Niggli reduction to preprocess the unit cell lattices, followed by the polar decomposition to ensure the lattice matrices are symmetric matrices" — [MatterGen HF model card](https://huggingface.co/microsoft/mattergen)
- Lattice diffusion runs on symmetric matrices "to ensure rotational invariance". Coordinate diffusion runs on fractional coordinates on a flat torus with a wrapped normal. The score network outputs equivariant scores (paper Appendix A.8) — as quoted in search snippets of [arXiv 2312.03687](https://pdf.arxiv.org/pdf/2312.03687); the appendix text was not read directly.

**ADiT, All-atom Diffusion Transformer** (Joshi, Fu, Liao, Gharakhanyan, Miller, Sriram, Ulissi; arXiv:2503.03965; ICML 2025)
- "We use Niggli reduction to uniquely determine the unit cell parameters" (citing Grosse-Kunstleve et al. 2004). The lattice is converted to (a,b,c,α,β,γ) with angles in [60°,120°] — [ADiT HTML](https://arxiv.org/html/2503.03965)
- The model is a non-equivariant Transformer that "learn[s] symmetries via data augmentation" (random rotations and translations). An equivariant EquiformerV2 variant did worse in ablation — [ADiT HTML](https://arxiv.org/html/2503.03965)

**CrystalFormer** (Cao, Luo, Lv, Wang; arXiv:2403.15734; accepted Sep 2025, journal name not visible on the fetched page)
- Representation: a sequence of symmetry-inequivalent atoms (Wyckoff letter, species, free fractional coordinates), e.g. "167-a-C-0-0-1/4-b-Ca-0-0-0-e-O-0.257-0-1/4". Atoms are ordered alphabetically by Wyckoff letter. The orbit representative is "the first of fractional coordinates that appear in the international tables"; ties are broken by "lexicographic order of fractional coordinates" — [CrystalFormer HTML](https://arxiv.org/html/2403.15734)
- Lattice: conventional-cell parameters constrained by crystal system. Origin: "it is often possible to place the origin of the unit cell at the inversion center". The **Euclidean normalizer is mentioned only as future work**. "Asymmetric unit" and "fundamental domain" are not used — [CrystalFormer HTML](https://arxiv.org/html/2403.15734)

**SymmCD** (Levy, Panigrahi, Kaba, Zhu, Lee, Galkin, Miret, Ravanbakhsh; arXiv:2502.03638; peer-reviewed venue not shown on page)
- The crystal is (G, k, X′, S, A′): one representative per crystallographic orbit (the asymmetric unit, which "will only contain a single atom from each orbit"), a site-symmetry binary matrix (15×13, oriented site-symmetry symbols), and atom types. It works in the conventional cell — [SymmCD HTML](https://arxiv.org/html/2502.03638)
- Lattice: log(L) = Σ kᵢBᵢ with k ∈ ℝ⁶ and a symmetric-matrix basis (DiffCSP++'s parameterisation, Jiao et al. 2024), with constraints masked per crystal system. Orthogonal factors are dropped because "orthogonal transformations do not change the lattice" — [SymmCD HTML](https://arxiv.org/html/2502.03638)
- It does **not** discuss which orbit member is chosen, origin choice, the normalizer, or Wyckoff-letter ambiguity. Sampled coordinates are projected to "the nearest Wyckoff position with that site symmetry" — [SymmCD HTML](https://arxiv.org/html/2502.03638)

**Wyckoff Transformer / WyFormer** (Kazeev, Nong, Romanov, Zhu, Ustyuzhanin, Yamazaki, Hippalgaonkar; arXiv:2503.02407; ICML 2025, PMLR 267:29495–29526)
- This is the clearest ML treatment of normalizer redundancy. It defines the Euclidean normalizer as "the largest symmetry group preserving" the space group. It notes that Wyckoff positions can be equivalent under normalizer operations (e.g. 4a/4b in Fm-3m via a centring-type shift) and shows SrTiO3 with two equivalent Wyckoff representations — [WyFormer HTML](https://arxiv.org/html/2503.02407v4)
- Handling: site symmetry replaces Wyckoff letters, plus an "enumeration" index that isolates the arbitrary "choice of the coset representative" and "leaves the rest of the representation strictly invariant to that choice". The residual non-invariance is left "as a training goal by picking a randomly selected equivalent representation at every training epoch". In MP-20, 96% of structures have <10 equivalent representations — [WyFormer HTML](https://arxiv.org/html/2503.02407v4)
- Counting all normalizer choices gives 10,904 unique tokens in MP-20, against 1,047 template tokens (search snippet of [arXiv 2503.02407v2](https://arxiv.org/html/2503.02407v2)). WyFormer generates no coordinates or lattice; pyXtal does that — [WyFormer HTML](https://arxiv.org/html/2503.02407v4)

**Wyckoff augmentation for WyCryst/FTCP** (Yamazaki, Nong, Zhu, Novoselov, Ustyuzhanin, Hippalgaonkar; arXiv:2503.16784, "Multi-property directed generative design of inorganic materials through Wyckoff-augmented transfer learning")
- It uses "Euclidean normalizers to enumerate all unique symmetry-equivalent crystal representations". Wyckoff sets and normalizer coset representatives (3×4 matrices) come from the Bilbao Crystallographic Server. For example, space group 2 has 8 normalizer cosets (8× augmentation). The normalizer is also applied to fractional coordinates for FTCP. It cites Müller (2013), "Conjugate subgroups, normalizers and equivalent descriptions of crystal structures" — [arXiv 2503.16784 PDF](https://arxiv.org/pdf/2503.16784) (text extracted locally)

**WyckoffDiff** (Ekström Kelvinius, Andersson, Parackal, Qian, Armiento, Lindsten; arXiv:2502.06485 v4, Oct 2025)
- It generates discrete Wyckoff "protostructures" and evaluates them via AFLOW protostructure labels, which are "canonicalized based on alphabetical element order". No normalizer discussion was found in the extracted text — [WyckoffDiff PDF](https://arxiv.org/pdf/2502.06485) (text extracted locally; grep for normali*, fundamental domain, origin returned no normalizer passages)

**Space Group Equivariant Crystal Diffusion** (Chang, Pak, Guerra, Zhan, Richardson, Ertekin, Adams; arXiv:2505.10994 v3, Oct 2025)
- It represents the crystal on "a convex polytope Π∈ℝ³, the asymmetric unit (ASU)", which "maximally reduces infinite crystals by removing all redundancies induced by the space group". The ASUs are "exact", i.e. they tile ℝ³ without boundary overlaps, using face, edge and vertex conditions from Grosse-Kunstleve et al. This is the closest ML construction to a true fundamental domain, but it is the **space group's** ASU, not the normalizer's — [SGEquiDiff HTML](https://arxiv.org/html/2505.10994)
- Wyckoff positions are intersected with the exact ASU to form "Wyckoff shapes" (points, segments, facets, interior). Noise is projected onto their tangent spaces. Equivariance comes from symmetrising over G/T_L. The lattice is in the conventional basis (contrasted with Niggli primitive cells). The visible text does not discuss origin choice or the normalizer — [SGEquiDiff HTML](https://arxiv.org/html/2505.10994)
- Search snippets: the paper "drops frame averaging and gets SE(3) invariance by canonicalizing crystals with an asymmetric-unit representation" — [arXiv 2505.10994 PDF](https://arxiv.org/pdf/2505.10994)

**Other ASU-based diffusion models seen only in search snippets (not fetched):** Symmetry-Breaking De Novo Crystal Generation via Markovian Jump Diffusion ([arXiv 2608.13457](https://arxiv.org/html/2608.13457)) calls the ASU "the minimal non-redundant subset that generates the full crystal via symmetry operations". SPARC does RL fine-tuning on the ASU representation ([arXiv 2609.13468](https://arxiv.org/html/2609.13468v1)). Symmetry-Aware Bayesian Flow Networks ([arXiv 2502.03146](https://arxiv.org/html/2502.03146v2)). MaskGXT is reported in a search summary to use a "continuous" Euclidean-normalizer origin-shift augmentation with precomputed normalizer translation cosets for all 230 groups; its URL and authors were not confirmed (see Gaps).

**Crystal-GFN** (Mila; arXiv:2310.04925)
- It samples the space group, then composition, then lattice parameters sequentially, and places no atoms — [Crystal-GFN HTML](https://arxiv.org/html/2310.04925v2). SHAFT is a hierarchical GFlowNet extension that generates space group, lattice and coordinates — [Digital Discovery d4dd00392f](https://pubs.rsc.org/en/content/articlehtml/2025/dd/d4dd00392f). Neither was read for redundancy handling.

### Inferences
- Across inorganic full-cell models the pattern is: rotation is quotiented by representation (lengths/angles, Gram matrix, symmetric polar factor); origin/translation is handled by invariance on the torus (never by fixing an origin); lattice basis is fixed by canonicalisation (Niggli). None of them handles lattice-basis (GL(3,ℤ)) invariance architecturally.
- Wyckoff/ASU models remove space-group redundancy by representation, but inherit a finite residual redundancy from the normalizer coset (origin shift, axis permutation, enantiomorphic setting). Only WyFormer and the Yamazaki augmentation paper name it, and they handle it by augmentation and random representative selection. None quotients it.
- The Chang et al. "exact ASU" polytope is technically a fundamental domain of the space group acting on ℝ³. A normalizer analogue (the Cheshire-group ASU) would be a direct generalisation; no paper found does this.

### Gaps
- DiffCSP++ (Jiao et al., ICLR 2024, arXiv:2402.03992), CrystalFlow (Luo et al.), OMatG (Höllmer et al. 2025), Chemeleon, UniMat, CDVAE, WyCryst, Gruver et al. 2024 crystal-text LLMs, and "Equivariant Symmetry Breaking" were not fetched in this pass. Indirectly: SymmCD states that DiffCSP++'s lattice is log(L) = ΣkᵢBᵢ (symmetric basis, orthogonal factor dropped), and the CG-OMatG appendix (Q2) describes OMatG-lineage handling. Representation details of the others are from prior knowledge only and are not recorded here as findings.
- The MaskGXT normalizer-augmentation claim comes from a search-engine summary; the primary source was not located.
- The tail of each long HTML page (e.g. the last 40–60k characters of FlowMM, DiffCSP, CrystalFormer and SGEquiDiff) was not read. Appendices may contain further discussion of normalizers.

## Q2. How do molecular-crystal generative/sampling models (2023–2026) represent cell, position and orientation, and handle redundancy?

### Takeaway
The 2025–2026 wave of molecular-crystal generators (OXtal, MolCrystalFlow, PackFlow, OrgFlow, Clari/unit-cell flow matching, CG-OMatG) works almost entirely in **P1 full cells or lattice-free clusters**. These models use translation and permutation invariance or augmentation, rotation equivariance or augmentation, and at most Niggli canonicalisation of the lattice. None imposes a space group or reduces to an asymmetric unit, and none quotients by the normalizer. The exceptions are the MXtalTools line (Kilgour, Tuckerman, Rogal): MXtalGFlow parameterises a single fixed-space-group Z′=1 crystal by 12 asymmetric-unit-scaled parameters with a hemisphere-restricted rotation vector. It explicitly acknowledges that normalizer and molecular symmetries leave residual redundancy, and it enforces cell uniqueness by penalty.

### Cited Findings
**MXtalGFlow, "A Distributional Framework for Generative Modeling of Molecular Crystals"** (Kilgour, Dong, Tuckerman, Rogal; arXiv:2607.05266, 6 Jul 2026)
- Parameterisation: Z′=1 rigid molecule, one fixed space group, 12 DoF: (a,b,c,α,β,γ), centroid (u,v,w), and a rotation vector of the "canonicalized principal inertial vectors". These map to [−1,1]¹². Lengths are normalised by molecular size and rescaled from cell to asymmetric-unit length. Jacobians include V/Z and the SO(3) Haar factor sin²(r/2) sinθ — [EmergentMind summary](https://www.emergentmind.com/papers/2607.05266); [arXiv HTML](https://arxiv.org/html/2607.05266v1)
- Centroid domain: restricted to the **space group's** asymmetric unit ("asymmetric unit is the only free object, assuming a fixed space group"; for P-1, half the cell along a), **not** the normalizer's. The paper acknowledges that normalizer action can produce "duplicate physical lattices", and that molecular plus normalizer symmetries give "identical physical structures" at different latent points, visible in the centroid marginals. It argues the normalizer's action on general positions is "generically uniform", so multiplicities should not bias sampled populations in expectation — [arXiv HTML](https://arxiv.org/html/2607.05266v1)
- Orientation: the rotation vector is restricted to the upper hemisphere (z ≥ 0) for "representational uniqueness". Chirality is a free parameter: in non-Sohncke groups such as P2₁/c, models would be trained for both chiralities and weighted by learned partition functions. It is not clear that this was done in the reported runs — [arXiv HTML](https://arxiv.org/html/2607.05266v1)
- Cell: the parameterisation is "not unique in the unit cell box degrees of freedom" (e.g. monoclinic shear). A "standardized" Niggli-type subspace (Křivý–Gruber, Spglib) is enforced by an energy penalty: "uniqueness is enforced by penalty rather than by construction". The section is titled "Canonical Map". "Fundamental domain", "Cheshire" and "origin choice" do not appear in the fetched text — [arXiv HTML](https://arxiv.org/html/2607.05266v1)
- Model: an energy-based GFlowNet / diffusion sampler (neural SDE) with reward exp(−E/k_BT). Energies are softened LJ or UMA esen-s; losses are MLE plus trajectory balance. The prior molecular-crystal generators it cites (OXtal, PackFlow, MolCrystalFlow, OrgFlow, unit-cell flow matching) are said not to target a thermodynamic distribution — [arXiv HTML](https://arxiv.org/html/2607.05266v1)

**MXtalTools** (Kilgour, Tuckerman, Rogal; arXiv:2511.20327, Nov 2025; JCIM 2026 per docs)
- A toolkit with fast differentiable supercell construction "from asymmetric unit parameters and space group symmetry", plus MLIP-driven search. It is not itself a generative model — [arXiv 2511.20327](https://arxiv.org/abs/2511.20327)

**MolCrystalFlow** (Zeng, Sullivan, Egg, Martirossyan, Höllmer, Jin, Roitberg, Hennig, Martiniani, Tadmor, Liu; arXiv:2602.16020 v3, Mar 2026)
- Representation: lattice L (3×3, det>0), centroids F on the 3-torus, orientations R ∈ SO(3) from PCA frames, flowed as axis-angle via Rodrigues. The lattice is normalised to lower-triangular form with a ≤ b ≤ c (QR). The network sees the Gram matrix. No Niggli reduction — [MolCrystalFlow HTML](https://arxiv.org/html/2602.16020v3)
- Symmetries are stated as invariances: periodic translation p(L, τ(F+T·1ᵀ), R) = p(L,F,R), and global rotation p(LQᵀ, F, QR) = p(L,F,R). Permutation is handled by the GNN and by OT matching within axis-flip groups χ. An axis-flip state χ resolves PCA sign ambiguity; symmetric molecules get a "symmetric-rotation prior" (details in SI) — [MolCrystalFlow HTML](https://arxiv.org/html/2602.16020v3)
- No space-group constraints ("do not impose explicit space-group constraints"). Sampling uses Z ∈ {2,4} in P1. "Reduced asymmetric unit representations" appear as future work. "Fundamental domain", "normalizer" and "canonical" are absent — [MolCrystalFlow HTML](https://arxiv.org/html/2602.16020v3)

**CG-OMatG, "Riemannian Flow Models with Reinforcement Learning for Molecular Crystal Structure Prediction"** (Egg, Sullivan, Martirossyan, Höllmer, Zeng, Roitberg, Liu, Hennig, Sarupria, Tadmor, Martiniani; arXiv:2609.39773; NeurIPS 2026 listing; header date May 2026 vs arXiv stamp Sep 2026)
- Manifold: ℳ = SO(3) × Sym₃⁺ × (𝕋³ × SO(3))^M, with the lattice by polar decomposition L′ = UP — [CG-OMatG HTML](https://arxiv.org/html/2609.39773)
- Appendix G gives the most complete symmetry inventory in this literature:
  - Lattice translations: a quotient ℝ³/L, with the representative chosen by fractional wrapping.
  - Global translation: invariant.
  - Rotation: equivariant.
  - Molecular point groups: a proof that a single-valued equivariant orientation map is impossible for nontrivial stabilisers, so a set-valued orbit map is used.
  - PCA sign flips: augmentation over 4 proper sign combinations.
  - GL(3,ℤ)/supercells: "noted but neglected"; training uses the dataset's cell; Niggli is mentioned only as an alternative preprocessing step.
  - Permutations: invariant.
  - Space groups: not enforced; P1, with 𝒢 ≤ 𝒢_CG.
  
  The paper writes "the quotient ℳ/G identifies configurations that lie in the same orbit" but chooses "an equivariant modeling approach" over quotient-space modelling. "Fundamental domain" and "normalizer" are absent — [CG-OMatG HTML](https://arxiv.org/html/2609.39773) (Appendix G read at offset 100000)

**Clari, "Fast Organic Crystal Structure Prediction with Unit Cell Flow Matching"** (Lo, Mucko, Cheng, Cai, Price, Matusik, Aspuru-Guzik; arXiv:2606.03199 v2, Nov 2026)
- An all-atom Cartesian model. The lattice vectors are "three additional virtual points". There are no rigid bodies and no space-group input. Z is an input (sweep 4,2,1) — [Clari HTML](https://arxiv.org/html/2606.03199)
- The stated invariances are signed permutations of lattice rows, graph-automorphism atom permutations, rotations, and "independent periodic translations of any connected molecular subgraph". They are handled by an invariant prior, augmentation, and OT alignment (brute-force signed-permutation lattice alignment, Hungarian matching, weighted Kabsch). There is no Niggli reduction, origin fixing, normalizer or fundamental domain. The ASU appears only in data parsing — [Clari HTML](https://arxiv.org/html/2606.03199)

**OXtal** (Jin et al.; arXiv:2512.06987, Dec 2025)
- Lattice-free: it discards "symmetry representations of lattice vectors" in favour of Cartesian coordinates with SE(3) augmentation (non-equivariant Transformer). Training uses S⁴ shell crops around a central ASU molecule. CSD cells are first converted to "unique Niggli-reduced forms". Equivalences (translation, rotation, permutation, unit-cell change) are named as a quotient "𝓜(g) = 𝓧(g)/∼". "Fundamental domain" is absent — [OXtal HTML](https://arxiv.org/html/2512.06987v1)

**OrgFlow** (Vahediahmar, McDonald, Liu; arXiv:2602.20195, Feb 2026)
- Atom-level, with fractional coordinates and lattice, conditioned on the ASU molecular graph, Z′=1. It applies "standard reductions (e.g., Niggli) and space-group refinement to obtain a canonical, symmetry-consistent cell". An EGNN gives "translation invariance and rotation equivariance" — [OrgFlow HTML](https://arxiv.org/html/2602.20195)

**PackFlow** (Subramanian et al., Gómez-Bombarelli group; arXiv:2602.20140, Feb 2026)
- Flow matching with RL "physics alignment" using MLIP energies and forces — [arXiv 2602.20140](https://arxiv.org/abs/2602.20140). Per CG-OMatG it "generates per-atom Cartesian coordinates for all heavy atoms" without rigid-body structure — [CG-OMatG HTML](https://arxiv.org/html/2609.39773). Its own symmetry handling was not read.

**Rigid Body Flows** (Köhler, Invernizzi, de Haan, Noé; ICML 2023, PMLR 202:17301–17326; arXiv:2301.11355)
- Normalizing flows on unit quaternions using the double-cover property, for Boltzmann generators of rigid-molecule systems (ice XI, TIP4P). Fixed-cell sampling, not de novo structure generation — [PMLR](https://proceedings.mlr.press/v202/kohler23a.html)

**Conditional Boltzmann generators** (Schebek, Invernizzi, Noé, Rogal; Mach. Learn.: Sci. Technol. 2024, doi 10.1088/2632-2153/ad849d; arXiv:2406.12378)
- A (T,P)-conditioned flow with a permutation-equivariant architecture. The test system appears to be atomic (LJ), not molecular — [arXiv 2406.12378](https://arxiv.org/abs/2406.12378)

**AssembleFlow** (Guo et al., ICLR 2025)
- Rigid-body flow matching with inertial frames for molecular cluster packing (non-periodic). Seen via search summaries only; a follow-up proposes a permutation-invariant geometric loss for the molecular assembly task ([arXiv 2509.00832](https://arxiv.org/html/2509.00832)).

### Inferences
- The molecular-CSP ML literature names the redundancies (translation, rotation, permutation, cell change, PCA sign, molecular point group) and sometimes uses quotient language (DiffCSP, OXtal, CG-OMatG). Each one handles the redundancies by invariance, augmentation or canonicalisation, never by sampling on a constructed quotient domain.
- MXtalGFlow is the only work found that (i) fixes a space group, (ii) restricts the centroid to an asymmetric unit, (iii) restricts orientation to a half-ball, and (iv) explicitly names the Euclidean/affine normalizer as a residual source of duplicates. It leaves (iv) unquotiented. The requester's construction (reduced cell × ASU of the Euclidean normalizer × orientation fundamental zone) is therefore a direct tightening of that framework, and appears to be without precedent in the published ML literature found here.
- A hemisphere-restricted rotation vector is a fundamental domain for SO(3) modulo nothing. It only removes the r ↔ 2π − r / axis-sign double description of the rotation-vector chart. A true orientation fundamental zone would quotient the molecular point group and any normalizer point operations. CG-OMatG's set-valued orbit map is the only formal ML treatment of molecular point groups found.

### Gaps
- Day-group ML samplers, Genarris-ML, CSPML, "OpenMolCSP"/"MCSP": no primary sources were located in this pass.
- The MolCrystalFlow "symmetric-rotation prior" details are in the SI and were not read.
- The final ~16k characters of MXtalGFlow and the SI (latent bounds for centroids) were not read.
- PackFlow's own text and the AssembleFlow paper were not fetched.

## Q3. Does any ML model quotient by the Euclidean normalizer or define a fundamental domain, and how is "fundamental domain" used?

### Takeaway
No crystal generative model found in this pass constructs a fundamental domain of the normalizer (a Cheshire-group ASU), or uses the phrase "fundamental domain" for its sampling domain. The field says "asymmetric unit" (SymmCD, SGEquiDiff, OXtal), "quotient space" (DiffCSP, OXtal, CG-OMatG) or "canonical" (Niggli; PCA frames). The normalizer appears in ML only as a source of equivalent Wyckoff descriptions, handled by enumeration or augmentation (WyFormer, Yamazaki et al.), or as future work (CrystalFormer). "Fundamental domain" as an ML term comes from general invariant-learning theory (Aslan, Platt & Sheard).

### Cited Findings
- Targeted searches for "fundamental domain" with crystal generative/diffusion models returned no paper using the term. Results consistently used "asymmetric unit" — [search results incl. SymmCD](https://arxiv.org/html/2502.03638v3), [SGEquiDiff](https://arxiv.org/pdf/2505.10994), [review 2509.02723](https://arxiv.org/html/2509.02723v1)
- Term checks on fetched pages:
  - "Fundamental domain" was explicitly absent from DiffCSP, FlowMM ("repeating fundamental volume element" only), CrystalFormer, SymmCD, WyFormer, SGEquiDiff (visible text), ADiT, MolCrystalFlow, CG-OMatG, Clari, OXtal, OrgFlow and MXtalGFlow (fetched portion). Sources are the respective links in Q1/Q2.
  - "Normalizer" was present in WyFormer, Yamazaki et al., MXtalGFlow, and CrystalFormer (future work only); it was absent from the others checked.
- Aslan, Platt & Sheard, "Group invariant machine learning by fundamental domain projections" (NeurIPS 2022 workshop on Symmetry and Geometry in Neural Representations; PMLR 197:181–218, 2023; arXiv:2202.02164). It is a preprocessing map from inputs into a space parametrising group orbits, after which any model can be applied; the authors present it as a unifying view of intrinsic invariant approaches. Applications are CICY Hodge numbers and Kreuzer–Skarke data, not crystals. Platt's talk says the approach is useful when the group is big and "useless if the group is small" — [PMLR](https://proceedings.mlr.press/v197/aslan23a.html); [seminar abstract](https://researchseminars.org/talk/CompAlg/16/)
- Canonicalisation and frame averaging for crystals:
  - A crystal-tensor property paper canonicalises by polar decomposition ([arXiv 2410.02372](https://arxiv.org/pdf/2410.02372v4)).
  - FAENet redefines frames so that frame averaging stays E(3)-equivariant for periodic cells ([arXiv 2305.05577](https://arxiv.org/pdf/2305.05577)).
  - An ICLR 2025 paper argues static frames are ill-defined for crystals and proposes dynamic local frames ([ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/187d94b3c93343f0e925b5cf729eadd5-Paper-Conference.pdf)).
  - A late-2025 paper proposes one architecture invariant under any space group via a symmetry-adapted Fourier basis ([alphaxiv 2512.13989](https://www.alphaxiv.org/abs/2512.13989)).
  
  All of these are from search snippets and were not read in full. None was found to target the normalizer.

### Inferences
- In crystallography, "fundamental domain" is standard for a space group's ASU. Hirshfeld's Cheshire groups are exactly the Euclidean normalizers, whose ASUs are the "Cheshire cells". Neither phrase appears in the ML generative literature found. A paper using "fundamental domain" for reduced cell × Cheshire ASU × orientation zone would be introducing the term to this subfield. It should cite Aslan et al. for the ML usage and Chang et al. (exact ASU polytopes) as the nearest ML construction.
- Aslan et al.'s observation that fundamental-domain projection pays off for large groups is relevant: the normalizer adds a factor of order 2–8+ for low-symmetry groups (e.g. 8 cosets for P-1 per Yamazaki et al.). That is modest compared with GL(3,ℤ) or S_N, but it is exact and finite.

### Gaps
- Kaba et al. 2023 (learned canonicalisation) and Puny et al. 2022 (frame averaging) were not fetched, and no application of either to space-group normalizers was found.
- No OpenReview discussion of normalizer redundancy was located.

## Q4. How do models parameterise molecular orientation and quotient point symmetry?

### Takeaway
The parameterisations in use are unit quaternions (Köhler et al. 2023, using the double cover), axis-angle/rotation vectors from PCA or inertial frames (MolCrystalFlow, MXtalGFlow), SO(3)-valued rigid-body frames in Riemannian flow matching (CG-OMatG, AssembleFlow, MOFFlow), or no orientation at all (all-atom models: OXtal, Clari, OrgFlow, PackFlow). Molecular point symmetry is handled by sign-flip states and symmetric priors (MolCrystalFlow), set-valued orbit maps plus augmentation (CG-OMatG), or a hemisphere restriction (MXtalGFlow). No model was found that quotients normalizer point operations from orientation.

### Cited Findings
- Köhler et al.: flows on unit quaternions with the double-cover property — [PMLR 202](https://proceedings.mlr.press/v202/kohler23a.html)
- MolCrystalFlow: PCA frame to axis-angle (Rodrigues), an axis-flip state χ for sign ambiguity, and a "symmetric-rotation prior" for symmetric molecules — [MolCrystalFlow HTML](https://arxiv.org/html/2602.16020v3)
- CG-OMatG: proves that no single-valued equivariant orientation map exists when the molecule's rotational stabiliser is nontrivial, and so uses a set-valued orbit map. PCA ordering is fixed by sorted eigenvalues (removing 6 permutations). The 4 proper sign combinations are augmented, which removes "the need for explicit molecular orientation canonicalization" — [CG-OMatG HTML](https://arxiv.org/html/2609.39773)
- MXtalGFlow: a rotation vector in spherical form (θ,φ,r) with θ ∈ [0,π/2] for "representational uniqueness". Chirality is handled per space-group type (Sohncke vs non-Sohncke) — [arXiv HTML](https://arxiv.org/html/2607.05266v1)
- MOFFlow (ICLR 2025): rigid-body SE(3) flow matching over building blocks — [OpenReview](https://openreview.net/forum?id=dNT3abOsLo)

### Inferences
- Combining an orientation fundamental zone with the normalizer's point part (which can map one molecular orientation to another while also moving the centroid) has not been done in the ML literature found. The CG-OMatG impossibility result is a useful citation for why a single-valued chart must either cut SO(3) at seams or accept a set-valued map.

### Gaps
- IGSO(3)-diffusion usage in molecular-crystal generators was not specifically confirmed in any fetched paper.
- Whether any model quotients the inversion/mirror part of the normalizer jointly with molecular handedness was not found and is presumed absent.
