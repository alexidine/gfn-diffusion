# Gap closure: IUCrJ 2026 "gq5020" and Hirshfeld (1968) Cheshire groups

Evidence tags: [V] read in the primary source by me this session; [S] secondary source or search snippet; [K] background knowledge, not verified this session.

## Q1. What is IUCrJ 2026 paper gq5020, and does it propose an asymmetric unit of the Euclidean normaliser / a parameter-space domain?

### Takeaway
gq5020 is Majumder, Widdowson, Elkin, Anosova, Cooper, Day & Kurlin (2026), "Continuous invariant-based asymmetries of periodic crystals quantify deviations from higher symmetry", IUCrJ 13(4), 456-470, doi:10.1107/S2052252526002988. It is a structure-analysis/comparison paper (a continuous asymmetry measure built on isometry invariants). It does not mention Euclidean normalisers, Cheshire groups or Hirshfeld, and proposes no search or parameter-space domain. The earlier search hit for "asymmetric unit" + "Euclidean normalizer" was a false positive.

### Cited Findings
- Identification [V]: Crossref record for doi:10.1107/S2052252526002988 lists the full-text link `https://journals.iucr.org/m/issues/2026/04/00/gq5020/gq5020.pdf`, reference keys `gq5020_bbNN`, volume 13, issue 4, pages 456-470, published 2026-06-01, licence CC BY 4.0 — [Crossref API](https://api.crossref.org/works/10.1107/s2052252526002988)
- Authors and affiliations [V]: Surya Majumder, Daniel Widdowson, Yury Elkin, Olga Anosova, Andrew I. Cooper, Graeme M. Day, Vitaliy A. Kurlin (University of Liverpool Computer Science / Materials Innovation Factory; Day at Southampton). PMID 42220268, PMCID PMC13324622; Europe PMC dates the issue "2026 Jul" — [Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1107/S2052252526002988&format=json&resultType=core)
- Abstract (summarised, [V]): ideal symmetry breaks under noise; Z' changes discontinuously under almost any displacement; the paper introduces a Continuous Invariant-based Asymmetry (CIA), in ångströms, measuring how far a periodic crystal deviates from a higher-symmetry form. On several CSP datasets about half of simulated crystals have high CIA while all experimental structures have CIA = 0; many high-Z' CSD crystals are close to more symmetric forms with Z' ≤ 1 — [Crossref abstract](https://api.crossref.org/works/10.1107/s2052252526002988)
- Full text read via Europe PMC (PMC13324622) [V]. Term search over the full text: "normaliz/normalis" appears once, only as "interatomic distances (usually normalized by frequencies...)"; "Cheshire" 0 hits; "Hirshfeld" 0 hits; "Euclidean" twice, both meaning ordinary Euclidean space/distance — [Europe PMC full text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13324622/fullTextXML)
- Its definition of asymmetric unit [V], close paraphrase: an asymmetric unit of S is a minimal, simply connected subset whose images under all symmetry operations of S tile the space. This is the ordinary space-group asymmetric unit, not a normaliser one — [Europe PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13324622/fullTextXML)
- Formal content [V]:
  - Definition 1 defines relative multiplicity Z' for a periodic point set whose motif is split into "geometric blocks" (molecules, atoms or ions).
  - Definitions 2-3 build on the Pointwise Distance Distribution PDD(S;k) and an adjusted version PDA(S;k).
  - Definition 4 is the Earth Mover's Distance between blocks.
  - Definition 5 defines CIA(S) as a min-max over EMDs between blocks in an asymmetric unit, with an average variant.
  - Lemma 7: CIA is isometry-invariant and independent of the choice of asymmetric unit.
  - Theorem 9: continuity, |CIA(S) − CIA(Q)| ≤ 4ε under ε-perturbation.
  - Theorem 10: computable in polynomial time, O(m³) for m single-point blocks.
  — [Europe PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13324622/fullTextXML)
- Application [V]: four CSP datasets (Pulido et al. 2017 nanoporous-cage molecules; over 50,000 simulated crystals in total) and a CSD snapshot of 12 Nov 2025 (1,394,755 entries; 907,223 without disorder; 69,196 with G ≥ 2 molecules of identical composition in the asymmetric unit). It concludes there is no need to search for a higher-symmetry group, because the CIAs are continuous — [Europe PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13324622/fullTextXML)

### Inferences
- gq5020 is not prior art for a symmetry-reduced search domain. Its relevance is indirect: it comes from the Liverpool isometry-invariant programme (Widdowson & Kurlin), which handles equivalence through complete invariants (PDD and similar) rather than a fundamental domain of a parameter space [K for the programme framing].
- It was probably matched to "Euclidean normalizer" by search-engine fuzziness (the words "asymmetric unit" and "Euclidean" both occur).

### Gaps
- None for identification. The journals.iucr.org HTML and PDF returned 403 in earlier attempts, but the PMC mirror made the full text available.

## Q2. Hirshfeld (1968): exact abstract, items (a)-(d), and post-1990 CSP-relevant citing papers

### Takeaway
I recovered the verbatim abstract from an archived IUCr abstract page. It confirms (a) a definition of the derivative symmetry, (b) "thirty distinct 'Cheshire' groups", and (c) that the direct product of the Cheshire group with the molecular point group is the symmetry of the 6-D trial-structure parameter space, whose asymmetric unit "is the region to be scanned". I could not verify (d) "Cheshire cell", because the full text was blocked. Most later citers are molecular-replacement, powder-solution and normaliser-theory papers. The CSP-relevant citers cluster in 1980-2005, chiefly Dzyabchenko and van Eijck & Kroon. Semantic Scholar shows no post-2005 CSP or ML-generation citer.

### Cited Findings
- Citation [V]: F. L. Hirshfeld, "Symmetry in the generation of trial structures", Acta Cryst. (1968) A24, 301-311, doi:10.1107/S0567739468000501 — [Wayback snapshot of IUCr abstract page, 2025-04-23](http://web.archive.org/web/20250423082147/https://journals.iucr.org/paper?S0567739468000501); identical text in [Wayback snapshot of scripts.iucr.org, 2024-04-19](http://web.archive.org/web/20240419170339/https://scripts.iucr.org/cgi-bin/paper?S0567739468000501)
- Verbatim abstract [V] (quoted in full here for the record; the downstream report should quote under 15 words):
  > "In the generation of trial structures by the systematic variation of the position and orientation of a molecule of known dimensions, the ranges to be scanned by the positional and orientational parameters depend on the symmetry of the molecule and on the space group. The set of transformations of the crystal axes that leave invariant the coordinates of equivalent positions for a given space group defines a corresponding derivative symmetry, conforming to one of thirty distinct `Cheshire' groups. The direct product of this group with the molecular point group specifies the symmetry of the six-dimensional space of the trial-structure parameters. The asymmetric unit in this space is the region to be scanned by the several parameters."
  — [archived IUCr abstract](http://web.archive.org/web/20250423082147/https://journals.iucr.org/paper?S0567739468000501)
- (a) Definition [V, abstract level]: the Cheshire group is the "derivative symmetry" defined by the set of transformations of the crystal axes that leave the coordinates of the space group's equivalent positions invariant. The modern name for this is the Euclidean normaliser [K: Koch & Fischer, International Tables A ch. 15/3.5; Koch & Fischer 2006, doi:10.1524/zkri.2006.221.1.1, which cites Hirshfeld 1968 per Semantic Scholar]. The abstract does not say whether improper (handedness-reversing) transformations are included. Whether his groups are the full Euclidean normaliser or the chirality-preserving one is therefore not settled by what I could read.
- (b) Count [V]: "one of thirty distinct 'Cheshire' groups" — [archived abstract](http://web.archive.org/web/20250423082147/https://journals.iucr.org/paper?S0567739468000501). The basis for counting 30 (for example, normalisers of general-metric space groups up to type) is not stated in the abstract.
- (c) 6-D space [V]: yes, nearly verbatim. The direct product of the Cheshire group with the molecular point group "specifies the symmetry of the six-dimensional space of the trial-structure parameters"; "the asymmetric unit in this space is the region to be scanned" — [archived abstract](http://web.archive.org/web/20250423082147/https://journals.iucr.org/paper?S0567739468000501). The cell is treated as given (a "molecule of known dimensions" in a fixed space group). Cell reduction is not part of the claim.
- (d) "Cheshire cell" [not verified]: the phrase is absent from the abstract. The full text (journals.iucr.org, the a05891.pdf link and Wiley) returned 403, and the Wayback Machine has no snapshot of the PDF. Later molecular-replacement work uses "Cheshire cell" and "anti-Cheshire cell", with the asymmetric unit of the Cheshire cell as the region scanned by rotation and translation parameters [S, search snippet]; one figure is captioned "Symmetry diagram of space group P3. The Cheshire cell and symmetry indicated in red" [S] — [ResearchGate figure](https://www.researchgate.net/figure/a-Symmetry-diagram-of-space-group-P3-The-Cheshire-cell-and-symmetry-indicated-in-red_fig3_236600222); see also Dauter (2013) "On optimal placement of molecules in the unit cell", Acta Cryst. D69, doi:10.1107/S0907444913002722 [S] — [PMC4005360](https://pmc.ncbi.nlm.nih.gov/articles/PMC4005360/). I could not confirm whether Hirshfeld coined "Cheshire cell".
- Citation counts [V]: Semantic Scholar lists 83 citing records; Crossref is-referenced-by-count = 58 — [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1107/S0567739468000501/citations?fields=title,year,authors,venue,externalIds&limit=200); [Crossref](https://api.crossref.org/works/10.1107/S0567739468000501)
- CSP and trial-structure-generation citers for molecular crystals, all [V] as listed by Semantic Scholar (I verified the citation link, not the content of these papers):
  - Post-1990:
    - Scaringe, R. (1991) "A Theoretical Technique for Layer Structure Prediction", doi:10.1007/978-94-011-3278-7_8
    - Dzyabchenko, A. V., D'Yachkov, P., Agafonov, V. (1995) "Potential energy surface and orientational disorder in solid fullerene C60", doi:10.1007/BF00714418
    - Dzyabchenko, A., Pivina, T., Arnautova, E. A. (1996) "Prediction of structure and density for organic nitramines", J. Mol. Struct. THEOCHEM, doi:10.1016/S0166-1280(96)91002-7
    - van Eijck, B. P. & Kroon, J. (1997) "Fast clustering of equivalent structures in crystal structure prediction", J. Comput. Chem. 18, 1036, doi:10.1002/(SICI)1096-987X(199706)18:8<1036::AID-JCC7>3.0.CO;2-U
    - Dzyabchenko, A. & Scheraga, H. (2004) "Model for the crystal packing and conformational changes of biphenyl in incommensurate phase transitions", Acta Cryst. B60, doi:10.1107/S010876810400312X
    - Papoular, R. J. et al. (2005) "X-ray powder diffraction structure determination of gamma-butyrolactone at 180 K: phase-problem solution from the lattice energy minimization with two independent molecules", Acta Cryst. B61, doi:10.1107/S0108768105005392
  - Pre-1990 CSP antecedents:
    - Hall & Wood (1980) "Prediction of the crystal structure of a cyclic tetrapeptide", J. Comput. Chem., doi:10.1002/jcc.540010408
    - Dzyabchenko (1984) "Theoretical structures of crystalline benzene: the search for a global minimum of the lattice energy in four space groups", doi:10.1007/BF00749334
    - Dzyabchenko & Bazilevskii (1985), doi:10.1007/BF00754129
    - Dzyabchenko (1988) "...VI. Global search in the bisystemic structural class", doi:10.1007/BF00745258
  — [Semantic Scholar citations](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1107/S0567739468000501/citations?fields=title,year,authors,venue,externalIds&limit=200)
- Adjacent non-CSP citers worth knowing [V, citation link only]:
  - Normaliser theory:
    - Gubler (1982) "Normalizer groups and automorphism groups of symmetry groups", doi:10.1524/zkri.1982.158.12.1
    - Kopský (1993) "Translation normalizers of Euclidean groups I/II", doi:10.1063/1.530173, doi:10.1063/1.530174
    - Koch & Fischer (2006) "Normalizers of space groups: a useful tool in crystal-structure description, comparison and determination", doi:10.1524/zkri.2006.221.1.1
    - Müller & de la Flor (2024) "Symmetry Relationships between Crystal Structures" (OUP), doi:10.1093/oso/9780192858320.001.0001
  - Molecular-replacement motion spaces, the closest formal analogue of a 6-D fundamental domain:
    - Chirikjian (2011) "Mathematical aspects of molecular replacement. I. Algebraic properties of motion spaces", Acta Cryst. A67, doi:10.1107/S0108767311021003
    - Shiffman, Lyu & Chirikjian (2020) "...V. Isolating feasible regions in motion spaces", Acta Cryst. A76, doi:10.1107/s2053273319014797
  - Standardised placement:
    - Dauter (2013), doi:10.1107/S0907444913002722
    - Kowiel, Jaskólski & Dauter (2014) "ACHESYM", doi:10.1107/S1399004714024572
  - Powder and direct-space solution: the EXPO, SIR and RAMM papers by Altomare, Burla, Giacovazzo et al., 2007-2018
  — [Semantic Scholar citations](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1107/S0567739468000501/citations?fields=title,year,authors,venue,externalIds&limit=200)

### Inferences
- For a fixed cell, the requester's "asymmetric unit of normaliser × orientation fundamental zone" is in substance Hirshfeld's 1968 prescription, scanning the asymmetric unit of (Cheshire group × molecular point group) acting on the 6-D position+orientation space. Hirshfeld's abstract does not cover:
  - the reduced-cell factor;
  - the coupling where the normaliser depends on metric specialisation of the cell;
  - the explicit chirality-preserving variant for chiral molecules;
  - use as a domain for a learned or generative sampler.
  Novelty, if any, lies in those additions and in their use, not in the position/orientation reduction itself.
- Caution, an inference not checked against the full text: the linear parts of the Cheshire group also rotate the molecular orientation. The 6-D action is therefore not two independent factor actions. Hirshfeld's "direct product" is a statement about the group, and the 6-D asymmetric unit is not, in general, simply (position ASU) × (orientation fundamental zone). If the requester's domain is a plain Cartesian product of the two, that difference should be checked against the full text.
- Semantic Scholar shows no CSP-method or ML-generation paper citing Hirshfeld after 2005, which suggests the 1968 idea has not been carried forward explicitly into modern CSP or generative work. Citation databases are incomplete, so this is weak evidence.

### Gaps
- Full text of Hirshfeld (1968) was blocked (journals.iucr.org 403, Wiley 403, no Wayback copy of the PDF). Still unverified: the phrase "Cheshire cell"; the exact basis for "thirty"; whether improper operations are included; whether the paper tabulates the asymmetric units.
- Not checked: Google Scholar "cited by" (likely more citers than Semantic Scholar's 83, and possibly modern CSP papers that cite without a Semantic Scholar link). Neither was the IUCr "cited by" list (blocked).
- The Online Dictionary of Crystallography "Cheshire group" entry returned 403 and has no Wayback snapshot.

## Q3. Any 2015-2026 paper that explicitly restricts molecular CSP or generative crystal search to the asymmetric unit of the Euclidean normaliser / Cheshire group?

### Takeaway
None found within a narrow search budget (three web searches plus targeted reads). The nearest modern uses of the Euclidean normaliser in crystal generation are canonicalisation and augmentation over origin choice, in inorganic or Wyckoff-token models. They do not restrict a molecular-crystal search domain.

### Cited Findings
- MaskGXT, from "Discovering Crystal Structure Prediction Algorithms with an AI Co-Scientist" (arXiv 2606.22866, 2026), uses a "Euclidean-normalizer origin shift" as a training augmentation [V]: a continuous origin shift r ~ U[0,1)³ with Wyckoff letters relabelled by the nearest normaliser coset. It precomputes normaliser translation cosets for all 230 space groups. It contrasts this with Kazeev et al. (2025), "who enumerate discrete coset representatives of the space group's Euclidean normalizer" — [arXiv 2606.22866 HTML](https://arxiv.org/html/2606.22866v1)
- Kazeev et al. (2025), the Wyckoff-based generator cited above, used normaliser coset representatives according to the MaskGXT paper [S, secondhand via MaskGXT; not read].
- Zhu & Weng, "Molecular Crystal Structure Prediction from Conditional Flow on the Unit Cells" (arXiv 2610.04193) [V, abstract only]:
  - learns invariant lattice descriptors conditioned on molecular graph, Hall setting and Z';
  - then runs quasi-random molecular packing search within the given symmetry and cell;
  - reproduces 83 of 84 systems with Z' ≤ 1.
  The abstract does not mention a normaliser or Cheshire-group reduction of the packing search. I did not read the full text — [arXiv abs](https://arxiv.org/abs/2610.04193)
- A web search for "Euclidean normalizer" + CSP search space returned no source that uses the normaliser to prune or parameterise the molecular CSP search space [S] — search results incl. [MolCrystalFlow arXiv 2602.16020](https://arxiv.org/html/2602.16020v3), [Beran-type CSP review PMC10685338](https://pmc.ncbi.nlm.nih.gov/articles/PMC10685338/)
- A web search for "Cheshire group" + CSP/generative model found Cheshire usage only in the molecular-replacement context (Cheshire cell, anti-Cheshire cell), and asymmetric-unit-based generators (SymmCD, CGWGAN) that use the space-group asymmetric unit, not the normaliser one [S] — [arXiv 2511.22652 review](https://arxiv.org/html/2511.22652); [ResearchGate figure](https://www.researchgate.net/figure/a-Symmetry-diagram-of-space-group-P3-The-Cheshire-cell-and-symmetry-indicated-in-red_fig3_236600222)

### Inferences
- Within this search, the explicit pairing "Euclidean-normaliser asymmetric unit × orientation fundamental zone × reduced cell" as a molecular-CSP or generative domain appears unpublished 2015-2026. In ML crystal generation, the normaliser shows up as an origin-equivalence augmentation or canonicalisation (MaskGXT, Kazeev et al.), not as a restricted sampling domain.
- The direct antecedent remains Hirshfeld (1968) for position+orientation at fixed cell. Molecular replacement (Cheshire cell; Chirikjian's motion-space series) is the closest continuing literature.

### Gaps
- The search was not exhaustive. Not run: Google Scholar; the exact phrase "permissible origins" + "crystal structure prediction"; full-text searches of GRACE, CrystalPredictor, UPACK, MOLPAK, PyXtal or Genarris documentation. Older CSP codes (for example MOLPAK, or Dzyabchenko's PMC program) may use normaliser-based reductions without the modern term [K, unverified].
- Kazeev et al. (2025) and the full text of Zhu & Weng (2026) were not read.
