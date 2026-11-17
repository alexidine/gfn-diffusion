# Molecular replacement and powder direct-space precedent for symmetry-reduced rigid-body search

Evidence labels used throughout:
- **[V]** verified this session from the primary source text, or from a documentation page fetched this session.
- **[S]** secondary: the claim comes from a citing source, a search snippet or a reference list, not from the original paper.
- **[K]** from prior knowledge and not re-verified this session. The writer should treat these as leads, not citations.

## Q1. Rotation function: how the asymmetric unit of orientation space is derived, and what it is called

### Takeaway
MR calls the symmetry of orientation space the **"rotation-function space group"** (Rao, Jih & Hartsuck 1980) or the **"symmetry of the rotation function"** (Tollin, Main & Rossmann 1966; Moss 1985). Its unique region is the **"asymmetric unit of the rotation function"** or the "unique region". Two things generate it:
- the rotational symmetry of the crystal Patterson. This is the Laue group with inversion set aside, so in practice the crystal point group's proper rotations.
- the rotational self-symmetry of the search-model Patterson.

With orders n and m, the redundancy is nm. Euler-angle box limits work for cyclic and dihedral groups. For cubic and icosahedral symmetry they fail, and Yeates (1993) solved the general case with Dirichlet (Voronoi) domains in rotation-matrix space. No source found uses "Cheshire group" for the rotation function; that term belongs to the translation function.

### Cited findings
- **[V]** Rao, S. N., Jih, J.-H. & Hartsuck, J. A. (1980). Rotation-function space groups. *Acta Cryst.* A36, 878–884. doi:10.1107/S0567739480001854. The paper tabulates the possible rotation-function space groups for all non-cubic space groups and gives their asymmetric units in Lattman's (θ+, θ−) coordinates (bibliographic data and abstract-level content via search). — [typeset.io PDF](https://typeset.io/pdf/rotation-function-space-groups-1ne3a7h92l.pdf); [IUCr ITA ref list](https://it.iucr.org/Ab/ch13o1v0001/sec13o1o2o1/)
- **[V]** The ROTFUN program manual (X.-J. Zhang, 1995) tells users to determine "the unique regions of a rotation function" from Rao et al. (1980) or with the program EDPDB. It does not mention Cheshire groups. — [ROTFUN manual](https://www.uoxray.uoregon.edu/local/manuals/mrchk/rotfun.html)
- **[V]** The Yeates lab page on the rotation-function asymmetric-unit problem makes these points:
  - With Patterson rotational symmetry of orders n and m, "only a fraction of the rotation function space (1/nm) is unique".
  - Patterson symmetry comes from the crystal's rotational symmetry with inversion set aside; for example, P2₁2₁2₁ gives 222.
  - Tollin, Main & Rossmann (1966) gave Euler-angle limits for the cyclic and dihedral cases, for example α from 0 to 60° for a 6-fold.
  - In cubic cases no Euler-angle limits work. Using the 422 subgroup leaves a 3-fold redundancy, and Burdina's method cut it to about 1.6 but is incomplete.
  - Yeates (1993) builds the unique regions as Dirichlet domains, also called Voronoi polyhedra, whose boundaries are linear in "the 9 elements of the 3 by 3 rotation matrix". The method covers arbitrarily high symmetry, including icosahedral.

  — [Yeates lab](https://yeateslab.mbi.ucla.edu/solution-of-the-rotation-function-asymmetric-unit-problem)
- **[V]** Yeates, T. O. (1993). The asymmetric regions of rotation functions between Patterson functions of arbitrarily high symmetry. *Acta Cryst.* A49, 138–141 (citation as given on the lab page). — [Yeates lab](https://yeateslab.mbi.ucla.edu/solution-of-the-rotation-function-asymmetric-unit-problem)
- **[S]** Moss, D. S. (1985). The symmetry of the rotation function. *Acta Cryst.* A41, 470–475. International Tables Vol. F ch. 13.2 cites it, together with Tollin et al. (1966), for how Patterson symmetries act on the rotation function. — [ITF 13.2.3.4](https://it.iucr.org/Fa/ch13o2v0001/sec13o2o3o4/)
- **[S]** Tollin, P., Main, P. & Rossmann, M. G. (1966). The symmetry of the rotation function. *Acta Cryst.* 20, 404–407. A search summary gives the pages as 404–417, while the Yeates page gives 404–407; the pages are unresolved. — [Yeates lab](https://yeateslab.mbi.ucla.edu/solution-of-the-rotation-function-asymmetric-unit-problem)
- **[V]** Phaser 2.1 keyword documentation for the brute rotation function: "Sample all unique angles (FULL) or restrict the search to the region of +/- RANGE degrees" (ROTAte FULL / AROUnd). — [Phaser 2.1 keywords](https://www-structmed.cimr.cam.ac.uk/phaser_obsolete/documentation/phaser-2.1_key.html)
- **[V]** EPMR does **not** use a rotation-function asymmetric unit. It uses quaternions so the initial population evenly samples the whole of rotation space, because uniform sampling in Euler angles is "highly biased". — Kissinger, Gehlhaar & Fogel (1999), *Acta Cryst.* D55, 484–491, doi:10.1107/S0907444998012517, full text read. [PDF](https://people.mbi.ucla.edu/sawaya/tutorials/Molrep/manuals/epmr.pdf)
- **[S]** Burdina, V. I. (1970/1971), working on the cubic-group rotation-function asymmetric unit, is mentioned without a full reference on the Yeates page; Crowther (1972) is also mentioned there. — [Yeates lab](https://yeateslab.mbi.ucla.edu/solution-of-the-rotation-function-asymmetric-unit-problem)

### Inferences
- The MR rotation-function asymmetric unit is the double coset G_Laue,proper \ SO(3) / G_model,self. It is the analogue of the requester's "orientation fundamental zone", with the molecule's point-group self-symmetry on the right and the crystal's proper point group on the left.
- MR uses the **Patterson (Laue) symmetry**, not the full Euclidean normaliser. The extra rotations a normaliser adds beyond the space group's point group are not symmetries of the observed Patterson. Examples are the perpendicular 2-folds of P4/mmm as the normaliser of P4, or axis permutations under metric specialisation. They show up instead as alternative indexing (see Q4, Dauter 2013, re-indexing h,k,−l for P4/P6).
  - In energy-only CSP there is no data to break this degeneracy. The full normaliser point part can therefore reduce orientation space, beyond anything MR does.
- Yeates's Dirichlet-domain construction in rotation-matrix coordinates is the closest MR precedent for a generic orientation fundamental zone. Euler-box limits are only valid for cyclic and dihedral symmetry.

### Gaps
- I could not access the full texts of Rao et al. (1980) and Moss (1985) for verbatim terminology, for example whether Moss defines a combined crystal-and-model group or whether either paper uses "normaliser".
- I found no source showing MR programs (AMoRe, MOLREP, Phaser) folding the **search model's** self-symmetry into the rotation search region automatically. Phaser's "unique angles" wording suggests some reduction, but its exact rule was not verified.

## Q2. Translation function and the Cheshire cell; which programs implement it

### Takeaway
The standard MR convention is to search the **first** molecule's translation over the **Cheshire cell**, Hirshfeld's (1968) term. The Cheshire cell is the unit cell of the Cheshire group, "the space group with its contents removed". Placing the first molecule fixes the origin, so later molecules are searched over the full cell, or the full primitive cell for centred lattices.
- In polar space groups the Cheshire cell has zero extent along the floating axis, and the first molecule's search is a plane.
- MR searches the Cheshire **cell**, the lattice of allowed origin shifts. It does not search the asymmetric unit of the Cheshire group, because the model's orientation and handedness are already fixed.
- AMoRe and Phaser are documented as doing this. MOLREP is not verified this session.

### Cited findings
- **[V]** CCP4 documentation page "Cheshire cells" (mirror):
  - It quotes the AMoRe documentation: "The Cheshire cell is the minimum volume which will allow a unique solution".
  - It describes the Cheshire group as the crystal's space group with its contents removed.
  - It states that only the first molecule's search is limited to the Cheshire cell, because placing that molecule fixes the origin. Later molecules are searched over the whole cell, or the primitive volume for centred cells.
  - It reproduces Hirshfeld's table for all 230 space groups, for example P2₁2₁2₁ → Cheshire group Pmmm, cell ½a × ½b × ½c, and P4₃2₁2 → P4₂22, cell ½(a−b) × ½(a+b) × ½c. P1's Cheshire cell is a single point.
  - It names AMoRe and Ian Tickle's TFFC as users.

  — [CCP4 Cheshire cells](https://staraniso.globalphasing.org/html/cheshirecell.html)
- **[S]** A CIMR (Cambridge) MR course page says the same: the Cheshire-cell limit applies to the first molecule only, and its placement fixes the origin. — [CIMR MolRep course](https://www-structmed.cimr.cam.ac.uk/Course/MolRep/molrep.html) (via search summary)
- **[V]** Phaser 2.1 keywords:
  - TRANslate FULL (default) is the "Search volume for brute force translation function. Cheshire cell or Primitive cell volume." LINE, REGIon and AROUnd are user-restricted alternatives.
  - Polar-group warning: "where the translation search covers only a plane, the TF Z-score for correct solutions can be low."

  — [Phaser 2.1 keywords](https://www-structmed.cimr.cam.ac.uk/phaser_obsolete/documentation/phaser-2.1_key.html)
- **[V]** Dauter, Z. (2013). On optimal placement of molecules in the unit cell. *Acta Cryst.* D69, 872–878. doi:10.1107/S0907444913002722.
  - The Cheshire cell's asymmetric unit is, quoting Hirshfeld, the region "to be scanned by the rotation and translation parameters".
  - In polar groups (classes 2, 3, 4, 6, m) and in P1 the origin floats, and the Cheshire cell is "infinitesimally small" along those axes. Table 2 marks these with an ε notation, and by convention the molecular centre is placed at 0 along them.
  - Most MR programs solve the rotation function first and then search translations over the Cheshire cell. According to the fetch summary, the fixed chirality means a restriction to a fraction of the cell arises only in I- or R-centred cases. This wording is paraphrased by the fetch tool and should be checked against the PDF.

  — [PMC4005360](https://pmc.ncbi.nlm.nih.gov/articles/PMC4005360/)
- **[S]** International Tables Vol. F ch. 13.3 (translation functions) defines Cheshire groups after Hirshfeld for all 230 space groups. The unit cell of the Cheshire group equals the asymmetric unit of the translation function, which reduces the grid that must be searched. — [ITF 13.3](https://it.iucr.org/Fa/ch13o3v0001) (403 on fetch; content via search snippet)
- **[V]** Hirshfeld, F. L. (1968). Symmetry in the generation of trial structures. *Acta Cryst.* A24, 301–311. The citation is confirmed from Dauter's reference list; the title is **[K]**. — [PMC4005360](https://pmc.ncbi.nlm.nih.gov/articles/PMC4005360/)
- **[S]** One search snippet, apparently Hirshfeld's abstract, says the transformations of the crystal axes that leave the coordinates of equivalent positions invariant define a derivative symmetry, conforming to one of thirty distinct "Cheshire" groups. The attribution and the count of 30 are unverified. — search result, no direct URL
- **[V]** CCP4 "Alternate origins for the 65 enantiomorphic space groups" lists discrete origin shifts plus free axes, sourced from ITB 2nd ed. (2001) Tables 2.2.3.2/2.2.3.4. Examples:
  - P2₁: (0,y,0), (0,y,½), (½,y,0), (½,y,½), with y free.
  - P2₁2₁2₁: 8 origins from (0,0,0) to (½,½,½).
  - P4₁2₁2: (0,0,0), (0,0,½), (½,½,0), (½,½,½).

  It does not mention Cheshire groups or normalisers. — [CCP4 alternate origins](https://smb.slac.stanford.edu/facilities/software/ccp4/html/alternate_origins.html)
- **[K]** Program papers for citation (not fetched this session):
  - Navaza, J. (1994). AMoRe: an automated package for molecular replacement. *Acta Cryst.* A50, 157–163.
  - Vagin, A. & Teplyakov, A. (1997). MOLREP. *J. Appl. Cryst.* 30, 1022–1025.
  - Vagin & Teplyakov (2010). *Acta Cryst.* D66, 22–25.
  - McCoy, A. J. et al. (2007). Phaser crystallographic software. *J. Appl. Cryst.* 40, 658–674.
  - Brünger, A. T. et al. (1998). CNS. *Acta Cryst.* D54, 905–921.
  - Evans, P. & McCoy, A. (2008). An introduction to molecular replacement. *Acta Cryst.* D64, 1–10. This is PMC2394790, surfaced by search.

### Inferences
- The MR Cheshire cell is exactly the **translation part** of the Euclidean normaliser N_E(G) modulo the lattice of G: the cell of additional origin shifts, continuous along polar axes. MR stops there.
  - Before the translation function the orientation is fixed by the rotation function, and proteins are chiral.
  - So MR never quotients by the normaliser's point operations: inversion, the extra rotations, metric-specialisation permutations.
  - The requester's "asymmetric unit of the Euclidean normaliser" applied jointly to (orientation, position) is therefore strictly finer than standard MR practice.
- Dauter's **"anti-Cheshire"** symmetry is the chirality-preserving normaliser: the Euclidean normaliser minus its improper operations, tabulated by Bilbao. It is the conceptually correct MR analogue for proper-rotation-only rigid-body placement. For achiral or racemic CSP the full Euclidean normaliser applies, with inversion mapping the molecule to its mirror image; that is useful only if the molecule's mirror image is congruent to it or the enantiomer is allowed.
- "First molecule in the Cheshire cell, later molecules in the full cell" is MR's sequential symmetry-breaking for Z′ > 1. The normaliser is used once, to pin the global gauge, and later molecules see only G.

### Gaps
- MOLREP's and CNS/X-PLOR's documented search volumes (Cheshire versus full or primitive cell) were not verified this session.
- No program page found in this session describes the Cheshire cell for **centred** cells and polar axes as an explicit algorithm. The behaviour is stated only at keyword level.

## Q3. 6-D stochastic searches (EPMR, Queen of Spades, SoMoRe, GA): bounds and symmetry or origin redundancy

### Takeaway
None of the 6-D or 6n-D stochastic MR programs checked this session uses the asymmetric unit of the normaliser.
- EPMR (1999) draws random positions "in the unit cell" and samples all of rotation space uniformly with quaternions. The paper never mentions Cheshire cells, polar origins or normalisers.
- Queen of Spades assigns "random orientations and translations to all molecules", with no symmetry-reduced bounds stated on the page checked.
- These programs accept redundancy: multiple symmetry-equivalent optima in a larger box. They do not reduce the domain.

### Cited findings
- **[V]** EPMR (Kissinger, C. R., Gehlhaar, D. K. & Fogel, D. B. (1999). Rapid automated molecular replacement by evolutionary search. *Acta Cryst.* D55, 484–491. doi:10.1107/S0907444998012517). From the full text:
  - The starting population is "generated by assigning random values to the six rigid-body parameters describing the orientation and position of the search model in the unit cell".
  - Rotations are quaternions, chosen so the initial population "evenly sample[s] rotation space" and mutated by Gaussian random rotations about random axes.
  - Initial mutation sizes are 6° and 3 Å.
  - Multiple molecules are searched sequentially, with earlier solutions fixed.
  - A full-text grep found no occurrence of "Cheshire", "normaliz" or "origin" in a symmetry sense.

  — [EPMR paper PDF](https://people.mbi.ucla.edu/sawaya/tutorials/Molrep/manuals/epmr.pdf)
- **[V]** The EPMR paper cites prior 6-D work: Rabinovich & Shakked (1984); Tong (1996); Chang & Lewis (1997), which used genetic algorithms; and DeLano & Brünger (1995) for the direct rotation function. — same PDF
- **[V]** Queen of Spades manual (Table 1, step 6): "Assign random orientations and translations to all molecules in the asymmetric unit". No ranges, Cheshire or polar handling appear on that page. — [Qs manual node23](https://utopia.duth.gr/glykos/Qs-html/node23.html)
- **[S]** Glykos, N. M. & Kokkinidis, M. (2000). A stochastic approach to molecular replacement. *Acta Cryst.* D56, 169–174. doi:10.1107/S0907444999015322. It is described as reverse Monte Carlo minimisation in a 6n-dimensional space, determining all molecules simultaneously, with all data expanded to P1. — [Wiley listing](https://onlinelibrary-wiley-com-443.webvpn.zafu.edu.cn/doi/abs/10.1107/S0907444999015322); [IUCr newsletter](https://www.iucr.org/news/newsletter/volume-11/number-2/structure-determination)
- **[S]** Glykos & Kokkinidis (2001). Multidimensional molecular replacement. *Acta Cryst.* D57, 1462–1473 (pages **[K]**). — [IUCr D 2001/10 ba5005](https://journals.iucr.org/d/issues/2001/10/00/ba5005)
- **[S]** Kissinger et al. (2001), CCP4 study weekend, *Acta Cryst.* D57 (ba5011): evolutionary and simulated-annealing methods are several orders of magnitude faster than a full 6-D search. — [IUCr D 2001 ba5011](https://journals.iucr.org/d/issues/2001/10/00/ba5011/)
- **[K]** Jamrog, D. C., Zhang, Y. & Phillips, G. N. Jr (2003). SoMoRe: a multi-dimensional search and optimization approach to molecular replacement. *Acta Cryst.* D59, 304–314. The search-region handling was not checked.
- **[K]** Chang, G. & Lewis, M. (1997). Molecular replacement using genetic algorithms. *Acta Cryst.* D53, 279–289.

### Inferences
- No 6-D MR program is known to use the asymmetric unit of the normaliser for the joint (R, t) space. The requester's reduced cell × normaliser asymmetric unit × orientation fundamental zone appears to go beyond MR practice. Its closest MR precedents are, separately:
  - the rotation-function asymmetric unit (Q1);
  - the Cheshire cell (Q2).
- Stochastic programs can afford redundancy, an extra factor of |N_E : G| times the rotation-function redundancy, because they report one optimum. A sampler that must estimate density or count distinct basins cannot afford it.

### Gaps
- I did not obtain the Qs papers' full text or the SoMoRe paper to confirm the absence of Cheshire or normaliser bounds. Treat "not used" for these two as unconfirmed.
- I did not check whether later EPMR versions (openEPMR) restrict translations to the Cheshire cell.

## Q4. Cheshire groups = Euclidean normalisers in MR literature; solution-comparison and alternative-origin tools

### Takeaway
Dauter (2013) states explicitly, citing Fischer & Koch (1983), that "the Cheshire symmetry corresponds to the Euclidean normalizer". He proposes **"anti-Cheshire"** (chirality-preserving normaliser) asymmetric units as a standard placement convention for macromolecules.

In MR, normaliser and origin equivalence is used mainly **after** the search, to compare and match solutions:
- phenix.find_alt_orig_sym_mate searches all symmetry operations × alternative origin shifts, and slides along polar axes;
- CCP4 alternate-origin tables list the allowed origin shifts.

### Cited findings
- **[V]** Dauter (2013), *Acta Cryst.* D69, 872–878:
  - "the Cheshire symmetry corresponds to the Euclidean normalizer of a particular space group", citing Fischer, W. & Koch, E. (1983), *Acta Cryst.* A39, 907–915, and Koch, E. & Fischer, W. (2006), *Z. Kristallogr.* 221, 1–14.
  - Cheshire groups and cells are tabulated in ITA Part 15 (2005).
  - Most Cheshire groups are centrosymmetric. The exceptions are those with a single chiral screw handedness (3₁, 3₂, 4₁, 4₃, 6₁, 6₂, 6₄, 6₅).
  - He proposes "anti-Cheshire" symmetry that keeps only proper operations, with chirality-preserving normalisers taken from the Bilbao server (Aroyo et al. 2006, *Z. Kristallogr.* 221, 15–27).
  - Table 1 lists groups needing re-indexing (for example h,k,−l for P4, P6) to place a structure in the optimal anti-Cheshire asymmetric unit.
  - On the field: "There are currently no rules for a unified, standard way of placing macromolecular structures in the crystal lattice."

  — [PMC4005360](https://pmc.ncbi.nlm.nih.gov/articles/PMC4005360/)
- **[V]** phenix.find_alt_orig_sym_mate:
  - It compares two MR solutions of the same data.
  - For each moving chain it finds the copy closest to the fixed chains over all space-group symmetry operations **and alternative origin shifts**, using SSM alignment and an MLAD score (good match < 1.5, bad > 2.0).
  - For floating-origin groups it slides the copy along the polar axis.
  - Citation: Oeffner, R. D., Bunkóczi, G. & Read, R. J., *Computational Crystallography Newsletter*, January 2012, 5–10.

  — [Phenix docs](https://phenix-online.org/documentation/reference/find_alt_orig_sym_mate.html)
- **[V]** CCP4 alternate-origins document for the 65 Sohncke ("enantiomorphic") groups: see Q2. — [CCP4](https://smb.slac.stanford.edu/facilities/software/ccp4/html/alternate_origins.html)
- **[V]** Phaser 2.1 documentation covers solution handling only as duplicate pruning, PURGe and PERMutations of search order, plus SGALternative for alternative space groups in the same point group, for example enantiomorph pairs. It does not discuss alternative origins. — [Phaser 2.1 keywords](https://www-structmed.cimr.cam.ac.uk/phaser_obsolete/documentation/phaser-2.1_key.html)
- **[K]** CCP4 csymmatch (K. Cowtan) moves one model onto another using symmetry and allowed origin shifts. Not fetched this session.

### Inferences
- MR literature recognises Cheshire = Euclidean normaliser, but mostly uses it as an equivalence relation for comparing solutions and depositing structures, not as a search-domain reduction beyond the translational Cheshire cell.
- Dauter's anti-Cheshire asymmetric unit is the closest published MR concept to the requester's normaliser asymmetric unit for position. It is not a joint (R, t) fundamental domain, because Dauter places only the molecular centre and the orientation is untouched.
- "Alternative indexing" (Dauter's re-indexing table) is the MR or data-processing name for normaliser rotations not in the Laue group. In CSP without data these become true redundancies to quotient out.

### Gaps
- I did not fetch ITA Part 15 (Koch & Fischer, "Normalizers of space groups") or Fischer & Koch (1983) directly.
- The exact page range of the Oeffner et al. newsletter is as stated in the Phenix docs only.

## Q5. Powder direct-space structure solution (DASH, FOX, TOPAS, PSSP, EXPO, Organa): do they restrict position or orientation by Cheshire cells or normalisers?

### Takeaway
I found **no** evidence that any powder direct-space program uses Cheshire cells or Euclidean normalisers to bound molecule position or orientation.
- FOX/ObjCryst++ has an `AsymmetricUnit` class, a box-shaped "pseudo-asymmetric unit" of the space group G built by grid sampling or from cctbx's brick. That box is G's asymmetric unit, not the normaliser's.
- DASH's 2005 constraints work reduces degrees of freedom through crystallographic constraints and notes floating origins (P1), but its exact bounds were not verified.

### Cited findings
- **[V]** Favre-Nicolin, V. & Černý, R. (2002). FOX, 'free objects for crystallography'. *J. Appl. Cryst.* 35, 734–743. doi:10.1107/S0021889802015236. It is a global-optimisation direct-space program with molecule and polyhedron building blocks and "automatic handling of special positions" (abstract-level). — [UNIGE archive](https://archive-ouverte.unige.ch/unige:35483); [Wiley](https://onlinelibrary-wiley-com-443.webvpn.zafu.edu.cn/doi/10.1107/S0021889802015236)
- **[S]** ObjCryst++ `AsymmetricUnit` has these properties (search summary of the source code docs):
  - It is constructed from a `SpaceGroup` and offers `IsInAsymmetricUnit(x,y,z)` plus Xmin..Zmax bounds.
  - For P1 it returns early.
  - Otherwise it grows a box until grid points in the cell are covered, or builds it from `cctbx::sgtbx::brick` under a different build flag.
  - How random moves use it was not confirmed.

  — [ObjCryst++ docs source](https://objcryst.readthedocs.io/en/latest/a00134_source.html)
- **[S]** DASH (2005 *J. Appl. Cryst.* paper, hx5022): crystallographic constraints "reduce number of degrees of freedom to be optimized during search", and the paper notes floating origins in P1. Snippet only; fetching the full text returned 403. — [IUCr hx5022 PDF](https://iucrdata.iucr.org/j/issues/2005/02/00/hx5022/hx5022.pdf)
- **[S]** Kabova et al. (2017), *J. Appl. Cryst.*, tuned DASH simulated-annealing parameters with irace for up to an order-of-magnitude higher success rate. It is not known whether those parameters include position bounds. — [Reading CentAUR](https://centaur.reading.ac.uk/72695/7/DASH_SA_Opimisation.pdf)
- **[S]** PSSP: simulated annealing against Le Bail–extracted intensities, aimed at rigid and flexible molecular solids. — [IUCr PSSP](https://journals.iucr.org/paper?S0021889810005509=)
- **[S]** EXPO: direct-space SA, parallelised, combined with direct methods. — [EXPO direct-space page](https://www.ba.ic.cnr.it/softwareic/expo/structure-solution-by-direct-space-methods/)
- **[K]** Citations not verified this session:
  - David, W. I. F., Shankland, K., van de Streek, J., Pidcock, E., Motherwell, W. D. S. & Cole, J. C. (2006). DASH: a program for crystal structure determination from powder diffraction data. *J. Appl. Cryst.* 39, 910–915.
  - Coelho, A. A. (2000). Whole-profile structure solution from powder diffraction data using simulated annealing. *J. Appl. Cryst.* 33, 899–908 (TOPAS).
  - Pagola, S. & Stephens, P. W. (2010). PSSP. *J. Appl. Cryst.* 43, 370–376.
  - Brodski, V., Peschar, R. & Schenk, H. (2005). Organa. *J. Appl. Cryst.* 38, 688–693.
  - Altomare, A. et al. (2013). EXPO2013. *J. Appl. Cryst.* 46, 1231–1235.

### Inferences
- Powder direct-space codes typically let molecular centres range over the unit cell, or at most over G's own asymmetric unit, and orientations over all of SO(3) (quaternions in DASH and TOPAS, **[K]**). They tolerate the |N_E : G| origin and normaliser redundancy because they seek one global χ² minimum.
- The precedent for normaliser-reduced direct-space search therefore seems to be MR's Cheshire cell together with Hirshfeld (1968), not the powder literature. This matches the requester's need for something stricter than either.

### Gaps
- I could not read the DASH, TOPAS, Organa or GRASP manuals or papers to state their default translation ranges or polar-axis handling.
- No source found mentions "Cheshire" or "normaliser" in any powder SDPD program paper. This is an absence-of-evidence finding from limited searches, not a proof of absence.
