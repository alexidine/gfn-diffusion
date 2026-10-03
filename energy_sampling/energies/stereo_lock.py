"""The STEREO LOCK: a flat-bottom energy term that makes one labelled stereoisomer the target.

WHY. Every term of the conformer force field is a function of distances, or of dihedrals
through terms that are even under reflection, so a mirrored conformer scores identically and
nothing in the reward knows which stereoisomer a condition names. At `full` (and `dihedral`,
`flex`) the chart reaches the mirror image of every tetrahedral centre and both sides of every
double bond inside its box, so an unlocked target is the mixture of every labelled
configuration the box holds -- not the condition's isomer. The lock restricts it to one.

WHAT IS LOCKED. Two kinds of element, both keyed on atoms of the condition graph:

  * every atom with FOUR bonded neighbours (``TETRAHEDRAL``) -- not only RDKit's
    stereocentres. At a CH2 or CH3 the two labelled parities are exact copies (a permutation
    of identical atoms), and the fitted prior's held rows visit only the reference's (it
    reflects three-coordinate centres only, energies/invertible_centres.py); locking the
    labelled parity everywhere makes the target exactly one labelled copy, whose partition
    function differs from the unlocked one by the molecule's symmetry count (ln 2 per
    independent centre), instead of 2^m copies the prior never proposes;
  * every double bond RDKit reports as POTENTIAL stereo (``DOUBLE_BOND``) on the input
    molecule, perceived on both the heavy-atom and the explicit-H graph (`tagged_elements`).

A three-coordinate centre is NOT locked by default: RDKit's perception from 3D does not recover
N stereo (measured again 2026-10-03 on the stereo nitrogens below: no N tag on any of 603
embedded QM9 isomers, under the legacy or the new perception), and the two sides of a
three-coordinate centre stay
together inside one condition (a working assumption, stated for N invertomers in
build_conformer_conditions.py::stereo_mol; see the wiki page on the force field). It covers
EVERY three-coordinate atom, not only an amine N: a sulfoxide S or a phosphine P is left free
too, although its inversion is slow and its configuration a real stereocentre, so a condition
containing one holds both configurations, and with the lock on a tag on one is refused
(`stereo_unsupported`, ConformerTorsions._init_stereo); QM9 has neither. The fitted prior draws
both sides of each such centre energies/invertible_centres.py qualifies: one with a substituent
offset whose sign can be negated as an exact inversion without turning its ring system, a ring
N at its ring's closure included. That module names what it misses: a ring N at the tree's
root, and one the tree enters from its substituent; a caged N has no second side. The tags are
stripped wherever isomers are compared (`_strip_invertible`). A four-coordinate N+ is an
ordinary locked centre.

STEREO NITROGENS (`ConformerTorsions(lock_stereo_nitrogen=True)`; off by default. Owner
decision 2026-10-03: lock the slow-flipping nitrogens by the rules the rest of the stereo
handling uses). A STEREO NITROGEN (`stereo_nitrogens`) is an N with three single bonds to three
non-hydrogen neighbours and no hydrogen, which `Chem.FindPotentialStereo` reports as a
tetrahedral element on the molecule AS PARSED from its SMILES (implicit hydrogens): RDKit's own
rule, an N in a three-membered ring or a bridgehead N, less the ones symmetry removes. An N-H
is never one: RDKit drops a tag on it at the parse, so no SMILES can name its configuration.
With the option on, a stereo nitrogen the condition's SMILES TAGS is a third kind of locked
atom: its tag stays in every isomer comparison (`_strip_invertible`'s ``keep``), a SMILES that
leaves a stereogenic one untagged is `stereo_unspecified`, and its element is a ``TETRAHEDRAL``
one whose quad is its three neighbours and itself -- the one triple product a three-coordinate
centre has, the form of a four-coordinate centre's "three bond directions" candidates. An
untagged stereo nitrogen that no enumeration distinguishes (a symmetric cage's) stays free, so
a SMILES without an N tag builds, with the option on, the member it builds with it off.
THE SIGN IS STILL READ OFF THE REFERENCE, and because 3D perception returns nothing for N the
reference is VERIFIED against the tag by the element's own sign. RDKit's tag convention is the
sign of the chiral volume of the centre's first three neighbours in its bond order
(``CHI_TETRAHEDRAL_CCW`` positive: `nitrogen_tag_sign`, the rule RDKit's 3D perception applies
to carbon, pinned by tests/conformer/test_stereo_nitrogen.py against that perception), carried
to the quad's neighbour order by the parity of the permutation (`quad_sign_of_tag`). A reference
whose element has the other sign is `stereo_verify_failed`; one below `MIN_MARGIN` is
`stereo_lock_in_band`: refused, not left free, because the tag is in the condition's name and a
near-planar reference cannot show which side it names. Measured on 611 tagged stereo nitrogens
of 400 QM9 molecules (ETKDGv3 seed 0 + MMFF94): |v| from 0.655 to 0.982, median 0.779, none
below 0.5; 610 on the tagged side, one inverted by the MMFF relaxation and so refused.

THE INDICATOR. Per element a signed, normalised four-point quantity of the positions of four
atoms ``p1..p4`` (the element's QUAD):

    tetrahedral  v = (p1-p4).[(p2-p4) x (p3-p4)] / (|p1-p4| |p2-p4| |p3-p4|)
    double bond  q = -[(a-b) x (c-b)].[(b-c) x (d-c)] / (|a-b| |c-b| |b-c| |d-c|)
                   = -sin(t1) sin(t2) cos(phi)          for the quad (a, b, c, d)

The tetrahedral form covers two families: p4 one of the four neighbours (the volume of the
neighbour tetrahedron), or p4 the centre itself with p1..p3 three of its neighbours (the
triple product of three bond directions). Eight candidates per centre; the one with the
largest |v| at the REFERENCE conformer is used (ties within `TIE_TOL` go to the lowest
candidate index, `_best_quad`). THIS CHOICE is what keeps the lock off the correct isomer at
strained fused three-ring bridgeheads, where the neighbour tetrahedron can be nearly flat. At
the untagged CC(C=O)N1C2CCC12's ETKDG reference one bridgehead's neighbour tetrahedron has |v|
0.090, and on an MMFF94 chain from it (`mmff_thermal_samples`, 20000 steps, 899 samples) its
sign flips on 516 samples that RDKit re-perceives as the same isomer, at the same mean MMFF
energy as the rest; the best candidate there has |v| 0.73 and never enters its band
(tests/conformer/test_stereo_lock.py). No single indicator is right at every strained centre,
though: at the untagged COCC1(O)C2COC12's reference the chain leaves a metastable basin for a
lower one RDKit also calls the same isomer, and there it is the best candidate that flips.
`thermal_check` measures exactly this, and build_conformer_conditions.py refuses a molecule it
fires on. For a double bond the four (a, d) choices are equivalent and the same rule picks
one.

THE SIGN IS READ OFF THE REFERENCE, never off CIP labels or tags: ``s = sign(v_ref)``, and the
lock is

    P = stereo_coeff * sum_e relu(lo_e - s_e * v_e)^2          [kcal/mol]

exactly zero wherever every element keeps its reference sign with margin ``lo_e``, so the
density and its gradient inside the locked stereoisomer's basins are untouched. It is C1 and
bounded (each |v| <= 1). ``lo_e = min(LO_MAX[kind], LO_FRAC * |v_ref|)``: a per-element band
that narrows only for an element whose best indicator is small at the reference. With the
refusal floor `MIN_MARGIN` at 0.4 it can bind only between 0.4 and 0.5 (tetrahedral) or 1.0
(double bond); on the QM9 census every tetrahedral element sits above 0.5, so there it is the
double bonds' band only, and the best-of-eight choice above does the tetrahedral work.

A POTENTIAL, NOT A MEASURE TERM. P is in kcal/mol and enters the potential BEFORE
`energy_clip`'s compression, so the owner's cap bounds the total, and it is divided by T
exactly as U is -- which is what makes a live row and a baked row (stored at T = 1, divided by
T on the read side) score it identically at every temperature. The consequence is stated: the
lock softens with T like any potential term does. Identically only at ONE coefficient: a baked
row holds the lock at the coefficient it was scored under, so `stereo_coeff` is fixed for the
life of an energy (its setter refuses a change) and every condition graph records it
(``ctree_stereo_coeff``).

WHERE IT IS COMPUTED. `ConformerTorsions.potential_energy` from its own `StereoTable`, and
`MultiConformerTorsions._energy_one_pass` graph-natively from the ``ctree_stereo_*`` fields a
condition graph carries (`graph_fields`, `batch_lock_energy`): the graph binds the sign to the
reference geometry it was read from, in one object.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

TETRAHEDRAL, DOUBLE_BOND = 1, 2
KIND_NAMES = {TETRAHEDRAL: 'tetrahedral', DOUBLE_BOND: 'double bond'}
#: ceiling of the in-band threshold per kind, in units of the normalised indicator. The
#: tetrahedral value is the design's, the double-bond one likewise; `LO_FRAC` narrows either
#: for an element whose reference value is small.
LO_MAX = {TETRAHEDRAL: 0.1, DOUBLE_BOND: 0.2}
#: the band as a fraction of the element's own reference margin |v_ref|
LO_FRAC = 0.2
#: an element whose best indicator is below this at the reference is REFUSED when the lock is
#: on (`stereo_lock_in_band`). Set by the penalty the WRONG configuration pays: at its mirror
#: point (v = -m) that is k (lo + m)^2, which with lo = min(LO_MAX, LO_FRAC m) is 1.44 k m^2 for
#: a thin element -- about 1 kcal/mol at m = 0.05 and k = 300, not a lock at all. At 0.4 it is
#: 300 (0.08 + 0.4)^2 = 69 kcal/mol at k = 300. The QM9 census's smallest reference margins
#: (realised isomers plus up to two other stereoisomers each) are 0.645 (tetrahedral) and 0.623
#: (double bond), so no census molecule sits near it.
MIN_MARGIN = 0.4
#: two candidate quads whose |v| at the reference differ by less than this are a TIE, broken by
#: the lowest candidate index. Symmetric substituents make exact ties (the three H-H-X triples
#: of a CH3 group), and an argmax over them follows roundoff in the reference geometry, so the
#: same molecule embedded on another machine could store another quad -- which `_resolve_rows`
#: compares atom by atom and would refuse as another stereoisomer. Indicator units. It moves
#: the sensitivity from gaps near 0, which symmetry makes common, to gaps near TIE_TOL itself:
#: 26 of 24,576 elements on the QM9 census (realised isomers plus up to two other
#: stereoisomers each) have a candidate within 1e-5 of that edge. There a reference that
#: differs by roundoff can still store another quad, and the row check refuses it loudly.
TIE_TOL = 1e-3
#: MMFF94 Metropolis steps of the build-time thermal check (`thermal_check`)
THERMAL_CHECK_STEPS = 20000
#: floor on every length in an indicator's normaliser, in Angstrom. Two substituents that
#: overlap make a normalising length vanish, and d(1/|x|) grows as 1/|x|^2; with the floor the
#: value stays continuous (its numerator vanishes with the length) and the gradient bounded.
#: Far below any bonded or nonbonded distance a physical conformer reaches, so the value is
#: unchanged wherever the force field is finite.
NORM_FLOOR = 0.1


# ----------------------------------------------------------------------------- perception


@contextmanager
def pinned_perception():
    """RDKit's stereo perception with the LEGACY algorithm, whatever the process default is.

    `Chem.SetUseLegacyStereoPerception` is process-global and RDKit has announced a change of
    its default. Under the non-legacy algorithm `AssignStereochemistryFrom3D` marks spurious
    'specified' centres (a gem-dimethyl carbon, a symmetric spiro carbon) on a measured 313
    of 2948 QM9 census molecules, and it enumerates cis-1,3-dimethylcyclobutane twice. Every
    perception in this module runs inside this context, and every `FindPotentialStereo` call
    passes ``cleanIt=True`` explicitly, so the lock set and the verify step cannot move with
    another module's setting.
    """
    from rdkit import Chem
    prev = Chem.GetUseLegacyStereoPerception()
    Chem.SetUseLegacyStereoPerception(True)
    try:
        yield
    finally:
        Chem.SetUseLegacyStereoPerception(prev)


def _find_potential(mol) -> List[tuple]:
    """``[(kind, key, specified, stereo_type_name)]`` on a COPY, keys in `mol`'s indexing.

    key is ``(atom,)`` for a tetrahedral element and ``(b, c)`` sorted for a double bond.
    """
    from rdkit import Chem
    m = Chem.Mol(mol)
    n_double = lambda a: sum(1 for b in m.GetAtomWithIdx(a).GetBonds()
                             if b.GetBondType() == Chem.BondType.DOUBLE)
    out = []
    for si in Chem.FindPotentialStereo(m, cleanIt=True):
        spec = si.specified == Chem.StereoSpecified.Specified
        if si.type == Chem.StereoType.Atom_Tetrahedral:
            out.append((TETRAHEDRAL, (int(si.centeredOn),), spec, 'tetrahedral'))
        elif si.type == Chem.StereoType.Bond_Double:
            bd = m.GetBondWithIdx(int(si.centeredOn))
            key = tuple(sorted((bd.GetBeginAtomIdx(), bd.GetEndAtomIdx())))
            # A CUMULATED double bond (an allene's) is axial stereo in RDKit's double-bond
            # form: its far substituent is collinear with the bond, so the E/Z indicator is
            # identically 0, and the two bonds of one allene share their centre atom. Kind 0:
            # not locked, refused when the lock is on.
            if max(n_double(key[0]), n_double(key[1])) > 1:
                out.append((0, key, spec, 'cumulated double bond'))
            else:
                out.append((DOUBLE_BOND, key, spec, 'double bond'))
        else:
            out.append((0, (int(si.centeredOn),), spec, str(si.type)))
    return out


def _is_stereo_nitrogen_atom(atom) -> bool:
    """The structural half of the STEREO NITROGEN rule (module docstring), on one atom."""
    from rdkit import Chem
    return (atom.GetAtomicNum() == 7 and atom.GetDegree() == 3 and atom.GetTotalNumHs() == 0
            and not atom.GetIsAromatic()
            and all(n.GetAtomicNum() != 1 for n in atom.GetNeighbors())
            and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()))


def stereo_nitrogens(mol) -> List[int]:
    """Atom indices of `mol`'s STEREO NITROGENS (module docstring), ascending.

    `mol` is the molecule AS PARSED (implicit hydrogens; an =NH hydrogen may be explicit). The
    explicit-H graph answers differently: there RDKit also reports an aziridine N-H, and does
    not report the N of 99 of the 9,866 QM9 constitutions that have a stereo nitrogen (census
    of all 133,641, 2026-10-03), so every caller passes the parsed view (`implicit_h_view`).
    Whether one is TAGGED is the caller's question.
    """
    with pinned_perception():
        return sorted(key[0] for kind, key, _, _ in _find_potential(mol)
                      if kind == TETRAHEDRAL
                      and _is_stereo_nitrogen_atom(mol.GetAtomWithIdx(key[0])))


def implicit_h_view(mol):
    """A copy of an explicit-H `mol` with its removable hydrogens removed, conformer kept, and
    its heavy atoms on THE SAME INDICES -- checked, since whatever is read off the view is
    carried back by index. True of ``AddHs(MolFromSmiles(s))``, which appends its hydrogens.
    """
    from rdkit import Chem
    h = Chem.RemoveHs(Chem.Mol(mol))
    for a in h.GetAtoms():
        if a.GetAtomicNum() != mol.GetAtomWithIdx(a.GetIdx()).GetAtomicNum():
            raise ValueError('removing hydrogens renumbered the heavy atoms; the molecule was '
                             'not built as AddHs(MolFromSmiles(smiles))')
    return h


def nitrogen_tag_sign(atom) -> int:
    """+1, -1 or 0: the sign RDKit's tetrahedral tag on a three-neighbour `atom` asks of the
    chiral volume ``(p1 - p0) . [(p2 - p0) x (p3 - p0)]`` of its neighbours p1..p3, taken in
    ITS BOND ORDER, about its own position p0. ``CHI_TETRAHEDRAL_CCW`` is positive: the rule
    RDKit's 3D perception applies to the first three neighbours of any centre.
    """
    from rdkit import Chem
    tag = atom.GetChiralTag()
    return (1 if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW
            else -1 if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else 0)


def bond_order_neighbours(atom) -> List[int]:
    """`atom`'s neighbours in its bond order, the order a chiral tag is written against."""
    return [int(b.GetOtherAtomIdx(atom.GetIdx())) for b in atom.GetBonds()]


def quad_sign_of_tag(tag_sign: int, neighbours: Sequence[int]) -> int:
    """The sign a tag asks of the lock's indicator on the quad ``sorted(neighbours) + [centre]``.

    ``neighbours`` are the centre's three neighbours IN BOND ORDER, in the numbering the quad
    is sorted in (placement slots, for a `StereoTable`). The triple product changes sign with
    each exchange of two neighbours, so the answer is `tag_sign` times the parity of the
    permutation that sorts them.
    """
    n = [int(a) for a in neighbours]
    if len(n) != 3 or len(set(n)) != 3:
        raise ValueError(f'need three distinct neighbours, got {n}')
    inversions = sum(1 for i in range(3) for j in range(i + 1, 3) if n[i] > n[j])
    return int(tag_sign) * (-1 if inversions % 2 else 1)


def assign_nitrogen_tags_from_3d(mol, atoms: Sequence[int]):
    """Tag each three-neighbour atom of `atoms` on `mol` (IN PLACE) from `mol`'s conformer.

    What RDKit's 3D perception does not do for N: the tag whose `nitrogen_tag_sign` is the
    sign of the chiral volume of the atom's three neighbours in its bond order. A volume of
    exactly zero leaves the atom untagged. Returns `mol`.
    """
    from rdkit import Chem
    pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    for i in atoms:
        a = mol.GetAtomWithIdx(int(i))
        nb = bond_order_neighbours(a)
        if len(nb) != 3:
            raise ValueError(f'atom {int(i)} has {len(nb)} neighbours; a stereo nitrogen has 3')
        u, v, w = (pos[k] - pos[int(i)] for k in nb)
        vol = float(np.dot(u, np.cross(v, w)))
        a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW if vol > 0.0
                       else Chem.ChiralType.CHI_TETRAHEDRAL_CW if vol < 0.0
                       else Chem.ChiralType.CHI_UNSPECIFIED)
    return mol


def tagged_elements(smiles: str) -> List[dict]:
    """The potential stereo elements of the INPUT molecule, and which ones its tags specify.

    Indexed like ``Chem.AddHs(Chem.MolFromSmiles(smiles))`` -- the molecule
    `ConformerTorsions` embeds, whose heavy atoms and heavy-atom bonds keep their parsed
    indices because AddHs appends.

    PERCEIVED TWICE AND UNITED, because each graph misses a class. On the explicit-H graph
    `FindPotentialStereo` does not report ring cis/trans centres that are unspecified
    (OC1CCC(O)CC1 comes back with none); on the heavy-atom graph it does not report a =NH
    imine (CC=N comes back with no double bond). An element reported on either is an element;
    it counts as specified only if no report calls it unspecified.

    ``[{'kind', 'atoms', 'specified', 'element', 'degree', 'type', 'stereo_nitrogen'}]``;
    ``degree`` is the key atom's total degree (hydrogens included), so a tetrahedral element
    of degree 3 is an invertible centre the lock leaves alone -- unless it is a
    ``stereo_nitrogen`` (module docstring: reported on the parsed graph, and passing
    `_is_stereo_nitrogen_atom`) and the energy was asked to hold those. ``kind`` 0 is a stereo
    type this module does not lock (an allene, an atropisomer), carried so a caller can refuse
    it.

    'Potential' is RDKit's word and is wider than stereogenic: a bridgehead of a symmetric
    cage, or a centre that is stereogenic only for some assignments of the others, is
    reported and never specified. Whether the tags pin ONE isomer is `consistent_isomers`'
    question, not this function's.
    """
    from rdkit import Chem
    with pinned_perception():
        mol0 = Chem.MolFromSmiles(smiles)
        if mol0 is None:
            raise ValueError(f'RDKit cannot parse {smiles!r}')
        molh = Chem.AddHs(mol0)
        merged: Dict[tuple, dict] = {}
        for m in (mol0, molh):
            for kind, key, spec, tname in _find_potential(m):
                k = (kind, key)
                if k in merged:
                    merged[k]['specified'] = merged[k]['specified'] and spec
                else:
                    at = molh.GetAtomWithIdx(key[0])
                    merged[k] = dict(kind=kind, atoms=key, specified=spec,
                                     element=at.GetSymbol(), degree=int(at.GetTotalDegree()),
                                     type=tname, stereo_nitrogen=False)
                if m is mol0 and kind == TETRAHEDRAL:
                    merged[k]['stereo_nitrogen'] = _is_stereo_nitrogen_atom(
                        mol0.GetAtomWithIdx(key[0]))
    return [merged[k] for k in sorted(merged)]


def _strip_invertible(mol, keep: Sequence[int] = ()):
    """`mol` (in place) with the chiral tag cleared on every three-coordinate atom not in `keep`.

    The lock does not pin an invertible centre, so a tag there names nothing the target
    distinguishes; every isomer comparison here is made without them. `keep` holds the atoms
    the lock DOES pin: the stereo nitrogens, under `lock_stereo_nitrogen`.
    """
    from rdkit import Chem
    keep = {int(i) for i in keep}
    for a in mol.GetAtoms():
        if (a.GetTotalDegree() == 3 and a.GetIdx() not in keep
                and a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED):
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return mol


def canonical_isomeric(smiles: str, lock_nitrogen: bool = False) -> str:
    """Canonical isomeric SMILES of the input, invertible-centre tags stripped, pinned
    perception. With `lock_nitrogen` a stereo nitrogen's tag is kept."""
    from rdkit import Chem
    with pinned_perception():
        m = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(_strip_invertible(
            m, stereo_nitrogens(m) if lock_nitrogen else ()))


def consistent_isomers(smiles: str, max_isomers: int = 4096,
                       lock_nitrogen: bool = False) -> List[str]:
    """Canonical isomeric SMILES of every stereoisomer the input's tags are consistent with.

    One entry means the tags pin a single isomer. RDKit's enumerator over the unassigned
    elements, on BOTH graphs for the reason `tagged_elements` gives (explicit H hides ring
    cis/trans, implicit H hides a =NH imine), each result canonicalised with invertible-centre
    tags stripped -- so an element that is 'potential' but not stereogenic given the others
    (a symmetric bridgehead, a centre whose two branches are made equivalent by the rest)
    collapses to one string instead of reading as unassigned.

    With `lock_nitrogen` the stereo nitrogens are stereo elements like any other: a tag on one
    is kept going in, so it is not re-enumerated, and kept coming out, so an input that leaves
    a stereogenic one untagged comes back as two isomers. Which atoms those are is read off
    each enumerated isomer's own hydrogen-free view, since symmetry can make it depend on the
    other centres' assignment.
    """
    from rdkit import Chem
    from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
                                                   StereoEnumerationOptions)
    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=max_isomers)
    out = set()
    with pinned_perception():
        mol0 = Chem.MolFromSmiles(smiles)
        if not lock_nitrogen:
            for m in (mol0, Chem.AddHs(mol0)):
                for iso in EnumerateStereoisomers(_strip_invertible(Chem.Mol(m)),
                                                  options=opts):
                    out.add(Chem.MolToSmiles(Chem.RemoveHs(_strip_invertible(Chem.Mol(iso)))))
            return sorted(out)
        keep0 = stereo_nitrogens(mol0)             # heavy-atom indices, which AddHs keeps
        for m in (mol0, Chem.AddHs(mol0)):
            for iso in EnumerateStereoisomers(_strip_invertible(Chem.Mol(m), keep0),
                                              options=opts):
                h = Chem.RemoveHs(Chem.Mol(iso))
                keep = stereo_nitrogens(h)
                if keep:
                    out.add(Chem.MolToSmiles(_strip_invertible(h, keep)))
                else:              # no nitrogen to keep: the string the option-off path writes
                    out.add(Chem.MolToSmiles(Chem.RemoveHs(_strip_invertible(Chem.Mol(iso)))))
    return sorted(out)


def realised_isomer(mol, lock_nitrogen: bool = False) -> str:
    """Canonical isomeric SMILES of the stereoisomer a 3D conformer REALISES.

    Stereo re-perceived from the coordinates of `mol`'s conformer (on a copy; `mol` is not
    touched), hydrogens removed except those RDKit keeps to define a double bond (the H of a
    =NH imine). Comparing it with `canonical_isomeric` of the input is the verify step: the
    reference embedding is the object the lock's signs are read from, so an embedding that
    realised another isomer would lock that one.

    With `lock_nitrogen`, a molecule that has a stereo nitrogen is perceived by
    `perceive_with_nitrogens` instead, which tags the nitrogens from the geometry too; one
    that has none returns the string the option-off path writes.
    """
    from rdkit import Chem
    with pinned_perception():
        m = Chem.Mol(mol)
        Chem.AssignStereochemistryFrom3D(m, replaceExistingTags=True)
        if not (lock_nitrogen and any(_is_stereo_nitrogen_atom(a) for a in m.GetAtoms())):
            return Chem.MolToSmiles(Chem.RemoveHs(_strip_invertible(m)))
        h = implicit_h_view(_strip_invertible(m))
        if not stereo_nitrogens(h):
            return Chem.MolToSmiles(h)
        h = perceive_with_nitrogens(mol)
        return Chem.MolToSmiles(_strip_invertible(h, stereo_nitrogens(h)))


def perceive_with_nitrogens(mol):
    """The hydrogen-free view of an explicit-H 3D `mol`, its stereo perceived from the
    geometry WITH the stereo nitrogens (a copy; `mol` is not touched).

    RDKit's `AssignStereochemistryFrom3D` is three steps: bond stereo from 3D, a raw tag on
    every centre from its chiral volume, then the legacy assignment, which drops the tags
    that are no stereocentre. It writes no raw tag on a three-coordinate N, and the last step
    then also drops a centre that is one only TOGETHER with that N (the bridgehead carbon
    facing a bridgehead N). So here the same three steps run with the N tags written between
    the second and the third (`assign_nitrogen_tags_from_3d`, on every N that passes
    `_is_stereo_nitrogen_atom`; the assignment drops the ones that are no stereocentre), and
    the third on the hydrogen-free view, the graph RDKit judges a stereo nitrogen on. Call it
    under `pinned_perception`.
    """
    from rdkit import Chem
    m = Chem.Mol(mol)
    Chem.DetectBondStereochemistry(m)
    Chem.AssignAtomChiralTagsFromStructure(m, replaceExistingTags=True)
    h = implicit_h_view(m)
    assign_nitrogen_tags_from_3d(
        h, [a.GetIdx() for a in h.GetAtoms() if _is_stereo_nitrogen_atom(a)])
    Chem.AssignStereochemistry(h, cleanIt=True, force=True)
    return h


# ----------------------------------------------------------------------------- indicator


def _length(v: torch.Tensor) -> torch.Tensor:
    """|v| floored at NORM_FLOOR, differentiable at v = 0 (the clamp stops the gradient there)."""
    return torch.sqrt((v * v).sum(-1).clamp_min(NORM_FLOOR ** 2))


def values_from_points(p: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
    """``p [..., 4, 3]`` quad positions, ``kind [...]`` -> ``[...]`` indicator values.

    Both forms are evaluated for every element and one is selected, so neither branch may
    produce a NaN: every normaliser is floored (`_length`).
    """
    p1, p2, p3, p4 = p.unbind(-2)
    v1, v2, v3 = p1 - p4, p2 - p4, p3 - p4
    vt = ((v1 * torch.linalg.cross(v2, v3, dim=-1)).sum(-1)
          / (_length(v1) * _length(v2) * _length(v3)))
    # (a, b, c, d) = (p1, p2, p3, p4); (b - c) = -(c - b)
    ab, cb, dc = p1 - p2, p3 - p2, p4 - p3
    n1 = torch.linalg.cross(ab, cb, dim=-1)
    n2 = torch.linalg.cross(-cb, dc, dim=-1)
    lcb = _length(cb)
    vb = -(n1 * n2).sum(-1) / (_length(ab) * lcb * lcb * _length(dc))
    return torch.where(kind == TETRAHEDRAL, vt, vb)


def _values_np(pos: np.ndarray, quads: np.ndarray, kinds: np.ndarray) -> np.ndarray:
    """The indicator on one conformer ``pos [N, 3]`` for candidate quads, float64 numpy."""
    p = torch.as_tensor(pos, dtype=torch.float64)[torch.as_tensor(quads, dtype=torch.long)]
    return values_from_points(p, torch.as_tensor(kinds)).numpy()


# ----------------------------------------------------------------------------- the table


@dataclass
class StereoTable:
    """The locked elements of one molecule, in PLACEMENT-SLOT numbering.

    ``kind [E]``, ``key [E]`` (the centre for a tetrahedral element, atom b for a double
    bond), ``quad [E, 4]``, ``sign [E]`` (+1/-1, off the reference), ``margin [E]`` = |v_ref|
    and ``lo [E]``. ``stereocentre [E]`` marks the tetrahedral elements RDKit perceives as
    stereo elements of the input molecule (the others are labelled-parity locks).
    """
    kind: np.ndarray
    key: np.ndarray
    quad: np.ndarray
    sign: np.ndarray
    margin: np.ndarray
    lo: np.ndarray
    stereocentre: np.ndarray
    _cache: Dict[tuple, tuple] = field(default_factory=dict, repr=False, compare=False)

    @property
    def n(self) -> int:
        return int(len(self.kind))

    def bonds(self) -> set:
        """``{frozenset((b, c))}`` of the locked double bonds, slot numbering."""
        return {frozenset((int(q[1]), int(q[2]))) for q, k in zip(self.quad, self.kind)
                if int(k) == DOUBLE_BOND}

    def element_of(self, key: int, kind: int = TETRAHEDRAL) -> int:
        """Index of the one element of `kind` keyed on slot `key`; KeyError otherwise."""
        hit = np.flatnonzero((self.key == int(key)) & (self.kind == int(kind)))
        if hit.size != 1:
            raise KeyError(f'{hit.size} elements of kind {kind} keyed on slot {int(key)}')
        return int(hit[0])

    def with_sign(self, sign) -> 'StereoTable':
        """A copy locking the given signs instead: another labelled configuration."""
        s = np.asarray(sign, dtype=np.int64).reshape(-1)
        if s.shape != self.sign.shape or not np.isin(s, (-1, 1)).all():
            raise ValueError(f'need {self.n} signs in {{-1, +1}}, got {s.tolist()}')
        return StereoTable(self.kind, self.key, self.quad, s, self.margin, self.lo,
                           self.stereocentre)

    def tensors(self, device, dtype):
        """``(quad, kind, sign, lo)`` as tensors, cached per (device, dtype)."""
        k = (str(device), dtype)
        if k not in self._cache:
            dev = torch.device(device)
            self._cache[k] = (torch.as_tensor(self.quad, dtype=torch.long, device=dev),
                              torch.as_tensor(self.kind, dtype=torch.long, device=dev),
                              torch.as_tensor(self.sign, dtype=dtype, device=dev),
                              torch.as_tensor(self.lo, dtype=dtype, device=dev))
        return self._cache[k]

    def values(self, pos: torch.Tensor) -> torch.Tensor:
        """``pos [B, N, 3]`` -> ``[B, E]`` indicator values."""
        quad, kind, _, _ = self.tensors(pos.device, pos.dtype)
        return values_from_points(pos[:, quad], kind)

    def lock_energy(self, pos: torch.Tensor, coeff: float) -> torch.Tensor:
        """``pos [B, N, 3]`` -> ``[B]`` lock potential, kcal/mol."""
        if self.n == 0:
            return pos.new_zeros(pos.shape[0])
        _, _, sign, lo = self.tensors(pos.device, pos.dtype)
        return float(coeff) * (torch.relu(lo - sign * self.values(pos)) ** 2).sum(-1)

    def describe_elements(self, z: Sequence[int]) -> str:
        from rdkit import Chem
        sym = Chem.GetPeriodicTable()
        parts = []
        for k, key, s, m, lo in zip(self.kind, self.key, self.sign, self.margin, self.lo):
            tag = 'T' if int(k) == TETRAHEDRAL else 'B'
            parts.append(f'{tag}{sym.GetElementSymbol(int(z[int(key)]))}{int(key)}'
                         f'{"+" if int(s) > 0 else "-"}(|v| {m:.2f}, lo {lo:.3f})')
        return ' '.join(parts)


def _best_quad(pos: np.ndarray, cands: List[List[int]], kind: int) -> Tuple[List[int], float]:
    """The candidate quad with the largest |indicator| at the reference, and its value.

    The LOWEST-INDEX candidate within `TIE_TOL` of the largest, not the argmax: the candidates
    come in a fixed order off the graph, so the choice among (near-)equal ones is a function of
    the graph, not of roundoff in the reference.
    """
    vals = _values_np(pos, np.asarray(cands, dtype=np.int64),
                      np.full(len(cands), kind, dtype=np.int64))
    a = np.abs(vals)
    i = int(np.flatnonzero(a >= a.max() - TIE_TOL)[0])
    return cands[i], float(vals[i])


def build_table(ref_pos: np.ndarray, bond_index_slot: np.ndarray,
                double_bonds: Sequence[Tuple[int, int]],
                stereocentres: Sequence[int] = (),
                nitrogens: Sequence[int] = ()) -> StereoTable:
    """The lock table of one molecule, everything in PLACEMENT-SLOT numbering.

    ``ref_pos [N, 3]`` the reference conformer; ``bond_index_slot [2, n_bonds]`` the full bond
    graph; ``double_bonds`` the potential-stereo double bonds; ``stereocentres`` the
    tetrahedral atoms RDKit calls stereo elements of the input (recorded, not used to select);
    ``nitrogens`` the three-neighbour stereo nitrogens to lock (module docstring, STEREO
    NITROGENS), empty unless the energy was asked to hold them.

    Tetrahedral elements are EVERY atom with four neighbours and each atom of ``nitrogens``,
    in ascending slot order, then the double bonds in ascending (b, c). A nitrogen's quad is
    its three neighbours, ascending, then itself: its one triple product. A deterministic
    function of the graph, the reference and the canonical placement order, so a condition
    graph built from the same member carries the same table.
    """
    pos = np.asarray(ref_pos, dtype=np.float64)
    n = len(pos)
    nbr: Dict[int, set] = {i: set() for i in range(n)}
    for u, v in np.asarray(bond_index_slot, dtype=np.int64).reshape(2, -1).T:
        nbr[int(u)].add(int(v))
        nbr[int(v)].add(int(u))
    kinds, keys, quads, signs, margins, los, sc = [], [], [], [], [], [], []
    sc_set = {int(a) for a in stereocentres}
    n_set = {int(a) for a in nitrogens}
    for c in sorted(n_set):
        if not 0 <= c < n or len(nbr[c]) != 3:
            raise ValueError(f'stereo nitrogen at slot {c}: its element needs exactly three '
                             f'bonded neighbours in the chart\'s graph, it has '
                             f'{len(nbr[c]) if 0 <= c < n else "no such slot"}')

    def add(kind, key, quad, val):
        kinds.append(kind)
        keys.append(key)
        quads.append(quad)
        signs.append(1 if val >= 0 else -1)
        margins.append(abs(val))
        los.append(min(LO_MAX[kind], LO_FRAC * abs(val)))
        sc.append(kind == TETRAHEDRAL and key in sc_set)

    for c in range(n):
        ns = sorted(nbr[c])
        if c in n_set:
            quad = ns + [c]                        # its three bond directions: one candidate
            add(TETRAHEDRAL, c, quad, float(_values_np(
                pos, np.asarray([quad], dtype=np.int64),
                np.full(1, TETRAHEDRAL, dtype=np.int64))[0]))
            continue
        if len(ns) != 4:
            continue
        cands = []
        for k in range(4):
            rest = [a for j, a in enumerate(ns) if j != k]
            cands.append(rest + [ns[k]])          # neighbour tetrahedron, apex ns[k]
            cands.append(rest + [c])              # three bond directions from the centre
        quad, val = _best_quad(pos, cands, TETRAHEDRAL)
        add(TETRAHEDRAL, c, quad, val)
    for b, c in sorted(tuple(sorted((int(x), int(y)))) for x, y in double_bonds):
        a_s = sorted(nbr[b] - {c})
        d_s = sorted(nbr[c] - {b})
        if not a_s or not d_s:
            raise ValueError(f'double bond {b}={c} has no substituent on one end; it cannot '
                             f'be a stereo element')
        quad, val = _best_quad(pos, [[a, b, c, d] for a in a_s for d in d_s], DOUBLE_BOND)
        add(DOUBLE_BOND, b, quad, val)

    as_i = lambda a, w=None: (np.asarray(a, dtype=np.int64) if w is None
                              else np.asarray(a, dtype=np.int64).reshape(-1, w))
    return StereoTable(kind=as_i(kinds), key=as_i(keys), quad=as_i(quads, 4),
                       sign=as_i(signs), margin=np.asarray(margins, dtype=np.float64),
                       lo=np.asarray(los, dtype=np.float64),
                       stereocentre=np.asarray(sc, dtype=bool))


# ----------------------------------------------------------------------------- build check


def mmff_thermal_samples(en, steps: int, seed: int = 0) -> torch.Tensor:
    """``[S, N, 3]`` float64, PLACEMENT order: RDKit MMFF94 Metropolis at kT = 1 kcal/mol.

    Cartesian single-atom moves from `en`'s reference conformer (``en.mol``), under RDKit's own
    force field -- a thermal sampler independent of the chart and of the lock, so a lock firing
    on its samples is the INDICATOR failing on the correct isomer, not the chart leaving it.
    Every 20th state after a 10% burn-in is kept.
    """
    from rdkit.Chem import rdForceFieldHelpers as FH
    props = FH.MMFFGetMoleculeProperties(en.mol)
    if props is None:
        raise ValueError(f'{en.smiles}: RDKit has no MMFF94 parameters for this molecule')
    ff = FH.MMFFGetMoleculeForceField(en.mol, props)
    x = np.asarray(en.mol.GetConformer().GetPositions(), dtype=np.float64)
    e = ff.CalcEnergy(x.ravel().tolist())
    rng = np.random.default_rng(seed)
    kept = []
    for s in range(int(steps)):
        i = rng.integers(len(x))
        y = x.copy()
        y[i] += rng.normal(0.0, 0.06, 3)
        ey = ff.CalcEnergy(y.ravel().tolist())
        if ey <= e or rng.random() < np.exp(-(ey - e)):
            x, e = y, ey
        if s > steps // 10 and s % 20 == 0:
            kept.append(x.copy())
    if not kept:
        raise ValueError(f'{steps} steps keep no sample; need more than 20')
    return torch.as_tensor(np.stack(kept)[:, np.asarray(en.spec.perm)])


def thermal_check(en, steps: int = THERMAL_CHECK_STEPS, seed: int = 0) -> dict:
    """Does the lock stay at EXACTLY 0 on thermal samples of the correct isomer?

    The build-time half of the zero claim, per molecule, because the QM9 census that
    calibrated the indicator is a SAMPLE (each molecule's ETKDG-realised isomer and up to two
    other stereoisomers) and the chain is the only check that sees a reference sitting in a
    metastable basin. Reads `en.stereo` whatever `en.stereo_coeff` is, since only whether
    P > 0 matters. Stochastic: a molecule whose ensemble just reaches its band can pass on one
    seed and fail on another. ``{'n', 'fired', 'min_excess'}``: samples, samples with P > 0,
    and the smallest ``s * v - lo`` over every element and sample (negative when one fired).
    """
    S = mmff_thermal_samples(en, steps, seed)
    if en.stereo.n == 0:
        return {'n': int(len(S)), 'fired': 0, 'min_excess': float('inf')}
    _, _, sign, lo = en.stereo.tensors('cpu', torch.float64)
    excess = sign * en.stereo.values(S) - lo
    return {'n': int(len(S)), 'fired': int((excess < 0).any(-1).sum()),
            'min_excess': float(excess.min())}


# ----------------------------------------------------------------------------- graph form


#: the per-atom condition-graph fields, all node-level and graph-relative (conformer_data's
#: storage rule): kind 0 = no element keyed on this atom
STEREO_FIELDS = ('ctree_stereo_kind', 'ctree_stereo_sign', 'ctree_stereo_nbr',
                 'ctree_stereo_lo')


def graph_fields(table: StereoTable, n_atoms: int, dtype) -> Dict[str, torch.Tensor]:
    """The table scattered onto its KEY atoms: kind, sign, quad as DELTAS from the key, lo.

    Deltas, like ``ctree_ref_*`` and ``ctree_closure``, because a buffer draw re-offsets atom
    indices by collation but never remaps their values. One element per key atom, which the
    two kinds guarantee: a tetrahedral key has four neighbours, or is a stereo nitrogen, whose
    three bonds are single, and a double-bond end has at most three with a double bond among
    them; an atom is the begin atom of at most one double bond unless it is cumulated --
    asserted rather than assumed.
    """
    kind = torch.zeros(n_atoms, dtype=torch.long)
    sign = torch.zeros(n_atoms, dtype=torch.long)
    nbr = torch.zeros(n_atoms, 4, dtype=torch.long)
    lo = torch.zeros(n_atoms, dtype=dtype)
    for k, key, q, s, l in zip(table.kind, table.key, table.quad, table.sign, table.lo):
        key = int(key)
        if int(kind[key]):
            raise ValueError(f'two stereo elements are keyed on atom {key}; the per-atom '
                             f'encoding holds one (see stereo_lock.graph_fields)')
        kind[key], sign[key] = int(k), int(s)
        nbr[key] = torch.as_tensor(np.asarray(q, dtype=np.int64) - key)
        lo[key] = float(l)
    return {'ctree_stereo_kind': kind, 'ctree_stereo_sign': sign, 'ctree_stereo_nbr': nbr,
            'ctree_stereo_lo': lo}


def atom_codes(table: StereoTable, n_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(kind * sign [N], nbr deltas [N, 4])`` per atom -- what the library check compares."""
    f = graph_fields(table, n_atoms, torch.float64)
    return ((f['ctree_stereo_kind'] * f['ctree_stereo_sign']).numpy(),
            f['ctree_stereo_nbr'].numpy())


def batch_lock_energy(batch, pos: torch.Tensor, coeff: float) -> torch.Tensor:
    """``[num_graphs]`` lock potential, graph-natively, from a batch's ``ctree_stereo_*``.

    ``pos [N_total, 3]`` in batch atom order (the one-pass build). The quad of the element on
    key atom i is ``i + ctree_stereo_nbr[i]``; each element's term is index-added onto its
    key atom's graph. The same `values_from_points` the member path uses, so the two agree to
    roundoff.
    """
    n_graphs = int(batch.num_graphs)
    kind = batch.ctree_stereo_kind.reshape(-1)
    key = torch.nonzero(kind, as_tuple=True)[0]
    out = pos.new_zeros(n_graphs)
    if key.numel() == 0:
        return out
    quad = key.unsqueeze(-1) + batch.ctree_stereo_nbr.reshape(-1, 4).index_select(0, key)
    v = values_from_points(pos[quad], kind.index_select(0, key))
    s = batch.ctree_stereo_sign.reshape(-1).index_select(0, key).to(pos.dtype)
    lo = batch.ctree_stereo_lo.reshape(-1).index_select(0, key).to(pos.dtype)
    term = float(coeff) * torch.relu(lo - s * v) ** 2
    return out.index_add(0, batch.batch.index_select(0, key), term)
