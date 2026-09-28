"""FREE INVERTIBLE CENTRES: the three-coordinate centres the prior draws on either side.

WHAT THE TARGET HOLDS. At `dihedral` and above the chart reaches both sides of every non-planar
centre: the two pyramids of a three-coordinate centre, and the two labelled parities of a
four-coordinate centre (docs/wiki/conformer-chart-and-internal-coordinates.md, the chirality
section). The stereo lock (energies/stereo_lock.py) pins every four-coordinate centre when
`stereo_coeff` > 0 and leaves every three-coordinate centre free: an amine N, a near-planar
amide N (NON-PLANAR below), a planar sp2 C, whose two sides are one geometry, and also a
sulfoxide S or a phosphine P, whose inversion is slow (QM9 has neither). The target holds both
sides of a FREE centre wherever both are low-energy minima; the barrier between them does not
enter, since the target is a distribution over the chart, not a dynamics.

THE SIDE IS A SIGN. A centre's side is the sign of the offset its substituent rows keep from
their group's leader or frame. `ConformerTorsions.sample_prior_states` holds that offset at its
reference value, sign included, because its magnitude is bond-angle geometry: a sibling at
`sibling_jitter_sigma` about the leader, a mixed group's substituent about the ring member that
leads it, a substituent hung off the ring frame by `ring_frame_groups` at the offset it has in
the reference conformer, an improper row at `improper_phi_sigma` about ph0. So on its own the
prior proposes the reference's side only. The two-sided draw NEGATES the offset on one fair coin
per centre and draw, magnitude and jitter unchanged (`reflect_phi`). At a planar centre the
offset v sits at 0 or pi, and the flip moves a drawn row by twice its distance from the plane.

ONE DEFINITION, EVERY READER. The prior draw and its density (`ConformerTorsions.
sample_prior_states`, `.prior_log_prob`), the oracle proposal
(energies/prior_diagnostics.py::oracle_logw), the offline eval's parity metric
(eval/conformer_model_eval.py::parity_coverage) and the log Z check's coverage label
(eval/conformer_logz_check.py::prior_held_parity_centres, `prior_coverage_bias`) all read
`centre_table`. Which centres have an entry, what each flip moves and what the prior flips are
functions of the chart's structure, its level and the lock's table: no energy is evaluated and
no reference value is read. The eval and the label also read each entry's `planar` mark
(NON-PLANAR below), which reads the reference values and the prior's widths; the prior's flip
does not. Where the lock's table can be read, the three-coordinate centres the eval requires on
both sides are exactly the non-planar ones among the centres the prior flips.

THE FLIP (`reflect_phi`), on the phi block in radians, by the centre's place in the tree
(`Centre.kind`):
  * ROOT: phi -> -phi on the improper rows at the root. Each is an angle at the root measured
    from the seed plane (slots 0, 1, 2), and `place_nerf` is odd in phi, so every
    improper-placed child lands on its mirror image through the seed plane;
  * SIBLING: phi_f -> 2 phi_p - phi_f on every row of the sibling group about (parent(c), c)
    but the PIVOT p, the mirror of c's star through the plane (parent(c), c, p). The pivot is
    the group's first sibling bonded to c by a RING bond when there is one, the ring member the
    draw leads a mixed group with, else the group's first row, the leader drawn from a
    histogram;
  * FRAME: phi_j -> 2 phi_ring - phi_j on every row of the group about (b, c) = (parent(c), c)
    when b and c are ring atoms, no row of the group places a child through a ring bond, and
    a ring bond joins c to a neighbour p outside the group (the lowest such slot), which is
    then c's ring-closure neighbour; phi_ring is the dihedral (a, b, c, p) on the draw's
    positions (`frame_dihedral`), and the flip the mirror through the plane (b, c, p) of c's
    two ring neighbours. This test is the table's own. `ConformerTorsions.ring_frame_groups`,
    which places such a group in the draw, applies its own, and the two differ where a
    substituent of c belongs to another ring system: at C3 and C4 of C1(C2CC2)CC1C1CC1 the
    table has FRAME entries for groups the draw leads from the substituent ring's row. The
    flip is an exact inversion either way (the next paragraph).
Every atom below a flipped child, and every atom placed on a frame that holds one, moves with
it as a rigid body turning about an axis through c (docs/design/internal_dof_ladder.md
section 5). The flip moves periodic phi columns only, so no box clamp can bind on it; it is its
own inverse, and |det| of the map is 1.

QUALIFIED: a centre has an entry only when its flip is an EXACT INVERSION of c -- every bond
length, bond angle and other centre's parity kept, and every stereo element's indicator but
that of a double bond c is an atom of -- by structure: every bonded neighbour of c the flip
does not move lies in the mirror plane, and the flip moves no atom of c's RING SYSTEM (the atoms
joined to c through ring bonds). So a centre qualifies when
  * every row the flip moves is `movable`: a row the level's state drives (`free_mask`) that is
    not the partner of a transverse row, which carries v rather than an angle. A ROOT flip's
    rows are held improper rows, which the state drives: `movable` asks whether the state
    drives a row, not whether the prior holds it;
  * its bonded neighbours are the flip's own frame (the seeds; the parent and the pivot; the
    parent and p) and the children the flip moves, so no other ring-closure neighbour sits off
    the mirror plane, as one would at a bridgehead;
  * the atoms the flip moves -- its rows' children, and every atom placed on a frame holding
    one (`_moved`) -- hold no atom of c's ring system: only substituents turn, a ring inside a
    substituent turning rigidly with it.
Planarity is no condition: a planar centre has an entry, marked `planar` (NON-PLANAR below). A
double bond c is an atom of is planar at c, and a flip there moves that bond's indicator by
turning a substituent through twice its distance from the plane. Measured 2026-09-28 on 2000
prior draws each (level full, MMFF, T = 1 kcal/mol, `stereo_coeff` 300, an empty
InternalPrior) of seven alkenes and an oxime (C/C=C/C, F/C=C/F, C/C=C/Cl, C/C=C/C(=O)N(C)C,
C/C=C/C=O, C/C(F)=C(/F)C, CC/C=C/CC, C/C=N/O): the locked double bond's indicator moved by at
most 0.13 between a draw and its unflipped twin, s v stayed at 0.37 or above against bands of
0.14 to 0.16, and the lock stayed 0 on every draw.
A DECISION made 2026-09-28 by the integrating agent under the owner's delegated judgement, not
an owner ruling (scope: the prior's flip, the eval's parity metric and the log Z check's label;
revisit when the owner rules on it): the rule is structural, with no energy bar. A proposal on
a high-energy side costs efficiency, not correctness. It replaced a 10 kT bar on the reflected
reference geometry, which with every rotor held there left one-sided the exact inversions of
crowded acyclic amines (the N of CCCN(C)C scored 150 kT, while the target holds half its mass on
each side). Under the same decision the prior flips a PLANAR centre too, with no width bar: a
planar centre is its own flip in distribution, so flipping it is harmless and costs one coin
per draw, while a width bar on the prior's flip would read the reference embedding and, the
widths scaling as sqrt(T), the temperature (NON-PLANAR below), and so make the flip set a
function of both rather than of the chart alone.
tests/conformer/test_invertible_centres.py measures the exactness: bond lengths and angles
unchanged at the reference and on perturbed states, the lock at 0 on flipped draws.

NON-PLANAR (`Centre.planar`, `MIRROR_WIDTHS`). The flip takes a row's offset v to -v,
|wrap(2 v)| away. An entry is `planar` when no flipped row lies more than MIRROR_WIDTHS of its
own prior width (`improper_phi_sigma`, `sibling_jitter_sigma`) from its flip; a planar centre
(v at 0 or pi) is its own flip. The mark decides what the eval requires on both sides
(`free_centres`) and what the log Z check's label may name (`prior_held_parity_centres`); the
prior's flip does not read it. The bar is in prior widths, so a NEAR-planar three-coordinate
centre whose reference sits a few degrees out of plane reads non-planar, and how far past the
bar it sits depends on the reference embedding and on T (MIRROR_WIDTHS gives the measurements).
No entry at a collective level (`torsion`), where one column drives several rows.

LOCK AND SCOPE. `LOCKED` when `stereo_coeff` > 0 and the lock's table holds a TETRAHEDRAL
element keyed on c (`StereoTable.kind`, `.key`); `UNKNOWN` when the lock is on and the table
cannot be read (`_lock_names`: no `key` or no `kind`, or the two of different lengths); `FREE`
otherwise. A DOUBLE_BOND element is keyed on one of its double-bond atoms, the lower slot, and
locks the bond's E/Z, not that atom's side, so it names no centre: the key atom of a locked
double bond is FREE (C0 of C/C=C/C(=O)N(C)C at `stereo_coeff` 300). Until 2026-09-28 every key
was read as a name, so that atom read LOCKED under the lock and FREE without it, neither flipped
nor required under the lock. The prior flips a FREE centre with THREE bonded neighbours, planar
or not (`Centre.invertible`): pyramidal inversion only, in every lock state, a second DECISION
of the same date and standing. The eval requires both sides of every centre the lock does not
name that is not planar (`free_centres`), and the log Z check's label names the non-planar
centres the prior does not flip (`prior_held_parity_centres`). A FREE four-coordinate centre --
every sp3 centre when `stereo_coeff` is 0, the labelled parities of a CH2 or CH3 included --
stays one-sided as before although the unlocked target holds both of its sides: a KNOWN
LIMITATION, which the eval's parity metric still reads and the log Z check's label names. The
eval counts an UNKNOWN centre as free; the prior does not flip it.

BITWISE WHERE NOTHING IS FLIPPED. Given [] from `invertible_centres`, the prior draw, its density
and the oracle make no call on the generator and move no row: a molecule with no FREE
three-coordinate centre in the table (CH4, ethanol, 2-butanol, locked or not) draws bitwise as
before this module, generator state included. A molecule whose only such centres are planar
(H2CO, acetone, benzene) draws differently bit for bit -- its coins consume the generator and
its flips move rows -- but not in distribution beyond its reference's own departure from the
plane: the two components of such a centre's mixture sit |wrap(2 v)| apart, 8e-7 rad at the C of
H2CO, 2e-6 at acetone's and at most 2.4e-5 at benzene's ring carbons (level full, MMFF,
measured 2026-09-28).

NOT COVERED: a centre whose chart offers no offset a flip may negate has no entry, so the prior
draws its reference's side only and neither the eval nor the log Z check's label reads it.
  * A centre without an exocyclic neighbour, such as DABCO's caged N, which has no second side.
  * A ring atom at the ROOT whose flip would turn its ring system: an improper-placed child that
    is a ring member, or a seed's ring subtree placed on frames that hold the flipped child,
    turns with the flip. Measured 2026-09-28 on the reference and 8 perturbed states (level
    full, MMFF, lock 300): the root N flips of CN1CCOCC1, CCN1CCCC1 and DABCO change a bond
    length by 3.0 to 3.7 Angstrom and that of CN1CC1 scores 1.07e3 kcal/mol at the reference;
    that of C1COCCN1, whose H is a seed, turns the whole ring rigidly about the root-seed bond,
    an exact inversion (bond lengths to 7e-16, 0.144 kcal/mol) the structural rule does not
    admit. Benzene's root C, planar, loses nothing; the root C of biphenyl, whose
    improper-placed child is the other ring's ipso C, has an entry.
  * A ring atom the tree ENTERS from its exocyclic substituent, both ring neighbours its tree
    children, as the N of CCN1CC1: its only offset moves a ring member.
  * The far atom of a double bond the lock holds: with `stereo_coeff` > 0 the rows about the
    bond are held (`ConformerTorsions.held_phi_rows`) and in no group, so no flip moves that
    atom's substituents; it is planar, its own flip (C2 of C/C=C/C(=O)N(C)C at 300).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

#: WORKING ASSUMPTION (scope: the eval's parity metric, `free_centres`, and the log Z check's
#: label, `prior_held_parity_centres`; the prior's flip does not read it; revisit if a correct
#: sampler is flagged at a near-planar centre): an entry is `planar`, so neither required on
#: both sides nor labelled, when no flipped row's reference offset v lies more than this many of
#: the row's own prior widths from its flip -v, |wrap(2 v)| away. The reading depends on the
#: reference embedding, and on T through widths that scale as sqrt(T). Measured 2026-09-28 at
#: level full, MMFF, T = 1 kcal/mol unless stated: the amide N of CC(=O)NC at 5.4 widths (its
#: substituent 17 degrees from the plane), 1.7 at T = 10; the root alkene C of
#: C/C=C/C(=O)N(C)C at 1.08 (3 degrees), just past the bar, at `stereo_coeff` 300; the aromatic
#: ipso C of CNc1ccccc1 at 2.9 (7 degrees), its other ring carbons at 0.05 to 0.31, under the
#: bar; the carboxyl C of OC(=O)[C@@H]1CCCN1 at 2.2 (5 degrees); the C of H2CO and of acetone
#: and benzene's ring carbons under 0.01; amine N, a sulfoxide S and tetrahedral C at 15.9 to 23.
MIRROR_WIDTHS = 1.0

#: a centre's lock state (module docstring, LOCK AND SCOPE)
LOCKED, FREE, UNKNOWN = 'locked', 'free', 'unknown'

#: a centre's flip (module docstring, THE FLIP)
ROOT, SIBLING, FRAME = 'root', 'sibling', 'frame'


@dataclass(frozen=True)
class Centre:
    """One qualified centre of a chart, its flip and its standing (module docstring)."""
    slot: int                                       # placement slot of the centre atom
    name: str                                       # element symbol and slot, e.g. 'N0'
    kind: str                                       # ROOT, SIBLING or FRAME
    rows: Tuple[int, ...]                           # the phi rows the flip moves
    pivot: Optional[int]                            # SIBLING: the pivot row
    frame: Optional[Tuple[int, int, int, int]]      # FRAME: slots (a, b, c, p) of phi_ring
    n_bonded: int                                   # bonded neighbours in the chart's graph
    planar: bool                                    # its own flip within MIRROR_WIDTHS
    lock: str                                       # LOCKED, FREE or UNKNOWN
    triple: Tuple[int, int, int]                    # three neighbours of the centre ...
    side_ref: float                                 # ... and the sign of their triple product
    #                                                 about it at the reference: the side

    @property
    def free(self) -> bool:
        """The lock does not name it (an unreadable table names nothing). The eval requires
        both sides of a free centre that is not `planar` (`free_centres`)."""
        return self.lock != LOCKED

    @property
    def invertible(self) -> bool:
        """What the prior flips (module docstring, LOCK AND SCOPE): free by a readable lock
        table, and three-coordinate, planar or not."""
        return self.lock == FREE and self.n_bonded == 3


def reflect_phi(ph: np.ndarray, centre: Centre, where=None, ring=None) -> np.ndarray:
    """Flip `centre` on the phi block ``ph [n, n_ph]`` (radians), IN PLACE, on the draws `where`
    selects (bool ``[n]``; every draw when None). Returns `ph`.

    A FRAME centre needs `ring` ``[n]``, phi_ring of each draw (`frame_dihedral`), which the
    flip does not move. Values are not wrapped: a flipped row can leave (-pi, pi], and every
    consumer reads phi on the circle (`state_from_dof` wraps the state columns).
    """
    idx = (np.arange(ph.shape[0]) if where is None
           else np.flatnonzero(np.asarray(where, dtype=bool)))
    if idx.size == 0:
        return ph
    cols = np.ix_(idx, list(centre.rows))
    if centre.kind == ROOT:
        ph[cols] = -ph[cols]
    elif centre.kind == SIBLING:
        ph[cols] = 2.0 * ph[idx, centre.pivot][:, None] - ph[cols]
    else:
        if ring is None:
            raise ValueError(f'{centre.name}: a FRAME flip reads phi_ring, the dihedral '
                             f'{centre.frame} of each draw (frame_dihedral)')
        ph[cols] = 2.0 * np.asarray(ring, dtype=np.float64).reshape(-1)[idx][:, None] - ph[cols]
    return ph


def frame_dihedral(pos, centre: Centre) -> np.ndarray:
    """phi_ring ``[n]`` of a FRAME centre on positions ``pos [n, n_atoms, 3]``, in the convention
    the chart's rows use (mxtaltools.conformers.geometry.dihedral)."""
    from mxtaltools.conformers.geometry import dihedral
    p = torch.as_tensor(pos).reshape(len(pos), -1, 3)
    a, b, c, q = centre.frame
    return dihedral(p[:, a], p[:, b], p[:, c], p[:, q]).detach().double().cpu().numpy()


def side(pos, centre: Centre) -> np.ndarray:
    """``[n]`` the sign of the triple product of `centre.triple` about the centre, on positions
    ``pos [n, n_atoms, 3]``. A flip mirrors some of these neighbours through a plane holding the
    centre and the rest, so it reverses the sign."""
    p = np.asarray(torch.as_tensor(pos).detach().double().cpu()).reshape(len(pos), -1, 3)
    u, v, w = (p[:, k] - p[:, centre.slot] for k in centre.triple)
    return np.sign(np.einsum('ij,ij->i', u, np.cross(v, w)))


def reference_side(pos, centre: Centre) -> np.ndarray:
    """bool ``[n]``: the draw sits on the reference's side of `centre` (`side`)."""
    return side(pos, centre) == centre.side_ref


def centre_table(member) -> List[Centre]:
    """Every qualified centre of `member`'s chart, locked and planar ones included, by slot.

    Which centres have an entry is a function of the chart's structure, its level and the
    lock's table; the `planar` mark reads the reference values and the prior's widths (module
    docstring, NON-PLANAR), and `side_ref` the reference positions. Built once per
    (temperature, stereo coefficient, lock table) and cached on the instance.
    """
    cache = vars(member).setdefault('_centre_table_cache', {})
    key = _cache_key(member)
    if key not in cache:
        cache[key] = _build_table(member)
    return list(cache[key])


def free_centres(member) -> List[Centre]:
    """What the eval requires on both sides: the centres the lock leaves free (or cannot be
    read to name) that are not planar."""
    return [c for c in centre_table(member) if c.free and not c.planar]


def invertible_centres(member) -> List[Centre]:
    """What the prior flips (module docstring, LOCK AND SCOPE): the free, three-coordinate
    centres, planar or not."""
    return [c for c in centre_table(member) if c.invertible]


# ------------------------------------------------------------------ construction

def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _lock_names(member):
    """``(coefficient, set of slots the lock names or None when unreadable)``.

    The lock names an atom when its table holds a TETRAHEDRAL element keyed on it
    (energies/stereo_lock.py, `StereoTable.kind` and `.key`). A DOUBLE_BOND element is keyed
    on one of its double-bond atoms, a three-coordinate atom, and locks the bond's E/Z, not
    that atom's side, so it names nothing here. A table without both fields, or with fields
    of different lengths, is unreadable.
    """
    from energies.stereo_lock import TETRAHEDRAL
    coeff = float(getattr(member, 'stereo_coeff', 0.0) or 0.0)
    if coeff <= 0.0:
        return coeff, set()
    stereo = getattr(member, 'stereo', None)
    keys, kinds = getattr(stereo, 'key', None), getattr(stereo, 'kind', None)
    if keys is None or kinds is None:
        return coeff, None
    keys, kinds = np.asarray(keys).reshape(-1), np.asarray(kinds).reshape(-1)
    if keys.shape != kinds.shape:
        return coeff, None
    return coeff, {int(k) for k, kd in zip(keys, kinds) if int(kd) == TETRAHEDRAL}


def _cache_key(member):
    coeff, named = _lock_names(member)
    return float(member.temperature), coeff, None if named is None else tuple(sorted(named))


def _ring_bonds(member) -> set:
    """The bonds of the chart's graph that lie on a cycle (not bridges), as slot frozensets."""
    import networkx as nx
    b = np.asarray(member.bond_index_slot, dtype=np.int64).reshape(2, -1).T
    edges = {frozenset((int(u), int(v))) for u, v in b if u != v}
    g = nx.Graph([tuple(e) for e in edges])
    return edges - {frozenset(e) for e in nx.bridges(g)}


def _neighbours(member):
    """``(graph, tree)``: each slot's bonded neighbours over the chart's whole bond graph
    (`bond_index_slot`, ring-closure bonds included) and over its spanning tree
    (`spec.bond_index`). A graph neighbour that is no tree neighbour is a RING-CLOSURE one."""
    graph: Dict[int, set] = {}
    tree: Dict[int, set] = {}
    for adj, bonds in ((graph, np.asarray(member.bond_index_slot).reshape(2, -1).T),
                       (tree, np.asarray(member.spec.bond_index).reshape(-1, 2))):
        for u, v in np.asarray(bonds, dtype=np.int64):
            if u != v:
                adj.setdefault(int(u), set()).add(int(v))
                adj.setdefault(int(v), set()).add(int(u))
    return graph, tree


def _moved(ti: np.ndarray, rows) -> set:
    """The slots a flip of `rows` moves: each row's placed atom, then, in placement order, every
    atom placed on a frame (a, b, c) that holds a moved one -- a subtree below a flipped child,
    and a root seed's children, which the canonical tree places on a frame holding the root's
    improper-placed child."""
    moved = {int(ti[j, 3]) for j in rows}
    for j in np.argsort(ti[:, 3], kind='stable'):
        a, b, c, n = (int(v) for v in ti[j])
        if n not in moved and moved & {a, b, c}:
            moved.add(n)
    return moved


def _ring_systems(member) -> Dict[int, frozenset]:
    """``{slot: its ring system}`` for every atom on a ring: the atoms joined to it through ring
    bonds (`_ring_bonds`), itself included."""
    adj: Dict[int, set] = {}
    for bond in _ring_bonds(member):
        u, v = tuple(bond)
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    out: Dict[int, frozenset] = {}
    for s in adj:
        if s in out:
            continue
        comp, todo = {s}, [s]
        while todo:
            for w in adj[todo.pop()] - comp:
                comp.add(w)
                todo.append(w)
        out.update(dict.fromkeys(comp, frozenset(comp)))
    return out


def _build_table(member) -> List[Centre]:
    """The qualified centres (module docstring, QUALIFIED), from the chart's structure and its
    reference values; nothing is built or scored."""
    if getattr(member, 'collective', False):
        return []                          # a column drives several rows: none moves alone
    ti = np.asarray(member.spec.torsion_index, dtype=np.int64).reshape(-1, 4)
    n_ph = int(ti.shape[0])
    if n_ph == 0:
        return []
    from mxtaltools.conformers.geometry import dihedral
    from rdkit import Chem
    sym = Chem.GetPeriodicTable()
    z = np.asarray(member.spec.z)
    n0 = int(member.n_r + member.n_th)
    ph0 = member.ph0.detach().cpu().numpy().astype(np.float64)
    ref = member.ref_pos.detach().cpu().double().reshape(-1, 3)
    t = float(member.temperature)
    movable = np.asarray(member.free_mask, dtype=bool)[n0:n0 + n_ph].copy()
    tv = np.asarray(getattr(member, 'transverse_angles', np.zeros(0)), dtype=bool)
    if tv.any():                           # a transverse row carries v, not an angle
        movable[np.asarray(member.transverse_partner)[tv]] = False
    in_ring = np.asarray(member.atom_in_ring, dtype=bool)
    graph, _ = _neighbours(member)
    ring, systems = _ring_bonds(member), _ring_systems(member)
    imp = sorted({int(j) for j in member.improper_phi_rows()})
    s_imp = float(member.improper_phi_sigma(t))
    groups = [[int(j) for j in rows if int(j) not in imp] for rows in member.torsion_groups()]
    widths = member.sibling_jitter_sigma(groups, t)
    kid = lambda j: int(ti[j, 3])

    # every candidate flip: (c, kind, rows, pivot, frame, [(offset, width)], neighbours it keeps)
    flips = []
    at_c: Dict[int, List[int]] = {}
    for j in imp:
        at_c.setdefault(int(ti[j, 2]), []).append(j)
    if 0 in at_c:                          # the root, slot 0: its seeds span the mirror plane
        rows = tuple(at_c[0])
        flips.append((0, ROOT, rows, None, None, [(ph0[j], s_imp) for j in rows], {1, 2}))
    for rows, width in zip(groups, widths):
        if not rows:
            continue
        a, b, c = (int(v) for v in ti[rows[0], :3])
        on_ring = [j for j in rows if frozenset((c, kid(j))) in ring]
        ps = [q for q in sorted(graph.get(c, ())) if q != b and frozenset((c, q)) in ring
              and q not in {kid(j) for j in rows}]
        if on_ring:                        # a mixed group: its ring member leads and pivots
            pivot = on_ring[0]
            moved = tuple(j for j in rows if j != pivot)
            flips.append((c, SIBLING, moved, pivot, None,
                          [(ph0[j] - ph0[pivot], width) for j in moved], {b, kid(pivot)}))
        elif ps and in_ring[b] and in_ring[c]:
            # the ring-frame rule (ConformerTorsions.ring_frame_groups): every row hangs off
            # phi_ring, the dihedral (a, b, c, p) of c's ring-closure neighbour p
            p = ps[0]
            ring0 = float(dihedral(*(ref[k][None] for k in (a, b, c, p)))[0])
            flips.append((c, FRAME, tuple(rows), None, (a, b, c, p),
                          [(ph0[j] - ring0, width) for j in rows], {b, p}))
        elif len(rows) >= 2:               # acyclic siblings: the leader pivots
            flips.append((c, SIBLING, tuple(rows[1:]), rows[0], None,
                          [(ph0[j] - ph0[rows[0]], width) for j in rows[1:]],
                          {b, kid(rows[0])}))

    coeff, named = _lock_names(member)
    per_c: Dict[int, int] = {}
    for f in flips:
        per_c[f[0]] = per_c.get(f[0], 0) + 1
    out = []
    for c, kind, rows, pivot, frame, offsets, keep in sorted(flips, key=lambda f: f[0]):
        if per_c[c] > 1 or (c in at_c and c != 0):
            continue                       # not the canonical tree's one flip per centre
        if not rows or not all(movable[j] for j in rows):
            continue
        if graph.get(c, set()) - {kid(j) for j in rows} - keep:
            continue                       # a neighbour stays put off the mirror plane
        if _moved(ti, rows) & systems.get(c, frozenset()):
            continue                       # the flip would turn c's ring system
        nb = sorted(graph[c])
        if len(nb) < 3:
            continue                       # no pyramid to have a side
        # planar: every flipped row's offset within MIRROR_WIDTHS of its own flip. Recorded,
        # never a reason to drop the centre: the eval and the log Z label read it, the prior's
        # flip does not (module docstring, NON-PLANAR)
        planar = not any(abs(_wrap(2.0 * v)) > MIRROR_WIDTHS * w for v, w in offsets)
        triple = (nb[0], nb[1], nb[2])
        u, v, w = ((ref[k] - ref[c]).numpy() for k in triple)
        lock = FREE if coeff <= 0.0 else (UNKNOWN if named is None
                                          else (LOCKED if c in named else FREE))
        out.append(Centre(slot=int(c), name=f'{sym.GetElementSymbol(int(z[c]))}{c}', kind=kind,
                          rows=tuple(int(j) for j in rows), pivot=pivot, frame=frame,
                          n_bonded=len(nb), planar=bool(planar), lock=lock, triple=triple,
                          side_ref=float(np.sign(np.dot(u, np.cross(v, w))))))
    return out
