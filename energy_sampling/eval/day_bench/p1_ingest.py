"""Turn a periodic cell holding whole molecules into an MXtalTools rigid-molecule crystal.

Every source of structures for the generator benchmark arrives as an ``ase.Atoms`` cell: a generator's P1
sample (Clari, Packora, CrystAF, MolCrystalFlow), a Day landscape CIF expanded by ASE's reader, or a CSD
structure. ``ingest`` then does the same steps for all of them:

1. ``split_molecules``: bond atoms by covalent radii under periodic boundaries, take connected components,
   and unwrap each component across the cell faces.
2. ``fit_reference``: match each component's graph to the reference molecule (element-labelled graph
   isomorphism, composed with every automorphism of the reference), and fit the rigid reference onto it by
   Kabsch, as given and mirrored. The better of the two is kept, so a cell that holds both enantiomers of a
   non-planar conformer keeps them.
3. ``symmetry_candidates``: run spglib on the cell rebuilt from the fitted copies, at increasing
   tolerances, and yield each description with exactly one molecule per asymmetric unit; where the full
   group puts a molecule on a special position, the Z' = 1 subgroups instead. Each group is re-detected at
   the Hall number whose operators are ``SYM_OPS[sg]``, so the cell comes out in MXtalTools' setting.
4. ``_build``: place one fitted molecule in that cell, build the unit cell from ``SYM_OPS[sg]``, read the
   canonical asymmetric-unit pose with ``reparameterize_unit_cell``, and rebuild the crystal from it.
5. The rebuilt unit cell is compared atom by atom with the fitted input copies, mapped into the same cell:
   ``residual_max`` is the largest distance from an input atom to the nearest rebuilt atom of its element.
   The first candidate within ``residual_limit(symprec)`` is kept; none, and the cell fails.

A CSD structure fitted with its own molecule should come back with a residual near zero; a generator
sample comes back with the distance its symmetrisation moved it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
import spglib
import torch
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList

from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.dataset_utils.data_class_methods.crystal_ops import spglib_hall_number
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

BOND_SCALE = 1.15  # bonded when distance < s_i r_cov_i + s_j r_cov_j, s = BOND_SCALE for heavy atoms
H_BOND_SCALE = 1.35  # and H_BOND_SCALE for hydrogen: C-H up to 1.29 A (CSD CAMWUI has 1.238), O-H up to 1.18 A
SYMPRECS = (1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.5)  # Angstrom, tried in order
MAX_AUTOMORPHISMS = 5000


@dataclass
class Molecule:
    """One molecule: atomic numbers, Cartesian positions (unwrapped), and its bond graph."""
    z: np.ndarray
    pos: np.ndarray
    graph: nx.Graph


@dataclass
class Reference:
    """The rigid conformer every structure is fitted with, and the automorphisms of its graph."""
    mol: Molecule
    automorphisms: list
    smiles: Optional[str] = None


@dataclass
class IngestResult:
    ok: bool
    reason: str = ''
    n_molecules: int = 0
    fit_rmsd_max: float = float('nan')     # Angstrom, worst Kabsch fit over the cell's molecules
    n_mirrored: int = 0                    # molecules fitted by the mirrored reference
    sg: Optional[int] = None
    z_prime: Optional[float] = None
    symprec: Optional[float] = None
    residual_max: float = float('nan')     # Angstrom, input atom to nearest rebuilt atom of its element
    n_subgroups: int = 0                   # Z' = 1 descriptions found (more than 1 only for special positions)
    canonical: bool = True                 # False where MXtalTools' ASYM_UNITS box does not pick one copy
    cell_strain: float = float('nan')      # largest relative cell-length change from idealising the metric
    angle_change_deg: float = float('nan')  # largest cell-angle change from idealising the metric, degrees
    crystal: Optional[MolCrystalData] = None
    notes: dict = field(default_factory=dict)


def _bond_radii(z: np.ndarray) -> np.ndarray:
    return covalent_radii[z] * np.where(z == 1, H_BOND_SCALE, BOND_SCALE)


def _bond_graph(z: np.ndarray, pos: np.ndarray) -> nx.Graph:
    g = nx.Graph()
    for i, zi in enumerate(z):
        g.add_node(i, z=int(zi))
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    r = _bond_radii(z)
    bonded = d < r[:, None] + r[None]
    i, j = np.nonzero(np.triu(bonded, 1))
    g.add_edges_from(zip(i.tolist(), j.tolist()))
    return g


def split_molecules(atoms: Atoms) -> list[Molecule]:
    """Connected components of the periodic bond graph, each unwrapped into one piece.

    A cell that carries per-atom molecule labels (``bb_indices``, as MolCrystalFlow writes them, each labelled
    molecule already whole) is split by the labels instead: a generator's sample can overlap so closely that
    bonds read across molecules, and the generator knows which atoms it placed together.

    Raises ValueError when a component bonds to its own periodic image (a polymer, or atoms too close).
    """
    z = atoms.get_atomic_numbers()
    if 'bb_indices' in atoms.arrays:
        labels = atoms.arrays['bb_indices']
        pos = atoms.get_positions()
        return [Molecule(z[labels == g], pos[labels == g], _bond_graph(z[labels == g], pos[labels == g]))
                for g in np.unique(labels)]
    cutoffs = _bond_radii(z).tolist()
    nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)
    cell = np.asarray(atoms.cell)
    pos = atoms.get_positions()
    shift = np.full((len(z), 3), np.iinfo(np.int64).min, dtype=np.int64)
    comp = np.full(len(z), -1)
    n_comp = 0
    for root in range(len(z)):
        if comp[root] >= 0:
            continue
        comp[root] = n_comp
        shift[root] = 0
        stack = [root]
        while stack:
            i = stack.pop()
            idx, offsets = nl.get_neighbors(i)
            for j, o in zip(idx, offsets):
                s = shift[i] + o
                if comp[j] < 0:
                    comp[j] = n_comp
                    shift[j] = s
                    stack.append(j)
                elif not np.array_equal(shift[j], s):
                    raise ValueError('a molecule bonds to its own periodic image')
        n_comp += 1
    unwrapped = pos + shift @ cell
    out = []
    for c in range(n_comp):
        members = np.nonzero(comp == c)[0]
        zc, pc = z[members], unwrapped[members]
        out.append(Molecule(zc, pc, _bond_graph(zc, pc)))
    return out


def _node_match(a, b):
    return a['z'] == b['z']


def make_reference(mol: Molecule, smiles: Optional[str] = None) -> Reference:
    gm = nx.algorithms.isomorphism.GraphMatcher(mol.graph, mol.graph, node_match=_node_match)
    autos = []
    for m in gm.isomorphisms_iter():
        autos.append(m)
        if len(autos) >= MAX_AUTOMORPHISMS:
            break
    return Reference(mol=mol, automorphisms=autos, smiles=smiles)


def _kabsch(p: np.ndarray, q: np.ndarray):
    """Proper rotation R and translation t minimising |R q + t - p|; returns R, t, rmsd."""
    pc, qc = p.mean(0), q.mean(0)
    h = (q - qc).T @ (p - pc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = pc - r @ qc
    rmsd = float(np.sqrt(((q @ r.T + t - p) ** 2).sum(1).mean()))
    return r, t, rmsd


def fit_reference(mol: Molecule, ref: Reference):
    """Fitted reference coordinates, in reference atom order, placed on ``mol``.

    Returns (coords, rmsd, mirrored) or raises ValueError when the graphs differ.
    """
    if len(mol.z) != len(ref.mol.z) or sorted(mol.z) != sorted(ref.mol.z):
        raise ValueError('atom count or composition differs from the reference')
    gm = nx.algorithms.isomorphism.GraphMatcher(ref.mol.graph, mol.graph, node_match=_node_match)
    base = next(gm.isomorphisms_iter(), None)  # reference atom -> sample atom
    if base is None:
        raise ValueError('bond graph differs from the reference')
    q0 = ref.mol.pos
    q_m = q0 * np.array([-1.0, 1.0, 1.0])
    best = (np.inf, None, False)
    for aut in ref.automorphisms:
        order = np.array([base[aut[k]] for k in range(len(q0))])  # sample atom for each reference atom
        p = mol.pos[order]
        for q, mirrored in ((q0, False), (q_m, True)):
            r, t, rmsd = _kabsch(p, q)
            if rmsd < best[0] - 1e-9:
                best = (rmsd, q @ r.T + t, mirrored)
    return best[1], best[0], best[2]


def _cell_handedness_fix(lattice: np.ndarray, frac: np.ndarray):
    """Swap a and b when the basis is left-handed; the structure is unchanged."""
    if np.linalg.det(lattice) > 0:
        return lattice, frac
    return lattice[[1, 0, 2]], frac[:, [1, 0, 2]]


# Preference among Z' = 1 subgroups of a special-position structure: the space groups by their count among
# the Day set's 1,040 observed structures (Day ESI Table S3), most common first; groups not listed rank last.
SG_PREFERENCE = (14, 19, 15, 61, 2, 4, 33, 62, 7, 92, 60, 29, 11, 9, 1, 12, 114, 56, 148, 41, 176, 43, 58, 64,
                 76, 18, 63, 5, 36, 96, 20, 88, 145, 154, 169, 170, 86, 13)


def _op_key(W, w):
    return tuple(np.rint(W).astype(int).ravel()), tuple(np.round(np.mod(w, 1.0), 4) % 1.0)


def _compose(a, b):
    """(W_a, w_a) o (W_b, w_b), translations reduced modulo 1."""
    return a[0] @ b[0], np.mod(a[0] @ b[1] + a[1], 1.0)


def _closure(gens, limit):
    """Group generated by ``gens`` modulo lattice translations; None once it grows past ``limit``."""
    ident = (np.eye(3), np.zeros(3))
    elems = {_op_key(*ident): ident}
    frontier = [ident]
    while frontier:
        new = []
        for x in frontier:
            for g in gens:
                y = _compose(g, x)
                k = _op_key(*y)
                if k not in elems:
                    elems[k] = y
                    new.append(y)
                    if len(elems) > limit:
                        return None
        frontier = new
    return elems


def _free_subgroups(ops, fixes, n_mol):
    """Subgroups of order ``n_mol`` in which only the identity maps a molecule onto itself."""
    free = [o for o, f in zip(ops, fixes) if not f]
    free_keys = {_op_key(*o) for o in free}
    ident_key = _op_key(np.eye(3), np.zeros(3))
    found, seen = [], set()

    def grow(gens, elems):
        key = frozenset(elems)
        if key in seen:
            return
        seen.add(key)
        if len(elems) == n_mol:
            found.append(elems)
            return
        for g in free:
            if _op_key(*g) in elems:
                continue
            closed = _closure(gens + [g], n_mol)
            if closed is None or n_mol % len(closed):
                continue
            if any(k != ident_key and k not in free_keys for k in closed):
                continue  # the closure contains an element that fixes a molecule
            grow(gens + [g], closed)

    if n_mol == 1:
        return [{ident_key: (np.eye(3), np.zeros(3))}]
    grow([], {ident_key: (np.eye(3), np.zeros(3))})
    return found


def _molecule_permutation_fixes(ops, centroids, lattice, tol):
    """For each operation, whether it maps some molecule's centroid onto itself (within ``tol`` Angstrom)."""
    fixes = []
    for W, w in ops:
        img = centroids @ W.T + w
        d = img - centroids
        d -= np.round(d)
        fixes.append(bool((np.linalg.norm(d @ lattice, axis=1) < tol).any()))
    return fixes


def _identify(lattice, frac, numbers, s, n_mol):
    """spglib's group in MXtalTools' setting: (sg, P, p, z_prime) or (None, reason)."""
    ds = spglib.get_symmetry_dataset((lattice, frac, numbers), symprec=s)
    if ds is None:
        return None, 'no symmetry found'
    sg = int(ds.number)
    ds = spglib.get_symmetry_dataset((lattice, frac, numbers), symprec=s, hall_number=spglib_hall_number(sg))
    if ds is None or int(ds.number) != sg:
        return None, f'sg {sg} not re-detected in MXtalTools setting at symprec {s}'
    P = np.asarray(ds.transformation_matrix, dtype=float)
    z_prime = n_mol * abs(np.linalg.det(np.linalg.inv(P))) / len(SYM_OPS[sg])
    return (sg, P, np.asarray(ds.origin_shift, dtype=float), z_prime, np.asarray(ds.std_lattice, dtype=float)), ds


def symmetry_candidates(lattice: np.ndarray, frac: np.ndarray, numbers: np.ndarray, n_mol: int,
                        centroids: np.ndarray, symprecs=SYMPRECS, prefer_sg: Optional[int] = None,
                        failures: Optional[list] = None):
    """Descriptions with one molecule per asymmetric unit, in MXtalTools' setting, in the order to try them.

    Tolerances are taken in increasing order. When the full group at a tolerance has Z' < 1 (a molecule on a
    special position), the structure is described in each subgroup acting freely on the molecules,
    ``prefer_sg`` first, then by ``SG_PREFERENCE``. A subgroup is identified by adding, for each of its
    operations, an image of one marker point placed off molecule 0's centroid, and running spglib on the
    marked structure. The caller keeps the first candidate whose rebuild reproduces the cell.

    Yields (sg, symprec, P, p, z_prime, n_candidates) with x_std = P x + p; reasons for tolerances that give
    nothing are appended to ``failures``.
    """
    failures = [] if failures is None else failures
    marker_type = int(numbers.max()) + 50
    for s in symprecs:
        out, ds = _identify(lattice, frac, numbers, s, n_mol)
        if out is None:
            failures.append(ds)
            continue
        sg, P, p, zp, std_lattice = out
        if abs(zp - 1) < 1e-6:
            yield sg, s, P, p, zp, 1, std_lattice
            continue
        if zp > 1:
            failures.append(f"sg {sg} at symprec {s} gives Z' = {zp:g}")
            continue
        # special position: descend to a subgroup in which every molecule is in general position
        ops = [(np.asarray(W, dtype=float), np.asarray(w, dtype=float)) for W, w in zip(ds.rotations, ds.translations)]
        fixes = _molecule_permutation_fixes(ops, centroids, lattice, tol=max(4 * s, 0.05))
        subs = _free_subgroups(ops, fixes, n_mol)
        offset = np.array([0.31, 0.17, 0.23]) @ np.linalg.inv(lattice)  # a generic 0.43 A step, fractional
        x0 = centroids[0] + offset
        cands = []
        for elems in subs:
            markers = np.array([W @ x0 + w for W, w in elems.values()])
            f2 = np.concatenate([frac, np.mod(markers, 1.0)])
            n2 = np.concatenate([numbers, np.full(len(markers), marker_type)])
            o2, _ = _identify(lattice, f2, n2, s, n_mol)
            if o2 is not None and abs(o2[3] - 1) < 1e-6:
                cands.append(o2)
        if not cands:
            failures.append(f"sg {sg} at symprec {s} gives Z' = {zp:g} and no subgroup with Z' = 1 was identified")
            continue
        rank = {g: i for i, g in enumerate(SG_PREFERENCE)}
        cands.sort(key=lambda c: (c[0] != prefer_sg, rank.get(c[0], len(rank))))
        for sg2, P2, p2, zp2, std2 in cands:
            yield sg2, s, P2, p2, zp2, len(cands), std2


def _lengths_angles(lattice_rows: np.ndarray):
    a, b, c = lattice_rows
    la, lb, lc = (np.linalg.norm(v) for v in (a, b, c))
    alpha = np.arccos(np.dot(b, c) / (lb * lc))
    beta = np.arccos(np.dot(a, c) / (la * lc))
    gamma = np.arccos(np.dot(a, b) / (la * lb))
    return np.array([la, lb, lc]), np.array([alpha, beta, gamma])


def _crystal(ref: Reference, sg: int, lengths, angles, pos, centroid, orientation=None, handedness=None,
             identifier=None) -> MolCrystalData:
    # x: one zero feature per atom; the cluster builder copies x to every image (column 0 is read as a partial
    # charge only by the electrostatic computes, which the eLJ route does not call)
    mol = MolData(z=torch.tensor(ref.mol.z, dtype=torch.long), pos=torch.tensor(pos, dtype=torch.float32),
                  x=torch.zeros(len(ref.mol.z), 1), smiles=ref.smiles, do_mol_analysis=True)
    return MolCrystalData(
        molecule=mol, sg_ind=int(sg),
        cell_lengths=torch.tensor(lengths, dtype=torch.float32),
        cell_angles=torch.tensor(angles, dtype=torch.float32),
        aunit_centroid=torch.as_tensor(centroid, dtype=torch.float32).flatten(),
        aunit_orientation=(torch.zeros(3) if orientation is None
                           else torch.as_tensor(orientation, dtype=torch.float32).flatten()),
        aunit_handedness=(torch.ones(1) if handedness is None
                          else torch.as_tensor(handedness, dtype=torch.float32).flatten()),
        identifier=identifier, z_prime=torch.ones(1, dtype=torch.long), max_z_prime=1, do_box_analysis=True)


def _build(ref: Reference, sg, P, p, std_lattice, lattice, frac, numbers, identifier):
    """The crystal in the standard cell of (sg, P, p), and its residual against the fitted atoms (Angstrom).

    The cell is spglib's idealised standard cell (``std_lattice``), so its metric obeys the crystal system;
    ``strain`` reports how far that is from the input cell expressed in the same basis,
    (a_s b_s c_s) = (a b c) P^-1: the largest relative length change and the largest angle change in degrees.
    Fractional coordinates x_s = P x + p are kept and placed in the idealised cell.
    """
    n_at = len(ref.mol.z)
    heavy = ref.mol.z > 1
    lengths_raw, angles_raw = _lengths_angles((lattice.T @ np.linalg.inv(P)).T)
    lengths, angles = _lengths_angles(std_lattice)
    strain = (float(np.max(np.abs(lengths / lengths_raw - 1))), float(np.degrees(np.max(np.abs(angles - angles_raw)))))
    frac_s = frac @ P.T + p  # unwrapped fractional coordinates of every fitted atom in the standard cell
    mol0 = frac_s[:n_at]
    c0 = mol0[heavy].mean(0)
    mol0 = mol0 - np.floor(c0)
    c0 = c0 - np.floor(c0)

    seed = _crystal(ref, sg, lengths, angles, ref.mol.pos, c0, identifier=identifier)
    T_fc = seed.T_fc[0].double().numpy()
    seed.pos = torch.tensor(mol0 @ T_fc.T, dtype=torch.float32)
    batch = collate_data_list([seed], max_z_prime=1)
    batch.build_unit_cell()
    cen, ori, hand, well_defined, pos = batch.reparameterize_unit_cell()
    crystal = _crystal(ref, sg, lengths, angles, pos.numpy(), cen, ori, hand, identifier=identifier)
    check = collate_data_list([crystal], max_z_prime=1)
    check.pose_aunit()
    check.build_unit_cell()
    uc_frac = check.unit_cell_pos.double().numpy() @ np.linalg.inv(T_fc).T
    uc_z = np.tile(ref.mol.z, int(check.sym_mult[0]))

    worst = 0.0
    for el in np.unique(numbers):
        a = frac_s[numbers == el]
        b = uc_frac[uc_z == el]
        df = a[:, None] - b[None]
        df -= np.round(df)
        d = np.linalg.norm(df @ T_fc.T, axis=-1).min(1)
        worst = max(worst, float(d.max()))
    # well_defined is False when no copy, or more than one, lies in ASYM_UNITS[sg]; that box is the whole
    # cell for sg 145, 148, 154, 169, 170. The copy MXtalTools then picks still describes the same crystal,
    # which the residual checks.
    return crystal, worst, bool(well_defined[0]), strain


def residual_limit(symprec: float) -> float:
    """Largest residual (Angstrom) at which a description found at ``symprec`` is accepted."""
    return 2.0 * symprec + 0.05


def ingest(atoms: Atoms, ref: Reference, symprecs=SYMPRECS, identifier: Optional[str] = None,
           prefer_sg: Optional[int] = None) -> IngestResult:
    """Run every step on one cell; failures return ``ok=False`` with the reason, never raise.

    Candidate descriptions are tried in the order ``symmetry_candidates`` yields them, and the first whose
    rebuild lies within ``residual_limit(symprec)`` of every fitted atom is kept. ``prefer_sg`` picks the
    description when a special-position structure has several Z' = 1 subgroups, for example the space group
    a Day landscape structure was searched in.
    """
    res = IngestResult(ok=False)
    try:
        mols = split_molecules(atoms)
    except ValueError as e:
        res.reason = f'split: {e}'
        return res
    res.n_molecules = len(mols)
    fitted, rmsds = [], []
    for m in mols:
        try:
            coords, rmsd, mirrored = fit_reference(m, ref)
        except ValueError as e:
            res.reason = f'fit: {e}'
            return res
        fitted.append(coords)
        rmsds.append(rmsd)
        res.n_mirrored += int(mirrored)
    res.fit_rmsd_max = float(max(rmsds))

    lattice = np.asarray(atoms.cell, dtype=float)
    cart = np.concatenate(fitted)
    numbers = np.tile(ref.mol.z, len(fitted))
    frac = cart @ np.linalg.inv(lattice)
    lattice, frac = _cell_handedness_fix(lattice, frac)
    n_at = len(ref.mol.z)
    heavy = ref.mol.z > 1
    centroids = frac.reshape(len(fitted), n_at, 3)[:, heavy].mean(1)

    failures, rejected = [], []
    for sg, s, P, p, zp, n_sub, std_lattice in symmetry_candidates(lattice, frac - np.floor(frac), numbers,
                                                                   len(fitted), centroids, symprecs, prefer_sg,
                                                                   failures):
        try:
            crystal, worst, canonical, strain = _build(ref, sg, P, p, std_lattice, lattice, frac, numbers,
                                                       identifier)
        except Exception as e:  # MXtalTools raises assorted types on bad cells; recorded, not raised
            rejected.append(f'sg {sg} at {s}: {type(e).__name__}')
            continue
        if worst > residual_limit(s):
            rejected.append(f'sg {sg} at {s}: residual {worst:.2f} A')
            continue
        res.sg, res.symprec, res.z_prime, res.n_subgroups = sg, s, zp, n_sub
        res.residual_max, res.canonical, res.crystal, res.ok = worst, canonical, crystal, True
        res.cell_strain, res.angle_change_deg = strain
        res.notes['rejected'] = rejected
        return res
    res.reason = 'symmetry: ' + ('; '.join(rejected[-2:]) if rejected else (failures[-1] if failures
                                                                            else 'no symmetry found'))
    res.notes['rejected'] = rejected
    return res
