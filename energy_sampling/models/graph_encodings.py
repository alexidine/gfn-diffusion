"""Structural encodings and graph observables for the conformer encoder.

TWO JOBS, ONE MODULE, ON PURPOSE. The functions here produce both the *inputs* the encoder
is given (``rwse``, ``lap_pe``) and the *labels* the self-supervision battery scores it
against (everything under OBSERVABLES). Keeping them together is what makes the battery
honest: a probe and its feature must never be computed by two different code paths that can
drift apart, and several observables are literally global versions of what an encoding holds
locally -- ``spectral_moments`` is the closed-walk count that ``rwse`` samples per node.

WHY ENCODINGS ARE NEEDED AT ALL. Message passing is bounded by 1-WL, so an MPNN cannot count
cycles: a carbon in a benzene ring and a carbon in a chain with the same local neighbourhood
are indistinguishable to it at any depth. That is not a range problem more layers can fix.
A structural encoding is computed exactly, offline, from the adjacency, and injected as node
features -- so it sidesteps both the 1-WL bound and the depth-vs-oversquashing trade
entirely.

RWSE IS THE DEFAULT AND LapPE IS NOT. See ``lap_pe``'s docstring: its output is a function of
what the eigensolver returned, not of the graph, and ``{f_j}`` is cached per molecule. A
non-deterministic encoding makes the cached condition a function of the solver -- the same
defect class as the chart problem in conformer_conditional_stack.md section 6.

Everything is pure-graph: an edge list and an atom count. No geometry, no force field, no
conformer. Chirality does not enter here at all -- every quantity below is a function of the
adjacency, which is identical for enantiomers, so parity has to arrive as a separate atom
feature. See docs/design/encoder_ssl_battery.md section 1.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    'to_dense_adjacency', 'rwse', 'lap_pe',
    'shortest_paths', 'eccentricity', 'diameter', 'wiener_index',
    'cycle_rank', 'degree_histogram', 'spectral_moments',
    'smallest_ring_size', 'ring_membership',
    'orbit_sizes', 'n_automorphisms',
    'graph_from_smiles', 'bond_features_from_smiles',
    'canonical_parity', 'canonical_root', 'cip_codes', 'wl_colours', 'pi_degree',
    'mol_for_labels',
]

#: shortest_paths marks unreachable pairs with this rather than inf, so the matrix stays
#: integral and every downstream consumer has to decide explicitly what to do about
#: disconnection instead of silently propagating a float sentinel.
UNREACHABLE = -1


# --------------------------------------------------------------------------- helpers

def to_dense_adjacency(edge_index, n: int) -> np.ndarray:
    """``[2, E] -> [n, n]`` float64, symmetric, zero diagonal.

    Accepts either direction convention: the edge list may carry each bond once or twice,
    and the result is the same, because the adjacency is symmetrised here. Relying on the
    caller to have doubled its edges is exactly the kind of convention that holds until one
    call site forgets.
    """
    e = np.asarray(edge_index)
    if e.ndim != 2 or e.shape[0] != 2:
        raise ValueError(f'edge_index must be [2, E], got {e.shape}')
    a = np.zeros((n, n), dtype=np.float64)
    if e.shape[1]:
        if e.max() >= n or e.min() < 0:
            raise ValueError(f'edge_index references atoms outside [0, {n})')
        a[e[0], e[1]] = 1.0
        a[e[1], e[0]] = 1.0
    np.fill_diagonal(a, 0.0)
    return a


def _degrees(a: np.ndarray) -> np.ndarray:
    return a.sum(axis=1)


# -------------------------------------------------------------------------- ENCODINGS

def rwse(edge_index, n: int, k: int = 16) -> np.ndarray:
    """Random-walk structural encoding. ``[n, k]``: return probability at each step count.

    ``RWSE(i)[s] = (A D^-1)^(s+1)_{ii}`` for s = 0 .. k-1 -- the probability that a walk
    leaving atom i is back at i after s+1 steps.

    WHY THIS ONE. It encodes ring membership and ring SIZE almost directly (a node in a
    6-ring has a characteristic spike at step 6), which is the structural fact that sets a
    torsion's rotamer profile. And it is a matrix power: fully determined by the graph, with
    no sign or basis ambiguity to repair. Since ``{f_j}`` is cached per molecule, that
    determinism is not a nicety -- it is what keeps the cached condition a function of the
    labelled graph.

    Isolated atoms get an all-zero row rather than a divide-by-zero: a walk cannot leave, so
    "probability of return" is not defined, and 0 is the honest filler. Molecules with
    explicit hydrogens never hit this; disconnected fragments can.
    """
    if k < 1:
        raise ValueError(f'k must be >= 1, got {k}')
    a = to_dense_adjacency(edge_index, n)
    deg = _degrees(a)
    live = deg > 0
    rw = np.zeros_like(a)
    rw[:, live] = a[:, live] / deg[live]          # column-stochastic: A D^-1
    out = np.zeros((n, k), dtype=np.float64)
    m = rw.copy()
    for s in range(k):
        out[:, s] = np.diag(m)
        if s + 1 < k:
            m = m @ rw
    return out


def lap_pe(edge_index, n: int, k: int = 8) -> np.ndarray:
    """Laplacian positional encoding. ``[n, k]`` from the lowest non-trivial eigenvectors.

    Node i's encoding is its row of the eigenvector matrix -- literally its coordinates in an
    eigenbasis of the global adjacency, the graph analogue of a Fourier position.

    ⚠ THE OUTPUT IS NOT A FUNCTION OF THE GRAPH, AND THAT IS NOT FIXABLE HERE. Each
    eigenvector is defined only up to sign, and inside an eigenspace of multiplicity m only
    up to an arbitrary rotation in O(m). Molecular symmetry produces such degeneracies
    constantly -- benzene's Laplacian is degenerate by symmetry. So two runs on the same
    molecule can return different encodings.

    This function is provided so the battery can MEASURE that, not because it is safe to
    use raw. The determinism gate in docs/design/encoder_ssl_battery.md is expected to fail
    on this and to pass on ``rwse``; a candidate wanting spectral coordinates needs SignNet
    over the eigenvectors rather than the eigenvectors themselves.
    """
    if k < 1:
        raise ValueError(f'k must be >= 1, got {k}')
    a = to_dense_adjacency(edge_index, n)
    deg = _degrees(a)
    dinv = np.zeros(n)
    live = deg > 0
    dinv[live] = deg[live] ** -0.5
    lap = np.eye(n) - (dinv[:, None] * a * dinv[None, :])
    vals, vecs = np.linalg.eigh(lap)
    # drop the trivial constant eigenvector(s); pad when the graph is smaller than k+1
    vecs = vecs[:, 1:k + 1]
    if vecs.shape[1] < k:
        vecs = np.pad(vecs, ((0, 0), (0, k - vecs.shape[1])))
    return np.ascontiguousarray(vecs, dtype=np.float64)


# ------------------------------------------------------------------------ OBSERVABLES

def shortest_paths(edge_index, n: int) -> np.ndarray:
    """``[n, n]`` int bond-count distances by BFS. Unreachable pairs are ``UNREACHABLE``.

    Plain BFS rather than repeated matrix powers: it is exact, O(n·E), and it makes the
    unreachable case explicit instead of leaving it as a never-updated sentinel.
    """
    a = to_dense_adjacency(edge_index, n)
    nbrs = [np.flatnonzero(a[i]).tolist() for i in range(n)]
    out = np.full((n, n), UNREACHABLE, dtype=np.int64)
    for src in range(n):
        out[src, src] = 0
        frontier, d = [src], 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in nbrs[u]:
                    if out[src, v] == UNREACHABLE:
                        out[src, v] = d
                        nxt.append(v)
            frontier = nxt
    return out


def eccentricity(spd: np.ndarray) -> np.ndarray:
    """``[n]`` -- the greatest distance from each atom to any atom it can reach.

    THE SHARPEST SINGLE PROBE in the battery: "where am I in the graph" reduced to one
    integer, and it depends on a SPECIFIC far atom rather than on any sum over atoms. That
    is exactly what an aggregate-and-broadcast encoder cannot produce, which is why this is
    the probe that separates broadcast from routing.
    """
    m = np.where(spd == UNREACHABLE, -1, spd)
    return m.max(axis=1).astype(np.int64)


def diameter(spd: np.ndarray) -> int:
    """Greatest eccentricity. Computed over the reachable set only."""
    return int(eccentricity(spd).max())


def wiener_index(spd: np.ndarray) -> int:
    """Sum of shortest-path distances over unordered reachable pairs."""
    m = np.where(spd == UNREACHABLE, 0, spd)
    return int(m.sum() // 2)


def cycle_rank(edge_index, n: int) -> int:
    """``|E| - |V| + C`` -- the number of independent cycles. Exact, and the cheapest ring
    signal available. Counts each bond once regardless of the caller's edge convention."""
    a = to_dense_adjacency(edge_index, n)
    n_edges = int(a.sum() // 2)
    spd = shortest_paths(edge_index, n)
    seen, comps = np.zeros(n, dtype=bool), 0
    for i in range(n):
        if not seen[i]:
            comps += 1
            seen |= (spd[i] != UNREACHABLE)
    return n_edges - n + comps


def degree_histogram(edge_index, n: int, max_degree: int = 5) -> np.ndarray:
    """``[max_degree + 1]`` counts. Degrees above ``max_degree`` land in the last bin."""
    a = to_dense_adjacency(edge_index, n)
    d = np.minimum(_degrees(a).astype(np.int64), max_degree)
    return np.bincount(d, minlength=max_degree + 1)[:max_degree + 1].astype(np.int64)


def spectral_moments(edge_index, n: int, ks: Sequence[int] = (2, 3, 4, 5)) -> np.ndarray:
    """``tr(L^k)`` for each k, on the normalized Laplacian.

    These are closed-walk counts over the whole graph -- the GLOBAL form of exactly what
    ``rwse`` holds per node. An encoder carrying RWSE has the local pieces and must have
    aggregated them correctly to answer this, which is the point of asking.
    """
    a = to_dense_adjacency(edge_index, n)
    deg = _degrees(a)
    dinv = np.zeros(n)
    live = deg > 0
    dinv[live] = deg[live] ** -0.5
    lap = np.eye(n) - (dinv[:, None] * a * dinv[None, :])
    vals = np.linalg.eigvalsh(lap)
    return np.array([float((vals ** k).sum()) for k in ks], dtype=np.float64)


def smallest_ring_size(edge_index, n: int) -> np.ndarray:
    """``[n]`` -- size of the smallest cycle through each atom; 0 if the atom is acyclic.

    Computed by the standard edge-deletion argument: the smallest cycle through bond (u, v)
    is ``spd(u, v) + 1`` with that bond removed. An atom's answer is the minimum over its
    incident bonds.

    This is graph girth-per-node, NOT RDKit's SSSR. They agree on the smallest ring but not
    on ring perception in fused systems, and the difference is deliberate: the encoder is
    scored on what the adjacency determines, so the label must come from the adjacency too.
    """
    a = to_dense_adjacency(edge_index, n)
    out = np.zeros(n, dtype=np.int64)
    best = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    us, vs = np.triu(a, 1).nonzero()
    for u, v in zip(us.tolist(), vs.tolist()):
        cut = a.copy()
        cut[u, v] = cut[v, u] = 0.0
        e = np.array(np.triu(cut, 1).nonzero())
        d = shortest_paths(e, n)[u, v]
        if d != UNREACHABLE:
            best[u] = min(best[u], d + 1)
            best[v] = min(best[v], d + 1)
    found = best != np.iinfo(np.int64).max
    out[found] = best[found]
    return out


def ring_membership(edge_index, n: int) -> np.ndarray:
    """``[n]`` bool -- is the atom on any cycle. The 1-WL breaker, in its simplest form."""
    return smallest_ring_size(edge_index, n) > 0


# -------------------------------------------------------------------------- SYMMETRY

def _nx_graph(edge_index, n: int, labels: Optional[Sequence] = None):
    import networkx as nx
    g = nx.Graph()
    for i in range(n):
        g.add_node(i, label=(None if labels is None else labels[i]))
    a = to_dense_adjacency(edge_index, n)
    us, vs = np.triu(a, 1).nonzero()
    g.add_edges_from(zip(us.tolist(), vs.tolist()))
    return g


def _automorphisms(edge_index, n: int, labels: Optional[Sequence], cap: int):
    """Enumerate label-preserving automorphisms, refusing rather than hanging past ``cap``."""
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher
    g = _nx_graph(edge_index, n, labels)
    match = (None if labels is None
             else lambda x, y: x.get('label') == y.get('label'))
    gm = GraphMatcher(g, g, node_match=match)
    found = []
    for m in gm.isomorphisms_iter():
        found.append(m)
        if len(found) > cap:
            raise RuntimeError(
                f'more than {cap} automorphisms on a {n}-atom graph; refusing to enumerate '
                f'further. Raise the cap deliberately if this molecule really is that '
                f'symmetric -- do not silently truncate, since a truncated set gives wrong '
                f'orbits rather than incomplete ones.')
    return found


def orbit_sizes(edge_index, n: int, labels: Optional[Sequence] = None,
                cap: int = 100_000) -> np.ndarray:
    """``[n]`` -- the size of each atom's automorphism orbit on the LABELLED graph.

    Ceiling-bearing, and that is the point. Leave-one-out node identification can only ever
    succeed up to automorphism, so its accuracy ceiling is set by these numbers and is BELOW
    1.0 on any symmetric molecule. Score against the ceiling or correct behaviour reads as
    failure -- see docs/design/encoder_ssl_battery.md section 4.

    ``labels`` should carry whatever the encoder is allowed to see -- atomic number, and
    parity where a stereocentre is labelled. Passing ``None`` computes orbits of the BARE
    adjacency, which is a strictly coarser partition and is the right control for asking
    what the labels bought.
    """
    autos = _automorphisms(edge_index, n, labels, cap)
    orbit = [{i} for i in range(n)]
    for m in autos:
        for i, j in m.items():
            orbit[i].add(j)
    # close under composition by union-find over the reachability just accumulated
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in orbit[i]:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    _, counts = np.unique(roots, return_counts=True)
    lookup = {r: c for r, c in zip(np.unique(roots), counts)}
    return np.array([lookup[r] for r in roots], dtype=np.int64)


def n_automorphisms(edge_index, n: int, labels: Optional[Sequence] = None,
                    cap: int = 100_000) -> int:
    """|Aut(G)| on the labelled graph."""
    return len(_automorphisms(edge_index, n, labels, cap))


# ----------------------------------------------------------------------------- INPUT

def _mol(smiles: str, explicit_h: bool):
    """One place where a SMILES becomes a molecule.

    Factored out because ``graph_from_smiles`` and ``bond_features_from_smiles`` must agree
    atom-for-atom and bond-for-bond: two independent constructions would drift the moment one
    of them changed an RDKit flag, and the symptom would be edge features silently attached
    to the wrong bonds.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'RDKit could not parse {smiles!r}')
    if explicit_h:
        mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def bond_features_from_smiles(smiles: str, explicit_h: bool = True) -> np.ndarray:
    """``[E, 4]`` aligned bond-for-bond with ``graph_from_smiles``' ``edge_index``.

    Columns: single, aromatic, double, triple. Bond ORDER rather than bond type, so an
    aromatic bond is not silently a fifth unrelated category.

    ⚠ ``IsInRing`` WAS A COLUMN HERE AND WAS REMOVED 2026-08-27. It leaked the battery's own
    block-B label: the flag reproduced ``ring_membership`` on 511/511 atoms in one hop with
    no learning, so every arm scored 1.000 and the block measured nothing. Ring membership is
    DERIVED TOPOLOGY, not a bond property the model is owed. Note the residual: an aromatic
    bond still implies a ring, so block B stays partially leaked on aromatic systems.
    """
    from rdkit import Chem
    mol = _mol(smiles, explicit_h)
    order = {Chem.BondType.SINGLE: 0, Chem.BondType.AROMATIC: 1,
             Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}
    out = np.zeros((mol.GetNumBonds(), 4), dtype=np.float64)
    for i, b in enumerate(mol.GetBonds()):
        k = order.get(b.GetBondType())
        if k is None:
            raise ValueError(f'unhandled bond type {b.GetBondType()} in {smiles!r}')
        out[i, k] = 1.0
    return out


def canonical_root(mol) -> int:
    """The atom of lowest canonical rank -- a function of the MOLECULE, not its serialisation.

    Two earlier root choices were wrong in opposite ways. Atom index 0 is a property of the
    SMILES text (label survived re-serialisation on 20.7% of atoms). Resampling the root per
    call fixed well-posedness but made the TRAINING SET NON-REPRODUCIBLE -- re-deriving it
    gave different roots and therefore a different `is_root` input column for every probe, so
    "train accuracy" stopped being a memorisation measure and the train/test decomposition
    the battery rests on quietly weakened.

    Canonical rank is both: invariant to atom ordering AND reproducible.
    """
    from rdkit import Chem
    return int(np.argmin(list(Chem.CanonicalRankAtoms(mol, breakTies=True))))


def canonical_parity(mol) -> np.ndarray:
    """``[n]`` in {-1, 0, +1} -- tetrahedral sign referenced to CANONICAL RANK order.

    ⚠ NO LONGER THE ENCODER INPUT. Superseded 2026-09-01 by :func:`cip_codes`, kept because
    the comparison between the two is what exposed the design error below.

    A tetrahedral sign is only a well-defined +/-1 once you FIX AN ORDERING of the four
    substituents; different orderings give different signs for the same physical molecule.
    This function references the sign to RDKit's canonical atom rank; ``cip_codes``
    references it to CIP substituent priority. Measured over 1,186 stereocentres the two
    AGREE ON ONLY 59.4%, because
        CIP = canonical_parity x sign(permutation between the two orderings)
    and that permutation is odd ~40% of the time.

    Feeding one convention as the INPUT while scoring against the other made the chirality
    probe a test of whether the model could re-derive RDKit's canonicalisation algorithm --
    arbitrary, chemically meaningless, and not the capability wanted. The model scored 0%
    (chance) while merely copying its own input would have scored 59.4%. ONE convention
    everywhere; CIP, because it is the chemical standard and equally order-stable (98.6%
    vs 98.5% survival under re-serialisation).
    WHY NOT ``GetChiralTag()`` DIRECTLY, WHICH IS WHAT THIS REPLACED. The CW/CCW tag is
    defined against the atom's BOND-LIST order, which is a property of the SMILES text.
    Measured: re-serialise the same molecule with a random atom order, align by isomorphism,
    and the raw tag survives on only **55.3%** of stereocentres -- a coin flip. The encoder's
    only chirality input was therefore noise correlated with serialisation, and every result
    claiming the parity channel was carried is void.

    Re-referencing the sign to ``CanonicalRankAtoms`` order makes it a function of the
    molecule. The permutation parity between the neighbour list and its rank-sorted order
    flips the tag's sign exactly when that permutation is odd.
    """
    from rdkit import Chem
    tags = {Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW: -1}
    rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    out = np.zeros(mol.GetNumAtoms(), dtype=np.int64)
    for a in mol.GetAtoms():
        base = tags.get(a.GetChiralTag(), 0)
        if base == 0:
            continue
        nbr = [n.GetIdx() for n in a.GetNeighbors()]
        order = sorted(range(len(nbr)), key=lambda i: rank[nbr[i]])
        # parity of the permutation, by counting inversions
        inv = sum(1 for i in range(len(order)) for j in range(i + 1, len(order))
                  if order[i] > order[j])
        out[a.GetIdx()] = base * (1 if inv % 2 == 0 else -1)
    return out


def cip_codes(mol) -> np.ndarray:
    """``[n]`` in {-1, 0, +1} -- CIP R/S per atom. R = +1, S = -1, unassigned = 0.

    THE ENCODER'S CHIRALITY INPUT, and also the tripwire probe's label -- deliberately the
    same quantity, so that reading handedness back out is TRIVIAL. If it is not ~100%, the
    parity channel has been dropped somewhere and everything chirality-related is void.

    A non-trivial chirality probe must require the sign to be COMBINED WITH STRUCTURE (see
    `chiral_moment`), not merely transported. Making the input and label disagree in order to
    manufacture difficulty is what broke the previous version.
    """
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler
    m = Chem.Mol(mol)
    try:
        rdCIPLabeler.AssignCIPLabels(m)
    except Exception:
        return np.zeros(m.GetNumAtoms(), dtype=np.int64)
    c = {'R': 1, 'S': -1}
    return np.array([c.get(a.GetPropsAsDict().get('_CIPCode'), 0) for a in m.GetAtoms()],
                    dtype=np.int64)


def wl_colours(z, edge_index, n: int, rounds: int = 3, modulus: int = 97) -> np.ndarray:
    """``[n]`` Weisfeiler-Lehman colours after ``rounds`` refinements, hashed to 1..modulus.

    A position-aware weight, so a chirality statistic built on it cannot collapse to a plain
    sum over centres.
    """
    a = to_dense_adjacency(edge_index, n)
    nbrs = [np.flatnonzero(a[i]).tolist() for i in range(n)]
    col = [int(x) for x in z]
    for _ in range(rounds):
        col = [hash((col[i], tuple(sorted(col[j] for j in nbrs[i])))) for i in range(n)]
    return np.array([abs(c) % modulus + 1 for c in col], dtype=np.int64)


def pi_degree(mol) -> np.ndarray:
    """``[n]`` -- sum over incident bonds of (bond order - 1); aromatic counts as 1.

    Exists so that at least ONE label depends on ``edge_attr``. Currently every probe target
    is a function of (z, adjacency) alone, so an encoder can score 100% across the whole
    battery while discarding all four bond-order columns -- and bond order is exactly what a
    torsion profile depends on.
    """
    from rdkit import Chem
    order = {Chem.BondType.SINGLE: 0, Chem.BondType.AROMATIC: 1,
             Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}
    out = np.zeros(mol.GetNumAtoms(), dtype=np.int64)
    for b in mol.GetBonds():
        k = order.get(b.GetBondType(), 0)
        out[b.GetBeginAtomIdx()] += k
        out[b.GetEndAtomIdx()] += k
    return out


def mol_for_labels(smiles: str, explicit_h: bool = True):
    """The same molecule object graph_from_smiles used, for label functions that need it."""
    return _mol(smiles, explicit_h)


def graph_from_smiles(smiles: str, explicit_h: bool = True
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``smiles -> (z [n], edge_index [2, E], parity [n])``.

    Hydrogens are EXPLICIT by default, because the conformer state includes hydrogen
    torsions -- an encoder trained on the heavy-atom skeleton would be blind to exactly the
    coordinates that were measured to dominate the descent behaviour.

    ``parity`` is the RDKit chiral tag reduced to {-1, 0, +1}. It is carried here so the
    labelled-graph symmetry functions can use it; note it is NOT the same convention as
    ``dof_features.atom_parity``, which is the improper-dihedral sign in TreeSpec's
    placement order. The two agree on WHICH atoms are stereocentres and may disagree on
    sign, so do not mix them within one experiment.
    """
    from rdkit import Chem
    mol = _mol(smiles, explicit_h)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    # CIP-REFERENCED, and this is the ONE convention. See canonical_parity's docstring for
    # the two conventions this replaced and why mixing them was a design error.
    parity = cip_codes(mol)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    edge_index = (np.array(bonds, dtype=np.int64).T if bonds
                  else np.zeros((2, 0), dtype=np.int64))
    return z, edge_index, parity
