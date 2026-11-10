"""Basin identity for the exp-6 recall test: the FD identity distance, minimised over the molecule's self-symmetries.

``mxtaltools.crystal_fd.metric`` ignores molecular point symmetry: a crystal and the same crystal with every molecule
replaced by its image under a symmetry S of the rigid conformer (acridine's C2, naphthalene's D2h, a planar
molecule's mirror) are one structure, and it sees them up to r_g * pi apart. Here each crystal is also described with
pose P S for every self-symmetry S of the conformer, and

    d_sym(A, B) = min over S of GluedFD(A, B S)

with GluedFD crystal_fd's glued distance. Self-symmetries are the orthogonal S (proper or improper) with
S x_i ~ x_pi(i) for a bond-graph automorphism pi, RMSD below ``SYM_TOL``, in the builder's standard frame
(``crystal_fd.metric.molecule_info``), where pose P = R diag(h, 1, 1) maps body coordinates to Cartesian.

``basins`` groups crystals into basins: pairs within ``df_kT`` in E/kT (polished minima of one energy agree to the
polish's tolerance) are linked when d_sym < ``FD_CUT`` (0.05 A, crystal_fd's identity cut), and basins are the linked
components.
"""
import networkx as nx
import numpy as np
import torch

F64 = torch.float64
SYM_TOL = 0.1  # A, RMSD of S x against x permuted by the automorphism (the X-ray conformers are symmetric only to ~0.05-0.1)
FD_CUT = 0.05  # A


def self_symmetries(mol_path, tol=SYM_TOL, max_auto=20000):
    """List of (S [3, 3] float64, rmsd A, det) for the conformer in ``mol_path``, identity first, distinct S only."""
    from eval.day_bench.p1_ingest import _bond_graph, _node_match
    from mxtaltools.crystal_fd.metric import molecule_info
    info = molecule_info(mol_path)
    X, z = info['X'].to(F64), info['z']
    g = _bond_graph(z.numpy(), X.numpy())
    gm = nx.algorithms.isomorphism.GraphMatcher(g, g, node_match=_node_match)
    out = [(torch.eye(3, dtype=F64), 0.0, 1.0)]
    for k, m in enumerate(gm.isomorphisms_iter()):
        if k >= max_auto:
            raise RuntimeError(f'more than {max_auto} automorphisms')
        Y = X[torch.tensor([m[i] for i in range(len(z))])]
        U, _, Vt = torch.linalg.svd(Y.T @ X)
        # both signs of the least singular direction: for a planar conformer the fit cannot tell the identity from
        # the mirror through the plane, and both are symmetries
        for flip in (1.0, -1.0):
            S = U @ torch.diag(torch.tensor([1.0, 1.0, flip], dtype=F64)) @ Vt
            rmsd = float((X @ S.T - Y).pow(2).sum(1).mean().sqrt())
            if rmsd < tol and all((S - s).abs().max() > 1e-3 for s, _, _ in out):
                out.append((S, rmsd, float(torch.det(S))))
    return out


def _can_dict(can):
    return {k: can[k] for k in ('cell', 't', 'R', 'h')}


def described(sg, params, hand, syms):
    """Canonical forms of every crystal (params [n, 12], hand [n]) under every self-symmetry: dict of cell, t, R, h
    with rows s * n + i for symmetry s (0 the identity) and crystal i."""
    from mxtaltools.crystal_fd.description import CrystalDescription
    from mxtaltools.crystal_fd.metric import canonical, canonical_desc
    can = _can_dict(canonical(sg, torch.as_tensor(params, dtype=F64), torch.as_tensor(hand).reshape(-1)))
    parts = [can]
    hh = torch.ones(len(can['t']), 3, dtype=F64)
    hh[:, 0] = can['h'].to(F64)
    P = can['R'] * hh[:, None, :]
    for S, _, _ in syms[1:]:
        desc = CrystalDescription.from_poses(sg, can['cell'], can['t'], P @ S)
        parts.append(_can_dict(canonical_desc(desc)))
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


class SymFD:
    """d_sym between crystals of one molecule and space group, all given at once (``params`` [n, 12], ``hand`` [n])."""

    def __init__(self, sg, mol_path, params, hand, device='cpu'):
        from mxtaltools.crystal_fd.metric import GluedFDMetric, molecule_info
        self.syms = self_symmetries(mol_path)
        self.n = len(params)
        self.k = len(self.syms)
        self.metric = GluedFDMetric(int(sg), molecule_info(mol_path)['rg'], device=device)
        self.P = self.metric.prepare_glued(described(int(sg), params, hand, self.syms))

    def pairs(self, ia, ib, cut=None):
        """[p] d_sym for pairs (ia[k], ib[k]); with ``cut``, pairs at or beyond it come back as inf."""
        ia = torch.as_tensor(ia, dtype=torch.long)
        ib = torch.as_tensor(ib, dtype=torch.long)
        best = torch.full((len(ia),), float('inf'), dtype=F64)
        for s in range(self.k):
            d = self.metric.distance(self.P, ia.to(self.metric.device), (s * self.n + ib).to(self.metric.device),
                                     cut=cut)
            best = torch.minimum(best, d.to(F64).cpu())
        return best


def basins(sg, mol_path, params, hand, f_kT, df_kT=0.1, cut=FD_CUT, device='cpu'):
    """Basin label [n] (0..B-1, in order of each basin's lowest E/kT) of each crystal, and the SymFD used."""
    f = torch.as_tensor(f_kT, dtype=F64)
    n = len(f)
    fd = SymFD(sg, mol_path, params, hand, device=device)
    order = torch.argsort(f)
    ia, ib = [], []
    fo = f[order]
    for a in range(n):  # candidate pairs: sorted by energy, within df_kT
        hi = int(torch.searchsorted(fo, fo[a] + df_kT, right=True))
        if hi > a + 1:
            ia.append(torch.full((hi - a - 1,), int(order[a])))
            ib.append(order[a + 1:hi])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if ia:
        ia, ib = torch.cat(ia), torch.cat(ib)
        for lo in range(0, len(ia), 200000):
            d = fd.pairs(ia[lo:lo + 200000], ib[lo:lo + 200000], cut=cut)
            for i, j in zip(ia[lo:lo + 200000][d < cut].tolist(), ib[lo:lo + 200000][d < cut].tolist()):
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    low = {}
    for i in range(n):
        low[roots[i]] = min(low.get(roots[i], float('inf')), float(f[i]))
    rank = {r: k for k, r in enumerate(sorted(low, key=low.get))}
    return torch.tensor([rank[r] for r in roots]), fd
