"""FD identity distance between two Z' = 1 crystals of one molecule in one space group, in Angstrom.

Both crystals are first put in canonical form by crystal_fd.canon.canonicalize (reduced cell; the +1 sheet through an
improper element of G; pose folded into ASU(Gamma+) x SO(3)). Then

    d(A, B) = min over g = (W, w) in Gamma+ and n in {-1, 0, 1}^3 of  sqrt( D_cell^2 + D_cent(g, n)^2 + D_rot(g)^2 )

    D_cell^2   = (a_A - a_B)^2 + (b_A - b_B)^2 + (c_A - c_B)^2
                 + (dalpha sqrt(b c))^2 + (dbeta sqrt(a c))^2 + (dgamma sqrt(a b))^2      [A^2; angles in rad,
                 lengths in the scale factors the means of A's and B's]
    D_cent     = | Tbar (u - n) |,  u = t_A - (W t_B + w) wrapped to [-1/2, 1/2)^3,  Tbar = (T_A + T_B) / 2   [A]
    D_rot      = r_g * theta(R_A, Q_W R_B),  theta the geodesic angle (rad) of R_A^T Q_W R_B,
                 Q_W = T_B W T_B^-1 projected on O(3),  r_g the molecule's radius of gyration           [A]

g runs over GroupTable.gamma_plus (crystal_fd.tables.group_table(sg)); r_g is the RMS distance of all atoms from the
heavy-atom centroid (the builder's centre). The cell term compares the two canonical cells directly: the faces of the
triclinic / monoclinic cell domains are NOT glued here (the chart does not record those gluings), so two nearly equal
crystals whose reduced cells lie on opposite sides of a cell-domain face are far apart in d. `face` below measures how
many campaign endpoints sit that close to a face.

Molecular point symmetry is ignored in d (owner decision). For a molecule with an approximate improper self-symmetry
sigma (a near-planar molecule: the mirror through its plane, which maps the atoms onto themselves up to a permutation
of like atoms that preserves the bond graph), d_twin(A, B) = min(d(A, B), d(A, B~)) with B~ the canonical form of
(t_B, P_B sigma) is offered as a flagged variant.

    python -m energy_sampling.eval.fd_analysis.fd_distance --campaign mipcas --out fd_distance_mipcas
"""
import argparse
import os
import time

import numpy as np
import torch

from energy_sampling.eval.fd_analysis import common as C

F64 = torch.float64


# ----------------------------------------------------------------------------
# molecule
# ----------------------------------------------------------------------------
def molecule_info(mol_path):
    """Body coordinates X in the builder's standard frame (align_mol_batch_to_standard_axes, h = +1, heavy-atom
    centred), r_g, the heavy-atom plane normal n there, sigma = I - 2 n n^T, and the mirror's fit: the RMSD of sigma X
    against X in identity atom order (mirror_rmsd), and with the atoms of each element matched by optimal assignment
    (Hungarian on squared distances; perm[i] = the atom sigma X_i lands on; mirror_rmsd_matched, mirror_maxdev_matched,
    all atoms). The mirror swaps out-of-plane H atoms onto each other, so the matched figure is the one that says
    whether the twin is the same crystal (common.molecule_template checks that perm preserves the bond graph)."""
    from mxtaltools.crystal_building.utils import align_mol_batch_to_standard_axes
    from mxtaltools.dataset_utils.utils import collate_data_list
    m = torch.load(mol_path, weights_only=False)
    m = m[0] if isinstance(m, list) else m
    b = collate_data_list([m])
    out = align_mol_batch_to_standard_axes(b.clone())
    pos = (out.pos if hasattr(out, 'pos') else out[0].pos).double()
    z = b.z
    heavy = z > 1
    X = pos - pos[heavy].mean(0)
    rg = float(X.pow(2).sum(1).mean().sqrt())
    U, S, Vt = torch.linalg.svd(X[heavy])
    n = Vt[-1]
    sigma = torch.eye(3, dtype=F64) - 2 * torch.outer(n, n)
    SX = X @ sigma.T
    rmsd = float((SX - X).pow(2).sum(1).mean().sqrt())
    from scipy.optimize import linear_sum_assignment
    perm = np.arange(len(z))
    for el in torch.unique(z).tolist():
        ids = torch.nonzero(z == el).flatten().numpy()
        cost = torch.cdist(SX[ids], X[ids]).pow(2).numpy()
        r, c = linear_sum_assignment(cost)
        perm[ids[r]] = ids[c]
    dev = (SX - X[torch.as_tensor(perm)]).norm(dim=1)
    from mxtaltools.crystal_fd.description import MoleculeRef
    return dict(X=X, z=z, rg=rg, normal=n, sigma=sigma, mirror_rmsd=rmsd, n_atoms=len(z), perm=perm,
                mirror_rmsd_matched=float(dev.pow(2).mean().sqrt()), mirror_maxdev_matched=float(dev.max()),
                perm_moves=int((perm != np.arange(len(z))).sum()),
                out_of_plane_max=float((X[heavy] @ n).abs().max()), ref=MoleculeRef.from_coords(X, z))


# ----------------------------------------------------------------------------
# canonical forms
# ----------------------------------------------------------------------------
def canonical_desc(desc, chunk=4096):
    """canon.canonicalize in chunks; returns dict(cell [n,6], t [n,3], R [n,3,3], status (list of names), near_face,
    n_auto, N [n,3,3])."""
    from mxtaltools.crystal_fd import canon
    from mxtaltools.crystal_fd.description import CrystalDescription
    parts = []
    for lo in range(0, desc.num_rows, chunk):
        c = canon.canonicalize(desc[lo:lo + chunk], moved_tol=1e-5)
        parts.append(dict(cell=c.desc.cell.double(), t=c.desc.t[:, 0].double(), R=c.desc.R[:, 0].double(),
                          h=c.desc.h[:, 0], status=c.names(), near_face=c.near_face, n_auto=c.n_auto, N=c.N))
    out = {k: torch.cat([p[k] for p in parts]) for k in ('cell', 't', 'R', 'h', 'near_face', 'n_auto', 'N')}
    out['status'] = sum((p['status'] for p in parts), [])
    return out


def canonical(sg, params, hand, chunk=4096):
    """Canonical forms of rows in MolCrystalData parameter layout (lengths, angles, centroid, rotation vector)."""
    from mxtaltools.crystal_fd.description import CrystalDescription
    p = torch.as_tensor(params, dtype=F64)
    desc = CrystalDescription.from_crystal_params(sg, p[:, :3], p[:, 3:6], p[:, 6:9], p[:, 9:12],
                                                  torch.as_tensor(hand).reshape(-1, 1))
    return canonical_desc(desc, chunk)


def twin_canonical(sg, can, sigma, chunk=4096):
    """Canonical forms of (t, R diag(h,1,1) sigma): the crystal with the mirror-image molecule at the same poses (the
    same crystal to within the matched mirror RMSD for a molecule with sigma X ~ X up to a bond-preserving permutation
    of like atoms)."""
    from mxtaltools.crystal_fd.description import CrystalDescription
    hh = torch.ones(len(can['t']), 3, dtype=F64)
    hh[:, 0] = can['h'].to(F64)
    P = (can['R'] * hh[:, None, :]) @ sigma
    desc = CrystalDescription.from_poses(sg, can['cell'], can['t'], P)
    return canonical_desc(desc, chunk)


# ----------------------------------------------------------------------------
# the distance
# ----------------------------------------------------------------------------
class FDMetric:
    """d(A, B) of the module docstring for one space group and molecule (r_g)."""

    def __init__(self, sg, rg):
        from mxtaltools.crystal_fd.tables import group_table
        tab = group_table(sg)
        self.sg = sg
        self.W = tab.gamma_plus_W.to(F64)
        self.w = tab.gamma_plus_w.to(F64)
        self.m = len(self.W)
        self.n27 = torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * 3, indexing='ij'), -1).reshape(-1, 3).to(F64)
        self.rg = float(rg)

    def prepare(self, can):
        """Per-crystal tensors: cell, T, t, R and Q_W R for every element."""
        from mxtaltools.crystal_fd.ops import cartesian_rotation, cell_matrices
        cell, t, R = can['cell'].to(F64), can['t'].to(F64), can['R'].to(F64)
        n, m = len(cell), self.m
        T = cell_matrices(cell)
        QR = torch.empty(n, m, 3, 3, dtype=F64)
        for lo in range(0, n, 8192):
            Tb = T[lo:lo + 8192]
            b = len(Tb)
            Q = cartesian_rotation(Tb[:, None].expand(b, m, 3, 3).reshape(-1, 3, 3),
                                   self.W[None].expand(b, m, 3, 3).reshape(-1, 3, 3)).reshape(b, m, 3, 3)
            QR[lo:lo + b] = Q @ R[lo:lo + b, None]
        return dict(cell=cell, T=T, t=t, R=R, QR=QR)

    @staticmethod
    def cell2(ca, cb):
        """[p] D_cell^2 for paired cells [p, 6]."""
        dl = ca[:, :3] - cb[:, :3]
        L = 0.5 * (ca[:, :3] + cb[:, :3])
        sc = torch.stack([torch.sqrt(L[:, 1] * L[:, 2]), torch.sqrt(L[:, 0] * L[:, 2]), torch.sqrt(L[:, 0] * L[:, 1])], 1)
        da = (ca[:, 3:] - cb[:, 3:]) * sc
        return (dl * dl).sum(1) + (da * da).sum(1)

    def theta(self, Ra, QRb):
        """[p, m] geodesic angles between R_A [p,3,3] and Q_W R_B [p,m,3,3]."""
        tr = torch.einsum('pij,pmij->pm', Ra, QRb)
        return torch.arccos(((tr - 1) / 2).clamp(-1, 1))

    def cent(self, ta, tb, Ta, Tb):
        """[p, m] min over the 27 lattice shifts of |Tbar (u - n)|."""
        y = torch.einsum('mij,pj->pmi', self.W, tb) + self.w
        u = ta[:, None] - y
        u = u - torch.round(u)
        Tm = 0.5 * (Ta + Tb)
        x = u[:, :, None, :] - self.n27
        return torch.einsum('pij,pmsj->pmsi', Tm, x).norm(dim=-1).amin(-1)

    def pairs(self, A, ia, B, ib, chunk=4096, parts=False):
        """d for pairs (A[ia[k]], B[ib[k]]); with parts also (D_cell, D_cent, D_rot, element) at the minimum."""
        dev, dt = A['cell'].device, A['cell'].dtype
        ia = torch.as_tensor(ia, dtype=torch.long, device=dev)
        ib = torch.as_tensor(ib, dtype=torch.long, device=dev)
        out = torch.empty(len(ia), dtype=dt, device=dev)
        P = torch.empty(len(ia), 4, dtype=dt, device=dev) if parts else None
        for lo in range(0, len(ia), chunk):
            a, b = ia[lo:lo + chunk], ib[lo:lo + chunk]
            c2 = self.cell2(A['cell'][a], B['cell'][b])
            th = self.theta(A['R'][a], B['QR'][b])
            ce = self.cent(A['t'][a], B['t'][b], A['T'][a], B['T'][b])
            tot = c2[:, None] + ce * ce + (self.rg * th) ** 2
            v, k = tot.min(1)
            out[lo:lo + chunk] = v.sqrt()
            if parts:
                r = torch.arange(len(a), device=dev)
                P[lo:lo + chunk] = torch.stack([c2.sqrt(), ce[r, k], self.rg * th[r, k], k.to(dt)], 1)
        return (out, P) if parts else out

    def lower_bound(self, A, ia, B, ib):
        """[p] sqrt(D_cell^2 + (r_g min_g theta)^2) <= d (the pre-screen: no centroid term)."""
        c2 = self.cell2(A['cell'][ia], B['cell'][ib])
        th = self.theta(A['R'][ia], B['QR'][ib]).amin(1)
        return (c2 + (self.rg * th) ** 2).sqrt()

    def one_to_many(self, A, i, B, idx, cut):
        """Distances from A[i] to B[idx] where the pre-screen allows d < cut; inf elsewhere."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        out = torch.full((len(idx),), float('inf'), dtype=F64)
        if not len(idx):
            return out
        ia = torch.full((len(idx),), int(i), dtype=torch.long)
        lb = self.lower_bound(A, ia, B, idx)
        cand = torch.nonzero(lb < cut).flatten()
        if len(cand):
            out[cand] = self.pairs(A, ia[cand], B, idx[cand])
        return out

    def matrix(self, A, cap, chunk=64):
        """[n, n] distances within one prepared set; entries whose lower bound is >= cap hold the lower bound (and are
        flagged in the returned bool mask exact=False)."""
        n = len(A['cell'])
        D = torch.empty(n, n, dtype=F64)
        exact = torch.zeros(n, n, dtype=torch.bool)
        allj = torch.arange(n)
        for lo in range(0, n, chunk):
            rows = torch.arange(lo, min(n, lo + chunk))
            ia = rows.repeat_interleave(n)
            ib = allj.repeat(len(rows))
            lb = self.lower_bound(A, ia, A, ib)
            cand = torch.nonzero(lb < cap).flatten()
            val = lb.clone()
            if len(cand):
                val[cand] = self.pairs(A, ia[cand], A, ib[cand])
            D[rows] = val.reshape(len(rows), n)
            ex = torch.zeros(len(ia), dtype=torch.bool)
            ex[cand] = True
            exact[rows] = ex.reshape(len(rows), n)
        return D, exact


def _setting_candidates(sg, max_entry=2):
    """Integer det +1 matrices with entries in [-max_entry, max_entry], identity first, whose rotational action maps
    the point operations of SYM_OPS[sg] onto themselves (N^-1 W N in the set): the candidates for ``setting_shift``.

    A setting group of sg 3-15 is a proper subgroup of GL(3, Z), and its cell domain's faces can be glued by matrices
    with an entry of 2 (P2_1/c: c -> c + 2a keeps the c glide, c -> c + a does not), which entries in {-1, 0, 1} miss.
    Triclinic groups take every unimodular matrix, and the Niggli faces are glued within {-1, 0, 1}."""
    from mxtaltools.crystal_search.standardize import _ops
    r = np.arange(-max_entry, max_entry + 1)
    allm = np.array(np.meshgrid(*[r] * 9, indexing='ij')).reshape(9, -1).T.reshape(-1, 3, 3)
    allm = allm[np.round(np.linalg.det(allm.astype(float))).astype(np.int64) == 1]
    W = np.round(_ops(sg)[0]).astype(np.int64)
    Wset = {w.tobytes() for w in W}
    keep = np.ones(len(allm), dtype=bool)
    inv = np.round(np.linalg.inv(allm.astype(float))).astype(np.int64)
    for w in np.unique(W, axis=0):
        conj = inv @ w[None] @ allm
        keep &= np.array([c.tobytes() in Wset for c in conj])
    m = allm[keep]
    eye = np.eye(3, dtype=np.int64)
    return np.concatenate([eye[None], m[~(m == eye).all(axis=(1, 2))]])


class GluedFDMetric(FDMetric):
    """``FDMetric`` with the faces of the cell domain glued (sg 1-15): d(A, B) is also minimised over B's near-minimal
    bases, so two crystals on opposite sides of a reduction face are as close as their atoms are.

    The cell domain's faces are glued by lattice basis changes N (integer, det +1, in the space group's setting group).
    Near a face, the reduced basis of a crystal is one of several bases of nearly the same total edge length, and a
    nearby crystal reduces to another of them. So each crystal B is expanded into its alternatives: every N of
    ``crystal_fd.ops._neighbourhood`` (entries in {-1, 0, 1}, det +1, 3480) for sg 1-2, of ``_setting_candidates``
    (entries in {-2, ..., 2}) for sg 3-15 -- the setting group's only, with its origin shift -- whose basis has
    a + b + c within ``margin`` A of the reduced one, applied jointly to cell, centre and
    orientation (``ops.change_basis``); the identity is always one. d(A, B) = min over B's alternatives of FDMetric's
    d(A, alternative). One side suffices: if A is near B, some basis of B is near A's reduced basis, and its edge sum
    is within O(d) of B's minimum.

    Exactness is local, a working assumption to be checked empirically: the minimum is the quotient distance for
    pairs whose nearest images lie within the expanded set -- lattice shifts in {-1, 0, 1}^3, Gamma+ elements, and
    basis changes of the candidate set within ``margin``. (With {-1, 0, 1} only on sg 14, five NEHZOR pairs that
    COMPACK matches at 0.02-0.09 A sat at 2.5-3.5 A: their cells lie on the face |c cos beta| = a, glued by
    c -> c + 2a; 2026-10-28.) For pairs farther apart it is an upper bound. sg >= 16
    (snapped cells, canonical axis order) gets the identity only: axis-order faces are not glued here.

    ``device``: tensors live there; CUDA runs float32 (distances to ~1e-5 A), CPU float64.
    """

    def __init__(self, sg, rg, margin=1.0, device='cpu'):
        super().__init__(sg, rg)
        self.margin = float(margin)
        self.device = torch.device(device)
        self.dtype = torch.float32 if self.device.type == 'cuda' else F64
        from mxtaltools.crystal_fd import ops
        from mxtaltools.crystal_search.standardize import setting_shift
        if sg <= 15:
            Ns, ss = [], []
            for N in (ops._neighbourhood().numpy() if sg <= 2 else _setting_candidates(sg)):
                o = setting_shift(N, sg)
                if o is not None:
                    Ns.append(N)
                    ss.append(o)
            self.N = torch.as_tensor(np.stack(Ns), dtype=F64)
            self.s = torch.as_tensor(np.stack(ss), dtype=F64)
        else:
            self.N = torch.eye(3, dtype=F64)[None]
            self.s = torch.zeros(1, 3, dtype=F64)
        if not torch.equal(self.N[0], torch.eye(3, dtype=F64)):
            raise RuntimeError('the identity must be the first basis change')
        self.W, self.w, self.n27 = (x.to(self.device, self.dtype) for x in (self.W, self.w, self.n27))

    def alternatives(self, can, chunk=2048):
        """B's near-minimal descriptions: dict(cell, t, R, h, owner [K], ptr [n + 1]) with rows ptr[j]:ptr[j+1] the
        alternatives of crystal j, its own description first."""
        from mxtaltools.crystal_fd import ops
        from mxtaltools.crystal_fd.description import CrystalDescription
        cell = can['cell'].to(F64)
        T = ops.cell_matrices(cell)
        owner, which = [], []
        for lo in range(0, len(cell), chunk):
            Tb = T[lo:lo + chunk]
            L = (Tb[:, None] @ self.N[None]).norm(dim=2).sum(-1)  # [b, K] a + b + c of each basis
            keep = L <= L[:, :1] + self.margin
            r, k = torch.nonzero(keep, as_tuple=True)
            owner.append(r + lo)
            which.append(k)
        owner, which = torch.cat(owner), torch.cat(which)
        order = torch.argsort(owner * len(self.N) + which)  # grouped by owner, identity first
        owner, which = owner[order], which[order]
        h = can['h'] if 'h' in can else torch.ones(len(cell), dtype=torch.long)
        base = CrystalDescription.from_parts(self.sg, cell[owner], can['t'].to(F64)[owner], can['R'].to(F64)[owner],
                                             h[owner])
        alt = ops.change_basis(base, self.N[which], self.s[which])
        counts = torch.bincount(owner, minlength=len(cell))
        ptr = torch.zeros(len(cell) + 1, dtype=torch.long)
        ptr[1:] = torch.cumsum(counts, 0)
        return dict(cell=alt.cell.double(), t=alt.t[:, 0].double(), R=alt.R[:, 0].double(), h=alt.h[:, 0],
                    owner=owner, ptr=ptr)

    def prepare(self, can):
        P = super().prepare(can)  # float64 on the CPU (crystal_fd's frame helpers)
        out = {k: v.to(self.device, self.dtype) for k, v in P.items()}
        for k in ('owner', 'ptr'):
            if k in can:
                out[k] = can[k].to(self.device)
        return out

    def prepare_glued(self, can):
        """(A, Balt): the crystals prepared as themselves (the A side) and with their alternatives (the B side)."""
        return self.prepare(can), self.prepare(self.alternatives(can))

    def _expand(self, B, ib):
        """Pair index -> alternative rows of B: (rep [E], alt [E]) over all alternatives of every ib."""
        ptr = B['ptr']
        start, cnt = ptr[ib], ptr[ib + 1] - ptr[ib]
        rep = torch.repeat_interleave(torch.arange(len(ib), device=ib.device), cnt)
        first = torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        alt = torch.repeat_interleave(start, cnt) + torch.arange(len(rep), device=ib.device) - first
        return rep, alt

    def distance(self, P, ia, ib, cut=None, chunk=8192):
        """Symmetric glued distance between crystals ia[k] and ib[k] of one prepared set P = ``prepare_glued(can)``:
        the smaller of the two one-sided minima (each expands one side's bases; the two differ by up to ~0.03 A on
        MIPCAS through the mean-cell centre term). With ``cut``, pairs farther than cut come back as inf."""
        A, B = P
        d1 = self.pairs(A, ia, B, ib, chunk=chunk, cut=cut)
        d2 = self.pairs(A, ib, B, ia, chunk=chunk, cut=cut)
        return torch.minimum(d1, d2)

    def pairs(self, A, ia, B, ib, chunk=8192, parts=False, cut=None):
        """d for pairs (A[ia[k]], crystal ib[k] of the glued set B); ``parts`` also gives (D_cell, D_cent, D_rot,
        element, alternative row) at the minimum. With ``cut``: pairs whose distance is >= cut return inf, and
        alternatives whose cell term alone reaches cut are never evaluated."""
        if 'ptr' not in B:
            return super().pairs(A, ia, B, ib, chunk=chunk, parts=parts)
        dev = self.device
        ia = torch.as_tensor(ia, dtype=torch.long, device=dev)
        ib = torch.as_tensor(ib, dtype=torch.long, device=dev)
        rep, alt = self._expand(B, ib)
        # exact pruning: D_cell <= d for every alternative, so (1) the full d through each pair's cell-closest
        # alternative is an upper bound, and (2) only alternatives whose D_cell is below it can lower the minimum
        c2 = torch.cat([self.cell2(A['cell'][ia[rep[lo:lo + chunk]]], B['cell'][alt[lo:lo + chunk]])
                        for lo in range(0, len(rep), chunk)]) if len(rep) else torch.zeros(0, dtype=self.dtype, device=dev)
        min_c2 = torch.full((len(ia),), float('inf'), dtype=self.dtype, device=dev)
        min_c2.scatter_reduce_(0, rep, c2, reduce='amin')
        at_min = c2 <= min_c2[rep]
        arg = torch.zeros(len(ia), dtype=torch.long, device=dev)
        arg[rep[at_min]] = alt[at_min]
        best = torch.full((len(ia),), float('inf'), dtype=self.dtype, device=dev)
        live = torch.ones(len(ia), dtype=torch.bool, device=dev) if cut is None else min_c2 < cut * cut
        li = torch.nonzero(live).flatten()
        if len(li):
            best[li] = FDMetric.pairs(self, A, ia[li], B, arg[li], chunk=chunk)
        bound = best[rep] ** 2 if cut is None else torch.minimum(best[rep] ** 2, torch.full_like(c2, cut * cut))
        cand = torch.nonzero((c2 < bound) & (alt != arg[rep])).flatten()
        for lo in range(0, len(cand), chunk):
            e = cand[lo:lo + chunk]
            r, a = rep[e], alt[e]
            v = FDMetric.pairs(self, A, ia[r], B, a, chunk=chunk)
            best.scatter_reduce_(0, r, v, reduce='amin')
            hit = v <= best[r]
            arg[r[hit]] = a[hit]
        self.last_full_evals = int(len(li) + len(cand))  # full distances computed (vs len(rep) without pruning)
        if cut is not None:
            best = torch.where(best < cut, best, torch.full_like(best, float('inf')))
        if parts:
            _, P = FDMetric.pairs(self, A, ia, B, arg, chunk=chunk, parts=True)
            return best, torch.cat([P, arg[:, None].to(P.dtype)], 1)
        return best


def merge_twin(A, At):
    """A prepared set holding both forms: rows [0, n) the canonical forms, [n, 2n) their twins."""
    return {k: torch.cat([A[k], At[k]]) for k in A}


# ----------------------------------------------------------------------------
# CLI: checks and the face census
# ----------------------------------------------------------------------------
def face_census(sg, can, metric, deltas, rng):
    """Axis-aligned cell perturbations of size delta (in D_cell's units) of each canonical row, t and R kept;
    re-canonicalised; a row is near a face at delta if some perturbation's canonical form is further than
    1.5 delta + 1e-4 A from the row (no face crossed gives d <= delta exactly). Returns per delta the counts."""
    from mxtaltools.crystal_fd.description import CrystalDescription
    n = len(can['cell'])
    axes = [0, 1, 2, 3, 4, 5] if sg <= 2 else [0, 1, 2, 4]
    L = can['cell'][:, :3]
    sc = torch.stack([torch.sqrt(L[:, 1] * L[:, 2]), torch.sqrt(L[:, 0] * L[:, 2]), torch.sqrt(L[:, 0] * L[:, 1])], 1)
    A = metric.prepare(can)
    res = {}
    for delta in deltas:
        split_any = torch.zeros(n, dtype=torch.bool)
        split_axis = torch.zeros(len(axes) * 2, dtype=torch.long)
        basis_changed_any = torch.zeros(n, dtype=torch.bool)
        dmax = torch.zeros(n, dtype=F64)
        k = 0
        for ax in axes:
            for sgn in (1.0, -1.0):
                cell = can['cell'].clone()
                if ax < 3:
                    cell[:, ax] += sgn * delta
                else:
                    cell[:, ax] += sgn * delta / sc[:, ax - 3]
                desc = CrystalDescription.from_parts(sg, cell, can['t'], can['R'], torch.ones(n, dtype=torch.long))
                pc = canonical_desc(desc)
                ok = torch.tensor([s in ('canonical', 'special_metric') for s in pc['status']])
                B = metric.prepare(pc)
                d = metric.pairs(A, torch.arange(n), B, torch.arange(n))
                d = torch.where(ok, d, torch.full_like(d, float('nan')))
                sp = ok & (d > 1.5 * delta + 1e-4)
                split_any |= sp
                split_axis[k] = int(sp.sum())
                basis_changed_any |= ok & (pc['N'] != torch.eye(3, dtype=torch.long)).flatten(1).any(1)
                dmax = torch.maximum(dmax, torch.nan_to_num(d, nan=0.0))
                k += 1
        res[delta] = dict(split=int(split_any.sum()), basis_changed=int(basis_changed_any.sum()),
                          per_axis=split_axis.tolist(), split_mask=split_any, dmax_median_split=float(
                dmax[split_any].median()) if split_any.any() else float('nan'))
    return res, axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True, choices=list(C.CAMPAIGNS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--band_kT', type=float, default=2.0)
    ap.add_argument('--n_face', type=int, default=5000)
    ap.add_argument('--n_check', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    C.setup()
    out = C.new_out_dir(a.out)
    rng = np.random.default_rng(a.seed)
    co, cfg, reg, ref = C.load_campaign(a.campaign)
    sg = cfg.sg
    mi = molecule_info(cfg.mol_path)
    lines = [f'# FD identity distance: {a.campaign} (sg {sg}, Z\'=1)', '', __doc__.split('\n\n    python')[0], '',
             f'Molecule ({cfg.mol_path}): {mi["n_atoms"]} atoms; r_g {mi["rg"]:.4f} A (all atoms about the heavy-atom '
             f'centroid, builder frame); heavy-atom plane normal {[round(float(v), 4) for v in mi["normal"]]}; '
             f'mirror through that plane: RMSD {mi["mirror_rmsd"]:.4f} A in identity atom order, '
             f'{mi["mirror_rmsd_matched"]:.4f} A (largest atom deviation {mi["mirror_maxdev_matched"]:.4f} A) with like '
             f'atoms matched by optimal assignment ({mi["perm_moves"]} atoms swapped); heavy-atom out-of-plane max '
             f'{mi["out_of_plane_max"]:.4f} A.', '']
    from mxtaltools.crystal_fd.tables import group_table
    tab = group_table(sg)
    lines += [f'Gamma+ of sg {sg}: {len(tab.gamma_plus)} elements (W distinct: '
              f'{len(torch.unique(tab.gamma_plus_W.reshape(len(tab.gamma_plus), -1), dim=0))}); lattice shifts '
              f'{{-1,0,1}}^3 after wrapping, so {27 * len(tab.gamma_plus)} images per pair.', '']
    st = C.shard_states(a.campaign, co, cfg, reg, ref, a.band_kT)
    del reg
    n_all = len(st['energy'])
    idx = np.sort(rng.choice(n_all, min(a.n_face, n_all), replace=False))
    s = C.subset(st, idx)
    C.log(f'{n_all} shard states within {a.band_kT} kT; {len(idx)} sampled')

    # canonicalisation, timed
    t0 = time.time()
    can = canonical(sg, s['params'], s['hand'])
    t_can = (time.time() - t0) / len(idx)
    stat = {k: can['status'].count(k) for k in set(can['status'])}
    nf = can['near_face']
    lines += [f'## Canonical forms of {len(idx)} endpoints (random, seed {a.seed}, of {n_all} shard states within '
              f'{a.band_kT} kT)', '',
              f'Status: {stat}. near_face at the default tolerance (1e-5, chart units): NEAR_CELL {int(((nf & 1) > 0).sum())}, '
              f'NEAR_CENTROID {int(((nf & 2) > 0).sum())}, NEAR_FOLD_BAND {int(((nf & 4) > 0).sum())}; special metrics '
              f'(n_auto > 1) {int((can["n_auto"] > 1).sum())}. canonicalize: {1e3 * t_can:.2f} ms per row.', '']
    metric = FDMetric(sg, mi['rg'])
    A = metric.prepare(can)

    # check 1: re-descriptions of the same crystal (a Gamma+ element, a lattice shift, sg 1-15 a basis change of the
    # setting group) come back at distance ~0
    from mxtaltools.crystal_fd import ops
    from mxtaltools.crystal_fd.description import CrystalDescription
    nc = min(a.n_check, len(idx))
    sel = torch.as_tensor(rng.choice(len(idx), nc, replace=False))
    base = CrystalDescription.from_parts(sg, can['cell'][sel], can['t'][sel], can['R'][sel], can['h'][sel])
    k = torch.as_tensor(rng.integers(0, metric.m, nc))
    img = ops.apply(base, metric.W[k], metric.w[k] + torch.as_tensor(rng.integers(-2, 3, (nc, 3)), dtype=F64))
    if sg <= 2:
        Ns = []
        for _ in range(nc):
            while True:
                N = rng.integers(-1, 2, (3, 3))
                if round(np.linalg.det(N)) == 1:
                    break
            Ns.append(N)
        img = ops.change_basis(img, torch.as_tensor(np.stack(Ns)), torch.zeros(nc, 3, dtype=F64))
    else:
        from mxtaltools.crystal_search.standardize import keeps_operators
        Ns = []
        for _ in range(nc):
            while True:
                p_, q_, r_, s_ = rng.integers(-1, 2, 4)
                N = np.array([[p_, 0, r_], [0, p_ * s_ - q_ * r_, 0], [q_, 0, s_]])
                if round(np.linalg.det(N)) == 1 and keeps_operators(N, sg):
                    break
            Ns.append(N)
        img = ops.change_basis(img, torch.as_tensor(np.stack(Ns)))
    ci = canonical_desc(img)
    d_self = metric.pairs(A, sel, metric.prepare(ci), torch.arange(nc))
    lines += [f'Check (same crystal re-described: random Gamma+ element, lattice shift in [-2,2]^3, random basis change '
              f'of the setting group; {nc} rows): d max {float(d_self.max()):.2e} A, median {float(d_self.median()):.2e} A; '
              f'statuses {dict((s_, ci["status"].count(s_)) for s_ in set(ci["status"]))}.', '']

    # check 2: d against the literal largest atom deviation for small moves of t and R (same cell)
    from mxtaltools.crystal_fd.checks import same_crystal
    from mxtaltools.crystal_fd.so3 import rotvec_to_matrix
    eps = torch.as_tensor(rng.normal(size=(nc, 3)), dtype=F64)
    eps = 0.05 * eps / eps.norm(dim=1, keepdim=True)
    rv = torch.as_tensor(rng.normal(size=(nc, 3)), dtype=F64)
    rv = (0.05 / mi['rg']) * rv / rv.norm(dim=1, keepdim=True)
    mv = CrystalDescription.from_parts(sg, base.cell, base.t[:, 0] + torch.linalg.solve(base.cell_matrix(), eps[..., None])[..., 0],
                                       rotvec_to_matrix(rv) @ base.R[:, 0], base.h[:, 0])
    dev = same_crystal(base, mv, mi['ref'])
    dm = metric.pairs(A, sel, metric.prepare(canonical_desc(mv)), torch.arange(nc))
    lines += [f'Scale check (t moved 0.05 A and R turned 0.05/r_g rad, random directions, same cell; {nc} rows): d '
              f'median {float(dm.median()):.4f} A (expected ~0.0707 = 0.05 sqrt 2); literal largest atom deviation '
              f'(checks.same_crystal) median {float(dev.median()):.4f} A, max {float(dev.max()):.4f} A.', '']

    # check 3: the molecular-mirror twin
    tw = twin_canonical(sg, can, mi['sigma'])
    At = metric.prepare(tw)
    d_tw = metric.pairs(A, torch.arange(len(idx)), At, torch.arange(len(idx)), parts=True)
    twd = CrystalDescription.from_poses(sg, can['cell'][sel], can['t'][sel], can['R'][sel] @ mi['sigma'])
    dev_tw = same_crystal(base, twd, mi['ref'])
    lines += [f'Mirror twin B~ = (t, R sigma) of every sampled endpoint: literal largest atom deviation from the endpoint '
              f'in identity atom order median {float(dev_tw.median()):.4f} A, max {float(dev_tw.max()):.4f} A ({nc} rows); '
              f'with like atoms matched it is the molecule\'s matched largest deviation, '
              f'{mi["mirror_maxdev_matched"]:.4f} A, for every row (a rigid motion keeps distances). d(endpoint, its own '
              f'twin) median {float(d_tw[0].median()):.3f} A, 10% {float(d_tw[0].quantile(0.1)):.3f} A, rotation part '
              f'median {float(d_tw[1][:, 2].median()):.3f} A. A twin is the same crystal to the matched mirror RMSD '
              f'({mi["mirror_rmsd_matched"]:.4f} A here) when the matching preserves the bond graph.', '']

    # timing of the distance
    ia = torch.as_tensor(rng.integers(0, len(idx), 20000))
    ib = torch.as_tensor(rng.integers(0, len(idx), 20000))
    t0 = time.time()
    metric.pairs(A, ia, A, ib)
    t_pair = (time.time() - t0) / 20000
    t0 = time.time()
    metric.lower_bound(A, ia, A, ib)
    t_lb = (time.time() - t0) / 20000
    lines += [f'Cost: d {1e6 * t_pair:.1f} us per pair ({27 * metric.m} images), pre-screen bound {1e6 * t_lb:.2f} us per '
              f'pair; {C.machine_note()}.', '']

    # face census
    deltas = (0.02, 0.05, 0.1, 0.2, 0.5)
    t0 = time.time()
    fc, axes = face_census(sg, can, metric, deltas, rng)
    names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    lines += ['## Endpoints near a cell-domain face', '',
              f'*Endpoints whose canonical form jumps under an axis-aligned cell perturbation of size delta (each of '
              f'{", ".join(names[x] for x in axes)} moved by +-delta in D_cell units, t and R kept, re-canonicalised): '
              f'the re-canonicalised form lies more than 1.5 delta + 1e-4 A from the endpoint in d, i.e. d would split '
              f'two crystals delta apart. {len(idx)} endpoints within {a.band_kT} kT; the same rows at every delta. '
              f'"basis changed" counts endpoints where some perturbation changed the reduced basis (N != I), split or '
              f'not. ({time.time() - t0:.0f} s)*', '',
              '| delta (A) | endpoints split | share | basis changed | median largest d of split rows (A) | splits per '
              'perturbation (' + ', '.join(f'{s}{names[x]}' for x in axes for s in '+-') + ') |',
              '|---|---|---|---|---|---|']
    for dlt in deltas:
        r = fc[dlt]
        lines.append(f'| {dlt} | {r["split"]} | {r["split"] / len(idx):.4f} | {r["basis_changed"]} | '
                     f'{r["dmax_median_split"]:.3f} | {r["per_axis"]} |')
    lines.append('')
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    torch.save(dict(idx=idx, can={k: v for k, v in can.items() if k != 'status'}, status=can['status'],
                    face={d: {k: v for k, v in r.items()} for d, r in fc.items()}, rg=mi['rg'],
                    mirror_rmsd=mi['mirror_rmsd'], mirror_rmsd_matched=mi['mirror_rmsd_matched'],
                    mirror_maxdev_matched=mi['mirror_maxdev_matched'], perm=mi['perm'], sigma=mi['sigma'], t_can=t_can, t_pair=t_pair, t_lb=t_lb,
                    d_self=d_self, d_twin_self=d_tw[0], scale_d=dm, scale_dev=dev),
               os.path.join(out, 'results.pt'))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
