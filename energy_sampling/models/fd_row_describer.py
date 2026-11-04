"""Typed FD rows for the AR flow's conditioner (docs/design/conditional_ar_architecture.md §5.3-§5.4, option C).

A policy row on crystal_fd's chart is s in [-1, 1]: the fraction u = (s + 1) / 2 of the coordinate's interval given the
earlier coordinates of its block (energies/fd_coordinates.py). The conditioner sees s; this module gives it, per row,
what s means physically:

- ``types``: the row's coordinate type, an index into FD_TYPES (the chart's coordinate name). The same name means
  the same quantity in every space group, so the conditioner's slot and value embeddings are shared across groups.
- ``__call__(s, radius, volume)`` -> [B, d, k], row i's block a function of rows < i only (the conditioner adds it
  to the slot that predicts row i):
    [lo_i, hi_i]        the physical interval of row i given rows < i (cell and orientation blocks: chart.bounds;
                        centroid block: PolytopeARChart.t_bounds, the fractional-coordinate interval, not u's [0, 1]);
    [y_{i-1}]           the previous row's physical value (zero at i = 0);
    [sin, cos]          of 2 pi u_{i-1} when the previous row is a circle, else zero.
  Lengths (log_a, log_b, log_c) are log(L / r_mol), r_mol the molecule radius; every other coordinate is in the
  chart's own units (ratios, cosine factors, fractional coordinates, radians).
- with ``prefix=True``, the prefix physics features of the blocks completed before row i (zero, flag 0, on rows of
  the block itself and before it):
    after the cell block (CELL_FEATS):  log(V_cell / (Z V_mol)) (= -log packing coefficient), log(d_100 / r_mol),
        log(d_010 / r_mol), log(d_001 / r_mol) (interplanar spacings from the reciprocal metric), flag;
    after the centroid block (POS_FEATS): over the images of the molecule's centroid under every non-identity
        operation of the group (minimum image over the 27 neighbouring cells, distances over r_mol): a soft minimum
        of the distance, a soft count within 2 r_mol, the soft minimum per operation class (inversion, other proper
        operations, other improper operations; translations count only in the first two), the class's presence (0/1),
        flag.
  Z is the number of operations of the conventional cell (centring included), V_mol the molecule volume
  (MolData.mol_volume). Soft, so the features are differentiable and a draw's rounding cannot switch them.

Computed in float64 on the CPU from the clamped rows, as energies/molecular_crystal.py::fd_rows_to_params clamps them,
so the density pass and the cached decode see the same numbers for the same float32 rows. Differentiable in s."""
import math

import torch

#: every coordinate name the FD charts of the Day groups use (energies/fd_coordinates.py::FDCoordinates.names);
#: a chart naming anything else is refused, so a new name is added here rather than silently aliased
FD_TYPES = ('log_a', 'log_b', 'log_c', 'r1', 'r2', 's', 's_beta', 's_gamma', 'v',
            't_x', 't_y', 't_z', 't_x+y', 't_x-y',
            'theta', 'phi', 'r', 'r_signed')
LENGTHS = ('log_a', 'log_b', 'log_c')
K_EXTRA = 5
CELL_FEATS = ('log_vcell_over_zvmol', 'log_d100', 'log_d010', 'log_d001', 'cell_done')
POS_FEATS = ('image_softmin', 'images_within_2r', 'inversion_softmin', 'proper_softmin', 'improper_softmin',
             'has_inversion', 'has_proper', 'has_improper', 'position_done')
SOFT_TAU = 0.05     # soft minimum temperature, in units of r_mol
COUNT_WIDTH = 0.1   # soft count's sigmoid width, in units of r_mol
F64 = torch.float64


def _op_classes(sg):
    """(W [n, 3, 3], w [n, 3], class [n]) of the group's non-identity operations, float64; class 0 inversion
    (W = -I), 1 other proper (det +1, translations included), 2 other improper (det -1)."""
    from mxtaltools.constants.space_group_info import SYM_OPS
    W, w, cls = [], [], []
    eye = torch.eye(3, dtype=F64)
    for op in SYM_OPS[int(sg)]:
        A = torch.as_tensor(op, dtype=F64)
        M, m = A[:3, :3], A[:3, 3] - torch.floor(A[:3, 3])
        if torch.equal(M, eye) and not bool(m.abs().gt(1e-9).any()):
            continue
        W.append(M)
        w.append(m)
        cls.append(0 if torch.equal(M, -eye) else (1 if float(torch.det(M)) > 0 else 2))
    if not W:
        return torch.zeros(0, 3, 3, dtype=F64), torch.zeros(0, 3, dtype=F64), torch.zeros(0, dtype=torch.long)
    return torch.stack(W), torch.stack(w), torch.tensor(cls)


def _softmin(x, mask, tau=SOFT_TAU):
    """Soft minimum over the last axis of x where mask (zero where nothing is masked in)."""
    big = torch.where(mask, -x / tau, torch.full_like(x, -float('inf')))
    any_ = mask.any(-1)
    out = -tau * torch.logsumexp(torch.where(any_[..., None], big, torch.zeros_like(big)), -1)
    return torch.where(any_, out, torch.zeros_like(out))


class FDRowDescriber:
    """The typed description of one space group's FD rows (module docstring). No parameters: the conditioner's
    embeddings read it."""

    def __init__(self, sg: int, prefix: bool = False):
        from energies.fd_coordinates import FDCoordinates
        from mxtaltools.constants.space_group_info import SYM_OPS
        from mxtaltools.crystal_fd.charts import CellChart, PolytopeARChart
        self.sg = int(sg)
        layout = FDCoordinates(self.sg, 3.0)
        unknown = [n for n in layout.names if n not in FD_TYPES]
        if unknown:
            raise NotImplementedError(f'sg {self.sg}: FD coordinates {unknown} have no type in FD_TYPES')
        self.names = list(layout.names)
        self.dim = len(self.names)
        self.types = [FD_TYPES.index(n) for n in self.names]
        self.wrap_mask = list(layout.wrap_mask)
        self.is_length = torch.tensor([n in LENGTHS for n in self.names])
        self.is_circle = torch.tensor(self.wrap_mask)
        self.prefix = bool(prefix)
        self.k = K_EXTRA + (len(CELL_FEATS) + len(POS_FEATS) if self.prefix else 0)
        # the first row after the cell block, and after the centroid block (the chart's blocks: cell, centroid,
        # orientation; a group with no centroid coordinates has a centroid block of no rows)
        ends = {}
        for off, blk in zip(layout.chart.offsets, layout.chart.blocks):
            if isinstance(blk, CellChart):
                ends['cell'] = off + blk.ndim
            elif isinstance(blk, PolytopeARChart):
                ends['t'] = off + blk.ndim
        if set(ends) != {'cell', 't'}:
            raise NotImplementedError(f'sg {self.sg}: the chart has no cell or centroid block ({sorted(ends)})')
        self.cell_end, self.pos_end = ends['cell'], ends['t']
        self.z = len(SYM_OPS[self.sg])
        self.op_W, self.op_w, self.op_cls = _op_classes(self.sg)

    def __call__(self, s: torch.Tensor, radius: torch.Tensor, volume=None) -> torch.Tensor:
        """[B, d, k] (module docstring) of rows s [B, d] at molecule radii radius [B] (Angstrom) and, with prefix
        features, molecule volumes volume [B] (cubic Angstrom)."""
        from energies.fd_coordinates import FDCoordinates
        from energies.molecular_crystal import FD_OPEN_EPS
        from mxtaltools.crystal_fd.charts import CellChart, PolytopeARChart
        if s.ndim != 2 or s.shape[1] != self.dim:
            raise ValueError(f'sg {self.sg}: expected rows [B, {self.dim}], got {tuple(s.shape)}')
        b = s.shape[0]
        r = torch.as_tensor(radius, dtype=F64).reshape(-1).cpu()
        if r.numel() != b:
            raise ValueError(f'{r.numel()} radii for {b} rows')
        u = ((s.to(F64).cpu().clamp(-1 + FD_OPEN_EPS, 1 - FD_OPEN_EPS)) + 1) / 2
        chart = FDCoordinates(self.sg, r).chart
        lo, hi, y = [], [], []
        cell = t = None
        for off, blk in zip(chart.offsets, chart.blocks):
            prev = []
            for i in range(blk.ndim):
                pre = torch.stack(prev, 1) if prev else u.new_zeros(b, 0)
                if isinstance(blk, PolytopeARChart):
                    a, z = blk.t_bounds(pre, i)        # the fractional interval, over the earlier fractional coordinates
                else:
                    a, z = blk.bounds(pre, i)
                prev.append(a + u[:, off + i] * (z - a))
                lo.append(a)
                hi.append(z)
            y.extend(prev)
            if self.prefix and isinstance(blk, CellChart):
                cell = blk.map(torch.stack(prev, 1))[0]                   # [B, 6] lengths, angles
            elif self.prefix and isinstance(blk, PolytopeARChart):
                t = blk.map(u[:, off:off + blk.ndim])[0]                 # [B, 3] fractional, free axes at the gauge
        lo, hi, y = torch.stack(lo, 1), torch.stack(hi, 1), torch.stack(y, 1)
        shift = torch.where(self.is_length, torch.log(r)[:, None], torch.zeros_like(lo))
        lo, hi, y = lo - shift, hi - shift, y - shift
        ang = 2 * math.pi * u
        circ = self.is_circle.expand(b, -1)
        sin = torch.where(circ, torch.sin(ang), torch.zeros_like(ang))
        cos = torch.where(circ, torch.cos(ang), torch.zeros_like(ang))
        prev_feats = torch.stack([y, sin, cos], -1)                                  # [B, d, 3] of row i
        prev_feats = torch.cat([prev_feats.new_zeros(b, 1, 3), prev_feats[:, :-1]], 1)   # shifted: row i - 1's
        parts = [torch.stack([lo, hi], -1), prev_feats]
        if self.prefix:
            if volume is None:
                raise ValueError('prefix features need the molecule volumes (MolData.mol_volume)')
            vol = torch.as_tensor(volume, dtype=F64).reshape(-1).cpu()
            rows = torch.arange(self.dim)
            cf = self._cell_features(cell, r, vol)                                   # [B, n_cell]
            pf = self._position_features(cell, t, r)                                 # [B, n_pos]
            parts.append(torch.where((rows >= self.cell_end)[None, :, None], cf[:, None], torch.zeros_like(cf[:, None])))
            parts.append(torch.where((rows >= self.pos_end)[None, :, None], pf[:, None], torch.zeros_like(pf[:, None])))
        out = torch.cat([p.expand(b, self.dim, -1) for p in parts], -1)
        return out.to(device=s.device, dtype=s.dtype)

    def _cell_features(self, cell, r, vol):
        from mxtaltools.crystal_fd.ops import cell_matrices, metric_tensor
        v_cell = torch.linalg.det(cell_matrices(cell)).abs()
        g_inv = torch.linalg.inv(metric_tensor(cell))
        d = 1.0 / torch.sqrt(torch.diagonal(g_inv, dim1=-2, dim2=-1))              # [B, 3] d_100, d_010, d_001
        return torch.cat([torch.log(v_cell / (self.z * vol))[:, None], torch.log(d / r[:, None]),
                          torch.ones_like(r)[:, None]], -1)

    def _position_features(self, cell, t, r):
        from mxtaltools.crystal_fd.ops import cell_matrices
        b = cell.shape[0]
        n_ops = self.op_W.shape[0]
        flags = torch.stack([(self.op_cls == k).any() for k in range(3)]).to(F64).expand(b, 3)
        if n_ops == 0:
            return torch.cat([torch.zeros(b, 5, dtype=F64), flags, torch.ones(b, 1, dtype=F64)], -1)
        T = cell_matrices(cell)                                                      # [B, 3, 3], cart = T frac
        img = torch.einsum('kij,bj->bki', self.op_W, t) + self.op_w                  # [B, n_ops, 3]
        delta = img - t[:, None]
        delta = delta - torch.round(delta)
        shifts = torch.stack(torch.meshgrid(*(torch.arange(-1, 2, dtype=F64),) * 3, indexing='ij'), -1).reshape(-1, 3)
        cart = torch.einsum('bij,bkmj->bkmi', T, delta[:, :, None] + shifts)          # [B, n_ops, 27, 3]
        dist = cart.norm(dim=-1).amin(-1) / r[:, None]                                # [B, n_ops] over r_mol
        every = torch.ones_like(dist, dtype=torch.bool)
        out = [_softmin(dist, every), torch.sigmoid((2.0 - dist) / COUNT_WIDTH).sum(-1)]
        for k in range(3):
            out.append(_softmin(dist, (self.op_cls == k).expand(b, -1)))
        return torch.cat([torch.stack(out, -1), flags, torch.ones(b, 1, dtype=F64)], -1)
