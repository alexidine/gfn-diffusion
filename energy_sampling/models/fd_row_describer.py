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
    [y_{i-1}]           the previous row's physical value (zero at i = 0, and when the previous row is a circle);
    [sin, cos]          of 2 pi u_{i-1} when the previous row is a circle, else zero. A circle row is given by these
                        alone: its raw value jumps at the seam, where they are continuous.
  Physical units: a cell row that sets a length (log_a, log_b, log_c, and the triclinic ratios r2 = b/c and r1 = a/b,
  which set b and a) is log(L / r_mol), r_mol the molecule radius; a cell row that sets an angle (the triclinic
  s_beta, s_gamma and v, which set cos beta, cos gamma and cos alpha; the monoclinic s, which sets cos beta) is that
  cosine, its interval mapped through the same (monotone, given the earlier rows) map; centroid rows are fractional
  coordinates and orientation rows radians.
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

Computed in float64 on the rows' device from the clamped rows, as energies/molecular_crystal.py::fd_rows_to_params
clamps them, so the density pass and the cached decode see the same numbers for the same float32 rows.
Differentiable in s."""
import math

import torch

#: every coordinate name the FD charts of the Day groups use (energies/fd_coordinates.py::FDCoordinates.names);
#: a chart naming anything else is refused, so a new name is added here rather than silently aliased
FD_TYPES = ('log_a', 'log_b', 'log_c', 'r1', 'r2', 's', 's_beta', 's_gamma', 'v',
            't_x', 't_y', 't_z', 't_x+y', 't_x-y',
            'theta', 'phi', 'r', 'r_signed')
LENGTHS = ('log_a', 'log_b', 'log_c', 'r1', 'r2')   # rows whose physical value is a log length
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


def _cell_row_map(blk, ys, i):
    """The map taking cell row i's chart value to its physical value (log Angstrom or a cosine; CellChart.map's
    formulas), given the block's earlier chart values ys (a list of [B]), or None for the identity. A function of rows
    < i only, monotone in its argument (a decreasing map swaps an interval's ends)."""
    if blk.family == 'triclinic':      # rows log_c, r2, r1, s_beta, s_gamma, v
        if i == 0:
            return None
        log_c = ys[0]
        if i == 1:
            return lambda x: log_c + torch.log(x.clamp_min(1e-12))                      # r2 -> log b
        r2 = ys[1]
        if i == 2:
            log_b = log_c + torch.log(r2)
            return lambda x: log_b + torch.log(x.clamp_min(1e-12))                      # r1 -> log a
        r1 = ys[2]
        if i == 3:
            return lambda x: x * r1 * r2 / 2                                            # s_beta -> cos beta
        if i == 4:
            return lambda x: x * r1 / 2                                                 # s_gamma -> cos gamma
        lam = blk.lam(r1, ys[3], ys[4])
        return lambda x: (lam + (1 - lam) * x) * r2 / 2                                 # v -> cos alpha
    if blk.family == 'monoclinic' and i == 3:  # rows log_c, log_a, log_b, s
        m = blk._mono_bound(torch.exp(ys[1]), torch.exp(ys[0]))
        return lambda x: -x * m                                                         # s -> cos beta (decreasing)
    return None                                                                         # log lengths already


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
        if s.ndim != 2 or s.shape[1] != self.dim:
            raise ValueError(f'sg {self.sg}: expected rows [B, {self.dim}], got {tuple(s.shape)}')
        st = self.stepper(radius, volume, s.shape[0], s.device)
        return torch.stack([st.row(s, i) for i in range(self.dim)], 1).to(dtype=s.dtype)

    def stepper(self, radius, volume=None, b=None, device=None) -> 'RowStepper':
        """A RowStepper for b rows at these radii and volumes: row i's features one row at a time, each row's chart
        work done once, so a decode that fixes one row per step pays for one row per step. __call__ is a stepper
        run over every row, so the two give the same numbers."""
        return RowStepper(self, radius, volume, b, device)

    def _cell_features(self, cell, r, vol):
        from mxtaltools.crystal_fd.ops import cell_matrices, metric_tensor
        v_cell = torch.linalg.det(cell_matrices(cell)).abs()
        g_inv = torch.linalg.inv(metric_tensor(cell))
        d = 1.0 / torch.sqrt(torch.diagonal(g_inv, dim1=-2, dim2=-1))              # [B, 3] d_100, d_010, d_001
        return torch.cat([torch.log(v_cell / (self.z * vol))[:, None], torch.log(d / r[:, None]),
                          torch.ones_like(r)[:, None]], -1)

    def _position_features(self, cell, t, r):
        from mxtaltools.crystal_fd.ops import cell_matrices
        b, dev = cell.shape[0], cell.device
        n_ops = self.op_W.shape[0]
        W, w, cls = self.op_W.to(dev), self.op_w.to(dev), self.op_cls.to(dev)
        flags = torch.stack([(cls == k).any() for k in range(3)]).to(F64).expand(b, 3)
        if n_ops == 0:
            return torch.cat([cell.new_zeros(b, 5), flags, cell.new_ones(b, 1)], -1)
        T = cell_matrices(cell)                                                      # [B, 3, 3], cart = T frac
        img = torch.einsum('kij,bj->bki', W, t) + w                                  # [B, n_ops, 3]
        delta = img - t[:, None]
        delta = delta - torch.round(delta)
        shifts = torch.stack(torch.meshgrid(*(torch.arange(-1, 2, dtype=F64, device=dev),) * 3, indexing='ij'),
                             -1).reshape(-1, 3)
        cart = torch.einsum('bij,bkmj->bkmi', T, delta[:, :, None] + shifts)          # [B, n_ops, 27, 3]
        dist = cart.norm(dim=-1).amin(-1) / r[:, None]                                # [B, n_ops] over r_mol
        every = torch.ones_like(dist, dtype=torch.bool)
        out = [_softmin(dist, every), torch.sigmoid((2.0 - dist) / COUNT_WIDTH).sum(-1)]
        for k in range(3):
            out.append(_softmin(dist, (cls == k).expand(b, -1)))
        return torch.cat([torch.stack(out, -1), flags, cell.new_ones(b, 1)], -1)


class RowStepper:
    """Row i's features [B, k] of FDRowDescriber (module docstring), rows in the chart's order. ``row(s, i)`` reads
    rows < i of s [B, d] (rows from i on may hold anything) and fixes each of them once: its chart value from the
    interval found at its own step, its physical value, and the prefix features when a block completes. Rows are
    asked for in increasing order; a row already fixed is not re-read, so s's rows < i must not change between
    calls (in a decode they do not)."""

    def __init__(self, desc: FDRowDescriber, radius, volume, b, device):
        from energies.fd_coordinates import ORIENTATION, length_box
        from mxtaltools.crystal_fd.charts import CellChart, CrystalChart, PolytopeARChart
        self.desc = desc
        r = torch.as_tensor(radius)
        dev = r.device if device is None else device
        r = r.to(device=dev, dtype=F64).reshape(-1)
        if b is not None and r.numel() != b:
            raise ValueError(f'{r.numel()} radii for {b} rows')
        self.r, self.b, self.dev = r, r.numel(), dev
        self.log_r = torch.log(r)
        self.vol = None
        if desc.prefix:
            if volume is None:
                raise ValueError('prefix features need the molecule volumes (MolData.mol_volume)')
            self.vol = torch.as_tensor(volume).to(device=dev, dtype=F64).reshape(-1)
        # FDCoordinates(sg, r).chart, built on the rows' device (the centroid block's coefficient tables included)
        chart = CrystalChart.for_length_box(desc.sg, length_box(desc.sg, r), ORIENTATION, device=dev)
        self.where = []                                   # row -> (block, offset, index in block)
        for off, blk in zip(chart.offsets, chart.blocks):
            self.where += [(blk, off, i) for i in range(blk.ndim)]
            if isinstance(blk, PolytopeARChart):
                self.pos_blk, self.pos_off = blk, off
        self._cell_type, self._pos_type = CellChart, PolytopeARChart
        self.bounds = {}                                  # row -> physical (lo, hi), [B] each
        self.chart_lo_hi = {}                             # row -> chart (lo, hi)
        self.ys = {}                                      # row -> chart value
        self.prev = {}                                    # row -> (physical value, sin, cos), as row + 1 reads them
        self.cell = self.cell_f = self.pos_f = None
        self.n_fixed = 0

    def _u(self, s, row):
        from energies.molecular_crystal import FD_OPEN_EPS
        return (s[:, row].to(device=self.dev, dtype=F64).clamp(-1 + FD_OPEN_EPS, 1 - FD_OPEN_EPS) + 1) / 2

    def _block_ys(self, row):
        blk, off, i = self.where[row]
        return [self.ys[off + k] for k in range(i)]

    def _interval(self, row):
        """Row's chart interval given the earlier rows of its block, and its physical interval."""
        if row in self.bounds:
            return
        blk, off, i = self.where[row]
        ys = self._block_ys(row)
        pre = torch.stack(ys, 1) if ys else self.r.new_zeros(self.b, 0)
        if isinstance(blk, self._pos_type):
            a, z = blk.t_bounds(pre, i)        # the fractional interval, over the earlier fractional coordinates
        else:
            a, z = blk.bounds(pre, i)
        self.chart_lo_hi[row] = (a, z)
        f = _cell_row_map(blk, ys, i) if isinstance(blk, self._cell_type) else None
        if f is not None:
            fa, fz = f(a), f(z)
            a, z = torch.minimum(fa, fz), torch.maximum(fa, fz)
        if bool(self.desc.is_length[row]):
            a, z = a - self.log_r, z - self.log_r
        self.bounds[row] = (a, z)

    def _position_done(self, s):
        t = self.pos_blk.map(torch.stack([self._u(s, self.pos_off + k) for k in range(self.pos_blk.ndim)], 1)
                             if self.pos_blk.ndim else self.r.new_zeros(self.b, 0))[0]   # [B, 3] fractional
        self.pos_f = self.desc._position_features(self.cell, t, self.r)

    def _fix(self, s, row):
        """Row's chart and physical value from s, and the prefix features of a block it completes."""
        self._interval(row)
        blk, off, i = self.where[row]
        u = self._u(s, row)
        a, z = self.chart_lo_hi[row]
        y = a + u * (z - a)
        ys = self._block_ys(row)
        self.ys[row] = y
        if bool(self.desc.is_circle[row]):     # a circle row's value is its sin, cos: no seam jump
            ang = 2 * math.pi * u
            self.prev[row] = (torch.zeros_like(y), torch.sin(ang), torch.cos(ang))
        else:
            f = _cell_row_map(blk, ys, i) if isinstance(blk, self._cell_type) else None
            v = y if f is None else f(y)
            if bool(self.desc.is_length[row]):
                v = v - self.log_r
            self.prev[row] = (v, torch.zeros_like(y), torch.zeros_like(y))
        if self.desc.prefix and i == blk.ndim - 1:
            if isinstance(blk, self._cell_type):
                self.cell = blk.map(torch.stack(ys + [y], 1))[0]                        # [B, 6] lengths, angles
                self.cell_f = self.desc._cell_features(self.cell, self.r, self.vol)
                if self.pos_blk.ndim == 0:      # a group with no centroid rows: its position is fixed with the cell
                    self._position_done(s)
            elif isinstance(blk, self._pos_type):
                self._position_done(s)

    def row(self, s, i):
        """Row i's features [B, k], float64, from rows < i of s."""
        d = self.desc
        if s.shape[0] != self.b:
            raise ValueError(f'{self.b} radii for {s.shape[0]} rows')
        if i < self.n_fixed:
            raise ValueError(f'row {i} asked after rows up to {self.n_fixed - 1} were fixed: rows go in order')
        while self.n_fixed < i:
            self._fix(s, self.n_fixed)
            self.n_fixed += 1
        self._interval(i)
        lo, hi = self.bounds[i]
        prev = self.prev[i - 1] if i else (torch.zeros_like(lo),) * 3
        parts = [torch.stack([lo, hi, *prev], -1)]
        if d.prefix:
            parts.append(self.cell_f if i >= d.cell_end else lo.new_zeros(self.b, len(CELL_FEATS)))
            parts.append(self.pos_f if i >= d.pos_end else lo.new_zeros(self.b, len(POS_FEATS)))
        return torch.cat(parts, -1)
