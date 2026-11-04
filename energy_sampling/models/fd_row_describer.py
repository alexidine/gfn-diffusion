"""Typed FD rows for the AR flow's conditioner (docs/design/conditional_ar_architecture.md §5.3, option C).

A policy row on crystal_fd's chart is s in [-1, 1]: the fraction u = (s + 1) / 2 of the coordinate's interval given the
earlier coordinates of its block (energies/fd_coordinates.py). The conditioner sees s; this module gives it, per row,
what s means physically:

- ``types``: the row's coordinate type, an index into FD_TYPES (the chart's coordinate name). The same name means
  the same quantity in every space group, so the conditioner's slot and value embeddings are shared across groups.
- ``__call__(s, radius)`` -> [B, d, K_EXTRA], column block i a function of rows < i only (the conditioner adds it to
  the slot that predicts row i):
    [lo_i, hi_i]        the physical interval of row i given rows < i (cell and orientation blocks: chart.bounds;
                        centroid block: PolytopeARChart.t_bounds, the fractional-coordinate interval, not u's [0, 1]);
    [y_{i-1}]           the previous row's physical value (zero at i = 0);
    [sin, cos]          of 2 pi u_{i-1} when the previous row is a circle, else zero.
  Lengths (log_a, log_b, log_c) are log(L / r_mol) Angstrom over the molecule radius; every other coordinate is in
  the chart's own units (ratios, cosine factors, fractional coordinates, radians).

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
F64 = torch.float64


class FDRowDescriber:
    """The typed description of one space group's FD rows (module docstring). No parameters: the conditioner's
    embeddings read it."""

    def __init__(self, sg: int):
        from energies.fd_coordinates import FDCoordinates
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

    def __call__(self, s: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """[B, d, K_EXTRA] (module docstring) of rows s [B, d] at molecule radii radius [B] (Angstrom)."""
        from energies.fd_coordinates import FDCoordinates
        from energies.molecular_crystal import FD_OPEN_EPS
        from mxtaltools.crystal_fd.charts import PolytopeARChart
        if s.ndim != 2 or s.shape[1] != self.dim:
            raise ValueError(f'sg {self.sg}: expected rows [B, {self.dim}], got {tuple(s.shape)}')
        b = s.shape[0]
        r = torch.as_tensor(radius, dtype=F64).reshape(-1).cpu()
        if r.numel() != b:
            raise ValueError(f'{r.numel()} radii for {b} rows')
        u = ((s.to(F64).cpu().clamp(-1 + FD_OPEN_EPS, 1 - FD_OPEN_EPS)) + 1) / 2
        chart = FDCoordinates(self.sg, r).chart
        lo, hi, y = [], [], []
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
        lo, hi, y = torch.stack(lo, 1), torch.stack(hi, 1), torch.stack(y, 1)
        shift = torch.where(self.is_length, torch.log(r)[:, None], torch.zeros_like(lo))
        lo, hi, y = lo - shift, hi - shift, y - shift
        ang = 2 * math.pi * u
        circ = self.is_circle.expand(b, -1)
        sin = torch.where(circ, torch.sin(ang), torch.zeros_like(ang))
        cos = torch.where(circ, torch.cos(ang), torch.zeros_like(ang))
        prev_feats = torch.stack([y, sin, cos], -1)                                  # [B, d, 3] of row i
        prev_feats = torch.cat([prev_feats.new_zeros(b, 1, 3), prev_feats[:, :-1]], 1)   # shifted: row i - 1's
        out = torch.cat([torch.stack([lo, hi], -1), prev_feats], -1)
        return out.to(device=s.device, dtype=s.dtype)
