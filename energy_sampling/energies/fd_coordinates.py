"""Fundamental-domain coordinates for a crystal sampler: s in [-1, 1]^d <-> MXtalTools cell parameters, Z' = 1.

One space group, one molecule radius (the trainer's length box at that radius, ``CrystalChart.for_latent_v1``), the
hemisphere orientation chart (sg 6-9: its K cut). The chart is crystal_fd's ``CrystalChart``: every crystal of the
space group (inside the length box) has exactly one point s, and every s is a canonical crystal -- reduced cell,
folded pose, handedness on the +1 sheet. The reduction penalty a latent sampler pays is therefore zero by
construction.

Maps, per row:

    s --(u = (s + 1) / 2)--> u in [0, 1]^d --(charts.from_unit: y_i = lo_i(y_<i) + u_i (hi_i - lo_i))--> y
      --(CrystalChart.forward)--> CrystalDescription --(centres wrapped into [0, 1))--> cell parameters, handedness

``log_reference(s)`` is the log-density, relative to Lebesgue measure on s, of the reference measure on crystals
(``measure.log_reference_density`` with ``cell_measure``: Haar on the orientation, Lebesgue on fractional centres
times Z' log V, and the cell measure): that density on y, plus log |dy/ds| = sum_i log(hi_i - lo_i) - d log 2. A
sampler on s whose log-reward is ``-E / T + log_reference(s)`` targets the Boltzmann density relative to the reference
measure, the currency of a latent sampler whose reward carries ``MolecularCrystal.compute_jacobian`` -- up to the
latent's own multiplicity (images of one crystal in the latent box) and a constant per space group.

Rows: ``wrap_mask`` marks circle coordinates (period 2 in s: the hemisphere azimuth, and centroid coordinates glued
by a pure translation); every other coordinate is an interval, bounded on [-1, 1] (a flow's bounded rows). There are
no dead rows: free centroid axes (sg 1, polar groups) are not coordinates.

``from_params`` takes stored rows (any description of each crystal) to s: ``canon.canonicalize`` (sheet +1), then
``CrystalChart.inverse``, then the unit map's inverse u_i = (y_i - lo_i) / (hi_i - lo_i). A row that does not
canonicalise (status other than ``OK_STATUS``), or whose canonical cell lies outside the length box, is reported
in ``ok`` and its s is NaN.
"""
import math
from typing import NamedTuple

import torch

from mxtaltools.crystal_fd import canon, charts, measure
from mxtaltools.crystal_fd.description import CrystalDescription

F64 = torch.float64
ORIENTATION = 'hemisphere'
#: an s row may sit this far outside [-1, 1] after a float32 round trip and still be taken as on the face
S_TOL = 1e-6
#: canonicalize statuses taken as charted: 'special_metric' rows (a cell with extra automorphisms) are folded on the
#: sheet too; their pose depends on the input basis, but the point is in the domain
OK_STATUS = ('canonical', 'special_metric')


class FDRows(NamedTuple):
    """``FDCoordinates.from_params`` output: s [B, d] float64 (NaN where not ok), ok [B] bool, status [B] str."""
    s: torch.Tensor
    ok: torch.Tensor
    status: list


class FDCoordinates:
    """s in [-1, 1]^d <-> Z' = 1 crystals of one space group at one molecule radius (module docstring).

    Parameters
    ----------
    sg : int
    radius : float
        The molecule radius (Angstrom) that sets the trainer's length box (``charts.latent_v1_length_box``).
    cell_measure : str
        ``measure.CELL_MEASURES``; ``'latent_v1_lebesgue'`` (default) is the latent trainer's own cell measure, flat
        in its cell rows (owner decision O4).
    """

    def __init__(self, sg: int, radius: float, cell_measure: str = 'latent_v1_lebesgue'):
        self.sg = int(sg)
        self.radius = float(radius)
        self.cell_measure = cell_measure
        self.chart = charts.CrystalChart.for_latent_v1(self.sg, torch.tensor([self.radius], dtype=F64), ORIENTATION)
        self.dim = self.chart.ndim
        self.wrap_mask = [d.kind == 'circle' for d in self.chart.dims]
        self.names = [d.name for d in self.chart.dims]
        bad = [d.name for d in self.chart.dims if d.kind not in ('interval', 'circle')]
        if bad:
            raise NotImplementedError(f'sg {self.sg}: chart coordinates of kind other than interval/circle: {bad}')
        self._check_circle_periods()

    def _check_circle_periods(self):
        """A circle coordinate must span exactly one period between its bounds, so that s = -1 and s = 1 are one
        point; checked on a few unit draws (the bounds of a circle do not depend on the prefix)."""
        g = torch.Generator().manual_seed(0)
        u = torch.rand(64, self.dim, generator=g, dtype=F64)
        cols = []
        for i in range(self.dim):
            prefix = torch.stack(cols, 1) if cols else u.new_zeros(64, 0)
            lo, hi = self.chart.bounds(prefix, i)
            d = self.chart.dims[i]
            if d.kind == 'circle' and not torch.allclose(hi - lo, torch.full_like(lo, float(d.period)), atol=1e-12):
                raise ValueError(f'sg {self.sg}: circle coordinate {d.name} spans {float((hi - lo).min())}-'
                                 f'{float((hi - lo).max())}, not its period {d.period}')
            cols.append(lo + u[:, i] * (hi - lo))

    # ------------------------------------------------------------------
    def to_description(self, s: torch.Tensor):
        """(CrystalDescription [B] with centres wrapped into [0, 1), log_reference [B]) of s [B, d] in [-1, 1]."""
        s = torch.as_tensor(s, dtype=F64)
        if s.ndim != 2 or s.shape[1] != self.dim:
            raise ValueError(f'expected s of shape [B, {self.dim}], got {tuple(s.shape)}')
        u = ((s + 1) / 2).clamp(0.0, 1.0)
        y, log_volume = charts.from_unit(self.chart, u)
        desc, _ = self.chart.forward(y)
        log_ref_y = measure.log_reference_density(desc, self.chart, self.cell_measure)
        log_ref = log_ref_y + log_volume - self.dim * math.log(2.0)
        t = desc.t - torch.floor(desc.t)
        desc = desc.replace(t=torch.where(t >= 1, t - 1, t))
        return desc, log_ref

    def to_params(self, s: torch.Tensor):
        """([B, 12] float32 cell parameters in MolCrystalData's layout, [B] float32 handedness, [B] float64
        log_reference) of s [B, d]."""
        desc, log_ref = self.to_description(s)
        L, A, t, rv, h = desc.crystal_params()
        return torch.cat([L, A, t, rv], 1).float(), h.reshape(-1).float(), log_ref

    def from_params(self, params: torch.Tensor, handedness: torch.Tensor) -> FDRows:
        """s of stored rows ([B, 12] cell parameters, [B] handedness): canonicalised first (module docstring)."""
        params = torch.as_tensor(params)
        B = params.shape[0]
        p = params.to(F64)
        desc = CrystalDescription.from_crystal_params(self.sg, p[:, :3], p[:, 3:6], p[:, 6:9], p[:, 9:12],
                                                      torch.as_tensor(handedness).to(F64).reshape(-1, 1))
        c = canon.canonicalize(desc)
        status = c.names()
        good = torch.tensor([st in OK_STATUS for st in status], dtype=torch.bool)
        s = torch.full((B, self.dim), float('nan'), dtype=F64)
        if bool(good.any()):
            gi = torch.nonzero(good).flatten()
            y, _ = self.chart.inverse(c.desc[gi])
            u = self._unit_of(y)
            inside = ((u >= -S_TOL) & (u <= 1 + S_TOL)).all(1) & torch.isfinite(u).all(1)
            for k, j in enumerate(gi.tolist()):
                if not bool(inside[k]):
                    status[j] = 'outside_length_box'
                    good[j] = False
            s[gi[inside]] = (2 * u[inside].clamp(0.0, 1.0) - 1)
        return FDRows(s=s, ok=good, status=status)

    def _unit_of(self, y: torch.Tensor) -> torch.Tensor:
        """The inverse of ``charts.from_unit``: u_i = (y_i - lo_i(y_<i)) / (hi_i - lo_i(y_<i)); circle coordinates
        are first wrapped into their [lo, hi) period."""
        u = torch.empty_like(y)
        for i in range(self.dim):
            lo, hi = self.chart.bounds(y[:, :i], i)
            yi = y[:, i]
            d = self.chart.dims[i]
            if d.kind == 'circle':
                yi = lo + torch.remainder(yi - lo, float(d.period))
            u[:, i] = (yi - lo) / (hi - lo)
        return u
