"""Fundamental-domain coordinates for a crystal sampler: s in [-1, 1]^d <-> MXtalTools cell parameters, Z' = 1.

One space group, one molecule radius (the length box ``length_box`` at that radius, ``CrystalChart.for_length_box``),
the hemisphere orientation chart (sg 6-9: its K cut). The chart is crystal_fd's ``CrystalChart``: every crystal of the
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
from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.crystal_fd.description import CrystalDescription

F64 = torch.float64
ORIENTATION = 'hemisphere'
#: an s row may sit this far outside [-1, 1] after a float32 round trip and still be taken as on the face
S_TOL = 1e-6
#: canonicalize statuses taken as charted: 'special_metric' rows (a cell with extra automorphisms) are folded on the
#: sheet too; their pose depends on the input basis, but the point is in the domain
OK_STATUS = ('canonical', 'special_metric')
#: the length box: every canonical cell length in [LENGTH_LO, 2 Z (r + LENGTH_CONTACT)] Angstrom (``length_box``)
LENGTH_LO = 2.5
LENGTH_CONTACT = 3.0


def length_box(sg: int, radius):
    """(lo, hi) [B, 3] float64 Angstrom, the same for the three cell lengths: lo = LENGTH_LO, hi = 2 Z (r +
    LENGTH_CONTACT), Z = len(SYM_OPS[sg]) molecules in the conventional cell at Z' = 1 (centring included), r the
    molecule radius [B] (``MolData.radius``). Owner decision 2026-11-02. Measured then: every canonical cell of the
    CSD's Z' = 1 rows (csd_fd_v5, 394,517) and of Day's FIT+DMA landscapes (<= 30 lowest per molecule within
    10 kJ/mol, 29,947) lies inside, the longest axis at most 0.97 of hi, the shortest 2.96 A."""
    r = torch.as_tensor(radius, dtype=F64).reshape(-1)
    hi = (2 * len(SYM_OPS[int(sg)]) * (r + LENGTH_CONTACT))[:, None].expand(-1, 3).clone()
    return torch.full_like(hi, LENGTH_LO), hi


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
    radius : float or [B] tensor
        The molecule radius (Angstrom) that sets the length box (``length_box``). A [B]
        tensor charts B rows of different molecules at once (one length box per row): every method is then called
        with exactly those B rows, in that order.
    cell_measure : str
        ``measure.CELL_MEASURES``; ``'latent_v1_lebesgue'`` (default) is the latent trainer's own cell measure, flat
        in its cell rows (owner decision O4).
    """

    def __init__(self, sg: int, radius, cell_measure: str = 'latent_v1_lebesgue'):
        self.sg = int(sg)
        r = torch.as_tensor(radius, dtype=F64).reshape(-1)
        self.rows = None if r.numel() == 1 else int(r.numel())   # None: one radius for any number of rows
        self.radius = float(r[0]) if self.rows is None else r
        self.cell_measure = cell_measure
        self.chart = charts.CrystalChart.for_length_box(self.sg, length_box(self.sg, r), ORIENTATION)
        self.dim = self.chart.ndim
        self.wrap_mask = [d.kind == 'circle' for d in self.chart.dims]
        self.names = [d.name for d in self.chart.dims]
        bad = [d.name for d in self.chart.dims if d.kind not in ('interval', 'circle')]
        if bad:
            raise NotImplementedError(f'sg {self.sg}: chart coordinates of kind other than interval/circle: {bad}')
        if self.rows is None:   # the check draws its own rows; per-row charts share the scalar chart's coordinates
            self._check_circle_periods()

    def _check_rows(self, n):
        if self.rows is not None and int(n) != self.rows:
            raise ValueError(f'this chart holds {self.rows} per-row length boxes, called with {int(n)} rows')

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
        self._check_rows(s.shape[0])
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
        self._check_rows(B)
        p = params.to(F64)
        desc = CrystalDescription.from_crystal_params(self.sg, p[:, :3], p[:, 3:6], p[:, 6:9], p[:, 9:12],
                                                      torch.as_tensor(handedness).to(F64).reshape(-1, 1))
        c = canon.canonicalize(desc)
        status = c.names()
        good = torch.tensor([st in OK_STATUS for st in status], dtype=torch.bool)
        s = torch.full((B, self.dim), float('nan'), dtype=F64)
        if bool(good.any()):
            gi = torch.nonzero(good).flatten()
            # per-row length boxes: the chart of the rows that canonicalised, in their order
            chart = self.chart if self.rows is None else charts.CrystalChart.for_length_box(
                self.sg, length_box(self.sg, self.radius[gi]), ORIENTATION)
            y, _ = chart.inverse(c.desc[gi])
            u = self._unit_of(y, chart)
            inside = ((u >= -S_TOL) & (u <= 1 + S_TOL)).all(1) & torch.isfinite(u).all(1)
            for k, j in enumerate(gi.tolist()):
                if not bool(inside[k]):
                    status[j] = 'outside_length_box'
                    good[j] = False
            s[gi[inside]] = (2 * u[inside].clamp(0.0, 1.0) - 1)
        return FDRows(s=s, ok=good, status=status)

    def reframe(self, s: torch.Tensor, R0: torch.Tensor) -> FDRows:
        """The chart rows of the crystals of s [B, d] with the molecule presented as R0 [B, 3, 3] (proper rotations)
        applied to its standard-frame coordinates: the builder rotates the recentred molecule by the orientation R,
        so R becomes R R0^T and no other parameter changes; then canonicalised as from_params. Rows on the +1
        handedness sheet only (every chart row is, after canonicalisation); one off it raises, since a mirrored
        molecule would take M R0 M (M the sheet's mirror) instead of R0."""
        from mxtaltools.common.geometry_utils import rotmat2rotvec_stable, rotvec2rotmat
        params, hand, _ = self.to_params(s)
        if bool((hand != 1).any()):
            raise NotImplementedError("reframe: a chart row off the +1 handedness sheet")
        p = params.to(F64)
        R = rotvec2rotmat(p[:, 9:12])
        p[:, 9:12] = rotmat2rotvec_stable(R @ torch.as_tensor(R0).to(F64).transpose(1, 2))
        return self.from_params(p, hand)

    def block_widths(self) -> list:
        """[cell, centroid, orientation] row counts of the chart (every group's chart has these three blocks, in
        this order; a group with no free centroid coordinate has a centroid block of 0 rows)."""
        kinds = [type(b).__name__ for b in self.chart.blocks]
        if len(kinds) != 3 or not (kinds[0] == 'CellChart' and kinds[1] == 'PolytopeARChart'
                                   and isinstance(self.chart.blocks[2], charts.OrientationChart)):
            raise NotImplementedError(f'sg {self.sg}: chart blocks {kinds}, not cell, centroid, orientation')
        return [int(b.ndim) for b in self.chart.blocks]

    def _unit_of(self, y: torch.Tensor, chart=None) -> torch.Tensor:
        """The inverse of ``charts.from_unit``: u_i = (y_i - lo_i(y_<i)) / (hi_i - lo_i(y_<i)); circle coordinates
        are first wrapped into their [lo, hi) period."""
        chart = self.chart if chart is None else chart
        u = torch.empty_like(y)
        for i in range(self.dim):
            lo, hi = chart.bounds(y[:, :i], i)
            yi = y[:, i]
            d = chart.dims[i]
            if d.kind == 'circle':
                yi = lo + torch.remainder(yi - lo, float(d.period))
            u[:, i] = (yi - lo) / (hi - lo)
        return u


#: the blocks of every group's chart, in chart order (FDCoordinates.block_widths)
FD_BLOCKS = ('cell', 'centroid', 'orientation')
PAD = 'pad'


#: groups whose orientation block is the K-cut hemisphere (crystal_fd KCutHemisphereChart): a frame change there can
#: leave the cut, and folding it back is not a change of the orientation rows alone, so rotate_orientation refuses them
KCUT_GROUPS = (6, 7, 8, 9)
#: the hemisphere chart's bounds per orientation coordinate (mxtaltools crystal_fd.charts.OrientationChart.bounds)
HEMISPHERE_BOUNDS = ((0.0, math.pi / 2), (-math.pi, math.pi), (0.0, 2 * math.pi))


def _orientation_map(s: torch.Tensor, left, right, rows):
    """(s' [n, d], log |ds'/ds| [n]) of FD chart rows s with the orientation R of rows (theta, phi, r) mapped to
    left R right (left, right [n, 3, 3] or None for the identity, their determinants' product +1, so the result is a
    rotation), every other row unmoved; the rotation in float64. The map preserves Haar measure, so log |ds'/ds| =
    log rho(y) - log rho(y'), rho the hemisphere chart's Haar density (the unit map's constant scale cancels)."""
    from mxtaltools.crystal_fd import so3
    rows = list(rows)
    lo = torch.tensor([b[0] for b in HEMISPHERE_BOUNDS], dtype=F64, device=s.device)
    hi = torch.tensor([b[1] for b in HEMISPHERE_BOUNDS], dtype=F64, device=s.device)
    y = lo + (s[:, rows].to(F64) + 1) / 2 * (hi - lo)
    R = so3.hemisphere_to_matrix(y)
    if left is not None:
        R = left.to(device=s.device, dtype=F64) @ R
    if right is not None:
        R = R @ right.to(device=s.device, dtype=F64)
    y2 = so3.matrix_to_hemisphere(R)
    y2[:, 1] = torch.remainder(y2[:, 1] - lo[1], 2 * math.pi) + lo[1]          # phi on its circle
    out = s.clone()
    out[:, rows] = (2 * (y2 - lo) / (hi - lo) - 1).to(s.dtype)
    log_jac = so3.haar_log_density('hemisphere', y) - so3.haar_log_density('hemisphere', y2)
    return out, log_jac.to(s.dtype)


def rotate_orientation(s: torch.Tensor, R0: torch.Tensor, rows=(9, 10, 11)):
    """(s' [n, d], log |ds'/ds| [n]) of FD chart rows s [n, d] with the molecule presented as R0 [n, 3, 3] applied to
    its standard-frame coordinates: the orientation R becomes R R0^T and no other row moves, on any group whose
    orientation block is the plain hemisphere chart (frame_rows; FDCoordinates.reframe is the exact, CPU-bound
    reference, which agrees on every such group tested, tests/crystal/test_fd_reframe.py). Torch on s's device and
    dtype; the rotation in float64.
    The map preserves Haar measure, so log |ds'/ds| = log rho(y) - log rho(y'), rho the hemisphere chart's Haar
    density (the unit map's constant scale cancels)."""
    return _orientation_map(s, None, torch.as_tensor(R0).transpose(1, 2), rows)


#: The mirror image of a chart row: the molecule reflected, X -> diag(-1, 1, 1) X (its standard-frame coordinates),
#: and the crystal re-described for it (in Sohncke-free groups the same crystal, the reflected molecule being its
#: inverted copy). Per space group: (the centroid chart rows negated, A, B), the orientation R -> A R B, every other
#: row unmoved. Measured 2026-11-09 against crystal_fd's canonicaliser on the same crystals with the handedness
#: flipped (canon.canonicalize(desc.replace(h=-h), sheet=1)): every row of MIPCAS's P-1 eLJ band set (42,258) and
#: NEHZOR's P2_1/c set (73,073) in description terms, and 4,000 of each in chart terms; tests/crystal/test_fd_mirror.py
#: pins it on random chart points. A group not listed is refused (mirror_rows).
MIRROR_MAPS = {
    2: (('t_x', 't_y', 't_z'), (1.0, 1.0, 1.0), (1.0, -1.0, -1.0)),     # t -> 1/2 - t,  R -> R diag(1, -1, -1)
    14: (('t_y',), (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0)),                 # t_y -> 1/2 - t_y,  R -> diag(1,-1,1) R diag(-1,1,1)
}


def mirror_rows(groups):
    """{sg: (flip [k] long chart-row indices, A [3, 3], B [3, 3])} for a run over ``groups`` (one group's chart rows,
    or the shared slots of several groups' FDSlotLayout): the inputs of mirror_state. Raises NotImplementedError for
    a group without a measured map (MIRROR_MAPS) or a K-cut group (frame_rows)."""
    groups = sorted({int(g) for g in groups})
    frame_rows(groups)   # the orientation rows must be the shared hemisphere chart
    missing = [g for g in groups if g not in MIRROR_MAPS]
    if missing:
        raise NotImplementedError(f"no mirror map for sg {missing} (energies/fd_coordinates.py::MIRROR_MAPS: derive "
                                  f"it against crystal_fd's canonicaliser and pin it in tests/crystal/test_fd_mirror.py)")
    names = ({groups[0]: FDCoordinates(groups[0], 3.0).names} if len(groups) == 1
             else FDSlotLayout(groups).group_names)
    out = {}
    for g in groups:
        flip, A, B = MIRROR_MAPS[g]
        out[g] = (torch.tensor([names[g].index(k) for k in flip], dtype=torch.long),
                  torch.diag(torch.tensor(A, dtype=F64)), torch.diag(torch.tensor(B, dtype=F64)))
    return out


def mirror_state(s: torch.Tensor, sg: torch.Tensor, maps: dict, rows):
    """(s' [n, d], log |ds'/ds| [n]) of chart rows s [n, d] of space groups sg [n] (one per row): each row's mirror
    image (MIRROR_MAPS; maps from mirror_rows, rows the orientation rows of frame_rows). The map is an involution and
    preserves the chart's reference measure on the centroid (a sign change) and Haar measure on the orientation."""
    sg = torch.as_tensor(sg).reshape(-1).to(s.device)
    bad = ~torch.isin(sg, torch.tensor(list(maps), device=s.device, dtype=sg.dtype))
    if bool(bad.any()):
        raise ValueError(f"mirror_state: rows of sg {sorted(set(sg[bad].tolist()))} without a map")
    out = s.clone()
    log_jac = torch.zeros(s.shape[0], dtype=s.dtype, device=s.device)
    for g, (flip, A, B) in maps.items():
        m = sg == int(g)
        if not bool(m.any()):
            continue
        idx = torch.nonzero(m).flatten()
        part = out[idx]
        f = flip.to(s.device)
        part[:, f] = -part[:, f]
        k = idx.numel()
        part, lj = _orientation_map(part, A.to(s.device).expand(k, 3, 3), B.to(s.device).expand(k, 3, 3), rows)
        out[idx] = part
        log_jac[idx] = lj
    return out, log_jac


def frame_rows(groups):
    """The state rows (theta, phi, r) rotate_orientation re-expresses, for a run over `groups`: one group's chart
    indices, or several groups' shared orientation slots (FDSlotLayout). Raises for a K-cut group, or a layout whose
    groups do not share those slots."""
    groups = sorted({int(g) for g in groups})
    bad = [g for g in groups if g in KCUT_GROUPS]
    if bad:
        raise NotImplementedError(f"a frame change on the K-cut orientation chart of sg {bad} is not a change of its "
                                  f"orientation rows alone")
    if len(groups) == 1:
        names = FDCoordinates(groups[0], 3.0).names
        return [names.index(k) for k in ('theta', 'phi', 'r')]
    layout = FDSlotLayout(groups)
    rows = None
    for g in groups:
        names = layout.group_names[g]
        r = [names.index(k) for k in ('theta', 'phi', 'r')]
        if rows is not None and r != rows:
            raise NotImplementedError(f"sg {groups}: the orientation rows sit in different slots")
        rows = r
    return rows


class FDSlotLayout:
    """The trainer state of an FD run over the space groups ``groups`` (Z' = 1), so that one batch can hold rows of
    several groups.

    Slot block k (FD_BLOCKS) is as wide as the widest group's block k. A group's chart rows fill the first slots of
    each block, in chart order; its remaining slots are pads, held at 0 and carrying no density (the policy pins
    them, ``live_mask``). With one group the slots are exactly its chart rows, in order: the state of a one-group run.

    ``wrap_mask``: a slot is a circle iff it is a circle in every group whose row it carries. A circle row modelled as
    an interval on [-1, 1] is still a density on it; an interval row modelled as a circle would be tied at its two
    ends. With one group it is the chart's own ``wrap_mask``.

    Parameters
    ----------
    groups : iterable of int
        The run's space groups (``MolecularCrystal.space_groups``); duplicates and order do not matter.
    radius : float
        Any molecule radius: the slots, widths and circle rows do not depend on it.
    """

    def __init__(self, groups, radius: float = 4.0):
        self.groups = tuple(sorted({int(g) for g in groups}))
        if not self.groups:
            raise ValueError('FDSlotLayout needs at least one space group')
        self.charts = {g: FDCoordinates(g, radius) for g in self.groups}
        widths = {g: fc.block_widths() for g, fc in self.charts.items()}
        self.block_width = [max(w[k] for w in widths.values()) for k in range(len(FD_BLOCKS))]
        self.block_start = [sum(self.block_width[:k]) for k in range(len(FD_BLOCKS))]
        self.dim = sum(self.block_width)
        lut = torch.full((max(self.groups) + 1,), -1, dtype=torch.long)
        lut[torch.tensor(self.groups)] = torch.arange(len(self.groups))
        self._index = lut
        self.slots, names = {}, {}
        live = torch.zeros(len(self.groups), self.dim, dtype=torch.bool)
        wrap = torch.zeros(len(self.groups), self.dim, dtype=torch.bool)
        for j, g in enumerate(self.groups):
            fc = self.charts[g]
            slot = []
            for k, n in enumerate(widths[g]):
                slot += list(range(self.block_start[k], self.block_start[k] + n))
            self.slots[g] = torch.tensor(slot, dtype=torch.long)
            live[j, self.slots[g]] = True
            wrap[j, self.slots[g]] = torch.tensor(fc.wrap_mask, dtype=torch.bool)
            row_names = [PAD] * self.dim
            for i, n in zip(slot, fc.names):
                row_names[i] = n
            names[g] = row_names
        self.group_names = names
        self.live = live                                  # [n_groups, dim]
        self.group_wrap = wrap                            # [n_groups, dim], each group's own circle rows
        self.wrap_mask = [bool(live[:, i].any()) and bool((wrap[:, i] | ~live[:, i]).all()) for i in range(self.dim)]
        #: per slot, the chart names it carries across the groups ('|'-joined); one group: the chart's names
        self.names = ['|'.join(dict.fromkeys(names[g][i] for g in self.groups if names[g][i] != PAD)) or PAD
                      for i in range(self.dim)]
        self.sg = self.groups[0] if len(self.groups) == 1 else None

    def group_index(self, sg) -> torch.Tensor:
        """[B] index into ``groups`` of each row's space group ``sg`` [B]; raises on a group outside the layout."""
        sg = torch.as_tensor(sg).long().reshape(-1).cpu()
        idx = torch.full_like(sg, -1)
        ok = (sg >= 0) & (sg < len(self._index))
        idx[ok] = self._index[sg[ok]]
        if bool((idx < 0).any()):
            bad = sorted(set(sg[idx < 0].tolist()))
            raise ValueError(f'space groups {bad} are not in this FD layout {list(self.groups)}')
        return idx

    def live_mask(self, sg) -> torch.Tensor:
        """[B, dim] bool, the slots that carry a chart row of each row's space group ``sg`` [B] (False: a pad)."""
        return self.live[self.group_index(sg)]

    def pack(self, sg: int, s: torch.Tensor) -> torch.Tensor:
        """[n, dim] state of chart rows s [n, d_sg] of group sg, pads 0."""
        slot = self.slots[int(sg)]
        if s.shape[-1] != len(slot):
            raise ValueError(f'sg {int(sg)}: chart rows are {len(slot)} wide, got {s.shape[-1]}')
        out = s.new_zeros(s.shape[0], self.dim)
        out[:, slot.to(s.device)] = s
        return out

    def unpack(self, sg: int, x: torch.Tensor) -> torch.Tensor:
        """[n, d_sg] chart rows of group sg from states x [n, dim] (the pads dropped, not checked)."""
        if x.shape[-1] != self.dim:
            raise ValueError(f'FD states are {self.dim} wide, got {x.shape[-1]}')
        return x[:, self.slots[int(sg)].to(x.device)]
