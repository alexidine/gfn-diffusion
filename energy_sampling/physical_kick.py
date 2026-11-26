"""noised_prior.live_kernel 'physical': noised_prior.py::metric_kick's step from an anchor's canonical description, with
the new crystal's canonical chart point certified from a small table of basis changes.

  1. step    metric_kick's arithmetic (chart 'fd') on the anchor's description, decoded in closed form from its chart row.
  2. cell    only the basis changes of BASIS_TABLES (axis_order.axis_orders on the orthorhombic groups) are tried. A cell
             passes a basis when it is strictly inside every wall of the reduced domain there (ops.tri_niggli_conditions,
             ops.mono_conditions, the axis-order faces) by CELL_MARGIN. Triclinic and monoclinic: the identity is
             accepted as soon as it passes, and the other entries are then not evaluated; otherwise exactly one of them
             must pass. Orthorhombic: exactly one axis order must pass.
  3. pose    the centroid is certified when exactly one Gamma+ image lies strictly inside the asymmetric-unit box by
             T_MARGIN.
  4. chart   the closed-form inverse of the FD chart. A cell coordinate strictly inside the length box is accepted, one
             outside it by more than BOX_BAND is a rejection, and one in the band between is not certified.

A row not certified at any step goes through the full canonicaliser (full_route: canon.canonicalize and the production
chart). A row with no chart point is redrawn, at most `tries` draws in all, and then keeps its anchor's point.
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch

from energies.fd_coordinates import LENGTH_CONTACT, LENGTH_LO, OK_STATUS, S_TOL, length_box
from energy_sampling import noised_prior as NP
from mxtaltools.common.geometry_utils import init_sym_info
from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.crystal_fd import axis_order, canon, charts, domains, ops, so3
from mxtaltools.crystal_fd.description import CrystalDescription

F64 = torch.float64
CELL_MARGIN = 1e-6      # dimensionless, on every reduced-cell wall condition
T_MARGIN = 1e-7         # fractional, on every face of the asymmetric-unit box
BOX_BAND = 1e-5         # unit coordinate: outside the length box by more is a rejection
BOX_MARGIN = 1e-7       # unit coordinate: inside the length box by more is an acceptance
ORI_MARGIN = 1e-7       # unit theta and r: closer to the orientation chart's seams is not certified
_PAD = 1e-6             # domains._PAD: the margin of the fold's lattice shift
FULL_ROUTE_WARN_SHARE = 0.05    # a run with more than this share of its rows on the full route is told once ...
FULL_ROUTE_WARN_ROWS = 4096     # ... once it has drawn this many rows (train.py::Modeller._physical_kernel_draw)

#: the basis changes tried per space group, each a row-major 3 x 3 integer matrix, identity first. None: the group's
#: whole allowed set comes from axis_order.axis_orders. A basis missing from a table costs a full-route fallback, never
#: a wrong row. data_processing/mine_basis_tables.py rebuilds an entry: every distinct basis change canon.canonicalize
#: chooses on kicks of a prior's anchors, identity first and then most frequent first. The entries below are that list
#: for 300,000 kicks per group (half at the radius law 10^U(-2, -0.375) Angstrom, half at 1.4 times it) of a random 60%
#: of the anchors of pool4_test_train_k4_metric.pt. tests/crystal/test_physical_kernel.py accepts a table: unimodular,
#: no repeated entry, and every row it certifies equal to the full canonicaliser's.
BASIS_TABLES = {
    2: (
        (1, 0, 0, 0, 1, 0, 0, 0, 1), (-1, 1, 0, 0, 1, 0, 0, 0, -1), (-1, 0, 0, 0, 0, -1, 0, -1, 0),
        (-1, 0, 0, 0, -1, 0, 0, 0, 1), (-1, 0, 1, 0, -1, 0, 0, 0, 1), (-1, 0, 0, 0, 1, 0, 0, 0, -1),
        (0, 1, 0, 1, 0, 0, 0, 0, -1), (0, -1, 0, -1, 0, 0, 0, 0, -1), (-1, 0, 1, 0, -1, 1, 0, 0, 1),
        (-1, 0, 0, 0, -1, -1, 0, 0, 1), (1, 0, 0, 0, 1, 1, 0, 0, 1), (1, 0, 0, 0, 1, -1, 0, 0, 1),
        (1, -1, 0, 0, -1, 0, 0, 0, -1), (0, -1, 0, -1, 0, 1, 0, 0, -1), (1, -1, 0, 0, 0, 1, 0, -1, 0),
        (1, -1, -1, 0, -1, 0, 0, 0, -1), (1, 0, 0, 0, -1, 0, 0, 0, -1), (1, -1, -1, 0, -1, -1, 0, 0, -1),
        (1, 0, 0, 0, 0, -1, 0, 1, 0), (1, 0, -1, 0, -1, 0, 0, 0, -1), (0, 1, 0, -1, 1, 0, 0, 0, 1),
        (1, -1, 0, 0, -1, 1, 0, 0, -1), (1, 0, -1, 0, 0, -1, 0, 1, 0), (0, 1, -1, 1, 0, -1, 0, 0, -1),
        (0, 1, 0, 1, 0, -1, 0, 0, -1), (1, 0, 0, 0, 0, 1, 0, -1, 0), (0, -1, -1, -1, 0, 0, 0, 0, -1),
        (-1, 0, 0, 0, 1, -1, 0, 0, -1), (0, -1, 0, 0, 0, 1, -1, 0, 0), (0, -1, 0, 1, 0, 0, 0, 0, 1),
        (0, 1, 0, -1, 0, 0, 0, 0, 1), (1, -1, 0, 0, -1, 1, 0, -1, 0), (0, 0, -1, -1, 0, 0, 0, 1, 0),
        (0, 1, 0, 0, 0, 1, 1, 0, 0), (0, 0, 1, 1, 0, 0, 0, 1, 0), (-1, 0, 0, 0, 1, 1, 0, 0, -1),
        (0, 1, -1, 1, 0, 0, 0, 0, -1), (0, 0, -1, 0, -1, 0, -1, 0, 0), (1, -1, 0, 1, 0, 0, 0, 0, 1),
        (1, 0, 0, 0, 0, -1, 0, 1, 1), (-1, 1, 0, -1, 0, 0, 0, 0, 1), (-1, 0, 0, -1, 1, 0, 0, 0, -1),
        (-1, 0, 0, 0, 0, -1, 0, -1, -1), (0, 0, 1, 0, -1, 0, 1, 0, 0), (0, -1, 0, 1, -1, 0, 0, 0, 1),
        (-1, 1, 0, 0, 1, 0, 0, 1, -1), (-1, 0, 0, 0, 0, -1, 0, -1, 1), (1, 0, -1, 0, 0, -1, 0, 1, -1),
        (1, 0, 0, 1, -1, 0, 0, 0, -1), (-1, 0, 0, 0, 1, -1, 0, -1, 0), (1, 0, 0, 0, 1, 1, 0, -1, 0),
        (-1, 0, 0, 0, -1, -1, 0, -1, 0), (-1, 1, 1, 0, 1, 1, 0, 0, -1), (-1, 0, 0, 0, 1, 0, 0, -1, -1),
        (1, 0, 0, 0, 1, 0, 0, 1, 1), (1, 0, 0, 0, 1, 0, 0, -1, 1), (-1, 1, -1, 0, 1, -1, 0, 0, -1),
        (-1, 0, 0, 0, 0, 1, 0, 1, 0), (1, 0, 0, 0, -1, -1, 0, 0, -1), (1, 0, -1, 0, -1, 1, 0, 0, -1),
        (-1, 1, 0, 0, 1, -1, 0, 0, -1), (0, 1, 1, -1, 1, 1, 0, 0, 1), (-1, 0, 0, 0, -1, 1, 0, 0, 1),
        (0, 0, -1, -1, 1, 0, 0, 1, 0), (0, -1, 0, 1, -1, -1, 0, 0, 1), (0, 1, 0, -1, 0, 1, 0, 0, 1),
        (1, 0, -1, 0, 1, -1, 0, 0, 1), (1, 0, 1, 0, 1, 1, 0, 0, 1), (-1, 0, 0, -1, 1, 1, 0, 0, -1),
        (0, 0, -1, 1, 0, 0, 0, -1, 0), (-1, 1, 1, 0, 0, 1, 0, 1, 1), (0, 0, -1, 0, -1, 0, -1, 1, 0),
        (1, -1, -1, 0, -1, 0, 0, -1, -1), (-1, 1, 1, -1, 0, 1, 0, 0, 1), (0, 0, 1, 0, -1, 0, 1, -1, 0),
        (0, 1, 0, 0, 0, 1, 1, 0, -1), (1, 0, -1, 1, -1, -1, 0, 0, -1), (-1, 0, 1, 0, -1, -1, 0, 0, 1),
        (0, 0, 1, 0, 1, 0, -1, 0, 0), (0, 0, -1, -1, 0, 0, -1, 1, 0), (1, 0, 0, 0, 0, 1, 0, -1, 1),
        (1, -1, 0, 1, 0, -1, 0, 0, 1), (0, 0, 1, 0, 1, 1, -1, 0, 0), (1, 0, 0, 0, 1, -1, 0, 1, 0),
        (0, 1, 1, -1, 0, 0, 0, 0, 1), (1, 0, -1, 0, -1, -1, 0, 0, -1), (0, 0, -1, 0, 1, 0, 1, 0, -1),
        (1, 0, 0, 1, -1, 0, 1, 0, -1), (-1, 1, 1, 0, 1, 1, 0, 1, 0), (0, -1, 0, 0, 0, -1, 1, -1, 0),
        (-1, 1, 0, 0, 0, -1, -1, 0, 0), (0, 0, -1, -1, 1, 0, 1, 0, 0), (0, 1, 0, 0, 1, 1, 1, 0, 0),
        (1, -1, -1, 0, 0, -1, 1, 0, -1), (-1, 1, 0, 0, 0, 1, 0, 1, 0), (0, -1, 0, 0, 0, -1, 1, 0, 0),
        (0, -1, 1, 1, 0, -1, 0, 0, 1), (0, -1, 0, -1, 0, 1, -1, 0, 0), (0, -1, 1, 1, 0, 0, 0, 0, 1),
        (-1, 0, 1, -1, 0, 0, 0, -1, 0), (-1, 1, 0, 0, 1, 1, 0, 0, -1), (0, -1, 0, 0, -1, 1, -1, 0, 0),
        (0, 0, 1, 1, -1, 0, 0, 1, 0), (0, 1, -1, -1, 1, 0, 0, 1, 0), (1, 0, 0, 0, -1, -1, 0, 1, 0),
        (0, -1, 0, 0, 0, 1, -1, 0, 1), (1, 0, -1, 1, -1, 0, 0, 0, -1), (0, 1, 0, 1, 0, -1, -1, 0, 0),
        (-1, 0, 1, 0, -1, 1, 0, 1, 0), (1, 0, 0, 1, 0, -1, 0, 1, 0), (1, -1, 0, 0, -1, 0, 0, 1, -1),
        (0, 0, 1, 0, 1, 0, -1, 0, 1), (0, 0, -1, 1, -1, 0, -1, 0, 0), (0, 1, -1, -1, 0, 0, 0, 1, 0),
        (-1, 1, 0, 0, -1, 1, 0, 1, 0), (0, -1, 1, 0, 0, 1, -1, 0, 0), (-1, 0, 1, -1, 1, 0, -1, 0, 0),
        (1, 0, -1, 0, 1, -1, 0, 1, 0), (1, -1, 0, 1, 0, -1, 1, 0, 0), (1, -1, -1, 1, 0, -1, 1, 0, 0),
        (-1, 1, 0, 0, 0, 1, 0, 1, -1), (0, -1, 1, 1, -1, 1, 0, 0, 1), (0, 0, 1, 1, 0, 0, -1, 1, 0),
        (-1, 1, 1, 0, 0, 1, 0, 1, 0), (-1, 0, 1, 0, 0, 1, 0, 1, 0), (0, 0, -1, 1, 0, -1, 0, -1, 0),
        (0, 1, 1, 0, 0, 1, 1, 0, 0), (0, -1, 0, 1, 0, -1, 0, 0, 1),
    ),
    14: (
        (1, 0, 0, 0, 1, 0, 0, 0, 1), (1, 0, 0, 0, -1, 0, 0, 0, -1), (1, 0, 0, 0, -1, 0, 1, 0, -1),
        (1, 0, -2, 0, -1, 0, 0, 0, -1), (1, 0, -2, 0, 1, 0, 1, 0, -1),
    ),
    19: None,
    61: None,
}


@dataclass
class GroupKit:
    """One space group's constants for the kick, all on the CPU in float64 unless named."""
    g: int
    family: str
    kind: str
    Z: int
    d: int                          # chart width
    nc: int                         # cell coordinates of the chart
    slots: torch.Tensor             # [d] long: the chart's slots in the run's FD layout
    live: torch.Tensor              # [12] bool: metric_kick's live coordinates
    cen_lo: torch.Tensor            # [3]
    cen_w: torch.Tensor             # [3]
    asu_lo: torch.Tensor            # [3]
    asu_hi: torch.Tensor            # [3]
    W: torch.Tensor                 # [m, 3, 3] Gamma+
    w: torch.Tensor                 # [m, 3]
    N: torch.Tensor                 # [K, 3, 3] basis changes, identity first
    Ninv: torch.Tensor              # [K, 3, 3]
    shift: torch.Tensor             # [K, 3]
    perm: Optional[torch.Tensor]    # [K, 3] long, orthorhombic axis orders
    faces: Optional[torch.Tensor]   # orthorhombic face rows
    mono_cls: Optional[int]         # ops.mono_class of a monoclinic group


def group_kit(g, slots, table=None) -> GroupKit:
    """The kit of space group g with chart slots `slots`. Raises, naming the group, when it has no basis table or its
    chart is not of the form the closed forms below are written for. `table`: basis changes read in place of
    BASIS_TABLES[g], for a group not yet in it (data_processing/mine_basis_tables.py)."""
    g = int(g)
    if table is None and g not in BASIS_TABLES:
        raise ValueError(f"noised_prior.live_kernel 'physical': space group {g} has no basis table "
                         f"(physical_kick.BASIS_TABLES holds {sorted(BASIS_TABLES)})")
    fam = charts.cell_family(g)
    if fam not in ('triclinic', 'monoclinic', 'orthorhombic'):
        raise ValueError(f"noised_prior.live_kernel 'physical': space group {g} is {fam}, outside the closed-form charts")
    chart = charts.CrystalChart.for_length_box(g, length_box(g, torch.tensor([4.0], dtype=F64)), 'hemisphere')
    cen = chart.centroid
    if cen.basis is not None or list(cen.order) != [0, 1, 2] or any(
            float(st.lower_coef.abs().sum()) or float(st.upper_coef.abs().sum()) for st in cen.stages):
        raise ValueError(f"noised_prior.live_kernel 'physical': space group {g}: the centroid chart is not a box")
    clo = torch.stack([st.lower_const.amax() for st in cen.stages]).to('cpu', F64)
    chi = torch.stack([st.upper_const.amin() for st in cen.stages]).to('cpu', F64)
    sec = domains.translation_section(g, torch.device('cpu'))
    if sec.shifts.shape[0] != 1 or sec.region.normals.shape[0] != 6 or bool(sec.free.any()):
        raise ValueError(f"noised_prior.live_kernel 'physical': space group {g}: the asymmetric unit is not a box "
                         f"reached by one lattice shift")
    code = int(init_sym_info()['lattice_code'][g])
    P = NP._metric_projector(torch.tensor([code]), F64, 'cpu')[0]
    if float((P - torch.diag(torch.diagonal(P))).abs().max()) > 0:
        raise ValueError(f"noised_prior.live_kernel 'physical': space group {g}: the metric projector is not diagonal")
    perm, faces, kind, mono_cls = None, None, 'none', None
    if fam == 'orthorhombic':
        o = axis_order.axis_orders(g)
        kind, perm, faces = o.kind, o.perm.cpu(), o.faces.to('cpu', F64)
        N, shift = o.N.to('cpu', F64), o.s.to('cpu', F64)
    else:
        N = torch.tensor(BASIS_TABLES[g] if table is None else table, dtype=F64).reshape(-1, 3, 3)
        shift = torch.zeros(len(N), 3, dtype=F64)
    if fam == 'monoclinic':
        mono_cls = int(ops.mono_class(torch.tensor([g]))[0])
    if not torch.equal(N[0], torch.eye(3, dtype=F64)):
        raise ValueError(f"noised_prior.live_kernel 'physical': space group {g}: the basis table must start at the "
                         f"identity")
    return GroupKit(g=g, family=fam, kind=kind, Z=len(SYM_OPS[g]), d=chart.ndim, nc=chart.cell.ndim,
                    slots=torch.as_tensor(slots).long().cpu(), live=torch.diagonal(P) > 0.5, cen_lo=clo, cen_w=chi - clo,
                    asu_lo=sec.region.lo.to('cpu', F64), asu_hi=sec.region.hi.to('cpu', F64),
                    W=sec.W.to('cpu', F64), w=sec.w.to('cpu', F64), N=N, Ninv=torch.round(torch.linalg.inv(N)),
                    shift=shift, perm=perm, faces=faces, mono_cls=mono_cls)


def group_kits(layout) -> dict:
    """{space group: GroupKit} of an energies/fd_coordinates.py::FDSlotLayout."""
    return {int(g): group_kit(g, layout.slots[int(g)]) for g in layout.groups}


# ---------------------------------------------------------------------------
# chart <-> description: closed forms of charts.CrystalChart and charts.from_unit / to_unit for these families
# ---------------------------------------------------------------------------

def _mono_bound(k: GroupKit, a, c):
    return ops.mono_bound(a, c, torch.full(a.shape, k.mono_cls, dtype=torch.long))


def decode(k: GroupKit, u, radius):
    """(cell [n, 6], t [n, 3], R [n, 3, 3]) of unit chart rows u [n, d] (charts.from_unit then CrystalChart.forward)."""
    hi = 2 * k.Z * (radius + LENGTH_CONTACT)
    llo, lhi = math.log(LENGTH_LO), torch.log(hi)
    half = torch.full_like(hi, math.pi / 2)
    if k.family == 'triclinic':
        c = torch.exp(llo + u[:, 0] * (lhi - llo))
        l2 = LENGTH_LO / c
        r2 = l2 + u[:, 1] * (torch.minimum(hi, c) / c - l2)
        b = r2 * c
        l1 = LENGTH_LO / b
        r1 = l1 + u[:, 2] * (torch.minimum(hi, b) / b - l1)
        a = r1 * b
        sb, sg = u[:, 3] - 1, u[:, 4] - 1
        lam = -1 + r1 * r1 * (-1 - sb - sg).clamp(min=0)
        sa = lam + (1 - lam) * u[:, 5]
        cell = torch.stack([a, b, c, torch.arccos(sa * r2 / 2), torch.arccos(sb * r1 * r2 / 2),
                            torch.arccos(sg * r1 / 2)], 1)
    elif k.family == 'monoclinic':
        wid = lhi - llo
        c, a, b = torch.exp(llo + u[:, 0] * wid), torch.exp(llo + u[:, 1] * wid), torch.exp(llo + u[:, 2] * wid)
        cell = torch.stack([a, b, c, half, torch.arccos(-u[:, 3] * _mono_bound(k, a, c)), half], 1)
    else:
        la = llo + u[:, 0] * (lhi - llo)
        blo = la.clamp(min=llo)
        lb = blo + u[:, 1] * (lhi - blo)
        clo = (lb if k.kind == 'all' else la).clamp(min=llo)
        lc = clo + u[:, 2] * (lhi - clo)
        cell = torch.stack([torch.exp(la), torch.exp(lb), torch.exp(lc), half, half, half], 1)
    t = k.cen_lo + u[:, k.nc:k.nc + 3] * k.cen_w
    th = u[:, k.nc + 3] * (math.pi / 2)
    ph = u[:, k.nc + 4] * (2 * math.pi) - math.pi
    r = u[:, k.nc + 5] * (2 * math.pi)
    return cell, t, so3.hemisphere_to_matrix(torch.stack([th, ph, r], 1))


def cell_unit(k: GroupKit, cell, radius):
    """[n, nc] unit coordinates of canonical cells (CellChart.unmap then charts.to_unit); not clamped."""
    hi = 2 * k.Z * (radius + LENGTH_CONTACT)
    llo, lhi = math.log(LENGTH_LO), torch.log(hi)
    a, b, c = cell[:, 0], cell[:, 1], cell[:, 2]
    if k.family == 'triclinic':
        r2, r1 = b / c, a / b
        sa, sb, sg = 2 * torch.cos(cell[:, 3]) / r2, 2 * torch.cos(cell[:, 4]) / (r1 * r2), 2 * torch.cos(cell[:, 5]) / r1
        lam = -1 + r1 * r1 * (-1 - sb - sg).clamp(min=0)
        l2 = LENGTH_LO / c
        l1 = LENGTH_LO / b
        return torch.stack([(torch.log(c) - llo) / (lhi - llo), (r2 - l2) / (torch.minimum(hi, c) / c - l2),
                            (r1 - l1) / (torch.minimum(hi, b) / b - l1), sb + 1, sg + 1, (sa - lam) / (1 - lam)], 1)
    if k.family == 'monoclinic':
        wid = lhi - llo
        s = -torch.cos(cell[:, 4]) / _mono_bound(k, a, c)
        return torch.stack([(torch.log(c) - llo) / wid, (torch.log(a) - llo) / wid, (torch.log(b) - llo) / wid, s], 1)
    la, lb, lc = torch.log(a), torch.log(b), torch.log(c)
    blo = la.clamp(min=llo)
    clo = (lb if k.kind == 'all' else la).clamp(min=llo)
    return torch.stack([(la - llo) / (lhi - llo), (lb - blo) / (lhi - blo), (lc - clo) / (lhi - clo)], 1)


def pose_unit(k: GroupKit, t, R):
    """[n, 6] unit coordinates of a folded centroid and a rotation (the hemisphere chart), phi wrapped as charts.to_unit."""
    y = so3.matrix_to_hemisphere(R)
    return torch.cat([(t - k.cen_lo) / k.cen_w, y[:, 0:1] / (math.pi / 2),
                      torch.remainder(y[:, 1:2] + math.pi, 2 * math.pi) / (2 * math.pi), y[:, 2:3] / (2 * math.pi)], 1)


def t_fc(cell):
    """[n, 3, 3] upper triangular T_fc (geometry_utils.batch_compute_fractional_transform)."""
    a, b, c = cell[:, 0], cell[:, 1], cell[:, 2]
    ca, cb, cg = torch.cos(cell[:, 3]), torch.cos(cell[:, 4]), torch.cos(cell[:, 5])
    sg = torch.sin(cell[:, 5])
    f = 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg
    z = torch.zeros_like(a)
    return torch.stack([a, b * cg, c * cb, z, b * sg, c * (ca - cb * cg) / sg, z, z,
                        c * torch.sqrt(f.clamp(min=0)) / sg], 1).reshape(-1, 3, 3)


def gram_factor(cell):
    """[n] 1 - cos^2 alpha - cos^2 beta - cos^2 gamma + 2 cos alpha cos beta cos gamma."""
    ca, cb, cg = torch.cos(cell[:, 3]), torch.cos(cell[:, 4]), torch.cos(cell[:, 5])
    return 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg


def gram(cell):
    """[n, 3, 3] metric tensors of cells [n, 6]."""
    a, b, c = cell[:, 0], cell[:, 1], cell[:, 2]
    x, y, z = b * c * torch.cos(cell[:, 3]), a * c * torch.cos(cell[:, 4]), a * b * torch.cos(cell[:, 5])
    return torch.stack([a * a, z, y, z, b * b, x, y, x, c * c], 1).reshape(-1, 3, 3)


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------

def draw_step(k: GroupKit, n, log_lo, log_hi, generator=None):
    """[n, 12] metric_kick's step rho u: rho = 10^U(log_lo, log_hi) Angstrom, u uniform on the live coordinates' sphere."""
    y = torch.randn(n, 12, dtype=F64, generator=generator) * k.live
    u = y / y.norm(dim=1, keepdim=True)
    rho = 10 ** (log_lo + (log_hi - log_lo) * torch.rand(n, dtype=F64, generator=generator))
    return rho[:, None] * u


def apply_step(st, cell, t, R, rg, T):
    """noised_prior.py::metric_kick's update of a description by the step st [n, 12]: the new (cell, t, R). T: T_fc of
    the description before the step."""
    a, b, c = cell[:, 0], cell[:, 1], cell[:, 2]
    new = torch.cat([cell[:, :3] + st[:, :3],
                     cell[:, 3:] + st[:, 3:6] / torch.stack([torch.sqrt(b * c), torch.sqrt(a * c), torch.sqrt(a * b)], 1)],
                    1)
    z = st[:, 8] / T[:, 2, 2]
    y = (st[:, 7] - T[:, 1, 2] * z) / T[:, 1, 1]
    x = (st[:, 6] - T[:, 0, 1] * y - T[:, 0, 2] * z) / T[:, 0, 0]
    return new, t + torch.stack([x, y, z], 1), NP._rodrigues(st[:, 9:12] / rg[:, None]) @ R


# ---------------------------------------------------------------------------
# certified canonicalisation
# ---------------------------------------------------------------------------

def _certify_cell(k: GroupKit, cell):
    """([n] index into the basis table, -1 where not certified; the Gram matrices, None on the orthorhombic route)."""
    n = len(cell)
    pick = torch.full((n,), -1, dtype=torch.long)
    if k.family == 'orthorhombic':
        keys = torch.log(cell[:, :3])[:, k.perm]                                    # [n, K, 3]
        fv = axis_order._face_dot(keys, k.faces).amin(-1)                           # [n, K]
        good = fv > CELL_MARGIN
        one = good.sum(1) == 1
        pick[one] = good[one].to(torch.int8).argmax(1)
        return pick, None
    G = gram(cell)
    cond = ops.tri_niggli_conditions if k.family == 'triclinic' else (lambda X: ops.mono_conditions(X, k.g))
    pick[cond(G).amin(-1) > CELL_MARGIN] = 0
    rest = torch.nonzero(pick < 0).flatten()
    if len(rest) and len(k.N) > 1:
        Gc = k.N.transpose(1, 2)[None, 1:] @ G[rest, None] @ k.N[None, 1:]          # [r, K - 1, 3, 3]
        good = cond(Gc).amin(-1) > CELL_MARGIN
        one = good.sum(1) == 1
        pick[rest[one]] = good[one].to(torch.int8).argmax(1) + 1
    return pick, G


def _new_cells(k: GroupKit, cell, G, idx):
    """[r, 6] the cells in basis idx [r] (>= 1) of the table."""
    if k.family == 'orthorhombic':
        return torch.cat([cell[:, :3].gather(1, k.perm[idx]), cell[:, 3:]], 1)
    N = k.N[idx]
    Gn = N.transpose(1, 2) @ G @ N
    L = torch.sqrt(torch.diagonal(Gn, dim1=1, dim2=2))
    if k.family == 'monoclinic':                                                    # ops.monoclinic_basis
        half = torch.full_like(L[:, 0], math.pi / 2)
        return torch.stack([L[:, 0], cell[:, 1], L[:, 2], half,
                            torch.arccos((Gn[:, 0, 2] / (L[:, 0] * L[:, 2])).clamp(-1, 1)), half], 1)
    return ops.cell_from_metric_tensor(Gn)


def certified_chart(k: GroupKit, cell, t, R, radius):
    """(u [n, d] unit chart rows, state [n] int8) of descriptions on the +1 sheet. state 1: certified, with a chart point;
    0: certified outside the length box, no chart point; -1: not certified, for the full route."""
    n = len(cell)
    state = torch.full((n,), -1, dtype=torch.int8)
    u = torch.full((n, k.d), float('nan'), dtype=F64)
    sane = torch.isfinite(cell).all(1) & (cell[:, :3] > 0).all(1) & torch.isfinite(t).all(1)
    sane &= gram_factor(torch.where(sane[:, None], cell, torch.ones_like(cell))) > 1e-9
    if k.family == 'monoclinic':
        sane &= ((cell[:, [3, 5]] - math.pi / 2).abs() <= 1e-9).all(1)
    idx, G = _certify_cell(k, torch.where(sane[:, None], cell, torch.ones_like(cell)))
    idx = torch.where(sane, idx, torch.full_like(idx, -1))
    live = torch.nonzero(idx >= 0).flatten()
    if not len(live):
        return u, state
    c1, t1, R1 = cell[live], t[live], R[live]
    move = torch.nonzero(idx[live] > 0).flatten()
    if len(move):                                                                   # ops.change_basis, without the SVD
        e = idx[live][move]
        cn = _new_cells(k, c1[move], None if G is None else G[live][move], e)
        Ni = k.Ninv[e]
        Q = torch.linalg.solve_triangular(t_fc(c1[move]), t_fc(cn) @ Ni, upper=True, left=False)
        t1, R1, c1 = t1.clone(), R1.clone(), c1.clone()
        t1[move] = (Ni @ (t1[move] - k.shift[e])[..., None])[..., 0]
        R1[move] = Q @ R1[move]
        c1[move] = cn
    cu = cell_unit(k, c1, radius[live])
    out = (~torch.isfinite(cu)).any(1) | (cu < -BOX_BAND).any(1) | (cu > 1 + BOX_BAND).any(1)
    inside = torch.isfinite(cu).all(1) & (cu > BOX_MARGIN).all(1) & (cu < 1 - BOX_MARGIN).all(1)
    state[live[out]] = 0
    keep = torch.nonzero(inside).flatten()
    if not len(keep):
        return u, state
    c2, t2, R2, cu = c1[keep], t1[keep], R1[keep], cu[keep]
    in_box = ((t2 > k.asu_lo + T_MARGIN) & (t2 < k.asu_hi - T_MARGIN)).all(1)
    fold = torch.nonzero(~in_box).flatten()
    done = in_box.clone()
    if len(fold):
        y = torch.einsum('mij,bj->bmi', k.W, t2[fold]) + k.w                        # domains._fold_block, S = 1
        x = y + torch.ceil(k.asu_lo - _PAD - y)
        ok = ((x > k.asu_lo + T_MARGIN) & (x < k.asu_hi - T_MARGIN)).all(-1)        # [b, m]
        one = ok.sum(1) == 1
        if bool(one.any()):
            fi, e = fold[one], ok[one].to(torch.int8).argmax(1)
            t2, R2 = t2.clone(), R2.clone()
            t2[fi] = x[one, e]
            moved = torch.nonzero(e > 0).flatten()
            if len(moved):                                                          # Q = T W T^-1 (domains._rotations)
                T = t_fc(c2[fi[moved]])
                Q = torch.linalg.solve_triangular(T, T @ k.W[e[moved]], upper=True, left=False)
                R2[fi[moved]] = Q @ R2[fi[moved]]
            done[fi] = True
    hit = torch.nonzero(done).flatten()
    if len(hit):
        pu = pose_unit(k, t2[hit], R2[hit])
        ori = pu[:, 3:]     # the orientation chart's own seams: rounding can pick either description there
        clear = (ori[:, 0] > ORI_MARGIN) & (ori[:, 0] < 1 - ORI_MARGIN) & (ori[:, 2] > ORI_MARGIN) \
            & (ori[:, 2] < 1 - ORI_MARGIN) & torch.isfinite(pu).all(1)
        rows = live[keep[hit[clear]]]
        u[rows] = torch.cat([cu[hit[clear]], pu[clear]], 1)
        state[rows] = 1
    return u, state


def full_route(k: GroupKit, cell, t, R, radius):
    """(u [n, d], ok [n] bool) through canon.canonicalize and the production chart: the fallback and the reference."""
    n = len(cell)
    u = torch.full((n, k.d), float('nan'), dtype=F64)
    ok = torch.zeros(n, dtype=torch.bool)
    sane = torch.nonzero(torch.isfinite(cell).all(1) & (cell[:, :3] > 0).all(1) & torch.isfinite(t).all(1)
                         & torch.isfinite(R).all(-1).all(-1)).flatten()
    if not len(sane):
        return u, ok
    c = canon.canonicalize(CrystalDescription.from_parts(torch.full((len(sane),), k.g), cell[sane], t[sane], R[sane],
                                                         torch.ones(len(sane), dtype=torch.long)))
    codes = torch.tensor([canon._CODE[x] for x in OK_STATUS], dtype=c.status.dtype)
    good = torch.nonzero(torch.isin(c.status, codes)).flatten()
    if not len(good):
        return u, ok
    ch = charts.CrystalChart.for_length_box(k.g, length_box(k.g, radius[sane[good]]), 'hemisphere', device='cpu')
    y, _ = ch.inverse(c.desc[good])
    v = charts.to_unit(ch, y)
    ins = ((v >= -S_TOL) & (v <= 1 + S_TOL)).all(1) & torch.isfinite(v).all(1)
    u[sane[good[ins]]] = v[ins].clamp(0, 1)
    ok[sane[good[ins]]] = True
    return u, ok


# ---------------------------------------------------------------------------
# the draw
# ---------------------------------------------------------------------------

def kick_group(k: GroupKit, s, radius, rg, log_lo, log_hi, tries, generator=None, rho_out=None):
    """([n, d] float64 chart rows, counts) of fresh kicks of one group's anchors s [n, d] in [-1, 1]. Each row takes at
    most `tries` draws, then keeps its anchor. counts: rows, full_route (rows that used the full route in some draw),
    redrawn (rows whose first draw had no chart point), kept (rows that kept their anchor). rho_out [n] (optional):
    filled with each row's accepted step length (Angstrom), 0 for a row that kept its anchor."""
    n = len(s)
    cell, t, R = decode(k, ((s + 1) / 2).clamp(0, 1), radius)
    T = t_fc(cell)
    out = s.clone()
    full = torch.zeros(n, dtype=torch.bool)
    todo = torch.arange(n)
    redrawn = 0
    for attempt in range(int(tries)):
        st = draw_step(k, len(todo), log_lo, log_hi, generator)
        c2, t2, R2 = apply_step(st, cell[todo], t[todo], R[todo], rg[todo], T[todo])
        u, state = certified_chart(k, c2, t2, R2, radius[todo])
        rest = torch.nonzero(state < 0).flatten()
        if len(rest):
            uf, okf = full_route(k, c2[rest], t2[rest], R2[rest], radius[todo][rest])
            u[rest] = uf
            state[rest] = okf.to(state.dtype)
            full[todo[rest]] = True
        good = state == 1
        out[todo[good]] = 2 * u[good].clamp(0, 1) - 1
        if rho_out is not None:
            rho_out[todo[good]] = st[good].norm(dim=1).to(rho_out.dtype)
        todo = todo[~good]
        if attempt == 0:
            redrawn = len(todo)
        if not len(todo):
            break
    return out, dict(rows=n, full_route=int(full.sum()), redrawn=redrawn, kept=len(todo))


@contextmanager
def single_thread():
    """Run the block on one torch thread and restore the global thread count on exit, also on an exception."""
    prev = torch.get_num_threads()
    if prev != 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if prev != 1:
            torch.set_num_threads(prev)


@torch.no_grad()
def kick(kits, centres, sg, radius, rg, log_lo, log_hi, tries, generator=None, rho_out=None):
    """([n, dim] float32 CPU states in the run's slot layout, counts summed over the groups) of fresh kicks of the anchors
    `centres` [n, dim] (any device; only their chart slots are read) with space groups sg [n], molecule radii and radii
    of gyration [n]. Runs on the CPU on one thread; a row of a group outside `kits` raises. rho_out [n] float64 CPU
    (optional): filled with each row's step length (Angstrom; 0 where the row kept its anchor), kick_group's."""
    centres, sg = centres.cpu(), torch.as_tensor(sg).reshape(-1).long().cpu()
    radius, rg = torch.as_tensor(radius).cpu().to(F64), torch.as_tensor(rg).cpu().to(F64)
    radius = torch.round(radius * 1e6) / 1e6        # the production chart's radius (MolecularCrystal.fd_coordinates_rows)
    x = torch.zeros_like(centres, dtype=torch.float32)
    total = dict(rows=0, full_route=0, redrawn=0, kept=0)
    with single_thread():
        for g in torch.unique(sg).tolist():
            if g not in kits:
                raise ValueError(f"noised_prior.live_kernel 'physical': space group {g} has no table in this run "
                                 f"(built for {sorted(kits)})")
            k, at = kits[g], torch.nonzero(sg == g).flatten()
            rho_g = None if rho_out is None else torch.zeros(len(at), dtype=F64)
            out, counts = kick_group(k, centres[at][:, k.slots].to(F64), radius[at], rg[at], log_lo, log_hi, tries,
                                     generator, rho_out=rho_g)
            if rho_out is not None:
                rho_out[at] = rho_g
            row = torch.zeros(len(at), x.shape[1], dtype=torch.float32)
            row[:, k.slots] = out.to(torch.float32)
            x[at] = row
            for name, v in counts.items():
                total[name] += v
    return x, total


def radius_of_gyration_rows(batch, rows, chunk: int = 1 << 16):
    """[len(rows)] float32 CPU: noised_prior.py::radius_of_gyration of the rows `rows` of `batch`, read in chunks through
    its atom pointers, so a store's atoms are never all converted to float64 at once."""
    rows = torch.as_tensor(rows).long().flatten().cpu()
    ptr = batch.ptr.cpu()
    out = torch.zeros(len(rows), dtype=torch.float32)
    for lo in range(0, len(rows), chunk):
        part = rows[lo:lo + chunk]
        start, count = ptr[part], ptr[part + 1] - ptr[part]
        first = torch.cat([torch.zeros(1, dtype=count.dtype), count.cumsum(0)[:-1]])
        group = torch.arange(len(part)).repeat_interleave(count)
        atoms = torch.arange(int(count.sum())) - first[group] + start[group]
        atoms = atoms.to(batch.pos.device)
        view = _AtomView(batch.pos[atoms], batch.z.reshape(-1)[atoms], group.to(batch.pos.device), len(part))
        out[lo:lo + chunk] = NP.radius_of_gyration(view).float().cpu()
    return out


@dataclass
class _AtomView:
    """The atom fields noised_prior.radius_of_gyration reads."""
    pos: torch.Tensor
    z: torch.Tensor
    batch: torch.Tensor
    num_graphs: int
