"""The THERMAL anchor tile: a per-coordinate Gaussian displacement at each coordinate's own
thermal width, for conformer states.

The isotropic tile (``ConformerModeller._noise_and_condition`` under ``tile: 'iso'``) draws one
radius in state units for every column. Conformer columns differ in stiffness by orders of
magnitude -- a bond length's thermal width is a few hundredths of an Angstrom, a free rotor's
tens of degrees -- so one radius strains the stiff columns by many kT while barely moving the
soft ones. This tile draws each coordinate at its own width instead.

WHAT IS DRAWN, per row of member m, in m's own chart and then placed in the carrier:

    delta = c * ( z * sigma  +  sum_g eta_g * w_g * e_g ),     z, eta standard normal

``c`` is the per-row multiplier, drawn by the caller. The DRAW BASIS is one direction per state
column: every column on its own axis, except that each sibling group's LEADER column (the
first row of a ``torsion_groups`` group) is replaced by ``e_g``, the RIGID ROTATION of the
whole group about its central bond (the same angle on every row of the group, i.e. 1 / scale
on each of its columns). The rotation is the soft torsional coordinate; moving one sibling
alone changes the angle between siblings and is stiff. This is the leader-plus-followers
structure ``sample_prior_states`` draws with.

THE WIDTH of each basis direction is ``sqrt(kT / kappa)``, kappa the second derivative of the
member's potential (``potential_energy`` at T = 1: every force-field term, the stereo lock,
the clip and the box wall) along that direction at the member's REFERENCE state (x = 0), by a
central difference (``curvatures``). In a quadratic basin this makes the expected excess of
each direction kT/2 whatever the couplings between directions, so the expected excess of a
row is exactly (k/2) kT for a member with k state columns.

Each column's width is additionally CAPPED at its coordinate's own-term width -- the width
the force field states outright for that coordinate alone:

  - bond length r and bond angle theta: ``ConformerTorsions.thermal_rtheta_sigma(T)``,
    sqrt(kT / 2k). The force field writes a bond or angle term as ``k (q - q0)^2`` with no
    factor 1/2 (mxtaltools/conformers/energy.py::intramolecular_energy), so exp(-k dq^2 / kT)
    is a Gaussian of variance kT / 2k: the factor 2 is right for this convention.
  - a transverse bend (u, v) at a linear centre: that angle row's theta width for both
    components. MMFF's linear form 2k (1 + cos theta) is k rho^2 + O(rho^4) in rho = pi -
    theta, and rho^2 = u^2 + v^2, so each component has variance kT / 2k.
  - a HELD phi row (an improper, or a locked double bond: ``held_phi_rows``):
    ``improper_phi_sigma(T)``, the width ``sample_prior_states`` holds it at.
  - a FOLLOWER phi row: ``sibling_jitter_sigma``, the width of the redundant angle between
    siblings.

The cap binds where the rest of the potential is locally concave at the reference, so the
width never exceeds what the coordinate's own term allows. For r the curvature and the own
term agree closely (the tree chart moves no other bonded term with a bond length); for theta,
followers and held rows they do NOT: a tree angle moved alone also moves the redundant angles
at its centre and, in a ring, stretches the closure bond, so the chart direction is stiffer
than the angle's own term. Measured on the 50-molecule QM9 database rung at c = 1 with the
own-term widths alone, theta directions carried 1.8x (acyclic) and 5.6x (ring) their kT/2 and
the row median excess was 2.0x k/2; with the curvature widths every class sits within 0.1 of
kT/2 per direction.

A GROUP ROTATION has no own term to cap it; where its curvature is at most
kT / PHI_SIGMA_MAX^2 (a flat or locally concave rotor at the reference) its width is
PHI_SIGMA_MAX = pi / sqrt(3), the standard deviation of a uniform angle on the circle. This
curvature is the torsional width the tile uses, chosen over the widths the chart already
provides because none of those is torsional: ``sibling_jitter_sigma`` and
``improper_phi_sigma`` are ANGLE widths (a few degrees), and InternalPrior's phi histograms
are rotamer POPULATIONS, not a local width about a state. Taken from the full potential, it
carries the torsion term, the 1-4 and longer-range non-bonded terms, and -- about a ring bond
-- the closure bond the rotation stretches, without a separate rule.

The curvatures are taken at the reference, not at each anchor: one width per direction per
member, so an anchor in another rotamer well or ring pucker sees the reference's widths. The
temperature is the energy's configured one (``temperature``), read at construction.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

#: std of a uniform angle on the circle, the width of a flat rotor
PHI_SIGMA_MAX = float(np.pi / np.sqrt(3.0))
#: central-difference steps for the curvatures: bond lengths (A), angles and single phi
#: rows (rad), group rotations (rad) -- each a fraction of the width it measures
FD_STEP_BOND = 0.005
FD_STEP_ANGLE = 0.01
FD_STEP_ROT = 0.05


def member_widths(member, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(sigma [k], rot [G, k])`` for one chart, in ITS OWN column order and state units.

    ``sigma`` is each column's independent width; row g of ``rot`` is ``w_g * e_g``, the
    group rotation's displacement per unit normal draw. See the module docstring.
    """
    if getattr(member, 'collective', False):
        raise NotImplementedError(
            f"the thermal tile needs a SELECTION chart (one DoF row per state column); "
            f"{member.smiles} is at level {member.level!r}, where a column rotates a whole "
            f"bond. Use tile 'iso' there.")
    T = float(temperature)
    k = int(member.data_ndim)
    n_r, n_th = int(member.n_r), int(member.n_th)
    n_phi0 = n_r + n_th
    scale = member._free_scale.detach().cpu().double().numpy()
    sel = member._sel_rows.detach().cpu().numpy()
    col_of_row = {int(r): c for c, r in enumerate(sel)}

    # OWN-TERM widths, in the coordinate's units, and the FD step for each column's kind
    s_r, s_th = member.thermal_rtheta_sigma(T)
    own = np.full(k, np.nan)
    step = np.full(k, np.nan)

    # TRANSVERSE PAIRS first: their u sits in an angle row's slot and their v in a phi row's,
    # and neither is a theta or a torsion
    tv_rows = np.flatnonzero(member.transverse_angles)
    u_cols = member._tv_u_cols.detach().cpu().numpy()
    v_cols = member._tv_v_cols.detach().cpu().numpy()
    tv_phi_rows = set()
    for i, j in enumerate(tv_rows[:len(u_cols)]):
        for c in (int(u_cols[i]), int(v_cols[i])):
            own[c], step[c] = s_th[int(j)], FD_STEP_ANGLE
        tv_phi_rows.add(int(member.transverse_partner[int(j)]))

    for row, c in col_of_row.items():
        if not np.isnan(own[c]):
            continue
        if row < n_r:
            own[c], step[c] = s_r[row], FD_STEP_BOND
        elif row < n_phi0:
            own[c], step[c] = s_th[row - n_r], FD_STEP_ANGLE

    held = [j for j in member.held_phi_rows()
            if j not in tv_phi_rows and (n_phi0 + j) in col_of_row]
    if held:
        s_imp = float(member.improper_phi_sigma(T))
        for j in held:
            c = col_of_row[n_phi0 + j]
            own[c], step[c] = s_imp, FD_STEP_ANGLE

    groups = []
    for rows in member.torsion_groups():
        rows = [j for j in rows if j not in tv_phi_rows and (n_phi0 + j) in col_of_row]
        if rows:
            groups.append(rows)
    s_sib = member.sibling_jitter_sigma(groups, T) if groups else []
    leaders = np.zeros(k, dtype=bool)
    directions = np.zeros((len(groups), k))
    for g, rows in enumerate(groups):
        for i, j in enumerate(rows):
            c = col_of_row[n_phi0 + j]
            directions[g, c] = 1.0 / scale[c]
            if i == 0:
                leaders[c] = True
                own[c], step[c] = 0.0, FD_STEP_ANGLE
            else:
                own[c], step[c] = s_sib[g], FD_STEP_ANGLE

    if np.isnan(own).any():
        raise RuntimeError(
            f'{member.smiles}: state column(s) {np.flatnonzero(np.isnan(own)).tolist()} were '
            f'given no thermal width -- a DoF row kind this tile does not know')

    # ONE FD PASS: every column along its own axis (per coordinate unit), then every rotation
    # (per radian). kappa is d^2 U / dq^2 in kcal/mol per coordinate unit squared.
    axes = np.diag(1.0 / scale)
    kappa = curvatures(member, np.concatenate([axes, directions]),
                       np.concatenate([step, np.full(len(groups), FD_STEP_ROT)]))
    k_col, k_rot = kappa[:k], kappa[k:]

    # sqrt(kT / kappa), never wider than the coordinate's own-term width: kappa below the
    # own term's 2k means the rest of the potential is locally concave there, and the own
    # term is the one curvature the force field states outright
    with np.errstate(divide='ignore', invalid='ignore'):
        fd = np.where(k_col > 0, np.sqrt(T / np.where(k_col > 0, k_col, 1.0)), np.inf)
    sigma = np.minimum(fd, own) / scale
    sigma[leaders] = 0.0

    floor = T / PHI_SIGMA_MAX ** 2
    w = np.where(k_rot > floor, np.sqrt(T / np.maximum(k_rot, floor)), PHI_SIGMA_MAX)
    return sigma, directions * w[:, None]


def curvatures(member, directions: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """``[M]`` second derivatives of the member's potential along each row of ``directions``.

    Central difference of ``potential_energy`` at T = 1 about the REFERENCE state (x = 0), with
    row m displaced by ``+/- steps[m] * directions[m]``; all 2M + 1 points in one call. The
    units are kcal/mol per (unit of ``directions``)^2.
    """
    M, k = directions.shape
    if M == 0:
        return np.zeros(0)
    d = torch.as_tensor(directions * steps[:, None], dtype=member.dtype, device=member.device)
    x = torch.cat([torch.zeros(1, k, dtype=member.dtype, device=member.device), d, -d])
    one = torch.tensor(1.0, dtype=member.dtype, device=member.device)
    with torch.no_grad():
        e = member.potential_energy(x, one).detach().double().cpu().numpy()
    return (e[1:M + 1] + e[M + 1:] - 2.0 * e[0]) / np.asarray(steps, dtype=np.float64) ** 2


class ThermalTile:
    """Per-member thermal widths at the ENERGY's state width, built once per identifier.

    Works on a single ``ConformerTorsions`` (one member, its own chart) and on a
    ``MultiConformerTorsions`` set, identity layout or carrier: each row is resolved to its
    member by ``member_groups`` -- the same identity checks the energy runs -- and takes that
    member's widths, placed at its carrier columns with zeros on its pads.
    """

    def __init__(self, energy):
        self.energy = energy
        self.temperature = float(energy.temperature)
        self._tables: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _member(self, ident: Optional[str]):
        members = getattr(self.energy, '_members', None)
        if not members or ident is None:
            return self.energy
        return members[ident]

    def widths(self, ident: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(sigma [K], rot [G, K])`` float64 on the CPU, cached by identifier."""
        if ident not in self._tables:
            member = self._member(ident)
            sigma, rot = member_widths(member, self.temperature)
            # A SET MEMBER KEEPS NO BATCH CACHE: `curvatures` evaluated it at 2M + 1 points,
            # and the set scores through its one-pass library, so that entry (1.7 MB a
            # member, on the energy's device) would never be read again
            if member is not self.energy:
                member.release_batch_cache()
            layout = getattr(self.energy, 'carrier', None)
            if layout is not None:
                cols = np.asarray(layout.cols[ident])
                K = int(layout.K)
                s_full = np.zeros(K)
                s_full[cols] = sigma
                r_full = np.zeros((rot.shape[0], K))
                r_full[:, cols] = rot
                sigma, rot = s_full, r_full
            self._tables[ident] = (torch.as_tensor(sigma), torch.as_tensor(rot))
        return self._tables[ident]

    def row_groups(self, batch, n: int) -> List[Tuple[Optional[str], torch.Tensor]]:
        """``[(identifier, row indices)]``; one group of every row for a single chart."""
        if hasattr(self.energy, 'member_groups') and getattr(self.energy, '_members', None):
            return self.energy.member_groups(batch)
        return [(None, torch.arange(n))]

    def draw(self, batch, state: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """``[n, K]`` displacement for ``state``'s rows, row i scaled by ``c[i]``.

        Pads are exactly 0 (their width is 0). Draws from torch's global generator on
        ``state``'s device, as the isotropic tile does.
        """
        n, K = state.shape
        out = torch.zeros_like(state)
        for ident, rows in self.row_groups(batch, n):
            rows = rows.to(state.device)
            sigma, rot = self.widths(ident)
            if int(sigma.numel()) != K:
                raise RuntimeError(f'thermal widths are {sigma.numel()} wide, the state {K}')
            sigma = sigma.to(device=state.device, dtype=state.dtype)
            rot = rot.to(device=state.device, dtype=state.dtype)
            z = torch.randn(len(rows), K, device=state.device, dtype=state.dtype)
            delta = z * sigma
            if rot.shape[0]:
                eta = torch.randn(len(rows), rot.shape[0], device=state.device,
                                  dtype=state.dtype)
                delta = delta + eta @ rot
            out[rows] = delta * c.to(state.dtype).index_select(0, rows)[:, None]
        return out
