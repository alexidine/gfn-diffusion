"""Fundamental-domain (FD) search starts for a Z' = 1 space group and molecule.

The law implemented. A description is first drawn uniformly on the canonical set (``crystal_fd``'s fundamental
domain) inside the chart's length box, with respect to the reference measure without its volume factor: Haar(dR) x
d^k t x mu_cell, mu_cell = ``'latent_v1_lebesgue'`` (flat in the free log lengths and free angles; monoclinic
d(log a) d(log b) d(log c) d(beta)). Its cell SIZE is then replaced by an isotropic rescale to a packing coefficient
drawn as the campaign's random stream draws its target (``crystal_search.utils.get_initial_state``,
``init_target_cp: std``). Pose (t, R, handedness) is uniform on the FD and independent of the cell. The SHAPE (the
log-length ratios and beta) is not the reference measure conditional on V: integrating the size out of a law that is
flat in (log a, log b, log c, beta) on the box gives each shape the density of mu_cell times the length of the
isotropic line (log a, log b, log c) + s (1, 1, 1) that lies inside the box. Shapes whose lengths span a wide ratio
leave the box sooner and are drawn less often, so the shape law depends on the box (default the trainer's at the
molecule radius; for NEHZOR a 0.62-24.3, b 2.47-97.0, c 0.82-32.3 A), and an FD-arm result holds for that box.

Why the volume factor V of the reference measure (Haar x V d^k t x mu_cell) is left out: the size is replaced by the
rescale, and with V the shape weight above would become the integral of V along that line, dominated by the largest
cell the box allows the shape, a stronger dependence on the box's upper edges than the line length.

Construction (``FDStartSampler.draw_shape_pose``). The chart is ``CrystalChart`` of the group (cell chart of its
family, ``PolytopeARChart`` centroid, ``OrientationChart('hemisphere')``) on a length box (default the trainer's,
``CrystalChart.for_latent_v1`` at the molecule radius). u ~ U(0, 1)^k is mapped block by block through the chart's AR
bounds (``charts.from_unit``); the log-density of the target relative to du is, per block,

- cell: sum log(hi - lo) of the cell coordinates' bounds + the cell chart's log|det| (monoclinic: log m(a, c) -
  log sin beta, ``CellChart.map``);
- centroid: the ``PolytopeARChart`` log|det| (sum log(hi - lo) of the t bounds; constant where every bound is);
- orientation: sum log(hi - lo) of (theta, phi, r) + the Haar log-density log(sin^2(r/2) sin theta) - log(2 pi^2).
  In sg 6-9 (K nontrivial; since 2026-10-26) the chart's orientation block is ``charts.KCutHemisphereChart``: r is
  signed and its bounds depend on the earlier rows, r in [-rho(theta, phi), rho(theta, phi)], rho in [pi/2, pi] (the
  fold's half of SO(3), R00 >= 0); ``from_unit`` fills it through those bounds, so log(hi - lo) of r is log(2 rho).

Their sum is ``CrystalChart.forward(y).logabsdet + from_unit(chart, u).log_volume``, and the chart's logabsdet is
``measure.log_reference_density(desc, chart, 'latent_v1_lebesgue') - log V`` (``verify_density`` checks both). The
blocks are independent, so each block is drawn by its own rejection (accept with exp(L - L_max)); the product is
exact. L_max is analytic for the monoclinic cell (m <= 1/sqrt(2) for the GLIDE and CENTRED classes, 1/2 for FREE, and
log m - 1/2 log(1 - s^2 m^2) increases in m and s) and for the hemisphere orientation (sin^2(r/2) sin theta <= 1;
on the K cut also 2 rho <= 2 pi, attained where u_x^2 >= 1/2, so the bound is the plain chart's and still tight; the
acceptance halves to 1 / (2 pi), the Haar mass of the cut, 1/2, over exp(L_max) = pi);
other cases use a sampled maximum plus a margin. Every proposal is checked against its bound (a value above it
raises: the draw would not be exact).

Rows for the search (``to_search_rows``): MolCrystalData built as run_search builds its seeds (the campaign
conformer, placeholders, then ``set_cell_parameters``), centres wrapped into [0, 1), with MolCrystalData's layout from
``CrystalDescription.crystal_params`` (rotation vectors on the +z hemisphere, handedness +1: the reference molecule is
the conformer aligned at handedness +1, which the builder's ``std_orientation`` alignment reproduces).
"""
import math
from typing import Optional

import numpy as np
import torch

from mxtaltools.crystal_fd import canon, charts, checks, measure, so3
from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef

F64 = torch.float64
CELL_MEASURE = 'latent_v1_lebesgue'
ORIENTATION = 'hemisphere'
#: the campaign random stream's size draw (crystal_search.utils.get_initial_state, init_target_cp 'std'):
#: target packing coefficient N(0.6226, 0.0447) clipped to [0.45, 0.95]
CP_MEAN, CP_STD, CP_LO, CP_HI = 0.6226, 0.0447, 0.45, 0.95
#: margin (nats) added to a sampled maximum where no analytic bound is built
EMPIRICAL_MARGIN = 0.5
#: proposals per accepted draw beyond which a block's rejection loop gives up (termination bound)
MAX_PROPOSALS_PER_DRAW = 2000


def fd_chart(sg: int, radius, length_box=None) -> charts.CrystalChart:
    """The FD chart of ``sg`` with the hemisphere orientation chart: the trainer's length box at ``radius`` (default)
    or an explicit (lo, hi) box in Angstrom. In sg 6-9 ``CrystalChart`` makes the orientation block the K cut
    (``charts.KCutHemisphereChart``; ``chart.k_cut`` True), so the chart is the fundamental domain there too."""
    if length_box is None:
        return charts.CrystalChart.for_latent_v1(int(sg), torch.as_tensor(radius, dtype=F64).reshape(-1), ORIENTATION)
    return charts.CrystalChart.for_length_box(int(sg), length_box, ORIENTATION)


def draw_target_cp(n: int, generator: torch.Generator) -> torch.Tensor:
    """[n] float64 target packing coefficients, drawn as the campaign's ``init_target_cp: std``."""
    return (torch.randn(n, generator=generator, dtype=F64) * CP_STD + CP_MEAN).clip(min=CP_LO, max=CP_HI)


def rescale_to_packing(desc: CrystalDescription, cp: torch.Tensor, mol_volume: float, sym_mult: int):
    """Isotropic rescale of each row's cell lengths to packing coefficient ``cp``: V = mol_volume * sym_mult / cp.

    Angles, fractional centroids, rotations and handedness are kept, so every reduction condition (homogeneous in
    the lengths: monoclinic -m(a/c) <= cos beta <= 0 and, FREE class, a <= c; the Niggli conditions) and the pose's
    place in ASU(Gamma+) x SO(3) are kept. Returns (desc, scale [B])."""
    d = desc.double()
    V = measure.cell_volume(d.cell)
    target = float(mol_volume) * float(sym_mult) / cp.to(F64)
    k = (target / V) ** (1.0 / 3.0)
    cell = d.cell.clone()
    cell[:, :3] = cell[:, :3] * k[:, None]
    return d.replace(cell=cell), k


class _Block:
    """One chart block (cell, centroid or orientation) with its log-density relative to du and its bound."""

    def __init__(self, name, chart_block, ndim, logp, lmax, exact_bound):
        self.name, self.block, self.ndim = name, chart_block, ndim
        self.logp, self.lmax, self.exact_bound = logp, lmax, exact_bound


class FDStartSampler:
    """Exact draws of shape and pose on the FD of ``sg``, uniform under the reference measure without V inside the
    length box, then sized to a packing coefficient (the module docstring states the resulting shape law).

    Parameters
    ----------
    sg : int
    mol : MolCrystalData
        The campaign conformer (``radius``, ``mol_volume`` and the conformer itself are read).
    length_box : (lo, hi), optional
        Angstrom; default the trainer's box at the molecule radius (``charts.latent_v1_length_box``).
    """

    def __init__(self, sg: int, mol, length_box=None, generator: Optional[torch.Generator] = None):
        self.sg = int(sg)
        self.mol = mol
        self.radius = float(mol.radius.reshape(-1)[0])
        self.mol_volume = float(mol.mol_volume.reshape(-1)[0])
        from mxtaltools.constants.space_group_info import SYM_OPS
        self.sym_mult = len(SYM_OPS[self.sg])
        self.chart = fd_chart(self.sg, self.radius, length_box)
        cell, cen, ori = self.chart.blocks
        g = generator if generator is not None else torch.Generator().manual_seed(12345)
        self.blocks = [self._cell_block(cell, g), self._centroid_block(cen, g), self._orientation_block(ori)]

    # ------------------------------------------------------------------ per-block densities and bounds

    def _cell_block(self, cell, g):
        def logp(u):
            y, logv = charts.from_unit(cell, u)
            _, ld = cell.map(y)
            return logv + ld, y
        if cell.family == 'monoclinic':
            from mxtaltools.common.sym_utils import MONO_FREE
            lo, hi = (v.reshape(-1, 3)[0].to(F64) for v in (cell.lo, cell.hi))
            free = cell.mono_class == MONO_FREE
            lo_c = torch.maximum(lo[2], lo[0]) if free else lo[2]
            # largest possible sum log(hi - lo): the coordinate ranges at their widest (FREE: log a's top is
            # min(log hi_a, log c) <= log hi_a); s in [0, 1]
            logv_max = (torch.log(torch.log(hi[2]) - torch.log(lo_c)) + torch.log(torch.log(hi[0]) - torch.log(lo[0]))
                        + torch.log(torch.log(hi[1]) - torch.log(lo[1])))
            m_max = 0.5 if free else 1 / math.sqrt(2)
            ld_max = math.log(m_max) - 0.5 * math.log(1 - m_max ** 2)  # log m - 1/2 log(1 - s^2 m^2) at s = 1
            return _Block('cell', cell, cell.ndim, logp, float(logv_max) + ld_max, True)
        return _Block('cell', cell, cell.ndim, logp, self._empirical_max(logp, cell.ndim, g), False)

    def _centroid_block(self, cen, g):
        def logp(u):
            t, ld = cen.map(u)
            return ld, u
        constant = all(float(st.lower_coef.abs().sum()) == 0 and float(st.upper_coef.abs().sum()) == 0
                       for st in cen.stages)
        if constant:
            _, ld = cen.map(torch.full((1, cen.ndim), 0.5, dtype=F64))
            return _Block('centroid', cen, cen.ndim, logp, float(ld[0]), True)
        return _Block('centroid', cen, cen.ndim, logp, self._empirical_max(logp, cen.ndim, g), False)

    def _orientation_block(self, ori):
        if ori.name != 'hemisphere':
            raise NotImplementedError('the analytic orientation bound is built for the hemisphere chart')

        def logp(u):
            y, logv = charts.from_unit(ori, u)
            _, ld = ori.map(y)
            return logv + ld, y
        # widest r range: 2 pi (plain hemisphere r in [0, 2 pi]; K cut r in [-rho, rho] with rho <= pi)
        r_width = 2 * math.pi
        if getattr(ori, 'cut', None) == 'K':
            rho_max = float(ori.rho(torch.tensor([math.pi / 2], dtype=F64), torch.tensor([0.0], dtype=F64))[0])
            if rho_max != math.pi:
                raise RuntimeError(f'K cut: rho at u_x = 1 is {rho_max!r}, not pi; the orientation bound needs it')
            r_width = 2 * rho_max
        logv = math.log(math.pi / 2) + math.log(2 * math.pi) + math.log(r_width)
        return _Block('orientation', ori, ori.ndim, logp, logv - math.log(2 * math.pi ** 2), True)

    @staticmethod
    def _empirical_max(logp, k, g, n=200_000):
        L, _ = logp(torch.rand(n, k, generator=g, dtype=F64))
        return float(L[torch.isfinite(L)].max()) + EMPIRICAL_MARGIN

    # ------------------------------------------------------------------ drawing

    def _reject(self, blk: _Block, n: int, g: torch.Generator, batch: int = 4096):
        """n accepted coordinates of one block (y) and their u; also (proposals, accepted)."""
        ys, us, tried, got = [], [], 0, 0
        cap = MAX_PROPOSALS_PER_DRAW * n + 10 * batch
        while got < n:
            if tried >= cap:  # termination bound: an acceptance this low means a broken bound, not bad luck
                raise RuntimeError(f'{blk.name}: {got} of {n} accepted after {tried} proposals')
            u = torch.rand(batch, blk.ndim, generator=g, dtype=F64)
            L, y = blk.logp(u)
            over = L > blk.lmax + 1e-9
            if bool(over.any()):
                raise RuntimeError(f'{blk.name}: log-density {float(L[over].max()):.6g} above the bound '
                                   f'{blk.lmax:.6g}; the rejection draw would not be exact')
            acc = torch.rand(batch, generator=g, dtype=F64) < torch.exp(L - blk.lmax)
            ys.append(y[acc])
            us.append(u[acc])
            tried += batch
            got += int(acc.sum())
        return torch.cat(ys)[:n], torch.cat(us)[:n], tried, got

    def draw_shape_pose(self, n: int, generator: torch.Generator):
        """n descriptions uniform on the FD under Haar x d^k t x mu_cell (no V), in the chart's length box.

        Returns (desc float64 [n], y [n, k], u [n, k], info dict with per-block acceptance rates)."""
        ys, us, info = [], [], {}
        for blk in self.blocks:
            y, u, tried, got = self._reject(blk, n, generator)
            ys.append(y)
            us.append(u)
            info[f'accept_{blk.name}'] = got / tried
            info[f'bound_{blk.name}'] = blk.lmax
            info[f'bound_{blk.name}_analytic'] = blk.exact_bound
        info['accept_joint_equivalent'] = float(np.prod([info[f'accept_{b.name}'] for b in self.blocks]))
        y = torch.cat(ys, 1)
        u = torch.cat(us, 1)
        fwd = self.chart.forward(y)
        return fwd.desc, y, u, info

    def draw(self, n: int, generator: torch.Generator):
        """n search starts: shape and pose on the FD, size from the campaign's packing-coefficient draw.

        Returns (desc float64 [n] sized, cp_target [n], scale [n], info)."""
        desc, _, _, info = self.draw_shape_pose(n, generator)
        cp = draw_target_cp(n, generator)
        sized, k = rescale_to_packing(desc, cp, self.mol_volume, self.sym_mult)
        return sized, cp, k, info

    # ------------------------------------------------------------------ rows

    def to_search_rows(self, desc: CrystalDescription):
        """A collated MolCrystalData batch of ``desc`` built as run_search builds its seeds (float32), every centre
        wrapped into [0, 1) (``wrap_centres``): the same crystal, not the FD representative where that has a centre
        outside the unit cube."""
        return rows_from_params(self.mol, self.sg, *desc_to_params(wrap_centres(desc)))


def wrap_centres(desc: CrystalDescription) -> CrystalDescription:
    """``desc`` with every centre wrapped into [0, 1). The builder clips a fractional centre to [0, 1 - 1e-4] instead
    of wrapping it, and ASU(Gamma+) leaves the unit cube in 84 groups (t down to -0.11 in sg 143), so an unwrapped FD
    row builds another crystal there (2026-10-26 stress sweep, D:/crystal_datasets/fd_stress_oct26/c1_all)."""
    t = desc.t - torch.floor(desc.t)
    return desc.replace(t=torch.where(t >= 1, t - 1, t))


def desc_to_params(desc: CrystalDescription):
    """([B, 12] float32 full_cell_parameters, [B] float32 handedness) in MolCrystalData's layout."""
    L, A, t, rv, h = desc.crystal_params()
    return torch.cat([L, A, t, rv], 1).float(), h.reshape(-1).float()


def rows_from_params(mol, sg: int, params: torch.Tensor, hand: torch.Tensor):
    """Collated MolCrystalData seeds with these parameters: placeholders as crystal_search.utils.init_samples_to_optim
    makes them (the conformer, sg, Z' = 1), then ``set_cell_parameters`` and the handedness (as run_search's hop
    batches set stored parameters)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    from mxtaltools.dataset_utils.utils import collate_data_list
    ones3, ones1 = torch.ones(3), torch.ones(1)
    proto = [MolCrystalData(molecule=mol.clone(), sg_ind=int(sg), aunit_handedness=ones1, cell_lengths=ones3,
                            cell_angles=ones3, aunit_centroid=ones3, aunit_orientation=ones3, skip_box_analysis=True,
                            max_z_prime=1, z_prime=1, do_box_analysis=True) for _ in range(len(params))]
    b = collate_data_list(proto)
    b.set_cell_parameters(params.to(torch.float32).clone())
    b.aunit_handedness = hand.to(b.aunit_handedness.dtype).reshape(b.aunit_handedness.shape).clone()
    b.box_analysis()
    return b


def rows_to_desc(sg: int, batch) -> CrystalDescription:
    """The float64 description of MolCrystalData rows' stored (float32) parameters."""
    return CrystalDescription.from_crystal_params(int(sg), batch.cell_lengths.detach(), batch.cell_angles.detach(),
                                                  batch.aunit_centroid.detach(), batch.aunit_orientation.detach(),
                                                  batch.aunit_handedness.detach().reshape(-1, 1))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def verify_density(sampler: FDStartSampler, y: torch.Tensor, u: torch.Tensor) -> dict:
    """The block log-densities against the chart and ``measure.log_reference_density``.

    Returns the largest absolute differences of (1) the sum of the three block log-densities against
    ``chart.forward(y).logabsdet + from_unit(chart, u).log_volume`` and (2) ``chart.forward(y).logabsdet`` against
    ``measure.log_reference_density(desc, chart, 'latent_v1_lebesgue') - log V``."""
    ch = sampler.chart
    offs = [0]
    for blk in sampler.blocks:
        offs.append(offs[-1] + blk.ndim)
    Lsum = sum(blk.logp(u[:, offs[i]:offs[i + 1]])[0] for i, blk in enumerate(sampler.blocks))
    _, logv = charts.from_unit(ch, u)
    fwd = ch.forward(y)
    lrd = measure.log_reference_density(fwd.desc, ch, CELL_MEASURE)
    logV = torch.log(measure.cell_volume(fwd.desc.cell))
    return dict(blocks_vs_chart=float((Lsum - (fwd.logabsdet + logv)).abs().max()),
                chart_vs_reference_minus_logV=float((fwd.logabsdet - (lrd - logV)).abs().max()))


def uniformity_checks(sampler: FDStartSampler, desc: CrystalDescription) -> dict:
    """KS p-values of statistics whose law under the target is known exactly (draws before the size rescale).

    beta_fraction: (beta - pi/2) / (arccos(-m) - pi/2), uniform given (a, c) (monoclinic only); log_b: uniform over
    its box (b enters no other bound; monoclinic); rotation angle omega in [0, pi], CDF (omega - sin omega) / pi;
    cos theta (hemisphere axis polar angle), uniform on [0, 1]; phi uniform; centroid coordinates uniform on their
    ASU(Gamma+) ranges (box groups only). On the K cut (sg 6-9, ``chart.k_cut``) R is Haar restricted to R00 >= 0:
    k_cut_R00, uniform on [0, 1] (R00 is uniform on [-1, 1] under Haar), and the three rotation statistics on the draws
    each replaced by its K image Q_K R with probability 1/2 (a coin of a fixed seed, independent of the draws), which
    is Haar."""
    from scipy.stats import kstest
    d = desc.double()
    out = {}
    cell = sampler.chart.cell
    if cell.family == 'monoclinic':
        a, b, c, beta = d.cell[:, 0], d.cell[:, 1], d.cell[:, 2], d.cell[:, 4]
        m = cell._mono_bound(a, c)
        frac = (beta - math.pi / 2) / (torch.arccos(-m) - math.pi / 2)
        out['beta_fraction'] = float(kstest(frac.numpy(), 'uniform').pvalue)
        lo, hi = (v.reshape(-1, 3)[0] for v in (cell.lo, cell.hi))
        lb = (torch.log(b) - torch.log(lo[1])) / (torch.log(hi[1]) - torch.log(lo[1]))
        out['log_b'] = float(kstest(lb.numpy(), 'uniform').pvalue)
    R = d.R[:, 0]
    if getattr(sampler.chart, 'k_cut', False):
        out['k_cut_R00'] = float(kstest(R[:, 0, 0].numpy(), 'uniform').pvalue)
        coin = torch.rand(len(R), generator=torch.Generator().manual_seed(2026_10_26), dtype=F64) < 0.5
        R = torch.where(coin[:, None, None], sampler.chart.orientation.k_image(R), R)
    omega = so3.matrix_to_axis_angle(R).norm(dim=1)
    out['rotation_angle'] = float(kstest(omega.numpy(), lambda w: (w - np.sin(w)) / np.pi).pvalue)
    hemi = so3.matrix_to_hemisphere(R)
    out['cos_theta'] = float(kstest(torch.cos(hemi[:, 0]).numpy(), 'uniform').pvalue)
    out['phi'] = float(kstest(((hemi[:, 1] + math.pi) / (2 * math.pi)).numpy(), 'uniform').pvalue)
    cen = sampler.chart.centroid
    if all(float(st.lower_coef.abs().sum()) == 0 and float(st.upper_coef.abs().sum()) == 0 for st in cen.stages) \
            and cen.basis is None:
        t = d.t[:, 0]
        for i, ax in enumerate(cen.order):
            st = cen.stages[i]
            lo_t, hi_t = float(st.lower_const.max()), float(st.upper_const.min())
            out[f't_{"xyz"[ax]}'] = float(kstest(((t[:, ax] - lo_t) / (hi_t - lo_t)).numpy(), 'uniform').pvalue)
    return out


def check_rows(sg: int, batch, desc_intended: CrystalDescription, cp_target: Optional[torch.Tensor] = None,
               n_builder: int = 8) -> dict:
    """Checks of search rows against the descriptions they were built from.

    - fd: every float32 row, its centres moved by whole cells back onto the intended description's (``to_search_rows``
      wraps them into [0, 1)), canonicalises to itself (``canon.fd_status`` at tol 1e-5, the float32 rounding), and
      the status names of its canonicalisation;
    - param_dev: largest |row - intended| over cell lengths (relative), angles, t (after the same whole-cell move) and
      R entries;
    - cp: largest |packing_coeff - cp_target|;
    - builder: on ``n_builder`` rows, the live builder's unit cell (``mol2ucell``, std_orientation, as the optimizer
      builds) against the float64 literal unit cell of the intended description with the conformer aligned at
      handedness +1 (``checks.unit_cell_fractional``): largest atom deviation, Angstrom.
    """
    out = {}
    di = desc_intended.double()
    d32 = rows_to_desc(sg, batch)
    d32 = d32.replace(t=d32.t + torch.round(di.t - d32.t))
    st = canon.fd_status(d32, tol=1e-5)
    out['n'] = d32.num_rows
    out['is_fd'] = int(st.is_fd.sum())
    names = [canon.STATUS[int(s)] for s in st.status.tolist()]
    out['status'] = {k: names.count(k) for k in sorted(set(names))}
    out['near_face'] = int(st.near_face.sum())
    dev_len = ((d32.cell[:, :3] - di.cell[:, :3]).abs() / di.cell[:, :3]).max()
    dev_ang = (d32.cell[:, 3:] - di.cell[:, 3:]).abs().max()
    dev_t = (d32.t - di.t).abs().max()
    dev_R = (d32.R - di.R).abs().max()
    out['param_dev'] = dict(length_rel=float(dev_len), angle=float(dev_ang), t=float(dev_t), R=float(dev_R))
    if cp_target is not None:
        out['cp_dev'] = float((batch.packing_coeff.double() - cp_target.double()).abs().max())
    if n_builder:
        out['builder_dev_A'] = builder_deviation(sg, batch, desc_intended, n_builder)
    return out


_NB = torch.tensor([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=F64)


def builder_deviation(sg: int, batch, desc: CrystalDescription, n: int = 8) -> float:
    """Largest atom deviation (Angstrom) over the first n rows: live builder unit cell vs literal unit cell."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    rows = batch.batch_to_list()[:n]
    b = collate_data_list([r.clone() for r in rows])
    ref_b = collate_data_list([r.clone() for r in rows])
    ref_b.orient_molecule(mode='standardized', target_handedness=torch.ones_like(ref_b.aunit_handedness))
    with torch.no_grad():
        b.mol2ucell(std_orientation=True)
    pos, bat = b.unit_cell_pos.double(), b.unit_cell_batch
    d = desc.double()[:len(rows)]
    worst = 0.0
    for i in range(len(rows)):
        sel = ref_b.batch == i
        X = MoleculeRef.from_coords(ref_b.pos[sel].double(), ref_b.z[sel])
        T = d[i].cell_matrix()[0]
        ref = checks.unit_cell_fractional(d[i], X)[0]
        got = torch.linalg.solve(T, pos[bat == i].T).T.reshape(-1, len(X.z), 3)
        df = got[:, None] - ref[None]
        df = df - torch.round(df)
        dist = torch.einsum('ij,abnkj->abnki', T, df[:, :, :, None] + _NB).norm(dim=-1).amin(-1).amax(-1)
        worst = max(worst, float(torch.maximum(dist.amin(1).amax(), dist.amin(0).amax())))
    return worst
