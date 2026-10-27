"""Minimal reproductions of the failure modes the FD stress sweep (``stress``) found; each prints its evidence.

usage, from gfn_diffusion/energy_sampling, CPU (CUDA_VISIBLE_DEVICES=-1):
    python -m eval.fd_stress.repro [all | raw_rows | k_groups | neighbour_cap | unreal_cell | mxt_box]

The molecule is the sweep's (``D:/crystal_datasets/fd_stress_oct26/c1_all/molecule.pt``, read only).
"""
import sys
import warnings

import torch

MOL = 'D:/crystal_datasets/fd_stress_oct26/c1_all/molecule.pt'


def _mol():
    return torch.load(MOL, weights_only=False)['mol']


def raw_rows(sg=143):
    """FD start rows hand MXtalTools centres outside [0, 1) wherever ASU(Gamma+) leaves the unit cube; the builder
    clips them to [0, 1 - 1e-4], so the rows build another crystal (fd_starts.check_rows' own builder_dev_A says so)."""
    from eval.fd_search import fd_starts, run_fd_search
    mol = _mol()
    smp = fd_starts.FDStartSampler(sg, mol)
    run_fd_search.SG = sg  # fd_start_rows' check_rows reads this module global
    batch, cp, desc, info, draw, clipped = run_fd_search.fd_start_rows(smp, 8, 0, 0, run_fd_search.length_floor())
    print(f'[raw_rows] sg {sg}: canonical t min {float(desc.t.min()):.4f} max {float(desc.t.max()):.4f}; '
          f'rows aunit_centroid min {float(batch.aunit_centroid.min()):.4f}; check_rows builder_dev_A '
          f'{info["check"]["builder_dev_A"]:.3f} A (gate: < 1e-3)')


def k_groups(sg=7):
    """FDStartSampler refused sg 6-9 (K nontrivial) until 2026-10-26: fd_chart builds the chart without
    orientation_cover, and CrystalChart had no SO(3)/K chart. Since then the chart's orientation block is the K cut
    (crystal_fd charts.KCutHemisphereChart), so this prints the chart it builds."""
    from eval.fd_search import fd_starts
    try:
        smp = fd_starts.FDStartSampler(sg, _mol())
        ch = smp.chart
        print(f'[k_groups] sg {sg}: no exception; domain {ch.domain}, multiplicity {ch.multiplicity}, orientation '
              f'block {type(ch.orientation).__name__}')
    except NotImplementedError as e:
        print(f'[k_groups] sg {sg}: NotImplementedError: {e}')


def neighbour_cap():
    """Two descriptions of one R3 crystal (a = b = 3.54 A, c = 134 A, molecules overlapping) get different eLJ from
    analyze(['elj']); with construct_radial_graph's neighbour cap raised from its default 1000 they agree."""
    from eval.fd_stress import stress
    mol, sg = _mol(), 146
    A = torch.tensor([[3.5382414, 3.5382414, 134.4295959, 1.5707964, 1.5707964, 2.0943952, 0.1767214, 0.0533576, 0.5,
                       1.4691917, 0.6771685, 1.4148455]])
    B = torch.tensor([[3.5382414, 3.5382414, 134.4295959, 1.5707964, 1.5707964, 2.0943952, 0.543303, 0.1566119,
                       0.2294973, -3.8586652, 2.7274837, 0.8770041]])
    h = torch.tensor([1.0])
    ea = stress.elj_of(mol, sg, A, h, 1)
    eb = stress.elj_of(mol, sg, B, h, 1)
    ua, na = stress.elj_uncapped(mol, sg, A, h)
    ub, nb = stress.elj_uncapped(mol, sg, B, h)
    from eval.fd_search import fd_starts
    from mxtaltools.crystal_fd import canon
    ca = canon.canonicalize(fd_starts.rows_to_desc(sg, stress.rows(mol, sg, A, h)))
    cb = canon.canonicalize(fd_starts.rows_to_desc(sg, stress.rows(mol, sg, B, h)))
    dev, _ = stress.param_diff(ca.desc, cb.desc)
    print(f'[neighbour_cap] same crystal (canonical forms differ by {float(dev[0]):.2e}); analyze eLJ '
          f'{float(ea[0]):.2f} vs {float(eb[0]):.2f}; cap 1e5: {float(ua[0]):.2f} vs {float(ub[0]):.2f} '
          f'(largest neighbour count {int(max(na[0], nb[0]))})')


def unreal_cell():
    """An endpoint of MXtalTools' eLJ relaxation (sg 1, 60 rprop steps from an FD start) whose angles admit no real
    cell (alpha + beta + gamma > 360 deg): MXtalTools builds it from a signed-volume T_fc (det < 0) and reports a
    finite eLJ; crystal_fd's canonicalize refuses it as unreal_cell."""
    from eval.fd_search import fd_starts
    from eval.fd_stress import stress
    from mxtaltools.crystal_fd import canon
    mol = _mol()
    end = torch.tensor([[4.8281507, 8.1873045, 9.927352, 2.1293156, 1.8550529, 2.3945322, 0.7025822, 0.6261722,
                         0.4949124, -0.270642, 0.805338, -0.5595294]])
    b = stress.rows(mol, 1, end, torch.tensor([1.0]))
    ca = torch.cos(end[0, 3:6].double())
    f = 1 - (ca ** 2).sum() + 2 * ca.prod()
    print(f'[unreal_cell] angle sum {float(torch.rad2deg(end[0, 3:6].sum())):.1f} deg, (V/abc)^2 = {float(f):.4f}; '
          f'MXT det T_fc {float(torch.linalg.det(b.T_fc[0])):.1f}, cell_volume {float(b.cell_volume[0]):.1f}, eLJ '
          f'{float(stress.elj_of(mol, 1, end, torch.tensor([1.0]), 1)[0]):.2f}; crystal_fd status '
          f'{canon.canonicalize(fd_starts.rows_to_desc(1, b)).names()[0]}')


def mxt_box(sg=70):
    """canonicalize_aunit raises where the ASYM_UNITS box is not covered by G's images of every centre."""
    from eval.fd_stress import stress
    cov = stress.mxt_box_coverage(sg, torch.Generator().manual_seed(0))
    print(f'[mxt_box] sg {sg}: ASYM_UNITS box {cov["box"]} holds an image of {cov["covered"]:.3f} of uniform centres '
          f'(mean images {cov["mean_images"]:.3f})')
    from mxtaltools.crystal_search.crystal_opt_utils import canonicalize_aunit
    # a centre in the uncovered part: search a few random centres for one with no image in the box (bounded)
    from mxtaltools.constants.space_group_info import SYM_OPS
    import numpy as np
    from mxtaltools.crystal_fd import tables
    box = tables.group_table(sg).trainer_box.double()
    ops = torch.as_tensor(np.array(SYM_OPS[sg]), dtype=torch.float64)
    g = torch.Generator().manual_seed(1)
    for _ in range(200):
        t = torch.rand(3, generator=g, dtype=torch.float64)
        c = torch.einsum('mij,j->mi', ops[:, :3, :3], t) + ops[:, :3, 3]
        c = c - torch.floor(c)
        if not bool(((1 - c / box).amin(-1) >= -1e-5).any()):
            break
    p = torch.tensor([[12.0, 14.0, 16.0, 1.5707964, 1.5707964, 1.5707964, *t.tolist(), 0.3, 0.2, 0.1]])
    b = stress.rows(_mol(), sg, p, torch.tensor([1.0]))
    try:
        canonicalize_aunit(b)
        print(f'[mxt_box] centre {t.tolist()}: no exception')
    except ValueError as e:
        print(f'[mxt_box] centre {[round(x, 4) for x in t.tolist()]}: ValueError: {e}')


CASES = dict(raw_rows=raw_rows, k_groups=k_groups, neighbour_cap=neighbour_cap, unreal_cell=unreal_cell,
             mxt_box=mxt_box)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    torch.set_num_threads(2)
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for name, fn in CASES.items():
        if which in ('all', name):
            fn()
