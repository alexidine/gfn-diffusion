"""Stress sweep of the fundamental-domain (FD) machinery with MXtalTools' builder, eLJ and search plumbing, Z' = 1.

A correctness and robustness sweep over space groups, not a performance comparison. Per group:

1. Starts. ``fd_starts.FDStartSampler`` (shape and pose uniform on the FD under the reference measure without V, size
   from the campaign's packing-coefficient draw) through ``run_fd_search.fd_start_rows`` (the draw, ``check_rows`` on
   the draw, then the 3 A length floor of ``length_floor``). ``fd_start_rows`` reads the module global
   ``run_fd_search.SG`` for ``check_rows``; this driver sets it to the group before each call. The groups whose K is
   nontrivial (sg 6-9) draw on the K-cut chart since 2026-10-26 (``fd_chart``: ``charts.KCutHemisphereChart``, the
   fundamental domain; ``rec['k_cut']``); before that ``fd_chart`` refused them, and should the sampler still refuse
   a group the refusal is recorded and the draw falls back to ``DoubleCoverSampler`` (the same sampler on the
   double-cover chart, ``orientation_cover='double'``, then ``canon.canonicalize``: uniform on the cover, so uniform
   on the FD after the K fold). Each block's bound is also probed on 20,000 fresh proposals (largest log-density
   minus the bound; a positive value is a violation).
2. Checks on the N floored rows (float32, as the search stores them):
   a. canonical: ``canon.fd_status(rows_to_desc(rows), tol=1e-5)``; reported over all rows, over the rows the floor
      left alone (``a_unfloored``: the real check; the floor may take a row out of the reduced set), and on the draw
      before the floor (``check_rows``' ``is_fd``);
   b. builds: eLJ (``analyze(['elj'])``, cutoff 10) finite, ``mol2ucell`` holds sym_mult molecules, and per row the
      live builder's unit cell against the float64 literal unit cell of the row's description
      (``fd_starts.builder_deviation``, pass below 1e-3 A, gate 3's bound);
   c. symmetry: spglib on the built unit cell (all atoms, symprec 1e-2 A; also 1e-3 A recorded) returns sg. A
      different number is classified by applying every operator of SYM_OPS[sg] to the built cell (largest atom
      mismatch, A): all hold within 1e-2 A -> ``supergroup``, else ``wrong``. The operator check also runs on the
      first 8 rows of every group;
   d. re-description: each canonical row is re-described by a random element of Gamma (the table's full Euclidean
      normaliser, improper elements included, which flip h), a random lattice shift in [-2, 2]^3, a random shift
      along every free axis, and a random setting change the canonicaliser absorbs (sg 1-2 a unimodular basis with
      entries in {-1, 0, 1}; sg 3-15 an ac-plane change the setting class allows, with its origin shift; sg 16-74 a
      random allowed axis order with its shift; none above). Checked: (d64) ``canonicalize`` of the float64
      re-description returns the original canonical description (``identical``: parameter difference below 1e-7;
      else ``equivalent``: FD identity distance of ``fd_analysis.fd_distance.FDMetric`` below 1e-4 A); in a Sohncke
      group an improper element gives the mirror-image crystal, so the row must come back ``off_sheet`` and its
      ``enantiomorph`` must canonicalise to the original; (d32) the same from the float32 rows the re-description
      builds (parameter 1e-4, FD distance 1e-2 A); (elj) those rows' eLJ equals the original row's (relative 1e-4
      of max(|E|, 1)); the canonical form's own rows likewise (``elj_canon``); the re-described rows' builder
      deviation as in (b);
   e. chart: ``chart.inverse`` then ``chart.forward`` of the canonical description within 1e-8 (cell lengths
      relative, angles, t, R), forward log|det| and ``measure.log_reference_density`` finite, inverse log|det| =
      -forward's; the pose coordinates (centroid, orientation) inside the chart's AR bounds (the cell coordinates
      leave the length box after the size rescale and the floor, so their bound count is recorded, not judged). On
      the K cut (sg 6-9) also ``k_image_outside``: the K image of each canonical row charts with its signed r outside
      [-rho, rho], except within the fold's tie band |R00| <= 5e-10, where both images chart at the fold's choice
      (recorded, not part of the pass).
3. Short relaxations of the first ``n_relax`` rows: MXtalTools' ``optimize_crystal_parameters`` with the campaign
   random stream's stage-1 settings (``campaign.stream_config('random').opt[0]``, eLJ, rprop, wrap) at
   ``max_num_steps`` 60. MXtalTools' end-of-run ``canonicalize_aunit`` is wrapped for the call (the module global in
   ``crystal_opt_utils`` is replaced and restored; MXtalTools is not edited): the wrapper keeps the endpoint before
   it, and a ValueError it raises is recorded instead of losing the run. Per endpoint: does ``canonicalize_aunit``
   raise on the row alone, and if not does it keep the crystal (eLJ relative 1e-4, FD distance of the crystal_fd
   canonical forms before and after 1e-2 A, the centre inside the ASYM_UNITS box); does ``canon.canonicalize`` of
   the endpoint succeed (status ``canonical``); is the endpoint's eLJ unchanged by the fold; is the folded endpoint
   its own canonical form in float64 (``fd_status``, 1e-9) and after the float32 round trip through rows (1e-5).

Every stage runs inside a guard: an exception is recorded (stage, type, message, traceback tail) and that stage's
pass counts stay 0, so a failure is never counted as a pass. Groups are written one JSON each (never overwritten;
``--resume`` adds the missing ones), then ``report`` writes ``table.csv``, ``summary.md`` and ``failures.md``.

usage, from gfn_diffusion/energy_sampling, CPU (CUDA_VISIBLE_DEVICES=-1):
    python -m eval.fd_stress.stress run --name NAME [--molecule c1|mipcas] [--groups 1-230] [--n 64] [--n-relax 16]
        [--seed 0] [--threads 4] [--resume]
    python -m eval.fd_stress.stress report --name NAME
"""
import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

GFN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_ROOT = 'D:/crystal_datasets/fd_stress_oct26'
F64 = torch.float64

# tolerances (module docstring)
CANON_TOL32 = 1e-5
BUILDER_TOL_A = 1e-3
SYMPREC = 1e-2
SYMPREC_STRICT = 1e-3
OPS_TOL_A = 1e-2
OPS_ROWS = 8
REDESC_PARAM_TOL64 = 1e-7
REDESC_FD_TOL64 = 1e-4
REDESC_PARAM_TOL32 = 1e-4
REDESC_FD_TOL32 = 1e-2
ELJ_RTOL = 1e-4
ELJ_ATOL = 1e-2  # absolute floor: float32 summation noise of the pair sum, measured up to ~7e-3 at |E| ~ 1e3
NEIGHBOUR_CAP_DIAG = 100_000  # diagnostic re-evaluation without MXtalTools' construct_radial_graph cap of 1000
NOISE_FACTOR = 3.0  # an eLJ mismatch within this multiple of the spread over 6 proper re-descriptions is float32 noise
CHART_TOL = 1e-8
BOUND_PROBE = 20_000
RELAX_STEPS = 60
SETTING_TRIES = 30
CLIP_EDGE = 1 - 1e-4  # the live builder clips fractional centres to [0, 1 - 1e-4]

MIPCAS_GROUPS = (1, 2, 4, 9, 14, 15, 19, 33, 61, 92, 148, 169, 205, 227)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

class Guard:
    """Runs stages, recording exceptions instead of raising."""

    def __init__(self, log):
        self.exceptions = []
        self.log = log

    def run(self, stage, fn, *a, **k):
        try:
            return True, fn(*a, **k)
        except Exception as e:  # noqa: BLE001 -- recorded, never counted as a pass
            tb = traceback.format_exc().strip().splitlines()
            rec = dict(stage=stage, type=type(e).__name__, message=str(e)[:600], traceback=tb[-14:])
            self.exceptions.append(rec)
            self.log(f'    EXCEPTION in {stage}: {type(e).__name__}: {str(e)[:200]}')
            return False, None


def _f(x):
    """JSON-safe float."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else (None if math.isnan(x) else (1e300 if x > 0 else -1e300))


def _max(t):
    t = torch.as_tensor(t)
    return None if t.numel() == 0 else _f(t.max())


def param_diff(a, b):
    """[B] largest of |dL|/L, |d angle|, |dt| (wrapped to [-1/2, 1/2)), |dR| between two Z' = 1 descriptions; and
    [B] bool h equal."""
    a, b = a.double(), b.double()
    dl = ((a.cell[:, :3] - b.cell[:, :3]).abs() / b.cell[:, :3].abs()).amax(1)
    da = (a.cell[:, 3:] - b.cell[:, 3:]).abs().amax(1)
    dt = a.t[:, 0] - b.t[:, 0]
    dt = (dt - torch.round(dt)).abs().amax(1)
    dR = (a.R[:, 0] - b.R[:, 0]).abs().reshape(len(dl), -1).amax(1)
    return torch.stack([dl, da, dt, dR], 1).amax(1), a.h[:, 0] == b.h[:, 0]


def elj_rel(e, ref):
    """|e - ref| / max(|ref|, ELJ_ATOL / ELJ_RTOL): the pass test is elj_rel <= ELJ_RTOL, i.e. |dE| <= max(ELJ_RTOL |E|,
    ELJ_ATOL)."""
    e, ref = torch.as_tensor(e, dtype=F64), torch.as_tensor(ref, dtype=F64)
    return (e - ref).abs() / ref.abs().clamp(min=ELJ_ATOL / ELJ_RTOL)


# ---------------------------------------------------------------------------
# molecule and samplers
# ---------------------------------------------------------------------------

def radius_of_gyration(mol):
    pos, z = mol.pos.double(), mol.z
    X = pos - pos[z > 1].mean(0)
    return float(X.pow(2).sum(1).mean().sqrt())


def make_double_cover_sampler(sg, mol, generator):
    """FDStartSampler on the sg 6-9 double-cover chart; ``draw`` folds K out with ``canon.canonicalize``."""
    from eval.fd_search import fd_starts
    from mxtaltools.crystal_fd import canon, charts
    from mxtaltools.constants.space_group_info import SYM_OPS

    class DoubleCoverSampler(fd_starts.FDStartSampler):
        def __init__(self, sg, mol, generator):  # noqa: the parent's __init__ builds the K-refused chart
            self.sg, self.mol = int(sg), mol
            self.radius = float(mol.radius.reshape(-1)[0])
            self.mol_volume = float(mol.mol_volume.reshape(-1)[0])
            self.sym_mult = len(SYM_OPS[self.sg])
            self.chart = charts.CrystalChart.for_latent_v1(
                self.sg, torch.as_tensor(self.radius, dtype=F64).reshape(-1), fd_starts.ORIENTATION,
                orientation_cover='double')
            cell, cen, ori = self.chart.blocks
            self.blocks = [self._cell_block(cell, generator), self._centroid_block(cen, generator),
                           self._orientation_block(ori)]

        def draw(self, n, generator):
            desc, cp, k, info = super().draw(n, generator)
            c = canon.canonicalize(desc)
            names = c.names()
            info = dict(info, double_cover_fold={s: names.count(s) for s in sorted(set(names))},
                        double_cover_moved=int(c.moved.sum()))
            return c.desc, cp, k, info

    return DoubleCoverSampler(sg, mol, generator)


def bound_probe(sampler, g):
    """Per block: largest log-density over BOUND_PROBE fresh proposals minus the bound (<= 0 holds), violations."""
    out = {}
    for blk in sampler.blocks:
        u = torch.rand(BOUND_PROBE, blk.ndim, generator=g, dtype=F64)
        L, _ = blk.logp(u)
        fin = torch.isfinite(L)
        out[blk.name] = dict(max_minus_bound=_f(L[fin].max() - blk.lmax) if bool(fin.any()) else None,
                             violations=int((L[fin] > blk.lmax + 1e-9).sum()), nonfinite=int((~fin).sum()),
                             analytic=bool(blk.exact_bound))
    return out


# ---------------------------------------------------------------------------
# building and energies
# ---------------------------------------------------------------------------

def energy_chunk(sym_mult):
    return 16 if sym_mult <= 16 else (8 if sym_mult <= 48 else 4)


def rows(mol, sg, params, hand):
    from eval.fd_search.fd_starts import rows_from_params
    return rows_from_params(mol, sg, params.float(), hand.float())


def elj_of(mol, sg, params, hand, chunk):
    """[B] float64 eLJ of rows built from (params, hand), recomputed (never a stored attribute)."""
    out = []
    with torch.no_grad():
        for lo in range(0, len(params), chunk):
            b = rows(mol, sg, params[lo:lo + chunk], hand[lo:lo + chunk])
            out.append(b.analyze(['elj'], cutoff=10, supercell_size=10)['elj'].detach().double().reshape(-1))
    return torch.cat(out) if out else torch.zeros(0, dtype=F64)


def elj_uncapped(mol, sg, params, hand):
    """Diagnostic: per row (eLJ with the pair search's neighbour cap raised to NEIGHBOUR_CAP_DIAG, the largest number
    of image neighbours an asymmetric-unit atom has within the cutoff). analyze() builds the pair list with
    construct_radial_graph's default cap of 1000 per atom, which truncates without warning."""
    out_e, out_n = [], []
    with torch.no_grad():
        for i in range(len(params)):
            b = rows(mol, sg, params[i:i + 1], hand[i:i + 1])
            cl = b.mol2cluster(10, 10, std_orientation=True)
            cl.construct_radial_graph(cutoff=10, max_num_neighbors=NEIGHBOUR_CAP_DIAG)
            cl._init_computes(override=True)
            out_e.append(float(cl.compute(requests=['elj'])['elj'].reshape(-1)[0]))
            ed = cl.edges_dict['edge_index_inter']
            out_n.append(int(torch.bincount(ed[1]).max()) if ed.shape[1] else 0)
    return torch.tensor(out_e, dtype=F64), torch.tensor(out_n)


def cap_diagnosis(mol, sg, pa, ha, pb, hb):
    """For paired failing rows a, b (two descriptions of one crystal): dict with the uncapped energies, whether they
    agree (ELJ test) and the neighbour counts (>= 1000 means the default cap truncated)."""
    Ea, na = elj_uncapped(mol, sg, pa, ha)
    Eb, nb = elj_uncapped(mol, sg, pb, hb)
    agree = elj_rel(Eb, Ea) <= ELJ_RTOL
    capped = (na >= 1000) | (nb >= 1000)
    return dict(n=len(Ea), agree_uncapped=int(agree.sum()), hit_cap=int(capped.sum()),
                explained_by_cap=int((agree & capped).sum()), max_neighbours=int(torch.maximum(na, nb).max()))


def desc_params(desc):
    """(params [B, 12] float32, hand [B]) of a description with centroids wrapped to [0, 1); and [B] bool, a centre
    in the builder's clip band (1 - 1e-4, 1)."""
    from eval.fd_search.fd_starts import desc_to_params
    t = desc.t - torch.floor(desc.t)
    t = torch.where(t >= 1, t - 1, t)
    d = desc.replace(t=t)
    p, h = desc_to_params(d)
    clip = (p[:, 6:9].double() > CLIP_EDGE).any(1)
    return p, h, clip


def unit_cells(mol, sg, params, hand, chunk):
    """Per row: (T_fc [3, 3] float64, Cartesian unit-cell positions [n, 3] float64)."""
    out = []
    with torch.no_grad():
        for lo in range(0, len(params), chunk):
            b = rows(mol, sg, params[lo:lo + chunk], hand[lo:lo + chunk])
            b.mol2ucell(std_orientation=True)
            pos, ub = b.unit_cell_pos.double(), b.unit_cell_batch
            for i in range(b.num_graphs):
                out.append((b.T_fc[i].double(), pos[ub == i]))
    return out


def spglib_number(T, pos, numbers, symprec):
    import spglib
    frac = torch.linalg.solve(T, pos.T).T.numpy()
    ds = spglib.get_symmetry_dataset((T.T.numpy(), frac % 1.0, numbers), symprec=symprec)
    return None if ds is None else int(ds.number)


def ops_mismatch(T, pos, numbers, sg):
    """Largest distance (A) from an atom's image under an operator of SYM_OPS[sg] to the nearest like atom."""
    from scipy.spatial import cKDTree
    from mxtaltools.constants.space_group_info import SYM_OPS
    frac = torch.linalg.solve(T, pos.T).T.numpy()
    frac = frac - np.floor(frac)
    tree = cKDTree(frac % 1.0, boxsize=1.0 + 1e-12)
    Tn = T.numpy()
    worst = 0.0
    for op in SYM_OPS[sg]:
        op = np.asarray(op, dtype=np.float64)
        img = frac @ op[:3, :3].T + op[:3, 3]
        img = img - np.floor(img)
        _, j = tree.query(np.clip(img, 0, 1.0), k=1)
        d = img - frac[j]
        d = d - np.round(d)
        dist = np.linalg.norm(d @ Tn.T, axis=1)
        dist = np.where(numbers[j] == numbers, dist, np.inf)
        worst = max(worst, float(dist.max()))
    return worst


def min_atom_distance(T, pos):
    """Smallest distance (A) between two distinct atoms of a unit cell, periodic (nearest-image through rounding of
    fractional differences; exact for distances well below the shortest lattice vector)."""
    from scipy.spatial import cKDTree
    frac = torch.linalg.solve(T, pos.T).T.numpy()
    frac = frac - np.floor(frac)
    tree = cKDTree(frac, boxsize=1.0 + 1e-12)
    _, j = tree.query(np.clip(frac, 0, 1.0), k=2)
    d = frac - frac[j[:, 1]]
    d = d - np.round(d)
    return float(np.linalg.norm(d @ T.numpy().T, axis=1).min())


def mxt_box_coverage(sg, g, n=20_000):
    """MXtalTools' canonicalize_aunit rule on n uniform centres: the fraction with an image under SYM_OPS[sg] in the
    ASYM_UNITS box (margin >= -1e-5, as it tests) and the mean number of such images. A covered fraction below 1 means
    the box is not a fundamental domain of G and canonicalize_aunit must raise on that fraction of crystals."""
    from mxtaltools.constants.space_group_info import SYM_OPS
    from mxtaltools.crystal_fd import tables
    box = tables.group_table(sg).trainer_box.to(F64)
    ops = torch.as_tensor(np.array(SYM_OPS[sg]), dtype=F64)
    t = torch.rand(n, 3, generator=g, dtype=F64)
    c = torch.einsum('mij,nj->nmi', ops[:, :3, :3], t) + ops[:, :3, 3]
    c = c - torch.floor(c)
    inside = (1 - c / box).amin(-1) >= -1e-5
    return dict(covered=float(inside.any(1).double().mean()), mean_images=float(inside.sum(1).double().mean()),
                box=[float(x) for x in box])


def noise_diagnosis(mol, sg, params, hand, g, k=6):
    """Spread (max - min) of the eLJ of each row over k random proper re-descriptions (Gamma+ element and lattice
    shift, float32 rows): the float32 noise floor of that crystal's eLJ under re-description."""
    from mxtaltools.crystal_fd import ops, tables
    from eval.fd_search import fd_starts
    tab = tables.group_table(sg)
    d = fd_starts.rows_to_desc(sg, rows(mol, sg, params, hand))
    Es = []
    for _ in range(k):
        j = tab.gamma_plus[torch.randint(len(tab.gamma_plus), (d.num_rows,), generator=g)]
        di = ops.apply(d, tab.gamma_W[j], tab.gamma_w[j])
        p, h, _ = desc_params(di)
        Es.append(elj_of(mol, sg, p, h, 4))
    E = torch.stack(Es)
    return E.amax(0) - E.amin(0)


# ---------------------------------------------------------------------------
# re-description
# ---------------------------------------------------------------------------

_UNIMOD = None


def unimodular():
    global _UNIMOD
    if _UNIMOD is None:
        allm = np.array(list(itertools.product((-1, 0, 1), repeat=9)), dtype=np.int64).reshape(-1, 3, 3)
        _UNIMOD = torch.as_tensor(allm[np.round(np.linalg.det(allm)) == 1])
    return _UNIMOD


def redescribe(sg, c, g):
    """A random re-description of each row of the canonical float64 description ``c`` (module docstring, check d).

    Returns (desc float64, info dict of per-row tensors / lists)."""
    from mxtaltools.crystal_fd import axis_order, ops, tables
    tab = tables.group_table(sg)
    B = c.num_rows
    k = torch.randint(len(tab.gamma_W), (B,), generator=g)
    d = ops.apply(c, tab.gamma_W[k], tab.gamma_w[k])
    n = torch.randint(-2, 3, (B, 3), generator=g).to(F64)
    fs = torch.rand(B, 3, generator=g, dtype=F64) * tab.free.to(F64)
    d = d.replace(t=d.t + (n + fs)[:, None])
    setting = ['none'] * B
    if sg <= 2:
        U = unimodular()
        N = U[torch.randint(len(U), (B,), generator=g)]
        d = ops.change_basis(d, N)
        setting = [f'N={N[i].tolist()}' for i in range(B)]
    elif sg <= 15:
        parts = []
        for i in range(B):
            di = d[i]
            for _ in range(SETTING_TRIES):  # bounded: about half the 2x2 changes are in a class's setting group
                M = torch.randint(-1, 2, (2, 2), generator=g)
                e = int(round(float(torch.linalg.det(M.double()))))
                if abs(e) != 1:
                    continue
                N = torch.zeros(3, 3, dtype=torch.long)
                N[0, 0], N[2, 0], N[0, 2], N[2, 2], N[1, 1] = M[0, 0], M[1, 0], M[0, 1], M[1, 1], e
                try:
                    di = ops.change_basis(di, N)
                except ValueError:
                    continue
                setting[i] = f'N={N.tolist()}'
                break
            parts.append(di)
        from mxtaltools.crystal_fd.description import CrystalDescription
        d = CrystalDescription.cat(parts)
    elif sg <= 74:
        ao = axis_order.axis_orders(sg)
        j = torch.randint(len(ao.N), (B,), generator=g)
        d = ops.change_basis(d, ao.N[j], ao.s[j])
        setting = [f'order={int(x)}' for x in j]
    det = tab.gamma_det[k]
    return d, dict(element=k, det=det, lattice=n, free_shift=fs, setting=setting)


def compare_canonical(sg, c_ref, d, metric, sohncke, improper, param_tol, fd_tol):
    """canonicalize(d) against the canonical reference c_ref, row by row (module docstring, check d).

    Returns dict of [B] tensors: ok, identical, equivalent, param_dev, fd_dist, status_ok, h_ok, path."""
    from mxtaltools.crystal_fd import canon
    from mxtaltools.crystal_fd.enantiomorph import enantiomorph
    B = d.num_rows
    c = canon.canonicalize(d)
    names = c.names()
    desc = c.desc
    expect_off = sohncke & improper
    status_ok = torch.tensor([(nm == 'off_sheet') if bool(expect_off[i]) else (nm == 'canonical')
                              for i, nm in enumerate(names)])
    path = ['direct'] * B
    if bool(expect_off.any()):
        idx = torch.nonzero(expect_off).flatten()
        ok_off = idx[torch.tensor([names[int(i)] == 'off_sheet' for i in idx], dtype=torch.bool)]
        if len(ok_off):
            e = canon.canonicalize(enantiomorph(d[ok_off]))
            en = e.names()
            for j, i in enumerate(ok_off.tolist()):
                path[i] = 'enantiomorph'
                if en[j] != 'canonical':
                    status_ok[i] = False
            parts_t, parts_R, parts_h, parts_cell = desc.t.clone(), desc.R.clone(), desc.h.clone(), desc.cell.clone()
            parts_t[ok_off], parts_R[ok_off], parts_h[ok_off] = e.desc.t, e.desc.R, e.desc.h
            parts_cell[ok_off] = e.desc.cell
            desc = desc.replace(t=parts_t, R=parts_R, h=parts_h, cell=parts_cell)
    pdev, h_ok = param_diff(desc, c_ref)
    fd = fd_distance(sg, metric, desc, c_ref)
    identical = status_ok & h_ok & (pdev <= param_tol)
    equivalent = status_ok & h_ok & ~identical & (fd <= fd_tol)
    return dict(ok=identical | equivalent, identical=identical, equivalent=equivalent, param_dev=pdev, fd_dist=fd,
                status_ok=status_ok, h_ok=h_ok, names=names, path=path)


def fd_distance(sg, metric, a, b):
    """[B] FD identity distance (A) between paired Z' = 1 descriptions (both canonical)."""
    def can(x):
        x = x.double()
        return dict(cell=x.cell, t=x.t[:, 0], R=x.R[:, 0])
    A, Bp = metric.prepare(can(a)), metric.prepare(can(b))
    idx = torch.arange(a.num_rows)
    return metric.pairs(A, idx, Bp, idx)


# ---------------------------------------------------------------------------
# one group
# ---------------------------------------------------------------------------

def run_group(sg, mol, rg, n, n_relax, seed, out_dir, log):
    from eval.fd_search import campaign, fd_starts, run_fd_search
    from mxtaltools.crystal_fd import canon, charts, io_mxt, measure, tables
    from mxtaltools.constants.space_group_info import SYM_OPS
    sys.path.insert(0, GFN_ROOT)
    from energy_sampling.eval.fd_analysis.fd_distance import FDMetric

    t_group = time.time()
    G = Guard(log)
    tab = tables.group_table(sg)
    sym_mult = len(SYM_OPS[sg])
    sohncke = io_mxt.is_sohncke(sg)
    chunk = energy_chunk(sym_mult)
    rec = dict(sg=sg, symbol=tab.symbol, lattice_type=tab.lattice_type, cell_family=charts.cell_family(sg),
               sym_mult=sym_mult, sohncke=sohncke, n_gamma=len(tab.gamma_W), n_gamma_plus=len(tab.gamma_plus),
               n_gamma_improper=int((tab.gamma_det < 0).sum()), free_axes=int(tab.free.sum()),
               n_kernel=len(tab.kernel), box_coverage=str(tab.box_coverage), n=n, n_relax=n_relax)
    metric = FDMetric(sg, rg)
    g = torch.Generator().manual_seed(int(seed) * 7919 + sg)
    ok, cov = G.run('mxt_box_coverage', mxt_box_coverage, sg, torch.Generator().manual_seed(sg))
    rec['mxt_box'] = cov

    # 1. sampler and starts -------------------------------------------------------------------------------
    t0 = time.time()
    ok, sampler = G.run('sampler', fd_starts.FDStartSampler, sg, mol, generator=g)
    rec['sampler'] = 'fd'
    if not ok:
        ok, sampler = G.run('sampler_double_cover', make_double_cover_sampler, sg, mol, g)
        rec['sampler'] = 'double_cover_fallback' if ok else 'failed'
    if sampler is None:
        rec['exceptions'] = G.exceptions
        rec['wall_s'] = time.time() - t_group
        return rec
    ch = sampler.chart
    rec['k_cut'] = bool(getattr(ch, 'k_cut', False))
    rec['cell_dims'], rec['centroid_dims'], rec['orientation_dims'] = (b.ndim for b in ch.blocks)
    ok, bp = G.run('bound_probe', bound_probe, sampler, g)
    rec['bound_probe'] = bp
    run_fd_search.SG = sg  # fd_start_rows' check_rows reads this module global
    floor = run_fd_search.length_floor()
    ok, res = G.run('fd_start_rows', run_fd_search.fd_start_rows, sampler, n, seed, sg, floor)
    rec['t_draw_s'] = time.time() - t0
    if not ok:
        rec['exceptions'] = G.exceptions
        rec['wall_s'] = time.time() - t_group
        return rec
    batch, cp, desc_draw, info, draw, clipped = res
    rec['draw_info'] = {k: (v if not isinstance(v, float) else _f(v)) for k, v in info.items()}
    floored = clipped.any(1)
    rec['n_floored'] = int(floored.sum())
    params_raw = batch.full_cell_parameters().detach().float().clone()
    hand0 = batch.aunit_handedness.detach().reshape(-1).float().clone()
    d32 = fd_starts.rows_to_desc(sg, batch)
    d32 = d32.replace(t=d32.t + torch.round(desc_draw.t.double() - d32.t))  # to_search_rows wraps centres into [0, 1)
    del batch
    # the rows with every centre wrapped into [0, 1) (the builder clips a centre outside [0, 1 - 1e-4] instead of
    # wrapping it); checks b-e and the references for d run on these, the raw rows' build is checked apart (b_raw)
    params0, _, clip0 = desc_params(d32)
    rec['n_t_outside_unit'] = int(((params_raw[:, 6:9] < 0) | (params_raw[:, 6:9] >= 1)).any(1).sum())
    rec['n_clip_band'] = int(clip0.sum())
    d32w = fd_starts.rows_to_desc(sg, rows(mol, sg, params0, hand0))

    def ex(i, p, h, **kw):
        return dict(row=int(i), params=[round(float(x), 7) for x in p[i]], hand=float(h[i]), **kw)

    # a. canonical ------------------------------------------------------------------------------------------
    a = dict(pass_all=0, pass_unfloored=0, n_unfloored=int((~floored).sum()), status={}, worst_delta=None,
             draw_is_fd=info.get('check', {}).get('is_fd', 0))
    c32 = None
    ok, c32 = G.run('canonicalize_rows', canon.canonicalize, d32)
    if ok:
        st_ok, st = G.run('fd_status_rows', canon.fd_status, d32, tol=CANON_TOL32)
        names = c32.names()
        a['status'] = {s: names.count(s) for s in sorted(set(names))}
        pdev, h_ok = param_diff(c32.desc, d32)
        a['worst_delta'] = _max(pdev)
        a['worst_delta_unfloored'] = _max(pdev[~floored])
        a['near_face'] = int((c32.near_face != 0).sum())
        if st_ok:
            a['pass_all'] = int(st.is_fd.sum())
            a['pass_unfloored'] = int(st.is_fd[~floored].sum())
            a['fail_examples_unfloored'] = [ex(i, params_raw, hand0, status=names[i])
                                            for i in torch.nonzero(~st.is_fd & ~floored).flatten().tolist()[:3]]
            a['fail_examples_floored'] = [ex(i, params_raw, hand0, status=names[i])
                                          for i in torch.nonzero(~st.is_fd & floored).flatten().tolist()[:3]]
    rec['a'] = a
    # rows whose canonical form is decided and unique (special metrics' canonical pose depends on the input basis)
    done = c32.done if c32 is not None else torch.zeros(n, dtype=torch.bool)
    canonical = torch.tensor([s == 'canonical' for s in c32.names()]) if c32 is not None else done

    # b. builds ---------------------------------------------------------------------------------------------
    b = dict(pass_=0, elj_finite=0, ucell_ok=0, builder_dev_worst=None, builder_pass=0)

    def _builder_devs(params, hand, desc):
        out = torch.full((len(params),), float('inf'), dtype=F64)
        for i in range(len(params)):
            bi = rows(mol, sg, params[i:i + 1], hand[i:i + 1])
            out[i] = fd_starts.builder_deviation(sg, bi, desc[i], 1)
        return out
    # the raw rows as fd_start_rows hands them to MXtalTools: do they build the described crystal?
    ok, r = G.run('builder_deviation_raw', _builder_devs, params_raw, hand0, d32)
    if ok:
        b['raw_builder_pass'] = int((r < BUILDER_TOL_A).sum())
        b['raw_builder_dev_worst'] = _max(r)
        b['raw_fail_examples'] = [ex(i, params_raw, hand0, dev_A=_f(r[i]))
                                  for i in torch.nonzero(~(r < BUILDER_TOL_A)).flatten().tolist()[:3]]
    ok, E0 = G.run('elj_rows', elj_of, mol, sg, params0, hand0, chunk)
    if ok:
        b['elj_finite'] = int(torch.isfinite(E0).sum())
        b['elj_median'] = _f(E0[torch.isfinite(E0)].median()) if b['elj_finite'] else None
    else:
        E0 = torch.full((n,), float('nan'), dtype=F64)
    ok_uc, ucs = G.run('unit_cells', unit_cells, mol, sg, params0, hand0, chunk)
    natoms = len(mol.z)
    numbers_cell = mol.z.repeat(sym_mult).numpy()
    uc_ok = torch.zeros(n, dtype=torch.bool)
    if ok_uc:
        uc_ok = torch.tensor([len(p) == sym_mult * natoms and bool(torch.isfinite(p).all()) for _, p in ucs])
        b['ucell_ok'] = int(uc_ok.sum())
    bdev = torch.full((n,), float('inf'), dtype=F64)
    ok, r = G.run('builder_deviation', _builder_devs, params0, hand0, d32w)
    if ok:
        bdev = r
        b['builder_dev_worst'] = _max(bdev)
        b['builder_pass'] = int((bdev < BUILDER_TOL_A).sum())
    b_pass = torch.isfinite(E0) & uc_ok & (bdev < BUILDER_TOL_A)
    b['pass_'] = int(b_pass.sum())
    b['fail_examples'] = [ex(i, params0, hand0, dev_A=_f(bdev[i]), elj=_f(E0[i]))
                          for i in torch.nonzero(~b_pass).flatten().tolist()[:3]]
    rec['b'] = b

    # c. symmetry -------------------------------------------------------------------------------------------
    c = dict(pass_=0, found={}, found_strict={}, supergroup=0, wrong=0, ops_dev_worst=None, n_checked=0)
    if ok_uc:
        def _sym():
            found, found_s, opsdev = [], [], []
            for i, (T, pos) in enumerate(ucs):
                if not bool(uc_ok[i]):
                    found.append(None)
                    found_s.append(None)
                    continue
                found.append(spglib_number(T, pos, numbers_cell, SYMPREC))
                found_s.append(spglib_number(T, pos, numbers_cell, SYMPREC_STRICT))
            for i, (T, pos) in enumerate(ucs):
                if bool(uc_ok[i]) and (i < OPS_ROWS or found[i] != sg):
                    opsdev.append((i, ops_mismatch(T, pos, numbers_cell, sg)))
            return found, found_s, opsdev
        ok, r = G.run('symmetry', _sym)
        if ok:
            found, found_s, opsdev = r
            od = dict(opsdev)
            c['n_checked'] = sum(1 for f in found if f is not None)
            c['found'] = {str(k): sum(1 for f in found if f == k) for k in sorted(set(found), key=lambda v: (v is None, v))}
            c['found_strict'] = {str(k): sum(1 for f in found_s if f == k)
                                 for k in sorted(set(found_s), key=lambda v: (v is None, v))}
            # pass: spglib returns sg at either tolerance (1e-2 A can fail on clashing starts whose atoms of
            # different molecules lie within it; 1e-3 A can fail on float32 rows of very long cells)
            passed = [f == sg or fs == sg for f, fs in zip(found, found_s)]
            c['pass_'] = sum(passed)
            c['pass_at_symprec'] = sum(1 for f in found if f == sg)
            c['pass_strict'] = sum(1 for f in found_s if f == sg)
            other = [i for i in range(len(found)) if not passed[i]]
            sup = [i for i in other if (found[i] is not None or found_s[i] is not None) and od.get(i, 1e9) < OPS_TOL_A]
            wrong = [i for i in other if i not in sup]  # od holds every row with found != sg (_sym)
            c['supergroup'], c['wrong'] = len(sup), len(wrong)
            c['supergroup_numbers'] = sorted(set(found[i] for i in sup))
            # spglib returned no dataset at either tolerance although every operator of G holds: a tool failure
            # (it refuses cells whose atoms of different molecules nearly coincide), recorded apart from 'wrong'
            spg_none = [i for i in wrong if found[i] is None and found_s[i] is None and od.get(i, 1e9) < OPS_TOL_A]
            wrong = [i for i in wrong if i not in spg_none]
            c['wrong'] = len(wrong)
            c['spglib_none_ops_hold'] = len(spg_none)
            c['spglib_none_min_dist_A'] = [_f(min_atom_distance(*ucs[i])) for i in spg_none[:4]]
            c['wrong_examples'] = [ex(i, params0, hand0, found=found[i], found_strict=found_s[i],
                                      ops_dev_A=_f(od.get(i)), min_dist_A=_f(min_atom_distance(*ucs[i])))
                                   for i in wrong[:3]]
            c['ops_dev_worst'] = _max(torch.tensor([v for _, v in opsdev])) if opsdev else None
    rec['c'] = c

    # d. re-description ---------------------------------------------------------------------------------------
    d = dict(n=0, pass64=0, identical64=0, equivalent64=0, pass32=0, identical32=0, equivalent32=0, elj_pass=0,
             elj_n=0, elj_canon_pass=0, clip_band=0, improper=0, setting_changed=0, builder_pass=0,
             excluded_not_canonical=int((~canonical).sum()))
    if c32 is not None and bool(canonical.any()):
        idx = torch.nonzero(canonical).flatten()
        cref = c32.desc[idx]
        ok, r = G.run('redescribe', redescribe, sg, cref, g)
        if ok:
            rd, rinfo = r

            def dex(i, **kw):  # a failing re-described row: the start row and the re-description applied
                return dict(ex(int(idx[i]), params_raw, hand0), element=int(rinfo['element'][i]),
                            det=int(rinfo['det'][i]), lattice=rinfo['lattice'][i].tolist(),
                            free_shift=[round(float(x), 6) for x in rinfo['free_shift'][i]],
                            setting=rinfo['setting'][i], **kw)
            improper = rinfo['det'] < 0
            sohn = torch.full((len(idx),), bool(sohncke))
            d['n'] = len(idx)
            d['improper'] = int(improper.sum())
            d['setting_changed'] = sum(1 for s in rinfo['setting'] if s != 'none')
            ok, cmp64 = G.run('redesc_canon64', compare_canonical, sg, cref, rd, metric, sohn, improper,
                              REDESC_PARAM_TOL64, REDESC_FD_TOL64)
            if ok:
                d['pass64'] = int(cmp64['ok'].sum())
                d['identical64'] = int(cmp64['identical'].sum())
                d['equivalent64'] = int(cmp64['equivalent'].sum())
                d['worst_param64'] = _max(cmp64['param_dev'])
                d['worst_fd64_A'] = _max(cmp64['fd_dist'])
                d['status_fail64'] = int((~cmp64['status_ok']).sum())
                d['h_fail64'] = int((~cmp64['h_ok']).sum())
                bad = torch.nonzero(~cmp64['ok']).flatten().tolist()
                d['fail64_examples'] = [dex(i, status=cmp64['names'][i], path=cmp64['path'][i],
                                            param_dev=_f(cmp64['param_dev'][i]), fd_A=_f(cmp64['fd_dist'][i]))
                                        for i in bad[:4]]
            # float32 rows of the re-description: canonical forms, energies, builder
            ok, pr = G.run('redesc_rows', desc_params, rd)
            if ok:
                p_r, h_r, clip = pr
                d['clip_band'] = int(clip.sum())
                ok, cmp32 = G.run('redesc_canon32', lambda: compare_canonical(
                    sg, cref, fd_starts.rows_to_desc(sg, rows(mol, sg, p_r, h_r)), metric, sohn, improper,
                    REDESC_PARAM_TOL32, REDESC_FD_TOL32))
                if ok:
                    d['pass32'] = int(cmp32['ok'].sum())
                    d['identical32'] = int(cmp32['identical'].sum())
                    d['equivalent32'] = int(cmp32['equivalent'].sum())
                    d['worst_param32'] = _max(cmp32['param_dev'])
                    d['worst_fd32_A'] = _max(cmp32['fd_dist'])
                    bad = torch.nonzero(~cmp32['ok']).flatten().tolist()
                    d['fail32_examples'] = [dex(i, status=cmp32['names'][i], path=cmp32['path'][i],
                                                param_dev=_f(cmp32['param_dev'][i]), fd_A=_f(cmp32['fd_dist'][i]),
                                                redesc_params=[round(float(x), 7) for x in p_r[i]],
                                                redesc_hand=float(h_r[i])) for i in bad[:4]]
                keep = ~clip
                ok, Er = G.run('redesc_elj', elj_of, mol, sg, p_r[keep], h_r[keep], chunk)
                if ok:
                    rel = elj_rel(Er, E0[idx][keep])
                    d['elj_n'] = int(keep.sum())
                    d['elj_pass'] = int((rel <= ELJ_RTOL).sum())
                    d['elj_worst_rel'] = _max(rel)
                    d['elj_worst_abs'] = _max((Er - E0[idx][keep]).abs())
                    badk = torch.nonzero(~(rel <= ELJ_RTOL)).flatten()
                    kidx = torch.nonzero(keep).flatten()
                    d['elj_fail_examples'] = [dex(int(kidx[j]), E0=_f(E0[idx][kidx[j]]), E=_f(Er[j]),
                                                  redesc_params=[round(float(x), 7) for x in p_r[kidx[j]]],
                                                  redesc_hand=float(h_r[kidx[j]])) for j in badk[:4].tolist()]
                    if len(badk):
                        fk = kidx[badk]
                        ok, cd = G.run('redesc_elj_cap_diag', cap_diagnosis, mol, sg, params0[idx[fk]], hand0[idx[fk]],
                                       p_r[fk], h_r[fk])
                        if ok:
                            ok2, nz = G.run('redesc_elj_noise_diag', noise_diagnosis, mol, sg, params0[idx[fk]],
                                            hand0[idx[fk]], g)
                            if ok2:
                                dE = (Er[badk] - E0[idx][fk]).abs()
                                cd['noise_spread'] = [_f(x) for x in nz[:4]]
                                cd['within_noise'] = int((dE <= NOISE_FACTOR * nz).sum())
                        d['elj_fail_diagnosis'] = cd
                ok, bd = G.run('redesc_builder', _builder_devs, p_r[keep], h_r[keep], rd[torch.nonzero(keep).flatten()])
                if ok:  # rows with a centre in the clip band are left out: the builder clips them (a known limit)
                    d['builder_n'] = int(keep.sum())
                    d['builder_pass'] = int((bd < BUILDER_TOL_A).sum())
                    d['builder_dev_worst'] = _max(bd)
            # the canonical form's own rows
            ok, pc = G.run('canon_rows', desc_params, cref)
            if ok:
                p_c, h_c, clip_c = pc
                ok, Ec = G.run('canon_elj', elj_of, mol, sg, p_c[~clip_c], h_c[~clip_c], chunk)
                if ok:
                    rel = elj_rel(Ec, E0[idx][~clip_c])
                    d['elj_canon_n'] = int((~clip_c).sum())
                    d['elj_canon_pass'] = int((rel <= ELJ_RTOL).sum())
                    d['elj_canon_worst_rel'] = _max(rel)
                    badc = torch.nonzero(~(rel <= ELJ_RTOL)).flatten()
                    if len(badc):
                        ci = torch.nonzero(~clip_c).flatten()[badc]
                        ok, cd = G.run('canon_elj_cap_diag', cap_diagnosis, mol, sg, params0[idx[ci]], hand0[idx[ci]],
                                       p_c[ci], h_c[ci])
                        d['elj_canon_fail_diagnosis'] = cd
    rec['d'] = d

    # e. chart round trip --------------------------------------------------------------------------------------
    e = dict(n=0, pass_=0, worst=None, logdens_finite=0, inv_fwd_consistent=0, pose_in_bounds=0, cell_in_bounds=0)
    if c32 is not None:
        sel = done & (c32.desc.h[:, 0] == 1)
        e['n'] = int(sel.sum())
        if bool(sel.any()):
            def _chart():
                cd = c32.desc[torch.nonzero(sel).flatten()]
                y, l_inv = ch.inverse(cd)
                f = ch.forward(y)
                dev = torch.stack([((f.desc.cell[:, :3] - cd.cell[:, :3]).abs() / cd.cell[:, :3]).amax(1),
                                   (f.desc.cell[:, 3:] - cd.cell[:, 3:]).abs().amax(1),
                                   (f.desc.t - cd.t).abs().reshape(len(y), -1).amax(1),
                                   (f.desc.R - cd.R).abs().reshape(len(y), -1).amax(1)], 1).amax(1)
                lrd = measure.log_reference_density(f.desc, ch, fd_starts.CELL_MEASURE)
                fin = torch.isfinite(f.logabsdet) & torch.isfinite(lrd)
                cons = (l_inv + f.logabsdet).abs() < 1e-8
                inb = torch.ones(len(y), len(ch.dims), dtype=torch.bool)
                for i in range(len(ch.dims)):
                    lo, hi = ch.bounds(y[:, :i], i)
                    inb[:, i] = (y[:, i] >= lo - 1e-9) & (y[:, i] <= hi + 1e-9)
                ncell = ch.blocks[0].ndim
                k_out = None
                if getattr(ch, 'k_cut', False):  # the K image (same crystal, the fold's other choice) is outside
                    yk, _ = ch.inverse(ch.k_image(cd))
                    lo, hi = ch.bounds(yk[:, :-1], ch.ndim - 1)
                    k_out = (yk[:, -1] < lo) | (yk[:, -1] > hi)
                return dev, fin, cons, inb[:, :ncell].all(1), inb[:, ncell:].all(1), k_out
            ok, r = G.run('chart', _chart)
            if ok:
                dev, fin, cons, cell_in, pose_in, k_out = r
                if k_out is not None:
                    e['k_image_outside'] = int(k_out.sum())
                e['worst'] = _max(dev)
                e['logdens_finite'] = int(fin.sum())
                e['inv_fwd_consistent'] = int(cons.sum())
                e['cell_in_bounds'] = int(cell_in.sum())
                e['pose_in_bounds'] = int(pose_in.sum())
                e['pass_'] = int(((dev <= CHART_TOL) & fin & cons & pose_in).sum())
    rec['e'] = e

    # 3. relaxations ------------------------------------------------------------------------------------------
    # from the raw rows, as the search gets them (wrap mode wraps the raw centre every step); start energies of the
    # described crystals (wrapped rows)
    rec['r'] = relax_group(sg, mol, params_raw[:n_relax], hand0[:n_relax], E0[:n_relax], metric, chunk, out_dir, G,
                           log)
    rec['exceptions'] = G.exceptions
    rec['wall_s'] = time.time() - t_group
    return rec


def relax_group(sg, mol, params, hand, E_start, metric, chunk, out_dir, G, log):
    from eval.fd_search import campaign, fd_starts, run_fd_search
    from mxtaltools.crystal_fd import canon
    from mxtaltools.crystal_search import crystal_opt_utils as cou
    from mxtaltools.dataset_utils.utils import collate_data_list
    m = len(params)
    r = dict(n=m, ran=False, mxt_canon_batch_raised=False, mxt_canon_row_raise=0, mxt_canon_row_ok=0,
             mxt_canon_misbehave=0, fold_canonical=0, fold_status={}, elj_fold_pass=0, canonical64=0, canonical32=0,
             mxt_vs_fd_consistent=0)
    if m == 0:
        return r
    cfg = campaign.stream_config('random')
    stage = dict(cfg.opt[0])
    stage['max_num_steps'] = RELAX_STEPS
    ok, oc = G.run('relax_config', run_fd_search._stage_config, stage, cfg)
    if not ok:
        return r
    r['stage'] = {k: v for k, v in oc.items() if isinstance(v, (int, float, str, bool)) or v is None}
    calls = []
    orig = cou.canonicalize_aunit

    def wrapped(cb):
        pre = dict(params=cb.full_cell_parameters().detach().float().clone(),
                   hand=cb.aunit_handedness.detach().reshape(-1).float().clone())
        try:
            orig(cb)
            calls.append(dict(pre=pre, exc=None))
        except ValueError as e:
            calls.append(dict(pre=pre, exc=f'{type(e).__name__}: {e}'))
    t0 = time.time()
    outs, steps = [], []
    n_chunks = 0
    cou.canonicalize_aunit = wrapped
    try:  # chunks of `chunk` rows (memory in the large groups); one canonicalize_aunit call per chunk
        for lo in range(0, m, chunk):
            n_chunks += 1
            b0 = rows(mol, sg, params[lo:lo + chunk], hand[lo:lo + chunk])
            ok, res = G.run('relax', b0.optimize_crystal_parameters, return_record=True,
                            intermediates_path=os.path.join(out_dir, '_opt_intermediates.pt'), **oc)
            if not ok:
                break
            outs += res[0]
            steps.append(int(res[1]['loss'].shape[0]))
    finally:
        cou.canonicalize_aunit = orig
    r['wall_s'] = time.time() - t0
    if not ok or len(calls) != n_chunks or len(outs) != m:
        r['wrapper_calls'] = len(calls)
        return r
    r['ran'] = True
    r['steps'] = max(steps)
    pre_p = torch.cat([c_['pre']['params'] for c_ in calls])
    pre_h = torch.cat([c_['pre']['hand'] for c_ in calls])
    excs = [c_['exc'] for c_ in calls if c_['exc'] is not None]
    # rows of chunks whose canonicalize_aunit raised come back wrapped but not re-expressed
    chunk_raised = torch.cat([torch.full((len(c_['pre']['hand']),), c_['exc'] is not None) for c_ in calls])
    r['mxt_canon_batch_raised'] = bool(excs)
    r['mxt_canon_batch_message'] = excs[0] if excs else None
    post = collate_data_list(outs)
    post_p = post.full_cell_parameters().detach().float().clone()
    post_h = post.aunit_handedness.detach().reshape(-1).float().clone()

    ok, E_end = G.run('relax_elj_end', elj_of, mol, sg, pre_p, pre_h, chunk)
    if not ok:
        return r
    r['elj_start_median'] = _f(E_start[torch.isfinite(E_start)].median()) if bool(torch.isfinite(E_start).any()) else None
    r['elj_end_median'] = _f(E_end[torch.isfinite(E_end)].median()) if bool(torch.isfinite(E_end).any()) else None
    r['elj_end_finite'] = int(torch.isfinite(E_end).sum())
    r['n_lowered'] = int((E_end < E_start).sum())

    # MXtalTools' canonicalize_aunit on each endpoint alone
    d_pre = fd_starts.rows_to_desc(sg, rows(mol, sg, pre_p, pre_h))
    ok_pre, c_pre = G.run('relax_fold', canon.canonicalize, d_pre)

    raise_rows, misb_rows = [], []

    def _mxt_rows():
        raised, misb, okc, msgs = 0, 0, 0, []
        for i in range(m):
            bi = rows(mol, sg, pre_p[i:i + 1], pre_h[i:i + 1])
            try:
                orig(bi)
            except ValueError as e:
                raised += 1
                raise_rows.append(i)
                msgs.append(str(e)[:160])
                continue
            okc += 1
            q_p = bi.full_cell_parameters().detach().float()
            q_h = bi.aunit_handedness.detach().reshape(-1).float()
            Eq = elj_of(mol, sg, q_p, q_h, 1)
            box = torch.stack([bi.asym_unit_dict[str(int(sg))]]).double()
            inside = bool(((q_p[:, 6:9].double() >= -1e-5) & (q_p[:, 6:9].double() <= box + 1e-5)).all())
            same = True
            if ok_pre:
                cq = canon.canonicalize(fd_starts.rows_to_desc(sg, rows(mol, sg, q_p, q_h)))
                fdd = float(fd_distance(sg, metric, cq.desc, c_pre.desc[i:i + 1])[0])
                same = cq.names()[0] == c_pre.names()[i] and bool(cq.desc.h[0, 0] == c_pre.desc.h[i, 0]) \
                    and fdd <= REDESC_FD_TOL32
            if not (float(elj_rel(Eq, E_end[i:i + 1])[0]) <= ELJ_RTOL and inside and same):
                misb += 1
                misb_rows.append(i)
                msgs.append(f'row {i}: misbehaved (dE rel {float(elj_rel(Eq, E_end[i:i + 1])[0]):.2e}, in box '
                            f'{inside}, same canonical {same})')
        return raised, misb, okc, msgs
    ok, mr = G.run('relax_mxt_canon_rows', _mxt_rows)
    if ok:
        r['mxt_canon_row_raise'], r['mxt_canon_misbehave'], r['mxt_canon_row_ok'], msgs = mr
        r['mxt_canon_messages'] = msgs[:4]
    # MXtalTools' returned rows against the crystal_fd fold of the endpoint
    r['mxt_vs_fd_eligible'] = int((~chunk_raised).sum())
    if ok_pre and bool((~chunk_raised).any()):
        ok, cpost = G.run('relax_fold_post', canon.canonicalize,
                          fd_starts.rows_to_desc(sg, rows(mol, sg, post_p, post_h)))
        if ok:
            fdd = fd_distance(sg, metric, cpost.desc, c_pre.desc)
            same = torch.tensor([a == b for a, b in zip(cpost.names(), c_pre.names())]) & \
                (cpost.desc.h[:, 0] == c_pre.desc.h[:, 0]) & (fdd <= REDESC_FD_TOL32) & ~chunk_raised
            fdd = fdd[~chunk_raised]
            r['mxt_vs_fd_consistent'] = int(same.sum())
            r['mxt_vs_fd_worst_A'] = _max(fdd)
    if not ok_pre:
        return r
    names = c_pre.names()
    r['fold_status'] = {s: names.count(s) for s in sorted(set(names))}
    r['fold_canonical'] = names.count('canonical')
    r['fold_moved'] = int(c_pre.moved.sum())

    def rex(i, **kw):  # a failing endpoint: its start row and the endpoint MXtalTools handed canonicalize_aunit
        ca = torch.cos(pre_p[i, 3:6].double())
        vf = 1 - (ca ** 2).sum() + 2 * ca.prod()  # (V / abc)^2; <= 0: the angles admit no real cell
        return dict(row=int(i), start=[round(float(x), 7) for x in params[i]], start_hand=float(hand[i]),
                    end=[round(float(x), 7) for x in pre_p[i]], end_hand=float(pre_h[i]), fold_status=names[i],
                    fold_message=c_pre.message[i][:200], elj_end=_f(E_end[i]), end_volume_factor=_f(vf),
                    end_angle_sum_deg=_f(torch.rad2deg(pre_p[i, 3:6].double().sum())), **kw)
    r['fold_fail_examples'] = [rex(i) for i, nm in enumerate(names) if nm != 'canonical'][:3]
    r['mxt_canon_raise_examples'] = [rex(i) for i in raise_rows[:2]]
    r['mxt_canon_misbehave_examples'] = [rex(i) for i in misb_rows[:2]]
    good = torch.tensor([nm == 'canonical' for nm in names]) & (c_pre.desc.h[:, 0] == 1)
    if not bool(good.any()):
        return r
    gi = torch.nonzero(good).flatten()
    ok, pc = G.run('relax_fold_rows', desc_params, c_pre.desc[gi])
    if not ok:
        return r
    p_c, h_c, clip_c = pc
    r['fold_clip_band'] = int(clip_c.sum())
    ok, Ec = G.run('relax_fold_elj', elj_of, mol, sg, p_c[~clip_c], h_c[~clip_c], chunk)
    if ok:
        rel = elj_rel(Ec, E_end[gi][~clip_c])
        r['elj_fold_pass'] = int((rel <= ELJ_RTOL).sum())
        r['elj_fold_worst_rel'] = _max(rel)
        gk = gi[~clip_c]
        r['elj_fold_fail_examples'] = [rex(int(gk[j]), elj_fold=_f(Ec[j]))
                                       for j in torch.nonzero(~(rel <= ELJ_RTOL)).flatten().tolist()[:3]]
    ok, st64 = G.run('relax_fd_status64', canon.fd_status, c_pre.desc[gi])
    if ok:
        r['canonical64'] = int(st64.is_fd.sum())
    # the float32 round trip of the folded endpoint, centres NOT wrapped (the FD representative may have t outside
    # [0, 1)): stored as MolCrystalData parameters and read back
    p_nw, h_nw = fd_starts.desc_to_params(c_pre.desc[gi])
    ok, st32 = G.run('relax_fd_status32', lambda: canon.fd_status(
        fd_starts.rows_to_desc(sg, rows(mol, sg, p_nw, h_nw)), tol=CANON_TOL32))
    if ok:
        r['canonical32'] = int(st32.is_fd.sum())
        r['canonical32_fail_examples'] = [rex(int(gi[j])) for j in torch.nonzero(~st32.is_fd).flatten().tolist()[:3]]
    return r


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def parse_groups(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    bad = [g for g in out if not 1 <= g <= 230]
    if bad:
        raise ValueError(f'space groups out of range: {bad}')
    return out


def _git_head(path):
    import subprocess
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _atomic_json(obj, path):
    if os.path.exists(path):
        raise FileExistsError(path)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def out_path(name):
    if not name or any(c in name for c in '/\\:') or name in ('.', '..'):
        raise ValueError(f'bad run name {name!r}')
    return os.path.join(OUT_ROOT, name)


def cmd_run(args):
    torch.set_num_threads(args.threads)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise SystemExit('run with CUDA_VISIBLE_DEVICES=-1 (CPU only)')
    out_dir = out_path(args.name)
    if args.resume:
        if not os.path.isdir(out_dir):
            raise FileNotFoundError(out_dir)
        with open(os.path.join(out_dir, 'meta.json')) as fh:
            meta = json.load(fh)
        for k in ('molecule', 'n', 'n_relax', 'seed'):
            if meta['args'][k] != getattr(args, k):
                raise ValueError(f'--resume with {k}={getattr(args, k)!r}; the run has {meta["args"][k]!r}')
    else:
        os.makedirs(OUT_ROOT, exist_ok=True)
        os.makedirs(out_dir, exist_ok=False)  # refuses an existing directory
        os.makedirs(os.path.join(out_dir, 'groups'))
    log_path = os.path.join(out_dir, 'run.log')

    def log(*a):
        print(*a, flush=True)
        with open(log_path, 'a') as fh:
            print(*a, file=fh)

    from eval.fd_stress import molecule as M
    mol_file = os.path.join(out_dir, 'molecule.pt')
    if os.path.exists(mol_file):
        saved = torch.load(mol_file, weights_only=False)
        mol, mol_rec = saved['mol'], saved['record']
    else:
        if args.molecule == 'c1':
            prior = M.load_conditions()
            mol, mol_rec = M.pick_c1_molecule(prior, prefer=args.prefer, log=log)
            del prior
        else:
            mol = M.load_mipcas()
            mol_rec = dict(chosen=dict(smiles=mol.smiles, identifier=mol.identifier, **M.point_symmetry(mol.pos, mol.z)))
        torch.save(dict(mol=mol, record=mol_rec), mol_file)
    if not args.resume:
        meta = dict(args=vars(args), started=time.strftime('%Y-%m-%d %H:%M:%S'), python=sys.version,
                    mxt_head=_git_head(os.path.dirname(os.path.dirname(os.path.abspath(
                        __import__('mxtaltools').__file__)))),
                    gfn_head=_git_head(GFN_ROOT), molecule=mol_rec, torch_threads=args.threads,
                    tolerances=dict(CANON_TOL32=CANON_TOL32, BUILDER_TOL_A=BUILDER_TOL_A, SYMPREC=SYMPREC,
                                    SYMPREC_STRICT=SYMPREC_STRICT, OPS_TOL_A=OPS_TOL_A,
                                    REDESC_PARAM_TOL64=REDESC_PARAM_TOL64, REDESC_FD_TOL64=REDESC_FD_TOL64,
                                    REDESC_PARAM_TOL32=REDESC_PARAM_TOL32, REDESC_FD_TOL32=REDESC_FD_TOL32,
                                    ELJ_RTOL=ELJ_RTOL, CHART_TOL=CHART_TOL, RELAX_STEPS=RELAX_STEPS))
        _atomic_json(meta, os.path.join(out_dir, 'meta.json'))
    log(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {"resumed" if args.resume else "started"} {vars(args)}; molecule '
        f'{mol_rec["chosen"].get("smiles")}')
    rg = radius_of_gyration(mol)
    import psutil
    proc = psutil.Process(os.getpid())
    groups = parse_groups(args.groups)
    for sg in groups:
        path = os.path.join(out_dir, 'groups', f'sg_{sg:03d}.json')
        if os.path.exists(path):
            continue
        t0 = time.time()
        log(f'[{args.name}] sg {sg}')
        try:
            rec = run_group(sg, mol, rg, args.n, args.n_relax, args.seed, out_dir, log)
        except Exception as e:  # noqa: BLE001 -- a harness-level failure is recorded as such
            rec = dict(sg=sg, harness_exception=dict(type=type(e).__name__, message=str(e)[:600],
                                                     traceback=traceback.format_exc().strip().splitlines()[-14:]))
            log(f'    HARNESS EXCEPTION sg {sg}: {type(e).__name__}: {e}')
        mi = proc.memory_info()
        rec['rss_gb'] = mi.rss / 1e9
        rec['rss_peak_gb'] = getattr(mi, 'peak_wset', mi.rss) / 1e9
        _atomic_json(rec, path)
        log(f'    sg {sg} {rec.get("symbol", "")}: {time.time() - t0:.1f}s, '
            f'a {rec.get("a", {}).get("pass_unfloored")}/{rec.get("a", {}).get("n_unfloored")} '
            f'b {rec.get("b", {}).get("pass_")} c {rec.get("c", {}).get("pass_")} '
            f'd64 {rec.get("d", {}).get("pass64")} d32 {rec.get("d", {}).get("pass32")} '
            f'dE {rec.get("d", {}).get("elj_pass")}/{rec.get("d", {}).get("elj_n")} e {rec.get("e", {}).get("pass_")} '
            f'relax fold {rec.get("r", {}).get("fold_canonical")} mxt_raise '
            f'{rec.get("r", {}).get("mxt_canon_row_raise")} exc {len(rec.get("exceptions", []))} '
            f'rss {rec["rss_gb"]:.2f} GB')
        import gc
        gc.collect()
    log(f'{time.strftime("%Y-%m-%d %H:%M:%S")} done')


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

FAMILY_ORDER = ('triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'hexagonal', 'cubic')


def failure_modes(rec):
    """List of (mode, detail) for one group record. A missing stage is a failure, never a pass."""
    out = []
    if 'harness_exception' in rec:
        return [('harness_exception', rec['harness_exception']['type'] + ': ' + rec['harness_exception']['message'])]
    n, m = rec['n'], rec['n_relax']
    for e in rec.get('exceptions', []):
        out.append((f'exception:{e["stage"]}', f'{e["type"]}: {e["message"][:200]}'))
    if rec.get('sampler') == 'double_cover_fallback':
        out.append(('sampler_refused_fd_chart', 'FDStartSampler refused (K nontrivial); double-cover fallback used'))
    a, b, c, d, e, r = (rec.get(k, {}) for k in 'abcder')
    if not a:
        return out + [('no_starts', 'no rows drawn')]
    if a.get('draw_is_fd', 0) != n:
        out.append(('draw_not_canonical', f'{a.get("draw_is_fd")}/{n} draws (before the floor) in the FD'))
    if a.get('pass_unfloored', 0) != a.get('n_unfloored', -1):
        out.append(('rows_not_canonical', f'{a.get("pass_unfloored")}/{a.get("n_unfloored")} unfloored rows in the FD '
                                          f'(worst delta {a.get("worst_delta_unfloored")})'))
    if a.get('pass_all', 0) != n and a.get('pass_unfloored', 0) == a.get('n_unfloored', -1):
        out.append(('floored_rows_not_canonical', f'{n - a.get("pass_all", 0)} of {rec.get("n_floored")} floored rows '
                                                  f'left the FD (statuses {a.get("status")})'))
    if b.get('raw_builder_pass', 0) != n:
        out.append(('raw_rows_misbuilt', f'{b.get("raw_builder_pass")}/{n} rows as fd_start_rows hands them build the '
                                         f'described crystal ({rec.get("n_t_outside_unit")} rows with a centre outside '
                                         f'[0, 1); worst {b.get("raw_builder_dev_worst")} A)'))
    if b.get('elj_finite', 0) != n:
        out.append(('elj_nonfinite', f'{b.get("elj_finite")}/{n} finite'))
    if b.get('ucell_ok', 0) != n:
        out.append(('ucell_build', f'{b.get("ucell_ok")}/{n} unit cells with sym_mult molecules'))
    if b.get('builder_pass', 0) != n:
        out.append(('builder_deviation', f'{b.get("builder_pass")}/{n} below {BUILDER_TOL_A} A (worst '
                                         f'{b.get("builder_dev_worst")})'))
    if c.get('wrong', 0) or c.get('pass_', 0) + c.get('supergroup', 0) + c.get('spglib_none_ops_hold', 0) != n:
        out.append(('symmetry_wrong', f'spglib found {c.get("found")}; {c.get("wrong")} not explained by a supergroup; '
                                      f'e.g. {c.get("wrong_examples", [])[:1]}'))
    if c.get('spglib_none_ops_hold', 0):
        out.append(('symmetry_spglib_none_ops_hold', f'{c["spglib_none_ops_hold"]} rows: spglib returned no dataset at '
                                                     f'{SYMPREC} and {SYMPREC_STRICT} A while every operator of G '
                                                     f'holds; closest atoms {c.get("spglib_none_min_dist_A")} A'))
    if c.get('supergroup', 0):
        out.append(('symmetry_supergroup', f'{c.get("supergroup")} rows in {c.get("supergroup_numbers")}'))
    if c.get('pass_', 0) and (c.get('pass_strict') != c.get('pass_') or c.get('pass_at_symprec') != c.get('pass_')):
        out.append(('symmetry_tolerance_sensitive', f'symprec {SYMPREC}: {c.get("found")}; symprec {SYMPREC_STRICT}: '
                                                    f'{c.get("found_strict")} (passes at one of them)'))
    nd = d.get('n', 0)
    if nd + d.get('excluded_not_canonical', 0) != n:
        out.append(('redesc_not_run', f'{nd}/{n} rows re-described'))
    if d.get('pass64', 0) != nd:
        out.append(('redesc_canonical64', f'{d.get("pass64")}/{nd} (worst param {d.get("worst_param64")}, FD '
                                          f'{d.get("worst_fd64_A")} A); e.g. {d.get("fail64_examples", [])[:2]}'))
    if d.get('pass32', 0) != nd:
        out.append(('redesc_canonical32', f'{d.get("pass32")}/{nd} (worst param {d.get("worst_param32")}, FD '
                                          f'{d.get("worst_fd32_A")} A); e.g. {d.get("fail32_examples", [])[:2]}'))
    def _elj_mode(base, npass, ntot, diag):
        nfail = ntot - npass
        if diag and diag.get('explained_by_cap') == nfail:
            return f'{base}_neighbour_cap'
        if diag and diag.get('within_noise') == nfail:
            return f'{base}_float32_noise'
        return base
    if d.get('elj_pass', 0) != d.get('elj_n', -1) or d.get('elj_n', 0) + d.get('clip_band', 0) != nd:
        out.append((_elj_mode('redesc_elj', d.get('elj_pass', 0), d.get('elj_n', 0), d.get('elj_fail_diagnosis')),
                    f'{d.get("elj_pass")}/{d.get("elj_n")} within max({ELJ_RTOL} |E|, {ELJ_ATOL}) (worst rel '
                    f'{d.get("elj_worst_rel")}, abs {d.get("elj_worst_abs")}); uncapped re-evaluation '
                    f'{d.get("elj_fail_diagnosis")}; e.g. {d.get("elj_fail_examples", [])[:1]}'))
    if d.get('elj_canon_pass', 0) != d.get('elj_canon_n', -1):
        out.append((_elj_mode('canon_elj', d.get('elj_canon_pass', 0), d.get('elj_canon_n', 0),
                              d.get('elj_canon_fail_diagnosis')),
                    f'{d.get("elj_canon_pass")}/{d.get("elj_canon_n")} (worst rel {d.get("elj_canon_worst_rel")}); '
                    f'uncapped re-evaluation {d.get("elj_canon_fail_diagnosis")}'))
    # records written before builder_n existed counted clip-band rows too: there, failures up to clip_band are the
    # clip band's (the builder clips those centres)
    if ('builder_n' in d and d.get('builder_pass', 0) != d['builder_n']) or             ('builder_n' not in d and nd - d.get('builder_pass', 0) > d.get('clip_band', 0)):
        out.append(('redesc_builder_deviation', f'{d.get("builder_pass")}/{nd} (worst {d.get("builder_dev_worst")} A)'))
    if e.get('n', 0) != n or e.get('pass_', 0) != e.get('n', -1):
        out.append(('chart', f'{e.get("pass_")}/{e.get("n")} of {n} (worst {e.get("worst")}, logdens finite '
                             f'{e.get("logdens_finite")}, pose in bounds {e.get("pose_in_bounds")})'))
    if not r.get('ran'):
        out.append(('relax_not_run', 'relaxation did not complete'))
        return out
    cov = (rec.get('mxt_box') or {}).get('covered')
    if r.get('mxt_canon_batch_raised') or r.get('mxt_canon_row_raise', 0):
        mode = 'mxt_canonicalize_aunit_raises' if (cov is not None and cov < 1) else \
            'mxt_canonicalize_aunit_raises_box_covers'
        out.append((mode, f'{r.get("mxt_canon_row_raise")}/{m} endpoints (ASYM_UNITS box covers {cov} of centres '
                          f'under G, mean images {(rec.get("mxt_box") or {}).get("mean_images")}): '
                          f'{(r.get("mxt_canon_messages") or [""])[0][:120]}; e.g. '
                          f'{r.get("mxt_canon_raise_examples", [])[:1]}'))
    if r.get('mxt_canon_misbehave', 0):
        out.append(('mxt_canonicalize_aunit_misbehaves', f'{r.get("mxt_canon_misbehave")}/{m}: '
                                                         f'{r.get("mxt_canon_messages")}'))
    if r.get('mxt_vs_fd_consistent', 0) != r.get('mxt_vs_fd_eligible', m):
        out.append(('mxt_endpoint_vs_fd', f'{r.get("mxt_vs_fd_consistent")}/{m} MXT-returned rows fold to the '
                                          f'endpoint\'s canonical form (worst {r.get("mxt_vs_fd_worst_A")} A)'))
    fs = dict(r.get('fold_status') or {})
    if fs.get('unreal_cell'):
        ur = [x for x in r.get('fold_fail_examples', []) if x.get('fold_status') == 'unreal_cell']
        out.append(('relax_endpoint_unreal_cell', f'{fs["unreal_cell"]}/{m} endpoints have angles admitting no real '
                                                  f'cell; e.g. {ur[:1]}'))
    fs.pop('unreal_cell', None)
    if sum(v for k, v in fs.items() if k != 'canonical'):
        out.append(('relax_fold_status', f'{r.get("fold_status")}; e.g. {r.get("fold_fail_examples", [])[:1]}'))
    nf = r.get('fold_canonical', 0) - r.get('fold_clip_band', 0)
    if r.get('elj_fold_pass', 0) != nf:
        out.append(('relax_fold_elj', f'{r.get("elj_fold_pass")}/{nf} (worst rel {r.get("elj_fold_worst_rel")})'))
    if r.get('canonical64', 0) != r.get('fold_canonical', 0) or r.get('canonical32', 0) != r.get('fold_canonical', 0):
        out.append(('relax_fold_not_canonical', f'float64 {r.get("canonical64")}, float32 {r.get("canonical32")} of '
                                                f'{r.get("fold_canonical")}'))
    return out


CSV_COLS = [
    ('sg', 'sg'), ('symbol', 'symbol'), ('family', 'lattice_type'), ('sohncke', 'sohncke'), ('sym_mult', 'sym_mult'),
    ('free_axes', 'free_axes'), ('cell_dims', 'cell_dims'), ('centroid_dims', 'centroid_dims'),
    ('sampler', 'sampler'), ('box_coverage', 'box_coverage'),
]


def flat_row(rec):
    row = {k: rec.get(v) for k, v in CSV_COLS}
    di = rec.get('draw_info', {})
    for blk in ('cell', 'centroid', 'orientation'):
        row[f'accept_{blk}'] = di.get(f'accept_{blk}')
        bp = (rec.get('bound_probe') or {}).get(blk, {})
        row[f'bound_margin_{blk}'] = None if bp.get('max_minus_bound') is None else -bp['max_minus_bound']
        row[f'bound_violations_{blk}'] = bp.get('violations')
    row['accept_joint'] = di.get('accept_joint_equivalent')
    row['t_draw_s'] = rec.get('t_draw_s')
    row['n_floored'] = rec.get('n_floored')
    a, b, c, d, e, r = (rec.get(k, {}) for k in 'abcder')
    row.update({
        'a_draw_is_fd': a.get('draw_is_fd'), 'a_pass_all': a.get('pass_all'), 'a_pass_unfloored': a.get('pass_unfloored'),
        'a_n_unfloored': a.get('n_unfloored'), 'a_worst_delta_unfloored': a.get('worst_delta_unfloored'),
        'b_pass': b.get('pass_'), 'b_elj_finite': b.get('elj_finite'), 'b_builder_dev_worst_A': b.get('builder_dev_worst'),
        'c_pass': c.get('pass_'), 'c_pass_strict': c.get('pass_strict'), 'c_supergroup': c.get('supergroup'),
        'c_wrong': c.get('wrong'), 'c_ops_dev_worst_A': c.get('ops_dev_worst'),
        'd_n': d.get('n'), 'd_improper': d.get('improper'), 'd_setting_changed': d.get('setting_changed'),
        'd_pass64': d.get('pass64'), 'd_equivalent64': d.get('equivalent64'), 'd_worst_param64': d.get('worst_param64'),
        'd_pass32': d.get('pass32'), 'd_worst_fd32_A': d.get('worst_fd32_A'),
        'd_elj_pass': d.get('elj_pass'), 'd_elj_n': d.get('elj_n'), 'd_elj_worst_rel': d.get('elj_worst_rel'),
        'd_elj_canon_pass': d.get('elj_canon_pass'), 'd_builder_pass': d.get('builder_pass'),
        'e_pass': e.get('pass_'), 'e_n': e.get('n'), 'e_worst': e.get('worst'), 'e_logdens_finite': e.get('logdens_finite'),
        'e_pose_in_bounds': e.get('pose_in_bounds'), 'e_cell_in_bounds': e.get('cell_in_bounds'),
        'r_ran': r.get('ran'), 'r_steps': r.get('steps'), 'r_n_lowered': r.get('n_lowered'),
        'r_elj_start_median': r.get('elj_start_median'), 'r_elj_end_median': r.get('elj_end_median'),
        'r_mxt_canon_batch_raised': r.get('mxt_canon_batch_raised'), 'r_mxt_canon_row_raise': r.get('mxt_canon_row_raise'),
        'r_mxt_canon_misbehave': r.get('mxt_canon_misbehave'), 'r_mxt_vs_fd_consistent': r.get('mxt_vs_fd_consistent'),
        'r_fold_canonical': r.get('fold_canonical'), 'r_elj_fold_pass': r.get('elj_fold_pass'),
        'r_canonical64': r.get('canonical64'), 'r_canonical32': r.get('canonical32'),
        'n_exceptions': len(rec.get('exceptions', [])) + (1 if 'harness_exception' in rec else 0),
        'wall_s': rec.get('wall_s'),
    })
    row['failure_modes'] = ';'.join(sorted(set(m for m, _ in failure_modes(rec))))
    return row


def cmd_report(args):
    out_dir = out_path(args.name)
    gdir = os.path.join(out_dir, 'groups')
    recs = []
    for fn in sorted(os.listdir(gdir)):
        if fn.endswith('.json'):
            with open(os.path.join(gdir, fn)) as fh:
                recs.append(json.load(fh))
    with open(os.path.join(out_dir, 'meta.json')) as fh:
        meta = json.load(fh)
    rows_ = [flat_row(r) for r in recs]
    for name in ('table.csv', 'summary.md', 'failures.md'):
        if os.path.exists(os.path.join(out_dir, name)) and not args.overwrite_report:
            raise FileExistsError(f'{name} exists; pass --overwrite-report to rewrite the report files')
    with open(os.path.join(out_dir, 'table.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_[0].keys()))
        w.writeheader()
        w.writerows(rows_)
    n = meta['args']['n']
    m = meta['args']['n_relax']
    fam = {}
    for rec, row in zip(recs, rows_):
        fam.setdefault(rec.get('lattice_type', '?'), []).append((rec, row))
    lines = []
    mol = meta['molecule']['chosen']
    lines.append(f'# FD stress sweep `{args.name}`\n')
    lines.append(f'Molecule {mol.get("smiles")} ({mol.get("n_atoms")} atoms; best proper symmetry RMSD '
                 f'{mol.get("proper_rmsd"):.3f} A, best improper {mol.get("improper_rmsd"):.3f} A). '
                 f'{len(recs)} space groups, Z\' = 1, {n} FD starts per group, {m} short relaxations. '
                 f'MXtalTools {meta.get("mxt_head")}, GFN {meta.get("gfn_head")}.\n')

    def d_ran(r):  # check d ran on every row whose canonical form is unique (special metrics excluded, see a)
        d = r.get('d', {})
        return d.get('n', 0) > 0 and d['n'] + d.get('excluded_not_canonical', 0) == n

    def cnt(recs_, key, sub, target):
        return sum(1 for r in recs_ if (r.get(key, {}) or {}).get(sub) == (target(r) if callable(target) else target))

    lines.append('## Groups passing each check, by crystal family\n')
    lines.append('Caption: each cell counts the space groups of the family whose every sampled row passed the check '
                 f'(of the groups in the family). Checks a-e on {n} FD starts per group (a over the rows the 3 A floor '
                 f'left unchanged), relaxation checks on {m} endpoints of a {RELAX_STEPS}-step eLJ stage. "MXT re-expr. '
                 'raises" counts groups where MXtalTools\' canonicalize_aunit raised on some endpoint (an expected '
                 'limit where ASYM_UNITS is not a fundamental domain), not a pass/fail. Failure modes per group are '
                 'in failures.md and table.csv.\n')
    hdr = ['family', 'groups', 'sampler ok', 'a canonical', 'b builds', 'c symmetry = sg', 'c not all = sg, none wrong',
           'c G holds on every row',
           'd canon (f64)', 'd canon (f32)', 'd eLJ', 'e chart', 'relax: fold canonical', 'relax: eLJ kept by fold',
           'MXT re-expr. raises', 'groups with exceptions']
    lines.append('| ' + ' | '.join(hdr) + ' |')
    lines.append('|' + '---|' * len(hdr))
    tot = None
    for f in FAMILY_ORDER + tuple(sorted(set(fam) - set(FAMILY_ORDER))):
        if f not in fam:
            continue
        rr = [x[0] for x in fam[f]]
        vals = [
            len(rr),
            sum(1 for r in rr if r.get('sampler') == 'fd'),
            sum(1 for r in rr if r.get('a', {}).get('pass_unfloored') == r.get('a', {}).get('n_unfloored')
                and r.get('a', {}).get('draw_is_fd') == n),
            sum(1 for r in rr if r.get('b', {}).get('pass_') == n),
            sum(1 for r in rr if r.get('c', {}).get('pass_') == n),
            sum(1 for r in rr if r.get('c', {}).get('pass_', 0) < n and r.get('c', {}).get('wrong', 1) == 0),
            sum(1 for r in rr if r.get('c', {}).get('wrong', 1) == 0 and r.get('c', {}).get('pass_', 0)
                + r.get('c', {}).get('supergroup', 0) + r.get('c', {}).get('spglib_none_ops_hold', 0) == n),
            sum(1 for r in rr if d_ran(r) and r['d'].get('pass64') == r['d']['n']),
            sum(1 for r in rr if d_ran(r) and r['d'].get('pass32') == r['d']['n']),
            sum(1 for r in rr if d_ran(r) and r['d'].get('elj_pass') == r['d'].get('elj_n', -1)
                and r['d'].get('elj_n', 0) + r['d'].get('clip_band', 0) == r['d']['n']),
            sum(1 for r in rr if r.get('e', {}).get('pass_') == n),
            sum(1 for r in rr if r.get('r', {}).get('fold_canonical') == m),
            sum(1 for r in rr if r.get('r', {}).get('ran') and r.get('r', {}).get('elj_fold_pass')
                == r.get('r', {}).get('fold_canonical', 0) - r.get('r', {}).get('fold_clip_band', 0)),
            sum(1 for r in rr if r.get('r', {}).get('mxt_canon_row_raise', 0) or r.get('r', {}).get('mxt_canon_batch_raised')),
            sum(1 for r in rr if r.get('exceptions') or 'harness_exception' in r),
        ]
        tot = vals if tot is None else [x + y for x, y in zip(tot, vals)]
        lines.append(f'| {f} | ' + ' | '.join(str(v) for v in vals) + ' |')
    lines.append('| **all** | ' + ' | '.join(str(v) for v in tot) + ' |\n')
    # failure modes
    modes = {}
    for rec in recs:
        for mode, detail in failure_modes(rec):
            modes.setdefault(mode, []).append((rec['sg'], rec.get('symbol', ''), detail))
    lines.append('## Failure modes\n')
    lines.append('Caption: every mode seen, the number of space groups it occurs in, and the groups (number and '
                 'symbol). Details per group in failures.md.\n')
    lines.append('| mode | groups | space groups |')
    lines.append('|---|---|---|')
    for mode in sorted(modes, key=lambda k: -len(modes[k])):
        sgs = ', '.join(f'{s} {sym}' for s, sym, _ in modes[mode][:60]) + (' ...' if len(modes[mode]) > 60 else '')
        lines.append(f'| {mode} | {len(modes[mode])} | {sgs} |')
    wall = sum(r.get('wall_s', 0) or 0 for r in recs)
    lines.append(f'\nTotal wall time over groups: {wall / 60:.1f} min; peak RSS '
                 f'{max((r.get("rss_peak_gb") or 0) for r in recs):.2f} GB.\n')
    with open(os.path.join(out_dir, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    fl = [f'# Failures per group, `{args.name}`\n']
    for rec in recs:
        fm = failure_modes(rec)
        if fm:
            fl.append(f'## sg {rec["sg"]} {rec.get("symbol", "")} ({rec.get("lattice_type", "")})')
            for mode, detail in fm:
                fl.append(f'- **{mode}**: {detail}')
            fl.append('')
    with open(os.path.join(out_dir, 'failures.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(fl))
    print('\n'.join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--name', required=True)
    r.add_argument('--molecule', choices=('c1', 'mipcas'), default='c1')
    r.add_argument('--prefer', default='O=C1OCC2CC1N2', help='c1: smiles tried first among the candidates')
    r.add_argument('--groups', default='1-230')
    r.add_argument('--n', type=int, default=64)
    r.add_argument('--n-relax', type=int, default=16)
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--threads', type=int, default=4)
    r.add_argument('--resume', action='store_true')
    p = sub.add_parser('report')
    p.add_argument('--name', required=True)
    p.add_argument('--overwrite-report', action='store_true',
                   help='rewrite table.csv / summary.md / failures.md (the group records are never touched)')
    args = ap.parse_args(argv)
    if args.cmd == 'run':
        cmd_run(args)
    else:
        cmd_report(args)


if __name__ == '__main__':
    main()
