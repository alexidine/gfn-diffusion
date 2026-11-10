"""Pre-noised prior files (docs/design/prior_dataset_build.md): the build manifest, the per-row tags, and the
load-time selection both the builder (data_processing/build_noised_prior.py) and the trainer
(train.py::Modeller.init_prior_dataset) read.

A pre-noised prior file is an ordinary train.py prior file ({'prior', 'equalized_prior', 'thermal_scaling_factor'})
whose rows are already scored through the run's analysis (energies stamped with the file's lj_coeff), carrying three
graph attributes and a manifest under MANIFEST_KEY:

  noised_kind    KIND_ANCHOR (a source row), KIND_CHILD (an anchor kicked by the iso tile), or KIND_HARVESTED
                 (a row a run scored, merged in from its harvest shards, not kicked)
  noised_split   SPLIT_TRAIN, or SPLIT_HELDOUT (an anchor held out whole, with every child of it)
  noised_parent  the anchor row a child was kicked from (an anchor: itself)

The loader keeps rows by kind and split, inside the packing window and within window_kT x T of their condition's
floor, and checks a stratified sample of rows against a fresh scoring instead of re-scoring the file. On the FD
route it also reads the rows' chart points and the metric kernel's bank from sidecar caches beside the file when
their keys match (FD_CACHE_VERSION, cache_path), and writes them when they do not.
"""
import math
import os

import numpy as np
import torch

MANIFEST_KEY = 'noised_build'
FORMAT_VERSION = 1
KIND_ANCHOR, KIND_CHILD, KIND_HARVESTED = 0, 1, 2
KIND_NAMES = {'anchor': KIND_ANCHOR, 'child': KIND_CHILD, 'harvested': KIND_HARVESTED}
SPLIT_TRAIN, SPLIT_HELDOUT = 0, 1
TAGS = ('noised_kind', 'noised_split', 'noised_parent')

# the stored terms the entry check compares, each against |fresh - stored| <= ABS_TOL + REL_TOL * |stored|; a row
# off by more than FAIL_TOL on any of them fails the run outright
CHECK_TERMS = ('reduction_en', 'packing_coeff', 'cell_volume')
ABS_TOL, REL_TOL = 0.1, 1e-4   # energy units: two scorings of one crystal differ by up to ~0.06 raw eLJ in float noise
CHECK_ROWS = 512
# the load-time defaults (cfg:noised_prior.*): an absent key reads these
DEFAULTS = dict(window_kT=6.0, packing_window=None, train_kinds=('child',), check_rows=CHECK_ROWS,
                live_kernel=None, live_tau=1.0, live_width_cap=0.25, live_caps=(1.0, 0.2, 0.1, 0.5), live_redraws=4,
                live_bank=64, live_fresh_frac=0.1, live_shape='hessian', window_by='row')
#: window_by: the energy a child's energy window reads (window_energy): 'row' its own (a basin-kernel build always
#: reads its basin's anchor), 'parent' its parent anchor's (noised_parent), so a minimum passes or fails the window
#: with every kick of it, long kicks included
WINDOW_BY = ('row', 'parent')
#: live_kernel: the Gaussian the MLE draw adds to each drawn row (train.py::Modeller._live_kernel_draw); 'metric'
#: redraws the prior's own build kick (metric_kick at the manifest's log_lo, log_hi) around each drawn row
LIVE_KERNELS = (None, 'iso', 'hessian', 'basin', 'metric')
#: live_shape: the basin draw's covariance (basin_factors); read by live_kernel 'basin' only
LIVE_SHAPES = ('hessian', 'iso')

# SIDECAR CACHES beside the prior file, for the two CPU conversions an FD run otherwise repeats at every launch: the
# chart point of every stored row (train.py::Modeller._fd_chart_points) and the metric kernel's bank
# (train.py::Modeller._metric_kernel_table). A cache is used only when its key equals this launch's; the chart cache
# is also re-checked against a fresh conversion of FD_CACHE_CHECK_ROWS rows, and the bank cache keys on that sample's
# digest, so a change to the chart code invalidates both. A write that fails (read-only folder) is reported, not raised.
FD_CACHE_VERSION = 1
FD_CACHE_CHECK_ROWS = 256


def file_identity(path) -> dict:
    """The size of the file at `path`: what a sidecar cache keys the file on. Not its modification time, which a copy
    to another host changes: the chart cache re-converts FD_CACHE_CHECK_ROWS rows against itself on every read, so a
    file of the same size with other contents still misses (and the bank keys on that check's digest)."""
    return dict(size=int(os.stat(path).st_size))


def digest(*tensors) -> str:
    """A short sha1 of the tensors' bytes (CPU, contiguous)."""
    import hashlib
    h = hashlib.sha1()
    for t in tensors:
        h.update(torch.as_tensor(t).detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def cache_path(prior_path, kind: str, key: dict) -> str:
    """`<prior_path>.<kind>-<hash of key>.pt`: the sidecar a cache of `kind` with this key lives in."""
    import hashlib
    import json
    h = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"{prior_path}.{kind}-{h}.pt"


def load_cache(path, key: dict):
    """The cache dict at `path` when it exists, loads and carries exactly `key` at FD_CACHE_VERSION; else None."""
    if not os.path.exists(path):
        return None
    try:
        d = torch.load(path, weights_only=False, map_location='cpu')
    except Exception as e:   # a truncated or foreign file is a miss, never a crash
        print(f"fd cache {path}: unreadable ({type(e).__name__}: {e}); recomputing", flush=True)
        return None
    if not isinstance(d, dict) or d.get('version') != FD_CACHE_VERSION or d.get('key') != key:
        return None
    return d


def save_cache(path, key: dict, **payload) -> bool:
    """Write the cache atomically (a temporary file, then a rename); False, with a printed reason, when it cannot."""
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        torch.save(dict(version=FD_CACHE_VERSION, key=key, **payload), tmp)
        os.replace(tmp, path)
        return True
    except OSError as e:
        print(f"fd cache {path}: not written ({e})", flush=True)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def settings(args):
    """cfg:noised_prior as a dict, absent keys at DEFAULTS; packing_window None reads density_penalty's range."""
    from energies.molecular_crystal import PACKING_COEFF_WINDOW
    cfg = getattr(args, 'noised_prior', None)
    cfg = dict(vars(cfg)) if cfg is not None and not isinstance(cfg, dict) else dict(cfg or {})
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"noised_prior: unknown keys {sorted(unknown)} (known: {sorted(DEFAULTS)})")
    s = {**DEFAULTS, **cfg}
    s['packing_window'] = tuple(float(v) for v in (s['packing_window'] or PACKING_COEFF_WINDOW))
    bad = [k for k in s['train_kinds'] if k not in KIND_NAMES]
    if bad or not s['train_kinds']:
        raise ValueError(f"noised_prior.train_kinds must be a non-empty subset of {sorted(KIND_NAMES)}, got "
                         f"{s['train_kinds']}")
    s['window_kT'] = None if s['window_kT'] is None else float(s['window_kT'])
    if s['live_kernel'] not in LIVE_KERNELS:
        raise ValueError(f"noised_prior.live_kernel must be one of {LIVE_KERNELS}, got {s['live_kernel']!r}")
    for k in ('live_tau', 'live_width_cap'):
        v = s[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"noised_prior.{k} must be a finite number > 0, got {v!r}")
    caps = s['live_caps']
    if (not isinstance(caps, (list, tuple)) or len(caps) != 4 or any(
            isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c <= 0 for c in caps)):
        raise ValueError(f"noised_prior.live_caps must be 4 finite numbers > 0 (A, rad, fraction, rad), got {caps!r}")
    r = s['live_redraws']
    if isinstance(r, bool) or not isinstance(r, int) or r < 0:
        raise ValueError(f"noised_prior.live_redraws must be an integer >= 0, got {r!r}")
    k = s['live_bank']
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError(f"noised_prior.live_bank must be an integer >= 0, got {k!r}")
    f = s['live_fresh_frac']
    if isinstance(f, bool) or not isinstance(f, (int, float)) or not 0 <= f <= 1:
        raise ValueError(f"noised_prior.live_fresh_frac must be a number in [0, 1], got {f!r}")
    if s['window_by'] not in WINDOW_BY:
        raise ValueError(f"noised_prior.window_by must be one of {WINDOW_BY}, got {s['window_by']!r}")
    if s['live_shape'] not in LIVE_SHAPES:
        raise ValueError(f"noised_prior.live_shape must be one of {LIVE_SHAPES}, got {s['live_shape']!r}")
    if s['live_shape'] != DEFAULTS['live_shape'] and s['live_kernel'] != 'basin':
        raise ValueError(f"noised_prior.live_shape {s['live_shape']!r} is read by live_kernel 'basin' only, got "
                         f"live_kernel {s['live_kernel']!r}")
    return s


def _metric_projector(lattice_codes, dtype, device):
    """[n, 12, 12]: the orthogonal projector onto the live coordinates of each row's crystal system, in the metric
    coordinates of metric_kick (Z' = 1). It mirrors mxtaltools enforce_crystal_system: lengths tied to their mean
    (tetragonal and hexagonal a = b; rhombohedral and cubic a = b = c) move together, fixed angles do not move,
    rhombohedral angles move together; the centroid and orientation rows are always live."""
    from mxtaltools.constants.space_group_info import LATTICE_TO_CODE as C
    eye3 = torch.eye(3, dtype=dtype, device=device)
    ab = torch.tensor([[.5, .5, 0.], [.5, .5, 0.], [0., 0., 1.]], dtype=dtype, device=device)
    abc = torch.full((3, 3), 1.0 / 3.0, dtype=dtype, device=device)
    zero = torch.zeros(3, 3, dtype=dtype, device=device)
    beta = torch.diag(torch.tensor([0., 1., 0.], dtype=dtype, device=device))
    blocks = {C['triclinic']: (eye3, eye3), C['monoclinic']: (eye3, beta), C['orthorhombic']: (eye3, zero),
              C['tetragonal']: (ab, zero), C['hexagonal']: (ab, zero), C['rhombohedral']: (abc, abc),
              C['cubic']: (abc, zero)}
    codes = torch.as_tensor(lattice_codes).reshape(-1).tolist()
    bad = sorted(set(codes) - set(blocks))
    if bad:
        raise ValueError(f"metric_kick: unknown lattice codes {bad}")
    P = torch.zeros(len(codes), 12, 12, dtype=dtype, device=device)
    for i, c in enumerate(codes):
        P[i, :3, :3], P[i, 3:6, 3:6] = blocks[c]
    P[:, 6:9, 6:9] = eye3
    P[:, 9:12, 9:12] = eye3
    return P


def _rodrigues(w):
    """[n, 3, 3] float64 rotation matrices of rotation vectors w [n, 3], exact at w = 0."""
    th = w.norm(dim=1).clamp_min(1e-300)[:, None, None]
    K = torch.zeros(len(w), 3, 3, dtype=w.dtype, device=w.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
    K = (K - K.transpose(1, 2)) / th
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def radius_of_gyration(batch):
    """[n] float64: each molecule's RMS atom distance from its heavy-atom centroid (Angstrom), as mxtaltools
    crystal_fd.metric.molecule_info defines r_g."""
    pos, z, g = batch.pos.double(), batch.z.reshape(-1), batch.batch.reshape(-1)
    n = int(batch.num_graphs)
    heavy = (z > 1).double()
    hn = torch.zeros(n, dtype=pos.dtype, device=pos.device).index_add_(0, g, heavy)
    c = torch.zeros(n, 3, dtype=pos.dtype, device=pos.device).index_add_(0, g, pos * heavy[:, None])
    c = c / hn.clamp_min(1)[:, None]
    d2 = ((pos - c[g]) ** 2).sum(1)
    cnt = torch.zeros(n, dtype=pos.dtype, device=pos.device).index_add_(0, g, torch.ones_like(d2))
    return torch.sqrt(torch.zeros(n, dtype=pos.dtype, device=pos.device).index_add_(0, g, d2) / cnt)


def fd_chart_points(ef, batch, device=None):
    """([n] bool has a chart point, [n, data_ndim] chart rows s packed in the run's FD slot layout, zero where none,
    [n] log reference density on s, zero where none) of the crystals of `batch` (any description of each), through
    the FD chart of each row's (space group, molecule radius) on the energy function `ef` (crystal_coordinates 'fd').
    A row has no point when it does not canonicalise or its canonical cell is outside the chart's length box. Shared
    by the trainer (train.py::Modeller._fd_chart_points) and the builder (data_processing/build_noised_prior.py).
    `device`: where the canonicalisation runs (its triclinic reduction tries 3,480 bases per lattice in float64, the
    bulk of the cost; None = the CPU); the results are on the CPU either way."""
    params = batch.full_cell_parameters().detach().cpu()
    hand = batch.aunit_handedness.reshape(-1).detach().cpu()
    sg = batch.sg_ind.reshape(-1).cpu()
    radius = batch.radius.reshape(-1).detach().cpu().double()
    n = len(sg)
    ok = torch.zeros(n, dtype=torch.bool)
    log_ref = torch.zeros(n, dtype=torch.float32)
    state = torch.zeros(n, int(ef.data_ndim), dtype=torch.float32)
    for g in sorted(set(sg.tolist())):
        for lo in range(0, n, 50000):   # one chart per space group and chunk, each row at its own radius
            rows = torch.nonzero(sg == g).flatten()
            rows = rows[(rows >= lo) & (rows < lo + 50000)]
            if rows.numel() == 0:
                continue
            fc = ef.fd_coordinates_rows(int(g), radius[rows])
            r = fc.from_params(params[rows], hand[rows], device=device)
            good = rows[r.ok]
            ok[good] = True
            state[good] = ef.fd_layout.pack(int(g), r.s[r.ok].float())
            if good.numel():
                _, _, lr = ef.fd_coordinates_rows(int(g), radius[good]).to_params(r.s[r.ok].double())
                log_ref[good] = lr.float()
    return ok, state, log_ref


KICK_CHARTS = ('latent', 'fd')


def metric_kick(batch, log_lo, log_hi, eps=1e-6, generator=None, chart='latent'):
    """Displace each Z' = 1 crystal of `batch` by a step isotropic in the FD identity metric (mxtaltools
    crystal_fd.metric: every coordinate in Angstrom), of length 10^U(log_lo, log_hi) Angstrom; IN PLACE. Returns [n]
    the step lengths (Angstrom).

    The metric coordinates of a displacement are
      lengths      (da, db, dc)
      angles       (dalpha sqrt(bc), dbeta sqrt(ac), dgamma sqrt(ab))
      centroid     T_fc du, the Cartesian displacement of the molecule centre
      orientation  r_g w, w the rotation vector of the turn applied to the molecule (radius_of_gyration)
    each at the crystal before the step. The direction is uniform on the unit sphere of the row's live coordinates
    (_metric_projector: the crystal system's tied and fixed cell parameters), so in every live direction a step of
    length rho moves the crystal rho in the metric to first order. A free polar axis of the space group is not
    removed: a step along it moves nothing physical. chart 'latent': after the step the row goes through the same three
    primitives as the latent iso kick (MolCrystalOps.log_noise_latent_parameters): latent_params, clipped to
    [-1 + eps, 1 - eps], latent_to_cell_params, clean_cell_parameters(mode='hard'), so a kicked crystal is one the
    trainer latent can hold (the hard clean also folds the centroid into the latent's asymmetric-unit box, which on
    P2_1/c is y <= 1/4: a molecule the search placed past it becomes another crystal). chart 'fd': the stepped
    parameters are written as they are, any description of the crystal; the caller canonicalises them through the FD
    chart (fd_chart_points), which drops a row with no chart point. The draws use `generator` (None: the global torch
    generator)."""
    if chart not in KICK_CHARTS:
        raise ValueError(f"metric_kick: chart must be one of {KICK_CHARTS}, got {chart!r}")
    from mxtaltools.common.geometry_utils import init_sym_info, rotmat2rotvec_stable
    from mxtaltools.crystal_building.utils import canonicalize_rotvec
    if int(batch.max_z_prime) != 1:
        raise ValueError(f"metric_kick: Z' = 1 batches only, got max_z_prime {int(batch.max_z_prime)}")
    dev, F = batch.cell_lengths.device, torch.float64
    n = int(batch.num_graphs)
    p = batch.full_cell_parameters().detach().to(F)
    codes = init_sym_info()['lattice_code'][torch.as_tensor(batch.sg_ind).reshape(-1).long().cpu()]
    P = _metric_projector(codes, F, dev)
    g = torch.randn(n, 12, dtype=F, generator=generator).to(dev)
    y = torch.einsum('nij,nj->ni', P, g)
    u = y / y.norm(dim=1, keepdim=True)
    t = torch.rand(n, dtype=F, generator=generator).to(dev)
    rho = 10 ** (float(log_lo) + (float(log_hi) - float(log_lo)) * t)
    s = rho[:, None] * u
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    q = p.clone()
    q[:, :3] = p[:, :3] + s[:, :3]
    q[:, 3:6] = p[:, 3:6] + s[:, 3:6] / torch.stack([torch.sqrt(b * c), torch.sqrt(a * c), torch.sqrt(a * b)], 1)
    T = batch.T_fc.detach().to(F).reshape(n, 3, 3)
    q[:, 6:9] = p[:, 6:9] + torch.linalg.solve(T, s[:, 6:9, None])[..., 0]
    w = s[:, 9:12] / radius_of_gyration(batch).to(dev)[:, None]
    # the latent's orientation rows hold the +z half-space description (canonicalize_rotvec), which an angle in
    # [0, pi] from rotmat2rotvec_stable need not be: the round trip below would clip z, a different rotation
    q[:, 9:12] = canonicalize_rotvec(rotmat2rotvec_stable(_rodrigues(w) @ _rodrigues(p[:, 9:12])).to(F))
    batch.set_cell_parameters(q.to(batch.cell_lengths.dtype))
    if chart == 'fd':
        return rho
    lat = batch.latent_params(gauge_fix_free_axes=False)
    batch.latent_to_cell_params(lat.clip(min=-1 + eps, max=1 - eps))
    batch.clean_cell_parameters(mode='hard')
    return rho


def basin_factors(H, live, caps, kT, tau, shape='hessian'):
    """[n, 12, 12] float64: the factor L of each basin's draw x = p + L z, z standard normal (docs/design/basin_set.md
    section 4), from its Hessian H [n, 12, 12] of the physical energy (energy units; MolCrystalData's cell parameters)
    on its live coordinates live [n, 12] bool. In coordinates scaled by caps (one cap per block: A, rad, fraction,
    rad), Hs = D H D / kT on the live block has eigenpairs (lambda_i, v_i), and each mode's width is
    w_i = sqrt(tau / max(|lambda_i|, tau)) <= 1: a mode with |lambda_i| above tau spends tau / 2 nats of the quadratic
    model |Hs| = V |Lambda| V^T (an unstable one at the width of |lambda_i|), a softer one is held at width 1 (its cap).
      'hessian'  L = D V diag(w): L L^T = tau kT H^-1 on the live block where H is positive and no mode is held
      'iso'      L = D sigma I on the live block, sigma^2 = sum_i |lambda_i| w_i^2 / sum_i |lambda_i|: the same
                 expected excess 1/2 tr(L^T D^-1 |Hs| D^-1 L) as 'hessian', spread over every scaled direction alike
                 (sigma <= 1)
    Dead coordinates get zero rows and columns."""
    if shape not in LIVE_SHAPES:
        raise ValueError(f"basin_factors: shape must be one of {LIVE_SHAPES}, got {shape!r}")
    H = torch.as_tensor(H, dtype=torch.float64).reshape(-1, 12, 12).cpu()
    live = torch.as_tensor(live).reshape(-1, 12).bool().cpu()
    Dg = torch.diag(torch.as_tensor(list(caps), dtype=torch.float64).repeat_interleave(3))
    L = torch.zeros_like(H)
    for i in range(H.shape[0]):
        m = live[i].nonzero().flatten()
        Hs = (Dg @ H[i] @ Dg)[m][:, m] / float(kT)
        lam, V = torch.linalg.eigh(0.5 * (Hs + Hs.T))
        w = torch.sqrt(float(tau) / lam.abs().clamp_min(float(tau))).clamp(max=1.0)
        Li = torch.zeros(12, 12, dtype=torch.float64)
        if shape == 'hessian':
            Li[m[:, None], m] = V * w[None, :]
        else:
            a = lam.abs()
            sigma2 = float((a * w ** 2).sum() / a.sum()) if float(a.sum()) > 0 else 1.0
            Li[m, m] = math.sqrt(min(sigma2, 1.0))
        L[i] = Dg @ Li
    return L


def basin_draw(p, hand, L, sg, radius, rows, chart, fd_coordinates=None, redraws=4, width=12, generator=None,
               pack=None):
    """(points [N, width] float32, ok [N] bool): fresh draws x = p[r] + L[r] z around the basins r in rows (p [n, 12],
    hand [n], L [n, 12, 12] from basin_factors, sg [n], radius [n]), each canonicalised into the chart:
      'fd'      fd_coordinates(sg, radius).from_params (energies/fd_coordinates.py::FDCoordinates; train.py passes
                MolecularCrystal.fd_coordinates_rows)
      'latent'  the trainer latent at a uniformly drawn trainer-box image
                (energies/molecular_crystal.py::latent_rows_from_params)
    A draw with no chart point is redrawn up to `redraws` times; a row still without one has ok False and zeros. The
    draws use `generator` (None: the global torch generator). pack(sg, points) -> [n, width] places a group's chart
    points in the state (an FD run's energies/fd_coordinates.py::FDSlotLayout.pack); None: in the first columns."""
    from energies.molecular_crystal import latent_rows_from_params
    if chart not in ('fd', 'latent'):
        raise ValueError(f"basin_draw: chart must be 'fd' or 'latent', got {chart!r}")
    rows = torch.as_tensor(rows).long().flatten().cpu()
    out = torch.zeros(len(rows), int(width), dtype=torch.float32)
    pending = torch.ones(len(rows), dtype=torch.bool)
    for _ in range(int(redraws) + 1):
        idx = torch.nonzero(pending).flatten()
        if idx.numel() == 0:
            break
        r = rows[idx]
        z = torch.randn(len(r), 12, dtype=torch.float64, generator=generator)
        params = p[r] + torch.einsum('nij,nj->ni', L[r], z)
        for g in sorted(set(sg[r].tolist())):
            sel = (sg[r] == g).nonzero().flatten()
            if chart == 'fd':
                res = fd_coordinates(int(g), radius[r[sel]]).from_params(params[sel], hand[r[sel]])
                ok = res.ok & torch.isfinite(res.s).all(1)
                pts = res.s[ok].float()
            else:
                lat, ok = latent_rows_from_params(params[sel], hand[r[sel]], int(g), radius[r[sel]],
                                                  generator=generator)
                pts = lat[ok]
            done = idx[sel[ok]]
            if pack is not None:
                out[done] = pack(int(g), pts)
            else:
                out[done, :pts.shape[1]] = pts
            pending[done] = False
    return out, ~pending


def identity(args, energy_function):
    """What is baked into a stored row's energy terms, so a file is read only by runs that would score it the same:
    the energy function and its weights, the analysis settings as resolved (MolecularCrystal.analyze_crystal_batch's
    defaults, then energy_config.analyze_kwargs), the MLIP density guard, the legacy triclinic wall switch, the space
    groups and Z' the run builds, and the eLJ coefficient the energies are scaled by. Not the state chart
    (energy_config.crystal_coordinates): the stored terms are the crystal's, and the chart's Jacobian is recomputed from
    the crystal whenever a reward is composed (prebuilt_sample_to_reward), so a latent-chart build serves an FD run."""
    analyze = dict(cutoff=10, supercell_size=10, std_orientation=False)
    analyze.update({k: v for k, v in dict(getattr(energy_function, 'analyze_kwargs', {}) or {}).items()
                    if k not in ('predictor', 'c', 'width')})
    mlip = getattr(args, 'mlip_path', None)
    ec = args.energy_config
    return {
        'energy_function': str(args.energy_function),
        'mlip': os.path.basename(str(mlip)) if mlip else None,
        'analyze_kwargs': {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                           for k, v in sorted(analyze.items())},
        'mlip_max_packing_coeff': getattr(ec, 'mlip_max_packing_coeff', None),
        'legacy_triclinic_walls': os.environ.get('MXT_LEGACY_TRICLINIC_WALLS'),
        'space_groups': [int(s) for s in args.space_groups],
        'z_primes': [int(z) for z in args.z_primes],
        'lj_coeff': (round(float(energy_function.lj_coeff), 9)
                     if getattr(energy_function, 'lj_coeff', None) is not None else None),
    }


def identity_mismatch(stored, current):
    """The identity keys on which a file and a run differ, as 'key: file X, run Y' strings (empty when they agree).
    The keys are the current definition's (identity): a key a file records that this code no longer counts as identity
    does not refuse it; a key the file lacks does. space_groups and z_primes agree when the file's are a subset of the
    run's: each stored row was scored under its own group and Z', which a run that also builds others scores the same
    (2026-11-04, for FD runs over several space groups, energies/fd_coordinates.py::FDSlotLayout)."""
    def same(k):
        a, b = stored.get(k), current.get(k)
        if k in ('space_groups', 'z_primes') and isinstance(a, list) and isinstance(b, list):
            return bool(a) and set(a) <= set(b)
        return a == b
    return [f"{k}: file {stored.get(k)!r}, run {current.get(k)!r}" for k in sorted(current) if not same(k)]


def condition_groups(batch):
    """[n] long: each row's condition as this file holds it -- (identifier, space group, Z') -- numbered in order of
    first appearance. The run's condition ids do not exist yet when the prior loads (init_identifiers runs after)."""
    ids = [str(m) for m in batch.identifier]
    sg = torch.as_tensor(batch.sg_ind).reshape(-1).tolist()
    zp = torch.as_tensor(batch.z_prime).reshape(-1).tolist()
    table, out = {}, []
    for key in zip(ids, sg, zp):
        out.append(table.setdefault(key, len(table)))
    return torch.tensor(out, dtype=torch.long)


def condition_floor(energy, groups, eligible):
    """[n] float64: each row's condition floor, the minimum of `energy` over the condition's eligible rows (+inf for a
    condition with none)."""
    n_groups = int(groups.max()) + 1 if groups.numel() else 0
    e = torch.where(eligible, energy.double(), torch.full_like(energy, float('inf'), dtype=torch.float64))
    floor = torch.full((n_groups,), float('inf'), dtype=torch.float64)
    floor = floor.scatter_reduce(0, groups, e, reduce='amin')
    return floor[groups]


def tag(batch, name):
    """The row tag `name` as a [n] long tensor on the CPU (raises when the batch does not carry it)."""
    if not hasattr(batch, name):
        raise ValueError(f"a pre-noised prior row carries no {name!r}: rebuild the file with "
                         f"data_processing/build_noised_prior.py")
    return torch.as_tensor(getattr(batch, name)).reshape(-1).long().cpu()


def check_rows(batch, n=CHECK_ROWS, seed=0):
    """Up to n row indices for the entry check, stratified: an equal share of each (kind, split, space group, Z')
    stratum present, drawn without replacement (a stratum smaller than its share is taken whole)."""
    kinds, splits = tag(batch, 'noised_kind'), tag(batch, 'noised_split')
    sg = torch.as_tensor(batch.sg_ind).reshape(-1).long().cpu()
    zp = torch.as_tensor(batch.z_prime).reshape(-1).long().cpu()
    key = ((kinds * 2 + splits) * 1000 + sg) * 10 + zp
    strata = torch.unique(key)
    share = max(1, n // max(len(strata), 1))
    rng = np.random.default_rng(seed)
    picked = []
    for s in strata.tolist():
        rows = torch.nonzero(key == s).flatten().numpy()
        picked.append(rows if len(rows) <= share else rng.choice(rows, share, replace=False))
    return np.sort(np.concatenate(picked))


def compare_terms(stored, fresh, energy_attr):
    """(worst relative-tolerance ratio, worst absolute miss, per-term worst misses) of a re-scored sample against
    its stored rows, over the energy attribute and CHECK_TERMS. A ratio <= 1 passes."""
    worst_ratio, worst_abs, per_term = 0.0, 0.0, {}
    for name in (energy_attr,) + CHECK_TERMS:
        a = torch.as_tensor(getattr(stored, name)).reshape(-1).double().cpu()
        b = torch.as_tensor(getattr(fresh, name)).reshape(-1).double().cpu()
        if a.shape != b.shape:
            raise ValueError(f"entry check: {name} has {a.numel()} stored and {b.numel()} fresh values")
        both = torch.isfinite(a) & torch.isfinite(b)
        if not bool((torch.isfinite(a) == torch.isfinite(b)).all()):
            return float('inf'), float('inf'), {name: float('inf')}
        d = (a[both] - b[both]).abs()
        tol = ABS_TOL + REL_TOL * a[both].abs()
        per_term[name] = float(d.max()) if d.numel() else 0.0
        if d.numel():
            worst_ratio = max(worst_ratio, float((d / tol).max()))
            worst_abs = max(worst_abs, float(d.max()))
    return worst_ratio, worst_abs, per_term


def window_energy(seed_energy, kinds, groups, basin_ids, manifest, by='row', parents=None):
    """[n] float64: the energy each row's energy window reads. A build whose children were drawn by the basin kernel
    (manifest child_kernel 'basin') windows a child by its basin's anchor's energy (the anchor of its condition group
    with its basin_id), so a basin passes or fails the energy window with every draw of it. Keyed on (condition group,
    basin_id), not on row indices, so it holds on a subset of the file's rows; a child whose basin has no anchor among
    the rows raises. Any other build: by 'row' (cfg:noised_prior.window_by) every row reads its own; by 'parent' a
    child reads its parent anchor's (parents: noised_parent, row indices into these same rows; a parent that is not an
    anchor raises). The packing window and the FD chart still cut draws one by one."""
    if (manifest or {}).get('child_kernel') != 'basin':
        if by == 'row':
            return seed_energy
        if by != 'parent':
            raise ValueError(f"window_energy: by must be one of {WINDOW_BY}, got {by!r}")
        if parents is None:
            raise ValueError("window_energy: by 'parent' needs the rows' noised_parent")
        kinds = torch.as_tensor(kinds).reshape(-1).long()
        parents = torch.as_tensor(parents).reshape(-1).long()
        child = kinds == KIND_CHILD
        bad = int((kinds[parents[child]] != KIND_ANCHOR).sum())
        if bad:
            raise ValueError(f"window_energy: {bad} children whose noised_parent is not an anchor among these rows")
        out = seed_energy.clone()
        out[child] = seed_energy[parents[child]]
        return out
    kinds = torch.as_tensor(kinds).reshape(-1).long()
    key = torch.as_tensor(groups).reshape(-1).long() * (1 << 32) + torch.as_tensor(basin_ids).reshape(-1).long()
    anchor = kinds == KIND_ANCHOR
    table = {}
    for k, e in zip(key[anchor].tolist(), seed_energy[anchor].tolist()):
        if k in table:
            raise ValueError(f"window_energy: two anchors of one basin (group, basin_id) {divmod(k, 1 << 32)}")
        table[k] = e
    child = kinds == KIND_CHILD
    missing = sorted({k for k in key[child].tolist() if k not in table})
    if missing:
        raise ValueError(f"window_energy: {len(missing)} basins have children but no anchor among these rows, e.g. "
                         f"(group, basin_id) {divmod(missing[0], 1 << 32)}")
    out = seed_energy.clone()
    idx = torch.nonzero(child).flatten()
    out[idx] = torch.tensor([table[k] for k in key[idx].tolist()], dtype=out.dtype)
    return out


def select_rows(seed_energy, packing, kinds, splits, groups, T, s, window_e=None):
    """The load-time selection, as boolean masks over the file's rows:
      floor_ok   rows a condition's floor is taken over: training split, finite, inside the packing window
      train      rows the run trains on: training split, a kind in train_kinds, inside both windows
      heldout    held-out anchor groups' rows of those kinds, inside both windows
      anchors    clean training anchors inside both windows (a readout when anchors are not trained on)
      buffer     every scored training row inside both windows, anchors and children whatever train_kinds holds: the
                 prior buffer's seed (train.py::Modeller.prior_buffer_source), so the noised rows the build already
                 scored serve the buffer-fed stages as well as, or instead of, MLE
    and the per-row floor. The energy window is window_kT x T above the floor (None: no energy window), both read on
    window_e (window_energy; None: seed_energy)."""
    lo, hi = s['packing_window']
    finite = torch.isfinite(seed_energy)
    in_pack = (packing >= lo) & (packing <= hi)
    floor_ok = (splits == SPLIT_TRAIN) & finite & in_pack
    we = seed_energy if window_e is None else window_e
    floor = condition_floor(we, groups, floor_ok & torch.isfinite(we))
    if s['window_kT'] is None:
        in_window = finite
    else:
        in_window = finite & ((we - floor) <= s['window_kT'] * T)
    kind_ok = torch.zeros_like(kinds, dtype=torch.bool)
    for name in s['train_kinds']:
        kind_ok |= kinds == KIND_NAMES[name]
    good = in_pack & in_window
    scored = (kinds == KIND_ANCHOR) | (kinds == KIND_CHILD)
    return dict(floor_ok=floor_ok, train=(splits == SPLIT_TRAIN) & kind_ok & good,
                buffer=(splits == SPLIT_TRAIN) & scored & good,
                heldout=(splits == SPLIT_HELDOUT) & kind_ok & good,
                anchors=(splits == SPLIT_TRAIN) & (kinds == KIND_ANCHOR) & good), floor


def strip_analysis_helpers(b):
    """Drop what the analysis leaves on a batch that is not row data: the lazy space-group tables, and its
    dict-valued attributes (`computes` holds bound methods, which pickle the whole batch object they belong to, on
    whatever device it was scored: a file saved with them is larger and does not load on a CPU-only host)."""
    from energies.molecular_crystal import SG_LOOKUP_TABLES
    for attr in SG_LOOKUP_TABLES:
        if hasattr(b, attr):
            delattr(b, attr)
    for key in [k for k in b.keys() if isinstance(b[k], dict)]:
        delattr(b, key)
    return b


def harvest_mask(resid, log_r, temperature, min_abs_resid, floor=None):
    """[n] bool, the rows train.py::Modeller.harvest_scored_rows keeps: a finite reward and a finite TB residual of at
    least min_abs_resid nats in magnitude, or (given each row's condition floor, energy units) a finite reward whose
    energy -T log R is at or below it."""
    keep = torch.isfinite(log_r) & torch.isfinite(resid) & (resid.abs() >= float(min_abs_resid))
    if floor is not None:
        keep |= torch.isfinite(log_r) & (-log_r * float(temperature) <= floor)
    return keep
