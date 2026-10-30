"""Build a pre-noised prior file (docs/design/prior_dataset_build.md): a source prior's anchors, K iso-tile children
of each, every row scored once through the run's own analysis, tagged (noised_prior.TAGS) and written in train.py's
prior layout with a build manifest.

    python -m data_processing.build_noised_prior --config configs/mk_dev.yaml --source SRC.pt --out OUT.pt \
        [--set energy_config.temperature=6.9 ...] [--k 10] [--log_lo -2.5] [--log_hi auto|-1.5] [--window_kT 6]
        [--heldout_frac 0.05] [--molecule_fraction 1] [--anchors all|is_anchor_image0] [--images auto|off]
        [--device cuda] [--chunk 4096] [--seed 0]

ANCHORS. Every row of SRC's equalized_prior, or (--anchors is_anchor_image0, the pooled MIPCAS priors) the rows its
'is_anchor' and 'image_id' tensors mark as anchor image 0. --molecule_fraction keeps the molecules whose molecule_u
is below it (data_processing.molecule_subset), as the trainers' --molecule_fraction does.

HELD OUT. Within each condition (identifier, space group, Z') the lowest-energy anchor stays in training; every
other anchor is held out with probability --heldout_frac, whole, with every child of it (noised_split SPLIT_HELDOUT).

CHILDREN. Each anchor is kicked K times by MolCrystalOps.log_noise_latent_parameters(log_lo, log_hi) (a unit
direction in the gauge-free latent times 10^U(log_lo, log_hi), clipped, decoded, hard-cleaned) and scored. With
--log_hi auto, log_hi is calibrated first: a pilot of PILOT_ROWS training anchors is kicked at each fixed radius
10^h for h in PILOT_GRID, and log_hi is the h whose median seed-energy excess over the parent
(MolecularCrystal.seed_energy_from, in kT at the run's temperature) lies nearest --window_kT; the table goes in the
manifest. Every finite child is kept whatever its energy: the energy and packing windows are load-time settings.
With --basins and --child_kernel basin, the K children of a basin are instead draws of its kernel
(noised_prior.basin_factors at --shape, --tau, --caps; noised_prior.basin_draw into the trainer latent at a uniformly
drawn trainer-box image; basin_kick): no calibration, no image step, a draw with no chart point after --redraws
redraws dropped and counted (manifest basin_kernel.misses). The manifest's child_kernel 'basin' makes the loader's energy
window read each child at its basin's energy (noised_prior.window_energy); the packing window still cuts draws one by one.

IMAGES (owner rule 2026-09-28: noise first, then the normaliser images). On space group 2, Z' 1, handedness +1
(data_processing/symmetrize_prior.py's scope), each child is written as one of its valid descriptions drawn
uniformly (the four y/z half shifts; eight on an x face), so the child set is image-symmetric in distribution.
Only aunit_centroid changes; stored energies carry over (symmetrize_prior proves them equal), and the trainer's entry
check re-scores a stratified sample of rows against them. --images off writes children as kicked.
"""
import argparse
import datetime
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import noised_prior as NP  # noqa: E402
from data_processing.molecule_subset import keep_molecules  # noqa: E402
from data_processing.symmetrize_prior import ROWS_PER_SOURCE, image_centroids  # noqa: E402

PILOT_ROWS = 2048
BASIN_ROUND_TRIP_TOL = 1e-3   # latent units: a decoded basin draw's latent_params against the latent it was decoded from
PHI_LATENT = 10               # Z' = 1 latent row of the orientation's azimuth, periodic on [-1, 1]
PILOT_GRID = tuple(-2.5 + 0.125 * i for i in range(17))   # 10^-2.5 .. 10^-0.5


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--config', required=True)
    p.add_argument('--source', default=None, help='a train.py prior file (its equalized_prior rows are the anchors)')
    p.add_argument('--basins', default=None, help='a basin set (mxtaltools crystal_search.basin_set, '
                   'docs/design/basin_set.md): its rows are the anchors, each tagged with its basin; exclusive of --source')
    p.add_argument('--thermal_scaling_factor', type=float, default=None,
                   help="with --basins on eLJ: the run's lj scale (a --source prior carries its own)")
    p.add_argument('--energy_tol', type=float, default=0.05,
                   help='with --basins: largest |rescored - converted basin energy| (run units) before a row counts as off')
    p.add_argument('--out', required=True)
    p.add_argument('--set', action='append', default=[], dest='overrides', metavar='KEY=VALUE')
    p.add_argument('--k', type=int, default=10, help='children per anchor')
    p.add_argument('--log_lo', type=float, default=None, help="--child_kernel iso: the iso tile's lowest log10 radius "
                   "(default -2.5)")
    p.add_argument('--log_hi', default='auto', help="'auto' (pilot calibration) or a number")
    p.add_argument('--window_kT', type=float, default=6.0, help='the calibration target, the load-time window')
    p.add_argument('--heldout_frac', type=float, default=0.05)
    p.add_argument('--molecule_fraction', type=float, default=1.0)
    p.add_argument('--anchors', choices=['all', 'is_anchor_image0'], default='all')
    p.add_argument('--images', choices=['auto', 'off'], default='auto')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--chunk', type=int, default=4096, help='rows per scoring call')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--child_kernel', choices=['iso', 'basin'], default='iso',
                   help="'iso': the latent iso tile (--log_lo/--log_hi); 'basin' (with --basins): draws of each "
                        "basin's kernel (noised_prior.basin_factors/basin_draw)")
    p.add_argument('--tau', type=float, default=None, help="--child_kernel basin: the kernel's tau (default 1: the "
                   "basin's thermal ellipsoid at the run's T)")
    p.add_argument('--shape', choices=list(NP.LIVE_SHAPES), default=None,
                   help='--child_kernel basin: its shape (default hessian)')
    p.add_argument('--caps', type=float, nargs=4, default=None,
                   help='--child_kernel basin: the width caps per block (A, rad, fraction, rad)')
    p.add_argument('--redraws', type=int, default=None,
                   help='--child_kernel basin: redraws of a draw with no chart point before it is dropped')
    p.add_argument('--limit_anchors', type=int, default=None, help='the first N anchors only (tests)')
    p.add_argument('--harvest', default=None, help='a glob of train.py harvest shards (cfg:harvest) to merge in')
    return p


def git_commit(path):
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None


def collate(batches):
    """One batch of every row of `batches`, in order, as batched_analyze_crystal_batch joins its own chunks
    (energies.molecular_crystal.merge_batches, after dropping the lazy space-group tables)."""
    from energies.molecular_crystal import merge_batches
    batches = list(batches)
    for b in batches:
        NP.strip_analysis_helpers(b)
    return merge_batches(batches)



def score(ef, batch, T, device, chunk):
    """The analysed batch (on the CPU) of every row of `batch`, through the energy function's own scoring of a stored
    prior row (train.py::Modeller._score_prior_rows), chunk rows at a time."""
    out = []
    for lo in range(0, batch.num_graphs, chunk):
        c = batch.subsample_new_batch(torch.arange(lo, min(lo + chunk, batch.num_graphs))).to(device)
        with torch.no_grad(), ef.allow_unreferenced():
            _, c = ef.batched_analyze_crystal_batch(
                c.latent_params(gauge_fix_free_axes=True), c,
                T * torch.ones(c.num_graphs, dtype=torch.float32, device=device),
                return_batch=True, internal_oom_recovery=True)
        out.append(c.cpu())
        del c
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    return out


def seed_energy(ef, batch, T, device, chunk=8192):
    """[n] float64: MolecularCrystal.seed_energy_from of every stored row (molecule + density + pressure terms; no
    Jacobian, penalty or clip), from the stored terms alone."""
    out = []
    for lo in range(0, batch.num_graphs, chunk):
        c = batch.subsample_new_batch(torch.arange(lo, min(lo + chunk, batch.num_graphs))).to(device)
        with torch.no_grad(), ef.allow_unreferenced():
            _, ens = ef.prebuilt_sample_to_reward(c, T, return_ens_dict=True)
        out.append(ef.seed_energy_from(ens).double().cpu().flatten())
    return torch.cat(out)


def load_harvest(pattern, identity, keys, molecules):
    """(the rows of every harvest shard matching `pattern` whose molecule is among `molecules`, reduced to the attribute
    names `keys` the source rows carry, or None; the files read). A shard scored under another energy definition
    (noised_prior.identity) raises."""
    import glob
    files = sorted(glob.glob(pattern))
    parts = []
    for f in files:
        d = torch.load(f, weights_only=False, map_location='cpu')
        bad = NP.identity_mismatch(d['identity'], identity)
        if bad:
            raise SystemExit(f"harvest shard {f} was scored under another energy definition: " + '; '.join(bad))
        b = NP.strip_analysis_helpers(d['rows'])
        for key in [k for k in b.keys() if k not in keys and k not in ('batch', 'ptr')]:
            delattr(b, key)
        keep = [i for i, m in enumerate(b.identifier) if str(m) in molecules]
        if keep:
            parts.append(b.subsample_new_batch(torch.tensor(keep)))
    return (collate(parts) if parts else None), files


def kick(batch, lo, hi, device):
    b = batch.to(device)
    mag = b.log_noise_latent_parameters(lo, hi)
    return b, mag


def check_child_kernel_args(args):
    """Refuse flags the chosen --child_kernel does not read, fill the defaults of those it does, and validate the basin
    kernel's (as noised_prior.settings does the trainer's)."""
    kernel_flags = {k: getattr(args, k) for k in ('tau', 'shape', 'caps', 'redraws') if getattr(args, k) is not None}
    if args.child_kernel == 'basin':
        if args.basins is None or str(args.log_hi) != 'auto' or args.images != 'auto' or args.log_lo is not None:
            raise SystemExit("--child_kernel basin needs --basins, and takes no --log_lo, --log_hi or --images (its "
                             "draws are the basin kernel's, each at a uniformly drawn trainer-box image)")
        args.tau = 1.0 if args.tau is None else args.tau
        args.shape = 'hessian' if args.shape is None else args.shape
        args.caps = list(NP.DEFAULTS['live_caps']) if args.caps is None else list(args.caps)
        args.redraws = NP.DEFAULTS['live_redraws'] if args.redraws is None else args.redraws
        if not (math.isfinite(args.tau) and args.tau > 0) or not all(math.isfinite(c) and c > 0 for c in args.caps):
            raise SystemExit(f"--tau and every --caps entry must be finite and > 0, got {args.tau}, {args.caps}")
        if args.redraws < 0:
            raise SystemExit(f"--redraws must be >= 0, got {args.redraws}")
    else:
        if kernel_flags:
            raise SystemExit(f"{sorted('--' + k for k in kernel_flags)} are read by --child_kernel basin only")
        args.log_lo = -2.5 if args.log_lo is None else args.log_lo


def basin_kick(anchors, rep, L, args, generator):
    """(batch, rows, misses) for --child_kernel basin: one draw of the basin kernel (noised_prior.basin_draw, factor L
    from noised_prior.basin_factors) per entry of rep (anchor indices), canonicalised into the trainer latent at a
    uniformly drawn trainer-box image and decoded onto a copy of its anchor (latent_to_cell_params; the latent lies
    inside the decode clamp, so the crystal is the draw). A draw with no latent after --redraws redraws is dropped:
    rows are the anchor index of each kept draw, misses [n_anchors] the dropped draws per anchor."""
    p = torch.as_tensor(anchors.basin_p, dtype=torch.float64).reshape(-1, 12).cpu()
    hand = torch.as_tensor(anchors.basin_hand, dtype=torch.float64).reshape(-1).cpu()
    sg = torch.as_tensor(anchors.sg_ind).reshape(-1).cpu()
    radius = torch.as_tensor(anchors.radius).reshape(-1).double().cpu()
    lat, ok = NP.basin_draw(p, hand, L, sg, radius, rep, 'latent', redraws=args.redraws, generator=generator)
    misses = torch.bincount(rep[~ok], minlength=anchors.num_graphs)
    rows = rep[ok]
    b = anchors.subsample_new_batch(rows).to(args.device)
    b.latent_to_cell_params(lat[ok].to(args.device))
    # score() builds from latent_params: it must read back the draw, or the scored crystal is another one
    back = b.latent_params(gauge_fix_free_axes=True).detach().float().cpu()
    d = (back - lat[ok]).abs()
    d[:, PHI_LATENT] = torch.minimum(d[:, PHI_LATENT], 2.0 - d[:, PHI_LATENT])   # the azimuth is periodic: -1 is +1
    off = d.max(1).values if len(rows) else torch.zeros(0)
    if bool((off > BASIN_ROUND_TRIP_TOL).any()):
        raise SystemExit(f"basin children: {int((off > BASIN_ROUND_TRIP_TOL).sum())} decoded draws do not read back "
                         f"their latent (worst {float(off.max()):.3g} > {BASIN_ROUND_TRIP_TOL})")
    return b, rows, misses


def calibrate(ef, anchors, e_anchor, train_idx, T, args, rng):
    """(log_hi, table): the pilot of PILOT_ROWS training anchors kicked at each fixed radius in PILOT_GRID."""
    pick = rng.choice(train_idx, min(PILOT_ROWS, len(train_idx)), replace=False)
    base = anchors.subsample_new_batch(torch.as_tensor(pick))
    table = []
    for h in PILOT_GRID:
        torch.manual_seed(args.seed + 11)
        b, _ = kick(base.subsample_new_batch(torch.arange(base.num_graphs)), h, h, args.device)
        scored = collate(score(ef, b, T, args.device, args.chunk))
        du = (seed_energy(ef, scored, T, args.device) - e_anchor[torch.as_tensor(pick)]) / T
        ok = torch.isfinite(du)
        q = torch.quantile(du[ok], torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)).tolist() if ok.any() else [math.nan] * 3
        inside = float((du[ok] <= args.window_kT).double().mean()) if ok.any() else math.nan
        table.append(dict(log_radius=h, excess_kT_p10=q[0], excess_kT_p50=q[1], excess_kT_p90=q[2],
                          inside_window=inside, finite=float(ok.double().mean())))
        print(f"  pilot radius 10^{h:+.2f}: excess over parent (kT) P10 {q[0]:.2f} P50 {q[1]:.2f} P90 {q[2]:.2f}, "
              f"{100 * inside:.1f}% within {args.window_kT:g} kT, {100 * float(ok.double().mean()):.1f}% finite",
              flush=True)
    best = min(table, key=lambda r: abs(r['excess_kT_p50'] - args.window_kT) if math.isfinite(r['excess_kT_p50'])
               else math.inf)
    log_hi = best['log_radius']
    if not (0.5 * args.window_kT <= best['excess_kT_p50'] <= 2.0 * args.window_kT):
        print(f"  pilot: no radius put the median excess within a factor 2 of {args.window_kT:g} kT; "
              f"log_hi falls back to -1.5", flush=True)
        log_hi = -1.5
    return max(log_hi, args.log_lo), table


def choose_images(batch, rng):
    """Each row rewritten as one of its valid descriptions, drawn uniformly (see IMAGES in the module docstring)."""
    images, _, _, face_copy, _ = image_centroids(batch.aunit_centroid)
    face = face_copy.view(-1, ROWS_PER_SOURCE)[:, 4]          # a source on an x face has a real copy at slot 4
    n = batch.num_graphs
    k = torch.where(face, torch.as_tensor(rng.integers(0, 8, n)), torch.as_tensor(rng.integers(0, 4, n)))
    batch.aunit_centroid = images.view(n, ROWS_PER_SOURCE, 3)[torch.arange(n), k]
    return int(face.sum())


def refuse_unscored_chart(run_args):
    """score() hands the analysis each row's trainer latent (latent_params) as x, which a run on
    energy_config.crystal_coordinates 'fd' would decode as an FD chart row: every row mis-built (seen 2026-10-30:
    packing coefficients 2.6-14). Build on the latent chart; the trainer's FD loader re-describes the rows."""
    chart = getattr(run_args.energy_config, 'crystal_coordinates', 'latent')
    if chart != 'latent':
        raise SystemExit(f"build_noised_prior scores rows on the latent chart only; got energy_config.crystal_coordinates "
                         f"{chart!r}. Build without it: an FD run loads a latent-scored build")


def load_source_anchors(args, run_args, ef):
    """(anchors, thermal_scaling_factor or None, description) of a train.py prior file: its equalized_prior rows (or
    its anchor image-0 rows), sets ef.lj_coeff from the file's thermal_scaling_factor."""
    src = torch.load(args.source, weights_only=False, map_location='cpu')
    if NP.MANIFEST_KEY in src:
        raise SystemExit(f"{args.source} is already a pre-noised build")
    tsf = src.get('thermal_scaling_factor', None)
    if tsf is not None:
        if run_args.energy_function != 'elj' and abs(float(tsf) - 1.0) > 1e-9:
            raise SystemExit(f"thermal_scaling_factor {tsf} on a {run_args.energy_function} prior: see "
                             f"train.py::Modeller.init_prior_dataset")
        ef.lj_coeff = float(tsf)
    full = src['equalized_prior']
    if args.anchors == 'is_anchor_image0':
        rows = torch.nonzero(src['is_anchor'] & (src['image_id'] == 0)).flatten()
        anchors = full.subsample_new_batch(rows)
        what = f"{rows.numel()} anchor image-0 rows of {full.num_graphs}"
    else:
        anchors = full
        what = f"all {full.num_graphs} rows"
    del src, full
    if args.molecule_fraction < 1.0:
        anchors, n_mol = keep_molecules(anchors, args.molecule_fraction)
        what += f", molecule fraction {args.molecule_fraction:g} ({n_mol} molecules)"
    if args.limit_anchors is not None and anchors.num_graphs > args.limit_anchors:
        anchors = anchors.subsample_new_batch(torch.arange(args.limit_anchors))
        what += f", first {args.limit_anchors}"
    anchors['smiles'] = None
    return anchors, tsf, what


def load_basin_anchors(args, run_args):
    """(anchors, thermal_scaling_factor or None, description, basin tags) of a basin set (docs/design/basin_set.md):
    one anchor per usable basin, built as the set describes it (crystal_fd ``to_mol_crystal_data``, mode 'body', the
    trainer's convention); the tags carry each basin's draw centre p, its handedness, its Hessian and energy in the
    run's currency, and its live coordinates. Refuses a set whose energy is not the run's."""
    from mxtaltools.crystal_search.basin_set import read_basin_set
    from mxtaltools.dataset_utils.utils import collate_data_list
    if args.harvest:
        raise SystemExit('--harvest is not a basin-set input (owner 2026-10-30)')
    if args.anchors != 'all' or args.molecule_fraction < 1.0:
        raise SystemExit('--anchors and --molecule_fraction apply to --source only')
    ds, header = read_basin_set(args.basins)
    energy = header['energy']
    if energy.get('target') != run_args.energy_function:
        raise SystemExit(f"basin set energy {energy.get('target')!r} is not the run's {run_args.energy_function!r}: "
                         f"transfer first (a seeded streaming search under the run's energy)")
    if getattr(run_args.energy_config, 'lj_rescale', None) is not None:
        raise SystemExit('--basins does not convert lj_rescale; build without it')
    tsf = args.thermal_scaling_factor
    if run_args.energy_function == 'elj':
        if tsf is None:
            raise SystemExit("--basins on eLJ needs --thermal_scaling_factor (the run's lj scale)")
        if energy.get('per') != 'asymmetric_unit':
            raise SystemExit(f"eLJ basin energies are per asymmetric unit, the set says {energy.get('per')!r}")
        factor = float(tsf)
    else:
        if tsf is not None:
            raise SystemExit('--thermal_scaling_factor is for eLJ only')
        factor = 1.0
    usable = torch.tensor([ds.status_names[int(c)] in ('canonical', 'special_metric') for c in ds.status])
    ds = ds.select(usable)
    rows = ds.to_mol_crystal_data(mode='body')
    anchors = collate_data_list(rows)
    anchors['smiles'] = None
    c = ds.columns
    tags = dict(basin_id=c['basin_id'].long(), basin_p=c['p'].double(), basin_hand=c['p_handedness'].double(),
                basin_H=c['H'].double() * factor, basin_live=c['live_mask'].bool(), basin_E=c['E'].double() * factor)
    what = (f"basin set {args.basins}: {len(ds)} usable of {int(usable.numel())} rows "
            f"({energy.get('target')}, x {factor:g} to run units, search kT {header['search'].get('kT')})")
    return anchors, tsf, what, dict(tags=tags, header=header, factor=factor, path=os.path.abspath(args.basins))


def check_basin_energies(anchors, basins, tol):
    """[B] bool, the anchors that rescore to their basin's converted search energy within ``tol`` (run units). A row
    that does not has no faithful chart point: score() builds from the trainer latent, which clips a cell outside the
    trainer's length box into another crystal. Such rows are dropped and counted, never kept clipped; more than 20%
    of rows off refuses the set (frame, chart or energy definition differs)."""
    got = torch.as_tensor(anchors.mol_energy, dtype=torch.float64).reshape(-1)
    want = basins['tags']['basin_E'].reshape(-1)
    off = (got - want).abs()
    keep = off <= tol
    print(f"basin energies: rescored vs converted |diff| median {float(off.median()):.3g}, max {float(off.max()):.3g}; "
          f"{int((~keep).sum())} of {off.numel()} rows past {tol:g} dropped (no faithful chart point)", flush=True)
    if float((~keep).double().mean()) > 0.2:
        raise SystemExit('basin set does not rescore to its stored energies: frame, chart or energy definition differs')
    return keep

def main(argv=None):
    args = build_parser().parse_args(argv)
    from inn_modeller import apply_overrides
    from train import energy_function_kwargs
    from energies.molecular_crystal import MolecularCrystal
    from utils import get_train_args

    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    run_args = get_train_args(['--config', args.config])
    apply_overrides(run_args, args.overrides)
    T = float(run_args.energy_config.temperature)
    refuse_unscored_chart(run_args)
    ef = MolecularCrystal(**energy_function_kwargs(run_args, args.device))

    if (args.source is None) == (args.basins is None):
        raise SystemExit('give exactly one of --source and --basins')
    basin_children = args.child_kernel == 'basin'
    check_child_kernel_args(args)
    basins = None
    if args.basins is not None:
        anchors, tsf, what, basins = load_basin_anchors(args, run_args)
        ef.lj_coeff = 1.0 if tsf is None else float(tsf)
    else:
        anchors, tsf, what = load_source_anchors(args, run_args, ef)
    print(f"anchors: {what}: {anchors.num_graphs} rows; T {T:g}, lj_coeff {ef.lj_coeff}", flush=True)

    # HARVEST (cfg:harvest shards of earlier runs): rows of the source's molecules join the pool, are scored with it,
    # and a harvested row below its condition's floor over the source anchors becomes an anchor (kicked like one);
    # the rest are written once, kind KIND_HARVESTED, no children
    n_src, harvest_files, rest = anchors.num_graphs, [], None
    if args.harvest:
        harvested, harvest_files = load_harvest(args.harvest, NP.identity(run_args, ef), set(anchors.keys()),
                                                {str(m) for m in anchors.identifier})
        if harvested is not None:
            harvested['smiles'] = None
            anchors = collate([anchors, harvested])
        print(f"harvest: {0 if harvested is None else harvested.num_graphs} rows of these molecules from "
              f"{len(harvest_files)} shard(s)", flush=True)

    t0 = time.time()
    anchors = collate(score(ef, anchors, T, args.device, args.chunk))
    e_anchor = seed_energy(ef, anchors, T, args.device)
    if basins is not None:
        keep = check_basin_energies(anchors, basins, args.energy_tol)
        basins['dropped_off_chart'] = int((~keep).sum())
        if not bool(keep.all()):
            rows_kept = torch.nonzero(keep).flatten()
            anchors, e_anchor = anchors.subsample_new_batch(rows_kept), e_anchor[rows_kept]
            basins['tags'] = {k: v[rows_kept] for k, v in basins['tags'].items()}
        for k, v in basins['tags'].items():
            anchors.add_graph_attr(v, k)
    print(f"anchors scored in {time.time() - t0:.0f} s; seed energy min {float(e_anchor.min()):.2f} "
          f"P50 {float(e_anchor.median()):.2f} max {float(e_anchor.max()):.2f}", flush=True)
    n_promoted = 0
    if anchors.num_graphs > n_src:
        groups0 = NP.condition_groups(anchors)
        is_src = torch.arange(anchors.num_graphs) < n_src
        floor0 = NP.condition_floor(e_anchor, groups0, is_src & torch.isfinite(e_anchor))
        promote = ~is_src & torch.isfinite(e_anchor) & (e_anchor < floor0)
        kick_rows = torch.nonzero(is_src | promote).flatten()
        rest_rows = torch.nonzero(~is_src & ~promote).flatten()
        n_promoted = int(promote.sum())
        rest = anchors.subsample_new_batch(rest_rows) if rest_rows.numel() else None
        anchors, e_anchor = anchors.subsample_new_batch(kick_rows), e_anchor[kick_rows]
        print(f"harvest: {n_promoted} rows below their condition's floor become anchors; "
              f"{rest_rows.numel()} are kept as harvested rows", flush=True)

    # held out whole, per condition, never a condition's lowest anchor
    groups = NP.condition_groups(anchors)
    lowest = torch.zeros(anchors.num_graphs, dtype=torch.bool)
    e_fin = torch.where(torch.isfinite(e_anchor), e_anchor, torch.full_like(e_anchor, float('inf')))
    order = torch.argsort(e_fin)
    seen = set()
    for i in order.tolist():
        g = int(groups[i])
        if g not in seen:
            seen.add(g)
            lowest[i] = True
    held = torch.as_tensor(rng.random(anchors.num_graphs) < args.heldout_frac) & ~lowest
    split = torch.where(held, NP.SPLIT_HELDOUT, NP.SPLIT_TRAIN)
    train_idx = torch.nonzero(~held).flatten().numpy()
    print(f"held out: {int(held.sum())} of {anchors.num_graphs} anchors over {len(seen)} conditions", flush=True)

    kernel = None
    if basin_children:
        log_hi, table = None, None
        kernel = dict(shape=args.shape, tau=args.tau, caps=list(args.caps), redraws=args.redraws)
        L = NP.basin_factors(anchors.basin_H, anchors.basin_live, args.caps, T, args.tau, args.shape)
        widths = torch.linalg.norm(L, dim=2)   # per coordinate: the std of (L z)_i
        print(f"children: K {args.k} per basin from its kernel ({args.shape}, tau {args.tau:g}, caps "
              f"{list(args.caps)}): width medians per block (A, rad, frac, rad) "
              f"{[round(float(widths[:, 3 * j:3 * j + 3].median()), 4) for j in range(4)]}", flush=True)
    elif str(args.log_hi) == 'auto':
        print(f"calibrating log_hi on {min(PILOT_ROWS, len(train_idx))} training anchors (target median excess "
              f"{args.window_kT:g} kT at T {T:g})", flush=True)
        log_hi, table = calibrate(ef, anchors, e_anchor, train_idx, T, args, rng)
    else:
        log_hi, table = float(args.log_hi), None
    if not basin_children:
        print(f"children: K {args.k} per anchor at 10^U({args.log_lo:g}, {log_hi:g})", flush=True)

    sg = set(torch.as_tensor(anchors.sg_ind).reshape(-1).tolist())
    zp = set(torch.as_tensor(anchors.z_prime).reshape(-1).tolist())
    hand = set(torch.as_tensor(anchors.aunit_handedness).reshape(-1).tolist())
    images = not basin_children and args.images == 'auto' and sg == {2} and zp == {1} and hand == {1.0}
    if args.images == 'auto' and not images and not basin_children:
        print(f"images: not applied (space groups {sorted(sg)}, Z' {sorted(zp)}, handedness {sorted(hand)}: outside "
              f"symmetrize_prior's scope)", flush=True)

    t0 = time.time()
    torch.manual_seed(args.seed + 1)
    gen = torch.Generator().manual_seed(args.seed + 1)
    children, parents, mags, n_face = [], [], [], 0
    misses = torch.zeros(anchors.num_graphs, dtype=torch.long)
    per_call = max(1, args.chunk // max(args.k, 1))
    for lo in range(0, anchors.num_graphs if args.k > 0 else 0, per_call):
        ia = torch.arange(lo, min(lo + per_call, anchors.num_graphs))
        rep = ia.repeat_interleave(args.k)
        if basin_children:
            b, rep, miss = basin_kick(anchors, rep, L, args, gen)
            misses += miss
            mag = torch.zeros(len(rep))
        else:
            b, mag = kick(anchors.subsample_new_batch(rep), args.log_lo, log_hi, args.device)
        scored = score(ef, b, T, args.device, args.chunk)
        for s in scored:
            if images:
                n_face += choose_images(s, rng)
        children.extend(scored)
        parents.append(rep)
        mags.append(mag.detach().cpu())
        if (lo // per_call) % 20 == 0:
            done = min(lo + per_call, anchors.num_graphs)
            print(f"  children of {done} / {anchors.num_graphs} anchors, {time.time() - t0:.0f} s", flush=True)
    children = collate(children) if children else None
    parents = torch.cat(parents) if parents else torch.zeros(0, dtype=torch.long)
    mags = torch.cat(mags) if mags else torch.zeros(0)
    print(f"children scored in {time.time() - t0:.0f} s: {0 if children is None else children.num_graphs} rows"
          + (f", images drawn ({n_face} on an x face)" if images else "")
          + (f"; {int(misses.sum())} draws with no chart point dropped (most of one basin: {int(misses.max())})"
             if basin_children and anchors.num_graphs else ""), flush=True)
    if kernel is not None:
        kernel.update(misses=int(misses.sum()), misses_max_per_basin=int(misses.max()) if misses.numel() else 0)

    n_a = anchors.num_graphs
    anchors.add_graph_attr(torch.full((n_a,), NP.KIND_ANCHOR, dtype=torch.long), 'noised_kind')
    anchors.add_graph_attr(split.long(), 'noised_split')
    anchors.add_graph_attr(torch.arange(n_a, dtype=torch.long), 'noised_parent')
    anchors.add_graph_attr(torch.zeros(n_a), 'noised_radius')
    n_c = 0 if children is None else children.num_graphs
    parts = [anchors]
    if children is not None:
        children.add_graph_attr(torch.full((n_c,), NP.KIND_CHILD, dtype=torch.long), 'noised_kind')
        children.add_graph_attr(split[parents].long(), 'noised_split')
        children.add_graph_attr(parents.long(), 'noised_parent')
        children.add_graph_attr(mags.float(), 'noised_radius')
        if basins is not None:   # a child's basin is its parent's
            for k, v in basins['tags'].items():
                children.add_graph_attr(v[parents], k)
        parts.append(children)
    n_h = 0 if rest is None else rest.num_graphs
    if rest is not None:
        rest.add_graph_attr(torch.full((n_h,), NP.KIND_HARVESTED, dtype=torch.long), 'noised_kind')
        rest.add_graph_attr(torch.full((n_h,), NP.SPLIT_TRAIN, dtype=torch.long), 'noised_split')
        rest.add_graph_attr(torch.full((n_h,), -1, dtype=torch.long), 'noised_parent')
        rest.add_graph_attr(torch.zeros(n_h), 'noised_radius')
        parts.append(rest)
    out = collate(parts)

    # the build's own view of the load-time window, for the record
    e_all = seed_energy(ef, out, T, args.device)
    groups_all = NP.condition_groups(out)
    kind, spl = NP.tag(out, 'noised_kind'), NP.tag(out, 'noised_split')
    we = NP.window_energy(e_all, kind, groups_all, getattr(out, 'basin_id', None), {'child_kernel': args.child_kernel})
    floor = NP.condition_floor(we, groups_all, (spl == NP.SPLIT_TRAIN) & torch.isfinite(we))
    du = (we - floor) / T
    child_tr = (kind == NP.KIND_CHILD) & (spl == NP.SPLIT_TRAIN) & torch.isfinite(du)
    inside = float((du[child_tr] <= args.window_kT).double().mean()) if child_tr.any() else math.nan
    below_parent = ((e_all[n_a:n_a + n_c] - e_anchor[parents]) / T < -args.window_kT)
    print(f"window check: {100 * inside:.1f}% of training children within {args.window_kT:g} kT of their condition "
          f"floor; {int(below_parent.sum())} children more than {args.window_kT:g} kT below their parent", flush=True)

    manifest = dict(
        format_version=NP.FORMAT_VERSION,
        identity=NP.identity(run_args, ef),
        source=os.path.abspath(args.source) if args.source else None, anchors=args.anchors,
        molecule_fraction=args.molecule_fraction,
        basin_set=None if basins is None else dict(path=basins['path'], header=basins['header'],
                                                   to_run_units=basins['factor'],
                                                   dropped_off_chart=basins['dropped_off_chart']),
        child_kernel=args.child_kernel, basin_kernel=kernel,
        k=args.k, log_lo=None if basin_children else args.log_lo, log_hi=log_hi, calibration=table,
        window_kT_target=args.window_kT,
        temperature=T, heldout_frac=args.heldout_frac, images=bool(images), seed=args.seed,
        harvest=dict(files=harvest_files, promoted_to_anchor=n_promoted, kept_as_harvested=n_h),
        counts=dict(anchors=n_a, children=n_c, harvested=n_h, heldout_anchors=int(held.sum()), conditions=len(seen),
                    children_within_window=inside, children_far_below_parent=int(below_parent.sum())),
        commits=dict(gfn=git_commit(HERE), mxtaltools=git_commit(os.path.dirname(os.path.dirname(
            __import__('mxtaltools').__file__)))),
        built=datetime.datetime.now().isoformat(timespec='seconds'),
        build_seconds=round(time.time() - t_start, 1),
    )
    payload = {'prior': out, 'equalized_prior': out, NP.MANIFEST_KEY: manifest}
    if tsf is not None:
        payload['thermal_scaling_factor'] = float(tsf)
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists: this tool never overwrites")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + '.partial'
    torch.save(payload, tmp)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}: {out.num_graphs} rows ({n_a} anchors, {n_c} children), "
          f"{os.path.getsize(args.out) / 1e6:.1f} MB, {time.time() - t_start:.0f} s", flush=True)
    return manifest


if __name__ == '__main__':
    main()
