"""Density engine: many draws (e.g. 100k) per problem, scored ONCE, with the intermediates the landscape analysis needs.

    ONLY_RUN=<run_name> NUM=100000 python eval/paper1_results/dens_engine.py

Reads the run's entry in new_analysis.yaml (checkpoint, dataset_path, rdf_mode, exp_sample_path, identifiers, batch
sizes) and writes a bundle to <results_dir>/dens/<run_name>_<N>/. Every stage is resumable: a finished artifact is
skipped on rerun and a partial one continues where it stopped. Every resume is checked against what the bundle was
started with (draw settings in draw_meta.json, stage-4 fingerprint in stream_progress.json) and refuses on a mismatch.

  1. draw + score, in shards (shards/shard_XXXX.pt, listed in shards/index.json): raw and clipped latents,
     log P_F / log P_B, the problem's energy (and elj), packing coefficient, reduction penalty, cell parameters, and
     each draw's normalised RDF CDF (fp16). That is everything needed to rebuild a crystal (latent_to_cell_params) or
     recompute any distance later.
  2. assemble: draws.pt (per-draw arrays, the 15 kT window and reduced-cell masks) and cdf.npy ([N, channels*bins]),
     one shard in memory at a time.
  3. references: experimental structures scored on the same footing as the draws (refs.pt).
  4. stream distances: every draw and every reference against every draw, in blocks; N x N is never stored. Keeps
       - kernel sums at sigma x SIGMA_MULTS over three draw sets -- all draws, the 15 kT window, window & reduced --
         leave-one-out for draws, in draws' worth of weight (sums.npy [N+R, n_sigma, 3]);
       - each draw's K nearest draws, self excluded (knn_idx.npy int32, knn_dist.npy float32, [N, K]);
       - every reference's full distance row (ref_dist.npy [R, N]);
       - every draw's neighbours WITHIN A CUTOFF (the bundle's own probe d_cut), self excluded, with distances
         (nbrs/nb_<r0>_<r1>.npz per row block: counts int32 [rows], idx int32, dist fp16, row-major);
       - a distance histogram per draw and reference (hist.npy uint16 [N+R, 3 sets, HIST_BINS], bins of equal width
         on [0, 4 x probe d_cut)), from which kernel sums at ANY width up to 2 x sigma0 follow without distances
         (kernel_sums_from_hist). Both are independent of the kernel width the pass was run with.
     Progress is recorded per [r0, r1) block (stream_progress.json) and coverage is checked before sigma.json is
     written, so an interrupted pass resumes and an incomplete one cannot pass for complete.

WHAT IT IS NOT. The K-neighbour lists serve embeddings, HDBSCAN and nearest-structure queries; they do NOT support the
analysis's mean-shift basin step at 100k (its radii hold tens of thousands of draws). Basins stay on a display subset
(its own small distance matrix, recomputable from cdf.npy) with densities read from sums.npy.

DISTANCE. compute_rdf_distance is a per-channel L1 between normalised RDF CDFs times the bin width, averaged over the
channels active (non-empty) in either structure. With no empty channel that is exactly bin_width / n_channels times
cdist_L1 on the flattened CDFs; when a channel is empty in both structures it is dropped from that pair's average,
which the stream corrects per pair from the stored emptiness masks. Stage 1 checks the fp16 CDFs against the original
RDFs on every shard, and stage 4 checks the block distance against compute_rdf_distance before it starts.

SIGMA. The kernel width is sigma0 = d_cut / 3, as in new_analysis.py. SIGMA_FROM selects d_cut: the run's 10k
new_analysis results (default; must exist and match the bundle's rdf_mode and checkpoint step, so the bundle sits on
the published figures' kernel), 'probes' (500 random window draws), or a number. The probe estimate is always recorded.
"""
import hashlib
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from energy_sampling.eval.paper1_results.new_analysis import (   # noqa: E402
    load_gfn, load_predictor, sample_latents, decode_and_score, score_experimental, resolve_molecule,
    LEGACY_TRICLINIC_WALLS)
from energy_sampling.utils import load_yaml, dict2namespace   # noqa: E402
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance   # noqa: E402
from mxtaltools.dataset_utils.utils import collate_data_list   # noqa: E402

SIGMA_MULTS = (0.5, 0.75, 1.0, 1.5, 2.0)
SETS = ('all', 'window', 'window_reduced')
MAX_REDUCTION_EN = 1e-6
HIST_BINS = 1024


# ----------------------------------------------------------------------------- helpers
def rdf_to_cdf(rdf):
    """Normalised per-channel CDF, flattened, fp16; plus the per-channel emptiness mask."""
    x = rdf.double()
    empty = x.sum(-1) <= 1e-12
    x = x / (x.sum(-1, keepdim=True) + 1e-10)
    return torch.cumsum(x, -1).flatten(1).half(), empty


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _atomic_json(path, obj):
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(path + '.tmp', path)


def _check_same(old, new, what):
    diff = {k: (old.get(k, '<absent>'), v) for k, v in new.items() if old.get(k, '<absent>') != v}
    assert not diff, (f"{what}: this bundle was started with different settings {diff} (existing, now). Use a new OUT "
                      f"directory, or restore the original settings.")


class PairDistance:
    """Exact compute_rdf_distance for blocks of rows against blocks of columns, from fp16 CDFs."""

    def __init__(self, n_channels, n_bins, any_empty):
        self.C, self.bw = n_channels, 10.0 / (n_bins - 1)
        self.any_empty = any_empty

    def __call__(self, a_cdf, a_empty, b_cdf, b_empty):
        l1 = torch.cdist(a_cdf.float(), b_cdf.float(), p=1)
        if not self.any_empty:
            return l1 * (self.bw / self.C)
        # channels empty in BOTH structures are inactive; their CDF difference is 0, so only the count changes
        both_empty = a_empty.float() @ b_empty.float().T
        return l1 * self.bw / (self.C - both_empty).clamp(min=1)


# ----------------------------------------------------------------------------- stage 1
def _pinned(cfg, num, shard_size, seed):
    """Draw settings that must not change within one bundle (checked on every resume)."""
    return {'checkpoint': cfg.checkpoint, 'rdf_mode': cfg.rdf_mode,
            'std_orientation': bool(getattr(cfg, 'std_orientation', True)), 'num': num, 'shard_size': shard_size,
            'seed': seed, 'dataset_path': os.path.basename(cfg.dataset_path),
            'molecule_path': os.path.basename(getattr(cfg, 'molecule_path', '') or '') or None,
            'kT': float(cfg.kT), 'max_reduction_en': MAX_REDUCTION_EN}


def stage_draws(cfg, out, num, shard_size, seed, device):
    shard_dir = os.path.join(out, 'shards')
    os.makedirs(shard_dir, exist_ok=True)
    assert cfg.rdf_mode in ('atomwise', 'envwise'), f"set rdf_mode to 'atomwise' or 'envwise' (got {cfg.rdf_mode!r})"
    pinned = _pinned(cfg, num, shard_size, seed)
    meta_path = os.path.join(out, 'draw_meta.json')
    if os.path.exists(meta_path):
        _check_same(json.load(open(meta_path)), pinned, 'stage 1')
    n_shards = int(np.ceil(num / shard_size))
    index_path = os.path.join(shard_dir, 'index.json')
    index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    todo = [s for s in range(n_shards) if str(s) not in index
            or not os.path.exists(os.path.join(shard_dir, f'shard_{s:04d}.pt'))]
    if not todo:
        _log(f"stage 1: all {n_shards} shards present")
        return
    ck = os.path.join(cfg.checkpoints_dir, cfg.checkpoint)
    model, problem_def, train_T, prov = load_gfn(ck, device)
    ef = problem_def['energy_function']
    assert len(problem_def['space_groups']) == 1 and len(problem_def['z_primes']) == 1, (
        f"one (sg, Z') cell per bundle; checkpoint trained on {problem_def['space_groups']} x {problem_def['z_primes']}")
    sg_ind, zp = int(problem_def['space_groups'][0]), int(problem_def['z_primes'][0])
    prior_file = torch.load(cfg.dataset_path, map_location='cpu', weights_only=False)
    lj_coeff = float(prior_file.get('thermal_scaling_factor', 1.0))
    if ef != 'elj' and abs(lj_coeff - 1.0) > 1e-9:
        raise ValueError(f"prior carries thermal_scaling_factor={lj_coeff} on a {ef} problem")
    molecule = resolve_molecule(cfg, prior_file)
    del prior_file
    predictor = load_predictor(ef, device)
    # SCORE_BATCH / SAMPLE_BATCH override the yaml's analysis sizes (set for the 10k runs' memory margins); at batch
    # 64 under a 0.5 cap UMA scores ~140 draws/s against ~90 at 16 (MIPCAS, measured 2026-09-28)
    score_batch = int(os.environ.get('SCORE_BATCH', cfg.analysis_batch_size))
    max_batch = int(os.environ.get('SCORE_BATCH', getattr(cfg, 'max_analysis_batch_size', cfg.analysis_batch_size)))
    sample_batch = int(os.environ.get('SAMPLE_BATCH', cfg.sample_batch_size))
    std_orientation = pinned['std_orientation']
    model_side = {'step_ind': prov['step_ind'], 'stage': prov['stage'], 'energy_function': ef, 'sg': sg_ind, 'zp': zp,
                  'train_T': train_T, 'lj_coeff': lj_coeff, 'legacy_triclinic_walls': LEGACY_TRICLINIC_WALLS,
                  'molecule_identifier': str(getattr(molecule, 'identifier', '?')),
                  'pb_frozen_installed': prov.get('pb_frozen_installed')}
    if os.path.exists(meta_path):   # a resumed stage 1 must be drawing from the same model
        _check_same(json.load(open(meta_path)), model_side, 'stage 1 (checkpoint state)')
    else:
        _atomic_json(meta_path, {**pinned, **model_side, 'learned_log_z': float(model.flow_model().item()),
                                 'seed_formula': '1_000_003 * (shard + 1) + 7919 * seed'})
    for s in todo:
        t0 = time.time()
        n = min(shard_size, num - s * shard_size)
        torch.manual_seed(1_000_003 * (s + 1) + 7919 * seed)   # reproducible per shard, independent of resume order
        lat, pf, pb, raw = sample_latents(model, n, sample_batch, train_T, device)
        samples, kept = decode_and_score(lat, molecule, model.max_z_prime, sg_ind, zp, ef, lj_coeff, predictor,
                                         score_batch, device, std_orientation, max_batch,
                                         cfg.rdf_mode)
        b = collate_data_list(samples)
        cdf, empty = rdf_to_cdf(b.rdf)
        # fp16 storage check: the stored CDFs must reproduce the library distance on the ORIGINAL RDFs
        C_, B_ = b.rdf.shape[1:]
        dist = PairDistance(C_, B_, bool(empty.any()))
        gen = torch.Generator().manual_seed(s)
        ii, jj = torch.randint(0, len(kept), (32,), generator=gen), torch.randint(0, len(kept), (32,), generator=gen)
        bins = torch.linspace(0, 10, B_)
        err = max(abs(float(compute_rdf_distance(b.rdf[i], b.rdf[j], bins))
                      - float(dist(cdf[i:i + 1], empty[i:i + 1], cdf[j:j + 1], empty[j:j + 1])))
                  for i, j in zip(ii.tolist(), jj.tolist()))
        assert err < 5e-4, f"fp16 CDFs change the RDF distance by {err:.2e} on shard {s}"
        shard = {
            'latents': lat[kept], 'latents_raw': raw[kept], 'log_pf': pf[kept], 'log_pb': pb[kept],
            'energy': b[ef].detach().float().flatten(),
            'elj': b.elj.detach().float().flatten() if hasattr(b, 'elj') else None,
            'packing_coeff': b.packing_coeff.detach().float().flatten(),
            'reduction_en': b.reduction_en.detach().float().flatten(),
            'cell_volume': b.cell_volume.detach().float().flatten(),
            'cell_lengths': b.cell_lengths.detach().float(), 'cell_angles': b.cell_angles.detach().float(),
            'aunit_centroid': b.aunit_centroid.detach().float(),
            'aunit_orientation': b.aunit_orientation.detach().float(),
            'cdf': cdf, 'empty': empty, 'rdf_shape': tuple(b.rdf.shape[1:]),
            'n_drawn': n, 'n_kept': len(kept), 'fp16_distance_err': err,
        }
        path = os.path.join(shard_dir, f'shard_{s:04d}.pt')
        torch.save(shard, path + '.tmp')
        os.replace(path + '.tmp', path)   # a shard exists only once complete
        index[str(s)] = {'n_kept': len(kept), 'n_drawn': n, 'rdf_shape': list(b.rdf.shape[1:]),
                         'size': os.path.getsize(path)}
        _atomic_json(index_path, index)
        _log(f"stage 1: shard {s + 1}/{n_shards}: {len(kept)}/{n} kept in {time.time() - t0:.0f}s "
             f"(fp16 distance error {err:.1e})")
        del samples, b, shard
        torch.cuda.empty_cache()
    del model, predictor
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- stage 2
def stage_assemble(out):
    meta = json.load(open(os.path.join(out, 'draw_meta.json')))
    n_shards = int(np.ceil(meta['num'] / meta['shard_size']))
    index = json.load(open(os.path.join(out, 'shards', 'index.json')))
    assert set(index) == {str(s) for s in range(n_shards)}, (
        f"shard index holds {sorted(index, key=int)}, expected 0..{n_shards - 1}: stage 1 is incomplete")
    assert sum(v['n_drawn'] for v in index.values()) == meta['num']
    draws_path, cdf_path = os.path.join(out, 'draws.pt'), os.path.join(out, 'cdf.npy')
    if os.path.exists(draws_path) and os.path.exists(cdf_path):
        if torch.load(draws_path, weights_only=False).get('shard_index') == index:
            _log("stage 2: draws.pt and cdf.npy present and match the shard index")
            return
    # rebuilding would change what the stage-4 outputs were computed against
    assert not os.path.exists(os.path.join(out, 'stream_progress.json')), (
        "stage 2 would rebuild draws.pt/cdf.npy under existing stage-4 outputs; delete sums.npy, knn_*.npy, "
        "ref_dist.npy and stream_progress.json first")
    shapes = {tuple(v['rdf_shape']) for v in index.values()}
    assert len(shapes) == 1, f"shards disagree on the RDF shape: {shapes}"
    C, B = shapes.pop()
    n = sum(index[str(s)]['n_kept'] for s in range(n_shards))
    cdf = np.lib.format.open_memmap(cdf_path + '.tmp', mode='w+', dtype=np.float16, shape=(n, C * B))
    keys = ('latents', 'latents_raw', 'log_pf', 'log_pb', 'energy', 'elj', 'packing_coeff', 'reduction_en',
            'cell_volume', 'cell_lengths', 'cell_angles', 'aunit_centroid', 'aunit_orientation', 'empty')
    parts = {k: [] for k in keys}
    c = 0
    for s in range(n_shards):   # one shard in memory at a time
        sh = torch.load(os.path.join(out, 'shards', f'shard_{s:04d}.pt'), weights_only=False)
        assert sh['n_kept'] == index[str(s)]['n_kept'], f"shard {s} does not match the index"
        cdf[c:c + sh['n_kept']] = sh['cdf'].numpy()
        c += sh['n_kept']
        for kk in keys:
            parts[kk].append(sh[kk])
        del sh
    cdf.flush()
    del cdf
    os.replace(cdf_path + '.tmp', cdf_path)
    draws = {kk: (torch.cat(v) if v[0] is not None else None) for kk, v in parts.items()}
    e = draws['energy']
    kT = float(meta['kT'])
    thr = float(e.min() + 15 * kT)
    window = e < thr
    reduced = draws['reduction_en'].clamp(min=0) <= meta['max_reduction_en']
    low = torch.sort(e).values[:10]
    draws.update({'window': window, 'window_reduced': window & reduced, 'energy_threshold': thr, 'kT': kT,
                  # a robust alternative anchor, recorded for comparison only (the masks use the minimum, as before)
                  'energy_threshold_q001': float(torch.quantile(e, 0.001) + 15 * kT),
                  'rdf_shape': (C, B), 'n_drawn': meta['num'], 'shard_index': index})
    torch.save(draws, draws_path + '.tmp')
    os.replace(draws_path + '.tmp', draws_path)
    _log(f"stage 2: {n} draws ({meta['num']} drawn); window {int(window.sum())}, window & reduced "
         f"{int(draws['window_reduced'].sum())}; 10 lowest energies {[round(float(v), 2) for v in low]}")


# ----------------------------------------------------------------------------- stage 3
def stage_refs(cfg, out, device):
    meta = json.load(open(os.path.join(out, 'draw_meta.json')))
    assert cfg.rdf_mode == meta['rdf_mode'], (
        f"the yaml now says rdf_mode {cfg.rdf_mode!r}; the draws were made with {meta['rdf_mode']!r}")
    path = os.path.join(out, 'refs.pt')
    if os.path.exists(path):
        refs = torch.load(path, weights_only=False)
        assert refs.get('rdf_mode') == meta['rdf_mode'], "refs.pt was scored in a different rdf_mode"
        _log("stage 3: refs.pt present")
        return
    assert not os.path.exists(os.path.join(out, 'stream_progress.json')), (
        "stage 3 would regenerate refs.pt under existing stage-4 outputs; delete them first")
    ef = meta['energy_function']
    predictor = load_predictor(ef, device)
    max_batch = int(getattr(cfg, 'max_analysis_batch_size', cfg.analysis_batch_size))
    eb = score_experimental(cfg, ef, meta['lj_coeff'], predictor, device, meta['std_orientation'], meta['sg'],
                            meta['zp'], max_batch, meta['rdf_mode'])
    cdf, empty = rdf_to_cdf(eb.rdf.cpu())
    refs = {'labels': [str(i) for i in eb.identifier], 'cdf': cdf, 'empty': empty,
            'energy': eb[ef].detach().float().cpu().flatten(),
            'packing_coeff': eb.packing_coeff.detach().float().cpu().flatten(),
            'rdf': eb.rdf.cpu(), 'rdf_mode': meta['rdf_mode']}
    torch.save(refs, path + '.tmp')
    os.replace(path + '.tmp', path)
    _log(f"stage 3: {len(refs['labels'])} references: {refs['labels']}")
    del predictor
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- stage 4
def _resolve_sigma(sigma_from, meta, d_cut_probes):
    """d_cut for the kernel: 'probes', a number, or a new_analysis results file (validated against the bundle)."""
    rec = {'d_cut_probes': d_cut_probes}
    if sigma_from == 'probes':
        rec.update(source='probes (500 random window draws)', d_cut=d_cut_probes)
        return rec
    try:
        rec.update(source='pinned value', d_cut=float(sigma_from))
        return rec
    except ValueError:
        pass
    if not os.path.exists(sigma_from):
        raise FileNotFoundError(f"SIGMA_FROM {sigma_from} does not exist; pass SIGMA_FROM=probes or a d_cut value")
    r = torch.load(sigma_from, map_location='cpu', weights_only=False)
    step = (r.get('provenance') or {}).get('step_ind')
    assert r.get('rdf_mode') == meta['rdf_mode'], (
        f"{sigma_from} was measured in rdf_mode {r.get('rdf_mode')!r}, the bundle in {meta['rdf_mode']!r}")
    assert step == meta['step_ind'], (
        f"{sigma_from} comes from checkpoint step {step}, the bundle from {meta['step_ind']}; pass SIGMA_FROM=probes "
        f"or the d_cut you want pinned")
    rec.update(source=os.path.basename(sigma_from), d_cut=float(r['d_cuts'][0]), results_step=step)
    ratio = rec['d_cut'] / d_cut_probes
    if abs(ratio - 1) > 0.05:
        _log(f"stage 4: !! the 10k results' d_cut {rec['d_cut']:.4f} differs from this bundle's probe estimate "
             f"{d_cut_probes:.4f} by {100 * (ratio - 1):+.1f}% -- using the 10k value (the published kernel)")
    return rec


def stage_stream(out, device, k, sigma_from, row_block, col_chunk, stop_after=None):
    meta = json.load(open(os.path.join(out, 'draw_meta.json')))
    draws = torch.load(os.path.join(out, 'draws.pt'), weights_only=False)
    refs = torch.load(os.path.join(out, 'refs.pt'), weights_only=False)
    cdf_all = np.load(os.path.join(out, 'cdf.npy'), mmap_mode='r')
    N, D = cdf_all.shape
    R = len(refs['labels'])
    C, B = draws['rdf_shape']
    assert N == len(draws['energy']) and D == C * B, "cdf.npy and draws.pt disagree"
    empty = draws['empty'].to(device)
    any_empty = bool(draws['empty'].any() or refs['empty'].any())
    dist = PairDistance(C, B, any_empty)
    mask_bool = torch.stack([torch.ones(N, dtype=torch.bool), draws['window'], draws['window_reduced']], 1)
    masks = mask_bool.float().to(device)

    # the whole CDF matrix stays resident when it fits this process's cap; otherwise columns stream from the memmap
    if device.startswith('cuda'):
        cap = float(os.environ.get('GPU_MEM_FRACTION', 0.9)) * torch.cuda.get_device_properties(0).total_memory
        budget = min(cap - torch.cuda.memory_reserved(), torch.cuda.mem_get_info()[0])
    else:
        budget = 0
    resident = N * D * 2 < 0.5 * budget
    cols_gpu = torch.from_numpy(np.ascontiguousarray(cdf_all)).to(device) if resident else None
    _log(f"stage 4: N {N}, D {D} ({C} channels x {B} bins), {R} references; CDFs "
         f"{'resident on GPU' if resident else 'streamed from disk'}; empty channels: {any_empty}")

    def cols(c0, c1):
        return cols_gpu[c0:c1] if resident else torch.from_numpy(np.ascontiguousarray(cdf_all[c0:c1])).to(device)

    # self-check against the library's distance on random pairs
    g = torch.Generator().manual_seed(0)
    ii, jj = torch.randint(0, N, (64,), generator=g), torch.randint(0, N, (64,), generator=g)
    bins = torch.linspace(0, 10, B)

    def pdf(r):
        return torch.diff(torch.from_numpy(np.asarray(cdf_all[r])).float().view(C, B), dim=-1,
                          prepend=torch.zeros(C, 1))
    diffs = [abs(float(compute_rdf_distance(pdf(i), pdf(j), bins))
                 - float(dist(cols(i, i + 1), empty[i:i + 1], cols(j, j + 1), empty[j:j + 1])))
             for i, j in zip(ii.tolist(), jj.tolist())]
    _log(f"stage 4: fast distance vs compute_rdf_distance on 64 random pairs: max |diff| {max(diffs):.2e}")
    assert max(diffs) < 5e-4, "the streamed distance does not reproduce compute_rdf_distance"

    # d_cut from 500 random window draws against every window draw (self pairs dropped), always recorded
    wnd = torch.nonzero(draws['window']).flatten()
    probes = torch.sort(wnd[torch.randperm(len(wnd), generator=g)[:500]]).values
    Ap = torch.from_numpy(np.ascontiguousarray(cdf_all[probes.numpy()])).to(device)
    wmask = draws['window'].to(device)
    pieces = []
    for c0 in range(0, N, col_chunk):
        c1 = min(N, c0 + col_chunk)
        Dp = dist(Ap, empty[probes.to(device)], cols(c0, c1), empty[c0:c1])
        selfhit = probes.to(device)[:, None] == torch.arange(c0, c1, device=device)[None, :]
        pieces.append(Dp[:, wmask[c0:c1]][~selfhit[:, wmask[c0:c1]]].cpu().numpy())
    sigma_record = _resolve_sigma(sigma_from, meta, float(np.quantile(np.concatenate(pieces), 0.15)))
    del Ap, pieces
    sigma0 = sigma_record['d_cut'] / 3
    sigmas = torch.tensor([m * sigma0 for m in SIGMA_MULTS], device=device, dtype=torch.float32)
    # kernel-independent stores, both scaled by the bundle's OWN probe d_cut so a re-pass at another width
    # reproduces them: the histogram range and the neighbour-list cutoff
    hist_max, r_nb = 4 * sigma_record['d_cut_probes'], sigma_record['d_cut_probes']
    sigma_record.update({'sigma0': sigma0, 'sigma_mults': SIGMA_MULTS, 'hist_bins': HIST_BINS,
                         'hist_max': hist_max, 'nbr_cutoff': r_nb})
    mask_dev = mask_bool.to(device)
    nb_dir = os.path.join(out, 'nbrs')
    os.makedirs(nb_dir, exist_ok=True)
    _log(f"stage 4: d_cut {sigma_record['d_cut']:.4f} ({sigma_record['source']}; from probes "
         f"{sigma_record['d_cut_probes']:.4f}); sigma0 {sigma0:.4f}")

    # everything a stored number depends on; a resume must match it exactly
    fingerprint = {'N': N, 'R': R, 'D': D, 'k': k, 'row_block': row_block,
                   'sigmas': [round(float(v), 9) for v in sigmas.cpu()], 'ref_labels': refs['labels'],
                   'hist_bins': HIST_BINS, 'hist_max': round(hist_max, 9), 'nbr_cutoff': round(r_nb, 9),
                   'energy_threshold': draws['energy_threshold'], 'n_window': int(draws['window'].sum()),
                   'n_window_reduced': int(draws['window_reduced'].sum()),
                   'mask_sha1': hashlib.sha1(mask_bool.numpy().tobytes()).hexdigest(),
                   'cdf_rows_sha1': hashlib.sha1(np.ascontiguousarray(cdf_all[:: max(1, N // 64)]).tobytes()).hexdigest()}
    prog_path = os.path.join(out, 'stream_progress.json')
    outputs = {'sums.npy': (np.float64, (N + R, len(SIGMA_MULTS), len(SETS)), 0.0),
               'knn_idx.npy': (np.int32, (N, k), -1), 'knn_dist.npy': (np.float32, (N, k), np.inf),
               'ref_dist.npy': (np.float32, (R, N), np.nan),
               'hist.npy': (np.uint16, (N + R, len(SETS), HIST_BINS), 0)}
    if os.path.exists(prog_path):
        prog = json.load(open(prog_path))
        _check_same(prog['fingerprint'], fingerprint, 'stage 4 resume')
        done = set(prog['done'])
    else:   # outputs without a progress file are leftovers of a pass that never recorded a block
        for name in outputs:
            if os.path.exists(os.path.join(out, name)):
                os.remove(os.path.join(out, name))
        done = set()
        _atomic_json(prog_path, {'done': [], 'fingerprint': fingerprint})
    arr = {}
    for name, (dtype, shape, fill) in outputs.items():
        p = os.path.join(out, name)
        if os.path.exists(p):
            a = np.load(p, mmap_mode='r+')
            assert a.shape == shape and a.dtype == dtype, f"{name} is {a.shape} {a.dtype}, expected {shape} {dtype}"
        else:
            a = np.lib.format.open_memmap(p, mode='w+', dtype=dtype, shape=shape)
            a[:] = fill
        arr[name] = a
    sums, knn_idx, knn_dist, ref_dist = (arr['sums.npy'], arr['knn_idx.npy'], arr['knn_dist.npy'],
                                         arr['ref_dist.npy'])
    hist = arr['hist.npy']

    ref_cdf, ref_empty = refs['cdf'].to(device), refs['empty'].to(device)
    blocks = [('draws', r0, min(N, r0 + row_block)) for r0 in range(0, N, row_block)] + [('refs', N, N + R)]
    keys = [f"{kind}:{r0}-{r1}" for kind, r0, r1 in blocks]
    n_blocks_run, t_start = 0, time.time()
    for (kind, r0, r1), key in zip(blocks, keys):
        if key in done:
            continue
        is_draw = kind == 'draws'
        A, Ae = (cols(r0, r1), empty[r0:r1]) if is_draw else (ref_cdf, ref_empty)
        nr = r1 - r0
        acc = torch.zeros(nr, len(SIGMA_MULTS), len(SETS), device=device, dtype=torch.float64)
        best_d = torch.full((nr, k), float('inf'), device=device)
        best_i = torch.full((nr, k), -1, device=device, dtype=torch.long)
        ref_rows = []
        hist_acc = torch.zeros(nr, len(SETS), HIST_BINS, device=device, dtype=torch.int64)
        row_base = torch.arange(nr, device=device)[:, None] * HIST_BINS
        nb_rows, nb_cols, nb_d = [], [], []
        for c0 in range(0, N, col_chunk):
            c1 = min(N, c0 + col_chunk)
            Dm = dist(A, Ae, cols(c0, c1), empty[c0:c1])
            if is_draw:   # leave-one-out: a draw is not its own neighbour
                lo, hi = max(r0, c0), min(r1, c1)
                if lo < hi:
                    idx = torch.arange(lo, hi, device=device)
                    Dm[idx - r0, idx - c0] = float('inf')
            else:
                ref_rows.append(Dm.cpu())
            in_hist = Dm < hist_max   # the diagonal is inf for draws, so a draw never counts itself
            flat = row_base + (Dm.clamp(max=hist_max) * (HIST_BINS / hist_max)).long().clamp(max=HIST_BINS - 1)
            for ci in range(len(SETS)):
                sel = in_hist & mask_dev[c0:c1, ci][None, :]
                hist_acc[:, ci, :] += torch.bincount(flat[sel], minlength=nr * HIST_BINS).view(nr, HIST_BINS)
            if is_draw:
                nz = (Dm < r_nb).nonzero()
                nb_rows.append(nz[:, 0].int().cpu())
                nb_cols.append((nz[:, 1] + c0).int().cpu())
                nb_d.append(Dm[nz[:, 0], nz[:, 1]].half().cpu())
            D2 = Dm * Dm
            for si in range(len(SIGMA_MULTS)):
                acc[:, si, :] += (torch.exp(-D2 / (2 * sigmas[si] ** 2)) @ masks[c0:c1]).double()
            if is_draw:
                cand_d = torch.cat([best_d, Dm], 1)
                cand_i = torch.cat([best_i, torch.arange(c0, c1, device=device).expand(nr, -1)], 1)
                best_d, pos = torch.topk(cand_d, k, dim=1, largest=False)
                best_i = torch.gather(cand_i, 1, pos)
        sums[r0:r1] = acc.cpu().numpy()
        assert int(hist_acc.max()) < 65536, "a histogram bin overflows uint16"
        hist[r0:r1] = hist_acc.cpu().numpy().astype(np.uint16)
        if is_draw:
            rows_all = torch.cat(nb_rows)
            order_nb = torch.argsort(rows_all, stable=True)   # chunks are column-ordered: row-major, columns ascending
            nb_path = os.path.join(nb_dir, f'nb_{r0:07d}_{r1:07d}.npz')
            np.savez(nb_path + '.tmp.npz', counts=torch.bincount(rows_all.long(), minlength=nr).int().numpy(),
                     idx=torch.cat(nb_cols)[order_nb].numpy(), dist=torch.cat(nb_d)[order_nb].numpy())
            os.replace(nb_path + '.tmp.npz', nb_path)
            del nb_rows, nb_cols, nb_d, rows_all
            order = torch.argsort(best_d, 1)
            knn_dist[r0:r1] = torch.gather(best_d, 1, order).cpu().numpy()
            knn_idx[r0:r1] = torch.gather(best_i, 1, order).cpu().numpy().astype(np.int32)
        else:
            rows = torch.cat(ref_rows, 1).numpy()
            assert rows.shape == ref_dist.shape
            ref_dist[:] = rows
        for a in arr.values():
            a.flush()
        done.add(key)   # recorded only after the block's rows are flushed
        _atomic_json(prog_path, {'done': sorted(done), 'fingerprint': fingerprint})
        n_blocks_run += 1
        el = time.time() - t_start
        _log(f"stage 4: block {len(done)}/{len(blocks)} ({key}) in {el / n_blocks_run:.1f}s/block; "
             f"eta {(len(blocks) - len(done)) * el / n_blocks_run / 60:.1f} min")
        if stop_after and n_blocks_run >= stop_after:
            _log(f"stage 4: stopping after {n_blocks_run} blocks (STOP_AFTER)")
            return None
    # coverage: nothing may still hold its fill value
    assert done == set(keys), f"blocks missing: {sorted(set(keys) - done)}"
    kk = min(k, N - 1)
    assert (np.asarray(knn_idx[:, :kk]) >= 0).all() and np.isfinite(np.asarray(knn_dist[:, :kk])).all(), \
        "some draws have unfilled neighbour slots"
    assert np.isfinite(np.asarray(ref_dist)).all() and np.isfinite(np.asarray(sums)).all(), "unfilled outputs"
    _atomic_json(os.path.join(out, 'sigma.json'), sigma_record)
    return sigma_record


# ----------------------------------------------------------------------------- summary / loader
def load_bundle(out):
    """Everything the analysis needs, without touching the big arrays until asked (memmaps)."""
    return {'draws': torch.load(os.path.join(out, 'draws.pt'), weights_only=False),
            'refs': torch.load(os.path.join(out, 'refs.pt'), weights_only=False),
            'sums': np.load(os.path.join(out, 'sums.npy'), mmap_mode='r'),
            'knn_idx': np.load(os.path.join(out, 'knn_idx.npy'), mmap_mode='r'),
            'knn_dist': np.load(os.path.join(out, 'knn_dist.npy'), mmap_mode='r'),
            'ref_dist': np.load(os.path.join(out, 'ref_dist.npy'), mmap_mode='r'),
            'cdf': np.load(os.path.join(out, 'cdf.npy'), mmap_mode='r'),
            'sigma': json.load(open(os.path.join(out, 'sigma.json'))),
            'meta': json.load(open(os.path.join(out, 'draw_meta.json'))),
            'sigma_mults': SIGMA_MULTS, 'sets': SETS}


def kernel_sums_from_hist(out, d_cut, set_name='window_reduced'):
    """Kernel sums of every draw (leave-one-out) and reference at sigma = d_cut / 3, from the stored distance
    histograms: [N + R] float64. Bins are sigma0-independent; each contributes its count at its centre distance,
    so the result is exact to the bin width (4 x probe d_cut / HIST_BINS, ~1% of a typical sigma)."""
    rec = json.load(open(os.path.join(out, 'sigma.json')))
    sigma = d_cut / 3
    assert 5 * sigma <= rec['hist_max'], f"sigma {sigma:.4f} needs distances beyond the stored range {rec['hist_max']:.4f}"
    centres = (np.arange(rec['hist_bins']) + 0.5) * rec['hist_max'] / rec['hist_bins']
    w = np.exp(-centres ** 2 / (2 * sigma ** 2))
    h = np.load(os.path.join(out, 'hist.npy'), mmap_mode='r')
    ci = SETS.index(set_name)
    return np.concatenate([np.asarray(h[r0:r0 + 8192, ci, :], dtype=np.float64) @ w for r0 in range(0, h.shape[0], 8192)])


def load_neighbours(out, r0, r1):
    """Within-cutoff neighbours of the draws in one stored row block: (counts, idx, dist); row i's neighbours are
    idx[counts[:i].sum() : counts[:i + 1].sum()], columns ascending."""
    z = np.load(os.path.join(out, 'nbrs', f'nb_{r0:07d}_{r1:07d}.npz'))
    return z['counts'], z['idx'], z['dist']


def point_sums(out, rdf=None, set_name='window_reduced', device='cuda', col_chunk=8192, cdf=None, empty=None):
    """Kernel sums of arbitrary structures against a bundle's draws, at the bundle's own sigma0.

    rdf: [P, channels, bins], scored in the bundle's rdf_mode. Returns the points' sums over the draws of
    `set_name` (a draw at distance ~0 is the point itself and is left out, as in the draws' leave-one-out sums),
    the draws' own leave-one-out sums over the same set, and the nearest-draw distance of each point."""
    b = load_bundle(out)
    draws, cdf_all = b['draws'], b['cdf']
    N = len(draws['energy'])
    C, B = draws['rdf_shape']
    if rdf is not None:
        assert tuple(rdf.shape[-2:]) == (C, B), f"points are {tuple(rdf.shape[-2:])}, the bundle is {(C, B)}"
    ci, si = SETS.index(set_name), SIGMA_MULTS.index(1.0)
    member = torch.ones(N, dtype=torch.bool) if set_name == 'all' else draws[set_name]
    p_cdf, p_empty = rdf_to_cdf(rdf.cpu()) if rdf is not None else (cdf, empty)   # or stored CDFs + emptiness masks
    p_cdf, p_empty = p_cdf.to(device), p_empty.to(device)
    dist = PairDistance(C, B, bool(draws['empty'].any() or p_empty.any()))
    empty, mask = draws['empty'].to(device), member.to(device).double()
    sigma0 = b['sigma']['sigma0']
    acc = torch.zeros(len(p_cdf), device=device, dtype=torch.float64)
    nearest = torch.full((len(p_cdf),), float('inf'), device=device)
    for c0 in range(0, N, col_chunk):
        c1 = min(N, c0 + col_chunk)
        Dm = dist(p_cdf, p_empty, torch.from_numpy(np.ascontiguousarray(cdf_all[c0:c1])).to(device), empty[c0:c1])
        w = torch.exp(-(Dm * Dm) / (2 * sigma0 ** 2)).double()
        w[Dm < 1e-7] = 0.0
        acc += w @ mask[c0:c1]
        nearest = torch.minimum(nearest, torch.where(mask[c0:c1].bool()[None, :], Dm, torch.inf).amin(1))
    return {'point_sums': acc.cpu(), 'draw_sums': torch.from_numpy(np.asarray(b['sums'][:N, si, ci]))[member],
            'nearest': nearest.cpu(), 'd_cut': b['sigma']['d_cut'], 'sigma0': sigma0, 'n_draws': int(member.sum()),
            'rdf_mode': b['meta']['rdf_mode'], 'step_ind': b['meta']['step_ind']}


def summarise(out):
    b = load_bundle(out)
    N = len(b['draws']['energy'])
    s = np.asarray(b['sums'])
    si = SIGMA_MULTS.index(1.0)
    print(f"\n== {os.path.basename(out)}: {N} draws; sigma0 {b['sigma']['sigma0']:.4f} "
          f"(d_cut {b['sigma']['d_cut']:.4f}, {b['sigma']['source']})")
    for ci, name in enumerate(SETS):
        member = (np.ones(N, bool) if name == 'all' else b['draws'][name].numpy())
        draw_s = s[:N, si, ci][member]
        print(f"   set {name:15s}: {member.sum():6d} draws; draws under one draw's weight {100 * np.mean(draw_s < 1):.2f}%; "
              f"peak sum {draw_s.max():.1f}")
        for r, lab in enumerate(b['refs']['labels']):
            v = s[N + r, si, ci]
            pct = 100 * (np.mean(draw_s < v) + 0.5 * np.mean(draw_s == v))
            print(f"      {lab:10s} sum {v:9.2f} draws   percentile {pct:6.2f}   -kT ln P/Pmax "
                  f"{-b['draws']['kT'] * np.log(max(v, 1e-300) / draw_s.max()):6.2f} kJ/mol   "
                  f"nearest draw {np.nanmin(b['ref_dist'][r]):.4f}")


if __name__ == '__main__':
    raw_cfg = load_yaml(os.path.join(HERE, 'new_analysis.yaml'))
    base, runs = raw_cfg['base'], raw_cfg['runs']
    run_name = os.environ['ONLY_RUN']
    run = [r for r in runs if r['run_name'] == run_name]
    assert len(run) == 1, f"run {run_name} not found once in new_analysis.yaml"
    cfg = dict2namespace({**base, **run[0]})
    num = int(os.environ.get('NUM', 100_000))
    shard_size = int(os.environ.get('SHARD', 5000))
    seed = int(os.environ.get('SEED', 0))
    k = int(os.environ.get('K', 512))
    device = cfg.device
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('GPU_MEM_FRACTION', 0.9)), device=0)
    out = os.environ.get('OUT') or os.path.join(cfg.results_dir, 'dens', f"{run_name}_{num}")
    os.makedirs(out, exist_ok=True)
    # SIGMA_FROM: a new_analysis results file (default: this run's 10k results), 'probes', or a d_cut value
    sigma_from = os.environ.get('SIGMA_FROM', os.path.join(cfg.results_dir, f"{run_name}.pt"))
    t0 = time.time()
    stage_draws(cfg, out, num, shard_size, seed, device)
    stage_assemble(out)
    stage_refs(cfg, out, device)
    rec = stage_stream(out, device, k, sigma_from, int(os.environ.get('ROW_BLOCK', 1024)),
                       int(os.environ.get('COL_CHUNK', 8192)),
                       stop_after=int(os.environ['STOP_AFTER']) if os.environ.get('STOP_AFTER') else None)
    if rec is not None:
        summarise(out)
    _log(f"total {time.time() - t0:.0f}s -> {out}")
