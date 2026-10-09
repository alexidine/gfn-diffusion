"""Landscape figures over the full dens_engine bundles (e.g. 100k draws) instead of a 10k analysis set.

Nothing here needs the N x N distance matrix. Three pieces:

  embed_bundles   UMAP of every draw of one bundle, or of two bundles jointly. A landmark subset is laid out exactly
                  as the 10k figures are (dense precomputed distances, same UMAP settings); every draw then starts
                  at its nearest landmark and the layout is refined on the stored k-nearest-neighbour graph.
  cross_knn       the neighbours each bundle's draws have in the OTHER bundle (one streamed distance pass), which
                  the joint embedding needs and a single bundle does not hold.
  bundle_landscape   everything figures.combo_fig takes, over a bundle's window & reduced draws: the 10k analysis's
                  mean-shift basins extended to every draw by the same anchor rule, densities and percentiles from
                  the bundle's kernel sums.
"""
import json
import os
import time

import numpy as np
import plotly.express as px
import torch

from energy_sampling.eval.paper1_results.dens_engine import (
    PairDistance, load_bundle, rdf_to_cdf, point_sums, kernel_sums_from_hist, SETS, SIGMA_MULTS, _log)


def _dist_to_draws(bundle, cdf, empty, device='cuda', col_chunk=8192, cols=None):
    """[P, N] (or [P, len(cols)]) distances from structures given as CDFs to a bundle's draws."""
    draws, cdf_all = bundle['draws'], bundle['cdf']
    C, B = draws['rdf_shape']
    dist = PairDistance(C, B, bool(draws['empty'].any() or empty.any()))
    cdf, empty = cdf.to(device), empty.to(device)
    cols = np.arange(len(draws['energy'])) if cols is None else np.asarray(cols)
    out = []
    for c0 in range(0, len(cols), col_chunk):
        sel = cols[c0:c0 + col_chunk]
        b = torch.from_numpy(np.ascontiguousarray(cdf_all[sel])).to(device)
        out.append(dist(cdf, empty, b, draws['empty'][sel].to(device)).cpu())
    return torch.cat(out, 1)


def cross_knn(out_a, out_b, k=300, device='cuda', row_block=1024, col_chunk=8192):
    """For every draw of bundle A its k nearest draws of bundle B, and the reverse, from one streamed pass.

    Written to <out_a>/cross_<basename(out_b)>_{idx,dist}.npy and <out_b>/cross_<basename(out_a)>_{idx,dist}.npy
    (skipped when all four exist)."""
    paths = [os.path.join(o, f'cross_{os.path.basename(p)}_{w}.npy')
             for o, p in ((out_a, out_b), (out_b, out_a)) for w in ('idx', 'dist')]
    if all(os.path.exists(p) for p in paths):
        return
    a, b = load_bundle(out_a), load_bundle(out_b)
    Na, Nb = len(a['draws']['energy']), len(b['draws']['energy'])
    C, B = a['draws']['rdf_shape']
    assert tuple(b['draws']['rdf_shape']) == (C, B) and a['meta']['rdf_mode'] == b['meta']['rdf_mode']
    dist = PairDistance(C, B, bool(a['draws']['empty'].any() or b['draws']['empty'].any()))
    ea, eb = a['draws']['empty'].to(device), b['draws']['empty'].to(device)
    row_d = np.empty((Na, k), np.float32)
    row_i = np.empty((Na, k), np.int32)
    col_d = torch.full((Nb, k), float('inf'), device=device)
    col_i = torch.full((Nb, k), -1, device=device, dtype=torch.long)
    t0 = time.time()
    for r0 in range(0, Na, row_block):
        r1 = min(Na, r0 + row_block)
        A = torch.from_numpy(np.ascontiguousarray(a['cdf'][r0:r1])).to(device)
        best_d = torch.full((r1 - r0, k), float('inf'), device=device)
        best_i = torch.full((r1 - r0, k), -1, device=device, dtype=torch.long)
        kk = min(k, r1 - r0)
        for c0 in range(0, Nb, col_chunk):
            c1 = min(Nb, c0 + col_chunk)
            Dm = dist(A, ea[r0:r1], torch.from_numpy(np.ascontiguousarray(b['cdf'][c0:c1])).to(device), eb[c0:c1])
            cand_d = torch.cat([best_d, Dm], 1)
            cand_i = torch.cat([best_i, torch.arange(c0, c1, device=device).expand(r1 - r0, -1)], 1)
            best_d, pos = torch.topk(cand_d, k, dim=1, largest=False)
            best_i = torch.gather(cand_i, 1, pos)
            # the same block, read column-wise: each B draw's nearest A draws within this row block
            bd, bp = torch.topk(Dm.T, kk, dim=1, largest=False)
            cd = torch.cat([col_d[c0:c1], bd], 1)
            cidx = torch.cat([col_i[c0:c1], bp + r0], 1)
            col_d[c0:c1], pos = torch.topk(cd, k, dim=1, largest=False)
            col_i[c0:c1] = torch.gather(cidx, 1, pos)
        order = torch.argsort(best_d, 1)
        row_d[r0:r1] = torch.gather(best_d, 1, order).cpu().numpy()
        row_i[r0:r1] = torch.gather(best_i, 1, order).cpu().numpy()
        if (r0 // row_block) % 10 == 0:
            el = time.time() - t0
            _log(f"cross kNN: rows {r1}/{Na}; eta {el / r1 * (Na - r1) / 60:.1f} min")
    order = torch.argsort(col_d, 1)
    col_dn, col_in = torch.gather(col_d, 1, order).cpu().numpy(), torch.gather(col_i, 1, order).cpu().numpy()
    assert (row_i >= 0).all() and (col_in >= 0).all()
    for p, arr in zip(paths, (row_i, row_d, col_in.astype(np.int32), col_dn)):
        np.save(p, arr)


def embed_bundles(outs, n_neighbors=300, min_dist=0.75, n_landmarks=10000, seed=0, device='cuda',
                  landmark_kwargs=None, n_epochs=200):
    """UMAP coordinates for every draw of one bundle or of two bundles jointly, plus the first bundle's references.

    Returns {'draws': [array [N_i, 2] per bundle], 'refs': [R, 2], 'landmarks': per-bundle draw indices}. Cached in
    the first bundle's directory (embedding_<tag>.npz), keyed by the settings."""
    from umap import UMAP
    tag = '+'.join(os.path.basename(o) for o in outs[1:]) or 'self'
    cache = os.path.join(outs[0], f'embedding_{tag}_nn{n_neighbors}_md{min_dist}_L{n_landmarks}_s{seed}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return {'draws': [z[f'draws{i}'] for i in range(len(outs))], 'refs': z['refs'],
                'landmarks': [z[f'landmarks{i}'] for i in range(len(outs))]}
    bundles = [load_bundle(o) for o in outs]
    Ns = [len(b['draws']['energy']) for b in bundles]
    offs = np.concatenate([[0], np.cumsum(Ns)])
    refs = bundles[0]['refs']
    R = len(refs['labels'])
    rng = np.random.default_rng(seed)

    # landmarks: window & reduced draws of each bundle, laid out as the 10k figures are
    per = n_landmarks // len(outs)
    lm = [np.sort(rng.choice(np.nonzero(b['draws']['window_reduced'].numpy())[0], per, replace=False))
          for b in bundles]
    lm_cdf = torch.cat([torch.from_numpy(np.ascontiguousarray(b['cdf'][i])) for b, i in zip(bundles, lm)]
                       + [refs['cdf']])
    lm_empty = torch.cat([b['draws']['empty'][i] for b, i in zip(bundles, lm)] + [refs['empty']])
    L = len(lm_cdf)
    dmat = torch.cat([_dist_to_draws(b, lm_cdf, lm_empty, device, cols=i) for b, i in zip(bundles, lm)]
                     + [torch.zeros(L, R)], 1)
    C, B = bundles[0]['draws']['rdf_shape']
    pd = PairDistance(C, B, bool(lm_empty.any()))
    dmat[:, L - R:] = pd(lm_cdf.to(device), lm_empty.to(device), refs['cdf'].to(device),
                         refs['empty'].to(device)).cpu()
    dmat = 0.5 * (dmat + dmat.T)
    dmat.fill_diagonal_(0.0)
    _log(f"embedding: {L} landmarks ({per} per bundle + {R} references); dense layout")
    # no random_state: it forces UMAP single-threaded. The layout is fixed by the cache instead.
    kw = dict(n_neighbors=n_neighbors, min_dist=min_dist, init='pca')
    kw.update(landmark_kwargs or {})
    lm_xy = UMAP(n_components=2, metric='precomputed', low_memory=True, n_jobs=-1, **kw).fit_transform(
        dmat.numpy().astype(np.float32))
    del dmat

    # every draw: its k nearest draws (own bundle, merged with the other bundle's when joint), self first
    k = n_neighbors - 1
    idx_all, dist_all, init = [], [], []
    for bi, (o, b) in enumerate(zip(outs, bundles)):
        ki, kd = np.asarray(b['knn_idx'][:, :k]).astype(np.int64) + offs[bi], np.asarray(b['knn_dist'][:, :k])
        if len(outs) == 2:
            other = 1 - bi
            ci = np.load(os.path.join(o, f'cross_{os.path.basename(outs[other])}_idx.npy'))[:, :k].astype(np.int64)
            cd = np.load(os.path.join(o, f'cross_{os.path.basename(outs[other])}_dist.npy'))[:, :k]
            ki, kd = np.concatenate([ki, ci + offs[other]], 1), np.concatenate([kd, cd], 1)
            order = np.argsort(kd, 1)[:, :k]
            ki, kd = np.take_along_axis(ki, order, 1), np.take_along_axis(kd, order, 1)
        idx_all.append(ki)
        dist_all.append(kd)
        # start every draw at its nearest landmark (any bundle)
        near = torch.cat([torch.cat([
            _dist_to_draws(lb, torch.from_numpy(np.ascontiguousarray(b['cdf'][r0:r0 + 4096])),
                           b['draws']['empty'][r0:r0 + 4096], device, cols=li)
            for lb, li in zip(bundles, lm)], 1).argmin(1) for r0 in range(0, Ns[bi], 4096)]).numpy()
        init.append(lm_xy[near])
    # references: nearest draws across the bundles
    rd = torch.cat([_dist_to_draws(b, refs['cdf'], refs['empty'], device) for b in bundles], 1)
    r_d, r_i = torch.topk(rd, k, dim=1, largest=False)
    n_tot = int(offs[-1]) + R
    knn_i = np.concatenate(idx_all + [r_i.numpy()], 0)
    knn_d = np.concatenate(dist_all + [r_d.numpy()], 0)
    knn_i = np.concatenate([np.arange(n_tot)[:, None], knn_i], 1)
    knn_d = np.concatenate([np.zeros((n_tot, 1), np.float32), knn_d], 1).astype(np.float32)
    xy0 = np.concatenate(init + [lm_xy[L - R:]], 0).astype(np.float32)
    xy0 += rng.normal(scale=1e-3 * np.ptp(lm_xy, axis=0).max(), size=xy0.shape).astype(np.float32)
    _log(f"embedding: refining {n_tot} points on the {n_neighbors}-neighbour graph")
    t0 = time.time()
    xy = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, init=xy0, metric='euclidean',
              precomputed_knn=(knn_i, knn_d), n_epochs=n_epochs, n_jobs=-1,
              **{kk: v for kk, v in (landmark_kwargs or {}).items() if kk in ('repulsion_strength',)}
              ).fit_transform(xy0)
    _log(f"embedding: done in {time.time() - t0:.0f}s")
    res = {'draws': [xy[offs[i]:offs[i + 1]] for i in range(len(outs))], 'refs': xy[int(offs[-1]):],
           'landmarks': lm}
    np.savez(cache, refs=res['refs'], **{f'draws{i}': d for i, d in enumerate(res['draws'])},
             **{f'landmarks{i}': l for i, l in enumerate(lm)})
    return res


def mean_shift_pointers(out, cutoffs, device='cuda', row_block=1024, col_chunk=8192, dens=None, tag=''):
    """utils.mean_shift_density's pointer table over a bundle's window & reduced draws, for every cutoff at once.

    A draw points at the DENSEST draw closer than the cutoff (itself included). With the draws ordered by density,
    that is the first one along the order whose distance is under the cutoff, so one streamed pass over the pairs
    (only columns at least as dense as the row are needed) answers every cutoff: the running minimum of distance
    along the order is non-increasing, and each cutoff's answer is where it first drops below the cutoff.

    Returns (rows, perm, ptr): rows = bundle indices of the set, perm = density order (descending) within the set,
    ptr [n_cutoffs, M] in ORDER positions. Cached in the bundle directory."""
    b = load_bundle(out)
    draws = b['draws']
    N = len(draws['energy'])
    si, ci = SIGMA_MULTS.index(1.0), SETS.index('window_reduced')
    rows = torch.nonzero(draws['window_reduced']).flatten().numpy()
    dens = np.asarray(b['sums'][:N, si, ci])[rows] if dens is None else np.asarray(dens)   # dens: over `rows`
    perm = np.argsort(-dens, kind='stable')
    cutoffs = np.asarray(cutoffs, dtype=np.float32)
    cache = os.path.join(out, f'meanshift_ptr{tag}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z['cutoffs']) == len(cutoffs) and np.allclose(z['cutoffs'], cutoffs) and len(z['perm']) == len(perm):
            return rows, z['perm'], z['ptr']
    M, K = len(rows), len(cutoffs)
    C, B = draws['rdf_shape']
    dist = PairDistance(C, B, bool(draws['empty'].any()))
    cdf = torch.from_numpy(np.asarray(b['cdf'])[rows[perm]]).to(device)   # density order
    empty = draws['empty'][rows[perm]].to(device)
    cut = torch.from_numpy(cutoffs).to(device)
    ptr = np.empty((K, M), np.int32)
    t0 = time.time()
    for r0 in range(0, M, row_block):
        r1 = min(M, r0 + row_block)
        nr = r1 - r0
        run_min = torch.full((nr, 1), float('inf'), device=device)
        found = torch.full((nr, K), -1, device=device, dtype=torch.long)
        for c0 in range(0, r1, col_chunk):   # columns denser than (or equal to) the block's rows
            c1 = min(r1, c0 + col_chunk)
            Dm = dist(cdf[r0:r1], empty[r0:r1], cdf[c0:c1], empty[c0:c1])
            lo, hi = max(r0, c0), min(r1, c1)
            if lo < hi:   # a draw is at distance exactly 0 from itself
                idx = torch.arange(lo, hi, device=device)
                Dm[idx - r0, idx - c0] = 0.0
            cm = torch.cummin(torch.cat([run_min, Dm], 1), dim=1).values[:, 1:]
            first = torch.searchsorted((-cm).contiguous(), (-cut).expand(nr, K).contiguous(), right=True)
            hit = (first < (c1 - c0)) & (found < 0)
            found = torch.where(hit, first + c0, found)
            run_min = cm[:, -1:]
        assert (found >= 0).all(), "a draw found no neighbour (itself included)"
        ptr[:, r0:r1] = found.T.cpu().numpy()
        if (r0 // row_block) % 10 == 0:
            el = time.time() - t0
            _log(f"mean-shift pointers: rows {r1}/{M}; eta {el * ((M / r1) ** 2 - 1) / 60:.1f} min")
    np.savez(cache, cutoffs=cutoffs, perm=perm, ptr=ptr)
    return rows, perm, ptr


def mean_shift_anchors(out, d_cut, max_n_clusters, device='cuda', dens=None, tag=''):
    """The mean-shift scan of utils.clustering over a bundle's window & reduced draws: the smallest cutoff in
    linspace(0.33, 2, 200) x d_cut that leaves at most max_n_clusters modes. Returns (rows, anchors, cutoff factor),
    anchors as positions within rows."""
    # utils.clustering scans to 2 x d_cut; the same step carried to 3 x, because a COMPACK-anchored d_cut can be
    # small against the distance between a molecule's packing families
    ccs = np.linspace(0.33, 2, 200)
    ccs = np.concatenate([ccs, ccs[-1] + (ccs[1] - ccs[0]) * np.arange(1, 120)])
    rows, perm, ptr = mean_shift_pointers(out, ccs * d_cut, device, dens=dens, tag=tag)
    for cc, p in zip(ccs, ptr):
        state = np.arange(len(p))
        for _ in range(100):
            new = p[state]
            if np.array_equal(new, state):
                break
            state = new
        modes = np.unique(state)
        if len(modes) <= max_n_clusters:
            break
    _log(f"mean shift over {len(rows)} draws: {len(modes)} modes at cutoff {cc:.3f} x d_cut")
    if len(modes) > 64:   # the scan ran out: keep the 64 modes that the most draws flow to
        ids, counts = np.unique(state, return_counts=True)
        modes = np.sort(ids[np.argsort(-counts)[:64]])
        _log(f"   scan exhausted; keeping the 64 largest modes ({int(np.sort(counts)[-64:].sum())} of {len(state)} draws)")
    return rows, perm[modes], float(cc)


class _BasinMinima:
    """The two things figures.fig_grid reads off the basin-minimum batch."""

    def __init__(self, energy_function, energy, packing_coeff):
        self.packing_coeff, self._e, self._ef = packing_coeff, energy, energy_function

    def __getitem__(self, key):
        assert key == self._ef
        return self._e


def _rank(raw_draws, raw_pts):
    """Percentile of each point's kernel sum among the draws' leave-one-out sums (mid-rank), the % of draws under one
    draw's weight, and which points are under it (same reading as utils.combo_fig_analysis)."""
    raw_draws, raw_pts = torch.as_tensor(raw_draws).double(), torch.as_tensor(raw_pts).double()
    srt = torch.sort(raw_draws).values
    lt = torch.searchsorted(srt, raw_pts, right=False).double() / len(srt)
    le = torch.searchsorted(srt, raw_pts, right=True).double() / len(srt)
    return (100 * 0.5 * (lt + le)).numpy(), float(100 * (raw_draws < 1).double().mean()), (raw_pts < 1).numpy()


def bundle_landscape(uma_bundle, elj_bundle, uma_results, uma_thermos, ebatch, energy_function, max_n_clusters,
                     min_basin_size, device='cuda', basins_from='bundle', d_cut=None, max_basins=None,
                     scale_label=None, extra_scales=None):
    """figures.combo_fig's inputs over the bundle's window & reduced draws.

    Basins are the 10k analysis's: its mean-shift anchors are reused and every bundle draw is assigned by the same
    rule (kernel weight to each anchor, normalised over anchors, >= 0.8 or unassigned; anchors of basins the 10k
    analysis dropped as too small leave their draws unassigned). Basins keep the 10k figure's numbering; a basin's probability
    maximum is its mode (the densest member within one kernel width of the anchor).
    Densities and the UMA percentiles are the bundle's leave-one-out kernel sums; the LJ percentiles are the same
    points against the eLJ bundle. Polymorph energies and packing coefficients come from `ebatch`, as in the 10k
    figure. Returns the tuple utils.combo_fig_analysis returns, then the thermo dict combo_fig takes.

    basins_from='bundle' instead runs the mean shift itself over the bundle's draws (mean_shift_anchors), with the
    same assignment rule; basins under min_basin_size per 10,000 draws are dropped and the rest are numbered by
    the density at their mode. basins_from='10k' is the extension described above."""
    from energy_sampling.eval.paper1_results.utils import clustering, dmat_row_indices
    b = load_bundle(uma_bundle)
    draws, refs = b['draws'], b['refs']
    N, R = len(draws['energy']), len(refs['labels'])
    # d_cut: another kernel width than the analysis's (basins_from='bundle' only). Every kernel sum then comes from
    # the stored distance histograms, and the mean shift, the assignment and the percentiles all use that width.
    rescaled = d_cut is not None
    assert not rescaled or basins_from == 'bundle', "a d_cut override needs the bundle's own mean shift"
    if not rescaled:
        assert abs(b['sigma']['d_cut'] / float(uma_results['d_cuts'][0]) - 1) < 1e-6, "bundle is not on this figure's kernel"
    d_cut = b['sigma']['d_cut'] if d_cut is None else float(d_cut)
    assert b['meta']['step_ind'] == uma_results['provenance']['step_ind'], "bundle and results differ in checkpoint"
    assert refs['labels'] == [str(i) for i in ebatch.identifier], "bundle references differ from the figure's"
    sigma0 = d_cut / 3
    si, ci = SIGMA_MULTS.index(1.0), SETS.index('window_reduced')
    rows = torch.nonzero(draws['window_reduced']).flatten()
    M = len(rows)
    all_sums = kernel_sums_from_hist(uma_bundle, d_cut) if rescaled else np.asarray(b['sums'][:, si, ci])
    dens = torch.from_numpy(np.asarray(all_sums[:N]))[rows]
    ref_sums = torch.from_numpy(np.asarray(all_sums[N:]))

    own = basins_from == 'bundle'
    if own:   # mean shift over the bundle's own draws
        ms_rows, anchors, _ = mean_shift_anchors(uma_bundle, d_cut, max_n_clusters, device,
                                                 dens=dens.numpy() if rescaled else None,
                                                 tag=f'_dcut{d_cut:.4f}' if rescaled else '')
        assert np.array_equal(ms_rows, rows.numpy())
        kept_10k = anchors
        a_rows = rows[torch.as_tensor(anchors)].numpy()
        a_cdf = torch.from_numpy(np.stack([np.asarray(b['cdf'][int(r)]) for r in a_rows]))
        a_empty = draws['empty'][a_rows]
    else:     # the 10k analysis's anchors, extended to every draw
        _, _, kept_10k, _, anchors = clustering(uma_results, uma_thermos, max_n_clusters=max_n_clusters,
                                                min_basin_size=min_basin_size, return_anchors=True)
        good = dmat_row_indices(uma_results['sample_batch'], uma_results, energy_function)
        a_cdf, a_empty = rdf_to_cdf(uma_results['sample_batch'].rdf[good][torch.as_tensor(anchors)].cpu())
    d_anchor = _dist_to_draws(b, a_cdf, a_empty, device, cols=rows.numpy()).double()
    w = torch.exp(-d_anchor ** 2 / (2 * sigma0 ** 2))
    tot = w.sum(0)
    share = torch.where(tot > 0, w / tot.clamp(min=1e-300), torch.zeros_like(w))
    best_share, best = share.max(0)
    kept = torch.as_tensor([int(a) in set(int(k) for k in kept_10k) for a in anchors])
    lab = torch.where((best_share >= 0.8) & kept[best], best, torch.full_like(best, -1))
    if own:   # the same share of the set that min_basin_size is of a 10k analysis
        small = [v for v in torch.unique(lab) if v >= 0 and int((lab == v).sum()) < min_basin_size * M / 10000]
        if len(small) == len([v for v in torch.unique(lab) if v >= 0]):
            print(f"  no basin holds {min_basin_size} per 10,000 draws at this kernel width; keeping the largest instead")
        else:
            for v in small:
                lab[lab == v] = -1
    basin_ids = [int(v) for v in torch.unique(lab) if v >= 0]
    if max_basins is not None and len(basin_ids) > max_basins:   # the figure grid holds this many: keep the largest
        basin_ids.sort(key=lambda v: -int((lab == v).sum()))
        for v in basin_ids[max_basins:]:
            lab[lab == v] = -1
        basin_ids = basin_ids[:max_basins]
    # a basin's probability maximum is its MODE: the densest member within one kernel width of the 10k anchor (the
    # mean-shift mode). The densest member anywhere would sit on the basin's border with a denser neighbour.
    p_max = {}
    for v in basin_ids:
        near = torch.nonzero((lab == v) & (d_anchor[v] <= sigma0)).flatten()
        p_max[v] = int(near[dens[near].argmax()]) if len(near) else int(d_anchor[v].argmin())
    # numbered as in the 10k figure (its density order), so a near-tie between two modes cannot renumber the basins
    if own:
        basin_ids.sort(key=lambda v: -float(dens[p_max[v]]))
    else:
        rank_10k = {int(a): list(int(k) for k in kept_10k).index(int(a)) for a in anchors
                    if int(a) in set(int(k) for k in kept_10k)}
        basin_ids.sort(key=lambda v: rank_10k[int(anchors[v])])
    n_basins = len(basin_ids)
    indexed = torch.full((M,), -1, dtype=torch.long)
    for new, v in enumerate(basin_ids):
        indexed[lab == v] = new
    p_maxima = np.array([p_max[v] for v in basin_ids])
    energy, cp = draws['energy'][rows].flatten(), draws['packing_coeff'][rows].flatten()
    e_min = np.array([int(torch.nonzero(indexed == k).flatten()[energy[indexed == k].argmin()])
                      for k in range(n_basins)])
    sizes = [int((indexed == k).sum()) for k in range(n_basins)]
    print(f"  bundle landscape: {M} draws; basin sizes {sizes}; unassigned {int((indexed < 0).sum())} "
          f"({100 * float((indexed < 0).float().mean()):.1f}%)")

    emb = embed_bundles([uma_bundle], device=device)
    xy = emb['draws'][0][rows.numpy()]
    sample_embedding = np.concatenate([xy, emb['refs'], xy[e_min]], 0)
    polymorph_inds = [M + r for r in range(R)]
    new_min_inds = [M + R + k for k in range(n_basins)]

    stats = {'energy_function': energy_function, 'paired_density': True}
    stats['sample_energy'] = torch.cat([energy[e_min], ebatch[energy_function].flatten().cpu()]).numpy()
    stats['sample_cp'] = torch.cat([cp[p_maxima], ebatch.packing_coeff.flatten().cpu()]).numpy()
    pts = torch.cat([dens[p_maxima], ref_sums])
    stats['density'] = (pts / dens.max()).float()
    stats['density_pct'], stats['density_pct_floor'], stats['below_resolution'] = _rank(dens, pts)
    pt_cdf = torch.cat([torch.from_numpy(np.ascontiguousarray(b['cdf'][np.sort(rows[p_maxima].numpy())])),
                        refs['cdf']])
    order = np.argsort(np.argsort(rows[p_maxima].numpy()))   # cdf rows were read sorted; restore basin order
    pt_cdf = torch.cat([pt_cdf[:n_basins][order], pt_cdf[n_basins:]])
    pt_empty = torch.cat([draws['empty'][rows[p_maxima]], refs['empty']])
    ps = point_sums(elj_bundle, cdf=pt_cdf, empty=pt_empty, device=device, d_cut=d_cut if rescaled else None)
    assert abs(ps['d_cut'] / d_cut - 1) < 1e-6, "the eLJ bundle is on a different kernel"
    (stats['elj_density_pct'], stats['elj_density_pct_floor'],
     stats['elj_below_resolution']) = _rank(ps['draw_sums'], ps['point_sums'])
    stats['elj_density'] = (ps['point_sums'] / ps['draw_sums'].max()).numpy()
    stats['uma_bundle_n_draws'], stats['elj_bundle_n_draws'] = M, ps['n_draws']
    # scale_label names the kernel the basins, shading and first percentile rows use; extra_scales {label: d_cut}
    # adds the SAME points' percentiles at other kernel widths (basins and points unchanged), for the table only
    stats['scale_label'] = scale_label
    stats['extra_scales'] = []
    for label, dc in (extra_scales or {}).items():
        s2 = kernel_sums_from_hist(uma_bundle, dc)
        d2 = torch.from_numpy(s2[:N])[rows]
        u = _rank(d2, torch.cat([d2[p_maxima], torch.from_numpy(s2[N:])]))
        p2 = point_sums(elj_bundle, cdf=pt_cdf, empty=pt_empty, device=device, d_cut=dc)
        e = _rank(p2['draw_sums'], p2['point_sums'])
        stats['extra_scales'].append({'label': label, 'uma_pct': u[0], 'uma_below': u[2], 'elj_pct': e[0],
                                      'elj_below': e[2]})
        print(f"  scale {label} (d_cut {dc:.4f}): UMA percentiles {np.round(u[0], 1).tolist()} "
              f"(<LOD {u[2].astype(int).tolist()}), LJ {np.round(e[0], 1).tolist()} (<LOD {e[2].astype(int).tolist()}); "
              f"draws below one draw's weight: UMA {u[1]:.1f}%, LJ {e[1]:.1f}%")

    ref_d = torch.from_numpy(np.asarray(b['ref_dist']))[:, rows]
    polymorph_basin_index = torch.stack([torch.stack([ref_d[r][indexed == k].min() for k in range(n_basins)])
                                         for r in range(R)]).argmin(1)
    basin_colorscale = px.colors.qualitative.Vivid[:n_basins + 2]
    basin_colorscale[0] = 'rgb(100, 100, 100)'
    sample_colors = [basin_colorscale[i + 1] for i in indexed.tolist()]
    polymorph_colorscale = [basin_colorscale[1 + int(i)] for i in polymorph_basin_index]
    basin_min_batch = _BasinMinima(energy_function, energy[e_min], cp[e_min])
    thermos = {'density': dens.numpy()}
    return (basin_colorscale, basin_min_batch, indexed, n_basins, new_min_inds, p_maxima, cp.numpy(),
            polymorph_basin_index, polymorph_colorscale, polymorph_inds, sample_colors, sample_embedding,
            energy.numpy(), np.arange(M), stats), thermos


def bundle_duo(uma_bundle, elj_bundle, device='cuda'):
    """The LJ-vs-UMA embedding over both bundles' window & reduced draws: what generate_figs.duo_embedding_fig
    plots and saves for the fused figure."""
    cross_knn(uma_bundle, elj_bundle, device=device)
    emb = embed_bundles([uma_bundle, elj_bundle], device=device,
                        landmark_kwargs=dict(init='spectral', repulsion_strength=2.0))
    out = {}
    for tag, o, xy in (('uma', uma_bundle, emb['draws'][0]), ('lj', elj_bundle, emb['draws'][1])):
        keep = torch.load(os.path.join(o, 'draws.pt'), weights_only=False)['window_reduced'].numpy()
        out[tag] = xy[keep]
    out['poly'] = emb['refs']
    return out
