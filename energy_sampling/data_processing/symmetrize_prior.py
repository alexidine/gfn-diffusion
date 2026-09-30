"""Symmetrise a crystal prior file under the Euclidean normaliser, in the chart the trainer actually uses.

    python -m data_processing.symmetrize_prior SRC.pt OUT.pt [--layout orbits|one] [--device cuda] [--rescore-chunk 1000] [--limit N]

SCOPE: space group 2 (P-1), Z' = 1, handedness +1 on every row. Anything else is refused.

WHAT IT WRITES. SRC's train.py layout ({prior, equalized_prior, thermal_scaling_factor, ...}; `prior_path` reads
equalized_prior, `molecules_path` reads prior) with every row of both datasets rewritten, and OUT's `_provenance.pt`
beside it (per row: source index, y/z shift, whether it is an opposite-face copy, whether it is a duplicate, and under
`one` the image index). Two layouts:
  orbits  every source row becomes ROWS_PER_SOURCE = 8 consecutive rows, identity first: exact symmetry, 8x the size.
  one     every source row is written ONCE, as one of its valid descriptions chosen by `choose_one`: the same size as
          the source, and the same distribution as `orbits` in the aggregate. The image index cycles within the rows
          that share a nearest anchor (the file stores noised rows shuffled; `nearest_anchor` recovers the grouping),
          so an anchor's dozen noised rows spread evenly over its descriptions; the anchor set itself cycles globally.
Only `aunit_centroid` differs between a row and its source; the stored energies are carried over, which the rescoring
below proves is right.

THE CHART, and why it is not simply the 8 cosets. N_E(P-1)/P-1 has 8 cosets, the half-cell origin shifts. The trainer
builds every crystal with aunit_handedness +1 (energies/molecular_crystal.py::MolecularCrystal.init_blank_crystal_batch)
and scores a stored row from its 12 latent coordinates alone, and its asymmetric-unit box spans x in [0, 1/2] and
y, z in [0, 1). In that chart
  * a half shift in y and/or z stays inside the box: 4 descriptions of every crystal, pure latent shifts;
  * a half shift in x puts the +1 molecule outside the box. The image that IS inside is the inversion partner, the
    molecule with handedness -1 -- a row the trainer would read as a different crystal (measured 2026-09-30 on 951
    such images from standardize.normalizer_images: eLJ off by a median 106 raw units, 15 nat, none within 0.01).
    In the +1 chart the x shift is the box's own period: the shifted crystal has no description there and is the same
    structure as the unshifted one. It adds nothing, EXCEPT
  * on an x face. A row with x on a face (x = 0 or x = 1/2; 16% of the MIPCAS noised rows, clamp-era anchors) has its
    x-shifted image on the opposite face with the same handedness. The two faces are the same points of a periodic
    axis the model does not wrap, so both are states it can produce; the row gets both.
So a face row has 8 descriptions and an interior row 4. Every source row is written as 8 rows all the same, the
interior ones as their 4 images twice, so that every crystal keeps equal weight (owner rule 2026-09-28: exactly N
images per row, duplicates allowed). Noise commutes with all of these maps (shifts and a face swap are latent
isometries), so noising a symmetrised anchor set is symmetric in distribution.

CHECKS, all run on every build and all asserted (a check that did not run is not reassurance):
  structure   8 rows per source; handedness, cell, orientation and every other stored tensor equal the source's;
              the latent differs from the source's only in the three centroid coordinates, by the intended maps
  energy      every DISTINCT output row rescored through the trainer's analyze call (a duplicate is bit-identical
              to its original); |eLJ(image) - eLJ(its source row, rescored)| under ENERGY_TOL, the stored source
              energies within ENERGY_TOL of their rescoring, reduction penalty zero
  crystal     a sample of images is the same crystal as its source by atomwise RDF distance
  tool        on interior rows the y/z images equal standardize.normalizer_images' cosets for the same shifts
"""
import argparse
import os
import sys
import time

import torch

ROWS_PER_SOURCE = 8
#: (dy, dz) of the four in-box images, identity first
YZ_SHIFTS = ((0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5))
X_BOX = 0.5          # the sg 2 asymmetric-unit box extent in x; y and z span the whole cell
FACE_TOL = 1e-6      # fractional: a row this close to x = 0 or x = X_BOX is ON the face (a clamped coordinate)
CELL_EDGE = 0.9999   # mxtaltools.crystal_search.crystal_opt_utils.CELL_EDGE: the builders clip a centre to [0, CELL_EDGE]
ENERGY_TOL = 0.25    # raw eLJ units. Float noise between two scorings of one crystal reaches ~0.06 over 1.6M MIPCAS rows;
                     # an image the trainer reads as a different crystal is off by a median 106
ANALYZE_KW = dict(cutoff=10, supercell_size=10, std_orientation=False)   # MolecularCrystal.analyze_crystal_batch


def image_centroids(centroid: torch.Tensor):
    """[n, 3] fractional centroids -> ([8n, 3] image centroids, source [8n], shift_id [8n], face_copy [8n] bool,
    duplicate [8n] bool). Row 8i + k is source i's image k: shift YZ_SHIFTS[k % 4]; for k >= 4 the opposite x face if
    the source is on one (face_copy), else the same image again (duplicate). Image 0 is the source, untouched."""
    if centroid.ndim != 2 or centroid.shape[1] != 3:
        raise ValueError(f'expected [n, 3] centroids (Z\' = 1), got {tuple(centroid.shape)}')
    c = centroid.double()
    if float(c[:, 0].min()) < -FACE_TOL or float(c[:, 0].max()) > X_BOX + FACE_TOL:
        raise ValueError(f'centroid x outside the sg 2 box [0, {X_BOX}]: [{float(c[:, 0].min())}, {float(c[:, 0].max())}]')
    if float(c[:, 1:].min()) < 0 or float(c[:, 1:].max()) >= 1:
        raise ValueError('centroid y/z outside [0, 1)')
    n = len(c)
    source = torch.arange(n).repeat_interleave(ROWS_PER_SOURCE)
    k = torch.arange(ROWS_PER_SOURCE).repeat(n)
    shift_id = k % 4
    second = k >= 4
    out = c[source].clone()
    shifts = torch.tensor(YZ_SHIFTS, dtype=torch.float64)[shift_id]            # [8n, 2]
    moved = shifts > 0
    f = out[:, 1:] + shifts
    f = f - torch.floor(f)
    # mxtaltools' standardize._snap_edge_sliver: a moved coordinate in (CELL_EDGE, 1) nearer 1 than CELL_EDGE is 0
    snap = moved & (f > CELL_EDGE) & ((1 - f) < (f - CELL_EDGE))
    f = torch.where(snap, torch.zeros_like(f), f)
    out[:, 1:] = torch.where(moved, f, out[:, 1:])
    x = out[:, 0]
    low, high = x <= FACE_TOL, x >= X_BOX - FACE_TOL
    face_copy = second & (low | high)
    out[:, 0] = torch.where(second & low, torch.full_like(x, X_BOX), torch.where(second & high, torch.zeros_like(x), x))
    duplicate = second & ~(low | high)
    return out.to(centroid.dtype), source, shift_id, face_copy, duplicate


def choose_one(face_copy_ok, group):
    """One image index in [0, 8) per source row for the `one` layout: rows are visited group by group (a group is the
    rows sharing a nearest anchor; the anchor set itself is one group) and, separately for a group's rows on an x face
    (all eight descriptions valid) and its interior rows (four), the image index cycles through the valid set, starting
    at an offset that itself cycles with the group, so that a group's rows spread evenly over the descriptions its
    members can take AND groups of one or two rows spread evenly across groups. Deterministic; no row is dropped."""
    n = len(face_copy_ok)
    key = group * 2 + face_copy_ok.long()
    order = torch.argsort(key, stable=True)
    pos = torch.empty(n, dtype=torch.long)
    g_sorted = key[order]
    starts = torch.cat([torch.tensor([0]), torch.nonzero(g_sorted[1:] != g_sorted[:-1]).flatten() + 1, torch.tensor([n])])
    for a, b in zip(starts[:-1].tolist(), starts[1:].tolist()):
        pos[order[a:b]] = torch.arange(b - a)
    return torch.where(face_copy_ok, (pos + group) % ROWS_PER_SOURCE, (pos + group) % 4)


def nearest_anchor(latents, anchors, chunk=1024):
    """Index of the nearest anchor in the 12-D latent (y and z wrapped), for grouping noised rows by the anchor they
    were noised from. The file stores them shuffled, so this is the only way back to that grouping."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    L, A = latents.float().to(dev), anchors.float().to(dev)
    out = torch.empty(len(L), dtype=torch.long)
    for lo in range(0, len(L), chunk):
        x = (L[lo:lo + chunk, None, :] - A[None, :, :]).abs()
        x[..., 7:9] = torch.minimum(x[..., 7:9], 2 - x[..., 7:9])
        out[lo:lo + chunk] = x.norm(dim=-1).argmin(1).cpu()
    return out


def wrapped_pairs(X, r):
    """Every pair (i < j) of rows of the [n, 12] latent array X within Euclidean distance r, with the centroid y and z
    coordinates (columns 7 and 8, whole-cell axes) periodic with period 2: rows within r of a seam are also entered as
    their images across it, so a pair straddling the wrap is found."""
    import numpy as np
    from scipy.spatial import cKDTree
    n = len(X)
    pts, idx = [X], [np.arange(n)]
    for dy in (-2.0, 0.0, 2.0):
        for dz in (-2.0, 0.0, 2.0):
            if dy == 0 and dz == 0:
                continue
            m = np.ones(n, bool)
            if dy:
                m &= X[:, 7] * np.sign(dy) < -1 + r
            if dz:
                m &= X[:, 8] * np.sign(dz) < -1 + r
            if m.any():
                Y = X[m].copy()
                Y[:, 7] += dy
                Y[:, 8] += dz
                pts.append(Y)
                idx.append(np.nonzero(m)[0])
    ids = np.concatenate(idx)
    pr = cKDTree(np.concatenate(pts)).query_pairs(r, output_type='ndarray')
    i, j = ids[pr[:, 0]], ids[pr[:, 1]]
    keep = i != j
    i, j = np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep])
    return np.unique(np.stack([i, j], 1), axis=0) if len(i) else np.zeros((0, 2), dtype=np.int64)


def greedy_thin(n, pairs, order):
    """Leader clustering: visit rows in `order`, keep a row unless a kept row already lies within the cutoff (a pair),
    and retire its neighbours. Returns the kept-row mask. With a group-invariant metric and an order in which every
    row of one source precedes every row of the next, the decision is the same at every image, so orbits survive or
    fall together (asserted by the caller)."""
    import numpy as np
    from scipy.sparse import coo_matrix
    adj = coo_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    alive = np.ones(n, bool)
    kept = np.zeros(n, bool)
    indptr, indices = adj.indptr, adj.indices
    for k in order:
        if alive[k]:
            kept[k] = True
            alive[indices[indptr[k]:indptr[k + 1]]] = False
    return kept


def thin_orbits(orbits, prov, cutoff):
    """The orbit batch thinned at `cutoff` in the wrapped 12-D latent: exact duplicates go first, then leader clustering
    over the distinct rows in (source energy, source index, image) order. Returns (batch, provenance, report) and
    asserts that every surviving row's in-box y/z orbit survived whole; opposite-face pairings may split (an interior
    row near one face retires the same-face images of a face row and cannot reach the other face) and are counted."""
    import numpy as np
    distinct = torch.nonzero(~prov['duplicate']).flatten().numpy()
    X = _latents(orbits).numpy()[distinct]
    src = prov['source'].numpy()[distinct]
    e = orbits.elj.double().flatten().numpy()[distinct]
    order = np.lexsort((np.arange(len(distinct)), src, e))        # energy, then source index, then image
    kept = greedy_thin(len(distinct), wrapped_pairs(X, cutoff), order)
    sel = distinct[kept]
    out = orbits.subsample_new_batch(torch.as_tensor(sel))
    newprov = {k: v[torch.as_tensor(sel)] for k, v in prov.items()}
    # y/z orbit closure: within each (source, face side) the four shifts live or die together
    side = newprov['face_copy'].long()
    key = newprov['source'] * 2 + side
    counts = torch.bincount(key)
    counts = counts[counts > 0]
    assert bool((counts == 4).all()), f'thinning split a y/z orbit: sides with {sorted(set(counts.tolist()))} rows'
    n_src = int(prov['source'].max()) + 1
    face_src = prov['source'][prov['face_copy']].unique()
    has_side = torch.zeros(n_src, 2, dtype=torch.bool)
    has_side[newprov['source'], side] = True
    kept_face = has_side[face_src].any(1)
    split = kept_face & ~has_side[face_src].all(1)
    return out, newprov, dict(cutoff=cutoff, distinct_in=len(distinct), rows=out.num_graphs, sources_kept=int(newprov['source'].unique().numel()),
                              face_sources_kept=int(kept_face.sum()), face_pairs_split=int(split.sum()))


def children_of(parent, kept_anchors, n_anchors):
    """Boolean mask over noised rows: True where the row's nearest anchor (`parent`, [n] long) is one of `kept_anchors`."""
    keep = torch.zeros(n_anchors, dtype=torch.bool)
    keep[torch.as_tensor(kept_anchors, dtype=torch.long)] = True
    return keep[parent]


def symmetrize_batch(batch):
    """The symmetrised batch (8 rows per source, see the module docstring) and its provenance dict."""
    sgs = set(batch.sg_ind.reshape(-1).tolist())
    zps = set(batch.z_prime.reshape(-1).tolist())
    hand = set(batch.aunit_handedness.reshape(-1).tolist())
    if sgs != {2} or zps != {1} or hand != {1.0}:
        raise NotImplementedError(f'only space group 2, Z\' = 1, handedness +1 is implemented; got sg {sorted(sgs)}, '
                                  f'z_prime {sorted(zps)}, handedness {sorted(hand)}')
    images, source, shift_id, face_copy, duplicate = image_centroids(batch.aunit_centroid)
    out = batch.subsample_new_batch(source)
    out.aunit_centroid = images
    return out, dict(source=source, shift_id=shift_id, face_copy=face_copy, duplicate=duplicate)


def _latents(batch):
    return batch.latent_transform(batch.full_cell_parameters()).double()


def check_structure(src, out, prov, name):
    n = src.num_graphs
    assert out.num_graphs == ROWS_PER_SOURCE * n, name
    assert torch.equal(torch.bincount(prov['source'], minlength=n), torch.full((n,), ROWS_PER_SOURCE)), name
    s = prov['source']
    for key in ('cell_lengths', 'cell_angles', 'aunit_orientation', 'aunit_handedness', 'sg_ind', 'z_prime', 'num_atoms',
                'elj', 'reduction_en'):
        if hasattr(src, key):
            assert torch.equal(getattr(out, key), getattr(src, key)[s]), f'{name}: {key} moved'
    assert torch.equal(out.z, src.z.reshape(n, -1)[s].reshape(-1)) and torch.equal(out.pos, src.pos.reshape(n, -1, 3)[s].reshape(-1, 3)), name
    ident = (prov['shift_id'] == 0) & ~prov['face_copy'] & ~prov['duplicate']
    assert int(ident.sum()) == n and torch.equal(out.aunit_centroid[ident], src.aunit_centroid), f'{name}: identity rows moved'
    L, Ls = _latents(out), _latents(src)[s]
    d = (L - Ls).abs()
    assert float(d[:, :6].max()) < 1e-6 and float(d[:, 9:].max()) < 1e-6, f'{name}: a non-centroid latent moved'
    shifts = torch.tensor(YZ_SHIFTS, dtype=torch.float64)[prov['shift_id']]
    for col, ax in ((7, 0), (8, 1)):
        want = torch.where(shifts[:, ax] > 0, torch.ones_like(d[:, col]), torch.zeros_like(d[:, col]))
        # a half cell is 1.0 in latent units; the CELL_EDGE clip and the sliver snap move a face coordinate by <= 2e-4
        assert float((d[:, col] - want).abs().max()) < 5e-4, f'{name}: latent {col} is not a half-cell shift'
    fc = prov['face_copy']
    assert float(d[~fc, 6].max()) < 1e-6, f'{name}: x moved on a row that is not a face copy'
    if fc.any():
        assert float((L[fc, 6] + Ls[fc, 6]).abs().max()) < 1e-3 and float(Ls[fc, 6].abs().min()) > 1 - 1e-3, f'{name}: face copy is not the opposite face'
    dup = torch.nonzero(prov['duplicate']).flatten()
    assert torch.equal(out.aunit_centroid[dup], out.aunit_centroid[dup - 4]), f'{name}: a duplicate differs from its original'
    return dict(rows=out.num_graphs, sources=n, face_sources=int(fc.sum()) // 4, duplicate_rows=int(prov['duplicate'].sum()))


def rescore(batch, device, chunk):
    """[n, 2]: eLJ and the reduction penalty of every row through the trainer's analyze call."""
    out, lo = [], 0
    while lo < batch.num_graphs:
        c = batch.subsample_new_batch(torch.arange(lo, min(lo + chunk, batch.num_graphs))).to(device)
        try:
            with torch.no_grad():
                o = c.analyze(['reduction_en', 'elj'], **ANALYZE_KW)
        except RuntimeError as err:
            # CUDA reports an exhausted card as OutOfMemoryError or as AcceleratorError, both RuntimeErrors. A few
            # high-energy noised rows build very large clusters: halve and retry the same rows. Below 50 rows the
            # problem is not the chunk.
            if 'out of memory' not in str(err).lower():
                raise
            del c
            torch.cuda.empty_cache()
            if chunk <= 50:
                raise
            chunk //= 2
            print(f'    rescore: out of memory, chunk -> {chunk}', flush=True)
            continue
        out.append(torch.stack([o['elj'].float().cpu().flatten(), o['reduction_en'].float().cpu().flatten()], 1))
        lo += c.num_graphs
        del c, o
        if device != 'cpu':
            torch.cuda.empty_cache()
    return torch.cat(out)


def check_energy(src, out, prov, device, chunk, name):
    t0 = time.time()
    # duplicates are bit-identical to their originals four rows earlier (check_structure): score each distinct row once
    distinct = torch.nonzero(~prov['duplicate']).flatten()
    e = torch.empty(out.num_graphs, 2)
    e[distinct] = rescore(out.subsample_new_batch(distinct), device, chunk)
    dup = torch.nonzero(prov['duplicate']).flatten()
    e[dup] = e[dup - 4]
    assert bool(torch.isfinite(e).all()), f'{name}: non-finite rescored energy'
    if out.num_graphs == ROWS_PER_SOURCE * src.num_graphs:
        # orbits layout: an image against ITS OWN SOURCE ROW rescored the same way (row 8i), and separately the float
        # noise between a source row's stored and rescored energy. A description the trainer reads as another crystal
        # is off by ~100.
        own = e[prov['source'] * ROWS_PER_SOURCE, 0]
        base = (e[::ROWS_PER_SOURCE, 0] - src.elj.float().flatten()).abs()
    else:
        # one layout: the source rows are not all present, so rescore them too (same call, same float regime)
        own = rescore(src, device, chunk)[prov['source'], 0]
        base = (own - src.elj.float().flatten()[prov['source']]).abs()
    d = (e[:, 0] - own).abs()
    worst = {k: float(d[m].max()) for k, m in (('in-box shift', ~prov['face_copy']), ('opposite-face copy', prov['face_copy'])) if m.any()}
    assert float(d.max()) < ENERGY_TOL, f'{name}: |eLJ(image) - eLJ(source)| up to {float(d.max()):.4f} raw (tolerance {ENERGY_TOL}); {worst}'
    assert float(base.max()) < ENERGY_TOL, f'{name}: stored vs rescored source eLJ differ by up to {float(base.max()):.4f} raw'
    assert float(e[:, 1].abs().max()) == 0.0, f'{name}: reduction penalty nonzero on {int((e[:, 1] != 0).sum())} rows'
    return dict(rescored_rows=len(distinct), seconds=time.time() - t0, max_image_vs_source=float(d.max()),
                p999_image_vs_source=float(d.quantile(0.999)) if len(d) < 16_000_000 else float('nan'),
                max_stored_vs_rescored_source=float(base.max()), **{f'max {k}': v for k, v in worst.items()})


def check_same_crystal(src, out, prov, name, n_sources=48):
    """atomwise RDF distance of a sample of images (half of them face sources when there are any) to their sources."""
    from mxtaltools.crystal_search import coordinator as co
    from mxtaltools.dataset_utils.utils import collate_data_list
    kw = dict(co.RDF_KW, rdf_mode='atomwise')
    face_src = torch.unique(prov['source'][prov['face_copy']])
    g = torch.Generator().manual_seed(0)
    pick = torch.cat([face_src[torch.randperm(len(face_src), generator=g)[:n_sources // 2]],
                      torch.randperm(src.num_graphs, generator=g)[:n_sources - min(len(face_src), n_sources // 2)]])
    rows = (pick[:, None] * ROWS_PER_SOURCE + torch.arange(ROWS_PER_SOURCE)[None]).flatten()

    def rdf(b):
        out_ = []
        items = b.batch_to_list()
        for lo in range(0, len(items), 32):
            c = collate_data_list([r.clone() for r in items[lo:lo + 32]], exclude_keys=co.RDF_DROP)
            with torch.no_grad():
                r = c.analyze(['rdf'], assign_outputs=False, **kw)['rdf']
            out_.append((r[0] if isinstance(r, (tuple, list)) else r).float())
        return torch.cat(out_)

    d = torch.diagonal(co.rdf_distance_matrix(rdf(out.subsample_new_batch(rows)), rdf(src.subsample_new_batch(pick))[torch.arange(len(pick)).repeat_interleave(ROWS_PER_SOURCE)]))
    assert float(d.max()) < 1e-3, f'{name}: an image is not its source crystal (atomwise RDF distance {float(d.max()):.2e})'
    return dict(images=len(rows), max_rdf_distance=float(d.max()))


def check_against_tool(src, out, prov, name, n_sources=64):
    """On interior rows the y/z images must equal standardize.normalizer_images' images for the same origin shifts."""
    import numpy as np
    from mxtaltools.crystal_search.standardize import normalizer_images
    lat = _latents(src)
    interior = torch.nonzero((lat[:, 6:9].abs() < 1 - 1e-3).all(1)).flatten()
    pick = interior[:: max(1, len(interior) // n_sources)][:n_sources]
    eye = np.eye(3)
    table = {'2': [(eye, [0.0, dy, dz]) for dy, dz in YZ_SHIFTS[1:]]}
    tool, source, coset = normalizer_images(src.subsample_new_batch(pick), table=table)
    assert bool((tool.aunit_handedness == 1).all()), f'{name}: the tool flipped handedness on an interior in-box shift'
    mine = out.subsample_new_batch((pick[source] * ROWS_PER_SOURCE + coset))
    d = (tool.full_cell_parameters().double() - mine.full_cell_parameters().double()).abs().max()
    assert float(d) < 1e-5, f'{name}: builder and normalizer_images disagree by {float(d):.2e} in cell parameters'
    return dict(sources=len(pick), images=tool.num_graphs, max_param_delta=float(d))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--rescore-chunk', type=int, default=1000)
    ap.add_argument('--limit', type=int, default=None, help='build from the first N rows of each dataset (a smoke; not a prior)')
    ap.add_argument('--layout', choices=('orbits', 'one'), default='orbits',
                    help="orbits: every source row as its 8 rows (exact; 8x the size). one: every source row ONCE, written as one "
                         "of its valid descriptions chosen by choose_one (same size as the source; symmetric in the aggregate, "
                         "balanced within each anchor's rows)")
    ap.add_argument('--thin', type=float, default=None,
                    help='orbits layout only: leader-cluster the distinct orbit rows at this cutoff in the wrapped 12-D latent, '
                         'lowest energy kept first (exact duplicates go first); y/z orbits are asserted to survive whole')
    ap.add_argument('--thin-anchors', type=float, default=None,
                    help='thin the ANCHOR set (prior) at this cutoff the same way, then keep only the noised rows whose nearest '
                         'anchor survived, with every image the layout gives them; the noised rows themselves are not thinned')
    args = ap.parse_args(argv)
    if args.thin is not None and args.layout != 'orbits':
        sys.exit('--thin applies to --layout orbits')
    if args.thin is not None and args.thin_anchors is not None:
        sys.exit('--thin and --thin-anchors are alternatives')
    if os.path.exists(args.out):
        sys.exit(f'{args.out} exists -- refusing to overwrite')
    data = torch.load(args.src, map_location='cpu', weights_only=False)
    out_data, provenance, report = dict(data), {}, {}
    anchors = data['prior']
    kept_anchors = None
    for key in ('prior', 'equalized_prior'):
        src = data[key]
        if args.limit:
            src = src.subsample_new_batch(torch.arange(min(args.limit, src.num_graphs)))
        kept_rows = None
        if args.thin_anchors is not None and key == 'equalized_prior':
            parent = nearest_anchor(_latents(src), _latents(anchors))
            keep = children_of(parent, kept_anchors, anchors.num_graphs)
            kept_rows = torch.nonzero(keep).flatten()
            print(f'[{key}] parents thinned at {args.thin_anchors}: {len(kept_anchors):,} of {anchors.num_graphs:,} anchors keep '
                  f'{int(keep.sum()):,} of {src.num_graphs:,} noised rows', flush=True)
            src = src.subsample_new_batch(kept_rows)
            parent = parent[kept_rows]
        t0 = time.time()
        orbits, prov = symmetrize_batch(src)
        print(f'[{key}] {src.num_graphs} rows -> {orbits.num_graphs} orbit rows ({time.time() - t0:.1f}s); checking...', flush=True)
        # the orbit checks prove every one of the 8 rows; the `one` layout then keeps a subset of proven rows
        report[key] = dict(structure=check_structure(src, orbits, prov, key),
                           tool=check_against_tool(src, orbits, prov, key),
                           crystal=check_same_crystal(src, orbits, prov, key))
        if args.layout == 'one':
            if key == 'prior':
                group = torch.zeros(src.num_graphs, dtype=torch.long)      # the anchors themselves: one group, cycle globally
            else:
                group = nearest_anchor(_latents(src), _latents(anchors))
            face_ok = prov['face_copy'].reshape(-1, ROWS_PER_SOURCE).any(1)
            k = choose_one(face_ok, group)
            sel = torch.arange(src.num_graphs) * ROWS_PER_SOURCE + k
            out = orbits.subsample_new_batch(sel)
            prov = {name: t[sel] for name, t in prov.items()}
            prov['image'] = k
            hist = torch.bincount(k, minlength=ROWS_PER_SOURCE).tolist()
            report[key]['layout'] = dict(rows=out.num_graphs, groups=int(group.unique().numel()), image_counts=hist,
                                         face_sources=int(face_ok.sum()), opposite_face_rows=int(prov['face_copy'].sum()))
            assert not prov['duplicate'].any() and out.num_graphs == src.num_graphs, key
        elif args.thin is not None or (args.thin_anchors is not None and key == 'prior'):
            t0 = time.time()
            out, prov, report[key]['thin'] = thin_orbits(orbits, prov, args.thin if args.thin is not None else args.thin_anchors)
            report[key]['thin']['seconds'] = time.time() - t0
            if key == 'prior':
                kept_anchors = prov['source'].unique()
        else:
            out = orbits
        if kept_rows is not None:
            prov['row_in_source_file'] = kept_rows[prov['source']]
            prov['parent_anchor'] = parent[prov['source']]
            report[key]['parents'] = dict(cutoff=args.thin_anchors, anchors_kept=int(len(kept_anchors)), noised_rows_kept=int(len(kept_rows)))
        report[key]['energy'] = check_energy(src, out, prov, args.device, args.rescore_chunk, key)
        for part, vals in report[key].items():
            print(f'    {part:<9} ' + ', '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in vals.items()), flush=True)
        out_data[key] = out
        provenance[key] = prov
    torch.save(out_data, args.out)
    stem = args.out[:-3] if args.out.endswith('.pt') else args.out
    torch.save(dict(source_file=os.path.abspath(args.src), layout=args.layout, thin=args.thin, thin_anchors=args.thin_anchors,
                    rows_per_source=ROWS_PER_SOURCE, yz_shifts=YZ_SHIFTS,
                    x_box=X_BOX, face_tol=FACE_TOL, limit=args.limit, report=report, **provenance), stem + '_provenance.pt')
    print(f'wrote {args.out} ({os.path.getsize(args.out):,} bytes) and {stem}_provenance.pt')


if __name__ == '__main__':
    main()
