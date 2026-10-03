"""Assemble a shipped prior file from polished anchors and their capped-MC flood (pooled prior rebuild, 2026-10).

    python -m data_processing.pool_assemble ANCHORS.pt FLOOD_DIR OUT.pt --sg SG --key elj|uma --mol MOL.pt
        [--dedupe 0.01] [--no-fold] [--target-rows N] [--check 2000] [--mlip_path P]

ANCHORS.pt   a prior-layout file from `pool_anchors export` (rows in the trainer's chart, handedness +1).
FLOOD_DIR    a capped_mc output directory, or a directory of them (every shard_*.pt below it is read): each accepted
             move's cell parameters and energy in the training currency.

Steps
  1. candidates = the anchors, then every flood state, as rows on the anchors' molecule at handedness +1.
  2. de-dupe in the trainer's latent (12-D; the chart's periodic coordinates wrapped, as capped_mc wraps them): greedy,
     anchors first in ascending energy, then flood states in ascending energy; a row within --dedupe (Euclidean) of a
     kept row is dropped. No anchor is dropped for a flood state; an anchor within --dedupe of a lower anchor is.
  3. fold (unless --no-fold): every kept row is written with all its normaliser images the +1 chart holds
     (pool_anchors.Chart.images: sg 2 four, eight for a centre on an x face; sg 14 eight), images carrying the row's
     energy. With --target-rows N the flood radius of step 2 is widened by x1.25 at a time (the anchors' stays at
     --dedupe) until kept rows x images <= N: a coarser cover of the same flooded volume, not an energy cut.
  4. write {'prior': batch, 'equalized_prior': the same batch, 'thermal_scaling_factor', ['uma_energy_state']} plus
     provenance (source row, image id, anchor flag). Energies are stored in the file's raw currency under --key (eLJ:
     the flood's training-currency energy divided by the anchors file's thermal_scaling_factor).
  5. check: --check rows drawn at random are re-scored the trainer's way against their stored energy (all asserted).
"""
import argparse
import glob
import os
import time

import numpy as np
import torch

from data_processing.pool_anchors import Chart

PHI, RMAG = 10, 11


def periodic_dims(sg):
    from mxtaltools.constants.asymmetric_units import ASYM_UNITS
    full = [ax for ax in range(3) if float(ASYM_UNITS[str(sg)][ax]) == 1.0]
    return sorted(set([6 + ax for ax in full] + ([6] if sg == 2 else []) + [PHI, RMAG]))


def load_flood(flood_dir):
    files = sorted(glob.glob(os.path.join(flood_dir, '**', 'shard_*.pt'), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise SystemExit(f'no shard_*.pt under {flood_dir}')
    cp, E = [], []
    for f in files:
        d = torch.load(f, weights_only=False)
        cp.append(np.asarray(d['cell_params'], dtype=np.float32))
        E.append(np.asarray(d['E'], dtype=np.float64))
    return torch.from_numpy(np.concatenate(cp)), torch.from_numpy(np.concatenate(E)), len(files)


def rows_on(template, cp):
    """A +1 batch of len(cp) rows on the template's molecule with the given cell parameters [n, 12]."""
    b = template.subsample_new_batch(torch.zeros(len(cp), dtype=torch.long))
    b.cell_lengths, b.cell_angles = cp[:, :3].clone(), cp[:, 3:6].clone()
    b.aunit_centroid, b.aunit_orientation = cp[:, 6:9].clone(), cp[:, 9:12].clone()
    b.aunit_handedness = torch.ones_like(b.aunit_handedness)
    b.box_analysis()
    return b


def latents_of(template, cp, chunk=20000):
    out = []
    for lo in range(0, len(cp), chunk):
        out.append(rows_on(template, cp[lo:lo + chunk]).latent_params().double())
    return torch.cat(out)


def wrapped(lat, per):
    """Latents shifted to [0, 2] with the periodic coordinates reduced mod 2, and the KD-tree box (period 2 on those
    coordinates, effectively none on the rest)."""
    x = (lat.numpy() + 1.0).copy()
    box = np.full(x.shape[1], 1e3)
    for j in per:
        box[j] = 2.0
        x[:, j] = np.mod(x[:, j], 2.0)
    return np.clip(x, 0.0, None), box


def greedy_dedupe(x, box, tree, order, n_anchor, r_anchor, r_flood):
    """Indices kept by a greedy pass in `order` (anchors are rows < n_anchor): a row is dropped when a kept row lies
    within its radius of it -- r_anchor between two anchors, r_flood for any pair with a flood state. Distances are
    Euclidean with the periodic coordinates wrapped. Returns (kept indices, close pairs)."""
    pairs = tree.query_pairs(max(r_anchor, r_flood), output_type='ndarray')
    if len(pairs):
        d = np.abs(x[pairs[:, 0]] - x[pairs[:, 1]])
        d = np.minimum(d, box - d)
        dist = np.sqrt((d * d).sum(1))
        both = (pairs[:, 0] < n_anchor) & (pairs[:, 1] < n_anchor)
        pairs = pairs[dist <= np.where(both, r_anchor, r_flood)]
    nbr = [[] for _ in range(len(x))]
    for i, j in pairs:
        nbr[i].append(j)
        nbr[j].append(i)
    state = np.zeros(len(x), dtype=np.int8)  # 0 undecided, 1 kept, -1 dropped
    for i in order:
        if state[i]:
            continue
        state[i] = 1
        for j in nbr[i]:
            if state[j] == 0:
                state[j] = -1
    return np.nonzero(state == 1)[0], len(pairs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('anchors')
    ap.add_argument('flood_dir')
    ap.add_argument('out')
    ap.add_argument('--sg', type=int, required=True)
    ap.add_argument('--key', required=True, choices=['elj', 'uma'])
    ap.add_argument('--mol', required=True)
    ap.add_argument('--dedupe', type=float, default=0.01)
    ap.add_argument('--no-fold', action='store_true')
    ap.add_argument('--target-rows', type=int, default=0,
                    help='widen the flood radius until the file holds at most this many rows (0: no limit)')
    ap.add_argument('--check', type=int, default=2000)
    ap.add_argument('--mlip_path', default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    src = torch.load(a.anchors, weights_only=False)
    anchors = src['prior']
    tsf = float(src.get('thermal_scaling_factor', 1.0))
    scale = tsf if a.key == 'elj' else 1.0
    eA = anchors[a.key].double().flatten()
    cpA = anchors.full_cell_parameters().float()
    cpF, eF, n_files = load_flood(a.flood_dir)
    eF = eF / scale
    cp = torch.cat([cpA, cpF])
    E = torch.cat([eA, eF])
    is_anchor = torch.zeros(len(cp), dtype=torch.bool)
    is_anchor[:len(cpA)] = True
    lat = latents_of(anchors, cp)
    per = periodic_dims(a.sg)
    order = np.concatenate([np.argsort(eA.numpy(), kind='stable'),
                            len(cpA) + np.argsort(eF.numpy(), kind='stable')])
    from scipy.spatial import cKDTree
    x, box = wrapped(lat, per)
    tree = cKDTree(x, boxsize=box)
    n_img = 1 if a.no_fold else (4 if a.sg == 2 else 8)
    r_flood = a.dedupe
    while True:
        kept, n_pairs = greedy_dedupe(x, box, tree, order, len(cpA), a.dedupe, r_flood)
        ka = is_anchor[kept]
        print(f'latent de-dupe (periodic dims {per}): anchors at {a.dedupe}, flood states at {r_flood:.4f} '
              f'({n_pairs} close pairs): kept {int(ka.sum())} of {len(cpA)} anchors and {int((~ka).sum())} of '
              f'{len(cpF)} flood states ({n_files} shard files) -> about {len(kept) * n_img} rows  '
              f'({time.time() - t0:.0f} s)', flush=True)
        if not a.target_rows or len(kept) * n_img <= a.target_rows or int((~ka).sum()) == 0:
            break
        r_flood *= 1.25
    kept = torch.from_numpy(np.sort(kept))
    ch = Chart(a.mol, a.sg, a.key)
    if a.key == 'uma':
        from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
        ch.device, ch.predictor = 'cuda', init_uma_crystal_predictor(a.mlip_path, device='cuda')
    base = rows_on(anchors, cp[kept])
    ch.setup(base)
    if a.no_fold:
        out, source, image = base, torch.arange(len(kept)), torch.zeros(len(kept), dtype=torch.long)
    else:
        parts, srcs, iids = [], [], []
        for lo in range(0, len(kept), 20000):
            b, s, i = ch.images(base.subsample_new_batch(torch.arange(lo, min(lo + 20000, len(kept)))))
            parts.append(b); srcs.append(s + lo); iids.append(i)
        out = parts[0]
        for p in parts[1:]:
            out = out.append_batch(p)
        source, image = torch.cat(srcs), torch.cat(iids)
    e_out = E[kept][source]
    setattr(out, a.key, e_out.float())
    hand = set(out.aunit_handedness.flatten().tolist())
    assert hand == {1.0}, hand
    lat_out = out.latent_params()
    print(f'rows: {out.num_graphs} ({out.num_graphs / len(kept):.2f} per kept row); energy {float(e_out.min()):.3f} .. '
          f'{float(e_out.max()):.3f}; latent range {float(lat_out.min()):.4f} .. {float(lat_out.max()):.4f}', flush=True)
    if a.check:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(out.num_graphs, generator=g)[:a.check]
        e, red = ch.read(out.subsample_new_batch(idx))
        d = (e - e_out[idx]).abs()
        kT = 2.494 / scale
        print(f'check on {len(idx)} rows: trainer-read energy vs stored: median |d| {float(d.median()):.4f}, max '
              f'{float(d.max()):.4f} ({float(d.max()) / kT:.3f} kT); reduction penalty max {float(red.max()):.3g}', flush=True)
        bad = torch.nonzero(d > 0.2 * kT).flatten()
        if len(bad):
            for j in bad[:12].tolist():
                r = int(idx[j])
                print(f'  row {r}: anchor {bool(is_anchor[kept][source][r])}, image {int(image[r])}, stored '
                      f'{float(e_out[r]):.3f}, read {float(e[j]):.3f}, reduction penalty {float(red[j]):.3g}, '
                      f'centre {[round(float(v), 4) for v in out.aunit_centroid[r]]}')
            raise SystemExit(f'refused: {len(bad)} of {len(idx)} checked rows do not read back at their stored energy')
    blob = {'prior': out, 'equalized_prior': out, 'thermal_scaling_factor': src.get('thermal_scaling_factor', 1),
            'source_row': kept[source], 'image_id': image, 'is_anchor': is_anchor[kept][source],
            'n_anchors_in': len(cpA), 'n_flood_in': len(cpF), 'dedupe': a.dedupe, 'flood_radius': r_flood,
            'folded': not a.no_fold,
            'anchors_file': os.path.abspath(a.anchors), 'flood_dir': os.path.abspath(a.flood_dir)}
    if 'uma_energy_state' in src:
        blob['uma_energy_state'] = src['uma_energy_state']
    torch.save(blob, a.out)
    print(f'wrote {a.out} ({os.path.getsize(a.out) / 2 ** 20:.0f} MiB, {time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
