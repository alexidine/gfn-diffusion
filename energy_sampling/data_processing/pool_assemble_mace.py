"""Assemble a shipped prior file from pooled anchors and their capped-MC walk, MACE energy (acridine; pool_acr_oct03).

    python -m data_processing.pool_assemble_mace ANCHORS.pt FLOOD_DIR OUT.pt --sg 14 --mol MOL.pt --mlip_path MODEL
        --radius R [--ceiling_kT C] [--no-fold] [--check 2000] [--device cuda]

This is data_processing.pool_assemble (GFN 54ea7e36: one physical radius per system under an energy ceiling, no row
budget) for a MACE energy, with pool_assemble's helpers used unchanged and its steps in its order:

  1. candidates = the anchors, then every walk state, as rows on the anchors' molecule at handedness +1.
  2. ceiling (--ceiling_kT C): a candidate more than C kT (2.494 kJ/mol) above the lowest anchor is left out. Then thin
     in the trainer's latent: greedy, anchors first in ascending energy, then walk states in ascending energy; a row
     within --radius of a kept row is dropped. One radius for every pair, no row budget.
  3. a kept state outside the trainer latent box (a cell parameter changes by more than 1e-3 on a pass through the
     latent and back) is left out, and so is a state the trainer's density penalty touches.
  4. fold: every kept state written with the 8 normaliser images the +1 chart holds, images carrying its energy.
  5. cached space-group lookup fields dropped; --check rows re-scored the trainer's way against their stored energy;
     OUT.summary.json written with pool_assemble's keys.

What differs from pool_assemble, and why:

  * the energy is the MACE lattice energy under --mlip_path (key 'mace', kJ/mol per molecule, no scaling factor);
  * DENSITY AND THE LATENT BOX ARE ALSO APPLIED BEFORE THE THINNING. pool_assemble filters only the kept states, which
    is the same thing when few states are affected. Here most are: under acr_newmodel the walk's 15 kT ceiling is
    above the whole binding energy (13.1 kT), walkers drift into expanded cells, and 55% of the walk states carry the
    density penalty. A penalised state that is kept by the thinning deletes every unpenalised state within the radius
    and is then removed itself, leaving a hole. Filtering the candidates first avoids that; step 3 still runs on the
    kept states and should find nothing.
  * the latent-box test is repeated on every image after the fold; a state one of whose images fails leaves with all
    its images.
  * no relabelling of the molecule's own symmetry (owner 2026-10-05: the model conditions on one labelled conformer).

The file records its inputs by name and size, never by path.
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

from data_processing.pool_anchors import Chart
from data_processing.pool_assemble import (LAZY_CACHES, greedy_dedupe, latents_of, load_flood, periodic_dims, rows_on,
                                           wrapped)

KEY = 'mace'
KT = 2.494
BOX_TOL = 1e-3  # pool_assemble's: the largest change of a cell parameter on a pass through the latent and back


def representable(batch, chunk=200000):
    """[n] bool: every one of the row's 12 cell parameters survives latent_params -> latent_to_cell_params."""
    out = []
    for lo in range(0, batch.num_graphs, chunk):
        rt = batch.subsample_new_batch(torch.arange(lo, min(lo + chunk, batch.num_graphs)))
        before = rt.full_cell_parameters().detach().double().clone()
        rt.latent_to_cell_params(rt.latent_params())
        out.append((rt.full_cell_parameters().detach().double() - before).abs().amax(1) <= BOX_TOL)
    return torch.cat(out)


def rows_ok(template, cp, chunk=200000):
    """([n] in the latent box, [n] free of the trainer's density penalty) for rows with the given cell parameters."""
    from energies.molecular_crystal import density_penalty
    box, dens = [], []
    for lo in range(0, len(cp), chunk):
        b = rows_on(template, cp[lo:lo + chunk])
        dens.append(density_penalty(b.packing_coeff.double().flatten()) <= 0)
        box.append(representable(b))
    return torch.cat(box), torch.cat(dens)


def _commit():
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001 -- provenance only
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('anchors')
    ap.add_argument('flood_dir')
    ap.add_argument('out')
    ap.add_argument('--sg', type=int, required=True)
    ap.add_argument('--mol', required=True)
    ap.add_argument('--mlip_path', required=True)
    ap.add_argument('--radius', type=float, required=True,
                    help='thinning radius in the trainer latent (Euclidean), for anchors and walk states alike')
    ap.add_argument('--ceiling_kT', type=float, default=None,
                    help='leave out candidates more than this many kT above the lowest anchor (default: no ceiling)')
    ap.add_argument('--no-fold', action='store_true')
    ap.add_argument('--check', type=int, default=2000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--vram', type=float, default=None)
    ap.add_argument('--chunk', type=int, default=32)
    a = ap.parse_args(argv)
    if a.device == 'cuda' and a.vram:
        torch.cuda.set_per_process_memory_fraction(a.vram, 0)
    t0 = time.time()
    src = torch.load(a.anchors, weights_only=False)
    anchors = src['prior']
    eA = anchors[KEY].double().flatten()
    cpA = anchors.full_cell_parameters().float()
    cpF, eF, n_files = load_flood(a.flood_dir)
    n_anchor_in, n_flood_in = len(cpA), len(cpF)
    if a.ceiling_kT is not None:
        top = float(eA.min()) + a.ceiling_kT * KT
        okA, okF = eA <= top, eF <= top
        print(f'ceiling {a.ceiling_kT:g} kT above the lowest anchor ({float(eA.min()):.3f} -> {top:.3f}): '
              f'{int(okA.sum())} of {len(eA)} anchors and {int(okF.sum())} of {len(eF)} walk states under it', flush=True)
        cpA, eA, cpF, eF = cpA[okA], eA[okA], cpF[okF], eF[okF]
    n_anchor_ceil, n_flood_ceil = len(cpA), len(cpF)
    # before the thinning (see the docstring): candidates outside the latent box or under the density penalty
    boxA, densA = rows_ok(anchors, cpA)
    boxF, densF = rows_ok(anchors, cpF)
    print(f'before thinning: outside the latent box {int((~boxA).sum())} anchors, {int((~boxF).sum())} walk states; '
          f'density-penalised {int((~densA).sum())} anchors, {int((~densF).sum())} of {len(densF)} walk states; left out',
          flush=True)
    pre = dict(anchors_outside_box=int((~boxA).sum()), flood_outside_box=int((~boxF).sum()),
               anchors_density=int((boxA & ~densA).sum()), flood_density=int((boxF & ~densF).sum()))
    cpA, eA, cpF, eF = cpA[boxA & densA], eA[boxA & densA], cpF[boxF & densF], eF[boxF & densF]
    cp = torch.cat([cpA, cpF])
    E = torch.cat([eA, eF])
    is_anchor = torch.zeros(len(cp), dtype=torch.bool)
    is_anchor[:len(cpA)] = True
    lat = latents_of(anchors, cp)
    per = periodic_dims(a.sg)
    order = np.concatenate([np.argsort(eA.numpy(), kind='stable'), len(cpA) + np.argsort(eF.numpy(), kind='stable')])
    from scipy.spatial import cKDTree
    x, box = wrapped(lat, per)
    tree = cKDTree(x, boxsize=box)
    kept, n_pairs = greedy_dedupe(x, box, tree, order, len(cpA), a.radius, a.radius)
    ka = is_anchor[kept]
    n_anchor_thin, n_flood_thin = int(ka.sum()), int((~ka).sum())
    print(f'latent thinning at radius {a.radius:g} (periodic dims {per}; {n_pairs} close pairs): kept {n_anchor_thin} '
          f'of {len(cpA)} anchors and {n_flood_thin} of {len(cpF)} walk states ({n_files} shard files)  '
          f'({time.time() - t0:.0f} s)', flush=True)
    kept = torch.from_numpy(np.sort(kept))
    # pool_assemble's filters on the kept states: nothing should be left for them here
    kbox, kdens = rows_ok(anchors, cp[kept])
    n_outside, n_density = int((~kbox).sum()), int((kbox & ~kdens).sum())
    if n_outside or n_density:
        print(f'kept states: {n_outside} outside the latent box, {n_density} density-penalised; left out', flush=True)
        kept = kept[kbox & kdens]
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    ch = Chart(a.mol, a.sg, KEY, a.device, load_mace_model(a.mlip_path, device=a.device, dtype=torch.float32))

    def fold(kept):
        base = rows_on(anchors, cp[kept])
        ch.setup(base)
        if a.no_fold:
            return base, torch.arange(len(kept)), torch.zeros(len(kept), dtype=torch.long)
        parts, srcs, iids = [], [], []
        for lo in range(0, len(kept), 20000):
            b, s, i = ch.images(base.subsample_new_batch(torch.arange(lo, min(lo + 20000, len(kept)))))
            parts.append(b)
            srcs.append(s + lo)
            iids.append(i)
        out = parts[0]
        for p in parts[1:]:
            out = out.append_batch(p)
        return out, torch.cat(srcs), torch.cat(iids)

    out, source, image = fold(kept)
    rep = representable(out)
    n_box_fold = 0
    if not bool(rep.all()):  # an image at the edge of the box: its state goes, with every image
        bad_states = torch.unique(source[~rep])
        n_box_fold = len(bad_states)
        print(f'latent box after the fold: {int((~rep).sum())} rows of {n_box_fold} states fail; those states are left '
              f'out with all their images ({int(is_anchor[kept][bad_states].sum())} of them anchors)', flush=True)
        keep_state = torch.ones(len(kept), dtype=torch.bool)
        keep_state[bad_states] = False
        kept = kept[keep_state]
        out, source, image = fold(kept)
        if not bool(representable(out).all()):
            raise SystemExit('refused: rows outside the latent box remain after the fold filter')
    e_out = E[kept][source].clone()
    setattr(out, KEY, e_out.float())
    hand = set(out.aunit_handedness.flatten().tolist())
    assert hand == {1.0}, hand
    lat_out = out.latent_params()
    print(f'rows: {out.num_graphs} ({out.num_graphs / len(kept):.2f} per kept state); energy {float(e_out.min()):.3f} .. '
          f'{float(e_out.max()):.3f} kJ/mol ({(float(e_out.max()) - float(e_out.min())) / KT:.1f} kT); latent range '
          f'{float(lat_out.min()):.4f} .. {float(lat_out.max()):.4f}', flush=True)
    check = None
    if a.check:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(out.num_graphs, generator=g)[:a.check]
        e, red = ch.read(out.subsample_new_batch(idx), chunk=a.chunk)
        d = (e - e_out[idx]).abs()
        check = dict(rows=len(idx), median_abs=float(d.median()), max_abs=float(d.max()), reduction_max=float(red.max()))
        print(f'check on {len(idx)} rows: trainer-read energy vs stored: median |d| {check["median_abs"]:.4f}, max '
              f'{check["max_abs"]:.4f} kJ/mol ({check["max_abs"] / KT:.3f} kT); reduction penalty max '
              f'{check["reduction_max"]:.3g}', flush=True)
        bad = torch.nonzero(d > 0.2 * KT).flatten()
        if len(bad):
            for j in bad[:12].tolist():
                r = int(idx[j])
                print(f'  row {r}: anchor {bool(is_anchor[kept][source][r])}, image {int(image[r])}, stored '
                      f'{float(e_out[r]):.3f}, read {float(e[j]):.3f}, reduction penalty {float(red[j]):.3g}')
            raise SystemExit(f'refused: {len(bad)} of {len(idx)} checked rows do not read back at their stored energy')
    anc = is_anchor[kept][source]
    print('Kept states (one per image set) by energy above the lowest row, kT = 2.494 kJ/mol: anchors, walk states.')
    first = image == image.min()
    e_min = float(e_out.min())
    bands = {}
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10), (10, None)):
        m = first & (e_out >= e_min + lo * KT)
        if hi is not None:
            m = m & (e_out < e_min + hi * KT)
        label = f'{lo} kT to ' + ('the ceiling' if hi is None else f'{hi} kT')
        bands[label] = (int((m & anc).sum()), int((m & ~anc).sum()))
        print(f'  {label}: {bands[label][0]} anchors, {bands[label][1]} walk states')
    for k in LAZY_CACHES:  # last, since any analysis rebuilds them
        if k in out.keys():
            delattr(out, k)
    blob = {'prior': out, 'equalized_prior': out, 'thermal_scaling_factor': 1, 'source_row': kept[source],
            'image_id': image, 'is_anchor': anc, 'n_anchors_in': n_anchor_in, 'n_flood_in': n_flood_in,
            'radius': a.radius, 'ceiling_kT': a.ceiling_kT, 'folded': not a.no_fold,
            'out_of_box_states_dropped': n_outside + n_box_fold, 'density_states_dropped': n_density,
            'dropped_before_thinning': pre, 'check': check,
            'anchors_file': (os.path.basename(a.anchors), os.path.getsize(a.anchors)),
            'mlip_file': (os.path.basename(a.mlip_path), os.path.getsize(a.mlip_path)),
            'mol_file': os.path.basename(a.mol), 'commit': _commit()}
    torch.save(blob, a.out)
    summary = {'file': os.path.basename(a.out), 'bytes': os.path.getsize(a.out), 'rows': int(out.num_graphs),
               'radius': a.radius, 'ceiling_kT': a.ceiling_kT, 'folded': not a.no_fold,
               'anchors_in': n_anchor_in, 'flood_states_in': n_flood_in,
               'anchors_under_ceiling': n_anchor_ceil, 'flood_states_under_ceiling': n_flood_ceil,
               'dropped_before_thinning': pre,
               'anchors_after_thinning': n_anchor_thin, 'flood_states_after_thinning': n_flood_thin,
               'states_outside_latent_box': n_outside, 'states_with_density_penalty': n_density,
               'states_outside_latent_box_after_fold': n_box_fold,
               'states_kept': len(kept), 'anchor_rows': int(anc.sum()),
               'energy_min': float(e_out.min()), 'energy_max': float(e_out.max()), 'energy_key': KEY,
               'kept_states_by_band_anchors_walk': bands, 'check': check, 'commit': blob['commit']}
    with open(a.out + '.summary.json', 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(f'wrote {a.out} ({summary["bytes"]} bytes, {summary["rows"]} rows, {time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
