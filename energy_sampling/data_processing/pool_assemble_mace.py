"""Assemble a shipped prior file from pooled anchors and their capped-MC walk, MACE energy (acridine; pool_acr_oct03).

    python -m data_processing.pool_assemble_mace ANCHORS.pt FLOOD_DIR OUT.pt --sg 14 --mol MOL.pt --mlip_path MODEL
        [--dedupe 0.01] [--no-fold] [--target-rows N] [--check 2000] [--device cuda]

The steps are those of data_processing.pool_assemble (whose helpers are used unchanged): candidates = the anchors, then
every walk state; greedy de-dupe in the trainer's 12-D latent (anchors first, each group in ascending energy; with
--target-rows the walk states' radius is widened x1.25 at a time until rows x images fit); every kept row written with
its normaliser images the +1 chart holds (sg 14: eight); a random --check rows re-scored the trainer's way against
their stored energy. What differs:

  * the energy is the MACE lattice energy under --mlip_path (key 'mace', kJ/mol per molecule, no scaling factor);
  * anchors the trainer's latent cannot represent are left out first (counted): an anchor whose cell lengths or angles
    change under latent_params -> latent_to_cell_params lies outside the latent box, and the trainer would read a
    different crystal from its row (acridine: 64 of 10,986 anchors were outside capped_mc's clamp box). Walk states
    are inside the box by construction; the written rows are checked the same way.
  * --pc-min F: anchors and walk states with a packing coefficient below F are left out before the de-dupe (owner
    2026-10-05: filter on density). Under acr_newmodel the walk's 15 kT ceiling lies above the whole binding energy
    (13.1 kT), so walkers drift into expanded, unbound cells: 55% of acridine's walk states have a packing coefficient
    below 0.55, where the MIPCAS and NEHZOR walks never go.
  * no relabelling of the molecule's own symmetry (owner 2026-10-05: the model conditions on one labelled conformer,
    so the C2-relabelled copies are not added).
"""
import argparse
import os
import time

import numpy as np
import torch

from data_processing.pool_anchors import Chart
from data_processing.pool_assemble import greedy_dedupe, latents_of, load_flood, periodic_dims, rows_on, wrapped

KEY = 'mace'
KT = 2.494


def representable(batch, rtol=1e-3, atol=1e-3):
    """[n] bool: the latent round trip returns the row's cell lengths (relative rtol) and angles (absolute atol)."""
    b = batch.clone()
    before = b.full_cell_parameters().detach().double().clone()
    b.latent_to_cell_params(b.latent_params())
    after = b.full_cell_parameters().detach().double()
    return (((after[:, :3] - before[:, :3]).abs() / before[:, :3]).amax(1) < rtol) & \
        ((after[:, 3:6] - before[:, 3:6]).abs().amax(1) < atol)


def packing_of(template, cp, chunk=20000):
    """[n] packing coefficient of rows with the given cell parameters on the template's molecule."""
    out = []
    for lo in range(0, len(cp), chunk):
        out.append(rows_on(template, cp[lo:lo + chunk]).packing_coeff.double().flatten())
    return torch.cat(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('anchors')
    ap.add_argument('flood_dir')
    ap.add_argument('out')
    ap.add_argument('--sg', type=int, required=True)
    ap.add_argument('--mol', required=True)
    ap.add_argument('--mlip_path', required=True)
    ap.add_argument('--dedupe', type=float, default=0.01)
    ap.add_argument('--no-fold', action='store_true')
    ap.add_argument('--target-rows', type=int, default=0,
                    help="widen the walk states' radius until the file holds at most this many rows (0: no limit)")
    ap.add_argument('--check', type=int, default=2000)
    ap.add_argument('--pc-min', type=float, default=0.0,
                    help='leave out anchors and walk states with a packing coefficient below this (0: no floor)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--vram', type=float, default=None)
    ap.add_argument('--chunk', type=int, default=32)
    a = ap.parse_args(argv)
    if a.device == 'cuda' and a.vram:
        torch.cuda.set_per_process_memory_fraction(a.vram, 0)
    t0 = time.time()
    src = torch.load(a.anchors, weights_only=False)
    anchors = src['prior']
    ok = representable(anchors)
    print(f'anchors: {anchors.num_graphs}; outside the latent box (left out): {int((~ok).sum())}', flush=True)
    if float((~ok).double().mean()) > 0.05:
        raise SystemExit(f'refused: {int((~ok).sum())} of {len(ok)} anchors are outside the latent box')
    anchors = anchors.subsample_new_batch(torch.nonzero(ok).flatten())
    eA = anchors[KEY].double().flatten()
    cpA = anchors.full_cell_parameters().float()
    cpF, eF, n_files = load_flood(a.flood_dir)
    if a.pc_min > 0:
        pa, pf = anchors.packing_coeff.double().flatten(), packing_of(anchors, cpF)
        ka, kf = pa >= a.pc_min, pf >= a.pc_min
        print(f'packing-coefficient floor {a.pc_min}: kept {int(ka.sum())} of {len(ka)} anchors and {int(kf.sum())} of '
              f'{len(kf)} walk states', flush=True)
        anchors = anchors.subsample_new_batch(torch.nonzero(ka).flatten())
        eA, cpA = eA[ka], cpA[ka]
        cpF, eF = cpF[kf], eF[kf]
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
    n_img = 1 if a.no_fold else (4 if a.sg == 2 else 8)
    r_flood = a.dedupe
    while True:
        kept, n_pairs = greedy_dedupe(x, box, tree, order, len(cpA), a.dedupe, r_flood)
        ka = is_anchor[kept]
        print(f'latent de-dupe (periodic dims {per}): anchors at {a.dedupe}, walk states at {r_flood:.4f} ({n_pairs} '
              f'close pairs): kept {int(ka.sum())} of {len(cpA)} anchors and {int((~ka).sum())} of {len(cpF)} walk '
              f'states ({n_files} shard files) -> about {len(kept) * n_img} rows  ({time.time() - t0:.0f} s)', flush=True)
        if not a.target_rows or len(kept) * n_img <= a.target_rows or int((~ka).sum()) == 0:
            break
        r_flood *= 1.25
    kept = torch.from_numpy(np.sort(kept))
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    ch = Chart(a.mol, a.sg, KEY, a.device, load_mace_model(a.mlip_path, device=a.device, dtype=torch.float32))
    base = rows_on(anchors, cp[kept])
    ch.setup(base)
    if a.no_fold:
        out, source, image = base, torch.arange(len(kept)), torch.zeros(len(kept), dtype=torch.long)
    else:
        parts, srcs, iids = [], [], []
        for lo in range(0, len(kept), 20000):
            b, s, i = ch.images(base.subsample_new_batch(torch.arange(lo, min(lo + 20000, len(kept)))))
            parts.append(b)
            srcs.append(s + lo)
            iids.append(i)
        out = parts[0]
        for p in parts[1:]:
            out = out.append_batch(p)
        source, image = torch.cat(srcs), torch.cat(iids)
    e_out = E[kept][source]
    setattr(out, KEY, e_out.float())
    hand = set(out.aunit_handedness.flatten().tolist())
    assert hand == {1.0}, hand
    rep = representable(out)
    lat_out = out.latent_params()
    print(f'rows: {out.num_graphs} ({out.num_graphs / len(kept):.2f} per kept row); energy {float(e_out.min()):.3f} .. '
          f'{float(e_out.max()):.3f} kJ/mol ({(float(e_out.max()) - float(e_out.min())) / KT:.1f} kT); latent range '
          f'{float(lat_out.min()):.4f} .. {float(lat_out.max()):.4f}; rows outside the latent box {int((~rep).sum())}',
          flush=True)
    if float((~rep).double().mean()) > 0.001:
        raise SystemExit(f'refused: {int((~rep).sum())} of {len(rep)} written rows are outside the latent box')
    if a.check:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(out.num_graphs, generator=g)[:a.check]
        e, red = ch.read(out.subsample_new_batch(idx), chunk=a.chunk)
        d = (e - e_out[idx]).abs()
        print(f'check on {len(idx)} rows: trainer-read energy vs stored: median |d| {float(d.median()):.4f}, max '
              f'{float(d.max()):.4f} kJ/mol ({float(d.max()) / KT:.3f} kT); reduction penalty max {float(red.max()):.3g}',
              flush=True)
        bad = torch.nonzero(d > 0.2 * KT).flatten()
        if len(bad):
            for j in bad[:12].tolist():
                r = int(idx[j])
                print(f'  row {r}: anchor {bool(is_anchor[kept][source][r])}, image {int(image[r])}, stored '
                      f'{float(e_out[r]):.3f}, read {float(e[j]):.3f}, reduction penalty {float(red[j]):.3g}')
            raise SystemExit(f'refused: {len(bad)} of {len(idx)} checked rows do not read back at their stored energy')
    anc = is_anchor[kept][source]
    for w in (1.0, 2.0, 5.0, 10.0, 15.0):
        m = e_out <= float(e_out.min()) + w * KT
        print(f'  rows within {w:g} kT: {int(m.sum())} (anchors {int((m & anc).sum())}, walk states {int((m & ~anc).sum())})')
    blob = {'prior': out, 'equalized_prior': out, 'thermal_scaling_factor': 1, 'source_row': kept[source],
            'image_id': image, 'is_anchor': anc, 'n_anchors_in': len(cpA), 'n_anchors_outside_box': int((~ok).sum()),
            'n_flood_in': len(cpF), 'pc_min': a.pc_min, 'dedupe': a.dedupe, 'flood_radius': r_flood, 'folded': not a.no_fold,
            'anchors_file': os.path.abspath(a.anchors), 'flood_dir': os.path.abspath(a.flood_dir)}
    torch.save(blob, a.out)
    print(f'wrote {a.out} ({os.path.getsize(a.out) / 2 ** 20:.0f} MiB, {time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
