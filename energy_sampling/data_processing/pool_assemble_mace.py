"""Assemble a shipped prior file from pooled anchors and their capped-MC walk, MACE energy (acridine; pool_acr_oct03).

    python -m data_processing.pool_assemble_mace ANCHORS.pt FLOOD_DIR OUT.pt --sg 14 --mol MOL.pt --mlip_path MODEL
        --walk-radius R [--e-max-kT X] [--dedupe 0.01] [--no-fold] [--check 2000] [--keep-penalised] [--device cuda]

The steps are those of data_processing.pool_assemble (whose helpers are used unchanged): candidates = the anchors, then
every walk state; greedy de-dupe in the trainer's 12-D latent (anchors first, each group in ascending energy: anchors
at --dedupe from one another, any pair with a walk state at --walk-radius); every kept state written
with its normaliser images the +1 chart holds (sg 14: eight); a random --check rows re-scored the trainer's way against
their stored energy; cached space-group lookup fields dropped from the rows (pool_assemble.LAZY_CACHES: a stored copy
breaks key parity with the trainer's batches). What differs:

  * the energy is the MACE lattice energy under --mlip_path (key 'mace', kJ/mol per molecule, no scaling factor);
  * LATENT BOX. A state the trainer's latent cannot represent (any of its 12 cell parameters changes by more than 1e-3
    on a pass latent_params -> latent_to_cell_params, pool_assemble's test) is left out: anchors before the de-dupe,
    and after the fold any state one of whose images fails, with all its images.
  * DENSITY, BEFORE the de-dupe. Anchors and walk states the trainer's density penalty touches
    (energies.molecular_crystal.density_penalty > 0) are left out (owner 2026-10-05). pool_assemble does this after
    its de-dupe, which is equivalent when few states are affected. Here most are: under acr_newmodel the walk's 15 kT
    ceiling is above the whole binding energy (13.1 kT), walkers drift into expanded cells, and 55% of the walk states
    carry a penalty; thinning first spent the row budget on them (walk radius 0.284) and then would discard 90% of
    what it had kept. --keep-penalised turns the filter off.
  * THE WALK RADIUS IS PHYSICAL, NOT A ROW BUDGET (owner 2026-10-05). --walk-radius is a fixed latent distance, for
    acridine the 1 kT kick (0.031: the latent step that raises a relaxed basin's energy by a median 1 kT under
    acr_newmodel); the file is as large as that makes it. pool_assemble's --target-rows widened the radius until the
    file fit 400,000 rows, which put acridine's at 0.146, five kicks, and deleted nearly every walk state near a low
    anchor. The size is controlled by --e-max-kT (anchors and walk states more than X kT above the lowest are left
    out) or by choosing another radius; --target-rows remains only as an explicit override.
  * no relabelling of the molecule's own symmetry (owner 2026-10-05: the model conditions on one labelled conformer,
    so the C2-relabelled copies are not added).

The file records what it was built from by name and size, never by path: the anchors and model file names and sizes,
the molecule file name, the counts left out at each step, the re-scoring check, and the repository commit when git can
report it.
"""
import argparse
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


def packing_of(template, cp, chunk=20000):
    """[n] packing coefficient of rows with the given cell parameters on the template's molecule."""
    out = []
    for lo in range(0, len(cp), chunk):
        out.append(rows_on(template, cp[lo:lo + chunk]).packing_coeff.double().flatten())
    return torch.cat(out)


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
    ap.add_argument('--dedupe', type=float, default=0.01)
    ap.add_argument('--no-fold', action='store_true')
    ap.add_argument('--walk-radius', type=float, required=True,
                    help='latent distance below which a walk state duplicates a kept state: a physical scale (the '
                         "system's 1 kT kick), not a row budget")
    ap.add_argument('--e-max-kT', type=float, default=None,
                    help='leave out anchors and walk states more than this many kT (2.494 kJ/mol) above the lowest')
    ap.add_argument('--target-rows', type=int, default=0,
                    help='override: widen the walk radius x1.25 at a time until the file holds at most this many rows')
    ap.add_argument('--check', type=int, default=2000)
    ap.add_argument('--keep-penalised', action='store_true',
                    help="keep anchors and walk states the trainer's density penalty touches")
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--vram', type=float, default=None)
    ap.add_argument('--chunk', type=int, default=32)
    a = ap.parse_args(argv)
    if a.device == 'cuda' and a.vram:
        torch.cuda.set_per_process_memory_fraction(a.vram, 0)
    t0 = time.time()
    src = torch.load(a.anchors, weights_only=False)
    anchors = src['prior']
    n_anchors_in = anchors.num_graphs
    ok = representable(anchors)
    n_box_anchors = int((~ok).sum())
    print(f'anchors: {n_anchors_in}; outside the latent box (left out): {n_box_anchors}', flush=True)
    if float((~ok).double().mean()) > 0.05:
        raise SystemExit(f'refused: {n_box_anchors} of {len(ok)} anchors are outside the latent box')
    anchors = anchors.subsample_new_batch(torch.nonzero(ok).flatten())
    eA = anchors[KEY].double().flatten()
    cpA = anchors.full_cell_parameters().float()
    cpF, eF, n_files = load_flood(a.flood_dir)
    n_flood_in = len(cpF)
    n_dens_anchors = n_dens_walk = 0
    if not a.keep_penalised:
        from energies.molecular_crystal import density_penalty
        pa, pf = anchors.packing_coeff.double().flatten(), packing_of(anchors, cpF)
        ka, kf = density_penalty(pa) <= 0, density_penalty(pf) <= 0
        n_dens_anchors, n_dens_walk = int((~ka).sum()), int((~kf).sum())
        print(f"density: {n_dens_anchors} of {len(ka)} anchors and {n_dens_walk} of {len(kf)} walk states carry the "
              f"trainer's density penalty and are left out; packing coefficient kept {float(pa[ka].min()):.3f} .. "
              f"{float(max(pa[ka].max(), pf[kf].max())):.3f}", flush=True)
        anchors = anchors.subsample_new_batch(torch.nonzero(ka).flatten())
        eA, cpA = eA[ka], cpA[ka]
        cpF, eF = cpF[kf], eF[kf]
    n_emax_anchors = n_emax_walk = 0
    if a.e_max_kT is not None:
        top = float(min(eA.min(), eF.min())) + a.e_max_kT * KT
        ka, kf = eA <= top, eF <= top
        n_emax_anchors, n_emax_walk = int((~ka).sum()), int((~kf).sum())
        print(f'energy ceiling {a.e_max_kT} kT above the lowest ({top:.3f} kJ/mol): {n_emax_anchors} anchors and '
              f'{n_emax_walk} of {len(kf)} walk states are above it and are left out', flush=True)
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
    r_flood = a.walk_radius
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
    n_box_states = 0
    if not bool(rep.all()):  # an image at the edge of the box: its state goes, with every image
        bad_states = torch.unique(source[~rep])
        n_box_states = len(bad_states)
        print(f'latent box after the fold: {int((~rep).sum())} rows of {n_box_states} states fail; those states are '
              f'left out with all their images ({int(is_anchor[kept][bad_states].sum())} of them anchors)', flush=True)
        keep_state = torch.ones(len(kept), dtype=torch.bool)
        keep_state[bad_states] = False
        kept = kept[keep_state]
        out, source, image = fold(kept)
        rep = representable(out)
        if not bool(rep.all()):
            raise SystemExit(f'refused: {int((~rep).sum())} rows are still outside the latent box')
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
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10), (10, None)):
        m = first & (e_out >= e_min + lo * KT)
        if hi is not None:
            m = m & (e_out < e_min + hi * KT)
        print(f'  {lo} kT to {"the ceiling" if hi is None else str(hi) + " kT"}: {int((m & anc).sum())} anchors, '
              f'{int((m & ~anc).sum())} walk states')
    for k in LAZY_CACHES:  # last, since any analysis rebuilds them
        if k in out.keys():
            delattr(out, k)
    blob = {'prior': out, 'equalized_prior': out, 'thermal_scaling_factor': 1, 'source_row': kept[source],
            'image_id': image, 'is_anchor': anc, 'dedupe': a.dedupe, 'flood_radius': r_flood, 'walk_radius_asked': a.walk_radius,
            'e_max_kT': a.e_max_kT, 'target_rows': a.target_rows, 'folded': not a.no_fold,
            'n_anchors_in': n_anchors_in, 'n_flood_in': n_flood_in, 'n_shard_files': n_files,
            'anchors_outside_box_dropped': n_box_anchors, 'states_outside_box_after_fold_dropped': n_box_states,
            'anchors_density_dropped': n_dens_anchors, 'walk_states_density_dropped': n_dens_walk,
            'anchors_above_ceiling_dropped': n_emax_anchors, 'walk_states_above_ceiling_dropped': n_emax_walk,
            'density_filter': not a.keep_penalised, 'check': check,
            'anchors_file': (os.path.basename(a.anchors), os.path.getsize(a.anchors)),
            'mlip_file': (os.path.basename(a.mlip_path), os.path.getsize(a.mlip_path)),
            'mol_file': os.path.basename(a.mol), 'commit': _commit()}
    torch.save(blob, a.out)
    print(f'wrote {a.out} ({os.path.getsize(a.out)} bytes, {out.num_graphs} rows, {time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
