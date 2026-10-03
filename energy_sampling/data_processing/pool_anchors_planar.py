"""Anchors of a crystal-search registry in the trainer's chart, for a PLANAR molecule in space group 14 (acridine).

    python -m data_processing.pool_anchors_planar REG_DIR OUT.pt --mol MOL.pt --mlip_path MODEL [--key mace]
        [--window 5] [--device cuda] [--vram F] [--sample N]

`pool_anchors.Chart` (the MIP/NEH pooled-prior tooling) is used unchanged; what a planar molecule adds is that both
of its sg 14 special cases are exact instead of approximate or dropped:

  * a handedness -1 row: Chart.to_trainer fits the +1 molecule onto the row's mirrored molecule by a proper rotation.
    A mirror image of a planar body is a rotation of it, so the row describes the same crystal (acridine: fit RMSD
    2.5e-4 A, energy change <= 0.005 kT on 37 basins, 2026-10-03).
  * a row whose +1 centre has y mod 1/2 in (1/4, 1/2): no proper operation of P2_1/c brings it into the box and
    Chart.to_trainer reports it as outside. Inversion through the centroid of a planar body is a half turn about its
    normal, so the inversion mate is again the +1 molecule: the row is re-described through it (Chart.describe on the
    negated fractional atoms) and then brought into the box. Energy change 0.001 kT on 12 rows.

Every row is then read the trainer's way (analyze ['reduction_en', KEY], std_orientation False) and compared with the
registry's stored energy; a row off by more than 0.15 kT, with a reduction penalty, or non-finite is left out, and the
export is refused when more than 2% are. Writes the prior layout of pool_anchors export: {'prior': batch,
'equalized_prior': the same batch, 'thermal_scaling_factor': 1, 'basin', 'embedded' (handedness -1 rows),
'via_inversion', 'registry'}, the energy under KEY.
"""
import argparse
import os
import time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('reg_dir')
    ap.add_argument('out')
    ap.add_argument('--mol', required=True)
    ap.add_argument('--mlip_path', required=True)
    ap.add_argument('--key', default='mace', choices=['mace'])
    ap.add_argument('--window', type=float, default=5.0)
    ap.add_argument('--sample', type=int, default=0, help='0 = every basin in the window; N: the N/2 lowest and N/2 at random')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--vram', type=float, default=None, help='cap this process to that fraction of the GPU memory')
    ap.add_argument('--chunk', type=int, default=None)
    a = ap.parse_args(argv)
    if a.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import torch

    from data_processing.pool_anchors import RED_TOL, Chart, basin_rows
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    if a.device == 'cuda' and a.vram:
        torch.cuda.set_per_process_memory_fraction(a.vram, 0)
    t = time.time()
    reg, cfg, keep, P, H, E = basin_rows(a.reg_dir, a.window)
    if a.sample and a.sample < len(keep):
        g = np.random.default_rng(0)
        sel = np.sort(np.concatenate([np.arange(a.sample // 2), g.choice(np.arange(a.sample // 2, len(keep)),
                                                                         a.sample - a.sample // 2, replace=False)]))
        sel = torch.as_tensor(sel)
        keep, P, H, E = keep[sel], P[sel], H[sel], E[sel]
    ch = Chart(a.mol, 14, a.key, a.device, load_mace_model(a.mlip_path, device=a.device, dtype=torch.float32))
    tb, emb, in_box, rms = ch.to_trainer(P, H)
    via_inv = ~in_box
    if bool(via_inv.any()):
        alt, rms2 = ch.describe(tb, -ch.frac_atoms(tb))
        alt, ok2 = ch.into_box(alt)
        if not bool(ok2[via_inv].all()):
            raise RuntimeError('an inversion mate outside the box')
        idx = torch.nonzero(via_inv).flatten()
        for k in ('aunit_centroid', 'aunit_orientation'):
            v = getattr(tb, k).clone()
            v[idx] = getattr(alt, k)[idx]
            setattr(tb, k, v)
        rms = torch.where(via_inv, torch.maximum(rms, rms2), rms)
    print(f'{a.reg_dir}: {len(keep)} basins within {a.window} kT of {float(E.min()):.4f} (kT {cfg.kT:.4g}); handedness -1 '
          f'rows {int((H < 0).sum())}; rows described through the inversion mate {int(via_inv.sum())}; fit RMSD max '
          f'{float(rms.max()):.2e} A', flush=True)
    if float(rms.max()) > 5e-3:
        raise SystemExit(f'the molecule is not planar to the precision this export assumes (fit RMSD {float(rms.max()):.3g} A)')
    e, red = ch.read(tb, chunk=a.chunk or (32 if a.device == 'cuda' else 8))  # ~0.6 GB of GPU memory per row
    d = (e - E).abs() / cfg.kT
    print('| rows | count | median abs(trainer-read - stored) (kT) | 99th percentile (kT) | max (kT) | reduction penalty max |')
    print('|---|---|---|---|---|---|')
    for name, m in (('handedness +1, in box', ~emb & ~via_inv), ('handedness -1 (fitted)', emb & ~via_inv),
                    ('through the inversion mate', via_inv), ('all', torch.ones_like(emb))):
        if bool(m.any()):
            print(f'| {name} | {int(m.sum())} | {float(d[m].median()):.4f} | {float(d[m].quantile(0.99)):.4f} | '
                  f'{float(d[m].max()):.4f} | {float(red[m].max()):.2e} |')
    bad = (d > 0.15) | (red > RED_TOL) | ~torch.isfinite(e)
    print(f'left out: {int(bad.sum())} of {len(bad)} rows (energy off by > 0.15 kT {int((d > 0.15).sum())}, reduction '
          f'penalty {int((red > RED_TOL).sum())}, non-finite {int((~torch.isfinite(e)).sum())})', flush=True)
    if float(bad.double().mean()) > 0.02:
        raise SystemExit(f'export refused: {int(bad.sum())} of {len(bad)} rows fail the read-back checks')
    good = torch.nonzero(~bad).flatten()
    tb = tb.subsample_new_batch(good)
    setattr(tb, a.key, e[good].float())
    for w in (1.0, 2.0, 3.0, 5.0):
        print(f'  anchors within {w:g} kT: {int((e[good] <= float(e[good].min()) + w * cfg.kT).sum())}')
    torch.save({'prior': tb, 'equalized_prior': tb, 'thermal_scaling_factor': 1, 'basin': keep[good],
                'embedded': emb[good], 'via_inversion': via_inv[good], 'registry': os.path.abspath(a.reg_dir)}, a.out)
    print(f'wrote {tb.num_graphs} rows to {a.out} ({time.time() - t:.0f} s)')


if __name__ == '__main__':
    main()
