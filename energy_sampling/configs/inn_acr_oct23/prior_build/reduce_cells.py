"""Re-describe the rows of an old-style prior file whose cell is outside the trainer's reduced-cell domain, in their
reduced cell: the same crystal, the cell the reduction walls accept.

    python reduce_cells.py PRIOR.pt OUT.pt --mlip_path MODEL [--mol MOL.pt] [--device cuda] [--chunk 8]

An old prior predates MXtalTools' monoclinic class walls (sym_utils.mono_reduction_penalty): in
acridine_sg14_zp1_mace_prior_dataset_w3.pt 29% of the anchors carry a reduction penalty, which the trainer scores at
reduction_coeff (1000 in the current recipes), and pool_anchors' export would leave them out (RED_TOL). Here each such
row ('prior' and 'equalized_prior' alike) is re-described instead:

  1. the basis change N (integer, det +1, in the sg's setting group) and origin shift s of
     mxtaltools.crystal_fd.ops.reduce_cell, the cell where mono_reduction_penalty at margin 0 is zero;
  2. the asymmetric unit's fractional atoms f, as the trainer builds them, go to f' = N^-1 (f - s) (ops.change_basis's
     convention for a fractional point) in the cell with metric N^T G N;
  3. the row is written in the trainer's chart by data_processing.pool_anchors.Chart (describe, into_box): handedness
     +1, the standard-frame molecule, centre in the asymmetric-unit box.

Every re-described row is then read the trainer's way (analyze ['reduction_en', 'mace']) and must give its stored
energy within 0.2 kT, a reduction penalty under RED_TOL and a cell that survives the latent round trip; otherwise the
script refuses. Rows under RED_TOL are copied unchanged. OUT.pt has the input's keys, the two batches rebuilt on the
input's molecule, plus 'reduced_rows' (indices per key).
"""
import argparse
import time

import numpy as np
import torch

from data_processing.pool_anchors import ANALYZE_KW, Chart, RED_TOL
from data_processing.pool_assemble import rows_on
from data_processing.pool_assemble_mace import representable

KEY = 'mace'
KT = 2.494


def reduction_of(batch, chunk=20000):
    out = []
    for lo in range(0, batch.num_graphs, chunk):
        c = batch.subsample_new_batch(torch.arange(lo, min(lo + chunk, batch.num_graphs)))
        with torch.no_grad():
            out.append(c.analyze(['reduction_en'], assign_outputs=False, **ANALYZE_KW)['reduction_en'].double().flatten())
    return torch.cat(out)


def reduce_rows(ch, template, cp):
    """cp [n, 12] cell parameters of +1 rows -> cell parameters [n, 12] of the same crystals in their reduced cell."""
    from mxtaltools.crystal_fd.description import CrystalDescription
    from mxtaltools.crystal_fd.ops import reduce_cell
    rows = rows_on(template, cp)
    desc = CrystalDescription.from_crystal_params(rows.sg_ind.long(), rows.cell_lengths, rows.cell_angles,
                                                  rows.aunit_centroid[:, :3], rows.aunit_orientation[:, :3],
                                                  rows.aunit_handedness[:, :1])
    red = reduce_cell(desc)
    N = red.N.double()
    f = ch.frac_atoms(rows)                                              # [n, atoms, 3], old basis
    f_new = torch.einsum('nij,naj->nai', torch.linalg.inv(N), f - red.s.double()[:, None])
    cell = red.desc.cell  # fractional coordinates do not depend on the Cartesian convention of the cell matrix
    like = rows_on(template, torch.cat([cell[:, :3].float(), cell[:, 3:].float(), cp[:, 6:]], 1))
    out, rms = ch.describe(like, f_new)
    out, in_box = ch.into_box(out)
    new = out.full_cell_parameters().float()
    if not bool(in_box.all()):
        # the +1 molecules sit at y in (1/4, 1/2) or (3/4, 1): the asymmetric unit is an inversion partner (-x, -y, -z),
        # whose molecule the planar acridine matches at +1 by a proper rotation (pool_anchors_planar's re-description)
        m = torch.nonzero(~in_box).flatten()
        o2, rms2 = ch.describe(like.subsample_new_batch(m), -f_new[m])
        o2, in2 = ch.into_box(o2)
        print(f'  {len(m)} rows described through the inversion partner: fit RMSD max {float(rms2.max()):.2e} A',
              flush=True)
        if not bool(in2.all()):
            raise SystemExit(f'refused: {int((~in2).sum())} re-described rows have no description in the box')
        new[m] = o2.full_cell_parameters().float()
        rms = rms.clone()
        rms[m] = rms2
    if float(rms.max()) > 1e-3:
        raise SystemExit(f'refused: a re-described molecule fits its atoms at RMSD {float(rms.max()):.3g}')
    return new


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('prior')
    ap.add_argument('out')
    ap.add_argument('--mlip_path', required=True)
    ap.add_argument('--sg', type=int, default=14)
    ap.add_argument('--mol', default='D:/crystal_datasets/acridine/acridine_conformer.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--chunk', type=int, default=8)
    a = ap.parse_args(argv)
    t0 = time.time()
    d = torch.load(a.prior, weights_only=False)
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    ch = Chart(a.mol, a.sg, KEY, a.device, load_mace_model(a.mlip_path, device=a.device, dtype=torch.float32))
    template = d['prior']
    ch.setup(rows_on(template, template.full_cell_parameters().float()[:1]))
    out = dict(d)
    out['reduced_rows'] = {}
    for k in ('prior', 'equalized_prior'):
        b = d[k]
        cp = b.full_cell_parameters().float()
        E = b[KEY].double().flatten()
        red = reduction_of(b)
        idx = torch.nonzero(red > RED_TOL).flatten()
        print(f'{k}: {len(idx)} of {b.num_graphs} rows with a reduction penalty over {RED_TOL} '
              f'(max {float(red.max()):.3f})', flush=True)
        if len(idx):
            new = reduce_rows(ch, template, cp[idx])
            chk = rows_on(template, new)
            e, r = ch.read(chk, chunk=a.chunk)
            dE = (e - E[idx]).abs()
            rep = representable(chk)
            print(f'  re-described: |E read - stored| median {float(dE.median()):.4f}, max {float(dE.max()):.4f} kJ/mol; '
                  f'reduction penalty max {float(r.max()):.3g}; outside the latent box {int((~rep).sum())}; '
                  f'cell length change median {float((new[:, :3] - cp[idx, :3]).abs().amax(1).median()):.3f} A '
                  f'({time.time() - t0:.0f} s)', flush=True)
            if float(dE.max()) > 0.2 * KT or float(r.max()) > RED_TOL or not bool(rep.all()):
                raise SystemExit('refused: a re-described row does not read back as the same crystal in the domain')
            cp[idx] = new
        nb = rows_on(template, cp)
        setattr(nb, KEY, E.float())
        out[k] = nb
        out['reduced_rows'][k] = idx
    torch.save(out, a.out)
    print(f'wrote {a.out} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
