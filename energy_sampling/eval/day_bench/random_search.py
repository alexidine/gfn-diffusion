"""Random-search baseline and reference minima: random Z' = 1 crystals in chosen space groups, relaxed in group.

Starts come from MXtalTools' own random initialiser (``MolCrystalOps.sample_random_crystal_parameters``: cell
angles, lengths, orientation and centroid at random, then hard cleaning), with each crystal's cell rescaled to a
packing coefficient drawn as the crystal search's ``init_target_cp: std`` draws it (normal 0.6226 +- 0.0447,
clipped to [0.45, 0.95]). The reduced-cell initialiser is not used: its loop has no attempt limit, and identity
checks downstream are independent of the cell setting. Each start is relaxed in its own space group by
``p1_relax``'s two stages (every row takes every step), and the folder is laid out as ``p1_relax`` lays out a
generator's, so ``compack_endpoints`` and ``rdf_compare`` read it unchanged: starts as P1 CIFs in ``cifs/``, relaxed
cells in ``relaxed_<energy>/`` beside the relaxed experimental forms, and ``p1_relax_<energy>.csv``.

usage (CPU, licensed CSD Python API):
    python -m eval.day_bench.random_search OUT_DIR REFCODE SG:N[,SG:N...] [elj|exp6] [OTHER_FORM,...] [SEED] [DEVICE] [CHUNK]

DEVICE (default cpu) runs the relaxation there; on a GPU the process is held to 40% of the card's memory. CHUNK
(default p1_relax.CHUNK) is the number of crystals relaxed per batch.
"""
import csv
import math
import os
import sys
import time

import numpy as np
import torch
from ase.io import write

from eval.day_bench.p1_ingest import _crystal, ingest, make_reference, split_molecules
from eval.day_bench.p1_relax import CHUNK, TIGHT, UNITS, relax, relax_experimental, settle_step, stages_for, to_atoms
from mxtaltools.dataset_utils.utils import collate_data_list


def starts(ref, sg, n, seed, chunk=CHUNK):
    """``n`` random Z' = 1 crystals of ``ref`` in space group ``sg``, as a list of batches of at most ``chunk``."""
    out = []
    for k, i in enumerate(range(0, n, chunk)):
        m = min(chunk, n - i)
        batch = collate_data_list([_crystal(ref, sg, [10.0] * 3, [math.pi / 2] * 3, ref.mol.pos, [0.5] * 3)
                                   for _ in range(m)], max_z_prime=1)
        g = seed + 1000 * sg + k
        torch.manual_seed(g)
        target = (torch.randn(m) * 0.0447 + 0.6226).clip(min=0.45, max=0.95)
        batch.sample_random_crystal_parameters(target_packing_coeff=target, seed=g)
        off = (batch.packing_coeff - target).abs() > 0.1  # as crystal_search.utils.get_initial_state
        if off.any():
            batch.cell_lengths = batch.cell_lengths * (batch.packing_coeff / target).pow(1.0 / 3.0)[:, None]
            batch.clean_cell_parameters(mode='hard')
            batch.box_analysis()
        out.append(batch)
    return out


def main(out_dir, refcode, plan, energy='exp6', extra_forms=(), seed=0, device='cpu', chunk=CHUNK):
    from ccdc import io as ccdc_io
    from eval.day_bench.csd_roundtrip import p1_cell
    torch.set_num_threads(2)
    if str(device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(0.4)
    reader = ccdc_io.EntryReader('CSD')
    atoms_ref, _, _ = p1_cell(refcode, reader)
    ref = make_reference(split_molecules(atoms_ref)[0], smiles=reader.entry(refcode).molecule.heaviest_component.smiles)
    stages = stages_for(energy)
    raw_dir, rel_dir = os.path.join(out_dir, 'cifs'), os.path.join(out_dir, f'relaxed_{energy}')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(rel_dir, exist_ok=True)
    scratch = os.path.join(out_dir, 'opt_intermediates.pt')

    rows = []
    t0 = time.time()
    for sg, n in plan:
        k = 0
        for batch in starts(ref, sg, n, seed, chunk):
            names = [f'sample_sg{sg}_{k + j:05d}.cif' for j in range(batch.num_graphs)]
            k += batch.num_graphs
            for name, c in zip(names, batch.batch_to_list()):
                write(os.path.join(raw_dir, name), to_atoms(c), format='cif')
            relaxed, e0, e1, traj = relax(batch, stages, device=device, scratch=scratch)
            for j, (name, c) in enumerate(zip(names, relaxed.batch_to_list())):
                cell = to_atoms(c)
                write(os.path.join(rel_dir, name), cell, format='cif')
                res = ingest(cell, ref, symprecs=TIGHT, identifier=name)
                rows.append(dict(file=name, built=True, build_residual=0.0, start_sg=sg, e_start=float(e0[j]),
                                 e_end=float(e1[j]), settle=settle_step(traj[:, j]),
                                 sg=res.sg if res.ok else None, symprec=res.symprec,
                                 residual=res.residual_max if res.ok else None, note='' if res.ok else res.reason))
        print(f'sg {sg}: {n} random starts relaxed on {device} ({time.time() - t0:.0f} s so far)', flush=True)

    forms = relax_experimental([refcode, *extra_forms], ref, stages, reader, rel_dir, scratch)
    ex1 = forms[refcode][2]
    for r in rows:
        r['e_vs_exp'] = r['e_end'] - ex1
    keys = ['file', 'built', 'build_residual', 'start_sg', 'e_start', 'e_end', 'e_vs_exp', 'settle', 'sg', 'symprec',
            'residual', 'note']
    with open(os.path.join(out_dir, f'p1_relax_{energy}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    e = np.array([r['e_vs_exp'] for r in rows])
    for rc, (sg, a, z) in forms.items():
        print(f'experimental {rc} relaxed in sg {sg}: {a:.1f} -> {z:.1f} ({UNITS[energy]})')
    print(f'{len(rows)} random starts relaxed in {time.time() - t0:.0f} s on CPU; energy above the relaxed experimental '
          f'form ({UNITS[energy]}): median {np.median(e):.1f}, lowest {e.min():.1f}; settle step median '
          f'{int(np.median([r["settle"] for r in rows]))}')


if __name__ == '__main__':
    plan = [tuple(int(v) for v in s.split(':')) for s in sys.argv[3].split(',')]
    main(sys.argv[1], sys.argv[2], plan, sys.argv[4] if len(sys.argv) > 4 else 'exp6',
         tuple(filter(None, sys.argv[5].split(','))) if len(sys.argv) > 5 else (),
         int(sys.argv[6]) if len(sys.argv) > 6 else 0, sys.argv[7] if len(sys.argv) > 7 else 'cpu',
         int(sys.argv[8]) if len(sys.argv) > 8 else CHUNK)
