"""Brief relaxation of generator samples: the first STEPS steps of ``p1_relax``'s first stage, nothing more.

A raw sample carries clashes and off-minimum distortion that a structural fingerprint reads as differences in
packing. A few steps of the same rigid P1 relaxation remove the clashes while leaving the sample near where the
generator put it, so a fingerprint of the brief copy separates coverage (where samples go) from quality (how far
they are from a minimum) better than either the raw or the fully relaxed structure. Same reference conformer,
energy, optimiser and settings as ``p1_relax``; only ``max_num_steps`` of stage 1 differs and stage 2 is skipped.

usage (CPU):
    python -m eval.day_bench.p1_brief SAMPLE_DIR REFCODE [N_SAMPLES] [elj|exp6] [STEPS]

Briefly relaxed cells go to SAMPLE_DIR/brief<STEPS>_<energy>/ as P1 CIFs, with p1_brief<STEPS>_<energy>.csv.
"""
import csv
import glob
import os
import sys
import time
import warnings

import numpy as np
import torch
from ase.io import read, write

from eval.day_bench.p1_ingest import make_reference, split_molecules
from eval.day_bench.p1_relax import CHUNK, UNITS, p1_crystal, relax, stages_for, to_atoms
from mxtaltools.dataset_utils.utils import collate_data_list


def main(sample_dir, refcode, n=100, energy='exp6', steps=25):
    from ccdc import io as ccdc_io
    from eval.day_bench.csd_roundtrip import p1_cell
    torch.set_num_threads(2)
    reader = ccdc_io.EntryReader('CSD')
    atoms_ref, _, _ = p1_cell(refcode, reader)
    ref = make_reference(split_molecules(atoms_ref)[0], smiles=reader.entry(refcode).molecule.heaviest_component.smiles)
    stages = [dict(stages_for(energy)[0], max_num_steps=steps)]

    # the generator's own samples only: never the relaxed or control CIFs that sit in other subfolders
    paths = sorted(glob.glob(os.path.join(sample_dir, 'cifs', '*.cif')))[:n]
    rows, crystals = [], []
    for path in paths:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            atoms = read(path)
        crystal, residual = p1_crystal(atoms, ref, identifier=os.path.basename(path))
        if crystal is None:
            rows.append({'file': os.path.basename(path), 'built': False, 'note': residual})
            continue
        crystals.append(crystal)
        rows.append({'file': os.path.basename(path), 'built': True})
    zps = sorted({int(c.z_prime) for c in crystals})
    assert len(zps) == 1, 'one Z per run'

    out_dir = os.path.join(sample_dir, f'brief{steps}_{energy}')
    os.makedirs(out_dir, exist_ok=True)
    scratch = os.path.join(sample_dir, f'opt_intermediates_brief{steps}.pt')
    t0 = time.time()
    parts = [relax(collate_data_list(crystals[i:i + CHUNK], max_z_prime=zps[0]), stages, scratch=scratch)
             for i in range(0, len(crystals), CHUNK)]
    relaxed = [c for p in parts for c in p[0].batch_to_list()]
    e0 = torch.cat([p[1] for p in parts])
    e1 = torch.cat([p[2] for p in parts])
    print(f'{energy}: {len(crystals)} of {len(rows)} samples built; brief relaxation of {steps} steps in '
          f'{time.time() - t0:.0f} s on CPU')

    built = [r for r in rows if r['built']]
    assert len(built) == len(relaxed)
    for k, (r, c) in enumerate(zip(built, relaxed)):
        write(os.path.join(out_dir, r['file']), to_atoms(c), format='cif')
        r.update(e_start=float(e0[k]), e_end=float(e1[k]))
    with open(os.path.join(sample_dir, f'p1_brief{steps}_{energy}.csv'), 'w', newline='') as f:
        keys = ['file', 'built', 'e_start', 'e_end', 'note']
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in keys} for r in rows])
    e = np.array([[r['e_start'], r['e_end']] for r in built])
    print(f'energy per molecule ({UNITS[energy]}), median: start {np.median(e[:, 0]):.1f}, after {steps} steps '
          f'{np.median(e[:, 1]):.1f}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 100,
         sys.argv[4] if len(sys.argv) > 4 else 'exp6', int(sys.argv[5]) if len(sys.argv) > 5 else 25)
