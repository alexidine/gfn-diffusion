"""Run a folder of generator samples (P1 CIFs) through ``p1_ingest.ingest`` and summarise what symmetrises.

The reference conformer is the first molecule of a CSD entry's expanded cell, read from the local CSD
(licensed CSD Python API). Writes ``ingest.csv`` beside the samples, one row per CIF.

usage (CPU):
    python -m eval.day_bench.ingest_samples SAMPLE_DIR=REFCODE [SAMPLE_DIR=REFCODE ...]
"""
import csv
import glob
import os
import sys
import warnings

import numpy as np
import torch
from ase.io import read

from eval.day_bench.csd_roundtrip import p1_cell
from eval.day_bench.p1_ingest import SYMPRECS, ingest, make_reference, split_molecules

FIELDS = ('file', 'ok', 'reason', 'n_molecules', 'fit_rmsd_max', 'n_mirrored', 'sg', 'symprec', 'z_prime',
          'n_subgroups', 'residual_max', 'cell_strain', 'angle_change_deg', 'canonical')


def run(sample_dir, refcode, reader, symprecs=SYMPRECS, out_name='ingest.csv'):
    atoms_ref, _, _ = p1_cell(refcode, reader)
    ref = make_reference(split_molecules(atoms_ref)[0])
    rows = []
    for path in sorted(glob.glob(os.path.join(sample_dir, '**', '*.cif'), recursive=True)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            atoms = read(path)
        r = ingest(atoms, ref, symprecs=symprecs, identifier=os.path.basename(path))
        rows.append({'file': os.path.relpath(path, sample_dir), 'ok': r.ok, 'reason': r.reason,
                     'n_molecules': r.n_molecules, 'fit_rmsd_max': r.fit_rmsd_max, 'n_mirrored': r.n_mirrored,
                     'sg': r.sg, 'symprec': r.symprec, 'z_prime': r.z_prime, 'n_subgroups': r.n_subgroups,
                     'residual_max': r.residual_max, 'cell_strain': r.cell_strain,
                     'angle_change_deg': r.angle_change_deg, 'canonical': r.canonical})
    with open(os.path.join(sample_dir, out_name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def summarise(name, rows):
    n = len(rows)
    ok = [r for r in rows if r['ok']]
    fit_fail = [r for r in rows if r['reason'].startswith(('split', 'fit'))]
    sym_fail = [r for r in rows if r['reason'].startswith('symmetry')]
    fitted = [r for r in rows if not r['reason'].startswith(('split', 'fit'))]

    def q(vals, p):
        return float(np.percentile(vals, p)) if vals else float('nan')

    rms = [r['fit_rmsd_max'] for r in fitted]
    res = [r['residual_max'] for r in ok]
    sgs = {}
    for r in ok:
        sgs[r['sg']] = sgs.get(r['sg'], 0) + 1
    top = ', '.join(f'{k}: {v}' for k, v in sorted(sgs.items(), key=lambda kv: -kv[1])[:4])
    reasons = {}
    for r in rows:
        if not r['ok']:
            key = r['reason'].split(':')[0] + ': ' + r['reason'].split(':')[1].strip()[:40] if ':' in r['reason'] else r['reason']
            reasons[key] = reasons.get(key, 0) + 1
    print(f'\n{name}: {n} samples; graph or fit failed {len(fit_fail)}; symmetry failed {len(sym_fail)}; '
          f'converted {len(ok)}')
    print(f'  fit RMSD to the reference conformer (A), median / 90th pct over fitted samples: '
          f'{q(rms, 50):.3f} / {q(rms, 90):.3f}')
    if ok:
        print(f'  converted: residual (A) median / 90th pct {q(res, 50):.2f} / {q(res, 90):.2f}; '
              f'cell length change median {q([r["cell_strain"] for r in ok], 50):.3f}; '
              f'angle change (deg) median {q([r["angle_change_deg"] for r in ok], 50):.2f}; '
              f'symprec used (A) median {q([r["symprec"] for r in ok], 50):.2f}')
        print(f'  space groups found (count): {top}')
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:4]:
        print(f'  failure x{v}: {k}')


WIDE = SYMPRECS + (0.7, 1.0, 1.5)  # diagnostic ladder: how far from symmetric are the samples?


def main(args):
    from ccdc import io as ccdc_io
    torch.set_num_threads(2)
    wide = args and args[0] == '--wide'
    pairs = args[1:] if wide else args
    reader = ccdc_io.EntryReader('CSD')
    for pair in pairs:
        sample_dir, refcode = pair.split('=')
        rows = run(sample_dir, refcode, reader, symprecs=WIDE if wide else SYMPRECS,
                   out_name='ingest_wide.csv' if wide else 'ingest.csv')
        tag = ' (tolerances up to 1.5 A)' if wide else ''
        summarise(f'{os.path.basename(os.path.normpath(sample_dir))} (reference {refcode}){tag}', rows)


if __name__ == '__main__':
    main(sys.argv[1:])
