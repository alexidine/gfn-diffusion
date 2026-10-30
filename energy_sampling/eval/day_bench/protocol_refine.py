"""Raw samples through the benchmark protocol's substitution and optimisation depths (``docs/design/
generator_benchmark_protocol.md``, steps 2, 2.1 and 2.2).

Each raw sample (``cifs/*.cif``, or ``cifs/*.extxyz`` with molecule labels) has every molecule replaced by the constant
conformer (``p1_relax.p1_crystal``: a P1 cell of Z rigid molecules keeping the sample's molecule positions and
orientations). A sample is *legit* when every molecule's bond graph matches the conformer's
(``p1_ingest.fit_reference``); its largest fit RMSD is recorded, not thresholded. Molecules come from the sample's
labels when it has them (``bb_indices``), else from the periodic bond graph, else, when overlapping molecules make that
graph join them, from consecutive blocks of the conformer's atom count in file order, each rejoined along its own bonds
(``molecules_of``); a sample none of these splits into intact molecules is not legit. A rebuilt cell that does not
reproduce the fitted molecules (largest error above ``REBUILD_TOL``) is a failure of this pipeline, not of the sampler,
and is recorded apart. Every legit sample is optimised by ``converge.fire`` (the shared rigid-body optimiser: FIRE on
the target energy, a row stopping once its largest generalised force is at most ``converge.F_TOL``) and written at each
depth in ``--depths`` (0 = raw, as substituted), with its energy per molecule and largest generalised force there.

usage (licensed CSD Python API; GPU):
    python -m eval.day_bench.protocol_refine SAMPLE_DIR OUT_DIR REFCODE CONFORMER [--depths 0,10,30,100,300,1000]
        [--energy exp6] [--device cuda] [--vram 0.3] [--max_step 0.1] [--chunk 100] [--limit N (evenly spaced)]
        [--glob PATTERN]

CONFORMER as ``day_landscape.load_reference``. Writes OUT_DIR/depth_<k>/<stem>.extxyz (the P1 cell with a molecule
label per atom, ``bb_indices``, so a cell whose molecules overlap still splits into its molecules) and
OUT_DIR/refine_<energy>.csv.
"""
import argparse
import csv
import glob
import os
import time
import warnings

import numpy as np
import torch
from ase.io import read, write

from eval.day_bench.converge import CHUNK, F_TOL, fire
from eval.day_bench.p1_ingest import fit_reference, split_molecules
from eval.day_bench.p1_relax import p1_crystal, stages_for, to_atoms
from mxtaltools.dataset_utils.utils import collate_data_list

REBUILD_TOL = 0.1  # Angstrom


def _rejoined(z, frac, cell):
    """Cartesian positions of one molecule's atoms, each placed by the minimum image of a bond from an atom already
    placed (breadth-first from the first atom), or None when its bonds do not reach every atom."""
    from eval.day_bench.p1_ingest import _bond_radii
    df = frac[None] - frac[:, None]
    df -= np.round(df)
    d = df @ cell  # d[i, j]: minimum-image vector from atom i to atom j
    r = _bond_radii(z)
    bonded = np.linalg.norm(d, axis=-1) < r[:, None] + r[None]
    np.fill_diagonal(bonded, False)
    pos = np.full((len(z), 3), np.nan)
    pos[0] = frac[0] @ cell
    stack = [0]
    while stack:
        i = stack.pop()
        for j in np.nonzero(bonded[i])[0]:
            if np.isnan(pos[j, 0]):
                pos[j] = pos[i] + d[i, j]
                stack.append(j)
    return None if np.isnan(pos).any() else pos


def molecules_of(atoms, ref):
    """(cell to rebuild, largest fit RMSD) for a sample whose molecules are all intact, else ValueError. The cell is
    ``atoms`` itself when its labels or its bond graph give the molecules, else a copy with each consecutive block of the
    conformer's atom count rejoined whole and labelled."""
    from ase import Atoms
    from eval.day_bench.p1_ingest import Molecule, _bond_graph
    try:
        return atoms, max(fit_reference(m, ref)[1] for m in split_molecules(atoms))
    except ValueError as err:
        if 'bb_indices' in atoms.arrays:
            raise
        first = err
    n_at, z = len(ref.mol.z), atoms.get_atomic_numbers()
    if len(z) % n_at:
        raise first
    frac, cell, pos, rmsd = atoms.get_scaled_positions(), np.asarray(atoms.cell), [], []
    for s in range(0, len(z), n_at):
        p = _rejoined(z[s:s + n_at], frac[s:s + n_at], cell)
        if p is None:
            raise ValueError(f'{first}; consecutive atom blocks are not molecules')
        rmsd.append(fit_reference(Molecule(z[s:s + n_at], p, _bond_graph(z[s:s + n_at], p)), ref)[1])
        pos.append(p)
    out = Atoms(numbers=z, positions=np.concatenate(pos), cell=atoms.cell, pbc=True)
    out.arrays['bb_indices'] = np.repeat(np.arange(len(z) // n_at), n_at)
    return out, max(rmsd)


def labelled(atoms, ref):
    """``atoms`` (``p1_relax.to_atoms`` of a P1 cell: molecules in consecutive blocks of the conformer's atoms) with a
    molecule label per atom."""
    n_at = len(ref.mol.z)
    labels = np.repeat(np.arange(len(atoms) // n_at), n_at)
    z = atoms.get_atomic_numbers()
    assert len(labels) == len(atoms) and all(sorted(z[labels == g]) == sorted(ref.mol.z) for g in np.unique(labels)), \
        'unit-cell atoms are not in consecutive molecule blocks'
    atoms.arrays['bb_indices'] = labels
    return atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sample_dir')
    ap.add_argument('out_dir')
    ap.add_argument('refcode')
    ap.add_argument('conformer')
    ap.add_argument('--depths', default='0,10,30,100,300,1000')
    ap.add_argument('--energy', default='exp6')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max_step', type=float, default=0.1)  # FIRE['max_step'] (0.02) is a polishing setting
    ap.add_argument('--chunk', type=int, default=2 * CHUNK)  # rows per batch at 4 molecules per cell, half at 8
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--glob', default='*', help='raw files to take, by name (for example sample_sg41_*)')
    ap.add_argument('--vram', type=float, default=0.3, help='share of GPU memory this process may use')
    a = ap.parse_args()
    from ccdc import io as ccdc_io
    from eval.day_bench.day_landscape import load_reference
    torch.set_num_threads(2)
    if str(a.device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(a.vram)
    depths = sorted(int(k) for k in a.depths.split(','))
    ref = load_reference(a.refcode, a.conformer, ccdc_io.EntryReader('CSD'))
    exp6_kwargs = stages_for(a.energy)[0].get('exp6_kwargs')

    paths = sorted(glob.glob(os.path.join(a.sample_dir, 'cifs', a.glob + '.cif'))
                   + glob.glob(os.path.join(a.sample_dir, 'cifs', a.glob + '.extxyz')))
    if a.limit:  # evenly spaced, so a folder of several kinds (random search's space groups) keeps its mix
        paths = [paths[i] for i in np.unique(np.linspace(0, len(paths) - 1, a.limit).round().astype(int))]
    rows, groups = [], {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        row = dict(file=f'{stem}.extxyz', legit=False, rebuilt=False)
        rows.append(row)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            atoms = read(p)
        try:
            atoms, row['fit_rmsd'] = molecules_of(atoms, ref)
        except ValueError as err:
            row['note'] = f'molecule: {err}'[:100]
            continue
        row['legit'] = True
        c, residual = p1_crystal(atoms, ref, identifier=row['file'])
        if c is None:
            row['note'] = f'rebuild: {residual}'[:100]
            continue
        row['rebuild_residual'] = residual
        if residual > REBUILD_TOL:
            row['note'] = 'rebuilt cell does not reproduce the sample'
            continue
        row['rebuilt'] = True
        row['z'] = int(c.z_prime)
        groups.setdefault(int(c.z_prime), []).append((row, c))
    n_legit = sum(r['legit'] for r in rows)
    print(f'{len(rows)} raw samples: {n_legit} legit, {sum(r["rebuilt"] for r in rows)} rebuilt; '
          f'{dict((z, len(m)) for z, m in groups.items())} by molecules per cell', flush=True)

    for k in depths:
        os.makedirs(os.path.join(a.out_dir, f'depth_{k}'), exist_ok=True)
    t0 = time.time()
    for zp, members in groups.items():
        chunk = a.chunk if zp <= 4 else a.chunk // 2
        for i in range(0, len(members), chunk):
            part = members[i:i + chunk]
            _, _, _, steps, snaps = fire(collate_data_list([c for _, c in part], max_z_prime=zp), a.device, a.energy,
                                         exp6_kwargs, F_TOL, max(depths), snapshots=depths, max_step=a.max_step)
            for k, (cs, e, f) in snaps.items():
                for j, ((row, _), c) in enumerate(zip(part, cs)):
                    write(os.path.join(a.out_dir, f'depth_{k}', row['file']), labelled(to_atoms(c), ref),
                          format='extxyz')
                    row[f'e_{k}'] = float(e[j])
                    row[f'f_{k}'] = float(f[j])
            for j, (row, _) in enumerate(part):
                row['steps'] = int(steps[j])
            print(f'  Z {zp}: {min(i + chunk, len(members))} of {len(members)} in {time.time() - t0:.0f} s; energy '
                  f'median by depth ' + ', '.join(f'{k}: {np.median([r[f"e_{k}"] for r, _ in part]):.1f}'
                                                  for k in depths), flush=True)

    keys = ['file', 'legit', 'rebuilt', 'note', 'fit_rmsd', 'rebuild_residual', 'z', 'steps'] + \
        [f'{q}_{k}' for k in depths for q in ('e', 'f')]
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f'refine_{a.energy}.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in keys} for r in rows])
    done = [r for r in rows if r['rebuilt']]
    print(f'{len(done)} refined in {time.time() - t0:.0f} s on {a.device}; energy per molecule ({a.energy}) median by '
          f'depth: ' + ', '.join(f'{k}: {np.median([r[f"e_{k}"] for r in done]):.1f}' for k in depths), flush=True)


if __name__ == '__main__':
    main()
