"""One molecule's Day landscape (Taylor, Butler & Day 2025, deposit 10.5258/SOTON/D3094) as a reference source.

Reads the molecule's FIT+DMA landscape (``<stem>.csv``: id, space group, density, FIT+DMA lattice energy; ``<stem>.cif``:
one block per structure, asymmetric unit plus symmetry operators) and, when given, the deposit's DFT-corrected
re-ranking (``corrected.csv`` of the Delta-ML archive: ``corrected_energy_kjmol``, B86bPBE+XDM level). Structures within
WINDOW kJ/mol of the lowest corrected energy (FIT+DMA energy when no correction is given) are refitted with the shared
reference conformer (``p1_ingest.ingest``, in their own space group) and relaxed under the benchmark energy by
``p1_relax``'s two stages, so they share the samples' coordinate space for RDF identity. The folder is laid out as
``p1_relax`` lays out a generator's; ``p1_relax_<energy>.csv`` adds ``day_fitdma`` and ``ref_energy`` (the corrected
energy, the one the reference ranking uses).

usage (licensed CSD Python API):
    python -m eval.day_bench.day_landscape OUT_DIR REFCODE STEM_CSV STEM_CIF [CORRECTED_CSV|-] WINDOW [energy] [DEVICE]
"""
import copy
import csv
import io
import os
import sys
import time
import warnings

import numpy as np
import torch
from ase.io import read, write

from eval.day_bench.p1_ingest import ingest, make_reference, split_molecules
from eval.day_bench.p1_relax import CHUNK, TIGHT, relax, relax_experimental, settle_step, stages_for, to_atoms
from mxtaltools.crystal_search.crystal_opt_utils import canonicalize_aunit
from mxtaltools.dataset_utils.utils import collate_data_list


def blocks(cif_path):
    """{structure id: CIF text of its data block}."""
    out, cur, name = {}, [], None
    for line in open(cif_path, encoding='utf-8', errors='replace'):
        if line.startswith('data_'):
            if name:
                out[name] = ''.join(cur)
            name, cur = line[5:].strip(), [line]
        elif name:
            cur.append(line)
    if name:
        out[name] = ''.join(cur)
    return out


def block_to_atoms(text):
    """The full unit cell of one cspy CIF block: the asymmetric unit expanded by its listed operators
    (``_symmetry_equiv_pos_as_xyz``), images within 1e-4 fractional of one another merged. ASE's reader refuses these
    blocks because they name no space-group symbol."""
    from ase import Atoms
    from ase.cell import Cell
    cell, ops, sites, mode = {}, [], [], None
    for line in text.splitlines():
        t = line.strip()
        if t.startswith('_cell_length_') or t.startswith('_cell_angle_'):
            k, v = t.split()[:2]
            cell[k] = float(v)
        elif t == 'loop_':
            mode = 'loop'
        elif t.startswith('_symmetry_equiv_pos_as_xyz'):
            mode = 'ops'
        elif t.startswith('_atom_site_occupancy'):
            mode = 'sites'
        elif t.startswith('_') or not t or t.startswith('#'):
            continue
        elif mode == 'ops':
            ops.append(t.split()[-1])
        elif mode == 'sites':
            f = t.split()
            sites.append((f[1], [float(x) for x in f[2:5]]))

    def affine(op):
        exprs = op.split(',')
        def ev(v):
            x, y, z = v
            return np.array([eval(e, {}, {'x': x, 'y': y, 'z': z}) for e in exprs], dtype=float)
        t = ev((0.0, 0.0, 0.0))
        return np.stack([ev(u) - t for u in np.eye(3)], 1), t

    frac, sym = [], []
    for R, t in (affine(o) for o in ops):
        for el, xyz in sites:
            frac.append((R @ np.array(xyz) + t) % 1.0)
            sym.append(el)
    frac = np.array(frac)
    keep = []
    for i in range(len(frac)):
        d = frac[keep] - frac[i] if keep else np.zeros((0, 3))
        d -= np.round(d)
        if not keep or (np.abs(d).max(1) > 1e-4).all():
            keep.append(i)
    c = Cell.fromcellpar([cell['_cell_length_a'], cell['_cell_length_b'], cell['_cell_length_c'],
                          cell['_cell_angle_alpha'], cell['_cell_angle_beta'], cell['_cell_angle_gamma']])
    return Atoms(symbols=[sym[i] for i in keep], scaled_positions=frac[keep], cell=c, pbc=True)


def main(out_dir, refcode, stem_csv, stem_cif, corrected_csv, window, energy='exp6', device='cpu'):
    from ccdc import io as ccdc_io
    from eval.day_bench.csd_roundtrip import p1_cell
    torch.set_num_threads(2)
    if str(device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(0.4)
    reader = ccdc_io.EntryReader('CSD')
    atoms_ref, _, _ = p1_cell(refcode, reader)
    ref = make_reference(split_molecules(atoms_ref)[0], smiles=reader.entry(refcode).molecule.heaviest_component.smiles)
    stages = stages_for(energy)

    land = {r['id']: r for r in csv.DictReader(open(stem_csv))}
    corr = {}
    if corrected_csv and corrected_csv != '-':
        corr = {r['structure']: float(r['corrected_energy_kjmol']) for r in csv.DictReader(open(corrected_csv))}
    rank = {i: corr.get(i, float(r['energy'])) for i, r in land.items()} if corr else \
        {i: float(r['energy']) for i, r in land.items()}
    e0 = min(rank.values())
    chosen = sorted((i for i in rank if rank[i] - e0 <= window), key=lambda i: rank[i])
    print(f'{len(land)} structures in the landscape; {len(chosen)} within {window} kJ/mol of the lowest '
          f'{"corrected (DFT-level)" if corr else "FIT+DMA"} energy', flush=True)

    texts = blocks(stem_cif)
    raw_dir, rel_dir = os.path.join(out_dir, 'cifs'), os.path.join(out_dir, f'relaxed_{energy}')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(rel_dir, exist_ok=True)
    rows, crystals, names = [], [], []
    for i in chosen:
        atoms = block_to_atoms(texts[i])
        name = f'{i}.cif'
        write(os.path.join(raw_dir, name), atoms, format='cif')
        res = ingest(atoms, ref, identifier=name)
        row = dict(file=name, day_sg=land[i]['spacegroup'], day_fitdma=float(land[i]['energy']), ref_energy=rank[i])
        if not res.ok:
            rows.append(dict(row, built=False, note=res.reason))
            continue
        try:  # the relaxer canonicalises the asymmetric unit first and refuses some settings (seen: P4_2/n)
            canonicalize_aunit(collate_data_list([copy.deepcopy(res.crystal)], max_z_prime=1))
        except ValueError as err:
            rows.append(dict(row, built=False, note=f'relaxer refuses: {err}'[:120]))
            continue
        rows.append(dict(row, built=True, build_residual=res.residual_max))
        crystals.append(res.crystal)
        names.append(name)
    print(f'{len(crystals)} of {len(chosen)} converted to Z\' = 1 crystals of the reference conformer', flush=True)

    t0 = time.time()
    parts = [relax(collate_data_list(crystals[k:k + CHUNK * 4], max_z_prime=1), stages, device=device,
                   scratch=os.path.join(out_dir, 'opt_intermediates.pt'))
             for k in range(0, len(crystals), CHUNK * 4)]
    relaxed = [c for p in parts for c in p[0].batch_to_list()]
    e_start = torch.cat([p[1] for p in parts])
    e_end = torch.cat([p[2] for p in parts])
    traj = np.concatenate([p[3] for p in parts], axis=1)
    forms = relax_experimental([refcode], ref, stages, reader, rel_dir, os.path.join(out_dir, 'opt_intermediates.pt'))
    ex1 = forms[refcode][2]
    by_name = {r['file']: r for r in rows}
    for k, (n, c) in enumerate(zip(names, relaxed)):
        cell = to_atoms(c)
        write(os.path.join(rel_dir, n), cell, format='cif')
        res = ingest(cell, ref, symprecs=TIGHT, identifier=n)
        by_name[n].update(e_start=float(e_start[k]), e_end=float(e_end[k]), e_vs_exp=float(e_end[k]) - ex1,
                          settle=settle_step(traj[:, k]), sg=res.sg if res.ok else None,
                          symprec=res.symprec if res.ok else None, note='' if res.ok else res.reason)
    keys = ['file', 'built', 'build_residual', 'day_sg', 'day_fitdma', 'ref_energy', 'e_start', 'e_end', 'e_vs_exp',
            'settle', 'sg', 'symprec', 'note']
    with open(os.path.join(out_dir, f'p1_relax_{energy}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in keys} for r in rows])
    kept = sum(1 for r in rows if r['built'] and str(r.get('sg')) == str(r['day_sg']))
    print(f'relaxed {len(crystals)} under {energy} on {device} in {time.time() - t0:.0f} s; {kept} kept their space '
          f'group; energy above relaxed {refcode} (kJ/mol per molecule): median '
          f'{np.median([r["e_vs_exp"] for r in rows if r["built"]]):.1f}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]),
         sys.argv[7] if len(sys.argv) > 7 else 'exp6', sys.argv[8] if len(sys.argv) > 8 else 'cpu')
