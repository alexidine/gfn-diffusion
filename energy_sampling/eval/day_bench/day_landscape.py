"""One molecule's Day landscape (Taylor, Butler & Day 2025, deposit 10.5258/SOTON/D3094) as a reference source.

A refcode's directory of the FIT+DMA archive holds one or more ``<stem>.csv`` (id, space group, density, FIT+DMA lattice
energy, ...) and ``<stem>.cif`` (one block per structure: asymmetric unit plus symmetry operators). A chiral molecule's
directory splits its structures over ``_all``, ``_racsg`` and ``_sohncke`` files, and ``_all.csv`` can hold the Sohncke
rows only, so ``landscape`` takes the union of ids over every CSV and finds each id's block in whichever CIF holds it.
The deposit's DFT-corrected re-ranking (the Delta-ML archive's ``<refcode>/corrected.csv``: ``corrected_energy_kjmol``,
B86bPBE+XDM level) is joined where it exists; it covers part of some landscapes only.

``main`` writes the reference for ``protocol_refine``: every structure within WINDOW kJ/mol of the lowest energy under
either ranking (FIT+DMA, or corrected where present), as its primitive P1 cell (``primitive``) in ``OUT_DIR/cifs/``,
with ``OUT_DIR/reference.csv`` (both energies, absolute and relative to the landscape's lowest, the space group and
the molecules per conventional and primitive cell) and ``OUT_DIR/conformer.txt`` (Day's own conformer as a
``load_reference`` spec: the rigid molecule of the landscape's lowest FIT+DMA structure). The reference then goes
through ``protocol_refine`` like any sampler's draws, so reference minima and samples share one substitution, one
relaxation and one description (P1 at Z' = Z); which minima count as modes, and their weights, is decided from Day's
energies in ``protocol_report``.

usage (licensed CSD Python API, for the SMILES of the refcode):
    python -m eval.day_bench.day_landscape OUT_DIR REFCODE LANDSCAPE_DIR [--corrected CSV_OR_DELTA_ML_ZIP]
        [--window 10]
"""
import csv
import glob
import io
import os
from collections import Counter

import numpy as np
from ase.io import write

from eval.day_bench.p1_ingest import make_reference, split_molecules


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


def day_reference(stem_cif, structure_id, smiles):
    """Day's rigid conformer: the first molecule of one landscape structure's full cell, as a ``p1_ingest`` reference
    (every structure of a landscape carries the same conformer)."""
    return make_reference(split_molecules(block_to_atoms(blocks(stem_cif)[structure_id]))[0], smiles=smiles)


def family_smiles(reader, refcode):
    """SMILES of ``refcode``'s heaviest component, or of the first numbered member of its family when the bare family
    name (Day's directory names, e.g. IMAZOL) is not itself an entry."""
    for code in [refcode] + [f'{refcode}{i:02d}' for i in range(1, 100)]:
        try:
            return reader.entry(code).molecule.heaviest_component.smiles
        except RuntimeError:
            continue
    raise KeyError(f'no CSD entry for {refcode} or its numbered family members')


def load_reference(refcode, conformer, reader):
    """The rigid reference conformer named by ``conformer``: 'csd' (the refcode's deposited molecule) or
    'day:<stem CIF>:<structure id>' (Day's own, ``day_reference``)."""
    from eval.day_bench.csd_roundtrip import p1_cell
    smiles = family_smiles(reader, refcode)
    if conformer == 'csd':
        return make_reference(split_molecules(p1_cell(refcode, reader)[0])[0], smiles=smiles)
    kind, rest = conformer.split(':', 1)  # the CIF path may itself hold a drive colon
    cif, structure_id = rest.rsplit(':', 1)
    assert kind == 'day', f'unknown conformer {conformer!r}'
    return day_reference(cif, structure_id, smiles)


def landscape(landscape_dir, corrected=None, refcode=None):
    """{id: row} over the union of every CSV in ``landscape_dir``: ``sg``, ``fitdma`` (kJ/mol), ``cif`` (the CIF holding
    its block) and, when ``corrected`` (a ``corrected.csv``, or the Delta-ML zip read at ``<refcode>/corrected.csv``)
    has the id, ``corrected`` (kJ/mol); ``fitdma_rel`` and ``corrected_rel`` are relative to the lowest of each, and
    ``corrected`` and ``corrected_rel`` are None for an id the correction does not cover."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(landscape_dir, '*.csv'))):
        for r in csv.DictReader(open(path)):
            rows.setdefault(r['id'], dict(id=r['id'], sg=int(r['spacegroup']), fitdma=float(r['energy'])))
    where = {}
    for path in sorted(glob.glob(os.path.join(landscape_dir, '*.cif'))):
        for line in open(path, encoding='utf-8', errors='replace'):
            if line.startswith('data_'):
                where.setdefault(line[5:].strip(), path)
    missing = [i for i in rows if i not in where]
    assert not missing, f'{len(missing)} ids have no CIF block, e.g. {missing[:3]}'
    corr = {}
    if corrected:
        if corrected.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(corrected) as z:
                name = [n for n in z.namelist() if n.endswith(f'/{refcode}/corrected.csv')]
                text = z.read(name[0]).decode() if name else ''
        else:
            text = open(corrected).read()
        corr = {r['structure']: float(r['corrected_energy_kjmol']) for r in csv.DictReader(io.StringIO(text))}
    f0 = min(r['fitdma'] for r in rows.values())
    c0 = min((corr[i] for i in rows if i in corr), default=None)
    for i, r in rows.items():
        r.update(cif=where[i], fitdma_rel=r['fitdma'] - f0, corrected=corr.get(i),
                 corrected_rel=corr[i] - c0 if i in corr else None)
    return rows


def primitive(atoms, symprec=1e-3):
    """The primitive P1 cell of a full conventional cell (``block_to_atoms``), by spglib; the cell itself when spglib
    finds no smaller one or the reduction does not divide the atom count."""
    import spglib
    from ase import Atoms
    cell = (np.asarray(atoms.cell), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    out = spglib.standardize_cell(cell, to_primitive=True, no_idealize=True, symprec=symprec)
    if out is None or len(out[2]) >= len(atoms) or len(atoms) % len(out[2]):
        return atoms
    lattice, frac, numbers = out
    return Atoms(numbers=numbers, scaled_positions=frac, cell=lattice, pbc=True)


def day_conformer_spec(rows):
    """'day:<stem CIF>:<structure id>' of the lowest FIT+DMA structure of ``landscape`` rows."""
    lowest = min(rows.values(), key=lambda r: r['fitdma'])
    return f'day:{lowest["cif"]}:{lowest["id"]}'


def in_window(r, window):
    """Within ``window`` kJ/mol of the lowest under FIT+DMA, or under the correction where the row has one."""
    return r['fitdma_rel'] <= window or (r['corrected_rel'] is not None and r['corrected_rel'] <= window)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('refcode')
    ap.add_argument('landscape_dir')
    ap.add_argument('--corrected', default=None)
    ap.add_argument('--window', type=float, default=10.0)
    a = ap.parse_args()
    from ccdc import io as ccdc_io
    reader = ccdc_io.EntryReader('CSD')
    rows = landscape(a.landscape_dir, a.corrected, a.refcode.lower())
    spec = day_conformer_spec(rows)
    ref = load_reference(a.refcode.upper(), spec, reader)
    n_at = len(ref.mol.z)
    chosen = sorted((r for r in rows.values() if in_window(r, a.window)), key=lambda r: r['fitdma'])
    n_corr = sum(r['corrected'] is not None for r in rows.values())
    print(f'{a.refcode}: {len(rows)} structures in the landscape ({n_corr} with a corrected energy); {len(chosen)} '
          f'within {a.window:g} kJ/mol of the lowest under either ranking', flush=True)
    raw = os.path.join(a.out_dir, 'cifs')
    os.makedirs(raw, exist_ok=True)
    texts, out = {}, []
    for r in chosen:
        if r['cif'] not in texts:
            texts[r['cif']] = blocks(r['cif'])
        conv = block_to_atoms(texts[r['cif']][r['id']])
        prim = primitive(conv)
        assert len(conv) % n_at == 0 and len(prim) % n_at == 0, f'{r["id"]}: atoms not a whole number of molecules'
        write(os.path.join(raw, f'{r["id"]}.cif'), prim, format='cif')
        out.append(dict(file=f'{r["id"]}.extxyz', id=r['id'], day_sg=r['sg'], z_conv=len(conv) // n_at,
                        z_prim=len(prim) // n_at, fitdma=r['fitdma'], fitdma_rel=r['fitdma_rel'],
                        corrected='' if r['corrected'] is None else r['corrected'],
                        corrected_rel='' if r['corrected_rel'] is None else r['corrected_rel'], cif=r['cif']))
    with open(os.path.join(a.out_dir, 'reference.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(os.path.join(a.out_dir, 'conformer.txt'), 'w') as fh:
        fh.write(spec + '\n')
    print(f'wrote {len(out)} primitive cells to {raw}; molecules per primitive cell '
          f'{dict(sorted(Counter(r["z_prim"] for r in out).items()))}; conformer {spec}', flush=True)


if __name__ == '__main__':
    main()
