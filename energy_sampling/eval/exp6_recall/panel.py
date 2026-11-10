"""The exp-6 recall test's panel: rigid conformers and experimental anchor crystals (docs/design/exp6_recall_test.md).

For each panel molecule with a CSD anchor: the entry's P1 cell, hydrogens moved along their bonds to the CSD's
neutron-normalised lengths (C-H 1.089, N-H 1.015, O-H 0.993 A), the cell's first molecule as the rigid reference, and
``eval.day_bench.p1_ingest.ingest`` to the Z' = 1 crystal in the searched space group. Norbornane, with no ordered CSD
entry, gets an RDKit ETKDG (seed 0) + MMFF94 conformer and no anchor.

Writes to OUT_DIR (a new folder; refuses to overwrite):
    <name>_mol.pt     [MolData], the search's mol_path; identifier = name, smiles from the CSD entry
    <name>_anchor.pt  [MolCrystalData] (one row), the anchor in the searched space group, the same conformer and identifier
    panel.json        per molecule: refcode, CSD space group and Z', searched sg, ingest residual and fit, notes

usage (licensed CSD Python API):
    python -m eval.exp6_recall.panel OUT_DIR
"""
import io
import json
import os
import sys

import numpy as np
import torch

# name, CSD refcode (None: no anchor), searched space group (international number), SMILES (None: the CSD entry's)
PANEL = [
    ('imidazole', 'IMAZOL06', 14, None),
    ('pyridone', 'PYRIDO13', 19, None),
    ('uracil', 'URACIL', 14, None),
    ('benzoic_acid', 'BENZAC12', 14, None),  # 123 K, ordered; BENZAC04 has no coordinates
    ('acetic_acid', 'ACETAC07', 33, None),
    ('naphthalene', 'NAPHTA04', 4, None),   # P2_1/a with Z' = 0.5; P2_1 is its Z' = 1 subgroup
    ('acridine', 'ACRDIN04', 14, None),  # 185 K, R 3.8%; ACRDIN11 has R 20.6%
    ('norbornane', None, 2, 'C1CC2CCC1C2'),
]

XH = {6: 1.089, 7: 1.015, 8: 0.993}  # A, CSD normalised X-H lengths


def normalise_h(atoms):
    """Move every H along its bond to its nearest heavy atom (minimum image) to that element's XH length."""
    z = atoms.numbers
    heavy = np.nonzero(z > 1)[0]
    pos = atoms.get_positions()
    for i in np.nonzero(z == 1)[0]:
        d = atoms.get_distances(i, heavy, mic=True, vector=True)
        k = int(np.argmin(np.linalg.norm(d, axis=1)))
        v = -d[k]  # from H to the heavy atom, minimum image
        L = XH.get(int(z[heavy[k]]))
        if L is None:
            raise ValueError(f'H bonded to element {z[heavy[k]]}')
        heavy_pos = pos[i] + v
        pos[i] = heavy_pos - v / np.linalg.norm(v) * L
    atoms.set_positions(pos)
    return atoms


def mol_from_reference(ref, name, smiles):
    from mxtaltools.dataset_utils.data_classes import MolData
    pos = torch.tensor(ref.mol.pos, dtype=torch.float32)
    return MolData(z=torch.tensor(ref.mol.z, dtype=torch.long), pos=pos, x=torch.zeros(len(ref.mol.z), 1),
                   smiles=smiles, identifier=name,
                   radius=(pos.double() - pos.double().mean(0)).norm(dim=1).max().float(), do_mol_analysis=True)


def rdkit_mol(name, smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from mxtaltools.dataset_utils.data_classes import MolData
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(m, randomSeed=0) != 0:
        raise RuntimeError(f'{name}: ETKDG failed')
    AllChem.MMFFOptimizeMolecule(m)
    pos = torch.tensor(m.GetConformer().GetPositions(), dtype=torch.float32)
    z = torch.tensor([a.GetAtomicNum() for a in m.GetAtoms()], dtype=torch.long)
    return MolData(z=z, pos=pos, x=torch.zeros(len(z), 1), smiles=smiles, identifier=name,
                   radius=(pos.double() - pos.double().mean(0)).norm(dim=1).max().float(), do_mol_analysis=True)


def main(out_dir):
    from ase.io import read
    from ccdc import io as ccdc_io
    from eval.day_bench.p1_ingest import ingest, make_reference, split_molecules
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise SystemExit(f'{out_dir} exists and is not empty')
    os.makedirs(out_dir, exist_ok=True)
    reader = ccdc_io.EntryReader('CSD')
    record = {}
    for name, refcode, sg, smiles in PANEL:
        rec = dict(refcode=refcode, sg_searched=sg)
        if refcode is None:
            mol = rdkit_mol(name, smiles)
            rec.update(smiles=smiles, anchor=None, note='no ordered CSD entry; RDKit ETKDG seed 0 + MMFF94')
        else:
            e = reader.entry(refcode)
            smiles = smiles or e.molecule.smiles
            atoms = normalise_h(read(io.StringIO(e.crystal.to_string('cif')), format='cif'))
            ref = make_reference(split_molecules(atoms)[0], smiles=smiles)
            r = ingest(atoms, ref, identifier=name, prefer_sg=sg)
            rec.update(smiles=smiles, csd_sg=e.crystal.spacegroup_symbol, csd_z_prime=float(e.crystal.z_prime),
                       ok=r.ok, reason=r.reason, sg_found=r.sg, n_subgroups=r.n_subgroups,
                       fit_rmsd_max_A=r.fit_rmsd_max, residual_max_A=r.residual_max, n_mirrored=r.n_mirrored,
                       cell_strain=r.cell_strain, angle_change_deg=r.angle_change_deg, canonical=r.canonical)
            if not r.ok or r.sg != sg:
                print(f'{name}: ingest gave ok={r.ok} sg={r.sg} ({r.reason}); no anchor written')
                rec['anchor'] = None
            else:
                torch.save([r.crystal], os.path.join(out_dir, f'{name}_anchor.pt'))  # a list, as stream seeds files are read
                rec['anchor'] = f'{name}_anchor.pt'
            mol = mol_from_reference(ref, name, smiles)
        torch.save([mol], os.path.join(out_dir, f'{name}_mol.pt'))
        rec.update(n_atoms=int(mol.num_atoms), mol=f'{name}_mol.pt')
        record[name] = rec
        print(name, {k: v for k, v in rec.items() if k not in ('smiles',)})
    with open(os.path.join(out_dir, 'panel.json'), 'w') as fh:
        json.dump(record, fh, indent=1, default=float)


if __name__ == '__main__':
    main(sys.argv[1])
