"""CPU tests for data_processing/random_conformers.py (random valid conformers for crystal-search molecules).

WHAT THIS FILE IS FOR. A random-torsion conformer that clashes, stretches a bond, flips a stereocentre or twists a
double bond still builds a crystal, so a broken generator or check would not fail downstream. Covered: which bonds are
spun (ring, double and terminal-F bonds never; methyl yes); that conformers are reproducible per (SMILES, replicate),
differ across replicates, are actually spun, and pass every check; NEGATIVE CONTROLS for the checks (a forced clash, a
stretched bond, a twisted double bond); and the MolData fields prep_qm9_anchor_mols.py and the prior builder read
(all-atom radius, identifier format, the stereo-free SMILES).

    python -m pytest -q tests/crystal/test_random_conformers.py
"""
import os
import sys

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

from data_processing import random_conformers as rc  # noqa: E402

MOLS = ['CCCCO', 'CC(=O)Nc1ccccc1', 'OCC1CCOC1', 'C/C=C/C(=O)OC', 'FC(F)c1ccncc1']


def _setup(smi, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, ps) == 0
    pos0 = mol.GetConformer().GetPositions()
    topo = Chem.GetDistanceMatrix(mol)
    bonds = np.array(sorted(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()))
    return mol, pos0, topo, bonds, rc.stereo_3d(mol, pos0)


def test_rotor_rule():
    mol = Chem.AddHs(Chem.MolFromSmiles('CC(=O)c1ccc(F)cc1'))
    spun = {tuple(sorted((u, v))) for u, v, _ in rc.rotors(mol)}
    for b in mol.GetBonds():
        key = tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        syms = {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()}
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE or 'H' in syms or 'F' in syms:
            assert key not in spun, (syms, b.GetBondType())
    heavy_single_acyclic = [b for b in mol.GetBonds() if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE
                            and 'H' not in {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()}
                            and 'F' not in {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()}]
    assert len(spun) == len(heavy_single_acyclic) == 2   # CH3-C(=O) and C(=O)-ring


def test_conformers_are_valid_reproducible_and_spun():
    for smi in MOLS:
        m1, i1 = rc.conformer(smi, 0)
        m2, _ = rc.conformer(smi, 0)
        m3, _ = rc.conformer(smi, 1)
        assert m1 is not None, (smi, i1)
        p1, p2, p3 = (m.GetConformer().GetPositions() for m in (m1, m2, m3))
        assert np.array_equal(p1, p2), smi                          # same (SMILES, replicate): same conformer
        assert not np.allclose(p1, p3), smi                         # another replicate: another conformer
        mol, pos0, topo, bonds, st0 = _setup(smi, seed=rc.seed_of(smi, 0))
        assert rc.check(mol, p1, pos0, st0, topo, bonds) == [], smi
        if i1['rotors']:
            assert not np.allclose(p1, pos0, atol=1e-3), smi           # the torsions were actually spun


def test_checks_catch_broken_geometry_negative_controls():
    mol, pos0, topo, bonds, st0 = _setup('CCCCC=CC')
    assert rc.check(mol, pos0, pos0, st0, topo, bonds) == []
    far = np.argwhere(topo >= 4)[0]
    bad = pos0.copy()
    bad[far[1]] = bad[far[0]] + np.array([0.5, 0.0, 0.0])            # two distant atoms 0.5 A apart
    assert 'clash' in rc.check(mol, bad, pos0, st0, topo, bonds)
    i, j = bonds[0]
    bad = pos0.copy()
    bad[j] = bad[i] + 1.4 * (bad[j] - bad[i])                         # one bond stretched by 40%
    assert 'bond_length' in rc.check(mol, bad, pos0, st0, topo, bonds)
    db = [b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE][0]
    a1, a2 = db.GetBeginAtomIdx(), db.GetEndAtomIdx()
    mask = np.zeros(mol.GetNumAtoms(), bool)                          # a double bond is never a rotor: twist by hand
    stack, seen = [a2], {a2}
    while stack:
        x = stack.pop()
        for nb in mol.GetAtomWithIdx(x).GetNeighbors():
            y = nb.GetIdx()
            if y != a1 and y not in seen:
                seen.add(y)
                stack.append(y)
    mask[list(seen - {a2})] = True
    bad = rc.rotate_about(pos0.copy(), a1, a2, mask, np.pi / 2)
    assert 'db_twist' in rc.check(mol, bad, pos0, st0, topo, bonds)


def test_molecule_without_distant_pairs_and_a_raising_smiles_do_not_abort(tmp_path, monkeypatch):
    m, info = rc.conformer('n12n3n1n23', 0)        # every atom pair bonded: no pair two bonds apart to clash-check
    assert m is not None, info
    real = rc.conformer
    monkeypatch.setattr(rc, 'conformer', lambda s, j, a: (_ for _ in ()).throw(RuntimeError()) if s == 'CCO' else
                        real(s, j, a))
    path = str(tmp_path / 't_raw_shard000.pt')
    n, made, failed = rc._shard((path, path[:-3] + '.tsv', str(tmp_path / 't_failed_shard000.tsv'), ['CCO', 'CCN'], 1,
                                 20))
    assert (n, made, failed) == (2, 1, 1)
    assert 'error:RuntimeError' in (tmp_path / 't_failed_shard000.tsv').read_text(encoding='utf-8')


def test_moldata_fields():
    smi = 'OCC1CCOC1'
    mol, _ = rc.conformer(smi, 0)
    md = rc.moldata(smi, mol, f'{smi}_c0')
    assert md.identifier == 'OCC1CCOC1_c0' and md.smiles == smi
    pos = md.pos.double()
    assert abs(float(md.radius) - float((pos - pos.mean(0)).norm(dim=1).max())) < 1e-5   # all-atom radius
    assert int(md.num_atoms) == mol.GetNumAtoms() and (md.z == 1).any()
    assert float(md.mass) > 0 and float(md.mol_volume) > 0


def test_one_shard_end_to_end(tmp_path):
    path = str(tmp_path / 't_raw_shard000.pt')
    n, made, failed = rc._shard((path, path[:-3] + '.tsv', str(tmp_path / 't_failed_shard000.tsv'), MOLS, 2, 20))
    assert (n, made + failed) == (5, 10) and made >= 9
    mols = torch.load(path, weights_only=False)
    assert len(mols) == made and {m.identifier for m in mols} <= {f'{s}_c{j}' for s in MOLS for j in range(2)}
    rows = open(path[:-3] + '.tsv', encoding='utf-8').read().splitlines()
    assert rows[0].startswith('identifier\tsmiles') and len(rows) == made + 1
