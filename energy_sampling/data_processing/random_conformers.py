"""
Random valid conformers for crystal-search molecules, the Mo3ENet on-the-fly recipe with its defects fixed, and the
MolData lists prep_qm9_anchor_mols.py reads.

Per (SMILES, replicate j): RDKit AddHs; ETKDGv3 embedding seeded by sha1(SMILES|j) (one retry with random starting
coordinates); every rotor -- a SINGLE bond in no ring whose smaller side, hydrogens included, holds more than one atom,
methyl, amine and hydroxyl rotors included -- spun by an independent uniform angle in [-pi, pi), the smaller side
moving; the spun conformer kept only when it passes every validity check, else the angles redrawn (seeded by
SMILES|j:attempt) up to --max-attempts times. No force-field relief: the conformer is the embedded geometry with random
torsions. mxtaltools mol_building.smiles2conformer differs: it spins double-bond bridges too, spins only when a
methyl/amine/hydroxyl rotor exists, relieves clashes with short MMFF cycles, and its scramble path raises at HEAD.

Validity of a conformer (check): finite coordinates; H/C/N/O/F only; every pair two or more bonds apart farther than
1.2 A; every pair three or more bonds apart at least 0.60 of the sum of their van der Waals radii (H 1.2, C 1.7, N 1.6,
O 1.55, F 1.5 A); every bond within 0.10 A of its embedded length; the connectivity perceived from the coordinates
alone keeps every bond and adds none between atoms three or more bonds apart; the stereochemistry perceived from 3D
equal to the embedded conformer's; every non-cumulated double bond within 30 degrees of planar.

The MolData of a conformer carries z, pos, Gasteiger charges as x, the canonical stereo-free SMILES as smiles (the key
the held-out split groups on), identifier f'{smiles}_c{j}', the ALL-ATOM radius about the all-atom centroid (the QM9
files' convention; mol_analysis would store a heavy-atom radius), and mass and mol_volume from mol_analysis.

    python -m data_processing.random_conformers --smiles selected.tsv --n-conf 1 --shard-size 50000 --out DIR --tag fam
writes DIR/raw/<tag>_raw_shard<s>.pt (a list of MolData) and DIR/raw/<tag>_raw_shard<s>.tsv (one row per conformer:
identifier, smiles, stereo of the embedded conformer, heavy atoms, atoms, rotors, attempts) for every shard not yet
written, and DIR/raw/<tag>_failed_shard<s>.tsv (SMILES and reason). Nothing is overwritten.
"""
import argparse
import csv
import hashlib
import os
import sys
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDetermineBonds

RDLogger.DisableLog('rdApp.*')

ALLOWED_Z = {1, 6, 7, 8, 9}
VDW = {1: 1.2, 6: 1.7, 7: 1.6, 8: 1.55, 9: 1.5}   # mxtaltools.constants.atom_properties VDW_RADII
CLASH = 1.2            # A, pairs >= 2 bonds apart (the Mo3ENet relief criterion)
VDW_RATIO = 0.60       # d / (R_i + R_j), pairs >= 3 bonds apart
BOND_TOL = 0.10        # A
DB_TWIST = 30.0        # degrees


def seed_of(smi, tag):
    return int(hashlib.sha1(f'{smi}|{tag}'.encode()).hexdigest()[:8], 16) & 0x7fffffff


def rotors(mol):
    """[(fixed atom, pivot atom, bool mask of the moving side)] for every rotor: SINGLE, in no ring, smaller side
    (hydrogens included) holding more than one atom; the smaller side moves."""
    n = mol.GetNumAtoms()
    adj = [[nb.GetIdx() for nb in a.GetNeighbors()] for a in mol.GetAtoms()]
    out = []
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        seen, stack = {j}, [j]
        while stack:
            a = stack.pop()
            for c in adj[a]:
                if {a, c} == {i, j} or c in seen:
                    continue
                seen.add(c)
                stack.append(c)
        if len(seen) <= n - len(seen):
            small, piv, fixed = seen, j, i
        else:
            small, piv, fixed = set(range(n)) - seen, i, j
        if len(small) <= 1:
            continue
        mask = np.zeros(n, bool)
        mask[list(small)] = True
        out.append((fixed, piv, mask))
    return out


def rotate_about(pos, u, v, mask, theta):
    k = pos[u] - pos[v]
    k = k / np.linalg.norm(k)
    p = pos[mask] - pos[v]
    c, s = np.cos(theta), np.sin(theta)
    pos[mask] = p * c + np.cross(k, p) * s + np.outer(p @ k, k) * (1 - c) + pos[v]
    return pos


def pair_d(pos):
    d = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def _with_pos(mol, pos):
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, pos[i].tolist())
    return m


def stereo_3d(mol, pos):
    m = _with_pos(mol, pos)
    Chem.AssignStereochemistryFrom3D(m)
    return Chem.MolToSmiles(Chem.RemoveHs(m))


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1n) * b1n
    w = b2 - (b2 @ b1n) * b1n
    return np.degrees(np.arctan2(np.cross(b1n, v) @ w, v @ w))


def check(mol, pos, pos0, stereo0, topo, bonds):
    """The failed validity checks of a conformer (an empty list = valid)."""
    if not np.isfinite(pos).all():
        return ['nonfinite']
    fails = []
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if not set(z) <= ALLOWED_Z:
        fails.append('elements')
    D = pair_d(pos)
    nb2 = topo >= 2    # empty for a molecule whose atoms are all bonded to each other (n12n3n1n23, in ZINC)
    if nb2.any() and D[nb2].min() <= CLASH:
        fails.append('clash')
    nb3 = topo >= 3
    if nb3.any():
        R = np.array([VDW[x] for x in z])
        if (D / (R[:, None] + R[None, :]))[nb3].min() < VDW_RATIO:
            fails.append('vdw_ratio')
    D0 = pair_d(pos0)
    bi, bj = bonds[:, 0], bonds[:, 1]
    if np.abs(D[bi, bj] - D0[bi, bj]).max() > BOND_TOL:
        fails.append('bond_length')
    xyz = f'{len(z)}\n\n' + '\n'.join(f'{Chem.GetPeriodicTable().GetElementSymbol(zz)} {p[0]:.6f} {p[1]:.6f} '
                                      f'{p[2]:.6f}' for zz, p in zip(z, pos))
    mx = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(mx)
    perceived = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mx.GetBonds()}
    graph = {tuple(x) for x in bonds.tolist()}
    if not graph <= perceived:
        fails.append('bond_broken')
    if any(topo[i, j] >= 3 for (i, j) in perceived - graph):
        fails.append('new_bond')
    if stereo_3d(mol, pos) != stereo0:
        fails.append('stereo_changed')
    tw = 0.0
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if Chem.HybridizationType.SP in (mol.GetAtomWithIdx(a1).GetHybridization(),
                                         mol.GetAtomWithIdx(a2).GetHybridization()):
            continue
        n1 = [x.GetIdx() for x in mol.GetAtomWithIdx(a1).GetNeighbors() if x.GetIdx() != a2]
        n2 = [x.GetIdx() for x in mol.GetAtomWithIdx(a2).GetNeighbors() if x.GetIdx() != a1]
        for x in n1:
            for y in n2:
                phi = abs(_dihedral(pos[x], pos[a1], pos[a2], pos[y]))
                tw = max(tw, min(phi, 180 - phi))
    if tw > DB_TWIST:
        fails.append('db_twist')
    return fails


def conformer(smi, rep, max_attempts=20):
    """(H-explicit RDKit Mol with one valid random conformer, info) or (None, {'reason': ...})."""
    m_in = Chem.MolFromSmiles(smi)
    if m_in is None:
        return None, {'reason': 'parse'}
    mol = Chem.AddHs(m_in)
    if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= ALLOWED_Z:
        return None, {'reason': 'elements'}
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed_of(smi, rep)
    if AllChem.EmbedMolecule(mol, ps) < 0:
        ps.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, ps) < 0:
            return None, {'reason': 'embed'}
    pos0 = mol.GetConformer().GetPositions()
    topo = Chem.GetDistanceMatrix(mol)
    bonds = np.array(sorted(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()))
    stereo0 = stereo_3d(mol, pos0)
    rot = rotors(mol)
    last = None
    for att in range(max_attempts):
        rng = np.random.default_rng(seed_of(smi, f'{rep}:{att}'))
        pos = pos0.copy()
        for (u, v, mk), th in zip(rot, rng.uniform(-np.pi, np.pi, size=len(rot))):
            pos = rotate_about(pos, u, v, mk, th)
        last = check(mol, pos, pos0, stereo0, topo, bonds)
        if not last:
            return _with_pos(mol, pos), {'stereo': stereo0, 'attempts': att + 1, 'rotors': len(rot)}
        if not rot:
            break   # nothing to redraw: the embedded geometry itself fails
    return None, {'reason': 'invalid:' + ','.join(last or [])}


def moldata(smiles, mol, identifier):
    """The MolData prep_qm9_anchor_mols.py reads, from an H-explicit RDKit Mol with one conformer (scout recipe)."""
    import torch
    from mxtaltools.dataset_utils.data_classes import MolData
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    AllChem.ComputeGasteigerCharges(mol)
    x = torch.tensor([float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms()], dtype=torch.float32)
    x = torch.nan_to_num(x)
    radius = (pos.double() - pos.double().mean(0)).norm(dim=1).max().float()
    return MolData(z=z, pos=pos, x=x, smiles=smiles, identifier=identifier, radius=radius, do_mol_analysis=True)


def _shard(job):
    path, tsv_path, fail_path, smiles, n_conf, max_attempts = job
    import torch
    out, rows, failed = [], [], []
    for smi in smiles:
        for j in range(n_conf):
            try:   # one SMILES that raises must not take its shard (and the worker pool) down with it
                mol, info = conformer(smi, j, max_attempts)
            except Exception as e:
                mol, info = None, {'reason': f'error:{type(e).__name__}'}
            if mol is None:
                failed.append((smi, j, info['reason']))
                continue
            ident = f'{smi}_c{j}'
            try:
                out.append(moldata(smi, mol, ident))
            except Exception as e:   # an MXtalTools analysis failure drops the conformer, recorded
                failed.append((smi, j, f'moldata:{type(e).__name__}'))
                continue
            rows.append((ident, smi, info['stereo'], mol.GetNumHeavyAtoms(), mol.GetNumAtoms(), info['rotors'],
                         info['attempts']))
    tables = ((tsv_path, 'identifier\tsmiles\tstereo_embedded\theavy_atoms\tatoms\trotors\tattempts', rows),
              (fail_path, 'smiles\treplicate\treason', failed))
    for p, header, table in tables:
        with open(p + '.tmp', 'w', encoding='utf-8') as f:
            f.write(header + '\n')
            for r in table:
                f.write('\t'.join(map(str, r)) + '\n')
        os.replace(p + '.tmp', p)
    torch.save(out, path + '.tmp')
    os.replace(path + '.tmp', path)   # the .pt last: its presence marks the shard done
    return len(smiles), len(out), len(failed)


def read_smiles(path):
    with open(path, encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        header = next(r)
        col = header.index('smiles')
        return [row[col] for row in r if row]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--smiles', required=True, help="a TSV with a 'smiles' column (canonical, stereo-free)")
    p.add_argument('--out', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--n-conf', type=int, default=1, help='conformers per SMILES (replicates 0..n-1)')
    p.add_argument('--max-attempts', type=int, default=20)
    p.add_argument('--shard-size', type=int, default=50000)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args(argv)
    smiles = read_smiles(args.smiles)
    raw = os.path.join(args.out, 'raw')
    os.makedirs(raw, exist_ok=True)
    jobs = []
    for s, start in enumerate(range(0, len(smiles), args.shard_size)):
        path = os.path.join(raw, f'{args.tag}_raw_shard{s:03d}.pt')
        if os.path.exists(path):
            continue
        jobs.append((path, path[:-3] + '.tsv', os.path.join(raw, f'{args.tag}_failed_shard{s:03d}.tsv'),
                     smiles[start:start + args.shard_size], args.n_conf, args.max_attempts))
    print(f'{len(smiles)} SMILES, {len(jobs)} shards to do, {args.n_conf} conformers each', flush=True)
    tot = np.zeros(3, int)
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_shard, jobs, chunksize=1)):
            tot += res
            print(f'shard {i + 1}/{len(jobs)}: SMILES {tot[0]}, conformers {tot[1]}, failed {tot[2]}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
