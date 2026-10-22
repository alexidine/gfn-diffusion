"""
Distribution of the molecules in smiles_pool.py pools against QM9: heavy-atom histogram, rings (count, sizes,
aromatic share), element fractions, sp3 carbon share and functional-group prevalence, plus each pool's share of
molecules already in QM9. Each pool is read from <pool dir>/<source>/pool.tsv, sampled down to --n molecules (seeded);
QM9 is the identifiers (SMILES) of the qm9full conditions files.

    python -m data_processing.smiles_pool_stats --pool D:/.../pool --sources gdb11 zinc20A pubchem csd --out stats.tsv
"""
import argparse
import os
import pickle
import random
import sys
from collections import Counter

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog('rdApp.*')

GROUPS = {
    'alcohol': '[CX4][OX2H]', 'ether': '[OD2]([#6])[#6]', 'aldehyde': '[CX3H1](=O)[#6,#1]', 'ketone': '[#6][CX3](=O)[#6]',
    'carboxylic acid': 'C(=O)[OH]', 'ester': '[#6]C(=O)O[#6]', 'amide': 'C(=O)[NX3]', 'amine': '[NX3;!$(N-C=[O,N]);!$(N-a)]',
    'aniline-type N': '[NX3]-a', 'nitrile': 'C#N', 'nitro': '[N+](=O)[O-]', 'alkene': '[CX3]=[CX3]', 'alkyne': 'C#C',
    'imine': '[CX3]=[NX2]', 'C-F': 'C-F', 'aromatic N': 'n', 'aromatic O': 'o', 'epoxide/aziridine': '[O,N]1CC1',
}
PATTS = {k: Chem.MolFromSmarts(v) for k, v in GROUPS.items()}


def features(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    ri = m.GetRingInfo()
    sizes = [len(r) for r in ri.AtomRings()]
    z = Counter(a.GetAtomicNum() for a in m.GetAtoms())
    n = m.GetNumHeavyAtoms()
    f = {'heavy atoms': n, 'rings': len(sizes), 'has aromatic ring': float(any(
        all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r) for r in ri.AtomRings())),
         'acyclic': float(not sizes), 'fraction sp3 C': rdMolDescriptors.CalcFractionCSP3(m)}
    for k in (3, 4, 5, 6):
        f[f'has {k}-ring'] = float(k in sizes)
    f['has 7+-ring'] = float(any(s >= 7 for s in sizes))
    for zz, name in ((6, 'C'), (7, 'N'), (8, 'O'), (9, 'F')):
        f[f'frac {name}'] = z.get(zz, 0) / n
    for k, p in PATTS.items():
        f[k] = float(m.HasSubstructMatch(p))
    return f


def qm9_smiles():
    out = set()
    for name in ('qm9full_conditions.pt', 'qm9full_test_conditions.pt'):
        d = torch.load(os.path.join(r'D:\crystal_datasets\conditional\priors', name), weights_only=False)['prior']
        for s in d.identifier:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                Chem.RemoveStereochemistry(m)
                out.add(Chem.MolToSmiles(m, isomericSmiles=False))
    return sorted(out)


def pool_smiles(pool, source):
    with open(os.path.join(pool, source, 'pool.tsv'), encoding='utf-8') as f:
        next(f)
        return [line.split('\t', 1)[0] for line in f]


def summarise(smiles, n, seed):
    rng = random.Random(seed)
    pick = smiles if len(smiles) <= n else rng.sample(smiles, n)
    rows = [r for r in map(features, pick) if r is not None]
    keys = list(rows[0])
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    hist = Counter(r['heavy atoms'] for r in rows)
    for h in range(4, 10):
        out[f'share {h} heavy'] = hist.get(h, 0) / len(rows)
    out['n sampled'] = len(rows)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pool', required=True)
    p.add_argument('--sources', nargs='+', required=True)
    p.add_argument('--n', type=int, default=100000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True, help='a TSV path that must not exist')
    args = p.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f'{args.out} exists')
    q = qm9_smiles()
    qset = set(q)
    cols = {'QM9': summarise(q, args.n, args.seed)}
    cols['QM9']['unique molecules'] = len(q)
    cols['QM9']['share in QM9'] = 1.0
    for s in args.sources:
        sm = pool_smiles(args.pool, s)
        c = summarise(sm, args.n, args.seed)
        c['unique molecules'] = len(sm)
        c['share in QM9'] = sum(x in qset for x in sm) / len(sm)
        c['new to QM9'] = sum(x not in qset for x in sm)
        cols[s] = c
    keys = list(cols['QM9']) + ['new to QM9']
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('quantity\t' + '\t'.join(cols) + '\n')
        for k in keys:
            f.write(k + '\t' + '\t'.join(f"{cols[c].get(k, float('nan')):.4g}" for c in cols) + '\n')
    width = max(len(k) for k in keys)
    print('quantity'.ljust(width), *[c.rjust(11) for c in cols])
    for k in keys:
        print(k.ljust(width), *[f"{cols[c].get(k, float('nan')):11.4g}" for c in cols])


if __name__ == '__main__':
    main(sys.argv[1:])
