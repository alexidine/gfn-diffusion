"""One MMFF94 conformer per molecule of training pool v1 up to a heavy-atom count, as MolData for the crystal search.

Per molecule (pool v1 row: key, smiles, split, n_heavy, n_rot):
  - ETKDGv3, N_CONF embeddings seeded by the key (sha1);
  - each optimised under MMFF94 (at most MAX_ITERS steps); the lowest-energy converged one is kept;
  - the kept conformer must give back the input's stereochemistry (canonical isomeric SMILES of the stereo assigned
    from 3D == of the input), else the molecule is dropped;
  - data_processing.random_conformers.moldata, identifier = pool key, smiles = pool smiles. MolData.radius is the
    all-atom max norm about the centroid (moldata's convention).
Mirror images share a pool key (InChIKey -SRel); the conformer is of the SMILES's own stereo. A centrosymmetric space
group holds both enantiomers, so one per key suffices there.
Molecules that fail (no embedding, no MMFF parameters, none converged, stereo changed) are listed with the reason.

usage, from energy_sampling/ (CPU only):
  python -m data_processing.training_pool_conformers MAX_HEAVY OUT_DIR [--workers 8]
writes OUT_DIR/pool<MAX_HEAVY>_all.pt (list of MolData, pool order), OUT_DIR/pool<MAX_HEAVY>_mols.tsv (key, smiles,
split, n_heavy, n_rot, mmff_kcal, n_converged) and OUT_DIR/pool<MAX_HEAVY>_failed.tsv (key, smiles, reason).
"""
import argparse
import hashlib
import os
from multiprocessing import Pool

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
POOL = 'D:/crystal_datasets/paper_review_nov02/pool/pool_v1.csv.gz'
N_CONF, MAX_ITERS = 8, 2000


def stereo_smiles(mol):
    m = Chem.RemoveHs(mol)
    Chem.AssignStereochemistryFrom3D(m)
    return Chem.MolToSmiles(m)


def build(row):
    key, smi = row
    from data_processing.random_conformers import moldata
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    ref = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3()
    p.randomSeed = int(hashlib.sha1(key.encode()).hexdigest()[:7], 16)  # < 2^31: RDKit takes a signed int
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=N_CONF, params=p))
    if not cids:
        return key, None, 'no_embedding', None, 0
    if not AllChem.MMFFHasAllMoleculeParams(m):
        return key, None, 'no_mmff_params', None, 0
    res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=MAX_ITERS)
    ok = [(e, c) for (nc, e), c in zip(res, cids) if nc == 0]
    if not ok:
        return key, None, 'not_converged', None, 0
    e, c = min(ok)
    keep = Chem.Mol(m)
    keep.RemoveAllConformers()
    keep.AddConformer(Chem.Conformer(m.GetConformer(c)), assignId=True)
    if stereo_smiles(keep) != ref:
        return key, None, 'stereo_changed', None, len(ok)
    try:
        md = moldata(smi, keep, key)
    except Exception as ex:  # an MXtalTools analysis failure drops the molecule, recorded
        return key, None, f'moldata:{type(ex).__name__}', None, len(ok)
    return key, md, None, float(e), len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('max_heavy', type=int)
    ap.add_argument('out_dir')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    import torch
    stem = f'{a.out_dir}/pool{a.max_heavy}'
    assert not os.path.exists(f'{stem}_all.pt'), f'{stem}_all.pt exists'
    os.makedirs(a.out_dir, exist_ok=True)
    d = pd.read_csv(POOL, keep_default_na=False, usecols=['key', 'smiles', 'split', 'n_heavy', 'n_rot'])
    d = d[d.n_heavy <= a.max_heavy].reset_index(drop=True)
    print(f'pool v1 molecules with at most {a.max_heavy} heavy atoms: {len(d)}', flush=True)
    with Pool(a.workers) as p:
        out = p.map(build, zip(d.key, d.smiles), chunksize=200)
    mols, rows, failed = [], [], []
    for (key, md, why, e, nok), r in zip(out, d.itertuples()):
        if md is None:
            failed.append((key, r.smiles, why))
        else:
            mols.append(md)
            rows.append((key, r.smiles, r.split, r.n_heavy, r.n_rot, e, nok))
    torch.save(mols, f'{stem}_all.pt')
    pd.DataFrame(rows, columns=['key', 'smiles', 'split', 'n_heavy', 'n_rot', 'mmff_kcal', 'n_converged']).to_csv(
        f'{stem}_mols.tsv', sep='\t', index=False)
    pd.DataFrame(failed, columns=['key', 'smiles', 'reason']).to_csv(f'{stem}_failed.tsv', sep='\t', index=False)
    print(f'kept {len(mols)}, failed {len(failed)}:', pd.Series([f[2] for f in failed]).value_counts().to_dict())


if __name__ == '__main__':
    main()
