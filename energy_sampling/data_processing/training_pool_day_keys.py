"""Day exclusion keys for the training pool: both spellings of every Day refcode, stereo-free.

Spellings: the SMILES perceived from Day's own geometry (`key` in day_molecules.csv.gz, written by
condition_set_stats.py) and the CSD SMILES of the same refcode in csd_fd_v5. Each is reduced to canonical stereo-free
SMILES and the standard InChIKey's first block (ik14), the skeleton training_pool_build.py excludes by.

usage, from energy_sampling/: python -m data_processing.training_pool_day_keys [OUT]   (default DATA/day_keys.csv)
"""
import sys

import pandas as pd
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
DAY = 'D:/crystal_datasets/paper_review_nov02/data/day_molecules.csv.gz'
CSD = 'D:/crystal_datasets/csd_fd_v5/data.pt'
DATA = 'D:/crystal_datasets/paper_review_nov02/pool'


def stereo_free(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    if m is None:
        return None, None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m, isomericSmiles=False), Chem.MolToInchiKey(m)[:14]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f'{DATA}/day_keys.csv'
    dm = pd.read_csv(DAY)
    d = torch.load(CSD, weights_only=True, mmap=True)
    by = {}
    for i, s in zip((str(x).upper() for x in d['identifier']), d['columns']['smiles']):
        if isinstance(s, str):
            by.setdefault(i, s)
    rows = []
    for _, r in dm.iterrows():
        k1, i1 = stereo_free(r.key)
        k2, i2 = stereo_free(by.get(r.refcode.upper()))
        rows.append(dict(refcode=r.refcode, key_perceived=k1, key_csd=k2, ik14_perceived=i1, ik14_csd=i2,
                         n_heavy=r.n_heavy))
    pd.DataFrame(rows).to_csv(out, index=False)
    print('refcodes', len(rows), 'without any key', sum(1 for r in rows if not (r['key_perceived'] or r['key_csd'])))


if __name__ == '__main__':
    main()
