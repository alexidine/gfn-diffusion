"""A small self-similar test family drawn from training pool v1: one Murcko scaffold, no symmetry, clear frame.

A molecule is kept when, in pool v1, it has no rotatable bond, 7-11 heavy atoms and the given Murcko scaffold, and:
  - its heavy-atom graph has no non-trivial automorphism (RDKit CanonicalRankAtoms, breakTies=False), so no point
    operation can map the molecule onto itself and each crystal is one chart point;
  - it carries no OH or NH (no hydrogen-bond donor whose H position a rigid-body search would have to fix);
  - on one MMFF94 conformer (ETKDG seed 0) its standard frame is clear and its heavy skeleton non-planar: the
    build_rc_prior.frame_margins margin (least axis overlap with the farthest heavy atom, or the relative gap between
    the two farthest) is at least MARGIN, and the RMS heavy-atom distance from the best plane is at least OOP.
test_split: 'heldout' for sha1(family:key) in the lowest HELD_FRAC, else 'train'. pool_split is pool v1's own.

usage, from energy_sampling/: python -m data_processing.training_pool_test_family NAME SCAFFOLD
  e.g. norbornane C1CC2CCC1C2   ->  DATA/test_families/<NAME>.csv
"""
import hashlib
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
DATA = 'D:/crystal_datasets/paper_review_nov02/pool'
MARGIN, OOP, HELD_FRAC = 0.02, 0.2, 0.2
DONOR = Chem.MolFromSmarts('[$([OX2H1]),$([NX3H2]),$([NX3H1])]')


def graph_asym(smi):
    m = Chem.MolFromSmiles(smi)
    r = list(Chem.CanonicalRankAtoms(m, breakTies=False))
    return len(set(r)) == len(r) and not m.HasSubstructMatch(DONOR)


def frame(smi):
    """(margin, rms out-of-plane in A) of the heavy atoms of one MMFF94 conformer, or (nan, nan)."""
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    if AllChem.EmbedMolecule(m, randomSeed=0) != 0:
        return np.nan, np.nan
    AllChem.MMFFOptimizeMolecule(m)
    p = m.GetConformer().GetPositions()[[a.GetAtomicNum() > 1 for a in m.GetAtoms()]]
    p = p - p.mean(0)
    r = np.linalg.norm(p, axis=1)
    rs = np.sort(r)[::-1]
    _, V = np.linalg.eigh(np.eye(3) * (p * p).sum() - p.T @ p)
    overlap = np.abs(V.T @ (p[np.argmax(r)] / r.max())).min()
    return float(min(overlap, (rs[0] - rs[1]) / rs[0])), float(np.linalg.svd(p, compute_uv=False)[-1] / np.sqrt(len(p)))


def main():
    name, scaffold = sys.argv[1], sys.argv[2]
    out = f'{DATA}/test_families/{name}.csv'
    assert not os.path.exists(out), out
    d = pd.read_csv(f'{DATA}/pool_v1.csv.gz', keep_default_na=False,
                    usecols=['key', 'smiles', 'ik14', 'n_rot', 'n_heavy', 'mw', 'scaffold', 'in_csd', 'split'])
    c = d[(d.n_rot == 0) & d.n_heavy.between(7, 11) & (d.scaffold == scaffold)].reset_index(drop=True)
    with Pool(8) as p:
        c = c[p.map(graph_asym, c.smiles, chunksize=200)].reset_index(drop=True)
        fr = p.map(frame, c.smiles, chunksize=50)
    c['frame_margin'] = [f[0] for f in fr]
    c['out_of_plane'] = [f[1] for f in fr]
    c = c[(c.frame_margin >= MARGIN) & (c.out_of_plane >= OOP)].copy()
    held = c.key.map(lambda k: int(hashlib.sha1(f'{name}:{k}'.encode()).hexdigest()[:12], 16) % 1000)
    c['test_split'] = np.where(held < HELD_FRAC * 1000, 'heldout', 'train')
    c = c.rename(columns={'split': 'pool_split'}).drop(columns=['n_rot', 'scaffold'])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    c.sort_values('key').to_csv(out, index=False)
    print(name, len(c), c.test_split.value_counts().to_dict(), out)


if __name__ == '__main__':
    main()
