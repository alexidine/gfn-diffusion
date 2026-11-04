"""Build the frozen SMILES training pool v1 from the training_pool_census outputs (owner decisions of 2026-11-03).

Steps, in order, with each step's count written to pool_v1_counts.json:
  1. union of DATA/v3/csd, v3/zinc20A, v3/pubchem by key (non-standard InChIKey, -SRel -FixedH); the sources a key was
     found in are kept in `sources`. ZINC20 tranche B is not on disk and is not part of v1.
  2. isotope labels (2H, 3H, 13C, ...) are stripped and the molecule re-judged, so a labelled spelling merges into
     its parent.
  3. every unassigned stereo element (centre or double bond) is enumerated (RDKit EnumerateStereoisomers,
     onlyUnassigned, at most MAX_ISO per molecule) and each isomer re-judged; the -SRel key merges mirror images, so
     each surviving key is one diastereomer. The CSD table carries no stereo, so every CSD molecule with a stereo
     element is enumerated. Isomers that cannot be embedded are not filtered here (no embedding attempted).
  4. Day exclusion: drop every key whose standard-InChIKey first block (ik14) matches either spelling of any Day
     refcode (day_keys.csv).
  5. in_csd: the ik14 also occurs in the CSD table (some CSD entry has this compound as a component).
  6. split, fixed by sha1 hashes alone:
       heldout_scaffold  generic Murcko scaffolds taken in hash order until 5% of molecules, skipping any scaffold
                         holding more than SCAF_CAP of the pool; acyclic molecules are not eligible
       heldout_random    5% of the remaining molecules by key hash
       train             the rest
usage, from energy_sampling/: python -m data_processing.training_pool_build [workers]
reads DATA/v3/<source>_v3.csv.gz and DATA/day_keys.csv; writes DATA/pool_v1.csv.gz and DATA/pool_v1_counts.json
"""
import collections
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.Scaffolds import MurckoScaffold

from data_processing import training_pool_census as census

RDLogger.DisableLog('rdApp.*')
DATA = 'D:/crystal_datasets/paper_review_nov02/pool'
SOURCES = ['csd', 'zinc20A', 'pubchem']
MAX_ISO = 64
HELD_FRAC, SCAF_CAP = 0.05, 0.001
OPTS = StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=MAX_ISO)


def expand(item):
    """(rows, counter) for one (smiles, sources): isotope-stripped, stereo-enumerated, re-judged."""
    s, srcs = item
    cnt = collections.Counter()
    census.ALLOW_SEPARATED = 'csd' in srcs.split('|')
    m = Chem.MolFromSmiles(s)
    if m is None:
        cnt['parse'] += 1
        return [], cnt
    if any(a.GetIsotope() for a in m.GetAtoms()):
        for a in m.GetAtoms():
            a.SetIsotope(0)
        m = Chem.RemoveHs(m)
        cnt['isotope_stripped'] += 1
    isos = list(EnumerateStereoisomers(m, options=OPTS))
    cnt['enumerated' if len(isos) > 1 else 'not_enumerated'] += 1
    cnt['hit_max_iso'] += len(isos) >= MAX_ISO
    out = []
    for iso in isos:
        r, why = census.judge(Chem.MolToSmiles(iso))
        if r is None:
            cnt['rejudge_' + why] += 1
            continue
        r['sources'] = srcs
        r['stereo_enumerated'] = int(len(isos) > 1)
        out.append(r)
    return out, cnt


def generic(scaf):
    if not scaf:
        return ''
    try:
        g = MurckoScaffold.MakeScaffoldGeneric(Chem.MolFromSmiles(scaf))
        return Chem.MolToSmiles(g, isomericSmiles=False)
    except Exception:
        return ''


def h(salt, x):
    return int(hashlib.sha1(f'{salt}:{x}'.encode()).hexdigest()[:12], 16)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out_csv, out_json = f'{DATA}/pool_v1.csv.gz', f'{DATA}/pool_v1_counts.json'
    assert not os.path.exists(out_csv), out_csv
    counts = {}

    frames = []
    for src in SOURCES:
        d = pd.read_csv(f'{DATA}/v3/{src}_v3.csv.gz', keep_default_na=False, usecols=['key', 'smiles', 'ik14'])
        d['src'] = src
        counts[f'1_rows_{src}'] = len(d)
        frames.append(d)
    allr = pd.concat(frames, ignore_index=True)
    csd_ik = set(allr.loc[allr.src == 'csd', 'ik14'])
    srcs = allr.groupby('key').src.agg(lambda x: '|'.join(sorted(set(x))))
    first = allr.drop_duplicates('key').set_index('key')
    first['sources'] = srcs
    counts['1_union_keys'] = len(first)
    del allr, frames

    rows, cnt = {}, collections.Counter()
    items = list(zip(first.smiles.values, first.sources.values))
    with Pool(workers) as p:
        for k, (o, c) in enumerate(p.imap_unordered(expand, items, chunksize=2000)):
            cnt.update(c)
            for r in o:
                old = rows.get(r['key'])
                if old is None:
                    rows[r['key']] = r
                elif r['sources'] not in old['sources']:
                    old['sources'] = '|'.join(sorted(set(old['sources'].split('|')) | set(r['sources'].split('|'))))
            if k % 200000 == 0:
                print(k, len(rows), flush=True)
    counts['2_3_expand'] = dict(cnt)
    df = pd.DataFrame(list(rows.values()))
    counts['3_keys_after_isotope_and_stereo'] = len(df)
    counts['3_keys_with_unassigned_centre_left'] = int((df.n_stereo_unassigned > 0).sum())

    dk = pd.read_csv(f'{DATA}/day_keys.csv', keep_default_na=False)
    day_ik = (set(dk.ik14_perceived) | set(dk.ik14_csd)) - {''}
    drop = df.ik14.isin(day_ik)
    counts['4_day_excluded_keys'] = int(drop.sum())
    counts['4_day_refcodes_matched'] = int((dk.ik14_perceived.isin(df.ik14) | dk.ik14_csd.isin(df.ik14)).sum())
    df = df[~drop].reset_index(drop=True)

    df['in_csd'] = df.ik14.isin(csd_ik).astype(int)
    counts['5_in_csd'] = int(df.in_csd.sum())

    df['generic_scaffold'] = [generic(s) for s in df.scaffold.values]
    n = len(df)
    sizes = df.loc[df.generic_scaffold != '', 'generic_scaffold'].value_counts()
    held_scaf, total = set(), 0
    for g in sorted(sizes.index, key=lambda g: h('scaffold', g)):
        if total >= HELD_FRAC * n:
            break
        if sizes[g] <= SCAF_CAP * n:
            held_scaf.add(g)
            total += sizes[g]
    split = pd.Series('train', index=df.index)
    split[df.generic_scaffold.isin(held_scaf)] = 'heldout_scaffold'
    rnd = (df.key.map(lambda k: h('random', k) % 10000) < HELD_FRAC * 10000) & (split == 'train')
    split[rnd] = 'heldout_random'
    df['split'] = split
    counts['6_split'] = df.split.value_counts().to_dict()
    counts['6_heldout_scaffolds'] = len(held_scaf)
    counts['6_acyclic_molecules'] = int((df.generic_scaffold == '').sum())

    counts['final'] = {'molecules': n, 'rigid (0 rotatable)': int((df.n_rot == 0).sum()),
                       'one torsion': int((df.n_rot == 1).sum()),
                       'strict (0 rotatable, no CH3/OH/NH2)': int(((df.n_rot == 0) & (df.rotor == 0)).sum()),
                       'by source (found in)': {s: int(df.sources.str.contains(s).sum()) for s in SOURCES}}
    cols = ['key', 'smiles', 'ik14', 'n_rot', 'rotor', 'n_heavy', 'mw', 'n_rings', 'n_arom_rings', 'n_stereo',
            'stereo_enumerated', 'scaffold', 'generic_scaffold', 'sources', 'in_csd', 'split']
    df[cols].to_csv(out_csv, index=False, compression='gzip')
    json.dump(counts, open(out_json, 'w'), indent=1)
    print(json.dumps(counts, indent=1))


if __name__ == '__main__':
    main()
