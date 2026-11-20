"""Size-step scaffold split of training pool v1 (owner 2026-11-20): replaces the split column training_pool_build.py
writes, whose scaffold arm was drawn over the whole pool and so, inside a size rung, could be a few scaffolds (at <= 8
heavy atoms 43% of the arm was one, C1C2CC12).

Rungs are cumulative heavy-atom counts t = FIRST_RUNG .. max (the molecules with n_heavy <= t). Arms, all fixed by
sha1 hashes (training_pool_build.h) and the molecule table alone:
  heldout_scaffold  generic Murcko scaffolds. A scaffold is eligible when, at every rung, its molecules number at most
                    cap x the rung's molecules. Rungs are filled smallest first: eligible scaffolds with a molecule in
                    the rung, in hash order, are added until the arm holds held_frac of the rung (or the eligible
                    scaffolds run out: the rung's shortfall is reported). A scaffold is held at every size, so the arm
                    is scaffold-disjoint from train and heldout_random. Acyclic molecules are not eligible.
  heldout_random    held_frac of the remaining molecules by key hash (the rule of training_pool_build).
  train             the rest.

Built rungs: `--built N TSV` restricts the molecules with n_heavy <= N to the keys of TSV (the molecules that got a
conformer, training_pool_conformers.py), so the arms are drawn on the molecules a run can use. `--frozen PREV` keeps a
previous split: its held scaffolds stay held, scaffolds with a training or random molecule in it are not eligible, and
every key of PREV keeps its arm (checked). A later rung is built with the earlier split frozen.

usage, from energy_sampling/:
    python -m data_processing.training_pool_split OUT_DIR [--built N TSV ...] [--frozen PREV.csv.gz] [--cap 0.002]
writes OUT_DIR/pool_v1_split.csv.gz (key, n_heavy, generic_scaffold, split), OUT_DIR/pool_v1_split_counts.json, and per
--built TSV a copy with its split column replaced, OUT_DIR/<tsv stem>_split.tsv (band_from_slim.py --split reads it).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from data_processing.training_pool_build import DATA, h

HELD_FRAC, CAP, FIRST_RUNG = 0.05, 0.002, 8
ARMS = ('train', 'heldout_random', 'heldout_scaffold')


def size_step_split(df, held_frac=HELD_FRAC, cap=CAP, first_rung=FIRST_RUNG, frozen=None):
    """split (Series on df.index) and a per-rung report, for a table with key, n_heavy, generic_scaffold ('' acyclic).

    frozen: optional {key: split} of a previous split, kept exactly (ValueError when it cannot be)."""
    top = int(df.n_heavy.max())
    rungs = list(range(min(first_rung, top), top + 1))
    cols = range(0, top + 1)
    cyc = df[df.generic_scaffold != '']
    cnt = pd.crosstab(cyc.generic_scaffold, cyc.n_heavy).reindex(columns=cols, fill_value=0).cumsum(axis=1)
    npool = df.n_heavy.value_counts().reindex(cols, fill_value=0).cumsum()
    elig = pd.Series(True, index=cnt.index)
    for t in rungs:
        elig &= cnt[t] <= cap * npool[t]

    held, barred = set(), set()
    if frozen:
        f = df[df.key.isin(frozen.keys()) & (df.generic_scaffold != '')]
        fs = f.key.map(frozen)
        held = set(f.generic_scaffold[fs == 'heldout_scaffold'])
        barred = set(f.generic_scaffold[fs != 'heldout_scaffold'])
        both = held & barred
        if both:
            raise ValueError(f'frozen split is not scaffold-disjoint: {sorted(both)[:5]}')
    arm = cnt.loc[sorted(held)].sum() if held else pd.Series(0, index=cnt.columns)
    order = sorted((g for g in cnt.index[elig] if g not in held and g not in barred), key=lambda g: h('scaffold', g))
    report = []
    for t in rungs:
        target = held_frac * npool[t]
        for g in order:
            if arm[t] >= target:
                break
            if g in held or cnt.at[g, t] == 0:
                continue
            held.add(g)
            arm = arm + cnt.loc[g]
        report.append({'rung': t, 'molecules': int(npool[t]), 'arm': int(arm[t]), 'target': round(target, 1),
                       'shortfall': bool(arm[t] < target)})

    split = pd.Series('train', index=df.index)
    split[df.generic_scaffold.isin(held)] = 'heldout_scaffold'
    rnd = (df.key.map(lambda k: h('random', k) % 10000) < held_frac * 10000) & (split == 'train')
    split[rnd] = 'heldout_random'
    if frozen:
        old = df.key.map(frozen)
        bad = old.notna() & (old != split)
        if bad.any():
            raise ValueError(f'{int(bad.sum())} frozen keys change arm, e.g. {df.key[bad].iloc[0]}')
    return split, report, {'eligible': int(elig.sum()), 'ineligible': int((~elig).sum()), 'held': len(held)}


def arm_table(df, split, rungs):
    rows = []
    for t in rungs:
        m = df.n_heavy <= t
        a = df[m & (split == 'heldout_scaffold')]
        vc = a.generic_scaffold.value_counts()
        rows.append({'rung': f'<={t}', 'molecules': int(m.sum()), 'heldout_scaffold': len(a),
                     'heldout_random': int((m & (split == 'heldout_random')).sum()), 'scaffolds': len(vc),
                     'top1_pct': round(100 * vc.iloc[0] / len(a), 1) if len(a) else 0.0,
                     'top3_pct': round(100 * vc.iloc[:3].sum() / len(a), 1) if len(a) else 0.0,
                     'top1': vc.index[0] if len(a) else ''})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--pool', default=f'{DATA}/pool_v1.csv.gz')
    p.add_argument('--built', nargs=2, action='append', default=[], metavar=('N', 'TSV'))
    p.add_argument('--frozen', default=None)
    p.add_argument('--cap', type=float, default=CAP)
    p.add_argument('--held_frac', type=float, default=HELD_FRAC)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.pool, usecols=['key', 'n_heavy', 'generic_scaffold'], keep_default_na=False)
    counts = {'pool': len(df), 'cap': a.cap, 'held_frac': a.held_frac, 'first_rung': FIRST_RUNG, 'built': {}}
    built = []
    for n, tsv in a.built:
        n = int(n)
        keys = set(pd.read_csv(tsv, sep='\t', usecols=['key']).key)
        small = df.n_heavy <= n
        missing = keys - set(df.key[small])
        if missing:
            raise ValueError(f'{tsv}: {len(missing)} keys are not pool molecules with n_heavy <= {n}')
        df = df[~small | df.key.isin(keys)].reset_index(drop=True)
        counts['built'][tsv] = {'n_heavy_max': n, 'kept': len(keys), 'dropped': int(small.sum() - len(keys))}
        built.append(tsv)
    frozen = None
    if a.frozen:
        fz = pd.read_csv(a.frozen, usecols=['key', 'split'])
        frozen = dict(zip(fz.key, fz.split))

    split, report, info = size_step_split(df, a.held_frac, a.cap, FIRST_RUNG, frozen)
    df['split'] = split
    counts.update(info, molecules=len(df), split=split.value_counts().to_dict(), rungs=report,
                  arms=arm_table(df, split, [8, 9, 10, 12, int(df.n_heavy.max())]))
    df.to_csv(f'{a.out}/pool_v1_split.csv.gz', index=False, compression='gzip')
    smap = dict(zip(df.key, df.split))
    for tsv in built:
        t = pd.read_csv(tsv, sep='\t', keep_default_na=False)
        t['split'] = t.key.map(smap)
        stem = os.path.splitext(os.path.basename(tsv))[0]
        t.to_csv(f'{a.out}/{stem}_split.tsv', sep='\t', index=False)
    json.dump(counts, open(f'{a.out}/pool_v1_split_counts.json', 'w'), indent=1)
    print(pd.DataFrame(counts['arms']).to_string(index=False))
    short = [r['rung'] for r in report if r['shortfall']]
    print(f"held scaffolds {info['held']}, ineligible {info['ineligible']}; rungs short of {a.held_frac:.0%}: {short}")


if __name__ == '__main__':
    main()
