"""
The SMILES list a random-conformer crystal search runs on: a mixture of smiles_pool.py pools by explicit per-source
quotas, every QM9 molecule excluded, each molecule drawn once, and the list shuffled so that any prefix (any run of
search chunks) is a random sample of the whole mixture. Alternatively (--qm9), the QM9 molecules of the qm9full prior
themselves, with the prior's split, for the single-source variant (random conformers in place of QM9's DFT geometries).

Mixture. Sources are taken in the order given; each draws its quota uniformly (seeded) from its pool minus QM9 minus
what earlier sources already took, so a molecule found in several pools is drawn once and keeps every pool's name in
`sources`. A quota is a count or 'all'. Smaller, rarer pools go first, so the large ones cannot crowd them out.

QM9 (the exclusion set and --qm9) is canonical stereo-free SMILES -- the canonical form smiles_pool.py writes -- of the
GDB and the relaxed-geometry SMILES of every QM9 xyz file, and of the identifiers of the qm9full conditions files.

    python -m data_processing.smiles_mixture --pool D:/.../pool --quota csd=all zinc20A=all gdb11=300000 \
        pubchem=600000 --out D:/.../selection/new_oct22
    python -m data_processing.smiles_mixture --qm9 --out D:/.../selection/qm9_oct22

Split. Every mixture molecule carries a fixed side, 'held_out' when blake2b(salt, SMILES) mod 1000 < --holdout-permille
(default 50, the qm9full prior's 5%), so a molecule's side never depends on which molecules or how many search chunks a
later prior build gathers. QM9 molecules keep the qm9full prior's side.

write <out>/selected.tsv (smiles first, shuffled; mixture columns: the pools' own, then source, sources and split;
QM9 columns: heavy_atoms, split) and <out>/selection.json (arguments and counts); a mixture also writes
<out>/by_source/<source>.tsv, its rows of each source, shuffled on their own: the SMILES and their order in a source's
file do not change when a later source's quota does or a source is added after it (its `sources` column does gain the
added pool's name), so each source can be searched as its own chunk family and the shares chosen by how many of each
family's chunks run. Adding a source to a selection already searched means re-running every earlier source with the
same quotas, order and seed and checking their SMILES columns are unchanged. An existing <out> is refused.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

QM9_XYZ = r'D:\crystal_datasets\qm9_xyz'
PRIORS = r'D:\crystal_datasets\conditional\priors'
CONDITIONS = {'train': 'qm9full_conditions.pt', 'held_out': 'qm9full_test_conditions.pt'}


def split_of(smiles, salt, permille):
    h = int.from_bytes(hashlib.blake2b(f'{salt}\0{smiles}'.encode(), digest_size=8, person=b'rc_split').digest(), 'big')
    return 'held_out' if h % 1000 < permille else 'train'


def canon(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m, isomericSmiles=False)


def qm9_split(priors=PRIORS):
    """{canonical SMILES: 'train' | 'held_out'} of the qm9full conditions files. The prior split on SMILES as written,
    so a molecule written in two Kekule forms can be in both files (8 in qm9full); it is held out here."""
    import torch
    out = {}
    for split, name in CONDITIONS.items():
        d = torch.load(os.path.join(priors, name), weights_only=False, map_location='cpu')['prior']
        for s in d.identifier:
            c = canon(s)
            if c is not None and out.get(c) != 'held_out':
                out[c] = split
    return out


def qm9_reference(xyz_dir=QM9_XYZ, priors=PRIORS):
    """Every canonical SMILES QM9 is known by: GDB and relaxed SMILES of each xyz file, and the qm9full identifiers."""
    ref = set(qm9_split(priors))
    files = glob.glob(os.path.join(xyz_dir, '*.xyz'))
    if not files:
        raise SystemExit(f'no QM9 xyz files in {xyz_dir}')
    for f in files:
        with open(f) as fh:
            lines = fh.read().splitlines()
        for s in lines[int(lines[0]) + 3].split()[:2]:
            c = canon(s)
            if c is not None:
                ref.add(c)
    return ref


def read_pool(pool, source):
    """(header, [(smiles, rest of the line)]) of <pool>/<source>/pool.tsv; the rest is kept as text (a pool can hold
    tens of millions of rows)."""
    with open(os.path.join(pool, source, 'pool.tsv'), encoding='utf-8') as f:
        header = next(f).rstrip('\r\n').split('\t')
        assert header[0] == 'smiles', header
        return header, [tuple(line.rstrip('\r\n').split('\t', 1)) for line in f]


def mix(pools, quotas, exclude, seed):
    """pools: {source: [(smiles, rest)]}; quotas: [(source, n or None for all)] in draw order. The selection as
    [(smiles, *rest's fields, source, every pool holding it)], unshuffled, and the counts per source."""
    members = {src: {r[0] for r in rows} for src, rows in pools.items()}
    taken, out, counts = set(), [], {}
    for src, n in quotas:
        free = [r for r in pools[src] if r[0] not in exclude and r[0] not in taken]
        pick = free if n is None or n >= len(free) else random.Random(f'{seed}|{src}').sample(free, n)
        for smi, rest in pick:
            taken.add(smi)
            out.append((smi, *rest.split('\t'), src, ','.join(s for s in members if smi in members[s])))
        counts[src] = dict(pool=len(pools[src]), eligible=len(free), drawn=len(pick))
    return out, counts


def _table(path, header, rows):
    with open(path + '.tmp', 'w', encoding='utf-8', newline='\n') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')
    os.replace(path + '.tmp', path)


def write(out_dir, rows, header, meta, seed, by=None):
    """rows: tuples of text fields in header order; by: the column whose values each get their own file too, shuffled
    on their own (seed and value), so a source's file does not change when other sources' quotas do."""
    os.makedirs(out_dir)
    if by is not None:
        col = header.index(by)
        os.makedirs(os.path.join(out_dir, f'by_{by}'))
        for v in sorted({r[col] for r in rows}):
            part = sorted(r for r in rows if r[col] == v)
            random.Random(f'{seed}|shuffle|{v}').shuffle(part)
            _table(os.path.join(out_dir, f'by_{by}', f'{v}.tsv'), header, part)
    rows = sorted(rows)
    random.Random(f'{seed}|shuffle').shuffle(rows)
    _table(os.path.join(out_dir, 'selected.tsv'), header, rows)
    with open(os.path.join(out_dir, 'selection.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)


def parse_quota(q):
    src, _, n = q.partition('=')
    if not src or not n:
        raise SystemExit(f'quota {q!r}: expected source=count or source=all')
    return src, None if n == 'all' else int(n)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, help='a directory that must not exist')
    p.add_argument('--pool', help='the smiles_pool.py output directory')
    p.add_argument('--quota', nargs='+', default=[], help='source=count or source=all, in draw order')
    p.add_argument('--qm9', action='store_true', help='write the qm9full molecules with their split instead')
    p.add_argument('--qm9-xyz', default=QM9_XYZ)
    p.add_argument('--priors', default=PRIORS, help='the directory holding the qm9full conditions files')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split-salt', default='rc_oct22')
    p.add_argument('--holdout-permille', type=int, default=50)
    args = p.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f'{args.out} exists')
    meta = dict(argv=sys.argv[1:] if argv is None else list(argv), seed=args.seed)
    if args.qm9:
        split = qm9_split(args.priors)
        rows = sorted((s, Chem.MolFromSmiles(s).GetNumHeavyAtoms(), v) for s, v in split.items())
        meta['counts'] = {k: sum(v == k for v in split.values()) for k in CONDITIONS}
        write(args.out, rows, ['smiles', 'heavy_atoms', 'split'], meta, args.seed)
    else:
        if not args.pool or not args.quota:
            raise SystemExit('a mixture needs --pool and --quota')
        quotas = [parse_quota(q) for q in args.quota]
        if len({s for s, _ in quotas}) != len(quotas):
            raise SystemExit('a source is named twice')
        exclude = qm9_reference(args.qm9_xyz, args.priors)
        read = {s: read_pool(args.pool, s) for s, _ in quotas}
        headers = {tuple(h) for h, _ in read.values()}
        if len(headers) != 1:
            raise SystemExit(f'the pools have different columns: {headers}')
        rows, counts = mix({s: r for s, (_, r) in read.items()}, quotas, exclude, args.seed)
        rows = [(*r, split_of(r[0], args.split_salt, args.holdout_permille)) for r in rows]
        meta.update(qm9_reference=len(exclude), counts=counts, total=len(rows),
                    held_out=sum(r[-1] == 'held_out' for r in rows))
        write(args.out, rows, [*headers.pop(), 'source', 'sources', 'split'], meta, args.seed, by='source')
    print(json.dumps({k: v for k, v in meta.items() if k != 'argv'}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
