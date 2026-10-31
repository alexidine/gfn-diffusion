"""Cut a train.py prior file into shards for a seeded streaming search: per shard, the molecules (one MolData per
identifier) and the seeds (every prior row of those molecules), so the search re-polishes the rows under its own
energy and writes the minima, with Hessians, as a basin set (mxtaltools crystal_search/stream_search.py, keys
``seeds``, ``basin_set``; docs/design/basin_set.md, the transfer route).

From ``energy_sampling/``, CPU only:

    python -m data_processing.prior_to_seeds SRC.pt --out DIR [--per_shard 2000] [--shards 0 1 ...]
        [--batch_key equalized_prior]

writes ``DIR/shard_<k>_mols.pt`` (list of MolData: z, pos and x of the identifier's first row, which every row of a
train.py prior shares at the orient_molecule(mode='std') fixed point; identifier and smiles), ``DIR/shard_<k>_seeds.pt``
(list of the rows, as stored) and ``DIR/manifest.json`` (source, layout, per-shard counts and identifier range).
Identifiers are sorted as strings and cut into consecutive shards of ``per_shard``, the order a stream run takes its
conditions in. A search config then names a shard: ``mol_path`` the mols file, ``sampling_mode: all``,
``stream.seeds`` the seeds file, ``num_samples`` the random starts at most per molecule after the seeds (0: seeds
only), ``stream.quota`` and ``stream.basin_set: true``. An existing output file is never overwritten.
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _ident(v) -> str:
    return str(v[0] if isinstance(v, (list, tuple)) and len(v) else v)


def shard_plan(identifiers, per_shard: int):
    """[(identifiers, row indices)] per shard: the distinct identifiers sorted as strings, cut into consecutive groups
    of ``per_shard``; each shard's row indices are every row of its identifiers, in row order."""
    if per_shard < 1:
        raise ValueError(f'per_shard must be positive, got {per_shard}')
    rows_of = {}
    for i, v in enumerate(identifiers):
        rows_of.setdefault(_ident(v), []).append(i)
    ids = sorted(rows_of)
    plan = []
    for lo in range(0, len(ids), per_shard):
        group = ids[lo:lo + per_shard]
        plan.append((group, sorted(i for g in group for i in rows_of[g])))
    return plan


def molecule_of(row, identifier: str):
    """A MolData of a prior row's molecule (z, pos, x as stored), with its identifier and smiles."""
    from mxtaltools.dataset_utils.data_classes import MolData
    smiles = getattr(row, 'smiles', None)
    smiles = smiles[0] if isinstance(smiles, (list, tuple)) else smiles
    return MolData(z=row.z.clone(), pos=row.pos.clone().float(), x=row.x.clone(), identifier=identifier,
                   smiles=smiles, do_mol_analysis=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('source')
    p.add_argument('--out', required=True)
    p.add_argument('--per_shard', type=int, default=2000)
    p.add_argument('--shards', type=int, nargs='*', default=None, help='shard indices to write (default: all)')
    p.add_argument('--batch_key', default='equalized_prior')
    a = p.parse_args(argv)
    t0 = time.time()
    src = torch.load(a.source, weights_only=False, map_location='cpu')
    batch = src[a.batch_key] if isinstance(src, dict) else src
    plan = shard_plan(batch.identifier, a.per_shard)
    todo = range(len(plan)) if a.shards is None else a.shards
    os.makedirs(a.out, exist_ok=True)
    width = max(4, len(str(len(plan) - 1)))
    shards = {}
    for k in todo:
        ids, idx = plan[k]
        mols_path = os.path.join(a.out, f'shard_{k:0{width}d}_mols.pt')
        seeds_path = os.path.join(a.out, f'shard_{k:0{width}d}_seeds.pt')
        for f_ in (mols_path, seeds_path):
            if os.path.exists(f_):
                raise SystemExit(f'{f_} exists; not overwritten')
        rows = batch.subsample_new_batch(torch.tensor(idx)).batch_to_list()
        first = {}
        for r in rows:
            first.setdefault(_ident(r.identifier), r)
        mols = [molecule_of(first[i], i) for i in ids]
        torch.save(mols, mols_path)
        torch.save(rows, seeds_path)
        shards[str(k)] = dict(mols=os.path.basename(mols_path), seeds=os.path.basename(seeds_path),
                              molecules=len(ids), seeds_count=len(rows), first=ids[0], last=ids[-1])
        print(f'shard {k}: {len(ids)} molecules, {len(rows)} seeds')
    man_path = os.path.join(a.out, 'manifest.json')
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    if manifest and (manifest.get('source') != os.path.abspath(a.source) or manifest.get('per_shard') != a.per_shard):
        raise SystemExit(f'{man_path} describes another cut ({manifest.get("source")}, per_shard '
                         f'{manifest.get("per_shard")}); use another --out')
    st = os.stat(a.source)
    manifest.update(source=os.path.abspath(a.source), source_bytes=st.st_size, source_mtime=st.st_mtime,
                    batch_key=a.batch_key, per_shard=a.per_shard, n_shards=len(plan),
                    n_molecules=sum(len(g) for g, _ in plan), n_rows=int(batch.num_graphs),
                    frame=src.get('frame') if isinstance(src, dict) else None,
                    energy_function=src.get('energy_function') if isinstance(src, dict) else None)
    manifest.setdefault('shards', {}).update(shards)
    with open(man_path, 'w') as fh:
        json.dump(manifest, fh, indent=1)
    print(f'{len(shards)} shard(s) of {len(plan)} written to {a.out} in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
