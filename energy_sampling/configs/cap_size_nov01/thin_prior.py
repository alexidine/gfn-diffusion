"""cap_size_nov01: write the 2- and 4-crystal-per-molecule priors from the uncapped rc_oct22 prior with
build_rc_prior.thin_rows (each molecule's lowest-eLJ crystal plus k - 1 others drawn by an identifier-seeded permutation,
so the cap-2 rows are a subset of the cap-4 rows). Per-row keys are subset, every other key is carried over. Each file
is written under a temporary name and hard-linked into place only if the name is still free, so a reader never sees a
partial file and a file once in place is never replaced: on the cluster's network filesystem, replacing a file another job
already has open gives that reader OSError 116 (stale file handle), which killed an arm of the first launch.
usage: python configs/cap_size_nov01/thin_prior.py SRC OUT_DIR"""
import os
import sys

import torch

from build_rc_prior import thin_rows

src, out_dir = sys.argv[1], sys.argv[2]
want = {k: os.path.join(out_dir, f'rcoct22i1_cap{k}_prior.pt') for k in (2, 4)}
if all(os.path.exists(p) for p in want.values()):
    print('caps present:', ', '.join(want.values()))
    sys.exit(0)
os.makedirs(out_dir, exist_ok=True)
d = torch.load(src, weights_only=False, map_location='cpu')
b = d['equalized_prior']
n = int(b.num_graphs)
for k, path in want.items():
    keep = torch.tensor(thin_rows(b, k), dtype=torch.long)
    sub = b.subsample_new_batch(keep)
    out = {}
    for key, v in d.items():
        if key in ('prior', 'equalized_prior'):
            out[key] = sub
        elif torch.is_tensor(v) and v.shape[:1] == (n,):
            out[key] = v[keep]
        elif isinstance(v, list) and len(v) == n:
            out[key] = [v[i] for i in keep.tolist()]
        else:
            out[key] = v
    out['n_structures'] = int(sub.num_graphs)
    out['rows_per_molecule_cap'] = k
    out['subset'] = dict(source=src, rule=f'build_rc_prior.thin_rows(k={k})', rows_in=n, rows_out=int(sub.num_graphs))
    if os.path.exists(path):
        print(f'cap {k}: present, kept -> {path}', flush=True)
        continue
    tmp = f'{path}.tmp{os.getpid()}'
    torch.save(out, tmp)
    try:
        os.link(tmp, path)   # fails if another job got there first; its file stays, so its readers keep their handle
        print(f'cap {k}: rows {n} -> {int(sub.num_graphs)} -> {path}', flush=True)
    except FileExistsError:
        print(f'cap {k}: written by another job meanwhile, kept -> {path}', flush=True)
    finally:
        os.remove(tmp)
