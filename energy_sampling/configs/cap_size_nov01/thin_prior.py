"""cap_size_nov01: write the 2- and 4-crystal-per-molecule priors from the uncapped rc_oct22 prior with
build_rc_prior.thin_rows (each molecule's lowest-eLJ crystal plus k - 1 others drawn by an identifier-seeded permutation,
so the cap-2 rows are a subset of the cap-4 rows). Per-row keys are subset, every other key is carried over. Each file
is written under a temporary name and moved into place, so a reader never sees a partial file and two jobs doing this at
once leave one complete copy.
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
    tmp = f'{path}.tmp{os.getpid()}'
    torch.save(out, tmp)
    os.replace(tmp, path)
    print(f'cap {k}: rows {n} -> {int(sub.num_graphs)} -> {path}', flush=True)
