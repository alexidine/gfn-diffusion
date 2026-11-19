"""Will merge_noised_priors.py accept these prior files together? A read-only check, seconds per file (memory-mapped).

    python configs/pool8sg4_nov20/check_priors.py FILE1.pt FILE2.pt [...]

Prints, per file, the rows, its space groups, whether it carries a force column (and its width), the thermal scaling
factor, and the manifest keys the merge compares (data_processing/merge_noised_priors.py READ, plus every identity
field but space_groups / z_primes). Then one line per compared key: 'same' or 'DIFFERS' with each file's value. Exits 1
on any difference the merge would refuse, so a job can stop before loading ~40M rows. The forces entry is compared as
the merge compares it (whole dict), with a note when only its 'width' differs."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'data_processing'))
import noised_prior as NP  # noqa: E402

READ = ('format_version', 'k', 'child_kernel', 'kick_units', 'log_lo', 'log_hi', 'basin_kernel', 'temperature',
        'window_kT_target', 'chart', 'heldout_frac', 'forces')


def summary(path):
    f = torch.load(path, weights_only=False, map_location='cpu', mmap=True)
    b, m = f['prior'], f[NP.MANIFEST_KEY]
    sg = torch.as_tensor(b.sg_ind).reshape(-1)
    force = getattr(b, 'force', None) if 'force' in b.keys() else None
    return dict(path=path, rows=int(b.num_graphs), groups=sorted(set(int(g) for g in torch.unique(sg).tolist())),
                first_rows_sg=sg[:8].tolist(), force=None if force is None else tuple(force.shape),
                tsf=f.get('thermal_scaling_factor'), read={k: m.get(k) for k in READ},
                identity={k: v for k, v in m['identity'].items() if k not in ('space_groups', 'z_primes')},
                same_object=f['prior'] is f['equalized_prior'])


def main(paths):
    s = [summary(p) for p in paths]
    for x in s:
        print(f"{x['path']}: {x['rows']:,} rows, groups {x['groups']}, first rows' groups {x['first_rows_sg']}, "
              f"force {x['force']}, thermal_scaling_factor {x['tsf']}, prior is equalized_prior {x['same_object']}")
    bad = []
    keys = [('read', k) for k in READ] + [('identity', k) for k in sorted({k for x in s for k in x['identity']})]
    for where, k in keys:
        vals = [x[where].get(k) for x in s]
        if all(v == vals[0] for v in vals):
            print(f'  same     {where}.{k} = {vals[0]!r}')
            continue
        note = ''
        if k == 'forces' and all(isinstance(v, dict) for v in vals):
            stripped = [{kk: vv for kk, vv in v.items() if kk != 'width'} for v in vals]
            if all(v == stripped[0] for v in stripped):
                note = '  (only the width differs)'
        print(f'  DIFFERS  {where}.{k}{note}: ' + ' | '.join(f'{os.path.basename(x["path"])}={v!r}' for x, v in zip(s, vals)))
        bad.append(f'{where}.{k}')
    if len({x['tsf'] for x in s}) > 1:
        print('  DIFFERS  thermal_scaling_factor')
        bad.append('thermal_scaling_factor')
    if any(x['force'] is None for x in s) and any(x['force'] is not None for x in s):
        print('  DIFFERS  some files carry a force column and others do not')
        bad.append('force column')
    if not all(x['same_object'] for x in s):
        bad.append('prior is not equalized_prior')
    print('merge would REFUSE: ' + ', '.join(bad) if bad else 'merge checks pass')
    return 1 if bad else 0


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1:]))
