"""Merge pre-noised prior files (build_noised_prior.py) into one, for a run that reads them as one prior_path: the
shards of one build split by molecule (a Slurm array over band-set lists), or one molecule set built in different
space groups.

    python data_processing/merge_noised_priors.py OUT.pt IN1.pt IN2.pt [...]
    python data_processing/merge_noised_priors.py OUT.pt --list FILES.txt      (one input path per line)

Rows are concatenated in input order (energies.molecular_crystal.merge_batches); each later file's noised_parent
indices are offset by the rows before it (an anchor's parent is itself). Refuses unless every file agrees on the
thermal_scaling_factor and on the manifest keys the trainer reads (format_version, k, child_kernel, kick_units, log_lo,
log_hi, basin_kernel, temperature, window_kT_target, chart, heldout_frac) and on every identity field but space_groups
/ z_primes (their union is written). Refuses a condition (identifier, space group, Z') in two files: a build holds
each condition's anchors whole (its floor, its never-held-out lowest anchor), so shards must split by molecule. The
manifest is the first file's, with identity.space_groups / z_primes the union, counts summed (children_within_window:
the row-weighted mean), seed and basin_set as per-file lists, and every source manifest under 'merged_from' with its
path. Never overwrites OUT; writes it atomically."""
import argparse
import copy
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import noised_prior as NP  # noqa: E402

READ = ('format_version', 'k', 'child_kernel', 'kick_units', 'log_lo', 'log_hi', 'basin_kernel', 'temperature',
        'window_kT_target', 'chart', 'heldout_frac')
LAYOUT_ROWS = ('fd_state', 'fd_log_reference')   # layout-specific chart rows: the trainer attaches its own


def merge(paths, out_path):
    from energies.molecular_crystal import merge_batches
    if os.path.exists(out_path):
        raise SystemExit(f'{out_path} exists: this tool never overwrites')
    t0 = time.time()
    batches, mans, tsf, conds, n = [], [], set(), {}, 0
    for p in paths:
        f = torch.load(p, weights_only=False, map_location='cpu')
        if f['prior'] is not f['equalized_prior']:
            raise SystemExit(f'{p}: prior and equalized_prior are different objects; this tool merges builds only')
        m = f[NP.MANIFEST_KEY]
        if mans:
            for key in READ:
                if m.get(key) != mans[0].get(key):
                    raise SystemExit(f'{p}: manifest {key} {m.get(key)!r} differs from {paths[0]}: {mans[0].get(key)!r}')
            for k, v in m['identity'].items():
                if k not in ('space_groups', 'z_primes') and v != mans[0]['identity'].get(k):
                    raise SystemExit(f"{p}: identity {k} {v!r} differs from {paths[0]}: {mans[0]['identity'].get(k)!r}")
        tsf.add(float(f['thermal_scaling_factor']) if 'thermal_scaling_factor' in f else None)
        if len(tsf) > 1:
            raise SystemExit(f'{p}: thermal_scaling_factor differs across files: {tsf}')
        b = f['prior']
        NP.refuse_stale_atom_pointers(b, p)
        ids = [str(x) for x in b.identifier]
        sg = torch.as_tensor(b.sg_ind).reshape(-1).tolist()
        zp = torch.as_tensor(b.z_prime).reshape(-1).tolist()
        for key in set(zip(ids, sg, zp)):
            if key in conds:
                raise SystemExit(f'condition {key} is in {conds[key]} and {p}: shards must split by molecule')
            conds[key] = p
        b = copy.copy(b)   # a new store: the offset must not touch the loaded batch
        b.noised_parent = f['prior'].noised_parent + n
        for key in LAYOUT_ROWS:
            if key in b.keys():
                delattr(b, key)
        NP.strip_analysis_helpers(b)
        n += int(b.num_graphs)
        batches.append(b)
        mans.append(m)
        del f
    out = merge_batches(batches)
    ident = dict(mans[0]['identity'])
    for k in ('space_groups', 'z_primes'):
        ident[k] = sorted({int(v) for m in mans for v in m['identity'][k]})
    counts = {}
    for m in mans:
        for k, v in (m.get('counts') or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != 'children_within_window':
                counts[k] = counts.get(k, 0) + v
    w = [(m['counts'].get('children') or 0, m['counts'].get('children_within_window')) for m in mans]
    w = [(c, x) for c, x in w if c and x is not None and x == x]
    counts['children_within_window'] = (sum(c * x for c, x in w) / sum(c for c, _ in w)) if w else float('nan')
    manifest = dict(mans[0], identity=ident, counts=counts, seed=[m.get('seed') for m in mans],
                    basin_set=[m.get('basin_set') for m in mans], build_seconds=sum(m.get('build_seconds') or 0 for m in mans),
                    merged_from=[dict(m, path=os.path.abspath(p)) for m, p in zip(mans, paths)])
    payload = {'prior': out, 'equalized_prior': out, NP.MANIFEST_KEY: manifest}
    if tsf != {None}:
        payload['thermal_scaling_factor'] = tsf.pop()
    tmp = out_path + '.partial'
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    print(f'wrote {out_path}: {int(out.num_graphs)} rows from {len(paths)} files ({len(conds)} conditions); counts '
          f'{counts}; {os.path.getsize(out_path) / 1e9:.2f} GB, {time.time() - t0:.0f} s', flush=True)
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out')
    ap.add_argument('inputs', nargs='*')
    ap.add_argument('--list', default=None, help='a text file of input paths, one per line')
    a = ap.parse_args(argv)
    paths = list(a.inputs)
    if a.list:
        paths += [ln.strip() for ln in open(a.list) if ln.strip()]
    if len(paths) < 2:
        raise SystemExit('give at least two input files')
    merge(paths, a.out)


if __name__ == '__main__':
    main()
