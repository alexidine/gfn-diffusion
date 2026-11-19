"""Re-score a pre-noised prior file (build_noised_prior.py) under a run's energy when the ONLY identity difference is
the LJ cutoff switch (analyze_kwargs.lj_envelope; the trainer's eLJ is switched by default since 2026-11-19), and
write it with the run's identity: the same rows, tags and order, each energy term replaced by its re-scored value.

    python data_processing/restamp_prior.py IN.pt OUT.pt --config CONFIG.yaml [--set KEY=VALUE ...] [--device cuda]
        [--chunk 4096] [--forces]                                       one pass over every row
    ... --part K/N    re-score part K of N (rows K*n/N ..), writing only the new columns to OUT.parts/part_<K>.pt
                      (the source memory-mapped, so a task reads its own rows); an existing part is kept
    ... --assemble N  check that the N parts cover every row under one energy, then write OUT

Refuses a file that differs from the run in anything else (rebuild that), a stale-pointer file
(noised_prior.refuse_stale_atom_pointers), a row the run's chart drops, and a crystal parameter that moves on
re-scoring (the rows are written as the analysis returns them, so a re-score reproduces them; a move means the row is
not the crystal the build scored). Every graph-level float column the analysis returns is replaced, the crystal
parameters and the build's tags excepted (a stored ``force`` column is re-taken under the run's energy; --forces adds
one); the manifest records each column's largest change. Never overwrites OUT.
Cost: one scoring pass over every row (build_noised_prior.score).
"""
import argparse
import datetime
import glob
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import noised_prior as NP  # noqa: E402

PARAMETER_KEYS = ('cell_lengths', 'cell_angles', 'aunit_centroid', 'aunit_orientation', 'aunit_handedness')
TAG_PREFIXES = ('noised_', 'basin_')
PARAM_TOL = 1e-4
PART_FORMAT = 'restamp_part/1'


def load(src, mmap=False):
    f = torch.load(src, weights_only=False, map_location='cpu', mmap=mmap)
    if f['prior'] is not f['equalized_prior']:
        raise SystemExit(f'{src}: prior and equalized_prior differ; this tool restamps builds only')
    NP.refuse_stale_atom_pointers(f['prior'], src)
    return f


def energy(a, f):
    """(run_args, energy function at the file's eLJ scale, stored identity, run identity); refuses a file whose
    identity differs from the run's in more than the LJ cutoff switch."""
    from inn_modeller import apply_overrides
    from train import energy_function_kwargs
    from energies.molecular_crystal import MolecularCrystal
    from utils import get_train_args
    run_args = get_train_args(['--config', a.config])
    apply_overrides(run_args, a.overrides)
    ef = MolecularCrystal(**energy_function_kwargs(run_args, a.device))
    if f.get('thermal_scaling_factor') is not None:
        ef.lj_coeff = float(f['thermal_scaling_factor'])
    stored, current = f[NP.MANIFEST_KEY]['identity'], NP.identity(run_args, ef)
    if not NP.restamp_hint(stored, current, a.src):
        raise SystemExit(f'{a.src}: the file and the run differ in more than the LJ cutoff switch (or not at all): '
                         + ('; '.join(NP.identity_mismatch(stored, current)) or 'identical identity'))
    return run_args, ef, stored, current


def rescore(b, lo, hi, ef, T, a, take_forces, t0):
    """({column: [hi - lo, ...] new values}, force [hi - lo, w] or None, {column: largest |change|}) of rows lo..hi."""
    from data_processing.build_noised_prior import collate, score
    n = int(b.num_graphs)
    cols, new, worst, force = None, {}, {}, []
    for c0 in range(lo, hi, a.chunk):
        idx = torch.arange(c0, min(c0 + a.chunk, hi))
        part = b.subsample_new_batch(idx)
        scored, kept = score(ef, part, T, a.device, a.chunk, forces=take_forces)
        if not bool(kept.all()):
            raise SystemExit(f'{a.src}: rows {idx[~kept][:5].tolist()} have no chart point under this run')
        s = scored[0] if len(scored) == 1 else collate(scored)
        if cols is None:
            cols = [k for k in s.keys() if k in b.keys() and torch.is_tensor(s[k]) and torch.is_tensor(b[k])
                    and s[k].dim() >= 1 and s[k].shape[0] == s.num_graphs and b[k].shape[0] == n
                    and s[k].is_floating_point() and k not in PARAMETER_KEYS and not k.startswith(TAG_PREFIXES)
                    and k != 'force']
            new = {k: [] for k in cols}
        for k in PARAMETER_KEYS:
            if k in s.keys() and k in b.keys():
                d = float((s[k].double() - part[k].double()).abs().max())
                if d > PARAM_TOL:
                    raise SystemExit(f'{a.src}: {k} moves by {d:.3g} on re-scoring rows {c0}..: not the crystals the '
                                     f'build scored')
        for k in cols:
            v = s[k].to(b[k].dtype).reshape((len(idx),) + tuple(b[k].shape[1:]))
            d = (v.double() - part[k].double()).abs()
            d = d[torch.isfinite(d)]
            worst[k] = max(worst.get(k, 0.0), float(d.max()) if d.numel() else 0.0)
            new[k].append(v)
        if take_forces:
            force.append(s.force.float())
        if ((c0 - lo) // a.chunk) % 50 == 0:
            print(f'  rows {lo:,}..{min(c0 + a.chunk, hi):,} of {lo:,}..{hi:,} re-scored, {time.time() - t0:.0f} s',
                  flush=True)
    return {k: torch.cat(v) for k, v in new.items()}, (torch.cat(force) if take_forces else None), worst


def write(a, f, cols, force, worst, ef, stored, current, t0):
    from data_processing.build_noised_prior import git_commit
    b, manifest = f['prior'], dict(f[NP.MANIFEST_KEY])
    for k, v in cols.items():
        b[k] = v
    if force is not None:
        if 'force' in b.keys():
            b['force'] = force
        else:
            b.add_graph_attr(force, 'force')
        manifest['forces'] = dict(column='force', of='mol_energy',
                                  wrt='the FD chart point (noised_prior.fd_chart_points)',
                                  units='run energy units per chart unit',
                                  lj_envelope=ef.analyze_kwargs.get('lj_envelope'))
    manifest.update(identity=current, restamped=dict(
        from_identity=stored, source=os.path.abspath(a.src), largest_change=worst,
        built=datetime.datetime.now().isoformat(timespec='seconds'), seconds=round(time.time() - t0, 1),
        commits=dict(gfn=git_commit(HERE), mxtaltools=git_commit(os.path.dirname(os.path.dirname(
            __import__('mxtaltools').__file__))))))
    payload = dict(f, prior=b, equalized_prior=b)
    payload[NP.MANIFEST_KEY] = manifest
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(payload, a.out + '.partial')
    os.replace(a.out + '.partial', a.out)
    big = sorted(worst.items(), key=lambda kv: -kv[1])[:6]
    print(f'wrote {a.out}: {int(b.num_graphs):,} rows re-scored under {current["analyze_kwargs"]}; largest changes '
          + ', '.join(f'{k} {v:.3g}' for k, v in big) + f'; {time.time() - t0:.0f} s', flush=True)


def part_range(n, k, parts):
    return (n * k) // parts, (n * (k + 1)) // parts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[], dest='overrides', metavar='KEY=VALUE')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--forces', action='store_true',
                    help="also store each row's force (build_noised_prior --forces) when the file has none")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--part', default=None, help='K/N: re-score part K of N into OUT.parts/')
    mode.add_argument('--assemble', type=int, default=None, help='N: write OUT from the N parts in OUT.parts/')
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} exists: this tool never overwrites')
    t0 = time.time()
    parts_dir = a.out + '.parts'

    if a.part is not None:
        k, parts = (int(x) for x in a.part.split('/'))
        path = os.path.join(parts_dir, f'part_{k:04d}.pt')
        if os.path.exists(path):
            print(f'{path} exists: kept', flush=True)
            return
        f = load(a.src, mmap=True)
        run_args, ef, stored, current = energy(a, f)
        b = f['prior']
        take_forces = a.forces or 'force' in b.keys()
        lo, hi = part_range(int(b.num_graphs), k, parts)
        cols, force, worst = rescore(b, lo, hi, ef, float(run_args.energy_config.temperature), a, take_forces, t0)
        os.makedirs(parts_dir, exist_ok=True)
        torch.save(dict(format=PART_FORMAT, src=os.path.abspath(a.src), src_bytes=os.path.getsize(a.src),
                        n=int(b.num_graphs), lo=lo, hi=hi, parts=parts, identity=current, cols=cols, force=force,
                        worst=worst), path + '.partial')
        os.replace(path + '.partial', path)
        print(f'{path}: rows {lo:,}..{hi:,} re-scored, {time.time() - t0:.0f} s', flush=True)
        return

    f = load(a.src)
    run_args, ef, stored, current = energy(a, f)
    b = f['prior']
    n = int(b.num_graphs)
    if a.assemble is None:
        take_forces = a.forces or 'force' in b.keys()
        cols, force, worst = rescore(b, 0, n, ef, float(run_args.energy_config.temperature), a, take_forces, t0)
        write(a, f, cols, force, worst, ef, stored, current, t0)
        return
    paths = sorted(glob.glob(os.path.join(parts_dir, 'part_*.pt')))
    if len(paths) != a.assemble:
        raise SystemExit(f'{parts_dir}: {len(paths)} parts, expected {a.assemble}')
    cols, force, worst, nxt = {}, [], {}, 0
    for k, p in enumerate(paths):
        d = torch.load(p, weights_only=False, map_location='cpu')
        expect = part_range(n, k, a.assemble)
        if (d.get('format') != PART_FORMAT or d['n'] != n or d['parts'] != a.assemble or (d['lo'], d['hi']) != expect
                or d['src_bytes'] != os.path.getsize(a.src) or d['identity'] != current or d['lo'] != nxt):
            raise SystemExit(f'{p}: not part {k} of {a.assemble} of {a.src} under this run')
        nxt = d['hi']
        for c, v in d['cols'].items():
            cols.setdefault(c, []).append(v)
        if d['force'] is not None:
            force.append(d['force'])
        for c, v in d['worst'].items():
            worst[c] = max(worst.get(c, 0.0), v)
    if nxt != n or len({len(v) for v in cols.values()}) != 1 or force and len(force) != len(paths):
        raise SystemExit(f'{parts_dir}: the parts do not cover the {n:,} rows with one set of columns')
    write(a, f, {c: torch.cat(v) for c, v in cols.items()}, torch.cat(force) if force else None, worst, ef, stored,
          current, t0)


if __name__ == '__main__':
    main()
