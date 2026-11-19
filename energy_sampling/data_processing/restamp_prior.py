"""Re-score a pre-noised prior file (build_noised_prior.py) under a run's energy when the ONLY identity difference is
the LJ cutoff switch (analyze_kwargs.lj_envelope; the trainer's eLJ is switched by default since 2026-11-19), and
write it with the run's identity: the same rows, tags and order, each energy term replaced by its re-scored value.

    python data_processing/restamp_prior.py IN.pt OUT.pt --config CONFIG.yaml [--set KEY=VALUE ...] [--device cuda]
        [--chunk 4096] [--forces]

Refuses a file that differs from the run in anything else (rebuild that), a stale-pointer file
(noised_prior.refuse_stale_atom_pointers), a row the run's chart drops, and a crystal parameter that moves on
re-scoring (the rows are written as the analysis returns them, so a re-score reproduces them; a move means the row is
not the crystal the build scored). Every graph-level float column the analysis returns is replaced, the crystal
parameters and the build's tags excepted (a stored ``force`` column is re-taken under the run's energy); the manifest
records each column's largest change. Never overwrites OUT.
Cost: one scoring pass over every row (build_noised_prior.score).
"""
import argparse
import datetime
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
    a = ap.parse_args(argv)
    from inn_modeller import apply_overrides
    from train import energy_function_kwargs
    from energies.molecular_crystal import MolecularCrystal
    from utils import get_train_args
    from data_processing.build_noised_prior import score, git_commit

    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} exists: this tool never overwrites')
    t0 = time.time()
    f = torch.load(a.src, weights_only=False, map_location='cpu')
    if f['prior'] is not f['equalized_prior']:
        raise SystemExit(f'{a.src}: prior and equalized_prior differ; this tool restamps builds only')
    b, manifest = f['prior'], dict(f[NP.MANIFEST_KEY])
    NP.refuse_stale_atom_pointers(b, a.src)
    run_args = get_train_args(['--config', a.config])
    apply_overrides(run_args, a.overrides)
    ef = MolecularCrystal(**energy_function_kwargs(run_args, a.device))
    tsf = f.get('thermal_scaling_factor')
    if tsf is not None:
        ef.lj_coeff = float(tsf)
    stored, current = manifest['identity'], NP.identity(run_args, ef)
    if not NP.restamp_hint(stored, current, a.src):
        raise SystemExit(f'{a.src}: the file and the run differ in more than the LJ cutoff switch (or not at all): '
                         + ('; '.join(NP.identity_mismatch(stored, current)) or 'identical identity'))
    T = float(run_args.energy_config.temperature)
    n = int(b.num_graphs)
    take_forces = a.forces or 'force' in b.keys()
    cols, worst = None, {}
    for lo in range(0, n, a.chunk):
        idx = torch.arange(lo, min(lo + a.chunk, n))
        part = b.subsample_new_batch(idx)
        scored, kept = score(ef, part, T, a.device, a.chunk, forces=take_forces)   # a stored force is re-taken
        if not bool(kept.all()):
            raise SystemExit(f'{a.src}: rows {idx[~kept][:5].tolist()} have no chart point under this run')
        s = scored[0] if len(scored) == 1 else None
        if s is None:
            from data_processing.build_noised_prior import collate
            s = collate(scored)
        if cols is None:
            cols = [k for k in s.keys() if k in b.keys() and torch.is_tensor(s[k]) and torch.is_tensor(b[k])
                    and s[k].dim() >= 1 and s[k].shape[0] == s.num_graphs and b[k].shape[0] == n
                    and s[k].is_floating_point() and k not in PARAMETER_KEYS and not k.startswith(TAG_PREFIXES)]
            if take_forces and 'force' not in b.keys():   # a new column: no stored value to compare
                cols = [k for k in cols if k != 'force']
                new_force = torch.empty((n, s.force.shape[1]), dtype=torch.float32)
            new = {k: torch.empty_like(b[k]) for k in cols}
        for k in PARAMETER_KEYS:
            if k in s.keys() and k in b.keys():
                d = float((s[k].double() - part[k].double()).abs().max())
                if d > PARAM_TOL:
                    raise SystemExit(f'{a.src}: {k} moves by {d:.3g} on re-scoring rows {lo}..: not the crystals the '
                                     f'build scored')
        for k in cols:
            v = s[k].to(new[k].dtype).reshape(new[k][idx].shape)
            d = (v.double() - b[k][idx].double()).abs()
            d = d[torch.isfinite(d)]
            worst[k] = max(worst.get(k, 0.0), float(d.max()) if d.numel() else 0.0)
            new[k][idx] = v
        if take_forces and 'force' not in b.keys():
            new_force[idx] = s.force.float()
        if (lo // a.chunk) % 50 == 0:
            print(f'  {min(lo + a.chunk, n):,} / {n:,} rows re-scored, {time.time() - t0:.0f} s', flush=True)
    for k in cols:
        b[k] = new[k]
    if take_forces and 'force' not in b.keys():
        b.add_graph_attr(new_force, 'force')
    if take_forces:
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
    print(f'wrote {a.out}: {n:,} rows re-scored under {current["analyze_kwargs"]}; largest changes '
          + ', '.join(f'{k} {v:.3g}' for k, v in big) + f'; {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
