"""Calibrate the RDF identity tolerance against COMPACK (the benchmark protocol's identity knob,
``docs/design/generator_benchmark_protocol.md`` step 2.1.1), once per molecule landscape.

Pairs of (sample structure at any depth, reference minimum) are drawn from ``protocol_score``'s distances, the same
number from each of ``--bins`` equal-width RDF distance bins on [0, ``--d_max``], so the fit sees every distance
range. Each pair is compared with the CSD Python API's packing similarity at Day's tolerances (30% and 30 degrees) with a
``--shell``-molecule shell (``compack_endpoints.engine``). For each match level m in ``--levels`` (a match being at
least m of the shell overlaid), a logistic model of P(match | RDF distance) is fitted and the tolerance is the distance
at which it falls to ``--p``; 200 bootstrap refits give its 5-95% range. A comparison the engine cannot make (no
result, or an error such as bond perception failing on a clashing raw structure) is recorded without a verdict and
left out of the fits. Comparisons run in ``--workers`` processes and are appended as they finish, so a stopped run
resumes.

usage (CPU, licensed CSD Python API):
    python -m eval.day_bench.protocol_compack SCORE_NPZ OUT_DIR [--shell 20] [--levels 15,10] [--p 0.9] [--bins 14]
        [--per_bin 25] [--d_max 0.35] [--workers 12] [--seed 0]

Writes OUT_DIR/compack_calibration_pairs.csv and OUT_DIR/compack_calibration.csv (one row per match level).
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_ENGINE, _CRYSTALS, _CIF_DIR = None, {}, None
KEYS = ['structure', 'minimum', 'depth', 'rdf', 'nmatched', 'rmsd', 'seconds', 'error']


def _init(shell, cif_dir):
    global _ENGINE, _CIF_DIR
    from eval.day_bench.compack_endpoints import engine
    _ENGINE = engine()
    _ENGINE.settings.packing_shell_size = shell
    _CIF_DIR = cif_dir


def _crystal(path):
    """The CSD API crystal of ``path``; an extended-XYZ structure is first written as a CIF beside the pairs file."""
    if path not in _CRYSTALS:
        from eval.day_bench.compack_endpoints import read
        if path.endswith('.extxyz'):
            import hashlib
            from ase.io import read as ase_read, write
            cif = os.path.join(_CIF_DIR, hashlib.md5(path.encode()).hexdigest() + '.cif')
            if not os.path.exists(cif):
                atoms = ase_read(path)
                del atoms.arrays['bb_indices']
                write(cif, atoms, format='cif')
            path_read = cif
        else:
            path_read = path
        _CRYSTALS[path] = read(path_read)
    return _CRYSTALS[path]


def _compare(job):
    ps, pr, depth, d = job
    from eval.day_bench import compack_endpoints as ce
    t, err, n, rmsd = time.time(), '', None, None
    try:
        n, rmsd = ce.compare(_ENGINE, _crystal(ps), _crystal(pr), ps, pr)
    except Exception as e:  # a crystal the engine cannot read or perceive: no verdict, recorded
        err = f'{type(e).__name__}: {e}'[:120]
    return dict(structure=ps, minimum=pr, depth=depth, rdf=f'{d:.4f}', nmatched='' if n is None else n,
                rmsd='' if rmsd is None else f'{rmsd:.4f}', seconds=f'{time.time() - t:.2f}', error=err)


def logistic_tolerance(d, y, p):
    """Fit P(y = 1 | d) = 1 / (1 + exp(-(a + b d))) by maximum likelihood; return (a, b, the d at which P = p)."""
    from scipy.optimize import minimize

    def nll(w):
        z = w[0] + w[1] * d
        return np.sum(np.logaddexp(0, z) - y * z)
    w = minimize(nll, x0=np.array([3.0, -30.0]), method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-9,
                                                                                    maxiter=4000)).x
    return w[0], w[1], (np.log(p / (1 - p)) - w[0]) / w[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('score_npz')
    ap.add_argument('out_dir')
    ap.add_argument('--shell', type=int, default=20)
    ap.add_argument('--levels', default='15,10')
    ap.add_argument('--p', type=float, default=0.9)
    ap.add_argument('--bins', type=int, default=14)
    ap.add_argument('--per_bin', type=int, default=25)
    ap.add_argument('--d_max', type=float, default=0.35)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    z = np.load(a.score_npz)
    D, path, depth, ref_path = z['D'], z['path'].astype(str), z['depth'], z['ref_path'].astype(str)
    rng = np.random.default_rng(a.seed)
    edges = np.linspace(0, a.d_max, a.bins + 1)
    jobs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        i, j = np.nonzero((D > lo) & (D <= hi))
        pick = rng.choice(len(i), size=min(a.per_bin, len(i)), replace=False)
        jobs += [(path[i[k]], ref_path[j[k]], int(depth[i[k]]), float(D[i[k], j[k]])) for k in pick]
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, 'compack_calibration_pairs.csv')
    done = {(r['structure'], r['minimum']) for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    print(f'{len(jobs)} pairs over {a.bins} RDF bins on [0, {a.d_max}]; {len(done)} already compared, {len(todo)} to '
          f'run on {a.workers} workers (shell {a.shell})', flush=True)
    t0 = time.time()
    cif_dir = os.path.join(a.out_dir, 'cif_cache')
    os.makedirs(cif_dir, exist_ok=True)
    with open(out, 'a', newline='') as fh, ProcessPoolExecutor(a.workers, initializer=_init,
                                                               initargs=(a.shell, cif_dir)) as pool:
        w = csv.DictWriter(fh, fieldnames=KEYS)
        if not done:
            w.writeheader()
        futures = [pool.submit(_compare, j) for j in todo]
        for k, f in enumerate(as_completed(futures), 1):
            w.writerow(f.result())
            fh.flush()
            if k % 50 == 0 or k == len(futures):
                print(f'  {k} of {len(futures)} in {time.time() - t0:.0f} s', flush=True)

    rows = [r for r in csv.DictReader(open(out)) if r['nmatched'] != '']
    d = np.array([float(r['rdf']) for r in rows])
    n = np.array([int(r['nmatched']) for r in rows])
    res = []
    for m in (int(x) for x in a.levels.split(',')):
        y = (n >= m).astype(float)
        if y.min() == y.max():  # one outcome only: no transition to fit
            print(f'match at >= {m}/{a.shell}: {int(y.sum())} of {len(d)} pairs; no tolerance (one outcome only)')
            res.append(dict(level=f'{m}/{a.shell}', pairs=len(d), matches=int(y.sum()), tolerance=float('nan'),
                            tolerance_5=float('nan'), tolerance_95=float('nan'), intercept=float('nan'),
                            slope=float('nan')))
            continue
        a0, b0, tol = logistic_tolerance(d, y, a.p)
        boot = []
        for _ in range(200):
            k = rng.integers(0, len(d), len(d))
            boot.append(logistic_tolerance(d[k], y[k], a.p)[2])
        lo, hi = np.percentile(boot, [5, 95])
        res.append(dict(level=f'{m}/{a.shell}', pairs=len(d), matches=int(y.sum()), tolerance=round(float(tol), 4),
                        tolerance_5=round(float(lo), 4), tolerance_95=round(float(hi), 4), intercept=round(a0, 3),
                        slope=round(b0, 3)))
        print(f'match at >= {m}/{a.shell}: {int(y.sum())} of {len(d)} pairs; RDF tolerance at P = {a.p}: {tol:.4f} '
              f'(5-95% {lo:.4f}-{hi:.4f})', flush=True)
    unsettled = sum(1 for r in csv.DictReader(open(out)) if r['nmatched'] == '')
    print(f'{unsettled} pairs without a verdict (left out of the fits)', flush=True)
    with open(os.path.join(a.out_dir, 'compack_calibration.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)


if __name__ == '__main__':
    main()
