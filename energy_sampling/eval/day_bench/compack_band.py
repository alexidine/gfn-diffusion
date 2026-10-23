"""COMPACK verdicts for the reference-minimum vs source-structure pairs whose RDF distance is ambiguous.

The envwise RDF distance identifies a structure with a reference minimum reliably below the calibrated cut (0.093 for
imidazole); between that cut and a looser one it mixes one minimum relaxed from two starts with distinct neighbouring
minima. Each pair in that band (``coverage_<energy>.npz``: a reference minimum against a structure of another source)
is compared with the CSD Python API's packing similarity under the Day duplicate rule (``compack_endpoints.engine``:
30-molecule shells, 30% and 30 degrees; a match is all 30 overlaid). Comparisons run in ``--workers`` processes, each
bounded by the engine's timeout; a timeout is written with no verdict, never as a non-match. Verdicts are appended to
``compack_band_<energy>.csv`` as they arrive, and pairs already there are skipped, so a stopped run resumes.

usage (CPU, licensed CSD Python API):
    python -m eval.day_bench.compack_band OUT_DIR REF_LABEL [--lo 0.093] [--hi 0.14] [--workers 12] [--energy exp6]
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from eval.day_bench.compack_endpoints import SHELL

_ENGINE, _CRYSTALS = None, {}


def _init():
    global _ENGINE
    from eval.day_bench.compack_endpoints import engine
    _ENGINE = engine()


def _crystal(path):
    if path not in _CRYSTALS:
        from eval.day_bench.compack_endpoints import read
        _CRYSTALS[path] = read(path)
    return _CRYSTALS[path]


def _compare(job):
    a, b, pa, pb, d = job
    from eval.day_bench import compack_endpoints as ce
    t = time.time()
    n, rmsd = ce.compare(_ENGINE, _crystal(pa), _crystal(pb), a, b)
    return dict(a=a, b=b, rdf=f'{d:.4f}', nmatched='' if n is None else n, rmsd='' if rmsd is None else f'{rmsd:.4f}',
                seconds=f'{time.time() - t:.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('ref_label')
    ap.add_argument('--lo', type=float, default=0.093)
    ap.add_argument('--hi', type=float, default=0.14)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--energy', default='exp6')
    a = ap.parse_args()
    z = np.load(os.path.join(a.out_dir, f'coverage_{a.energy}.npz'))
    D, names, kind, paths = z['D'], z['names'].astype(str), z['kind'].astype(str), z['paths'].astype(str)
    label = np.array([n.split(':', 1)[0] for n in names])
    ref = np.nonzero((label == a.ref_label) & (kind == 'sample'))[0]
    other = np.nonzero((label != a.ref_label) & (kind == 'sample'))[0]
    band = (D[np.ix_(ref, other)] > a.lo) & (D[np.ix_(ref, other)] <= a.hi)
    jobs = [(names[ref[i]], names[other[j]], paths[ref[i]], paths[other[j]], float(D[ref[i], other[j]]))
            for i, j in zip(*np.nonzero(band))]
    out = os.path.join(a.out_dir, f'compack_band_{a.energy}.csv')
    done = set()
    if os.path.exists(out):
        done = {(r['a'], r['b']) for r in csv.DictReader(open(out))}
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    per = {lab: int(sum(names[other[j]].startswith(lab + ':') for _, j in zip(*np.nonzero(band))))
           for lab in sorted(set(label[other]))}
    print(f'{len(jobs)} pairs between RDF {a.lo} and {a.hi} ({per}); {len(done)} already settled, {len(todo)} to run '
          f'on {a.workers} workers', flush=True)
    keys = ['a', 'b', 'rdf', 'nmatched', 'rmsd', 'seconds']
    new = not os.path.exists(out)
    t0, n_match, n_none = time.time(), 0, 0
    # jobs sorted by reference minimum keep each worker's crystal cache small
    with open(out, 'a', newline='') as fh, ProcessPoolExecutor(a.workers, initializer=_init) as pool:
        w = csv.DictWriter(fh, fieldnames=keys)
        if new:
            w.writeheader()
        futures = [pool.submit(_compare, j) for j in sorted(todo, key=lambda j: j[0])]
        for k, f in enumerate(as_completed(futures), 1):
            r = f.result()
            w.writerow(r)
            n_match += r['nmatched'] == SHELL
            n_none += r['nmatched'] == ''
            if k % 200 == 0 or k == len(futures):
                fh.flush()
                print(f'  {k} of {len(futures)} in {time.time() - t0:.0f} s; matched 30/30 {n_match}, unsettled '
                      f'{n_none}', flush=True)


if __name__ == '__main__':
    main()
