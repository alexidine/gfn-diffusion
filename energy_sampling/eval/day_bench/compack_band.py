"""COMPACK verdicts for reference minima whose nearest source structures sit at an ambiguous RDF distance.

The envwise RDF distance identifies a structure with a reference minimum reliably below the calibrated cut (0.093 for
imidazole); between that cut and a looser one (0.14) it mixes one minimum relaxed from two starts with distinct
neighbouring minima. COMPACK settles such pairs (the CSD Python API's packing similarity under the Day duplicate rule,
``compack_endpoints.engine``: 30-molecule shells, 30% and 30 degrees; a match is all 30 overlaid), but one comparison of
P1 cells with 4-8 independent molecules takes tens of seconds, so every band pair cannot be checked. Instead, for each
reference minimum within ``--window`` kJ/mol of the lowest (``ref_energy`` of ``--ref_dir``'s first-pass CSV) and each
other source, that source's band structures are compared in ascending RDF distance until one matches, at most
``--per`` of them. ``coverage`` carries each verdict to the source's near-identical structures.

Comparisons run in ``--workers`` processes. The engine's own timeout does not bound a comparison (seen: 123 s at a
20 s setting), so each group is also bounded by its ``--per`` comparisons. Verdicts are appended to
``compack_band_<energy>.csv`` as each group finishes; pairs already there (or in ``--reuse`` files) are not compared
again, so a stopped run resumes.

usage (CPU, licensed CSD Python API):
    python -m eval.day_bench.compack_band OUT_DIR REF_LABEL REF_DIR [--lo 0.093] [--hi 0.14] [--window 7] [--per 3]
        [--workers 12] [--energy exp6] [--reuse CSV,CSV]
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from eval.day_bench.compack_endpoints import SHELL

_ENGINE, _CRYSTALS = None, {}
KEYS = ['a', 'b', 'rdf', 'nmatched', 'rmsd', 'seconds']


def _init():
    global _ENGINE
    from eval.day_bench.compack_endpoints import engine
    _ENGINE = engine()


def _crystal(path):
    if path not in _CRYSTALS:
        from eval.day_bench.compack_endpoints import read
        _CRYSTALS[path] = read(path)
    return _CRYSTALS[path]


def _group(jobs):
    """Compare in order until one matches; the rows of every comparison made."""
    from eval.day_bench import compack_endpoints as ce
    rows = []
    for a, b, pa, pb, d in jobs:
        t = time.time()
        n, rmsd = ce.compare(_ENGINE, _crystal(pa), _crystal(pb), a, b)
        rows.append(dict(a=a, b=b, rdf=f'{d:.4f}', nmatched='' if n is None else n,
                         rmsd='' if rmsd is None else f'{rmsd:.4f}', seconds=f'{time.time() - t:.2f}'))
        if n == SHELL:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('ref_label')
    ap.add_argument('ref_dir')
    ap.add_argument('--lo', type=float, default=0.093)
    ap.add_argument('--hi', type=float, default=0.14)
    ap.add_argument('--window', type=float, default=7.0)
    ap.add_argument('--per', type=int, default=3)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--energy', default='exp6')
    ap.add_argument('--reuse', default='')
    a = ap.parse_args()
    z = np.load(os.path.join(a.out_dir, f'coverage_{a.energy}.npz'))
    D, names, kind, paths = z['D'], z['names'].astype(str), z['kind'].astype(str), z['paths'].astype(str)
    label = np.array([n.split(':', 1)[0] for n in names])
    e_ref = {r['file']: float(r['ref_energy']) for r in csv.DictReader(
        open(os.path.join(a.ref_dir, f'p1_relax_{a.energy}.csv'))) if r['built'] == 'True'}
    ref = np.nonzero((label == a.ref_label) & (kind == 'sample'))[0]
    e = np.array([e_ref[names[i].split('/', 1)[1]] for i in ref])
    ref = ref[e - e.min() <= a.window]

    out = os.path.join(a.out_dir, f'compack_band_{a.energy}.csv')
    known = {}
    for path in [out] + [p for p in a.reuse.split(',') if p]:
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                if r.get('seconds'):  # a row cut short by a stopped run has no timing
                    known[(r['a'], r['b'])] = r
    in_out = {(r['a'], r['b']) for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    reused = [r for k, r in known.items() if k not in in_out]

    groups, settled = [], 0
    for lab in sorted(set(label[(label != a.ref_label) & (kind == 'sample')])):
        s = np.nonzero((label == lab) & (kind == 'sample'))[0]
        for i in ref:
            if (D[i, s] <= a.lo).any():
                continue  # already identified below the cut
            cand = s[(D[i, s] > a.lo) & (D[i, s] <= a.hi)]
            cand = cand[np.argsort(D[i, cand])]
            left, jobs, done = a.per, [], False
            for j in cand:
                if left == 0:
                    break
                r = known.get((names[i], names[j]))
                if r is not None:
                    left -= 1
                    if r['nmatched'] != '' and int(r['nmatched']) == SHELL:
                        done = True
                        break
                    continue
                jobs.append((names[i], names[j], paths[i], paths[j], float(D[i, j])))
                left -= 1
            if done or not jobs:
                settled += bool(len(cand))
                continue
            groups.append(jobs)
    print(f'{len(ref)} {a.ref_label} minima within {a.window:g} kJ/mol; {len(groups)} (minimum, source) groups to check '
          f'({sum(len(g) for g in groups)} comparisons at most), {settled} settled already; {a.workers} workers',
          flush=True)

    new = not os.path.exists(out)
    t0, n_cmp, n_match = time.time(), 0, 0
    with open(out, 'a', newline='') as fh, ProcessPoolExecutor(a.workers, initializer=_init) as pool:
        w = csv.DictWriter(fh, fieldnames=KEYS)
        if new:
            w.writeheader()
        for r in reused:  # verdicts reused from other runs
            w.writerow({k: r[k] for k in KEYS})
        fh.flush()
        futures = [pool.submit(_group, g) for g in groups]
        for k, f in enumerate(as_completed(futures), 1):
            rows = f.result()
            w.writerows(rows)
            fh.flush()
            n_cmp += len(rows)
            n_match += any(r['nmatched'] == SHELL for r in rows)
            if k % 20 == 0 or k == len(futures):
                print(f'  {k} of {len(futures)} groups, {n_cmp} comparisons, {time.time() - t0:.0f} s; groups matched '
                      f'{n_match}', flush=True)


if __name__ == '__main__':
    main()
