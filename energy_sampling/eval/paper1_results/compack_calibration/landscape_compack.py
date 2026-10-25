"""RDF distance against COMPACK on one analysis results file: the kernel scales of a landscape for a system with no
calibration yet (p23_compack.py + compack_scales.py did MIPCAS and NEHZOR, behind generate_figs.SCALE_RADII).

Sample-sample pairs only, stratified over the stored RDF distance (K per bin) of the results file's analysed set (its
dmat rows, good_ens), so the scale is an average over the landscape, not over the experimental forms. Each pair is
written as two unit-cell CIFs (box_analysis + mol2ucell + write_cif, as p23_compack.py) and compared by mxtaltools'
single_compack_run (cp_worker.py: 20-molecule shell, CCDC defaults); results are cached per pair, so a rerun resumes.
Then, for N = 20, 15 and 10, the distance at which a logistic fit gives P(at least N of 20 matched) = 0.95 ('scales',
the definition behind SCALE_RADII) and = 0.5 ('scales_p50'), each with a 200-resample bootstrap interval, written to
<out_dir>/<tag>_scales.json. CPU only.

usage: python landscape_compack.py <results.pt> <out_dir> <tag> [workers]"""
import json
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # cp_worker
EDGES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 0.90, 1.20]
K = 20


def fit(d, y):
    def nll(p):
        z = p[1] * (d - p[0])
        return np.sum(np.logaddexp(0, z) * y + np.logaddexp(0, -z) * (1 - y))
    best = None
    for d0 in np.linspace(0.05, d.max(), 12):
        r = minimize(nll, [d0, 20.0], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 4000})
        best = r if best is None or r.fun < best.fun else best
    return best.x


def at(d, y, p, rng):
    d50, s = fit(d, y)
    v, b = d50 + np.log(1 / p - 1) / s, []
    for _ in range(200):
        i = rng.integers(0, len(d), len(d))
        if y[i].min() != y[i].max():
            a = fit(d[i], y[i])
            b.append(a[0] + np.log(1 / p - 1) / a[1])
    return float(v), float(np.quantile(b, .025)), float(np.quantile(b, .975))


if __name__ == '__main__':
    import cp_worker
    from mxtaltools.dataset_utils.utils import collate_data_list
    res_path, out, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261025)
    build_path = os.path.join(out, f'{tag}_pairs.pt')
    cif_dir = os.path.join(out, f'{tag}_cif')
    os.makedirs(cif_dir, exist_ok=True)
    if os.path.exists(build_path):
        B = torch.load(build_path, weights_only=False)
    else:
        R = torch.load(res_path, map_location='cpu', weights_only=False)
        good = torch.as_tensor(R['good_ens']).long()
        n = len(good)
        D = R['dmat'].float()
        D = 0.5 * (D + D.T)
        iu = torch.triu_indices(n, n, 1)
        dv = D[iu[0], iu[1]]
        pair_q = {e: float((dv < e).float().mean()) for e in EDGES[1:]}
        pairs = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            idx = torch.nonzero((dv >= lo) & (dv < hi)).flatten().numpy()
            for t in (rng.choice(idx, min(K, len(idx)), replace=False) if len(idx) else []):
                i, j = int(iu[0][t]), int(iu[1][t])
                pairs.append({'a': int(good[i]), 'b': int(good[j]), 'd': float(D[i, j]), 'bin': (lo, hi)})
            print(f"  bin [{lo:.2f},{hi:.2f}): {len(idx):,} pairs available", flush=True)
        need = sorted({p[k] for p in pairs for k in 'ab'})
        cwd = os.getcwd()
        os.chdir(cif_dir)
        try:
            for s0 in range(0, len(need), 32):
                chunk = need[s0:s0 + 32]
                cb = collate_data_list([R['samples'][i].clone() for i in chunk], max_z_prime=1)
                cb.box_analysis()
                cb.mol2ucell()
                cb.write_cif(list(range(len(chunk))), 'tmp_s', mode='unit cell')
                for i, k in enumerate(chunk):
                    os.replace(f'tmp_s_{i}.cif', f's_{k}.cif')
        finally:
            os.chdir(cwd)
        for p in pairs:
            p['cif_a'], p['cif_b'] = (os.path.join(cif_dir, f"s_{p[k]}.cif") for k in 'ab')
            assert os.path.exists(p['cif_a']) and os.path.exists(p['cif_b'])
        B = {'pairs': pairs, 'pair_q': pair_q, 'n': n, 'd_cut': float(R['d_cuts'][0]), 'results': res_path}
        torch.save(B, build_path)
        print(f"{tag}: {len(pairs)} pairs, {len(need)} sample CIFs", flush=True)
    pairs = B['pairs']
    cache = os.path.join(out, f'{tag}_compack.pt')
    done = torch.load(cache, weights_only=False) if os.path.exists(cache) else {}
    todo = [(k, p['cif_a'], p['cif_b']) for k, p in enumerate(pairs) if k not in done]
    T0 = time.time()
    pool = mp.Pool(workers)
    handles = [(t[0], pool.apply_async(cp_worker.work, (t,))) for t in todo]
    for nd, (k, h) in enumerate(handles):
        try:
            key, rmsd, nm, status, dt, tail = h.get(timeout=1800)
        except mp.TimeoutError:
            key, rmsd, nm, status = k, float('nan'), -1, 'timeout'
        done[key] = {'rmsd': rmsd, 'n_matched': nm, 'status': status}
        if nd % 25 == 0:
            torch.save(done, cache)
            print(f"[{time.time() - T0:6.0f}s] {nd + 1}/{len(handles)}", flush=True)
    pool.terminate()
    pool.join()
    torch.save(done, cache)
    ok = [k for k in range(len(pairs)) if done[k]['status'] == 'ok']
    print(f"{tag}: {len(ok)} of {len(pairs)} pairs compared ({time.time() - T0:.0f} s)")
    d = np.array([pairs[k]['d'] for k in ok])
    nm = np.array([done[k]['n_matched'] for k in ok])
    rm = np.array([done[k]['rmsd'] for k in ok])
    scales, scales_p50 = {}, {}
    for thr in (20, 15, 10):
        v5, lo5, hi5 = at(d, (nm >= thr).astype(float), 0.5, rng)
        scales_p50[f'match{thr}of20'] = {'r': v5, 'ci95': [lo5, hi5]}
        v, lo, hi = at(d, (nm >= thr).astype(float), 0.95, rng)
        near = (np.abs(d - v) < 0.03) & (nm >= thr)
        scales[f'match{thr}of20'] = {'r': v, 'ci95': [lo, hi], 'median_rmsd_near': float(np.median(rm[near])) if near.any() else None}
        print(f"  >= {thr} of 20 at 95%: r = {v:.3f} [{lo:.3f}, {hi:.3f}]; median RMSD of matching pairs near r: "
              f"{np.median(rm[near]) if near.any() else float('nan'):.2f} A (n={int(near.sum())}); at 50%: r = {v5:.3f} "
              f"[{lo5:.3f}, {hi5:.3f}]; analysis d_cut {B['d_cut']:.3f}")
    json.dump({'results': B['results'], 'pairs_compared': len(ok), 'scales': scales, 'scales_p50': scales_p50},
              open(os.path.join(out, f'{tag}_scales.json'), 'w'), indent=1)
