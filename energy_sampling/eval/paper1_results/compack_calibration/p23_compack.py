"""Fresh RDF-distance vs COMPACK calibration on the p23 paper runs (10k analysis set: reduced cells, 15 kT window).

Per run: sample-sample pairs stratified over the stored atomwise RDF distance (K per bin), plus every experimental form
against its nearest sample and K_REF samples per bin. Each pair is written as two unit-cell CIFs (mol2ucell +
write_cif, the route the p20 calibration used) and compared by mxtaltools' single_compack_run (PackingSimilarity,
20-molecule shell, CCDC defaults). Results are cached per pair. usage: python p23_compack.py <mipu|nehu> <out_dir>
(pairs, CIFs and the COMPACK cache go to out_dir; compack_scales.py fits the N-of-20 distances from them)."""
import os, sys, time
import multiprocessing as mp
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # cp_worker
OUT = sys.argv[2] if __name__ == '__main__' else None
RES = r'D:\crystal_datasets\gfn_results'
REFS = {'mipu': r'D:\crystal_datasets\mipcas\MIPCAS_standardized_niggli.pt',
        'nehu': r'D:\crystal_datasets\nehzor\NEHZOR_structures_std_conf.pt'}
EDGES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 0.90, 1.20]
K, K_REF = 20, 5

if __name__ == '__main__':
    import cp_worker
    from mxtaltools.dataset_utils.utils import collate_data_list
    from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
    run = sys.argv[1]
    rng = np.random.default_rng(20261002)
    build_path = os.path.join(OUT, f'p23_{run}_pairs.pt')
    cif_dir = os.path.join(OUT, f'p23_{run}_cif'); os.makedirs(cif_dir, exist_ok=True)
    if os.path.exists(build_path):
        B = torch.load(build_path, weights_only=False)
    else:
        R = torch.load(os.path.join(RES, f'p23_{run}_ft_running_10k_red0.pt'), map_location='cpu', weights_only=False)
        good = torch.as_tensor(R['good_ens']).long(); n = len(good)
        D = R['dmat'].float(); D = 0.5 * (D + D.T)
        iu = torch.triu_indices(n, n, 1); dv = D[iu[0], iu[1]]
        pair_q = {e: float((dv < e).float().mean()) for e in EDGES[1:]}
        pairs = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            idx = torch.nonzero((dv >= lo) & (dv < hi)).flatten().numpy()
            for t in (rng.choice(idx, min(K, len(idx)), replace=False) if len(idx) else []):
                i, j = int(iu[0][t]), int(iu[1][t])
                pairs.append({'cls': 'ss', 'a': ('s', int(good[i])), 'b': ('s', int(good[j])), 'd': float(D[i, j]), 'bin': (lo, hi)})
            print(f"  ss bin [{lo:.2f},{hi:.2f}): {len(idx):,} pairs available", flush=True)
        exp = torch.load(REFS[run], map_location='cpu', weights_only=False)
        exp = exp if isinstance(exp, list) else exp.batch_to_list()
        if run == 'nehu':
            exp = [e for e in exp if str(e.identifier) in ('NEHZOR', 'NEHZOR01')]
        refs = torch.load(os.path.join(RES, 'dens', f'p23_{run}_ft_running_10k_100000_k_red0', 'refs.pt'), weights_only=False)
        assert refs['labels'] == [str(e.identifier) for e in exp]
        bins = torch.linspace(0, 10, refs['rdf'].shape[-1]); rdf_s = R['sample_batch'].rdf[good]
        for r, lab in enumerate(refs['labels']):
            d = compute_rdf_distance(refs['rdf'][r], rdf_s, bins).float(); near = int(d.argmin())
            pairs.append({'cls': 'ref', 'a': ('r', r), 'b': ('s', int(good[near])), 'd': float(d[near]), 'bin': 'nearest', 'ref': lab})
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                idx = torch.nonzero((d >= lo) & (d < hi)).flatten().numpy(); idx = idx[idx != near]
                for t in (rng.choice(idx, min(K_REF, len(idx)), replace=False) if len(idx) else []):
                    pairs.append({'cls': 'ref', 'a': ('r', r), 'b': ('s', int(good[int(t)])), 'd': float(d[int(t)]), 'bin': (lo, hi), 'ref': lab})
        need = sorted({p[k][1] for p in pairs for k in 'ab' if p[k][0] == 's'})
        cwd = os.getcwd(); os.chdir(cif_dir)
        try:
            for s0 in range(0, len(need), 32):
                chunk = need[s0:s0 + 32]
                cb = collate_data_list([R['samples'][i].clone() for i in chunk], max_z_prime=1)
                cb.box_analysis(); cb.mol2ucell(); cb.write_cif(list(range(len(chunk))), 'tmp_s', mode='unit cell')
                for i, k in enumerate(chunk):
                    os.replace(f'tmp_s_{i}.cif', f's_{k}.cif')
            cb = collate_data_list([e.clone() for e in exp], max_z_prime=1)
            cb.box_analysis(); cb.mol2ucell(); cb.write_cif(list(range(len(exp))), 'tmp_r', mode='unit cell')
            for i in range(len(exp)):
                os.replace(f'tmp_r_{i}.cif', f'r_{i}.cif')
        finally:
            os.chdir(cwd)
        for p in pairs:
            p['cif_a'] = os.path.join(cif_dir, f"{p['a'][0]}_{p['a'][1]}.cif"); p['cif_b'] = os.path.join(cif_dir, f"{p['b'][0]}_{p['b'][1]}.cif")
            assert os.path.exists(p['cif_a']) and os.path.exists(p['cif_b'])
        nn = D.clone(); nn.fill_diagonal_(float('inf'))
        B = {'pairs': pairs, 'pair_q': pair_q, 'n': n, 'd_cut': float(R['d_cuts'][0]),
             'nn_median': float(nn.amin(1).median()), 'pair_quantiles': {q: float(torch.quantile(dv[torch.randperm(len(dv))[:2_000_000]], q)) for q in (0.01, 0.05, 0.15, 0.5)}}
        torch.save(B, build_path)
        print(f"{run}: {len(pairs)} pairs, {len(need)} sample CIFs", flush=True)
    pairs = B['pairs']
    cache = os.path.join(OUT, f'p23_{run}_compack.pt')
    done = torch.load(cache, weights_only=False) if os.path.exists(cache) else {}
    todo = [(k, p['cif_a'], p['cif_b']) for k, p in enumerate(pairs) if k not in done]
    T0 = time.time(); pool = mp.Pool(6)
    handles = [(t[0], pool.apply_async(cp_worker.work, (t,))) for t in todo]
    for nd, (k, h) in enumerate(handles):
        try:
            key, rmsd, nm, status, dt, tail = h.get(timeout=1800)
        except mp.TimeoutError:
            key, rmsd, nm, status, dt = k, float('nan'), -1, 'timeout', float('nan')
        done[key] = {'rmsd': rmsd, 'n_matched': nm, 'status': status}
        if nd % 25 == 0:
            torch.save(done, cache); print(f"[{time.time() - T0:6.0f}s] {nd + 1}/{len(handles)}", flush=True)
    pool.terminate(); pool.join(); torch.save(done, cache)
    st = {}
    for v in done.values():
        st[v['status'].split(' ')[0]] = st.get(v['status'].split(' ')[0], 0) + 1
    print(f"{run}: finished in {time.time() - T0:.0f}s; status {st}", flush=True)
