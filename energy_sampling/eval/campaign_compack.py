"""
COMPACK identity calibration and deduplication for a crystal_search campaign (mxtaltools.crystal_search.coordinator).

The coordinator puts states into basins by RDF distance (coordinator.RDF_KW: envwise, 10 A; coordinator.
rdf_distance_matrix) at a hard identity cut. What a distance means as packing identity depends on the molecule, space
group and Z', so the cut is measured per problem against COMPACK (ccdc PackingSimilarity, 20-molecule shell, through
mxtaltools' single_compack_run in compack_pair; a (0, 0) engine return is 'failed' and never read as a result), by one
rule:

    RULE  Two states are one packing when COMPACK matches MATCH_MIN = 20 of the SHELL = 20 molecules. The identity cut is
          the largest RDF distance at which the monotone (isotonic, non-increasing in distance) estimate of
          P(match | d) over the calibration pairs is at least P_SAME = 0.95; the dedup distance limit d_hi is the
          largest at which it is at least P_DIFF = 0.05. Both come with 95% pair-bootstrap intervals.

The logistic fit (fit_logistic, logistic_summary: the same forms as the paper calibration script
eval/paper1_results/calibrate_rdf_metric.py, which is local and not in the repository) is reported beside it and does
not set the cut: its symmetric shape does not follow a curve that is flat near zero distance and falls later.

calibrate  pairs of the campaign's own states (read from its shards, within band_kT of the reference), stratified over
           RDF distance and compared by COMPACK in two rounds under one budget: round 1 spreads half the budget evenly
           over [0, span x cut]; round 2 spends the rest in the transition the round-1 fit finds (fitted P between 0.99
           and 0.01), so the comparisons land where they move the cut. Reports the RULE's cut and d_hi, the logistic
           fit, the energy difference of matched pairs (the dedup energy gate), and same-crystal controls -- a state against itself re-described in another cell setting of its setting
           group (mxtaltools.crystal_search.standardize.apply_basis_change) -- which the pipeline must call matches.
dedup      the campaign's product: the basins within band_kT; every pair of them closer than d_hi in RDF distance and
           within dE in energy is compared (lowest energies first, up to the budget; the number left unchecked is
           reported), and matched pairs are merged (union-find; the lowest-energy member represents the group).

Set the cut BEFORE a campaign: calibrate a short pilot campaign (random stream only), then start the campaign with the
measured cut; the pilot's shards may be copied into the new campaign's shards/ (they carry no basin ids). Changing
identity_cut under a running campaign would reassign basins the registry has already built incrementally, and hop
shards refer to basin ids.

    python -m energy_sampling.eval.campaign_compack calibrate CAMPAIGN_DIR [--budget 240] [--band_kT 3]
    python -m energy_sampling.eval.campaign_compack dedup CAMPAIGN_DIR --band_kT 2 [--calibration CAL.pt] --out OUT.pt
    (either with --mol_path LOCAL_CONFORMER.pt on a campaign directory copied from the cluster)

Run with energy_sampling's parent directory and mxtaltools on PYTHONPATH. CPU only (CUDA_VISIBLE_DEVICES=-1); needs
the ccdc Python API. Works in CAMPAIGN_DIR/compack/: one CIF per crystal and a per-pair result cache, both keyed by the
crystal's stored parameters, so a rerun, a larger budget or a dedup after a calibration reuses every comparison made.
Only final COMPACK results are cached (a match count, or the engine's own (0, 0) failure); an error or a timeout is run
again next time. dedup refuses a registry that has not ingested every shard on disk (run the coordinator's export, which
catches up, first).
"""
import argparse
import contextlib
import glob
import hashlib
import io
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from scipy.optimize import minimize

SHELL = 20  # PackingSimilarity.packing_shell_size in single_compack_run
MATCH_MIN = 20  # the RULE: one packing when COMPACK matches this many of the SHELL molecules
P_SAME = 0.95  # the RULE: identity cut = largest distance with isotonic P(match) >= P_SAME
P_DIFF = 0.05  # the RULE: dedup d_hi = largest distance with isotonic P(match) >= P_DIFF


def compack_pair(args):
    """mxtaltools' single_compack_run on one pair, the first CIF as reference; its (0, 0) return is 'failed'."""
    key, ref_path, test_path = args
    from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import single_compack_run
    t = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rmsd, nm = single_compack_run(key, test_path, ref_path)
        return key, float(rmsd), int(nm), 'failed' if (nm == 0 and rmsd == 0) else 'ok', time.time() - t
    except Exception as e:  # noqa: BLE001 - recorded, never read as a result
        return key, float('nan'), -1, f'error {type(e).__name__}: {str(e)[:200]}', time.time() - t


def fit_logistic(d, y):
    """P(match) = 1 / (1 + exp(b (d - d50))), maximum likelihood; returns (d50, b)."""
    def nll(p):
        z = p[1] * (d - p[0])
        return np.sum(np.logaddexp(0, z) * y + np.logaddexp(0, -z) * (1 - y))
    best = None
    for d0 in np.linspace(max(d.min(), 1e-3), d.max(), 12):
        r = minimize(nll, [d0, 20.0], method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 4000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def d_at(p, d50, b):
    return d50 + np.log(1 / p - 1) / b


def logistic_summary(d, y, nboot, seed=0):
    """The fit, and 95% pair-bootstrap intervals on d(P=0.95), d(P=0.5), d(P=0.05); one-class resamples dropped."""
    rng = np.random.default_rng(seed)
    d50, b = fit_logistic(d, y)
    boots = []
    for _ in range(nboot):
        i = rng.integers(0, len(d), len(d))
        if y[i].min() == y[i].max():
            continue
        boots.append(fit_logistic(d[i], y[i]))
    boots = np.array(boots)
    out = {}
    for p in (0.95, 0.5, 0.05):
        bv = d_at(p, boots[:, 0], boots[:, 1])
        out[p] = (float(d_at(p, d50, b)), float(np.percentile(bv, 2.5)), float(np.percentile(bv, 97.5)))
    return (float(d50), float(b)), out, len(boots)


def thresholds(d, y):
    """Cumulative thresholds: >= 95% matched below d_same_95, <= 5% above d_diff_05; first unmatched, last matched."""
    o = np.argsort(d)
    ds, ys = d[o], y[o]
    ok = np.nonzero(np.cumsum(ys) / np.arange(1, len(ys) + 1) >= 0.95)[0]
    ok2 = np.nonzero(np.cumsum(ys[::-1])[::-1] / np.arange(len(ys), 0, -1) <= 0.05)[0]
    return {'d_same_95': float(ds[ok.max()]) if len(ok) else float('nan'),
            'd_diff_05': float(ds[ok2.min()]) if len(ok2) else float('nan'),
            'first_unmatched': float(ds[np.nonzero(ys == 0)[0].min()]) if (ys == 0).any() else float('nan'),
            'last_matched': float(ds[np.nonzero(ys == 1)[0].max()]) if (ys == 1).any() else float('nan')}


def auc(d, y):
    """P(an unmatched pair lies farther than a matched pair); ties count 1/2."""
    a, b = d[y == 1], d[y == 0]
    if not len(a) or not len(b):
        return float('nan')
    return float(((b[None, :] > a[:, None]).sum() + 0.5 * (b[None, :] == a[:, None]).sum()) / (len(a) * len(b)))


def isotonic_decreasing(y):
    """Least-squares non-increasing fit of y (already ordered by distance): pool adjacent violators."""
    blocks = []
    for v in y:
        blocks.append([float(v), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            m2, w2 = blocks.pop()
            m1, w1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    return np.concatenate([np.full(w, m) for m, w in blocks]) if blocks else np.zeros(0)


def rule_distance(d, y, level):
    """Largest compared distance at which the isotonic estimate of P(match | d) is >= level; nan if none is."""
    o = np.argsort(d, kind='stable')
    fit = isotonic_decreasing(y[o])
    ok = np.nonzero(fit >= level)[0]
    return float(d[o][ok.max()]) if len(ok) else float('nan')


def rule_summary(d, y, nboot, seed=0):
    """The RULE's identity cut and d_hi, each (point, 2.5%, 97.5%) over pair-bootstrap resamples."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, level in (('cut', P_SAME), ('d_hi', P_DIFF)):
        boots = [rule_distance(d[i], y[i], level) for i in (rng.integers(0, len(d), len(d)) for _ in range(nboot))]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if np.isfinite(boots).any() else (np.nan, np.nan)
        out[name] = (rule_distance(d, y, level), float(lo), float(hi))
    return out


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ----------------------------------------------------------------------------
# states, keys, CIFs
# ----------------------------------------------------------------------------
def crystal_key(params, hand):
    """Identity of a stored crystal: SHA1 of its float32 parameters and handedness."""
    h = hashlib.sha1(np.asarray(params, dtype=np.float32).tobytes())
    h.update(np.asarray(hand, dtype=np.float32).tobytes())
    return h.hexdigest()[:20]


def load_campaign(coord_dir, mol_path=None):
    """(coordinator module, CampaignConfig, registry or None). mol_path replaces coord.yaml's conformer path, which
    names a cluster path in a campaign copied from the cluster."""
    from mxtaltools.crystal_search import coordinator as co
    cfg = co.CampaignConfig.load(os.path.join(coord_dir, 'coord.yaml'))
    if mol_path:
        cfg.mol_path = mol_path
    rp = os.path.join(coord_dir, 'registry.pt')
    reg = torch.load(rp, weights_only=False) if os.path.exists(rp) else None
    return co, cfg, reg


def shard_states(co, cfg, reg, coord_dir, band_kT):
    """Every physical state in the campaign's shards (this energy model only) within band_kT of the reference, exact
    repeats removed. Returns dict(params [n,18], hand [n,zp], energy [n], key [n], run [n])."""
    P, H, E, R = [], [], [], []
    skipped = 0
    for p in sorted(glob.glob(os.path.join(coord_dir, 'shards', '*', '*.pt'))):
        s = torch.load(p, weights_only=False)
        if s.get('energy_model_id') != cfg.energy_model_id:
            skipped += 1
            continue
        if len(s['energy']) == 0:
            continue
        keep = co.physical(s['params'], s['lj']) & torch.isfinite(s['energy'])
        P.append(s['params'][keep])
        H.append(s['handedness'][keep])
        E.append(s['energy'][keep])
        R += [s['run']] * int(keep.sum())
    if skipped:
        log(f'{skipped} shards from another energy model ignored')
    if not P:
        raise SystemExit(f'no shard states in {coord_dir}')
    P, H, E = torch.cat(P), torch.cat(H), torch.cat(E)
    ref = co._energy_ref(cfg, reg) if reg is not None else (cfg.energy_ref if cfg.energy_ref is not None
                                                            else float(E.min()))
    win = E <= ref + band_kT * cfg.kT
    keys, idx = [], []
    seen = set()
    for i in torch.nonzero(win).flatten().tolist():
        k = crystal_key(P[i], H[i])
        if k not in seen:
            seen.add(k)
            keys.append(k)
            idx.append(i)
    idx = torch.as_tensor(idx, dtype=torch.long)
    log(f'{len(E)} shard states, {int(win.sum())} within {band_kT} kT of {ref:.4f}, {len(idx)} distinct')
    return dict(params=P[idx], hand=H[idx], energy=E[idx].double().numpy(), key=keys,
                run=[R[i] for i in idx.tolist()], ref=ref)


def write_cifs(co, cfg, params, hand, keys, cif_dir, crystals=None):
    """One unit-cell CIF per crystal (mol2ucell + write_cif(mode='unit cell'), the route calibrate_rdf_metric and
    compare.compack_confirm use), named <key>.cif; existing files are kept."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    os.makedirs(cif_dir, exist_ok=True)
    todo = [i for i, k in enumerate(keys) if not os.path.exists(os.path.join(cif_dir, f'{k}.cif'))]
    tmp = f'tmp_{os.getpid()}_cif'  # per process: two runs on one campaign never rename each other's files
    cwd = os.getcwd()
    try:
        os.chdir(cif_dir)
        for lo in range(0, len(todo), 32):
            chunk = todo[lo:lo + 32]
            if crystals is None:
                cl = co.rebuild_crystals(cfg, params[chunk], hand[chunk])
            else:
                cl = [crystals[i] for i in chunk]
            cb = collate_data_list([c.clone() for c in cl], exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
            cb.box_analysis()
            cb.mol2ucell()
            with contextlib.redirect_stdout(io.StringIO()):
                cb.write_cif(list(range(len(chunk))), tmp, mode='unit cell')
            for j, i in enumerate(chunk):
                os.replace(f'{tmp}_{j}.cif', f'{keys[i]}.cif')
    finally:
        os.chdir(cwd)
    return [os.path.join(cif_dir, f'{k}.cif') for k in keys]


# ----------------------------------------------------------------------------
# COMPACK with a keyed cache
# ----------------------------------------------------------------------------
def pair_key(ka, kb):
    return f'{ka}|{kb}' if ka <= kb else f'{kb}|{ka}'


FINAL = ('ok', 'failed')  # COMPACK outcomes fixed for a given pair of CIFs; errors and timeouts are retried


def load_cache(path):
    return torch.load(path, weights_only=False) if os.path.exists(path) else {}


def compare(pairs, cif_dir, cache_path, n_proc, pair_timeout):
    """pairs: list of (key_a, key_b). Returns {pair_key: dict(rmsd, n_matched, status, seconds)}; results are cached
    under pair_key. A pair with a final result ('ok', or the engine's deterministic (0, 0) 'failed') is never run again;
    an 'error' or 'timeout' is reported now and run again next time. The lower key is the reference."""
    cache = load_cache(cache_path)
    want = list(dict.fromkeys(pair_key(a, b) for a, b in pairs))
    todo = [k for k in want if k not in cache or cache[k]['status'] not in FINAL]
    log(f'COMPACK: {len(want)} pairs, {len(want) - len(todo)} cached, {len(todo)} to run on {n_proc} processes')
    if todo:
        pool = mp.Pool(n_proc)
        try:
            handles = []
            for k in todo:
                a, b = k.split('|')
                handles.append((k, pool.apply_async(compack_pair, ((k, os.path.join(cif_dir, f'{a}.cif'),
                                                                     os.path.join(cif_dir, f'{b}.cif')),))))
            t0 = time.time()
            for n, (k, h) in enumerate(handles):
                try:
                    _, rmsd, nm, status, dt = h.get(timeout=pair_timeout)
                except mp.TimeoutError:
                    rmsd, nm, status, dt = float('nan'), -1, 'timeout', float('nan')
                cache[k] = dict(rmsd=rmsd, n_matched=nm, status=status, seconds=dt)
                if (n + 1) % 25 == 0 or n + 1 == len(handles):
                    tmp = f'{cache_path}.{os.getpid()}.tmp'
                    torch.save(cache, tmp)
                    os.replace(tmp, cache_path)
                    log(f'   {n + 1}/{len(handles)} compared ({(time.time() - t0) / (n + 1):.2f} s per pair wall)')
        finally:
            pool.terminate()
            pool.join()
    return {k: cache[k] for k in want}


# ----------------------------------------------------------------------------
# pair selection
# ----------------------------------------------------------------------------
def draw_pairs(D, lo, hi, n, rng, uses, cap, taken):
    """Up to n pairs (i < j) with lo <= D < hi, uniformly at random among those whose crystals are each in fewer than
    `cap` chosen pairs (so a few dense basins cannot fill a bin), none already taken."""
    iu, ju = np.nonzero(np.triu((D >= lo) & (D < hi), 1))
    out = []
    for k in rng.permutation(len(iu)):
        i, j = int(iu[k]), int(ju[k])
        if uses[i] >= cap or uses[j] >= cap or (i, j) in taken:
            continue
        out.append((i, j))
        uses[i] += 1
        uses[j] += 1
        taken.add((i, j))
        if len(out) == n:
            break
    return out


def spread(D, lo, hi, n_bins, budget, rng, uses, cap, taken):
    edges = np.linspace(lo, hi, n_bins + 1)
    per = max(1, budget // n_bins)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out += draw_pairs(D, a, b, per, rng, uses, cap, taken)
    return out, edges


def setting_controls(co, cfg, st, rows, rng):
    """Each row re-described in another cell setting of its setting group (apply_basis_change); returns the crystals
    and their keys. Same crystal, so COMPACK must match it to the original 20/20."""
    from mxtaltools.crystal_search.standardize import apply_basis_change, cell_from_metric, keeps_operators, metric
    from mxtaltools.dataset_utils.utils import collate_data_list
    base = collate_data_list(co.rebuild_crystals(cfg, st['params'][rows], st['hand'][rows]))
    Ns = []
    for i in range(len(rows)):
        L = base.cell_lengths[i].double().numpy()
        A = base.cell_angles[i].double().numpy()
        for _ in range(2000):
            p, q, r, s = rng.integers(-1, 2, size=4)
            N = np.array([[p, 0, r], [0, p * s - q * r, 0], [q, 0, s]]) if 3 <= cfg.sg <= 15 else \
                rng.integers(-1, 2, size=(3, 3))
            if round(np.linalg.det(N)) != 1 or (N == np.eye(3)).all() or not keeps_operators(N, cfg.sg):
                continue
            Ln, An = cell_from_metric(N.T @ metric(L, A) @ N)
            if 37 < np.degrees(An).min() and np.degrees(An).max() < 143:
                break
        else:
            N = np.eye(3, dtype=np.int64)
        Ns.append(N)
    skew = apply_basis_change(base, np.stack(Ns).astype(np.int64))
    crystals = skew.batch_to_list()
    keys = [crystal_key(c.full_cell_parameters().detach().float().reshape(-1),
                        c.aunit_handedness.detach().float().reshape(-1)) + '_ctl' for c in crystals]
    return crystals, keys, np.stack(Ns)


# ----------------------------------------------------------------------------
# calibrate
# ----------------------------------------------------------------------------
def calibrate(a):
    co, cfg, reg = load_campaign(a.coord_dir, a.mol_path)
    work = os.path.join(a.coord_dir, 'compack')
    cif_dir, cache_path = os.path.join(work, 'cif'), os.path.join(work, 'cache.pt')
    os.makedirs(cif_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    st = shard_states(co, cfg, reg, a.coord_dir, a.band_kT)
    n = len(st['key'])
    if n > a.max_pool:
        sub = np.sort(rng.choice(n, a.max_pool, replace=False))
        st = dict(params=st['params'][sub], hand=st['hand'][sub], energy=st['energy'][sub],
                  key=[st['key'][i] for i in sub], run=[st['run'][i] for i in sub], ref=st['ref'])
        n = a.max_pool
    t = time.time()
    crystals = co.rebuild_crystals(cfg, st['params'], st['hand'])
    R = co.compute_rdfs(crystals, cfg.rdf_batch)
    D = co.rdf_distance_matrix(R, R).double().numpy()
    log(f'{n} states: RDFs and distances in {time.time() - t:.0f} s')
    cut0 = float(a.cut if a.cut is not None else cfg.identity_cut)
    hi = a.span * cut0
    uses, taken = np.zeros(n, dtype=np.int64), set()

    # round 1
    r1, edges1 = spread(D, 0.0, hi, a.bins, a.budget // 2, rng, uses, a.cap, taken)
    ctl_rows = rng.choice(n, min(a.controls, n), replace=False)
    ctl_crystals, ctl_keys, ctl_N = setting_controls(co, cfg, st, ctl_rows, rng)
    def cifs_for(pairs):  # only the states that are compared
        need = sorted({i for p in pairs for i in p} | set(int(i) for i in ctl_rows))
        write_cifs(co, cfg, None, None, [st['key'][i] for i in need], cif_dir, crystals=[crystals[i] for i in need])
    cifs_for(r1)
    write_cifs(co, cfg, None, None, ctl_keys, cif_dir, crystals=ctl_crystals)
    ctl_pairs = [(st['key'][i], k) for i, k in zip(ctl_rows, ctl_keys)]
    res = compare([(st['key'][i], st['key'][j]) for i, j in r1] + ctl_pairs, cif_dir, cache_path, a.n_proc,
                  a.pair_timeout)

    def table(pairs):
        d = np.array([D[i, j] for i, j in pairs])
        r = [res[pair_key(st['key'][i], st['key'][j])] for i, j in pairs]
        ok = np.array([x['status'] == 'ok' for x in r])
        nm = np.array([x['n_matched'] for x in r])
        dE = np.array([abs(st['energy'][i] - st['energy'][j]) for i, j in pairs])
        return d, ok, nm, dE, np.array([x['seconds'] for x in r])

    d1, ok1, nm1, _, _ = table(r1)
    y1 = (nm1[ok1] >= MATCH_MIN).astype(float)
    fit_ok = len(y1) > 4 and 0 < y1.sum() < len(y1)
    if fit_ok:
        d50, b = fit_logistic(d1[ok1], y1)
        lo2, hi2 = max(0.0, d_at(0.99, d50, b)), min(2 * hi, d_at(0.01, d50, b))
        why = f'round-1 fit: d50 {d50:.4f}, slope {b:.1f}'
    elif len(y1) and y1.min() == 1:
        lo2, hi2, why = hi, 2 * hi, 'round 1 matched everywhere: the transition is above span x cut'
    else:
        lo2, hi2, why = 0.0, hi, 'round 1 had no match or too few pairs: the transition is not bracketed'
    # round 2
    r2 = []
    if hi2 > lo2:
        r2, _ = spread(D, lo2, hi2, max(2, a.bins // 2), a.budget - len(r1), rng, uses, a.cap, taken)
    log(f'round 2 on [{lo2:.4f}, {hi2:.4f}) ({why}): {len(r2)} pairs')
    cifs_for(r2)
    res.update(compare([(st['key'][i], st['key'][j]) for i, j in r2] + ctl_pairs, cif_dir, cache_path, a.n_proc,
                       a.pair_timeout))

    pairs = r1 + r2
    d, ok, nm, dE, sec = table(pairs)
    y = (nm >= MATCH_MIN).astype(float)
    dd, yy = d[ok], y[ok]
    ctl = [res[pair_key(*p)] for p in ctl_pairs]
    lines = [f'# COMPACK identity calibration: {os.path.basename(os.path.abspath(a.coord_dir))}', '',
             f'Energy model `{cfg.energy_model_id}`; sg {cfg.sg}, Z\'={cfg.z_prime}; states within {a.band_kT} kT of '
             f'{st["ref"]:.4f}: {n} (from the shards, exact repeats removed); RDF as the coordinator computes it '
             f'(envwise, 10 A); campaign identity cut {cfg.identity_cut}. Pairs stratified over RDF distance, at most '
             f'{a.cap} pairs per state, round 1 on [0, {hi:.3f}), round 2 on [{lo2:.3f}, {hi2:.3f}) ({why}). '
             f'A match is {MATCH_MIN}/{SHELL} molecules (COMPACK PackingSimilarity, shell {SHELL}). Seed {a.seed}.', '',
             f'Pairs compared: {len(pairs)}; engine failures or timeouts {int((~ok).sum())} (excluded). COMPACK wall '
             f'time per pair: median {np.nanmedian(sec):.1f} s on one process.', '',
             f'*P(match) per RDF distance bin; pairs of distinct states from this campaign; the bins are the '
             f'round-1 width (round 2 filled the transition more densely).*', '',
             f'| RDF distance bin | pairs | matched {MATCH_MIN}/{SHELL} | matched >= 15/{SHELL} | P(match) |',
             '|---|---|---|---|---|']
    w = edges1[1] - edges1[0]
    edges = np.arange(0.0, max(float(dd.max()) if len(dd) else hi, hi) + w, w)
    for lo, up in zip(edges[:-1], edges[1:]):
        m = (dd >= lo) & (dd < up)
        if m.sum():
            lines.append(f'| [{lo:.4f}, {up:.4f}) | {int(m.sum())} | {int(yy[m].sum())} | '
                         f'{int((nm[ok][m] >= 15).sum())} | {yy[m].mean():.2f} |')
    summary = dict(pairs=[dict(a=st['key'][i], b=st['key'][j], d=float(D[i, j]), E_a=float(st['energy'][i]),
                               E_b=float(st['energy'][j]), **res[pair_key(st['key'][i], st['key'][j])])
                          for i, j in pairs],
                   controls=[dict(key=st['key'][i], N=N.tolist(), **c) for i, N, c in zip(ctl_rows, ctl_N, ctl)],
                   campaign_cut=cfg.identity_cut, energy_model_id=cfg.energy_model_id, band_kT=a.band_kT, n_states=n)
    lines.append('')
    if len(yy) > 4 and 0 < yy.sum() < len(yy):
        (d50, b), s, nb = logistic_summary(dd, yy, nboot=a.nboot, seed=a.seed)
        th = thresholds(dd, yy)
        rs = rule_summary(dd, yy, a.nboot, seed=a.seed)
        rec, d_hi = rs['cut'][0], rs['d_hi'][0]
        lines += [f'**RULE (one packing = {MATCH_MIN}/{SHELL} COMPACK molecules; isotonic P(match | d)): identity_cut '
                  f'{rec:.3f} [{rs["cut"][1]:.3f}, {rs["cut"][2]:.3f}]** (largest distance with P >= {P_SAME}); dedup '
                  f'd_hi {d_hi:.3f} [{rs["d_hi"][1]:.3f}, {rs["d_hi"][2]:.3f}] (largest with P >= {P_DIFF}); 95% pair '
                  f'bootstrap, {a.nboot} resamples. Compared pairs at or below the cut: {int((dd <= rec).sum())}, of '
                  f'which matched {int(yy[dd <= rec].sum())}. The campaign uses {cfg.identity_cut}.', '']
        lines += [f'Logistic fit (descriptive) on {len(yy)} pairs ({int(yy.sum())} matched): slope {b:.1f} per unit, d50 {d50:.4f} '
                  f'[{s[0.5][1]:.4f}, {s[0.5][2]:.4f}]; d(P=0.95) {s[0.95][0]:.4f} [{s[0.95][1]:.4f}, '
                  f'{s[0.95][2]:.4f}]; d(P=0.05) {s[0.05][0]:.4f} [{s[0.05][1]:.4f}, {s[0.05][2]:.4f}] (95% pair '
                  f'bootstrap, {nb} usable resamples); AUC {auc(dd, yy):.3f}. Cumulative: >= 95% matched below '
                  f'{th["d_same_95"]:.4f}, <= 5% matched above {th["d_diff_05"]:.4f}; first unmatched '
                  f'{th["first_unmatched"]:.4f}, last matched {th["last_matched"]:.4f}. Fitted P(match) at the campaign '
                  f'cut: {1 / (1 + np.exp(b * (cfg.identity_cut - d50))):.2f}.']
        summary.update(d50=d50, slope=b, fit=s, thresholds=th, rule=rs, recommended_cut=rec, d_hi=d_hi,
                       rule_def=dict(match_min=MATCH_MIN, shell=SHELL, p_same=P_SAME, p_diff=P_DIFF))
    else:
        lines.append('No logistic fit: the compared pairs are all matched or all unmatched -- widen --span or move '
                     '--cut and rerun (every comparison made is cached).')
    dEm = dE[ok][yy == 1]
    if len(dEm):
        q = np.percentile(dEm, [50, 90, 99, 100])
        lines += ['', f'Energy difference of matched pairs (energy units as stored, per molecule): median {q[0]:.4f}, '
                      f'90th pct {q[1]:.4f}, 99th {q[2]:.4f}, max {q[3]:.4f}; dedup dE default 2 x max = '
                      f'{2 * q[3]:.4f}.']
        summary['dE_gate'] = float(2 * q[3])
    cm = sum(c['n_matched'] >= MATCH_MIN for c in ctl)
    lines += ['', f'Controls (a state against itself in another cell setting): {cm}/{len(ctl)} matched 20/20, '
                  f'RMSD max {max((c["rmsd"] for c in ctl), default=float("nan")):.2e} A. '
                  + ('' if cm == len(ctl) else '**A control did not match: the CIF/COMPACK pipeline is not measuring '
                                               'packing identity here; do not use this calibration.**')]
    out = a.out or os.path.join(work, 'calibration')
    torch.save(summary, out + '.pt')
    with open(out + '.md', 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


# ----------------------------------------------------------------------------
# dedup
# ----------------------------------------------------------------------------
def dedup(a):
    co, cfg, reg = load_campaign(a.coord_dir, a.mol_path)
    if reg is None:
        raise SystemExit('no registry.pt: run the curate pass first')
    left = co.pending_shards(a.coord_dir, reg)
    if left and not a.allow_pending:
        raise SystemExit(f'{len(left)} shard(s) on disk are not in registry.pt (e.g. {left[:3]}): run the coordinator '
                         f'export (it ingests them first), or pass --allow_pending to deduplicate the registry as it is')
    work = os.path.join(a.coord_dir, 'compack')
    cif_dir, cache_path = os.path.join(work, 'cif'), os.path.join(work, 'cache.pt')
    cal = torch.load(a.calibration, weights_only=False) if a.calibration else {}
    if cal and cal.get('energy_model_id') not in (None, cfg.energy_model_id):
        raise SystemExit(f"calibration scored by {cal.get('energy_model_id')!r}, campaign by {cfg.energy_model_id!r}")
    d_hi = a.d_hi if a.d_hi is not None else cal.get('d_hi', 2 * cfg.identity_cut)
    dE = a.dE if a.dE is not None else cal.get('dE_gate', float('inf'))
    if not np.isfinite(d_hi) or d_hi <= 0:
        raise SystemExit(f'd_hi {d_hi} is not a usable distance: pass --d_hi')
    ref = co._energy_ref(cfg, reg)
    E = np.asarray(reg['basin_E'], dtype=float)
    idx = np.nonzero(E <= ref + a.band_kT * cfg.kT)[0]
    idx = idx[np.argsort(E[idx])]
    params = torch.stack([reg['basin_params'][j] for j in idx]).float()
    hand = torch.stack([reg['basin_hand'][j] for j in idx]).float()
    keys = [crystal_key(p, h) for p, h in zip(params, hand)]
    crystals = co.rebuild_crystals(cfg, params, hand)
    D = co.rdf_distance_matrix(*(2 * [co.compute_rdfs(crystals, cfg.rdf_batch)])).double().numpy()
    Eb = E[idx]
    iu, ju = np.nonzero(np.triu(D < d_hi, 1) & (np.abs(Eb[:, None] - Eb[None, :]) <= dE))
    order = np.lexsort((D[iu, ju], np.maximum(Eb[iu], Eb[ju])))  # lowest energies first, then closest
    cand = [(int(iu[k]), int(ju[k])) for k in order]
    run = cand if a.budget is None else cand[:a.budget]
    skipped = cand[len(run):]
    log(f'{len(idx)} basins within {a.band_kT} kT; {len(cand)} pairs with d < {d_hi:.4f} and |dE| <= {dE:.4g}; '
        f'comparing {len(run)}')
    need = sorted({i for p in run for i in p})
    write_cifs(co, cfg, None, None, [keys[i] for i in need], cif_dir, crystals=[crystals[i] for i in need])
    res = compare([(keys[i], keys[j]) for i, j in run], cif_dir, cache_path, a.n_proc, a.pair_timeout)
    parent = list(range(len(idx)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    matched = 0
    for i, j in run:
        r = res[pair_key(keys[i], keys[j])]
        if r['status'] == 'ok' and r['n_matched'] >= MATCH_MIN:
            matched += 1
            a_, b_ = find(i), find(j)
            if a_ != b_:
                parent[max(a_, b_)] = min(a_, b_)  # rows are in energy order: the lower index is the lower energy
    groups = {}
    for i in range(len(idx)):
        groups.setdefault(find(i), []).append(i)
    reps = sorted(groups)
    out = []
    unchecked = np.zeros(len(idx), dtype=np.int64)  # candidate pairs a budget left out, per group
    for i, j in skipped:
        if find(i) != find(j):
            unchecked[find(i)] += 1
            unchecked[find(j)] += 1
    rows = ['group,basin,energy,above_ref_kT,merged_basins,uncompared_candidate_pairs']
    for g in reps:
        c = crystals[g].clone()
        setattr(c, cfg.energy_key, torch.tensor([Eb[g]]))
        c.basin_id = torch.tensor([int(idx[g])])
        c.merged_basins = [int(idx[m]) for m in groups[g]]
        c.uncompared_candidate_pairs = torch.tensor([int(unchecked[g])])
        c.energy_model_id = cfg.energy_model_id
        out.append(c)
        rows.append(f'{len(out) - 1},{idx[g]},{Eb[g]:.5f},{(Eb[g] - ref) / cfg.kT:.4f},'
                    f'{" ".join(str(int(idx[m])) for m in groups[g])},{int(unchecked[g])}')
    torch.save(out, a.out)
    with open(os.path.splitext(a.out)[0] + '.csv', 'w') as fh:
        fh.write('\n'.join(rows) + '\n')
    fails = sum(res[pair_key(keys[i], keys[j])]['status'] != 'ok' for i, j in run)
    log(f'dedup: {len(idx)} basins -> {len(reps)} distinct packings ({matched} matched pairs of {len(run)} compared, '
        f'{fails} engine failures or timeouts); {len(skipped)} candidate pairs NOT compared (budget '
        f'{a.budget or "none"}); '
        f'wrote {a.out} and its .csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('calibrate', 'dedup'):
        p = sub.add_parser(name)
        p.add_argument('coord_dir')
        p.add_argument('--n_proc', type=int, default=6)
        p.add_argument('--pair_timeout', type=float, default=300.0)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--mol_path', default=None,
                       help="local copy of the campaign conformer (coord.yaml's may be a cluster path)")
    c = sub.choices['calibrate']
    c.add_argument('--budget', type=int, default=240, help='COMPACK pairs over both rounds (controls extra)')
    c.add_argument('--band_kT', type=float, default=3.0)
    c.add_argument('--max_pool', type=int, default=3000, help='states sampled for pairing')
    c.add_argument('--cut', type=float, default=None, help='starting guess (default: the campaign identity_cut)')
    c.add_argument('--span', type=float, default=3.0, help='round 1 covers [0, span x cut]')
    c.add_argument('--bins', type=int, default=12)
    c.add_argument('--cap', type=int, default=2, help='pairs per state')
    c.add_argument('--controls', type=int, default=4)
    c.add_argument('--nboot', type=int, default=500)
    c.add_argument('--out', default=None)
    d = sub.choices['dedup']
    d.add_argument('--band_kT', type=float, required=True)
    d.add_argument('--calibration', default=None, help='a calibrate output (.pt): the RULE\'s d_hi and the dE gate')
    d.add_argument('--d_hi', type=float, default=None)
    d.add_argument('--dE', type=float, default=None)
    d.add_argument('--budget', type=int, default=None, help='cap on COMPACK pairs (default: every candidate pair)')
    d.add_argument('--allow_pending', action='store_true',
                   help='deduplicate even though some shards on disk are not in the registry')
    d.add_argument('--out', required=True)
    a = ap.parse_args()
    calibrate(a) if a.cmd == 'calibrate' else dedup(a)


if __name__ == '__main__':
    main()
