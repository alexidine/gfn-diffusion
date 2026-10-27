"""Clusters of one campaign's endpoints by RDF identity and by FD identity, judged against COMPACK.

Endpoints: the physical shard states within band_kT of the reference (every one a registry hit), at most n_max at random.
Each method is leader clustering in ascending energy (the lowest-energy state of a cluster leads it; a state joins its
nearest leader if closer than the cut, else leads a new cluster), as coordinator._assign does in arrival order:
  reg        the campaign's own registry basins (RDF at its cut, arrival order), read from the hits
  rdf_own    RDF distance (coordinator route, atomwise) at the campaign cut
  rdf_cal    RDF distance at its COMPACK-calibrated RULE cut (calibrate.py's cuts.json: the own-design cut)
  fd_cal     FD distance (fd_distance.FDMetric) at its calibrated RULE cut; pre-screened by the cell + rotation bound
  fd_tw_cal  FD distance allowing the molecular-mirror twin, at its calibrated RULE cut
  fd_m       FD distance at the cut whose cluster count matches rdf_own's (bisection on the cut, at most 10 runs)
  fd_tw_m    FD distance with the twin, cut matched to rdf_own's cluster count the same way
The twin variants run only for a molecule whose mirror maps it onto itself to a matched RMSD below 0.1 A by a
bond-preserving permutation of like atoms (fd_distance.molecule_info, common.molecule_template); molecular point
symmetry is otherwise ignored (owner decision).
COMPACK (20/20, repaired CIFs: common.compack_pair_repaired) on sampled pairs per method: within-cluster pairs (false
merge if unmatched) and boundary pairs, each state with the leader of the nearest OTHER cluster under that method (false
split if matched). Methods are compared at two matched operating points, each on one common pair sample: equal cluster
counts (rdf_own, fd_m, fd_tw_m on the pairs those three drew) and each method's own calibrated cut, i.e. one precision
target (rdf_cal, fd_cal, fd_tw_cal on the pairs those three drew).

    python -m energy_sampling.eval.fd_analysis.compare --campaign mipcas --calibration calibrate_mipcas --out compare_mipcas
"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch

from energy_sampling.eval.fd_analysis import common as C
from energy_sampling.eval.fd_analysis import fd_distance as F

F64 = torch.float64


class FDLeaders:
    """Leader clustering in the given order with the FD distance; leaders' prepared tensors kept contiguous."""

    def __init__(self, metric, A, cut, At=None):
        self.metric, self.A, self.cut, self.At = metric, A, cut, At
        n = len(A['cell'])
        self.store = {k: torch.empty((n,) + tuple(v.shape[1:]), dtype=v.dtype) for k, v in A.items()}
        self.store_t = {k: torch.empty((n,) + tuple(v.shape[1:]), dtype=v.dtype) for k, v in At.items()} \
            if At is not None else None
        self.k = 0
        self.leader_rows = []

    def view(self, s):
        return {kk: v[:self.k] for kk, v in s.items()}

    def dist(self, i, cut):
        """Distances from state i to every leader (inf where the pre-screen excludes d < cut)."""
        idx = torch.arange(self.k)
        d = self.metric.one_to_many(self.A, i, self.view(self.store), idx, cut)
        if self.store_t is not None:
            d = torch.minimum(d, self.metric.one_to_many(self.A, i, self.view(self.store_t), idx, cut))
        return d

    def run(self, order):
        lab = torch.full((len(self.A['cell']),), -1, dtype=torch.long)
        for i in order:
            if self.k:
                d = self.dist(i, self.cut)
                j = int(d.argmin())
                if d[j] < self.cut:
                    lab[i] = j
                    continue
            for kk, v in self.A.items():
                self.store[kk][self.k] = v[i]
            if self.store_t is not None:
                for kk, v in self.At.items():
                    self.store_t[kk][self.k] = v[i]
            self.leader_rows.append(int(i))
            lab[i] = self.k
            self.k += 1
        return lab


class RDFLeaders:
    """Leader clustering with the coordinator's RDF distance at several cuts at once; RDFs arrive in batches. The
    clustering time of each cut is kept apart (t)."""

    def __init__(self, cuts):
        self.cuts = list(cuts)
        self.CL = [None] * len(cuts)   # [C, L, bins] CDFs of the leaders, per cut
        self.AL = [None] * len(cuts)   # [L, C] active masks
        self.rows = [[] for _ in cuts]
        self.lab = [dict() for _ in cuts]
        self.t = [0.0] * len(cuts)

    def add_batch(self, rows, cb, ab):
        for c, cut in enumerate(self.cuts):
            t0 = time.time()
            L = 0 if self.CL[c] is None else self.CL[c].shape[1]
            Db = C.rdf_dist_prepped(cb, ab, self.CL[c], self.AL[c]) if L else torch.full((len(rows), 0), float('inf'))
            new = []
            for r in range(len(rows)):
                best, j = (float(Db[r].min()), int(Db[r].argmin())) if L else (float('inf'), -1)
                if new:
                    dn = C.rdf_dist_prepped(cb[:, [r]], ab[[r]], cb[:, new], ab[new])[0]
                    k = int(dn.argmin())
                    if float(dn[k]) < best:
                        best, j = float(dn[k]), L + k
                if best < cut:
                    self.lab[c][rows[r]] = j
                else:
                    self.lab[c][rows[r]] = L + len(new)
                    new.append(r)
            if new:
                self.CL[c] = cb[:, new].clone() if not L else torch.cat([self.CL[c], cb[:, new]], 1)
                self.AL[c] = ab[new].clone() if not L else torch.cat([self.AL[c], ab[new]], 0)
                self.rows[c] += [rows[r] for r in new]
            self.t[c] += time.time() - t0

    def labels(self, c, n):
        lab = torch.full((n,), -1, dtype=torch.long)
        for i, j in self.lab[c].items():
            lab[i] = j
        return lab

    def dist_to_leaders(self, c, cb, ab):
        return C.rdf_dist_prepped(cb, ab, self.CL[c], self.AL[c])


def matched_cut(metric, A, At, n, target, c0, L0, max_iter=10):
    """The FD cut whose leader-clustering count is closest to target: geometric bisection from (c0, its run L0),
    doubling or halving until the target is bracketed. At most max_iter further clusterings, so it always returns;
    stops early once the count is within max(1, 1% of target). Returns (cut, FDLeaders, labels, history)."""
    tol = max(1, round(0.01 * target))
    best = (abs(L0[0].k - target), c0, L0[0], L0[1])
    hist = [(c0, L0[0].k, None)]
    lo, hi = (c0, None) if L0[0].k > target else (None, c0)
    for _ in range(max_iter):
        if best[0] <= tol:
            break
        c = hi / 2 if lo is None else lo * 2 if hi is None else math.sqrt(lo * hi)
        t0 = time.time()
        L = FDLeaders(metric, A, c, At)
        lab = L.run(range(n))
        hist.append((c, L.k, time.time() - t0))
        C.log(f'   matched-count search: cut {c:.4g} A -> {L.k} clusters (target {target})')
        if abs(L.k - target) < best[0]:
            best = (abs(L.k - target), c, L, lab)
        if L.k > target:
            lo = c
        else:
            hi = c
    return best[1], best[2], best[3], hist


def judge(rec, methods, title):
    """2x2 counts per method and pairwise disagreements on the usable pairs of rec (deduplicated by pair)."""
    seen, us = set(), []
    for r in rec:
        k = (min(r['i'], r['j']), max(r['i'], r['j']))
        if r['status'] == 'ok' and k not in seen:
            seen.add(k)
            us.append(r)
    yv = np.array([r['n_matched'] >= 20 for r in us])
    lines = [title, '', '| method | clusters | same & matched | same & unmatched (false merges) | different & matched '
             '(false splits) | different & unmatched | errors |', '|---|---|---|---|---|---|---|']
    out = {}
    for m, ncl in methods:
        sv = np.array([r['same'][m] for r in us])
        tp, fp = int((sv & yv).sum()), int((sv & ~yv).sum())
        fn, tn = int((~sv & yv).sum()), int((~sv & ~yv).sum())
        lines.append(f'| {m} | {ncl} | {tp} | {fp} | {fn} | {tn} | {fp + fn} |')
        out[m] = dict(tp=tp, fp=fp, fn=fn, tn=tn)
    lines += ['', '| method A | method B | A same, B different: pairs (COMPACK matched) | A different, B same: pairs '
              '(COMPACK matched) |', '|---|---|---|---|']
    for (m1, _), (m2, _) in itertools.combinations(methods, 2):
        s1 = np.array([r['same'][m1] for r in us])
        s2 = np.array([r['same'][m2] for r in us])
        a_, b_ = s1 & ~s2, ~s1 & s2
        lines.append(f'| {m1} | {m2} | {int(a_.sum())} ({int((a_ & yv).sum())}) | {int(b_.sum())} '
                     f'({int((b_ & yv).sum())}) |')
    lines.append('')
    return lines, dict(n=len(us), matched=int(yv.sum()), per_method=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True, choices=list(C.CAMPAIGNS))
    ap.add_argument('--calibration', required=True, help='calibrate.py output folder name under OUT_ROOT')
    ap.add_argument('--out', required=True)
    ap.add_argument('--band_kT', type=float, default=2.0)
    ap.add_argument('--n_max', type=int, default=20000)
    ap.add_argument('--n_pairs', type=int, default=60, help='COMPACK pairs per method per kind (merge, boundary)')
    ap.add_argument('--rdf_batch', type=int, default=256)
    ap.add_argument('--timeout', type=float, default=120.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    C.setup()
    cal_dir = os.path.join(C.OUT_ROOT, a.calibration)
    with open(os.path.join(cal_dir, 'cuts.json')) as fh:
        cuts = json.load(fh)
    out = C.new_out_dir(a.out)
    C.log('args', vars(a))
    cif_dir, cache_path = os.path.join(out, 'cif'), os.path.join(out, 'compack_cache.pt')
    os.makedirs(cif_dir)
    rng = np.random.default_rng(a.seed)
    from energy_sampling.eval.campaign_compack import crystal_key
    co, cfg, reg, ref = C.load_campaign(a.campaign)
    sg = cfg.sg
    st = C.shard_states(a.campaign, co, cfg, reg, ref, a.band_kT)
    n_reg_band = int(sum(1 for e in reg['hits']['energy'] if e <= ref + a.band_kT * cfg.kT))
    n_basins_band = int(len(set(b for b, e in zip(reg['hits']['basin'], reg['hits']['energy'])
                                if e <= ref + a.band_kT * cfg.kT)))
    del reg
    n_all = len(st['energy'])
    sub = np.sort(rng.choice(n_all, min(a.n_max, n_all), replace=False))
    s = C.subset(st, sub)
    del st
    n = len(sub)
    order = torch.argsort(s['energy'])               # ascending energy
    s = C.subset(s, order)                           # rows now in ascending energy
    s['key'] = [crystal_key(p, h) for p, h in zip(s['params'], s['hand'])]

    def cal_cut(k):
        v = cuts[k].get('rule_cut_own', cuts[k]['rule_cut'])
        return float(v[0]), ('own design' if 'rule_cut_own' in cuts[k] else 'pooled')

    c_own = float(cfg.identity_cut)
    (c_rdf, src_r), (c_fd, src_f), (c_tw, src_t) = cal_cut('d_rdf'), cal_cut('d_fd'), cal_cut('d_tw')
    mi = F.molecule_info(cfg.mol_path)
    tmpl = C.molecule_template(mi)
    use_tw = mi['mirror_rmsd_matched'] < 0.1 and tmpl['perm_is_automorphism']
    lines = [f'# RDF vs FD identity on {a.campaign} endpoints (sg {sg}, eLJ)', '',
             f'Endpoints: physical shard states within {a.band_kT} kT of the reference {ref:.4f} (kT {cfg.kT:.4f}): '
             f'{n_all} (the registry counts {n_reg_band} hits in the band, in {n_basins_band} basins; '
             f'{int((s["basin"] < 0).sum())} of the sampled states have no matching hit); {n} drawn at random (seed '
             f'{a.seed}), clustered in ascending energy. Cuts: campaign {c_own} (RDF); calibrated RULE cuts from '
             f'{cal_dir}: RDF {c_rdf:.4g} ({src_r}), FD {c_fd:.4g} A ({src_f}), FD with twin {c_tw:.4g} A ({src_t}). '
             f'Mirror: matched RMSD {mi["mirror_rmsd_matched"]:.3f} A (largest atom deviation '
             f'{mi["mirror_maxdev_matched"]:.3f} A; identity-order RMSD {mi["mirror_rmsd"]:.3f} A), matching '
             f'{"preserves" if tmpl["perm_is_automorphism"] else "does not preserve"} the bond graph, so the twin '
             f'variants are {"run" if use_tw else "not run"} (gate: matched RMSD < 0.1 A and bond-preserving). '
             f'COMPACK route {C.RUNNER}. Arguments {vars(a)}.', '']

    # ---- FD: canonicalise, prepare, cluster at the calibrated cuts (each timed end to end)
    metric = F.FDMetric(sg, mi['rg'])
    t0 = time.time()
    can = F.canonical(sg, s['params'], s['hand'])
    t_can_all = time.time() - t0
    t0 = time.time()
    A = metric.prepare(can)
    t_prep = time.time() - t0
    t0 = time.time()
    fdl = FDLeaders(metric, A, c_fd)
    lab_fd = fdl.run(range(n))
    t_fd_cl = time.time() - t0
    C.log(f'FD clustering: {fdl.k} clusters in {t_fd_cl:.1f} s')
    leaders = {'fd_cal': fdl}
    labs = {}
    tw_t = {}
    if use_tw:
        t0 = time.time()
        twc = F.twin_canonical(sg, can, mi['sigma'])
        tw_t['can'] = time.time() - t0
        t0 = time.time()
        At = metric.prepare(twc)
        tw_t['prep'] = time.time() - t0
        del twc
        t0 = time.time()
        fdt = FDLeaders(metric, A, c_tw, At)
        labs['fd_tw_cal'] = fdt.run(range(n))
        tw_t['cl'] = time.time() - t0
        leaders['fd_tw_cal'] = fdt
        C.log(f'FD+twin clustering: {fdt.k} clusters in {tw_t["cl"]:.1f} s')
    else:
        At = None

    # ---- RDF (featurise in energy order, cluster at two cuts in one pass, clustering time per cut kept apart)
    rl = RDFLeaders([c_own, c_rdf])
    t_feat, t_rprep = 0.0, 0.0
    t0_all = time.time()
    for lo in range(0, n, a.rdf_batch):
        rows = list(range(lo, min(n, lo + a.rdf_batch)))
        t0 = time.time()
        R = C.rdfs(co, cfg, s['params'][rows], s['hand'][rows])
        t_feat += time.time() - t0
        t0 = time.time()
        cb, ab = C.rdf_prep(R)
        t_rprep += time.time() - t0
        rl.add_batch(rows, cb, ab)
        if (lo // a.rdf_batch) % 20 == 0:
            C.log(f'RDF {lo + len(rows)}/{n}: clusters {[len(r) for r in rl.rows]}; {C.machine_note()}')
    t_rdf_all = time.time() - t0_all
    labs = dict(reg=s['basin'].clone(), rdf_own=rl.labels(0, n), rdf_cal=rl.labels(1, n), fd_cal=lab_fd, **labs)
    n_own = len(rl.rows[0])

    # ---- FD cuts matched to rdf_own's cluster count
    search = {}
    c_fd_m, L, lab, hist = matched_cut(metric, A, None, n, n_own, c_fd, (fdl, lab_fd))
    labs['fd_m'], leaders['fd_m'], search['fd_m'] = lab, L, dict(cut=c_fd_m, history=hist)
    if use_tw:
        c_tw_m, L, lab, hist = matched_cut(metric, A, At, n, n_own, c_tw, (fdt, labs['fd_tw_cal']))
        labs['fd_tw_m'], leaders['fd_tw_m'], search['fd_tw_m'] = lab, L, dict(cut=c_tw_m, history=hist)
    methods = list(labs)
    cnt = {m: int(len(torch.unique(v[v >= 0]))) for m, v in labs.items()}
    C.log('clusters', cnt)

    # timing of single distances
    ia = torch.as_tensor(rng.integers(0, n, 20000))
    ib = torch.as_tensor(rng.integers(0, n, 20000))
    t0 = time.time()
    metric.pairs(A, ia, A, ib)
    t_fd_pair = (time.time() - t0) / 20000
    Rs = C.rdfs(co, cfg, s['params'][:200], s['hand'][:200])
    cs, as_ = C.rdf_prep(Rs)
    t0 = time.time()
    C.rdf_dist_prepped(cs, as_, cs, as_)
    t_rdf_pair = (time.time() - t0) / 200 ** 2

    # ---- counts and agreement
    cutv = dict(reg=c_own, rdf_own=c_own, rdf_cal=c_rdf, fd_cal=c_fd, fd_tw_cal=c_tw, fd_m=search['fd_m']['cut'])
    if use_tw:
        cutv['fd_tw_m'] = search['fd_tw_m']['cut']
    lines += [f'*Clusters per method on the {n} endpoints, and adjusted Rand index between methods (rows with a registry '
              f'hit only where "reg" is involved: {int((labs["reg"] >= 0).sum())}). reg = the campaign registry as run '
              f'(arrival order); the others are re-run here in ascending energy. Order alone moves counts as much as '
              f'the method does (reg against rdf_own, same distance and cut), so counts and ARI describe granularity, '
              f'not accuracy.*', '',
              '| method | cut | clusters | singletons | largest cluster | ' + ' | '.join(f'ARI vs {m}' for m in methods) + ' |',
              '|---|---|---|---|---|' + '---|' * len(methods)]
    for m in methods:
        v = labs[m]
        u, c = torch.unique(v[v >= 0], return_counts=True)
        aris = []
        for m2 in methods:
            ok = (labs[m] >= 0) & (labs[m2] >= 0)
            aris.append(f'{C.ari(labs[m][ok].numpy(), labs[m2][ok].numpy()):.3f}')
        lines.append(f'| {m} | {cutv[m]:.4g}{" A" if m.startswith("fd") else ""} | {len(u)} | {int((c == 1).sum())} | '
                     f'{int(c.max())} | ' + ' | '.join(aris) + ' |')
    lines.append('')
    for m, sv in search.items():
        lines.append(f'{m}: cut {sv["cut"]:.4g} A from the search (cut A -> clusters, target {n_own}): '
                     + ', '.join(f'{c:.4g} -> {k}' for c, k, _ in sv['history']) + '.')
    lines.append('')

    # ---- COMPACK on sampled pairs
    def nearest_other_leader(m, i):
        """Index (state row) of the leader of the nearest other cluster of state i under method m."""
        if m.startswith('fd'):
            Lm = leaders[m]
            d = Lm.dist(i, float('inf'))
            own = int(labs[m][i])
            d[own] = float('inf')
            j = int(d.argmin())
            return Lm.leader_rows[j], float(d[j])
        c = 0 if m == 'rdf_own' else 1
        R1 = C.rdfs(co, cfg, s['params'][[i]], s['hand'][[i]])
        c1, a1 = C.rdf_prep(R1)
        d = rl.dist_to_leaders(c, c1, a1)[0]
        own = int(labs[m][i])
        d[own] = float('inf')
        j = int(d.argmin())
        return rl.rows[c][j], float(d[j])

    eval_methods = [m for m in methods if m != 'reg']
    sample = []  # (method, kind, i, j)
    for m in eval_methods:
        v = labs[m]
        u, c = torch.unique(v, return_counts=True)
        big = set(u[c > 1].tolist())
        cand = [i for i in range(n) if int(v[i]) in big]
        for i in rng.choice(cand, min(a.n_pairs, len(cand)), replace=False):
            mem = torch.nonzero(v == v[i]).flatten().tolist()
            mem.remove(int(i))
            sample.append((m, 'merge', int(i), int(rng.choice(mem))))
        for i in rng.choice(n, a.n_pairs, replace=False):
            j, _ = nearest_other_leader(m, int(i))
            sample.append((m, 'boundary', int(i), int(j)))
    need = sorted({x for p in sample for x in p[2:]})
    for lo in range(0, len(need), 64):
        nd = need[lo:lo + 64]
        C.write_cifs(co, cfg, C.rebuild(co, cfg, s['params'][nd], s['hand'][nd]), [s['key'][i] for i in nd], cif_dir)
    extra = [torch.load(os.path.join(cal_dir, 'compack_cache.pt'), weights_only=False)]
    cache = {}
    t0 = time.time()
    res = C.compack([(s['key'][i], s['key'][j]) for _, _, i, j in sample], cif_dir, cache, cache_path, tmpl, a.timeout,
                    extra)
    t_cmp = time.time() - t0
    rec = []
    for m, kind, i, j in sample:
        r = res[C.pair_key(s['key'][i], s['key'][j])]
        same = {mm: bool(labs[mm][i] == labs[mm][j] and labs[mm][i] >= 0) for mm in methods}
        rec.append(dict(method=m, kind=kind, i=i, j=j, status=r['status'], n_matched=r['n_matched'], rmsd=r['rmsd'],
                        same=same, dE=float(abs(s['energy'][i] - s['energy'][j]))))
    lines += [f'*COMPACK (20/20) on sampled pairs, per method: "within" = a random state of a non-singleton cluster and a '
              f'random other member (a false merge if COMPACK does not match them); "boundary" = a random state and the '
              f'leader of the nearest other cluster under the method (a false split if COMPACK matches them). '
              f'{a.n_pairs} of each per method; pairs COMPACK could not judge are excluded from the rates and counted.*',
              '', '| method | clusters | within: usable | within: unmatched (false merges) | false-merge rate | '
              'boundary: usable | boundary: matched (false splits) | false-split rate | not judged |',
              '|---|---|---|---|---|---|---|---|---|']
    for m in eval_methods:
        w = [r for r in rec if r['method'] == m and r['kind'] == 'merge']
        b = [r for r in rec if r['method'] == m and r['kind'] == 'boundary']
        wu = [r for r in w if r['status'] == 'ok']
        bu = [r for r in b if r['status'] == 'ok']
        fm = sum(r['n_matched'] < 20 for r in wu)
        fs = sum(r['n_matched'] >= 20 for r in bu)
        nf = sum(r['status'] != 'ok' for r in w + b)
        lines.append(f'| {m} | {cnt[m]} | {len(wu)} | {fm} | {fm / max(1, len(wu)):.2f} | {len(bu)} | {fs} | '
                     f'{fs / max(1, len(bu)):.2f} | {nf} |')
    lines.append('')

    # ---- matched operating points, each on one common pair sample
    judged = {}
    sets = [('equal cluster counts', ['rdf_own', 'fd_m'] + (['fd_tw_m'] if use_tw else []),
             f'*Operating point 1, equal cluster counts: RDF at the campaign cut and FD (and FD+twin) at the cut that '
             f'gives the same number of clusters on these {n} endpoints. Common sample: every usable pair drawn by these '
             f'methods (within and boundary, duplicates once), each judged by every method; "same" = one cluster. '
             f'Rates are on this sample, which sits on these methods\' decision boundaries, not on all pairs.*'),
            ('calibrated cuts', ['rdf_cal', 'fd_cal'] + (['fd_tw_cal'] if use_tw else []),
             f'*Operating point 2, each distance at its own calibrated RULE cut (P(COMPACK 20/20) >= 0.95 on its own '
             f'pair design), i.e. one precision target. Common sample: every usable pair drawn by these methods, judged '
             f'by every method, as above.*')]
    for name, ms, cap in sets:
        sub_rec = [r for r in rec if r['method'] in ms]
        l2, jd = judge(sub_rec, [(m, cnt[m]) for m in ms], cap)
        lines += [f'## {name}', ''] + l2
        judged[name] = jd
    l2, jd = judge(rec, [(m, cnt[m]) for m in methods], f'*All usable sampled pairs of every method, judged by every '
                                                         f'method (for reference; the sample mixes operating points).*')
    lines += ['## All sampled pairs', ''] + l2
    judged['all'] = jd

    # ---- cost
    tw_tot = (t_can_all + tw_t['can'] + t_prep + tw_t['prep'] + tw_t['cl']) if use_tw else float('nan')
    fd_tot = t_can_all + t_prep + t_fd_cl
    rdf_tot = t_feat + t_rprep + rl.t[0]
    lines += [f'*Cost on this machine for the {n} endpoints, each method end to end at one cut (FD: canonicalise, '
              f'prepare, cluster; RDF: rebuild, standardize, featurise, prepare, cluster). One process, torch threads 2, '
              f'other jobs on the machine: {C.machine_note()}.*', '',
              '| step | FD at its calibrated cut | FD+twin at its calibrated cut | RDF at the campaign cut |',
              '|---|---|---|---|',
              f'| describe (FD canonicalise; RDF rebuild + standardize + featurise) (s) | {t_can_all:.1f} | '
              f'{t_can_all + tw_t.get("can", float("nan")):.1f} | {t_feat:.1f} |',
              f'| prepare (FD per-crystal tensors; RDF CDFs) (s) | {t_prep:.2f} | {t_prep + tw_t.get("prep", float("nan")):.2f} '
              f'| {t_rprep:.2f} |',
              f'| leader clustering (s) | {t_fd_cl:.1f} ({cnt["fd_cal"]} clusters) | {tw_t.get("cl", float("nan")):.1f} '
              f'({cnt.get("fd_tw_cal", "-")} clusters) | {rl.t[0]:.1f} ({cnt["rdf_own"]} clusters) |',
              f'| end to end (s) | {fd_tot:.1f} | {tw_tot:.1f} | {rdf_tot:.1f} |',
              f'| ratio RDF / method | {rdf_tot / fd_tot:.1f} | {rdf_tot / tw_tot:.1f} | 1 |', '',
              f'Per item: FD canonicalise {1e3 * t_can_all / n:.2f} ms per structure; FD distance {1e6 * t_fd_pair:.1f} us '
              f'per pair ({27 * metric.m} images); RDF featurise {1e3 * t_feat / n:.2f} ms per structure; RDF distance '
              f'{1e6 * t_rdf_pair:.1f} us per pair; RDF clustering at the calibrated cut {rl.t[1]:.1f} s '
              f'({cnt["rdf_cal"]} clusters); RDF featurise + clustering at both cuts in one pass {t_rdf_all:.1f} s. '
              f'COMPACK, {len(set(C.pair_key(s["key"][i], s["key"][j]) for _, _, i, j in sample))} pairs: {t_cmp:.1f} s.',
              '']
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    torch.save(dict(labels=labs, energy=s['energy'], keys=s['key'], basin=s['basin'], records=rec, cuts=cutv,
                    leaders={m: L.leader_rows for m, L in leaders.items()}, rdf_leaders=rl.rows, search=search,
                    judged=judged, use_tw=use_tw, runner=C.RUNNER,
                    timing=dict(t_can_all=t_can_all, t_prep=t_prep, t_fd_cl=t_fd_cl, tw=tw_t, t_feat=t_feat,
                                t_rprep=t_rprep, t_rdf_cl=rl.t, t_fd_pair=t_fd_pair, t_rdf_pair=t_rdf_pair,
                                t_rdf_all=t_rdf_all, t_cmp=t_cmp)),
               os.path.join(out, 'compare.pt'))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
