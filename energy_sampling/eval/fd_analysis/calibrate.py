"""COMPACK calibration of crystal identity for one campaign, by RDF distance (the campaign's atomwise metric) and by
the FD distance (fd_distance.FDMetric, Angstrom), with and without the molecular-mirror twin.

Pool: distinct shard states (exact repeats removed) within band_kT of the campaign reference, max_pool of them at random.
Pairs: three designs under one RULE, each stratified by one distance only. RDF-stratified (campaign_compack's: round 1
evenly over [0, span x campaign cut] in bins, round 2 over the round-1 logistic transition), FD-stratified and
FD+twin-stratified (round 1 over log-spaced bins on [0, 10] A, round 2 over the round-1 logistic transition in log d;
the twin design draws from its own random stream after the other two, so they are the draws of the earlier build).
Every pair gets all three distances, exact (the pool matrices hold the pre-screen lower bound above fd_cap; the sampled
pairs are recomputed in full). A match is COMPACK 20/20 molecules on repaired CIFs (common.compack_pair_repaired); a
'failed' or 'repair_failed' pair is excluded from the RULE (a sensitivity line counts 'failed' as a non-match). RULE cut:
the largest distance at which the isotonic P(match | d) is >= 0.95 (campaign_compack.rule_summary), with a pair
bootstrap. Reported pooled over all pairs and per design; the own design of a distance (RDF: rdf + on-disk; FD: fd;
FD+twin: tw) is the one whose selection depends on that distance alone, and its cut is the one compare.py uses. Also:
precision / recall of 'same' = d <= cut on the calibration pairs (a stratified design, not all pairs), the best-F1 cut,
AUC, recall at a 5% false-merge rate, and paired-bootstrap AUC differences.

Controls: a state against itself in another cell setting (campaign_compack.setting_controls), and against its own FD
canonical form rebuilt from crystal_params (must match 20/20). Twin test: each sampled state against its molecular-mirror
twin (t, R sigma) and, for the record, against (-t, R C2_z) built from its canonical form.

On NEHZOR the on-disk calibration's pairs (campaigns/nehzor_elj_sep28/compack, read only) are found in the shards by
crystal_key, written to CIFs here and rerun on the repaired route; their stored results (the old route) are kept only
for the before/after table. --previous names an earlier calibrate output folder whose COMPACK cache (old route) gives
the same table for the pairs it shares with this run.

    python -m energy_sampling.eval.fd_analysis.calibrate --campaign mipcas --out calibrate_mipcas
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from energy_sampling.eval.fd_analysis import common as C
from energy_sampling.eval.fd_analysis import fd_distance as F

F64 = torch.float64
OWN = {'d_rdf': ('rdf', 'old'), 'd_fd': ('fd',), 'd_tw': ('tw',)}
NAMES = (('d_rdf', 'RDF distance'), ('d_fd', 'FD distance (A)'), ('d_tw', 'FD distance with mirror twin (A)'))


def draw(D, lo, hi, n, rng, uses, cap, taken):
    from energy_sampling.eval.campaign_compack import draw_pairs
    return draw_pairs(D, lo, hi, n, rng, uses, cap, taken)


def spread(D, edges, budget, rng, uses, cap, taken):
    per = max(1, budget // (len(edges) - 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out += draw(D, lo, hi, per, rng, uses, cap, taken)
    return out


def leader_rdf_check(co, cfg, reg, name, n=12, rng=None):
    """Recompute the RDF of n registry leaders (the first hit of each basin, found in its shard by run, cursor and
    energy) by the coordinator's route at this MXT commit; return the RDF distances to the stored leaders."""
    h = reg['hits']
    first = {}
    for i, b in enumerate(h['basin']):
        if b not in first:
            first[b] = i
    js = rng.choice(len(reg['basin_E']), n, replace=False)
    P, H, J = [], [], []
    for j in js:
        i = first[int(j)]
        run, cur, E = h['run'][i], int(h['cursor'][i]), float(h['energy'][i])
        f = glob.glob(os.path.join(C.CAMPAIGNS[name]['dir'], 'shards', run, f'{cur:08d}_*.pt'))
        if len(f) != 1:
            continue
        s = torch.load(f[0], map_location='cpu', weights_only=False)
        r = torch.nonzero(s['energy'] == E).flatten()
        if len(r) < 1:
            continue
        P.append(s['params'][r[0]].float())
        H.append(s['handedness'][r[0]].float())
        J.append(int(j))
    R = C.rdfs(co, cfg, torch.stack(P), torch.stack(H))
    d = co.rdf_distance_matrix(R, reg['leaders'][J])
    return torch.diagonal(d), J


def analyse(name, d, y, ok, nm, nboot, seed, edges=None):
    """Lines of the per-distance report (pooled pairs) and a dict of cuts."""
    from energy_sampling.eval.campaign_compack import auc, rule_summary
    dd, yy = d[ok], y[ok]
    lines = []
    if edges is None:
        edges = np.linspace(0, dd.max() + 1e-9, 13)
    lines += [f'| {name} bin | pairs | matched 20/20 | matched >= 15/20 | P(match) |', '|---|---|---|---|---|']
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dd >= lo) & (dd < hi)
        if m.sum():
            lines.append(f'| [{lo:.4g}, {hi:.4g}) | {int(m.sum())} | {int(yy[m].sum())} | {int((nm[ok][m] >= 15).sum())} | '
                         f'{yy[m].mean():.2f} |')
    m = dd >= edges[-1]
    if m.sum():
        lines.append(f'| >= {edges[-1]:.4g} | {int(m.sum())} | {int(yy[m].sum())} | {int((nm[ok][m] >= 15).sum())} | '
                     f'{yy[m].mean():.2f} |')
    out = dict(n=int(len(dd)), matched=int(yy.sum()))
    if len(yy) > 4 and 0 < yy.sum() < len(yy):
        rs = rule_summary(dd, yy, nboot, seed=seed)
        bc, brow = C.best_f1_cut(dd, yy)
        out.update(rule_cut=rs['cut'], d_hi=rs['d_hi'], auc=auc(dd, yy), best_f1_cut=bc, best_f1=brow)
        # sensitivity: engine failures counted as non-matches
        fail = ~ok & (nm == 0)
        d2 = np.concatenate([dd, d[fail]])
        y2 = np.concatenate([yy, np.zeros(int(fail.sum()))])
        rs2 = rule_summary(d2, y2, nboot, seed=seed)
        out.update(rule_cut_fail_as_nonmatch=rs2['cut'])
        cuts = sorted(set([rs['cut'][0], bc] + list(np.quantile(dd[yy == 1], [0.5, 0.8, 0.9, 0.95, 0.99]))))
        lines += ['', f'RULE on all usable pairs (isotonic P(match) >= 0.95): **{name} cut {rs["cut"][0]:.4g} '
                      f'[{rs["cut"][1]:.4g}, {rs["cut"][2]:.4g}]**; d_hi (P >= 0.05) {rs["d_hi"][0]:.4g} '
                      f'[{rs["d_hi"][1]:.4g}, {rs["d_hi"][2]:.4g}] (95% pair bootstrap, {nboot}); AUC {out["auc"]:.3f}; '
                      f'best-F1 cut {bc:.4g} (F1 {brow[-1]:.3f}, precision {brow[5]:.3f}, recall {brow[6]:.3f}). '
                      f'With the {int(fail.sum())} engine failures counted as non-matches the RULE cut is '
                      f'{rs2["cut"][0]:.4g} [{rs2["cut"][1]:.4g}, {rs2["cut"][2]:.4g}].', '',
                  f'*Precision and recall of "same" = {name} <= cut against COMPACK 20/20, on the {len(dd)} usable '
                  f'calibration pairs of all designs (stratified designs: these are rates on the sampled pairs, weighted '
                  f'toward near pairs, not on all pairs of the campaign). Cuts: the RULE cut, the best-F1 cut, and '
                  f'quantiles (50/80/90/95/99%) of {name} over matched pairs.*', '',
                  f'| {name} cut | TP | FP | FN | TN | precision | recall | F1 |', '|---|---|---|---|---|---|---|---|']
        for r in C.pr_table(dd, yy, cuts):
            lines.append(f'| {r[0]:.4g} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} |')
    else:
        lines.append('No cut: compared pairs all matched or all unmatched.')
    return lines, out


def design_tables(rows, ok, y, nboot, seed):
    """Per-design and own-design RULE cuts, AUCs and recall at a 5% false-merge rate for every distance; paired-bootstrap
    AUC differences against RDF. Returns (lines, own-design summaries)."""
    from energy_sampling.eval.campaign_compack import auc, rule_summary
    des = np.array([r['design'] for r in rows])
    D = {k: np.array([r[k] for r in rows]) for k, _ in NAMES}
    sets = [(dz, des == dz) for dz in ('rdf', 'fd', 'tw', 'old') if (des == dz).any()] + [('all', np.ones(len(rows), bool))]
    lines = ['*Each distance judged on each pair design separately, on its own design (the pairs selected by that '
             'distance alone: RDF rdf + on-disk, FD fd, FD+twin tw), and on all pairs pooled. RULE cut: largest distance '
             f'with isotonic P(COMPACK 20/20 | d) >= 0.95, 95% pair bootstrap ({nboot}). AUC: P(an unmatched pair lies '
             'farther than a matched one). Recall at 5% false merges: share of matched pairs with d <= c at the largest '
             'c with at most 5% of the unmatched pairs at d <= c. All rates are on the stratified sample named in the '
             'row, not on all pairs of the campaign; designs that sample near pairs of one distance favour that distance '
             'in its own row.*', '',
             '| distance | pair set | usable pairs | matched 20/20 | RULE cut [95% CI] | AUC | recall at 5% false merges '
             '(at d) |', '|---|---|---|---|---|---|---|']
    own_out = {}
    for key, name in NAMES:
        d = D[key]
        own_mask = np.isin(des, OWN[key])
        for dz, msk in sets + [('own', own_mask)]:
            m = ok & msk
            if not m.any() or y[m].min() == y[m].max():
                lines.append(f'| {name} | {dz} | {int(m.sum())} | {int(y[m].sum())} | - | - | - |')
                continue
            rs = rule_summary(d[m], y[m], nboot, seed=seed)
            rc, cc = C.recall_at_fpr(d[m], y[m], 0.05)
            a = auc(d[m], y[m])
            lines.append(f'| {name} | {dz} | {int(m.sum())} | {int(y[m].sum())} | {rs["cut"][0]:.4g} '
                         f'[{rs["cut"][1]:.4g}, {rs["cut"][2]:.4g}] | {a:.3f} | {rc:.3f} ({cc:.4g}) |')
            if dz == 'own':
                own_out[key] = dict(rule_cut=rs['cut'], auc=a, n=int(m.sum()), matched=int(y[m].sum()),
                                    designs=list(OWN[key]))
    lines.append('')
    rng = np.random.default_rng(seed + 7)
    lines += ['*Paired-bootstrap AUC differences (1000 resamples of the usable pairs of each set; one-class resamples '
              'dropped). Positive: the FD variant separates COMPACK matches from non-matches better than RDF on that '
              'set.*', '', '| pair set | usable pairs | AUC(FD) - AUC(RDF) [95% CI] | AUC(FD+twin) - AUC(RDF) [95% CI] |',
              '|---|---|---|---|']
    for dz, msk in sets:
        idx = np.nonzero(ok & msk)[0]
        if not len(idx) or y[idx].min() == y[idx].max():
            continue
        cells = []
        for k1 in ('d_fd', 'd_tw'):
            bs = []
            for _ in range(1000):
                b = rng.choice(idx, len(idx))
                if y[b].min() == y[b].max():
                    continue
                bs.append(auc(D[k1][b], y[b]) - auc(D['d_rdf'][b], y[b]))
            pt = auc(D[k1][idx], y[idx]) - auc(D['d_rdf'][idx], y[idx])
            cells.append(f'{pt:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]')
        lines.append(f'| {dz} | {len(idx)} | {cells[0]} | {cells[1]} |')
    lines.append('')
    return lines, own_out


def outcome(status, nm):
    if status != 'ok':
        return 'failed' if status == 'failed' else 'other'
    return 'matched' if nm >= 20 else 'unmatched'


def before_after(pairs_old_new, title):
    """Counts of (old route outcome, repaired route outcome) for pairs compared both ways."""
    cats = ['matched', 'unmatched', 'failed', 'other']
    cnt = {(a, b): 0 for a in cats for b in cats}
    for a, b in pairs_old_new:
        cnt[(a, b)] += 1
    lines = [title, '', '| old route \\ repaired route | ' + ' | '.join(cats) + ' | total |', '|---' * (len(cats) + 2) + '|']
    for a in cats:
        tot = sum(cnt[(a, b)] for b in cats)
        if tot:
            lines.append(f'| {a} | ' + ' | '.join(str(cnt[(a, b)]) for b in cats) + f' | {tot} |')
    lines.append('')
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True, choices=list(C.CAMPAIGNS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--band_kT', type=float, default=2.0)
    ap.add_argument('--max_pool', type=int, default=3000)
    ap.add_argument('--budget_rdf', type=int, default=240)
    ap.add_argument('--budget_fd', type=int, default=240)
    ap.add_argument('--budget_tw', type=int, default=240)
    ap.add_argument('--span', type=float, default=3.0)
    ap.add_argument('--bins', type=int, default=12)
    ap.add_argument('--cap', type=int, default=3, help='pairs per state per design')
    ap.add_argument('--fd_cap', type=float, default=6.0, help='pool-matrix FD entries above this hold lower bounds')
    ap.add_argument('--n_twin', type=int, default=12)
    ap.add_argument('--n_controls', type=int, default=4)
    ap.add_argument('--nboot', type=int, default=500)
    ap.add_argument('--timeout', type=float, default=120.0)
    ap.add_argument('--previous', default=None, help='earlier calibrate output folder (old COMPACK route), for the '
                                                    'before/after table only')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    C.setup()
    out = C.new_out_dir(a.out)
    C.log('args', vars(a))
    cif_dir, cache_path = os.path.join(out, 'cif'), os.path.join(out, 'compack_cache.pt')
    os.makedirs(cif_dir)
    rng = np.random.default_rng(a.seed)
    from energy_sampling.eval.campaign_compack import crystal_key, fit_logistic, d_at, setting_controls
    co, cfg, reg, ref = C.load_campaign(a.campaign)
    sg = cfg.sg
    mi = F.molecule_info(cfg.mol_path)
    tmpl = C.molecule_template(mi)
    t_start = time.time()
    lines = [f'# Identity calibration against COMPACK: {a.campaign} (sg {sg}, Z\'=1, eLJ)', '',
             f'Campaign {C.CAMPAIGNS[a.campaign]["dir"]} (read only); reference {ref:.4f}, kT {cfg.kT:.4f}; campaign '
             f'identity cut {cfg.identity_cut} ({cfg.rdf_mode} RDF, calibrated: {cfg.identity_calibrated}). '
             f'Conformer {cfg.mol_path}. Arguments {vars(a)}.', '',
             f'COMPACK route {C.RUNNER}: each CIF repaired before the comparison (one atom per site at the written '
             f'coordinates, each molecule given the conformer\'s {len(tmpl["bonds"])}-bond graph as the CSD reader '
             f'perceives it on the isolated molecule; the result must be Z whole molecules), then PackingSimilarity at '
             f'shell 20 and CCDC defaults.', '']

    # ---- RDF route check against the registry's stored leaders
    dchk, J = leader_rdf_check(co, cfg, reg, a.campaign, rng=rng)
    lines += [f'RDF route check: {len(J)} registry leaders rebuilt from their shard rows and re-featurised by the '
              f'coordinator route at this MXT commit: RDF distance to the stored leader RDF max {float(dchk.max()):.2e} '
              f'(campaign cut {cfg.identity_cut}).', '']
    C.log('leader RDF check', dchk.tolist())

    # ---- pool
    band_old = 3.0 if 'old_compack' in C.CAMPAIGNS[a.campaign] else a.band_kT
    st_all = C.shard_states(a.campaign, co, cfg, reg, ref, max(a.band_kT, band_old))
    del reg
    in_band = torch.nonzero(st_all['energy'] <= ref + a.band_kT * cfg.kT).flatten()
    st = C.distinct(C.subset(st_all, in_band))
    n_band, n_dist = len(in_band), len(st['key'])
    sub = np.sort(rng.choice(n_dist, min(a.max_pool, n_dist), replace=False))
    pool = C.subset(st, sub)
    n = len(sub)
    C.log(f'{n_band} shard states within {a.band_kT} kT, {n_dist} distinct, pool {n}')

    # ---- distances on the pool
    t0 = time.time()
    R = C.rdfs(co, cfg, pool['params'], pool['hand'])
    t_rdf = (time.time() - t0) / n
    ca, aa = C.rdf_prep(R)
    del R
    D_rdf = C.rdf_dist_prepped(ca, aa, ca, aa).double().numpy()
    del ca, aa
    metric = F.FDMetric(sg, mi['rg'])
    t0 = time.time()
    can = F.canonical(sg, pool['params'], pool['hand'])
    t_can = (time.time() - t0) / n
    A = metric.prepare(can)
    D_fd_t, exact = metric.matrix(A, a.fd_cap)
    tw = F.twin_canonical(sg, can, mi['sigma'])
    At = metric.prepare(tw)
    X = torch.empty(n, n, dtype=F64)
    for lo in range(0, n, 64):
        rows = torch.arange(lo, min(n, lo + 64))
        ia, ib = rows.repeat_interleave(n), torch.arange(n).repeat(len(rows))
        lb = metric.lower_bound(A, ia, At, ib)
        cand = torch.nonzero(lb < a.fd_cap).flatten()
        v = lb.clone()
        if len(cand):
            v[cand] = metric.pairs(A, ia[cand], At, ib[cand])
        X[rows] = v.reshape(len(rows), n)
    D_fd = D_fd_t.numpy()
    D_tw = torch.minimum(D_fd_t, torch.minimum(X, X.T)).numpy()
    del X
    stat = {k: can['status'].count(k) for k in set(can['status'])}
    lines += [f'Pool: {n_band} shard states within {a.band_kT} kT of the reference, {n_dist} distinct (exact repeats '
              f'removed); {n} drawn at random. Canonical status {stat}. RDF {1e3 * t_rdf:.1f} ms per state (rebuild, '
              f'standardize, featurise); canonicalise {1e3 * t_can:.2f} ms per state. Pool-matrix FD entries above '
              f'{a.fd_cap} A hold the pre-screen lower bound ({int((~exact).sum())} of {n * n}; used only to draw '
              f'pairs); every sampled pair\'s FD distances are exact.', '']
    C.log('distances done')

    # ---- pair designs (rdf and fd: the earlier build's draws; tw: its own stream, drawn after them)
    uses_r, uses_f, uses_t, taken = (np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64),
                                     np.zeros(n, dtype=np.int64), set())
    cut0 = float(cfg.identity_cut)
    edges_r = np.linspace(0, a.span * cut0, a.bins + 1)
    edges_f = np.concatenate([[0.0], np.logspace(-2.5, 1.0, a.bins)])
    r1 = spread(D_rdf, edges_r, a.budget_rdf // 2, rng, uses_r, a.cap, taken)
    f1 = spread(D_fd, edges_f, a.budget_fd // 2, rng, uses_f, a.cap, taken)

    def cifs(idx_list):
        need = sorted(set(idx_list))
        cr = C.rebuild(co, cfg, pool['params'][need], pool['hand'][need])
        C.write_cifs(co, cfg, cr, [pool['key'][i] for i in need], cif_dir)

    cache = {}

    def run(pairs):
        cifs([i for p in pairs for i in p])
        return C.compack([(pool['key'][i], pool['key'][j]) for i, j in pairs], cif_dir, cache, cache_path, tmpl,
                         a.timeout)

    t_cmp0 = time.time()
    res = run(r1 + f1)

    def y_of(pairs):
        r = [res[C.pair_key(pool['key'][i], pool['key'][j])] for i, j in pairs]
        ok = np.array([x['status'] == 'ok' for x in r])
        nm = np.array([x['n_matched'] for x in r])
        return ok, (nm >= 20).astype(float), nm

    # round 2, each design in its own transition
    def round2(pairs, D, logd, budget, uses, rng_):
        d = np.array([D[i, j] for i, j in pairs])
        ok, y, _ = y_of(pairs)
        x = np.log10(np.maximum(d, 1e-4)) if logd else d
        if ok.sum() > 4 and 0 < y[ok].sum() < ok.sum():
            d50, b = fit_logistic(x[ok], y[ok])
            lo, hi = d_at(0.99, d50, b), d_at(0.01, d50, b)
            if logd:
                lo, hi = 10 ** lo, 10 ** hi
            lo, hi = max(0.0, lo), min(float(np.max(D)), hi)
            why = f'round-1 logistic in {"log10 d" if logd else "d"}: d50 {10 ** d50 if logd else d50:.4g}'
        else:
            lo, hi, why = float(edges_r[0] if not logd else 0.0), float(edges_r[-1] if not logd else 10.0), 'no fit'
        if hi <= lo:
            return [], (lo, hi, why)
        e = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), a.bins // 2 + 1) if logd else np.linspace(lo, hi, a.bins // 2 + 1)
        return spread(D, e, budget, rng_, uses, a.cap, taken), (lo, hi, why)

    r2, why_r = round2(r1, D_rdf, False, a.budget_rdf - len(r1), uses_r, rng)
    f2, why_f = round2(f1, D_fd, True, a.budget_fd - len(f1), uses_f, rng)
    C.log(f'round 2: RDF {len(r2)} pairs on [{why_r[0]:.4g}, {why_r[1]:.4g}) ({why_r[2]}); FD {len(f2)} pairs on '
          f'[{why_f[0]:.4g}, {why_f[1]:.4g}) ({why_f[2]})')
    res.update(run(r2 + f2))
    rng_t = np.random.default_rng(a.seed + 1)
    t1 = spread(D_tw, edges_f, a.budget_tw // 2, rng_t, uses_t, a.cap, taken)
    res.update(run(t1))
    t2, why_t = round2(t1, D_tw, True, a.budget_tw - len(t1), uses_t, rng_t)
    C.log(f'twin design: round 1 {len(t1)} pairs; round 2 {len(t2)} pairs on [{why_t[0]:.4g}, {why_t[1]:.4g}) '
          f'({why_t[2]})')
    res.update(run(t2))
    pairs = r1 + r2 + f1 + f2 + t1 + t2
    design = ['rdf'] * (len(r1) + len(r2)) + ['fd'] * (len(f1) + len(f2)) + ['tw'] * (len(t1) + len(t2))

    # ---- exact FD distances of every sampled pair
    ii = torch.tensor([p[0] for p in pairs])
    jj = torch.tensor([p[1] for p in pairs])
    dfd_x = metric.pairs(A, ii, A, jj)
    dtw_x = torch.minimum(dfd_x, torch.minimum(metric.pairs(A, ii, At, jj), metric.pairs(A, jj, At, ii)))
    ex = exact[ii, jj]
    lines += [f'Exact FD distances of the {len(pairs)} sampled pairs: {int((~ex).sum())} held only the lower bound in '
              f'the pool matrix (all >= {a.fd_cap} A); recomputed exactly, they lie at {float(dfd_x[~ex].min()) if (~ex).any() else float("nan"):.3f} '
              f'A or more. Where the matrix was exact the recomputation agrees to '
              f'{float((dfd_x[ex] - torch.as_tensor(D_fd)[ii[ex], jj[ex]]).abs().max()) if ex.any() else 0.0:.1e} A.', '']

    # ---- on-disk calibration pairs (NEHZOR): rerun on the repaired route
    old_rows = []
    if 'old_compack' in C.CAMPAIGNS[a.campaign]:
        oc = torch.load(os.path.join(C.CAMPAIGNS[a.campaign]['old_compack'], 'calibration.pt'), weights_only=False)
        win3 = torch.nonzero(st_all['energy'] <= ref + 3.0 * cfg.kT).flatten()
        st3 = C.distinct(C.subset(st_all, win3))
        kidx = {k: i for i, k in enumerate(st3['key'])}
        found = [(kidx.get(p['a']), kidx.get(p['b']), p) for p in oc['pairs']]
        found = [f for f in found if f[0] is not None and f[1] is not None]
        if found:
            ii3 = sorted({f[0] for f in found} | {f[1] for f in found})
            loc = {g: k for k, g in enumerate(ii3)}
            sub3 = C.subset(st3, ii3)
            for lo in range(0, len(ii3), 64):
                C.write_cifs(co, cfg, C.rebuild(co, cfg, sub3['params'][lo:lo + 64], sub3['hand'][lo:lo + 64]),
                             sub3['key'][lo:lo + 64], cif_dir)
            R3 = C.rdfs(co, cfg, sub3['params'], sub3['hand'])
            c3, a3 = C.rdf_prep(R3)
            del R3
            can3 = F.canonical(sg, sub3['params'], sub3['hand'])
            A3 = metric.prepare(can3)
            A3t = metric.prepare(F.twin_canonical(sg, can3, mi['sigma']))
            ia = torch.tensor([loc[f[0]] for f in found])
            ib = torch.tensor([loc[f[1]] for f in found])
            drdf = torch.stack([C.rdf_dist_prepped(c3[:, [i]], a3[[i]], c3[:, [j]], a3[[j]])[0, 0]
                                for i, j in zip(ia.tolist(), ib.tolist())]).double().numpy()
            dfd = metric.pairs(A3, ia, A3, ib).numpy()
            dtw = np.minimum(dfd, np.minimum(metric.pairs(A3, ia, A3t, ib).numpy(), metric.pairs(A3, ib, A3t, ia).numpy()))
            dstored = np.array([f[2]['d'] for f in found])
            ro = C.compack([(f[2]['a'], f[2]['b']) for f in found], cif_dir, cache, cache_path, tmpl, a.timeout)
            for k, f in enumerate(found):
                p = f[2]
                r = ro[C.pair_key(p['a'], p['b'])]
                old_rows.append(dict(d_rdf=float(drdf[k]), d_fd=float(dfd[k]), d_tw=float(dtw[k]),
                                     status=r['status'], n_matched=r['n_matched'], rmsd=r['rmsd'],
                                     seconds=r.get('seconds', np.nan), old_status=p['status'],
                                     old_n_matched=p['n_matched'], old_rmsd=p['rmsd'],
                                     E_a=p['E_a'], E_b=p['E_b'], key_a=p['a'], key_b=p['b']))
            lines += [f'On-disk NEHZOR calibration ({C.CAMPAIGNS[a.campaign]["old_compack"]}, read only): '
                      f'{len(oc["pairs"])} pairs, {len(found)} found in the shards by crystal_key, written to CIFs here and '
                      f'rerun on the repaired route; recomputed RDF distance minus the stored one: max |diff| '
                      f'{float(np.abs(drdf - dstored).max()):.2e} (a second RDF route check). Pooled below as design '
                      f'"old" (RDF-stratified by the campaign tool, 3 kT band).', '']
            del st3, sub3, c3, a3

    # ---- controls: other cell setting; own canonical form rebuilt
    ctl_rows = rng.choice(n, a.n_controls, replace=False)
    ctl_cr, ctl_keys, ctl_N = setting_controls(co, cfg, pool, ctl_rows, rng)
    C.write_cifs(co, cfg, ctl_cr, ctl_keys, cif_dir)
    from mxtaltools.crystal_fd.description import CrystalDescription
    cd = CrystalDescription.from_parts(sg, can['cell'][ctl_rows], can['t'][ctl_rows], can['R'][ctl_rows],
                                       can['h'][ctl_rows])
    L_, A_, t_, rv_, h_ = cd.crystal_params()
    cp = torch.cat([L_, A_, t_, rv_], 1).float()
    can_keys = [crystal_key(p, h.float()) + '_can' for p, h in zip(cp, h_)]
    C.write_cifs(co, cfg, C.rebuild(co, cfg, cp, h_.float()), can_keys, cif_dir)
    cifs(list(ctl_rows))
    ctl_pairs = [(pool['key'][i], k) for i, k in zip(ctl_rows, ctl_keys)] + \
                [(pool['key'][i], k) for i, k in zip(ctl_rows, can_keys)]
    rc = C.compack(ctl_pairs, cif_dir, cache, cache_path, tmpl, a.timeout)
    ctl = [rc[C.pair_key(*p)] for p in ctl_pairs]
    lines += [f'Controls (must match 20/20): a state against itself in another cell setting '
              f'{sum(c["n_matched"] >= 20 for c in ctl[:a.n_controls])}/{a.n_controls}; against its FD canonical form '
              f'rebuilt from crystal_params {sum(c["n_matched"] >= 20 for c in ctl[a.n_controls:])}/{a.n_controls}; '
              f'RMSD max {max(c["rmsd"] for c in ctl):.2e} A.', '']

    # ---- twin test
    tw_rows = rng.choice(n, a.n_twin, replace=False)
    twd = CrystalDescription.from_parts(sg, tw['cell'][tw_rows], tw['t'][tw_rows], tw['R'][tw_rows], tw['h'][tw_rows])
    L_, A_, t_, rv_, th_ = twd.crystal_params()
    tp, th_ = torch.cat([L_, A_, t_, rv_], 1).float(), th_.float()
    tw_keys = [crystal_key(p, h) + '_twin' for p, h in zip(tp, th_)]
    C.write_cifs(co, cfg, C.rebuild(co, cfg, tp, th_), tw_keys, cif_dir)
    # (-t, R C2_z) from the canonical form, as the earlier probes wrote the twin
    c2z = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=F64))
    zd = CrystalDescription.from_parts(sg, can['cell'][tw_rows], -can['t'][tw_rows], can['R'][tw_rows] @ c2z,
                                       can['h'][tw_rows])
    L_, A_, t_, rv_, h_ = zd.crystal_params()
    zp = torch.cat([L_, A_, t_ - torch.floor(t_), rv_], 1).float()
    z_keys = [crystal_key(p, h.float()) + '_c2z' for p, h in zip(zp, h_)]
    C.write_cifs(co, cfg, C.rebuild(co, cfg, zp, h_.float()), z_keys, cif_dir)
    cifs(list(tw_rows))
    tw_pairs = [(pool['key'][i], k) for i, k in zip(tw_rows, tw_keys)] + \
               [(pool['key'][i], k) for i, k in zip(tw_rows, z_keys)]
    rt = C.compack(tw_pairs, cif_dir, cache, cache_path, tmpl, a.timeout)
    Az = metric.prepare(F.canonical_desc(zd))
    d_tz = metric.pairs(At, torch.as_tensor(tw_rows), Az, torch.arange(a.n_twin))
    d_self_tw = metric.pairs(A, torch.as_tensor(tw_rows), At, torch.as_tensor(tw_rows))
    Rt = C.rdfs(co, cfg, torch.cat([pool['params'][tw_rows], tp]), torch.cat([pool['hand'][tw_rows], th_]))
    ct, at = C.rdf_prep(Rt)
    d_rdf_tw = torch.diagonal(C.rdf_dist_prepped(ct[:, :a.n_twin], at[:a.n_twin], ct[:, a.n_twin:], at[a.n_twin:]))
    lines += [f'## Molecular-mirror twin ({a.campaign}: r_g {mi["rg"]:.3f} A)', '',
              f'The mirror sigma through the heavy-atom plane maps the molecule onto itself to RMSD '
              f'{mi["mirror_rmsd_matched"]:.4f} A (largest atom deviation {mi["mirror_maxdev_matched"]:.4f} A) with like '
              f'atoms matched by optimal assignment ({mi["perm_moves"]} atoms swapped; the matching '
              f'{"preserves" if tmpl["perm_is_automorphism"] else "does NOT preserve"} the bond graph); in identity atom '
              f'order the RMSD is {mi["mirror_rmsd"]:.4f} A. Heavy-atom out-of-plane max {mi["out_of_plane_max"]:.4f} A.',
              '',
              f'*Each of {a.n_twin} random pool states against its twin built two ways: (t, R sigma) canonicalised, and '
              f'(-t, R C2_z) from the canonical form (the campaign-frame twin of the earlier probes). FD distance '
              f'between the two constructions: max {float(d_tz.max()):.3g} A. COMPACK 20/20 = same crystal.*', '',
              '| state | E (eLJ) | d_FD(state, twin) (A) | RDF dist (state, twin) | COMPACK (t, R sigma): matched / RMSD (A) '
              '| COMPACK (-t, R C2_z): matched / RMSD (A) |', '|---|---|---|---|---|---|']
    for k, i in enumerate(tw_rows):
        x, z = rt[C.pair_key(*tw_pairs[k])], rt[C.pair_key(*tw_pairs[a.n_twin + k])]
        lines.append(f'| {pool["key"][i][:8]} | {float(pool["energy"][i]):.2f} | {float(d_self_tw[k]):.3f} | '
                     f'{float(d_rdf_tw[k]):.4f} | {x["n_matched"]} ({x["status"]}) / {x["rmsd"]:.3f} | '
                     f'{z["n_matched"]} ({z["status"]}) / {z["rmsd"]:.3f} |')
    lines.append('')

    # ---- rows
    rows = []
    for k, ((i, j), dz) in enumerate(zip(pairs, design)):
        r = res[C.pair_key(pool['key'][i], pool['key'][j])]
        rows.append(dict(design=dz, i=int(i), j=int(j), key_a=pool['key'][i], key_b=pool['key'][j],
                         d_rdf=float(D_rdf[i, j]), d_fd=float(dfd_x[k]), d_tw=float(dtw_x[k]),
                         fd_exact_in_matrix=bool(ex[k]), E_a=float(pool['energy'][i]), E_b=float(pool['energy'][j]),
                         **{kk: r[kk] for kk in ('status', 'n_matched', 'rmsd', 'seconds')}))
    for r in old_rows:
        rows.append(dict(design='old', **r))
    st_ = np.array([r['status'] for r in rows])
    ok = st_ == 'ok'
    nm = np.array([r['n_matched'] for r in rows])
    y = (nm >= 20).astype(float)
    sec = np.array([r.get('seconds', np.nan) for r in rows], dtype=float)
    lines += ['## Pairs', '',
              f'{len(rows)} pairs: RDF design {design.count("rdf")} (round 2 on [{why_r[0]:.4g}, {why_r[1]:.4g}), '
              f'{why_r[2]}), FD design {design.count("fd")} (round 2 on [{why_f[0]:.4g}, {why_f[1]:.4g}) A, '
              f'{why_f[2]}), FD+twin design {design.count("tw")} (round 2 on [{why_t[0]:.4g}, {why_t[1]:.4g}) A, '
              f'{why_t[2]}), on-disk {len(old_rows)}; at most {a.cap} pairs per state per design. COMPACK status: '
              f'{dict((str(s), int((st_ == s).sum())) for s in set(st_))}; matched 20/20 {int(y[ok].sum())} of '
              f'{int(ok.sum())} usable. COMPACK wall time per pair: median {np.nanmedian(sec):.2f} s (one worker). Total '
              f'wall {(time.time() - t_start) / 60:.1f} min ({(time.time() - t_cmp0) / 60:.1f} min from the first COMPACK '
              f'call). {C.machine_note()}.', '']

    # ---- repair census and before/after
    reps = {}
    for pk, v in cache.items():
        ka, kb = pk.split('|')
        for key_, inf in ((ka, v.get('repair_a')), (kb, v.get('repair_b'))):
            if inf is not None:
                reps[key_] = inf
    nchg = sum(1 for v in reps.values() if v['changed'])
    comp_read = {}
    for v in reps.values():
        comp_read[v['components_as_read']] = comp_read.get(v['components_as_read'], 0) + 1
    lines += [f'Repair census over the {len(reps)} CIFs compared here: changed by the repair {nchg}; components as the '
              f'CSD reader returned them {dict(sorted(comp_read.items()))} (Z = {next(iter(reps.values()))["Z"] if reps else "-"}); '
              f'inter-molecule bonds removed {sum(v["inter_bonds_removed"] for v in reps.values())}, image atoms removed '
              f'{sum(v["image_atoms_removed"] for v in reps.values())}, atoms moved back to the written coordinates '
              f'{sum(v["atoms_moved"] for v in reps.values())}, intra-molecule bonds added '
              f'{sum(v["intra_bonds_added"] for v in reps.values())} / removed '
              f'{sum(v["intra_bonds_removed"] for v in reps.values())}. Pairs not "ok": '
              f'{[(r.get("key_a", "")[:6], r.get("key_b", "")[:6], r["status"]) for r in rows if r["status"] != "ok"][:10]}.',
              '']
    ba = []
    if old_rows:
        ba += before_after([(outcome(r['old_status'], r['old_n_matched']), outcome(r['status'], r['n_matched']))
                            for r in old_rows],
                           f'*The {len(old_rows)} on-disk calibration pairs: outcome stored by the old route (CIFs read as '
                           f'the CSD reader perceives them) against the repaired route, same states. Matched = 20/20.*')
    if a.previous:
        prev = torch.load(os.path.join(C.OUT_ROOT, a.previous, 'compack_cache.pt'), weights_only=False)
        shared = [(r, prev[C.pair_key(r['key_a'], r['key_b'])]) for r in rows if r['design'] != 'old'
                  and C.pair_key(r['key_a'], r['key_b']) in prev]
        ba += before_after([(outcome(p['status'], p['n_matched']), outcome(r['status'], r['n_matched']))
                            for r, p in shared],
                           f'*The {len(shared)} pairs this run shares with {a.previous} (the earlier build, old route): '
                           f'outcome there against the repaired route here, same CIF keys. Matched = 20/20.*')
        prev_ctl = [prev.get(C.pair_key(*p)) for p in ctl_pairs + tw_pairs]
        ba += [f'Controls and twins shared with {a.previous}: '
               f'{sum(1 for p in prev_ctl if p is not None)} of {len(prev_ctl)}; of them matched 20/20 there '
               f'{sum(1 for p in prev_ctl if p is not None and p["n_matched"] >= 20)}, here '
               f'{sum(1 for p, q in zip(prev_ctl, ctl + [rt[C.pair_key(*x)] for x in tw_pairs]) if p is not None and q["n_matched"] >= 20)}.',
               '']
    lines += ba

    # ---- per-distance analysis
    cuts = {}
    for key, name in NAMES:
        edges = np.linspace(0, a.span * cut0 * 2, 25) if key == 'd_rdf' else np.concatenate([[0.0], np.logspace(-2.5, 1.0, 15)])
        d = np.array([r[key] for r in rows])
        lines += [f'### {name}', '', f'*P(match) per {name} bin over all {int(ok.sum())} usable pairs (all designs'
                                     f'{" and the on-disk pairs" if old_rows else ""}).*', '']
        l2, o = analyse(name, d, y, ok, nm, a.nboot, a.seed, edges)
        lines += l2 + ['']
        cuts[key] = o
    lines += ['## By pair design', '']
    l3, own = design_tables(rows, ok, y, min(a.nboot, 300), a.seed)
    lines += l3
    for key, o in own.items():
        cuts[key]['rule_cut_own'] = o['rule_cut']
        cuts[key]['auc_own'] = o['auc']
        cuts[key]['own_designs'] = o['designs']
    # the matched pairs that FD calls far
    if 'rule_cut' in cuts['d_fd']:
        fc = cuts['d_fd']['rule_cut'][0]
        rc_ = cuts['d_rdf'].get('rule_cut', [np.nan])[0]
        d_fd = np.array([r['d_fd'] for r in rows])
        d_tw = np.array([r['d_tw'] for r in rows])
        d_r = np.array([r['d_rdf'] for r in rows])
        far = ok & (y == 1) & (d_fd > fc)
        lines += [f'Matched pairs (20/20) beyond the pooled FD RULE cut {fc:.4g} A: {int(far.sum())} of '
                  f'{int((ok & (y == 1)).sum())}; of them within the cut once the mirror twin is allowed: '
                  f'{int((far & (d_tw <= fc)).sum())}; within the pooled RDF RULE cut {rc_:.4g}: '
                  f'{int((far & (d_r <= rc_)).sum())}.', '']
        parts_idx = [k for k in np.nonzero(far)[0] if rows[k]['design'] != 'old']
        if parts_idx:
            ii_ = torch.tensor([rows[k]['i'] for k in parts_idx])
            jj_ = torch.tensor([rows[k]['j'] for k in parts_idx])
            _, P = metric.pairs(A, ii_, A, jj_, parts=True)
            lines += [f'Their FD terms at the minimising image (median over {len(parts_idx)} pairs): cell '
                      f'{float(P[:, 0].median()):.3f} A, centroid {float(P[:, 1].median()):.3f} A, rotation '
                      f'{float(P[:, 2].median()):.3f} A; with cell term > 0.5 A: {int((P[:, 0] > 0.5).sum())}; '
                      f'rotation > 1 A: {int((P[:, 2] > 1).sum())}.', '']

    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    torch.save(dict(rows=rows, cuts=cuts, controls=ctl, twin=[rt[C.pair_key(*p)] for p in tw_pairs],
                    pool_keys=pool['key'], rg=mi['rg'], t_rdf=t_rdf, t_can=t_can, runner=C.RUNNER,
                    mirror=dict(rmsd=mi['mirror_rmsd'], rmsd_matched=mi['mirror_rmsd_matched'],
                                maxdev_matched=mi['mirror_maxdev_matched'],
                                perm_is_automorphism=tmpl['perm_is_automorphism'])),
               os.path.join(out, 'calibration.pt'))
    with open(os.path.join(out, 'cuts.json'), 'w') as fh:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'best_f1'} for k, v in cuts.items()}, fh, indent=1,
                  default=float)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
