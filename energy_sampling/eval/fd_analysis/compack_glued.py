"""The glued FD distance against COMPACK, on the COMPACK-judged pairs of an earlier calibrate.py run (no new COMPACK).

calibrate.py (calibrate_<campaign>_v2) judged pairs of campaign endpoints with COMPACK (repaired CIFs, shell 20, CCDC
default tolerances) and scored them with the RDF distance and the unglued FD distance (FDMetric, with and without the
molecular-mirror twin). Its pairs were drawn on the decision boundaries of those two metrics, so absolute match rates
describe that design, not all pairs; every metric here is scored on the same pairs, so the comparison between metrics
is fair. This script rebuilds each pair's endpoints from the campaign shards (by crystal_key), adds the glued FD
distance (fd_distance.GluedFDMetric, margin 1 A; with the twin when the molecule's matched mirror RMSD is < 0.1 A, as
COMPACK matches a planar molecule's twin), and reports, per metric:

  - how well it predicts a COMPACK match (20/20): ROC AUC, the best-F1 cut, recall at 2% false matches, a logistic
    P(match) curve in log10 d;
  - how it tracks COMPACK's RMSD on matched pairs (the short-distance regime): Spearman and Pearson correlation;
  - pairs whose unglued and glued distances differ by more than 0.5 A (they cross a cell face), and their match rate.

    CUDA_VISIBLE_DEVICES=-1 python -m energy_sampling.eval.fd_analysis.compack_glued --campaign mipcas --out compack_glued_mipcas
"""
import argparse
import os

import numpy as np
import torch

from energy_sampling.eval.fd_analysis import common as C
from energy_sampling.eval.fd_analysis import fd_distance as F

F64 = torch.float64


def endpoints(campaign, keys):
    """params [n, 12], hand [n, 1] of the endpoints with these crystal_keys, in key order (calibrate.py's pool route)."""
    co, cfg, reg, ref = C.load_campaign(campaign)
    band = 3.0 if 'old_compack' in C.CAMPAIGNS[campaign] else 2.0
    st = C.shard_states(campaign, co, cfg, reg, ref, band)
    del reg
    want = set(keys)
    from energy_sampling.eval.campaign_compack import crystal_key
    at = {}
    for i, (p, h) in enumerate(zip(st['params'], st['hand'])):
        k = crystal_key(p, h)
        if k in want and k not in at:
            at[k] = i
    missing = [k for k in keys if k not in at]
    if missing:
        raise SystemExit(f'{len(missing)} keys not found in the shards (first {missing[:3]})')
    idx = torch.as_tensor([at[k] for k in keys])
    return cfg, st['params'][idx], st['hand'][idx]


def auc(d, y):
    """P(d of a matched pair < d of an unmatched pair), ties counted half."""
    pos, neg = d[y == 1], d[y == 0]
    if not len(pos) or not len(neg):
        return float('nan')
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float(1 - (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logistic(d, y):
    """(d50, d at P = 0.9, d at P = 0.1) of P(match) = sigmoid(a + b log10 d)."""
    from sklearn.linear_model import LogisticRegression
    x = np.log10(np.maximum(d, 1e-6))[:, None]
    m = LogisticRegression(C=1e6).fit(x, y)
    a, b = float(m.intercept_[0]), float(m.coef_[0, 0])
    at = lambda p: 10 ** ((np.log(p / (1 - p)) - a) / b)
    return at(0.5), at(0.9), at(0.1), (a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True, choices=list(C.CAMPAIGNS))
    ap.add_argument('--calibration', default=None, help='calibrate.py output folder (default calibrate_<campaign>_v2)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    C.setup()
    out = C.new_out_dir(a.out)
    cal_dir = os.path.join(C.OUT_ROOT, a.calibration or f'calibrate_{a.campaign}_v2')
    cal = torch.load(os.path.join(cal_dir, 'calibration.pt'), weights_only=False)
    rows = [r for r in cal['rows'] if r['status'] == 'ok']
    keys = list(dict.fromkeys([r['key_a'] for r in rows] + [r['key_b'] for r in rows]))
    cfg, params, hand = endpoints(a.campaign, keys)
    pos = {k: i for i, k in enumerate(keys)}
    ia = torch.as_tensor([pos[r['key_a']] for r in rows])
    ib = torch.as_tensor([pos[r['key_b']] for r in rows])

    mi = F.molecule_info(cfg.mol_path)
    can = F.canonical(cfg.sg, params, hand)
    gm = F.GluedFDMetric(cfg.sg, mi['rg'], margin=1.0, device='cpu')
    P = gm.prepare_glued(can)
    glued = gm.distance(P, ia, ib).numpy()
    twin = mi['mirror_rmsd_matched'] < 0.1
    if twin:
        Pt = gm.prepare_glued(F.twin_canonical(cfg.sg, can, mi['sigma']))
        g_tw = np.minimum.reduce([glued, gm.pairs(P[0], ia, Pt[1], ib).numpy(), gm.pairs(P[0], ib, Pt[1], ia).numpy()])
    # the stored unglued FD distance, recomputed here as a check on the rebuild
    pm = F.FDMetric(cfg.sg, mi['rg'])
    A = pm.prepare(can)
    plain = pm.pairs(A, ia, A, ib).numpy()
    stored = np.array([r['d_fd'] for r in rows])
    rebuild_dev = float(np.abs(plain - stored).max())

    y = np.array([int(r['n_matched'] == 20) for r in rows])
    nm = np.array([r['n_matched'] for r in rows])
    rmsd = np.array([r['rmsd'] for r in rows])
    design = np.array([r['design'] for r in rows])
    metrics = [('RDF distance', np.array([r['d_rdf'] for r in rows])), ('FD unglued (A)', stored),
               ('FD unglued + twin (A)', np.array([r['d_tw'] for r in rows])), ('FD glued (A)', glued)]
    if twin:
        metrics.append(('FD glued + twin (A)', g_tw))

    from scipy.stats import pearsonr, spearmanr
    lines = [f'# Glued FD distance against COMPACK: {a.campaign} (sg {cfg.sg}, Z\'=1)', '',
             f'Pairs: the {len(rows)} COMPACK-judged pairs of {cal_dir} (designs {dict(zip(*np.unique(design, return_counts=True)))}; '
             f'drawn on the RDF and unglued-FD decision boundaries, so match rates describe that design). COMPACK '
             f'route {cal["runner"]} (shell 20, CCDC default tolerances); match = 20/20: {int(y.sum())} of {len(y)}. '
             f'Endpoints rebuilt from the shards by crystal_key: unglued FD recomputed against the stored value, max '
             f'deviation {rebuild_dev:.1e} A. r_g {mi["rg"]:.3f} A; matched mirror RMSD {mi["mirror_rmsd_matched"]:.3f} A '
             f'(twin {"counted" if twin else "not counted"}; COMPACK matched every tested twin 20/20 in the calibration run).', '',
             f'*How each distance predicts a COMPACK match and tracks COMPACK\'s RMSD, on the same {len(y)} pairs. AUC: '
             f'probability that a matched pair is closer than an unmatched one (1 = perfect separation). Best-F1 cut and '
             f'its F1 for "same" = d <= cut. Recall at 2% FPR: share of matched pairs kept when at most 2% of unmatched '
             f'pairs are called the same. Logistic in log10 d: d at P(match) = 0.9 / 0.5 / 0.1. Spearman and Pearson: '
             f'against COMPACK RMSD over the {int(y.sum())} matched pairs.*', '',
             '| distance | AUC | best-F1 cut | F1 | recall at 2% FPR | d at P 0.9 / 0.5 / 0.1 | Spearman vs COMPACK RMSD (matched) | Pearson (matched) |',
             '|---|---|---|---|---|---|---|---|']
    fits = {}
    for name, d in metrics:
        cut, row = C.best_f1_cut(d, y)
        rec, _ = C.recall_at_fpr(d, y, 0.02)
        d50, d90, d10, ab = logistic(d, y)
        fits[name] = ab
        m = y == 1
        rs = spearmanr(d[m], rmsd[m]).correlation
        rp = pearsonr(d[m], rmsd[m])[0]
        lines.append(f'| {name} | {auc(d, y):.3f} | {cut:.4g} | {row[-1]:.3f} | {rec:.3f} | {d90:.3g} / {d50:.3g} / {d10:.3g} '
                     f'| {rs:.3f} | {rp:.3f} |')
    lines.append('')

    # face crossings
    base = np.array([r['d_tw'] for r in rows]) if twin else stored
    gl = g_tw if twin else glued
    cross = base - gl > 0.5
    lines += [f'*Pairs whose glued distance is more than 0.5 A below the unglued one (they meet across a cell face): '
              f'{int(cross.sum())} of {len(y)}; COMPACK matched {int(y[cross].sum())} of them. Their unglued / glued '
              f'distances (A), median: {np.median(base[cross]) if cross.any() else float("nan"):.3f} / '
              f'{np.median(gl[cross]) if cross.any() else float("nan"):.3f}.*', '']

    # matched-pair ratio: how the FD scale compares with COMPACK's RMSD
    m = (y == 1) & (rmsd > 0)
    q = np.quantile(gl[m] / rmsd[m], [0.1, 0.5, 0.9])
    lines += [f'*Scale: glued FD{" + twin" if twin else ""} over COMPACK RMSD on matched pairs, P10 / P50 / P90: '
              f'{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} (COMPACK RMSD is over the overlaid 20-molecule shells; the FD '
              f'distance is a one-molecule RMSD estimate).*', '']

    # P(match) and graded n_matched by glued-FD bin
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]
    lines += [f'*COMPACK outcome per glued FD{" + twin" if twin else ""} bin ({a.campaign}).*', '',
              '| glued FD bin (A) | pairs | matched 20/20 | P(match) | median molecules matched | median COMPACK RMSD, matched (A) |',
              '|---|---|---|---|---|---|']
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = (gl >= lo) & (gl < hi)
        if not mm.any():
            continue
        r_m = rmsd[mm & (y == 1)]
        lines.append(f'| [{lo}, {hi}) | {int(mm.sum())} | {int(y[mm].sum())} | {y[mm].mean():.2f} | {np.median(nm[mm]):.0f} '
                     f'| {np.median(r_m) if len(r_m) else float("nan"):.3f} |')
    lines.append('')

    plot(a.campaign, metrics, y, rmsd, fits, out)
    np.savez_compressed(os.path.join(out, 'pairs.npz'), keys_a=np.array([r['key_a'] for r in rows]),
                        keys_b=np.array([r['key_b'] for r in rows]), y=y, n_matched=nm, rmsd=rmsd, design=design,
                        **{n.split(' (')[0].replace(' ', '_').replace('+', 'plus'): d for n, d in metrics})
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    for l in lines:
        C.log(l)
    C.log(f'wrote {out}')


def plot(name, metrics, y, rmsd, fits, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sel = [m for m in metrics if m[0] in ('RDF distance', 'FD unglued + twin (A)', 'FD glued + twin (A)')] or metrics
    fig, axs = plt.subplots(2, len(sel), figsize=(4.8 * len(sel), 8.4), squeeze=False)
    for k, (mn, d) in enumerate(sel):
        ax = axs[0, k]
        x = np.log10(np.maximum(d, 1e-6))
        edges = np.quantile(x, np.linspace(0, 1, 13))
        cen, pm, nn = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x <= hi)
            if m.any():
                cen.append(10 ** x[m].mean()); pm.append(y[m].mean()); nn.append(m.sum())
        ax.plot(cen, pm, 'o', label='binned (12 equal-count bins)')
        a, b = fits[mn]
        g = np.logspace(x.min(), x.max(), 200)
        ax.plot(g, 1 / (1 + np.exp(-(a + b * np.log10(g)))), '-', label='logistic in log10 d')
        ax.set_xscale('log')
        ax.set_xlabel(mn)
        ax.set_ylabel('P(COMPACK 20/20)')
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f'{name}: P(match) vs {mn.split(" (")[0]}', fontsize=9)
        ax.legend(fontsize=7)
        ax = axs[1, k]
        ax.scatter(d[y == 1], rmsd[y == 1], s=8, alpha=0.6, label='matched 20/20')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(mn)
        ax.set_ylabel('COMPACK RMSD, matched pairs (A)')
        ax.set_title(f'{name}: COMPACK RMSD vs {mn.split(" (")[0]}', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'{name}_compack_curves.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
