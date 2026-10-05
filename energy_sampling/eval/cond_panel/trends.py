"""
Trends across condition space, read off one atlas (eval/cond_panel/atlas.py): which molecules
the model handles well and what that follows, how its crystals vary with the molecule, how a
held-out molecule compares with the training molecule nearest to it, and how the run's own
level estimates vary between like and unlike molecules. No model is loaded.

QUALITY is a molecule's mean excess: the lattice energy of its draws above its own best
search minimum, in kT (MolecularCrystal.seed_energy_from against the energy reference). The
forward Jensen J_F is carried beside it. Both are means over the atlas's draws, so each
molecule's value carries sampling noise; `reliability` reports how much of the
molecule-to-molecule variance is real (split-half, Spearman-Brown), and every explained-
variance number is to be read against it.

Populations: "training" is the atlas's random training sample; "held-out" its held-out sample.
The neighbour molecules enter only through the pair analyses.

    cd energy_sampling
    python -m eval.cond_panel.trends --atlas <dir>/atlas_step50000.pt --out <dir>/figures
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval.cond_panel import figstyle as fs
from eval.cond_panel.latents import BLOCKS, NAMES, PERIOD, circ_mean_spread, energy_distance, fold

LINEAR_DIMS = [k for k in range(12) if PERIOD[k] == 0]
SHAPE_KEYS = ('asphericity', 'planarity_rms', 'extent1', 'extent2', 'extent3', 'frame_gap')
SIZE_KEYS = ('n_heavy', 'n_H', 'mol_volume', 'radius')
TOPOLOGY_KEYS = ('n_rot', 'n_rings', 'n_arom_rings', 'heavy_automorphisms', 'n_equiv_atoms')
POLAR_KEYS = ('n_NO', 'hbd', 'hba')
REFERENCE_KEYS = ('n_ref', 'ref_gap2_kT', 'ref_within_5kT', 'ref_spread_kT', 'ref_cp_best')


def load(path):
    A = torch.load(path, weights_only=False)
    T, D = float(A['temperature']), A['draws']
    split = np.array(A['split'])
    M = len(split)
    ex = (D['seed_energy'].astype(np.float64) - A['e_ref'][:, None]) / T
    lw = (D['log_r'] + D['log_pb'] - D['log_pf']).astype(np.float64)
    lat = fold(D['latent'].astype(np.float64))
    centre, spread = np.empty((M, 12)), np.empty((M, 12))
    for m in range(M):
        centre[m], spread[m] = circ_mean_spread(lat[m])
    A['u_star'] = 4.0 * float(A['data_ndim'])
    A['m'] = {
        'excess': ex.mean(1), 'excess_med': np.median(ex, 1), 'excess_p10': np.quantile(ex, 0.1, axis=1),
        'hot': (ex > A['u_star']).mean(1), 'jf': lw.mean(1), 'logr': D['log_r'].astype(np.float64).mean(1),
        'path': (D['log_pb'] - D['log_pf']).astype(np.float64).mean(1), 'logw_sd': lw.std(1),
        'reasonable': D['reasonable'].mean(1), 'cp': np.median(D['packing_coeff'], 1),
        'red': (D['reduction_en'] > 1e-6).mean(1), 'centre': centre, 'spread': spread,
        'fwd_level': A['tracker']['fwd_level_ema'], 'bwd_level': A['tracker']['bwd_level_ema'],
        'head': A['head_log_z'].astype(np.float64), 'depth': -A['e_ref'].astype(np.float64) / T,
    }
    A['ex_draws'], A['lat'] = ex, lat
    A['train'] = (split == 'train') & A['is_random_train']
    A['held'] = split == 'held_out'
    A['is_train_split'] = split == 'train'
    return A


def reliability(draws):
    """Share of the molecule-to-molecule variance of a per-molecule MEAN that is not sampling
    noise: the odd/even split-half correlation, stepped up to the full draw count."""
    a, b = draws[:, 0::2].mean(1), draws[:, 1::2].mean(1)
    r = np.corrcoef(a, b)[0, 1]
    return 2 * r / (1 + r)


def binned(x, y, edges):
    """Mean, standard error and count of y in each [edge_i, edge_i+1) bin of x."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi) & np.isfinite(y)
        n = int(sel.sum())
        rows.append((float(y[sel].mean()) if n else np.nan, float(y[sel].std() / np.sqrt(n)) if n > 1 else np.nan, n))
    return np.array(rows)


def cv_r2(X, y, seed=0):
    """Out-of-fold R^2 of a ridge fit (5 folds, penalty chosen inside each fold)."""
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=seed))
    return 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# --------------------------------------------------------------------------------------
def quality_by_feature(A, F):
    d, m = A['descriptors'], A['m']
    specs = (('n_heavy', 'heavy atoms', np.arange(2.5, 10.5, 1.0), None),
             ('n_rot', 'rotatable bonds', np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 20]), ('0', '1', '2', '3', '4+')),
             ('n_rings', 'rings', np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 20]), ('0', '1', '2', '3', '4+')),
             ('asphericity', 'asphericity (fifths of the training sample)',
              np.quantile(d['asphericity'][A['train']], np.linspace(0, 1, 6)) + np.array([0, 0, 0, 0, 0, 1e-9]), None))
    fig, axes = fs.panels(4, ncols=4, width=3.4, height=3.2, sharey=True)
    header, rows, points = ['feature', 'bin', 'training mean excess (kT)', 'SE', 'n', 'held-out mean excess (kT)', 'SE', 'n'], [], []
    for ax, (key, label, edges, ticks) in zip(axes, specs):
        fs.style(ax)
        centres = 0.5 * (edges[:-1] + edges[1:]) if ticks is None else np.arange(len(edges) - 1)
        out = {}
        for name, mask, colour in (('training', A['train'], fs.TRAIN), ('held-out', A['held'], fs.HELD)):
            b = binned(d[key][mask], m['excess'][mask], edges)
            keep = b[:, 2] >= 15
            ax.errorbar(centres[keep], b[keep, 0], yerr=b[keep, 1], color=colour, marker='o', capsize=2, label=name)
            out[name] = b
        if ticks is not None:
            ax.set_xticks(np.arange(len(ticks)), ticks)
        elif key == 'n_heavy':
            ax.set_xticks(centres[b[:, 2] >= 15])
        ax.set_xlabel(label)
        for i in range(len(edges) - 1):
            name_bin = ticks[i] if ticks is not None else f'{edges[i]:.3g} to {edges[i + 1]:.3g}'
            rows.append([key, name_bin] + [round(float(v), 3) for v in out['training'][i]] + [round(float(v), 3) for v in out['held-out'][i]])
        b = out['training']
        good = b[:, 2] >= 15
        points.append(f'{label}: training mean excess runs from {np.nanmin(b[good, 0]):.1f} to {np.nanmax(b[good, 0]):.1f} kT across bins')
    axes[0].set_ylabel('mean excess above best search minimum (kT)')
    axes[0].legend(loc='upper left')
    F.save(fig, 'quality_by_feature', 'Sample quality against molecule size, flexibility, rings and shape',
           f'Mean excess energy per molecule (kT above its own best search minimum), averaged over the molecules in each bin; '
           f'bars are one standard error over molecules; bins with fewer than 15 molecules are not drawn. '
           f'{int(A["train"].sum())} training and {int(A["held"].sum())} held-out molecules, {A["n_draws"]} draws each, step {A["step"]}.',
           (header, rows), points)


def group_effects(A, F, min_count=25):
    d, m = A['descriptors'], A['m']
    names = [g for g in A['group_names'] if g not in ('any ring', 'acyclic')]
    cols = [A['group_names'].index(g) for g in names]
    cov = np.column_stack([d['n_heavy'], d['n_rot'], d['n_rings'], d['asphericity']])
    header = ['group', 'training molecules with it', 'training effect (kT)', 'SE', 'held-out molecules with it', 'held-out effect (kT)', 'SE']
    res = {}
    for name, mask in (('training', A['train']), ('held-out', A['held'])):
        G = A['groups'][mask][:, cols].astype(float)
        keep = G.sum(0) >= min_count
        X = np.column_stack([np.ones(mask.sum()), (cov[mask] - cov[mask].mean(0)) / cov[mask].std(0), G[:, keep]])
        y = m['excess'][mask]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * resid.var(ddof=X.shape[1]))
        eff = np.full(len(names), np.nan)
        err = np.full(len(names), np.nan)
        eff[keep], err[keep] = beta[5:], se[5:]
        res[name] = (eff, err, G.sum(0))
    order = np.argsort(np.where(np.isfinite(res['training'][0]), res['training'][0], -np.inf))
    order = [i for i in order if np.isfinite(res['training'][0][i])]
    fig, ax = fs.plt.subplots(figsize=(7.2, 0.33 * len(order) + 1.6))
    fs.style(ax, grid='x')
    ypos = np.arange(len(order))
    for k, (name, colour) in enumerate((('training', fs.TRAIN), ('held-out', fs.HELD))):
        eff, err, _ = res[name]
        ax.errorbar(eff[order], ypos + (0.14 if k else -0.14), xerr=2 * err[order], fmt='o', color=colour, capsize=2,
                    label=name, linewidth=1.5)
    ax.axvline(0, color=fs.AXIS, linewidth=1)
    ax.set_yticks(ypos, [names[i] for i in order])
    ax.set_xlabel('change in mean excess with the group present (kT)')
    ax.legend(loc='lower right')
    rows = [[names[i], int(res['training'][2][i]), round(float(res['training'][0][i]), 2), round(float(res['training'][1][i]), 2),
             int(res['held-out'][2][i]), round(float(res['held-out'][0][i]), 2), round(float(res['held-out'][1][i]), 2)] for i in order[::-1]]
    eff, err, _ = res['training']
    worst, best = order[-1], order[0]
    points = [f'largest penalty in training: {names[worst]} at {eff[worst]:+.1f} kT (SE {err[worst]:.1f})',
              f'largest gain in training: {names[best]} at {eff[best]:+.1f} kT (SE {err[best]:.1f})']
    F.save(fig, 'group_effects', 'Functional groups and ring topology, at fixed size, flexibility and shape',
           f'Coefficient of each group flag in one least-squares fit of the molecule\'s mean excess (kT) on heavy-atom count, '
           f'rotatable bonds, ring count, asphericity and every flag at once, fitted separately on the training and held-out '
           f'samples; bars are two standard errors; groups carried by fewer than {min_count} molecules of a sample are left out. '
           f'A positive value means molecules with the group sit higher above their best search minimum.',
           (header, rows), points)


def variance_explained(A, F):
    d, m = A['descriptors'], A['m']
    ref = A['reference']
    col = lambda keys, src: np.column_stack([src[k] for k in keys])
    blocks = [
        ('size', col(SIZE_KEYS, d)),
        ('+ flexibility and rings', col(SIZE_KEYS + TOPOLOGY_KEYS, d)),
        ('+ shape', col(SIZE_KEYS + TOPOLOGY_KEYS + SHAPE_KEYS, d)),
        ('+ polar atoms and groups', np.column_stack([col(SIZE_KEYS + TOPOLOGY_KEYS + SHAPE_KEYS + POLAR_KEYS, d), A['groups'].astype(float)])),
        ('search minima only', np.column_stack([np.nan_to_num(col(REFERENCE_KEYS, ref), nan=0.0), m['depth']])),
        ('encoder embedding', A['enc_emb'].astype(np.float64)),
        ('conditioner output', A['cond_emb'].astype(np.float64)),
    ]
    targets = (('excess', 'mean excess (kT)'), ('jf', 'forward Jensen (nats)'))
    fig, axes = fs.panels(2, ncols=2, width=5.2, height=3.6)
    header, rows, points = ['target', 'predictors', 'training R2', 'held-out R2', 'training ceiling', 'held-out ceiling'], [], []
    draws = {'excess': A['ex_draws'], 'jf': (A['draws']['log_r'] + A['draws']['log_pb'] - A['draws']['log_pf']).astype(np.float64)}
    for ax, (key, label) in zip(axes, targets):
        fs.style(ax, grid='x')
        ceil = {n: reliability(draws[key][mask]) for n, mask in (('training', A['train']), ('held-out', A['held']))}
        vals = {n: [cv_r2(X[mask], m[key][mask]) for _, X in blocks] for n, mask in (('training', A['train']), ('held-out', A['held']))}
        ypos = np.arange(len(blocks))[::-1]
        ax.barh(ypos + 0.19, vals['training'], height=0.34, color=fs.TRAIN, label='training')
        ax.barh(ypos - 0.19, vals['held-out'], height=0.34, color=fs.HELD, label='held-out')
        for n, colour in (('training', fs.TRAIN), ('held-out', fs.HELD)):
            ax.axvline(ceil[n], color=colour, linewidth=1)
        ax.text(min(ceil.values()), len(blocks) - 0.45, 'noise ceiling ', ha='right', va='bottom', color=fs.INK2, fontsize=8)
        ax.set_yticks(ypos, [b[0] for b in blocks])
        ax.set_xlim(0, 1)
        ax.set_xlabel(f'cross-validated R2 of {label}')
        for i, (bname, _) in enumerate(blocks):
            rows.append([label, bname, round(vals['training'][i], 3), round(vals['held-out'][i], 3), round(ceil['training'], 3), round(ceil['held-out'], 3)])
        points.append(f'{label}: size alone explains {vals["training"][0]:.2f} of the training variance, all descriptors '
                      f'{vals["training"][3]:.2f}, the conditioner output {vals["training"][6]:.2f}; ceiling {ceil["training"]:.2f}')
    axes[0].legend(loc='lower right', bbox_to_anchor=(0.9, 0.0))
    F.save(fig, 'variance_explained', 'How much of the molecule-to-molecule variation each kind of information explains',
           'Out-of-fold R2 (5 folds) of a ridge regression predicting each molecule\'s value from the named predictors, fitted '
           'separately within the training sample and within the held-out sample. The first four rows are cumulative descriptor '
           'sets; "search minima only" uses the count, spacing and depth of the molecule\'s search minima. The vertical lines are '
           'the noise ceiling: the share of variance that is not sampling noise of the per-molecule mean.',
           (header, rows), points)


def neighbour_matched(A, F):
    m = A['m']
    sets = (('held-out molecule minus its nearest training molecule', A['pairs_held'], A['pair_dist_held'], fs.HELD),
            ('training molecule minus its nearest training molecule', A['pairs_train'], A['pair_dist_train'], fs.TRAIN))
    metrics = (('excess', 'mean excess (kT)'), ('jf', 'forward Jensen (nats)'), ('hot', 'share of draws above u*'))
    fig, axes = fs.panels(4, ncols=4, width=3.5, height=3.2)
    header = ['quantity', 'pairs', 'n pairs', 'mean difference', 'SE', 'median pair distance']
    rows, points = [], []
    for ax, (key, label) in zip(axes, metrics):
        fs.style(ax)
        for name, pairs, dist, colour in sets:
            diff = m[key][pairs[:, 0]] - m[key][pairs[:, 1]]
            lo, hi = np.quantile(diff, [0.005, 0.995])
            ax.hist(diff, bins=np.linspace(lo, hi, 41), histtype='step', color=colour, linewidth=1.6, density=True)
            ax.axvline(diff.mean(), color=colour, linewidth=1)
            rows.append([label, name, len(diff), round(float(diff.mean()), 3), round(float(diff.std() / np.sqrt(len(diff))), 3), round(float(np.median(dist)), 3)])
        ax.set_xlabel(f'difference in {label}')
        ax.set_yticks([])
        h = m[key][A['pairs_held'][:, 0]] - m[key][A['pairs_held'][:, 1]]
        t = m[key][A['pairs_train'][:, 0]] - m[key][A['pairs_train'][:, 1]]
        points.append(f'{label}: held-out minus its training neighbour {h.mean():+.2f} (SE {h.std() / np.sqrt(len(h)):.2f}); '
                      f'training minus its training neighbour {t.mean():+.2f} (SE {t.std() / np.sqrt(len(t)):.2f})')
    ax = axes[3]
    fs.style(ax)
    edges = np.quantile(np.concatenate([A['pair_dist_held'], A['pair_dist_train']]), np.linspace(0, 1, 7))
    edges[-1] += 1e-9
    for name, pairs, dist, colour in sets:
        diff = m['excess'][pairs[:, 0]] - m['excess'][pairs[:, 1]]
        b = binned(dist, diff, edges)
        ax.errorbar(0.5 * (edges[:-1] + edges[1:]), b[:, 0], yerr=b[:, 1], color=colour, marker='o', capsize=2,
                    label=name.split(' minus')[0])
        for i in range(len(edges) - 1):
            rows.append([f'mean excess difference, pair distance {edges[i]:.2f} to {edges[i + 1]:.2f}', name, int(b[i, 2]), round(b[i, 0], 3), round(b[i, 1], 3), ''])
    ax.axhline(0, color=fs.AXIS, linewidth=1)
    ax.set_xlabel('distance to the neighbour (conditioner space)')
    ax.set_ylabel('difference in mean excess (kT)')
    ax.legend(loc='upper left', fontsize=8)
    h = m['excess'][A['pairs_held'][:, 0]] - m['excess'][A['pairs_held'][:, 1]]
    near, far = A['pair_dist_held'] < edges[1], A['pair_dist_held'] >= edges[-2]
    points.append(f'held-out penalty in mean excess by distance to the nearest training molecule: {h[near].mean():+.1f} kT in the '
                  f'nearest sixth (below {edges[1]:.2f}), {h[far].mean():+.1f} kT in the farthest sixth (above {edges[-2]:.2f})')
    F.save(fig, 'neighbour_matched', 'A held-out molecule against the training molecule most like it',
           f'Per pair, the molecule\'s value minus its nearest training neighbour\'s (nearest in the checkpoint\'s conditioner output). '
           f'Orange: {len(A["pairs_held"])} held-out molecules. Blue, the control: {len(A["pairs_train"])} training molecules against '
           f'their own nearest training neighbour. Vertical lines are the means. Last panel: the mean excess difference against the '
           f'distance to the neighbour, in sixths of the pooled distances, one standard error.',
           (header, rows), points)


def condition_map(A, F):
    import umap
    m, d = A['m'], A['descriptors']
    emb = StandardScaler().fit_transform(A['cond_emb'].astype(np.float64))
    xy = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(emb)
    A['map_xy'] = xy
    fields = (('heavy atoms', d['n_heavy'], fs.SEQ), ('rotatable bonds', np.minimum(d['n_rot'], 4), fs.SEQ),
              ('asphericity', d['asphericity'], fs.SEQ), ('mean excess (kT)', m['excess'], fs.SEQ),
              ('median packing coefficient', m['cp'], fs.SEQ), ('depth of the best search minimum (kT)', m['depth'], fs.SEQ),
              ('head log Z (nats)', m['head'], fs.SEQ), ('forward level, tracker (nats)', m['fwd_level'], fs.SEQ),
              ('backward level, tracker (nats)', m['bwd_level'], fs.SEQ))
    fig, axes = fs.panels(len(fields), ncols=3, width=4.2, height=3.6)
    header, rows = ['field', 'Spearman rho with map axis 1', 'Spearman rho with map axis 2', 'kNN R2 in conditioner space'], []
    for ax, (label, v, cmap) in zip(axes, fields):
        ok = np.isfinite(v)
        if 'tracker' in label:
            ok &= A['is_train_split'] & (A['tracker']['count'] > 0)
        lo, hi = np.quantile(v[ok], [0.02, 0.98])
        sc = ax.scatter(xy[ok, 0], xy[ok, 1], c=np.clip(v[ok], lo, hi), cmap=cmap, s=3, linewidths=0, rasterized=True)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label)
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).outline.set_visible(False)
        rows.append([label, round(float(spearmanr(xy[ok, 0], v[ok])[0]), 3), round(float(spearmanr(xy[ok, 1], v[ok])[0]), 3),
                     round(float(knn_r2(A['cond_emb'][ok], v[ok])), 3)])
    points = [f'{r[0]}: {r[3]:.2f} of its variance is predicted by the 10 nearest molecules in conditioner space' for r in rows]
    F.save(fig, 'condition_map', 'Condition space, coloured by the molecule and by what the model does with it',
           f'UMAP (30 neighbours) of the conditioner output of the {len(xy)} atlas molecules, one point each; colour is clipped at '
           f'the 2nd and 98th percentiles. Tracker levels exist for training molecules only. The table gives, per field, how much of '
           f'its variance the 10 nearest molecules in the full conditioner space predict (leave-one-out).',
           (header, rows), points)


def knn_r2(X, y, k=10):
    """Leave-one-out R^2 of predicting y from the mean of its k nearest rows of X."""
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    pred = y[idx].mean(1)
    return 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def latent_feature_corr(A, F):
    d, m = A['descriptors'], A['m']
    feats = (('heavy atoms', d['n_heavy']), ('hydrogens', d['n_H']), ('rotatable bonds', d['n_rot']), ('rings', d['n_rings']),
             ('aromatic rings', d['n_arom_rings']), ('N and O atoms', d['n_NO']), ('asphericity', d['asphericity']),
             ('planarity rms', d['planarity_rms']), ('shortest extent', d['extent1']), ('middle extent', d['extent2']),
             ('longest extent', d['extent3']), ('frame gap', d['frame_gap']))
    lat_rows = [(f'{NAMES[k]} centre', m['centre'][:, k]) for k in LINEAR_DIMS] + \
               [(f'{NAMES[k]} spread', m['spread'][:, k]) for k in range(12)] + \
               [('packing coefficient', m['cp']), ('mean excess (kT)', m['excess'])]
    sel = A['train'] | A['held']
    R = np.array([[spearmanr(v[sel], f[sel])[0] for _, f in feats] for _, v in lat_rows])
    fig, ax = fs.plt.subplots(figsize=(8.6, 0.3 * len(lat_rows) + 2.2))
    im = ax.imshow(R, cmap=fs.DIV, vmin=-0.8, vmax=0.8, aspect='auto')
    ax.set_xticks(np.arange(len(feats)), [f[0] for f in feats], rotation=40, ha='right')
    ax.set_yticks(np.arange(len(lat_rows)), [r[0] for r in lat_rows])
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if abs(R[i, j]) >= 0.3:
                ax.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center', fontsize=7, color=fs.INK if abs(R[i, j]) < 0.55 else 'white')
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('Spearman rho')
    cb.outline.set_visible(False)
    header = ['latent summary'] + [f[0] for f in feats]
    rows = [[lat_rows[i][0]] + [round(float(x), 3) for x in R[i]] for i in range(len(lat_rows))]
    flat = [(abs(R[i, j]), lat_rows[i][0], feats[j][0], R[i, j]) for i in range(R.shape[0] - 2) for j in range(R.shape[1])]
    points = [f'{a} against {b}: rho {r:+.2f}' for _, a, b, r in sorted(flat, reverse=True)[:6]]
    F.save(fig, 'latent_feature_corr', 'What the model\'s crystals follow in the molecule',
           f'Spearman correlation across {int(sel.sum())} molecules (training and held-out samples pooled) between a summary of '
           f'each molecule\'s draws and a descriptor of the molecule. "Centre" is the mean latent (in units of the latent\'s range) '
           f'for the non-periodic latents; "spread" is the standard deviation of the draws, circular on periodic latents, v and w '
           f'folded onto one origin image. Values of magnitude 0.3 and above are printed.',
           (header, rows), points)


def model_vs_reference(A, F):
    m, ref = A['m'], A['reference']
    best = fold(ref['ref_best_latent'].astype(np.float64))
    panels = [(NAMES[k], m['centre'][:, k] * 2.0, best[:, k]) for k in range(6)]
    panels.insert(4, ('|alpha|', np.abs(A['lat'][:, :, 3]).mean(1), np.abs(best[:, 3])))
    panels.append(('packing coefficient', m['cp'], ref['ref_cp_best']))
    fig, axes = fs.panels(len(panels), ncols=4, width=3.3, height=3.3)
    header, rows, points = ['quantity', 'training Pearson r', 'held-out Pearson r', 'training mean (model - reference)', 'held-out mean (model - reference)'], [], []
    for ax, (label, model, target) in zip(axes, panels):
        fs.style(ax, grid='both')
        out = []
        for name, mask, colour in (('training', A['train'], fs.TRAIN), ('held-out', A['held'], fs.HELD)):
            ok = mask & np.isfinite(target) & np.isfinite(model)
            ax.scatter(target[ok], model[ok], s=3, color=colour, alpha=0.35, linewidths=0, rasterized=True, label=name)
            out += [float(np.corrcoef(target[ok], model[ok])[0, 1]), float((model[ok] - target[ok]).mean())]
        lo, hi = np.nanquantile(np.concatenate([target[A['train'] | A['held']], model[A['train'] | A['held']]]), [0.002, 0.998])
        ax.plot([lo, hi], [lo, hi], color=fs.AXIS, linewidth=1)
        ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
        ax.set_title(f'{label}   r = {out[0]:.2f} / {out[2]:.2f}')
        ax.set_xlabel('best search minimum')
        rows.append([label, round(out[0], 3), round(out[2], 3), round(out[1], 4), round(out[3], 4)])
    axes[0].set_ylabel('mean of the model\'s draws')
    leg = axes[0].legend(loc='upper left', markerscale=3)
    for h in leg.legend_handles:
        h.set_alpha(1)
    points = [f'{r[0]}: r = {r[1]:.2f} training, {r[2]:.2f} held-out' for r in rows]
    F.save(fig, 'model_vs_reference', 'Does the model put each molecule\'s cell where its best search minimum is?',
           f'One point per molecule: the mean of the model\'s draws against the value at the molecule\'s best search minimum, for '
           f'the six cell latents (latent units; |alpha| is the distance of alpha from 90 degrees, since the draws sit on both sides '
           f'of it for every molecule) and the packing coefficient (median of the draws). The title gives Pearson r for '
           f'the training and held-out samples ({int(A["train"].sum())} and {int(A["held"].sum())} molecules). The diagonal is equality.',
           (header, rows), points)


def latent_widths(A, F):
    m, ref = A['m'], A['reference']
    fig, ax = fs.plt.subplots(figsize=(8.4, 3.4))
    fs.style(ax)
    x = np.arange(12)
    header, rows = ['latent', 'training median spread', 'held-out median spread', 'spread of a uniform draw'], []
    for k, (name, mask, colour) in enumerate((('training', A['train'], fs.TRAIN), ('held-out', A['held'], fs.HELD))):
        med = np.median(m['spread'][mask], 0)
        q = np.quantile(m['spread'][mask], [0.25, 0.75], axis=0)
        ax.bar(x + (k - 0.5) * 0.36, med, width=0.34, color=colour, label=name)
        ax.vlines(x + (k - 0.5) * 0.36, q[0], q[1], color=fs.INK2, linewidth=1)
    uniform = 1.0 / np.sqrt(12.0)
    ax.axhline(uniform, color=fs.AXIS, linewidth=1)
    ax.text(-0.6, uniform, ' a uniform draw, linear latent', ha='left', va='bottom', fontsize=8, color=fs.INK2)
    ax.set_xticks(x, NAMES)
    ax.set_ylabel('spread of a molecule\'s draws\n(fraction of the latent\'s range)')
    tr, he = np.median(m['spread'][A['train']], 0), np.median(m['spread'][A['held']], 0)
    top = np.quantile(m['spread'][A['train'] | A['held']], 0.75, axis=0).max()
    ax.set_ylim(0, max(0.34, 1.08 * top))
    ax.legend(loc='upper left')
    for k in range(12):
        rows.append([NAMES[k], round(float(tr[k]), 4), round(float(he[k]), 4), round(uniform, 4) if PERIOD[k] == 0 else 'unbounded'])
    blocks = {b: float(np.mean([tr[k] for k in idx])) for b, idx in BLOCKS.items()}
    points = [f'{b}: median spread {v:.3f} of the range' for b, v in blocks.items()]
    F.save(fig, 'latent_widths', 'How wide each molecule\'s distribution is, latent by latent',
           'Median over molecules of the spread of that molecule\'s draws in each latent, as a fraction of the latent\'s range '
           '(standard deviation; circular on v, w, phi and r, with v and w folded onto one origin image); thin lines span the '
           'quartiles over molecules. The horizontal line is what a uniform draw over a linear latent gives.',
           (header, rows), points)


def levels(A, F):
    """What the run's own estimates say about Z(c): the forward level bounds log Z(c) from
    below; the backward level and the head are estimates, not bounds."""
    m = A['m']
    tr = A['is_train_split'] & np.isfinite(m['fwd_level']) & np.isfinite(m['bwd_level']) & (A['tracker']['count'] > 0)
    rng = np.random.default_rng(0)
    quantities = (('forward level (nats)', m['fwd_level'], tr), ('backward level (nats)', m['bwd_level'], tr),
                  ('backward minus forward (nats)', m['bwd_level'] - m['fwd_level'], tr),
                  ('head log Z (nats)', m['head'], np.ones(len(tr), bool)),
                  ('depth of best search minimum (kT)', m['depth'], np.ones(len(tr), bool)),
                  ('mean excess (kT)', m['excess'], np.ones(len(tr), bool)))
    pairs_nn = np.concatenate([A['pairs_train'], A['pairs_held']])
    dist_nn = np.concatenate([A['pair_dist_train'], A['pair_dist_held']])
    fig, axes = fs.panels(3, ncols=3, width=4.3, height=3.4)
    header, rows, points = ['quantity', 'SD over molecules', 'RMS difference, nearest-neighbour pairs', 'RMS difference, random pairs',
                            'neighbour / random', 'kNN R2 in conditioner space', 'n pairs'], [], []
    ax = axes[0]
    fs.style(ax)
    edges = np.quantile(dist_nn, np.linspace(0, 1, 7))
    edges[-1] += 1e-9
    for (label, v, ok), colour in zip(quantities[:4], (fs.TRAIN, fs.HELD, fs.THIRD, fs.INK2)):
        good = ok[pairs_nn[:, 0]] & ok[pairs_nn[:, 1]]
        sq = (v[pairs_nn[good, 0]] - v[pairs_nn[good, 1]]) ** 2
        idx = np.flatnonzero(ok)
        ra, rb = rng.choice(idx, 20000), rng.choice(idx, 20000)
        rand_rms = np.sqrt(((v[ra] - v[rb]) ** 2).mean())
        b = binned(dist_nn[good], sq, edges)
        ax.plot(0.5 * (edges[:-1] + edges[1:]), np.sqrt(b[:, 0]) / rand_rms, color=colour, marker='o', label=label.split(' (')[0])
    for label, v, ok in quantities:
        good = ok[pairs_nn[:, 0]] & ok[pairs_nn[:, 1]]
        nn_rms = np.sqrt(((v[pairs_nn[good, 0]] - v[pairs_nn[good, 1]]) ** 2).mean())
        idx = np.flatnonzero(ok)
        ra, rb = rng.choice(idx, 20000), rng.choice(idx, 20000)
        rand_rms = np.sqrt(((v[ra] - v[rb]) ** 2).mean())
        rows.append([label, round(float(v[ok].std()), 3), round(float(nn_rms), 3), round(float(rand_rms), 3), round(float(nn_rms / rand_rms), 3),
                     round(float(knn_r2(A['cond_emb'][ok], v[ok])), 3), int(good.sum())])
        points.append(f'{label}: neighbours differ by {nn_rms:.2f} RMS against {rand_rms:.2f} for random pairs (ratio {nn_rms / rand_rms:.2f})')
    ax.axhline(1.0, color=fs.AXIS, linewidth=1)
    ax.set_xlabel('distance between the two molecules (conditioner space)')
    ax.set_ylabel('RMS difference / RMS difference of random pairs')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right', fontsize=7)
    ax = axes[1]
    fs.style(ax, grid='both')
    ax.scatter(m['fwd_level'][tr], m['bwd_level'][tr], s=3, color=fs.TRAIN, alpha=0.3, linewidths=0, rasterized=True)
    ax.set_xlim(*np.quantile(m['fwd_level'][tr], [0.002, 0.998]))
    ax.set_ylim(*np.quantile(m['bwd_level'][tr], [0.002, 0.998]))
    ax.set_xlabel('forward level (nats)')
    ax.set_ylabel('backward level (nats)')
    ax.set_title(f'r = {np.corrcoef(m["fwd_level"][tr], m["bwd_level"][tr])[0, 1]:.2f}')
    ax = axes[2]
    fs.style(ax, grid='both')
    mid = 0.5 * (m['fwd_level'] + m['bwd_level'])
    ax.scatter(mid[tr], m['head'][tr], s=3, color=fs.TRAIN, alpha=0.3, linewidths=0, rasterized=True)
    lo, hi = np.quantile(np.concatenate([mid[tr], m['head'][tr]]), [0.002, 0.998])
    ax.plot([lo, hi], [lo, hi], color=fs.AXIS, linewidth=1)
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_xlabel('midpoint of forward and backward levels (nats)')
    ax.set_ylabel('head log Z (nats)')
    ax.set_title(f'r = {np.corrcoef(mid[tr], m["head"][tr])[0, 1]:.2f}')
    points.append(f'forward and backward levels across training molecules: r = {np.corrcoef(m["fwd_level"][tr], m["bwd_level"][tr])[0, 1]:.2f}; '
                  f'mean gap {np.mean(m["bwd_level"][tr] - m["fwd_level"][tr]):.1f} nats')
    F.save(fig, 'levels', 'The run\'s level estimates between like and unlike molecules',
           f'Left: RMS difference of each quantity between a molecule and its nearest training neighbour, in sixths of the '
           f'neighbour distance, divided by the RMS difference of random pairs (1 = unrelated). Middle and right: one point per '
           f'training molecule with tracker levels ({int(tr.sum())}; axes span the central 99.6%). The forward level is the tracker\'s per-molecule mean log '
           f'weight on forward rollouts, a lower bound on log Z(c); the backward level is the same mean on backward rows and is '
           f'not a bound.', (header, rows), points)


def pairs_shallow(A, F):
    """Distance between two molecules' latent distributions against their distance in
    condition space, from the atlas draws. The floor is the same distance between two
    halves of ONE molecule's draws."""
    lat = A['lat']
    rng = np.random.default_rng(1)
    D = lat.shape[1]
    half = D // 2

    def dist(i, j, own=False):
        a = lat[i][:half] if own else lat[i][rng.permutation(D)[:half]]
        b = lat[i][half:] if own else lat[j][rng.permutation(D)[:half]]
        return energy_distance(a, b)

    sets = [('held-out, nearest training molecule', A['pairs_held'], A['pair_dist_held'], fs.HELD),
            ('training, nearest training molecule', A['pairs_train'], A['pair_dist_train'], fs.TRAIN)]
    n_use = min(600, len(A['pairs_train']), len(A['pairs_held']))
    cond = A['cond_emb'].astype(np.float64)
    out = {}
    for name, pairs, d, _ in sets:
        pick = rng.choice(len(pairs), n_use, replace=False)
        out[name] = (d[pick], np.array([dist(i, j) for i, j in pairs[pick]]))
    pool = np.flatnonzero(A['train'] | A['held'])
    ra, rb = rng.choice(pool, n_use), rng.choice(pool, n_use)
    keep = ra != rb
    ra, rb = ra[keep], rb[keep]
    rand_d = np.linalg.norm(cond[ra] - cond[rb], axis=1)
    rand_e = np.array([dist(i, j) for i, j in zip(ra, rb)])
    floor = np.array([dist(i, i, own=True) for i in rng.choice(pool, n_use, replace=False)])
    fig, axes = fs.panels(2, ncols=2, width=5.0, height=3.6)
    ax = axes[0]
    fs.style(ax, grid='both')
    all_d = np.concatenate([out[s[0]][0] for s in sets] + [rand_d])
    all_e = np.concatenate([out[s[0]][1] for s in sets] + [rand_e])
    ax.scatter(rand_d, rand_e, s=5, color=fs.MUTED, alpha=0.45, linewidths=0, label='random pair')
    for name, _, _, colour in sets:
        ax.scatter(out[name][0], out[name][1], s=5, color=colour, alpha=0.45, linewidths=0, label=name)
    ax.axhline(np.median(floor), color=fs.INK, linewidth=1)
    ax.text(all_d.max(), np.median(floor), 'two halves of one molecule\'s draws ', ha='right', va='bottom', fontsize=8, color=fs.INK2)
    ax.set_xlabel('distance in conditioner space')
    ax.set_ylabel('energy distance between latent distributions')
    leg = ax.legend(loc='upper left', markerscale=2.5, fontsize=7)
    for h in leg.legend_handles:
        h.set_alpha(1)
    rho = spearmanr(all_d, all_e)[0]
    ax.set_title(f'Spearman rho = {rho:.2f}')
    ax = axes[1]
    fs.style(ax)
    edges = np.quantile(all_d, np.linspace(0, 1, 9))
    edges[-1] += 1e-9
    b = binned(all_d, all_e, edges)
    ax.errorbar(0.5 * (edges[:-1] + edges[1:]), b[:, 0], yerr=b[:, 1], color=fs.INK, marker='o', capsize=2)
    ax.axhline(np.median(floor), color=fs.AXIS, linewidth=1)
    ax.set_xlabel('distance in conditioner space (eighths of all pairs)')
    ax.set_ylabel('mean energy distance')
    ax.set_ylim(0, None)
    header = ['pair set', 'n pairs', 'median conditioner distance', 'median energy distance', 'mean energy distance']
    rows = [[name, len(out[name][0]), round(float(np.median(out[name][0])), 3), round(float(np.median(out[name][1])), 4), round(float(out[name][1].mean()), 4)]
            for name, *_ in sets]
    rows.append(['random pair', len(rand_d), round(float(np.median(rand_d)), 3), round(float(np.median(rand_e)), 4), round(float(rand_e.mean()), 4)])
    rows.append(['two halves of one molecule (floor)', len(floor), 0.0, round(float(np.median(floor)), 4), round(float(floor.mean()), 4)])
    points = [f'{r[0]}: median energy distance {r[3]:.3f} at median conditioner distance {r[2]:.2f}' for r in rows]
    points.append(f'rank correlation of distribution distance with conditioner distance over all pairs: {rho:.2f}')
    F.save(fig, 'pairs_shallow', 'Do molecules that are close in condition space get similar crystal distributions?',
           f'Energy distance between two molecules\' draws ({half} against {half}, folded latents, each coordinate in units of its '
           f'range) against their distance in the conditioner output. {n_use} nearest-neighbour pairs of each kind and as many random '
           f'pairs. The horizontal line is the floor: the same distance between two halves of one molecule\'s draws. Right: the mean '
           f'over all pairs in eighths of the distance, one standard error.',
           (header, rows), points)


ALL = (quality_by_feature, group_effects, variance_explained, neighbour_matched, latent_feature_corr, model_vs_reference,
       latent_widths, levels, pairs_shallow, condition_map)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--atlas', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', nargs='*', default=None, help='figure names to draw; default all')
    args = ap.parse_args()
    A = load(args.atlas)
    F = fs.Figures(args.out)
    print(f'atlas step {A["step"]}: {int(A["train"].sum())} training, {int(A["held"].sum())} held-out, '
          f'{len(A["split"]) - int(A["train"].sum()) - int(A["held"].sum())} neighbour-only molecules; {A["n_draws"]} draws each. '
          f'Reliability of the per-molecule mean excess: {reliability(A["ex_draws"][A["train"]]):.2f} training, '
          f'{reliability(A["ex_draws"][A["held"]]):.2f} held-out.', flush=True)
    for fn in ALL:
        if args.only is None or fn.__name__ in args.only:
            fn(A, F)


if __name__ == '__main__':
    main()
