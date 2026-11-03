"""The benchmark protocol's analysis (``docs/design/day_benchmark_scope.md``; protocol steps in
``generator_benchmark_protocol.md``): the coverage surface C(E, N) over optimisation effort and draws, coverage against
energy calls, the topline numbers and a per-source ledger, from ``protocol_score`` distances and ``protocol_refine``
records.

Modes: reference minima (Day's structures after ``protocol_refine``) merged by complete linkage at the identity
tolerance, so every mode's members lie within the tolerance of one another (``modes``). A mode's energies are its
members' lowest under each of Day's rankings, FIT+DMA and the Delta-ML correction, relative to the landscape's lowest;
it is in the window when both are at most ``--window`` kJ/mol (FIT+DMA alone for a mode the correction does not cover,
counted in the report). Weights for the Boltzmann-weighted coverage beside the headline are 300 K factors on the
corrected energy (FIT+DMA where none), summed over members.

A draw *finds* a mode at depth k when the draw is legit and rebuilt, its depth-k structure is not severe (its van der
Waals overlap, ``protocol_refine.overlap``, at least ``--clash``) and it lies within the tolerance of one of the mode's
members. A draw's cost to depth k is min(steps, k) energy-and-gradient calls (a row stops once converged); a draw that
was not rebuilt costs none and finds nothing.

Outputs, per source, medians over ``--orders`` random orders of its draws:
- the surface C(E, N): unweighted coverage of the window's modes after N draws, each optimised for E steps (E = the
  depths written by ``protocol_refine``; E = 0 is raw), as ``protocol_surface<tag>.csv`` and ``protocol_surface<tag>.png``;
- coverage against cumulative energy calls, one curve per depth and the best depth at each call count
  (``protocol_effort<tag>.png``), and against draws (``protocol_draws<tag>.png``);
- ``protocol_topline<tag>.csv``: raw coverage, reach (coverage at the deepest depth over all draws) over all modes and
  over the modes the source's cell sizes can represent (a mode with p molecules per primitive cell is representable
  when some draw's cell holds a multiple of p), calls to half coverage, clash rates, dropped modes (found at the deepest
  depth by another source, never by this one), and the effective number of modes among
  its hits (1 / sum of squared hit shares, each finding draw assigned to its nearest mode) beside the reference's
  Boltzmann effective number;
- ``protocol_ledger<tag>.csv``: draws, legit, rebuilt, and energy calls to each depth.

usage:
    python -m eval.day_bench.protocol_report SCORE_NPZ OUT_DIR LABEL=DIR[+DIR...] [...] (--tolerance D | --calibration
        CSV --level 15/20) [--clash 0.75] [--window 8] [--orders 200] [--energy exp6]
"""
import argparse
import csv
import os

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

KT = 0.0083144626 * 300  # kJ/mol
STYLE = {'Clari-M': '#eb6834', 'Clari-H': '#b5431b', 'MolCrystalFlow': '#008300', 'random': '#6b6b6b'}
INK, GRID = '#1f2328', '#c8ccd1'
N_GRID = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)


def modes(D_ref, tol):
    """Mode label per reference minimum: complete linkage on the minima's distances, cut at ``tol``."""
    if len(D_ref) == 1:
        return np.zeros(1, dtype=int)
    D = np.maximum(D_ref, D_ref.T)
    np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), method='complete'), t=tol, criterion='distance') - 1


def mode_table(z, lab_of, window):
    """Per mode: FIT+DMA and corrected energy (lowest member, kJ/mol above the landscape's lowest; NaN when no member
    is corrected), in-window flag, Boltzmann weight (300 K, corrected energy where present), molecules per primitive
    cell (smallest member's)."""
    n = lab_of.max() + 1
    fit, cor, zp = z['ref_fitdma_rel'], z['ref_corrected_rel'], z['ref_zprim']
    out = dict(fitdma=np.empty(n), corrected=np.empty(n), zprim=np.empty(n, dtype=int), weight=np.empty(n))
    for m in range(n):
        k = lab_of == m
        out['fitdma'][m] = fit[k].min()
        c = cor[k][np.isfinite(cor[k])]
        out['corrected'][m] = c.min() if len(c) else np.nan
        out['zprim'][m] = zp[k].min()
        e = np.where(np.isfinite(cor[k]), cor[k], fit[k])
        out['weight'][m] = np.exp(-e / KT).sum()
    out['in_window'] = (out['fitdma'] <= window) & (~np.isfinite(out['corrected']) | (out['corrected'] <= window))
    out['fitdma_only'] = out['in_window'] & ~np.isfinite(out['corrected'])
    return out


def representable(zprim, cell_sizes):
    """Bool per mode: some draw's cell holds a whole multiple of the mode's molecules per primitive cell."""
    return np.array([any(z % p == 0 for z in cell_sizes) for p in zprim], dtype=bool)


def draw_costs(rows, depths):
    """{depth: calls per draw}: min(steps, k) for a rebuilt draw, 0 otherwise."""
    st = np.array([int(r['steps']) if r['rebuilt'] == 'True' and r.get('steps', '') != '' else 0 for r in rows])
    return {k: np.minimum(st, k) for k in depths}


def found_matrices(z, rows, folders, lab, tol, lab_of, clash, depths):
    """{depth: bool [draws, modes]} and {depth: int [draws]} (the nearest found mode, -1 for none) for one source."""
    idx = {(f, fl, int(k)): i for i, (f, fl, k) in enumerate(zip(z['folder'], z['fname'], z['depth'])) if
           z['label'][i] == lab}
    n_modes = lab_of.max() + 1
    F, nearest = {}, {}
    for k in depths:
        Fk = np.zeros((len(rows), n_modes), dtype=bool)
        nk = np.full(len(rows), -1)
        for d, (folder, r) in enumerate(zip(folders, rows)):
            i = idx.get((folder, r['file'], k))
            if i is None or r['legit'] != 'True' or r['rebuilt'] != 'True':
                continue
            if f'ov_{k}' not in r:
                raise KeyError(f'{folder}/{r["file"]}: no ov_{k} column; rerun protocol_refine (overlap per depth)')
            if float(r[f'ov_{k}']) < clash:
                continue
            row = z['D'][i]
            near = row <= tol
            Fk[d, np.unique(lab_of[near])] = True
            if near.any():
                nk[d] = lab_of[np.argmin(row)]
        F[k], nearest[k] = Fk, nk
    return F, nearest


def orders_of(n, orders, rng):
    return [rng.permutation(n) for _ in range(orders)]


def coverage_runs(Fk, score, perms):
    """[orders, draws] coverage after each prefix of each order."""
    return np.stack([np.maximum.accumulate(Fk[p], axis=0) @ score for p in perms])


def at_calls(cov, cost, perms, grid):
    """[orders, len(grid)] coverage once cumulative calls reach each grid value (0 before the first draw)."""
    out = np.zeros((len(perms), len(grid)))
    for o, p in enumerate(perms):
        calls = np.cumsum(cost[p])
        i = np.searchsorted(calls, grid, side='right') - 1
        out[o] = np.where(i >= 0, cov[o][np.clip(i, 0, None)], 0.0)
    return out


def surface(runs, depths, n):
    """Median coverage [depths, N] at the draw counts of ``N_GRID`` up to ``n``."""
    ns = [m for m in N_GRID if m <= n] + ([n] if n not in N_GRID else [])
    return ns, np.array([[np.median(runs[k][:, m - 1]) for m in ns] for k in depths])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('score_npz')
    ap.add_argument('out_dir')
    ap.add_argument('sources', nargs='+')
    ap.add_argument('--tolerance', type=float)
    ap.add_argument('--calibration')
    ap.add_argument('--level', default='15/20')
    ap.add_argument('--clash', type=float, default=0.75,
                    help='severe below this smallest intermolecular heavy-atom distance / Bondi radii sum')
    ap.add_argument('--window', type=float, default=8.0)
    ap.add_argument('--orders', type=int, default=200)
    ap.add_argument('--energy', default='exp6')
    a = ap.parse_args()
    if a.tolerance is None:
        cal = {r['level']: r for r in csv.DictReader(open(a.calibration))}
        tol, tag = float(cal[a.level]['tolerance']), '_' + a.level.replace('/', 'of')
        basis = f'COMPACK {a.level} at P 0.9'
    else:
        tol, tag, basis = a.tolerance, f'_tol{a.tolerance:g}', 'set by hand'
    z = {k: v for k, v in np.load(a.score_npz).items()}
    for k in ('label', 'folder', 'fname'):
        z[k] = z[k].astype(str)
    depths = sorted(set(int(k) for k in z['depth']))
    deep = max(depths)

    lab_of = modes(z['D_ref'], tol)
    mt = mode_table(z, lab_of, a.window)
    win = mt['in_window']
    score_win = win / win.sum()
    score_mass = mt['weight'] / mt['weight'].sum()
    w_win = mt['weight'][win] / mt['weight'][win].sum()
    neff_ref = 1.0 / (w_win ** 2).sum()
    print(f'identity tolerance {tol:.4f} ({basis}); {len(lab_of)} reference minima -> {len(mt["weight"])} modes '
          f'(complete linkage); {win.sum()} within {a.window:g} kJ/mol under both Day rankings '
          f'({mt["fitdma_only"].sum()} by FIT+DMA alone, no correction), holding {score_mass[win].sum():.0%} of the '
          f'Boltzmann weight (300 K); reference effective number of modes in the window {neff_ref:.1f}', flush=True)

    rng = np.random.default_rng(0)
    res, found_deep, info = {}, {}, {}
    for spec in a.sources:
        lab, dirs = spec.split('=', 1)
        rows, folders = [], []
        for d in dirs.split('+'):
            rr = list(csv.DictReader(open(os.path.join(d, f'refine_{a.energy}.csv'))))
            rows += rr
            folders += [os.path.basename(os.path.normpath(d))] * len(rr)
        F, nearest = found_matrices(z, rows, folders, lab, tol, lab_of, a.clash, depths)
        cost = draw_costs(rows, depths)
        perms = orders_of(len(rows), a.orders, rng)
        runs = {k: (coverage_runs(F[k], score_win, perms), coverage_runs(F[k], score_mass, perms)) for k in depths}
        res[lab] = dict(runs=runs, cost=cost, perms=perms, n=len(rows))
        found_deep[lab] = F[deep].any(0)
        info[lab] = dict(rows=rows, nearest=nearest[deep], zs={int(r['z']) for r in rows if r.get('z', '') != ''})

    union = np.zeros_like(win)
    for f in found_deep.values():
        union |= f
    top, ledger = [], []
    for lab, r in res.items():
        rows, n = info[lab]['rows'], r['n']
        ok = [x for x in rows if x['legit'] == 'True' and x['rebuilt'] == 'True']
        reach_mask = win & representable(mt['zprim'], info[lab]['zs'])
        cov_deep = found_deep[lab]
        total = int(r['cost'][deep].sum())
        grid = np.unique(np.geomspace(1, max(1, total), 400).round())
        env = envelope(r, depths, grid, 0)
        half = grid[np.argmax(env >= 0.5)] if (env >= 0.5).any() else np.inf
        hits = info[lab]['nearest']
        hits = hits[(hits >= 0) & win[np.clip(hits, 0, None)]]
        share = np.bincount(hits, minlength=len(win))[win] / max(1, len(hits))
        top.append({
            'source': lab, 'draws': n,
            'legit (%)': 100 * sum(x['legit'] == 'True' for x in rows) / n,
            'rebuilt (% of legit)': 100 * len(ok) / max(1, sum(x['legit'] == 'True' for x in rows)),
            'clash raw (% of rebuilt)': 100 * np.mean([float(x['ov_0']) < a.clash for x in ok]) if ok else np.nan,
            f'clash at {deep} steps (% of rebuilt)': 100 * np.mean([float(x[f'ov_{deep}']) < a.clash for x in ok])
            if ok else np.nan,
            'raw coverage (% of window modes)': 100 * np.median(r['runs'][0][0][:, -1]),
            f'reach at {deep} steps (% of window modes)': 100 * np.median(r['runs'][deep][0][:, -1]),
            'reachable window modes (count)': int(reach_mask.sum()),
            f'reach at {deep} steps (% of reachable)': 100 * (cov_deep & reach_mask).sum() / max(1, reach_mask.sum()),
            'calls to half the window modes': half,
            f'calls to {deep} steps, all draws': total,
            f'dropped (window modes others found, this source never)': int((union & win & ~cov_deep).sum()),
            'effective modes among hits': 1.0 / (share ** 2).sum() if share.sum() > 0 else 0.0,
            'reference effective modes (Boltzmann)': neff_ref,
            'raw mass (% Boltzmann)': 100 * np.median(r['runs'][0][1][:, -1]),
            f'mass at {deep} steps (% Boltzmann)': 100 * np.median(r['runs'][deep][1][:, -1])})
        ledger.append({'source': lab, 'draws': n, 'legit': sum(x['legit'] == 'True' for x in rows),
                       'rebuilt': len(ok), **{f'calls to depth {k}': int(r['cost'][k].sum()) for k in depths},
                       'median FIRE steps (rebuilt)': float(np.median([int(x['steps']) for x in ok])) if ok else np.nan})
    os.makedirs(a.out_dir, exist_ok=True)
    for name, table in (('topline', top), ('ledger', ledger)):
        with open(os.path.join(a.out_dir, f'protocol_{name}{tag}.csv'), 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(table[0]))
            wr.writeheader()
            wr.writerows(table)
    with open(os.path.join(a.out_dir, f'protocol_surface{tag}.csv'), 'w', newline='') as fh:
        wr = csv.writer(fh)
        for lab, r in res.items():
            ns, S = surface({k: v[0] for k, v in r['runs'].items()}, depths, r['n'])
            wr.writerow([f'{lab}: rows E (optimisation steps per draw), columns N (draws)'] + ns)
            for k, row in zip(depths, S):
                wr.writerow([k] + [f'{v:.4f}' for v in row])
    print(f'topline: identity tolerance {tol:.4f} ({basis}); severe = van der Waals overlap below {a.clash:g}; '
          f'coverage = share of the {win.sum()} window modes found, medians over {a.orders} orders of the draws:')
    for t in top:
        print('  ' + '; '.join(f'{k} {v:.0f}' if isinstance(v, float) and np.isfinite(v) and v > 100 else
                               f'{k} {v:.1f}' if isinstance(v, float) else f'{k} {v}' for k, v in t.items()))
    title = (f'identity: RDF within {tol:.3f} ({basis}); found = legit, no van der Waals overlap below {a.clash:g}, '
             f'within tolerance;\n{win.sum()} modes within {a.window:g} kJ/mol of Day\'s lowest under FIT+DMA and the '
             f'Delta-ML correction; Boltzmann mass over all {len(win)} modes at 300 K')
    draw_surface(os.path.join(a.out_dir, f'protocol_surface{tag}.png'), res, depths, title, a.orders)
    draw_draws(os.path.join(a.out_dir, f'protocol_draws{tag}.png'), res, depths, title, a.orders)
    draw_effort(os.path.join(a.out_dir, f'protocol_effort{tag}.png'), res, depths, title, a.orders)


def envelope(r, depths, grid, q):
    """Best median coverage over depths once cumulative calls reach each ``grid`` value; raw coverage of all draws
    costs no calls."""
    best = np.full(len(grid), np.median(r['runs'][0][q][:, -1]))
    for k in depths[1:]:
        best = np.maximum(best, np.median(at_calls(r['runs'][k][q], r['cost'][k], r['perms'], grid), 0))
    return best


def draw_surface(path, res, depths, title, orders):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(res), figsize=(4.4 * len(res) + 0.8, 3.8), dpi=160, squeeze=False)
    for c, (lab, r) in enumerate(res.items()):
        ax = axes[0, c]
        ns, S = surface({k: v[0] for k, v in r['runs'].items()}, depths, r['n'])
        im = ax.imshow(S, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='Blues')
        for i in range(len(depths)):
            for j in range(len(ns)):
                ax.text(j, i, f'{S[i, j]:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if S[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(ns)), [str(m) for m in ns], fontsize=7)
        ax.set_yticks(range(len(depths)), [str(k) for k in depths], fontsize=7)
        ax.set_xlabel('N: draws', fontsize=8)
        if c == 0:
            ax.set_ylabel('E: optimisation steps per draw (0 = raw)', fontsize=8)
        ax.set_title(lab, fontsize=10, loc='left', color=STYLE.get(lab, INK))
    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.02, label='window modes found (share)')
    fig.suptitle(f'coverage surface C(E, N); medians over {orders} random orders of the draws\n{title}', fontsize=7.5,
                 x=0.01, y=1.1, ha='left')
    fig.savefig(path, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'figure: {path}')


def draw_draws(path, res, depths, title, orders):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('Blues')
    shade = {k: cmap(0.3 + 0.7 * i / max(1, len(depths) - 1)) for i, k in enumerate(depths)}
    fig, axes = plt.subplots(2, len(res), figsize=(4.6 * len(res), 7.4), dpi=160, squeeze=False)
    for c, (lab, r) in enumerate(res.items()):
        for q, name in ((0, 'window modes found (share)'), (1, 'Boltzmann mass found (share)')):
            ax = axes[q, c]
            for k in depths:
                y = r['runs'][k][q]
                x = np.arange(1, y.shape[1] + 1)
                ax.fill_between(x, np.percentile(y, 10, 0), np.percentile(y, 90, 0), color=shade[k], alpha=0.15, lw=0)
                ax.plot(x, np.median(y, 0), color=shade[k], lw=1.6, label='raw' if k == 0 else f'{k} steps')
            ax.set_xscale('log')
            ax.set_ylim(0, 1)
            ax.grid(color=GRID, lw=0.5)
            ax.tick_params(labelsize=7.5)
            ax.set_xlabel('draws (raw samples, log scale)', fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=8.5)
            if q == 0:
                ax.set_title(lab, fontsize=10, loc='left', color=STYLE.get(lab, INK))
            if q == 0 and c == 0:
                ax.legend(title='optimisation depth', fontsize=7, title_fontsize=7.5, loc='upper left')
    fig.suptitle(f'coverage against draws at each optimisation depth; medians over {orders} random orders of the draws, '
                 f'bands 10-90%\n{title}', fontsize=8, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    print(f'figure: {path}')


def draw_effort(path, res, depths, title, orders):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)
    total = max(int(r['cost'][max(depths)].sum()) for r in res.values())
    grid = np.unique(np.geomspace(1, max(1, total), 300).round())
    for q, (ax, name) in enumerate(zip(axes, ('window modes found (share)', 'Boltzmann mass found (share)'))):
        for lab, r in res.items():
            col = STYLE.get(lab, INK)
            for k in depths[1:]:
                ax.plot(grid, np.median(at_calls(r['runs'][k][q], r['cost'][k], r['perms'], grid), 0), color=col,
                        lw=0.7, alpha=0.3)
            ax.plot(grid, envelope(r, depths, grid, q), color=col, lw=2.0,
                    label=f'{lab}: best depth at each call count (raw, all draws: '
                          f'{np.median(r["runs"][0][q][:, -1]):.2f})')
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.5)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel('energy-and-gradient calls, cumulative over draws (each draw min(steps, depth); log scale)',
                      fontsize=8)
        ax.set_ylabel(name, fontsize=8.5)
        ax.legend(fontsize=7, loc='upper left')
    fig.suptitle(f'coverage against optimisation effort; thin lines: one depth each, thick: the best depth at each '
                 f'call count; medians over {orders} random orders of the draws\n{title}', fontsize=8, x=0.01,
                 ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    print(f'figure: {path}')


if __name__ == '__main__':
    main()
