"""The benchmark protocol's analysis (``docs/design/generator_benchmark_protocol.md`` step 3): coverage against draws
and against optimisation effort, and the topline numbers, from ``protocol_score`` distances and ``protocol_refine``
records.

Modes: reference minima within the identity tolerance of each other (single linkage) are one mode; its energy is its
lowest member's ``ref_energy`` (Day's DFT-corrected ranking) above the lowest minimum, its Boltzmann weight (300 K) the
sum of its members'. A draw *finds* a mode at depth k when the draw is legit, its depth-k structure is not severe
(energy per molecule above ``--severe``) and lies within the tolerance of one of the mode's minima. Coverage is the
share of modes within ``--window`` kJ/mol found (unweighted, the headline), and the Boltzmann-weighted share of all
modes beside it. Curves are medians (bands 10-90%) over ``--orders`` random orders of a source's draws, illegit draws
included (they find nothing). Effort at depth k with n draws is n x k energy-and-gradient calls; the effort envelope is
the best coverage over depths at each effort, raw coverage of all draws costing no energy calls.

usage:
    python -m eval.day_bench.protocol_report SCORE_NPZ OUT_DIR LABEL=DIR[+DIR...] [...] (--tolerance D | --calibration
        CSV --level 15/20) [--severe 0] [--window 7] [--orders 200] [--energy exp6]

Writes OUT_DIR/protocol_topline<tag>.csv, protocol_draws<tag>.png and protocol_effort<tag>.png (tag from the level).
"""
import argparse
import csv
import os

import numpy as np
from scipy.sparse.csgraph import connected_components

KT = 0.0083144626 * 300  # kJ/mol
STYLE = {'Clari-M': '#eb6834', 'MolCrystalFlow': '#008300', 'random': '#6b6b6b'}
INK, GRID = '#1f2328', '#c8ccd1'


def found_matrices(z, rows, folders, lab, tol, members, severe, depths):
    """{depth: bool [draws, modes]} for one source; draws are the refine records of its folders, in order."""
    idx = {(f, fl, int(k)): i for i, (f, fl, k) in enumerate(zip(z['folder'], z['fname'], z['depth'])) if
           z['label'][i] == lab}
    near = z['D'] <= tol
    out = {}
    for k in depths:
        F = np.zeros((len(rows), len(members)), dtype=bool)
        for d, (folder, r) in enumerate(zip(folders, rows)):
            i = idx.get((folder, r['file'], k))
            if i is None or r['legit'] != 'True' or r['rebuilt'] != 'True' or float(r[f'e_{k}']) > severe:
                continue
            F[d] = [near[i, m].any() for m in members]
        out[k] = F
    return out


def curves(F, score, rng, orders):
    """[orders, draws] coverage after each prefix of random orders of the draws; ``score`` maps found modes to a value."""
    out = np.empty((orders, len(F)))
    for o in range(orders):
        out[o] = np.maximum.accumulate(F[rng.permutation(len(F))], axis=0) @ score
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('score_npz')
    ap.add_argument('out_dir')
    ap.add_argument('sources', nargs='+')
    ap.add_argument('--tolerance', type=float)
    ap.add_argument('--calibration')
    ap.add_argument('--level', default='15/20')
    ap.add_argument('--severe', type=float, default=0.0)
    ap.add_argument('--window', type=float, default=7.0)
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

    e = z['ref_energy'] - z['ref_energy'].min()
    n_modes, lab_of = connected_components(z['D_ref'] <= tol, directed=False)
    members = [np.nonzero(lab_of == m)[0] for m in range(n_modes)]
    e_mode = np.array([e[m].min() for m in members])
    w_mode = np.array([np.exp(-e[m] / KT).sum() for m in members])
    in_win = e_mode <= a.window
    score_win = in_win / in_win.sum()
    score_mass = w_mode / w_mode.sum()
    print(f'identity tolerance {tol:.4f} ({basis}); {len(e)} reference minima -> {n_modes} modes, {in_win.sum()} within '
          f'{a.window:g} kJ/mol holding {score_mass[in_win].sum():.0%} of the Boltzmann weight (300 K)')

    rng = np.random.default_rng(0)
    res, top = {}, []
    for spec in a.sources:
        lab, dirs = spec.split('=', 1)
        rows, folders = [], []
        for d in dirs.split('+'):
            rr = list(csv.DictReader(open(os.path.join(d, f'refine_{a.energy}.csv'))))
            rows += rr
            folders += [os.path.basename(os.path.normpath(d))] * len(rr)
        F = found_matrices(z, rows, folders, lab, tol, members, a.severe, depths)
        res[lab] = {k: (curves(F[k], score_win, rng, a.orders), curves(F[k], score_mass, rng, a.orders))
                    for k in depths}
        n = len(rows)
        ok = [r for r in rows if r['legit'] == 'True' and r['rebuilt'] == 'True']
        grid = np.geomspace(min(depths[1:]), max(depths) * n, 300)
        env = envelope(res[lab], depths, grid)
        half = grid[np.argmax(env[0] >= 0.5)] if (env[0] >= 0.5).any() else np.inf
        top.append({'source': lab, 'draws': n,
                    'legit (%)': 100 * sum(r['legit'] == 'True' for r in rows) / n,
                    'rebuilt (% of legit)': 100 * len(ok) / max(1, sum(r['legit'] == 'True' for r in rows)),
                    'severe raw (% of rebuilt)': 100 * np.mean([float(r['e_0']) > a.severe for r in ok]),
                    f'severe at {max(depths)} (% of rebuilt)': 100 * np.mean([float(r[f'e_{max(depths)}']) > a.severe
                                                                               for r in ok]),
                    'raw coverage (% of modes in window)': 100 * np.median(res[lab][0][0][:, -1]),
                    f'reach at {max(depths)} steps (% of modes in window)': 100 * np.median(
                        res[lab][max(depths)][0][:, -1]),
                    'effort to half the modes (calls)': half,
                    'raw mass (% Boltzmann)': 100 * np.median(res[lab][0][1][:, -1]),
                    f'mass at {max(depths)} steps (% Boltzmann)': 100 * np.median(res[lab][max(depths)][1][:, -1])})
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f'protocol_topline{tag}.csv'), 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(top[0]))
        wr.writeheader()
        wr.writerows(top)
    print(f'topline, identity tolerance {tol:.4f} ({basis}), severe = {a.energy} energy above {a.severe:g} kJ/mol per '
          f'molecule; coverage = modes within {a.window:g} kJ/mol found, medians over {a.orders} orders of the draws:')
    for t in top:
        print('  ' + '; '.join(f'{k} {v:.0f}' if isinstance(v, float) and np.isfinite(v) and v > 100 else
                               f'{k} {v:.1f}' if isinstance(v, float) else f'{k} {v}' for k, v in t.items()))
    title = (f'identity: RDF within {tol:.3f} ({basis}); found = legit, not severe ({a.energy} above {a.severe:g} '
             f'kJ/mol per molecule), within tolerance;\n{in_win.sum()} modes within {a.window:g} kJ/mol of Day\'s '
             f'lowest (DFT-corrected); Boltzmann mass over all {n_modes} modes at 300 K')
    draw_draws(os.path.join(a.out_dir, f'protocol_draws{tag}.png'), res, depths, title, a.orders)
    draw_effort(os.path.join(a.out_dir, f'protocol_effort{tag}.png'), res, depths, title, a.orders)


def envelope(r, depths, grid):
    """Best median coverage over depths at each effort in ``grid`` (calls), [unweighted, weighted]; raw coverage of all
    draws costs no calls."""
    out = []
    for q in (0, 1):
        raw = np.median(r[0][q][:, -1])
        best = np.full(len(grid), raw)
        for k in depths[1:]:
            med = np.median(r[k][q], 0)
            n = np.minimum((grid // k).astype(int), len(med))
            ok = n >= 1
            best[ok] = np.maximum(best[ok], med[n[ok] - 1])
        out.append(best)
    return out


def draw_draws(path, res, depths, title, orders):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('Blues')
    shade = {k: cmap(0.3 + 0.7 * i / max(1, len(depths) - 1)) for i, k in enumerate(depths)}
    fig, axes = plt.subplots(2, len(res), figsize=(4.6 * len(res), 7.4), dpi=160, squeeze=False)
    for c, (lab, r) in enumerate(res.items()):
        for q, name in ((0, 'modes in window found (share)'), (1, 'Boltzmann mass found (share)')):
            ax = axes[q, c]
            for k in depths:
                y = r[k][q]
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
    n_max = max(r[0][0].shape[1] for r in res.values())
    grid = np.geomspace(min(depths[1:]), max(depths) * n_max, 300)
    for q, (ax, name) in enumerate(zip(axes, ('modes in window found (share)', 'Boltzmann mass found (share)'))):
        for lab, r in res.items():
            col = STYLE.get(lab, INK)
            for k in depths[1:]:
                med = np.median(r[k][q], 0)
                ax.plot(k * np.arange(1, len(med) + 1), med, color=col, lw=0.7, alpha=0.3)
            env = envelope(r, depths, grid)[q]
            ax.plot(grid, env, color=col, lw=2.0, label=f'{lab}: best depth at each effort (raw, all draws: '
                                                         f'{np.median(r[0][q][:, -1]):.2f})')
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.5)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel('effort: energy-and-gradient calls = draws x optimisation steps (log scale)', fontsize=8)
        ax.set_ylabel(name, fontsize=8.5)
        ax.legend(fontsize=7, loc='upper left')
    fig.suptitle(f'coverage against optimisation effort; thin lines: one depth each, thick: the best depth at each '
                 f'effort; medians over {orders} random orders of the draws\n{title}', fontsize=8, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    print(f'figure: {path}')


if __name__ == '__main__':
    main()
