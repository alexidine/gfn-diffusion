"""2-D UMAP of one molecule's RDF distances: raw and relaxed samples, relaxed experimental forms, key minima labelled.

Reads SAMPLE_DIR/rdf_embed_<energy>.npz (``rdf_embed``) and compack_<energy>.csv (``compack_endpoints``). Only the
structures drawn enter the embedding: raw samples, relaxed samples and the experimental forms relaxed with the same
energy (the deposited forms, brief copies and the P1 control are left out). UMAP runs on the precomputed distances
(n_neighbors 15, min_dist 0.3, seed 0); it keeps neighbourhoods, not distances, so read the picture for which
structures sit together, and the RDF distances themselves for how far apart.

usage:
    python -m eval.day_bench.rdf_umap SAMPLE_DIR [energy] [REFCODE]
"""
import csv
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#c3c2b7'
SOURCE = {  # categorical slots 1-2 of the reference palette, fixed order
    'relaxed': ('#2a78d6', 'relaxed sample'),
    'raw': ('#eb6834', 'raw sample'),
}
SHOWN = ('raw', 'relaxed', 'exp_relaxed')


def main(sample_dir, energy='exp6', refcode=None):
    import umap
    z = np.load(os.path.join(sample_dir, f'rdf_embed_{energy}.npz'))
    keep = np.isin(z['kind'].astype(str), SHOWN)
    D = z['dist'][np.ix_(keep, keep)]
    kind, name = z['kind'].astype(str)[keep], z['name'].astype(str)[keep]
    ends = {r['file']: r for r in csv.DictReader(open(os.path.join(sample_dir, f'compack_{energy}.csv')))}
    forms = list(dict.fromkeys(name[kind == 'exp_relaxed']))
    refcode = refcode or forms[0]
    xy = umap.UMAP(metric='precomputed', n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(D)
    idx = {(k, n): i for i, (k, n) in enumerate(zip(kind, name))}

    fig, ax = plt.subplots(figsize=(9.0, 6.4), dpi=200)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # samples that reached the reference form: raw -> relaxed, thin connectors under the points
    hit = [f for f, r in ends.items() if r.get(f'{refcode}_matched') == '30']
    for f in hit:
        path = [idx.get((k, f)) for k in ('raw', 'relaxed')]
        if None not in path:
            ax.plot(xy[path, 0], xy[path, 1], color=MUTED, lw=0.8, zorder=1, alpha=0.8)

    for k in ('raw', 'relaxed'):
        sel = kind == k
        c, lab = SOURCE[k]
        ax.scatter(xy[sel, 0], xy[sel, 1], s=18, color=c, edgecolors='white', linewidths=0.6, zorder=2,
                   label=f'{lab} ({sel.sum()})')

    # landmarks: the relaxed experimental forms, the lowest minimum found and the most-hit minimum
    marks = [(idx[('exp_relaxed', rc)], f'{rc} (relaxed)', 'star') for rc in forms]
    rel = [(float(r['energy_kjmol']), int(r['minimum']), f) for f, r in ends.items()]
    lowest = min(rel)
    sizes = np.bincount([m for _, m, _ in rel])
    top = int(np.argmax(sizes))
    top_e, _, top_f = min(x for x in rel if x[1] == top)
    marks.append((idx[('relaxed', lowest[2])], f'lowest found, {lowest[0]:.1f} kJ/mol', 'ring'))
    marks.append((idx[('relaxed', top_f)], f'most hit ({sizes[top]}x), {top_e:.1f} kJ/mol', 'ring'))
    for i, _, style in marks:
        if style == 'ring':
            ax.scatter(*xy[i], s=170, facecolors='none', edgecolors=INK, linewidths=1.2, zorder=4)
        else:
            ax.scatter(*xy[i], s=170, marker='*', facecolors=INK, edgecolors='white', linewidths=0.8, zorder=6)

    # labels in two columns beside the cloud, each at its landmark's height, pushed apart where they crowd
    centre = np.median(xy, axis=0)
    span = np.ptp(xy, axis=0).max()
    gap = 0.055 * span
    cols = {'left': [m for m in marks if xy[m[0], 0] < centre[0]],
            'right': [m for m in marks if xy[m[0], 0] >= centre[0]]}
    for side, group in cols.items():
        group = sorted(group, key=lambda m: -xy[m[0], 1])
        ys = []
        for m in group:
            y = xy[m[0], 1]
            if ys and ys[-1] - y < gap:
                y = ys[-1] - gap
            ys.append(y)
        x = xy[:, 0].min() - 0.06 * span if side == 'left' else xy[:, 0].max() + 0.06 * span
        for (i, lab, _), y in zip(group, ys):
            ax.annotate(lab, xy[i], xytext=(x, y), fontsize=8, color=INK, va='center', zorder=7,
                        ha='right' if side == 'left' else 'left',
                        arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7, shrinkA=1, shrinkB=5),
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_xlabel('UMAP 1', color=MUTED, fontsize=9)
    ax.set_ylabel('UMAP 2', color=MUTED, fontsize=9)
    ax.set_title(f'{refcode}: Clari-M samples before and after relaxation, UMAP of RDF distances', fontsize=10,
                 color=INK, loc='left')
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, lw=0.8))
    labels.append(f'raw to relaxed, the {len(hit)} samples that reached {refcode}')
    leg = ax.legend(handles, labels, loc='lower left', fontsize=8, frameon=True, framealpha=0.9, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK)
    lo = np.vstack([xy.min(0), xy.max(0)])
    ax.set_xlim(lo[0, 0] - 0.42 * span, lo[1, 0] + 0.42 * span)
    ax.set_ylim(lo[0, 1] - 0.12 * span, lo[1, 1] + 0.12 * span)
    fig.tight_layout()
    out = os.path.join(sample_dir, f'rdf_umap_{energy}.png')
    fig.savefig(out, facecolor='white')
    print(f'wrote {out}; {int(keep.sum())} structures embedded; {len(hit)} samples reached {refcode}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'exp6', sys.argv[3] if len(sys.argv) > 3 else None)
