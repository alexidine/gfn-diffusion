"""
The whole of a pre-noised prior file (data_processing/build_noised_prior.py) on the two bases of panel_dists.py:
the 12 latent marginals, and lattice energy against packing coefficient. No model is loaded.

Groups, overlaid: the training split's rows of the prior (blue), its held-out crystals of training molecules (aqua,
noised_split SPLIT_HELDOUT), and, with --heldout, the rows of a held-out-molecules file (orange). Latents are
latent_params() of the stored crystals, as panel_dists.py reads its reference rows. Energies are the stored mol_energy,
which a build scales by its thermal_scaling_factor (printed on the axis); packing is the stored packing_coeff.

    cd energy_sampling
    python -m eval.cond_panel.prior_dists --prior PRIOR.pt [--heldout HELDOUT_ANCHORS.pt] --out DIR [--name NAME]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import noised_prior as NP
from eval.cond_panel.figstyle import HELD, INK, INK2, SEQ, SURFACE, THIRD, TRAIN, Figures, plt, style

LATENT_NAMES = ('a', 'b', 'c', 'alpha', 'beta', 'gamma', 'u', 'v', 'w', 'theta', 'phi', 'r')
CHUNK = 50000


def rows_of(path):
    """(dict of numpy arrays per row: latent [n, 12], energy, packing, kind, split, identifier; the file's lj scale)."""
    f = torch.load(path, weights_only=False, map_location='cpu')
    b = f['prior']
    NP.refuse_stale_atom_pointers(b, path)
    lat = []
    for lo in range(0, b.num_graphs, CHUNK):
        c = b.subsample_new_batch(torch.arange(lo, min(lo + CHUNK, b.num_graphs)))
        lat.append(c.latent_params().detach().double().cpu())
    n = b.num_graphs
    tag = lambda k, d: NP.tag(b, k).numpy() if k in b.keys() else np.full(n, d)
    return dict(latent=torch.cat(lat).numpy(), energy=b.mol_energy.double().flatten().numpy(),
                packing=b.packing_coeff.double().flatten().numpy(), kind=tag('noised_kind', NP.KIND_ANCHOR),
                split=tag('noised_split', NP.SPLIT_TRAIN), identifier=np.asarray([str(x) for x in b.identifier])), \
        f.get('thermal_scaling_factor')


def pick(d, mask):
    return {k: v[mask] for k, v in d.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--prior', required=True)
    ap.add_argument('--heldout', default=None, help='a held-out-molecules file (K 0 anchors)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default='prior')
    a = ap.parse_args(argv)
    d, tsf = rows_of(a.prior)
    groups = [('training split', TRAIN, pick(d, d['split'] == NP.SPLIT_TRAIN)),
              ('held-out crystals of training molecules', THIRD, pick(d, d['split'] == NP.SPLIT_HELDOUT))]
    if a.heldout:
        h, tsf_h = rows_of(a.heldout)
        if tsf_h != tsf:
            raise SystemExit(f'{a.heldout}: thermal_scaling_factor {tsf_h} is not the prior\'s {tsf}')
        groups.append(('held-out molecules', HELD, h))
    groups = [(n, c, g) for n, c, g in groups if len(g['energy'])]
    counts = [(n, len(g['energy']), len(set(g['identifier'].tolist()))) for n, _, g in groups]
    scale = 'raw eLJ' if tsf is None else f'eLJ x {tsf:g}'
    figs = Figures(a.out)

    # energy vs packing: the training split as a log-density map, the other groups as point samples on top
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    style(ax, grid=None)
    tr = groups[0][2]
    ok = np.isfinite(tr['energy']) & np.isfinite(tr['packing'])
    e_lo, e_hi = np.quantile(tr['energy'][ok], [0.0005, 0.9995])
    hb = ax.hexbin(tr['packing'][ok], tr['energy'][ok], gridsize=90, bins='log', cmap=SEQ, mincnt=1,
                   extent=(0.5, 1.0, e_lo, e_hi), linewidths=0)
    fig.colorbar(hb, ax=ax, label=f'{groups[0][0]}: rows per cell (log)')
    rng = np.random.default_rng(0)
    for name, colour, g in groups[1:]:
        keep = rng.permutation(len(g['energy']))[:3000]
        ax.scatter(g['packing'][keep], g['energy'][keep], s=4, c=colour, alpha=0.5, lw=0,
                   label=f'{name} ({min(3000, len(g["energy"])):,} of {len(g["energy"]):,} shown)')
    for x in (0.55, 0.95):
        ax.axvline(x, color=INK2, lw=0.8, ls='--')
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(e_lo, e_hi)
    ax.set_xlabel('packing coefficient')
    ax.set_ylabel(f'lattice energy per asymmetric unit ({scale})')
    if len(groups) > 1:
        ax.legend(loc='upper left', fontsize=8, markerscale=3)
    caption = (f'Stored lattice energy ({scale}) against packing coefficient for every row of {os.path.basename(a.prior)}'
               + (f' and {os.path.basename(a.heldout)}' if a.heldout else '') + '. ' +
               '; '.join(f'{n}: {r:,} rows over {m:,} molecules' for n, r, m in counts) +
               '. Dashed lines: the packing window the trainer loads (0.55-0.95). Energies are absolute, so molecules '
               'of different size sit at different heights.')
    figs.save(fig, f'{a.name}_energy_vs_packing', 'Lattice energy against packing coefficient', caption,
              table=(['group', 'rows', 'molecules', 'energy P5', 'energy P50', 'energy P95', 'packing P5',
                      'packing P50', 'packing P95'],
                     [[n, r, m, *np.round(np.quantile(g['energy'], [0.05, 0.5, 0.95]), 3).tolist(),
                       *np.round(np.quantile(g['packing'], [0.05, 0.5, 0.95]), 4).tolist()]
                      for (n, r, m), (_, _, g) in zip(counts, groups)]),
              points=[f'{n}: {r:,} rows, {m:,} molecules, {r / max(m, 1):.1f} rows per molecule' for n, r, m in counts])

    # the 12 latent marginals, each group as a density over the [-1, 1] latent box
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.6))
    bins = np.linspace(-1, 1, 61)
    rows = []
    for k, name in enumerate(LATENT_NAMES):
        ax = axes.flat[k]
        style(ax, grid=None)
        for gname, colour, g in groups:
            x = g['latent'][:, k]
            x = x[np.isfinite(x)]
            ax.hist(x, bins=bins, density=True, histtype='step', color=colour, lw=1.4, label=gname)
            rows.append([name, gname, len(x), *np.round(np.quantile(x, [0.05, 0.5, 0.95]), 4).tolist(),
                         round(float((np.abs(x) > 0.99).mean()), 5)])
        ax.set_xlim(-1, 1)
        ax.set_yticks([])
        ax.set_title(name, fontsize=9, color=INK)
    axes.flat[0].legend(loc='upper left', fontsize=7)
    caption = ('Latent marginals (latent_params of the stored crystals, the [-1, 1] box) of every row, one density '
               'curve per group: ' + '; '.join(f'{n} ({r:,} rows)' for n, r, _ in counts) + '. The table gives each '
               'group\'s 5/50/95 percentiles and the fraction of rows within 0.01 of a wall.')
    figs.save(fig, f'{a.name}_latent_marginals', 'Latent marginals', caption,
              table=(['latent', 'group', 'rows', 'P5', 'P50', 'P95', 'frac |x| > 0.99'], rows))


if __name__ == '__main__':
    main()
