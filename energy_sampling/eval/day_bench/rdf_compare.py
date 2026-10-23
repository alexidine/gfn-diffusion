"""Several sources of crystals for one molecule in one RDF space: shared minima, coverage of a reference, figure.

A source is a folder laid out as ``p1_relax`` (or ``random_search``) lays one out: starts in ``cifs/``, relaxed cells
in ``relaxed_<energy>/`` beside the relaxed experimental forms, energies in ``p1_relax_<energy>.csv``, and, when
``compack_endpoints`` has run, ``compack_<energy>.csv``. Every structure becomes a P1 crystal of the shared
reference conformer and gets the repo's RDF (``rdf_embed``: envwise, 10 Angstrom), computed on DEVICE. Relaxed
endpoints of all sources are grouped into minima by average linkage at the RDF identity cut CUT (calibrated against
COMPACK on this molecule by ``rdf_report``). A source marked ``:relaxed`` contributes its relaxed endpoints only.

Experimental matches are COMPACK's (all 30 of 30 molecules) where the source has ``compack_<energy>.csv`` and the RDF
cut otherwise; the table says which. With ``--ref LABEL``, the reference's minima within ``--window`` kJ/mol of the
lowest energy found by any source are the target set, and each other source's coverage is the fraction of them it
reaches (an endpoint in the same RDF minimum).

usage (CPU or GPU, licensed CSD Python API):
    python -m eval.day_bench.rdf_compare OUT_DIR REFCODE OTHER_FORM[,..] ENERGY CUT LABEL=DIR[:relaxed] ...
           [--ref LABEL] [--window KJMOL] [--device cuda]

Writes OUT_DIR/rdf_compare_<energy>.npz (cached: delete it after changing the sources), .md and .png.
"""
import argparse
import csv
import glob
import os

import matplotlib
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from eval.day_bench.p1_ingest import make_reference, split_molecules
from eval.day_bench.p1_relax import p1_crystal
from eval.day_bench.rdf_embed import load, rdfs
from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#c3c2b7'
COLOURS = ['#2a78d6', '#eb6834', '#1baf7a']  # categorical slots 1-3 for generators, fixed order
NEUTRAL = {'random': '#b9b7ae', 'Day': INK}  # reference landscapes drawn in ink and grey, not series colours


def parse_source(arg):
    label, rest = arg.split('=', 1)
    relaxed_only = rest.endswith(':relaxed')
    return label, rest[:-len(':relaxed')] if relaxed_only else rest, relaxed_only


def build(sources, refcode, forms, energy, device):
    from ccdc import io as ccdc_io
    from eval.day_bench.csd_roundtrip import p1_cell
    reader = ccdc_io.EntryReader('CSD')
    atoms_ref, _, _ = p1_cell(refcode, reader)
    ref = make_reference(split_molecules(atoms_ref)[0], smiles=reader.entry(refcode).molecule.heaviest_component.smiles)
    entries = []  # (source, stage, name, path)
    for label, d, relaxed_only in sources:
        if not relaxed_only:
            for p in sorted(glob.glob(os.path.join(d, 'cifs', '*.cif')) + glob.glob(os.path.join(d, 'cifs', '*.extxyz'))):
                entries.append((label, 'raw', os.path.splitext(os.path.basename(p))[0] + '.cif', p))
        for r in csv.DictReader(open(os.path.join(d, f'p1_relax_{energy}.csv'))):
            if r['built'] == 'True':
                entries.append((label, 'relaxed', r['file'], os.path.join(d, f'relaxed_{energy}', r['file'])))
    for rc in (refcode, *forms):
        entries.append(('experimental', 'relaxed', rc, os.path.join(sources[0][1], f'relaxed_{energy}', f'exp_{rc}.cif')))
    src, stage, name, crystals = [], [], [], []
    for s, st, n, p in entries:
        c, why = p1_crystal(load(p), ref, identifier=f'{s}:{st}:{n}')
        if c is None:
            print(f'not built: {s} {st} {n} ({why})')
            continue
        src.append(s), stage.append(st), name.append(n), crystals.append(c)
    print(f'{len(crystals)} structures built; computing RDFs on {device}', flush=True)
    rdf, bins = rdfs(crystals, device=device, chunk=32 if str(device).startswith('cuda') else 4)
    assert torch.isfinite(rdf).all(), 'non-finite RDF'
    return np.array(src), np.array(stage), np.array(name), compute_rdf_distmat(rdf, bins, show_tqdm=False).numpy()


def main(a):
    sources = [parse_source(s) for s in a.sources]
    labels = [s for s, _, _ in sources]
    torch.set_num_threads(2)
    path = os.path.join(a.out_dir, f'rdf_compare_{a.energy}.npz')
    if os.path.exists(path):
        z = np.load(path)
        src, stage, name, D = z['src'].astype(str), z['stage'].astype(str), z['name'].astype(str), z['dist']
    else:
        src, stage, name, D = build(sources, a.refcode, a.forms, a.energy, a.device)
        np.savez_compressed(path, src=src, stage=stage, name=name, dist=D)

    energy, compack, refE = {}, {}, {}
    for label, d, _ in sources:
        for r in csv.DictReader(open(os.path.join(d, f'p1_relax_{a.energy}.csv'))):
            if r['built'] == 'True':
                energy[(label, r['file'])] = float(r['e_end'])
                if r.get('ref_energy'):
                    refE[(label, r['file'])] = float(r['ref_energy'])  # the reference's own (better) ranking energy
        cp = os.path.join(d, f'compack_{a.energy}.csv')
        if os.path.exists(cp):
            for r in csv.DictReader(open(cp)):
                compack[(label, r['file'])] = r
    rel = np.nonzero((stage == 'relaxed') & (src != 'experimental'))[0]
    e = np.array([energy[(src[i], name[i])] for i in rel])
    groups = fcluster(linkage(squareform(D[np.ix_(rel, rel)], checks=False), method='average'), t=a.cut,
                      criterion='distance')
    exp_idx = {name[i]: i for i in np.nonzero(src == 'experimental')[0]}
    floor = e.min()
    gmin = {g: e[groups == g].min() for g in set(groups)}
    reach = {lab: set(groups[src[rel] == lab]) for lab in labels}

    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    out(f'## {a.refcode}: {", ".join(labels)} in one RDF space ({a.energy})')
    out()
    out(f'Relaxed endpoints grouped into minima by average linkage at RDF distance {a.cut} (identity cut calibrated '
        f'against COMPACK on this molecule). Energies per molecule relative to the lowest endpoint of any source '
        f'({floor:.1f} kJ/mol). Experimental matches: COMPACK 30/30 where a source has COMPACK identities (C), else '
        f'RDF distance to the relaxed form <= {a.cut} (R).')
    out()
    out('| source | relaxed endpoints (count) | minima reached (count) | reached by no other source | '
        'minima within 2 kJ/mol | within 5 kJ/mol | ' + ' | '.join(f'endpoints matching {rc}' for rc in exp_idx) + ' |')
    out('|---|---|---|---|---|---|' + '---|' * len(exp_idx))
    for lab, d, _ in sources:
        mine = rel[src[rel] == lab]
        own = reach[lab] - set().union(*(reach[o] for o in labels if o != lab))
        hits = []
        for rc, j in exp_idx.items():
            if all((lab, name[i]) in compack for i in mine):
                hits.append(f"{sum(compack[(lab, name[i])].get(f'{rc}_matched') == '30' for i in mine)} (C)")
            else:
                hits.append(f'{int((D[mine, j] <= a.cut).sum())} (R)')
        out(f'| {lab} | {len(mine)} | {len(reach[lab])} | {len(own)} | '
            f'{sum(gmin[g] - floor <= 2 for g in reach[lab])} | {sum(gmin[g] - floor <= 5 for g in reach[lab])} | '
            + ' | '.join(hits) + ' |')
    out()
    if a.ref:
        own = [(refE[(a.ref, name[i])], gi) for i, gi in zip(rel, groups) if (a.ref, name[i]) in refE]
        if own:  # rank the reference's minima by its own energy
            r0 = min(v for v, _ in own)
            gref = {}
            for v, gi in own:
                gref[gi] = min(v - r0, gref.get(gi, np.inf))
            target = {gi for gi, v in gref.items() if v <= a.window}
            out(f'(reference minima ranked by the ref_energy column of {a.ref})')
        else:
            gref = {gi: gmin[gi] - floor for gi in reach[a.ref]}
            target = {g for g in reach[a.ref] if gmin[g] - floor <= a.window}
        out(f'**Coverage of {a.ref}\'s minima within {a.window} kJ/mol of the floor** ({len(target)} minima): '
            + '; '.join(f'{lab} {len(reach[lab] & target)} ({len(reach[lab] & target) / max(len(target), 1):.0%})'
                        for lab in labels if lab != a.ref) + '.')
        out()
        missed = sorted((gref[g], g) for g in target - set().union(*(reach[o] for o in labels if o != a.ref)))
        out(f'{a.ref} minima in the window that no other source reached: {len(missed)}'
            + (f'; lowest at +{missed[0][0]:.1f} kJ/mol' if missed else '') + '.')
    with open(os.path.join(a.out_dir, f'rdf_compare_{a.energy}.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    figure(a, labels, src, stage, D, rel, e, exp_idx)
    a.refE = refE
    a.name = name
    if a.fig_window is not None:
        for by_energy in (False, True):
            figure_low(a, labels, src, D, rel, e, exp_idx, by_energy)
        if a.ref:
            figure_landscape(a, labels, src, D, rel, e, exp_idx)


def figure(a, labels, src, stage, D, rel, e, exp_idx):
    import umap
    xy = umap.UMAP(metric='precomputed', n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(D)
    fig, ax = plt.subplots(figsize=(9.0, 6.4), dpi=200)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    k = 0
    for lab in labels:
        if lab in NEUTRAL:
            sel = (src == lab) & (stage == 'relaxed')
            marker, size = ('+', 22) if lab == 'Day' else ('o', 8)
            ax.scatter(xy[sel, 0], xy[sel, 1], s=size, marker=marker, color=NEUTRAL[lab], linewidths=0.8,
                       zorder=1 if lab == 'random' else 4, label=f'{lab}, relaxed ({sel.sum()})')
            continue
        c = COLOURS[k]
        k += 1
        sel = (src == lab) & (stage == 'raw')
        if sel.any():
            ax.scatter(xy[sel, 0], xy[sel, 1], s=16, facecolors='none', edgecolors=c, linewidths=0.8, zorder=2,
                       label=f'{lab}, raw ({sel.sum()})')
        sel = (src == lab) & (stage == 'relaxed')
        ax.scatter(xy[sel, 0], xy[sel, 1], s=18, color=c, edgecolors='white', linewidths=0.6, zorder=3,
                   label=f'{lab}, relaxed ({sel.sum()})')
    marks = [(i, f'{n} (relaxed)', 'star') for n, i in exp_idx.items()]
    low = rel[np.argmin(e)]
    marks.append((low, f'lowest found ({src[low]}), {e.min():.1f} kJ/mol', 'ring'))
    for i, _, style in marks:
        if style == 'ring':
            ax.scatter(*xy[i], s=170, facecolors='none', edgecolors=INK, linewidths=1.2, zorder=5)
        else:
            ax.scatter(*xy[i], s=170, marker='*', facecolors=INK, edgecolors='white', linewidths=0.8, zorder=6)
    centre, span = np.median(xy, axis=0), np.ptp(xy, axis=0).max()
    gap = 0.055 * span
    for side in ('left', 'right'):
        group = sorted([m for m in marks if (xy[m[0], 0] < centre[0]) == (side == 'left')], key=lambda m: -xy[m[0], 1])
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
    ax.set_title(f'{a.refcode}: {", ".join(labels)}, 2-D UMAP of RDF distances ({a.energy})', fontsize=10, color=INK,
                 loc='left')
    leg = ax.legend(loc='lower left', fontsize=8, frameon=True, framealpha=0.9, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK)
    lo = np.vstack([xy.min(0), xy.max(0)])
    ax.set_xlim(lo[0, 0] - 0.42 * span, lo[1, 0] + 0.42 * span)
    ax.set_ylim(lo[0, 1] - 0.12 * span, lo[1, 1] + 0.12 * span)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, f'rdf_compare_{a.energy}.png'), facecolor='white')


MARKERS = ['o', '^', 'D']  # generator markers, fixed order, for the energy-coloured view


def figure_low(a, labels, src, D, rel, e, exp_idx, by_energy):
    """Relaxed endpoints within ``a.fig_window`` kJ/mol of the lowest energy found, plus the relaxed experimental forms,
    in a UMAP of their own distances. Colour = source (``by_energy`` False) or energy above the lowest (True, one-hue
    sequential, darker = lower; the source is then the marker)."""
    import umap
    from matplotlib.colors import LinearSegmentedColormap
    floor = e.min()
    keep = e - floor <= a.fig_window
    idx = np.concatenate([rel[keep], np.array(list(exp_idx.values()))])
    de = e[keep] - floor
    xy = umap.UMAP(metric='precomputed', n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(D[np.ix_(idx, idx)])
    n = int(keep.sum())
    s_keep = src[rel][keep]
    cmap = LinearSegmentedColormap.from_list('ink_blue', ['#0d366b', '#2a78d6', '#9ec5f4'])  # low energy = dark
    fig, ax = plt.subplots(figsize=(9.0, 6.4), dpi=200)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    k = 0
    sc = None
    for lab in labels:
        sel = np.nonzero(s_keep == lab)[0]
        if not len(sel):
            continue
        if lab in NEUTRAL:
            marker, size = ('+', 30) if lab == 'Day' else ('s', 12)
        else:
            marker, size = MARKERS[k], 26
        if by_energy:
            sc = ax.scatter(xy[sel, 0], xy[sel, 1], c=de[sel], cmap=cmap, vmin=0, vmax=a.fig_window, s=size,
                            marker=marker, edgecolors=INK if marker != '+' else None, linewidths=0.4,
                            zorder=1 if lab == 'random' else 3, label=f'{lab} ({len(sel)})')
        else:
            colour = NEUTRAL[lab] if lab in NEUTRAL else COLOURS[k]
            ax.scatter(xy[sel, 0], xy[sel, 1], s=size, marker=marker if lab in NEUTRAL else 'o', color=colour,
                       edgecolors='white' if marker != '+' else None, linewidths=0.6,
                       zorder=1 if lab == 'random' else 3, label=f'{lab} ({len(sel)})')
        if lab not in NEUTRAL:
            k += 1
    marks = [(n + j, f'{nm} (relaxed)', 'star') for j, nm in enumerate(exp_idx)]
    marks.append((int(np.argmin(de)), f'lowest found ({s_keep[np.argmin(de)]}), {floor:.1f} kJ/mol', 'ring'))
    for i, _, style in marks:
        if style == 'ring':
            ax.scatter(*xy[i], s=170, facecolors='none', edgecolors=INK, linewidths=1.2, zorder=5)
        else:
            ax.scatter(*xy[i], s=190, marker='*', facecolors='#eda100' if by_energy else INK, edgecolors=INK,
                       linewidths=0.8, zorder=6)
    centre, span = np.median(xy, axis=0), np.ptp(xy, axis=0).max()
    gap = 0.055 * span
    for side in ('left', 'right'):
        group = sorted([m for m in marks if (xy[m[0], 0] < centre[0]) == (side == 'left')], key=lambda m: -xy[m[0], 1])
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
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(f'{a.refcode}: relaxed endpoints within {a.fig_window:g} kJ/mol of the lowest found, 2-D UMAP of RDF '
                 f'distances ({a.energy})', fontsize=10, color=INK, loc='left')
    leg = ax.legend(loc='lower left', fontsize=8, frameon=True, framealpha=0.9, edgecolor=GRID)
    if by_energy:
        for h in leg.legend_handles:
            h.set_facecolor('white')
            h.set_edgecolor(INK)
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
        cb.set_label('energy above the lowest found (kJ/mol per molecule)', fontsize=8, color=INK)
        cb.ax.tick_params(labelsize=7)
    for t in leg.get_texts():
        t.set_color(INK)
    lo = np.vstack([xy.min(0), xy.max(0)])
    ax.set_xlim(lo[0, 0] - 0.42 * span, lo[1, 0] + 0.42 * span)
    ax.set_ylim(lo[0, 1] - 0.12 * span, lo[1, 1] + 0.12 * span)
    fig.tight_layout()
    tag = '_energy' if by_energy else ''
    fig.savefig(os.path.join(a.out_dir, f'rdf_compare_{a.energy}_low{a.fig_window:g}{tag}.png'), facecolor='white')


GEN_STYLE = [dict(marker='o', color='#eb6834'), dict(marker='^', color='#008300'), dict(marker='D', color='#e87ba4')]
# one colour and one symbol per generator, fixed order; relaxed = filled, unrelaxed = hollow


def figure_landscape(a, labels, src, D, rel, e, exp_idx):
    """The reference's relaxed endpoints as an energy terrain, the generators' relaxed endpoints on top.

    The 2-D map is a UMAP fitted to the reference endpoints' RDF distances alone; generator endpoints and the relaxed
    experimental forms are placed by ``transform`` from their distances to the reference. The terrain at a map point
    is the lower envelope over reference endpoints i of E_i + (window / r0^2) |x - x_i|^2 (one well per endpoint, as
    deep as its energy, r0 = 6% of the map's span), blank beyond 10% of the span from every reference endpoint. Only
    endpoints within ``a.fig_window`` kJ/mol of the lowest found are drawn."""
    import umap
    from matplotlib.colors import LinearSegmentedColormap
    floor = e.min()
    is_ref = src[rel] == a.ref
    own = np.array([a.refE.get((a.ref, a.name[i]), np.nan) for i in rel])
    use_own = bool(np.isfinite(own[is_ref]).all())
    re_ = own - np.nanmin(own[is_ref]) if use_own else e - floor
    ref_i = rel[is_ref & (re_ <= a.fig_window)]
    ref_e = re_[is_ref & (re_ <= a.fig_window)]
    terrain_label = (f"{a.ref} terrain: {a.ref}'s own energy above its lowest (kJ/mol per molecule)" if use_own else
                     f'{a.ref} terrain: energy above the lowest found (kJ/mol per molecule)')
    gen_labels = [lab for lab in labels if lab != a.ref and lab not in NEUTRAL]
    reducer = umap.UMAP(metric='precomputed', n_neighbors=15, min_dist=0.3, random_state=0)
    xy_ref = reducer.fit_transform(D[np.ix_(ref_i, ref_i)])
    span = np.ptp(xy_ref, axis=0).max()
    gx, gy = np.meshgrid(np.linspace(xy_ref[:, 0].min() - 0.1 * span, xy_ref[:, 0].max() + 0.1 * span, 400),
                         np.linspace(xy_ref[:, 1].min() - 0.1 * span, xy_ref[:, 1].max() + 0.1 * span, 400))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    d2 = ((grid[:, None, :] - xy_ref[None, :, :]) ** 2).sum(-1)
    r0 = 0.06 * span
    terrain = (ref_e[None, :] + a.fig_window / r0 ** 2 * d2).min(1)
    terrain[np.sqrt(d2.min(1)) > 0.10 * span] = np.nan
    terrain = np.minimum(terrain, a.fig_window).reshape(gx.shape)
    cmap = LinearSegmentedColormap.from_list('terrain_blue', ['#0d366b', '#2a78d6', '#cde2fb'])

    fig, ax = plt.subplots(figsize=(9.0, 6.6), dpi=200)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    levels = np.arange(0, a.fig_window + 1e-6, 1.0)
    cf = ax.contourf(gx, gy, terrain, levels=levels, cmap=cmap, zorder=0)
    ax.contour(gx, gy, terrain, levels=levels, colors='white', linewidths=0.3, zorder=1)
    cb = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label(terrain_label, fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7)

    def place(idx):
        return reducer.transform(D[np.ix_(idx, ref_i)])
    for k, lab in enumerate(gen_labels):
        sel = rel[(src[rel] == lab) & (e - floor <= a.fig_window)]
        if not len(sel):
            continue
        p = place(sel)
        # every endpoint is drawn; one with no reference endpoint within the identity cut is placed by transform beside
        # its nearest reference structures, so its position is approximate (the legend counts how many are in basins)
        near = D[np.ix_(sel, ref_i)].min(1) <= a.cut
        st = GEN_STYLE[k]
        ax.scatter(p[:, 0], p[:, 1], zorder=3, s=36, marker=st['marker'], facecolors=st['color'], edgecolors=INK,
                   linewidths=0.6, label=f'{lab}, relaxed, within {a.fig_window:g} kJ/mol under {a.energy} ({len(sel)}; '
                                      f'{near.sum()} in mapped basins)')
    xs = place(np.array(list(exp_idx.values())))
    ax.scatter(xs[:, 0], xs[:, 1], s=220, marker='*', facecolors='white', edgecolors=INK, linewidths=1.0, zorder=5)
    for (nm, _), q in zip(exp_idx.items(), xs):
        ax.annotate(f'{nm} (relaxed)', q, xytext=(q[0] + 0.05 * span, q[1] + 0.05 * span), fontsize=8, color=INK,
                    zorder=6, arrowprops=dict(arrowstyle='-', color=INK, lw=0.6),
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(f'{a.refcode}: generator endpoints on the {a.ref} terrain (UMAP of RDF distances, {a.energy})',
                 fontsize=10, color=INK, loc='left')
    leg = ax.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.95, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, f'rdf_landscape_{a.energy}_{a.ref}_w{a.fig_window:g}.png'), facecolor='white')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('refcode')
    ap.add_argument('forms', type=lambda s: tuple(filter(None, s.split(','))))
    ap.add_argument('energy')
    ap.add_argument('cut', type=float)
    ap.add_argument('sources', nargs='+')
    ap.add_argument('--ref', default=None)
    ap.add_argument('--window', type=float, default=7.0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--fig_window', type=float, default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
