"""The reference landscape as a map, one panel per source, marking which reference minima the source matched.

Reuses ``coverage``'s cached distances and identity, so the map and the coverage numbers agree: a reference minimum is
matched by a source when one of the source's structures is within the RDF cut of it or COMPACK-matched to it
(``coverage.settled_identity``).

The map: UMAP (precomputed RDF distances, 15 neighbours, min_dist 0.3, seed 0) fitted on the reference minima within
``--window`` kJ/mol of the lowest (``ref_energy``: Day's DFT-corrected energy) and ``--copies`` rigidly jittered copies
of each (``rdf_terrain.noised``: each molecule shifted ~N(0, SHIFT) Angstrom per axis and turned ~N(0, TURN) degrees,
no relaxation), which give each minimum a well of its own size. The terrain is the lower envelope over those points of
E_i + (window / r0^2)|x - x_i|^2, r0 = 6% of the map's span, blank beyond 10%. A source's structures within the window
of the lowest structure under the benchmark energy are placed at the inverse-distance-weighted mean of the positions of
their 5 nearest map points (minima and copies, so a structure lands on the well it is closest to); filled when the
structure is matched to a drawn minimum, hollow otherwise (it is then not any drawn minimum, wherever it lands).

usage (licensed CSD Python API; GPU for the copies' RDFs):
    python -m eval.day_bench.coverage_map COVERAGE_DIR REFCODE REF_LABEL REF_DIR [--window 7] [--cut 0.093]
        [--copies 10] [--shift 0.05] [--turn 1.25] [--forms NAME=FILE,...] [--energy exp6] [--device cuda]

Writes COVERAGE_DIR/coverage_map_<energy>.npz (the map's distances, cached) and coverage_map_<energy>.png.
"""
import argparse
import csv
import os

import numpy as np
import torch

from eval.day_bench.coverage import KT, STYLE, settled_identity

INK, GRID = '#1f2328', '#c8ccd1'
MARKER = {'Clari-M': 'o', 'MolCrystalFlow': '^', 'random': 'D'}


def map_distances(paths, energies, copies, shift, turn, refcode, device, extra=()):
    """RDF distances among the given minima, their jittered copies and the ``extra`` structures (in that order); the
    energy and parent minimum of each map point."""
    from ccdc import io as ccdc_io
    from eval.day_bench.csd_roundtrip import p1_cell
    from eval.day_bench.p1_ingest import make_reference, split_molecules
    from eval.day_bench.p1_relax import p1_crystal
    from eval.day_bench.rdf_embed import load, rdfs
    from eval.day_bench.rdf_terrain import noised
    from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat
    reader = ccdc_io.EntryReader('CSD')
    atoms_ref, _, _ = p1_cell(refcode, reader)
    ref = make_reference(split_molecules(atoms_ref)[0], smiles=reader.entry(refcode).molecule.heaviest_component.smiles)
    rng = np.random.default_rng(0)
    crystals, energy, parent = [], [], []
    for k, (p, e) in enumerate(zip(paths, energies)):
        atoms = load(p)
        for c in [atoms] + [noised(atoms, rng, shift, turn) for _ in range(copies)]:
            cr, why = p1_crystal(c, ref)
            assert cr is not None, f'{p}: {why}'
            crystals.append(cr)
            energy.append(e)
            parent.append(k)
    for p in extra:
        cr, why = p1_crystal(load(p), ref)
        assert cr is not None, f'{p}: {why}'
        crystals.append(cr)
    rdf, bins = rdfs(crystals, device=device, chunk=32)
    return compute_rdf_distmat(rdf, bins, show_tqdm=False).numpy(), np.array(energy), np.array(parent)


def source_energies(paths, energy):
    """Each structure's energy after convergence (``converge_<energy>.csv`` ``e_after`` beside its folder)."""
    tables, out = {}, []
    for p in paths:
        folder = os.path.dirname(os.path.dirname(p))
        if folder not in tables:
            tables[folder] = {r['file']: float(r['e_after']) for r in
                              csv.DictReader(open(os.path.join(folder, f'converge_{energy}.csv'))) if r.get('e_after')}
        out.append(tables[folder][os.path.basename(p)])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cov_dir')
    ap.add_argument('refcode')
    ap.add_argument('ref_label')
    ap.add_argument('ref_dir')
    ap.add_argument('--window', type=float, default=7.0)
    ap.add_argument('--cut', type=float, default=0.093)
    ap.add_argument('--copies', type=int, default=10)
    ap.add_argument('--shift', type=float, default=0.05)
    ap.add_argument('--turn', type=float, default=1.25)
    ap.add_argument('--forms', default='')
    ap.add_argument('--energy', default='exp6')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    import umap
    from ase.spacegroup import Spacegroup

    z = np.load(os.path.join(a.cov_dir, f'coverage_{a.energy}.npz'))
    D, names, kind, paths = z['D'], z['names'].astype(str), z['kind'].astype(str), z['paths'].astype(str)
    label = np.array([n.split(':', 1)[0] for n in names])
    S, _ = settled_identity(D, names, kind, a.ref_label, a.cut,
                            os.path.join(a.cov_dir, f'compack_band_{a.energy}.csv'))
    if S is None:
        S = D <= a.cut
    rows = {r['file']: r for r in csv.DictReader(open(os.path.join(a.ref_dir, f'p1_relax_{a.energy}.csv')))}
    ref = np.nonzero((label == a.ref_label) & (kind == 'sample'))[0]
    files = [names[i].split('/', 1)[1] for i in ref]
    e_ref = np.array([float(rows[f]['ref_energy']) for f in files])
    e_ref -= e_ref.min()
    w = np.exp(-e_ref / KT)
    win = e_ref <= a.window
    wref = ref[win]

    src_idx = np.nonzero((label != a.ref_label) & (kind == 'sample'))[0]
    e6 = source_energies(paths[src_idx], a.energy)
    floor = e6.min()
    drawn = src_idx[e6 - floor <= a.window]  # the source structures the map shows
    cache = os.path.join(a.cov_dir, f'coverage_map_{a.energy}.npz')
    if os.path.exists(cache) and list(np.load(cache)['minima']) == list(names[wref]) \
            and 'drawn' in np.load(cache).files and list(np.load(cache)['drawn']) == list(names[drawn]):
        c = np.load(cache)
        D_all, e_fit, parent = c['D'], c['energy'], c['parent']
    else:
        if str(a.device).startswith('cuda'):
            torch.cuda.set_per_process_memory_fraction(0.4)
        D_all, e_fit, parent = map_distances(paths[wref], e_ref[win], a.copies, a.shift, a.turn, a.refcode, a.device,
                                             extra=paths[drawn])
        np.savez_compressed(cache, D=D_all, energy=e_fit, parent=parent, minima=names[wref], drawn=names[drawn])
    n_fit = len(e_fit)
    D_fit, D_place = D_all[:n_fit, :n_fit], D_all[n_fit:, :n_fit]
    place_row = {i: k for k, i in enumerate(drawn)}
    xy = umap.UMAP(metric='precomputed', n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(D_fit)
    own = np.array([np.nonzero(parent == k)[0][0] for k in range(len(wref))])  # each minimum's own point
    xy_min = xy[own]

    sources = [lab for lab in STYLE if (label == lab).any()]
    span = np.ptp(xy, axis=0).max()
    lo, hi = xy.min(0) - 0.1 * span, xy.max(0) + 0.1 * span
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 360), np.linspace(lo[1], hi[1], 360))
    d2 = ((np.stack([gx.ravel(), gy.ravel()], 1)[:, None, :] - xy[None]) ** 2).sum(-1)
    r0 = 0.06 * span
    terrain = (np.minimum(e_fit, a.window)[None, :] + a.window / r0 ** 2 * d2).min(1)
    terrain[np.sqrt(d2.min(1)) > 0.10 * span] = np.nan
    terrain = np.minimum(terrain, a.window).reshape(gx.shape)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('terrain_blue', ['#0d366b', '#2a78d6', '#cde2fb'])
    fig, axes = plt.subplots(1, len(sources), figsize=(6.2 * len(sources), 6.4), dpi=160)
    forms = dict(f.split('=', 1) for f in a.forms.split(',') if f)
    deep = np.nonzero(e_ref[win] <= 4.5)[0]
    for ax, lab in zip(np.atleast_1d(axes), sources):
        cf = ax.contourf(gx, gy, terrain, levels=np.arange(0, a.window + 1e-6, 1.0), cmap=cmap, zorder=0)
        ax.contour(gx, gy, terrain, levels=np.arange(0, a.window + 1e-6, 1.0), colors='white', linewidths=0.3,
                   zorder=1)
        s = np.nonzero(label == lab)[0]
        s = s[kind[s] == 'sample']
        e_s = e6[np.searchsorted(src_idx, s)] - floor
        s, e_s = s[e_s <= a.window], e_s[e_s <= a.window]
        dd = D_place[[place_row[i] for i in s]]
        nn = np.argsort(dd, 1)[:, :5]
        wt = 1.0 / np.maximum(np.take_along_axis(dd, nn, 1), 1e-6)
        pos = (xy[nn] * wt[..., None]).sum(1) / wt.sum(1, keepdims=True)
        on = S[np.ix_(s, wref)].any(1)
        col, mk = STYLE[lab], MARKER.get(lab, 'o')
        ax.scatter(pos[~on, 0], pos[~on, 1], s=16, marker=mk, facecolors='none', edgecolors=col, linewidths=0.7,
                   alpha=0.8, zorder=3, label=f'not any drawn {a.ref_label} minimum ({(~on).sum()})')
        ax.scatter(pos[on, 0], pos[on, 1], s=22, marker=mk, facecolors=col, edgecolors=INK, linewidths=0.5,
                   zorder=4, label=f'is a drawn {a.ref_label} minimum ({on.sum()})')
        hit = S[np.ix_(wref, np.nonzero((label == lab) & (kind == 'sample'))[0])].any(1)
        ax.scatter(xy_min[hit, 0], xy_min[hit, 1], s=30, marker='o', facecolors='white', edgecolors=INK,
                   linewidths=1.0, zorder=5, label=f'{a.ref_label} minimum matched ({hit.sum()} of {len(wref)})')
        ax.scatter(xy_min[~hit, 0], xy_min[~hit, 1], s=34, marker='x', c=INK, linewidths=1.2, zorder=5,
                   label=f'{a.ref_label} minimum missed ({(~hit).sum()})')
        for k in deep:
            sg = Spacegroup(int(rows[files[np.nonzero(win)[0][k]]]['day_sg'])).symbol.replace(' ', '')
            ax.annotate(f'+{e_ref[win][k]:.1f} {sg}', xy_min[k], xytext=(4, 4), textcoords='offset points',
                        fontsize=6.5, color=INK, zorder=6,
                        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.7))
        for nm, f in forms.items():
            k = [j for j, i in enumerate(wref) if names[i].endswith('/' + f)]
            if k:
                ax.annotate(nm, xy_min[k[0]], xytext=(-14, -14), textcoords='offset points', fontsize=8, color=INK,
                            ha='right', zorder=6, arrowprops=dict(arrowstyle='-', color=INK, lw=0.6),
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
        all_hit = S[np.ix_(ref, np.nonzero((label == lab) & (kind == 'sample'))[0])].any(1)
        ax.set_title(f'{lab}: {len(s)} structures within {a.window:g} kJ/mol under {a.energy}\nBoltzmann weight '
                     f'covered: {w[win][hit].sum() / w[win].sum():.0%} of the drawn minima, '
                     f'{w[all_hit].sum() / w.sum():.0%} of all {len(ref)}', fontsize=8.5, color=INK, loc='left')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.legend(loc='upper left', fontsize=6.5, frameon=True, framealpha=0.9, edgecolor=GRID)
    cb = fig.colorbar(cf, ax=np.atleast_1d(axes).tolist(), orientation='horizontal', fraction=0.04, pad=0.03,
                      aspect=60)
    cb.set_label(f'blue fill: {a.ref_label}\'s minima within {a.window:g} kJ/mol (DFT-corrected energy above the lowest, '
                 f'kJ/mol per molecule); matched = within RDF {a.cut} or COMPACK 30/30', fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7)
    out = os.path.join(a.cov_dir, f'coverage_map_{a.energy}.png')
    fig.savefig(out, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'figure: {out}')
    print(f'Boltzmann weight (300 K) of the {len(ref)} {a.ref_label} minima held by the {win.sum()} drawn: '
          f'{w[win].sum() / w.sum():.0%}')


if __name__ == '__main__':
    main()
