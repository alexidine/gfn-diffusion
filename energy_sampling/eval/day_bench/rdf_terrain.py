"""A reference landscape as an energy terrain in a map fitted on the reference alone, generator endpoints placed on it.

Three ways to build the map, to compare how stable each is:
- ``A``: UMAP fitted on the reference minima (one point each), ``n_neighbors`` K, samples placed by ``transform``;
- ``B``: the reference densified with COPIES rigidly noised copies of every minimum (each molecule shifted ~N(0,
  SHIFT) Angstrom per axis and turned ~N(0, TURN) degrees about its centroid, no relaxation), fitted with
  ``n_neighbors`` 15, samples placed by ``transform``;
- ``C``: as B, samples placed at the inverse-distance-weighted mean of their 5 nearest reference points' positions.

The terrain is the lower envelope over reference points (copies inherit their minimum's energy) of
E_i + (window / r0^2)|x - x_i|^2, r0 = 6% of the map's span, blank beyond 10%. Reference energy = the reference's
``ref_energy`` column (Day: DFT-corrected) above its lowest; every reference minimum shapes the map, the terrain is
capped at WINDOW. Generator endpoints within WINDOW kJ/mol of the lowest endpoint under the benchmark energy are
all drawn, one colour and symbol per generator.

usage (licensed CSD Python API; GPU for the RDFs):
    python -m eval.day_bench.rdf_terrain OUT_DIR REFCODE OTHER_FORM REF_LABEL=REF_DIR LABEL=DIR [LABEL=DIR ...]
"""
import csv
import glob
import os
import sys

import matplotlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from eval.day_bench.p1_ingest import make_reference, split_molecules
from eval.day_bench.p1_relax import p1_crystal
from eval.day_bench.rdf_embed import load, rdfs
from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#c3c2b7'
GEN_STYLE = [dict(marker='o', color='#eb6834'), dict(marker='^', color='#008300'), dict(marker='D', color='#e87ba4')]
ENERGY, WINDOW, COPIES, SHIFT, TURN, K_A = 'exp6', 7.0, 10, 0.2, 5.0, 5


def noised(atoms, rng, shift, turn):
    """``atoms`` with every molecule rigidly shifted and turned about its centroid (same cell)."""
    from ase import Atoms
    mols = split_molecules(atoms)
    z, pos = [], []
    for m in mols:
        c = m.pos.mean(0)
        R = Rotation.from_rotvec(rng.normal(size=3) / np.sqrt(3) * np.radians(turn)).as_matrix()
        pos.append((m.pos - c) @ R.T + c + rng.normal(scale=shift, size=3))
        z.append(m.z)
    return Atoms(numbers=np.concatenate(z), positions=np.concatenate(pos), cell=atoms.cell, pbc=True)


def main(out_dir, refcode, other_form, ref_src, sources, shift=SHIFT, turn=TURN, copies=COPIES, tag_sfx=''):
    from ccdc import io as ccdc_io
    from eval.day_bench.csd_roundtrip import p1_cell
    import umap
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(2)
    reader = ccdc_io.EntryReader('CSD')
    atoms_ref, _, _ = p1_cell(refcode, reader)
    ref = make_reference(split_molecules(atoms_ref)[0], smiles=reader.entry(refcode).molecule.heaviest_component.smiles)
    rng = np.random.default_rng(0)
    ref_label, ref_dir = ref_src
    os.makedirs(out_dir, exist_ok=True)

    rows = [r for r in csv.DictReader(open(os.path.join(ref_dir, f'p1_relax_{ENERGY}.csv'))) if r['built'] == 'True']
    cache = os.path.join(out_dir, f'rdf_terrain{tag_sfx}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        D, kind, label, energy, e6 = z['D'], z['kind'].astype(str), z['label'].astype(str), z['energy'], z['e6']
        return figures(D, kind, label, energy, e6, out_dir, refcode, ref_label, sources, copies, shift, turn, tag_sfx,
                       parent_sg(kind, rows))
    kind, label, energy, e6, crystals = [], [], [], [], []

    def add(k, lab, en, atoms, en6=np.nan):
        c, why = p1_crystal(atoms, ref)
        if c is not None:
            kind.append(k), label.append(lab), energy.append(en), e6.append(en6), crystals.append(c)

    e_ref = np.array([float(r['ref_energy']) for r in rows])
    e_ref -= e_ref.min()
    e_ref6 = np.array([float(r['e_end']) for r in rows])
    e_ref6 -= e_ref6.min()
    for r, er, er6 in zip(rows, e_ref, e_ref6):
        atoms = load(os.path.join(ref_dir, f'relaxed_{ENERGY}', r['file']))
        add('ref', ref_label, er, atoms, er6)
        for _ in range(copies):
            add('copy', ref_label, er, noised(atoms, rng, shift, turn), er6)
    floor = min(float(r['e_end']) for _, d in sources for r in csv.DictReader(open(os.path.join(d, f'p1_relax_{ENERGY}.csv')))
                if r['built'] == 'True')
    for lab, d in sources:
        for r in csv.DictReader(open(os.path.join(d, f'p1_relax_{ENERGY}.csv'))):
            if r['built'] == 'True' and float(r['e_end']) - floor <= WINDOW:
                add('gen', lab, float(r['e_end']) - floor, load(os.path.join(d, f'relaxed_{ENERGY}', r['file'])))
    for rc in (refcode, other_form):
        add('exp', rc, np.nan, load(os.path.join(sources[0][1], f'relaxed_{ENERGY}', f'exp_{rc}.cif')))
    kind, label, energy, e6 = np.array(kind), np.array(label), np.array(energy), np.array(e6)
    print(f'{len(crystals)} structures: {dict(zip(*np.unique(kind, return_counts=True)))}', flush=True)
    rdf, bins = rdfs(crystals, device='cuda', chunk=32)
    D = compute_rdf_distmat(rdf, bins, show_tqdm=False).numpy()
    np.savez_compressed(os.path.join(out_dir, f'rdf_terrain{tag_sfx}.npz'), D=D, kind=kind, label=label, energy=energy, e6=e6)
    figures(D, kind, label, energy, e6, out_dir, refcode, ref_label, sources, copies, shift, turn, tag_sfx,
            parent_sg(kind, rows))


def parent_sg(kind, rows):
    """Each reference minimum's space group (Day's ``day_sg``) at its own entry, 0 at copies and other entries; the
    entries were added in the reference CSV's row order, each minimum followed by its copies."""
    refs = np.nonzero(kind == 'ref')[0]
    assert len(refs) == len(rows), f'{len(refs)} reference entries for {len(rows)} reference rows'
    out = np.zeros(len(kind), dtype=int)
    out[refs] = [int(r['day_sg']) for r in rows]
    return out


def primitive_z(sg):
    """Molecules in the primitive cell of a Z' = 1 crystal in space group ``sg``: its point-group order."""
    from ase.spacegroup import Spacegroup
    return len(Spacegroup(int(sg)).get_rotations())


def figures(D, kind, label, energy, e6, out_dir, refcode, ref_label, sources, copies, shift, turn, tag_sfx, ref_sg):
    import umap
    ref_only, ref_all = np.nonzero(kind == 'ref')[0], np.nonzero((kind == 'ref') | (kind == 'copy'))[0]
    placed = np.nonzero((kind == 'gen') | (kind == 'exp'))[0]
    in_win = np.nonzero(((kind == 'ref') | (kind == 'copy')) & (energy <= WINDOW))[0]
    variants = [('A', ref_only, K_A, 'transform')] + ([('B', ref_all, 15, 'transform'), ('C', ref_all, 15, 'knn'),
                                                     ('D', in_win, 15, 'transform'), ('E', in_win, 15, 'knn')]
                                                    if copies else [])
    for tag, fit, k, how in variants:
        reducer = umap.UMAP(metric='precomputed', n_neighbors=k, min_dist=0.3, random_state=0)
        xy_fit = reducer.fit_transform(D[np.ix_(fit, fit)])
        if how == 'transform':
            xy_p = reducer.transform(D[np.ix_(placed, fit)])
        else:
            dd = D[np.ix_(placed, fit)]
            nn = np.argsort(dd, 1)[:, :5]
            w = 1.0 / np.maximum(np.take_along_axis(dd, nn, 1), 1e-6)
            xy_p = (xy_fit[nn] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        title = {'A': f'A: fitted on the {len(fit)} {ref_label} minima, {k} neighbours, samples by transform',
                 'B': f'B: + {copies} copies per minimum noised {shift} A / {turn} deg ({len(fit)} points), samples by transform',
                 'C': f'C: as B, samples at the weighted mean of their 5 nearest reference points',
                 'D': f'D: only {ref_label} minima within {WINDOW:g} kJ/mol, with copies ({len(fit)} points), transform',
                 'E': f'E: as D, samples at the weighted mean of their 5 nearest reference points'}[tag]
        draw(os.path.join(out_dir, f'terrain_{ref_label}_{tag}{tag_sfx}.png'), title, xy_fit, energy[fit], xy_p,
             kind[placed], label[placed], [lab for lab, _ in sources], ref_label, refcode, e6[fit], ref_sg[fit])
        print(f'variant {tag} drawn', flush=True)


def draw(path, title, xy_ref, e_ref, xy_p, kind_p, label_p, gens, ref_label, refcode, e6_ref=None, sg_ref=None):
    span = np.ptp(xy_ref, axis=0).max()
    lo, hi = xy_ref.min(0) - 0.1 * span, xy_ref.max(0) + 0.1 * span
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 360), np.linspace(lo[1], hi[1], 360))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    d2 = ((grid[:, None, :] - xy_ref[None, :, :]) ** 2).sum(-1)
    r0 = 0.06 * span
    terrain = (np.minimum(e_ref, WINDOW)[None, :] + WINDOW / r0 ** 2 * d2).min(1)
    terrain[np.sqrt(d2.min(1)) > 0.10 * span] = np.nan
    terrain = np.minimum(terrain, WINDOW).reshape(gx.shape)
    cmap = LinearSegmentedColormap.from_list('terrain_blue', ['#0d366b', '#2a78d6', '#cde2fb'])
    fig, ax = plt.subplots(figsize=(9.0, 6.6), dpi=170)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    levels = np.arange(0, WINDOW + 1e-6, 1.0)
    cf = ax.contourf(gx, gy, terrain, levels=levels, cmap=cmap, zorder=0)
    ax.contour(gx, gy, terrain, levels=levels, colors='white', linewidths=0.3, zorder=1)
    cb = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label(f"blue fill: {ref_label}'s own (DFT-corrected) energy above its lowest (kJ/mol per molecule)",
                 fontsize=7, color=INK)
    cb.ax.tick_params(labelsize=7)
    if e6_ref is not None and np.isfinite(e6_ref).all():
        # the same reference minima under the benchmark energy, as line contours on a second (purple) scale: where
        # its deep rings sit on pale blue, exp-6 ranks a basin lower than Day does, and the reverse
        t6 = (np.minimum(e6_ref, WINDOW)[None, :] + WINDOW / r0 ** 2 * d2).min(1)
        t6[np.sqrt(d2.min(1)) > 0.10 * span] = np.nan
        t6 = np.minimum(t6, WINDOW).reshape(gx.shape)
        purples = LinearSegmentedColormap.from_list('exp6_purple', ['#2d1b69', '#6250d6', '#c9c2f2'])
        cl = ax.contour(gx, gy, t6, levels=[0.5, 1.5, 2.5], cmap=purples, vmin=0, vmax=4, linewidths=1.2,
                        zorder=2)
        cb2 = fig.colorbar(cl, ax=ax, orientation='horizontal', fraction=0.035, pad=0.02, aspect=50)
        cb2.set_label(f'purple rings: the same {ref_label} minima re-relaxed under {ENERGY}, energy above their lowest '
                      f'(kJ/mol per molecule)', fontsize=7, color=INK)
        cb2.ax.tick_params(labelsize=7)
    if sg_ref is not None:
        # which reference minima a 4-molecule sample cannot be (primitive cell of 8), and the deep wells by name
        from ase.spacegroup import Spacegroup
        par = sg_ref > 0
        big = par & np.array([s > 0 and primitive_z(s) > 4 for s in sg_ref])
        ax.scatter(xy_ref[big, 0], xy_ref[big, 1], marker='x', s=22, c=INK, linewidths=0.9, zorder=2.5,
                   label=f'{ref_label} minimum with 8 molecules in its primitive cell (out of reach of a 4-molecule sample)')
        for q, s, e in zip(xy_ref[par & (e_ref <= 4.5)], sg_ref[par & (e_ref <= 4.5)], e_ref[par & (e_ref <= 4.5)]):
            ax.annotate(f'+{e:.1f} {Spacegroup(int(s)).symbol.replace(" ", "")}', q, xytext=(4, 4),
                        textcoords='offset points', fontsize=6.5, color=INK, zorder=6,
                        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.7))
    for k, g in enumerate(gens):
        sel = (kind_p == 'gen') & (label_p == g)
        st = GEN_STYLE[k]
        ax.scatter(xy_p[sel, 0], xy_p[sel, 1], s=32, marker=st['marker'], facecolors=st['color'], edgecolors=INK,
                   linewidths=0.6, zorder=3, label=f'{g}, relaxed, within {WINDOW:g} kJ/mol under {ENERGY} ({sel.sum()})')
    sel = kind_p == 'exp'
    ax.scatter(xy_p[sel, 0], xy_p[sel, 1], s=220, marker='*', facecolors='white', edgecolors=INK, linewidths=1.0, zorder=5)
    for q, nm in zip(xy_p[sel], label_p[sel]):
        ax.annotate(f'{nm} (relaxed)', q, xytext=(q[0] + 0.05 * span, q[1] + 0.05 * span), fontsize=8, color=INK, zorder=6,
                    arrowprops=dict(arrowstyle='-', color=INK, lw=0.6),
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    pts = np.vstack([xy_ref, xy_p])
    pad = 0.08 * span
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(f'{refcode}: {title}', fontsize=9, color=INK, loc='left')
    leg = ax.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.95, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    out_dir, refcode, other_form = sys.argv[1], sys.argv[2], sys.argv[3]
    ref_src = tuple(sys.argv[4].split('=', 1))
    sources = [tuple(s.split('=', 1)) for s in sys.argv[5:]]
    kw = {}
    for a in list(sources):
        if a[0].startswith('--'):
            sources.remove(a)
            kw[a[0][2:]] = float(a[1]) if a[0][2:] in ('shift', 'turn') else (int(a[1]) if a[0][2:] == 'copies' else a[1])
    main(out_dir, refcode, other_form, ref_src, sources, **kw)
