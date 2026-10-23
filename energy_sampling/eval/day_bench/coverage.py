"""How much of a reference landscape each sampling source covers, against the energy calls it spends.

Every structure is a converged endpoint (``converge``: ``converged_<energy>/`` and ``converge_<energy>.csv``, whose
``calls`` column is the structure's cost in energy-and-gradient evaluations). Distances are the envwise RDF distance
(``rdf_embed.rdfs``, ``compute_rdf_distmat``) between the endpoints as P1 cells of rigid reference conformers.

The reference is one source's minima (Day's landscape relaxed under the benchmark energy), each weighted by its
Boltzmann factor ``exp(-(E - E_min) / kT)`` on the reference's own ranking energy (``ref_energy``: Day's
DFT-corrected energy), kT = R x 300 K. Metrics, borrowed from conformer generation (COV and MAT, Xu et al. 2021):

- COV-R(delta): the Boltzmann-weighted share of reference minima with a source structure within ``delta``;
- MAT-R: the Boltzmann-weighted mean distance from a reference minimum to its nearest source structure (no cut);
- COV-P(delta): the share of a source's structures within ``delta`` of a reference minimum inside ``WINDOW``;
- MAT-P: the mean distance from a source structure to its nearest reference minimum inside ``WINDOW``.

Identity ("a source structure is that reference minimum") is the RDF distance below each cut in ``--cuts``, and, when
``compack_band.py`` has written COMPACK verdicts for the pairs between the first two cuts
(``compack_band_<energy>.csv``), also "COMPACK-settled": within the first cut, or between the cuts and matched by
COMPACK on all 30 molecules. A band pair COMPACK did not settle (timeout, or not compared) counts as no match.

A budget is a prefix of the source's structures in a random order; COV-R and MAT-R are reported against the prefix's
summed energy calls, as the median and 10-90% band over ``ORDERS`` orders. The ceiling drawn on COV-R is the
Boltzmann-weighted share of reference minima whose primitive cell holds at most 4 molecules (the cell size the
generators were run at).

usage (licensed CSD Python API):
    python -m eval.day_bench.coverage OUT_DIR REFCODE REF_LABEL=DIR LABEL=DIR[+DIR...] [LABEL=DIR ...] [--energy exp6]
        [--cuts 0.093,0.14] [--orders 200] [--device cuda]

A source given as DIR+DIR pools the folders (for example one generator's runs at two cell sizes).

Writes OUT_DIR/coverage_<energy>.npz (the distance matrix with each structure's name, kind and CIF path, cached and
reused), coverage_<energy>.csv (the metrics at each source's full budget and at the smallest source's budget) and
coverage_<energy>.png.
"""
import argparse
import csv
import os
import warnings

import numpy as np
import torch
from ase.io import read

KT = 0.0083144626 * 300  # kJ/mol
from eval.day_bench.converge import F_TOL

WINDOW = 7.0  # kJ/mol above the reference's lowest, for the precision metrics
ORDERS = 200
STYLE = {'Clari-M': '#eb6834', 'MolCrystalFlow': '#008300', 'random': '#6b6b6b'}
INK, GRID = '#1f2328', '#c8ccd1'


def load_source(label, dirs, energy):
    """The converged structures of every folder in ``dirs`` ('+'-separated), named <label>:<folder>/<file>."""
    out = []
    for d in dirs.split('+'):
        out += _load_folder(label, d, energy)
    return out


def _load_folder(label, d, energy):
    rows = [r for r in csv.DictReader(open(os.path.join(d, f'converge_{energy}.csv'))) if r.get('calls')]
    first = {r['file']: r for r in csv.DictReader(open(os.path.join(d, f'p1_relax_{energy}.csv')))}
    out = []
    for r in rows:
        path = os.path.join(d, f'converged_{energy}', r['file'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            atoms = read(path)
        out.append(dict(label=label, file=f'{os.path.basename(os.path.normpath(d))}/{r["file"]}', path=path,
                        kind=r['kind'], calls=int(r['calls']),
                        converged=r['converged'] == 'True', e=float(r['e_after']), atoms=atoms,
                        ref_energy=float(first[r['file']]['ref_energy']) if r['kind'] == 'sample'
                        and 'ref_energy' in first.get(r['file'], {}) else np.nan,
                        day_sg=int(first[r['file']]['day_sg']) if r['kind'] == 'sample'
                        and first.get(r['file'], {}).get('day_sg') else 0, z=len(atoms) // 9))
    return out


def distances(entries, refcode, device):
    from ccdc import io as ccdc_io
    from eval.day_bench.csd_roundtrip import p1_cell
    from eval.day_bench.p1_ingest import make_reference, split_molecules
    from eval.day_bench.p1_relax import p1_crystal
    from eval.day_bench.rdf_embed import rdfs
    from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat
    reader = ccdc_io.EntryReader('CSD')
    atoms_ref, _, _ = p1_cell(refcode, reader)
    ref = make_reference(split_molecules(atoms_ref)[0], smiles=reader.entry(refcode).molecule.heaviest_component.smiles)
    crystals = []
    for e in entries:
        c, why = p1_crystal(e['atoms'], ref, identifier=e['file'])
        assert c is not None, f'{e["label"]}:{e["file"]} not rebuilt: {why}'
        crystals.append(c)
    rdf, bins = rdfs(crystals, device=device, chunk=32)
    assert torch.isfinite(rdf).all(), 'non-finite RDF'
    return compute_rdf_distmat(rdf, bins, show_tqdm=False).numpy()


def primitive_z(sg):
    from ase.spacegroup import Spacegroup
    return len(Spacegroup(int(sg)).get_rotations())


def curves(D_rs, M_rs, w, calls, rng, orders=ORDERS):
    """COV-R (identity given by the boolean [n_ref, n_src] ``M_rs``) and MAT-R after each prefix of ``orders`` random
    orders of the source's structures: arrays [orders, n_src] of the two metrics and of the prefix's summed calls."""
    n = D_rs.shape[1]
    cov, mat, cost = np.empty((orders, n)), np.empty((orders, n)), np.empty((orders, n))
    wn = w / w.sum()
    for o in range(orders):
        perm = rng.permutation(n)
        cov[o] = wn @ np.maximum.accumulate(M_rs[:, perm], axis=1)
        mat[o] = wn @ np.minimum.accumulate(D_rs[:, perm], axis=1)
        cost[o] = np.cumsum(calls[perm])
    return cov, mat, cost


def compack_matches(path, names):
    """The boolean matrix [n, n] of pairs COMPACK matched on every shell molecule (``compack_band``), or None when no
    verdict file exists; also the number of pairs it lists and how many of them it left unsettled."""
    if not os.path.exists(path):
        return None, 0, 0
    from eval.day_bench.compack_endpoints import SHELL
    index = {n: i for i, n in enumerate(names)}
    M = np.zeros((len(names), len(names)), dtype=bool)
    listed = unsettled = 0
    for r in csv.DictReader(open(path)):
        listed += 1
        if r['nmatched'] == '':
            unsettled += 1
            continue
        if int(r['nmatched']) == SHELL:
            i, j = index[r['a']], index[r['b']]
            M[i, j] = M[j, i] = True
    return M, listed, unsettled


def on_grid(metric, cost, grid):
    """Each order's metric at each budget on ``grid`` (the longest prefix within it; NaN before the first)."""
    out = np.full((metric.shape[0], len(grid)), np.nan)
    for o in range(metric.shape[0]):
        k = np.searchsorted(cost[o], grid, side='right') - 1
        ok = k >= 0
        out[o, ok] = metric[o, k[ok]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('refcode')
    ap.add_argument('reference')
    ap.add_argument('sources', nargs='+')
    ap.add_argument('--energy', default='exp6')
    ap.add_argument('--cuts', default='0.093,0.14')
    ap.add_argument('--orders', type=int, default=ORDERS)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    cuts = [float(c) for c in a.cuts.split(',')]
    ref_label, ref_dir = a.reference.split('=', 1)
    sources = [s.split('=', 1) for s in a.sources]
    os.makedirs(a.out_dir, exist_ok=True)

    entries = load_source(ref_label, ref_dir, a.energy)
    for lab, d in sources:
        entries += load_source(lab, d, a.energy)
    label = np.array([e['label'] for e in entries])
    kind = np.array([e['kind'] for e in entries])
    names = np.array([f'{e["label"]}:{e["file"]}' for e in entries])
    cache = os.path.join(a.out_dir, f'coverage_{a.energy}.npz')
    if os.path.exists(cache) and list(np.load(cache)['names']) == list(names):
        D = np.load(cache)['D']
    else:
        if str(a.device).startswith('cuda'):
            torch.cuda.set_per_process_memory_fraction(0.4)
        D = distances(entries, a.refcode, a.device)
        np.savez_compressed(cache, D=D, names=names, kind=kind, paths=np.array([e['path'] for e in entries]))

    ref = np.nonzero((label == ref_label) & (kind == 'sample'))[0]
    e_ref = np.array([entries[i]['ref_energy'] for i in ref])
    e_ref -= e_ref.min()
    w = np.exp(-e_ref / KT)
    reach = np.array([primitive_z(entries[i]['day_sg']) <= 4 for i in ref])
    in_win = ref[e_ref <= WINDOW]
    print(f'reference: {len(ref)} {ref_label} minima (to {e_ref.max():.1f} kJ/mol), {len(in_win)} within {WINDOW:g}; '
          f'Boltzmann weight at {KT:.3f} kJ/mol held by minima with at most 4 molecules per primitive cell: '
          f'{w[reach].sum() / w.sum():.1%}; unconverged structures: '
          f'{sum(not e["converged"] for e in entries)} of {len(entries)}')
    dd = D[np.ix_(ref, ref)] + np.eye(len(ref)) * 9
    print(f'{ref_label} minima closer than {cuts[0]} to another after convergence: {int((dd.min(1) <= cuts[0]).sum())}; '
          f'nearest-neighbour median {np.median(dd.min(1)):.3f}')
    exp = np.nonzero(kind == 'exp')[0]
    for i in exp:
        j = ref[np.argmin(D[i, ref])]
        print(f'  experimental {names[i]}: nearest {ref_label} minimum {names[j]} at {D[i, j]:.3f} '
              f'(+{entries[j]["ref_energy"] - min(entries[k]["ref_energy"] for k in ref):.2f} kJ/mol)')

    identities = {f'RDF <= {c}': D <= c for c in cuts}
    C, listed, unsettled = compack_matches(os.path.join(a.out_dir, f'compack_band_{a.energy}.csv'), names)
    if C is not None:
        identities = {'COMPACK-settled': (D <= cuts[0]) | C, **identities}
        print(f'COMPACK verdicts: {listed} pairs between {cuts[0]} and {cuts[1]}, {int(C.sum() // 2)} matched, '
              f'{unsettled} unsettled (counted as no match)')
    main_id = next(iter(identities))
    rng = np.random.default_rng(0)
    res, rows = {}, []
    small = min(sum(entries[i]['calls'] for i in np.nonzero((label == lab) & (kind == 'sample'))[0])
                for lab, _ in sources)
    for lab, _ in sources:
        s = np.nonzero((label == lab) & (kind == 'sample'))[0]
        calls = np.array([entries[i]['calls'] for i in s], dtype=float)
        D_rs = D[np.ix_(ref, s)]
        res[lab] = {k: curves(D_rs, M[np.ix_(ref, s)], w, calls, rng, a.orders) for k, M in identities.items()}
        row = dict(source=lab, structures=len(s),
                   cell_sizes='/'.join(str(z) for z in sorted({entries[i]['z'] for i in s})),
                   calls=int(calls.sum()), unconverged=int(sum(not entries[i]['converged'] for i in s)),
                   mat_p=float(D[np.ix_(s, in_win)].min(1).mean()))
        for k, M in identities.items():
            cov, mat, cost = res[lab][k]
            row[f'cov_r [{k}]'] = float(cov[0, -1])
            row[f'cov_r at {small} calls [{k}]'] = float(np.nanmedian(on_grid(cov, cost, np.array([small]))[:, 0]))
            row[f'cov_p [{k}]'] = float(M[np.ix_(s, in_win)].any(1).mean())
            row[f'ref minima matched [{k}]'] = int(M[np.ix_(ref, s)].any(1).sum())
        row['mat_r'] = float(res[lab][main_id][1][0, -1])
        row[f'mat_r at {small} calls'] = float(np.nanmedian(on_grid(res[lab][main_id][1], res[lab][main_id][2],
                                                                    np.array([small]))[:, 0]))
        rows.append(row)
    with open(os.path.join(a.out_dir, f'coverage_{a.energy}.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print('  ' + ', '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in r.items()))
    figure(os.path.join(a.out_dir, f'coverage_{a.energy}.png'), res, main_id, cuts, w, reach, a.refcode, ref_label,
           len(ref), e_ref.max(), a.energy, rows)


def figure(path, res, main_id, cuts, w, reach, refcode, ref_label, n_ref, e_top, energy, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hi = max(r['calls'] for r in rows)
    lo = min(min(res[lab][main_id][2][:, 0].min() for lab in res), hi)
    grid = np.geomspace(lo, hi, 200)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=170)
    for ax in axes:
        ax.set_facecolor('white')
        ax.set_xscale('log')
        ax.grid(color=GRID, lw=0.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(labelsize=8, colors=INK)
        ax.set_xlabel(f'energy calls spent ({energy} energy-and-gradient evaluations, log scale)', fontsize=8, color=INK)
    for lab in res:
        cov, mat, cost = res[lab][main_id]
        n = next(r['structures'] for r in rows if r['source'] == lab)
        for ax, metric in ((axes[0], cov), (axes[1], mat)):
            g = on_grid(metric, cost, grid)
            ok = (np.isfinite(g).mean(0) >= 0.5) & (grid <= cost[0, -1])  # each line ends at its own budget
            med = np.nanmedian(g[:, ok], 0)
            lo_b, hi_b = np.nanpercentile(g[:, ok], 10, 0), np.nanpercentile(g[:, ok], 90, 0)
            ax.fill_between(grid[ok], lo_b, hi_b, color=STYLE.get(lab, INK), alpha=0.18, lw=0)
            cells = next(r['cell_sizes'] for r in rows if r['source'] == lab)
            ax.plot(grid[ok], med, color=STYLE.get(lab, INK), lw=1.8,
                    label=f'{lab} ({n} structures; {cells} molecules per cell)')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel(f'COV-R: Boltzmann-weighted share of {ref_label} minima\nmatched by a structure '
                       f'(identity: {main_id})', fontsize=8, color=INK)
    axes[1].set_ylabel(f'MAT-R: Boltzmann-weighted mean RDF distance\nfrom a {ref_label} minimum to the nearest '
                       f'structure (lower is better)', fontsize=8, color=INK)
    axes[1].axhline(cuts[0], color=INK, lw=0.6, ls=':')
    axes[1].annotate(f'identity cut {cuts[0]}', (grid[-1], cuts[0]), xytext=(-2, 3), textcoords='offset points',
                     fontsize=7, color=INK, ha='right')
    axes[0].legend(loc='upper left', fontsize=7.5, frameon=True, edgecolor=GRID)
    fig.suptitle(f'{refcode}: coverage of {n_ref} {ref_label} minima (to {e_top:.0f} kJ/mol), Boltzmann-weighted at '
                 f'300 K on {ref_label}\'s DFT-corrected energies\nevery structure relaxed under {energy} to a largest '
                 f'force of {F_TOL} kJ/mol per A or rad; lines: median of {cov.shape[0]} random orders of each '
                 f'source\'s structures, bands 10-90%', fontsize=8, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    print(f'figure: {path}')


if __name__ == '__main__':
    main()
