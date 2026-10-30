"""FD distance against RDF distance on every pair of ~1k crystals of one molecule (sg 2, Z' = 1).

Two sets: MIPCAS campaign endpoints (physical shard states within --band_kT of the reference, drawn at random) and the
rows of one QM9 molecule of the 20-molecule prior (qm9_20_prior.pt). For every unordered pair:

  - RDF distance: the campaign's route (coordinator.rebuild_crystals for MIPCAS / the stored crystal for QM9, then
    standardize_cells, coordinator.compute_rdfs in MIPCAS's rdf_mode, coordinator.rdf_distance_matrix arithmetic);
  - FD distance, glued (fd_distance.GluedFDMetric.distance: cell faces glued, margin 1 A) and unglued (FDMetric);
  - FD with the mirror twin (min over the twin), only when the molecule's matched mirror RMSD is < 0.1 A.

Reports Spearman rank correlations (all pairs; pairs with RDF distance below 0.2; pairs with glued FD below 2 A),
quantiles of FD within RDF bins, and hexbin scatters. Outputs under OUT_ROOT/<--out> (refused if it exists).

    CUDA_VISIBLE_DEVICES=0 python -m energy_sampling.eval.fd_analysis.rdf_correlation --out rdf_corr_oct28
"""
import argparse
import os
import time

import numpy as np
import torch

from energy_sampling.eval.fd_analysis import common as C
from energy_sampling.eval.fd_analysis import fd_distance as FD

F64 = torch.float64
QM9_PRIOR = r'D:\crystal_datasets\conditional\priors\qm9_20_prior.pt'
QM9_MOLECULE = 'NC1(COC11CC1)C#N'


def spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).correlation), int(ok.sum())


def mipcas_set(n, band_kT, rng):
    co, cfg, reg, ref = C.load_campaign('mipcas')
    st = C.shard_states('mipcas', co, cfg, reg, ref, band_kT)
    del reg
    idx = np.sort(rng.choice(len(st['energy']), min(n, len(st['energy'])), replace=False))
    s = C.subset(st, idx)
    rd = C.rdfs(co, cfg, s['params'], s['hand'])
    mi = FD.molecule_info(cfg.mol_path)
    return dict(name='mipcas', sg=cfg.sg, params=s['params'], hand=s['hand'], energy=s['energy'], rdf=rd, mi=mi,
                rdf_mode=cfg.rdf_mode, note=f'{len(st["energy"])} shard states within {band_kT} kT; {len(idx)} drawn')


def qm9_set(n, rdf_mode, rng, scratch):
    from mxtaltools.crystal_search import coordinator as co
    from mxtaltools.crystal_search.standardize import standardize_cells
    from mxtaltools.dataset_utils.utils import collate_data_list
    d = torch.load(QM9_PRIOR, weights_only=False)
    b = d['prior']
    rows = [i for i, s in enumerate(b.identifier) if str(s) == QM9_MOLECULE]
    pick = np.sort(rng.choice(rows, min(n, len(rows)), replace=False))
    crystals = b.index_select(torch.as_tensor(pick)) if hasattr(b, 'index_select') else [b.get_example(int(i)) for i in pick]
    lst = list(crystals)
    cb = collate_data_list(lst)
    params = cb.full_cell_parameters().detach().float()
    hand = cb.aunit_handedness.reshape(-1, 1).float()
    energy = cb.elj.detach().double() if hasattr(cb, 'elj') else torch.zeros(len(pick), dtype=F64)
    mol_path = os.path.join(scratch, 'qm9_molecule.pt')
    torch.save(lst[0].clone(), mol_path)
    # the campaign's own RDF route (rebuild from parameters, standardize_cells, compute_rdfs), with this molecule
    cfg = co.CampaignConfig.load(os.path.join(C.CAMPAIGNS['mipcas']['dir'], 'coord.yaml'))
    cfg.mol_path, cfg.rdf_batch, cfg.sg = mol_path, C.RDF_BATCH, int(cb.sg_ind[0])
    rdf = C.rdfs(co, cfg, params, hand)
    mi = FD.molecule_info(mol_path)
    return dict(name='qm9', sg=int(cb.sg_ind[0]), params=params, hand=hand, energy=energy, rdf=rdf, mi=mi,
                rdf_mode=rdf_mode, note=f'{QM9_MOLECULE}: {len(rows)} prior rows, {len(pick)} drawn')


def analyse(S, device, out, log):
    n = len(S['params'])
    iu = torch.triu_indices(n, n, 1)
    ia, ib = iu[0], iu[1]
    ca, aa = C.rdf_prep(S['rdf'])
    R = C.rdf_dist_prepped(ca, aa, ca, aa).numpy()
    rdf = R[ia.numpy(), ib.numpy()]
    can = FD.canonical(S['sg'], S['params'], S['hand'])
    stat = {k: can['status'].count(k) for k in set(can['status'])}
    keep_rows = np.array([s in ('canonical', 'special_metric') for s in can['status']])
    rg = S['mi']['rg']
    t0 = time.time()
    gm = FD.GluedFDMetric(S['sg'], rg, margin=1.0, device=device)
    P = gm.prepare_glued(can)
    glued = torch.cat([gm.distance(P, ia[lo:lo + 200000], ib[lo:lo + 200000], chunk=65536).double().cpu()
                       for lo in range(0, len(ia), 200000)]).numpy()
    t_glued = time.time() - t0
    pm = FD.FDMetric(S['sg'], rg)
    A = pm.prepare(can)
    t0 = time.time()
    plain = torch.cat([pm.pairs(A, ia[lo:lo + 100000], A, ib[lo:lo + 100000]).double()
                       for lo in range(0, len(ia), 100000)]).numpy()
    t_plain = time.time() - t0
    twin = None
    if S['mi']['mirror_rmsd_matched'] < 0.1:
        tw = FD.twin_canonical(S['sg'], can, S['mi']['sigma'])
        Pt = gm.prepare_glued(tw)
        # d(A, twin of B): A's own rows against the twin set's glued alternatives
        d1 = torch.cat([gm.pairs(P[0], ia[lo:lo + 200000], Pt[1], ib[lo:lo + 200000], chunk=65536).double().cpu()
                        for lo in range(0, len(ia), 200000)])
        twin = np.minimum(glued, d1.numpy())
    ok = keep_rows[ia.numpy()] & keep_rows[ib.numpy()]
    lines = [f'## {S["name"]} (sg {S["sg"]}, Z\'=1): {S["note"]}', '',
             f'Canonical status {stat}; pairs used {int(ok.sum())} of {len(ia)} (both rows charted). r_g {rg:.3f} A; '
             f'matched mirror RMSD {S["mi"]["mirror_rmsd_matched"]:.3f} A (twin variant '
             f'{"run" if twin is not None else "not run: >= 0.1 A"}). RDF mode {S["rdf_mode"]}. Wall: glued FD '
             f'{t_glued:.1f} s on {device}, unglued FD {t_plain:.1f} s on the CPU, for {len(ia)} pairs.', '']
    variants = [('FD glued', glued), ('FD unglued', plain)] + ([('FD glued + twin', twin)] if twin is not None else [])
    lines += [f'*Spearman rank correlation with the RDF distance over the pairs named in each column ({S["name"]}, '
              f'{n} crystals, every unordered pair of charted rows). "near (RDF)": RDF distance < 0.2; "near (FD)": the '
              f'variant\'s own FD distance < 2 A; the count of pairs is in brackets.*', '',
              '| FD variant | all pairs | near (RDF < 0.2) | near (FD < 2 A) |', '|---|---|---|---|']
    for name, v in variants:
        r_all, n_all = spearman(rdf[ok], v[ok])
        m1 = ok & (rdf < 0.2)
        r1, n1 = spearman(rdf[m1], v[m1])
        m2 = ok & (v < 2.0)
        r2, n2 = spearman(rdf[m2], v[m2])
        lines.append(f'| {name} | {r_all:.3f} ({n_all}) | {r1:.3f} ({n1}) | {r2:.3f} ({n2}) |')
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]
    lines += ['', f'*FD distance (A) quantiles (P10 / P50 / P90) within RDF-distance bins ({S["name"]}); pairs per bin '
              f'in brackets. A metric that agrees with RDF rises monotonically down the rows with a narrow spread.*', '',
              '| RDF distance bin | pairs | ' + ' | '.join(n for n, _ in variants) + ' |',
              '|---|---|' + '---|' * len(variants)]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = ok & (rdf >= lo) & (rdf < hi)
        cells = [(' / '.join(f'{q:.2f}' for q in np.quantile(v[m], [0.1, 0.5, 0.9])) if m.any() else '-')
                 for _, v in variants]
        lines.append(f'| [{lo}, {hi}) | {int(m.sum())} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    plot(S['name'], rdf[ok], [(nm, v[ok]) for nm, v in variants], out)
    np.savez_compressed(os.path.join(out, f'{S["name"]}_pairs.npz'), ia=ia.numpy(), ib=ib.numpy(), rdf=rdf,
                        glued=glued, plain=plain, twin=twin if twin is not None else np.zeros(0), ok=ok)
    for l in lines:
        log(l)
    return lines


def plot(name, rdf, variants, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 4.4), squeeze=False)
    for ax, (vn, v) in zip(axs[0], variants):
        m = np.isfinite(v)
        hb = ax.hexbin(rdf[m], v[m], gridsize=70, bins='log', mincnt=1, cmap='viridis')
        ax.set_xlabel('RDF distance (campaign route)')
        ax.set_ylabel(f'{vn} (A)')
        ax.set_title(f'{name}: {vn}')
        fig.colorbar(hb, ax=ax, label='pairs (log)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'{name}_rdf_vs_fd.png'), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--band_kT', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--sets', default='mipcas,qm9', help='comma list of mipcas, qm9 (qm9 uses atomwise RDFs alone)')
    a = ap.parse_args()
    torch.set_num_threads(8)
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.4)  # shared card
    out = C.new_out_dir(a.out)
    rng = np.random.default_rng(a.seed)
    lines = [f'# FD distance vs RDF distance (args {vars(a)})', '']
    sets = a.sets.split(',')
    rdf_mode = 'atomwise'  # MIPCAS's campaign mode
    if 'mipcas' in sets:
        M = mipcas_set(a.n, a.band_kT, rng)
        rdf_mode = M['rdf_mode']
        lines += analyse(M, a.device, out, C.log)
    if 'qm9' in sets:
        Q = qm9_set(a.n, rdf_mode, rng, out)
        lines += analyse(Q, a.device, out, C.log)
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    C.log(f'wrote {out}')


if __name__ == '__main__':
    main()
