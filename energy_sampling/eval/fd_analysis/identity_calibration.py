"""Where should an FD identity cut sit? Nearest-neighbour FD distances among relaxed crystals of one molecule.

For each set, every crystal's nearest other crystal under the glued FD distance (fd_distance.GluedFDMetric, cell faces
glued, margin 1 A; with the molecule's mirror twin as well when its matched mirror RMSD is < 0.1 A), searched over all
pairs on the GPU with a cutoff. Two relaxations into one minimum land close together (relaxation noise); distinct
minima sit farther apart; an identity cut belongs in the gap between the two populations, if there is one.

Sets: the MIPCAS (sg 2) and NEHZOR (sg 14) campaigns (distinct physical shard states within --band_kT, with the
campaign registry's basin of each state and its energy), and the rows of three QM9 molecules of the 20-molecule prior
(sg 2, no basin labels). Reports, per set: quantiles and a log-binned histogram of the nearest-neighbour distance;
for the campaigns, the same split by whether the neighbour is in the same registry basin (RDF clustering at the
campaign's cut), and |dE| of nearest pairs by distance band.

    CUDA_VISIBLE_DEVICES=0 python -m energy_sampling.eval.fd_analysis.identity_calibration --out fd_identity_oct28
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
QM9_MOLECULES = ('NC1(COC11CC1)C#N', 'OCC(=O)N1C=NN=C1', 'CC(=O)C1C2NC(=O)C12')
SEARCH_CUT = 3.0  # A: nearest neighbours farther than this are reported as > SEARCH_CUT


def campaign_set(name, n, band_kT, rng):
    co, cfg, reg, ref = C.load_campaign(name)
    st = C.distinct(C.shard_states(name, co, cfg, reg, ref, band_kT))
    del reg
    idx = np.sort(rng.choice(len(st['energy']), min(n, len(st['energy'])), replace=False))
    s = C.subset(st, idx)
    return dict(name=name, sg=cfg.sg, params=s['params'], hand=s['hand'], energy=s['energy'].double(),
                basin=s['basin'], mi=FD.molecule_info(cfg.mol_path), kT=float(cfg.kT),
                note=f'{len(st["energy"])} distinct physical shard states within {band_kT} kT; {len(idx)} drawn; '
                     f'campaign RDF identity cut {cfg.identity_cut}')


def qm9_sets(scratch):
    from mxtaltools.dataset_utils.utils import collate_data_list
    d = torch.load(QM9_PRIOR, weights_only=False)
    b = d['prior']
    ids = [str(s) for s in b.identifier]
    out = []
    for smi in QM9_MOLECULES:
        rows = [i for i, s in enumerate(ids) if s == smi]
        lst = [b.get_example(i) for i in rows]
        cb = collate_data_list(lst)
        mol_path = os.path.join(scratch, f'qm9_{len(out)}.pt')
        torch.save(lst[0].clone(), mol_path)
        e = cb.elj.detach().double() if hasattr(cb, 'elj') and cb.elj is not None else torch.zeros(len(rows), dtype=F64)
        out.append(dict(name=f'qm9:{smi}', sg=int(cb.sg_ind[0]), params=cb.full_cell_parameters().detach().float(),
                        hand=cb.aunit_handedness.reshape(-1, 1).float(), energy=e, basin=None,
                        mi=FD.molecule_info(mol_path), kT=None, note=f'{len(rows)} prior rows (all)'))
    return out


def nearest(S, device, cut=SEARCH_CUT, rows_per=50):
    """Per crystal: nearest other crystal's index and glued FD distance (inf beyond cut), plain and with the twin."""
    can = FD.canonical(S['sg'], S['params'], S['hand'])
    n = len(can['cell'])
    gm = FD.GluedFDMetric(S['sg'], S['mi']['rg'], margin=1.0, device=device)
    P = gm.prepare_glued(can)
    twin = S['mi']['mirror_rmsd_matched'] < 0.1
    Pt = gm.prepare_glued(FD.twin_canonical(S['sg'], can, S['mi']['sigma'])) if twin else None
    nn_d = torch.full((n,), float('inf'), dtype=F64)
    nn_i = torch.full((n,), -1, dtype=torch.long)
    nn_dt = torch.full((n,), float('inf'), dtype=F64)
    nn_it = torch.full((n,), -1, dtype=torch.long)
    t0 = time.time()
    allj = torch.arange(n)
    for lo in range(0, n, rows_per):
        rows = torch.arange(lo, min(n, lo + rows_per))
        a = rows.repeat_interleave(n)
        b = allj.repeat(len(rows))
        d = gm.distance(P, a, b, cut=cut, chunk=65536).double().cpu().reshape(len(rows), n)
        d[torch.arange(len(rows)), rows] = float('inf')
        v, j = d.min(1)
        nn_d[rows], nn_i[rows] = v, j
        if twin:
            dt = gm.pairs(P[0], a, Pt[1], b, chunk=65536, cut=cut).double().cpu().reshape(len(rows), n)
            dt[torch.arange(len(rows)), rows] = float('inf')
            dd = torch.minimum(d, dt)
            v, j = dd.min(1)
            nn_dt[rows], nn_it[rows] = v, j
    status = {k: can['status'].count(k) for k in set(can['status'])}
    return dict(d=nn_d, i=nn_i, dt=nn_dt if twin else None, it=nn_it if twin else None, twin=twin, status=status,
                wall=time.time() - t0, n=n)


BANDS = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]


def report(S, R, out):
    lines = [f'## {S["name"]} (sg {S["sg"]}): {S["note"]}', '',
             f'Canonical status {R["status"]}; r_g {S["mi"]["rg"]:.3f} A; matched mirror RMSD '
             f'{S["mi"]["mirror_rmsd_matched"]:.3f} A (twin {"counted in the second column set" if R["twin"] else "not counted: >= 0.1 A"}). '
             f'Nearest-neighbour search over all {R["n"] * (R["n"] - 1)} ordered pairs, cut {SEARCH_CUT} A: {R["wall"]:.0f} s.', '']
    variants = [('plain', R['d'], R['i'])] + ([('with twin', R['dt'], R['it'])] if R['twin'] else [])
    for vn, d, i in variants:
        dn = d.numpy()
        fin = np.isfinite(dn)
        q = np.quantile(dn[fin], [0.1, 0.25, 0.5, 0.75, 0.9]) if fin.any() else [np.nan] * 5
        lines += [f'*Nearest-neighbour glued FD distance, {vn} ({S["name"]}, {R["n"]} crystals): crystals per distance '
                  f'band; for the campaigns also the share whose nearest neighbour is in the same registry basin and '
                  f'the median |dE| (eLJ units) to it. Quantiles over neighbours within {SEARCH_CUT} A '
                  f'({int(fin.sum())} crystals): P10 {q[0]:.3f}, P25 {q[1]:.3f}, P50 {q[2]:.3f}, P75 {q[3]:.3f}, '
                  f'P90 {q[4]:.3f} A.*', '']
        head = '| nearest-neighbour FD (A) | crystals |'
        sep = '|---|---|'
        if S['basin'] is not None:
            head += ' same registry basin (share) | median abs dE (eLJ) |'
            sep += '---|---|'
        lines += [head, sep]
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            m = (dn >= lo) & (dn < hi) if np.isfinite(hi) else ~(dn < lo)
            row = f'| [{lo}, {hi}) | {int(m.sum())} |'
            if S['basin'] is not None and m.any():
                mm = torch.as_tensor(m)
                j = i[mm].clamp(min=0)
                same = (S['basin'][mm] == S['basin'][j]) & (S['basin'][mm] >= 0)
                de = (S['energy'][mm] - S['energy'][j]).abs()
                row += f' {float(same.double().mean()):.2f} | {float(de.median()):.3g} |'
            elif S['basin'] is not None:
                row += ' - | - |'
            lines.append(row)
        lines.append('')
    plot(S, R, out)
    return lines


def plot(S, R, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants = [('plain', R['d'], R['i'])] + ([('with twin', R['dt'], R['it'])] if R['twin'] else [])
    fig, axs = plt.subplots(1, len(variants), figsize=(5.5 * len(variants), 4), squeeze=False)
    bins = np.logspace(-4, np.log10(SEARCH_CUT), 60)
    for ax, (vn, d, i) in zip(axs[0], variants):
        dn = d.numpy()
        fin = np.isfinite(dn) & (dn > 0)
        if S['basin'] is not None:
            same = ((S['basin'] == S['basin'][i.clamp(min=0)]) & (S['basin'] >= 0)).numpy()
            ax.hist([dn[fin & same], dn[fin & ~same]], bins=bins, stacked=True, label=['same registry basin', 'other basin'])
            ax.legend(fontsize=8)
        else:
            ax.hist(dn[fin], bins=bins)
        ax.set_xscale('log')
        ax.set_xlabel('nearest-neighbour glued FD distance (A)')
        ax.set_ylabel('crystals')
        ax.set_title(f'{S["name"][:34]}: {vn}', fontsize=9)
        zero = int((dn == 0).sum())
        if zero:
            ax.text(0.02, 0.95, f'{zero} at exactly 0', transform=ax.transAxes, fontsize=8, va='top')
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'nn_{S["name"].replace(":", "_").replace("(", "").replace(")", "").replace("=", "").replace("#", "")[:40]}.png'), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--band_kT', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sets', default='mipcas,nehzor,qm9')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    torch.set_num_threads(8)
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.4)  # shared card
    out = C.new_out_dir(a.out)
    rng = np.random.default_rng(a.seed)
    sets = []
    for nm in a.sets.split(','):
        if nm in ('mipcas', 'nehzor'):
            sets.append(campaign_set(nm, a.n, a.band_kT, rng))
        elif nm == 'qm9':
            sets += qm9_sets(out)
    lines = [f'# Nearest-neighbour FD distances: where an identity cut can sit (args {vars(a)})', '']
    for S in sets:
        C.log(f'{S["name"]}: {S["note"]}')
        R = nearest(S, a.device)
        L = report(S, R, out)
        for l in L:
            C.log(l)
        lines += L
        torch.save(dict(S={k: v for k, v in S.items() if k != 'mi'}, R=R), os.path.join(out, f'{S["name"].split(":")[0]}_{len(lines)}.pt'))
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    C.log(f'wrote {out}')


if __name__ == '__main__':
    main()
