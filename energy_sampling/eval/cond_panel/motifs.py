"""
Packing motifs, read from geometry, for the model's draws and for the molecules' search
minima: which neighbours a molecule touches and how, compared between the two.

Every crystal is rebuilt from its latents through MolecularCrystal.instantiate_crystals (the
trainer's own construction) and expanded with mol2cluster; only the neighbours MXtalTools
marks as near (aux_ind 1) are read. In P-1 with Z' = 1 a neighbour is either a translate of
the central molecule or its inverse, and that relation is recorded for every contact.

What is measured, per crystal:

  contacts          molecules with any atom pair closer than the sum of van der Waals radii
                    plus 0.3 A (the coordination), and whether the closest of them is the
                    inverse or a translate
  donor contacts    N-H / O-H hydrogens of the central molecule within the H...N/O van der
                    Waals sum (2.75 A) of an N or O of a neighbour, at a D-H...A angle of at
                    least 130 degrees. ELJ has no hydrogen-bond term, so this is the geometry
                    of a hydrogen bond, not evidence that one is rewarded
  ring stacks       a planar unsaturated five- or six-ring of the central molecule over one
                    of a neighbour: planes within 20 degrees, 3.0-4.2 A apart, centroids
                    slipped by at most 2.5 A

A motif is a yes/no reading of thresholds, never a nearest-prototype assignment: a crystal
with no donor contact or no stack is recorded as having none. Bonds, donors and rings come
from the stored geometry (covalent radii), so they are in the file's own atom order; the ring
count is checked against RDKit's aromatic-ring count of the SMILES and the agreement printed.

    cd energy_sampling
    CUDA_VISIBLE_DEVICES=-1 python -m eval.cond_panel.motifs --checkpoint ... --config ... \
        --atlas <dir>/atlas/atlas_step50000.pt --test-prior D:/.../qm9full_test_prior.pt --out <dir>/motifs
"""
from __future__ import annotations

import argparse
import glob
import os

import networkx as nx
import numpy as np
import torch

from eval.cond_panel.sampler import load_run

COVALENT = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}
VDW = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47}
CONTACT_MARGIN, DONOR_CUT, DONOR_ANGLE = 0.3, 2.75, 130.0
STACK_ANGLE, STACK_PLANES, STACK_SLIP = 20.0, (3.0, 4.2), 2.5
KEYS = ('coordination', 'closest_is_inverse', 'closest_gap', 'n_donor_contacts', 'donor_to_inverse', 'donor_to_translate',
        'shortest_donor_dist', 'stacked', 'stack_to_inverse', 'stack_slip')


def template(z, pos):
    """Bonds, donor hydrogens, acceptors and planar unsaturated rings of one molecule, from
    its stored geometry."""
    z, pos = np.asarray(z), np.asarray(pos, dtype=np.float64)
    cov = np.array([COVALENT[int(a)] for a in z])
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    bonded = (d < 1.3 * (cov[:, None] + cov[None])) & ~np.eye(len(z), dtype=bool)
    donors = [(int(np.flatnonzero(bonded[h] & np.isin(z, (7, 8)))[0]), int(h))
              for h in np.flatnonzero(z == 1) if (bonded[h] & np.isin(z, (7, 8))).any()]
    acceptors = np.flatnonzero(np.isin(z, (7, 8)))
    heavy = np.flatnonzero(z > 1)
    g = nx.Graph()
    g.add_nodes_from(heavy.tolist())
    g.add_edges_from((int(i), int(j)) for i in heavy for j in heavy if i < j and bonded[i, j])
    rings = []
    for cyc in nx.minimum_cycle_basis(g):
        if len(cyc) not in (5, 6) or any(bonded[a].sum() > 3 for a in cyc):
            continue
        p = pos[cyc] - pos[cyc].mean(0)
        if np.sqrt(np.linalg.svd(p, compute_uv=False)[-1] ** 2 / len(cyc)) < 0.06:   # rms distance from the best plane
            rings.append(np.array(sorted(cyc)))
    vdw = np.array([VDW[int(a)] for a in z])
    return {'donors': donors, 'acceptors': acceptors, 'rings': rings, 'vdw_sum': vdw[:, None] + vdw[None]}


def ring_frame(pos, ring):
    p = pos[..., ring, :]
    c = p.mean(-2)
    normal = np.linalg.svd(p - c[..., None, :])[2][..., -1, :]
    return c, normal


def describe(tpl, central, images):
    """The motif readings of one crystal: `central` [n, 3], `images` [m, n, 3] near neighbours."""
    out = dict.fromkeys(KEYS, np.nan)
    c0 = central - central.mean(0)
    rel = images - images.mean(1, keepdims=True)
    inverse = np.abs(rel + c0[None]).max((1, 2)) < np.abs(rel - c0[None]).max((1, 2))
    d = np.linalg.norm(images[:, :, None, :] - central[None, None, :, :], axis=-1)       # [m, image atom, central atom]
    gap = (d - tpl['vdw_sum'][None]).min((1, 2))
    out['coordination'] = float((gap < CONTACT_MARGIN).sum())
    out['closest_gap'] = float(gap.min())
    out['closest_is_inverse'] = float(inverse[gap.argmin()])
    if tpl['donors'] and len(tpl['acceptors']):
        n_c, to_inv, to_tr, shortest = 0, False, False, np.inf
        for dn, h in tpl['donors']:
            v_ha = images[:, tpl['acceptors'], :] - central[h]                              # [m, acceptors, 3]
            dist = np.linalg.norm(v_ha, axis=-1)
            v_hd = central[dn] - central[h]
            cos = (v_ha @ v_hd) / (dist * np.linalg.norm(v_hd) + 1e-12)
            ok = np.degrees(np.arccos(np.clip(cos, -1, 1))) >= DONOR_ANGLE
            if ok.any():
                shortest = min(shortest, float(dist[ok].min()))
            hit = ok & (dist <= DONOR_CUT)
            n_c += int(hit.sum())
            to_inv |= bool(hit[inverse].any())
            to_tr |= bool(hit[~inverse].any())
        out.update(n_donor_contacts=float(n_c), donor_to_inverse=float(to_inv), donor_to_translate=float(to_tr),
                   shortest_donor_dist=shortest if np.isfinite(shortest) else np.nan)
    if tpl['rings']:
        stacked, to_inv, slip_best = False, False, np.inf
        for ring in tpl['rings']:
            c, n = ring_frame(central, ring)
            for other in tpl['rings']:
                cj, nj = ring_frame(images, other)                                          # [m, 3]
                sep = cj - c
                planes = np.abs(sep @ n)
                slip = np.sqrt(np.clip((sep ** 2).sum(-1) - planes ** 2, 0, None))
                angle = np.degrees(np.arccos(np.clip(np.abs(nj @ n), 0, 1)))
                hit = (angle <= STACK_ANGLE) & (planes >= STACK_PLANES[0]) & (planes <= STACK_PLANES[1]) & (slip <= STACK_SLIP)
                if hit.any():
                    stacked = True
                    best = np.flatnonzero(hit)[slip[hit].argmin()]
                    if slip[best] < slip_best:
                        slip_best, to_inv = float(slip[best]), bool(inverse[best])
        out.update(stacked=float(stacked), stack_to_inverse=float(to_inv) if stacked else np.nan,
                   stack_slip=slip_best if stacked else np.nan)
    return out


@torch.no_grad()
def motifs_of(run, rows, latents, templates):
    """Motif readings [n crystals] for latents built with the molecules of `rows` (one row
    per crystal; `templates[i]` is the template of rows' i-th molecule)."""
    ef = run.energy_function
    mb = rows.to(run.device)
    mb.orient_molecule(mode='standard')
    T = run.temperature * torch.ones(mb.num_graphs, dtype=torch.float32, device=run.device)
    mb, _, _, _ = ef.condition_samples(mb, temperature=T)
    crystals = ef.instantiate_crystals(latents.to(run.device).float(), mb)
    cl = crystals.mol2cluster(cutoff=6, supercell_size=10, std_orientation=False)
    near = cl.aux_ind <= 1
    pos, batch, mol = cl.pos[near].cpu().numpy().astype(np.float64), cl.batch[near].cpu().numpy(), cl.mol_ind[near].cpu().numpy()
    n_atoms = rows.num_atoms.cpu().numpy()
    bounds = np.searchsorted(batch, np.arange(mb.num_graphs + 1))
    out = {k: np.full(mb.num_graphs, np.nan) for k in KEYS}
    for g in range(mb.num_graphs):
        p, mi = pos[bounds[g]:bounds[g + 1]], mol[bounds[g]:bounds[g + 1]]
        n = int(n_atoms[g])
        assert len(p) % n == 0 and (mi[:n] == 0).all(), 'cluster is not whole molecules with the central one first'
        blocks = p.reshape(-1, n, 3)
        r = describe(templates[g], blocks[0], blocks[1:])
        for k in KEYS:
            out[k][g] = r[k]
    return out


def run_all(run, atlas, args):
    ident = atlas['identifier']
    use = np.flatnonzero((np.array(atlas['split']) == 'held_out') | atlas['is_random_train'])
    rows_of = {}
    for batch in (run.conditions, run.test_conditions):
        for i, name in enumerate(batch.identifier):
            rows_of[name] = (batch, i)
    refs = {}
    for path in (run.config['prior_path'], args.test_prior):
        data = torch.load(path, map_location='cpu', weights_only=False)
        b = data.get('equalized_prior', data.get('prior')) if isinstance(data, dict) else data
        wanted = {ident[m] for m in use}
        idx = [i for i, name in enumerate(b.identifier) if name in wanted]
        sub = b.subsample_new_batch(torch.tensor(idx))
        lat = sub.latent_params().detach().cpu()
        e = (sub.elj.double().flatten() / sub.z_prime.double().flatten()).numpy()
        for j, name in enumerate(sub.identifier):
            refs.setdefault(name, []).append((lat[j], float(e[j])))
        del data, b, sub
    n_draw = min(args.draws, atlas['draws']['latent'].shape[1])
    for start in range(0, len(use), args.chunk_molecules):
        path = os.path.join(args.out, 'motif_chunks', f'chunk_{start:06d}.pt')
        if os.path.exists(path):
            continue
        part = use[start:start + args.chunk_molecules]
        by_split = {}
        for m in part:
            by_split.setdefault(id(rows_of[ident[m]][0]), []).append(m)
        res = {'atlas_index': part, 'draws': {}, 'refs': {}, 'ref_energy': {}, 'template': {}}
        for group in by_split.values():
            batch = rows_of[ident[group[0]]][0]
            tpl = {}
            for m in group:
                i = rows_of[ident[m]][1]
                sl = slice(int(batch.ptr[i]), int(batch.ptr[i + 1]))
                tpl[m] = template(batch.z[sl].numpy(), batch.pos[sl].numpy())
                res['template'][int(m)] = (len(tpl[m]['donors']), len(tpl[m]['acceptors']), len(tpl[m]['rings']))
            rows_idx = torch.tensor([rows_of[ident[m]][1] for m in group])
            # the model's draws
            out = motifs_of(run, batch.subsample_new_batch(rows_idx.repeat_interleave(n_draw)),
                            torch.as_tensor(atlas['draws']['latent'][group][:, :n_draw].reshape(-1, 12)),
                            [tpl[m] for m in group for _ in range(n_draw)])
            for j, m in enumerate(group):
                res['draws'][int(m)] = {k: v[j * n_draw:(j + 1) * n_draw] for k, v in out.items()}
            # the search minima
            counts = [len(refs.get(ident[m], [])) for m in group]
            if sum(counts):
                lat = torch.stack([l for m in group for l, _ in refs.get(ident[m], [])])
                out = motifs_of(run, batch.subsample_new_batch(rows_idx.repeat_interleave(torch.tensor(counts))), lat,
                                [tpl[m] for m, c in zip(group, counts) for _ in range(c)])
                at = 0
                for m, c in zip(group, counts):
                    res['refs'][int(m)] = {k: v[at:at + c] for k, v in out.items()}
                    res['ref_energy'][int(m)] = np.array([e for _, e in refs.get(ident[m], [])])
                    at += c
        torch.save(res, path + '.tmp')
        os.replace(path + '.tmp', path)
        print(f'molecules {start}-{start + len(part) - 1} of {len(use)} done', flush=True)


def figures(atlas_path, motif_path, fig_dir):
    """The motif figures; reads the atlas and the motif file, loads no model."""
    from scipy.stats import spearmanr

    from eval.cond_panel import figstyle as fs
    A = torch.load(atlas_path, weights_only=False)
    M = torch.load(motif_path, weights_only=False)
    F = fs.Figures(fig_dir)
    idx = np.array(sorted(M['draws']))
    split = np.array(A['split'])[idx]
    n_draw = len(next(iter(M['draws'].values()))['coordination'])
    T = float(A['temperature'])
    excess = (A['draws']['seed_energy'].astype(np.float64) - A['e_ref'][:, None]) / T
    has_ring = np.array([M['template'][int(i)][2] > 0 for i in idx])
    has_donor = np.array([M['template'][int(i)][0] > 0 and M['template'][int(i)][1] > 0 for i in idx])
    everyone = np.ones(len(idx), bool)
    names = {'train': 'training', 'held_out': 'held-out'}

    def per_mol(src, key, which='mean'):
        """One value per molecule: the mean over its crystals, or the best minimum's."""
        out = np.full(len(idx), np.nan)
        for j, i in enumerate(idx):
            r = M[src].get(int(i))
            if r is None:
                continue
            v = r[key]
            if which == 'best':
                out[j] = v[int(np.argmin(M['ref_energy'][int(i)]))]
            elif np.isfinite(v).any():
                out[j] = np.nanmean(v)
        return out

    # ---- prevalence: model draws against search minima
    readings = (('coordination', 'molecules in contact', everyone),
                ('closest_gap', 'closest contact minus\nvan der Waals sum (A)', everyone),
                ('closest_is_inverse', 'closest neighbour is\nthe inverse (share)', everyone),
                ('n_donor_contacts', 'donor contacts\nper crystal', has_donor),
                ('stacked', 'ring stack present\n(share)', has_ring))
    fig, axes = fs.panels(len(readings), ncols=5, width=2.9, height=3.4)
    header = ['reading', 'molecules', 'model draws, training', 'model draws, held-out', 'all search minima', 'best search minimum']
    rows, points = [], []
    for ax, (key, label, sel) in zip(axes, readings):
        fs.style(ax)
        drawn, minima, best = per_mol('draws', key), per_mol('refs', key), per_mol('refs', key, 'best')
        vals = [np.nanmean(drawn[sel & (split == 'train')]), np.nanmean(drawn[sel & (split == 'held_out')]),
                np.nanmean(minima[sel]), np.nanmean(best[sel])]
        ax.bar(np.arange(4), vals, width=0.7, color=[fs.TRAIN, fs.HELD, fs.THIRD, fs.MUTED])
        ax.axhline(0, color=fs.AXIS, linewidth=0.8)
        ax.set_xticks(np.arange(4), ['model\ntraining', 'model\nheld-out', 'search\nminima', 'best\nminimum'], fontsize=7.5)
        ax.set_title(label, fontsize=9)
        flat = label.replace('\n', ' ')
        rows.append([flat, int(sel.sum())] + [round(float(v), 3) for v in vals])
        points.append(f'{flat}: model {vals[0]:.2f} (training) and {vals[1]:.2f} (held-out) against {vals[2]:.2f} in the '
                      f'search minima and {vals[3]:.2f} in the best one')
    F.save(fig, 'motif_prevalence', 'Packing motifs in the model\'s crystals and in the search minima',
           f'Mean over molecules of each reading, for {n_draw} model draws per molecule and for the molecule\'s search minima '
           f'(all of them, and the lowest one). {len(idx)} molecules; donor contacts are read on the {int(has_donor.sum())} '
           f'molecules with an N-H or O-H and an acceptor, ring stacks on the {int(has_ring.sum())} with a planar unsaturated ring. '
           f'A donor contact is a hydrogen-bond geometry; ELJ has no term that rewards it.', (header, rows), points)

    # ---- does a molecule's motif in the model follow its motif in the search minima?
    specs = (('stacked', 'share of crystals with a ring stack', has_ring),
             ('closest_is_inverse', 'share whose closest neighbour\nis the inverse', everyone),
             ('n_donor_contacts', 'donor contacts per crystal', has_donor), ('coordination', 'molecules in contact', everyone))
    fig, axes = fs.panels(len(specs), ncols=4, width=3.5, height=3.4)
    header, rows, points = ['reading', 'split', 'molecules', 'Spearman rho, model against search minima'], [], []
    for ax, (key, label, sel) in zip(axes, specs):
        fs.style(ax, grid='both')
        x, y = per_mol('refs', key), per_mol('draws', key)
        flat = label.replace('\n', ' ')
        for name, colour in (('train', fs.TRAIN), ('held_out', fs.HELD)):
            ok = sel & (split == name) & np.isfinite(x) & np.isfinite(y)
            edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, 7)))
            edges[-1] += 1e-9
            mid, mean, err = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                b = ok & (x >= lo) & (x < hi)
                if b.sum() >= 10:
                    mid.append(x[b].mean()), mean.append(y[b].mean()), err.append(y[b].std() / np.sqrt(b.sum()))
            ax.errorbar(mid, mean, yerr=err, color=colour, marker='o', capsize=2, label=names[name])
            rho = spearmanr(x[ok], y[ok])[0]
            rows.append([flat, names[name], int(ok.sum()), round(float(rho), 3)])
            points.append(f'{flat}, {names[name]}: rho {rho:.2f} over {int(ok.sum())} molecules')
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color=fs.AXIS, linewidth=1)
        ax.set_xlabel('in the molecule\'s search minima')
        ax.set_title(label, fontsize=9)
    axes[0].set_ylabel('in the model\'s draws')
    axes[0].legend(loc='upper left')
    F.save(fig, 'motif_agreement', 'Does a molecule get the motif its own search minima show?',
           f'One value per molecule on each axis: the mean of the reading over its search minima (x) and over {n_draw} model '
           f'draws (y); points are means over sixths of the x values, one standard error; the line is equality. Rank correlations '
           f'over molecules are in the table.', (header, rows), points)

    # ---- motif against energy, within molecule
    fig, axes = fs.panels(3, ncols=3, width=4.0, height=3.3)
    header, rows, points = ['comparison', 'split', 'n', 'mean excess or difference (kT)', 'SE'], [], []
    ax = axes[0]
    fs.style(ax)
    for name, colour in (('train', fs.TRAIN), ('held_out', fs.HELD)):
        sel = idx[split == name]
        coord = np.concatenate([M['draws'][int(i)]['coordination'] for i in sel])
        ex = np.concatenate([excess[i, :n_draw] for i in sel])
        ks = np.arange(4, 18)
        mean = np.array([ex[coord == k].mean() if (coord == k).sum() >= 50 else np.nan for k in ks])
        ax.plot(ks, mean, color=colour, marker='o', label=names[name])
        rows += [[f'draws with {k} molecules in contact', names[name], int((coord == k).sum()), round(float(v), 2), '']
                 for k, v in zip(ks, mean) if np.isfinite(v)]
        lo, hi = ks[np.nanargmax(mean)], ks[np.nanargmin(mean)]
        points.append(f'{names[name]}: mean excess {np.nanmax(mean):.1f} kT at {lo} molecules in contact, {np.nanmin(mean):.1f} kT at {hi}')
    ax.set_xlabel('molecules in contact in the draw')
    ax.set_ylabel('mean excess of those draws (kT)')
    ax.legend(loc='upper right')
    for ax, (key, label, sel) in zip(axes[1:], (('stacked', 'a ring stack', has_ring), ('n_donor_contacts', 'a donor contact', has_donor))):
        fs.style(ax)
        for name, colour in (('train', fs.TRAIN), ('held_out', fs.HELD)):
            diffs = []
            for j, i in enumerate(idx):
                if not sel[j] or split[j] != name:
                    continue
                flag = M['draws'][int(i)][key] > 0
                if 2 <= flag.sum() <= n_draw - 2:
                    diffs.append(excess[i, :n_draw][flag].mean() - excess[i, :n_draw][~flag].mean())
            diffs = np.array(diffs)
            ax.hist(diffs, bins=np.linspace(-40, 40, 41), histtype='step', color=colour, linewidth=1.6, density=True)
            ax.axvline(diffs.mean(), color=colour, linewidth=1)
            se = diffs.std() / np.sqrt(len(diffs))
            rows.append([f'draws with {label} minus draws without, same molecule', names[name], len(diffs), round(float(diffs.mean()), 2), round(float(se), 2)])
            points.append(f'{label}, {names[name]}: draws with it sit {diffs.mean():+.1f} kT (SE {se:.1f}) from draws without, within a molecule')
        ax.axvline(0, color=fs.AXIS, linewidth=1)
        ax.set_xlabel(f'excess of draws with {label} minus without (kT)')
        ax.set_yticks([])
    F.save(fig, 'motif_energy', 'How a draw\'s contacts go with its energy',
           f'Left: mean excess of the model\'s draws by the number of molecules the central one touches, pooled over molecules '
           f'(counts of at least 50 draws). Middle and right: per molecule, the mean excess of its draws that show the motif minus '
           f'that of its draws that do not (molecules with at least two draws of each kind); vertical lines are the means.',
           (header, rows), points)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--atlas', required=True)
    ap.add_argument('--test-prior', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=16, help='draws per molecule read (the first of the atlas)')
    ap.add_argument('--chunk-molecules', type=int, default=16)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--figures', default=None, help='figure directory: draw the motif figures from the motif file in --out and stop')
    args = ap.parse_args()
    if args.figures:
        figures(args.atlas, glob.glob(os.path.join(args.out, 'motifs_step*.pt'))[0], args.figures)
        return
    assert not torch.cuda.is_available(), 'CPU only: set CUDA_VISIBLE_DEVICES=-1'
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(os.path.join(args.out, 'motif_chunks'), exist_ok=True)
    atlas = torch.load(args.atlas, weights_only=False)
    run = load_run(args.checkpoint, args.config, device='cpu')
    run_all(run, atlas, args)
    merged = {'draws': {}, 'refs': {}, 'ref_energy': {}, 'template': {}}
    for f in sorted(glob.glob(os.path.join(args.out, 'motif_chunks', 'chunk_*.pt'))):
        c = torch.load(f, weights_only=False)
        for k in merged:
            merged[k].update(c[k])
    merged['keys'], merged['step'] = KEYS, run.step
    torch.save(merged, os.path.join(args.out, f'motifs_step{run.step}.pt'))
    print(f'motifs written for {len(merged["draws"])} molecules', flush=True)


if __name__ == '__main__':
    main()
