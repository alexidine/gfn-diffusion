"""
Pairs of molecules, drawn deeply: how different are the model's crystal distributions for two
molecules, measured three ways, against how far apart the molecules are in condition space.

For a pair (A, B), with `--draws` draws each:

  DISTRIBUTION DISTANCE. Energy distance between the two sets of draws on the folded latents
  (eval/cond_panel/latents.py), with the floor given by two halves of one molecule's draws.

  SWAP PENALTY, in energy. A's crystals are rebuilt with molecule B in them (the same
  latents: the cell lengths are in units of the molecule's radius, so the cell rescales
  with B) and scored against B's best search minimum. The penalty is that excess minus the
  excess of B's own draws, in kT. A model that used one packing for everything would show
  no penalty.

  DENSITY LIFT, no energies. eval/xcond.py::cross_condition_delta on the two molecules: the
  model's own log-density of A's crystals under A minus under B, in nats.

Pair sets: a molecule with its nearest training neighbour (training and held-out molecules,
taken from the atlas), random pairs, and isosteric pairs (one heavy-atom skeleton, different
elements, both rigid and of equal extent: under ELJ their landscapes should nearly coincide,
so a large penalty there would mean the model reads the label rather than the shape).

Resumable: results are written a block of pairs at a time under <out>/pair_chunks.

    cd energy_sampling
    CUDA_VISIBLE_DEVICES=-1 python -m eval.cond_panel.pairs --checkpoint ... --config ... \
        --atlas <dir>/atlas/atlas_step50000.pt --descriptors <dir>/panel/all_descriptors.csv --out <dir>/pairs
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from eval.cond_panel.latents import energy_distance, fold, w1_per_dim
from eval.cond_panel.sampler import SAMPLE_FIELDS, draw, load_run
from eval.xcond import cross_condition_delta
from utils import uniform_discretizer


def isosteric_pairs(csv_path, n, rng):
    """`n` training pairs sharing one heavy-atom skeleton with different elements, by the
    rule select_panel.py uses for its control pair."""
    import pandas as pd
    d = pd.read_csv(csv_path)
    d = d[(d.split == 'train') & (d.n_rot == 0)]
    out = []
    groups = [g for _, g in d.groupby('generic_key') if len(g) > 1]
    for gi in rng.permutation(len(groups)):
        g = groups[gi]
        rows = g.to_dict('records')
        found = None
        for i in rng.permutation(len(rows))[:12]:
            for j in rng.permutation(len(rows))[:12]:
                a, b = rows[i], rows[j]
                ext = max(abs(a[k] - b[k]) / max(a[k], b[k], 1e-6) for k in ('extent1', 'extent2', 'extent3'))
                if i < j and abs(a['n_NO'] - b['n_NO']) >= 2 and abs(a['n_H'] - b['n_H']) <= 2 and ext <= 0.10:
                    found = (a['identifier'], b['identifier'])
                    break
            if found:
                break
        if found:
            out.append(found)
        if len(out) == n:
            break
    return out


def choose_pairs(atlas, csv_path, n_each, n_iso, seed):
    """[(kind, identifier A, identifier B)], seeded. Neighbour pairs are spread evenly over
    the range of neighbour distances rather than drawn at the typical one."""
    rng = np.random.default_rng(seed)
    ident = atlas['identifier']
    out = []
    for kind, pairs, dist in (('training, nearest training', atlas['pairs_train'], atlas['pair_dist_train']),
                              ('held-out, nearest training', atlas['pairs_held'], atlas['pair_dist_held'])):
        order = np.argsort(dist)
        pick = order[np.linspace(0, len(order) - 1, n_each).round().astype(int)]
        out += [(kind, ident[pairs[p, 0]], ident[pairs[p, 1]]) for p in pick]
    pool = np.flatnonzero(np.array(atlas['split']) == 'train')
    a, b = rng.choice(pool, n_each, replace=False), rng.choice(pool, n_each, replace=False)
    out += [('random training pair', ident[i], ident[j]) for i, j in zip(a, b) if i != j]
    if csv_path and n_iso:
        out += [('isosteric training pair', x, y) for x, y in isosteric_pairs(csv_path, n_iso, rng)]
    return out


def seed_excess(run, batch, cid):
    """kT above the best search minimum of the molecule the crystals were built with."""
    ef = run.energy_function
    fields = {k: getattr(batch, k).detach().flatten().cpu() for k in SAMPLE_FIELDS
              if torch.is_tensor(getattr(batch, k, None)) and getattr(batch, k).numel() == batch.num_graphs}
    return ((ef.seed_energy_from(fields) - ef.energy_reference_for(cid.cpu())) / run.temperature).double().numpy()


@torch.no_grad()
def score_under(run, rows, terminals):
    """The excess (kT) of `terminals` rebuilt with the molecules of `rows`, one row each."""
    ef = run.energy_function
    mol_batch = rows.to(run.device)
    mol_batch.orient_molecule(mode='standard')
    T = run.temperature * torch.ones(mol_batch.num_graphs, dtype=torch.float32, device=run.device)
    mol_batch, log_T, _, cid = ef.condition_samples(mol_batch, temperature=T)
    _, built = ef.log_reward(terminals.to(run.device), mol_batch=mol_batch, log_temperature=log_T, return_exp=True)
    return seed_excess(run, built.cpu().detach(), cid)


@torch.no_grad()
def density_lift(run, rows2, terminals, n_per, k):
    """Mean log-density of each molecule's crystals under its own condition minus under the
    other's (nats), and its standard error over crystals. `terminals` is condition-major."""
    ef = run.energy_function
    mol_batch = rows2.to(run.device)
    mol_batch.orient_molecule(mode='standard')
    T = run.temperature * torch.ones(2, dtype=torch.float32, device=run.device)
    mol_batch, _, condition, _ = ef.condition_samples(mol_batch, temperature=T)
    disc = lambda bsz: uniform_discretizer(bsz, run.eval_T)
    delta, _, _ = cross_condition_delta(run.gfn, mol_batch, condition.to(run.device), terminals.to(run.device), disc, k)
    delta = delta.cpu().numpy()
    own = np.repeat([0, 1], n_per)
    lift = delta[np.arange(2 * n_per), own] - delta[np.arange(2 * n_per), 1 - own]
    return lift[:n_per].mean(), lift[n_per:].mean(), lift.std() / np.sqrt(len(lift))


def run_pairs(run, pairs, args, out_dir):
    rows_of = {}
    for batch in (run.conditions, run.test_conditions):
        for i, ident in enumerate(batch.identifier):
            rows_of[ident] = (batch, i)
    cond_of = {i: c for i, c in zip(args.atlas_ident, args.atlas_cond)}
    for start in range(0, len(pairs), args.block):
        path = os.path.join(out_dir, 'pair_chunks', f'block_{start:05d}.pt')
        if os.path.exists(path):
            continue
        block = pairs[start:start + args.block]
        mols = sorted({m for _, a, b in block for m in (a, b)})
        draws = {}
        for m in mols:  # one molecule at a time keeps the row bookkeeping trivial; draw() batches inside
            batch, i = rows_of[m]
            d = draw(run, batch.subsample_new_batch(torch.full((args.draws,), i, dtype=torch.long)),
                     batch_size=args.draws, seed=args.seed + start + len(draws))
            draws[m] = {'terminal': d['terminal_raw'], 'latent': fold(d['sample_batch'].latent_params().detach().cpu().numpy().astype(np.float64)),
                        'excess': seed_excess(run, d['sample_batch'], d['condition_id']),
                        'jf': float(d['log_w'].double().mean())}
        results = []
        for kind, a, b in block:
            da, db = draws[a], draws[b]
            n, half = args.draws, args.draws // 2
            r = {'kind': kind, 'a': a, 'b': b,
                 'energy_distance': energy_distance(da['latent'], db['latent']),
                 'floor': 0.5 * (energy_distance(da['latent'][:half], da['latent'][half:]) + energy_distance(db['latent'][:half], db['latent'][half:])),
                 'w1': w1_per_dim(da['latent'], db['latent']),
                 'excess_a': float(np.median(da['excess'])), 'excess_b': float(np.median(db['excess'])),
                 'jf_a': da['jf'], 'jf_b': db['jf']}
            for own, other, tag in ((a, b, 'a_in_b'), (b, a, 'b_in_a')):
                batch, i = rows_of[other]   # the OTHER molecule, with the first one's crystals
                ex = score_under(run, batch.subsample_new_batch(torch.full((n,), i, dtype=torch.long)), draws[own]['terminal'])
                r[f'swap_{tag}'] = float(np.median(ex))
            r['swap_penalty'] = 0.5 * ((r['swap_a_in_b'] - r['excess_b']) + (r['swap_b_in_a'] - r['excess_a']))
            (ba, ia), (bb, ib) = rows_of[a], rows_of[b]
            two = ba.subsample_new_batch(torch.tensor([ia])).append_batch(bb.subsample_new_batch(torch.tensor([ib]))) if ba is not bb \
                else ba.subsample_new_batch(torch.tensor([ia, ib]))
            term = torch.cat([da['terminal'][:args.lift_n], db['terminal'][:args.lift_n]])
            r['lift_a'], r['lift_b'], r['lift_se'] = density_lift(run, two, term, args.lift_n, args.lift_k)
            r['lift'] = 0.5 * (r['lift_a'] + r['lift_b'])
            if a in cond_of and b in cond_of:
                r['cond_dist'] = float(np.linalg.norm(cond_of[a] - cond_of[b]))
            results.append(r)
        torch.save(results, path + '.tmp')
        os.replace(path + '.tmp', path)
        print(f'pairs {start}-{start + len(block) - 1} done', flush=True)


def figures(pairs_path, fig_dir):
    """The pair figures; reads the pair file, loads no model."""
    from scipy.stats import spearmanr

    from eval.cond_panel import figstyle as fs
    from eval.cond_panel.latents import BLOCKS
    R = torch.load(pairs_path, weights_only=False)
    F = fs.Figures(fig_dir)
    kinds = list(dict.fromkeys(r['kind'] for r in R))
    colour = dict(zip(('training, nearest training', 'held-out, nearest training', 'random training pair', 'isosteric training pair'),
                      (fs.TRAIN, fs.HELD, fs.MUTED, fs.THIRD)))
    get = lambda key, kind=None: np.array([r[key] for r in R if kind is None or r['kind'] == kind], dtype=np.float64)
    dist = get('cond_dist')
    measures = (('swap_penalty', 'swap penalty (kT)', 'symlog'), ('lift', 'density lift (nats)', 'symlog'),
                ('energy_distance', 'energy distance between latent distributions', 'linear'))
    fig, axes = fs.panels(3, ncols=3, width=4.4, height=3.6)
    header = ['pair set', 'pairs', 'median conditioner distance', 'median swap penalty (kT)', 'median density lift (nats)',
              'median energy distance', 'median floor of the energy distance']
    rows, points = [], []
    for ax, (key, label, scale) in zip(axes, measures):
        fs.style(ax, grid='both')
        for kind in kinds:
            ax.scatter(get('cond_dist', kind), get(key, kind), s=14, color=colour[kind], alpha=0.75, linewidths=0, label=kind)
        if scale == 'symlog':
            ax.set_yscale('symlog', linthresh=10.0)
        ax.axhline(0, color=fs.AXIS, linewidth=1)
        ax.set_xlabel('distance between the two molecules (conditioner space)')
        ax.set_ylabel(label)
        ax.set_title(f'Spearman rho = {spearmanr(dist, get(key))[0]:.2f}')
    axes[0].legend(loc='upper left', fontsize=7)
    for kind in kinds:
        rows.append([kind, int((np.array([r['kind'] for r in R]) == kind).sum()), round(float(np.median(get('cond_dist', kind))), 2),
                     round(float(np.median(get('swap_penalty', kind))), 2), round(float(np.median(get('lift', kind))), 2),
                     round(float(np.median(get('energy_distance', kind))), 4), round(float(np.median(get('floor', kind))), 4)])
        points.append(f'{kind}: median swap penalty {rows[-1][3]:.1f} kT, density lift {rows[-1][4]:.1f} nats, at median distance {rows[-1][2]:.2f}')
    nn = np.array([r['kind'].endswith('nearest training') for r in R])
    near, far = nn & (dist < np.quantile(dist[nn], 1 / 3)), nn & (dist > np.quantile(dist[nn], 2 / 3))
    points.append(f'nearest-neighbour pairs, closest third (distance below {np.quantile(dist[nn], 1 / 3):.2f}): median swap penalty '
                  f'{np.median(get("swap_penalty")[near]):.1f} kT, lift {np.median(get("lift")[near]):.1f} nats; farthest third: '
                  f'{np.median(get("swap_penalty")[far]):.1f} kT, {np.median(get("lift")[far]):.1f} nats')
    F.save(fig, 'pairs_deep', 'How different two molecules\' crystal distributions are, against their distance in condition space',
           'One point per pair of molecules. Swap penalty: the excess (kT above the best search minimum) of one molecule\'s '
           'crystals rebuilt with the other molecule in them, minus the excess of that molecule\'s own draws, averaged over the two '
           'directions. Density lift: the model\'s own log-density of a molecule\'s crystals under its own condition minus under the '
           'other\'s. Energy distance: between the two sets of draws on the folded latents. The first two axes are linear within '
           '+-10 and logarithmic beyond. Isosteric pairs share a heavy-atom skeleton and differ in elements.', (header, rows), points)

    # which latents differ between the two molecules of a pair
    w1 = np.array([r['w1'] for r in R])
    names = list(BLOCKS)
    fig, ax = fs.plt.subplots(figsize=(7.4, 3.4))
    fs.style(ax)
    width = 0.8 / len(kinds)
    header, rows = ['pair set'] + [f'{b}: mean W1 per latent' for b in names], []
    for k, kind in enumerate(kinds):
        sel = np.array([r['kind'] == kind for r in R])
        vals = [w1[sel][:, list(BLOCKS[b])].mean() for b in names]
        ax.bar(np.arange(len(names)) + (k - (len(kinds) - 1) / 2) * width, vals, width=width * 0.92, color=colour[kind], label=kind)
        rows.append([kind] + [round(float(v), 4) for v in vals])
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel('mean 1-D Wasserstein distance per latent\n(fraction of the latent\'s range)')
    ax.legend(loc='upper left', fontsize=7)
    random_row, near_row = rows[kinds.index('random training pair')], rows[kinds.index('training, nearest training')]
    points = [f'{b}: {random_row[i + 1]:.3f} between random molecules, {near_row[i + 1]:.3f} between nearest neighbours'
              for i, b in enumerate(names)]
    F.save(fig, 'pairs_which_latents', 'Which latents differ between two molecules',
           'Mean over pairs of the 1-D Wasserstein distance between the two molecules\' draws, averaged over the latents of each '
           'block, in units of the latent\'s range (circular on v, w, phi and r; v and w folded onto one origin image). Two finite '
           'samples of one broad distribution already give a positive value, so read the bars against each other, not against zero.',
           (header, rows), points)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--atlas', default=None)
    ap.add_argument('--descriptors', default=None, help='select_panel all_descriptors.csv, for isosteric pairs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-each', type=int, default=60)
    ap.add_argument('--n-iso', type=int, default=40)
    ap.add_argument('--draws', type=int, default=128)
    ap.add_argument('--lift-n', type=int, default=24, help='crystals per molecule scored for the density lift')
    ap.add_argument('--lift-k', type=int, default=8, help='backward rollouts per cell (even)')
    ap.add_argument('--block', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--figures', default=None, help='figure directory: draw the pair figures from the pair file in --out and stop')
    args = ap.parse_args()
    if args.figures:
        figures(glob.glob(os.path.join(args.out, 'pairs_step*.pt'))[0], args.figures)
        return
    assert not torch.cuda.is_available(), 'CPU only: set CUDA_VISIBLE_DEVICES=-1'
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(os.path.join(args.out, 'pair_chunks'), exist_ok=True)

    atlas = torch.load(args.atlas, weights_only=False)
    plan_path = os.path.join(args.out, 'pairs_plan.pt')
    if os.path.exists(plan_path):
        pairs = torch.load(plan_path, weights_only=False)
    else:
        pairs = choose_pairs(atlas, args.descriptors, args.n_each, args.n_iso, args.seed)
        torch.save(pairs, plan_path)
    print(f'{len(pairs)} pairs: ' + ', '.join(f'{sum(k == kind for k, _, _ in pairs)} {kind}' for kind in dict.fromkeys(k for k, _, _ in pairs)), flush=True)

    run = load_run(args.checkpoint, args.config, device='cpu')
    args.atlas_ident, args.atlas_cond = atlas['identifier'], atlas['cond_emb'].astype(np.float64)
    # isosteric pairs come from outside the atlas: give them conditioner coordinates too
    missing = sorted({m for _, a, b in pairs for m in (a, b)} - set(args.atlas_ident))
    if missing:
        from eval.cond_panel.select_panel import conditioner_coords
        idx = {ident: i for i, ident in enumerate(run.conditions.identifier)}
        sub = run.conditions.subsample_new_batch(torch.tensor([idx[m] for m in missing]))
        args.atlas_ident = list(args.atlas_ident) + missing
        args.atlas_cond = np.concatenate([args.atlas_cond, conditioner_coords(run, sub).numpy()])
    run_pairs(run, pairs, args, args.out)

    results = [r for f in sorted(glob.glob(os.path.join(args.out, 'pair_chunks', 'block_*.pt'))) for r in torch.load(f, weights_only=False)]
    torch.save(results, os.path.join(args.out, f'pairs_step{run.step}.pt'))
    print(f'{len(results)} pairs written', flush=True)


if __name__ == '__main__':
    main()
