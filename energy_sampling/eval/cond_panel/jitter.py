"""
How smooth is the model in the geometry it is conditioned on? A noise ladder.

Each molecule's stored geometry is perturbed by Gaussian noise of a few sizes (angstrom per
coordinate), standardised to the trainer's frame as any new molecule would be, embedded by
the frozen encoder among stored molecules in a batch of the builder's size, and drawn from.
Against the stored condition it records how far the encoder's embedding and the
conditioner's output move, and the excess of the draws.

THE FRAME. The trainer re-standardises the molecule at every draw, so a condition is only
consistent when its embedding was taken in that same frame; the perturbed geometry is
therefore standardised before it is embedded. For a molecule whose principal axes are nearly
degenerate, or whose axis signs hang on a small asymmetry, that frame can turn over under
the smallest perturbation, which changes the condition wholesale. Each level records whether
it did (the RMSD to the stored geometry as both sit in the frame, against their RMSD after
the best rotation), and the figure reads the levels on the molecules whose frame was kept.

The contrast is between training and held-out molecules. A held-out molecule's stored
condition was never trained on, so a perturbed copy of it is just another unseen condition.
A training molecule's stored condition WAS trained on: if a perturbation far below any
physical difference between two molecules costs it sample quality, what the model learned
for that molecule is tied to the exact embedding and not to the molecule.

    cd energy_sampling
    CUDA_VISIBLE_DEVICES=-1 python -m eval.cond_panel.jitter --checkpoint ... --config ... \
        --atlas <dir>/atlas/atlas_step50000.pt --out <dir>/jitter
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from eval.cond_panel.conformers import DEFAULT_ENCODER, encode, frame_rmsds, frame_state, molecule_items, rows_from
from eval.cond_panel.pairs import seed_excess
from eval.cond_panel.sampler import draw, load_run, reasonable_mask
from eval.cond_panel.select_panel import conditioner_coords


def figures(path, atlas_path, fig_dir):
    from eval.cond_panel import figstyle as fs
    J = torch.load(path, weights_only=False)
    A = torch.load(atlas_path, weights_only=False)
    F = fs.Figures(fig_dir)
    levels = J['levels']
    labels = ['stored'] + ['0\n(re-embedded)' if s == 0 else f'{s:g}' for s in levels]
    nn_dist = float(np.median(A['pair_dist_train']))
    series = (('train', 'training', fs.TRAIN), ('held_out', 'held-out', fs.HELD))
    quantities = (('enc_rel', 'encoder embedding: change relative to its size'), ('cond_dist', 'conditioner distance from the stored condition'),
                  ('excess', 'median excess of the draws (kT)'), ('reasonable', 'bound and sensibly packed (share of draws)'))
    fig, axes = fs.panels(4, ncols=4, width=3.7, height=3.5)
    header, rows, points = ['quantity', 'split', 'molecules'] + [l.replace('\n', ' ') + (' A' if l[0].isdigit() and '(' not in l else '') for l in labels], [], []
    for ax, (key, label) in zip(axes, quantities):
        fs.style(ax)
        for split, name, colour in series:
            v = np.array([r[key] for r in J['results'] if r['split'] == split], dtype=np.float64)   # [molecules, 1 + levels]
            kept = np.array([r['turned'] for r in J['results'] if r['split'] == split]) == 0
            v = np.where(kept, v, np.nan)
            med = np.nanmedian(v, 0)
            q = np.nanquantile(v, [0.25, 0.75], axis=0)
            ax.errorbar(np.arange(v.shape[1]), med, yerr=[med - q[0], q[1] - med], color=colour, marker='o', capsize=2, label=name)
            rows.append([label, name, len(v)] + [round(float(x), 3) for x in med])
            if key == 'excess':
                points.append(f'{name}: median excess {med[0]:.1f} kT on the stored condition, {med[1]:.1f} re-embedded, '
                              + ', '.join(f'{m:.1f} at {s:g} A' for s, m in zip(levels[1:], med[2:])))
            if key == 'cond_dist':
                points.append(f'{name}: conditioner distance ' + ', '.join(f'{m:.2f} at {s:g} A' for s, m in zip(levels, med[1:]))
                              + f' (nearest training molecule: {nn_dist:.2f})')
        if key == 'cond_dist':
            ax.axhline(nn_dist, color=fs.AXIS, linewidth=1)
            ax.text(len(labels) - 1, nn_dist, 'nearest training molecule ', ha='right', va='bottom', fontsize=7.5, color=fs.INK2)
        ax.set_xticks(np.arange(len(labels)), labels, fontsize=7.5)
        ax.set_xlabel('noise on each coordinate (A)')
        ax.set_title(label, fontsize=9)
    axes[0].legend(loc='upper left')
    flips = np.array([r['turned'] for r in J['results']], dtype=np.float64)
    points.append('share of molecules whose frame turned over on standardising the perturbed geometry: '
                  + ', '.join(f'{f:.0%} at {s:g} A' for s, f in zip(levels[1:], flips.mean(0)[2:])))
    rows.append(['share whose frame turned over', 'both', len(flips)] + [round(float(x), 3) for x in flips.mean(0)])
    for split, name, _ in series:
        ex = np.array([r['excess'] for r in J['results'] if r['split'] == split], dtype=np.float64)
        turned = np.array([r['turned'] for r in J['results'] if r['split'] == split]) > 0
        if turned[:, 2:].any():
            points.append(f'{name}, frame turned over ({int(turned[:, 2:].sum())} molecule-levels): median excess '
                          f'{np.median(ex[:, 2:][turned[:, 2:]]):.1f} kT against {np.median(ex[:, 0]):.1f} on the stored condition')
    F.save(fig, 'jitter_ladder', 'A molecule moved by a fraction of an angstrom, as a condition',
           f'{sum(r["split"] == "train" for r in J["results"])} training and {sum(r["split"] == "held_out" for r in J["results"])} held-out '
           f'molecules. Each stored geometry is perturbed by Gaussian noise of the given size per coordinate, standardised to the trainer\'s frame, embedded '
           f'by the frozen encoder in a batch of {J["embed_batch"]} molecules, and drawn from {J["draws"]} times. "stored" is the condition in the '
           f'conditions file; "0 (re-embedded)" is the unperturbed geometry embedded again. Points are medians over the molecules whose frame was kept at that level, bars the '
           f'quartiles. Excess is kT above the molecule\'s best search minimum.', (header, rows), points)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--atlas', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--encoder', default=DEFAULT_ENCODER)
    ap.add_argument('--n-each', type=int, default=40, help='training molecules, and held-out molecules')
    ap.add_argument('--levels', type=float, nargs='+', default=[0.0, 0.0005, 0.002, 0.005, 0.02, 0.05])
    ap.add_argument('--embed-batch', type=int, default=200)
    ap.add_argument('--draws', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--figures', default=None, help='figure directory: draw the figure from the result file in --out and stop')
    args = ap.parse_args()
    if args.figures:
        figures(glob.glob(os.path.join(args.out, 'jitter_step*.pt'))[0], args.atlas, args.figures)
        return
    assert not torch.cuda.is_available(), 'CPU only: set CUDA_VISIBLE_DEVICES=-1'
    assert args.levels[0] == 0.0, 'the first level is the unperturbed geometry re-embedded: the gate'
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    from mxtaltools.common.training_utils import load_molecule_autoencoder

    atlas = torch.load(args.atlas, weights_only=False)
    run = load_run(args.checkpoint, args.config, device='cpu')
    encoder = load_molecule_autoencoder(args.encoder, 'cpu')
    rng = np.random.default_rng(args.seed)
    n_new = len(args.levels)
    filler = molecule_items(run.conditions, rng.choice(run.conditions.num_graphs, args.embed_batch - n_new, replace=False).tolist())
    split = np.array(atlas['split'])
    picks = [('train', run.conditions, rng.choice(np.flatnonzero(atlas['is_random_train']), args.n_each, replace=False)),
             ('held_out', run.test_conditions, rng.choice(np.flatnonzero(split == 'held_out'), args.n_each, replace=False))]
    results = []
    for name, batch, chosen in picks:
        row_of = {ident: i for i, ident in enumerate(batch.identifier)}
        for m in chosen:
            ident = atlas['identifier'][m]
            i = row_of[ident]
            sl = slice(int(batch.ptr[i]), int(batch.ptr[i + 1]))
            stored = batch.pos[sl].numpy().astype(np.float64)
            geoms = [stored] + [stored + rng.normal(0.0, s, stored.shape) if s > 0 else stored for s in args.levels]
            rows = rows_from(batch, i, geoms)
            if rows is None:
                continue
            heavy = batch.z[sl].numpy() > 1
            framed = [rows.pos[int(rows.ptr[k]):int(rows.ptr[k + 1])].numpy().astype(np.float64) for k in range(len(geoms))]
            in_frame, proper, either = zip(*[frame_rmsds(f[heavy], framed[0][heavy]) for f in framed])
            state = frame_state(in_frame, proper, either)
            stored_emb = batch.embedding[i]
            emb = encode(encoder, molecule_items(batch, [i] * n_new, framed[1:]) + filler)[:n_new]
            rel = [0.0] + [float((e - stored_emb).norm() / stored_emb.norm()) for e in emb]
            # the gate on this module's embedding path. Its bound is the encoder's own batch dependence: 800 stored molecules
            # re-embedded in batches of 200 sat a median 0.6-0.9% from their stored embeddings and at most 4.4% (2026-10-05)
            assert rel[1] < 0.06, f'{ident}: the unperturbed geometry re-embedded is {rel[1]:.3f} (relative) from its stored embedding'
            rows.add_graph_attr(torch.cat([stored_emb[None], emb]), 'embedding')
            rows.add_graph_attr(torch.full((rows.num_graphs,), run.registry[ident], dtype=torch.long), 'mol_id')
            cond = conditioner_coords(run, rows).numpy()
            d = draw(run, rows.subsample_new_batch(torch.arange(rows.num_graphs).repeat_interleave(args.draws)),
                     batch_size=rows.num_graphs * args.draws, seed=args.seed + len(results))
            ex = seed_excess(run, d['sample_batch'], d['condition_id']).reshape(rows.num_graphs, args.draws)
            good = reasonable_mask(run, d['sample_batch']).float().reshape(rows.num_graphs, args.draws)
            results.append({'identifier': ident, 'split': name, 'enc_rel': rel, 'cond_dist': np.linalg.norm(cond - cond[0], axis=1).tolist(),
                            'excess': np.median(ex, 1).tolist(), 'excess_mean': ex.mean(1).tolist(), 'reasonable': good.mean(1).tolist(),
                            'turned': [float(v != 'kept') for v in state], 'frame_state': state.tolist(), 'frame_rmsd': list(in_frame),
                            'shape_rmsd': list(either)})
            r = results[-1]
            print(f'{len(results):3d} {name:8s} {ident:22s} frame ' + ''.join(v[0] for v in r['frame_state']) + ' | conditioner '
                  + ' '.join(f'{v:5.2f}' for v in r['cond_dist'])
                  + ' | excess ' + ' '.join(f'{v:6.1f}' for v in r['excess']), flush=True)
            torch.save({'results': results, 'levels': args.levels, 'draws': args.draws, 'embed_batch': args.embed_batch, 'step': run.step},
                       os.path.join(args.out, f'jitter_step{run.step}.pt'))
    print(f'{len(results)} molecules written', flush=True)


if __name__ == '__main__':
    main()
