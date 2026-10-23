"""
Where do the model's draws go when they are relaxed?

A draw far above its molecule's best search minimum can be far above it for two different
reasons: it sits high on the wall of a good basin, or it sits in a poor basin. Relaxing it
tells the two apart. For a set of molecules this module relaxes three kinds of start with
ONE relaxer and compares where they end:

  model      draws from the checkpoint (eval/cond_panel/sampler.py::draw)
  random     starts from the crystal search's own initialiser (get_initial_state at the
             search config's settings): what a relaxation finds knowing nothing
  reference  the molecule's stored search minima: they should not move, which checks that
             this relaxer is the one that made them

THE RELAXER is the search's: the `opt` stages of the search config that built the prior,
read from its YAML and passed to MolCrystalOps.optimize_crystal_parameters unchanged, with
one exception. The search stops a batch when 95% of its rows have met `convergence_eps`,
which truncates the rest, so what a row returns depends on its batch mates. Here
`convergence_eps` is set to 0 and every row takes every step (`--search-stopping` restores
the search's rule). Under Rprop the rows are then independent, and `--check-batching`
verifies it by relaxing the same starts whole and in halves.

Energies are the relaxer's own eLJ, the currency of the stored minima. A molecule's FLOOR is
the lowest energy known for it: its stored reference, or anything lower that a relaxation
here reaches. Everything is reported in kT above that floor.

TWO CHECKS ON THE RELAXER ITSELF. `--check-converged` relaxes the kept ends of a run again
and prints the energy they lose: an end that is a minimum of the relaxer stays where it is.
`--stage-plan` runs chosen stages of the search config by index (`1 1` is its fine stage
twice), and `--figures <dir> --against <other run>` sets two such runs side by side, start
by start: whether a start's end depends on the relaxer that took it there.

    cd energy_sampling
    python -m eval.cond_panel.relax --checkpoint ... --config ... --test-prior D:/.../qm9full_test_prior.pt \
        --search-config <MXtalTools>/configs/crystal_searches/qm9_full_sep29/tasks/0.yaml --out <dir>/relax --keep-crystals
    python -m eval.cond_panel.relax --out <dir>/relax --figures <dir>/figures
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from eval.cond_panel.sampler import draw, load_run

KINDS = ('model', 'random', 'reference')
DISTINCT_TOL = 0.1  # raw energy units: two relaxed ends closer than this are counted as one minimum


def search_stages(path, fixed_steps, plan=None):
    """The `opt` stages of a search config as optimiser keyword sets, in the config's order
    or, with `plan`, the stages at those indices in that order; and the config."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    stages = [dict(cfg['opt'][i]) for i in (plan if plan else range(len(cfg['opt'])))]
    for s in stages:
        assert s['optimizer_func'] == 'rprop', 'row independence at fixed steps is a property of Rprop'
        s['show_tqdm'] = False
        if fixed_steps:
            s['convergence_eps'] = 0.0
    return stages, cfg


def relax(batch, stages, device, scratch, trajectory=False):
    """Run the stages on a crystal batch. Returns the relaxed crystals (CPU batch) and, per
    row, the energy and packing coefficient before the first step and at the end. With
    `trajectory`, the end dict also carries 'traj': the energy at every recorded step of
    every stage in order, [steps, rows] float32 (one step is one energy and force call a row)."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    b = batch.to(device)
    key = str(stages[-1]['optim_target'])
    start, traj = {}, []
    for stage in stages:
        out, rec = b.optimize_crystal_parameters(return_record=True, intermediates_path=scratch, **stage)
        assert len(out) == b.num_graphs, 'the optimiser dropped rows'
        if not start:
            start = {'e': rec[key][0].detach().cpu().double().flatten(), 'cp': rec['cp'][0].detach().cpu().double().flatten()}
        if trajectory:
            traj.append(rec[key].detach().cpu().float().reshape(rec[key].shape[0], -1))
        b = collate_data_list(out).to(device)
    b.box_analysis()
    b = b.cpu().detach()
    end = {'e': getattr(b, key).double().flatten(), 'cp': b.packing_coeff.double().flatten()}
    if trajectory:
        end['traj'] = torch.cat(traj).numpy()
    return b, start, end


def cell_rows(batch):
    return torch.cat([batch.cell_lengths.double(), batch.cell_angles.double()], dim=1).cpu().numpy()


def random_starts(rows, search_cfg, device, seed):
    """`rows` re-initialised by the search's initialiser at the search config's settings."""
    from mxtaltools.crystal_search.utils import get_initial_state
    ns = SimpleNamespace(init_sample_method=search_cfg['init_sample_method'], init_target_cp=search_cfg['init_target_cp'],
                         init_reduced=search_cfg['init_reduced'], opt_seed=seed)
    return get_initial_state(ns, rows.to(device), device, 0)


def reference_rows(prior, identifiers):
    """Each identifier's stored minima, as one crystal batch cut from a loaded prior batch,
    with the molecule index of each row."""
    want = {ident: m for m, ident in enumerate(identifiers)}
    idx = [i for i, ident in enumerate(prior.identifier) if ident in want]
    sub = prior.subsample_new_batch(torch.tensor(idx))
    owner = np.array([want[ident] for ident in sub.identifier])
    assert set(owner.tolist()) == set(range(len(identifiers))), 'a molecule has no stored minimum in this file'
    return sub, owner


def load_prior(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data.get('equalized_prior', data.get('prior')) if isinstance(data, dict) else data


def run(args):
    """Relax the three kinds of start for the chosen molecules, one chunk file per
    `--chunk-molecules` molecules; a chunk already on disk is skipped."""
    dev = args.device
    run_ = load_run(args.checkpoint, args.config, device=dev)
    stages, search_cfg = search_stages(args.search_config, not args.search_stopping, args.stage_plan)
    scratch = os.path.join(args.out, f'opt_intermediates_{os.getpid()}.pt')
    g = torch.Generator().manual_seed(args.seed)
    picks = [('train', run_.conditions, torch.randperm(run_.conditions.num_graphs, generator=g)[:args.n_train]),
             ('held_out', run_.test_conditions, torch.randperm(run_.test_conditions.num_graphs, generator=g)[:args.n_test])]
    meta = {'step': run_.step, 'temperature': run_.temperature, 'stages': stages, 'fixed_steps': not args.search_stopping,
            'search_config': os.path.basename(os.path.dirname(args.search_config)), 'checkpoint': os.path.basename(args.checkpoint),
            'draws': args.draws, 'randoms': args.randoms}
    t0 = time.time()
    priors = {'train': run_.config['prior_path'], 'held_out': args.test_prior}
    for split, batch, rows in picks:
        chunk_path = lambda start: os.path.join(args.out, 'relax_chunks', f'{split}_{start:05d}.pt')
        todo = [start for start in range(0, rows.numel(), args.chunk_molecules) if not os.path.exists(chunk_path(start))]
        if not todo:
            continue
        prior = load_prior(priors[split])
        for start in todo:
            path = chunk_path(start)
            sub = rows[start:start + args.chunk_molecules]
            ident = [batch.identifier[int(i)] for i in sub]
            res = {'identifier': ident, 'split': split, 'e_ref_table': run_.energy_function.energy_reference_for(
                torch.tensor([run_.registry[i] for i in ident])).double().numpy()}
            # model draws: the trainer's own built crystals
            d = draw(run_, batch.subsample_new_batch(sub.repeat_interleave(args.draws)), batch_size=sub.numel() * args.draws,
                     seed=args.seed + start)
            sets = {'model': (d['sample_batch'], np.repeat(np.arange(sub.numel()), args.draws)),
                    'random': (random_starts(batch.subsample_new_batch(sub.repeat_interleave(args.randoms)), search_cfg, dev,
                                             args.seed + 17 * start + 1), np.repeat(np.arange(sub.numel()), args.randoms)),
                    'reference': reference_rows(prior, ident)}
            for kind, (crystals, owner) in sets.items():
                cells0 = cell_rows(crystals)
                relaxed, s, e = relax(crystals, stages, dev, scratch)
                assert list(relaxed.identifier) == [ident[m] for m in owner], f'{kind}: rows came back in another order'
                res[kind] = {'mol': owner, 'e_start': s['e'].numpy(), 'e_end': e['e'].numpy(), 'cp_start': s['cp'].numpy(),
                             'cp_end': e['cp'].numpy(), 'cell_start': cells0, 'cell_end': cell_rows(relaxed),
                             'crystals': relaxed if args.keep_crystals else None}
                if kind == 'reference':
                    res[kind]['e_stored'] = (crystals.elj.double().flatten() / crystals.z_prime.double().flatten()).cpu().numpy()
            torch.save(dict(res, meta=meta), path + '.tmp')
            os.replace(path + '.tmp', path)
            n = sum(len(res[k]['mol']) for k in KINDS)
            print(f'{split} molecules {start}-{start + sub.numel() - 1}: {n} relaxations; {time.time() - t0:.0f} s elapsed', flush=True)
    if os.path.exists(scratch):
        os.remove(scratch)


def check_batching(args):
    """Relax the same starts whole and in two halves; with fixed steps the ends must agree."""
    dev = args.device
    run_ = load_run(args.checkpoint, args.config, device=dev)
    stages, search_cfg = search_stages(args.search_config, not args.search_stopping, args.stage_plan)
    scratch = os.path.join(args.out, f'opt_intermediates_{os.getpid()}.pt')
    rows = torch.arange(8).repeat_interleave(4)
    d = draw(run_, run_.conditions.subsample_new_batch(rows), batch_size=rows.numel(), seed=args.seed)
    sets = {'model draws': d['sample_batch'],
            'random starts': random_starts(run_.conditions.subsample_new_batch(rows), search_cfg, dev, args.seed + 1).cpu()}
    for name, crystals in sets.items():
        n = crystals.num_graphs
        t = time.time()
        whole, s_whole, e_whole = relax(crystals, stages, dev, scratch)
        dt = time.time() - t
        order = torch.randperm(n, generator=torch.Generator().manual_seed(1))
        e_split, cell_split = np.empty(n), np.empty((n, 6))
        for h in (order[:n // 2], order[n // 2:]):
            part, _, e = relax(crystals.subsample_new_batch(h), stages, dev, scratch)
            e_split[h.numpy()], cell_split[h.numpy()] = e['e'].numpy(), cell_rows(part)
        de = np.abs(e_split - e_whole['e'].numpy())
        dc = np.abs(cell_split - cell_rows(whole)).max(1)
        print(f'{name}: {n} starts relaxed whole ({dt:.0f} s) and in two halves ({"search stopping" if args.search_stopping else "fixed steps"}): '
              f'energy differs by a median {np.median(de):.2e} and at most {de.max():.2e} raw units; cell lengths and angles by at most '
              f'{dc.max():.2e} (A or rad); rows differing by more than 0.01 raw units: {(de > 0.01).sum()} | energy start median '
              f'{s_whole["e"].median():.1f}, end median {e_whole["e"].median():.1f}', flush=True)
        if name == 'model draws':  # the "as drawn" energy here against the one topline's excess is built from
            gap = np.abs(s_whole['e'].numpy() - run_.energy_function.seed_energy_from(d).double().cpu().numpy())
            print(f'model draws: the relaxer\'s energy before any step and the trainer\'s seed energy differ by a median {np.median(gap):.3f} '
                  f'and at most {gap.max():.3f} raw units', flush=True)
    if os.path.exists(scratch):
        os.remove(scratch)


def check_converged(args, paths):
    """Relax every kept end again. An end that is a minimum of the relaxer stays where it is;
    the energy an end loses here is what the first relaxation left on the table."""
    stages, _ = search_stages(args.search_config, not args.search_stopping, args.stage_plan)
    scratch = os.path.join(args.out, f'opt_intermediates_{os.getpid()}.pt')
    fell = {k: [] for k in KINDS}
    T = None
    for path in paths:
        c = torch.load(path, weights_only=False)
        T = float(c['meta']['temperature'])
        for k in KINDS:
            assert c[k]['crystals'] is not None, f'{path} was written without --keep-crystals'
            _, s, e = relax(c[k]['crystals'], stages, args.device, scratch)
            assert np.abs(s['e'].numpy() - c[k]['e_end']).max() < 0.05, 'a kept end does not have its recorded energy'
            fell[k].append((c[k]['e_end'] - e['e'].numpy()) / T)
        print(f'{os.path.basename(path)} done', flush=True)
    print(f'Kept ends relaxed again ({"; ".join(str(s["max_num_steps"]) + " steps from step size " + str(s["init_lr"]) for s in stages)}); '
          f'energy lost on the second relaxation, in kT (1 kT = {T:g} raw units):', flush=True)
    print(f'{"ends of":>22} | {"count":>6} | {"median":>7} | {"90th pct":>8} | {"99th pct":>8} | {"max":>7} | {"share over 0.1 kT":>17} | {"share over 1 kT":>15}')
    for k in KINDS:
        d = np.concatenate(fell[k])
        print(f'{k + " starts" if k != "reference" else "stored minima":>22} | {len(d):6d} | {np.median(d):7.3f} | {np.quantile(d, 0.9):8.3f} | '
              f'{np.quantile(d, 0.99):8.3f} | {d.max():7.2f} | {(d > 0.1).mean():17.3f} | {(d > 1).mean():15.3f}', flush=True)
    if os.path.exists(scratch):
        os.remove(scratch)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--test-prior', default=None, help='stored minima of the held-out molecules')
    ap.add_argument('--search-config', default=None, help='the search YAML whose opt stages are the relaxer')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-train', type=int, default=60)
    ap.add_argument('--n-test', type=int, default=60)
    ap.add_argument('--draws', type=int, default=32, help='model draws per molecule')
    ap.add_argument('--randoms', type=int, default=32, help='random starts per molecule')
    ap.add_argument('--chunk-molecules', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--search-stopping', action='store_true', help="keep the search's batch-wide convergence stop")
    ap.add_argument('--stage-plan', type=int, nargs='+', default=None,
                    help="indices of the search config's opt stages to run, in order (default: all of them once)")
    ap.add_argument('--keep-crystals', action='store_true', help='store every relaxed crystal (for structure comparison)')
    ap.add_argument('--check-batching', action='store_true', help='relax 32 starts of each kind whole and in halves, print the difference, stop')
    ap.add_argument('--check-converged', action='store_true', help='relax the ends kept in --out again by --stage-plan, print how much '
                                                                   'further they fall, stop (needs chunks written with --keep-crystals)')
    ap.add_argument('--figures', default=None, help='figure directory: draw from the chunks in --out and stop')
    ap.add_argument('--against', default=None, help='with --figures: a second run of the same starts under another relaxer, compared '
                                                    'start by start with the run in --out')
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'relax_chunks'), exist_ok=True)
    chunks = lambda d: sorted(glob.glob(os.path.join(d, 'relax_chunks', '*.pt')))
    if args.figures and args.against:
        compare(chunks(args.out), chunks(args.against), args.figures)
    elif args.figures:
        figures(chunks(args.out), args.figures)
    elif args.check_batching:
        check_batching(args)
    elif args.check_converged:
        check_converged(args, chunks(args.out))
    else:
        run(args)


def collect(paths):
    """Chunks joined into flat per-relaxation arrays with molecules numbered across chunks.
    Returns them with, per molecule, its identifier, split, stored best energy (the run's
    energy reference) and floor. Raises unless the three energies are one currency."""
    assert paths, 'no relaxation chunks found'
    fields = ('mol', 'e_start', 'e_end', 'cp_start', 'cp_end', 'cell_start', 'cell_end')
    out = {k: {f: [] for f in fields} for k in KINDS}
    ident, split, e_ref, stored, meta, n = [], [], [], [], None, 0
    for path in paths:
        c = torch.load(path, weights_only=False)
        meta = c['meta']
        ident += list(c['identifier'])
        split += [c['split']] * len(c['identifier'])
        e_ref.append(c['e_ref_table'])
        for k in KINDS:
            out[k]['mol'].append(c[k]['mol'] + n)
            for f in fields[1:]:
                out[k][f].append(c[k][f])
        stored.append(c['reference']['e_stored'])
        n += len(c['identifier'])
    out = {k: {f: np.concatenate(v) for f, v in d.items()} for k, d in out.items()}
    ref = out['reference']
    ref['e_stored'] = np.concatenate(stored)
    e_ref = np.concatenate(e_ref)
    lowest = np.full(n, np.inf)
    np.minimum.at(lowest, ref['mol'], ref['e_stored'])
    gap_table = float(np.abs(lowest - e_ref).max())
    gap_relaxer = float(np.abs(ref['e_start'] - ref['e_stored']).max())
    print(f'currency: the run\'s energy reference and the lowest stored minimum differ by at most {gap_table:.3f} raw units over {n} '
          f'molecules; the relaxer\'s energy of a stored minimum before any step and its stored energy by at most {gap_relaxer:.3f}', flush=True)
    assert gap_table < 0.1 and gap_relaxer < 0.1, 'the energy reference, the stored minima and the relaxer are not in one currency'
    for k in KINDS:
        assert np.isfinite(out[k]['e_end']).all(), f'{k}: a relaxation ended at a non-finite energy'
    floor = e_ref.copy()
    for k in KINDS:
        np.minimum.at(floor, out[k]['mol'], out[k]['e_end'])
    return out, np.array(ident), np.array(split), e_ref, floor, meta


def distinct_minima(e, tol=DISTINCT_TOL):
    """End energies grouped where neighbours in sorted order lie within `tol`: the lowest
    energy of each group and each input's group index."""
    order = np.argsort(e)
    new = np.concatenate([[True], np.diff(e[order]) > tol])
    group = np.empty(len(e), dtype=int)
    group[order] = np.cumsum(new) - 1
    return e[order][new], group


def describe_relaxer(meta):
    return '; '.join(f'Rprop from step size {s["init_lr"]}, {s["max_num_steps"]} steps' for s in meta['stages'])


def _weighted_quantile(v, w, q):
    order = np.argsort(v)
    cw = np.cumsum(w[order]) / w.sum()
    return float(v[order][min(np.searchsorted(cw, q), len(v) - 1)])


def figures(paths, fig_dir):
    """The relaxation figures and tables, from the chunk files alone."""
    from scipy.stats import spearmanr

    from eval.cond_panel import figstyle as fs
    R, ident, split, e_ref, floor, meta = collect(paths)
    F = fs.Figures(fig_dir)
    T = float(meta['temperature'])
    n_mol = len(ident)
    names = {'train': 'training', 'held_out': 'held-out'}
    present = [sp for sp in names if (split == sp).any()]
    colours = {'train': fs.TRAIN, 'held_out': fs.HELD}
    labels = {'model': 'model draws', 'random': 'random starts', 'reference': 'stored search minima'}
    ex = {k: (R[k]['e_end'] - floor[R[k]['mol']]) / T for k in KINDS}
    ex0 = {k: (R[k]['e_start'] - floor[R[k]['mol']]) / T for k in KINDS}

    # a reference for the target exp(-E/kT): each molecule's random-start ends reweighted by it,
    # which is where relaxed draws would end if a random start reached a minimum in proportion
    # to its thermal weight. With a few dozen ends a molecule the lowest ones carry the weight,
    # so it leans toward the floor.
    thermal = []  # per molecule: (excess, weight summing to 1)
    reached_also, n_distinct, top_share = ({k: np.zeros(n_mol) for k in ('model', 'random')} for _ in range(3))
    for m in range(n_mol):
        ends = {k: R[k]['e_end'][R[k]['mol'] == m] for k in KINDS}
        _, group = distinct_minima(np.concatenate([ends[k] for k in KINDS]))
        u = (ends['random'] - floor[m]) / T
        w = np.exp(-(u - u.min()))
        thermal.append((u, w / w.sum()))
        g = dict(zip(KINDS, np.split(group, np.cumsum([len(ends[k]) for k in KINDS])[:-1])))
        for k, other in (('model', 'random'), ('random', 'model')):
            reached_also[k][m] = np.isin(g[k], g[other]).mean()
            counts = np.unique(g[k], return_counts=True)[1]
            n_distinct[k][m], top_share[k][m] = len(counts) / len(g[k]), counts.max() / len(g[k])

    def selection_strength(of_split):
        """The exponent at which the random starts' ends, reweighted by exp(-exponent *
        excess), have the model's mean relaxed excess: 0 is the random starts' own spread
        over minima, 1 is the target if a random start reaches a minimum in proportion to
        its thermal weight."""
        mols = np.flatnonzero(of_split)
        target = np.mean([ex['model'][R['model']['mol'] == m].mean() for m in mols])

        def mean_at(beta):
            vals = []
            for m in mols:
                u = ex['random'][R['random']['mol'] == m]
                w = np.exp(-beta * (u - u.min()))
                vals.append((w * u).sum() / w.sum())
            return np.mean(vals)
        lo, hi = -2.0, 6.0
        if not mean_at(hi) < target < mean_at(lo):
            return float('nan')
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if mean_at(mid) > target else (lo, mid)
        return 0.5 * (lo + hi)

    header = ['molecules', 'structures', 'count', 'median kT above the floor, before', 'median kT above the floor, after',
              'mean kT above the floor, after', 'share within 1 kT of the floor', 'within 3 kT', 'within 5 kT', 'within 10 kT',
              'share ending more than 0.1 kT below the stored best']
    rows, points = [], []
    fig, axes = fs.panels(4, ncols=4, width=3.8, height=3.6)
    grid = np.linspace(0, 45, 451)
    for ax in axes[len(present):2]:
        ax.remove()
    for ax, sp in zip(axes[:2], present):
        fs.style(ax, grid='both')
        of_split = split == sp
        for k, colour in (('model', fs.TRAIN), ('random', fs.MUTED), ('reference', None)):
            sel = of_split[R[k]['mol']]
            v, v0 = ex[k][sel], ex0[k][sel]
            below = ((R[k]['e_end'][sel] - e_ref[R[k]['mol'][sel]]) / T < -0.1).mean()
            rows.append([names[sp], f'{labels[k]}, relaxed', int(sel.sum()), round(float(np.median(v0)), 2), round(float(np.median(v)), 2),
                         round(float(v.mean()), 2)] + [round(float((v <= c).mean()), 3) for c in (1, 3, 5, 10)] + [round(float(below), 3)])
            if colour:
                ax.plot(grid, [(v <= g).mean() for g in grid], color=colour, label=f'{labels[k]}, relaxed')
        v0 = ex0['model'][of_split[R['model']['mol']]]
        ax.plot(grid, [(v0 <= g).mean() for g in grid], color=fs.TRAIN, linestyle=(0, (1.5, 1.5)), linewidth=1.5, label='model draws, as drawn')
        sets = [thermal[m] for m in np.flatnonzero(of_split)]
        u = np.concatenate([s[0] for s in sets])
        w = np.concatenate([s[1] for s in sets]) / len(sets)
        rows.append([names[sp], 'random starts, relaxed, reweighted by exp(-E/kT)', len(u), '', round(_weighted_quantile(u, w, 0.5), 2),
                     round(float((u * w).sum()), 2)] + [round(float(w[u <= c].sum()), 3) for c in (1, 3, 5, 10)] + [''])
        ax.plot(grid, [w[u <= g].sum() for g in grid], color=fs.THIRD, linestyle=(0, (5, 2)), linewidth=1.5,
                label='random starts, reweighted')
        ax.set_xlabel('kT above the molecule\'s floor')
        ax.set_title(f'{names[sp]} molecules', fontsize=9)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel('share of structures at or below')
    axes[0].legend(loc='lower right', fontsize=6.5)

    share3, best, mean_end = {}, {}, {}
    for k in ('model', 'random'):
        hit, cnt, tot = np.zeros(n_mol), np.zeros(n_mol), np.zeros(n_mol)
        np.add.at(hit, R[k]['mol'], ex[k] <= 3.0)
        np.add.at(cnt, R[k]['mol'], 1.0)
        np.add.at(tot, R[k]['mol'], ex[k])
        share3[k], mean_end[k] = hit / cnt, tot / cnt
        low = np.full(n_mol, np.inf)
        np.minimum.at(low, R[k]['mol'], R[k]['e_end'])
        best[k] = (low - e_ref) / T
    ax = axes[2]
    fs.style(ax, grid='both')
    for sp in present:
        m, colour = split == sp, colours[sp]
        ax.scatter(share3['random'][m], share3['model'][m], s=16, color=colour, alpha=0.8, linewidths=0, label=names[sp])
    ax.plot([0, 1], [0, 1], color=fs.AXIS, linewidth=1)
    ax.set_xlabel('random starts ending within 3 kT of the floor (share)')
    ax.set_ylabel('model draws ending within 3 kT of the floor (share)')
    ax.legend(loc='upper left', fontsize=8)
    ax = axes[3]
    fs.style(ax, grid='both')
    lo = float(min(best['model'].min(), best['random'].min())) - 0.3
    hi = float(np.quantile(np.concatenate([best['model'], best['random']]), 0.99)) + 0.3
    for sp in present:
        m, colour = split == sp, colours[sp]
        ax.scatter(best['random'][m], best['model'][m], s=16, color=colour, alpha=0.8, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color=fs.AXIS, linewidth=1)
    ax.axhline(0, color=fs.AXIS, linewidth=1)
    ax.axvline(0, color=fs.AXIS, linewidth=1)
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_xlabel('lowest relaxed random start minus the stored best (kT)')
    ax.set_ylabel('lowest relaxed model draw minus the stored best (kT)')

    drop = (R['model']['e_start'] - R['model']['e_end']) / T
    for sp in present:
        m = split == sp
        sm, sr = m[R['model']['mol']], m[R['random']['mol']]
        points.append(f'{names[sp]}: a model draw sits a median {np.median(ex0["model"][sm]):.1f} kT above the floor as drawn and '
                      f'{np.median(ex["model"][sm]):.1f} kT after relaxing (relaxation removes a median {np.median(drop[sm]):.1f} kT; a harmonic '
                      f'basin in 12 coordinates holds 6 kT); a relaxed random start ends a median {np.median(ex["random"][sr]):.1f} kT above the floor')
        d3 = (share3['model'] - share3['random'])[m]
        du = (mean_end['model'] - mean_end['random'])[m]
        points.append(f'{names[sp]}: {(ex["model"][sm] <= 3).mean():.0%} of relaxed model draws end within 3 kT of the floor against '
                      f'{(ex["random"][sr] <= 3).mean():.0%} of relaxed random starts; per molecule the model\'s share is the larger in '
                      f'{(share3["model"][m] > share3["random"][m]).mean():.0%} of molecules and the smaller in '
                      f'{(share3["model"][m] < share3["random"][m]).mean():.0%}')
        points.append(f'{names[sp]}: model minus random starts, molecule by molecule ({int(m.sum())} molecules, mean and its standard error): '
                      f'share within 3 kT {d3.mean():+.3f} +- {d3.std(ddof=1) / np.sqrt(len(d3)):.3f}; mean relaxed excess '
                      f'{du.mean():+.2f} +- {du.std(ddof=1) / np.sqrt(len(du)):.2f} kT')
        points.append(f'{names[sp]}: selection strength {selection_strength(m):.2f} (the exponent at which random starts\' ends, reweighted by '
                      f'exp(-exponent x excess), match the model\'s mean relaxed excess; 0 = the random starts\' own spread over minima, 1 = '
                      f'the target if a random start reaches a minimum in proportion to its thermal weight)')
        win = (best['model'][m] < best['random'][m] - 0.1).mean()
        lose = (best['random'][m] < best['model'][m] - 0.1).mean()
        points.append(f'{names[sp]}: the lowest of {meta["draws"]} relaxed model draws is more than 0.1 kT below the lowest of {meta["randoms"]} '
                      f'relaxed random starts in {win:.0%} of molecules, above it in {lose:.0%}, level in {1 - win - lose:.0%}; it is below the '
                      f'stored best in {(best["model"][m] < -0.1).mean():.0%} of molecules (random starts: {(best["random"][m] < -0.1).mean():.0%}), '
                      f'and the floor sits a median {np.median((e_ref[m] - floor[m]) / T):.2f} kT below the stored best')
    stop = 'every row takes every step' if meta['fixed_steps'] else 'the search\'s batch-wide stop'
    F.save(fig, 'relax_outcomes', 'Where model draws end when relaxed, against random starts',
           f'{n_mol} molecules ({int((split == "train").sum())} training, {int((split == "held_out").sum())} held-out), checkpoint step '
           f'{meta["step"]}. Per molecule: {meta["draws"]} model draws, {meta["randoms"]} starts from the search\'s random initialiser and '
           f'the stored search minima, all relaxed by one relaxer ({describe_relaxer(meta)}; {stop}). Energies are the relaxer\'s eLJ; a '
           f'molecule\'s floor is the lowest energy known for it, stored or reached here; 1 kT = {T:g} raw units. First two panels: cumulative '
           f'share of structures within a distance of the floor. The green line reweights each molecule\'s random-start ends by exp(-E/kT): '
           f'where relaxed draws would end if a random start reached a minimum in proportion to its thermal weight. It is a reference, not '
           f'a measurement of the target, and with {meta["randoms"]} ends a molecule it leans toward the floor. Third: one point per molecule, the share of relaxed model draws within 3 kT of the floor against the same share for '
           f'random starts. Fourth: one point per molecule, the lowest relaxed model draw and the lowest relaxed random start, each minus '
           f'the stored best; below zero is a minimum lower than any the search stored.', (header, rows), points)

    # how many minima the relaxed draws fall into, and what relaxation changes
    fig, axes = fs.panels(3, ncols=3, width=4.0, height=3.5)
    header, rows, points = ['quantity', 'molecules', 'model draws', 'random starts'], [], []
    ax = axes[0]
    fs.style(ax, grid='both')
    for sp in present:
        m, colour = split == sp, colours[sp]
        ax.scatter(n_distinct['random'][m], n_distinct['model'][m], s=16, color=colour, alpha=0.8, linewidths=0, label=names[sp])
        for label, q in ((f'distinct end energies per relaxation ({DISTINCT_TOL} raw units), median over molecules', n_distinct),
                         ('share of relaxations in the most visited minimum, median over molecules', top_share),
                         ('share of relaxations ending in a minimum the other kind of start also reached, median over molecules', reached_also)):
            rows.append([label, names[sp], round(float(np.median(q['model'][m])), 3), round(float(np.median(q['random'][m])), 3)])
        points.append(f'{names[sp]}: relaxed model draws end at {np.median(n_distinct["model"][m]):.2f} distinct energies per relaxation and '
                      f'random starts at {np.median(n_distinct["random"][m]):.2f}; a median {np.median(reached_also["model"][m]):.0%} of a '
                      f'molecule\'s relaxed model draws end in a minimum a random start also reached, and {np.median(reached_also["random"][m]):.0%} '
                      f'of its random starts in a minimum a model draw reached')
    ax.plot([0, 1], [0, 1], color=fs.AXIS, linewidth=1)
    ax.set_xlim(0, 1.03), ax.set_ylim(0, 1.03)
    ax.set_xlabel('distinct end energies per relaxation, random starts')
    ax.set_ylabel('distinct end energies per relaxation, model draws')
    ax.legend(loc='lower right', fontsize=8)
    ax = axes[1]
    fs.style(ax)
    bins = np.linspace(0.45, 0.95, 51)
    ax.hist(R['model']['cp_start'], bins=bins, histtype='step', color=fs.TRAIN, linewidth=1.5, linestyle=(0, (1.5, 1.5)), density=True, label='model draws, as drawn')
    ax.hist(R['model']['cp_end'], bins=bins, histtype='step', color=fs.TRAIN, linewidth=1.8, density=True, label='model draws, relaxed')
    ax.hist(R['random']['cp_end'], bins=bins, histtype='step', color=fs.MUTED, linewidth=1.8, density=True, label='random starts, relaxed')
    ax.hist(R['reference']['cp_start'], bins=bins, histtype='step', color=fs.THIRD, linewidth=1.8, density=True, label='stored search minima')
    ax.set_xlabel('packing coefficient')
    ax.set_yticks([])
    ax.legend(loc='upper left', fontsize=7)
    ax = axes[2]
    fs.style(ax, grid='both')
    keep = np.random.default_rng(0).permutation(len(drop))[:3000]
    ax.scatter(ex0['model'][keep], ex['model'][keep], s=4, color=fs.TRAIN, alpha=0.35, linewidths=0)
    ax.set_xlim(0, float(np.quantile(ex0['model'], 0.99)))
    ax.set_ylim(0, float(np.quantile(ex['model'], 0.995)))
    ax.set_xlabel('a model draw as drawn: kT above the floor')
    ax.set_ylabel('the same draw relaxed: kT above the floor')
    rho = float(spearmanr(ex0['model'], ex['model'])[0])
    ax.set_title(f'Spearman rho = {rho:.2f}', fontsize=9)
    moved = {k: np.abs(R[k]['cell_end'] - R[k]['cell_start']) for k in KINDS}
    rows.append(['packing coefficient, median, as drawn and relaxed', 'both',
                 f'{np.median(R["model"]["cp_start"]):.3f} and {np.median(R["model"]["cp_end"]):.3f}',
                 f'{np.median(R["random"]["cp_start"]):.3f} and {np.median(R["random"]["cp_end"]):.3f}'])
    rows.append(['cell length change on relaxing, median over rows and axes (angstrom)', 'both',
                 round(float(np.median(moved['model'][:, :3])), 3), round(float(np.median(moved['random'][:, :3])), 3)])
    rows.append(['cell angle change on relaxing, median over rows and angles (rad)', 'both',
                 round(float(np.median(moved['model'][:, 3:])), 3), round(float(np.median(moved['random'][:, 3:])), 3)])
    points.append(f'a draw\'s excess as drawn and after relaxing correlate at Spearman rho {rho:.2f} over {len(drop)} draws')
    points.append(f'median packing coefficient: model draws {np.median(R["model"]["cp_start"]):.3f} as drawn and {np.median(R["model"]["cp_end"]):.3f} '
                  f'relaxed; relaxed random starts {np.median(R["random"]["cp_end"]):.3f}; stored minima {np.median(R["reference"]["cp_start"]):.3f}')
    points.append(f'relaxing changes a model draw\'s cell lengths by a median {np.median(moved["model"][:, :3]):.2f} angstrom and its angles by '
                  f'{np.median(moved["model"][:, 3:]):.2f} rad; a random start\'s by {np.median(moved["random"][:, :3]):.2f} angstrom and '
                  f'{np.median(moved["random"][:, 3:]):.2f} rad')
    ref_move = (R['reference']['e_end'] - R['reference']['e_stored']) / T
    points.append(f'the {len(ref_move)} stored minima under this relaxer: energy changes by a median {np.median(ref_move):+.3f} kT; '
                  f'{(ref_move < -1).mean():.0%} fall by more than 1 kT (lowest {ref_move.min():+.1f} kT); cell lengths move a median '
                  f'{np.median(moved["reference"][:, :3]):.3f} angstrom')
    F.save(fig, 'relax_spread', 'How many minima the relaxed draws fall into, and what relaxation changes',
           f'The relaxations of the figure above. Left: one point per molecule, the number of distinct end energies (ends within '
           f'{DISTINCT_TOL} raw units of a neighbour counted as one) over the number of relaxations, model draws against random starts; 1 '
           f'means every relaxation ended somewhere different. Middle: packing coefficient, all molecules. Right: 3,000 model draws, kT '
           f'above the floor as drawn against the same draw relaxed.', (header, rows), points)


def compare(paths_a, paths_b, fig_dir):
    """The same starts under two relaxers: the run in `--out` against the run in `--against`."""
    from eval.cond_panel import figstyle as fs
    A, ident, split, e_ref, floor_a, meta_a = collect(paths_a)
    B, ident_b, _, _, floor_b, meta_b = collect(paths_b)
    assert list(ident) == list(ident_b), 'the two runs hold different molecules'
    F = fs.Figures(fig_dir)
    T = float(meta_a['temperature'])
    floor = np.minimum(floor_a, floor_b)
    labels = {'model': 'model draws', 'random': 'random starts', 'reference': 'stored search minima'}
    header = ['starts', 'count', 'median kT above the floor, first relaxer', 'median kT above the floor, second relaxer',
              'share ending within 0.1 kT of each other', 'share ending more than 1 kT higher under the second',
              'share ending more than 1 kT lower under the second']
    rows, points = [], []
    fig, axes = fs.panels(3, ncols=3, width=3.9, height=3.6)
    for ax, k in zip(axes, KINDS):
        fs.style(ax, grid='both')
        paired = A[k]['cell_start'].shape == B[k]['cell_start'].shape and bool(np.abs(A[k]['cell_start'] - B[k]['cell_start']).max() < 1e-3)
        assert paired, f'{k}: the two runs did not relax the same starts'
        a, b = (A[k]['e_end'] - floor[A[k]['mol']]) / T, (B[k]['e_end'] - floor[B[k]['mol']]) / T
        rows.append([labels[k], len(a), round(float(np.median(a)), 2), round(float(np.median(b)), 2), round(float((np.abs(a - b) <= 0.1).mean()), 3),
                     round(float((b > a + 1).mean()), 3), round(float((b < a - 1).mean()), 3)])
        points.append(f'{labels[k]}: median {np.median(a):.2f} kT above the floor under the first relaxer and {np.median(b):.2f} under the second; '
                      f'{(np.abs(a - b) <= 0.1).mean():.0%} of starts end within 0.1 kT of each other, {(b > a + 1).mean():.0%} more than 1 kT '
                      f'higher under the second and {(b < a - 1).mean():.0%} more than 1 kT lower')
        hi = float(np.quantile(np.concatenate([a, b]), 0.995))
        ax.scatter(a, b, s=4, color=fs.TRAIN, alpha=0.35, linewidths=0, rasterized=True)
        ax.plot([0, hi], [0, hi], color=fs.AXIS, linewidth=1)
        ax.set_xlim(0, hi), ax.set_ylim(0, hi)
        ax.set_title(labels[k], fontsize=9)
        ax.set_xlabel('kT above the floor, first relaxer')
    axes[0].set_ylabel('kT above the floor, second relaxer')
    F.save(fig, 'relax_two_relaxers', 'The same starts under two relaxers',
           f'{len(ident)} molecules, checkpoint step {meta_a["step"]}; every start of the relaxation figures relaxed twice. First relaxer: '
           f'{describe_relaxer(meta_a)}. Second relaxer: {describe_relaxer(meta_b)}. One point per start: kT above the molecule\'s floor (the '
           f'lowest energy known for it across both runs) at the end of each; 1 kT = {T:g} raw units.', (header, rows), points)


if __name__ == '__main__':
    main()
