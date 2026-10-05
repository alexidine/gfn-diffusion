"""
A mock crystal-structure-prediction run: what does it cost a sampler to recover a molecule's
low-energy minima?

THE TASK. For one molecule the TARGET is every distinct minimum within `--window` kT of the
lowest energy known for it. A SAMPLER proposes starts: a checkpoint's draws; the crystal
search's random initialiser; or the unconditional prior, the latents of every OTHER
molecule's stored search minima rebuilt with this molecule. A STRATEGY decides which starts
are relaxed and for how long.
Two budgets are counted per molecule: DRAWS taken from the sampler, and ENERGY CALLS, one
energy-and-force evaluation of one crystal (one relaxation step of one row is one call).
The result is the share of the target recovered as a function of each budget.

TWO PHASES.
  collect  (GPU) every sampler proposes `--starts` starts for every molecule and EVERY start
           is relaxed by the search's own schedule, every row taking every step, with the
           energy at every step kept (eval/cond_panel/relax.py). One file per sampler and
           molecule; a file on disk is skipped.
  replay   (CPU) strategies are read off the stored trajectories: relaxing a start for k
           steps is that start's first k recorded steps. Any number of strategies then
           costs no further energy call. The price is that a strategy can choose only among
           starts that were collected.
  collect --cluster-from M   the strategy a replay cannot reach, run directly: M starts are
           drawn per molecule with no energy call and put in their canonical latents; the
           modes of their local density are found by mean shift (density_modes); the modes
           of the `--starts` largest clusters are relaxed. Stored as its own pool,
           `<sampler>+modes<M>`, largest cluster first, and charged M draws. M belongs in
           the tens of thousands: at a few thousand draws the nearest draw is farther than
           the distance over which two draws relax to one minimum, and no cluster shows.

WHICH MINIMUM a start leads to is where its full relaxation ends, if it had SETTLED: its
best energy fell by no more than E_TOL over its last SETTLE_WINDOW recorded steps. The end
of a relaxation still descending is not a minimum to the tolerance that tells minima apart;
it defines none and is credited with none, and its energy calls are spent all the same. Two
settled ends are one minimum when their energies agree within E_TOL and their packing
coefficients within CP_TOL, both invariant under any re-description of the crystal. A
molecule's reference minima are the union over every sampler collected for it, so every
sampler is scored against one set, and that set grows when a sampler is added.

A STRATEGY is a selection rule and a stopping rule (STRATEGIES):
  selection   all      relax every start drawn
              screen   one energy call on every start, relax the lowest fraction
              cascade  relax every start, retiring a relaxation that is too far above the
                       best in hand at set steps: the crystal search's own early stop
                       (mxtaltools crystal_opt_utils.py, `early_stop`) at the steps and
                       margins of its campaigns, CASCADE. The first quarter of the starts
                       runs without it and its lowest energy is the reference, as the
                       search's first batch sets its own.
  stopping    full     every recorded step
              plateau  stop a row when its best energy so far has improved by less than
                       PLATEAU_EPS kT over the last PLATEAU_WINDOW steps
              oracle   stop a row at the first step within CREDIT kT of where it ends: no
                       rule that cannot see the end spends fewer calls
A relaxation is credited with its minimum only if it stopped within CREDIT kT of where the
full relaxation ends; its calls are spent either way.

CANONICAL LATENTS, for clustering before any energy call. A crystal has many descriptions;
`canonical` picks one, for P-1 at Z' = 1 in the trainer's chart (the standard-frame molecule
at handedness +1):
  1. the reduced cell: mxtaltools crystal_search/standardize.py::standardize_cells chooses
     the change of lattice basis N and origin o;
  2. the trainer's molecule in that cell: its atoms carried over by x' = N^-1 (x - o) and
     described again at +1 by data_processing/pool_anchors.py::Chart.describe and
     .into_box. The standardised batch itself is NOT what the trainer reads: putting the
     molecule back in its asymmetric unit can return the inversion mate, which the trainer
     builds at +1 as another crystal;
  3. the origin image with the centroid's v and w on one period (latents.fold).
`check-fold` scores each of these through the trainer's reward function.

    cd energy_sampling
    python -m eval.cond_panel.csp collect --model final=<ckpt.pt> --config <run.yaml> --random --prior \\
        --search-config <MXtalTools>/configs/crystal_searches/qm9_full_sep29/tasks/0.yaml --out <dir>/csp
    python -m eval.cond_panel.csp collect --model final=<ckpt.pt> --config <run.yaml> --prior \\
        --search-config <...>/tasks/0.yaml --out <dir>/csp --cluster-from 32768 --starts 32
    python -m eval.cond_panel.csp figures --out <dir>/csp --figures <dir>/figures
"""
from __future__ import annotations

import argparse
import glob
import os
import socket
import time

import numpy as np
import torch

from eval.cond_panel.latents import PERIOD, RANGE, fold

E_TOL = 0.1         # raw energy units: ends closer than this ...
CP_TOL = 0.002      # ... with packing coefficients closer than this are one minimum
SETTLE_WINDOW = 50  # steps: a relaxation whose best energy still fell by more than E_TOL over its last ones has reached no minimum
CREDIT = 0.1        # kT: a stopped relaxation this close to its full end has found that minimum
PLATEAU_WINDOW = 40
PLATEAU_EPS = 0.003  # kT
# density clusters of the draws (density_modes): kernel sigma = D_CUT / 3, and a draw points at the densest draw within
# D_CUT. In units of a latent's full range. Two model draws 0.05 apart relax to one minimum about half the time
# (qf30_fwdF_lr5, 2026-10-05), which sets the kernel.
D_CUT = 0.15
# (step, margin in kT): at each step a relaxation more than the margin above the reference is retired. The values of
# MXtalTools configs/crystal_searches/acr_campaign_sep28 and acr_zp1_sep30 (make_campaign.py, CASCADE)
CASCADE = ((25, 10.0), (50, 8.0), (100, 5.0))
# (label, selection, share of the starts relaxed, stopping)
STRATEGIES = (('every start, every step', 'all', 1.0, 'full'),
              ('every start, stop on a plateau', 'all', 1.0, 'plateau'),
              ('every start, stopped by an oracle', 'all', 1.0, 'oracle'),
              ('screen by starting energy, relax the lowest quarter', 'screen', 0.25, 'plateau'),
              ("every start, the search's energy cascade", 'cascade', 1.0, 'plateau'))


# ---------------------------------------------------------------------------------------- collect

def reduced_latents(crystals, trainer_chart=True):
    """The 12 latents of each crystal in its reduced cell, and standardize_cells' report. A
    row whose cell is already reduced, or could not be standardised, keeps its own latents.
    With `trainer_chart` (crystals the trainer built: space group 2, Z' = 1, handedness +1)
    a row whose cell changes is described again in the trainer's chart; without it the
    standardised row's own latents are returned, which is the search's description."""
    from mxtaltools.crystal_search.standardize import standardize_cells
    crystals = crystals.cpu()
    std, info = standardize_cells(crystals, on_failure='flag')
    lat = crystals.latent_params().detach().cpu().double()
    moved = np.flatnonzero(info['changed'] & info['ok'])
    if not len(moved):
        return lat, info
    if not trainer_chart:
        lat[moved] = std.latent_params().detach().cpu().double()[moved]
        return lat, info
    from data_processing.pool_anchors import Chart
    assert bool((crystals.sg_ind.flatten() == 2).all()) and bool((crystals.aunit_handedness.flatten() == 1).all()), \
        'the trainer-chart description is written for space group 2 at handedness +1'
    by_molecule = {}
    for i in moved:
        by_molecule.setdefault(crystals.identifier[i], []).append(int(i))
    for rows in by_molecule.values():       # Chart holds one molecule's standard frame at a time
        idx = torch.tensor(rows)
        before, after = crystals.subsample_new_batch(idx), std.subsample_new_batch(idx)
        chart = Chart(None, 2, 'elj')
        chart.setup(before)
        n_inv = torch.linalg.inv(torch.as_tensor(info['N'][rows], dtype=torch.double))
        atoms = torch.einsum('nij,naj->nai', n_inv, chart.frac_atoms(before) - torch.as_tensor(info['origin'][rows])[:, None, :])
        out, _ = chart.describe(after, atoms)
        out, _ = chart.into_box(out)
        lat[idx] = out.latent_params().detach().cpu().double()
    return lat, info


def canonical(crystals, trainer_chart=True):
    """One description per crystal (the module docstring, CANONICAL LATENTS): [n, 12] numpy,
    and standardize_cells' report."""
    lat, info = reduced_latents(crystals, trainer_chart)
    return fold(lat.numpy()), info


def molecules(run, n_train, n_test, seed):
    """The molecules of eval/cond_panel/relax.py::run under the same seed: (split, conditions
    batch, row indices), a prefix of that run's selection for a smaller count."""
    g = torch.Generator().manual_seed(seed)
    return [('train', run.conditions, torch.randperm(run.conditions.num_graphs, generator=g)[:n_train]),
            ('held_out', run.test_conditions, torch.randperm(run.test_conditions.num_graphs, generator=g)[:n_test])]


def density_modes(features, device=None, block=2048):
    """The modes of the draws' local density in canonical latents: the mean shift of the
    landscape figures (eval/paper1_results/utils.py::mean_shift_density) with `embed`
    distance in place of the RDF distance. A draw's density is the Gaussian kernel sum over
    all the draws, sigma = D_CUT / 3. Every draw points at the densest draw closer than
    D_CUT, itself included, and pointers are followed to their fixed points, the modes.
    Returns the modes (draw indices) in descending mass, their masses (the draws that reach
    each) and every draw's kernel sum."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    z = torch.as_tensor(embed(np.asarray(features, dtype=np.float64)), dtype=torch.float32, device=device)
    n, sigma = z.shape[0], D_CUT / 3
    dens = torch.cat([torch.exp(-torch.cdist(z[r:r + block], z) ** 2 / (2 * sigma ** 2)).sum(1) for r in range(0, n, block)])
    ptr = torch.empty(n, dtype=torch.long, device=device)
    for r in range(0, n, block):
        d = torch.cdist(z[r:r + block], z)
        rows = torch.arange(r, min(r + block, n), device=device)
        d[rows - r, rows] = 0.0                                       # a draw is its own neighbour
        ptr[r:r + block] = torch.where(d < D_CUT, dens[None, :], torch.full_like(d, float('-inf'))).argmax(1)
    ptr, state = ptr.cpu().numpy(), np.arange(n)
    # a pointer never lowers the density, and among equal densities argmax takes the lowest index, so no path cycles:
    # n steps bound the walk
    for _ in range(n):
        new = ptr[state]
        if np.array_equal(new, state):
            break
        state = new
    else:
        raise RuntimeError('mean-shift pointers did not reach their fixed points')
    modes, mass = np.unique(state, return_counts=True)
    order = np.argsort(-mass, kind='stable')
    return modes[order], mass[order], dens.cpu().numpy()


def collect(args):
    from eval.cond_panel.relax import load_prior, random_starts, relax, search_stages
    from eval.cond_panel.sampler import crystals_from, draw, draw_latents, load_run
    dev = args.device
    stages, search_cfg = search_stages(args.search_config, True)
    # one scratch file per process: jobs on different nodes can share --out and a process id
    scratch = os.path.join(args.out, f'opt_intermediates_{socket.gethostname()}_{os.getpid()}.pt')
    models = [spec.split('=', 1) for spec in args.model]
    assert models, 'at least one --model: it supplies the molecules'
    samplers = (([] if args.skip_models else [(name, 'model', ckpt) for name, ckpt in models])
                + ([('random', 'random', None)] if args.random else []) + ([('prior', 'prior', None)] if args.prior else []))
    assert samplers, 'nothing to collect: --skip-models without --random or --prior'
    assert len({name for name, _, _ in samplers}) == len(samplers), 'sampler names must differ'
    assert not args.cluster_from or args.cluster_from > args.starts, '--cluster-from must exceed the number of modes relaxed (--starts)'
    per_call = max(1, args.batch_rows // args.starts)
    prior_ids = prior_lat = None
    if args.prior:      # every stored search minimum of the run's prior file, as latents: one pool for all molecules
        import yaml
        with open(args.config) as fh:
            prior = load_prior(yaml.safe_load(fh)['prior_path'])
        prior_ids, prior_lat = np.array(prior.identifier), prior.latent_params().detach().cpu().float()
        del prior
    t0 = time.time()
    run = None
    for name, kind, ckpt in samplers:
        if ckpt is not None or run is None:     # the random and prior samplers read their molecules off the last run loaded
            run = load_run(ckpt or models[0][1], args.config, device=dev)
        pool_name = f'{name}+modes{args.cluster_from}' if args.cluster_from else name

        def propose(one, seed, scored):
            """This sampler's crystals for `one`, rows of ONE molecule, and the latents the trainer
            built them from (None for the random initialiser). Unscored, no energy call is spent."""
            if kind == 'random':    # the initialiser builds every crystal it is handed at once: a thousand at a time
                out = None
                for lo in range(0, one.num_graphs, 1024):
                    part = random_starts(one.subsample_new_batch(torch.arange(lo, min(lo + 1024, one.num_graphs))), search_cfg,
                                         dev, seed + lo).cpu()
                    out = part if out is None else out.append_batch(part)
                return out, None
            if kind == 'model':
                if scored:
                    return draw(run, one, batch_size=min(one.num_graphs, 1024), seed=seed)['sample_batch'], None
                x = draw_latents(run, one, seed=seed)
            else:                   # the unconditional prior: other molecules' stored minima, rebuilt with this molecule
                other = np.flatnonzero(prior_ids != one.identifier[0])
                x = prior_lat[torch.as_tensor(np.random.default_rng(seed).choice(other, one.num_graphs, replace=False))]
            return crystals_from(run, one, x, scored=scored), x
        for split, batch, rows in molecules(run, args.n_train, args.n_test, args.seed):
            path = lambda k: os.path.join(args.out, 'pools', pool_name, f'{split}_{k:03d}.pt')
            todo = [k for k in range(rows.numel()) if not os.path.exists(path(k))]
            for lo in range(0, len(todo), per_call):
                ks = todo[lo:lo + per_call]
                sub = rows[ks]
                seed = args.seed + 7919 * ks[0] + (0 if split == 'train' else 1)
                kept, feats, sizes, changed, ok = [], [], [], [], []
                for j in range(len(ks)):
                    one = lambda m: batch.subsample_new_batch(sub[j:j + 1].repeat_interleave(m))
                    if args.cluster_from:
                        # no energy call before the relaxations: draw, canonicalise, find the density's modes, and keep
                        # the modes of the largest clusters
                        many, x = propose(one(args.cluster_from), seed + j, False)
                        f, info = canonical(many, trainer_chart=kind != 'random')
                        modes, mass, _ = density_modes(f, dev)
                        assert len(modes) >= args.starts, f'{len(modes)} density clusters, fewer than --starts'
                        chosen = torch.as_tensor(modes[:args.starts])
                        kept.append(many.subsample_new_batch(chosen) if x is None else crystals_from(run, one(args.starts), x[chosen]))
                        sizes.append(mass[:args.starts])
                        f, info = f[chosen.numpy()], {key: info[key][chosen.numpy()] for key in ('changed', 'ok')}
                    else:
                        kept.append(propose(one(args.starts), seed + j, True)[0])
                        f, info = canonical(kept[-1], trainer_chart=kind != 'random')
                    feats.append(f), changed.append(info['changed']), ok.append(info['ok'])
                crystals = kept[0]
                for more in kept[1:]:
                    crystals = crystals.append_batch(more)
                features, sizes = np.concatenate(feats), np.concatenate(sizes) if sizes else None
                info = {'changed': np.concatenate(changed), 'ok': np.concatenate(ok)}
                # at most --batch-rows crystals on the device at once: every row takes every step, so a row's relaxation
                # does not depend on the rows it shares a call with
                parts = [relax(crystals.subsample_new_batch(torch.arange(r, min(r + args.batch_rows, crystals.num_graphs))),
                               stages, dev, scratch, trajectory=True) for r in range(0, crystals.num_graphs, args.batch_rows)]
                relaxed_ids = [ident for part, _, _ in parts for ident in part.identifier]
                s = {'cp': torch.cat([start['cp'] for _, start, _ in parts])}
                e = {'e': torch.cat([end['e'] for _, _, end in parts]), 'cp': torch.cat([end['cp'] for _, _, end in parts]),
                     'traj': np.concatenate([end['traj'] for _, _, end in parts], axis=1)}
                assert len(relaxed_ids) == len(ks) * args.starts == e['traj'].shape[1], 'a molecule is short of starts'
                os.makedirs(os.path.dirname(path(ks[0])), exist_ok=True)
                for j, k in enumerate(ks):
                    sl = slice(j * args.starts, (j + 1) * args.starts)
                    ident = batch.identifier[int(sub[j])]
                    assert set(relaxed_ids[sl]) == {ident}, f'{ident}: its rows hold another molecule'
                    torch.save({'identifier': ident, 'split': split, 'sampler': pool_name,
                                'from_draws': args.cluster_from or None, 'cluster_size': None if sizes is None else sizes[sl],
                                'features': features[sl].astype(np.float32), 'standardised': info['ok'][sl],
                                'cell_changed': info['changed'][sl], 'traj': e['traj'][:, sl], 'e_end': e['e'][sl].numpy(),
                                'cp_end': e['cp'][sl].numpy(), 'cp_start': s['cp'][sl].numpy(),
                                'e_ref': float(run.energy_function.energy_reference_for(torch.tensor([run.registry[ident]]))),
                                'meta': {'step': run.step if ckpt else None, 'temperature': run.temperature, 'stages': stages,
                                         'checkpoint': os.path.basename(ckpt) if ckpt else None, 'starts': args.starts,
                                         'kind': kind, 'd_cut': D_CUT if args.cluster_from else None,
                                         'search_config': '/'.join(os.path.abspath(args.search_config).split(os.sep)[-3:])}},
                               path(k) + '.tmp')
                    os.replace(path(k) + '.tmp', path(k))
                print(f'{pool_name} {split} molecules {ks[0]}-{ks[-1]}: {len(ks) * args.starts} relaxations of {e["traj"].shape[0]} '
                      f'steps; {time.time() - t0:.0f} s elapsed', flush=True)
    if os.path.exists(scratch):
        os.remove(scratch)


def check_fold(args):
    """Score the canonical description through the trainer's reward function. A draw, the
    latents of the crystal built from it, its reduced cell, its three other origin images
    and its canonical latents must all have one energy, and the reduced cell no reduction
    penalty."""
    from mxtaltools.crystal_search.standardize import same_crystal_deviation, standardize_cells
    from eval.cond_panel.pairs import seed_excess
    from eval.cond_panel.sampler import draw, load_run
    name, ckpt = args.model[0].split('=', 1)
    run = load_run(ckpt, args.config, device=args.device)
    ef = run.energy_function
    (_, batch, rows), _ = molecules(run, 16, 0, args.seed)
    starts = batch.subsample_new_batch(rows.repeat_interleave(32))
    d = draw(run, starts, batch_size=starts.num_graphs, seed=args.seed)
    built = d['sample_batch']
    x0 = seed_excess(run, built, d['condition_id'])

    @torch.no_grad()
    def score(latents):
        """The trainer's excess energy (kT) and reduction penalty of each row rebuilt from `latents`."""
        mol = starts.clone().to(run.device)
        mol.orient_molecule(mode='standard')
        T = run.temperature * torch.ones(mol.num_graphs, dtype=torch.float32, device=run.device)
        mol, log_T, _, cid = ef.condition_samples(mol, temperature=T)
        _, out = ef.log_reward(latents.float().to(run.device), mol_batch=mol, log_temperature=log_T, return_exp=True)
        out = out.cpu().detach()
        return seed_excess(run, out, cid.cpu()), out.reduction_en.double().flatten().numpy()

    lat = built.latent_params().detach().cpu().double()
    red, info = reduced_latents(built)
    std, _ = standardize_cells(built, on_failure='flag')
    dev_a = same_crystal_deviation(built, std, info['N'], info['origin'])
    moved, ok = info['changed'] & info['ok'], info['ok']

    def image(v, w):     # an origin shift of half a cell along b and/or c: a latent shift of 1, wrapped into [-1, 1)
        out = red.clone()
        out[:, 7] = (out[:, 7] + v + 1.0) % 2.0 - 1.0
        out[:, 8] = (out[:, 8] + w + 1.0) % 2.0 - 1.0
        return out
    cases = (('latents of the built crystal, scored again', lat, np.ones(len(x0), bool)),
             ('standardised batch read as it is, changed rows', std.latent_params().detach().cpu().double(), moved),
             ("reduced cell in the trainer's chart, changed rows", red, moved),
             ('origin image, v + 1', image(1.0, 0.0), ok),
             ('origin image, w + 1', image(0.0, 1.0), ok),
             ('origin image, v + 1 and w + 1', image(1.0, 1.0), ok),
             ('canonical latents (reduced cell, folded)', torch.as_tensor(fold(red.numpy())), ok))
    r0 = score(lat)[1]
    print(f'{name}, step {run.step}: {len(x0)} draws of 16 training molecules. Draws with a reduction penalty above 1e-6: {(r0 > 1e-6).mean():.1%}; '
          f'cells changed by standardisation: {moved.mean():.1%}; not handled: {(~ok).mean():.1%}; largest atom displacement between a '
          f'draw and its standardised cell: {dev_a[moved].max() if moved.any() else 0.0:.2e} A')
    print("Caption: the trainer's excess energy (kT) of each re-description of a draw, minus the draw's own, and the share of the rows whose")
    print('reduction penalty is above 1e-6 when the trainer rebuilds them. A description of the same crystal has a zero difference.')
    print(f'{"description":>48} | {"rows":>5} | {"median |difference| (kT)":>25} | {"90th pct":>9} | {"max":>9} | {"share over 0.01 kT":>18} | '
          f'{"reduction penalty > 1e-6":>24}')
    for label, x, sel in cases:
        ex, red_pen = score(x)
        dx = np.abs(ex - x0)[sel]
        print(f'{label:>48} | {int(sel.sum()):5d} | {np.median(dx):25.2e} | {np.quantile(dx, 0.9):9.2e} | {dx.max():9.2e} | '
              f'{(dx > 0.01).mean():18.3f} | {(red_pen[sel] > 1e-6).mean():24.3f}', flush=True)


# ----------------------------------------------------------------------------------------- replay

def load_pools(out, only=None):
    """{identifier: {sampler: pool}} from the files of `collect`, with the temperature; with
    `only`, the pools of those samplers."""
    pools, T = {}, None
    for path in sorted(glob.glob(os.path.join(out, 'pools', '*', '*.pt'))):
        if only and os.path.basename(os.path.dirname(path)) not in only:
            continue
        p = torch.load(path, weights_only=False)
        T = float(p['meta']['temperature'])
        pools.setdefault(p['identifier'], {})[p['sampler']] = p
    assert pools, f'no pools under {out}'
    return pools, T


def assign_minima(e, cp):
    """Group ends into minima (E_TOL and CP_TOL from a group's first member, lowest energy
    first): each end's group index, and the groups' energies in ascending order."""
    order = np.argsort(e)
    rep_e, rep_cp, group = [], [], np.empty(len(e), dtype=int)
    for i in order:
        mine = -1
        for g in range(len(rep_e) - 1, -1, -1):     # representatives are in ascending energy: stop once out of reach
            if e[i] - rep_e[g] > E_TOL:
                break
            if abs(cp[i] - rep_cp[g]) <= CP_TOL:
                mine = g
                break
        if mine < 0:
            rep_e.append(e[i]), rep_cp.append(cp[i])
            mine = len(rep_e) - 1
        group[i] = mine
    return group, np.array(rep_e)


def settled(traj):
    """Per start: whether its relaxation had finished at the last recorded step, its best
    energy having fallen by no more than E_TOL over the last SETTLE_WINDOW steps. `traj` is
    [steps, starts]."""
    best = np.minimum.accumulate(traj.astype(np.float64), axis=0)
    return best[-min(SETTLE_WINDOW, traj.shape[0] - 1) - 1] - best[-1] <= E_TOL


def known_minima(by, samplers, T, window):
    """One molecule's minima over the settled ends of every sampler collected for it (`by`,
    {sampler: pool}): each sampler's minimum index per start, -1 for a start whose
    relaxation had not settled; the floor (the lowest energy known for the molecule, stored
    or reached here by any end); and the indices of the target minima."""
    ok = {s: settled(by[s]['traj']) for s in samplers}
    group, rep_e = assign_minima(np.concatenate([by[s]['e_end'][ok[s]] for s in samplers]),
                                 np.concatenate([by[s]['cp_end'][ok[s]] for s in samplers]))
    floor = min(min(p['e_end'].min() for p in by.values()), min(p['e_ref'] for p in by.values()))
    groups, at = {}, 0
    for s in samplers:
        groups[s] = np.full(len(ok[s]), -1)
        groups[s][ok[s]] = group[at:at + int(ok[s].sum())]
        at += int(ok[s].sum())
    return groups, floor, np.flatnonzero((rep_e - floor) / T <= window)


def stop_steps(traj, e_end, T, rule):
    """Per start: the energy calls a stopping rule spends, and whether the relaxation is then
    within CREDIT kT of where the full one ends. `traj` is [steps, starts]."""
    best = np.minimum.accumulate(traj.astype(np.float64), axis=0)
    steps = traj.shape[0]
    if rule == 'full':
        calls = np.full(traj.shape[1], steps)
    elif rule == 'plateau':
        flat = best[:-PLATEAU_WINDOW] - best[PLATEAU_WINDOW:] < PLATEAU_EPS * T     # [steps - w, starts]
        calls = np.where(flat.any(0), flat.argmax(0) + PLATEAU_WINDOW + 1, steps)
    elif rule == 'oracle':
        calls = (best <= (np.minimum(e_end, best[-1]) + CREDIT * T)[None, :]).argmax(0) + 1
    else:
        raise ValueError(rule)
    at_stop = best[calls - 1, np.arange(traj.shape[1])]
    return calls, at_stop - np.minimum(e_end, best[-1]) <= CREDIT * T, at_stop


def spend(pool, subset, how, share, stops, T):
    """One strategy on the starts drawn (`subset`, in the order drawn): the energy calls it
    spends, the starts whose minima it is credited with, and the lowest energy it holds.
    `stops` is stop_steps' result for the strategy's stopping rule."""
    calls, credited, at_stop = stops
    if how != 'cascade':
        chosen, overhead = select(pool, subset, how, share)
        return overhead + calls[chosen].sum(), chosen[credited[chosen]], at_stop[chosen].min()
    lead = subset[:max(1, len(subset) // 4)]
    rest = subset[len(lead):]
    reference = at_stop[lead].min()
    best = np.minimum.accumulate(pool['traj'].astype(np.float64), axis=0)
    stop, alive = calls[rest].copy(), np.ones(len(rest), dtype=bool)
    for step, margin in CASCADE:
        retire = alive & (stop > step) & (best[step - 1, rest] > reference + margin * T)
        stop[retire] = step
        alive &= ~retire
    held = best[stop - 1, rest].min() if len(rest) else np.inf
    return (calls[lead].sum() + stop.sum(), np.concatenate([lead[credited[lead]], rest[alive & credited[rest]]]),
            min(at_stop[lead].min(), held))


def embed(x):
    """Canonical latents as Euclidean coordinates in units of each latent's range: a periodic
    latent becomes a point on a circle whose arc length is that fraction of its range."""
    cols = []
    for k in range(x.shape[1]):
        if PERIOD[k] > 0:
            a = 2 * np.pi * x[:, k] / PERIOD[k]
            cols += [np.cos(a) / (2 * np.pi), np.sin(a) / (2 * np.pi)]
        else:
            cols.append(x[:, k] / RANGE[k])
    return np.stack(cols, axis=1)


def select(pool, subset, how, share):
    """Rows of `subset` (the starts drawn) that a selection rule relaxes, and the energy calls
    the rule itself spends."""
    k = max(1, int(np.ceil(share * len(subset))))
    if how == 'all':
        return subset, 0
    if how == 'screen':     # the screening call is the first step of the relaxations that follow
        return subset[np.argsort(pool['traj'][0, subset])[:k]], len(subset) - k
    raise ValueError(how)


def replay(pools, T, window, repeats, seed=0):
    """Every sampler under every strategy that applies to it. Returns {(sampler, strategy):
    {'draws' [points], 'calls' / 'recall' / 'best' / 'floor' [molecules, points]}}: per
    molecule, the energy calls spent, the share of the target recovered, the lowest energy in
    hand (kT above the floor) and whether that is within CREDIT kT of the floor, after that
    many draws. A pool collected start by start is replayed on
    `repeats` random subsets at each number of draws; a pool of density modes was charged
    all its draws up front and is replayed in its stored order, largest cluster first, under
    the strategies that relax every start."""
    rng = np.random.default_rng(seed)
    samplers = sorted({s for by in pools.values() for s in by})
    out, target_size, splits = {}, [], []
    for ident, by in pools.items():
        assert set(by) == set(samplers), f'{ident}: collected for {sorted(by)} only'
        groups, floor, target = known_minima(by, samplers, T, window)
        assert len(target), f'{ident}: no settled minimum within {window:g} kT of its floor, so no share of a target to report'
        target_size.append(len(target))
        splits.append(by[samplers[0]]['split'])
        for s in samplers:
            pool = by[s]
            n = len(pool['e_end'])
            mine = groups[s]
            preselected = pool.get('from_draws')
            sizes = [m for m in (8, 16, 32, 64, 128, 256, 512, 1024, 2048) if m < n] + [n]
            for lab, how, share, rule in STRATEGIES:
                if preselected and how not in ('all', 'cascade'):
                    continue
                stops = stop_steps(pool['traj'], pool['e_end'], T, rule)
                rows = {k: [] for k in ('calls', 'recall', 'best', 'floor')}
                for m in sizes:
                    acc = {k: [] for k in rows}
                    for _ in range(1 if (preselected or m == n) else repeats):
                        subset = np.arange(m) if (preselected or m == n) else rng.choice(n, m, replace=False)
                        spent, got, held = spend(pool, subset, how, share, stops, T)
                        acc['calls'].append(spent)
                        acc['recall'].append(np.isin(target, np.unique(mine[got])).mean())
                        acc['best'].append((held - floor) / T)
                        acc['floor'].append(float((held - floor) / T <= CREDIT))
                    for k in rows:
                        rows[k].append(np.mean(acc[k]))
                d = out.setdefault((s, lab), {'draws': [preselected] * len(sizes) if preselected else sizes,
                                              'calls': [], 'recall': [], 'best': [], 'floor': []})
                for k in rows:
                    d[k].append(rows[k])
    out = {key: {k: np.array(v) for k, v in d.items()} for key, d in out.items()}
    return out, np.array(target_size), np.array(splits), samplers


def selection_bounds(pools, T, window):
    """What a selection rule could save at most on each sampler's starts, and whether nearby
    starts share a minimum. Per pool collected start by start, means over molecules, every
    relaxation cut on a plateau:
      starts       starts collected
      unsettled    share of them whose full relaxation had not settled (`settled`)
      every        energy calls of relaxing every start
      distinct     minima recovered (a credited relaxation ends there)
      in_target    of those, target minima
      low          share of the starts that end in the target
      per_minimum  energy calls of the cheapest credited start of each minimum recovered: a
                   rule that knew, before any energy call, which starts share a minimum
      per_target   the same for the target minima recovered: a rule that also knew which
                   minima are low
      neighbour    chance that a start and its nearest other start (`embed` distance of their
                   canonical latents) end in one minimum
      any_two      that chance for any two starts of the molecule"""
    samplers = sorted({s for by in pools.values() for s in by})
    out = {}
    for by in pools.values():
        groups, _, target = known_minima(by, samplers, T, window)
        for s in samplers:
            pool, g = by[s], groups[s]
            if pool.get('from_draws'):
                continue
            n = len(g)
            calls, credited, _ = stop_steps(pool['traj'], pool['e_end'], T, 'plateau')
            ids = np.unique(g[credited & (g >= 0)])
            cheapest = np.array([calls[credited & (g == i)].min() for i in ids])
            hit = np.isin(ids, target)
            z = embed(pool['features'].astype(np.float64))
            d = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d, np.inf)
            same = (g[:, None] == g[None, :]) & (g[:, None] >= 0)       # an unsettled end shares a minimum with none
            row = {'starts': n, 'unsettled': (g < 0).mean(), 'every': calls.sum(), 'distinct': len(ids), 'in_target': hit.sum(),
                   'low': np.isin(g, target).mean(), 'per_minimum': cheapest.sum(), 'per_target': cheapest[hit].sum(),
                   'neighbour': same[np.arange(n), d.argmin(1)].mean() if n > 1 else np.nan,
                   'any_two': (same.sum() - (g >= 0).sum()) / (n * (n - 1)) if n > 1 else np.nan}
            for k, v in row.items():
                out.setdefault(s, {}).setdefault(k, []).append(float(v))
    return {s: {k: float(np.mean(v)) for k, v in d.items()} for s, d in out.items()}


def figures(args):
    """The curves and their table, from the pools alone."""
    from matplotlib.lines import Line2D

    from eval.cond_panel import figstyle as fs
    pools, T = load_pools(args.out, args.only)
    res, target_size, splits, samplers = replay(pools, T, args.window, args.repeats)
    F = fs.Figures(args.figures)
    greys = iter((fs.MUTED, fs.INK2, fs.AXIS))             # the random search and its variants in grey, checkpoints in colour
    hues = iter((fs.TRAIN, fs.HELD, fs.THIRD, '#8e5bd9', '#c49a00', '#d6408f'))
    colours = {s: next(greys) if s.split('+')[0] == 'random' else next(hues) for s in samplers}
    n_mol = len(target_size)
    full, plateau, oracle = (STRATEGIES[i][0] for i in (0, 1, 2))
    # one curve per sampler in every panel. A panel per strategy cut on a plateau and one for the oracle stop: recall against
    # energy calls. Then recall against draws with every step taken, and two readings of the floor under the plateau stop.
    shown = [lab for lab, _, _, rule in STRATEGIES if rule == 'plateau'] + [oracle]
    fig, axes = fs.panels(len(shown) + 3, ncols=4, width=4.1, height=3.6)
    ax_draws, ax_floor, ax_best = axes[len(shown):]
    recall_label = f'share of the minima within {args.window:g} kT recovered'
    kw = dict(marker='o', markersize=3)
    for ax, lab in zip(axes, shown):
        for s in samplers:
            if (s, lab) in res:
                ax.plot(res[(s, lab)]['calls'].mean(0), res[(s, lab)]['recall'].mean(0), color=colours[s], **kw)
        ax.set_title(lab, fontsize=8.5)
        ax.set_ylabel(recall_label)
    for s in samplers:
        ax_draws.plot(res[(s, full)]['draws'], res[(s, full)]['recall'].mean(0), color=colours[s], **kw)
        d = res[(s, plateau)]
        ax_floor.plot(d['calls'].mean(0), d['floor'].mean(0), color=colours[s], **kw)
        ax_best.plot(d['calls'].mean(0), np.median(d['best'], 0), color=colours[s], **kw)
    for ax, title, yl in ((ax_draws, full, recall_label),
                          (ax_floor, plateau, f'share of molecules with the floor in hand ({CREDIT} kT)'),
                          (ax_best, plateau, 'lowest energy in hand, kT above the floor (median)')):
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel(yl)
    span = (min(d['calls'].mean(0)[0] for d in res.values()) * 0.8, max(d['calls'].mean(0)[-1] for d in res.values()) * 1.25)
    for ax in axes:
        fs.style(ax, grid='both')
        ax.set_xscale('log')
        if ax is ax_draws:
            ax.set_xlabel('draws per molecule')
        else:
            ax.set_xlabel('energy calls per molecule')
            ax.set_xlim(*span)
        if ax is not ax_best:
            ax.set_ylim(0, 1)
    ax_best.set_yscale('symlog', linthresh=0.1)
    ax_best.set_ylim(bottom=0)
    axes[0].legend(handles=[Line2D([], [], color=colours[s], linewidth=2.5, label=s) for s in samplers], loc='upper left',
                   fontsize=8, title='sampler', title_fontsize=8.5)

    def calls_for(key, level, what='recall', mols=slice(None)):
        """Energy calls at which a curve (the mean of `what` over the molecules `mols`) reaches a level, by interpolation in
        log calls; None if it never does."""
        c, r = res[key]['calls'][mols].mean(0), res[key][what][mols].mean(0)
        return float(np.exp(np.interp(level, r, np.log(c)))) if r[0] <= level <= r[-1] else None

    resampled = np.random.default_rng(0).integers(0, n_mol, (1000, n_mol))      # the molecules drawn again with replacement

    def against(key, base, level, what='recall'):
        """The energy calls of two curves at a level and their ratio, base over key, with the 16th to 84th percentile of the
        ratio over the resampled molecules when at least four resamples in five reach the level; None if either never does."""
        a, b = calls_for(key, level, what), calls_for(base, level, what)
        if not (a and b):
            return None
        pairs = [(calls_for(key, level, what, m), calls_for(base, level, what, m)) for m in resampled]
        ratios = [y / x for x, y in pairs if x and y]
        spread = (f', {np.quantile(ratios, 0.16):.2f} to {np.quantile(ratios, 0.84):.2f} over resampled molecules'
                  if len(ratios) >= 0.8 * len(resampled) else ', no range: a curve ends near this level')
        return f'{a:,.0f} energy calls against {b:,.0f} (ratio {b / a:.2f}{spread})'

    budgets, levels = (1000, 3000, 10000, 30000), (0.25, 0.5)
    header = (['sampler', 'strategy', 'draws at the last point'] + [f'energy calls to recover {v:.0%} of the target' for v in levels] +
              ['energy calls until the floor is in hand for half the molecules (0.1 kT)'] +
              [f'recall at {b:,} energy calls' for b in budgets] +
              ['energy calls at the last point', 'recall at the last point', 'its standard error over molecules',
               'energy calls per relaxation', 'relaxations stopped within 0.1 kT of their full end'])
    rows, points = [], []
    for (s, lab), d in res.items():
        calls, recall = d['calls'].mean(0), d['recall'].mean(0)
        rule = next(r for label, _, _, r in STRATEGIES if label == lab)
        per, ok = [], []
        for by in pools.values():
            c, cr, _ = stop_steps(by[s]['traj'], by[s]['e_end'], T, rule)
            per.append(c), ok.append(cr)
        at = [round(float(np.interp(np.log(b), np.log(calls), recall)), 3) if calls[0] <= b <= calls[-1] else '' for b in budgets]
        need = [calls_for((s, lab), v) for v in levels] + [calls_for((s, lab), 0.5, 'floor')]     # 'floor': the chance it is in hand
        rows.append([s, lab, int(d['draws'][-1])] + ['' if v is None else int(round(v)) for v in need] + at +
                    [int(round(calls[-1])), round(float(recall[-1]), 3),
                    round(float(d['recall'].std(0, ddof=1)[-1] / np.sqrt(n_mol)), 3), round(float(np.concatenate(per).mean()), 1),
                    round(float(np.concatenate(ok).mean()), 3)])
    points.append(f'the target: a mean {target_size.mean():.1f} minima within {args.window:g} kT of the floor per molecule (median '
                  f'{np.median(target_size):.0f}, from {target_size.min()} to {target_size.max()}), pooled over '
                  f'{", ".join(samplers)}; it is what these starts found, not every minimum that exists')

    # every sampler against the reference sampler under the same strategy, and the reference sampler's strategies against its
    # plateau stop
    ref = 'random' if 'random' in samplers else samplers[-1]
    practical = {lab for lab, _, _, rule in STRATEGIES if rule == 'plateau'}
    for key in res:
        base = (ref, key[1] if key[0] != ref else STRATEGIES[1][0])
        if key == base or base not in res or key[1] not in practical:
            continue
        parts = [(f'{v:.0%} of the target', against(key, base, v)) for v in levels]
        parts.append(('the floor in hand for half the molecules', against(key, base, 0.5, 'floor')))
        if any(text for _, text in parts):
            points.append(f'{key[0]}, {key[1]}, against {base[0]}' + ('' if base[1] == key[1] else f', {base[1]}') + ': ' +
                          '; '.join(f'{name} for {text}' for name, text in parts if text))
    for s, b in selection_bounds(pools, T, args.window).items():
        points.append(f'{s}: {b["unsettled"]:.1%} of its {b["starts"]:.0f} relaxations had not settled at the last step. Each start '
                      f'relaxed to a plateau ({b["every"]:,.0f} energy calls), they recover '
                      f'{b["distinct"]:.1f} distinct minima, {b["in_target"]:.1f} of them in the target, where {b["low"]:.0%} of the starts '
                      f'end. The cheapest start of each minimum recovered costs {b["per_minimum"]:,.0f} energy calls together, and of each '
                      f'target minimum recovered {b["per_target"]:,.0f}: the most a selection rule could save if it knew, before any energy '
                      f'call, which starts share a minimum, and which minima are low')
        points.append(f'{s}: a start and its nearest other start in canonical latents end in one minimum with chance {b["neighbour"]:.3f}; '
                      f'any two of its starts, {b["any_two"]:.3f}')
    any_pool = next(iter(next(iter(pools.values())).values()))
    counts = sorted({len(p['e_end']) for by in pools.values() for p in by.values()})
    F.save(fig, 'csp_recall', 'Mock structure prediction: minima recovered against draws and energy calls',
           f'{n_mol} molecules ({int((splits == "train").sum())} training, {int((splits == "held_out").sum())} held-out); '
           f'{" or ".join(map(str, counts))} starts per molecule relaxed for each sampler, each once by the search schedule '
           f'({any_pool["meta"]["search_config"]}, {any_pool["traj"].shape[0]} recorded steps), and strategies replayed on the stored '
           f'per-step energies. A minimum is an end energy and packing coefficient ({E_TOL} raw units, {CP_TOL}) of a relaxation that '
           f'had settled (best energy down by at most {E_TOL} raw units over its last {SETTLE_WINDOW} steps); the end of any other '
           f'relaxation is no minimum. The target is every '
           f'minimum within {args.window:g} kT of the molecule\'s floor found by any sampler. One energy call is one energy-and-force '
           f'evaluation of one crystal. Each panel holds one curve per sampler under the strategy in its title: the first '
           f'{len(shown)} against energy calls, the next against draws, the last two the floor under the plateau stop. '
           f'A point is a number of draws; for a sampler collected start by start it is averaged over '
           f'{args.repeats} random subsets of its starts, and a "+modes" sampler drew all its draws first with no energy call, relaxed '
           f'the density mode of each of its largest clusters and is read largest cluster first. A relaxation stopped on a plateau (best energy improved by less than '
           f'{PLATEAU_EPS} kT in {PLATEAU_WINDOW} steps) is credited with its minimum only within {CREDIT} kT of its full end; the oracle '
           f'stops at the first such step. 1 kT = {T:g} raw units.', (header, rows), points)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('collect', 'figures', 'check-fold'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', action='append', default=[], metavar='NAME=CHECKPOINT', help='a checkpoint as a sampler (repeatable)')
    ap.add_argument('--skip-models', action='store_true',
                    help='collect: the --model checkpoints supply the molecules and are not collected themselves')
    ap.add_argument('--random', action='store_true', help="the search's random initialiser as a sampler")
    ap.add_argument('--prior', action='store_true',
                    help="the unconditional prior as a sampler: latents of other molecules' stored minima (the run config's prior_path)")
    ap.add_argument('--config', default=None, help='the run YAML the checkpoints were trained under')
    ap.add_argument('--search-config', default=None, help='the search YAML: its opt stages are the relaxer')
    ap.add_argument('--n-train', type=int, default=12)
    ap.add_argument('--n-test', type=int, default=12)
    ap.add_argument('--starts', type=int, default=256, help='starts relaxed per molecule from each sampler')
    ap.add_argument('--cluster-from', type=int, default=0,
                    help='collect: draw this many starts per molecule with no energy call and relax the density modes of the '
                         '--starts largest clusters (tens of thousands of draws)')
    ap.add_argument('--batch-rows', type=int, default=512,
                    help='crystals relaxed in one call at most; molecules are taken as many at a time as fit in it, at least one')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--window', type=float, default=3.0, help='kT above the floor that defines the target')
    ap.add_argument('--repeats', type=int, default=20, help='random subsets of the starts per point')
    ap.add_argument('--figures', default=None, help='figure directory')
    ap.add_argument('--only', nargs='+', default=None, help='figures: the samplers (pool names) to replay; default all collected')
    args = ap.parse_args()
    if args.mode == 'collect':
        os.makedirs(args.out, exist_ok=True)
        collect(args)
    elif args.mode == 'check-fold':
        check_fold(args)
    else:
        figures(args)


if __name__ == '__main__':
    main()
