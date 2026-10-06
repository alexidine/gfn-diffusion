"""Pre-train the atoms-visible conformer energy model (models/conformer_trunk.py) on the sampler's potential.

    python -u pretrain_conformer_trunk.py --rung <condition set dir> --out-dir <dir> --name <arm>

WHAT IS FITTED. The force-field part of the trainer's potential under the trainer's clip,
clip(U), minus the molecule's potential at its reference conformer, in kcal/mol (= kT at
T = 1), and its gradient with respect to the state. Left out of the target:
  * the stereo lock, which the trainer adds to U before the clip. It holds every
    four-neighbour centre and stereo double bond to its labelled handedness, which the model
    cannot tell from the other one (models/conformer_trunk.py). It is zero while every centre
    is on its own side, as nearly all thermal states are and some bridge states are not; the
    share of each test set carrying one is printed;
  * the box wall and the measure terms, functions of the state and not of the geometry.
The model's gradient is its energy differentiated through the build, so fitting it trains
the energy.

DATA, SPLIT AND ENERGY TABLE are models/energy_probe.py's, through its own functions: the
same condition set, the same held-out molecules and held-out stored conformers, the same
three test sets (seen, conf, mol) of stored rows displaced by their thermal widths, and the
same error table. A number here and a number there are on one footing.

STATES. A training state is, with probability --bridge-share, a BRIDGE state: t/T of the way
from the reference conformer (state 0) to a stored row, t uniform on 0..T, plus the sampler's
reference noise sqrt(m t_scale (t/T)(1 - t/T)) with m drawn from --noise-scales. Otherwise it
is energy_probe's thermal state: a stored row displaced by its coordinates' thermal widths
times a multiplier log-uniform over --noise-range. Bridge states of held-out molecules at
fixed t/T are scored beside the three test sets.

LOSS. energy_probe's energy term, plus --force-weight times the gradient term: per row, the
mean squared error of the gradient over (the mean squared reference gradient + 1), every
coordinate scaled by its thermal width (so a coordinate counts by the energy it moves over
its own thermal range, and a stiff bond does not outvote a torsion). --force-weight 0 fits
the energy alone and skips the reference gradient.

A run resumes from <out-dir>/<name>_running.pt when that file exists.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import yaml
from torch import nn

from models.conformer_trunk import MIN_DISTANCE, ConformerTrunk, MoleculeTopology
from models.energy_probe import (BAND, LOSS_SCALE, TABLE_CAPTION, build_energy, load_rung, metrics,
                                 molecule_split, table_header, table_row, thermal_sigma)

FORCE_CAPTION = (
    'Gradient of the potential with respect to the state: the model (its energy differentiated '
    'through the build) against the trainer. Rows of the same three test sets, then bridge states '
    'of held-out molecules at the t/T named (reference noise at 1x). cos and rel: per row, the '
    'cosine and |error| / |reference|, every coordinate scaled by its thermal width; medians over '
    'rows. step err/ref: sqrt(t_scale / T) times the RMS over a row\'s coordinates of the gradient '
    'error / of the reference gradient, i.e. what a full-strength force term moves the mean by in '
    'one step, in units of that step\'s noise standard deviation; medians. Compare err with ref.')


# ----------------------------------------------------------------------------- target

def potential_and_gradient(multi, sub, x, need_gradient=True, with_lock=False):
    """(clip(U) [B], its gradient with respect to the state [B, K] or None, lock [B]), kcal/mol;
    clip(U + lock), the trainer's own clipped potential, under `with_lock`.

    MultiConformerTorsions._energy_one_pass up to its clip, repeated here because that method
    returns this part detached and with the lock in it. `check_potential` holds the
    `with_lock` form to the trainer's own value.
    """
    from mxtaltools.common.utils import log_rescale_positive
    from mxtaltools.conformers.builder import build
    from mxtaltools.conformers.energy import intramolecular_energy

    from energies.conformer_data import batch_tree, dummy_frame_mask, state_to_dof, transverse_mask

    lib_ids, ptr = multi._resolve_rows(sub, int(x.shape[0]), x)
    xg = x.detach().to(multi.dtype).requires_grad_(need_gradient)
    with torch.enable_grad() if need_gradient else torch.no_grad():
        tree = batch_tree(sub)
        pos = build(tree, *state_to_dof(sub, xg), transverse=transverse_mask(sub),
                    dummy_frame=dummy_frame_mask(sub))
        e = intramolecular_energy(tree, pos, multi._lib.gather(lib_ids, ptr[:-1]))
        lock = torch.zeros_like(e)
        if multi.stereo_coeff > 0.0:
            from energies.stereo_lock import batch_lock_energy
            lock = batch_lock_energy(sub, pos, multi.stereo_coeff)
            if with_lock:
                e = e + lock
        if multi.energy_clip is not None:
            cutoff = (multi.energy_clip if multi.energy_clip_origin == 'absolute'
                      else multi._clip_floor_of_lib.index_select(0, lib_ids) + multi.energy_clip)
            e = log_rescale_positive(e, cutoff)
        grad = torch.autograd.grad(e.sum(), xg)[0] if need_gradient else None
    return e.detach(), grad, lock.detach()


def check_potential(multi, sub, x, what):
    """Raise unless `potential_and_gradient`, lock included, is the trainer's baked potential
    on these in-box states.

    The baked value is clip(U + lock) + wall. Inside the box the wall is zero except the disc
    wall of a transverse centre, so rows without one must agree and the others may only exceed.
    """
    mine = potential_and_gradient(multi, sub, x, need_gradient=False, with_lock=True)[0]
    with torch.no_grad():
        baked = multi.energy(x, sub, return_exp=True)[1].conformer_energy.flatten().to(mine.dtype)
    tv = getattr(sub, 'ctree_transverse', None)
    has_tv = torch.zeros_like(mine, dtype=torch.bool)
    if tv is not None and bool(tv.any()):
        has_tv = torch.zeros_like(mine).index_add_(0, sub.batch, tv.to(mine.dtype)) > 0
    diff = baked - mine
    tol = 1e-3 + 1e-4 * mine.abs()
    if bool((diff[~has_tv].abs() > tol[~has_tv]).any()) or bool((diff[has_tv] < -tol[has_tv]).any()):
        raise SystemExit(f'{what}: the target potential is not the trainer\'s. Largest |difference| '
                         f'{float(diff.abs().max()):.3e} kcal/mol on {int(mine.numel())} rows '
                         f'({int(has_tv.sum())} with a transverse centre). The trainer\'s potential '
                         f'(energies/multi_conformer.py::_energy_one_pass) has moved; '
                         f'potential_and_gradient must follow it.')
    return float(diff[~has_tv].abs().max()) if bool((~has_tv).any()) else 0.0, int(mine.numel())


# ---------------------------------------------------------------------------- scoring

def gradient_terms(pred, ref, sigma):
    """Per row: (mean squared thermal-scaled error, mean squared thermal-scaled reference, cosine,
    mean squared error, mean squared reference); means over the row's own coordinates."""
    live = (sigma > 0).to(pred.dtype)
    count = live.sum(1).clamp_min(1.0)
    u, v, w = (pred - ref) * sigma, ref * sigma, pred * sigma
    cos = (w * v).sum(1) / ((w.square().sum(1) * v.square().sum(1)).sqrt() + 1e-12)
    return (u.square().sum(1) / count, v.square().sum(1) / count, cos,
            ((pred - ref) * live).square().sum(1) / count, (ref * live).square().sum(1) / count)


def force_loss(pred, ref, sigma):
    err, size = gradient_terms(pred, ref, sigma)[:2]
    return (err / (size + 1.0)).mean()


def force_metrics(pred, ref, sigma, step_sd):
    err, size, cos, raw_err, raw_size = gradient_terms(pred, ref, sigma)
    med = lambda v: float(v.double().median())
    return {'cos': med(cos), 'rel': med((err / size.clamp_min(1e-12)).sqrt()),
            'step_err': med(step_sd * raw_err.sqrt()), 'step_ref': med(step_sd * raw_size.sqrt())}


def force_header(names):
    return ' '.join([f'{"step":>8}'] + [f'{n + " cos":>10} {"rel":>6} {"step err/ref":>13}' for n in names])


def force_row(step, res, names):
    cells = [f'{step:>8d}']
    for n in names:
        r = res[n]
        cells.append(f"{r['cos']:>10.4f} {r['rel']:>6.3f} {r['step_err']:>6.3f}/{r['step_ref']:<6.3f}")
    return ' '.join(cells)


# ------------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rung', required=True, help='condition set directory')
    ap.add_argument('--config', default='configs/conformer_mk.yaml',
                    help='the run config: its energy_config builds the energy, its model.t_scale and integrator.T '
                         'the bridge')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--node-dim', type=int, default=128)
    ap.add_argument('--message-dim', type=int, default=64)
    ap.add_argument('--num-convs', type=int, default=3)
    ap.add_argument('--num-radial', type=int, default=48)
    ap.add_argument('--cutoff', type=float, default=10.0, help='Angstrom; a pair beyond it has no edge')
    ap.add_argument('--force-weight', type=float, default=1.0, help='0 = fit the energy alone')
    ap.add_argument('--bridge-share', type=float, default=0.5, help='share of training states drawn on the bridge')
    ap.add_argument('--noise-scales', default='1,2,4',
                    help='multipliers on the bridge variance, one drawn per bridge state')
    ap.add_argument('--bridge-eval', type=float, nargs='*', default=[0.3, 0.5, 0.7, 0.9],
                    help='t/T of the bridge test sets')
    ap.add_argument('--bridge-eval-samples', type=int, default=5000)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=150000)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lr-final', type=float, default=3e-6, help='cosine floor')
    ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--noise-range', type=float, nargs=2, default=[0.3, 2.0],
                    help='thermal-width multiplier per thermal row, log-uniform over this range')
    ap.add_argument('--heldout-frac', type=float, default=0.10)
    ap.add_argument('--conformer-frac', type=float, default=0.10)
    ap.add_argument('--eval-samples', type=int, default=20000)
    ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--save-every', type=int, default=5000)
    ap.add_argument('--max-conditions', type=int, default=None, help='first N conditions only')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--vram-fraction', type=float, default=None)
    ap.add_argument('--wandb-project', default=None)
    a = ap.parse_args(argv)

    dev = torch.device(a.device)
    torch.set_default_dtype(torch.float32)
    if dev.type == 'cuda' and a.vram_fraction:
        torch.cuda.set_per_process_memory_fraction(a.vram_fraction)
    os.makedirs(a.out_dir, exist_ok=True)
    print('arguments: ' + json.dumps(vars(a)), flush=True)

    run_cfg = yaml.safe_load(open(a.config))
    t_scale, T = float(run_cfg['model']['t_scale']), int(run_cfg['integrator']['T'])
    step_sd = math.sqrt(t_scale / T)
    mults = torch.tensor([float(v) for v in a.noise_scales.split(',')], device=dev)

    cond, idents, smiles, row_slot, states = load_rung(a.rung, a.max_conditions)
    C, (n_rows, K) = len(idents), states.shape
    multi, ref, ec = build_energy(cond, idents, smiles, a.config, dev)
    sigma = thermal_sigma(cond, C, K, ec)
    periodic = torch.as_tensor(multi.periodic_dims, dtype=torch.bool)
    valid = cond.state_mask.reshape(C, K).bool()
    topology = MoleculeTopology(cond, dev)

    # --- split: energy_probe's
    held_mol = molecule_split(smiles, a.heldout_frac)
    g = torch.Generator().manual_seed(a.seed)
    held_row = torch.rand(n_rows, generator=g) < a.conformer_frac
    row_mol_out = held_mol[row_slot]
    pools = {'train': torch.nonzero(~row_mol_out & ~held_row).flatten(),
             'conf': torch.nonzero(~row_mol_out & held_row).flatten(),
             'mol': torch.nonzero(row_mol_out).flatten()}
    print(f'{C} conditions ({len(set(smiles))} distinct SMILES), {n_rows} stored rows, width {K}, up to '
          f'{topology.A} atoms. Held-out molecules: {int(held_mol.sum())} conditions. Rows -- train '
          f'{len(pools["train"])}, held-out conformers {len(pools["conf"])}, held-out molecules '
          f'{len(pools["mol"])}. Bridge: t_scale {t_scale:g}, T {T}', flush=True)
    if min(len(v) for v in pools.values()) == 0:
        raise SystemExit('an empty split; the set is too small for these fractions')

    # the file is written in float64 and the energy runs in float32 (ConformerModeller._as_run_dtype)
    for key, val in list(cond._store.items()):
        if torch.is_tensor(val) and val.is_floating_point() and val.dtype != torch.float32:
            cond[key] = val.float()
    cond = cond.to(dev)
    states, sigma, valid, row_slot_d = states.to(dev), sigma.to(dev), valid.to(dev), row_slot.to(dev)
    ref, periodic = ref.to(dev), periodic.to(dev)
    lo, hi = math.log10(a.noise_range[0]), math.log10(a.noise_range[1])

    def boxed(x, rows, slots):
        x = torch.where(periodic, (x + 1) % 2 - 1, x.clamp(-1, 1))
        return x * (sigma[slots] > 0) + states[rows] * (sigma[slots] == 0)

    def thermal_states(rows, gen):
        """energy_probe's draw: the same two calls on the generator, in the same order."""
        slots = row_slot_d[rows]
        mult = 10 ** (lo + (hi - lo) * torch.rand(len(rows), 1, device=dev, generator=gen))
        x = states[rows] + mult * sigma[slots] * torch.randn(len(rows), K, device=dev, generator=gen)
        return boxed(x, rows, slots)

    def bridge_states(rows, frac, mult, gen):
        """t/T = frac of the way from the reference conformer to each stored row, plus the
        reference bridge's noise at `mult` times its variance."""
        slots = row_slot_d[rows]
        sd = (mult * t_scale * frac * (1 - frac)).sqrt()
        x = (states[rows] * frac + sd * torch.randn(len(rows), K, device=dev, generator=gen)) * valid[slots]
        return boxed(x, rows, slots)

    def labelled(rows, x, need_gradient):
        slots = row_slot_d[rows]
        sub = cond.subsample_new_batch(slots)
        e, grad, lock = potential_and_gradient(multi, sub, x, need_gradient)
        return sub, slots, e.float() - ref[slots], None if grad is None else grad.float(), lock

    # --- the fixed test sets: the same draws on every start
    eg = torch.Generator(device=dev).manual_seed(a.seed + 1)
    cg = torch.Generator().manual_seed(a.seed + 2)
    tests, checked = {}, [0.0, 0]

    def add_test(name, pool, n, make):
        rows = pool[torch.randint(0, len(pool), (n,), generator=cg)].to(dev)
        chunks = []
        for i in range(0, n, a.batch):
            r = rows[i:i + a.batch]
            x = make(r)
            sub, slots, y, grad, lock = labelled(r, x, True)
            if i == 0:
                worst, count = check_potential(multi, sub, x, f'test set {name}')
                checked[0], checked[1] = max(checked[0], worst), checked[1] + count
            chunks.append((x, slots, y, grad, lock))
        tests[name] = chunks
        y = torch.cat([c[2] for c in chunks])
        q = torch.quantile(y.double().cpu(), torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
        clip = '' if multi.energy_clip is None or multi.energy_clip_origin != 'reference' else (
            f', {100 * float((y > multi.energy_clip).float().mean()):.1f}% in the compressed range above the clip')
        print(f'test set {name}: {len(y)} samples, target p5 {q[0]:.1f} p50 {q[1]:.1f} p95 {q[2]:.1f} max '
              f'{float(y.max()):.1f} kcal/mol, {100 * float((y < BAND).float().mean()):.0f}% below {BAND:g}{clip}, '
              f'{100 * float((torch.cat([c[4] for c in chunks]) > 1e-6).float().mean()):.2f}% with a nonzero '
              f'stereo lock', flush=True)

    for name, pool in (('seen', pools['train']), ('conf', pools['conf']), ('mol', pools['mol'])):
        add_test(name, pool, a.eval_samples, lambda r: thermal_states(r, eg))
    bridge_names = []
    for frac in a.bridge_eval:
        bridge_names.append(f't{frac:g}')
        add_test(bridge_names[-1], pools['mol'], a.bridge_eval_samples,
                 lambda r, f=frac: bridge_states(r, torch.full((len(r), 1), f, device=dev),
                                                 torch.ones(len(r), 1, device=dev), eg))
    print(f'target potential, with the stereo lock put back, against the trainer\'s baked potential on '
          f'{checked[1]} test rows: largest |difference| {checked[0]:.2e} kcal/mol', flush=True)

    model = ConformerTrunk(node_dim=a.node_dim, message_dim=a.message_dim, num_convs=a.num_convs,
                           num_radial=a.num_radial, cutoff=a.cutoff, energy_scale=LOSS_SCALE).to(dev)
    print(f'model: {int(sum(p.numel() for p in model.parameters()))} parameters; {model.args}', flush=True)

    # how much of the geometry the cutoff and the distance floor leave out, on the test states
    with torch.no_grad():
        from energies.conformer_data import states_to_positions
        out_n = in_n = low_n = 0
        for name in ('mol', *bridge_names):
            x, slots = tests[name][0][0], tests[name][0][1]
            pos = states_to_positions(cond.subsample_new_batch(slots), x)
            ei = topology.batch(slots)[0]
            d = (pos[ei[0]] - pos[ei[1]]).norm(dim=1)
            out_n, in_n, low_n = out_n + int((d > a.cutoff).sum()), in_n + d.numel(), low_n + int((d < MIN_DISTANCE).sum())
    print(f'pairs on {1 + len(bridge_names)} test batches: {100 * out_n / in_n:.3f}% beyond the {a.cutoff:g} A '
          f'cutoff (no edge), {100 * low_n / in_n:.4f}% closer than {MIN_DISTANCE:g} A (presented at that '
          f'distance)', flush=True)

    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.0)

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        frac = (step - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr_final + 0.5 * (a.lr - a.lr_final) * (1 + math.cos(math.pi * min(1.0, frac)))

    run_path = os.path.join(a.out_dir, f'{a.name}_running.pt')
    hist_path = os.path.join(a.out_dir, f'{a.name}_history.jsonl')
    step0 = 0
    tg = torch.Generator(device=dev).manual_seed(a.seed + 3)
    ig = torch.Generator().manual_seed(a.seed + 4)
    if os.path.exists(run_path):
        ck = torch.load(run_path, map_location=dev, weights_only=False)
        if ck['trunk_args'] != model.args:
            raise SystemExit(f'{run_path} holds a model built with {ck["trunk_args"]}; this run asks for {model.args}')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        tg.set_state(ck['noise_generator'].cpu())
        ig.set_state(ck['index_generator'].cpu())
        step0 = int(ck['step'])
        print(f'RESUMED {run_path} at step {step0}', flush=True)
    wb = None
    if a.wandb_project:
        # the curves are a convenience: the tables below and the history file are the record
        try:
            import wandb
            wb = wandb.init(project=a.wandb_project, name=a.name, id=a.name, resume='allow',
                            tags=['conformer_trunk'], config=vars(a))
        except Exception as exc:                                  # noqa: BLE001
            print(f'wandb unavailable ({type(exc).__name__}: {exc}); continuing without it', flush=True)

    def evaluate():
        model.eval()
        energy, force = {}, {}
        for name, chunks in tests.items():
            pr, ys, sl, gp, gt, sg = [], [], [], [], [], []
            for x, slots, y, grad, _ in chunks:
                e, gpred, _ = model.energy_and_gradient(x, cond.subsample_new_batch(slots), slots, topology)
                pr.append(e.detach()); ys.append(y); sl.append(slots)
                gp.append(gpred.detach()); gt.append(grad); sg.append(sigma[slots])
            energy[name] = metrics(torch.cat(pr).cpu(), torch.cat(ys).cpu(), torch.cat(sl).cpu())
            force[name] = force_metrics(torch.cat(gp), torch.cat(gt), torch.cat(sg), step_sd)
        model.train()
        return energy, force

    def save(step):
        tmp = run_path + '.tmp'
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step,
                    'noise_generator': tg.get_state().cpu(), 'index_generator': ig.get_state().cpu(),
                    'trunk_args': model.args, 'args': vars(a)}, tmp)
        os.replace(tmp, run_path)

    force_names = ['seen', 'conf', 'mol', *bridge_names]
    print('\nENERGY (lines E). ' + TABLE_CAPTION + f' Set: {a.rung}; arm {a.name}; {a.eval_samples} samples per '
          f'test set.')
    print('E ' + table_header())
    print('\nFORCE (lines F). ' + FORCE_CAPTION + f' t_scale {t_scale:g}, T {T}; {a.bridge_eval_samples} samples '
          f'per bridge set.')
    print('F ' + force_header(force_names), flush=True)
    model.train()
    train_pool = pools['train']
    t_last, s_last = time.time(), step0
    acc = {'loss': 0.0, 'energy': 0.0, 'force': 0.0, 'n': 0}
    energy = force = None
    for step in range(step0, a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            energy, force = evaluate()
            rate = (step - s_last) / max(time.time() - t_last, 1e-9)
            mean = {k: acc[k] / max(acc['n'], 1) for k in ('loss', 'energy', 'force')}
            print('E ' + table_row(step, mean['energy'], energy, rate))
            print('F ' + force_row(step, force, force_names), flush=True)
            with open(hist_path, 'a') as f:
                f.write(json.dumps({'step': step, 'loss': mean['loss'], 'energy_loss': mean['energy'],
                                    'force_loss': mean['force'], 'it_per_s': rate, 'lr': lr_at(step),
                                    'energy': energy, 'force': force}) + '\n')
            if wb is not None:
                wb.log({'loss': mean['loss'], 'energy_loss': mean['energy'], 'force_loss': mean['force'],
                        'lr': lr_at(step), 'it_per_s': rate,
                        **{f'{n_}/{k_}': v for n_, r in energy.items() for k_, v in r.items()},
                        **{f'{n_}/force_{k_}': v for n_, r in force.items() for k_, v in r.items()}}, step=step)
            t_last, s_last = time.time(), step
            acc = {'loss': 0.0, 'energy': 0.0, 'force': 0.0, 'n': 0}
        if step == a.steps:
            break
        for grp in opt.param_groups:
            grp['lr'] = lr_at(step)
        rows = train_pool[torch.randint(0, len(train_pool), (a.batch,), generator=ig)].to(dev)
        x = thermal_states(rows, tg)
        if a.bridge_share > 0:
            on_bridge = torch.rand(len(rows), 1, device=dev, generator=tg) < a.bridge_share
            frac = torch.randint(0, T + 1, (len(rows), 1), device=dev, generator=tg).float() / T
            mult = mults[torch.randint(0, len(mults), (len(rows), 1), device=dev, generator=tg)]
            x = torch.where(on_bridge, bridge_states(rows, frac, mult, tg), x)
        fit_force = a.force_weight > 0
        sub, slots, y, grad, _ = labelled(rows, x, fit_force)
        opt.zero_grad()
        if fit_force:
            pred, gpred, _ = model.energy_and_gradient(x, sub, slots, topology, create_graph=True)
            f_loss = force_loss(gpred, grad, sigma[slots])
        else:
            from energies.conformer_data import states_to_positions
            pred = model(states_to_positions(sub, x), sub, slots, topology)['energy']
            f_loss = torch.zeros((), device=dev)
        e_loss = nn.functional.smooth_l1_loss(pred / LOSS_SCALE, y / LOSS_SCALE, beta=0.2)
        loss = e_loss + a.force_weight * f_loss
        if not torch.isfinite(loss):
            raise SystemExit(f'non-finite loss at step {step}')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        acc['loss'] += float(loss.detach()); acc['energy'] += float(e_loss.detach())
        acc['force'] += float(f_loss.detach()); acc['n'] += 1
        if (step + 1) % a.save_every == 0:
            save(step + 1)
    save(a.steps)
    final = os.path.join(a.out_dir, f'{a.name}_final.pt')
    torch.save({'model': model.state_dict(), 'trunk_args': model.args, 'args': vars(a),
                'energy': energy, 'force': force, 'step': a.steps}, final)
    print(f'wrote {final}', flush=True)


if __name__ == '__main__':
    main()
