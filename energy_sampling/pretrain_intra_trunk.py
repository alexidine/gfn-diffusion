"""Pre-train the intra trunk (models/intra_trunk.py) on intramolecular energies and forces.

    python -u pretrain_intra_trunk.py --set <build_intra_energy_set.py output> --out <dir> --tag <name>

WHAT IS FITTED. The force field's total intramolecular energy of a geometry, as a sum of
per-atom terms, and its Cartesian force, the model's energy differentiated with respect to
the positions. Nothing in the target is clipped and no term of the force field is treated
apart from another.

UNITS AND REFERENCES, fixed from the training molecules before the first step and stored in
the checkpoint (the convention of machine-learned potentials):
  * `e0`, one reference energy per element: the least-squares fit of the relaxed minima's
    energies to their element counts, no intercept. The model fits what is left.
  * `scale`, the root mean square force component over geometries displaced by at most
    --scale-noise Angstrom. The network's per-atom output is in these units (times 1 A).
  * `energy_scale`, the root mean square of the per-atom energy left after `e0` on the same
    geometries: the unit the energy error is counted in.

LOSS, per geometry, averaged over the batch:
  * force: the mean squared error of the force components over (their mean squared reference
    + 1), both in units of `scale`. Geometries here run from relaxed minima (zero force) to
    displacements of 0.2 A (a thousand kcal/mol/A); a plain squared error would be fitted to
    the hot ones alone, a purely relative one is undefined at a minimum.
  * energy: --energy-weight times the Huber loss of the per-atom energy error in units of
    `energy_scale`. The forces fix the energy within a basin; this term fixes the level of
    each molecule and of each conformer against the others.

SPLIT. Molecules the set marks `heldout` (the crystal work's held-out molecules) are never
trained on. Each evaluation scores a fixed sample of their geometries and an equal sample
of training molecules' geometries, by displacement width, and the energy difference between
two relaxed conformers of a held-out molecule.

The checkpoint `<tag>.pt` holds the averaged weights (--ema) as `model`, with `trunk_args`
to rebuild the trunk (`models.intra_trunk.load_intra_trunk`). A run resumes from
`<tag>_running.pt` when that file exists, and only under the arguments it was written with.
"""
import argparse
import copy
import json
import math
import os
import time

import torch

from models.intra_trunk import IntraTrunk

#: arguments a resumed run must share with the run that wrote the checkpoint
RESUME_KEYS = ('set', 'seed', 'batch', 'steps', 'lr', 'lr_final', 'warmup', 'energy_weight', 'max_noise',
               'scale_noise', 'ema', 'node_dim', 'message_dim', 'num_convs', 'num_radial', 'cutoff')

CAPTION = (
    "{what}: the trunk's energy and force against MMFF94 on {n} geometries per split, {mols} molecules, averaged "
    "weights. Rows are displacement widths (0 = relaxed minima). Compare a split's rows with each other and the two "
    "splits row by row: held-out molecules were never trained on. Force error is per component; the relative error "
    "and the cosine are medians over geometries of the whole-molecule force; at a minimum the force is zero and both "
    "are left out. The last line is the energy difference between two relaxed conformers of one held-out molecule."
)


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True, help='build_intra_energy_set.py output')
    ap.add_argument('--out', required=True, help='directory for the checkpoint and the evaluation json')
    ap.add_argument('--tag', required=True, help='file stem of this run inside --out')
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--batch', type=int, default=1024, help='geometries per step')
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--lr-final', type=float, default=1e-5, help='cosine floor')
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--energy-weight', type=float, default=1.0)
    ap.add_argument('--max-noise', type=float, default=0.2, help='train on displacement widths up to this (A)')
    ap.add_argument('--scale-noise', type=float, default=0.05, help='widths up to this (A) set the units')
    ap.add_argument('--ema', type=float, default=0.999, help='decay of the averaged weights')
    ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--eval-geometries', type=int, default=20000, help='geometries scored per split')
    ap.add_argument('--save-every', type=int, default=2500, help='steps between running checkpoints')
    ap.add_argument('--node-dim', type=int, default=128)
    ap.add_argument('--message-dim', type=int, default=64)
    ap.add_argument('--num-convs', type=int, default=3)
    ap.add_argument('--num-radial', type=int, default=48)
    ap.add_argument('--cutoff', type=float, default=10.0, help='range (A) of the radial basis; every pair keeps its edge')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--vram-frac', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args(argv)


class EnergySet:
    """The set on one device; a batch of geometries is a gather."""

    def __init__(self, path, device):
        blob = torch.load(path, map_location='cpu', weights_only=False)
        self.build = blob['build']
        self.heldout = blob['heldout'].to(device)
        self.z = blob['z'].to(device)
        self.mol_ptr = blob['mol_ptr'].to(device)
        self.geom_mol = blob['geom_mol'].to(device)
        self.geom_ptr = blob['geom_ptr'].to(device)
        self.geom_n = self.geom_ptr[1:] - self.geom_ptr[:-1]
        self.pos = blob['pos'].to(device)
        self.force = blob['force'].to(device)
        self.energy = blob['energy'].to(device)            # float64: the reference is taken off before the cast
        self.noise = blob['noise'].to(device)
        self.conformer = blob['conformer'].to(device)
        self.num_molecules = int(self.heldout.numel())
        self.num_geometries = int(self.energy.numel())
        counts = torch.zeros(self.num_molecules, 101, dtype=torch.float64, device=device)
        atom_mol = torch.repeat_interleave(torch.arange(self.num_molecules, device=device),
                                           self.mol_ptr[1:] - self.mol_ptr[:-1])
        counts.index_put_((atom_mol, self.z), torch.ones_like(self.z, dtype=torch.float64), accumulate=True)
        self.counts = counts
        self.excess = None

    def fit_reference(self, geometries):
        """Per-element reference energies [101] from `geometries`: least squares on element counts."""
        a = self.counts[self.geom_mol[geometries]]
        present = a.sum(0) > 0
        sol = torch.linalg.lstsq(a[:, present], self.energy[geometries, None]).solution.flatten()
        e0 = torch.zeros(101, dtype=torch.float64, device=a.device)
        e0[present] = sol
        return e0

    def set_reference(self, e0):
        self.excess = (self.energy - self.counts[self.geom_mol] @ e0).float()

    def batch(self, g):
        """Geometries g [B] -> (z [B, A], pos [B, A, 3], force [B, A, 3], mask [B, A], excess energy [B], n [B])."""
        n = self.geom_n[g]
        ar = torch.arange(int(n.max()), device=g.device)
        mask = ar[None] < n[:, None]
        at = (self.geom_ptr[g][:, None] + ar[None]).clamp(max=self.pos.shape[0] - 1)
        zat = (self.mol_ptr[self.geom_mol[g]][:, None] + ar[None]).clamp(max=self.z.shape[0] - 1)
        keep = mask[..., None]
        return (self.z[zat] * mask, self.pos[at] * keep, self.force[at] * keep, mask, self.excess[g], n)


def losses(model, batch, scale, energy_scale, energy_weight, training):
    """(loss, force term, energy term, predicted excess energy [B], predicted force [B, A, 3])."""
    z, pos, force, mask, excess, n = batch
    _, f_pred, out = model.energy_and_force(z, pos, mask, create_graph=training)
    e_pred = model.scale * torch.zeros_like(excess).index_add_(0, out['node_graph'], out['e_model'])
    comps = 3.0 * n
    err = ((f_pred - force) / scale).square().sum((1, 2)) / comps
    ref = (force / scale).square().sum((1, 2)) / comps
    f_loss = (err / (ref + 1.0)).mean()
    e_loss = torch.nn.functional.huber_loss((e_pred - excess) / n / energy_scale, torch.zeros_like(excess), delta=1.0)
    return f_loss + energy_weight * e_loss, f_loss, e_loss, e_pred, f_pred


def evaluate(model, data, geometries, scale, energy_scale, chunk=2048):
    """Per displacement width over `geometries`: errors in kcal/mol and kcal/mol/A."""
    rows = {}
    acc = {}
    for lo in range(0, geometries.numel(), chunk):
        g = geometries[lo:lo + chunk]
        batch = data.batch(g)
        z, pos, force, mask, excess, n = batch
        _, _, _, e_pred, f_pred = losses(model, batch, scale, energy_scale, 1.0, training=False)
        f_pred = f_pred.detach()
        diff = (f_pred - force).square().sum((1, 2))
        fsq = force.square().sum((1, 2))
        dot = (f_pred * force).sum((1, 2))
        cos = dot / (f_pred.square().sum((1, 2)).sqrt() * fsq.sqrt()).clamp(min=1e-12)
        for key, val in (('noise', data.noise[g]), ('n', n.float()), ('diff', diff), ('fsq', fsq), ('cos', cos),
                         ('abs_e', (e_pred.detach() - excess).abs() / n)):
            acc.setdefault(key, []).append(val)
    acc = {k: torch.cat(v) for k, v in acc.items()}
    for width in sorted(set(acc['noise'].tolist())):
        sel = acc['noise'] == width
        comps = 3.0 * acc['n'][sel]
        row = {'geometries': int(sel.sum()),
               'force_rmse': float((acc['diff'][sel].sum() / comps.sum()).sqrt()),
               'force_rms': float((acc['fsq'][sel].sum() / comps.sum()).sqrt()),
               'energy_mae_per_atom': float(acc['abs_e'][sel].mean())}
        if width > 0:
            row['force_relative'] = float((acc['diff'][sel] / acc['fsq'][sel].clamp(min=1e-12)).sqrt().median())
            row['force_cosine'] = float(acc['cos'][sel].median())
        rows[f'{width:g}'] = row
    return rows


def conformer_gaps(model, data, pairs, scale, energy_scale, chunk=2048):
    """Energy difference between two relaxed conformers of a molecule: error and size, kcal/mol."""
    if pairs.numel() == 0:
        return None
    pred = []
    for lo in range(0, pairs.shape[0], chunk):
        p = pairs[lo:lo + chunk]
        e = [losses(model, data.batch(p[:, k]), scale, energy_scale, 1.0, training=False)[3].detach() for k in (0, 1)]
        pred.append(e[1] - e[0])
    pred = torch.cat(pred)
    true = (data.excess[pairs[:, 1]] - data.excess[pairs[:, 0]])
    return {'pairs': int(pairs.shape[0]), 'gap_mae': float((pred - true).abs().mean()),
            'gap_rms': float(true.square().mean().sqrt())}


def print_eval(step, results, gaps, a):
    n = sum(r['geometries'] for r in next(iter(results.values())).values())
    print(CAPTION.format(what=f'Evaluation at step {step}', n=n, mols='QM9'), flush=True)
    print('split | displacement (A) | geometries | force RMS error (kcal/mol/A) | reference force RMS (kcal/mol/A) | '
          'relative force error | force cosine | energy error per atom (kcal/mol)', flush=True)
    for split, rows in results.items():
        for width, r in rows.items():
            rel = f"{r['force_relative']:.4f}" if 'force_relative' in r else '-'
            cos = f"{r['force_cosine']:.5f}" if 'force_cosine' in r else '-'
            print(f"{split} | {width} | {r['geometries']} | {r['force_rmse']:.3f} | {r['force_rms']:.2f} | {rel} | "
                  f"{cos} | {r['energy_mae_per_atom']:.4f}", flush=True)
    if gaps is not None:
        print(f"conformer energy difference, held-out molecules, {gaps['pairs']} pairs: mean absolute error "
              f"{gaps['gap_mae']:.3f} kcal/mol against a root mean square difference of {gaps['gap_rms']:.3f}",
              flush=True)


def main(argv=None):
    a = parse(argv)
    dev = torch.device(a.device)
    if dev.type == 'cuda' and a.vram_frac:
        torch.cuda.set_per_process_memory_fraction(a.vram_frac)
    os.makedirs(a.out, exist_ok=True)
    running = os.path.join(a.out, f'{a.tag}_running.pt')

    data = EnergySet(a.set, dev)
    gen = torch.Generator().manual_seed(a.seed)
    held_geom = data.heldout[data.geom_mol]
    train_geom = (~held_geom & (data.noise <= a.max_noise + 1e-9)).nonzero().flatten()
    minima = (~held_geom & (data.noise == 0)).nonzero().flatten()

    def sample(pool, n):
        return pool[torch.randperm(pool.numel(), generator=gen)[:n].to(pool.device)]

    eval_sets = {'held-out molecules': sample(held_geom.nonzero().flatten(), a.eval_geometries),
                 'training molecules': sample((~held_geom).nonzero().flatten(), a.eval_geometries)}
    # two relaxed conformers of one held-out molecule: consecutive minima rows of the same molecule
    hm = (held_geom & (data.noise == 0)).nonzero().flatten()
    same = data.geom_mol[hm[1:]] == data.geom_mol[hm[:-1]]
    gap_pairs = torch.stack((hm[:-1][same], hm[1:][same]), dim=1)[:a.eval_geometries]

    # the model is built before any sampling that matters: mxtaltools' scalarMLP reseeds torch at construction
    trunk_args = dict(node_dim=a.node_dim, message_dim=a.message_dim, num_convs=a.num_convs,
                      num_radial=a.num_radial, cutoff=a.cutoff)
    model = IntraTrunk(**trunk_args).to(dev)

    resume = torch.load(running, map_location=dev, weights_only=False) if os.path.exists(running) else None
    if resume is not None:
        moved = [k for k in RESUME_KEYS if resume['args'][k] != getattr(a, k)]
        if moved:
            raise SystemExit(f"{running} was written under other arguments ({', '.join(moved)}); remove it or "
                             f"restore them")
        e0, scale, energy_scale = resume['e0'].to(dev), float(resume['scale']), float(resume['energy_scale'])
    else:
        e0 = data.fit_reference(minima)
        data.set_reference(e0)
        cal = sample(train_geom[data.noise[train_geom] <= a.scale_noise + 1e-9], 50000)
        fsq = comps = esq = 0.0
        for lo in range(0, cal.numel(), 4096):
            _, _, force, _, excess, n = data.batch(cal[lo:lo + 4096])
            fsq += float(force.double().square().sum())
            comps += float(3 * n.sum())
            esq += float((excess.double() / n).square().sum())
        scale, energy_scale = math.sqrt(fsq / comps), math.sqrt(esq / cal.numel())
    data.set_reference(e0)
    model.set_units(e0.float(), scale)
    ema = copy.deepcopy(model).requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    present = (e0 != 0).nonzero().flatten().tolist()
    print(f"[intra] {a.tag}: {data.num_molecules} molecules ({int(data.heldout.sum())} held out), "
          f"{train_geom.numel()} training geometries of {data.num_geometries}; {data.build['force_field']}; "
          f"batch {a.batch}, {a.steps} steps, device {dev.type}; {n_params:,} parameters; {a.num_convs} stages, "
          f"node width {a.node_dim}, message width {a.message_dim}, {a.num_radial} radial functions over "
          f"{a.cutoff:g} A", flush=True)
    print(f"[intra] reference energy per element (kcal/mol): "
          + ", ".join(f"Z={z}: {float(e0[z]):.3f}" for z in present)
          + f"; force scale {scale:.2f} kcal/mol/A, energy scale {energy_scale:.3f} kcal/mol per atom", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        frac = (step - a.warmup) / max(a.steps - a.warmup, 1)
        return a.lr_final + 0.5 * (a.lr - a.lr_final) * (1 + math.cos(math.pi * min(frac, 1.0)))

    start, history, log = 0, {}, []
    if resume is not None:
        model.load_state_dict(resume['model_raw'])
        ema.load_state_dict(resume['model'])
        opt.load_state_dict(resume['opt'])
        gen.set_state(resume['gen'])
        start, history = resume['step'], resume['history']
        print(f"[intra] resumed from {running} at step {start}", flush=True)

    def save(step, final):
        blob = {'model': ema.state_dict(), 'model_raw': model.state_dict(), 'trunk_args': trunk_args,
                'args': vars(a), 'e0': e0.cpu(), 'scale': scale, 'energy_scale': energy_scale, 'step': step,
                'history': history, 'set_build': data.build}
        torch.save({**blob, 'opt': opt.state_dict(), 'gen': gen.get_state()}, running)
        if final:
            torch.save(blob, os.path.join(a.out, f'{a.tag}.pt'))
            with open(os.path.join(a.out, f'{a.tag}_eval.json'), 'w') as f:
                json.dump({'args': vars(a), 'parameters': n_params, 'scale': scale, 'energy_scale': energy_scale,
                           'evaluations': history}, f, indent=1)

    def score(step):
        results = {name: evaluate(ema, data, g, scale, energy_scale) for name, g in eval_sets.items()}
        gaps = conformer_gaps(ema, data, gap_pairs, scale, energy_scale)
        print_eval(step, results, gaps, a)
        history[str(step)] = {'splits': results, 'conformer_gap': gaps}

    t0 = time.perf_counter()
    for step in range(start + 1, a.steps + 1):
        for group in opt.param_groups:
            group['lr'] = lr_at(step - 1)
        g = train_geom[torch.randint(0, train_geom.numel(), (a.batch,), generator=gen).to(dev)]
        loss, f_loss, e_loss, _, _ = losses(model, data.batch(g), scale, energy_scale, a.energy_weight, training=True)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print(f"[intra] step {step}: non-finite loss, batch skipped", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            decay = min(a.ema, (1 + step) / (10 + step))      # the average follows closely at first
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.lerp_(p, 1 - decay)
        log.append((float(f_loss.detach()), float(e_loss.detach())))
        if step % 100 == 0 or step == start + 1:
            recent = torch.tensor(log[-100:])
            print(f"[intra] step {step:6d}  force loss {float(recent[:, 0].mean()):.4f}  energy loss "
                  f"{float(recent[:, 1].mean()):.4f}  lr {lr_at(step - 1):.2e}  "
                  f"{(time.perf_counter() - t0) / (step - start):.3f} s/step"
                  + (f"  peak GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB" if dev.type == 'cuda' else ""),
                  flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            score(step)
            save(step, final=True)
        elif step % a.save_every == 0:
            save(step, final=False)
    print(f"[intra] done: step {a.steps}; checkpoint and evaluation in {a.out} as {a.tag}.pt / {a.tag}_eval.json",
          flush=True)


if __name__ == '__main__':
    main()
