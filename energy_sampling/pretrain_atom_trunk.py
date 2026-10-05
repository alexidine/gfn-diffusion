"""Offline regression of the eLJ energy and its force on the crystal latent, on full-QM9 molecules.

    python -u pretrain_atom_trunk.py --arm trunk    --out <dir> [--target full|short] [--steps N] ...
    python -u pretrain_atom_trunk.py --arm embedded --out <dir> ...
    python -u pretrain_atom_trunk.py --arm head     --out <dir> --trunk <dir>/trunk.pt ...

No trainer, no policy, no rollout. Three models are fitted to the same targets and scored on
molecules the conditional run holds out (`test_molecules_path`), so the trunk never sees a
molecule the sampler is later evaluated on:

  trunk     `models/atom_trunk.py`: per-atom energies from pairs within --feature-cutoff, the
            force on the latent by autograd through `mxtaltools.crystal_building.image_pairs`.
  embedded  an MLP on the latent and the molecule's stored Mo3ENet embedding, the inputs the
            policy has today; energy and force are direct outputs.
  head      a forward-only MLP from the frozen trunk's pooled node states, the latent and the
            density to the force rows.

STATES. A stored crystal is walked back toward the source by the trainer's reference kernel to
a random step t of T: x_t = x_T t/T + sqrt(m t_scale (t/T)(1 - t/T)) eps, the marginal of the
Brownian bridge pinned at the latent origin. Step T is the stored crystal, step 0 the origin.
The loss is per state, so drawing this marginal is the same as simulating whole backward
trajectories and keeping one state of each. m is a per-state multiplier on the bridge's
variance drawn from --noise-scales: 1 is the run's own bridge, larger values widen the tube
of states around the path, toward what a policy that is not yet good would visit.
--conditions may be the conditions file (one crystal per molecule) or the prior file (every
kept crystal of every training molecule); both hold their batch under --split-key.

REFERENCE. Per atom, the eLJ energy of that atom with every atom of every other molecule within
--label-cutoff, times lj_coeff, over the temperature (so in kT), log-compressed above
--compress-at (`mxtaltools.common.utils.log_rescale_positive`); the crystal's energy is the sum
over its atoms and the force its gradient with respect to the latent. Near a minimum this is
the trainer's eLJ leg; in an overlapped cell its gradients stay bounded. The cell-reduction,
bounding and Jacobian terms of the reward are functions of the latent alone and are left out.

TARGET. What a model is fitted to:
  --target full   the reference itself. Its pairs reach past the feature cutoff, so the model
                  has to infer the remainder from more message-passing stages and the density.
  --target short  the same per-atom sum over pairs inside --feature-cutoff only, each pair
                  switched smoothly to zero over the last --switch-width
                  (`vdw_analysis.lj_cutoff_envelope`): a function of exactly what the model is
                  given. The pairs between the feature cutoff and the label cutoff are replaced
                  by a closed-form mean-field term (`tail_energy`) that depends on the cell
                  volume alone; it is not learned, and is added to the model's force when the
                  model is scored against the reference.

LOSS. Energy: Huber (delta 1 kT) on per-atom energies (trunk) or on energy per atom
(embedded). Force: with each latent row divided by a fixed scale measured once on late
states of the reference, mean_r(err^2) / (mean_r(target^2) + 1): absolute where forces are
small, relative where they are large. No batch statistics enter any model or loss.
"""
import argparse
import json
import math
import os
import time

import torch
from torch import nn

from buffer import strip_lazy_sg_caches
from models.atom_trunk import AtomTrunk, crystal_density, intramolecular_edges, pooled_features
from mxtaltools.analysis.vdw_analysis import exponential_edgewise_lj_energy, lj_cutoff_envelope
from mxtaltools.common.utils import log_rescale_positive
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_building.image_pairs import build_image_tables, select_images, pair_distances

EVAL_STEPS_OF_T = (1.0, 0.8, 0.5, 0.2, 0.0)      # trajectory positions scored at evaluation, as fractions of T
CELL_ROWS, POSE_ROWS = slice(0, 6), slice(6, 12)


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--arm', choices=('trunk', 'embedded', 'head'), required=True)
    ap.add_argument('--out', required=True, help='directory for the checkpoint, the log and the evaluation json')
    ap.add_argument('--tag', default=None, help='file stem for this run inside --out (default: the arm)')
    ap.add_argument('--trunk', default=None, help='trunk checkpoint, required by --arm head')
    ap.add_argument('--conditions', default=r'D:/crystal_datasets/conditional/priors/qm9full_conditions.pt')
    ap.add_argument('--test-conditions', default=r'D:/crystal_datasets/conditional/priors/qm9full_test_conditions.pt')
    ap.add_argument('--train-molecules', type=int, default=0, help='use only this many training molecules (0 = all)')
    ap.add_argument('--eval-molecules', type=int, default=512, help='molecules per split scored at each evaluation')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--vram-frac', type=float, default=0.4)
    ap.add_argument('--seed', type=int, default=0)
    # the conditional run's own values (qf30_fwdF_lr5_local.yaml): energy_config and integrator
    ap.add_argument('--temperature', type=float, default=6.9)
    ap.add_argument('--lj-coeff', type=float, default=1.0)
    ap.add_argument('--T', type=int, default=50)
    ap.add_argument('--t-scale', type=float, default=0.05)
    ap.add_argument('--noise-scales', default='1',
                    help='comma-separated multipliers on the bridge variance, one drawn per training state')
    ap.add_argument('--split-key', default='prior', help='key of the crystal batch inside --conditions')
    ap.add_argument('--label-cutoff', type=float, default=10.0)
    ap.add_argument('--feature-cutoff', type=float, default=5.0)
    ap.add_argument('--target', choices=('full', 'short'), default='full')
    ap.add_argument('--switch-width', type=float, default=1.0, help='width (A) of the smooth switch of the short target')
    ap.add_argument('--compress-at', type=float, default=5.0, help='per-atom energy (kT) above which the target is log-compressed')
    ap.add_argument('--node-dim', type=int, default=128)
    ap.add_argument('--message-dim', type=int, default=64)
    ap.add_argument('--num-convs', type=int, default=3)
    ap.add_argument('--unfolded', action='store_true', help="use MXtalTools' MConv as written instead of FoldedMConv")
    ap.add_argument('--hidden', type=int, default=512, help='width of the embedded and head MLPs')
    return ap.parse_args()


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden, layers=4):
        super().__init__()
        dims = [n_in] + [hidden] * layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.GELU()]
        self.net = nn.Sequential(*mods, nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def load_split(path, n, seed, key='prior'):
    """The crystal batch of a split, optionally cut to `n` rows by a seeded permutation."""
    batch = strip_lazy_sg_caches(torch.load(path, weights_only=False, map_location='cpu')[key])
    if n and n < batch.num_graphs:
        batch = batch.subsample_new_batch(torch.randperm(batch.num_graphs, generator=torch.Generator().manual_seed(seed))[:n])
    return batch


def bridge_state(x_end, frac, t_scale, gen):
    """x_t given the stored latent x_end at t = T: the reference bridge's marginal at t = frac * T.
    `t_scale` is the bridge's total variance, a float or one value per row."""
    frac = frac.to(x_end)[:, None]
    if torch.is_tensor(t_scale):
        t_scale = t_scale.to(x_end)[:, None]
    noise = torch.randn(x_end.shape, generator=gen).to(x_end)
    return x_end * frac + (t_scale * frac * (1 - frac)).sqrt() * noise


def atom_energies(pairs, vdw, a, n_nodes, switch_at=None):
    """Per-atom eLJ energy in kT over a pair list, optionally switched to zero at `switch_at`, then compressed."""
    e_pair = exponential_edgewise_lj_energy(
        vdw, {'intermolecular_dist': pairs['dist'],
              'intermolecular_dist_atoms': [pairs['z_src'], pairs['z_tgt']]}, 2.5) * (a.lj_coeff / a.temperature)
    if switch_at is not None:
        e_pair = e_pair * lj_cutoff_envelope(pairs['dist'], switch_at, a.switch_width)
    raw = torch.zeros(n_nodes, dtype=e_pair.dtype, device=e_pair.device).index_add_(0, pairs['node_ref'], e_pair)
    return log_rescale_positive(raw, a.compress_at)


def tail_energy(tables, T_fc, vdw, a):
    """Mean-field energy, in kT per crystal, of the pairs between the feature cutoff c and the label cutoff R.

    Every reference atom a sees each atom type b of the molecule at the cell's mean number
    density Z / V, so the 12-6 pair energy integrates in closed form:
        (Z / V) sum_a sum_b 16 pi [ s^12 (c^-9 - R^-9) / 9 - s^6 (c^-3 - R^-3) / 3 ],  s = r_vdW(a) + r_vdW(b).
    Both cutoffs lie outside every s, where eLJ is the plain 12-6 form. A function of the cell
    volume alone, so its force acts on the cell rows only.
    """
    rad = vdw[tables.z] * tables.amask
    s = rad[:, :, None] + rad[:, None, :]
    mask = (tables.amask[:, :, None] & tables.amask[:, None, :]).to(s.dtype)
    c, r = a.feature_cutoff, a.label_cutoff
    integral = 16 * math.pi * (s ** 12 * (c ** -9 - r ** -9) / 9 - s ** 6 * (c ** -3 - r ** -3) / 3)
    per_volume = tables.kmask.sum(1).to(s.dtype) / torch.linalg.det(T_fc).abs()
    return (a.lj_coeff / a.temperature) * per_volume * (integral * mask).sum((1, 2))


def build_example(sub, x_t, a, vdw, need_geometry_grad, with_reference=False, labels=True):
    """Targets and model inputs of one batch of crystals at the latents x_t.

    Returns a dict. `x` is the leaf the geometry was built from. `force` [B, 12] is the
    gradient of the TARGET energy with respect to it; `tail_force` the gradient of the
    closed-form tail (zero under --target full); `ref_force`, with `with_reference`, the
    gradient of the reference energy. The inter-edge distances and the density keep their
    graph to `x` when `need_geometry_grad`. With `labels` False only the model inputs are built.
    """
    x = x_t.clone().requires_grad_(True)
    cb = sub.clone()
    cb.latent_to_cell_params(x)
    tables = build_image_tables(cb)
    geom = (cb.T_fc, cb.T_cf, cb.aunit_centroid, cb.aunit_orientation)
    n_nodes, n_graphs = int(tables.nat.sum()), cb.num_graphs
    node_graph = cb.batch
    short = getattr(a, 'target', 'full') == 'short'

    def summed(e_atom):
        return torch.zeros(n_graphs, dtype=e_atom.dtype, device=x.device).index_add_(0, node_graph, e_atom)

    def gradient(energy):
        return torch.autograd.grad(energy.sum(), x, retain_graph=True)[0].detach()

    out = {'x': x, 'n_graphs': n_graphs, 'node_graph': node_graph, 'z': cb.z.long(), 'nat': tables.nat,
           'tail_force': torch.zeros_like(x_t),
           'embedding': cb.embedding.reshape(n_graphs, -1) if hasattr(cb, 'embedding') else None}
    wide = None
    if short or not labels:
        # the target's pairs are the model's pairs
        sel = select_images(tables, *geom, a.feature_cutoff)
        pairs = pair_distances(tables, sel, a.feature_cutoff)
        inter_ref, inter_img, inter_dist = pairs['node_ref'], pairs['node_img'], pairs['dist']
        out['pairs_label'] = pairs['dist'].numel() / n_graphs
        if labels:
            e_atom = atom_energies(pairs, vdw, a, n_nodes, switch_at=a.feature_cutoff)
            out['e_atom'], out['energy'] = e_atom.detach(), summed(e_atom).detach()
            out['force'] = gradient(summed(e_atom))
            out['tail_force'] = gradient(tail_energy(tables, cb.T_fc, vdw, a))
    else:
        sel = select_images(tables, *geom, a.label_cutoff)
        wide = pair_distances(tables, sel, a.label_cutoff)
        e_atom = atom_energies(wide, vdw, a, n_nodes)
        out['e_atom'], out['energy'] = e_atom.detach(), summed(e_atom).detach()
        out['force'] = gradient(summed(e_atom))
        out['pairs_label'] = wide['dist'].numel() / n_graphs
        # model inputs: the pairs inside the feature cutoff, their distances recomputed on their own so
        # that a later backward pass touches these pairs only (same arithmetic as image_pairs.pair_distances)
        near = (wide['dist'].detach() <= a.feature_cutoff).nonzero().flatten()
        image, graph = wide['image'][near], wide['graph'][near]
        vec = sel['rel'][image] + sel['skel'][graph, sel['op'][image], wide['ib'][near]] \
            - sel['skel'][graph, 0, wide['ia'][near]]
        inter_ref, inter_img, inter_dist = wide['node_ref'][near], wide['node_img'][near], vec.square().sum(-1).sqrt()
    if with_reference and labels:
        if wide is None:
            wide = pair_distances(tables, select_images(tables, *geom, a.label_cutoff), a.label_cutoff)
            out['ref_force'] = gradient(summed(atom_energies(wide, vdw, a, n_nodes)))
        else:
            out['ref_force'] = out['force']
    density = crystal_density(tables, cb.T_fc)
    intra_index, intra_dist = intramolecular_edges(tables, a.feature_cutoff)
    out.update({'intra_index': intra_index, 'intra_dist': intra_dist, 'inter_ref': inter_ref, 'inter_img': inter_img,
                'inter_dist': inter_dist if need_geometry_grad else inter_dist.detach(),
                'density': density if need_geometry_grad else density.detach(),
                'pairs_feature': inter_dist.numel() / n_graphs})
    return out


def force_loss(pred, target, scale):
    """Per crystal: mean over latent rows of the squared scaled error over (mean squared scaled target + 1)."""
    u, v = (pred - target) / scale, target / scale
    return u.square().mean(1) / (v.square().mean(1) + 1.0)


def trunk_energy(model, ex):
    return model(ex['z'], ex['node_graph'], ex['intra_index'], ex['intra_dist'],
                 ex['inter_ref'], ex['inter_img'], ex['inter_dist'], ex['density'])


def predict(arm, models, ex, scale, training):
    """(force prediction [B, 12], energy loss term or None, energy). Geometry gradients are used by the trunk only."""
    if arm == 'trunk':
        out = trunk_energy(models['trunk'], ex)
        force, = torch.autograd.grad(out['energy'].sum(), ex['x'], create_graph=training)
        e_loss = nn.functional.huber_loss(out['e_atom'], ex['e_atom'], delta=1.0)
        return force, e_loss, out['energy']
    if arm == 'embedded':
        out = models['embedded'](torch.cat((ex['x'].detach(), ex['embedding']), dim=1))
        energy = out[:, 0] * ex['nat']
        e_loss = nn.functional.huber_loss(out[:, 0], ex['energy'] / ex['nat'], delta=1.0)
        return out[:, 1:] * scale, e_loss, energy
    with torch.no_grad():
        out = trunk_energy(models['trunk'], ex)
        feats = torch.cat((pooled_features(out['h'], ex['node_graph'], ex['n_graphs']),
                           ex['x'].detach(), ex['density'][:, None]), dim=1)
    return models['head'](feats) * scale, None, out['energy']


def _relative(u, v, rows=slice(None)):
    return (u - v)[:, rows].norm(dim=1) / v[:, rows].norm(dim=1).clamp(min=1e-6)


def evaluate(arm, models, split, a, vdw, scale, dev, noise_mult=1.0):
    """Scoring of one split at fixed trajectory positions; one dict of medians over molecules per position.

    Two comparisons: the model's force against its own target, and the model's force plus
    the closed-form tail against the reference (the same thing under --target full).
    """
    rows = {}
    gen = torch.Generator().manual_seed(1234)
    n = min(a.eval_molecules, split.num_graphs)
    for frac in EVAL_STEPS_OF_T:
        acc = {k: [] for k in ('cos', 'pose', 'cell', 'rms', 'size', 'own_pose', 'own_cell', 'e')}
        bad = 0
        for start in range(0, n, a.batch):
            sub = split.subsample_new_batch(torch.arange(start, min(start + a.batch, n))).to(dev)
            x_t = bridge_state(sub.latent_params(gauge_fix_free_axes=True),
                               torch.full((sub.num_graphs,), frac), a.t_scale * noise_mult, gen)
            ex = build_example(sub, x_t, a, vdw, need_geometry_grad=(arm == 'trunk'), with_reference=True)
            pred, _, energy = predict(arm, models, ex, scale, training=False)
            pred, energy = pred.detach(), energy.detach()
            ok = torch.isfinite(ex['ref_force']).all(1) & torch.isfinite(ex['force']).all(1) & torch.isfinite(pred).all(1)
            bad += int((~ok).sum())
            u, v = (pred + ex['tail_force'])[ok] / scale, ex['ref_force'][ok] / scale
            acc['cos'].append(nn.functional.cosine_similarity(u, v, dim=1))
            acc['pose'].append(_relative(u, v, POSE_ROWS))
            acc['cell'].append(_relative(u, v, CELL_ROWS))
            acc['rms'].append((u - v).square().mean(1).sqrt())
            acc['size'].append(v.square().mean(1).sqrt())
            uo, vo = pred[ok] / scale, ex['force'][ok] / scale
            acc['own_pose'].append(_relative(uo, vo, POSE_ROWS))
            acc['own_cell'].append(_relative(uo, vo, CELL_ROWS))
            acc['e'].append(((energy - ex['energy']).abs() / ex['nat'])[ok])
        acc = {k: torch.cat(v) for k, v in acc.items()}
        rows[frac] = dict(cos_median=float(acc['cos'].median()), cos_p10=float(acc['cos'].quantile(0.1)),
                          err_pose=float(acc['pose'].median()), err_cell=float(acc['cell'].median()),
                          rms_scaled=float(acc['rms'].median()), target_scaled=float(acc['size'].median()),
                          own_pose=float(acc['own_pose'].median()), own_cell=float(acc['own_cell'].median()),
                          e_per_atom=float(acc['e'].median()), n=int(acc['cos'].numel()), nonfinite=bad)
    return rows


def print_eval(tag, step, arm, a, results, noise_mult=1.0):
    print(f"\nTable. {tag} at step {step}: arm '{arm}', target '{a.target}', {a.feature_cutoff:g} A features. Force on "
          f"the 12 latent rows for QM9 conditions crystals walked back to a trajectory position by the reference "
          f"bridge at {noise_mult:g}x its variance (trained on multipliers {a.noise_scales}; position 1 is the stored "
          f"crystal, 0 the latent origin); medians over molecules (count in the last "
          f"column), rows scaled by fixed per-row constants. 'vs reference' compares the model's force, plus the "
          f"closed-form tail under target 'short', with the {a.label_cutoff:g} A compressed-eLJ force; 'vs own target' "
          f"compares the model's force with what it was fitted to. Relative error is |prediction - target| / "
          f"|target| and is ill-conditioned where the target is near zero (position 1, the stored minimum), so the "
          f"RMS error over rows is given beside the RMS size of the reference. Energy error is |E - E_target| per atom"
          + (", the frozen trunk's for the head arm." if arm == 'head' else "."))
    print("split | position (t/T) | cosine vs reference, median | cosine vs reference, 10th pct | pose rows vs reference | "
          "cell rows vs reference | RMS error / RMS reference (row-scale units) | pose rows vs own target | "
          "cell rows vs own target | energy error (kT per atom) | molecules")
    for split, rows in results.items():
        for frac, r in rows.items():
            print(f"{split} | {frac:g} | {r['cos_median']:.4f} | {r['cos_p10']:.4f} | {r['err_pose']:.3f} | "
                  f"{r['err_cell']:.3f} | {r['rms_scaled']:.3f} / {r['target_scaled']:.3f} | {r['own_pose']:.3f} | "
                  f"{r['own_cell']:.3f} | {r['e_per_atom']:.3f} | {r['n']}")


def time_inference(model, split, a, vdw, dev, n=1000, chunk=250, reps=5):
    """Wall time per `n` crystal states at trajectory position 0.9, as a rollout would pay it: geometry
    (latent map on a cloned batch, image selection, pair distances), the trunk's forward pass without
    gradients, and forward plus one backward pass to the latent. Chunks of `chunk` states; median of `reps`."""
    n = min(n, split.num_graphs)
    sub_all = split.subsample_new_batch(torch.arange(n)).to(dev)
    x_all = bridge_state(sub_all.latent_params(gauge_fix_free_axes=True), torch.full((n,), 0.9), a.t_scale,
                         torch.Generator().manual_seed(7))
    chunks = [(sub_all.subsample_new_batch(torch.arange(lo, min(lo + chunk, n))), x_all[lo:lo + chunk])
              for lo in range(0, n, chunk)]

    def sync():
        if dev.type == 'cuda':
            torch.cuda.synchronize()

    def med(fn):
        fn()
        ts = []
        for _ in range(reps):
            sync()
            t = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - t)
        ts.sort()
        return ts[len(ts) // 2] * 1e3

    def geometry(grad):
        return [build_example(s, x, a, vdw, need_geometry_grad=grad, labels=False) for s, x in chunks]

    def forward():
        with torch.no_grad():
            for ex in geometry(False):
                trunk_energy(model, ex)

    def forward_force():
        for ex in geometry(True):
            torch.autograd.grad(trunk_energy(model, ex)['energy'].sum(), ex['x'])

    model.eval()
    pairs = sum(ex['pairs_feature'] * ex['n_graphs'] for ex in geometry(False)) / n
    t_geo = med(lambda: geometry(False))
    t_fwd = med(forward)
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    t_force = med(forward_force)
    mem = (torch.cuda.max_memory_allocated() - base) / 2 ** 30 if dev.type == 'cuda' else float('nan')
    model.train()
    out = dict(states=n, chunk=chunk, pairs_per_crystal=pairs, geometry_ms=t_geo, forward_ms=t_fwd - t_geo,
               forward_force_ms=t_force - t_geo, peak_gib_per_chunk=mem)
    print(f"\nTable. Cost of the trunk per {n} held-out crystal states at trajectory position 0.9, in chunks of {chunk}, "
          f"{dev.type}" + (f" ({torch.cuda.get_device_name(0)}, shared)" if dev.type == 'cuda' else "")
          + f", median of {reps} calls after a warm-up: {a.num_convs} stage(s), message width {a.message_dim}, "
          f"{a.feature_cutoff:g} A features, {'MConv as written' if a.unfolded else 'folded messages'}, "
          f"{pairs:.0f} image pairs per crystal. Geometry is timed alone and subtracted from the two trunk rows; "
          f"peak memory is per chunk, for forward + force.")
    print("stage | wall time (ms)")
    print(f"geometry | {t_geo:.1f}")
    print(f"trunk forward, no gradients | {t_fwd - t_geo:.1f}")
    print(f"trunk forward + force on the latent | {t_force - t_geo:.1f}")
    print(f"peak memory per chunk (GiB) | {mem:.2f}")
    return out


def main():
    a = parse()
    dev = torch.device(a.device)
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.vram_frac)
    os.makedirs(a.out, exist_ok=True)
    stem = a.tag or a.arm
    vdw = torch.tensor(list(VDW_RADII.values()), device=dev)

    train = load_split(a.conditions, a.train_molecules, a.seed, a.split_key)
    test = load_split(a.test_conditions, 0, a.seed)
    mults = torch.tensor([float(v) for v in a.noise_scales.split(',')])
    train_eval = train.subsample_new_batch(torch.arange(min(a.eval_molecules, train.num_graphs)))
    print(f"[pretrain] arm {a.arm} ({stem}): {train.num_graphs} training molecules, {test.num_graphs} held-out molecules, "
          f"batch {a.batch}, {a.steps} steps, device {dev.type}; kT = {a.temperature:g} raw eLJ units, "
          f"lj_coeff {a.lj_coeff:g}; target '{a.target}', reference cutoff {a.label_cutoff:g} A, feature cutoff "
          f"{a.feature_cutoff:g} A, per-atom compression above {a.compress_at:g} kT; bridge variance multipliers "
          f"{mults.tolist()}", flush=True)

    # models are built before any sampling: mxtaltools' scalarMLP reseeds torch at construction
    models = {}
    if a.arm in ('trunk', 'head'):
        models['trunk'] = AtomTrunk(node_dim=a.node_dim, message_dim=a.message_dim, num_convs=a.num_convs,
                                    cutoff=a.feature_cutoff, folded=not a.unfolded).to(dev)
    if a.arm == 'head':
        if not a.trunk:
            raise SystemExit('--arm head needs --trunk')
        ck = torch.load(a.trunk, map_location=dev, weights_only=False)
        models['trunk'].load_state_dict(ck['model'])
        models['trunk'].requires_grad_(False).eval()
        models['head'] = MLP(2 * a.node_dim + 12 + 1, 12, a.hidden).to(dev)
    if a.arm == 'embedded':
        emb_dim = int(train.embedding.reshape(train.num_graphs, -1).shape[1])
        models['embedded'] = MLP(12 + emb_dim, 13, a.hidden).to(dev)
    fitted = models[a.arm]
    n_params = sum(p.numel() for p in fitted.parameters())
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)

    # per-row force scale: median |reference force| over crystals at late positions, measured once and then
    # fixed. Taken from the reference under either target, so runs with different targets share one scale.
    if a.arm == 'head':
        scale = ck['row_scale'].to(dev)
    else:
        cal = []
        for _ in range(4):
            sub = train.subsample_new_batch(torch.randint(0, train.num_graphs, (a.batch,), generator=gen)).to(dev)
            x_t = bridge_state(sub.latent_params(gauge_fix_free_axes=True),
                               0.8 + 0.2 * torch.rand(sub.num_graphs, generator=gen), a.t_scale, gen)
            f = build_example(sub, x_t, a, vdw, False, with_reference=True)['ref_force']
            cal.append(f[torch.isfinite(f).all(1)])
        scale = torch.cat(cal).abs().median(0).values.clamp(min=1e-3)
    print(f"[pretrain] {n_params:,} fitted parameters; row scales (kT per latent unit): "
          + ", ".join(f"{v:.1f}" for v in scale.tolist()), flush=True)

    opt = torch.optim.Adam(fitted.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.05)
    log, t0, skipped = [], time.perf_counter(), 0
    history, timing = {}, None

    def save(step):
        torch.save({'model': fitted.state_dict(), 'row_scale': scale.cpu(), 'args': vars(a), 'step': step},
                   os.path.join(a.out, f'{stem}.pt'))
        with open(os.path.join(a.out, f'{stem}_eval.json'), 'w') as f:
            json.dump({'args': vars(a), 'parameters': n_params, 'timing': timing,
                       'seconds_per_step': (time.perf_counter() - t0) / max(step, 1),
                       'evaluations': {str(k): {s: {str(p): v for p, v in r.items()} for s, r in res.items()}
                                       for k, res in history.items()}}, f, indent=1)

    for step in range(1, a.steps + 1):
        sub = train.subsample_new_batch(torch.randint(0, train.num_graphs, (a.batch,), generator=gen)).to(dev)
        frac = torch.randint(0, a.T + 1, (sub.num_graphs,), generator=gen).float() / a.T
        mult = mults[torch.randint(0, len(mults), (sub.num_graphs,), generator=gen)]
        x_t = bridge_state(sub.latent_params(gauge_fix_free_axes=True), frac, a.t_scale * mult, gen)
        ex = build_example(sub, x_t, a, vdw, need_geometry_grad=(a.arm == 'trunk'))
        pred, e_loss, _ = predict(a.arm, models, ex, scale, training=True)
        ok = torch.isfinite(ex['force']).all(1)
        skipped += int((~ok).sum())
        f_loss = force_loss(pred[ok], ex['force'][ok], scale).mean()
        loss = f_loss + (e_loss if e_loss is not None else 0.0)
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print(f"[pretrain] step {step}: non-finite loss, batch skipped", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fitted.parameters(), 10.0)
        opt.step()
        sched.step()
        log.append((float(f_loss), float(e_loss) if e_loss is not None else float('nan')))
        if step % 50 == 0 or step == 1:
            recent = torch.tensor(log[-50:])
            print(f"[pretrain] step {step:6d}  force loss {float(recent[:, 0].mean()):.4f}  energy loss "
                  f"{float(recent[:, 1].mean()):.4f}  {(time.perf_counter() - t0) / step:.3f} s/step  "
                  f"pairs/crystal label {ex['pairs_label']:.0f} feature {ex['pairs_feature']:.0f}  "
                  f"rows skipped so far {skipped}"
                  + (f"  peak GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB" if dev.type == 'cuda' else ""),
                  flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            fitted.eval()
            results = {'held-out molecules': evaluate(a.arm, models, test, a, vdw, scale, dev),
                       'training molecules': evaluate(a.arm, models, train_eval, a, vdw, scale, dev)}
            fitted.train()
            print_eval('Evaluation', step, a.arm, a, results)
            history[step] = results
            if step == a.steps and float(mults.max()) > 1.0:
                # the widest states the model was trained on, scored once at the end
                fitted.eval()
                wide = {'held-out molecules': evaluate(a.arm, models, test, a, vdw, scale, dev, float(mults.max()))}
                fitted.train()
                print_eval('Evaluation on widened states', step, a.arm, a, wide, float(mults.max()))
                history[f'{step}_wide'] = wide
            save(step)
    train_minutes = (time.perf_counter() - t0) / 60
    if a.arm == 'trunk':
        timing = time_inference(fitted, test, a, vdw, dev)
        save(a.steps)
    print(f"[pretrain] done: {a.steps} steps in {train_minutes:.1f} min; checkpoint and evaluation in {a.out} "
          f"as {stem}.pt / {stem}_eval.json", flush=True)


if __name__ == '__main__':
    main()
