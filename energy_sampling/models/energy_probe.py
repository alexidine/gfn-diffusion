"""Can the sampler's own policy network regress E(x | c)?

The sampler has to produce x ~ exp(-E(x | c)). Regressing E(x | c) from the same inputs is
the easier problem, so a network that cannot do it cannot sample either. This trains exactly
that: the forward policy's token network (models/ragged_set_policy.py, one scalar per
coordinate token, summed over the molecule's coordinates) on the run's own energy function.

    python -u -m models.energy_probe --rung <condition set dir> --out-dir <dir> --name <arm>

DATA. A condition set as build_conformer_set.py writes it (conditions_train.pt,
prior_train.pt; the compact prior form or the graph form). Every training step draws stored
prior rows, displaces each coordinate by its stored thermal width (energies/dof_features.py,
`log_thermal_sigma`) times one multiplier per row, log-uniform over --noise-range, and scores
the result with the energy function the trainer builds from --config (MultiConformerTorsions,
energy_clip and its origin included). The target is that potential minus the molecule's
potential at its reference conformer, kcal/mol (= kT at T = 1). No sample is seen twice.

SPLIT. Molecules by parent skeleton (stereoisomers stay together): --heldout-frac of them
are never trained on. Of a training molecule's stored rows, --conformer-frac are never drawn.
Three fixed test sets, the same at every evaluation and after a resume:
    seen       training rows, displacements the fit did not see
    new conf.  held-out stored rows of training molecules
    new mol.   rows of the held-out molecules

ENCODER (--encoder).
    stored      the per-atom and pooled embeddings the conditions file carries: the frozen
                encoder the sampler runs on
    pretrained  that encoder (--encoder-ckpt) run inside the step and trained with the rest;
                at step 0 its output is the stored embedding, which is checked
    scratch     the same network at --enc-hidden / --enc-layers / --stereo-features, random
                weights, trained with the rest

A run resumes from <out-dir>/<name>_running.pt when that file exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import yaml
from torch import nn

LOSS_SCALE = 10.0          # kcal/mol per unit of the regression target
BAND = 30.0                # "low" samples: within this many kcal/mol of the reference conformer


# ------------------------------------------------------------------------------- data

def load_rung(rung, max_conditions=None):
    """``(conditions batch, identifiers, smiles, row -> condition slot, row states [n, K])``."""
    from energies.conformer_data import check_compact_prior, is_compact_prior

    blob = torch.load(os.path.join(rung, 'conditions_train.pt'), weights_only=False,
                      map_location='cpu')
    cond = blob['prior'] if isinstance(blob, dict) and 'prior' in blob else blob
    idents = list(cond.identifier)
    if len(set(idents)) != len(idents):
        raise SystemExit(f'{rung}/conditions_train.pt lists a condition twice')
    smiles = list(cond.smiles)
    pblob = torch.load(os.path.join(rung, 'prior_train.pt'), weights_only=False,
                       map_location='cpu')
    slot_of = {i: j for j, i in enumerate(idents)}
    if is_compact_prior(pblob):
        check_compact_prior(pblob, os.path.join(rung, 'prior_train.pt'))
        own = torch.tensor([slot_of[i] for i in pblob['identifiers']])
        row_slot = own[pblob['condition_index']]
        states = pblob['torsion_state'].float()
    else:
        p = pblob['prior']
        row_slot = torch.tensor([slot_of[i] for i in p.identifier])
        states = p.torsion_state.float()
    if max_conditions is not None and max_conditions < len(idents):
        keep = row_slot < max_conditions
        row_slot, states = row_slot[keep], states[keep]
        cond = cond.subsample_new_batch(torch.arange(max_conditions))
        idents, smiles = idents[:max_conditions], smiles[:max_conditions]
    return cond, idents, smiles, row_slot, states


def build_energy(cond, idents, smiles, config, device):
    """The run's energy over the set's conditions, each member built from the reference
    conformer the conditions file stores, with the clip origin the config names installed."""
    from build_conformer_references import member_kwargs
    from energies.conformer_torsions import rdkit_order_reference
    from energies.multi_conformer import MultiConformerTorsions

    ec = yaml.safe_load(open(config))['energy_config']
    kw = member_kwargs(ec)
    refs = {}
    for j, (smi, ident) in enumerate(zip(smiles, idents)):
        at = slice(int(cond.ptr[j]), int(cond.ptr[j + 1]))
        refs[ident] = rdkit_order_reference(smi, cond.pos[at].double().numpy(),
                                            level=kw['level'], z=cond.z[at].numpy())[0]
    t = time.time()
    multi = MultiConformerTorsions(smiles, identifiers=idents, reference_positions=refs,
                                   device=str(device), **kw)
    if (ec.get('energy_clip_origin') or 'absolute') == 'reference':
        multi.install_clip_floor()
    ref = multi.reference_potentials().float()
    lib = torch.tensor([multi._lib_index[i] for i in idents])
    print(f'energy: {len(idents)} conditions, carrier width {multi.data_ndim}, clip '
          f'{multi.energy_clip} ({multi.energy_clip_origin}); built in {time.time() - t:.0f} s',
          flush=True)
    return multi, ref[lib], ec


def thermal_sigma(cond, n_cond, K, ec):
    """``[n_cond, K]`` latent-unit thermal width of each coordinate, 0 on pads: the stored
    `log_thermal_sigma` (physical units) over the coordinate's box half-width."""
    from energies.dof_features import feature_names

    names = feature_names()
    static = cond.dof_static.reshape(n_cond, K, -1).float()
    kind = static[..., :5].argmax(-1)
    half = torch.tensor([float(ec.get('delta_r_max', 0.30)), float(ec.get('delta_theta_max', 0.50)),
                         math.pi, 0.50, 0.50])
    sig = static[..., names.index('log_thermal_sigma')].exp() / half[kind]
    return sig * cond.state_mask.reshape(n_cond, K).float()


def molecule_split(smiles, frac):
    """True where the condition's molecule is held out: a hash of its parent skeleton."""
    from models.encoder_probe import parent_skeleton

    seen = {}
    out = []
    for smi in smiles:
        if smi not in seen:
            h = hashlib.sha1(parent_skeleton(smi).encode()).hexdigest()[:8]
            seen[smi] = int(h, 16) / 16 ** 8 < frac
        out.append(seen[smi])
    return torch.tensor(out)


class GraphTable:
    """Every condition's encoder inputs, padded, on the device: a batch of them is a gather."""

    def __init__(self, smiles, perms, z_tree, encoding, k, stereo, want_spd, device):
        from models.encoder_cache import _features
        from models.graph_encodings import graph_from_smiles

        feats = [_features(s, encoding, k, stereo) for s in smiles]
        C = len(feats)
        A = max(f.n for f in feats)
        E = max(f.edge_index.shape[1] for f in feats)
        fx = feats[0].x.shape[1] + feats[0].struct.shape[1]
        self.node_dim, self.edge_dim = feats[0].x.shape[1], feats[0].edge_attr.shape[1]
        x = torch.zeros(C, A, fx)
        e = torch.zeros(C, E, 2, dtype=torch.long)
        ea = torch.zeros(C, E, self.edge_dim)
        spd = torch.full((C, A, A), -1, dtype=torch.long)
        perm = torch.zeros(C, A, dtype=torch.long)
        n = torch.zeros(C, dtype=torch.long)
        ne = torch.zeros(C, dtype=torch.long)
        for c, f in enumerate(feats):
            p = np.asarray(perms[c]).astype(int)
            z_enc = np.asarray(graph_from_smiles(smiles[c])[0]).astype(int)
            if p.shape[0] != f.n or not np.array_equal(z_enc[p], np.asarray(z_tree[c]).astype(int)):
                raise SystemExit(f'{smiles[c]}: the encoder graph does not map onto the '
                                 f"member's atoms through its permutation")
            n[c], ne[c] = f.n, f.edge_index.shape[1]
            x[c, :f.n] = torch.as_tensor(np.concatenate([f.x, f.struct], axis=1), dtype=torch.float32)
            e[c, :ne[c]] = torch.as_tensor(f.edge_index.T, dtype=torch.long)
            ea[c, :ne[c]] = torch.as_tensor(f.edge_attr, dtype=torch.float32)
            spd[c, :f.n, :f.n] = torch.as_tensor(f.spd, dtype=torch.long)
            perm[c, :f.n] = torch.as_tensor(p)
        self.x, self.e, self.ea, self.perm = x.to(device), e.to(device), ea.to(device), perm.to(device)
        self.spd = spd.to(device) if want_spd else None
        self.n, self.ne = n.to(device), ne.to(device)
        self.A, self.E = A, E

    def batch(self, slots):
        """``(x, edge_index, edge_attr, batch, n_graphs, spd, tree_index)``; ``h[tree_index]``
        is the per-atom output in each member's placement order, graph after graph."""
        dev = slots.device
        B = slots.numel()
        n, ne = self.n[slots], self.ne[slots]
        amask = torch.arange(self.A, device=dev)[None] < n[:, None]
        emask = torch.arange(self.E, device=dev)[None] < ne[:, None]
        offs = torch.cumsum(n, 0) - n
        x = self.x[slots][amask]
        batch = torch.arange(B, device=dev)[:, None].expand(B, self.A)[amask]
        ei = (self.e[slots] + offs[:, None, None])[emask].T.contiguous()
        ea = self.ea[slots][emask]
        L = int(n.max())
        spd = None if self.spd is None else self.spd[slots][:, :L, :L]
        tree = (self.perm[slots] + offs[:, None])[amask]
        return x, ei, ea, batch, B, spd, tree


# ------------------------------------------------------------------------------ model

class EnergyModel(nn.Module):
    """Optional graph encoder, then the policy's token network with one output per token."""

    def __init__(self, policy, encoder=None):
        super().__init__()
        self.policy = policy
        self.encoder = encoder

    def embeddings(self, graphs, slots):
        x, ei, ea, batch, B, spd, tree = graphs.batch(slots)
        h, g = self.encoder(x, ei, ea, batch, B, spd=spd)
        return h[tree], g

    def forward(self, x, sub, graphs=None, slots=None):
        from energies.dof_features import MAX_FRAME
        from models.ragged_set_policy import shared_atom_relation

        B, K = x.shape
        if self.encoder is None:
            atom_emb, mol_emb = sub.atom_embedding.float(), sub.embedding.reshape(B, -1).float()
        else:
            atom_emb, mol_emb = self.embeddings(graphs, slots)
        # the layout ConformerGFN.bind_molecular_conditioning gives the policy
        atoms = sub.dof_atoms.reshape(B, -1)
        R = atoms.shape[1] // (K * MAX_FRAME)
        atoms = atoms.reshape(B * K, R, MAX_FRAME).long()
        offs = sub.ptr[:-1].repeat_interleave(K).view(-1, 1, 1)
        mask = sub.state_mask.reshape(B, K).bool()
        flat = mask.reshape(-1).nonzero().squeeze(1)
        dof_atoms = (atoms + offs).reshape(B, K, R, MAX_FRAME)
        cond = dict(atom_emb=atom_emb, dof_atoms=dof_atoms,
                    dof_mask=sub.dof_mask.reshape(B, K, R), mol_emb=mol_emb, state_mask=mask,
                    dof_static=sub.dof_static.reshape(B, K, -1).float(), flat_idx=flat,
                    dof_batch=torch.div(flat, K, rounding_mode='floor'))
        if self.policy.mix_layers:
            cond['token_rel'] = shared_atom_relation(dof_atoms)
        t = torch.zeros(B, self.policy_t_dim, device=x.device)
        return self.policy(x, t, **cond).sum(1) * LOSS_SCALE

    policy_t_dim = 8


def build_model(a, cond, periodic, n_cond, K, graphs, device):
    from models.graph_encoder import MPNNEncoder
    from models.ragged_set_policy import RaggedConditionalSetPolicy

    encoder = None
    if a.encoder == 'stored':
        enc_dim = int(cond.atom_embedding.shape[1])
        mol_dim = int(cond.embedding.reshape(n_cond, -1).shape[1])
    else:
        if a.encoder == 'pretrained':
            from models.encoder_cache import load_encoder
            encoder = load_encoder(a.encoder_ckpt, device)['encoder']
            encoder.train()
        else:
            encoder = MPNNEncoder(graphs.node_dim + a.enc_k, graphs.edge_dim, hidden=a.enc_hidden,
                                  layers=a.enc_layers, attention=True, n_heads=4, max_spd=8).to(device)
        with torch.no_grad():
            h, g = EnergyModel(None, encoder).embeddings(graphs, torch.arange(2, device=device))
        enc_dim, mol_dim = int(h.shape[1]), int(g.shape[1])
    static_dim = int(cond.dof_static.reshape(n_cond, K, -1).shape[-1])
    policy = RaggedConditionalSetPolicy(static_dim, [bool(p) for p in periodic], EnergyModel.policy_t_dim,
                                        enc_dim=enc_dim, mol_dim=mol_dim, corr_dim=a.corr_dim,
                                        frame_size=4, hidden_dim=a.hidden, layers=a.layers,
                                        out_per_token=1, mix_layers=a.mix_layers,
                                        mix_heads=a.mix_heads).to(device)
    return EnergyModel(policy, encoder).to(device)


# ---------------------------------------------------------------------------- scoring

def metrics(pred, y, slots):
    """Errors of ``pred`` against ``y`` (kcal/mol), over all samples and over the low band.

    `centred`: each molecule's mean error over its own low-band samples removed first, and the
    molecule's own mean target likewise in the reference. A sampler needs E(x | c) only up to
    a constant per molecule (log Z(c) absorbs it), so this is the error that matters to it.
    """
    err = pred - y
    lo = y < BAND
    out = {'n': int(y.numel()), 'sd': float(y.std()), 'rmse': float(err.pow(2).mean().sqrt()),
           'med_abs': float(err.abs().median()), 'low_n': int(lo.sum())}
    if int(lo.sum()) > 1:
        el, yl, sl = err[lo], y[lo], slots[lo]
        uniq, inv, cnt = torch.unique(sl, return_inverse=True, return_counts=True)
        mean_e = torch.zeros(len(uniq), dtype=el.dtype).index_add_(0, inv, el) / cnt
        mean_y = torch.zeros(len(uniq), dtype=yl.dtype).index_add_(0, inv, yl) / cnt
        many = (cnt > 1)[inv]
        out.update(low_sd=float(yl.std()), low_rmse=float(el.pow(2).mean().sqrt()),
                   low_med_abs=float(el.abs().median()))
        if int(many.sum()) > 1:
            out.update(low_centred_n=int(many.sum()),
                       low_centred_sd=float((yl - mean_y[inv])[many].pow(2).mean().sqrt()),
                       low_centred_rmse=float((el - mean_e[inv])[many].pow(2).mean().sqrt()))
    return out


TABLE_CAPTION = (
    'Errors in kcal/mol (= kT) of the predicted potential above the reference conformer, on '
    'three fixed test sets (seen: training rows, unseen displacements; conf: unseen stored '
    'conformers of training molecules; mol: unseen molecules). "low" columns keep samples '
    f'within {BAND:g} kcal/mol of the reference; "ctr" removes each molecule\'s mean error, '
    'the part log Z(c) absorbs. Compare each rmse with the sd beside it: sd is what '
    'predicting the mean scores.')


def table_row(step, loss, res, rate):
    cells = [f'{step:>8d}', f'{loss:>9.4f}']
    for name in ('seen', 'conf', 'mol'):
        r = res[name]
        cells.append(f"{r.get('low_rmse', float('nan')):>7.2f}/{r.get('low_sd', float('nan')):<6.2f}")
        cells.append(f"{r.get('low_centred_rmse', float('nan')):>7.2f}/{r.get('low_centred_sd', float('nan')):<6.2f}")
        cells.append(f"{r['rmse']:>7.1f}/{r['sd']:<6.1f}")
    cells.append(f'{rate:>6.1f}')
    return ' '.join(cells)


def table_header():
    cols = [f'{"step":>8}', f'{"loss":>9}']
    for name in ('seen', 'conf', 'mol'):
        cols += [f'{name + " low rmse/sd":>14}', f'{name + " ctr rmse/sd":>14}',
                 f'{name + " all rmse/sd":>14}']
    cols.append(f'{"it/s":>6}')
    return ' '.join(cols)


# ------------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rung', required=True, help='condition set directory')
    ap.add_argument('--config', default='configs/conformer_mk.yaml')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--encoder', default='stored', choices=['stored', 'pretrained', 'scratch'])
    ap.add_argument('--encoder-ckpt', default=None, help='default: models/encoder_cache.DEFAULT_CKPT')
    ap.add_argument('--stereo-features', type=int, default=2, choices=[1, 2], help='scratch only')
    ap.add_argument('--enc-hidden', type=int, default=128, help='scratch only')
    ap.add_argument('--enc-layers', type=int, default=4, help='scratch only')
    ap.add_argument('--enc-k', type=int, default=16, help='scratch only: random-walk encoding width')
    ap.add_argument('--hidden', type=int, default=128, help='per-token width (model.set_policy_hidden)')
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--corr-dim', type=int, default=32)
    ap.add_argument('--mix-layers', type=int, default=2)
    ap.add_argument('--mix-heads', type=int, default=4)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=300000)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lr-final', type=float, default=3e-6, help='cosine floor')
    ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--noise-range', type=float, nargs=2, default=[0.3, 2.0],
                    help='thermal-width multiplier per row, log-uniform over this range')
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

    cond, idents, smiles, row_slot, states = load_rung(a.rung, a.max_conditions)
    C, (n_rows, K) = len(idents), states.shape
    multi, ref, ec = build_energy(cond, idents, smiles, a.config, dev)
    sigma = thermal_sigma(cond, C, K, ec)
    periodic = torch.as_tensor(multi.periodic_dims, dtype=torch.bool)

    graphs = None
    if a.encoder != 'stored':
        from models.encoder_cache import DEFAULT_CKPT, load_encoder
        if a.encoder == 'pretrained':
            a.encoder_ckpt = a.encoder_ckpt or DEFAULT_CKPT
            bundle = load_encoder(a.encoder_ckpt, 'cpu')
            encoding, k, stereo, spd = (bundle['cfg']['encoding'], bundle['k'],
                                        bundle['stereo_features'], bundle['cfg']['spd'])
        else:
            encoding, k, stereo, spd = 'rwse', a.enc_k, a.stereo_features, True
        t = time.time()
        members = multi._members
        graphs = GraphTable(smiles, [members[i].spec.perm for i in idents],
                            [members[i].spec.z for i in idents], encoding, k, stereo, spd, dev)
        print(f'encoder inputs for {C} conditions (stereo features {stereo}) in '
              f'{time.time() - t:.0f} s', flush=True)

    # --- split
    held_mol = molecule_split(smiles, a.heldout_frac)
    g = torch.Generator().manual_seed(a.seed)
    held_row = torch.rand(n_rows, generator=g) < a.conformer_frac
    row_mol_out = held_mol[row_slot]
    pools = {'train': torch.nonzero(~row_mol_out & ~held_row).flatten(),
             'conf': torch.nonzero(~row_mol_out & held_row).flatten(),
             'mol': torch.nonzero(row_mol_out).flatten()}
    n_mol = len({s for s in smiles})
    print(f'{C} conditions ({n_mol} distinct SMILES), {n_rows} stored rows, width {K}. '
          f'Held-out molecules: {int(held_mol.sum())} conditions. Rows -- train '
          f'{len(pools["train"])}, held-out conformers {len(pools["conf"])}, held-out molecules '
          f'{len(pools["mol"])}', flush=True)
    if min(len(v) for v in pools.values()) == 0:
        raise SystemExit('an empty split; the set is too small for these fractions')

    # the file is written in float64 and the energy runs in float32 (ConformerModeller._as_run_dtype)
    for key, val in list(cond._store.items()):
        if torch.is_tensor(val) and val.is_floating_point() and val.dtype != torch.float32:
            cond[key] = val.float()
    cond = cond.to(dev)
    states, sigma, row_slot_d = states.to(dev), sigma.to(dev), row_slot.to(dev)
    ref, periodic = ref.to(dev), periodic.to(dev)
    lo, hi = math.log10(a.noise_range[0]), math.log10(a.noise_range[1])

    def draw(rows, gen):
        """Noised states of stored ``rows`` and their targets."""
        slots = row_slot_d[rows]
        mult = 10 ** (lo + (hi - lo) * torch.rand(len(rows), 1, device=dev, generator=gen))
        x = states[rows] + mult * sigma[slots] * torch.randn(len(rows), K, device=dev, generator=gen)
        x = torch.where(periodic, (x + 1) % 2 - 1, x.clamp(-1, 1))
        x = x * (sigma[slots] > 0) + states[rows] * (sigma[slots] == 0)
        sub = cond.subsample_new_batch(slots)
        with torch.no_grad():
            _, out = multi.energy(x, sub, return_exp=True)
        return x, sub, slots, out.conformer_energy.flatten().float() - ref[slots]

    # --- the fixed test sets: the same draws on every start
    eg = torch.Generator(device=dev).manual_seed(a.seed + 1)
    cg = torch.Generator().manual_seed(a.seed + 2)
    tests = {}
    for name, pool in (('seen', pools['train']), ('conf', pools['conf']), ('mol', pools['mol'])):
        rows = pool[torch.randint(0, len(pool), (a.eval_samples,), generator=cg)].to(dev)
        chunks = [draw(rows[i:i + a.batch], eg) for i in range(0, len(rows), a.batch)]
        tests[name] = [(x, slots, y) for x, _, slots, y in chunks]
        y = torch.cat([c[2] for c in tests[name]])
        q = torch.quantile(y.double().cpu(), torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()
        print(f'test set {name}: {len(y)} samples, target p5 {q[0]:.1f} p50 {q[1]:.1f} p95 '
              f'{q[2]:.1f} max {float(y.max()):.1f} kcal/mol, {100 * float((y < BAND).float().mean()):.0f}% '
              f'below {BAND:g}', flush=True)

    model = build_model(a, cond, periodic.tolist(), C, K, graphs, dev)
    n_par = {k_: int(sum(p.numel() for p in m.parameters())) for k_, m in
             (('token network', model.policy), ('encoder', model.encoder)) if m is not None}
    print(f'model: {n_par}; encoder {a.encoder}', flush=True)
    if a.encoder == 'pretrained':
        # the encoder inside the step reproduces the embeddings the conditions file carries
        probe = torch.arange(min(64, C), device=dev)
        sub = cond.subsample_new_batch(probe)
        model.encoder.eval()
        with torch.no_grad():
            h, gm = model.embeddings(graphs, probe)
        model.encoder.train()
        dh = float((h - sub.atom_embedding.float()).abs().max())
        dg = float((gm - sub.embedding.reshape(len(probe), -1).float()).abs().max())
        print(f'pretrained encoder against the stored embeddings on {len(probe)} conditions: max '
              f'|difference| per-atom {dh:.2e}, pooled {dg:.2e}', flush=True)
        if max(dh, dg) > 1e-3:
            raise SystemExit('the encoder checkpoint is not the one the conditions file was '
                             'embedded with (or the atom order differs)')

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
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        tg.set_state(ck['noise_generator'].cpu())
        ig.set_state(ck['index_generator'].cpu())
        step0 = int(ck['step'])
        print(f'RESUMED {run_path} at step {step0}', flush=True)
    wb = None
    if a.wandb_project:
        # the curves are a convenience: the table below and the history file are the record
        try:
            import wandb
            wb = wandb.init(project=a.wandb_project, name=a.name, id=a.name, resume='allow',
                            tags=['energy_probe'], config=vars(a))
        except Exception as exc:                                  # noqa: BLE001
            print(f'wandb unavailable ({type(exc).__name__}: {exc}); continuing without it', flush=True)

    def evaluate():
        model.eval()
        res = {}
        with torch.no_grad():
            for name, chunks in tests.items():
                pr, ys, sl = [], [], []
                for x, slots, y in chunks:
                    pr.append(model(x, cond.subsample_new_batch(slots), graphs, slots))
                    ys.append(y); sl.append(slots)
                res[name] = metrics(torch.cat(pr).cpu(), torch.cat(ys).cpu(), torch.cat(sl).cpu())
        model.train()
        return res

    def save(step):
        tmp = run_path + '.tmp'
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step,
                    'noise_generator': tg.get_state().cpu(),
                    'index_generator': ig.get_state().cpu(),
                    'args': vars(a)}, tmp)
        os.replace(tmp, run_path)

    print('\n' + TABLE_CAPTION + f' Set: {a.rung}; arm {a.name}; {a.eval_samples} samples per test set.')
    print(table_header(), flush=True)
    model.train()
    train_pool = pools['train']
    t_last, s_last, loss_acc, loss_n = time.time(), step0, 0.0, 0
    for step in range(step0, a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            res = evaluate()
            rate = (step - s_last) / max(time.time() - t_last, 1e-9)
            mean_loss = loss_acc / max(loss_n, 1)
            print(table_row(step, mean_loss, res, rate), flush=True)
            with open(hist_path, 'a') as f:
                f.write(json.dumps({'step': step, 'loss': mean_loss, 'it_per_s': rate,
                                    'lr': lr_at(step), **res}) + '\n')
            if wb is not None:
                wb.log({'loss': mean_loss, 'lr': lr_at(step), 'it_per_s': rate,
                        **{f'{n_}/{k_}': v for n_, r in res.items() for k_, v in r.items()}}, step=step)
            t_last, s_last, loss_acc, loss_n = time.time(), step, 0.0, 0
        if step == a.steps:
            break
        for grp in opt.param_groups:
            grp['lr'] = lr_at(step)
        rows = train_pool[torch.randint(0, len(train_pool), (a.batch,), generator=ig)].to(dev)
        x, sub, slots, y = draw(rows, tg)
        opt.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(model(x, sub, graphs, slots) / LOSS_SCALE,
                                                  y / LOSS_SCALE, beta=0.2)
        if not torch.isfinite(loss):
            raise SystemExit(f'non-finite loss at step {step}')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        loss_acc += float(loss.detach()); loss_n += 1
        if (step + 1) % a.save_every == 0:
            save(step + 1)
    save(a.steps)
    final = os.path.join(a.out_dir, f'{a.name}_final.pt')
    torch.save({'model': model.state_dict(), 'args': vars(a), 'results': res, 'step': a.steps}, final)
    print(f'wrote {final}', flush=True)


if __name__ == '__main__':
    main()
