"""
Per-molecule atlas of one checkpoint: for a seeded set of training and held-out molecules,
the model's draws, the molecule's descriptors, its place in condition space and the run's own
level estimates, written once so that every trend analysis reads the same rows.

WHICH MOLECULES. `--n-test` held-out molecules and `--n-train` training molecules drawn at
random, plus the nearest training molecule (in the checkpoint's conditioner output) of every
one of those. The neighbours are what make a held-out reading interpretable: a held-out
molecule is compared with a training molecule that looks like it to the model, and the same
comparison between a training molecule and ITS nearest training neighbour is the control.

WHAT IS STORED, per molecule: identifier, split, RDKit graph descriptors and functional-group
flags, shape descriptors of the stored geometry, the encoder embedding (192) and the
conditioner output, the tracker's per-condition levels, the head's log Z, the energy
reference, and the molecule's search minima (count, gaps, the best one's latents). Per draw
(`--draws` each, through eval/cond_panel/sampler.py::draw): the built crystal's latents, log
P_F, log P_B, log R, the lattice energy, packing coefficient, reduction penalty and the
trainer's reasonable flag.

RESUMABLE. Draws are written a chunk of molecules at a time under <out>/chunks and a rerun
skips the chunks already on disk; the selection is written first and reloaded, so a restart
continues the same atlas. `--assemble` (run automatically at the end) joins the chunks.

    cd energy_sampling
    CUDA_VISIBLE_DEVICES=-1 python -m eval.cond_panel.atlas --checkpoint D:/.../<stem>_step50000.pt \
        --config <local yaml> --test-prior D:/.../qm9full_test_prior.pt --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from eval.cond_panel.sampler import draw, head_log_z, load_run, reasonable_mask
from eval.cond_panel.select_panel import conditioner_coords, graph_descriptors, shape_descriptors

RDLogger.DisableLog('rdApp.*')

# SMARTS over the QM9 element set (C, N, O, F, H); a molecule can carry several
FUNCTIONAL_GROUPS = {
    'hydroxyl': '[OX2H][#6;!$(C=O)]',
    'carboxylic acid': 'C(=O)[OX2H1]',
    'ester': '[#6]C(=O)O[#6]',
    'ether': '[OD2]([#6;!$(C=O)])[#6;!$(C=O)]',
    'aldehyde': '[CX3H1](=O)',
    'ketone': '[#6][CX3](=O)[#6]',
    'amide': 'C(=O)[NX3]',
    'primary amine': '[NX3;H2;!$(NC=[O,N])][#6]',
    'secondary amine': '[NX3;H1;!$(NC=[O,N])]([#6])[#6]',
    'tertiary amine': '[NX3;H0;!$(NC=[O,N])]([#6])([#6])[#6]',
    'nitrile': 'C#N',
    'alkyne': 'C#C',
    'alkene': '[CX3]=[CX3]',
    'imine': '[CX3]=[NX2]',
    'fluorine': '[F]',
    'aromatic ring': 'a',
    'three-membered ring': '[r3]',
    'four-membered ring': '[r4]',
}
RING_FLAGS = ('bridged', 'spiro', 'any ring', 'acyclic')
TRACKER_KEYS = ('ema_logw', 'fwd_level_ema', 'bwd_level_ema', 'count', 'fwd_level_visits', 'bwd_level_visits',
                'best_energy', 'z_grad_ema', 'z_bias_ema')
DRAW_KEYS = ('latent', 'log_pf', 'log_pb', 'log_r', 'mol_energy', 'seed_energy', 'packing_coeff', 'reduction_en',
             'reasonable')


def group_flags(smiles):
    """Functional-group and ring-topology flags of one SMILES, in FUNCTIONAL_GROUPS + RING_FLAGS order."""
    mol = Chem.MolFromSmiles(smiles)
    flags = [mol.HasSubstructMatch(_PATTERNS[name]) for name in FUNCTIONAL_GROUPS]
    n_ring = rdMolDescriptors.CalcNumRings(mol)
    flags += [rdMolDescriptors.CalcNumBridgeheadAtoms(mol) > 0, rdMolDescriptors.CalcNumSpiroAtoms(mol) > 0,
              n_ring > 0, n_ring == 0]
    return flags, n_ring


_PATTERNS = {name: Chem.MolFromSmarts(s) for name, s in FUNCTIONAL_GROUPS.items()}
assert all(p is not None for p in _PATTERNS.values()), 'a functional-group SMARTS did not parse'


def nearest_training(query, train, own_col=None, chunk=512):
    """Index and distance of each query row's nearest row of `train`; `own_col` gives, for a
    query that IS a training row, its own column, masked by index (cdist's matrix-product
    form returns a self-distance of order 1e-3, so a `d > 0` test would keep the row itself)."""
    idx = torch.empty(query.shape[0], dtype=torch.long)
    dist = torch.empty(query.shape[0])
    for start in range(0, query.shape[0], chunk):
        d = torch.cdist(query[start:start + chunk].float(), train.float())
        if own_col is not None:
            rows = torch.arange(d.shape[0])
            d[rows, own_col[start:start + chunk]] = float('inf')
        dist[start:start + chunk], idx[start:start + chunk] = d.min(dim=1)
    return idx, dist


def select(run, n_train, n_test, seed):
    """The atlas molecules: (split, row) per molecule, the two neighbour pair lists, and every
    molecule's conditioner coordinates."""
    e_tr = conditioner_coords(run, run.conditions)
    e_te = conditioner_coords(run, run.test_conditions)
    g = torch.Generator().manual_seed(seed)
    held = torch.randperm(run.test_conditions.num_graphs, generator=g)[:n_test]
    rand = torch.randperm(run.conditions.num_graphs, generator=g)[:n_train]
    nn_held, d_held = nearest_training(e_te[held], e_tr)
    nn_rand, d_rand = nearest_training(e_tr[rand], e_tr, own_col=rand)

    train_rows = torch.unique(torch.cat([rand, nn_held, nn_rand]))
    pos = {int(r): i for i, r in enumerate(train_rows.tolist())}            # training row -> atlas index
    n_tr = train_rows.numel()
    sel = {
        'train_rows': train_rows, 'held_rows': held,
        'is_random_train': torch.isin(train_rows, rand),
        # pairs are (atlas index, atlas index of its nearest training molecule, conditioner distance)
        'pairs_held': torch.tensor([[n_tr + i, pos[int(j)]] for i, j in enumerate(nn_held.tolist())]),
        'pairs_train': torch.tensor([[pos[int(r)], pos[int(j)]] for r, j in zip(rand.tolist(), nn_rand.tolist())]),
        'pair_dist_held': d_held, 'pair_dist_train': d_rand,
        'cond_emb': torch.cat([e_tr[train_rows], e_te[held]]).float(),
    }
    return sel


def reference_table(batch, identifiers, temperature):
    """Per identifier, from a reference batch (prior rows): how many search minima it has,
    their spacing, and the best one's packing coefficient and latents."""
    rows = {}
    for i, ident in enumerate(batch.identifier):
        rows.setdefault(ident, []).append(i)
    e_all = (batch.elj.double().flatten() / batch.z_prime.double().flatten())
    out = {k: np.full(len(identifiers), np.nan) for k in ('n_ref', 'ref_e_min', 'ref_gap2_kT', 'ref_within_5kT',
                                                         'ref_spread_kT', 'ref_cp_best')}
    best_rows = []
    for m, ident in enumerate(identifiers):
        r = rows.get(ident)
        if not r:
            best_rows.append(-1)
            continue
        e = e_all[r]
        order = torch.argsort(e)
        out['n_ref'][m] = len(r)
        out['ref_e_min'][m] = float(e[order[0]])
        out['ref_gap2_kT'][m] = float((e[order[1]] - e[order[0]]) / temperature) if len(r) > 1 else np.nan
        out['ref_within_5kT'][m] = float(((e - e[order[0]]) / temperature <= 5.0).sum())
        out['ref_spread_kT'][m] = float((e.max() - e.min()) / temperature)
        out['ref_cp_best'][m] = float(batch.packing_coeff.flatten()[r[int(order[0])]])
        best_rows.append(r[int(order[0])])
    have = [b for b in best_rows if b >= 0]
    lat = np.full((len(identifiers), 12), np.nan, dtype=np.float32)
    if have:
        best = batch.subsample_new_batch(torch.tensor(have))
        lat[[m for m, b in enumerate(best_rows) if b >= 0]] = best.latent_params().detach().cpu().numpy()
    out['ref_best_latent'] = lat
    return out


def molecule_table(run, sel, prior_path, test_prior_path):
    """Everything stored per molecule except the draws."""
    tr, te = run.conditions, run.test_conditions
    parts = [(tr, sel['train_rows'], 'train'), (te, sel['held_rows'], 'held_out')]
    ident, split, cid, enc, head = [], [], [], [], []
    desc = {}
    flags, n_ring = [], []
    for batch, rows, name in parts:
        sub = batch.subsample_new_batch(rows)
        ident += list(sub.identifier)
        split += [name] * rows.numel()
        cid.append(torch.tensor([run.registry[i] for i in sub.identifier]))
        enc.append(sub.embedding.reshape(sub.num_graphs, -1).float())
        head.append(head_log_z(run, sub))
        for i, smi in enumerate(sub.identifier):
            gd = graph_descriptors(smi)
            sl = slice(int(sub.ptr[i]), int(sub.ptr[i + 1]))
            row = {k: v for k, v in gd.items() if k != 'generic_key'}
            row.update(shape_descriptors(sub.pos[sl].double(), sub.z[sl]))
            row['radius'] = float(sub.radius[i])
            row['mol_volume'] = float(sub.mol_volume[i])
            row['n_atoms'] = int(sub.num_atoms[i])
            for k, v in row.items():
                desc.setdefault(k, []).append(float(v))
            f, nr = group_flags(smi)
            flags.append(f)
            n_ring.append(nr)
    cid = torch.cat(cid)
    desc['n_rings'] = n_ring
    table = {
        'identifier': ident, 'split': split, 'condition_id': cid.numpy(),
        'is_random_train': np.concatenate([sel['is_random_train'].numpy(), np.zeros(sel['held_rows'].numel(), bool)]),
        'descriptors': {k: np.asarray(v, dtype=np.float64) for k, v in desc.items()},
        'group_names': list(FUNCTIONAL_GROUPS) + list(RING_FLAGS),
        'groups': np.asarray(flags, dtype=bool),
        'enc_emb': torch.cat(enc).numpy(), 'cond_emb': sel['cond_emb'].numpy(),
        'head_log_z': torch.cat(head).numpy(),
        'e_ref': run.energy_function.energy_reference_for(cid).numpy() if run.energy_function.energy_reference is not None else None,
        'tracker': {k: run.tracker[k][cid].double().numpy() for k in TRACKER_KEYS if k in run.tracker},
    }
    n_tr = sel['train_rows'].numel()
    refs = {}
    for path, lo, hi in ((prior_path, 0, n_tr), (test_prior_path, n_tr, len(ident))):
        if path is None or hi == lo:
            continue
        data = torch.load(path, map_location='cpu', weights_only=False)
        batch = data.get('equalized_prior', data.get('prior')) if isinstance(data, dict) else data
        part = reference_table(batch, ident[lo:hi], run.temperature)
        for k, v in part.items():
            refs.setdefault(k, np.full((len(ident),) + v.shape[1:], np.nan, dtype=v.dtype))[lo:hi] = v
        del data, batch
    table['reference'] = refs
    return table


def draw_chunk(run, batch, rows, n_draws, batch_size, seed):
    """`n_draws` draws for each conditions row in `rows`; arrays are [molecule, draw, ...]."""
    d = draw(run, batch.subsample_new_batch(rows.repeat_interleave(n_draws)), batch_size=batch_size, seed=seed)
    ef = run.energy_function
    out = {
        'latent': d['sample_batch'].latent_params().detach().cpu().float(),
        'log_pf': d['log_pf'].float(), 'log_pb': d['log_pb'].float(), 'log_r': d['log_r'].float(),
        'mol_energy': d['mol_energy'].float(), 'seed_energy': ef.seed_energy_from(d).float(),
        'packing_coeff': d['packing_coeff'].float(), 'reduction_en': d['reduction_en'].float(),
        'reasonable': reasonable_mask(run, d['sample_batch']).cpu().bool(),
    }
    return {k: v.reshape(rows.numel(), n_draws, *v.shape[1:]).numpy() for k, v in out.items()}


def assemble(out_dir, step):
    table = torch.load(os.path.join(out_dir, 'molecules.pt'), weights_only=False)
    files = sorted(glob.glob(os.path.join(out_dir, 'chunks', 'chunk_*.pt')))
    chunks = [torch.load(f, weights_only=False) for f in files]
    order = np.concatenate([c['atlas_index'] for c in chunks])
    assert np.array_equal(np.sort(order), np.arange(len(table['identifier']))), \
        f'{len(order)} molecules drawn, {len(table["identifier"])} in the table: chunks are missing'
    draws = {k: np.concatenate([c[k] for c in chunks])[np.argsort(order)] for k in DRAW_KEYS}
    atlas = dict(table, draws=draws, step=step)
    path = os.path.join(out_dir, f'atlas_step{step}.pt')
    torch.save(atlas, path)
    print(f'atlas written: {path} ({len(order)} molecules x {draws["log_r"].shape[1]} draws)', flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--test-prior', default=None, help='reference rows of the held-out molecules')
    ap.add_argument('--n-train', type=int, default=2000)
    ap.add_argument('--n-test', type=int, default=2000)
    ap.add_argument('--draws', type=int, default=24)
    ap.add_argument('--chunk-molecules', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=960)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0, help='torch threads; 0 leaves the default')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    if args.device == 'cpu':
        assert not torch.cuda.is_available(), 'CPU run with a visible GPU: set CUDA_VISIBLE_DEVICES=-1'
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(os.path.join(args.out, 'chunks'), exist_ok=True)

    run = load_run(args.checkpoint, args.config, device=args.device)
    sel_path = os.path.join(args.out, 'selection.pt')
    if os.path.exists(sel_path):
        sel = torch.load(sel_path, weights_only=False)
        assert sel['args'] == (args.n_train, args.n_test, args.seed, run.step), \
            f'{sel_path} was written for {sel["args"]}; use another --out for a different atlas'
    else:
        sel = select(run, args.n_train, args.n_test, args.seed)
        sel['args'] = (args.n_train, args.n_test, args.seed, run.step)
        torch.save(sel, sel_path)
    n_tr, n_te = sel['train_rows'].numel(), sel['held_rows'].numel()
    print(f'atlas at step {run.step}: {n_tr} training molecules ({int(sel["is_random_train"].sum())} random, the rest '
          f'nearest neighbours) and {n_te} held-out, {args.draws} draws each', flush=True)

    mol_path = os.path.join(args.out, 'molecules.pt')
    if not os.path.exists(mol_path):
        table = molecule_table(run, sel, run.config['prior_path'], args.test_prior)
        table.update(pairs_held=sel['pairs_held'].numpy(), pairs_train=sel['pairs_train'].numpy(),
                     pair_dist_held=sel['pair_dist_held'].numpy(), pair_dist_train=sel['pair_dist_train'].numpy(),
                     temperature=run.temperature, data_ndim=run.energy_function.data_ndim,
                     checkpoint=os.path.basename(args.checkpoint), n_draws=args.draws, seed=args.seed)
        torch.save(table, mol_path)
        print(f'molecule table written: {mol_path}', flush=True)

    # molecule-major chunks over (split batch, conditions row, atlas index)
    jobs = [(run.conditions, sel['train_rows'], 0), (run.test_conditions, sel['held_rows'], n_tr)]
    k, t0, done = 0, time.time(), 0
    for batch, rows, offset in jobs:
        for start in range(0, rows.numel(), args.chunk_molecules):
            path = os.path.join(args.out, 'chunks', f'chunk_{k:05d}.pt')
            sub = rows[start:start + args.chunk_molecules]
            if not os.path.exists(path):
                out = draw_chunk(run, batch, sub, args.draws, args.batch_size, args.seed + 1000 + k)
                out['atlas_index'] = np.arange(offset + start, offset + start + sub.numel())
                torch.save(out, path + '.tmp')
                os.replace(path + '.tmp', path)
                done += sub.numel() * args.draws
                if k % 10 == 0:
                    rate = done / max(time.time() - t0, 1e-9)
                    print(f'chunk {k}: {rate:.0f} draws/s this session', flush=True)
            k += 1
    assemble(args.out, run.step)


if __name__ == '__main__':
    main()
