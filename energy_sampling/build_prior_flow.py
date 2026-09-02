r"""
Fit the density proxy for a trained prior policy, from a checkpoint.

    python build_prior_flow.py \
        --checkpoint D:\crystal_datasets\gfn_checkpoints\<phase1_exit>.pt \
        --conditions D:\crystal_datasets\conditional\priors\qm9c100k_conditions.pt \
        --out D:\crystal_datasets\conditional\priors\qm9c100k_prior_flow.pt

WHY IT SAMPLES THE POLICY RATHER THAN READING prior_path. The stored prior
DATASET is not the prior: 42% of its rows have a coordinate pinned to exactly
+-1 in a non-periodic dim, an artifact of how that file was built. The policy's
own draws have none, so the dataset's distribution is not absolutely continuous
and no density is even defined on it. Sampling the policy also lifts the data
limit -- the overfit gap fell 0.86 -> 0.20 nats going from the stored 62k rows
to 350k policy draws.

The geometry is read off the CONSTRUCTED policy (`ang_mask`, `dead_rows`) rather
than reconstructed from the space group, so there is no second derivation to
drift. T is taken from the checkpoint unless overridden, and is stored: the
terminal distribution depends on it and T is absent from problem_def, so nothing
else would catch a mismatch.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here),
           os.path.join(os.path.dirname(_here), 'mxtaltools')):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from energies.prior_flow import PriorFlow  # noqa: E402
from models.gfn import GFN  # noqa: E402
from utils import uniform_discretizer, get_gfn_init_state  # noqa: E402


def load_policy(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = dict(blob['gfn_config'])
    cfg['device'] = device
    gfn = GFN(**cfg).to(device)
    missing, unexpected = gfn.load_state_dict(blob['model_eval'], strict=False)
    if missing:
        raise ValueError(f'checkpoint is missing {len(missing)} policy tensors: {missing[:4]}')
    gfn.eval()
    return gfn, blob, cfg


def condition_embeddings(path):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    batch = blob.get('prior') if isinstance(blob, dict) else blob
    if batch is None or not hasattr(batch, 'embedding'):
        raise ValueError(f'{path} carries no `embedding`; it is not a conditions file')
    return batch.embedding.reshape(batch.num_graphs, -1).float()


@torch.no_grad()
def draw_terminals(gfn, emb, n, T, dim, device, batch=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    disc = (lambda bs: uniform_discretizer(bs, T))
    out, done = [], 0
    while done < n:
        b = min(batch, n - done)
        cond = emb[torch.randint(0, emb.shape[0], (b,), generator=g)].to(device)
        states, *_ = gfn.get_traj_fwd(get_gfn_init_state(b, dim, device), disc,
                                      None, cond, None, detach_traj=True)
        out.append(states[:, -1].detach().cpu())
        done += b
        print(f'\r  sampled {done}/{n}', end='', flush=True)
    print()
    return torch.cat(out)[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--conditions', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-samples', type=int, default=400_000)
    ap.add_argument('--traj-T', type=int, default=None, help='defaults to the checkpoint train_T')
    ap.add_argument('--blocks', type=int, default=4)
    ap.add_argument('--bins', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--time-budget', type=float, default=200.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--note', default='')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gfn, blob, cfg = load_policy(args.checkpoint, dev)
    T = int(args.traj_T if args.traj_T is not None else blob.get('train_T'))
    ang = gfn.ang_mask.detach().cpu().tolist()
    dead = tuple(int(r) for r in getattr(gfn, 'dead_rows', ()))
    print(f'policy: dim={cfg["dim"]} T={T} wrapped={[i for i,w in enumerate(ang) if w]} dead={dead or "()"}')

    emb = condition_embeddings(args.conditions)
    x = draw_terminals(gfn, emb, args.n_samples, T, cfg['dim'], dev, seed=args.seed)

    atoms = ((x == -1.0) | (x == 1.0)).any(1).float().mean()
    if atoms > 0.01:
        raise ValueError(
            f'{float(atoms)*100:.1f}% of policy draws have a coordinate at exactly +-1. '
            f'A continuous density is not defined on a distribution with atoms; refusing '
            f'to fit. (The stored prior DATASET has this; the policy should not.)')

    flow = PriorFlow(n_blocks=args.blocks, n_bins=args.bins,
                     hidden=(args.hidden, args.hidden), seed=args.seed,
                     time_budget=args.time_budget, device=dev, verbose=True)
    flow.fit(x, ang, period=2.0)
    flow.save(args.out, traj_T=T, dead_rows=dead, provenance={
        'checkpoint': os.path.basename(args.checkpoint),
        'conditions': os.path.basename(args.conditions),
        'n_samples': int(x.shape[0]), 'seed': int(args.seed), 'note': args.note,
        'problem_hash': blob.get('problem_hash'), 'run_name': blob.get('run_name'),
    })
    check = PriorFlow.load(args.out, device=dev)
    check.verify_against_policy(ang, dead_rows=dead, traj_T=T)
    held = x[:20000]
    print(f'\nwrote {args.out}')
    print(f'  {check.describe()}')
    print(f'  held-out mean energy {float(check.energy(held).mean()):.3f} nats/sample')
    print(f'  reload verified against the policy it was fitted to')


if __name__ == '__main__':
    main()
