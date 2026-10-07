"""Re-embed crystal condition files with the intra trunk: the molecule condition read off the trunk.

    python -u build_trunk_conditions.py --trunk <stacked trunk checkpoint> --out-dir <dir> <file.pt> [<file.pt> ...]

A crystal run's molecule condition ships as a vector baked onto every row of its prior and
condition files (`embedding`; `MolecularCrystal.condition_samples` flattens it and
`embedding_conditioning_dim` names its width). Those vectors were a frozen Mo3ENet's. This
writes copies of the files whose `embedding` is instead the intra trunk's description of the
row's molecule:

    per-atom states of the isolated molecule, as the stacked trunk reads them (IntraTrunk, over
    its stored scale)  ->  [mean over atoms, sum over atoms / 16]          width 2 * node_dim

Nothing else in a file changes, so a run takes the new condition by pointing `prior_path`,
`molecules_path` and `test_molecules_path` at the copies and setting
`embedding_conditioning_dim` to the printed width. The trunk is frozen, so embedding once is
embedding at every step; a row depends on its own molecule only (no normalisation layer, no
batch statistic), which the Mo3ENet vectors did not offer.
"""
import argparse
import os

import torch

from models.stacked_trunk import StackedTrunk

#: divides the summed pool, as models.graph_state.SUM_SCALE does
SUM_SCALE = 16.0


def load_trunk(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('intra_trunk_args') is None:
        raise SystemExit(f'{path} is not a stacked trunk: it holds no intra trunk')
    a = ck['args']
    trunk = StackedTrunk(ck['intra_trunk_args'], node_dim=a['node_dim'], message_dim=a['message_dim'],
                         num_convs=a['num_convs'], cutoff=a['feature_cutoff'], folded=not a['unfolded'])
    trunk.load_state_dict(ck['model'])
    return trunk.to(device).requires_grad_(False).eval()


def molecule_conditions(trunk, batch, device, chunk=4096):
    """[num_graphs, 2 * node_dim]: the pooled intra states of every row's molecule."""
    ptr = batch.ptr
    n = ptr[1:] - ptr[:-1]
    out = []
    for lo in range(0, batch.num_graphs, chunk):
        hi = min(lo + chunk, batch.num_graphs)
        width = int(n[lo:hi].max())
        ar = torch.arange(width)
        mask = ar[None] < n[lo:hi, None]
        at = (ptr[lo:hi, None] + ar[None]).clamp(max=batch.pos.shape[0] - 1)
        z = (batch.z[at].long() * mask).to(device)
        pos = (batch.pos[at] * mask[..., None]).to(device)
        mask = mask.to(device)
        states = trunk.molecule_states(z, pos, mask)
        graph = mask.nonzero(as_tuple=True)[0]
        total = torch.zeros(hi - lo, states.shape[1], dtype=states.dtype, device=device).index_add_(0, graph, states)
        out.append(torch.cat((total / n[lo:hi, None].to(total), total / SUM_SCALE), dim=1).cpu())
    return torch.cat(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+', help='prior / condition files to re-embed')
    ap.add_argument('--trunk', required=True, help='stacked trunk checkpoint (pretrain_atom_trunk.py --intra)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--key', default='prior', help='key of the crystal batch inside each file')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    trunk = load_trunk(a.trunk, a.device)
    for path in a.files:
        blob = torch.load(path, map_location='cpu', weights_only=False)
        batch = blob[a.key]
        emb = molecule_conditions(trunk, batch, a.device)
        if not bool(torch.isfinite(emb).all()):
            raise SystemExit(f'{path}: non-finite conditions')
        old = tuple(batch.embedding.shape) if hasattr(batch, 'embedding') else None
        batch.embedding = emb
        blob['embedding_dim'] = int(emb.shape[1])
        blob['encoder'] = f'intra trunk of {os.path.basename(a.trunk)}: [mean, sum / {SUM_SCALE:g}] of its per-atom states'
        out = os.path.join(a.out_dir, os.path.basename(path))
        torch.save(blob, out)
        check = batch.subsample_new_batch(torch.tensor([0, batch.num_graphs - 1]))
        assert torch.equal(check.embedding.reshape(2, -1), emb[[0, batch.num_graphs - 1]]), 'a row lost its condition'
        distinct = torch.unique(emb.round(decimals=4), dim=0).shape[0]
        print(f"[trunk-conditions] {path}: {batch.num_graphs} rows, embedding {old} -> {tuple(emb.shape)} "
              f"(embedding_conditioning_dim: {emb.shape[1]}); {distinct} distinct conditions; root mean square "
              f"{float(emb.square().mean().sqrt()):.3f} -> {out}", flush=True)


if __name__ == '__main__':
    main()
