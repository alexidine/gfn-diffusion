"""Space groups as sets of symmetry operations, for the AR policy (docs/design/conditional_ar_architecture.md §5.2).

A group g is the set of its affine operations (M, m) in the standard setting (mxtaltools.constants.space_group_info
SYM_OPS: M the 3x3 integer rotation in the fractional basis, m the translation mod 1, centring included). Groups that
share operations share their tokens, which is the route to a sensible input for a group not trained on.

- ``op_features(M, m)``: one operation's features (OP_FEATS): M's 9 entries, sin/cos of 2 pi m, det M, the rotation
  order, the intrinsic (screw/glide) translation (1/n) sum_k M^k m mod 1 as sin/cos, and a pure-translation flag.
- ``group_tokens(sg)``: [n_ops, OP_DIM] float32 of the group's operations, identity included.
- ``SymOpEncoder``: a per-operation MLP then mxtaltools' AugSoftmaxAggregation over the group's operations -> e_g.
- ``SymOpAutoencoder``: the encoder plus a decoder giving one membership logit per operation of the vocabulary (the
  distinct operations of all 230 groups, translations on a 1/24 grid); ``pretrain`` fits it by binary cross-entropy
  on the 230 groups, no energy data. Its encoder is the pretrained e_g.

Not yet read by the policy (the SG step is built later; §2.4)."""
import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

OP_FEATS = tuple([f'M{i}{j}' for i in range(3) for j in range(3)] + ['sin_m0', 'sin_m1', 'sin_m2', 'cos_m0', 'cos_m1',
                 'cos_m2', 'det', 'order', 'sin_tau0', 'sin_tau1', 'sin_tau2', 'cos_tau0', 'cos_tau1', 'cos_tau2',
                 'translation'])
OP_DIM = len(OP_FEATS)
GRID = 24   # every SYM_OPS translation is a multiple of 1/24 (checked when the vocabulary is built)


def _key(M, m):
    """(M as 9 ints, m on the 1/GRID grid mod GRID): an operation's identity."""
    mm = np.mod(np.round(np.asarray(m, dtype=float) * GRID), GRID).astype(int)
    return tuple(np.round(M).astype(int).ravel().tolist()), tuple(mm.tolist())


def _rotation_order(M):
    P, eye = np.eye(3), np.eye(3)
    for n in range(1, 7):
        P = P @ M
        if np.allclose(P, eye):
            return n
    raise ValueError(f'a rotation of order above 6: {M.tolist()}')


def op_features(M, m):
    """[OP_DIM] float64 features of one operation (module docstring)."""
    M, m = np.asarray(M, dtype=float), np.mod(np.asarray(m, dtype=float), 1.0)
    n = _rotation_order(M)
    tau, P = np.zeros(3), np.eye(3)
    for _ in range(n):            # intrinsic translation: (1/n) sum_k M^k m
        tau += P @ m
        P = M @ P
    tau = np.mod(tau / n, 1.0)
    trans = float(np.allclose(M, np.eye(3)) and np.abs(m).max() > 1e-9)
    return np.concatenate([M.ravel(), np.sin(2 * np.pi * m), np.cos(2 * np.pi * m), [np.linalg.det(M), n],
                           np.sin(2 * np.pi * tau), np.cos(2 * np.pi * tau), [trans]])


def group_ops(sg):
    """[(M [3, 3], m [3])] of space group sg, translations mod 1."""
    from mxtaltools.constants.space_group_info import SYM_OPS
    out = []
    for A in SYM_OPS[int(sg)]:
        A = np.asarray(A, dtype=float)
        out.append((A[:3, :3], np.mod(A[:3, 3], 1.0)))
    return out


@lru_cache(maxsize=None)
def group_tokens(sg) -> torch.Tensor:
    """[n_ops, OP_DIM] float32 features of the group's operations, identity included."""
    return torch.tensor(np.stack([op_features(M, m) for M, m in group_ops(sg)]), dtype=torch.float32)


@lru_cache(maxsize=None)
def vocabulary():
    """(keys, index) of the distinct operations of all 230 groups: keys a sorted tuple of _key's, index key -> row."""
    keys = set()
    for g in range(1, 231):
        for M, m in group_ops(g):
            if not np.allclose(np.round(m * GRID), m * GRID, atol=1e-6):
                raise ValueError(f'sg {g}: a translation {m.tolist()} off the 1/{GRID} grid')
            keys.add(_key(M, m))
    keys = tuple(sorted(keys))
    return keys, {k: i for i, k in enumerate(keys)}


def membership(sg) -> torch.Tensor:
    """[len(vocabulary)] 0/1 float32: which operations of the vocabulary the group has."""
    keys, index = vocabulary()
    out = torch.zeros(len(keys))
    for M, m in group_ops(sg):
        out[index[_key(M, m)]] = 1.0
    return out


def batch_tokens(groups):
    """(tokens [n_total, OP_DIM], group index [n_total]) of a list of space groups, for the encoder."""
    toks = [group_tokens(int(g)) for g in groups]
    idx = torch.cat([torch.full((t.shape[0],), i, dtype=torch.long) for i, t in enumerate(toks)])
    return torch.cat(toks), idx


class SymOpEncoder(nn.Module):
    """e_g [n_groups, out_dim] from the operations of each group: an MLP per operation, then AugSoftmaxAggregation
    (sum_i x_i (softmax_i(t x_i) + b) per channel) over the group's operations, then a linear map."""

    def __init__(self, out_dim=64, hidden=128):
        super().__init__()
        from mxtaltools.models.modules.augmented_softmax_aggregator import AugSoftmaxAggregation
        self.op = nn.Sequential(nn.Linear(OP_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                nn.Linear(hidden, hidden))
        self.pool = AugSoftmaxAggregation(channels=hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, tokens, index, n_groups):
        return self.out(self.pool(self.op(tokens), index, dim_size=int(n_groups)))

    def embed(self, groups):
        """e_g of a list of space groups."""
        tok, idx = batch_tokens(groups)
        p = next(self.parameters())
        return self(tok.to(p), idx.to(p.device), len(groups))


class SymOpAutoencoder(nn.Module):
    """SymOpEncoder plus a decoder from e_g to one membership logit per vocabulary operation."""

    def __init__(self, out_dim=64, hidden=128):
        super().__init__()
        self.encoder = SymOpEncoder(out_dim, hidden)
        self.decoder = nn.Sequential(nn.Linear(out_dim, hidden), nn.SiLU(), nn.Linear(hidden, len(vocabulary()[0])))

    def forward(self, groups):
        return self.decoder(self.encoder.embed(groups))


def pretrain(steps=4000, lr=3e-3, out_dim=64, hidden=256, seed=0, log_every=0):
    """A SymOpAutoencoder fitted to the membership of all 230 groups (full batch, Adam). Returns (model, {'loss':
    final BCE, 'bit_errors': wrong membership bits at threshold 0.5, 'groups_exact': groups reconstructed exactly})."""
    torch.manual_seed(seed)
    model = SymOpAutoencoder(out_dim, hidden)
    groups = list(range(1, 231))
    target = torch.stack([membership(g) for g in groups])
    pos = target.mean()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(steps))
    lossf = nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)   # members are ~2% of the bits
    for step in range(int(steps)):
        opt.zero_grad()
        loss = lossf(model(groups), target)
        loss.backward()
        opt.step()
        sched.step()
        if log_every and step % log_every == 0:
            print(f'step {step} bce {float(loss.detach()):.4f}')
    with torch.no_grad():
        wrong = ((model(groups) > 0).float() != target)
    return model, {'loss': float(loss.detach()), 'bit_errors': int(wrong.sum()),
                   'groups_exact': int((~wrong.any(-1)).sum())}


def save_encoder(model, path):
    """The pretrained encoder's weights and shape, for nf sym_ops_init."""
    enc = model.encoder
    torch.save({'state_dict': enc.state_dict(), 'out_dim': enc.out.out_features,
                'hidden': enc.out.in_features, 'vocabulary': len(vocabulary()[0])}, path)


def encoder_shape(path):
    """(e_g width, hidden width) of a save_encoder file."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return int(ck['out_dim']), int(ck['hidden'])


def load_encoder_into(encoder, path):
    """Load a save_encoder file into encoder (a SymOpEncoder of the same shape); raises on a shape mismatch."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if (ck['out_dim'], ck['hidden']) != (encoder.out.out_features, encoder.out.in_features):
        raise ValueError(f"{path}: an e_g encoder of width {ck['out_dim']} (hidden {ck['hidden']}), the flow's is "
                         f"{encoder.out.out_features} (hidden {encoder.out.in_features})")
    encoder.load_state_dict(ck['state_dict'])


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='pretrain the space-group autoencoder and save its encoder')
    ap.add_argument('--out', required=True)
    ap.add_argument('--out_dim', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--steps', type=int, default=4000)
    a = ap.parse_args()
    m, res = pretrain(steps=a.steps, out_dim=a.out_dim, hidden=a.hidden, log_every=500)
    print(res)
    save_encoder(m, a.out)
