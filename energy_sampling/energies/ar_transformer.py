"""
Causal-transformer conditioner for the autoregressive spline flow of energies/prior_flow.py.

A drop-in for `_MADE` inside `_Block`: it takes the block's conditioner features (one
column per linear dim, sin and cos columns per wrapped dim, as `_Block._feats` builds
them) and returns the spline parameters of every dim, laid out in DIM order, exactly as
`_MADE` does. The parameters of dim order[j] depend only on dims order[:j], so the block's
forward pass, `invert_block` and `invert_differentiable` work unchanged.

Layout: one token per slot of the order. Slot j predicts dim order[j] and carries the value
embedding of dim order[j-1]; slot 0 carries a learned start token. Pre-norm transformer
layers with causal self-attention, then a shared head. Every slot gets the full width,
unlike MADE, whose hidden units are split among degrees. With context_dim set, forward takes
the condition's embedding as a second argument (as _ContextMADE does) and adds its projection
to every token.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from energies.prior_flow import _SOFTPLUS_ONE


class _CausalLayer(nn.Module):
    def __init__(self, width, heads, mlp_ratio=4):
        super().__init__()
        if width % heads:
            raise ValueError(f"transformer width {width} is not a multiple of {heads} heads")
        self.heads = heads
        self.ln1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_ratio * width), nn.SiLU(),
                                 nn.Linear(mlp_ratio * width, width))

    def forward(self, h):
        b, n, w = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(b, n, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.proj(a.transpose(1, 2).reshape(b, n, w))
        return h + self.mlp(self.ln2(h))


class CausalTransformerConditioner(nn.Module):
    """Features of x in, spline parameters of every dim out (dim order), autoregressive in `order`."""

    def __init__(self, order, feat_dim, n_bins, width=256, layers=4, heads=8, context_dim=None):
        super().__init__()
        # conditional runs: the condition's embedding, projected to the width, is added to every token
        self.ctx = nn.Linear(context_dim, width) if context_dim else None
        d = len(order)
        self.d, self.n_params = d, 3 * n_bins + 1
        self.register_buffer('order', torch.as_tensor(order, dtype=torch.long))
        self.register_buffer('slot_of_dim', torch.argsort(torch.as_tensor(order, dtype=torch.long)))
        self.register_buffer('feat_dim', torch.as_tensor(feat_dim, dtype=torch.long))
        self.feat_w = nn.Parameter(torch.randn(len(feat_dim), width) / math.sqrt(2.0))
        self.dim_emb = nn.Parameter(torch.zeros(d, width))           # which dim a value belongs to
        self.slot_emb = nn.Parameter(0.02 * torch.randn(d, width))   # which dim a slot predicts
        self.start = nn.Parameter(torch.zeros(width))
        self.layers = nn.ModuleList(_CausalLayer(width, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(width)
        self.bf16 = True
        self.head = nn.Linear(width, self.n_params)
        # identity spline at step 0, as _MADE: zero weights, unit knot slopes, zero shift
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[2 * n_bins:3 * n_bins] = _SOFTPLUS_ONE

    def forward(self, feats, ctx=None):
        b = feats.shape[0]
        width = self.feat_w.shape[1]
        val = feats.new_zeros(b, self.d, width).index_add_(1, self.feat_dim, feats.unsqueeze(-1) * self.feat_w)
        val = val + self.dim_emb
        h = torch.cat((self.start.expand(b, 1, width), val[:, self.order[:-1]]), dim=1) + self.slot_emb
        if (ctx is None) != (self.ctx is None):
            raise ValueError("a conditional transformer conditioner needs ctx, an unconditional one takes none")
        if ctx is not None:
            h = h + self.ctx(ctx).unsqueeze(1)
        # layers in bf16 on CUDA (2x faster on the local card; causal masking stays exact),
        # head and spline in fp32. One block round-trips as well as fp32; stacked blocks
        # round-trip worse than fp32 (3 blocks, head std 0.02: median 1e-4 vs 5e-7 in base space)
        # bf16 = False (inn_modeller --tf_fp32) runs the layers in the input's precision: a trained stack of
        # blocks can fail the working-precision round trip in bf16
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=feats.is_cuda and self.bf16):
            for layer in self.layers:
                # under autograd each layer is recomputed in backward: a full-width layer's
                # internals at 8192 rows x 12 tokens are ~1.7 GB, stored they exceed a shared card
                h = checkpoint(layer, h, use_reentrant=False) if torch.is_grad_enabled() else layer(h)
        out = self.head(self.ln(h.to(feats.dtype)))       # [b, d, n_params], slot order
        return out[:, self.slot_of_dim].reshape(b, self.d * self.n_params)

    @torch.no_grad()
    def decode(self, feats_of, solve_row, b, dtype):
        """Autoregressive decoding with a key/value cache, one slot per step, for sampling.

        At step j the cache holds every layer's keys and values for slots 0..j-1. Slot j's token -- the start token at
        j = 0, else the value embedding of row order[j-1], fixed at the step before -- runs through each layer
        attending to the cache and to itself, the head gives row order[j]'s parameters, and `solve_row(row, params)`
        fixes that row. `feats_of()` returns the block's conditioner features [b, n_feat] of the rows fixed so far
        (`_Block._feats`); only the columns of the row just fixed are read. Each token's computation is the one
        forward() does for its slot on the completed rows, so the result matches the d-pass inverse (invert_block)
        to rounding, at about one forward pass of conditioner work instead of d. Unconditional conditioners only."""
        if self.ctx is not None:
            raise NotImplementedError("decode: a conditional conditioner's context is not carried; use the d-pass inverse")
        width = self.feat_w.shape[1]
        cache = [None] * len(self.layers)
        prev = None
        for j in range(self.d):
            if j == 0:
                tok = self.start.to(dtype).expand(b, 1, width)
            else:
                cols = self.feat_dim == prev
                f = feats_of()[:, cols]                                           # [b, columns of row prev]
                tok = ((f.unsqueeze(-1) * self.feat_w[cols]).sum(1) + self.dim_emb[prev]).unsqueeze(1)
            tok = tok + self.slot_emb[j]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=tok.is_cuda and self.bf16):
                for li, layer in enumerate(self.layers):
                    tok, cache[li] = _layer_step(layer, tok, cache[li])
            params = self.head(self.ln(tok.to(dtype)))[:, 0]                      # [b, n_params]
            prev = int(self.order[j])
            solve_row(prev, params)


def _layer_step(layer, h, kv):
    """One token through a _CausalLayer, attending to the cached keys and values `kv` of the earlier slots; returns
    the token's output and the cache extended by its own keys and values."""
    b, n, w = h.shape
    q, k, v = layer.qkv(layer.ln1(h)).view(b, n, 3, layer.heads, w // layer.heads).permute(2, 0, 3, 1, 4)
    if kv is not None:
        k, v = torch.cat((kv[0], k), dim=2), torch.cat((kv[1], v), dim=2)
    a = F.scaled_dot_product_attention(q, k, v)      # one query, every cached key and its own: the causal row
    h = h + layer.proj(a.transpose(1, 2).reshape(b, n, w))
    return h + layer.mlp(layer.ln2(h)), (k, v)
