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

Typed rows (docs/design/conditional_ar_architecture.md §5.3): with `row_types` (one index per dim into a shared
table of `n_types` coordinate types, models/fd_row_describer.py::FD_TYPES) the slot and value embeddings are looked
up by the coordinate's type instead of its position, so a type means the same thing in every chart; with
`extra_dim`, forward and decode also take per-row features extra [b, d, extra_dim] whose row i is a function of the
rows before row i in the order only (the caller's contract), projected and added to the slot that predicts row i.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from energies.prior_flow import _SOFTPLUS_ONE

#: the normalisation of every pre-norm slot and of the final one: LayerNorm, RMSNorm, or none (identity)
NORMS = ('layer', 'rms', 'none')


def _norm(kind, width):
    if kind == 'layer':
        return nn.LayerNorm(width)
    if kind == 'rms':
        return nn.RMSNorm(width)
    if kind == 'none':
        return nn.Identity()
    raise ValueError(f"transformer norm {kind!r}: one of {NORMS}")


class _CausalLayer(nn.Module):
    """Pre-norm causal self-attention and MLP; with `cross`, a cross-attention block between them in which every slot
    attends to a memory of tokens (the molecule's atoms), all of which every slot may see. `norm`: one of NORMS."""

    def __init__(self, width, heads, mlp_ratio=4, cross=False, norm='layer'):
        super().__init__()
        if width % heads:
            raise ValueError(f"transformer width {width} is not a multiple of {heads} heads")
        self.heads = heads
        self.ln1 = _norm(norm, width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.cross = bool(cross)
        if self.cross:
            self.ln_x = _norm(norm, width)
            self.q_x = nn.Linear(width, width)
            self.kv_x = nn.Linear(width, 2 * width)
            self.proj_x = nn.Linear(width, width)
            nn.init.zeros_(self.proj_x.weight)  # a new cross block starts as the identity
            nn.init.zeros_(self.proj_x.bias)
        self.ln2 = _norm(norm, width)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_ratio * width), nn.SiLU(),
                                 nn.Linear(mlp_ratio * width, width))

    def forward(self, h, mem=None, mem_mask=None):
        b, n, w = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(b, n, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.proj(a.transpose(1, 2).reshape(b, n, w))
        if self.cross and mem is not None:
            h = h + self.proj_x(_cross_attend(self.q_x(self.ln_x(h)), self.kv_x(mem), mem_mask, self.heads))
        return h + self.mlp(self.ln2(h))


def _cross_attend(q, kv, mem_mask, heads):
    """Multi-head attention of queries q [b, n, w] over memory keys and values kv [b, m, 2w]; mem_mask [b, m] is True
    on real tokens (padding is never attended)."""
    b, n, w = q.shape
    m = kv.shape[1]
    q = q.view(b, n, heads, w // heads).transpose(1, 2)
    k, v = kv.view(b, m, 2, heads, w // heads).permute(2, 0, 3, 1, 4)
    bias = None if mem_mask is None else mem_mask[:, None, None, :]
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    return a.transpose(1, 2).reshape(b, n, w)


class CausalTransformerConditioner(nn.Module):
    """Features of x in, spline parameters of every dim out (dim order), autoregressive in `order`."""

    def __init__(self, order, feat_dim, n_bins, width=256, layers=4, heads=8, norm='layer', context_dim=None,
                 cross=False, row_types=None, n_types=None, extra_dim=0):
        super().__init__()
        # conditional runs: the condition's embedding, projected to the width, is added to every token; with `cross`,
        # every layer also cross-attends to a memory of tokens passed with it (forward's ctx = (vector, memory, mask))
        self.ctx = nn.Linear(context_dim, width) if context_dim else None
        self.cross = bool(cross)
        d = len(order)
        self.d, self.n_params = d, 3 * n_bins + 1
        self.register_buffer('order', torch.as_tensor(order, dtype=torch.long))
        self.register_buffer('slot_of_dim', torch.argsort(torch.as_tensor(order, dtype=torch.long)))
        self.register_buffer('feat_dim', torch.as_tensor(feat_dim, dtype=torch.long))
        self.feat_w = nn.Parameter(torch.randn(len(feat_dim), width) / math.sqrt(2.0))
        if row_types is None:
            self.dim_emb = nn.Parameter(torch.zeros(d, width))           # which dim a value belongs to
            self.slot_emb = nn.Parameter(0.02 * torch.randn(d, width))   # which dim a slot predicts
            self.row_types = None
        else:   # the same two tables, keyed by coordinate type (shared across charts) rather than by dim
            if len(row_types) != d or n_types is None or not all(0 <= int(t) < int(n_types) for t in row_types):
                raise ValueError(f"row_types {list(row_types)}: one type in [0, {n_types}) per dim ({d})")
            self.register_buffer('row_types', torch.as_tensor(row_types, dtype=torch.long))
            self.type_val_emb = nn.Parameter(torch.zeros(int(n_types), width))
            self.type_slot_emb = nn.Parameter(0.02 * torch.randn(int(n_types), width))
        self.extra_dim = int(extra_dim)
        self.extra = nn.Linear(self.extra_dim, width) if self.extra_dim else None
        if self.extra is not None:   # zero: the conditioner starts as the untyped one. The no-peeking tests randomise
            nn.init.zeros_(self.extra.weight)   # every weight, so the zero hides no leak there
            nn.init.zeros_(self.extra.bias)
        self.start = nn.Parameter(torch.zeros(width))
        self.layers = nn.ModuleList(_CausalLayer(width, heads, cross=cross, norm=norm) for _ in range(layers))
        self.ln = _norm(norm, width)
        self.bf16 = True
        self.head = nn.Linear(width, self.n_params)
        # identity spline at step 0, as _MADE: zero weights, unit knot slopes, zero shift
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[2 * n_bins:3 * n_bins] = _SOFTPLUS_ONE

    def _value_emb(self):
        """[d, width] by dim: the embedding of the dim a value belongs to."""
        return self.dim_emb if self.row_types is None else self.type_val_emb[self.row_types]

    def _slot_emb(self):
        """[d, width] by slot: the embedding of the dim slot j predicts."""
        return self.slot_emb if self.row_types is None else self.type_slot_emb[self.row_types[self.order]]

    def _check_extra(self, extra, b):
        if (extra is None) != (self.extra is None):
            raise ValueError(f"this conditioner takes per-row features of width {self.extra_dim}"
                             if self.extra is not None else "per-row features passed to a conditioner built without")
        if extra is not None and tuple(extra.shape) != (b, self.d, self.extra_dim):
            raise ValueError(f"per-row features of shape {tuple(extra.shape)}, expected {(b, self.d, self.extra_dim)}")

    def forward(self, feats, ctx=None, extra=None):
        b = feats.shape[0]
        width = self.feat_w.shape[1]
        self._check_extra(extra, b)
        val = feats.new_zeros(b, self.d, width).index_add_(1, self.feat_dim, feats.unsqueeze(-1) * self.feat_w)
        val = val + self._value_emb()
        h = torch.cat((self.start.expand(b, 1, width), val[:, self.order[:-1]]), dim=1) + self._slot_emb()
        if extra is not None:
            h = h + self.extra(extra.to(h.dtype))[:, self.order]
        if (ctx is None) != (self.ctx is None):
            raise ValueError("a conditional transformer conditioner needs ctx, an unconditional one takes none")
        mem = mem_mask = None
        if isinstance(ctx, tuple):
            ctx, mem, mem_mask = ctx
            if not self.cross:
                raise ValueError("a memory of tokens was passed to a conditioner built without cross-attention")
        elif self.cross and ctx is not None:
            raise ValueError("a cross-attending conditioner needs ctx = (vector, memory, mask)")
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
                h = (checkpoint(layer, h, mem, mem_mask, use_reentrant=False) if torch.is_grad_enabled()
                     else layer(h, mem, mem_mask))
        out = self.head(self.ln(h.to(feats.dtype)))       # [b, d, n_params], slot order
        return out[:, self.slot_of_dim].reshape(b, self.d * self.n_params)

    @torch.no_grad()
    def decode(self, feats_of, solve_row, b, dtype, ctx=None, extra_of=None):
        """Autoregressive decoding with a key/value cache, one slot per step, for sampling.

        At step j the cache holds every layer's keys and values for slots 0..j-1. Slot j's token -- the start token at
        j = 0, else the value embedding of row order[j-1], fixed at the step before -- runs through each layer
        attending to the cache and to itself, the head gives row order[j]'s parameters, and `solve_row(row, params)`
        fixes that row. `feats_of()` returns the block's conditioner features [b, n_feat] of the rows fixed so far
        (`_Block._feats`); only the columns of the row just fixed are read. Each token's computation is the one
        forward() does for its slot on the completed rows, so the result matches the d-pass inverse (invert_block)
        to rounding, at about one forward pass of conditioner work instead of d. A conditional conditioner takes the
        same ctx as forward(): the context vector's projection is added to every token, and a cross-attending one's
        memory is projected to keys and values once per layer for the whole decode. `extra_of(row)` returns that row's
        per-row features [b, extra_dim] from the rows fixed so far, called for row order[j] at slot j; by the contract
        they depend on rows already fixed."""
        if (extra_of is None) != (self.extra is None):
            raise ValueError("decode's extra_of must be given exactly when the conditioner takes per-row features")
        if (ctx is None) != (self.ctx is None):
            raise ValueError("a conditional transformer conditioner needs ctx, an unconditional one takes none")
        vec, mem, mem_mask = ctx if isinstance(ctx, tuple) else (ctx, None, None)
        if mem is not None and not self.cross:
            raise ValueError("a memory of tokens was passed to a conditioner built without cross-attention")
        if self.cross and vec is not None and mem is None:
            raise ValueError("a cross-attending conditioner needs ctx = (vector, memory, mask)")
        width = self.feat_w.shape[1]
        add = self.ctx(vec).unsqueeze(1) if vec is not None else None
        on_cuda = self.start.is_cuda
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=on_cuda and self.bf16):
            mem_kv = [layer.kv_x(mem) if mem is not None else None for layer in self.layers]
        cache = [None] * len(self.layers)
        value_emb, slot_emb = self._value_emb(), self._slot_emb()
        prev = None
        for j in range(self.d):
            if j == 0:
                tok = self.start.to(dtype).expand(b, 1, width)
            else:
                cols = self.feat_dim == prev
                f = feats_of()[:, cols]                                           # [b, columns of row prev]
                tok = ((f.unsqueeze(-1) * self.feat_w[cols]).sum(1) + value_emb[prev]).unsqueeze(1)
            tok = tok + slot_emb[j]
            if extra_of is not None:
                tok = tok + self.extra(extra_of(int(self.order[j])).to(dtype)).unsqueeze(1)
            if add is not None:
                tok = tok + add
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=on_cuda and self.bf16):
                for li, layer in enumerate(self.layers):
                    tok, cache[li] = _layer_step(layer, tok, cache[li], mem_kv[li], mem_mask)
            params = self.head(self.ln(tok.to(dtype)))[:, 0]                      # [b, n_params]
            prev = int(self.order[j])
            solve_row(prev, params)


def _layer_step(layer, h, kv, mem_kv=None, mem_mask=None):
    """One token through a _CausalLayer, attending to the cached keys and values `kv` of the earlier slots and, in a
    cross-attending layer, to the memory's keys and values `mem_kv` [b, m, 2w]; returns the token's output and the
    cache extended by its own keys and values."""
    b, n, w = h.shape
    q, k, v = layer.qkv(layer.ln1(h)).view(b, n, 3, layer.heads, w // layer.heads).permute(2, 0, 3, 1, 4)
    if kv is not None:
        k, v = torch.cat((kv[0], k), dim=2), torch.cat((kv[1], v), dim=2)
    a = F.scaled_dot_product_attention(q, k, v)      # one query, every cached key and its own: the causal row
    h = h + layer.proj(a.transpose(1, 2).reshape(b, n, w))
    if layer.cross and mem_kv is not None:
        h = h + layer.proj_x(_cross_attend(layer.q_x(layer.ln_x(h)), mem_kv, mem_mask, layer.heads))
    return h + layer.mlp(layer.ln2(h)), (k, v)
