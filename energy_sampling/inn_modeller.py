"""
Exact-density normalizing flow on the crystal latent: a PROTOTYPE sampler that
replaces the SDE policy and P_B with one invertible map.

InnModeller inherits Modeller's construction -- config load, energy function,
prior and condition datasets, identifier registry, energy reference -- so log R
is scored in the same currency as a crystal training run on the same config. It
replaces the step loop, and uses none of the stage protocol, LR controller, Z
calibration or buffers.

The sampler is energies/prior_flow.py's autoregressive spline flow over the live
latent rows. The wrap mask and the dead rows are read off a GFN built from the
same config, so the geometry is the SDE policy's own. On a conditional config
(any *_conditioning flag) every block's conditioner also reads an embedding of
the condition vector, and a small head carries the per-condition level log Z(c).

Two stages:
  mle     -log q on stored prior rows (optionally jittered).
  energy  the TB residual  log q - log R + log Z  on fresh draws (the draw is
          detached; the level is the head's log Z(c) plus the batch's own Huber
          root) and on stored prior rows (against an EMA of that root), plus an
          optional likelihood term and an optional pathwise term: the gradient
          of KL(q || R/Z) through a differentiable copy of the draw and the
          energy's own force (w_path). The coefficients are loss weights on
          full batches. Three further options, each off by default: stored
          rows displaced and scored again each step (data_noise_log_range),
          temperature as a flow input trained over a range and read at the
          run's own (t_factor), and a likelihood of the flow's own draws
          weighted by their importance weights within each molecule (w_iw).

    python -u inn_modeller.py --config configs/mk_dev.yaml --run_name mip_a

Writes nothing under checkpoints_dir. Metrics go to wandb and to
artifacts/inn_proto/<tag>_<run_name>/metrics.jsonl; the weights sit beside it.
A metric carries the crystal trainer's name wherever that trainer logs the same
quantity, and a reading it has no counterpart for sits in the family it belongs
to (fwd/ bwd/ per branch, eval_fwd/ eval_bwd/ eval_test/, energy/, rollout/,
rewardgrad/, gradnorm/ per part of the flow). log P_B is zero for a one-step map, so every TB statistic reads
log R - log q exactly; "bwd"/"Backward" means the stored prior rows.
"""
import argparse
import contextlib
import copy
import gc
import json
import math
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml

from energies.ar_transformer import CausalTransformerConditioner
from energies.prior_flow import _Block, _Flow, _MADE, _rq_spline, _wrap, invert_block, rq_spline_inverse
from energy_sampling.eval.evaluations import (FUNNEL_MAX_POINTS, boltzmann_fig, flow_parity_plot, vargrad_error,
                                              xy_scatter_plot)
from energy_sampling.eval.xcond import xcond_metrics
from gflownet_losses import update_and_lookup_condition_log_z, update_condition_best_energy, winsorized_z_root
from models.gfn import GFN
from paths import artifact
from train import (Modeller, adjust_fig_filesize, condition_tracker_figs, equipartition_bar, fig_guard,
                   flatten_wandb_params, loggable_array, quick_tb_stats)
from utils import get_train_args, is_cuda_oom


@torch.no_grad()
def invert_block_cached(blk, y, ctx=None):
    """invert_block for a block whose conditioner is a CausalTransformerConditioner: the conditioner decodes one row
    per step through its key/value cache (CausalTransformerConditioner.decode) instead of running d full passes, and
    each step inverts that row's spline at y. Same arithmetic per row as invert_block, so the same x to rounding; with
    the layers in bf16 the decode's one-token path and forward's full pass round differently, and a trained map's
    conditioning amplifies that (--tf_fp32 runs them in the input's precision).
    ctx: a _ContextBlock's context, as its forward takes it (_ContextFlow.context_input's first output)."""
    b = y.shape[0]
    x = torch.zeros_like(y)
    bnd = blk.is_bounded if getattr(blk, 'has_bounded', False) else None
    mh = getattr(blk, 'min_height', None)
    td = getattr(blk, 'tied', False)
    nb = blk.n_bins

    def solve_row(i, p):
        shift, sp = p[:, -1:], p[:, :3 * nb].view(b, 1, 3, nb)
        sl = slice(i, i + 1)
        mhi = mh[sl] if mh is not None else None
        if bnd is None:
            z = rq_spline_inverse(y[:, sl], sp, blk.bound[sl], blk.is_wrap[sl], min_height=mhi, tied=td)
        else:
            z = rq_spline_inverse(y[:, sl], sp, blk.bound[sl], blk.is_wrap[sl], bnd[sl], shift, mhi, td)
            shift = torch.where(bnd[sl], torch.zeros_like(shift), shift)
        x[:, sl] = torch.where(blk.is_wrap[sl], _wrap(z - shift), z - shift)

    blk.made.decode(lambda: blk._feats(x), solve_row, b, y.dtype, ctx)
    return x


def invert_differentiable(blk, y, ctx=None):
    """A block's inverse at y with the autograd graph kept (prior_flow.invert_block and
    _ContextBlock.invert run without one). After blk.d passes the result is the exact
    preimage for any parameter value, so the graph through the passes carries its exact
    derivative."""
    x = torch.zeros_like(y)
    bnd = blk.is_bounded if getattr(blk, 'has_bounded', False) else None
    mh = getattr(blk, 'min_height', None)
    td = getattr(blk, 'tied', False)
    for _ in range(blk.d):
        if ctx is None:
            p = blk.made(blk._feats(x)).view(y.shape[0], blk.d, blk.n_params)
            shift, sp = p[..., -1], p[..., :3 * blk.n_bins].view(y.shape[0], blk.d, 3, blk.n_bins)
        else:
            shift, sp = blk.spline_params(x, ctx)
        if bnd is None:
            z = rq_spline_inverse(y, sp, blk.bound, blk.is_wrap, min_height=mh, tied=td)
        else:
            z = rq_spline_inverse(y, sp, blk.bound, blk.is_wrap, bnd, shift, mh, td)
            shift = torch.where(bnd, torch.zeros_like(shift), shift)
        x = torch.where(blk.is_wrap, _wrap(z - shift), z - shift)
    return x


# ---------------------------------------------------------------- mixture-CDF layer on the wrapped rows
_MIX_EPS = 1e-6               # weight of a uniform component: keeps the map invertible and its log-det bounded
_MIX_S = (0.002, 0.5)         # component scale range on the [-1, 1) circle


def _mix_parts(mp, k):
    """Mixture parameters from raw head outputs [.., 3k]: log weights, centres on [-1, 1), scales."""
    logw = torch.log_softmax(mp[..., :k], dim=-1)
    off = -1.0 + (2.0 * torch.arange(k, device=mp.device, dtype=mp.dtype) + 1.0) / k   # evenly spaced at init
    c = _wrap(mp[..., k:2 * k] + off)
    s = _MIX_S[0] + (_MIX_S[1] - _MIX_S[0]) * torch.sigmoid(mp[..., 2 * k:3 * k])
    return logw, c, s


def _mix_cdf(z, mp, k):
    """u = 2 G(z) - 1 and log du/dz, for G the CDF on [-1, 1) of a mixture of k wrapped logistics (five periodic
    images each) plus a uniform component of weight _MIX_EPS. z, u: [.., d]; mp: [.., d, 3k]. The density of the
    mixture on [-1, 1) is dG/dz, so du/dz = 2 dG/dz."""
    logw, c, s = _mix_parts(mp, k)
    zz = z.unsqueeze(-1)
    cdf = torch.zeros_like(c)
    tot = torch.zeros_like(c)
    lpdf = []
    for n in (-4.0, -2.0, 0.0, 2.0, 4.0):   # five images: the density is periodic to < 1e-6 nats at scale 0.5
        a, lo, hi = (zz - c + n) / s, (-1.0 - c + n) / s, (1.0 - c + n) / s
        cdf = cdf + torch.sigmoid(a) - torch.sigmoid(lo)
        tot = tot + torch.sigmoid(hi) - torch.sigmoid(lo)
        lpdf.append(F.logsigmoid(a) + F.logsigmoid(-a) - torch.log(s))
    cdf = cdf / tot
    lpdf = torch.logsumexp(torch.stack(lpdf, -1), -1) - torch.log(tot)          # [.., d, k] component log densities
    g = (1.0 - _MIX_EPS) * (torch.exp(logw) * cdf).sum(-1) + _MIX_EPS * 0.5 * (z + 1.0)
    log_dens = torch.logaddexp(math.log(1.0 - _MIX_EPS) + torch.logsumexp(logw + lpdf, -1),
                               torch.full_like(z, math.log(_MIX_EPS * 0.5)))
    return (2.0 * g - 1.0).clamp(-1.0, 1.0), log_dens + math.log(2.0)


def _mix_inverse(u, mp, k, iters=40):
    """z with _mix_cdf(z) = u, by bisection on [-1, 1] (G is monotone)."""
    lo, hi = torch.full_like(u, -1.0), torch.full_like(u, 1.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = _mix_cdf(mid, mp, k)[0] < u
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


class _MixBlock(_Block):
    """_Block with a mixture-CDF layer on its wrapped rows before the spline (data side): x -> rotate -> u = G(z) ->
    spline -> y. The conditioner's head carries 3k extra outputs per row (weights, centres, scales). The density in a
    gap between components falls like the logistic tail, with no learned bin height needed to empty it."""

    def __init__(self, order, wrap_mask, n_bins, hidden, bound, bounded_mask=None, mix_k=8):
        super().__init__(order, wrap_mask, n_bins, hidden, bound, bounded_mask)
        if self.has_bounded:
            raise NotImplementedError("--mix_k with --bounded_rows")
        self.mix_k = int(mix_k)
        self.n_out = self.n_params + 3 * self.mix_k

    def widen_head(self):
        """Give the transformer conditioner's head the 3k mixture outputs per row (zero weights, zero bias)."""
        m = self.made
        old = m.head
        new = nn.Linear(old.in_features, self.n_out).to(old.weight.device, old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.bias.zero_()
            new.bias[:self.n_params] = old.bias
        m.head, m.n_params = new, self.n_out

    def _split(self, p):
        b = p.shape[0]
        p = p.view(b, self.d, self.n_out)
        shift = p[..., self.n_params - 1]
        sp = p[..., :3 * self.n_bins].view(b, self.d, 3, self.n_bins)
        return shift, sp, p[..., self.n_params:]

    def forward_per_dim(self, x):
        shift, sp, mp = self._split(self.made(self._feats(x)))
        z = torch.where(self.is_wrap, _wrap(x + shift), x + shift)
        u, ldm = _mix_cdf(z, mp, self.mix_k)
        u = torch.where(self.is_wrap, u, z)
        ldm = torch.where(self.is_wrap, ldm, torch.zeros_like(ldm))
        y, ld = _rq_spline(u, sp, self.bound, self.is_wrap, min_height=self.min_height)
        return y, ld + ldm

    def forward(self, x):
        y, ld = self.forward_per_dim(x)
        return y, ld.sum(-1)

    def invert(self, y, d):
        x = torch.zeros_like(y)
        for _ in range(d):
            shift, sp, mp = self._split(self.made(self._feats(x)))
            u = rq_spline_inverse(y, sp, self.bound, self.is_wrap, min_height=self.min_height)
            z = torch.where(self.is_wrap, _mix_inverse(u, mp, self.mix_k), u)
            x = torch.where(self.is_wrap, _wrap(z - shift), z - shift)
        return x


class LatentFlow(nn.Module):
    """q(x) over the full latent: a spline flow on the live rows, dead rows pinned at 0."""

    def __init__(self, dim, wrap_mask, dead_rows, n_blocks, n_bins, hidden, bound, seed, transformer=None,
                 block_orders=None, half_rows=(), bounded_rows=(), mix_k=0):
        super().__init__()
        self.mix_k = int(mix_k)
        dead = {int(d) for d in dead_rows}
        live = [i for i in range(dim) if i not in dead]
        # bounded rows: linear latent rows modelled on the closed interval [-1, 1] (a spline with no tails and a
        # uniform base), so q has no mass outside the latent box on them
        self.bounded_rows = [int(r) for r in bounded_rows]
        if any(r not in live or wrap_mask[r] for r in self.bounded_rows):
            raise ValueError(f"--bounded_rows {self.bounded_rows}: each must be a live linear (unwrapped) row")
        live_bnd = [i in self.bounded_rows for i in live]
        # rows whose distribution repeats every half period (x and x + 1 are images of one crystal): the
        # flow models the folded row stretched back onto the full circle, u = 2 (((x + 1) mod 1) - 1) + 1,
        # and a draw picks one of the two images uniformly. Two images at half the measure each leave the
        # density unchanged, q(x) = q_flow(u), so log_prob needs no correction term.
        self.half_rows = [int(r) for r in half_rows]
        if any(r not in live or not wrap_mask[r] for r in self.half_rows):
            raise ValueError(f"--fold_half_rows {self.half_rows}: each must be a live wrapped row")
        self.dim = dim
        self.register_buffer('live_idx', torch.tensor(live, dtype=torch.long))
        live_wrap = [bool(wrap_mask[i]) for i in live]
        self.flow = _Flow(live_wrap, n_blocks=n_blocks, n_bins=n_bins, hidden=tuple(hidden), bound=bound, seed=seed,
                          bounded_mask=live_bnd)
        if block_orders is not None:  # one order per block, replacing _Flow's alternation
            d = len(live)
            if len(block_orders) != n_blocks:
                raise ValueError(f"--block_orders needs {n_blocks} entries, got {block_orders}")
            orders = []
            for o in block_orders:
                if o in ('id', 'rev'):
                    orders.append(list(range(d)) if o == 'id' else list(range(d))[::-1])
                    continue
                # an explicit order: '.'-separated latent row indices, every live row once
                rows = [int(r) for r in o.split('.')]
                if sorted(rows) != sorted(live):
                    raise ValueError(f"--block_orders entry {o!r} must list each live row {live} exactly once")
                orders.append([live.index(r) for r in rows])
            self.flow.orders = orders
            self.flow.blocks = nn.ModuleList(
                (_MixBlock(o, live_wrap, n_bins, tuple(hidden), bound, live_bnd, self.mix_k) if self.mix_k
                 else _Block(o, live_wrap, n_bins, tuple(hidden), bound, live_bnd)) for o in self.flow.orders)
        elif self.mix_k:
            raise NotImplementedError("--mix_k needs --block_orders")
        if transformer is not None:  # (width, layers, heads): each block's MADE replaced by a causal transformer
            for blk, order in zip(self.flow.blocks, self.flow.orders):
                blk.made = CausalTransformerConditioner(order, blk.feat_dim.tolist(), n_bins, *transformer)
                if isinstance(blk, _MixBlock):
                    blk.widen_head()
        elif self.mix_k:
            raise NotImplementedError("--mix_k needs --conditioner transformer")

    def set_bf16(self, on: bool):
        """The transformer conditioners' layers in bf16 (the default) or in the input's precision."""
        for blk in self.flow.blocks:
            if isinstance(blk.made, CausalTransformerConditioner):
                blk.made.bf16 = bool(on)

    def _fold(self, x):
        if not self.half_rows:
            return x
        x = x.clone()
        x[:, self.half_rows] = 2.0 * (torch.remainder(x[:, self.half_rows] + 1.0, 1.0) - 1.0) + 1.0
        return x

    def _unfold(self, x):
        if not self.half_rows:
            return x
        x = x.clone()
        image = torch.randint(0, 2, (x.shape[0], len(self.half_rows)), device=x.device).to(x.dtype)
        x[:, self.half_rows] = (x[:, self.half_rows] - 1.0) / 2.0 + image
        return x

    def to_base(self, x, c=None):
        """The base-space point of x, and log|det| of the map that takes it there."""
        f = self.flow
        x = self._fold(x)[:, self.live_idx]
        z = torch.where(f.is_wrap, _wrap(x), (x - f.mu) / f.sd)
        ld = x.new_zeros(x.shape[0]) - torch.log(f.sd[~f.is_wrap]).sum()
        for blk in f.blocks:
            z, l = blk(z)
            ld = ld + l
        return z, ld

    def log_prob(self, x, c=None):
        return self.flow.log_prob(self._fold(x)[:, self.live_idx])

    @torch.no_grad()
    def sample(self, n, c=None, return_base: bool = False):
        f = self.flow
        dev, dtype = f.mu.device, f.mu.dtype
        z0 = torch.where(f.unit_base, 2.0 * torch.rand(n, f.d, device=dev, dtype=dtype) - 1.0,
                         torch.randn(n, f.d, device=dev, dtype=dtype))
        z = z0
        for blk in reversed(list(f.blocks)):
            if isinstance(blk, _MixBlock):
                z = blk.invert(z, f.d)
            elif isinstance(blk.made, CausalTransformerConditioner):
                z = invert_block_cached(blk, z)       # one decode step per row, not d full passes
            else:
                z = invert_block(blk, z, f.d)
        x = torch.zeros(n, self.dim, device=dev, dtype=dtype)
        x[:, self.live_idx] = torch.where(f.is_wrap, _wrap(z), z * f.sd + f.mu)
        x = self._unfold(x)
        return (x, z0) if return_base else x

    def rsample(self, z0, c=None):
        """The draw sample() makes from base points z0, differentiable in the parameters."""
        if self.half_rows:
            raise NotImplementedError("rsample with --fold_half_rows: the image choice is not part of z0")
        if self.mix_k:
            raise NotImplementedError("rsample with --mix_k: the mixture inverse is a bisection")
        f = self.flow
        z = z0
        for blk in reversed(list(f.blocks)):
            z = invert_differentiable(blk, z)
        x = torch.zeros(z0.shape[0], self.dim, device=z0.device, dtype=z0.dtype)
        x[:, self.live_idx] = torch.where(f.is_wrap, _wrap(z), z * f.sd + f.mu)
        return x


class _ContextMADE(_MADE):
    """_MADE whose first hidden layer and whose output also read a context vector, unmasked."""

    def __init__(self, order, wrap_mask, hidden, n_bins, context_dim):
        super().__init__(order, wrap_mask, hidden, n_bins)
        self.ctx_hidden = nn.Linear(context_dim, hidden[0], bias=False)
        self.ctx_out = nn.Linear(context_dim, self.net[-1].out_features, bias=False)
        nn.init.zeros_(self.ctx_out.weight)  # with the zero head, each block is still the identity at step 0

    def forward(self, feats, ctx):
        h = self.net[0](feats) + self.ctx_hidden(ctx)
        for layer in self.net[1:]:
            h = layer(h)
        return h + self.ctx_out(ctx)


class _ContextBlock(_Block):
    """_Block whose conditioner is a _ContextMADE."""

    def __init__(self, order, wrap_mask, n_bins, hidden, bound, context_dim):
        super().__init__(order, wrap_mask, n_bins, hidden, bound)
        self.made = _ContextMADE(order, wrap_mask, hidden, n_bins, context_dim)

    def spline_params(self, x, ctx):
        p = self.made(self._feats(x), ctx).view(x.shape[0], self.d, self.n_params)
        return p[..., -1], p[..., :3 * self.n_bins].view(x.shape[0], self.d, 3, self.n_bins)

    def forward(self, x, ctx):
        shift, sp = self.spline_params(x, ctx)
        z = x + shift
        z = torch.where(self.is_wrap, _wrap(z), z)
        y, ld = _rq_spline(z, sp, self.bound, self.is_wrap, min_height=self.min_height, tied=self.tied)
        return y, ld.sum(-1)

    @torch.no_grad()
    def invert(self, y, ctx):
        """x with forward(x, ctx) == y: one more coordinate exact per pass, whatever the order."""
        x = torch.zeros_like(y)
        for _ in range(self.d):
            shift, sp = self.spline_params(x, ctx)
            z = rq_spline_inverse(y, sp, self.bound, self.is_wrap, min_height=self.min_height, tied=self.tied)
            x = torch.where(self.is_wrap, _wrap(z - shift), z - shift)
        return x


class AtomEncoder(nn.Module):
    """A molecule as tokens, one per atom, for the conditional flow to cross-attend to.

    Node inputs are the atomic number and the atom's coordinates in the frame the latent's orientation rows rotate
    from (the standard frame), PointNet style; mxtaltools' ScalarGNN runs radial-basis message passing over a radius
    graph within each molecule. Each atom's token is its node embedding with an embedding of its coordinates
    re-appended, projected to the transformer width; an attention pool gives one vector per molecule. Not
    rotation-invariant by design: the condition and its orientation target share the frame. Norms see the batch index,
    so a molecule's tokens do not depend on the molecules encoded with it."""

    def __init__(self, width, node_dim=128, convs=3, cutoff=10.0, radial=32, pooled_dim=128, max_neighbors=64):
        super().__init__()
        from mxtaltools.models.graph_models.graph_neural_network import ScalarGNN
        self.cutoff, self.max_neighbors = float(cutoff), int(max_neighbors)
        self.gnn = ScalarGNN(input_node_dim=4, node_dim=node_dim, fcs_per_gc=1, message_dim=node_dim // 2,
                             embedding_dim=node_dim, num_convs=convs, num_radial=radial, cutoff=cutoff,
                             max_num_neighbors=max_neighbors, atom_type_embedding_dim=16, norm='layer')
        self.pos_emb = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64))
        self.tok = nn.Linear(node_dim + 64, width)
        self.pool_q = nn.Parameter(torch.zeros(node_dim))
        self.pool_out = nn.Linear(node_dim, pooled_dim)

    def forward(self, z, pos, n):
        """z [u, a] atomic numbers (0 = padding), pos [u, a, 3], n [u] atoms per molecule -> tokens [u, a, width],
        mask [u, a] (True on atoms), pooled [u, pooled_dim]."""
        from mxtaltools.models.functions.radial_graph import build_radial_graph
        u, a = z.shape
        mask = torch.arange(a, device=z.device)[None, :] < n[:, None]
        batch = torch.arange(u, device=z.device)[:, None].expand(u, a)[mask]
        zf, pf = z[mask], pos[mask]
        ptr = F.pad(torch.cumsum(n, 0), (1, 0))
        edges = build_radial_graph(pf, batch, ptr, self.cutoff, self.max_neighbors)['edge_index']
        h = self.gnn(torch.cat([zf.to(pf.dtype)[:, None], pf], -1), pf, batch, edges)       # [nodes, node_dim]
        node = h.new_zeros(u, a, h.shape[-1])
        node[mask] = h
        tokens = self.tok(torch.cat([node, self.pos_emb(pos)], -1))
        score = (node * self.pool_q).sum(-1).masked_fill(~mask, float('-inf'))
        pooled = self.pool_out((torch.softmax(score, -1)[..., None] * node).sum(1))
        return tokens, mask, pooled


class _ResMLP(nn.Module):
    """Linear in, residual blocks (LayerNorm, Linear, SiLU, Linear), then LayerNorm and linear out."""

    def __init__(self, n_in, width, n_out, blocks=2):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(blocks))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, n_out))

    def forward(self, x):
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


class _ContextFlow(nn.Module):
    """The blocks of a conditional flow, the condition's embedding, and the level head."""

    def __init__(self, wrap_mask, cond_dim, context_dim, context_hidden, n_blocks, n_bins, hidden, bound, seed,
                 transformer=None, atoms=None):
        super().__init__()
        d = len(wrap_mask)
        # atoms: AtomEncoder settings (dict). The condition is then the molecule itself: column 0 of c is the condition
        # id, looked up in an atom table (set_atom_table), and any further columns (temperature) pass through
        self.atoms = atoms is not None
        self.d, self.bound = d, bound
        self.register_buffer('is_wrap', torch.tensor(wrap_mask, dtype=torch.bool))
        self.register_buffer('mu', torch.zeros(d))
        self.register_buffer('sd', torch.ones(d))
        self.register_buffer('c_mu', torch.zeros(cond_dim))
        self.register_buffer('c_sd', torch.ones(cond_dim))

        g = torch.Generator().manual_seed(seed)
        orders = []
        for b in range(n_blocks):  # the same alternation as _Flow
            if b % 3 == 0:
                orders.append(list(range(d)))
            elif b % 3 == 1:
                orders.append(list(range(d))[::-1])
            else:
                orders.append(torch.randperm(d, generator=g).tolist())
        self.orders = orders
        self.blocks = nn.ModuleList(
            _ContextBlock(o, wrap_mask, n_bins, hidden, bound, context_dim) for o in orders)
        if transformer is not None:  # (width, layers, heads): each block's _ContextMADE replaced by a causal transformer
            for blk, order in zip(self.blocks, orders):
                blk.made = CausalTransformerConditioner(order, blk.feat_dim.tolist(), n_bins, *transformer,
                                                        context_dim=context_dim, cross=self.atoms)
        elif self.atoms:
            raise NotImplementedError("atom tokens need --conditioner transformer (the MADE has no cross-attention)")
        n_in = cond_dim
        if self.atoms:
            self.atom_encoder = AtomEncoder(transformer[0], **atoms)
            n_in = self.atom_encoder.pool_out.out_features + cond_dim - 1   # pooled molecule + the columns after the id
            for name in ('atom_z', 'atom_pos', 'atom_n'):
                self.register_buffer(name, None, persistent=False)
        self.context = _ResMLP(n_in, context_hidden, context_dim)
        self.level = _ResMLP(n_in, 256, 1)
        nn.init.zeros_(self.level.out[-1].weight)
        nn.init.zeros_(self.level.out[-1].bias)
        self._shared = None   # context_input's results by condition tensor while ConditionalLatentFlow.shared_context is open

    def set_atom_table(self, z, pos, n):
        """Per condition id: atomic numbers [n_cond, a] (0 = padding), standard-frame coordinates [n_cond, a, 3] and
        atom counts [n_cond]; data, not weights (rebuilt from the conditions files on every start)."""
        self.atom_z, self.atom_pos, self.atom_n = z, pos, n

    def context_input(self, c):
        """What every block and the level head read: (embedded vector, atom tokens, atom mask) and the level head's
        input in atom mode, else the embedded standardised condition and the standardised condition. While
        ConditionalLatentFlow.shared_context is open, a repeat call on the same tensor with gradients enabled returns
        the first such call's result (no-grad calls, the draws', are not kept)."""
        memo = self._shared
        if memo is None or not torch.is_grad_enabled():
            return self._context_input(c)
        hit = memo.get(id(c))
        if hit is None or hit[0] is not c:
            hit = memo[id(c)] = (c, self._context_input(c))   # c is held, so its id is not reused inside the block
        return hit[1]

    def _context_input(self, c):
        if not self.atoms:
            s = self.standardise(c)
            return self.context(s), s
        cid = c[:, 0].round().long()
        if self.atom_z is None or bool((self.atom_n[cid] == 0).any()):
            raise RuntimeError("a condition id with no atoms in the atom table")
        uniq, inv = torch.unique(cid, return_inverse=True)
        dtype = self.c_mu.dtype   # the flow's own precision (a double copy for the round-trip check), as standardise
        tokens, mask, pooled = self.atom_encoder(self.atom_z[uniq], self.atom_pos[uniq].to(dtype), self.atom_n[uniq])
        s = torch.cat([pooled[inv], c[:, 1:].to(dtype)], -1)
        return (self.context(s), tokens[inv], mask[inv]), s

    def set_scaler(self, x):
        """Standardise the LINEAR dims only; wrapped dims must stay on the circle."""
        mu, sd = x.mean(0), x.std(0).clamp_min(1e-4)
        self.mu.copy_(torch.where(self.is_wrap, torch.zeros_like(mu), mu))
        self.sd.copy_(torch.where(self.is_wrap, torch.ones_like(sd), sd))

    def set_condition_scaler(self, c):
        self.c_mu.copy_(c.mean(0))
        self.c_sd.copy_(c.std(0).clamp_min(1e-4))

    def standardise(self, c):
        return (c.to(self.c_mu.dtype) - self.c_mu) / self.c_sd


class ConditionalLatentFlow(nn.Module):
    """q(x | c): LatentFlow with every block reading an embedding of the condition vector c."""

    def __init__(self, dim, wrap_mask, dead_rows, cond_dim, context_dim, context_hidden,
                 n_blocks, n_bins, hidden, bound, seed, transformer=None, atoms=None):
        super().__init__()
        dead = {int(d) for d in dead_rows}
        live = [i for i in range(dim) if i not in dead]
        self.dim = dim
        self.register_buffer('live_idx', torch.tensor(live, dtype=torch.long))
        self.flow = _ContextFlow([bool(wrap_mask[i]) for i in live], cond_dim, context_dim, context_hidden,
                                 n_blocks, n_bins, tuple(hidden), bound, seed, transformer=transformer, atoms=atoms)

    def to_base(self, x, c):
        f = self.flow
        ctx = f.context_input(c)[0]
        x = x[:, self.live_idx]
        z = torch.where(f.is_wrap, _wrap(x), (x - f.mu) / f.sd)
        ld = x.new_zeros(x.shape[0]) - torch.log(f.sd[~f.is_wrap]).sum()
        for blk in f.blocks:
            z, l = blk(z, ctx)
            ld = ld + l
        return z, ld

    def log_prob(self, x, c):
        f = self.flow
        z, ld = self.to_base(x, c)
        zl = z[:, ~f.is_wrap]
        base = (-math.log(2.0) * int(f.is_wrap.sum())
                - 0.5 * (zl * zl).sum(-1) - 0.5 * math.log(2 * math.pi) * zl.shape[1])
        return base + ld

    @torch.no_grad()
    def sample(self, n, c, return_base: bool = False):
        f = self.flow
        dev, dtype = f.mu.device, f.mu.dtype
        ctx = f.context_input(c)[0]
        z0 = torch.where(f.is_wrap, 2.0 * torch.rand(n, f.d, device=dev, dtype=dtype) - 1.0,
                         torch.randn(n, f.d, device=dev, dtype=dtype))
        z = z0
        for blk in reversed(list(f.blocks)):
            if isinstance(blk.made, CausalTransformerConditioner):
                z = invert_block_cached(blk, z, ctx)       # one decode step per row, not d full passes
            else:
                z = blk.invert(z, ctx)
        x = torch.zeros(n, self.dim, device=dev, dtype=dtype)
        x[:, self.live_idx] = torch.where(f.is_wrap, _wrap(z), z * f.sd + f.mu)
        return (x, z0) if return_base else x

    def set_bf16(self, on: bool):
        """The transformer conditioners' layers in bf16 (the default) or in the input's precision, as LatentFlow."""
        for blk in self.flow.blocks:
            if isinstance(blk.made, CausalTransformerConditioner):
                blk.made.bf16 = bool(on)

    @contextlib.contextmanager
    def shared_context(self):
        """Within the block each condition tensor's context (_ContextFlow.context_input: the atom encoder and the
        context network) is computed once with gradients and reused, so an energy step that reads the same conditions
        for log q and for the level head encodes them once. The parameters must not change inside the block."""
        self.flow._shared = {}
        try:
            yield
        finally:
            self.flow._shared = None

    def rsample(self, z0, c):
        """The draw sample() makes from base points z0, differentiable in the parameters."""
        f = self.flow
        ctx = f.context_input(c)[0]
        z = z0
        for blk in reversed(list(f.blocks)):
            z = invert_differentiable(blk, z, ctx)
        x = torch.zeros(z0.shape[0], self.dim, device=z0.device, dtype=z0.dtype)
        x[:, self.live_idx] = torch.where(f.is_wrap, _wrap(z), z * f.sd + f.mu)
        return x

    def level(self, c):
        """log Z(c) up to the batch's shared level: the head, zero at initialisation."""
        return self.flow.level(self.flow.context_input(c)[1]).squeeze(-1)


def tb_loss(resid, beta):
    """gflownet_losses.get_tb_loss's Huber form, per row."""
    return beta * F.smooth_l1_loss(resid, torch.zeros_like(resid), reduction='none', beta=beta)


def tb_loss_asym(resid, beta, beta_pos=None):
    """tb_loss with its own knee for positive residuals (draws the flow over-weights, log q above log R - level);
    beta_pos inf = quadratic there. Inside both knees it is 0.5 r^2, as tb_loss."""
    if beta_pos is None:
        return tb_loss(resid, beta)
    pos = 0.5 * resid * resid if math.isinf(beta_pos) else tb_loss(resid, beta_pos)
    return torch.where(resid > 0, pos, tb_loss(resid, beta))


def quantiles(x, qs=(0.1, 0.5, 0.9)):
    return [float(v) for v in torch.quantile(x.double().flatten(), torch.tensor(qs, dtype=torch.float64,
                                                                               device=x.device))]


def grad_parts(sampler):
    """The sampler's parameters by part, for gradnorm/*: each block (a transformer conditioner split into its input
    embeddings, each causal layer, and its output norm and head), then every other child of the flow under its own
    name (context, level, atom_encoder in conditional runs). A parameter none of these holds goes to 'other', so a
    part added later is still reported."""
    f, parts = sampler.flow, {}
    for i, blk in enumerate(f.blocks):
        m = blk.made
        if isinstance(m, CausalTransformerConditioner):
            for j, layer in enumerate(m.layers):
                parts[f'block{i}/layer{j}'] = list(layer.parameters())
            parts[f'block{i}/head'] = [*m.ln.parameters(), *m.head.parameters()]
            taken = {id(p) for ps in parts.values() for p in ps}
            parts[f'block{i}/embed'] = [p for p in blk.parameters() if id(p) not in taken]
        else:
            parts[f'block{i}'] = list(blk.parameters())
    for name, mod in f.named_children():
        if name != 'blocks':
            parts[name] = list(mod.parameters())
    taken = {id(p) for ps in parts.values() for p in ps}
    rest = [p for p in sampler.parameters() if id(p) not in taken]
    if rest:
        parts['other'] = rest
    return parts


def part_grad_norms(sampler):
    """gradnorm/<part>: the gradient L2 norm of each part of grad_parts, the crystal trainer's per-submodel family
    (train.py's _submodel_grad_norms). A part with no gradient reads 0.0 rather than going missing, so a part the loss
    does not reach is a line pinned at zero. The squares sum to the square of the norm clip_grad_norm_ returns; call it
    before that clip, which rescales the gradients in place. One device sync for the set."""
    dev = next(sampler.parameters()).device
    names, norms = [], []
    for name, ps in grad_parts(sampler).items():
        g = [torch.linalg.vector_norm(p.grad.detach().float()) for p in ps if p.grad is not None]
        names.append(name)
        norms.append(torch.linalg.vector_norm(torch.stack(g)) if g else torch.zeros((), device=dev))
    return {f'gradnorm/{n}': v for n, v in zip(names, torch.stack(norms).cpu().tolist())}


class InnModeller(Modeller):

    def __init__(self, args, inn):
        self.inn = inn
        super().__init__(args=args)
        self.out_dir = artifact('inn_proto') / self.run_name
        os.makedirs(self.out_dir, exist_ok=True)
        self.energy_samples_scored = 0  # training draws only; eval draws are not counted
        self.oom_waits = 0

    # -- construction ----------------------------------------------------------

    def init_gfn(self):
        probe = GFN(**self._build_gfn_config())  # geometry only; never trained or sampled
        wrap_mask = probe.ang_mask.cpu().tolist()
        dead_rows = tuple(probe.dead_rows)
        dim = int(probe.dim)
        del probe
        inn = self.inn
        mol_dim = int(self.get_conditioning_dim())
        self.mol_conditional = mol_dim > 0        # the config gives every row a condition vector
        self.t_conditional = inn.t_factor > 1.0   # this trainer's own temperature conditioning
        if self.t_conditional and bool(self.args.temperature_conditioning):
            raise ValueError("--t_factor with the config's temperature_conditioning: one or the other")
        self.conditional = self.mol_conditional or self.t_conditional  # the flow reads a context, and has a level head
        ck_inn = None
        if inn.init_from is not None:
            ck_inn = torch.load(inn.init_from, map_location='cpu', weights_only=False).get('inn') or {}
        account = resolve_auto_options(inn, self.conditional, ck_inn,
                                       live_rows=[i for i in range(dim) if i not in set(dead_rows)])
        if account:
            print(f"options left unset, resolved: {account}", flush=True)
        cond_dim = mol_dim + int(self.t_conditional)
        self.log_T0 = math.log10(float(self.args.energy_config.temperature))
        self.t_span = math.log10(inn.t_factor) if self.t_conditional else 1.0
        shape = (inn.n_blocks, inn.n_bins, (inn.hidden,) * inn.hidden_layers, inn.bound)
        transformer = (inn.tf_width, inn.tf_layers, inn.tf_heads) if inn.conditioner == 'transformer' else None
        block_orders = inn.block_orders.split(',') if inn.block_orders else None
        half_rows = [int(r) for r in inn.fold_half_rows.split(',')] if inn.fold_half_rows else []
        if inn.mix_k and inn.w_path > 0:
            raise NotImplementedError("--mix_k with --w_path: rsample cannot pass through the mixture's bisection")
        if half_rows and inn.w_path > 0:
            raise NotImplementedError("--fold_half_rows with --w_path: rsample does not carry the image choice")
        if inn.bounded_rows == 'linear':
            bounded_rows = [i for i in range(dim) if i not in set(int(r) for r in dead_rows) and not wrap_mask[i]]
        else:
            bounded_rows = [int(r) for r in inn.bounded_rows.split(',')] if inn.bounded_rows else []
        if self.conditional:
            if block_orders is not None or half_rows or bounded_rows or inn.mix_k:
                raise NotImplementedError("--block_orders, --fold_half_rows, --bounded_rows and --mix_k are built "
                                          "for unconditional runs only")
            atoms = None
            if inn.atom_tokens:
                # the condition is the molecule itself: c = [condition id] (+ the temperature column), the atoms looked
                # up in the table set from the conditions files (init_data_rows) and encoded by the flow
                if not self.mol_conditional:
                    raise NotImplementedError("--atom_tokens needs molecule conditions (embedding_conditioning)")
                if transformer is None:
                    raise NotImplementedError("--atom_tokens needs --conditioner transformer")
                atoms = dict(node_dim=inn.atom_node_dim, convs=inn.atom_convs, cutoff=inn.atom_cutoff,
                             radial=inn.atom_radial)
                cond_dim = 1 + int(self.t_conditional)
            self.sampler = ConditionalLatentFlow(dim, wrap_mask, dead_rows, cond_dim, inn.context_dim,
                                                 inn.context_hidden, *shape, seed=int(self.args.seed),
                                                 transformer=transformer, atoms=atoms)
        else:
            self.sampler = LatentFlow(dim, wrap_mask, dead_rows, *shape, seed=int(self.args.seed),
                                      transformer=transformer, block_orders=block_orders, half_rows=half_rows,
                                      bounded_rows=bounded_rows, mix_k=inn.mix_k)
        self.sampler = self.sampler.to(self.device)
        if inn.min_bin_height is not None:
            # a lower floor on the spline's bin heights on the chosen rows: with a uniform base (wrapped and bounded
            # rows) each bin holds at least its height floor of the row's mass whatever the weights, so the default
            # 1e-3 x n_bins is mass no training can remove
            h0 = float(inn.min_bin_height)  # an arm-override file may hand a YAML '1e-6' over as a string
            if not 0.0 < h0 < 1.0 / inn.n_bins:
                raise ValueError(f"--min_bin_height {h0:g} must lie in (0, 1/n_bins = {1.0 / inn.n_bins:g}): at or "
                                 f"above 1/n_bins the heights cannot move or go negative")
            if inn.min_bin_height_rows not in ('uniform', 'all'):
                raise ValueError(f"--min_bin_height_rows {inn.min_bin_height_rows!r}: 'uniform' or 'all'")
            if inn.w_path > 0:
                # a draw inside a floored bin has |dx/dparams| up to ~1/floor, and the path term does not clip it
                raise NotImplementedError("--min_bin_height with --w_path > 0: the reparameterised path gradient "
                                          "is unbounded inside floored bins")
            inn.min_bin_height = h0
            for blk in self.sampler.flow.blocks:
                rows = (blk.unit_base_rows() if inn.min_bin_height_rows == 'uniform'
                        else torch.ones_like(blk.is_wrap))
                blk.min_height = torch.where(rows, torch.full_like(blk.bound, h0), torch.full_like(blk.bound, 1e-3))
            print(f"bin-height floor {h0:g} on {inn.min_bin_height_rows} rows (else 1e-3)")
        if inn.tied_derivatives:
            # knot derivatives derived from the neighbouring bin slopes (prior_flow._tied_derivatives): no knot far
            # steeper than both bins it joins, so near-empty bins carry no density spikes
            if inn.mix_k:
                raise NotImplementedError("--tied_derivatives with --mix_k")
            tf = float(inn.tied_factor)
            if not tf >= 1.0:
                raise ValueError(f"--tied_factor {tf:g} must be at least 1 (the learned factor's range is [1/F, F])")
            for blk in self.sampler.flow.blocks:
                blk.tied = tf
            print(f"knot derivatives tied to the neighbouring bin slopes (harmonic mean x a factor in [1/{tf:g}, {tf:g}])")
        if inn.tf_fp32:
            if inn.conditioner != 'transformer':
                raise NotImplementedError("--tf_fp32 needs --conditioner transformer")
            self.sampler.set_bf16(False)
        n_par = sum(p.numel() for p in self.sampler.parameters())
        print(f"flow sampler: dim {dim}, wrapped rows {[i for i, w in enumerate(wrap_mask) if w]}, "
              f"dead rows {list(dead_rows)}, {inn.n_blocks} blocks x {inn.n_bins} bins, "
              + (f"block orders {block_orders}, " if block_orders is not None else "")
              + (f"half-period images folded on rows {half_rows}, " if half_rows else "")
              + (f"bounded (no-tail, uniform-base) rows {bounded_rows}, " if bounded_rows else "")
              + (f"causal transformer width {inn.tf_width} x {inn.tf_layers} layers, {inn.tf_heads} heads, "
                 if transformer is not None else f"hidden {inn.hidden} x {inn.hidden_layers}, ")
              + (f"condition width {cond_dim} embedded to {inn.context_dim}, " if self.conditional else "")
              + (f"temperature-conditioned over T x [1, {inn.t_factor:g}], " if self.t_conditional else "")
              + f"{n_par / 1e6:.2f}M parameters", flush=True)

    @torch.no_grad()
    def init_data_rows(self, chunk: int = 8192):
        """Latent, stored log R and condition id of every prior row, split into train and
        held-out rows; per condition, its condition vector. The lowest energy known for a
        training condition goes to the trainer's tracker (condition_log_z), as its own runs
        record it; held-out conditions keep theirs in test_floor, apart from training state."""
        ef = self.energy_function
        T = float(self.args.energy_config.temperature)
        n_cond = int(ef.condition_library_size)
        self.test_floor = torch.full((n_cond,), float('inf'), device=self.device)
        if self.mol_conditional:
            self.cond_table = torch.zeros(n_cond, int(self.get_conditioning_dim()), device=self.device)
            self.cond_known = torch.zeros(n_cond, dtype=torch.bool, device=self.device)

        batch = self.prior_dataset.batch
        xs, lrs, lrs_hot, cids = [], [], [], []
        for start in range(0, batch.num_graphs, chunk):
            sub = batch.subsample_new_batch(
                torch.arange(start, min(start + chunk, batch.num_graphs), device=batch.device)).to(self.device)
            x = self._batch_latents(sub)
            sub, log_T, cond, cid = ef.condition_samples(sub)
            lrs.append(ef.prebuilt_sample_to_reward(sub, 10 ** log_T).float().flatten().to(self.device))
            if self.t_conditional:  # the same rows at the top of the temperature range (see stored_log_r)
                lrs_hot.append(ef.prebuilt_sample_to_reward(
                    sub, 10 ** log_T * self.inn.t_factor).float().flatten().to(self.device))
            xs.append(x.float().to(self.device))
            cids.append(cid.long().to(self.device))
            self.note_conditions(cids[-1], cond)
        x, lr, cid = torch.cat(xs), torch.cat(lrs), torch.cat(cids)
        lr_hot = torch.cat(lrs_hot) if self.t_conditional else lr
        ok = torch.isfinite(lr) & torch.isfinite(lr_hot) & torch.isfinite(x).all(-1)
        if not bool(ok.all()):
            print(f"data rows: dropping {int((~ok).sum())} of {ok.numel()} rows with a non-finite latent or reward")
            x, lr, lr_hot, cid = x[ok], lr[ok], lr_hot[ok], cid[ok]
        self.condition_log_z.update_best_energy(cid, -lr * T)

        # the held-out split's seed: --split_seed, else the run's seed + 1 (so a second training seed can keep the
        # start checkpoint's split)
        g = torch.Generator(device='cpu').manual_seed(int(self.inn.split_seed) if self.inn.split_seed is not None
                                                      else int(self.args.seed) + 1)
        n_val = max(1, min(10000, int(self.inn.val_frac * x.shape[0])))
        if self.inn.val_group_ignore_rows:
            # held out by CRYSTAL: rows that agree (to 1e-3, same condition) on every row but the named ones --
            # the prior's exact image copies -- are one crystal, and whole crystals go to one side of the split
            ign = {int(r) for r in self.inn.val_group_ignore_rows.split(',')}
            keep = [i for i in range(x.shape[1]) if i not in ign]
            key = torch.cat([torch.round(x[:, keep].double() * 1e3), cid.double().unsqueeze(-1)], 1).cpu()
            _, grp = torch.unique(key, dim=0, return_inverse=True)
            n_grp = int(grp.max()) + 1
            order = torch.randperm(n_grp, generator=g)
            rank = torch.empty(n_grp, dtype=torch.long)
            rank[order] = torch.arange(n_grp)
            n_take = int(torch.searchsorted(torch.bincount(grp, minlength=n_grp)[order].cumsum(0),
                                            torch.tensor(n_val))) + 1
            is_val = (rank[grp] < n_take).to(self.device)
            va, tr = torch.nonzero(is_val).flatten(), torch.nonzero(~is_val).flatten()
            # shuffled: the prior file is ordered (anchors first, blocked by image), and evaluation reads
            # x_train[:n] and cid_val[:n] as random samples
            va = va[torch.randperm(va.numel(), generator=g).to(va.device)]
            tr = tr[torch.randperm(tr.numel(), generator=g).to(tr.device)]
            n_val = int(va.numel())
            print(f"data rows: held out by crystal: {n_take} of {n_grp} crystals ({n_val} rows), rows agreeing off "
                  f"latent rows {sorted(ign)} counted as one crystal", flush=True)
        else:
            perm = torch.randperm(x.shape[0], generator=g).to(self.device)
            va, tr = perm[:n_val], perm[n_val:]
        self.x_val, self.lr_val, self.cid_val = x[va], lr[va], cid[va]
        self.x_train, self.lr_train, self.cid_train = x[tr], lr[tr], cid[tr]
        if self.t_conditional:
            # a stored row's reward energy is a + b T, so two temperatures give both parts
            self.a_train = (lr_hot[tr] - lr[tr]) * T / (1.0 - 1.0 / self.inn.t_factor)
            self.b_train = -lr[tr] - self.a_train / T
        self.sampler.flow.set_scaler(self.x_train[:, self.sampler.live_idx])
        # the stored-row pool the energy stage draws from: the prior's own rows first; with --grow_stored every
        # scored row at the run's own temperature joins behind them (append_stored). Evaluation reads the prior's
        # rows only (the first n_prior_rows).
        self.n_prior_rows = int(self.x_train.shape[0])
        self.src_train = torch.zeros(self.n_prior_rows, dtype=torch.long, device=self.device)
        self.n_appended = 0
        if self.inn.grow_stored and self.inn.w_mle > 0:
            raise NotImplementedError("--grow_stored with --w_mle: the likelihood term would train on the flow's own draws")
        if self.inn.grow_stored and self.t_conditional:
            raise NotImplementedError("--grow_stored with temperature conditioning: appended rows carry no a + b T split")
        if self.inn.level_mode in ('is', 'blend') and self.conditional:
            raise NotImplementedError(f"--level_mode {self.inn.level_mode} on a conditional run: one shared level "
                                      "does not fit rows centred on a per-condition head")
        if self.inn.bank_frac > 0 and (self.conditional or not 0.0 < self.inn.bank_frac < 1.0):
            raise NotImplementedError("--bank_frac: unconditional runs only, and a share strictly between 0 and 1")
        if self.inn.mle_plateau_evals > 0 and (self.inn.energy_steps > 0 or self.inn.budget_minutes is not None):
            raise NotImplementedError("--mle_plateau_evals ends a likelihood-only process (--energy_steps 0, no budget)")
        if not 0.0 <= self.inn.level_blend <= 1.0:
            raise ValueError(f"--level_blend must lie in [0, 1], got {self.inn.level_blend}")
        if self.inn.stored_batch is not None and self.inn.stored_batch < 1:
            raise ValueError(f"--stored_batch must be positive, got {self.inn.stored_batch}")

        # where a condition's molecule sits in mol_dataset, for building a named condition's crystals
        mol_cid = ef.condition_ids_of(self.mol_dataset.batch).to(self.device)
        self.mol_row_of_cid = torch.full((n_cond,), -1, dtype=torch.long, device=self.device)
        self.mol_row_of_cid[mol_cid] = torch.arange(mol_cid.numel(), device=self.device)

        if self.inn.atom_tokens:
            # per condition id, its molecule's atoms in the standard frame, filled from the conditions files below
            sets = [self.mol_dataset] + ([self.test_mol_dataset]
                                         if getattr(self, 'test_mol_dataset', None) is not None else [])
            a_max = max(int(torch.diff(ds.batch.ptr).max()) for ds in sets)
            self.atom_table = (torch.zeros(n_cond, a_max, dtype=torch.long, device=self.device),
                               torch.zeros(n_cond, a_max, 3, device=self.device),
                               torch.zeros(n_cond, dtype=torch.long, device=self.device))
        if self.mol_conditional:
            # the conditions files' own crystals: one reference structure per molecule, held-out included
            self.fold_condition_crystals(self.mol_dataset, held_out=False)
            if getattr(self, 'test_mol_dataset', None) is not None:
                self.fold_condition_crystals(self.test_mol_dataset, held_out=True)
        if self.inn.atom_tokens:
            z_t, pos_t, n_t = self.atom_table
            need = torch.unique(torch.cat([self.cid_train, self.cid_val]))
            if bool((n_t[need] == 0).any()):
                raise RuntimeError(f"--atom_tokens: {int((n_t[need] == 0).sum())} stored-row conditions have no "
                                   f"molecule in the conditions files")
            self.sampler.flow.set_atom_table(z_t, pos_t, n_t)
            print(f"atom tokens: {int((n_t > 0).sum())} molecules in the atom table, up to {z_t.shape[1]} atoms, "
                  f"standard-frame coordinates", flush=True)
        if self.conditional:
            ids = (self.cond_known.nonzero().flatten() if self.mol_conditional
                   else torch.zeros(4096, dtype=torch.long, device=self.device))
            self.sampler.flow.set_condition_scaler(self.context_of(ids, self.training_log_T(ids.numel())))
        if self.mol_conditional and getattr(self, 'mol_crystals_test', None) is not None:
            self.save_condition_neighbours()

        e = -lr * T
        q10, q50, q90 = quantiles(e)
        lin = self.sampler.live_idx[~self.sampler.flow.is_wrap]
        edge = (self.x_train[:, lin].abs() >= 1 - 1e-6).float().mean(0)
        print(f"data rows: {self.x_train.shape[0]} train, {n_val} held out, over "
              f"{int(torch.unique(cid).numel())} conditions; stored energy (-log R x T, T {T}) min "
              f"{float(e.min()):.2f}, P10 {q10:.2f}, P50 {q50:.2f}, P90 {q90:.2f}, max {float(e.max()):.2f}",
              flush=True)
        print(f"data rows: per-row std {[round(float(v), 3) for v in self.x_train.std(0)]}")
        print(f"data rows: fraction at the box edge per linear row {[round(float(v), 4) for v in edge]}", flush=True)
        if self.mol_conditional:
            print(f"conditions: {int(self.cond_known.sum())} of {n_cond} have a condition vector; "
                  f"{int(torch.isfinite(self.condition_log_z.best_energy).sum())} training and "
                  f"{int(torch.isfinite(self.test_floor).sum())} held-out conditions an energy floor", flush=True)
        self.data_energy = dict(min=float(e.min()), p10=q10, p50=q50, p90=q90)

    @torch.no_grad()
    def save_condition_neighbours(self, chunk: int = 1024):
        """condition_neighbours.npz: for each molecule of molecule_nll (held-out, and the fixed training draw), its
        nearest OTHER training molecule and the distance to it in the stored condition vectors (the conditions
        file's embedding), the same distance for every arm whatever its conditioner reads."""
        train = torch.unique(self.cid_train)
        e_tr = self.cond_table[train].float()
        out = {}
        for side in ('train', 'test'):
            cid = torch.unique(getattr(self, f'mol_crystals_{side}')[1])
            near, dist = [], []
            for start in range(0, cid.numel(), chunk):
                q = cid[start:start + chunk]
                d = torch.cdist(self.cond_table[q].float(), e_tr)
                d[q[:, None] == train[None, :]] = float('inf')   # a training molecule's own row
                v, i = d.min(1)
                dist.append(v)
                near.append(train[i])
            out.update({f'{side}_cid': cid.cpu().numpy(), f'{side}_near': torch.cat(near).cpu().numpy(),
                        f'{side}_dist': torch.cat(dist).cpu().numpy()})
        np.savez(os.path.join(self.out_dir, 'condition_neighbours.npz'), **out)

    def note_atoms(self, cid, mol_batch):
        """Each molecule's atomic numbers and coordinates, as the batch holds them (the standard frame), into the
        atom table at its condition id."""
        z_t, pos_t, n_t = self.atom_table
        nodes = mol_batch.batch.to(self.device)
        ptr = mol_batch.ptr.to(self.device)
        idx = torch.arange(nodes.numel(), device=self.device) - ptr[nodes]
        rows = cid[nodes]
        z_t[rows, idx] = mol_batch.z.long().to(self.device).flatten()
        pos_t[rows, idx] = mol_batch.pos.float().to(self.device)
        n_t[cid] = torch.diff(ptr)

    def note_conditions(self, cid, cond):
        if self.mol_conditional:
            self.cond_table[cid] = cond.float().to(self.device)
            self.cond_known[cid] = True

    @torch.no_grad()
    def fold_condition_crystals(self, dataset, held_out: bool, chunk: int = 4096):
        """A conditions file's own crystals: their condition vectors into the table, and the
        reward energy of each, rebuilt from its latent, into its condition's floor (the
        tracker's for training conditions, test_floor for held-out ones)."""
        ef = self.energy_function
        T = float(self.args.energy_config.temperature)
        batch = dataset.batch
        xs, cids = [], []
        for start in range(0, batch.num_graphs, chunk):
            sub = batch.subsample_new_batch(
                torch.arange(start, min(start + chunk, batch.num_graphs), device=batch.device)).to(self.device)
            x = self._batch_latents(sub).float()
            sub.orient_molecule(mode='std')
            sub, log_T, cond, cid = ef.condition_samples(sub)
            cid = cid.long().to(self.device)
            log_r = ef.log_reward(x, sub, log_T, False, internal_oom_recovery=True).flatten().to(self.device)
            self.note_conditions(cid, cond)
            if self.inn.atom_tokens:
                self.note_atoms(cid, sub)
            ok = torch.isfinite(log_r)
            if held_out:
                self.test_floor.scatter_reduce_(0, cid[ok], -log_r[ok] * T, reduce='amin')
            else:
                self.condition_log_z.update_best_energy(cid[ok], -log_r[ok] * T)
            xs.append(x.to(self.device)[ok])
            cids.append(cid[ok])
        # kept for the per-molecule likelihood (molecule_nll): every held-out crystal, and a fixed
        # draw of 10,000 training ones (seed + 2) so the two sides rest on similar counts
        x, cid = torch.cat(xs), torch.cat(cids)
        if not held_out and x.shape[0] > 10000:
            g = torch.Generator(device='cpu').manual_seed(int(self.args.seed) + 2)
            keep = torch.randperm(x.shape[0], generator=g)[:10000].to(self.device)
            x, cid = x[keep], cid[keep]
        setattr(self, 'mol_crystals_test' if held_out else 'mol_crystals_train', (x, cid))

    def pack(self, cond, log_T, cid=None):
        """The flow's context input: the row's condition vector, when the config gives one (under
        --atom_tokens its condition id instead, whose atoms the flow looks up), and under
        temperature conditioning where log T sits in the trained range (0 at the run's own
        temperature, 1 at t_factor times it)."""
        if not self.conditional:
            return None
        if self.inn.atom_tokens:
            parts = [cid.float().to(self.device).reshape(-1, 1)]
        else:
            parts = [cond.float().to(self.device)] if self.mol_conditional else []
        if self.t_conditional:
            parts.append(((log_T.flatten().float().to(self.device) - self.log_T0) / self.t_span).unsqueeze(-1))
        return torch.cat(parts, -1)

    def context_of(self, cid, log_T=None):
        """Context input of stored conditions, at the run's own temperature unless log_T is given."""
        if not self.conditional:
            return None
        if log_T is None:
            log_T = torch.full((cid.shape[0],), self.log_T0, device=self.device)
        return self.pack(self.cond_table[cid] if self.mol_conditional else None, log_T, cid)

    def level_of(self, c):
        return self.sampler.level(c) if self.conditional else 0.0

    def training_log_T(self, n, repeats: int = 1):
        """log10 T of n training rows: the run's own temperature, or under temperature
        conditioning a log-uniform draw up to t_factor times it with t_target_frac of the
        draws left at the run's own. `repeats` rows in a row share one draw."""
        log_T = torch.full((n // repeats,), self.log_T0, device=self.device)
        if self.t_conditional:
            u = torch.rand(n // repeats, device=self.device)
            at_target = torch.rand(n // repeats, device=self.device) < self.inn.t_target_frac
            log_T = log_T + torch.where(at_target, torch.zeros_like(u), u) * self.t_span
        return log_T.repeat_interleave(repeats)

    def stored_log_r(self, idx, log_T=None):
        """Stored log R of training rows at the run's own temperature, or at log_T. A stored
        row's reward energy is a + b T -- a the physical energy less its reference, b T the terms
        the energy function scales by T so that they stay fixed in nats (the Jacobian, the
        reduction term) -- so log R(T) = -a / T - b. check_temperature_form tests this."""
        if log_T is None or not self.t_conditional:
            return self.lr_train[idx]
        return -self.a_train[idx] / 10 ** log_T.flatten() - self.b_train[idx]

    def fresh_mol_batch(self, n, dataset=None, hot: bool = False, repeats: int = 1):
        """n condition rows drawn from a conditions dataset, ready to build crystals on: with
        `repeats`, n // repeats molecules, each `repeats` times in a row. hot: at training
        temperatures (training_log_T) rather than the run's own."""
        dataset = self.mol_dataset if dataset is None else dataset
        mol_batch = next(dataset.loader(n // repeats, mode='graphs'))
        if repeats > 1:
            mol_batch = mol_batch.subsample_new_batch(
                torch.arange(n // repeats, device=mol_batch.device).repeat_interleave(repeats))
        return self.conditioned(mol_batch, self.training_log_T(n, repeats) if hot else None)

    def mol_batch_for(self, cid, log_T=None):
        """The same, for named training conditions."""
        rows = self.mol_row_of_cid[cid]
        if bool((rows < 0).any()):
            raise ValueError("a condition id with no row in mol_dataset")
        batch = self.mol_dataset.batch
        return self.conditioned(batch.subsample_new_batch(rows.to(batch.device)), log_T)

    def conditioned(self, mol_batch, log_T=None):
        """(mol_batch, log10 T per row, context input, condition id); log T is the run's own
        unless given."""
        mol_batch = mol_batch.to(self.device)
        mol_batch.orient_molecule(mode='std')
        mol_batch, run_log_T, cond, cid = self.energy_function.condition_samples(mol_batch)
        if log_T is not None:
            run_log_T = log_T.to(run_log_T).reshape(run_log_T.shape)
        return mol_batch, run_log_T, self.pack(cond, run_log_T, cid), cid.long().to(self.device)

    def jitter(self, x, log_range):
        """Stored rows displaced along a random direction by a log10-uniform magnitude in
        log_range, modelled on the trainer's anchor noise; wrapped rows are folded, bounded rows
        reflected back into [-1, 1] (their density is zero outside), other linear rows left where
        they land."""
        lo, hi = log_range
        d = torch.randn_like(x)
        d = d / d.norm(dim=-1, keepdim=True)
        y = x + d * 10 ** (lo + (hi - lo) * torch.rand(x.shape[0], 1, device=x.device))
        wrapped = self.sampler.live_idx[self.sampler.flow.is_wrap]
        y[:, wrapped] = _wrap(y[:, wrapped])
        bounded = getattr(self.sampler, 'bounded_rows', [])
        if bounded:
            yb = y[:, bounded]
            yb = torch.where(yb > 1.0, 2.0 - yb, yb)
            yb = torch.where(yb < -1.0, -2.0 - yb, yb)
            y[:, bounded] = yb.clamp(-1.0, 1.0)
        return y

    # -- startup checks: each prints what it measured, and raises past its bar ----

    @torch.no_grad()
    def check_sampler_round_trip(self, n: int = 4096, bar: float = 1e-6, bar_working: float = 1e-2):
        """sample() must invert to_base(), or log q is not the density of the draws.

        Judged in double precision, where the map's conditioning does not enter. The
        working-precision error is reported and bounded at its 99th percentile: a trained
        flow contracts volume by e^30 or more, and that Jacobian multiplies single-precision
        rounding on the way back to the base, so the maximum over draws is not a logic test.
        """
        c = self.context_of(self.cid_train[torch.randint(0, self.cid_train.shape[0], (n,), device=self.device)])

        def per_draw_error(sampler):
            x, z0 = sampler.sample(n, c, return_base=True)
            z1, _ = sampler.to_base(x, c)
            f = sampler.flow
            return torch.where(f.is_wrap, _wrap(z1 - z0), z1 - z0).abs().max(-1).values

        exact = float(per_draw_error(copy.deepcopy(self.sampler).double()).max())
        working = per_draw_error(self.sampler)
        med, p99 = quantiles(working, (0.5, 0.99))
        print(f"sampler round trip, |base - to_base(sample(base))| over {n} draws: double precision max "
              f"{exact:.2e} (bar {bar:.0e}); working precision median {med:.2e}, P99 {p99:.2e} (bar "
              f"{bar_working:.0e}), max {float(working.max()):.2e}", flush=True)
        if not exact < bar:
            raise RuntimeError(f"the flow's sampler does not invert its density map: {exact:.3e} >= {bar:.0e} "
                               f"in double precision")
        if not p99 < bar_working:
            raise RuntimeError(f"the flow's sampler is too ill-conditioned for working precision: the 99th "
                               f"percentile round-trip error is {p99:.3e} >= {bar_working:.0e} base units")

    @torch.no_grad()
    def check_stored_reward_currency(self, n: int = 512, bar: float = 0.05):
        """log R of a stored row's latent rebuilt on its condition's mol_dataset molecule -- the
        way a fresh draw is built -- against that row's stored log R."""
        idx = torch.randint(0, self.x_train.shape[0], (n,), device=self.device)
        mol_batch, log_T, _, _ = self.mol_batch_for(self.cid_train[idx])
        fresh = self.energy_function.log_reward(self.x_train[idx], mol_batch, log_T, False,
                                                 internal_oom_recovery=True).flatten().to(self.device)
        d = (fresh - self.lr_train[idx]).abs()
        d = d[torch.isfinite(d)]
        med, p99 = quantiles(d, (0.5, 0.99))
        print(f"reward currency: |log R rebuilt from the latent - stored log R| median {med:.2e}, "
              f"P99 {p99:.2e}, max {float(d.max()):.2e} nats over {d.numel()} prior rows (bar on the "
              f"median {bar})", flush=True)
        if not med < bar:
            raise RuntimeError(
                f"a stored prior row and the crystal rebuilt from its latent score differently "
                f"(median {med:.3g} nats): the data term and the draws are not in one currency")

    @torch.no_grad()
    def check_temperature_form(self, n: int = 512, bar: float = 0.05):
        """Temperature conditioning only. log R of a stored row's latent, rebuilt and scored at
        a temperature inside the trained range, against stored_log_r's a + b T form at that
        temperature: the form holds, and the energy function scores at the temperature it is
        handed rather than the run's own."""
        idx = torch.randint(0, self.x_train.shape[0], (n,), device=self.device)
        log_T = torch.full((n,), self.log_T0 + 0.5 * self.t_span, device=self.device)
        mol_batch, log_T_rows, _, _ = self.mol_batch_for(self.cid_train[idx], log_T)
        fresh = self.energy_function.log_reward(self.x_train[idx], mol_batch, log_T_rows, False,
                                                 internal_oom_recovery=True).flatten().to(self.device)
        d = (fresh - self.stored_log_r(idx, log_T)).abs()
        moved = (fresh - self.lr_train[idx]).abs()
        keep = torch.isfinite(d)
        med, p99 = quantiles(d[keep], (0.5, 0.99))
        print(f"temperature form: |log R rebuilt at T x {10 ** (0.5 * self.t_span):.3g} - (-a/T - b)| median "
              f"{med:.2e}, P99 {p99:.2e} nats over {int(keep.sum())} prior rows (bar on the median {bar}); "
              f"that temperature moves log R by a median {float(moved[keep].median()):.2f} nats", flush=True)
        if not med < bar:
            raise RuntimeError(f"a stored row's log R is not -a/T - b in the temperature (median miss "
                               f"{med:.3g} nats): stored rows cannot be re-scored at a drawn temperature")
        if not float(moved[keep].median()) > bar:
            raise RuntimeError("the energy function returned the same log R at another temperature: "
                               "it is not scoring at the temperature it is handed")

    # -- the loop ---------------------------------------------------------------

    def train(self):
        inn = self.inn
        with wandb.init(project="GFN Energy", config=flatten_wandb_params(self.args), name=self.run_name,
                        tags=[self.args.tag], mode=inn.wandb_mode):
            wandb.config.update({f'inn/{k}': v for k, v in vars(inn).items()})
            self.init_energy_function()
            self.init_gfn()
            self.init_mol_dataset()
            with self.energy_function.allow_unreferenced():
                self.init_prior_dataset()
            self.init_identifiers()
            self.init_energy_reference()
            # the trainer's per-condition tracker: energy floors, level streams, and what its
            # eval and figure code reads. Nothing in this loop's loss reads it.
            self.init_condition_log_z()
            self.init_data_rows()
            # train.py's data/ keys, and this trainer's split of the prior rows: the stored training rows
            # (prior_buffer_length) and the rows held out of training molecules (eval_bwd/mle_heldout_rows)
            self.publish_data_set_sizes(**{'data/train_rows': int(self.n_prior_rows),
                                           'data/val_rows': int(self.x_val.shape[0])})
            if inn.init_from is not None:
                # after init_data_rows: the saved standardisation replaces the one just fitted
                ck = torch.load(inn.init_from, map_location='cpu')
                self.sampler.load_state_dict(ck['state_dict'])
                # the bin-height floor is a run setting, not a weight: a different floor on the same weights is a
                # different density (the eval-only floor checks do this on purpose), so say so when they differ
                ck_floor = (ck.get('inn') or {}).get('min_bin_height')
                ck_rows = (ck.get('inn') or {}).get('min_bin_height_rows')
                if (ck_floor, ck_rows if ck_floor is not None else None) !=                         (inn.min_bin_height, inn.min_bin_height_rows if inn.min_bin_height is not None else None):
                    print(f"WARNING --init_from: the checkpoint was trained with bin-height floor {ck_floor} on "
                          f"{ck_rows} rows; this run uses {inn.min_bin_height} on {inn.min_bin_height_rows} rows "
                          f"(None = 1e-3 everywhere)", flush=True)
                ck_inn = ck.get('inn') or {}
                ck_tied = (float(ck_inn.get('tied_factor', 2.0)) if ck_inn.get('tied_derivatives', False) else None)
                run_tied = float(inn.tied_factor) if inn.tied_derivatives else None
                if ck_tied != run_tied:
                    print(f"WARNING --init_from: the checkpoint's knot derivatives were "
                          f"{'free' if ck_tied is None else f'tied, factor range {ck_tied:g}'}; this run's are "
                          f"{'free' if run_tied is None else f'tied, factor range {run_tied:g}'}: the derivative slot "
                          f"changes meaning", flush=True)
                for i, blk in enumerate(self.sampler.flow.blocks):
                    if isinstance(blk.made, CausalTransformerConditioner) and \
                            blk.made.order.tolist() != list(self.sampler.flow.orders[i]):
                        raise ValueError(f"--init_from block {i} order {blk.made.order.tolist()} differs from this "
                                         f"run's {list(self.sampler.flow.orders[i])}: pass the checkpoint's --block_orders")
                print(f"flow sampler: weights loaded from {inn.init_from}", flush=True)
            if inn.tf_fp32:
                print("flow sampler: transformer layers in fp32 (--tf_fp32)", flush=True)
            self.check_sampler_round_trip()
            if inn.dump_draws > 0:
                self.dump_draws(int(inn.dump_draws))
                return
            self.check_stored_reward_currency()
            if self.t_conditional:
                self.check_temperature_form()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()  # vram/peak_train_mb reports the loop, not the prior scan

            if inn.util_target > 0 and inn.grow_dwell_s < 90:
                raise ValueError("--grow_dwell_s must be at least 90 s: the occupancy window has to clear the sensor's "
                                 "60 s minimum span")
            self.batch_now = {'mle': int(inn.mle_batch), 'energy': self.stored_batch_base()}
            self.sizer = dict(frozen={'mle': False, 'energy': False}, t_change=time.time())
            if inn.util_target > 0:
                self._start_gpu_util_thread()  # the crystal trainer's occupancy sensor, on its own thread
            self.log_path = os.path.join(self.out_dir, 'metrics.jsonl')
            open(self.log_path, 'w').close()
            # the run's random streams from here on follow its own seed: MXtalTools constructors built during start-up
            # (the GFN geometry probe in init_gfn) call torch.manual_seed and would otherwise leave every run, whatever
            # --set seed says, on one fixed stream (step-0 evals identical across seeds)
            torch.manual_seed(int(self.args.seed) + 7919)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.args.seed) + 7919)
            # with a budget the energy stage ends on the clock (energy_fraction), not on a step count
            total = inn.mle_steps + (10 ** 9 if inn.budget_minutes is not None else inn.energy_steps)
            opt = sched = None
            log_z = self.log_z_level = None
            self.is_level = None  # --level_mode is: the stored rows' level (advance_is_level)
            self.bank = None  # --bank_frac: the tail bank (bank_add, bank_term)
            self.bank_window = dict(admitted=0, evicted=0, retired=0, retired_visits=0, retired_undrawn=0,
                                    retired_delta_sum=0.0, retired_delta_n=0)
            # per stored training row: draws so far and an EMA of |residual| (train.py's select_counts / ema_loss)
            self.stored_counts = torch.zeros(self.n_prior_rows, dtype=torch.long, device=self.device)
            self.stored_ema = torch.full((self.n_prior_rows,), float('nan'), device=self.device)
            self.mle_val_hist = []
            self.noised_ref_energy = (self.score_noised_reference(4 * int(inn.eval_samples))
                                      if inn.mle_noise_log_range is not None and inn.mle_steps > 0 else None)
            self.rows = self.scored = self.scored_d = None
            self.reset_window()
            t0 = self.t_loop0 = time.time()
            bad_streak = n_evals = 0
            best_val_nll = math.inf
            for step in range(total + 1):
                self.step_ind = step  # read by the inherited progress_metrics
                stage = 'mle' if step < inn.mle_steps else 'energy'
                if step == inn.mle_steps:
                    self.t_energy0 = time.time()
                    if inn.budget_minutes is not None:
                        print(f"energy stage: {inn.budget_minutes - (self.t_energy0 - t0) / 60:.1f} of the "
                              f"{inn.budget_minutes:g} budgeted minutes remain", flush=True)
                if step in (0, inn.mle_steps) and step < total:
                    opt = torch.optim.Adam(self.sampler.parameters(),
                                           lr=inn.lr_mle if stage == 'mle' else inn.lr_energy)
                    if stage == 'mle':
                        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                            opt, T_max=inn.mle_steps, eta_min=inn.lr_mle * 0.05)
                    else:
                        sched = torch.optim.lr_scheduler.LambdaLR(opt, self.energy_lr_factor)

                # every way out is on the step count or the clock, and both advance whatever the loss does
                done = (step == total or (time.time() - t0) / 60 > inn.max_minutes
                        or (stage == 'energy' and self.energy_fraction(step - inn.mle_steps) >= 1.0))
                if step == inn.mle_steps and step > 0:
                    self.save('mle_exit.pt', step, log_z)
                if step % inn.eval_every == 0 or step == inn.mle_steps or done:
                    self.save('latest.pt', step, log_z)
                    figs = inn.figs_every > 0 and (n_evals % inn.figs_every == 0 or done)
                    n_evals += 1
                    m = self.yielding(lambda: self.evaluate(figs), f'eval at step {step}')
                    # the eval at step mle_steps comes after the last likelihood update and before any energy update
                    eval_stage = 'mle' if 0 < inn.mle_steps and step <= inn.mle_steps else stage
                    m.update({'step': step, 'stage': eval_stage, 'phase': 1 if eval_stage == 'mle' else 2,
                              'minutes': (time.time() - t0) / 60,
                              'energy/samples_total': float(self.energy_samples_scored),
                              'batch/oom_events': float(self.oom_waits),
                              'vram/peak_train_mb': torch.cuda.max_memory_allocated() / 2 ** 20
                              if torch.cuda.is_available() else 0.0})
                    # the clash tail, which P99 does not see (clashes are ~1 draw in 500-1000): shares of the draws more
                    # than 30 and 50 kT above the best stored row, and P99.9
                    ex = m.get('Excess Energy Nats')
                    if ex is not None and np.size(ex) > 1:
                        u = 10.0 ** np.asarray(ex, dtype=np.float64) - 1.0
                        u = u[np.isfinite(u)]
                        if u.size:
                            m['Excess Energy Frac Above 30'] = float((u > 30.0).mean())
                            m['Excess Energy Frac Above 50'] = float((u > 50.0).mean())
                            m['Excess Energy Nats P99.9'] = float(np.quantile(u, 0.999))
                    self.write(m, step)
                    rows = getattr(self, 'molecule_nll_rows', {})
                    if rows:   # -log q of each molecule's own crystal, by condition id (non-finite kept)
                        np.savez(os.path.join(self.out_dir, f'molecule_nll_step{step:06d}.npz'),
                                 **{f'{side}_{k}': v for side, (c, l) in rows.items()
                                    for k, v in (('cid', c), ('nll', l))})
                    # the draws' per-row excess, log10(1 + excess kT): write() sends arrays to wandb as a 32-bin
                    # histogram only, and a bootstrap error on a tail quantile needs the rows themselves
                    ex = m.get('Excess Energy Nats')
                    if eval_stage == 'energy' and ex is not None and np.size(ex) > 1:
                        np.save(os.path.join(self.out_dir, f'excess_log10_step{step:06d}.npy'),
                                np.asarray(ex, dtype=np.float32))
                    # likelihood stage: keep the best on held-out MOLECULES where there are any, else on held-out rows
                    val_nll = float(m.get('eval_test/mle', m.get('eval_bwd/mle_heldout_rows', math.inf)))
                    if eval_stage == 'mle' and val_nll < best_val_nll:
                        best_val_nll = val_nll
                        self.save('best.pt', step, log_z)
                    self.sampler.train()
                    if eval_stage == 'mle' and inn.mle_plateau_evals > 0 and not done:
                        self.mle_val_hist.append(val_nll)
                        if self.mle_plateau(m):
                            self.save('mle_exit.pt', step, log_z)
                            break
                if done:
                    break

                self.sampler.train()
                t_step = time.perf_counter()

                def update(log_z=log_z):
                    # each condition tensor is encoded once with gradients for the step (log q, level head)
                    with (self.sampler.shared_context() if self.conditional else contextlib.nullcontext()):
                        stats, loss, level = self.step_loss(stage, log_z)
                    if loss is None or not bool(torch.isfinite(loss)):
                        return stats, False, level
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if step % inn.log_every == 0:  # before the clip, which rescales the gradients in place
                        stats.update(part_grad_norms(self.sampler))
                    stats['grad_norm_pre_clip'] = float(
                        torch.nn.utils.clip_grad_norm_(self.sampler.parameters(), inn.grad_clip))
                    if not math.isfinite(stats['grad_norm_pre_clip']):  # a finite loss with a non-finite gradient
                        opt.zero_grad(set_to_none=True)
                        return stats, False, level
                    opt.step()
                    return stats, True, level

                stats, stepped, log_z = self.yielding(update, f'step {step}')
                self.log_z_level = log_z
                self.win['seconds'] += time.perf_counter() - t_step
                if not stepped:
                    self.win['nonfinite_steps'] += 1
                    bad_streak += 1
                    if bad_streak >= 50:
                        raise RuntimeError(f"50 consecutive steps without a finite loss at step {step}")
                    continue
                bad_streak = 0
                sched.step()
                if inn.util_target > 0:
                    self.maybe_grow(stage, step)
                if step % inn.log_every == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    # phase: the crystal trainer's numbering, 1 = likelihood, 2 = energy
                    row = {'step': step, 'stage': stage, 'phase': 1 if stage == 'mle' else 2,
                           'lr_fwd': opt.param_groups[0]['lr'],
                           'Batch Size': self.batch_now['mle'] if stage == 'mle' else inn.batch,
                           'batch/stored_rows_per_update': float(self.batch_now[stage]),
                           'gpu/util_recent': (self._gpu_util_mean(900.0) if inn.util_target > 0 else None),
                           'train_step_time': time.perf_counter() - t_step}
                    row.update(self.window_metrics())
                    row.update(self.rolling_stats(stats, step))
                    self.write(row, step)

            self.flush_wandb()
            self.save('final.pt', step, log_z)
            print(f"done: {step} steps, {(time.time() - t0) / 60:.1f} min, outputs in {self.out_dir}", flush=True)

    def energy_fraction(self, s):
        """How far through the energy stage its step s is: by steps, or under budget_minutes by
        the clock -- the stage ends budget_minutes after the loop started, however long the
        likelihood stage took, so a run on a slower card still finishes its rate schedule."""
        inn = self.inn
        if inn.budget_minutes is None:
            return s / max(inn.energy_steps, 1)
        end = self.t_loop0 + 60.0 * inn.budget_minutes
        return (time.time() - self.t_energy0) / max(end - self.t_energy0, 1.0)

    def energy_lr_factor(self, s):
        """Multiplier on lr_energy at energy-stage step s: a linear warm-up, then a cosine
        from 1 to lr_energy_final / lr_energy over the stage (constant when that is unset)."""
        inn = self.inn
        warm = min(1.0, (s + 1) / max(inn.energy_warmup, 1))
        if inn.lr_energy_final is None:
            return warm
        end = inn.lr_energy_final / inn.lr_energy
        return warm * (end + (1 - end) * 0.5 * (1 + math.cos(math.pi * min(self.energy_fraction(s), 1.0))))

    def yielding(self, fn, what):
        """fn(), retried after a wait when the card is full. The GPU is shared, so a CUDA
        out-of-memory here is a co-tenant's burst rather than this run's own size. Bounded:
        the error is raised once inn.oom_retries waits have not cleared it."""
        inn = self.inn
        for attempt in range(inn.oom_retries + 1):
            try:
                return fn()
            except Exception as e:
                if not is_cuda_oom(e) or attempt == inn.oom_retries:
                    raise
                self.oom_waits += 1
                if self.shrink_after_oom():
                    del e
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                print(f"{what}: CUDA out of memory (wait {attempt + 1} of {inn.oom_retries}, "
                      f"{inn.oom_wait:.0f} s)" + (f": {str(e)[:400]}" if attempt == 0 else ""), flush=True)
                del e
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(inn.oom_wait)

    def maybe_grow(self, stage, step):
        """The occupancy batch sizer. The cluster cancels a job whose trailing ~2 h device occupancy falls under
        ~55%, and this trainer replaces the crystal trainer's loop, so its select_batch_size never runs. Rows per
        update grow -- the likelihood batch, or in the energy stage the STORED-row batch, which costs a density
        pass and no energy call; the fresh-draw batch, and so the energy-call count, is never touched. Every
        --grow_dwell_s seconds the occupancy since the last change is read; under --util_target the batch is
        multiplied by --grow_factor, up to --max_rows_per_update, while the run's peak memory since the last change
        stays under --grow_mem_frac of the card share. Bounded: at most log(max / base) / log(factor) growths per stage,
        and a stage freezes on its first out-of-memory (shrink_after_oom) or when the memory guard refuses."""
        inn, sz = self.inn, self.sizer
        now = time.time()
        if sz['frozen'][stage] or now - sz['t_change'] < inn.grow_dwell_s:
            return
        # the window must clear the sensor's own minimum span (train._UTIL_MIN_SPAN_S, 60 s) or it reads None
        u = self._gpu_util_mean(max(90.0, inn.grow_dwell_s - 20.0))
        if u is None:
            return
        cur = self.batch_now[stage]
        if u >= 100.0 * inn.util_target or cur >= inn.max_rows_per_update:
            sz['t_change'] = now  # read again a dwell later: occupancy can fall when the stage changes
            return
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(self.device).total_memory
            share = total * float(inn.cuda_memory_fraction or getattr(self.args, 'cuda_memory_fraction', None) or 1.0)
            # peak memory is set mostly by the fresh draws' sampling and scoring, not by the rows per update, so
            # growth is held only once the peak is already near the share; an out-of-memory shrinks and freezes
            if torch.cuda.max_memory_allocated() > inn.grow_mem_frac * share:
                sz['frozen'][stage] = True
                print(f"batch sizer ({stage}): held at {cur} rows per update, occupancy {u:.0f}% (memory guard)",
                      flush=True)
                return
        new = min(int(inn.max_rows_per_update), int(math.ceil(cur * inn.grow_factor)))
        self.batch_now[stage] = new
        sz['t_change'] = now
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"batch sizer ({stage}): occupancy {u:.0f}% under {100 * inn.util_target:.0f}% at step {step}: "
              f"{cur} -> {new} rows per update", flush=True)

    def shrink_after_oom(self):
        """An out-of-memory on a batch the sizer grew: back one rung and freeze that stage; True to retry at once."""
        if self.inn.util_target <= 0 or not hasattr(self, 'batch_now'):
            return False
        for stage, base in (('mle', int(self.inn.mle_batch)), ('energy', self.stored_batch_base())):
            if self.batch_now[stage] > base and not self.sizer['frozen'][stage]:
                old = self.batch_now[stage]
                self.batch_now[stage] = max(base, int(old / self.inn.grow_factor))
                self.sizer['frozen'][stage] = True
                print(f"batch sizer ({stage}): out of memory at {old} rows per update -> {self.batch_now[stage]}, "
                      f"frozen", flush=True)
                return True
        return False

    @torch.no_grad()
    @torch.no_grad()
    def per_dim_log_q(self, x):
        """log q(x) split by live latent row: each row's log|d spline/d input| summed over blocks, its share of the
        standardisation, and its base density. Sums to self.sampler.log_prob(x) (unconditional runs)."""
        s = self.sampler
        f = s.flow
        xs = s._fold(x)[:, s.live_idx]
        z = torch.where(f.is_wrap, _wrap(xs), (xs - f.mu) / f.sd)
        contrib = torch.where(f.is_wrap, torch.zeros_like(z), -torch.log(f.sd).expand_as(z))
        for blk in f.blocks:
            if isinstance(blk, _MixBlock):
                z, ld = blk.forward_per_dim(z)
                contrib = contrib + ld
                continue
            b = z.shape[0]
            pr = blk.made(blk._feats(z)).view(b, blk.d, blk.n_params)
            shift, sp = pr[..., -1], pr[..., :3 * blk.n_bins].view(b, blk.d, 3, blk.n_bins)
            if getattr(blk, 'has_bounded', False):
                # as _Block.forward: no shift on bounded rows, whose slot carries the last knot derivative
                zz = z + torch.where(blk.is_bounded, torch.zeros_like(shift), shift)
                zz = torch.where(blk.is_wrap, _wrap(zz), zz)
                z, ld = _rq_spline(zz, sp, blk.bound, blk.is_wrap, blk.is_bounded, shift,
                                   getattr(blk, 'min_height', None), getattr(blk, 'tied', False))
            else:
                zz = torch.where(blk.is_wrap, _wrap(z + shift), z + shift)
                z, ld = _rq_spline(zz, sp, blk.bound, blk.is_wrap, min_height=getattr(blk, 'min_height', None),
                                   tied=getattr(blk, 'tied', False))
            contrib = contrib + ld
        base = torch.where(f.unit_base, torch.full_like(z, -math.log(2.0)), -0.5 * z * z - 0.5 * math.log(2 * math.pi))
        return contrib + base

    @torch.no_grad()
    def dump_draws(self, n, chunk=4000):
        """--dump_draws N: N fresh draws with their log R, log q and per-row log q terms, saved to dump_draws.npz in
        the run directory; the run then ends without training (energy calls counted nowhere)."""
        if self.conditional:
            raise NotImplementedError("--dump_draws is built for unconditional runs only")
        out = {k: [] for k in ('x', 'log_r', 'log_q', 'contrib')}
        for i in range(0, n, chunk):
            m = min(chunk, n - i)
            mol, log_T, c, cid = self.fresh_mol_batch(m, hot=False)
            x = self.sampler.sample(m, c)
            lr = self.energy_function.log_reward(x, mol, log_T, False, internal_oom_recovery=True).flatten().to(x)
            lq = self.sampler.log_prob(x, c)
            pd = self.per_dim_log_q(x)
            gap = float((pd.sum(-1) - lq).abs().max())
            if gap > 1e-2:
                raise RuntimeError(f"per-row log q terms do not sum to log q (max gap {gap:.3g})")
            for k, v in (('x', x), ('log_r', lr), ('log_q', lq), ('contrib', pd)):
                out[k].append(v.float().cpu())
        arrs = {k: torch.cat(v).numpy() for k, v in out.items()}
        arrs['lr_floor'] = float(self.lr_train[:self.n_prior_rows].max())
        arrs['order'] = [list(map(int, o)) for o in self.sampler.flow.orders]
        arrs['wrap'] = self.sampler.flow.is_wrap.cpu().numpy()
        import numpy as _np
        path = os.path.join(self.out_dir, 'dump_draws.npz')
        _np.savez(path, **{k: _np.asarray(v, dtype=object) if k == 'order' else v for k, v in arrs.items()})
        print(f"dumped {n} draws to {path}", flush=True)

    def score_noised_reference(self, n):
        """Energies (-T log R) of n stored training rows displaced as stage 1 displaces them (--mle_noise_log_range),
        scored once at start-up: the reference for the stage-1 energy-marginal readout E/noised_emarg_*. Counted in
        energy/startup_calls, not in energy/samples_total."""
        g = torch.Generator(device='cpu').manual_seed(int(self.args.seed) + 7)
        idx = torch.randint(0, self.n_prior_rows, (n,), generator=g).to(self.device)
        x = self.jitter(self.x_train[idx], self.inn.mle_noise_log_range)
        log_T = self.training_log_T(n)
        mol, log_T_rows, _, _ = self.mol_batch_for(self.cid_train[idx], log_T)
        log_r = self.energy_function.log_reward(x, mol, log_T_rows, False,
                                                internal_oom_recovery=True).flatten().to(self.device)
        self.energy_startup_calls = getattr(self, 'energy_startup_calls', 0) + n
        e = (-log_r * 10 ** log_T_rows.flatten().to(log_r)).double().cpu()
        e = e[torch.isfinite(e)]
        q = torch.quantile(e, torch.tensor([0.1, 0.5, 0.9], dtype=e.dtype))
        print(f"noised reference for the stage-1 energy marginal: {e.numel()} of {n} displaced stored rows finite, "
              f"energy P10 {float(q[0]):.2f} P50 {float(q[1]):.2f} P90 {float(q[2]):.2f}", flush=True)
        return e.numpy()

    def mle_plateau(self, m):
        """--mle_plateau_evals N: True when the best held-out NLL of the last N stage-1 evals improves on the best
        before them by less than --mle_plateau_nats and w1r/worst is within --mle_plateau_w1_factor of its
        perfect-sample value. Bounded: the stage still ends at --mle_steps whatever this returns."""
        inn, h = self.inn, self.mle_val_hist
        n = int(inn.mle_plateau_evals)
        if len(h) <= n:
            return False
        gain = min(h[:-n]) - min(h[-n:])
        w1, w1p = m.get('w1r/worst'), m.get('w1r/perfect_worst')
        w1_ok = w1 is not None and w1p is not None and math.isfinite(w1) and w1 <= inn.mle_plateau_w1_factor * w1p
        if gain < inn.mle_plateau_nats and w1_ok:
            print(f"stage 1 plateau: the best held-out NLL of the last {n} evals improves on the best before them by "
                  f"{gain:.3f} nats (< {inn.mle_plateau_nats}), w1r/worst {w1:.2f} <= {inn.mle_plateau_w1_factor} x "
                  f"{w1p:.2f}; ending the likelihood stage", flush=True)
            return True
        return False

    @torch.no_grad()
    def flow_buffer_stats(self):
        """train.py's buffer keys, where this trainer reports the same quantity (Modeller.log_buffer_stats,
        _buffer_core_stats). prior_buffer_*: the stored training rows the backward branch draws from (static here, so
        added / evicted / expired are 0). replay_buffer_* and replay/*: the tail bank, the reused draws (stage 2 with
        --bank_frac); per-eval-window counts are drained here, as train.py drains its churn."""
        out = {}
        n = int(self.n_prior_rows)
        T0 = 10 ** self.log_T0
        if not hasattr(self, '_stored_energy'):
            self._stored_energy = (-self.lr_train[:n] * T0).detach()
        counts = getattr(self, 'stored_counts', None)
        out.update({'prior_buffer_length': n, 'prior_buffer_added': 0, 'prior_buffer_evicted': 0,
                    'prior_buffer_expired': 0})
        if counts is not None:
            out['prior_buffer_mean_steps'] = float(counts.float().mean())
            out['prior_buffer_median_steps'] = float(counts.float().median())
            ema = self.stored_ema[torch.isfinite(self.stored_ema)]
            if ema.numel():
                out['prior_buffer_mean_loss'] = float(ema.mean())
                out['prior_buffer_median_loss'] = float(ema.median())
        out.update(self.energy_stats('prior_buffer', energy=self._stored_energy.cpu()))
        b = getattr(self, 'bank', None)
        if b is not None:
            act = b['active']
            n_act = int(act.sum())
            w = self.bank_window
            out.update({'replay_buffer_length': n_act, 'replay_buffer_admitted': w['admitted'],
                        'replay_buffer_evicted': w['evicted'] + w['retired'],
                        'replay_buffer_turnover': w['admitted'] / max(n_act, 1)})
            resolved = w['retired'] + w['evicted']
            if resolved > 0:
                out['replay_buffer_expired_undrawn_frac'] = w['retired_undrawn'] / resolved
            if w['retired'] > 0:
                out['replay_buffer_expired_draws'] = w['retired_visits'] / w['retired']
            if w['retired_delta_n'] > 0:
                out['replay_buffer_expired_delta'] = w['retired_delta_sum'] / w['retired_delta_n']
            for k in w:
                w[k] = 0.0 if isinstance(w[k], float) else 0
            if n_act:
                vis = b['visits'][act].float()
                age = (int(self.step_ind) - b['birth'][act]).float()
                out.update({'replay_buffer_mean_steps': float(vis.mean()), 'replay_buffer_median_steps': float(vis.median()),
                            'replay_buffer_mean_age': float(age.mean()), 'replay_buffer_max_age': float(age.max()),
                            'replay_buffer_age_cv': float(age.std(unbiased=False) / age.mean().clamp(min=1e-6))
                            if n_act > 1 else 0.0})
                ema = b['ema_loss'][act]
                ema = ema[torch.isfinite(ema)]
                if ema.numel():
                    out['replay_buffer_mean_loss'] = float(ema.mean())
                    out['replay_buffer_median_loss'] = float(ema.median())
                out.update(self.energy_stats('replay_buffer', energy=(-b['lr'][act] * T0).cpu()))
                # train.py's absorption sensor: residual now against at admission, over resident rows drawn at least once
                drawn = act & torch.isfinite(b['ema_loss']) & torch.isfinite(b['birth_loss'])
                if bool(drawn.any()):
                    em, bl = float(b['ema_loss'][drawn].mean()), float(b['birth_loss'][drawn].mean())
                    ratio = em / max(bl, 1e-12)
                    out.update({'replay/ema_loss_mean': em, 'replay/birth_loss_mean': bl,
                                'replay/resid_vs_intake': ratio, 'replay/absorbed_frac': 1.0 - ratio,
                                'replay/lambda_tau': -math.log(max(ratio, 1e-12)),
                                'replay/absorption_n': float(drawn.sum())})
        return out

    def stored_batch_base(self):
        """Stored rows per energy-stage update before any growth: --stored_batch, else --batch."""
        return int(self.inn.batch if self.inn.stored_batch is None else self.inn.stored_batch)

    def advance_is_level(self, logw):
        """--level_mode is: one step of the stored rows' level from a scored batch's finite log weights on
        their first use,
            L_t = log[a e^L_{t-1} + (1 - a) mean_i exp(min(logw_i, L_{t-1} + beta))],  a = 0.5^(1 / z_half_life),
        started at the first batch's untruncated log-mean-exp. Capping a term only lowers it, so E[e^L] <= Z
        for draws of the current flow; at q = p every log weight is log Z and L is exact."""
        n = torch.tensor(float(logw.numel()), device=logw.device, dtype=logw.dtype)
        if self.is_level is None:
            self.is_level = torch.logsumexp(logw, 0) - torch.log(n)
            return self.is_level
        prev = self.is_level
        lme = torch.logsumexp(torch.minimum(logw, prev + self.inn.beta), 0) - torch.log(n)
        a = 0.5 ** (1.0 / self.inn.z_half_life)
        self.is_level = torch.logaddexp(prev + math.log(a), lme + math.log(1.0 - a))
        return self.is_level

    def bank_add(self, x, log_r, cid, birth_resid):
        """--bank_frac: admit scored draws to the tail bank, a ring of --bank_cap rows (the oldest overwritten).
        A row keeps its latent, its energy label and its condition; it starts unvisited and active."""
        n = int(x.shape[0])
        if n == 0:
            return
        cap = int(self.inn.bank_cap)
        if self.bank is None:
            dev = x.device
            self.bank = dict(x=torch.zeros(cap, x.shape[1], device=dev, dtype=x.dtype),
                             lr=torch.zeros(cap, device=dev, dtype=log_r.dtype),
                             cid=torch.zeros(cap, dtype=torch.long, device=dev),
                             visits=torch.zeros(cap, dtype=torch.long, device=dev),
                             birth=torch.zeros(cap, dtype=torch.long, device=dev),
                             birth_loss=torch.full((cap,), float('nan'), device=dev),
                             ema_loss=torch.full((cap,), float('nan'), device=dev),
                             last=torch.full((cap,), -10 ** 9, dtype=torch.long, device=dev),
                             active=torch.zeros(cap, dtype=torch.bool, device=dev), ptr=0, admitted=0, retired=0)
        b = self.bank
        if n > cap:
            x, log_r, cid, birth_resid, n = x[-cap:], log_r[-cap:], cid[-cap:], birth_resid[-cap:], cap
        pos = (b['ptr'] + torch.arange(n, device=b['x'].device)) % cap
        over = b['active'][pos]  # active rows overwritten by the ring before retiring: train.py's evictions
        b['evicted_active'] = b.get('evicted_active', 0) + int(over.sum())
        w = self.bank_window
        w['evicted'] += int(over.sum())
        w['retired_undrawn'] += int((over & (b['visits'][pos] == 0)).sum())
        w['admitted'] += n
        b['x'][pos], b['lr'][pos], b['cid'][pos] = x.to(b['x'].dtype), log_r.to(b['lr'].dtype), cid.long()
        # 'last' = the admitting update, so a row's first reuse comes at least --bank_min_gap updates after the
        # update that already trained on it as a fresh draw
        b['visits'][pos], b['last'][pos], b['active'][pos] = 0, int(getattr(self, 'step_ind', 0)), True
        b['birth'][pos] = int(getattr(self, 'step_ind', 0))
        b['birth_loss'][pos] = birth_resid.detach().abs().float()
        b['ema_loss'][pos] = float('nan')
        b['ptr'] = (b['ptr'] + n) % cap
        b['admitted'] += n

    def bank_term(self, stats, level):
        """--bank_frac: (the bank's loss this update, its share of the stored-side loss), or (None, 0) when no
        row is eligible -- nothing waits on the bank. Up to --bank_batch active rows not visited in the last
        --bank_min_gap updates are drawn WITHOUT replacement (a row is visited at most once per update), uniformly or,
        with --bank_alpha, with probability proportional to 1 + alpha x the row's latest |residual| (its running
        mean once visited, else its residual at admission), and scored against the stored level with knee
        --bank_beta. The share is --bank_frac scaled by the
        fraction of --bank_batch actually drawn, so a few eligible rows never carry the full share. A row retires
        once its residual is under --bank_retire nats or after --bank_max_visits visits."""
        inn, b = self.inn, self.bank
        if b is None:
            return None, 0.0
        step = int(getattr(self, 'step_ind', 0))
        elig = torch.nonzero(b['active'] & (step - b['last'] >= int(inn.bank_min_gap))).flatten()
        loss_b, share = None, 0.0
        if elig.numel() > 0:
            m = min(int(inn.bank_batch), int(elig.numel()))
            if inn.bank_alpha > 0 and m < int(elig.numel()):
                q = torch.where(torch.isnan(b['ema_loss'][elig]), b['birth_loss'][elig], b['ema_loss'][elig])
                wts = 1.0 + float(inn.bank_alpha) * torch.nan_to_num(q, nan=0.0).clamp(min=0.0)
                pick = elig[torch.multinomial(wts, m, replacement=False)]
            else:
                pick = elig[torch.randperm(elig.numel(), device=elig.device)[:m]]
            log_q_b = self.sampler.log_prob(b['x'][pick], self.context_of(b['cid'][pick]))
            resid_b = log_q_b - b['lr'][pick] + level
            ok_b = torch.isfinite(resid_b.detach())
            if bool(ok_b.any()):
                knee = inn.beta if inn.bank_beta is None else inn.bank_beta
                loss_b = tb_loss(resid_b[ok_b], knee).mean()
                share = float(inn.bank_frac) * min(1.0, m / float(inn.bank_batch))
                if inn.bank_anneal_from is not None:
                    # the replay weight falls linearly to 0 from this fraction of the energy stage to its end
                    frac = self.energy_fraction(step - inn.mle_steps)
                    if frac > inn.bank_anneal_from:
                        share *= max(0.0, (1.0 - frac) / max(1.0 - inn.bank_anneal_from, 1e-9))
                with torch.no_grad():
                    b['visits'][pick] += 1
                    b['last'][pick] = step
                    p_ok, a_ok = pick[ok_b], resid_b.detach()[ok_b].abs().float()
                    old = b['ema_loss'][p_ok]
                    b['ema_loss'][p_ok] = torch.where(torch.isnan(old), a_ok, 0.9 * old + 0.1 * a_ok)
                    done = torch.zeros_like(b['active'])
                    done[pick[ok_b & (resid_b.detach().abs() < inn.bank_retire)]] = True
                    done |= b['visits'] >= int(inn.bank_max_visits)
                    newly = b['active'] & done
                    b['retired'] += int(newly.sum())
                    w = self.bank_window
                    w['retired'] += int(newly.sum())
                    w['retired_visits'] += int(b['visits'][newly].sum())
                    delta = (b['ema_loss'] - b['birth_loss'])[newly]
                    delta = delta[torch.isfinite(delta)]
                    w['retired_delta_sum'] += float(delta.sum())
                    w['retired_delta_n'] += int(delta.numel())
                    b['active'] &= ~done
                    # the replay branch's rows, for rolling_stats (train.py's replay/ family)
                    self.rows_replay = dict(step=step, log_q=log_q_b.detach()[ok_b].cpu(),
                                            log_r=b['lr'][pick][ok_b].cpu(),
                                            level=torch.full((int(ok_b.sum()),), float(level)),
                                            x=b['x'][pick][ok_b].cpu(), cid=b['cid'][pick][ok_b].cpu(), beta=knee)
                stats.update({'replay/loss': float(loss_b.detach()), 'bank/share': share})
        stats['bank/eligible'] = float(elig.numel())
        return loss_b, share

    def save(self, name, step, log_z):
        torch.save({'state_dict': self.sampler.state_dict(), 'inn': vars(self.inn), 'step': step,
                    'log_z_level': log_z}, os.path.join(self.out_dir, name))

    def append_stored(self, x, log_r, cid, src):
        """Scored rows join the stored-row pool for good: an energy label never goes stale, and TB on a
        stored row needs no particular sampling distribution. Appended rows sit in a ring of stored_cap
        rows behind the prior's own; once it is full the oldest appended row is overwritten."""
        n = int(x.shape[0])
        if n == 0:
            return
        P, cap = self.n_prior_rows, int(self.inn.stored_cap)
        if n > cap:
            raise ValueError(f"--stored_cap {cap} is smaller than one scored batch ({n} rows)")
        if not hasattr(self, '_store'):
            def widen(t):
                out = torch.empty((P + cap,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
                out[:P] = t
                return out
            self._store = dict(x=widen(self.x_train), lr=widen(self.lr_train), cid=widen(self.cid_train),
                               src=widen(self.src_train))
        st = self._store
        pos = P + (self.n_appended + torch.arange(n, device=st['x'].device)) % cap
        st['x'][pos] = x.to(st['x'].dtype)
        st['lr'][pos] = log_r.to(st['lr'].dtype)
        st['cid'][pos] = cid.to(st['cid'].dtype)
        st['src'][pos] = int(src)
        self.n_appended += n
        m = P + min(self.n_appended, cap)
        self.x_train, self.lr_train, self.cid_train, self.src_train = (st[k][:m] for k in ('x', 'lr', 'cid', 'src'))

    def molecule_grouped_rows(self, n, k):
        """About n stored-row indices as n // k molecules with k rows each: molecules in proportion to their row
        counts, rows uniformly within a molecule (with replacement), so every row's marginal is the uniform draw's.
        Under --atom_tokens the encoder then runs on n // k molecules instead of about n."""
        m = self.cid_train.shape[0]
        if getattr(self, '_by_molecule', (None,))[0] != m:
            order = torch.argsort(self.cid_train)
            _, counts = torch.unique_consecutive(self.cid_train[order], return_counts=True)
            self._by_molecule = (m, order, counts, F.pad(torch.cumsum(counts, 0), (1, 0))[:-1])
        _, order, counts, start = self._by_molecule
        mol = torch.multinomial(counts.float(), max(1, n // k), replacement=True)
        off = (torch.rand(mol.numel(), k, device=counts.device) * counts[mol, None]).long()
        return order[(start[mol, None] + off).flatten()]

    def step_loss(self, stage, log_z):
        inn = self.inn
        stats = {}
        if inn.rows_per_molecule > 1:
            idx = self.molecule_grouped_rows(self.batch_now[stage], inn.rows_per_molecule)
        else:
            idx = torch.randint(0, self.x_train.shape[0],
                                (self.batch_now['mle'] if stage == 'mle' else self.batch_now['energy'],),
                                device=self.device)
        x_d, cid_d, src_all = self.x_train[idx], self.cid_train[idx], self.src_train[idx]
        log_T_d = self.training_log_T(idx.numel())
        c_d = self.context_of(cid_d, log_T_d)
        if stage == 'mle':
            if inn.mle_noise_log_range is not None:
                x_d = self.jitter(x_d, inn.mle_noise_log_range)
            mle = -self.sampler.log_prob(x_d, c_d).mean()
            stats['bwd/mle'] = stats['bwd/loss'] = float(mle.detach())
            self.win['samples'] += idx.numel()
            return stats, mle, log_z

        # A SCORED BATCH -- the fresh draws, and with noise the displaced stored rows -- costs the
        # step's energy calls and serves updates_per_batch updates. TB on detached rows needs no
        # particular sampling distribution, so a row drawn one update earlier is still a valid row.
        fresh = self.scored is None or self.scored['uses'] >= inn.updates_per_batch

        # stored rows: as stored (new rows every update: no energy call), or displaced and scored
        # again where they land (once per scored batch)
        noised = inn.data_noise_log_range is not None
        with torch.set_grad_enabled(inn.w_mle > 0 or not noised):
            log_q_stored = self.sampler.log_prob(x_d, c_d)
        mle = -log_q_stored.mean()  # always of the rows as stored
        stats['bwd/mle'] = float(mle.detach())
        if noised:
            if fresh:
                with torch.no_grad():
                    x_n = self.jitter(x_d, inn.data_noise_log_range)
                    mol_d, log_T_rows, _, _ = self.mol_batch_for(cid_d, log_T_d)
                    t = time.perf_counter()
                    lr_n = self.energy_function.log_reward(x_n, mol_d, log_T_rows, False,
                                                            internal_oom_recovery=True).flatten().to(self.device)
                    self.win['energy_seconds'] += time.perf_counter() - t
                self.energy_samples_scored += idx.numel()
                self.scored_d = dict(x=x_n, cid=cid_d, log_T=log_T_d, c=c_d, log_r=lr_n)
                if inn.grow_stored:
                    keep = torch.isfinite(lr_n) & (log_T_d.flatten() <= self.log_T0 + 1e-6)
                    self.append_stored(x_n[keep], lr_n[keep], cid_d[keep], src=2)
            x_d, cid_d, log_T_d, c_d, lr_d = (self.scored_d[k] for k in ('x', 'cid', 'log_T', 'c', 'log_r'))
            log_q_d = self.sampler.log_prob(x_d, c_d)
        else:
            lr_d, log_q_d = self.stored_log_r(idx, log_T_d), log_q_stored
        ok_d = torch.isfinite(lr_d) & torch.isfinite(log_q_d.detach())
        stats['bwd/nonfinite_frac'] = 1.0 - float(ok_d.float().mean())

        # fresh draws: iw_k per molecule when the importance-weighted term is on
        if fresh:
            K = int(inn.iw_k) if inn.w_iw > 0 else int(inn.rows_per_molecule)
            n = (inn.batch // K) * K
            with torch.no_grad():
                t = time.perf_counter()
                mol_batch, log_T, c, cid = self.fresh_mol_batch(n, hot=True, repeats=K)
                x, z0 = self.sampler.sample(n, c, return_base=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                stats['rollout/seconds'] = time.perf_counter() - t
                t = time.perf_counter()
                log_r = self.energy_function.log_reward(x, mol_batch, log_T, False,
                                                         internal_oom_recovery=not inn.fast_scoring
                                                         ).flatten().to(self.device)
                self.win['energy_seconds'] += time.perf_counter() - t
            self.energy_samples_scored += n
            self.win['samples'] += n
            if inn.grow_stored:
                keep = torch.isfinite(log_r) & (log_T.flatten() <= self.log_T0 + 1e-6)
                self.append_stored(x[keep], log_r[keep], cid[keep], src=1)
            self.scored = dict(uses=0, K=K, n=n, mol_batch=mol_batch, log_T=log_T, c=c, cid=cid, x=x, z0=z0,
                               log_r=log_r)
        sc = self.scored
        sc['uses'] += 1
        first = sc['uses'] == 1  # the draws are the current flow's own only on their first update
        K, n, mol_batch, log_T, c, cid, x, z0, log_r = (sc[k] for k in ('K', 'n', 'mol_batch', 'log_T', 'c', 'cid',
                                                                         'x', 'z0', 'log_r'))
        log_q = self.sampler.log_prob(x, c)
        ok = torch.isfinite(log_r) & torch.isfinite(log_q.detach())
        stats['fwd/nonfinite_frac'] = 1.0 - float(ok.float().mean())
        if not bool(ok.any()) or not bool(ok_d.any()):
            return stats, None, log_z
        # level of a row = the head's log Z(c) (0 on an unconditional run) + the batch's own Huber root
        h = self.level_of(c)
        h_fixed = h.detach() if self.conditional else 0.0
        h_d = self.level_of(c_d)
        h_d_fixed = h_d.detach() if self.conditional else 0.0
        logw = (log_r - log_q.detach() - h_fixed)[ok]
        if inn.level_mode == 'union':
            # one level for both branches, refit every update: the Huber root of the draws' and the stored
            # rows' log weights together, so neither branch carries a pull from a lagged or biased level
            logw_d = (lr_d - log_q_d.detach() - h_d_fixed)[ok_d]
            root = winsorized_z_root(torch.cat([logw, logw_d]), inn.beta)[0]
            level_now = root
        else:
            root = winsorized_z_root(logw, inn.beta)[0]
        fwd = tb_loss_asym((log_q - log_r + h_fixed + root)[ok], inn.beta, inn.beta_pos).mean()
        # the level the stored rows are scored against: a lagged copy of the draws' own ('tied'), or this
        # update's shared root ('union'); log_z is the lagged copy either way (it is what evaluation reads)
        log_z = root if log_z is None else log_z + (root - log_z) * (1 - 0.5 ** (1 / inn.z_half_life))
        if inn.level_mode == 'tied':
            level_now = log_z
        elif inn.level_mode == 'is':
            # advanced once per scored batch (its first use); a reused batch is scored against the same level
            level_now = (self.advance_is_level(logw.detach()) if first or self.is_level is None
                         else self.is_level)
            stats['bwd/level_is'] = float(level_now)
        elif inn.level_mode == 'blend':
            # a fixed-weight blend of the two branches' implied levels: the draws' (tied, below log Z by about
            # KL(q||p)) and the stored rows' own Huber root (above it when the rows resemble p), whatever the
            # number of rows in each
            root_d = winsorized_z_root((lr_d - log_q_d.detach() - h_d_fixed)[ok_d], inn.beta_data)[0]
            level_now = inn.level_blend * log_z + (1.0 - inn.level_blend) * root_d
            stats['bwd/level_blend'], stats['bwd/root_stored'] = float(level_now), float(root_d)
        resid_d = (log_q_d - lr_d + h_d_fixed + level_now)[ok_d]
        data = tb_loss(resid_d, inn.beta_data).mean()
        with torch.no_grad():  # train.py's per-row select_counts and ema_loss (beta 0.9) for the stored rows
            rows_d = idx[ok_d]
            prior_d = rows_d < self.n_prior_rows
            if bool(prior_d.any()):
                r_i, a_i = rows_d[prior_d], resid_d.detach().abs()[prior_d].float()
                self.stored_counts.index_add_(0, r_i, torch.ones_like(r_i))
                old = self.stored_ema[r_i]
                self.stored_ema[r_i] = torch.where(torch.isnan(old), a_i, 0.9 * old + 0.1 * a_i)
        data_stored = data  # bwd/tb reports the stored rows alone, as runs without the bank did
        if inn.bank_frac > 0:
            # the tail bank: first-use draws whose residual against their own batch's root is large are kept and
            # reused like stored rows, scored against the stored level, at a fixed share of the stored-side loss
            if first:
                # admitted: past --bank_admit nats AND among the --bank_admit_top most surprising of the batch, so
                # inflow is bounded while the whole batch is still far from the target
                resid_f = (log_q.detach() - log_r + h_fixed + root).abs()
                adm = ok & (resid_f > inn.bank_admit) & (log_T.flatten() <= self.log_T0 + 1e-6)
                if not hasattr(self, '_lr_floor'):
                    self._lr_floor = float(self.lr_train[:self.n_prior_rows].max())
                if inn.bank_max_excess is not None:
                    # no clash draws: a draw more than --bank_max_excess nats (kT) below the best stored row's log R
                    # gets its single fresh push and is never reused
                    adm = adm & ((self._lr_floor - log_r) <= inn.bank_max_excess)
                if inn.bank_min_excess is not None:
                    # clash draws only: at least --bank_min_excess kT above the best stored row, and over-weighted
                    adm = adm & ((self._lr_floor - log_r) >= inn.bank_min_excess) &                         ((log_q.detach() - log_r + h_fixed + root) > 0)
                k_max = max(1, int(math.ceil(inn.bank_admit_top * n)))
                if int(adm.sum()) > k_max:
                    top = torch.topk(torch.where(adm, resid_f, torch.full_like(resid_f, -1.0)), k_max).indices
                    adm = torch.zeros_like(adm)
                    adm[top] = True
                self.bank_add(x[adm], log_r[adm], cid[adm], resid_f[adm])
                stats['bank/admit_frac'] = float(adm.float().mean())
            bank_loss, bank_share = self.bank_term(stats, level_now)
            if bank_loss is not None:
                # mixed: the bank takes its share from the stored rows; additive: the stored rows keep their weight
                data = (data + bank_share * bank_loss if inn.bank_additive
                        else (1.0 - bank_share) * data + bank_share * bank_loss)
                stats['bwd/tb_with_bank'] = float(data.detach())
        if inn.grow_stored and not noised:  # the residual of each kind of stored row (prior, own draws, noised)
            src_d = src_all[ok_d]
            rd = resid_d.detach()
            for s_id, nm in ((0, 'prior'), (1, 'draws'), (2, 'noised')):
                sel = src_d == s_id
                if bool(sel.any()):
                    stats[f'bwd/resid_{nm}_mean'] = float(rd[sel].mean())
                    stats[f'bwd/resid_{nm}_std'] = float(rd[sel].std()) if int(sel.sum()) > 1 else 0.0
            stats['bwd/frac_appended'] = float((src_d > 0).float().mean())
            prior = src_d == 0
            stats['bwd/tb_pool'] = float(data.detach())
            stats['bwd/mle_pool'] = stats['bwd/mle']
            if bool(prior.any()):
                stats['bwd/mle'] = float(-log_q_stored.detach()[ok_d][prior].mean())
        loss = inn.w_fwd * fwd + inn.w_data * data + inn.w_mle * mle
        level = (h_fixed + log_z) if self.conditional else torch.full_like(log_r, log_z)
        # the stored rows' level as they are trained against it ('is' trains them on its own level; tied and union
        # report the tied one, as before)
        level_d = ((h_d.detach() + log_z) if self.conditional else
                   torch.full_like(log_q_d.detach(),
                                   float(level_now) if inn.level_mode in ('is', 'blend') else log_z))
        # the trainer's tracker and train-cadence series take rows at the run's own temperature;
        # stored rows go in as the branch trained on them, noised or not, as a GFN's backward
        # branch takes its noised anchors
        cold = ok & (log_T.flatten() <= self.log_T0 + 1e-6)
        cold_d = ok_d & (log_T_d <= self.log_T0 + 1e-6)
        if inn.grow_stored and not noised:  # the tracker's backward stream and the bwd/ series: prior rows only
            cold_d = cold_d & (src_all == 0)
        self.note_step(cid[cold], log_r[cold], log_q.detach()[cold], level[cold], log_T.flatten()[cold],
                       cid_d[cold_d], lr_d[cold_d], log_q_d.detach()[cold_d], level_d[cold_d],
                       x[cold], x_d[cold_d], feed_tracker=first)

        # per-branch values under the trainer's names; rolling_stats smooths them with its own
        fwd_loss, bwd_loss = inn.w_fwd * fwd, inn.w_data * data + inn.w_mle * mle
        data_rep = data_stored
        if inn.grow_stored and not noised and bool((src_all[ok_d] == 0).any()):
            data_rep = tb_loss(resid_d[src_all[ok_d] == 0], inn.beta_data).mean()
        stats.update({'fwd/tb': float(fwd.detach()), 'bwd/tb': float(data_rep.detach()),
                      'fwd/log_z_global': float(root), 'bwd/log_z_global': float(log_z)})
        if self.conditional:
            # the head regresses onto each row's log weight less the shared root, so it carries the
            # differences between conditions and the root carries the level they share
            head = tb_loss((h - (log_r - log_q.detach() - root))[ok], inn.beta).mean()
            loss = loss + inn.w_head * head
            fwd_loss = fwd_loss + inn.w_head * head
            stats['fwd/log_Z_learned_std'] = float(h.detach().std())
        if inn.w_iw > 0 and first:
            # likelihood of the flow's own draws, each weighted by its self-normalised importance
            # weight among the draws that share its condition: one group per molecule (its K
            # draws share molecule and temperature), or the whole batch when every row shares one
            group = (torch.arange(n // K, device=self.device).repeat_interleave(K) if self.conditional
                     else torch.zeros(n, dtype=torch.long, device=self.device))
            lw = torch.where(ok, log_r - log_q.detach(), torch.full_like(log_r, -float('inf')))
            top = torch.full((int(group.max()) + 1,), -float('inf'), device=self.device).scatter_reduce(
                0, group, lw, reduce='amax')
            w = torch.where(ok, torch.exp(lw - top[group]), torch.zeros_like(lw))
            w = w / torch.zeros_like(top).index_add(0, group, w)[group].clamp_min(1e-30)
            n_groups = int(torch.isfinite(top).sum())
            iw = -(w[ok] * log_q[ok]).sum() / n_groups
            loss = loss + inn.w_iw * iw
            fwd_loss = fwd_loss + inn.w_iw * iw
            ess = 1.0 / torch.zeros_like(top).index_add(0, group, w * w)[torch.isfinite(top)]
            stats.update({'fwd/iw': float(iw.detach()), 'fwd/iw_ess_mean': float(ess.mean())})
        # the pathwise term's surrogate has no meaningful value, so the branch losses leave it out
        stats.update({'fwd/loss': float(fwd_loss.detach()), 'bwd/loss': float(bwd_loss.detach())})
        if inn.w_path > 0 and first:
            k = min(int(inn.path_batch), n)
            t = time.perf_counter()
            path = self.path_term(stats, z0[:k], None if c is None else c[:k],
                                  mol_batch.subsample_new_batch(np.arange(k)), log_T[:k])
            stats['rewardgrad/seconds'] = time.perf_counter() - t
            loss = loss + inn.w_path * path
        return stats, loss, log_z

    def note_step(self, cid, log_r, log_q, level, log_T, cid_d, log_r_d, log_q_d, level_d, x, x_d,
                  feed_tracker: bool = True):
        """This step's rows to the trainer's per-condition tracker, the way its losses feed it:
        the draws' log weights (level and level streams) and energies (the running minimum), and
        the stored rows' log weights (the backward level stream). Kept for rolling_stats too,
        with their latents. Either set may be empty (step_loss passes only rows at the run's own
        temperature). A reused scored batch (feed_tracker False) refreshes the kept rows only, so
        the tracker counts each draw once."""
        cpu = lambda t: t.detach().cpu()
        cid, log_r, log_q, level, log_T, x = map(cpu, (cid, log_r, log_q, level, log_T, x))
        cid_d, log_r_d, log_q_d, level_d, x_d = map(cpu, (cid_d, log_r_d, log_q_d, level_d, x_d))
        tracker, step = self.condition_log_z, int(self.step_ind)
        if cid.numel() and feed_tracker:
            update_and_lookup_condition_log_z(tracker, cid, log_r, torch.zeros_like(log_q), log_q, 'cpu', step,
                                              log_Z_learned=level, mode_level_stream='fwd')
            update_condition_best_energy(tracker, cid, log_r, log_T)
        if cid_d.numel() and feed_tracker:
            tracker.update_mode_level('bwd', cid_d, log_r_d - log_q_d, step=step)
        self.rows = dict(step=step, cid=cid, log_r=log_r, log_q=log_q, level=level, x=x,
                         cid_d=cid_d, log_r_d=log_r_d, log_q_d=log_q_d, level_d=level_d, x_d=x_d)

    def path_term(self, stats, z0, c, mol_batch, log_T):
        """The pathwise gradient of KL(q || R/Z) through the sampler: the draws x = f(z0) are
        re-made differentiably, and each is pushed along -d(log q - log R)/dx, taken with the
        flow's parameters held fixed (the path-gradient form, which leaves out the score term
        whose expectation is zero). A surrogate: its value means nothing, its gradient is the
        term. Rows whose direction is not finite are dropped; a row's direction is clipped at
        path_clip times the batch's median norm."""
        s = self.sampler
        x = s.rsample(z0, c)
        leaf = x.detach().clone().requires_grad_(True)
        t = time.perf_counter()
        log_r = self.energy_function.log_reward(leaf, mol_batch, log_T, False, keep_grads=True).flatten()
        self.win['energy_seconds'] += time.perf_counter() - t
        force = torch.autograd.grad(log_r.sum(), leaf, allow_unused=True)[0]
        force = torch.zeros_like(leaf) if force is None else force
        score = torch.autograd.grad(s.log_prob(leaf, c).sum(), leaf)[0]
        g = (score - force.to(score.device)).detach()
        ok = torch.isfinite(g).all(-1)
        g = torch.where(ok.unsqueeze(-1), g, torch.zeros_like(g))
        norm = g.norm(dim=-1)
        bar = float(self.inn.path_clip) * float(norm[ok].median()) if bool(ok.any()) else 0.0
        scale = (bar / norm.clamp_min(1e-12)).clamp(max=1.0)
        f_ok = torch.isfinite(force).all(-1)
        f_norm = torch.nan_to_num(force.detach(), nan=0.0, posinf=0.0, neginf=0.0).norm(dim=-1)
        stats.update({'rewardgrad/force_nonfinite_frac': 1.0 - float(f_ok.float().mean()),
                      'rewardgrad/force_norm_mean': float(f_norm.mean()),
                      'rewardgrad/force_norm_max': float(f_norm.max()),
                      'rewardgrad/push_norm_mean': float((norm * scale).mean()),
                      'rewardgrad/push_nonfinite_frac': 1.0 - float(ok.float().mean()),
                      'rewardgrad/push_clipped_frac': float((scale < 1.0).float().mean()),
                      'rewardgrad/score_norm_mean': float(score.detach().norm(dim=-1).mean())})
        return ((g * scale.unsqueeze(-1)) * x).sum(-1).mean()

    # -- evaluation: the crystal trainer's own eval and figure code, fed a flow's numbers ----
    #
    # A flow is a one-step trajectory whose forward log-probability is log q(x | c) and whose
    # backward log-probability is zero, so a stats dict in the trainer's shape -- log_pfs [N, 1],
    # log_pbs [N, 1] of zeros, log_Z_learned the level a row is scored against -- goes straight
    # through Modeller's TB statistics, thermal and physical readings and figures. What has no
    # counterpart is left out: per-step drift and variance, trajectory figures, P_B, buffers.

    @torch.no_grad()
    def draw(self, n, dataset=None):
        """n fresh draws, one condition row each, scored: a stats dict in the trainer's shape
        (on the CPU, like its own eval's), the scored sample batch, and the fraction of draws
        dropped for a non-finite reward or density."""
        inn, ef, s = self.inn, self.energy_function, self.sampler
        acc = defaultdict(list)
        sample_batch = None
        for start in range(0, n, inn.batch):
            k = min(inn.batch, n - start)
            mol_batch, log_T, c, cid = self.fresh_mol_batch(k, dataset)
            x = s.sample(k, c)
            log_r, sb = ef.log_reward(x, mol_batch, log_T, True, internal_oom_recovery=True)
            sb = sb.detach().cpu()
            sample_batch = sb if sample_batch is None else sample_batch.append_batch(sb)
            acc['log_r'].append(log_r.flatten().cpu())
            acc['log_q'].append(s.log_prob(x, c).cpu())
            acc['head'].append((self.level_of(c) if self.conditional else torch.zeros(k)).cpu())
            acc['log_T_tensor'].append(log_T.flatten().cpu())
            acc['condition_id'].append(cid.cpu())
        a = {k: torch.cat(v) for k, v in acc.items()}
        ok = torch.isfinite(a['log_r']) & torch.isfinite(a['log_q'])
        nonfinite = 1.0 - float(ok.float().mean())
        if nonfinite > 0:  # the trainer's statistics take every row; a non-finite one would blank them all
            keep = ok.nonzero().flatten()
            a = {k: v[keep] for k, v in a.items()}
            sample_batch = sample_batch.subsample_new_batch(keep)
        return self.as_stats(a), sample_batch, nonfinite

    def as_stats(self, a, shared=None):
        """log q, log R and the head's level as the trainer's stats dict. The level every row is
        scored against is the head's log Z(c) plus the shared level: `shared` when given, else the
        training loop's lagged Huber root once the energy stage has one, else this batch's own."""
        if shared is None:
            shared = self.log_z_level
        if shared is None:
            shared = winsorized_z_root(a['log_r'] - a['log_q'] - a['head'], self.inn.beta)[0]
        out = dict(a)
        out['log_pfs'] = a['log_q'].unsqueeze(-1)
        out['log_pbs'] = torch.zeros_like(out['log_pfs'])
        out['log_Z_learned'] = a['head'] + float(shared)
        return out

    @torch.no_grad()
    def stored_rows(self, n, shared=None):
        """The same stats dict for n stored prior rows: the trainer's backward stream, scored against
        the shared level given (see as_stats)."""
        idx = torch.randint(0, self.n_prior_rows, (min(n, self.n_prior_rows),), device=self.device)
        c = self.context_of(self.cid_train[idx])
        return self.as_stats({
            'log_r': self.lr_train[idx].cpu(), 'log_q': self.sampler.log_prob(self.x_train[idx], c).cpu(),
            'head': (self.level_of(c) if self.conditional else torch.zeros(idx.numel())).cpu(),
            'condition_id': self.cid_train[idx].cpu()}, shared)

    def rolling_stats(self, stats, step):
        """The trainer's train-cadence series for the step just taken. The step's per-branch values
        (fwd/..., bwd/...: losses, levels) join the trainer's own statistics of the step's rows --
        quick_tb_stats, the mean log-probabilities and log R, the level, and the terminal box and
        spread readings of Modeller's per-branch stats -- in its metric tracker, so they are
        smoothed and named as its runs' are. The other values pass through unsmoothed."""
        rows, wq = self.rows, self.args.conditional_worst_quantile
        branch, out = {'fwd': {}, 'bwd': {}, 'replay': {}}, {}
        for k, v in stats.items():
            head, _, name = k.partition('/')
            if head in branch and name:
                branch[head][name] = v
            else:
                out[k] = v
        if rows is not None and rows['step'] == step:
            for b, sfx, beta in (('fwd', '', self.inn.beta), ('bwd', '_d', self.inn.beta_data)):
                log_q, log_r, level, x = (rows[k + sfx] for k in ('log_q', 'log_r', 'level', 'x'))
                if log_q.numel() < 2:
                    continue
                branch[b].update(quick_tb_stats(log_q, torch.zeros_like(log_q), level, log_r, clip_beta=beta,
                                                condition_id=rows['cid' + sfx], worst_quantile=wq))
                viol = (x.abs() - 1.0).clamp(min=0.0)
                branch[b].update(logpf_mean=float(log_q.mean()), logr_mean=float(log_r.mean()),
                                 log_Z_learned=float(level.mean()),
                                 box_violation=float((viol ** 2).sum(-1).mean()),
                                 box_contact_frac=float((viol > 0).any(-1).float().mean()),
                                 terminal_var=float(x[:, self.sampler.live_idx.cpu()].var(0).mean()))
        rp = getattr(self, 'rows_replay', None)
        if rp is not None and rp['step'] == step and rp['log_q'].numel() >= 2:
            branch['replay'].update(quick_tb_stats(rp['log_q'], torch.zeros_like(rp['log_q']), rp['level'],
                                                   rp['log_r'], clip_beta=rp['beta'], condition_id=rp['cid'],
                                                   worst_quantile=wq))
            branch['replay'].update(logpf_mean=float(rp['log_q'].mean()), logr_mean=float(rp['log_r'].mean()),
                                    log_Z_learned=float(rp['level'].mean()))
        for b, d in branch.items():
            if d:
                self.metric_tracker.update(b, d, step)
        out.update(self.metric_tracker.snapshot(changed_only=True))
        return out

    def reset_window(self):
        self.win = dict(seconds=0.0, energy_seconds=0.0, samples=0, nonfinite_steps=0)

    def window_metrics(self):
        """The trainer's throughput and energy-cost readings over the steps since the last report:
        samples_per_sec, the energy function's own call counters (energy/seconds, calls,
        ms_per_sample: every call, eval included), the share of step time spent in training
        energy calls (the reward calls alone), and the skipped non-finite steps.
        energy/samples_total counts every training draw and re-scored row scored so far, and
        rollout/every_eff is the updates each scored batch serves (the trainer's name for the
        steps between rollouts)."""
        w = self.win
        out = {'gradnorm/nonfinite_steps': float(w['nonfinite_steps']),
               'energy/samples_total': float(self.energy_samples_scored)}
        if w['seconds'] > 0:
            out['samples_per_sec'] = w['samples'] / w['seconds']
        if self.scored is not None:
            out['rollout/every_eff'] = float(self.inn.updates_per_batch)
        timing = self.energy_function.drain_energy_timing()
        if timing:
            timing['energy/seconds_in_step'] = w['energy_seconds']
            if w['seconds'] > 0:
                timing['energy/frac_of_step'] = w['energy_seconds'] / w['seconds']
                timing['energy/frac_outside_step'] = max(
                    0.0, timing['energy/seconds'] - w['energy_seconds']) / w['seconds']
        out.update(timing)
        self.reset_window()
        return out

    @torch.no_grad()
    def evaluate(self, figs: bool = False):
        inn, s = self.inn, self.sampler
        s.eval()
        arr = lambda t: t.cpu().detach().numpy()
        val = lambda t: t.cpu().detach().item()
        m = {}

        fwd, sample_batch, m['eval_fwd/nonfinite_frac'] = self.draw(inn.eval_samples)
        # one level for draws and stored rows, as the crystal trainer's learned log Z is one. Before
        # the energy stage has a level that is the draws' own; scoring the stored rows against theirs
        # centred eval_bwd/* on the rows themselves, so it jumped at the stage switch for no change
        # in the sampler (inn12, 2026-10-14: eval_bwd/under_coverage 2.8 -> 33)
        bwd = self.stored_rows(inn.eval_samples, shared=float((fwd['log_Z_learned'] - fwd['head']).mean()))
        # Emin(c) takes this batch first, as the trainer's fwd_eval_sampling does, so no draw sits below its floor
        energy = -fwd['log_r'] * 10 ** fwd['log_T_tensor']
        self.condition_log_z.update_best_energy(fwd['condition_id'], energy)

        m.update({f'eval_fwd/{k}': v for k, v in
                  self._eval_conditional_stats(fwd, self.args.fwd_loss_coeffs).items()})
        self.log_thermo_properties(arr, fwd, fwd['log_T_tensor'], fwd['log_Z_learned'], fwd['log_r'],
                                   m, sample_batch, val)
        m.update({f'eval_bwd/{k}': v for k, v in
                  self._eval_conditional_stats(bwd, self.args.bwd_loss_coeffs).items()})
        self.log_dist_stats(bwd['log_q'], m, sample_batch)
        self.progress_metrics(m, sample_batch)
        m.update(self.log_condition_log_z_stats())
        self.drop_degenerate(m)

        # this trainer's own readings, which the crystal trainer has no counterpart for
        logw = fwd['log_r'] - fwd['log_q']
        mad = lambda v: float(1.4826 * (v - v.median()).abs().median())
        m.update(self.flow_buffer_stats())
        if self.noised_ref_energy is not None and self.step_ind <= self.inn.mle_steps:
            # stage 1 fits the stored rows displaced by --mle_noise_log_range, so the reachable energy marginal is
            # that of the displaced rows (scored once at start-up), not of the stored rows themselves
            from energies.conformer_eval_metrics import energy_marginal_overlap
            if not hasattr(self, '_w1_floor_cache'):
                self._w1_floor_cache = {}
            T0 = 10 ** self.log_T0
            m.update(energy_marginal_overlap(energy.cpu().double().numpy(), self.noised_ref_energy, temperature=T0,
                                             cache=self._w1_floor_cache, prefix='E/noised_'))
        m['energy/startup_calls'] = float(getattr(self, 'energy_startup_calls', 0))
        m['eval_fwd/logw_mad'] = mad(logw)
        m['eval_bwd/logw_mad'] = mad(bwd['log_r'] - bwd['log_q'])
        m['Sample Energy Min'] = float(energy.min())
        m['Sample Energy P10'], m['Sample Energy P50'], m['Sample Energy P90'] = quantiles(energy)
        if self.mol_conditional:  # the spread left once the head's per-condition level is taken out
            m['eval_fwd/logw_std_head_centred'] = float((logw - fwd['head']).std())
            m.update(self.cross_condition_metrics())
        # working-precision conditioning of the sampler (see check_sampler_round_trip)
        c = self.context_of(self.cid_val[:2048])
        x, z0 = s.sample(c.shape[0] if self.conditional else 2048, c, return_base=True)
        z1, _ = s.to_base(x, c)
        err = torch.where(s.flow.is_wrap, _wrap(z1 - z0), z1 - z0).abs().max(-1).values
        m['eval_fwd/round_trip_p99'] = quantiles(err, (0.99,))[0]

        # in chunks, as molecule_nll: under --atom_tokens each call encodes its rows' molecules at once
        rows = lambda fn, n, k=4096: torch.cat([fn(slice(i, min(i + k, n))) for i in range(0, n, k)])
        lq_val = rows(lambda j: s.log_prob(self.x_val[j], self.context_of(self.cid_val[j])), self.x_val.shape[0])
        sub = torch.arange(min(self.x_val.shape[0], self.n_prior_rows), device=self.device)
        # -log q of stored rows: the training split, and held-out rows of the same molecules
        m['eval_bwd/mle_heldout_rows'] = float(-lq_val.mean())
        m['eval_bwd/mle'] = float(-rows(lambda j: s.log_prob(self.x_train[sub[j]], self.context_of(self.cid_train[sub[j]])),
                                        sub.numel()).mean())
        head_val = (rows(lambda j: self.level_of(self.context_of(self.cid_val[j])), self.cid_val.shape[0])
                    if self.conditional else 0.0)

        # calibration of the exact density: log q (+ the row's level) regressed on log R, on the draws
        # and on stored rows -- the parity plots' fitted line. Slope 1 where q is Boltzmann-shaped;
        # below 1 the model is flatter (hotter) than the target, above 1 sharper
        for side, st in (('fwd', fwd), ('bwd', bwd)):
            x_r = st['log_r'].double().flatten()
            y_q = (st['log_q'] + st['log_Z_learned']).double().flatten()
            ok = torch.isfinite(x_r) & torch.isfinite(y_q)
            if int(ok.sum()) > 2 and float(x_r[ok].std()) > 0:
                xc, yc = x_r[ok] - x_r[ok].mean(), y_q[ok] - y_q[ok].mean()
                m[f'eval_{side}/slope'] = float((xc * yc).sum() / (xc * xc).sum())
                m[f'eval_{side}/pearson_r'] = float((xc * yc).sum() / ((xc * xc).sum() * (yc * yc).sum()).sqrt())
        m.update(self.molecule_nll())
        if getattr(self, 'test_mol_dataset', None) is not None:
            m.update(self.held_out_metrics(arr))
        if figs:
            m.update(self.eval_figures(fwd, bwd, sample_batch, m))
        return m

    @torch.no_grad()
    def molecule_nll(self, chunk: int = 4096):
        """Exact -log q of each molecule's own conditions-file crystal under its condition: on
        held-out molecules (eval_test/mle, the likelihood on molecules never trained on) and
        on a fixed 10,000 training molecules (eval_bwd/mle_mol_crystals). eval_bwd/mle_heldout_rows
        is on held-out ROWS of training molecules, so it cannot see generalisation across molecules."""
        m = {}
        self.molecule_nll_rows = {}   # per crystal, for the gap against distance to the training molecules
        for side in ('train', 'test'):
            held = getattr(self, f'mol_crystals_{side}', None)
            if held is None or held[0].shape[0] == 0:
                continue
            x, cid = held
            lq = torch.cat([self.sampler.log_prob(x[i:i + chunk], self.context_of(cid[i:i + chunk]))
                            for i in range(0, x.shape[0], chunk)])
            self.molecule_nll_rows[side] = (cid.cpu().numpy(), (-lq).double().cpu().numpy())
            lq = lq[torch.isfinite(lq)]
            key = 'eval_test/mle' if side == 'test' else 'eval_bwd/mle_mol_crystals'
            m[key] = float(-lq.mean())
            m[f'{key}_p90'] = quantiles(-lq, (0.9,))[0]
        if len(m) == 4:
            m['eval_test/mle_gap'] = m['eval_test/mle'] - m['eval_bwd/mle_mol_crystals']
        return m

    @torch.no_grad()
    def held_out_metrics(self, arr):
        """Modeller.log_test_metrics on held-out conditions: the same draw and the same statistics
        under 'eval_test/', touching no training state. Beyond the trainer's, the excess energy
        above each held-out molecule's own conditions-file crystal (it has no other floor)."""
        T = float(self.args.energy_config.temperature)
        n = int(getattr(self.args, 'test_eval_num_samples', None) or self.args.eval_num_samples)
        test, test_batch, nonfinite = self.draw(n, self.test_mol_dataset)
        m = {f'eval_test/{k}': v for k, v in
             self._eval_conditional_stats(test, self.args.fwd_loss_coeffs).items()}
        m['eval_test/nonfinite_frac'] = nonfinite
        is_good = self._reasonable_sample_mask(test_batch)
        m['eval_test/Reasonable Sample Fraction'] = is_good.float().mean().item()
        self.log_condition_fraction(m, arr, 'Reasonable', is_good, test['condition_id'],
                                    getattr(self.args, 'reasonable_cond_bar', 0.5),
                                    higher_is_worse=False, prefix='eval_test/')
        energy = -test['log_r'] * 10 ** test['log_T_tensor']
        u = ((energy - self.test_floor.cpu()[test['condition_id']]) / T).clamp_min(0.0)
        u = u[torch.isfinite(u)]
        if u.numel():
            m['eval_test/Mean Sample Energy'] = float(test_batch.gfn_energy.mean())
            m['eval_test/Excess Energy Nats Mean'] = float(u.mean())
            (m['eval_test/Excess Energy Nats P50'], m['eval_test/Excess Energy Nats P90'],
             m['eval_test/Excess Energy Nats P99']) = quantiles(u, (0.5, 0.9, 0.99))
            p_eq = getattr(self.args, 'nonthermal_equipartition_p', 1e-6)
            if p_eq:
                bar = float(equipartition_bar(
                    int(self.sampler.live_idx.numel()), float(p_eq),
                    float(getattr(self.args, 'nonthermal_basin_window_kT', 10.0) or 0.0))[0])
                m['eval_test/Nonthermal Fraction (equipartition)'] = float((u > bar).float().mean())
        return m

    @torch.no_grad()
    def cross_condition_metrics(self):
        """eval/xcond.py's cross-conditional set. Each draw is scored under every one of C
        conditions; the flow's density is exact, so the matrix carries no estimation noise and
        the two replicates the GFN needs are the matrix itself."""
        cfg = getattr(self.args, 'xcond_eval', None)
        if cfg is None or not getattr(cfg, 'enabled', False):
            return {}
        C, n = int(getattr(cfg, 'conditions', 20)), int(getattr(cfg, 'per_condition', 4))
        known = self.cond_known.nonzero().flatten()
        if known.numel() < C:
            return {'xcond/skipped': 1.0}
        cols = self.context_of(known[torch.randperm(known.numel(), device=self.device)[:C]])   # [C, cond]
        x = self.sampler.sample(C * n, cols.repeat_interleave(n, 0))                          # condition-major
        delta = self.sampler.log_prob(x.repeat_interleave(C, 0), cols.repeat(C * n, 1)).view(C * n, C).double()
        out = xcond_metrics(delta, delta, delta, worst_quantile=float(self.args.conditional_worst_quantile))
        out.pop('xcond/noise_sd', None)  # the density is exact, so the replicate noise is zero by construction
        out['xcond/skipped'] = 0.0
        return out

    def drop_degenerate(self, m):
        """Remove the crystal trainer's readings that are constant or duplicated by construction for a
        flow: its protocol gates (gates/*, progress/reason; this loop has no protocol), and, while
        every condition gets one draw per update (no importance-weighted groups), the per-condition
        tracker's log-mean-exp level and its gap to the mean log weight -- the log-mean-exp of one
        number is that number, so the two levels are the same series and the gap is zero."""
        one_draw = not self.inn.w_iw > 0
        for k in [k for k in m if k.startswith('gates/') or k == 'progress/reason'
                  or (one_draw and ('ema_log_z_emp' in k or k.startswith('condition_log_z_')
                                    and '_gap' in k))]:
            m.pop(k)

    def eval_figures(self, fwd, bwd, sample_batch, metrics):
        """The figures of eval.evaluations.eval_figs that a flow has: the Boltzmann fit, the TB
        parity, VarGrad-error and residual plots for draws ('Forward') and stored rows
        ('Backward'), the latent distribution against the prior rows, the density funnel, and the
        per-condition tracker's cross-sections. Each block is guarded, as there."""
        fig_dict = {}
        with fig_guard('Boltzmann Fit'):
            fig_dict['Boltzmann Fit'], metrics['Boltzmann Temp Estimate'] = boltzmann_fig(fwd['log_r'])
        for prefix, st in (('Forward', fwd), ('Backward', bwd)):
            with fig_guard(f'{prefix} TB diagnostics'):
                fig_dict[f'{prefix} TB Parity Plot'], _ = flow_parity_plot(
                    st['log_r'], st['log_Z_learned'], st['log_pbs'], st['log_pfs'],
                    cond_inds=st['condition_id'] if self.mol_conditional else None)
                fig_dict[f'{prefix} VG Error'] = vargrad_error(st['log_r'], st['log_pbs'], st['log_pfs'])
                resid = torch.abs(st['log_r'] - st['log_Z_learned'] - st['log_q'])
                fig_dict[f'{prefix} TB Residual vs R'] = xy_scatter_plot(st['log_r'], resid, 'Reward', 'TB Residual')
        scaled_energy = getattr(sample_batch, 'mol_energy', None)
        if scaled_energy is None:
            scaled_energy = sample_batch[self.args.energy_function]
        with fig_guard('Lattice Latents Distribution'):
            prior_latents, _ = next(self.prior_dataset.loader(batch_size=10000, mode='tensors'))
            fig_dict['Lattice Latents Distribution'] = sample_batch.plot_batch_cell_params(
                space='latent', ref_dist=prior_latents, quantiles=[0.1], show=False, return_fig=True,
                override_energy=scaled_energy, gauge_fix_free_axes=True, aux_dists=None)
        with fig_guard('Sample Scatter'):
            n = sample_batch.num_graphs
            scatter_batch, scatter_energy = sample_batch, scaled_energy
            if n > FUNNEL_MAX_POINTS:
                idx = torch.randperm(n, device=scaled_energy.device)[:FUNNEL_MAX_POINTS]
                scatter_batch, scatter_energy = sample_batch.subsample_new_batch(idx), scaled_energy[idx]
            fig_dict['Sample Scatter'] = scatter_batch.plot_batch_density_funnel(
                show=False, return_fig=True, override_energy=scatter_energy)
        with fig_guard('condition tracker figs'):
            fig_dict.update(condition_tracker_figs(self.condition_log_z, self.energy_function, self.step_ind,
                                                   worst_quantile=self.args.conditional_worst_quantile))
        self.drop_degenerate(fig_dict)
        with fig_guard('fig filesize adjustment'):
            adjust_fig_filesize(fig_dict)
        return fig_dict

    def write(self, metrics, step):
        """Scalars to metrics.jsonl; scalars, histograms and figures to wandb, cleaned up as
        Modeller.evaluation cleans them and published through its failure-tolerant logger. The
        names are the crystal trainer's wherever it logs the same quantity; this trainer's own
        readings sit in the family they belong to. step, stage and minutes go to the file only."""
        row, out = {}, {}
        for k, v in metrics.items():
            if isinstance(v, (str, type(None))) or k in ('step', 'minutes'):  # the file's own bookkeeping
                row[k] = v if isinstance(v, (str, type(None))) else float(v)
                continue
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            if isinstance(v, np.ndarray) and v.size > 1:
                out[k] = loggable_array(v)
                continue
            try:
                row[k] = out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v  # a figure or a histogram
        if len(out) > sum(k in out for k in row):  # name what went to wandb only: histograms and figures
            row['wandb_only'] = sorted(k for k in out if k not in row)
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(row) + '\n')
        # wandb takes one commit per step: an eval row and the training row of the same step
        # go out together, when a later step's row arrives or at the end (flush_wandb)
        pending = getattr(self, '_wandb_pending', None)
        if pending is not None and pending[0] != step:
            self.flush_wandb()
            pending = None
        self._wandb_pending = (step, {**(pending[1] if pending else {}), **out})
        if 'eval_fwd/emp_z' in row:
            g = lambda key: row.get(key, float('nan'))
            test = (f"  held-out excess P50 {g('eval_test/Excess Energy Nats P50'):.1f} reasonable "
                    f"{g('eval_test/Reasonable Sample Fraction'):.3f}" if 'eval_test/emp_z' in row else '')
            print(f"[{row.get('stage')} {step}] emp_z {g('eval_fwd/emp_z'):.2f}  jensen_z "
                  f"{g('eval_fwd/jensen_z'):.2f}  ess {g('eval_fwd/ess_frac'):.4f}  E mean "
                  f"{g('Mean Sample Energy'):.1f} P50 {g('Sample Energy P50'):.1f}  excess kT P50 "
                  f"{g('Excess Energy Nats P50'):.1f} P90 {g('Excess Energy Nats P90'):.1f}  reasonable "
                  f"{g('Reasonable Sample Fraction'):.3f}{test}  val nll {g('eval_bwd/mle_heldout_rows'):.3f}  "
                  f"train nll {g('eval_bwd/mle'):.3f}"
                  + (f"  mol nll train {g('eval_bwd/mle_mol_crystals'):.3f} held-out {g('eval_test/mle'):.3f}"
                     if 'eval_test/mle' in row else '')
                  + f"  {g('minutes'):.1f} min", flush=True)

    def flush_wandb(self):
        pending, self._wandb_pending = getattr(self, '_wandb_pending', None), None
        if pending is None:
            return
        live, self.step_ind = self.step_ind, pending[0]  # _log_metrics publishes at self.step_ind
        try:
            self._log_metrics(pending[1])
        finally:
            self.step_ind = live


def build_parser():
    p = argparse.ArgumentParser(description='exact-density flow sampler on the crystal latent (prototype)')
    p.add_argument('--config', required=True, metavar='PATH')
    p.add_argument('--run_name', required=True)
    p.add_argument('--tag', default='inn')
    p.add_argument('--wandb_mode', default='online')
    p.add_argument('--prior_path', default=None,
                   help='replaces the config\'s prior_path, and its molecules_path too unless that is given')
    p.add_argument('--molecules_path', default=None, help='replaces the config\'s molecules_path')
    p.add_argument('--test_molecules_path', default=None, help='replaces the config\'s test_molecules_path')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', dest='overrides',
                   help='replace one existing config value, dotted for a nested key (the value is read as YAML)')
    p.add_argument('--cuda_memory_fraction', type=float, default=None,
                   help='replaces the config\'s hard per-process VRAM cap')
    p.add_argument('--init_from', default=None,
                   help='a mle_exit.pt / best.pt / latest.pt / final.pt of this trainer: start from its weights')
    # the flow
    p.add_argument('--n_blocks', type=int, default=1)
    p.add_argument('--n_bins', type=int, default=32)
    p.add_argument('--mix_k', type=int, default=0,
                   help='unconditional transformer runs: a mixture-CDF layer of this many wrapped-logistic components on '
                        'each wrapped row before its spline (empty gaps by construction); 0 = off')
    p.add_argument('--min_bin_height', default='auto',
                   help="floor on the spline bin heights (fraction of the interval) on --min_bin_height_rows; 'auto' "
                        "= 1e-7 on unconditional runs, the --init_from checkpoint's own floor, off on conditional "
                        "runs; 'none' = 1e-3 everywhere")
    p.add_argument('--min_bin_height_rows', choices=['uniform', 'all'], default='all',
                   help='rows --min_bin_height applies to: the uniform-base (wrapped and bounded) rows, or every row')
    p.add_argument('--tied_derivatives', action=argparse.BooleanOptionalAction, default=None,
                   help="derive each spline knot derivative from its two bins' mean slopes (harmonic mean x a learned "
                        "factor in [1/F, F], F = --tied_factor) instead of a free output; unset = on for unconditional "
                        "runs, the --init_from checkpoint's own mode, off on conditional runs")
    p.add_argument('--tied_factor', type=float, default=None,
                   help="with --tied_derivatives: the learned factor's range [1/F, F]; 1 = the pure harmonic mean; "
                        "unset = 2, or the --init_from checkpoint's own factor")
    p.add_argument('--atom_tokens', action='store_true',
                   help='conditional runs: condition on the molecule itself, one token per atom (atomic number and '
                        'standard-frame coordinates) from a ScalarGNN, which every causal layer cross-attends to, '
                        'instead of the stored embedding vector')
    p.add_argument('--atom_node_dim', type=int, default=128, help="--atom_tokens: the atom encoder's node width")
    p.add_argument('--atom_convs', type=int, default=3, help='--atom_tokens: radial message-passing rounds')
    p.add_argument('--atom_cutoff', type=float, default=10.0,
                   help='--atom_tokens: radius-graph cutoff within a molecule (angstrom)')
    p.add_argument('--atom_radial', type=int, default=32, help='--atom_tokens: radial basis functions per edge')
    p.add_argument('--rows_per_molecule', type=int, default=1,
                   help='draw each batch of stored rows as batch / k molecules with k rows each (molecules in '
                        'proportion to their rows, so each row is as likely as in a uniform draw), and in the energy '
                        'stage the fresh draws as batch / k molecules with k draws each (unless --w_iw sets the groups '
                        'by --iw_k); with --atom_tokens the encoder runs on k times fewer molecules')
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--hidden_layers', type=int, default=2)
    p.add_argument('--bound', type=float, default=5.0)
    p.add_argument('--conditioner', choices=('made', 'transformer'), default='transformer',
                   help='per-block autoregressive conditioner; transformer = causal self-attention over the order')
    p.add_argument('--tf_width', type=int, default=256)
    p.add_argument('--tf_layers', type=int, default=4)
    p.add_argument('--tf_heads', type=int, default=8)
    p.add_argument('--tf_fp32', action='store_true',
                   help="transformer layers in fp32 instead of bf16 (a trained stack of blocks can fail the "
                        "working-precision round trip in bf16)")
    p.add_argument('--block_orders', default='auto',
                   help="unconditional runs: comma-separated order per block, 'id' (stored rows), 'rev', or "
                        "'.'-separated latent row indices (e.g. 6.7.8.9.10.11.3.4.5.0.1.2); 'none' = alternate "
                        "stored, reversed, random; 'auto' = Z1_ORDER on an unconditional one-block run over the "
                        "12-row Z'=1 latent, the --init_from checkpoint's own orders, else 'none'")
    p.add_argument('--fold_half_rows', default=None,
                   help="unconditional runs: comma-separated wrapped latent rows whose distribution repeats every "
                        "half period (the prior's exact image copies, e.g. 7,8 for centroid y, z in sg 2); the flow "
                        "models each folded row and a draw picks its image uniformly (density unchanged)")
    p.add_argument('--mle_plateau_evals', type=int, default=0,
                   help="stage 1 only (--energy_steps 0): end the likelihood stage when the best held-out NLL of the "
                        "last N evals improves on the best before them by less than --mle_plateau_nats, provided "
                        "w1r/worst <= --mle_plateau_w1_factor x w1r/perfect_worst; 0 = off (run to --mle_steps)")
    p.add_argument('--mle_plateau_nats', type=float, default=0.1)
    p.add_argument('--mle_plateau_w1_factor', type=float, default=1.5)
    p.add_argument('--bounded_rows', default='auto',
                   help="unconditional runs: linear latent rows modelled on the closed interval [-1, 1] by a spline "
                        "with no tails and a uniform base, so q has no mass outside the latent box on them; "
                        "'linear' = every live unwrapped row, or a comma-separated list; 'none' = off; 'auto' = "
                        "'linear' on unconditional runs, the --init_from checkpoint's own rows, off on conditional runs")
    p.add_argument('--context_dim', type=int, default=128, help='conditional runs: width of the condition embedding')
    p.add_argument('--context_hidden', type=int, default=512, help='conditional runs: width of the embedding MLP')
    # the two stages
    p.add_argument('--mle_steps', type=int, default=5000)
    p.add_argument('--energy_steps', type=int, default=5000)
    p.add_argument('--mle_batch', type=int, default=8192)
    p.add_argument('--batch', type=int, default=2000)
    p.add_argument('--stored_batch', type=int, default=None,
                   help='energy stage: stored rows per update (the occupancy sizer grows it from here); '
                        'unset = --batch')
    p.add_argument('--lr_mle', type=float, default=1e-3)
    p.add_argument('--lr_energy', type=float, default=2e-4)
    p.add_argument('--energy_warmup', type=int, default=200)
    p.add_argument('--lr_energy_final', type=float, default=None,
                   help='cosine-decay the energy-stage rate to this value; unset = constant after warm-up')
    p.add_argument('--grad_clip', type=float, default=10.0)
    p.add_argument('--mle_noise_log_range', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                   help='mle stage: displace each stored row by a log10-uniform magnitude in latent units; unset = off')
    # energy-stage loss weights and Huber knees
    p.add_argument('--w_fwd', type=float, default=1.0)
    p.add_argument('--w_data', type=float, default=1.0)
    p.add_argument('--w_mle', type=float, default=0.0)
    p.add_argument('--w_head', type=float, default=1.0, help='conditional runs: weight of the level head\'s own term')
    p.add_argument('--w_path', type=float, default=0.0,
                   help='weight of the pathwise energy-gradient term (differentiable draws and energy); 0 = off')
    p.add_argument('--path_batch', type=int, default=512, help='draws per step the pathwise term re-makes')
    p.add_argument('--path_clip', type=float, default=10.0,
                   help='pathwise term: per-row clip of the push direction, in multiples of the batch median norm')
    p.add_argument('--beta', type=float, default=10.0)
    p.add_argument('--beta_data', type=float, default=80.0)
    p.add_argument('--z_half_life', type=float, default=20.0)
    p.add_argument('--level_mode', choices=['tied', 'union', 'is', 'blend'], default='tied',
                   help="energy stage: 'tied' scores stored rows against a lagged copy of the draws' Huber root; "
                        "'union' scores both branches against one root refit every update over draws and stored "
                        "rows together; 'is' scores stored rows against an exponentially averaged importance "
                        "estimate of log Z from each scored batch's draws on first use, every log weight capped at "
                        "the previous level + --beta (see InnModeller.advance_is_level); 'blend' scores stored (and "
                        "bank) rows against --level_blend x the tied level + (1 - --level_blend) x the stored "
                        "batch's own Huber root (knee --beta_data). Evaluation reads the tied level in every mode")
    p.add_argument('--level_blend', type=float, default=0.5,
                   help="--level_mode blend: weight of the draws' (tied) level; the rest is the stored rows' own root")
    # the replay bank: first-use draws with a large residual, reused as stored-like rows
    p.add_argument('--bank_frac', type=float, default=None,
                   help='energy stage: weight of the replay bank (its own mean); 0 = off; unset = 0.1 on unconditional '
                        'runs, 0 on conditional runs')
    p.add_argument('--bank_admit', type=float, default=2.0,
                   help='bank: a first-use draw is admitted when |log q - log R + root| exceeds this (nats)')
    p.add_argument('--bank_max_excess', type=float, default=1e9,
                   help='bank: only draws within this many kT (nats of log R) of the best stored row are admitted; '
                        'clash draws further out get their one fresh push and are never reused')
    p.add_argument('--bank_admit_top', type=float, default=1.0,
                   help='bank: at most this fraction of a scored batch is admitted per update, the largest residuals')
    p.add_argument('--bank_retire', type=float, default=1.0,
                   help='bank: a row retires once its residual against the stored level is under this (nats)')
    p.add_argument('--bank_max_visits', type=int, default=8, help='bank: a row retires after this many visits')
    p.add_argument('--bank_min_gap', type=int, default=10,
                   help='bank: a row is not revisited within this many updates of its last visit')
    p.add_argument('--bank_batch', type=int, default=256, help='bank: rows per update (with replacement when fewer)')
    p.add_argument('--bank_cap', type=int, default=200000, help='bank: ring capacity in rows')
    p.add_argument('--bank_beta', type=float, default=None, help='bank: Huber knee; unset = --beta')
    p.add_argument('--bank_anneal_from', type=float, default=None,
                   help='bank: from this fraction of the energy stage, the replay weight falls linearly to 0 at its end')
    p.add_argument('--bank_additive', action=argparse.BooleanOptionalAction, default=True,
                   help='bank: add the bank loss at weight --bank_frac x (fraction drawn) to the stored-side loss, '
                        'instead of taking that share from the stored rows')
    p.add_argument('--split_seed', type=int, default=None,
                   help='seed of the held-out-crystal split; unset = the run seed + 1')
    p.add_argument('--bank_alpha', type=float, default=1.0,
                   help='bank: draw eligible rows with probability proportional to 1 + alpha |residual|; 0 = uniform')
    p.add_argument('--bank_min_excess', type=float, default=None,
                   help='bank: admit only over-weighted draws at least this many kT above the best stored row (clashes)')
    p.add_argument('--beta_pos', type=float, default=None,
                   help='fresh branch: Huber knee for positive residuals (over-weighted draws); inf = quadratic; '
                        'unset = --beta')
    p.add_argument('--dump_draws', type=int, default=0,
                   help='after start-up, save N fresh draws with log R, log q and per-row log q terms, then stop')
    p.add_argument('--fast_scoring', action='store_true',
                   help="energy stage: score fresh draws without the energy function's per-call out-of-memory "
                        "recovery (its gc.collect / empty_cache cost ~0.26 s per update); an OOM still reaches "
                        "the trainer's own retry")
    # energy-stage additions, each off by default
    p.add_argument('--data_noise_log_range', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                   help='energy stage: displace each stored row by a log10-uniform magnitude in latent units and '
                        'score it again where it lands (one more energy call per step); unset = rows as stored')
    p.add_argument('--t_factor', type=float, default=1.0,
                   help='temperature conditioning: train over T in [1, t_factor] x the run\'s temperature with '
                        'log T as a flow input, evaluate at the run\'s own; 1 = off')
    p.add_argument('--t_target_frac', type=float, default=0.25,
                   help='temperature conditioning: share of training rows kept at the run\'s own temperature')
    p.add_argument('--updates_per_batch', type=int, default=1,
                   help='energy stage: updates each scored batch serves (fresh draws and re-scored rows are scored '
                        'once per batch; the importance-weighted and pathwise terms apply on its first update only)')
    p.add_argument('--w_iw', type=float, default=0.0,
                   help='weight of the importance-weighted likelihood of the flow\'s own draws; 0 = off')
    p.add_argument('--iw_k', type=int, default=8,
                   help='importance-weighted term: draws per molecule (the batch is batch // iw_k molecules)')
    # reporting
    p.add_argument('--val_frac', type=float, default=0.05)
    p.add_argument('--val_group_ignore_rows', default=None,
                   help="hold out whole crystals: rows agreeing on every latent row but these (comma-separated; the "
                        "prior's image rows, e.g. 7,8 in sg 2) are one crystal. Unset = rows held out at random")
    p.add_argument('--grow_stored', action='store_true',
                   help="energy stage: every scored row (fresh draws, noised stored rows) joins the stored-row pool "
                        "for good, so later updates draw it again with no energy call")
    p.add_argument('--stored_cap', type=int, default=3000000,
                   help="--grow_stored: appended rows kept behind the prior's (a ring; the oldest overwritten)")
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--eval_samples', type=int, default=10000)
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--figs_every', type=int, default=5,
                   help='draw the eval figures on every Nth eval and on the last; 0 = never')
    p.add_argument('--max_minutes', type=float, default=240.0, help='hard stop on the loop\'s wall time')
    p.add_argument('--budget_minutes', type=float, default=None,
                   help='end the energy stage this many minutes after the loop starts, with its rate schedule '
                        'stretched over the time that leaves; --energy_steps is then ignored')
    p.add_argument('--util_target', type=float, default=0.0,
                   help='occupancy batch sizer: grow rows per update while the card is busier than this FRACTION '
                        'less often (0 = off; see maybe_grow)')
    p.add_argument('--grow_dwell_s', type=float, default=180.0, help='sizer: seconds between occupancy reads')
    p.add_argument('--grow_factor', type=float, default=1.5, help='sizer: multiplier per growth')
    p.add_argument('--max_rows_per_update', type=int, default=60000,
                   help='sizer: ceiling on rows per update (under 65,535: bf16 attention puts the batch on a CUDA grid '
                        'axis of that size)')
    p.add_argument('--grow_mem_frac', type=float, default=0.8,
                   help="sizer: grow only while peak memory since the last change stays under this fraction of the card share")
    p.add_argument('--oom_retries', type=int, default=120, help='waits on a full card before the error is raised')
    p.add_argument('--oom_wait', type=float, default=10.0, help='seconds per wait')
    return p


def apply_overrides(args, overrides):
    """--set a.b=value: replace an EXISTING config value. A key the config does not carry is
    refused, so a misspelling cannot pass as an override."""
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        node, *path = args, *key.split('.')
        for name in path[:-1]:
            node = getattr(node, name)
        if not hasattr(node, path[-1]):
            raise KeyError(f"--set {key}: the config has no such key")
        setattr(node, path[-1], yaml.safe_load(value))
        print(f"config override: {key} = {getattr(node, path[-1])!r}", flush=True)


ARM_OVERRIDES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'inn_arm_overrides.yaml')

# the unconditional route's defaults for the options left unset ('auto' / None) on the command line: bounded linear rows,
# tied knot derivatives, the bin-height floor and error-filtered replay. The conditional route refuses all four.
UNCONDITIONAL_DEFAULTS = dict(bounded_rows='linear', min_bin_height=1e-7, tied_derivatives=True, tied_factor=2.0,
                              bank_frac=0.1)
# the default row order of a one-block flow over the 12-row crystal latent at Z' = 1 (rows 0-11 all live): centroid z,
# the orientation rows, centroid y and x, then the cell angles and lengths
Z1_ORDER = '8.11.10.9.7.6.5.4.3.2.1.0'


def resolve_auto_options(inn, conditional, ck_inn=None, live_rows=None):
    """Fill the options left unset with a value and return a one-line account of where each came from.

    Unset = 'auto' (`--bounded_rows`, `--min_bin_height`) or None (`--tied_derivatives`, `--tied_factor`,
    `--bank_frac`). The density options (bounded rows, floor, derivative mode and factor) take the `--init_from`
    checkpoint's own setting when `ck_inn` (its saved `inn` entry) is given, a key absent there meaning the setting it
    was trained under before the key existed (off, 1e-3 everywhere, free); otherwise the unconditional route's default
    (UNCONDITIONAL_DEFAULTS), or off on a conditional run. Replay (`--bank_frac`) is a training setting and takes the
    route's default either way. 'none' turns `--bounded_rows` or `--min_bin_height` off explicitly. `--block_orders`
    'auto' is Z1_ORDER on an unconditional one-block run whose live rows `live_rows` are exactly 0-11, the
    checkpoint's own orders under `--init_from`, else none (the alternating orders); 'none' sets the alternation."""
    src = []

    def pick(key, from_ck, route, off):
        if ck_inn is not None and from_ck:
            val, where = ck_inn.get(key, off), 'checkpoint'
        elif conditional:
            val, where = off, 'conditional route'
        else:
            val, where = route, 'unconditional default'
        src.append(f"{key}={val} ({where})")
        return val

    if inn.bounded_rows == 'auto':
        inn.bounded_rows = pick('bounded_rows', True, UNCONDITIONAL_DEFAULTS['bounded_rows'], None)
    elif inn.bounded_rows == 'none':
        inn.bounded_rows = None
    if inn.min_bin_height == 'auto':
        inn.min_bin_height = pick('min_bin_height', True, UNCONDITIONAL_DEFAULTS['min_bin_height'], None)
    elif inn.min_bin_height == 'none':
        inn.min_bin_height = None
    elif inn.min_bin_height is not None:
        inn.min_bin_height = float(inn.min_bin_height)
    if inn.tied_derivatives is None:
        inn.tied_derivatives = bool(pick('tied_derivatives', True, UNCONDITIONAL_DEFAULTS['tied_derivatives'], False))
    if inn.tied_factor is None:
        inn.tied_factor = float(pick('tied_factor', True, UNCONDITIONAL_DEFAULTS['tied_factor'], 2.0)
                                if inn.tied_derivatives else 2.0)
    if inn.bank_frac is None:
        inn.bank_frac = float(pick('bank_frac', False, UNCONDITIONAL_DEFAULTS['bank_frac'], 0.0))
    if inn.block_orders == 'auto':
        z1 = int(inn.n_blocks) == 1 and live_rows is not None and list(live_rows) == list(range(12))
        inn.block_orders = pick('block_orders', True, Z1_ORDER if z1 else None, None)
    elif inn.block_orders == 'none':
        inn.block_orders = None
    return ', '.join(src)


def apply_arm_overrides(inn, path=ARM_OVERRIDES):
    """Changes to an arm that is already queued. A job script is fixed at submission, but the code
    and this file are read when the job starts, so the entry keyed `<tag>_<run_name>` -- the names
    the arm was submitted under -- replaces those flags of its command line. Every replacement is
    printed, and the run's wandb config holds the flags as finally applied (arm_overrides names the
    entry). A flag this trainer does not have is refused; an arm with no entry is untouched."""
    inn.arm_overrides = None
    if not os.path.exists(path):
        return
    key = f'{inn.tag}_{inn.run_name}'
    entry = (yaml.safe_load(open(path)) or {}).get(key)
    if not entry:
        return
    unknown = sorted(set(entry) - set(vars(inn)))
    if unknown:
        raise SystemExit(f"{path}: entry {key} names flags this trainer does not have: {unknown}")
    for k, v in entry.items():
        old = getattr(inn, k)
        new = float(v) if isinstance(old, float) and isinstance(v, int) else v
        if old is not None and new is not None and type(new) is not type(old):
            raise SystemExit(f"{path}: entry {key}: {k} = {v!r} is a {type(v).__name__}, the flag a "
                             f"{type(old).__name__}")
        setattr(inn, k, new)
        print(f"arm override ({key} in {os.path.basename(path)}): {k} = {new!r} (submitted as {old!r})", flush=True)
    inn.arm_overrides = key


def refuse_foreign_paths(args, windows: bool = os.name == 'nt'):
    """Off Windows, a config value that is still a Windows drive path is one of the canonical
    config's local settings that the launch did not replace. Raise before anything reads it or
    creates a directory of that name."""
    if windows:
        return
    left = sorted(k for k, v in vars(flatten_wandb_params(args)).items()
                  if isinstance(v, str) and re.match(r'^[A-Za-z]:[\\/]', v))
    if left:
        raise SystemExit(f"config values still hold Windows drive paths on this host: {left}. "
                         f"Replace each with --prior_path / --molecules_path / --test_molecules_path or --set.")


if __name__ == '__main__':
    _inn = build_parser().parse_args()
    apply_arm_overrides(_inn)  # before anything reads the flags, run_name included
    _args = get_train_args(['--config', _inn.config])
    _args.tag, _args.run_name = _inn.tag, _inn.run_name
    # this trainer loads no checkpoint and writes none
    _args.checkpoint_name, _args.continue_from_checkpoint, _args.checkpoint_read_only = None, False, True
    if _inn.prior_path is not None:
        _args.prior_path = _inn.prior_path
        _args.molecules_path = _inn.prior_path
    if _inn.molecules_path is not None:
        _args.molecules_path = _inn.molecules_path
    if _inn.test_molecules_path is not None:
        _args.test_molecules_path = _inn.test_molecules_path
    if _inn.cuda_memory_fraction is not None:
        _args.cuda_memory_fraction = _inn.cuda_memory_fraction
    apply_overrides(_args, _inn.overrides)
    refuse_foreign_paths(_args)
    if _inn.budget_minutes is not None and _inn.max_minutes <= _inn.budget_minutes:
        raise SystemExit("--max_minutes must exceed --budget_minutes: it is the hard stop behind the budget")

    from gpu_guard import require_free_gpu, GPUBusy

    try:
        require_free_gpu()
    except GPUBusy as _e:
        raise SystemExit(str(_e))

    InnModeller(args=_args, inn=_inn).train()
