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
rewardgrad/). log P_B is zero for a one-step map, so every TB statistic reads
log R - log q exactly; "bwd"/"Backward" means the stored prior rows.
"""
import argparse
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


def invert_differentiable(blk, y, ctx=None):
    """A block's inverse at y with the autograd graph kept (prior_flow.invert_block and
    _ContextBlock.invert run without one). After blk.d passes the result is the exact
    preimage for any parameter value, so the graph through the passes carries its exact
    derivative."""
    x = torch.zeros_like(y)
    mh = getattr(blk, 'min_height', None)
    for _ in range(blk.d):
        if ctx is None:
            p = blk.made(blk._feats(x)).view(y.shape[0], blk.d, blk.n_params)
            shift, sp = p[..., -1], p[..., :3 * blk.n_bins].view(y.shape[0], blk.d, 3, blk.n_bins)
        else:
            shift, sp = blk.spline_params(x, ctx)
        z = rq_spline_inverse(y, sp, blk.bound, blk.is_wrap, min_height=mh)
        x = torch.where(blk.is_wrap, _wrap(z - shift), z - shift)
    return x


class LatentFlow(nn.Module):
    """q(x) over the full latent: a spline flow on the live rows, dead rows pinned at 0."""

    def __init__(self, dim, wrap_mask, dead_rows, n_blocks, n_bins, hidden, bound, seed, transformer=None,
                 block_orders=None, half_rows=()):
        super().__init__()
        dead = {int(d) for d in dead_rows}
        live = [i for i in range(dim) if i not in dead]
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
        self.flow = _Flow(live_wrap, n_blocks=n_blocks, n_bins=n_bins, hidden=tuple(hidden), bound=bound, seed=seed)
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
                _Block(o, live_wrap, n_bins, tuple(hidden), bound) for o in self.flow.orders)
        if transformer is not None:  # (width, layers, heads): each block's MADE replaced by a causal transformer
            for blk, order in zip(self.flow.blocks, self.flow.orders):
                blk.made = CausalTransformerConditioner(order, blk.feat_dim.tolist(), n_bins, *transformer)

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
        z0 = torch.where(f.is_wrap, 2.0 * torch.rand(n, f.d, device=dev, dtype=dtype) - 1.0,
                         torch.randn(n, f.d, device=dev, dtype=dtype))
        z = z0
        for blk in reversed(list(f.blocks)):
            z = invert_block(blk, z, f.d)
        x = torch.zeros(n, self.dim, device=dev, dtype=dtype)
        x[:, self.live_idx] = torch.where(f.is_wrap, _wrap(z), z * f.sd + f.mu)
        x = self._unfold(x)
        return (x, z0) if return_base else x

    def rsample(self, z0, c=None):
        """The draw sample() makes from base points z0, differentiable in the parameters."""
        if self.half_rows:
            raise NotImplementedError("rsample with --fold_half_rows: the image choice is not part of z0")
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
        y, ld = _rq_spline(z, sp, self.bound, self.is_wrap)
        return y, ld.sum(-1)

    @torch.no_grad()
    def invert(self, y, ctx):
        """x with forward(x, ctx) == y: one more coordinate exact per pass, whatever the order."""
        x = torch.zeros_like(y)
        for _ in range(self.d):
            shift, sp = self.spline_params(x, ctx)
            z = rq_spline_inverse(y, sp, self.bound, self.is_wrap)
            x = torch.where(self.is_wrap, _wrap(z - shift), z - shift)
        return x


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
                 transformer=None):
        super().__init__()
        d = len(wrap_mask)
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
        self.blocks = nn.ModuleList(
            _ContextBlock(o, wrap_mask, n_bins, hidden, bound, context_dim) for o in orders)
        if transformer is not None:  # (width, layers, heads): each block's _ContextMADE replaced by a causal transformer
            for blk, order in zip(self.blocks, orders):
                blk.made = CausalTransformerConditioner(order, blk.feat_dim.tolist(), n_bins, *transformer,
                                                        context_dim=context_dim)
        self.context = _ResMLP(cond_dim, context_hidden, context_dim)
        self.level = _ResMLP(cond_dim, 256, 1)
        nn.init.zeros_(self.level.out[-1].weight)
        nn.init.zeros_(self.level.out[-1].bias)

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
                 n_blocks, n_bins, hidden, bound, seed, transformer=None):
        super().__init__()
        dead = {int(d) for d in dead_rows}
        live = [i for i in range(dim) if i not in dead]
        self.dim = dim
        self.register_buffer('live_idx', torch.tensor(live, dtype=torch.long))
        self.flow = _ContextFlow([bool(wrap_mask[i]) for i in live], cond_dim, context_dim, context_hidden,
                                 n_blocks, n_bins, tuple(hidden), bound, seed, transformer=transformer)

    def to_base(self, x, c):
        f = self.flow
        ctx = f.context(f.standardise(c))
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
        ctx = f.context(f.standardise(c))
        z0 = torch.where(f.is_wrap, 2.0 * torch.rand(n, f.d, device=dev, dtype=dtype) - 1.0,
                         torch.randn(n, f.d, device=dev, dtype=dtype))
        z = z0
        for blk in reversed(list(f.blocks)):
            z = blk.invert(z, ctx)
        x = torch.zeros(n, self.dim, device=dev, dtype=dtype)
        x[:, self.live_idx] = torch.where(f.is_wrap, _wrap(z), z * f.sd + f.mu)
        return (x, z0) if return_base else x

    def rsample(self, z0, c):
        """The draw sample() makes from base points z0, differentiable in the parameters."""
        f = self.flow
        ctx = f.context(f.standardise(c))
        z = z0
        for blk in reversed(list(f.blocks)):
            z = invert_differentiable(blk, z, ctx)
        x = torch.zeros(z0.shape[0], self.dim, device=z0.device, dtype=z0.dtype)
        x[:, self.live_idx] = torch.where(f.is_wrap, _wrap(z), z * f.sd + f.mu)
        return x

    def level(self, c):
        """log Z(c) up to the batch's shared level: the head, zero at initialisation."""
        return self.flow.level(self.flow.standardise(c)).squeeze(-1)


def tb_loss(resid, beta):
    """gflownet_losses.get_tb_loss's Huber form, per row."""
    return beta * F.smooth_l1_loss(resid, torch.zeros_like(resid), reduction='none', beta=beta)


def quantiles(x, qs=(0.1, 0.5, 0.9)):
    return [float(v) for v in torch.quantile(x.double().flatten(), torch.tensor(qs, dtype=torch.float64,
                                                                               device=x.device))]


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
        cond_dim = mol_dim + int(self.t_conditional)
        self.log_T0 = math.log10(float(self.args.energy_config.temperature))
        self.t_span = math.log10(inn.t_factor) if self.t_conditional else 1.0
        shape = (inn.n_blocks, inn.n_bins, (inn.hidden,) * inn.hidden_layers, inn.bound)
        transformer = (inn.tf_width, inn.tf_layers, inn.tf_heads) if inn.conditioner == 'transformer' else None
        block_orders = inn.block_orders.split(',') if inn.block_orders else None
        half_rows = [int(r) for r in inn.fold_half_rows.split(',')] if inn.fold_half_rows else []
        if half_rows and inn.w_path > 0:
            raise NotImplementedError("--fold_half_rows with --w_path: rsample does not carry the image choice")
        if self.conditional:
            if block_orders is not None or half_rows:
                raise NotImplementedError("--block_orders and --fold_half_rows are built for unconditional runs only")
            self.sampler = ConditionalLatentFlow(dim, wrap_mask, dead_rows, cond_dim, inn.context_dim,
                                                 inn.context_hidden, *shape, seed=int(self.args.seed),
                                                 transformer=transformer)
        else:
            self.sampler = LatentFlow(dim, wrap_mask, dead_rows, *shape, seed=int(self.args.seed),
                                      transformer=transformer, block_orders=block_orders, half_rows=half_rows)
        self.sampler = self.sampler.to(self.device)
        if inn.min_bin_height is not None:
            # a lower floor on the spline's bin heights on the chosen rows: with a uniform base (wrapped and bounded
            # rows) each bin holds at least its height floor of the row's mass whatever the weights, so the default
            # 1e-3 x n_bins is mass no training can remove
            if not isinstance(self.sampler, LatentFlow):
                raise NotImplementedError("--min_bin_height is built for unconditional runs only")
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
        if inn.tf_fp32:
            if not (isinstance(self.sampler, LatentFlow) and inn.conditioner == 'transformer'):
                raise NotImplementedError("--tf_fp32 needs an unconditional run with --conditioner transformer")
            self.sampler.set_bf16(False)
        n_par = sum(p.numel() for p in self.sampler.parameters())
        print(f"flow sampler: dim {dim}, wrapped rows {[i for i, w in enumerate(wrap_mask) if w]}, "
              f"dead rows {list(dead_rows)}, {inn.n_blocks} blocks x {inn.n_bins} bins, "
              + (f"block orders {block_orders}, " if block_orders is not None else "")
              + (f"half-period images folded on rows {half_rows}, " if half_rows else "")
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

        g = torch.Generator(device='cpu').manual_seed(int(self.args.seed) + 1)
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

        # where a condition's molecule sits in mol_dataset, for building a named condition's crystals
        mol_cid = ef.condition_ids_of(self.mol_dataset.batch).to(self.device)
        self.mol_row_of_cid = torch.full((n_cond,), -1, dtype=torch.long, device=self.device)
        self.mol_row_of_cid[mol_cid] = torch.arange(mol_cid.numel(), device=self.device)

        if self.mol_conditional:
            # the conditions files' own crystals: one reference structure per molecule, held-out included
            self.fold_condition_crystals(self.mol_dataset, held_out=False)
            if getattr(self, 'test_mol_dataset', None) is not None:
                self.fold_condition_crystals(self.test_mol_dataset, held_out=True)
        if self.conditional:
            ids = (self.cond_known.nonzero().flatten() if self.mol_conditional
                   else torch.zeros(4096, dtype=torch.long, device=self.device))
            self.sampler.flow.set_condition_scaler(self.context_of(ids, self.training_log_T(ids.numel())))

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

    def pack(self, cond, log_T):
        """The flow's context input: the row's condition vector, when the config gives one, and
        under temperature conditioning where log T sits in the trained range (0 at the run's
        own temperature, 1 at t_factor times it)."""
        if not self.conditional:
            return None
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
        return self.pack(self.cond_table[cid] if self.mol_conditional else None, log_T)

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
        return mol_batch, run_log_T, self.pack(cond, run_log_T), cid.long().to(self.device)

    def jitter(self, x, log_range):
        """Stored rows displaced along a random direction by a log10-uniform magnitude in
        log_range, modelled on the trainer's anchor noise; wrapped rows are folded, linear rows
        left where they land."""
        lo, hi = log_range
        d = torch.randn_like(x)
        d = d / d.norm(dim=-1, keepdim=True)
        y = x + d * 10 ** (lo + (hi - lo) * torch.rand(x.shape[0], 1, device=x.device))
        wrapped = self.sampler.live_idx[self.sampler.flow.is_wrap]
        y[:, wrapped] = _wrap(y[:, wrapped])
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
            if inn.init_from is not None:
                # after init_data_rows: the saved standardisation replaces the one just fitted
                ck = torch.load(inn.init_from, map_location='cpu')
                self.sampler.load_state_dict(ck['state_dict'])
                # the bin-height floor is a run setting, not a weight: a different floor on the same weights is a
                # different density, so say so when they differ
                ck_floor = (ck.get('inn') or {}).get('min_bin_height')
                ck_rows = (ck.get('inn') or {}).get('min_bin_height_rows')
                if (ck_floor, ck_rows if ck_floor is not None else None) !=                         (inn.min_bin_height, inn.min_bin_height_rows if inn.min_bin_height is not None else None):
                    print(f"WARNING --init_from: the checkpoint was trained with bin-height floor {ck_floor} on "
                          f"{ck_rows} rows; this run uses {inn.min_bin_height} on {inn.min_bin_height_rows} rows "
                          f"(None = 1e-3 everywhere)", flush=True)
                for i, blk in enumerate(self.sampler.flow.blocks):
                    if isinstance(blk.made, CausalTransformerConditioner) and \
                            blk.made.order.tolist() != list(self.sampler.flow.orders[i]):
                        raise ValueError(f"--init_from block {i} order {blk.made.order.tolist()} differs from this "
                                         f"run's {list(self.sampler.flow.orders[i])}: pass the checkpoint's --block_orders")
                print(f"flow sampler: weights loaded from {inn.init_from}", flush=True)
            if inn.tf_fp32:
                print("flow sampler: transformer layers in fp32 (--tf_fp32)", flush=True)
            self.check_sampler_round_trip()
            self.check_stored_reward_currency()
            if self.t_conditional:
                self.check_temperature_form()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()  # vram/peak_train_mb reports the loop, not the prior scan

            if inn.util_target > 0 and inn.grow_dwell_s < 90:
                raise ValueError("--grow_dwell_s must be at least 90 s: the occupancy window has to clear the sensor's "
                                 "60 s minimum span")
            self.batch_now = {'mle': int(inn.mle_batch), 'energy': int(inn.batch)}
            self.sizer = dict(frozen={'mle': False, 'energy': False}, t_change=time.time())
            if inn.util_target > 0:
                self._start_gpu_util_thread()  # the crystal trainer's occupancy sensor, on its own thread
            self.log_path = os.path.join(self.out_dir, 'metrics.jsonl')
            open(self.log_path, 'w').close()
            # with a budget the energy stage ends on the clock (energy_fraction), not on a step count
            total = inn.mle_steps + (10 ** 9 if inn.budget_minutes is not None else inn.energy_steps)
            opt = sched = None
            log_z = self.log_z_level = None
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
                    self.write(m, step)
                    # likelihood stage: keep the best on held-out MOLECULES where there are any, else on held-out rows
                    val_nll = float(m.get('eval_test/mle', m.get('eval_bwd/mle_heldout_rows', math.inf)))
                    if eval_stage == 'mle' and val_nll < best_val_nll:
                        best_val_nll = val_nll
                        self.save('best.pt', step, log_z)
                    self.sampler.train()
                if done:
                    break

                self.sampler.train()
                t_step = time.perf_counter()

                def update(log_z=log_z):
                    stats, loss, level = self.step_loss(stage, log_z)
                    if loss is None or not bool(torch.isfinite(loss)):
                        return stats, False, level
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
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
        for stage, base in (('mle', int(self.inn.mle_batch)), ('energy', int(self.inn.batch))):
            if self.batch_now[stage] > base and not self.sizer['frozen'][stage]:
                old = self.batch_now[stage]
                self.batch_now[stage] = max(base, int(old / self.inn.grow_factor))
                self.sizer['frozen'][stage] = True
                print(f"batch sizer ({stage}): out of memory at {old} rows per update -> {self.batch_now[stage]}, "
                      f"frozen", flush=True)
                return True
        return False

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

    def step_loss(self, stage, log_z):
        inn = self.inn
        stats = {}
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
            K = int(inn.iw_k) if inn.w_iw > 0 else 1
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
                                                         internal_oom_recovery=True).flatten().to(self.device)
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
        fwd = tb_loss((log_q - log_r + h_fixed + root)[ok], inn.beta).mean()
        # the level the stored rows are scored against: a lagged copy of the draws' own ('tied'), or this
        # update's shared root ('union'); log_z is the lagged copy either way (it is what evaluation reads)
        log_z = root if log_z is None else log_z + (root - log_z) * (1 - 0.5 ** (1 / inn.z_half_life))
        if inn.level_mode != 'union':
            level_now = log_z
        resid_d = (log_q_d - lr_d + h_d_fixed + level_now)[ok_d]
        data = tb_loss(resid_d, inn.beta_data).mean()
        stats['bwd/stored_rows'] = float(self.x_train.shape[0])
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
        level_d = (h_d.detach() + log_z) if self.conditional else torch.full_like(log_q_d.detach(), log_z)
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
        data_rep = data
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
        branch, out = {'fwd': {}, 'bwd': {}}, {}
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

        lq_val = s.log_prob(self.x_val, self.context_of(self.cid_val))
        sub = torch.arange(min(self.x_val.shape[0], self.n_prior_rows), device=self.device)
        # -log q of stored rows: the training split, and held-out rows of the same molecules
        m['eval_bwd/mle_heldout_rows'] = float(-lq_val.mean())
        m['eval_bwd/mle'] = float(-s.log_prob(self.x_train[sub], self.context_of(self.cid_train[sub])).mean())
        head_val = self.level_of(self.context_of(self.cid_val)) if self.conditional else 0.0

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
        for side in ('train', 'test'):
            held = getattr(self, f'mol_crystals_{side}', None)
            if held is None or held[0].shape[0] == 0:
                continue
            x, cid = held
            lq = torch.cat([self.sampler.log_prob(x[i:i + chunk], self.context_of(cid[i:i + chunk]))
                            for i in range(0, x.shape[0], chunk)])
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
    p.add_argument('--n_blocks', type=int, default=6)
    p.add_argument('--n_bins', type=int, default=16)
    p.add_argument('--min_bin_height', type=float, default=None,
                   help='floor on the spline bin heights (fraction of the interval) on --min_bin_height_rows; '
                        'unset = 1e-3 everywhere')
    p.add_argument('--min_bin_height_rows', choices=['uniform', 'all'], default='all',
                   help='rows --min_bin_height applies to: the uniform-base (wrapped and bounded) rows, or every row')
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--hidden_layers', type=int, default=2)
    p.add_argument('--bound', type=float, default=5.0)
    p.add_argument('--conditioner', choices=('made', 'transformer'), default='made',
                   help='per-block autoregressive conditioner; transformer = causal self-attention over the order')
    p.add_argument('--tf_width', type=int, default=256)
    p.add_argument('--tf_layers', type=int, default=4)
    p.add_argument('--tf_heads', type=int, default=8)
    p.add_argument('--tf_fp32', action='store_true',
                   help="transformer layers in fp32 instead of bf16 (a trained stack of blocks can fail the "
                        "working-precision round trip in bf16)")
    p.add_argument('--block_orders', default=None,
                   help="unconditional runs: comma-separated order per block, 'id' (stored rows), 'rev', or "
                        "'.'-separated latent row indices (e.g. 6.7.8.9.10.11.3.4.5.0.1.2); "
                        "unset = alternate stored, reversed, random")
    p.add_argument('--fold_half_rows', default=None,
                   help="unconditional runs: comma-separated wrapped latent rows whose distribution repeats every "
                        "half period (the prior's exact image copies, e.g. 7,8 for centroid y, z in sg 2); the flow "
                        "models each folded row and a draw picks its image uniformly (density unchanged)")
    p.add_argument('--context_dim', type=int, default=128, help='conditional runs: width of the condition embedding')
    p.add_argument('--context_hidden', type=int, default=512, help='conditional runs: width of the embedding MLP')
    # the two stages
    p.add_argument('--mle_steps', type=int, default=5000)
    p.add_argument('--energy_steps', type=int, default=5000)
    p.add_argument('--mle_batch', type=int, default=8192)
    p.add_argument('--batch', type=int, default=2000)
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
    p.add_argument('--level_mode', choices=['tied', 'union'], default='tied',
                   help="energy stage: 'tied' scores stored rows against a lagged copy of the draws' Huber root; "
                        "'union' scores both branches against one root refit every update over draws and stored "
                        "rows together")
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
