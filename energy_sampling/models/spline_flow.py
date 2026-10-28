"""The exact-density spline flows over the crystal latent: the unconditional LatentFlow and the conditional
ConditionalLatentFlow (an autoregressive rational-quadratic spline per block, a masked-MLP or causal-transformer
conditioner, an optional atom-token molecule encoder, and the level head log Z(c)), with their block inverses.

Moved verbatim from inn_modeller.py (2026-10-27) so a policy class in models/ can build them without importing the
INN trainer; inn_modeller.py re-imports every name. Wiki: ar-spline-flow.md."""
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from energies.ar_transformer import NORMS, CausalTransformerConditioner
from energies.prior_flow import _Block, _Flow, _MADE, _rq_spline, _wrap, invert_block, rq_spline_inverse

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
    """_Block whose conditioner is a _ContextMADE. Bounded rows as _Block's: the closed [-1, 1], no tails, no shift."""

    def __init__(self, order, wrap_mask, n_bins, hidden, bound, context_dim, bounded_mask=None):
        super().__init__(order, wrap_mask, n_bins, hidden, bound, bounded_mask)
        self.made = _ContextMADE(order, wrap_mask, hidden, n_bins, context_dim)

    def spline_params(self, x, ctx):
        p = self.made(self._feats(x), ctx).view(x.shape[0], self.d, self.n_params)
        return p[..., -1], p[..., :3 * self.n_bins].view(x.shape[0], self.d, 3, self.n_bins)

    def forward(self, x, ctx):
        shift, sp = self.spline_params(x, ctx)
        if self.has_bounded:
            z = x + torch.where(self.is_bounded, torch.zeros_like(shift), shift)
            z = torch.where(self.is_wrap, _wrap(z), z)
            y, ld = _rq_spline(z, sp, self.bound, self.is_wrap, self.is_bounded, shift, self.min_height, self.tied)
            return y, ld.sum(-1)
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
            if self.has_bounded:
                z = rq_spline_inverse(y, sp, self.bound, self.is_wrap, self.is_bounded, shift, self.min_height,
                                      self.tied)
                shift = torch.where(self.is_bounded, torch.zeros_like(shift), shift)
            else:
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
                 transformer=None, atoms=None, bounded_mask=None):
        super().__init__()
        d = len(wrap_mask)
        # bounded dims (see _Block) are not standardised and take a uniform base, as wrapped dims do
        bmask = [False] * d if bounded_mask is None else [bool(b) for b in bounded_mask]
        self.register_buffer('is_bounded', torch.tensor(bmask, dtype=torch.bool), persistent=False)
        self.register_buffer('unit_base', torch.tensor([bool(w) or b for w, b in zip(wrap_mask, bmask)],
                                                       dtype=torch.bool), persistent=False)
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
            _ContextBlock(o, wrap_mask, n_bins, hidden, bound, context_dim, bmask) for o in orders)
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
        """Standardise the LINEAR dims only; wrapped dims must stay on the circle, and bounded dims on their
        interval."""
        mu, sd = x.mean(0), x.std(0).clamp_min(1e-4)
        self.mu.copy_(torch.where(self.unit_base, torch.zeros_like(mu), mu))
        self.sd.copy_(torch.where(self.unit_base, torch.ones_like(sd), sd))

    def set_condition_scaler(self, c):
        self.c_mu.copy_(c.mean(0))
        self.c_sd.copy_(c.std(0).clamp_min(1e-4))

    def standardise(self, c):
        return (c.to(self.c_mu.dtype) - self.c_mu) / self.c_sd


class ConditionalLatentFlow(nn.Module):
    """q(x | c): LatentFlow with every block reading an embedding of the condition vector c."""

    def __init__(self, dim, wrap_mask, dead_rows, cond_dim, context_dim, context_hidden,
                 n_blocks, n_bins, hidden, bound, seed, transformer=None, atoms=None, bounded_rows=()):
        super().__init__()
        dead = {int(d) for d in dead_rows}
        live = [i for i in range(dim) if i not in dead]
        # bounded rows: linear rows modelled on the closed interval [-1, 1], as LatentFlow's
        self.bounded_rows = [int(r) for r in bounded_rows]
        if any(r not in live or wrap_mask[r] for r in self.bounded_rows):
            raise ValueError(f"bounded_rows {self.bounded_rows}: each must be a live linear (unwrapped) row")
        self.dim = dim
        self.register_buffer('live_idx', torch.tensor(live, dtype=torch.long))
        self.flow = _ContextFlow([bool(wrap_mask[i]) for i in live], cond_dim, context_dim, context_hidden,
                                 n_blocks, n_bins, tuple(hidden), bound, seed, transformer=transformer, atoms=atoms,
                                 bounded_mask=[i in self.bounded_rows for i in live])

    def to_base(self, x, c, detach_context: bool = False):
        f = self.flow
        ctx = f.context_input(c)[0]
        if detach_context:   # the blocks train, the condition's encoder does not (scramble_conditions)
            ctx = tuple(t.detach() for t in ctx) if isinstance(ctx, tuple) else ctx.detach()
        x = x[:, self.live_idx]
        z = torch.where(f.is_wrap, _wrap(x), (x - f.mu) / f.sd)
        ld = x.new_zeros(x.shape[0]) - torch.log(f.sd[~f.is_wrap]).sum()
        for blk in f.blocks:
            z, l = blk(z, ctx)
            ld = ld + l
        return z, ld

    def log_prob(self, x, c, detach_context: bool = False):
        f = self.flow
        z, ld = self.to_base(x, c, detach_context=detach_context)
        zl = z[:, ~f.unit_base]
        base = (-math.log(2.0) * int(f.unit_base.sum())
                - 0.5 * (zl * zl).sum(-1) - 0.5 * math.log(2 * math.pi) * zl.shape[1])
        return base + ld

    @torch.no_grad()
    def sample(self, n, c, return_base: bool = False):
        f = self.flow
        dev, dtype = f.mu.device, f.mu.dtype
        ctx = f.context_input(c)[0]
        z0 = torch.where(f.unit_base, 2.0 * torch.rand(n, f.d, device=dev, dtype=dtype) - 1.0,
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
        """log Z(c) up to the batch's shared level: the head, zero at initialisation. It reads a detached copy of
        the condition features, so its regression trains the head alone and not the atom encoder every flow block
        reads (the invariant models/gfn.py::GFN._condition_flow keeps on the SDE route)."""
        return self.flow.level(self.flow.context_input(c)[1].detach()).squeeze(-1)
