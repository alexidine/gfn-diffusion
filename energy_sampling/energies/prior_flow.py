"""
Normalizing-flow proxy for the trained prior policy's density.

WHAT THIS IS FOR. The prior is a SAMPLER: you can draw from it, but there is no
tractable log p. A masked autoregressive flow fitted to its draws supplies one --
exactly normalised by construction, deterministic, and cheap to evaluate. That is
what makes a lambda=0 null test possible: run the conditional objective against
the prior's OWN implied energy, where the policy should converge to the fitted
density and stop rather than wander.

WHY THIS FAMILY. Seven were gated against an importance-weighted ground truth on
real prior draws (2026-08-30). Total deviation, sqrt(resid^2 + ((slope-1)*sd)^2):
autoregressive flow 0.72, coupling flow 0.88, copula 1.25, low-rank mixture 1.54,
gaussian mixture 1.68, k-nearest-neighbour 1.94, kernel density 2.52. Every
smoothing-based estimator fails at d=12 for the same structural reason (its
neighbourhood is not local); a flow cannot, because its normaliser is exact, so
its only possible error is underfitting. Note SMALLER IS BETTER here: 4 blocks of
256 beat 6x512 and a 10x1024 diverged outright, and 50000 training steps scored
worse than 12000.

CALIBRATION IS NOT OPTIONAL. Goodness of fit is the wrong criterion. The consumer
reads only the SPREAD of log-weights within a condition group, so an additive
constant cancels while a MULTIPLICATIVE error on log p does not -- and correlation
does not discriminate between them (a rejected estimator correlated 0.99 at slope
0.70). Gate any refit through energies/density_calibration.py.

THE ARTIFACT IS TIED TO ITS SAMPLING GEOMETRY *AND* TO T. The wrap mask and dead
rows follow from (space group, max_z_prime, periodic_centroids); the terminal
distribution also depends on the integrator's trajectory length. T is ABSENT from
problem_def, so a config that changes it would silently invalidate this file with
nothing to catch it. Both are stored here and re-checked against the live policy
in train.py's init_gfn.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "maf_circular"

_MIN_BIN = 1e-3
_MIN_DERIV = 1e-3
_SOFTPLUS_ONE = math.log(math.e - 1.0)  # softplus(_SOFTPLUS_ONE) == 1


def _wrap(u: torch.Tensor, period: float = 2.0) -> torch.Tensor:
    """Fold onto [-period/2, period/2]. The seam is a coordinate artifact only."""
    return u - period * torch.round(u / period)


_TRIU_CACHE = {}


def _cumsum_last(t):
    """Prefix sums along a SHORT last axis, as a matmul against upper-triangular 1s.

    torch.cumsum along a length-12 last axis of a [B, D, 2, K] tensor measured
    3.8-8.0 ms at B=8192 here -- the single largest cost in the whole flow, an
    order of magnitude above the 512x512 matmuls. The equivalent [.., K] @ [K, K]
    runs in 0.22 ms and agrees to 2e-6.
    """
    k = t.shape[-1]
    key = (k, t.device, t.dtype)
    u = _TRIU_CACHE.get(key)
    if u is None:
        u = torch.triu(torch.ones(k, k, device=t.device, dtype=t.dtype))
        _TRIU_CACHE[key] = u
    return t @ u


def _knot_derivatives(params, circular, bounded=None, end_raw=None):
    """The K + 1 knot derivatives of a spline: a circular dim ties the ends (d_K = d_0); a
    linear dim pins both ends to 1 (C1 identity tails); a BOUNDED dim (a closed interval,
    no tails) leaves both ends free, its last derivative read from `end_raw` (the slot a
    linear dim spends on its shift), offset so that a zero parameter gives slope 1."""
    n_bins = params.shape[-1]
    circ = circular.view(1, -1, 1)
    d = _MIN_DERIV + F.softplus(params[..., 2, :])
    out = torch.where(circ,
                      torch.cat([d, d[..., :1]], dim=-1),                  # d_K == d_0
                      F.pad(d[..., :n_bins - 1], (1, 1), value=1.0))       # C1 identity tails
    if bounded is not None:
        d_end = (_MIN_DERIV + F.softplus(end_raw + _SOFTPLUS_ONE)).unsqueeze(-1)
        out = torch.where(bounded.view(1, -1, 1), torch.cat([d, d_end], dim=-1), out)
    return out


def _tied_derivatives(params, slopes, circular, bounded=None, end_raw=None, max_factor=2.0):
    """The K + 1 knot derivatives DERIVED from the bins' mean slopes `slopes` [B, D, K] (height / width): at an
    interior knot the harmonic mean of its two bins' slopes (the Fritsch-Butland choice of monotone interpolation; it
    lies between the smaller slope and twice it), times a learned factor in [1/max_factor, max_factor] from the
    derivative slot (1 at a zero-initialised head; max_factor 1 is the pure harmonic mean). No knot is then far steeper
    than both bins it joins, so a near-empty bin cannot carry a density spike at its edge: inside any bin with no
    pinned end the density stays below 8 x max_factor x its mean slope. Ends as _knot_derivatives: a linear dim pins
    both to 1, a circular dim joins bin K - 1 to bin 0 across the seam, a bounded dim gives each end its own bin's
    slope times a factor (the last read from `end_raw`)."""
    lf = math.log(float(max_factor))
    f = torch.exp(lf * torch.tanh(params[..., 2, :] - _SOFTPLUS_ONE))
    prev = torch.roll(slopes, 1, dims=-1)                    # knot k joins bin k - 1 (bin K - 1 at k = 0) and bin k
    d = 2.0 * prev * slopes / (prev + slopes) * f
    out = torch.where(circular.view(1, -1, 1),
                      torch.cat([d, d[..., :1]], dim=-1),               # d_K == d_0
                      F.pad(d[..., 1:], (1, 1), value=1.0))             # C1 identity tails
    if bounded is not None:
        f_end = torch.exp(lf * torch.tanh(end_raw)).unsqueeze(-1)
        ends = torch.cat([slopes[..., :1] * f[..., :1], d[..., 1:], slopes[..., -1:] * f_end], dim=-1)
        out = torch.where(bounded.view(1, -1, 1), ends, out)
    return out


def _bin_fractions(params, n_bins, min_height=None):
    """Bin widths and heights as fractions of the interval, each at least _MIN_BIN; `min_height` (optional [D]
    tensor) replaces the heights' floor per dim."""
    wh = F.softmax(params[..., :2, :], dim=-1)
    if min_height is None:
        return _MIN_BIN + (1.0 - _MIN_BIN * n_bins) * wh
    lo = torch.stack((torch.full_like(min_height, _MIN_BIN), min_height), -1).view(1, -1, 2, 1)
    return lo + (1.0 - lo * n_bins) * wh


def _rq_spline(inputs, params, bound, circular, bounded=None, end_raw=None, min_height=None, tied=False):
    """Monotone rational-quadratic spline; returns (outputs, log|dy/dx|).

    `inputs` [B, D]; `params` [B, D, 3, K] -> unnormalised widths, heights,
    derivatives. `bound` and `circular` are PER-DIM ([D] tensors), so the wrapped
    and the linear dims of one block go through a single call: on a contended GPU
    this flow is launch-bound, and two calls cost twice the kernels for the same
    arithmetic. A circular dim ties the end derivatives (d_K = d_0), giving a
    diffeomorphism of the circle [-bound, bound]; a linear dim pins them to 1 and
    is the identity outside the box, a diffeomorphism of the line. A `bounded` dim
    (optional per-dim mask, with `end_raw` [B, D] its last-derivative parameter) is a
    diffeomorphism of the closed interval [-bound, bound] with no tails: an input
    outside it has zero density (log-derivative -inf). `tied` derives the knot
    derivatives from the bin slopes (_tied_derivatives) instead of reading them free:
    True for a learned factor in [1/2, 2], or a number F >= 1 for [1/F, F].
    """
    n_bins = params.shape[-1]
    b = bound.view(1, -1, 1)

    wh = _bin_fractions(params, n_bins, min_height)
    cum = _cumsum_last(wh)
    cum = cum / cum[..., -1:]                       # pin the far knot exactly at 1
    cum = F.pad(cum, (1, 0))
    cum = (2.0 * b.unsqueeze(-1)) * cum - b.unsqueeze(-1)
    lens = cum[..., 1:] - cum[..., :-1]
    cumw, cumh = cum[..., 0, :], cum[..., 1, :]
    w, h = lens[..., 0, :], lens[..., 1, :]

    d = (_tied_derivatives(params, h / w, circular, bounded, end_raw, 2.0 if tied is True else float(tied)) if tied
         else _knot_derivatives(params, circular, bounded, end_raw))

    x = torch.maximum(torch.minimum(inputs, b.squeeze(-1)), -b.squeeze(-1))
    idx = torch.searchsorted(cumw.detach().contiguous(),
                             x.unsqueeze(-1).contiguous(), right=True)
    idx = (idx - 1).clamp_(0, n_bins - 1)

    stacked = torch.stack((cumw[..., :-1], w, cumh[..., :-1], h,
                           d[..., :-1], d[..., 1:]), dim=-2)          # [B, D, 6, K]
    sel = stacked.gather(-1, idx.unsqueeze(-2).expand(idx.shape[:-1] + (6, 1)))
    xk, wk, yk, hk, dk, dk1 = sel.squeeze(-1).unbind(-1)

    s = hk / wk
    xi = ((x - xk) / wk).clamp(0.0, 1.0)
    xi1 = xi - xi * xi

    den = s + (dk + dk1 - 2.0 * s) * xi1
    y = yk + hk * (s * xi * xi + dk * xi1) / den
    dnum = s * s * (dk1 * xi * xi + 2.0 * s * xi1 + dk * (1.0 - xi) ** 2)
    logdet = torch.log(dnum.clamp_min(1e-30)) - 2.0 * torch.log(den.clamp_min(1e-30))

    # linear dims are the identity outside their box; circular dims have no
    # outside (their input is folded onto [-bound, bound] before the call);
    # bounded dims have no mass outside
    outside = (~circular).view(1, -1) & (inputs != x)
    if bounded is None:
        y = torch.where(outside, inputs, y)
        logdet = torch.where(outside, torch.zeros_like(logdet), logdet)
    else:
        bnd = bounded.view(1, -1)
        tails = outside & ~bnd
        y = torch.where(tails, inputs, y)
        logdet = torch.where(tails, torch.zeros_like(logdet), logdet)
        logdet = torch.where(outside & bnd, torch.full_like(logdet, -float('inf')), logdet)
    return y, logdet


class _MaskedLinear(nn.Linear):
    def __init__(self, n_in, n_out, mask):
        super().__init__(n_in, n_out)
        self.register_buffer('mask', mask.to(self.weight.dtype))

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class _MADE(nn.Module):
    """Standard MADE, with a per-dim FEATURE GROUP on the input side.

    A wrapped dim contributes two input features (sin, cos) that share one
    autoregressive degree; a linear dim contributes one. Output block i carries
    3*n_bins spline parameters plus one shift, and is masked to depend only on
    dims strictly earlier in `order`.
    """

    def __init__(self, order, wrap_mask, hidden, n_bins):
        super().__init__()
        d = len(order)
        n_params = 3 * n_bins + 1
        deg = [0] * d
        for slot, dim in enumerate(order):
            deg[dim] = slot + 1                      # degrees 1..d

        feat_deg = []
        for i in range(d):
            feat_deg += [deg[i], deg[i]] if wrap_mask[i] else [deg[i]]
        feat_deg = torch.tensor(feat_deg)
        out_deg = torch.tensor(deg).repeat_interleave(n_params)

        widths = list(hidden)
        hid_deg = [torch.arange(w) % max(d - 1, 1) + 1 for w in widths]

        layers = [_MaskedLinear(feat_deg.numel(), widths[0],
                                (hid_deg[0][:, None] >= feat_deg[None, :]))]
        for a in range(1, len(widths)):
            layers += [nn.SiLU(),
                       _MaskedLinear(widths[a - 1], widths[a],
                                     (hid_deg[a][:, None] >= hid_deg[a - 1][None, :]))]
        head = _MaskedLinear(widths[-1], d * n_params,
                             (out_deg[:, None] > hid_deg[-1][None, :]))
        # start every block at the identity map -- zero head weights give uniform
        # bins, and a derivative bias of softplus^-1(1) makes every knot slope 1,
        # which is exactly the identity spline. A stack of these is the identity
        # at step 0, so the flow never has to climb out of a random warp.
        nn.init.zeros_(head.weight)
        with torch.no_grad():
            head.bias.zero_()
            head.bias.view(d, 3 * n_bins + 1)[:, 2 * n_bins:3 * n_bins] = _SOFTPLUS_ONE
        layers += [nn.SiLU(), head]
        self.net = nn.Sequential(*layers)

    def forward(self, feats):
        return self.net(feats)


class _Block(nn.Module):
    """One autoregressive layer over all d dims, in the given `order`."""

    def __init__(self, order, wrap_mask, n_bins, hidden, bound, bounded_mask=None):
        super().__init__()
        d = len(order)
        self.d, self.n_bins = d, n_bins
        self.n_params = 3 * n_bins + 1
        self.register_buffer('is_wrap', torch.tensor(wrap_mask, dtype=torch.bool))
        # bounded dims: an interval [-1, 1] with no tails and no shift (its slot carries the last
        # knot derivative). Not persistent, so existing state dicts load unchanged.
        bmask = [False] * d if bounded_mask is None else [bool(b) for b in bounded_mask]
        self.has_bounded = any(bmask)
        self.register_buffer('is_bounded', torch.tensor(bmask, dtype=torch.bool), persistent=False)
        self.register_buffer('bound',
                             torch.tensor([1.0 if (w or bb) else bound for w, bb in zip(wrap_mask, bmask)]))
        # per-dim floor on the bin heights ([d] tensor on the block's device), None = _MIN_BIN; set by the caller,
        # not part of the state dict
        self.min_height = None
        # knot derivatives derived from the bin slopes (_tied_derivatives) instead of free; set by the caller, not
        # part of the state dict
        self.tied = False
        self.made = _MADE(order, wrap_mask, hidden, n_bins)

        # gather plan for the conditioner features: column f reads channel
        # `feat_kind[f]` (0 raw, 1 sin, 2 cos) of dim `feat_dim[f]`
        fd, fk = [], []
        for i in range(d):
            if wrap_mask[i]:
                fd += [i, i]
                fk += [1, 2]
            else:
                fd += [i]
                fk += [0]
        self.register_buffer('feat_dim', torch.tensor(fd, dtype=torch.long))
        self.register_buffer('feat_kind', torch.tensor(fk, dtype=torch.long))

    def unit_base_rows(self):
        """[d] mask of the rows whose base is uniform on [-1, 1] (wrapped and bounded)."""
        return self.is_wrap | self.is_bounded

    def _feats(self, x):
        ang = math.pi * x
        trip = torch.stack((x, torch.sin(ang), torch.cos(ang)), dim=-1)   # [B, d, 3]
        return trip[:, self.feat_dim, self.feat_kind]

    def forward(self, x):
        b = x.shape[0]
        p = self.made(self._feats(x)).view(b, self.d, self.n_params)
        shift = p[..., -1]
        sp = p[..., :3 * self.n_bins].view(b, self.d, 3, self.n_bins)

        # conditional rotation on the wrapped dims (|detJ| = 1, breaks the seam's
        # fixed point); plain translation on the linear dims; none on bounded dims
        if self.has_bounded:
            z = x + torch.where(self.is_bounded, torch.zeros_like(shift), shift)
            z = torch.where(self.is_wrap, _wrap(z), z)
            y, ld = _rq_spline(z, sp, self.bound, self.is_wrap, self.is_bounded, shift, self.min_height,
                               getattr(self, 'tied', False))
            return y, ld.sum(-1)
        z = x + shift
        z = torch.where(self.is_wrap, _wrap(z), z)
        y, ld = _rq_spline(z, sp, self.bound, self.is_wrap, min_height=self.min_height, tied=getattr(self, 'tied', False))
        return y, ld.sum(-1)


class _Flow(nn.Module):
    def __init__(self, wrap_mask, n_blocks=6, n_bins=10, hidden=(384, 384),
                 bound=5.0, seed=0, bounded_mask=None):
        super().__init__()
        d = len(wrap_mask)
        self.d, self.bound = d, bound
        self.register_buffer('is_wrap', torch.tensor(wrap_mask, dtype=torch.bool))
        self.register_buffer('mu', torch.zeros(d))
        self.register_buffer('sd', torch.ones(d))
        # bounded dims (see _Block) are not standardised and take a uniform base, as wrapped dims do
        bmask = [False] * d if bounded_mask is None else [bool(b) for b in bounded_mask]
        self.bounded_mask = bmask
        self.register_buffer('is_bounded', torch.tensor(bmask, dtype=torch.bool), persistent=False)
        self.register_buffer('unit_base', torch.tensor([bool(w) or b for w, b in zip(wrap_mask, bmask)],
                                                       dtype=torch.bool), persistent=False)

        g = torch.Generator().manual_seed(seed)
        orders = []
        for b in range(n_blocks):
            if b % 3 == 0:
                orders.append(list(range(d)))
            elif b % 3 == 1:
                orders.append(list(range(d))[::-1])
            else:
                orders.append(torch.randperm(d, generator=g).tolist())
        self.orders = orders  # read by callers that swap a block's conditioner (inn_modeller --conditioner)
        self.blocks = nn.ModuleList(
            _Block(o, wrap_mask, n_bins, hidden, bound, bmask) for o in orders)

    def set_scaler(self, x):
        """Standardise the LINEAR dims only; wrapped dims must stay on the circle, and
        bounded dims on their interval."""
        mu = x.mean(0)
        sd = x.std(0).clamp_min(1e-4)
        mu = torch.where(self.unit_base, torch.zeros_like(mu), mu)
        sd = torch.where(self.unit_base, torch.ones_like(sd), sd)
        self.mu.copy_(mu)
        self.sd.copy_(sd)

    def log_prob(self, x):
        """log p in INTERNAL coordinates: wrapped dims on [-1,1], linear standardised."""
        z = torch.where(self.is_wrap, _wrap(x), (x - self.mu) / self.sd)
        # the fixed per-dim standardisation of the linear dims is itself a change
        # of variable and owes a (constant) log-jacobian
        ld = x.new_zeros(x.shape[0]) - torch.log(self.sd[~self.is_wrap]).sum()
        for blk in self.blocks:
            z, l = blk(z)
            ld = ld + l
        base = x.new_zeros(x.shape[0])
        if self.unit_base.any():
            base = base - math.log(2.0) * int(self.unit_base.sum())
        if (~self.unit_base).any():
            zl = z[:, ~self.unit_base]
            base = base - 0.5 * (zl * zl).sum(-1) \
                - 0.5 * math.log(2 * math.pi) * zl.shape[1]
        return base + ld




# ---------------------------------------------------------------- inverse

def _spline_knots(params, bound, circular, bounded=None, end_raw=None, min_height=None, tied=False):
    n_bins = params.shape[-1]
    b = bound.view(1, -1, 1)
    wh = _bin_fractions(params, n_bins, min_height)
    cum = _cumsum_last(wh); cum = cum / cum[..., -1:]
    cum = F.pad(cum, (1, 0))
    cum = (2.0 * b.unsqueeze(-1)) * cum - b.unsqueeze(-1)
    lens = cum[..., 1:] - cum[..., :-1]
    cumw, cumh = cum[..., 0, :], cum[..., 1, :]
    w, h = lens[..., 0, :], lens[..., 1, :]
    d = (_tied_derivatives(params, h / w, circular, bounded, end_raw, 2.0 if tied is True else float(tied)) if tied
         else _knot_derivatives(params, circular, bounded, end_raw))
    return cumw, cumh, w, h, d, n_bins

def rq_spline_inverse(y, params, bound, circular, bounded=None, end_raw=None, min_height=None, tied=False):
    cumw, cumh, w, h, d, n_bins = _spline_knots(params, bound, circular, bounded, end_raw, min_height, tied)
    bb = bound.view(1, -1)
    yc = torch.maximum(torch.minimum(y, bb), -bb)
    idx = torch.searchsorted(cumh.detach().contiguous(),
                             yc.unsqueeze(-1).contiguous(), right=True)
    idx = (idx - 1).clamp_(0, n_bins - 1)
    stacked = torch.stack((cumw[..., :-1], w, cumh[..., :-1], h,
                           d[..., :-1], d[..., 1:]), dim=-2)
    sel = stacked.gather(-1, idx.unsqueeze(-2).expand(idx.shape[:-1] + (6, 1)))
    xk, wk, yk, hk, dk, dk1 = sel.squeeze(-1).unbind(-1)
    s = hk / wk
    Y = yc - yk
    D = dk + dk1 - 2.0 * s
    a = hk * (s - dk) + Y * D
    b_ = hk * dk - Y * D
    c = -s * Y
    disc = (b_ * b_ - 4.0 * a * c).clamp_min(0.0)
    xi = (2.0 * c / (-b_ - torch.sqrt(disc))).clamp(0.0, 1.0)
    x = xk + xi * wk
    outside = (~circular).view(1, -1) & (y != yc)
    if bounded is not None:  # a bounded dim's base point is inside its interval by construction
        outside = outside & ~bounded.view(1, -1)
    return torch.where(outside, y, x)

@torch.no_grad()
def invert_block(blk, y, d):
    """x such that blk.forward(x) == y.

    The block order is not stored, and does not need to be: for an
    autoregressive map, iterating the update d times fixes one more coordinate
    per pass whatever the ordering, so after d passes every dim is exact.
    """
    x = torch.zeros_like(y)
    bnd = blk.is_bounded if getattr(blk, 'has_bounded', False) else None
    mh = getattr(blk, 'min_height', None)
    td = getattr(blk, 'tied', False)
    for _ in range(d):
        p = blk.made(blk._feats(x)).view(y.shape[0], d, blk.n_params)
        shift = p[..., -1]
        sp = p[..., :3 * blk.n_bins].view(y.shape[0], d, 3, blk.n_bins)
        if bnd is None:
            z = rq_spline_inverse(y, sp, blk.bound, blk.is_wrap, min_height=mh, tied=td)
        else:
            z = rq_spline_inverse(y, sp, blk.bound, blk.is_wrap, bnd, shift, mh, td)
            shift = torch.where(bnd, torch.zeros_like(shift), shift)
        x = torch.where(blk.is_wrap, _wrap(z - shift), z - shift)
    return x


@torch.no_grad()
def sample(flow, n, device, seed=0):
    d = flow.d
    g = torch.Generator(device='cpu').manual_seed(seed)
    zn = torch.randn(n, d, generator=g)
    zu = 2.0 * torch.rand(n, d, generator=g) - 1.0
    unit = getattr(flow, 'unit_base', flow.is_wrap)
    z = torch.where(unit.cpu(), zu, zn).to(device)
    for blk in reversed(list(flow.blocks)):
        z = invert_block(blk, z, d)
    return torch.where(flow.is_wrap, _wrap(z), z * flow.sd + flow.mu)


# --------------------------------------------------------------- artifact

import hashlib   # noqa: E402


def _digest(t):
    return hashlib.sha256(t.detach().to('cpu', torch.float32).contiguous()
                          .numpy().tobytes()).hexdigest()


class PriorFlow:
    """Fitted density proxy for one policy at one trajectory length.

    The geometry (wrap mask, dead rows) and the trajectory length T are stored
    and re-checked against the live policy. Neither is recoverable from the
    weights, and a mismatch in either is SILENT: the flow would keep returning
    finite, plausible log-densities for a distribution the policy no longer has.
    """

    def __init__(self, n_blocks=4, n_bins=8, hidden=(256, 256), batch=8192,
                 steps=12000, lr=2e-3, val_frac=0.05, seed=0, time_budget=200.0,
                 device=None, verbose=False):
        self.cfg = dict(n_blocks=n_blocks, n_bins=n_bins, hidden=tuple(hidden),
                        batch=batch, steps=steps, lr=lr, val_frac=val_frac,
                        seed=seed, time_budget=time_budget, verbose=verbose)
        self.device = torch.device(device) if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.flow = None
        self.period = 2.0
        self.wrap_mask = None
        self.dead_rows = ()
        self.traj_T = None
        self.provenance = {}
        self.val_logl = None
        self.fit_seconds = None

    # -- fit ---------------------------------------------------------------
    def fit(self, samples, wrap_mask, period=2.0):
        cfg = self.cfg
        t0 = time.time()
        torch.manual_seed(cfg['seed'])
        self.period = float(period)
        wrap_mask = [bool(w) for w in wrap_mask]
        self.wrap_mask = wrap_mask

        x = torch.as_tensor(samples, dtype=torch.float32).to(self.device)
        x = self._to_internal(x)

        n = x.shape[0]
        g = torch.Generator(device='cpu').manual_seed(cfg['seed'] + 1)
        perm = torch.randperm(n, generator=g).to(self.device)
        n_val = max(1, min(20000, int(cfg['val_frac'] * n)))
        val, tr = x[perm[:n_val]], x[perm[n_val:]]

        flow = _Flow(wrap_mask, n_blocks=cfg['n_blocks'], n_bins=cfg['n_bins'],
                     hidden=cfg['hidden'], seed=cfg['seed']).to(self.device)
        flow.set_scaler(tr)
        self.flow = flow

        opt = torch.optim.Adam(flow.parameters(), lr=cfg['lr'])
        ntr = tr.shape[0]
        gb = torch.Generator(device=self.device).manual_seed(cfg['seed'] + 2)
        bs = min(cfg['batch'], ntr)

        def one_step():
            idx = torch.randint(0, ntr, (bs,), generator=gb, device=self.device)
            l = -flow.log_prob(tr[idx]).mean()
            opt.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            opt.step()
            return l

        # SIZE THE RUN TO THE MACHINE, not to a step count. This box shares one
        # GPU with whatever else is running, and measured step time varied 2x
        # between identical configs; a fixed step count either overshoots the
        # budget or truncates the cosine schedule with the LR still hot, which
        # leaves the flow underfitted for a reason that has nothing to do with
        # the model. So: time a short warm-up, then build the schedule to fit.
        n_probe = 25
        for _ in range(n_probe):
            one_step()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        per_step = max((time.time() - t0) / n_probe, 1e-4)
        steps = int(max(600, min(cfg['steps'], (cfg['time_budget'] - (time.time() - t0)) / per_step)))
        if cfg['verbose']:
            print(f'    probe: {per_step*1000:.0f} ms/step -> {steps} steps', flush=True)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                           eta_min=cfg['lr'] * 0.02)
        best, best_state = -float('inf'), None
        every = max(100, steps // 12)
        for step in range(steps):
            loss = one_step()
            sched.step()
            last = (step == steps - 1)
            if (step + 1) % every == 0 or last:
                v = self._val_logl(flow, val)
                if v > best:
                    best = v
                    best_state = {k: t.detach().clone() for k, t in flow.state_dict().items()}
                if cfg['verbose']:
                    print(f'    step {step+1:5d}  train {-float(loss.detach()):8.3f}  '
                          f'val {v:8.3f}  {time.time()-t0:5.1f}s', flush=True)

        if best_state is not None:
            flow.load_state_dict(best_state)
        flow.eval()
        for p in flow.parameters():
            p.requires_grad_(False)
        self.val_logl = float(best) + self._jac_const()
        self.fit_seconds = time.time() - t0
        return self

    # -- energy ------------------------------------------------------------
    def energy(self, x):
        """-log p(x) + const. Deterministic; a single masked forward pass.

        THE GRAPH IS BUILT ONLY WHEN THE CALLER ASKS FOR IT, i.e. when x carries grad and
        grad mode is on. This used to be an unconditional `with torch.no_grad()`, which made
        d(flow leg)/dx identically zero: under lambda mixing the terminal reward force was
        then entirely lambda * d(physical)/dx at every lambda, with no flow contribution
        even at lambda -> 0 where the flow IS the target (measured 2026-09-21). Every
        scoring path that does not want a graph -- buffer rescoring, eval, admission --
        passes a detached x and is unaffected, so this keeps their cost exactly as it was.
        """
        dev_in = x.device if torch.is_tensor(x) else None
        xt = torch.as_tensor(x, dtype=torch.float32).to(self.device)
        xt = self._to_internal(xt)
        want_grad = torch.is_grad_enabled() and xt.requires_grad
        out = []
        with torch.enable_grad() if want_grad else torch.no_grad():
            for i in range(0, xt.shape[0], 65536):
                out.append(self.flow.log_prob(xt[i:i + 65536]))
        lp = torch.cat(out) + self._jac_const()
        e = -lp
        return e.to(dev_in) if dev_in is not None else e

    def log_prob(self, x):
        return -self.energy(x)

    # -- internals ---------------------------------------------------------
    def _to_internal(self, x):
        """Rescale wrapped dims from period `p` onto [-1, 1] and fold them."""
        if self.period == 2.0:
            return x
        m = torch.tensor(self.wrap_mask, device=x.device)
        return torch.where(m, _wrap(x * (2.0 / self.period)), x)

    def _jac_const(self):
        """d(internal)/d(x) for the wrapped rescale. Zero at the shipped period."""
        if self.period == 2.0:
            return 0.0
        return sum(self.wrap_mask) * math.log(2.0 / self.period)

    @staticmethod
    def _val_logl(flow, val):
        with torch.no_grad():
            tot, n = 0.0, 0
            for i in range(0, val.shape[0], 65536):
                b = val[i:i + 65536]
                tot += float(flow.log_prob(b).sum())
                n += b.shape[0]
        return tot / n


    # -- artifact ----------------------------------------------------------

    def save(self, path, traj_T, dead_rows=(), provenance=None):
        if self.flow is None:
            raise RuntimeError('nothing fitted yet')
        blob = {
            'state_dict': {k: v.cpu() for k, v in self.flow.state_dict().items()},
            'cfg': self.cfg, 'wrap_mask': list(self.wrap_mask),
            'dead_rows': tuple(int(r) for r in (dead_rows or ())),
            'period': float(self.period), 'traj_T': int(traj_T),
            'data_ndim': len(self.wrap_mask), 'val_logl': self.val_logl,
            'provenance': dict(provenance or {}),
        }
        torch.save(blob, path)
        return blob

    @classmethod
    def load(cls, path, device=None):
        blob = torch.load(path, map_location='cpu', weights_only=False)
        for key in ('state_dict', 'wrap_mask', 'cfg', 'traj_T'):
            if key not in blob:
                raise ValueError(
                    f"{path} is missing '{key}'; it was not written by "
                    f"build_prior_flow.py, or predates its current format")
        obj = cls(device=device, **{k: v for k, v in blob['cfg'].items()
                                    if k in ('n_blocks', 'n_bins', 'hidden', 'batch',
                                             'steps', 'lr', 'val_frac', 'seed',
                                             'time_budget', 'verbose')})
        obj.wrap_mask = [bool(w) for w in blob['wrap_mask']]
        obj.dead_rows = tuple(blob.get('dead_rows', ()))
        obj.period = float(blob.get('period', 2.0))
        obj.traj_T = int(blob['traj_T'])
        obj.provenance = blob.get('provenance', {})
        obj.val_logl = blob.get('val_logl')
        flow = _Flow(obj.wrap_mask, n_blocks=obj.cfg['n_blocks'], n_bins=obj.cfg['n_bins'],
                     hidden=obj.cfg['hidden'], seed=obj.cfg['seed']).to(obj.device)
        flow.load_state_dict({k: v.to(obj.device) for k, v in blob['state_dict'].items()})
        flow.eval()
        for p in flow.parameters():
            p.requires_grad_(False)
        obj.flow = flow
        return obj

    def verify_against_policy(self, ang_mask, dead_rows=(), traj_T=None):
        """Raise unless the live policy matches what this flow was fitted to."""
        theirs = torch.as_tensor(ang_mask, dtype=torch.bool).cpu()
        ours = torch.as_tensor(self.wrap_mask, dtype=torch.bool)
        if theirs.numel() != ours.numel() or bool((theirs != ours).any()):
            raise ValueError(
                f"prior_flow was fitted with wrapped dims "
                f"{sorted(torch.nonzero(ours).flatten().tolist())} but the policy wraps "
                f"{sorted(torch.nonzero(theirs).flatten().tolist())}; the flow does not "
                f"describe this problem's latent space")
        their_dead = tuple(sorted(set(int(r) for r in (dead_rows or ()))))
        if their_dead != self.dead_rows:
            raise ValueError(
                f"prior_flow was fitted with dead rows {self.dead_rows} but the policy "
                f"pins {their_dead}")
        if traj_T is not None and int(traj_T) != int(self.traj_T):
            raise ValueError(
                f"prior_flow was fitted to draws at T={self.traj_T} but this run "
                f"integrates T={int(traj_T)}. The terminal distribution depends on T, "
                f"so this flow describes a distribution the policy no longer has. "
                f"T is absent from problem_def, so nothing else would catch this -- "
                f"refit with build_prior_flow.py at the new T.")

    def describe(self):
        wrapped = [i for i, w in enumerate(self.wrap_mask or []) if w]
        return (f"PriorFlow(d={len(self.wrap_mask or [])}, T={self.traj_T}, "
                f"blocks={self.cfg['n_blocks']}x{self.cfg['hidden'][0]}/{self.cfg['n_bins']}b, "
                f"wrapped={wrapped}, dead={self.dead_rows or '()'}, "
                f"val_logl={self.val_logl:.3f})" if self.val_logl is not None else "PriorFlow(unfitted)")

    def draw(self, n, seed=0):
        """Samples from the fitted density. Diagnostics only -- never needed by the energy."""
        return sample(self.flow, n, self.device, seed=seed)
