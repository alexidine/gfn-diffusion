"""A molecular graph encoder -- ``E_GNN`` in the conditional stack.

Produces PER-ATOM embeddings ``{g_i}``, not only a pooled vector, because the n-body
correlators that build the per-DoF identities ``f_j`` consume ``g~_i = [g_i, g_global]``.
A model that only emits a molecular latent puts the representational pressure in the wrong
place -- see conformer_conditional_stack.md section 5.

WHAT g_i HAS TO CONTAIN. The 2D graph and the atom's place in it -- formally, a complete
invariant of the chirality-labelled pointed graph (G, i), up to automorphism of that labelled
graph. Two independent obstructions stand between message passing and that target, and they
want different remedies:

  * EXPRESSIVENESS. A k-round MPNN computes exactly the 1-WL colour, which cannot count
    cycles. Fixed by a STRUCTURAL ENCODING computed exactly offline from the adjacency and
    fed in as node features (models/graph_encodings.py). Not by more layers.
  * RADIUS. g_i cannot encode what it never received, so completeness needs depth >= the
    atom's eccentricity. A tetraglycine backbone is ~13 bonds end to end, and message passing
    oversquashes long before 13 rounds. Fixed by ATTENTION, which makes the radius 1.

So the two halves are not competing options; they answer different questions, and this module
can be built with either, both, or neither, which is what makes the battery in
docs/design/encoder_ssl_battery.md a measurement rather than an argument. ``attention=False``
is the aggregate-and-broadcast arm -- rank-one global context, able to express "this molecule
contains an amide" but not "atom i specifically needs to know about atom k".

NO EQUIVARIANT MACHINERY, and none is needed: the input is a 2D graph, the outputs are
scalars, and internal coordinates are themselves invariants, so the whole policy is
SE(3)-invariant by construction. The single asymmetry is chirality, which enters as the
parity pseudoscalar in the atom features -- without it the encoder cannot tell enantiomers
apart, because their 2D graphs are identical. Note that every structural encoding here is a
function of the adjacency alone and is therefore enantiomer-blind too; parity has to be
passed in, it is never recovered.

EVERY POOL IN THIS FILE IS AUGMENTED SOFTMAX -- neighbour aggregation, the global broadcast,
the attention block and the final readout. Owner's rule, and the reason is asymmetric cost:
the augmented form STRICTLY CONTAINS sum (its unnormalised half), mean (uniform logits) and
max (large logit scale), so the right behaviour is LEARNED rather than assumed. Picking wrong
in the intensive direction is a silent bug this codebase has already hit twice -- a mean pool
in the log Z head would discard exactly the size dependence log Z needs, and set_policy.py
shipped mean-and-sum whose selective half was a plain average, invisible on a fixed molecule.
A redundant channel costs a linearly dependent input the next MLP can ignore for free. Only
depart from this where a quantity is KNOWN to be exactly intensive.

AGGREGATION IS AUGMENTED SOFTMAX, matching models/set_policy.py: a softmax-weighted term
that can emphasise, plus an unnormalised sum that is extensive and carries molecular size.
Section 5 makes the same argument for the log Z head, where a mean pool would discard
exactly the size dependence log Z needs.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


def scatter_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """``[E, F] -> [n, F]`` summed by index. Plain torch; no PyG dependency here."""
    out = torch.zeros(n, src.shape[-1], dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


def scatter_softmax(score: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax over each graph's own nodes. ``[N, 1] -> [N, 1]``.

    Max-subtracted per graph, not globally: a global max makes the weights depend on which
    other molecules happen to share the batch, which is a silent batch-composition leak.
    """
    m = torch.full((n, 1), float('-inf'), dtype=score.dtype, device=score.device)
    m = m.index_reduce_(0, index, score, 'amax', include_self=True)
    e = torch.exp(score - m[index])
    denom = scatter_sum(e, index, n)
    return e / denom[index].clamp(min=1e-12)


def to_dense_batch(h: torch.Tensor, batch: torch.Tensor, n_graphs: int):
    """``[N, F], [N] -> ([G, L, F], [G, L] bool)`` with L = the batch's largest graph.

    Written out rather than imported from PyG so this module keeps its plain-torch
    dependency, and so the padding convention is visible at the one place attention masks
    against it.
    """
    counts = torch.bincount(batch, minlength=n_graphs)
    length = int(counts.max().item()) if counts.numel() else 0
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    pos = torch.arange(h.shape[0], device=h.device) - ptr[batch]
    dense = h.new_zeros(n_graphs, length, h.shape[-1])
    mask = torch.zeros(n_graphs, length, dtype=torch.bool, device=h.device)
    dense[batch, pos] = h
    mask[batch, pos] = True
    return dense, mask


def from_dense_batch(dense: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Inverse of :func:`to_dense_batch`'s scatter. ``[G, L, F] -> [N, F]``."""
    counts = torch.bincount(batch, minlength=n_graphs)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    pos = torch.arange(batch.shape[0], device=dense.device) - ptr[batch]
    return dense[batch, pos]


class SPDAttention(nn.Module):
    """Dense self-attention over one graph's atoms, biased by shortest-path distance.

    THE BIAS IS THE POINT, not the attention. Bare attention on a 2D graph is strictly worse
    than message passing, because it discards bond topology entirely -- atoms two bonds apart
    and eight bonds apart become indistinguishable. Graphormer's spatial encoding restores it
    as a learned scalar per SPD bucket added to the logits, so the model is told where it is
    rather than having to infer it.

    Distances at or beyond ``max_spd`` share the last real bucket, and unreachable or padded
    pairs get their own bucket -- distinct from "very far", because a disconnected fragment
    is a different statement from a long chain.
    """

    def __init__(self, hidden: int, n_heads: int = 4, max_spd: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        if hidden % n_heads:
            raise ValueError(f'hidden {hidden} is not divisible by n_heads {n_heads}')
        self.h, self.n_heads, self.dh = hidden, n_heads, hidden // n_heads
        self.max_spd = int(max_spd)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        # SPLIT PROJECTION, and the summed branch is ZERO-INITIALISED. Concatenating the two
        # halves and projecting jointly looks equivalent but is not: the unnormalised sum is
        # L times the magnitude of the convex combination -- MEASURED at 9x on a 9-atom
        # molecule, 31x on Gly4, 63x at L=63 -- so at init the routed signal is swamped and
        # gets a vanishing share of the gradient. Zero-init makes the block start as pure
        # attention and LEARN the extensive channel in, which is the same residual-branch
        # trick spd_bias already uses two lines below.
        self.proj = nn.Linear(hidden, hidden)
        self.proj_sum = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.proj_sum.weight)
        nn.init.zeros_(self.proj_sum.bias)
        self.drop = nn.Dropout(dropout)
        #: buckets 0..max_spd are real distances; max_spd + 1 is unreachable/padding
        self.spd_bias = nn.Embedding(self.max_spd + 2, n_heads)
        nn.init.zeros_(self.spd_bias.weight)

    def bucket(self, spd: torch.Tensor) -> torch.Tensor:
        """``[G, L, L]`` raw distances (negative = unreachable) -> bias indices."""
        far = self.max_spd + 1
        b = spd.clamp(min=0, max=self.max_spd)
        return torch.where(spd < 0, torch.full_like(spd, far), b)

    def forward(self, dense: torch.Tensor, mask: torch.Tensor,
                spd: Optional[torch.Tensor] = None) -> torch.Tensor:
        g, length, _ = dense.shape
        q, k, v = self.qkv(dense).chunk(3, dim=-1)
        shape = (g, length, self.n_heads, self.dh)
        q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))   # [G, H, L, dh]
        logits = (q @ k.transpose(-1, -2)) / (self.dh ** 0.5)           # [G, H, L, L]
        if spd is not None:
            if spd.shape != (g, length, length):
                raise ValueError(
                    f'spd must be [G, L, L] = {(g, length, length)} matching the dense '
                    f'batch, got {tuple(spd.shape)}; build it with dense_spd_batch so the '
                    f'padding convention cannot drift from to_dense_batch')
            logits = logits + self.spd_bias(self.bucket(spd)).permute(0, 3, 1, 2)
        pad = ~mask                                                     # [G, L]
        logits = logits.masked_fill(pad[:, None, None, :], float('-inf'))
        attn = self.drop(torch.softmax(logits, dim=-1))
        # a fully padded row softmaxes to NaN; it is masked out downstream but must not
        # poison the graph through the residual, so zero it here rather than later
        attn = torch.nan_to_num(attn, nan=0.0)
        routed = (attn @ v).transpose(1, 2).reshape(g, length, self.h)
        # AUGMENTED, for the same reason every other pool here is. A softmax is a convex
        # combination and therefore size-blind; on its own it cannot represent an extensive
        # quantity at all. The unnormalised term is the rank-one broadcast, so this block
        # STRICTLY CONTAINS the broadcast arm and attention can only win by using routing.
        summed = (v.transpose(1, 2).reshape(g, length, self.h)
                  * mask.unsqueeze(-1)).sum(dim=1, keepdim=True).expand(-1, length, -1)
        return (self.proj(routed) + self.proj_sum(summed)) * mask.unsqueeze(-1)


def dense_spd_batch(spd_list: Sequence, n_graphs: int, length: int,
                    device=None) -> torch.Tensor:
    """Pack per-graph ``[n_i, n_i]`` distance matrices into ``[G, L, L]``.

    Padding is filled with -1, i.e. the unreachable bucket, so a padded key is never read as
    an atom sitting at distance zero -- which would be the maximally attractive bias.
    """
    out = torch.full((n_graphs, length, length), -1, dtype=torch.long, device=device)
    for i, spd in enumerate(spd_list):
        t = torch.as_tensor(spd, dtype=torch.long, device=device)
        n = t.shape[0]
        if n > length:
            raise ValueError(f'graph {i} has {n} atoms but the dense length is {length}')
        out[i, :n, :n] = t
    return out


class MPNNEncoder(nn.Module):
    """Message passing over a 2D molecular graph, optionally with global attention.

    Returns ``(g_i, g_global)``: per-atom embeddings ``[N, hidden]`` and the augmented-softmax
    pooled vector ``[G, 2 * hidden]``.

    Both arms carry a GLOBAL step per layer: rank-one aggregate-and-broadcast when
    ``attention=False``, SPD-biased pairwise attention when ``attention=True``. That is the
    comparison section 5 actually asks for -- every atom receives the *same* vector in the
    first, and a *routed* one in the second.

    ``struct_dim`` reserves input width for a structural encoding (RWSE or LapPE, from
    models/graph_encodings.py). It is part of ``node_dim`` from the caller's side -- the
    encoding is concatenated onto the atom features before the call -- and is named here only
    so the arm that runs WITHOUT one is expressed by passing zeros of the right width rather
    than by building a differently shaped model. That keeps parameter counts comparable
    across battery arms, so a win cannot be a capacity artefact.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64, layers: int = 4,
                 attention: bool = False, n_heads: int = 4, max_spd: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(layers)])
        #: one score per MESSAGE, softmaxed over each atom's own incoming edges
        self.mscore = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(layers)])
        self.upd = nn.ModuleList([
            nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.attention = bool(attention)
        if self.attention:
            self.attn = nn.ModuleList([
                SPDAttention(hidden, n_heads=n_heads, max_spd=max_spd, dropout=dropout)
                for _ in range(layers)])
            self.attn_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        else:
            # THE AGGREGATE-AND-BROADCAST ARM, and it has to exist as an arm. Section 5's
            # static branch is g~_i = [g_i, g_global]; without this the non-attention model
            # has no global channel AT ALL, and the comparison silently becomes
            # "global vs none" instead of "rank-one vs pairwise". Placed per layer, at the
            # same point attention sits, so the two arms reach global context at equal depth.
            self.bscore = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(layers)])
            self.bcast = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
            #: the extensive half, zero-initialised for the scale reason in SPDAttention
            self.bcast_sum = nn.ModuleList([nn.Linear(hidden, hidden)
                                            for _ in range(layers)])
            for lin in self.bcast_sum:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            self.bcast_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.score = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, x, edge_index, edge_attr, batch, n_graphs, spd=None):
        """``x [N, node_dim]``, ``edge_index [2, E]`` (both directions), ``batch [N]``.

        ``spd`` is ``[G, L, L]`` from :func:`dense_spd_batch`, required only when this was
        built with ``attention=True``; passing it to a non-attention encoder is refused
        rather than ignored, because silently dropping the one input that distinguishes the
        battery's arms is how a comparison comes back null for the wrong reason.
        """
        if spd is not None and not self.attention:
            raise ValueError('spd was supplied but this encoder has attention=False; the '
                             'bias would be silently discarded and the arm would be '
                             'mislabelled')
        h = self.embed(x)
        src, dst = edge_index[0], edge_index[1]
        for i, (msg, upd, nrm) in enumerate(zip(self.msg, self.upd, self.norm)):
            m = msg(torch.cat([h[src], h[dst], edge_attr], dim=-1))
            wm = scatter_softmax(self.mscore[i](m), dst, h.shape[0])
            agg = torch.cat([scatter_sum(wm * m, dst, h.shape[0]),
                             scatter_sum(m, dst, h.shape[0])], dim=-1)
            h = nrm(h + upd(torch.cat([h, agg], dim=-1)))          # residual
            if self.attention:
                dense, mask = to_dense_batch(h, batch, n_graphs)
                a = from_dense_batch(self.attn[i](dense, mask, spd), batch, n_graphs)
                h = self.attn_norm[i](h + a)                       # residual, GPS-shaped
            else:
                w = scatter_softmax(self.bscore[i](h), batch, n_graphs)
                sel = scatter_sum(w * h, batch, n_graphs)[batch]
                ext = scatter_sum(h, batch, n_graphs)[batch]
                h = self.bcast_norm[i](h + self.bcast[i](sel) + self.bcast_sum[i](ext))
        a = scatter_softmax(self.score(h), batch, n_graphs)
        g = torch.cat([scatter_sum(a * h, batch, n_graphs),
                       scatter_sum(h, batch, n_graphs)], dim=-1)   # [n_graphs, 2H]
        return h, g

    @property
    def out_dim(self):
        return self.hidden

    @property
    def global_dim(self):
        return 2 * self.hidden
