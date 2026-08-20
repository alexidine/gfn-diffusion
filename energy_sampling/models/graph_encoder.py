"""A minimal molecular graph encoder -- ``E_GNN`` in the conditional stack.

Produces PER-ATOM embeddings ``{g_i}``, not only a pooled vector, because the n-body
correlators that build the per-DoF identities ``f_j`` consume ``g~_i = [g_i, g_global]``.
A model that only emits a molecular latent puts the representational pressure in the wrong
place -- see conformer_conditional_stack.md section 5.

WHAT THIS IS AND IS NOT. This is the local message-passing half of the hybrid that section 5
argues for; the global attention half is deliberately absent. That section's own
discriminator says why the choice has to be measured rather than assumed: aggregate-and-
broadcast is RANK-ONE global context -- every atom receives the same vector, so it can
express "this molecule contains an amide" but not "atom i specifically needs to know about
atom k". Attention is pairwise routing and can express the second. Starting here means the
attention variant has something to beat.

NO EQUIVARIANT MACHINERY, and none is needed: the input is a 2D graph, the outputs are
scalars, and internal coordinates are themselves invariants, so the whole policy is
SE(3)-invariant by construction. The single asymmetry is chirality, which enters as the
parity pseudoscalar in the atom features -- without it the encoder cannot tell enantiomers
apart, because their 2D graphs are identical.

AGGREGATION IS AUGMENTED SOFTMAX, matching models/set_policy.py: a softmax-weighted term
that can emphasise, plus an unnormalised sum that is extensive and carries molecular size.
Section 5 makes the same argument for the log Z head, where a mean pool would discard
exactly the size dependence log Z needs.
"""
from __future__ import annotations

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


class MPNNEncoder(nn.Module):
    """Message passing over a 2D molecular graph. Returns ``(g_i, g_global)``."""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64, layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(layers)])
        self.upd = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.score = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, x, edge_index, edge_attr, batch, n_graphs):
        """``x [N, node_dim]``, ``edge_index [2, E]`` (both directions), ``batch [N]``."""
        h = self.embed(x)
        src, dst = edge_index[0], edge_index[1]
        for msg, upd, nrm in zip(self.msg, self.upd, self.norm):
            m = msg(torch.cat([h[src], h[dst], edge_attr], dim=-1))
            agg = scatter_sum(m, dst, h.shape[0])
            h = nrm(h + upd(torch.cat([h, agg], dim=-1)))          # residual
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
