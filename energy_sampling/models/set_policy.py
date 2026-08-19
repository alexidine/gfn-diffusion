"""A set-structured forward policy over internal coordinates.

WHAT IT REPLACES. ``PolicyModel`` maps a compressed state embedding to ``[B, K * dim]`` with
a flat MLP, so its width is tied to one molecule and nothing tells it that column 7 is a
ring torsion and column 8 is a methyl spin. This module computes the same output from the
same inputs, but per COORDINATE:

    d_j = rho( phi(q_j) , Agg_k phi(q_k) , t )        q_j = [ f_j , x_j , sin, cos ]

``f_j`` is the static per-coordinate identity. Identity travels in the features, so the model
is permutation equivariant in the coordinate table's storage order -- a spanning-tree
traversal artifact that means nothing physically.

WHERE ``f_j`` COMES FROM IS NOT SETTLED HERE. conformer_conditional_stack.md section 5
derives it as ``f_j = F_tau(g~_i1, ..., g~_in, e_j)`` -- n-body correlators over per-atom GNN
embeddings, one correlator per DoF class, computed ONCE per molecule and cached for the whole
trajectory. This module takes ``f_j`` as a given ``[dim, F]`` array precisely so that the
handcrafted version in energies/dof_features.py and the learned version are interchangeable
inputs. The owner has chosen the learned route; the handcrafted one survives as a control.

NOT WIRED INTO ``GFN`` HERE, DELIBERATELY. ``GFN`` is shared with the crystal workflow,
which is priority 1, and ``_forward_kernel`` reaches the policy with ``s_emb`` -- already
compressed, per-token information gone. Wiring needs ``predict_next_state`` to receive the
raw state as well, which is a small additive change to shared code and belongs in its own
reviewable step rather than buried under a new module.

THE OUTPUT LAYOUT IS LOAD-BEARING AND IS NOT PER-TOKEN. ``split_params`` slices the policy
output into CONTIGUOUS BLOCKS of width ``dim`` -- ``[mean(dim), logvar(dim)]``, or
``[mean, logvar, rho_logit, U(dim*rank)]`` under DPLR. A per-token model naturally produces
``[B, dim, K]``, which is the same numbers in the wrong order; emitting that directly would
feed the mean of coordinate 1 into the log-variance slot of coordinate 0. ``_to_blocks``
does the transpose, and ``test_set_policy.py`` pins it against the real ``split_params``.

AGGREGATION IS AUGMENTED SOFTMAX, per conformer_conditional_stack.md section 5. Two things
have to survive the reduction and no single pool keeps both. A softmax-weighted term is
SELECTIVE -- it can pick out the one strained torsion that should dominate the step -- while
an unnormalised sum is EXTENSIVE and carries "how big is this molecule" at all. A plain mean
supplies neither: it is size-blind by construction, and it cannot emphasise anything.

An earlier version of this module used mean-and-sum, which gets the extensive half right and
replaces the selective half with an average. Kept as a note because the failure is invisible
on a fixed molecule -- both variants train, and the difference only shows up when the model
has to notice that one coordinate matters more than the rest.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from mxtaltools.models.modules.components import scalarMLP


class SetPolicy(nn.Module):
    """Per-coordinate policy. ``forward(state, t_emb) -> [B, out_per_token * dim]``.

    ``static_features`` is ``[dim, F]`` and is registered as a BUFFER, not a parameter: it
    is a property of the molecule, not something to learn, and it must travel with the
    module to whatever device the trajectory runs on.
    """

    def __init__(self,
                 static_features,
                 angular_mask: Sequence[bool],
                 t_dim: int,
                 hidden_dim: int = 64,
                 layers: int = 4,
                 out_per_token: int = 2,
                 dropout: Optional[float] = 0,
                 norm: Optional[str] = None,
                 zero_init: bool = False,
                 device=None):
        super().__init__()
        f = torch.as_tensor(static_features, dtype=torch.get_default_dtype())
        if f.ndim != 2:
            raise ValueError(f'static_features must be [dim, F], got {tuple(f.shape)}')
        if len(angular_mask) != f.shape[0]:
            raise ValueError(f'angular_mask has {len(angular_mask)} entries for '
                             f'{f.shape[0]} coordinates')
        self.dim, self.n_static = int(f.shape[0]), int(f.shape[1])
        self.out_per_token = int(out_per_token)
        self.register_buffer('static', f)
        self.register_buffer('is_ang',
                             torch.as_tensor(list(angular_mask), dtype=f.dtype).unsqueeze(-1))

        # per token: static features, the coordinate's own value, its sin/cos lift, and a
        # flag saying whether that lift means anything
        token_in = self.n_static + 4
        self.phi = scalarMLP(layers=layers, input_dim=token_in, filters=hidden_dim,
                             output_dim=hidden_dim, dropout=dropout, norm=norm)
        # one scalar score per token, softmaxed over the set -> the selective channel
        self.score = nn.Linear(hidden_dim, 1)
        # rho sees its own token, both pooled channels, and time
        self.rho = scalarMLP(layers=layers, input_dim=3 * hidden_dim + t_dim,
                             filters=hidden_dim, output_dim=self.out_per_token,
                             dropout=dropout, norm=norm)
        if zero_init:
            self.rho.output_layer.weight.data.fill_(0.0)
        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------ helpers

    def tokens(self, state: torch.Tensor) -> torch.Tensor:
        """``[B, dim, n_static + 4]``. The sin/cos lift is ZEROED on linear coordinates.

        The state is natively on [-1, 1] representing (-pi, pi] for the angular block, which
        is why the lift multiplies by pi -- the same convention ``expand_state_for_policy``
        uses. Feeding sin/cos of a bond length would be meaningless, so the flag rides along
        and the lift is masked rather than left as noise the model has to learn to ignore.
        """
        b = state.shape[0]
        x = state.unsqueeze(-1)                                  # [B, dim, 1]
        ang = self.is_ang.unsqueeze(0)                           # [1, dim, 1]
        lift = torch.cat([torch.sin(x * torch.pi), torch.cos(x * torch.pi)], dim=-1) * ang
        stat = self.static.unsqueeze(0).expand(b, -1, -1)
        return torch.cat([stat, x, lift, ang.expand(b, -1, -1)], dim=-1)

    def _to_blocks(self, per_token: torch.Tensor) -> torch.Tensor:
        """``[B, dim, K] -> [B, K * dim]`` as CONTIGUOUS blocks, matching split_params."""
        return torch.cat([per_token[..., k] for k in range(self.out_per_token)], dim=-1)

    # ------------------------------------------------------------------ forward

    def forward(self, state: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.dim:
            raise ValueError(f'state has {state.shape[-1]} coordinates, expected {self.dim}')
        h = self.phi(self.tokens(state))                          # [B, dim, H]
        a = torch.softmax(self.score(h), dim=1)                   # [B, dim, 1]
        pooled = torch.cat([(a * h).sum(dim=1), h.sum(dim=1)], dim=-1)   # [B, 2H]
        ctx = torch.cat([pooled, t_emb], dim=-1).unsqueeze(1).expand(-1, self.dim, -1)
        return self._to_blocks(self.rho(torch.cat([h, ctx], dim=-1)))


def set_policy_for(energy, t_dim: int, prior=None, **kw) -> SetPolicy:
    """Build a SetPolicy against the layout the ENERGY declares.

    Mirrors ``train_conformer.build_gfn``'s contract: the angular mask comes from
    ``periodic_dims`` rather than being inferred, because the fallback for an unknown state
    is "nothing is periodic", which is a silently unnormalizable target rather than a merely
    degraded one.
    """
    from energies.dof_features import state_features
    return SetPolicy(state_features(energy, prior), energy.periodic_dims, t_dim, **kw)
