"""State encoding that reads the atoms of the asymmetric unit.

`architectures.StateEncoding` gives the policies [state, condition vector] through an MLP. Here
the condition vector is replaced by the molecule's own atoms as they sit in the crystal the
state builds:

    atoms [B, A, F], elements [B, A], mask [B, A]     per row, padded to the widest molecule
        -> AtomSetPool: every atom reads every other atom of its row, then three pools
        -> MLP([state, pooled, extra])                -> s_emb [B, s_emb_dim]

What the per-atom features ARE is the caller's business (node states of an energy trunk,
positions in the cell, per-atom energies); nothing here knows about crystals. `s_emb` has
`StateEncoding`'s shape and meaning, so the policies that read it are unchanged.

A ROW'S OUTPUT IS ITS OWN. Padding is no key in attention and has no weight in a pool, and
the only normalisation is per token, so a row encodes the same alone, in any batch, and
under any padding width. The crystal condition this replaces lacked that property.
"""
from typing import Optional

import torch
from torch import nn

from mxtaltools.models.modules.components import scalarMLP

#: divides the summed pool, so a molecule of this many atoms sums to about its mean
SUM_SCALE = 16.0


class AtomSetPool(nn.Module):
    """Atoms of one row, all reading one another, pooled to one vector [B, out_dim].

    A token is a linear map of [atom features, element embedding, the row's `context`].
    Each of `blocks` rounds is self-attention over the row's atoms and a two-layer MLP, both
    residual and both behind a per-token LayerNorm. Three pools leave: a mean under learned
    weights (softmax of one score per atom), the plain mean, and the sum over `SUM_SCALE`.

    Parameters
    ----------
    atom_dim : width F of the per-atom features.
    context_dim : width of a per-row vector every token of the row is also given (0: none).
    hidden_dim : token width; must divide by `heads`.
    blocks, heads : rounds of self-attention and heads per round.
    type_dim : width of the element embedding.
    """

    def __init__(self, atom_dim: int, context_dim: int = 0, hidden_dim: int = 128, blocks: int = 2,
                 heads: int = 4, type_dim: int = 16, dropout: Optional[float] = 0):
        super().__init__()
        self.atom_dim, self.context_dim = int(atom_dim), int(context_dim)
        self.embed = nn.Embedding(101, type_dim)
        self.token = nn.Linear(self.atom_dim + type_dim + self.context_dim, hidden_dim)
        self.attn = nn.ModuleList([nn.MultiheadAttention(hidden_dim, heads, dropout=dropout or 0.0,
                                                         batch_first=True) for _ in range(blocks)])
        self.norm_attn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(blocks)])
        self.norm_mlp = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(blocks)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
                                                nn.Linear(2 * hidden_dim, hidden_dim)) for _ in range(blocks)])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        self.out_dim = 3 * hidden_dim

    def forward(self, atoms: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """atoms [B, A, F]; z [B, A] atomic numbers; mask [B, A], True on real atoms; context [B, C]."""
        if atoms.shape[-1] != self.atom_dim:
            raise ValueError(f'atoms carry {atoms.shape[-1]} features, expected {self.atom_dim}')
        if (context is None) != (self.context_dim == 0):
            raise ValueError(f'this pool reads a context of width {self.context_dim}; `context` is '
                             f'{"absent" if context is None else "given"}')
        if not bool(mask.any(dim=1).all()):
            raise ValueError('a row has no atoms')
        keep = mask.unsqueeze(-1)
        # padding may hold anything, NaN included: it is zeroed here and after every round, because a
        # masked key still multiplies its (zero) attention weight and 0 * NaN is NaN
        parts = [torch.where(keep, atoms, torch.zeros_like(atoms)), self.embed(torch.where(mask, z, 0).long())]
        if context is not None:
            parts.append(context.unsqueeze(1).expand(-1, atoms.shape[1], -1))
        h = torch.where(keep, self.token(torch.cat(parts, dim=-1)), 0.0)
        pad = ~mask
        for attn, norm_attn, norm_mlp, mlp in zip(self.attn, self.norm_attn, self.norm_mlp, self.mlp):
            q = norm_attn(h)
            h = h + attn(q, q, q, key_padding_mask=pad, need_weights=False)[0]
            h = h + mlp(norm_mlp(h))
            h = torch.where(keep, h, 0.0)
        h = torch.where(keep, self.norm_out(h), 0.0)
        weight = torch.softmax(self.score(h).masked_fill(pad.unsqueeze(-1), float('-inf')), dim=1)
        total = h.sum(dim=1)
        count = mask.sum(dim=1, keepdim=True).to(h.dtype)
        return torch.cat([(weight * h).sum(dim=1), total / count, total / SUM_SCALE], dim=-1)


class AtomStateEncoding(nn.Module):
    """`StateEncoding` whose conditioning is the row's atoms: forward -> s_emb [B, s_emb_dim].

    The state goes in twice: to every token, so an atom is read knowing the cell and pose it
    sits in, and to the MLP after the pool, beside `extra` (a per-row vector of width
    `extra_dim`, e.g. a force on the state; 0: none).
    """

    def __init__(self, s_dim: int, atom_dim: int, layers: int, hidden_dim: int = 64, s_emb_dim: int = 64,
                 extra_dim: int = 0, atom_hidden_dim: int = 128, blocks: int = 2, heads: int = 4,
                 dropout: Optional[float] = 0, norm: Optional[str] = None, bias: Optional[bool] = True):
        super().__init__()
        self.extra_dim = int(extra_dim)
        self.pool = AtomSetPool(atom_dim, context_dim=s_dim, hidden_dim=atom_hidden_dim, blocks=blocks,
                                heads=heads, dropout=dropout)
        self.x_model = scalarMLP(
            layers=layers,
            input_dim=s_dim + self.pool.out_dim + self.extra_dim,
            filters=hidden_dim,
            output_dim=s_emb_dim,
            dropout=dropout,
            norm=norm,
            bias=bias,
        )

    def forward(self, s: torch.Tensor, atoms: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
                extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (extra is None) != (self.extra_dim == 0):
            raise ValueError(f'this encoding reads an extra vector of width {self.extra_dim}; `extra` is '
                             f'{"absent" if extra is None else "given"}')
        parts = [s, self.pool(atoms, z, mask, context=s)]
        if extra is not None:
            parts.append(extra)
        return self.x_model(torch.cat(parts, dim=-1))


class FlatAtomStateEncoding(nn.Module):
    """`AtomStateEncoding` behind `StateEncoding`'s call: forward(s, conditioning, extra) -> s_emb.

    `extra` is a state's flattened atom features as a provider hands them to GFN (the layout of
    `models.crystal_force.TrunkForce.state_info`): [B, atoms * (atom_dim + 2)], per atom its
    `atom_dim` features, then its atomic number, then 1 on a real atom and 0 on padding. The
    condition vector, when there is one, joins the state after the pool.
    """

    def __init__(self, s_dim: int, layers: int, hidden_dim: int = 64, conditioning_dim: int = 0,
                 s_emb_dim: int = 64, atoms: int = 1, atom_dim: int = 1, atom_hidden_dim: int = 128,
                 blocks: int = 2, heads: int = 4, dropout: Optional[float] = 0, norm: Optional[str] = None,
                 bias: Optional[bool] = True):
        super().__init__()
        self.atoms, self.atom_dim = int(atoms), int(atom_dim)
        self.conditioning_dim = int(conditioning_dim)
        self.extra_dim = self.atoms * (self.atom_dim + 2)
        self.encoding = AtomStateEncoding(s_dim, self.atom_dim, layers, hidden_dim=hidden_dim, s_emb_dim=s_emb_dim,
                                          extra_dim=self.conditioning_dim, atom_hidden_dim=atom_hidden_dim,
                                          blocks=blocks, heads=heads, dropout=dropout, norm=norm, bias=bias)

    def forward(self, s, conditioning=None, extra=None):
        if extra is None:
            raise ValueError('this state encoder reads atom features and was given none')
        if extra.shape[-1] != self.extra_dim:
            raise ValueError(f'atom features of width {extra.shape[-1]}; this encoder was built for '
                             f'{self.atoms} atoms of {self.atom_dim} features ({self.extra_dim})')
        per_atom = extra.reshape(extra.shape[0], self.atoms, self.atom_dim + 2)
        mask = per_atom[..., -1] > 0.5
        z = per_atom[..., -2].round().long()
        return self.encoding(s, per_atom[..., :self.atom_dim], z, mask,
                             extra=conditioning if self.conditioning_dim > 0 else None)
