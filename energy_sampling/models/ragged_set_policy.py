"""A set-structured forward policy over internal coordinates, RAGGED across molecules.

PARALLEL TO ``models/set_policy.py``, NOT A REPLACEMENT (owner decision 2026-09-10: keep
crystal on the dense route, subclass or parallel the ragged version for molecules; merging
them is a problem for another day). ``SetPolicy`` is dense ``[B, dim, ...]`` and binds itself
to ONE molecule by registering ``static_features`` as a ``[dim, F]`` buffer. This module
computes the same function on a batch whose rows have DIFFERENT coordinate counts.

THE LAYOUT IS PyG's, one level down. ``MolDataBatch`` already carries atoms as
``[sum_atoms, ...]`` with a ``batch`` index vector; here coordinates are ``[sum_k, ...]`` with
a ``dof_batch`` index vector, and the two compose. Nothing is padded, so no mask can be
dropped somewhere and read as a real coordinate at value 0.

RAGGED DELETES THE OUTPUT-LAYOUT PROBLEM RATHER THAN INHERITING IT. ``set_policy.py`` records
that a per-token model "naturally produces ``[B, dim, K]``, which is the same numbers in the
wrong order", and carries ``_to_blocks`` to re-emit contiguous blocks for
``GFN.split_params``. Here the natural output IS ``[sum_k, out_per_token]`` and the split is a
slice on the last axis -- ``mean = out[:, 0]``, ``logvar = out[:, 1]``. There is no
reordering step and none is needed.

WHAT MOVED FROM BUFFERS TO ARGUMENTS. ``static`` (``f_j``) and ``is_ang`` are per-COORDINATE
properties of whichever molecule that row came from, so they arrive with the batch instead of
being registered once at construction. That is the whole reason this module exists: the dense
version's ``self.dim = f.shape[0]`` is what pins a run to a single ``k``.
"""

from typing import Optional

import torch
from torch import nn

from .architectures import scalarMLP


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Softmax within each segment of a ragged batch.

    The per-segment max is subtracted before exponentiating for the usual reason, and it
    matters more here than in the dense case: segments have different lengths, so a shared
    global max would bias short molecules against long ones.
    """
    if scores.ndim != 1:
        raise ValueError(f'scores must be 1-D [sum_k], got {tuple(scores.shape)}')
    m = torch.full((n_seg,), float('-inf'), dtype=scores.dtype, device=scores.device)
    m = m.scatter_reduce(0, index, scores, reduce='amax', include_self=True)
    e = torch.exp(scores - m[index])
    z = torch.zeros(n_seg, dtype=scores.dtype, device=scores.device).index_add_(0, index, e)
    # a segment with no rows would divide by zero; it also cannot contribute, so clamp
    return e / z.clamp_min(torch.finfo(scores.dtype).tiny)[index]


def segment_sum(src: torch.Tensor, index: torch.Tensor, n_seg: int) -> torch.Tensor:
    """``[sum_k, H] -> [n_seg, H]``, summing rows within each segment."""
    out = torch.zeros(n_seg, src.shape[-1], dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


class RaggedSetPolicy(nn.Module):
    """Per-coordinate policy over a ragged batch.

    ``forward(state, dof_batch, static, is_ang, t_emb) -> [sum_k, out_per_token]``

    ``state``     ``[sum_k]``            the raw pre-expansion coordinate values
    ``dof_batch`` ``[sum_k]`` long       which molecule each coordinate belongs to
    ``static``    ``[sum_k, n_static]``  ``f_j``, the per-coordinate identity
    ``is_ang``    ``[sum_k]`` bool/float 1 where the coordinate is angular
    ``t_emb``     ``[B, t_dim]``         one row per molecule in the batch

    ``n_static`` is fixed at construction because it sizes ``phi``'s input layer; the NUMBER
    of coordinates is not, which is the point.
    """

    #: Read by the rollout to decide whether to hand over the raw state -- same contract as
    #: SetPolicy.wants_raw_state, so the crystal route stays invisible to this branch.
    wants_raw_state = True
    #: This policy also needs the ragged index and the per-row molecule features, which a
    #: dense caller has no way to supply. Duck-typed for the same reason.
    wants_ragged_state = True

    def __init__(self,
                 n_static: int,
                 t_dim: int,
                 hidden_dim: int = 64,
                 layers: int = 4,
                 out_per_token: int = 2,
                 dropout: Optional[float] = 0,
                 norm: Optional[str] = None,
                 zero_init: bool = False,
                 device=None):
        super().__init__()
        self.n_static = int(n_static)
        self.out_per_token = int(out_per_token)
        # per token: static features, the coordinate's own value, its sin/cos lift, and a
        # flag saying whether that lift means anything
        token_in = self.n_static + 4
        self.phi = scalarMLP(layers=layers, input_dim=token_in, filters=hidden_dim,
                             output_dim=hidden_dim, dropout=dropout, norm=norm)
        # one scalar score per token, softmaxed WITHIN ITS MOLECULE -> the selective channel
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

    def tokens(self, state, static, is_ang):
        """``[sum_k, n_static + 4]``. The sin/cos lift is ZEROED on linear coordinates.

        The state is natively on [-1, 1] representing (-pi, pi] for the angular block, which
        is why the lift multiplies by pi -- the same convention ``expand_state_for_policy``
        uses. Feeding sin/cos of a bond length would be meaningless, so the flag rides along
        and the lift is masked rather than left as noise the model has to learn to ignore.
        """
        x = state.unsqueeze(-1)                                        # [sum_k, 1]
        ang = is_ang.to(dtype=x.dtype).reshape(-1, 1)                  # [sum_k, 1]
        lift = torch.cat([torch.sin(x * torch.pi), torch.cos(x * torch.pi)], dim=-1) * ang
        return torch.cat([static, x, lift, ang], dim=-1)

    # ------------------------------------------------------------------ forward

    def forward(self, state, dof_batch, static, is_ang, t_emb):
        state = state.reshape(-1)
        dof_batch = dof_batch.reshape(-1).long()
        if static.shape[0] != state.shape[0]:
            raise ValueError(f'static has {static.shape[0]} rows for {state.shape[0]} '
                             f'coordinates')
        if static.shape[-1] != self.n_static:
            raise ValueError(f'static has width {static.shape[-1]}, expected {self.n_static}')
        if dof_batch.shape[0] != state.shape[0]:
            raise ValueError(f'dof_batch has {dof_batch.shape[0]} entries for '
                             f'{state.shape[0]} coordinates')
        n_seg = int(t_emb.shape[0])
        if int(dof_batch.max()) >= n_seg:
            raise ValueError(f'dof_batch indexes molecule {int(dof_batch.max())} but t_emb '
                             f'has {n_seg} rows')

        h = self.phi(self.tokens(state, static, is_ang))                # [sum_k, H]
        a = segment_softmax(self.score(h).reshape(-1), dof_batch, n_seg).unsqueeze(-1)
        pooled = torch.cat([segment_sum(a * h, dof_batch, n_seg),
                            segment_sum(h, dof_batch, n_seg)], dim=-1)  # [B, 2H]
        ctx = torch.cat([pooled, t_emb], dim=-1)[dof_batch]             # [sum_k, 2H + t_dim]
        return self.rho(torch.cat([h, ctx], dim=-1))                    # [sum_k, K]


# ------------------------------------------------------------------ conditional, on a carrier

class RaggedConditionalSetPolicy(RaggedSetPolicy):
    """`ConditionalSetPolicy`'s function over the VALID columns of a carrier state.

    The rollout keeps a dense ``[B, K]`` carrier (energies/conformer_carrier.py), so the GFN's
    contiguous-block ``split_params`` is unchanged. Inside the policy the valid tokens are
    gathered flat -- ``flat_idx`` / ``dof_batch``, bound once per trajectory by
    ``ConformerGFN.bind_molecular_conditioning`` -- run through the ragged set machinery, and
    scattered back to ``[B, 2K]`` blocks. PAD columns get mean 0 and logvar 0: they never enter
    a pooled channel, and their output is discarded by ``_pin_dead`` and the masked log-probs.

    Differences from the unconditional ragged parent, each mirroring ConditionalSetPolicy:
    ``f_j`` gains a learned part from ``DoFCorrelator`` over per-atom embeddings, and the pooled
    molecular embedding joins ``rho``'s context.
    """

    wants_molecular_conditioning = True
    wants_carrier = True

    def __init__(self, n_static: int, angular_mask, t_dim: int, enc_dim: int, mol_dim: int,
                 corr_dim: int = 32, frame_size: int = 4, hidden_dim: int = 64,
                 layers: int = 4, out_per_token: int = 2, dropout: Optional[float] = 0,
                 norm: Optional[str] = None, zero_init: bool = False, device=None):
        super().__init__(n_static, t_dim, hidden_dim=hidden_dim, layers=layers,
                         out_per_token=out_per_token, dropout=dropout, norm=norm,
                         zero_init=False)
        from .set_policy import DoFCorrelator
        self.dim = len(angular_mask)
        self.register_buffer('is_ang', torch.as_tensor(list(angular_mask), dtype=torch.bool))
        self.enc_dim, self.mol_dim, self.corr_dim = int(enc_dim), int(mol_dim), int(corr_dim)
        self.correlator = DoFCorrelator(enc_dim, corr_dim, frame_size=frame_size,
                                        hidden=hidden_dim, dropout=dropout, norm=norm)
        self.phi = scalarMLP(layers=layers, input_dim=self.n_static + 4 + self.corr_dim,
                             filters=hidden_dim, output_dim=hidden_dim, dropout=dropout,
                             norm=norm)
        self.rho = scalarMLP(layers=layers, input_dim=3 * hidden_dim + t_dim + self.mol_dim,
                             filters=hidden_dim, output_dim=self.out_per_token,
                             dropout=dropout, norm=norm)
        if zero_init:
            self.rho.output_layer.weight.data.fill_(0.0)
        if device is not None:
            self.to(device)

    def forward(self, state, t_emb, atom_emb=None, dof_atoms=None, dof_mask=None,
                mol_emb=None, state_mask=None, dof_static=None, flat_idx=None,
                dof_batch=None):
        if any(v is None for v in (atom_emb, dof_atoms, mol_emb, flat_idx, dof_batch,
                                   dof_static)):
            raise ValueError('RaggedConditionalSetPolicy needs the carrier bindings '
                             '(atom_emb, dof_atoms, mol_emb, dof_static, flat_idx, dof_batch)')
        B, K = state.shape
        if K != self.dim:
            raise ValueError(f'state has {K} carrier columns, expected {self.dim}')
        s = state.reshape(-1).index_select(0, flat_idx)
        st = dof_static.reshape(B * K, -1).index_select(0, flat_idx)
        ang = self.is_ang.repeat(B).index_select(0, flat_idx)
        f = self.correlator(atom_emb,
                            dof_atoms.reshape(B * K, *dof_atoms.shape[-2:]).index_select(0, flat_idx),
                            dof_mask.reshape(B * K, -1).index_select(0, flat_idx).bool())
        h = self.phi(torch.cat([self.tokens(s, st, ang), f], dim=-1))            # [n, H]
        a = segment_softmax(self.score(h).reshape(-1), dof_batch, B).unsqueeze(-1)
        pooled = torch.cat([segment_sum(a * h, dof_batch, B),
                            segment_sum(h, dof_batch, B)], dim=-1)               # [B, 2H]
        ctx = torch.cat([pooled, t_emb, mol_emb], dim=-1).index_select(0, dof_batch)
        out = self.rho(torch.cat([h, ctx], dim=-1))                              # [n, P]
        full = out.new_zeros(B * K, self.out_per_token).index_copy(0, flat_idx, out)
        full = full.reshape(B, K, self.out_per_token)
        return torch.cat([full[..., p] for p in range(self.out_per_token)], dim=-1)
