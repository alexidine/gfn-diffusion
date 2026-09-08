"""The GFN specialisation for the conditional conformer route.

WHY A SUBCLASS. `models/gfn.py` is shared with the crystal workflow, and the crystal route is
priority 1. Everything here is conformer-specific, so it lives outside that file rather than
adding conformer branches to code the crystal path also runs.

WHAT IT ADDS, and it is deliberately small:

    the policy call carries the MOLECULE. `_forward_kernel` sees only
    `(state, t, condition_embedding, t_next, dts)` -- no `mol_batch` -- so a policy whose
    per-coordinate tokens are derived from per-atom embeddings has no way to reach them. The
    three `get_traj_*` entry points DO receive `mol_batch`, so this binds the tensors there,
    once per trajectory, and `predict_next_state` reads them on every step.

WHAT IT DOES NOT ADD, because it already exists:

    the `log Z(c)` head. conformer_conditional_stack.md section 5 asks for
    `log Z(c) = Z_MLP(Agg_i g_i)`; on the conditional route `GFN.init_flow_model` already
    builds `scalarMLP(input_dim=condition_embedding_dim, output_dim=1)` and `_condition_flow`
    already reads it. Since the condition vector now carries the pooled molecular embedding --
    which IS `[sum softmax(s_i) h_i || sum h_i]`, the augmented aggregation over per-atom
    embeddings the design asks for -- that head is `Z_MLP(Agg_i g_i)` as written. Adding a
    second one would be two heads competing to normalise the same object.

BINDING IS PER TRAJECTORY, NOT PER STEP, and that is the point of precomputing at all: the
molecular embedding is frozen and static, so it is gathered once and read T times.
"""
from __future__ import annotations

from typing import Optional

import torch

from models.gfn import GFN


class ConformerGFN(GFN):
    """`GFN` that can hand the forward policy the molecule it is sampling for."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mol_cond: Optional[dict] = None

    # ------------------------------------------------------------------ binding

    @property
    def _policy_wants_molecule(self) -> bool:
        return bool(getattr(self.forward_policy, 'wants_molecular_conditioning', False))

    def bind_molecular_conditioning(self, mol_batch) -> None:
        """Gather the per-sample conditioning tensors off the batch, once.

        `dof_atoms` is stored per graph FLAT and in LOCAL (tree) atom numbering, while
        `atom_embedding` collates to ``[N_total, E]``. So the local
        indices have to be shifted by each graph's atom offset or every graph after the first
        reads graph 0's atoms -- silently, since the indices stay in range. `ptr` is the
        offset table PyG already maintains for exactly this.
        """
        if not self._policy_wants_molecule:
            self._mol_cond = None
            return
        missing = [k for k in ('atom_embedding', 'dof_atoms', 'dof_mask', 'embedding')
                   if getattr(mol_batch, k, None) is None]
        if missing:
            raise RuntimeError(
                f'the forward policy is conditional but this batch is missing {missing}. '
                f'Bake them with: python build_conformer_conditions.py --smiles ... '
                f'--encoder-ckpt <ckpt> --out <conditions.pt>')

        from energies.dof_features import MAX_FRAME
        from energies.conformer_data import state_dim

        n_graphs = int(mol_batch.num_graphs)
        atom_emb = mol_batch.atom_embedding
        k = int(state_dim(mol_batch))
        # stored FLAT as [n_graphs, k*R*MAX_FRAME] -- see the builder on why. R is recovered
        # rather than stored: it is a property of the FILE (the widest collective column in
        # it), so carrying it separately would be a second source of truth that can disagree.
        atoms = mol_batch.dof_atoms.reshape(n_graphs, -1)
        mask = mol_batch.dof_mask.reshape(n_graphs, -1)
        per = atoms.shape[1]
        if per % (k * MAX_FRAME):
            raise RuntimeError(
                f'dof_atoms carries {per} entries per graph, which is not a whole number of '
                f'[k={k} x R x frame={MAX_FRAME}] blocks; the file was written by a different '
                f'builder version')
        R = per // (k * MAX_FRAME)
        atoms = atoms.reshape(n_graphs * k, R, MAX_FRAME)
        mask = mask.reshape(n_graphs * k, R)

        ptr = getattr(mol_batch, 'ptr', None)
        if ptr is None:
            raise RuntimeError('batch has no `ptr`; cannot offset per-graph atom indices')
        offs = ptr[:-1].to(atoms.device).repeat_interleave(k).view(-1, 1, 1)
        atoms = atoms.long()
        self._mol_cond = {
            'atom_emb': atom_emb.to(torch.get_default_dtype()),
            'dof_atoms': (atoms + offs).reshape(n_graphs, k, *atoms.shape[-2:]),
            'dof_mask': mask.reshape(n_graphs, k, mask.shape[-1]),
            'mol_emb': mol_batch.embedding.reshape(n_graphs, -1).to(torch.get_default_dtype()),
        }

    # ------------------------------------------------------------------ policy call

    def predict_next_state(self, s_emb, t_emb, state=None):
        """As the parent, but a molecule-conditional policy also receives its molecule."""
        if not self._policy_wants_molecule:
            return super().predict_next_state(s_emb, t_emb, state)
        if state is None:
            raise ValueError(
                'the conditional set policy needs the RAW state; predict_next_state was '
                'called without it. s_emb cannot substitute -- StateEncoding has already '
                'mixed the coordinates the policy tokenises.')
        if self._mol_cond is None:
            raise RuntimeError(
                'molecular conditioning was never bound. Every trajectory entry point calls '
                'bind_molecular_conditioning; reaching here without it means the policy was '
                'driven through a path this subclass does not override.')
        if self.dplr_rank > 0:
            # the parent's reasoning applies unchanged: a set policy emits the low-rank
            # factor rank-major while split_params views it [dim, rank], so u_raw would be
            # silently transposed
            raise NotImplementedError(
                f'a set policy with dplr_rank {self.dplr_rank} would have its low-rank '
                f'factor silently transposed; set dplr_rank: 0 for this run')
        s_new = self.forward_policy(state, t_emb, **self._mol_cond)
        if self.clipping:
            s_new = torch.clip(s_new, -self.gfn_clip, self.gfn_clip)
        return s_new

    # ------------------------------------------------------------------ entry points

    def get_traj_fwd(self, initial_state, discretizer, exploration_std, condition, mol_batch,
                     *args, **kwargs):
        self.bind_molecular_conditioning(mol_batch)
        return super().get_traj_fwd(initial_state, discretizer, exploration_std, condition,
                                    mol_batch, *args, **kwargs)

    def get_traj_bwd(self, terminal_state, discretizer, condition, mol_batch,
                     *args, **kwargs):
        self.bind_molecular_conditioning(mol_batch)
        return super().get_traj_bwd(terminal_state, discretizer, condition, mol_batch,
                                    *args, **kwargs)

    def get_traj_replay(self, trajectory, discretizer, condition, mol_batch,
                        *args, **kwargs):
        self.bind_molecular_conditioning(mol_batch)
        return super().get_traj_replay(trajectory, discretizer, condition, mol_batch,
                                       *args, **kwargs)
