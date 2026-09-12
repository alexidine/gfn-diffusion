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

from models.gfn import GFN, logtwopi


class ConformerGFN(GFN):
    """`GFN` that can hand the forward policy the molecule it is sampling for."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mol_cond: Optional[dict] = None
        self._state_mask: Optional[torch.Tensor] = None
        #: set by the modeller when the energy is a CARRIER state; a batch without
        #: `state_mask` is then refused rather than scored as though every column were real
        self._carrier = False

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
        self._bind_state_mask(mol_batch)
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
        if getattr(self.forward_policy, 'wants_carrier', False):
            # the ragged policy tokenises VALID columns only. Its index vectors are fixed for
            # the trajectory, so they are computed here once rather than on each of T steps.
            mask = self._state_mask
            if mask is None:
                raise RuntimeError('the carrier policy needs `state_mask` on the batch; '
                                   'rebuild the conditions file with --carrier')
            static = getattr(mol_batch, 'dof_static', None)
            if static is None:
                raise RuntimeError('the carrier policy needs `dof_static` on the batch; '
                                   'rebuild the conditions file with --carrier')
            flat = mask.reshape(-1).nonzero().squeeze(1)
            self._mol_cond.update({
                'state_mask': mask,
                'dof_static': static.reshape(n_graphs, k, -1).to(torch.get_default_dtype()),
                'flat_idx': flat,
                'dof_batch': torch.div(flat, k, rounding_mode='floor'),
            })

    def _bind_state_mask(self, mol_batch) -> None:
        """Per-row carrier validity ``[B, K]``, or None on a non-carrier batch."""
        m = getattr(mol_batch, 'state_mask', None) if mol_batch is not None else None
        if m is None:
            if getattr(self, '_carrier', False):
                raise RuntimeError(
                    'this GFN runs on a CARRIER state but the batch has no `state_mask`; '
                    'every pad column would be scored as a real coordinate')
            self._state_mask = None
            return
        dev = next(self.parameters()).device
        self._state_mask = m.reshape(int(mol_batch.num_graphs), -1).bool().to(dev)

    # ------------------------------------------------------------------ carrier masking
    #
    # A carrier row owns only some of the K columns. The rest are pinned to 0 by _pin_dead
    # and excluded from every log-prob SUM below -- the per-row analogue of the parent's
    # dead-dim handling (D33), which is global over columns and so cannot express it.
    # Each override is the parent's arithmetic with the final reduction masked; with no mask
    # bound they return the parent's result unchanged.

    def _row_mask(self, x):
        m = getattr(self, '_state_mask', None)
        if m is None:
            return None
        if m.shape != x.shape:
            raise RuntimeError(
                f'state_mask {tuple(m.shape)} does not match the state {tuple(x.shape)}; the '
                f'bound batch is not the one being rolled out')
        return m

    def _pin_dead(self, state):
        state = super()._pin_dead(state)
        m = self._row_mask(state)
        # torch.where, not a multiply: a NaN in a pad must become 0, not stay NaN
        return state if m is None else torch.where(m, state, torch.zeros_like(state))

    def gauss_logprob(self, delta_x, drift, var):
        m = self._row_mask(delta_x)
        if m is None:
            return super().gauss_logprob(delta_x, drift, var)
        if self.dead_idx.numel():
            raise NotImplementedError('dead latent rows and a carrier state together')
        z = self._wrap_ang(delta_x - drift)
        per = -0.5 * ((z / var.sqrt()) ** 2 + logtwopi + var.log())
        return torch.where(m, per, torch.zeros_like(per)).sum(1)

    def _pb_logprob(self, prev_state, next_state, drift_coeff, back_mean_correction,
                    back_var, t_next):
        m = self._row_mask(next_state)
        if m is None or not (self.pb_exact_reversal and self.ang_dim > 0):
            # the single-image path goes through gauss_logprob, which is already masked
            return super()._pb_logprob(prev_state, next_state, drift_coeff,
                                       back_mean_correction, back_var, t_next)
        prev_state = self._wrap_ang(prev_state)
        next_state = self._wrap_ang(next_state)
        lin, ang = self.lin_idx, self.ang_idx
        next_lin = next_state.index_select(1, lin)
        back_drift_lin = -next_lin * drift_coeff * back_mean_correction.index_select(1, lin)
        z_lin = (prev_state.index_select(1, lin) - next_lin) - back_drift_lin
        var_lin = back_var.index_select(1, lin)
        per_lin = -0.5 * (z_lin ** 2 / var_lin + logtwopi + var_lin.log())
        per_ang = self._pb_mixture_ang_terms(
            prev_state.index_select(1, ang), next_state.index_select(1, ang),
            drift_coeff, back_mean_correction.index_select(1, ang),
            back_var.index_select(1, ang), t_next)
        m_lin, m_ang = m.index_select(1, lin), m.index_select(1, ang)
        return (torch.where(m_lin, per_lin, torch.zeros_like(per_lin)).sum(1)
                + torch.where(m_ang, per_ang, torch.zeros_like(per_ang)).sum(1))

    def _pb_mixture_ang_terms(self, x, y, drift_coeff, kappa, beta_sq, t_next):
        """``GFN._pb_mixture_ang_logprob`` WITHOUT its final sum over dims: ``[B, ang]``.

        A replica, not a refactor, because models/gfn.py is shared with crystal.
        tests/models/test_carrier_gfn.py asserts ``terms.sum(1)`` equals the parent's value.
        """
        period = 2.0
        lifts = torch.tensor(self.PB_LIFTS, device=x.device, dtype=x.dtype) * period
        y_lifts = y.unsqueeze(-1) + lifts
        v_next = self.accum_var(t_next).clamp(min=self.var_floor)
        log_pi = -0.5 * y_lifts.pow(2) / v_next.view(-1, 1, 1)
        log_pi = log_pi - torch.logsumexp(log_pi, dim=-1, keepdim=True)
        contraction = 1.0 - drift_coeff * kappa
        component_mean = contraction.unsqueeze(-1) * y_lifts
        image_lifts = torch.tensor(self.PB_IMAGE_LIFTS, device=x.device, dtype=x.dtype) * period
        x_lifts = x.unsqueeze(-1).unsqueeze(-1) + image_lifts.view(1, 1, 1, -1)
        z = x_lifts - component_mean.unsqueeze(-1)
        beta = beta_sq.unsqueeze(-1).unsqueeze(-1)
        comp_logp = -0.5 * (z.pow(2) / beta + logtwopi + beta.log())
        wrapped_comp_logp = torch.logsumexp(comp_logp, dim=-1)
        return torch.logsumexp(log_pi + wrapped_comp_logp, dim=-1)

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
