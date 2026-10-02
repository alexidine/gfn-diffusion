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

    the `log Z(c)` head can REPLACE its input, opt-in (`install_flow_head`). By default the
    head is `GFN.init_flow_model`'s `scalarMLP(input_dim=condition_embedding_dim)`, read by
    `_condition_flow` off the conditioner's output, detached. That output is shaped by
    whatever trains the conditioner, and on the set route nothing does once P_B is frozen:
    the set P_F never reads `s_emb`, and the frozen P_B snapshot (`GFN._pb_net`) consumes
    the conditioner detached. The head then regresses on a feature fixed at its phase-1
    value, yet `get_tb_loss` uses it as the TB centre for every condition the tracker does
    not yet trust. `install_flow_head('mol_emb')`
    swaps in a head over the pooled baked molecular embedding instead -- a frozen,
    per-molecule feature that does not depend on the conditioner training at all. It is an
    A/B arm, not a default; still ONE head, so the two never compete to normalise the same
    object.

BINDING IS PER TRAJECTORY, NOT PER STEP, and that is the point of precomputing at all: the
molecular embedding is frozen and static, so it is gathered once and read T times.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from mxtaltools.models.modules.components import scalarMLP

from models.gfn import GFN, logtwopi


class MolEmbFlowHead(scalarMLP):
    """``log Z(c)`` over ``[pooled molecular embedding || per-block valid-column counts]``.

    A scalarMLP whose forward REFUSES any other input width. The width check is the whole
    reason for the subclass: `train.py`'s `bootstrap_log_z` (protocol action `bootstrap_z`
    on a conditional run) calls ``flow_model(get_condition_embedding(...))`` directly, and
    this head must not read the conditioner's output as though it were a molecule.
    `ConformerGFN.install_flow_head` refuses the one width at which that check could not
    tell the two apart.
    """

    def __init__(self, in_dim: int, block_onehot: Optional[torch.Tensor], **mlp_kw):
        super().__init__(input_dim=in_dim, output_dim=1, **mlp_kw)
        # NON-persistent: which block each column belongs to is the energy's layout, not a
        # learned weight. install_flow_head rebuilds it on every construction, so the
        # checkpoint carries weights only and a reload cannot overwrite the current layout
        self.register_buffer('block_onehot', block_onehot, persistent=False)

    def forward(self, x, *args, **kwargs):
        if x.dim() != 2 or x.shape[-1] != self.input_dim:
            raise RuntimeError(
                f'the mol_emb log Z head reads [B, {self.input_dim}] -- the pooled molecular '
                f'embedding (+ per-block valid-column counts) that ConformerGFN binds per '
                f'trajectory -- and was called on {tuple(x.shape)}. A caller that passes the '
                f'conditioner output straight to flow_model (train.py bootstrap_log_z, i.e. '
                f'the `bootstrap_z` stage action on a conditional run) is not wired for this '
                f'head: drop that action from the arm, or keep the default head')
        return super().forward(x, *args, **kwargs)


class ConformerGFN(GFN):
    """`GFN` that can hand the forward policy the molecule it is sampling for."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mol_cond: Optional[dict] = None
        self._state_mask: Optional[torch.Tensor] = None
        #: set by the modeller when the energy is a CARRIER state; a batch without
        #: `state_mask` is then refused rather than scored as though every column were real
        self._carrier = False
        # the modeller RE-CLASSES a built GFN rather than constructing one, so this __init__
        # never runs there: every read of these two goes through getattr with this default
        self._flow_head_kind = 'condition'
        self._flow_in: Optional[torch.Tensor] = None

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

    def fwd_gauss_logprob(self, delta_x, drift, d, dt, V=None):
        """The parent's, but DPLR on a carrier is refused rather than scored unmasked.

        With a low-rank factor the parent takes the Woodbury path, which sums over all K
        columns and never calls `gauss_logprob` -- so the per-row mask above never applies,
        and every pad column (pinned to 0, residual ``-drift``) enters log P_F. Finite,
        plausible and wrong. The set policy already refuses DPLR in `predict_next_state`;
        this closes the FLAT policy on a carrier. It keys on the carrier, not on whether this
        batch happens to have pads, so the refusal cannot come and go with the batch.
        """
        if V is not None and (getattr(self, '_carrier', False)
                              or getattr(self, '_state_mask', None) is not None):
            raise NotImplementedError(
                f'dplr_rank {self.dplr_rank} on a CARRIER state: the DPLR (Woodbury) forward '
                f'density sums over every one of the {self.dim} columns and never reaches the '
                f'per-row pad mask, so pad columns would enter log P_F. Set model.dplr_rank: 0 '
                f'for a carrier run')
        return super().fwd_gauss_logprob(delta_x, drift, d, dt, V)

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

    # ------------------------------------------------------------------ log Z(c) head

    FLOW_HEAD_KINDS = ('condition', 'mol_emb')

    @property
    def flow_head_kind(self) -> str:
        """'condition' (the parent's head over the conditioner) or 'mol_emb'."""
        return getattr(self, '_flow_head_kind', 'condition')

    def install_flow_head(self, kind: str = 'condition', mol_dim: Optional[int] = None,
                          column_blocks: Optional[Sequence[int]] = None) -> None:
        """Choose what the ``log Z(c)`` head reads. Call BEFORE the optimizers are built.

        'condition' is the default and a no-op: the parent's head, its weights and its
        reading are untouched.

        'mol_emb' REPLACES ``self.flow_model`` -- the attribute name is kept, so every router
        keyed on ``flow_model.parameters()`` (the lr_flow group, the fused optimizer's flow
        group, the z_calibration sidecars' clip) picks the new head up unchanged -- with a
        `MolEmbFlowHead` over the batch's pooled ``embedding`` (``mol_dim`` wide). With
        ``column_blocks`` (the energy's per-column block code, e.g. ``_free_block``: 0 r,
        1 theta, 2 phi, 3 transverse, 4 bounded double-bond dihedral) it also reads each row's count of VALID columns per block, because
        log Z(c) grows with the number of free coordinates of each kind and a carrier row's
        count is otherwise only implicit in the embedding. Depth, width, norm, dropout and
        activation are copied from the head it replaces, so an A/B differs in the input only.

        The optimizers and the EMA copy hold references to the OLD head's parameters: a
        caller installing after either was built must rebuild the optimizers, and install on
        the EMA model too unless it is deep-copied afterwards.
        """
        if kind not in self.FLOW_HEAD_KINDS:
            raise ValueError(f'flow head kind must be one of {self.FLOW_HEAD_KINDS}, got {kind!r}')
        current = self.flow_head_kind
        if current != 'condition':
            # a second install would re-initialise a head that may already be trained (or
            # loaded), and 'condition' cannot be restored once the parent's head is gone
            raise RuntimeError(f'the {current!r} log Z head is already installed on this model')
        if kind == 'condition':
            return

        if not self.conditional:
            why_not = 'the model is unconditional, so log Z is one number'
        elif self.full_flow:
            why_not = 'full_flow reads flow_model per step over [s_emb, t_emb] (GFN._step_flow)'
        elif not isinstance(self.flow_model, scalarMLP):
            why_not = ('a condition set of one keeps the LearnableScalar head, which '
                       'z_level_fill writes through `.scalar`')
        else:
            why_not = None
        if why_not:
            raise NotImplementedError(f"flow head 'mol_emb' refused: {why_not}")
        if not mol_dim or int(mol_dim) <= 0:
            raise ValueError(f"flow head 'mol_emb' needs the pooled embedding width "
                             f"(embedding_conditioning_dim), got mol_dim={mol_dim!r}")
        onehot = None
        if column_blocks is not None:
            # a numpy array on both energies today (ConformerTorsions, CarrierLayout)
            blocks = torch.as_tensor(column_blocks, dtype=torch.long).reshape(-1).cpu()
            if blocks.numel() != self.dim:
                raise ValueError(f'column_blocks has {blocks.numel()} entries for a '
                                 f'{self.dim}-wide state')
            # sorted, so one count per block code PRESENT, in code order (r, theta, phi,
            # transverse); a code absent from this energy gets no input column
            codes = torch.unique(blocks)
            onehot = (blocks.view(-1, 1) == codes.view(1, -1)).to(torch.get_default_dtype())
        in_dim = int(mol_dim) + (0 if onehot is None else onehot.shape[1])
        if in_dim == self.condition_embedding_dim:
            # the head's width check is what refuses bootstrap_log_z's direct
            # flow_model(condition_embedding) call; at this width it could not tell the
            # conditioner's output from a molecule and would read it silently
            raise ValueError(
                f"flow head 'mol_emb' input width {in_dim} equals condition_embedding_dim; "
                f"a direct flow_model(condition_embedding) call would then be read as a "
                f"molecule without error. Change either width")

        old = self.flow_model
        p = next(old.parameters())
        # scalarMLP's constructor calls torch.manual_seed(seed). Forked so that installing a
        # head after construction does not rewind the caller's CPU random stream to seed 0
        with torch.random.fork_rng(devices=[]):
            head = MolEmbFlowHead(in_dim, onehot, layers=old.n_layers,
                                  filters=old.fc_layers[0].out_features,
                                  activation=old.activation, dropout=old.dropout_p,
                                  norm=old.norm_mode, bias=old.bias,
                                  norm_after_linear=old.norm_after_linear)
        self.flow_model = head.to(device=p.device, dtype=p.dtype)
        self._flow_head_kind = kind
        self._flow_in = None

    def _bind_flow_input(self, mol_batch, condition) -> None:
        """The mol_emb head's per-row input ``[B, in_dim]``, bound once per trajectory.

        DETACHED, for the same invariant as the parent's `_condition_flow`: a Z-side gradient
        reaches flow_model's own parameters and nothing else. The baked embedding carries no
        graph today; the detach keeps that true if the encoder is ever run live.
        """
        if self.flow_head_kind != 'mol_emb':
            self._flow_in = None
            return
        if condition is False:
            # the parent's no-conditioning pass reads log Z at a zero condition; this head
            # is a function of the molecule and has no molecule-free value to report
            raise RuntimeError("the mol_emb log Z head has no value for a no-conditioning "
                               "(condition=False) pass")
        emb = getattr(mol_batch, 'embedding', None) if mol_batch is not None else None
        if emb is None:
            raise RuntimeError(
                "the mol_emb log Z head needs the batch's pooled `embedding`. Bake it with: "
                "python build_conformer_conditions.py --smiles ... --encoder-ckpt <ckpt> "
                "--out <conditions.pt>")
        head = self.flow_model
        p = next(head.parameters())
        n = int(mol_batch.num_graphs)
        x = emb.reshape(n, -1).detach().to(device=p.device, dtype=p.dtype)
        onehot = head.block_onehot
        if onehot is not None:
            m = self._state_mask                 # bound just before, by _bind_state_mask
            valid = (m.to(p.dtype) if m is not None
                     else torch.ones(n, onehot.shape[0], device=p.device, dtype=p.dtype))
            if valid.shape[1] != onehot.shape[0]:
                raise RuntimeError(
                    f'state_mask is {valid.shape[1]} wide but the head was installed for '
                    f'{onehot.shape[0]} columns')
            x = torch.cat([x, valid @ onehot.to(p.dtype)], dim=1)
        if x.shape[1] != head.input_dim:
            raise RuntimeError(
                f'the batch gives the mol_emb head a {x.shape[1]}-wide input; it was '
                f'installed for {head.input_dim}. The conditions file and '
                f'embedding_conditioning_dim disagree')
        self._flow_in = x

    def _condition_flow(self, condition_embedding):
        """The parent's read unless the mol_emb head is installed; then that head's.

        Every caller of this is one of the three `get_traj_*` entry points, and each binds
        `_flow_in` for its own batch first. Under the bwd/replay `scramble_conditions` stage
        the parent's head reads the SCRAMBLED conditioner rows; this one reads the true
        molecule, the same pairing `condition_id` and the tracker keep.

        The binding is CONSUMED by the read. A path that reaches here without rebinding --
        an entry point that skips `_bind_trajectory`, or the parent's `get_traj_*` called
        directly -- would otherwise score its rows against the PREVIOUS batch's molecules,
        and the row check below cannot see that when both batches have the same size, which
        the branch batches usually do. Consumed, it raises instead.
        """
        if self.flow_head_kind != 'mol_emb':
            return super()._condition_flow(condition_embedding)
        x = getattr(self, '_flow_in', None)
        if x is None:
            raise RuntimeError('the mol_emb log Z head was read with no batch bound for this '
                               'trajectory: each get_traj_* entry point binds its own, and a '
                               'binding is read once')
        self._flow_in = None
        if condition_embedding is not None and condition_embedding.shape[0] != x.shape[0]:
            raise RuntimeError(
                f'the bound flow input has {x.shape[0]} rows and this trajectory '
                f'{condition_embedding.shape[0]}; the bound batch is not the one being scored')
        return self.flow_model(x).flatten()

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

    def _bind_trajectory(self, mol_batch, condition) -> None:
        self.bind_molecular_conditioning(mol_batch)
        self._bind_flow_input(mol_batch, condition)

    def get_traj_fwd(self, initial_state, discretizer, exploration_std, condition, mol_batch,
                     *args, **kwargs):
        self._bind_trajectory(mol_batch, condition)
        flow_in = self._flow_in                  # taken now: the parent's read consumes it
        out = super().get_traj_fwd(initial_state, discretizer, exploration_std, condition,
                                   mol_batch, *args, **kwargs)
        if flow_in is not None:
            # the parent stashed the CONDITIONER rows for z_calibration's regression mode,
            # which re-feeds them to flow_model(...) directly; this head reads its own input
            self._z_cal_embedding = flow_in
        return out

    def get_traj_bwd(self, terminal_state, discretizer, condition, mol_batch,
                     *args, **kwargs):
        self._bind_trajectory(mol_batch, condition)
        return super().get_traj_bwd(terminal_state, discretizer, condition, mol_batch,
                                    *args, **kwargs)

    def get_traj_replay(self, trajectory, discretizer, condition, mol_batch,
                        *args, **kwargs):
        self._bind_trajectory(mol_batch, condition)
        return super().get_traj_replay(trajectory, discretizer, condition, mol_batch,
                                       *args, **kwargs)
