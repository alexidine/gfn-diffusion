"""A normalizing flow (NF) as a train.py policy -- here the exact-density autoregressive spline flow -- a one-step sampler with the interface of models/gfn.py::GFN.

A trajectory is one step, s0 -> x. Forward, x is drawn from q(x | c) and log P_F = log q(x | c), exact; backward, the
path x -> s0 is deterministic, log P_B = 0. log_flow[:, 0] is the level head log Z(c); there is no per-step flow. With
these, the trainer's losses (gflownet_losses.py) read a length-1 trajectory and need no change, and the trainer's
loop, stage protocol, buffers, clipping, LR control, evaluation and checkpointing run as they do for the SDE policy.
Plan: docs/design/flow_policy_port.md.

The geometry -- the wrapped (angular) rows, the dead rows and the values they are pinned at -- is read off a GFN built
from the same config, so it is the SDE policy's own. The flow itself is models/spline_flow.py's ConditionalLatentFlow
(conditional runs) or LatentFlow (unconditional), built from the `nf` settings dict (NF_DEFAULTS; mk_dev's model.nf).

Parameter groups, under the names the trainer's optimizers read: forward_policy = the flow's blocks;
conditions_embedding_model = what reads the condition (the context network, and the atom encoder in molecule mode);
flow_model = the level head; t_model, s_model and backward_policy are empty (a one-step flow has no time or state
embedding and a fixed P_B).

Not supported (raise): a pathwise gradient through the draw (detach_traj False, path_grad_last_k > 0), replay-tail
re-propagation (resample_last_k, implied_noise_last_k, stored forces), state forces and features, exploration noise.
"""
import contextlib

import torch
import torch.nn as nn

from models.architectures import LearnableScalar
from models.spline_flow import ConditionalLatentFlow, LatentFlow

# the flow's settings: inn_modeller.py's flags of the same names, with its defaults, so model.kind nf reproduces
# its flow. 'auto' / None are resolved per route as it resolves them (resolve_flow_settings)
NF_DEFAULTS = dict(
    conditioner='transformer',      # 'transformer' (causal transformer) or 'made'
    n_blocks=1, n_bins=32, bound=5.0,
    hidden=512, hidden_layers=2,    # the MADE conditioner's width and depth
    tf_width=256, tf_layers=4, tf_heads=8, tf_norm='layer', tf_fp32=False,
    context_dim=128, context_hidden=512,
    atom_tokens=False, atom_node_dim=128, atom_convs=3, atom_cutoff=10.0, atom_radial=32,
    min_bin_height='auto', min_bin_height_rows='all',
    tied_derivatives=None, tied_factor=None,
    bounded_rows='auto',            # 'auto', 'none'/None, 'linear' (every live unwrapped row on the closed [-1, 1]), or rows '0,3'
    block_orders='auto',            # 'auto', 'none'/None, or one order per block: 'id', 'rev' or rows '8.11.10...'
    fold_half_rows=None,            # rows '7,8' whose distribution repeats every half period
    mix_k=0,                        # > 0: a mixture-of-logistics block of k components (unconditional)
)
# as inn_modeller.py::UNCONDITIONAL_DEFAULTS and ::Z1_ORDER: the unconditional route's defaults for the 'auto' settings
UNCONDITIONAL_DEFAULTS = dict(bounded_rows='linear', min_bin_height=1e-7, tied_derivatives=True, tied_factor=2.0)
Z1_ORDER = '8.11.10.9.7.6.5.4.3.2.1.0'


def resolve_flow_settings(s: dict, conditional: bool, live_rows) -> dict:
    """inn_modeller.py::resolve_auto_options for a fresh build: an unset setting takes the unconditional route's
    default, or off on a conditional policy; block_orders 'auto' is Z1_ORDER on an unconditional one-block flow whose
    live rows are exactly 0-11 (the Z' = 1 crystal latent's own rows), else the alternating orders."""
    s = dict(s)
    for key in ('bounded_rows', 'min_bin_height', 'block_orders'):
        if s[key] == 'none':
            s[key] = None
    if s['bounded_rows'] == 'auto':
        s['bounded_rows'] = None if conditional else UNCONDITIONAL_DEFAULTS['bounded_rows']
    if s['min_bin_height'] == 'auto':
        s['min_bin_height'] = None if conditional else UNCONDITIONAL_DEFAULTS['min_bin_height']
    if s['tied_derivatives'] is None:
        s['tied_derivatives'] = False if conditional else UNCONDITIONAL_DEFAULTS['tied_derivatives']
    if s['tied_factor'] is None:
        s['tied_factor'] = 2.0
    if s['block_orders'] == 'auto':
        z1 = not conditional and int(s['n_blocks']) == 1 and list(live_rows) == list(range(12))
        s['block_orders'] = Z1_ORDER if z1 else None
    return s


def build_policy(gfn_config: dict):
    """The policy a gfn_config describes: models/gfn.py::GFN unless the config carries a `policy` entry whose kind is
    'nf' (train.py writes that entry only for a non-SDE policy, so an SDE gfn_config, its checkpoints and problem
    hashes are unchanged)."""
    cfg = dict(gfn_config)
    pol = cfg.pop('policy', None)
    kind = 'sde' if pol is None else pol.get('kind', 'sde')
    if kind == 'sde':
        from models.gfn import GFN
        return GFN(**cfg)
    if kind == 'nf':
        return NFPolicy(cfg, dict(pol.get('nf') or {}), seed=int(pol.get('seed', 0)))
    raise ValueError(f"model kind {kind!r}: 'sde' or 'nf'")


class _Empty(nn.Module):
    """A parameter group with nothing in it (t_model, s_model, backward_policy of a one-step flow)."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("a one-step flow policy has no time model, state model or backward policy")


class _LevelHead(nn.Module):
    """log Z(c) from the condition features get_condition_embedding returns (the head models/gfn.py calls flow_model:
    it reads a detached copy, so its regression trains the head alone)."""

    def __init__(self, sampler):
        super().__init__()
        self.head = sampler.flow.level

    def forward(self, condition_embedding):
        return self.head(condition_embedding.detach()).squeeze(-1)


class NFPolicy(nn.Module):
    GAUSS_PARAM_KEYS = ('means_f', 'logvars_f', 'diag_logvars_f', 'rho_f', 'means_b', 'logvars_b')

    def __init__(self, gfn_config: dict, flow: dict, seed: int = 0):
        super().__init__()
        from models.gfn import GFN
        unknown = set(flow) - set(NF_DEFAULTS)
        if unknown:
            raise KeyError(f"unknown nf settings {sorted(unknown)}; known: {sorted(NF_DEFAULTS)}")
        s = {**NF_DEFAULTS, **flow}
        probe = GFN(**gfn_config)   # geometry only
        self.device = probe.device
        self.dim = int(probe.dim)
        self.conditional = bool(probe.conditional)
        self.conditions_type = probe.conditions_type
        self.conditions_dim = int(gfn_config.get('conditions_dim') or 0)   # GFN keeps it in its conditioner only
        self.register_buffer('ang_mask', probe.ang_mask.clone())
        self.ang_dim = int(probe.ang_dim)
        self.register_buffer('ang_idx', probe.ang_idx.clone())
        self.dead_rows = tuple(probe.dead_rows)
        self.register_buffer('dead_idx', probe.dead_idx.clone())
        self.register_buffer('live_idx', probe.live_idx.clone())
        self.live_dim = int(probe.live_dim)
        self.register_buffer('_dead_values', probe._dead_values.clone(), persistent=False)
        del probe
        s = resolve_flow_settings(s, self.conditional, self.live_idx.tolist())
        self.settings = s
        if s['atom_tokens'] and not (self.conditional and self.conditions_type == 'molecule'):
            raise ValueError("flow atom_tokens reads the molecule from mol_batch: it needs molecule conditioning")

        wrap_mask = self.ang_mask.cpu().tolist()
        transformer = ((s['tf_width'], s['tf_layers'], s['tf_heads'], s['tf_norm'])
                       if s['conditioner'] == 'transformer' else None)
        shape = (s['n_blocks'], s['n_bins'], (s['hidden'],) * s['hidden_layers'], s['bound'])
        bounded = []
        if s['bounded_rows'] == 'linear':
            dead = set(int(r) for r in self.dead_rows)
            bounded = [i for i in range(self.dim) if i not in dead and not wrap_mask[i]]
        elif s['bounded_rows'] is not None:
            bounded = [int(r) for r in str(s['bounded_rows']).split(',')]
        orders = str(s['block_orders']).split(',') if s['block_orders'] is not None else None
        half_rows = [int(r) for r in str(s['fold_half_rows']).split(',')] if s['fold_half_rows'] else []
        if s['mix_k'] and s['tied_derivatives']:
            raise NotImplementedError("flow tied_derivatives with mix_k")
        if self.conditional:
            if bounded or orders is not None or half_rows or s['mix_k']:
                raise NotImplementedError("flow bounded_rows, block_orders, fold_half_rows and mix_k are built for "
                                          "unconditional policies only (as inn_modeller.py)")
            atoms = (dict(node_dim=s['atom_node_dim'], convs=s['atom_convs'], cutoff=s['atom_cutoff'],
                          radial=s['atom_radial']) if s['atom_tokens'] else None)
            if self.conditions_type == 'molecule' and atoms is None:
                raise NotImplementedError("molecule conditioning on the flow policy needs flow atom_tokens true")
            cond_dim = 1 if atoms is not None else self.conditions_dim
            self.sampler = ConditionalLatentFlow(self.dim, wrap_mask, self.dead_rows, cond_dim, s['context_dim'],
                                                 s['context_hidden'], *shape, seed=int(seed),
                                                 transformer=transformer, atoms=atoms)
            self.flow_model = _LevelHead(self.sampler)
            blocks = self.sampler.flow.blocks
            embed = nn.ModuleDict({'context': self.sampler.flow.context})
            if atoms is not None:
                embed['atom_encoder'] = self.sampler.flow.atom_encoder
            self.conditions_embedding_model = embed
        else:
            self.sampler = LatentFlow(self.dim, wrap_mask, self.dead_rows, *shape, seed=int(seed),
                                      transformer=transformer, block_orders=orders, half_rows=half_rows,
                                      bounded_rows=bounded, mix_k=int(s['mix_k']))
            self.flow_model = LearnableScalar(0.0)
            blocks = self.sampler.flow.blocks
            self.conditions_embedding_model = _Empty()
        self.forward_policy = blocks
        self.t_model, self.s_model, self.backward_policy = _Empty(), _Empty(), _Empty()

        if s['min_bin_height'] is not None:
            h0 = float(s['min_bin_height'])
            if not 0.0 < h0 < 1.0 / s['n_bins']:
                raise ValueError(f"flow min_bin_height {h0:g} must lie in (0, 1/n_bins)")
            for blk in blocks:
                rows = blk.unit_base_rows() if s['min_bin_height_rows'] == 'uniform' else torch.ones_like(blk.is_wrap)
                blk.min_height = torch.where(rows, torch.full_like(blk.bound, h0), torch.full_like(blk.bound, 1e-3))
        if s['tied_derivatives']:
            for blk in blocks:
                blk.tied = float(s['tied_factor'])
        if s['tf_fp32'] and transformer is not None:
            self.sampler.set_bf16(False)

        # the trainer's switches and stashes, as GFN has them
        self.full_flow = False
        self.force_on = self.force_fwd_on = self.force_bwd_on = self.features_on = False
        self.pb_frozen = False
        self._pb_frozen = None
        self._stash_live_branches = False
        self._live_fwd = self._live_bwd = self._live_replay = None
        self._z_cal_embedding = None
        self._force_stats = self._reward_grad_stats = self._path_grad_stats = self._replay_force_stats = None
        self.path_grad_scale_live = False
        self.traj_checkpoint, self.traj_checkpoint_modes = False, None   # runtime flags train.py sets on every build
        self.register_buffer('scalers_fitted', torch.tensor(False))   # persistent: a restored checkpoint keeps its own
        self.to(self.device)

    # -- data-dependent set-up the trainer calls once the prior is loaded ------------------------------------------

    @torch.no_grad()
    def fit_data_scalers(self, latents, conditions=None):
        """Standardise the flow's linear rows on the training latents [N, dim] and, in vector mode, its condition
        input on the training condition vectors [M, conditions_dim]."""
        self.sampler.flow.set_scaler(latents.to(self.device, torch.float32)[:, self.live_idx])
        if self.conditional and not self.settings['atom_tokens'] and conditions is not None:
            self.sampler.flow.set_condition_scaler(conditions.to(self.device, torch.float32))
        self.scalers_fitted.fill_(True)

    # -- the condition -------------------------------------------------------------------------------------------

    def _context(self, condition, mol_batch, n):
        """The flow's condition input for n rows: the condition vector (vector mode), or row indices into an atom
        table set from mol_batch's molecules (molecule mode, one graph per row)."""
        if not self.conditional:
            return None
        if not self.settings['atom_tokens']:
            return condition.to(self.device, torch.float32)
        if int(mol_batch.num_graphs) != n:
            raise ValueError(f"molecule mode needs one graph per row: {int(mol_batch.num_graphs)} graphs, {n} rows")
        ptr = mol_batch.ptr.to(self.device)
        nodes = mol_batch.batch.to(self.device)
        counts = torch.diff(ptr)
        a = int(counts.max())
        idx = torch.arange(nodes.numel(), device=self.device) - ptr[nodes]
        z = torch.zeros(n, a, dtype=torch.long, device=self.device)
        pos = torch.zeros(n, a, 3, device=self.device)
        z[nodes, idx] = mol_batch.z.long().to(self.device).flatten()
        pos[nodes, idx] = mol_batch.pos.float().to(self.device)
        self.sampler.flow.set_atom_table(z, pos, counts)
        return torch.arange(n, device=self.device, dtype=torch.float32)[:, None]

    def get_condition_embedding(self, condition, mol_batch):
        """The level head's input features for these conditions (what flow_model reads)."""
        n = int(condition.shape[0]) if condition is not None and torch.is_tensor(condition) else int(mol_batch.num_graphs)
        return self.sampler.flow.context_input(self._context(condition, mol_batch, n))[1]

    def _scramble(self, c, n, k):
        """train.py's scramble_conditions on a one-step flow: the condition rows permuted tile-wise (tile size k), so
        likelihood training learns the unconditional mixture; as GFN._maybe_scramble_condition_embedding, the
        condition's encoder is not trained through the scrambled rows (_one_step detaches the context when k > 0)."""
        if k <= 0 or c is None:
            return c
        if n % k:
            raise ValueError(f"batch size {n} not divisible by scramble tile size {k}")
        perm = torch.randperm(n // k, device=c.device)
        return c.reshape(n // k, k, -1)[perm].reshape(n, -1)

    # -- geometry helpers --------------------------------------------------------------------------------------------

    def _wrap_ang(self, state):
        if self.ang_dim == 0:
            return state
        x = state.index_select(1, self.ang_idx)
        return state.index_copy(1, self.ang_idx, torch.remainder(x + 1.0, 2.0) - 1.0)

    def _pin_dead(self, state):
        if self.dead_idx.numel() == 0:
            return state
        vals = (self._dead_values.index_select(0, self.dead_idx).to(state.dtype)
                .unsqueeze(0).expand(state.shape[0], -1).contiguous())
        return state.index_copy(1, self.dead_idx, vals)

    def _mean_over_live(self, t):
        return t.mean(-1) if self.dead_idx.numel() == 0 else t.index_select(-1, self.live_idx).mean(-1)

    # -- the trajectory interface ---------------------------------------------------------------------------------

    def _one_step(self, s0, x, c, freeze_policy, return_gauss_params, scrambled: bool = False):
        n = x.shape[0]
        with (torch.no_grad() if freeze_policy else contextlib.nullcontext()):
            log_q = (self.sampler.log_prob(x, c, detach_context=scrambled) if self.conditional
                     else self.sampler.log_prob(x))
        if freeze_policy:
            log_q = log_q.detach()
        states = torch.stack([s0.to(x.dtype), x], 1)
        log_pfs = log_q.reshape(n, 1)
        log_pbs = torch.zeros_like(log_pfs)
        log_flow = torch.zeros(n, 2, device=x.device, dtype=log_q.dtype)
        log_flow[:, 0] = self.sampler.level(c) if self.conditional else self.flow_model()
        if self.conditional:
            self._z_cal_embedding = self.sampler.flow.context_input(c)[1].detach()
        if return_gauss_params:
            gp = {k: torch.zeros(n, 1, device=x.device) for k in self.GAUSS_PARAM_KEYS}
            return states, log_pfs, log_pbs, log_flow, gp
        return states, log_pfs, log_pbs, log_flow

    def get_traj_fwd(self, initial_state, discretizer, exploration_std, condition, mol_batch,
                     return_gauss_params: bool = False, detach_traj: bool = True, freeze_policy: bool = False,
                     path_grad_last_k: int = 0, state_grad_hook=None, drift_context=None):
        if not detach_traj or path_grad_last_k > 0:
            raise NotImplementedError("a pathwise gradient through the flow policy's draw is not wired")
        if exploration_std:
            raise NotImplementedError("exploration noise on the flow policy")
        n = initial_state.shape[0]
        c = self._context(condition, mol_batch, n)
        with torch.no_grad():
            x = self.sampler.sample(n, c) if self.conditional else self.sampler.sample(n)
        x = self._pin_dead(self._wrap_ang(x.to(torch.float32)))
        return self._one_step(initial_state, x, c, freeze_policy, return_gauss_params)

    def get_traj_bwd(self, terminal_state, discretizer, condition, mol_batch, return_gauss_params: bool = False,
                     detach_traj: bool = False, freeze_policy: bool = False, scramble_condition_tiles: int = 0,
                     drift_context=None):
        n = terminal_state.shape[0]
        x = self._pin_dead(self._wrap_ang(terminal_state.detach().to(self.device, torch.float32)))
        c = self._scramble(self._context(condition, mol_batch, n), n, scramble_condition_tiles)
        s0 = torch.zeros_like(x)
        return self._one_step(s0, x, c, freeze_policy, return_gauss_params, scrambled=scramble_condition_tiles > 0)

    def get_traj_replay(self, trajectory, discretizer, condition, mol_batch, return_gauss_params: bool = False,
                        freeze_policy: bool = False, scramble_condition_tiles: int = 0, resample_last_k: int = 0,
                        state_grad_hook=None, implied_noise_last_k: int = 0, drift_context=None, state_forces=None):
        if resample_last_k or implied_noise_last_k or state_forces is not None:
            raise NotImplementedError("replay-tail re-propagation does not apply to a one-step flow policy")
        n = trajectory.shape[0]
        x = self._pin_dead(self._wrap_ang(trajectory[:, -1].detach().to(self.device, torch.float32)))
        c = self._scramble(self._context(condition, mol_batch, n), n, scramble_condition_tiles)
        return self._one_step(trajectory[:, 0].detach().to(self.device, torch.float32), x, c, freeze_policy,
                              return_gauss_params, scrambled=scramble_condition_tiles > 0)

    # -- P_B snapshots and the SDE-only calls ------------------------------------------------------------------------

    def freeze_backward_policy(self, source_state=None):
        """P_B is fixed (log P_B = 0): nothing to freeze."""
        return None

    def unfreeze_backward_policy(self):
        return None

    def install_pb_snapshot(self, frozen):
        return None

    def pb_snapshot_state(self):
        return None

    def install_drift_force(self, provider):
        """train.py installs its drift-force provider on every build; a one-step flow has no drift to force."""
        if provider is not None:
            raise NotImplementedError("a drift force on the flow policy (force terms and state features are SDE-only)")

    def compile_step_kernels(self):
        raise NotImplementedError("compile_policy step kernels are SDE-only")

    def _no_force(self, *args, **kwargs):
        raise NotImplementedError("force terms are SDE-only: a flow policy never has force_on")

    accum_var = force_gate_values = force_nonfinite_rows = restart_force_gates = _no_force
    force_drift_t_min = float('inf')

    def last_step_mean_shift(self, *args, **kwargs):
        raise NotImplementedError("stored-force replay does not apply to a one-step flow policy")

    def bwd_propagate(self, *args, **kwargs):
        raise NotImplementedError("a one-step flow policy has no backward SDE")
