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
    atom_tokens=False,              # False; True (AtomEncoder: xyz in the node input, cross-attended tokens); 'split'
                                    # (SplitAtomEncoder: invariant trunk s for the level head, point head p, tokens
                                    # cross-attended); 'split_pooled' (the same, pooled s and p only, no cross-attention)
    atom_node_dim=128, atom_convs=3, atom_cutoff=10.0, atom_radial=32,
    atom_point_convs=None,          # the split encoder's point-stream message-passing rounds; None = atom_convs, 0 = a
                                    # per-atom MLP
    atom_chirality=True,            # the split encoder's invariant trunk also reads CHIRAL_CHANNELS learned per-atom
                                    # pseudoscalars (models/spline_flow.py::ChiralChannels), so the level head and the
                                    # context tell a molecule from its mirror image (owner, 2026-11-04: on by default).
                                    # Read by the split encoder only: the xyz encoder reads coordinates already, and a
                                    # policy without atom tokens has no molecule encoder
    min_bin_height='auto', min_bin_height_rows='all',
    tied_derivatives=None, tied_factor=None,
    bounded_rows='auto',            # 'auto', 'none'/None, 'linear' (every live unwrapped row on the closed [-1, 1]), or rows '0,3'
    block_orders='auto',            # 'auto' (one-block unconditional: 'id', the chart's own order, cell first; else
                                    # the alternating orders), 'none'/None, or one order per block: 'id', 'rev' or rows
    fold_half_rows=None,            # rows '7,8' whose distribution repeats every half period
    mix_k=0,                        # > 0: a mixture-of-logistics block of k components (unconditional)
    row_tokens='plain',             # 'plain' (each row its value only); 'typed' (crystal_fd's chart rows as typed tokens
                                    # with their physical values and allowed ranges, models/fd_row_describer.py; one
                                    # block); 'prefix' (typed, and each row also reads the prefix physics of the blocks
                                    # completed before it: packing, interplanar spacings, symmetry-image distances)
    mol_scalars=False,              # True (molecule route): the level head and the context also read the molecule's
                                    # log volume, log radius and log principal extents (MOL_SCALARS)
    sym_ops='none',                 # 'none'; 'pooled' (molecule route): the level head and the context also read e_g of
                                    # the row's space group from its symmetry operations (models/sym_ops.py)
    sym_ops_init=None,              # with sym_ops pooled: a models/sym_ops.py save_encoder file, whose pretrained encoder
                                    # e_g starts from (width and hidden from the file); None = a new one, SYM_OPS_SHAPE
)
# settings built after mk_dev's nf block was last edited (mk_dev is owner-edited): accepted and defaulted like
# NF_DEFAULTS, absent from mk_dev until the owner adds them, then moved into NF_DEFAULTS (tests/models/test_nf_policy.py
# pins both halves)
NF_PENDING_DEFAULTS = dict()
ROW_TOKENS = ('plain', 'typed', 'prefix')
CHIRAL_CHANNELS = 8
#: sym_ops: 'pooled' is built; 'tokens' and 'tokens_dynamic' (op tokens in the cross-attention memory, static or
#: refreshed per completed block; docs/design/conditional_ar_architecture.md §5.2) are designed and not built
SYM_OPS = ('none', 'pooled', 'tokens', 'tokens_dynamic')
#: (e_g width, hidden width) of a new e_g encoder: models/sym_ops.py::pretrain's defaults, so a fresh and a pretrained
#: encoder have one shape
SYM_OPS_SHAPE = (64, 256)
#: the molecule scalars of nf mol_scalars: log MolData.mol_volume, log MolData.radius, and the log square roots of the
#: heavy-atom gyration tensor's eigenvalues, largest first (invariant to rotation and reflection)
MOL_SCALARS = ('log_volume', 'log_radius', 'log_extent_1', 'log_extent_2', 'log_extent_3')
NF_ALL_DEFAULTS = {**NF_DEFAULTS, **NF_PENDING_DEFAULTS}
ATOM_TOKEN_MODES = (False, True, 'split', 'split_pooled')
# as inn_modeller.py::UNCONDITIONAL_DEFAULTS and ::Z1_ORDER: the unconditional route's defaults for the 'auto' settings
UNCONDITIONAL_DEFAULTS = dict(bounded_rows='linear', min_bin_height=1e-7, tied_derivatives=True, tied_factor=2.0)
Z1_ORDER = '8.11.10.9.7.6.5.4.3.2.1.0'   # pose first (centroid z, orientation, centroid y, x, then the cell)


def resolve_flow_settings(s: dict, conditional: bool, live_rows=None) -> dict:
    """An unset setting takes UNCONDITIONAL_DEFAULTS on both routes (owner, 2026-11-03: the conditional spline is the
    unconditional one unless there is a reason; inn_modeller.py::resolve_auto_options still leaves them off on its
    conditional route); block_orders 'auto' is 'id' (the chart's own row order, cell block first) on an unconditional
    one-block flow, else the alternating orders, whose first block is also the stored order. Z1_ORDER, the pose-first
    order inn_modeller.py's 'auto' still gives the Z' = 1 latent, is taken only when named. live_rows is not read."""
    s = dict(s)
    for key in ('bounded_rows', 'min_bin_height', 'block_orders'):
        if s[key] == 'none':
            s[key] = None
    if s['bounded_rows'] == 'auto':   # both flows: every row the policy writes lies in [-1, 1] (latent and FD chart)
        s['bounded_rows'] = UNCONDITIONAL_DEFAULTS['bounded_rows']
    if s['min_bin_height'] == 'auto':
        s['min_bin_height'] = UNCONDITIONAL_DEFAULTS['min_bin_height']
    if s['tied_derivatives'] is None:
        s['tied_derivatives'] = UNCONDITIONAL_DEFAULTS['tied_derivatives']
    if s['tied_factor'] is None:
        s['tied_factor'] = 2.0
    if s['block_orders'] == 'auto':   # the chart's own order: the cell block first (owner, 2026-11-04)
        s['block_orders'] = 'id' if not conditional and int(s['n_blocks']) == 1 else None
    if s['sym_ops'] in (None, False, 0, 'off'):   # YAML reads a bare `off` as false
        s['sym_ops'] = 'none'
    return s


# settings that set the flow's arithmetic precision and no parameter: a resume takes them from the run's config
# (checkpointing.py::Checkpointer._gfn_config_from)
PRECISION_ONLY_KEYS = ('tf_fp32',)


def resolve_policy_entry(policy, gfn_config: dict):
    """A gfn_config's `policy` entry with its nf settings as NFPolicy resolves them: NF_DEFAULTS under the entry's
    settings, then resolve_flow_settings on gfn_config's route (conditional) and live rows (range(dim) less
    dead_latent_rows, none when hold_dead_latent_rows is false, as models/gfn.py::GFN reads them). None and an SDE
    entry are returned as given. Resolving a resolved entry returns it unchanged."""
    if policy is None or policy.get('kind', 'sde') != 'nf':
        return policy
    held = gfn_config.get('hold_dead_latent_rows', True)
    dead = {int(r) for r in (gfn_config.get('dead_latent_rows') or ())} if held else set()
    live = [i for i in range(int(gfn_config['dim'])) if i not in dead]
    nf = resolve_flow_settings({**NF_ALL_DEFAULTS, **dict(policy.get('nf') or {})}, bool(gfn_config.get('conditional')),
                               live)
    return {**policy, 'nf': nf}


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
        return NFPolicy(cfg, dict(pol.get('nf') or {}), seed=int(pol.get('seed', 0)),
                        fd_space_group=pol.get('fd_space_group'), fd_space_groups=pol.get('fd_space_groups'))
    raise ValueError(f"model kind {kind!r}: 'sde' or 'nf'")


def _mol_scalars(mol_batch, z, pos, counts):
    """[n, len(MOL_SCALARS)] float32 molecule scalars (MOL_SCALARS) of the n molecules of an atom table (z [n, a],
    pos [n, a, 3], counts [n]) whose MolData fields are on mol_batch."""
    vol, rad = getattr(mol_batch, 'mol_volume', None), getattr(mol_batch, 'radius', None)
    if vol is None or rad is None:
        raise ValueError("flow mol_scalars need each molecule's mol_volume and radius (MolData) on mol_batch")
    n, a = z.shape
    real = torch.arange(a, device=z.device)[None, :] < counts[:, None]
    heavy = real & (z > 1)
    heavy = torch.where(heavy.any(1, keepdim=True), heavy, real)   # a molecule of hydrogens only: every atom
    w = heavy.to(pos.dtype)[..., None]
    mu = (pos * w).sum(1) / w.sum(1)
    d = (pos - mu[:, None]) * w
    gyr = d.transpose(1, 2) @ d / w.sum(1)[..., None]
    ext = 0.5 * torch.log(torch.linalg.eigvalsh(gyr.double()).flip(-1).clamp_min(1e-6)).to(pos.dtype)
    v = torch.as_tensor(vol).reshape(-1).to(pos)
    r = torch.as_tensor(rad).reshape(-1).to(pos)
    return torch.cat([torch.log(v)[:, None], torch.log(r)[:, None], ext], -1).float()


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

    def __init__(self, gfn_config: dict, flow: dict, seed: int = 0, fd_space_group=None, fd_space_groups=None):
        """fd_space_group: the one space group of an FD run (typed rows describe its chart). fd_space_groups: the
        space groups of a mixed-group FD run (two or more), whose state is energies/fd_coordinates.py::FDSlotLayout's
        slots: each row's pad slots are pinned at 0 with no density (spline_flow.py::_ContextFlow.set_row_live, from
        mol_batch.sg_ind) and typed rows describe each row's own group (models/fd_slot_describer.py)."""
        super().__init__()
        from models.gfn import GFN
        unknown = set(flow) - set(NF_ALL_DEFAULTS)
        if unknown:
            raise KeyError(f"unknown nf settings {sorted(unknown)}; known: {sorted(NF_ALL_DEFAULTS)}")
        s = {**NF_ALL_DEFAULTS, **flow}
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
        if s['atom_tokens'] not in ATOM_TOKEN_MODES:
            raise ValueError(f"flow atom_tokens {s['atom_tokens']!r}: one of {ATOM_TOKEN_MODES}")
        if s['atom_tokens'] and not (self.conditional and self.conditions_type == 'molecule'):
            raise ValueError("flow atom_tokens reads the molecule from mol_batch: it needs molecule conditioning")
        describer = None
        self.fd_slots = None
        if fd_space_groups is not None and len(set(int(g) for g in fd_space_groups)) > 1:
            from energies.fd_coordinates import FDSlotLayout
            self.fd_slots = FDSlotLayout(fd_space_groups)
            if self.fd_slots.dim != self.dim or self.dead_rows:
                raise ValueError(f"fd_space_groups {list(self.fd_slots.groups)}: the slot layout has "
                                 f"{self.fd_slots.dim} rows, the policy {self.dim} ({len(self.dead_rows)} dead)")
            if not (self.conditional and s['atom_tokens']):
                raise NotImplementedError("a mixed-group FD state needs a conditional policy on the molecule route "
                                          "(atom_tokens): each row's space group is read from mol_batch")
        if s['row_tokens'] not in ROW_TOKENS:
            raise ValueError(f"flow row_tokens {s['row_tokens']!r}: one of {ROW_TOKENS}")
        if s['sym_ops'] not in SYM_OPS:
            raise ValueError(f"flow sym_ops {s['sym_ops']!r}: one of {SYM_OPS}")
        if s['sym_ops'] in ('tokens', 'tokens_dynamic'):
            raise NotImplementedError(f"flow sym_ops {s['sym_ops']!r}: op tokens are designed (docs/design/"
                                      f"conditional_ar_architecture.md §5.2) and not built; 'pooled' is")
        if s['row_tokens'] != 'plain' and self.fd_slots is not None:
            from models.fd_slot_describer import SlotRowDescriber
            describer = SlotRowDescriber(self.fd_slots, prefix=s['row_tokens'] == 'prefix')
        elif s['row_tokens'] != 'plain':
            # the policy entry names the chart's space group (train.py::Modeller._build_gfn_config on crystal_fd)
            if fd_space_group is None:
                raise ValueError(f"flow row_tokens {s['row_tokens']!r} describes crystal_fd's chart rows: it "
                                 f"needs energy_config.crystal_coordinates fd")
            if self.conditional and not s['atom_tokens']:
                raise NotImplementedError(f"flow row_tokens {s['row_tokens']!r} on a conditional policy needs the "
                                          f"molecule route (atom_tokens): the vector route carries no molecule radius")
            from models.fd_row_describer import FDRowDescriber
            describer = FDRowDescriber(int(fd_space_group), prefix=s['row_tokens'] == 'prefix')
            if describer.dim != self.dim or self.dead_rows:
                raise ValueError(f"flow row_tokens: sg {int(fd_space_group)}'s chart has {describer.dim} rows, the "
                                 f"policy {self.dim} ({len(self.dead_rows)} dead)")

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
        if not self.conditional and (s['mol_scalars'] or s['sym_ops'] != 'none' or s['sym_ops_init']):
            raise ValueError("flow mol_scalars and sym_ops are read beside the condition's molecule: they need a "
                             "conditional policy on the molecule route")
        if self.conditional:
            if orders is not None or half_rows or s['mix_k']:
                raise NotImplementedError("flow block_orders, fold_half_rows and mix_k are built for "
                                          "unconditional policies only (as inn_modeller.py)")
            if s['sym_ops_init'] and s['sym_ops'] != 'pooled':
                raise ValueError("flow sym_ops_init loads e_g's encoder: it needs sym_ops pooled")
            sym_w, sym_h = SYM_OPS_SHAPE
            if s['sym_ops_init']:
                from models.sym_ops import encoder_shape
                sym_w, sym_h = encoder_shape(s['sym_ops_init'])
            atoms = (dict(node_dim=s['atom_node_dim'], convs=s['atom_convs'], cutoff=s['atom_cutoff'],
                          radial=s['atom_radial'],
                          encoder='split' if str(s['atom_tokens']).startswith('split') else 'xyz',
                          cross=s['atom_tokens'] != 'split_pooled',
                          mol_scalars=len(MOL_SCALARS) if s['mol_scalars'] else 0,
                          sym_ops=sym_w if s['sym_ops'] == 'pooled' else 0, sym_hidden=sym_h)
                     if s['atom_tokens'] else None)
            if atoms is not None and atoms['encoder'] == 'split':
                atoms['point_convs'] = s['atom_point_convs']
                atoms['chirality'] = CHIRAL_CHANNELS if s['atom_chirality'] else 0
            elif s['atom_point_convs'] is not None:
                raise ValueError("flow atom_point_convs sets the split encoder's point stream: it needs atom_tokens "
                                 "'split' or 'split_pooled'")
            if s['mol_scalars'] and atoms is None:
                raise ValueError("flow mol_scalars are read beside the molecule's atoms: they need atom_tokens")
            if s['sym_ops'] == 'pooled' and atoms is None:
                raise ValueError("flow sym_ops are read beside the molecule's atoms: they need atom_tokens")
            if self.conditions_type == 'molecule' and atoms is None:
                raise NotImplementedError("molecule conditioning on the flow policy needs flow atom_tokens true")
            # molecule mode: the row id, then the condition vector's columns (temperature, space group, Z')
            cond_dim = 1 + self.conditions_dim if atoms is not None else self.conditions_dim
            self.sampler = ConditionalLatentFlow(self.dim, wrap_mask, self.dead_rows, cond_dim, s['context_dim'],
                                                 s['context_hidden'], *shape, seed=int(seed),
                                                 transformer=transformer, atoms=atoms, bounded_rows=bounded,
                                                 describer=describer)
            self.flow_model = _LevelHead(self.sampler)
            blocks = self.sampler.flow.blocks
            embed = nn.ModuleDict({'context': self.sampler.flow.context})
            if atoms is not None:
                embed['atom_encoder'] = self.sampler.flow.atom_encoder
            if getattr(self.sampler.flow, 'n_sym', 0):   # e_g's encoder trains with the condition's other encoders
                embed['sym_encoder'] = self.sampler.flow.sym_encoder
                if s['sym_ops_init']:
                    from models.sym_ops import load_encoder_into
                    load_encoder_into(self.sampler.flow.sym_encoder, s['sym_ops_init'])
            self.conditions_embedding_model = embed
        else:
            self.sampler = LatentFlow(self.dim, wrap_mask, self.dead_rows, *shape, seed=int(seed),
                                      transformer=transformer, block_orders=orders, half_rows=half_rows,
                                      bounded_rows=bounded, mix_k=int(s['mix_k']), describer=describer)
            self._mol_radius = self._mol_volume = None   # typed rows: the one molecule's, from the first batch
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
                floor = torch.where(rows, torch.full_like(blk.bound, h0), torch.full_like(blk.bound, 1e-3))
                # a buffer, so it follows the policy's .to(device) (a plain attribute stayed on the CPU); not
                # persistent: it is rebuilt from the settings, so checkpoints are unchanged
                if 'min_height' in blk.__dict__:
                    delattr(blk, 'min_height')
                blk.register_buffer('min_height', floor, persistent=False)
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
        """Standardise the flow's linear rows on the training latents [N, dim] and its condition input on the
        training condition vectors [M, conditions_dim]: the whole vector in vector mode, the columns beside the
        molecule in molecule mode."""
        self.sampler.flow.set_scaler(latents.to(self.device, torch.float32)[:, self.live_idx])
        # conditions_dim 0 (the molecule alone): condition_samples' one-column zero placeholder is no column, as
        # _context reads it
        if self.conditional and self.conditions_dim > 0 and conditions is not None and conditions.shape[-1] > 0:
            c = conditions.to(self.device, torch.float32)
            if self.settings['atom_tokens']:   # column 0 is the row id, never standardised (_context)
                c = torch.cat([torch.zeros_like(c[:, :1]), c], -1)
            self.sampler.flow.set_condition_scaler(c)
        self.scalers_fitted.fill_(True)

    # -- the condition -------------------------------------------------------------------------------------------

    def _context(self, condition, mol_batch, n):
        """The flow's condition input for n rows: the condition vector (vector mode), or row indices into an atom
        table set from mol_batch's molecules (molecule mode, one graph per row) followed by the condition vector's
        columns, standardised (temperature, space group, Z': the encoder reads the molecule, these pass through)."""
        if not self.conditional:
            if getattr(self.sampler, 'describer', None) is not None:
                r, v = self._unconditional_molecule(mol_batch)
                self.sampler.set_row_radius(r.expand(n), None if v is None else v.expand(n))
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
        if self.settings['mol_scalars']:
            self.sampler.flow.set_mol_scalars(_mol_scalars(mol_batch, z, pos, counts).to(self.device))
        if self.settings['sym_ops'] == 'pooled':
            sg = getattr(mol_batch, 'sg_ind', None)
            if sg is None:
                raise ValueError("flow sym_ops need each row's space group (sg_ind on mol_batch)")
            self.sampler.flow.set_row_groups(torch.as_tensor(sg).reshape(-1).long())
        if self.fd_slots is not None:   # each row's pads, by the row's own group (a scramble moves conditions only)
            sg = getattr(mol_batch, 'sg_ind', None)
            if sg is None:
                raise ValueError("a mixed-group FD state needs each row's space group (sg_ind on mol_batch)")
            sg = torch.as_tensor(sg).reshape(-1).long()
            self.sampler.flow.set_row_live(self.fd_slots.live_mask(sg).to(self.device))
            if self.sampler.flow.describer is not None:
                self.sampler.flow.describer.set_row_groups(sg)
        if self.sampler.flow.describer is not None:
            radius = getattr(mol_batch, 'radius', None)
            if radius is None:
                raise ValueError("flow row_tokens typed/prefix need each molecule's radius (MolData.radius) on "
                                 "mol_batch")
            volume = None
            if self.sampler.flow.describer.prefix:
                volume = getattr(mol_batch, 'mol_volume', None)
                if volume is None:
                    raise ValueError("flow row_tokens prefix needs each molecule's volume (MolData.mol_volume) on "
                                     "mol_batch")
                volume = torch.as_tensor(volume).reshape(-1).to(self.device, torch.float64)
            self.sampler.flow.set_row_radius(torch.as_tensor(radius).reshape(-1).to(self.device, torch.float64),
                                             volume)
        ids = torch.arange(n, device=self.device, dtype=torch.float32)[:, None]
        if self.conditions_dim == 0:
            return ids
        if condition is None or condition.shape[-1] != self.conditions_dim:
            raise ValueError(f"molecule mode with {self.conditions_dim} condition columns (temperature, space group, "
                             f"Z') needs them passed: got {None if condition is None else tuple(condition.shape)}")
        f = self.sampler.flow   # standardised here, not in the flow, which inn_modeller also drives raw
        rest = (condition.to(self.device, torch.float32) - f.c_mu[1:]) / f.c_sd[1:]
        return torch.cat([ids, rest], -1)

    def _unconditional_molecule(self, mol_batch):
        """Typed rows on an unconditional policy: the one molecule's radius [] (Angstrom) and, with prefix features,
        volume [] (cubic Angstrom, else None), float64, read from mol_batch.radius / mol_volume when the call carries
        them (every row must agree with the values first seen) and the held values otherwise."""
        prefix = self.sampler.describer.prefix
        for name, attr in (('radius', '_mol_radius'), ('mol_volume', '_mol_volume')):
            if name == 'mol_volume' and not prefix:
                continue
            val = getattr(mol_batch, name, None) if mol_batch is not None else None
            if val is not None:
                v = torch.as_tensor(val).reshape(-1).to(self.device, torch.float64)
                if getattr(self, attr) is None:
                    setattr(self, attr, v[0].clone())
                if not bool(torch.allclose(v, getattr(self, attr).expand_as(v), rtol=1e-6, atol=1e-6)):
                    raise ValueError(f"flow typed_rows on an unconditional policy: the batch's molecule {name}s "
                                     f"differ from the one molecule's (an unconditional run has one molecule)")
            if getattr(self, attr) is None:
                raise ValueError(f"flow typed_rows needs the molecule's {name} (MolData.{name} on mol_batch) on the "
                                 f"first call")
        return self._mol_radius, (self._mol_volume if prefix else None)

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

    @torch.no_grad()
    def log_density(self, x, condition, mol_batch):
        """Exact log q(x | c) of terminal states x [N, dim] (the trainer's latents), as the backward step scores
        them: angles wrapped and dead rows pinned first. Read by train.py::Modeller.density_readouts."""
        n = x.shape[0]
        x = self._pin_dead(self._wrap_ang(x.detach().to(self.device, torch.float32)))
        c = self._context(condition, mol_batch, n)
        return self.sampler.log_prob(x, c) if self.conditional else self.sampler.log_prob(x)

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
