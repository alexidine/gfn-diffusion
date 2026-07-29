import math
from argparse import Namespace
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from energy_sampling.utils import gaussian_params
from mxtaltools.models.graph_models.molecule_graph_model import VectorMoleculeGraphModel
from mxtaltools.models.modules.components import scalarMLP
from .architectures import NoneModule, LearnableScalar, TimeEncoding, StateEncoding, PolicyModel

logtwopi = math.log(2 * math.pi)


class GFN(nn.Module):  # todo add seeding
    def __init__(self, dim: int, s_emb_dim: int, conditions_dim: int,
                 harmonics_dim: int, t_dim: int,
                 t_hidden_dim: int = 64,
                 s_hidden_dim: int = 64, s_layers: int = 4,
                 policy_hidden_dim: int = 64, policy_layers: int = 4,
                 flow_hidden_dim: int = 64, flow_layers: int = 4,
                 cond_hidden_dim: int = 64, cond_layers: int = 4,
                 log_var_range: float = 4.,
                 t_scale: float = 1., learned_variance: bool = True,
                 t_scale_ratio: Optional[float] = None,
                 t_scale_power: float = 4.,
                 t_scale_preserve_budget: bool = True,
                 condition_embedding_dim: int = 0,
                 conditions_type: str = 'vector',
                 clipping: bool = False,
                 gfn_clip: float = 1e4, pb_drift_range: float = 0.1,
                 pb_var_range: float = 0.1,
                 conditional: bool = False,
                 learn_pb: bool = False,
                 dropout: Optional[float] = 0, norm: Optional[str] = None,
                 zero_init: bool = False, device=torch.device('cuda'),
                 max_z_prime: int = 1,
                 full_flow: bool = False,
                 do_periodic_angles: bool = True,
                 periodic_centroids: bool = False,
                 periodic_centroid_axes: Optional[Sequence[int]] = None,
                 dplr_rank: int = 0,
                 dplr_rho_max: float = 0.9,
                 dplr_mask_angular: bool = True,
                 ):
        super(GFN, self).__init__()
        self.dim = dim
        self.harmonics_dim = harmonics_dim
        self.t_dim = t_dim
        self.s_emb_dim = s_emb_dim

        self.learned_variance = learned_variance
        self.t_scale = t_scale

        self.clipping = clipping
        self.gfn_clip = gfn_clip  # clipping maximum step size

        self.conditional = conditional
        self.learn_pb = learn_pb

        self.pf_std_per_traj = np.sqrt(self.t_scale)
        self._log_var_base = 2.0 * float(np.log(self.pf_std_per_traj))  # constant-rate additive shift onto any policy logvar
        self.log_var_range = log_var_range
        self.var_clip = 16
        self.var_floor = 1e-12  # guards V(t) in denominators; the pinned t=0 step is special-cased upstream
        self.init_var_schedule(t_scale_ratio, t_scale_power, t_scale_preserve_budget)
        self.device = device
        self.max_z_prime = max_z_prime
        self.conditions_type = conditions_type
        self.full_flow = full_flow

        # diagonal-plus-low-rank (DPLR) forward covariance: C = diag(d) + V V^T,
        # with (d, V) built from a fixed per-dim marginal-variance budget s^2,
        # a correlated fraction rho in [0, rho_max), and unit directions U (see
        # get_dplr_cov). rank 0 disables it and the forward kernel is exactly
        # the original diagonal SDE.
        self.dplr_rank = dplr_rank
        self.dplr_rho_max = dplr_rho_max
        self.dplr_mask_angular = dplr_mask_angular
        self.dplr_eps = 1e-6
        # scalarMLP's output layer has no bias, so we can't init a learnable
        # bias toward "rho ~ 0"; instead we zero-init the rho/U weight block
        # (below) and subtract this fixed constant from the raw logit, which
        # has the same effect: sigmoid(0 - 4) ~ 0.018 at initialization.
        self.dplr_rho_init_bias = 4.0

        # `periodic_centroids` is the on/off switch (config-level); the axes themselves are
        # a space group property and so are resolved by the caller (train.py) rather than here
        self.periodic_centroids = periodic_centroids
        self.get_periodic_dimensions(
            device, do_periodic_angles=do_periodic_angles,
            periodic_centroid_axes=periodic_centroid_axes if periodic_centroids else None)
        self.condition_embedding_dim = condition_embedding_dim

        self.init_conditioner(cond_hidden_dim, cond_layers, condition_embedding_dim, conditions_dim,
                              dropout, norm)

        self.init_flow_model(condition_embedding_dim, dropout, flow_hidden_dim, flow_layers,
                             norm, self.full_flow, s_emb_dim, t_dim)

        self.t_model = TimeEncoding(harmonics_dim, t_dim, t_hidden_dim,
                                    norm=norm, dropout=dropout)
        self.s_model = StateEncoding(self.expanded_dim,
                                     s_layers,
                                     s_hidden_dim,
                                     condition_embedding_dim if self.conditional else 0,
                                     s_emb_dim,
                                     norm=norm, dropout=dropout)
        self.init_policies(s_emb_dim, t_dim, policy_hidden_dim, policy_layers, zero_init, norm, dropout)

        self.pb_drift_range = pb_drift_range
        self.pb_var_range = pb_var_range

        # runtime flag, set post-construction (train.py) like compile_policy --
        # deliberately NOT a constructor arg, so it stays out of gfn_config and
        # therefore out of checkpoints/problem hashing. When on, each trajectory
        # step is gradient-checkpointed (activations recomputed in backward):
        # rollout activation memory becomes ~O(1) in T for one extra policy
        # forward per step. Values and gradients are identical either way.
        self.traj_checkpoint = False

    def init_policies(self, s_emb_dim, t_dim, policy_hidden_dim, policy_layers, zero_init, norm, dropout):
        # head layout when dplr_rank > 0: [mean(dim), log(s^2)(dim), rho_logit(dim), U(dim*rank)]
        fwd_out_dim = 3 * self.dim + self.dim * self.dplr_rank if self.dplr_rank > 0 else 2 * self.dim
        self.forward_policy = PolicyModel(self.dim, s_emb_dim, t_dim,
                                          policy_hidden_dim, policy_layers,
                                          fwd_out_dim,
                                          zero_init=zero_init,
                                          norm=norm, dropout=dropout)
        if self.dplr_rank > 0 and not zero_init:
            # zero-init only the rho_logit block, so raw rho_logit ~ 0 and,
            # combined with dplr_rho_init_bias, rho ~ 0 (input-independent) at
            # init -- DPLR starts at (near-)diagonal covariance. U is left at
            # its default init (small but not exactly 0): normalizing an
            # exactly-zero row has an unbounded (1/eps) gradient, and zeroing
            # U isn't needed anyway -- sqrt(rho) ~ 0.13 already suppresses V's
            # magnitude at init regardless of U's direction.
            with torch.no_grad():
                self.forward_policy.model.output_layer.weight.data[2 * self.dim:3 * self.dim].zero_()
        self.backward_policy = PolicyModel(self.dim, s_emb_dim, t_dim,
                                           policy_hidden_dim, policy_layers, 2 * self.dim,
                                           zero_init=zero_init,
                                           norm=norm, dropout=dropout)

    def init_conditioner(self, cond_hidden_dim, cond_layers, condition_embedding_dim, conditions_dim,
                         dropout, norm):
        if self.conditional:
            if self.conditions_type == 'vector':
                self.conditions_embedding_model = scalarMLP(input_dim=conditions_dim,
                                                            norm=norm,
                                                            dropout=dropout,
                                                            layers=cond_layers,
                                                            filters=cond_hidden_dim,
                                                            output_dim=condition_embedding_dim,
                                                            )

            elif self.conditions_type == 'molecule':
                # will return nice o(3) invariant embeddings from vector inputs
                self.conditions_embedding_model = VectorMoleculeGraphModel(
                    input_node_dim=1,
                    num_mol_feats=conditions_dim,
                    output_dim=condition_embedding_dim,
                    concat_pos_to_node_dim=True,
                    concat_mol_to_node_dim=False,
                    fc_config=Namespace(
                        num_layers=cond_layers,
                        hidden_dim=cond_hidden_dim,
                        norm=norm,
                        vector_norm='vector ' + norm if norm is not None else None,
                        dropout=dropout,
                    ),
                    graph_config=Namespace(
                        node_dim=cond_hidden_dim,
                        fcs_per_gc=1,
                        message_dim=cond_hidden_dim // 4,
                        embedding_dim=cond_hidden_dim,
                        num_convs=cond_layers,
                        num_radial=32,
                        cutoff=6.0,
                        max_num_neighbors=100,
                        norm='graph ' + norm if norm is not None else None,
                        vector_norm='graph vector ' + norm if norm is not None else None,
                        dropout=0.0,
                        v_embedding_dim=None,
                        v_input_node_dim=1,
                    )
                )


        else:  # we can pass arguments to this (conditions) but nothing will happen
            self.conditions_embedding_model = NoneModule()

    def init_flow_model(self, condition_embedding_dim, dropout, flow_hidden_dim, flow_layers, norm,
                        full_flow, s_emb_dim, t_emb_dim):
        if full_flow:  # time and state-dependent flow model
            if self.conditional:
                self.flow_model = scalarMLP(layers=flow_layers,
                                            filters=flow_hidden_dim,
                                            input_dim=s_emb_dim + t_emb_dim,
                                            output_dim=1,
                                            norm=norm,
                                            dropout=dropout,
                                            )
            else:
                self.flow_model = scalarMLP(layers=flow_layers,
                                            filters=flow_hidden_dim,
                                            input_dim=s_emb_dim + t_emb_dim,
                                            output_dim=1,
                                            norm=norm,
                                            dropout=dropout,
                                            )
        else:
            if self.conditional:
                self.flow_model = scalarMLP(layers=flow_layers,
                                            filters=flow_hidden_dim,
                                            input_dim=condition_embedding_dim,
                                            output_dim=1,
                                            norm=norm,
                                            dropout=dropout,
                                            )
            else:
                self.flow_model = LearnableScalar()  # unified syntax with this instead of nn.Parameter

    def get_periodic_dimensions(self, device, do_periodic_angles: bool = True,
                                periodic_centroid_axes: Optional[Sequence[int]] = None):
        """
        Build the mask of state dims that live on a circle. ang_mask means exactly
        "this dim is wrapped": it drives the sin/cos policy encoding
        (expand_state_for_policy) and the post-step wrap in get_traj_fwd/bwd.

        State layout is
            [6 box params | 3*max_z_prime aunit centroids | 3*max_z_prime aunit orientations]
        Orientation dims: phi and r are periodic in the rotational basis.

        Centroid dims are periodic ONLY for the axes in `periodic_centroid_axes`. The
        aunit is generally not periodic -- crossing a face re-enters through a symmetry
        operation rather than a translation (e.g. P21/c at y=1/4), a genuine
        discontinuity -- but on axes where the shift by the aunit width is itself a
        lattice translation the coordinate really is circular, and wrapping lets the SDE
        flow through the face instead of being held off it by the bounding-energy wall.
        Which axes qualify depends on the space group, so the caller supplies them; see
        models/aunit_periodicity.py for the derivation and its empirical validation.

        Note this also extends `dplr_mask_angular` (get_dplr_cov) to the wrapped centroid
        dims, which is the consistent reading: that mask exists to keep the shared
        low-rank noise direction off circular coordinates, and these are now circular.
        """
        if do_periodic_angles:
            angs = [False] * 6
            for zp in range(self.max_z_prime):
                angs.extend([False, False, False])
            for zp in range(self.max_z_prime):
                angs.extend([False, True, True])
                # phi and r dimensions arein rotational basis
        else:
            angs = [False] * 12

        self.periodic_centroid_axes = tuple(sorted(set(periodic_centroid_axes or ())))
        if self.periodic_centroid_axes:
            if not do_periodic_angles:
                raise ValueError(
                    "periodic centroid axes are a molecular-crystal property, but this GFN was "
                    "built with do_periodic_angles=False (i.e. a toy/latent energy whose state "
                    "is not a crystal parameterization)")
            if any(a not in (0, 1, 2) for a in self.periodic_centroid_axes):
                raise ValueError(f"centroid axes must be in (0, 1, 2), got {self.periodic_centroid_axes}")
            expected_dim = 6 + 6 * self.max_z_prime
            if self.dim != expected_dim:
                raise ValueError(
                    f"expected crystal state dim {expected_dim} for max_z_prime={self.max_z_prime}, "
                    f"got {self.dim}; refusing to index centroid dims into an unknown layout")
            for zp in range(self.max_z_prime):
                for axis in self.periodic_centroid_axes:
                    angs[6 + 3 * zp + axis] = True

        self.ang_mask = torch.tensor(angs, device=device)
        self.ang_dim = (self.ang_mask == True).sum().item()
        self.lin_dim = self.dim - self.ang_dim
        self.expanded_dim = self.lin_dim + self.ang_dim * 2
        # integer twins of ang_mask for the per-step hot paths: bool-mask
        # indexing on CUDA runs nonzero() and forces a host sync every call
        self.ang_idx = self.ang_mask.nonzero(as_tuple=False).flatten()
        self.lin_idx = (~self.ang_mask).nonzero(as_tuple=False).flatten()

    VAR_SCHEDULE_GRID = 8192

    def init_var_schedule(self, ratio, power, preserve_budget):
        """
        Optional in-rollout schedule for the forward noise RATE (variance per
        unit trajectory time), decaying from t=0 to t=1:

            sigma^2(t) = sigma^2(0) * ratio ** (t ** power)

        `ratio` = sigma^2(1)/sigma^2(0); None disables the schedule entirely
        and every method below short-circuits to the constant-rate expression
        the original code wrote, verbatim (see the parity note there).

        Everything downstream runs on the ACCUMULATED variance
        V(t) = int_0^t sigma^2(s) ds rather than on t. With a constant rate
        V(t) = t_scale * t, so the two are proportional and t was a valid
        stand-in; once the rate varies they come apart, and every place the
        old code wrote a time ratio it meant a variance ratio. P_B is the
        Brownian bridge of this process, so it has to move with the schedule
        -- its drift coefficient can only be rescaled by +-40%
        (pb_drift_range), which is nowhere near enough to absorb the
        difference over the last third of the rollout.

        preserve_budget rescales sigma^2(0) so V(1) = t_scale exactly. The
        trajectory starts from a deterministic x_0 = 0 under a deterministic
        drift, so V(1) is the entire stock of randomness available to build
        the terminal distribution: without this, a schedule shrinks the
        reachable support at the same time as the terminal step, and the two
        effects can't be separated afterwards.
        """
        self.t_scale_ratio = ratio
        self.t_scale_power = power
        self.t_scale_preserve_budget = preserve_budget
        self.var_scheduled = ratio is not None
        if not self.var_scheduled:
            return
        if not 0 < ratio <= 1:
            raise ValueError(f"t_scale_ratio is sigma^2(1)/sigma^2(0) and must lie in (0, 1], got {ratio}")
        if power <= 0:
            raise ValueError(f"t_scale_power must be positive, got {power}")

        n = self.VAR_SCHEDULE_GRID
        grid = torch.linspace(0, 1, n + 1, dtype=torch.float64)
        rate = torch.exp(math.log(ratio) * grid ** power)  # sigma^2(t) / sigma^2(0)
        increments = torch.zeros(n + 1, dtype=torch.float64)
        increments[1:] = 0.5 * (rate[1:] + rate[:-1]) / n  # cumulative trapezoid; V(0) = 0 exactly
        accum = increments.cumsum(0)
        scale = self.t_scale / accum[-1].item() if preserve_budget else self.t_scale
        # derived wholly from config, so non-persistent: it stays out of the
        # state_dict and checkpoints are identical with or without a schedule
        self.register_buffer('_var_accum', (accum * scale).to(torch.float32), persistent=False)

    def var_accum(self, t):
        """
        V(t), by linear interpolation on the precomputed grid. Exact zero at
        t=0, so the pinned first backward step keeps its existing branch.
        """
        x = t.clamp(min=0.0, max=1.0) * self.VAR_SCHEDULE_GRID
        idx = x.floor().long().clamp(max=self.VAR_SCHEDULE_GRID - 1)
        frac = x - idx.to(x.dtype)
        v0 = self._var_accum[idx]
        return v0 + frac * (self._var_accum[idx + 1] - v0)

    def var_log_rate(self, t_prev, t_next, dts):
        """
        Additive baseline on a policy logvar for the step [t_prev, t_next]:
        the log of that step's mean rate, so the per-step variance a caller
        forms as exp(logvar) * dt is exactly the variance the schedule
        allocates to the step. Constant rate -> the original scalar
        log(t_scale), returned as a float so the caller's add is unchanged.
        """
        if not self.var_scheduled:
            return self._log_var_base
        dv = self.var_accum(t_next) - self.var_accum(t_prev)
        return dv.clamp(min=self.var_floor).div(dts).log().unsqueeze(1)

    def var_drift_coeff(self, t_prev, t_next, dts):
        """
        dV/V(t_next): the fraction of accumulated variance the backward bridge
        removes in one step, i.e. how hard P_B pulls toward the origin.
        Constant rate -> dt/t_next, the original expression verbatim.
        """
        if not self.var_scheduled:
            return dts / t_next
        v_next = self.var_accum(t_next).clamp(min=self.var_floor)
        return (v_next - self.var_accum(t_prev)) / v_next

    def var_bridge_step(self, t_prev, t_next, dts):
        """
        The whole factor multiplying P_B's variance, dt * V(t_prev)/V(t_next)
        -- bridge contraction toward the pin at the origin. Constant rate ->
        dt * t_prev/t_next, written with the original grouping so the result
        is bit-identical, not merely equal.
        """
        if not self.var_scheduled:
            return dts * t_prev / t_next
        v_prev = self.var_accum(t_prev)
        v_next = self.var_accum(t_next).clamp(min=self.var_floor)
        return dts * v_prev / v_next

    def split_params(self, tensor, log_var_base):
        if self.dplr_rank > 0:
            mean, logvar_i, rho_logit, u_raw = torch.split(
                tensor, [self.dim, self.dim, self.dim, self.dim * self.dplr_rank], dim=-1)
            u_raw = u_raw.view(-1, self.dim, self.dplr_rank)
        else:
            mean, logvar_i = gaussian_params(tensor)
            rho_logit, u_raw = None, None
        if not self.learned_variance:
            logvar = torch.zeros_like(logvar_i)
        else:
            if self.log_var_range == -1:
                logvar = logvar_i
            else:
                logvar = torch.tanh(logvar_i / self.log_var_range) * self.log_var_range
        logvar = (logvar + log_var_base).clip(min=-self.var_clip, max=self.var_clip)
        return mean, logvar, rho_logit, u_raw

    def get_dplr_cov(self, logvar, rho_logit, u_raw):
        """
        DPLR forward covariance, scheme B (fixed marginal-variance budget).
        s^2 = exp(logvar) is the total per-dim variance budget -- the same
        quantity the pure-diagonal policy variance always was. rho in
        [0, rho_max) redistributes that budget between a private diagonal d
        and a shared low-rank part V, holding the marginal fixed exactly:
            d_k = (1 - rho_k) * s_k^2
            V_{k,:} = sqrt(rho_k) * s_k * Uhat_{k,:}          (Uhat: unit rows)
            C_kk = d_k + ||V_{k,:}||^2 = s_k^2
        so "how big a step" (s) and "how tilted" (rho, U) are orthogonal
        knobs. Returns (d, V); V is None when DPLR is disabled.
        """
        s2 = logvar.exp()
        if rho_logit is None:
            return s2, None
        rho = torch.sigmoid(rho_logit - self.dplr_rho_init_bias) * self.dplr_rho_max
        if self.dplr_mask_angular and self.ang_dim > 0:
            # mask the fraction, not V directly: the diagonal then reabsorbs
            # the freed budget automatically (d_k = s_k^2 exactly on these dims)
            rho = rho.masked_fill(self.ang_mask.view(1, -1), 0.0)
        u_hat = u_raw / (u_raw.norm(dim=-1, keepdim=True) + self.dplr_eps)
        d = (1.0 - rho) * s2
        V = (rho.sqrt() * s2.sqrt()).unsqueeze(-1) * u_hat
        return d, V

    def eval_forward_head(self, state_update, log_var_base):
        """split_params + get_dplr_cov, chained: every caller needs both together."""
        pf_mean, logvar, rho_logit, u_raw = self.split_params(state_update, log_var_base)
        d, V = self.get_dplr_cov(logvar, rho_logit, u_raw)
        return pf_mean, logvar, d, V

    def predict_next_state(self, s_emb, t_emb):
        s_new = self.forward_policy(s_emb, t_emb)

        if self.clipping:
            s_new = torch.clip(s_new, -self.gfn_clip, self.gfn_clip)

        return s_new

    def _forward_kernel(self, state, t, condition_embedding, t_next, dts):
        """
        Evaluate the forward policy at `state`/`t`: this is the one density
        (mean, d, V) shared by get_traj_fwd, get_traj_bwd's pf term, and
        get_traj_replay -- DPLR (when enabled) always applies here, since
        it's always the same forward_policy network being scored.

        The policy network sees the step's START time `t` (unchanged), while
        the variance baseline is taken over the whole step [t, t_next]: the
        noise on this step is what lands the sample at t_next, so that's the
        interval whose variance budget it should spend. It also makes P_F and
        P_B read the same baseline for the same interval.
        """
        expanded_state = self.expand_state_for_policy(state)
        s_emb = self.s_model(expanded_state, condition_embedding)
        t_emb = self.t_model(t)
        state_update = self.predict_next_state(s_emb, t_emb)
        pf_mean, logvar, d, V = self.eval_forward_head(
            state_update, self.var_log_rate(t, t_next, dts))
        return pf_mean, logvar, d, V, s_emb, t_emb

    def _use_traj_checkpoint(self):
        return self.traj_checkpoint and torch.is_grad_enabled()

    def _run_step(self, use_ckpt, step_fn, *args):
        """
        Execute one trajectory step, optionally under gradient checkpointing.
        Step functions must be pure w.r.t. their inputs (all randomness is
        pre-drawn at loop level and passed in as eps tensors) so the backward
        recompute replays them exactly. NB the conditioner (PyG scatter ops,
        nondeterministic on CUDA) must stay OUTSIDE the step functions:
        condition_embedding is computed once per trajectory and passed in.
        """
        if use_ckpt:
            return grad_checkpoint(step_fn, *args, use_reentrant=False)
        return step_fn(*args)

    def _wrap_ang(self, state):
        """Out-of-place twin of the old in-place post-step angular wrap."""
        if self.ang_dim == 0:
            return state
        wrapped = self.wrap_to_pi(state.index_select(1, self.ang_idx) * torch.pi) / torch.pi  # latent space is on [-1, 1]
        return state.index_copy(1, self.ang_idx, wrapped)

    def _step_flow(self, s_emb, t_emb):
        """
        Per-step flow value under full_flow (state/time-dependent head); None
        otherwise -- the constant-flow case is state-independent, so the traj
        functions write log_flow[:, 0] once from the condition embedding,
        outside the step loop.

        Z-only training (freeze_policy) is handled upstream by detaching
        condition_embedding at its source (see get_traj_fwd/bwd/replay), so the
        flow head trains its own parameters while the conditioner stays frozen.
        NB: that source detach isolates flow_model only while full_flow is off;
        under full_flow the flow head reads [s_emb, t_emb], so it would still
        train s_model -- revisit here if full_flow + freeze_policy is ever used.
        """
        if not self.full_flow:
            return None
        return self.flow_model(torch.cat([s_emb, t_emb], dim=1)).flatten()

    def _fwd_step(self, current_state, dts, ts, condition_embedding, eps, eps_r,
                  exploration_std, i: int, detach_traj: bool):
        """
        One forward-rollout step: propagate current_state -> next_state and
        score the transition. Returns the (angular-wrapped) next state plus
        everything the loop needs downstream; the trailing Gaussian-parameter
        tensors are only consumed under return_gauss_params.
        """
        pf_mean, pflogvars, d, V, s_emb, t_emb = self._forward_kernel(
            current_state, ts[:, i], condition_embedding, ts[:, i + 1], dts)
        pflogvars_sample = self.fwd_get_logvars(detach_traj, dts, exploration_std, i, d.log())
        # exploration only inflates the diagonal; V is never inflated for sampling
        V_sample = V.detach() if (detach_traj and V is not None) else V
        next_state = self.fwd_propagate(current_state, detach_traj, dts, pf_mean, pflogvars_sample,
                                        V_sample, eps=eps, eps_r=eps_r)

        flow_i = self._step_flow(s_emb, t_emb)

        # compute forward logprobs under the actual (un-inflated) policy density
        fwd_drift = dts.unsqueeze(1) * pf_mean
        logpf_i = self.fwd_gauss_logprob(next_state - current_state, fwd_drift, d, dts, V)

        # compute backward logprobs
        back_drift, back_var, logpb_i = self._eval_pb_logprob(
            condition_embedding, i, current_state, next_state, dts, ts, logpf_i)

        # only wrap after logprob calculations
        next_state = self._wrap_ang(next_state)
        return next_state, logpf_i, logpb_i, flow_i, back_drift, back_var, fwd_drift, pflogvars, d

    def _bwd_step(self, current_state, dts, ts, condition_embedding, eps,
                  i: int, trajectory_length: int, detach_traj: bool):
        """
        One backward-rollout step: propagate current_state -> prev_state under
        P_B (deterministically to the source on the final step) and score both
        directions. Returns the (angular-wrapped) previous state.
        """
        if i < trajectory_length - 1:
            # backward propagate
            expanded_current_state = self.expand_state_for_policy(current_state)

            back_mean_correction, back_var_correction \
                = self.get_bwd_correction(
                condition_embedding, expanded_current_state, i, trajectory_length, ts)

            t_prev, t_next = ts[:, trajectory_length - i - 1], ts[:, trajectory_length - i]
            drift_coeff = self.var_drift_coeff(t_prev, t_next, dts).unsqueeze(1)

            # directly x_t-u\Delta t
            back_mean = current_state - current_state * drift_coeff * back_mean_correction

            back_drift = - current_state * drift_coeff * back_mean_correction
            var = (back_var_correction + self.var_log_rate(t_prev, t_next, dts)).clip(
                min=-self.var_clip, max=self.var_clip).exp()
            back_var = var * self.var_bridge_step(t_prev, t_next, dts).unsqueeze(1)
            prev_state = self.bwd_propagate(back_mean, back_var, current_state, detach_traj, eps=eps)

            # log Pb calculation
            logpb_i = self.gauss_logprob(prev_state - current_state, back_drift, back_var)
        else:
            # send it identically to the source state (0)
            prev_state = torch.zeros_like(current_state)
            back_drift = -current_state
            back_var = torch.ones_like(back_drift) * 1e-3 * dts.unsqueeze(1)
            logpb_i = current_state.new_zeros(current_state.shape[0])

        """log pf calculation"""
        # forward kernel: same policy density as get_traj_fwd/get_traj_replay, so DPLR
        # applies here too when enabled. Only the backward policy (P_B, above) stays diagonal.
        pf_mean, pflogvars, d, V, s_emb, t_emb = self._forward_kernel(
            prev_state, ts[:, trajectory_length - i - 1], condition_embedding,
            ts[:, trajectory_length - i], dts)

        flow_i = self._step_flow(s_emb, t_emb)

        fwd_drift = dts.unsqueeze(1) * pf_mean
        logpf_i = self.fwd_gauss_logprob(current_state - prev_state, fwd_drift, d, dts, V)

        prev_state = self._wrap_ang(prev_state)
        return prev_state, logpf_i, logpb_i, flow_i, back_drift, back_var, fwd_drift, pflogvars, d

    def _replay_step(self, current_state, next_state, dts, ts, condition_embedding, i: int):
        """
        Score one fixed transition (replayed trajectory): no propagation, no
        wrap -- states are read as given.
        """
        # PROPAGATION (evaluated against the given transition, not sampled)
        pf_mean, pflogvars, d, V, s_emb, t_emb = self._forward_kernel(
            current_state, ts[:, i], condition_embedding, ts[:, i + 1], dts)

        flow_i = self._step_flow(s_emb, t_emb)

        # compute forward logprobs (mirrors get_traj_fwd's un-inflated policy density)
        fwd_drift = dts.unsqueeze(1) * pf_mean
        logpf_i = self.fwd_gauss_logprob(next_state - current_state, fwd_drift, d, dts, V)

        # compute backward logprobs
        back_drift, back_var, logpb_i = self._eval_pb_logprob(
            condition_embedding, i, current_state, next_state, dts, ts, logpf_i)
        return logpf_i, logpb_i, flow_i, back_drift, back_var, fwd_drift, pflogvars, d

    def get_traj_fwd(self, initial_state, discretizer, exploration_std, condition, mol_batch,
                     return_gauss_params: bool = False, detach_traj: bool = True,
                     freeze_policy: bool = False):
        batch_size = initial_state.shape[0]
        ts = discretizer(batch_size).to(self.device)
        trajectory_length = ts.shape[1] - 1
        logpb, logpf, states, gauss_params, log_flow = self.init_traj_tensors(batch_size, trajectory_length)

        initial_state.requires_grad_(not detach_traj)
        current_state = initial_state
        states[:, 0] = current_state

        if self.conditional:
            if condition is not False:
                condition_embedding = self.get_condition_embedding(condition, mol_batch)
            else:  # constant embedding
                condition_embedding = torch.zeros((batch_size, self.condition_embedding_dim),
                                                  dtype=torch.float32, device=self.device)
            if freeze_policy:  # Z-only training: cut gradient to the conditioner at the source
                condition_embedding = condition_embedding.detach()
        else:
            condition_embedding = None

        # detached copy for the interspersed z-calibration step (train.py
        # z_calibration_tick), paired with condition_id by fwd_train_step
        # immediately after this rollout's loss returns. fwd rollouts are
        # never condition-scrambled (that lives on the bwd-side paths), so
        # this is always the TRUE embedding/condition pairing.
        self._z_cal_embedding = (condition_embedding.detach()
                                 if condition_embedding is not None else None)

        use_ckpt = self._use_traj_checkpoint()
        if not self.full_flow:
            log_flow[:, 0] = self.flow_model(condition_embedding).flatten()

        for i in range(trajectory_length):
            dts = ts[:, i + 1] - ts[:, i]
            # pre-draw the step's noise at loop level (outside any checkpoint
            # region) so recompute-in-backward replays the step exactly
            eps = torch.randn(batch_size, self.dim, dtype=current_state.dtype, device=self.device)
            eps_r = (torch.randn(batch_size, self.dplr_rank, device=self.device)
                     if self.dplr_rank > 0 else None)

            (next_state, logpf_i, logpb_i, flow_i,
             back_drift, back_var, fwd_drift, pflogvars, d) = self._run_step(
                use_ckpt, self._fwd_step, current_state, dts, ts, condition_embedding,
                eps, eps_r, exploration_std, i, detach_traj)

            logpf.append(logpf_i)
            logpb.append(logpb_i)
            if self.full_flow:
                log_flow[:, i] = flow_i
            if return_gauss_params:
                self.log_gauss_params(gauss_params, i, back_drift, back_var, dts, fwd_drift, pflogvars, d)

            current_state = next_state
            states[:, i + 1] = current_state

        logpfs = torch.stack(logpf).T
        logpbs = torch.stack(logpb).T
        if return_gauss_params:
            return states, logpfs, logpbs, log_flow, {k: v.mean(-1) for k, v in gauss_params.items()}
        else:
            return states, logpfs, logpbs, log_flow

    def log_gauss_params(self, gauss_params, i, back_drift, back_var, dts, fwd_drift, pflogvars, d):
        """
        Record per-step diagnostic Gaussian parameters into `gauss_params`
        (see GAUSS_PARAM_KEYS). logvars_f is the total per-dim variance
        budget s^2 (== the true forward marginal C_kk, DPLR or not) --
        the same quantity this always logged, kept as-is since
        eval/traj_reporting.py's diagonal-Gaussian KL approximation wants
        each dim's actual marginal variance. diag_logvars_f is the private
        diagonal d alone, and rho_f is the correlated fraction (1 - d/s^2)
        DPLR redistributes into V. When DPLR is disabled, d == s^2 exactly,
        so diag_logvars_f == logvars_f and rho_f == 0.
        """
        gauss_params['means_b'][:, i, :] = back_drift.detach()
        gauss_params['logvars_b'][:, i, :] = (back_var / dts[:, None]).log().detach()
        gauss_params['means_f'][:, i, :] = fwd_drift.detach()
        gauss_params['logvars_f'][:, i, :] = pflogvars.detach()
        gauss_params['diag_logvars_f'][:, i, :] = d.detach().log()
        gauss_params['rho_f'][:, i, :] = (1.0 - d / pflogvars.exp()).detach()

    def _maybe_scramble_condition_embedding(self, condition_embedding, batch_size,
                                            scramble_condition_tiles: int):
        """
        Unconditional-prior training (the scramble_conditions stage flag,
        gated by Modeller.scramble_applicable): the
        conditioner runs on the TRUE, correctly-paired conditions -- so the
        state model sees embeddings of exactly the scale and distribution
        later (conditional) phases will feed it -- but its output is detached
        (the conditioner itself must stay at init) and its ROWS are permuted
        tile-wise before the state model ever consumes them, so MLE/TBC
        actively train the trunk to ignore the embedding. Tile size = the
        caller's K-repeats grouping: same-terminal rollouts keep sharing one
        (scrambled) condition, preserving the exact-MLE (IWAE) and TBC group
        semantics. Deliberately contained HERE, at the conditioner->trunk
        seam, so no scrambled tensor can leak back into buffers, per-sample
        losses, or condition bookkeeping upstream. 0 = off (default).
        """
        if scramble_condition_tiles <= 0:
            return condition_embedding
        k = scramble_condition_tiles
        assert batch_size % k == 0, \
            f"batch size {batch_size} not divisible by scramble tile size {k}"
        perm = torch.randperm(batch_size // k, device=condition_embedding.device)
        return (condition_embedding.detach()
                .reshape(batch_size // k, k, -1)[perm]
                .reshape(batch_size, -1))

    def get_condition_embedding(self, condition, mol_batch):
        if self.conditions_type == 'molecule':
            scalar_embedding, vector_embedding = self.conditions_embedding_model(
                mol_batch.z,
                mol_batch.pos,
                mol_batch.batch,
                mol_batch.ptr,
                mol_batch.num_graphs,
                condition
            )
        elif self.conditions_type == 'vector':
            scalar_embedding = self.conditions_embedding_model(condition)
        else:
            assert False, "invalid condition type"

        return scalar_embedding

    def get_traj_bwd(self, terminal_state, discretizer, condition, mol_batch,
                     return_gauss_params: bool = False, detach_traj: bool = False,
                     freeze_policy: bool = False, scramble_condition_tiles: int = 0):
        batch_size = terminal_state.shape[0]
        ts = discretizer(batch_size).to(self.device)
        trajectory_length = ts.shape[1] - 1

        logpb, logpf, states, gauss_params, log_flow = self.init_traj_tensors(batch_size, trajectory_length)

        states[:, -1] = terminal_state.detach()
        current_state = terminal_state.detach()

        if self.conditional:
            if condition is not False:
                condition_embedding = self.get_condition_embedding(condition, mol_batch)
            else:  # constant embedding
                condition_embedding = torch.zeros((batch_size, self.condition_embedding_dim),
                                                  dtype=torch.float32, device=self.device)
            condition_embedding = self._maybe_scramble_condition_embedding(
                condition_embedding, batch_size, scramble_condition_tiles)
            if freeze_policy:  # Z-only training: cut gradient to the conditioner at the source
                condition_embedding = condition_embedding.detach()
        else:
            condition_embedding = None

        use_ckpt = self._use_traj_checkpoint()
        # matches the old (i - 1) == 0 gating: a T=1 backward traj never wrote
        # the constant flow and left log_flow[:, 0] at zero
        if not self.full_flow and trajectory_length > 1:
            log_flow[:, 0] = self.flow_model(condition_embedding).flatten()

        for i in range(trajectory_length):
            dts = ts[:, trajectory_length - i] - ts[:, trajectory_length - i - 1]
            # pre-draw the step's noise at loop level (outside any checkpoint
            # region); the final step is deterministic and draws none
            eps = (torch.randn(batch_size, self.dim, dtype=current_state.dtype, device=self.device)
                   if i < trajectory_length - 1 else None)

            (prev_state, logpf_i, logpb_i, flow_i,
             back_drift, back_var, fwd_drift, pflogvars, d) = self._run_step(
                use_ckpt, self._bwd_step, current_state, dts, ts, condition_embedding,
                eps, i, trajectory_length, detach_traj)

            logpf.append(logpf_i)
            logpb.append(logpb_i)
            if self.full_flow:
                log_flow[:, trajectory_length - i - 1] = flow_i
            if return_gauss_params:
                self.log_gauss_params(gauss_params, i, back_drift, back_var, dts, fwd_drift, pflogvars, d)

            current_state = prev_state
            states[:, trajectory_length - i - 1] = current_state.detach()

        logpfs = torch.stack(logpf).T
        logpbs = torch.stack(logpb).T
        if return_gauss_params:
            return states, logpfs, logpbs, log_flow, {k: v.mean(-1) for k, v in gauss_params.items()}
        else:
            return states, logpfs, logpbs, log_flow

    def get_traj_replay(self, trajectory, discretizer, condition, mol_batch,
                        return_gauss_params: bool = False, freeze_policy: bool = False,
                        scramble_condition_tiles: int = 0):
        """
        Recompute log_flow, logpf and logpb for a fixed batch of trajectories
        (e.g., replayed from a buffer), instead of generating them. Mirrors
        get_traj_fwd's semantics exactly, but reads states from `trajectory`
        rather than sampling them, so the output is naturally detached from
        the state-generating computation graph (only the policy/flow model
        evaluations carry gradient).

        trajectory: [batch_size, trajectory_length + 1, dim]
        """
        trajectory = trajectory.detach()
        batch_size = trajectory.shape[0]
        ts = discretizer(batch_size).to(self.device)
        trajectory_length = ts.shape[1] - 1
        assert trajectory.shape[1] == trajectory_length + 1, \
            f"trajectory has {trajectory.shape[1]} states, expected {trajectory_length + 1}"

        logpb, logpf, states, gauss_params, log_flow = self.init_traj_tensors(batch_size, trajectory_length)

        states = trajectory
        current_state = states[:, 0]

        if self.conditional:
            if condition is not False:
                condition_embedding = self.get_condition_embedding(condition, mol_batch)
            else:  # constant embedding
                condition_embedding = torch.zeros((batch_size, self.condition_embedding_dim),
                                                  dtype=torch.float32, device=self.device)
            condition_embedding = self._maybe_scramble_condition_embedding(
                condition_embedding, batch_size, scramble_condition_tiles)
            if freeze_policy:  # Z-only training: cut gradient to the conditioner at the source
                condition_embedding = condition_embedding.detach()
        else:
            condition_embedding = None

        use_ckpt = self._use_traj_checkpoint()
        if not self.full_flow:
            log_flow[:, 0] = self.flow_model(condition_embedding).flatten()

        for i in range(trajectory_length):
            dts = ts[:, i + 1] - ts[:, i]
            next_state = states[:, i + 1]

            (logpf_i, logpb_i, flow_i,
             back_drift, back_var, fwd_drift, pflogvars, d) = self._run_step(
                use_ckpt, self._replay_step, current_state, next_state, dts, ts, condition_embedding, i)

            logpf.append(logpf_i)
            logpb.append(logpb_i)
            if self.full_flow:
                log_flow[:, i] = flow_i
            if return_gauss_params:
                self.log_gauss_params(gauss_params, i, back_drift, back_var, dts, fwd_drift, pflogvars, d)

            current_state = next_state

        logpfs = torch.stack(logpf).T
        logpbs = torch.stack(logpb).T
        if return_gauss_params:
            return states, logpfs, logpbs, log_flow, {k: v.mean(-1) for k, v in gauss_params.items()}
        else:
            return states, logpfs, logpbs, log_flow

    def _eval_pb_logprob(self, condition_embedding, i, current_state, next_state, dts, ts, fallback_logpf):
        """
        P_B step log-prob for the forward-direction transition current_state
        -> next_state at forward step i. Shared by get_traj_fwd and
        get_traj_replay (both walk forward through the same time grid);
        get_traj_bwd computes its own logpb directly since it's driven by
        backward propagation rather than scored against a given transition.
        """
        expanded_next_state = self.expand_state_for_policy(next_state)
        back_mean_correction, back_var_correction = self.fwd_get_back_correction(
            condition_embedding, i, expanded_next_state, ts)
        t_prev, t_next = ts[:, i], ts[:, i + 1]
        back_drift = -next_state * self.var_drift_coeff(t_prev, t_next, dts).unsqueeze(1) * back_mean_correction
        if i > 0:  # variance is exactly zero for the first step, so we can't use it
            var = (back_var_correction + self.var_log_rate(t_prev, t_next, dts)).clip(
                min=-self.var_clip, max=self.var_clip).exp()
            back_var = var * self.var_bridge_step(t_prev, t_next, dts).unsqueeze(1)
            logpb_i = self.gauss_logprob(current_state - next_state, back_drift, back_var)
        else:  # instead set this as a constant the model will have to learn around
            back_var = torch.ones_like(back_drift) * 1e-3 * dts.unsqueeze(1)
            logpb_i = torch.zeros_like(fallback_logpf)
        return back_drift, back_var, logpb_i

    def fwd_get_back_correction(self, condition_embedding, i, expanded_next_state, ts):
        if self.learn_pb:
            t_emb = self.t_model(ts[:, i + 1])
            pbs = self.backward_policy(self.s_model(expanded_next_state, condition_embedding), t_emb)
            dmean, dvar = gaussian_params(pbs)
            back_mean_correction = 1 + torch.tanh(dmean / self.pb_drift_range) * self.pb_drift_range

            if self.learned_variance:
                back_additive_logvar = torch.tanh(dvar / self.pb_var_range) * self.pb_var_range

                # back_var_correction = (1 + torch.tanh(dvar/self.pb_scale_range) * self.pb_scale_range)
            else:
                back_additive_logvar = torch.zeros_like(dvar)

        else:
            back_mean_correction, back_additive_logvar = (torch.ones((len(expanded_next_state), self.dim),
                                                                     dtype=torch.float32, device=self.device),
                                                          torch.zeros((len(expanded_next_state), self.dim),
                                                                      dtype=torch.float32, device=self.device)
                                                          )
        return back_mean_correction, back_additive_logvar

    def fwd_propagate(self, current_state, detach_traj, dts, pf_mean, pflogvars_sample, V_sample=None,
                      eps=None, eps_r=None):
        # eps ~ N(0, I_n): diagonal contribution; eps_r ~ N(0, I_r): low-rank contribution (if DPLR is active).
        # The trajectory loops pre-draw and pass both (checkpoint-safe: the step must
        # be deterministic given its inputs); drawn here only if a caller omits them.
        if eps is None:
            eps = torch.randn_like(current_state, device=self.device)
        noise = (pflogvars_sample / 2).exp() * eps
        if V_sample is not None:
            if eps_r is None:
                eps_r = torch.randn(current_state.shape[0], V_sample.shape[-1], device=self.device)
            noise = noise + torch.einsum('bnr,br->bn', V_sample, eps_r)
        if detach_traj:
            next_state = (current_state +
                          dts.unsqueeze(1) * pf_mean.detach() +
                          dts.sqrt().unsqueeze(1) * noise)
        else:
            next_state = (current_state +
                          dts.unsqueeze(1) * pf_mean +
                          dts.sqrt().unsqueeze(1) * noise)
        return next_state

    def fwd_get_logvars(self, detach_traj, dts, exploration_std, i, pflogvars):
        # log_coeff: (batch_size,) per-trajectory log-multiplier on policy std
        #   log_coeff = 0 → no change (multiplier 1)
        #   log_coeff > 0 → widen by exp(log_coeff)
        if exploration_std is not None:
            pflogvars_sample = pflogvars + 2 * exploration_std.unsqueeze(-1)
        else:
            pflogvars_sample = pflogvars
        if detach_traj:
            return pflogvars_sample.detach()
        else:
            return pflogvars_sample

    def bwd_propagate(self, back_mean, back_var, current_state, detach_traj, eps=None):
        # eps is pre-drawn by the trajectory loop (checkpoint-safe); drawn here only if omitted
        if eps is None:
            eps = torch.randn_like(current_state, device=self.device)
        if detach_traj:
            s_ = back_mean.detach() + back_var.sqrt().detach() * eps
        else:
            s_ = back_mean + back_var.sqrt() * eps
        return s_

    def get_bwd_correction(self, condition_embedding, expanded_current_state, i, trajectory_length, ts):
        if self.learn_pb:
            t = self.t_model(ts[:, trajectory_length - i])
            pbs = self.backward_policy(self.s_model(expanded_current_state, condition_embedding), t)
            dmean, dvar = gaussian_params(pbs)
            back_mean_correction = 1 + torch.tanh(dmean / self.pb_drift_range) * self.pb_drift_range

            if self.learned_variance:
                back_additive_logvar = torch.tanh(dvar / self.pb_var_range) * self.pb_var_range
            else:
                back_additive_logvar = torch.zeros_like(dvar)

        else:
            back_mean_correction, back_additive_logvar = (torch.ones((len(expanded_current_state), self.dim),
                                                                     dtype=torch.float32, device=self.device),
                                                          torch.zeros((len(expanded_current_state), self.dim),
                                                                      dtype=torch.float32, device=self.device)
                                                          )
        return back_mean_correction, back_additive_logvar

    # keys of the optional per-step diagnostic dict (see log_gauss_params)
    GAUSS_PARAM_KEYS = ('means_f', 'logvars_f', 'diag_logvars_f', 'rho_f', 'means_b', 'logvars_b')

    def init_traj_tensors(self, batch_size, trajectory_length):
        logpf = []
        logpb = []
        states = torch.zeros((batch_size, trajectory_length + 1, self.dim), device=self.device)
        gauss_params = {key: torch.zeros((batch_size, trajectory_length, self.dim), device=self.device)
                        for key in self.GAUSS_PARAM_KEYS}
        log_flow = torch.zeros((batch_size, trajectory_length + 1), device=self.device)

        return logpb, logpf, states, gauss_params, log_flow

    def wrap_to_pi(self, x):
        # (-pi, pi]
        return (x + torch.pi) % (2 * torch.pi) - torch.pi

    def expand_state_for_policy(self, state):
        lin = state.index_select(-1, self.lin_idx)  # [B, lin_dim]
        ang = state.index_select(-1, self.ang_idx) * torch.pi  # [B, ang_dim]  # latent space is natively defined on [-1, 1]
        sin, cos = torch.sin(ang), torch.cos(ang)  # [B, ang_dim] each
        orient = torch.stack([sin, cos], dim=-1).reshape(state.size(0), self.ang_dim * 2)
        return torch.cat([lin, orient], dim=-1)  # [B, expanded_dim]

    def gauss_logprob(self, delta_x, drift, var):
        noise = (delta_x - drift) / var.sqrt()
        # noise_raw[:, self.ang_mask] = self.wrap_to_pi(noise_raw[:, self.ang_mask])

        return -0.5 * (noise ** 2 + logtwopi + var.log()).sum(1)

    def fwd_gauss_logprob(self, delta_x, drift, d, dt, V=None):
        """
        Log density of delta_x under N(drift, dt * C), with C = diag(d), or,
        if V ([B, n, r]) is given, C = diag(d) + V @ V^T (DPLR forward
        covariance). Reduces exactly to the gauss_logprob diagonal path when
        V is None. Uses the Woodbury identity / matrix determinant lemma so
        no n x n solve/determinant is needed: cost is O(n r^2 + r^3).
        """
        if V is None:
            var = dt.unsqueeze(1) * d
            return self.gauss_logprob(delta_x, drift, var)

        z = delta_x - drift
        r = V.shape[-1]
        Dinv_V = V / d.unsqueeze(-1)  # [B, n, r]
        M = torch.eye(r, device=V.device, dtype=V.dtype) + torch.einsum('bnr,bns->brs', V, Dinv_V)
        w = torch.einsum('bnr,bn->br', Dinv_V, z)  # V^T D^-1 z

        L = torch.linalg.cholesky(M)
        Minv_w = torch.cholesky_solve(w.unsqueeze(-1), L).squeeze(-1)
        quad_correction = (w * Minv_w).sum(-1)  # w^T M^-1 w
        logdet_M = 2 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)

        # C >= D (PSD add) guarantees this is >= 0; clamp only guards fp round-off
        quad = (z.pow(2) / d).sum(-1) - quad_correction
        quad = quad.clamp(min=0)

        n = d.shape[-1]
        logdet_C = d.log().sum(-1) + logdet_M
        return -0.5 * (quad / dt + n * logtwopi + n * dt.log() + logdet_C)
