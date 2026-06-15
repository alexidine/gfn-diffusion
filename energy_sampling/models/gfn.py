import math
from argparse import Namespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from energy_sampling.utils import gaussian_params
from mxtaltools.models.graph_models.molecule_graph_model import VectorMoleculeGraphModel, ScalarMoleculeGraphModel
from mxtaltools.models.modules.components import scalarMLP, vectorMLP
from .architectures import FlowModel, NoneModule, LearnableScalar, TimeEncoding, StateEncoding, PolicyModel

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
        self.log_var_range = log_var_range
        self.var_clip = 16
        self.device = device
        self.max_z_prime = max_z_prime
        self.conditions_type = conditions_type

        self.get_periodic_dimensions(device)

        self.init_conditioner(cond_hidden_dim, cond_layers, condition_embedding_dim, conditions_dim,
                              dropout, norm)

        self.init_flow_model(condition_embedding_dim, dropout, flow_hidden_dim, flow_layers,
                             norm)

        self.t_model = TimeEncoding(harmonics_dim, t_dim, t_hidden_dim,
                                    norm=norm, dropout=dropout)
        self.s_model = StateEncoding(self.expanded_dim, s_layers, s_hidden_dim, condition_embedding_dim, s_emb_dim,
                                     norm=norm, dropout=dropout)
        self.forward_policy = PolicyModel(dim, s_emb_dim, t_dim,
                                          policy_hidden_dim, policy_layers, 2 * dim,
                                          zero_init=zero_init,
                                          norm=norm, dropout=dropout)
        self.backward_policy = PolicyModel(dim, s_emb_dim, t_dim,
                                           policy_hidden_dim, policy_layers, 2 * dim,
                                           zero_init=zero_init,
                                           norm=norm, dropout=dropout)

        self.pb_drift_range = pb_drift_range
        self.pb_var_range = pb_var_range

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

    def init_flow_model(self, condition_embedding_dim, dropout, flow_hidden_dim, flow_layers, norm):
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

    def get_periodic_dimensions(self, device):
        angs = [False] * 6
        for zp in range(self.max_z_prime):
            angs.extend([False, False, False])
        for zp in range(self.max_z_prime):
            angs.extend([False, True, True])
            # phi and r dimensions arein rotational basis
        self.ang_mask = torch.tensor(angs, device=device)
        self.ang_dim = (self.ang_mask == True).sum().item()
        self.lin_dim = self.dim - self.ang_dim
        self.expanded_dim = self.lin_dim + self.ang_dim * 2

    def split_params(self, tensor):
        mean, logvar_i = gaussian_params(tensor)
        if not self.learned_variance:
            logvar = torch.zeros_like(logvar_i)
        else:
            if self.log_var_range == -1:
                logvar = logvar_i
            else:
                logvar = torch.tanh(logvar_i / self.log_var_range) * self.log_var_range
        return mean, (logvar + np.log(self.pf_std_per_traj) * 2.0).clip(min=-self.var_clip, max=self.var_clip)

    def predict_next_state(self, s, t, condition_embedding):
        s_new = self.forward_policy(self.s_model(s, condition_embedding), self.t_model(t))

        if self.clipping:
            s_new = torch.clip(s_new, -self.gfn_clip, self.gfn_clip)

        return s_new

    def get_traj_fwd(self, initial_state, discretizer, exploration_std, condition, mol_batch,
                     return_gauss_params: bool = False, detach_traj: bool = True):
        batch_size = initial_state.shape[0]
        ts = discretizer(batch_size).to(self.device)
        trajectory_length = ts.shape[1] - 1
        logpb, logpf, states, means_f, logvars_f, means_b, logvars_b = self.init_traj_tensors(batch_size,
                                                                                              trajectory_length)

        initial_state.requires_grad_(not detach_traj)
        current_state = initial_state
        states[:, 0] = current_state

        condition_embedding = self.get_condition_embedding(condition, mol_batch)
        logf = self.flow_model(condition_embedding).flatten()

        for i in range(trajectory_length):
            dts = ts[:, i + 1] - ts[:, i]

            # PROPAGATION
            expanded_current_state = self.expand_state_for_policy(current_state)
            state_update = self.predict_next_state(expanded_current_state, ts[:, i], condition_embedding)
            pf_mean, pflogvars = self.split_params(state_update)
            pflogvars_sample = self.fwd_get_logvars(detach_traj, dts, exploration_std, i, pflogvars)
            next_state = self.fwd_propagate(current_state, detach_traj, dts, pf_mean, pflogvars_sample)

            # noise = ((next_state - current_state) - dts.unsqueeze(1) * pf_mean) / (
            #        dts.sqrt().unsqueeze(1) * (pflogvars / 2).exp())

            # compute forward logprobs
            fwd_drift = dts.unsqueeze(1) * pf_mean
            fwd_var = dts.unsqueeze(1) * pflogvars.exp()
            logpf.append(self.gauss_logprob(next_state - current_state, fwd_drift, fwd_var))
            # -0.5 * (noise ** 2 + logtwopi + dts.log().unsqueeze(1) + pflogvars).sum(1))

            # compute backward logprobs
            expanded_next_state = self.expand_state_for_policy(next_state)
            back_mean_correction, back_var_correction = self.fwd_get_back_correction(
                condition_embedding, i, expanded_next_state, ts)
            back_drift = -next_state * (dts / ts[:, i + 1]).unsqueeze(1) * back_mean_correction
            if i > 0:  # variance is exactly zero for the first step, so we can't use it
                var = (back_var_correction + np.log(self.pf_std_per_traj) * 2.0).clip(min=-self.var_clip,
                                                                                      max=self.var_clip).exp()
                back_var = var * (dts * ts[:, i] / ts[:, i + 1]).unsqueeze(1)
                logpb.append(self.gauss_logprob(current_state - next_state, back_drift, back_var))

            else:  # instead set this as a constant the model will have to learn around
                back_var = torch.ones_like(back_drift) * 1e-3 * dts.unsqueeze(1)

            current_state = next_state
            # only wrap after logprob calculations
            if self.ang_dim > 0:
                current_state[:, self.ang_mask] = self.wrap_to_pi(
                    current_state[:, self.ang_mask] * torch.pi) / torch.pi  # latent space is on [-1, 1]
            states[:, i + 1] = current_state

            if return_gauss_params:
                self.log_gauss_params(back_drift, back_var, dts, fwd_drift, i,
                                      logvars_b, logvars_f, means_b, means_f,
                                      pflogvars)

        logpfs = torch.stack(logpf).T
        logpbs = torch.stack(logpb).T
        if return_gauss_params:
            return (states, logpfs, logpbs, logf,
                    means_f.mean(-1), logvars_f.mean(-1),
                    means_b.mean(-1), logvars_b.mean(-1))
        else:
            return states, logpfs, logpbs, logf

    def log_gauss_params(self, back_drift, back_var, dts, fwd_drift, i, logvars_b, logvars_f, means_b, means_f,
                         pflogvars):
        means_b[:, i, :] = back_drift.detach()
        logvars_b[:, i, :] = (back_var / dts[:, None]).log().detach()
        means_f[:, i, :] = fwd_drift.detach()
        logvars_f[:, i, :] = pflogvars.detach()

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
                     return_gauss_params: bool = False, detach_traj: bool = True):
        batch_size = terminal_state.shape[0]
        ts = discretizer(batch_size).to(self.device)
        trajectory_length = ts.shape[1] - 1

        logpb, logpf, states, means_f, logvars_f, means_b, logvars_b = (
            self.init_traj_tensors(batch_size, trajectory_length))

        states[:, -1] = terminal_state.detach()
        current_state = terminal_state.detach()

        condition_embedding = self.get_condition_embedding(condition, mol_batch)

        logf = self.flow_model(condition_embedding).flatten()
        log_var_base = 2.0 * math.log(float(self.pf_std_per_traj))

        for i in range(trajectory_length):
            dts = ts[:, trajectory_length - i] - ts[:, trajectory_length - i - 1]

            if i < trajectory_length - 1:
                # backward propagate
                expanded_current_state = self.expand_state_for_policy(current_state)

                back_mean_correction, back_var_correction \
                    = self.bwd_get_correction(
                    condition_embedding, expanded_current_state, i, trajectory_length, ts)

                # directly x_t-u\Delta t
                back_mean = (current_state -
                             current_state * (dts / ts[:, trajectory_length - i]).unsqueeze(1) * back_mean_correction)

                back_drift = - current_state * (dts / ts[:, trajectory_length - i]).unsqueeze(1) * back_mean_correction
                var = (back_var_correction + log_var_base).clip(min=-self.var_clip, max=self.var_clip).exp()
                back_var = var * (dts * ts[:, trajectory_length - i - 1] / ts[:, trajectory_length - i]).unsqueeze(1)
                prev_state = self.bwd_propagate(back_mean, back_var, current_state, detach_traj)

                # log Pb calculation
                # noise_backward = (prev_state - back_mean) / back_var.sqrt()
                # logpb.append(-0.5 * (noise_backward ** 2 + logtwopi + back_var.log()).sum(1))
                logpb.append(self.gauss_logprob(prev_state - current_state, back_drift, back_var))
            else:
                # send it identically to the source state (0)
                prev_state = torch.zeros_like(current_state)
                back_drift = -current_state
                # back_mean = prev_state  # remember the back_mean in pb is for some reason the actual mean not the drift
                back_var = torch.ones_like(back_drift) * 1e-3 * dts.unsqueeze(1)

            """log pf calculation"""
            expanded_prev_state = self.expand_state_for_policy(prev_state)
            pfs = self.predict_next_state(expanded_prev_state, ts[:, trajectory_length - i - 1], condition_embedding)
            pf_mean, pflogvars = self.split_params(pfs)

            # noise = ((current_state - prev_state) - dts.unsqueeze(1) * pf_mean) / (
            #         dts.sqrt().unsqueeze(1) * (pflogvars / 2).exp())
            #
            # logpf.append(-0.5 * (
            #         noise ** 2 + logtwopi + dts.log().unsqueeze(1) + pflogvars).sum(
            #     1))

            fwd_drift = dts.unsqueeze(1) * pf_mean
            fwd_var = dts.unsqueeze(1) * pflogvars.exp()

            logpf.append(self.gauss_logprob(current_state - prev_state, fwd_drift, fwd_var))

            if return_gauss_params:
                self.log_gauss_params(back_drift, back_var, dts, fwd_drift, i, logvars_b, logvars_f, means_b, means_f,
                                      pflogvars)

            current_state = prev_state
            if self.ang_dim > 0:
                current_state[:, self.ang_mask] = self.wrap_to_pi(
                    current_state[:, self.ang_mask] * torch.pi) / torch.pi  # latent space is on [-1, 1]
            states[:, trajectory_length - i - 1] = current_state.detach()

        logpfs = torch.stack(logpf).T
        logpbs = torch.stack(logpb).T
        if return_gauss_params:
            return (states, logpfs, logpbs, logf,
                    means_f.mean(-1), logvars_f.mean(-1),
                    means_b.mean(-1), logvars_b.mean(-1))
        else:
            return states, logpfs, logpbs, logf

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

    def fwd_propagate(self, current_state, detach_traj, dts, pf_mean, pflogvars_sample):
        if detach_traj:
            next_state = (current_state +
                          dts.unsqueeze(1) * pf_mean.detach() +
                          dts.sqrt().unsqueeze(1) * (pflogvars_sample / 2).exp() * torch.randn_like(current_state,
                                                                                                    device=self.device))
        else:
            next_state = (current_state +
                          dts.unsqueeze(1) * pf_mean +
                          dts.sqrt().unsqueeze(1) * (pflogvars_sample / 2).exp() * torch.randn_like(current_state,
                                                                                                    device=self.device))
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

    def bwd_propagate(self, back_mean, back_var, current_state, detach_traj):
        if detach_traj:
            s_ = (back_mean.detach() +
                  back_var.sqrt().detach() * torch.randn_like(current_state, device=self.device))
        else:
            s_ = (back_mean +
                  back_var.sqrt() * torch.randn_like(current_state, device=self.device))
        return s_

    def bwd_get_correction(self, condition_embedding, expanded_current_state, i, trajectory_length, ts):
        if self.learn_pb:
            t = self.t_model(ts[:, trajectory_length - i])
            pbs = self.backward_policy(self.s_model(expanded_current_state, condition_embedding), t)
            dmean, dvar = gaussian_params(pbs)
            back_mean_correction = 1 + dmean.tanh() * self.pb_drift_range

            if self.learned_variance:
                back_additive_logvar = torch.tanh(dvar / self.pb_var_range) * self.pb_var_range

                # back_var_correction = (1 + torch.tanh(dvar/self.pb_scale_range) * self.pb_scale_range)
            else:
                back_additive_logvar = torch.zeros_like(dvar)

        else:
            # back_mean_correction, back_additive_logvar = torch.ones_like(current_state), torch.zeros_like(current_state)
            back_mean_correction, back_additive_logvar = (torch.ones((len(expanded_current_state), self.dim),
                                                                     dtype=torch.float32, device=self.device),
                                                          torch.zeros((len(expanded_current_state), self.dim),
                                                                      dtype=torch.float32, device=self.device)
                                                          )
        return back_mean_correction, back_additive_logvar

    def init_traj_tensors(self, batch_size, trajectory_length):
        logpf = []
        logpb = []
        states = torch.zeros((batch_size, trajectory_length + 1, self.dim), device=self.device)
        means_f = torch.zeros((batch_size, trajectory_length, self.dim), device=self.device)
        logvars_f = torch.zeros((batch_size, trajectory_length, self.dim), device=self.device)
        means_b = torch.zeros((batch_size, trajectory_length, self.dim), device=self.device)
        logvars_b = torch.zeros((batch_size, trajectory_length, self.dim), device=self.device)

        return logpb, logpf, states, means_f, logvars_f, means_b, logvars_b

    def wrap_to_pi(self, x):
        # (-pi, pi]
        return (x + torch.pi) % (2 * torch.pi) - torch.pi

    def expand_state_for_policy(self, state):
        lin = state[..., ~self.ang_mask]  # [B, 10]
        ang = state[..., self.ang_mask] * torch.pi  # [B, 2]  # latent space is natively defined on [-1, 1]
        sin, cos = torch.sin(ang), torch.cos(ang)  # [B, 2] each
        orient = torch.stack([sin, cos], dim=-1).reshape(state.size(0), self.ang_dim * 2)  # [B, 6]
        return torch.cat([lin, orient], dim=-1)  # [B, 6 + 8*zp]

    def gauss_logprob(self, delta_x, drift, var):
        noise = (delta_x - drift) / var.sqrt()
        # noise_raw[:, self.ang_mask] = self.wrap_to_pi(noise_raw[:, self.ang_mask])

        return -0.5 * (noise ** 2 + logtwopi + var.log()).sum(1)
