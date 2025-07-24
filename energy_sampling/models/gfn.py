from typing import Optional

import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .architectures import FlowModel, NoneModule, LearnableScalar, TimeEncoding, StateEncoding, PolicyModel
from utils import gaussian_params, get_gfn_init_state
from mxtaltools.models.modules.components import scalarMLP

logtwopi = math.log(2 * math.pi)


class GFN(nn.Module):
    def __init__(self, dim: int, s_emb_dim: int, hidden_dim: int, conditions_dim: int,
                 harmonics_dim: int, t_dim: int, log_var_range: float = 4.,
                 t_scale: float = 1., learned_variance: bool = True,
                 trajectory_length: int = 100,
                 condition_embedding_dim: int = 32,
                 clipping: bool = False,
                 gfn_clip: float = 1e4, pb_scale_range: float = 1.,
                 conditional_flow_model: bool = False,
                 learn_pb: bool = False, joint_layers: int = 2,
                 dropout: Optional[float] = 0, norm: Optional[str] = None,
                 zero_init: bool = False, device=torch.device('cuda')):
        super(GFN, self).__init__()
        self.dim = dim
        self.harmonics_dim = harmonics_dim
        self.t_dim = t_dim
        self.s_emb_dim = s_emb_dim

        self.trajectory_length = trajectory_length
        self.learned_variance = learned_variance
        self.t_scale = t_scale

        self.clipping = clipping
        self.gfn_clip = gfn_clip

        self.conditional_flow_model = conditional_flow_model
        self.learn_pb = learn_pb

        self.joint_layers = joint_layers

        self.pf_std_per_traj = np.sqrt(self.t_scale)
        self.dt = 1. / trajectory_length
        self.log_var_range = log_var_range
        self.device = device

        if self.conditional_flow_model:
            self.conditions_embedding_model = scalarMLP(input_dim=conditions_dim,
                                                        norm=norm,
                                                        dropout=dropout,
                                                        layers=joint_layers,
                                                        filters=hidden_dim,
                                                        output_dim=condition_embedding_dim,
                                                        )
            self.flow_model = FlowModel(condition_embedding_dim,
                                        hidden_dim,
                                        2,
                                        norm=norm,
                                        dropout=dropout,
                                        )
        else:  # we can pass arguments to this (conditions) but nothing will happen
            self.conditions_embedding_model = NoneModule()  # ditto
            self.flow_model = LearnableScalar()  # unified syntax with this instead of nn.Parameter
            condition_embedding_dim = 0

        self.t_model = TimeEncoding(harmonics_dim, t_dim, hidden_dim,
                                    norm=norm, dropout=dropout)
        self.s_model = StateEncoding(dim, joint_layers, hidden_dim, condition_embedding_dim, s_emb_dim,
                                     norm=norm, dropout=dropout)
        self.forward_policy = PolicyModel(dim, s_emb_dim, t_dim,
                                          hidden_dim, joint_layers, 2 * dim, zero_init=zero_init,
                                          norm=norm, dropout=dropout)
        self.backward_policy = PolicyModel(dim, s_emb_dim, t_dim,
                                           hidden_dim, joint_layers, 2 * dim, zero_init=zero_init,
                                           norm=norm, dropout=dropout)

        self.pb_scale_range = pb_scale_range

    def split_params(self, tensor):
        mean, logvar_i = gaussian_params(tensor)
        if not self.learned_variance:
            logvar = torch.zeros_like(logvar_i)
        else:
            if self.log_var_range == -1:
                logvar = logvar_i
            else:
                logvar = torch.tanh(logvar_i / self.log_var_range) * self.log_var_range
        return mean, (logvar + np.log(self.pf_std_per_traj) * 2.0).clip(min=-8, max=8)

    #

    def predict_next_state(self, s, t, condition_embedding):
        t = self.t_model(t)
        s = self.s_model(s, condition_embedding)
        s_new = self.forward_policy(s, t)

        if self.clipping:
            s_new = torch.clip(s_new, -self.gfn_clip, self.gfn_clip)
        return s_new

    def get_trajectory_fwd(self, initial_state, discretizer, exploration_std, condition,
                           return_gauss_params: bool = False, detach_traj: bool = True):
        batch_size = initial_state.shape[0]

        ts = discretizer(batch_size).to(self.device)
        trajectory_length = ts.shape[1] - 1

        logf, logpb, logpf, states, means_f, logvars_f, means_b, logvars_b = self.init_traj_tensors(batch_size,
                                                                                                    trajectory_length)

        states[:, 0] = initial_state.clone().detach()  # set correct initial state
        current_state = initial_state.clone().detach()

        condition_embedding = self.conditions_embedding_model(condition)

        logf[:, 0] = self.flow_model(condition_embedding).squeeze(-1).squeeze(-1)

        for i in range(trajectory_length):
            dts = ts[:, i + 1] - ts[:, i]

            state_update = self.predict_next_state(current_state, ts[:, i], condition_embedding)
            pf_mean, pflogvars = self.split_params(state_update)
            pflogvars_sample = self.fwd_get_logvars(detach_traj, dts, exploration_std, i, pflogvars)
            next_state = self.fwd_propagate(current_state, detach_traj, dts, pf_mean, pflogvars_sample)

            noise = ((next_state - current_state) - dts.unsqueeze(1) * pf_mean) / (
                    dts.sqrt().unsqueeze(1) * (pflogvars / 2).exp())
            logpf[:, i] = -0.5 * (noise ** 2 + logtwopi + dts.log().unsqueeze(1) + pflogvars).sum(1)

            back_mean_correction, back_var_correction = self.fwd_get_back_correction(condition_embedding, i, next_state,
                                                                                     ts)
            back_mean = (next_state - next_state * (dts / ts[:, i + 1]).unsqueeze(1) * back_mean_correction)
            if i > 0:  # variance is exactly zero for the first step, so we can't use it
                back_var = ((self.pf_std_per_traj ** 2) * (dts * ts[:, i] / ts[:, i + 1]).unsqueeze(
                    1) * back_var_correction)
                noise_backward = (current_state - back_mean) / back_var.sqrt()
                logpb[:, i] = -0.5 * (noise_backward ** 2 + logtwopi + back_var.log()).sum(1)
            else:  # instead set this as a constant the model will have to learn around
                back_var = torch.ones_like(back_mean) * 1e-3 * dts.unsqueeze(1)

            current_state = next_state
            states[:, i + 1] = current_state

            if return_gauss_params:
                means_b[:, i, :] = back_mean - current_state
                logvars_b[:, i, :] = (back_var / dts[:, None]).log()
                means_f[:, i, :] = pf_mean * dts[:, None]
                logvars_f[:, i, :] = pflogvars

        if return_gauss_params:
            return (states, logpf, logpb, logf,
                    means_f.mean(-1), logvars_f.mean(-1),
                    means_b.mean(-1), logvars_b.mean(-1))
        else:
            return states, logpf, logpb, logf

    def get_trajectory_bwd(self, terminal_state, discretizer, condition,
                           return_gauss_params: bool = False, detach_traj: bool = False):
        batch_size = terminal_state.shape[0]

        ts = discretizer(batch_size).to(self.device)
        trajectory_length = ts.shape[1] - 1

        logf, logpb, logpf, states, means_f, logvars_f, means_b, logvars_b = self.init_traj_tensors(batch_size,
                                                                                                    trajectory_length)

        states[:, -1] = terminal_state.detach().clone()
        current_state = terminal_state.detach().clone()
        condition_embedding = self.conditions_embedding_model(condition)

        flow = self.flow_model(condition_embedding).squeeze(-1).squeeze(-1)
        logf[:, 0] = flow

        for i in range(trajectory_length):
            dts = ts[:, trajectory_length - i] - ts[:, trajectory_length - i - 1]

            if i < trajectory_length - 1:
                back_mean_correction, back_var_correction = self.bwd_get_correction(condition_embedding, current_state,
                                                                                    i, trajectory_length, ts)

                back_mean = (current_state -
                             current_state * (dts / ts[:, trajectory_length - i]).unsqueeze(1) * back_mean_correction)
                back_var = ((self.pf_std_per_traj ** 2) *
                            (dts * ts[:, trajectory_length - i - 1] / ts[:, trajectory_length - i]).unsqueeze(
                                1) * back_var_correction)

                prev_state = self.bwd_propagate(back_mean, back_var, current_state, detach_traj)
                noise_backward = (prev_state - back_mean) / back_var.sqrt()
                logpb[:, trajectory_length - i - 1] = -0.5 * (noise_backward ** 2 + logtwopi + back_var.log()).sum(1)
            else:
                # send it identically to the source state (0)
                prev_state = torch.zeros_like(current_state)
                back_mean = prev_state  # remember the back_mean in pb is for some reson the actual mean not the drift
                back_var = torch.ones_like(back_mean) * 1e-3 * dts.unsqueeze(1)

            pfs = self.predict_next_state(prev_state, ts[:, trajectory_length - i - 1], condition_embedding)
            pf_mean, pflogvars = self.split_params(pfs)

            noise = ((current_state - prev_state) - dts.unsqueeze(1) * pf_mean) / (
                    dts.sqrt().unsqueeze(1) * (pflogvars / 2).exp())
            logpf[:, trajectory_length - i - 1] = -0.5 * (
                    noise ** 2 + logtwopi + dts.log().unsqueeze(1) + pflogvars).sum(
                1)

            if return_gauss_params:
                means_b[:, i, :] = back_mean - current_state
                logvars_b[:, i, :] = (back_var / dts[:, None]).log()
                means_f[:, i, :] = pf_mean * dts[:, None]
                logvars_f[:, i, :] = pflogvars

            current_state = prev_state
            states[:, trajectory_length - i - 1] = current_state

        if return_gauss_params:
            return (states, logpf, logpb, logf,
                    means_f.mean(-1), logvars_f.mean(-1),
                    means_b.mean(-1), logvars_b.mean(-1))
        else:
            return states, logpf, logpb, logf

    def fwd_get_back_correction(self, condition_embedding, i, next_state, ts):
        if self.learn_pb:
            t_emb = self.t_model(ts[:, i + 1])
            pbs = self.backward_policy(self.s_model(next_state, condition_embedding), t_emb)
            dmean, dvar = gaussian_params(pbs)
            back_mean_correction = 1 + torch.tanh(dmean) * self.pb_scale_range
            if self.learned_variance:
                back_var_correction = (1 + torch.tanh(dvar) * self.pb_scale_range)
            else:
                back_var_correction = torch.ones_like(next_state)
        else:
            back_mean_correction, back_var_correction = torch.ones_like(next_state), torch.ones_like(next_state)
        return back_mean_correction, back_var_correction

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
        if exploration_std is not None:
            expl = exploration_std(None)
            if expl > 0:
                add_log_var = torch.full_like(pflogvars, np.log(exploration_std(i)) * 2) / dts.sqrt().unsqueeze(1)
                pflogvars_sample = torch.logaddexp(pflogvars, add_log_var)
            else:
                pflogvars_sample = pflogvars
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

    def bwd_get_correction(self, condition_embedding, current_state, i, trajectory_length, ts):
        if self.learn_pb:
            t = self.t_model(ts[:, trajectory_length - i])
            pbs = self.backward_policy(self.s_model(current_state, condition_embedding), t)
            dmean, dvar = gaussian_params(pbs)
            back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
            if self.learned_variance:
                back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
            else:
                back_var_correction = torch.ones_like(current_state)
        else:
            back_mean_correction, back_var_correction = torch.ones_like(current_state), torch.ones_like(
                current_state)
        return back_mean_correction, back_var_correction


    def init_traj_tensors(self, batch_size, trajectory_length):
        logpf = torch.zeros((batch_size, trajectory_length), device=self.device)
        logpb = torch.zeros((batch_size, trajectory_length), device=self.device)
        logf = torch.zeros((batch_size, trajectory_length + 1), device=self.device)
        states = torch.zeros((batch_size, trajectory_length + 1, self.dim), device=self.device)
        means_f = torch.zeros((batch_size, trajectory_length, self.dim), device=self.device)
        logvars_f = torch.zeros((batch_size, trajectory_length, self.dim), device=self.device)
        means_b = torch.zeros((batch_size, trajectory_length, self.dim), device=self.device)
        logvars_b = torch.zeros((batch_size, trajectory_length, self.dim), device=self.device)

        return logf, logpb, logpf, states, means_f, logvars_f, means_b, logvars_b

    def sample(self, batch_size, log_r, condition=None):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, None, condition)[0][:, -1]

    def sleep_phase_sample(self, batch_size, exploration_std, condition=None):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, exploration_std, condition=condition)[0][:, -1]

    def forward(self, s, exploration_std=None, log_r=None, condition=None):
        return self.get_trajectory_fwd(s, exploration_std, condition)
