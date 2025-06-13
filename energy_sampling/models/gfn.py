import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .architectures import *
from utils import gaussian_params, get_gfn_init_state
from mxtaltools.models.modules.components import scalarMLP

logtwopi = math.log(2 * math.pi)


class GFN(nn.Module):
    def __init__(self, dim: int, s_emb_dim: int, hidden_dim: int, conditions_dim: int,
                 harmonics_dim: int, t_dim: int, log_var_range: float = 4.,
                 t_scale: float = 1., learned_variance: bool = True,
                 trajectory_length: int = 100, partial_energy: bool = False,
                 condition_embedding_dim: int = 32,
                 clipping: bool = False,
                 gfn_clip: float = 1e4, pb_scale_range: float = 1.,
                 conditional_flow_model: bool = False,
                 learn_pb: bool = False, lgv_layers: int = 3, joint_layers: int = 2,
                 dropout: Optional[float] = 0, norm: Optional[str] = None,
                 zero_init: bool = False, device=torch.device('cuda')):
        super(GFN, self).__init__()
        self.dim = dim
        self.harmonics_dim = harmonics_dim
        self.t_dim = t_dim
        self.s_emb_dim = s_emb_dim

        self.trajectory_length = trajectory_length
        self.learned_variance = learned_variance
        self.partial_energy = partial_energy
        self.t_scale = t_scale

        self.clipping = clipping
        self.gfn_clip = gfn_clip

        self.conditional_flow_model = conditional_flow_model
        self.learn_pb = learn_pb

        self.lgv_layers = lgv_layers
        self.joint_layers = joint_layers

        self.pf_std_per_traj = np.sqrt(self.t_scale)
        self.dt = 1. / trajectory_length
        self.log_var_range = log_var_range
        self.device = device

        self.t_model = TimeEncoding(harmonics_dim, t_dim, hidden_dim,
                                    norm=norm, dropout=dropout)
        self.s_model = StateEncoding(dim, hidden_dim, condition_embedding_dim, s_emb_dim,
                                     norm=norm, dropout=dropout)
        self.joint_model = JointPolicy(dim, s_emb_dim, t_dim,
                                       hidden_dim, joint_layers, 2 * dim, zero_init=zero_init,
                                       norm=norm, dropout=dropout)
        if learn_pb:
            self.back_model = JointPolicy(dim, s_emb_dim, t_dim, hidden_dim, joint_layers, 2 * dim, zero_init=zero_init,
                                          norm=norm, dropout=dropout)
        self.pb_scale_range = pb_scale_range

        if self.conditional_flow_model:
            self.conditions_embedding_model = scalarMLP(input_dim=conditions_dim, norm=None, dropout=0,
                                                        layers=1, filters=hidden_dim,
                                                        output_dim=condition_embedding_dim)
            self.flow_model = FlowModel(condition_embedding_dim, hidden_dim, 1,
                                        norm='layer', dropout=0)
        else:
            self.flow_model = torch.nn.Parameter(torch.tensor(0.).to(self.device))

    def split_params(self, tensor):
        mean, logvar = gaussian_params(tensor)
        if not self.learned_variance:
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.log_var_range
        return mean, logvar + np.log(self.pf_std_per_traj) * 2.

    def predict_next_state(self, state, time, condition):
        batch_size = state.shape[0]
        if self.conditional_flow_model:
            condition_embedding = self.conditions_embedding_model(condition)
            log_flow = self.flow_model(condition_embedding).squeeze(-1)
        else:
            condition_embedding = None
            log_flow = self.flow_model
        time_encoding = self.t_model(time).repeat(batch_size, 1)
        state_encoding = self.s_model(state, condition_embedding)
        state_update = self.joint_model(state_encoding, time_encoding)  # nx(2d) with d drift and d noise parameters

        if self.clipping:
            state_update = torch.clip(state_update, -self.gfn_clip, self.gfn_clip)
        return state_update, log_flow.squeeze(-1)

    def get_trajectory_fwd(self, state, exploration_std, log_r, condition):
        batch_size = state.shape[0]

        logpf = torch.zeros((batch_size, self.trajectory_length), device=self.device)
        logpb = torch.zeros((batch_size, self.trajectory_length), device=self.device)
        logf = torch.zeros((batch_size, self.trajectory_length + 1), device=self.device)
        states = torch.zeros((batch_size, self.trajectory_length + 1, self.dim), device=self.device)

        for i in range(self.trajectory_length):
            state_update, log_flow = self.predict_next_state(state, i * self.dt, condition)
            pf_mean, pflogvars = self.split_params(state_update)  # drift and log variance terms

            logf[:, i] = log_flow

            if exploration_std is None:
                pflogvars_sample = pflogvars.detach()
            else:
                expl = exploration_std(i)
                if expl <= 0.0:
                    pflogvars_sample = pflogvars.detach()
                else:
                    add_log_var = torch.full_like(pflogvars, np.log(exploration_std(i) / np.sqrt(self.dt)) * 2)
                    pflogvars_sample = torch.logaddexp(pflogvars, add_log_var).detach()

            next_state = state + self.dt * pf_mean.detach() + np.sqrt(self.dt) * (
                    pflogvars_sample / 2).exp() * torch.randn_like(state, device=self.device)

            noise = ((next_state - state) - self.dt * pf_mean) / (
                        np.sqrt(self.dt) * (pflogvars / 2).exp())  # seems unnecessary, as we have the noise above
            logpf[:, i] = -0.5 * (noise ** 2 + logtwopi + np.log(self.dt) + pflogvars).sum(1)

            if self.learn_pb:
                t = self.t_model((i + 1) * self.dt).repeat(batch_size, 1)
                if self.conditional_flow_model:
                    condition_embedding = self.conditions_embedding_model(condition)
                else:
                    condition_embedding = None
                pbs = self.back_model(self.s_model(next_state, condition_embedding), t)
                dmean, dvar = gaussian_params(pbs)
                back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
            else:
                back_mean_correction, back_var_correction = torch.ones_like(next_state), torch.ones_like(next_state)

            if i > 0:
                back_mean = next_state - self.dt * next_state / ((i + 1) * self.dt) * back_mean_correction
                back_var = (self.pf_std_per_traj ** 2) * self.dt * i / (i + 1) * back_var_correction
                noise_backward = (state - back_mean) / back_var.sqrt()
                logpb[:, i] = -0.5 * (noise_backward ** 2 + logtwopi + back_var.log()).sum(1)

            state = next_state
            states[:, i + 1] = state

        return states, logpf, logpb, logf

    def get_trajectory_bwd(self, s, exploration_std, condition):
        batch_size = s.shape[0]
        logpf = torch.zeros((batch_size, self.trajectory_length), device=self.device)
        logpb = torch.zeros((batch_size, self.trajectory_length), device=self.device)
        logf = torch.zeros((batch_size, self.trajectory_length + 1), device=self.device)
        states = torch.zeros((batch_size, self.trajectory_length + 1, self.dim), device=self.device)
        states[:, -1] = s

        for i in range(self.trajectory_length):
            if i < self.trajectory_length - 1:
                if self.learn_pb:
                    t = self.t_model(1. - i * self.dt).repeat(batch_size, 1)
                    if self.conditional_flow_model:
                        condition_embedding = self.conditions_embedding_model(condition)
                    else:
                        condition_embedding = None
                    pbs = self.back_model(self.s_model(s, condition_embedding), t)
                    dmean, dvar = gaussian_params(pbs)
                    back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                    back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
                else:
                    back_mean_correction, back_var_correction = torch.ones_like(s), torch.ones_like(s)

                mean = s - self.dt * s / (1. - i * self.dt) * back_mean_correction
                var = ((self.pf_std_per_traj ** 2) * self.dt * (1. - (i + 1) * self.dt)) / (
                        1 - i * self.dt) * back_var_correction
                s_ = mean.detach() + var.sqrt().detach() * torch.randn_like(s, device=self.device)
                noise_backward = (s_ - mean) / var.sqrt()
                logpb[:, self.trajectory_length - i - 1] = -0.5 * (noise_backward ** 2 + logtwopi + var.log()).sum(1)
            else:
                s_ = get_gfn_init_state(len(s), s.shape[1], s.device)  # call initial state from function

            pfs, flow = self.predict_next_state(s_, (1. - (i + 1) * self.dt), condition)
            pf_mean, pflogvars = self.split_params(pfs)
            logf[:, self.trajectory_length - i - 1] = flow
            noise = ((s - s_) - self.dt * pf_mean) / (np.sqrt(self.dt) * (pflogvars / 2).exp())
            logpf[:, self.trajectory_length - i - 1] = -0.5 * (noise ** 2 + logtwopi + np.log(self.dt) + pflogvars).sum(
                1)

            s = s_
            states[:, self.trajectory_length - i - 1] = s

        return states, logpf, logpb, logf

    def sample(self, batch_size, log_r, condition=None):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, None, log_r, condition)[0][:, -1]

    def sleep_phase_sample(self, batch_size, exploration_std, condition=None):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, exploration_std, log_r=None, condition=condition)[0][:, -1]

    def forward(self, s, exploration_std=None, log_r=None, condition=None):
        return self.get_trajectory_fwd(s, exploration_std, log_r, condition)
