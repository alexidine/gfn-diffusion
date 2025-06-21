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
            self.conditions_embedding_model = scalarMLP(input_dim=conditions_dim,
                                                        norm=None,
                                                        dropout=0,
                                                        layers=1,
                                                        filters=hidden_dim,
                                                        output_dim=condition_embedding_dim,
                                                        )
            self.flow_model = FlowModel(condition_embedding_dim,
                                        hidden_dim,
                                        1,
                                        norm='layer',
                                        dropout=0,
                                        )
        else:
            self.flow_model = LearnableScalar()  # unified syntax with this instead of nn.Parameter

    def split_params(self, tensor):
        mean, logvar = gaussian_params(tensor)
        if not self.learned_variance:
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.log_var_range
        return mean, logvar + np.log(self.pf_std_per_traj) * 2.

    def call_forward_policy(self, state, time, condition_embedding):
        batch_size = state.shape[0]
        log_flow = self.flow_model(condition_embedding).squeeze(-1)
        time_encoding = self.t_model(time).repeat(batch_size, 1)
        state_encoding = self.s_model(state, condition_embedding)
        state_update = self.joint_model(state_encoding, time_encoding)  # nx(2d) with d drift and d noise parameters

        if self.clipping:
            state_update = torch.clip(state_update, -self.gfn_clip, self.gfn_clip)

        pf_mean, pf_logvars = self.split_params(state_update)  # drift and log variance terms

        return pf_mean, pf_logvars, log_flow.squeeze(-1)

    def get_trajectory_fwd(self,
                           initial_state,
                           exploration_std,
                           log_reward_fn,
                           condition,
                           return_gauss_params: bool = False
                           ):

        batch_size = initial_state.shape[0]
        logf, logpb, logpf, logvars_b, logvars_f, means_b, means_f, states = (
            self.init_traj_tensors(batch_size, return_gauss_params))

        current_state = initial_state.clone().detach()
        states[:, 0] = initial_state.detach()  # set correct initial state
        condition_embedding = self.conditions_embedding_model(condition)

        for i in range(self.trajectory_length):
            pf_mean, pf_logvar, logf[:, i] = self.call_forward_policy(current_state, i * self.dt, condition_embedding)

            # no longer detaching added variance - I don't know why it was the original code and I don't think it makes sense
            pf_logvar = self.add_expl_var(exploration_std, i, pf_logvar)
            forward_std = (pf_logvar / 2).exp() * np.sqrt(self.dt)
            # propagate SDE
            next_state = (current_state +
                          self.dt * pf_mean +
                          forward_std * torch.randn_like(current_state, device=self.device))

            # get forward probabilities
            noise = ((next_state - current_state) - self.dt * pf_mean) / forward_std
            logpf[:, i] = -0.5 * (noise ** 2 + logtwopi + np.log(self.dt) + pf_logvar).sum(1)

            back_mean_correction, back_var_correction = self.call_backward_policy(batch_size,
                                                                                  condition_embedding,
                                                                                  (i + 1) * self.dt,
                                                                                  next_state)

            if i > 0:  # get backward probabilities
                # pb_mean here is the actual mean of the SDE, not the drift (correction)
                pb_mean = ((i / (i + 1) * next_state + (1 / (i + 1)) * initial_state)) * back_mean_correction
                pb_var = ((self.pf_std_per_traj ** 2) * (i / (i + 1))) * back_var_correction
                noise_backward = (current_state - pb_mean) / (pb_var * self.dt).sqrt()
                logpb[:, i] = -0.5 * (noise_backward ** 2 + logtwopi + np.log(self.dt) + pb_var.log()).sum(1)

            if return_gauss_params:
                means_f[:, i] = pf_mean.mean(dim=1).detach()
                logvars_f[:, i] = pf_logvar.mean(dim=1).detach()
                if i > 0:
                    means_b[:, i] = pb_mean.mean(dim=1).detach()
                    logvars_b[:, i] = pb_var.log().mean(dim=1).detach()

            current_state = next_state
            states[:, i + 1] = current_state

        if return_gauss_params:
            return states, logpf, logpb, logf, means_f, logvars_f, means_b, logvars_b
        else:
            return states, logpf, logpb, logf

    def get_trajectory_bwd(self, terminal_state, exploration_std, condition, return_gauss_params: bool = False):
        initial_state = get_gfn_init_state(len(terminal_state), terminal_state.shape[1], terminal_state.device)
        batch_size = terminal_state.shape[0]
        logf, logpb, logpf, logvars_b, logvars_f, means_b, means_f, states = (
            self.init_traj_tensors(batch_size, return_gauss_params))

        states[:, -1] = terminal_state
        condition_embedding = self.conditions_embedding_model(condition)

        current_state = terminal_state.clone().detach()  # todo clean up and unify logic between forward and backward trajs
        for i in range(self.trajectory_length):
            if i < self.trajectory_length - 1:
                # index of the equivalent forward trajectory
                traj_ind = self.trajectory_length - i
                back_mean_correction, back_var_correction = self.call_backward_policy(
                    batch_size,
                    condition_embedding,
                    1 - i * self.dt,
                    current_state
                )

                # simplified and incorporates connection to nonzero initial state
                pb_mean = ((traj_ind - 1) / traj_ind * current_state + (
                        1 / traj_ind) * initial_state) * back_mean_correction
                pb_var = (((traj_ind - 1) / traj_ind) * self.pf_std_per_traj ** 2) * back_var_correction

                # current_state omitted here as it's implicit in pb_mean above
                prev_state = (pb_mean +
                              (pb_var * self.dt).sqrt() * torch.randn_like(terminal_state, device=self.device))
                pb_mean = ((traj_ind - 1) / traj_ind * current_state + (
                        1 / traj_ind) * initial_state) * back_mean_correction
                noise_backward = (prev_state - pb_mean) / (pb_var * self.dt).sqrt()
                logpb[:, self.trajectory_length - i - 1] = -0.5 * (noise_backward ** 2 + logtwopi + np.log(self.dt) + pb_var.log()).sum(
                    1)
            else:
                prev_state = initial_state  # call initial state from function
                # at t=0 brownian bridge variance goes to zero and the SDE adopts the initial state
                pb_mean = initial_state
                pb_var = torch.zeros_like(pb_var)
                traj_ind = 0

            pf_mean, pf_logvar, flow = self.call_forward_policy(prev_state,
                                                                (1. - (i + 1) * self.dt),
                                                                condition_embedding)
            logf[:, self.trajectory_length - i - 1] = flow
            forward_std = (pf_logvar / 2).exp() * np.sqrt(self.dt)
            noise = ((current_state - prev_state) - self.dt * pf_mean) / forward_std
            logpf[:, self.trajectory_length - i - 1] = -0.5 * (noise ** 2 + logtwopi + np.log(self.dt) + pf_logvar).sum(
                1)

            current_state = prev_state
            states[:, self.trajectory_length - i - 1] = current_state

            if return_gauss_params:
                means_f[:, traj_ind - 1] = pf_mean.mean(dim=1).detach()
                logvars_f[:, traj_ind - 1] = pf_logvar.mean(dim=1).detach()
                if i < self.trajectory_length - 1:
                    means_b[:, traj_ind - 1] = pb_mean.mean(dim=1).detach()
                    logvars_b[:, traj_ind - 1] = pb_var.log().mean(dim=1).detach()

        if return_gauss_params:
            return states, logpf, logpb, logf, means_f, logvars_f, means_b, logvars_b
        else:
            return states, logpf, logpb, logf

    def call_backward_policy(self, batch_size, condition_embedding, time_step, state):
        if self.learn_pb:
            t = self.t_model(time_step).repeat(batch_size, 1)
            pbs = self.back_model(self.s_model(state, condition_embedding), t)
            dmean, dvar = gaussian_params(pbs)
            back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
            back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
        else:
            back_mean_correction, back_var_correction = torch.ones_like(state), torch.ones_like(state)
        return back_mean_correction, back_var_correction

    def add_expl_var(self, exploration_std, i, pf_logvar):
        if exploration_std is None:
            pflogvars_sample = pf_logvar
        else:
            expl = exploration_std(i)
            if expl <= 0.0:
                pflogvars_sample = pf_logvar
            else:
                add_log_var = torch.full_like(pf_logvar, np.log(exploration_std(i) / np.sqrt(self.dt)) * 2)
                pflogvars_sample = torch.logaddexp(pf_logvar, add_log_var)
        return pflogvars_sample

    def init_traj_tensors(self, batch_size, return_gauss_params):
        logpf = torch.zeros((batch_size, self.trajectory_length), device=self.device)
        logpb = torch.zeros((batch_size, self.trajectory_length), device=self.device)
        logf = torch.zeros((batch_size, self.trajectory_length + 1), device=self.device)
        states = torch.zeros((batch_size, self.trajectory_length + 1, self.dim), device=self.device)
        if return_gauss_params:
            means_f = torch.zeros((batch_size, self.trajectory_length), device=self.device)
            logvars_f = torch.zeros((batch_size, self.trajectory_length), device=self.device)
            means_b = torch.zeros((batch_size, self.trajectory_length), device=self.device)
            logvars_b = torch.zeros((batch_size, self.trajectory_length), device=self.device)
        else:
            means_f, logvars_f, means_b, logvars_b = None, None, None, None
        return logf, logpb, logpf, logvars_b, logvars_f, means_b, means_f, states

    def sample(self, batch_size, log_r, condition=None):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, None, log_r, condition)[0][:, -1]

    def sleep_phase_sample(self, batch_size, exploration_std, condition=None):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, exploration_std, log_reward_fn=None, condition=condition)[0][:, -1]

    def forward(self, s, exploration_std=None, log_r=None, condition=None):
        return self.get_trajectory_fwd(s, exploration_std, log_r, condition)
