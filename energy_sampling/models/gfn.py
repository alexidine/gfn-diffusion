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
                                                        output_dim=condition_embedding_dim)
            self.flow_model = FlowModel(
               condition_embedding_dim, hidden_dim, 1,
                                        norm='layer', dropout=0)
            #self.flow_model = LearnableScalar()
        else:
            self.flow_model = torch.nn.Parameter(torch.tensor(0.).to(self.device))

    def split_params(self, tensor):
        mean, logvar = gaussian_params(tensor)
        if not self.learned_variance:
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.log_var_range
        return mean, logvar + np.log(self.pf_std_per_traj) * 2.

    def predict_next_state(self, state, time, condition_embedding):
        batch_size = state.shape[0]
        if self.conditional_flow_model:
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

    def get_trajectory_fwd(self, initial_state, exploration_std, log_reward_fn, condition, keep_step_grads: bool = False,
                           return_gauss_params: bool=False):

        batch_size = initial_state.shape[0]

        logpf = torch.zeros((batch_size, self.trajectory_length), device=self.device)
        logpb = torch.zeros((batch_size, self.trajectory_length), device=self.device)
        logf = torch.zeros((batch_size, self.trajectory_length + 1), device=self.device)
        states = torch.zeros((batch_size, self.trajectory_length + 1, self.dim), device=self.device)
        if return_gauss_params:
            means = torch.zeros((batch_size, self.trajectory_length), device=self.device)
            logvars = torch.zeros((batch_size, self.trajectory_length), device=self.device)

        states[:, 0] = initial_state.detach()  # set correct initial state
        if self.conditional_flow_model:
            condition_embedding = self.conditions_embedding_model(condition)
        else:
            condition_embedding = None

        current_state = initial_state.clone().detach()
        for i in range(self.trajectory_length):
            state_update, log_flow = self.predict_next_state(current_state, i * self.dt, condition_embedding)
            pf_mean, pflogvars = self.split_params(state_update)  # drift and log variance terms
            logf[:, i] = log_flow

            if exploration_std is None:  # todo clean up this logic
                pflogvars_sample = pflogvars
            else:
                expl = exploration_std(i)
                if expl <= 0.0:
                    pflogvars_sample = pflogvars
                else:
                    add_log_var = torch.full_like(pflogvars, np.log(exploration_std(i) / np.sqrt(self.dt)) * 2)
                    pflogvars_sample = torch.logaddexp(pflogvars, add_log_var)

            if keep_step_grads:
                pf_mean_sample = pf_mean
            else:
                pf_mean_sample = pf_mean.detach()

                pflogvars_sample = pflogvars.detach()

            if return_gauss_params:
                means[:, i] = pf_mean.mean(dim=1).detach()
                logvars[:, i] = pflogvars.mean(dim=1).detach()

            next_state = (current_state +
                          self.dt * pf_mean_sample +
                          np.sqrt(self.dt) * (pflogvars_sample / 2).exp() * torch.randn_like(current_state, device=self.device))

            # need to back the noise out explicitly here to get gradients to pf_mean and pflogvars
            noise = ((next_state - current_state) - self.dt * pf_mean) / (np.sqrt(self.dt) * (pflogvars / 2).exp())
            logpf[:, i] = -0.5 * (noise ** 2 + logtwopi + np.log(self.dt) + pflogvars).sum(1)

            if self.learn_pb:
                t = self.t_model((i + 1) * self.dt).repeat(batch_size, 1)
                pbs = self.back_model(self.s_model(next_state, condition_embedding), t)
                dmean, dvar = gaussian_params(pbs)
                back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
            else:
                back_mean_correction, back_var_correction = torch.ones_like(next_state), torch.ones_like(next_state)

            if i > 0:
                # back_mean = next_state - self.dt * next_state / ((i + 1) * self.dt) * back_mean_correction
                # back_var = (self.pf_std_per_traj ** 2) * self.dt * i / (i + 1) * back_var_correction

                # correcting and simplifying - the second term corrects for the nonzero initial state
                back_mean = (i/(i+1) * next_state + 1/(i+1) * initial_state) * back_mean_correction
                back_var = (self.pf_std_per_traj ** 2) * self.dt * i / (i + 1) * back_var_correction

                noise_backward = (current_state - back_mean) / back_var.sqrt()
                logpb[:, i] = -0.5 * (noise_backward ** 2 + logtwopi + back_var.log()).sum(1)

            current_state = next_state
            states[:, i + 1] = current_state

        if return_gauss_params:
            return states, logpf, logpb, logf, means, logvars
        else:
            return states, logpf, logpb, logf

    def get_trajectory_bwd(self, terminal_state, exploration_std, condition):
        initial_state = get_gfn_init_state(len(terminal_state), terminal_state.shape[1], terminal_state.device)
        batch_size = terminal_state.shape[0]
        logpf = torch.zeros((batch_size, self.trajectory_length), device=self.device)
        logpb = torch.zeros((batch_size, self.trajectory_length), device=self.device)
        logf = torch.zeros((batch_size, self.trajectory_length + 1), device=self.device)
        states = torch.zeros((batch_size, self.trajectory_length + 1, self.dim), device=self.device)
        states[:, -1] = terminal_state
        if self.conditional_flow_model:
            condition_embedding = self.conditions_embedding_model(condition)
        else:
            condition_embedding = None

        current_state = terminal_state.clone().detach()  # todo clean up and unify logic between forward and backward trajs
        for i in range(self.trajectory_length):
            if i < self.trajectory_length - 1:
                # index of the equivalent forward trajectory
                traj_ind = self.trajectory_length - i
                if self.learn_pb:
                    t = self.t_model(1. - i * self.dt).repeat(batch_size, 1)
                    pbs = self.back_model(self.s_model(current_state, condition_embedding), t)
                    dmean, dvar = gaussian_params(pbs)
                    back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                    back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
                else:
                    back_mean_correction, back_var_correction = torch.ones_like(current_state), torch.ones_like(current_state)

                # mean = s - self.dt * s / (1. - i * self.dt) * back_mean_correction
                # var = ((self.pf_std_per_traj ** 2) * self.dt * (1. - (i + 1) * self.dt)) / (
                #         1 - i * self.dt) * back_var_correction
                # simplified and incorporates connection to nonzero initial state
                mean = ((traj_ind - 1) / traj_ind * current_state + (1 / traj_ind) * initial_state) * back_mean_correction  # not sure about this one
                var = ((traj_ind - 1) / traj_ind * self.dt * self.pf_std_per_traj ** 2) * back_var_correction

                prev_state = mean.detach() + var.sqrt().detach() * torch.randn_like(terminal_state, device=self.device)
                noise_backward = (prev_state - mean) / var.sqrt()
                logpb[:, self.trajectory_length - i - 1] = -0.5 * (noise_backward ** 2 + logtwopi + var.log()).sum(1)  # note here delta T folded into the var term
            else:
                prev_state = initial_state  # call initial state from function

            pfs, flow = self.predict_next_state(prev_state, (1. - (i + 1) * self.dt), condition_embedding)
            pf_mean, pflogvars = self.split_params(pfs)
            logf[:, self.trajectory_length - i - 1] = flow
            noise = ((current_state - prev_state) - self.dt * pf_mean) / (np.sqrt(self.dt) * (pflogvars / 2).exp())
            logpf[:, self.trajectory_length - i - 1] = -0.5 * (noise ** 2 + logtwopi + np.log(self.dt) + pflogvars).sum(1)

            current_state = prev_state
            states[:, self.trajectory_length - i - 1] = current_state

        return states, logpf, logpb, logf

    def sample(self, batch_size, log_r, condition=None):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, None, log_r, condition)[0][:, -1]

    def sleep_phase_sample(self, batch_size, exploration_std, condition=None):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, exploration_std, log_reward_fn=None, condition=condition)[0][:, -1]

    def forward(self, s, exploration_std=None, log_r=None, condition=None):
        return self.get_trajectory_fwd(s, exploration_std, log_r, condition)
