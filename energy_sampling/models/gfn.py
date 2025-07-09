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
                 harmonics_dim: int, t_dim: int, bwd_policy: str, log_var_range: float = 4.,
                 t_scale: float = 1., learned_variance: bool = True,
                 trajectory_length: int = 100,
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
        self.t_scale = t_scale

        self.clipping = clipping
        self.gfn_clip = gfn_clip

        self.conditional_flow_model = conditional_flow_model
        self.learn_pb = learn_pb

        self.lgv_layers = lgv_layers
        self.joint_layers = joint_layers
        self.bwd_policy = bwd_policy

        self.pf_std_per_traj = np.sqrt(self.t_scale)
        self.dt = 1. / trajectory_length
        self.log_var_range = log_var_range
        self.device = device

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
                logvar = torch.tanh(logvar_i) * self.log_var_range
        return mean, (logvar + np.log(self.pf_std_per_traj) * 2.0).clip(min=-10, max=10)

    def call_forward_policy(self, state, time, condition_embedding):
        batch_size = state.shape[0]

        if torch.is_tensor(time):
            time_encoding = self.t_model(time)
        else:
            time_encoding = self.t_model(time).repeat(batch_size, 1)
        state_encoding = self.s_model(state, condition_embedding)
        state_update = self.forward_policy(state_encoding,
                                           time_encoding)

        if self.clipping:
            state_update = torch.clip(state_update, -self.gfn_clip, self.gfn_clip)

        pf_mean, pf_logvars = self.split_params(state_update)  # drift and log variance terms

        return pf_mean, pf_logvars

    def call_backwards_gaussian_policy(self, state, time, condition_embedding):
        batch_size = state.shape[0]
        time_encoding = self.t_model(time).repeat(batch_size, 1)
        state_encoding = self.s_model(state, condition_embedding)
        state_update = self.backward_policy(state_encoding,
                                            time_encoding)  # nx(2d) with d drift and d noise parameters

        if self.clipping:
            state_update = torch.clip(state_update, -self.gfn_clip, self.gfn_clip)

        pb_mean, pb_logvars = self.split_params(state_update)  # drift and log variance terms

        return pb_mean, pb_logvars

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
            else:  # instead set this as a constant the model will have to learn around
                back_var = torch.ones_like(back_mean) * 1e-1 * dts.unsqueeze(1)
            noise_backward = (current_state - back_mean) / back_var.sqrt()
            logpb[:, i] = -0.5 * (noise_backward ** 2 + logtwopi + back_var.log()).sum(1)

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

            else:
                # send it identically to the source state (0)
                prev_state = torch.zeros_like(current_state)
                back_mean = prev_state  # remember the back_mean in pb is for some reson the actual mean not the drift
                back_var = torch.ones_like(back_mean) * 1e-1 * dts.unsqueeze(1)

            noise_backward = (prev_state - back_mean) / back_var.sqrt()
            logpb[:, trajectory_length - i - 1] = -0.5 * (noise_backward ** 2 + logtwopi + back_var.log()).sum(1)

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
            back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
            if self.learned_variance:
                back_var_correction = (1 + dvar.tanh() * self.pb_scale_range)
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
        if exploration_std is None:
            if detach_traj:
                pflogvars_sample = pflogvars.detach()
            else:
                pflogvars_sample = pflogvars
        else:
            expl = exploration_std(
                None)  # currently not using this arg -- could use ts here, would need changes to utils get_exploration_std
            if expl <= 0.0:
                pflogvars_sample = pflogvars.detach()
            else:
                add_log_var = torch.full_like(pflogvars, np.log(exploration_std(i)) * 2) / dts.sqrt().unsqueeze(1)
                if detach_traj:
                    pflogvars_sample = torch.logaddexp(pflogvars, add_log_var).detach()
                else:
                    pflogvars_sample = torch.logaddexp(pflogvars, add_log_var)
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

    #
    # def mk_get_trajectory_fwd(self,
    #                           initial_state,
    #                           exploration_std,
    #                           condition,
    #                           return_gauss_params: bool = False,
    #                           compute_pf: bool = True,
    #                           compute_pb: bool = True,
    #                           ):
    #
    #     batch_size = initial_state.shape[0]
    #     logf, logpb, logpf, states, means_f, logvars_f, means_b, logvars_b = self.init_traj_tensors(batch_size)
    #
    #     current_state = initial_state.clone().detach()
    #     states[:, 0] = initial_state.detach()  # set correct initial state
    #     condition_embedding = self.conditions_embedding_model(condition)
    #
    #     # get exploration std per-trajectory, and then distribute it randomly across each
    #     if exploration_std is not None:
    #         per_path_expl_std = torch.rand(len(initial_state), device=initial_state.device) * exploration_std(0)
    #         per_step_expl_std = torch.rand_like(states[:, :, 0]) * per_path_expl_std[:, None] * 2
    #     else:
    #         per_step_expl_std = torch.zeros_like(states[:, :, 0])
    #
    #     logf[:, 0] = self.flow_model(condition_embedding).squeeze(-1).squeeze(-1)
    #
    #     with torch.no_grad():
    #         for i in range(self.trajectory_length):  # propagate the SDE
    #             pf_mean, pf_logvar = self.call_forward_policy(current_state, i * self.dt, condition_embedding)
    #             forward_std = (pf_logvar / 2).exp() * np.sqrt(self.dt)
    #             expl_std = per_step_expl_std[:, i, None]
    #             next_state = (current_state +
    #                           self.dt * pf_mean +
    #                           (forward_std + expl_std) * torch.randn_like(current_state, device=self.device))
    #
    #             current_state = next_state
    #             states[:, i + 1] = current_state
    #
    #     if compute_pb:
    #         logpb, logvars_b, means_b = self.compute_traj_pb(condition_embedding, initial_state, logpb, states)
    #
    #     if compute_pf:
    #         logpf, logvars_f, means_f = self.compute_traj_pf(condition_embedding, logpf, states)
    #
    #     if return_gauss_params:
    #         return (states, logpf, logpb, logf,
    #                 means_f.detach().mean(-1), logvars_f.detach().mean(-1), means_b.detach().mean(-1),
    #                 logvars_b.detach().mean(-1))
    #     else:
    #         return states, logpf, logpb, logf
    #
    # def mk_get_trajectory_bwd(self,
    #                           terminal_state,
    #                           condition,
    #                           return_gauss_params: bool = False,
    #                           compute_pf: bool = True,
    #                           compute_pb: bool = True):
    #     initial_state = get_gfn_init_state(len(terminal_state), terminal_state.shape[1], terminal_state.device)
    #     batch_size = terminal_state.shape[0]
    #     logf, logpb, logpf, states, means_f, logvars_f, means_b, logvars_b = self.init_traj_tensors(batch_size)
    #
    #     states[:, 0] = initial_state.clone().detach()
    #     states[:, -1] = terminal_state.clone().detach()
    #     condition_embedding = self.conditions_embedding_model(condition)
    #     current_state = terminal_state.clone().detach()
    #
    #     logf[:, 0] = self.flow_model(condition_embedding).squeeze(-1).squeeze(-1)
    #
    #     with torch.no_grad():  # pure sampling
    #         for i in range(self.trajectory_length - 1):  # the final traj step is deterministic
    #             # equivalent index of current_state in the forward trajectory
    #             traj_ind = self.trajectory_length - i
    #             pb_logvar, pb_mean = self.get_bwd_params_for_state(
    #                 condition_embedding,
    #                 initial_state,
    #                 states[:, traj_ind, :],
    #                 traj_ind * self.dt,
    #                 states[:, -1]
    #             )
    #             backward_std = (pb_logvar / 2).exp() * np.sqrt(self.dt)
    #             prev_state = (current_state +
    #                           self.dt * pb_mean +
    #                           backward_std * torch.randn_like(current_state, device=self.device))
    #
    #             current_state = prev_state
    #             states[:, self.trajectory_length - i - 1] = current_state
    #
    #     if compute_pb:
    #         logpb, logvars_b, means_b = self.compute_traj_pb(condition_embedding, initial_state, logpb, states)
    #
    #     if compute_pf:
    #         logpf, logvars_f, means_f = self.compute_traj_pf(condition_embedding, logpf, states)
    #
    #     if return_gauss_params:
    #         return (states, logpf, logpb, logf,
    #                 means_f.detach().mean(-1), logvars_f.detach().mean(-1),
    #                 means_b.detach().mean(-1), logvars_b.detach().mean(-1))
    #     else:
    #         return states, logpf, logpb, logf

    # def compute_traj_pf(self, condition_embedding, logpf, states):
    #     logvars_f, means_f = self.get_fwd_params_for_traj(condition_embedding, states)
    #     # get forward probabilities in a single parallel step
    #     forward_delta_x = states.diff(dim=1)
    #     fwd_std_step = (logvars_f / 2).exp() * np.sqrt(self.dt)
    #     forward_noise = (forward_delta_x - self.dt * means_f) / fwd_std_step
    #     logpf = (-0.5 * (forward_noise ** 2 + logtwopi * np.log(self.dt) + logvars_f)).sum(2)
    #     return logpf, logvars_f, means_f

    # def compute_traj_pb(self, condition_embedding, initial_state, logpb, states):
    #     # call the policy on all the states in the trajectory at once
    #     logvars_b, means_b = self.get_bwd_params_for_traj(condition_embedding, initial_state, states)
    #     # get backward probabilities in a single parallel step
    #     backward_delta_x = -states.diff(dim=1)  # prev_step - current_step
    #     bwd_std_step = (logvars_b / 2).exp() * np.sqrt(self.dt)
    #     backward_noise = (backward_delta_x - self.dt * means_b) / bwd_std_step
    #     logpb = (-0.5 * (backward_noise ** 2 + logtwopi * np.log(self.dt) + logvars_b)).sum(2)
    #     return logpb, logvars_b, means_b
    #
    # def get_fwd_params_for_traj(self, condition_embedding, states):
    #     # call forward policy on all the states in the trajectory at once
    #     states_for_fwd = states[:, :-1].reshape(states.shape[0] * (states.shape[1] - 1), states.shape[2])
    #     traj_times = torch.linspace(0, 1, self.trajectory_length + 1, device=self.device)
    #     times_for_fwd = traj_times[:-1].repeat(states.shape[0])
    #     if condition_embedding is not None:
    #         conditions_for_fwd = condition_embedding.repeat(self.trajectory_length, 1)
    #     else:
    #         conditions_for_fwd = None
    #     means_f_i, logvars_f_i = self.call_forward_policy(states_for_fwd,
    #                                                       times_for_fwd[:, None],
    #                                                       conditions_for_fwd)
    #     means_f = means_f_i.reshape(states[:, :-1].shape)
    #     logvars_f = logvars_f_i.reshape(states[:, :-1].shape)
    #     return logvars_f, means_f
    #
    # def get_bwd_params_for_traj(self, condition_embedding, initial_state, states):
    #     states_for_bwd = states[:, 1:].reshape(states.shape[0] * (states.shape[1] - 1), states.shape[2])
    #     traj_times = torch.linspace(0, 1, self.trajectory_length + 1, device=self.device)
    #     prev_times = traj_times[:-1]  # t
    #
    #     times_for_bwd = traj_times[1:].repeat(states.shape[0])
    #     if condition_embedding is not None:
    #         conditions_for_bwd = condition_embedding.repeat(self.trajectory_length, 1)
    #     else:
    #         conditions_for_bwd = None
    #     if self.bwd_policy == 'brownian_bridge':  # original implementation of the brownian bridge
    #         back_mean_correction, back_var_correction = self.call_bb_policy(conditions_for_bwd,
    #                                                                         states_for_bwd,
    #                                                                         times_for_bwd[:, None])
    #
    #         # back to traj basis
    #         back_mean_correction = back_mean_correction.reshape(states[:, 1:].shape)
    #         back_var_correction = back_var_correction.reshape(states[:, 1:].shape)
    #
    #         # 'local' brownian bridge, where the drift points always towards the initial state
    #         # todo timing is fucked up
    #         local_slope = - (states[:, 1:] - initial_state[:, None, :]) / (traj_times[None, :-1, None] + self.dt)
    #         means_b = local_slope * back_mean_correction
    #         # sigma^2 * (1- (t - \Delta t))/(1-t) # a constant value that crashes at t=0
    #         # var_at_t = self.pf_std_per_traj ** 2 * self.dt * (1 - traj_times[1:].flip(0)) / (
    #         #         1 - traj_times[:-1].flip(0))  # not sure what I was thinking here, maybe the time-reversed version
    #         var_at_t = self.pf_std_per_traj ** 2 * self.dt * prev_times / (prev_times + self.dt)
    #         logvars_b = ((var_at_t.clip(min=1e-2).log()
    #                       .reshape(1, self.trajectory_length, 1)
    #                       .repeat(states.shape[0], 1, states.shape[2])
    #                       ) * back_var_correction)  # clip for so edges see nonzero probs
    #
    #     elif self.bwd_policy == 'new_brownian_bridge':
    #         back_mean_correction, back_var_correction = self.call_bb_policy(conditions_for_bwd, states_for_bwd,
    #                                                                         times_for_bwd[:, None])
    #
    #         # back to traj basis
    #         back_mean_correction = back_mean_correction.reshape(states[:, 1:].shape)
    #         back_var_correction = back_var_correction.reshape(states[:, 1:].shape)
    #
    #         # proper global brownian bridge
    #         bridge_means = (initial_state[:, None, :] -
    #                         traj_times[None, :, None] * (states[:, -1] - initial_state)[:, None,
    #                                                     :])  # dt scaling might be wrong here
    #
    #         # this is how much we need to adjust the current state such that the previous state has the appropriate mean
    #         # delta mu = mu_(t-1)-mu_t
    #         means_b = -bridge_means.diff(dim=1) * back_mean_correction
    #         var_at_t = (traj_times[1:] * (1 - traj_times[1:]) / 1 * self.pf_std_per_traj ** 2)
    #         logvars_b = ((var_at_t.clip(min=1e-2).log()
    #                       .reshape(1, self.trajectory_length, 1)
    #                       .repeat(states.shape[0], 1, states.shape[2])
    #                       ) * back_var_correction)  # clip for so edges see nonzero probs
    #
    #     elif self.bwd_policy == 'gaussian':
    #         time_encoding = self.t_model(times_for_bwd[:, None])
    #         state_encoding = self.s_model(states_for_bwd, conditions_for_bwd)
    #         state_update = self.backward_policy(state_encoding, time_encoding)
    #
    #         if self.clipping:
    #             state_update = torch.clip(state_update, -self.gfn_clip, self.gfn_clip)
    #
    #         pb_mean, pb_logvar = self.split_params(state_update)  # drift and log variance terms
    #
    #         means_b = pb_mean.reshape(states[:, 1:].shape)
    #         logvars_b = pb_logvar.reshape(states[:, 1:].shape)
    #
    #     return logvars_b, means_b

    # def get_bwd_params_for_state(self, condition_embedding, initial_state, current_state, current_time, terminal_state):
    #     if self.bwd_policy == 'brownian_bridge':  # original implementation of the brownian bridge
    #         back_mean_correction, back_var_correction = self.call_bb_policy(condition_embedding,
    #                                                                         current_state,
    #                                                                         current_time)
    #
    #         # 'local' brownian bridge, where the drift points always towards the initial state
    #         local_slope = - (current_state - initial_state) / (current_time + self.dt)
    #         means_b = local_slope * back_mean_correction
    #         # sigma^2 * (1- (t - \Delta t))/(1-t) # a constant value that crashes at t=0
    #         prev_time = current_time - self.dt
    #         # var_at_t = torch.tensor(self.pf_std_per_traj ** 2 * self.dt * (1 - current_time) / (
    #         #        1 - (current_time - self.dt)), device=self.device)
    #         var_at_t = torch.tensor(self.pf_std_per_traj ** 2 * self.dt * prev_time / (prev_time + self.dt),
    #                                 device=self.device)
    #         logvars_b = (var_at_t.clip(min=1e-2).log() * back_var_correction)  # clip for so edges see nonzero probs
    #
    #     elif self.bwd_policy == 'new_brownian_bridge':
    #         back_mean_correction, back_var_correction = self.call_bb_policy(condition_embedding, current_state,
    #                                                                         current_time)
    #
    #         # proper global brownian bridge
    #         bridge_means = (initial_state[:, None, :] -
    #                         current_time * (terminal_state - initial_state)[:, None,
    #                                        :])  # dt scaling might be wrong here
    #
    #         # this is how much we need to adjust the current state such that the previous state has the appropriate mean
    #         # delta mu = mu_(t-1)-mu_t
    #         means_b = -bridge_means.diff(dim=1) * back_mean_correction
    #         var_at_t = (current_time * (1 - current_time) / 1 * self.pf_std_per_traj ** 2)
    #         logvars_b = (var_at_t.clip(min=1e-2).log() * back_var_correction)  # clip for so edges see nonzero probs
    #
    #     elif self.bwd_policy == 'gaussian':
    #         time_encoding = self.t_model(current_time)
    #         state_encoding = self.s_model(current_state, condition_embedding)
    #         state_update = self.backward_policy(state_encoding, time_encoding)
    #
    #         if self.clipping:
    #             state_update = torch.clip(state_update, -self.gfn_clip, self.gfn_clip)
    #
    #         means_b, logvars_b = self.split_params(state_update)  # drift and log variance terms
    #
    #     return logvars_b, means_b
    #
    # def call_bb_policy(self, condition_embedding, current_state, current_time):
    #     if self.learn_pb:
    #         if isinstance(current_time, float):
    #             time_encoding = self.t_model(current_time).repeat(current_state.shape[0], 1)
    #         else:
    #             time_encoding = self.t_model(current_time)
    #
    #         state_encoding = self.s_model(current_state, condition_embedding)
    #         state_update = self.backward_policy(state_encoding, time_encoding)
    #
    #         dmean, dvar = gaussian_params(state_update)
    #         back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
    #         back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
    #     else:
    #         back_mean_correction, back_var_correction = torch.ones_like(current_state), torch.ones_like(
    #             current_state)
    #     return back_mean_correction, back_var_correction
    #
    # def get_backward_correction(self, batch_size, condition_embedding, time_step, state):
    #     if self.learn_pb:
    #         t = self.t_model(time_step).repeat(batch_size, 1)
    #         pbs = self.backward_policy(self.s_model(state, condition_embedding), t)
    #         dmean, dvar = gaussian_params(pbs)
    #         back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
    #         back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
    #     else:
    #         back_mean_correction, back_var_correction = torch.ones_like(state), torch.ones_like(state)
    #     return back_mean_correction, back_var_correction

    def get_expl_std(self, exploration_std, i):
        if exploration_std is None:
            expl = 0
        else:
            expl = exploration_std(i)

        return expl

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
