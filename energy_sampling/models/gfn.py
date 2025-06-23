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

        self.t_model = TimeEncoding(harmonics_dim, t_dim, hidden_dim,
                                    norm=norm, dropout=dropout)
        self.s_model = StateEncoding(dim, hidden_dim, condition_embedding_dim, s_emb_dim,
                                     norm=norm, dropout=dropout)
        self.policy_model = JointPolicy(dim, s_emb_dim, t_dim,
                                        hidden_dim, joint_layers, 2 * dim, zero_init=zero_init,
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

        time_encoding = self.t_model(time).repeat(batch_size, 1)
        state_encoding = self.s_model(state, condition_embedding)
        state_update = self.policy_model.forward_policy(state_encoding,
                                                        time_encoding)  # nx(2d) with d drift and d noise parameters

        if self.clipping:
            state_update = torch.clip(state_update, -self.gfn_clip, self.gfn_clip)

        pf_mean, pf_logvars = self.split_params(state_update)  # drift and log variance terms

        return pf_mean, pf_logvars

    def call_backward_policy(self, state, time, condition_embedding):
        batch_size = state.shape[0]
        time_encoding = self.t_model(time).repeat(batch_size, 1)
        state_encoding = self.s_model(state, condition_embedding)
        state_update = self.policy_model.backward_policy(state_encoding,
                                                         time_encoding)  # nx(2d) with d drift and d noise parameters

        if self.clipping:
            state_update = torch.clip(state_update, -self.gfn_clip, self.gfn_clip)

        pb_mean, pb_logvars = self.split_params(state_update)  # drift and log variance terms

        return pb_mean, pb_logvars

    def get_trajectory_fwd(self,
                           initial_state,
                           exploration_std,
                           log_reward_fn,
                           condition,
                           return_gauss_params: bool = False,
                           compute_pb: bool = True,
                           ):

        batch_size = initial_state.shape[0]
        logf, logpb, logpf, logvars_b, logvars_f, means_b, means_f, states = (
            self.init_traj_tensors(batch_size, return_gauss_params))

        current_state = initial_state.clone().detach()
        states[:, 0] = initial_state.detach()  # set correct initial state
        condition_embedding = self.conditions_embedding_model(condition)

        # get exploration std per-trajectory, and then distribute it randomly across each
        if exploration_std is not None:
            per_path_expl_std = torch.rand(len(initial_state), device=initial_state.device) * exploration_std(0)
            per_step_expl_std = torch.rand_like(states[:, :, 0]) * per_path_expl_std[:, None] * 2
        else:
            per_step_expl_std = torch.zeros_like(states[:, :, 0])

        for i in range(self.trajectory_length):
            logf[:, i] = self.flow_model(condition_embedding).squeeze(-1).squeeze(-1)
            pf_mean, pf_logvar = self.call_forward_policy(current_state, i * self.dt, condition_embedding)
            forward_std = (pf_logvar / 2).exp() * np.sqrt(self.dt)

            # expl_std = self.get_expl_std(exploration_std, i)
            expl_std = per_step_expl_std[:, i, None]

            # propagate SDE
            # add manually the exploratory variance, and do not consider it in the Pf calculation
            # we want to know the probability of the move under the policy model, which is not the same
            # as the path generation policy (which here is just pflogvars + expl)
            next_state = (current_state +
                          self.dt * pf_mean +
                          (forward_std + expl_std) * torch.randn_like(current_state, device=self.device))

            # get forward probabilities
            forward_noise = ((next_state - current_state) - self.dt * pf_mean) / forward_std  # extra variance not included here as this is the noise under the policy
            logpf[:, i] = -0.5 * (forward_noise ** 2 + logtwopi + np.log(self.dt) + pf_logvar).sum(1)

            if compute_pb:
                if self.bwd_policy == 'brownian_bridge':
                    back_mean_correction, back_var_correction = self.get_backward_correction(batch_size,
                                                                                             condition_embedding,
                                                                                             (i + 1) * self.dt,
                                                                                             next_state)

                    # pb_mean here is the actual mean of the SDE, not the drift (correction)
                    pb_mean = ((i / (i + 1) * next_state + (1 / (i + 1)) * initial_state)) * back_mean_correction
                    pb_var = ((self.pf_std_per_traj ** 2) * (i / (i + 1))) * back_var_correction
                    pb_logvar = pb_var.log()
                    # also, this is the noise of the induced backward step, not the actual one
                    # not clear to me this is right, as we want Pb for the real trajectory
                    # not a trajectory induced by Pb
                    # todo fix this
                    noise_backward = (current_state - next_state) / (pb_var * self.dt).sqrt()
                    logpb[:, i] = -0.5 * (noise_backward ** 2 + logtwopi + np.log(self.dt) + pb_logvar).sum(1)
                elif self.bwd_policy == 'gaussian':
                    pb_mean, pb_logvar = self.call_backward_policy(next_state,
                                                                   (i + 1) * self.dt,
                                                                   condition_embedding
                                                                   )
                    backward_std = (pb_logvar / 2).exp() * np.sqrt(self.dt)
                    backward_noise = ((next_state - current_state) - self.dt * pb_mean) / backward_std
                    logpb[:, i] = -0.5 * (backward_noise ** 2 + logtwopi + np.log(self.dt) + pb_logvar).sum(1)
                else:
                    assert False, 'Unknown bwd_policy'
            else:
                pb_mean, pb_logvar = torch.zeros_like(pf_mean), torch.zeros_like(pf_logvar)

            if return_gauss_params:
                means_f[:, i] = pf_mean.mean(dim=1).detach()
                logvars_f[:, i] = pf_logvar.mean(dim=1).detach()
                means_b[:, i] = pb_mean.mean(dim=1).detach()
                logvars_b[:, i] = pb_logvar.mean(dim=1).detach()

            current_state = next_state
            states[:, i + 1] = current_state

        if return_gauss_params:
            return states, logpf, logpb, logf, means_f, logvars_f, means_b, logvars_b
        else:
            return states, logpf, logpb, logf

    def get_trajectory_bwd(self, terminal_state, exploration_std, condition,
                           return_gauss_params: bool = False,
                           compute_pf: bool = True):
        initial_state = get_gfn_init_state(len(terminal_state), terminal_state.shape[1], terminal_state.device)
        batch_size = terminal_state.shape[0]
        logf, logpb, logpf, logvars_b, logvars_f, means_b, means_f, states = (
            self.init_traj_tensors(batch_size, return_gauss_params))

        states[:, -1] = terminal_state
        condition_embedding = self.conditions_embedding_model(condition)

        current_state = terminal_state.clone().detach()  # todo clean up and unify logic between forward and backward trajs
        for i in range(self.trajectory_length):
            traj_ind = self.trajectory_length - i

            #  no need for separate termination logic - put it in the policies
            #   if True: # roll up the trajectory termination logic here. i < self.trajectory_length - 1:
            if self.bwd_policy == 'brownian_bridge':  # I still think there might be issues here
                # index of the equivalent forward trajectory
                back_mean_correction, back_var_correction = self.get_backward_correction(
                    batch_size,
                    condition_embedding,
                    1 - i * self.dt,
                    current_state
                )

                # simplified and incorporates connection to nonzero initial state
                pb_mean = ((traj_ind - 1) / traj_ind * current_state + (
                        1 / traj_ind) * initial_state) * back_mean_correction
                pb_var = (((traj_ind - 1) / traj_ind) * self.pf_std_per_traj ** 2) * back_var_correction
                pb_logvar = pb_var.log()
                # current_state omitted here as it's implicit in pb_mean above
                prev_state = (pb_mean +
                              (pb_var * self.dt).sqrt() * torch.randn_like(terminal_state, device=self.device))
                pb_mean = ((traj_ind - 1) / traj_ind * current_state + (
                        1 / traj_ind) * initial_state) * back_mean_correction
                noise_backward = (prev_state - pb_mean) / (pb_var * self.dt).sqrt()
                logpb[:, self.trajectory_length - i - 1] = -0.5 * (
                        noise_backward ** 2 + logtwopi + np.log(self.dt) + pb_logvar).sum(
                    1)
            elif self.bwd_policy == 'gaussian':

                pb_mean, pb_logvar = self.call_backward_policy(current_state,
                                                               1 - i * self.dt,
                                                               condition_embedding
                                                               )
                backward_std = (pb_logvar / 2).exp() * np.sqrt(self.dt)
                prev_state = (current_state +
                              self.dt * pb_mean +
                              backward_std * torch.randn_like(current_state, device=self.device))

                backward_noise = ((prev_state - current_state) - self.dt * pb_mean) / backward_std
                logpb[:, i] = -0.5 * (backward_noise ** 2 + logtwopi + np.log(self.dt) + pb_logvar).sum(1)
            else:
                assert False, 'Unknown bwd_policy'

            # let it propagate
            # else:
            #     prev_state = initial_state  # call initial state from function
            #     # at t=0 brownian bridge variance goes to zero and the SDE adopts the initial state
            #     pb_mean = initial_state
            #     pb_var = torch.zeros_like(pb_mean)
            #     pb_logvar = pb_var.log()
            #     traj_ind = 0

            if compute_pf:
                logf[:, self.trajectory_length - i - 1] = self.flow_model(condition_embedding).squeeze(-1).squeeze(-1)
                pf_mean, pf_logvar = self.call_forward_policy(prev_state,
                                                              (1. - (i + 1) * self.dt),
                                                              condition_embedding)
                forward_std = (pf_logvar / 2).exp() * np.sqrt(self.dt)
                noise = ((current_state - prev_state) - self.dt * pf_mean) / forward_std
                logpf[:, self.trajectory_length - i - 1] = -0.5 * (
                        noise ** 2 + logtwopi + np.log(self.dt) + pf_logvar).sum(
                    1)
            else:
                pf_mean, pf_logvar = torch.zeros_like(pb_mean), torch.zeros_like(pb_logvar)

            current_state = prev_state
            states[:, self.trajectory_length - i - 1] = current_state

            if return_gauss_params:
                means_f[:, traj_ind - 1] = pf_mean.mean(dim=1).detach()
                logvars_f[:, traj_ind - 1] = pf_logvar.mean(dim=1).detach()
                means_b[:, traj_ind - 1] = pb_mean.mean(dim=1).detach()
                logvars_b[:, traj_ind - 1] = pb_logvar.mean(dim=1).detach()

        if return_gauss_params:
            return states, logpf, logpb, logf, means_f, logvars_f, means_b, logvars_b
        else:
            return states, logpf, logpb, logf

    def get_backward_correction(self, batch_size, condition_embedding, time_step, state):
        if self.learn_pb:
            t = self.t_model(time_step).repeat(batch_size, 1)
            pbs = self.backward_policy(self.s_model(state, condition_embedding), t)
            dmean, dvar = gaussian_params(pbs)
            back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
            back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
        else:
            back_mean_correction, back_var_correction = torch.ones_like(state), torch.ones_like(state)
        return back_mean_correction, back_var_correction

    def get_expl_std(self, exploration_std, i):
        if exploration_std is None:
            expl = 0
        else:
            expl = exploration_std(i)

        return expl

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
