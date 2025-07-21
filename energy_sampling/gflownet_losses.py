import math
from typing import Optional

import torch.nn.functional as F
import torch
from mxtaltools.dataset_utils.utils import collate_data_list

from utils import compute_sample_overlap


def get_loss_reward(log_T_tensor, log_reward_fn, mol_batch, return_exp, states, no_grad: bool = True):
    if log_T_tensor is not None:
        log_temperature = log_T_tensor
    else:
        log_temperature = None
    with torch.set_grad_enabled(not no_grad):
        if return_exp:
            log_r, crystal_batch = log_reward_fn(states[:, -1], mol_batch, log_temperature, return_exp)
            crystal_batch = crystal_batch.detach()
        else:
            log_r = log_reward_fn(states[:, -1], mol_batch, log_temperature, return_exp)
            crystal_batch = None
    return crystal_batch, log_r


def normed_smoothness_loss(x, eps=1e-5):
    second_diff = torch.diff(x, n=2, dim=-1)
    curvature = (second_diff ** 4).mean(dim=-1)  # specifically punish very large changes
    variance = torch.var(x, dim=-1, correction=0)
    return curvature / (variance + eps)


def soft_saturate(x, scale: Optional[float] = 10.0):
    return torch.log(torch.abs(x / scale) + 1) * torch.sign(x)


def get_gfn_forward_loss(loss_coeffs,
                         initial_state,
                         gfn,
                         log_reward_fn,
                         discretizer,
                         mol_batch,
                         buffer,
                         log_T_tensor,
                         exploration_std=None, return_exp=False, condition=None,
                         repeats=10, reweight_T: Optional[float] = None):
    if gfn.conditional_flow_model and any([
        loss_coeffs.var > 0, loss_coeffs.vg_lb > 0,
        loss_coeffs.vg_lme > 0, loss_coeffs.buffer > 0
    ]):
        condition = condition.repeat(repeats, 1)
        initial_state = initial_state.repeat(repeats, 1)
        reassign_sgs = False
        log_T_tensor = log_T_tensor.repeat(repeats)
        if hasattr(mol_batch,'sg_ind'):
            sg_inds = mol_batch.sg_ind.repeat(repeats)
            reassign_sgs = True
        mol_batch = collate_data_list(mol_batch.to_data_list() * repeats)
        if reassign_sgs:
            mol_batch.sg_ind = sg_inds

    if loss_coeffs.greedy > 0 or loss_coeffs.var > 0 or loss_coeffs.buffer > 0:  # 0 or loss_coeffs.drift > 0:
        keep_grads = True
    else:
        keep_grads = False

    (states, log_pfs, log_pbs, log_fs,
     means_f, logvars_f, means_b, logvars_b) = gfn.get_trajectory_fwd(initial_state,
                                                                      discretizer,
                                                                      exploration_std,
                                                                      condition,
                                                                      detach_traj=not keep_grads,
                                                                      return_gauss_params=True,
                                                                      )

    crystal_batch, log_r = get_loss_reward(log_T_tensor,
                                           log_reward_fn,
                                           mol_batch,
                                           return_exp,
                                           states,
                                           no_grad=loss_coeffs.greedy == 0
                                           )
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_flow = log_fs[:, 0]

    losses = []
    """greedy loss"""
    if loss_coeffs.greedy > 0:
        greedy_loss = soft_saturate(-log_r)
        losses.append(greedy_loss * loss_coeffs.greedy)

    """trajectory smoothing loss"""
    if loss_coeffs.smoothed > 0:
        smoothness_loss = normed_smoothness_loss(torch.stack([means_f, logvars_f, means_b, logvars_b])).mean(dim=0)
        losses.append(smoothness_loss * loss_coeffs.smoothed)

    """TB loss"""
    if loss_coeffs.tb > 0:
        tb = (log_pf + log_flow - log_pb - log_r.detach())
        tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')
        losses.append(tb_loss * loss_coeffs.tb)

    """VarGrad - lower bound loss"""
    if loss_coeffs.vg_lb > 0:
        assert not (loss_coeffs.vg_lb > 0 and loss_coeffs.vg_lme > 0), \
            "Cannot use both vg_lb and vg_lme simultaneously"
        log_ratio = log_r.detach() + log_pb - log_pf

        if gfn.conditional_flow_model:
            log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
            vg_loss = (0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2).view(-1)
        else:
            log_Z = log_ratio.mean(dim=0, keepdim=True)
            vg_loss = 0.5 * (log_Z - log_ratio) ** 2
        losses.append(vg_loss * loss_coeffs.vg_lb)

        """VarGrad - log mean exp loss"""
    elif loss_coeffs.vg_lme > 0:
        log_ratio = log_r.detach() + log_pb - log_pf

        if gfn.conditional_flow_model:
            log_Z = torch.logsumexp(log_ratio.view(repeats, -1), dim=0, keepdim=True) - math.log(repeats)
            vg_loss = (0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2).view(-1)
        else:
            log_Z = torch.logsumexp(log_ratio, dim=0, keepdim=True) - math.log(repeats)
            vg_loss = 0.5 * (log_Z - log_ratio) ** 2
        losses.append(vg_loss * loss_coeffs.vg_lme)

    """MLE/TPM loss"""
    if loss_coeffs.mle > 0:
        mle_loss = soft_saturate(-log_pbs.sum(-1))
        losses.append(mle_loss * loss_coeffs.mle)

    """Variance loss"""
    if loss_coeffs.var > 0:
        states_to_compare = states[:, -1].clip(min=-6, max=6)  # states[:, -1, 3:].clip(min=-6, max=6)

        if gfn.conditional_flow_model:
            states_reshaped = states_to_compare.view(repeats, -1, states_to_compare.shape[-1])

            # Compute variance within each condition
            batch_var = states_reshaped.var(dim=1, keepdim=True)  # (repeats, 1, features)
            var_loss = (F.relu(-(batch_var - loss_coeffs.var_cutoff)) / loss_coeffs.var_cutoff
                        ).expand(-1, states_reshaped.shape[1], -1).mean(dim=2)  # (repeats, batch_size_per_condition)

            # Compute overlap within each condition
            overlap_loss = torch.zeros_like(var_loss)
            for i in range(repeats):
                condition_states = states_reshaped[i]
                overlap_loss[i] = compute_sample_overlap(
                    condition_states,
                    condition_states,
                    ga=loss_coeffs.var_gamma,
                    agg='mean',
                )

            total_var_loss = (overlap_loss + var_loss).view(-1)
        else:
            states_to_compare = states[:, -1].clip(min=-6, max=6) #states[:, -1, 3:].clip(min=-6, max=6)
            batch_var = states_to_compare.var(dim=0, keepdim=True)
            var_loss = (F.relu(-(batch_var - loss_coeffs.var_cutoff))/loss_coeffs.var_cutoff).repeat(len(states), 1).mean(dim=1)
            overlap_loss = compute_sample_overlap(
                states_to_compare,  # don't let it escape - it could cheat
                states_to_compare,  # don't let it escape - it could cheat
                ga=loss_coeffs.var_gamma,
                agg='mean',
            )
            total_var_loss = overlap_loss + var_loss
        losses.append(total_var_loss * loss_coeffs.var)

    """Buffer distance loss"""
    if loss_coeffs.buffer > 0:
        assert not gfn.conditional_flow_model, "Buffer loss not yet set up for conditional sampling"
        states_to_compare = states[:, -1].clip(min=-6, max=6) #states[:, -1, 3:].clip(min=-6, max=6)
        buffer_states = torch.stack(buffer.x_list).to(gfn.device)
        buffer_to_compare = buffer_states[:, -1, 3:].clip(min=-6, max=6)
        buffer_loss = compute_sample_overlap(
            buffer_to_compare,
            states_to_compare,  # don't let it escape - it could cheat
            ga=loss_coeffs.buffer_gamma,
        )
        losses.append(buffer_loss * loss_coeffs.buffer)

    combined_losses = torch.stack(losses).mean(dim=0)

    loss = reweight_losses(combined_losses, losses, reweight_T)

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return loss


def reweight_losses(combined_losses, losses, reweight_T):
    if reweight_T is not None:  # optionally reweight losses to minimize large outliers
        weights = (torch.softmax(-combined_losses.detach() / reweight_T, dim=0) * len(losses)).clamp(min=1e-4)
        weights /= weights.sum()
        loss = (weights * combined_losses).mean()
    else:
        loss = combined_losses.mean()
    return loss


def get_gfn_backward_loss(loss_coeffs,
                          samples,
                          gfn,
                          log_r,
                          discretizer,
                          exploration_std=None,
                          condition=None,
                          repeats=10,
                          return_exp=False, reweight_T: Optional[float] = None):
    if gfn.conditional_flow_model and any([
        loss_coeffs.vg_lb > 0, loss_coeffs.vg_lme > 0
    ]):
        condition = condition.repeat(repeats, 1)
        samples = samples.repeat(repeats, 1)
        log_r = log_r.repeat(repeats)

    (states, log_pfs, log_pbs, log_fs,
     means_f, logvars_f, means_b, logvars_b) = gfn.get_trajectory_bwd(
        samples, discretizer, condition, return_gauss_params=True)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_flow = log_fs[:, 0]

    losses = []
    if loss_coeffs.smoothed > 0:
        smoothness_loss = normed_smoothness_loss(torch.stack([means_f, logvars_f, means_b, logvars_b])).mean(dim=0)
        losses.append(smoothness_loss * loss_coeffs.smoothed)

    if loss_coeffs.tb > 0:
        tb = (log_pf + log_flow - log_pb - log_r)
        tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')
        losses.append(tb_loss * loss_coeffs.tb)

    if loss_coeffs.vg_lb > 0:
        assert not (loss_coeffs.vg_lb > 0 and loss_coeffs.vg_lme > 0), \
            "Cannot use both vg_lb and vg_lme simultaneously"
        log_ratio = log_r + log_pb - log_pf

        if gfn.conditional_flow_model:
            log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
            vg_loss = (0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2).view(-1)
        else:
            log_Z = log_ratio.mean(dim=0, keepdim=True)
            vg_loss = 0.5 * (log_Z - log_ratio) ** 2
        losses.append(vg_loss * loss_coeffs.vg_lb)

    elif loss_coeffs.vg_lme > 0:
        log_ratio = log_r + log_pb - log_pf

        if gfn.conditional_flow_model:
            log_Z = torch.logsumexp(log_ratio.view(repeats, -1), dim=0, keepdim=True) - math.log(repeats)
            vg_loss = (0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2).view(-1)
        else:
            log_Z = torch.logsumexp(log_ratio, dim=0, keepdim=True) - math.log(repeats)
            vg_loss = 0.5 * (log_Z - log_ratio) ** 2
        losses.append(vg_loss * loss_coeffs.vg_lme)

    if loss_coeffs.mle > 0:
        mle_loss = soft_saturate(-log_pfs.sum(-1))
        losses.append(mle_loss * loss_coeffs.mle)

    combined_losses = torch.stack(losses).mean(dim=0)

    loss = reweight_losses(combined_losses, losses, reweight_T)

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
    else:
        return loss


#
# def fwd_combo(initial_state, gfn, log_reward_fn, discretizer, mol_batch,
#               exploration_std=None, return_exp=False, condition=None, repeats=10):
#     """
#     Loss function that dynamically combines several losses on-the-fly.
#     """
#     if gfn.conditional_flow_model:
#         condition = condition.repeat(repeats, 1)
#         initial_state = initial_state.repeat(repeats, 1)
#         mol_batch = collate_data_list(mol_batch.to_data_list() * repeats)
#
#     (states, log_pfs, log_pbs, log_fs,
#      means_f, logvars_f, means_b, logvars_b) = gfn.get_trajectory_fwd(initial_state,
#                                                                       discretizer,
#                                                                       exploration_std,
#                                                                       condition,
#                                                                       return_gauss_params=True)
#     crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)
#
#     log_pf = log_pfs.sum(-1)
#     log_pb = log_pbs.sum(-1)
#     log_flow = log_fs[:, 0]
#     log_ratio = log_r + log_pb - log_pf
#
#     # trajectory balance loss
#     tb = log_flow - log_ratio
#     tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')
#
#     # VarGrad loss
#     if gfn.conditional_flow_model:
#         log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
#         vg_loss = 0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2
#
#     else:
#         log_Z = log_ratio.mean(dim=0, keepdim=True)
#         vg_loss = 0.5 * (log_Z - log_ratio) ** 2
#
#     # greedy loss
#     greedy_loss = soft_saturate(-log_r)  # this is just the system energy
#
#     # backward MLE
#     mle_loss = soft_saturate(-log_pbs.sum(-1))
#
#     loss = vg_loss.mean() + tb_loss.mean() + greedy_loss.mean() + mle_loss.mean()
#
#     if return_exp:
#         return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
#     else:
#         return loss
#
#
# def bwd_combo(terminal_state, gfn, log_r, discretizer, condition=None, repeats=10,
#               return_exp: bool = False):
#     if gfn.conditional_flow_model:  # do repeats if there are conditions, otherwise skip
#         condition = condition.repeat(repeats, 1)
#         terminal_state = terminal_state.repeat(repeats, 1)
#         log_r = log_r.repeat(repeats)
#
#     (states, log_pfs, log_pbs, log_fs,
#      means_f, logvars_f, means_b, logvars_b) \
#         = gfn.get_trajectory_bwd(terminal_state, discretizer, condition, return_gauss_params=True)
#
#     log_pf = log_pfs.sum(-1)
#     log_pb = log_pbs.sum(-1)
#     log_flow = log_fs[:, 0]
#     log_ratio = log_r + log_pb - log_pf
#
#     # trajectory balance loss
#     tb = log_flow - log_ratio
#     tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')
#
#     # VarGrad loss
#     if gfn.conditional_flow_model:
#         log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
#         vg_loss = 0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2
#     else:
#         log_Z = log_ratio.mean(dim=0, keepdim=True)
#         vg_loss = 0.5 * (log_Z - log_ratio) ** 2
#
#     # forward MLE
#     loss = soft_saturate(-log_pfs.sum(-1))
#
#     # mle loss
#
#     loss = vg_loss.mean() + tb_loss.mean()
#
#     if return_exp:
#         return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
#     else:
#         return loss
#
#
# def fwd_tb(initial_state, gfn, log_reward_fn, discretizer, mol_batch,
#            exploration_std=None, return_exp=False, condition=None):
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer, exploration_std, condition)
#     crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)
#
#     log_pf = log_pfs.sum(-1)
#     log_pb = log_pbs.sum(-1)
#     log_flow = log_fs[:, 0]
#
#     tb = (log_pf + log_flow - log_pb - log_r)
#     tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')
#
#     loss = tb_loss
#
#     if return_exp:
#         return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
#     else:
#         return loss
#
#
# def fwd_greedy(initial_state, gfn, log_reward_fn, discretizer, mol_batch,
#                exploration_std=None, return_exp=False, condition=None,
#                traj_midpoint: int = 0,
#                repeats: int = 1, entropy_penalty: float = 1.0,
#                ):
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer,
#                                                               None, condition,
#                                                               detach_traj=False)
#     # optionally only evaluate performance from some mid-trajectory initial state, rather from the global init s0
#     if traj_midpoint > 0:
#         states[:, :traj_midpoint] = states[:, :traj_midpoint].detach()
#
#     # # optionally, evaluate over a minibatch and penalize low entropy
#     # loss = -torch.logsumexp(torch.stack(log_r_list), dim=0)
#     # mean_reward = torch.stack(log_r_list).mean(dim=0)
#     # var_reward = torch.stack(log_r_list).var(dim=0)
#     # loss = -mean_reward + alpha * var_reward
#     crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states, no_grad=False)
#     loss = soft_saturate(-log_r)  # this is just the system energy
#
#     if return_exp:
#         return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
#     else:
#         return loss
#
#
# def fwd_tb_greedy(initial_state, gfn, log_reward_fn, discretizer, mol_batch,
#                   exploration_std=None, return_exp=False, condition=None,
#                   traj_midpoint: int = 0,
#                   repeats: int = 1, entropy_penalty: float = 1.0,
#                   ):
#     """standard TB loss, but at zero expl, will add a greedy loss term to boost exploitation"""
#     skip_greedy = exploration_std(0) != 0
#
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer,
#                                                               exploration_std, condition,
#                                                               detach_traj=skip_greedy)
#     # optionally only evaluate performance from some mid-trajectory initial state, rather from the global init s0
#     if traj_midpoint > 0:
#         states[:, :traj_midpoint] = states[:, :traj_midpoint].detach()
#
#     crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states, no_grad=skip_greedy)
#
#     if skip_greedy:
#         greedy_loss = torch.zeros_like(log_r)
#     else:
#         greedy_loss = -log_r
#
#     log_pf = log_pfs.sum(-1)
#     log_pb = log_pbs.sum(-1)
#     log_flow = log_fs[:, 0]
#
#     tb = (log_pf + log_flow - log_pb - log_r.detach())
#     tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')
#
#     loss = tb_loss + greedy_loss
#
#     if return_exp:
#         return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
#     else:
#         return loss
#
#
# def fwd_vg_greedy(initial_state, gfn, log_reward_fn, discretizer, mol_batch,
#                   exploration_std=None, return_exp=False, condition=None,
#                   traj_midpoint: int = 0,
#                   repeats: int = 1, entropy_penalty: float = 1.0,
#                   ):
#     skip_greedy = exploration_std(0) != 0
#
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer,
#                                                               exploration_std, condition,
#                                                               detach_traj=skip_greedy)
#     # optionally only evaluate performance from some mid-trajectory initial state, rather from the global init s0
#     if traj_midpoint > 0:
#         states[:, :traj_midpoint] = states[:, :traj_midpoint].detach()
#
#     crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states, no_grad=skip_greedy)
#
#     log_pf = log_pfs.sum(-1)
#     log_pb = log_pbs.sum(-1)
#     log_ratio = log_r.detach() + log_pb - log_pf
#
#     if gfn.conditional_flow_model:
#         # reshape and take the mean over repeats
#         # minimize the variance over repeats w.r.t., the norm
#         log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
#         if skip_greedy:
#             greedy_loss = torch.zeros_like(log_Z)
#         else:
#             greedy_loss = -log_r
#         greedy_loss = -log_r.view(repeats, -1).mean(dim=0, keepdim=True)
#
#         vg_loss = 0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2
#     else:
#         # take the variance over the full unconditional batch
#         log_Z = log_ratio.mean(dim=0, keepdim=True)
#         if skip_greedy:
#             greedy_loss = torch.zeros_like(log_r)
#         else:
#             greedy_loss = -log_r
#         vg_loss = 0.5 * (log_Z - log_ratio) ** 2
#
#     loss = vg_loss + greedy_loss
#
#     if return_exp:
#         return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
#     else:
#         return loss
#
#
# def bwd_tb(terminal_state, gfn, log_r, discretizer, condition=None, return_exp: bool = False):
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, discretizer, condition)
#     log_pf = log_pfs.sum(-1)
#     log_pb = log_pbs.sum(-1)
#     log_flow = log_fs[:, 0]
#
#     tb = (log_pf + log_flow - log_pb - log_r)
#     tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')
#
#     loss = tb_loss
#
#     if return_exp:
#         return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
#     else:
#         return loss
#
#
# def fwd_vg(initial_state, gfn,
#            log_r,
#            discretizer,
#            mol_batch,
#            exploration_std=None,
#            return_exp=False,
#            condition=None,
#            repeats=10):
#     if gfn.conditional_flow_model:
#         condition = condition.repeat(repeats, 1)
#         initial_state = initial_state.repeat(repeats, 1)
#         mol_batch = collate_data_list(mol_batch.to_data_list() * repeats)
#
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer, exploration_std, condition)
#     crystal_batch, log_r = get_loss_reward(condition, log_r, mol_batch, return_exp, states)
#
#     log_pf = log_pfs.sum(-1)
#     log_pb = log_pbs.sum(-1)
#     log_ratio = log_r + log_pb - log_pf
#
#     if gfn.conditional_flow_model:
#         # reshape and take the mean over repeats
#         # minimize the variance over repeats w.r.t., the norm
#         log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
#         vg_loss = 0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2
#     else:
#         # take the variance over the full unconditional batch
#         log_Z = log_ratio.mean(dim=0, keepdim=True)
#         vg_loss = 0.5 * (log_Z - log_ratio) ** 2
#
#     loss = vg_loss
#
#     if return_exp:
#         return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
#     else:
#         return loss
#
#
# def bwd_vg(terminal_state, gfn, log_r, discretizer, condition=None, repeats=10,
#            return_exp: bool = False):
#     if gfn.conditional_flow_model:  # do repeats if there are conditions, otherwise skip
#         condition = condition.repeat(repeats, 1)
#         terminal_state = terminal_state.repeat(repeats, 1)
#         log_r = log_r.repeat(repeats)
#
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, discretizer, condition)
#     log_pf = log_pfs.sum(-1)
#     log_pb = log_pbs.sum(-1)
#     log_ratio = log_r + log_pb - log_pf
#
#     if gfn.conditional_flow_model:
#         # reshape and take the mean over repeats
#         # minimize the variance over repeats w.r.t., the norm
#         log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
#         vg_loss = 0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2
#     else:
#         # take the variance over the full unconditional batch
#         log_Z = log_ratio.mean(dim=0, keepdim=True)
#         vg_loss = 0.5 * (log_Z - log_ratio) ** 2
#
#     loss = vg_loss
#
#     if return_exp:
#         return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
#     else:
#         return loss
#
#
# def db(initial_state, gfn, log_reward_fn, exploration_std=None, condition=None):
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, condition)
#     with torch.no_grad():
#         log_fs[:, -1] = log_reward_fn(states[:, -1], condition).detach()
#
#     loss = 0.5 * ((log_pfs + log_fs[:, :-1] - log_pbs - log_fs[:, 1:]) ** 2).sum(-1)
#     return loss
#
#
# def subtb(initial_state, gfn, log_reward_fn, coef_matrix, exploration_std=None, condition=None):
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, condition)
#     with torch.no_grad():
#         log_fs[:, -1] = log_reward_fn(states[:, -1], condition).detach()
#
#     diff_logp = log_pfs - log_pbs
#     diff_logp_padded = torch.cat(
#         (torch.zeros((diff_logp.shape[0], 1)).to(diff_logp),
#          diff_logp.cumsum(dim=-1)),
#         dim=1)
#     A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)
#     A2 = log_fs[:, :, None] - log_fs[:, None, :] + A1
#     A2 = A2 ** 2
#     return torch.stack([torch.triu(A2[i] * coef_matrix, diagonal=1).sum() for i in range(A2.shape[0])]).sum()
#
#
# def fwd_mle(initial_state, gfn, log_reward_fn, discretizer, mol_batch,
#             exploration_std=None, return_exp=False, condition=None):
#     """maximize the probability of forward-sampled trajectories under the backward model"""
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer, None, condition)
#     crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)
#     loss = soft_saturate(-log_pbs.sum(-1))
#     if return_exp:
#         return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
#     else:
#         return loss
#
#
# def bwd_mle(terminal_state, gfn, discretizer, log_r, condition=None, return_exp: bool = False):
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, discretizer, condition)
#     loss = soft_saturate(-log_pfs.sum(-1))
#     if return_exp:
#         return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
#     else:
#         return loss
#
#
# def bwd_mle_batch(terminal_state, gfn, discretizer, log_r, condition=None, return_exp: bool = False, repeats: int = 10):
#     """
#     Use importance sampling on the backwards trajectories to weight the forward policy probability loss
#     """
#
#     condition = condition.repeat(repeats, 1)
#     terminal_state = terminal_state.repeat(repeats, 1)
#     log_r = log_r.repeat(repeats)
#
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, discretizer, condition)
#
#     log_pb = log_pbs.sum(-1).view(repeats, -1)
#     log_pf = log_pfs.sum(-1).view(repeats, -1)
#
#     log_weights = log_pf - log_pb
#     log_weights = log_weights - log_weights.max(dim=0, keepdim=True).values  # shape [repeats, B]
#
#     # Softmax weights for marginal likelihood
#     weights = torch.softmax(log_weights, dim=0)  # shape: [repeats, B]
#
#     # Use weighted average of log_pf for stability
#     loss = soft_saturate(-torch.sum(weights * log_pf, dim=0))  # shape: [B]
#
#     if return_exp:
#         return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
#     else:
#         return loss
#


#
# def old_get_gfn_forward_loss(mode,
#                              init_state,
#                              gfn_model,
#                              log_reward,
#                              discretizer,
#                              mol_batch,
#                              exploration_std=None, return_exp=False, condition=None,
#                              repeats=10, reweight_T: Optional[float] = None):
#     if mode == 'tb':
#         out = fwd_tb(init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std,
#                      return_exp=return_exp,
#                      condition=condition)
#     elif mode == 'vg':
#         out = fwd_vg(init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std, return_exp=return_exp,
#                      condition=condition, repeats=repeats)
#     elif mode == 'combo':
#         out = fwd_combo(init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std,
#                         return_exp=return_exp,
#                         condition=condition, repeats=repeats)
#     elif mode == 'greedy':
#         out = fwd_greedy(init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std,
#                          return_exp=return_exp,
#                          condition=condition)
#     elif mode == 'tb_greedy':
#         out = fwd_tb_greedy(
#             init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std,
#             return_exp=return_exp,
#             condition=condition
#         )
#     elif mode == 'vg_greedy':
#         out = fwd_vg_greedy(
#             init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std,
#             return_exp=return_exp,
#             condition=condition
#         )
#
#     elif mode == 'mle':
#         out = fwd_mle(init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std,
#                       return_exp=return_exp,
#                       condition=condition)
#     else:
#         assert False
#     losses, *rest = out
#
#     if reweight_T is not None:  # optionally reweight losses to minimize large outliers
#         weights = torch.softmax(-losses.detach() / reweight_T, dim=0) * len(losses)
#         weights += 1e-2  # minimum relative contribution
#         weights /= weights.sum()
#         loss = (weights * losses).mean()
#     else:
#         loss = losses.mean()
#
#     return loss, *rest


# def old_get_gfn_backward_loss(mode, samples, gfn_model, rewards, discretizer, exploration_std=None, condition=None,
#                               repeats=10,
#                               return_exp=False, reweight_T: Optional[float] = None):
#     if mode == 'tb':
#         out = bwd_tb(samples, gfn_model, rewards, discretizer, condition=condition, return_exp=return_exp)
#     elif mode == 'vg':
#         out = bwd_vg(samples, gfn_model, rewards, discretizer, condition=condition, repeats=repeats,
#                      return_exp=return_exp)
#     elif mode == 'combo':
#         out = bwd_combo(samples, gfn_model, rewards, discretizer, condition=condition, repeats=repeats,
#                         return_exp=return_exp)
#     elif mode == 'mle':
#         out = bwd_mle(samples, gfn_model, discretizer, rewards, condition=condition, return_exp=return_exp)
#     elif mode == 'mle_batch':
#         out = bwd_mle_batch(samples, gfn_model, discretizer, rewards, condition=condition, return_exp=return_exp,
#                             repeats=repeats)
#     else:
#         assert False
#
#     losses, *rest = out
#     if reweight_T is not None:  # optionally reweight losses to minimize large outliers
#         weights = torch.softmax(-losses.detach() / reweight_T, dim=0) * len(losses)
#         weights += 1e-2  # minimum relative contribution
#         weights /= weights.sum()
#         loss = (weights * losses).mean()
#
#     else:
#         loss = losses.mean()
#
#     return loss, *rest

