from typing import Optional

import numpy as np
import torch.nn.functional as F
import torch
from mxtaltools.dataset_utils.utils import collate_data_list


def get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states, no_grad: bool = True):
    if condition is not None:
        log_temperature = condition[:, 0]
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


def fwd_combo(initial_state, gfn, log_reward_fn, discretizer, mol_batch,
              exploration_std=None, return_exp=False, condition=None, repeats=10):
    if gfn.conditional_flow_model:
        condition = condition.repeat(repeats, 1)
        initial_state = initial_state.repeat(repeats, 1)
        mol_batch = collate_data_list(mol_batch.to_data_list() * repeats)

    (states, log_pfs, log_pbs, log_fs,
     means_f, logvars_f, means_b, logvars_b) = gfn.get_trajectory_fwd(initial_state,
                                                                      discretizer,
                                                                      exploration_std,
                                                                      condition,
                                                                      return_gauss_params=True)
    crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_flow = log_fs[:, 0]
    log_ratio = log_r + log_pb - log_pf

    # trajectory balance loss
    tb = log_flow - log_ratio
    tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')

    # VarGrad loss
    if gfn.conditional_flow_model:
        # reshape and take the mean over repeats
        # minimize the variance over repeats w.r.t., the norm
        log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
        vg_loss = 0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2
        #z_matching_loss = F.mse_loss(log_Z, log_flow.view(repeats, -1).mean(dim=0, keepdim=True), reduction='mean')

    else:
        # take the variance over the full unconditional batch
        log_Z = log_ratio.mean(dim=0, keepdim=True)
        vg_loss = 0.5 * (log_Z - log_ratio) ** 2
        #z_matching_loss = F.mse_loss(log_Z.repeat(len(log_flow)), log_flow, reduction='mean')

    # regularize policies for local smoothness with a small penalty
    #smoothness_loss = normed_smoothness_loss(torch.stack([means_f, logvars_f, means_b, logvars_b]))

    loss = vg_loss.mean() + tb_loss.mean()  #+ z_matching_loss  #+ smoothness_loss.mean() * 1

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return loss


def bwd_combo(terminal_state, gfn, log_r, discretizer, condition=None, repeats=10,
              return_exp: bool = False):
    if gfn.conditional_flow_model:  # do repeats if there are conditions, otherwise skip
        condition = condition.repeat(repeats, 1)
        terminal_state = terminal_state.repeat(repeats, 1)
        log_r = log_r.repeat(repeats)

    (states, log_pfs, log_pbs, log_fs,
     means_f, logvars_f, means_b, logvars_b) \
        = gfn.get_trajectory_bwd(terminal_state, discretizer, condition, return_gauss_params=True)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_flow = log_fs[:, 0]
    log_ratio = log_r + log_pb - log_pf

    # trajectory balance loss
    tb = log_flow - log_ratio
    tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')

    # VarGrad loss
    if gfn.conditional_flow_model:
        # reshape and take the mean over repeats
        # minimize the variance over repeats w.r.t., the norm
        log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
        vg_loss = 0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2
        #z_matching_loss = F.mse_loss(log_Z, log_flow.view(repeats, -1).mean(dim=0, keepdim=True), reduction='mean')

    else:
        # take the variance over the full unconditional batch
        log_Z = log_ratio.mean(dim=0, keepdim=True)
        vg_loss = 0.5 * (log_Z - log_ratio) ** 2
        #z_matching_loss = F.mse_loss(log_Z.repeat(len(log_flow)), log_flow, reduction='mean')

    # regularize policies for local smoothness with a small penalty
    #smoothness_loss = normed_smoothness_loss(torch.stack([means_f, logvars_f, means_b, logvars_b]))

    loss = vg_loss.mean() + tb_loss.mean()  #+ z_matching_loss  #+ smoothness_loss.mean() * 1

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
    else:
        return loss


def fwd_tb(initial_state, gfn, log_reward_fn, discretizer, mol_batch,
           exploration_std=None, return_exp=False, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer, exploration_std, condition)
    crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_flow = log_fs[:, 0]

    tb = (log_pf + log_flow - log_pb - log_r)
    tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')

    loss = tb_loss

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return loss


def fwd_greedy(initial_state, gfn, log_reward_fn, discretizer, mol_batch,
               exploration_std=None, return_exp=False, condition=None,
               traj_midpoint: int = 0,
               repeats: int = 1, entropy_penalty: float = 1.0,
               ):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer,
                                                              None, condition,
                                                              detach_traj=False)
    # optionally only evaluate performance from some mid-trajectory initial state, rather from the global init s0
    if traj_midpoint > 0:
        states[:, :traj_midpoint] = states[:, :traj_midpoint].detach()

    # # optionally, evaluate over a minibatch and penalize low entropy
    # loss = -torch.logsumexp(torch.stack(log_r_list), dim=0)
    # mean_reward = torch.stack(log_r_list).mean(dim=0)
    # var_reward = torch.stack(log_r_list).var(dim=0)
    # loss = -mean_reward + alpha * var_reward
    crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states, no_grad=False)
    loss = -log_r

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return loss


def fwd_tb_greedy(initial_state, gfn, log_reward_fn, discretizer, mol_batch,
                  exploration_std=None, return_exp=False, condition=None,
                  traj_midpoint: int = 0,
                  repeats: int = 1, entropy_penalty: float = 1.0,
                  ):
    skip_greedy = exploration_std(0) != 0

    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer,
                                                              exploration_std, condition,
                                                              detach_traj=skip_greedy)
    # optionally only evaluate performance from some mid-trajectory initial state, rather from the global init s0
    if traj_midpoint > 0:
        states[:, :traj_midpoint] = states[:, :traj_midpoint].detach()

    crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states, no_grad=skip_greedy)

    if skip_greedy:
        greedy_loss = torch.zeros_like(log_r)
    else:
        greedy_loss = -log_r

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_flow = log_fs[:, 0]

    tb = (log_pf + log_flow - log_pb - log_r.detach())
    tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')

    loss = tb_loss + greedy_loss

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return loss


def fwd_vg_greedy(initial_state, gfn, log_reward_fn, discretizer, mol_batch,
                  exploration_std=None, return_exp=False, condition=None,
                  traj_midpoint: int = 0,
                  repeats: int = 1, entropy_penalty: float = 1.0,
                  ):
    skip_greedy = exploration_std(0) != 0

    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer,
                                                              exploration_std, condition,
                                                              detach_traj=skip_greedy)
    # optionally only evaluate performance from some mid-trajectory initial state, rather from the global init s0
    if traj_midpoint > 0:
        states[:, :traj_midpoint] = states[:, :traj_midpoint].detach()

    crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states, no_grad=skip_greedy)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_ratio = log_r.detach() + log_pb - log_pf

    if gfn.conditional_flow_model:
        # reshape and take the mean over repeats
        # minimize the variance over repeats w.r.t., the norm
        log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
        if skip_greedy:
            greedy_loss = torch.zeros_like(log_Z)
        else:
            greedy_loss = -log_r
        greedy_loss = -log_r.view(repeats, -1).mean(dim=0, keepdim=True)

        vg_loss = 0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2
    else:
        # take the variance over the full unconditional batch
        log_Z = log_ratio.mean(dim=0, keepdim=True)
        if skip_greedy:
            greedy_loss = torch.zeros_like(log_r)
        else:
            greedy_loss = -log_r
        vg_loss = 0.5 * (log_Z - log_ratio) ** 2

    loss = vg_loss + greedy_loss

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return loss


def bwd_tb(terminal_state, gfn, log_r, discretizer, condition=None, return_exp: bool = False):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, discretizer, condition)
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_flow = log_fs[:, 0]

    tb = (log_pf + log_flow - log_pb - log_r)
    tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')

    loss = tb_loss

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
    else:
        return loss


def fwd_vg(initial_state, gfn,
           log_r,
           discretizer,
           mol_batch,
           exploration_std=None,
           return_exp=False,
           condition=None,
           repeats=10):
    if gfn.conditional_flow_model:
        condition = condition.repeat(repeats, 1)
        initial_state = initial_state.repeat(repeats, 1)
        mol_batch = collate_data_list(mol_batch.to_data_list() * repeats)

    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer, exploration_std, condition)
    crystal_batch, log_r = get_loss_reward(condition, log_r, mol_batch, return_exp, states)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_ratio = log_r + log_pb - log_pf

    if gfn.conditional_flow_model:
        # reshape and take the mean over repeats
        # minimize the variance over repeats w.r.t., the norm
        log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
        vg_loss = 0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2
    else:
        # take the variance over the full unconditional batch
        log_Z = log_ratio.mean(dim=0, keepdim=True)
        vg_loss = 0.5 * (log_Z - log_ratio) ** 2

    loss = vg_loss

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return loss


def bwd_vg(terminal_state, gfn, log_r, discretizer, condition=None, repeats=10,
           return_exp: bool = False):
    if gfn.conditional_flow_model:  # do repeats if there are conditions, otherwise skip
        condition = condition.repeat(repeats, 1)
        terminal_state = terminal_state.repeat(repeats, 1)
        log_r = log_r.repeat(repeats)

    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, discretizer, condition)
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_ratio = log_r + log_pb - log_pf

    if gfn.conditional_flow_model:
        # reshape and take the mean over repeats
        # minimize the variance over repeats w.r.t., the norm
        log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
        vg_loss = 0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2
    else:
        # take the variance over the full unconditional batch
        log_Z = log_ratio.mean(dim=0, keepdim=True)
        vg_loss = 0.5 * (log_Z - log_ratio) ** 2

    loss = vg_loss

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
    else:
        return loss


def db(initial_state, gfn, log_reward_fn, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, condition)
    with torch.no_grad():
        log_fs[:, -1] = log_reward_fn(states[:, -1], condition).detach()

    loss = 0.5 * ((log_pfs + log_fs[:, :-1] - log_pbs - log_fs[:, 1:]) ** 2).sum(-1)
    return loss


def subtb(initial_state, gfn, log_reward_fn, coef_matrix, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, condition)
    with torch.no_grad():
        log_fs[:, -1] = log_reward_fn(states[:, -1], condition).detach()

    diff_logp = log_pfs - log_pbs
    diff_logp_padded = torch.cat(
        (torch.zeros((diff_logp.shape[0], 1)).to(diff_logp),
         diff_logp.cumsum(dim=-1)),
        dim=1)
    A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)
    A2 = log_fs[:, :, None] - log_fs[:, None, :] + A1
    A2 = A2 ** 2
    return torch.stack([torch.triu(A2[i] * coef_matrix, diagonal=1).sum() for i in range(A2.shape[0])]).sum()


def bwd_mle(terminal_state, gfn, discretizer, log_r, condition=None, return_exp: bool = False):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, discretizer, condition)
    loss = -log_pfs.sum(-1)
    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
    else:
        return loss


def bwd_mle_batch(terminal_state, gfn, discretizer, log_r, condition=None, return_exp: bool = False, repeats: int = 10):
    """
    Use importance sampling on the backwards trajectories to weight the forward policy probability loss
    """

    condition = condition.repeat(repeats, 1)
    terminal_state = terminal_state.repeat(repeats, 1)
    log_r = log_r.repeat(repeats)

    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, discretizer, condition)

    log_pb_rep = log_pbs.sum(-1).view(repeats, -1)
    log_pf_rep = log_pfs.sum(-1).view(repeats, -1)

    # Shape: (repeats, batch_size)
    log_w = log_pf_rep - log_pb_rep

    # log-sum-exp over repeats (dim 0), then subtract log(repeats)
    log_p_marginal = torch.logsumexp(log_w, dim=0) - np.log(repeats)

    # Final loss: negative log marginal likelihood for each x_T
    loss = -log_p_marginal

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
    else:
        return loss


def get_gfn_forward_loss(mode, init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std=None,
                         return_exp=False, condition=None, repeats=10, reweight_T: Optional[float] = None):
    if mode == 'tb':
        out = fwd_tb(init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std,
                     return_exp=return_exp,
                     condition=condition)
    elif mode == 'vg':
        out = fwd_vg(init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std, return_exp=return_exp,
                     condition=condition, repeats=repeats)
    elif mode == 'combo':
        out = fwd_combo(init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std,
                        return_exp=return_exp,
                        condition=condition, repeats=repeats)
    elif mode == 'greedy':
        out = fwd_greedy(init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std,
                         return_exp=return_exp,
                         condition=condition)
    elif mode == 'tb_greedy':
        out = fwd_tb_greedy(
            init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std,
            return_exp=return_exp,
            condition=condition
        )
    elif mode == 'vg_greedy':
        out = fwd_vg_greedy(
            init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std,
            return_exp=return_exp,
            condition=condition
        )
    else:
        assert False

    losses, *rest = out
    if reweight_T is not None:  # optionally reweight losses to minimize large outliers
        weights = torch.softmax(-losses.detach() / reweight_T, dim=0) * len(losses)
        weights += 1e-2  # minimum relative contribution
        weights /= weights.sum()
        loss = (weights * losses).mean()

    else:
        loss = losses.mean()


    return loss, *rest


def get_gfn_backward_loss(mode, samples, gfn_model, rewards, discretizer, exploration_std=None, condition=None,
                          repeats=10,
                          return_exp=False, reweight_T: Optional[float] = None):
    if mode == 'tb':
        out = bwd_tb(samples, gfn_model, rewards, discretizer, condition=condition, return_exp=return_exp)
    elif mode == 'vg':
        out = bwd_vg(samples, gfn_model, rewards, discretizer, condition=condition, repeats=repeats,
                     return_exp=return_exp)
    elif mode == 'combo':
        out = bwd_combo(samples, gfn_model, rewards, discretizer, condition=condition, repeats=repeats,
                        return_exp=return_exp)
    elif mode == 'mle':
        out = bwd_mle(samples, gfn_model, discretizer, rewards, condition=condition, return_exp=return_exp)
    elif mode == 'mle_batch':
        out = bwd_mle_batch(samples, gfn_model, discretizer, rewards, condition=condition, return_exp=return_exp,
                            repeats=repeats)
    else:
        assert False

    losses, *rest = out
    if reweight_T is not None:  # optionally reweight losses to minimize large outliers
        weights = torch.softmax(-losses.detach() / reweight_T, dim=0) * len(losses)
        weights += 1e-2  # minimum relative contribution
        weights /= weights.sum()
        loss = (weights * losses).mean()

    else:
        loss = losses.mean()

    return loss, *rest
