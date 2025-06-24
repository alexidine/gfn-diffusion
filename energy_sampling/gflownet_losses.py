import torch.nn.functional as F
import torch
from mxtaltools.dataset_utils.utils import collate_data_list

from utils import get_gfn_init_state


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


def linear_trajectory_penalty(trajectory_length, initial_state, states, linear_penalty_coeff: float = 0.0):
    terminal_state = states[:, -1, :]
    n_steps = trajectory_length
    steps = torch.linspace(0, 1, n_steps + 1, device=initial_state.device)[None, :, None]
    linear_interp = initial_state[:, None, :] + steps * (terminal_state - initial_state)[:, None, :]
    # penalize the MSE against a linear path
    linear_penalty = (linear_interp - states).norm(dim=2).pow(2).mean(1)
    return linear_penalty * linear_penalty_coeff


def fwd_tb(initial_state, gfn, log_reward_fn, mol_batch, exploration_std=None, return_exp=False, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn, condition)
    crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_ratio = (log_pf + log_fs[:, 0] - log_pb - log_r)
    tb_loss = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio),
                               reduction='none')  # a more stable loss, though we lose some theoretical guarantees

    if gfn.bwd_policy == 'gaussian':
        # apply a penalty to regularize the policy towards linear paths
        linear_penalty = linear_trajectory_penalty(gfn.trajectory_length, initial_state, states)

        loss = tb_loss.mean() + linear_penalty.mean()
    else:
        loss = tb_loss.mean()

    if return_exp:
        return loss.mean(), states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return loss.mean()


def bwd_tb(terminal_state, gfn, log_r, exploration_std=None, condition=None, return_exp: bool = False):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, exploration_std, condition)
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_ratio = (log_pf + log_fs[:, 0] - log_pb - log_r)  #.clip(min=-100, max=100)    #loss = 0.5 * (log_ratio ** 2)
    tb_loss = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio),
                               reduction='none')  # a more stable loss, though we lose some theoretical guarantees

    if gfn.bwd_policy == 'gaussian':
        # guide the TB loss towards the initial state
        initial_state = get_gfn_init_state(len(terminal_state), terminal_state.shape[1], terminal_state.device)
        initial_state_loss = (initial_state - states[:, 0]).norm(dim=1).pow(2)

        # apply a penalty to regularize the policy towards linear paths
        linear_penalty = linear_trajectory_penalty(gfn.trajectory_length, initial_state, states)

        # only train the flow loss when the trajectory is already well behaved
        tb_coeff = 2 * F.sigmoid(-initial_state_loss.mean() * 100)  # increase the scalar inside to tighten the fit
        loss = tb_coeff * tb_loss.mean() + initial_state_loss.mean() + linear_penalty.mean()
    else:
        loss = tb_loss.mean()

    if return_exp:
        return loss.mean(), states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
    else:
        return loss.mean()


# def fwd_greedy(initial_state, gfn, log_reward_fn, mol_batch, exploration_std=None, return_exp=False, condition=None):
#     # connect forward policy model gradients to reward model
#     states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn, condition,
#                                                               keep_step_grads=True)
#     # keep gradients from reward model
#     crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states, no_grad=False)
#
#     loss = -log_r
#     if return_exp:
#         return loss.mean(), states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
#     else:
#         return loss.mean()


def fwd_vg(initial_state, gfn, log_reward_fn, mol_batch, exploration_std=None, return_exp=False,
           condition=None,
           repeats=10):
    condition = condition.repeat(repeats, 1)
    initial_state = initial_state.repeat(repeats, 1)
    mol_batch = collate_data_list(mol_batch.to_data_list() * repeats)
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn, condition)
    crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    # reshape and take the mean over repeats
    log_Z = (log_r + log_pb - log_pf).view(repeats, -1).mean(dim=0, keepdim=True)
    # minimize the variance over repeats w.r.t., the norm
    vg_loss = 0.5 * (log_Z + (log_pf - log_r - log_pb).view(repeats, -1)) ** 2

    if gfn.bwd_policy == 'gaussian':
        # apply a penalty to regularize the policy towards linear paths
        linear_penalty = linear_trajectory_penalty(gfn.trajectory_length, initial_state, states)
        loss = vg_loss.mean() + linear_penalty.mean()
    else:
        loss = vg_loss.mean()

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return loss


def bwd_vg(terminal_state, gfn, log_r, exploration_std=None, condition=None, repeats=10,
           return_exp: bool = False):
    condition = condition.repeat(repeats, 1)
    terminal_state = terminal_state.repeat(repeats, 1)
    log_r = log_r.repeat(repeats)

    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, exploration_std, condition)
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_Z = (log_r + log_pb - log_pf).view(repeats, -1).mean(dim=0, keepdim=True)
    vg_loss = 0.5 * (log_Z + (log_pf - log_r - log_pb).view(repeats, -1)) ** 2

    if gfn.bwd_policy == 'gaussian':
        # guide the TB loss towards the initial state
        initial_state = get_gfn_init_state(len(terminal_state), terminal_state.shape[1], terminal_state.device)
        initial_state_loss = (initial_state - states[:, 0]).norm(dim=1).pow(2)

        # apply a penalty to regularize the policy towards linear paths
        linear_penalty = linear_trajectory_penalty(gfn.trajectory_length, initial_state, states)

        # only train the flow loss when the trajectory is already well behaved
        vg_coeff = 2 * F.sigmoid(-initial_state_loss.mean() * 10)  # increase the scalar inside to tighten the fit
        loss = vg_coeff * vg_loss.mean() + initial_state_loss.mean() #+ linear_penalty.mean()
    else:
        loss = vg_loss.mean()

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
    else:
        return loss


### NOTE none of the below are up-to-date


def fwd_tb_avg(initial_state, gfn, log_reward_fn, mol_batch, exploration_std=None, return_exp=False, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn, condition)
    crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_Z = (log_r + log_pb - log_pf).mean(dim=0, keepdim=True)
    loss = log_Z + (log_pf - log_r - log_pb)
    if return_exp:
        return 0.5 * (
                loss ** 2).mean(), states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return 0.5 * (loss ** 2).mean()


def bwd_tb_avg(terminal_state, gfn, log_r, exploration_std=None, condition=None, return_exp=False):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, exploration_std, condition)
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_Z = (log_r + log_pb - log_pf).mean(dim=0, keepdim=True)
    loss = log_Z + (log_pf - log_r - log_pb)
    if return_exp:
        return 0.5 * (
                loss ** 2).mean(), states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
    else:
        return 0.5 * (loss ** 2).mean()


def db(initial_state, gfn, log_reward_fn, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn, condition)
    with torch.no_grad():
        log_fs[:, -1] = log_reward_fn(states[:, -1], condition).detach()

    loss = 0.5 * ((log_pfs + log_fs[:, :-1] - log_pbs - log_fs[:, 1:]) ** 2).sum(-1)
    return loss.mean()


def subtb(initial_state, gfn, log_reward_fn, coef_matrix, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn, condition)
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


def bwd_mle(terminal_state, gfn, log_reward_fn, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, exploration_std, log_reward_fn, condition)
    loss = -log_pfs.sum(-1)
    return loss.mean()


def get_gfn_forward_loss(mode, init_state, gfn_model, log_reward, coeff_matrix, mol_batch, exploration_std=None,
                         return_exp=False, condition=None, repeats=10):
    if mode == 'tb':
        return fwd_tb(init_state, gfn_model, log_reward, mol_batch, exploration_std,
                      return_exp=return_exp,
                      condition=condition)
    # if mode == 'greedy':
    #     return fwd_greedy(init_state, gfn_model, log_reward, mol_batch, exploration_std,
    #                   return_exp=return_exp,
    #                   condition=condition)
    elif mode == 'tb-avg':
        return fwd_tb_avg(init_state, gfn_model, log_reward, mol_batch, exploration_std, return_exp=return_exp,
                          condition=condition)
    elif mode == 'cond-tb-avg':
        return fwd_vg(init_state, gfn_model, log_reward, mol_batch, exploration_std, return_exp=return_exp,
                      condition=condition, repeats=repeats)
    elif mode == 'db':
        return db(init_state, gfn_model, log_reward, exploration_std, condition=condition)
    elif mode == 'subtb':
        return subtb(init_state, gfn_model, log_reward, coeff_matrix, exploration_std, condition=condition)
    else:
        assert False


def get_gfn_backward_loss(mode, samples, gfn_model, rewards, exploration_std=None, condition=None, repeats=10,
                          return_exp=False):
    if mode == 'tb':
        return bwd_tb(samples, gfn_model, rewards, exploration_std, condition=condition, return_exp=return_exp)
    elif mode == 'tb-avg':
        return bwd_tb_avg(samples, gfn_model, rewards, exploration_std, condition=condition, return_exp=return_exp)
    elif mode == 'cond-tb-avg':
        return bwd_vg(samples, gfn_model, rewards, exploration_std, condition=condition, repeats=repeats,
                      return_exp=return_exp)
    elif mode == 'mle':
        return bwd_mle(samples, gfn_model, rewards, exploration_std, condition=condition)
    else:
        assert False
