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


def fwd_tb(initial_state, gfn, log_reward_fn, mol_batch, exploration_std=None, return_exp=False, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn, condition)
    crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_ratio = (log_pf + log_fs[:, 0] - log_pb - log_r)  #.clip(min=-10, max=10)
    #loss = 0.5 * (log_ratio ** 2)
    loss = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio),
                            reduction='none')  # a more stable loss, though we lose some theoretical guarantees
    if return_exp:
        return loss.mean(), states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return loss.mean()


def bwd_tb(terminal_state, gfn, log_r, exploration_std=None, condition=None, return_exp: bool = False):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, exploration_std, condition)
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_ratio = (log_pf + log_fs[:, 0] - log_pb - log_r)  #.clip(min=-100, max=100)    #loss = 0.5 * (log_ratio ** 2)
    loss = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio),
                            reduction='none')  # a more stable loss, though we lose some theoretical guarantees

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


def fwd_tb_avg_cond(initial_state, gfn, log_reward_fn, mol_batch, exploration_std=None, return_exp=False,
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
    loss = log_Z + (log_pf - log_r - log_pb).view(repeats, -1)
    if return_exp:
        return 0.5 * (
                loss ** 2).mean(), states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return 0.5 * (loss ** 2).mean()


def bwd_tb_avg_cond(terminal_state, gfn, log_r, exploration_std=None, condition=None, repeats=10,
                    return_exp: bool = False):
    condition = condition.repeat(repeats, 1)
    terminal_state = terminal_state.repeat(repeats, 1)
    log_r = log_r.repeat(repeats)

    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, exploration_std, condition)
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_Z = (log_r + log_pb - log_pf).view(repeats, -1).mean(dim=0, keepdim=True)
    loss = log_Z + (log_pf - log_r - log_pb).view(repeats, -1)
    if return_exp:
        return 0.5 * (
                loss ** 2).mean(), states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
    else:
        return 0.5 * (loss ** 2).mean()


### NOTE none of the below are up-to-date
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
