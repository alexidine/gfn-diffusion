import torch
from torch.distributions import Normal
from mxtaltools.dataset_utils.utils import collate_data_list


def get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states):
    if condition is not None:
        log_temperature = condition[:, 0]
    else:
        log_temperature = None
    with torch.no_grad():
        if return_exp:
            log_r, crystal_batch = log_reward_fn(states[:, -1], mol_batch, log_temperature, return_exp)
            log_r = log_r.detach()
            crystal_batch = crystal_batch.detach()
        else:
            log_r = log_reward_fn(states[:, -1], mol_batch, log_temperature, return_exp).detach()
            crystal_batch = None
    return crystal_batch, log_r


def fwd_tb(initial_state, gfn, log_reward_fn, mol_batch, exploration_std=None, return_exp=False, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn, condition)
    crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_ratio = log_pf + log_fs[:, 0] - log_pb - log_r
    #log_ratio = log_ratio.clip(min=-10, max=10)  # clip extremely large losses
    loss = 0.5 * (log_ratio ** 2)
    if return_exp:
        return loss.mean(), states, log_pfs, log_pbs, log_r, crystal_batch
    else:
        return loss.mean()


def bwd_tb(initial_state, gfn, log_r, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(initial_state, exploration_std, condition)
    log_ratio = log_pfs.sum(-1) + log_fs[:, 0] - log_pbs.sum(-1) - log_r
    #log_ratio = log_ratio.clip(min=-10, max=10)  # clip extremely large losses
    loss = 0.5 * (log_ratio ** 2)

    return loss.mean()


def fwd_tb_avg(initial_state, gfn, log_reward_fn, mol_batch, exploration_std=None, return_exp=False, condition=None):
        states, log_pfs, log_pbs, _ = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn, condition)
        crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)
        log_pf = log_pfs.sum(-1)
        log_pb = log_pbs.sum(-1)
        log_Z = (log_r + log_pb - log_pf).mean(dim=0, keepdim=True)
        loss = log_Z + (log_pf - log_r - log_pb)
        if return_exp:
            return 0.5 * (loss ** 2).mean(), states, log_pfs, log_pbs, log_r, crystal_batch
        else:
            return 0.5 * (loss ** 2).mean()


def bwd_tb_avg(initial_state, gfn, log_r, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, _ = gfn.get_trajectory_bwd(initial_state, exploration_std, condition)
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_Z = (log_r + log_pb - log_pf).mean(dim=0, keepdim=True)
    loss = log_Z + (log_pf - log_r - log_pb)
    return 0.5 * (loss ** 2).mean()


def fwd_tb_avg_cond(initial_state, gfn, log_reward_fn, mol_batch, exploration_std=None, return_exp=False,
                    condition=None,
                    repeats=10):
    """
    This is the VarGrad forward loss
    :param initial_state:
    :param gfn:
    :param log_reward_fn:
    :param mol_batch:
    :param exploration_std:
    :param return_exp:
    :param condition:
    :param repeats:
    :return:
    """
    condition = condition.repeat(repeats, 1)
    initial_state = initial_state.repeat(repeats, 1)
    mol_batch = collate_data_list(mol_batch.to_data_list() * repeats)
    states, log_pfs, log_pbs, _ = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn, condition)
    crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    # reshape and take the mean over repeats
    log_Z = (log_r + log_pb - log_pf).view(repeats, -1).mean(dim=0, keepdim=True)
    # minimize the variance over repeats w.r.t., the norm
    loss = log_Z + (log_pf - log_r - log_pb).view(repeats, -1)
    #loss = loss.clip(min=-10, max=10)  # clip extreme outliers for stability
    if return_exp:
        return 0.5 * (loss ** 2).mean(), states, log_pfs, log_pbs, log_r, crystal_batch
    else:
        return 0.5 * (loss ** 2).mean()


def bwd_tb_avg_cond(initial_state, gfn, log_r, exploration_std=None, condition=None, repeats=10):
    condition = condition.repeat(repeats, 1)
    initial_state = initial_state.repeat(repeats, 1)
    log_r = log_r.repeat(repeats)

    states, log_pfs, log_pbs, _ = gfn.get_trajectory_bwd(initial_state, exploration_std, condition)
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_Z = (log_r + log_pb - log_pf).view(repeats, -1).mean(dim=0, keepdim=True)
    loss = log_Z + (log_pf - log_r - log_pb).view(repeats, -1)
    #loss = loss.clip(min=-10, max=10)  # clip extreme outliers for stability
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


def bwd_mle(samples, gfn, log_reward_fn, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(samples, exploration_std, log_reward_fn, condition)
    loss = -log_pfs.sum(-1)
    return loss.mean()
