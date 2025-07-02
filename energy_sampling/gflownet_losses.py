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


def fwd_tb(initial_state, gfn, log_reward_fn, discretizer, mol_batch,
           exploration_std=None, return_exp=False, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, discretizer, exploration_std, condition)
    crystal_batch, log_r = get_loss_reward(condition, log_reward_fn, mol_batch, return_exp, states)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_ratio = (log_pf + log_fs[:, 0] - log_pb - log_r)
    #tb_loss = F.mse_loss(log_ratio, torch.zeros_like(log_ratio), reduction='none')
    tb_loss = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), reduction='none')

    loss = tb_loss.mean()

    if return_exp:
        return loss.mean(), states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch
    else:
        return loss.mean()


def bwd_tb(terminal_state, gfn, log_r, discretizer, condition=None, return_exp: bool = False):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, discretizer, condition)
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    log_ratio = (log_pf + log_fs[:, 0] - log_pb - log_r)
    #tb_loss = F.mse_loss(log_ratio, torch.zeros_like(log_ratio), reduction='none')
    tb_loss = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), reduction='none')

    loss = tb_loss.mean()

    if return_exp:
        return loss.mean(), states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
    else:
        return loss.mean()


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

    if gfn.conditional_flow_model:
        # reshape and take the mean over repeats
        # minimize the variance over repeats w.r.t., the norm
        log_ratio = log_r + log_pb - log_pf
        log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
        vg_loss = 0.5 * (log_Z + (log_pf - log_r - log_pb).view(repeats, -1)) ** 2
        #vg_loss = F.smooth_l1_loss(log_Z, log_ratio, reduction='none')  # smoother
    else:
        # take the variance over the full unconditional batch
        log_ratio = log_r + log_pb - log_pf
        log_Z = log_ratio.mean(dim=0, keepdim=True)
        vg_loss = 0.5 * (log_Z + (log_pf - log_r - log_pb)) ** 2
        #vg_loss = F.smooth_l1_loss(log_Z.repeat(len(log_ratio)), log_ratio, reduction='none')  # smoother

    loss = vg_loss.mean()

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

    if gfn.conditional_flow_model:
        # reshape and take the mean over repeats
        # minimize the variance over repeats w.r.t., the norm
        log_ratio = log_r + log_pb - log_pf
        log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
        vg_loss = 0.5 * (log_Z + (log_pf - log_r - log_pb).view(repeats, -1)) ** 2
        #vg_loss = F.smooth_l1_loss(log_Z, log_ratio, reduction='none')  # smoother
    else:
        # take the variance over the full unconditional batch
        log_ratio = log_r + log_pb - log_pf
        log_Z = log_ratio.mean(dim=0, keepdim=True)
        vg_loss = 0.5 * (log_Z + (log_pf - log_r - log_pb)) ** 2
        #vg_loss = F.smooth_l1_loss(log_Z.repeat(len(log_ratio)), log_ratio, reduction='none')  # smoother

    loss = vg_loss.mean()

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach()
    else:
        return loss


def db(initial_state, gfn, log_reward_fn, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, condition)
    with torch.no_grad():
        log_fs[:, -1] = log_reward_fn(states[:, -1], condition).detach()

    loss = 0.5 * ((log_pfs + log_fs[:, :-1] - log_pbs - log_fs[:, 1:]) ** 2).sum(-1)
    return loss.mean()


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


def bwd_mle(terminal_state, gfn, log_reward_fn, exploration_std=None, condition=None):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, condition)
    loss = -log_pfs.sum(-1)
    return loss.mean()


def get_gfn_forward_loss(mode, init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std=None,
                         return_exp=False, condition=None, repeats=10):
    if mode == 'tb':
        return fwd_tb(init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std,
                      return_exp=return_exp,
                      condition=condition)
    elif mode == 'vg':
        return fwd_vg(init_state, gfn_model, log_reward, discretizer, mol_batch, exploration_std, return_exp=return_exp,
                      condition=condition, repeats=repeats)
    # elif mode == 'db':
    #     return db(init_state, gfn_model, log_reward, exploration_std, condition=condition)
    # elif mode == 'subtb':
    #     return subtb(init_state, gfn_model, log_reward, coeff_matrix, exploration_std, condition=condition)
    else:
        assert False


def get_gfn_backward_loss(mode, samples, gfn_model, rewards, discretizer, exploration_std=None, condition=None, repeats=10,
                          return_exp=False):
    if mode == 'tb':
        return bwd_tb(samples, gfn_model, rewards, discretizer, condition=condition, return_exp=return_exp)
    elif mode == 'vg':
        return bwd_vg(samples, gfn_model, rewards, discretizer, condition=condition, repeats=repeats,
                      return_exp=return_exp)

    else:
        assert False
