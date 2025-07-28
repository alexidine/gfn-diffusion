import math
from typing import Optional

import numpy as np
import torch.nn.functional as F
import torch
from mxtaltools.dataset_utils.utils import collate_data_list

from utils import compute_sample_overlap


def diagonal_gaussian_log_density(x, mu, sigma):
    """
    x     : (batch_size, d)
    mu    : (d,) or (batch_size, d)
    sigma : (d,) or (batch_size, d) — standard deviation (not variance!)

    Returns: (batch_size,) log-probability under diagonal Gaussian
    """
    var = sigma ** 2
    log_term = torch.log(2 * math.pi * var)
    sq_term = ((x - mu) ** 2) / var
    log_density = -0.5 * (log_term + sq_term)
    return log_density.sum(dim=1)  # sum over dimensions


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
                         repeats=10, reweight_T: Optional[float] = None,
                         report_losses: bool = False,
                         ):
    if gfn.conditional_flow_model and any([
        loss_coeffs.var > 0, loss_coeffs.vg_lb > 0,
        loss_coeffs.vg_lme > 0, loss_coeffs.buffer > 0,
        loss_coeffs.overlap > 0
    ]):
        condition = condition.repeat(repeats, 1)
        initial_state = initial_state.repeat(repeats, 1)
        reassign_sgs = False
        log_T_tensor = log_T_tensor.repeat(repeats)
        if hasattr(mol_batch, 'sg_ind'):
            sg_inds = mol_batch.sg_ind.repeat(repeats)
            reassign_sgs = True
        mol_batch = collate_data_list(mol_batch.to_data_list() * repeats)
        if reassign_sgs:
            mol_batch.sg_ind = sg_inds

    loss_dict = {}

    if loss_coeffs.greedy > 0 or loss_coeffs.var > 0 or loss_coeffs.buffer > 0:
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

    if loss_coeffs.reinforce > 0:
        log_r_det = log_r.detach()
        centered_log_r = log_r_det - log_r_det.mean()
        reinforce_loss = -centered_log_r * log_pf
        losses.append(reinforce_loss * loss_coeffs.reinforce)

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

    if loss_coeffs.emp_z > 0:  # train the flow model to match the empirical log Z distribution
        if gfn.conditional_flow_model:
            emp_z_loss = (0.5 * (log_Z - log_flow.view(repeats, -1)) ** 2).view(-1)
        else:
            emp_z_loss = 0.5 * (log_Z - log_flow) ** 2
        losses.append(emp_z_loss * loss_coeffs.emp_z)

    """MLE/TPM loss"""
    if loss_coeffs.mle > 0:
        mle_loss = soft_saturate(-log_pbs.sum(-1))
        losses.append(mle_loss * loss_coeffs.mle)

    """Variance loss"""
    if loss_coeffs.var > 0:
        states_to_compare = states[:, -1].clip(min=-6, max=6)  # states[:, -1, 3:].clip(min=-6, max=6)
        dimwise_var_cutoff = torch.ones(12, device=states.device) * loss_coeffs.var_cutoff
        dimwise_var_cutoff[2] /= 5  # be gentle on the c-dimension
        if gfn.conditional_flow_model:
            states_reshaped = states_to_compare.view(repeats, -1, states_to_compare.shape[-1])

            # Compute variance within each condition
            batch_var = states_reshaped.var(dim=0)  # (repeats, batch, features)
            # penalize any variant dimensions below the minimum value
            small_var_loss = F.relu(dimwise_var_cutoff[None, :] - batch_var) / dimwise_var_cutoff[None, :]
            large_var_loss = F.relu(batch_var - 16)  # penalize very high variances
            var_gap = small_var_loss + large_var_loss
            var_loss = ((var_gap ** 2) * F.softmax(var_gap, dim=1)).sum(dim=1, keepdim=True).repeat(1, repeats).view(-1)
        else:
            batch_var = states_to_compare.var(dim=0)  # (batch, features)
            small_var_loss = F.relu(dimwise_var_cutoff - batch_var) / dimwise_var_cutoff
            large_var_loss = F.relu(batch_var - 16)
            var_gap = small_var_loss + large_var_loss
            var_loss = ((var_gap ** 2) * F.softmax(var_gap, dim=0)).sum(dim=0, keepdim=True).repeat(len(states))

        losses.append(var_loss * loss_coeffs.var)

    if loss_coeffs.overlap > 0:
        states_to_compare = states[:, -1].clip(min=-6, max=6)  # states[:, -1, 3:].clip(min=-6, max=6)

        if gfn.conditional_flow_model:
            states_reshaped = states_to_compare.view(repeats, -1, states_to_compare.shape[-1])

            # Compute overlap within each condition
            overlap_loss = torch.zeros((len(states) // repeats, repeats), device=states.device)
            for i in range(len(overlap_loss)):
                condition_states = states_reshaped[:, i]
                overlap_loss[i] = compute_sample_overlap(
                    condition_states,
                    ga=loss_coeffs.var_gamma,
                    agg='mean',
                )
            overlap_loss = overlap_loss.view(-1)
        else:
            overlap_loss = compute_sample_overlap(
                states_to_compare,  # don't let it escape - it could cheat
                ga=loss_coeffs.var_gamma,
                agg='mean',
            )
        losses.append(overlap_loss * loss_coeffs.overlap)

    """Buffer distance loss"""
    if loss_coeffs.buffer > 0:
        assert not gfn.conditional_flow_model, "Buffer loss not yet set up for conditional sampling"
        states_to_compare = states[:, -1].clip(min=-6, max=6)  #states[:, -1, 3:].clip(min=-6, max=6)
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

    if report_losses:
        loss_dict = {}
        if loss_coeffs.greedy > 0:
            loss_dict['greedy'] = greedy_loss.mean().detach()
        if loss_coeffs.reinforce > 0:
            loss_dict['reinforce'] = reinforce_loss.mean().detach()
        if loss_coeffs.smoothed > 0:
            loss_dict['smoothed'] = smoothness_loss.mean().detach()
        if loss_coeffs.tb > 0:
            loss_dict['tb'] = tb_loss.mean().detach()
        if loss_coeffs.vg_lb > 0:
            loss_dict['vg_lb'] = vg_loss.mean().detach()
        if loss_coeffs.vg_lme > 0:
            loss_dict['vg_lme'] = vg_loss.mean().detach()
        if loss_coeffs.emp_z > 0:
            loss_dict['emp_z'] = emp_z_loss.mean().detach()
        if loss_coeffs.mle > 0:
            loss_dict['mle'] = mle_loss.mean().detach()
        if loss_coeffs.var > 0:
            loss_dict['var'] = var_loss.mean().detach()
        if loss_coeffs.overlap > 0:
            loss_dict['overlap'] = overlap_loss.mean().detach()
        if loss_coeffs.buffer > 0:
            loss_dict['buffer'] = buffer_loss.mean().detach()
    else:
        loss_dict = None

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), crystal_batch, loss_dict
    else:
        return loss, loss_dict


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
                          return_exp=False,
                          reweight_T: Optional[float] = None,
                          report_losses: bool = False):
    if loss_coeffs.mle_prior_fraction > 0:
        # replace buffer samples with a random prior
        prior_samples = (torch.randn_like(samples) * loss_coeffs.pmle_std).clip(min=-6, max=6)
        if loss_coeffs.mle_prior_fraction < 1:
            num_to_replace = max(1, int(len(samples) * loss_coeffs.mle_prior_fraction))
            inds_to_replace = np.random.choice(len(samples), num_to_replace, replace=False)
            samples[inds_to_replace] = prior_samples[inds_to_replace]
        else:
            samples = prior_samples

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

    if loss_coeffs.emp_z > 0:  # train the flow model to match the empirical log Z distribution
        if gfn.conditional_flow_model:
            emp_z_loss = (0.5 * (log_Z - log_flow.view(repeats, -1)) ** 2).view(-1)
        else:
            emp_z_loss = 0.5 * (log_Z - log_flow) ** 2
        losses.append(emp_z_loss * loss_coeffs.emp_z)

    if loss_coeffs.mle > 0:
        mle_loss = soft_saturate(-log_pfs.sum(-1))
        losses.append(mle_loss * loss_coeffs.mle)

    combined_losses = torch.stack(losses).mean(dim=0)

    loss = reweight_losses(combined_losses, losses, reweight_T)

    if report_losses:
        loss_dict = {}
        if loss_coeffs.smoothed > 0:
            loss_dict['smoothed'] = smoothness_loss.mean().detach()
        if loss_coeffs.tb > 0:
            loss_dict['tb'] = tb_loss.mean().detach()
        if loss_coeffs.vg_lb > 0:
            loss_dict['vg_lb'] = vg_loss.mean().detach()
        if loss_coeffs.vg_lme > 0:
            loss_dict['vg_lme'] = vg_loss.mean().detach()
        if loss_coeffs.emp_z > 0:
            loss_dict['emp_z'] = emp_z_loss.mean().detach()
        if loss_coeffs.mle > 0:
            loss_dict['mle'] = mle_loss.mean().detach()
    else:
        loss_dict = None

    if return_exp:
        return loss, states.detach(), log_pfs.detach(), log_pbs.detach(), log_r.detach(), log_fs.detach(), loss_dict
    else:
        return loss, loss_dict
