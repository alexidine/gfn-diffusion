import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

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
    log_temperature = log_T_tensor if log_T_tensor is not None else None

    x_T = states[:, -1]
    if no_grad:
        x_T = x_T.detach()
        ctx = torch.no_grad()
    else:
        ctx = torch.enable_grad()

    with ctx:
        if return_exp:
            log_r, crystal_batch = log_reward_fn(x_T, mol_batch, log_temperature, return_exp)
            crystal_batch = crystal_batch.detach().to('cpu')
        else:
            log_r = log_reward_fn(states[:, -1], mol_batch, log_temperature, return_exp)
            crystal_batch = None

    if no_grad:
        log_r = log_r.detach()

    return crystal_batch, log_r


def normed_smoothness_loss(x, eps=1e-5):
    second_diff = torch.diff(x, n=2, dim=-1)
    curvature = (second_diff ** 4).mean(dim=-1)  # specifically punish very large changes
    variance = torch.var(x, dim=-1, correction=0)
    return curvature / (variance + eps)


def soft_saturate(x, scale: Optional[float] = 10.0):
    return torch.log(torch.abs(x / scale) + 1) * torch.sign(x)


def soft_clip(x, cutoff):
    abs_x = x.abs()
    sign_x = x.sign()
    # Match value and slope at cutoff using a shifted log
    delta = abs_x - cutoff
    clipped = cutoff + torch.log1p(delta.clip(min=1e-3))  # log1p = log(1 + x), safer numerically
    return torch.where(abs_x <= cutoff, x, sign_x * clipped)


def get_gfn_forward_loss(loss_coeffs,
                         initial_state,
                         gfn,
                         log_reward_fn,
                         discretizer,
                         mol_batch,
                         log_T_tensor,
                         exploration_std=None, return_exp=False, condition=None,
                         repeats=10,
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
        z_primes = mol_batch.z_prime
        mol_batch.max_z_prime = int(torch.amax(mol_batch.z_prime))
        mol_batch.is_well_defined = torch.ones(mol_batch.num_graphs, dtype=bool, device=mol_batch.device)
        mol_batch = collate_data_list(mol_batch.batch_to_list() * repeats)
        mol_batch.z_prime = z_primes.repeat(repeats)
        mol_batch.max_z_prime = int(torch.amax(mol_batch.z_prime))
        if reassign_sgs:
            mol_batch.sg_ind = sg_inds

    if loss_coeffs.greedy > 0 or loss_coeffs.var > 0 or loss_coeffs.buffer > 0:
        keep_grads = True
    else:
        keep_grads = False

    condition = condition.to(gfn.device)
    log_T_tensor = log_T_tensor.to(gfn.device)
    (states, log_pfs, log_pbs, log_flow) = gfn.get_traj_fwd(initial_state,
                                                            discretizer,
                                                            exploration_std,
                                                            condition,
                                                            detach_traj=not keep_grads,
                                                            return_gauss_params=False,
                                                            )

    crystal_batch, log_r = get_loss_reward(log_T_tensor,
                                           log_reward_fn,
                                           mol_batch,
                                           return_exp,
                                           states,
                                           no_grad=(loss_coeffs.greedy == 0)
                                           )
    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)
    # del log_pfs, log_pbs

    losses = []
    """greedy loss"""
    if loss_coeffs.greedy > 0:
        assert False, "Grads are hard detached in the energy function currently. Greedy training will fail."
        greedy_loss = -power_saturate(log_r, 0.7)
        losses.append(greedy_loss * loss_coeffs.greedy)

    if loss_coeffs.reinforce > 0:  # todo maybe do repeats over conditioning here for the weight
        log_r_det = log_r.detach()
        centered_log_r = log_r_det - log_r_det.mean()
        weight = F.softmax(centered_log_r / 10, dim=0) + 1e-2 / len(log_r)
        weight /= weight.sum()
        weight *= len(weight)
        reinforce_loss = -power_saturate(weight * log_pf, 0.7)
        losses.append(reinforce_loss * loss_coeffs.reinforce)

    """VarGrad - lower bound loss"""
    if loss_coeffs.vg_lb > 0:
        log_Z, vg_loss = vg_lb(gfn, log_pb, log_pf, log_r, loss_coeffs, repeats)

        losses.append(vg_loss * loss_coeffs.vg_lb)

        """VarGrad - log mean exp loss"""
    elif loss_coeffs.vg_lme > 0:
        log_Z, vg_loss = vg_lme(gfn, log_pb, log_pf, log_r, repeats)
        losses.append(vg_loss * loss_coeffs.vg_lme)

    if loss_coeffs.emp_z > 0:  # train the flow model to match the empirical log Z distribution
        emp_z_loss = emp_Z(gfn, log_Z, log_flow, repeats)
        losses.append(emp_z_loss * loss_coeffs.emp_z)
    else:
        emp_z_loss = None

    """TB loss"""
    if loss_coeffs.tb > 0:
        tb_loss = get_tb_loss(log_flow, log_pb, log_pf, log_r,
                              detach_z = exploration_std>0 if exploration_std is not None else False)
        losses.append(tb_loss * loss_coeffs.tb)

    """MLE/TPM loss"""
    if loss_coeffs.mle > 0:
        mle_loss = -power_saturate(log_pb, 0.7)
        losses.append(mle_loss * loss_coeffs.mle)

    # """Variance loss"""
    # if loss_coeffs.var > 0:
    #     var_loss = fwd_var_loss(gfn, loss_coeffs, repeats, states)
    #     losses.append(var_loss * loss_coeffs.var)
    #
    # """self overlap loss"""
    # if loss_coeffs.overlap > 0:
    #     overlap_loss = fwd_overlap_loss(gfn, loss_coeffs, repeats, states)
    #     losses.append(overlap_loss * loss_coeffs.overlap)
    #
    # """Buffer distance loss"""
    # if loss_coeffs.buffer > 0:
    #     buffer_loss = fwd_buffer_loss(buffer, gfn, loss_coeffs, states)
    #     losses.append(buffer_loss * loss_coeffs.buffer)

    if loss_coeffs.loss_clip != -1:
        combined_losses = soft_clip(torch.stack(losses), loss_coeffs.loss_clip).mean(dim=0)
    else:
        combined_losses = torch.stack(losses).mean(dim=0)

    assert combined_losses.isfinite().all()
    loss = combined_losses.mean()

    X_side = log_pf - log_pb
    Y_side = log_r - log_flow
    normed_tb_residual = (X_side - Y_side).abs() / torch.maximum(torch.ones_like(Y_side), Y_side.abs())
    normed_tb_residual = torch.nan_to_num(normed_tb_residual.detach())
    log_importance_weight = (Y_side - X_side).detach()

    tb_y = (log_flow + log_pf).cpu().detach().numpy()
    tb_x = (log_r + log_pb).cpu().detach().numpy()
    m, b = np.polyfit(tb_x, tb_y, 1)
    slope_err = abs(m - 1)
    intercept_err = abs(b) / np.std(tb_y)
    scatter_err = np.std(tb_x - tb_y)

    log_weight = log_r + log_pb - log_pf
    log_Z_lb = log_weight.mean()

    if report_losses:
        loss_dict = {}
        loss_dict['log_Z_lb'] = log_Z_lb
        loss_dict['normed_tb'] = normed_tb_residual.clip(max=normed_tb_residual.quantile(0.95)).mean()
        loss_dict['scatter_err'] = scatter_err
        loss_dict['slope_err'] = slope_err
        loss_dict['intercept_err'] = intercept_err
        if loss_coeffs.greedy > 0:
            loss_dict['greedy'] = greedy_loss.mean().detach()
        if loss_coeffs.reinforce > 0:
            loss_dict['reinforce'] = reinforce_loss.mean().detach()
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
        # if loss_coeffs.var > 0:
        #     loss_dict['var'] = var_loss.mean().detach()
        # if loss_coeffs.overlap > 0:
        #     loss_dict['overlap'] = overlap_loss.mean().detach()
        # if loss_coeffs.buffer > 0:
        #     loss_dict['buffer'] = buffer_loss.mean().detach()
    else:
        loss_dict = None

    if return_exp:
        return loss, crystal_batch.cpu().detach(), loss_dict, log_r, log_importance_weight
    else:
        return loss, loss_dict, log_r, log_importance_weight


def get_gfn_backward_loss(loss_coeffs,
                          samples,
                          gfn,
                          log_r,
                          discretizer,
                          condition=None,
                          repeats=10,
                          report_losses: bool = False):
    if gfn.conditional_flow_model and any([
        loss_coeffs.vg_lb > 0, loss_coeffs.vg_lme > 0
    ]):
        condition = condition.repeat(repeats, 1)
        samples = samples.repeat(repeats, 1)
        log_r = log_r.repeat(repeats)
        conditional_repeats = True
    else:
        conditional_repeats = False

    condition = condition.to(gfn.device)
    states, log_pfs, log_pbs, log_flow = gfn.get_traj_bwd(
        samples, discretizer, condition, return_gauss_params=False)

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)

    losses = []
    """VarGrad - lower bound loss"""
    if loss_coeffs.vg_lb > 0:
        log_Z, vg_loss = vg_lb(gfn, log_pb, log_pf, log_r, loss_coeffs, repeats)

        losses.append(vg_loss * loss_coeffs.vg_lb)

        """VarGrad - log mean exp loss"""
    elif loss_coeffs.vg_lme > 0:
        log_Z, vg_loss = vg_lme(gfn, log_pb, log_pf, log_r, repeats)
        losses.append(vg_loss * loss_coeffs.vg_lme)

    if loss_coeffs.emp_z > 0:  # train the flow model to match the empirical log Z distribution
        emp_z_loss = emp_Z(gfn, log_Z, log_flow, repeats)
        losses.append(emp_z_loss * loss_coeffs.emp_z)
    else:
        emp_z_loss = None

    """TB loss"""
    tb_loss = get_tb_loss(log_flow, log_pb, log_pf, log_r,
                          detach_z=True if loss_coeffs.bwd_tb_z == 0 else False,
                          z_only=True if loss_coeffs.bwd_tb_z == 2 else False)
    if loss_coeffs.tb > 0:
        losses.append(tb_loss * loss_coeffs.tb)

    if loss_coeffs.mle > 0:
        mle_loss = terminal_mle(
            log_pf, log_pb,
            repeats,
            conditional_repeats,
            estimator='exact' if conditional_repeats else 'bound',
            dreg=True
        )
        losses.append(mle_loss * loss_coeffs.mle)

    if loss_coeffs.loss_clip != -1:
        combined_losses = soft_clip(torch.stack(losses), loss_coeffs.loss_clip).mean(dim=0)
    else:
        combined_losses = torch.stack(losses).mean(dim=0)
    loss = combined_losses.mean()

    'normed tb reporting'
    X_side = log_pf - log_pb
    Y_side = log_r - log_flow
    normed_tb_residual = (X_side - Y_side).abs() / torch.maximum(torch.ones_like(Y_side), Y_side.abs())
    normed_tb_residual = torch.nan_to_num(normed_tb_residual.detach())

    log_weight = log_r + log_pb - log_pf
    log_Z_lb = log_weight.mean()

    tb_y = (log_flow + log_pf).cpu().detach().numpy()
    tb_x = (log_r + log_pb).cpu().detach().numpy()
    m, b = np.polyfit(tb_x, tb_y, 1)
    slope_err = abs(m - 1)
    intercept_err = abs(b) / np.std(tb_y)
    scatter_err = np.std(tb_x - tb_y)

    if report_losses:
        loss_dict = {}
        loss_dict['log_Z_lb'] = log_Z_lb
        loss_dict['normed_tb'] = normed_tb_residual.clip(
            max=normed_tb_residual.quantile(0.95)).mean()  # exclude outliers
        loss_dict['scatter_err'] = scatter_err
        loss_dict['slope_err'] = slope_err
        loss_dict['intercept_err'] = intercept_err
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

    return loss, loss_dict, tb_loss.detach()


def terminal_mle(
        log_pf, log_pb,
        reps: int = None,
        do_repeats: bool = False,
        estimator: str = "bound",  # "bound" (Jensen, eq. 28) or "exact" (IWAE, eq. 27)
        dreg: bool = True,  # use detached responsibilities for "exact"
):
    """
    Returns:
        loss: scalar tensor
        stats: dict with diagnostics
    """
    if not do_repeats:
        repeats = 1
    else:
        repeats = 1 * reps

    # reshape into [B, K] where B = number of distinct terminals (before repeating)
    if log_pf.numel() % repeats != 0:
        raise ValueError(f"log_pf size {log_pf.numel()} not divisible by repeats={repeats}")

    B = log_pf.numel() // repeats
    log_pf = log_pf.view(B, repeats)
    log_pb = log_pb.view(B, repeats)

    logw = log_pf - log_pb  # [B, K]  ; log importance weights for each path

    if estimator == "bound":
        # Eq. (28): E_{τ~Pb}[ log Pf(τ) - log Pb(τ|x) ]  (sample mean over paths)
        # No importance weights needed; just average over the K samples.
        loss = -logw.mean(dim=1)  # [B]

        return loss

    elif estimator == "exact":
        assert do_repeats, "Exact MLE estimator requires minibatch repeats > 1"
        # Eq. (27): log E_{τ~Pb}[ exp(logw) ] ≈ logsumexp(logw, dim=1) - log K
        if dreg:
            # DReG-style gradient: responsibilities detached to tame variance
            with torch.no_grad():
                alpha = torch.softmax(logw, dim=1)  # [B, K]
            # Equivalent gradient to IWAE objective under DReG; stable
            loss = -((alpha * logw).sum(dim=1))  # [B]
        else:
            # Plain IWAE objective (can be higher variance)
            lse = torch.logsumexp(logw, dim=1)  # [B]
            loss = -(lse - math.log(repeats))

        return loss.repeat(repeats)

    else:
        raise ValueError("estimator must be 'bound' or 'exact'")


def power_saturate(x, power):
    return torch.sign(x) * (torch.abs(x) ** power)


def get_tb_loss(log_flow, log_pb, log_pf, log_r, detach_z: Union[bool, torch.Tensor]=False, z_only=False):
    log_reward = log_r.detach()
    if z_only:
        tb = (log_pf.detach() + log_flow - log_pb.detach() - log_reward)
    elif isinstance(detach_z, bool):
        if detach_z:
            tb = (log_pf + log_flow.detach() - log_pb - log_reward)
        else:
            tb = (log_pf + log_flow - log_pb - log_reward)
    elif torch.is_tensor(detach_z):
        log_Z_per_traj = torch.where(detach_z, log_flow.detach(), log_flow)
        # TB residual uses per-trajectory log_Z
        tb = (log_pf + log_Z_per_traj - log_pb - log_reward)
    else:
        tb = (log_pf + log_flow - log_pb - log_reward)

    # tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')
    tb_loss = F.smooth_l1_loss(tb, torch.zeros_like(tb), reduction='none', beta=10)
    return tb_loss


def emp_Z(gfn, log_Z, log_flow, repeats):
    if gfn.conditional_flow_model:
        emp_z_loss = F.smooth_l1_loss(log_Z.repeat(repeats, 1), log_flow.view(repeats, -1), reduction='none',
                                      beta=10).view(-1)
    else:
        emp_z_loss = F.smooth_l1_loss(log_Z.repeat(len(log_flow)), log_flow, reduction='none', beta=10)
    return emp_z_loss


def vg_lme(gfn, log_pb, log_pf, log_r, repeats):
    log_ratio = log_r.detach() + log_pb - log_pf
    if gfn.conditional_flow_model:
        log_Z = torch.logsumexp(log_ratio.view(repeats, -1), dim=0, keepdim=True) - math.log(repeats)
        # vg_loss = (0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2).view(-1)
        vg_loss = F.smooth_l1_loss(log_Z.repeat(repeats, 1), log_ratio.view(repeats, -1), reduction='none',
                                   beta=10).view(-1)
    else:
        log_Z = torch.logsumexp(log_ratio, dim=0, keepdim=True) - math.log(repeats)
        # vg_loss = 0.5 * (log_Z - log_ratio) ** 2
        vg_loss = F.smooth_l1_loss(log_Z.repeat(len(log_ratio)), log_ratio, reduction='none', beta=10)
    return log_Z, vg_loss


def vg_lb(gfn, log_pb, log_pf, log_r, loss_coeffs, repeats):
    assert not (loss_coeffs.vg_lb > 0 and loss_coeffs.vg_lme > 0), \
        "Cannot use both vg_lb and vg_lme simultaneously"
    log_ratio = log_r.detach() + log_pb - log_pf
    if gfn.conditional_flow_model:
        log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
        # vg_loss = (0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2).view(-1)
        vg_loss = F.smooth_l1_loss(log_Z.repeat(repeats, 1), log_ratio.view(repeats, -1), reduction='none',
                                   beta=10).view(-1)

    else:
        log_Z = log_ratio.mean(dim=0, keepdim=True)
        # vg_loss = 0.5 * (log_Z - log_ratio) ** 2
        vg_loss = F.smooth_l1_loss(log_Z.repeat(len(log_ratio)), log_ratio, reduction='none', beta=10)
    return log_Z, vg_loss

#
# def fwd_buffer_loss(buffer, gfn, loss_coeffs, states):
#     assert not gfn.conditional_flow_model, "Buffer loss not yet set up for conditional sampling"
#     states_to_compare = states[:, -1].clip(min=-6, max=6)  # states[:, -1, 3:].clip(min=-6, max=6)
#     buffer_states = torch.stack(buffer.x_list).to(gfn.device)
#     buffer_to_compare = buffer_states[:, -1, 3:].clip(min=-6, max=6)
#     buffer_loss = compute_sample_overlap(
#         buffer_to_compare,
#         states_to_compare,  # don't let it escape - it could cheat
#         ga=loss_coeffs.buffer_gamma,
#     )
#     return buffer_loss

#
# def fwd_var_loss(gfn, loss_coeffs, repeats, states):
#     states_to_compare = states[:, -1].clip(min=-6, max=6)  # states[:, -1, 3:].clip(min=-6, max=6)
#     dimwise_var_cutoff = torch.ones(12, device=states.device) * loss_coeffs.var_cutoff
#     dimwise_var_cutoff[2] /= 5  # be gentle on the c-dimension
#     if gfn.conditional_flow_model:
#         states_reshaped = states_to_compare.view(repeats, -1, states_to_compare.shape[-1])
#
#         # Compute variance within each condition
#         batch_var = states_reshaped.var(dim=0)  # (repeats, batch, features)
#         # penalize any variant dimensions below the minimum value
#         small_var_loss = F.relu(dimwise_var_cutoff[None, :] - batch_var) / dimwise_var_cutoff[None, :]
#         large_var_loss = F.relu(batch_var - 16)  # penalize very high variances
#         var_gap = small_var_loss + large_var_loss
#         var_loss = ((var_gap ** 2) * F.softmax(var_gap, dim=1)).sum(dim=1, keepdim=True).repeat(1, repeats).view(-1)
#     else:
#         batch_var = states_to_compare.var(dim=0)  # (batch, features)
#         small_var_loss = F.relu(dimwise_var_cutoff - batch_var) / dimwise_var_cutoff
#         large_var_loss = F.relu(batch_var - 16)
#         var_gap = small_var_loss + large_var_loss
#         var_loss = ((var_gap ** 2) * F.softmax(var_gap, dim=0)).sum(dim=0, keepdim=True).repeat(len(states))
#     return var_loss

#
# def fwd_overlap_loss(gfn, loss_coeffs, repeats, states):
#     states_to_compare = states[:, -1].clip(min=-6, max=6)  # states[:, -1, 3:].clip(min=-6, max=6)
#     if gfn.conditional_flow_model:
#         states_reshaped = states_to_compare.view(repeats, -1, states_to_compare.shape[-1])
#
#         # Compute overlap within each condition
#         overlap_loss = torch.zeros((len(states) // repeats, repeats), device=states.device)
#         for i in range(len(overlap_loss)):
#             condition_states = states_reshaped[:, i]
#             overlap_loss[i] = compute_sample_overlap(
#                 condition_states,
#                 ga=loss_coeffs.var_gamma,
#                 agg='mean',
#             )
#         overlap_loss = overlap_loss.view(-1)
#     else:
#         overlap_loss = compute_sample_overlap(
#             states_to_compare,  # don't let it escape - it could cheat
#             ga=loss_coeffs.var_gamma,
#             agg='mean',
#         )
#     return overlap_loss
