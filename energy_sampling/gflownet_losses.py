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
    x_T = states[:, -1]  # terminal state
    if no_grad:
        x_T = x_T.detach()
        ctx = torch.no_grad()
    else:
        ctx = torch.enable_grad()

    with ctx:
        if return_exp:
            log_r, crystal_batch = log_reward_fn(x_T, mol_batch, log_T_tensor, return_exp, keep_grads=not no_grad)
            crystal_batch = crystal_batch.detach().to('cpu')
        else:
            log_r = log_reward_fn(states[:, -1], mol_batch, log_T_tensor, return_exp, keep_grads=not no_grad)
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
                         exploration_std=None,
                         return_exp=False,
                         condition=None,
                         repeats=10,
                         report_losses: bool = False,
                         ):
    if gfn.conditional and any([
        loss_coeffs.vg_lb > 0,
        loss_coeffs.vg_lme > 0]):  # todo rewrite all this
        assert False, "Rewrite repeats method from sampler"

    condition = condition.to(gfn.device)
    log_T_tensor = log_T_tensor.to(gfn.device)
    (states, log_pfs, log_pbs, log_flow) = gfn.get_traj_fwd(initial_state,
                                                            discretizer,
                                                            exploration_std,
                                                            condition,
                                                            mol_batch,
                                                            detach_traj=loss_coeffs.traj_grads == 0,
                                                            return_gauss_params=False,
                                                            )
    log_Z_learned = log_flow[:, 0]

    crystal_batch, log_r = get_loss_reward(log_T_tensor,
                                           log_reward_fn,
                                           mol_batch,
                                           return_exp,
                                           states,
                                           no_grad=loss_coeffs.reward_grads == 0
                                           )
    log_flow[:, -1] = log_r

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)

    beta = loss_coeffs.beta

    losses = []
    """VarGrad - lower bound loss"""
    if loss_coeffs.vg_lb > 0:
        log_Z, vg_loss = vg_lb(gfn, log_pb, log_pf, log_r, loss_coeffs, repeats,
                               beta=beta)
        losses.append(vg_loss * loss_coeffs.vg_lb)

        """VarGrad - log mean exp loss"""
    elif loss_coeffs.vg_lme > 0:
        log_Z, vg_loss = vg_lme(gfn, log_pb, log_pf, log_r, repeats,
                                beta=beta)
        losses.append(vg_loss * loss_coeffs.vg_lme)

    if loss_coeffs.emp_z > 0:  # train the flow model to match the empirical log Z distribution
        emp_z_loss = emp_Z(gfn, log_Z, log_Z_learned, repeats,
                           beta=beta)
        losses.append(emp_z_loss * loss_coeffs.emp_z)
    else:
        emp_z_loss = None


    if loss_coeffs.db > 0:
        db_loss = get_db_loss(log_pfs, log_pbs, log_flow, beta=beta)
        losses.append(db_loss * loss_coeffs.db)

    if loss_coeffs.subtb > 0:
        subtb_loss = get_subtb_loss(log_pfs, log_pbs, log_flow, loss_coeffs.coeff_matrix, beta=beta)
        losses.append(subtb_loss * loss_coeffs.subtb)

    """TB loss"""
    if loss_coeffs.tb > 0:
        tb_loss = get_tb_loss(log_Z_learned, log_pb, log_pf, log_r,
                              detach_z=exploration_std > 0 if exploration_std is not None else False,
                              beta=beta)
        losses.append(tb_loss * loss_coeffs.tb)

    if loss_coeffs.loss_clip != -1:
        combined_losses = soft_clip(torch.stack(losses), loss_coeffs.loss_clip).mean(dim=0)
    else:
        combined_losses = torch.stack(losses).mean(dim=0)

    assert combined_losses.isfinite().all()
    loss = combined_losses.mean()

    if report_losses:
        loss_dict = {'log_pf': log_pf, 'log_pb': log_pb, 'log_Z': log_Z_learned, 'log_r': log_r}
        if loss_coeffs.tb > 0:
            loss_dict['tb'] = tb_loss.mean().detach()
        if loss_coeffs.vg_lb > 0:
            loss_dict['vg_lb'] = vg_loss.mean().detach()
        if loss_coeffs.vg_lme > 0:
            loss_dict['vg_lme'] = vg_loss.mean().detach()
        if loss_coeffs.emp_z > 0:
            loss_dict['emp_z'] = emp_z_loss.mean().detach()
        if loss_coeffs.db > 0:
            loss_dict['db'] = db_loss.mean().detach()
        if loss_coeffs.subtb > 0:
            loss_dict['subtb'] = subtb_loss.mean().detach()

    else:
        loss_dict = None

    if return_exp:
        return loss, crystal_batch.cpu().detach(), loss_dict
    else:
        return loss, loss_dict


def get_gfn_backward_loss(loss_coeffs,
                          samples,
                          gfn,
                          log_r,
                          discretizer,
                          mol_batch,
                          condition=None,
                          repeats=10,
                          report_losses: bool = False,
                          ):
    if gfn.conditional and repeats > 1:
        assert False, "Not yet implemented"

    states, log_pfs, log_pbs, log_flow = gfn.get_traj_bwd(
        samples, discretizer, condition, mol_batch,
        return_gauss_params=False,
        detach_traj=loss_coeffs.traj_grads == 0)
    log_Z_learned = log_flow[:, 0]
    log_flow[:, -1] = log_r

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)

    beta = loss_coeffs.beta
    losses = []
    """VarGrad - lower bound loss"""
    if loss_coeffs.vg_lb > 0:
        log_Z_emp, vg_loss = vg_lb(gfn, log_pb, log_pf, log_r, loss_coeffs, repeats,
                                   beta=beta)

        losses.append(vg_loss * loss_coeffs.vg_lb)

        """VarGrad - log mean exp loss"""
    elif loss_coeffs.vg_lme > 0:
        log_Z_emp, vg_loss = vg_lme(gfn, log_pb, log_pf, log_r, repeats,
                                    beta=beta)
        losses.append(vg_loss * loss_coeffs.vg_lme)

    if loss_coeffs.pf_boost > 0:
        pf_boost_loss = get_pf_retention_loss(log_Z_learned, log_pb, log_pf, log_r, beta=beta)
        losses.append(pf_boost_loss * loss_coeffs.pf_boost)

    if loss_coeffs.db > 0:
        db_loss = get_db_loss(log_pfs, log_pbs, log_flow, beta=beta)
        losses.append(db_loss * loss_coeffs.db)

    if loss_coeffs.subtb > 0:
        subtb_loss = get_subtb_loss(log_pfs, log_pbs, log_flow, loss_coeffs.coeff_matrix, beta=beta)
        losses.append(subtb_loss * loss_coeffs.subtb)

    if loss_coeffs.emp_z > 0:  # train the flow model to match the empirical log Z distribution
        emp_z_loss = emp_Z(gfn, log_Z_emp, log_Z_learned, repeats,
                           beta=beta)
        losses.append(emp_z_loss * loss_coeffs.emp_z)
    else:
        emp_z_loss = None

    """TB loss"""
    tb_loss = get_tb_loss(log_Z_learned, log_pb, log_pf, log_r,
                          detach_z=True if loss_coeffs.bwd_tb_z == 0 else False,
                          z_only=True if loss_coeffs.bwd_tb_z == 2 else False,
                          beta=beta)
    if loss_coeffs.tb > 0:
        losses.append(tb_loss * loss_coeffs.tb)

    if loss_coeffs.mle > 0:
        mle_loss = terminal_mle(
            log_pf, log_pb,
            repeats,
            repeats > 1,
            estimator='exact' if repeats > 1 else 'bound',
            dreg=True
        )
        losses.append(mle_loss * loss_coeffs.mle)

    if loss_coeffs.loss_clip != -1:
        combined_losses = soft_clip(torch.stack(losses), loss_coeffs.loss_clip).mean(dim=0)
    else:
        combined_losses = torch.stack(losses).mean(dim=0)
    loss = combined_losses.mean()

    if report_losses:
        loss_dict = {'losses': combined_losses.detach(),
                     'log_pf': log_pf, 'log_pb': log_pb,
                     'log_Z': log_Z_learned, 'log_r': log_r}
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
        if loss_coeffs.db > 0:
            loss_dict['db'] = db_loss.mean().detach()
        if loss_coeffs.subtb > 0:
            loss_dict['subtb'] = subtb_loss.mean().detach()
        if loss_coeffs.pf_boost > 0:
            loss_dict['pf_boost'] = pf_boost_loss.mean().detach()

    else:
        loss_dict = None

    return loss, loss_dict


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


def get_db_loss(log_pfs, log_pbs, log_flow, beta: float = 10.0):
    residual = log_pfs + log_flow[:, :-1] - log_pbs - log_flow[:, 1:]

    loss = beta * F.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        reduction='none',
        beta=beta,
    ).sum(-1)
    return loss


def get_subtb_loss(log_pfs, log_pbs, log_flow, coeff_matrix, beta: float = 10.0):
    diff_logp = log_pfs - log_pbs
    diff_logp_padded = torch.cat(
        (torch.zeros((diff_logp.shape[0], 1)).to(diff_logp),
         diff_logp.cumsum(dim=-1)),
        dim=1)
    A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)
    A2 = log_flow[:, :, None] - log_flow[:, None, :] + A1
    A2 = beta * F.smooth_l1_loss(
        A2,
        torch.zeros_like(A2),
        reduction='none',
        beta=beta,
    )
    mask = torch.triu(torch.ones_like(coeff_matrix), diagonal=1)
    w = coeff_matrix * mask  # precompute once, ideally outside
    loss = (A2 * w).sum(dim=(-2, -1))  # or .mean() over batch
    return loss


def get_tb_loss(log_Z_learned, log_pb, log_pf, log_r, detach_z: Union[bool, torch.Tensor] = False, z_only=False,
                beta: float = 10):
    log_reward = log_r
    if z_only:
        tb = (log_pf.detach() + log_Z_learned - log_pb.detach() - log_reward)
    elif isinstance(detach_z, bool):
        if detach_z:
            tb = (log_pf + log_Z_learned.detach() - log_pb - log_reward)
        else:
            tb = (log_pf + log_Z_learned - log_pb - log_reward)
    elif torch.is_tensor(detach_z):
        log_Z_per_traj = torch.where(detach_z, log_Z_learned.detach(), log_Z_learned)
        # TB residual uses per-trajectory log_Z
        tb = (log_pf + log_Z_per_traj - log_pb - log_reward)
    else:
        tb = (log_pf + log_Z_learned - log_pb - log_reward)

    # tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')
    tb_loss = beta * F.smooth_l1_loss(tb, torch.zeros_like(tb), reduction='none', beta=beta)
    return tb_loss


def emp_Z(gfn, log_Z, log_Z_learned, repeats, beta: float = 10):
    if gfn.conditional:
        emp_z_loss = beta * F.smooth_l1_loss(log_Z.repeat(repeats, 1), log_Z_learned.view(repeats, -1),
                                             reduction='none',
                                             beta=beta).view(-1)
    else:
        emp_z_loss = beta * F.smooth_l1_loss(log_Z.repeat(len(log_Z_learned)), log_Z_learned, reduction='none',
                                             beta=beta)
    return emp_z_loss


def vg_lme(gfn, log_pb, log_pf, log_r, repeats, beta: float = 10):
    log_ratio = log_r + log_pb - log_pf
    if gfn.conditional:
        log_Z = torch.logsumexp(log_ratio.view(repeats, -1), dim=0, keepdim=True) - math.log(repeats)
        # vg_loss = (0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2).view(-1)
        vg_loss = beta * F.smooth_l1_loss(log_Z.repeat(repeats, 1), log_ratio.view(repeats, -1), reduction='none',
                                          beta=beta).view(-1)
    else:
        log_Z = torch.logsumexp(log_ratio, dim=0, keepdim=True) - math.log(repeats)
        # vg_loss = 0.5 * (log_Z - log_ratio) ** 2
        vg_loss = beta * F.smooth_l1_loss(log_Z.repeat(len(log_ratio)), log_ratio, reduction='none', beta=beta)
    return log_Z, vg_loss



def vg_lb(gfn, log_pb, log_pf, log_r, loss_coeffs, repeats, beta: float = 10):
    assert not (loss_coeffs.vg_lb > 0 and loss_coeffs.vg_lme > 0), \
        "Cannot use both vg_lb and vg_lme simultaneously"
    log_ratio = log_r + log_pb - log_pf
    if gfn.conditional:
        log_Z = log_ratio.view(repeats, -1).mean(dim=0, keepdim=True)
        # vg_loss = (0.5 * (log_Z - log_ratio.view(repeats, -1)) ** 2).view(-1)
        vg_loss = beta * F.smooth_l1_loss(log_Z.repeat(repeats, 1), log_ratio.view(repeats, -1), reduction='none',
                                          beta=beta).view(-1)

    else:
        log_Z = log_ratio.mean(dim=0, keepdim=True)
        # vg_loss = 0.5 * (log_Z - log_ratio) ** 2
        vg_loss = beta * F.smooth_l1_loss(log_Z.repeat(len(log_ratio)), log_ratio, reduction='none', beta=beta)
    return log_Z, vg_loss


def get_pf_retention_loss(log_Z_learned, log_pb, log_pf, log_r, beta: float = 10):
    # detached target: what log_pf "should" be at this terminal under DB.
    C = (log_Z_learned.detach() - log_pb.detach() - log_r.detach())
    delta = log_pf + C  # TB residual; grad flows ONLY via log_pf

    neg = delta.clamp(max=0.0)  # = delta where forgotten (delta<0), else 0
    # half-wave Huber: only the up-push; the mode-sharpening branch is gone
    loss = beta * F.smooth_l1_loss(neg, torch.zeros_like(neg),
                                   reduction='none', beta=beta)
    return loss
