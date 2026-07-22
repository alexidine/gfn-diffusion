import math
from typing import Optional

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


def update_and_lookup_condition_log_z(condition_log_z, condition_id, log_r, log_pb, log_pf, device,
                                      step: int, log_Z_learned=None, do_update: bool = True,
                                      mode_level_stream: Optional[str] = None):
    """
    Shared by get_gfn_forward_loss/get_gfn_backward_loss: whenever a
    ConditionLogZTracker is supplied, optionally feed it this step's
    empirical log importance weight (regardless of tb_z_source -- so the
    estimate keeps warming up even for loss groups still on the learned
    log_Z) and look up the current per-condition target for this batch.
    do_update=False skips the update but still performs the lookup, e.g.
    for backward trajectories outside phase 1/2, where log_Z_learned is
    detached from the backward TB loss (freeze_z) and the resulting
    importance weights are no longer trustworthy signal for the persistent
    estimate. Returns (None, None) when no tracker/condition_id is
    provided, so downstream code can pass these straight through to
    get_tb_loss unconditionally. step is the caller's current global
    training step (Modeller.step_ind), forwarded to
    ConditionLogZTracker.update() -- see its docstring for why the tracker's
    decay is keyed off elapsed training steps rather than elapsed calls.
    log_Z_learned, if given, also feeds ConditionLogZTracker.update_z_residual
    (same do_update gate -- an untrustworthy logw is untrustworthy for
    judging network calibration too), the per-condition monitoring signal
    behind rms_z_lag(). Only get_gfn_forward_loss passes it -- the z-residual
    monitor is deliberately on-policy only: the Z model is trained and judged
    on forward rollouts, bwd/replay are off-policy and it's the policy
    model's job (mode retention/coverage) to fix those, not the Z model's.
    mode_level_stream ('fwd'/'bwd'/None) additionally feeds this batch's logw
    into the tracker's matching per-mode level EMA (the z_match delta gate's
    two sides) -- NOT gated by do_update, deliberately: do_update guards
    ema_logw (a gradient-feeding blend of whatever touches it), while the
    level streams are pure measurements of the CURRENT policy's mean log w on
    that stream, which is exactly what delta(c) = J_B(c) - J_F(c) compares.
    """
    if condition_log_z is None or condition_id is None:
        return None, None

    logw = (log_r + log_pb - log_pf).detach()
    if mode_level_stream is not None:
        condition_log_z.update_mode_level(mode_level_stream, condition_id, logw, step=step)
    if do_update:
        condition_log_z.update(condition_id, logw, step=step)
        if log_Z_learned is not None:
            condition_log_z.update_z_residual(condition_id, logw, log_Z_learned, step=step)
    log_z_target, target_mask = condition_log_z.lookup(condition_id)
    return log_z_target.to(device), target_mask.to(device)


def update_condition_best_energy(condition_log_z, condition_id, log_r, log_T_tensor):
    """
    Feed this step's implied energy back to ConditionLogZTracker's running
    per-condition minimum (see ConditionLogZTracker.update_best_energy).
    log_r is exactly -E/T by construction (MolecularCrystal.energy()
    divides by temperature before base_set.py's log_reward() negates it,
    and prebuilt_sample_to_reward follows the identical convention), so
    recovering E = -log_r * T here is exact, not an approximation, and
    free -- no extra energy computation or crystal_batch materialization
    beyond what the loss already computed. No-op when no tracker/
    condition_id is provided.
    """
    if condition_log_z is None or condition_id is None:
        return

    temperature = 10 ** log_T_tensor.detach()
    energy = -log_r.detach() * temperature
    condition_log_z.update_best_energy(condition_id, energy)


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
                         repeats=1,
                         report_losses: bool = False,
                         condition_log_z=None,
                         condition_id=None,
                         tb_z_source: str = 'learned',
                         step: int = 0,
                         ):
    """
    freeze_policy/freeze_z (read from loss_coeffs, the single source of
    truth -- scheduled per phase like every other coefficient): forcibly
    detach log_pf/log_pb (all downstream loss terms) or log_Z_learned/
    log_flow, applied uniformly right after they're computed. They replace
    the old tb_z (0/1/2) flag entirely: freeze_z == old tb_z 0 (train the
    policy, Z detached), freeze_policy == old tb_z 2 (train Z only), neither
    == old tb_z 1 (train both). Because the detach happens once, at the
    source, the freeze holds regardless of which other loss terms (vg_lb,
    db, subtb, emp_z, mle...) happen to be active, rather than relying on
    every term individually respecting it. freeze_policy is additionally
    threaded into get_traj_fwd so the conditioner->flow path is detached
    too (gfn._update_log_flow), leaving only flow_model's own parameters to
    receive gradient.
    """
    freeze_policy = getattr(loss_coeffs, 'freeze_policy', 0) > 0.5
    freeze_z = getattr(loss_coeffs, 'freeze_z', 0) > 0.5

    condition = condition.to(gfn.device)
    log_T_tensor = log_T_tensor.to(gfn.device)
    (states, log_pfs, log_pbs, log_flow) = gfn.get_traj_fwd(initial_state,
                                                            discretizer,
                                                            exploration_std,
                                                            condition,
                                                            mol_batch,
                                                            detach_traj=loss_coeffs.traj_grads == 0,
                                                            return_gauss_params=False,
                                                            freeze_policy=freeze_policy,
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

    if freeze_policy:
        log_pf = log_pf.detach()
        log_pb = log_pb.detach()
        log_pfs = log_pfs.detach()
        log_pbs = log_pbs.detach()
    if freeze_z:
        log_Z_learned = log_Z_learned.detach()
        log_flow = log_flow.detach()

    beta = loss_coeffs.beta

    # mode_level_stream='fwd': forward-loss trajectories are always fresh
    # on-policy rollouts (no replay path through here), so this is the J_F
    # side of the z_match delta gate
    log_z_target, log_z_target_mask = update_and_lookup_condition_log_z(
        condition_log_z, condition_id, log_r, log_pb, log_pf, gfn.device, step=step,
        log_Z_learned=log_Z_learned, mode_level_stream='fwd')
    update_condition_best_energy(condition_log_z, condition_id, log_r, log_T_tensor)

    losses = []
    """VarGrad losses + empirical-Z regression"""
    vg_by_condition = getattr(loss_coeffs, 'vg_by_condition', 0) > 0.5
    if vg_by_condition and condition_id is not None and (
            loss_coeffs.vg_lb > 0 or loss_coeffs.vg_lme > 0 or loss_coeffs.emp_z > 0):
        # condition-grouped estimation (see condition_grouped_empirical_z): every
        # batch row sharing a condition_id pools into one large-sample per-batch
        # estimate, instead of one estimate per K-repeats tile -- so with a small
        # condition library, fwd repeats stays 1 (fwd tiling, unlike bwd, pays the
        # full energy cost per extra rollout). emp_z here does NOT require a VG
        # loss to be active, unlike the repeats-grouped branch below.
        assert not (loss_coeffs.vg_lb > 0 and loss_coeffs.vg_lme > 0), \
            "Cannot use both vg_lb and vg_lme simultaneously"

        """VarGrad - lower bound loss (condition-grouped): center = group MEAN of log w"""
        if loss_coeffs.vg_lb > 0:
            log_Z_emp_rows, emp_mask, vg_loss = condition_grouped_empirical_z(
                log_pb, log_pf, log_r, condition_id, lme=False, beta=beta)
            losses.append(vg_loss * loss_coeffs.vg_lb)

            """VarGrad - log mean exp loss (condition-grouped): center = group LOGMEANEXP of log w"""
        elif loss_coeffs.vg_lme > 0:
            log_Z_emp_rows, emp_mask, vg_loss = condition_grouped_empirical_z(
                log_pb, log_pf, log_r, condition_id, lme=True, beta=beta)
            losses.append(vg_loss * loss_coeffs.vg_lme)

        else:  # emp_z standalone: Jensen-mean target (the vargrad-Z experiment -- drive Z exactly at the empirical lower bound)
            log_Z_emp_rows, emp_mask, _ = condition_grouped_empirical_z(
                log_pb, log_pf, log_r, condition_id, lme=False, beta=beta)

        if loss_coeffs.emp_z > 0:
            # regress Z(c) onto this batch's per-condition estimate; the target's
            # flavor follows the active VG branch (mean under vg_lb, lme under
            # vg_lme), same inheritance as the repeats-grouped path below
            emp_z_loss = beta * F.smooth_l1_loss(log_Z_emp_rows, log_Z_learned,
                                                 reduction='none', beta=beta) * emp_mask.float()
            losses.append(emp_z_loss * loss_coeffs.emp_z)
        else:
            emp_z_loss = None
    else:
        """VarGrad - lower bound loss (repeats-grouped)"""
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
        use_persistent_z = tb_z_source == 'persistent'
        tb_loss = get_tb_loss(log_Z_learned, log_pb, log_pf, log_r,
                              beta=beta,
                              log_Z_target=log_z_target if use_persistent_z else None,
                              target_mask=log_z_target_mask if use_persistent_z else None)
        losses.append(tb_loss * loss_coeffs.tb)

    """sidecar regression of the learned flow onto the per_z_errsistent per-condition target"""
    emp_z_persistent_coeff = getattr(loss_coeffs, 'emp_z_persistent', 0)
    if emp_z_persistent_coeff > 0 and log_z_target_mask is not None:
        emp_z_persistent_loss = beta * F.smooth_l1_loss(
            log_z_target, log_Z_learned, reduction='none', beta=beta) * log_z_target_mask.float()
        losses.append(emp_z_persistent_loss * emp_z_persistent_coeff)
    else:
        emp_z_persistent_loss = None

    if loss_coeffs.loss_clip != -1:
        combined_losses = soft_clip(torch.stack(losses), loss_coeffs.loss_clip).mean(dim=0)
    else:
        combined_losses = torch.stack(losses).mean(dim=0)

    assert combined_losses.isfinite().all()
    loss = combined_losses.mean()

    if report_losses:
        loss_dict = {'log_pf': log_pf.detach(),
                     'log_pb': log_pb.detach(),
                     'log_Z': log_Z_learned.detach(),
                     'log_r': log_r.detach(),
                     'flow_states': states.detach()}
        if condition_id is not None:
            loss_dict['condition_id'] = condition_id.detach()
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
        if emp_z_persistent_loss is not None:
            loss_dict['emp_z_persistent'] = emp_z_persistent_loss.mean().detach()
        if log_z_target_mask is not None:
            loss_dict['condition_log_z_visited_frac'] = log_z_target_mask.float().mean().detach()

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
                          trajectories: Optional[torch.Tensor] = None,
                          condition_log_z=None,
                          condition_id=None,
                          tb_z_source: str = 'learned',
                          update_log_z: bool = True,
                          step: int = 0,
                          scramble_condition_tiles: int = 0,
                          mode_level_stream: Optional[str] = None,
                          ):
    """
    freeze_policy/freeze_z (read from loss_coeffs): see get_gfn_forward_loss's
    docstring -- same contract, applied here right after log_pf/log_pb/
    log_Z_learned are computed, and freeze_policy threaded into the traj
    sampler so the conditioner->flow path is detached too.

    scramble_condition_tiles: passed straight through to the traj sampler --
    unconditional-prior phase-1 MLE detaches and tile-permutes the condition
    embedding INSIDE the model, at the conditioner->trunk seam; `condition`
    itself stays correctly paired everywhere in this function (see
    GFN._maybe_scramble_condition_embedding). 0 = off.
    """
    freeze_policy = getattr(loss_coeffs, 'freeze_policy', 0) > 0.5
    freeze_z = getattr(loss_coeffs, 'freeze_z', 0) > 0.5

    if trajectories is not None:
        # replay a fixed trajectory (e.g. from a buffer) instead of resampling one
        states, log_pfs, log_pbs, log_flow = gfn.get_traj_replay(
            trajectories, discretizer, condition, mol_batch,
            return_gauss_params=False, freeze_policy=freeze_policy,
            scramble_condition_tiles=scramble_condition_tiles)
    else:
        states, log_pfs, log_pbs, log_flow = gfn.get_traj_bwd(
            samples, discretizer, condition, mol_batch,
            return_gauss_params=False,
            detach_traj=loss_coeffs.traj_grads == 0, freeze_policy=freeze_policy,
            scramble_condition_tiles=scramble_condition_tiles)
    log_Z_learned = log_flow[:, 0]
    log_flow[:, -1] = log_r

    log_pf = log_pfs.sum(-1)
    log_pb = log_pbs.sum(-1)

    if freeze_policy:
        log_pf = log_pf.detach()
        log_pb = log_pb.detach()
        log_pfs = log_pfs.detach()
        log_pbs = log_pbs.detach()
    if freeze_z:
        log_Z_learned = log_Z_learned.detach()
        log_flow = log_flow.detach()

    beta = loss_coeffs.beta

    # log_Z_learned deliberately omitted (unlike the forward call site): the
    # z-residual monitor is on-policy only -- bwd/replay are off-policy, and
    # it's the policy model's job to fix those (mode retention/coverage), not
    # the Z model's, which is trained and judged on forward rollouts only.
    log_z_target, log_z_target_mask = update_and_lookup_condition_log_z(
        condition_log_z, condition_id, log_r, log_pb, log_pf, gfn.device, step=step,
        do_update=update_log_z, mode_level_stream=mode_level_stream)

    losses = []
    """VarGrad losses"""
    vg_by_condition = getattr(loss_coeffs, 'vg_by_condition', 0) > 0.5
    if vg_by_condition and condition_id is not None and (
            loss_coeffs.vg_lb > 0 or loss_coeffs.vg_lme > 0):
        # condition-grouped BACKWARD VarGrad: cross-terminal, pooling every row
        # sharing a condition (buffer draws AND their K repeats). Safe off-policy
        # because zero variance is a measure-independent fixed point -- log w
        # constant on the buffer's support forces P_F prop. to R*P_B there, the
        # same optimum as TB/forward VarGrad -- and doubles as mode retention: a
        # terminal P_F has dropped shows up as an extreme positive log w outlier,
        # i.e. the variance's loudest gradient. Only the group CENTER (the
        # off-policy mean of log w) is biased as a log Z estimate, which is why
        # it must never feed a Z regression (assert below); as the variance's
        # centering constant it's harmless. NB the repeats-grouped legacy branch
        # below groups per K-tile, which for same-terminal tiles is TBC in
        # disguise (reward cancels) -- cross-terminal pressure requires this
        # condition grouping.
        assert not (loss_coeffs.vg_lb > 0 and loss_coeffs.vg_lme > 0), \
            "Cannot use both vg_lb and vg_lme simultaneously"
        assert getattr(loss_coeffs, 'emp_z', 0) <= 0, \
            "bwd emp_z with vg_by_condition is unsupported: the off-policy group center is a biased log Z target"
        if loss_coeffs.vg_lb > 0:
            _, _, vg_loss = condition_grouped_empirical_z(
                log_pb, log_pf, log_r, condition_id, lme=False, beta=beta)
            losses.append(vg_loss * loss_coeffs.vg_lb)
        else:
            _, _, vg_loss = condition_grouped_empirical_z(
                log_pb, log_pf, log_r, condition_id, lme=True, beta=beta)
            losses.append(vg_loss * loss_coeffs.vg_lme)

        """VarGrad - lower bound loss (legacy repeats-grouped)"""
    elif loss_coeffs.vg_lb > 0:
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
    if loss_coeffs.tb > 0:
        use_persistent_z = tb_z_source == 'persistent'
        tb_loss = get_tb_loss(log_Z_learned, log_pb, log_pf, log_r,
                              beta=beta,
                              log_Z_target=log_z_target if use_persistent_z else None,
                              target_mask=log_z_target_mask if use_persistent_z else None)
        losses.append(tb_loss * loss_coeffs.tb)

    """sidecar regression of the learned flow onto the persistent per-condition target"""
    emp_z_persistent_coeff = getattr(loss_coeffs, 'emp_z_persistent', 0)
    if emp_z_persistent_coeff > 0 and log_z_target_mask is not None:
        emp_z_persistent_loss = beta * F.smooth_l1_loss(
            log_z_target, log_Z_learned, reduction='none', beta=beta) * log_z_target_mask.float()
        losses.append(emp_z_persistent_loss * emp_z_persistent_coeff)
    else:
        emp_z_persistent_loss = None

    if loss_coeffs.mle > 0:
        mle_loss = terminal_mle(
            log_pf, log_pb,
            repeats,
            repeats > 1,
            estimator='exact' if repeats > 1 else 'bound',
            dreg=True
        )
        losses.append(mle_loss * loss_coeffs.mle)

    tbc_coeff = getattr(loss_coeffs, 'tbc', 0)
    if tbc_coeff > 0:
        tbc_loss = get_tbc_loss(log_pf, log_pb, repeats, beta=beta)
        losses.append(tbc_loss * tbc_coeff)
    else:
        tbc_loss = None

    if loss_coeffs.loss_clip != -1:
        combined_losses = soft_clip(torch.stack(losses), loss_coeffs.loss_clip).mean(dim=0)
    else:
        combined_losses = torch.stack(losses).mean(dim=0)
    loss = combined_losses.mean()

    if report_losses:
        loss_dict = {'losses': combined_losses.detach(),
                     'log_pf': log_pf.detach(), 'log_pb': log_pb.detach(),
                     'log_Z': log_Z_learned.detach(), 'log_r': log_r.detach(),
                     'flow_states': states.detach(),
                     'resid': ((log_pf - log_pb) - (log_r - log_Z_learned)).detach()}
        if condition_id is not None:
            loss_dict['condition_id'] = condition_id.detach()
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
        if tbc_loss is not None:
            loss_dict['tbc'] = tbc_loss.mean().detach()
        if loss_coeffs.db > 0:
            loss_dict['db'] = db_loss.mean().detach()
        if loss_coeffs.subtb > 0:
            loss_dict['subtb'] = subtb_loss.mean().detach()
        if loss_coeffs.pf_boost > 0:
            loss_dict['pf_boost'] = pf_boost_loss.mean().detach()
        if emp_z_persistent_loss is not None:
            loss_dict['emp_z_persistent'] = emp_z_persistent_loss.mean().detach()
        if log_z_target_mask is not None:
            loss_dict['condition_log_z_visited_frac'] = log_z_target_mask.float().mean().detach()

    else:
        loss_dict = None

    return loss, loss_dict


def log_pf_estimate(log_pf, log_pb, repeats: int):
    """
    IWAE-style (eq. 27) estimate of log p_f(x), the forward policy's marginal
    probability of reaching terminal x, from K = repeats independent backward
    rollouts x -> ... -> x_0 with teacher-forced forward log-probs:

        log p_hat_f(x) = logsumexp_k(log_pf_k - log_pb_k) - log K

    log_pf/log_pb: [B * repeats] (K-tiled terminal-major layout, i.e. each
    terminal's K rollouts are contiguous -- see CrystalBuffer's `repeats`
    tiling convention). Returns log_w [B, K] (per-rollout log importance
    weight, for confidence/spread diagnostics) and log_p_hat [B].

    Reused by both terminal_mle's "exact" estimator (a training loss) and
    train.py's anchor-buffer confirmation step (a no_grad novelty check) --
    same math, different callers.
    """
    if log_pf.numel() % repeats != 0:
        raise ValueError(f"log_pf size {log_pf.numel()} not divisible by repeats={repeats}")
    B = log_pf.numel() // repeats
    log_w = (log_pf - log_pb).view(B, repeats)
    log_p_hat = torch.logsumexp(log_w, dim=1) - math.log(repeats)
    return log_w, log_p_hat


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
            # Plain IWAE objective (can be higher variance) -- same reduction as log_pf_estimate
            _, log_p_hat = log_pf_estimate(log_pf.reshape(-1), log_pb.reshape(-1), repeats)
            loss = -log_p_hat

        return loss.repeat(repeats)

    else:
        raise ValueError("estimator must be 'bound' or 'exact'")


def get_tbc_loss(log_pf, log_pb, repeats: int, beta: float = 10.0):
    """
    Trajectory balance consistency (TBC) -- the data-driven, reward-free
    VarGrad of "Unifying Generative Models with GFlowNets and Beyond"
    (arXiv:2209.02606, eq. 33): at the TB fixed point, log(Pf(tau)/Pb(tau|x))
    is the SAME value (log R(x) - log Z) for every trajectory tau terminating
    in the same state x, so for K backward rollouts per terminal the reward
    cancels and consistency can be trained with no reward signal at all.
    Penalizes each rollout's deviation of logw = log_pf - log_pb from its
    terminal group's mean -- i.e. the intra-terminal variance of logw, exactly
    the trajectory-marginalization component of Var(log w) that MLE (a bound
    on the K-sample average, not on its spread) leaves unconstrained.

    log_pf/log_pb: [B*K] trajectory-summed log-probs in the terminal-major
    K-tiled layout (each terminal's K rollouts contiguous -- the same
    CrystalBuffer `repeats` convention terminal_mle/log_pf_estimate rely on).

    Mean-centered rather than pairwise (eq. 33): the two differ only by a
    constant factor, and in the quadratic (|resid| < beta) regime the attached
    group mean contributes exactly zero gradient (residuals sum to zero per
    group), so no explicit center-detach is needed.

    Returns per-rollout losses [B*K], in input order.
    """
    assert repeats > 1, "tbc needs repeats > 1: the consistency residual is defined over K same-terminal backward rollouts"
    if log_pf.numel() % repeats != 0:
        raise ValueError(f"log_pf size {log_pf.numel()} not divisible by repeats={repeats}")
    logw = (log_pf - log_pb).view(-1, repeats)
    center = logw.mean(dim=1, keepdim=True)
    loss = beta * F.smooth_l1_loss(logw, center.expand_as(logw), reduction='none', beta=beta)
    return loss.view(-1)


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


def get_tb_loss(log_Z_learned, log_pb, log_pf, log_r,
                beta: float = 10,
                log_Z_target: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None
                ):
    """
    Policy/Z freezing is handled upstream by get_gfn_forward_loss /
    get_gfn_backward_loss before this is called: freeze_policy detaches
    log_pf/log_pb and freeze_z detaches log_Z_learned (and the whole
    conditioner->flow path, see gfn._update_log_flow). The TB residual just
    consumes whatever gradient state the inputs already carry, so there are
    no per-term detach_z/z_only flags here anymore.

    log_Z_target/target_mask (from ConditionLogZTracker.lookup): where
    target_mask is True, the persistent per-condition empirical log Z
    estimate is substituted for log_Z_learned in the TB residual --
    log_Z_target is always detached (a running statistic, never a network
    output), so the flow model gets no TB gradient for those trajectories
    (see emp_z_persistent for its sidecar regression target instead). Where
    target_mask is False (or is None), behavior is unchanged from before.
    """
    log_Z_per_traj = log_Z_learned
    if target_mask is not None:
        log_Z_per_traj = torch.where(target_mask, log_Z_target.detach(), log_Z_per_traj)

    tb = (log_pf + log_Z_per_traj - log_pb - log_r)

    # tb_loss = F.mse_loss(tb, torch.zeros_like(tb), reduction='none')
    tb_loss = beta * F.smooth_l1_loss(tb, torch.zeros_like(tb), reduction='none', beta=beta)
    return tb_loss


def emp_Z(gfn, log_Z, log_Z_learned, repeats, beta: float = 10):
    if gfn.conditional:
        # log_Z is one empirical estimate per condition group [B]; broadcast each
        # group's value back out over its `repeats` trajectories (x-repeated-K-times
        # layout, matching how mol_batch/condition were tiled) to compare row-wise.
        emp_z_loss = beta * F.smooth_l1_loss(log_Z.repeat_interleave(repeats), log_Z_learned, reduction='none',
                                             beta=beta)
    else:
        emp_z_loss = beta * F.smooth_l1_loss(log_Z.repeat(len(log_Z_learned)), log_Z_learned, reduction='none',
                                             beta=beta)
    return emp_z_loss


def condition_grouped_empirical_z(log_pb, log_pf, log_r, condition_id,
                                  lme: bool = False, beta: float = 10.0,
                                  min_group_count: int = 2):
    """
    Condition-grouped VarGrad / empirical-Z estimation for the forward loss.

    The repeats-grouped vg_lb/vg_lme assume each K-tiled block of rows is one
    estimation group -- the right shape when every condition appears once per
    batch, but wasteful for small condition libraries, where many independent
    draws share a condition. Scatter-grouping by condition_id pools BOTH the
    K repeats of a draw and every other same-condition draw in the batch into
    one estimate, so the two mechanisms compose: a 2-condition library gets
    ~B/2-sample estimates with fwd repeats = 1 (fwd tiling, unlike bwd, pays
    the full energy cost per extra rollout), while a large library where each
    condition lands ~once per batch still needs repeats > 1 to give every
    group >= K samples. No row-ordering/contiguity is assumed.

    Returns (log_Z_emp_rows, mask_rows, vg_loss), all [N] and row-aligned:
    - log_Z_emp_rows: each row's condition-group empirical log Z -- Jensen
      flavor (group mean of log_ratio, a lower bound) when lme=False,
      logmeanexp flavor when lme=True. Carries whatever grad state
      log_pf/log_pb/log_r already have (the upstream freeze flags govern
      detaching, same contract as the repeats-grouped path).
    - mask_rows (bool): rows whose group had >= min_group_count samples. A
      singleton group's "estimate" is the row's own ratio -- regressing Z
      onto it degenerates to per-trajectory TB, exactly what emp_z exists to
      avoid -- so those rows carry no emp_z gradient.
    - vg_loss: per-row huber deviation of log_ratio from its group estimate
      (the classic VarGrad policy loss; identically zero on singleton groups).
    """
    log_ratio = log_r + log_pb - log_pf
    uniq, inverse = torch.unique(condition_id.to(log_ratio.device), return_inverse=True)
    k = uniq.numel()

    counts = torch.zeros(k, device=log_ratio.device, dtype=log_ratio.dtype).scatter_add_(
        0, inverse, torch.ones_like(log_ratio))

    if lme:
        # detached max-shift: the standard logsumexp trick -- the shift cancels
        # exactly in the value and the gradient still flows through exp/sum
        group_max = torch.full((k,), float('-inf'), device=log_ratio.device,
                               dtype=log_ratio.dtype).scatter_reduce_(
            0, inverse, log_ratio.detach(), reduce='amax', include_self=True)
        shifted = (log_ratio - group_max[inverse]).exp()
        sum_exp = torch.zeros(k, device=log_ratio.device, dtype=log_ratio.dtype).scatter_add_(
            0, inverse, shifted)
        group_z = group_max + sum_exp.log() - counts.log()
    else:
        group_z = torch.zeros(k, device=log_ratio.device, dtype=log_ratio.dtype).scatter_add_(
            0, inverse, log_ratio) / counts.clamp(min=1)

    log_Z_emp_rows = group_z[inverse]
    mask_rows = counts[inverse] >= min_group_count
    vg_loss = beta * F.smooth_l1_loss(log_Z_emp_rows, log_ratio, reduction='none', beta=beta)
    return log_Z_emp_rows, mask_rows, vg_loss


def vg_lme(gfn, log_pb, log_pf, log_r, repeats, beta: float = 10):
    log_ratio = log_r + log_pb - log_pf
    if gfn.conditional:
        # [B*repeats] -> [B, repeats]; each row is one condition's K trajectories
        log_ratio_grouped = log_ratio.view(-1, repeats)
        log_Z = torch.logsumexp(log_ratio_grouped, dim=1, keepdim=True) - math.log(repeats)  # [B, 1]
        vg_loss = beta * F.smooth_l1_loss(log_Z.expand_as(log_ratio_grouped), log_ratio_grouped, reduction='none',
                                          beta=beta).view(-1)
        log_Z = log_Z.view(-1)  # [B]
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
        # [B*repeats] -> [B, repeats]; each row is one condition's K trajectories
        log_ratio_grouped = log_ratio.view(-1, repeats)
        log_Z = log_ratio_grouped.mean(dim=1, keepdim=True)  # [B, 1]
        vg_loss = beta * F.smooth_l1_loss(log_Z.expand_as(log_ratio_grouped), log_ratio_grouped, reduction='none',
                                          beta=beta).view(-1)
        log_Z = log_Z.view(-1)  # [B]

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
