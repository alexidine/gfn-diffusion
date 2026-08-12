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
    behind rms_z_grad()/worst_tb_err(). Only get_gfn_forward_loss passes it -- the z-residual
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
                                                            path_grad_last_k=getattr(
                                                                loss_coeffs, 'path_grad_last_k', 0),
                                                            )
    log_Z_learned = log_flow[:, 0]

    crystal_batch, log_r = get_loss_reward(log_T_tensor,
                                           log_reward_fn,
                                           mol_batch,
                                           return_exp,
                                           states,
                                           no_grad=loss_coeffs.reward_grads == 0
                                           )
    # OPTIONAL per-sample clip on the REWARD's own gradient path, separate from
    # the global grad clip. Defaults to 0 = off, so this is inert unless asked
    # for. Rationale: with reward_grads on, d log R / d x_T for an LJ-type
    # energy is near-singular whenever atoms clash, so a handful of overlapping
    # samples can dominate the whole batch gradient. That is a DIFFERENT
    # destabilizer from the BPTT-Jacobian one path_grad_last_k addresses, and
    # the global clip cannot separate them -- by the time gradients are summed
    # at the parameters the reward path is indistinguishable from the density
    # path. Clipping at the source keeps a clashed sample's contribution
    # bounded without silencing the rest of the batch.
    _rg_clip = float(getattr(loss_coeffs, 'reward_grad_clip', 0) or 0)
    if _rg_clip > 0 and log_r.requires_grad:
        log_r.register_hook(lambda g: g.clamp(-_rg_clip, _rg_clip))
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

    # Z-only condition-grouped level regression (see z_level_loss). Rides the
    # fwd rollout that ALREADY EXISTS -- no extra sampling, no extra reward
    # call -- and detaches log w, so it adds Z gradient WITHOUT adding
    # on-policy policy gradient. That separation is the point: it isolates
    # "Z tracks each condition" from "the policy samples itself", which are
    # otherwise confounded in any change to fwd's share. Scalar, so expand to
    # per-row for the shared stack/mean below (the mean recovers it exactly).
    z_level_coeff = getattr(loss_coeffs, 'z_level', 0)
    if z_level_coeff > 0 and condition_id is not None:
        z_lvl = z_level_loss(log_pf, log_pb, log_r, log_Z_learned, condition_id)
        losses.append(z_lvl.expand(log_pf.shape[0]) * z_level_coeff)

    if loss_coeffs.loss_clip != -1:
        combined_losses = soft_clip(torch.stack(losses), loss_coeffs.loss_clip).mean(dim=0)
    else:
        combined_losses = torch.stack(losses).mean(dim=0)

    # No finiteness assert here (there never was one on the backward path, and
    # the asymmetry meant forward NaNs crashed the process while backward NaNs
    # were contained). Non-finite losses are handled where they can be handled
    # gracefully: step_loss clips and checks the GRADIENT norm, skips the
    # optimizer step on a non-finite reading, and feeds a consecutive-streak
    # counter to _frozen_training_state -- so weights are never stepped on a
    # NaN, and the LR controller's reset tier sees the loss on its own clock.
    loss = combined_losses.mean()

    if report_losses:
        loss_dict = {'log_pf': log_pf.detach(),
                     'log_pb': log_pb.detach(),
                     'log_Z': log_Z_learned.detach(),
                     'log_r': log_r.detach(),
                     'flow_states': states.detach()}
        if condition_id is not None:
            loss_dict['condition_id'] = condition_id.detach()
            loss_dict.update(condition_group_stats(condition_id))
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
                          sample_weights: Optional[torch.Tensor] = None,
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

    """Level-gap term: the z_match delta as a training signal, not only a gate"""
    # Pulls J_B(c) down onto J_F(c) using the tracker's EMA gap as a DETACHED
    # per-row coefficient: sign and magnitude come from the well-averaged level
    # stream, only the gradient direction comes from this batch, so a noisy
    # per-step gap estimate is never squared. Descending raises E_q[log P_F] on
    # the buffer's support. Deliberately ONE-SIDED -- the matching forward half
    # (-coeff * gap * log w, which would raise J_F) is not added to the forward
    # loss, so the on-policy level is a target, never a follower. No log_Z term
    # appears, so this cannot move Z.
    #
    # Self-limiting: gap is re-read each step, so the coefficient -- and the
    # term -- vanish once the levels match. Structurally this is a PROPORTIONAL
    # CONTROLLER on J_B with gain level_gap and lag = the level stream's
    # half_life_visits, so the failure mode is limit-cycling at high gain, not
    # divergence. level_gap_clamp bounds the per-row force to |gap| nats.
    # The term's VALUE is g*log w, not a distance -- either sign, dominated by
    # |log w|, and not a convergence signal. level_gap_coeff_rms (= rms |gap|,
    # reported below) is the one that goes to zero on success.
    level_gap_coeff = getattr(loss_coeffs, 'level_gap', 0)
    if level_gap_coeff > 0 and condition_log_z is not None and condition_id is not None:
        level_gap_clamp = getattr(loss_coeffs, 'level_gap_clamp', 10.0)
        gap, gap_mask = condition_log_z.lookup_delta(condition_id)
        gap = gap.to(gfn.device).clamp(-level_gap_clamp, level_gap_clamp) \
              * gap_mask.to(gfn.device).float()
        level_gap_loss = gap * (log_r + log_pb - log_pf)
        losses.append(level_gap_loss * level_gap_coeff)
    else:
        level_gap_loss = None

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

    if sample_weights is None:
        loss = combined_losses.mean()
    else:
        # Self-normalised importance weighting for a prioritised draw
        # (docs/to_do_rebuild.md B5). Applied at the FINAL reduction so it
        # covers every active term at once, and self-normalised so the overall
        # loss scale -- and therefore the LR the run is tuned at -- is unchanged
        # by turning prioritisation on.
        #
        # NB per B5b this belongs with a QUADRATIC branch loss. Combining it
        # with an active Huber knee makes per-row push ~ beta/delta, so the
        # deepest rows push LEAST; set beta inactive wherever this is used.
        w = sample_weights.to(combined_losses.device, combined_losses.dtype).flatten()
        if w.numel() != combined_losses.numel():
            # K repeats are tiled over the batch: broadcast one weight per row
            # across its tile rather than silently mis-pairing.
            reps = combined_losses.numel() // max(w.numel(), 1)
            assert reps * w.numel() == combined_losses.numel(), (
                f"sample_weights length {w.numel()} does not tile into "
                f"{combined_losses.numel()} losses")
            w = w.repeat_interleave(reps) if reps > 1 else w
        w = torch.clamp(w, min=0.0)
        denom = w.sum()
        loss = (combined_losses.flatten() * w).sum() / torch.clamp(denom, min=1e-12)

    if report_losses:
        loss_dict = {'losses': combined_losses.detach(),
                     'log_pf': log_pf.detach(), 'log_pb': log_pb.detach(),
                     'log_Z': log_Z_learned.detach(), 'log_r': log_r.detach(),
                     'flow_states': states.detach(),
                     'resid': ((log_pf - log_pb) - (log_r - log_Z_learned)).detach()}
        if condition_id is not None:
            loss_dict['condition_id'] = condition_id.detach()
            loss_dict.update(condition_group_stats(condition_id))
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
        if level_gap_loss is not None:
            loss_dict['level_gap'] = level_gap_loss.mean().detach()
            # the clamped+masked coefficient actually applied, so a level_gap
            # reading ~0 can be told apart from an untrusted-mask no-op
            loss_dict['level_gap_coeff_rms'] = gap.pow(2).mean().sqrt().detach()
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


def z_level_loss(log_pf, log_pb, log_r, log_Z_learned, condition_id):
    """
    Condition-grouped regression of log_Z(c) onto the LIVE per-condition mean
    importance weight:

        L = mean_c ( log_Z(c) - mean_{i in c} log w_i )^2

    Z-ONLY: log w is detached, so this never touches the policy. Rides an
    existing forward rollout, so it costs no extra sampling and no extra
    reward call -- and Z must be regressed on capital-F FRESH on-policy log w,
    which is why it piggybacks rather than sampling its own.

    MAGNITUDE, NOT DISPERSION (corrected 2026-07-27). An earlier version
    mean-centered this into Var_c(e_c), on the reasoning that a uniform offset
    is "just a global Z shift TB already owns". That was wrong twice over.
    First, a uniform offset is NOT harmless: if every condition sits at
    e_c = +4 then every sample's TB residual is +4, so log_pf is pushed down
    EVERYWHERE, and since P_F is normalized the only way to comply is to move
    mass off-support -- the policy inflates at every condition at once, which
    is the same uphill mechanism as the dispersion case, just global. Second,
    TB is empirically NOT driving that mean to zero: the per-condition z_bias
    histogram sits centered near -4, not 0. Centering therefore discarded the
    single largest component of the level error. The target is every condition
    on its OWN right level, i.e. z_bias_rms / z_bias_worst -> 0, so the loss
    is the plain squared magnitude.

    WHAT IT STILL ADDS OVER TB's Z GRADIENT. It overlaps TB rather than being
    orthogonal to it (both pull log_Z toward log w), so part of its effect is
    simply a Z learning-rate multiplier. The parts that are NOT redundant:
    (a) it is QUADRATIC where TB's Huber has gone LINEAR -- past |resid| >
    beta the Huber caps magnitude but not sign, and Adam normalizes a
    persistent sign-consistent gradient to a full step, so a condition 30 nats
    off pushes exactly as hard as one 10 nats off, forever, without tripping a
    grad-norm wire. Here it pulls ~3x harder, in proportion to how wrong it
    actually is. (b) it groups by condition BEFORE squaring, so within-
    condition sampling noise averages out instead of being fitted.

    Bias note: with ~1 sample per condition per batch the group means are
    noisy, so the LOSS VALUE is biased upward by within-condition variance.
    The GRADIENT w.r.t. log_Z(c) is unbiased, which is what is used. No
    internal clip -- the tail's pull is the point -- so gradient_norm_clip is
    the backstop.

    Returns a scalar; callers expand it to per-row for the usual stack/mean.
    """
    logw = (log_r + log_pb - log_pf).detach()
    err = log_Z_learned - logw
    uniq, inverse = torch.unique(condition_id.to(err.device), return_inverse=True)
    k = uniq.numel()
    counts = torch.zeros(k, device=err.device, dtype=err.dtype).scatter_add_(
        0, inverse, torch.ones_like(err))
    group_mean = torch.zeros(k, device=err.device, dtype=err.dtype).scatter_add_(
        0, inverse, err) / counts.clamp(min=1)
    return (group_mean ** 2).mean()


def condition_group_stats(condition_id, min_group_count: int = 2):
    """
    Draw-composition readout for condition-grouped VarGrad: how a batch's rows
    actually distribute over conditions. Measures the DRAW rather than a
    downstream mask, so it reports identically whichever estimator branch ran,
    and both knobs that move group occupancy land in it -- `repeats` tiling and
    prior_buffer.condition_block_m.

    Row-weighted deliberately. 'The average ROW sits in a group of size g' is
    what the estimator's variance depends on; an unweighted mean over groups is
    dominated by the singletons, which contribute no gradient at all (vg_loss is
    identically zero there, see condition_grouped_empirical_z).

    vg_live_frac is the load-bearing one: it is the fraction of the batch that
    carries any VarGrad gradient. A configuration that looks well-tuned but runs
    at live_frac 0.3 is training on a third of what it paid to roll out.
    """
    uniq, inverse, counts = torch.unique(condition_id, return_inverse=True,
                                         return_counts=True)
    per_row = counts[inverse].float()
    return {'vg_n_groups': per_row.new_tensor(float(uniq.numel())),
            'vg_group_size_mean': per_row.mean().detach(),
            'vg_live_frac': (per_row >= min_group_count).float().mean().detach()}


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
        # the group here is the WHOLE batch, so log-mean-exp normalizes by the
        # batch size -- NOT by `repeats`, which is the conditional branch's
        # group size and was copied down here. Subtracting log(repeats) left the
        # centre high by log(N/repeats) = log(B) nats (~5.7 at a 300-row batch
        # with repeats 1); since the shift is constant it survives into the
        # gradient, pushing every log_ratio up, i.e. log_pf DOWN everywhere --
        # and a normalized P_F can only comply by moving mass off-support (the
        # uphill mechanism in z_level_loss's docstring). vg_lb's unconditional
        # branch takes a plain mean over the same group and was always correct.
        log_Z = torch.logsumexp(log_ratio, dim=0, keepdim=True) - math.log(log_ratio.shape[0])
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
