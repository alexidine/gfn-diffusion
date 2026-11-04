"""The space-group step under option A (docs/design/conditional_ar_architecture.md §5.7).

The level head learns log Z(m, g) with g given, from e_g (nf sym_ops pooled or tokens) or the space-group condition
column (sg_conditioning). The space-group distribution of a molecule is then its levels over the allowed groups,

    p(g | m) = softmax_{g in G_m} L(m, g),        log q(x, g | m) = log p(g | m) + log q(x | g, m),

read detached: a TB residual of a drawn (g, x) is L(m, g) + log q(x | g, m) - log R(x; g, m), the residual of the
forced draw of that g, so drawing g changes which (m, g) pairs train, never the loss. While the flow is unconverged
the level sits near log Z(m, g) - KL_g, so reported space-group weights come from importance sampling within each
group (group_weights_is), not from the head.

- ``level_group_logits(policy, condition_of, mol_batch, groups)``: [n, G] L(m, g) of each row under each candidate g.
- ``draw_groups(logits, groups, uniform_frac, allowed)``: g per row from (1 - eps) p + eps Uniform(G_m), with log p(g | m).
- ``group_weights_is(log_w, group_index, n_groups)``: per-group importance-sampling log Z_g and their normalised
  weights, with each group's effective sample size.

The allowed set G_m is every candidate unless ``allowed`` [n, G] says otherwise; no per-molecule rule (Sohncke groups
for an enantiopure chiral molecule, enantiomorph closure) is applied here (owner, 2026-11-04: left for now).
"""
import math

import torch


@torch.no_grad()
def level_group_logits(policy, condition_of, mol_batch, groups):
    """[n, G] float32: the policy's level L(m, g) of each of mol_batch's n rows with its space group set to each
    candidate g in `groups`. condition_of(g) -> (mol_batch_g, condition_g): the rows conditioned under g alone (the
    trainer passes energy_function.condition_samples with the rows' own temperatures and Z' and sg_inds = g)."""
    cols = []
    for g in groups:
        mb, cond = condition_of(int(g))
        cols.append(policy.flow_model(policy.get_condition_embedding(cond, mb)).reshape(-1).float())
    return torch.stack(cols, -1)


@torch.no_grad()
def draw_groups(logits, groups, uniform_frac=0.0, allowed=None, generator=None):
    """(g [n] long, log p(g | m) [n] float64 under the head alone, p [n, G] float64 the head's distribution).

    g is drawn from (1 - uniform_frac) p + uniform_frac Uniform(G_m), p = softmax over the allowed groups of logits
    [n, G]; the uniform share keeps every allowed group drawn whatever the head says. allowed [n, G] bool (None =
    every group); a row with no allowed group raises."""
    logits = logits.double()
    n, k = logits.shape
    if len(groups) != k:
        raise ValueError(f"{len(groups)} groups for {k} logit columns")
    if not 0.0 <= float(uniform_frac) <= 1.0:
        raise ValueError(f"uniform_frac {uniform_frac} outside [0, 1]")
    allowed = torch.ones_like(logits, dtype=torch.bool) if allowed is None else allowed.to(logits.device).bool()
    if not bool(allowed.any(-1).all()):
        raise ValueError("a row with no allowed space group")
    logp = torch.log_softmax(logits.masked_fill(~allowed, -math.inf), -1)
    p = logp.exp()
    uni = allowed.double() / allowed.sum(-1, keepdim=True)
    mix = (1.0 - float(uniform_frac)) * p + float(uniform_frac) * uni
    pick = torch.multinomial(mix, 1, generator=generator).squeeze(-1)
    g = torch.as_tensor(list(groups), dtype=torch.long, device=logits.device)[pick]
    return g, logp.gather(-1, pick[:, None]).squeeze(-1), p


@torch.no_grad()
def group_weights_is(log_w, group_index, n_groups):
    """Per-group importance sampling of draws x ~ q(x | g, m) for one molecule: log_w [N] = log R(x; g, m) -
    log q(x | g, m), group_index [N] in [0, n_groups). Returns (log Z_g [G] float64, the normalised weights
    Z_g / sum Z [G], ESS_g [G] = (sum w)^2 / sum w^2 per group). A group with no draw has log Z -inf, weight 0, ESS 0."""
    log_w = torch.as_tensor(log_w, dtype=torch.float64).reshape(-1)
    gi = torch.as_tensor(group_index, dtype=torch.long, device=log_w.device).reshape(-1)
    log_z = torch.full((n_groups,), -math.inf, dtype=torch.float64, device=log_w.device)
    ess = torch.zeros(n_groups, dtype=torch.float64, device=log_w.device)
    for k in range(n_groups):
        lw = log_w[gi == k]
        if lw.numel() == 0:
            continue
        log_z[k] = torch.logsumexp(lw, 0) - math.log(lw.numel())
        ess[k] = torch.exp(2 * torch.logsumexp(lw, 0) - torch.logsumexp(2 * lw, 0))
    finite = torch.isfinite(log_z)
    weights = torch.zeros_like(log_z)
    if bool(finite.any()):
        weights[finite] = torch.softmax(log_z[finite], 0)
    return log_z, weights, ess
