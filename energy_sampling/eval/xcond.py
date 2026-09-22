r"""Cross-conditional sharpening: the metric set, ready to be called from an eval hook.

WHAT IT ANSWERS. Are the per-condition distributions annealing into their own shapes, or
settling into one shared basin wearing C different labels? For a terminal x drawn under c_i,
score it under every c_j:

    Delta[i, j] = log p_hat_f(x_i | c_j) = logsumexp_k(log_pf_k - log_pb_k) - log K

over K backward rollouts tau ~ P_b(.|x, c_j). That is the model's OWN marginal log-density.
No reward, no energy call, no Z(c) -- P_f is normalised by construction, so the IWAE identity
E_{tau~P_b}[P_f/P_b] = p(x|c) needs no partition function.

THE NOISE CORRECTION, which is the reason this file exists and xcond_matrix.py is not enough.
Var(Delta) splits into row (which crystal), column (which condition) and interaction (the
pairing). Only the interaction carries conditional information -- and per-cell ESTIMATION
noise is idiosyncratic, so it has no row or column structure and lands 100% in the
interaction bucket. The headline sensor is therefore biased upward by its own noise, by an
amount that grows as K shrinks: comparing two runs at different K compares their noise as
much as their conditioning. So K is drawn as TWO INDEPENDENT REPLICATES of K/2. Their
difference estimates the noise variance directly, it is subtracted from the interaction
term, and it also yields the standard error on the diagonal lift.

WHY COLUMNS GET THEIR OWN ROLLOUTS. Re-scoring ONE stored trajectory under every c_j is
cheaper and perfectly paired, and it is wrong: that trajectory was drawn under c_i, so the
bound is tight on the diagonal and loose off it, manufacturing the diagonal dominance the
metric exists to detect. Every column draws its own tau ~ P_b(.|x, c_j).

BATCHING. Cost is C^2 * n * K trajectories, but the shape matters more than the count on this
codebase: the step is launch-overhead bound (profiled 2026-09-22), so the (terminal, column,
replicate) grid is flattened and run in a few large batches rather than one call per column.
log_pf_estimate's terminal-major contiguity contract does not survive that flattening, so its
one line of math is reproduced explicitly over an [N, C, K] view.
"""
import torch


@torch.no_grad()
def cross_condition_delta(model, cond_batch, cond_vectors, latents, discretizer,
                          k: int, chunk_rows: int = 1024):
    """
    Delta [N, C] and its two half-K replicates.

    cond_batch / cond_vectors: the COLUMN source, C rows -- one condition each, carrying that
        condition's molecule graph and condition vector.
    latents: [N, D] terminals, condition-major, N = C * per_cond: rows
        [j*per_cond : (j+1)*per_cond] were drawn under condition j.

    The two are indexed independently: a cell pairs terminal i with column j, and column j's
    graph comes from cond_batch regardless of which condition produced terminal i. That
    separation is the whole measurement.
    """
    assert k % 2 == 0 and k >= 2, f'k must be even (two replicates of k/2), got {k}'
    dev = latents.device
    N, C, half = latents.shape[0], cond_vectors.shape[0], k // 2
    assert N % C == 0, f'{N} terminals is not a whole number of blocks of {C} conditions'

    # flat grid of (terminal, column, rollout); replicate id is rollout >= half
    t_idx = torch.arange(N, device=dev).repeat_interleave(C * k)
    c_idx = torch.arange(C, device=dev).repeat_interleave(k).repeat(N)
    total = t_idx.numel()

    log_w = torch.empty(total, device=dev)
    for s in range(0, total, chunk_rows):
        e = min(s + chunk_rows, total)
        cols = c_idx[s:e]
        _, lpf, lpb, _ = model.get_traj_bwd(
            latents[t_idx[s:e]], discretizer, cond_vectors[cols],
            cond_batch.subsample_new_batch(cols))
        log_w[s:e] = lpf.sum(-1) - lpb.sum(-1)

    w = log_w.view(N, C, k)
    lse = lambda x: torch.logsumexp(x, dim=-1) - torch.log(torch.tensor(float(x.shape[-1]), device=dev))
    return lse(w).double(), lse(w[..., :half]).double(), lse(w[..., half:]).double()


def xcond_metrics(delta, rep_a, rep_b, worst_quantile: float = 0.9):
    """The reported set. delta/rep_a/rep_b: [N, C], N = C*per_cond, condition-major rows."""
    N, C = delta.shape
    per_cond = N // C
    own = torch.arange(C, device=delta.device).repeat_interleave(per_cond)   # each row's true column
    rows = torch.arange(N, device=delta.device)
    diag = delta[rows, own]                                    # [N] own-condition density

    # --- noise floor. rep_a/rep_b are independent K/2 estimates of the same cell, so
    # Var(rep) = E[(a-b)^2]/2 and the full-K estimate has about half that variance.
    noise_half = ((rep_a - rep_b) ** 2).mean() / 2.0
    noise_full = noise_half / 2.0

    # --- two-way decomposition of Var(Delta)
    gm = delta.mean()
    r = delta.mean(1, keepdim=True) - gm
    c = delta.mean(0, keepdim=True) - gm
    inter = delta - gm - r - c
    tot = ((delta - gm) ** 2).mean()
    inter_var = (inter ** 2).mean()
    inter_corr = (inter_var - noise_full).clamp(min=0.0)       # noise lands ENTIRELY here
    tot_corr = (tot - noise_full).clamp(min=1e-12)

    # --- diagonal lift, as an interaction contrast (row/column main effects removed)
    # DE-SHRINK BY (1 - 1/C). The planted lift also raises each row mean and each column
    # mean by L/C, so removing the main effects leaves L*(1 - 1/C) on the diagonal, not L.
    # Verified against a synthetic matrix with a known planted lift. Without this the metric
    # is a function of the matrix size and two runs at different `conditions` are not
    # comparable; with it, this equals the raw diagonal-minus-off-diagonal contrast.
    deshrink = 1.0 / (1.0 - 1.0 / C) if C > 1 else 1.0
    lift_rows = inter[rows, own] * deshrink
    lift = lift_rows.mean()
    lift_se = lift_rows.std() / max(N ** 0.5, 1.0)
    # LOWER tail, not upper. cfg:conditional_worst_quantile is the fraction of conditions
    # allowed past the bar, and the *_worst family takes the UPPER tail for tb_err-style
    # keys where large is bad. A lift is the other polarity -- large is GOOD -- so the worst
    # conditions sit at quantile q, not 1-q. Caught by the first in-train run returning a
    # `worst` of 7.52 against a mean of 4.95.
    per_cond_lift = lift_rows.view(C, per_cond).mean(1)        # [C]
    worst = torch.quantile(per_cond_lift, worst_quantile)

    # --- rank statistics: human-readable, NOT sensors (flat across 10k steps while the
    # interaction share moved, because a proportional contraction preserves ordering)
    pred = delta.argmax(1)
    acc = (pred == own).double().mean()
    logpost = (diag - torch.logsumexp(delta, 1)).mean()
    chance = -torch.log(torch.tensor(float(C)))
    sd = delta.std()

    return {
        # primary
        'xcond/interaction_frac': (inter_corr / tot_corr).item(),
        'xcond/diag_lift_nats': lift.item(),
        'xcond/diag_lift_std': (lift / sd.clamp(min=1e-9)).item(),
        # context -- diag_lift_nats cannot be read without these
        'xcond/delta_sd': sd.item(),
        # r is [N,1] and broadcasts across C columns, so its mean-square over the FULL
        # matrix already equals (r**2).mean() -- no factor of C. (The square-matrix
        # spelling that had one agreed numerically only because N == C there.)
        'xcond/row_frac': ((r ** 2).mean() / tot_corr).item(),
        'xcond/col_frac': ((c ** 2).mean() / tot_corr).item(),
        'xcond/diag_lift_se': lift_se.item(),
        'xcond/noise_sd': noise_full.sqrt().item(),
        # per-condition tail, in the *_worst idiom
        'xcond/diag_lift_worst': worst.item(),
        # communication
        'xcond/accuracy': acc.item(),
        'xcond/accuracy_chance': 1.0 / C,
        'xcond/logpost_true': logpost.item(),
        'xcond/logpost_chance': chance.item(),
        'xcond/n_conditions': float(C),
        'xcond/n_per_condition': float(per_cond),
    }
