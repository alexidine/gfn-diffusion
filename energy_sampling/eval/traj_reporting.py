"""
traj_overlap.py
===============

Diagnostics for overlap / drift between FORWARD and BACKWARD trajectory
distributions, designed for diffusion-sampler / continuous-GFlowNet style
dicts with keys like:

    means_f, logvars_f, means_b, logvars_b,
    flow_states, log_r, log_pfs, log_pbs, log_flow, log_T_tensor

It answers two distinct questions that are easy to conflate:

  (A) LIKELIHOOD overlap  -- do F and B agree as distributions over
      trajectories?  (ESS of the likelihood ratio, KL both directions,
      Jeffreys divergence, Crooks histograms.)

  (B) GEOMETRIC overlap   -- do the visited STATE CLOUDS live on the same
      manifold, and where along t do they separate?  (per-step policy KL,
      MMD-vs-t, nearest-neighbour two-sample test.)

Two policies can pass (A) and fail (B) or vice-versa, so it reports both.

Usage
-----
    import pickle
    from traj_overlap import report

    data_F = pickle.load(open("eval_forward.pkl", "rb"))   # traj sampled from F
    data_B = pickle.load(open("eval_backward.pkl", "rb"))  # traj sampled from B (optional)

    report(data_F, data_B)          # full bidirectional report
    report(data_F)                  # single-direction (subset of metrics)

If your tensors aren't (N, T, D) / (N, T), set the axis hints in report().
"""

from __future__ import annotations
import numpy as np
import torch

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _np(x):
    """torch.Tensor | np.ndarray | list -> np.ndarray (float64, on cpu)."""
    if x is None:
        return None
    if hasattr(x, "detach"):          # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _t(x):
    """torch.Tensor | np.ndarray | list -> float torch.Tensor, KEEPING the
    device it already lives on. The O(n^2) state-cloud metrics use this so GPU
    trajectories are scored on the GPU instead of being downloaded first."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _choice_idx(rng, n, size, device, replace=False, p=None):
    """numpy-rng index draw (keeps the historical rng stream) as a device
    tensor, so subsampling never moves the data itself."""
    idx = rng.choice(n, size=size, replace=replace, p=p)
    return torch.as_tensor(idx, dtype=torch.long, device=device)


def _logsumexp(a, axis=None):
    a = np.asarray(a, dtype=np.float64)
    if axis is None:
        m = np.max(a)
        m = m if np.isfinite(m) else 0.0
        return float(np.log(np.sum(np.exp(a - m))) + m)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)


def _per_traj_logprob(lp, step_axis=1):
    """Reduce (N, T) per-step log-probs to (N,) trajectory log-probs.
    Pass-through if already 1-D."""
    lp = _np(lp)
    if lp.ndim == 1:
        return lp
    return lp.sum(axis=step_axis)


def _flatten_states(states, time_axis=1):
    """(N, T, D) -> dict t -> (N, D) torch tensors, device preserved."""
    states = _t(states)
    if states.ndim == 2:              # (N, D) single snapshot
        return {0: states}
    states = states.movedim(time_axis, 0)             # (T, N, D)
    return {t: states[t] for t in range(states.shape[0])}


# -----------------------------------------------------------------------------
# (A) likelihood-ratio diagnostics
# -----------------------------------------------------------------------------

def ess(log_w):
    """Effective sample size of self-normalised importance weights.
    Returns (ess, ess_fraction). ess_frac ~ 1 good, -> 0 means one weight
    dominates = effectively disjoint support."""
    log_w = _np(log_w).ravel()
    log_w = log_w[np.isfinite(log_w)]
    n = log_w.size
    if n == 0:
        return 0.0, 0.0
    logZ = _logsumexp(log_w)
    log_wn = log_w - logZ                 # normalised log weights
    ess_val = float(np.exp(-_logsumexp(2.0 * log_wn)))  # 1 / sum w_i^2
    return ess_val, ess_val / n


def kl_from_logratio(r_under_F=None, r_under_B=None):
    """r = log(P_F/P_B) per trajectory.
        KL(F||B) =  E_{F}[r]
        KL(B||F) = -E_{B}[r]
    Pass whichever you have. Jeffreys = KL(F||B)+KL(B||F)."""
    out = {}
    if r_under_F is not None:
        rF = _np(r_under_F).ravel()
        out["KL(F||B)"] = float(np.mean(rF[np.isfinite(rF)]))
    if r_under_B is not None:
        rB = _np(r_under_B).ravel()
        out["KL(B||F)"] = float(-np.mean(rB[np.isfinite(rB)]))
    if "KL(F||B)" in out and "KL(B||F)" in out:
        out["Jeffreys"] = out["KL(F||B)"] + out["KL(B||F)"]
    return out


def _pick_logZ(stats, override=None):
    if override is not None:
        return float(override)
    for k in ("log_Z_learned", "log_Z", "log_Z_lb"):
        if k in stats and stats[k] is not None:
            return float(_np(stats[k]).reshape(-1)[0])
    raise KeyError("no log_Z[_learned/_lb] in dict; pass log_Z= explicitly")


def tb_residual(stats, log_Z=None, step_axis=1):
    """GFlowNet trajectory-balance residual per trajectory:

        delta = log_Z + sum(log_pfs) - log_r - sum(log_pbs)
              = log( P_F(tau) / pi(tau) )

    where pi is the target-induced (reward x backward) distribution. delta is
    the log importance weight from the forward sampler to the target, so:
        E_F[delta]  =  KL(P_F || pi)   (forward TB gap, >= 0)
       -E_B[delta]  =  KL(pi || P_F)
    and Var_F[delta] is essentially the TB loss. Returns (delta, log_Z_used).
    """
    lpf = _per_traj_logprob(stats["log_pfs"], step_axis)
    lpb = _per_traj_logprob(stats["log_pbs"], step_axis)
    logR = _np(stats["log_r"]).ravel()
    Z = _pick_logZ(stats, log_Z)
    delta = Z + lpf - logR - lpb
    return delta, Z



def crooks_histograms(r_under_F, r_under_B, bins=60):
    """Returns (edges, hist_F, hist_B, overlap). For the TB residual delta the
    two histograms should cross at delta=0; small overlap = poor coverage."""
    rF = _np(r_under_F).ravel(); rF = rF[np.isfinite(rF)]
    rB = _np(r_under_B).ravel(); rB = rB[np.isfinite(rB)]
    lo = min(rF.min(), rB.min()); hi = max(rF.max(), rB.max())
    edges = np.linspace(lo, hi, bins + 1)
    hF, _ = np.histogram(rF, bins=edges, density=True)
    hB, _ = np.histogram(rB, bins=edges, density=True)
    # crude overlap coefficient of the two densities
    centers = 0.5 * (edges[:-1] + edges[1:])
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    overlap = float(_trapz(np.minimum(hF, hB), centers))
    return edges, hF, hB, overlap


# -----------------------------------------------------------------------------
# (B-1) closed-form per-step policy KL  (diagonal Gaussians)
# -----------------------------------------------------------------------------

def gaussian_step_kl(means_f, logvars_f, means_b, logvars_b, dim_axis=-1):
    """KL( N(mu_f,var_f) || N(mu_b,var_b) ) for diagonal Gaussians, averaged
    over trajectories at each step -> array of length T.

    Handles params shaped (N, T)        -> each entry a univariate Gaussian
                      or  (N, T, D)      -> diagonal D-dim Gaussian (summed).
    A rising curve vs t = the two policies drift apart, and argmax tells you
    *where* along the trajectory.
    """
    mf, lf = _np(means_f), _np(logvars_f)
    mb, lb = _np(means_b), _np(logvars_b)

    # align step counts defensively (fwd/bwd can differ by one)
    T = min(mf.shape[1], mb.shape[1])
    mf, lf, mb, lb = mf[:, :T], lf[:, :T], mb[:, :T], lb[:, :T]
    vf, vb = np.exp(lf), np.exp(lb)

    def _kl(m1, v1, lv1, m2, v2, lv2):
        return 0.5 * (lv2 - lv1 + (v1 + (m1 - m2) ** 2) / v2 - 1.0)

    e_fb = _kl(mf, vf, lf, mb, vb, lb)
    e_bf = _kl(mb, vb, lb, mf, vf, lf)
    if e_fb.ndim == 3:                       # (N, T, D) -> sum over D
        e_fb = e_fb.sum(axis=dim_axis)
        e_bf = e_bf.sum(axis=dim_axis)

    kl_fb_t = e_fb.mean(axis=0)              # (T,)
    kl_bf_t = e_bf.mean(axis=0)
    return kl_fb_t, kl_bf_t, (kl_fb_t + kl_bf_t)


# -----------------------------------------------------------------------------
# (B-2) MMD-vs-t between the two state clouds
# -----------------------------------------------------------------------------

def _pairwise_sq(X, Y):
    xx = (X * X).sum(1)[:, None]
    yy = (Y * Y).sum(1)[None, :]
    return (xx + yy - 2.0 * X @ Y.T).clamp_(min=0.0)


def mmd2_rbf(X, Y, gamma=None, max_n=800, rng=None):
    """Unbiased-ish RBF MMD^2 with median-heuristic bandwidth.
    Subsamples to max_n per cloud for tractability."""
    rng = rng or np.random.default_rng(0)
    X, Y = _t(X), _t(Y)
    if X.shape[0] > max_n:
        X = X[_choice_idx(rng, X.shape[0], max_n, X.device)]
    if Y.shape[0] > max_n:
        Y = Y[_choice_idx(rng, Y.shape[0], max_n, Y.device)]
    Dxx, Dyy, Dxy = _pairwise_sq(X, X), _pairwise_sq(Y, Y), _pairwise_sq(X, Y)
    if gamma is None:
        med = torch.cat([Dxx.reshape(-1), Dyy.reshape(-1), Dxy.reshape(-1)]).median()
        gamma = 1.0 / (float(med) + 1e-12)
    Kxx, Kyy, Kxy = torch.exp(-gamma * Dxx), torch.exp(-gamma * Dyy), torch.exp(-gamma * Dxy)
    m, n = X.shape[0], Y.shape[0]
    # remove diagonal for the within terms
    Kxx.fill_diagonal_(0.0); Kyy.fill_diagonal_(0.0)
    return float(Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1))
                 - 2.0 * Kxy.mean())


def mmd_vs_t(states_F, states_B, time_axis=1, **kw):
    """MMD^2 between forward and backward state clouds at each timestep."""
    SF = _flatten_states(states_F, time_axis)
    SB = _flatten_states(states_B, time_axis)
    ts = sorted(set(SF) & set(SB))
    return np.array(ts), np.array([mmd2_rbf(SF[t], SB[t], **kw) for t in ts])


# -----------------------------------------------------------------------------
# (B-3) nearest-neighbour two-sample test (Schilling-Henze)
# -----------------------------------------------------------------------------

def nn_overlap(X, Y, k=5, max_n=1500, rng=None):
    """Fraction of each point's k-NN that share its own label.
    chance level = max(|X|,|Y|)/(|X|+|Y|)  -> clouds well mixed (overlap)
    near 1.0                               -> cleanly separated (diff manifold)
    """
    rng = rng or np.random.default_rng(0)
    X, Y = _t(X), _t(Y)
    if X.shape[0] > max_n:
        X = X[_choice_idx(rng, X.shape[0], max_n, X.device)]
    if Y.shape[0] > max_n:
        Y = Y[_choice_idx(rng, Y.shape[0], max_n, Y.device)]
    Z = torch.cat([X, Y], dim=0)
    lbl = torch.cat([torch.zeros(len(X), dtype=torch.long, device=Z.device),
                     torch.ones(len(Y), dtype=torch.long, device=Z.device)])
    D = _pairwise_sq(Z, Z)
    D.fill_diagonal_(float("inf"))
    nn = D.topk(k, dim=1, largest=False).indices
    same = (lbl[nn] == lbl[:, None]).float().mean()
    chance = max(len(X), len(Y)) / (len(X) + len(Y))
    return float(same), float(chance)


def nn_vs_t(states_F, states_B, time_axis=1, k=5, **kw):
    SF = _flatten_states(states_F, time_axis)
    SB = _flatten_states(states_B, time_axis)
    ts = sorted(set(SF) & set(SB))
    frac, chance = [], None
    for t in ts:
        f, c = nn_overlap(SF[t], SB[t], k=k, **kw)
        frac.append(f); chance = c
    return np.array(ts), np.array(frac), chance


# -----------------------------------------------------------------------------
# (B-4) one-sided k-NN coverage  (precision / recall, directional)
# -----------------------------------------------------------------------------

def _knn_radius(X, k):
    """For each x in X, distance to its k-th nearest neighbour within X.
    These radii define X's manifold estimate: union of balls B(x_i, r_i)."""
    D = _pairwise_sq(X, X)
    D.fill_diagonal_(float("inf"))
    kk = min(k, X.shape[0] - 1)
    r2 = D.kthvalue(kk, dim=1).values      # squared radius
    return r2


def _covered_mask(Q, R, r2_R):
    """Boolean over Q: is q inside the union of balls B(r_i, sqrt(r2_R[i]))?
    i.e. does q fall within *any* reference point's own k-NN ball."""
    D = _pairwise_sq(Q, R)                 # (nq, nr)
    return (D <= r2_R[None, :]).any(dim=1)


def _resample_weighted(X, w, n, rng):
    w = np.clip(_np(w).ravel(), 0.0, None)
    if w.sum() <= 0 or not np.isfinite(w.sum()):
        idx = _choice_idx(rng, len(X), min(n, len(X)), X.device)
    else:
        idx = _choice_idx(rng, len(X), n, X.device, replace=True, p=w / w.sum())
    return X[idx]


def coverage(X_F, X_B, k=5, weights_B=None, max_n=1000, rng=None):
    """Directional manifold coverage (Kynkaanniemi et al. style).

    precision = fraction of FORWARD points inside BACKWARD's manifold
              = 'forward in backward'   (1 - precision = forward OUTSIDE backward)
    recall    = fraction of BACKWARD points inside FORWARD's manifold
              = 'backward in forward'   (1 - recall    = backward OUTSIDE forward,
                                          i.e. dropped / uncovered data regions)

    If weights_B is given (per-point importance weights for the backward cloud,
    e.g. from corrected bridges), the backward cloud is weight-resampled so both
    its manifold estimate and its query points reflect the true target mass.
    """
    rng = rng or np.random.default_rng(0)
    X_F, X_B = _t(X_F), _t(X_B)

    # forward: plain subsample (model samples are what they are)
    if X_F.shape[0] > max_n:
        X_F = X_F[_choice_idx(rng, X_F.shape[0], max_n, X_F.device)]
    # backward: weight-resample if weights given, else plain subsample
    if weights_B is not None:
        X_B = _resample_weighted(X_B, weights_B, min(max_n, len(X_B)), rng)
    elif X_B.shape[0] > max_n:
        X_B = X_B[_choice_idx(rng, X_B.shape[0], max_n, X_B.device)]

    r2_B = _knn_radius(X_B, k)
    r2_F = _knn_radius(X_F, k)
    precision = float(_covered_mask(X_F, X_B, r2_B).float().mean())   # forward in backward
    recall    = float(_covered_mask(X_B, X_F, r2_F).float().mean())   # backward in forward
    return precision, recall


def coverage_vs_t(states_F, states_B, k=5, weights_B=None, time_axis=1,
                  max_n=1000, rng=None):
    """Per-timestep precision/recall. Returns (ts, precision, recall) arrays.
    Both clouds are pinned at the source, so precision~1 at t=0 and the
    interesting structure is the profile toward the terminal end."""
    rng = rng or np.random.default_rng(0)
    SF = _flatten_states(states_F, time_axis)
    SB = _flatten_states(states_B, time_axis)
    ts = sorted(set(SF) & set(SB))
    prec, rec = [], []
    for t in ts:
        p, r = coverage(SF[t], SB[t], k=k, weights_B=weights_B,
                        max_n=max_n, rng=rng)
        prec.append(p); rec.append(r)
    return np.array(ts), np.array(prec), np.array(rec)


# -----------------------------------------------------------------------------
# top-level report
# -----------------------------------------------------------------------------

def _logw_from_dict(d, step_axis=1):
    """Trajectory log likelihood-ratio r = log P_F - log P_B (per traj)."""
    if "log_pfs" in d and "log_pbs" in d:
        lpf = _per_traj_logprob(d["log_pfs"], step_axis)
        lpb = _per_traj_logprob(d["log_pbs"], step_axis)
        return lpf - lpb
    if "log_r" in d:                       # fallback if that's your ratio
        return _per_traj_logprob(d["log_r"], step_axis)
    raise KeyError("need log_pfs & log_pbs (or log_r) in the dict")


def to_scalars(res):
    """Collapse a report() result into a flat {name: float} dict suitable for
    logging once per eval (wandb.log / csv row). See notes in the chat for
    which ones actually track 'are forward & backward on the same manifold'.
    """
    out = {}

    d = _np(res.get("delta_F"))
    if d is not None:
        d = d[np.isfinite(d)]
        out["tb_gap"] = float(d.mean())          # KL(P_F||pi); depends on log_Z
        out["tb_resid_std"] = float(d.std())     # ~ sqrt(TB loss); Z-invariant

    # 'ess_frac' dropped: it duplicated eval_fwd/ess_frac (quick_tb_stats
    # computes the same Kish ESS on the same forward batch) to 1e-7.

    pr = _np(res.get("policy_ratio_F"))
    if pr is not None:
        pr = pr[np.isfinite(pr)]
        out["policy_ratio_mean"] = float(pr.mean())
        out["policy_ratio_std"] = float(pr.std())

    ck = res.get("crooks")
    if ck is not None:
        edges, hF, hB = ck
        c = 0.5 * (edges[:-1] + edges[1:])
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        out["delta_overlap"] = float(_trapz(np.minimum(hF, hB), c))  # 0..1

    # 'stepkl_sum/max/argmax_t/drift' retired 2026-08-23 (owner decision). They
    # reduced the per-timestep fwd/bwd step-kernel KL curve to four numbers; the
    # curve itself is what carries the information and nothing read the
    # reductions. `res['step_kl']` is still computed by traj_overlap_report for
    # any caller that wants the curve -- only the logging is gone.

    mv = res.get("mmd_vs_t")
    if mv is not None:
        m = _np(mv[1])
        out["mmd_mean"] = float(m.mean())
        out["mmd_max"] = float(m.max())
        out["mmd_final"] = float(m[-1])
        # no 'mmd_drift': both clouds are pinned at the source, so m[0] is
        # identically 0 and m[-1] - m[0] was a bit-exact copy of mmd_final
        # (verified 0.0 max deviation on 0j0tg0iq and 1xz7zd9n). Same for
        # nn_sep_drift below.

    nv = res.get("nn_vs_t")
    if nv is not None:
        frac, chance = _np(nv[1]), float(nv[2])
        # normalised manifold separation: 0 = clouds mixed, 1 = fully disjoint
        sep = np.clip((frac - chance) / max(1e-9, 1.0 - chance), 0.0, 1.0)
        out["nn_sep_mean"] = float(sep.mean())
        out["nn_sep_final"] = float(sep[-1])     # late-t separation: THE metric
        out["nn_sep_max"] = float(sep.max())
        # no 'nn_sep_drift': sep is clipped at 0 and the clouds start mixed, so
        # sep[0] is exactly 0 every eval and the drift was a copy of nn_sep_final

    cv = res.get("coverage_pr")
    if cv is not None:
        ts, prec, rec = _np(cv[0]), _np(cv[1]), _np(cv[2])
        # precision = forward-in-backward ; recall = backward-in-forward.
        # Only the 'outside' complements are logged -- precision_final and
        # recall_final were exactly 1 - these, i.e. the same two numbers twice.
        out["fwd_outside_bwd_final"] = float(1.0 - prec[-1])   # the headline
        out["bwd_outside_fwd_final"] = float(1.0 - rec[-1])    # dropped data
        # area under the 'outside' profiles over t (mean across steps)
        out["fwd_outside_bwd_auc"] = float((1.0 - prec).mean())
        out["bwd_outside_fwd_auc"] = float((1.0 - rec).mean())

    return out


def traj_overlap_report(data_F, data_B=None, step_axis=1, dim_axis=2, time_axis=1,
                        bins=60, k=5, log_Z=None, weights_B=None):
    line = "=" * 70

    # ---- (A) TB-residual / target overlap ----------------------------------
    # print("\n" + line)
    # print("(A) LIKELIHOOD OVERLAP  (trajectory-balance residual)")
    # print(line)
    dF, Zf = tb_residual(data_F, log_Z, step_axis)
    # print(f"  log_Z used (F) = {Zf:.4f}")
    # print(f"  delta under F: mean={dF.mean():.4f}  std={dF.std():.4f}  "
    #       f"(mean = KL(P_F||pi) = fwd TB gap; std^2 ~ TB loss)")
    eF, fF = ess(-dF)        # forward sampler reweighted to target pi
    # print(f"  ESS (forward -> target pi): {eF:9.1f}  frac={fF:.4f}")
    # print("     frac~1: forward covers target; frac->0: forward misses mass")

    crooks = None
    if data_B is not None:
        dB, Zb = tb_residual(data_B, log_Z, step_axis)
        klFpi = float(dF.mean())
        klpiF = float(-dB.mean())
        # print(f"  delta under B: mean={dB.mean():.4f}  std={dB.std():.4f}")
        eB, fB = ess(dB)     # backward (target) samples reweighted to forward
        # print(f"  ESS (target -> forward)   : {eB:9.1f}  frac={fB:.4f}")
        # print(f"  KL(P_F||pi) = {klFpi:.4f}")
        # print(f"  KL(pi||P_F) = {klpiF:.4f}")
        # print(f"  Jeffreys    = {klFpi + klpiF:.4f}")
        edges, hF, hB, ov = crooks_histograms(dF, dB, bins=bins)
        crooks = (edges, hF, hB)
        # print(f"  delta-histogram overlap coeff = {ov:.3f}  "
        #       f"(0 disjoint, 1 identical)")

    # ---- (A') raw policy ratio (secondary) ---------------------------------
    # print("\n" + "-" * 70)
    # print("(A') pure policy ratio  r = sum(log_pfs) - sum(log_pbs)")
    rF = _logw_from_dict(data_F, step_axis)
    e1, f1 = ess(-rF); e2, f2 = ess(rF)
    # print(f"  r under F: mean={rF.mean():.4f} std={rF.std():.4f}")
    # print(f"  ESS(F as proposal for B) frac={f1:.4f} | "
    #       f"ESS(B as proposal for F) frac={f2:.4f}")

    # ---- (B-1) per-step policy KL ------------------------------------------
    # print("\n" + line); print("(B) GEOMETRIC OVERLAP / DRIFT"); print(line)
    step_kl = None
    if all(kk in data_F for kk in ("means_f", "logvars_f",
                                   "means_b", "logvars_b")):
        kl_fb, kl_bf, kl_sym = gaussian_step_kl(
            data_F["means_f"], data_F["logvars_f"],
            data_F["means_b"], data_F["logvars_b"], dim_axis=dim_axis)
        step_kl = (kl_fb, kl_bf, kl_sym)
        # print(f"  per-step policy KL(F||B): first={kl_fb[0]:.3f} "
        #       f"last={kl_fb[-1]:.3f} max={kl_fb.max():.3f} "
        #       f"@t={int(kl_fb.argmax())}")
        # print(f"  per-step Jeffreys sum = {kl_sym.sum():.3f}  "
        #       f"(rising curve = drift; plot the returned arrays vs t)")

    # ---- (B-2 / B-3) state-cloud tests -------------------------------------
    mmd, nn, coverage_pr = None, None, None
    if data_B is not None and "flow_states" in data_F and "flow_states" in data_B:
        ts, mmds = mmd_vs_t(data_F["flow_states"], data_B["flow_states"],
                            time_axis=time_axis)
        mmd = (ts, mmds)
        # print(f"  MMD^2 vs t: first={mmds[0]:.4f} last={mmds[-1]:.4f} "
        #       f"max={mmds.max():.4f} @t={int(ts[mmds.argmax()])}")
        tsn, frac, chance = nn_vs_t(data_F["flow_states"],
                                    data_B["flow_states"],
                                    time_axis=time_axis, k=k)
        nn = (tsn, frac, chance)
        # print(f"  NN same-label frac vs t: mean={frac.mean():.3f} "
        #       f"(chance={chance:.3f}; ->1 = separated manifolds)")

        # ---- (B-4) directional coverage (precision/recall) -----------------
        tsc, prec, rec = coverage_vs_t(
            data_F["flow_states"], data_B["flow_states"],
            k=k, weights_B=weights_B, time_axis=time_axis)
        coverage_pr = (tsc, prec, rec)
        # print(f"  forward-in-backward (precision): final={prec[-1]:.3f} "
        #       f"min={prec.min():.3f}  -> 1-this = forward OUTSIDE backward")
        # print(f"  backward-in-forward (recall)   : final={rec[-1]:.3f} "
        #       f"min={rec.min():.3f}  -> 1-this = data regions forward MISSES")
        if weights_B is None:
            pass # print("    (backward cloud UNWEIGHTED: measures vs raw bridge, "
                  # "not true pi; pass weights_B= if corrected bridges)")
    elif "flow_states" in data_F:
        pass
        # print("  (MMD / NN / coverage need both data_F and data_B)")

    # print("\n" + line); print("done"); print(line)
    return {
        "delta_F": dF,
        "ess_target": {"F_to_pi": (eF, fF)},
        "policy_ratio_F": rF,
        "crooks": crooks,
        "step_kl": step_kl,
        "mmd_vs_t": mmd,
        "nn_vs_t": nn,
        "coverage_pr": coverage_pr,
    }


# -----------------------------------------------------------------------------
# optional plotting (only if matplotlib present)
# -----------------------------------------------------------------------------

def plot(res, savepath=None):
    import matplotlib.pyplot as plt
    panels = [p for p in ("crooks", "step_kl", "mmd_vs_t", "nn_vs_t",
                           "coverage_pr") if res.get(p) is not None]
    if not panels:
        print("nothing to plot"); return
    fig, ax = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 3.6))
    ax = np.atleast_1d(ax); i = 0
    if res.get("crooks") is not None:
        edges, hF, hB = res["crooks"]; c = 0.5 * (edges[:-1] + edges[1:])
        ax[i].plot(c, hF, label="under F"); ax[i].plot(c, hB, label="under B")
        ax[i].set_title("Crooks histograms"); ax[i].set_xlabel("r=logP_F/P_B")
        ax[i].legend(); i += 1
    if res.get("step_kl") is not None:
        kl_fb, kl_bf, kl_sym = res["step_kl"]
        ax[i].plot(kl_fb, label="KL(F||B)"); ax[i].plot(kl_bf, label="KL(B||F)")
        ax[i].set_title("per-step policy KL"); ax[i].set_xlabel("t")
        ax[i].legend(); i += 1
    if res.get("mmd_vs_t") is not None:
        ts, m = res["mmd_vs_t"]; ax[i].plot(ts, m)
        ax[i].set_title("MMD$^2$ vs t"); ax[i].set_xlabel("t"); i += 1
    if res.get("nn_vs_t") is not None:
        ts, frac, chance = res["nn_vs_t"]; ax[i].plot(ts, frac)
        ax[i].axhline(chance, ls="--", c="k", lw=1, label="chance")
        ax[i].set_title("NN same-label vs t"); ax[i].set_xlabel("t")
        ax[i].legend(); i += 1
    if res.get("coverage_pr") is not None:
        ts, prec, rec = res["coverage_pr"]
        ax[i].plot(ts, 1 - prec, label="fwd outside bwd")
        ax[i].plot(ts, 1 - rec, label="bwd outside fwd")
        ax[i].set_title("directional gap vs t"); ax[i].set_xlabel("t")
        ax[i].set_ylim(-0.02, 1.02); ax[i].legend(); i += 1
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140); print("saved", savepath)
    else:
        plt.show()


if __name__ == "__main__":
    # synthetic data matching the real shapes:
    #   means/logvars (N, T_f)=(N,10), flow_states (N, 11, 12),
    #   log_pfs (N,10), log_pbs (N,9), log_r (N,), log_Z_learned scalar
    rng = np.random.default_rng(0)
    N, Tf, Tb, S, D = 10000, 10, 9, 11, 12
    def fake(shift):
        return {
            "means_f":   rng.normal(0, 1, (N, Tf)),
            "logvars_f": rng.normal(-1, .2, (N, Tf)),
            "means_b":   rng.normal(shift, 1, (N, Tf)),
            "logvars_b": rng.normal(-1, .2, (N, Tf)),
            "flow_states": rng.normal(shift, 1, (N, S, D)),
            "log_pfs": rng.normal(-3, 1, (N, Tf)),
            "log_pbs": rng.normal(-3, 1, (N, Tb)),
            "log_r":   rng.normal(-5, 1, (N,)),
            "log_Z_learned": np.array(2.0),
        }
    res = traj_overlap_report(fake(0.0), fake(0.0),
                              weights_B=rng.random(N))   # exercise weighted coverage path
    print("\nLOGGABLE SCALARS:")
    for kk, vv in to_scalars(res).items():
        print(f"  {kk:20s} {vv}")
    # plot(res)  # uncomment if matplotlib available