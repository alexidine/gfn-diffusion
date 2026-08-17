"""Interpretable quality metrics for an internal-coordinate prior.

WHY NOT JUST REPORT ESS. A raw ESS fraction is not comparable to anything. It falls with
dimension, so two molecules cannot be compared; and it is bounded above by a ceiling that
is a property of the PRODUCT FORM rather than of the fit, so it cannot say whether a
number is good. Measured on ethanol, the best product-of-1D-marginals proposal that can be
built from the exact energy reaches 6.55%, not ~100% -- so a fitted prior at 1.09% is
using 17% of what is achievable, which is the useful statement, and "1.09%" on its own is
not.

So every number here is reported against the ORACLE: the same machinery driven by each
coordinate's exact 1-D Boltzmann marginal instead of the fitted tables. That splits the
total mismatch into a part the fit could remove and a part inherent to a per-coordinate
proposal.

    D          = log(1 / ESS_fraction), in nats. Additive and roughly a log-chi-squared
                 divergence, so differences are meaningful where ratios of ESS are not.
    D_oracle   = the product form's own cost. Irreducible without joint structure.
    D_avoidable= D_fitted - D_oracle. THE ACTIONABLE NUMBER: nats the fit is leaving.
    eta        = ESS_fitted / ESS_oracle in [0, 1]. The same thing as a fraction.
    D_oracle/d = per-dimension cost of the product form; says whether a per-coordinate
                 proposal is viable for this molecule AT ALL, independent of the fit.

ESS is a ratio of noisy sums, so all of these are reported with bootstrap intervals. At
n=6000 and ESS=1% only ~60 draws carry the estimate, and an interval that straddles a
factor of two is the honest reading of that.
"""
from typing import Optional

import numpy as np
import torch

NGRID = 721


def ess_fraction(logw: np.ndarray) -> float:
    w = np.exp(logw - logw.max())
    return float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))


def _boot(logw: np.ndarray, n_boot: int, rng) -> tuple:
    idx = rng.integers(0, len(logw), (n_boot, len(logw)))
    vals = np.array([ess_fraction(logw[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _gauss_lp(x, mu, s):
    return -0.5 * ((x - mu) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)


def _wrapped_lp(x, mu, s):
    acc = 0.0
    for k in (-1, 0, 1):
        acc = acc + np.exp(_gauss_lp(x + 2 * np.pi * k, mu, s))
    return np.log(np.clip(acc, 1e-300, None))


def oracle_logw(en, n: int = 6000, seed: int = 0, report_modes: bool = False):
    """Log importance weights for the best product-form proposal we can construct.

    Each group leader is sampled from a 1-D scan of the TRUE energy with its followers
    tracking, so the rotamer structure is measured rather than fitted. r, theta and the
    improper rows use the exact Gaussian marginal of their harmonic term.

    THIS IS A LOWER BOUND ON THE PRODUCT-FORM CEILING, not the ceiling. The r/theta
    marginals are exact only for their own term in isolation, and the redundant graph
    angles couple them -- which is measurably where most of the remaining cost sits, so a
    better construction would raise this number. Treat eta as "fraction of a conservative
    ceiling", and do not read eta = 1 as "nothing left to gain".
    """
    T = float(en.temperature)
    n_r, n_th = en.n_r, en.n_th
    n0 = n_r + n_th
    r0 = en.r0.detach().cpu().numpy()
    th0 = en.th0.detach().cpu().numpy()
    ph0 = en.ph0.detach().cpu().numpy()
    s_r, s_th = en.thermal_rtheta_sigma(T)
    s_imp = en.improper_phi_sigma(T)
    imp = en.improper_phi_rows()
    groups = en.torsion_groups()
    g_sigma = en.sibling_jitter_sigma(groups, T)
    ref = np.concatenate([r0, th0, ph0])
    t = lambda a: torch.as_tensor(a, dtype=en.dtype, device=en.device)

    def energy_of(d):
        x = en.state_from_dof(t(d[:, :n_r]), t(d[:, n_r:n0]), t(d[:, n0:])).clamp(-1, 1)
        return en.energy(x).detach().cpu().numpy()

    grid = np.linspace(-np.pi, np.pi, NGRID, endpoint=False)
    tables, modes = {}, {}
    for gi, rows_j in enumerate(groups):
        lead = rows_j[0]
        d = np.repeat(ref[None], NGRID, 0)
        d[:, n0 + lead] = grid
        disp = (grid - ph0[lead] + np.pi) % (2 * np.pi) - np.pi
        for i in rows_j[1:]:
            d[:, n0 + i] = ph0[i] + disp
        lp = -energy_of(d)
        p = np.exp(lp - lp.max())
        p = p / p.sum()
        tables[gi] = (grid, p, np.cumsum(p))
        if report_modes:
            modes[gi] = [float(np.degrees(grid[k])) for k in range(NGRID)
                         if p[k] > p[(k - 1) % NGRID] and p[k] > p[(k + 1) % NGRID]
                         and p[k] > 0.05 * p.max()]

    rng = np.random.default_rng(seed)
    dof = np.repeat(ref[None], n, 0)
    logq = np.zeros(n)
    for j in range(n_r):
        dof[:, j] = rng.normal(r0[j], s_r[j], n)
        logq += _gauss_lp(dof[:, j], r0[j], s_r[j])
    for j in range(n_th):
        dof[:, n_r + j] = rng.normal(th0[j], s_th[j], n)
        logq += _gauss_lp(dof[:, n_r + j], th0[j], s_th[j])
    for j in imp:
        dof[:, n0 + j] = ph0[j] + rng.normal(0.0, s_imp, n)
        logq += _wrapped_lp(dof[:, n0 + j], ph0[j], s_imp)
    step = grid[1] - grid[0]
    for gi, rows_j in enumerate(groups):
        g_, p, cdf = tables[gi]
        lead = rows_j[0]
        idx = np.searchsorted(cdf, rng.uniform(0, 1, n)).clip(0, NGRID - 1)
        val = (g_[idx] + rng.uniform(0, step, n) + np.pi) % (2 * np.pi) - np.pi
        dof[:, n0 + lead] = val
        dens = np.interp(val, g_, p / step, period=2 * np.pi)
        logq += np.log(np.clip(dens, 1e-300, None))
        disp = (val - ph0[lead] + np.pi) % (2 * np.pi) - np.pi
        for i in rows_j[1:]:
            dof[:, n0 + i] = ph0[i] + disp + rng.normal(0.0, g_sigma[gi], n)
            logq += _wrapped_lp(dof[:, n0 + i], ph0[i] + disp, g_sigma[gi])

    logw = -energy_of(dof) - logq
    return (logw, modes) if report_modes else logw


def prior_report(en, prior, n: int = 6000, seed: int = 0, n_boot: int = 200,
                 blocks: bool = True, oracle: bool = True) -> dict:
    """Fitted-vs-oracle quality for one molecule. Every field is defined in the module docstring."""
    rng = np.random.default_rng(seed)
    x, stats = en.sample_prior_states(prior, n, rng, report=False, joint_rings=False)
    logw_fit = (-en.energy(x).detach().cpu().numpy()
                - en.prior_log_prob(prior, stats['dof']))

    clip = max([v for v in stats.get('clip_frac', {}).values()] or [0.0])
    out = {'d': int(en.ndim), 'n': n, 'clip_frac': clip,
           'ess_fitted': ess_fraction(logw_fit),
           'sd_logw_fitted': float(logw_fit.std())}
    out['ci_fitted'] = _boot(logw_fit, n_boot, np.random.default_rng(seed + 1))
    out['D_fitted'] = float(-np.log(out['ess_fitted']))

    if oracle:
        logw_or = oracle_logw(en, n=n, seed=seed)
        out['ess_oracle'] = ess_fraction(logw_or)
        out['sd_logw_oracle'] = float(logw_or.std())
        out['ci_oracle'] = _boot(logw_or, n_boot, np.random.default_rng(seed + 2))
        out['D_oracle'] = float(-np.log(out['ess_oracle']))
        out['D_avoidable'] = out['D_fitted'] - out['D_oracle']
        out['eta'] = out['ess_fitted'] / out['ess_oracle']
        out['D_oracle_per_dim'] = out['D_oracle'] / out['d']

    if blocks:
        n_r, n_th = en.n_r, en.n_th
        n0 = n_r + n_th
        ref = np.concatenate([en.r0.detach().cpu().numpy(),
                              en.th0.detach().cpu().numpy(),
                              en.ph0.detach().cpu().numpy()])
        t = lambda a: torch.as_tensor(a, dtype=en.dtype, device=en.device)
        dof = stats['dof']
        for tag, sl in (('r', slice(0, n_r)), ('theta', slice(n_r, n0)),
                        ('phi', slice(n0, en.ndim))):
            d = np.repeat(ref[None], n, 0)
            d[:, sl] = dof[:, sl]
            xx = en.state_from_dof(t(d[:, :n_r]), t(d[:, n_r:n0]),
                                   t(d[:, n0:])).clamp(-1, 1)
            lw = -en.energy(xx).detach().cpu().numpy() - en.prior_log_prob(prior, d)
            out[f'ess_{tag}'] = ess_fraction(lw)
    return out


def rotamer_modes(en, min_rel: float = 0.05):
    """Per-group rotamer mode centres, in radians, from a 1-D scan of the TRUE energy.

    Returns ``[(rows, centres)]`` -- one entry per rotatable group. Improper rows are not
    included: they are angles, not rotamers, and have a single minimum by construction.
    """
    T = float(en.temperature)
    n_r, n_th = en.n_r, en.n_th
    n0 = n_r + n_th
    ph0 = en.ph0.detach().cpu().numpy()
    ref = np.concatenate([en.r0.detach().cpu().numpy(),
                          en.th0.detach().cpu().numpy(), ph0])
    t = lambda a: torch.as_tensor(a, dtype=en.dtype, device=en.device)
    grid = np.linspace(-np.pi, np.pi, NGRID, endpoint=False)
    out = []
    for rows_j in en.torsion_groups():
        lead = rows_j[0]
        d = np.repeat(ref[None], NGRID, 0)
        d[:, n0 + lead] = grid
        disp = (grid - ph0[lead] + np.pi) % (2 * np.pi) - np.pi
        for i in rows_j[1:]:
            d[:, n0 + i] = ph0[i] + disp
        x = en.state_from_dof(t(d[:, :n_r]), t(d[:, n_r:n0]), t(d[:, n0:])).clamp(-1, 1)
        p = np.exp(-en.energy(x).detach().cpu().numpy() + 0.0)
        p = p / p.max()
        pk = [k for k in range(NGRID)
              if p[k] > p[(k - 1) % NGRID] and p[k] >= p[(k + 1) % NGRID]
              and p[k] > min_rel]
        out.append((rows_j, grid[pk] if pk else np.array([ph0[lead]])))
    return out


def coverage_report(en, prior, n: int = 6000, seed: int = 0, max_modes: int = 512,
                    accessible_kt: float = 10.0) -> dict:
    """Does the prior REACH every energetically accessible rotamer basin?

    This is the question ESS cannot answer. A self-normalised ESS is built from the draws
    you got, so a basin the prior never proposes contributes no large weight and no
    warning -- the estimate looks healthy precisely where the prior is broken. Coverage
    has to be measured in the REVERSE direction: enumerate the target's basins first, then
    ask what the prior assigns them.

    Modes are the product of per-group rotamer centres, so 3^(rotatable groups). Each is
    realised with r/theta/impropers at the reference and every group's leader at its
    centre, and counted ACCESSIBLE if it sits within `accessible_kt` of the best mode.
    Prior draws are assigned to a mode by nearest centre in each group's leader dihedral.

    A methyl's three rotamers are physically indistinguishable by symmetry, so they are
    counted three times here. That inflates the mode count but does not bias the failure
    the metric is for -- a symmetric triple is either all reached or all missed.
    """
    import itertools
    T = float(en.temperature)
    n_r, n_th = en.n_r, en.n_th
    n0 = n_r + n_th
    ph0 = en.ph0.detach().cpu().numpy()
    ref = np.concatenate([en.r0.detach().cpu().numpy(),
                          en.th0.detach().cpu().numpy(), ph0])
    t = lambda a: torch.as_tensor(a, dtype=en.dtype, device=en.device)
    groups = rotamer_modes(en)
    n_comb = int(np.prod([len(c) for _, c in groups])) if groups else 1
    if n_comb > max_modes:
        return {'skipped': f'{n_comb} modes exceeds max_modes={max_modes}'}

    combos = list(itertools.product(*[range(len(c)) for _, c in groups]))
    d = np.repeat(ref[None], len(combos), 0)
    for m, combo in enumerate(combos):
        for gi, (rows_j, centres) in enumerate(groups):
            lead = rows_j[0]
            val = centres[combo[gi]]
            d[m, n0 + lead] = val
            disp = (val - ph0[lead] + np.pi) % (2 * np.pi) - np.pi
            for i in rows_j[1:]:
                d[m, n0 + i] = ph0[i] + disp
    x = en.state_from_dof(t(d[:, :n_r]), t(d[:, n_r:n0]), t(d[:, n0:])).clamp(-1, 1)
    e_mode = en.potential_energy(x, T).detach().cpu().numpy()
    e_best = float(e_mode.min())
    accessible = (e_mode - e_best) / T <= accessible_kt

    # ---- where do prior draws land? ----
    rng = np.random.default_rng(seed)
    xs, stats = en.sample_prior_states(prior, n, rng, report=False, joint_rings=False)
    dof = stats['dof']
    lab = np.zeros(n, dtype=np.int64)
    stride = 1
    for gi in range(len(groups) - 1, -1, -1):
        rows_j, centres = groups[gi]
        v = dof[:, n0 + rows_j[0]]
        dist = np.abs((v[:, None] - centres[None, :] + np.pi) % (2 * np.pi) - np.pi)
        lab += stride * np.argmin(dist, axis=1)
        stride *= len(centres)
    counts = np.bincount(lab, minlength=len(combos))
    frac = counts / n

    acc_idx = np.where(accessible)[0]
    missed = [int(i) for i in acc_idx if counts[i] == 0]
    # binomial 95% upper bound on the probability of a basin with zero draws
    p_upper = 3.0 / n

    e_draw = en.potential_energy(xs, T).detach().cpu().numpy()
    excess = (e_draw - e_best) / T - en.ndim / 2.0

    return {
        'd': int(en.ndim), 'n_modes': len(combos),
        'n_accessible': int(accessible.sum()),
        'n_missed': len(missed),
        'worst_frac': float(frac[acc_idx].min()) if len(acc_idx) else float('nan'),
        'expected_frac': 1.0 / max(int(accessible.sum()), 1),
        'p_upper_if_zero': p_upper,
        'excess_median_kt': float(np.median(excess)),
        'excess_p90_kt': float(np.percentile(excess, 90)),
        'frac_within_equipartition': float((excess <= 0).mean()),
        'mode_fracs': frac, 'mode_energies': (e_mode - e_best) / T,
        'accessible': accessible,
    }


def format_coverage(name: str, r: dict) -> str:
    if 'skipped' in r:
        return f'{name:<20} skipped: {r["skipped"]}'
    return (f'{name:<20} d={r["d"]:<3} modes {r["n_modes"]:>4} '
            f'accessible {r["n_accessible"]:>4}  MISSED {r["n_missed"]:>3}  '
            f'worst {r["worst_frac"] * 100:6.3f}% (even={r["expected_frac"] * 100:5.2f}%)  '
            f'excess med {r["excess_median_kt"]:+7.1f} kT  p90 {r["excess_p90_kt"]:+7.1f}  '
            f'thermal-or-better {r["frac_within_equipartition"] * 100:5.1f}%')


def format_report(name: str, rep: dict) -> str:
    s = (f'{name:<20} d={rep["d"]:<3} '
         f'ESS {rep["ess_fitted"] * 100:6.2f}% '
         f'[{rep["ci_fitted"][0] * 100:5.2f},{rep["ci_fitted"][1] * 100:5.2f}]')
    if 'eta' in rep:
        s += (f'  oracle {rep["ess_oracle"] * 100:6.2f}%'
              f'  eta {rep["eta"] * 100:5.1f}%'
              f'  D_avoid {rep["D_avoidable"]:5.2f}'
              f'  D_or/dim {rep["D_oracle_per_dim"]:.3f}')
    if 'ess_r' in rep:
        s += (f'   [r {rep["ess_r"] * 100:.0f}% th {rep["ess_theta"] * 100:.0f}% '
              f'ph {rep["ess_phi"] * 100:.0f}%]')
    if rep['clip_frac'] > 1e-3:
        s += f'  !! clip {rep["clip_frac"]:.3f}'
    return s
