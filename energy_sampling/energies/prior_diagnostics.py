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


def dof_to_state(en, d):
    """``(r, theta, phi)`` -> state ``x``, at ANY level. THE ONLY THING THAT WAS MISSING.

    ``state_from_dof`` refuses at a collective level and its message names
    ``build_prior_states.draw_states`` as the torsion route. This IS that route, and it is
    a translation rather than a derivation: there is no row-wise inverse because one column
    drives several dihedrals, but the column is simply the LEADER'S DISPLACEMENT over the
    free scale -- exactly what ``draw_states`` writes as
    ``x[:, j] = wrap((phi - phi_ref) / pi)``, with the leader taken from ``mask[:, j]`` and
    ``phi_ref = ph0[leader]``. No new machinery; one branch.
    """
    n_r, n0_ = en.n_r, en.n_r + en.n_th
    t = lambda a: torch.as_tensor(a, dtype=en.dtype, device=en.device)
    if not en.collective:
        return en.state_from_dof(t(d[:, :n_r]), t(d[:, n_r:n0_]), t(d[:, n0_:])).clamp(-1, 1)
    mask = en.mask.detach().cpu().numpy()
    ph0 = en.ph0.detach().cpu().numpy()
    x = np.empty((d.shape[0], mask.shape[1]))
    for j in range(mask.shape[1]):
        lead = int(np.flatnonzero(mask[:, j] != 0)[0])
        x[:, j] = ((d[:, n0_ + lead] - ph0[lead]) / np.pi + 1.0) % 2.0 - 1.0
    return t(x).clamp(-1, 1)


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
        return en.energy(dof_to_state(en, d)).detach().cpu().numpy()

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
    # AT A COLLECTIVE LEVEL THIS MACHINERY DROPS OUT, it is not translated. r, theta and
    # every improper row are FROZEN there, and dof_to_state reads only the leader columns --
    # so perturbing them would add logq terms with no counterpart in the state and the
    # weights would be nonsense (measured: eta of 171440%, i.e. an "oracle" 1700x WORSE
    # than the fitted prior). What remains at torsion is one 1-D energy scan per rotatable
    # bond, which is the whole of the oracle there.
    if not en.collective:
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
            if en.collective:
                # the column drives every follower rigidly: no jitter, and a deterministic
                # follower contributes no density term
                dof[:, n0 + i] = ph0[i] + disp
                continue
            dof[:, n0 + i] = ph0[i] + disp + rng.normal(0.0, g_sigma[gi], n)
            logq += _wrapped_lp(dof[:, n0 + i], ph0[i] + disp, g_sigma[gi])

    logw = -energy_of(dof) - logq
    return (logw, modes) if report_modes else logw


def prior_report(en, prior, n: int = 6000, seed: int = 0, n_boot: int = 200,
                 blocks: bool = True, oracle: bool = True) -> dict:
    """Fitted-vs-oracle quality for one molecule. Every field is defined in the module docstring."""
    rng = np.random.default_rng(seed)
    if en.collective:
        # THE TORSION ROUTE, named by state_from_dof's own refusal. draw_states samples the
        # leader per rotatable bond and writes the state column directly, so it never needs
        # the inverse that does not exist. The followers are then DETERMINISTIC given the
        # leader, which makes their prior_log_prob terms constant across draws -- and a
        # constant cancels in a self-normalised ESS, which is why the same density scores
        # both routes.
        import contextlib
        import io
        from build_prior_states import draw_states
        with contextlib.redirect_stdout(io.StringIO()):
            xt, _ = draw_states(en, prior, n, rng)
        x = xt.to(en.dtype)
        dof = np.concatenate([a.detach().cpu().numpy()
                              for a in en.dof_from_state(x)], axis=1)
        # draw_states writes wrapped columns, so nothing can land outside the box
        clip = 0.0
    else:
        # joint_rings left at its DEFAULT. This function scores draws with prior_log_prob,
        # which refuses on any ring block, so a ring molecule is skipped below either way --
        # but passing joint_rings=False here said "measure the disabled path", and that is
        # what the reference table inherited.
        x, stats = en.sample_prior_states(prior, n, rng, report=False)
        dof = stats['dof']
        clip = max([v for v in stats.get('clip_frac', {}).values()] or [0.0])
    # TWO LOUD EXCLUSIONS, stated rather than routed around.
    # (1) prior_log_prob raises on ANY ring block at EVERY level: a bank/pucker density is a
    #     mixture that is singular in the held directions, so there is no weight to quote.
    #     That is Track C1. A labelled skip, never an approximation.
    try:
        logq_fit = en.prior_log_prob(prior, dof)
    except NotImplementedError as exc:
        return {'d': int(en.ndim), 'n': n, 'clip_frac': clip,
                'skipped': 'prior_log_prob: {}'.format(str(exc).split('.')[0].strip())}
    # (2) the box clamp is NOT represented in the density -- it puts finite mass exactly on
    #     the wall, where no continuous density can put any. Weights are valid only when
    #     essentially nothing was clipped.
    if clip > 1e-3:
        return {'d': int(en.ndim), 'n': n, 'clip_frac': clip,
                'skipped': 'clip_frac={:.4f}: mass sits ON the box wall, which the density '
                           'does not represent, so these weights would be quietly wrong'
                           .format(clip)}
    logw_fit = -en.energy(x).detach().cpu().numpy() - logq_fit
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
        for tag, sl in (('r', slice(0, n_r)), ('theta', slice(n_r, n0)),
                        ('phi', slice(n0, en.ndim))):
            d = np.repeat(ref[None], n, 0)
            d[:, sl] = dof[:, sl]
            xx = dof_to_state(en, d)
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
        x = dof_to_state(en, d)
        p = np.exp(-en.energy(x).detach().cpu().numpy() + 0.0)
        p = p / p.max()
        pk = [k for k in range(NGRID)
              if p[k] > p[(k - 1) % NGRID] and p[k] >= p[(k + 1) % NGRID]
              and p[k] > min_rel]
        out.append((rows_j, grid[pk] if pk else np.array([ph0[lead]])))
    return out


def basin_reference(en, max_modes: int = 512, accessible_kt: float = 10.0) -> dict:
    """Enumerate the target's rotamer basins and score them. SAMPLER-INDEPENDENT.

    Split out of coverage_report so the SAME basin definition can be applied to draws that
    did not come from an InternalPrior -- a trained policy's samples, most importantly.
    Coverage is only meaningful measured in this direction (enumerate the target's basins
    first, then ask what a sampler assigns them), so a second, differently-defined basin
    set would silently make two coverage numbers incomparable.

    Depends only on the molecule and the force field, so a caller evaluating repeatedly
    should compute it ONCE and keep it.
    """
    import itertools

    T = float(en.temperature)
    n0 = en.n_r + en.n_th
    ph0 = en.ph0.detach().cpu().numpy()
    ref = np.concatenate([en.r0.detach().cpu().numpy(),
                          en.th0.detach().cpu().numpy(), ph0])
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
    e_mode = en.potential_energy(dof_to_state(en, d), T).detach().cpu().numpy()
    e_best = float(e_mode.min())
    return {
        'groups': groups, 'combos': combos, 'n0': n0,
        'mode_energies_raw': e_mode, 'e_best': e_best,
        'mode_energies': (e_mode - e_best) / T,
        'accessible': (e_mode - e_best) / T <= accessible_kt,
    }


def rotamer_basin_labels(groups, dof, n0: int) -> np.ndarray:
    """Per-draw ROTAMER basin index, ``[n]``, by nearest centre in each group's leader.

    NOT ``basin_labels`` -- that name is taken, by ``ring_metrics.basin_labels``, for RING
    PUCKER identity. The two are different partitions of different coordinates, and giving
    them one name is how a coverage number and a pucker number end up silently compared.

    Takes ``dof`` directly rather than a sampler, so prior draws and policy samples are
    labelled by identical code -- see basin_reference for why that matters.
    """
    L = rotamer_group_labels(groups, dof, n0)
    lab = np.zeros(len(dof), dtype=np.int64)
    stride = 1
    # group g-1 is the LEAST significant digit, which is what makes this agree with
    # basin_reference's itertools.product ordering (product varies the last factor
    # fastest). basin_coverage indexes `combos` by this label, so the two must not drift.
    for gi in range(len(groups) - 1, -1, -1):
        lab += stride * L[:, gi]
        stride *= len(groups[gi][1])
    return lab


def rotamer_group_labels(groups, dof, n0: int) -> np.ndarray:
    """PER-GROUP rotamer labels, ``[n, n_groups]`` -- the un-collapsed labelling.

    ``rotamer_basin_labels`` reduces this to one mixed-radix integer, which is what
    coverage needs and which DISCARDS the per-group structure. Anything asking whether the
    groups move INDEPENDENTLY (basin_coupling) needs the array before that collapse, so it
    is factored here rather than recomputed -- one definition of "which rotamer is this
    group in", used by both.
    """
    out = np.zeros((len(dof), len(groups)), dtype=np.int64)
    for gi, (rows_j, centres) in enumerate(groups):
        v = dof[:, n0 + rows_j[0]]
        dist = np.abs((v[:, None] - centres[None, :] + np.pi) % (2 * np.pi) - np.pi)
        out[:, gi] = np.argmin(dist, axis=1)
    return out


def basin_counts(groups, dof, n0: int, n_combos: int) -> np.ndarray:
    """Occupancy per rotamer basin, ``[n_combos]``. One definition, via the labeller."""
    return np.bincount(rotamer_basin_labels(groups, dof, n0), minlength=n_combos)


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
    ref = basin_reference(en, max_modes=max_modes, accessible_kt=accessible_kt)
    if 'skipped' in ref:
        return ref
    groups, combos, n0, e_mode, e_best, accessible = (
        ref['groups'], ref['combos'], ref['n0'], ref['mode_energies_raw'],
        ref['e_best'], ref['accessible'])
    T = float(en.temperature)

    # ---- where do prior draws land? ----
    rng = np.random.default_rng(seed)
    # coverage needs no density, so unlike prior_report it CAN run on a ring molecule --
    # and must therefore draw from the real ring path, not the disabled one.
    xs, stats = en.sample_prior_states(prior, n, rng, report=False)
    counts = basin_counts(groups, stats['dof'], n0, len(combos))
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


def is_log_z(en, prior, n: int = 20000, seed: int = 0, chunk: int = 4096) -> dict:
    """Importance-sampled ``log Z`` using the fitted prior as the proposal.

    THE POINT. A trained flow head reports a log Z, but nothing in the TB objective forces
    that number to be RIGHT -- a policy and a flow head can agree with each other at the
    wrong constant. This is an independent estimate that never touches the model, so
    ``log Z_learned - log Z_IS`` is a real error bar rather than a self-consistency check.
    ``brute_force_log_z`` is the exact answer but only at tiny d (64**30 at propanol/full),
    which is why this exists.

    THE MEASURE HAS TO MATCH OR THE ANSWER IS OFF BY A CONSTANT. ``log_reward`` is a density
    on the LATENT BOX x (it carries ``log|dq/dx|``), while ``prior_log_prob`` is a density on
    the internal coordinates q. Changing variables, ``log q_x(x) = log q_dof(q) +
    log|dq/dx|``, so the chart term is added to the proposal before subtracting. Getting
    this wrong shifts every estimate by exactly ``log_chart_jacobian`` -- a constant, which
    is precisely the kind of error that looks like a plausible number.

    ACYCLIC ONLY, by inheritance from ``prior_log_prob``: a ring block's density is a
    mixture that is singular in the held directions. It RAISES there rather than returning
    a number, and the caller is expected to publish an unavailable cell.

    Returns the estimate plus the diagnostics that say whether to believe it: ``ess_frac``
    and ``w_max_frac`` (one sample carrying most of the weight means the estimate is a
    lower bound in practice), and ``clip_frac``, which must be ~0 -- the box clamp in
    ``sample_prior_states`` puts finite mass exactly on the wall and no continuous density
    can express that.
    """
    rng = np.random.default_rng(seed)
    xs, stats = en.sample_prior_states(prior, n, rng, report=False)
    dof = stats['dof']
    log_q = np.asarray(en.prior_log_prob(prior, dof), dtype=np.float64)

    x = torch.as_tensor(xs, dtype=en.dtype, device=en.device)
    one = torch.tensor(1.0, dtype=en.dtype, device=en.device)
    log_r = []
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk]
        e = en.potential_energy(xb, one) + en.jacobian_energy(xb, one)
        log_r.append((-(e - en.log_chart_jacobian)).detach().cpu().numpy())
    log_r = np.concatenate(log_r).astype(np.float64)

    # log q on the BOX, not on the internal coordinates -- see docstring
    log_w = log_r - (log_q + float(en.log_chart_jacobian))
    m = log_w.max()
    log_z = m + np.log(np.exp(log_w - m).mean())

    w = np.exp(log_w - m)
    ess = float(w.sum() ** 2 / max(float((w ** 2).sum()), 1e-300))
    # clip_frac is a dict keyed by DoF class ({'r': .., 'theta': ..}); the WORST class is
    # what invalidates the weights, so reduce by max rather than averaging it away
    _clip = stats.get('clip_frac', 0.0) or 0.0
    clip = float(max(_clip.values())) if isinstance(_clip, dict) else float(_clip)
    return {
        'log_z': float(log_z), 'n': int(n), 'ess': ess, 'ess_frac': ess / n,
        'w_max_frac': float(w.max() / max(w.sum(), 1e-300)),
        'clip_frac': clip, 'log_w_std': float(log_w.std()),
    }
