"""Distributional, energetic and geometric eval statistics for the conformer route.

WHY THIS EXISTS. The ported protocol publishes ~227 metrics, but they are overwhelmingly
the TB family -- residuals, coverage of the importance weights, log Z parity. Those say the
OBJECTIVE is being optimised. They do not say the SAMPLES are right: a policy and a flow
head can agree with each other at the wrong constant, and every TB residual would fall
while the geometry drifted. Everything here is a function of the samples and the force
field ONLY, so none of it can be satisfied by the model agreeing with itself.

FOUR RULES THIS MODULE FOLLOWS, all of them lessons already paid for elsewhere in the repo:

  * ABSENT IS NOT ZERO. A quantity a molecule does not have (ring closure on an acyclic
    molecule, a correlation across one condition) is reported as an explicit
    ``*_available = 0`` plus no value, never as 0.0 or as a nan that averages into
    something. A metric that abstains silently passes exactly the case it exists to catch.
    The flag is published ONLY WHEN IT IS 0: a live metric is identified by its value
    being there, and a companion ``= 1`` beside every number was a second key per metric
    carrying no information. Absence is still labelled, which is the whole rule -- what is
    gone is the redundant announcement of presence. (``ring/``, ``ringtor/`` and
    ``corr_E/`` are deliberately left on the older both-ways convention; they abstain on
    every acyclic single-molecule run, which is the case the flag exists for.)
  * A DEGENERATE BAR IS LABELLED. Where a fraction is 1.0 BY CONSTRUCTION -- bond and
    angle ranges at `torsion`/`dihedral`, where r and theta are frozen at the reference --
    the value is suppressed and an ``*_frozen`` flag published instead. Publishing 1.0
    there is the same failure as a test that cannot fail.
  * ONE DEFINITION. Basin identity comes from ``prior_diagnostics.basin_reference``,
    closure from ``ring_metrics.closure_error``, DoF classes from
    ``ConformerTorsions._free_block``. Nothing here re-derives a second, incompatible one.
  * GROUPINGS ARE BOUNDED. Per-DoF and per-atom breakdowns explode with molecule size, so
    the groupings are by DoF CLASS (3) and by central-atom ELEMENT (<= 8) -- both bounded,
    and both meaning the same thing across different molecules, which per-index groupings
    do not.
"""
from __future__ import annotations

import numpy as np
import torch

#: MMFF terms, in the order ``intramolecular_energy(components=True)`` returns them.
ENERGY_TERMS = ('bond', 'angle', 'lj', 'torsion', 'stretch_bend', 'oop', 'electrostatic')

#: element -> symbol, for naming the per-element breakdown. Matches dof_features.ELEMENTS.
_SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'}

#: Energy terms that get the full quantile block; every other term gets its mean and its
#: share only. The distinction is between a term whose SHAPE is read -- the clash tail on
#: LJ, the angle strain that dominates a strained conformer -- and one whose contribution
#: is small enough that only its level matters.
#:
#: FIXED, NOT CHOSEN PER EVAL from the measured shares. A threshold on the live share
#: would make a term drifting across it appear and disappear from the charts, which is a
#: worse failure than logging it always. The choice is auditable rather than hidden
#: because ``{term}_share`` is still published for EVERY term including these two, so a
#: molecule whose spectrum puts the mass somewhere else shows up here as a large share on
#: a term with no histogram -- which is the signal to change this tuple.
FULL_QUANTILE_TERMS = ('lj', 'angle')


def _host(t, dtype=None):
    """-> numpy on the HOST. Every public entry point funnels its inputs through this.

    Buffers live on `buffer_device`, which is 'cuda' in the canonical config, so any of
    these arguments can arrive as a CUDA tensor -- and `np.asarray` on one raises rather
    than copying. Centralised so a new metric cannot reintroduce the same failure.
    """
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    return a.astype(dtype) if dtype is not None else a


def _mean_only(v, prefix, out):
    """The mean alone, for a term outside ``FULL_QUANTILE_TERMS``.

    Abstains explicitly on an empty or all-nonfinite array rather than publishing a nan,
    on the same rule as ``_quantiles`` -- a small term is still a term that can go absent.
    """
    v = _host(v, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        out[f'{prefix}_available'] = 0
        return out
    out[f'{prefix}_mean'] = float(v.mean())
    return out


def _quantiles(v, prefix, out, hist=True):
    """mean / p10 / p50 / p90 / max, plus the raw array so wandb draws the histogram."""
    v = _host(v, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        out[f'{prefix}_available'] = 0
        return out
    out[f'{prefix}_mean'] = float(v.mean())
    out[f'{prefix}_p10'] = float(np.percentile(v, 10))
    out[f'{prefix}_p50'] = float(np.percentile(v, 50))
    out[f'{prefix}_p90'] = float(np.percentile(v, 90))
    out[f'{prefix}_max'] = float(v.max())
    if hist:
        out[prefix] = v.astype(np.float32)
    return out


# --------------------------------------------------------------------- energy


def energy_components(en, x, chunk: int = 8192) -> dict:
    """Per-term MMFF energies, ``{term: [B]}``, at T = 1 in kcal/mol. RAW -- UNCLIPPED.

    Reuses ``intramolecular_energy(components=True)`` rather than re-summing the terms. The
    box wall is NOT a component -- it is not part of the force field, it is the chart's
    boundary condition -- so it is returned separately as 'wall'.

    ⚠ WITH ``energy_clip`` SET THESE DO NOT SUM TO WHAT THE TRAINER OPTIMISES, and they
    cannot: the clip acts on the SUM (potential_energy), and a clipped total has no
    per-term decomposition to hand back. Measured on tetraglycine at clip 250, the
    component sum runs to 1.4e7 while potential_energy tops out at 266, disagreeing on 217
    of 400 rows. That is deliberate and useful -- the raw LJ is how you SEE the clash level
    the clip is hiding -- but read these as diagnostics of the geometry, never as the
    objective. ``energy_component_stats`` publishes 'E/clipped_total_*' and
    'E/clip_active_frac' alongside so the difference is visible rather than implied.
    """
    from mxtaltools.conformers.energy import intramolecular_energy

    x = torch.as_tensor(x, dtype=en.dtype, device=en.device)
    one = torch.tensor(1.0, dtype=en.dtype, device=en.device)
    acc = {}
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            xb = x[i:i + chunk]
            tree, ff = en._batch(xb.shape[0])
            _, comp = intramolecular_energy(tree, en.build_positions(xb), ff, components=True)
            if en._lin_free_idx.numel():
                comp = dict(comp)
                comp['wall'] = en.bounding_energy(xb, one)
            for k, v in comp.items():
                acc.setdefault(k, []).append(_host(v))
    return {k: np.concatenate(v) for k, v in acc.items()}


def energy_component_stats(en, x, prefix: str = 'E/') -> dict:
    """Histogram + quantiles for the total and the major components; means for the rest.

    Which terms are 'major' is ``FULL_QUANTILE_TERMS``, a fixed tuple -- see its comment
    for why the set is not derived from the live shares. Every term keeps its mean and
    its share, so the partition over terms is still complete and a shift in which term
    carries the energy is still visible; what the minor terms lose is their percentile
    block and their histogram.

    When ``energy_clip`` is active the components are RAW and do not sum to the optimised
    potential (see energy_components), so the clipped total is published beside them along
    with the fraction of samples the clip actually bit on. Without that, a rising raw
    'E/lj_mean' reads as the objective degrading when the objective is bounded.
    """
    comp = energy_components(en, x)
    out = {}
    total = np.zeros(len(next(iter(comp.values()))))
    for name, v in comp.items():
        if name in FULL_QUANTILE_TERMS:
            _quantiles(v, f'{prefix}{name}', out)
        else:
            _mean_only(v, f'{prefix}{name}', out)
        total = total + v
    _quantiles(total, f'{prefix}total', out)
    # the share each term carries, so a component that starts dominating is visible without
    # reading seven histograms side by side. Normalised by the sum of the term MAGNITUDES,
    # not by |total|: the terms cancel (LJ negative against bonded positive), so dividing by
    # the total gives shares that sum well past 1 and read as nonsense.
    mags = {name: float(np.abs(v).mean()) for name, v in comp.items()}
    denom = sum(mags.values()) or 1.0
    for name, m in mags.items():
        out[f'{prefix}{name}_share'] = m / denom
    # THE OPTIMISED POTENTIAL, beside the raw sum above. Identical when energy_clip is off;
    # when it is on, `total` is the uncompressed sum (which can reach 1e7) while this is
    # what the loss actually sees (bounded near the cutoff). Publishing only the first
    # makes a bounded objective look like a diverging one.
    if getattr(en, 'energy_clip', None) is not None:
        one = torch.tensor(1.0, dtype=en.dtype, device=en.device)
        with torch.no_grad():
            clipped = _host(en.potential_energy(
                torch.as_tensor(x, dtype=en.dtype, device=en.device), one), np.float64)
        _quantiles(clipped, f'{prefix}clipped_total', out)
        out[f'{prefix}clip_active_frac'] = float((total > en.energy_clip).mean())
        out[f'{prefix}energy_clip'] = float(en.energy_clip)
    return out


def energy_vs_reference(sample_e, reference_e, prefix: str = 'E/') -> dict:
    """Is the sampler producing BETTER energies than its reference population?

    ``frac_below_ref_median`` is the headline: 0.5 means indistinguishable from the
    reference, 1.0 means every sample beats its median. This is the one number that says
    training bought something in energy terms rather than in residual terms.
    """
    s, r = _host(sample_e, np.float64), _host(reference_e, np.float64)
    s, r = s[np.isfinite(s)], r[np.isfinite(r)]
    if s.size == 0 or r.size == 0:
        return {f'{prefix}vs_ref_available': 0}
    ref_med = float(np.median(r))
    return {
        f'{prefix}ref_median': ref_med,
        f'{prefix}frac_below_ref_median': float((s < ref_med).mean()),
        f'{prefix}median_gain_vs_ref': ref_med - float(np.median(s)),
        f'{prefix}min_gain_vs_ref': float(r.min()) - float(s.min()),
    }


def energy_marginal_overlap(sample_e, reference_e, temperature: float = 1.0,
                            reps: int = 4, cache: dict = None, prefix: str = 'E/') -> dict:
    """Does the sampler's ENERGY distribution match the reference's? Not "is it better".

    WHY THIS IS NOT ``energy_vs_reference``. That one is one-sided by design -- it asks
    whether training bought lower energies, and ``frac_below_ref_median`` reads 0.5 for a
    sampler that is indistinguishable from the reference. But it reads 0.5 just as happily
    for a sampler with the RIGHT median and TWICE the spread, which is a badly wrong
    distribution scoring the converged value. Overlap needs a two-sided statistic.

    WHY IT IS NOT REDUNDANT WITH THE PER-COLUMN MARGINALS. Energy is a scalar function of
    all coordinates jointly, so it is sensitive to CORRELATIONS that no 1-D marginal can
    see: a sampler can reproduce every column's marginal exactly and still assemble them
    into combinations the target never visits, which shows up here and nowhere else. The
    two checks are complementary, and on a molecule with real coupling neither substitutes
    for the other.

    THREE READINGS, because one number cannot carry both questions:
      w1_kT     the displacement in units of kT -- an EFFECT SIZE. Physical, n-independent,
                and the number to answer "is this close".
      w1_ratio  the same W1 over its own two-sample floor -- a SIGNIFICANCE measure, 1.0 =
                indistinguishable at this n. Falls as ~n^-1/2, so it tightens with sample
                size and must not be gated on without pinning n.
      overlap   the literal shared area of the two densities, in [0, 1], reported beside
                the floor a perfect sampler would score -- which is NOT 1.0 at finite n,
                and reads lower the more bins are used, so the bare value means nothing
                without its floor next to it.
    """
    s = _host(sample_e, np.float64)
    r = _host(reference_e, np.float64)
    s, r = s[np.isfinite(s)], r[np.isfinite(r)]
    n = int(s.size)
    if n < 32 or r.size < 4 * n:
        return {}

    def _w1(a, b):
        q = np.linspace(0.0, 1.0, 512)
        return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())

    def _ovl(a, b, lo, hi, bins=64):
        pa, _ = np.histogram(a, bins=bins, range=(lo, hi), density=False)
        pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
        pa = pa / max(pa.sum(), 1)
        pb = pb / max(pb.sum(), 1)
        return float(np.minimum(pa, pb).sum())

    if cache is None:
        cache = {}
    key = ('e', n)
    got = cache.get(key)
    if got is None:
        rng = np.random.default_rng(0)
        fw, fo = [], []
        for _ in range(reps):
            idx = rng.permutation(r.size)
            a, b = r[idx[:n]], r[idx[n:]]
            fw.append(_w1(a, b))
            lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
            fo.append(_ovl(a, b, lo, hi))
        got = (float(np.median(fw)), float(np.median(fo)))
        cache[key] = got
    floor_w1, floor_ovl = got

    w1 = _w1(s, r)
    lo, hi = float(min(s.min(), r.min())), float(max(s.max(), r.max()))
    ovl = _ovl(s, r, lo, hi)
    kT = float(temperature) if float(temperature) > 0 else 1.0
    return {
        f'{prefix}emarg_w1_kT': w1 / kT,
        f'{prefix}emarg_w1_ratio': w1 / max(floor_w1, 1e-12),
        f'{prefix}emarg_w1_floor': floor_w1,
        f'{prefix}emarg_overlap': ovl,
        f'{prefix}emarg_overlap_floor': floor_ovl,
        # overlap normalised by what perfection scores: 1.0 = as overlapping as two draws
        # of the target are with each other
        f'{prefix}emarg_overlap_rel': ovl / max(floor_ovl, 1e-12),
        f'{prefix}emarg_n': float(n),
    }


def thermal_stats(en, energies, e_min: float, prefix: str = 'E/') -> dict:
    """Excess over the tier's own minimum, and ``T_eff/T`` built from it.

    ``T_eff = 1 + 2 * median_excess / d`` is the repo's existing definition
    (prior_baselines). Its known degeneracy -- k cancelling between draw width and score --
    applies to a RAW PRIOR draw with free r/theta, NOT to a trained policy's samples, which
    is why it is meaningful here and marked `*deg` there.

    ``e_min`` must be the multi-start local minimum for THIS tier (prior_baselines
    .tier_minimum). It is an upper bound on the true minimum, so every excess here is a
    lower bound -- a uniform shift, which leaves comparisons within a run intact.
    """
    e = _host(energies, np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0 or not np.isfinite(e_min):
        return {f'{prefix}excess_available': 0}
    T = float(en.temperature)
    excess = (e - e_min) / T
    out = {f'{prefix}e_min_reference': float(e_min)}
    _quantiles(excess, f'{prefix}excess_kt', out)
    med = float(np.median(excess))
    out[f'{prefix}T_eff_over_T'] = 1.0 + 2.0 * med / max(int(en.ndim), 1)
    # equipartition puts <E - E_min> at d/2 kT for a harmonic well; below it means the
    # sampler is COLDER than the target, which over-optimisation looks like
    out[f'{prefix}frac_within_equipartition'] = float((excess - en.ndim / 2.0 <= 0).mean())
    return out


# ------------------------------------------------------------------ geometry


def _dof_class_columns(en):
    """State columns per DoF class, as ``{class_name: index array}``."""
    block = _host(en._free_block)
    return {'r': np.flatnonzero(block == 0),
            'theta': np.flatnonzero(block == 1),
            'phi': np.flatnonzero(block == 2)}


def geometry_stats(en, x, prefix: str = 'geom/') -> dict:
    """Are bond lengths and angles inside their physically reasonable window?

    THE WINDOW IS THE CHART'S OWN, not a second opinion: the state is a displacement from
    the force field's reference in units of ``delta_r_max`` / ``delta_theta_max``, so
    |x| <= 1 on a linear column IS "this bond is within delta_r_max of its equilibrium".
    Re-deriving a chemical range in angstroms here would be a second, disagreeing
    definition of the same thing.

    THREE KEYS, AND THAT IS THE WHOLE GROUP. This is a GUARD, not a distribution, so it
    has exactly two questions to answer:
      * ``all_in_range`` -- DID IT FIRE. Fraction of MOLECULES with every bond AND angle
        in range. The harsh reading, and the one that decides whether a sample is usable:
        one bad bond in 11 makes the whole conformer wrong.
      * ``*_worst_abs``  -- HOW CLOSE IS IT to firing. The worst |x| anywhere in the batch,
        against a limit of 1.
    The per-column in-range fractions and the per-molecule worst-case quantile blocks that
    used to sit here were eighteen further keys, and on a tier where the sampler stays an
    order of magnitude inside the window they could only restate a guard that has never
    fired. ``all_in_range`` already carries the per-molecule tail this group needs.

    FROZEN TIERS ARE LABELLED, NOT SCORED. At `torsion` and `dihedral` the r and theta
    blocks are held at the reference, so a worst-case excursion is 0 by construction. That
    is published as ``*_frozen = 1`` INSTEAD OF the value, never beside it, because a 0
    that cannot be anything else is not evidence.
    """
    x = _host(x)
    cols = _dof_class_columns(en)
    lin_free = set(_host(en._lin_free_idx).tolist())
    out = {}
    ok_all = np.ones(x.shape[0], dtype=bool)
    any_scored = False
    for name in ('r', 'theta'):
        idx = np.array([c for c in cols[name] if c in lin_free], dtype=int)
        if idx.size == 0:
            out[f'{prefix}{name}_frozen'] = 1
            continue
        any_scored = True
        a = np.abs(x[:, idx])
        out[f'{prefix}{name}_worst_abs'] = float(a.max())
        ok_all &= (a <= 1.0).all(axis=1)
    if any_scored:
        out[f'{prefix}all_in_range'] = float(ok_all.mean())
    else:
        # every linear block frozen -> the question is not askable at this tier
        out[f'{prefix}all_in_range_frozen'] = 1
    return out


def dof_class_stats(en, x, reference=None, prefix: str = 'dof/') -> dict:
    """Per-DoF-class spread, wall-piling and drift against a reference population.

    Three named pathologies, each as a scalar, per the agreed scalars-first framing:
      * VARIANCE EXPLOSION -- ``sd_ratio`` against the reference, max over columns in the
        class. The max, not the mean: one column blowing up is the failure, and averaging
        over 11 healthy columns hides it.
      * WALL PILING -- mass within 1% of the box edge, on LINEAR columns only. The phi
        block wraps, so there is no wall to pile against and the number would be noise.
      * DRIFT -- |mean shift| in reference sd units, max over columns.
    """
    x = _host(x)
    ref = _host(reference) if reference is not None else None
    cols = _dof_class_columns(en)
    periodic = _host(en.periodic_dims).astype(bool)
    out = {}
    for name, idx in cols.items():
        if idx.size == 0:
            out[f'{prefix}{name}_available'] = 0
            continue
        sub = x[:, idx]
        # the pooled distribution over every column of the class, as a histogram. Pooled
        # rather than per-column on purpose: per-column is d histograms (30 at
        # propanol/full) and unreadable, and the per-column view is what the DoF-class
        # figure is for. phi is published in DEGREES, r/theta in box units -- they are not
        # the same kind of quantity and a shared axis would be meaningless.
        scale = 180.0 if periodic[idx].all() else 1.0
        out[f'{prefix}{name}_hist'] = (scale * sub.reshape(-1)).astype(np.float32)
        out[f'{prefix}{name}_sd_mean'] = float(sub.std(axis=0).mean())
        if not periodic[idx].any():
            out[f'{prefix}{name}_wall_mass'] = float((np.abs(sub) >= 0.99).mean())
        if ref is not None and ref.shape[1] == x.shape[1]:
            rsd = ref[:, idx].std(axis=0)
            good = rsd > 1e-9
            if good.any():
                out[f'{prefix}{name}_sd_ratio_max'] = float(
                    (sub.std(axis=0)[good] / rsd[good]).max())
                out[f'{prefix}{name}_drift_max'] = float(
                    (np.abs(sub.mean(axis=0) - ref[:, idx].mean(axis=0))[good]
                     / rsd[good]).max())
    return out


def _central_elements(en):
    """Element of the atom that OWNS each state column, in SPEC numbering.

    Bond -> the heavier of its two atoms; angle -> its vertex; torsion -> the central bond's
    first atom. Grouping by ELEMENT rather than by atom index is deliberate: atom index
    explodes with molecule size and means nothing across two different molecules, while
    element is bounded at 8 and is directly comparable.
    """
    spec = en.spec
    z = np.asarray(spec.z)
    bi, ai, ti = (np.asarray(spec.bond_index), np.asarray(spec.angle_index),
                  np.asarray(spec.torsion_index))
    per_class = {
        0: (np.maximum(z[bi[:, 0]], z[bi[:, 1]]) if len(bi) else np.zeros(0, int)),
        1: (z[ai[:, 1]] if len(ai) else np.zeros(0, int)),
        2: (z[ti[:, 1]] if len(ti) else np.zeros(0, int)),
    }
    block = _host(en._free_block)
    keep = _host(en._keep_idx) if hasattr(en, '_keep_idx') else None
    owner = np.zeros(block.shape[0], dtype=int)
    for cls in (0, 1, 2):
        cols = np.flatnonzero(block == cls)
        src = per_class[cls]
        if cols.size == 0 or src.size == 0:
            continue
        # columns of a class appear in the spec's own order, so the j-th free column of a
        # class is the j-th entry of that class's index table when nothing was dropped
        take = (keep[cols] if keep is not None and keep.shape[0] == block.shape[0]
                else np.arange(cols.size))
        owner[cols] = src[np.clip(take[:cols.size] if take.size >= cols.size
                                  else np.arange(cols.size), 0, src.size - 1)]
    return owner


def dof_element_stats(en, x, reference=None, prefix: str = 'dof_elem/') -> dict:
    """Spread per (DoF class, central-atom element). Bounded at 3 x 8 groups.

    THE DRILL-DOWN FOR ``dof/*_sd_ratio_max``. That number says one column of a class
    carries k times the prior's spread; this says which ELEMENT owns it, and it does so
    without a per-atom breakdown that grows with the molecule.

    A RATIO, NOT A RAW SPREAD, whenever a reference population is given. A bare
    ``r_N_sd = 0.010`` cannot be judged right or wrong: there is no bar for it, and a
    dozen such numbers side by side are unreadable -- which is what made this group
    useless for the drill-down it exists to serve. Against the prior's own spread on the
    SAME columns, 1.0 means "matches the prior" on every group of every molecule.

    MAX over the group's columns, matching ``dof_class_stats``. One column blowing up is
    the failure; a mean over the group's healthy columns is exactly what hides it.

    The raw sd is published only where there is no reference to divide by, and under a
    DIFFERENT key name, so a ratio and an unreferenced spread can never land on one axis.
    """
    x = _host(x)
    ref = _host(reference) if reference is not None else None
    if ref is not None and (ref.ndim != 2 or ref.shape[1] != x.shape[1]):
        ref = None
    block = _host(en._free_block)
    try:
        owner = _central_elements(en)
    except Exception:
        return {f'{prefix}available': 0}
    out = {}
    for cls, cname in ((0, 'r'), (1, 'theta'), (2, 'phi')):
        for zval in np.unique(owner):
            idx = np.flatnonzero((block == cls) & (owner == zval))
            if idx.size == 0:
                continue
            sym = _SYM.get(int(zval), f'Z{int(zval)}')
            sd = x[:, idx].std(axis=0)
            out[f'{prefix}{cname}_{sym}_n'] = int(idx.size)
            rsd = ref[:, idx].std(axis=0) if ref is not None else None
            good = rsd > 1e-9 if rsd is not None else None
            if good is not None and bool(np.any(good)):
                out[f'{prefix}{cname}_{sym}_sd_ratio'] = float((sd[good] / rsd[good]).max())
            else:
                out[f'{prefix}{cname}_{sym}_sd'] = float(sd.mean())
    return out


# ---------------------------------------------------------------------- rings


def ring_stats(en, x, prefix: str = 'ring/') -> dict:
    """Ring closure error. UNAVAILABLE, explicitly, on an acyclic molecule.

    Delegates to ``ring_metrics.closure_error``, which is the sampler's own monitor, so
    this cannot drift from the number the prior draw reports. An acyclic molecule has no
    closure bond and gets ``available = 0`` rather than 0.0 -- a zero closure error and a
    molecule with nothing to close are opposite readings and must not share a value.
    """
    from energies.ring_metrics import closure_error

    x = torch.as_tensor(x, dtype=en.dtype, device=en.device)
    ang, sigma, n_bonds = closure_error(en, x)
    if not n_bonds:
        return {f'{prefix}available': 0, f'{prefix}n_closure_bonds': 0}
    # closure_error counts closure bonds over the BATCHED force field, so its third return
    # is n_bonds x n_molecules. Divided here rather than in closure_error, which the prior
    # benchmark already reports and whose numbers must not move.
    per_mol = int(round(n_bonds / max(int(x.shape[0]), 1)))
    return {f'{prefix}available': 1, f'{prefix}n_closure_bonds': per_mol,
            f'{prefix}closure_err_a': float(ang), f'{prefix}closure_err_sigma': float(sigma)}


# ------------------------------------------------------------------- coverage


def basin_coverage(en, x, basin_ref, prefix: str = 'cover/') -> dict:
    """Which of the target's accessible rotamer basins does the SAMPLER reach?

    This is the question ESS structurally cannot answer: a basin never proposed contributes
    no large weight and no warning, so the importance-weight diagnostics look healthiest
    exactly where the sampler is broken. Coverage has to be measured in the reverse
    direction, against a basin set enumerated from the target -- which is what `basin_ref`
    (prior_diagnostics.basin_reference) is.

    KNOWN FALSE-PASS, and it is not fixed here: basins are the product of per-group rotamer
    centres, so on a molecule whose coordinates are COUPLED the product over-counts
    reachable combinations and the metric can report full coverage for a sampler that never
    reaches a genuinely distinct conformer. Read `n_missed` as a lower bound on what is
    missing, never as proof nothing is.

    THREE KEYS, and the denominators they are read against live in ``cover_constants``.
    ``n_modes``, ``n_accessible`` and the uniform-occupancy expectation do not move within
    a run -- they are fixed by the molecule -- so logging them every eval produced flat
    traces, while ``missed_frac`` was ``n_missed`` over one of them. They still have to be
    RECORDED, because ``n_missed = 0`` says nothing without knowing how many basins were on
    the table, so they are printed once at startup instead of deleted.
    """
    from energies.prior_diagnostics import basin_counts

    if basin_ref is None or 'skipped' in basin_ref:
        why = (basin_ref or {}).get('skipped', 'no basin reference')
        return {f'{prefix}available': 0, f'{prefix}skipped': why}
    x_t = torch.as_tensor(x, dtype=en.dtype, device=en.device)
    r, th, ph = en.dof_from_state(x_t)
    dof = np.concatenate([_host(r), _host(th), _host(ph)], axis=1)
    combos, groups, n0 = basin_ref['combos'], basin_ref['groups'], basin_ref['n0']
    counts = basin_counts(groups, dof, n0, len(combos))
    acc = np.asarray(basin_ref['accessible'], dtype=bool)
    acc_idx = np.flatnonzero(acc)
    n = int(counts.sum())
    if acc_idx.size == 0 or n == 0:
        return {f'{prefix}available': 0}
    frac = counts / n
    res = {
        f'{prefix}n_missed': int((counts[acc_idx] == 0).sum()),
        f'{prefix}worst_frac': float(frac[acc_idx].min()),
    }
    # occupancy entropy over accessible basins, normalised: 1 = uniform over them, -> 0 =
    # collapsed onto one. Mode collapse as a scalar -- but ONLY defined with something to
    # collapse from. A molecule with one accessible basin would score 0, i.e. maximally
    # collapsed, when it is in fact fully covered; that is a false alarm, so it abstains.
    if acc_idx.size >= 2:
        res[f'{prefix}occupancy_entropy'] = _norm_entropy(frac[acc_idx])
    else:
        res[f'{prefix}occupancy_entropy_available'] = 0
    return res


def _entropy_nats(counts) -> float:
    """Plug-in Shannon entropy in NATS. Deliberately NOT normalised.

    ``_norm_entropy`` divides by log(k), which is right for "how spread is this" but
    destroys ADDITIVITY -- and total correlation is a difference of entropies, so
    normalising each term separately makes the difference meaningless.
    """
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * np.log(p)).sum())


def _mixed_radix(L, sizes):
    """Per-group labels -> one basin index. Group g-1 is the LEAST significant digit,
    which is what makes this agree with basin_reference's itertools.product ordering."""
    lab, stride = np.zeros(len(L), dtype=np.int64), 1
    for gi in range(L.shape[1] - 1, -1, -1):
        lab += stride * L[:, gi]
        stride *= sizes[gi]
    return lab


def _total_correlation(L, sizes, n_combos) -> float:
    """sum_i H(group_i) - H(joint), in nats. 0 iff the groups are independent."""
    h_marg = sum(_entropy_nats(np.bincount(L[:, i], minlength=sizes[i]))
                 for i in range(L.shape[1]))
    joint = np.bincount(_mixed_radix(L, sizes), minlength=n_combos)
    return h_marg - _entropy_nats(joint)


def _circ_mean_deg(a_deg):
    r = np.radians(np.asarray(a_deg, dtype=np.float64))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def _circ_sd_deg(a_deg):
    """Circular standard deviation, degrees. Linear sd on wrapped angles is meaningless --
    a distribution straddling +/-180 would report a huge spread while being tight."""
    r = np.radians(np.asarray(a_deg, dtype=np.float64))
    R = np.abs(np.exp(1j * r).mean())
    return float(np.degrees(np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0)))))


def _ring_corr(t_deg, k):
    """Correlation among a cycle's torsions, sin/cos embedded first.

    A linear correlation on wrapped angles is not meaningful, so the angles are embedded
    before correlating and the sin-sin block is taken as the coupling structure.
    """
    r = np.radians(np.asarray(t_deg, dtype=np.float64))
    z = np.concatenate([np.sin(r), np.cos(r)], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        c = np.corrcoef(z.T)
    c = np.nan_to_num(c)
    return c[:k, :k]


def ring_torsion_stats(en, x, reference=None, prefix='ringtor/', cycles=None) -> dict:
    """Per ring cycle: is the sampler's RING TORSION distribution the prior's?

    WHY THIS AND NOT THE POOLED DoF HISTOGRAM. `dof/phi_hist` pools every torsion column
    into one distribution, so two rings failing in OPPOSITE directions -- one too wide,
    one collapsed -- average into a single blob that looks mildly wrong. Measured per
    cycle they separate immediately, which is exactly how the phenyl-tetrahydropyran
    split was found (2026-08-20): the aromatic ring uniformly 3.6x too WIDE with its
    correlation structure intact, the saturated ring NARROWER than the prior with shifted
    means and degraded correlations.

    Measured in the ring's own torsion space (ring_metrics.ring_torsions, read off the
    built geometry) rather than in state columns, so it does not depend on which state
    column happens to drive which ring dihedral -- and it is therefore comparable across
    molecules whose charts differ.

    THREE NUMBERS PER CYCLE, because width and structure fail independently:
      ``sd_ratio_max``   widest torsion relative to the reference. > 1 too broad, < 1
                         collapsed. Both are failures and the direction matters.
      ``mean_shift_max`` largest circular mean displacement, degrees -- catches a ring
                         sitting in the wrong pucker rather than the wrong width.
      ``corr_dist``      ||C_sampler - C_reference||_F / ||C_reference||_F. Ring closure is
                         a property of the JOINT, so a ring can match every marginal and
                         still never close; this is the term that sees that.

    Without a reference only the sampler's own spreads are published -- a labelled
    half-measurement rather than a ratio against nothing.
    """
    from energies.ring_metrics import ring_cycles, ring_torsions

    if cycles is None:
        cycles = ring_cycles(en)
    if not cycles:
        return {f'{prefix}available': 0, f'{prefix}n_cycles': 0}

    x_t = torch.as_tensor(_host(x), dtype=en.dtype, device=en.device)
    deg = 180.0 / np.pi
    ts = ring_torsions(en, x_t, cycles)
    tr = None
    if reference is not None:
        ref = _host(reference)
        if ref.shape[1] == x_t.shape[1] and len(ref) >= 2:
            # matched n: a correlation matrix estimated from a different sample size is
            # not comparable to one estimated from this batch
            ref = ref[:len(x_t)] if len(ref) >= len(x_t) else ref
            tr = ring_torsions(en, torch.as_tensor(ref, dtype=en.dtype, device=en.device),
                               cycles)

    out = {f'{prefix}available': 1, f'{prefix}n_cycles': len(cycles)}
    for ci, cyc in enumerate(cycles):
        k = len(cyc)
        a = np.asarray(ts[ci]) * deg
        tag = f'{prefix}c{ci}'
        out[f'{tag}_size'] = k
        sd_s = np.array([_circ_sd_deg(a[:, j]) for j in range(k)])
        out[f'{tag}_sd_max_deg'] = float(sd_s.max())
        out[f'{tag}_sd_med_deg'] = float(np.median(sd_s))
        out[f'{tag}_hist'] = a.reshape(-1).astype(np.float32)
        if tr is None:
            out[f'{tag}_ref_available'] = 0
            continue
        b = np.asarray(tr[ci]) * deg
        sd_r = np.array([_circ_sd_deg(b[:, j]) for j in range(k)])
        good = sd_r > 1e-6
        out[f'{tag}_ref_available'] = 1
        out[f'{tag}_ref_sd_med_deg'] = float(np.median(sd_r))
        if good.any():
            ratio = sd_s[good] / sd_r[good]
            out[f'{tag}_sd_ratio_max'] = float(ratio.max())
            out[f'{tag}_sd_ratio_min'] = float(ratio.min())
        shift = np.array([abs(((_circ_mean_deg(a[:, j]) - _circ_mean_deg(b[:, j]) + 180.0)
                               % 360.0) - 180.0) for j in range(k)])
        out[f'{tag}_mean_shift_max_deg'] = float(shift.max())
        cs, cr = _ring_corr(a, k), _ring_corr(b, k)
        nr = np.linalg.norm(cr)
        out[f'{tag}_corr_dist'] = float(np.linalg.norm(cs - cr) / nr) if nr > 1e-9 else 0.0
        off = ~np.eye(k, dtype=bool)
        out[f'{tag}_corr_absmean'] = float(np.abs(cs[off]).mean())
        out[f'{tag}_ref_corr_absmean'] = float(np.abs(cr[off]).mean())
    return out


def basin_coupling(en, x, basin_ref, target_tc=None, n_null: int = 8, seed: int = 0,
                   prefix: str = 'cover/') -> dict:
    """Do the rotamer groups move INDEPENDENTLY, or are specific combinations missing?

    THIS IS THE QUALIFIER ON ``basin_coverage``. That metric's documented false pass is on
    molecules whose coordinates are COUPLED -- the basin set is a product over per-group
    centres, so if the groups are not independent the product over-counts genuinely
    reachable conformers and coverage can read full while real states are unreachable.
    Coupling is exactly the condition under which that happens, so measuring it is what
    makes the coverage number interpretable rather than merely reassuring.

    Reported as total correlation, ``sum_i H(group_i) - H(joint)``, in nats: 0 means the
    per-group marginals combine independently.

    THE FIRST-ORDER KEY IS ``coupling_n_suppressed`` -- joint combinations with ZERO
    samples that the marginals predict should be populated. Total correlation is a
    magnitude with no direction; the suppressed count is the alarm.

    MODE COLLAPSE READS AS ZERO COUPLING, and that is the severe failure mode. Collapse
    onto one basin makes every marginal a delta: all entropies vanish, TC is 0, and the
    metric announces "the marginals are trustworthy" precisely when the sampler is most
    broken. Two mitigations, both required and both present: it ABSTAINS unless at least
    two groups are non-degenerate, and it is emitted from the same call block as
    ``n_missed`` / ``occupancy_entropy`` so the pair is never read apart.

    SMALL n MANUFACTURES COUPLING: plug-in entropy is biased low, and the joint has far
    more bins than any marginal, so its bias is larger and TC is biased UP. ``tc_null`` is
    TC on column-shuffled labels -- destroys the coupling, preserves every marginal and n --
    and ``tc_debiased``, the only one published, is the number to read. Same null/debiased
    convention the route already uses for wass.

    THE TARGET'S OWN COUPLING is a run constant and lives in ``cover_constants``. A
    ``tc_debiased`` above it means the policy learned a DIFFERENT dependence structure than
    the target has -- which every per-column statistic in ``dof_class_stats`` is blind to
    by construction, since the marginals can all match while the joint is wrong.
    """
    from energies.prior_diagnostics import rotamer_group_labels

    if basin_ref is None or 'skipped' in basin_ref:
        return {f'{prefix}coupling_available': 0}
    groups, combos, n0 = basin_ref['groups'], basin_ref['combos'], basin_ref['n0']
    if len(groups) < 2:
        return {f'{prefix}coupling_available': 0, f'{prefix}coupling_n_groups': len(groups)}

    x_t = torch.as_tensor(x, dtype=en.dtype, device=en.device)
    r, th, ph = en.dof_from_state(x_t)
    dof = np.concatenate([_host(r), _host(th), _host(ph)], axis=1)
    L = rotamer_group_labels(groups, dof, n0)
    sizes = [len(c) for _, c in groups]
    n = len(L)

    non_degenerate = sum(1 for i in range(L.shape[1]) if len(np.unique(L[:, i])) >= 2)
    if non_degenerate < 2:
        # one moving group is not a joint distribution. Abstaining rather than publishing
        # TC = 0, which would read as "independent, marginals trustworthy" for a collapsed
        # sampler -- the exact false pass this metric exists to prevent.
        return {f'{prefix}coupling_available': 0,
                f'{prefix}coupling_n_nondegenerate': int(non_degenerate)}

    tc = _total_correlation(L, sizes, len(combos))
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(max(int(n_null), 1)):
        S = np.column_stack([rng.permutation(L[:, i]) for i in range(L.shape[1])])
        nulls.append(_total_correlation(S, sizes, len(combos)))
    tc_null = float(np.mean(nulls))

    # combos the marginals say should be populated but which have NO samples
    marg = [np.bincount(L[:, i], minlength=sizes[i]) / n for i in range(L.shape[1])]
    pred = np.array([np.prod([marg[i][c[i]] for i in range(len(sizes))]) for c in combos])
    seen = np.bincount(_mixed_radix(L, sizes), minlength=len(combos))
    suppressed = int(((seen == 0) & (pred * n >= 10.0)).sum())

    # TWO KEYS. `tc` and `tc_null` are the halves of `tc_debiased`, which is the one that
    # is READ (raw TC is biased up and is not comparable across n); `tc_norm` rescales the
    # same quantity a second way; and `tc_gap` is `tc_debiased` minus a run constant, so it
    # is the same trace shifted. The target's own coupling moves to `cover_constants`,
    # which is where the subtraction can be done by eye.
    out = {
        f'{prefix}coupling_tc_debiased': float(tc - tc_null),
        f'{prefix}coupling_n_suppressed': suppressed,
    }
    return out


def target_coupling(basin_ref) -> float:
    """Total correlation of the TARGET's own rotamer landscape. Sampler-independent.

    Built from ``basin_reference``'s mode energies as a Boltzmann weight over combos, so it
    costs nothing extra and answers "is this molecule coupled at all" -- the number that
    says whether coverage was ever trustworthy on this system.

    HARMONIC-MODE APPROXIMATION. Each combo is realised with everything else at the
    reference geometry, so this understates entropic and steric coupling that only appears
    off-reference. It is a floor on the true coupling, not a measurement of it.
    """
    if basin_ref is None or 'skipped' in basin_ref:
        return float('nan')
    groups, combos = basin_ref['groups'], basin_ref['combos']
    if len(groups) < 2:
        return float('nan')
    e = np.asarray(basin_ref['mode_energies'], dtype=np.float64)
    w = np.exp(-(e - e.min()))
    w = w / w.sum()
    sizes = [len(c) for _, c in groups]
    h_marg = 0.0
    for gi in range(len(sizes)):
        pm = np.zeros(sizes[gi])
        for ci, combo in enumerate(combos):
            pm[combo[gi]] += w[ci]
        h_marg += _entropy_nats(pm)
    return float(h_marg - _entropy_nats(w))


def basin_nonthermal(en, x, energies, e_min, basin_ref, u_star, prefix: str = 'cover/') -> dict:
    """The non-thermal tail, grouped by ROTAMER BASIN instead of by condition.

    train.py's per-condition non-thermal family is correctly ABSENT on this route: it
    groups on condition_id, and one molecule means one condition, so its k >= 2 guard
    abstains. The question still transfers -- "is the bad tail concentrated somewhere, or
    spread evenly" -- and the one axis that genuinely partitions a single-molecule batch is
    the rotamer basin. Same reduction, different grouping label.

    READ THIS BESIDE ``n_missed``, NEVER ALONE. A basin the sampler ABANDONS contributes no
    samples and therefore drops out of the grouping entirely -- so if the abandoned basin
    was the bad one, ``worst_basin_frac`` FALLS while coverage collapses. On its own this
    metric rewards mode collapse. ``n_missed`` and ``occupancy_entropy`` are published from
    the same call site precisely so the pair is always visible together.

    Two more ways it under-reports, both deliberate and both flagged rather than patched:
    ``e_min`` is a multi-start local minimum and hence an upper bound, so every excess is a
    lower bound; and non-finite energies are dropped rather than counted as tail, so a
    blown-up geometry LEAVES the metric instead of failing it -- which is why
    'Finite Energy Fraction' belongs on the same panel.
    """
    from energies.prior_diagnostics import rotamer_basin_labels

    if basin_ref is None or 'skipped' in basin_ref:
        return {f'{prefix}nonthermal_available': 0}
    e = _host(energies, np.float64)
    finite = np.isfinite(e)
    if finite.sum() == 0 or not np.isfinite(e_min):
        return {f'{prefix}nonthermal_available': 0}

    x_t = torch.as_tensor(x, dtype=en.dtype, device=en.device)
    r, th, ph = en.dof_from_state(x_t)
    dof = np.concatenate([_host(r), _host(th), _host(ph)], axis=1)[finite]
    lab = rotamer_basin_labels(basin_ref['groups'], dof, basin_ref['n0'])
    excess = (e[finite] - e_min) / float(en.temperature)
    bad = excess > float(u_star)

    occupied = np.unique(lab)
    if occupied.size < 2:
        # one occupied basin is not a partition; the pooled 'Nonthermal Fraction' already
        # says everything a single group could. Abstaining rather than publishing a
        # degenerate spread, which would read as "uniform across basins".
        return {f'{prefix}nonthermal_available': 0,
                f'{prefix}nonthermal_n_basins': int(occupied.size)}
    fracs = np.array([bad[lab == b].mean() for b in occupied], dtype=np.float64)
    # ONE KEY: the worst basin. `pooled_frac` is the modeller's own 'Nonthermal Fraction'
    # recomputed on this route's grouping, `basin_spread` is the worst minus a minimum that
    # is 0 whenever anything is clean, `basins_failing` counts what `worst > 0` already
    # announces, and `u_star` is the threshold from the config. The worst basin is the
    # alarm; the rest were four ways of restating it.
    return {f'{prefix}nonthermal_worst_basin_frac': float(fracs.max())}


def cover_constants(en, basin_ref, u_star=None, target_tc=None,
                    prefix: str = 'cover/') -> dict:
    """The coverage group's RUN CONSTANTS -- for printing ONCE, not for logging.

    Every value here is fixed by the molecule and the config for the whole run, so logged
    per eval they were flat traces taking chart space from the three numbers that move.
    They are not redundant, though, and that is why this exists rather than a deletion:
    ``n_missed = 0`` is meaningless without ``n_accessible``, ``worst_frac`` is only
    readable against the uniform expectation ``1/n_accessible``, and ``coupling_tc_debiased``
    is only interpretable against the target's own coupling. Print this line at startup and
    the three live keys become readable; drop it and they do not.
    """
    out = {}
    if basin_ref is None or 'skipped' in basin_ref:
        out[f'{prefix}available'] = 0
        out[f'{prefix}skipped'] = (basin_ref or {}).get('skipped', 'no basin reference')
        return out
    acc = np.asarray(basin_ref['accessible'], dtype=bool)
    n_acc = int(acc.sum())
    out[f'{prefix}n_modes'] = int(len(basin_ref['combos']))
    out[f'{prefix}n_accessible'] = n_acc
    if n_acc:
        out[f'{prefix}expected_frac'] = 1.0 / n_acc
    out[f'{prefix}coupling_n_groups'] = int(len(basin_ref['groups']))
    if target_tc is not None and np.isfinite(target_tc):
        out[f'{prefix}coupling_tc_target'] = float(target_tc)
    if u_star is not None and np.isfinite(u_star):
        out[f'{prefix}nonthermal_u_star'] = float(u_star)
    return out


def _norm_entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(p.size))


# ------------------------------------------------- across-molecule correlations


def feature_correlations(values, features: dict, prefix: str, min_groups: int = 3) -> dict:
    """Correlate a per-sample quantity with per-MOLECULE features (size, n_rings, ...).

    REFUSES BELOW `min_groups` DISTINCT MOLECULES rather than returning a number. On an
    unconditional single-molecule run every feature is constant, so a correlation is 0/0 --
    and numpy would hand back a nan that reads as "measured, no relationship" rather than
    "not measurable". The unavailable flag is the honest reading, and this becomes live
    unchanged as soon as the conditional route trains on a library.
    """
    v = _host(values, np.float64)
    out = {}
    for name, f in features.items():
        f = _host(f, np.float64)
        n_groups = len(np.unique(f[np.isfinite(f)]))
        if n_groups < min_groups or f.shape[0] != v.shape[0]:
            out[f'{prefix}{name}_available'] = 0
            out[f'{prefix}{name}_n_distinct'] = int(n_groups)
            continue
        good = np.isfinite(v) & np.isfinite(f)
        if good.sum() < 3 or f[good].std() < 1e-12 or v[good].std() < 1e-12:
            out[f'{prefix}{name}_available'] = 0
            continue
        out[f'{prefix}{name}_available'] = 1
        out[f'{prefix}{name}_pearson'] = float(np.corrcoef(v[good], f[good])[0, 1])
    return out
