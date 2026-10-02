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

import re

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


def energy_component_stats(en, x, prefix: str = 'E/', hist: bool = True) -> dict:
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

    ``hist=False`` drops the raw arrays and keeps every scalar unchanged -- the per-molecule
    block's form, where a histogram per molecule per eval is exactly the unbounded output
    that block exists to avoid.
    """
    comp = energy_components(en, x)
    out = {}
    total = np.zeros(len(next(iter(comp.values()))))
    for name, v in comp.items():
        if name in FULL_QUANTILE_TERMS:
            _quantiles(v, f'{prefix}{name}', out, hist=hist)
        else:
            _mean_only(v, f'{prefix}{name}', out)
        total = total + v
    _quantiles(total, f'{prefix}total', out, hist=hist)
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
        _quantiles(clipped, f'{prefix}clipped_total', out, hist=hist)
        out[f'{prefix}clip_active_frac'] = float((total > en.clip_cutoff).mean())
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
    out = {'r': np.flatnonzero(block == 0),
           'theta': np.flatnonzero(block == 1),
           'phi': np.flatnonzero(block == 2)}
    # a bounded double-bond dihedral (block 4) is its own class, present only on a chart
    # that has one, so every other chart's key set is unchanged
    if (block == 4).any():
        out['double_bond'] = np.flatnonzero(block == 4)
    return out


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
    first atom; a TRANSVERSE column (block 3, u or v) -> the vertex of the linear bend it
    carries, for BOTH components, since u and v are one 2-D bend at that atom (v sits in a
    torsion row, whose central atom is a different one). Grouping by ELEMENT rather than by
    atom index is deliberate: atom index explodes with molecule size and means nothing
    across two different molecules, while element is bounded at 8 and is directly comparable.

    THE COLUMN -> ROW MAP IS ``_M``'s, the one `dof_from_state` applies and
    conformer_data._dof_state_map stores: column j drives DoF row
    ``_driven_idx[nonzero(_M[:, j])]``. Assigning the j-th free column of a class to the j-th
    row of that class's table is right only when no row of the class is held and none is
    re-coded; a transverse u leaves class 1 (and its v class 2), so every later theta and phi
    column took its neighbour's atom. At `torsion` a column drives several dihedrals about
    ONE bond, which share ``torsion_index[:, 1]``, so its first row names it.
    """
    spec = en.spec
    z = np.asarray(spec.z)
    bi, ai, ti = (np.asarray(spec.bond_index), np.asarray(spec.angle_index),
                  np.asarray(spec.torsion_index))
    n_r, n_th = int(en.n_r), int(en.n_th)
    m = _host(en._M)
    driven = _host(en._driven_idx)
    block = _host(en._free_block)
    tv = np.asarray(en.transverse_angles, dtype=bool)
    # a v column's torsion row -> the angle row whose bend it carries
    bend_of = {int(en.transverse_partner[j]): int(j) for j in np.flatnonzero(tv)}
    owner = np.zeros(block.shape[0], dtype=int)
    for j in range(m.shape[1]):
        rows = np.flatnonzero(m[:, j])
        if rows.size == 0:
            continue
        row = int(driven[rows[0]])
        if row < n_r:
            owner[j] = max(int(z[bi[row, 0]]), int(z[bi[row, 1]]))
        elif row < n_r + n_th:
            owner[j] = int(z[ai[row - n_r, 1]])
        elif int(block[j]) == 3:
            owner[j] = int(z[ai[bend_of[row - n_r - n_th], 1]])
        else:
            owner[j] = int(z[ti[row - n_r - n_th, 1]])
    return owner


def dof_element_stats(en, x, reference=None, prefix: str = 'dof_elem/') -> dict:
    """Spread per (DoF class, central-atom element). Bounded at 4 x 8 groups.

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
    # 'transverse' groups a linear bend's u and v by the element of its vertex; it appears
    # only on a molecule that has one, so every other molecule's key set is unchanged
    # 'double_bond' likewise groups the bounded dihedrals of locked double bonds (block 4)
    for cls, cname in ((0, 'r'), (1, 'theta'), (2, 'phi'), (3, 'transverse'),
                       (4, 'double_bond')):
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


# ============================================================ per molecule, on a set
#
# WHY THIS SECTION EXISTS. Every function above reads ONE chart -- `en._batch`,
# `build_positions`, `dof_from_state`, `spec`, `_free_block` -- and on a carrier set the
# dispatcher refuses every one of those by name (multi_conformer._CHART_METHODS). So the
# block above cannot be un-gated on a set: it has to run once per MOLECULE, on that
# member's own chart and its own member-width rows. Pooling across molecules is not a
# substitute. A pooled carrier statistic is a MIXTURE reading, diluted by 1/M: one
# molecule of 40 collapsed to 0.3x its spread left pooled w1r at its null while that
# molecule alone scored ~4x its perfect median (scratchpad eval_audit/w1r_dilution.py).
#
# THREE KINDS OF REFERENCE, kept apart because they answer different questions:
#   * NONE -- the samples and the force field only: energy composition, the geometry
#     guard (``geom/all_in_range`` IS the in-box fraction, the same `_lin_free_idx`
#     columns ConformerModeller._in_box walls), the finite fraction, ring closure.
#     Always available.
#   * TARGET -- properties of the Boltzmann target, fixed per molecule and computed
#     offline: the tier floor ``e_min`` and the rotamer basin table. T_eff/T, the
#     non-thermal tail and basin coverage read against these, so they stay correctness
#     bars in every phase.
#   * POPULATION -- a set of reference DRAWS: sd ratios, drift, ring-torsion ratios, w1r.
#     These are only as right as the population. Against prior draws they measure
#     distance FROM THE PRIOR, which is the MLE phase's objective and NOT a correctness
#     bar once phase 2 moves the policy off the prior. So a population reference is never
#     implicit: it is an explicit argument with a LABEL, and every key it produces carries
#     that label in its NAME (``vs_prior/...``). A prior-referenced number then cannot be
#     read as a target-referenced one, on a dashboard or in a gate.
#
# BOUNDED OUTPUT. `per_molecule_block` returns one flat scalar row per molecule, for a
# run-dir table; `aggregate_per_condition` reduces those rows to a fixed headline list
# whose key count does not depend on M. Nothing here publishes an array or a key named
# after a molecule.

#: A label for a reference population: one lowercase token, because it becomes part of
#: every key the population produces (``vs_<label>/``).
_REF_LABEL = re.compile(r'^[a-z][a-z0-9_]*$')

#: Floor/ceiling on an sd ratio before its log is taken. A column the sampler holds
#: exactly fixed has ratio 0, and |log 0| = inf would turn every quantile it enters into
#: inf or nan; 1e-6 still reads as |log| = 13.8, far past any bar.
_RATIO_CLIP = 1e-6


def _scalars(d: dict) -> dict:
    """Numeric scalars only -- no arrays, tensors or strings (e.g. ``cover/skipped``)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (bool, np.bool_, int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = float(v)
    return out


def _population_prefix(reference_x, reference_label):
    """``'vs_<label>/'`` for a labelled population reference, None without one.

    REFUSES an unlabelled population, and a label with no population: the label is the only
    thing that says what a ratio is against, and a label with nothing behind it is a caller
    that forgot to pass the draws -- every referenced key would silently go missing.
    """
    if reference_x is None:
        if reference_label is not None:
            raise ValueError(f'reference_label {reference_label!r} given without reference_x')
        return None
    if reference_label is None or not _REF_LABEL.match(str(reference_label)):
        raise ValueError(
            f'a reference population needs a label matching {_REF_LABEL.pattern} (got '
            f'{reference_label!r}); it names every key the population produces, e.g. '
            f'"prior" -> vs_prior/w1r/worst')
    return f'vs_{reference_label}/'


def _label_prefix(reference_label):
    """The same prefix for a caller that only READS labelled rows (aggregation)."""
    if reference_label is None:
        return None
    if not _REF_LABEL.match(str(reference_label)):
        raise ValueError(f'reference_label {reference_label!r} does not match '
                         f'{_REF_LABEL.pattern}')
    return f'vs_{reference_label}/'


def _slot(cache, ident):
    return None if cache is None else cache.setdefault(ident, {})


def _w1r_cache(slot, reference):
    """The member's w1r floor cache, reset if the reference population changed under it.

    _column_w1_ratio keys its floor by n only, which is right for ONE fixed reference. A
    cache kept across evals must therefore be invalidated when the draws change, or the
    floor of the old population would divide the new one's distances.
    """
    if slot is None:
        return None
    # hosted, not np.asarray: the draws can live on a CUDA buffer (see _host)
    r = _host(reference, np.float64)
    fp = (r.shape, float(r.sum()), float(np.square(r).sum()))
    if slot.get('w1r_fp') != fp:
        slot['w1r_fp'] = fp
        slot['w1r'] = {}
    return slot['w1r']


def split_by_member(en, state, mol_batch, energy=None):
    """A set's eval rows -> per-member views: ``(members, states, energies, rows)``.

    Four dicts keyed by identifier, in first-appearance order: the member chart, its rows
    read through its OWN columns (member width k, not the carrier's K), their energies
    (None when `energy` is None), and the batch row indices.

    Built on ``_row_identifiers``, ``_members`` and ``carrier`` only. The one-pass energy
    makes its row checks in ``MultiConformerTorsions._resolve_rows``; the two that
    matter for a per-member SLICE are repeated here, because a wrong slice returns
    plausible numbers:
      * every PAD column is exactly 0 -- a nonzero pad means the row was not produced in
        this member's layout;
      * when the batch carries ``state_mask``, each row's mask IS this member's.

    A single chart (plain ConformerTorsions) is returned as one group keyed by its SMILES.
    """
    x = torch.as_tensor(state)
    n = int(x.shape[0])
    e = None if energy is None else torch.as_tensor(energy).reshape(-1)
    if e is not None and int(e.shape[0]) != n:
        raise ValueError(f'{int(e.shape[0])} energies for {n} rows')
    members = getattr(en, '_members', None)
    if members is None:
        ident = str(getattr(en, 'smiles', 'molecule'))
        rows = torch.arange(n, device=x.device)
        return {ident: en}, {ident: x}, (None if e is None else {ident: e}), {ident: rows}

    order = {}
    for i, ident in enumerate(en._row_identifiers(mol_batch, n)):
        order.setdefault(ident, []).append(i)
    unknown = [k for k in order if k not in members]
    if unknown:
        raise RuntimeError(f'batch carries {len(unknown)} molecule(s) this energy was not '
                           f'built for, e.g. {unknown[0]!r}')
    lay = getattr(en, 'carrier', None)
    mask = getattr(mol_batch, 'state_mask', None) if mol_batch is not None else None
    mem, xs, es, rows = {}, {}, ({} if e is not None else None), {}
    for ident, r in order.items():
        idx = torch.as_tensor(r, dtype=torch.long, device=x.device)
        xi = x.index_select(0, idx)
        if lay is not None:
            pads = torch.as_tensor(lay.pad_cols(ident), dtype=torch.long, device=x.device)
            if pads.numel() and bool((xi.index_select(1, pads) != 0).any()):
                raise RuntimeError(
                    f'{ident}: carrier rows carry nonzero PAD columns (max |x| '
                    f'{float(xi.index_select(1, pads).abs().max()):.3g}); these rows were not '
                    f'produced in {ident}\'s layout')
            if mask is not None:
                want = torch.as_tensor(lay.valid(ident), device=mask.device)
                got = mask.reshape(-1, lay.K).index_select(0, idx.to(mask.device)).bool()
                if not bool((got == want).all()):
                    raise RuntimeError(f'{ident}: the batch\'s state_mask disagrees with the '
                                       f'carrier layout')
            xi = lay.from_carrier(ident, xi)
        mem[ident], xs[ident], rows[ident] = members[ident], xi, idx
        if es is not None:
            es[ident] = e.index_select(0, idx.to(e.device))
    return mem, xs, es, rows


def _member_w1r(member, x, reference, lab, cache=None) -> dict:
    """w1r over ONE member's own columns, with its own periodic mask. Keys under `lab`.

    ``_column_w1_ratio`` unchanged -- same floor, same self-calibrated perfect values --
    so a per-molecule reading means exactly what the pooled one means, minus the mixture.
    Its own row requirement (n >= 32, n_ref >= 4n) abstains here as
    ``<lab>w1r_available = 0`` with both counts beside it, never as a silent gap.
    """
    from progress_metrics import _column_w1_ratio

    xs, r = _host(x, np.float64), _host(reference, np.float64)
    periodic = np.asarray(member.periodic_dims, dtype=bool)
    k = int(periodic.size)
    if xs.ndim != 2 or r.ndim != 2 or xs.shape[1] != k or r.shape[1] != k:
        raise ValueError(f'w1r needs member-width [n, {k}] samples and reference; got '
                         f'{tuple(xs.shape)} and {tuple(r.shape)}')
    w = _column_w1_ratio(xs, r, periodic, cache=cache)
    if not w:
        return {f'{lab}w1r_available': 0, f'{lab}w1r/n_eval': int(xs.shape[0]),
                f'{lab}w1r/n_ref': int(r.shape[0])}
    return {f'{lab}{key}': float(v) for key, v in w.items()}


def per_molecule_w1r(members, states, reference_x, reference_label, cache=None) -> dict:
    """``{identifier: {vs_<label>/w1r/*: value}}`` -- w1r per molecule, own columns only.

    The per-molecule counterpart of the pooled carrier w1r, which is a mixture reading
    (see the section note). Its WORST-MOLECULE reading, the one a gate on "no collapsed
    molecule" reads, is ``aggregate_per_condition(...)['<ns>/vs_<label>/w1r/median/max']``:
    the largest per-molecule MEDIAN over columns. Not ``w1r/worst/max``: at panel-sized n
    each molecule's worst column is an extreme value over k noisy columns, and the max of
    those over molecules is dominated by noise. Measured on the real prior (40 molecules,
    62 rows each, one molecule collapsed to 0.3x): median/max 1.29 -> 3.2-3.8 with the
    collapsed molecule on top in all three arms, while worst/max read 4.70 on the control
    and did not move at all for one of the three.

    The reference is a POPULATION and is labelled accordingly: against prior draws this is
    an MLE-phase diagnostic, not a phase-2 correctness bar.
    """
    lab = _population_prefix(reference_x, reference_label)
    if lab is None:
        raise ValueError('per_molecule_w1r needs a labelled reference population')
    out = {}
    for ident, x in states.items():
        rx = reference_x.get(ident)
        if rx is None:
            out[ident] = {f'{lab}w1r_available': 0}
            continue
        out[ident] = _member_w1r(members[ident], x, rx, lab,
                                 _w1r_cache(_slot(cache, ident), rx))
    return out


def _abs_log_sd_ratio_max(x, ref, periodic):
    """Per-column sd ratio against the reference, reduced TWO-SIDED: max_j |log ratio_j|.

    ``dof/*_sd_ratio_max`` is a max of the ratio, so it sees the widest column and is blind
    to a single COLLAPSED one. |log ratio| scores both failures on one scale.

    THE SPREAD OF A WRAPPED COLUMN IS CIRCULAR. A phi state is the displacement from the
    reference dihedral in units of pi, wrapped to [-1, 1], so a rotamer 180 degrees from
    the reference (a two-fold OH, amide or aryl flip) straddles the wrap and its LINEAR sd
    reads about 1 whatever its width. On an 8-degree mode there (ten draws of 64 rows
    against 256), a 0.3x collapse read 0.01-0.04 where |log 0.3| = 1.20, and a pure
    15-degree shift at unchanged width read 0.87-3.16 where 0 is right; the circular sd
    reads 1.17-1.31 and 0.01-0.09. Periodic columns therefore take
    ``_circ_sd_deg`` with ring_torsion_stats' 1e-6-degree guard; linear columns keep
    dof_class_stats' np.std and 1e-9 guard. The ratio is unit-free, so the two kinds of
    column share the one reduction.
    """
    x, ref = _host(x, np.float64), _host(ref, np.float64)
    per = np.asarray(periodic, dtype=bool)
    sd_s, sd_r = x.std(axis=0), ref.std(axis=0)
    floor = np.full(per.shape, 1e-9)
    for j in np.flatnonzero(per):
        sd_s[j], sd_r[j] = _circ_sd_deg(180.0 * x[:, j]), _circ_sd_deg(180.0 * ref[:, j])
        floor[j] = 1e-6
    good = sd_r > floor
    if not good.any():
        return None
    ratio = np.clip(sd_s[good] / sd_r[good], _RATIO_CLIP, 1.0 / _RATIO_CLIP)
    return float(np.abs(np.log(ratio)).max())


def _reduce_ringtor(rt: dict, lab, prefix: str = 'ringtor/') -> dict:
    """``ring_torsion_stats`` reduced WITHIN the molecule: max over its cycles.

    Cycle index is molecule-local -- c0 of one molecule and c0 of another are unrelated
    rings -- so per-cycle keys cannot be aggregated across molecules. The reductions keep
    each failure's direction-free magnitude: the worst |log sd ratio| (either side of 1),
    the worst correlation distance, the worst circular mean shift.
    """
    out = {f'{prefix}available': int(rt.get(f'{prefix}available', 0)),
           f'{prefix}n_cycles': int(rt.get(f'{prefix}n_cycles', 0))}
    tags = [f'{prefix}c{ci}' for ci in range(out[f'{prefix}n_cycles'])]
    if not out[f'{prefix}available'] or not tags:
        return out
    out[f'{prefix}sd_max_deg'] = max(float(rt[f'{t}_sd_max_deg']) for t in tags)
    if lab is None:
        return out
    ratios, corr, shift = [], [], []
    for t in tags:
        if f'{t}_sd_ratio_max' in rt:
            lo, hi = (np.clip([rt[f'{t}_sd_ratio_min'], rt[f'{t}_sd_ratio_max']],
                              _RATIO_CLIP, 1.0 / _RATIO_CLIP))
            ratios.append(max(abs(float(np.log(lo))), abs(float(np.log(hi)))))
        if f'{t}_corr_dist' in rt:
            corr.append(float(rt[f'{t}_corr_dist']))
        if f'{t}_mean_shift_max_deg' in rt:
            shift.append(float(rt[f'{t}_mean_shift_max_deg']))
    if not (ratios or corr or shift):
        out[f'{lab}{prefix}ref_available'] = 0
        return out
    if ratios:
        out[f'{lab}{prefix}abs_log_sd_ratio_max'] = max(ratios)
    if corr:
        out[f'{lab}{prefix}corr_dist_max'] = max(corr)
    if shift:
        out[f'{lab}{prefix}mean_shift_max_deg'] = max(shift)
    return out


def per_molecule_block(members, states, energies=None, refs=None, *, n_min: int,
                       reference_x=None, reference_label=None,
                       nonthermal_entropy_per_dim=None, cache=None) -> dict:
    """The physics block, once per molecule on its own chart. ``{identifier: {key: scalar}}``.

    INPUTS, all keyed by identifier and all explicit (pure: nothing is read off a modeller):
      members      the member ConformerTorsions (``split_by_member``'s first return)
      states       MEMBER-WIDTH rows ``[n_c, k_c]``; the block scores whatever is here
      energies     ``[n_c]`` raw potential in kcal/mol -- the baked ``conformer_energy``,
                   the same currency as ``e_min``. None: every energy reading abstains.
      refs         TARGET references per molecule: ``e_min`` (the tier floor, e.g. from the
                   offline references table), ``basin_ref``
                   (prior_diagnostics.basin_reference) and optionally ``target_tc``. A
                   missing entry makes its readings abstain with ``*_available = 0``.
      reference_x  a POPULATION of member-width reference draws per molecule, with
                   `reference_label` naming it; every key it produces is published under
                   ``vs_<label>/``. None: reference-free and target readings only.
      n_min        molecules with fewer rows are NOT scored: their row is
                   ``{'phys/n_rows': n, 'phys/below_n_min': 1}``, and the aggregation counts
                   them apart. An sd, a coverage or a w1r over a handful of rows is noise
                   that would read as a verdict.
      nonthermal_entropy_per_dim  s in the non-thermal bar u* = s * k, with k THIS
                   molecule's own DoF count (member.ndim), never the carrier width K, which
                   is the widest member's and would make every smaller molecule's bar too
                   lenient by K/k. None or 0 = the channel is off, as in the config.
      cache        optional dict the caller keeps across evals: ring cycles and w1r floors
                   per identifier, both fixed by the molecule and its reference.

    Per scored molecule the row holds the scalars of energy_component_stats (hist off),
    geometry_stats, dof_class_stats and dof_element_stats (reference-free halves; the
    referenced halves under the label), ring_stats, ring_torsion_stats reduced within the
    molecule, basin_coverage plus ``cover/missed_frac``, basin_coupling, basin_nonthermal,
    thermal_stats, per-molecule w1r, and ``phys/`` / ``thermal/`` readings of its own:
    rows, k, finite fraction, u* and the non-thermal fraction. Each copied key has exactly
    the value the single-molecule call gives on these rows.
    """
    n_min = int(n_min)
    if n_min < 1:
        raise ValueError(f'n_min must be >= 1, got {n_min}')
    lab = _population_prefix(reference_x, reference_label)
    refs = {} if refs is None else refs
    s = 0.0 if nonthermal_entropy_per_dim is None else float(nonthermal_entropy_per_dim)
    unknown = [i for i in states if i not in members]
    if unknown:
        raise KeyError(f'states for {len(unknown)} identifier(s) with no member, e.g. '
                       f'{unknown[0]!r}')
    if energies is not None:
        missing = [i for i in states if i not in energies]
        if missing:
            raise KeyError(f'energies missing for {len(missing)} molecule(s), e.g. '
                           f'{missing[0]!r}')

    from energies.ring_metrics import ring_cycles

    out = {}
    for ident, x in states.items():
        member = members[ident]
        k = int(member.ndim)
        n = int(x.shape[0])
        row = {'phys/n_rows': n, 'phys/k': k}
        if n < n_min:
            row['phys/below_n_min'] = 1
            out[ident] = row
            continue
        if x.ndim != 2 or int(x.shape[1]) != k:
            raise ValueError(f'{ident}: states must be member-width [n, {k}], got '
                             f'{tuple(x.shape)} -- carrier rows go through split_by_member')
        e = None
        if energies is not None:
            e = _host(energies[ident], np.float64).reshape(-1)
            if e.shape[0] != n:
                raise ValueError(f'{ident}: {e.shape[0]} energies for {n} rows')
            if getattr(member, 'temperature_conditioning', False):
                # thermal_stats and the non-thermal bar divide by ONE temperature; with
                # temperature conditioning each row carries its own, and dividing by the
                # member's nominal one would publish a plausible wrong excess
                raise ValueError(f'{ident}: per-molecule thermal readings assume a fixed '
                                 f'temperature; temperature_conditioning is on')
        rx = None
        if lab is not None:
            rx = reference_x.get(ident)
            if rx is None:
                row[f'{lab}ref_available'] = 0
            else:
                rx = _host(rx)
                if rx.ndim != 2 or rx.shape[1] != k:
                    raise ValueError(f'{ident}: reference draws must be member-width '
                                     f'[m, {k}], got {tuple(rx.shape)}')
        slot = _slot(cache, ident)

        # ---- reference-free: the samples and the force field only
        row.update(_scalars(energy_component_stats(member, x, hist=False)))
        row.update(_scalars(geometry_stats(member, x)))
        dof_free = _scalars(dof_class_stats(member, x))
        elem_free = _scalars(dof_element_stats(member, x))
        row.update(dof_free)
        row.update(elem_free)
        row.update(_scalars(ring_stats(member, x)))
        if slot is not None and 'cycles' in slot:
            cycles = slot['cycles']
        else:
            cycles = ring_cycles(member)
            if slot is not None:
                slot['cycles'] = cycles
        row.update(_reduce_ringtor(ring_torsion_stats(member, x, reference=rx, cycles=cycles),
                                   lab if rx is not None else None))

        # ---- population-referenced, every key under the label. Only the keys the
        # reference ADDS are kept: the reference-free half is already in the row above,
        # unlabelled, because it does not depend on the reference.
        if rx is not None:
            dof_ref = _scalars(dof_class_stats(member, x, reference=rx))
            row.update({f'{lab}{key}': v for key, v in dof_ref.items() if key not in dof_free})
            elem_ref = _scalars(dof_element_stats(member, x, reference=rx))
            row.update({f'{lab}{key}': v for key, v in elem_ref.items()
                        if key not in elem_free})
            two_sided = _abs_log_sd_ratio_max(x, rx, member.periodic_dims)
            if two_sided is not None:
                row[f'{lab}dof/abs_log_sd_ratio_max'] = two_sided
            row.update(_member_w1r(member, x, rx, lab, _w1r_cache(slot, rx)))

        # ---- target-referenced: the floor and the basin table
        ref = refs.get(ident) or {}
        e_min = ref.get('e_min')
        e_min = float('nan') if e_min is None else float(e_min)
        basin_ref = ref.get('basin_ref')
        if e is None:
            row['phys/energy_available'] = 0
        else:
            finite = np.isfinite(e)
            row['phys/finite_frac'] = float(finite.mean())
            # thermal_stats reads member.ndim: THIS molecule's d, so T_eff/T is per-molecule
            row.update(_scalars(thermal_stats(member, e, e_min)))
        if s > 0:
            u_star = s * k
            row['thermal/nonthermal_u_star'] = u_star
            fin = np.isfinite(e) if e is not None else None
            if e is None or not np.isfinite(e_min) or not fin.any():
                row['thermal/nonthermal_available'] = 0
            else:
                # train.py's reduction, per molecule: excess in nats against this
                # molecule's own floor, clamped at 0 (a row below the floor is a new
                # record, not tail), non-finite rows dropped and counted in finite_frac
                u = np.maximum((e[fin] - e_min) / float(member.temperature), 0.0)
                row['thermal/nonthermal_frac'] = float((u > u_star).mean())
                row.update(_scalars(basin_nonthermal(member, x, e, e_min, basin_ref,
                                                     u_star)))
        cov = _scalars(basin_coverage(member, x, basin_ref))
        row.update(cov)
        if basin_ref is not None and 'skipped' not in basin_ref:
            n_acc = int(np.asarray(basin_ref['accessible'], dtype=bool).sum())
            row['cover/n_modes'] = int(len(basin_ref['combos']))
            row['cover/n_accessible'] = n_acc
            if 'cover/n_missed' in cov and n_acc:
                # the fraction, not the count: n_missed = 3 is a disaster on a molecule
                # with 3 accessible basins and noise on one with 300, and only the
                # fraction means the same thing across molecules
                row['cover/missed_frac'] = cov['cover/n_missed'] / n_acc
        row.update(_scalars(basin_coupling(member, x, basin_ref,
                                           target_tc=ref.get('target_tc'))))
        out[ident] = row
    return out


# ------------------------------------------------------------ bounded aggregation

#: The per-condition HEADLINES: about a dozen per-molecule readings, each turned into a
#: BADNESS (0 = ideal, larger = worse) so that one bad side and one quantile convention
#: serve every key. ``reference`` says what the reading is against:
#:   None          samples and force field only
#:   'target'      the target's own floor / basin table -- a correctness bar in any phase
#:   'population'  labelled reference draws; published under ``vs_<label>/`` and only when
#:                 a label is given. Against the prior these are MLE-phase diagnostics.
#: Transforms: identity (lower is better, ideal 0 or self-calibrated), abs_dev
#: (|v - ideal|), shortfall (ideal - v, for a fraction whose ideal is 1).
#:
#: T_eff/T's ideal is 2, not 1: thermal_stats defines T_eff = 1 + 2 * median_excess / d
#: WITHOUT the -d/2 offset prior_baselines' excess carries, so a harmonic well sampled at
#: the target temperature (median excess d/2) reads 2.0 there. The ideal follows the
#: function as it stands; the three zeros are laid side by side on the wiki's
#: conformer-force-field-and-prior page.
PANEL_HEADLINES = (
    {'name': 'thermal/T_eff_over_T_dev', 'key': 'E/T_eff_over_T',
     'transform': 'abs_dev', 'ideal': 2.0, 'reference': 'target'},
    {'name': 'thermal/equipartition_dev', 'key': 'E/frac_within_equipartition',
     'transform': 'abs_dev', 'ideal': 0.5, 'reference': 'target'},
    {'name': 'thermal/nonthermal_frac', 'key': 'thermal/nonthermal_frac',
     'transform': 'identity', 'ideal': 0.0, 'reference': 'target'},
    {'name': 'cover/missed_frac', 'key': 'cover/missed_frac',
     'transform': 'identity', 'ideal': 0.0, 'reference': 'target'},
    {'name': 'cover/nonthermal_worst_basin_frac', 'key': 'cover/nonthermal_worst_basin_frac',
     'transform': 'identity', 'ideal': 0.0, 'reference': 'target'},
    {'name': 'geom/out_of_box_frac', 'key': 'geom/all_in_range',
     'transform': 'shortfall', 'ideal': 1.0, 'reference': None},
    {'name': 'phys/nonfinite_frac', 'key': 'phys/finite_frac',
     'transform': 'shortfall', 'ideal': 1.0, 'reference': None},
    {'name': 'ring/closure_err_sigma', 'key': 'ring/closure_err_sigma',
     'transform': 'identity', 'ideal': 0.0, 'reference': None},
    {'name': 'dof/abs_log_sd_ratio', 'key': 'dof/abs_log_sd_ratio_max',
     'transform': 'identity', 'ideal': 0.0, 'reference': 'population'},
    {'name': 'ringtor/abs_log_sd_ratio', 'key': 'ringtor/abs_log_sd_ratio_max',
     'transform': 'identity', 'ideal': 0.0, 'reference': 'population'},
    {'name': 'ringtor/corr_dist', 'key': 'ringtor/corr_dist_max',
     'transform': 'identity', 'ideal': 0.0, 'reference': 'population'},
    # w1r's ideal is its own self-calibrated perfect value (w1r/perfect_*, in each
    # molecule's row), not a constant; the raw ratio is aggregated. The worst-molecule gate
    # reading is w1r/median's max -- see per_molecule_w1r for why not w1r/worst's
    {'name': 'w1r/median', 'key': 'w1r/median',
     'transform': 'identity', 'ideal': None, 'reference': 'population'},
    {'name': 'w1r/worst', 'key': 'w1r/worst',
     'transform': 'identity', 'ideal': None, 'reference': 'population'},
)

_TRANSFORMS = {
    'identity': lambda v, ideal: v,
    'abs_dev': lambda v, ideal: abs(v - ideal),
    'shortfall': lambda v, ideal: ideal - v,
}

#: Per-molecule features the correlation block reads, and the headlines it correlates:
#: one per kind of failure (thermal, coverage, spread, marginal fit), so "which kind of
#: molecule fails" is asked once of each.
MOLECULE_FEATURES = ('k', 'n_rings', 'n_rotors', 'n_modes')
CORR_HEADLINES = ('thermal/T_eff_over_T_dev', 'cover/missed_frac', 'dof/abs_log_sd_ratio',
                  'w1r/median')


def _check_row_labels(per_mol, reference_label):
    """REFUSE rows carrying a population label the caller did not name.

    `_resolve` publishes population headlines only under the label passed, so rows made
    with ``reference_label='prior'`` and read with no label, or with another one, would
    lose every ``vs_prior/`` headline -- a collapsed molecule's w1r gone from the aggregate
    with nothing raised, and a later gate on the missing key reading as a pass. The block
    refuses an unlabelled population; this is the same refusal on the reading side.
    """
    found = {key[3:key.index('/')] for r in per_mol.values() for key in r
             if key.startswith('vs_') and '/' in key}
    stray = sorted(found - ({str(reference_label)} if reference_label is not None else set()))
    if stray:
        raise ValueError(f'rows carry population keys labelled {stray} but reference_label is '
                         f'{reference_label!r}; pass the label the rows were made with, one '
                         f'label per call')


def _resolve(spec, reference_label):
    """Spec entries -> ``[(published name, per-molecule key, entry)]``. Validates first.

    Population entries resolve under ``vs_<label>/`` and are DROPPED without a label: a
    key named after no reference would be the unlabelled number this section refuses.
    Rows that DO carry a label are checked against it first (`_check_row_labels`), so the
    drop only ever applies to rows with no population keys to lose.
    """
    lab = _label_prefix(reference_label)
    out = []
    for h in PANEL_HEADLINES if spec is None else spec:
        if h['transform'] not in _TRANSFORMS:
            raise ValueError(f"{h['name']}: unknown transform {h['transform']!r}")
        if h.get('reference') == 'population':
            if lab is None:
                continue
            out.append((f"{lab}{h['name']}", f"{lab}{h['key']}", h))
        else:
            out.append((h['name'], h['key'], h))
    return out


def _badness(h, row, key):
    """One molecule's transformed value for one headline, or None if it has none."""
    v = row.get(key)
    if v is None:
        return None
    v = float(v)
    if not np.isfinite(v):
        return None
    return float(_TRANSFORMS[h['transform']](v, h.get('ideal')))


def _scored(per_mol):
    return {i: r for i, r in per_mol.items() if not r.get('phys/below_n_min')}


def aggregate_per_condition(per_mol: dict, spec, worst_quantile: float, *, ns: str,
                            reference_label=None) -> dict:
    """Per-molecule rows -> a FIXED key set: per headline median / worst / max / n /
    n_unavailable, plus the molecule counts. The count does not depend on M.

    ``worst`` follows ``utils.per_condition_fraction``: the ``worst_quantile`` tail on the
    BAD side, i.e. the (1 - worst_quantile) quantile of the badness, with worst_quantile
    the fraction of molecules allowed beyond it (``conditional_worst_quantile``). ``max``
    is the single worst molecule -- the gate reading, because a quantile tail lets up to
    worst_quantile of the panel fail unseen and "no collapsed molecule" is a max.

    ABSENT IS NOT ZERO. A scored molecule with no finite value for a headline -- its
    ``*_available = 0``, an acyclic molecule on a ring key, no reference given -- is
    counted in ``<key>/n_unavailable`` and never averaged in. Molecules below n_min are
    counted in ``<ns>/n_below_n_min`` and enter no headline at all. With no molecule
    carrying a value, only ``n`` and ``n_unavailable`` are published for that headline.

    Rows made against a labelled population must be read with that label: a row carrying
    ``vs_<other>/`` keys raises rather than dropping its population headlines unseen.
    """
    wq = float(worst_quantile)
    if not 0.0 <= wq <= 1.0:
        raise ValueError(f'worst_quantile must be in [0, 1], got {wq}')
    _check_row_labels(per_mol, reference_label)
    scored = _scored(per_mol)
    out = {f'{ns}/n_molecules': len(scored), f'{ns}/n_below_n_min': len(per_mol) - len(scored)}
    for name, key, h in _resolve(spec, reference_label):
        vals = [b for b in (_badness(h, r, key) for r in scored.values()) if b is not None]
        base = f'{ns}/{name}'
        out[f'{base}/n'] = len(vals)
        out[f'{base}/n_unavailable'] = len(scored) - len(vals)
        if vals:
            a = np.asarray(vals, dtype=np.float64)
            out[f'{base}/median'] = float(np.median(a))
            out[f'{base}/worst'] = float(np.quantile(a, 1.0 - wq))
            out[f'{base}/max'] = float(a.max())
    return out


def worst_molecules(per_mol: dict, name: str, k: int = 3, spec=None,
                    reference_label=None) -> list:
    """The `k` worst molecules on one headline, ``[(identifier, badness)]``, worst first.

    For a summary line or the drill-down table, NOT for logging: identifiers as keys would
    make the key set grow with M.
    """
    _check_row_labels(per_mol, reference_label)
    for pub, key, h in _resolve(spec, reference_label):
        if pub == name:
            vals = [(i, _badness(h, r, key)) for i, r in _scored(per_mol).items()]
            vals = [(i, b) for i, b in vals if b is not None]
            return sorted(vals, key=lambda t: (-t[1], t[0]))[:int(k)]
    raise KeyError(f'{name!r} is not a headline of this spec (label {reference_label!r})')


def _n_rotors(member) -> int:
    """Torsion groups about an ACYCLIC SINGLE bond: the molecule's genuine rotors.

    ``torsion_groups`` groups phi rows by central bond with no rotatability test, so its
    length also counts ring bonds, aromatic bonds and double bonds (toluene 6, THF 3,
    2-butene 3) and as a "flexibility" feature would mostly re-measure size and ring count.
    The bond order and ring membership are RDKit's, on the member's own molecule, whose
    atoms the placement slots index through ``spec.perm``. Methyl, hydroxyl and amide C-N
    groups ARE counted: at explicit hydrogens each is a free torsional coordinate with its
    own rotamer modes (basin_reference enumerates them). Not ``member.rotatable`` either --
    that is the torsion tier's tree-bond set, which drops terminal spins and so depends on
    where the spanning tree is rooted.
    """
    from rdkit import Chem

    perm = np.asarray(member.spec.perm)
    ti = np.asarray(member.spec.torsion_index)
    n = 0
    for rows in member.torsion_groups():
        b, c = int(perm[ti[rows[0], 1]]), int(perm[ti[rows[0], 2]])
        bond = member.mol.GetBondBetweenAtoms(b, c)
        if bond is None:
            # the tree's bonds are perceived from geometry, the molecule's from the SMILES;
            # a central bond RDKit does not have means the two graphs disagree
            raise RuntimeError(f'{member.smiles}: torsion central bond {b}-{c} is not a bond '
                               f'of the RDKit molecule')
        n += int(bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing())
    return n


def molecule_features(member, n_modes=None) -> dict:
    """``{k, n_rings, n_rotors, n_modes}`` for one member: the correlation and panel axes.

    `n_rotors` counts torsion groups about acyclic single bonds (`_n_rotors`). `n_modes` is
    the rotamer-mode count from the references table (None when the basin reference was
    skipped or not computed); it is passed in rather than recomputed here because
    enumerating it is the expensive part of basin_reference.
    """
    from energies.ring_metrics import ring_cycles

    return {'k': int(member.ndim), 'n_rings': int(len(ring_cycles(member))),
            'n_rotors': _n_rotors(member),
            'n_modes': None if n_modes is None else int(n_modes)}


def _feature(features, ident, name):
    if ident not in features:
        raise KeyError(f'no features for {ident!r}')
    v = features[ident].get(name)
    return float('nan') if v is None else float(v)


def per_molecule_correlations(per_mol: dict, features: dict, *, ns: str, spec=None,
                              headlines=CORR_HEADLINES, reference_label=None,
                              min_groups: int = 3) -> dict:
    """``feature_correlations`` over MOLECULES: which kind of molecule fails?

    Each headline's per-molecule badness against each per-molecule feature
    (``MOLECULE_FEATURES``). Bounded: len(headlines) x 4 features, whatever M is.
    ``feature_correlations`` refuses a feature with fewer than `min_groups` distinct
    values as ``*_available = 0`` -- a Pearson over two molecule sizes is a line through
    two points. A molecule without a value for a headline is left out of that
    headline's correlation, never entered as 0.
    """
    _check_row_labels(per_mol, reference_label)
    want = set(headlines)
    scored = _scored(per_mol)
    out = {}
    for name, key, h in _resolve(spec, reference_label):
        bare = name.split('/', 1)[1] if name.startswith('vs_') else name
        if bare not in want:
            continue
        ids = [i for i, r in scored.items() if _badness(h, r, key) is not None]
        v = np.asarray([_badness(h, scored[i], key) for i in ids], dtype=np.float64)
        feats = {f: np.asarray([_feature(features, i, f) for i in ids], dtype=np.float64)
                 for f in MOLECULE_FEATURES}
        out.update(feature_correlations(v, feats, prefix=f'{ns}/corr/{name}/',
                                        min_groups=min_groups))
    return out


def select_panel(identifiers, features: dict, size: int, seed: int,
                 n_k_bins: int = 3) -> list:
    """A fixed, stratified, requeue-stable panel of molecules. Sorted identifiers.

    DETERMINISTIC FROM (seed, the identifier SET, their features) alone: identifiers are
    sorted before anything else and the features are looked up by identifier, so input
    order is irrelevant and a requeued leg rebuilds the identical panel with no checkpoint
    state. Returns every molecule when M <= size.

    STRATIFIED by k-bin x ring/acyclic x basin-available, because the failure modes differ
    by stratum -- ring closure only exists on rings, coverage only where the basin table
    exists -- and a plain random panel of 32 from thousands can miss a whole stratum. k-bins
    are equal-count bins over the CANDIDATE set's own k (a fixed edge list would be a
    constant tuned to one dataset). basin-available means n_modes >= 2: a one-mode molecule
    is trivially covered by any sample, so it cannot exercise the coverage reading.

    Allocation is largest-remainder proportional to stratum size, after one slot for every
    non-empty stratum when size allows, in exact integer arithmetic so no float tie can
    push a stratum past its size.
    """
    ids = [str(i) for i in identifiers]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{len(ids) - len(set(ids))} duplicate identifier(s); a panel of '
                         f'duplicates would score one molecule twice')
    ids = sorted(ids)
    size = int(size)
    if size < 1:
        raise ValueError(f'panel size must be >= 1, got {size}')
    if len(ids) <= size:
        return ids
    missing = [i for i in ids if i not in features]
    if missing:
        raise KeyError(f'no features for {len(missing)} candidate(s), e.g. {missing[0]!r}; '
                       f'stratifying on a guess would make the panel depend on it')

    k = np.asarray([float(features[i]['k']) for i in ids])
    edges = np.quantile(k, np.linspace(0.0, 1.0, int(n_k_bins) + 1)[1:-1])
    kbin = np.searchsorted(edges, k, side='right')
    strata = {}
    for j, ident in enumerate(ids):
        n_rings = features[ident].get('n_rings') or 0
        n_modes = features[ident].get('n_modes') or 0
        key = (int(kbin[j]), int(n_rings > 0), int(n_modes >= 2))
        strata.setdefault(key, []).append(ident)
    keys = sorted(strata)
    sizes = [len(strata[s]) for s in keys]

    if size >= len(keys):
        base, cap, rem = [1] * len(keys), [c - 1 for c in sizes], size - len(keys)
    else:
        base, cap, rem = [0] * len(keys), list(sizes), size
    total = sum(cap)
    quota = [rem * c // total if total else 0 for c in cap]
    frac = [rem * c % total if total else 0 for c in cap]
    left = rem - sum(quota)
    for j in sorted(range(len(keys)), key=lambda j: (-frac[j], j))[:left]:
        quota[j] += 1

    rng = np.random.default_rng(int(seed))
    chosen = []
    for s, b, q in zip(keys, base, quota):
        pool = strata[s]
        chosen += [pool[j] for j in rng.permutation(len(pool))[:b + q]]
    return sorted(chosen)
