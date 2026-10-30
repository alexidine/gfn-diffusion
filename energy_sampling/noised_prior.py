"""Pre-noised prior files (docs/design/prior_dataset_build.md): the build manifest, the per-row tags, and the
load-time selection both the builder (data_processing/build_noised_prior.py) and the trainer
(train.py::Modeller.init_prior_dataset) read.

A pre-noised prior file is an ordinary train.py prior file ({'prior', 'equalized_prior', 'thermal_scaling_factor'})
whose rows are already scored through the run's analysis (energies stamped with the file's lj_coeff), carrying three
graph attributes and a manifest under MANIFEST_KEY:

  noised_kind    KIND_ANCHOR (a source row), KIND_CHILD (an anchor kicked by the iso tile), or KIND_HARVESTED
                 (a row a run scored, merged in from its harvest shards, not kicked)
  noised_split   SPLIT_TRAIN, or SPLIT_HELDOUT (an anchor held out whole, with every child of it)
  noised_parent  the anchor row a child was kicked from (an anchor: itself)

The loader keeps rows by kind and split, inside the packing window and within window_kT x T of their condition's
floor, and checks a stratified sample of rows against a fresh scoring instead of re-scoring the file.
"""
import math
import os

import numpy as np
import torch

MANIFEST_KEY = 'noised_build'
FORMAT_VERSION = 1
KIND_ANCHOR, KIND_CHILD, KIND_HARVESTED = 0, 1, 2
KIND_NAMES = {'anchor': KIND_ANCHOR, 'child': KIND_CHILD, 'harvested': KIND_HARVESTED}
SPLIT_TRAIN, SPLIT_HELDOUT = 0, 1
TAGS = ('noised_kind', 'noised_split', 'noised_parent')

# the stored terms the entry check compares, each against |fresh - stored| <= ABS_TOL + REL_TOL * |stored|; a row
# off by more than FAIL_TOL on any of them fails the run outright
CHECK_TERMS = ('reduction_en', 'packing_coeff', 'cell_volume')
ABS_TOL, REL_TOL = 0.1, 1e-4   # energy units: two scorings of one crystal differ by up to ~0.06 raw eLJ in float noise
CHECK_ROWS = 512
# the load-time defaults (cfg:noised_prior.*): an absent key reads these
DEFAULTS = dict(window_kT=6.0, packing_window=None, train_kinds=('child',), check_rows=CHECK_ROWS,
                live_kernel=None, live_tau=1.0, live_width_cap=0.25)
#: live_kernel: the Gaussian the MLE draw adds to each drawn row (train.py::Modeller._live_kernel_draw)
LIVE_KERNELS = (None, 'iso', 'hessian')


def settings(args):
    """cfg:noised_prior as a dict, absent keys at DEFAULTS; packing_window None reads density_penalty's range."""
    from energies.molecular_crystal import PACKING_COEFF_WINDOW
    cfg = getattr(args, 'noised_prior', None)
    cfg = dict(vars(cfg)) if cfg is not None and not isinstance(cfg, dict) else dict(cfg or {})
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"noised_prior: unknown keys {sorted(unknown)} (known: {sorted(DEFAULTS)})")
    s = {**DEFAULTS, **cfg}
    s['packing_window'] = tuple(float(v) for v in (s['packing_window'] or PACKING_COEFF_WINDOW))
    bad = [k for k in s['train_kinds'] if k not in KIND_NAMES]
    if bad or not s['train_kinds']:
        raise ValueError(f"noised_prior.train_kinds must be a non-empty subset of {sorted(KIND_NAMES)}, got "
                         f"{s['train_kinds']}")
    s['window_kT'] = None if s['window_kT'] is None else float(s['window_kT'])
    if s['live_kernel'] not in LIVE_KERNELS:
        raise ValueError(f"noised_prior.live_kernel must be one of {LIVE_KERNELS}, got {s['live_kernel']!r}")
    for k in ('live_tau', 'live_width_cap'):
        v = s[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"noised_prior.{k} must be a finite number > 0, got {v!r}")
    return s


def identity(args, energy_function):
    """What is baked into a stored row's energy terms, so a file is read only by runs that would score it the same:
    the energy function and its weights, the analysis settings as resolved (MolecularCrystal.analyze_crystal_batch's
    defaults, then energy_config.analyze_kwargs), the MLIP density guard, the legacy triclinic wall switch, the space
    groups and Z' the run builds, and the eLJ coefficient the energies are scaled by. Not the state chart
    (energy_config.crystal_coordinates): the stored terms are the crystal's, and the chart's Jacobian is recomputed from
    the crystal whenever a reward is composed (prebuilt_sample_to_reward), so a latent-chart build serves an FD run."""
    analyze = dict(cutoff=10, supercell_size=10, std_orientation=False)
    analyze.update({k: v for k, v in dict(getattr(energy_function, 'analyze_kwargs', {}) or {}).items()
                    if k not in ('predictor', 'c', 'width')})
    mlip = getattr(args, 'mlip_path', None)
    ec = args.energy_config
    return {
        'energy_function': str(args.energy_function),
        'mlip': os.path.basename(str(mlip)) if mlip else None,
        'analyze_kwargs': {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                           for k, v in sorted(analyze.items())},
        'mlip_max_packing_coeff': getattr(ec, 'mlip_max_packing_coeff', None),
        'legacy_triclinic_walls': os.environ.get('MXT_LEGACY_TRICLINIC_WALLS'),
        'space_groups': [int(s) for s in args.space_groups],
        'z_primes': [int(z) for z in args.z_primes],
        'lj_coeff': (round(float(energy_function.lj_coeff), 9)
                     if getattr(energy_function, 'lj_coeff', None) is not None else None),
    }


def identity_mismatch(stored, current):
    """The identity keys on which a file and a run differ, as 'key: file X, run Y' strings (empty when they agree).
    The keys are the current definition's (identity): a key a file records that this code no longer counts as identity
    does not refuse it; a key the file lacks does."""
    keys = sorted(current)
    return [f"{k}: file {stored.get(k)!r}, run {current.get(k)!r}" for k in keys if stored.get(k) != current.get(k)]


def condition_groups(batch):
    """[n] long: each row's condition as this file holds it -- (identifier, space group, Z') -- numbered in order of
    first appearance. The run's condition ids do not exist yet when the prior loads (init_identifiers runs after)."""
    ids = [str(m) for m in batch.identifier]
    sg = torch.as_tensor(batch.sg_ind).reshape(-1).tolist()
    zp = torch.as_tensor(batch.z_prime).reshape(-1).tolist()
    table, out = {}, []
    for key in zip(ids, sg, zp):
        out.append(table.setdefault(key, len(table)))
    return torch.tensor(out, dtype=torch.long)


def condition_floor(energy, groups, eligible):
    """[n] float64: each row's condition floor, the minimum of `energy` over the condition's eligible rows (+inf for a
    condition with none)."""
    n_groups = int(groups.max()) + 1 if groups.numel() else 0
    e = torch.where(eligible, energy.double(), torch.full_like(energy, float('inf'), dtype=torch.float64))
    floor = torch.full((n_groups,), float('inf'), dtype=torch.float64)
    floor = floor.scatter_reduce(0, groups, e, reduce='amin')
    return floor[groups]


def tag(batch, name):
    """The row tag `name` as a [n] long tensor on the CPU (raises when the batch does not carry it)."""
    if not hasattr(batch, name):
        raise ValueError(f"a pre-noised prior row carries no {name!r}: rebuild the file with "
                         f"data_processing/build_noised_prior.py")
    return torch.as_tensor(getattr(batch, name)).reshape(-1).long().cpu()


def check_rows(batch, n=CHECK_ROWS, seed=0):
    """Up to n row indices for the entry check, stratified: an equal share of each (kind, split, space group, Z')
    stratum present, drawn without replacement (a stratum smaller than its share is taken whole)."""
    kinds, splits = tag(batch, 'noised_kind'), tag(batch, 'noised_split')
    sg = torch.as_tensor(batch.sg_ind).reshape(-1).long().cpu()
    zp = torch.as_tensor(batch.z_prime).reshape(-1).long().cpu()
    key = ((kinds * 2 + splits) * 1000 + sg) * 10 + zp
    strata = torch.unique(key)
    share = max(1, n // max(len(strata), 1))
    rng = np.random.default_rng(seed)
    picked = []
    for s in strata.tolist():
        rows = torch.nonzero(key == s).flatten().numpy()
        picked.append(rows if len(rows) <= share else rng.choice(rows, share, replace=False))
    return np.sort(np.concatenate(picked))


def compare_terms(stored, fresh, energy_attr):
    """(worst relative-tolerance ratio, worst absolute miss, per-term worst misses) of a re-scored sample against
    its stored rows, over the energy attribute and CHECK_TERMS. A ratio <= 1 passes."""
    worst_ratio, worst_abs, per_term = 0.0, 0.0, {}
    for name in (energy_attr,) + CHECK_TERMS:
        a = torch.as_tensor(getattr(stored, name)).reshape(-1).double().cpu()
        b = torch.as_tensor(getattr(fresh, name)).reshape(-1).double().cpu()
        if a.shape != b.shape:
            raise ValueError(f"entry check: {name} has {a.numel()} stored and {b.numel()} fresh values")
        both = torch.isfinite(a) & torch.isfinite(b)
        if not bool((torch.isfinite(a) == torch.isfinite(b)).all()):
            return float('inf'), float('inf'), {name: float('inf')}
        d = (a[both] - b[both]).abs()
        tol = ABS_TOL + REL_TOL * a[both].abs()
        per_term[name] = float(d.max()) if d.numel() else 0.0
        if d.numel():
            worst_ratio = max(worst_ratio, float((d / tol).max()))
            worst_abs = max(worst_abs, float(d.max()))
    return worst_ratio, worst_abs, per_term


def select_rows(seed_energy, packing, kinds, splits, groups, T, s):
    """The load-time selection, as boolean masks over the file's rows:
      floor_ok   rows a condition's floor is taken over: training split, finite, inside the packing window
      train      rows the run trains on: training split, a kind in train_kinds, inside both windows
      heldout    held-out anchor groups' rows of those kinds, inside both windows
      anchors    clean training anchors inside both windows (a readout when anchors are not trained on)
    and the per-row floor. The energy window is window_kT x T above the floor (None: no energy window)."""
    lo, hi = s['packing_window']
    finite = torch.isfinite(seed_energy)
    in_pack = (packing >= lo) & (packing <= hi)
    floor_ok = (splits == SPLIT_TRAIN) & finite & in_pack
    floor = condition_floor(seed_energy, groups, floor_ok)
    if s['window_kT'] is None:
        in_window = finite
    else:
        in_window = finite & ((seed_energy - floor) <= s['window_kT'] * T)
    kind_ok = torch.zeros_like(kinds, dtype=torch.bool)
    for name in s['train_kinds']:
        kind_ok |= kinds == KIND_NAMES[name]
    good = in_pack & in_window
    return dict(floor_ok=floor_ok, train=(splits == SPLIT_TRAIN) & kind_ok & good,
                heldout=(splits == SPLIT_HELDOUT) & kind_ok & good,
                anchors=(splits == SPLIT_TRAIN) & (kinds == KIND_ANCHOR) & good), floor


def strip_analysis_helpers(b):
    """Drop what the analysis leaves on a batch that is not row data: the lazy space-group tables, and its
    dict-valued attributes (`computes` holds bound methods, which pickle the whole batch object they belong to, on
    whatever device it was scored: a file saved with them is larger and does not load on a CPU-only host)."""
    from energies.molecular_crystal import SG_LOOKUP_TABLES
    for attr in SG_LOOKUP_TABLES:
        if hasattr(b, attr):
            delattr(b, attr)
    for key in [k for k in b.keys() if isinstance(b[k], dict)]:
        delattr(b, key)
    return b


def harvest_mask(resid, log_r, temperature, min_abs_resid, floor=None):
    """[n] bool, the rows train.py::Modeller.harvest_scored_rows keeps: a finite reward and a finite TB residual of at
    least min_abs_resid nats in magnitude, or (given each row's condition floor, energy units) a finite reward whose
    energy -T log R is at or below it."""
    keep = torch.isfinite(log_r) & torch.isfinite(resid) & (resid.abs() >= float(min_abs_resid))
    if floor is not None:
        keep |= torch.isfinite(log_r) & (-log_r * float(temperature) <= floor)
    return keep
