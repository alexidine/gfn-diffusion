"""
Config invariants: the rules that are currently only comments.

WHY. Several relations between config keys are load-bearing, documented in
configs/mk_dev.yaml as prose, and enforced nowhere. Each one fails SILENTLY --
the run starts, trains, and produces numbers, with a mechanism quietly inert or a
schedule that never fires. That is the most expensive failure mode this codebase
has, because the result looks like an answer.

This module states them mechanically, as pure functions of a raw config dict. No
torch, no model construction: everything here is provable from the YAML alone,
which is what makes it cheap enough to run over the whole config corpus.

TWO SEVERITIES, deliberately distinguished:

  ERROR    a relation that is wrong under any circumstances -- the config
           contradicts itself, or guarantees a mechanism cannot work.
  BASELINE a project default that a run may knowingly depart from. Reported so
           the departure is deliberate, never failed on.

Checks that need to know which MODE is active (whether a metric exists on this
route, whether a bar applies to the ruler currently configured) are NOT here.
They arrive with the Phase 1 mode-safety work, once inactive-mode behavior is
pinned down. Guessing at them now would produce exactly the false reassurance
this module exists to prevent.

    python -m config_invariants <config.yaml> [<config.yaml> ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

ERROR = 'ERROR'
BASELINE = 'BASELINE'

# §9's current effective-optimization-batch baseline. A number to revise on
# profiling evidence, not a law -- hence BASELINE.
MIN_EFFECTIVE_BATCH = 1000


@dataclass(frozen=True)
class Violation:
    severity: str
    rule: str
    detail: str

    def __str__(self):
        return f'{self.severity:8s} {self.rule}: {self.detail}'


def _get(cfg: dict, dotted: str, default=None):
    node: Any = cfg
    for p in dotted.split('.'):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def _num(v) -> Optional[float]:
    """A float, or None for absent/`auto`/non-numeric. `auto` is resolved later
    from the primitives, so a check cannot judge it here and must abstain rather
    than guess."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


# ---------------------------------------------------------------------------
# The rules. Each returns a list of Violations; an empty list means it passed OR
# abstained, and abstention is reported separately by `check` so a rule that can
# never fire is visible rather than reassuring.
# ---------------------------------------------------------------------------

def growth_gain_below_growth_factor(cfg: dict) -> list[Violation]:
    """`batch_growth_min_throughput_gain` must be < `batch_growth_factor - 1`.

    At growth factor f, a jump multiplies work by f. Demanding a throughput gain
    at or above f-1 demands more than the jump can deliver, so EVERY jump is
    rejected and the batch freezes at batch_size -- with auto_batch_throughput_opt
    still reporting itself as on."""
    gain = _num(_get(cfg, 'batch_growth_min_throughput_gain'))
    factor = _num(_get(cfg, 'batch_growth_factor'))
    if gain is None or factor is None:
        return []
    if gain >= factor - 1.0:
        return [Violation(ERROR, 'growth_gain_below_growth_factor',
                          f'batch_growth_min_throughput_gain={gain} >= '
                          f'batch_growth_factor-1={factor - 1:.4g}. No jump can ever '
                          f'clear this bar; the batch freezes at batch_size while the '
                          f'throughput optimizer reports itself active.')]
    return []


def figs_period_fires(cfg: dict) -> list[Violation]:
    """`figs_period` must be a positive multiple of `eval_period`.

    Figure logging is checked inside the eval branch, so a non-multiple simply
    never coincides and no figures are ever produced."""
    figs = _num(_get(cfg, 'figs_period'))
    ev = _num(_get(cfg, 'eval_period'))
    if figs is None or ev is None or figs == 0 or ev == 0:
        return []
    if figs % ev != 0:
        return [Violation(ERROR, 'figs_period_fires',
                          f'figs_period={figs:g} is not a multiple of '
                          f'eval_period={ev:g}; the two schedules never coincide and '
                          f'no figures are logged.')]
    return []


def batch_ceiling_above_floor(cfg: dict) -> list[Violation]:
    """`max_batch_size` must be >= `batch_size`: the configured batch is the walk's
    floor, and a ceiling below it is unsatisfiable."""
    b = _num(_get(cfg, 'batch_size'))
    mx = _num(_get(cfg, 'max_batch_size'))
    if b is None or mx is None:
        return []
    if mx < b:
        return [Violation(ERROR, 'batch_ceiling_above_floor',
                          f'max_batch_size={mx:g} < batch_size={b:g}.')]
    return []


# Energy functions whose state carries cell angles, hence ang_dim > 0. Listed
# explicitly rather than inferred by excluding the toys: an unrecognised name
# must make the angular check ABSTAIN, not assume. A false pass costs a crash at
# model construction; a false failure costs trust in every other rule here.
_ANGULAR_ENERGY_FUNCTIONS = frozenset({
    'elj', 'lj', 'qlj', 'uma', 'mace', 'combo', 'silu', 'silu_energy',
    'simple_density', 'ellipsoid_overlap', 'crystal_harmonic', 'crystal_multiharmonic',
})


def dplr_is_well_formed(cfg: dict) -> list[Violation]:
    """DPLR forward covariance C = diag(d) + V V^T.

    `dplr_rho_max` caps the correlated variance fraction and is also an exact cap
    on pairwise |corr|, so it must be strictly < 1.

    The angular rule is a CONFIG-LEVEL PROXY for the real one. `models/gfn.py`
    asserts `dplr_mask_angular` whenever `ang_dim > 0 and dplr_rank > 0`, and
    ang_dim is only known once the model is built. Two config facts each imply
    ang_dim > 0: a crystal energy function (the state carries cell angles), or
    `periodic_centroids: true` (which extends the wrapped set to centroid dims).
    Either one is sufficient; neither is necessary, so this rule is sound but
    incomplete -- it never fires falsely, and a config it passes may still hit
    the construction assert."""
    out = []
    rank = _num(_get(cfg, 'model.dplr_rank'))
    if rank is None or rank <= 0:
        return out  # DPLR disabled -- the rest does not apply
    rho = _num(_get(cfg, 'model.dplr_rho_max'))
    if rho is not None and not (0.0 <= rho < 1.0):
        out.append(Violation(ERROR, 'dplr_is_well_formed',
                             f'model.dplr_rho_max={rho} must lie in [0, 1); it caps a '
                             f'variance fraction and a pairwise correlation.'))
    if _get(cfg, 'model.dplr_mask_angular') is not True:
        periodic = _get(cfg, 'model.periodic_centroids') is True
        crystal = _get(cfg, 'energy_function') in _ANGULAR_ENERGY_FUNCTIONS
        if periodic or crystal:
            why = 'periodic_centroids: true' if periodic else \
                  f"energy_function {_get(cfg, 'energy_function')!r} carries cell angles"
            out.append(Violation(
                ERROR, 'dplr_is_well_formed',
                f'model.dplr_mask_angular must be true when DPLR (rank {rank:g}) runs '
                f'with angular dims ({why}). models/gfn.py asserts this at '
                f'construction, so the run dies at init rather than training wrong.'))
    return out


def deactivate_threshold_is_sane(cfg: dict) -> list[Violation]:
    """A branch is skipped entirely when its frac falls below
    `deactivate_threshold`. At >= 1/3 the threshold can deactivate every branch of
    a three-way split at once, leaving a step with no loss."""
    out = []
    for path in ('controller.deactivate_threshold',):
        v = _num(_get(cfg, path))
        if v is not None and v >= 1.0 / 3.0:
            out.append(Violation(ERROR, 'deactivate_threshold_is_sane',
                                 f'{path}={v} >= 1/3; a three-way frac split can be '
                                 f'fully deactivated, leaving no active branch.'))
    for st in _get(cfg, 'protocol.stages', []) or []:
        if not isinstance(st, dict):
            continue
        v = _num(st.get('deactivate_threshold'))
        if v is not None and v >= 1.0 / 3.0:
            out.append(Violation(ERROR, 'deactivate_threshold_is_sane',
                                 f"stage {st.get('name')!r} deactivate_threshold={v} "
                                 f'>= 1/3.'))
    return out


def pinned_frac_matches_fracs(cfg: dict) -> list[Violation]:
    """For a `ratio` balance controller, `balance.pinned.<mode>` must equal
    `fracs.<mode>`.

    The pinned entry declares the frac the controller holds fixed while it steers
    the remaining pair. If it disagrees with the stage's own `fracs`, the two
    describe different splits and which one a step sees depends on evaluation
    order."""
    out = []
    for st in _get(cfg, 'protocol.stages', []) or []:
        if not isinstance(st, dict):
            continue
        bal = st.get('balance') or {}
        pinned = bal.get('pinned') or {}
        fracs = st.get('fracs') or {}
        if not isinstance(pinned, dict) or not isinstance(fracs, dict):
            continue
        for mode, val in pinned.items():
            pv, fv = _num(val), _num(fracs.get(mode))
            if pv is None or fv is None:
                continue
            if abs(pv - fv) > 1e-12:
                out.append(Violation(
                    ERROR, 'pinned_frac_matches_fracs',
                    f"stage {st.get('name')!r}: balance.pinned.{mode}={pv} != "
                    f'fracs.{mode}={fv}; the two declare different splits.'))
    return out


def effective_batch_meets_baseline(cfg: dict) -> list[Violation]:
    """The effective optimization batch should reach MIN_EFFECTIVE_BATCH.

    On fused steps `fused_grad_accum_min_samples` sets the floor directly:
    batches below it accumulate micro-steps until that many samples contribute.
    A value of 0 disables accumulation, in which case the floor is `batch_size`
    itself. BASELINE, not ERROR -- a run may depart from it deliberately."""
    accum = _num(_get(cfg, 'fused_grad_accum_min_samples'))
    batch = _num(_get(cfg, 'batch_size'))
    if batch is None:
        return []
    effective = batch if not accum else max(batch, accum)
    if effective < MIN_EFFECTIVE_BATCH:
        return [Violation(BASELINE, 'effective_batch_meets_baseline',
                          f'effective optimization batch {effective:g} < '
                          f'{MIN_EFFECTIVE_BATCH} (batch_size={batch:g}, '
                          f'fused_grad_accum_min_samples={accum if accum else 0:g}).')]
    return []


# Sensor kinds that actually move the learning rate. `none` (and an omitted
# block, which means the same thing silently) does not.
_ADAPTIVE_SENSOR_KINDS = frozenset({'ray', 'plateau', 'hyper'})
_LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')


def _is_auto(v) -> bool:
    return isinstance(v, str) and v.strip().lower() == 'auto'


def auto_lr_requires_an_adaptive_sensor(cfg: dict,
                                        auto_keys: Optional[list] = None) -> list[Violation]:
    """`auto` must yield to an adaptive scheme. A float must not.

    THE TWO SPELLINGS MEAN OPPOSITE THINGS and resolve to the same number, which
    is what makes the failure silent:

      auto   -> servo-managed. The rate is seeded at adaptive_lr.seed_lr and an
                adaptive sensor owns it from there.
      float  -> a fixed peak. It takes the warmup envelope and divergence
                handling, and `peak_scale` never applies to it
                (controller.py::_apply_lrs: `env * (peak if managed else 1.0)`).

    So `auto` with no adaptive sensor anywhere is a contradiction: the key is
    marked servo-managed, nothing ever moves peak_scale off 1.0, and the run
    trains at the seed for its whole life while the config reads as adaptive.
    Since the sensor is OPT-IN PER STAGE and an omitted block means `none`
    SILENTLY, this is reachable by leaving something out rather than by writing
    anything wrong.

    Checked per stage, not globally: a sensor on the terminal stage does nothing
    for a phase-1 LR.

    `auto_keys` EXISTS BECAUSE THE EVIDENCE IS DESTROYED BY RESOLUTION.
    `resolve_derived_config` overwrites the string `auto` with the seed float in
    place, so a caller running after it sees four ordinary numbers and this rule
    would find nothing to complain about -- silently, which is the failure mode it
    was written to catch. Such a caller passes the managed-key list it already
    computed. Callers working on raw YAML pass nothing and the keys are read from
    the config."""
    if auto_keys is None:
        auto_keys = [k for k in _LR_KEYS if _is_auto(cfg.get(k))]
    if not auto_keys:
        return []                      # every rate explicitly pinned: nothing to own

    stages = _get(cfg, 'protocol.stages', []) or []
    if not stages:
        return []                      # no protocol to reason about

    out = []
    for st in stages:
        if not isinstance(st, dict):
            continue
        sensor = st.get('lr_sensor')
        kind = sensor.get('kind') if isinstance(sensor, dict) else None
        if kind in _ADAPTIVE_SENSOR_KINDS:
            continue
        how = 'declares no lr_sensor' if sensor is None else f"declares lr_sensor kind {kind!r}"
        out.append(Violation(
            ERROR, 'auto_lr_requires_an_adaptive_sensor',
            f"stage {st.get('name')!r} {how}, but {', '.join(auto_keys)} "
            f"{'are' if len(auto_keys) > 1 else 'is'} `auto`. `auto` hands the "
            f"rate to a servo; with no adaptive sensor in this stage nothing "
            f"moves it, so it trains at adaptive_lr.seed_lr throughout while the "
            f"config reads as adaptive. Either declare a sensor (ray / plateau / "
            f"hyper) or write an explicit float, which takes the warmup envelope "
            f"and divergence handling without pretending to adapt."))
    return out


def ray_sensor_needs_a_coherent_stage(cfg: dict) -> list[Violation]:
    """`ray` is only coherent in a fused stage that trains replay TB.

    The probe draws from the replay buffer (so it needs stored trajectories) and
    scores with replay_loss_coeffs, so anywhere else it rates a loss nobody is
    optimising -- and does so silently, tallying skips rather than raising."""
    out = []
    base_replay_tb = _num(_get(cfg, 'replay_loss_coeffs.tb')) or 0.0
    for st in (_get(cfg, 'protocol.stages', []) or []):
        if not isinstance(st, dict):
            continue
        sensor = st.get('lr_sensor')
        if not (isinstance(sensor, dict) and sensor.get('kind') == 'ray'):
            continue
        if st.get('train_mode') != 'fused':
            out.append(Violation(
                ERROR, 'ray_sensor_needs_a_coherent_stage',
                f"stage {st.get('name')!r} declares lr_sensor kind 'ray' but "
                f"train_mode is {st.get('train_mode')!r}. The ray probe draws "
                f"from replay and scores replay_loss_coeffs; outside a fused "
                f"stage training replay TB it rates a loss nobody is optimising."))
            continue
        override = ((st.get('loss_coeffs') or {}).get('replay') or {}).get('tb')
        replay_tb = _num(override) if override is not None else base_replay_tb
        if not replay_tb:
            out.append(Violation(
                ERROR, 'ray_sensor_needs_a_coherent_stage',
                f"stage {st.get('name')!r} declares lr_sensor kind 'ray' but its "
                f"effective replay tb coefficient is 0 -- the probe would score a "
                f"loss this stage does not optimise."))
    return out


def periodic_centroids_needs_one_crystal_space_group(cfg: dict) -> list[Violation]:
    """`periodic_centroids` makes the model SPACE-GROUP SPECIFIC.

    Which centroid axes span the full cell width is a property of the space
    group, so the feature resolves a per-SG axis set and bakes it into the
    model's `expanded_dim`. Two space groups would have to be intersected, which
    "works" while quietly handing back a weaker or empty wrap instead of saying
    the config asked for something unsupported.

    It is also a molecular-crystal feature: a toy has no cell to wrap.

    Both are enforced at model construction, which is late -- on an MLIP route
    that is after the predictor has loaded. Checking at config level costs
    nothing and fails in the right place."""
    if _get(cfg, 'model.periodic_centroids') is not True:
        return []
    out = []
    sgs = list(_get(cfg, 'space_groups', []) or [])
    if len(sgs) != 1:
        out.append(Violation(
            ERROR, 'periodic_centroids_needs_one_crystal_space_group',
            f'model.periodic_centroids is on, which makes the model space-group '
            f'specific, so space_groups needs exactly one entry; got {sgs}.'))
    ef = _get(cfg, 'energy_function')
    if ef is not None and ef not in _ANGULAR_ENERGY_FUNCTIONS:
        out.append(Violation(
            ERROR, 'periodic_centroids_needs_one_crystal_space_group',
            f'model.periodic_centroids is a molecular-crystal feature but '
            f'energy_function is {ef!r}, which has no cell to wrap.'))
    return out


RULES = (
    auto_lr_requires_an_adaptive_sensor,
    periodic_centroids_needs_one_crystal_space_group,
    ray_sensor_needs_a_coherent_stage,
    growth_gain_below_growth_factor,
    figs_period_fires,
    batch_ceiling_above_floor,
    dplr_is_well_formed,
    deactivate_threshold_is_sane,
    pinned_frac_matches_fracs,
    effective_batch_meets_baseline,
)


def check(cfg: dict) -> list[Violation]:
    """Every rule against one raw config dict, worst severity first."""
    out = []
    for rule in RULES:
        out.extend(rule(cfg))
    return sorted(out, key=lambda v: (v.severity != ERROR, v.rule))


def errors(cfg: dict) -> list[Violation]:
    return [v for v in check(cfg) if v.severity == ERROR]


def _main():
    import argparse
    import sys
    import yaml

    ap = argparse.ArgumentParser(prog='config_invariants')
    ap.add_argument('paths', nargs='+')
    args = ap.parse_args()

    worst = 0
    for p in args.paths:
        with open(p, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            print(f'{p}: not a config mapping, skipped')
            continue
        vs = check(cfg)
        if not vs:
            print(f'{p}: ok')
            continue
        print(f'{p}:')
        for v in vs:
            print(f'  {v}')
        if any(v.severity == ERROR for v in vs):
            worst = 1
    sys.exit(worst)


if __name__ == '__main__':
    _main()
