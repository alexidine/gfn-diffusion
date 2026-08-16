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


RULES = (
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
