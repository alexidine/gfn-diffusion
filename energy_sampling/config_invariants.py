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

import ast
import functools
import math
import os
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


# ---------------------------------------------------------------------------
# THE PROTOCOL SELECTOR -- one resolution point, imported by everything that
# needs the stage list.
#
# The canonical config carries EVERY protocol under `protocols:` and names the
# live one in `protocol:`. Switching route is then a one-word edit rather than a
# stage-list rewrite, and both alternatives sit in the same file where they can
# be compared.
#
# This function exists because the stage list used to be read from the literal
# path `protocol.stages` in six places. Four of those are validators, and
# `auto_lr_requires_lr_control` returns [] on absence while utils makes
# it a RAISING load gate -- so a restructure that moved the list would have
# silently disarmed the gate rather than tripping it. One resolver means the next
# move has one place to change.
# ---------------------------------------------------------------------------

PROTOCOL_SELECTOR = 'protocol'      # names the live protocol
PROTOCOL_LIBRARY = 'protocols'      # holds every protocol, keyed by name


def _member(node, name, default=None):
    """Read `name` off a dict OR an argparse.Namespace, so one resolver serves
    the raw-YAML callers and the trainer alike."""
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(name, default)
    return getattr(node, name, default)


def active_protocol_name(cfg) -> Optional[str]:
    sel = _member(cfg, PROTOCOL_SELECTOR)
    return sel if isinstance(sel, str) else None


def active_stages(cfg) -> list:
    """The stage specs the run will actually execute, or [] if unresolvable.

    Returns [] rather than raising: a validator's job is to report a broken
    config, not to die alongside it. The callers that MUST have stages
    (protocol.StageProtocol) raise on the empty result themselves, with a message
    about what is missing."""
    name = active_protocol_name(cfg)
    if name is None:
        return []
    entry = _member(_member(cfg, PROTOCOL_LIBRARY), name)
    return list(_member(entry, 'stages', []) or [])


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
# THE MODEL SECTION. mk_dev writes it by kind: `model: {kind, shared, sde, nf}` -- the model the trainer trains, the
# geometry both kinds read, the SDE policy's architecture, the normalizing flow's settings. The code reads the
# resolved form: `model` = shared + sde (the SDE builds from it, and the NF's geometry probe needs it too) and
# `policy` = {kind, nf}. utils.load_yaml resolves on load and `check` resolves before judging; a config already in the
# resolved form (a generated battery file, an older config) passes through unchanged.
# ---------------------------------------------------------------------------
MODEL_KINDS = ('sde', 'nf')
MODEL_SECTIONS = ('kind', 'shared', 'sde', 'nf')


def canonical_model_section(cfg):
    """The config with a sectioned `model:` block resolved (see above); anything else returned as it is."""
    if not isinstance(cfg, dict):
        return cfg
    model = cfg.get('model')
    if not isinstance(model, dict) or 'kind' not in model:
        return cfg
    unknown = set(model) - set(MODEL_SECTIONS)
    if unknown:
        raise ValueError(f"model: unknown sections {sorted(unknown)}; a sectioned model block holds "
                         f"{list(MODEL_SECTIONS)}")
    if model['kind'] not in MODEL_KINDS:
        raise ValueError(f"model.kind {model['kind']!r}: one of {list(MODEL_KINDS)}")
    if 'policy' in cfg:
        raise ValueError("a `policy` block beside a sectioned `model:` block: the model section carries the kind "
                         "and the NF settings")
    shared, sde = dict(model.get('shared') or {}), dict(model.get('sde') or {})
    both = sorted(set(shared) & set(sde))
    if both:
        raise ValueError(f"model.shared and model.sde both set {both}")
    out = dict(cfg)
    out['model'] = {**shared, **sde}
    out['policy'] = {'kind': model['kind'], 'nf': dict(model.get('nf') or {})}
    return out


# ---------------------------------------------------------------------------
# The rules. Each returns a list of Violations; an empty list means it passed OR
# abstained, and abstention is reported separately by `check` so a rule that can
# never fire is visible rather than reassuring.
# ---------------------------------------------------------------------------

def util_target_actuable(cfg: dict) -> list[Violation]:
    """A set `batch_util_target` must be able to act.

    The occupancy ladder (train.select_batch_size) only runs when growth is on
    and there is headroom above the base batch. A target written into a config
    where `grow_batch_size` is false, or where `max_batch_size <= batch_size`,
    is an inert flag: it loads clean, logs nothing, and reads as 'the occupancy
    constraint is being served' while nothing serves it.

    THE UNIT IS A FRACTION, and the range clause is what enforces it. This key
    was percent-valued until state 9 and the two spellings are not
    distinguishable by inspection: 0.6 is a legal 0.6% target under the old
    reading -- one that EVERY rung clears, so the ladder stops at the first rung
    and the constraint reports itself as served while serving nothing. That is
    the exact defect the rest of this rule exists to catch, and a range of
    (0, 100] could not see it. (0, 1] can: a leftover `60` now fails loudly
    instead of asking for 6000% occupancy and calling every batch INFEASIBLE."""
    target = _num(_get(cfg, 'batch_util_target'))
    if target is None or target <= 0:
        return []                       # unset / off is the shipping default
    out = []
    if not (0 < target <= 1):
        out.append(Violation(ERROR, 'util_target_actuable',
                             f'batch_util_target={target} is not a fraction in '
                             f'(0, 1] -- it is a fraction of the card, not a '
                             f'percentage (0.6 = 60% busy). A value above 1 is '
                             f'the pre-state-9 percent spelling.'))
    if not bool(_get(cfg, 'grow_batch_size')):
        out.append(Violation(ERROR, 'util_target_actuable',
                             f'batch_util_target={target} is set but grow_batch_size '
                             f'is false: the occupancy ladder never runs, so the '
                             f'target is an inert flag.'))
    batch = _num(_get(cfg, 'batch_size'))
    max_batch = _num(_get(cfg, 'max_batch_size'))
    if batch is not None and max_batch is not None and max_batch <= batch:
        out.append(Violation(ERROR, 'util_target_actuable',
                             f'batch_util_target={target} is set but max_batch_size='
                             f'{max_batch:g} <= batch_size={batch:g}: the ladder has '
                             f'one rung, so no occupancy-driven growth can happen.'))
    return out


def figs_period_fires(cfg: dict) -> list[Violation]:
    """On the conformer route, `figs_period` must be a positive multiple of `eval_period`.

    `train.py::Modeller.figs_due` reads `figs_period` (a step period) only on
    `conformer_torsions`; it is checked inside the eval branch, so a non-multiple
    never coincides and no figures are produced. Every other route reads
    `figs_every_n_evals` and refuses `figs_period` (step_cadence_keys_are_retired);
    there the rule returns []."""
    if _get(cfg, 'energy_function') not in _CONFORMER_ENERGY_FUNCTIONS:
        return []
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
    # latent_knn scores a REAL crystal parameterization (is_crystal True), so it
    # carries cell angles despite being scored in latent space.
    'latent_knn',
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
    for st in active_stages(cfg):
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
    for st in active_stages(cfg):
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


_LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')


def _is_auto(v) -> bool:
    return isinstance(v, str) and v.strip().lower() == 'auto'


def auto_lr_requires_lr_control(cfg: dict,
                                auto_keys: Optional[list] = None) -> list[Violation]:
    """`auto` must yield to something that can move it. A float must not.

    THE TWO SPELLINGS MEAN OPPOSITE THINGS and resolve to the same number, which
    is what makes the failure silent:

      auto   -> scale-managed. The rate is seeded at lr_control.seed_lr and the
                LR controller's scale (lr_control.schedule) multiplies it.
      float  -> a fixed rate. It takes divergence handling and the max_lr rail,
                and the controller's scale never applies to it
                (controller.py::_apply_lrs: `scale if managed else 1.0`).

    So `auto` with no `lr_control: {mode: fixed}` block is a contradiction: the
    key is marked managed, nothing sets the scale, and the run trains at the seed
    for its whole life.

    A RUN-LEVEL CHECK: `lr_control` is not declared per stage.

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

    node = _get(cfg, 'lr_control')
    if isinstance(node, dict) and node.get('mode', 'fixed') == 'fixed':
        return []                      # absent `mode` reads as 'fixed' (LRController)
    how = ('there is no `lr_control` block' if node is None
           else f"`lr_control.mode` is {(node or {}).get('mode')!r} (the only mode "
                f"is 'fixed'; 'bracket' was deleted 2026-10-28)")
    return [Violation(
        ERROR, 'auto_lr_requires_lr_control',
        f"{', '.join(auto_keys)} {'are' if len(auto_keys) > 1 else 'is'} `auto` "
        f"(scale-managed), but {how}. Nothing would set the scale, so the run "
        f"trains at lr_control.seed_lr throughout. Add `lr_control: {{mode: fixed, "
        f"schedule: [...]}}`, or write an explicit float for any rate meant to be "
        f"fixed.")]


def lr_control_is_well_formed(cfg: dict) -> list[Violation]:
    """`lr_control` as `controller.py::LRController` will read it.

    ERROR on: a `mode` other than 'fixed' (absent reads as 'fixed'; 'bracket' was
    deleted 2026-10-28), a negative `hard_failure.cruise_settle_steps`, no
    `schedule` (on the conformer route the LR_SCHEDULE_REPLACED_KEYS stand in for
    it), a malformed `schedule` (lr_schedule_problems), and a segment with
    `steps: null` other than a hold on a stage that declares no `max_steps`. The
    controller raises on each. A stage's own `lr_schedule` is judged where the
    stage is parsed (protocol.py::Stage, every_protocol_parses)."""
    node = _get(cfg, 'lr_control')
    if not isinstance(node, dict):
        return []
    rule = 'lr_control_is_well_formed'
    out = []
    mode = node.get('mode', 'fixed')
    if mode != 'fixed':
        out.append(Violation(
            ERROR, rule,
            f"lr_control.mode is {mode!r}; the only mode is 'fixed' ('bracket', the "
            f"candidate-grid search, was deleted 2026-10-28). Write `mode: fixed` "
            f"with an explicit schedule."))
    settle = _num(_get(cfg, 'lr_control.hard_failure.cruise_settle_steps'))
    if settle is not None and settle < 0:
        out.append(Violation(
            ERROR, rule,
            f"lr_control.hard_failure.cruise_settle_steps = {settle:g} is negative. "
            f"It is the number of steps discarded after a rate change before the "
            f"excursion bars are refitted at the new rate."))
    sched = node.get('schedule')
    conformer = _member(cfg, 'energy_function') in _CONFORMER_ENERGY_FUNCTIONS
    if sched is None:
        if not (conformer and any(k in node for k in LR_SCHEDULE_REPLACED_KEYS)):
            out.append(Violation(
                ERROR, rule,
                "lr_control needs an explicit lr_control.schedule -- the list of "
                "{to, steps, kind} segments that sets the multiplier on the auto "
                "rates over each stage."))
        return out
    both = [k for k in LR_SCHEDULE_REPLACED_KEYS if k in node]
    if conformer and both:
        out.append(Violation(
            ERROR, rule,
            f"lr_control carries both `schedule` and {both}, which `schedule` "
            f"replaced. Remove {both}."))
    problems = lr_schedule_problems(sched, 'lr_control.schedule')
    out.extend(Violation(ERROR, rule, p) for p in problems)
    if problems:
        return out
    open_ended = [i for i, s in enumerate(sched)
                  if _member(s, 'steps') is None and _member(s, 'kind') != 'hold']
    if open_ended:
        library = _member(cfg, PROTOCOL_LIBRARY) or {}
        for pname, proto in (library.items() if isinstance(library, dict) else ()):
            for st in _member(proto, 'stages', []) or []:
                if (_member(st, 'lr_schedule') is None
                        and _member(st, 'max_steps') is None):
                    out.append(Violation(
                        ERROR, rule,
                        f"lr_control.schedule[{open_ended[0]}] runs to the stage's "
                        f"max_steps (steps: null), and stage "
                        f"'{_member(st, 'name')}' of protocol '{pname}' declares "
                        f"no max_steps and no lr_schedule of its own."))
    return out


# ---------------------------------------------------------------------------
# THE LR SCHEDULE: `lr_control.schedule` (run level) and `stage.lr_schedule`
# (overrides it for that stage). An ordered list of segments
# {to: multiplier on seed_lr, steps: int >= 0 or null, kind}. Segment 0 starts
# at its own `to`; each later segment starts at the previous one's `to`; a
# null `steps` runs to the stage's max_steps; past the last segment the value
# holds. Evaluated by controller.py::lr_schedule_value.
# ---------------------------------------------------------------------------

#: hold = `to` throughout; linear = a + (b - a) p; geometric =
#: a (b / a)^p; cosine = b + (a - b)(1 + cos(pi p)) / 2, for a segment from a to
#: b at progress p in [0, 1].
LR_SCHEDULE_KINDS = ('hold', 'linear', 'geometric', 'cosine')

#: The `lr_control` keys `schedule` replaced (2026-10-28). Refused off the
#: conformer route (lr_schedule_replaces_burn_in_keys, LRController); on
#: energy_function conformer_torsions, whose configs still carry them, a block
#: with no `schedule` is read through legacy_lr_schedule.
LR_SCHEDULE_REPLACED_KEYS = ('burn_in_steps', 'burn_in_scale',
                             'promotion_ramp_steps', 'fixed_scale')

_LR_SEGMENT_KEYS = frozenset({'to', 'steps', 'kind'})


def lr_schedule_problems(node, where: str, max_steps=None,
                         judge_open_ended: bool = False) -> list:
    """Messages for everything wrong with one schedule; [] when well formed.

    Refused: not a non-empty list; a segment that is not a mapping, carries a
    key other than to/steps/kind, or lacks `to` or `kind`; a kind outside
    LR_SCHEDULE_KINDS; a `to` that is not a positive finite number; a `steps`
    that is not null or an integer >= 0; a first segment whose kind is not
    `hold` (it starts at its own `to`); a `steps: null` segment that is not the
    last. With `judge_open_ended`, also a non-hold `steps: null` segment when
    `max_steps` is None. Segments may be dicts or Namespaces."""
    if not isinstance(node, (list, tuple)) or not node:
        return [f"{where} must be a non-empty list of {{to, steps, kind}} "
                f"segments, got {node!r}"]
    out = []
    last = len(node) - 1
    for i, seg in enumerate(node):
        at = f'{where}[{i}]'
        keys = (set(seg) if isinstance(seg, dict)
                else set(vars(seg)) if hasattr(seg, '__dict__') else None)
        if keys is None:
            out.append(f"{at} must be a mapping {{to, steps, kind}}, got {seg!r}")
            continue
        bad = sorted(keys - _LR_SEGMENT_KEYS)
        if bad:
            out.append(f"{at} has unknown key(s) {bad}; a segment is {{to, steps, kind}}")
        kind = _member(seg, 'kind')
        if kind not in LR_SCHEDULE_KINDS:
            out.append(f"{at}.kind is {kind!r}; one of {list(LR_SCHEDULE_KINDS)}")
        to = _member(seg, 'to')
        if (isinstance(to, bool) or not isinstance(to, (int, float))
                or not math.isfinite(float(to)) or not float(to) > 0):
            out.append(f"{at}.to is {to!r}; a positive finite multiplier on seed_lr")
        steps = _member(seg, 'steps')
        if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int)
                                  or steps < 0):
            out.append(f"{at}.steps is {steps!r}; an integer >= 0, or null for the "
                       f"stage's max_steps")
        if i == 0 and kind in LR_SCHEDULE_KINDS and kind != 'hold':
            out.append(f"{at}.kind is {kind!r}; the first segment starts at its own "
                       f"`to`, so it must be a hold")
        if steps is None and i != last:
            out.append(f"{at}.steps is null (to the stage's max_steps) and segments "
                       f"follow it")
        if (judge_open_ended and steps is None and kind != 'hold'
                and max_steps is None):
            out.append(f"{at} is a {kind} segment with steps null on a stage with no "
                       f"max_steps")
    return out


def normalize_lr_schedule(node) -> list:
    """A well-formed schedule (lr_schedule_problems is empty) as a list of plain
    {'to': float, 'steps': int | None, 'kind': str} dicts."""
    return [{'to': float(_member(s, 'to')),
             'steps': None if _member(s, 'steps') is None else int(_member(s, 'steps')),
             'kind': str(_member(s, 'kind'))} for s in node]


def legacy_lr_schedule(burn_in_steps=3000, burn_in_scale=0.05,
                       promotion_ramp_steps=1000, fixed_scale=1.0) -> list:
    """The schedule the LR_SCHEDULE_REPLACED_KEYS describe: a hold at
    burn_in_scale for burn_in_steps, then a geometric move to fixed_scale over
    promotion_ramp_steps (0 = a jump). The defaults are LRController's for a run
    with no `lr_control` block."""
    return [{'to': float(burn_in_scale), 'steps': int(burn_in_steps), 'kind': 'hold'},
            {'to': float(fixed_scale), 'steps': int(promotion_ramp_steps or 0),
             'kind': 'geometric'}]


def replaced_lr_schedule_keys(cfg) -> list:
    """The LR_SCHEDULE_REPLACED_KEYS present under `lr_control`, off the
    conformer route; [] on energy_function conformer_torsions. Reads a raw dict
    or a Namespace."""
    if _member(cfg, 'energy_function') in _CONFORMER_ENERGY_FUNCTIONS:
        return []
    node = _member(cfg, 'lr_control')
    if node is None:
        return []
    present = node if isinstance(node, dict) else vars(node)
    return [k for k in LR_SCHEDULE_REPLACED_KEYS if k in present]


def lr_schedule_replaces_burn_in_keys(cfg: dict) -> list[Violation]:
    """`burn_in_steps`, `burn_in_scale`, `promotion_ramp_steps` and `fixed_scale`
    are replaced by `lr_control.schedule` (2026-10-28).

    A crystal-route rule rather than entries in utils._RETIRED_KEYS: the
    conformer configs still carry these keys, `preflight_config` is shared with
    that route, and a retired key raises at load. On `conformer_torsions` the
    rule returns []."""
    present = replaced_lr_schedule_keys(cfg)
    if not present:
        return []
    return [Violation(
        ERROR, 'lr_schedule_replaces_burn_in_keys',
        f"lr_control.{{{', '.join(present)}}}: replaced 2026-10-28 by "
        f"lr_control.schedule; LRController raises on them. The old burn-in and "
        f"ramp are `schedule: [{{to: <burn_in_scale>, steps: <burn_in_steps>, "
        f"kind: hold}}, {{to: <fixed_scale>, steps: <promotion_ramp_steps>, "
        f"kind: geometric}}]`.")]


#: `lr_control` keys deleted 2026-10-28 with `lr_control.mode: bracket` (the
#: candidate grid, its trials, boundary confirmation and densification, the
#: re-bracket clock, and the root's bias-correction check). Read by nothing.
LR_CONTROL_RETIRED_KEYS = (
    'candidate_scales', 'trial_steps', 'safety_rungs', 'repeat_every',
    'trial_settle_steps', 'logz_detour_nats', 'boundary_confirm_repeats',
    'boundary_densify', 'min_root_bias_correction',
)


def retired_lr_control_keys(cfg) -> list:
    """The LR_CONTROL_RETIRED_KEYS present under `lr_control`, off the conformer
    route; [] on energy_function conformer_torsions. Reads a raw dict or a
    Namespace (`controller.py::LRController._check_schema` raises on a
    non-empty result)."""
    if _member(cfg, 'energy_function') in _CONFORMER_ENERGY_FUNCTIONS:
        return []
    node = _member(cfg, 'lr_control')
    if node is None:
        return []
    present = node if isinstance(node, dict) else vars(node)
    return [k for k in LR_CONTROL_RETIRED_KEYS if k in present]


def lr_bracket_keys_are_retired(cfg: dict) -> list[Violation]:
    """The bracket's `lr_control` keys are deleted (2026-10-28).

    A crystal-route rule rather than entries in utils._RETIRED_KEYS: the
    conformer configs still carry some of these keys, `preflight_config` is
    shared with that route, and a retired key raises at load. On
    `conformer_torsions` the rule returns []."""
    present = retired_lr_control_keys(cfg)
    if not present:
        return []
    return [Violation(
        ERROR, 'lr_bracket_keys_are_retired',
        f"lr_control.{{{', '.join(present)}}}: deleted 2026-10-28 with "
        f"lr_control.mode 'bracket' and read by nothing; LRController raises on "
        f"them. Remove them.")]


def lr_probe_is_retired(cfg: dict) -> list[Violation]:
    """`lr_probe` -- Replay Racing -- is deleted, and an inert block must say so.

    A LOAD-TIME GATE RATHER THAN A RETIRED-KEY MIGRATION, following state 9's
    precedent for `batch_util_target`: a migration moves
    `project_state_version`, which every config in the tree then has to be
    restamped for, and there is nothing to migrate -- no tracked config outside
    the deleted `configs/race_*.yaml` ever carried the block. What is needed is
    only that a stray one cannot be SILENTLY IGNORED, which is what an unknown
    key would otherwise be.

    The design it configured trained candidate rates on RECORDED trajectories
    with fixed rewards: supervised regression toward frozen targets, a different
    objective from live on-policy progress, whose optimum sits elsewhere and
    which drifts toward the stability edge rather than the operating point
    (docs/design/lr_handoff_2026-08-21.md section 4).
    """
    if _get(cfg, 'lr_probe') is None:
        return []
    return [Violation(
        ERROR, 'lr_probe_is_retired',
        "`lr_probe` is deleted -- Replay Racing is retired (see "
        "docs/design/lr_handoff_2026-08-21.md section 4). Its HARVEST is what "
        "survived, as lr_larder.py, and it now feeds the generalised `ray` "
        "sensor: declare `lr_sensor: {kind: ray}` on the stage instead. Remove "
        "the block -- left in place it would configure nothing.")]


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


#: The conformer route is the one place a null data path is a DECLARED
#: configuration rather than a defect. It runs `ConformerModeller`, launched from
#: conformer_modeller.py rather than train.py, and that subclass overrides both
#: data-init methods: `init_mol_dataset` branches on `if path:` and falls back to
#: a condition built from the energy function, and `init_prior_dataset` SAMPLES
#: the fitted internal prior instead of loading a file. Measured over the config
#: corpus on 2026-09-09: 21 of the 26 configs carrying a null data path are this
#: energy function, and every one of them is legitimate.
_CONFORMER_ENERGY_FUNCTIONS = frozenset({'conformer_torsions'})

#: The paths `train.py` torch.loads with NO null branch, and the method that does
#: it. Both run during startup, before the first training step, in this order.
_UNCONDITIONALLY_LOADED_PATHS = (
    ('molecules_path', 'Modeller._load_condition_file, called from init_mol_dataset'),
    ('prior_path', 'Modeller.init_prior_dataset'),
)


def loaded_data_paths_are_not_null(cfg: dict) -> list[Violation]:
    """A null data path on the crystal route is a run that cannot start.

    `torch.load(None)` does not raise a path error. It raises

        AttributeError: 'NoneType' object has no attribute 'seek'. You can only
        torch.load from a file that is seekable.

    which names neither the key nor the config, and arrives after the energy
    function and the model have been built -- on an MLIP route, after the
    predictor has loaded. Nothing between generation and that line looks at the
    value, so the config validates clean, gets written, gets queued, and dies at
    startup.

    THE CASE THIS WAS WRITTEN FROM. `configs/problems.yaml`'s `latent_gaussian`
    entry carried `prior_path: null`, `molecules_path: null` AND
    `buffers.anchor_buffer.seed_source: prior_dataset` -- a self-contradiction
    provable from the file, since that seed source reads `self.prior_dataset`,
    which only exists if `prior_path` loaded. Every arm generated from it carries
    both nulls -- `configs/utilphase_sep09/lgauss.yaml` is one, generated the same
    day the fault was reported. The registry's own docstring calls that problem
    the reference workload for exact before/after comparison, so the entry that
    could not launch was the one a comparison was supposed to stand on.

    THE RULE IS NOT WRITTEN ON THE SEED SOURCE, deliberately. Anchoring it there
    would state a true relation and still miss most of the fault: both loads are
    UNCONDITIONAL, so `seed_source: generated` with a null `prior_path` is
    equally fatal and would have passed. What is provable from the file is the
    stronger and simpler claim -- these two keys are read by a `torch.load` with
    no null branch, so null is never a valid value for them here.

    ABSENT IS NOT NULL. A key that is simply missing is a partial dict, which is
    what generation and migration both produce, and this module's policy is that
    no rule fires on absence. `null` is an assertion, and it is the assertion
    that is wrong."""
    ef = _get(cfg, 'energy_function')
    if ef is None or ef in _CONFORMER_ENERGY_FUNCTIONS:
        # No energy_function means the route cannot be established from the file,
        # and a rule that cannot prove which loader runs has nothing to say.
        return []
    out = []
    for key, site in _UNCONDITIONALLY_LOADED_PATHS:
        if key in cfg and cfg[key] is None:
            out.append(Violation(
                ERROR, 'loaded_data_paths_are_not_null',
                f'{key} is null, but {site} torch.loads it with no null branch '
                f'on the {ef!r} route -- the run dies at startup with '
                f"AttributeError: 'NoneType' object has no attribute 'seek'."))
    # `prior_path` present-and-null already produced a violation above, so this
    # adds a SECOND statement about the same config rather than re-detecting it.
    if (_get(cfg, 'prior_path', 'ABSENT') is None
            and _get(cfg, 'buffers.anchor_buffer.seed_source') == 'prior_dataset'):
        out.append(Violation(
            ERROR, 'loaded_data_paths_are_not_null',
            'buffers.anchor_buffer.seed_source is prior_dataset, which seeds from '
            'self.prior_dataset -- an object that exists only if prior_path loaded. '
            'Point prior_path at a prior, or set seed_source: generated.'))
    return out


def grad_geometry_is_retired(cfg: dict) -> list[Violation]:
    """`grad_geometry` is deleted (2026-10-28) on the crystal route.

    It armed `Modeller._log_fused_gradient_geometry`, a periodic per-branch
    autograd.grad probe on fused steps that logged `fused_grad/*`. The method,
    its cadence gate and its metrics drain are gone, so nothing reads the block.

    A crystal-route rule rather than an entry in utils._RETIRED_KEYS: the
    conformer configs still carry the block, `preflight_config` is shared with
    that route, and a retired key raises at load. On `conformer_torsions` the
    rule returns []."""
    if _get(cfg, 'grad_geometry') is None:
        return []
    if _get(cfg, 'energy_function') in _CONFORMER_ENERGY_FUNCTIONS:
        return []
    return [Violation(
        ERROR, 'grad_geometry_is_retired',
        '`grad_geometry` is deleted (2026-10-28): the fused-step gradient '
        'geometry probe and its `fused_grad/*` metrics are gone from train.py, '
        'so the block configures nothing. Remove it.')]


def hold_dead_rows_is_retired(cfg: dict) -> list[Violation]:
    """`model.hold_dead_latent_rows` is retired (2026-10-28) on the crystal route.

    Modeller._resolve_dead_latent_rows always holds the rows that carry no degree
    of freedom for the configured (space group, Z'); there is no opt-out, so the
    key configures nothing. A crystal-route rule rather than an entry in
    utils._RETIRED_KEYS: the conformer configs still carry it. On
    `conformer_torsions` the rule returns []."""
    if _get(cfg, 'model.hold_dead_latent_rows') is None:
        return []
    if _on_conformer_route(cfg):
        return []
    return [Violation(
        ERROR, 'hold_dead_rows_is_retired',
        '`model.hold_dead_latent_rows` is retired (2026-10-28): dead latent rows '
        'are always held on this route, so the key configures nothing. Remove it.')]


def _present(node, name) -> bool:
    if node is None:
        return False
    return name in node if isinstance(node, dict) else hasattr(node, name)


def trajectory_length_problems(cfg) -> list[str]:
    """What is wrong with the trajectory-length keys of `cfg` (a dict or a Namespace tree, the model block
    resolved) off the conformer route: a top-level `eval_T` or an `integrator` block present (both retired
    2026-10-28: training and evaluation integrate `model.T`, mk_dev's `model.sde.T`, read by
    utils.trajectory_T), or, unless model.kind is nf, no positive integer `model.T`. [] on `conformer_torsions`
    and where no energy_function is set (train_conformer.py's conformer_dev), whose configs keep
    `integrator.T`. train.py::Modeller.init_gfn raises on a non-empty result."""
    ef = _member(cfg, 'energy_function')
    if ef is None or ef in _CONFORMER_ENERGY_FUNCTIONS:
        return []
    out = []
    if _present(cfg, 'eval_T'):
        out.append("eval_T is retired (2026-10-28): evaluation integrates model.sde.T steps, the training "
                   "length. Remove it.")
    if _present(cfg, 'integrator'):
        out.append("the integrator block is retired (2026-10-28): its T is model.sde.T. Move the value and "
                   "remove the block.")
    if _member(_member(cfg, 'policy'), 'kind', 'sde') != 'nf':
        T = _member(_member(cfg, 'model'), 'T')
        if T is None:
            out.append("model.sde.T is unset: the SDE policy's steps per rollout, train and eval.")
        elif isinstance(T, bool) or not isinstance(T, int) or T < 1:
            out.append(f"model.sde.T = {T!r}: a positive integer step count.")
    return out


def trajectory_length_is_model_T(cfg: dict) -> list[Violation]:
    """Off the conformer route `model.T` (mk_dev `model.sde.T`) is the one trajectory length
    (trajectory_length_problems). A crystal-route rule rather than entries in utils._RETIRED_KEYS: the
    conformer configs still set `integrator.T` and `eval_T`, and `preflight_config` is shared with that
    route."""
    return [Violation(ERROR, 'trajectory_length_is_model_T', p) for p in trajectory_length_problems(cfg)]


#: Step-unit eval-cadence keys replaced 2026-10-28 by counts of eval periods, and
#: their replacements: `figs_period` -> `figs_every_n_evals`,
#: `xcond_eval.period` -> `xcond_eval.every_n_evals`.
STEP_CADENCE_KEYS = {'figs_period': 'figs_every_n_evals',
                     'xcond_eval.period': 'xcond_eval.every_n_evals'}


def retired_step_cadence_keys(cfg) -> list[str]:
    """The keys of STEP_CADENCE_KEYS present in `cfg` (a dict or a Namespace tree) off the
    conformer route; [] on `conformer_torsions`, whose configs still set them."""
    if _member(cfg, 'energy_function') in _CONFORMER_ENERGY_FUNCTIONS:
        return []
    present = []
    for dotted in STEP_CADENCE_KEYS:
        node, parts = cfg, dotted.split('.')
        for part in parts[:-1]:
            node = _member(node, part)
        if node is not None and (parts[-1] in node if isinstance(node, dict)
                                 else hasattr(node, parts[-1])):
            present.append(dotted)
    return present


def step_cadence_keys_are_retired(cfg: dict) -> list[Violation]:
    """`figs_period` and `xcond_eval.period` are retired (2026-10-28) on the crystal route.

    Figures and the cross-condition matrix fire on `figs_every_n_evals` and
    `xcond_eval.every_n_evals`, counted in eval periods
    (`train.py::eval_cadence_fires`). `Modeller.init_gfn` raises on the same
    condition at startup.

    A crystal-route rule rather than an entry in utils._RETIRED_KEYS: the
    conformer configs still set `figs_period`, which `Modeller.figs_due` reads on
    that route, and `preflight_config` is shared with it. On `conformer_torsions`
    the rule returns []."""
    return [Violation(
        ERROR, 'step_cadence_keys_are_retired',
        f'{key} is retired (2026-10-28): set {STEP_CADENCE_KEYS[key]} (a count of '
        f'eval periods; null or 0 is off) and remove {key}.')
        for key in retired_step_cadence_keys(cfg)]


def ema_decay_is_conformer_only(cfg: dict) -> list[Violation]:
    """A non-null `ema_decay` is refused off the conformer route (2026-10-28).

    There `ema_model` IS `gfn_model` (utils.keeps_ema_copy): no EMA copy is
    built, saved or loaded, so a decay would configure nothing.
    `Modeller.init_gfn` raises on the same condition at startup.

    A crystal-route rule rather than an entry in utils._RETIRED_KEYS: the
    conformer route keeps its EMA copy and its configs and tests still set the
    key, and `preflight_config` is shared with that route. On
    `conformer_torsions` the rule returns []."""
    decay = _get(cfg, 'ema_decay')
    if decay is None or _get(cfg, 'energy_function') in _CONFORMER_ENERGY_FUNCTIONS:
        return []
    return [Violation(
        ERROR, 'ema_decay_is_conformer_only',
        f'ema_decay={decay!r}: this route has no EMA model (ema_model is gfn_model) '
        f'and Modeller.init_gfn refuses a non-null decay. Remove the key.')]


# ---------------------------------------------------------------------------
# EXIT TRIGGERS. Two ways a declared exit condition ships dead, both found in
# the 2026-08-16 audit of prod0810 / qm9anchor_aug14 (docs/design/next_battery.md
# 1.1a and 1.3), and both provable from the YAML.
# ---------------------------------------------------------------------------

# The exit trigger's check cadence in train steps: train.py runs
# `protocol.tick()` inside `if self.step_ind % 10 == 0`. Hardcoded there, so it
# is a constant here too -- if that block moves, this must move with it.
EXIT_TICK_STEPS = 10


#: gates published from the EVAL path, not the 10-step tick block. Their patience is
#: therefore denominated in eval_period. See _exit_metric_cadence.
_EVAL_CADENCE_GATES = frozenset({'progress_done', 'heldout_flat'})


def _exit_metric_cadence(cfg: dict, metric: str) -> Optional[float]:
    """Train steps between successive WRITES of `metric`, or None if the config
    does not determine it.

    `eval/*` is written once per evaluation. MOST `gates/*` are published from the
    same 10-step block that runs the tick, and `dir/*` rides the metric tracker,
    written by whichever branch produced the sample -- both are tick-rate or
    faster, so the tick is the binding cadence for them.

    THE EXCEPTION IS `_EVAL_CADENCE_GATES`, and it is not a special case so much as
    a correction. A gate is only as fresh as its INPUTS: gates/progress_done is
    published by Modeller.progress_metrics off the eval path, because the marginal
    and energy metrics it reads exist only there. Classifying it at tick rate would
    understate its patience cost by eval_period/10 -- 50x at the shipped 500 --
    and the obvious "fix" of publishing it every tick would be worse: it would
    republish a STALE verdict, and _advance_term counts fresh writes, so a single
    measurement would advance the streak many times. That is exactly the bug this
    whole rule exists to prevent."""
    if not isinstance(metric, str):
        return None
    if metric.startswith('eval/'):
        return _num(_get(cfg, 'eval_period'))
    if metric.startswith('gates/') and metric[len('gates/'):] in _EVAL_CADENCE_GATES:
        return _num(_get(cfg, 'eval_period'))
    return float(EXIT_TICK_STEPS)


def exit_patience_is_reachable(cfg: dict) -> list[Violation]:
    """An exit term's `patience` counts WRITES of its metric, not checks of it.

    protocol._advance_term only moves a streak on a tick where the metric was
    freshly written, because every value source it reads persists its last
    value: counting checks counted one sample many times, so a `patience: 5` on
    a metric written every 500 steps was cleared by a SINGLE clean write 50
    steps later. Patience is therefore denominated in the term's own metric
    cadence, and the same integer means different things on different terms of
    one `exit:` block.

    Two consequences, both readable off the YAML:

      ERROR     `patience * cadence` exceeds `epochs`. The term cannot reach
                its patience before the run ends, so the stage's exit is
                governed by its OTHER terms while the config reads as if this
                one participates. That is the shape 1.3 describes: a trigger
                whose declared conditions and effective conditions differ.
      BASELINE  a coarse-cadence term (eval/*) carrying patience > 1 sits in a
                block alongside tick terms whose identical integer costs
                cadence/10 times fewer steps. Legal, and worth stating, because
                nothing at the point of writing it says the units differ.

    NOT an error merely for being coarse. The engine handles a slow metric
    correctly now; it just costs `patience * eval_period` steps to satisfy."""
    epochs = _num(_get(cfg, 'epochs'))
    out = []
    for st in active_stages(cfg):
        if not isinstance(st, dict):
            continue
        for i, term in enumerate(st.get('exit') or []):
            if not isinstance(term, dict):
                continue
            metric = term.get('metric')
            patience = _num(term.get('patience'))
            if patience is None or patience <= 1:
                continue                # patience 1 is one measurement, always reachable
            cadence = _exit_metric_cadence(cfg, metric)
            if cadence is None or cadence <= 0:
                continue                # eval_period absent: nothing to reason about
            cost = patience * cadence
            where = f"stage {st.get('name')!r} exit term {i} ({metric}, patience {patience:g})"
            if epochs is not None and cost > epochs:
                out.append(Violation(
                    ERROR, 'exit_patience_is_reachable',
                    f'{where} needs {patience:g} writes of {metric} at one every '
                    f'{cadence:g} steps = {cost:g} train steps, but the run is '
                    f'{epochs:g} steps. The term can never reach its patience, so '
                    f'the stage exits on its remaining terms while the config reads '
                    f'as if this one gates.'))
            elif cadence > EXIT_TICK_STEPS:
                out.append(Violation(
                    BASELINE, 'exit_patience_is_reachable',
                    f'{where} is denominated in that metric\'s own cadence: '
                    f'{patience:g} writes at one every {cadence:g} steps = {cost:g} '
                    f'train steps. The same patience on a tick-cadence term in the '
                    f'same block costs {patience * EXIT_TICK_STEPS:g} steps.'))
    return out


# ---------------------------------------------------------------------------
# MEASURED METRIC RANGES -- the read-time R14 "bar inside its own scatter"
# check (analysis/checks.py, _R14_TAG_BAR), moved to config load.
#
# THIS TABLE IS DATA, AND DATA ROTS. Each entry is a measurement with a stated
# provenance, not a law: it says what a metric was observed to do on a named
# battery, and a bar underneath that floor is a bar no run in that battery would
# have cleared. Retire or re-measure an entry when the route it was measured on
# changes -- and prefer re-measuring to deleting, because an absent entry is a
# silent pass.
#
# WHICH IS WHY EVERY FINDING HERE IS BASELINE. A floor measured on a battery
# whose controls were railed describes the railed regime, not the metric; the
# configs most likely to trip this rule are the ones written to escape that
# regime. The table's job is to make a reader look, not to decide for them.
#
# Deliberately SMALL. A metric belongs here only when its floor has been
# measured across enough arms and ticks to be a property of the metric rather
# than of one run.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeasuredRange:
    """What a metric was observed to do, and where that was measured."""
    minimum: Optional[float]        # smallest value observed
    sigma: Optional[float]          # detrended scatter, R14's noise floor
    source: str                     # battery, arms, ticks, date
    maximum: Optional[float] = None  # largest value observed, where relevant


MEASURED_METRIC_RANGES: dict[str, MeasuredRange] = {
    # next_battery.md 1.1. `var_conditioning` declares `fwd/logw_std_within <
    # 6.0`; the measured minimum is 17.1, so no arm came within 2.9x of it in
    # 4,348 ticks. That block turned out to be VESTIGIAL rather than mis-set --
    # var_conditioning is terminal by design -- which is the case this rule is
    # most useful for and the reason its message names deletion as a remedy.
    'fwd/logw_std_within': MeasuredRange(
        minimum=17.1, sigma=9.9,
        source='qm9anchor_aug14, 6 arms, 4,348 ticks in var_conditioning, read 2026-08-16'),
}


def exit_bar_is_within_measured_range(cfg: dict) -> list[Violation]:
    """An exit bar must sit somewhere its metric has actually been observed.

    A `below: X` bar under the metric's measured MINIMUM asks for a value the
    metric has never taken, so the condition cannot fire. TWO DIFFERENT FAULTS
    PRODUCE THAT, and the rule cannot tell them apart -- which is why it reports
    the fact and names both remedies rather than assuming one:

      the bar is wrong        raise it to something the metric reaches
      the BLOCK is vestigial  the stage is terminal by design and the `exit:`
                              should be deleted, along with whatever stage sits
                              behind it

    The second is what `var_conditioning` turned out to be (next_battery.md
    1.1), and it is the case that most needs saying out loud: an exit block
    nobody intends to fire still reads as a live gate to every reader that walks
    the config, which is how the same stage got diagnosed as railed, as dead,
    and as terminal-by-design across three drafts.

    The read-time version of this is R14 (`analysis/checks.py`), which flags a
    bar inside the metric's own detrended sigma. R14 needs a run; by the time it
    speaks the arms are spent. Both conditions are reported, at two strengths:

      below the measured minimum   no run has come near the bar
      inside sigma of the minimum  whether it fires is decided by scatter

    ALWAYS BASELINE, NEVER ERROR, and the distinction is the whole reason the
    two severities exist. An ERROR is a config contradicting ITSELF -- provable
    from the file, true under any circumstances. A measured floor is EVIDENCE
    FROM ONE BATTERY, and evidence about a metric is not a property of the
    config in front of you: the same bar may be trivially reachable on another
    molecule set, temperature, or (W, T). `MIN_EFFECTIVE_BATCH` is the
    precedent -- a number to revise on evidence, not a law.

    The measurement here makes the point sharply. 17.1 was measured on runs
    where next_battery.md 1.1 finds FIVE controls railed or inert, and fixing
    those is the point of that section: the floor was read off a regime that is
    deliberately about to stop existing. A rule that blocked a run on it would
    forbid exactly the configs written to move it.

    ABSTAINS on any metric not in MEASURED_METRIC_RANGES, which is nearly all of
    them. A missing entry is not evidence the bar is fine."""
    out = []
    for st in active_stages(cfg):
        if not isinstance(st, dict):
            continue
        for i, term in enumerate(st.get('exit') or []):
            if not isinstance(term, dict):
                continue
            rng = MEASURED_METRIC_RANGES.get(term.get('metric'))
            if rng is None:
                continue
            where = f"stage {st.get('name')!r} exit term {i} ({term.get('metric')})"
            below = _num(term.get('below'))
            if below is not None and rng.minimum is not None:
                if below < rng.minimum:
                    out.append(Violation(
                        BASELINE, 'exit_bar_is_within_measured_range',
                        f'{where} requires < {below:g}, below the measured minimum '
                        f'{rng.minimum:g} ({rng.source}). No run in that battery came '
                        f'near it, so on that evidence the condition does not fire and '
                        f'any stage behind it is never entered. Three things this can '
                        f'mean: the bar wants raising, the exit block is vestigial and '
                        f'wants deleting (a stage terminal by design should not carry '
                        f'a gate that reads as live), or the run intends to reach a '
                        f'regime the measurement never saw.'))
                elif rng.sigma is not None and below < rng.minimum + rng.sigma:
                    out.append(Violation(
                        BASELINE, 'exit_bar_is_within_measured_range',
                        f'{where} requires < {below:g}, within one sigma '
                        f'({rng.sigma:g}) of the measured minimum {rng.minimum:g} '
                        f'({rng.source}); whether it fires is decided by scatter.'))
            above = _num(term.get('above'))
            if above is not None and rng.maximum is not None and above > rng.maximum:
                out.append(Violation(
                    BASELINE, 'exit_bar_is_within_measured_range',
                    f'{where} requires > {above:g}, above the measured maximum '
                    f'{rng.maximum:g} ({rng.source}).'))
    return out


def _coeff(cfg: dict, stage: dict, mode: str, key: str):
    """A stage's effective coefficient: its own override, else the base block."""
    override = ((stage.get('loss_coeffs') or {}).get(mode) or {})
    if key in override:
        return _num(override[key])
    return _num(_get(cfg, f'{mode}_loss_coeffs.{key}'))


# THE COEFFICIENTS THAT ARM CONDITION GROUPING, named in ONE place.
#
# Three of them now mean "a grouped variance objective reads this branch's
# condition groups", and they are NOT all on the branch they arm:
#   vg_lb / vg_lme  branch-local VarGrad flavours, read off that branch's block
#   pooled_vg       the CROSS-BRANCH term: it concatenates the forward and
#                   backward rows into ONE set of condition groups, so it arms
#                   grouping on BOTH branches while living on fwd_loss_coeffs
#                   alone. A gate that only widens its test on `blc` therefore
#                   still reads 0 for a pooled-only arm and still looks right.
#
# This list has now been wrong TWICE for the same reason -- vg_lme (2026-08-26)
# and pooled_vg (2026-08-28) -- each time because a caller open-coded the test
# instead of asking here, and each time the failure was silent: the blocked draw
# switched off, groups collapsed to singletons, and the term became a near-no-op
# with no error. `runs_grouped_vargrad` is the ONLY place that names them;
# tests/test_grouped_vargrad_predicate.py fails if a caller re-opens the list.
BRANCH_VARGRAD_COEFFS = ('vg_lb', 'vg_lme')
CROSS_BRANCH_VARGRAD_COEFF = 'pooled_vg'   # on fwd_loss_coeffs; arms BOTH branches


def runs_grouped_vargrad(branch_get, fwd_get) -> bool:
    """Does a grouped-variance objective read this branch's condition groups?

    Written over GETTERS rather than a config dict so the raw-YAML rules here and
    train.py's resolved-Namespace gates share one implementation instead of two
    copies that drift. Each getter takes a coefficient name and returns a number
    or None.
    """
    if any((branch_get(k) or 0.0) > 0 for k in BRANCH_VARGRAD_COEFFS):
        return True
    return (fwd_get(CROSS_BRANCH_VARGRAD_COEFF) or 0.0) > 0


def _runs_vargrad(cfg: dict, stage: dict, mode: str) -> bool:
    return runs_grouped_vargrad(
        lambda k: _coeff(cfg, stage, mode, k),
        lambda k: _coeff(cfg, stage, 'fwd', k))


def vargrad_needs_groups(cfg: dict) -> list[Violation]:
    """VarGrad needs a GROUP of >= 2 rows to centre over, per branch that runs it.

    The group centre replaces log Z, so it is estimated from the rows that share
    a group. A SINGLETON GROUP CONTRIBUTES NO GRADIENT AT ALL -- vg_loss is
    identically zero there (gflownet_losses.condition_grouped_empirical_z, and
    condition_group_stats' `vg_live_frac` exists to measure exactly this). So a
    misgrouped branch does not crash: it trains on a fraction of what it paid to
    roll out, and the run looks busy while learning from part of the batch.
    That is why this is a load-time rule and not something a reader spots.

    THE TWO BRANCHES REACH >= 2 BY DIFFERENT MEANS, which is why the second
    condition is a DISJUNCTION and cannot be written as `repeats >= 2`:

      fwd   only `repeats` can do it. Forward tiles share a condition and roll
            out to DISTINCT terminals, which is the cross-terminal group VarGrad
            needs. Nothing else supplies one.
      bwd   EITHER `repeats >= 2` (K backward rollouts from one terminal) OR
            `condition_block_m >= 2` (M distinct same-condition terminals per
            block). Either yields a group; neither is required when the other
            holds.

    Measured across every conditional battery that RAN (2026-08-17): aug14 and
    aug11 satisfy the bwd side via `repeats: 2.0` at `condition_block_m: 1`;
    aug13 satisfies it via `condition_block_m: 2` at `repeats: 1.0`. A rule
    written as a conjunction would reject two of the three, and a naive config
    diff reads those two spellings as a disagreement when they are the same
    constraint met two ways.

    Both terms resolve through `_coeff`, so a stage that overrides either one
    is judged on what IT runs rather than on the base block.

    ABSTAINS on any branch not running vg_lb/vg_lme -- on a TB route `repeats`
    means something else entirely and 1 is correct."""
    out = []
    for st in active_stages(cfg):
        if not isinstance(st, dict):
            continue
        name = st.get('name')
        if _runs_vargrad(cfg, st, 'fwd'):
            reps = _coeff(cfg, st, 'fwd', 'repeats')
            if reps is not None and reps < 2:
                out.append(Violation(
                    ERROR, 'vargrad_needs_groups',
                    f"stage {name!r} runs fwd VarGrad at fwd repeats={reps:g}. Forward "
                    f"tiling is the ONLY source of a forward group -- at repeats 1 every "
                    f"group is a singleton, vg_loss is identically zero, and the forward "
                    f"branch trains on nothing while reporting a loss. Set fwd "
                    f"repeats >= 2."))
        if _runs_vargrad(cfg, st, 'bwd'):
            reps = _coeff(cfg, st, 'bwd', 'repeats')
            cbm = _coeff(cfg, st, 'bwd', 'condition_block_m') or 0.0
            # a third source: the aligned per-condition draw (stage condition_draw)
            # serves prior_rows distinct same-condition terminals per condition
            cd = st.get('condition_draw')
            drawn = (_num(cd.get('prior_rows')) or 0.0) if isinstance(cd, dict) else 0.0
            if reps is not None and reps < 2 and cbm < 2 and drawn < 2:
                out.append(Violation(
                    ERROR, 'vargrad_needs_groups',
                    f"stage {name!r} runs bwd VarGrad at bwd repeats={reps:g} AND "
                    f"bwd condition_block_m={cbm:g}. The backward group "
                    f"needs ONE of the two >= 2 -- repeats gives K rollouts from one "
                    f"terminal, condition_block_m gives M same-condition terminals per "
                    f"block. With neither, every backward group is a singleton and the "
                    f"branch contributes no VarGrad gradient."))
    return out


# keys that act only through the SDE policy's machinery; under model.kind nf each is inert, raises mid-run, or
# (gradient_norm_clip auto) derives a value from SDE quantities. key -> the values a flow run may carry
FLOW_REFUSED_GLOBALS = {
    'compile_policy': (False, None),
    'traj_checkpoint': (False, None),
    'model.full_flow': (False, None),
    'model.force_drift_fwd': (False, None, 0),
    'model.force_drift_bwd': (False, None, 0),
    'model.state_atoms': (0, None, False),
    'model.state_force_input': (False, None),
}
# per-branch loss coefficients an NF run must leave at 0 (or the stated value), resolved through each stage
FLOW_ZERO_COEFFS = {'fwd': ('subtb', 'db', 'exploration_std', 'traj_grads', 'path_grad_last_k', 'reward_grads'),
                    'bwd': ('subtb', 'db', 'tbc'),
                    'replay': ('subtb', 'db', 'stored_force_k', 'resample_last_k', 'reward_grads')}
FLOW_REFUSED_ACTIONS = ('set_traj_checkpoint',)


def nf_policy_refuses_sde_only_settings(cfg: dict) -> list[Violation]:
    """Under model.kind nf (models/nf_policy.py: a one-step policy, log P_B = 0, no time grid, no drift or
    variance heads, no force terms) the SDE-only settings refuse at load: each would otherwise sit in the run's config
    looking live while doing nothing (compile_policy, traj_checkpoint, the force and state-atom keys, full_flow, tbc,
    backward repeats > 1, db as a second TB), raise at its first use mid-run (subtb, the forward path-gradient keys,
    replay stored forces and resampling), or derive a value from SDE quantities (gradient_norm_clip auto: the (W, T)
    rule fitted to SDE gradients). Absent policy block or kind sde: nothing to judge."""
    cfg = canonical_model_section(cfg)
    if _get(cfg, 'policy.kind', 'sde') != 'nf':
        return []
    out = []
    for key, ok in FLOW_REFUSED_GLOBALS.items():
        v = _get(cfg, key)
        if v not in ok:
            out.append(Violation(ERROR, 'nf_policy_refuses_sde_only_settings',
                                 f'{key} = {v!r} acts only through the SDE policy (allowed under model.kind nf: {ok})'))
    if _is_auto(_get(cfg, 'gradient_norm_clip', 'auto')):
        out.append(Violation(ERROR, 'nf_policy_refuses_sde_only_settings',
                             "gradient_norm_clip 'auto' is fitted to SDE gradients from (W, model.sde.T): give the "
                             "NF policy a number"))
    for stage in active_stages(cfg) or [{}]:
        name = stage.get('name', '(base)')
        for mode, keys in FLOW_ZERO_COEFFS.items():
            for key in keys:
                v = _coeff(cfg, stage, mode, key)
                if v not in (None, 0.0):
                    out.append(Violation(ERROR, 'nf_policy_refuses_sde_only_settings',
                                         f"stage '{name}': {mode} {key} = {v:g} is SDE-only (0 under model.kind nf)"))
        rep = _coeff(cfg, stage, 'bwd', 'repeats')
        if rep is not None and rep > 1:
            out.append(Violation(ERROR, 'nf_policy_refuses_sde_only_settings',
                                 f"stage '{name}': bwd repeats = {rep:g}: a one-step policy's backward copies are "
                                 f"identical (1 under model.kind nf)"))
        for act in list(stage.get('on_enter') or []) + list(stage.get('on_exit') or []):
            if str(act).split(':')[0] in FLOW_REFUSED_ACTIONS:
                out.append(Violation(ERROR, 'nf_policy_refuses_sde_only_settings',
                                     f"stage '{name}': action {act!r} acts only on the SDE policy"))
    return out


#: buffers.store, the NF-route row store (row_store.py; docs/wiki/row-store.md): every key and its code default.
#: An absent key reads its default; an unknown key is refused.
ROW_STORE_DEFAULTS = dict(enabled=False, bar=1.0, admit_reward_min=-200, cap_per_condition=50000, max_rows=200000,
                          eps=0.1, records_kept=8, kappa=1.0, floor_frac=0.25, uniform_frac=0.1, is_exponent=1.0,
                          refresh_chunk=8192)
#: the off-cadence rollout bars whose reading is a replay_buffer statistic the store never writes
ROW_STORE_REFUSED_TRIGGERS = ('drift_std_max', 'ess_min', 'val_gap_max')


def row_store_settings(node) -> Optional[dict]:
    """buffers.store merged over ROW_STORE_DEFAULTS when its `enabled` is true, else None. `node` is a raw config dict
    or the trainer's Namespace. Raises on unknown keys."""
    st = _member(_member(node, 'buffers'), 'store')
    if st is None:
        return None
    st = dict(st) if isinstance(st, dict) else dict(vars(st))
    unknown = sorted(set(st) - set(ROW_STORE_DEFAULTS))
    if unknown:
        raise ValueError(f"buffers.store: unknown keys {unknown} (known: {sorted(ROW_STORE_DEFAULTS)})")
    s = {**ROW_STORE_DEFAULTS, **st}
    return s if s['enabled'] is True else None


def _real(v) -> bool:
    return not isinstance(v, (bool, str)) and isinstance(v, (int, float)) and math.isfinite(v)


def _whole(v) -> bool:
    return _real(v) and int(v) == v


def row_store_problems(cfg: dict) -> list[tuple[str, str]]:
    """(severity, message) for every way an enabled buffers.store conflicts with the rest of the config. [] when the
    store is absent or off."""
    try:
        s = row_store_settings(cfg)
    except ValueError as e:
        return [(ERROR, str(e))]
    if s is None:
        return []
    out = []

    def err(msg):
        out.append((ERROR, msg))
    if _get(cfg, 'policy.kind', 'sde') != 'nf':
        err(f"buffers.store needs model.kind nf, got {_get(cfg, 'policy.kind', 'sde')!r}")
    if _get(cfg, 'energy_function') in _CONFORMER_ENERGY_FUNCTIONS:
        err('buffers.store is refused on the conformer route')
    if _get(cfg, 'temperature_conditioning', False) is True:
        err('buffers.store stores log R at one temperature: refused with temperature_conditioning')
    if _get(cfg, 'energy_config.prior_flow_path') is not None:
        err('buffers.store stores log R at admission: refused with energy_config.prior_flow_path (a lambda path)')
    if _get(cfg, 'policy.nf.atom_tokens', False) is True:
        err('buffers.store keeps no molecule graph per row: refused with model.nf.atom_tokens')
    if _get(cfg, 'buffers.replay_buffer.prioritise.enabled', False) is True:
        err('buffers.store draws replay by its own priority: set buffers.replay_buffer.prioritise.enabled false')
    if (_num(_get(cfg, 'buffers.replay_buffer.val_frac')) or 0) > 0:
        err('buffers.store holds no replay held-out split: set buffers.replay_buffer.val_frac 0')
    if _get(cfg, 'buffers.anchor_buffer.frozen', False) is not True:
        err('buffers.store seeds once from the prior: set buffers.anchor_buffer.frozen true')
    if 'harvested' in list(_get(cfg, 'noised_prior.train_kinds', ('child',)) or ()):
        err("buffers.store seeds data tags anchor and child only: remove 'harvested' from noised_prior.train_kinds")
    if _get(cfg, 'condition_log_z.rollout_condition_draw', 'iid') == 'under_drawn':
        err("buffers.store: condition_log_z.rollout_condition_draw 'under_drawn' reads replay_buffer occupancy")

    for st in active_stages(cfg):
        if not isinstance(st, dict):
            continue
        name = st.get('name', '?')
        if st.get('condition_draw'):
            err(f"stage {name!r}: condition_draw draws from prior_buffer and replay_buffer, not the store")
        if (_num(st.get('replay_warmup_rows')) or 0) > 0:
            err(f"stage {name!r}: replay_warmup_rows counts replay_buffer rows")
        if (_coeff(cfg, st, 'replay', 'mle') or 0) > 0:
            err(f"stage {name!r}: replay mle > 0 (the store's MLE rows are drawn by the bwd branch)")
        if (_coeff(cfg, st, 'fwd', CROSS_BRANCH_VARGRAD_COEFF) or 0) > 0:
            err(f"stage {name!r}: fwd {CROSS_BRANCH_VARGRAD_COEFF} > 0 (the store has no condition-blocked draw)")
        for mode in ('bwd', 'replay'):
            for key in BRANCH_VARGRAD_COEFFS:
                if (_coeff(cfg, st, mode, key) or 0) > 0:
                    err(f"stage {name!r}: {mode} {key} > 0 (the store has no condition-blocked draw)")
        if (st.get('flags') or {}).get('weighted_bwd_sampling'):
            err(f"stage {name!r}: flags.weighted_bwd_sampling reads prior_buffer losses")
        if st.get('coeff_schedule'):
            err(f"stage {name!r}: coeff_schedule moves the reward; the store keeps log R from admission")
        if ((st.get('balance') or {}).get('anneal_coeffs')):
            err(f"stage {name!r}: balance.anneal_coeffs moves the reward; the store keeps log R from admission")
        trig = st.get('fwd_rollout_triggers') or {}
        for k in ROW_STORE_REFUSED_TRIGGERS:
            if isinstance(trig, dict) and (_num(trig.get(k)) or 0) > 0:
                err(f"stage {name!r}: fwd_rollout_triggers.{k} reads a replay_buffer statistic the store never writes")
        if (st.get('lr_sensor') or {}).get('kind') == 'ray':
            err(f"stage {name!r}: lr_sensor kind ray draws from the larder, which keeps no store row")
        if ((st.get('flags') or {}).get('z_calibration') is True
                and _get(cfg, 'z_calibration.mode', 'rollout') == 'replay'):
            err(f"stage {name!r}: z_calibration.mode replay draws from replay_buffer, which the store never fills "
                f"(Modeller._z_replay_step returns without a step)")

    if not (_real(s['kappa']) and s['kappa'] >= 0):
        err(f"buffers.store.kappa must be a number >= 0, got {s['kappa']!r}")
    for k in ('floor_frac', 'uniform_frac', 'is_exponent'):
        if not (_real(s[k]) and 0 <= s[k] <= 1):
            err(f"buffers.store.{k} must lie in [0, 1], got {s[k]!r}")
    if not (_real(s['bar']) and s['bar'] >= 0):
        err(f"buffers.store.bar must be a number >= 0, got {s['bar']!r}")
    if not (_real(s['eps']) and s['eps'] > 0):
        err(f"buffers.store.eps must be a number > 0, got {s['eps']!r}")
    for k in ('cap_per_condition', 'max_rows'):
        if not (_whole(s[k]) and s[k] >= 0):
            err(f"buffers.store.{k} must be an integer >= 0, got {s[k]!r}")
    if s['records_kept'] is not None and not (_whole(s['records_kept']) and s['records_kept'] >= 1):
        err(f"buffers.store.records_kept must be an integer >= 1 or null, got {s['records_kept']!r}")
    if not (_whole(s['refresh_chunk']) and s['refresh_chunk'] >= 1):
        err(f"buffers.store.refresh_chunk must be an integer >= 1, got {s['refresh_chunk']!r}")
    if s['admit_reward_min'] is not None and not _real(s['admit_reward_min']):
        err(f"buffers.store.admit_reward_min must be a number or null, got {s['admit_reward_min']!r}")
    if s['records_kept'] is None:
        out.append((BASELINE, 'buffers.store.records_kept null: protected records are unbounded (one per condition '
                              'per admission call at most)'))
    return out


def row_store_is_well_formed(cfg: dict) -> list[Violation]:
    """buffers.store (row_store.py) on the raw config: row_store_problems. Modeller.init_row_store raises on the
    ERRORs, since this module only reports at load."""
    return [Violation(sev, 'row_store_is_well_formed', msg) for sev, msg in row_store_problems(canonical_model_section(cfg))]


def conditional_z_settings_are_conditional(cfg: dict) -> list[Violation]:
    """Three Z-side keys are documented as UNCONDITIONAL-route settings and must
    be switched when the route is conditional.

    `configs/mk_dev.yaml` says it in its own comments -- the condition_log_z
    block is "MONITORING ONLY ... it becomes load-bearing on the conditional
    route", and the z_calibration block's parameters "are set for the
    UNCONDITIONAL route". Neither statement was executable, so a conditional
    config spawned from the canonical one inherits all three silently.

    THE EVIDENCE IS THE RUN RECORD, NOT A DERIVATION. Every conditional battery
    that ran to completion -- qm9_aug11, qm9_anchor_aug13, qm9anchor_aug14 --
    carries z_calibration off and all three tb_z_source keys `persistent`. Every
    conditional run that detonated in var_conditioning carried the unconditional
    values. That includes one with a 3.2k-step MLE warm start, which is what
    rules out "not enough MLE" as the explanation (findings F-042).

    HALF_LIFE_VISITS WAS THE THIRD KEY AND IS NO LONGER ROUTE-DEPENDENT. Those
    batteries ran it at 28 against a canonical 7, so it read as a per-route
    setting; hyperslope_aug17 showed 7 is wrong on BOTH routes -- it makes
    ema_logw the last few batches everywhere, and only on the conditional route
    is the consequence a detonation rather than a noisy target. The default is
    now 200 on both, so what survives here is a floor on an explicit short value,
    not a route difference.

    BASELINE, not ERROR, and deliberately: this is a per-route default backed by
    three batteries, not a self-contradiction provable from the file. A run may
    depart from it knowingly -- but not by accident, which is what happened.

    ABSENCE IS JUDGED, NOT ABSTAINED ON, and that distinction is the whole point
    of the rule. Elsewhere this module abstains where it cannot know (see
    exit_bar_is_within_measured_range, which only judges metrics it has a measured
    range for). Here there is nothing to abstain about: both keys have a code
    fallback, both fallbacks are the UNCONDITIONAL value, so an unset key is not an
    unmade choice -- it is the wrong setting, silently. Judging only present values
    is what let all five configs/a100_stab_aug16 arms report clean while the two
    conditional ones resolved to the F-042 detonation regime, because their
    generator wrote `persistent` into the pre-migration home where nothing reads
    it. A rule that passes the exact config it was written to catch is not a
    conservative rule, it is a broken one."""
    conditional = any(_get(cfg, k) is True for k in
                      ('embedding_conditioning', 'molecule_conditioning', 'vector_conditioning'))
    if not conditional:
        return []
    out = []
    src = 'qm9_aug11 / qm9_anchor_aug13 / qm9anchor_aug14, every conditional battery that ran'
    sel = _get(cfg, 'protocol')
    stages = (_get(cfg, f'protocols.{sel}.stages') or []) if sel else []
    for st in stages:
        if isinstance(st, dict) and (st.get('flags') or {}).get('z_calibration') is True:
            out.append(Violation(
                BASELINE, 'conditional_z_settings_are_conditional',
                f"stage '{st.get('name')}' declares the z_calibration flag on a "
                f'CONDITIONAL route; every battery that ran left it off ({src}). Its '
                f'parameters are set for the unconditional route, and here it drives up '
                f'to max_steps_per_step Z-only steps into a per-condition flow NETWORK '
                f'that no stage has trained.'))
    # tb_z_source moved out of condition_log_z into {mode}_loss_coeffs, so the
    # question is no longer "what does the global say" but "what does the SELECTED
    # protocol's stage resolve to" -- base value overlaid with the stage override.
    # Checking the base alone would now fail every conditional config, since the
    # base is deliberately `learned` and the conditional stages override it.
    sel = _get(cfg, 'protocol')
    stages = (_get(cfg, f'protocols.{sel}.stages') or []) if sel else []
    for st in stages:
        if not isinstance(st, dict):
            continue
        overrides = st.get('loss_coeffs') or {}
        # Only the branches this stage actually trains: a train_mode: bwd stage
        # never reads fwd's Z source, and a fused stage skips any branch its fracs
        # zero out. Flagging those would be noise, and noise is what gets a
        # BASELINE rule ignored.
        tm = st.get('train_mode')
        fracs = st.get('fracs') or {}
        if tm == 'fused':
            live = [m for m in ('fwd', 'bwd', 'replay')
                    if not fracs or (_num(fracs.get(m)) or 0) > 0]
        elif tm in ('fwd', 'bwd', 'replay'):
            live = [tm]
        else:
            live = ['fwd', 'bwd', 'replay']
        for mode in live:
            over = overrides.get(mode) or {}
            eff = over.get('tb_z_source', _get(cfg, f'{mode}_loss_coeffs.tb_z_source'))
            # ABSENCE IS NOT ABSTENTION HERE. train.py:1105 falls back to
            # 'learned', so a config that sets the key nowhere trains in exactly
            # the regime this rule exists to catch. Judging only present values
            # made the rule blind to the most likely way to get it wrong -- a
            # config that wrote `persistent` into the PRE-MIGRATION home, where it
            # loads clean and is read by nothing.
            if eff is None:
                stale = _get(cfg, f'condition_log_z.{mode}_tb_z_source')
                where = (f'; it is set to {stale!r} at the pre-migration '
                         f'`condition_log_z.{mode}_tb_z_source`, which is read by nothing'
                         if stale is not None else '')
                out.append(Violation(
                    BASELINE, 'conditional_z_settings_are_conditional',
                    f"stage '{st.get('name')}' sets {mode}_loss_coeffs.tb_z_source "
                    f'NOWHERE on a CONDITIONAL route, so it resolves to the code '
                    f'fallback `learned` (train.py:1105){where}. Every battery that ran '
                    f'used `persistent` ({src}). Set it in that stage\'s loss_coeffs.'))
            elif eff == 'learned':
                out.append(Violation(
                    BASELINE, 'conditional_z_settings_are_conditional',
                    f"stage '{st.get('name')}' resolves {mode}_loss_coeffs.tb_z_source "
                    f'to `learned` on a CONDITIONAL route; every battery that ran used '
                    f'`persistent` ({src}) -- the conditional persistent-Z regime. Set it '
                    f"in that stage's loss_coeffs so selecting the protocol adopts it."))
    # THE ABSENCE BRANCH IS GONE, DELIBERATELY. It fired because the code default
    # was 7.0 -- the unconditional value -- so an unset key silently resolved to
    # the wrong setting. `buffer.DEFAULT_HALF_LIFE_VISITS` is now 200.0, safe on
    # this route by construction, so an absent key is no longer a defect and a
    # rule that still complained about it would be reporting a hazard that no
    # longer exists. What remains reachable, and still wrong, is an EXPLICIT
    # short value: measured 2026-08-17 (hyperslope_aug17, QM9 conditional, rate
    # pinned at 2e-4) an explicit 7 detonates var_conditioning at step 1560 while
    # 28 runs 2000 steps clean, a single-key difference. So the rule keeps its
    # teeth exactly where the hazard is.
    hl = _num(_get(cfg, 'condition_log_z.half_life_visits'))
    if hl is not None and hl < 28.0:
        out.append(Violation(
            BASELINE, 'conditional_z_settings_are_conditional',
            f'condition_log_z.half_life_visits={hl:g} on a CONDITIONAL route. Measured '
            f'2026-08-17 (hyperslope_aug17, QM9 conditional, rate pinned at 2e-4): an '
            f'explicit 7 detonates var_conditioning at step 1560 while 28 runs 2000 '
            f'steps clean, a single-key difference. The default is now 200 '
            f'(buffer.DEFAULT_HALF_LIFE_VISITS) and this bar sits at the lowest value '
            f'shown to survive, not at the default.'))
    return out


def protocol_selector_resolves(cfg: dict) -> list[Violation]:
    """`protocol` must name a protocol in `protocols` that has stages.

    THIS RULE EXISTS BECAUSE THE SELECTOR CREATED A NEW WAY TO GO QUIET. Every
    stage-scoped check reads the ACTIVE stage list, so a selector naming a
    protocol that is not defined resolves to zero stages -- and a rule with
    nothing to iterate reports nothing wrong. Measured while making the change:
    with the selector pointed at a missing name, the auto-LR gate stopped firing
    on a config it had rejected a moment earlier.

    One mistyped word therefore used to disarm every stage check at once. This
    rule is what makes that word fail loudly instead, so it must be checked
    FIRST and must never itself depend on the stage list."""
    library = _get(cfg, PROTOCOL_LIBRARY)
    selector = _get(cfg, PROTOCOL_SELECTOR)
    if library is None and selector is None:
        return []                       # a fragment with no protocol at all
    known = sorted(library) if isinstance(library, dict) else []
    if not isinstance(selector, str):
        return [Violation(
            ERROR, 'protocol_selector_resolves',
            f'`{PROTOCOL_SELECTOR}` must name one of the protocols in '
            f'`{PROTOCOL_LIBRARY}` ({known or "none defined"}); got '
            f'{selector!r}. Without it no stage is live and EVERY stage-scoped '
            f'check silently has nothing to inspect.')]
    if selector not in known:
        return [Violation(
            ERROR, 'protocol_selector_resolves',
            f'`{PROTOCOL_SELECTOR}: {selector}` names no protocol. Defined: '
            f'{known or "none"}. Every stage-scoped check would inspect an empty '
            f'list and report nothing wrong.')]
    if not active_stages(cfg):
        return [Violation(
            ERROR, 'protocol_selector_resolves',
            f'protocol {selector!r} defines no stages.')]
    return []


def every_protocol_parses(cfg: dict) -> list[Violation]:
    """Every protocol in the library must parse, not just the selected one.

    THE PROBLEM THIS SOLVES. Every other stage-scoped rule reads the ACTIVE stage
    list, so an inactive protocol is unexamined until it is selected -- and then
    it fails at load, on the switch, which is the worst moment to discover it.
    Switching route is meant to be one word.

    DELIBERATELY SHALLOW. This parses each stage through `protocol.Stage`, which
    is the validation the trainer already does: unknown keys, train_mode, flags,
    lr_sensor shape (ray takes only period/n_sub, retired kinds refuse), the
    balance kind. That catches the trivially-broken protocol. It does NOT check
    whether a stage's exit metrics are ever published, whether a handover's
    on_enter actions are coherent, or anything else that needs a run -- those are
    not cheap and are not attempted here.

    Cheap enough for load and for config generation: Stage parsing is plain
    Python, no torch, no model."""
    library = _get(cfg, PROTOCOL_LIBRARY)
    if not isinstance(library, dict):
        return []
    from protocol import Stage           # local: protocol imports this module back

    out = []
    for name, entry in sorted(library.items()):
        specs = _member(entry, 'stages') or []
        if not specs:
            out.append(Violation(ERROR, 'every_protocol_parses',
                                 f'protocol {name!r} defines no stages.'))
            continue
        seen = []
        for i, spec in enumerate(specs):
            try:
                st = Stage(spec, i)
            except Exception as e:
                out.append(Violation(
                    ERROR, 'every_protocol_parses',
                    f'protocol {name!r} stage {i} does not parse: '
                    f'{type(e).__name__}: {e}'))
                continue
            seen.append(st.name)
        dupes = {n for n in seen if seen.count(n) > 1}
        if dupes:
            out.append(Violation(
                ERROR, 'every_protocol_parses',
                f'protocol {name!r} repeats stage name(s) {sorted(dupes)}; the '
                f'trainer identifies the live stage BY NAME, so a duplicate makes '
                f'the run\'s position ambiguous.'))
    return out


def fwd_rollout_cadence_is_well_formed(cfg: dict) -> list[Violation]:
    """A stage running forward rollouts on 1 in N steps (`fwd_rollout_every`)
    relies on two things the rest of the config can silently undo.

    `flags.z_calibration` must be off: its `rollout` mode performs its own forward
    rollout + energy call per Z step, off a sensor that is frozen between
    rollouts, so it would fire on every skipped step -- restoring the cost the
    cadence removes and moving Z in the interval the design keeps frozen.
    `z_calibration.fill_threshold` must be > 0: that is the switch for
    `z_level_fill`, the ONLY thing pinning log Z at each rollout once the servo is
    off; 0 disables it, and an unpinned Z under rare rollouts is the failure mode
    the design exists to avoid (docs/design/rarer_rollouts.md, invariants 1-2).
    Stage.__init__ refuses the first case at load; this rule is the audit-path
    twin and adds the second.

    It also covers `fwd_rollout_triggers`, the off-cadence rollout bars: every
    bar must be a known name and >= 0, and they only have meaning on a stage that
    skips forward rollouts in the first place. `fwd_rollout_drift_max` is the
    retired spelling of `drift_std_max` and is refused, not ignored -- a dead
    trigger key reads as an armed trigger."""
    known = {'drift_std_max', 'ess_min', 'val_gap_max',
             'occupancy_min_batches'}
    out = []
    for st in active_stages(cfg):
        if not isinstance(st, dict):
            continue
        if 'fwd_rollout_drift_max' in st:
            out.append(Violation(ERROR, 'fwd_rollout_cadence_is_well_formed',
                                 f"stage {st.get('name')!r} sets fwd_rollout_drift_max; it "
                                 f"moved into fwd_rollout_triggers as 'drift_std_max'."))
        trig = st.get('fwd_rollout_triggers')
        if not isinstance(trig, dict) or not trig:
            continue
        for k, v in trig.items():
            n = _num(v)
            if k not in known:
                out.append(Violation(ERROR, 'fwd_rollout_cadence_is_well_formed',
                                     f"stage {st.get('name')!r}: fwd_rollout_triggers.{k} is "
                                     f"not a known bar; known are {sorted(known)}."))
            elif n is None or n < 0:
                out.append(Violation(ERROR, 'fwd_rollout_cadence_is_well_formed',
                                     f"stage {st.get('name')!r}: fwd_rollout_triggers.{k}={v!r} "
                                     f"must be a number >= 0 (0 = that bar is off)."))
        if any((_num(v) or 0) > 0 for v in trig.values())                 and (_num(st.get('fwd_rollout_every')) or 0) <= 0:
            out.append(Violation(ERROR, 'fwd_rollout_cadence_is_well_formed',
                                 f"stage {st.get('name')!r} sets fwd_rollout_triggers without "
                                 f"fwd_rollout_every > 0. The triggers only add rollouts to "
                                 f"steps a cadence skipped, so they are dead config here."))
    cadenced = [st for st in active_stages(cfg)
                if isinstance(st, dict) and (_num(st.get('fwd_rollout_every')) or 0) > 0]
    for st in cadenced:
        flags = st.get('flags') or {}
        if bool(flags.get('z_calibration', False)):
            out.append(Violation(ERROR, 'fwd_rollout_cadence_is_well_formed',
                                 f"stage {st.get('name')!r} sets fwd_rollout_every="
                                 f"{st.get('fwd_rollout_every')} with flags.z_calibration "
                                 f"true; the servo would call the energy function on every "
                                 f"skipped step. Set it false."))
    # A stage declaring fwd_z_sidecar pins Z(c) with the forward emp_z on every
    # rollout instead (Stage refuses the sidecar without emp_z > 0), and the fill
    # refuses a Z(c) head anyway (_z_fill_head_is_fillable), so the fill
    # requirement is waived there. The z_calibration refusal above is not.
    fill_pinned = [st for st in cadenced if not st.get('fwd_z_sidecar')]
    if fill_pinned:
        ft = _num(_get(cfg, 'z_calibration.fill_threshold'))
        if ft is None or ft <= 0:
            out.append(Violation(ERROR, 'fwd_rollout_cadence_is_well_formed',
                                 f'z_calibration.fill_threshold={ft} but a stage runs '
                                 f'fwd_rollout_every > 0; the fill is the only thing '
                                 f'pinning log Z at each rollout and 0/absent disables it.'))
    return out

def z_fill_mode_is_well_formed(cfg: dict) -> list[Violation]:
    """z_calibration.fill_mode is 'snap' or 'absorb'; fill_process_var and
    fill_moment_reset are non-negative numbers; fill_from_eval is off, report
    or fill. A typo here would surface as a ValueError on the first fill --
    hundreds of steps into a cluster leg."""
    out = []
    zc = cfg.get('z_calibration') or {}
    if not isinstance(zc, dict):
        return out
    mode = zc.get('fill_mode', 'snap')
    if mode not in ('snap', 'absorb'):
        out.append(Violation(ERROR, 'z_fill_mode_is_well_formed',
                             f"z_calibration.fill_mode={mode!r}; must be 'snap' or 'absorb'."))
    for key in ('fill_process_var', 'fill_moment_reset'):
        v = zc.get(key)
        if v is not None and (_num(v) is None or _num(v) < 0):
            out.append(Violation(ERROR, 'z_fill_mode_is_well_formed',
                                 f"z_calibration.{key}={v!r}; must be a number >= 0."))
    fe = zc.get('fill_from_eval', 'off')
    if fe not in ('off', 'report', 'fill'):
        out.append(Violation(ERROR, 'z_fill_mode_is_well_formed',
                             f"z_calibration.fill_from_eval={fe!r}; must be off, report or fill."))
    return out


def batch_root_forward_is_well_formed(cfg: dict) -> list[Violation]:
    """tb_z_source 'batch_root' (the level-blind forward policy step) is a
    FORWARD-branch mode for cadenced stages only: the batch root stands in for
    the learned Z in the forward TB loss so the forward branch can train the
    policy without carrying a level error, and z_level_fill keeps setting the
    head. On bwd/replay it is meaningless (those branches freeze Z and train
    against the head); without a cadence every step would train the policy
    against its own batch's level with nothing bounding how often."""
    out = []
    base = (cfg.get('fwd_loss_coeffs') or {})
    for mode in ('bwd', 'replay'):
        if (cfg.get(f'{mode}_loss_coeffs') or {}).get('tb_z_source') == 'batch_root':
            out.append(Violation(ERROR, 'batch_root_forward_is_well_formed',
                                 f"{mode}_loss_coeffs.tb_z_source='batch_root' -- forward only."))
    for st in active_stages(cfg):
        if not isinstance(st, dict):
            continue
        lc = st.get('loss_coeffs') or {}
        for mode in ('bwd', 'replay'):
            if (lc.get(mode) or {}).get('tb_z_source') == 'batch_root':
                out.append(Violation(ERROR, 'batch_root_forward_is_well_formed',
                                     f"stage {st.get('name')!r}: loss_coeffs.{mode}.tb_z_source="
                                     f"'batch_root' -- forward only."))
        fwd = dict(base); fwd.update(lc.get('fwd') or {})
        if fwd.get('tb_z_source') == 'batch_root':
            if (_num(st.get('fwd_rollout_every')) or 0) <= 0:
                out.append(Violation(ERROR, 'batch_root_forward_is_well_formed',
                                     f"stage {st.get('name')!r}: fwd tb_z_source 'batch_root' "
                                     f"needs fwd_rollout_every > 0."))
            if (_num(fwd.get('freeze_policy')) or 0) > 0.5:
                out.append(Violation(ERROR, 'batch_root_forward_is_well_formed',
                                     f"stage {st.get('name')!r}: fwd tb_z_source 'batch_root' with "
                                     f"freeze_policy on -- the branch would train nothing."))
    return out


def replay_seat_problems(stage, coeff, glob) -> list[str]:
    """What the replay-seat keys need from the RESOLVED config: the part a stage
    parse cannot see. protocol.Stage._parse_replay_seat refuses what the stage
    itself declares; this judges the stage's effective loss coefficients (its
    override, else the base block) and the global keys.

      pooled_source 'replay'  needs the pooled term on (else a dead key), a stage
                              whose replay branch can carry weight, an
                              UNprioritised replay draw (the pooled term uses the
                              replay rows unweighted, so a prioritised draw would
                              hand it rows from p with no IS correction), and the
                              stage's condition_draw: nothing else aligns replay's
                              conditions with bwd's.
      fwd_z_sidecar           needs the forward branch Z-only (freeze_policy >
                              0.5), emp_z > 0, and forward groups of >= 2 rows
                              (repeats >= 2 AND a condition- or tile-grouped
                              estimate to centre on; with vg_by_condition 0 and
                              no vg flavour on, emp_z has no log Z estimate at
                              all). The fwd frac floor (min_fracs.fwd, else
                              controller.min_mode_frac) must sit below the
                              deactivate threshold, or fwd trains in the mix too.
      replay_warmup_rows      must be below buffers.replay_buffer.max_size, or the
                              warm-up can never release and every step rolls out.

    Written over GETTERS, like runs_grouped_vargrad, so this rule and the
    trainer's refusal in Modeller.set_loss_coeffs share one implementation:
      stage   a parsed protocol.Stage
      coeff   (mode, key) -> the effective coefficient, raw (strings kept)
      glob    dotted global key -> its value, or None when absent
    Returns one message per problem; [] = clean."""
    from protocol import POOLED_SOURCES   # local: protocol imports this module back
    out = []

    src = coeff('fwd', 'pooled_source')
    src = 'fwd' if src is None else src
    pooled_on = (_num(coeff('fwd', CROSS_BRANCH_VARGRAD_COEFF)) or 0.0) > 0
    if src not in POOLED_SOURCES:
        out.append(f"fwd pooled_source={src!r}; must be one of {POOLED_SOURCES}.")
    elif src == 'replay':
        if not pooled_on:
            out.append(f"fwd pooled_source 'replay' with {CROSS_BRANCH_VARGRAD_COEFF} "
                       f"off; the source only picks the pooled term's first side, so "
                       f"it is a dead key here. Set it on the stage that runs the term.")
        if not getattr(stage, 'replay_trains', True):
            out.append("fwd pooled_source 'replay' on a stage whose replay branch can "
                       "never carry weight; the pooled term would read INERT every step.")
        if bool(glob('buffers.replay_buffer.prioritise.enabled')):
            out.append("fwd pooled_source 'replay' with buffers.replay_buffer.prioritise"
                       ".enabled true. The pooled term takes the replay rows unweighted, "
                       "so its groups would be drawn from the prioritised p with no IS "
                       "correction. Turn prioritise off for this route.")
        if not getattr(stage, 'condition_draw', None):
            out.append("fwd pooled_source 'replay' on a stage without condition_draw. bwd "
                       "is aligned to the forward batch only under the fwd source and "
                       "replay runs after bwd, so without the aligned per-condition draw "
                       "the pooled groups share a condition by chance collision only.")

    if getattr(stage, 'fwd_z_sidecar', False):
        fp = _num(coeff('fwd', 'freeze_policy'))
        if fp is None or fp <= 0.5:
            out.append(f"fwd_z_sidecar with fwd freeze_policy={fp}; at weight 1 outside "
                       f"the frac mix the forward loss would train the policy.")
        ez = _num(coeff('fwd', 'emp_z'))
        if ez is None or ez <= 0:
            out.append(f"fwd_z_sidecar with fwd emp_z={ez}; emp_z is the only Z(c) pin "
                       f"on the sidecar.")
        reps = _num(coeff('fwd', 'repeats'))
        grouped = ((_num(coeff('fwd', 'vg_by_condition')) or 0.0) > 0.5
                   or any((_num(coeff('fwd', k)) or 0.0) > 0 for k in BRANCH_VARGRAD_COEFFS))
        if reps is None or reps < 2 or not grouped:
            out.append(f"fwd_z_sidecar with fwd repeats={reps} and "
                       f"{'a' if grouped else 'NO'} grouped estimate (vg_by_condition or a "
                       f"vg flavour). emp_z's target needs forward groups of >= 2 rows: at "
                       f"repeats 1 every group is a singleton and the sidecar trains "
                       f"nothing, and with no grouped estimate emp_z has no log Z target.")
        deact = getattr(stage, 'deactivate_threshold', None)
        if deact is None:
            deact = _num(glob('controller.deactivate_threshold'))
        floor = (getattr(stage, 'min_fracs', None) or {}).get('fwd')
        if floor is None:
            floor = _num(glob('controller.min_mode_frac'))
        if deact is not None and floor is not None and floor >= deact:
            out.append(f"fwd_z_sidecar with a fwd frac floor {floor:g} >= deactivate "
                       f"threshold {deact:g}; fwd would train in the mix AND as the sidecar.")

    rows = int(getattr(stage, 'replay_warmup_rows', 0) or 0)
    if rows > 0:
        cap = _num(glob('buffers.replay_buffer.max_size'))
        if cap is not None and rows >= cap:
            out.append(f"replay_warmup_rows={rows} >= buffers.replay_buffer.max_size="
                       f"{cap:g}; the buffer can never hold that many trainable rows, so "
                       f"the warm-up never releases and every step rolls out.")
    return out


def _coeff_raw(cfg: dict, stage: dict, mode: str, key: str):
    """`_coeff` without the numeric filter, so string coefficients survive."""
    override = ((stage.get('loss_coeffs') or {}).get(mode) or {})
    if key in override:
        return override[key]
    return _get(cfg, f'{mode}_loss_coeffs.{key}')


def replay_seat_is_well_formed(cfg: dict) -> list[Violation]:
    """The audit-path twin of Modeller.set_loss_coeffs' refusal: every active
    stage's replay-seat keys against the resolved config (replay_seat_problems).
    A stage that does not parse is skipped here -- every_protocol_parses
    reports it."""
    from protocol import Stage           # local: protocol imports this module back
    out = []
    for i, st in enumerate(active_stages(cfg)):
        if not isinstance(st, dict):
            continue
        try:
            parsed = Stage(st, i)
        except Exception:
            continue
        for msg in replay_seat_problems(
                parsed, lambda mode, key, st=st: _coeff_raw(cfg, st, mode, key),
                lambda dotted: _get(cfg, dotted)):
            out.append(Violation(ERROR, 'replay_seat_is_well_formed',
                                 f"stage {parsed.name!r}: {msg}"))
    return out


def condition_draw_problems(stage, coeff, glob) -> list[str]:
    """What one stage's condition draws need from the RESOLVED config: the part
    protocol.Stage._parse_condition_draw cannot see.

      condition_draw          (the aligned per-condition draw) needs an
                              UNprioritised replay draw -- its rows are drawn
                              uniformly within the chosen conditions, so IS
                              weights from buffers.replay_buffer.prioritise would
                              divide by a measure the draw never used; an
                              effective condition_block_m of 0 on bwd AND replay
                              (the stage override, else the base block -- mk_dev's
                              base bwd value is 2), since both set the rows per
                              condition; and one space group x Z' per molecule,
                              because every draw re-runs condition_samples, which
                              redraws both, so the stored condition_id it groups
                              by would not be the one the rows train under.
      rollout_condition_draw  (global) a non-'iid' value refuses a stage declaring
                              flags.weighted_condition_sampling: both set the
                              forward rollout's condition draw.

    Written over getters, like replay_seat_problems (same arguments), so this
    rule and Modeller.set_loss_coeffs' refusal share one implementation. [] = clean."""
    out = []
    if getattr(stage, 'condition_draw', None):
        if bool(glob('buffers.replay_buffer.prioritise.enabled')):
            out.append("condition_draw with buffers.replay_buffer.prioritise.enabled true. "
                       "The aligned draw is uniform within its conditions, so the "
                       "prioritised IS weights would divide by a measure it never used. "
                       "Turn prioritise off for this route.")
        for mode in ('bwd', 'replay'):
            cbm = coeff(mode, 'condition_block_m')
            if cbm is not None and _num(cbm) != 0.0:
                out.append(f"condition_draw with an effective {mode} condition_block_m="
                           f"{cbm!r} (the stage's override, else the base block). Both set "
                           f"the rows per condition -- one switch only; set it 0 on the "
                           f"stage.")
        sgs, zps = glob('space_groups'), glob('z_primes')
        n_combos = (len(sgs) if isinstance(sgs, (list, tuple)) else 1) * \
                   (len(zps) if isinstance(zps, (list, tuple)) else 1)
        if n_combos > 1:
            out.append(f"condition_draw on a route with {n_combos} space-group x Z' "
                       f"combinations per molecule. Every draw re-runs condition_samples, "
                       f"which redraws them, so rows grouped by stored condition_id train "
                       f"under different conditions.")
    mode = glob('condition_log_z.rollout_condition_draw')
    if mode not in (None, 'iid') and bool((getattr(stage, 'flags', None) or {}).get(
            'weighted_condition_sampling', False)):
        out.append(f"condition_log_z.rollout_condition_draw={mode!r} with "
                   f"flags.weighted_condition_sampling: both set the forward rollout's "
                   f"condition draw. Drop the flag or set the draw to 'iid'.")
    return out


def rollout_condition_draw_problems(glob) -> list[str]:
    """condition_log_z.rollout_condition_draw is one of protocol's
    ROLLOUT_CONDITION_DRAWS (absent = 'iid'), and rollout_under_drawn_power is a
    finite number: > 0 under 'under_drawn' (0 would be 'iid' spelled another
    way), and at its default 1.0 under any other draw, where nothing reads it.
    `glob` as in condition_draw_problems. [] = clean."""
    from protocol import ROLLOUT_CONDITION_DRAWS   # local: protocol imports this module back
    out = []
    mode = glob('condition_log_z.rollout_condition_draw')
    if mode is not None and mode not in ROLLOUT_CONDITION_DRAWS:
        out.append(f"condition_log_z.rollout_condition_draw={mode!r}; must be one of "
                   f"{ROLLOUT_CONDITION_DRAWS}.")
    power = glob('condition_log_z.rollout_under_drawn_power')
    if power is not None:
        p = _num(power)
        if p is None or not math.isfinite(p):
            out.append(f"condition_log_z.rollout_under_drawn_power={power!r}; must be a "
                       f"finite number.")
        elif mode == 'under_drawn' and p <= 0:
            out.append(f"condition_log_z.rollout_under_drawn_power={p:g} under "
                       f"'under_drawn'; must be > 0 (0 weights every condition equally, "
                       f"which is the 'iid' draw).")
        elif mode != 'under_drawn' and p != 1.0:
            out.append(f"condition_log_z.rollout_under_drawn_power={p:g} with "
                       f"rollout_condition_draw={mode or 'iid'!r}; only 'under_drawn' reads "
                       f"it, so a non-default value is a dead key.")
    return out


def condition_draw_is_well_formed(cfg: dict) -> list[Violation]:
    """The audit-path twin of Modeller.set_loss_coeffs' refusal of the two
    condition draws: rollout_condition_draw_problems once, and
    condition_draw_problems for every active stage. A stage that does not parse
    is skipped here -- every_protocol_parses reports it."""
    from protocol import Stage           # local: protocol imports this module back
    glob = lambda dotted: _get(cfg, dotted)   # noqa: E731
    out = [Violation(ERROR, 'condition_draw_is_well_formed', msg)
           for msg in rollout_condition_draw_problems(glob)]
    for i, st in enumerate(active_stages(cfg)):
        if not isinstance(st, dict):
            continue
        try:
            parsed = Stage(st, i)
        except Exception:
            continue
        for msg in condition_draw_problems(
                parsed, lambda mode, key, st=st: _coeff_raw(cfg, st, mode, key), glob):
            out.append(Violation(ERROR, 'condition_draw_is_well_formed',
                                 f"stage {parsed.name!r}: {msg}"))
    return out


# ---------------------------------------------------------------------------
# THE CONFORMER ROUTE. conformer_modeller.py runs `ConformerModeller` over the
# crystal trainer's config schema, and a config ported from the crystal lineage
# carries keys that are inert there, refused only after the queue wait, or
# quietly replaced by a constructor default. Every rule below returns [] unless
# energy_function is conformer_torsions, so no crystal config changes verdict.
#
# A SET is a config with `molecules_path` set: ConformerModeller reads the
# condition set from that file (`_condition_set_molecules`) and builds a
# MultiConformerTorsions over it when it holds more than one molecule. Whether a
# set is a CARRIER -- members whose coordinate layouts differ, padded to one width
# -- is decided by the file's CONTENTS, which a torch-free rule cannot read. A
# hazard that exists only on a carrier, or only above some member count, is
# therefore BASELINE here: an ERROR must never fire falsely (dplr_is_well_formed
# states the same policy), and a set whose members share one layout is a real
# configuration (an equal-N rung, a held-out file). `config_snapshot --check`
# fails on both severities, so a canonical conformer config is held to all of
# them either way; generation refuses only the ERRORs.
# ---------------------------------------------------------------------------

#: The three flags that make the route conditional, as train.py's
#: log_xcond_metrics and conditional_z_settings_are_conditional read them.
_CONDITIONING_FLAGS = ('embedding_conditioning', 'molecule_conditioning',
                       'vector_conditioning')

#: Where the conformer energy's contract is read from: the files the route imports,
#: beside this module. PARSED, not imported -- both import torch.
_ENERGY_SAMPLING_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFORMER_TORSIONS_SRC = os.path.join(_ENERGY_SAMPLING_DIR, 'energies',
                                       'conformer_torsions.py')
_CONFORMER_MODELLER_SRC = os.path.join(_ENERGY_SAMPLING_DIR, 'conformer_modeller.py')

#: energy_config keys ConformerModeller drops that are still LEGAL on this route.
#: `temperature` is the crystal's; the conformer derives kT from log_temperature,
#: but utils.problem_slug reads energy_config.temperature to name the checkpoint.
_CONFORMER_TOLERATED_ENERGY_KEYS = frozenset({'temperature'})


def _on_conformer_route(cfg: dict) -> bool:
    return _get(cfg, 'energy_function') in _CONFORMER_ENERGY_FUNCTIONS


def _conformer_set(cfg: dict) -> bool:
    """`molecules_path` names a condition file. The loader branches on the path's
    truthiness, so an empty string is no set."""
    path = _get(cfg, 'molecules_path')
    return isinstance(path, str) and bool(path.strip())


def _conditional(cfg: dict) -> bool:
    return any(bool(_get(cfg, k)) for k in _CONDITIONING_FLAGS)


def _policy_kind(cfg: dict) -> str:
    """model.policy_kind as ConformerModeller._install_set_policy reads it:
    lower-cased, 'flat' when absent."""
    return str(_get(cfg, 'model.policy_kind', 'flat')).lower()


@functools.lru_cache(maxsize=None)
def _conformer_energy_contract() -> tuple:
    """(ConformerTorsions.__init__ parameter names, conformer_modeller's
    _NON_ENERGY_KEYS, ConformerTorsions.LEVELS), by AST parse of the source.

    THE SAME SETS THE RUN FILTERS WITH, NOT A COPY OF THEM. init_energy_function
    pops _NON_ENERGY_KEYS and filters the rest through
    `inspect.signature(ConformerTorsions.__init__)`; a hand-kept list here would be
    the thing that drifts. Raises when either file is missing or has lost the
    expected shape -- the rules that call this report that as a violation, so a
    check that could not run never reads as a pass."""
    def parse(path):
        with open(path, 'r', encoding='utf-8') as f:
            return ast.parse(f.read(), path)

    def assigned(body, name, where):
        for node in body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return ast.literal_eval(node.value)
        raise LookupError(f'no literal assignment to {name} in {where}')

    tree = parse(_CONFORMER_TORSIONS_SRC)
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == 'ConformerTorsions'), None)
    init = next((n for n in (cls.body if cls is not None else [])
                 if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
    if init is None:
        raise LookupError(f'ConformerTorsions.__init__ not found in {_CONFORMER_TORSIONS_SRC}')
    a = init.args
    params = frozenset(x.arg for x in a.posonlyargs + a.args + a.kwonlyargs) - {'self'}
    levels = tuple(assigned(cls.body, 'LEVELS', 'ConformerTorsions'))
    non_energy = frozenset(assigned(parse(_CONFORMER_MODELLER_SRC).body,
                                    '_NON_ENERGY_KEYS', _CONFORMER_MODELLER_SRC))
    return params, non_energy, levels


def _contract_unreadable(rule: str, e: Exception) -> list[Violation]:
    return [Violation(ERROR, rule,
                      f'could not read the conformer energy contract from '
                      f'{_CONFORMER_TORSIONS_SRC} / {_CONFORMER_MODELLER_SRC} '
                      f'({type(e).__name__}: {e}); this check did not run.')]


def conformer_dplr_is_off_on_a_set(cfg: dict) -> list[Violation]:
    """DPLR (model.dplr_rank > 0) on the conformer route, in two cases.

    SET POLICY -> ERROR. _install_set_policy refuses it: SetPolicy emits the
    low-rank factor rank-major while GFN.split_params reads it dim-major, so u_raw
    would be silently transposed. The refusal fires at init, after the queue wait.

    FLAT POLICY ON A SET -> BASELINE. On a carrier, ConformerGFN masks the pad
    columns only inside gauss_logprob, and the DPLR forward density takes the
    Woodbury path in GFN.fwd_gauss_logprob, which never calls it -- so pad columns
    enter log P_F and nothing reports it. The config cannot say whether the set is
    a carrier (section note above); a set whose members share one layout has no
    pads to leak, so this is reported rather than refused."""
    if not _on_conformer_route(cfg):
        return []
    rank = _num(_get(cfg, 'model.dplr_rank'))
    if rank is None or rank <= 0:
        return []
    kind = _policy_kind(cfg)
    if kind == 'set':
        return [Violation(
            ERROR, 'conformer_dplr_is_off_on_a_set',
            f'model.policy_kind set with model.dplr_rank {rank:g}. The set head emits '
            f'the low-rank factor rank-major and GFN.split_params reads it dim-major, '
            f'so _install_set_policy refuses this at init. Set dplr_rank: 0.')]
    if kind == 'flat' and _conformer_set(cfg):
        return [Violation(
            BASELINE, 'conformer_dplr_is_off_on_a_set',
            f'a flat policy over the condition set in molecules_path with '
            f'model.dplr_rank {rank:g}. If the set is a carrier, the DPLR forward '
            f'density bypasses ConformerGFN\'s pad mask and pad columns enter log P_F '
            f'silently. Set dplr_rank: 0 unless every member shares one layout.')]
    return []


def conformer_set_policy_is_conditioned(cfg: dict) -> list[Violation]:
    """A set policy over a condition SET must see the molecule, at a width its
    pooled readout can split.

    Without embedding_conditioning the set head is built from ONE molecule's static
    features. On a carrier _install_set_policy refuses; on a set whose members
    share one layout it builds the dense head bound to the reference member and
    scores every row through it -- finite, plausible, and the wrong molecule.
    Both conditional set heads split the embedding into two equal blocks and raise
    on an odd embedding_conditioning_dim, and _install_set_policy raises on an
    unset one; this moves both from init to generation."""
    if not _on_conformer_route(cfg) or _policy_kind(cfg) != 'set':
        return []
    out = []
    conditioned = bool(_get(cfg, 'embedding_conditioning'))
    if _conformer_set(cfg) and not conditioned:
        out.append(Violation(
            ERROR, 'conformer_set_policy_is_conditioned',
            'model.policy_kind set over the condition set in molecules_path with '
            'embedding_conditioning false. The set head would be built from one '
            'member\'s static features: refused at init on a carrier, and silently '
            'bound to the reference member otherwise. Set embedding_conditioning: true.'))
    if conditioned:
        dim = _num(_get(cfg, 'embedding_conditioning_dim'))
        if dim is None or dim <= 0 or dim != int(dim) or int(dim) % 2:
            out.append(Violation(
                ERROR, 'conformer_set_policy_is_conditioned',
                f"embedding_conditioning_dim={_get(cfg, 'embedding_conditioning_dim')!r} "
                f'under a conditional set policy. The pooled readout is two equal '
                f'blocks, so it must be a positive even integer, and it must match the '
                f'embeddings stored in molecules_path.'))
    return out


def conformer_level_and_force_field_are_stated(cfg: dict) -> list[Violation]:
    """energy_config must name a known `level` and a `force_field`.

    `level` is keyword-only with no default, so an absent or unknown one fails at
    construction: loudly, but after the queue wait. `force_field` is the silent
    half. Its constructor default is 'reference', so a config that omits it -- or
    misspells the key, which conformer_energy_config_keys_are_read reports --
    trains on the reference force field while saying nothing about it. Absence is
    judged rather than abstained on because the fallback is a live value, as in
    conditional_z_settings_are_conditional."""
    if not _on_conformer_route(cfg):
        return []
    rule = 'conformer_level_and_force_field_are_stated'
    try:
        _, _, levels = _conformer_energy_contract()
    except Exception as e:           # reported, never swallowed into a pass
        return _contract_unreadable(rule, e)
    ec = _get(cfg, 'energy_config')
    ec = ec if isinstance(ec, dict) else {}
    out = []
    level = ec.get('level')
    if level is None:
        out.append(Violation(
            ERROR, rule,
            f'energy_config.level is not set. ConformerTorsions takes it keyword-only '
            f'with no default, so the run dies at construction. Name one of {levels}; '
            f'`full` is the training target, the others are helper tiers.'))
    elif level not in levels:
        out.append(Violation(
            ERROR, rule,
            f'energy_config.level={level!r} is not one of ConformerTorsions.LEVELS '
            f'{levels}; construction raises.'))
    if ec.get('force_field') is None:
        out.append(Violation(
            ERROR, rule,
            "energy_config.force_field is not set. The ConformerTorsions default is "
            "'reference', so the run would train on the reference force field while "
            "nothing in the config says so. Write force_field: mmff or reference."))
    return out


def conformer_energy_config_keys_are_read(cfg: dict) -> list[Violation]:
    """Every energy_config key must reach something on this route.

    init_energy_function pops _NON_ENERGY_KEYS, filters the rest against
    ConformerTorsions.__init__ (which takes no **kwargs, deliberately) and DROPS
    whatever is left with one printed line. A crystal-lineage key is then inert
    while the config reads as setting it, and a misspelt one is worse:
    `forcefield: mmff` is dropped and the run trains on the constructor default.
    `temperature` is the one tolerated drop (_CONFORMER_TOLERATED_ENERGY_KEYS)."""
    if not _on_conformer_route(cfg):
        return []
    rule = 'conformer_energy_config_keys_are_read'
    ec = _get(cfg, 'energy_config')
    if not isinstance(ec, dict) or not ec:
        return []
    try:
        params, non_energy, _ = _conformer_energy_contract()
    except Exception as e:           # reported, never swallowed into a pass
        return _contract_unreadable(rule, e)
    unknown = sorted(set(ec) - params - non_energy - _CONFORMER_TOLERATED_ENERGY_KEYS)
    if not unknown:
        return []
    return [Violation(
        ERROR, rule,
        f'energy_config carries {unknown}, which is neither a ConformerTorsions '
        f'parameter nor a key ConformerModeller consumes (_NON_ENERGY_KEYS). '
        f'init_energy_function drops them with one printed line, so each is inert '
        f'here, and a misspelt key leaves its parameter at the constructor default. '
        f'Remove them, or fix the spelling.')]


def conformer_protocol_omits_snapshot_prior(cfg: dict) -> list[Violation]:
    """No active stage may name `snapshot_prior` on this route.

    The conformer route samples backward from a fitted InternalPrior, and
    protocol._snapshot_prior refuses the action when one is loaded -- at the
    phase-1 EXIT, after the whole MLE phase has run. The crystal train_prior
    stage carries it, so a literal port of a crystal protocol inherits it."""
    if not _on_conformer_route(cfg):
        return []
    out = []
    for st in active_stages(cfg):
        if not isinstance(st, dict):
            continue
        for where in ('on_enter', 'on_exit'):
            for action in (st.get(where) or []):
                if str(action).partition(':')[0].strip() == 'snapshot_prior':
                    out.append(Violation(
                        ERROR, 'conformer_protocol_omits_snapshot_prior',
                        f"stage {st.get('name')!r} {where} names snapshot_prior. It "
                        f"freezes the MLE policy as the prior model; this route's prior "
                        f"is the fitted InternalPrior, and protocol._snapshot_prior "
                        f"refuses the action at the phase-1 exit, after the MLE phase "
                        f"has run. Drop it (snapshot:<tag> keeps the checkpoint)."))
    return out


def conformer_prior_model_name_is_null(cfg: dict) -> list[Violation]:
    """`prior_model_name` is inert on this route and must be null.

    It is read only by Modeller.init_prior_dataset, which ConformerModeller
    overrides without calling super(). A named prior therefore loads nothing, and
    `skip_if: prior_loaded` -- the stage skip the crystal route sets the key to
    trip -- never holds, so a crystal-style phase-2 arm silently re-runs phase 1
    from its loaded weights."""
    if not _on_conformer_route(cfg):
        return []
    name = _get(cfg, 'prior_model_name')
    if name is None:
        return []
    return [Violation(
        ERROR, 'conformer_prior_model_name_is_null',
        f'prior_model_name={name!r} on the conformer route. ConformerModeller\'s '
        f'init_prior_dataset never reads it, so nothing loads and skip_if: '
        f'prior_loaded cannot hold -- the first stage runs anyway. Set it null.')]


def crystal_prior_model_is_retired(cfg: dict) -> list[Violation]:
    """No frozen prior MODEL exists off the conformer route.

    Modeller loads no `prior_model_name`, `snapshot_prior` builds nothing, and
    `skip_if: prior_loaded` never holds, so each is refused here rather than
    ignored. The prior buffer's only source is `anchors`: under 'prior_model'
    (the code default when the key is absent) _prior_churn_cycle finds no prior
    sampler and admits nothing. The conformer route keeps its own rules
    (conformer_prior_model_name_is_null, conformer_protocol_omits_snapshot_prior)
    and its InternalPrior source."""
    if _on_conformer_route(cfg):
        return []
    rule = 'crystal_prior_model_is_retired'
    out = []
    name = _get(cfg, 'prior_model_name')
    if name is not None:
        out.append(Violation(
            ERROR, rule,
            f'prior_model_name={name!r}: the frozen prior model is retired on this '
            f'route and nothing loads it. Set it null; warm-start the policy with '
            f'checkpoint_name + load_weights_only.'))
    pb = _get(cfg, 'buffers.prior_buffer')
    if isinstance(pb, dict):
        source = pb.get('source', 'prior_model')
        if source != 'anchors':
            out.append(Violation(
                ERROR, rule,
                f"buffers.prior_buffer.source={source!r} (absent = 'prior_model'): "
                f"'anchors' is the only source on this route; no prior model exists "
                f"to draw from."))
    for st in active_stages(cfg):
        if not isinstance(st, dict):
            continue
        if st.get('skip_if') == 'prior_loaded':
            out.append(Violation(
                ERROR, rule,
                f"stage {st.get('name')!r} skip_if: prior_loaded never holds (no prior "
                f"model is loaded). Use skip_if: weights_loaded with checkpoint_name + "
                f"load_weights_only."))
        for where in ('on_enter', 'on_exit'):
            for action in (st.get(where) or []):
                if str(action).partition(':')[0].strip() == 'snapshot_prior':
                    out.append(Violation(
                        ERROR, rule,
                        f"stage {st.get('name')!r} {where} names snapshot_prior, which is "
                        f"retired: no prior model is built. Drop it (snapshot:<tag> keeps "
                        f"the checkpoint)."))
    return out


def conformer_set_policy_is_not_scrambled(cfg: dict) -> list[Violation]:
    """`flags.scramble_conditions` cannot act on a set policy's forward half.

    The scramble permutes the conditioner's output at the conditioner-to-trunk
    seam, and on the set route that seam feeds only s_model -- read by P_B and the
    flow head -- while the set policy takes the true molecule through mol_cond. A
    stage declaring it trains a conditional P_F against a P_B shown scrambled
    conditions, and because the scramble also detaches the conditioner's output,
    the conditioner gets no gradient from that stage."""
    if not _on_conformer_route(cfg) or _policy_kind(cfg) != 'set':
        return []
    out = []
    for st in active_stages(cfg):
        if isinstance(st, dict) and (st.get('flags') or {}).get('scramble_conditions') is True:
            out.append(Violation(
                ERROR, 'conformer_set_policy_is_not_scrambled',
                f"stage {st.get('name')!r} sets flags.scramble_conditions under "
                f"model.policy_kind set. The set policy reads the molecule through "
                f"mol_cond, not the scrambled seam, so only P_B and the flow head see "
                f"the scramble and the conditioner is detached for the stage. Drop "
                f"the flag."))
    return out


def conformer_set_clips_energy(cfg: dict) -> list[Violation]:
    """A condition SET should set energy_config.energy_clip; absent is null.

    The per-condition log Z tracker trims floor(trim_frac x rows) from each side
    of a visit, which is zero whenever a visit holds fewer than 1/trim_frac rows
    (100 at the 0.01 the conformer configs carry). Past a handful of members every
    visit is that small, so one clash row at U = 1e4 kcal/mol (T = 1) enters its
    condition's evidence at about -1e4 nats untrimmed and drags that condition's
    TB centre for hundreds of visits. The clip bounds the row.

    The VALUE is not judged here. The clip compresses ABSOLUTE U, so it has to sit
    above every member's energy floor plus its thermal tail -- a census over the
    set, not a property of the file. BASELINE because the hazard needs enough
    members to starve each visit, and the member count is in the conditions file."""
    if not _on_conformer_route(cfg) or not _conformer_set(cfg):
        return []
    if _get(cfg, 'energy_config.energy_clip') is not None:
        return []
    return [Violation(
        BASELINE, 'conformer_set_clips_energy',
        'energy_config.energy_clip is null (absent = null, the constructor default) '
        'over the condition set in molecules_path. Once a visit holds fewer than '
        '1/condition_log_z.trim_frac rows per condition the tracker trims nothing, '
        'and one clash row moves that condition\'s log Z centre by tens of nats. Set '
        'the clip above every member\'s energy floor plus its thermal tail.')]


def conformer_set_xcond_is_off(cfg: dict) -> list[Violation]:
    """xcond_eval should be off on a conditional conformer SET.

    The cross-condition matrix scores terminals drawn under condition a through
    backward rollouts on condition b's graph, and ConformerGFN binds b's state mask.
    On a carrier each cell is then a density over b's valid columns with b's pads
    pinned, so the matrix's columns live on different coordinate sets and the
    dimension terms dominate the delta -- and it returns finite numbers. BASELINE
    because a set whose members share one layout is well posed, and the layout is
    in the file."""
    if not (_on_conformer_route(cfg) and _conformer_set(cfg) and _conditional(cfg)):
        return []
    if not bool(_get(cfg, 'xcond_eval.enabled')):
        return []
    return [Violation(
        BASELINE, 'conformer_set_xcond_is_off',
        'xcond_eval.enabled over the condition set in molecules_path. On a carrier '
        'each matrix cell scores a terminal under another member\'s state mask, so '
        'cells are densities over different coordinate sets and the result is finite '
        'and meaningless. Set xcond_eval.enabled: false.')]


def conformer_set_policy_is_not_compiled(cfg: dict) -> list[Violation]:
    """compile_policy off under a set policy. A WORKING ASSUMPTION, not a finding.

    Scope: model.policy_kind set on the conformer route. maybe_compile_policy runs
    at the end of the base init_gfn and compiles the trunk submodules in place;
    _install_set_policy then swaps an uncompiled set head into forward_policy and
    deep-copies the partly compiled model into the EMA. No conformer run has
    executed that path -- compile is Linux + CUDA only, so every local run is
    eager -- and suppress_errors turns a failed compile into a silent fallback.
    `auto` and `step` count as on, because they compile on the cluster; a quoted
    'off' is a truthy string and compiles as well.

    Revisit when a cluster smoke measures compile on this route (launch count and
    step time, bench.compile_rollout). BASELINE so that smoke stays expressible."""
    if not _on_conformer_route(cfg) or _policy_kind(cfg) != 'set':
        return []
    setting = _get(cfg, 'compile_policy')
    if not setting:                      # absent, null, false: eager
        return []
    return [Violation(
        BASELINE, 'conformer_set_policy_is_not_compiled',
        f'compile_policy={setting!r} under model.policy_kind set. The compile-then-swap '
        f'path (trunk compiled in init_gfn, set head installed after it) has never run '
        f'on this route and a failed compile degrades to eager silently. Working '
        f'assumption until a cluster smoke measures it: set compile_policy: false.')]


def energy_reference_is_consistent(cfg: dict) -> list[Violation]:
    """energy_config.energy_reference and condition_log_z.untrusted_z, judged together.

    untrusted_z 'global' stands ONE all-condition level in for every condition below
    min_visits. That is only a usable Z once energy_reference has removed each
    condition's depth from its log Z; without it the conditions' log Z differ by tens
    of nats. The reference itself is refused with temperature_conditioning, because the
    absolute reward floors undo E_ref/T at the run's single temperature. The trainer
    raises on all three at init; this reports them before a launch.

    Absence is judged: an unset untrusted_z is the code default 'head', an unset
    energy_reference is null -- both fine on their own."""
    out = []
    ref = _get(cfg, 'energy_config.energy_reference', None)
    unz = _get(cfg, 'condition_log_z.untrusted_z', 'head')
    if ref not in (None, 'seed_min'):
        out.append(Violation(ERROR, 'energy_reference_is_consistent',
                             f"energy_config.energy_reference {ref!r} is not one of null, 'seed_min'"))
    if unz not in ('head', 'global'):
        out.append(Violation(ERROR, 'energy_reference_is_consistent',
                             f"condition_log_z.untrusted_z {unz!r} is not one of 'head', 'global'"))
    if unz == 'global' and ref is None:
        out.append(Violation(ERROR, 'energy_reference_is_consistent',
                             "condition_log_z.untrusted_z 'global' needs energy_config.energy_reference: "
                             "without it one all-condition level stands in for conditions whose log Z "
                             "differ by their depth"))
    if ref is not None and _get(cfg, 'temperature_conditioning', False) is True:
        out.append(Violation(ERROR, 'energy_reference_is_consistent',
                             "energy_config.energy_reference is refused with temperature_conditioning"))
    return out


#: buffer.Z_FEED_MODES and the code defaults Modeller.init_condition_log_z reads (buffer imports torch, this module
#: does not; tests/config/test_z_feed_invariants.py pins the two copies equal)
Z_FEED_MODES = ('mixed', 'forward')
Z_FEED_DEFAULTS = dict(z_feed='mixed', eval_half_life_visits=2.0)


def z_feed_is_well_formed(cfg: dict) -> list[Violation]:
    """condition_log_z.z_feed is one of buffer.Z_FEED_MODES and condition_log_z.eval_half_life_visits is a finite
    number > 0 (Modeller.init_condition_log_z raises on either; this reports them before a launch). Absence is
    judged: unset keys are the code defaults 'mixed' and buffer.DEFAULT_EVAL_HALF_LIFE_VISITS."""
    out = []
    feed = _get(cfg, 'condition_log_z.z_feed', Z_FEED_DEFAULTS['z_feed'])
    hl = _get(cfg, 'condition_log_z.eval_half_life_visits', Z_FEED_DEFAULTS['eval_half_life_visits'])
    if feed not in Z_FEED_MODES:
        out.append(Violation(ERROR, 'z_feed_is_well_formed',
                             f"condition_log_z.z_feed {feed!r} is not one of {Z_FEED_MODES}"))
    if isinstance(hl, bool) or not isinstance(hl, (int, float)) or not math.isfinite(hl) or hl <= 0:
        out.append(Violation(ERROR, 'z_feed_is_well_formed',
                             f"condition_log_z.eval_half_life_visits {hl!r} is not a finite number > 0"))
    return out

def heldout_gate_is_well_formed(cfg: dict) -> list[Violation]:
    """`heldout_gate` parses through progress_metrics.parse_heldout_gate, the parser
    Modeller.__init__ runs (absent = the code defaults, progress_metrics._HELDOUT_GATE)."""
    from progress_metrics import parse_heldout_gate
    try:
        parse_heldout_gate(_get(cfg, 'heldout_gate'))
    except (TypeError, ValueError) as e:
        return [Violation(ERROR, 'heldout_gate_is_well_formed', str(e))]
    return []


RULES = (
    protocol_selector_resolves,
    every_protocol_parses,
    vargrad_needs_groups,
    conditional_z_settings_are_conditional,
    energy_reference_is_consistent,
    z_feed_is_well_formed,
    heldout_gate_is_well_formed,
    auto_lr_requires_lr_control,
    lr_control_is_well_formed,
    # every route but the conformer one; returns [] on energy_function conformer_torsions
    lr_bracket_keys_are_retired,
    # every route but the conformer one; returns [] on energy_function conformer_torsions
    lr_schedule_replaces_burn_in_keys,
    periodic_centroids_needs_one_crystal_space_group,
    loaded_data_paths_are_not_null,
    lr_probe_is_retired,
    grad_geometry_is_retired,
    # every route but the conformer one; returns [] on energy_function conformer_torsions
    hold_dead_rows_is_retired,
    # every route but the conformer one; returns [] on energy_function conformer_torsions or unset
    trajectory_length_is_model_T,
    # every route but the conformer one; returns [] on energy_function conformer_torsions
    step_cadence_keys_are_retired,
    ema_decay_is_conformer_only,
    exit_patience_is_reachable,
    exit_bar_is_within_measured_range,
    util_target_actuable,
    # the conformer route; returns [] off energy_function conformer_torsions
    figs_period_fires,
    batch_ceiling_above_floor,
    dplr_is_well_formed,
    deactivate_threshold_is_sane,
    pinned_frac_matches_fracs,
    effective_batch_meets_baseline,
    fwd_rollout_cadence_is_well_formed,
    z_fill_mode_is_well_formed,
    batch_root_forward_is_well_formed,
    replay_seat_is_well_formed,
    condition_draw_is_well_formed,
    nf_policy_refuses_sde_only_settings,
    # returns [] when buffers.store is absent or off
    row_store_is_well_formed,
    # every route but the conformer one; returns [] on energy_function conformer_torsions
    crystal_prior_model_is_retired,
    # the conformer route; each returns [] off energy_function conformer_torsions
    conformer_dplr_is_off_on_a_set,
    conformer_set_policy_is_conditioned,
    conformer_level_and_force_field_are_stated,
    conformer_energy_config_keys_are_read,
    conformer_protocol_omits_snapshot_prior,
    conformer_prior_model_name_is_null,
    conformer_set_policy_is_not_scrambled,
    conformer_set_clips_energy,
    conformer_set_xcond_is_off,
    conformer_set_policy_is_not_compiled,
)


def check(cfg: dict) -> list[Violation]:
    """Every rule against one raw config dict, worst severity first."""
    cfg = canonical_model_section(cfg)
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
