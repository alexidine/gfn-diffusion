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
# `auto_lr_requires_an_adaptive_sensor` returns [] on absence while utils makes
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

    stages = active_stages(cfg)
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
    for st in active_stages(cfg):
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


# ---------------------------------------------------------------------------
# EXIT TRIGGERS. Two ways a declared exit condition ships dead, both found in
# the 2026-08-16 audit of prod0810 / qm9anchor_aug14 (docs/design/next_battery.md
# 1.1a and 1.3), and both provable from the YAML.
# ---------------------------------------------------------------------------

# The exit trigger's check cadence in train steps: train.py runs
# `protocol.tick()` inside `if self.step_ind % 10 == 0`. Hardcoded there, so it
# is a constant here too -- if that block moves, this must move with it.
EXIT_TICK_STEPS = 10


def _exit_metric_cadence(cfg: dict, metric: str) -> Optional[float]:
    """Train steps between successive WRITES of `metric`, or None if the config
    does not determine it.

    `eval/*` is written once per evaluation. `gates/*` is published from the
    same 10-step block that runs the tick, and `dir/*` rides the metric tracker,
    written by whichever branch produced the sample -- both are tick-rate or
    faster, so the tick is the binding cadence for them."""
    if not isinstance(metric, str):
        return None
    if metric.startswith('eval/'):
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


def _runs_vargrad(cfg: dict, stage: dict, mode: str) -> bool:
    return bool((_coeff(cfg, stage, mode, 'vg_lb') or 0.0) > 0
                or (_coeff(cfg, stage, mode, 'vg_lme') or 0.0) > 0)


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
            `prior_buffer.condition_block_m >= 2` (M distinct same-condition
            terminals per block). Either yields a group; neither is required
            when the other holds.

    Measured across every conditional battery that RAN (2026-08-17): aug14 and
    aug11 satisfy the bwd side via `repeats: 2.0` at `condition_block_m: 1`;
    aug13 satisfies it via `condition_block_m: 2` at `repeats: 1.0`. A rule
    written as a conjunction would reject two of the three, and a naive config
    diff reads those two spellings as a disagreement when they are the same
    constraint met two ways.

    ABSTAINS on any branch not running vg_lb/vg_lme -- on a TB route `repeats`
    means something else entirely and 1 is correct."""
    out = []
    cbm = _num(_get(cfg, 'buffers.prior_buffer.condition_block_m')) or 0.0
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
            if reps is not None and reps < 2 and cbm < 2:
                out.append(Violation(
                    ERROR, 'vargrad_needs_groups',
                    f"stage {name!r} runs bwd VarGrad at bwd repeats={reps:g} AND "
                    f"buffers.prior_buffer.condition_block_m={cbm:g}. The backward group "
                    f"needs ONE of the two >= 2 -- repeats gives K rollouts from one "
                    f"terminal, condition_block_m gives M same-condition terminals per "
                    f"block. With neither, every backward group is a singleton and the "
                    f"branch contributes no VarGrad gradient."))
    return out


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
    carries z_calibration off, all three tb_z_source keys `persistent`, and
    half_life_visits 28. Every conditional run that detonated in
    var_conditioning carried the unconditional trio. That includes one with a
    3.2k-step MLE warm start, which is what rules out "not enough MLE" as the
    explanation (findings F-042).

    BASELINE, not ERROR, and deliberately: this is a per-route default backed by
    three batteries, not a self-contradiction provable from the file. A run may
    depart from it knowingly -- but not by accident, which is what happened."""
    conditional = any(_get(cfg, k) is True for k in
                      ('embedding_conditioning', 'molecule_conditioning', 'vector_conditioning'))
    if not conditional:
        return []
    out = []
    src = 'qm9_aug11 / qm9_anchor_aug13 / qm9anchor_aug14, every conditional battery that ran'
    if _get(cfg, 'z_calibration.enabled') is True:
        out.append(Violation(
            BASELINE, 'conditional_z_settings_are_conditional',
            f'z_calibration.enabled is true on a CONDITIONAL route; every battery that '
            f'ran turned it off ({src}). Its parameters are documented as set for the '
            f'unconditional route, and on the conditional one it drives up to '
            f'max_steps_per_step Z-only steps into a per-condition flow NETWORK that '
            f'no stage has trained.'))
    for key in ('fwd_tb_z_source', 'bwd_tb_z_source', 'replay_tb_z_source'):
        if _get(cfg, f'condition_log_z.{key}') == 'learned':
            out.append(Violation(
                BASELINE, 'conditional_z_settings_are_conditional',
                f'condition_log_z.{key} is `learned` on a CONDITIONAL route; every '
                f'battery that ran used `persistent` ({src}) -- the conditional '
                f'persistent-Z regime the canonical config names in its own comment.'))
    hl = _num(_get(cfg, 'condition_log_z.half_life_visits'))
    if hl is not None and hl < 28.0:
        out.append(Violation(
            BASELINE, 'conditional_z_settings_are_conditional',
            f'condition_log_z.half_life_visits={hl:g} on a CONDITIONAL route; every '
            f'battery that ran used 28 ({src}). The canonical 7 is reasoned for a '
            f'ONE-condition run, where 7 visits == 7 steps; across a large library a '
            f'condition is revisited far more sparsely.'))
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
    lr_sensor shape (hyper needs an explicit beta, ray takes no other keys), the
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


RULES = (
    protocol_selector_resolves,
    every_protocol_parses,
    vargrad_needs_groups,
    conditional_z_settings_are_conditional,
    auto_lr_requires_an_adaptive_sensor,
    periodic_centroids_needs_one_crystal_space_group,
    ray_sensor_needs_a_coherent_stage,
    exit_patience_is_reachable,
    exit_bar_is_within_measured_range,
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
