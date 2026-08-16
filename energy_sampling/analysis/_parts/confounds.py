"""§4 -- the confounds named routinely. The only check that spans arms.

`run_all` calls this FIRST. A comparison across arms that are not comparable is
not a weaker result, it is not a result, so the arm table has to be settled
before any metric is shown rather than caveated afterwards.

Everything here is read from `config` and the `phase` series, so it answers on a
run that logged almost nothing -- which is exactly the run whose comparability
is in doubt.

WHAT IS DELIBERATELY NOT HERE. §4 names ten confounds; three of them do not
belong to this check:

  * 'a knob that was retired or inert in that tree' is R2's subject. Two checks
    detecting one condition can disagree about it, and then the reader has to
    adjudicate the tool.
  * 'another process on the GPU' is NOT READABLE from wandb output. The run
    record carries this process's utilisation, not the machine's tenancy, and a
    proxy built from it would manufacture findings. It stays something the
    reader has to know about the box.
  * 'the LR sitting in a different part of its cycle at read time' needs the LR
    series and a cycle model -- that is `features.py`'s oscillation extraction,
    not a config assertion.

A SINGLE RUN STILL GETS CHECKED. Its cross-arm subjects are skipped and the
result SAYS so, as a row: one arm is a fact about the battery, not an absence of
findings about it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import keys as K
from .base import (CheckResult, Context, Finding, State, as_float, context,
                   series)

_CONF_CHECK = '§4 confounds'

# A read taken within this many steps of a stage ENTRY is a read of the
# injection point rather than of the stage: at a transition the optimiser state
# is fresh, the LR ramp (`adaptive_lr.warmup_steps`) is still running, and log Z
# is still relevelling. Under this many steps, whatever the metrics say is the
# transient. It is a constant and not a fraction of the run because the
# transient's length is set by the ramp, not by how long the run went on for.
_CONF_MIN_STAGE_STEPS = 1000.0

_CONF_IDENTITY = frozenset(K.CFG_IDENTITY)

# What decides whether two arms started from the same place. T is here as well
# as in the per-run subject: two arms can each be self-consistent (T == eval_T)
# and still be incomparable to each other.
_CONF_START_KEYS = (K.CFG_PRIOR_PATH, K.CFG_CONTINUE_FROM_CHECKPOINT,
                    K.CFG_SEED, K.CFG_ENERGY_FUNCTION,
                    K.CFG_TRAIN_T, K.CFG_EVAL_T)

# A key absent from the config and a key present holding null are DIFFERENT and
# are rendered differently: the first is a config from another tree (the knob
# takes its default), the second is a knob explicitly set to nothing.
_CONF_MISSING = '<missing>'
_CONF_NULL = '<null>'

# Knob names spelled out in the sweep row. The exact count is in `numbers`, so
# the cap shortens the line without hiding the size of the sweep.
_CONF_SWEEP_NAMES_SHOWN = 24


# ---------------------------------------------------------------------------
# Reading one config entry
# ---------------------------------------------------------------------------

def _conf_get(config: dict, key: str) -> tuple:
    """`(present, value)`. Both halves are needed: `K._value` answers None for a
    key that is absent and for a key holding null, and those are different
    findings -- the first says this config came from a different tree."""
    return key in config, K._value(config, key)


def _conf_show(present: bool, value) -> str:
    if not present:
        return _CONF_MISSING
    return _CONF_NULL if value is None else str(value)


def _conf_equal(a, b) -> bool:
    """Config values compared for the sweep table.

    NaN equals NaN here. A yaml `.nan` reaching the default comparison makes
    that knob differ between every pair of arms, which reads as a sweep
    dimension nobody swept."""
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


def _conf_label(run) -> str:
    return str(getattr(run, 'name', None) or getattr(run, 'run_id', '?'))


def _conf_normalise(runs) -> list:
    """One Run, a list of them, or any iterable of them -> a list."""
    if runs is None:
        return []
    if hasattr(runs, 'config'):
        return [runs]
    return list(runs)


def _conf_knobs(config: dict) -> set:
    """Config keys that CONFIGURE the run. Identity keys are dropped: every arm
    differs in its name, and a sweep table that lists the name as a swept knob
    is listing the thing the sweep is indexed BY."""
    return {k for k in config if k not in _CONF_IDENTITY}


# ---------------------------------------------------------------------------
# The stage series
# ---------------------------------------------------------------------------

def _conf_stage_series(run):
    """`(steps, values)` for the stage metric, or None.

    A one-point series is refused. `base.series` falls back to the SUMMARY,
    which hands back a single point at `last_step`; read as a residence that
    says the run entered its stage this instant, and the barely-started flag
    then fires on a hole in the data instead of on a short stage."""
    got = series(run, K.STAGE_METRIC)
    if got is None:
        return None
    s, v = got
    m = np.isfinite(s) & np.isfinite(v)
    s, v = s[m], v[m]
    return (s, v) if len(s) >= 2 else None


def _conf_boundaries(s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Steps at which the stage metric CHANGED. Any change counts, in either
    direction -- a rewind that puts a run back into an earlier stage is a
    boundary for the reader in exactly the same way an advance is."""
    idx = np.nonzero(np.diff(v) != 0)[0]
    return s[idx + 1]


# ---------------------------------------------------------------------------
# Per-run subjects
# ---------------------------------------------------------------------------

def _conf_per_run(res: CheckResult, run, ctx: Optional[Context],
                  window: Optional[float]) -> None:
    cfg = run.config or {}
    label = _conf_label(run)

    # --- T. 'different problem, different T (T dominates; keep eval_T = train
    # T)'. A config fact, so it is answerable on a run that logged nothing.
    p_t, t = _conf_get(cfg, K.CFG_TRAIN_T)
    p_e, e = _conf_get(cfg, K.CFG_EVAL_T)
    ft, fe = as_float(t), as_float(e)
    subj = f'{label}/T'
    if np.isnan(ft) or np.isnan(fe):
        res.add(Finding(_CONF_CHECK, subj, State.UNREADABLE,
                        f'{K.CFG_TRAIN_T}={_conf_show(p_t, t)}  '
                        f'{K.CFG_EVAL_T}={_conf_show(p_e, e)}'))
    else:
        nums = {K.CFG_TRAIN_T: ft, K.CFG_EVAL_T: fe}
        if ft != fe:
            res.add(Finding(_CONF_CHECK, subj, State.FLAG,
                            f'{K.CFG_EVAL_T} is not {K.CFG_TRAIN_T} -- the run '
                            f'is evaluated on a different integrator than the '
                            f'one it trains on', nums))
        else:
            res.add(Finding(_CONF_CHECK, subj, State.OK, '', nums))

    # --- code version. Absent is a FLAG in its own right: without the stamp,
    # drift against a sibling cannot be ruled out, and §4 opens with drift.
    commit = K.git_commit(cfg)
    subj = f'{label}/code_version'
    if commit:
        res.add(Finding(_CONF_CHECK, subj, State.OK, commit))
    else:
        res.add(Finding(_CONF_CHECK, subj, State.FLAG,
                        'no commit stamp in the config -- version drift against '
                        'another arm cannot be ruled out'))

    # --- start condition, REPORTED not judged. Resuming is normal here; what
    # makes it a confound is a sibling that started somewhere else, and that
    # comparison is a battery subject.
    p_c, cont = _conf_get(cfg, K.CFG_CONTINUE_FROM_CHECKPOINT)
    p_n, ckpt = _conf_get(cfg, K.CFG_CHECKPOINT_NAME)
    subj = f'{label}/start_condition'
    if not p_c and not p_n:
        res.add(Finding(_CONF_CHECK, subj, State.UNREADABLE,
                        'neither start-condition key is in the config'))
    else:
        res.add(Finding(_CONF_CHECK, subj, State.OK,
                        f'{K.CFG_CONTINUE_FROM_CHECKPOINT}='
                        f'{_conf_show(p_c, cont)}  '
                        f'{K.CFG_CHECKPOINT_NAME}={_conf_show(p_n, ckpt)}'))

    _conf_stage_subjects(res, run, ctx, window, label)


def _conf_stage_subjects(res: CheckResult, run, ctx: Optional[Context],
                         window: Optional[float], label: str) -> None:
    subj_r, subj_b = f'{label}/stage_residence', f'{label}/stage_boundary'
    got = _conf_stage_series(run)
    if got is None:
        reason = (f'{K.STAGE_METRIC} has fewer than two logged points -- a '
                  f'stage entry cannot be located')
        res.add(Finding(_CONF_CHECK, subj_r, State.UNREADABLE, reason))
        res.add(Finding(_CONF_CHECK, subj_b, State.UNREADABLE, reason))
        return

    s, v = got
    bounds = _conf_boundaries(s, v)
    now = max(as_float(getattr(run, 'last_step', 0.0), 0.0), float(s[-1]))
    stage = (ctx.stage_name if ctx is not None and ctx.stage_name
             else f'{K.STAGE_METRIC}={int(v[-1])}')

    # No boundary in the history is NOT 'the run never transitioned'. Runs here
    # restart from a checkpoint with the step counter carried over, so the
    # history can begin mid-stage; the residence is then bounded below by the
    # span, and that span is also all there is to read.
    exact = bool(len(bounds))
    entered = float(bounds[-1]) if exact else float(s[0])
    resid = now - entered
    nums = {'steps_in_stage': resid, 'entered_at': entered, 'last_step': now,
            'n_boundaries': len(bounds)}
    if exact:
        where = f'{stage}, entered at {entered:.0f}'
        # The stage entry IS in the history, so the residence is the residence
        # and a short one means the metrics are still the transition's.
        flag = (f'{where}; under {_CONF_MIN_STAGE_STEPS:.0f} steps in the '
                f'stage, so the read is of the injection point')
    else:
        where = (f'{stage}, no stage change in the history -- a LOWER BOUND on '
                 f'residence, and the whole readable span')
        # A different sentence, because the residence is NOT known here: the
        # run may have been in this stage for a hundred thousand steps and be
        # readable for four hundred of them. Claiming an injection point would
        # be asserting something the data does not say.
        flag = (f'{where}; under {_CONF_MIN_STAGE_STEPS:.0f} steps of history '
                f'in this stage, whatever the true residence is')
    res.add(Finding(_CONF_CHECK, subj_r,
                    State.FLAG if resid < _CONF_MIN_STAGE_STEPS else State.OK,
                    flag if resid < _CONF_MIN_STAGE_STEPS else where, nums))

    last_b = float(bounds[-1]) if exact else float('nan')
    if window is None:
        # Not a flag. With no window the read is the whole history, so every
        # boundary is inside it by construction and flagging that would fire on
        # every multi-stage run while saying nothing about the read.
        res.add(Finding(_CONF_CHECK, subj_b, State.OK,
                        'no window given -- the read spans the whole history '
                        'and every stage boundary in it',
                        {'n_boundaries': len(bounds), 'last_boundary': last_b}))
        return
    inside = bounds[bounds > now - float(window)]
    if len(inside):
        res.add(Finding(_CONF_CHECK, subj_b, State.FLAG,
                        'the trailing window straddles a stage boundary -- '
                        'features over it mix two stages plus the transition '
                        'transient (fresh optimiser, LR ramp)',
                        {'n_in_window': len(inside),
                         'nearest': float(inside[-1]),
                         'window': float(window), 'last_step': now}))
    else:
        res.add(Finding(_CONF_CHECK, subj_b, State.OK, '',
                        {'n_in_window': 0, 'last_boundary': last_b,
                         'window': float(window)}))


# ---------------------------------------------------------------------------
# Battery subjects
# ---------------------------------------------------------------------------

def _conf_group(runs, fn) -> dict:
    """Arms grouped by a hashable reading of their config, insertion-ordered."""
    out = {}
    for run in runs:
        out.setdefault(fn(run), []).append(_conf_label(run))
    return out


def _conf_battery(res: CheckResult, runs: list) -> None:
    _conf_battery_commit(res, runs)
    _conf_battery_checkpoint(res, runs)
    _conf_battery_start(res, runs)
    _conf_battery_duplicates(res, runs)
    _conf_battery_sweep(res, runs)


def _conf_battery_commit(res: CheckResult, runs: list) -> None:
    groups = _conf_group(runs, lambda r: K.git_commit(r.config or {}))
    detail = ' | '.join(f'{c or "no stamp"}: {", ".join(a)}'
                        for c, a in groups.items())
    nums = {'n_commits': len(groups), 'n_arms': len(runs)}
    if len(groups) > 1:
        res.add(Finding(_CONF_CHECK, 'battery/code_version', State.FLAG,
                        f'arms are on different code -- {detail}', nums))
    else:
        res.add(Finding(_CONF_CHECK, 'battery/code_version', State.OK,
                        detail, nums))


def _conf_battery_checkpoint(res: CheckResult, runs: list) -> None:
    """`checkpoint_name` mixed null / non-null across a battery.

    The worked case: three arms running with `checkpoint_name: None` beside nine
    that carried an explicit checkpoint are two batches, not one battery of
    twelve -- the second nine started from a trained model and the three did
    not, so every metric between them is offset by that."""
    missing, null, named = [], [], []
    for run in runs:
        present, value = _conf_get(run.config or {}, K.CFG_CHECKPOINT_NAME)
        (missing if not present else null if value is None else named).append(
            _conf_label(run))
    nums = {'n_named': len(named), 'n_null': len(null),
            'n_missing': len(missing), 'n_arms': len(runs)}
    parts = [f'named: {", ".join(named)}' if named else '',
             f'null: {", ".join(null)}' if null else '',
             f'key missing: {", ".join(missing)}' if missing else '']
    detail = '  |  '.join(p for p in parts if p)
    if named and (null or missing):
        res.add(Finding(_CONF_CHECK, 'battery/checkpoint_name', State.FLAG,
                        f'arms started from different things -- {detail}', nums))
    else:
        res.add(Finding(_CONF_CHECK, 'battery/checkpoint_name', State.OK,
                        detail, nums))


def _conf_battery_start(res: CheckResult, runs: list) -> None:
    for key in _CONF_START_KEYS:
        groups = _conf_group(
            runs, lambda r, k=key: _conf_show(*_conf_get(r.config or {}, k)))
        detail = ' | '.join(f'{val}: {", ".join(a)}' for val, a in groups.items())
        nums = {'n_values': len(groups), 'n_arms': len(runs)}
        state = State.FLAG if len(groups) > 1 else State.OK
        res.add(Finding(_CONF_CHECK, f'battery/start/{key}', state,
                        detail, nums))


def _conf_battery_duplicates(res: CheckResult, runs: list) -> None:
    """Arms whose SHARED knobs all agree are the same arm written twice.

    An absent knob takes its default, so a pair that differs only in which keys
    are PRESENT is a pair of duplicates -- the sweep dimension the author
    thought they were varying is not in the config at all. The stricter case,
    where the two configs are equal outright, is the same finding and is named
    as such rather than being folded in silently."""
    dup = 0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            ca, cb = a.config or {}, b.config or {}
            ka, kb = _conf_knobs(ca), _conf_knobs(cb)
            differ = [k for k in (ka & kb)
                      if not _conf_equal(K._value(ca, k), K._value(cb, k))]
            if differ:
                continue
            dup += 1
            only_a, only_b = sorted(ka - kb), sorted(kb - ka)
            omitted = only_a + only_b
            nums = {'n_shared': len(ka & kb), 'n_differing': 0,
                    'n_present_only_in_one': len(omitted)}
            if omitted:
                detail = ('same arm written two ways -- every shared knob '
                          'agrees and the rest are absent, taking their '
                          f'defaults: {", ".join(omitted[:_CONF_SWEEP_NAMES_SHOWN])}')
            else:
                detail = ('identical configs -- not one knob differs outside '
                          'the identity keys')
            res.add(Finding(_CONF_CHECK,
                            f'battery/duplicate/{_conf_label(a)}~{_conf_label(b)}',
                            State.FLAG, detail, nums))
    n_pairs = len(runs) * (len(runs) - 1) // 2
    res.add(Finding(_CONF_CHECK, 'battery/duplicates', State.OK,
                    f'{dup} duplicate pair(s) of {n_pairs} compared',
                    {'n_pairs': n_pairs, 'n_duplicate': dup}))


def _conf_battery_sweep(res: CheckResult, runs: list) -> None:
    """The sweep table: which knobs actually differ across the battery.

    Not a finding -- it is the table the reader needs in order to say which arm
    is which. A knob differing by PRESENCE counts: absent means the default, and
    a default that differs from a sibling's explicit value is a swept knob
    whether or not anyone meant to sweep it."""
    keys = set()
    for run in runs:
        keys |= _conf_knobs(run.config or {})
    by_presence, by_value = [], []
    for key in sorted(keys):
        seen = [_conf_get(run.config or {}, key) for run in runs]
        p0, v0 = seen[0]
        if any(p != p0 for p, _ in seen[1:]):
            by_presence.append(key)
        elif any(not _conf_equal(v, v0) for _, v in seen[1:]):
            by_value.append(key)
    names = by_value + by_presence
    shown = ', '.join(names[:_CONF_SWEEP_NAMES_SHOWN])
    more = len(names) - _CONF_SWEEP_NAMES_SHOWN
    res.add(Finding(_CONF_CHECK, 'battery/sweep', State.OK,
                    (shown + (f'  (+{more} more)' if more > 0 else ''))
                    or 'no knob differs across the battery',
                    {'n_knobs': len(names), 'by_value': len(by_value),
                     'by_presence': len(by_presence), 'n_arms': len(runs)}))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def check_confounds(runs, *, ctx: Optional[Context] = None,
                    window: Optional[float] = None) -> CheckResult:
    """§4's confounds, over one run or a battery.

    Battery subjects run first: 'these arms are not comparable' outranks
    anything true of one of them. A single run gets the per-run subjects and a
    row saying the cross-arm ones were skipped -- returning `not_run` there
    would throw away the T, code-version, start-condition and stage-residence
    answers, which are properties of the run and not of the battery.
    """
    runs = _conf_normalise(runs)
    if not runs:
        return CheckResult.not_run(
            _CONF_CHECK, 'no runs given -- nothing to read or to compare')
    res = CheckResult(check=_CONF_CHECK)
    if len(runs) > 1:
        _conf_battery(res, runs)
    else:
        res.add(Finding(_CONF_CHECK, 'battery', State.OK,
                        'cross-arm subjects skipped -- one arm, and a confound '
                        'between arms needs a sibling to be a confound of',
                        {'n_arms': 1}))
    for run in runs:
        rctx = ctx if (ctx is not None and len(runs) == 1) else context(run)
        _conf_per_run(res, run, rctx, window)
    return res
