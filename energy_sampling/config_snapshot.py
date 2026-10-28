"""
What a config actually resolves to, and whether an edit changed it.

THE PROBLEM THIS SOLVES. Consolidating the canonical config means restructuring
it -- moving keys, adding blocks for modes that are inactive. So a raw text diff
is guaranteed to be noisy, and "the file changed" says nothing about whether the
RUN changed. What has to be preserved is the resolved, effective configuration
for the selected mode, and that is a different object from the YAML.

So the comparison is three-way, and the middle column is the one that matters:

    CHANGED   a key present in both, with a different value    <- DANGER
    ADDED     a key only in the new config                     <- expected during
                                                                  consolidation;
                                                                  must be inert
    REMOVED   a key only in the old config                     <- check it was dead

A consolidation pass is correct when CHANGED is empty and every ADDED key belongs
to a mode that is off. The first half is mechanical and is what this module
automates; the second is judgment, and the report puts the list in front of you
rather than deciding for you.

WHAT IT CAPTURES. Everything a config determines that needs neither a GPU nor the
data drive:

  * the RESOLVED config -- after preflight and derived-value resolution, so
    `auto` has become the number it will actually train at
  * the PARSED protocol -- stages as `protocol.Stage` reads them
  * the EFFECTIVE loss coefficients per stage, computed by the trainer's own
    `StageProtocol.coeffs` rather than a reimplementation of it

That last point is deliberate. The overlay rule (base block + stage overrides) is
four lines and would be trivial to copy; copying it is how a comparator comes to
certify a config against a rule the trainer no longer follows.

    python -m config_snapshot cfg.yaml                  # show what it resolves to
    python -m config_snapshot old.yaml new.yaml         # compare two configs
    python -m config_snapshot cfg.yaml --save ref.json  # bank a reference
    python -m config_snapshot cfg.yaml --against ref.json

THE CURRENT CONTRACT. Validate the canonical operational surface without a GPU,
data drive, checkpoint, W&B, or model construction:

    python -m config_snapshot configs/mk_dev.yaml --check

For a deliberate before/after comparison, bank the reference explicitly with
``--save`` and use ``--against``.  Do not infer an authoritative baseline from
Git chronology; the reference is evidence only when the person making the
comparison chose it for that purpose.
"""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

MODES = ('fwd', 'bwd', 'replay')

# Keys whose value is a machine-local path or otherwise not part of what the run
# COMPUTES. Compared separately so a path difference does not hide in the same
# list as a changed learning rate.
_ENVIRONMENT_KEYS = frozenset({
    'checkpoints_dir', 'device', 'cuda_memory_fraction', 'run_name', 'tag',
})


def _to_plain(obj):
    """Namespace tree -> plain data, recursively."""
    if isinstance(obj, Namespace):
        return {k: _to_plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def flatten(obj, prefix='') -> dict[str, Any]:
    """Nested config -> {dotted path: leaf}. Lists are indexed, so a reordered
    protocol shows up as changed stages rather than as one opaque blob."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f'{prefix}.{k}' if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f'{prefix}[{i}]'))
    else:
        out[prefix] = obj
    return out


class _StubModeller:
    """The minimum `StageProtocol` touches to compute coefficients: `args`, and
    `stage` as a NAME (the protocol looks the name up in its own parse of
    `args.protocol.stages`).

    A stub rather than a real Modeller because building one needs the energy
    function, which needs the data drive and, on the MLIP route, a GPU. The
    overlay itself depends on nothing but the base blocks and the stage, so the
    real code path runs end to end on this."""

    def __init__(self, args, stage_name=None):
        self.args = args
        self.stage = stage_name


def _effective_coeffs(args, stage_name: str) -> dict[str, dict]:
    """Effective loss coefficients for one stage, via the trainer's own overlay.

    Raises whatever `StageProtocol.coeffs` raises -- notably on an override key
    absent from the base block, which is a config error the snapshot should
    surface rather than smooth over."""
    import protocol as P

    sp = P.StageProtocol(_StubModeller(args, stage_name))
    return {m: sp.coeffs(m) for m in MODES}


def _stage_summary(spec: dict, index: int, args, coeffs: Optional[dict] = None) -> dict:
    """`coeffs` given ({mode: effective coefficients}) is used as is, in place of
    the stub overlay -- the trainer passes its live StageProtocol.coeffs, since
    by then args carries an overlay already applied."""
    import protocol as P

    st = P.Stage(spec, index)
    out = {
        'name': st.name,
        'train_mode': st.train_mode,
        'bwd_sampling_mode': getattr(st, 'bwd_sampling_mode', None),
        'skip_if': getattr(st, 'skip_if', None),
        'flags': dict(getattr(st, 'flags', {}) or {}),
        'fracs': dict(getattr(st, 'fracs', {}) or {}),
        'min_fracs': dict(getattr(st, 'min_fracs', {}) or {}),
        'deactivate_threshold': getattr(st, 'deactivate_threshold', None),
        # WHICH LR DIAGNOSTIC this stage runs, and it is per stage. The two
        # kinds (ray / none) behave completely differently, so two configs with
        # identical seeds and identical servo maps can still train nothing
        # alike. Recorded explicitly as 'absent' rather than left
        # as None because OMITTING the block means "no sensor" SILENTLY -- and it
        # did not always: omission used to arm the ray probe under the global
        # ray_calibration.enabled. A consolidation that dropped an lr_sensor
        # block would otherwise pass this comparator clean.
        'lr_sensor': (_to_plain(st.lr_sensor) if getattr(st, 'lr_sensor', None)
                      else 'absent (no sensor)'),
        'on_enter': list(getattr(st, 'on_enter', []) or []),
        'on_exit': list(getattr(st, 'on_exit', []) or []),
        'exit': _to_plain(getattr(st, 'exit', None)),
        'balance': _to_plain(getattr(st, 'balance', None)),
    }
    if coeffs is not None:
        out['effective_loss_coeffs'] = {m: dict(coeffs.get(m) or {}) for m in MODES}
        return out
    try:
        out['effective_loss_coeffs'] = _effective_coeffs(args, st.name)
    except Exception as e:
        # Recorded, not raised: a snapshot that dies on one malformed stage is
        # useless for the comparison that would have shown WHY it is malformed.
        out['effective_loss_coeffs'] = f'ERROR: {type(e).__name__}: {e}'
    return out


def _periodic_centroid_axes(args):
    """Resolved wrap axes, or a string saying why there are none.

    Returns a STRING rather than None for each 'no axes' case, so the reasons are
    distinguishable in a diff: 'feature off' and 'this space group has no
    full-width axes' produce the same empty wrap but are different states, and a
    consolidation that silently converted one into the other would compare
    equal."""
    model = getattr(args, 'model', None)
    if not getattr(model, 'periodic_centroids', False):
        return 'off'
    sgs = list(getattr(args, 'space_groups', []) or [])
    if len(sgs) != 1:
        # The model becomes space-group specific, so more than one is refused at
        # construction. Recorded, not raised: a snapshot exists to explain a
        # broken config, not to die alongside it.
        return f'INVALID: needs exactly one space group, got {sgs}'
    try:
        from energy_sampling.models.aunit_periodicity import sg_periodic_centroid_axes
        axes = set(sg_periodic_centroid_axes(int(sgs[0])))
    except Exception as e:
        return f'UNRESOLVED: {type(e).__name__}: {e}'
    # A MEMBERSHIP MAP, not a list, for the same reason lr_servo_managed is one:
    # this is a SET, and flatten() indexes lists positionally. [1,2] -> [2] would
    # report "element 0 changed 1->2, element 1 removed" instead of "axis 1 is no
    # longer wrapped". Keyed by axis, the diff names the axis that moved.
    return {f'axis_{i}': (i in axes) for i in range(3)}


def _fd_wrap_rows(args):
    """energy_config.crystal_coordinates 'fd': {chart row name: wrapped}, the FD chart's circle rows
    (energies/fd_coordinates.py::FDCoordinates.wrap_mask) for the one space group; the trainer passes these as the
    angular mask and does not read model.periodic_centroids. A string when the chart cannot be built."""
    sgs = list(getattr(args, 'space_groups', []) or [])
    if len(sgs) != 1:
        return f'INVALID: needs exactly one space group, got {sgs}'
    try:
        from energy_sampling.energies.fd_coordinates import FDCoordinates
        from energy_sampling.energies.molecular_crystal import FD_LAYOUT_RADIUS
        fd = FDCoordinates(int(sgs[0]), FD_LAYOUT_RADIUS)
    except Exception as e:
        return f'UNRESOLVED: {type(e).__name__}: {e}'
    return {name: bool(w) for name, w in zip(fd.names, fd.wrap_mask)}


def snapshot(yaml_path: str) -> dict:
    """Resolve a config and capture everything it determines, GPU-free.

    A config that FAILS TO LOAD returns a snapshot carrying `load_error` rather
    than raising. This matters more than it looks: the reference side of a
    comparison is usually the committed config, and the most common reason to
    tighten a rule is that the committed config violates it. A comparator that
    dies on "before" is useless in exactly the case it was built for -- so an
    unloadable config is a REPORTED state, not a crash."""
    return _snapshot(yaml_path)[0]


def _snapshot(yaml_path: str):
    """`snapshot` plus the resolved args Namespace it was taken from (None when
    the config does not load)."""
    try:
        import utils
        args = utils.resolve_derived_config(
            utils.preflight_config(utils.dict2namespace(utils.load_yaml(yaml_path))))
    except Exception as e:
        return {'source': str(yaml_path), 'config': {}, 'stages': [],
                'load_error': f'{type(e).__name__}: {e}'}, None
    resolved = _to_plain(args)

    # SERVO MANAGEMENT IS A PER-KEY PROPERTY, and it is recorded as one so the
    # comparison can see it.
    #
    # `auto` and an explicit float RESOLVE TO THE SAME NUMBER and mean opposite
    # things: `auto` hands the rate to the ray-calibration servo, which then owns
    # it; a float is a fixed peak the servo never touches. Swapping one for the
    # other changes the run completely while leaving `lr_policy` identical, so the
    # resolved value cannot detect it and `lr_servo_managed` is the only witness.
    #
    # Stored as {key: bool} rather than the raw tuple because a positional list
    # compares badly: dropping one entry reports as N shifted elements plus a
    # removal, which buries "lr_policy is no longer servo-managed" under noise and
    # would also fire spuriously on a reordering.
    managed = set(getattr(args, 'lr_servo_managed', ()) or ())
    resolved['lr_servo_managed'] = {k: (k in managed) for k in utils._LR_KEYS}

    # PERIODIC CENTROID AXES -- which aunit centroid axes get wrapped. Derived
    # from the space group, and it sets the model's `expanded_dim`, so a change
    # here is not merely a different run: it produces a model of a different
    # SHAPE, and the config's own note says such a model is not checkpoint
    # compatible with one trained without it. That makes it the highest-value
    # tier-B quantity to capture -- the config keys behind it (space_groups,
    # periodic_centroids) are individually visible, but the axis set they
    # RESOLVE TO is what the model is actually built against.
    #
    # Computed here rather than via the trainer's `_resolve_periodic_centroid_axes`
    # because that method needs a constructed energy function for an `is_crystal`
    # guard; the axes themselves are a pure function of the space group.
    #
    # On energy_config.crystal_coordinates 'fd' the state is the FD chart, whose
    # own circle rows are wrapped and model.periodic_centroids is not read: the
    # snapshot records those rows (fd_wrap_rows) and no periodic_centroid_axes.
    energy_config = getattr(args, 'energy_config', None)
    if getattr(energy_config, 'crystal_coordinates', 'latent') == 'fd':
        resolved['fd_wrap_rows'] = _fd_wrap_rows(args)
    else:
        resolved['periodic_centroid_axes'] = _periodic_centroid_axes(args)

    # Stage specs come from the RESOLVED args, not from a second read of the
    # YAML: dict2namespace converts in place, so the loaded dict no longer holds
    # dicts, and re-reading the file would snapshot the pre-preflight text rather
    # than what the trainer will actually parse.
    stages = []
    # Through the shared resolver, so the snapshot describes the stages the run
    # would ACTUALLY execute rather than whatever list happens to be present.
    from config_invariants import active_protocol_name, active_stages
    for i, spec in enumerate(active_stages(resolved)):
        stages.append(_stage_summary(spec, i, args))
    resolved['_active_protocol'] = active_protocol_name(resolved)

    return {'source': str(yaml_path), 'config': resolved, 'stages': stages}, args


def contract(yaml_path: str) -> tuple[dict, list[str]]:
    """Validate one config as a complete CPU-only operational contract.

    This composes the existing sources of proof rather than restating their
    rules: ``snapshot`` exercises the trainer's load/preflight/derived-config and
    active ``StageProtocol`` paths, while ``config_invariants`` parses every
    declared protocol and checks the cross-field claims.  Baseline departures
    are failures here because this command is the canonical-control-surface
    check; experiment generation has its own path that reports them instead.
    """
    return _contract(yaml_path)[:2]


def _contract(yaml_path: str):
    """`contract` plus the resolved args Namespace (None when it does not load)."""
    snap, args = _snapshot(yaml_path)
    issues = []
    if snap.get('load_error'):
        issues.append(f"load: {snap['load_error']}")
        return snap, issues, args

    try:
        import config_invariants
        import utils
        raw = utils.load_yaml(yaml_path)
        issues.extend(str(v) for v in config_invariants.check(raw))
    except Exception as e:
        issues.append(f'invariants: {type(e).__name__}: {e}')

    stages = snap.get('stages') or []
    if not stages:
        issues.append('active protocol resolves to no stages')
    for stage in stages:
        coeffs = stage.get('effective_loss_coeffs')
        if isinstance(coeffs, str) and coeffs.startswith('ERROR:'):
            issues.append(f"stage {stage.get('name')!r} loss coefficients: {coeffs}")
    return snap, issues, args


# ---------------------------------------------------------------------------
# Prose summary of a protocol: a few plain sentences per stage, generated from
# the parsed Stage and its effective loss coefficients. A stage key the
# sentences do not phrase is listed verbatim in an 'also:' tail.

# loss_coeffs keys that are loss-term weights: phrased when nonzero
_LOSS_TERMS = ('tb', 'mle', 'tbc', 'z_level', 'vg_lb', 'vg_lme', 'vg_by_condition', 'db',
               'subtb', 'emp_z', 'emp_z_persistent', 'level_gap', 'pf_boost')
# loss_coeffs keys the sentences phrase; any other stage override goes in the tail
_PHRASED_COEFFS = set(_LOSS_TERMS) | {'freeze_policy', 'freeze_z', 'pooled_vg',
                                      'pooled_source', 'pooled_ratio'}
# stage keys never phrased: listed in the tail whenever the stage declares them
_TAIL_KEYS = ('hot_lr_sensor', 'lr_sensor', 'mle_gate', 'min_fracs', 'deactivate_threshold',
              'lr_schedule')
_BRANCH_DATA = {'fwd': 'its own on-policy rollouts',
                'replay': 'stored forward rollouts in the replay buffer'}
_BWD_DATA = {'dataset': 'terminals drawn from the prior dataset',
             'prior': 'terminals drawn from the prior_buffer'}


def _num_text(v) -> str:
    return f'{v:g}' if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _verbatim(v) -> str:
    """One-line YAML flow form of a config value, keys in declared order."""
    import yaml
    if not isinstance(v, (dict, list)):
        return _num_text(v)
    return yaml.safe_dump(_to_plain(v), default_flow_style=True, sort_keys=False,
                          width=10 ** 6).strip()


def _trained_params(c: dict) -> str:
    freeze_policy = float(c.get('freeze_policy', 0) or 0) > 0.5
    freeze_z = float(c.get('freeze_z', 0) or 0) > 0.5
    if freeze_policy and freeze_z:
        return 'nothing (policy and log Z both frozen)'
    if freeze_policy:
        return 'log Z only'
    if freeze_z:
        return 'the policy only'
    return 'the policy and log Z'


def _terms_text(c: dict) -> str:
    terms = [f'{k} {_num_text(c[k])}' for k in _LOSS_TERMS
             if isinstance(c.get(k), (int, float)) and not isinstance(c.get(k), bool)
             and float(c[k]) != 0.0]
    return ', '.join(terms) if terms else 'no loss term of its own'


def _exit_text(terms) -> str:
    out = []
    for t in terms:
        side, sym = ('above', '>') if 'above' in t else ('below', '<')
        metric = f"|{t['metric']}|" if t.get('abs') else t['metric']
        out.append(f"{metric} {sym} {_num_text(t[side])} for {int(t.get('patience', 1))} "
                   f"fresh measurement(s)")
    return ' and '.join(out)


def _balance_text(spec_bal: dict, parsed: dict) -> tuple[str, set]:
    """(clause, the spec balance keys it phrases)."""
    kind = parsed['kind']
    if kind == 'lexicographic':
        metrics = [r['metric'] for r in parsed.get('rules') or []]
        text = (f"a lexicographic balance over {len(metrics)} rule(s)"
                + (f" on {', '.join(metrics)}" if metrics else '')
                + f", default boost {_verbatim(parsed['default_boost'])}")
        ann = parsed.get('anneal_coeffs') or {}
        if ann:
            text += ('; each all-clean streak anneals '
                     + ', '.join(f"{k} toward {_num_text(v['target'])}"
                                 + (f" (rate {_num_text(v['rate'])})" if 'rate' in v else '')
                                 for k, v in ann.items()))
        if parsed.get('anneal_cooldown_steps'):
            text += f", at least {parsed['anneal_cooldown_steps']} steps apart"
        return text, {'kind', 'rules', 'default_boost', 'anneal_coeffs',
                      'anneal_cooldown_steps'}
    if kind == 'gated_ramp':
        text = (f"a gated_ramp balance moves share to {parsed['ramp']} while "
                f"{parsed['metric']} <= {_num_text(parsed['bar'])} and back to "
                f"{parsed['guard']} otherwise")
        if parsed.get('ratchet_metric'):
            text += f", the ramp also held unless {parsed['ratchet_metric']} is near its stage best"
        return text, {'kind', 'ramp', 'guard', 'metric', 'bar', 'ratchet_metric'}
    metrics = parsed.get('metrics') or {}
    text = (f"a {kind} balance splits the {'/'.join(metrics)} share on "
            f"{', '.join(metrics.values())}")
    if 'setpoint' in parsed:
        text += f" at setpoint {_num_text(parsed['setpoint'])}"
    return text, {'kind', 'metrics', 'setpoint'}


def describe_stage(spec: dict, index: int, args=None, *, coeffs: Optional[dict] = None,
                   successor: Optional[str] = None, n_stages: Optional[int] = None) -> str:
    """Plain sentences for one stage: what trains which parameters on which data,
    then cadence / draw / balance / flags, then how the stage starts and ends.

    Fields come from `_stage_summary` (effective loss coefficients included) and
    from the Stage attributes it does not carry. `coeffs` overrides the summary's
    coefficients (see _stage_summary); `successor` is the next stage's name, None
    on the last stage."""
    import copy
    import protocol as P

    spec = copy.deepcopy(spec)
    st = P.Stage(copy.deepcopy(spec), index)
    summ = _stage_summary(copy.deepcopy(spec), index, args, coeffs=coeffs)
    eff = summ['effective_loss_coeffs']
    if isinstance(eff, str):
        eff = {m: dict(st.loss_coeffs.get(m, {})) for m in MODES}
        coeff_note = f' (base coefficients unavailable: {summ["effective_loss_coeffs"]})'
    else:
        coeff_note = ''
    tail = []
    for k in _TAIL_KEYS:
        if spec.get(k) is not None:
            tail.append(f'{k}={_verbatim(spec[k])}')
    for mode, over in (spec.get('loss_coeffs') or {}).items():
        for k, v in (over or {}).items():
            if k not in _PHRASED_COEFFS:
                tail.append(f'loss_coeffs.{mode}.{k}={_verbatim(v)}')

    # -- sentence 1: who trains what on which data
    data = dict(_BRANCH_DATA, bwd=_BWD_DATA[st.bwd_sampling_mode])
    of_n = f'{index + 1}/{n_stages}' if n_stages else f'{index + 1}'
    head = f"Stage {of_n} '{st.name}' ({st.train_mode})"
    parts = []
    if st.train_mode == 'bwd':
        c = eff.get('bwd') or {}
        parts.append(f"bwd alone trains {_trained_params(c)} on {data['bwd']} with "
                     f"{_terms_text(c)}")
    else:
        total = float(sum(st.fracs.values())) if st.fracs else 0.0
        # stage override, else controller.deactivate_threshold as train.py reads it;
        # 0.01 only when no args are given
        thr = st.deactivate_threshold
        if thr is None:
            thr = getattr(getattr(args, 'controller', None), 'deactivate_threshold', None)
            thr = 0.01 if thr is None else float(thr)
        for mode in MODES:
            c = eff.get(mode) or {}
            if mode == 'fwd' and st.fwd_z_sidecar:
                parts.append(f"fwd trains {_trained_params(c)} on {data['fwd']} with "
                             f"{_terms_text(c)}, as a weight-1 sidecar outside the frac mix")
                continue
            share = float(st.fracs.get(mode, 0.0)) / total if total > 0 else None
            weighted = (share is None or share >= thr
                        or float(st.min_fracs.get(mode, 0.0)) >= thr
                        or st.balance_can_raise(mode, thr))
            if not weighted:
                parts.append(f"{mode} carries no loss weight")
                continue
            at = (f" at entry share {share:.2f}" if share is not None
                  else ' at the share carried from the previous stage')
            parts.append(f"{mode} trains {_trained_params(c)} on {data[mode]} with "
                         f"{_terms_text(c)}{at}")
        fc = eff.get('fwd') or {}
        pv = fc.get('pooled_vg')
        if isinstance(pv, (int, float)) and float(pv) > 0:
            parts.append(f"a pooled VarGrad term (pooled_vg {_num_text(pv)}) pairs "
                         f"{fc.get('pooled_source', 'fwd')} rows with bwd rows, bwd share "
                         f"{_num_text(fc.get('pooled_ratio', 0.5))}")
    sentences = [f"{head}: {'; '.join(parts)}{coeff_note}."]

    # -- sentence 2: cadence, draw, balance, schedule, flags
    mech = []
    if st.train_mode == 'fused':
        if st.fwd_rollout_every > 0:
            r = f"forward rollouts run on 1 step in {st.fwd_rollout_every}"
            trig = {k: v for k, v in st.fwd_rollout_triggers.items() if v > 0}
            if trig:
                r += ', or early on ' + ', '.join(f'{k} {_num_text(v)}' for k, v in trig.items())
            if st.replay_warmup_rows:
                r += (f", and on every step until the replay buffer holds "
                      f"{st.replay_warmup_rows} trainable rows")
            if st.z_pin_rollout_every:
                r += f"; a z-pin rollout (log Z fill only) every {st.z_pin_rollout_every} steps"
            mech.append(r)
        else:
            mech.append('forward rollouts run on every step')
    if st.condition_draw is not None:
        cd = st.condition_draw
        cn = (cd['conditions'] or
              f"batch_size // {max(cd['replay_rows'], cd['prior_rows'])}")
        mech.append(f"each step draws {cn} conditions ({cd['pick']}) with exactly "
                    f"{cd['replay_rows']} replay and {cd['prior_rows']} bwd rows each")
    if st.balance is not None:
        btext, phrased = _balance_text(spec.get('balance') or {}, st.balance)
        mech.append(btext)
        for k, v in (spec.get('balance') or {}).items():
            if k not in phrased:
                tail.append(f'balance.{k}={_verbatim(v)}')
    for name, s in st.coeff_schedule.items():
        mech.append(f"{name} ramps to {_num_text(s['target'])} over {s['steps']} steps "
                    f"({s['kind']}" + (f", from step {s['from_step']}" if 'from_step' in s else '')
                    + ')')
    on = [k for k, v in st.flags.items() if v]
    if on:
        mech.append(f"flags on: {', '.join(on)}")
    if mech:
        sentences.append(mech[0][0].upper() + mech[0][1:]
                         + ''.join(f'; {m}' for m in mech[1:]) + '.')

    # -- sentence 3: how it starts and ends
    start = []
    if st.skip_if:
        start.append(f"a fresh run skips it when {st.skip_if} holds")
    actions = [f'{n}:{a}' if a else n for n, a in st.on_enter]
    if actions:
        if index == 0:
            start.append(f"its on_enter ({', '.join(actions)}) never fires, since the first "
                         f"stage is not entered by a transition")
        else:
            start.append(f"on entry it runs {', '.join(actions)}")
    if not start:
        start.append('it starts with no entry action' if index else 'a fresh run starts here')
    exits = [f'{n}:{a}' if a else n for n, a in st.on_exit]
    stops = 'stop' in [n for n, _ in st.on_exit]
    if st.exit or st.max_steps is not None:
        when = []
        if st.exit:
            when.append(f"when {_exit_text(st.exit)}")
        if st.max_steps is not None:
            when.append(f"after {st.max_steps} of its own steps")
        end = f"it ends {' or '.join(when)}"
        run = [a for a in exits if a != 'stop']
        if run:
            end += f", runs {', '.join(run)}"
        if stops:
            end += ' and ends the run'
        elif successor:
            end += f" and advances to '{successor}'"
        else:
            end += ' with no successor stage'
    else:
        end = "it has no exit and runs until the run's epochs are spent"
    s3 = '; '.join(start) + '; ' + end + '.'
    sentences.append(s3[0].upper() + s3[1:])

    text = ' '.join(sentences)
    if tail:
        text += ' also: ' + '; '.join(tail)
    return text


def describe_protocol(args, name: str) -> str:
    """`describe_stage` for every stage of the declared protocol `name`, one
    paragraph per stage. Coefficients come from the trainer's overlay with `name`
    selected, so an inactive protocol is described as it would run."""
    from config_invariants import PROTOCOL_LIBRARY, PROTOCOL_SELECTOR
    library = _to_plain(getattr(args, PROTOCOL_LIBRARY, None)) or {}
    specs = list((library.get(name) or {}).get('stages') or [])
    if not specs:
        return f"protocol '{name}' declares no stages."
    view = Namespace(**vars(args))
    setattr(view, PROTOCOL_SELECTOR, name)
    paras = []
    for i, spec in enumerate(specs):
        successor = specs[i + 1].get('name') if i + 1 < len(specs) else None
        paras.append(describe_stage(spec, i, view, successor=successor, n_stages=len(specs)))
    return '\n'.join(paras)


# ---------------------------------------------------------------------------

@dataclass
class Comparison:
    changed: list[tuple[str, Any, Any]] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    environment: list[tuple[str, Any, Any]] = field(default_factory=list)
    reference_error: Optional[str] = None
    candidate_error: Optional[str] = None

    @property
    def behaviour_preserved(self) -> bool:
        """True iff nothing a run reads changed value. ADDED keys do not
        disqualify -- proving they are inert is the mode-safety test's job, not
        this one's, and conflating the two would make a passing comparison mean
        less than it appears to.

        An unloadable config on either side is never 'preserved': there is no
        comparison to be preserved BY."""
        if self.reference_error or self.candidate_error:
            return False
        return not self.changed

    def render(self, limit: int = 40) -> str:
        lines = []
        if self.candidate_error:
            lines.append(f'CANDIDATE DOES NOT LOAD: {self.candidate_error}\n')
        if self.reference_error:
            # Expected and not an error in the tool: tightening a rule the
            # committed config violates is a normal reason to be running this.
            lines.append(
                f'REFERENCE DOES NOT LOAD under current code:\n'
                f'  {self.reference_error}\n'
                f'  No value comparison is possible. If the candidate is the fix '
                f'for this, that is the intended direction -- re-baseline once it '
                f'is committed.\n')
        if self.reference_error or self.candidate_error:
            return '\n'.join(lines)
        verdict = ('CHANGED VALUES: 0 -- no key a run reads has a different value'
                   if self.behaviour_preserved else
                   f'CHANGED VALUES: {len(self.changed)}  <-- these alter behaviour')
        lines.append(verdict)
        for path, old, new in self.changed[:limit]:
            lines.append(f'  ~ {path}: {old!r} -> {new!r}')
        if len(self.changed) > limit:
            lines.append(f'  ... {len(self.changed) - limit} more')

        lines.append(f'\nADDED {len(self.added)} (expected while consolidating -- '
                     f'each must belong to an INACTIVE mode):')
        for p in self.added[:limit]:
            lines.append(f'  + {p}')
        if len(self.added) > limit:
            lines.append(f'  ... {len(self.added) - limit} more')

        lines.append(f'\nREMOVED {len(self.removed)} (each must be dead, not just unused):')
        for p in self.removed[:limit]:
            lines.append(f'  - {p}')
        if len(self.removed) > limit:
            lines.append(f'  ... {len(self.removed) - limit} more')

        if self.environment:
            lines.append(f'\nENVIRONMENT {len(self.environment)} (paths/identity, '
                         f'not computation):')
            for path, old, new in self.environment[:limit]:
                lines.append(f'  = {path}: {old!r} -> {new!r}')
        return '\n'.join(lines)


def _shape_changes(fa: dict, fb: dict) -> list[str]:
    """Paths that are a LEAF on one side and a SUBTREE on the other.

    Without this they are invisible. Flattening turns `x: 'none'` into the path
    `x`, and `x: {kind: ray}` into `x.kind` -- no path exists in both, so the
    comparison files one REMOVED and one ADDED and reports nothing CHANGED. Since
    `behaviour_preserved` reads CHANGED only, ANY edit that turns a scalar into a
    block or back would pass clean. That is the shape of half the interesting
    config edits, `lr_sensor` among them."""
    a_leaves, b_leaves = set(fa), set(fb)
    out = []
    for leaves, others in ((a_leaves - b_leaves, b_leaves),
                           (b_leaves - a_leaves, a_leaves)):
        for leaf in leaves:
            if any(k.startswith(leaf + '.') or k.startswith(leaf + '[')
                   for k in others):
                out.append(leaf)
    return sorted(set(out))


def compare(a: dict, b: dict) -> Comparison:
    """Compare two snapshots. `a` is the reference, `b` the candidate."""
    if a.get('load_error') or b.get('load_error'):
        return Comparison(reference_error=a.get('load_error'),
                          candidate_error=b.get('load_error'))

    fa = {**flatten(a.get('config'), 'config'), **flatten(a.get('stages'), 'stages')}
    fb = {**flatten(b.get('config'), 'config'), **flatten(b.get('stages'), 'stages')}

    cmp = Comparison()
    shape = set(_shape_changes(fa, fb))
    for path in sorted(shape):
        cmp.changed.append((path,
                            fa.get(path, '<block>'),
                            fb.get(path, '<block>')))

    for path in sorted(set(fa) | set(fb)):
        if path in shape:
            continue        # already reported as a shape change
        in_a, in_b = path in fa, path in fb
        if in_a and not in_b:
            cmp.removed.append(path)
        elif in_b and not in_a:
            cmp.added.append(path)
        elif fa[path] != fb[path]:
            leaf = path.rsplit('.', 1)[-1]
            if leaf in _ENVIRONMENT_KEYS:
                cmp.environment.append((path, fa[path], fb[path]))
            else:
                cmp.changed.append((path, fa[path], fb[path]))
    return cmp


def _main():
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog='config_snapshot')
    ap.add_argument('config')
    ap.add_argument('other', nargs='?', help='second config to compare against')
    ap.add_argument('--save', metavar='PATH', help='write this config\'s snapshot as JSON')
    ap.add_argument('--against', metavar='PATH', help='compare against a saved snapshot')
    ap.add_argument('--check', action='store_true',
                    help='validate a canonical config and print a concise operational summary')
    a = ap.parse_args()

    if a.check and (a.other or a.save or a.against):
        ap.error('--check cannot be combined with comparison or snapshot-output options')

    if a.check:
        import textwrap
        checked, issues, args = _contract(a.config)
        if issues:
            print(f'{a.config}: contract FAILED')
            for issue in issues:
                print(f'  - {issue}')
            return 1
        cfg = checked['config']
        active = cfg.get('_active_protocol')
        declared = ', '.join(sorted((cfg.get('protocols') or {}).keys()))
        stages = ' -> '.join(str(s.get('name')) for s in checked['stages'])
        print(f'{a.config}: contract ok')
        print(f'  project_state_version: {cfg.get("project_state_version")}')
        print(f'  active protocol: {active}')
        print(f'  declared protocols: {declared}')
        print(f'  active stages: {stages}')
        print('  runtime services touched: none (GPU/data/checkpoint/W&B/model not constructed)')
        for name in sorted((cfg.get('protocols') or {}).keys()):
            print(f"\n  protocol '{name}'{' (active)' if name == active else ''}:")
            try:
                text = describe_protocol(args, name)
            except Exception as e:
                text = f'DESCRIPTION FAILED: {type(e).__name__}: {e}'
            for para in text.split('\n'):
                print(textwrap.fill(para, width=100, initial_indent='    ',
                                    subsequent_indent='      '))
        return 0

    first = snapshot(a.config)

    if a.save:
        with open(a.save, 'w', encoding='utf-8') as f:
            json.dump(first, f, indent=1, sort_keys=True, default=str)
        print(f'wrote {a.save}')
        return 0

    # WHICH IS THE REFERENCE depends on the form, and getting it backwards
    # inverts ADDED and REMOVED -- which reads as "the consolidation deleted the
    # block it just added", i.e. exactly wrong in the direction that matters.
    #   two configs:  A B      -> A is the reference (old), B the candidate (new)
    #   --against:    cfg ref  -> the saved snapshot is the reference
    if a.other:
        reference, candidate = first, snapshot(a.other)
    elif a.against:
        with open(a.against, encoding='utf-8') as f:
            reference, candidate = json.load(f), first
    else:
        print(json.dumps(first, indent=1, sort_keys=True, default=str))
        return 0

    cmp = compare(reference, candidate)
    print(f'reference (old): {reference.get("source")}\n'
          f'candidate (new): {candidate.get("source")}\n')
    print(cmp.render())
    return 0 if cmp.behaviour_preserved else 1


if __name__ == '__main__':
    raise SystemExit(_main())
