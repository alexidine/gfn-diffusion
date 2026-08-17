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

THE CONSOLIDATION LOOP. The reference does not need banking -- git already holds
it, so the baseline is always the committed config rather than a snapshot file
someone has to remember to refresh:

    git show HEAD:energy_sampling/configs/mk_dev.yaml > /tmp/base.yaml
    python -m config_snapshot /tmp/base.yaml configs/mk_dev.yaml

Exit status is 0 when no key a run reads changed value, 1 otherwise, so this
drops straight into a loop and every edit is one command away from a verdict.
That is what makes it safe to consolidate in small steps instead of one leap.
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


def _stage_summary(spec: dict, index: int, args) -> dict:
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
        # WHICH LR CONTROLLER this stage runs, and it is per stage. The four
        # kinds (ray / plateau / hyper / none) behave completely differently, so
        # two configs with identical seeds and identical servo maps can still
        # train nothing alike. Recorded explicitly as 'absent' rather than left
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


def snapshot(yaml_path: str) -> dict:
    """Resolve a config and capture everything it determines, GPU-free.

    A config that FAILS TO LOAD returns a snapshot carrying `load_error` rather
    than raising. This matters more than it looks: the reference side of a
    comparison is usually the committed config, and the most common reason to
    tighten a rule is that the committed config violates it. A comparator that
    dies on "before" is useless in exactly the case it was built for -- so an
    unloadable config is a REPORTED state, not a crash."""
    import utils

    try:
        args = utils.resolve_derived_config(
            utils.preflight_config(utils.dict2namespace(utils.load_yaml(yaml_path))))
    except Exception as e:
        return {'source': str(yaml_path), 'config': {}, 'stages': [],
                'load_error': f'{type(e).__name__}: {e}'}
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

    return {'source': str(yaml_path), 'config': resolved, 'stages': stages}


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
    a = ap.parse_args()

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
