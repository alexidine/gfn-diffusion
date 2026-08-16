"""
Project state versioning: the semantic history of config changes, and the
executable migrations that implement them, as ONE artifact.

WHY THEY ARE THE SAME FILE. A history that only describes a change drifts from
the code that performs it, and a migration with no recorded reason cannot be
audited. Every entry below carries both: the semantic delta a future reader needs
in order to understand the transition, and the mechanical transform that applies
it. A transition whose change cannot be applied mechanically says so explicitly
(`manual`) and is REPORTED rather than guessed at.

RELATION TO docs/PROTOCOL.md. PROTOCOL puts Log -- "what happened when" -- in git
history only. These records are not Log. They answer "how does a config at state
N reach state N+1", which is State about the migration path, and unlike Log it is
checkable: a migration either produces a loadable config or it does not.

RELATION TO utils._RETIRED_KEYS. That dict is the LOAD-TIME GATE: it rejects a
config carrying a key the schema no longer has, so a stale value can never be
silently ignored. This module is the REPAIR for the same event. The gate stays
authoritative for rejection; the v0 -> v1 record below carries the subset of
those retirements that can be repaired mechanically, and names the rest.

RELATION TO the checkpoint `schema_version` (utils.py, problem_def). Different
axis, deliberately not merged: that one versions the stored problem identity so a
resume can refuse a checkpoint trained under different physics.
`project_state_version` versions the CONFIG SCHEMA, and a migration across it
must leave the problem identity untouched.

CLI:
    python -m config_state history                  # the chronology, as Markdown
    python -m config_state migrate <config.yaml>    # report what a migration would do
    python -m config_state migrate <config.yaml> --write
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# The state the canonical config (configs/mk_dev.yaml) is written against. A
# production config records the state it was generated from, so a later migration
# knows which transitions to apply.
# What an unstamped config is assumed to be. Every config predating the
# introduction of the key is state 0 by definition.
UNSTAMPED_VERSION = 0

VERSION_KEY = 'project_state_version'


# ---------------------------------------------------------------------------
# TWO DIFFERENT THINGS, deliberately not one thing
#
# `Change`     -- the semantic history. EVERY material functional change to code
#                 or config gets one. This is what a future reader consults to
#                 find out what a change meant.
#
# `Transition` -- the migration payload. Only a change that alters how PERSISTED
#                 state is INTERPRETED carries one, and only such a change moves
#                 project_state_version.
#
# WHY THEY ARE SEPARATE. If every functional edit bumped the state integer, the
# project would reach state 483 with 450 transitions that have nothing to say
# about migrating a config, and the number would stop meaning anything. The
# integer earns its keep precisely by being rare: `state N -> N+1` should be a
# statement about durable state, not a changelog entry.
#
# THE TEST for whether a change carries a Transition: could a config or
# checkpoint written before it be READ WRONG afterwards? A renamed, removed or
# reinterpreted key, or a default whose meaning moved -- yes. A bug fix, a
# performance change, a new metric, a refactor -- no, however material.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transition:
    """The mechanical part of a state change.

    Applied in the order added -> renamed -> removed. `manual` is the judgment
    part: never applied, only reported, so a migration cannot quietly make a
    decision that needed a human."""

    added: dict[str, Any] = field(default_factory=dict)      # dotted -> value reproducing prior behavior
    renamed: dict[str, str] = field(default_factory=dict)    # old dotted -> new dotted, value carried unchanged
    removed: dict[str, str] = field(default_factory=dict)    # dotted -> why
    manual: dict[str, str] = field(default_factory=dict)     # dotted/topic -> what a human must decide
    migrate_fn: Optional[Callable[[dict], None]] = None      # escape hatch, mutates in place


@dataclass(frozen=True)
class Change:
    """One material functional change.

    `state` is the project state this change LANDED IN. A change carrying a
    `transition` is what created that state (so it is one higher than the change
    before it); a change without one sits at the current state and does not move
    it."""

    state: int
    summary: str                                  # what functional behavior changed
    components: tuple[str, ...] = ()              # principal files/modules affected
    invariants: tuple[str, ...] = ()              # what must hold afterwards
    validation: tuple[str, ...] = ()              # what was run to demonstrate it
    commit: Optional[str] = None
    transition: Optional[Transition] = None       # present iff this change moves state

    @property
    def moves_state(self) -> bool:
        return self.transition is not None


# ---------------------------------------------------------------------------
# The history. Append; never edit a shipped record.
# ---------------------------------------------------------------------------

STATE_HISTORY: tuple[StateTransition, ...] = (
    StateTransition(
        version=1,
        summary=(
            "Baseline. Introduces `project_state_version` and this module. State 1 "
            "is the config schema as it stands at the start of the infrastructure "
            "stabilization pass, with configs/mk_dev.yaml as the canonical master. "
            "The key changes recorded here are the retirements that had accumulated "
            "in utils._RETIRED_KEYS without version stamps: previously a config "
            "carrying any of them was rejected at load with no route forward, and "
            "the mechanical subset can now be repaired instead."
        ),
        components=('config_state.py', 'utils.py', 'configs/mk_dev.yaml'),
        added={VERSION_KEY: 1},
        # Pure renames ONLY: same meaning, same units, value carried across
        # untouched. Anything whose interpretation moved is in `manual` below,
        # because carrying a value across a reinterpretation is exactly the silent
        # corruption the retired-key gate exists to prevent.
        renamed={
            'adaptive_lr.reset_loss_abs': 'adaptive_lr.divergence_loss_abs',
            'adaptive_lr.reset_grad_abs': 'adaptive_lr.divergence_grad_abs',
            'adaptive_lr.cut_ratio': 'adaptive_lr.divergence_cut',
            'buffers.anchor_buffer.health_gate_r2': 'buffers.anchor_buffer.health_gate_floor',
        },
        # Deleted outright: the mechanism the key drove no longer exists, so
        # dropping the key reproduces current behavior exactly. Reasons are held
        # in utils._RETIRED_KEYS and not duplicated here (PROTOCOL: one fact, one
        # home) -- these are the keys, that dict is the argument.
        removed={
            k: 'see utils._RETIRED_KEYS'
            for k in (
                'gpu_util_floor',
                'batch_growth_min_gain',
                'step_probe',
                'adaptive_lr.servo.ceiling_halflife_steps',
                'adaptive_lr.servo.trigger',
                'adaptive_lr.servo.discovery',
                'adaptive_lr.trigger',
                'adaptive_lr.boost',
                'adaptive_lr.discovery',
                'adaptive_lr.damage',
                'adaptive_lr.enabled',
                'adaptive_lr.hold_steps',
                'adaptive_lr.decay_halflife_steps',
                'adaptive_lr.decay_floor_scale',
                'adaptive_lr.cut_loss_abs',
                'adaptive_lr.cut_grad_abs',
                'adaptive_lr.fire_cooldown_steps',
                'adaptive_lr.recovery_target_frac',
                'adaptive_lr.recovery_wait_steps',
                'adaptive_lr.recovery_ramp_steps',
                'reuse_prior',
                'terminal_logw_std',
                'terminal_box_violation',
                'terminal_frozen_steps',
                'integrator.min_traj_length',
                'integrator.max_traj_length',
                'integrator.traj_length_strategy',
                'integrator.discretizer',
                'integrator.discretizer_max_ratio',
                'z_calibration.unclipped',
                'buffers.anchor_buffer.mcmc',
                'buffers.replay_buffer.admit_cap_max',
                'buffers.replay_buffer.admit_cap_min',
                'buffers.replay_buffer.admit_cap_health_h0',
                'buffers.replay_buffer.admit_temperature',
            )
        },
        # Renamed AND reinterpreted, or replaced by a key with different
        # semantics. The old value is not transferable and the migration stops
        # short of deciding a new one.
        manual={
            'max_reloads': (
                "renamed -> max_reloads_per_1k_steps AND changed from a COUNT to a "
                "RATE. The old integer is not a valid value for the new key: a "
                "budget of 5 reloads is not 5 reloads per 1000 steps. Choose the "
                "rate the run should carry (canonical: 0.2)."
            ),
            'buffers.anchor_buffer.health_gate_zerr': (
                "renamed -> health_gate_ceiling, but the RULER changed with it. The "
                "bar now applies to tb_resid_clipped (signed, beta-bounded) rather "
                "than tb_err_worst (unbounded RMS, ~18-21 when healthy). A bar does "
                "not survive a ruler swap; carrying the old number across would "
                "gate on a threshold that means something else. Canonical: 0.5."
            ),
            'adaptive_lr.servo': (
                "the block was split, not renamed: seed_lr -> adaptive_lr.seed_lr "
                "and bounds -> adaptive_lr.bounds carry across unchanged, while "
                "target/clip/period/min_readings/max_bad_rate belonged to the online "
                "median servo and have no successor. Migrate the two survivors by "
                "hand and drop the rest."
            ),
            'batch_growth_max_step_regression': (
                "replaced by batch_growth_min_throughput_gain, which is a DIFFERENT "
                "criterion: the old key bounded step-time regression, the new one "
                "sets a throughput-saturation floor. The values are not "
                "interchangeable. Canonical: 0.05, and it must stay below "
                "batch_growth_factor - 1 or every jump is rejected and the batch "
                "freezes."
            ),
        },
        invariants=(
            "A config at CURRENT_STATE_VERSION passes migrate() unchanged.",
            "Migration never alters the problem identity (energy_function, paths, "
            "conditioning flags, space_groups, temperature) -- those are the "
            "checkpoint schema_version's axis, not this one.",
            "A key appears in at most one of added/renamed/removed/manual.",
        ),
        validation=(
            "test_config_state.py: round-trip at current version is a no-op; each "
            "mechanical class applies; manual keys are reported and never rewritten; "
            "the canonical config migrates clean.",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Dotted-path helpers. Migration operates on the RAW dict from load_yaml, before
# dict2namespace: dotted paths and add/rename/remove are dict operations, and
# doing them pre-namespace keeps the transform independent of how the trainer
# happens to consume the config.
# ---------------------------------------------------------------------------

def _get(cfg: dict, dotted: str):
    """(container, leaf, present) for a dotted path, or (None, leaf, False)."""
    node = cfg
    parts = dotted.split('.')
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            return None, parts[-1], False
        node = node[p]
    if not isinstance(node, dict):
        return None, parts[-1], False
    return node, parts[-1], parts[-1] in node


def _set(cfg: dict, dotted: str, value) -> None:
    """Set a dotted path, creating intermediate dicts as needed."""
    node = cfg
    parts = dotted.split('.')
    for p in parts[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    node[parts[-1]] = value


def config_version(cfg: dict) -> int:
    """The state a config declares. Unstamped configs are state 0 by definition."""
    v = cfg.get(VERSION_KEY, None)
    if v is None:
        return UNSTAMPED_VERSION
    return int(v)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

@dataclass
class MigrationReport:
    """What a migration did, and what it refused to do.

    `needs_judgment` is the load-bearing field: it is the honest answer to
    "what remains ambiguous", and a caller that ignores it has turned a
    reported migration into a silent guess.
    """

    from_version: int
    to_version: int
    applied: list[str] = field(default_factory=list)
    needs_judgment: list[str] = field(default_factory=list)
    unchanged: bool = True

    def render(self) -> str:
        lines = [f"migrate: state {self.from_version} -> {self.to_version}"]
        if self.unchanged:
            lines.append("  no changes required")
            return "\n".join(lines)
        for a in self.applied:
            lines.append(f"  applied   {a}")
        for m in self.needs_judgment:
            lines.append(f"  JUDGMENT  {m}")
        if self.needs_judgment:
            lines.append(
                f"\n{len(self.needs_judgment)} item(s) need a decision -- the migration "
                f"did NOT choose for you. Resolve each against configs/mk_dev.yaml."
            )
        return "\n".join(lines)


def migrate(cfg: dict, from_version: Optional[int] = None,
            to_version: int = CURRENT_STATE_VERSION) -> tuple[dict, MigrationReport]:
    """Transform a config from its recorded state to `to_version`.

    Returns a NEW dict plus the report; the input is not mutated. Transitions
    apply in version order, each one add -> rename -> remove, then any escape
    hatch, and finally the version stamp.

    A key listed in `manual` is left exactly as found and reported. That is the
    whole point: the alternative -- picking a plausible value -- produces a
    config that loads and trains on a number nobody chose.
    """
    cfg = copy.deepcopy(cfg)
    if from_version is None:
        from_version = config_version(cfg)
    report = MigrationReport(from_version=from_version, to_version=to_version)

    if from_version > to_version:
        raise ValueError(
            f"config declares state {from_version}, which is AHEAD of this code's "
            f"state {to_version}. Downgrade migrations are not defined: the config "
            f"may carry keys this code has no schema for. Check out the matching "
            f"revision instead.")

    for tr in STATE_HISTORY:
        if not (from_version < tr.version <= to_version):
            continue

        for dotted, value in tr.added.items():
            node, leaf, present = _get(cfg, dotted)
            if not present:
                _set(cfg, dotted, value)
                report.applied.append(f"v{tr.version} add    {dotted} = {value!r}")

        for old, new in tr.renamed.items():
            node, leaf, present = _get(cfg, old)
            if present:
                value = node.pop(leaf)
                _set(cfg, new, value)
                report.applied.append(f"v{tr.version} rename {old} -> {new} (value kept: {value!r})")

        for dotted, why in tr.removed.items():
            node, leaf, present = _get(cfg, dotted)
            if present:
                dropped = node.pop(leaf)
                report.applied.append(f"v{tr.version} drop   {dotted} (was {dropped!r}) -- {why}")

        for dotted, why in tr.manual.items():
            _, _, present = _get(cfg, dotted)
            if present:
                report.needs_judgment.append(f"v{tr.version} {dotted}: {why}")

        if tr.migrate_fn is not None:
            tr.migrate_fn(cfg)
            report.applied.append(f"v{tr.version} custom transform")

    cfg[VERSION_KEY] = to_version
    report.unchanged = not (report.applied or report.needs_judgment)
    return cfg, report


def render_history_markdown() -> str:
    """The chronology, generated from the records. Never hand-edit the output:
    edit the records and regenerate, so the description and the transform that
    implements it cannot disagree."""
    out = [
        "# Project state history",
        "",
        "Generated from `config_state.STATE_HISTORY` -- do not edit by hand.",
        "One entry per project-state transition. The line-level diff is in git;",
        "what is recorded here is the semantic delta needed to migrate state.",
        "",
    ]
    for tr in STATE_HISTORY:
        out.append(f"## State {tr.version}")
        out.append("")
        out.append(tr.summary)
        out.append("")
        if tr.commit:
            out.append(f"**Commit:** `{tr.commit}`")
            out.append("")
        if tr.components:
            out.append(f"**Components:** {', '.join(f'`{c}`' for c in tr.components)}")
            out.append("")
        for title, items, fmt in (
            ("Added", tr.added, lambda k, v: f"`{k}` = `{v!r}`"),
            ("Renamed", tr.renamed, lambda k, v: f"`{k}` -> `{v}`"),
            ("Removed", tr.removed, lambda k, v: f"`{k}` -- {v}"),
            ("Requires judgment", tr.manual, lambda k, v: f"`{k}` -- {v}"),
        ):
            if items:
                out.append(f"**{title}:**")
                out.append("")
                for k, v in items.items():
                    out.append(f"- {fmt(k, v)}")
                out.append("")
        if tr.invariants:
            out.append("**Invariants:**")
            out.append("")
            out.extend(f"- {i}" for i in tr.invariants)
            out.append("")
        if tr.validation:
            out.append("**Validation:**")
            out.append("")
            out.extend(f"- {v}" for v in tr.validation)
            out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------

def _main():
    import argparse
    import yaml

    ap = argparse.ArgumentParser(prog='config_state', description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('history', help='print the state history as Markdown')
    mg = sub.add_parser('migrate', help='migrate a config to the current state')
    mg.add_argument('path')
    mg.add_argument('--from-version', type=int, default=None,
                    help='override the state the config declares')
    mg.add_argument('--write', action='store_true',
                    help='write the migrated config back in place (default: report only)')
    args = ap.parse_args()

    if args.cmd == 'history':
        print(render_history_markdown())
        return

    with open(args.path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    migrated, report = migrate(cfg, from_version=args.from_version)
    print(report.render())
    if args.write:
        if report.needs_judgment:
            # Writing a config that still carries an unresolved key would produce
            # a file stamped at the current state while holding a value from a
            # retired interpretation -- a provenance claim that is false.
            raise SystemExit(
                "\nrefusing to write: unresolved items above. Fix them in the source "
                "config first, then re-run.")
        with open(args.path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(migrated, f, sort_keys=False, default_flow_style=False)
        print(f"\nwrote {args.path}")


if __name__ == '__main__':
    _main()
