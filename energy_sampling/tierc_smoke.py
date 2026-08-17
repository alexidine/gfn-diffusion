"""
Tier C of the consolidation acceptance criterion: does the RUN change?

`config_snapshot` covers tiers A and B -- the resolved config and the
deterministic pre-runtime state, both pure functions of the YAML and the seed.
Tier C is the one that needs a process: losses, energies and metrics over N
steps. `docs/design/infrastructure_stabilization.md` §1 states it, and §1.1
records that the runtime half of the mode-safety audit is blocked on the same
instrument -- proving the trainer never READS a key needs a run.

    python -m tierc_smoke --null configs/mk_dev.yaml
    python -m tierc_smoke --negative-control configs/mk_dev.yaml
    python -m tierc_smoke base.yaml configs/mk_dev.yaml

`latent_gaussian` only, deliberately. §1 picks it as the sharpest instrument
available: an analytic target, so the same-config spread is ZERO and tier C
collapses from "within a measured floor" to an exact test. The MLIP route's
tier C needs a measured repeat-run spread and is a different job.

=============================================================================
WHY THIS REUSES `benchmarks/registry.yaml` RATHER THAN DECLARING ITS OWN SETUP
=============================================================================
Five config settings silently unfix a run's work quantity, and the registry's
`defaults.overrides` already neutralises all five. The sharpest is not in that
list, though -- it is in `registry.epochs_for`, and it is this:

    `epochs` IS AN ABSOLUTE STEP INDEX, NOT A COUNT.

`train.py` runs `trange(init_step, args.epochs + 1)` with `init_step` restored
from the checkpoint. A harness that writes `epochs: 30` against a step-6680
resume runs ZERO steps, raises nothing, and reports a clean empty trace -- and
two clean empty traces compare equal, so a from-scratch tier-C harness passes
while measuring nothing. That is the exact shape of failure this file exists to
avoid, so the budget is computed from the resume step and then VERIFIED against
what actually executed (`meta.executed_steps`), because a computed budget is
still a belief until something counts the steps.

Note the OFF-BY-ONE the registry's formula carries: `trange(a, b + 1)` runs
`b - a + 1` iterations, so `epochs = resume + warmup + measure` executes one
step more than `warmup + measure`. Harmless for a 500-step throughput
benchmark; not harmless for an exact comparison, so `epochs_for_steps` below
subtracts it and the verification catches it if that is ever wrong.

=============================================================================
WHAT IS COMPARED, AND WHAT IS RECORDED BUT NOT COMPARED
=============================================================================
A trace holds two kinds of number and conflating them makes an exact test
impossible: `train_step_time` differs between two identical runs on the first
try, every time. So every value is captured and then SPLIT --

    deterministic   losses, coefficients, learning rates, counts, gate values
    wallclock       times, rates, occupancy, VRAM

-- and only the first half is compared. The split is by explicit rule
(`WALLCLOCK_*` below), the wallclock half is still written to the trace file,
and the classification is printed. Dropping a key silently is how a comparator
comes to certify a run against a quantity it stopped looking at.

=============================================================================
WHY `--deterministic strict` IS THE DEFAULT
=============================================================================
Because the null test measured it, not because it seemed prudent. On this box
(RTX 5080, torch 2.8.0+cu128, `latent_gaussian`, 30 steps, seed 12345) two
launches of the identical config produce:

    default torch settings   7 of 243 logged values differ
    --deterministic strict   0 differ

Two things about that 7 are worth keeping. First, they are ALL grouped
reductions -- `bwd/relative_under`, `bwd/relative_under_wcen`,
`bwd/z_grad_worst`, `bwd/cond_tb_err`, `bwd/tb_err_worst` -- differing in the
last one or two decimal digits, i.e. float32 rounding under a
reduction whose ORDER is not fixed. Second, and more useful: every loss, every
learning rate and the other 236 values were already bit-identical without the
flag. The training path was reproducible; a handful of *metrics* computed off
it were not, and which ones differ changes between runs, which is the signature
of atomics rather than of anything the config did.

`strict` (rather than `warn_only=True`) did NOT raise here. That is the whole
result: every op on this path HAS a deterministic implementation and torch
simply does not select it by default. So the flag costs nothing but a kernel
choice, and `--deterministic off` reproduces the non-zero null on demand.

This is also why the tier-C bar can be `torch.equal`-grade on this route and
must not be on the MLIP route. There, the spread is a property of the model
(UMA is not bit-reproducible on GPU); here it was a property of the default
kernel selection, and it went away when asked.

=============================================================================
THE ORDER THE INSTRUMENT IS BUILT IN
=============================================================================
1. `--null`         same config twice, traces must be BIT-IDENTICAL. On a
                    deterministic target the same-config spread is zero; a
                    non-zero null means the instrument is broken and nothing
                    downstream means anything.
2. `--negative-control`
                    one loss coefficient altered, traces must DIFFER. A harness
                    that cannot see a real change is worse than none: it
                    launders an unvalidated config as validated.
3. the comparison   only then.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

#: The registry entry whose overrides this harness borrows. Its `work` block
#: describes a 500-step throughput benchmark; only the neutralising `overrides`
#: and the `epochs` arithmetic are reused here, not the step budget.
BENCHMARK_ID = 'toy-latentgauss-fused-uncond'
#: `configs/problems.yaml` key. Scope is this problem and no other.
PROBLEM = 'latent_gaussian'
#: Small enough that no evaluation fires: `train.py` forces one at step 50
#: regardless of `eval_period`, and an eval pulls in figures, buffer saves and
#: array-valued metrics that have nothing to do with the question here.
DEFAULT_STEPS = 30
DEFAULT_SEED = 12345

# Keys that describe how long something took or how much of the machine it used.
# Recorded, never compared. Two identical runs disagree on every one of them.
WALLCLOCK_EXACT = frozenset({
    'samples_per_sec', 'initialization_time',
})
#: Wins over every rule below, including the registry's own grouping. `Batch
#: Size` sits in the registry's `cost` group because it is the DENOMINATOR of a
#: throughput number, not because it is a timing -- it is an exact integer and
#: one of the most informative things in the trace, since a config that changed
#: the batch changed everything downstream of it.
DETERMINISTIC_OVERRIDE = frozenset({'Batch Size'})
WALLCLOCK_PREFIXES = ('gpu/', 'vram/')
#: `_ms` is here because the NULL TEST put it here. `probe/churn_add_ms_max` and
#: `probe/churn_purge_ms_max` are millisecond timers that matched none of the
#: other rules, and they were the entire content of a 42-value non-zero null at
#: 600 steps -- a length that first reaches the buffer-churn code they time.
#: Note it does NOT catch `_rms` (`tracker/logw_std_rms` and friends survive as
#: deterministic), which was checked against all 522 logged keys rather than
#: assumed.
#:
#: THE NULL IS WHAT CALIBRATES THIS LIST, not judgement about what a name looks
#: like. Every entry should be traceable to two identical runs disagreeing on it;
#: a name added because it seemed timing-ish is a quantity silently dropped from
#: the comparison.
WALLCLOCK_SUBSTRINGS = ('_time', 'time_', 'seconds', 'ms_per_sample', '_ms',
                        'frac_of_step', 'frac_outside_step', 'per_sec',
                        '_s_', 'wall')
#: MACE phase-split timers end in `_s`; matched separately so the substring rule
#: above does not have to guess at a one-character suffix.
WALLCLOCK_SUFFIXES = ('_s',)


# ---------------------------------------------------------------- classify --

def registry_wallclock_metrics() -> set[str]:
    """Timing/occupancy metric names the benchmark registry already names.

    Taken from the registry rather than re-listed, so a metric renamed there is
    renamed here. It is not the whole answer -- the registry catalogues the
    metrics a BENCHMARK reports, and a training run logs several hundred more --
    which is why `is_wallclock` also carries pattern rules."""
    try:
        from benchmarks import registry
        reg = registry.load()
    except Exception:
        return set()
    out: set[str] = set()
    for group in ('cost', 'occupancy', 'eval_cost'):
        out.update(reg['metrics'].get(group, {}))
    for name in reg['metrics'].get('energy', {}):
        if 'calls' not in name:          # energy/calls is a COUNT, and exact
            out.add(name)
    for name in reg['metrics'].get('energy_phase_split', {}):
        if 'calls' not in name:
            out.add(name)
    return out


def is_wallclock(name: str, registry_set: Optional[set] = None) -> bool:
    if name in DETERMINISTIC_OVERRIDE:
        return False
    if registry_set and name in registry_set:
        return True
    if name in WALLCLOCK_EXACT:
        return True
    if any(name.startswith(p) for p in WALLCLOCK_PREFIXES):
        return True
    if any(s in name for s in WALLCLOCK_SUBSTRINGS):
        return True
    return any(name.endswith(s) for s in WALLCLOCK_SUFFIXES)


# ------------------------------------------------------------------ config --

def _deep_merge(base: dict, over: dict) -> dict:
    """Deep, for the reason `registry.resolved_overrides` gives: a shallow merge
    drops one side of a nested block silently, which is the shape of every config
    bug this project has logged."""
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


#: Keys in a `configs/problems.yaml` entry that describe the problem to a READER
#: rather than to the trainer. Merging them in would inject unknown top-level
#: keys into the config; they are prose, not settings.
_PROBLEM_PROSE_KEYS = ('description', 'domain', 'conditioning')


def problem_overlay(problem: str = PROBLEM) -> dict:
    """The problem-intrinsic settings, from `configs/problems.yaml`.

    That file is Phase 1.4's replacement for `mode_presets.yaml` and is the
    declared home for exactly this. It is NOT yet read by `train.py` -- nothing
    outside its own tests loads it -- so this harness reads it directly."""
    path = os.path.join(HERE, 'configs', 'problems.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        doc = yaml.safe_load(f)
    problems = doc.get('problems', doc)
    if problem not in problems:
        raise KeyError(f'no problem {problem!r} in {path}; have {sorted(problems)}')
    entry = dict(problems[problem])
    for k in _PROBLEM_PROSE_KEYS:
        entry.pop(k, None)
    return entry


def problem_gap_fill(problem: str = PROBLEM) -> tuple[dict, list[str]]:
    """What `problems.yaml` is missing before the problem can actually RUN.

    Returned separately from `problem_overlay` and reported, not folded in
    quietly: this is a gap in the Phase 1.4 registry, and a harness that patches
    it invisibly is a harness that stops the gap from ever being closed.

    Two keys, both fatal as they stand:

      * `prior_path: null` -- `Modeller.init_prior_dataset` opens it with
        `torch.load(self.args.prior_path)`, which raises on None. There is no
        null branch.
      * `analyze_kwargs: {}` -- the analytic gaussian needs its centre `c` and
        width `w`; without them the target is not the target the closed form
        describes.

    The values come from `configs/gauss_aug12/spec.py`, which is the single
    source of truth for this target (prior generation, config generation and the
    closed-form check all import it), rather than being retyped here. A prior
    drawn at one width and scored at another trains perfectly well and reports a
    wrong log Z, with nothing to see in either file on its own."""
    if problem != 'latent_gaussian':
        return {}, []
    sys.path.insert(0, os.path.join(HERE, 'configs', 'gauss_aug12'))
    try:
        import spec                                          # noqa: E402
    finally:
        sys.path.pop(0)

    sg = 1                       # problems.yaml declares space_groups: [1]
    fill = {
        'prior_path': spec.prior_path(sg),
        'molecules_path': spec.prior_path(sg),
        'energy_config': {
            'temperature': spec.T,
            'log_temperature_range': [0.0, 0.0],
            'bounding_coeff': spec.BOUNDING_COEFF,
            'reduction_coeff': spec.REDUCTION_COEFF,
            'density_coeff': 0.0,
            'lj_coeff': 1.0,
            'lj_rescale': None,
            'reward_range': None,
            'internal_oom_recovery': False,
            'analyze_kwargs': {'c': spec.target_c(sg), 'width': spec.WIDTH},
        },
        'model': {'hold_dead_latent_rows': True, 'periodic_centroids': False},
    }
    notes = [
        f'problems.yaml:{problem}.prior_path is null; init_prior_dataset '
        f'torch.loads it unconditionally -> supplied {spec.PRIOR_STEM.format(sg=sg)}.pt '
        f'from gauss_aug12/spec.py',
        f'problems.yaml:{problem}.analyze_kwargs is empty; the analytic target '
        f'needs c and width -> supplied from gauss_aug12/spec.py '
        f'(MODE={spec.MODE}, WIDTH={spec.WIDTH}, k={spec.BOUNDING_COEFF})',
    ]
    return fill, notes


def _retired_paths_in(cfg: dict) -> list[tuple[str, str]]:
    """Dotted paths in `cfg` that current code retires, with the stated reason.

    Reads `utils._RETIRED_KEYS` -- the same table `preflight_config` raises on --
    rather than keeping a second list, because a second list is a list that goes
    stale exactly when it matters."""
    import utils
    found = []
    for dotted, why in utils._RETIRED_KEYS.items():
        node: Any = cfg
        parts = dotted.split('.')
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                node = None
                break
            node = node[p]
        if isinstance(node, dict) and parts[-1] in node:
            found.append((dotted, why))
    return found


def pin_auto_lr_without_sensor(cfg: dict) -> list[str]:
    """Translate a PRE-SENSOR config's `auto` rates into the floats they trained
    at, in place. Returns the notes; returns [] and touches nothing otherwise.

    WHY THIS IS NEEDED AT ALL. The pre-consolidation canonical config does not
    LOAD under current code, and migration cannot repair it. Two blockers, and
    they are different in kind:

      * five retired keys -- mechanical, and `config_state.migrate` fixes them;
      * `lr_policy/lr_back/lr_replay/lr_fused: auto` with no stage declaring an
        `lr_sensor`. Phase 1 made that a RAISING load gate. Migration refuses
        it, correctly: picking a sensor is judgment, and a migration that picks
        one produces a config that loads and trains on a number nobody chose.

    WHY PINNING IS A TRANSLATION AND NOT A GUESS. The gate's own rule
    (`config_invariants.auto_lr_requires_an_adaptive_sensor`) documents what the
    two spellings do:

      auto   servo-managed; seeded at adaptive_lr.seed_lr, owned by a sensor
      float  a fixed peak; takes the warmup envelope and divergence handling,
             and `peak_scale` never applies to it

    In the pre-consolidation config there was NO sensor, so nothing ever moved
    `peak_scale` off 1.0 and the rate sat at `seed_lr` for the whole run -- which
    the plan states independently ('previously neither did, so all four `auto`
    rates sat at the seed for entire runs'). Envelope: applies to both. Divergence
    handling: applies to both. `peak_scale`: 1.0 in both. So the float and the
    sensorless `auto` are the same run, and writing the float is how that run is
    spelled under the current schema.

    It is still an assumption, it is still reported on every run that triggers
    it, and it is deliberately NOT silent -- because what it does not preserve is
    the thing tier C is about to measure: the current config DOES declare
    sensors, so the LR trajectories diverge by design from here."""
    import config_invariants
    import utils

    auto = [k for k in utils._LR_KEYS if utils._is_auto(cfg.get(k))]
    if not auto:
        return []
    violations = config_invariants.auto_lr_requires_an_adaptive_sensor(
        cfg, auto_keys=auto)
    if not violations:
        return []
    seed = (cfg.get('adaptive_lr') or {}).get('seed_lr')
    if not isinstance(seed, (int, float)) or isinstance(seed, bool):
        raise ValueError(
            f'{auto} are `auto` with no stage sensor, and adaptive_lr.seed_lr is '
            f'{seed!r} -- there is no number to pin them to, so what this config '
            f'trained at cannot be reconstructed.')
    for k in auto:
        cfg[k] = float(seed)
    stages = sorted({getattr(v, 'detail', str(v)).split("'")[1]
                     for v in violations if "'" in getattr(v, 'detail', str(v))})
    return [f'REPAIR: {", ".join(auto)} were `auto` with no lr_sensor on '
            f'stage(s) {stages or "?"} -- pinned to adaptive_lr.seed_lr='
            f'{float(seed):g}, the rate this config actually trained at. '
            f'`auto` without a sensor is a raising load gate under current code; '
            f'see pin_auto_lr_without_sensor.']


def registry_overrides() -> tuple[dict, list[str]]:
    """`defaults.overrides` + the toy benchmark's own, via the registry's own
    resolver -- MINUS any key current code has retired.

    This is the block that neutralises checkpoint writes, archiving, batch
    growth, the throughput optimiser, the runaway-step guard, the knee recheck
    and figures. Three of those are the ones that make an exact comparison
    possible at all, because they are actuated by WALL CLOCK: `grow_batch_size`
    and `auto_batch_throughput_opt` read measured throughput, and
    `max_step_seconds` is a stopwatch. Left on, two identical runs take
    different actions and the null test can never be zero.

    THE DROP IS A STANDING GUARD, not a workaround for a known break. The
    registry used to set `ray_calibration.enabled: false`, and under current code
    both `ray_calibration` (moved under `adaptive_lr`) and
    `ray_calibration.enabled` (deleted -- a stage declaring
    `lr_sensor: {kind: ray}` IS the switch) are retired keys that hard-fail at
    load, so the overrides could not be applied verbatim: a harness that did so
    would die at preflight, and one that silenced the error would be running with
    the block dropped and not know it. That entry is now gone from the registry
    and this function drops nothing. It stays because the registry is a hand-kept
    file on the other side of a schema boundary and will drift again; the
    alternative is discovering the next retirement as a preflight failure 90
    seconds into a subprocess.

    Dropped keys are RETURNED, not swallowed -- the drop is what makes the
    override block differ from the registry's stated intent, and that difference
    belongs in the provenance rather than in this function's head.

    On what a ray drop would cost, should one recur: the registry disables
    periodic calibrations because their period is not the report period, so a
    fixed-length TIMING window contains a variable amount of them. That argument
    is about aliasing in a throughput measurement. It does not apply to an exact
    trace comparison, where periodic work is deterministic and appears
    identically on both sides -- and where a live ray probe makes the instrument
    MORE sensitive, not less."""
    from benchmarks import registry
    ov = registry.resolved_overrides(BENCHMARK_ID)
    dropped = []
    # Deepest first. `ray_calibration` and `ray_calibration.enabled` are BOTH
    # retired, and popping the parent first leaves the child's path dangling.
    for dotted, why in sorted(_retired_paths_in(ov),
                              key=lambda kv: -kv[0].count('.')):
        node: Any = ov
        parts = dotted.split('.')
        for p in parts[:-1]:
            node = node.get(p) if isinstance(node, dict) else None
        if not isinstance(node, dict) or parts[-1] not in node:
            continue                     # already removed with its parent
        value = node.pop(parts[-1])
        dropped.append(f'registry override {dotted}={value!r} DROPPED -- '
                       f'retired: {why.splitlines()[0]}')
    # A block emptied by the drop is removed as well: `ray_calibration: {}` is
    # still the retired top-level key, and preflight fires on PRESENCE.
    for key in [k for k, v in ov.items() if isinstance(v, dict) and not v]:
        ov.pop(key)
        dropped.append(f'registry override {key}={{}} removed -- emptied by the '
                       f'drops above, and the bare key is itself retired')
    return ov, dropped


def epochs_for_steps(resume_step: int, steps: int) -> int:
    """`epochs` such that exactly `steps` training steps execute.

    ABSOLUTE INDEX, not a count -- see the module docstring. The `- 1` is the
    `trange(init_step, epochs + 1)` inclusive bound; `registry.epochs_for` omits
    it because one extra step does not matter to a throughput measurement, and
    it does matter here."""
    return int(resume_step) + int(steps) - 1


def build_config(base_yaml: str,
                 steps: int = DEFAULT_STEPS,
                 seed: int = DEFAULT_SEED,
                 resume_step: int = 0,
                 device: str = 'cuda',
                 run_tag: str = 'tierc',
                 extra: Optional[dict] = None) -> tuple[dict, dict]:
    """Base config -> the config this harness actually runs, plus provenance.

    Layer order, each winning over the one before:

        base YAML  <  problem overlay  <  gap fill  <  registry overrides
                   <  harness pins  <  caller `extra`

    then `config_state.migrate`, then the LR pin, then a retired-key check.

    MIGRATION CANNOT BE THE PLACE THE REGISTRY'S RETIRED KEYS ARE HANDLED, which
    is why `registry_overrides` drops them before they are merged at all. Two
    reasons, and the second is the one that bites:

      * `migrate` applies only the transitions ABOVE the config's declared
        `project_state_version`. The current canonical config is already stamped
        at the current state, so every transition is skipped -- and a retired key
        a merge layer introduced sails straight through to a load failure.
      * even where migration does fire, its repair is wrong here. The
        `ray_calibration -> adaptive_lr.ray_calibration` rename MOVES THE BLOCK
        WHOLESALE, so merging the registry's `{enabled: false}` on top of a
        pre-consolidation config and then migrating would overwrite the real
        calibration parameters with the stub. The config would load, and the ray
        probe would be running on values nobody wrote.

    Migration is still applied, LAST, for the job it is actually for: repairing a
    baseline written against an older schema. Both sides of a comparison go
    through this identical pipeline, so nothing here can favour one of them."""
    import config_state

    with open(base_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    prov: dict[str, Any] = {'base_yaml': os.path.abspath(base_yaml),
                            'layers': [], 'notes': []}

    overlay = problem_overlay()
    _deep_merge(cfg, overlay)
    prov['layers'].append({'name': 'problems.yaml', 'keys': sorted(overlay)})

    fill, notes = problem_gap_fill()
    _deep_merge(cfg, fill)
    prov['layers'].append({'name': 'gap_fill(gauss_aug12/spec.py)', 'keys': sorted(fill)})
    prov['notes'].extend(notes)

    ov, dropped = registry_overrides()
    _deep_merge(cfg, ov)
    prov['layers'].append({'name': f'registry[{BENCHMARK_ID}]', 'keys': sorted(ov)})
    prov['notes'].extend(dropped)

    pins = {
        'seed': int(seed),
        'epochs': epochs_for_steps(resume_step, steps),
        'device': device,
        # Cold start, explicitly. The mk_dev defaults resolve
        # `continue_from_checkpoint: true` + `checkpoint_name: null` to a
        # rolling '{tag}_{run_name}_..._running.pt', so a harness that forgets
        # these resumes its own previous invocation -- and then the null test
        # compares run 2 against a warm start of run 1 and is meaningless.
        'continue_from_checkpoint': False,
        'checkpoint_name': None,
        'prior_model_name': None,
        'load_weights_only': False,
        'run_name': run_tag,
        'tag': 'tierc',
        # No eval inside the window. `eval_T` must equal `integrator.T` or
        # preflight refuses the config, so it is set from whatever T survived
        # the merge rather than pinned to a literal.
        'eval_period': 100000000,
        'figs_period': 100000000,
        'figure_period': 100000000,
        'wandb_mode': 'disabled',
    }
    _deep_merge(cfg, pins)
    cfg['eval_T'] = cfg.get('integrator', {}).get('T', cfg.get('eval_T'))
    prov['layers'].append({'name': 'harness pins', 'keys': sorted(pins)})

    if extra:
        _deep_merge(cfg, extra)
        prov['layers'].append({'name': 'caller extra', 'keys': sorted(extra)})

    migrated, report = config_state.migrate(cfg)
    prov['migration'] = {'applied': list(report.applied),
                         'manual': list(getattr(report, 'manual', []) or []),
                         'from_version': report.from_version,
                         'to_version': report.to_version}

    prov['notes'].extend(pin_auto_lr_without_sensor(migrated))

    # Fail HERE, not 90 seconds into a subprocess. Migration only fires for
    # transitions above the config's declared `project_state_version`, so a
    # config already stamped at the current state has every transition skipped
    # -- including the one that would have repaired a retired key a merge layer
    # introduced. Generating a config is not loading it, so the check that
    # loading performs is repeated at generation time.
    left = _retired_paths_in(migrated)
    if left:
        raise ValueError(
            'the assembled config still carries retired keys after migration '
            f'(declared state {report.from_version} -> {report.to_version}, so '
            f'earlier transitions were skipped):\n'
            + '\n'.join(f'  {k}' for k, _ in left))
    return migrated, prov


# --------------------------------------------------------------------- run --

def _venv_python() -> str:
    """The interpreter that has torch. `sys.executable` is right when this module
    is already running under it, which is the normal case."""
    return sys.executable


def run_trace(cfg: dict, out_path: str, *, workdir: Optional[str] = None,
              deterministic: bool = False, timeout: int = 3600) -> dict:
    """Launch one run in a FRESH PROCESS and return its trace.

    A separate process per run, not a loop inside one. The registry says the
    same thing about noise floors and for the same reason -- 're-timing inside
    one process measures step-to-step scatter, which is not the quantity a
    run-to-run comparison is tested against'. Two runs in one interpreter share
    the RNG, the allocator, cuDNN's autotune cache and every module-level
    singleton, so an in-process null test can pass for reasons that have nothing
    to do with the config."""
    workdir = workdir or tempfile.mkdtemp(prefix='tierc_')
    cfg_path = os.path.join(workdir, 'config.yaml')
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    env = dict(os.environ)
    # Offline is sanctioned for smoke tests and pure functionality checks only;
    # this is one. `disabled` goes further than `offline` -- no run directory is
    # written at all, so repeat invocations cannot differ by what the last one
    # left behind.
    env['WANDB_MODE'] = 'disabled'
    env['WANDB_SILENT'] = 'true'
    env['PYTHONHASHSEED'] = '0'
    env.setdefault('PYTHONPATH', os.pathsep.join([
        os.path.abspath(os.path.join(HERE, '..', '..', 'mxtaltools')),
        os.path.abspath(os.path.join(HERE, '..')),
    ]))
    if deterministic and deterministic != 'off':
        env['TIERC_DETERMINISTIC'] = str(deterministic)
        # Required by torch before deterministic cuBLAS reductions are allowed.
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    cmd = [_venv_python(), '-m', 'tierc_smoke', '_child', cfg_path, out_path]
    proc = subprocess.run(cmd, cwd=HERE, env=env, timeout=timeout,
                          capture_output=True, text=True)
    if not os.path.exists(out_path):
        raise RuntimeError(
            f'child produced no trace (exit {proc.returncode})\n'
            f'--- stdout tail ---\n{proc.stdout[-4000:]}\n'
            f'--- stderr tail ---\n{proc.stderr[-4000:]}')
    with open(out_path, 'r', encoding='utf-8') as f:
        trace = json.load(f)
    trace['meta']['returncode'] = proc.returncode
    trace['meta']['config_path'] = cfg_path
    if trace['meta'].get('error'):
        raise RuntimeError(
            f"run failed: {trace['meta']['error']}\n"
            f"--- stderr tail ---\n{proc.stderr[-4000:]}")
    return trace


# ------------------------------------------------------------------- child --

def _jsonable(v):
    """Value -> something JSON can hold EXACTLY, or a deterministic digest.

    Floats go through `json` unchanged, which uses `repr` and therefore round
    trips a float64 bit for bit -- so an equality test on the written file is an
    equality test on the number. Arrays are digested rather than expanded: a
    10,000-element histogram in a diff is unreadable, and a hash of its bytes
    answers 'did it change' exactly."""
    import numpy as np
    import torch

    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.detach().cpu().item()
        arr = v.detach().cpu().numpy()
        return {'__digest__': hashlib.sha256(
            np.ascontiguousarray(arr).tobytes()).hexdigest()[:16],
            'shape': list(arr.shape), 'dtype': str(arr.dtype)}
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return v.reshape(-1)[0].item()
        return {'__digest__': hashlib.sha256(
            np.ascontiguousarray(v).tobytes()).hexdigest()[:16],
            'shape': list(v.shape), 'dtype': str(v.dtype)}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    # wandb.Histogram and anything else: name the TYPE. Not silently dropped --
    # a key that vanishes from a trace reads as a key that never existed.
    return {'__unsupported__': type(v).__name__}


def _child(cfg_path: str, out_path: str) -> int:
    """One run, in this process. Never called directly -- `run_trace` spawns it.

    Everything captured here is captured by WRAPPING the trainer, never by
    editing it. A smoke harness that requires production code to carry hooks for
    it is a harness that changes the thing it measures."""
    trace: dict[str, Any] = {'meta': {}, 'steps': [], 'logged': []}
    meta = trace['meta']
    meta['cfg_path'] = cfg_path
    meta['error'] = None

    try:
        import torch

        mode = os.environ.get('TIERC_DETERMINISTIC', '')
        if mode:
            # STRICT by default (`warn_only=False`). `warn_only=True` turns an
            # op with no deterministic implementation into a warning and then
            # runs the nondeterministic kernel anyway -- so the flag reads as
            # 'this run is deterministic' while the thing that made it not
            # deterministic is untouched. If an op has no deterministic path,
            # the useful outcome is a raise that NAMES it.
            torch.use_deterministic_algorithms(True, warn_only=(mode == 'warn'))
            torch.backends.cudnn.benchmark = False
            meta['deterministic_algorithms'] = mode
        else:
            meta['deterministic_algorithms'] = False

        # `remaining[1]` -- utils.get_train_args reads the config path out of
        # argparse's leftovers positionally, so argv has to look like a launch.
        sys.argv = ['train.py', '--config', cfg_path]

        import train                                          # noqa: E402

        recorded_steps: list[dict] = []
        logged: list[dict] = []

        _orig_train_step = train.Modeller.train_step

        def _traced_train_step(self, step_type):
            self._tierc_sub = None
            loss = _orig_train_step(self, step_type)
            rec = {
                'step': int(self.step_ind),
                'step_type': str(step_type),
                'stage': str(getattr(self.protocol.stage, 'name', '?')),
                'loss': _jsonable(loss),
                'batch_size': int(self.batch_size),
            }
            # Learning rates per optimizer, every step. The LR is where a
            # config change shows up FIRST and most legibly -- an `auto` rate
            # and an explicit float resolve to the same number and mean
            # opposite things (config_snapshot's `lr_servo_managed` note), and
            # the difference is only visible once something moves.
            rec['lr'] = {k: float(o.param_groups[0]['lr'])
                         for k, o in sorted(self.optimizers.items())}
            if 'fused' in self.optimizers:
                rec['lr']['fused_flow'] = float(
                    self.optimizers['fused'].param_groups[-1]['lr'])
            for name in ('fwd_frac', 'bwd_frac', 'replay_frac'):
                if hasattr(self, name):
                    rec[name] = _jsonable(getattr(self, name))
            if getattr(self, '_tierc_sub', None) is not None:
                rec['sub_losses'] = self._tierc_sub
                self._tierc_sub = None
            recorded_steps.append(rec)
            return loss

        train.Modeller.train_step = _traced_train_step

        # Fused sub-losses: the per-branch decomposition of the one scalar the
        # step returns. Without it a change that moves fwd up and bwd down by
        # the same amount is invisible in the total.
        #
        # STASHED, NOT APPENDED. This fires from INSIDE `train_step`, before it
        # returns, so at this moment the last entry in `recorded_steps` is the
        # PREVIOUS step. An earlier version appended its own record when the
        # step indices did not match, which silently turned 1200 steps into 2019
        # records -- one extra per fused step. `verify_step_count` caught it,
        # which is the only reason it is not still there: the 30-step null never
        # reaches the fused stage, so no amount of running the short null would
        # have found this.
        _orig_sub = getattr(train.Modeller, 'record_fused_substep_losses', None)
        if _orig_sub is not None:
            def _traced_sub(self, sub_losses):
                self._tierc_sub = _jsonable(sub_losses)
                return _orig_sub(self, sub_losses)
            train.Modeller.record_fused_substep_losses = _traced_sub

        import wandb                                          # noqa: E402

        class _WandbProxy:
            """Stands in for the `wandb` module inside `train`'s namespace.

            NOT a patch of `wandb.log`. `wandb.init()` REBINDS the module-level
            `log`, `config` and `summary` onto the freshly created run, so a
            wrapper installed before `init` is silently replaced by the real
            one the moment training starts -- and the trace then comes back
            with zero logged points and no error, which is the failure this
            harness exists to not have. Substituting the module reference that
            `train` resolves against survives the rebind, because `self._real
            .log` is looked up at CALL time and therefore picks up whatever
            `init` installed."""

            def __init__(self, real, sink):
                self._real = real
                self._sink = sink

            def __getattr__(self, name):
                return getattr(self._real, name)

            def log(self, data=None, step=None, **kw):
                if isinstance(data, dict):
                    self._sink.append(
                        {'step': (int(step) if step is not None else None),
                         'values': {str(k): _jsonable(v)
                                    for k, v in sorted(data.items())}})
                return self._real.log(data, step=step, **kw)

        train.wandb = _WandbProxy(wandb, logged)

        modeller = train.Modeller()
        meta['seed'] = int(modeller.args.seed)
        meta['epochs'] = int(modeller.args.epochs)
        meta['device'] = str(modeller.args.device)
        meta['energy_function'] = str(modeller.args.energy_function)
        meta['batch_size_cfg'] = int(modeller.args.batch_size)
        meta['problem_hash'] = str(modeller.problem_hash)
        meta['init_step'] = int(modeller.step_ind or 0)
        meta['torch'] = torch.__version__
        meta['cuda_device'] = (torch.cuda.get_device_name(0)
                               if torch.cuda.is_available() else None)

        modeller.train()

        meta['final_step_ind'] = int(modeller.step_ind)
    except BaseException as e:                       # noqa: BLE001
        import traceback
        meta['error'] = f'{type(e).__name__}: {e}'
        meta['traceback'] = traceback.format_exc()[-6000:]
        try:
            trace['steps'] = recorded_steps          # noqa: F821
            trace['logged'] = logged                 # noqa: F821
        except Exception:
            pass
    else:
        trace['steps'] = recorded_steps
        trace['logged'] = logged

    meta['executed_steps'] = len(trace['steps'])
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(trace, f, indent=1, sort_keys=True)
    return 0 if not meta['error'] else 1


# ------------------------------------------------------------------ split --

def split_trace(trace: dict) -> tuple[dict, dict]:
    """Trace -> (deterministic, wallclock). Both are returned; only the first is
    ever compared, and the second is written out so nothing is hidden."""
    reg = registry_wallclock_metrics()
    det: dict[str, Any] = {'steps': [], 'logged': []}
    wall: dict[str, Any] = {'steps': [], 'logged': []}

    for rec in trace.get('steps', []):
        d, w = {}, {}
        for k, v in rec.items():
            (w if is_wallclock(k, reg) else d)[k] = v
        det['steps'].append(d)
        wall['steps'].append(w)

    for entry in trace.get('logged', []):
        d = {'step': entry.get('step'), 'values': {}}
        w = {'step': entry.get('step'), 'values': {}}
        for k, v in (entry.get('values') or {}).items():
            (w if is_wallclock(k, reg) else d)['values'][k] = v
        det['logged'].append(d)
        wall['logged'].append(w)
    return det, wall


# --------------------------------------------------------------- compare ---

@dataclass
class TraceComparison:
    steps_a: int = 0
    steps_b: int = 0
    diffs: list[tuple[str, Any, Any]] = field(default_factory=list)
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return (self.steps_a == self.steps_b
                and not self.diffs and not self.only_a and not self.only_b)

    def render(self, limit: int = 60) -> str:
        lines = []
        if self.steps_a != self.steps_b:
            lines.append(f'STEP COUNT DIFFERS: {self.steps_a} vs {self.steps_b}')
        n = len(self.diffs) + len(self.only_a) + len(self.only_b)
        lines.append('TRACES IDENTICAL' if self.identical
                     else f'TRACES DIFFER in {n} quantities')
        for path, a, b in self.diffs[:limit]:
            lines.append(f'  ~ {path}: {a!r} -> {b!r}')
        if len(self.diffs) > limit:
            lines.append(f'  ... {len(self.diffs) - limit} more changed')
        for p in self.only_a[:limit]:
            lines.append(f'  - {p} (reference only)')
        for p in self.only_b[:limit]:
            lines.append(f'  + {p} (candidate only)')
        return '\n'.join(lines)


def _flatten(obj, prefix='') -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f'{prefix}.{k}' if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f'{prefix}[{i}]'))
    else:
        out[prefix] = obj
    return out


def _same(a, b) -> bool:
    """Equality for trace values, with ONE deviation from `==`: NaN equals NaN.

    Not a tolerance. `float('nan') != float('nan')` is correct IEEE behaviour and
    wrong here -- a metric that is NaN in both runs is a metric that did not
    change, and reporting `nan -> nan` as a difference is the comparator failing,
    not the run. `zmatch/*_level` is NaN before the z-match servo has anything to
    report, which is every short run.

    A metric that is NaN on one side and a number on the other is still caught:
    only the both-NaN case is folded."""
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return True
    return a == b


def compare_traces(a: dict, b: dict) -> TraceComparison:
    """Exact comparison of the deterministic halves. No tolerance, on purpose:
    §1 says the same-config spread on a deterministic target is ZERO, so a
    tolerance here would be a number picked by eye -- the failure mode the whole
    three-tier criterion is written to avoid."""
    da, _ = split_trace(a)
    db, _ = split_trace(b)
    cmp = TraceComparison(steps_a=len(a.get('steps', [])),
                          steps_b=len(b.get('steps', [])))
    fa, fb = _flatten(da), _flatten(db)
    for path in sorted(set(fa) | set(fb)):
        if path not in fb:
            cmp.only_a.append(path)
        elif path not in fa:
            cmp.only_b.append(path)
        elif not _same(fa[path], fb[path]):
            cmp.diffs.append((path, fa[path], fb[path]))
    return cmp


def verify_step_count(trace: dict, expected: int) -> tuple[bool, str]:
    """Did the run execute the number of steps it was asked for?

    Checked explicitly rather than assumed, because the failure it guards is
    silent: `epochs` is an absolute index, so a mis-computed budget produces a
    run of ZERO steps that raises nothing. Three independent readings have to
    agree -- the budget, the loop's final index, and the number of train_step
    calls that actually returned."""
    meta = trace.get('meta', {})
    got = int(meta.get('executed_steps', 0))
    init = int(meta.get('init_step', 0))
    epochs = int(meta.get('epochs', -1))
    final = meta.get('final_step_ind')
    want_epochs = epochs_for_steps(init, expected)
    problems = []
    if got != expected:
        problems.append(f'executed {got} steps, asked for {expected}')
    if epochs != want_epochs:
        problems.append(f'epochs={epochs}, expected {want_epochs} '
                        f'(init_step={init} + {expected} - 1)')
    if final is not None and int(final) != epochs:
        problems.append(f'loop ended at step_ind={final}, epochs={epochs}')
    if got == 0:
        problems.append('ZERO STEPS -- this is the absolute-step-index trap; '
                        'a zero-step trace compares equal to another zero-step '
                        'trace and certifies nothing')
    # Every record must come from the `train_step` wrapper, which is the only
    # thing that sets `step_type`. A capture hook that appends its own record
    # inflates the count without executing anything, and the inflation is upward
    # -- so it reads as MORE work rather than as a broken instrument.
    stray = [i for i, s in enumerate(trace.get('steps') or [])
             if 'step_type' not in s]
    if stray:
        problems.append(f'{len(stray)} step records were not written by the '
                        f'train_step wrapper (first at index {stray[0]})')
    ok = not problems
    return ok, ('step count verified: %d steps, init_step=%d, epochs=%d'
                % (got, init, epochs)) if ok else '; '.join(problems)


def verify_capture(trace: dict, expected_steps: int) -> tuple[bool, str]:
    """Did the instrument actually RECORD, as opposed to complete?

    A separate question from `verify_step_count`, and it has to be asked
    separately. The first version of this harness ran all 30 steps, verified the
    count, and captured ZERO metrics -- `wandb.init` had rebound `wandb.log` out
    from under the wrapper. Nothing raised. Two empty capture sets compare equal,
    so the null test would have passed and reported that the instrument was
    sound.

    So: an empty channel is a FAILURE, not an absence."""
    problems = []
    steps = trace.get('steps') or []
    logged = trace.get('logged') or []

    losses = [s.get('loss') for s in steps if 'loss' in s]
    if len(losses) != len(steps):
        problems.append(f'{len(steps) - len(losses)} step records carry no loss')
    if losses and not all(isinstance(v, float) and v == v and abs(v) != float('inf')
                          for v in losses):
        problems.append('non-finite or non-numeric loss in the trace')

    # `train.py` reports on a 10-step grid, so any window of 10 or more steps
    # must produce at least one logged point. Zero is the signature of a capture
    # hook that is no longer attached to anything.
    want_logged = expected_steps // 10
    if expected_steps >= 10 and len(logged) < want_logged:
        problems.append(f'{len(logged)} logged points, expected at least '
                        f'{want_logged} on the 10-step reporting grid -- the '
                        f'metric capture is detached')
    det, _ = split_trace(trace)
    n_det = sum(len(e['values']) for e in det['logged'])
    if logged and n_det == 0:
        problems.append('every logged value classified as wallclock; nothing '
                        'deterministic left to compare')
    ok = not problems
    return ok, (f'capture verified: {len(steps)} losses, {len(logged)} logged '
                f'points, {n_det} deterministic values') if ok else '; '.join(problems)


# ------------------------------------------------------------------- CLI ----

def _outdir() -> str:
    d = os.environ.get('TIERC_OUTDIR') or tempfile.mkdtemp(prefix='tierc_out_')
    os.makedirs(d, exist_ok=True)
    return d


def _run_one(base_yaml, label, steps, seed, device, deterministic, extra=None,
             outdir=None):
    outdir = outdir or _outdir()
    cfg, prov = build_config(base_yaml, steps=steps, seed=seed, device=device,
                             run_tag=label, extra=extra)
    work = os.path.join(outdir, label)
    os.makedirs(work, exist_ok=True)
    # Checkpoints are already suppressed by `checkpoint_read_only`, but point
    # the directory at the scratch tree as well -- defence in depth against a
    # write path the flag does not cover.
    cfg['checkpoints_dir'] = os.path.join(work, 'ckpt')
    os.makedirs(cfg['checkpoints_dir'], exist_ok=True)
    # Scaled to the budget, with a floor for process start and the prior scan.
    # A fixed cap silently turns a longer window into a timeout, and a timeout
    # produces no trace at all -- which reads as a crash rather than as the run
    # having been cut off.
    timeout = 600 + int(steps) * 5
    out = os.path.join(outdir, f'{label}.trace.json')
    print(f'[tierc] running {label}: {base_yaml} ({steps} steps, seed {seed})',
          flush=True)
    # Printed, every run. These record where the assembled config departs from
    # the file on disk -- a gap filled, an override dropped, an `auto` pinned.
    # A comparison whose two sides were assembled differently is not a
    # comparison, and the only defence is that the assembly is stated out loud.
    for note in prov['notes']:
        print(f'[tierc]   {label} note: {note}', flush=True)
    for line in prov['migration']['applied']:
        print(f'[tierc]   {label} migrate: {line}', flush=True)
    trace = run_trace(cfg, out, workdir=work, deterministic=deterministic,
                      timeout=timeout)
    trace['meta']['provenance'] = prov
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(trace, f, indent=1, sort_keys=True)
    ok, msg = verify_step_count(trace, steps)
    print(f'[tierc] {label}: {msg}', flush=True)
    if not ok:
        raise SystemExit(f'[tierc] step-count verification FAILED for {label}')
    ok, msg = verify_capture(trace, steps)
    print(f'[tierc] {label}: {msg}', flush=True)
    if not ok:
        raise SystemExit(f'[tierc] capture verification FAILED for {label}')
    return trace, out


def _main(argv=None):
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '_child':
        return _child(argv[1], argv[2])

    ap = argparse.ArgumentParser(prog='tierc_smoke')
    ap.add_argument('config', nargs='?', help='config to run (the reference)')
    ap.add_argument('other', nargs='?', help='second config -- compare against it')
    ap.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--device', default='cuda')
    # DEFAULT ON, because the null test says it has to be. See DETERMINISM_NOTE.
    ap.add_argument('--deterministic', default='strict',
                    choices=['strict', 'warn', 'off'],
                    help='torch.use_deterministic_algorithms + cuBLAS workspace. '
                         '`strict` (default) raises and names an op with no '
                         'deterministic implementation; `warn` runs it anyway; '
                         '`off` reproduces the non-zero null.')
    ap.add_argument('--null', action='store_true',
                    help='run the SAME config twice and require identical traces')
    ap.add_argument('--negative-control', action='store_true',
                    help='perturb one loss coefficient and require the traces to differ')
    ap.add_argument('--coeff-scale', type=float, default=2.0,
                    help='negative control: multiplier on the perturbed coefficient')
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args(argv)

    if not a.config:
        ap.error('a config is required')
    outdir = a.outdir or _outdir()
    print(f'[tierc] output dir: {outdir}')
    common = dict(steps=a.steps, seed=a.seed, device=a.device,
                  deterministic=a.deterministic, outdir=outdir)

    if a.null:
        t1, _ = _run_one(a.config, 'null_a', **common)
        t2, _ = _run_one(a.config, 'null_b', **common)
        cmp = compare_traces(t1, t2)
        print('\n=== NULL CONTROL: same config, two launches ===')
        print(cmp.render())
        if cmp.identical:
            # ASCII only: this prints to a Windows console under cp1252, where a
            # section sign comes out as a replacement character.
            print('\nThe same-config spread is ZERO. Tier C is an exact test on '
                  'this target, as the plan says it should be.')
            return 0
        print('\nTHE INSTRUMENT IS BROKEN. A non-zero null means every downstream '
              'comparison is measuring the harness, not the config. Do not '
              'proceed.')
        return 1

    if a.negative_control:
        t1, _ = _run_one(a.config, 'neg_base', **common)
        # One loss coefficient, on the stage that actually trains. Multiplying a
        # coefficient is a REAL behavioural change of the smallest kind the
        # criterion is meant to catch -- config_snapshot validated on exactly
        # this case at tiers A/B, so tier C is held to the same bar.
        perturbed, which = _perturb_one_coeff(a.config, a.coeff_scale)
        t2, _ = _run_one(a.config, 'neg_perturbed', extra=perturbed,
                         **common)
        cmp = compare_traces(t1, t2)
        print(f'\n=== NEGATIVE CONTROL: {which} scaled by {a.coeff_scale} ===')
        print(cmp.render(limit=20))
        if not cmp.identical:
            print('\nThe harness SEES a one-coefficient change. It is capable of '
                  'failing, which is what makes a pass mean something.')
            return 0
        print('\nTHE HARNESS IS BLIND. It cannot see a real change, so it would '
              'launder an unvalidated config as validated.')
        return 1

    if not a.other:
        t, out = _run_one(a.config, 'single', **common)
        det, wall = split_trace(t)
        print(f'\ntrace: {out}')
        print(f'deterministic step records: {len(det["steps"])}, '
              f'logged points: {len(det["logged"])}')
        wall_keys = sorted({k for e in wall['logged'] for k in e['values']})
        print(f'wallclock keys recorded but NOT compared ({len(wall_keys)}): '
              f'{wall_keys}')
        return 0

    t1, _ = _run_one(a.config, 'ref', **common)
    t2, _ = _run_one(a.other, 'cand', **common)
    cmp = compare_traces(t1, t2)
    print(f'\nreference (old): {a.config}\ncandidate (new): {a.other}\n')
    print(cmp.render())
    return 0 if cmp.identical else 1


def _perturb_one_coeff(base_yaml: str, scale: float) -> tuple[dict, str]:
    """Find one live loss coefficient in the config and scale it.

    Reads the config that will actually run (after the same merge the comparison
    uses), so the perturbed key is guaranteed to be one the trainer reads, not
    one that was overwritten by a later layer."""
    from config_invariants import active_protocol_name, active_stages

    cfg, _ = build_config(base_yaml, steps=1)
    # Through the SHARED resolver, the same one `config_snapshot` uses, rather
    # than a second reading of `protocol:` here. The two configs being compared
    # do not even carry the same protocol SHAPE -- the pre-consolidation file
    # has an inline `protocol: {stages: [...]}` and the current one names a
    # protocol out of a library -- so a hand-rolled reader would work on one
    # side and silently return nothing on the other.
    name = active_protocol_name(cfg)
    stages = active_stages(cfg)
    if not stages:
        raise RuntimeError(f'no active stages resolved from {base_yaml}')
    for st in stages:
        for mode, block in (st.get('loss_coeffs') or {}).items():
            for key, val in (block or {}).items():
                if isinstance(val, (int, float)) and not isinstance(val, bool) and val:
                    dotted = f'protocols.{name}.stages[{st.get("name")}].loss_coeffs.{mode}.{key}'
                    extra = {'protocols': {name: {'stages': _stage_patch(
                        stages, st.get('name'), mode, key, val * scale)}}}
                    return extra, f'{dotted} ({val} -> {val * scale})'
    raise RuntimeError('no numeric loss coefficient found to perturb')


def _stage_patch(stages, stage_name, mode, key, new_value):
    """A full stage list with one coefficient replaced.

    A whole list, not a sparse patch: `_deep_merge` replaces lists wholesale, so
    a partial list would silently delete every stage it omitted -- and a config
    with one stage missing still runs, which is the worst version of this bug."""
    out = copy.deepcopy(stages)
    for st in out:
        if st.get('name') == stage_name:
            st['loss_coeffs'][mode][key] = new_value
    return out


if __name__ == '__main__':
    raise SystemExit(_main())
