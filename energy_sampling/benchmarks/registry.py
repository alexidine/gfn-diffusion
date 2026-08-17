"""
THE REGISTRY IS DATA AND IT IS LOADED. That is the whole point of this module.

`configs/mode_presets.yaml` carries the header "Reference only -- never loaded by
train.py" and nothing consumes it, so it drifts from the thing it describes and
nothing says so. A registry that is only read by humans is a document with a
misleading extension. This module loads `registry.yaml`, validates it, and is the
only supported way to ask what a benchmark is.

WHAT THIS IS NOT. It does not run anything. There is no runner, no scheduler, no
result store. A benchmark is a *specification*; executing one is `train.py` with
the resolved overrides applied, which is a workflow question and not this file's.
Every previous generation of measurement machinery in this project grew a harness
and then spent its budget on the harness.

THE THREE RULES THE VALIDATOR EXISTS TO ENFORCE, each of which this project has
already paid for:

  1. NO METRIC MAY DEPEND ON A REFERENCE RATE. A headline number that is a ratio
     against another run makes the denominator load-bearing, and a denominator
     chosen for one purpose gets reused for three others. `forbidden_primary_
     patterns` is checked against every `primary` entry.
  2. A COMPARISON NEEDS A MEASURED FLOOR. `noise_floor.measured` starts `null`
     and `floor_for` raises rather than defaulting. A tolerance picked by eye
     yields a test that passes because it is loose.
  3. CATASTROPHES ARE COUNTED. Every benchmark must name its catastrophe
     counters, and `score_repeats` refuses to summarise a set of repeats in which
     any run did not complete its work quantity.
"""
from __future__ import annotations

import copy
import os
import re
from typing import Any

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(_HERE, 'registry.yaml')

#: Hardware classes a benchmark may declare. `both` means the benchmark is
#: defined on either class and its numbers do NOT transfer between them.
HARDWARE_CLASSES = ('local', 'a100', 'both')
#: `partial` is a real answer and not a hedge: local runs of the MLIP routes give
#: valid step-cost deltas and invalid absolute throughput.
LOCAL_ADEQUACY = (True, False, 'partial')
#: The cluster averages occupancy over this window before deciding to cancel.
#: A benchmark shorter than it cannot report `gpu/util_policy` -- the window
#: never fills, and a partially-filled window reads as a number.
POLICY_WINDOW_S = 7200
#: Fused steps force-evaluate every non-dormant branch every `controller.
#: refresh_every` steps (canonically 10). A measurement window that is not a
#: whole number of those periods contains a different amount of refresh work in
#: each repeat, which shows up as noise floor that is really aliasing.
FUSED_REFRESH_PERIOD = 10
#: Fewer than this and a spread is not a spread.
MIN_FLOOR_REPEATS = 3

_ID_RE = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')


class RegistryError(ValueError):
    """The registry is malformed. Always raised with the offending id named."""


# --------------------------------------------------------------------- load --

def load(path: str = REGISTRY_PATH) -> dict:
    """Parse and validate. Every caller goes through here."""
    with open(path, 'r', encoding='utf-8') as f:
        reg = yaml.safe_load(f)
    validate(reg)
    return reg


def benchmark(bid: str, reg: dict | None = None) -> dict:
    reg = reg or load()
    for b in reg['benchmarks']:
        if b['id'] == bid:
            return b
    raise KeyError(f'no benchmark {bid!r}; have {[b["id"] for b in reg["benchmarks"]]}')


def suite(name: str, reg: dict | None = None) -> list[dict]:
    reg = reg or load()
    if name not in reg['suites']:
        raise KeyError(f'no suite {name!r}; have {sorted(reg["suites"])}')
    return [benchmark(i, reg) for i in reg['suites'][name]['benchmarks']]


def known_metrics(reg: dict) -> set[str]:
    """Every metric literal the registry declares, across all groups."""
    out = set()
    for group in reg['metrics'].values():
        out.update(group)
    return out


def resolved_overrides(bid: str, reg: dict | None = None) -> dict:
    """
    `defaults.overrides` deep-merged with the benchmark's own, benchmark wins.

    Deep, not shallow: the defaults disable `z_calibration.enabled` while several
    benchmarks set other keys under blocks of their own, and a shallow merge would
    drop one side of that silently -- which is the shape of every config bug this
    project has logged.
    """
    reg = reg or load()
    return _deep_merge(copy.deepcopy(reg['defaults']['overrides']),
                       copy.deepcopy(benchmark(bid, reg).get('overrides') or {}))


def _deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def epochs_for(bid: str, resume_step: int, reg: dict | None = None) -> int:
    """
    The value to put in `epochs`, WHICH IS AN ABSOLUTE STEP INDEX AND NOT A COUNT.

    `train.py` runs `trange(init_step, args.epochs + 1)` with `init_step` restored
    from the checkpoint, so a warm-started benchmark carrying `epochs: 400` against
    a step-6680 resume runs ZERO steps and reports a clean, empty result. Every
    benchmark that warm-starts must compute its budget through here.
    """
    w = benchmark(bid, reg)['work']
    return int(resume_step) + int(w['warmup_steps']) + int(w['measure_steps'])


# -------------------------------------------------------------------- floor --

def relative_span(values) -> float:
    """
    The dispersion the registry declares: (max - min) / |median|, over REPEATS.

    A span rather than an sd because the repeat counts here are 3-5, where an sd
    is an estimate of nothing. A span is what was actually observed, it is
    conservative, and it gets tighter -- never looser -- as repeats are added.

    Pure function of a list of numbers, so a floor can be recomputed from stored
    repeat statistics without re-running anything. `bench/metrics.py` is the
    precedent and the reason: four re-runs were spent on metric definitions that
    were entangled with execution.
    """
    xs = [float(v) for v in values]
    if len(xs) < 2:
        raise ValueError('a spread needs at least two repeats')
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    med = (xs_sorted[n // 2] if n % 2 else 0.5 * (xs_sorted[n // 2 - 1] + xs_sorted[n // 2]))
    if med == 0:
        raise ValueError('median is zero; relative span is undefined')
    return (xs_sorted[-1] - xs_sorted[0]) / abs(med)


def floor_for(bid: str, metric: str, reg: dict | None = None) -> float:
    """
    The measured floor, or a refusal.

    NEVER returns a default. An unmeasured floor is not a small floor; it is the
    absence of the thing a comparison rests on, and returning 0.0 here would turn
    every difference into a finding.
    """
    b = benchmark(bid, reg)
    m = b['noise_floor'].get('measured')
    if not m:
        raise RegistryError(
            f'{bid}: noise floor for {metric!r} has NOT been measured. Run '
            f'{b["noise_floor"]["repeats"]} repeat launches and record '
            f'noise_floor.measured before making any comparison on this benchmark.')
    if metric not in m.get('per_metric', {}):
        raise RegistryError(
            f'{bid}: floor recorded, but not for {metric!r} '
            f'(have {sorted(m.get("per_metric", {}))})')
    return float(m['per_metric'][metric])


def exceeds_floor(a: float, b: float, floor: float) -> bool:
    """
    Is the difference between two runs bigger than the same-config spread?

    The comparison is symmetric and scale-free: |a - b| against `floor` times the
    midpoint. There is no reference run -- neither argument is a denominator.
    """
    mid = 0.5 * (abs(float(a)) + abs(float(b)))
    if mid == 0:
        return False
    return abs(float(a) - float(b)) > float(floor) * mid


def score_repeats(bid: str, repeats: list[dict], reg: dict | None = None) -> dict:
    """
    Summarise a set of repeat launches into the run statistic and its spread.

    `repeats` is a list of {'metrics': {name: value}, 'completed': bool,
    'catastrophes': {name: count}}.

    AN INCOMPLETE RUN IS NOT A SLOW RUN. A benchmark that OOMed, was cut by the
    runaway guard, or ended early did not execute the work quantity, so its timing
    describes different work. It is reported and excluded, never averaged in --
    the same reason `bench/metrics.py` scores an aborted run as infinite rather
    than on its healthy-looking tail.
    """
    b = benchmark(bid, reg)
    good = [r for r in repeats if r.get('completed')]
    dropped = [r for r in repeats if not r.get('completed')]
    counts: dict[str, int] = {}
    for r in repeats:
        for k, v in (r.get('catastrophes') or {}).items():
            counts[k] = counts.get(k, 0) + int(v)
    out = {'benchmark': bid, 'repeats': len(repeats), 'usable': len(good),
           'dropped_incomplete': len(dropped), 'catastrophes': counts,
           'per_metric': {}}
    for metric in b['metrics']['primary']:
        vals = [r['metrics'][metric] for r in good if metric in (r.get('metrics') or {})]
        if len(vals) < 2:
            out['per_metric'][metric] = {'median': None, 'relative_span': None,
                                         'n': len(vals)}
            continue
        s = sorted(float(v) for v in vals)
        n = len(s)
        med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
        out['per_metric'][metric] = {'median': med,
                                     'relative_span': relative_span(s),
                                     'n': n}
    return out


# ---------------------------------------------------------------- validate --

def validate(reg: dict) -> None:
    """Raise `RegistryError` on the first violation, naming the benchmark."""
    if reg.get('schema_version') != 1:
        raise RegistryError(f'schema_version must be 1, got {reg.get("schema_version")!r}')
    for block in ('defaults', 'metrics', 'hardware', 'benchmarks', 'suites',
                  'forbidden_primary_patterns'):
        if block not in reg:
            raise RegistryError(f'missing top-level block {block!r}')

    _validate_defaults(reg)

    catalogue = known_metrics(reg)
    forbidden = [re.compile(p) for p in reg['forbidden_primary_patterns']]
    seen: set[str] = set()

    for b in reg['benchmarks']:
        bid = b.get('id')
        if not bid or not _ID_RE.match(bid):
            raise RegistryError(f'bad benchmark id {bid!r}: want kebab-case')
        if bid in seen:
            raise RegistryError(f'duplicate benchmark id {bid!r}')
        seen.add(bid)
        for block in ('workload', 'training_mode', 'hardware', 'work', 'metrics',
                      'liveness', 'noise_floor', 'correctness', 'comparison'):
            if block not in b:
                raise RegistryError(f'{bid}: missing block {block!r}')
        _validate_metrics(bid, b, catalogue, forbidden)
        _validate_hardware(bid, b, reg)
        _validate_work(bid, b)
        _validate_floor(bid, b)
        _validate_correctness(bid, b)
        if not b['liveness']:
            raise RegistryError(
                f'{bid}: liveness is empty. An inert branch posts a plausible row '
                f'rather than an error, so every benchmark must name how it proves '
                f'the thing it times actually ran.')

    _validate_suites(reg, seen)


def _validate_defaults(reg: dict) -> None:
    ov = reg['defaults'].get('overrides') or {}
    # PERIODIC extra work whose period is not the report period, so a fixed-length
    # window contains a variable amount of it. z_calibration is worse than
    # aliasing: its rollouts are inside the step timing while only the training
    # batch is charged to the throughput denominator (train.py:2168-2170), so
    # `samples_per_sec` moves at constant batch as the sensor converges.
    #
    # ray_calibration USED TO BE REQUIRED HERE TOO and no longer can be: its
    # `enabled` flag is retired (utils._RETIRED_KEYS) and the probe now arms from
    # the stage declarations, which an override cannot reach. Requiring the flag
    # meant this validator raised unless the registry set a key the trainer
    # refuses at load -- a schema the two halves could not both satisfy. See the
    # note at defaults.overrides in registry.yaml for what that costs a window.
    for path, want in ((('z_calibration', 'enabled'), False),):
        node: Any = ov
        for k in path:
            if not isinstance(node, dict) or k not in node:
                raise RegistryError(f'defaults.overrides must set {".".join(path)}')
            node = node[k]
        if node is not want:
            raise RegistryError(f'defaults.overrides.{".".join(path)} must be {want}')
    for key, want in (('checkpoint_read_only', True),
                      ('grow_batch_size', False),
                      ('auto_batch_throughput_opt', False),
                      ('max_step_seconds', 0)):
        if ov.get(key) != want:
            raise RegistryError(f'defaults.overrides.{key} must be {want!r}')


def _validate_metrics(bid, b, catalogue, forbidden) -> None:
    m = b['metrics']
    for field in ('primary', 'secondary', 'catastrophes'):
        if field not in m:
            raise RegistryError(f'{bid}: metrics.{field} missing')
    if not m['primary']:
        raise RegistryError(f'{bid}: metrics.primary is empty')
    if not m['catastrophes']:
        raise RegistryError(
            f'{bid}: metrics.catastrophes is empty. Catastrophes are counted, never '
            f'averaged, and a benchmark that names none cannot report the tail.')
    unusable = m.get('unusable') or {}
    for name in list(m['primary']) + list(m['secondary']) + list(m['catastrophes']):
        if name not in catalogue:
            raise RegistryError(f'{bid}: metric {name!r} is not in the metrics catalogue')
    for name in m['primary']:
        for pat in forbidden:
            if pat.search(name):
                raise RegistryError(
                    f'{bid}: primary metric {name!r} matches forbidden pattern '
                    f'{pat.pattern!r} -- no headline metric may depend on a reference rate')
    for name, reason in unusable.items():
        if not reason or not str(reason).strip():
            raise RegistryError(f'{bid}: unusable metric {name!r} carries no reason')
        if name in m['primary'] or name in m['secondary']:
            raise RegistryError(
                f'{bid}: {name!r} is listed both as reportable and as unusable')


def _validate_hardware(bid, b, reg) -> None:
    h = b['hardware']
    if h.get('class') not in HARDWARE_CLASSES:
        raise RegistryError(f'{bid}: hardware.class must be one of {HARDWARE_CLASSES}')
    if h.get('local_adequate') not in LOCAL_ADEQUACY:
        raise RegistryError(f'{bid}: hardware.local_adequate must be one of {LOCAL_ADEQUACY}')
    if h['class'] == 'a100' and h['local_adequate'] is not False:
        raise RegistryError(
            f'{bid}: hardware.class is a100 but local_adequate is not False')
    if h['class'] == 'local' and h.get('a100_required'):
        raise RegistryError(f'{bid}: class local cannot also require the A100')
    if not str(h.get('reason', '')).strip():
        raise RegistryError(
            f'{bid}: hardware.reason is empty -- "requires the A100" without a reason '
            f'is how a convenience becomes a law')


def _validate_work(bid, b) -> None:
    w = b['work']
    for field in ('kind', 'pin_batch', 'warmup_steps', 'measure_steps',
                  'epochs_formula', 'wallclock_cap_s'):
        if field not in w:
            raise RegistryError(f'{bid}: work.{field} missing')
    if w['warmup_steps'] < 1:
        raise RegistryError(
            f'{bid}: work.warmup_steps must be >= 1. The first steps carry '
            f'allocator warmup and, on a compiling host, a recompile stall.')
    if w['measure_steps'] < 1:
        raise RegistryError(f'{bid}: work.measure_steps must be >= 1')
    if 'resume_step' not in w:
        raise RegistryError(f'{bid}: work.resume_step missing (null is a valid value)')
    if 'resume_step' not in str(w['epochs_formula']):
        raise RegistryError(
            f'{bid}: work.epochs_formula must be expressed against resume_step -- '
            f'`epochs` is an ABSOLUTE step index, so a warm-started benchmark with a '
            f'count in it runs zero steps and reports nothing')
    ov = b.get('overrides') or {}
    if w['kind'] == 'fixed_steps_per_rung':
        # The batch varies BY DESIGN here, one launch per rung, so a single pinned
        # value in `overrides` would be a lie that the rung loop then silently
        # overrides. `batch_size` must be null and the ladder must be explicit.
        if not w.get('batch_rungs'):
            raise RegistryError(f'{bid}: kind is fixed_steps_per_rung but batch_rungs is empty')
        if w.get('batch_size') is not None:
            raise RegistryError(
                f'{bid}: a rung benchmark must leave work.batch_size null -- the batch '
                f'is per-rung, and a value here reads as a pin that is not one')
        for key in ('batch_size', 'max_batch_size'):
            if key in ov:
                raise RegistryError(
                    f'{bid}: overrides must not fix {key!r} on a rung benchmark; each '
                    f'rung sets batch_size and max_batch_size together to its own value')
    elif w['pin_batch'] and w.get('batch_size') is not None:
        if ov.get('max_batch_size') != w['batch_size'] or ov.get('batch_size') != w['batch_size']:
            raise RegistryError(
                f'{bid}: pin_batch is true, so overrides must set batch_size and '
                f'max_batch_size both to {w["batch_size"]} -- they are independent '
                f'hard stops and setting one alone does not pin anything')
    if b['training_mode']['train_mode'] == 'fused':
        if w['measure_steps'] % FUSED_REFRESH_PERIOD:
            raise RegistryError(
                f'{bid}: fused measure_steps must be a multiple of '
                f'{FUSED_REFRESH_PERIOD} (the force-refresh period), or repeats '
                f'differ in how much refresh work they contain')
    if 'gpu/util_policy' in b['metrics']['primary']:
        if int(w.get('min_wallclock_s') or 0) < POLICY_WINDOW_S:
            raise RegistryError(
                f'{bid}: gpu/util_policy is primary but work.min_wallclock_s is under '
                f'{POLICY_WINDOW_S} s -- the window would never fill and a partial '
                f'window still prints a number')


def _validate_floor(bid, b) -> None:
    nf = b['noise_floor']
    for field in ('method', 'repeats', 'seed_policy', 'run_statistic', 'dispersion'):
        if field not in nf:
            raise RegistryError(f'{bid}: noise_floor.{field} missing')
    if nf['method'] != 'repeat_launch':
        raise RegistryError(
            f'{bid}: noise_floor.method must be repeat_launch. Re-timing inside one '
            f'process measures step-to-step scatter, which is not the quantity a '
            f'run-to-run comparison is tested against.')
    if int(nf['repeats']) < MIN_FLOOR_REPEATS:
        raise RegistryError(f'{bid}: noise_floor.repeats must be >= {MIN_FLOOR_REPEATS}')
    if 'measured' not in nf:
        raise RegistryError(f'{bid}: noise_floor.measured missing (null until measured)')
    m = nf['measured']
    if m is None:
        return
    for field in ('date', 'host', 'repeats', 'per_metric'):
        if field not in m:
            raise RegistryError(f'{bid}: noise_floor.measured.{field} missing')
    if int(m['repeats']) < int(nf['repeats']):
        raise RegistryError(
            f'{bid}: floor recorded from {m["repeats"]} repeats, fewer than the '
            f'{nf["repeats"]} the benchmark declares')
    missing = [k for k in b['metrics']['primary'] if k not in m['per_metric']]
    if missing:
        raise RegistryError(
            f'{bid}: floor does not cover primary metrics {missing} -- a primary '
            f'metric without a floor cannot support a comparison')


def _validate_correctness(bid, b) -> None:
    c = b['correctness']
    if not c.get('reference'):
        raise RegistryError(f'{bid}: correctness.reference missing')
    if c.get('exactness') not in ('exact', 'floor'):
        raise RegistryError(f'{bid}: correctness.exactness must be exact or floor')
    if c['exactness'] == 'exact' and c['reference'] != 'closed_form':
        raise RegistryError(
            f'{bid}: exactness `exact` is only defensible against a closed form. On '
            f'any GPU/MLIP path the same run disagrees with itself, so an exact bar '
            f'measures reduction order rather than the code.')
    if c['reference'] == 'control_comparison' and not c.get('gate'):
        raise RegistryError(
            f'{bid}: a control comparison must name the harness that performs it')


def _validate_suites(reg, ids) -> None:
    for name, s in reg['suites'].items():
        if not _ID_RE.match(name):
            raise RegistryError(f'bad suite name {name!r}: want kebab-case')
        if not str(s.get('description', '')).strip():
            raise RegistryError(f'suite {name!r}: description is empty')
        if not s.get('benchmarks'):
            raise RegistryError(f'suite {name!r} is empty')
        unknown = [i for i in s['benchmarks'] if i not in ids]
        if unknown:
            raise RegistryError(f'suite {name!r} names unknown benchmarks {unknown}')
        if len(set(s['benchmarks'])) != len(s['benchmarks']):
            raise RegistryError(f'suite {name!r} lists a benchmark twice')
    local = reg['suites'].get('local-dev', {}).get('benchmarks', [])
    by_id = {b['id']: b for b in reg['benchmarks']}
    for i in local:
        if by_id[i]['hardware']['local_adequate'] is False:
            raise RegistryError(
                f'suite local-dev contains {i!r}, which declares local_adequate false')
    covered = {i for s in reg['suites'].values() for i in s['benchmarks']}
    orphans = sorted(ids - covered)
    if orphans:
        raise RegistryError(
            f'benchmarks in no suite: {orphans}. A benchmark nothing names is a '
            f'benchmark nobody reruns.')


if __name__ == '__main__':
    _reg = load()
    print(f'registry ok: {len(_reg["benchmarks"])} benchmarks, '
          f'{len(_reg["suites"])} suites')
    for _s, _v in _reg['suites'].items():
        print(f'  {_s:<20} {len(_v["benchmarks"])} benchmarks')
    _un = [b['id'] for b in _reg['benchmarks'] if not b['noise_floor']['measured']]
    if _un:
        print(f'\nNO MEASURED NOISE FLOOR ({len(_un)}/{len(_reg["benchmarks"])}) -- '
              f'no comparison is supportable on these yet:')
        for _i in _un:
            print(f'  {_i}')
