"""
Production config generation: canonical + problem + overrides -> config.

    from configs import generate
    arms = [generate.arm('f1_elj', problem='mipcas_elj', batch_size=1000)]
    generate.emit(arms, outdir='configs/my_battery')

WHY THIS EXISTS. Twenty-odd generators under `configs/*/make.py` were each forked
from the last, so every convention -- deep-merge, cluster paths, index emission,
the validation block -- exists in as many copies as there are batteries, and
`load_yaml` alone is defined 82 times in the tree. The conventions are real and
worth keeping; having them live only in a fork lineage is what makes each new
battery an archaeology exercise. This is that lineage's spine, extracted once.

=============================================================================
FROM CANONICAL, ALWAYS -- AND THE STAMP IS WHAT MAKES THAT SAFE
=============================================================================
Every arm derives from `configs/mk_dev.yaml` and records the hash of the exact
canonical file it came from. The alternative -- a battery keeping a frozen
snapshot of the canonical config, which is what every battery has done until now
-- fails in both directions, and both failures were observed on the same day:

  * a100_stab_aug16's arms were generated from a base that predated a schema
    migration, so they wrote three keys into homes nothing reads and stamped a
    state version claiming otherwise. STALE AGAINST THE SCHEMA.
  * the same battery's base was re-snapshotted mid-flight, so the arms on disk
    and the base beside them disagreed for six hours. STALE AGAINST ITSELF.

Regenerating from canonical makes the first unrepresentable. The stamp makes the
second VISIBLE rather than preventing it: an arm carries the canonical hash and
the project state it was built against, so "was this arm built from the config I
am looking at" is a string comparison instead of a guess.

The cost is real and is accepted deliberately: a battery cannot rely on a frozen
copy to hold still while the canonical config moves. It re-runs the generator and
diffs. That trade is only sound because the canonical config is not churned in
ways that change what a running battery measures -- a discipline, not a mechanism,
and named here so it is a discipline someone chose rather than one they inherited.

=============================================================================
OVERRIDES ARE NOT WHITELISTED
=============================================================================
The plan for this module originally said it should accept "only genuinely
run-specific overrides". That is refused, and the reason is the same one
`config_invariants` gives for grading most of its rules BASELINE rather than
ERROR: a battery exists to vary something, and a list of permitted axes written
in advance is a list of experiments nobody can run. `exit_bar_is_within_measured
_range`'s docstring puts it exactly -- "a rule built on evidence blocks the next
experiment".

So any dotted key may be overridden, including `protocol` and per-stage
`loss_coeffs`. What replaces the whitelist is not permissiveness:

  * `config_invariants.errors()` is FATAL at generation. An ERROR is a
    self-contradiction provable from the file, and generation is exactly where
    that should cost nothing to fix.
  * BASELINE violations are REPORTED, never fatal -- a departure from a measured
    default is what an experiment often is.
  * every deviation from canonical is summarised, so an override that was not
    intended is visible in the generator's own output rather than discovered in
    the run.

A gate that refuses is replaced by a gate that SHOWS.
"""

import copy
import datetime as _datetime
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config_invariants                       # noqa: E402
import config_snapshot                         # noqa: E402
import config_state                            # noqa: E402

CANONICAL = _HERE / 'mk_dev.yaml'
PROBLEMS = _HERE / 'problems.yaml'

#: Where the provenance block lives. A single top-level key, so it is one line to
#: strip and cannot collide with a config section. Nothing reads it at runtime --
#: it is inert by construction, and `test_generate.py` proves that rather than
#: asserting it, because "inert-looking flag" is this codebase's signature defect.
PROVENANCE_KEY = 'provenance'


# --------------------------------------------------------------------- loading

def load_yaml(path) -> dict:
    """The project's loader, not a local copy.

    A local `open(path)` is what every generator in the corpus wrote, and it is
    wrong twice over: the default encoding is the LOCALE's (cp1252 here), so any
    non-ASCII byte in a path raises on one machine and loads on another; and a
    UTF-8 BOM makes yaml parse the whole header as one scalar and then fail at
    the first real key. `configs/shakeout_aug16/qm9_cond.yaml` carries a BOM
    today -- found by pointing this module at the corpus, which is what the
    corpus is for. Reusing `utils.load_yaml` means the generator and the trainer
    cannot disagree about what a config file says."""
    import utils
    return utils.load_yaml(path)


def merge(base: dict, over: dict) -> dict:
    """Deep-merge `over` onto a COPY of `base`; scalars and lists replace.

    A list replaces rather than extends, which is the behaviour every generator
    in the corpus implemented and the only one that makes sense here: `alphas`,
    `space_groups` and `bounds` are all lists whose meaning is the whole
    sequence, so appending to one produces a value nobody wrote."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_INDEXED = __import__('re').compile(r'^(.*?)\[(\d+)\]$')


def _descend(node, part: str, dotted: str, sofar: str, create: bool):
    """One step of a dotted path, handling an optional `[n]` list index.

    LIST INDICES EXIST BECAUSE STAGES ARE A LIST. `protocol.stages[0].mle_gate`
    is how a battery reaches one stage, and without it the only way to change a
    single stage is to restate the whole stage list -- which is what every
    generator in the corpus did, and how a stage list drifts from canonical
    without anyone deciding to."""
    m = _INDEXED.match(part)
    key, idx = (m.group(1), int(m.group(2))) if m else (part, None)
    if key:
        nxt = node.get(key) if isinstance(node, dict) else None
        if nxt is None and create and idx is None:
            nxt = node[key] = {}
        if nxt is None:
            raise ValueError(f'override {dotted!r}: {sofar} does not exist')
        node = nxt
    if idx is not None:
        if not isinstance(node, list):
            raise ValueError(
                f'override {dotted!r}: {sofar} is {type(node).__name__}, not a list')
        if idx >= len(node):
            raise ValueError(
                f'override {dotted!r}: {sofar} has {len(node)} entries, no index {idx}')
        node = node[idx]
    return node


def set_dotted(cfg: dict, dotted: str, value: Any) -> dict:
    """Set `a.b.c` (or `a.b[0].c`) on a copy of cfg, creating missing dicts.

    REFUSES TO TUNNEL THROUGH A NON-DICT. `lr_sensor.beta` on a stage whose
    lr_sensor is None would otherwise silently replace the None with a dict
    carrying only `beta` -- a sensor with no kind, which is not what anyone
    writing that override meant. An out-of-range list index is refused for the
    same reason: silently appending a stage is not what `stages[7]` asks for."""
    out = copy.deepcopy(cfg)
    parts = dotted.split('.')
    node = out
    for i, p in enumerate(parts[:-1]):
        sofar = '.'.join(parts[:i + 1])
        nxt = _descend(node, p, dotted, sofar, create=True)
        if not isinstance(nxt, (dict, list)):
            raise ValueError(
                f'override {dotted!r}: {sofar} is {type(nxt).__name__}, not a '
                f'section -- cannot set a key inside it')
        node = nxt
    last = parts[-1]
    m = _INDEXED.match(last)
    if m:
        node = _descend(node, last, dotted, dotted, create=False)
        raise ValueError(
            f'override {dotted!r}: ends in a list index; name the key to set')
    node[last] = value
    return out


def canonical() -> dict:
    """The canonical config, with its state version checked against the code.

    A canonical config at a different state than `config_state` means one of the
    two moved without the other, and every arm generated in between would carry a
    version stamp that is a lie. Fatal here, where it costs nothing."""
    cfg = load_yaml(CANONICAL)
    got, want = cfg.get('project_state_version'), config_state.CURRENT_STATE_VERSION
    if got != want:
        raise SystemExit(
            f'{CANONICAL} is state {got} against code state {want}. Generation is '
            f'blocked: every arm would stamp a version its contents do not match.\n'
            f'    python -m config_state migrate {CANONICAL}')
    return cfg


def canonical_hash() -> str:
    return hashlib.sha256(CANONICAL.read_bytes()).hexdigest()[:12]


#: Problem-registry keys that are DOCUMENTATION, not config. `test_problems.py`
#: allows them; a config must not carry them.
_PROBLEM_METADATA = ('description', 'domain', 'conditioning')

#: Problem-registry keys whose config home is not their own name. The registry is
#: written flat because a problem is described flat -- "this problem runs at
#: T=2.5" -- while the config groups by consumer. Verified against
#: configs/mk_dev.yaml: every other allowed key sits at top level under its own
#: name, and `model`/`buffers`/`protocol` merge as the sections they already are.
_PROBLEM_KEY_PATHS = {
    'temperature': 'energy_config.temperature',
    'analyze_kwargs': 'energy_config.analyze_kwargs',
}


def problems() -> dict:
    """The problem registry. `problems.yaml` is {schema, problems}, and this
    returns the inner mapping -- the schema version guards the outer shape."""
    doc = load_yaml(PROBLEMS) or {}
    return doc.get('problems') or {}


def problem_block(name: str) -> dict:
    """`name`'s problem settings, translated to the paths a config uses.

    THIS TRANSLATION IS THE REASON problems.yaml WAS NEVER LOADED. 1.4 built the
    registry and declared it loaded rather than reference-only, but nothing
    consumed it -- so the map from a problem's flat description to the config's
    grouped-by-consumer layout had never been written down, and a registry no
    code reads is the exact failure mode `mode_presets.yaml` was retired for.
    Two keys need it; the rest are already at the path they name."""
    known = problems()
    if name not in known:
        raise SystemExit(
            f'unknown problem {name!r}. configs/problems.yaml defines: '
            f'{", ".join(sorted(known))}')
    block, out = dict(known[name] or {}), {}
    for key in _PROBLEM_METADATA:
        block.pop(key, None)
    for key, value in block.items():
        path = _PROBLEM_KEY_PATHS.get(key)
        if path is None:
            out[key] = copy.deepcopy(value)
        else:
            out = set_dotted(out, path, value)
    return out


# ------------------------------------------------------------------ generation

def arm(name: str, problem: Optional[str] = None, tag: Optional[str] = None,
        base: Optional[dict] = None, **overrides) -> dict:
    """One config: canonical, then the problem block, then run-specific overrides.

    ORDER IS THE CONTRACT. Canonical carries the defaults, the problem block says
    what the problem IS, and overrides are what this run varies -- so an override
    beats the problem block, which beats canonical. Any of the three may set any
    key; that is what makes "the problem chose it" and "this arm chose it"
    distinguishable in the deviation summary rather than a merge accident.

    Dotted keys are accepted alongside nested dicts, since `loss_coeffs__fwd__tb`
    is unreadable and `{'fwd_loss_coeffs': {'tb': 1.0}}` is verbose for one value:

        arm('a1', problem='latent_gaussian', **{'integrator.T': 25})
    """
    cfg = copy.deepcopy(base) if base is not None else canonical()
    if problem is not None:
        cfg = merge(cfg, problem_block(problem))
    cfg['run_name'] = name
    if tag is not None:
        cfg['tag'] = tag
    nested = {k: v for k, v in overrides.items() if '.' not in k}
    cfg = merge(cfg, nested)
    for dotted, value in overrides.items():
        if '.' in dotted:
            cfg = set_dotted(cfg, dotted, value)
    return stamp(cfg, problem=problem)


def stamp(cfg: dict, problem: Optional[str] = None) -> dict:
    """Record what this config was made from.

    NOT A TIMESTAMP-ONLY RECORD. The load-bearing field is `canonical_sha`: it
    answers "was this arm built from the canonical config I am looking at now",
    which is the question that went unanswerable when a battery's base moved
    under its arms. `generated_utc` is provenance for humans; the hash is
    provenance for a comparison."""
    out = copy.deepcopy(cfg)
    out[PROVENANCE_KEY] = {
        'canonical': str(CANONICAL.name),
        'canonical_sha': canonical_hash(),
        'project_state_version': config_state.CURRENT_STATE_VERSION,
        'problem': problem,
        'generated_utc': _datetime.datetime.now(_datetime.timezone.utc)
                                  .strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    return out


# ------------------------------------------------------------------ validation

def validate(cfg: dict, name: str = '<arm>') -> list:
    """ERROR is fatal, BASELINE is reported. Returns the BASELINE violations.

    The split is `config_invariants`' own and is not re-litigated here: an ERROR
    is a self-contradiction provable from the file, a BASELINE is a departure
    from something measured. Generation is the right place to make the first
    fatal -- it costs a rerun rather than a queue slot -- and the wrong place to
    make the second fatal, because a config written to depart from a measured
    default is the normal shape of an experiment."""
    violations = config_invariants.check(cfg)
    errors = [v for v in violations if v.severity == config_invariants.ERROR]
    if errors:
        raise SystemExit(
            f'{name}: {len(errors)} config ERROR(s) -- generation refused:\n' +
            '\n'.join(f'  {v}' for v in errors))
    return [v for v in violations if v.severity != config_invariants.ERROR]


def deviations(path, reference=None) -> 'config_snapshot.Comparison':
    """How a written arm differs from canonical, on the RESOLVED config.

    Resolved, not textual, so `auto` appears as the number it will train at and a
    reordering is not a deviation. Provenance is excluded: it differs by
    construction on every arm, and a summary whose first line is always noise is
    a summary people stop reading."""
    cand = config_snapshot.snapshot(str(path))
    ref = config_snapshot.snapshot(str(reference or CANONICAL))
    for snap in (ref, cand):
        for key in list(snap.get('config', {})):
            if key == PROVENANCE_KEY or key.startswith(PROVENANCE_KEY + '.'):
                snap['config'].pop(key)
    return config_snapshot.compare(ref, cand)


# --------------------------------------------------------------------- emission

def emit(arms: dict, outdir, index: bool = True, quiet: bool = False) -> list:
    """Write `{name: cfg}` to `outdir`, validate each, and report deviations.

    Writes first and validates the WRITTEN FILE, rather than validating the dict
    in memory and writing afterwards. The two can differ -- yaml round-trips
    floats and `1.0e-6` style scalars -- and the file is what trains, so the file
    is what gets checked."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, cfg in arms.items():
        path = outdir / f'{name}.yaml'
        with open(path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        baselines = validate(load_yaml(path), name)
        cmp = deviations(path)
        # A CONFIG THAT DOES NOT LOAD MUST NOT REPORT AS CLEAN, and it did: an
        # unloadable candidate leaves changed/added/removed all empty, so the
        # deviation count read "0 deviations from canonical" for an arm that
        # could never train. `snapshot` returns load_error instead of raising on
        # purpose (its docstring says why -- the reference side of a comparison
        # is often expected to fail), which puts the burden on the caller. Here
        # the candidate is a file we just wrote, so there is no such excuse.
        if cmp.candidate_error:
            raise SystemExit(
                f'{name}: generated config DOES NOT LOAD -- {cmp.candidate_error}\n'
                f'  written to {path} for inspection; fix the overrides and rerun.')
        written.append(path)
        if quiet:
            continue
        n = len(cmp.changed) + len(cmp.added) + len(cmp.removed)
        print(f'{name}: {n} deviation(s) from canonical'
              f'{f", {len(baselines)} baseline note(s)" if baselines else ""}')
        for v in baselines:
            print(f'    {v}')
    if index:
        _write_index(arms, outdir)
    return written


def _write_index(arms: dict, outdir: Path) -> Path:
    """INDEX.tsv -- the convention every battery in the corpus emits by hand."""
    path = Path(outdir) / 'INDEX.tsv'
    cols = ('name', 'problem', 'batch', 'epochs', 'state', 'canonical_sha')
    with open(path, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for name, cfg in arms.items():
            prov = cfg.get(PROVENANCE_KEY, {})
            f.write('\t'.join(str(x) for x in (
                name, prov.get('problem') or '-', cfg.get('batch_size', '-'),
                cfg.get('epochs', '-'), prov.get('project_state_version', '-'),
                prov.get('canonical_sha', '-'))) + '\n')
    return path
