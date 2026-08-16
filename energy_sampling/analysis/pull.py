"""
Run resolution and history fetching, for both local `.wandb` datastores and the
cloud API.

Cluster runs sync, so the cloud path reaches them; the local path exists because
a run in progress on this box is readable before it syncs, and because it works
with no network.

Everything here is shaped by traps that have already produced a wrong or empty
read. They are marked H1-H4 and each is load-bearing.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

# Cache lives in the system temp dir, deliberately not in the repo: run data in a
# working tree is a commit accident waiting to happen. A cold pull is cheap
# anyway (~2 s for a 14k-step run), so this is for repeat analysis, not latency.
CACHE_DIR = os.path.join(tempfile.gettempdir(), 'gfn_analysis_cache')

DEFAULT_PROJECT = 'mkilgour/GFN Energy'
CONFORMER_PROJECT = 'mkilgour/GFN Conformers'

# H3: a stub run directory is small AND old. A freshly launched run is also
# small, so a size-only filter races launches and hides the run you just started.
_GHOST_BYTES = 65536
_GHOST_AGE_S = 600


class EmptyPull(RuntimeError):
    """A pull that returned no rows.

    Its own exception type because this is the failure this package exists to
    prevent: `scan_history` answers an unresolved key with zero rows and no
    error, which reads exactly like a run that did no work."""


@dataclass
class Run:
    """One run's data. `history` maps key -> (steps, values) as float arrays."""

    run_id: str
    name: str
    source: str                       # 'local' or 'cloud'
    config: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    path: Optional[str] = None

    @property
    def last_step(self) -> float:
        steps = [s[-1] for s, _ in self.history.values() if len(s)]
        return max(steps) if steps else 0.0

    def available_keys(self) -> set[str]:
        return set(self.summary) | set(self.history)


# ---------------------------------------------------------------------------
# Local runs
# ---------------------------------------------------------------------------

def _run_dirs(base='wandb') -> list[str]:
    """Local run directories, oldest first.

    H3: ordered by the launch timestamp in the directory NAME
    (`run-YYYYMMDD_HHMMSS-id` sorts chronologically). Directory mtime and
    `.wandb` file mtime are BOTH unusable -- the sync service sweeps old runs,
    touching and even growing their files, so a recently-synced run from last
    week outranks one launched an hour ago."""
    out = []
    for p in glob.glob(os.path.join(base, 'run-*', 'run-*.wandb')):
        try:
            sz = os.path.getsize(p)
            age = time.time() - os.path.getmtime(p)
        except OSError:
            continue
        if sz > _GHOST_BYTES or age < _GHOST_AGE_S:
            out.append(os.path.dirname(p))
    return sorted(out, key=os.path.basename)


def _local_config(run_dir: str) -> dict:
    p = os.path.join(run_dir, 'files', 'config.yaml')
    if not os.path.exists(p):
        return {}
    import yaml
    try:
        with open(p, encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _local_summary(run_dir: str) -> dict:
    p = os.path.join(run_dir, 'files', 'wandb-summary.json')
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception:
        return {}


def scan_local_history(run_dir: str, keys: Optional[set[str]] = None) -> dict:
    """Parse a local `.wandb` datastore into {key: (steps, values)}.

    H4, all four of them, each observed:
      * `item.key` is EMPTY in this wandb version for nested metrics -- the name
        lives in `item.nested_key` and must be joined.
      * not every history row carries `_step`; the last seen value is carried
        forward, or rows silently vanish.
      * `scan_data()`/`ParseFromString` are wrapped and BREAK on error: a live
        run's final record is partially written, and raising there would discard
        every row already parsed.
      * a just-restarted run's `.wandb` can be 0 bytes (header race).

    `keys=None` collects everything, which is how the available-key set is
    discovered before anything is requested."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore

    files = glob.glob(os.path.join(run_dir, 'run-*.wandb'))
    if not files or os.path.getsize(files[0]) == 0:
        return {}

    ds = datastore.DataStore()
    ds.open_for_scan(files[0])
    series: dict[str, tuple[list, list]] = {}
    step = None
    try:
        while True:
            data = ds.scan_data()
            if data is None:
                break
            rec = wandb_internal_pb2.Record()
            try:
                rec.ParseFromString(data)
            except Exception:
                break
            if rec.WhichOneof('record_type') != 'history':
                continue
            row = {}
            for item in rec.history.item:
                k = item.key or '.'.join(item.nested_key)
                if keys is not None and k not in keys and k != '_step':
                    continue
                try:
                    row[k] = json.loads(item.value_json)
                except Exception:
                    pass
            if '_step' in row:
                step = row['_step']
            if step is None:
                continue
            for k, v in row.items():
                if k == '_step' or isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)) and np.isfinite(v):
                    s, vals = series.setdefault(k, ([], []))
                    s.append(step)
                    vals.append(v)
    except Exception:
        pass

    return {k: (np.asarray(s, float), np.asarray(v, float))
            for k, (s, v) in series.items() if len(s) >= 3}


# ---------------------------------------------------------------------------
# Cloud runs
# ---------------------------------------------------------------------------

def _cloud_run(spec: str, project: str):
    import wandb
    api = wandb.Api()
    if re.fullmatch(r'[a-z0-9]{8}', spec):
        try:
            return api.run(f'{project}/{spec}')
        except Exception:
            pass
    runs = api.runs(project, filters={'$or': [
        {'display_name': spec}, {'config.run_name': spec}, {'tags': spec}]})
    runs = list(runs)
    if not runs:
        raise LookupError(f'no run matching {spec!r} in {project}')
    return runs[0]


def scan_cloud_history(run, keys: list[str], samples: int = 100000) -> dict:
    """Fetch history for keys already resolved against `run.summary`.

    H1: pass only RESOLVED keys. `scan_history` returns zero rows silently when
    any one requested key is absent -- measured: seven keys of which two were
    absent returned 0 rows in 0.4 s and looked like a run with no data."""
    if not keys:
        return {}
    want = list(dict.fromkeys(list(keys) + ['_step']))
    series: dict[str, tuple[list, list]] = {}
    step = None
    for row in run.scan_history(keys=want, page_size=10000):
        if '_step' in row and row['_step'] is not None:
            step = row['_step']
        if step is None:
            continue
        for k, v in row.items():
            if k == '_step' or v is None or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and np.isfinite(v):
                s, vals = series.setdefault(k, ([], []))
                s.append(step)
                vals.append(v)
    return {k: (np.asarray(s, float), np.asarray(v, float))
            for k, (s, v) in series.items() if len(s) >= 3}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_path(run_id: str, last_step: float) -> str:
    return os.path.join(CACHE_DIR, f'{run_id}_{int(last_step)}.pkl')


def _cache_load(run_id: str, last_step: float) -> Optional[Run]:
    p = _cache_path(run_id, last_step)
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None  # a corrupt cache entry is a cache miss, never an error


def _cache_store(run: Run) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(run.run_id, run.last_step), 'wb') as f:
            pickle.dump(run, f)
    except Exception:
        pass  # caching is an optimisation; never fail a pull over it


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def pull(spec: str = 'newest', *, project: str = DEFAULT_PROJECT,
         wanted: Optional[Iterable[str]] = None, base: str = 'wandb',
         use_cache: bool = True) -> Run:
    """Fetch one run by spec.

    `spec` is 'newest', a local run directory, a run id, a display name, or a
    tag. Local is tried first: it needs no network and reaches a run that has not
    synced yet.

    `wanted=None` pulls every scalar (the sane default -- a full history is ~2 s).
    Pass a key list only when the cloud path makes breadth expensive, and pass
    RESOLVED keys (see keys.resolve) or the pull comes back empty.

    Raises EmptyPull when a run yields no rows. An empty pull is never returned
    as an empty result: silence here is indistinguishable from a run that did no
    work, and that ambiguity is the thing this package removes."""
    run = _pull_local(spec, wanted, base) if _looks_local(spec, base) else None
    if run is None:
        run = _pull_cloud(spec, project, wanted, use_cache)
    if not run.history:
        raise EmptyPull(
            f'{run.name}: no scalar rows. Either the run logged nothing, or a '
            f'requested key was unresolved -- an unresolved key silently zeroes '
            f'the WHOLE pull, it does not just drop that key. Resolve against '
            f'run.summary first (analysis.keys.resolve).')
    return run


def _looks_local(spec: str, base: str) -> bool:
    if spec == 'newest':
        return bool(_run_dirs(base))
    if os.path.isdir(spec):
        return True
    return any(spec in os.path.basename(d) for d in _run_dirs(base))


def _pull_local(spec: str, wanted, base: str) -> Optional[Run]:
    if os.path.isdir(spec):
        run_dir = spec
    else:
        dirs = _run_dirs(base)
        if not dirs:
            return None
        if spec == 'newest':
            run_dir = dirs[-1]
        else:
            match = [d for d in dirs if spec in os.path.basename(d)]
            if not match:
                return None
            run_dir = match[-1]

    name = os.path.basename(run_dir)
    run_id = name.rsplit('-', 1)[-1]
    config = _local_config(run_dir)
    summary = _local_summary(run_dir)
    history = scan_local_history(run_dir, set(wanted) if wanted else None)
    display = config.get('run_name')
    if isinstance(display, dict):
        display = display.get('value')
    return Run(run_id=run_id, name=display or name, source='local',
               config=config, summary=summary, history=history, path=run_dir)


def _pull_cloud(spec: str, project: str, wanted, use_cache: bool) -> Run:
    cr = _cloud_run(spec, project)
    summary = dict(cr.summary)
    config = dict(cr.config)
    last = summary.get('_step', 0) or 0

    if use_cache:
        cached = _cache_load(cr.id, last)
        if cached is not None:
            return cached

    from . import keys as K
    want = list(wanted) if wanted else list(summary)
    resolved = K.resolve(set(summary), want, K.detect_route(config))
    history = scan_cloud_history(cr, K.live_keys(resolved))

    run = Run(run_id=cr.id, name=cr.name, source='cloud',
              config=config, summary=summary, history=history)
    if use_cache:
        _cache_store(run)
    return run


def list_local(base='wandb', limit: int = 20) -> list[str]:
    """Newest local run directories first."""
    return list(reversed(_run_dirs(base)))[:limit]
