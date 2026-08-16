"""Real captured runs, loaded as `pull.Run` objects. Network-free.

`_capture.py` documents how these were made and why each run is here. Use
`load(name)` in a test; use `mutate(run, ...)` to build the counter-case a check
must FIRE on -- a check that has never fired has not been tested.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def names() -> list[str]:
    return sorted(f[:-len('.config.json')] for f in os.listdir(HERE)
                  if f.endswith('.config.json'))


def load(name: str):
    """One captured run, as the `pull.Run` the rest of the package consumes."""
    from analysis.pull import Run

    with open(os.path.join(HERE, f'{name}.config.json'), encoding='utf-8') as f:
        config = json.load(f)
    with open(os.path.join(HERE, f'{name}.summary.json'), encoding='utf-8') as f:
        summary = json.load(f)

    history = {}
    with np.load(os.path.join(HERE, f'{name}.history.npz')) as z:
        for k in z.files:
            key, part = k.rsplit('#', 1)
            s, v = history.setdefault(key, [None, None])
            if part == 's':
                history[key][0] = np.asarray(z[k], float)
            else:
                history[key][1] = np.asarray(z[k], float)
    history = {k: (s, v) for k, (s, v) in history.items()}

    run_id = name
    return Run(run_id=run_id, name=name, source='local',
               config=config, summary=summary, history=history)


def mutate(run, *, config=None, summary=None, history=None, drop=()):
    """A deep copy of `run` with edits applied -- the mutation half of a
    mutation test.

    `config`/`summary` are shallow-merged (config values are wrapped as
    {'value': x} to match the captured form). `history` maps key -> (steps,
    values). `drop` removes keys from history AND summary, which is how a
    'the trace was never logged' case is built.

    Deep-copied because the module-scoped fixtures are shared: an in-place edit
    in one test silently changes what every later test is looking at, and that
    failure mode reads as a flaky check rather than a corrupted fixture.
    """
    out = copy.deepcopy(run)
    for k, v in (config or {}).items():
        out.config[k] = {'value': v}
    for k, v in (summary or {}).items():
        out.summary[k] = v
    for k, (s, v) in (history or {}).items():
        out.history[k] = (np.asarray(s, float), np.asarray(v, float))
    for k in drop:
        out.history.pop(k, None)
        out.summary.pop(k, None)
    return out


def pin(run, key, value=None):
    """Replace a series with a CONSTANT, keeping its steps -- the R14 dead-sensor
    mutation. Defaults to pinning at the series' own last value."""
    s, v = run.history[key]
    const = float(v[-1]) if value is None else float(value)
    return mutate(run, history={key: (s, np.full_like(v, const))},
                  summary={key: const})
