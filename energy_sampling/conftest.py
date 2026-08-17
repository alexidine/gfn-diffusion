"""
Test tiering, so there is a dev loop that is not the whole suite.

MEASURED, 2026-08-16, 721 tests:

    torch-free modules       491 tests    66 s     (0.13 s/test)
    torch-importing modules  230 tests   ~17 min   (~4.4 s/test)

A third of the tests own ~94% of the wall clock. Those are the ones that build
models, run rollouts, or read real priors off the data drive; they are worth
having, but not worth paying for on every edit.

    pytest -m fast     the dev loop -- no torch, no GPU, no data drive
    pytest             everything, unchanged

THE DEFAULT IS STILL EVERYTHING, deliberately. A suite whose default silently
skips things is how this repo ended up with six invariant tests that had not run
in months (they were collection errors, and the runner reported success). The
fast lane is opt-in and named; it is not a quieter default.

HOW A TEST IS CLASSIFIED. Automatically, by whether its module imported `torch`.
Mechanical rather than a hand-maintained list, because a list of slow files goes
stale the first time someone adds one -- and a stale tier is worse than none,
since it promises a coverage it no longer has.

`torch` is a proxy for cost, not the cost itself. It is a good proxy here (the
measurement above is exactly this split) and a cheap one. Where it is wrong, say
so in the file with an explicit marker, which always wins:

    pytestmark = pytest.mark.fast   # torch imported, but nothing is built
    pytestmark = pytest.mark.slow   # no torch, but reads the data drive
"""

import re
from pathlib import Path

import pytest

_TIER_MARKS = {'fast', 'slow'}

# Matches a torch import ANYWHERE in the file, including inside a function.
# A module-namespace check misses those, and the miss is not academic:
# bench/test_surface_fitness.py and bench/test_tracking.py both defer
# `import torch` into the test body and together take 180 s for 11 tests -- the
# most expensive per-test in the suite. A namespace check put them in the FAST
# lane, which would have made the fast lane a lie.
_TORCH_IMPORT = re.compile(r'^\s*(?:import\s+torch\b|from\s+torch\b)', re.M)

_cache: dict[str, bool] = {}


def _module_imports_torch(module) -> bool:
    """True if the test module's SOURCE imports torch, at any indentation.

    Source-scanned rather than namespace-checked, per the note above. Read once
    per file and cached; a module whose source cannot be read is treated as slow,
    since guessing 'fast' is the direction that silently under-reports cost.

    A module that pulls torch only TRANSITIVELY (config_snapshot -> utils ->
    torch) is deliberately NOT caught: it pays the import once, which is seconds,
    not the per-test cost this tiering exists to separate."""
    path = getattr(module, '__file__', None)
    if not path:
        return True
    hit = _cache.get(path)
    if hit is None:
        try:
            hit = bool(_TORCH_IMPORT.search(Path(path).read_text(encoding='utf-8')))
        except Exception:
            hit = True
        _cache[path] = hit
    return hit


def pytest_collection_modifyitems(config, items):
    for item in items:
        # An explicit marker in the file always wins over the proxy.
        if _TIER_MARKS & {m.name for m in item.iter_markers()}:
            continue
        module = getattr(item, 'module', None)
        item.add_marker(pytest.mark.slow if _module_imports_torch(module)
                        else pytest.mark.fast)
