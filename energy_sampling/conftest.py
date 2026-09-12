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

import functools
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


# ---------------------------------------------------------------------------
# REPORTED CHECKS MUST FAIL THE TEST THAT REPORTED THEM
#
# Several suites in this repo report through a helper of the shape
#
#     def check(name, ok, detail=''):
#         _RESULTS.append((name, bool(ok), detail))
#         print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
#
# written so that one failing case still prints every OTHER case in the same
# test -- which is what makes a partial failure readable while you are working
# on the thing under test. The cost was that the log was read only by the file's
# own `main()`, so under pytest a test full of FAILs passed. `test_latent_gaussian
# .py` sat that way with six genuinely failing checks.
#
# WHY THE CALL PHASE AND NOT A TEARDOWN FIXTURE. A fixture is the obvious place
# and it is the wrong one: pytest classifies an assertion raised in teardown as
# an ERROR and still counts the test itself as PASSED, so those six broken checks
# would read as `8 passed, 2 errors` -- a pass with noise attached.
# `test_batch_invariance.py` measured that and rejected it, choosing instead to
# raise from inside `check`, which does fail properly but stops at the first bad
# case and so gives up the print-the-rest property.
#
# Wrapping `pytest_pyfunc_call` gets both. The wrapper runs INSIDE the call
# phase, so the body has already finished (every check ran and printed) and an
# exception raised here is an ordinary FAILURE, attributed to the test.
# ---------------------------------------------------------------------------

#: Module attributes holding a suite's own check log, as [(name, ok, detail), ...].
#: Two spellings are in use -- `_RESULTS` in test_latent_gaussian.py, `_R` in
#: test_batch_invariance.py and test_gpu_guard.py -- and both are read, so
#: adopting this in another file is not also a rename someone has to remember.
#: Those three are the only module-level bindings of either name in the suite,
#: so neither is at risk of catching an unrelated list.
_CHECK_LOG_ATTRS = ('_RESULTS', '_R')


def _check_log(module):
    for attr in _CHECK_LOG_ATTRS:
        log = getattr(module, attr, None)
        if isinstance(log, list):
            return log
    return None


def _failed_since(log, start):
    """Entries appended during this test whose `ok` is false.

    Entries that do not unpack to the (name, ok, detail) shape are IGNORED
    rather than guessed at -- a wrong guess here invents a failure, and this
    hook's whole purpose is that its verdict can be trusted."""
    out = []
    for entry in log[start:]:
        if isinstance(entry, (tuple, list)) and len(entry) == 3:
            name, ok, detail = entry
            if not ok:
                out.append((name, detail))
    return out


@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """Fail a test that reported a failed `check`, or that RETURNED a false verdict.

    TWO SHAPES, ONE HOOK, because this repo has both. The log shape is described
    above. The other is a test that ends `return ok` -- pytest discards the value
    (it warns, and pytest 9 will make it an error, but today the test passes), so
    a suite written that way is exactly as blind as one whose log nobody reads.
    `test_vg_detach_center.py` is written that way. `test_replay_gating.py` was
    too, and is the case for this hook: three of its checks reported FAIL for
    months while pytest called the file green.

    ONLY A FALSE RETURN FAILS, never a merely non-None one. A test that returns a
    truthy value is using `return` loosely and pytest's own warning covers it;
    inventing a failure there would make this hook something to switch off. The
    verdict is taken from the value the body produced, captured by wrapping
    `pyfuncitem.obj` for the duration of the call -- pytest's own
    `pytest_pyfunc_call` throws the return value away, so there is nothing to read
    downstream of it.

    Scoped to the entries this test appended, not the whole log, because the log
    is module-level and accumulates across the file.

    Does nothing when the test raised on its own: `wrapper=True` re-raises at the
    `yield`, so the body below never runs and a real error is never masked."""
    log = _check_log(getattr(pyfuncitem, 'module', None))
    start = len(log) if log is not None else 0

    captured, original = {}, pyfuncitem.obj

    @functools.wraps(original)
    def _capture_verdict(*args, **kwargs):
        out = original(*args, **kwargs)
        captured['value'] = out
        return out

    pyfuncitem.obj = _capture_verdict
    try:
        result = yield
    finally:
        pyfuncitem.obj = original

    failed = _failed_since(log, start) if log is not None else []
    if failed:
        raise AssertionError(
            f'{len(failed)} of {len(log) - start} reported checks failed:\n' +
            '\n'.join(f'  FAIL {name}   {detail}' for name, detail in failed))

    verdict = captured.get('value')
    if verdict is not None and not verdict:
        raise AssertionError(
            f'the test returned {verdict!r}, a FALSE verdict. pytest discards a '
            f'return value, so this would otherwise have passed. Assert instead of '
            f'returning, or the next reader has no way to know it failed.')
    return result
