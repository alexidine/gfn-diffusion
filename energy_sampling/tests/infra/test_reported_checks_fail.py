"""
The conftest hook that makes a reported `check` fail the test that reported it.

WHY THIS FILE EXISTS. The hook is the thing standing between "a suite prints FAIL
and pytest says passed" and a real verdict, and it is invisible from the suites it
protects -- they contain no assertion, by design. So a broken hook restores
exactly the blindness it was written to remove, and it restores it SILENTLY: every
protected suite goes green. Nothing else in the repo would notice.

The tests below therefore run pytest AS A SUBPROCESS on generated modules, and
read its exit status and summary line. That is deliberately end-to-end rather than
calling `_failed_since` directly: the unit is trivial and was never the risk. The
risk is the wiring -- a hook name pytest no longer calls, a `wrapper=True`
signature it rejects, a conftest that stops being loaded for a subdirectory -- and
none of that is reachable from a unit test of the helper.

Run: python -m pytest tests/infra/test_reported_checks_fail.py -q
"""

import contextlib
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[2]   # tests/<area>/x.py -> energy_sampling/

pytestmark = pytest.mark.slow   # spawns pytest; not the dev loop


#: A module in the shape the protected suites use: a module-level log, a `check`
#: that records and prints without raising, and tests carrying no assertion of
#: their own. `{spelling}` is the log's attribute name and `{oks}` the outcomes
#: the single test reports.
_MODULE = textwrap.dedent('''
    {spelling} = []

    def check(name, ok, detail=''):
        {spelling}.append((name, bool(ok), detail))
        print(f"  {{'PASS' if ok else 'FAIL'}}  {{name}}   {{detail}}")

    def test_reports():
        for i, ok in enumerate({oks}):
            check(f'case{{i}}', ok, 'detail')

    def test_clean():
        check('later test', True, 'must not inherit the earlier failure')
''')


@contextlib.contextmanager
def _module_under_the_repo_conftest(source: str):
    """Write `source` to a throwaway module INSIDE the repo tree, and yield it.

    Inside, not in `tmp_path`, and that is the whole subtlety: pytest finds
    conftest.py by walking UP from the collected file to rootdir, so a module in
    the system temp directory is collected with no repo conftest at all -- and
    the hook under test simply never runs. Every assertion here would then pass
    or fail for reasons having nothing to do with the hook.

    `SCRATCH/` is the home because pytest.ini already lists it in
    `norecursedirs`, so a file called `test_*.py` can live there for the length
    of one subprocess without the real suite ever collecting it."""
    scratch = HERE / 'SCRATCH' / f'checkhook_{os.getpid()}'
    scratch.mkdir(parents=True, exist_ok=True)
    mod = scratch / 'test_generated_checks.py'
    mod.write_text(source, encoding='utf-8')
    try:
        yield mod
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _pytest(mod: Path):
    """Run pytest on one module, from the repo root, as the developer would."""
    return subprocess.run(
        [sys.executable, '-m', 'pytest', str(mod), '-p', 'no:cacheprovider', '-q'],
        capture_output=True, text=True, cwd=str(HERE))


def _run(spelling='_RESULTS', oks=(True, True)):
    with _module_under_the_repo_conftest(
            _MODULE.format(spelling=spelling, oks=list(oks))) as mod:
        return _pytest(mod)


def test_a_reported_failure_fails_the_test():
    """THE POINT. Two of three checks report False; the test body itself asserts
    nothing. It must come out FAILED."""
    proc = _run(oks=(True, False, False))
    assert proc.returncode != 0, proc.stdout
    assert '1 failed' in proc.stdout, proc.stdout
    assert '2 of 3 reported checks failed' in proc.stdout, proc.stdout


def test_it_is_a_failure_and_not_a_teardown_error():
    """THE DISTINCTION THE FIXTURE APPROACH COULD NOT MAKE, and the reason this
    hook wraps the call phase instead. An assertion raised in fixture teardown is
    reported by pytest as `1 passed, 1 error` -- the test itself still counts as
    PASSED, with noise attached, which is barely better than silence. The
    reporting test must be named on a `FAILED` line and on no `ERROR` line."""
    proc = _run(oks=(False,))
    lines = [ln.strip() for ln in proc.stdout.splitlines()]
    assert any(ln.startswith('FAILED') and '::test_reports' in ln for ln in lines), proc.stdout
    assert not any(ln.startswith('ERROR') for ln in lines), proc.stdout
    # ...and the test that reported nothing is untouched, which is why the tally
    # reads 1 failed / 1 passed rather than 2 failed.
    assert '1 failed, 1 passed' in proc.stdout, proc.stdout


def test_every_check_in_the_test_still_runs_and_prints():
    """The property the raise-immediately approach gives up: a failing case must
    not stop the ones after it, or a partial failure stops being readable."""
    proc = _run(oks=(False, True, False))
    for i in range(3):
        assert f'case{i}' in proc.stdout, proc.stdout


def test_a_later_test_does_not_inherit_an_earlier_failure():
    """The log is module-level and accumulates, so the hook must scope its verdict
    to the entries THIS test appended. Without that, every test after the first
    failure fails too and the report points at the wrong one."""
    proc = _run(oks=(False,))
    assert '1 failed, 1 passed' in proc.stdout, proc.stdout


def test_all_passing_checks_leave_the_test_passing():
    """The other direction: a hook that fails everything would also make the
    suites 'not blind', and would be useless."""
    proc = _run(oks=(True, True))
    assert proc.returncode == 0, proc.stdout
    assert '2 passed' in proc.stdout, proc.stdout


@pytest.mark.parametrize('spelling', ['_RESULTS', '_R'])
def test_both_log_spellings_are_honoured(spelling):
    """`test_latent_gaussian.py` calls it `_RESULTS`; `test_batch_invariance.py`
    and `test_gpu_guard.py` call it `_R`. One mechanism, both names -- otherwise
    adopting it in a second file is a rename nobody remembers to do."""
    proc = _run(spelling=spelling, oks=(False,))
    assert proc.returncode != 0, proc.stdout
    assert '1 of 1 reported checks failed' in proc.stdout, proc.stdout


#: The OTHER blind shape: no log at all, a verdict handed back with `return`.
#: pytest discards it (with a warning), so the test passes whatever it computed.
_RETURNING = 'def test_verdict():\n    return {value}\n'


def test_a_false_return_fails_the_test():
    """`return ok` is as blind as an unread log: `test_vg_detach_center.py` still
    ends its tests that way. `test_replay_gating.py` did until this hook caught
    it -- three failing checks, a green file, and nothing else in the repo
    looking."""
    with _module_under_the_repo_conftest(_RETURNING.format(value='False')) as mod:
        proc = _pytest(mod)
    assert proc.returncode != 0, proc.stdout
    assert '1 failed' in proc.stdout, proc.stdout
    assert 'FALSE verdict' in proc.stdout, proc.stdout


@pytest.mark.parametrize('value', ['True', 'None', "(1, 2)", "'ok'"])
def test_a_true_or_absent_verdict_is_left_alone(value):
    """ONLY FALSE FAILS. A truthy return is loose style, not a failure -- pytest's
    own PytestReturnNotNoneWarning covers that -- and inventing a failure there is
    how a guard becomes something people switch off. `(1, 2)` is the real case:
    tests/crystal/test_periodic_scoring.py returns a tuple."""
    with _module_under_the_repo_conftest(_RETURNING.format(value=value)) as mod:
        proc = _pytest(mod)
    assert proc.returncode == 0, proc.stdout
    assert '1 passed' in proc.stdout, proc.stdout


def test_a_falsy_non_bool_verdict_also_fails():
    """0 and '' are verdicts too. A rule written on `is False` would miss a test
    that accumulates with `ok &= ...` and hands back an int."""
    with _module_under_the_repo_conftest(_RETURNING.format(value='0')) as mod:
        proc = _pytest(mod)
    assert proc.returncode != 0, proc.stdout
    assert 'FALSE verdict' in proc.stdout, proc.stdout


def test_the_two_shapes_do_not_interfere():
    """A module with BOTH a log and a return: the log's verdict is reported first,
    because it names the individual checks and is the more useful message."""
    src = ("_RESULTS = []\n"
           "def check(name, ok, detail=''):\n"
           "    _RESULTS.append((name, bool(ok), detail))\n"
           "def test_both():\n"
           "    check('a', False, 'detail')\n"
           "    return False\n")
    with _module_under_the_repo_conftest(src) as mod:
        proc = _pytest(mod)
    assert proc.returncode != 0, proc.stdout
    assert '1 of 1 reported checks failed' in proc.stdout, proc.stdout


def test_a_module_with_no_check_log_is_untouched():
    """The hook runs on EVERY test in the repo. A module that reports nothing must
    be exactly as it was."""
    with _module_under_the_repo_conftest(
            'def test_ok():\n    assert True\n') as mod:
        proc = _pytest(mod)
    assert proc.returncode == 0, proc.stdout
    assert '1 passed' in proc.stdout, proc.stdout
