"""The wiki's references resolve against the code and the canonical config.

The writing protocol says every symbol a page cites exists and that the
notation is "checked by a script". This is the script running in the suite, so
a page that cites a symbol a refactor renamed fails here rather than being
found by the next reader.

FAST TIER. `docs/wiki/check_refs.py` resolves by `ast` and `yaml`: no torch, no
CUDA, no data drive. `test_checker_does_not_import_the_trainer` pins that,
because the cheapest way to lose it is to import a trainer module for a
constant -- `utils` pulls torch transitively, and the tier is assigned by
whether the module imports torch at all.

A reference that is deliberately unresolvable belongs in
docs/wiki/check_refs_allow.txt with its reason, not in a skip here."""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / 'docs' / 'wiki' / 'check_refs.py'


def _load():
    spec = importlib.util.spec_from_file_location('wiki_check_refs', CHECKER)
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: `@dataclass` resolves annotations through
    # sys.modules, and raises on a module that is not in it.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_every_reference_resolves():
    failures, counts = _load().check()
    assert counts['symbol'] > 1000 and counts['cfg'] > 500 and counts['link'] > 100, (
        f'the checker found almost nothing to check ({counts}), which is how a '
        f'broken extractor reads as a green suite')
    assert not failures, 'unresolved references:\n' + '\n'.join(
        f'  {r.page}:{r.line}  {r.text}  --  {why}' for r, why in failures)


def test_allowlisted_entries_carry_a_reason():
    allow = _load().load_allowlist()
    assert allow, 'the allowlist is empty; it should still hold the notation example'
    missing = [ref for ref, reason in allow.items() if not reason]
    assert not missing, f'allowlist entries with no reason: {missing}'


def test_checker_does_not_import_the_trainer():
    """Importing the checker must not pull torch or a trainer module in."""
    probe = (
        'import importlib.util, sys\n'
        f'spec = importlib.util.spec_from_file_location("c", r"{CHECKER}")\n'
        'm = importlib.util.module_from_spec(spec)\n'
        'sys.modules["c"] = m\n'
        'spec.loader.exec_module(m)\n'
        'bad = [n for n in ("torch", "train", "utils", "gflownet_losses", "protocol",\n'
        '                   "buffer", "controller") if n in sys.modules]\n'
        'print(",".join(bad))\n')
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True,
                         text=True, cwd=str(ROOT), timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '', (
        f'importing the checker pulled in {out.stdout.strip()}; it resolves by '
        f'ast and yaml so that it runs in the fast tier')
