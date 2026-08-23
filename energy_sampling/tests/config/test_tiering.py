"""
Tests for the test tiering itself (conftest.py).

WHY THIS EXISTS. `pytest -m fast` is only useful if "fast" is true. A tier that
quietly mis-classifies an expensive file gives a dev loop that is slower than
advertised, and -- worse -- invites the belief that the fast lane covers more
than it does.

The first version of the classifier checked the module NAMESPACE for `torch`.
That silently put `bench/test_surface_fitness.py` and `bench/test_tracking.py`
in the fast lane, because both defer `import torch` into the test body. Together
they are 180 s for 11 tests, the most expensive per-test in the suite. The rule
below is the regression test for that.

Run: python -m pytest test_tiering.py -q
"""

import re
from pathlib import Path

import pytest

import conftest as C

HERE = Path(__file__).parent

# Files known to build models, run rollouts, or read the data drive. Each must
# land in `slow` however the classifier is implemented. This is a floor, not the
# definition -- the classifier is mechanical and covers more than this list.
KNOWN_EXPENSIVE = (
    'test_dead_latent_rows_deep.py',   # explicitly a statistical suite
    'test_conformer_levels.py',
    'test_periodic_scoring.py',
    'test_batch_invariance.py',        # loads real priors off the data drive
    'test_mxtaltools_crystal_boundary.py',  # CPU synthetic boundary; constructs crystal graphs
    'test_latent_gaussian.py',
    'bench/test_surface_fitness.py',   # defers `import torch` into the body
    'bench/test_tracking.py',          # ditto
)

# Files that must stay in the fast lane, or the dev loop stops being a loop.
KNOWN_CHEAP = (
    'test_crystal_operational_contract.py',
    'test_config_state.py',
    'test_config_invariants.py',
    'test_problems.py',
    'analysis/tests/test_keys.py',
    'analysis/tests/test_features.py',
)


class _FakeModule:
    def __init__(self, path):
        self.__file__ = str(path)


@pytest.mark.parametrize('rel', KNOWN_EXPENSIVE)
def test_known_expensive_files_classify_slow(rel):
    path = HERE / rel
    if not path.exists():
        pytest.skip(f'{rel} not present')
    assert C._module_imports_torch(_FakeModule(path)), \
        f'{rel} must classify as slow'


@pytest.mark.parametrize('rel', KNOWN_CHEAP)
def test_known_cheap_files_classify_fast(rel):
    path = HERE / rel
    if not path.exists():
        pytest.skip(f'{rel} not present')
    assert not C._module_imports_torch(_FakeModule(path)), \
        f'{rel} must stay in the fast lane'


def test_a_deferred_import_inside_a_function_is_caught(tmp_path):
    """THE bug the namespace check had. A module-level namespace lookup finds
    nothing here, because the import never runs at import time."""
    p = tmp_path / 'deferred.py'
    p.write_text('def test_x():\n    import torch\n    assert torch\n', encoding='utf-8')
    assert C._module_imports_torch(_FakeModule(p))


def test_a_from_import_is_caught(tmp_path):
    p = tmp_path / 'fromimp.py'
    p.write_text('from torch import Tensor\n', encoding='utf-8')
    assert C._module_imports_torch(_FakeModule(p))


def test_the_word_torch_in_prose_is_not_an_import(tmp_path):
    """The mutation for the tests above: a source scan must not fire on comments
    and strings, or everything classifies slow and the tier is vacuous."""
    p = tmp_path / 'prose.py'
    p.write_text('"""We do not import torch here."""\n'
                 '# torch would be too slow\n'
                 'MSG = "import torch to use the GPU"\n', encoding='utf-8')
    assert not C._module_imports_torch(_FakeModule(p))


def test_an_unreadable_module_is_slow_not_fast(tmp_path):
    """Guessing 'fast' is the direction that silently under-reports cost, so an
    unreadable or path-less module defaults to slow."""
    assert C._module_imports_torch(_FakeModule(tmp_path / 'does_not_exist.py'))
    assert C._module_imports_torch(_FakeModule(''))


def test_every_test_file_gets_exactly_one_tier():
    """No file may be both, and none may be neither -- either would make the two
    lanes stop partitioning the suite."""
    for path in list(HERE.glob('test_*.py')) + list(HERE.glob('*/test_*.py')):
        slow = C._module_imports_torch(_FakeModule(path))
        assert isinstance(slow, bool), path
