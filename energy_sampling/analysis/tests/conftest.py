"""Real captured runs, as pytest fixtures.

Every fixture is a REAL RUN, not a mock. What each one proves is recorded in
`fixtures/_capture.py`; the short version:

  tb_ramp     TB/unconditional, terminal stage. Ray calibration FIRED (26
              events), z-cal fired, ratio balance controller live, and Fwd Frac
              pinned at 0.05 BY DECLARATION -- a pin that is not a fault.
  vg_normal   Conditional VarGrad. `ray_calibration.enabled` is true and the
              probe never calibrated: `lr_ctrl/calibrations` is 0 for the whole
              run and no `raycal/*` series exists. A real inert mechanism.
  vg_blowup   Its sibling arm.
  mle_only    Died in phase 1, so it is on the MLE/prior route.
  buildout    Five stages, and the only fixture carrying `protocol/thr_*`.
  tb_resumed  Resumed from an explicit checkpoint -- the §4 chaining confound.
  ring_probe  Two arms of one battery, for the §4 comparability checks.
  ring_cal

Module-scoped, so the deep copy in `fixtures.mutate` is what protects a test
from another test's edits.
"""

import pytest

from analysis.tests import fixtures


def _fixture(name):
    @pytest.fixture(scope='module')
    def _f():
        return fixtures.load(name)
    return _f


tb_ramp = _fixture('tb_ramp')
vg_normal = _fixture('vg_normal')
vg_blowup = _fixture('vg_blowup')
mle_only = _fixture('mle_only')
buildout = _fixture('buildout')
tb_resumed = _fixture('tb_resumed')
ring_probe = _fixture('ring_probe')
ring_cal = _fixture('ring_cal')


@pytest.fixture(scope='module')
def all_runs():
    """Every captured run. Use for invariants that must hold on all real data --
    'this check never crashes' and 'this check does not fire on a run where the
    mechanism plainly ran' are both properties of the corpus, not of one run."""
    return {n: fixtures.load(n) for n in fixtures.names()}
