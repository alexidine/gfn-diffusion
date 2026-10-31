"""buffer.py::FDChartState, the buffers' x_fn on crystal_coordinates 'fd'. What each test would catch:

  * an FD buffer whose x_fn cannot be pickled, so its checkpoint sidecar fails to save once the run's live
    objects hold locks (2026-10-30: a bound Modeller method did, on every FD run that saved);
  * the reader not returning a row's cached fd_state, or canonicalising a crystal to another chart point than
    FDCoordinates.from_params does, or giving a crystal with no chart point anything but NaN.

CPU, no model or energy.

    python -m pytest -q tests/crystal/test_fd_buffer_state.py
"""
import inspect
import pickle
from types import SimpleNamespace

import pytest
import torch

from energy_sampling.buffer import FDChartState
from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast

P_MIPCAS = [3.42406, 8.815858, 8.946602, 1.665358, 1.578231, 1.610735, 0.250891, 0.32719, 0.261217,
            1.072443, 1.292957, 1.133738]
RADIUS = 3.369637966156006


class _Rows(SimpleNamespace):
    def keys(self):
        return [k for k in vars(self) if not k.startswith('_')]

    def full_cell_parameters(self):
        return self._params


def _rows(params, fd_state=None):
    n = len(params)
    r = _Rows(_params=torch.tensor(params, dtype=torch.float32), aunit_handedness=torch.ones(n, 1),
              sg_ind=torch.full((n,), 2), radius=torch.full((n,), RADIUS), num_graphs=n)
    if fd_state is not None:
        r.fd_state = fd_state
    return r


def test_the_fd_buffer_reader_pickles_and_the_modeller_hands_it_out():
    m = Modeller.__new__(Modeller)
    m.args = SimpleNamespace(z_primes=[1])
    m.energy_function = SimpleNamespace(crystal_coordinates='fd', data_ndim=12, is_crystal=True)
    x_fn = m._buffer_kwargs()['x_fn']
    assert isinstance(x_fn, FDChartState) and not inspect.ismethod(x_fn)
    back = pickle.loads(pickle.dumps(x_fn))
    assert back.data_ndim == 12


def test_the_reader_returns_the_cached_chart_point_else_canonicalises_else_nan():
    pytest.importorskip('mxtaltools.crystal_fd')
    from energy_sampling.energies.fd_coordinates import FDCoordinates
    cached = torch.rand(2, 12)
    assert torch.equal(FDChartState(12)(_rows([P_MIPCAS] * 2, fd_state=cached)), cached)
    far = list(P_MIPCAS)
    far[2] = 45.0                                          # no chart point: outside the length box
    x = FDChartState(12)(_rows([P_MIPCAS, far]))
    want = FDCoordinates(2, RADIUS).from_params(torch.tensor([P_MIPCAS], dtype=torch.float64),
                                                torch.ones(1, dtype=torch.float64))
    assert bool(want.ok.all())
    assert torch.allclose(x[0], want.s[0].float(), atol=1e-5)
    assert bool(torch.isnan(x[1]).all())
