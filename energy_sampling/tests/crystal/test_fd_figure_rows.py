"""The 'Lattice Latents Distribution' figure on FD runs reads the run's own chart rows (eval/evaluations.py::eval_figs
passes each draw's fd_state as plot_batch_cell_params' samples).

Why: mxtaltools' fd_chart_coordinates charts at the trainer's latent length box (CrystalChart.for_latent_v1), the run
at energies/fd_coordinates.py::length_box, so the cell-length rows of one crystal land on other s there; the reference
and anchor rows are the run's. Run from ``energy_sampling/``:

    python -m pytest -q tests/crystal/test_fd_figure_rows.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fd_energy_option import _energy, _molecules  # noqa: E402

pytestmark = pytest.mark.fast


def _scored(capsys, n=32):
    fd = _energy(capsys, crystal_coordinates='fd')
    s = torch.rand(n, 12, generator=torch.Generator().manual_seed(4)) * 1.6 - 0.8
    _, b = fd.analyze_crystal_batch(s, _molecules(n), torch.full((n,), 2.5), return_batch=True)
    return b


def test_mxtaltools_charts_the_length_rows_elsewhere_and_the_figure_takes_the_runs_rows(capsys):
    b = _scored(capsys)
    state = b.fd_state.double()
    mxt, ok, _ = b.fd_chart_coordinates()
    names = [d.name for d in b._fd_chart(rows=[0]).dims]
    diff = (mxt[ok] - state[ok]).abs().amax(0)
    lengths = [names.index(k) for k in ('log_c', 'r2', 'r1')]
    rest = [i for i in range(len(names)) if i not in lengths]
    assert float(diff[lengths].max()) > 0.1          # the two length boxes disagree ...
    assert float(diff[rest].max()) < 1e-4            # ... and nothing else does

    def sample_trace(fig, col):
        axis = 'x' if col == 0 else f'x{col + 1}'
        t = next(t for t in fig.data if t.name == 'Samples' and (t.xaxis or 'x') == axis)
        return np.asarray(t.y)
    own = b.plot_batch_cell_params(space='fd', show=False, return_fig=True, samples=state)
    mx = b.plot_batch_cell_params(space='fd', show=False, return_fig=True)
    c = names.index('log_c')
    assert not np.allclose(sample_trace(own, c), sample_trace(mx, c))
    with pytest.raises(ValueError, match='rows for a batch'):
        b.plot_batch_cell_params(space='fd', show=False, return_fig=True, samples=state[:3])
