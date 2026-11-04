"""noised_prior.live_kernel 'metric' (train.py::Modeller._metric_fresh, _metric_kernel_table): the prior's own build
kick, noised_prior.metric_kick, redrawn live around each drawn row and charted in the run's chart.

    python -m pytest -q tests/crystal/test_live_metric_kernel.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fd_energy_option import _energy, _molecules  # noqa: E402

pytestmark = pytest.mark.fast


def _stub(capsys, redraws=4):
    import train
    ef = _energy(capsys, crystal_coordinates='fd')
    # crystals built through the trainer latent (inside its box), as test_fd_energy_option's route comparison:
    # metric_kick passes every kick back through the latent box, which would move a crystal outside it
    n = 8
    g = torch.Generator().manual_seed(3)
    states = torch.rand(n, 12, generator=g) * 1.6 - 0.8
    states[:, :3] = 0.75 + 0.2 * torch.rand(n, 3, generator=g)
    _, batch = _energy(capsys).analyze_crystal_batch(states, _molecules(n), torch.full((n,), 2.5), return_batch=True)
    ok = ef.fd_coordinates_for(2, float(batch.radius[0])).from_params(
        batch.full_cell_parameters(), batch.aunit_handedness.reshape(-1)).ok
    batch = batch.subsample_new_batch(torch.nonzero(ok).flatten()[:6])
    m = SimpleNamespace(energy_function=ef, prior_dataset=SimpleNamespace(batch=batch),
                        _noised_prior_settings={'live_redraws': redraws, 'live_bank': 3, 'live_fresh_frac': 0.5},
                        _noised_prior_manifest={'child_kernel': 'metric', 'log_lo': -2.0, 'log_hi': -0.375})
    m._batch_latents = lambda b: train.Modeller._batch_latents(m, b)
    m._metric_fresh = lambda rows, chunk=4096: train.Modeller._metric_fresh(m, rows, chunk)
    return train, m, train.Modeller._batch_latents(m, batch).float()


def test_metric_draws_are_fresh_chart_points_around_their_rows(capsys):
    train, m, s0 = _stub(capsys)
    train.Modeller._metric_kernel_table(m)
    B = m._basin
    assert B['kind'] == 'metric' and (B['lo'], B['hi']) == (-2.0, -0.375)
    assert B['bank'].shape == (6, 3, 12) and torch.isfinite(B['bank']).all()
    assert B['draws'] == 18 and B['misses'] == 0
    from energies.fd_coordinates import FDCoordinates
    wrap = torch.tensor(FDCoordinates(2, 3.0).wrap_mask)
    diff = B['bank'] - s0[:, None]
    diff = torch.where(wrap, torch.remainder(diff + 1, 2) - 1, diff)   # a circle row's seam is no move
    d = diff.norm(dim=-1)
    assert bool((d > 0).all()) and float(d.max()) < 1.0, d   # moved, and near its row (kicks up to 10^-0.375 A)
    a, b = m._metric_fresh(torch.tensor([0, 0, 1])), m._metric_fresh(torch.tensor([0, 0, 1]))
    assert not torch.equal(a, b) and not torch.equal(a[0], a[1])   # fresh at every call and every row


def test_metric_live_kernel_needs_a_metric_build(capsys):
    train, m, _ = _stub(capsys)
    m._noised_prior_manifest = {'child_kernel': 'iso', 'log_lo': -2.0, 'log_hi': -0.5}
    with pytest.raises(ValueError, match="child_kernel 'iso'"):
        train.Modeller._metric_kernel_table(m)
