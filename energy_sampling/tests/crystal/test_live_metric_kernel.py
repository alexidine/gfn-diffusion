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


# -- the sidecar caches (noised_prior.cache_path) ------------------------------------------------------------------

def _cached_stub(capsys, tmp_path, seed=0):
    train, m, s0 = _stub(capsys)
    prior_file = tmp_path / 'prior.pt'
    prior_file.write_bytes(b'stand-in for a pre-noised prior file: the caches key on its size and mtime')
    m.args = SimpleNamespace(prior_path=str(prior_file), seed=seed)
    m._fd_chart_points = lambda b: train.Modeller._fd_chart_points(m, b)
    return train, m, prior_file


def test_chart_points_are_cached_beside_the_prior_and_read_back(capsys, tmp_path):
    train, m, prior_file = _cached_stub(capsys, tmp_path)
    batch = m.prior_dataset.batch
    first = train.Modeller._fd_chart_points_cached(m, batch, str(prior_file))
    side = list(tmp_path.glob('prior.pt.fdchart-*.pt'))
    assert len(side) == 1, side
    assert 'converted in' in capsys.readouterr().out
    calls = []
    m._fd_chart_points = lambda b: calls.append(int(b.num_graphs)) or train.Modeller._fd_chart_points(m, b)
    second = train.Modeller._fd_chart_points_cached(m, batch, str(prior_file))
    assert 'read from' in capsys.readouterr().out
    assert calls == [int(batch.num_graphs)]   # only the check sample was converted (here every row: 6 < 256)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_a_cache_that_disagrees_with_a_fresh_conversion_is_recomputed(capsys, tmp_path):
    train, m, prior_file = _cached_stub(capsys, tmp_path)
    batch = m.prior_dataset.batch
    good = train.Modeller._fd_chart_points_cached(m, batch, str(prior_file))
    side = next(tmp_path.glob('prior.pt.fdchart-*.pt'))
    d = torch.load(side, weights_only=False)
    d['state'] = d['state'] + 0.1   # as if written by other chart code
    torch.save(d, side)
    capsys.readouterr()
    again = train.Modeller._fd_chart_points_cached(m, batch, str(prior_file))
    assert 'disagree with the cache' in capsys.readouterr().out
    assert torch.equal(again[1], good[1])
    assert torch.equal(torch.load(side, weights_only=False)['state'], good[1])   # rewritten


def test_a_changed_prior_file_misses_the_cache(capsys, tmp_path):
    train, m, prior_file = _cached_stub(capsys, tmp_path)
    batch = m.prior_dataset.batch
    train.Modeller._fd_chart_points_cached(m, batch, str(prior_file))
    prior_file.write_bytes(b'a rebuilt prior file of another size')
    capsys.readouterr()
    train.Modeller._fd_chart_points_cached(m, batch, str(prior_file))
    assert 'converted in' in capsys.readouterr().out
    assert len(list(tmp_path.glob('prior.pt.fdchart-*.pt'))) == 2


def test_the_metric_bank_is_cached_per_seed(capsys, tmp_path):
    train, m, prior_file = _cached_stub(capsys, tmp_path)
    train.Modeller._fd_chart_points_cached(m, m.prior_dataset.batch, str(prior_file))   # sets the chart digest
    train.Modeller._metric_kernel_table(m)
    bank = m._basin['bank'].clone()
    assert 'written to' in capsys.readouterr().out
    train.Modeller._metric_kernel_table(m)
    assert 'read from' in capsys.readouterr().out
    assert torch.equal(m._basin['bank'], bank) and m._basin['draws'] == 18
    m.args.seed = 1
    train.Modeller._metric_kernel_table(m)
    assert 'drawn in' in capsys.readouterr().out and not torch.equal(m._basin['bank'], bank)


def test_no_bank_cache_without_a_checked_chart(capsys, tmp_path):
    train, m, prior_file = _cached_stub(capsys, tmp_path)   # no _fd_chart_points_cached call: no chart digest
    train.Modeller._metric_kernel_table(m)
    assert not list(tmp_path.glob('prior.pt.metricbank-*.pt'))
