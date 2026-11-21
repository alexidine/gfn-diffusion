"""noised_prior.live_kernel 'chart': noised_prior.chart_kick (a fresh kick by arithmetic on a chart point),
chart_step_lengths (its Angstrom scale) and the settings that select it.

    python -m pytest -q tests/crystal/test_chart_kernel.py
"""
from types import SimpleNamespace

import pytest
import torch

import noised_prior as NP
from energies.fd_coordinates import FDCoordinates

pytestmark = pytest.mark.fast


def _kick(center, per_A=None, live=None, wrap=None, lo=-2.0, hi=-0.375, redraws=4):
    n, d = center.shape
    per_A = torch.ones(n, d) if per_A is None else per_A
    live = torch.ones(n, d, dtype=torch.bool) if live is None else live
    wrap = torch.zeros(d, dtype=torch.bool) if wrap is None else wrap
    return NP.chart_kick(center, per_A, live, wrap, lo, hi, redraws)


def test_every_point_stays_in_the_box_and_the_step_follows_the_radius_law():
    torch.manual_seed(0)
    center = torch.zeros(20000, 12)
    x, out, kept = _kick(center, per_A=torch.full((20000, 12), 0.5), lo=-1.0, hi=-0.5)
    assert out == 0.0 and kept == 0.0 and float(x.abs().max()) <= 1
    r = (x / 0.5).norm(dim=1)      # back in Angstrom
    assert float(r.min()) >= 10 ** -1.0 - 1e-5 and float(r.max()) <= 10 ** -0.5 + 1e-5
    # log-uniform radius: the median is at the middle of the log range
    assert abs(float(torch.log10(r).median()) + 0.75) < 0.01


def test_the_direction_is_isotropic_over_the_live_coordinates():
    torch.manual_seed(1)
    x, _, _ = _kick(torch.zeros(40000, 4), lo=-1.0, hi=-1.0)
    u = x / x.norm(dim=1, keepdim=True)
    assert float(u.mean(0).abs().max()) < 0.01
    assert float(((u ** 2).mean(0) - 0.25).abs().max()) < 0.01


def test_a_pad_coordinate_never_moves():
    torch.manual_seed(2)
    live = torch.ones(500, 4, dtype=torch.bool)
    live[:, 2] = False
    center = torch.full((500, 4), 0.25)
    x, _, _ = _kick(center, live=live)
    assert torch.equal(x[:, 2], center[:, 2]) and float((x[:, 0] - center[:, 0]).abs().max()) > 0


def test_a_circle_coordinate_wraps_and_an_interval_row_is_redrawn():
    torch.manual_seed(3)
    center = torch.tensor([[0.999, 0.0]]).repeat(4000, 1)
    x, out, kept = _kick(center, wrap=torch.tensor([True, False]), lo=-1.0, hi=-1.0)
    assert out == 0.0 and kept == 0.0
    assert bool((x[:, 0] < 0).any()) and float(x.abs().max()) <= 1      # crossed the seam to the other side
    y, out, kept = _kick(center, lo=-1.0, hi=-1.0)                      # the same rows on an interval
    assert 0.3 < out < 0.7 and float(y.abs().max()) <= 1                # about half step past the wall on the first try


def test_a_row_that_cannot_land_keeps_its_centre_and_is_counted():
    torch.manual_seed(4)
    center = torch.tensor([[0.5, 0.5], [0.0, 0.0]])
    per_A = torch.tensor([[1e7, 1e7], [1.0, 1.0]])
    x, out, kept = _kick(center, per_A=per_A, lo=-1.0, hi=-1.0, redraws=2)
    assert torch.equal(x[0], center[0]) and not torch.equal(x[1], center[1])
    assert out == pytest.approx(0.5) and kept == pytest.approx(0.5)


def test_a_row_near_a_wall_keeps_its_radius_and_almost_never_returns_its_centre():
    torch.manual_seed(6)
    center = torch.zeros(40000, 12)
    center[:, 0] = 0.95        # one wall within the kick's reach
    x, out, kept = _kick(center, lo=-1.0, hi=-0.2)
    assert out > 0.02 and kept == 0.0
    r = (x - center).norm(dim=1)
    assert float(r.min()) > 0
    assert abs(float(torch.log10(r).mean()) + 0.6) < 0.01     # the log-uniform law's mean, not shortened by the wall


def test_the_step_scale_predicts_the_fd_distance_of_a_small_move():
    from mxtaltools.crystal_fd.metric import FDMetric
    gen = torch.Generator().manual_seed(5)
    n = 64
    radius = torch.full((n,), 4.0, dtype=torch.float64)
    rg = torch.full((n,), 2.0, dtype=torch.float64)
    fc = FDCoordinates(2, radius)
    s = (torch.rand(n, 12, generator=gen, dtype=torch.float64) * 1.2 - 0.6)
    steps = NP.chart_step_lengths(fc, s, rg, 2)
    assert bool(torch.isfinite(steps).all()) and bool((steps > 0).all())
    k = 5
    x = s.clone()
    x[:, k] += 2e-3
    m = FDMetric(2, 1.0)
    a, _ = fc.to_description(s)
    b, _ = fc.to_description(x)
    A, B = (m.prepare(dict(cell=d.cell, t=d.t[:, 0], R=d.R[:, 0])) for d in (a, b))
    P = m.frame(A['T'], B['T'])
    th = m.theta(A['R'], P[:, None] @ B['QR'])
    ce = m.cent(A['t'], B['t'], A['T'], P @ B['T'])
    moved = (m.cell2(A['cell'], B['cell'])[:, None] + ce * ce + (rg[:, None] * th) ** 2).min(1).values.sqrt()
    assert float((moved / (2e-3 * steps[:, k]) - 1).abs().median()) < 0.02


def _settings(**block):
    return NP.settings(SimpleNamespace(noised_prior=block))


def test_settings_accept_the_chart_kernel_and_refuse_a_stray_or_bad_range():
    assert _settings(live_kernel='chart')['live_log_range'] is None
    assert list(_settings(live_kernel='chart', live_log_range=[-1.0, -0.2])['live_log_range']) == [-1.0, -0.2]
    with pytest.raises(ValueError, match="read by live_kernel 'chart' only"):
        _settings(live_kernel='metric', live_log_range=[-1.0, -0.2])
    with pytest.raises(ValueError, match='lo <= hi'):
        _settings(live_kernel='chart', live_log_range=[-0.2, -1.0])
    with pytest.raises(ValueError, match='live_log_range'):
        _settings(live_kernel='chart', live_log_range=[-1.0])
