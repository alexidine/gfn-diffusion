"""
CPU tests for the rational-quadratic spline's knot-derivative options (energies/prior_flow.py: _rq_spline,
rq_spline_inverse, _knot_derivatives, _tied_derivatives).

WHAT THIS FILE IS FOR. With free knot derivatives, a bin holding almost no mass can still meet a steep knot: the
density then spikes at that knot and collapses inside the bin, and nothing in the forward map, the inverse or the
log-determinant fails -- the map stays monotone and invertible. So the contract checked here is quantitative: with
`tied` the derivative at a knot is bounded by the slopes of the two bins it joins, and the density inside any bin
with no pinned end stays within 8 x F of the bin's mean slope. Each bound is paired with a NEGATIVE CONTROL: the free
derivatives on the same parameters must violate it, or the test would pass with tying deleted.

    python test_rq_spline_tied.py
"""
import os
import sys

import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for p in (_here, os.path.dirname(_here)):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energies.prior_flow import _rq_spline, _spline_knots, rq_spline_inverse  # noqa: E402

K = 32
CIRC = torch.tensor([False, False, True, True, False, False])
BND = torch.tensor([False, False, False, False, True, True])
BOUND = torch.tensor([5.0, 5.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
FLOOR = torch.full((6,), 1e-5, dtype=torch.float64)


def _params(n, scale=3.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    params = torch.randn(n, 6, 3, K, generator=g, dtype=torch.float64) * scale
    end_raw = torch.randn(n, 6, generator=g, dtype=torch.float64) * scale
    x = (torch.rand(n, 6, generator=g, dtype=torch.float64) * 2 - 1) * BOUND
    return params, end_raw, x


def _interior_ratio(params, end_raw, tied):
    """max over interior knots of (knot derivative / mean slope of the bin to its right)."""
    _, _, w, h, d, _ = _spline_knots(params, BOUND, CIRC, BND, end_raw, FLOOR, tied)
    return float((d[..., 1:-1] / (h / w)[..., 1:]).max())


def test_tied_inverts_and_logdet_matches_autograd():
    for tied in (True, 1.0, 3.0):
        params, end_raw, x = _params(1024)
        xr = x.clone().requires_grad_(True)
        y, ld = _rq_spline(xr, params, BOUND, CIRC, BND, end_raw, FLOOR, tied=tied)
        slope = torch.autograd.grad(y.sum(), xr)[0]
        assert float((ld - torch.log(slope)).abs().max()) < 1e-8, tied
        xi = rq_spline_inverse(y.detach(), params, BOUND, CIRC, BND, end_raw, FLOOR, tied)
        assert float((xi - x).abs().max()) < 1e-5, tied


def test_tied_true_is_factor_two():
    params, end_raw, x = _params(256)
    y1, l1 = _rq_spline(x, params, BOUND, CIRC, BND, end_raw, FLOOR, tied=True)
    y2, l2 = _rq_spline(x, params, BOUND, CIRC, BND, end_raw, FLOOR, tied=2.0)
    assert torch.equal(y1, y2) and torch.equal(l1, l2)


def test_pure_factor_is_the_harmonic_mean():
    params, end_raw, _ = _params(64)
    _, _, w, h, d, _ = _spline_knots(params, BOUND, CIRC, BND, end_raw, FLOOR, 1.0)
    s = h / w
    hm = 2 * s[..., :-1] * s[..., 1:] / (s[..., :-1] + s[..., 1:])
    assert torch.allclose(d[..., 1:-1], hm, rtol=1e-12, atol=0)


def test_knot_derivative_bound_with_negative_control():
    params, end_raw, _ = _params(512, scale=8.0)   # very uneven bins, many near the height floor
    for f in (1.0, 2.0):
        assert _interior_ratio(params, end_raw, f) <= 2 * f * (1 + 1e-9), f
    assert _interior_ratio(params, end_raw, False) > 100.0   # free derivatives: far above the bound


def test_density_inside_bins_bounded_with_negative_control():
    params, end_raw, _ = _params(256, scale=8.0)
    xs = torch.linspace(0, 1, 101, dtype=torch.float64)
    worst = {}
    for tied in (2.0, False):
        cumw, _, w, h, _, _ = _spline_knots(params, BOUND, CIRC, BND, end_raw, FLOOR, tied)
        s = h / w
        m = 0.0
        for k in range(1, K - 1):   # interior bins: neither end pinned to the identity tails
            xin = (cumw[..., k:k + 1] + w[..., k:k + 1] * xs).permute(2, 0, 1).reshape(-1, 6)
            pr = params.unsqueeze(0).expand(xs.numel(), -1, -1, -1, -1).reshape(-1, 6, 3, K)
            er = end_raw.unsqueeze(0).expand(xs.numel(), -1, -1).reshape(-1, 6)
            _, ld = _rq_spline(xin, pr, BOUND, CIRC, BND, er, FLOOR, tied=tied)
            m = max(m, float((ld.exp().view(xs.numel(), -1, 6) / s[..., k].unsqueeze(0)).max()))
        worst[tied] = m
    assert worst[2.0] <= 16.0 * (1 + 1e-6), worst
    assert worst[False] > 1e3, worst   # free derivatives: a knot spike hundreds of times the bin's mean density


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
