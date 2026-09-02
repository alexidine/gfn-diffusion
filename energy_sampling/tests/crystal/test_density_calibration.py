"""
The calibration gate and its analytic targets.

WHY THE TARGETS NEED THEIR OWN TESTS. `calibrate` compares a candidate energy
against `true_log_p`. If that reference density is not actually normalised, or
its periodic images are wrong, then every verdict the gate issues is wrong --
and wrong in the most expensive direction, because a broken reference makes a
GOOD model look like it has a slope error. The gate is only as trustworthy as
its targets, so each one is checked against grid integration below.

    python test_density_calibration.py
"""
import math
import os
import sys

import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here)):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energies.density_calibration import (  # noqa: E402
    Calibration, calibrate, wrapped_gaussian_draw, wrapped_mixture_draw,
    warped_mixture_draw, _wrapped_mixture_log_p)


def _grid_integral_2d(log_p_fn, wrap, lo, hi, g=700):
    ax = [torch.linspace(lo[i], hi[i], g, dtype=torch.float64) for i in range(2)]
    X, Y = torch.meshgrid(ax[0], ax[1], indexing='ij')
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    lp = log_p_fn(pts.float())
    cell = ((hi[0] - lo[0]) / (g - 1)) * ((hi[1] - lo[1]) / (g - 1))
    return float(lp.exp().sum() * cell)


def test_wrapped_gaussian_target_is_normalised():
    width = 0.45
    for wrap in ([True, True], [True, False]):
        norm = -0.5 * math.log(2 * math.pi * width * width)
        offs = torch.arange(-3, 4, dtype=torch.float64) * 2.0

        def log_p(pts, wrap=wrap):
            out = torch.zeros(pts.shape[0], dtype=torch.float64)
            for d in range(2):
                col = pts[:, d].to(torch.float64)
                if wrap[d]:
                    out += torch.logsumexp(
                        norm - 0.5 * ((col.unsqueeze(1) + offs) / width) ** 2, dim=1)
                else:
                    out += norm - 0.5 * (col / width) ** 2
            return out

        lo = [-1.0 if w else -8.0 for w in wrap]
        hi = [1.0 if w else 8.0 for w in wrap]
        got = _grid_integral_2d(log_p, wrap, lo, hi)
        assert abs(got - 1.0) < 5e-3, (wrap, got)


def test_wrapped_mixture_target_is_normalised():
    for wrap in ([True, True], [True, False]):
        wm = torch.as_tensor(wrap)
        mg = torch.Generator().manual_seed(1234)
        means = 0.55 * torch.randn(8, 2, generator=mg)
        means[:, wm] = means[:, wm] - 2.0 * torch.round(means[:, wm] / 2.0)
        logits = 0.5 * torch.randn(8, generator=mg)
        log_w = torch.log_softmax(logits.to(torch.float64), dim=0)

        lo = [-1.0 if w else -6.0 for w in wrap]
        hi = [1.0 if w else 6.0 for w in wrap]
        got = _grid_integral_2d(
            lambda pts: _wrapped_mixture_log_p(pts, means, log_w, wm, 0.18, 2.0, 3),
            wrap, lo, hi)
        assert abs(got - 1.0) < 5e-3, (wrap, got)


def test_warped_target_is_a_bijection_of_the_circle():
    """The warp must fix +-1, or the period is destroyed and log p is nonsense."""
    g = torch.Generator().manual_seed(3)
    y, lp = warped_mixture_draw(20000, 2, [True, True], amp=0.25, generator=g)
    assert float(y.min()) >= -1.0 - 1e-5 and float(y.max()) <= 1.0 + 1e-5, \
        (float(y.min()), float(y.max()))
    assert bool(torch.isfinite(lp).all())

    try:
        warped_mixture_draw(10, 2, [True, True], amp=0.5)   # amp*pi > 1
    except ValueError as exc:
        assert 'invertible' in str(exc)
    else:
        raise AssertionError('a non-invertible warp amplitude was accepted')


def test_warped_target_is_out_of_the_gaussian_family():
    """Otherwise it adds nothing over the mixture target it is built from."""
    g = torch.Generator().manual_seed(4)
    z, _ = wrapped_mixture_draw(40000, 2, [True, True], generator=g)
    g2 = torch.Generator().manual_seed(4)
    y, _ = warped_mixture_draw(40000, 2, [True, True], amp=0.25, generator=g2)
    # the warp is componentwise and monotone, so it shows up as skew, not spread
    def skew(t):
        c = t - t.mean(0)
        return float((c ** 3).mean() / c.std(0).pow(3).mean().clamp_min(1e-9))
    assert abs(skew(y) - skew(z)) > 0.05, (skew(z), skew(y))


def test_calibrate_recovers_a_known_slope():
    """A model that is exactly s * (-log p) + c must measure as slope s."""
    g = torch.Generator().manual_seed(5)
    x, lp = wrapped_gaussian_draw(3000, 4, [True, False, True, False], 0.4, generator=g)
    for s in (0.5, 1.0, 1.7):
        cal = calibrate(lambda q, s=s: s * (-lp) + 11.0, x, lp)
        assert abs(cal.slope - s) < 1e-6, (s, cal.slope)
        assert cal.passes == (abs(s - 1.0) <= cal.tolerance)


def test_calibrate_is_not_fooled_by_correlation():
    """The whole point of the gate: high correlation, wrong slope."""
    g = torch.Generator().manual_seed(6)
    x, lp = wrapped_gaussian_draw(3000, 4, [True, False, True, False], 0.4, generator=g)
    noise = 0.02 * torch.randn(3000, generator=g).to(torch.float64)
    cal = calibrate(lambda q: 0.7 * (-lp) + noise, x, lp)
    assert cal.corr > 0.99 and not cal.passes, str(cal)


def test_calibrate_reconciles_devices():
    if not torch.cuda.is_available():
        return
    g = torch.Generator().manual_seed(7)
    x, lp = wrapped_gaussian_draw(500, 3, [True, True, False], 0.4, generator=g)
    cal = calibrate(lambda q: (-lp).to('cuda'), x.cuda(), lp)
    assert abs(cal.slope - 1.0) < 1e-6, str(cal)


if __name__ == '__main__':
    passed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            passed += 1
            print(f'  ok  {name}')
    print(f'\n{passed} passed')
