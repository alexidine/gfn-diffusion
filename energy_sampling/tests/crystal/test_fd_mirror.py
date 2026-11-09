"""energies/fd_coordinates.py::MIRROR_MAPS, mirror_rows, mirror_state: the mirror image of an FD chart row (the molecule
reflected, X -> diag(-1, 1, 1) X) as a closed form per space group.

  * the closed form equals crystal_fd's canonicaliser on the same crystal with its handedness flipped, then charted
    (the slow, general route), on random chart points of P-1 and P2_1/c;
  * the map is an involution, and its log-Jacobian is the numerical one;
  * a group without a measured map, and a K-cut group, are refused.

    python -m pytest -q tests/crystal/test_fd_mirror.py
"""
import math

import pytest
import torch

from energies.fd_coordinates import FDCoordinates, MIRROR_MAPS, frame_rows, mirror_rows, mirror_state

F64 = torch.float64
RADIUS = 3.0


def _reference(fc, s):
    """The slow route: description of s, handedness flipped, canonicalised and charted (FDCoordinates.from_params)."""
    from mxtaltools.crystal_fd import canon
    desc, _ = fc.to_description(s)
    c = canon.canonicalize(desc.replace(h=-desc.h), sheet=1)
    L, A, t, rv, h = c.desc.crystal_params()
    return fc.from_params(torch.cat([L, A, t, rv], 1), h.reshape(-1))


def _circle_diff(a, b, wrap):
    d = (a - b).abs()
    w = torch.tensor(wrap, dtype=torch.bool)
    return torch.where(w, torch.minimum(d, 2 - d), d)   # a circle row's chart span is 2 (s = -1 and 1 one point)


@pytest.mark.parametrize('sg', sorted(MIRROR_MAPS))
def test_closed_form_matches_the_canonicaliser(sg):
    fc = FDCoordinates(sg, RADIUS)
    g = torch.Generator().manual_seed(sg)
    s = torch.rand(400, fc.dim, generator=g, dtype=F64) * 1.9 - 0.95   # off the chart's faces
    ref = _reference(fc, s)
    ok = ref.ok
    assert int(ok.sum()) > 300
    maps, rows = mirror_rows([sg]), frame_rows([sg])
    got, _ = mirror_state(s, torch.full((s.shape[0],), sg), maps, rows)
    diff = _circle_diff(got[ok], ref.s[ok], fc.wrap_mask).amax(1)
    assert float(diff.max()) < 1e-6, f'sg {sg}: {int((diff > 1e-6).sum())} rows differ, max {float(diff.max()):.3g}'


@pytest.mark.parametrize('sg', sorted(MIRROR_MAPS))
def test_involution_and_jacobian(sg):
    fc = FDCoordinates(sg, RADIUS)
    g = torch.Generator().manual_seed(100 + sg)
    s = torch.rand(64, fc.dim, generator=g, dtype=F64) * 1.8 - 0.9
    maps, rows = mirror_rows([sg]), frame_rows([sg])
    sgs = torch.full((s.shape[0],), sg)
    m, lj = mirror_state(s, sgs, maps, rows)
    back, lj_back = mirror_state(m, sgs, maps, rows)
    assert float(_circle_diff(back, s, fc.wrap_mask).max()) < 1e-9
    assert float((lj + lj_back).abs().max()) < 1e-9           # the inverse's log-Jacobian cancels
    for i in range(4):                                          # against the numerical Jacobian
        x = s[i:i + 1].clone()
        J = torch.autograd.functional.jacobian(lambda v: mirror_state(v, sgs[:1], maps, rows)[0], x)
        J = J.reshape(fc.dim, fc.dim)
        assert math.isclose(float(torch.linalg.slogdet(J)[1]), float(lj[i]), abs_tol=1e-6)


def test_unmapped_and_kcut_groups_are_refused():
    with pytest.raises(NotImplementedError, match='no mirror map'):
        mirror_rows([4])
    with pytest.raises(NotImplementedError):
        mirror_rows([7])
    maps, rows = mirror_rows([2]), frame_rows([2])
    with pytest.raises(ValueError, match='without a map'):
        mirror_state(torch.zeros(2, 12, dtype=F64), torch.tensor([2, 14]), maps, rows)
