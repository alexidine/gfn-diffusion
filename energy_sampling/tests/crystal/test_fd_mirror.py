"""energies/fd_coordinates.py::MIRROR_MAPS, mirror_rows, mirror_state: the mirror image of an FD chart row (the molecule
reflected, X -> diag(-1, 1, 1) X) as a closed form per space group.

  * the closed form equals crystal_fd's canonicaliser on random chart points, against two slow references: the same
    crystal with its handedness flipped (groups with an improper operation: 2, 14, 15, 61, 33), and the crystal
    inverted through the origin described for the reflected molecule (every mapped group; in the Sohncke groups 4 and
    19 the enantiomorph crystal, the handedness flip being off the +1 sheet there); the two references agree where
    both exist;
  * the map is an involution, and its log-Jacobian is the numerical one;
  * a mixed-group batch on an FDSlotLayout state maps each row by its own group's map, in its own slots;
  * a group without a map, and a K-cut group (Cc), are refused.

    python -m pytest -q tests/crystal/test_fd_mirror.py
"""
import math

import pytest
import torch

from energies.fd_coordinates import (FDCoordinates, FDSlotLayout, KCUT_GROUPS, MIRROR_MAPS, frame_rows, mirror_rows,
                                     mirror_state)

F64 = torch.float64
RADIUS = 3.0
SOHNCKE = (4, 19)   # no improper operation: the mirror image is the enantiomorph crystal
M = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=F64))


def _chart(fc, c):
    L, A, t, rv, h = c.desc.crystal_params()
    return fc.from_params(torch.cat([L, A, t, rv], 1), h.reshape(-1))


def _reference_flip(fc, s):
    """The slow route: description of s, handedness flipped, canonicalised and charted (FDCoordinates.from_params)."""
    from mxtaltools.crystal_fd import canon
    desc, _ = fc.to_description(s)
    return _chart(fc, canon.canonicalize(desc.replace(h=-desc.h), sheet=1))


def _reference_inversion(fc, s):
    """The slow route: the crystal of s inverted through the origin (t -> -t, pose P -> -P M), described for the
    reflected molecule (handedness unchanged), canonicalised and charted."""
    from mxtaltools.crystal_fd import canon
    from mxtaltools.crystal_fd.description import CrystalDescription
    desc, _ = fc.to_description(s)
    inv = CrystalDescription.from_poses(desc.sg, desc.cell, -desc.t[:, 0], -desc.pose[:, 0] @ M)
    return _chart(fc, canon.canonicalize(inv, sheet=1))


def _circle_diff(a, b, wrap):
    d = (a - b).abs()
    w = torch.tensor(wrap, dtype=torch.bool)
    return torch.where(w, torch.minimum(d, 2 - d), d)   # a circle row's chart span is 2 (s = -1 and 1 one point)


def _points(fc, sg, n=400):
    g = torch.Generator().manual_seed(sg)
    return torch.rand(n, fc.dim, generator=g, dtype=F64) * 1.9 - 0.95   # off the chart's faces


def _check(fc, sg, s, ref):
    ok = ref.ok
    assert int(ok.sum()) > 0.75 * s.shape[0]
    got, _ = mirror_state(s, torch.full((s.shape[0],), sg), mirror_rows([sg]), frame_rows([sg]))
    diff = _circle_diff(got[ok], ref.s[ok], fc.wrap_mask).amax(1)
    assert float(diff.max()) < 1e-6, f'sg {sg}: {int((diff > 1e-6).sum())} rows differ, max {float(diff.max()):.3g}'


@pytest.mark.parametrize('sg', sorted(g for g in MIRROR_MAPS if g not in SOHNCKE))
def test_closed_form_matches_the_canonicaliser(sg):
    fc = FDCoordinates(sg, RADIUS)
    s = _points(fc, sg)
    _check(fc, sg, s, _reference_flip(fc, s))


@pytest.mark.parametrize('sg', sorted(MIRROR_MAPS))
def test_closed_form_matches_the_inverted_crystal(sg):
    fc = FDCoordinates(sg, RADIUS)
    s = _points(fc, sg)
    _check(fc, sg, s, _reference_inversion(fc, s))


@pytest.mark.parametrize('sg', sorted(MIRROR_MAPS))
def test_the_two_references_agree_or_the_flip_is_off_sheet(sg):
    """Where the group has an improper operation the inverted crystal is the same crystal, so the two references
    are one row; in a Sohncke group the flip cannot be put on the +1 sheet and the inversion is the only reference."""
    fc = FDCoordinates(sg, RADIUS)
    s = _points(fc, sg, 64)
    flip, inv = _reference_flip(fc, s), _reference_inversion(fc, s)
    assert int(inv.ok.sum()) > 48
    if sg in SOHNCKE:
        assert int(flip.ok.sum()) == 0
        return
    both = flip.ok & inv.ok
    assert int(both.sum()) > 48
    assert float(_circle_diff(flip.s[both], inv.s[both], fc.wrap_mask).max()) < 1e-9


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


def test_a_mixed_group_batch_maps_each_row_by_its_own_group():
    """On one FDSlotLayout state over every mapped group, a row's mirror image is its own group's, computed on that
    group's chart rows and packed back into its slots; pads stay 0; the batch map is an involution."""
    groups = sorted(MIRROR_MAPS)
    layout = FDSlotLayout(groups)
    sgs = torch.tensor([g for g in groups for _ in range(3)])
    g = torch.Generator().manual_seed(11)
    x = torch.stack([layout.pack(int(sg), torch.rand(1, len(layout.slots[int(sg)]), generator=g, dtype=F64)
                                 * 1.8 - 0.9)[0] for sg in sgs])
    maps, rows = mirror_rows(groups), frame_rows(groups)
    got, lj = mirror_state(x, sgs, maps, rows)
    live = layout.live_mask(sgs)
    assert bool((got[~live] == 0).all())
    for i, sg in enumerate(sgs.tolist()):
        fc = layout.charts[sg]
        want, lj_w = mirror_state(layout.unpack(sg, x[i:i + 1]), torch.tensor([sg]), mirror_rows([sg]),
                                  frame_rows([sg]))
        d = _circle_diff(layout.unpack(sg, got[i:i + 1]), want, fc.wrap_mask)
        assert float(d.max()) < 1e-12, (sg, float(d.max()))
        assert abs(float(lj[i] - lj_w[0])) < 1e-12
    back, _ = mirror_state(got, sgs, maps, rows)
    d = (back - x).abs()
    wrap = torch.tensor(layout.wrap_mask)
    d[:, wrap] = torch.minimum(d[:, wrap], 2 - d[:, wrap])
    assert float(d.max()) < 1e-9


def test_unmapped_and_kcut_groups_are_refused():
    with pytest.raises(NotImplementedError, match='no mirror map for sg'):
        mirror_rows([1])
    for g in KCUT_GROUPS:   # Cc (9) included: its fold moves more than the orientation rows
        with pytest.raises(NotImplementedError, match='K-cut'):
            mirror_rows([g])
    with pytest.raises(NotImplementedError, match='K-cut'):
        mirror_rows([2, 9])
    maps, rows = mirror_rows([2]), frame_rows([2])
    with pytest.raises(ValueError, match='without a map'):
        mirror_state(torch.zeros(2, 12, dtype=F64), torch.tensor([2, 14]), maps, rows)
