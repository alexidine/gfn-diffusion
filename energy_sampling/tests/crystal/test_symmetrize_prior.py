"""data_processing.symmetrize_prior.image_centroids: the image rule of the normaliser-symmetric prior for space
group 2, Z' = 1, in the trainer's handedness +1 chart (x in [0, 1/2], y and z in [0, 1)).

Every source gets 8 rows: its four half-cell shifts in y/z, then the same four again -- on the OPPOSITE x face when
the source sits on one, as plain duplicates otherwise. Synthetic centroids only; the build itself rescoring every row
through the energy function is the proof on real data (the module's own checks).
"""
import pytest
import torch

from energy_sampling.data_processing.symmetrize_prior import (CELL_EDGE, ROWS_PER_SOURCE, X_BOX, YZ_SHIFTS, choose_one,
                                                               image_centroids)

INTERIOR = [0.20, 0.30, 0.70]
LOW_FACE = [0.00, 0.10, 0.60]
HIGH_FACE = [0.50, 0.90, 0.20]


def _images(rows):
    return image_centroids(torch.tensor(rows, dtype=torch.float32))


def test_eight_rows_per_source_identity_first():
    img, source, shift_id, face_copy, duplicate = _images([INTERIOR, LOW_FACE])
    assert img.shape == (2 * ROWS_PER_SOURCE, 3) and img.dtype == torch.float32
    assert source.tolist() == [0] * 8 + [1] * 8
    assert shift_id.tolist() == [0, 1, 2, 3] * 4
    assert torch.equal(img[0], torch.tensor(INTERIOR)) and torch.equal(img[8], torch.tensor(LOW_FACE))


def test_an_interior_row_gets_its_four_yz_images_twice():
    img, _, _, face_copy, duplicate = _images([INTERIOR])
    want = torch.tensor([[0.20, (0.30 + dy) % 1, (0.70 + dz) % 1] for dy, dz in YZ_SHIFTS])
    assert torch.allclose(img[:4], want, atol=1e-6)
    assert torch.equal(img[4:], img[:4])                     # x never moves off a face
    assert not face_copy.any() and duplicate.tolist() == [False] * 4 + [True] * 4


@pytest.mark.parametrize('row, other_x', [(LOW_FACE, X_BOX), (HIGH_FACE, 0.0)])
def test_a_face_row_gets_the_opposite_face_as_its_second_four(row, other_x):
    img, _, _, face_copy, duplicate = _images([row])
    assert torch.allclose(img[:4, 0], torch.full((4,), row[0])) and torch.allclose(img[4:, 0], torch.full((4,), other_x))
    assert torch.equal(img[4:, 1:], img[:4, 1:])             # same y/z images on both faces
    assert face_copy.tolist() == [False] * 4 + [True] * 4 and not duplicate.any()


def test_a_row_near_but_not_on_the_face_is_interior():
    img, _, _, face_copy, duplicate = _images([[2e-4, 0.1, 0.1]])
    assert not face_copy.any() and duplicate[4:].all() and torch.allclose(img[:, 0], torch.full((8,), 2e-4))


def test_the_image_set_is_closed():
    # the images of the images are the same set of points, for interior and face rows alike
    for row in (INTERIOR, LOW_FACE, HIGH_FACE):
        first = _images([row])[0]
        again = image_centroids(first)[0]
        d = torch.cdist(again.double(), first.double())
        assert float(d.min(1).values.max()) < 1e-6


def test_a_shift_landing_in_the_clip_sliver_snaps_to_zero():
    # 0.49996 + 0.5 = 0.99996 lies in (CELL_EDGE, 1) nearer 1: mxtaltools' builders would clip it to CELL_EDGE, so
    # it is written as 0, the same point by a lattice translation
    img, *_ = _images([[0.2, 0.49996, 0.3]])
    assert CELL_EDGE == 0.9999 and float(img[2, 1]) == 0.0 and float(img[1, 1]) == pytest.approx(0.49996)


@pytest.mark.parametrize('bad', [[[0.6, 0.1, 0.1]], [[0.1, 1.0, 0.1]], [[0.1, -0.1, 0.1]]])
def test_a_centroid_outside_the_box_is_refused(bad):
    with pytest.raises(ValueError):
        _images(bad)


def test_one_layout_cycles_the_valid_images_within_each_group():
    # group 0: five interior rows cycle 0..3 from offset 0; group 1: three face rows cycle 0..7 from offset 1
    face = torch.tensor([False, True, False, True, False, True, False, False])
    group = torch.tensor([0, 1, 0, 1, 0, 1, 0, 0])
    k = choose_one(face, group)
    assert k[group == 0].tolist() == [0, 1, 2, 3, 0]
    assert k[group == 1].tolist() == [1, 2, 3]
    assert bool((k[~face] < 4).all())                        # an interior row never takes an x-shifted image


def test_one_layout_mixed_group_cycles_interior_and_face_rows_separately():
    # one group, interior and face rows interleaved: each class cycles its own valid set from its own offset
    face = torch.tensor([False, True] * 8)
    k = choose_one(face, torch.zeros(16, dtype=torch.long))
    assert k[~face].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert k[face].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_one_layout_spreads_singleton_groups_across_the_images():
    # 4000 groups of one interior row and 800 of one face row: the offsets alone must balance the images
    face = torch.tensor([False] * 4000 + [True] * 800)
    k = choose_one(face, torch.arange(4800))
    assert torch.bincount(k[~face], minlength=4).tolist() == [1000] * 4
    assert torch.bincount(k[face], minlength=8).tolist() == [100] * 8


def test_the_constant_matches_mxtaltools():
    from mxtaltools.crystal_search.crystal_opt_utils import CELL_EDGE as MXT_EDGE
    assert CELL_EDGE == MXT_EDGE
