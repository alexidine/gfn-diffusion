"""build_qm9_full_prior.py::to_trainer_chart keeps a row's orientation when its rotation is near pi.

The refit poses each row's asymmetric unit as the search scored it, fits the source molecule onto it (or onto its
inversion partner) by a proper rotation, and stores that rotation as the row's orientation. With MXtalTools
rotmat2rotvec, a rotation within ~2.4e-4 rad of pi came back as pi about (1, 1, 1) -- a different crystal from the
one whose energy the row stores. What each test would catch:

  * a refitted row (generic, or near pi) whose stored orientation no longer poses the molecule as it was posed;
  * NEGATIVE CONTROL: the same refit through the old conversion fails on the near-pi row, so a passing refit is not
    a test that cannot fail.

CPU, synthetic molecules, no energy computed.

    python -m pytest -q tests/crystal/test_trainer_chart_rotation.py
"""
import math

import pytest
import torch

import energy_sampling.build_qm9_full_prior as bqp
from mxtaltools.common import geometry_utils
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list


def _molecule(i):
    pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1], [0.4, 0.3, 0.9]]) * (1.0 + 0.1 * i)
    return MolData(z=torch.tensor([6, 7, 8, 6]), pos=pos, x=torch.zeros((4, 1)), identifier=f'syn-{i}',
                   mol_volume=torch.tensor(55.0), mass=torch.tensor(54.0), radius=torch.tensor(1.2),
                   z_prime=torch.tensor(1, dtype=torch.long))


def _batch(rotvecs):
    rows = []
    for i, rv in enumerate(rotvecs):
        rows.append(MolCrystalData(molecule=_molecule(i), sg_ind=2, z_prime=torch.tensor(1),
                                   cell_lengths=torch.tensor([8.0, 9.0, 10.0]),
                                   cell_angles=torch.tensor([1.5, 1.6, 1.7]),
                                   aunit_centroid=torch.tensor([0.2, 0.3, 0.6]),
                                   aunit_orientation=torch.as_tensor(rv, dtype=torch.float32),
                                   aunit_handedness=torch.tensor([1.0]), identifier=f'syn-{i}', max_z_prime=1))
    b = collate_data_list(rows, max_z_prime=1)
    b.box_analysis()   # the cell's fractional-to-Cartesian transforms, which pose_aunit reads
    return b


def _relative_pose(b):
    """Each row's posed asymmetric unit, centred on its heavy-atom centroid: the orientation alone."""
    p = b.clone()
    p.pose_aunit(std_orientation=True)
    heavy = (p.z.reshape(-1) > 1).double()[:, None]
    w = torch.zeros(p.num_graphs, 1, dtype=torch.float64).index_add_(0, p.batch, heavy)
    c = torch.zeros(p.num_graphs, 3, dtype=torch.float64).index_add_(0, p.batch, p.pos.double() * heavy) / w
    return p.pos.double() - c[p.batch], p.batch


def _rotvec(angle, axis):
    a = torch.tensor(axis, dtype=torch.float64)
    return (a / a.norm() * angle).tolist()


ROTVECS = [_rotvec(1.1, [0.3, -0.5, 0.8]),                     # generic
           _rotvec(math.pi - 2e-5, [0.6, 0.2, 0.77]),           # within the old conversion's failure band of pi
           _rotvec(math.pi - 1e-6, [-0.2, 0.9, 0.4])]


def _refit_errors(b):
    """Max per-row deviation (Angstrom) of the refitted pose from the original pose or its inversion partner."""
    std = b.clone()
    std.orient_molecule(mode='std')
    mol_pos = {ident: std.pos[std.batch == i].clone() for i, ident in enumerate(std.identifier)}
    before, batch = _relative_pose(b)
    keep, *_ = bqp.to_trainer_chart(b, mol_pos)
    assert bool(keep.all())
    after, _ = _relative_pose(b)
    err = []
    for i in range(b.num_graphs):
        x, y = before[batch == i], after[batch == i]
        err.append(min(float((y - x).norm(dim=1).max()), float((y + x).norm(dim=1).max())))
    return err


def test_refit_keeps_every_rows_pose_including_near_pi():
    err = _refit_errors(_batch(ROTVECS))
    assert max(err) < 1e-4, err


def test_old_conversion_loses_the_near_pi_rows_negative_control(monkeypatch):
    old = lambda R: geometry_utils.rotmat2rotvec(R.float(), warn_on_bad_determinant=False)
    monkeypatch.setattr(geometry_utils, 'rotmat2rotvec_stable', old)
    err = _refit_errors(_batch(ROTVECS))
    assert err[0] < 1e-4, err                       # a generic rotation survives either conversion
    assert max(err[1:]) > 0.1, err                  # the near-pi rows come back as a different orientation
