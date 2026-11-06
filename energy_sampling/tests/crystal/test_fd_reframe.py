"""FDCoordinates.reframe: the FD chart rows of the same crystals with the molecule presented in a rotated frame R0 (the
rows nf frames trains on). Two failures would run without an error: a re-expressed row describing another crystal
(the wrong side of the composition, a canonicalisation slip), and a row that no longer canonicalises. So:

  * the unit cell built from (molecule, s) equals the unit cell built from (R0 molecule, s'), atom for atom up to
    lattice translations and the order of symmetry images; NEGATIVE CONTROLS: s unchanged, and the wrong side of the
    composition (R R0), both build another crystal;
  * reframing by R0 then by R0^T returns s.

    python -m pytest -q tests/crystal/test_fd_reframe.py
"""
import math

import pytest
import torch

from energies.fd_coordinates import FDCoordinates
from models.spline_flow import quat_to_rotmat
from mxtaltools.common.geometry_utils import apply_rotation_to_batch
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

pytestmark = pytest.mark.fast
RADIUS = 1.6


def _crystals(n=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1], [0.4, 0.3, 0.9]]) * (1 + 0.03 * i)
        mol = MolData(z=torch.tensor([6, 7, 8, 6]), pos=pos, x=torch.zeros((4, 1)), identifier=f'syn-{i}',
                      mol_volume=torch.tensor(55.0), mass=torch.tensor(54.0), radius=torch.tensor(RADIUS),
                      z_prime=torch.tensor(1, dtype=torch.long))
        axis = torch.randn(3, generator=g)
        rv = axis / axis.norm() * (0.2 + 2.8 * torch.rand(1, generator=g))
        rows.append(MolCrystalData(molecule=mol, sg_ind=2, z_prime=torch.tensor(1),
                                   cell_lengths=torch.tensor([5.0, 5.5, 6.0]) + torch.rand(3, generator=g),
                                   cell_angles=torch.tensor([1.5, 1.6, 1.7]),
                                   aunit_centroid=torch.tensor([0.2, 0.3, 0.4]),
                                   aunit_orientation=rv.float(),
                                   aunit_handedness=torch.tensor([1.0]), identifier=f'syn-{i}', max_z_prime=1))
    return collate_data_list(rows, max_z_prime=1)


def _frac_cell(b, s, fd, R0=None):
    """(cell parameters [n, 6], per crystal the unit cell's fractional atom coordinates wrapped into [0, 1)) of the
    rows s with the molecule presented as R0 applied to its coordinates (None: as stored), posed as given."""
    c = b.clone()
    if R0 is not None:
        c.pos = apply_rotation_to_batch(c.pos, R0.float(), c.batch)
    p, h, _ = fd.to_params(s)
    c.cell_lengths, c.cell_angles = p[:, :3], p[:, 3:6]
    c.aunit_centroid, c.aunit_orientation, c.aunit_handedness = p[:, 6:9], p[:, 9:12], h.reshape(-1, 1)
    c.box_analysis()
    c.pose_aunit(std_orientation=False)
    c.build_unit_cell()
    frac = []
    sizes = (torch.diff(c.ptr) * c.sym_mult.reshape(-1)).tolist()   # unit_cell_pos: each crystal's atoms in turn
    assert sum(sizes) == c.unit_cell_pos.shape[0]
    for i, x in enumerate(torch.split(c.unit_cell_pos.double(), sizes)):
        f = x @ torch.linalg.inv(c.T_fc[i].double()).T
        frac.append(f - torch.floor(f))
    return p[:, :6].double(), frac


def _same_cell(a, b, tol=1e-3):
    """Every atom of a has an atom of b within tol (fractional, minimum image) and the counts agree."""
    if a.shape != b.shape:
        return False
    d = a[:, None, :] - b[None, :, :]
    d = d - torch.round(d)
    return bool((d.norm(dim=-1).min(1).values < tol).all())


def _frames(n, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    return quat_to_rotmat(q / q.norm(dim=1, keepdim=True))


def test_reframed_rows_build_the_same_crystal():
    b = _crystals()
    fd = FDCoordinates(2, RADIUS)
    r = fd.from_params(b.full_cell_parameters(), b.aunit_handedness.reshape(-1))
    assert bool(r.ok.all())
    s = r.s
    R0 = _frames(int(b.num_graphs), 1)
    r2 = fd.reframe(s, R0)
    assert bool(r2.ok.all()), r2.status
    cell, ref = _frac_cell(b, s, fd)
    cell2, got = _frac_cell(b, r2.s, fd, R0)
    assert torch.allclose(cell, cell2, atol=1e-4)                      # the cell does not depend on the frame
    assert all(_same_cell(a, g) for a, g in zip(ref, got))
    # negative controls: the row left as it was, and the wrong side of the composition (R R0, not R R0^T)
    _, stale = _frac_cell(b, s, fd, R0)
    assert not any(_same_cell(a, g) for a, g in zip(ref, stale))
    wrong = fd.reframe(s, R0.transpose(1, 2))
    _, bad = _frac_cell(b, wrong.s, fd, R0)
    assert not any(_same_cell(a, g) for a, g in zip(ref, bad))


def test_reframe_round_trips():
    b = _crystals(seed=3)
    fd = FDCoordinates(2, RADIUS)
    s = fd.from_params(b.full_cell_parameters(), b.aunit_handedness.reshape(-1)).s
    R0 = _frames(int(b.num_graphs), 4)
    back = fd.reframe(fd.reframe(s, R0).s, R0.transpose(1, 2))
    assert bool(back.ok.all())
    d = back.s - s
    wrap = torch.tensor(fd.wrap_mask)
    d = torch.where(wrap, d - 2 * torch.round(d / 2), d)
    assert float(d.abs().max()) < 1e-6, float(d.abs().max())
