"""CPU tests for rotational augmentation of the atom-token flow (inn_modeller.rotate_latent, quat_to_rotmat,
random_frames, and the frame columns of _ContextFlow.context_input under --rotate_frames).

WHAT THIS FILE IS FOR. With --rotate_frames the likelihood stage presents each molecule in a random frame R0 and
trains on its rows re-expressed for that frame. Two failures would run without an error: (1) the re-expressed row
describing a different crystal from the stored one (a wrong side of the composition, a canonicalisation slip), and
(2) the encoder not seeing the frame the row was re-expressed for. So:

  * the crystal built from (molecule, x) equals the crystal built from (R0 molecule, x'), atom for atom; rows 0-8 do
    not change; x -> x' -> x is exact; NEGATIVE CONTROLS: x unchanged, and the wrong-side composition R0^T R, both
    build a different crystal;
  * a row's context in frame q equals its context in the identity frame with the atom table rotated by q, and
    differs from its standard-frame context (negative control); a batch mixing frames of one molecule gives each row
    its own frame's context.

    python -m pytest -q tests/models/test_inn_rotate_frames.py
"""
import math
import os
import sys

import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for p in (_here, os.path.dirname(_here)):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

import inn_modeller as im  # noqa: E402
from mxtaltools.common.geometry_utils import apply_rotation_to_batch  # noqa: E402
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData  # noqa: E402
from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402

WRAP = [False] * 7 + [True, True, False, True, True]


def _crystals(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1], [0.4, 0.3, 0.9]]) * (1 + 0.05 * i)
        mol = MolData(z=torch.tensor([6, 7, 8, 6]), pos=pos, x=torch.zeros((4, 1)), identifier=f'syn-{i}',
                      mol_volume=torch.tensor(55.0), mass=torch.tensor(54.0), radius=torch.tensor(1.2),
                      z_prime=torch.tensor(1, dtype=torch.long))
        axis = torch.randn(3, generator=g)
        rv = axis / axis.norm() * (0.2 + 2.8 * torch.rand(1, generator=g))
        if rv[2] < 0:
            rv = -rv / rv.norm() * (2 * math.pi - rv.norm())   # the chart's canonical form: axis in the upper half
        rows.append(MolCrystalData(molecule=mol, sg_ind=2, z_prime=torch.tensor(1),
                                   cell_lengths=torch.tensor([8.0, 9.0, 10.0]) + torch.rand(3, generator=g),
                                   cell_angles=torch.tensor([1.5, 1.6, 1.7]),
                                   aunit_centroid=torch.tensor([0.2, 0.3, 0.6]),
                                   aunit_orientation=rv.float(),
                                   aunit_handedness=torch.tensor([1.0]), identifier=f'syn-{i}', max_z_prime=1))
    b = collate_data_list(rows, max_z_prime=1)
    b.box_analysis()
    return b


def _unit_cells(b, x, R0=None):
    c = b.clone()
    if R0 is not None:
        c.pos = apply_rotation_to_batch(c.pos, R0.float(), c.batch)
    c.latent_to_cell_params(x.float())
    c.pose_aunit(std_orientation=False)
    c.build_unit_cell()
    return c.unit_cell_pos


def _random_R(n, seed):
    torch.manual_seed(seed)
    return im.quat_to_rotmat(im.random_frames(n, 'cpu', torch.float64))


def test_rotated_rows_build_the_same_crystal():
    b = _crystals()
    x = b.latent_params(gauge_fix_free_axes=True)
    R0 = _random_R(b.num_graphs, 1)
    xp = im.rotate_latent(x, R0)
    assert float((xp - x)[:, :9].abs().max()) == 0.0                     # only the orientation rows change
    d = (_unit_cells(b, x) - _unit_cells(b, xp, R0)).norm(dim=-1)
    assert float(d.max()) < 1e-4, float(d.max())
    # negative controls: the latent left as it was, and the wrong side of the composition
    assert float((_unit_cells(b, x) - _unit_cells(b, x, R0)).norm(dim=-1).max()) > 0.1
    wrong = x.clone()
    wrong[:, 9:12] = im.rotate_latent(x, R0.transpose(1, 2))[:, 9:12]   # R R0, not R R0^T
    assert float((_unit_cells(b, x) - _unit_cells(b, wrong, R0)).norm(dim=-1).max()) > 0.1


def test_round_trip_is_exact():
    b = _crystals(seed=3)
    x = b.latent_params(gauge_fix_free_axes=True).double()
    R0 = _random_R(b.num_graphs, 4)
    back = im.rotate_latent(im.rotate_latent(x, R0), R0.transpose(1, 2))
    d = back - x
    d[:, 10:12] = d[:, 10:12] - 2 * torch.round(d[:, 10:12] / 2)        # wrapped rows
    assert float(d.abs().max()) < 1e-9, float(d.abs().max())


def test_quaternions_are_proper_rotations():
    torch.manual_seed(5)
    R = im.quat_to_rotmat(im.random_frames(500, 'cpu', torch.float64))
    eye = torch.eye(3, dtype=torch.float64).expand(500, 3, 3)
    assert float((R @ R.transpose(1, 2) - eye).abs().max()) < 1e-12
    assert float((torch.linalg.det(R) - 1).abs().max()) < 1e-12


def _flow():
    torch.manual_seed(6)
    f = im.ConditionalLatentFlow(12, WRAP, [], 5, 32, 64, 1, 8, (64,), 5.0, 0, transformer=(64, 2, 4),
                                 atoms=dict(node_dim=32, convs=2, cutoff=10.0, radial=8), frames=True)
    with torch.no_grad():
        for prm in f.parameters():
            prm.add_(0.1 * torch.randn_like(prm))
    f = f.double().eval()
    g = torch.Generator().manual_seed(7)
    n_mol, a = 4, 6
    z = torch.zeros(n_mol, a, dtype=torch.long)
    pos = torch.zeros(n_mol, a, 3, dtype=torch.float64)
    n = torch.tensor([4, 6, 5, 3])
    for i in range(n_mol):
        z[i, :n[i]] = torch.tensor([6, 7, 8, 6, 1, 1])[:n[i]]
        pos[i, :n[i]] = 1.5 * torch.randn(int(n[i]), 3, generator=g, dtype=torch.float64)
    f.flow.set_atom_table(z, pos, n)
    return f, z, pos, n


def test_frame_rotates_the_atoms_the_encoder_sees():
    f, z, pos, n = _flow()
    torch.manual_seed(8)
    q = im.random_frames(4, 'cpu', torch.float64)
    ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).expand(4, 4)
    cid = torch.arange(4, dtype=torch.float64)[:, None]
    with torch.no_grad():
        (vec_q, tok_q, _), _ = f.flow.context_input(torch.cat([cid, q], 1))
        f.flow.set_atom_table(z, pos @ im.quat_to_rotmat(q).transpose(1, 2), n)     # the table itself rotated
        (vec_r, tok_r, _), _ = f.flow.context_input(torch.cat([cid, ident], 1))
        f.flow.set_atom_table(z, pos, n)
        (vec_i, _, _), _ = f.flow.context_input(torch.cat([cid, ident], 1))
    assert float((vec_q - vec_r).abs().max()) < 1e-10 and float((tok_q - tok_r).abs().max()) < 1e-10
    assert float((vec_q - vec_i).abs().max()) > 1e-4               # negative control: the frame matters


def test_rows_of_one_molecule_in_different_frames_keep_their_own_frame():
    f, *_ = _flow()
    torch.manual_seed(9)
    q = im.random_frames(3, 'cpu', torch.float64)
    c = torch.cat([torch.zeros(3, 1, dtype=torch.float64), q], 1)      # molecule 0 in three frames, one batch
    with torch.no_grad():
        (vec, _, _), _ = f.flow.context_input(c)
        single = torch.cat([f.flow.context_input(c[i:i + 1])[0][0] for i in range(3)])
    assert float((vec - single).abs().max()) < 1e-10
    assert float((vec[0] - vec[1]).abs().max()) > 1e-4


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
