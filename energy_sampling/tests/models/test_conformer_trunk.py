"""The atoms-visible conformer energy model and the target it is pre-trained on.

Each claim is one a wrong implementation would still return plausible numbers for:
  * `AtomTrunk`'s defaults are still the crystal layout, and an absent edge class is class 0;
  * pair classes count bonds apart, ring closures included, and the per-atom inputs count bonds;
  * the energy does not change under a rotation, a translation or a mirror image of the
    positions, nor with what else is in the batch;
  * the gradient with respect to the state, taken through the build, is the energy's finite
    difference, and is zero on a coordinate the molecule does not have;
  * the pre-training target is the trainer's own clipped potential once the stereo lock is
    put back, differs from it by exactly the lock below the clip, and its gradient is its
    finite difference.
"""
import pytest
import torch

from energies.conformer_carrier import carrier_pad_condition
from energies.conformer_data import batch_tree, collate_conditions, condition_from_energy, states_to_positions
from energies.dof_features import free_dof_atom_index
from energies.multi_conformer import MultiConformerTorsions
from models.atom_trunk import AtomTrunk
from models.conformer_trunk import MAX_DEGREE, SPD_KINDS, ConformerTrunk, MoleculeTopology
from pretrain_conformer_trunk import check_potential, potential_and_gradient

# butane, ethanol, cyclopropanol (a ring), (S)-2-butanol (a stereocentre)
SMIS = ['CCCC', 'CCO', 'OC1CC1', 'CC[C@H](C)O']
F64 = torch.float64


@pytest.fixture(scope='module')
def multi():
    m = MultiConformerTorsions(SMIS, identifiers=SMIS, device='cpu', level='full', force_field='mmff',
                               dtype=F64, energy_clip=300.0, stereo_coeff=300.0)
    m.install_clip_floor()
    return m


@pytest.fixture(scope='module')
def cond(multi):
    rows = []
    for ident, m in multi._members.items():
        a, msk = free_dof_atom_index(m)
        rows.append(carrier_pad_condition(condition_from_energy(m, identifier=ident), multi.carrier, ident, m,
                                          atoms=a, mask=msk, R=1))
    return collate_conditions(rows)


@pytest.fixture(scope='module')
def topology(cond):
    return MoleculeTopology(cond, 'cpu')


def _states(cond, slots, scale, seed):
    C = int(cond.num_graphs)
    mask = cond.state_mask.reshape(C, -1)[slots]
    return scale * torch.randn(mask.shape, generator=torch.Generator().manual_seed(seed), dtype=F64) * mask


def _model(seed=0):
    torch.manual_seed(seed)
    return ConformerTrunk(node_dim=24, message_dim=12, num_convs=2, num_radial=8).double().eval()


def test_default_atom_trunk_is_the_crystal_layout():
    trunk = AtomTrunk(node_dim=16, message_dim=8, num_convs=1, num_radial=6).eval()
    assert trunk.convs[0].edge2message.in_features == 6 + 2
    assert trunk.embed.linear.in_features == 16 and trunk.tail_head is not None
    g = torch.Generator().manual_seed(0)
    z = torch.tensor([6, 1, 8, 6, 1])
    graph = torch.tensor([0, 0, 0, 1, 1])
    intra = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    d_intra = 1.0 + torch.rand(6, generator=g)
    ref, img, d_inter = torch.tensor([0, 2, 3]), torch.tensor([2, 0, 4]), 2.0 + torch.rand(3, generator=g)
    density = torch.tensor([1.0, 1.2])
    with torch.no_grad():
        absent = trunk(z, graph, intra, d_intra, ref, img, d_inter, density)
        zero = trunk(z, graph, intra, d_intra, ref, img, d_inter, density,
                     intra_kind=torch.zeros(6, dtype=torch.long))
    assert torch.equal(absent['energy'], zero['energy']) and torch.equal(absent['h'], zero['h'])
    with pytest.raises(ValueError, match='density'):
        trunk(z, graph, intra, d_intra, ref, img, d_inter)


def test_pair_classes_count_bonds_apart_and_inputs_count_bonds(cond, topology):
    ptr = cond.ptr
    for c, smi in enumerate(SMIS):
        n = int(ptr[c + 1] - ptr[c])
        z = cond.z[ptr[c]:ptr[c + 1]]
        carbons = (z == 6).nonzero().flatten()
        kind = topology.kind[c, :n, :n]
        assert torch.equal(kind, kind.T)
        cc = kind[carbons][:, carbons][torch.triu(torch.ones(len(carbons), len(carbons)), 1).bool()]
        counts = [int((cc == s).sum()) for s in range(SPD_KINDS)]
        degree = topology.node_feat[c, :n, :MAX_DEGREE + 1].argmax(1)
        to_h = topology.node_feat[c, :n, MAX_DEGREE + 1:].argmax(1)
        assert int(topology.n[c]) == n
        assert degree[z == 1].tolist() == [1] * int((z == 1).sum()) and (degree[carbons] == 4).all()
        if smi == 'CCCC':
            assert counts == [3, 2, 1, 0, 0]
            assert sorted(to_h[carbons].tolist()) == [2, 2, 3, 3]
        if smi == 'OC1CC1':
            # the three ring carbons are bonded pairwise: the closing bond is a bond
            assert counts == [3, 0, 0, 0, 0]
    # an independent count of bonds apart, from the bond list
    bonds = batch_tree(cond).graph_bond_index
    c = 3
    n = int(ptr[c + 1] - ptr[c])
    far = torch.full((n, n), 99)
    far.fill_diagonal_(0)
    for i, j in (bonds[(bonds[:, 0] >= ptr[c]) & (bonds[:, 0] < ptr[c + 1])] - ptr[c]).tolist():
        far[i, j] = far[j, i] = 1
    for k in range(n):
        far = torch.minimum(far, far[:, k:k + 1] + far[k:k + 1, :])
    want = (far - 1).clamp(max=SPD_KINDS - 1)
    off = ~torch.eye(n, dtype=torch.bool)
    assert torch.equal(topology.kind[c, :n, :n].long()[off], want[off])


def test_energy_ignores_rigid_motion_mirror_image_and_batch_mates(cond, topology):
    model = _model()
    slots = torch.tensor([0, 3, 1, 2, 3, 0])
    sub = cond.subsample_new_batch(slots)
    pos = states_to_positions(sub, _states(cond, slots, 0.2, 1))
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(2), dtype=F64))
    with torch.no_grad():
        e = model(pos, sub, slots, topology)['energy']
        moved = model(pos @ q.T + torch.tensor([1.0, -2.0, 0.5], dtype=F64), sub, slots, topology)['energy']
        mirrored = model(pos * torch.tensor([1.0, 1.0, -1.0], dtype=F64), sub, slots, topology)['energy']
        first = model(pos[:int(sub.ptr[2])], cond.subsample_new_batch(slots[:2]), slots[:2], topology)['energy']
    assert e.std() > 1e-3
    assert torch.allclose(moved, e, atol=1e-10) and torch.allclose(mirrored, e, atol=1e-10)
    assert torch.allclose(first, e[:2], atol=1e-10)
    with pytest.raises(ValueError, match='not the conditions'):
        model(pos, sub, slots.flip(0), topology)


def test_state_gradient_is_the_finite_difference_and_zero_off_the_molecule(cond, topology):
    model = _model(3)
    slots = torch.tensor([2, 0, 3])
    sub = cond.subsample_new_batch(slots)
    x = _states(cond, slots, 0.15, 4)
    mask = cond.state_mask.reshape(int(cond.num_graphs), -1)[slots]
    energy, grad, h = model.energy_and_gradient(x, sub, slots, topology)
    assert h.shape == (int(sub.z.shape[0]), 24) and not grad.requires_grad
    assert (grad[~mask] == 0).all() and grad[mask].abs().max() > 1e-4
    eps = 1e-5
    for row in range(3):
        for col in mask[row].nonzero().flatten()[::7].tolist():
            d = torch.zeros_like(x)
            d[row, col] = eps
            with torch.no_grad():
                up = model(states_to_positions(sub, x + d), sub, slots, topology)['energy'][row]
                dn = model(states_to_positions(sub, x - d), sub, slots, topology)['energy'][row]
            assert abs(float((up - dn) / (2 * eps) - grad[row, col])) < 1e-6 * (1 + abs(float(grad[row, col])))


def test_target_is_the_trainers_potential_less_the_stereo_lock(multi, cond):
    slots = torch.tensor([0, 1, 2, 3] * 16)
    sub = cond.subsample_new_batch(slots)
    near = _states(cond, slots, 0.05, 5)
    worst, rows = check_potential(multi, sub, near, 'test')
    assert rows == 64 and worst < 1e-9
    e, grad, lock = potential_and_gradient(multi, sub, near)
    with_lock = potential_and_gradient(multi, sub, near, need_gradient=False, with_lock=True)[0]
    assert (lock == 0).all() and torch.equal(e, with_lock)

    # far from the reference the lock is on for some rows: it holds every four-neighbour centre
    far = _states(cond, slots, 0.35, 6).clamp(-1, 1)
    check_potential(multi, sub, far, 'test')
    e, grad, lock = potential_and_gradient(multi, sub, far)
    with_lock = potential_and_gradient(multi, sub, far, need_gradient=False, with_lock=True)[0]
    assert (lock > 1e-6).any() and (lock == 0).any()
    cutoff = multi._clip_floor_of_lib.index_select(0, multi._resolve_rows(sub, 64, far)[0]) + multi.energy_clip
    below = with_lock < cutoff
    assert below.any() and torch.allclose((with_lock - e)[below], lock[below], atol=1e-9)
    assert (with_lock >= e - 1e-12).all()

    mask = cond.state_mask.reshape(int(cond.num_graphs), -1)[slots]
    assert (grad[~mask] == 0).all()
    eps = 1e-6
    for row in (0, 1, 2, 3):
        for col in mask[row].nonzero().flatten()[::9].tolist():
            d = torch.zeros_like(far)
            d[row, col] = eps
            up = potential_and_gradient(multi, sub, far + d, need_gradient=False)[0][row]
            dn = potential_and_gradient(multi, sub, far - d, need_gradient=False)[0][row]
            assert abs(float((up - dn) / (2 * eps) - grad[row, col])) < 1e-5 * (1 + abs(float(grad[row, col])))
