"""The intra trunk (models/intra_trunk.py), its fitting script and its data builder. CPU, no checkpoint.

What is pinned:
  * the pair list is every ordered pair of a row's real atoms and no padded one, with distances
    held to [MIN_DISTANCE, cutoff];
  * a molecule's energy is its own (alone, in a batch, under any padding) and does not change
    under a rotation, a translation or a mirror image, while its force turns with it;
  * the force is minus the energy's gradient, to a central difference;
  * the stored units are applied per atom: energy = scale * network + reference of each element;
  * two atoms beyond the radial range still exchange their states;
  * the fitting script's set gathers the rows it was given, its per-element reference recovers
    planted values, and a few steps of its loss lower it;
  * the builder's force is the energy's own gradient (RDKit present), and its check refuses one that is not.
"""
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _root in (_here, os.path.dirname(_here),
              os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from energy_sampling.models.intra_trunk import MIN_DISTANCE, IntraTrunk, molecule_pairs  # noqa: E402

pytestmark = pytest.mark.fast

SIZES = (5, 9, 3, 12)


def _trunk(cutoff=10.0):
    torch.manual_seed(0)
    model = IntraTrunk(node_dim=32, message_dim=16, num_convs=2, num_radial=16, cutoff=cutoff).double()
    e0 = torch.zeros(101, dtype=torch.float64)
    e0[[1, 6, 7, 8]] = torch.tensor([-2.5, 8.0, -4.0, -2.0], dtype=torch.float64)
    model.set_units(e0, 30.0)
    return model


def _molecules(width=None, seed=1, spread=2.5):
    g = torch.Generator().manual_seed(seed)
    width = max(SIZES) if width is None else width
    b = len(SIZES)
    z = torch.zeros(b, width, dtype=torch.long)
    pos = torch.full((b, width, 3), 77.0, dtype=torch.float64)       # padding holds a far-off point
    mask = torch.zeros(b, width, dtype=torch.bool)
    choices = torch.tensor([1, 6, 7, 8])
    for i, n in enumerate(SIZES):
        z[i, :n] = choices[torch.randint(0, 4, (n,), generator=g)]
        pos[i, :n] = spread * torch.randn(n, 3, generator=g, dtype=torch.float64)
        mask[i, :n] = True
    return z, pos, mask


def test_pair_list_is_every_ordered_pair_of_real_atoms():
    z, pos, mask = _molecules()
    pos[0, 0], pos[0, 1] = torch.zeros(3, dtype=torch.float64), torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
    pos[1, 0], pos[1, 1] = torch.zeros(3, dtype=torch.float64), torch.tensor([40.0, 0.0, 0.0], dtype=torch.float64)
    edge_index, dist, node_graph = molecule_pairs(pos, mask, cutoff=10.0)
    assert edge_index.shape[1] == sum(n * (n - 1) for n in SIZES)
    assert node_graph.tolist() == [i for i, n in enumerate(SIZES) for _ in range(n)]
    assert bool((node_graph[edge_index[0]] == node_graph[edge_index[1]]).all())      # no pair crosses molecules
    assert bool((edge_index[0] != edge_index[1]).all())
    assert len({(int(a), int(b)) for a, b in edge_index.T}) == edge_index.shape[1]    # each ordered pair once
    assert float(dist.min()) == pytest.approx(MIN_DISTANCE) and float(dist.max()) == pytest.approx(10.0)
    free = (dist > MIN_DISTANCE + 1e-9) & (dist < 10.0 - 1e-9)
    real = pos[mask]
    assert torch.allclose(dist[free], (real[edge_index[1]] - real[edge_index[0]]).norm(dim=1)[free])


def test_a_molecules_energy_is_its_own_and_ignores_pose_and_handedness():
    model = _trunk()
    z, pos, mask = _molecules()
    energy, force, out = model.energy_and_force(z, pos, mask)
    assert bool(torch.isfinite(energy).all()) and bool((force[~mask] == 0).all())
    for i, n in enumerate(SIZES):
        alone, f_alone, _ = model.energy_and_force(z[i:i + 1, :n], pos[i:i + 1, :n], mask[i:i + 1, :n])
        assert torch.allclose(alone, energy[i:i + 1], atol=1e-9) and torch.allclose(f_alone[0], force[i, :n], atol=1e-9)
    z2, pos2, mask2 = _molecules(width=max(SIZES) + 4)
    assert torch.allclose(model.energy_and_force(z2, pos2, mask2)[0], energy, atol=1e-9)

    g = torch.Generator().manual_seed(3)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.linalg.det(q))                       # a proper rotation
    shift = torch.randn(3, generator=g, dtype=torch.float64)
    e_rot, f_rot, _ = model.energy_and_force(z, pos @ q.T + shift, mask)
    assert torch.allclose(e_rot, energy, atol=1e-9)
    assert torch.allclose(f_rot[mask], (force @ q.T)[mask], atol=1e-8)
    mirror = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))
    e_mir, f_mir, _ = model.energy_and_force(z, pos @ mirror, mask)
    assert torch.allclose(e_mir, energy, atol=1e-9) and torch.allclose(f_mir[mask], (force @ mirror)[mask], atol=1e-8)


def test_force_is_minus_the_energy_gradient():
    model = _trunk()
    z, pos, mask = _molecules()
    _, force, _ = model.energy_and_force(z, pos, mask)
    g = torch.Generator().manual_seed(4)
    direction = torch.randn(pos.shape, generator=g, dtype=torch.float64) * mask[..., None]
    h = 1e-5
    up = model(z, pos + h * direction, mask)['energy']
    down = model(z, pos - h * direction, mask)['energy']
    assert torch.allclose((up - down) / (2 * h), -(force * direction).sum((1, 2)), rtol=1e-5, atol=1e-6)


def test_units_are_applied_per_atom():
    model = _trunk()
    z, pos, mask = _molecules()
    out = model(z, pos, mask)
    zr = z[mask]
    assert torch.allclose(out['e_atom'], 30.0 * out['e_model'] + model.e0[zr])
    total = torch.zeros(len(SIZES), dtype=torch.float64).index_add_(0, out['node_graph'], out['e_atom'])
    assert torch.allclose(out['energy'], total)
    assert out['h'].shape == (int(mask.sum()), 32)


def test_atoms_beyond_the_radial_range_still_exchange_states():
    model = _trunk(cutoff=4.0)
    z = torch.tensor([[6, 8]])
    pos = torch.tensor([[[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]]], dtype=torch.float64)
    mask = torch.ones(1, 2, dtype=torch.bool)
    near = model(z, pos, mask)['e_atom'][0]
    other = model(torch.tensor([[6, 7]]), pos, mask)['e_atom'][0]
    assert abs(float(near - other)) > 1e-8                       # the carbon reads which element sits far off
    moved = model(z, pos * 2.0, mask)['e_atom'][0]
    assert float(near) == pytest.approx(float(moved), abs=1e-12)  # but not how far beyond the range


def _blob(tmp_path, planted):
    """A tiny set in the builder's layout: energy = planted per-element values + a harmonic pair term."""
    g = torch.Generator().manual_seed(7)
    sizes = [4, 6, 3, 5, 7, 4]
    z, pos, force, energy, geom_mol, geom_n, noise, conf = [], [], [], [], [], [], [], []
    for m, n in enumerate(sizes):
        zm = torch.tensor([1, 6, 7, 8])[torch.randint(0, 4, (n,), generator=g)]
        base = 1.5 * torch.randn(n, 3, generator=g, dtype=torch.float64)
        z.append(zm)
        for k, sigma in enumerate((0.0, 0.05, 0.1)):
            x = (base + sigma * torch.randn(n, 3, generator=g, dtype=torch.float64)).requires_grad_(True)
            d = (x[:, None] - x[None]).square().sum(-1).add(torch.eye(n, dtype=torch.float64)).sqrt()
            e = 2.0 * ((d - 2.0).square() * (1 - torch.eye(n, dtype=torch.float64))).sum() * (sigma > 0) \
                + planted[zm].sum()
            f = -torch.autograd.grad(e, x)[0] if sigma > 0 else torch.zeros(n, 3, dtype=torch.float64)
            pos.append(x.detach().float())
            force.append(f.float())
            energy.append(float(e))
            geom_mol.append(m)
            geom_n.append(n)
            noise.append(sigma)
            conf.append(0)

    def ptr(c):
        return torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(c).cumsum(0)))

    blob = {'smiles': [f'm{i}' for i in range(len(sizes))],
            'heldout': torch.tensor([False, False, False, False, True, True]), 'z': torch.cat(z),
            'mol_ptr': ptr(sizes), 'geom_mol': torch.tensor(geom_mol), 'geom_ptr': ptr(geom_n),
            'pos': torch.cat(pos), 'force': torch.cat(force), 'energy': torch.tensor(energy, dtype=torch.float64),
            'noise': torch.tensor(noise), 'conformer': torch.tensor(conf),
            'build': {'force_field': 'planted'}}
    path = tmp_path / 'set.pt'
    torch.save(blob, path)
    return blob, str(path)


def test_the_fitting_scripts_set_reference_and_loss(tmp_path):
    from pretrain_intra_trunk import EnergySet, losses

    planted = torch.zeros(101, dtype=torch.float64)
    planted[[1, 6, 7, 8]] = torch.tensor([-2.0, 9.0, -5.0, 3.0], dtype=torch.float64)
    blob, path = _blob(tmp_path, planted)
    data = EnergySet(path, torch.device('cpu'))
    minima = (data.noise == 0).nonzero().flatten()
    e0 = data.fit_reference(minima)
    assert torch.allclose(e0, planted, atol=1e-6)                # six molecules, four elements: determined
    data.set_reference(e0)
    assert torch.allclose(data.excess[minima], torch.zeros(len(minima)), atol=1e-4)

    g = torch.tensor([4, 0, 17])
    z, pos, force, mask, excess, n = data.batch(g)
    assert n.tolist() == [int(data.geom_n[i]) for i in g] and mask.sum(1).tolist() == n.tolist()
    for row, gi in enumerate(g.tolist()):
        lo, hi = int(blob['geom_ptr'][gi]), int(blob['geom_ptr'][gi + 1])
        m = int(blob['geom_mol'][gi])
        assert torch.equal(pos[row, :hi - lo], blob['pos'][lo:hi]) and torch.equal(force[row, :hi - lo], blob['force'][lo:hi])
        assert torch.equal(z[row, :hi - lo], blob['z'][int(blob['mol_ptr'][m]):int(blob['mol_ptr'][m + 1])])
        assert bool((pos[row, hi - lo:] == 0).all()) and bool((z[row, hi - lo:] == 0).all())

    torch.manual_seed(0)
    model = IntraTrunk(node_dim=32, message_dim=16, num_convs=2, num_radial=16, cutoff=8.0)
    model.set_units(e0.float(), 5.0)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    train = (~data.heldout[data.geom_mol]).nonzero().flatten()
    first = last = None
    for step in range(40):
        loss, f_loss, e_loss, e_pred, f_pred = losses(model, data.batch(train), 5.0, 1.0, 1.0, training=True)
        assert bool(torch.isfinite(loss))
        opt.zero_grad()
        loss.backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
    assert last < 0.7 * first, (first, last)


def test_the_builders_force_is_its_energys_gradient():
    pytest.importorskip('rdkit')
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from build_intra_energy_set import force_is_consistent, label_molecule

    smiles, z, rows, dropped = label_molecule(('CCO', 2, (0.05, 0.1), 500, 50.0))
    assert z.tolist().count(1) == 6 and sorted(set(z.tolist())) == [1, 6, 8]
    assert {r[1] for r in rows} == {0.0, 0.05, 0.1} and sum(dropped.values()) == 0
    minimum = [r for r in rows if r[1] == 0.0][0]
    assert float(np.abs(minimum[4]).max()) < 0.05                    # a relaxed minimum has no force
    assert all(r[3] >= minimum[3] - 1e-6 for r in rows if r[0] == minimum[0])

    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
    ff.Initialize()
    x = np.asarray(ff.Positions(), dtype=np.float64).reshape(-1, 3) + 0.05 * np.random.default_rng(0).standard_normal((9, 3))
    force = -np.asarray(ff.CalcGrad(x.ravel().tolist())).reshape(-1, 3)
    assert force_is_consistent(ff, x, force)
    assert not force_is_consistent(ff, x, 3.0 * force)               # the check has power
    assert not force_is_consistent(ff, x, force + 1e4 * np.eye(9, 3))
    assert math.isfinite(ff.CalcEnergy(x.ravel().tolist()))
