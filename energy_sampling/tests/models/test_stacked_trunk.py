"""The crystal trunk stacked on the intra trunk (models/stacked_trunk.py), on experimental crystals. CPU.

What is pinned:
  * the fitted stages read the intra trunk's states: another intra trunk gives another energy,
    and the stages hold no intramolecular edge;
  * a crystal's energy is its own, whatever else is in the batch;
  * force matching through the stack runs with finite gradients and lowers the loss, and the
    intra trunk's parameters neither receive a gradient nor move;
  * `calibrate` sets the stored scale so the states read have unit root mean square.

The crystals are MXtalTools' own fixture (`tests/datasets/mini_new_csd.pt`), read from the
sibling checkout; the module is skipped where that file is absent.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import mxtaltools
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_building.image_pairs import build_image_tables
from mxtaltools.dataset_utils.utils import collate_data_list

from models.stacked_trunk import StackedTrunk
from pretrain_atom_trunk import build_example, force_loss, trunk_energy

FIXTURE = Path(mxtaltools.__file__).resolve().parents[1] / 'tests' / 'datasets' / 'mini_new_csd.pt'
INTRA_ARGS = dict(node_dim=24, message_dim=12, num_convs=2, num_radial=12, cutoff=8.0)


def _args():
    return SimpleNamespace(label_cutoff=10.0, feature_cutoff=5.0, lj_coeff=1.0, temperature=1.0, compress_at=5.0,
                           target='full', switch_width=1.0)


@pytest.fixture(scope='module')
def crystals():
    if not FIXTURE.exists():
        pytest.skip(f'{FIXTURE} not present')
    rows = [c for c in torch.load(FIXTURE, weights_only=False, map_location='cpu') if int(c.z_prime) == 1]
    rows = [c for c in rows if int(c.num_atoms) <= 40][:8]
    if len(rows) < 6:
        pytest.skip("too few small Z'=1 crystals in the fixture")
    return rows


@pytest.fixture(scope='module')
def vdw():
    return torch.tensor(list(VDW_RADII.values()))


def _model(seed=0):
    torch.manual_seed(seed)
    return StackedTrunk(INTRA_ARGS, node_dim=32, message_dim=16, num_convs=2, cutoff=5.0)


def _example(crystals, vdw, idx=None, grad=False):
    batch = collate_data_list([crystals[i].clone() for i in (range(len(crystals)) if idx is None else idx)])
    return build_example(batch, batch.latent_params(gauge_fix_free_axes=True), _args(), vdw, need_geometry_grad=grad)


def test_the_stages_read_the_intra_states_and_hold_no_intramolecular_edge(crystals, vdw):
    model = _model()
    ex = _example(crystals, vdw)
    tables_states = model.molecule_states(ex['mol_z'], ex['mol_pos'], ex['mol_mask'])
    assert tables_states.shape == (int(ex['mol_mask'].sum()), INTRA_ARGS['node_dim'])
    base = trunk_energy(model, ex)['energy']
    assert bool(torch.isfinite(base).all())
    other = _model()
    torch.manual_seed(9)
    with torch.no_grad():
        for p in other.intra.parameters():
            p.add_(0.05 * torch.randn_like(p))
    assert float((trunk_energy(other, ex)['energy'] - base).abs().max()) > 1e-6

    seen = {}
    hook = model.inter.convs[0].register_forward_hook(lambda mod, inp, out: seen.update(edges=inp[1].shape[1]))
    trunk_energy(model, ex)
    hook.remove()
    assert seen['edges'] == ex['inter_dist'].numel()             # intermolecular pairs and nothing else


def test_a_crystals_energy_does_not_depend_on_its_batch_mates(crystals, vdw):
    model = _model().double()
    whole = trunk_energy(model, _to_double(_example(crystals, vdw)))['energy']
    for i in (0, 3, len(crystals) - 1):
        alone = trunk_energy(model, _to_double(_example(crystals, vdw, idx=[i])))['energy']
        assert torch.allclose(alone, whole[i:i + 1], atol=1e-8), i


def _to_double(ex):
    return {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in ex.items()}


def test_force_matching_trains_the_stages_and_leaves_the_intra_trunk_alone(crystals, vdw):
    model = _model()
    batch = collate_data_list([c.clone() for c in crystals])
    tables = build_image_tables(batch)
    model.calibrate(tables.z, tables.p, tables.amask)
    states = model.molecule_states(tables.z, tables.p, tables.amask)
    assert float(states.square().mean().sqrt()) == pytest.approx(1.0, rel=1e-4)

    before = {k: v.clone() for k, v in model.intra.state_dict().items()}
    x = batch.latent_params(gauge_fix_free_axes=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-3)
    scale = None
    first = last = None
    for step in range(25):
        ex = build_example(batch, x, _args(), vdw, need_geometry_grad=True)
        if scale is None:
            scale = ex['force'].abs().median(0).values.clamp(min=1e-3)
        out = trunk_energy(model, ex)
        force, = torch.autograd.grad(out['energy'].sum(), ex['x'], create_graph=True)
        loss = force_loss(force, ex['force'], scale).mean()
        opt.zero_grad()
        loss.backward()
        assert all(p.grad is None for p in model.intra.parameters())
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.inter.parameters() if p.grad is not None)
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
    assert last < first, (first, last)
    after = model.intra.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not model.intra.training
    assert model.train().inter.training and not model.intra.training     # held in evaluation mode under .train()


def test_trunk_conditions_are_the_pooled_intra_states_of_each_rows_molecule(crystals):
    from build_trunk_conditions import SUM_SCALE, molecule_conditions

    model = _model()
    batch = collate_data_list([c.clone() for c in crystals])
    width = INTRA_ARGS['node_dim']
    emb = molecule_conditions(model, batch, 'cpu', chunk=3)
    assert emb.shape == (len(crystals), 2 * width) and bool(torch.isfinite(emb).all())
    # the image tables hold each molecule in another frame: the states, and so the conditions, do not care
    tables = build_image_tables(batch)
    states = model.molecule_states(tables.z, tables.p, tables.amask)
    graph = tables.amask.nonzero(as_tuple=True)[0]
    total = torch.zeros(len(crystals), width).index_add_(0, graph, states)
    assert torch.allclose(emb[:, :width], total / tables.nat[:, None], atol=1e-4)
    assert torch.allclose(emb[:, width:], total / SUM_SCALE, atol=1e-4)
    alone = molecule_conditions(model, collate_data_list([crystals[2].clone()]), 'cpu')
    assert torch.allclose(alone, emb[2:3], atol=1e-5)
