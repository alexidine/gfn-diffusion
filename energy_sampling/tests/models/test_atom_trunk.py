"""The atom trunk's targets and the trunk itself, on experimental crystals. CPU, no checkpoint.

WHAT IS PINNED.

1. THE TARGET IS THE TRAINER'S eLJ. `pretrain_atom_trunk.build_example` builds its energy from
   `mxtaltools.crystal_building.image_pairs`, regrouped per atom. With the per-atom compression
   switched off and kT = 1 it must equal `MolCrystalData.analyze(['elj'])`, the call
   `MolecularCrystal.analyze_crystal_batch` makes, and its force must be that energy's gradient
   with respect to the latent. A target that drifted from the trainer's energy would pre-train
   the trunk on a different potential without any loss noticing.

2. A CRYSTAL'S OUTPUT IS ITS OWN. No normalisation layer, no batch statistic: the energy of a
   crystal must not change with what else is in the batch. This is the property the molecule
   encoder lacks (its embedding moves with its batch-mates), and a rollout that re-scores a
   stored state depends on it.

3. FORCE MATCHING TRAINS. The force is a gradient of the model's energy, so its loss needs a
   second derivative through MXtalTools' layers and through the image geometry; a few steps on
   one batch must run with finite gradients and lower the loss.

4. THE FOLDED CONVOLUTION IS `MConv`. `FoldedMConv` reuses `MConv`'s parameters and must return
   its output, its gradient and its second derivative, with and without a normalisation layer,
   on a graph with isolated nodes and repeated edges. The trunk's speed rests on that identity.

5. THE SHORT TARGET IS WHAT THE MODEL SEES. Under `target='short'` the fitted energy uses the
   model's own pair list, the closed-form tail pushes on the cell rows only, and the reference
   is still the full-cutoff force.

The crystals are MXtalTools' own fixture (`tests/datasets/mini_new_csd.pt`), read from the
sibling checkout; the module is skipped where that file is absent.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import mxtaltools
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.dataset_utils.utils import collate_data_list

from models.atom_trunk import AtomTrunk, FoldedMConv
from mxtaltools.models.modules.graph_convolution import MConv
from pretrain_atom_trunk import build_example, force_loss

FIXTURE = Path(mxtaltools.__file__).resolve().parents[1] / 'tests' / 'datasets' / 'mini_new_csd.pt'


def _args(**over):
    base = dict(label_cutoff=10.0, feature_cutoff=5.0, lj_coeff=1.0, temperature=1.0, compress_at=1e12,
                target='full', switch_width=1.0)
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture(scope='module')
def crystals():
    if not FIXTURE.exists():
        pytest.skip(f'{FIXTURE} not present')
    rows = [c for c in torch.load(FIXTURE, weights_only=False, map_location='cpu') if int(c.z_prime) == 1]
    rows = [c for c in rows if int(c.num_atoms) <= 40][:8]       # small molecules keep this in seconds
    if len(rows) < 6:
        pytest.skip("too few small Z'=1 crystals in the fixture")
    return rows


@pytest.fixture(scope='module')
def vdw():
    return torch.tensor(list(VDW_RADII.values()))


def _batch(crystals, idx=None):
    rows = crystals if idx is None else [crystals[i] for i in idx]
    return collate_data_list([c.clone() for c in rows])


def test_target_energy_and_force_are_the_trainers_elj(crystals, vdw):
    base = _batch(crystals)
    x0 = base.latent_params(gauge_fix_free_axes=True)
    ex = build_example(base, x0, _args(), vdw, need_geometry_grad=False)

    x = x0.clone().requires_grad_(True)
    cb = base.clone()
    cb.latent_to_cell_params(x)
    elj = cb.analyze(['elj'], cutoff=10.0, supercell_size=10, std_orientation=False)['elj']
    grad, = torch.autograd.grad(elj.sum(), x)

    rel_e = (ex['energy'] - elj.detach()).abs() / elj.detach().abs().clamp(min=1.0)
    assert float(rel_e.max()) < 1e-3, f'target energy differs from analyze([elj]) by up to {float(rel_e.max()):.2e}'
    per_graph = torch.zeros(base.num_graphs).index_add_(0, ex['node_graph'], ex['e_atom'])
    torch.testing.assert_close(per_graph, ex['energy'], rtol=1e-5, atol=1e-4)
    live = grad.norm(dim=1) > 0
    assert int(live.sum()) >= 4
    rel_g = (ex['force'][live] - grad[live]).norm(dim=1) / grad[live].norm(dim=1)
    assert float(rel_g.max()) < 1e-3, f'target force differs from the eLJ gradient by up to {float(rel_g.max()):.2e}'


def test_compression_leaves_bound_atoms_alone_and_caps_the_rest(crystals, vdw):
    base = _batch(crystals)
    x0 = base.latent_params(gauge_fix_free_axes=True)
    raw = build_example(base, x0, _args(), vdw, False)['e_atom']
    squeezed = build_example(base, x0, _args(compress_at=0.5), vdw, False)['e_atom']
    low = raw <= 0.5
    assert bool(low.any())
    torch.testing.assert_close(squeezed[low], raw[low])
    if bool((~low).any()):
        assert bool((squeezed[~low] < raw[~low]).all()) and bool((squeezed[~low] > 0.5).all())


def test_model_inputs_lie_inside_the_feature_cutoff(crystals, vdw):
    base = _batch(crystals)
    ex = build_example(base, base.latent_params(gauge_fix_free_axes=True), _args(), vdw, False)
    assert ex['inter_dist'].numel() > 0 and float(ex['inter_dist'].max()) <= 5.0 + 1e-4
    assert float(ex['intra_dist'].max()) <= 5.0 + 1e-4 and float(ex['intra_dist'].min()) > 0
    # an image atom is addressed by the atom it copies, in the same crystal as its reference atom
    assert torch.equal(ex['node_graph'][ex['inter_ref']], ex['node_graph'][ex['inter_img']])
    src, tgt = ex['intra_index']
    assert torch.equal(ex['node_graph'][src], ex['node_graph'][tgt]) and bool((src != tgt).all())


def _energy(model, ex):
    return model(ex['z'], ex['node_graph'], ex['intra_index'], ex['intra_dist'],
                 ex['inter_ref'], ex['inter_img'], ex['inter_dist'], ex['density'])


def test_a_crystals_energy_does_not_depend_on_its_batch_mates(crystals, vdw):
    model = AtomTrunk(node_dim=32, message_dim=16, num_convs=2, num_radial=8).eval()
    a = _args(temperature=6.9, compress_at=5.0)
    full = _batch(crystals)
    part = _batch(crystals, [0, 1])
    # build_example differentiates its own target, so it runs with gradients on; only the model call is frozen
    ex_full = build_example(full, full.latent_params(gauge_fix_free_axes=True), a, vdw, False)
    ex_part = build_example(part, part.latent_params(gauge_fix_free_axes=True), a, vdw, False)
    with torch.no_grad():
        e_full, e_part = _energy(model, ex_full), _energy(model, ex_part)
    torch.testing.assert_close(e_full['energy'][:2], e_part['energy'], rtol=1e-4, atol=1e-4)
    n = int(part.num_atoms.sum())
    torch.testing.assert_close(e_full['e_atom'][:n], e_part['e_atom'], rtol=1e-4, atol=1e-4)


def test_force_matching_runs_through_a_second_derivative_and_lowers_the_loss(crystals, vdw):
    model = AtomTrunk(node_dim=32, message_dim=16, num_convs=2, num_radial=8)
    torch.manual_seed(0)
    a = _args(temperature=6.9, compress_at=5.0)
    base = _batch(crystals)
    x0 = base.latent_params(gauge_fix_free_axes=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    scale = None
    losses = []
    for _ in range(12):
        ex = build_example(base, x0, a, vdw, need_geometry_grad=True)
        out = _energy(model, ex)
        force, = torch.autograd.grad(out['energy'].sum(), ex['x'], create_graph=True)
        if scale is None:
            scale = ex['force'].abs().median(0).values.clamp(min=1e-3)
        loss = force_loss(force, ex['force'], scale).mean() \
            + torch.nn.functional.huber_loss(out['e_atom'], ex['e_atom'], delta=1.0)
        opt.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(float(g.abs().max()) > 0 for g in grads)
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0], f'loss did not fall over 12 steps on one batch: {losses[0]:.3f} -> {losses[-1]:.3f}'


@pytest.mark.parametrize('norm', [None, 'layer'])
def test_folded_convolution_equals_mconv_to_second_order(norm):
    torch.manual_seed(0)
    n, e, node_dim, msg_dim, edge_dim = 40, 600, 24, 12, 10
    ref = MConv(message_dim=msg_dim, node_dim=node_dim, edge_embedding_dim=edge_dim, norm=norm).double()
    new = FoldedMConv(message_dim=msg_dim, node_dim=node_dim, edge_embedding_dim=edge_dim, norm=norm).double()
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.3 * torch.randn_like(p))         # away from the constant-filled defaults (t = 1, b = 0.1)
    new.load_state_dict(ref.state_dict())             # the same keys and shapes: the fold adds no parameter
    edge_index = torch.randint(0, n - 5, (2, e))       # the last five nodes receive and send nothing
    edge_index = torch.cat((edge_index, edge_index[:, :50]), dim=1)        # repeated edges
    edge_attr = torch.randn(edge_index.shape[1], edge_dim, dtype=torch.double, requires_grad=True)
    x = torch.randn(n, node_dim, dtype=torch.double, requires_grad=True)

    outs = []
    for conv in (ref, new):
        y = conv(x, edge_index, edge_attr)
        g, = torch.autograd.grad(y.square().sum(), edge_attr, create_graph=True)
        params = list(conv.parameters())
        gg = torch.autograd.grad(g.square().sum(), params, allow_unused=True)
        outs.append((y, g, [None if t is None else t.clone() for t in gg]))
    (y0, g0, gg0), (y1, g1, gg1) = outs
    torch.testing.assert_close(y1, y0, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(g1, g0, rtol=1e-9, atol=1e-10)
    assert torch.equal(y1[-5:], x[-5:].detach()) or torch.allclose(y1[-5:], x[-5:])   # isolated nodes: residual only
    compared = 0
    for a, b in zip(gg0, gg1):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(b, a, rtol=1e-8, atol=1e-9)
            compared += 1
    assert compared >= 6, 'the second-derivative check reached too few parameters to mean anything'


def test_short_target_is_the_models_own_pairs_and_the_tail_moves_the_cell_only(crystals, vdw):
    base = _batch(crystals)
    x0 = base.latent_params(gauge_fix_free_axes=True)
    a_full, a_short = _args(temperature=6.9, compress_at=5.0), _args(temperature=6.9, compress_at=5.0, target='short')
    full = build_example(base, x0, a_full, vdw, False, with_reference=True)
    short = build_example(base, x0, a_short, vdw, False, with_reference=True)
    assert short['pairs_label'] == short['pairs_feature'] < full['pairs_label']
    torch.testing.assert_close(short['ref_force'], full['force'], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(full['ref_force'], full['force'])
    assert float(full['tail_force'].abs().max()) == 0.0
    tail = short['tail_force']
    assert float(tail[:, 6:].abs().max()) == 0.0, 'the volume-only tail pushed on a centroid or orientation row'
    assert float(tail[:, :6].abs().max()) > 0.0
    # with the tail, the short target is closer to the reference on the cell rows than without it
    err_with = (short['force'] + tail - full['force'])[:, :6].norm(dim=1)
    err_without = (short['force'] - full['force'])[:, :6].norm(dim=1)
    assert float(err_with.median()) < float(err_without.median())
