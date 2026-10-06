"""The trunk as a force provider (`models/crystal_force.TrunkForce`), on experimental crystals. CPU.

WHAT IS PINNED.

1. THE PROVIDER'S FORCE IS MINUS THE GRADIENT OF THE TRUNK'S ENERGY, as `pretrain_atom_trunk`
   computes it from a fresh crystal batch: the provider reuses one batch and one set of image
   tables per context and splits the rows into chunks, and none of that may change a number.
2. A CONTEXT IS REUSABLE. A trajectory calls the provider once per state on the same context;
   a later state's force must not depend on an earlier one.
3. IT WORKS WHERE THE SAMPLER CALLS IT: with gradients disabled, and with `create_graph` the
   force carries its graph back to the state.
4. A SAMPLER WITH THE PROVIDER INSTALLED rolls forward, rolls backward and replays to the same
   log-probabilities, on the crystal layout with its wrapped and dead coordinates.
5. A RUN WITH ANOTHER ENERGY IS REFUSED: the trunk's target is compressed in kT of one
   temperature and coefficient.

The crystals are MXtalTools' own fixture (`tests/datasets/mini_new_csd.pt`); the trunk is a
randomly initialised one saved in the pre-training script's checkpoint format.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import mxtaltools
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.dataset_utils.utils import collate_data_list

from models.atom_trunk import AtomTrunk
from models.crystal_force import TrunkForce
from pretrain_atom_trunk import build_example, trunk_energy

pytestmark = pytest.mark.fast

FIXTURE = Path(mxtaltools.__file__).resolve().parents[1] / 'tests' / 'datasets' / 'mini_new_csd.pt'
TRUNK_ARGS = dict(arm='trunk', target='full', node_dim=32, message_dim=16, num_convs=2, unfolded=False,
                  feature_cutoff=5.0, label_cutoff=10.0, temperature=6.9, lj_coeff=1.0, compress_at=5.0)


@pytest.fixture(scope='module')
def crystals():
    if not FIXTURE.exists():
        pytest.skip(f'{FIXTURE} not present')
    rows = [c for c in torch.load(FIXTURE, weights_only=False, map_location='cpu') if int(c.z_prime) == 1]
    rows = [c for c in rows if int(c.num_atoms) <= 40][:7]
    if len(rows) < 6:
        pytest.skip("too few small Z'=1 crystals in the fixture")
    return rows


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    trunk = AtomTrunk(node_dim=32, message_dim=16, num_convs=2, cutoff=5.0)
    path = tmp_path_factory.mktemp('trunk') / 'trunk.pt'
    torch.save({'model': trunk.state_dict(), 'row_scale': torch.ones(12), 'step': 7, 'args': dict(TRUNK_ARGS)}, path)
    return path


def _batch(crystals):
    return collate_data_list([c.clone() for c in crystals])


def _states(batch, seed, spread=0.03):
    """Latents near the stored crystals, as late trajectory states are."""
    x = batch.latent_params(gauge_fix_free_axes=True)
    return x + spread * torch.randn(x.shape, generator=torch.Generator().manual_seed(seed))


def _reference(provider, batch, x):
    """Energy and its gradient the way the pre-training script builds them: a fresh batch, no chunks."""
    a = SimpleNamespace(feature_cutoff=provider.cutoff, label_cutoff=10.0, lj_coeff=1.0, temperature=6.9,
                        compress_at=5.0, target='full', switch_width=1.0)
    ex = build_example(batch, x, a, torch.tensor(list(VDW_RADII.values())), need_geometry_grad=True, labels=False)
    energy = trunk_energy(provider.trunk, ex)['energy']
    return energy.detach(), torch.autograd.grad(energy.sum(), ex['x'])[0]


@pytest.mark.parametrize('chunk', [1, 3, 1000])
def test_force_is_minus_the_trunk_energy_gradient(crystals, checkpoint, chunk):
    batch = _batch(crystals)
    provider = TrunkForce(checkpoint, 'cpu', chunk=chunk)
    ctx = provider.context(batch)
    assert len(ctx.chunks) == -(-len(crystals) // chunk)
    for seed in (1, 2):                      # the second state reuses the context the first one wrote into
        x = _states(batch, seed)
        energy, force = provider.energy_and_force(x, ctx)
        e_ref, g_ref = _reference(provider, batch, x)
        assert torch.allclose(energy, e_ref, atol=1e-4, rtol=1e-5)
        assert torch.allclose(force, -g_ref, atol=1e-3, rtol=1e-4)
        assert force.abs().max() > 1e-3, 'a zero force would pass the comparison and test nothing'
    assert provider.calls == 2 and provider.rows == 2 * len(crystals)
    assert not any(p.requires_grad for p in provider.trunk.parameters())


def test_runs_without_gradients_and_carries_a_graph_on_request(crystals, checkpoint):
    batch = _batch(crystals)
    provider = TrunkForce(checkpoint, 'cpu')
    ctx = provider.context(batch)
    x = _states(batch, 3)
    with torch.no_grad():
        quiet = provider(x, ctx, False)
    assert not quiet.requires_grad and torch.isfinite(quiet).all()

    live = x.clone().requires_grad_(True)
    force = provider(live, ctx, True)
    assert torch.allclose(force.detach(), quiet, atol=1e-4, rtol=1e-5)
    second, = torch.autograd.grad(force.pow(2).sum(), live)        # through the trunk's Hessian
    assert torch.isfinite(second).all() and second.abs().max() > 0


def test_context_does_not_touch_the_callers_batch(crystals, checkpoint):
    batch = _batch(crystals)
    x = _states(batch, 4) + 0.1          # read first: latent_params itself fixes the gauge of its batch
    before = {k: getattr(batch, k).clone() for k in ('pos', 'cell_lengths', 'cell_angles', 'aunit_centroid')}
    provider = TrunkForce(checkpoint, 'cpu')
    provider(x, provider.context(batch), False)
    for k, v in before.items():
        assert torch.equal(getattr(batch, k), v), k


def test_refusals(crystals, checkpoint, tmp_path):
    provider = TrunkForce(checkpoint, 'cpu')
    provider.check_energy(6.9, 1.0)
    with pytest.raises(ValueError, match='temperature'):
        provider.check_energy(5.0, 1.0)
    with pytest.raises(ValueError, match='lj_coeff'):
        provider.check_energy(6.9, 0.5)
    batch = _batch(crystals)
    with pytest.raises(ValueError, match='context'):
        provider(_states(batch, 5), None, False)
    with pytest.raises(ValueError, match='states for a context'):
        provider(_states(batch, 5)[:3], provider.context(batch), False)
    ck = torch.load(checkpoint, weights_only=False)
    ck['args']['target'] = 'short'
    torch.save(ck, tmp_path / 'short.pt')
    with pytest.raises(ValueError, match='full target'):
        TrunkForce(tmp_path / 'short.pt', 'cpu')


class _Builder:
    """Stands in for `MolecularCrystal`: the fixture rows already carry their space group and operators."""

    def __init__(self):
        self.calls = 0

    def init_blank_crystal_batch(self, mol_batch):
        self.calls += 1
        return mol_batch.clone()


def test_sampler_with_the_provider_scores_every_route_alike(crystals, checkpoint):
    from energy_sampling.models.gfn import GFN
    from energy_sampling.models.crystal_force import CrystalDriftForce, TrunkForce as _TrunkForce
    from energy_sampling.utils import uniform_discretizer

    T, B = 6, len(crystals)
    mol_batch = _batch(crystals)
    torch.manual_seed(0)
    g = GFN(dim=12, s_emb_dim=32, conditions_dim=1, harmonics_dim=8, t_dim=8, t_hidden_dim=32, s_hidden_dim=32,
            s_layers=2, policy_hidden_dim=32, policy_layers=2, flow_hidden_dim=16, flow_layers=2,
            learned_variance=True, learn_pb=True, conditional=False, device=torch.device('cpu'),
            t_scale=0.05, max_z_prime=1, do_periodic_angles=True, hold_dead_latent_rows=False,
            force_drift_fwd=0.5, force_drift_bwd=0.2, force_drift_t_min=0.5).eval()
    builder = _Builder()
    trunk = _TrunkForce(checkpoint, 'cpu')
    g.install_drift_force(CrystalDriftForce(trunk, builder))
    disc = lambda b: uniform_discretizer(b, T)

    with torch.no_grad():
        torch.manual_seed(1)
        s, pf, pb, _ = g.get_traj_fwd(torch.zeros(B, 12), disc, None, None, mol_batch)
        assert builder.calls == 1, 'one context per trajectory batch'
        assert trunk.calls == T // 2 + 1, 'one force evaluation per state inside the window'
        _, rpf, rpb, _ = g.get_traj_replay(s, disc, None, mol_batch)
        assert torch.allclose(rpf, pf, atol=1e-4) and torch.allclose(rpb, pb, atol=1e-4)
        torch.manual_seed(2)
        bs, bpf, bpb, _ = g.get_traj_bwd(s[:, -1], disc, None, mol_batch)
        _, rpf2, rpb2, _ = g.get_traj_replay(bs, disc, None, mol_batch)
        assert torch.allclose(rpf2, bpf.flip(1), atol=1e-4) and torch.allclose(rpb2, bpb.flip(1), atol=1e-4)
        assert torch.isfinite(s).all() and torch.isfinite(pf).all() and torch.isfinite(pb).all()
        assert g.force_nonfinite_rows() == 0

        # a context handed in by the caller is used as it is
        calls = builder.calls
        ctx = g.drift_force_fn.context(mol_batch)
        builder.calls = calls
        _, cpf, cpb, _ = g.get_traj_replay(s, disc, None, None, drift_context=ctx)
        assert builder.calls == calls and torch.allclose(cpf, pf, atol=1e-4) and torch.allclose(cpb, pb, atol=1e-4)

        # the term is there: the same weights without it score the late steps differently
        torch.manual_seed(0)
        bare = GFN(dim=12, s_emb_dim=32, conditions_dim=1, harmonics_dim=8, t_dim=8, t_hidden_dim=32,
                   s_hidden_dim=32, s_layers=2, policy_hidden_dim=32, policy_layers=2, flow_hidden_dim=16,
                   flow_layers=2, learned_variance=True, learn_pb=True, conditional=False,
                   device=torch.device('cpu'), t_scale=0.05, max_z_prime=1, do_periodic_angles=True,
                   hold_dead_latent_rows=False).eval()
        bare.load_state_dict(g.state_dict(), strict=False)
        _, npf, npb, _ = bare.get_traj_replay(s, disc, None, mol_batch)
        assert (npf - pf)[:, T // 2:].abs().max() > 1e-3 and (npb - pb)[:, T // 2:].abs().max() > 1e-3
        assert torch.allclose(npf[:, :T // 2], pf[:, :T // 2], atol=1e-5)

    with pytest.raises(ValueError, match='mol_batch=None'):
        g.get_traj_fwd(torch.zeros(B, 12), disc, None, None, None)


def test_target_force_is_the_pretraining_reference_and_agreement_reads_it(crystals, checkpoint):
    """`TrunkForce.target_force` is what the trunk was fitted to (the pre-training script's reference
    force), and `agreement` compares the trunk with it in the sampler's units."""
    batch = _batch(crystals)
    provider = TrunkForce(checkpoint, 'cpu', chunk=3)
    ctx = provider.context(batch)
    x = _states(batch, 6)
    a = SimpleNamespace(feature_cutoff=provider.cutoff, label_cutoff=provider.label_cutoff, lj_coeff=provider.lj_coeff,
                        temperature=provider.temperature, compress_at=provider.compress_at, target='full',
                        switch_width=1.0)
    ref = build_example(batch, x, a, torch.tensor(list(VDW_RADII.values())), False, with_reference=True)['ref_force']
    target, capped = provider.target_force(x, ctx)
    assert torch.allclose(target, -ref, atol=1e-3, rtol=1e-4) and target.abs().max() > 1e-2
    assert not bool(capped.any()), 'a real crystal must sit under the default caps'

    stats = provider.agreement(x, ctx, step_variance=1e-3)
    assert stats['rows'] == len(crystals)
    force = provider(x, ctx, False)
    want = (1e-3 ** 0.5 * (force - target).square().mean(1).sqrt()).median()
    assert abs(stats['err_sigma'] - float(want)) < 1e-6 * max(1.0, float(want))
    assert -1.0 <= stats['cosine'] <= 1.0 and stats['target_sigma'] > 0


def _squeezed(batch):
    """Stored crystals with their three cell-length rows pushed far down: cells no crystal has."""
    x = batch.latent_params(gauge_fix_free_axes=True).clone()
    x[:, :3] = -0.9
    return x


def _same(a, b, rel=1e-4):
    """Two evaluations of one quantity agree to `rel` of each row's own size. On squeezed cells a force
    is of order 1e5 with components near zero, and two summation orders differ in those by more than
    any per-element tolerance."""
    a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    return bool(((a - b).norm(dim=1) <= rel * b.norm(dim=1) + 1e-6).all())


def test_memory_bound_caps_squeezed_cells_and_splits_calls_without_changing_values(crystals, checkpoint):
    """On cells far below any physical density the pair list is capped per crystal and the trunk is
    called on groups of crystals under the pair budget. The split changes no number; the caps are
    counted; and a real crystal is under them, so its force is the uncapped one."""
    batch = _batch(crystals)
    x = _states(batch, 8)
    free = TrunkForce(checkpoint, 'cpu', max_images=10 ** 6, max_pairs=10 ** 8, max_pairs_per_call=10 ** 9)
    default = TrunkForce(checkpoint, 'cpu')
    f_free = free(x, free.context(batch), False)
    # the same pair lists (MXtalTools' tests/test_image_pairs.py holds them entry for entry); two evaluations differ only in rounding
    assert torch.allclose(default(x, default.context(batch), False), f_free, atol=1e-5, rtol=1e-6)
    assert default.capped_rows == 0 and free.capped_rows == 0

    x = _squeezed(batch)
    one = TrunkForce(checkpoint, 'cpu', max_images=60, max_pairs=500, max_pairs_per_call=10 ** 9)
    e1, f1 = one.energy_and_force(x, one.context(batch))
    assert one.capped_rows == len(crystals) and one.extra_calls == 0, (one.capped_rows, one.extra_calls)
    assert torch.isfinite(f1).all() and torch.isfinite(e1).all() and f1.abs().max() > 0

    split = TrunkForce(checkpoint, 'cpu', max_images=60, max_pairs=500, max_pairs_per_call=1200)
    e2, f2 = split.energy_and_force(x, split.context(batch))
    assert split.extra_calls >= 2, 'the pair budget was meant to force several trunk calls'
    assert _same(e2, e1) and _same(f2, f1)
    # chunks and the budget compose
    both = TrunkForce(checkpoint, 'cpu', chunk=3, max_images=60, max_pairs=500, max_pairs_per_call=1200)
    e3, f3 = both.energy_and_force(x, both.context(batch))
    assert _same(e3, e1) and _same(f3, f1)

    # the agreement check leaves capped rows out rather than comparing two truncated lists
    stats = one.agreement(x, one.context(batch), step_variance=1e-3)
    assert stats['rows'] + stats['capped'] == len(crystals) and stats['capped'] > 0

    with pytest.raises(ValueError, match='must fit in a call'):
        TrunkForce(checkpoint, 'cpu', max_pairs=500, max_pairs_per_call=100)


def test_copying_a_sampler_shares_the_provider(crystals, checkpoint):
    import copy
    from energy_sampling.models.crystal_force import CrystalDriftForce
    provider = CrystalDriftForce(TrunkForce(checkpoint, 'cpu'), _Builder())
    assert copy.deepcopy({'p': provider})['p'] is provider
