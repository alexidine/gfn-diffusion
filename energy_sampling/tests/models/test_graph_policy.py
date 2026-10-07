"""The sampler reading atoms: the provider's per-state atom features and the GFN that encodes them.

On experimental crystals (MXtalTools' `tests/datasets/mini_new_csd.pt`), CPU, with randomly
initialised trunks saved in the pre-training scripts' checkpoint formats. What is pinned:

  * a stacked checkpoint loads as a force provider, and its force is the stacked trunk's energy
    gradient however the rows are chunked;
  * `state_info` returns that force and, per atom, exactly what the layout says: the trunk's node
    state, the intra trunk's state, the fractional position (its heavy-atom mean is the state's
    fractional centroid), the offset from the centre, the energy, the element and the flag, zero
    on padding; the same with or without the force, under any chunking and any call budget;
  * a sampler built with `state_atoms` scores a trajectory the same rolled forward, rolled
    backward or replayed, with a force term and without, and with P_B frozen;
  * the policies read the features (another trunk gives other log-probabilities), a loss reaches
    the state encoder's atom layers, and the trunk receives no gradient.
"""
from pathlib import Path

import pytest
import torch

import mxtaltools
from mxtaltools.dataset_utils.utils import collate_data_list

FIXTURE = Path(mxtaltools.__file__).resolve().parents[1] / 'tests' / 'datasets' / 'mini_new_csd.pt'
INTRA_ARGS = dict(node_dim=24, message_dim=12, num_convs=2, num_radial=12, cutoff=8.0)
TRUNK_ARGS = dict(arm='trunk', target='full', node_dim=32, message_dim=16, num_convs=2, unfolded=False,
                  feature_cutoff=5.0, label_cutoff=10.0, temperature=6.9, lj_coeff=1.0, compress_at=5.0)
ATOM_DIM = 32 + 24 + 7


@pytest.fixture(scope='module')
def crystals():
    if not FIXTURE.exists():
        pytest.skip(f'{FIXTURE} not present')
    rows = [c for c in torch.load(FIXTURE, weights_only=False, map_location='cpu') if int(c.z_prime) == 1]
    rows = [c for c in rows if int(c.num_atoms) <= 40][:6]
    if len(rows) < 6:
        pytest.skip("too few small Z'=1 crystals in the fixture")
    return rows


def _stacked_checkpoint(folder, seed):
    from energy_sampling.models.stacked_trunk import StackedTrunk

    torch.manual_seed(seed)
    trunk = StackedTrunk(INTRA_ARGS, node_dim=32, message_dim=16, num_convs=2, cutoff=5.0)
    trunk.feature_scale.fill_(0.7)
    path = folder / f'stacked_{seed}.pt'
    torch.save({'model': trunk.state_dict(), 'row_scale': torch.ones(12), 'step': 3, 'args': dict(TRUNK_ARGS),
                'intra_trunk_args': dict(INTRA_ARGS)}, path)
    return path


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    return _stacked_checkpoint(tmp_path_factory.mktemp('stacked'), 0)


def _batch(crystals):
    return collate_data_list([c.clone() for c in crystals])


def _states(batch, seed, spread=0.03):
    x = batch.latent_params(gauge_fix_free_axes=True)
    return x + spread * torch.randn(x.shape, generator=torch.Generator().manual_seed(seed))


class _Builder:
    """Stands in for `MolecularCrystal`: the fixture rows already carry their space group and operators."""

    def init_blank_crystal_batch(self, mol_batch):
        return mol_batch.clone()


def test_stacked_checkpoint_is_a_force_provider_and_describes_a_state(crystals, checkpoint):
    from energy_sampling.models.crystal_force import COORD_SCALE, ENERGY_CLAMP, ENERGY_SCALE, TrunkForce

    batch = _batch(crystals)
    x = _states(batch, 1)
    widest = max(int(c.num_atoms) for c in crystals)
    A = widest + 3
    whole = TrunkForce(checkpoint, 'cpu', max_atoms=A)
    assert whole.stacked and whole.atom_dim == ATOM_DIM and whole.features_dim == A * (ATOM_DIM + 2)
    assert TrunkForce.atom_feature_dim(checkpoint) == ATOM_DIM
    ctx = whole.context(batch)
    force, feats = whole.state_info(x, ctx, need_force=True)
    assert torch.allclose(force, whole(x, ctx), atol=1e-5)
    assert not feats.requires_grad and feats.shape == (len(crystals), A * (ATOM_DIM + 2))
    none, feats_only = whole.state_info(x, ctx, need_force=False)
    assert none is None and torch.allclose(feats_only, feats, atol=1e-5)

    per_atom = feats.reshape(len(crystals), A, ATOM_DIM + 2)
    chunk = ctx.chunks[0]
    am = chunk.tables.amask
    mask = per_atom[..., -1] > 0.5
    assert torch.equal(mask[:, :am.shape[1]], am) and not bool(mask[:, am.shape[1]:].any())
    assert bool((per_atom[~mask] == 0).all())
    assert torch.equal(per_atom[..., -2][mask].long(), chunk.z)
    # the intra trunk's states, as the trunk itself reads them
    assert torch.allclose(per_atom[..., 32:56][mask], chunk.e_m, atol=1e-6)
    assert torch.allclose(chunk.e_m, whole.trunk.molecule_states(chunk.tables.z, chunk.tables.p, am), atol=1e-6)
    # the heavy atoms' mean fractional position is the fractional centroid the state names
    cb = ctx.crystals
    cb.latent_to_cell_params(x)
    heavy = mask & (per_atom[..., -2] > 1.5)
    frac = per_atom[..., 56:59]
    mean = (frac * heavy[..., None]).sum(1) / heavy.sum(1, keepdim=True)
    assert torch.allclose(mean, cb.aunit_centroid[:, :3], atol=1e-4)
    # offsets are a rigid copy of the molecule: pair distances among a row's atoms are the molecule's own
    off = per_atom[..., 59:62] * COORD_SCALE
    def pair_distances(q):
        return (q[:, None] - q[None]).norm(dim=-1)

    for b in range(len(crystals)):
        n = int(am[b].sum())
        assert torch.allclose(pair_distances(off[b, :n]), pair_distances(chunk.tables.p[b, :n]), atol=1e-3)
    energy = per_atom[..., 62][mask] * ENERGY_SCALE
    assert float(energy.abs().max()) <= ENERGY_CLAMP * ENERGY_SCALE + 1e-6 and float(energy.abs().sum()) > 0

    # chunking and the call budget change nothing
    for kwargs in (dict(chunk=2), dict(chunk=1000, max_pairs=20_000, max_pairs_per_call=20_000)):
        other = TrunkForce(checkpoint, 'cpu', max_atoms=A, **kwargs)
        f2, feats2 = other.state_info(x, other.context(batch), need_force=True)
        scale = feats.abs().amax(dim=1, keepdim=True)
        assert torch.allclose(feats2 / scale, feats / scale, atol=1e-4), kwargs
        assert torch.allclose(f2, force, rtol=1e-3, atol=1e-3 * float(force.abs().max())), kwargs

    with pytest.raises(ValueError, match='sized for'):
        TrunkForce(checkpoint, 'cpu', max_atoms=widest - 1).context(batch)
    with pytest.raises(ValueError, match='without state features'):
        TrunkForce(checkpoint, 'cpu').state_info(x, ctx, need_force=False)


def _sampler(features_dim, atoms, seed=0, **over):
    from energy_sampling.models.gfn import GFN

    torch.manual_seed(seed)
    kwargs = dict(dim=12, s_emb_dim=32, conditions_dim=1, harmonics_dim=8, t_dim=8, t_hidden_dim=32, s_hidden_dim=32,
                  s_layers=2, policy_hidden_dim=32, policy_layers=2, flow_hidden_dim=16, flow_layers=2,
                  learned_variance=True, learn_pb=True, conditional=False, device=torch.device('cpu'),
                  t_scale=0.05, max_z_prime=1, do_periodic_angles=True, hold_dead_latent_rows=False,
                  state_features_dim=features_dim, state_atoms=atoms, state_atom_hidden_dim=32,
                  state_atom_blocks=1, state_atom_heads=2)
    kwargs.update(over)
    return GFN(**kwargs).eval()


@pytest.mark.parametrize('force', [False, True])
def test_sampler_reading_atoms_scores_every_route_alike(crystals, checkpoint, tmp_path, force):
    from energy_sampling.models.crystal_force import CrystalDriftForce, TrunkForce
    from energy_sampling.models.graph_state import FlatAtomStateEncoding
    from energy_sampling.utils import uniform_discretizer

    T, B = 6, len(crystals)
    mol_batch = _batch(crystals)
    A = max(int(c.num_atoms) for c in crystals)
    over = dict(force_drift_fwd=0.5, force_drift_t_min=0.5) if force else {}
    trunk = TrunkForce(checkpoint, 'cpu', max_atoms=A)
    g = _sampler(trunk.features_dim, A, **over)
    assert isinstance(g.s_model, FlatAtomStateEncoding) and g.features_on
    g.install_drift_force(CrystalDriftForce(trunk, _Builder()))
    disc = lambda b: uniform_discretizer(b, T)

    with torch.no_grad():
        torch.manual_seed(1)
        s, pf, pb, _ = g.get_traj_fwd(torch.zeros(B, 12), disc, None, None, mol_batch)
        assert trunk.calls == T + 1, 'one trunk pass per state of the trajectory'
        _, rpf, rpb, _ = g.get_traj_replay(s, disc, None, mol_batch)
        assert torch.allclose(rpf, pf, atol=1e-4) and torch.allclose(rpb, pb, atol=1e-4)
        torch.manual_seed(2)
        bs, bpf, bpb, _ = g.get_traj_bwd(s[:, -1], disc, None, mol_batch)
        _, rpf2, rpb2, _ = g.get_traj_replay(bs, disc, None, mol_batch)
        assert torch.allclose(rpf2, bpf.flip(1), atol=1e-4) and torch.allclose(rpb2, bpb.flip(1), atol=1e-4)
        assert torch.isfinite(s).all() and torch.isfinite(pf).all() and torch.isfinite(pb).all()
        assert g.force_nonfinite_rows() == 0

        # P_B frozen: the snapshot holds the state encoder, atom layers included, and scores as before
        g.freeze_backward_policy()
        _, fpf, fpb, _ = g.get_traj_replay(s, disc, None, mol_batch)
        assert torch.allclose(fpf, pf, atol=1e-4) and torch.allclose(fpb, pb, atol=1e-4)
        g.unfreeze_backward_policy()

        # the policies read the features: the same sampler on another trunk scores the trajectory differently
        other = TrunkForce(_stacked_checkpoint(tmp_path, 5), 'cpu', max_atoms=A)
        g.install_drift_force(CrystalDriftForce(other, _Builder()))
        _, opf, opb, _ = g.get_traj_replay(s, disc, None, mol_batch)
        assert float((opf - pf).abs().max()) > 1e-3 and float((opb - pb).abs().max()) > 1e-3
        g.install_drift_force(CrystalDriftForce(trunk, _Builder()))

    # a loss on the forward log-probabilities trains the atom layers and leaves the trunk alone
    g.train()
    _, tpf, tpb, _ = g.get_traj_replay(s, disc, None, mol_batch)
    (tpf.sum() + tpb.sum()).backward()
    atom_grads = [p.grad for n, p in g.s_model.named_parameters() if 'pool' in n]
    assert atom_grads and all(q is not None and bool(torch.isfinite(q).all()) for q in atom_grads)
    assert sum(float(q.abs().sum()) for q in atom_grads) > 0
    assert all(p.grad is None for p in trunk.trunk.parameters())


def test_state_atoms_needs_a_matching_feature_width():
    with pytest.raises(ValueError, match='state_atoms'):
        _sampler(100, 7)
    with pytest.raises(ValueError, match='state_atoms'):
        _sampler(0, 7)


def test_nonfinite_features_are_zeroed_in_place_and_the_scramble_is_refused(crystals, checkpoint):
    from energy_sampling.models.crystal_force import CrystalDriftForce, TrunkForce
    from energy_sampling.utils import uniform_discretizer

    T, B = 4, len(crystals)
    mol_batch = _batch(crystals)
    A = max(int(c.num_atoms) for c in crystals)
    trunk = TrunkForce(checkpoint, 'cpu', max_atoms=A)

    class Poisoned(CrystalDriftForce):
        """The provider with one row's first atom features not finite, as a collapsed cell gives them."""

        def state_info(self, state, ctx, need_force, create_graph=False):
            force, feats = super().state_info(state, ctx, need_force, create_graph)
            feats = feats.clone()
            feats[1, :4] = float('nan')
            feats[1, 4] = float('inf')
            return force, feats

    g = _sampler(trunk.features_dim, A)
    g.install_drift_force(Poisoned(trunk, _Builder()))
    disc = lambda b: uniform_discretizer(b, T)
    with torch.no_grad():
        torch.manual_seed(1)
        s, pf, pb, _ = g.get_traj_fwd(torch.zeros(B, 12), disc, None, None, mol_batch)
    assert torch.isfinite(s).all() and torch.isfinite(pf).all() and torch.isfinite(pb).all()
    assert g.force_nonfinite_rows() == T + 1                    # the one row, at every state
    # the scramble (a conditional model's stage flag) cannot hide a molecule the features carry
    assert g._maybe_scramble_condition_embedding(None, B, 0) is None
    with pytest.raises(ValueError, match='scramble_conditions'):
        g._maybe_scramble_condition_embedding(torch.zeros(B, 4), B, 2)


def test_a_checkpoint_of_another_state_encoder_is_refused():
    from types import SimpleNamespace

    from energy_sampling.checkpointing import Checkpointer

    def config_for(asked, held):
        fake = SimpleNamespace(modeller=SimpleNamespace(args=SimpleNamespace(model=SimpleNamespace(state_atoms=asked))),
                               RECONFIGURABLE_GFN_KEYS=(), FORCE_DRIFT_GFN_KEYS=(),
                               _assert_dead_rows_match=lambda config: None)
        stored = {'dim': 12} if held is None else {'dim': 12, 'state_atoms': held}
        return Checkpointer._gfn_config_from(fake, {'gfn_config': stored})

    assert config_for(0, None) == {'dim': 12}                   # a checkpoint from before the key
    assert config_for(29, 29)['state_atoms'] == 29
    for asked, held in ((29, None), (29, 0), (0, 29), (20, 29)):
        with pytest.raises(ValueError, match='state_atoms'):
            config_for(asked, held)
