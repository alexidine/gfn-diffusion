"""`model.backward_policy_kind: set` -- P_B reads the molecule the way P_F does.

WHAT WAS WRONG. Under `policy_kind: set` only the FORWARD policy was replaced by the
per-coordinate set head. The backward policy stayed the base model's flat network over the
padded state encoding, for which a coordinate is nothing but its column index on the carrier:
no element, no coordinate kind, no atom embedding. Swapping two bond columns of a row together
with their features permutes P_F's output exactly and changed P_B's.

WHAT IS PINNED HERE, through the real build, freeze and checkpoint paths (helpers from
test_set_policy_resume): the set backward head is built, stamped, trained by the optimizer,
snapshotted by the freeze and rebuilt by a reload; its output follows a coordinate, not a
column; and a flat checkpoint is not silently loaded as a set one or the reverse.
"""
import pytest
import torch

from models.conformer_gfn import ConformerGFN
from models.ragged_set_policy import RaggedConditionalSetPolicy
from test_set_policy_resume import (KW, SMIS, _args, _condition_batch, _disc, _flat, _load,
                                    _modeller, _perturb, _rollouts, _synthetic_step)
from energies.multi_conformer import MultiConformerTorsions

pytestmark = pytest.mark.fast


@pytest.fixture(scope='module', autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture(scope='module')
def carrier():
    return MultiConformerTorsions(SMIS, identifiers=SMIS, **KW)


def _build(carrier, ckdir, kind='set', **kw):
    m = _modeller(carrier, _args(ckdir, model__backward_policy_kind=kind, **kw))
    m.init_gfn()
    _perturb(m.gfn_model, 1, 0.05)          # off the init, so outputs are not trivially 0
    return m


def _pb_out(model, batch, state, t=0.5):
    """(dmean, dvar) of the backward network on `state`, with the batch bound."""
    n = batch.num_graphs
    model.bind_molecular_conditioning(batch)
    return model._pb_net(model.expand_state_for_policy(state), _cond(model, batch),
                         torch.full((n,), t, dtype=state.dtype))


def _cond(model, batch):
    """The condition embedding the flat backward policy's state encoder reads."""
    n = batch.num_graphs
    return model.get_condition_embedding(batch.embedding.reshape(n, -1), batch)


def _state(model, batch, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = 0.6 * (2 * torch.rand(batch.num_graphs, model.dim, generator=g) - 1)
    return x * batch.state_mask.reshape(batch.num_graphs, -1).to(x.dtype)


# ------------------------------------------------------------------ construction


def test_default_is_the_flat_backward_policy(carrier, tmp_path):
    m = _modeller(carrier, _args(tmp_path))
    m.init_gfn()
    assert not isinstance(m.gfn_model.backward_policy, RaggedConditionalSetPolicy)
    assert m.gfn_config['conformer']['backward_policy_kind'] == 'flat'
    assert not m.gfn_model._pb_wants_molecule


def test_set_builds_a_second_head_with_its_own_weights(carrier, tmp_path):
    m = _build(carrier, tmp_path)
    for model in (m.gfn_model, m.ema_model):
        assert type(model) is ConformerGFN
        assert isinstance(model.backward_policy, RaggedConditionalSetPolicy)
        assert model._pb_wants_molecule
    fwd, bwd = m.gfn_model.forward_policy, m.gfn_model.backward_policy
    assert bwd is not fwd
    assert [tuple(p.shape) for p in bwd.parameters()] == [tuple(p.shape) for p in fwd.parameters()]
    held = {p.data_ptr() for p in fwd.parameters()}
    assert not any(p.data_ptr() in held for p in bwd.parameters())
    assert m.gfn_config['conformer']['backward_policy_kind'] == 'set'


def test_an_unknown_kind_is_refused(carrier, tmp_path):
    m = _modeller(carrier, _args(tmp_path, model__backward_policy_kind='ragged'))
    with pytest.raises(ValueError, match='backward_policy_kind'):
        m.init_gfn()


def test_the_optimizer_trains_the_set_backward_head(carrier, tmp_path):
    m = _build(carrier, tmp_path)
    before = [p.detach().clone() for p in m.gfn_model.backward_policy.parameters()]
    _synthetic_step(m, 5)
    after = list(m.gfn_model.backward_policy.parameters())
    assert all(not torch.equal(a, b) for a, b in zip(before, after))


# ------------------------------------------------------------------ what it computes


def test_raw_state_is_recovered_from_the_expansion(carrier, tmp_path):
    m = _build(carrier, tmp_path)
    model = m.gfn_model
    g = torch.Generator().manual_seed(3)
    x = 2 * torch.rand(16, model.dim, generator=g) - 1
    x[:, model.lin_idx] *= 1.4                           # non-periodic columns leave the box
    x[0, model.ang_idx] = 1.0                            # the seam
    back = model._raw_from_expanded(model.expand_state_for_policy(x))
    assert torch.allclose(back, x, rtol=0, atol=1e-12)


def test_output_follows_the_coordinate_not_the_column(carrier, tmp_path):
    """Swap two bond columns of one row with everything the policy is told about them: the
    set backward head's two outputs swap with them. The flat one's change."""
    batch = _condition_batch(carrier, n_rep=1)
    row = 1                                              # 'CO': two bond columns to swap
    mask = batch.state_mask.reshape(batch.num_graphs, -1)
    j, k = [int(c) for c in torch.nonzero(mask[row])[:2].flatten()]

    def swapped(model, state):
        model.bind_molecular_conditioning(batch)
        cond = dict(model._mol_cond)
        b, kk = state.shape
        perm = torch.arange(kk)
        perm[j], perm[k] = k, j
        s2 = state.clone()
        s2[row] = state[row, perm]
        for key in ('dof_static', 'dof_atoms', 'dof_mask'):
            v = cond[key].reshape(b, kk, *cond[key].shape[1:][1:]) if cond[key].shape[0] != b \
                else cond[key].reshape(b, kk, -1)
            v = v.clone()
            v[row] = v[row, perm]
            cond[key] = v.reshape(cond[key].shape)
        model._mol_cond = cond
        out = model._pb_net(model.expand_state_for_policy(s2), _cond(model, batch),
                            torch.full((b,), 0.5, dtype=state.dtype))
        return out, perm

    for kind, equivariant in (('set', True), ('flat', False)):
        m = _build(carrier, tmp_path / kind, kind=kind)
        model = m.gfn_model
        x = _state(model, batch)
        base = _pb_out(model, batch, x)
        (dm, dv), perm = swapped(model, x)
        same = all(torch.allclose(a[row, perm], b[row], rtol=0, atol=1e-12)
                   for a, b in ((base[0], dm), (base[1], dv)))
        assert same is equivariant, kind
        if equivariant:                                   # and the other rows did not move
            others = [r for r in range(batch.num_graphs) if r != row]
            assert torch.allclose(base[0][others], dm[others], rtol=0, atol=1e-12)


def test_pads_get_no_correction(carrier, tmp_path):
    m = _build(carrier, tmp_path)
    batch = _condition_batch(carrier, n_rep=1)
    dm, dv = _pb_out(m.gfn_model, batch, _state(m.gfn_model, batch))
    pad = ~batch.state_mask.reshape(batch.num_graphs, -1).bool()
    assert bool(pad.any()) and not dm[pad].any() and not dv[pad].any()


def test_rollouts_run_and_the_three_paths_score_one_kernel(carrier, tmp_path):
    m = _build(carrier, tmp_path)
    batch = _condition_batch(carrier)
    fwd, bwd, rep = _rollouts(m.gfn_model, batch)
    assert all(torch.isfinite(t).all() for leg in (fwd, bwd, rep) for t in leg[:3])
    # replay re-scores the forward trajectory: the same P_F and the same P_B
    assert torch.allclose(rep[1], fwd[1], rtol=0, atol=1e-10)
    assert torch.allclose(rep[2], fwd[2], rtol=0, atol=1e-10)
    pads = ~batch.state_mask.reshape(batch.num_graphs, -1).bool()
    assert not fwd[0][:, :, :][pads.unsqueeze(1).expand_as(fwd[0])].any()
    assert not bwd[0][pads.unsqueeze(1).expand_as(bwd[0])].any()


# ------------------------------------------------------------------ freeze and checkpoint


def test_freeze_holds_the_set_backward_head(carrier, tmp_path):
    m = _build(carrier, tmp_path)
    model = m.gfn_model
    batch = _condition_batch(carrier, n_rep=1)
    x = _state(model, batch)
    m.set_pb_freeze('full')
    assert isinstance(model._pb_frozen['backward_policy'], RaggedConditionalSetPolicy)
    held = _pb_out(model, batch, x)
    assert not held[0].requires_grad
    _perturb(model.backward_policy, 9, 0.2)              # the live head moves; P_B does not
    again = _pb_out(model, batch, x)
    assert torch.equal(held[0], again[0]) and torch.equal(held[1], again[1])
    m.set_pb_freeze(None)
    live = _pb_out(model, batch, x)
    assert not torch.equal(held[0], live[0])


@pytest.fixture(scope='module')
def saved(carrier, tmp_path_factory):
    ckdir = tmp_path_factory.mktemp('ckpb')
    a = _build(carrier, ckdir)
    _perturb(a.ema_model, 2, 0.01)
    a.set_pb_freeze('full')
    _perturb(a.gfn_model.backward_policy, 3, 0.05)
    _synthetic_step(a, 11)
    a.checkpointer.save('probe')
    return a, ckdir, a.checkpointer.path_for('probe')


def test_full_load_rebuilds_the_set_backward_head(saved, carrier):
    a, ckdir, path = saved
    b = _load(carrier, ckdir, path, model__backward_policy_kind='set')
    for model in (b.gfn_model, b.ema_model):
        assert isinstance(model.backward_policy, RaggedConditionalSetPolicy)
    assert b.gfn_config['conformer'] == a.gfn_config['conformer']
    batch = _condition_batch(carrier)
    x, y = _flat(_rollouts(a.gfn_model, batch)), _flat(_rollouts(b.gfn_model, batch))
    assert torch.equal(x, y), f'max|diff| {float((x - y).abs().max()):.3g}'


def test_config_and_checkpoint_must_agree_on_the_backward_kind(saved, carrier, tmp_path):
    _, ckdir, path = saved
    with pytest.raises(ValueError, match='backward_policy_kind'):
        _load(carrier, ckdir, path)                      # config: flat (absent); file: set
    flat = _build(carrier, tmp_path, kind='flat')
    flat.checkpointer.save('probe')
    with pytest.raises(ValueError, match='backward_policy_kind'):
        _load(carrier, tmp_path, flat.checkpointer.path_for('probe'),
              model__backward_policy_kind='set')         # config: set; file: flat


def test_a_stamp_written_before_the_key_loads_as_flat(carrier, tmp_path):
    a = _build(carrier, tmp_path, kind='flat')
    a.gfn_config['conformer'].pop('backward_policy_kind')    # as an older checkpoint stores it
    a.checkpointer.save('probe')
    b = _load(carrier, tmp_path, a.checkpointer.path_for('probe'))
    assert not isinstance(b.gfn_model.backward_policy, RaggedConditionalSetPolicy)
    with pytest.raises(ValueError, match='backward_policy_kind'):
        _load(carrier, tmp_path, a.checkpointer.path_for('probe'),
              model__backward_policy_kind='set')
