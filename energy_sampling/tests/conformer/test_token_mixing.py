"""`model.set_policy_mix_layers`: coordinate tokens of one molecule attend to each other.

WHY. Without it each coordinate's head sees its own value plus two sums over ALL of the
molecule's tokens, identical for every coordinate, so a ring dihedral cannot read the current
value of the dihedral next to it. TokenMixer gives every token attention over the others,
biased by how many defining atoms the two share.

PINNED: the relation it is biased by; a fresh mixer is the identity (so turning it on starts
from the unmixed head); the output still follows a coordinate, not a column; rows do not see
each other; pads get nothing; the setting is stamped, rebuilt by a reload, and absent in an
older checkpoint means off.
"""
import pytest
import torch

from energies.multi_conformer import MultiConformerTorsions
from models.ragged_set_policy import N_SHARED_BUCKETS, shared_atom_relation
from test_set_policy_resume import (KW, SMIS, _args, _condition_batch, _flat, _load,
                                    _modeller, _perturb, _rollouts, _synthetic_step)

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


def _build(carrier, ckdir, mix=2, perturb=True, **kw):
    m = _modeller(carrier, _args(ckdir, model__backward_policy_kind='set',
                                 model__set_policy_mix_layers=mix, **kw))
    m.init_gfn()
    if perturb:
        _perturb(m.gfn_model, 1, 0.05)
    return m


def _forward_out(model, batch, state, t=0.5):
    model.bind_molecular_conditioning(batch)
    n = batch.num_graphs
    return model.forward_policy(state, model.t_model(torch.full((n,), t)), **model._mol_cond)


def _state(model, batch, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = 0.6 * (2 * torch.rand(batch.num_graphs, model.dim, generator=g) - 1)
    return x * batch.state_mask.reshape(batch.num_graphs, -1).to(x.dtype)


def test_relation_counts_shared_atoms(carrier, tmp_path):
    m = _build(carrier, tmp_path, perturb=False)
    batch = _condition_batch(carrier, n_rep=1)
    m.gfn_model.bind_molecular_conditioning(batch)
    rel = m.gfn_model._mol_cond['token_rel']
    atoms = m.gfn_model._mol_cond['dof_atoms'][:, :, 0, :]
    b, k = rel.shape[:2]
    assert torch.equal(rel, rel.transpose(1, 2))
    assert bool((rel.diagonal(dim1=1, dim2=2) == N_SHARED_BUCKETS).all())
    off = ~torch.eye(k, dtype=torch.bool)
    assert int(rel[:, off].max()) <= 4 and int(rel.min()) >= 0
    # brute force on every valid pair of the methanol row
    row = 1
    valid = torch.nonzero(batch.state_mask.reshape(b, -1)[row]).flatten().tolist()
    for i in valid:
        for j in valid:
            if i != j:
                want = len(set(atoms[row, i].tolist()) & set(atoms[row, j].tolist()))
                assert int(rel[row, i, j]) == want, (i, j)


def test_a_fresh_mixer_is_the_identity(carrier, tmp_path):
    plain = _build(carrier, tmp_path / 'a', mix=0, perturb=False)
    mixed = _build(carrier, tmp_path / 'b', mix=2, perturb=False)
    batch = _condition_batch(carrier, n_rep=1)
    x = _state(plain.gfn_model, batch)
    for name in ('forward_policy', 'backward_policy'):
        a, b = getattr(plain.gfn_model, name), getattr(mixed.gfn_model, name)
        shared = {k: v for k, v in b.state_dict().items() if not k.startswith('mixers.')}
        assert set(shared) == set(a.state_dict())
        a.load_state_dict(shared)                       # same non-mixer weights in both
    out_a = _forward_out(plain.gfn_model, batch, x)
    out_b = _forward_out(mixed.gfn_model, batch, x)
    assert torch.equal(out_a, out_b)


def test_output_follows_the_coordinate_and_rows_stay_apart(carrier, tmp_path):
    m = _build(carrier, tmp_path)
    model = m.gfn_model
    batch = _condition_batch(carrier, n_rep=1)
    x = _state(model, batch)
    base = _forward_out(model, batch, x)
    b, k = x.shape
    row = 1
    mask = batch.state_mask.reshape(b, -1)
    i, j = [int(c) for c in torch.nonzero(mask[row])[:2].flatten()]
    perm = torch.arange(k)
    perm[i], perm[j] = j, i
    model.bind_molecular_conditioning(batch)
    cond = dict(model._mol_cond)
    for key in ('dof_static', 'dof_atoms', 'dof_mask'):
        v = cond[key].clone()
        v[row] = v[row, perm]
        cond[key] = v
    cond['token_rel'] = shared_atom_relation(cond['dof_atoms'])
    x2 = x.clone()
    x2[row] = x[row, perm]
    out = model.forward_policy(x2, model.t_model(torch.full((b,), 0.5)), **cond)
    for block in (slice(0, k), slice(k, 2 * k)):
        assert torch.allclose(base[:, block][row, perm], out[:, block][row], rtol=0, atol=1e-12)
    # another row's state moves; this row's output does not
    x3 = x.clone()
    others = [r for r in range(b) if r != row]
    x3[others] = -x3[others]
    out3 = _forward_out(model, batch, x3)
    assert torch.allclose(out3[row], base[row], rtol=0, atol=1e-12)
    assert not torch.allclose(out3[others], base[others])


def test_the_mixer_reads_other_coordinates(carrier, tmp_path):
    """With pooling cut out of the head, a token's output still moves when another token of
    its row moves -- only the mixer can carry that."""
    m = _build(carrier, tmp_path)
    model = m.gfn_model
    pol = model.forward_policy
    batch = _condition_batch(carrier, n_rep=1)
    x = _state(model, batch)
    b, k = x.shape
    mask = batch.state_mask.reshape(b, -1)
    row = 1
    i, j = [int(c) for c in torch.nonzero(mask[row])[:2].flatten()]
    model.bind_molecular_conditioning(batch)
    t = model.t_model(torch.full((b,), 0.5))
    seen = {}
    for nm, layers in (('mixed', pol.mix_layers), ('plain', 0)):
        held = pol.mix_layers
        pol.mix_layers = layers
        hs = []
        for xv in (x, x.clone().index_put_((torch.tensor([row]), torch.tensor([j])),
                                            torch.tensor([0.9]))):
            captured = {}
            # a hook that RETURNS a value replaces the module's output, so this one returns None
            hook = pol.score.register_forward_hook(lambda mod, inp, out: captured.update(h=inp[0]))
            pol(xv, t, **model._mol_cond)
            hook.remove()
            hs.append(captured['h'])
        pol.mix_layers = held
        # token i's representation before pooling: flat position of (row, i)
        flat = model._mol_cond['flat_idx'].tolist().index(row * k + i)
        seen[nm] = not torch.allclose(hs[0][flat], hs[1][flat], rtol=0, atol=1e-12)
    assert seen == {'mixed': True, 'plain': False}


def test_pads_get_nothing(carrier, tmp_path):
    m = _build(carrier, tmp_path)
    batch = _condition_batch(carrier, n_rep=1)
    out = _forward_out(m.gfn_model, batch, _state(m.gfn_model, batch))
    k = m.gfn_model.dim
    pad = ~batch.state_mask.reshape(batch.num_graphs, -1).bool()
    assert not out[:, :k][pad].any() and not out[:, k:][pad].any()


def test_trains_and_round_trips(carrier, tmp_path):
    a = _build(carrier, tmp_path)
    before = [p.detach().clone() for p in a.gfn_model.forward_policy.mixers.parameters()]
    _synthetic_step(a, 4)
    after = list(a.gfn_model.forward_policy.mixers.parameters())
    assert all(not torch.equal(x, y) for x, y in zip(before, after))
    assert a.gfn_config['conformer']['set_policy_mix_layers'] == 2
    a.set_pb_freeze('full')
    a.checkpointer.save('probe')
    b = _load(carrier, tmp_path, a.checkpointer.path_for('probe'),
              model__backward_policy_kind='set', model__set_policy_mix_layers=2)
    assert len(b.gfn_model.forward_policy.mixers) == 2 == len(b.gfn_model.backward_policy.mixers)
    batch = _condition_batch(carrier)
    x, y = _flat(_rollouts(a.gfn_model, batch)), _flat(_rollouts(b.gfn_model, batch))
    assert torch.equal(x, y)
    with pytest.raises(ValueError, match='set_policy_mix_layers'):
        _load(carrier, tmp_path, a.checkpointer.path_for('probe'),
              model__backward_policy_kind='set')


def test_an_older_stamp_means_no_mixing(carrier, tmp_path):
    a = _build(carrier, tmp_path, mix=0)
    for key in ('set_policy_mix_layers', 'set_policy_mix_heads'):
        a.gfn_config['conformer'].pop(key)
    a.checkpointer.save('probe')
    b = _load(carrier, tmp_path, a.checkpointer.path_for('probe'),
              model__backward_policy_kind='set')
    assert b.gfn_model.forward_policy.mix_layers == 0
