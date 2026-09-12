"""RaggedSetPolicy computes SetPolicy's function on a batch of mixed coordinate counts.

THE LOAD-BEARING TEST IS test_a_molecules_output_does_not_depend_on_its_batchmates. The
pooled channels are sums over a molecule's OWN coordinates, so the whole construction is
wrong -- silently, and in a way no shape check catches -- if a scatter leaks across the
segment boundary. Dense `SetPolicy` gets that for free from `dim=1` reductions; ragged has
to earn it with the index vector, so it is asserted directly.

The equivalence test pins the rest: on a batch every molecule of which has the same k, the
ragged answer must equal the dense one to float tolerance, with weights copied across. That
is what makes this a re-implementation rather than a new model.
"""
import pytest
import torch

from energy_sampling.models.set_policy import SetPolicy
from energy_sampling.models.ragged_set_policy import (RaggedSetPolicy, segment_softmax,
                                                      segment_sum)

F, H, T = 5, 16, 8
torch.manual_seed(0)


def _pair(dim, n_static=F, t_dim=T, seed=0):
    """A dense policy and a ragged one carrying THE SAME weights."""
    torch.manual_seed(seed)
    static = torch.randn(dim, n_static)
    ang = [bool(i % 3 == 0) for i in range(dim)]
    dense = SetPolicy(static, ang, t_dim, hidden_dim=H, layers=2)
    ragged = RaggedSetPolicy(n_static, t_dim, hidden_dim=H, layers=2)
    ragged.load_state_dict({k: v for k, v in dense.state_dict().items()
                            if not k.startswith(('static', 'is_ang'))})
    return dense, ragged, static, torch.tensor(ang)


# ------------------------------------------------------------------ the segment ops

def test_segment_softmax_normalises_within_each_segment():
    idx = torch.tensor([0, 0, 0, 1, 2, 2])
    a = segment_softmax(torch.randn(6), idx, 3)
    got = torch.zeros(3).index_add_(0, idx, a)
    assert torch.allclose(got, torch.ones(3), atol=1e-6)


def test_segment_softmax_is_shift_invariant_per_segment():
    """Subtracting the per-segment max, not a global one: segments have different lengths,
    so a shared max would bias short molecules against long ones."""
    idx = torch.tensor([0, 0, 1, 1])
    s = torch.tensor([1.0, 2.0, 101.0, 102.0])          # segment 1 offset by +100
    a = segment_softmax(s, idx, 2)
    assert torch.allclose(a[:2], a[2:], atol=1e-6)


def test_segment_softmax_survives_a_large_magnitude():
    idx = torch.tensor([0, 0])
    a = segment_softmax(torch.tensor([1e4, 1e4 + 1.0]), idx, 1)
    assert torch.isfinite(a).all() and torch.allclose(a.sum(), torch.tensor(1.0), atol=1e-6)


def test_segment_sum_matches_a_dense_reduction():
    idx = torch.tensor([0, 0, 1])
    src = torch.randn(3, 4)
    got = segment_sum(src, idx, 2)
    assert torch.allclose(got[0], src[:2].sum(0), atol=1e-6)
    assert torch.allclose(got[1], src[2], atol=1e-6)


# ------------------------------------------------------------------ equivalence

@pytest.mark.parametrize('B,dim', [(1, 7), (4, 7), (3, 12)])
def test_ragged_equals_dense_when_every_row_has_the_same_k(B, dim):
    """The re-implementation claim, stated as an assertion."""
    dense, ragged, static, ang = _pair(dim)
    state = torch.randn(B, dim)
    t_emb = torch.randn(B, T)

    want = dense(state, t_emb)                                   # [B, out*dim] blocks
    dof_batch = torch.arange(B).repeat_interleave(dim)
    got = ragged(state.reshape(-1), dof_batch,
                 static.repeat(B, 1), ang.repeat(B), t_emb)      # [B*dim, out]

    # dense emits CONTIGUOUS BLOCKS [mean(dim), logvar(dim)]; ragged emits per-token
    # columns. Same numbers, and the reshape is the whole difference between them.
    got_blocks = torch.cat([got[:, k].reshape(B, dim)
                            for k in range(ragged.out_per_token)], dim=-1)
    assert torch.allclose(want, got_blocks, atol=1e-5), (want - got_blocks).abs().max()


# ------------------------------------------------------------------ raggedness

def test_a_ragged_batch_of_mixed_k_runs_and_keeps_its_rows():
    ks = [4, 9, 6]
    _, ragged, _, _ = _pair(max(ks))
    dof_batch = torch.cat([torch.full((k,), i) for i, k in enumerate(ks)])
    n = sum(ks)
    out = ragged(torch.randn(n), dof_batch, torch.randn(n, F),
                 torch.randint(0, 2, (n,)), torch.randn(len(ks), T))
    assert out.shape == (n, 2) and torch.isfinite(out).all()


def test_a_molecules_output_does_not_depend_on_its_batchmates():
    """THE LOAD-BEARING PROPERTY. Pooling is within-molecule, so re-batching the same
    molecule beside different partners must not move a single one of its outputs. A scatter
    that leaked across the segment boundary would still return the right SHAPE."""
    _, ragged, _, _ = _pair(9)
    torch.manual_seed(1)
    k_a = 5
    s_a, f_a, g_a, t_a = (torch.randn(k_a), torch.randn(k_a, F),
                          torch.randint(0, 2, (k_a,)), torch.randn(1, T))

    alone = ragged(s_a, torch.zeros(k_a, dtype=torch.long), f_a, g_a, t_a)

    for k_b in (3, 11):                       # a shorter and a longer partner
        s_b, f_b, g_b = torch.randn(k_b), torch.randn(k_b, F), torch.randint(0, 2, (k_b,))
        together = ragged(torch.cat([s_a, s_b]),
                          torch.cat([torch.zeros(k_a, dtype=torch.long),
                                     torch.ones(k_b, dtype=torch.long)]),
                          torch.cat([f_a, f_b]), torch.cat([g_a, g_b]),
                          torch.cat([t_a, torch.randn(1, T)]))
        assert torch.allclose(alone, together[:k_a], atol=1e-6), \
            f'batchmate of size {k_b} changed the first molecule'


def test_molecule_order_in_the_batch_does_not_matter():
    _, ragged, _, _ = _pair(9)
    torch.manual_seed(2)
    ka, kb = 4, 7
    sa, fa, ga, ta = torch.randn(ka), torch.randn(ka, F), torch.randint(0, 2, (ka,)), torch.randn(1, T)
    sb, fb, gb, tb = torch.randn(kb), torch.randn(kb, F), torch.randint(0, 2, (kb,)), torch.randn(1, T)
    ab = ragged(torch.cat([sa, sb]),
                torch.cat([torch.zeros(ka, dtype=torch.long), torch.ones(kb, dtype=torch.long)]),
                torch.cat([fa, fb]), torch.cat([ga, gb]), torch.cat([ta, tb]))
    ba = ragged(torch.cat([sb, sa]),
                torch.cat([torch.zeros(kb, dtype=torch.long), torch.ones(ka, dtype=torch.long)]),
                torch.cat([fb, fa]), torch.cat([gb, ga]), torch.cat([tb, ta]))
    assert torch.allclose(ab[:ka], ba[kb:], atol=1e-6)
    assert torch.allclose(ab[ka:], ba[:kb], atol=1e-6)


def test_the_sin_cos_lift_is_zeroed_on_linear_coordinates():
    """Same contract as the dense version: feeding sin/cos of a bond length is meaningless,
    so the lift is masked and the flag rides along instead."""
    _, ragged, _, _ = _pair(4)
    n = 3
    state, static = torch.randn(n), torch.randn(n, F)
    tok = ragged.tokens(state, static, torch.zeros(n))
    assert torch.allclose(tok[:, F + 1:F + 3], torch.zeros(n, 2)), 'lift must be zeroed'
    assert torch.allclose(tok[:, -1], torch.zeros(n)), 'and the flag must say so'
    tok_ang = ragged.tokens(state, static, torch.ones(n))
    assert not torch.allclose(tok_ang[:, F + 1:F + 3], torch.zeros(n, 2))


def test_shape_disagreements_are_refused_not_broadcast():
    _, ragged, _, _ = _pair(6)
    n = 5
    ok = dict(state=torch.randn(n), dof_batch=torch.zeros(n, dtype=torch.long),
              static=torch.randn(n, F), is_ang=torch.zeros(n), t_emb=torch.randn(1, T))
    with pytest.raises(ValueError, match='static has 4 rows'):
        ragged(**{**ok, 'static': torch.randn(4, F)})
    with pytest.raises(ValueError, match='width'):
        ragged(**{**ok, 'static': torch.randn(n, F + 1)})
    with pytest.raises(ValueError, match='dof_batch indexes molecule'):
        ragged(**{**ok, 'dof_batch': torch.full((n,), 3, dtype=torch.long)})


# ------------------------------------------------------------------ conditional, on a carrier

from energy_sampling.models.ragged_set_policy import RaggedConditionalSetPolicy

K_C, ENC, MOL = 6, 4, 8


def _carrier_inputs(mask, seed=0, n_atoms=7):
    g = torch.Generator().manual_seed(seed)
    B = mask.shape[0]
    flat = mask.reshape(-1).nonzero().squeeze(1)
    return dict(
        state=torch.rand(B, K_C, generator=g) * 2 - 1,
        t_emb=torch.randn(B, T, generator=g),
        atom_emb=torch.randn(n_atoms * B, ENC, generator=g),
        dof_atoms=(torch.randint(0, n_atoms, (B, K_C, 1, 4), generator=g)
                   + (torch.arange(B) * n_atoms).view(B, 1, 1, 1)),
        dof_mask=torch.ones(B, K_C, 1, dtype=torch.bool),
        mol_emb=torch.randn(B, MOL, generator=g),
        state_mask=mask,
        dof_static=torch.randn(B, K_C, F, generator=g),
        flat_idx=flat,
        dof_batch=torch.div(flat, K_C, rounding_mode='floor'))


def _cpol(seed=0):
    torch.manual_seed(seed)
    return RaggedConditionalSetPolicy(F, [False] * 3 + [True] * 3, T, enc_dim=ENC,
                                      mol_dim=MOL, corr_dim=4, hidden_dim=H, layers=2)


def test_carrier_policy_emits_dense_blocks_with_zero_pads():
    pol = _cpol()
    mask = torch.tensor([[1, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 0, 0]]).bool()
    out = pol(**_carrier_inputs(mask))
    assert out.shape == (3, 2 * K_C)
    mean, logvar = out[:, :K_C], out[:, K_C:]
    assert torch.equal(mean[~mask], torch.zeros_like(mean[~mask]))
    assert torch.equal(logvar[~mask], torch.zeros_like(logvar[~mask]))
    assert (mean[mask] != 0).all()


def test_carrier_policy_row_does_not_depend_on_its_batchmates():
    """The same molecule's output must not move when its batchmates change size."""
    pol = _cpol()
    m0 = torch.tensor([[1, 1, 0, 1, 0, 1]]).bool()
    alone = _carrier_inputs(m0, seed=5, n_atoms=7)
    want = pol(**alone)
    mix = torch.tensor([[1, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]]).bool()
    other = _carrier_inputs(mix, seed=9, n_atoms=7)
    for key in ('state', 't_emb', 'mol_emb', 'dof_static'):
        other[key][0] = alone[key][0]
    other['atom_emb'][:7] = alone['atom_emb']
    other['dof_atoms'][0] = alone['dof_atoms'][0]
    got = pol(**other)[0:1]
    assert torch.allclose(got, want, atol=1e-6)


def test_carrier_policy_trains_the_correlator():
    pol = _cpol()
    mask = torch.tensor([[1, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1]]).bool()
    pol(**_carrier_inputs(mask)).pow(2).sum().backward()
    g = [p.grad for p in pol.correlator.parameters() if p.grad is not None]
    assert g and any(float(x.abs().sum()) > 0 for x in g)
