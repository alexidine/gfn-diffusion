"""ConformerGFN's per-row carrier masking is the parent's arithmetic with the sum masked.

The identity used throughout: for any row mask m, ``masked(m) + masked(~m)`` equals the
parent's unmasked value. That pins both halves at once -- nothing dropped, nothing counted
twice -- without re-deriving the kernels. The P_B mixture is a REPLICA of the shared-file
method minus its final sum, so it is checked against the parent directly.
"""
import pytest
import torch

from energy_sampling.models.gfn import GFN
from models.conformer_gfn import ConformerGFN

DIM = 9
MASK = [False] * 6 + [True] * 3          # r, theta linear; phi block wraps
B = 16


def _gfn(pb_exact_reversal=True, seed=0):
    torch.manual_seed(seed)
    g = GFN(dim=DIM, angular_mask=MASK, s_emb_dim=32, conditions_dim=1, harmonics_dim=8,
            t_dim=8, t_hidden_dim=32, s_hidden_dim=32, s_layers=2, policy_hidden_dim=32,
            policy_layers=2, flow_hidden_dim=16, flow_layers=2, t_scale=1.0,
            learned_variance=True, learn_pb=True, conditional=False, device='cpu',
            zero_init=False, dplr_rank=0, pb_exact_reversal=pb_exact_reversal).eval()
    g.__class__ = ConformerGFN
    g._mol_cond, g._state_mask, g._carrier = None, None, True
    return g


def _mask(seed=1):
    torch.manual_seed(seed)
    m = torch.rand(B, DIM) > 0.4
    m[:, 0] = True
    m[:, -1] = False                     # every row has a real and a pad column
    return m


def _pb_args(g):
    torch.manual_seed(2)
    prev, nxt = torch.rand(B, DIM) * 2 - 1, torch.rand(B, DIM) * 2 - 1
    dc = torch.rand(B, 1) * 0.1
    bmc = torch.rand(B, DIM) + 0.5
    bv = torch.rand(B, DIM) * 0.05 + 0.01
    t = torch.rand(B) * 0.8 + 0.1
    return prev, nxt, dc, bmc, bv, t


def test_gauss_logprob_splits_exactly_over_the_mask():
    g = _gfn()
    torch.manual_seed(3)
    dx, dr, v = torch.randn(B, DIM), torch.randn(B, DIM), torch.rand(B, DIM) + 0.1
    full = GFN.gauss_logprob(g, dx, dr, v)
    m = _mask()
    g._state_mask = m
    a = g.gauss_logprob(dx, dr, v)
    g._state_mask = ~m
    b = g.gauss_logprob(dx, dr, v)
    assert torch.allclose(a + b, full, atol=1e-5)
    g._state_mask = torch.ones_like(m)
    assert torch.allclose(g.gauss_logprob(dx, dr, v), full, atol=1e-6)


@pytest.mark.parametrize('exact', [True, False])
def test_pb_logprob_splits_exactly_over_the_mask(exact):
    g = _gfn(pb_exact_reversal=exact)
    args = _pb_args(g)
    full = GFN._pb_logprob(g, *args)
    m = _mask()
    g._state_mask = m
    a = g._pb_logprob(*args)
    g._state_mask = ~m
    b = g._pb_logprob(*args)
    assert torch.allclose(a + b, full, atol=1e-5)


def test_mixture_replica_equals_the_shared_method():
    g = _gfn()
    prev, nxt, dc, bmc, bv, t = _pb_args(g)
    ang = g.ang_idx
    x, y = g._wrap_ang(prev).index_select(1, ang), g._wrap_ang(nxt).index_select(1, ang)
    k, bs = bmc.index_select(1, ang), bv.index_select(1, ang)
    want = GFN._pb_mixture_ang_logprob(g, x, y, dc, k, bs, t)
    got = g._pb_mixture_ang_terms(x, y, dc, k, bs, t).sum(1)
    assert torch.allclose(got, want, atol=1e-6)


def test_pads_are_pinned_to_exact_zero_even_from_nan():
    g = _gfn()
    m = _mask()
    g._state_mask = m
    s = torch.randn(B, DIM)
    s[~m] = float('nan')
    out = g._pin_dead(s)
    assert torch.equal(out[~m], torch.zeros_like(out[~m]))
    assert torch.equal(out[m], s[m])


def test_no_mask_is_the_parent_unchanged():
    g = _gfn()
    torch.manual_seed(4)
    dx, dr, v = torch.randn(B, DIM), torch.randn(B, DIM), torch.rand(B, DIM) + 0.1
    assert torch.equal(g.gauss_logprob(dx, dr, v), GFN.gauss_logprob(g, dx, dr, v))
    s = torch.randn(B, DIM)
    assert g._pin_dead(s) is s


def test_a_mask_for_a_different_batch_is_refused():
    g = _gfn()
    g._state_mask = _mask()[:4]
    with pytest.raises(RuntimeError, match='does not match'):
        g._pin_dead(torch.randn(B, DIM))


def test_a_carrier_gfn_refuses_a_batch_without_state_mask():
    g = _gfn()

    class _Batch:
        num_graphs = B
    with pytest.raises(RuntimeError, match='state_mask'):
        g._bind_state_mask(_Batch())
