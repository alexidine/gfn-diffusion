"""RaggedConformerGFN.split_params is GFN.split_params in a different layout.

The claim under test is numerical equality, not similarity: given the same numbers arranged
as dense contiguous blocks and as ragged per-token columns, the two must return the same
mean and logvar. Everything downstream of the slice -- learned_variance, log_var_range,
var_clip, the baseline addition -- is shared source, so this pins the slice itself and the
fact that nothing was dropped in the copy.

Driven through a STUB carrying only the four attributes split_params reads, because building
a real GFN needs an energy, a prior and a device. The property is arithmetic.
"""
import types

import pytest
import torch

from energy_sampling.models.gfn import GFN
from energy_sampling.models.ragged_gfn import RaggedConformerGFN


def _stub(cls, dim, learned_variance=True, log_var_range=6.0, var_clip=8.0, dplr_rank=0):
    m = types.SimpleNamespace(dim=dim, learned_variance=learned_variance,
                              log_var_range=log_var_range, var_clip=var_clip,
                              dplr_rank=dplr_rank)
    m.split_params = types.MethodType(cls.split_params, m)
    return m


@pytest.mark.parametrize('log_var_range', [6.0, -1])
@pytest.mark.parametrize('learned_variance', [True, False])
@pytest.mark.parametrize('B,dim', [(1, 5), (4, 9), (3, 12)])
def test_ragged_split_equals_dense_split(B, dim, learned_variance, log_var_range):
    torch.manual_seed(0)
    mean = torch.randn(B, dim)
    logvar_i = torch.randn(B, dim)
    base = torch.randn(B, 1)

    dense = _stub(GFN, dim, learned_variance, log_var_range)
    d_mean, d_logvar, d_rho, d_u = dense.split_params(
        torch.cat([mean, logvar_i], dim=-1), base)          # contiguous blocks

    ragged = _stub(RaggedConformerGFN, dim, learned_variance, log_var_range)
    r_mean, r_logvar, r_rho, r_u = ragged.split_params(
        torch.stack([mean.reshape(-1), logvar_i.reshape(-1)], dim=-1),   # per-token columns
        base.repeat_interleave(dim, dim=0).reshape(-1))

    assert torch.allclose(r_mean, d_mean.reshape(-1), atol=1e-6)
    assert torch.allclose(r_logvar, d_logvar.reshape(-1), atol=1e-6)
    assert d_rho is None and d_u is None and r_rho is None and r_u is None


def test_the_clip_is_applied_and_is_the_same_clip():
    """var_clip is what stops a runaway logvar; it must survive the copy."""
    ragged = _stub(RaggedConformerGFN, 3, var_clip=2.0, log_var_range=-1)
    _, logvar, _, _ = ragged.split_params(
        torch.tensor([[0.0, 50.0], [0.0, -50.0]]), torch.zeros(2))
    assert torch.allclose(logvar, torch.tensor([2.0, -2.0]))


def test_a_dense_block_layout_reaching_the_ragged_split_is_REFUSED():
    """The failure this guard exists for: installing the flat policy on a ragged GFN gives
    [B, 2*dim], which would slice as two coordinates and train on nonsense."""
    ragged = _stub(RaggedConformerGFN, 6)
    with pytest.raises(ValueError, match=r'must be \[sum_k, 2\]'):
        ragged.split_params(torch.randn(4, 12), torch.zeros(4, 1))
    with pytest.raises(ValueError, match=r'must be \[sum_k, 2\]'):
        ragged.split_params(torch.randn(4, 6, 2), torch.zeros(4))


def test_dplr_is_refused_here_too_not_only_in_predict_next_state():
    """ConformerGFN.predict_next_state already refuses it, but split_params is reachable
    without going through that method, so the refusal is repeated rather than assumed."""
    ragged = _stub(RaggedConformerGFN, 6, dplr_rank=2)
    with pytest.raises(NotImplementedError, match='dplr_rank'):
        ragged.split_params(torch.randn(24, 2), torch.zeros(24))


def test_a_ragged_batch_of_mixed_k_splits_without_knowing_k():
    """The point of the layout: split_params never needs to know how the rows divide."""
    ragged = _stub(RaggedConformerGFN, 0)          # dim is not consulted at all
    n = 4 + 9 + 6
    mean, logvar, _, _ = ragged.split_params(torch.randn(n, 2), torch.zeros(n))
    assert mean.shape == (n,) and logvar.shape == (n,)
