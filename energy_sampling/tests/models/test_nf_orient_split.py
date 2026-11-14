"""nf orient_loss_weight / orient_grad_balance / orient_grad_rho (models/nf_policy.py NF_PENDING_DEFAULTS): the MLE
term split into its FD orientation rows x_O and the other rows x_C, log q = log q(x_C | c) + log q(x_O | x_C, c). So:
  * the defaults take today's path (no split, nothing armed, the backward loss unchanged);
  * the split leaves log q bitwise, and its orientation part is the conditional: the other rows' part does not move
    when x_O does;
  * a weight w trains -(log q + (w - 1) log q(x_O | x_C, c)) while bwd/mle still reports -log q;
  * a balance hands the trainer the two blocks' batch-mean losses, which sum to the step's loss, and the trainer's
    gradient is rho g_O/|g_O| + g_C/|g_C| (gflownet_losses.py::combine_block_grads, hand values on a toy; pcgrad
    removes the conflicting components; train.py::Modeller._balanced_backward writes it as the parameters' gradient);
  * malformed or unsplittable settings are refused.

    python -m pytest -q tests/models/test_nf_orient_split.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_typed_rows import _typed_policy, _mols  # noqa: E402
from test_nf_policy import _bwd_coeffs  # noqa: E402
from energies.fd_coordinates import frame_rows  # noqa: E402
from gflownet_losses import combine_block_grads, get_gfn_backward_loss  # noqa: E402
from models.nf_policy import NF_PENDING_DEFAULTS, orient_split_problems  # noqa: E402

pytestmark = pytest.mark.fast
N = 4
O = frame_rows([2])


def _setup(dtype=torch.float64, **flow):
    """float64 for the flow-level identities; float32 wherever get_traj_bwd runs (it casts states to float32)."""
    pol = _typed_policy(2, scale=0.05, atom_tokens='split', rotate_frames=True, **flow).to(dtype)
    mb = _mols(N, 2)
    cond = torch.randn(N, 4, generator=torch.Generator().manual_seed(2), dtype=dtype)
    with torch.no_grad():
        x = pol.sampler.sample(N, pol._context(cond, mb, N).to(dtype))
    return pol, mb, cond, x


def _hand_split(pol, mb, cond, x):
    """log q and log q(x_O | x_C, c) of the backward step's rows in the frames torch.manual_seed(7) draws."""
    torch.manual_seed(7)
    q = pol._frames(N)
    c = pol._context(cond, mb, N, frames=q)
    xm, lj = pol._present(pol._wrap_ang(x), q)
    log_q, log_q_o = pol.sampler.log_prob(xm, c, rows=O)
    lj = lj.to(log_q.dtype)
    return log_q + lj, log_q_o + lj


def test_the_defaults_take_todays_path():
    from config_invariants import ORIENT_SPLIT_DEFAULTS
    assert {k: NF_PENDING_DEFAULTS[k] for k in ORIENT_SPLIT_DEFAULTS} == ORIENT_SPLIT_DEFAULTS == dict(
        orient_loss_weight=1.0, orient_grad_balance='off', orient_grad_rho=1.0)
    pol, mb, cond, x = _setup(torch.float32)
    assert pol.orient_cols is None and pol.mle_orient_split() is None
    pol.train()
    torch.manual_seed(7)
    loss, d = get_gfn_backward_loss(_bwd_coeffs(mle=1), x, pol, None, None, mb, condition=cond, repeats=1,
                                    report_losses=True, step=0)
    log_q, _ = _hand_split(pol, mb, cond, x)
    assert torch.allclose(loss, -log_q.mean(), atol=1e-5) and 'mle_orient' not in d
    assert pol._mle_parts is None
    with pytest.raises(RuntimeError):
        pol.take_orient_log_q(N)


def test_the_split_leaves_log_q_bitwise_and_its_orientation_part_is_the_conditional():
    pol, mb, cond, x = _setup(orient_loss_weight=10.0)
    assert pol.orient_cols == O                         # P-1: no dead rows, flow positions = chart rows
    c = pol._context(cond, mb, N).double()
    with torch.no_grad():
        whole = pol.sampler.log_prob(x, c)
        total, part = pol.sampler.log_prob(x, c, rows=O)
        assert torch.equal(total, whole)
        x2 = x.clone()
        x2[:, O] = pol.sampler.sample(N, c)[:, O]       # another orientation, same cell and centroid
        total2, part2 = pol.sampler.log_prob(x2, c, rows=O)
    assert torch.allclose(total - part, total2 - part2, atol=1e-10)   # log q(x_C | c) does not read x_O
    assert float((part2 - part).abs().max()) > 1e-3


def test_a_weight_trains_the_weighted_loss_and_bwd_mle_stays_minus_log_q():
    w = 30.0
    pol, mb, cond, x = _setup(torch.float32, orient_loss_weight=w)
    pol.train()
    params = [p for p in pol.parameters() if p.requires_grad]
    torch.manual_seed(7)
    loss, d = get_gfn_backward_loss(_bwd_coeffs(mle=1), x, pol, None, None, mb, condition=cond, repeats=1,
                                    report_losses=True, step=0)
    g1 = torch.autograd.grad(loss, params, allow_unused=True)
    log_q, log_q_o = _hand_split(pol, mb, cond, x)
    g2 = torch.autograd.grad(-(log_q + (w - 1) * log_q_o).mean(), params, allow_unused=True)
    for a, b in zip(g1, g2):
        assert (a is None and b is None) or torch.allclose(a, b, atol=1e-4, rtol=1e-4)
    assert torch.allclose(d['mle'], -log_q.mean().detach(), atol=1e-5)
    assert torch.allclose(d['mle_orient'], -log_q_o.mean().detach(), atol=1e-5)
    assert torch.allclose(d['mle_rest'], -(log_q - log_q_o).mean().detach(), atol=1e-5)
    assert pol._orient_log_q is None and pol._mle_parts is None


@pytest.mark.parametrize('balance', ('norm', 'pcgrad'))
def test_a_balance_hands_the_trainer_the_two_block_losses(balance):
    pol, mb, cond, x = _setup(torch.float32, orient_grad_balance=balance, orient_grad_rho=3.0)
    pol.train()
    torch.manual_seed(7)
    loss, _ = get_gfn_backward_loss(_bwd_coeffs(mle=1), x, pol, None, None, mb, condition=cond, repeats=1,
                                    report_losses=True, step=0)
    held, l_c, l_o, rho, pcgrad = pol._mle_parts
    assert held is loss and rho == 3.0 and pcgrad == (balance == 'pcgrad')
    log_q, log_q_o = _hand_split(pol, mb, cond, x)
    assert torch.allclose(l_c + l_o, loss, atol=1e-5)
    assert torch.allclose(l_o, -log_q_o.mean(), atol=1e-5)
    with torch.no_grad():                               # a no-grad call (a readout, a probe) clears the hand-over
        get_gfn_backward_loss(_bwd_coeffs(mle=1), x, pol, None, None, mb, condition=cond, repeats=1, step=0)
    assert pol._mle_parts is None


def test_the_combined_gradient_is_its_hand_value():
    g_c = [torch.tensor([3.0, 4.0]), None, torch.tensor([0.0])]       # |g_C| = 5
    g_o = [torch.tensor([0.0, 1.0]), torch.tensor([2.0]), None]       # |g_O| = sqrt 5, g_C . g_O = 4
    out, st = combine_block_grads(g_c, g_o, rho=2.0)
    r5 = 5 ** 0.5
    want = [torch.tensor([3 / 5, 4 / 5 + 2 / r5]), torch.tensor([4 / r5]), torch.tensor([0.0])]
    for a, b in zip(out, want):
        assert torch.allclose(a, b, atol=1e-6)
    assert st['norm_C'] == pytest.approx(5.0) and st['norm_O'] == pytest.approx(r5)
    assert st['cos'] == pytest.approx(4 / (5 * r5))
    # equal component norms at rho 1: the result is the sum of two unit vectors
    out, _ = combine_block_grads([torch.tensor([100.0, 0.0])], [torch.tensor([0.0, 0.01])])
    assert torch.allclose(out[0], torch.tensor([1.0, 1.0]))
    # pcgrad on agreeing gradients: unchanged
    agree, _ = combine_block_grads(g_c, g_o, rho=2.0, pcgrad=True)
    for a, b in zip(agree, want):
        assert torch.allclose(a, b, atol=1e-6)


def test_pcgrad_removes_the_conflicting_components():
    g_c, g_o = [torch.tensor([1.0, 0.0])], [torch.tensor([-1.0, 1.0])]   # cos = -1/sqrt 2
    out, st = combine_block_grads(g_c, g_o, rho=1.0, pcgrad=True)
    u_c, u_o = g_c[0], g_o[0] / 2 ** 0.5
    cos = st['cos']
    assert cos == pytest.approx(-2 ** -0.5)
    want = (u_c - cos * u_o) + (u_o - cos * u_c)
    assert torch.allclose(out[0], want, atol=1e-6)
    assert float(((u_c - cos * u_o) * u_o).sum()) == pytest.approx(0.0, abs=1e-6)   # u_C' is orthogonal to u_O


def test_the_trainer_writes_the_combined_gradient():
    from train import Modeller
    lin = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[1.0, -2.0]]))
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = lin(x).squeeze(1)
    l_c, l_o = (y[0] ** 2), 10.0 * y[1]                 # g_C = [2, 0], g_O = [0, 10]
    stub = SimpleNamespace(gfn_model=lin, step_ind=10)
    Modeller._balanced_backward(stub, l_c, l_o, 1.0, False)
    assert torch.allclose(lin.weight.grad, torch.tensor([[1.0, 1.0]]))
    assert stub._grad_balance_stats['gradbal/ratio'] == pytest.approx(0.2)


@pytest.mark.parametrize('bad', [dict(orient_loss_weight=0.0), dict(orient_loss_weight=float('nan')),
                                 dict(orient_grad_balance='sideways'), dict(orient_grad_rho=-1.0),
                                 dict(orient_loss_weight=True),
                                 dict(orient_grad_balance='norm', orient_loss_weight=2.0),
                                 dict(orient_grad_rho=3.0)])
def test_malformed_settings_are_refused(bad):
    assert orient_split_problems(bad)
    with pytest.raises(ValueError):
        _typed_policy(2, scale=0.0, atom_tokens='split', **bad)


def test_unsplittable_routes_are_refused():
    from test_nf_policy import flow_policy
    with pytest.raises((NotImplementedError, ValueError)):   # two blocks (the typed-rows describer refuses it first)
        _typed_policy(2, scale=0.0, atom_tokens='split', n_blocks=2, orient_loss_weight=10.0)
    with pytest.raises(NotImplementedError, match='conditional route'):
        flow_policy(False, orient_loss_weight=10.0)
    with pytest.raises(NotImplementedError, match='crystal_coordinates fd'):
        flow_policy(True, orient_grad_balance='norm')        # the latent chart: no FD orientation rows
