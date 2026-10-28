"""The forward-branch tb term under models/nf_policy.py::NFPolicy (docs/wiki/trajectory-balance.md, 'Under the flow
policy'): on on-policy draws x ~ q the residual is r = log Z + log q(x) - log R(x), the draw is detached, and the
policy gradient of the Huber loss is mean_i clip(r_i, +-beta) grad log q(x_i). Inside the knee that is the
score-function estimator of grad KL(q || pi), log Z entering as a baseline whose term has zero expectation."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gflownet_losses import get_gfn_forward_loss  # noqa: E402
from models.nf_policy import NFPolicy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_policy import SMALL, gfn_config  # noqa: E402

pytestmark = pytest.mark.fast
DIM = 12


def unconditional_policy(log_z=0.0):
    torch.manual_seed(0)
    pol = NFPolicy(gfn_config(False, dead=None), dict(SMALL), seed=0)
    assert pol.dead_idx.numel() == 0 and pol.live_dim == DIM
    with torch.no_grad():
        pol.flow_model.scalar.fill_(float(log_z))
    return pol


def fwd_coeffs(beta):
    return SimpleNamespace(tb=1.0, beta=float(beta), traj_grads=0.0, path_grad_last_k=0, reward_grads=0.0,
                           freeze_policy=0.0, freeze_z=0.0, vg_lb=0.0, vg_lme=0.0, emp_z=0.0, db=0.0, subtb=0.0,
                           loss_clip=-1)


class Reward:
    """log R(x) = -2 |x|^2 + 0.3 sum(x): a smooth target; records the terminals the loss scored."""

    def __init__(self):
        self.x = None

    def __call__(self, x, mol_batch, log_T, return_exp, keep_grads=False):
        self.x = x.detach().clone()
        return self.log_r(x)

    @staticmethod
    def log_r(x):
        return -2.0 * (x ** 2).sum(-1) + 0.3 * x.sum(-1)


def fwd_tb_grads(pol, n, beta):
    """The forward loss's gradient on the flow's blocks and on log Z, and the draws it was taken on."""
    reward = Reward()
    pol.zero_grad(set_to_none=True)
    loss, _ = get_gfn_forward_loss(fwd_coeffs(beta), torch.zeros(n, DIM), pol, reward, None, None,
                                   torch.zeros(n), condition=torch.zeros(n, 0))
    loss.backward()
    g_policy = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in pol.forward_policy.parameters()]
    return g_policy, pol.flow_model.scalar.grad.detach().clone(), reward.x


def score_function_grads(pol, x, beta):
    """mean_i clip(log Z + log q(x_i) - log R(x_i), +-beta) grad log q(x_i), on the given draws."""
    log_q = pol.sampler.log_prob(x)
    r = (pol.flow_model.scalar.detach() + log_q - Reward.log_r(x)).detach()
    surrogate = (r.clamp(-beta, beta) * log_q).mean()
    grads = torch.autograd.grad(surrogate, list(pol.forward_policy.parameters()), allow_unused=True)
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, pol.forward_policy.parameters())], r


def flat(gs):
    return torch.cat([g.flatten() for g in gs])


def test_fwd_tb_gradient_is_the_score_function_reverse_kl_gradient():
    pol = unconditional_policy(log_z=1.5)
    torch.manual_seed(1)
    g_tb, g_z, x = fwd_tb_grads(pol, 256, beta=1e6)
    g_sf, r = score_function_grads(pol, x, beta=1e6)
    assert float(r.abs().max()) < 1e6                                    # every row inside the knee
    assert flat(g_tb).norm() > 0
    assert torch.allclose(flat(g_tb), flat(g_sf), rtol=1e-4, atol=1e-6)
    assert torch.allclose(g_z, r.mean(), rtol=1e-4, atol=1e-6)          # d L / d log Z = mean residual


def test_outside_the_knee_the_weight_is_the_clipped_residual():
    pol = unconditional_policy(log_z=1.5)
    torch.manual_seed(1)
    beta = 2.0
    g_tb, g_z, x = fwd_tb_grads(pol, 256, beta=beta)
    g_sf, r = score_function_grads(pol, x, beta=beta)
    clipped = float((r.abs() > beta).float().mean())
    assert 0.05 < clipped < 0.95                                         # both regimes present
    assert torch.allclose(flat(g_tb), flat(g_sf), rtol=1e-4, atol=1e-6)
    g_unclipped, _ = score_function_grads(pol, x, beta=1e6)
    assert not torch.allclose(flat(g_tb), flat(g_unclipped), rtol=1e-2)  # the clip changes the gradient
    assert torch.allclose(g_z, r.clamp(-beta, beta).mean(), rtol=1e-4, atol=1e-6)


def test_log_z_enters_the_policy_gradient_as_a_baseline():
    """Same draws, two levels: the gradients differ by (Delta log Z) mean_i grad log q(x_i), whose expectation is
    zero (E_q grad log q = 0); on many draws its norm is small next to the per-row score norm."""
    pol = unconditional_policy(log_z=0.0)
    n = 4096
    torch.manual_seed(2)
    g0, _, x = fwd_tb_grads(pol, n, beta=1e6)
    with torch.no_grad():
        pol.flow_model.scalar.fill_(5.0)
    torch.manual_seed(2)
    g5, _, x5 = fwd_tb_grads(pol, n, beta=1e6)
    assert torch.equal(x, x5)                                            # the level does not move the draw
    params = list(pol.forward_policy.parameters())
    mean_score = flat([g if g is not None else torch.zeros_like(p) for g, p in
                       zip(torch.autograd.grad(pol.sampler.log_prob(x).mean(), params, allow_unused=True), params)])
    assert torch.allclose(flat(g5) - flat(g0), 5.0 * mean_score, rtol=1e-3, atol=1e-5)
    row_norms = []
    for i in range(16):
        gi = torch.autograd.grad(pol.sampler.log_prob(x[i:i + 1]).sum(), params, allow_unused=True)
        row_norms.append(float(flat([g if g is not None else torch.zeros_like(p) for g, p in zip(gi, params)]).norm()))
    assert float(mean_score.norm()) < 0.1 * (sum(row_norms) / len(row_norms))
