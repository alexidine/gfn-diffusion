"""eval/nf_fit.py's per-row terms of a conditional flow policy: rows_given_ctx re-runs ConditionalLatentFlow.to_base
with the per-row log-derivatives kept, so its rows must sum to the policy's own exact log q on every route the policy
builds (MADE and transformer conditioners, bounded and standardised rows, the bin floor and tied derivatives, several
blocks), and a free context equal to the encoder's must reproduce it. An untrained policy is enough: the identity is
algebraic, and drawing x from the policy keeps every row inside its box."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.nf_fit import _ctx_vec, _with_vec, rows_given_ctx  # noqa: E402
from models.nf_policy import NFPolicy  # noqa: E402
from test_nf_policy import SMALL, gfn_config  # noqa: E402

pytestmark = pytest.mark.fast

ROUTES = {
    'made': {},
    'transformer': dict(conditioner='transformer', tf_width=32, tf_layers=1, tf_heads=2, tf_fp32=True),
    'floor_tied': dict(min_bin_height=1e-5, tied_derivatives=True),
    'standardised_rows': dict(bounded_rows='none'),
    'three_blocks': dict(n_blocks=3),
}


@pytest.mark.parametrize('route', sorted(ROUTES))
def test_row_terms_sum_to_the_policy_log_q(route):
    torch.manual_seed(0)
    pol = NFPolicy(gfn_config(conditional=True), {**SMALL, **ROUTES[route]}, seed=0)
    n = 64
    g = torch.Generator().manual_seed(1)
    pol.fit_data_scalers(torch.rand(512, 12, generator=g) * 1.6 - 0.8, torch.randn(512, 4, generator=g) * 3 + 1)
    # a few random steps so the conditioner is not at its identity initialisation
    with torch.no_grad():
        for p in pol.sampler.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    cond = torch.randn(n, 4, generator=g)
    c = pol._context(cond, None, n)
    x = pol._pin_dead(pol._wrap_ang(pol.sampler.sample(n, c).float()))
    with torch.no_grad():
        ctx = pol.sampler.flow.context_input(c)[0]
        terms = rows_given_ctx(pol.sampler, x, ctx)
        exact = pol.log_density(x, cond, None)
        assert terms.shape == (n, pol.live_dim)
        assert torch.isfinite(exact).all()
        assert torch.allclose(terms.sum(-1), exact, atol=1e-4, rtol=1e-5), float((terms.sum(-1) - exact).abs().max())
        again = rows_given_ctx(pol.sampler, x, _with_vec(ctx, _ctx_vec(ctx).clone()))
        assert torch.equal(again, terms)


def test_a_free_context_moves_log_q_and_carries_a_gradient():
    """The amortisation probe optimises the context vector alone: log q must depend on it and be differentiable in
    it with the flow's parameters frozen."""
    torch.manual_seed(0)
    pol = NFPolicy(gfn_config(conditional=True), dict(SMALL), seed=0)
    with torch.no_grad():
        for p in pol.sampler.parameters():
            p.add_(0.05 * torch.randn_like(p))
    for p in pol.parameters():
        p.requires_grad_(False)
    n = 16
    cond = torch.randn(n, 4)
    c = pol._context(cond, None, n)
    x = pol._pin_dead(pol._wrap_ang(pol.sampler.sample(n, c).float()))
    vec = _ctx_vec(pol.sampler.flow.context_input(c)[0]).detach().clone().requires_grad_(True)
    nll = -rows_given_ctx(pol.sampler, x, vec).sum()
    nll.backward()
    assert vec.grad is not None and float(vec.grad.abs().sum()) > 0
