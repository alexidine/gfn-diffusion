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


@pytest.mark.parametrize('shape', [0.2, 0.5, 0.9])
def test_pareto_khat_recovers_the_tail_shape_of_known_weights(shape):
    """PSIS's k-hat on weights whose upper tail is generalised Pareto of a known shape (log w = log of the GPD draw):
    the fitted shape must come back within 0.15 at 20,000 draws."""
    import numpy as np
    from eval.nf_fit import pareto_khat
    u = np.random.default_rng(0).random(20000)
    w = ((1 - u) ** (-shape) - 1) / shape
    assert abs(pareto_khat(np.log(w)) - shape) < 0.15


def test_condition_is_on_an_exact_proposal_has_no_divergence():
    """q = pi up to the constant: every log w equals log Z, so both KLs are 0 and the ESS is full."""
    from eval.nf_fit import condition_is
    r = condition_is(torch.full((1000,), 3.0) + 1e-9 * torch.randn(1000))
    assert abs(r['log_z'] - 3.0) < 1e-6 and abs(r['kl_q_pi']) < 1e-6 and abs(r['kl_pi_q']) < 1e-6
    assert r['ess_frac'] > 0.999
