"""NF score matching and pathwise reverse KL (gflownet_losses.py::score_matching_rows / ::pathwise_rows, NFPolicy.score /
._pathwise_draw; trained by train.py::Modeller.fused_train_step outside the frac mix). So:
  * the policy's score is d log q / d x (finite differences), zero on dead rows, and a loss on it trains the policy;
  * the score-matching row loss is the Huber of |g_q - g_R| over the live rows, a row with no force dropped;
  * the pathwise draw is the draw sample() makes from the same base points, its log q has log q's value, and the
    pathwise loss's gradient is the sticking-the-landing estimator: the total derivative of log q(x_theta(z)) minus
    its explicit parameter derivative at fixed x;
  * the forward and backward losses put the terms in gfn._aux_terms only on a trained (grad-enabled) call, the TB
    reward unchanged; and the loader refuses malformed or inert settings.

    python -m pytest -q tests/models/test_nf_score_pathwise.py
"""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_policy import _bwd_coeffs, flow_policy  # noqa: E402
from gflownet_losses import (get_gfn_backward_loss, get_gfn_forward_loss, pathwise_rows,  # noqa: E402
                             score_matching_rows)

pytestmark = pytest.mark.fast
N = 6
DEAD = (9,)


def _setup(dtype=torch.float32):
    pol = flow_policy(True, dead=DEAD).to(dtype)
    c = torch.randn(N, 4, generator=torch.Generator().manual_seed(3), dtype=dtype)
    with torch.no_grad():
        x = pol.get_traj_fwd(torch.zeros(N, 12, dtype=dtype), None, 0, c, None)[0][:, 1]
    return pol, c, x


def _reward(x, mol_batch, log_T, return_exp, keep_grads=False):
    return -3.0 * (x ** 2).sum(-1) + x[:, 0]


def _reward_grad(x):
    g = -6.0 * x
    g[:, 0] += 1.0
    return g


def test_the_score_is_d_log_q_dx_and_zero_on_dead_rows():
    pol, c, x = _setup(torch.float64)
    x = x.double()                                           # get_traj_fwd hands back float32 states
    g = pol.score(x, c, None)
    assert g.requires_grad                                   # create_graph: a loss on it trains the policy
    fd = torch.zeros_like(x)
    with torch.no_grad():
        for j in range(12):
            e = torch.zeros_like(x)
            e[:, j] = 1e-6
            lq = lambda y: pol.sampler.log_prob(pol._pin_dead(pol._wrap_ang(y)), c)   # noqa: E731
            fd[:, j] = (lq(x + e) - lq(x - e)) / 2e-6
    assert torch.allclose(g.detach(), fd, atol=1e-5, rtol=1e-4)
    assert (g.detach()[:, list(DEAD)] == 0).all()
    params = [p for p in pol.parameters() if p.requires_grad]
    grads = torch.autograd.grad((g ** 2).sum(), params, allow_unused=True)
    assert any(gr is not None and gr.abs().sum() > 0 for gr in grads)


def test_score_matching_rows_are_the_huber_of_the_residual_norm():
    g_q = torch.tensor([[3.0, 4.0], [0.3, 0.4], [1.0, 1.0]], requires_grad=True)
    g_r = torch.tensor([[0.0, 0.0], [0.0, 0.0], [float('nan'), 0.0]])
    loss, st = score_matching_rows(g_q, g_r, beta=1.0)
    # row 0: |r| = 5 > 1 -> 1 * (5 - 0.5) = 4.5; row 1: |r| = 0.5 -> 0.125; row 2: no force, dropped
    assert float(loss) == pytest.approx((4.5 + 0.125) / 2)
    assert float(st['score_bad_frac']) == pytest.approx(1 / 3)
    assert float(st['score_capped_frac']) == pytest.approx(1 / 3)
    loss0, _ = score_matching_rows(g_q, g_q.detach(), beta=1.0)
    assert float(loss0) == 0.0
    lossq, _ = score_matching_rows(g_q[:2], g_r[:2], beta=0.0)          # no cap: 0.5 |r|^2
    assert float(lossq) == pytest.approx((12.5 + 0.125) / 2)
    live, _ = score_matching_rows(g_q[:2], g_r[:2], beta=0.0, live_idx=torch.tensor([1]))
    assert float(live) == pytest.approx((8.0 + 0.08) / 2)


def test_the_pathwise_draw_is_samples_draw_and_its_loss_is_the_stl_gradient():
    pol, c, _ = _setup()
    torch.manual_seed(11)
    with torch.no_grad():
        x_ref = pol._pin_dead(pol._wrap_ang(pol.sampler.sample(N, c)))
    torch.manual_seed(11)
    x, log_q_stl = pol._pathwise_draw(N, pol._context(c, None, N))
    assert torch.allclose(x.detach(), x_ref, atol=1e-4)
    assert torch.allclose(log_q_stl.detach(), pol.sampler.log_prob(x.detach(), c), atol=1e-4)
    params = [p for p in pol.sampler.parameters() if p.requires_grad]
    loss, st = pathwise_rows(torch.zeros(N), log_q_stl, x)
    g_stl = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = torch.autograd.grad(pol.sampler.log_prob(x, c).mean(), params, retain_graph=True, allow_unused=True)
    explicit = torch.autograd.grad(pol.sampler.log_prob(x.detach(), c).mean(), params, allow_unused=True)
    for p, a, t, e in zip(params, g_stl, total, explicit):
        z = torch.zeros_like(p)
        want = (z if t is None else t) - (z if e is None else e)
        assert torch.allclose(z if a is None else a, want, atol=1e-4, rtol=1e-3)
    assert float(st['pathwise']) == pytest.approx(float(log_q_stl.mean()), abs=1e-5)


def test_a_pathwise_row_with_no_reward_is_dropped_and_the_clip_caps_each_row():
    log_r = torch.tensor([0.0, float('nan'), 0.0, 0.0])
    x2 = torch.randn(4, 3, requires_grad=True)
    loss2, st2 = pathwise_rows(log_r, (100 * x2).sum(-1), x2, grad_clip=1.0)
    g, = torch.autograd.grad(loss2, x2)
    assert float(st2['pathwise_bad_frac']) == pytest.approx(0.25)
    assert (g[1] == 0).all()                                   # the dropped row
    assert torch.allclose(g[[0, 2, 3]].norm(dim=-1) * 3, torch.ones(3), atol=1e-5)   # each row's own norm, capped


def _fwd_coeffs(**on):
    return _bwd_coeffs(**{'tb': 1, 'beta': 10.0, 'traj_grads': 0, 'reward_grads': 0, 'pooled_source': None,
                          'coeff_matrix': None, **on})


def test_the_forward_loss_arms_the_terms_and_keeps_the_tb_reward():
    pol, c, _ = _setup()
    seen = {}

    def reward(x, mol_batch, log_T, return_exp, keep_grads=False):
        seen.setdefault('calls', 0)
        seen['calls'] += 1
        seen['keep'] = keep_grads
        return _reward(x, mol_batch, log_T, return_exp, keep_grads)

    pol._aux_terms = None
    loss, d = get_gfn_forward_loss(_fwd_coeffs(score=0.5, score_beta=0.0, pathwise=2.0), torch.zeros(N, 12), pol,
                                   reward, None, None, torch.zeros(N), condition=c, report_losses=True)
    assert seen == {'calls': 1, 'keep': True}                     # one reward call serves TB, the force and pathwise
    assert set(pol._aux_terms) == {'fwd_score', 'fwd_pathwise'}
    assert pol._pathwise is None and pol._aux_terms['fwd_pathwise'].requires_grad
    assert {'score', 'score_cos', 'pathwise', 'pathwise_log_r'} <= set(d)
    # the pathwise term is 2 * mean(log q - log R) at the live draw; TB is scored on the same (detached) draw
    assert torch.allclose(pol._aux_terms['fwd_pathwise'].detach(), 2.0 * (d['pathwise_log_q'] - d['pathwise_log_r']),
                          atol=1e-4)
    pol._aux_terms = None
    with torch.no_grad():                                        # a readout arms nothing
        get_gfn_forward_loss(_fwd_coeffs(score=0.5, pathwise=2.0), torch.zeros(N, 12), pol, reward, None, None,
                             torch.zeros(N), condition=c)
    assert pol._aux_terms is None


@pytest.mark.parametrize('source', ('live', 'stored'))
def test_the_backward_loss_scores_its_rows_against_their_force(source):
    pol, c, x = _setup()
    mb = SimpleNamespace(replay_force=_reward_grad(x)) if source == 'stored' else None
    pol._aux_terms = None
    pol.train()
    get_gfn_backward_loss(_bwd_coeffs(mle=1, score=1.0, score_beta=0.0, score_force=source), x, pol,
                          _reward(x, None, None, False), None, mb, condition=c, repeats=1, live_stash='replay',
                          log_reward_fn=_reward, log_T_tensor=torch.zeros(N))
    term = pol._aux_terms['replay_score']
    g_q = pol.score(x, c, None).detach()
    live = pol.live_idx
    want = 0.5 * ((g_q - _reward_grad(x))[:, live] ** 2).sum(-1).mean()
    assert torch.allclose(term.detach(), want, atol=1e-3, rtol=1e-4)
    pol._aux_terms = None
    get_gfn_backward_loss(_bwd_coeffs(mle=1, score=1.0, score_force=source), x, pol, _reward(x, None, None, False),
                          None, mb, condition=c, repeats=1, live_stash=None, log_reward_fn=_reward,
                          log_T_tensor=torch.zeros(N))
    assert pol._aux_terms is None                                # a probe call (no stash) arms nothing


def test_framed_policies_refuse_the_score_and_the_draw():
    pol = flow_policy(True, dead=DEAD)
    pol.settings = {**pol.settings, 'canonical_frame': 'inertial'}
    with pytest.raises(NotImplementedError):
        pol.score(torch.zeros(2, 12), torch.zeros(2, 4), None)
    with pytest.raises(NotImplementedError):
        pol._pathwise_draw(2, torch.zeros(2, 16))


def _canonical_nf():
    import yaml
    from config_invariants import canonical_model_section
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml', encoding='utf-8'))
    cfg = copy.deepcopy(canonical_model_section(cfg))
    cfg['policy']['kind'] = 'nf'
    return cfg


def _fires(cfg):
    from config_invariants import nf_aux_terms_are_well_formed
    return nf_aux_terms_are_well_formed(cfg)


def test_the_loader_refuses_malformed_or_unconsumable_terms():
    cfg = _canonical_nf()
    assert not _fires(cfg)                                       # absent keys: nothing to judge
    stages = cfg['protocols'][cfg['protocol']]['stages']
    fused = [st for st in stages if st.get('train_mode') == 'fused']
    other = [st for st in stages if st.get('train_mode') != 'fused']
    assert fused or other
    if fused:
        st = fused[0]
        st.setdefault('loss_coeffs', {}).setdefault('replay', {})['score'] = 0.1
        if not any(cfg.get('policy', {}).get('nf', {}).get(k) for k in ('rotate_frames', 'mirror_frames')):
            assert not [v for v in _fires(cfg) if st.get('name') in v.detail]
        st['loss_coeffs']['replay']['score'] = -1.0
        assert _fires(cfg)
        st['loss_coeffs']['replay']['score'] = 0.1
        st['loss_coeffs'].setdefault('bwd', {})['pathwise'] = 0.1
        assert _fires(cfg)
        del st['loss_coeffs']['bwd']['pathwise']
        st['loss_coeffs']['replay']['score_force'] = 'sideways'
        assert _fires(cfg)
        del st['loss_coeffs']['replay']
    if other:
        st = other[0]
        st.setdefault('loss_coeffs', {}).setdefault('bwd', {})['score'] = 0.1
        assert any('fused step' in v.detail for v in _fires(cfg))


def test_score_rows_scores_a_subset():
    from gflownet_losses import branch_score_term
    pol, c, x = _setup()
    seen = {}
    real = pol.score

    def spy(xx, cc, mb):
        seen['n'] = xx.shape[0]
        return real(xx, cc, mb)
    pol.score = spy
    t, st = branch_score_term(_bwd_coeffs(score=1.0, score_rows=4), pol, x, c, None, _reward_grad(x))
    assert seen['n'] == 4 and t.requires_grad
    branch_score_term(_bwd_coeffs(score=1.0, score_rows=0), pol, x, c, None, _reward_grad(x))
    assert seen['n'] == N


def test_descending_the_pathwise_loss_fits_a_gaussian_target():
    """Adam on the pathwise loss alone, toy flow, target log R = -20 |x - 0.5|^2 on the live rows: the loss falls from
    ~116 toward its optimum -log Z (~10 here) and log q sits near the target's own (~4.7). At lr 1e-3 the same toy
    diverges (the target is sharp); 3e-4 converges for both this estimator and the full reparameterised gradient."""
    pol = flow_policy(True, dead=DEAD)
    opt = torch.optim.Adam([p for p in pol.parameters() if p.requires_grad], lr=3e-4)
    c = torch.randn(64, 4, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    for _ in range(300):
        x, log_q = pol._pathwise_draw(64, pol._context(c, None, 64))
        loss, st = pathwise_rows(-20.0 * ((x - 0.5) ** 2).sum(-1), log_q, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 25.0 and 2.0 < float(st['pathwise_log_q']) < 8.0


def test_pathwise_entropy_zero_is_energy_descent_along_the_draw():
    """pathwise_entropy 0: the loss is -mean log R at the draw, its gradient d(-log R)/dx . dx/dtheta alone."""
    pol, c, _ = _setup()
    torch.manual_seed(3)
    x, log_q = pol._pathwise_draw(N, pol._context(c, None, N))
    params = [p for p in pol.sampler.parameters() if p.requires_grad]
    lr = _reward(x, None, None, False)
    loss, st = pathwise_rows(lr, log_q, x, entropy=0.0)
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    want = torch.autograd.grad(-lr.mean(), params, allow_unused=True)
    for p, a, b in zip(params, g, want):
        z = torch.zeros_like(p)
        assert torch.allclose(z if a is None else a, z if b is None else b, atol=1e-5)
