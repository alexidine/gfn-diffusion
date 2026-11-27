"""NF score matching and pathwise reverse KL (gflownet_losses.py::score_matching_rows / ::pathwise_term, NFPolicy.score /
.pathwise_draw; score trained by train.py::Modeller.fused_train_step outside the frac mix, pathwise added after the fused
backward at its dose by train.py::Modeller._apply_pathwise_grads). So:
  * the policy's score is d log q / d x (finite differences), zero on dead rows, and a loss on it trains the policy;
  * the score-matching row loss is the Huber of |g_q - g_R| over the live rows, a row with no force dropped;
  * the fresh pathwise draw is the draw sample() makes from the same base points with its log q and score; the
    off-policy draw regenerates stored rows (x' = x) with dx'/dtheta reaching the flow and the condition's encoder;
  * the term's gradient is the sticking-the-landing estimator (on- and off-policy, entropy 1 and 0); a row with no
    reward is dropped; the x clip, the row subsample and the per-row parameter cap act as stated; descending the term
    alone lowers fresh draws' energy on a toy target, on- and off-policy;
  * the backward loss stashes the off-policy term only on trained calls, at the row's own temperature; zero dose is the
    loss without the keys (no reward call, same RNG); the forward loss no longer makes a pathwise term;
  * the dose: a norm fraction is realised against this backward's own main gradient (mid-accumulation too), a
    coefficient carries the accumulation scale, and a fraction with no main gradient adds nothing;
  * the loader refuses malformed or inert settings.

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
from gflownet_losses import get_gfn_backward_loss, get_gfn_forward_loss, score_matching_rows  # noqa: E402

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
    loss, d = get_gfn_forward_loss(_fwd_coeffs(score=0.5, score_beta=0.0), torch.zeros(N, 12), pol,
                                   reward, None, None, torch.zeros(N), condition=c, report_losses=True)
    assert seen == {'calls': 1, 'keep': True}                     # one reward call serves TB and the force
    assert set(pol._aux_terms) == {'fwd_score'} and pol._aux_terms['fwd_score'].requires_grad
    assert {'score', 'score_cos'} <= set(d)
    pol._aux_terms = None
    with torch.no_grad():                                        # a readout arms nothing
        get_gfn_forward_loss(_fwd_coeffs(score=0.5), torch.zeros(N, 12), pol, reward, None, None,
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
        pol.pathwise_draw(torch.zeros(2, 4), None)


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
        if not any(cfg.get('policy', {}).get('nf', {}).get(k) for k in ('rotate_frames', 'mirror_frames')):
            assert not [v for v in _fires(cfg) if st.get('name') in v.detail]     # off-policy pathwise on bwd
        st['loss_coeffs']['bwd']['pathwise_norm_frac'] = 0.1
        assert any('one dose' in v.detail for v in _fires(cfg))
        st['loss_coeffs']['bwd'].update(pathwise=0.0, pathwise_row_cap=3.0)
        assert any('pathwise_rows 0' in v.detail for v in _fires(cfg))
        st['loss_coeffs']['bwd'].update(pathwise_rows=2.5)
        assert any('whole number' in v.detail for v in _fires(cfg))
        st['loss_coeffs']['bwd'].update(pathwise_rows=32, pathwise_norm_frac=-0.1)
        assert _fires(cfg)
        del st['loss_coeffs']['bwd']
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


# -- the pathwise reverse-KL term (gflownet_losses.py::pathwise_term, NFPolicy.pathwise_draw) -------------------------

def _pw(**on):
    from gflownet_losses import PATHWISE_DEFAULTS
    return {**PATHWISE_DEFAULTS, 'pathwise': 1.0, **on}


def _policy_params(pol):
    from gflownet_losses import policy_parameters
    return policy_parameters(pol)


def test_the_fresh_draw_is_samples_draw_with_its_log_q_and_score():
    pol, c, _ = _setup()
    torch.manual_seed(11)
    with torch.no_grad():
        x_ref = pol._pin_dead(pol._wrap_ang(pol.sampler.sample(N, c)))
    torch.manual_seed(11)
    x, log_q, g_q = pol.pathwise_draw(c, None)
    assert x.requires_grad and not log_q.requires_grad and not g_q.requires_grad
    assert torch.allclose(x.detach(), x_ref, atol=1e-4)
    assert torch.allclose(log_q, pol.sampler.log_prob(x.detach(), c), atol=1e-4)
    assert torch.allclose(g_q, pol.score(x.detach(), c, None).detach(), atol=1e-4)    # d log q / dx, fixed theta


def test_the_off_policy_draw_regenerates_the_stored_rows():
    """z = T^-1(x) without gradient, x' = T(z) with it: x' equals the stored x (wrapped, pinned) to float error and
    carries dx'/dtheta into the flow blocks and the condition's encoder."""
    pol, c, x = _setup()
    x2, log_q, _ = pol.pathwise_draw(c, None, x)
    assert torch.allclose(x2.detach(), x, atol=2e-4)
    assert torch.allclose(log_q, pol.sampler.log_prob(x, c), atol=1e-3)
    grads = torch.autograd.grad(x2.sum(), _policy_params(pol), allow_unused=True)
    names = [n for n, p in pol.named_parameters() if any(p is q for q in _policy_params(pol))]
    reached = {n.split('.')[0] for n, g in zip(names, grads) if g is not None and g.abs().sum() > 0}
    assert reached, 'no gradient reached the policy'


@pytest.mark.parametrize('off_policy', [False, True])
@pytest.mark.parametrize('entropy', [1.0, 0.0])
def test_the_term_is_the_sticking_the_landing_gradient(off_policy, entropy):
    """pathwise_term's gradient = d/dtheta mean(entropy log q_thetabar(x') - log R(x')) at the reparameterised x': the
    total derivative of log q(x'(theta)) minus its explicit derivative at fixed x, minus d mean log R(x'(theta))."""
    from gflownet_losses import pathwise_term
    pol, c, x = _setup()
    params = _policy_params(pol)
    torch.manual_seed(5)
    out = pathwise_term(_pw(pathwise_entropy=entropy), pol, _reward, c, None, torch.zeros(N),
                        x=x if off_policy else None)
    torch.manual_seed(5)
    x2, _, _ = pol.pathwise_draw(c, None, x if off_policy else None)
    lr = _reward(x2, None, None, False)

    def z(ts):
        return [torch.zeros_like(p) if t is None else t for p, t in zip(params, ts)]
    want = z(torch.autograd.grad(-lr.mean(), params, retain_graph=True, allow_unused=True))
    if entropy:
        total = z(torch.autograd.grad(pol.sampler.log_prob(x2, c).mean(), params, allow_unused=True))
        explicit = z(torch.autograd.grad(pol.sampler.log_prob(x2.detach(), c).mean(), params, allow_unused=True))
        want = [w + t - e for w, t, e in zip(want, total, explicit)]
    got = z(out['grads'])
    for g, w in zip(got, want):
        assert torch.allclose(g, w, atol=1e-4, rtol=1e-3)
    st = out['stats']
    assert st['grad_norm'] > 0 and st['bad_frac'] == 0.0 and st['rows'] == N
    assert st['log_r'] == pytest.approx(float(lr.mean()), abs=1e-3)
    assert ('regen_err' in st) == off_policy and st.get('regen_err', 0.0) < 2e-4


def test_a_row_with_no_reward_is_dropped_the_clip_caps_dl_dx_and_rows_subsample():
    from gflownet_losses import pathwise_term

    def reward(x, mol_batch, log_T, return_exp, keep_grads=False):
        r = _reward(x, mol_batch, log_T, return_exp)
        return torch.where(torch.arange(x.shape[0]) == 1, torch.full_like(r, float('nan')), r)
    pol, c, x = _setup()
    out = pathwise_term(_pw(), pol, reward, c, None, torch.zeros(N), x=x)
    assert out['stats']['bad_frac'] == pytest.approx(1 / N)
    clipped = pathwise_term(_pw(pathwise_grad_clip=1e-3), pol, _reward, c, None, torch.zeros(N), x=x)
    assert clipped['stats']['x_clipped_frac'] == 1.0 and clipped['stats']['grad_norm'] < out['stats']['grad_norm']
    seen = {}
    real = pol.pathwise_draw

    def spy(cc, mb, xx=None):
        seen['n'] = cc.shape[0]
        return real(cc, mb, xx)
    pol.pathwise_draw = spy
    sub = pathwise_term(_pw(pathwise_rows=4), pol, _reward, c, None, torch.zeros(N), x=x)
    assert seen['n'] == 4 and sub['stats']['rows'] == 4


def test_the_row_cap_bounds_each_rows_parameter_gradient():
    """pathwise_row_cap r: row i's parameter gradient J_i^T g_i is scaled to at most r x the median row norm, and the
    term is the mean of the scaled rows; at a cap no row reaches, the plain gradient."""
    from gflownet_losses import pathwise_param_grads
    pol, c, x = _setup()
    params = _policy_params(pol)
    x2, _, g_q = pol.pathwise_draw(c, None, x)
    g = g_q - _reward_grad(x2.detach())
    g[0] *= 1e3                                                    # one row dominates
    ok = torch.ones(N, dtype=torch.bool)
    rows = [torch.autograd.grad((x2[i] * g[i]).sum(), params, retain_graph=True, allow_unused=True) for i in range(N)]
    nu = torch.stack([sum((t.double() ** 2).sum() for t in r if t is not None).sqrt() for r in rows])
    keep = torch.clamp(2.0 * nu.median() / nu, max=1.0)
    want = [sum(keep[i] * (rows[i][j] if rows[i][j] is not None else 0) for i in range(N)) / N
            for j in range(len(params))]
    got, st = pathwise_param_grads(x2, g, ok, params, row_cap=2.0)
    for gg, w in zip(got, want):
        if gg is not None:
            assert torch.allclose(gg.double(), torch.as_tensor(w).double(), atol=1e-6, rtol=1e-4)
    assert st['row_top_share'] > 0.9 and 0 < st['row_capped_frac'] < 1
    plain, _ = pathwise_param_grads(pol.pathwise_draw(c, None, x)[0], g, ok, params)    # a fresh graph per call
    loose, _ = pathwise_param_grads(pol.pathwise_draw(c, None, x)[0], g, ok, params, row_cap=1e9)
    for a, b in zip(plain, loose):
        if a is not None:
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('off_policy', [False, True])
def test_descending_the_term_lowers_the_energy_of_fresh_draws(off_policy):
    """Adam on the pathwise term alone, toy flow, target log R = -20 |x - 0.5|^2 (energy 20 |x - 0.5|^2): fresh draws'
    mean energy falls. On-policy: fresh rows each step; off-policy: a fixed set of stored rows drawn once from the
    initial policy, regenerated every step (the regenerated rows move, the stored ones do not)."""
    from gflownet_losses import pathwise_term
    pol = flow_policy(True, dead=DEAD)
    params = _policy_params(pol)
    opt = torch.optim.Adam(params, lr=3e-4)
    n = 64
    c = torch.randn(n, 4, generator=torch.Generator().manual_seed(1))

    def reward(x, mol_batch, log_T, return_exp, keep_grads=False):
        return -20.0 * ((x - 0.5) ** 2).sum(-1)
    torch.manual_seed(0)
    with torch.no_grad():
        stored = pol._pin_dead(pol._wrap_ang(pol.sampler.sample(n, c)))
        e0 = float(-reward(stored, None, None, False).mean())
    for _ in range(150):
        out = pathwise_term(_pw(), pol, reward, c, None, torch.zeros(n), x=stored if off_policy else None)
        opt.zero_grad()
        for p, g in zip(params, out['grads']):
            p.grad = None if g is None else g
        opt.step()
    with torch.no_grad():
        e1 = float(-reward(pol._pin_dead(pol._wrap_ang(pol.sampler.sample(n, c))), None, None, False).mean())
    assert e1 < 0.7 * e0, (e0, e1)


def test_the_backward_loss_stashes_the_off_policy_term_on_trained_calls_only():
    pol, c, x = _setup()
    calls = {'n': 0}

    def reward(xx, mol_batch, log_T, return_exp, keep_grads=False):
        calls['n'] += 1
        return _reward(xx, mol_batch, log_T, return_exp)
    log_r = _reward(x, None, None, False)
    pol._pathwise_grads = None
    get_gfn_backward_loss(_bwd_coeffs(mle=1, pathwise_norm_frac=0.1, pathwise_rows=4), x, pol, log_r, None, None,
                          condition=c, repeats=1, live_stash='replay', log_reward_fn=reward,
                          log_T_tensor=torch.zeros(N))
    assert set(pol._pathwise_grads) == {'replay'} and calls['n'] == 1
    term = pol._pathwise_grads['replay']
    assert term['stats']['rows'] == 4 and term['stats']['regen_err'] < 2e-4 and term['settings']['pathwise'] == 0.0
    pol._pathwise_grads = None
    get_gfn_backward_loss(_bwd_coeffs(mle=1, pathwise=1.0), x, pol, log_r, None, None, condition=c, repeats=1,
                          live_stash=None, log_reward_fn=reward, log_T_tensor=torch.zeros(N))
    with torch.no_grad():
        get_gfn_backward_loss(_bwd_coeffs(mle=1, pathwise=1.0), x, pol, log_r, None, None, condition=c, repeats=1,
                              live_stash='bwd', log_reward_fn=reward, log_T_tensor=torch.zeros(N))
    assert pol._pathwise_grads is None and calls['n'] == 1          # a probe or a readout arms nothing
    mb = SimpleNamespace(draw_log_T=torch.full((N,), 0.5))           # the row's own temperature wins
    seen = {}

    def reward_t(xx, mol_batch, log_T, return_exp, keep_grads=False):
        seen['log_T'] = log_T
        return _reward(xx, None, log_T, return_exp)
    get_gfn_backward_loss(_bwd_coeffs(mle=1, pathwise=1.0), x, pol, log_r, None, mb, condition=c, repeats=1,
                          live_stash='bwd', log_reward_fn=reward_t, log_T_tensor=torch.zeros(N))
    assert torch.allclose(seen['log_T'], torch.full((N,), 0.5))
    pol._pathwise_grads = None
    with pytest.raises(ValueError, match='one dose'):
        get_gfn_backward_loss(_bwd_coeffs(mle=1, pathwise=1.0, pathwise_norm_frac=0.1), x, pol, log_r, None, None,
                              condition=c, repeats=1, live_stash='bwd', log_reward_fn=reward,
                              log_T_tensor=torch.zeros(N))


def test_zero_dose_is_the_loss_without_the_keys():
    """pathwise 0 and pathwise_norm_frac 0 (whatever the other pathwise keys say): the same loss, no reward call, no
    stash, and the global RNG where the keyless call leaves it."""
    pol, c, x = _setup()
    log_r = _reward(x, None, None, False)

    def reward(*a, **k):
        raise AssertionError('no reward call at zero dose')
    torch.manual_seed(7)
    pol._pathwise_grads = None
    l0, _ = get_gfn_backward_loss(_bwd_coeffs(mle=1), x, pol, log_r, None, None, condition=c, repeats=1,
                                  live_stash='bwd', log_reward_fn=reward, log_T_tensor=torch.zeros(N))
    s0 = torch.get_rng_state()
    torch.manual_seed(7)
    l1, _ = get_gfn_backward_loss(_bwd_coeffs(mle=1, pathwise=0.0, pathwise_norm_frac=0.0, pathwise_rows=3,
                                              pathwise_row_cap=2.0, pathwise_entropy=0.0, pathwise_grad_clip=5.0),
                                  x, pol, log_r, None, None, condition=c, repeats=1, live_stash='bwd',
                                  log_reward_fn=reward, log_T_tensor=torch.zeros(N))
    assert torch.equal(l0, l1) and torch.equal(s0, torch.get_rng_state()) and pol._pathwise_grads is None


def test_the_forward_loss_no_longer_makes_the_pathwise_term():
    """The forward branch's pathwise term draws its own rows every fused step (train.py::Modeller.
    _pathwise_fresh_step); get_gfn_forward_loss ignores the keys: one ordinary (no-grad) reward call, nothing armed."""
    pol, c, _ = _setup()
    seen = []

    def reward(x, mol_batch, log_T, return_exp, keep_grads=False):
        seen.append(keep_grads)
        return _reward(x, mol_batch, log_T, return_exp)
    pol._aux_terms = None
    get_gfn_forward_loss(_fwd_coeffs(pathwise=2.0, pathwise_norm_frac=0.0), torch.zeros(N, 12), pol, reward, None,
                         None, torch.zeros(N), condition=c, report_losses=True)
    assert seen == [False] and pol._aux_terms is None and getattr(pol, '_pathwise_grads', None) is None


# -- the dose (train.py::Modeller._apply_pathwise_grads) -----------------------------------------------------------

def _dose_case(seed=0):
    from gflownet_losses import PATHWISE_DEFAULTS
    g = torch.Generator().manual_seed(seed)
    params = [torch.nn.Parameter(torch.zeros(3, 2)), torch.nn.Parameter(torch.zeros(4))]
    main = [torch.randn(3, 2, generator=g), torch.randn(4, generator=g)]
    term = [torch.randn(3, 2, generator=g), None]
    return params, main, term, PATHWISE_DEFAULTS


def _norm(ts):
    return float(torch.sqrt(sum((t.double() ** 2).sum() for t in ts if t is not None)))


@pytest.mark.parametrize('frac', [0.1, 0.3])
def test_the_norm_fraction_is_realised_on_the_policy_parameters(frac):
    from train import Modeller
    params, main, term, d = _dose_case()
    for p, m in zip(params, main):
        p.grad = m.clone()
    ns = SimpleNamespace()
    pend = {'bwd': {'params': params, 'grads': term, 'settings': {**d, 'pathwise_norm_frac': frac},
                    'stats': {'grad_norm': _norm(term), 'loss': 1.0}}}
    Modeller._apply_pathwise_grads(ns, pend, None, 0.25)          # aux_scale does not touch a fraction
    added = [p.grad - m for p, m in zip(params, main)]
    assert _norm(added) == pytest.approx(frac * _norm(main), rel=1e-6)
    assert torch.allclose(added[0] / added[0].norm(), term[0] / term[0].norm(), atol=1e-6)
    out = Modeller._drain_pathwise_running(ns)
    assert out['pathwise/bwd_frac'] == pytest.approx(frac, rel=1e-6)
    assert out['pathwise/bwd_main_grad_norm'] == pytest.approx(_norm(main), rel=1e-6)
    assert out['pathwise/bwd_grad_norm'] == pytest.approx(_norm(term), rel=1e-6)
    assert out['pathwise/bwd_steps'] == 1.0 and Modeller._drain_pathwise_running(ns) == {}


def test_mid_accumulation_the_main_norm_is_this_backwards_own_and_a_coefficient_carries_the_scale():
    from train import Modeller
    params, main, term, d = _dose_case(1)
    held = [torch.full_like(m, 5.0) for m in main]                  # gradient from an earlier accumulation sub-step
    for p, m, h in zip(params, main, held):
        p.grad = m + h
    ns = SimpleNamespace()
    pend = {'fwd': {'params': params, 'grads': term, 'settings': {**d, 'pathwise_norm_frac': 0.2},
                    'stats': {'grad_norm': _norm(term)}}}
    Modeller._apply_pathwise_grads(ns, pend, [h.clone() for h in held], 1.0)
    assert _norm([p.grad - m - h for p, m, h in zip(params, main, held)]) == pytest.approx(0.2 * _norm(main), rel=1e-6)
    params, main, term, d = _dose_case(2)
    for p, m in zip(params, main):
        p.grad = m.clone()
    pend = {'replay': {'params': params, 'grads': term, 'settings': {**d, 'pathwise': 0.5},
                       'stats': {'grad_norm': _norm(term)}}}
    Modeller._apply_pathwise_grads(ns, pend, None, 0.25)
    assert torch.allclose(params[0].grad - main[0], 0.125 * term[0])
    assert torch.equal(params[1].grad, main[1])                       # an unreached parameter keeps its gradient


def test_a_fraction_with_no_main_gradient_adds_nothing():
    from train import Modeller
    params, _, term, d = _dose_case(3)
    ns = SimpleNamespace()
    pend = {'bwd': {'params': params, 'grads': term, 'settings': {**d, 'pathwise_norm_frac': 0.1},
                    'stats': {'grad_norm': _norm(term)}}}
    Modeller._apply_pathwise_grads(ns, pend, None, 1.0)
    assert all(p.grad is None for p in params)


def _step_loss_stub():
    import types
    from train import Modeller

    class Pol(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.forward_policy = torch.nn.Linear(3, 2)
            self.conditions_embedding_model = torch.nn.Linear(2, 2)
            self.flow_model = torch.nn.Linear(2, 1)
            self._aux_terms = self._pathwise_grads = self._mle_parts = None
    pol = Pol()
    guard = SimpleNamespace(threshold=lambda step_type: 1e30, observe=lambda step_type, v: None)
    ns = SimpleNamespace(gfn_model=pol, _pathwise_pending=None, step_ind=10, device='cpu', grad_guard=guard,
                         optimizers={'fused': torch.optim.SGD(pol.parameters(), lr=0.0), 'bwd': None},
                         args=SimpleNamespace(), _grad_nonfinite=0)
    for name in ('_apply_pathwise_grads', '_drain_pathwise_running', '_submodel_grad_norms'):
        setattr(ns, name, types.MethodType(getattr(Modeller, name), ns))
    return ns, pol, types.MethodType(Modeller.step_loss, ns)


def test_step_loss_adds_the_pathwise_gradient_after_the_fused_backward_at_its_fraction():
    """train.py::Modeller.step_loss end to end on a stub: the fused backward, then the pending term at its norm fraction
    of that backward's policy gradient (the level head untouched), then the clip and the optimizer step; the 10-step
    readouts carry the realised fraction. A pending term on another step type raises."""
    from gflownet_losses import PATHWISE_DEFAULTS, policy_parameters
    ns, pol, step_loss = _step_loss_stub()
    params = policy_parameters(pol)
    x = torch.randn(5, 3)

    def loss_fn():
        return (pol.conditions_embedding_model(pol.forward_policy(x)) ** 2).sum() + pol.flow_model(torch.ones(1, 2)).sum()
    main = torch.autograd.grad(loss_fn(), params)
    term = [torch.randn_like(p) for p in params]
    ns._pathwise_pending = {'bwd': {'params': params, 'grads': term, 'settings': {**PATHWISE_DEFAULTS,
                                                                                  'pathwise_norm_frac': 0.2},
                                    'stats': {'grad_norm': _norm(term)}}}
    step_loss('fused', loss_fn())
    added = [p.grad - m for p, m in zip(params, main)]
    assert _norm(added) == pytest.approx(0.2 * _norm(main), rel=1e-5)
    assert ns._pathwise_pending is None
    assert ns._last_grad_norms['pathwise/bwd_frac'] == pytest.approx(0.2, rel=1e-5)
    assert pol.flow_model.weight.grad is not None                 # the fused loss still trains the level head
    ns._pathwise_pending = {'bwd': {'params': params, 'grads': term, 'settings': {**PATHWISE_DEFAULTS, 'pathwise': 1.0},
                                    'stats': {'grad_norm': _norm(term)}}}
    with pytest.raises(RuntimeError, match='pending'):
        step_loss('bwd', loss_fn())
