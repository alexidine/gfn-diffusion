"""Force terms in the kernel means (GFN.init_force_drift): P_F and P_B each take an optional
term gate(t) * tame(variance * force), the force coming from an installed provider.

What is pinned here, all on toy energies on CPU:
  * with both options None nothing is built and nothing is called;
  * the term changes each kernel's log density by exactly what a shift of
    gate * variance * force of a Gaussian's mean must (sign and mobility);
  * a trajectory scores the same whether it was rolled forward, rolled backward or
    replayed, with stored forces or computed ones, on linear, wrapped and dead coordinates;
  * log Z by importance sampling from an untrained sampler is unbiased for any pair of
    gates, and is NOT when P_B reads the force at the wrong state (the test has power);
  * the gates train in their policy's parameter group and P_B's gate freezes with P_B.
"""
import math
import os
import sys

import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _root in (os.path.dirname(_here),
              os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from energy_sampling.models.gfn import GFN  # noqa: E402
from energy_sampling.utils import uniform_discretizer  # noqa: E402

pytestmark = pytest.mark.fast

CPU = torch.device('cpu')
NET = dict(s_emb_dim=32, conditions_dim=1, harmonics_dim=8, t_dim=8, t_hidden_dim=32, s_hidden_dim=32,
           s_layers=2, policy_hidden_dim=32, policy_layers=2, flow_hidden_dim=16, flow_layers=2,
           learned_variance=True, learn_pb=True, conditional=False, device=CPU)
LAYOUTS = {
    'linear': dict(dim=4, t_scale=0.25, do_periodic_angles=False, hold_dead_latent_rows=False),
    'crystal': dict(dim=12, t_scale=0.3, max_z_prime=1, do_periodic_angles=True, pb_exact_reversal=True),
    'crystal_dplr_dead_schedule': dict(dim=12, t_scale=0.3, max_z_prime=1, do_periodic_angles=True,
                                       pb_exact_reversal=True, dplr_rank=2, periodic_centroids=True,
                                       periodic_centroid_axes=(1,), dead_latent_rows=(3, 5),
                                       t_scale_ratio=0.3),
    'crystal_single_image': dict(dim=12, t_scale=0.3, max_z_prime=1, do_periodic_angles=True,
                                 pb_exact_reversal=False),
}


class ToyEnergy:
    """E(x) = sum_lin (x - mu)^2 / (2 s^2) - kappa * sum_ang cos(pi (x - m)), in kT.

    Period 2 in the wrapped coordinates, as the sampler's are. `mu` may be given per row
    through the context, which is how a per-system target reaches a provider.
    """

    def __init__(self, gfn, mu=0.2, s=0.6, kappa=1.0, m=0.3):
        # dead coordinates are given a (large) force too, which the sampler must ignore
        self.lin = torch.cat([gfn.lin_idx, gfn.dead_idx])
        self.ang = gfn.ang_idx
        self.mu, self.s, self.kappa, self.m = mu, s, kappa, m
        self.calls = 0

    def energy(self, x, ctx=None):
        mu = self.mu if ctx is None else ctx['mu'].unsqueeze(1)
        e = ((x.index_select(1, self.lin) - mu) ** 2).sum(1) / (2 * self.s ** 2)
        if self.ang.numel():
            e = e - self.kappa * torch.cos(math.pi * (x.index_select(1, self.ang) - self.m)).sum(1)
        return e

    def log_z(self, n_lin, n_ang):
        return (0.5 * n_lin * math.log(2 * math.pi * self.s ** 2)
                + n_ang * math.log(2 * float(torch.special.i0(torch.tensor(self.kappa, dtype=torch.float64)))))

    def force(self, state, ctx, create_graph):
        self.calls += 1
        with torch.enable_grad():
            x = state if state.requires_grad else state.detach().requires_grad_(True)
            return -torch.autograd.grad(self.energy(x, ctx).sum(), x, create_graph=create_graph)[0]


def build(layout='linear', seed=0, zero_init=False, **force_kw):
    torch.manual_seed(seed)
    return GFN(**NET, **LAYOUTS[layout], zero_init=zero_init, **force_kw).eval()


def randomise_gates(g, seed=7, scale=0.3):
    """Give the learned gates a time and coordinate dependence, as training would."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for pol in (g.forward_policy, g.backward_policy):
            if hasattr(pol, 'force_gate'):
                for p in pol.force_gate.parameters():
                    p.add_(scale * torch.randn_like(p))


def disc(T):
    return lambda b: uniform_discretizer(b, T)


def test_off_builds_nothing_and_calls_nothing():
    def never(*a):
        raise AssertionError('the provider was called with no force term configured')

    for layout in LAYOUTS:
        plain = build(layout)
        with_provider = build(layout)
        with_provider.install_drift_force(never)
        assert not plain.force_on
        assert sorted(plain.state_dict()) == sorted(with_provider.state_dict())
        assert not any('force_gate' in k for k in plain.state_dict())
        outs = []
        for g in (plain, with_provider):
            torch.manual_seed(1)
            s, pf, pb, _ = g.get_traj_fwd(torch.zeros(8, g.dim), disc(6), None, None, None)
            torch.manual_seed(2)
            bs, bpf, bpb, _ = g.get_traj_bwd(s[:, -1], disc(6), None, None)
            _, rpf, rpb, _ = g.get_traj_replay(s, disc(6), None, None)
            outs.append((s, pf, pb, bs, bpf, bpb, rpf, rpb))
        assert all(torch.equal(a, b) for a, b in zip(*outs))


def test_missing_provider_and_step_compile_are_refused():
    g = build(force_drift_fwd=0.5)
    with pytest.raises(RuntimeError, match='install_drift_force'):
        g.get_traj_fwd(torch.zeros(4, g.dim), disc(4), None, None, None)
    with pytest.raises(ValueError, match="compile_policy: 'step'"):
        g.compile_step_kernels()
    with pytest.raises(ValueError, match='max_sigma'):
        build(force_drift_fwd=0.5, force_drift_max_sigma=0.0)
    with pytest.raises(ValueError, match='t_min'):
        build(force_drift_fwd=0.5, force_drift_t_min=1.0)


@pytest.mark.parametrize('a_f,a_b', [(0.7, None), (None, 0.4), (1.0, -0.5)])
def test_log_density_change_is_that_of_a_mean_shift(a_f, a_b):
    """Untamed, fixed gates, linear coordinates: against the same network without the term,

        d log P_F = a_F r_F . F(x)  - a_F^2/2 sum(v_F F(x)^2),   r_F = x' - x - dt mu(x)
        d log P_B = a_B r_B . F(x') - a_B^2/2 sum(v_B F(x')^2),  r_B = x - (1 - c k) x'

    v the kernel's own step variance. A positive gate therefore raises the density of a
    step whose residual points along the force, in either kernel.
    """
    T, B = 6, 32
    off = build(seed=3)
    on = build(seed=3, force_drift_fwd=a_f, force_drift_bwd=a_b, force_drift_learned=False,
               force_drift_max_sigma=None)
    on.load_state_dict(off.state_dict())          # fixed gates carry no parameters
    energy = ToyEnergy(on)
    on.install_drift_force(energy.force)

    torch.manual_seed(4)
    traj = torch.randn(B, T + 1, off.dim) * 0.4
    traj[:, 0] = 0.0
    ts = uniform_discretizer(B, T)
    with torch.no_grad():
        _, pf0, pb0, _ = off.get_traj_replay(traj, disc(T), None, None)
        _, pf1, pb1, _ = on.get_traj_replay(traj, disc(T), None, None)
        for i in range(T):
            x, y, dts = traj[:, i], traj[:, i + 1], ts[:, i + 1] - ts[:, i]
            mu, _, d, _, _, _ = off._forward_kernel(x, ts[:, i], None, ts[:, i + 1], dts)
            if a_f is not None:
                f, v = energy.force(x, None, False), dts[:, None] * d
                want = a_f * ((y - x - dts[:, None] * mu) * f).sum(1) - 0.5 * a_f ** 2 * (v * f ** 2).sum(1)
                assert torch.allclose(pf1[:, i] - pf0[:, i], want, atol=2e-4, rtol=1e-4)
            else:
                assert torch.equal(pf1[:, i], pf0[:, i])
            if a_b is not None and i > 0:
                kappa, logvar_corr = off.fwd_get_back_correction(None, off.expand_state_for_policy(y), ts[:, i + 1])
                c = off.var_drift_coeff(ts[:, i], ts[:, i + 1], dts)[:, None]
                v = (logvar_corr + off.var_log_rate(ts[:, i], ts[:, i + 1], dts)).exp() \
                    * off.var_bridge_step(ts[:, i], ts[:, i + 1], dts)[:, None]
                f = energy.force(y, None, False)
                want = a_b * ((x - (1 - c * kappa) * y) * f).sum(1) - 0.5 * a_b ** 2 * (v * f ** 2).sum(1)
                assert torch.allclose(pb1[:, i] - pb0[:, i], want, atol=2e-4, rtol=1e-4)
            else:
                assert torch.equal(pb1[:, i], pb0[:, i])   # the step into the source has no P_B term


@pytest.mark.parametrize('layout', list(LAYOUTS))
@pytest.mark.parametrize('t_min', [0.0, 0.4])
def test_forward_backward_and_replay_score_alike(layout, t_min):
    T, B = 10, 48
    g = build(layout, force_drift_fwd=0.6, force_drift_bwd=0.3, force_drift_t_min=t_min)
    randomise_gates(g)
    energy = ToyEnergy(g)
    g.install_drift_force(energy.force)
    ctx = {'mu': torch.linspace(-0.3, 0.3, B)}

    with torch.no_grad():
        torch.manual_seed(1)
        s, pf, pb, _ = g.get_traj_fwd(torch.zeros(B, g.dim), disc(T), None, None, None, drift_context=ctx)
        rs, rpf, rpb, _ = g.get_traj_replay(s, disc(T), None, None, drift_context=ctx)
        assert torch.equal(rs, s)
        assert torch.allclose(rpf, pf, atol=1e-4) and torch.allclose(rpb, pb, atol=1e-4)

        torch.manual_seed(2)
        bs, bpf, bpb, _ = g.get_traj_bwd(s[:, -1], disc(T), None, None, drift_context=ctx)
        _, rpf, rpb, _ = g.get_traj_replay(bs, disc(T), None, None, drift_context=ctx)
        # backward log-probs are listed from the terminal step down
        assert torch.allclose(rpf, bpf.flip(1), atol=1e-4) and torch.allclose(rpb, bpb.flip(1), atol=1e-4)

        # stored forces stand in for the provider on a fixed replay
        ts = uniform_discretizer(B, T)
        stored = torch.stack([energy.force(s[:, j], ctx, False) for j in range(T + 1)], dim=1)
        calls = energy.calls
        _, spf, spb, _ = g.get_traj_replay(s, disc(T), None, None, state_forces=stored)
        assert energy.calls == calls, 'stored forces were given and the provider was still called'
        assert torch.allclose(spf, pf, atol=1e-4) and torch.allclose(spb, pb, atol=1e-4)

        # the term is really there: the same weights without it score differently
        bare = build(layout)
        bare.load_state_dict(g.state_dict(), strict=False)
        _, npf, npb, _ = bare.get_traj_replay(s, disc(T), None, None)
        in_f = ts[0, :-1] >= t_min                # P_F reads the state the step leaves
        in_b = ts[0, 1:] >= t_min                 # P_B the state it is conditioned on,
        in_b[0] = False                           # and has no term on the step into the source
        assert (npf - pf)[:, in_f].abs().max() > 1e-3 and (npb - pb)[:, in_b].abs().max() > 1e-3
        # and a state before the window carries none
        assert torch.allclose(npf[:, ~in_f], pf[:, ~in_f], atol=1e-5)
        assert torch.allclose(npb[:, ~in_b], pb[:, ~in_b], atol=1e-5)
    assert g.dead_invariant_violation(s) == 0.0 and g.dead_invariant_violation(bs) == 0.0


def test_each_state_force_is_computed_once():
    T, B = 8, 4
    g = build(force_drift_fwd=0.5, force_drift_bwd=0.5)
    energy = ToyEnergy(g)
    g.install_drift_force(energy.force)
    with torch.no_grad():
        s, *_ = g.get_traj_fwd(torch.zeros(B, g.dim), disc(T), None, None, None)
        assert energy.calls == T + 1
        energy.calls = 0
        g.get_traj_replay(s, disc(T), None, None)
        assert energy.calls == T + 1
        energy.calls = 0
        g.get_traj_bwd(s[:, -1], disc(T), None, None)
        assert energy.calls == T + 1

    only_f = build(force_drift_fwd=0.5)
    energy = ToyEnergy(only_f)
    only_f.install_drift_force(energy.force)
    with torch.no_grad():
        only_f.get_traj_fwd(torch.zeros(B, only_f.dim), disc(T), None, None, None)
    assert energy.calls == T, 'P_F alone never reads the terminal state'

    windowed = build(force_drift_fwd=0.5, force_drift_bwd=0.5, force_drift_t_min=0.5)
    energy = ToyEnergy(windowed)
    windowed.install_drift_force(energy.force)
    with torch.no_grad():
        windowed.get_traj_fwd(torch.zeros(B, windowed.dim), disc(T), None, None, None)
    assert energy.calls == T // 2 + 1, 'states before t_min must not reach the provider'


class _WrongStatePB(GFN):
    """P_B's force read at the state being SCORED instead of the one conditioned on: no
    longer a normalised kernel of the earlier state. The control for the log Z test."""

    def _eval_pb_logprob(self, condition_embedding, current_state, next_state, dts,
                         t_prev, t_next, is_first, fallback_logpf, force_next=None):
        if force_next is not None:
            force_next = self._state_force(current_state, t_next, None)
        return super()._eval_pb_logprob(condition_embedding, current_state, next_state, dts,
                                        t_prev, t_next, is_first, fallback_logpf, force_next)


def _log_z_by_importance_sampling(g, energy, n=40000, T=16, seed=11):
    torch.manual_seed(seed)
    with torch.no_grad():
        s, pf, pb, _ = g.get_traj_fwd(torch.zeros(n, g.dim), disc(T), None, None, None)
        log_w = (-energy.energy(s[:, -1]) + pb.sum(1) - pf.sum(1)).double()
    log_z = torch.logsumexp(log_w, 0) - math.log(n)
    w = (log_w - log_w.max()).exp()
    stderr = float(w.std() / w.mean() / math.sqrt(n))
    ess = float(w.sum() ** 2 / (w ** 2).sum() / n)
    return float(log_z), stderr, ess


IS_LAYOUT = dict(dim=4, t_scale=0.25, angular_mask=[False, False, False, True],
                 do_periodic_angles=False, hold_dead_latent_rows=False)


NARROW = dict(mu=0.15, s=0.35, kappa=2.5, m=0.3)      # narrower than the reference process
WIDE = dict(mu=0.15, s=0.6, kappa=1.0, m=0.3)         # wider than it


@pytest.mark.parametrize('a_f,a_b,max_sigma,target', [
    (None, None, 2.0, NARROW),      # the reference: no force term
    (0.4, None, 2.0, NARROW),
    (0.3, 0.3, None, NARROW),       # untamed
    (0.4, 0.3, 0.5, NARROW),        # tamed hard enough to bite
    (None, 0.4, 2.0, WIDE),
    (None, -0.4, 2.0, WIDE),        # P_B uphill
])
def test_log_z_is_unbiased_for_any_gates(a_f, a_b, max_sigma, target):
    """An untrained sampler with any force terms is still a pair of normalised kernels, so
    mean(R * P_B / P_F) over its rollouts is Z. Three linear coordinates and one wrapped
    (the mixture P_B), a target whose Z is known in closed form. The target is chosen per
    case only so that the weights are even enough for the standard error to be trusted."""
    torch.manual_seed(0)
    g = GFN(**NET, **IS_LAYOUT, zero_init=True, force_drift_fwd=a_f, force_drift_bwd=a_b,
            force_drift_max_sigma=max_sigma, force_drift_t_min=0.25).eval()
    randomise_gates(g, scale=0.05)
    energy = ToyEnergy(g, **target)
    g.install_drift_force(energy.force)
    log_z, stderr, ess = _log_z_by_importance_sampling(g, energy)
    truth = energy.log_z(3, 1)
    assert ess > 0.2, f'weights too uneven for the standard error to mean anything (ESS {ess:.2f})'
    assert abs(log_z - truth) < 4 * stderr + 2e-3, (log_z, truth, stderr, ess)


def test_log_z_test_detects_a_kernel_that_is_not_normalised():
    torch.manual_seed(0)
    g = _WrongStatePB(**NET, **IS_LAYOUT, zero_init=True, force_drift_fwd=None, force_drift_bwd=0.4,
                      force_drift_t_min=0.25).eval()
    energy = ToyEnergy(g, **WIDE)
    g.install_drift_force(energy.force)
    log_z, stderr, ess = _log_z_by_importance_sampling(g, energy)
    assert abs(log_z - energy.log_z(3, 1)) > 10 * stderr + 0.02, (log_z, energy.log_z(3, 1), stderr, ess)


def test_tame_bounds_the_step_and_keeps_its_direction():
    g = build(force_drift_fwd=1.0, force_drift_learned=False, force_drift_max_sigma=1.5)
    big = lambda state, ctx, create_graph: torch.full_like(state, 1e4) * torch.arange(1, state.shape[1] + 1)
    g.install_drift_force(big)
    B, T = 5, 4
    ts = uniform_discretizer(B, T)
    dts = ts[:, 2] - ts[:, 1]
    x = torch.randn(B, g.dim) * 0.1
    with torch.no_grad():
        mu0, logvar, d, _, _, _ = g._forward_kernel(x, ts[:, 1], None, ts[:, 2], dts)
        f = g._state_force(x, ts[:, 1], None)
        mu1 = g._forward_kernel(x, ts[:, 1], None, ts[:, 2], dts, f)[0]
    step = dts[:, None] * (mu1 - mu0)
    rms = (step / (dts[:, None] * d).sqrt()).pow(2).mean(1).sqrt()
    assert (rms <= 1.5 + 1e-4).all() and (rms > 1.49).all()
    cos = torch.nn.functional.cosine_similarity(step, d * f, dim=1)
    assert (cos > 1 - 1e-5).all()


def test_nonfinite_force_rows_are_dropped_and_counted():
    g = build(force_drift_fwd=0.5, force_drift_bwd=0.5)
    energy = ToyEnergy(g)

    def flaky(state, ctx, create_graph):
        f = energy.force(state, ctx, create_graph).clone()
        f[0, 1] = float('nan')
        f[1, 0] = float('inf')
        return f

    g.install_drift_force(flaky)
    with torch.no_grad():
        s, pf, pb, _ = g.get_traj_fwd(torch.zeros(6, g.dim), disc(5), None, None, None)
    assert torch.isfinite(s).all() and torch.isfinite(pf).all() and torch.isfinite(pb).all()
    assert g.force_nonfinite_rows() == 2 * 6      # two rows at each of the six states


def test_gates_train_with_their_policy_and_freeze_with_pb():
    T, B = 6, 16
    g = build(force_drift_fwd=0.0, force_drift_bwd=0.0).train()
    energy = ToyEnergy(g)
    g.install_drift_force(energy.force)
    fwd_ids = {id(p) for p in g.forward_policy.parameters()}
    bwd_ids = {id(p) for p in g.backward_policy.parameters()}
    f_gate = list(g.forward_policy.force_gate.parameters())
    b_gate = list(g.backward_policy.force_gate.parameters())
    assert f_gate and all(id(p) in fwd_ids for p in f_gate)
    assert b_gate and all(id(p) in bwd_ids for p in b_gate)

    torch.manual_seed(5)
    traj = torch.randn(B, T + 1, g.dim) * 0.4
    traj[:, 0] = 0.0
    _, pf, pb, _ = g.get_traj_replay(traj, disc(T), None, None)
    (pf.sum() - pb.sum()).backward()
    # a gate at zero still has a gradient: it is what lets the term switch itself on
    assert f_gate[-1].grad.abs().max() > 0 and b_gate[-1].grad.abs().max() > 0

    a_f, a_b = g.force_gate_values(uniform_discretizer(1, T)[0])
    assert a_f.shape == (T + 1, g.dim) and torch.equal(a_f, torch.zeros_like(a_f)) and torch.equal(a_b, a_f)

    with torch.no_grad():
        before = g.get_traj_replay(traj, disc(T), None, None)[2]
    g.freeze_backward_policy()
    assert any('force_gate' in k for k in g.pb_snapshot_state())
    randomise_gates(g, scale=1.0)                 # moves the LIVE gates only
    _, pf2, pb2, _ = g.get_traj_replay(traj, disc(T), None, None)
    assert torch.allclose(pb2, before, atol=1e-6), 'a frozen P_B must keep its gate'
    assert not pb2.requires_grad and pf2.requires_grad
    assert not torch.allclose(pf2, pf.detach(), atol=1e-4), 'the live forward gate did move'


def test_differentiable_force_carries_the_state_gradient():
    """One step from x0 with a zero-output policy and E = |x|^2 / (2 s^2): the mean is
    x0 (1 - a dt sigma^2 / s^2), so d x1 / d x0 is that factor when the force is
    differentiable and 1 when it is a constant of the step."""
    s2, a, t_scale = 0.36, 0.8, 0.25
    for differentiable, want in ((True, 1 - a * 1.0 * t_scale / s2), (False, 1.0)):
        g = build(zero_init=True, force_drift_fwd=a, force_drift_learned=False, force_drift_max_sigma=None,
                  force_drift_differentiable=differentiable)
        energy = ToyEnergy(g, mu=0.0, s=math.sqrt(s2))
        g.install_drift_force(energy.force)
        x0 = torch.randn(3, g.dim) * 0.2
        torch.manual_seed(1)
        s, *_ = g.get_traj_fwd(x0, disc(1), None, None, None, detach_traj=False)
        grad = torch.autograd.grad(s[:, 1].sum(), x0)[0]
        assert torch.allclose(grad, torch.full_like(grad, want), atol=1e-5), (differentiable, grad)


@pytest.mark.parametrize('layout', ['linear', 'crystal'])
def test_gradient_checkpointed_steps_reproduce_values_and_gradients(layout):
    """Each step is a pure function of its inputs with the force passed in or recomputed
    inside, so checkpointing the trajectory changes neither the values nor the gradients."""
    T, B = 8, 24
    g = build(layout, force_drift_fwd=0.5, force_drift_bwd=0.3).train()
    randomise_gates(g)
    energy = ToyEnergy(g)
    g.install_drift_force(energy.force)
    ctx = {'mu': torch.linspace(-0.2, 0.2, B)}
    params = [p for p in g.parameters() if p.requires_grad]
    results = []
    for checkpointed in (False, True):
        g.traj_checkpoint = checkpointed
        torch.manual_seed(21)
        s, pf, pb, _ = g.get_traj_fwd(torch.zeros(B, g.dim), disc(T), None, None, None, drift_context=ctx)
        torch.manual_seed(22)
        _, bpf, bpb, _ = g.get_traj_bwd(s[:, -1].detach(), disc(T), None, None, drift_context=ctx)
        _, rpf, rpb, _ = g.get_traj_replay(s.detach(), disc(T), None, None, drift_context=ctx)
        loss = ((pf.sum(1) - pb.sum(1)).pow(2).mean() + (bpf.sum(1) - bpb.sum(1)).pow(2).mean()
                + (rpf.sum(1) - rpb.sum(1)).pow(2).mean())
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        results.append((s.detach(), pf.detach(), pb.detach(), grads))
    (s0, pf0, pb0, g0), (s1, pf1, pb1, g1) = results
    assert torch.equal(s0, s1) and torch.equal(pf0, pf1) and torch.equal(pb0, pb1)
    for a, b in zip(g0, g1):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    gate = [p for p in g.forward_policy.force_gate.parameters()][-1]
    assert g1[[id(p) for p in params].index(id(gate))].abs().max() > 0
