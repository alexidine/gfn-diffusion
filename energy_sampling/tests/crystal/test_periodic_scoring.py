"""
Consistency tests for periodic (angular-dim) scoring in models/gfn.py.

Covers the fix specified in docs/periodic_scoring_fix.html:
  R1 nearest-image residuals, R2 canonical P_B conditioning, R3
  wrap-after-sampling, and the pb_exact_reversal mixture kernel (eq. 4).

Run from energy_sampling with the csd_mxt_gfn venv:
  python test_periodic_scoring.py
"""
import os
import sys

import pytest

_here = os.path.dirname(os.path.abspath(__file__))
for _root in (os.path.dirname(_here),                                   # gfn_diffusion
              os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import torch

from energy_sampling.models.gfn import GFN
from energy_sampling.utils import uniform_discretizer

DEVICE = torch.device('cpu')
DIM = 12          # crystal layout, max_z_prime=1 -> ang dims are (10, 11)
T = 25
B = 64
PERIOD = 2.0


def build_gfn(t_scale=1.0, pb_exact_reversal=True, dplr_rank=0, dplr_mask_angular=True,
              seed=0):
    torch.manual_seed(seed)
    return GFN(dim=DIM, s_emb_dim=64, conditions_dim=1, harmonics_dim=16, t_dim=16,
               t_hidden_dim=64, s_hidden_dim=64, s_layers=2,
               policy_hidden_dim=64, policy_layers=2,
               flow_hidden_dim=32, flow_layers=2,
               t_scale=t_scale, learned_variance=True, learn_pb=True,
               conditional=False, device=DEVICE, max_z_prime=1,
               do_periodic_angles=True, zero_init=False,
               dplr_rank=dplr_rank, dplr_mask_angular=dplr_mask_angular,
               pb_exact_reversal=pb_exact_reversal).to(DEVICE).eval()


def discretizer(bsz):
    return uniform_discretizer(bsz, T)


def count_crossings(states, gfn):
    """Steps whose stored (wrapped) angular displacement exceeds half a period."""
    ang = states[:, :, gfn.ang_idx]
    return (ang.diff(dim=1).abs() > PERIOD / 2).sum().item()


def test_guardrail():
    build_gfn(dplr_rank=2, dplr_mask_angular=True)  # must construct
    try:
        build_gfn(dplr_rank=2, dplr_mask_angular=False)
    except AssertionError:
        print("PASS guardrail: DPLR + periodic dims requires dplr_mask_angular")
        return
    raise AssertionError("guardrail: expected construction to fail with mask off")


@torch.no_grad()
@pytest.mark.parametrize('pb_exact_reversal', [True, False])
def test_representative_invariance(pb_exact_reversal):
    """Shifting any endpoint by a full period must not change any score."""
    gfn = build_gfn(pb_exact_reversal=pb_exact_reversal)
    torch.manual_seed(1)
    ts = discretizer(B).to(DEVICE)
    i = T // 2
    dts = ts[:, i + 1] - ts[:, i]
    current = torch.randn(B, DIM) * 0.4
    nxt = torch.randn(B, DIM) * 0.4
    current[:, gfn.ang_idx] = torch.rand(B, gfn.ang_dim) * 2 - 1
    nxt[:, gfn.ang_idx] = torch.rand(B, gfn.ang_dim) * 2 - 1

    shift = torch.zeros(B, DIM)
    shift[:, gfn.ang_idx[0]] = PERIOD
    shift[:, gfn.ang_idx[1]] = -PERIOD

    _, _, logpb = gfn._eval_pb_logprob(None, i, current, nxt, dts, ts, None)
    _, _, logpb_s1 = gfn._eval_pb_logprob(None, i, current, nxt + shift, dts, ts, None)
    _, _, logpb_s2 = gfn._eval_pb_logprob(None, i, current - shift, nxt, dts, ts, None)
    assert torch.allclose(logpb, logpb_s1, atol=1e-5), \
        f"logpb not invariant to next-state shift: {(logpb - logpb_s1).abs().max():.2e}"
    assert torch.allclose(logpb, logpb_s2, atol=1e-5), \
        f"logpb not invariant to prev-state shift: {(logpb - logpb_s2).abs().max():.2e}"

    pf_mean, _, d, V, _, _ = gfn._forward_kernel(current, ts[:, i], None, ts[:, i + 1], dts)
    fwd_drift = dts.unsqueeze(1) * pf_mean
    logpf = gfn.fwd_gauss_logprob(nxt - current, fwd_drift, d, dts, V)
    logpf_s = gfn.fwd_gauss_logprob((nxt + shift) - current, fwd_drift, d, dts, V)
    assert torch.allclose(logpf, logpf_s, atol=1e-5), \
        f"logpf not invariant to shift: {(logpf - logpf_s).abs().max():.2e}"
    print(f"PASS representative invariance (pb_exact_reversal={pb_exact_reversal})")


@torch.no_grad()
@pytest.mark.parametrize('pb_exact_reversal', [True, False])
def test_fwd_replay_roundtrip(pb_exact_reversal, t_scale=1.0, exploration=1.0,
                              require_crossings=True):
    """A fwd rollout's own logpf/logpb must be exactly recomputable from its
    stored (wrapped) states -- the exact-replay contract, incl. crossings."""
    gfn = build_gfn(t_scale=t_scale, pb_exact_reversal=pb_exact_reversal)
    torch.manual_seed(2)
    init = torch.zeros(B, DIM)
    expl = torch.full((B,), exploration) if exploration else None
    states, logpf, logpb, _ = gfn.get_traj_fwd(init, discretizer, expl, None, None)
    n_cross = count_crossings(states, gfn)
    if require_crossings:
        assert n_cross > 0, "test setup failed to produce any boundary crossings"
    else:
        assert n_cross == 0, f"no-crossing setup produced {n_cross} crossings"
    r_states, r_logpf, r_logpb, _ = gfn.get_traj_replay(states, discretizer, None, None)
    assert torch.equal(states, r_states)
    dpf = (logpf - r_logpf).abs().max().item()
    dpb = (logpb - r_logpb).abs().max().item()
    assert dpf < 1e-4, f"fwd->replay logpf mismatch {dpf:.2e} ({n_cross} crossings)"
    assert dpb < 1e-4, f"fwd->replay logpb mismatch {dpb:.2e} ({n_cross} crossings)"
    print(f"PASS fwd->replay round trip (pb_exact_reversal={pb_exact_reversal}, "
          f"crossings={n_cross}, max dev pf {dpf:.1e} pb {dpb:.1e})")
    return gfn, states


@torch.no_grad()
@pytest.mark.parametrize('pb_exact_reversal', [True, False])
def test_bwd_replay_roundtrip(pb_exact_reversal):
    """A bwd rollout scored by the forward-direction scorer must reproduce the
    rollout's own values -- the fwd/bwd kernel matchup, incl. crossings."""
    gfn = build_gfn(pb_exact_reversal=pb_exact_reversal)
    torch.manual_seed(3)
    terminal = torch.randn(B, DIM) * 0.4
    # park angular dims against the seam so lift sampling / bridge noise cross
    terminal[:, gfn.ang_idx] = (torch.rand(B, gfn.ang_dim) * 0.1 + 0.9) \
        * torch.where(torch.rand(B, gfn.ang_dim) > 0.5, 1.0, -1.0)
    states, logpf, logpb, _ = gfn.get_traj_bwd(terminal, discretizer, None, None,
                                               detach_traj=True)
    n_cross = count_crossings(states, gfn)
    assert n_cross > 0, "bwd test setup failed to produce any boundary crossings"
    r_states, r_logpf, r_logpb, _ = gfn.get_traj_replay(states, discretizer, None, None)
    # bwd stacks its per-step columns in backward iteration order
    dpf = (logpf.flip(1) - r_logpf).abs().max().item()
    dpb = (logpb.flip(1) - r_logpb).abs().max().item()
    assert dpf < 1e-4, f"bwd->replay logpf mismatch {dpf:.2e} ({n_cross} crossings)"
    assert dpb < 1e-4, f"bwd->replay logpb mismatch {dpb:.2e} ({n_cross} crossings)"
    print(f"PASS bwd->replay round trip (pb_exact_reversal={pb_exact_reversal}, "
          f"crossings={n_cross}, max dev pf {dpf:.1e} pb {dpb:.1e})")


@torch.no_grad()
def test_mixture_inert_off_boundary():
    """With no crossings and states well inside the domain, the mixture must
    coincide with the single-image kernel -- nothing moves off the seam."""
    gfn, states = test_fwd_replay_roundtrip(pb_exact_reversal=True, t_scale=0.02,
                                            exploration=None, require_crossings=False)
    _, _, logpb_mix, _ = gfn.get_traj_replay(states, discretizer, None, None)
    gfn.pb_exact_reversal = False
    _, _, logpb_single, _ = gfn.get_traj_replay(states, discretizer, None, None)
    d = (logpb_mix - logpb_single).abs().max().item()
    assert d < 1e-5, f"mixture vs single-image differ off-boundary: {d:.2e}"
    print(f"PASS mixture inert off-boundary (max dev {d:.1e})")


@torch.no_grad()
def test_seam_continuity():
    """The mixture kernel must be continuous across the seam; the single-image
    kernel carries its known L*c-sized kink there."""
    gfn = build_gfn(pb_exact_reversal=True)
    torch.manual_seed(4)
    ts = discretizer(2).to(DEVICE)
    i = T // 2
    dts = ts[:, i + 1] - ts[:, i]
    eps = 1e-4
    prev = torch.zeros(2, DIM)
    prev[:, gfn.ang_idx[0]] = 0.9
    nxt = torch.zeros(2, DIM)
    nxt[0, gfn.ang_idx[0]] = 1.0 - eps
    nxt[1, gfn.ang_idx[0]] = -(1.0 - eps)

    def gap():
        _, _, logpb = gfn._eval_pb_logprob(None, i, prev, nxt, dts, ts, None)
        return (logpb[0] - logpb[1]).abs().item()

    mix_gap = gap()
    gfn.pb_exact_reversal = False
    single_gap = gap()
    assert mix_gap < 2e-2, f"mixture kernel discontinuous at seam: {mix_gap:.3f}"
    assert single_gap > 5 * mix_gap, \
        f"expected single-image kink >> mixture gap (single {single_gap:.3f}, mix {mix_gap:.3f})"
    print(f"PASS seam continuity (mixture gap {mix_gap:.1e}, single-image kink {single_gap:.3f})")


@torch.no_grad()
def test_mixture_normalization():
    """The mixture density must integrate to 1 over one period, even with the
    conditioning state parked on the seam."""
    gfn = build_gfn(pb_exact_reversal=True)
    n = 4001
    x = torch.linspace(-1.0, 1.0, n).unsqueeze(1)          # [n, 1] grid over one period
    y = torch.full((n, 1), 0.97)
    drift_coeff = torch.full((n, 1), 0.08)
    kappa = torch.ones(n, 1)
    beta_sq = torch.full((n, 1), 0.04)
    t_next = torch.full((n,), 0.8)
    logp = gfn._pb_mixture_ang_logprob(x, y, drift_coeff, kappa, beta_sq, t_next)
    mass = torch.trapz(logp.exp(), x.squeeze(1)).item()
    assert abs(mass - 1.0) < 1e-3, f"mixture mass over one period = {mass:.5f}"
    print(f"PASS mixture normalization (mass {mass:.5f})")


@pytest.mark.parametrize('pb_exact_reversal', [True, False])
def test_traj_checkpoint_and_grads(pb_exact_reversal):
    """Gradient-checkpointed steps must replay the new pre-drawn randomness
    (incl. the mixture's u_lift) bitwise, and the mixture logsumexp must pass
    finite gradients to both policies."""
    gfn = build_gfn(pb_exact_reversal=pb_exact_reversal).train()
    init = torch.zeros(B, DIM)
    torch.manual_seed(11)
    s1, pf1, pb1, _ = gfn.get_traj_fwd(init, discretizer, None, None, None)
    gfn.traj_checkpoint = True
    torch.manual_seed(11)
    s2, pf2, pb2, _ = gfn.get_traj_fwd(init, discretizer, None, None, None)
    assert torch.equal(s1, s2) and torch.equal(pf1, pf2) and torch.equal(pb1, pb2), \
        "traj_checkpoint changed fwd rollout values"

    terminal = torch.zeros(B, DIM)
    terminal[:, gfn.ang_idx] = 0.95
    torch.manual_seed(12)
    _, bpf, bpb, _ = gfn.get_traj_bwd(terminal, discretizer, None, None, detach_traj=True)
    loss = (pf2.sum(1) - pb2.sum(1)).pow(2).mean() + (bpf.sum(1) - bpb.sum(1)).pow(2).mean()
    loss.backward()
    for name, model in (('forward_policy', gfn.forward_policy),
                        ('backward_policy', gfn.backward_policy)):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, f"no gradients reached {name}"
        assert all(torch.isfinite(g).all() for g in grads), f"non-finite grads in {name}"
    print(f"PASS traj_checkpoint parity + finite grads (pb_exact_reversal={pb_exact_reversal})")


if __name__ == '__main__':
    test_guardrail()
    for mode in (True, False):
        test_representative_invariance(mode)
        test_fwd_replay_roundtrip(mode)
        test_bwd_replay_roundtrip(mode)
        test_traj_checkpoint_and_grads(mode)
    test_mixture_inert_off_boundary()
    test_seam_continuity()
    test_mixture_normalization()
    print("\nALL PERIODIC SCORING TESTS PASSED")
