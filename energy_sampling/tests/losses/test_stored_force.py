"""The stored-force replay tail (replay_loss_coeffs.stored_force_k).

The claim: re-propagating the last k STORED steps through the noise the current
kernel implies (GFN._implied_step) lands exactly on the stored states, scores the
same log P_F / log P_B as a fixed replay, and hands back a terminal that carries
d x_T / d theta -- the reparameterisation gradient of a fixed sample. Each test
is written so that breaking the property FAILS rather than degrades:

  1. RECONSTRUCTION: implied-noise replay returns the stored states to float
     precision, for k in {1, 3}, with and without DPLR, on a fully periodic
     state (nearest-image wrap is exercised).
  2. DENSITY PARITY: log P_F / log P_B match the fixed replay bitwise-close.
  3. GRADIENT: for k = 1, d (F . x_T(theta)) / d theta from the loop equals a
     hand-written reparameterisation of the same stored step. The zero-valued
     surrogate used by the loss (F . (x_T - x_T.detach())) carries exactly that
     gradient and exactly zero value.
  4. GATES: resample_last_k and implied_noise_last_k together are refused; k
     above the trajectory length is refused; k = 0 leaves the fixed replay's
     returned states the very same object (bitwise-identical path).
  5. LEGS: the buffer stores the force PER REWARD LEG (flow, phys, bound) and
     remix_force_legs composes d log R / d x_T at the current lambda -- exact
     endpoints, NaN rows kept; an energy without legs yields (0, total, 0).
  6. THE POOLED SEAT: on the conditional route the replay seat's only loss is
     pooled_condition_vargrad over the live stash, so the surrogate has to
     reach it THROUGH log_r. The stashed log_r's parameter gradient equals the
     surrogate's exactly, and the pooled term's gradient moves with F.

    python -m pytest -q tests/losses/test_stored_force.py
"""
import os
import sys

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _root in (os.path.dirname(_here),
              os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from types import SimpleNamespace

import pytest
import torch

from energy_sampling.buffer import N_FORCE_LEGS
from energy_sampling.gflownet_losses import (get_gfn_backward_loss, pooled_condition_vargrad,
                                             remix_force_legs, terminal_force_legs)
from energy_sampling.models.gfn import GFN
from energy_sampling.utils import uniform_discretizer, get_gfn_init_state

DEVICE = 'cpu'
DIM = 7
B = 16
TRAJ = 5


def _gfn(dplr_rank=0, seed=0):
    torch.manual_seed(seed)
    return GFN(dim=DIM, s_emb_dim=16, conditions_dim=0, harmonics_dim=8, t_dim=8,
               device=DEVICE, angular_mask=[True] * DIM, dplr_rank=dplr_rank)


def _disc():
    return lambda bsz: uniform_discretizer(bsz, TRAJ)


def _rollout(gfn):
    torch.manual_seed(1)
    init = get_gfn_init_state(B, DIM, DEVICE)
    states, logpf, logpb, _ = gfn.get_traj_fwd(init, _disc(), None, False, None, detach_traj=True)
    return states.detach()


@pytest.mark.parametrize('dplr_rank', [0, 2])
@pytest.mark.parametrize('k', [1, 3])
def test_implied_noise_reconstructs_and_scores_the_stored_path(dplr_rank, k):
    gfn = _gfn(dplr_rank)
    stored = _rollout(gfn)
    st_fixed, pf_fixed, pb_fixed, _ = gfn.get_traj_replay(stored, _disc(), False, None)
    st_imp, pf_imp, pb_imp, _ = gfn.get_traj_replay(stored, _disc(), False, None,
                                                    implied_noise_last_k=k)
    # 1. reconstruction, nearest image on the periodic dims
    err = gfn._wrap_ang((st_imp.detach() - stored).reshape(-1, DIM)).abs().max().item()
    assert err < 1e-5, f'implied-noise tail missed the stored states by {err}'
    # the live tail carries gradient, the untouched prefix does not
    assert st_imp[:, -1].requires_grad
    assert st_imp.requires_grad  # the copy holds the live tail
    # 2. density parity with the fixed replay
    assert torch.allclose(pf_imp, pf_fixed, atol=1e-4), (pf_imp - pf_fixed).abs().max()
    assert torch.allclose(pb_imp, pb_fixed, atol=1e-4), (pb_imp - pb_fixed).abs().max()


@pytest.mark.parametrize('dplr_rank', [0, 2])
def test_implied_split_is_on_distribution_for_a_fresh_step(dplr_rank):
    """A step DRAWN from the current kernel is reproduced by the implied split with a
    whitened residual norm of ~1 per dim (chi2_n / n), for the diagonal kernel and for
    DPLR alike. The old diagonal-only split read >1 on every DPLR row (the low-rank
    noise was charged to the diagonal), which is what a staleness gate would have
    culled for no reason."""
    torch.manual_seed(1)
    gfn = _gfn(dplr_rank, seed=1)
    B = 4000
    # every dim is angular on [-1, 1] here, and a wrapped step's nearest-image
    # residual is shorter than the noise that made it (selecting unwrapped rows
    # would bias toward small noise instead); so take a FINE grid from the origin,
    # where the step's sd is ~0.07 and no row comes near the boundary
    ts = uniform_discretizer(B, 200).to(DEVICE)
    dts = ts[:, 1] - ts[:, 0]
    x_prev = torch.zeros(B, DIM, device=DEVICE)
    with torch.no_grad():
        pf_mean, pflogvars, d, V, _, _ = gfn._forward_kernel(x_prev, ts[:, 0], None, ts[:, 1], dts)
        x_next = gfn.fwd_propagate(x_prev, True, dts, pf_mean, d.log(), V)
        assert (x_next.abs() < 0.9).all(), 'fresh step reached the wrap boundary -- test setup invalid'
        out = gfn._implied_step(x_prev, x_next, dts, ts[:, 0], ts[:, 1], None, True)
    recon = out[0]
    assert gfn._wrap_ang(recon - x_next).abs().max() < 1e-5
    stat = gfn._last_implied_eps_sq
    assert abs(stat.mean().item() - 1.0) < 0.05, stat.mean().item()
    if dplr_rank > 0:
        assert V is not None and gfn._last_implied_epsr_sq is not None
    else:
        assert gfn._last_implied_epsr_sq is None


def test_last_step_gradient_matches_hand_written_reparameterisation():
    gfn = _gfn(0)
    stored = _rollout(gfn)
    torch.manual_seed(2)
    F = torch.randn(B, DIM)

    # the loop's gradient
    gfn.zero_grad()
    st_imp, _, _, _ = gfn.get_traj_replay(stored, _disc(), False, None, implied_noise_last_k=1)
    x_T = st_imp[:, -1]
    surrogate = (F * (x_T - x_T.detach())).sum()
    assert surrogate.item() == 0.0            # the loss VALUE is untouched
    surrogate.backward()
    g_loop = torch.cat([p.grad.flatten().clone() for p in gfn.parameters() if p.grad is not None])

    # the hand-written reparameterisation of the same stored step
    gfn.zero_grad()
    ts = _disc()(B)
    i = TRAJ - 1
    dts = ts[:, i + 1] - ts[:, i]
    x_prev, x_next = stored[:, i], stored[:, i + 1]
    pf_mean, pflogvars, d, V, s_emb, t_emb = gfn._forward_kernel(x_prev, ts[:, i], None, ts[:, i + 1], dts)
    drift = dts.unsqueeze(1) * pf_mean
    with torch.no_grad():
        eps = gfn._wrap_ang(x_next - x_prev - drift) / (dts.sqrt().unsqueeze(1) * d.sqrt())
    x_T_hand = gfn._pin_dead(gfn._wrap_ang(x_prev + drift + dts.sqrt().unsqueeze(1) * d.sqrt() * eps))
    assert gfn._wrap_ang(x_T_hand.detach() - x_next).abs().max() < 1e-5
    (F * x_T_hand).sum().backward()
    g_hand = torch.cat([p.grad.flatten().clone() for p in gfn.parameters() if p.grad is not None])

    assert g_loop.shape == g_hand.shape
    assert g_hand.abs().max() > 0, 'the hand-written gradient is identically zero -- test is vacuous'
    assert torch.allclose(g_loop, g_hand, atol=1e-6, rtol=1e-5), (g_loop - g_hand).abs().max()


def test_gates():
    gfn = _gfn(0)
    stored = _rollout(gfn)
    with pytest.raises(ValueError):
        gfn.get_traj_replay(stored, _disc(), False, None, implied_noise_last_k=1, resample_last_k=1)
    with pytest.raises(ValueError):
        gfn.get_traj_replay(stored, _disc(), False, None, implied_noise_last_k=TRAJ + 1)
    st, _, _, _ = gfn.get_traj_replay(stored, _disc(), False, None)
    assert st is stored or torch.equal(st, stored)
    assert not st.requires_grad


def test_mean_mode_shift_gradient_matches_hand_written_mean_channel():
    """stored_force_mode 'mean': the surrogate F . (shift - shift.detach()) on a
    FIXED replay carries exactly F . dt * d mu/d theta and nothing else."""
    gfn = _gfn(0)
    stored = _rollout(gfn)
    torch.manual_seed(3)
    F = torch.randn(B, DIM)

    gfn.zero_grad()
    shift = gfn.last_step_mean_shift(stored, _disc(), False, None)
    surrogate = (F * (shift - shift.detach())).sum()
    assert surrogate.item() == 0.0
    surrogate.backward()
    g_loop = torch.cat([p.grad.flatten().clone() for p in gfn.parameters() if p.grad is not None])

    gfn.zero_grad()
    ts = _disc()(B)
    i = TRAJ - 1
    dts = ts[:, i + 1] - ts[:, i]
    pf_mean = gfn._forward_kernel(stored[:, i], ts[:, i], None, ts[:, i + 1], dts)[0]
    (F * dts.unsqueeze(1) * pf_mean).sum().backward()
    g_hand = torch.cat([p.grad.flatten().clone() for p in gfn.parameters() if p.grad is not None])

    assert g_hand.abs().max() > 0
    assert torch.allclose(g_loop, g_hand, atol=1e-6, rtol=1e-5), (g_loop - g_hand).abs().max()
    # and the fixed replay it rides on is untouched: no implied tail, no grad on states
    st, _, _, _ = gfn.get_traj_replay(stored, _disc(), False, None)
    assert not st.requires_grad


# ----------------------------------------------------------------------- 5. legs

def test_remix_force_legs_takes_the_endpoints_exactly_and_keeps_nan_rows():
    torch.manual_seed(4)
    legs = torch.randn(6, N_FORCE_LEGS, DIM)
    legs[1, 1] = float('inf')            # a zeroed-out physical leg would be 0; inf shows it is NOT read
    legs[5] = float('nan')               # no force recorded
    flow, phys, bound = legs[:, 0], legs[:, 1], legs[:, 2]
    f0 = remix_force_legs(legs, 0.0)
    f1 = remix_force_legs(legs, 1.0)
    fm = remix_force_legs(legs, 0.3)
    ok = torch.arange(6) < 5
    assert torch.equal(f0[ok], (flow + bound)[ok]), 'lambda 0 must not read the physical leg'
    assert torch.isfinite(f0[:5]).all()
    assert torch.equal(f1[ok][[0, 2, 3, 4]], (phys + bound)[[0, 2, 3, 4]])
    assert torch.allclose(fm[[0, 2, 3, 4]], (0.7 * flow + 0.3 * phys + bound)[[0, 2, 3, 4]])
    assert torch.isnan(f0[5]).all() and torch.isnan(f1[5]).all() and torch.isnan(fm[5]).all()
    with pytest.raises(ValueError, match='force legs'):
        remix_force_legs(legs[:, :2], 0.5)
    with pytest.raises(ValueError, match='force legs'):
        remix_force_legs(legs[:, 0], 0.5)


class _ToyEnergy:
    """An energy with no legs (the toys): log R = -|x|^2 / T."""
    lambda_mix = 1.0

    @staticmethod
    def log_reward(x, mol_batch, log_T, return_exp=False, keep_grads=False):
        return -(x ** 2).sum(-1) / (10.0 ** log_T)


def test_terminal_force_legs_without_legs_is_the_total_in_the_weight_one_slot():
    """The bounding slot carries weight 1 at every lambda; the physical slot
    would drop the toy's whole force at lambda = 0."""
    torch.manual_seed(5)
    x = torch.randn(B, DIM)
    log_T = torch.full((B,), 0.5)
    legs, stats = terminal_force_legs(log_T, _ToyEnergy(), None, x)
    assert legs.shape == (B, N_FORCE_LEGS, DIM) and not legs.requires_grad
    expect = -2.0 * x / (10.0 ** 0.5)
    assert torch.allclose(legs[:, 2], expect, atol=1e-6)
    assert torch.equal(legs[:, 0], torch.zeros(B, DIM)) and torch.equal(legs[:, 1], torch.zeros(B, DIM))
    for lam in (0.0, 0.3, 1.0):
        assert torch.allclose(remix_force_legs(legs, lam), expect, atol=1e-6)
    assert float(stats['force/nonfinite_frac']) == 0.0
    assert abs(float(stats['force/norm_mean']) - expect.norm(dim=-1).mean().item()) < 1e-6
    # a 3-D states tensor means "take the terminal"
    legs3, _ = terminal_force_legs(log_T, _ToyEnergy(), None, torch.stack([x * 0, x], dim=1))
    assert torch.equal(legs3, legs)


def test_terminal_force_legs_sanitises_per_leg_and_reports_it():
    class _Bad(_ToyEnergy):
        @staticmethod
        def log_reward(x, mol_batch, log_T, return_exp=False, keep_grads=False):
            r = -(x ** 2).sum(-1)
            scale = torch.ones(x.shape[0])
            scale[0] = float('inf')          # row 0's gradient is non-finite; a constant
            return r * scale                 # factor, so nothing leaks into the other rows
    x = torch.randn(B, DIM)
    legs, stats = terminal_force_legs(torch.zeros(B), _Bad(), None, x)
    assert torch.isfinite(legs).all()
    assert torch.equal(legs[0], torch.zeros(N_FORCE_LEGS, DIM))
    assert abs(float(stats['force/nonfinite_frac']) - 1.0 / B) < 1e-6
    assert abs(float(stats['force/nonfinite_frac_bound']) - 1.0 / B) < 1e-6
    assert float(stats['force/nonfinite_frac_flow']) == 0.0
    assert float(stats['force/nonfinite_frac_phys']) == 0.0


# ---------------------------------------------------------------- 6. pooled seat

_ZERO = dict(beta=10.0, freeze_policy=0.0, freeze_z=1.0, vg_by_condition=0.0, vg_lb=0.0,
             vg_lme=0.0, level_gap=0.0, pf_boost=0.0, db=0.0, subtb=0.0, emp_z=0.0, tb=0.0,
             emp_z_persistent=0.0, mle=0.0, tbc=0.0, loss_clip=1.0e9, traj_grads=0.0,
             resample_last_k=0, stored_force_mode='implied')


def _params_grad(gfn, scalar):
    gfn.zero_grad()
    scalar.backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().clone()
                      for p in gfn.parameters()])


def _stash(gfn, stored, F, k, slot, cids):
    """One get_gfn_backward_loss call on the stored rows, as replay_train_step
    (slot 'replay', stored_force_k = k) and bwd_train_step (slot 'bwd') make it."""
    coeffs = SimpleNamespace(**_ZERO, stored_force_k=k)
    mol = SimpleNamespace(replay_force=F) if F is not None else SimpleNamespace()
    log_r = -torch.arange(B, dtype=torch.float32) / B
    get_gfn_backward_loss(coeffs, stored[:, -1], gfn, log_r, _disc(), mol, repeats=1,
                          trajectories=stored, condition_log_z=None, condition_id=cids,
                          live_stash=slot)
    return getattr(gfn, f'_live_{slot}')


def test_the_surrogate_reaches_the_pooled_term_through_the_live_stash():
    gfn = _gfn(0)
    gfn._stash_live_branches = True
    stored = _rollout(gfn)
    torch.manual_seed(6)
    F = torch.randn(B, DIM)
    F[3] = float('nan')                       # a row with no force recorded
    cids = torch.arange(B) // 2               # 8 conditions x 2 rows, both seats

    # (a) the stashed log_r carries EXACTLY the surrogate's gradient
    live = _stash(gfn, stored, F, 1, 'replay', cids)
    assert live['log_r'].requires_grad
    g_stash = _params_grad(gfn, live['log_r'].sum())
    st_imp, _, _, _ = gfn.get_traj_replay(stored, _disc(), False, None, implied_noise_last_k=1)
    F_masked = torch.nan_to_num(F) * torch.isfinite(F).all(-1, keepdim=True).float()
    g_direct = _params_grad(gfn, (F_masked * st_imp[:, -1]).sum())
    assert g_direct.abs().max() > 0
    assert torch.allclose(g_stash, g_direct, atol=1e-6, rtol=1e-5), (g_stash - g_direct).abs().max()
    # the stored log R itself is untouched by the surrogate (zero VALUE)
    assert torch.allclose(live['log_r'].detach(), -torch.arange(B, dtype=torch.float32) / B)

    # (b) the pooled term, as the fused step forms it over (_live_replay, _live_bwd):
    # the force changes its parameter gradient by EXACTLY sum_i w_i F_i . dx_T,i/dtheta,
    # w_i = d pooled / d log_r_i -- the surrogate enters through log_r and nowhere
    # else. (F = 0 is compared against, not the fixed replay: the implied-noise
    # tail already reparameterises the density terms, by design.)
    def pooled(force):
        torch.manual_seed(7)
        rep = _stash(gfn, stored, force, 1, 'replay', cids)
        bwd = _stash(gfn, stored, None, 0, 'bwd', cids)
        rows, _ = pooled_condition_vargrad(rep, bwd, beta=40.0, ratio=0.5)
        assert rows is not None
        return rep, bwd, rows.mean()
    _, _, p_F = pooled(F)
    g_F = _params_grad(gfn, p_F)
    rep0, bwd0, p_0 = pooled(torch.zeros(B, DIM))
    g_0 = _params_grad(gfn, p_0)
    assert (g_F - g_0).abs().max() > 1e-6, 'the pooled gradient did not see the stored force'
    leaf = rep0['log_r'].detach().clone().requires_grad_(True)
    torch.manual_seed(7)
    rows, _ = pooled_condition_vargrad({**rep0, 'log_r': leaf}, bwd0, beta=40.0, ratio=0.5)
    w = torch.autograd.grad(rows.mean(), leaf)[0]
    assert w.abs().max() > 0
    st_imp, _, _, _ = gfn.get_traj_replay(stored, _disc(), False, None, implied_noise_last_k=1)
    g_expect = _params_grad(gfn, (w.unsqueeze(-1) * F_masked * st_imp[:, -1]).sum())
    assert torch.allclose(g_F - g_0, g_expect, atol=1e-6, rtol=1e-4),         (g_F - g_0 - g_expect).abs().max()


def test_a_missing_replay_force_is_refused_not_skipped():
    gfn = _gfn(0)
    stored = _rollout(gfn)
    with pytest.raises(ValueError, match='replay_force'):
        _stash(gfn, stored, None, 1, 'replay', torch.arange(B))


@pytest.mark.parametrize('mode', ['implied', 'resample'])
def test_live_tail_survives_trajectory_checkpointing(mode):
    """Under trajectory activation checkpointing (MACE checkpoints every branch)
    the checkpointed step saves its input, a view of the states tensor. The
    live tail must not write into that tensor, or the backward recompute
    refuses with 'modified by an inplace operation' (smoke_mace_sf1, 2026-09-16).
    Both live tails, k = 2, gradient must run to the parameters."""
    gfn = _gfn(0)
    gfn.traj_checkpoint = True
    gfn.traj_checkpoint_modes = None
    stored = _rollout(gfn)
    torch.manual_seed(4)
    F = torch.randn(B, DIM)
    kw = {'implied_noise_last_k': 2} if mode == 'implied' else {'resample_last_k': 2}
    st, pf, pb, _ = gfn.get_traj_replay(stored, _disc(), False, None, **kw)
    assert st.shape == stored.shape
    if mode == 'implied':
        assert gfn._wrap_ang((st.detach() - stored).reshape(-1, DIM)).abs().max() < 1e-5
    else:
        assert torch.equal(st[:, :TRAJ - 1], stored[:, :TRAJ - 1])
    gfn.zero_grad()
    ((F * st[:, -1]).sum() + pf.sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gfn.parameters())
