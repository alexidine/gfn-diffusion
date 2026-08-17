"""
Tests for structurally-dead latent rows (docs/decisions.md D33, findings.md F-009).

Rows that `enforce_crystal_system` overwrites with a constant carry no information:
the crystal, and so the energy, does not depend on them, yet `latent_params()`
round-trips them to a canonical value so every prior/buffer row is pinned there while
the energy is flat across the whole box. `bwd` then trains P_F toward a delta while
`fwd` gets no gradient. The fix holds them out of the diffusion entirely.

The risk this suite exists to cover is NOT a crash. A single-digit error in an index
set produces a model that trains, converges, and reports a plausible but wrong log Z.
So the assertions are:

  1. no-dead-rows is BITWISE identical to the pre-change code (guards shared paths)
  2. (ang, lin, dead) is a three-way partition of range(dim)
  3. expanded_dim follows the FINAL sets: dead-angular removes 2, dead-linear removes 1
  4. dead dims never move along any trajectory -- measured, not inferred from the masks
  5. the restricted density matches an INDEPENDENT MultivariateNormal
  6. log-probs are exactly invariant to the value the dead dims are held at
  7. the dead dims' policy-mean units receive exactly zero gradient

(1) is compared against numbers recorded from the pre-change tree, inlined below.

Run from energy_sampling with the csd_mxt_gfn venv:
  python test_dead_latent_rows.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _root in (os.path.dirname(_here),                                   # gfn_diffusion
              os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import pytest
import torch
from torch.distributions import MultivariateNormal

from energy_sampling.models.gfn import GFN
from energy_sampling.models.dead_latent_rows import (
    dead_latent_rows, resolve_dead_rows, live_latent_rows, latent_ndim,
    free_centroid_rows)
from energy_sampling.utils import uniform_discretizer, get_gfn_init_state

DEVICE = torch.device('cpu')
B = 5
TRAJ = 6

# Recorded from the tree immediately BEFORE the dead-row change, via this same
# harness. Any drift here means a shared code path moved, not just the new one.
PRECHANGE = {
    # name: (expanded_dim, fwd_logpf.sum(), bwd_logpb.sum())
    'sg2_zp1':  (16, 392.2286071777, 371.9884338379),
    'sg14_zp1': (16, 384.2695312500, 370.9296875000),
    'sg14_zp2': (26, 506.5727233887, 528.9957275391),
    'nodplr':   (16, 398.9316101074, 371.9884338379),
}
CFG = {
    'sg2_zp1':  dict(dim=12, max_z_prime=1, periodic_centroid_axes=(1, 2), dplr_rank=6),
    'sg14_zp1': dict(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2), dplr_rank=6),
    'sg14_zp2': dict(dim=18, max_z_prime=2, periodic_centroid_axes=(0, 2), dplr_rank=6),
    'nodplr':   dict(dim=12, max_z_prime=1, periodic_centroid_axes=(1, 2), dplr_rank=0),
}

# The PRECHANGE table is float32, and so is every `torch.equal` in this file. Seeding
# is already explicit (build_gfn and roll both call manual_seed), so the RNG is not the
# exposure -- the DEFAULT DTYPE is. Build these models in double and all four pinned
# pairs move by tens of nats, and the mxtaltools batches in the end-to-end test stop
# accepting index writes at all.
#
# That is reachable from outside this file: pytest imports every collected module
# before running any test, so one module-scope `torch.set_default_dtype` anywhere in
# the session rebuilds these models in float64. mxtaltools' conformer suite had exactly
# that, which made this file's bitwise guarantee a function of collection order --
# passing alone, failing beside a module it never references. Pin it per test rather
# than inheriting whatever the process was left in.
DTYPE = torch.float32


@pytest.fixture(autouse=True)
def _pinned_default_dtype():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(1, 2), dplr_rank=6,
              dead_latent_rows=None, dead_latent_values=None, seed=0, **extra):
    torch.manual_seed(seed)
    kwargs = dict(
        dim=dim, s_emb_dim=32, conditions_dim=4, harmonics_dim=16, t_dim=8,
        t_hidden_dim=32, s_hidden_dim=32, s_layers=2,
        policy_hidden_dim=32, policy_layers=2,
        flow_hidden_dim=32, flow_layers=2, cond_hidden_dim=32, cond_layers=2,
        log_var_range=6.0, t_scale=0.05, learned_variance=True,
        condition_embedding_dim=0, conditions_type='vector',
        clipping=True, gfn_clip=200.0, pb_drift_range=0.4, pb_var_range=6.0,
        conditional=False, learn_pb=True, dropout=0, norm='layer', zero_init=False,
        device=DEVICE, max_z_prime=max_z_prime, full_flow=False,
        do_periodic_angles=True,
        periodic_centroids=periodic_centroid_axes is not None,
        periodic_centroid_axes=periodic_centroid_axes,
        dead_latent_rows=dead_latent_rows, dead_latent_values=dead_latent_values,
        dplr_rank=dplr_rank, dplr_rho_max=0.5, dplr_mask_angular=True,
        pb_exact_reversal=True,
    )
    kwargs.update(extra)
    gfn = GFN(**kwargs).to(DEVICE)
    gfn.eval()
    return gfn


def roll(gfn, seed=1234):
    disc = lambda bsz: uniform_discretizer(bsz, TRAJ)
    out = {}
    torch.manual_seed(seed)
    init = get_gfn_init_state(B, gfn.dim, DEVICE)
    with torch.no_grad():
        s, pf, pb, _ = gfn.get_traj_fwd(init, disc, None, False, None, detach_traj=True)
    out['fwd_states'], out['fwd_logpf'], out['fwd_logpb'] = s, pf, pb
    torch.manual_seed(seed + 1)
    terminal = torch.randn(B, gfn.dim, device=DEVICE) * 0.3
    with torch.no_grad():
        s, pf, pb, _ = gfn.get_traj_bwd(terminal, disc, False, None)
    out['bwd_states'], out['bwd_logpf'], out['bwd_logpb'] = s, pf, pb
    torch.manual_seed(seed + 2)
    with torch.no_grad():
        _, pf, pb, _ = gfn.get_traj_replay(out['fwd_states'], disc, False, None)
    out['replay_logpf'], out['replay_logpb'] = pf, pb
    return out


def test_table_matches_known_space_groups():
    """The tabulated rows, independent of any crystal batch."""
    # centrosymmetric: clobbered angles only, no free axes
    assert dead_latent_rows(2) == ()               # triclinic, no-op branch
    for sg in (14, 15):                            # monoclinic: alpha, gamma
        assert dead_latent_rows(sg) == (3, 5), sg
    for sg in (19, 61, 62):                        # orthorhombic: all three
        assert dead_latent_rows(sg) == (3, 4, 5), sg
    for sg in (92, 96, 128):                       # tetragonal: angles only, a=b is diagonal
        assert dead_latent_rows(sg) == (3, 4, 5), sg

    # polar / Sohncke: clobbered angles PLUS free centroid axes, at Z'=1
    assert dead_latent_rows(1) == (6, 7, 8)        # P1: no angles, all three axes free
    for sg in (3, 4, 5):                           # free y -> row 7
        assert dead_latent_rows(sg) == (3, 5, 7), sg
    for sg in (7, 9):                              # free x and z -> rows 6, 8
        assert dead_latent_rows(sg) == (3, 5, 6, 8), sg
    for sg in (29, 33, 75):                        # polar ortho/tetragonal: free z -> row 8
        assert dead_latent_rows(sg) == (3, 4, 5, 8), sg
    assert free_centroid_rows(9) == (6, 8)
    assert free_centroid_rows(14) == ()

    # the ANGLE block is Z'-independent; the FREE rows are dropped at Z'>1, because
    # the free translation there is one global shift and the relative offsets are real
    assert dead_latent_rows(14) == (3, 5)
    assert dead_latent_rows(14, 2) == (3, 5)
    assert dead_latent_rows(9, 1) == (3, 5, 6, 8)
    assert dead_latent_rows(9, 2) == (3, 5)
    assert free_centroid_rows(9, 2) == ()
    assert live_latent_rows(14, 2) == tuple(d for d in range(18) if d not in (3, 5))
    assert latent_ndim(2) == 18
    print("PASS dead-row table over known space groups")


def test_toy_gate():
    """Toys carry space_groups:[1] as a placeholder and must get no dead rows."""
    assert resolve_dead_rows(1, is_crystal=False) == ()
    assert resolve_dead_rows(14, is_crystal=False) == ()
    assert resolve_dead_rows(14, is_crystal=True) == (3, 5)
    assert live_latent_rows(1, 1, is_crystal=False) == tuple(range(12))
    print("PASS toy gate (is_crystal=False yields no dead rows)")


def test_prechange_bitwise_identity():
    """No dead rows must reproduce the pre-change tree exactly, not approximately."""
    for name, kw in CFG.items():
        exp_dim, exp_pf, exp_pb = PRECHANGE[name]
        for dead in (None, ()):
            gfn = build_gfn(dead_latent_rows=dead, **kw)
            r = roll(gfn)
            assert gfn.expanded_dim == exp_dim, (name, dead, gfn.expanded_dim, exp_dim)
            pf, pb = float(r['fwd_logpf'].sum()), float(r['bwd_logpb'].sum())
            assert abs(pf - exp_pf) < 1e-5, (name, dead, pf, exp_pf)
            assert abs(pb - exp_pb) < 1e-5, (name, dead, pb, exp_pb)
    print("PASS no-dead-rows is identical to the pre-change tree (4 configs x 2 spellings)")


def test_three_way_partition():
    """(ang, lin, dead) partitions range(dim). The check that catches an off-by-one."""
    for dead in [(), (3,), (3, 5), (6,), (6, 8), (3, 6), (3, 5, 6, 8)]:
        gfn = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2),
                        dead_latent_rows=dead)
        parts = torch.cat([gfn.ang_idx, gfn.lin_idx, gfn.dead_idx]).sort().values
        assert torch.equal(parts, torch.arange(12)), dead
        assert gfn.ang_dim + gfn.lin_dim + len(dead) == 12, dead
        assert gfn.live_dim + len(dead) == 12, dead
        assert gfn.dead_idx.tolist() == list(dead), dead
        # dead must appear in NEITHER live block
        for d in dead:
            assert d not in gfn.ang_idx.tolist() and d not in gfn.lin_idx.tolist(), (dead, d)
    # a bad index is rejected rather than silently clamped
    for bad in [(12,), (-1,), (99,)]:
        try:
            build_gfn(dim=12, dead_latent_rows=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"dead_latent_rows={bad} was accepted")
    print("PASS three-way partition invariant + bad-index rejection")


def test_expanded_dim_arithmetic():
    """dead-ANGULAR removes 2 policy-input slots (sin, cos); dead-LINEAR removes 1."""
    # periodic axes (0, 2) -> centroid rows 6 and 8 are angular; rows 3, 5 are linear
    for dead in [(), (3,), (3, 5), (6,), (6, 8), (3, 6), (3, 5, 6, 8)]:
        gfn = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2),
                        dead_latent_rows=dead)
        n_ang_dead = sum(1 for d in dead if d in (6, 8))
        n_lin_dead = len(dead) - n_ang_dead
        assert gfn.expanded_dim == 16 - 2 * n_ang_dead - n_lin_dead, (dead, gfn.expanded_dim)
        # and the policy input really is that wide
        state = torch.randn(B, 12)
        assert gfn.expand_state_for_policy(state).shape[-1] == gfn.expanded_dim, dead
    print("PASS expanded_dim follows the final index sets, not dim - n_dead")


def test_dead_dims_never_move():
    """The property we actually want, measured over real trajectories."""
    for dead, vals in [((3, 5), None), ((6, 8), None), ((3, 6), None),
                       ((3, 5), (0.0, 0.0)), ((3, 5), (0.37, -0.62)), ((3, 5), (-1.0, 1.0))]:
        gfn = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2),
                        dead_latent_rows=dead, dead_latent_values=vals)
        r = roll(gfn)
        for key in ('fwd_states', 'bwd_states'):
            v = gfn.dead_invariant_violation(r[key])
            assert v == 0.0, (dead, vals, key, v)
    print("PASS dead dims are bitwise constant over fwd and bwd trajectories")


def test_logprobs_invariant_to_held_value():
    """log Z must not depend on the arbitrary constant the dead dims sit at."""
    ref = None
    for vals in [(0.0, 0.0), (0.37, -0.62), (0.9, 0.9), (-1.0, 1.0)]:
        gfn = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2),
                        dead_latent_rows=(3, 5), dead_latent_values=vals)
        r = roll(gfn)
        cur = (r['fwd_logpf'], r['bwd_logpb'], r['replay_logpf'])
        if ref is None:
            ref = tuple(t.clone() for t in cur)
        for a, b in zip(cur, ref):
            assert torch.equal(a, b), vals
    print("PASS log-probs are exactly invariant to the held constant")


def test_density_matches_independent_mvn():
    """
    Validates the live-dim restriction AND the Woodbury identity against an
    independent implementation, so 'excluded two dims' is distinguished from
    'excluded two dims that happened to be nearly free'.
    """
    for rank, dead in [(6, ()), (6, (3, 5)), (6, (6, 8)), (6, (3, 6)),
                       (0, ()), (0, (3, 5)), (3, (3, 5, 6))]:
        gfn = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2),
                        dplr_rank=rank, dead_latent_rows=dead)
        torch.manual_seed(7)
        state = gfn._pin_dead(gfn._wrap_ang(torch.randn(4, 12) * 0.2))
        dts = torch.full((4,), 1.0 / TRAJ)
        t, t_next = torch.full((4,), 0.3), torch.full((4,), 0.3 + 1.0 / TRAJ)
        with torch.no_grad():
            head = gfn.predict_next_state(
                gfn.s_model(gfn.expand_state_for_policy(state), None), gfn.t_model(t))
            _, _, d, V = gfn.eval_forward_head(head, gfn.var_log_rate(t, t_next, dts))
            delta = torch.randn(4, 12) * 0.1
            if gfn.dead_idx.numel():
                delta = delta.index_copy(1, gfn.dead_idx, torch.zeros(4, gfn.dead_idx.numel()))
            drift = dts.unsqueeze(1) * torch.randn(4, 12) * 0.05
            got = gfn.fwd_gauss_logprob(delta, drift, d, dts, V)

            z = gfn._wrap_ang(delta - drift).index_select(1, gfn.live_idx)
            d_l = d.index_select(1, gfn.live_idx)
            C = torch.diag_embed(d_l)
            if V is not None:
                V_l = V.index_select(1, gfn.live_idx)
                C = C + V_l @ V_l.transpose(-1, -2)
            want = MultivariateNormal(torch.zeros_like(z),
                                      covariance_matrix=C * dts.view(-1, 1, 1)).log_prob(z)
        err = (got - want).abs().max().item()
        assert err < 2e-4, (rank, dead, err)
    print("PASS restricted density matches an independent MultivariateNormal")


def test_dead_policy_units_get_no_gradient():
    for dead in [(3, 5), (6, 8), (3, 6)]:
        gfn = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2),
                        dead_latent_rows=dead)
        gfn.zero_grad()
        torch.manual_seed(3)
        init = get_gfn_init_state(4, gfn.dim, DEVICE)
        _, pf, pb, _ = gfn.get_traj_fwd(init, lambda b: uniform_discretizer(b, 5),
                                        None, False, None, detach_traj=False)
        (pf.sum() + pb.sum()).backward()
        gn = gfn.forward_policy.model.output_layer.weight.grad[:gfn.dim].norm(dim=1)
        assert gn[torch.tensor(dead)].max().item() == 0.0, dead
        assert gn[gfn.live_idx].min().item() > 0.0, dead
    print("PASS dead dims' policy-mean units receive exactly zero gradient")


def test_replay_and_checkpoint_parity():
    for dead in [(), (3, 5)]:
        gfn = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2),
                        dead_latent_rows=dead)
        r = roll(gfn)
        assert torch.equal(r['fwd_logpf'], r['replay_logpf']), dead
        assert torch.equal(r['fwd_logpb'], r['replay_logpb']), dead
        outs = []
        for ckpt in (False, True):
            g = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2),
                          dead_latent_rows=dead)
            g.traj_checkpoint = ckpt
            g.zero_grad()
            torch.manual_seed(3)
            init = get_gfn_init_state(4, g.dim, DEVICE)
            _, pf, pb, _ = g.get_traj_fwd(init, lambda b: uniform_discretizer(b, 5),
                                          None, False, None, detach_traj=False)
            (pf.sum() + pb.sum()).backward()
            outs.append((pf.detach().clone(),
                         g.forward_policy.model.output_layer.weight.grad.clone()))
        assert torch.equal(outs[0][0], outs[1][0]), dead
        assert (outs[0][1] - outs[1][1]).abs().max().item() < 1e-6, dead
    print("PASS fwd/replay agreement and traj_checkpoint parity")


def test_knob_off_restores_full_width():
    gfn = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2),
                    dead_latent_rows=(3, 5), hold_dead_latent_rows=False)
    assert gfn.dead_rows == () and gfn.expanded_dim == 16
    assert gfn.dead_invariant_violation(roll(gfn)['fwd_states']) == 0.0  # vacuous, no dead dims
    print("PASS hold_dead_latent_rows=False restores the full-width behaviour")


def _lowrank_rows_for_dead(gfn):
    """Max |V| over dead rows, at a representative state/time."""
    torch.manual_seed(5)
    state = gfn._pin_dead(torch.randn(4, gfn.dim) * 0.2)
    dts = torch.full((4,), 1.0 / TRAJ)
    t, t_next = torch.full((4,), 0.3), torch.full((4,), 0.3 + 1.0 / TRAJ)
    with torch.no_grad():
        head = gfn.predict_next_state(
            gfn.s_model(gfn.expand_state_for_policy(state), None), gfn.t_model(t))
        _, _, _, V = gfn.eval_forward_head(head, gfn.var_log_rate(t, t_next, dts))
    return V.index_select(1, gfn.dead_idx).abs().max().item()


def test_dead_dims_stay_out_of_dplr_lowrank():
    """
    A dead dim must carry no correlated component: its row of V would otherwise enter
    the Woodbury logdet for a dim that is not part of the process.

    Two branches of get_dplr_cov are covered. The crystal path always has angular dims
    (orientation phi and r), so dplr_mask_angular=False is unconstructible there -- the
    existing guardrail rejects it -- and dplr_zero_mask (angular | dead) is what applies.
    The second branch is only reachable with ang_dim == 0, which for a crystal cannot
    coincide with dead rows, so it is exercised synthetically.
    """
    # branch 1: the real crystal path, via dplr_zero_mask
    for dead in [(3, 5), (6, 8), (3, 6)]:
        gfn = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2),
                        dplr_rank=4, dead_latent_rows=dead, dplr_mask_angular=True)
        assert _lowrank_rows_for_dead(gfn) == 0.0, ('crystal path', dead)

    # branch 2: no angular dims at all (do_periodic_angles=False), dead rows present
    gfn = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=None,
                    dplr_rank=4, dead_latent_rows=(3, 5), do_periodic_angles=False)
    assert gfn.ang_dim == 0, gfn.ang_dim
    assert _lowrank_rows_for_dead(gfn) == 0.0, 'ang_dim == 0 branch'
    print("PASS dead dims carry no DPLR low-rank component (both get_dplr_cov branches)")


def test_dead_values_pair_with_caller_ordering():
    """
    dead_latent_rows is SORTED for the index sets, so writing dead_latent_values straight
    into vals[dead_idx] silently reverses the caller's intent for unsorted input.
    """
    gfn = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2),
                    dead_latent_rows=(5, 3), dead_latent_values=(0.7, -0.2))
    assert abs(float(gfn._dead_values[5]) - 0.7) < 1e-6
    assert abs(float(gfn._dead_values[3]) + 0.2) < 1e-6
    for rows, vals in [((3, 5), (1.0,)), ((3, 3), (1.0, 2.0))]:
        try:
            build_gfn(dim=12, dead_latent_rows=rows, dead_latent_values=vals)
        except ValueError:
            pass
        else:
            raise AssertionError(f"rows={rows} values={vals} accepted")
    print("PASS dead_latent_values pair with the caller's row ordering")


def test_bad_space_group_raises_valueerror():
    for sg in (0, 231, 999):
        try:
            dead_latent_rows(sg)
        except ValueError:
            pass
        except KeyError:
            raise AssertionError(f"sg={sg} raised KeyError, not ValueError")
        else:
            raise AssertionError(f"sg={sg} was accepted")
    print("PASS out-of-range space group raises ValueError")


def test_per_dim_diagnostics_use_live_dims():
    """
    step_var / terminal_var / the Gaussian diagnostics are per-dim means. Counting dims
    that structurally cannot move would make them drop DISCONTINUOUSLY at this change
    for monoclinic+, reading as a coverage regression that never happened.
    """
    for dead in [(), (3, 5), (3, 4, 5)]:
        gfn = build_gfn(dim=12, max_z_prime=1, periodic_centroid_axes=(0, 2),
                        dead_latent_rows=dead)
        states = roll(gfn)['fwd_states']
        full = (states[:, 1:] - states[:, :-1]).pow(2).mean().item()
        live = states.index_select(-1, gfn.live_idx)
        live_only = (live[:, 1:] - live[:, :-1]).pow(2).mean().item()
        # dead increments are exactly zero, so the full-width mean is diluted by exactly
        # live_dim/dim. Relative tolerance: the two reductions sum in a different order,
        # so they agree to float32 rounding rather than bitwise.
        assert abs(live_only * gfn.live_dim / 12 - full) <= 1e-6 * max(full, 1e-30), \
            (dead, full, live_only)
        if not dead:
            assert full == live_only
        # and _mean_over_live matches an explicit restriction
        t = torch.randn(4, 7, 12)
        assert torch.allclose(gfn._mean_over_live(t),
                              t.index_select(-1, gfn.live_idx).mean(-1)), dead
    print("PASS per-dim diagnostics average over live dims only")


def test_checkpoint_dead_row_mismatch_is_loud():
    """
    Dead rows fix the weight layout, so the checkpoint's value must win on resume -- which
    makes a STALE value dangerous: a pre-change monoclinic checkpoint carries no
    dead_latent_rows, so the model would rebuild with them live while the startup probe
    prints reassurance, and log Z would revert to the old scale silently.

    Pre-change sg-1/sg-2 checkpoints must still load, since they resolve to ().
    """
    from argparse import Namespace
    import train as train_mod
    import checkpointing as ckpt_mod

    class StubModeller:
        _resolve_dead_latent_rows = train_mod.Modeller._resolve_dead_latent_rows

        def __init__(self, sgs):
            self.args = Namespace(model=Namespace(hold_dead_latent_rows=True),
                                  space_groups=list(sgs), z_primes=[1])
            self.energy_function = Namespace(is_crystal=True)

    class StubCkpt:
        _assert_dead_rows_match = ckpt_mod.Checkpointer._assert_dead_rows_match

        def __init__(self, sgs):
            self.modeller = StubModeller(sgs)

    cases = [
        # triclinic sg 2 has nothing dead, so a pre-change checkpoint (no key) matches
        ((2,), None, False), ((2,), (), False),
        # matching stored rows always load
        ((14,), (3, 5), False), ((19,), (3, 4, 5), False), ((1,), (6, 7, 8), False),
        # stale or wrong stored rows are refused. sg 1 AS A CRYSTAL has three free
        # centroid axes, so a pre-change P1 checkpoint must NOT silently load either.
        ((14,), None, True), ((14,), (3, 4, 5), True), ((19,), (3, 5), True),
        ((1,), None, True), ((1,), (), True),
    ]
    for sgs, stored, should_raise in cases:
        try:
            StubCkpt(sgs)._assert_dead_rows_match({'dead_latent_rows': stored})
            raised = False
        except ValueError:
            raised = True
        assert raised == should_raise, (sgs, stored, raised, should_raise)
    print("PASS stale checkpoint dead rows are refused; triclinic resumes still load")


def test_explicit_angular_mask_layout():
    """
    Non-crystal state layouts come in through GFN(angular_mask=...), which REPLACED the
    TorsionGFN subclass. Two things are asserted, and the second is the reason the first
    exists.

    (1) An all-True mask reproduces the old subclass layout exactly, and still rolls with
        fwd/replay agreement.
    (2) The FALLBACK is unusable, so the mask is not optional. GFN's non-crystal branch
        (do_periodic_angles=False) writes [False] * dim, i.e. ZERO wrapped dims, with no
        error. For a torsion state that is not a degraded layout: the reward is exactly
        2-periodic in every dim, so with no wrap the integral diverges and no finite log Z
        exists. This test pins that the two paths genuinely differ, so nobody "simplifies"
        the mask away by leaning on the default.
    """
    for dim in (4, 7, 12):
        gfn = GFN(dim=dim, s_emb_dim=64, conditions_dim=0, harmonics_dim=16,
                  t_dim=16, device=DEVICE, angular_mask=[True] * dim)
        gfn.eval()
        for attr in ('dead_rows', 'dead_mask', 'dplr_zero_mask', 'dead_idx', 'live_idx',
                     'live_dim', '_dead_values', 'ang_mask', 'ang_idx', 'lin_idx',
                     'ang_dim', 'lin_dim', 'expanded_dim'):
            assert hasattr(gfn, attr), (dim, attr)
        # exactly the layout the retired subclass produced
        assert gfn.ang_dim == dim and gfn.lin_dim == 0 and gfn.expanded_dim == 2 * dim
        assert gfn.ang_idx.tolist() == list(range(dim)) and gfn.lin_idx.tolist() == []
        assert gfn.dead_rows == () and gfn.live_dim == dim
        torch.manual_seed(0)
        init = get_gfn_init_state(4, dim, DEVICE)
        disc = lambda b: uniform_discretizer(b, 5)
        with torch.no_grad():
            s, pf, pb, _ = gfn.get_traj_fwd(init, disc, None, False, None, detach_traj=True)
            _, rpf, rpb, _ = gfn.get_traj_replay(s, disc, False, None)
        assert torch.equal(pf, rpf) and torch.equal(pb, rpb), dim

    # (2) the fallback really does silently unwrap -- re-introduce the bug and require the
    # difference, rather than trusting the comment
    fallback = GFN(dim=7, s_emb_dim=32, conditions_dim=0, harmonics_dim=8, t_dim=8,
                   device=DEVICE, do_periodic_angles=False)
    assert fallback.ang_dim == 0 and fallback.lin_dim == 7, \
        "the non-crystal fallback is supposed to be the WRONG layout; if it now wraps " \
        "correctly on its own, angular_mask's justification has changed and this test " \
        "and the design doc both need revisiting"

    # a mixed mask partitions on the mask, not on a layout guess: r/theta linear, phi wrapped
    mixed = [False] * 3 + [True] * 4
    gfn = GFN(dim=7, s_emb_dim=32, conditions_dim=0, harmonics_dim=8, t_dim=8,
              device=DEVICE, angular_mask=mixed)
    assert gfn.ang_idx.tolist() == [3, 4, 5, 6] and gfn.lin_idx.tolist() == [0, 1, 2]
    assert gfn.expanded_dim == 3 + 2 * 4

    # a mask of the wrong width must be rejected, not silently mis-partitioned
    for bad in ([True] * 6, [True] * 8):
        try:
            GFN(dim=7, s_emb_dim=32, conditions_dim=0, harmonics_dim=8, t_dim=8,
                device=DEVICE, angular_mask=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"an angular_mask of width {len(bad)} was accepted for dim 7")

    # crystal-layout arguments and an explicit mask are mutually exclusive
    try:
        GFN(dim=12, s_emb_dim=32, conditions_dim=0, harmonics_dim=8, t_dim=8,
            device=DEVICE, angular_mask=[True] * 12,
            periodic_centroids=True, periodic_centroid_axes=(0,))
    except ValueError:
        pass
    else:
        raise AssertionError("periodic_centroid_axes was accepted alongside angular_mask")
    print("PASS explicit angular_mask layout (replaces TorsionGFN; fallback still unwraps)")


def test_end_to_end_physically_inert():
    """
    The property the whole change rests on: a terminal state with the dead rows pinned
    builds the SAME crystal as one where the SDE diffused freely in them. If this fails,
    the rows were not dead and holding them constant is destroying real freedom.

    Needs real structures, so it is skipped when the mini dataset is unavailable.
    """
    import os
    # _here is .../gfn_diffusion/energy_sampling, so two levels up is the repo root that
    # holds mxtaltools -- same expression as the sys.path setup at the top of this file
    path = os.path.join(os.path.dirname(os.path.dirname(_here)),
                        'mxtaltools', 'mini_datasets', 'mini_new_csd.pt')
    if not os.path.exists(path):
        print(f"SKIP end-to-end inertness (dataset not found at {path})")
        return
    from mxtaltools.dataset_utils.utils import collate_data_list
    data = torch.load(path, weights_only=False)
    checked = 0
    for sg in (14, 15, 19, 61):
        dl = [e for e in data if int(e.sg_ind) == sg and int(e.z_prime) == 1][:8]
        if len(dl) < 3:
            continue
        dead = dead_latent_rows(sg)
        batch = collate_data_list(dl)
        batch.pose_aunit()
        batch.build_unit_cell()
        base = batch.latent_params().clone()

        pinned = base.clone()
        pinned[:, list(dead)] = 0.0
        torch.manual_seed(11)
        freed = base.clone()
        freed[:, list(dead)] = torch.rand(len(dl), len(dead)) * 1.8 - 0.9

        got = {}
        for tag, lat in (('pinned', pinned), ('freed', freed)):
            c = batch.clone()
            c.latent_to_cell_params(lat)
            got[tag] = (c.cell_lengths.clone(), c.cell_angles.clone(),
                        c.analyze(['reduction_en'])['reduction_en'].clone())
        for i in range(3):
            assert torch.equal(got['pinned'][i], got['freed'][i]), (sg, i)
        # and the constrained angles really are the constant
        ang = got['pinned'][1]
        for row in dead:
            assert abs(float(ang[:, row - 3].max()) - torch.pi / 2) < 1e-6, (sg, row)
        checked += 1
    assert checked >= 2, f"only {checked} space groups exercised"
    print(f"PASS end-to-end: pinned vs freely-diffused dead rows build identical crystals ({checked} SGs)")


def test_multi_space_group_disagreement_raises():
    """
    A multi-SG run must be refused when the configured space groups need DIFFERENT
    index sets -- and the discriminator has to be the RESOLVED SETS, not the crystal
    system.

    Monoclinic alone carries three sets, because free aunit axes do not follow the
    crystal system:
        sg 3, 4, 5    -> (3, 5, 7)      2-fold along b: free translation along b
        sg 6, 7, 8, 9 -> (3, 5, 6, 8)   mirror perp b: free translation in a-c
        sg 10 - 15    -> (3, 5)         centrosymmetric: no free axis
    So [4, 14] is same-system and still ambiguous. A crystal-system comparison would
    wave it through and serve ONE index set to space groups needing two -- pinning a
    row that is live for half the batch. This test exists so that "simplification"
    cannot be made silently.
    """
    from argparse import Namespace
    import train as train_mod

    class StubModeller:
        _resolve_dead_latent_rows = train_mod.Modeller._resolve_dead_latent_rows

        def __init__(self, sgs, zp=(1,), hold=True, is_crystal=True):
            self.args = Namespace(model=Namespace(hold_dead_latent_rows=hold),
                                  space_groups=list(sgs), z_primes=list(zp))
            self.energy_function = Namespace(is_crystal=is_crystal)

    # agreeing sets resolve fine, including across DIFFERENT crystal systems that
    # happen to coincide -- agreement is what matters, not system identity
    ok_cases = [((14,), (3, 5)), ((14, 15), (3, 5)), ((10, 11, 12, 13, 14, 15), (3, 5)),
                ((3, 4, 5), (3, 5, 7)), ((6, 7, 8, 9), (3, 5, 6, 8)),
                ((19,), (3, 4, 5)), ((2,), ())]
    for sgs, want in ok_cases:
        got = StubModeller(sgs)._resolve_dead_latent_rows(quiet=True)
        assert got == want, f"sgs {sgs}: got {got}, want {want}"

    # disagreeing sets must RAISE -- these are the same-crystal-system traps
    bad_cases = [(4, 14), (14, 4), (3, 6), (6, 10), (4, 6, 14), (2, 14), (14, 19)]
    for sgs in bad_cases:
        try:
            got = StubModeller(sgs)._resolve_dead_latent_rows(quiet=True)
        except ValueError as e:
            assert 'disagree' in str(e), f"sgs {sgs}: raised but not the expected message: {e}"
            continue
        raise AssertionError(
            f"space_groups {sgs} resolve to different dead rows but were accepted, "
            f"returning {got} -- one index set cannot serve them")

    # an empty list is an error, not an implicit ()
    try:
        StubModeller(())._resolve_dead_latent_rows(quiet=True)
        raise AssertionError('empty space_groups was accepted')
    except ValueError as e:
        assert 'at least one entry' in str(e), str(e)

    # and the knob-off / toy paths still short-circuit before any of this
    assert StubModeller((4, 14), hold=False)._resolve_dead_latent_rows(quiet=True) is None
    assert StubModeller((4, 14), is_crystal=False)._resolve_dead_latent_rows(quiet=True) is None

    print(f"PASS multi-SG disagreement raises ({len(bad_cases)} ambiguous sets refused, "
          f"{len(ok_cases)} agreeing sets accepted)")


def test_probe_actually_RUNS_on_real_priors_including_zprime_2():
    """
    The probe's ANSWER was always tested; its ABILITY TO RUN was not. Those are
    different failures, and the second one is worse because it is silent.

    `_verify_dead_latent_rows` swallows any exception and prints "the tabulated rows are
    UNVERIFIED this run", which is the right call for a diagnostic (better than killing a
    multi-day run) but means a broken probe looks like a warning nobody reads. It was
    broken at Z'>1 for exactly that reason: the prep called `build_unit_cell()`, and
    `aunit2ucell` assumes a 3-wide centroid while Z'=2 stores `aunit_centroid` FLATTENED
    as (n, 6) -- appending the affine 1 gives a 7-vector against a 4x4 operator
    ("einsum(): subscript j has size 7 ... previously seen size 4"). So EVERY Z'>1 run
    had no runtime check on the dead-row table, which is the layout with the least other
    coverage. Found by an sg 9 Z'=2 smoke run, 2026-08-12.

    This test asserts the probe COMPLETES and prints its confirmation, not merely that
    the table is right. Skips cleanly when the priors are not on this machine.
    """
    import os
    from argparse import Namespace
    import io, contextlib
    import train as train_mod

    PRIORS = r'D:\crystal_datasets\conditional\priors'
    cases = [('deadrow10k_sg9_zp2_elj.pt', 9, 2, (3, 5)),
             ('deadrow10k_sg14_zp2_elj.pt', 14, 2, (3, 5)),
             ('nehzor_sg14_zp1_elj_prior_dataset.pt', 14, 1, (3, 5)),
             ('gauss_latent_sg1_zp1_prior.pt', 1, 1, (6, 7, 8))]

    class Stub:
        _verify_dead_latent_rows = train_mod.Modeller._verify_dead_latent_rows

        def __init__(self, sg, zp):
            self.args = Namespace(model=Namespace(hold_dead_latent_rows=True),
                                  space_groups=[sg], z_primes=[zp])
            self.energy_function = Namespace(is_crystal=True)

    checked = 0
    for fname, sg, zp, want in cases:
        path = os.path.join(PRIORS, fname)
        if not os.path.exists(path):
            continue
        batch = torch.load(path, weights_only=False)['equalized_prior']
        assert batch.latent_params().shape[-1] == 6 + 6 * zp,             f"{fname}: latent width {batch.latent_params().shape[-1]} != {6 + 6 * zp}"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            Stub(sg, zp)._verify_dead_latent_rows(batch, n_probe=8)
        out = buf.getvalue()
        assert 'UNVERIFIED' not in out,             f"{fname} (sg {sg}, Z'={zp}): probe could not run -- {out.strip()}"
        assert f'confirms rows {want}' in out,             f"{fname} (sg {sg}, Z'={zp}): expected rows {want}, got: {out.strip()}"
        checked += 1

    if checked == 0:
        print("  SKIP probe-runs-on-real-priors (no priors on this machine)")
        return
    zp2 = sum(1 for f, _, z, _ in cases if z == 2 and os.path.exists(os.path.join(PRIORS, f)))
    assert zp2 >= 1, "no Z'=2 prior exercised -- the regression this test exists for"
    print(f"PASS probe RUNS (not just answers) on {checked} real priors, {zp2} of them Z'=2")


def test_rotation_magnitude_clamp_covers_every_aunit():
    """
    mxtaltools' latent_to_cell_params floors each aunit's rotation MAGNITUDE away from 0,
    because r=0 is a singularity of compute_jacobian's log(sin(r/2)) (clamped at ~37
    nats). The index was `5 + 6*(1 + ind)` = 11 + 6*ind, correct at Z'=1 and only by
    coincidence: the layout is [3 lengths | 3 angles | 3*Zp centroids | 3*Zp orientations]
    with both aunit blocks FLATTENED, so aunit ind's magnitude is at
    6 + 3*Zp + 3*ind + 2. At Z'=2 the old form clamped [11, 17] where [14, 17] was needed
    -- 11 is centroid[1][z] -- so only the LAST aunit was ever protected and row 14 could
    reach |rotvec| EXACTLY 0.0. It never IndexErrored because the wrong index 5+6*Zp is
    exactly width-1: silent by construction.

    Not a D33 defect, but D33's Z'=2 work is what reached it, and the dead-row machinery
    shares this layout -- so the assertion belongs next to the layout tests.
    """
    import os
    path = os.path.abspath(os.path.join(_here, '..', '..', 'mxtaltools',
                                        'mini_datasets', 'mini_new_csd.pt'))
    if not os.path.exists(path):
        # LOUD, not a quiet skip. The first version of this test had a garbled path, so it
        # skipped and the suite still printed ALL TESTS PASSED -- a test written to catch
        # silent-skip failures, failing silently. Fail instead: the dataset is part of the
        # repo, so its absence is a broken checkout, not a valid configuration.
        raise AssertionError(
            f"mini_new_csd.pt not found at {path} -- this test cannot be skipped, it is "
            f"the only cover for the rotation-magnitude clamp indices")
    from mxtaltools.dataset_utils.utils import collate_data_list
    mol = next(e for e in torch.load(path, weights_only=False)
               if int(e.z_prime) == 1 and bool(e.is_well_defined))

    for zp in (1, 2, 3):
        width = 6 + 6 * zp
        for aunit in range(zp):
            b = collate_data_list([mol.clone() for _ in range(2)], max_z_prime=zp)
            b.reset_sg_info(9)
            x = torch.zeros(2, width)
            # drive this aunit's whole orientation triple to the box floor
            for c in range(3):
                x[:, 6 + 3 * zp + 3 * aunit + c] = -1.0
            b.latent_to_cell_params(x.clone())
            r = b.aunit_orientation.reshape(2, zp, 3)[0, aunit].norm().item()
            assert r > 1e-3, (
                f"Z'={zp} aunit {aunit}: |rotvec| = {r:.6g} -- the r=0 singularity is "
                f"reachable, so the clamp is on the wrong row (expected index "
                f"{6 + 3 * zp + 3 * aunit + 2})")
    # and the fix must be a NO-OP at Z'=1, where the old formula was already right
    assert [5 + 6 * (1 + i) for i in range(1)] == [6 + 3 * 1 + 3 * i + 2 for i in range(1)]
    print("PASS rotation-magnitude clamp floors every aunit at Z'=1/2/3 (no-op at Z'=1)")


if __name__ == '__main__':
    # the autouse fixture above is pytest-only, so the direct runner sets it itself
    torch.set_default_dtype(DTYPE)
    test_table_matches_known_space_groups()
    test_toy_gate()
    test_prechange_bitwise_identity()
    test_three_way_partition()
    test_expanded_dim_arithmetic()
    test_dead_dims_never_move()
    test_logprobs_invariant_to_held_value()
    test_density_matches_independent_mvn()
    test_dead_policy_units_get_no_gradient()
    test_replay_and_checkpoint_parity()
    test_knob_off_restores_full_width()
    test_dead_dims_stay_out_of_dplr_lowrank()
    test_dead_values_pair_with_caller_ordering()
    test_bad_space_group_raises_valueerror()
    test_per_dim_diagnostics_use_live_dims()
    test_checkpoint_dead_row_mismatch_is_loud()
    test_explicit_angular_mask_layout()
    test_multi_space_group_disagreement_raises()
    test_probe_actually_RUNS_on_real_priors_including_zprime_2()
    test_rotation_magnitude_clamp_covers_every_aunit()
    test_end_to_end_physically_inert()
    print("\nALL DEAD-LATENT-ROW TESTS PASSED")
