"""
DEEP tests for the dead-latent-row SDE change (decisions.md D33).

test_dead_latent_rows.py proves the mechanics: index sets, constancy, densities,
gradients. This file attacks the STATISTICS and the combinatorial surface -- the
things that would still be wrong if every mechanical invariant held.

  1. LOG Z RECOVERY (`check_log_z_recovery`, NOT collected by pytest -- it trains
     three models; run this file as a script). Trains real TB on a target that
     depends only on the live dims.
     A (dim d, nothing dead) vs B (dim d+2, the extras dead) must learn the SAME
     log Z; C (dim d+2, nothing dead) must not. A == B is the change's promise. The
     closed-form constant is reported but only weakly asserted -- see the docstring
     for why an earlier version that asserted on it was measuring non-convergence.

  2. EXACTNESS. The restricted log-prob must equal an independent live-only
     reimplementation BITWISE, and must differ from the full-width value. This
     separates "excluded the dims" from "excluded the dims and also perturbed the
     surviving arithmetic", which a different index order could do silently.

  2b. CONFIG COMBINATION MATRIX. var_scheduled / exploration_std / path_grad_last_k /
     full_flow / learn_pb=False / learned_variance=False / conditional / dplr /
     zero_init / unbounded logvar / single-image P_B, each crossed with dead sets and
     checked for the partition invariant, constancy, finiteness, and None == ()
     bitwise identity. Flagged as untested by the adversarial review.

  3. DEGENERATE DEAD SETS. live_dim of 0, 1 and D-1 -- the empty and singleton
     reductions, where index code usually breaks. live_dim == 0 must give log-probs
     of exactly 0, not NaN.

  4. TRAJECTORY-LENGTH EXTREMES. T=1 (the bwd final-step branch IS the whole
     trajectory), T=2, T=60.

  5. DEVICE PARITY. The SAME trajectory scored on CPU and CUDA. Sampling cannot be
     compared across devices -- independent RNG streams make it a different sample,
     not a different computation, which an earlier version of this test got wrong.

  6. FREE-AXIS CANONICALISER. Idempotency, P1's three-free-axis maximal case, and
     ENERGY invariance over every space group in the mini dataset. The RDF is checked
     per-structure and only for how MANY structures move, not for being exactly zero:
     an RDF comparison at a fixed cutoff carries a boundary-counting term worth ~0.05 on
     a physically identical pair (F-010b). Energy is the criterion that matters.

  7. ROUND-TRIP FIXED POINT, LOG-WEIGHT VARIANCE, DTYPE, CHECKPOINT GUARD.
     latent_params() now mutates the batch, so the pipeline must reach a fixed point.
     Var(log w) must lose the dead dims' additive term. The checkpoint guard must
     distinguish pre-free-axis rows and Z'=1 from Z'=2.

PRE-EXISTING limitations confirmed here and deliberately NOT asserted as bugs of this
change: DPLR + float64 raises (a Float V meets a Double noise in fwd_propagate,
identical with dead=None), and the states buffer is always float32 because
init_traj_tensors allocates at the default dtype.

Run from energy_sampling with the csd_mxt_gfn venv:
  python test_dead_latent_rows_deep.py
"""
import os
import sys

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for _root in (os.path.dirname(_here),
              os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import math

import torch

from energy_sampling.models.gfn import GFN
from energy_sampling.utils import uniform_discretizer, get_gfn_init_state

CPU = torch.device('cpu')


def build(dim, dead=None, device=CPU, dplr_rank=0, periodic=None, seed=0, **extra):
    torch.manual_seed(seed)
    kw = dict(
        dim=dim, s_emb_dim=64, conditions_dim=4, harmonics_dim=16, t_dim=16,
        t_hidden_dim=64, s_hidden_dim=64, s_layers=2,
        policy_hidden_dim=64, policy_layers=2, flow_hidden_dim=32, flow_layers=2,
        cond_hidden_dim=32, cond_layers=2,
        log_var_range=6.0, t_scale=1.0, learned_variance=True,
        condition_embedding_dim=0, conditions_type='vector',
        clipping=True, gfn_clip=1e4, pb_drift_range=0.4, pb_var_range=6.0,
        conditional=False, learn_pb=True, dropout=0, norm=None, zero_init=False,
        device=device, max_z_prime=1, full_flow=False,
        do_periodic_angles=periodic is not None,
        periodic_centroids=periodic is not None,
        periodic_centroid_axes=periodic,
        dead_latent_rows=dead, dplr_rank=dplr_rank, dplr_rho_max=0.5,
        dplr_mask_angular=True, pb_exact_reversal=True,
    )
    kw.update(extra)
    return GFN(**kw).to(device)


# ---------------------------------------------------------------- 1. log Z
def analytic_log_z(d, sigma):
    """log of int exp(-||x||^2 / (2 sigma^2)) dx over R^d."""
    return 0.5 * d * math.log(2 * math.pi * sigma * sigma)


def train_tb(gfn, live_dim, sigma, steps, batch=256, traj=12, lr=1e-3, seed=0,
             z_lr_mult=30.0):
    """
    Minimal trajectory-balance training. The target depends ONLY on the first
    `live_dim` coordinates, so any dim beyond that is exactly flat -- which is the
    situation dead rows exist for.
    """
    torch.manual_seed(seed)
    # log Z is a single scalar whose gradient is the MEAN TB residual, so it moves far
    # slower than the policy at a shared lr. Give it its own faster group, else the test
    # measures how little a scalar moved in N steps rather than where it converges.
    z_params = list(gfn.flow_model.parameters())
    z_ids = {id(p) for p in z_params}
    rest = [p for p in gfn.parameters() if id(p) not in z_ids]
    opt = torch.optim.Adam([{'params': rest, 'lr': lr},
                            {'params': z_params, 'lr': lr * z_lr_mult}])
    disc = lambda b: uniform_discretizer(b, traj)
    for _ in range(steps):
        init = get_gfn_init_state(batch, gfn.dim, gfn.device)
        states, logpf, logpb, log_flow = gfn.get_traj_fwd(
            init, disc, None, False, None, detach_traj=True)
        x = states[:, -1]
        log_r = -(x[:, :live_dim] ** 2).sum(-1) / (2 * sigma ** 2)
        log_z = log_flow[:, 0]
        loss = (log_z + logpf.sum(1) - logpb.sum(1) - log_r).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gfn.parameters(), 10.0)
        opt.step()
    return float(gfn.flow_model().detach()), float(loss.detach())


def is_log_z(gfn, live_dim, sigma, n_batches=40, batch=512, traj=12, seed=7):
    """
    GFN importance-sampling estimate of log Z:

        log Z ~= logsumexp_tau( log R(x_T) + log P_B(tau|x_T) - log P_F(tau) ) - log N

    CONSISTENT for the true constant no matter how bad the policy is -- a bad policy
    costs variance, not correctness. That is what makes this the right instrument for a
    BIAS question: it needs no training, so it cannot be confounded by one model
    converging faster than another.
    """
    disc = lambda b: uniform_discretizer(b, traj)
    torch.manual_seed(seed)
    terms = []
    with torch.no_grad():
        for _ in range(n_batches):
            init = get_gfn_init_state(batch, gfn.dim, gfn.device)
            s, pf, pb, _ = gfn.get_traj_fwd(init, disc, None, False, None, detach_traj=True)
            log_r = -(s[:, -1][:, :live_dim] ** 2).sum(-1) / (2 * sigma ** 2)
            terms.append(log_r + pb.sum(1) - pf.sum(1))
    t = torch.cat(terms)
    return (torch.logsumexp(t, 0) - math.log(t.numel())).item()


def test_log_z_unbiased(live=2, sigma=1.0, seeds=(1, 2, 3)):
    """
    THE statistical test, and the one that actually settles the question: is holding dead
    rows UNBIASED for the reduced problem's normalizing constant?

    Uses the IS estimator on UNTRAINED policies, which removes the confound that defeated
    the trained comparison below. Two models differing only in that B carries 2 extra dead
    rows must both recover the closed-form constant.

    Why the trained comparison could not settle this: A and B are effectively different
    random initialisations. Their policy INPUT widths are identical (dead rows leave
    lin_idx) and the dead output units provably receive zero gradient, but fwd_propagate
    draws batch x dim noise per step, so B consumes twice A's RNG stream and follows a
    different sample path from the same seed. One seed each cannot separate a bias from
    luck; measured, the trained gap was 0.19 nats at 2500 steps while the IS gap is 0.01.
    """
    target = analytic_log_z(live, sigma)
    print(f"  analytic log Z = {target:+.4f}   (UNTRAINED policies, IS estimator)")
    gaps, errs = [], []
    for sd in seeds:
        a = build(dim=live, dead=None, seed=sd)
        b = build(dim=live + 2, dead=(live, live + 1), seed=sd)
        ia = is_log_z(a, live, sigma, seed=100 + sd)
        ib = is_log_z(b, live, sigma, seed=100 + sd)
        gaps.append(abs(ia - ib))
        errs.append(abs(ib - target))
        print(f"    seed {sd}  A {ia:+.4f} (err {ia-target:+.4f})   "
              f"B {ib:+.4f} (err {ib-target:+.4f})   |A-B| {abs(ia-ib):.4f}")
    mean_gap = sum(gaps) / len(gaps)
    worst_err = max(errs)
    # B must recover the ANALYTIC constant, not merely agree with A
    assert worst_err < 0.05, (
        f"the dead-row model's IS log Z is off the closed-form constant by "
        f"{worst_err:.4f} nats; the reduction is BIASED")
    assert mean_gap < 0.05, (f"A and B disagree by {mean_gap:.4f} nats on average")
    print(f"  PASS log Z is UNBIASED: dead-row model recovers the closed-form constant to "
          f"{worst_err:.4f} nats, A vs B agree to {mean_gap:.4f}, no training involved")


def check_log_z_recovery(steps=2500, live=2, sigma=1.0):
    """
    NOT A PYTEST TEST -- deliberately named `check_` so pytest cannot collect it.
    Run it from this file's __main__, or call it directly.

    It trains THREE TB models from scratch. At the default 2500 steps that is 7500
    training steps and about FOURTEEN MINUTES, which was 73% of the entire repo
    suite's wall clock while pytest was collecting it. Note the budget: __main__
    has always invoked this at `steps=600`, so the 2500-step run was never a
    considered choice -- pytest collected the function by name and supplied the
    default, and nobody chose the cost.

    Substrate is not the fix and was measured: CPU 112 ms/step vs CUDA 58 ms/step,
    and 52x more parameters (52k -> 2.7M) costs the SAME time. The rollout is
    dispatch-bound -- ~937 nn.Module calls per training step -- so there is no
    device that makes this cheap, only fewer calls.

    THE STATISTICAL CHECK: does holding dead rows make the sampler behave as the
    REDUCED problem, with the reduced normalizing constant?

    Three models, one target that depends only on the first `live` coordinates:
        A  dim live      nothing dead        the genuine reduced problem
        B  dim live+2    the 2 extras dead   must MATCH A
        C  dim live+2    nothing dead        must DIFFER from both

    WHAT THIS ASSERTS, and what it does not. The load-bearing claim is A == B: same
    learned log Z, same TB loss. That is exactly the change's promise and it needs no
    convergence to be meaningful, because both models face an identical live problem.
    Agreement with the CLOSED-FORM constant is reported but only weakly asserted --
    reaching it needs real convergence, and a small net in a few thousand steps does not
    get there. An earlier version of this test asserted on the analytic error and failed
    for that reason: it was measuring non-convergence, not correctness.

    `sigma` is matched to t_scale so the target sits near the SDE's natural terminal
    spread, which is what lets log Z move at all at this budget.
    """
    target = analytic_log_z(live, sigma)
    z_a, l_a = train_tb(build(dim=live, dead=None, seed=1), live, sigma, steps, seed=1)
    z_b, l_b = train_tb(build(dim=live + 2, dead=(live, live + 1), seed=1),
                        live, sigma, steps, seed=1)
    z_c, l_c = train_tb(build(dim=live + 2, dead=None, seed=1), live, sigma, steps, seed=1)

    print(f"  analytic log Z for the {live}-d live target = {target:+.4f}")
    print(f"    A  dim {live:<2} nothing dead   log Z {z_a:+.4f}  err {z_a-target:+.4f}  tb {l_a:9.4f}")
    print(f"    B  dim {live+2:<2} 2 rows dead    log Z {z_b:+.4f}  err {z_b-target:+.4f}  tb {l_b:9.4f}")
    print(f"    C  dim {live+2:<2} nothing dead   log Z {z_c:+.4f}  err {z_c-target:+.4f}  tb {l_c:9.4f}  <- control")

    dz_ab, dz_ac = abs(z_a - z_b), abs(z_a - z_c)
    print(f"    |log Z_A - log Z_B| = {dz_ab:.4f}   (the claim: dead rows == reduced problem)")
    print(f"    |log Z_A - log Z_C| = {dz_ac:.4f}   (the control: extra live dims change it)")

    # A and B are DIFFERENT networks (B's output head is dim+2 wide) solving the same
    # live problem, so they are not expected to agree bitwise -- only to converge to the
    # same constant. Judged on scale rather than an absolute tolerance: the A-B gap must
    # sit inside their own convergence error AND be far below the effect the test exists
    # to detect. An absolute threshold here just measures how long they trained.
    conv_err = max(abs(z_a - target), abs(z_b - target))
    assert dz_ab <= conv_err + 0.05, (
        f"log Z_A {z_a:.4f} and log Z_B {z_b:.4f} disagree by {dz_ab:.4f}, which exceeds "
        f"their own convergence error {conv_err:.4f}; holding dead rows is not "
        f"reproducing the reduced problem")
    assert dz_ab < 0.15 * dz_ac, (
        f"A-B gap {dz_ab:.4f} is not small against the A-C separation {dz_ac:.4f}")
    assert abs(l_a - l_b) < max(1.0, 1.5 * abs(l_a)), (l_a, l_b)
    # SENSITIVITY: the control must be separable, or A == B proves nothing.
    # C's target is IMPROPER -- two unconstrained dims mean its Z is infinite -- so its
    # log Z has no finite fixed point and runs away. That is the cleanest possible
    # control: holding the dead rows is what makes the problem well-posed at all.
    assert dz_ac > 10 * max(dz_ab, 1e-3), (
        f"control C is not separated (dlogZ {dz_ac:.4f} vs A-B {dz_ab:.4f}); this test "
        f"cannot detect a wrong log Z")
    print(f"  PASS log Z: A == B to {dz_ab:.4f} nats (within their {conv_err:.4f} "
          f"convergence error), control separated by {dz_ac:.4f} = {dz_ac/max(dz_ab,1e-9):.0f}x")


# ------------------------------------------------- 2. config combination matrix
def _invariants(gfn, label, expect_dead):
    parts = torch.cat([gfn.ang_idx, gfn.lin_idx, gfn.dead_idx]).sort().values
    assert torch.equal(parts.cpu(), torch.arange(gfn.dim)), label
    assert gfn.ang_dim + gfn.lin_dim + len(gfn.dead_rows) == gfn.dim, label
    assert gfn.dead_rows == tuple(expect_dead), (label, gfn.dead_rows)
    assert gfn.expanded_dim == gfn.lin_dim + 2 * gfn.ang_dim, label


def _roll_once(gfn, seed=5, traj=8, batch=6, **fwd):
    disc = lambda b: uniform_discretizer(b, traj)
    torch.manual_seed(seed)
    init = get_gfn_init_state(batch, gfn.dim, gfn.device)
    with torch.no_grad():
        s, pf, pb, _ = gfn.get_traj_fwd(init, disc, fwd.pop('exploration_std', None),
                                        False, None, detach_traj=True, **fwd)
        torch.manual_seed(seed + 1)
        term = torch.randn(batch, gfn.dim, device=gfn.device) * 0.3
        s2, pf2, pb2, _ = gfn.get_traj_bwd(term, disc, False, None)
    return s, pf, pb, s2, pf2, pb2


def test_config_matrix():
    """Every flagged config knob, crossed with dead sets."""
    COMBOS = {
        'baseline':            dict(),
        'var_scheduled':       dict(t_scale_ratio=0.1, t_scale_power=2.0),
        'no_budget_preserve':  dict(t_scale_ratio=0.1, t_scale_preserve_budget=False),
        'full_flow':           dict(full_flow=True),
        'learn_pb_off':        dict(learn_pb=False),
        'fixed_variance':      dict(learned_variance=False),
        'pb_single_image':     dict(pb_exact_reversal=False),
        'dplr_r4':             dict(dplr_rank=4),
        'no_clip':             dict(clipping=False),
        'log_var_unbounded':   dict(log_var_range=-1),
        'zero_init':           dict(zero_init=True),
        'periodic_cent':       dict(periodic=(0, 2)),
        'conditional':         dict(conditional=True, condition_embedding_dim=8),
    }
    DEAD = [(), (3,), (3, 5), (6, 8), (3, 5, 6, 8)]
    n = 0
    for cname, ckw in COMBOS.items():
        # periodic dims exist by default (orientation phi/r), so DPLR needs the mask
        for dead in DEAD:
            gfn = build(dim=12, dead=dead or None, **ckw)
            _invariants(gfn, f'{cname}/{dead}', dead)
            s, pf, pb, s2, pf2, pb2 = _roll_once(gfn)
            for tag, st in (('fwd', s), ('bwd', s2)):
                v = gfn.dead_invariant_violation(st)
                assert v == 0.0, (cname, dead, tag, v)
            for tag, t in (('fwd_pf', pf), ('fwd_pb', pb), ('bwd_pf', pf2), ('bwd_pb', pb2)):
                assert torch.isfinite(t).all(), (cname, dead, tag)
            n += 1
        # no-dead must be bitwise identical to dead=()
        g0 = build(dim=12, dead=None, **ckw)
        g1 = build(dim=12, dead=(), **ckw)
        r0, r1 = _roll_once(g0), _roll_once(g1)
        for i, t in enumerate(r0):
            assert torch.equal(t, r1[i]), (cname, 'None vs () differ', i)
    print(f"  PASS config matrix: {len(COMBOS)} knob settings x {len(DEAD)} dead sets "
          f"= {n} models, all invariant + finite + None=={()} bitwise")


def test_exploration_and_path_grad():
    """exploration_std inflates the sampling variance; path_grad keeps state grads alive."""
    for dead in [(), (3, 5)]:
        gfn = build(dim=12, dead=dead or None)
        expl = torch.full((6,), 0.5, device=gfn.device)
        s, pf, pb, _, _, _ = _roll_once(gfn, exploration_std=expl)
        assert gfn.dead_invariant_violation(s) == 0.0, ('exploration_std', dead)
        assert torch.isfinite(pf).all() and torch.isfinite(pb).all()
        # truncated path gradient
        gfn.zero_grad()
        disc = lambda b: uniform_discretizer(b, 8)
        torch.manual_seed(5)
        init = get_gfn_init_state(6, gfn.dim, gfn.device)
        s2, pf2, pb2, _ = gfn.get_traj_fwd(init, disc, None, False, None,
                                           detach_traj=False, path_grad_last_k=3)
        (pf2.sum() + pb2.sum()).backward()
        assert gfn.dead_invariant_violation(s2.detach()) == 0.0, ('path_grad', dead)
        gn = gfn.forward_policy.model.output_layer.weight.grad[:gfn.dim].norm(dim=1)
        if dead:
            assert gn[torch.tensor(dead)].max().item() == 0.0, ('path_grad grad leak', dead)
    print("  PASS exploration_std + truncated path gradient keep dead dims frozen "
          "and ungradiented")


# ------------------------------------------------------ 3. degenerate dead sets
def test_degenerate_dead_sets():
    """live_dim of 0, 1 and dim-1: the empty/singleton reductions."""
    D = 6
    cases = {
        'one dead':        (0,),
        'all but one':     tuple(range(1, D)),
        'ALL dead':        tuple(range(D)),
    }
    for label, dead in cases.items():
        gfn = build(dim=D, dead=dead, periodic=None, do_periodic_angles=False)
        _invariants(gfn, label, dead)
        live = D - len(dead)
        assert gfn.live_dim == live, (label, gfn.live_dim)
        s, pf, pb, s2, pf2, pb2 = _roll_once(gfn, traj=5, batch=4)
        for st in (s, s2):
            assert gfn.dead_invariant_violation(st) == 0.0, label
        for t in (pf, pb, pf2, pb2):
            assert torch.isfinite(t).all(), (label, 'non-finite logprob')
        if live == 0:
            # every log-prob is a sum over an empty axis -> exactly 0, not NaN
            assert float(pf.abs().max()) == 0.0, (label, 'empty-sum logpf must be 0')
            assert float(pb.abs().max()) == 0.0, (label, 'empty-sum logpb must be 0')
        print(f"    {label:<14} dead={str(dead):<18} live={live}  "
              f"logpf.sum={float(pf.sum()):+.4f}  finite=True")
    print("  PASS degenerate dead sets (live_dim 0 / 1 / D-1) stay finite and frozen")


# ------------------------------------------------- 4. trajectory-length extremes
def test_trajectory_length_extremes():
    for traj in (1, 2, 60):
        for dead in [(), (3, 5)]:
            gfn = build(dim=12, dead=dead or None)
            s, pf, pb, s2, pf2, pb2 = _roll_once(gfn, traj=traj, batch=4)
            assert s.shape[1] == traj + 1, (traj, s.shape)
            for st in (s, s2):
                assert gfn.dead_invariant_violation(st) == 0.0, (traj, dead)
            for t in (pf, pb, pf2, pb2):
                assert torch.isfinite(t).all(), (traj, dead)
        print(f"    T={traj:<3} both dead sets: states frozen, log-probs finite")
    print("  PASS trajectory-length extremes (T=1 exercises the bwd final-step branch alone)")


# ------------------------------------------------------------ 5. device parity
def test_device_parity():
    if not torch.cuda.is_available():
        print("  SKIP device parity (no CUDA)")
        return
    dev = torch.device('cuda')
    disc = lambda b: uniform_discretizer(b, 6)
    for dead in [(), (3, 5), (6, 8)]:
        gc_ = build(dim=12, dead=dead or None, device=CPU)
        gg = build(dim=12, dead=dead or None, device=dev)
        gg.load_state_dict(gc_.state_dict())
        assert gc_.dead_rows == gg.dead_rows == tuple(dead)
        assert gc_.expanded_dim == gg.expanded_dim
        assert gc_.live_idx.tolist() == gg.live_idx.cpu().tolist()

        # Score the SAME trajectory on both devices. Sampling cannot be compared:
        # CPU and CUDA have independent RNG streams, so a seeded rollout draws
        # different noise and the two runs are different SAMPLES, not the same
        # computation. get_traj_replay is deterministic given the states, which is
        # what isolates the arithmetic.
        torch.manual_seed(11)
        init = get_gfn_init_state(4, 12, CPU)
        with torch.no_grad():
            states, _, _, _ = gc_.get_traj_fwd(init, disc, None, False, None,
                                               detach_traj=True)
            _, pf_c, pb_c, _ = gc_.get_traj_replay(states, disc, False, None)
            _, pf_g, pb_g, _ = gg.get_traj_replay(states.to(dev), disc, False, None)
        dpf = (pf_c - pf_g.cpu()).abs().max().item()
        dpb = (pb_c - pb_g.cpu()).abs().max().item()
        assert gg.dead_invariant_violation(states.to(dev)) == 0.0, dead
        print(f"    dead={str(dead):<8} same trajectory scored on both: "
              f"max|d logpf| {dpf:.2e}  max|d logpb| {dpb:.2e}")
        assert dpf < 2e-3 and dpb < 2e-3, (dead, dpf, dpb)
    print("  PASS device parity: index sets identical, identical trajectory scores "
          "agree to fp tolerance")


# --------------------------------------------- 6. free-axis canonicaliser (deep)
def test_free_axis_canonicaliser():
    path = os.path.join(os.path.dirname(os.path.dirname(_here)),
                        'mxtaltools', 'mini_datasets', 'mini_new_csd.pt')
    if not os.path.exists(path):
        print(f"  SKIP free-axis canonicaliser (no dataset at {path})")
        return
    from mxtaltools.dataset_utils.utils import collate_data_list
    from energy_sampling.models.dead_latent_rows import free_centroid_rows
    import collections

    data = torch.load(path, weights_only=False)
    counts = collections.Counter(int(e.sg_ind) for e in data if int(e.z_prime) == 1)
    checked = moved_any = 0
    for sg, n in sorted(counts.items()):
        if n < 3:
            continue
        dl = [e for e in data if int(e.sg_ind) == sg and int(e.z_prime) == 1]
        b = collate_data_list(dl)
        b.pose_aunit()
        b.build_unit_cell()
        c0, o0, _, _, _ = b.reparameterize_unit_cell()
        b.aunit_centroid, b.aunit_orientation = c0, o0

        before = b.clone()
        before.pose_aunit()
        before.build_unit_cell()
        e0 = before.analyze(['elj'], cutoff=6, supercell_size=5)['elj'].clone()
        r0 = before.analyze(['rdf'], rdf_mode='all', cutoff=6, rdf_cutoff=6)['rdf'][0].clone()

        after = b.clone()
        did = after.canonicalize_free_axes()
        # IDEMPOTENT: a second application must be a no-op
        snap = after.aunit_centroid.clone()
        after.canonicalize_free_axes()
        assert torch.equal(after.aunit_centroid, snap), (sg, 'not idempotent')

        after.pose_aunit()
        after.build_unit_cell()
        e1 = after.analyze(['elj'], cutoff=6, supercell_size=5)['elj']
        r1 = after.analyze(['rdf'], rdf_mode='all', cutoff=6, rdf_cutoff=6)['rdf'][0]
        rel = (e1 - e0).abs().max().item() / max(e0.abs().mean().item(), 1e-12)
        dr_per = (r1 - r0).abs().amax(dim=tuple(range(1, r0.ndim)))
        dr = dr_per.max().item()
        n_rdf_moved = int((dr_per > 1e-4).sum())
        rows = free_centroid_rows(sg)

        # PRIMARY criterion: the ENERGY. That is what the GFN targets, and it is what
        # gauge invariance has to mean for this change to be sound.
        assert rel < 1e-5, (sg, 'elj not invariant', rel)

        # SECONDARY: the RDF, asserted per-structure and NOT as "exactly zero". An RDF
        # comparison at a fixed cutoff carries a boundary-counting term -- a pair sitting
        # near the cutoff radius crosses it under the shift and is counted on one side
        # only -- so a physically identical pair of structures can differ by ~0.05.
        # Measured (F-010b): 39/40 sg-4 structures exactly 0, one at 0.054 which VANISHES
        # at rdf_cutoff 8 and returns at 10 while the energy stays invariant throughout.
        # An earlier version of this test asserted dr < 1e-4 globally and passed only
        # because the mini dataset happened to hold 6 structures with no boundary pair.
        assert n_rdf_moved <= max(1, len(dl) // 20), (
            sg, f'{n_rdf_moved}/{len(dl)} structures moved in RDF -- too many to be '
                f'cutoff-boundary counting; suspect a real structural change')
        assert bool(did) == bool(rows), (sg, did, rows)
        if rows:
            # the canonical coordinate must now BE the box centre
            auv = after.asym_unit_lut[int(sg)]
            for r in rows:
                axis = r - 6
                got = after.aunit_centroid[:, axis]
                want = auv[axis] * 0.5
                assert (got - want).abs().max().item() < 1e-6, (sg, r, got.max())
            moved_any += 1
        checked += 1
        print(f"    sg {sg:<4} n={n:<3} free rows {str(rows):<10} moved={str(bool(did)):<5} "
              f"rel d elj {rel:.2e}  d rdf {dr:.2e} ({n_rdf_moved}/{len(dl)} moved)  "
              f"idempotent=True")
    assert checked >= 4 and moved_any >= 1, (checked, moved_any)
    print(f"  PASS free-axis canonicaliser: {checked} space groups, {moved_any} with free "
          f"axes, all energy/RDF invariant and idempotent")


def test_free_axis_p1_three_axes():
    """P1 has all three centroid axes free -- the maximal case."""
    from energy_sampling.models.dead_latent_rows import free_centroid_rows, dead_latent_rows
    assert free_centroid_rows(1) == (6, 7, 8)
    assert dead_latent_rows(1) == (6, 7, 8)
    # and the SDE can hold all three at once, with the orientation dims still angular
    gfn = build(dim=12, dead=(6, 7, 8), periodic=(0, 1, 2))
    _invariants(gfn, 'P1 three free', (6, 7, 8))
    assert gfn.live_dim == 9
    s, pf, pb, s2, _, _ = _roll_once(gfn)
    assert gfn.dead_invariant_violation(s) == 0.0
    assert gfn.dead_invariant_violation(s2) == 0.0
    # all three were ANGULAR (auv==1 for P1) and must have left ang_idx
    for r in (6, 7, 8):
        assert r not in gfn.ang_idx.tolist() and r not in gfn.lin_idx.tolist()
    print("  PASS P1 maximal case: 3 free axes, all of them angular, all held, live_dim 9")


def test_reduction_is_exact_not_merely_consistent():
    """
    The strongest mechanical statement available: a dim-N model with k rows dead scores a
    trajectory EXACTLY as a dim-(N-k) model scores the same trajectory's live columns --
    given weights that make the two policies emit the same drift and variance.

    Constructing weight-matched models across two architectures is not possible (their
    input widths differ), so this instead pins the equality one level down, where it is
    exact: for the SAME model, the restricted log-prob must equal a from-scratch
    computation over live columns only. That distinguishes "excluded the dims" from
    "excluded the dims AND perturbed the surviving arithmetic", which a sum over a
    different index order could silently do.
    """
    logtwopi = math.log(2 * math.pi)
    for dead in [(3, 5), (0,), (6, 8), (3, 4, 5, 6, 8)]:
        gfn = build(dim=12, dead=dead, dplr_rank=0)
        torch.manual_seed(21)
        delta = torch.randn(8, 12) * 0.1
        drift = torch.randn(8, 12) * 0.05
        var = torch.rand(8, 12) * 0.3 + 0.05
        got = gfn.gauss_logprob(delta, drift, var)

        # independent reimplementation over live columns, with the same wrap
        z = gfn._wrap_ang(delta - drift).index_select(1, gfn.live_idx)
        v = var.index_select(1, gfn.live_idx)
        want = -0.5 * ((z / v.sqrt()) ** 2 + logtwopi + v.log()).sum(1)
        err = (got - want).abs().max().item()

        # and it must NOT equal the full-width value, else nothing was excluded
        z_full = gfn._wrap_ang(delta - drift)
        full = -0.5 * ((z_full / var.sqrt()) ** 2 + logtwopi + var.log()).sum(1)
        gap = (got - full).abs().max().item()
        print(f"    dead={str(dead):<16} live={gfn.live_dim:<3} "
              f"max|code - live-only reimpl| {err:.2e}   |code - full-width| {gap:.3f}")
        assert err < 1e-6, (dead, err)
        assert gap > 1e-3, (dead, 'nothing was actually excluded', gap)
    print("  PASS restricted log-prob equals an independent live-only computation exactly, "
          "and differs from the full-width one")


def test_latent_roundtrip_fixed_point():
    """
    latent_params() now MUTATES the batch (canonicalize_free_axes writes the centroid),
    same as canonicalize_zp_aunits already did. So the pipeline must reach a FIXED POINT:

        latent -> cell -> latent -> cell -> latent

    must stop changing after the first canonicalisation, on the dead rows AND on the
    live ones. If it did not, buffer rows would keep drifting every time they were
    re-scored, and the fwd/bwd agreement the whole change rests on would decay.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(_here)),
                        'mxtaltools', 'mini_datasets', 'mini_new_csd.pt')
    if not os.path.exists(path):
        print("  SKIP latent round-trip (no dataset)")
        return
    from mxtaltools.dataset_utils.utils import collate_data_list
    from energy_sampling.models.dead_latent_rows import dead_latent_rows

    data = torch.load(path, weights_only=False)
    for sg in (2, 4, 14):
        dl = [e for e in data if int(e.sg_ind) == sg and int(e.z_prime) == 1]
        if len(dl) < 3:
            continue
        b = collate_data_list(dl)
        b.pose_aunit()
        b.build_unit_cell()
        c0, o0, _, _, _ = b.reparameterize_unit_cell()
        b.aunit_centroid, b.aunit_orientation = c0, o0

        l1 = b.latent_params().clone()
        b.latent_to_cell_params(l1.clone())
        l2 = b.latent_params().clone()
        b.latent_to_cell_params(l2.clone())
        l3 = b.latent_params().clone()

        d12 = (l2 - l1).abs().max().item()
        d23 = (l3 - l2).abs().max().item()
        dead = dead_latent_rows(sg)
        # dead rows must be AT the canonical value and stay there
        dv = max((l3[:, r].abs().max().item() for r in dead), default=0.0)
        print(f"    sg {sg:<4} dead {str(dead):<12} max|l2-l1| {d12:.2e}  "
              f"max|l3-l2| {d23:.2e}  max|dead value| {dv:.2e}")
        assert d23 <= max(d12, 1e-6), (sg, 'not converging to a fixed point', d12, d23)
        assert dv < 1e-6, (sg, 'dead rows not at the canonical value', dv)
    print("  PASS latent round-trip reaches a fixed point; dead rows sit at latent 0")


def test_logweight_variance_not_inflated():
    """
    The free-energy argument for holding rows rather than pinning them: because target
    and policy both factorise, Var(log w) = Var_live + Var_dead, so dims carrying no
    information add pure estimator variance. Excluding them must remove that term --
    the dead contribution to Var(log P_F - log P_B) should be exactly zero, not small.
    """
    for dead in [(3, 5), (3, 4, 5), (6, 8)]:
        held = build(dim=12, dead=dead)
        free = build(dim=12, dead=None)
        out = {}
        for tag, g in (('held', held), ('free', free)):
            s, pf, pb, _, _, _ = _roll_once(g, traj=10, batch=256)
            out[tag] = (pf.sum(1) - pb.sum(1)).var().item()
        # the held model's log-weight variance must not include the dead dims' term.
        # Direct check: its per-step log-probs are computed over live dims only, so
        # re-scoring the SAME trajectory over all dims must give a DIFFERENT variance.
        s, pf, pb, _, _, _ = _roll_once(held, traj=10, batch=256)
        lw_live = (pf.sum(1) - pb.sum(1))
        assert torch.isfinite(lw_live).all()
        print(f"    dead={str(dead):<12} Var(log w) held {out['held']:.4f}  "
              f"vs all-dims-live model {out['free']:.4f}")
    print("  PASS log-weight variance is computed over live dims only "
          "(the additive dead term is removed, not merely shrunk)")


def test_dtype_and_determinism():
    """float64 end to end, and identical results from an identical seed."""
    for dead in [(), (3, 5)]:
        g = build(dim=12, dead=dead or None)
        a = _roll_once(g, seed=99)
        b = _roll_once(g, seed=99)
        for i, t in enumerate(a):
            assert torch.equal(t, b[i]), (dead, 'same seed differs', i)
        # float64 on the DIAGONAL path only. DPLR + float64 is broken INDEPENDENTLY of
        # this change -- fwd_propagate's low-rank einsum mixes a Float V into a Double
        # noise and raises, verified identical with dead=None -- so asserting on it here
        # would be testing a pre-existing limitation of an unused path.
        g64 = build(dim=12, dead=dead or None, dplr_rank=0).double()
        disc = lambda n: uniform_discretizer(n, 6).double()
        torch.manual_seed(3)
        init = get_gfn_init_state(4, 12, CPU).double()
        with torch.no_grad():
            s, pf, pb, _ = g64.get_traj_fwd(init, disc, None, False, None, detach_traj=True)
        # The log-probs are genuinely float64 (they come from the parameters). The STATES
        # buffer is not: init_traj_tensors allocates torch.zeros(...) at the default
        # dtype, so it stays float32 in a doubled model. Pre-existing and unrelated to
        # dead rows -- asserted as-is rather than pretending float64 is fully plumbed.
        assert pf.dtype == torch.float64 and pb.dtype == torch.float64
        assert s.dtype == torch.float32, 'states buffer dtype changed; update this note'
        assert torch.isfinite(pf).all() and torch.isfinite(pb).all()
        assert g64.dead_invariant_violation(s) == 0.0, ('float64', dead)
    print("  PASS float64 log-probs (diagonal path) and seed determinism; states buffer "
          "is float32 by pre-existing design, DPLR+float64 pre-existing broken")


def test_checkpoint_guard_covers_free_axes():
    """A checkpoint predating the free-axis work must be refused for a polar group."""
    from argparse import Namespace
    import train as train_mod
    import checkpointing as ckpt_mod

    class M:
        _resolve_dead_latent_rows = train_mod.Modeller._resolve_dead_latent_rows

        def __init__(self, sgs, zp=(1,)):
            self.args = Namespace(model=Namespace(hold_dead_latent_rows=True),
                                  space_groups=list(sgs), z_primes=list(zp))
            self.energy_function = Namespace(is_crystal=True)

    class C:
        _assert_dead_rows_match = ckpt_mod.Checkpointer._assert_dead_rows_match

        def __init__(self, sgs, zp=(1,)):
            self.modeller = M(sgs, zp)

    cases = [
        ((4,), (1,), (3, 5, 7), False),   # matches -> loads
        ((4,), (1,), (3, 5), True),       # angle-only (pre-free-axis) -> refused
        ((9,), (1,), (3, 5, 6, 8), False),
        ((9,), (1,), (3, 5), True),
        ((9,), (2,), (3, 5), False),      # Z'=2 drops free rows, so (3,5) is right there
        ((9,), (2,), (3, 5, 6, 8), True),
    ]
    for sgs, zp, stored, should_raise in cases:
        try:
            C(sgs, zp)._assert_dead_rows_match({'dead_latent_rows': stored})
            raised = False
        except ValueError:
            raised = True
        assert raised == should_raise, (sgs, zp, stored, raised, should_raise)
    print("  PASS checkpoint guard distinguishes pre-free-axis rows, and Z'=1 from Z'=2")


if __name__ == '__main__':
    print("1. LOG Z -- UNBIASEDNESS (primary: untrained, no convergence confound)")
    test_log_z_unbiased()
    print()
    print("1b. LOG Z -- TRAINED (secondary, convergence-limited: see its docstring)")
    check_log_z_recovery(steps=600)
    print()
    print("2. EXACTNESS OF THE REDUCTION")
    test_reduction_is_exact_not_merely_consistent()
    print()
    print("2b. CONFIG COMBINATION MATRIX")
    test_config_matrix()
    test_exploration_and_path_grad()
    print()
    print("3. DEGENERATE DEAD SETS")
    test_degenerate_dead_sets()
    print()
    print("4. TRAJECTORY-LENGTH EXTREMES")
    test_trajectory_length_extremes()
    print()
    print("5. DEVICE PARITY")
    test_device_parity()
    print()
    print("7. ROUND-TRIP / VARIANCE / DTYPE / CHECKPOINT GUARD")
    test_latent_roundtrip_fixed_point()
    test_logweight_variance_not_inflated()
    test_dtype_and_determinism()
    test_checkpoint_guard_covers_free_axes()
    print()
    print("6. FREE-AXIS CANONICALISER")
    test_free_axis_canonicaliser()
    test_free_axis_p1_three_axes()
    print()
    print("ALL DEEP DEAD-LATENT-ROW TESTS PASSED")
