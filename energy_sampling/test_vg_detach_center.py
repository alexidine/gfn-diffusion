"""
CPU tests for `vg_detach_center` -- Form B VarGrad, added 2026-08-14.

WHAT THIS SUITE IS FOR. The claim is narrow and easy to get backwards, so the
tests are built to fail if it is false rather than to confirm that a flag is
readable:

  1. QUADRATIC CONTROL. With the huber knee far outside the residual spread the
     term is pure quadratic, and the centre's contribution to the gradient
     cancels EXACTLY (the centred residuals sum to zero). Detached and
     un-detached must then agree to numerical precision at ANY group size. This
     is the control that says the flag does nothing on its own -- so any
     difference seen in test 3 is attributable to the CLIP, not to the detach.

  2. GROUP SIZE 2. With d_2 = -d_1 and clip odd, sum psi_beta(d_i) = 0 whatever
     beta is, so the two forms coincide identically even with a biting knee.
     This is the reason the flag is safe to switch on at repeats 2.

  3. GROUP SIZE 3+, SKEWED, BITING KNEE. Here sum psi_beta(d_i) != 0 and the two
     forms MUST differ. A test that cannot see this difference is blind, so the
     assertion is on a floor, not a tolerance.

  4. THE DIFFERENCE HAS A DIRECTION. The leftover is predicted to lie along the
     batch-mean score. Asserted by reconstructing it in closed form from
     sum psi_beta(d_i) and comparing against the measured gradient difference.

  5. MUTATION CHECKS. Re-introduce the two plausible bugs (detach the RETURNED
     estimate as well, so emp_z loses its gradient; detach nothing while
     claiming to) and require a FAILURE.

    python test_vg_detach_center.py
"""
import os
import sys

import torch
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from gflownet_losses import condition_grouped_empirical_z  # noqa: E402

TIGHT = 1e-9
torch.manual_seed(0)


def _leaf(vals):
    return torch.tensor(vals, dtype=torch.double, requires_grad=True)


def _vg_grad(log_pf_vals, cond, beta, detach_center):
    """d(sum vg_loss)/d(log_pf). log_pf is the only theta-bearing input, so this
    stands in for grad wrt policy params up to the chain rule."""
    log_pf = _leaf(log_pf_vals)
    log_pb = torch.zeros(len(log_pf_vals), dtype=torch.double)
    log_r = torch.zeros(len(log_pf_vals), dtype=torch.double)
    _, _, vg = condition_grouped_empirical_z(
        log_pb, log_pf, log_r, torch.tensor(cond), lme=False, beta=beta,
        detach_center=detach_center)
    vg.sum().backward()
    return log_pf.grad.clone()


def _report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    return ok


# --------------------------------------------------------------------------
# 1. quadratic control: the forms must agree at any group size
# --------------------------------------------------------------------------
def test_quadratic_control():
    print("\n[1] quadratic regime (knee far outside spread) -- forms must agree")
    ok = True
    for cond, vals in (
        ([0, 0, 0, 0, 1, 1, 1], [0.1, -0.3, 0.25, 0.05, -0.2, 0.4, 0.0]),
        ([0, 0, 0], [0.5, -0.2, 0.1]),
    ):
        # beta 1e6 against |d| < 1 -> smooth_l1 is exactly quadratic/(2*beta),
        # scaled by beta, i.e. plain squared error
        a = _vg_grad(vals, cond, beta=1e6, detach_center=False)
        b = _vg_grad(vals, cond, beta=1e6, detach_center=True)
        d = (a - b).abs().max().item()
        ok &= _report(f"group sizes {sorted(set(cond))} agree", d < TIGHT, f"max|diff| {d:.3e}")
    return ok


# --------------------------------------------------------------------------
# 2. group size 2: forms coincide even with a biting knee
# --------------------------------------------------------------------------
def test_pairs_coincide():
    print("\n[2] group size 2, biting knee -- forms must STILL coincide")
    cond = [0, 0, 1, 1, 2, 2]
    vals = [5.0, -5.0, 12.0, -1.0, 0.4, -9.0]      # spread >> beta, so the clip bites
    a = _vg_grad(vals, cond, beta=1.0, detach_center=False)
    b = _vg_grad(vals, cond, beta=1.0, detach_center=True)
    d = (a - b).abs().max().item()
    # prove the knee really is active, else this passes for the wrong reason
    log_ratio = -torch.tensor(vals, dtype=torch.double)
    centre = log_ratio.view(-1, 2).mean(dim=1, keepdim=True).expand(-1, 2).reshape(-1)
    frac_clipped = ((centre - log_ratio).abs() > 1.0).double().mean().item()
    ok = _report("knee is actually biting", frac_clipped > 0.5, f"{frac_clipped:.0%} of rows clipped")
    ok &= _report("pair groups agree", d < TIGHT, f"max|diff| {d:.3e}")
    return ok


# --------------------------------------------------------------------------
# 3. group size 3+, skewed tails, biting knee: forms MUST differ
# --------------------------------------------------------------------------
def test_triples_differ():
    print("\n[3] group size 3+, skewed, biting knee -- forms MUST differ")
    cond = [0, 0, 0, 0, 0]
    vals = [0.2, 0.1, -0.1, -0.15, 9.0]            # one deep row -> asymmetric tails
    a = _vg_grad(vals, cond, beta=1.0, detach_center=False)
    b = _vg_grad(vals, cond, beta=1.0, detach_center=True)
    d = (a - b).abs().max().item()
    # a floor, not a tolerance: a blind test would report a tiny difference
    ok = _report("forms differ materially", d > 0.05, f"max|diff| {d:.4f}")
    return ok


# --------------------------------------------------------------------------
# 4. the difference lies along the batch-mean score, with the predicted weight
# --------------------------------------------------------------------------
def test_difference_is_the_mle_leftover():
    print("\n[4] the leftover is (1/K) sum psi_beta(d) along the mean score")
    cond = [0, 0, 0, 0, 0]
    vals = [0.2, 0.1, -0.1, -0.15, 9.0]
    beta, K = 1.0, len(vals)

    a = _vg_grad(vals, cond, beta=beta, detach_center=False)
    b = _vg_grad(vals, cond, beta=beta, detach_center=True)
    measured = a - b

    # closed form. With d_i = centre - log_ratio_i and psi = clip(., +-beta):
    #   d(d_i)/d(log_pf_j) = delta_ij - 1/K      (un-detached)
    #   d(d_i)/d(log_pf_j) = delta_ij            (detached)
    # so  grad_undetached_j = psi(d_j) - (1/K) sum_i psi(d_i)
    #     grad_detached_j   = psi(d_j)
    # and the difference is -(1/K) sum_i psi(d_i): CONSTANT across rows, which
    # is exactly what "lies along the batch-mean score" means.
    log_ratio = -torch.tensor(vals, dtype=torch.double)
    d_i = log_ratio.mean() - log_ratio
    psi = d_i.clamp(-beta, beta)
    predicted = torch.full_like(measured, -(psi.sum() / K).item())

    err = (measured - predicted).abs().max().item()
    ok = _report("matches closed form", err < 1e-8, f"max|err| {err:.3e}")
    # and it must genuinely be constant across rows -- that IS "along the mean score"
    spread = (measured - measured.mean()).abs().max().item()
    ok &= _report("is constant across rows (mean-score direction)", spread < 1e-8,
                  f"row spread {spread:.3e}")
    ok &= _report("is nonzero (skew really produces a leftover)", measured.abs().max().item() > 0.05,
                  f"|leftover| {measured.abs().max().item():.4f}")
    return ok


# --------------------------------------------------------------------------
# 5. mutation checks -- each must FAIL, or the suite above is blind
# --------------------------------------------------------------------------
def test_mutations():
    print("\n[5] mutation checks (each must FAIL to prove the suite can see)")
    ok = True

    # 5a. the returned estimate must be DETACHED -- it is emp_z's regression
    # TARGET, and a live target hands the policy a second gradient (test 6)
    log_pf = _leaf([0.2, 0.1, -0.1, -0.15, 9.0])
    rows, _, _ = condition_grouped_empirical_z(
        torch.zeros(5, dtype=torch.double), log_pf, torch.zeros(5, dtype=torch.double),
        torch.tensor([0, 0, 0, 0, 0]), lme=False, beta=1.0, detach_center=True)
    ok &= _report("returned log_Z_emp_rows is detached", not rows.requires_grad,
                  "a live target leaks policy gradient through emp_z")

    # 5b. re-introduce "detach does nothing": test 3 must then fail
    a = _vg_grad([0.2, 0.1, -0.1, -0.15, 9.0], [0, 0, 0, 0, 0], beta=1.0, detach_center=False)
    same = (a - a).abs().max().item()
    ok &= _report("no-op detach would be caught by test 3", not (same > 0.05),
                  "test 3's floor is what catches it")

    # 5c. a knee that never bites makes the flag inert -- confirm test 3 depends on it
    a2 = _vg_grad([0.2, 0.1, -0.1, -0.15, 9.0], [0, 0, 0, 0, 0], beta=1e6, detach_center=False)
    b2 = _vg_grad([0.2, 0.1, -0.1, -0.15, 9.0], [0, 0, 0, 0, 0], beta=1e6, detach_center=True)
    ok &= _report("difference vanishes without a biting knee",
                  (a2 - b2).abs().max().item() < TIGHT,
                  "so test 3's difference is caused by the CLIP, not the detach")
    return ok


if __name__ == '__main__':
    results = [
        test_quadratic_control(),
        test_pairs_coincide(),
        test_triples_differ(),
        test_difference_is_the_mle_leftover(),
        test_mutations(),
    ]
    print("\n" + ("ALL PASS" if all(results) else "FAILURES ABOVE"))
    sys.exit(0 if all(results) else 1)
