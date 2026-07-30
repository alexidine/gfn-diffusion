"""
Mode-covering buffer for the conformational GFlowNet, built on CPU.

Random torsion states -> local optimization -> deduplicate into distinct basins. The
result is a set of *modes*, not a Boltzmann sample, which is exactly what TB needs:
trajectory balance is off-policy consistent, so the buffer's weights are irrelevant and
only its **support** matters. Deliberately over-representing rare basins is a feature --
it equalises constraint density per mode, where a Boltzmann-weighted buffer would leave
the rare ones effectively unconstrained no matter how long you train.

Each basin also gets a harmonic free-energy estimate,

    F_m  =  E_m + (1/2beta) * log det(beta * H_m / 2pi)

from the Hessian of the energy in torsion space (d x d autograd, free at this size).
Those give predicted relative populations to check a trained sampler's mode occupancy
against -- ground truth that survives past the point where brute-force quadrature dies.

    python build_conformer_buffer.py --smiles CCCCO --n-starts 20000 --threads 2
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from energies.conformer_torsions import ConformerTorsions


def wrap(x):
    """Wrap to the state space (-1, 1], NOT to radians.

    The torsion state is a delta from the reference conformer in units where 1 == pi
    (see conformer_torsions.build_positions). Wrapping with the radian period 2*pi
    instead of 2 leaves states anywhere in +/-pi state units -- i.e. +/-565 degrees --
    which are outside the sampler's domain entirely.
    """
    return (x + 1.0) % 2.0 - 1.0


def local_optimize(energy, x0, steps=250, lr=0.05, patience=40):
    """Batched gradient descent on the torsion state. Adam on a torus.

    Returns the optimized states and their energies. Runs the whole population at once;
    per-sample early stopping isn't worth the bookkeeping when a step is this cheap.
    """
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    best = torch.full((len(x0),), float("inf"), dtype=x0.dtype)
    best_x = x0.clone()
    stale = 0

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        e = energy.energy(x, None, None, keep_grads=True)
        e.sum().backward()
        opt.step()
        with torch.no_grad():
            x.data = wrap(x.data)
            e_now = energy.energy(x.data, None, None)
            improved = e_now < best
            best = torch.where(improved, e_now, best)
            best_x[improved] = x.data[improved]
            stale = 0 if improved.any() else stale + 1
            if stale > patience:
                break
    return best_x.detach(), best


def deduplicate(x, e, tol_deg=20.0):
    """Collapse states that land in the same basin.

    Two torsion vectors are the same mode when every component agrees to within
    ``tol_deg`` under periodic wrapping. Sorted by energy so each basin keeps its best
    representative.
    """
    order = torch.argsort(e)
    x, e = x[order], e[order]
    tol = tol_deg / 180.0            # state units, where 1 == 180 degrees

    keep_x, keep_e, counts = [], [], []
    for xi, ei in zip(x, e):
        hit = None
        for j, kx in enumerate(keep_x):
            if wrap(xi - kx).abs().max() < tol:
                hit = j
                break
        if hit is None:
            keep_x.append(xi)
            keep_e.append(ei)
            counts.append(1)
        else:
            counts[hit] += 1
    return torch.stack(keep_x), torch.stack(keep_e), torch.tensor(counts)


def harmonic_weights(energy, modes, beta=1.0, eps=1e-8):
    """Relative populations from the harmonic approximation at each basin.

    p_m proportional to exp(-beta E_m) / sqrt(det H_m). Non-positive-definite Hessians
    (a saddle rather than a minimum) are reported as nan rather than silently given a
    weight, since that means the optimizer did not actually converge there.
    """
    log_w, cond = [], []
    for m in modes:
        h = torch.autograd.functional.hessian(
            lambda v: energy.energy(v.unsqueeze(0), None, None, keep_grads=True).squeeze(),
            m.clone().requires_grad_(True))
        h = 0.5 * (h + h.T)
        evals = torch.linalg.eigvalsh(h)
        if (evals <= eps).any():
            log_w.append(float("nan"))
            cond.append(float("nan"))
            continue
        e_m = energy.energy(m.unsqueeze(0), None, None).item()
        log_w.append(-beta * e_m - 0.5 * torch.log(evals).sum().item())
        cond.append((evals.max() / evals.min()).item())
    log_w = torch.tensor(log_w)
    finite = torch.isfinite(log_w)
    p = torch.full_like(log_w, float("nan"))
    p[finite] = torch.softmax(log_w[finite], 0)
    return p, log_w, torch.tensor(cond)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles", default="CCCCO")
    ap.add_argument("--n-starts", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--opt-steps", type=int, default=250)
    ap.add_argument("--dedup-deg", type=float, default=20.0)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)     # stay off the GPU and out of the way
    torch.manual_seed(args.seed)

    energy = ConformerTorsions(smiles=args.smiles, device="cpu", epsilon=args.epsilon)
    print(energy.describe())
    k = energy.data_ndim
    out = args.out or Path(f"conformer_buffer_{args.smiles.replace('/', '_')}.pt")

    t0 = time.time()
    all_x, all_e = [], []
    done = 0
    while done < args.n_starts:
        n = min(args.chunk, args.n_starts - done)
        x0 = torch.rand(n, k) * 2 - 1        # uniform on the state space, not on radians
        x, e = local_optimize(energy, x0, steps=args.opt_steps)
        all_x.append(x)
        all_e.append(e)
        done += n
        print(f"  optimized {done:>7}/{args.n_starts}   "
              f"E [{e.min():.3f}, {e.max():.3f}]   [{time.time() - t0:.0f} s]")

    x = torch.cat(all_x)
    e = torch.cat(all_e)
    modes, mode_e, counts = deduplicate(x, e, tol_deg=args.dedup_deg)
    print(f"\n{len(x)} optimized states -> {len(modes)} distinct basins "
          f"(dedup at {args.dedup_deg:.0f} deg)")

    p_harm, log_w, cond = harmonic_weights(energy, modes)
    print(f"\n{'mode':>5} {'E':>9} {'basin hits':>11} {'p_harmonic':>11} {'cond(H)':>10}")
    for i in range(min(len(modes), 12)):
        ph = f"{p_harm[i]:.4f}" if torch.isfinite(p_harm[i]) else "  saddle"
        cd = f"{cond[i]:.1f}" if torch.isfinite(cond[i]) else "     -"
        print(f"{i:>5} {mode_e[i]:>9.4f} {counts[i]:>11} {ph:>11} {cd:>10}")
    if len(modes) > 12:
        print(f"  ... {len(modes) - 12} more")

    # empirical basin frequency is a *search* statistic, not a population -- kept only
    # to see how uniformly the random starts covered the basins
    torch.save(dict(smiles=args.smiles, k=k, states=x, energies=e,
                    modes=modes, mode_energies=mode_e, basin_hits=counts,
                    harmonic_p=p_harm, harmonic_log_w=log_w, hessian_cond=cond,
                    n_starts=args.n_starts, dedup_deg=args.dedup_deg,
                    epsilon=args.epsilon), out)
    print(f"\nwrote {out}  ({len(x)} states, {len(modes)} modes, "
          f"{time.time() - t0:.0f} s wall)")


if __name__ == "__main__":
    main()
