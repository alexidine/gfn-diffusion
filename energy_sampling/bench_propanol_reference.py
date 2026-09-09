"""An INDEPENDENT reference distribution for the propanol benchmark.

Both benchmark arms are trained to sample `p(x) ~ exp(-energy(x))` over the 30-dimensional
`full` chart. Comparing them only to EACH OTHER would establish that they agree, not that
either is right -- two samplers sharing a prior, a chart and a force field can agree at the
wrong distribution. So this draws from the same target by a route with no learning in it at
all: Metropolis-adjusted Langevin, on the identical `ConformerTorsions.energy`.

WHY THIS IS A FAIR REFERENCE. The target is taken from the energy object the runs use, so the
Jacobian, the chart scaling, the reward clip and the box wall are all exactly the ones being
trained against. What differs is only HOW the samples are produced. A reference that used a
different measure would be measuring the measure.

WHY MALA AND NOT REJECTION OR PLAIN RANDOM WALK. 30 dimensions: an isotropic random walk needs
a step small enough that its autocorrelation time swamps any practical budget, and rejection
sampling has no usable envelope. MALA uses the gradient the force field already provides and
is exactly corrected by its Metropolis step, so it is unbiased for any step size -- the step
size costs efficiency, never correctness.

WHAT IS REPORTED, AND WHAT IS NOT. R-hat and per-chain ESS are printed for the energy and for
every coordinate. A reference whose own convergence is unestablished is not a reference, so if
R-hat is not close to 1 the right conclusion is that this needs a longer run, NOT that the
benchmark arms disagree with the truth.

    python bench_propanol_reference.py --out <dir>/reference.pt --chains 16 --steps 200000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from energies.conformer_torsions import ConformerTorsions

SMILES = 'CCCO'


def log_p(energy, x, grad=False):
    """``log p(x)`` up to a constant. `energy()` IS `-log_reward`, measure terms included.

    `keep_grads` is NOT optional for the Langevin drift: `energy()` detaches by default (it
    is normally called for a reward, not for a force), so without it the graph is severed and
    autograd raises rather than silently returning zeros.
    """
    return -energy.energy(x, keep_grads=grad)


def grad_log_p(energy, x):
    x = x.detach().requires_grad_(True)
    lp = log_p(energy, x, grad=True).sum()
    g, = torch.autograd.grad(lp, x)
    return lp.detach(), g.detach()


def wrap(x, periodic):
    """phi columns live on a circle; folding them keeps the walk from fighting the seam."""
    if periodic is None or not periodic.any():
        return x
    y = x.clone()
    y[:, periodic] = (y[:, periodic] + 1.0) % 2.0 - 1.0
    return y


def mala(energy, x0, n_steps, step0, periodic, adapt_frac=0.3, target_acc=0.574, seed=0):
    """Metropolis-adjusted Langevin. Returns ``(samples [S, C, d], acc, step)``.

    Step size is adapted only during the first `adapt_frac` of the run and then FROZEN --
    an adapting chain is not a Markov chain, so samples from the adaptation phase are
    discarded rather than merely down-weighted.
    """
    g = torch.Generator(device=x0.device).manual_seed(seed)
    x = x0.clone()
    step = torch.full((x.shape[0], 1), float(step0), dtype=x.dtype, device=x.device)
    n_adapt = int(n_steps * adapt_frac)
    lp, gr = grad_log_p(energy, x)
    kept, n_acc = [], torch.zeros(x.shape[0], device=x.device)

    for t in range(n_steps):
        noise = torch.randn(x.shape, generator=g, dtype=x.dtype, device=x.device)
        prop = wrap(x + 0.5 * step ** 2 * gr + step * noise, periodic)
        lp_p, gr_p = grad_log_p(energy, prop)

        # the MALA correction: q(x|x') / q(x'|x), both Gaussians centred on a gradient step
        back = x - prop - 0.5 * step ** 2 * gr_p
        fwd = prop - x - 0.5 * step ** 2 * gr
        log_q = (-(back ** 2).sum(-1) + (fwd ** 2).sum(-1)) / (2 * step[:, 0] ** 2)
        log_a = (lp_p - lp) + log_q
        u = torch.rand(x.shape[0], generator=g, dtype=x.dtype, device=x.device)
        acc = (torch.log(u) < log_a) & torch.isfinite(lp_p)

        x = torch.where(acc[:, None], prop, x)
        lp = torch.where(acc, lp_p, lp)
        gr = torch.where(acc[:, None], gr_p, gr)
        n_acc += acc.to(n_acc.dtype)

        if t < n_adapt:
            # per-chain adaptation on LOG step, at a CONSTANT rate. A Robbins-Monro 1/(t+10)
            # decays far too fast here: measured, it left acceptance at 0.022 against a 0.574
            # target, and a chain that rejects 98% of proposals has not moved -- at which
            # point R-hat reads a healthy 0.998 because the chains have not yet found
            # anything to disagree about. That is the exact failure this file's docstring
            # warns of, so the diagnostic cannot be trusted to catch it.
            step = step * torch.exp(0.05 * (acc.to(x.dtype)[:, None] - target_acc))
        elif (t - n_adapt) % 10 == 0:
            kept.append(x.clone())

    return torch.stack(kept), (n_acc / n_steps), step[:, 0]


def rhat(chains):
    """Split R-hat over ``[S, C]``. > 1.01 means the chains have not mixed."""
    s, c = chains.shape
    half = s // 2
    parts = torch.stack([chains[:half], chains[half:2 * half]]).reshape(2 * c, half)
    m, v = parts.mean(1), parts.var(1, unbiased=True)
    b = half * m.var(unbiased=True)
    w = v.mean()
    if float(w) <= 0:
        return float('nan')
    return float(torch.sqrt(((half - 1) / half * w + b / half) / w))


def ess(chains):
    """Crude per-chain ESS from the lag-1 autocorrelation -- an upper bound, reported as one."""
    s, c = chains.shape
    x = chains - chains.mean(0, keepdim=True)
    denom = (x * x).sum(0)
    r1 = (x[:-1] * x[1:]).sum(0) / denom.clamp_min(1e-30)
    return float((s * (1 - r1) / (1 + r1).clamp_min(1e-6)).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--chains', type=int, default=16)
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--step0', type=float, default=0.02)
    ap.add_argument('--internal-prior', type=Path, default=Path('conformer_prior_v2.pt'),
                    help='used ONLY to disperse the chain starts, never as the target')
    ap.add_argument('--level', default='full')
    ap.add_argument('--force-field', default='mmff')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    energy = ConformerTorsions(smiles=SMILES, device=args.device, level=args.level,
                               force_field=args.force_field, seed=args.seed)
    print(energy.describe().splitlines()[0])
    d = int(energy.data_ndim)
    periodic = torch.as_tensor(energy.periodic_dims, dtype=torch.bool, device=args.device)
    print(f'   d = {d}, {int(periodic.sum())} periodic column(s), '
          f'{args.chains} chains x {args.steps} steps')

    # DISPERSED BUT PHYSICAL starts. Chains that all begin in one basin agree early and
    # R-hat reads healthy while the sampler has seen one mode -- so the starts must be
    # spread. But uniform-on-the-box in 30 dimensions is not spread, it is absurd: measured,
    # its p90 energy is +542 against a target median near +19, so every chain spends its
    # budget falling out of a region the target gives no mass. The fitted prior samples
    # rotamers, so it is dispersed ACROSS BASINS while staying physical.
    rng = np.random.default_rng(args.seed)
    if args.internal_prior is not None and Path(args.internal_prior).exists():
        from build_prior_states import fit_or_load
        fitted = fit_or_load(Path(args.internal_prior), [], 0.15)
        x0, _ = energy.sample_prior_states(fitted, args.chains, rng, report=False)
        x0 = x0.to(torch.float64).to(args.device)
        print(f'   starts: fitted InternalPrior ({args.chains} draws)')
    else:
        x0 = torch.as_tensor(rng.uniform(-1, 1, (args.chains, d)),
                             dtype=torch.float64, device=args.device)
        print('   starts: UNIFORM on the box (no fitted prior found -- expect a long burn-in)')

    t0 = time.time()
    samples, acc, step = mala(energy, x0, args.steps, args.step0, periodic, seed=args.seed)
    dt = time.time() - t0
    print(f'   {dt:.1f}s  acceptance {float(acc.mean()):.3f} '
          f'[{float(acc.min()):.3f}, {float(acc.max()):.3f}]  '
          f'final step {float(step.mean()):.4f}')
    print(f'   kept {samples.shape[0]} x {samples.shape[1]} = '
          f'{samples.shape[0] * samples.shape[1]} samples (thinned 10x, adaptation discarded)')

    with torch.no_grad():
        flat = samples.reshape(-1, d)
        e = torch.cat([energy.energy(flat[i:i + 4096]) for i in range(0, flat.shape[0], 4096)])
        e_chain = e.reshape(samples.shape[0], samples.shape[1])

    r_e = rhat(e_chain)
    r_x = max(rhat(samples[:, :, k]) for k in range(d))
    print(f'\n   R-hat  energy {r_e:.4f}   worst coordinate {r_x:.4f}   '
          f'(> 1.01 means NOT converged)')
    print(f'   ESS    energy {ess(e_chain):.0f} of {e.numel()} draws')
    print(f'   energy  median {e.median():+8.3f}  p10 {torch.quantile(e, 0.1):+8.3f}'
          f'  p90 {torch.quantile(e, 0.9):+8.3f}')
    if max(r_e, r_x) > 1.01:
        print('\n   *** NOT CONVERGED. This is not yet a reference; run it longer. ***')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'samples': samples.cpu(), 'energy': e_chain.cpu(), 'smiles': SMILES,
                'level': args.level, 'force_field': args.force_field,
                'rhat_energy': r_e, 'rhat_worst_coord': r_x,
                'acceptance': acc.cpu(), 'step': step.cpu(),
                'chains': args.chains, 'steps': args.steps, 'seed': args.seed}, args.out)
    print(f'\n   wrote -> {args.out}')


if __name__ == '__main__':
    main()
