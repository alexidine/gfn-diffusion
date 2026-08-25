"""Build a SEMI-OPTIMIZED conformer prior dataset: minima plus shoulders, no thermal smear.

WHY THIS EXISTS. `init_prior_dataset` samples the fitted InternalPrior and relaxes it to
T_eff/T = 2.0 -- deliberately thermal, and correct as a Boltzmann-shaped proposal. But that
set is also what seeds the ANCHOR buffer and what 45% of the fused mixture trains on, and
measured on tetraglycine it contained no good conformer at all: min 45.9 against a tier
minimum of 28.3, median 84.4. The model was never shown the answer.

TB is off-policy -- its fixed point does not depend on where backward terminals come from --
so there is no correctness reason for the terminal set to be thermal. Terminals control
where the learning signal lands. Putting them on good states is the point.

SEMI-optimized, not converged. A fully-relaxed set has very little entropy, and with
`freeze_policy: 1.0` on the forward branch the policy is trained only by bwd and replay, so
the thermal shape has to be learned rather than handed over. Stopping the descent short
keeps residual spread for free: measured on tetraglycine, median energy by depth runs
253.1 (0) -> 69.3 (10) -> 51.9 (30) -> 47.1 (60) -> 44.5 (120) -> 43.4 (250). A MIXTURE of
depths therefore spans minima through shoulders in one set, with no reheating machinery.

DIVERSITY COMES FROM THE DRAW, NEVER FROM THE DESCENT. Local optimization refines the basin
it starts in; it cannot discover one the sampler missed. So the yield is deliberately built
by drawing many and filtering, not by optimizing few very hard, and the per-torsion circular
spread is reported so a collapse is visible rather than assumed.
"""
import argparse
import json
import math
import time
import warnings

warnings.filterwarnings('ignore')
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
import numpy as np
import torch

torch.set_default_dtype(torch.float32)

from energies.conformer_torsions import ConformerTorsions
from energies.prior_baselines import descend

ap = argparse.ArgumentParser()
ap.add_argument('--smiles', default='NCC(=O)NCC(=O)NCC(=O)NCC(=O)O')
ap.add_argument('--level', default='full')
ap.add_argument('--force-field', default='mmff')
ap.add_argument('--energy-clip', type=float, default=250.0)
ap.add_argument('--n', type=int, default=20000, help='target number of KEPT states')
ap.add_argument('--ceiling', type=float, default=10.0,
                help='keep states within this many kcal/mol of the global minimum found')
ap.add_argument('--depths', default='30,60,120,250,400',
                help='descent-step ladder; shallow = shoulders, deep = minima')
ap.add_argument('--chunk', type=int, default=4096)
ap.add_argument('--max-draws', type=int, default=1_500_000)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default=None)
cli = ap.parse_args()

DEPTHS = [int(d) for d in cli.depths.split(',')]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
en = ConformerTorsions(smiles=cli.smiles, level=cli.level, device=dev,
                       force_field=cli.force_field, energy_clip=cli.energy_clip)
prior = torch.load('conformer_prior_v2.pt', weights_only=False)
one = torch.tensor(1.0, device=dev)
rng = np.random.default_rng(cli.seed)
print(f'{cli.smiles}  level {cli.level}  d={en.data_ndim}  clip={cli.energy_clip}  device {dev}')
print(f'depth ladder {DEPTHS}   target {cli.n:,} kept within {cli.ceiling} kcal/mol of the '
      f'global minimum\n')

kept_x, kept_e, kept_d = [], [], []
best = math.inf
drawn = 0
t0 = time.perf_counter()
print(f'{"drawn":>10s}{"kept":>9s}{"best E":>9s}{"ceiling":>9s}{"yield%":>8s}{"sec":>7s}')
while sum(len(k) for k in kept_x) < cli.n and drawn < cli.max_draws:
    for depth in DEPTHS:
        xs, _ = en.sample_prior_states(prior, cli.chunk, rng, report=False)
        x = torch.as_tensor(xs, dtype=en.dtype, device=dev)
        drawn += cli.chunk
        with torch.enable_grad():
            x = descend(en, x, depth)[0].detach()
        e = en.potential_energy(x, one).double()
        good = torch.isfinite(e)
        x, e = x[good], e[good]
        if e.numel():
            best = min(best, float(e.min()))
        # re-filter everything kept so far whenever the floor drops: the ceiling is
        # RELATIVE, so a new global minimum invalidates earlier admissions
        keep = e <= best + cli.ceiling
        if keep.any():
            kept_x.append(x[keep].cpu())
            kept_e.append(e[keep].cpu())
            kept_d.append(torch.full((int(keep.sum()),), depth, dtype=torch.int32))
    if kept_e:
        all_e = torch.cat(kept_e)
        m = all_e <= best + cli.ceiling
        if not bool(m.all()):
            X, E, D = torch.cat(kept_x)[m], all_e[m], torch.cat(kept_d)[m]
            kept_x, kept_e, kept_d = [X], [E], [D]
    n_kept = sum(len(k) for k in kept_x)
    print(f'{drawn:>10,}{n_kept:>9,}{best:>9.2f}{best + cli.ceiling:>9.2f}'
          f'{100 * n_kept / max(drawn, 1):>8.2f}{time.perf_counter() - t0:>7.0f}', flush=True)

X = torch.cat(kept_x)[:cli.n]
E = torch.cat(kept_e)[:cli.n]
D = torch.cat(kept_d)[:cli.n]
print(f'\nKEPT {len(X):,} of {drawn:,} drawn ({100 * len(X) / drawn:.2f}%) in '
      f'{time.perf_counter() - t0:.0f} s')

e = E.numpy()
print(f'\nenergy (kcal/mol, potential only -- no measure terms):')
print(f'  min {e.min():7.2f}   p10 {np.percentile(e, 10):7.2f}   median {np.median(e):7.2f}'
      f'   p90 {np.percentile(e, 90):7.2f}   max {e.max():7.2f}')
print(f'  global minimum found {best:.2f};  T_eff/T = '
      f'{1 + 2 * (np.median(e) - best) / en.ndim:.2f}  (2.0 = thermal, lower = colder)')
print(f'\nby descent depth:')
print(f'  {"depth":>7s}{"kept":>9s}{"share":>8s}{"median E":>10s}{"min E":>9s}')
for depth in DEPTHS:
    m = (D == depth).numpy()
    if m.sum():
        print(f'  {depth:>7d}{m.sum():>9,}{100 * m.mean():>7.1f}%'
              f'{np.median(e[m]):>10.2f}{e[m].min():>9.2f}')

# DIVERSITY, reported not assumed. Descent cannot create modes, so if the draw+filter
# collapsed onto one basin this is where it shows: a torsion whose circular sd falls to a
# few degrees is a coordinate the set no longer explores.
_, _, ph = en.dof_from_state(X.to(dev))
ph = ph.double().cpu().numpy()


def circ_sd_deg(a):
    r = math.hypot(float(np.sin(a).mean()), float(np.cos(a).mean()))
    return math.degrees(math.sqrt(-2.0 * math.log(max(r, 1e-12))))


sds = np.array([circ_sd_deg(ph[:, k]) for k in range(ph.shape[1])])
live = sds[sds > 2.0]
print(f'\nper-torsion circular spread over the kept set ({len(live)} of {len(sds)} live):')
print(f'  median {np.median(live):.1f} deg   p10 {np.percentile(live, 10):.1f}   '
      f'max {live.max():.1f}   (a collapsed set reads a few degrees everywhere)')

out = cli.out or f'conformer_prior_dataset_{cli.n // 1000}k.pt'
torch.save({
    'states': X, 'energies': E, 'depth': D,
    'smiles': cli.smiles, 'level': cli.level, 'force_field': cli.force_field,
    'energy_clip': cli.energy_clip, 'data_ndim': int(en.data_ndim),
    'global_min': float(best), 'ceiling': float(cli.ceiling), 'depths': DEPTHS,
    'n_drawn': int(drawn), 'seed': int(cli.seed),
    # the clip is part of the problem definition, so a set built under one must never be
    # loaded under another -- the loader is expected to check these
    'provenance': 'build_conformer_prior_dataset.py',
}, out)
print(f'\nwrote {out}')
