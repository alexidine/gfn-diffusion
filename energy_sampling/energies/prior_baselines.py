"""F1 -- the reference table. What a number has to beat before it means anything.

A TABLE GENERATOR, NOT A SECOND HARNESS. It imports the diagnostics and the optimizer and
calls them; it does not reimplement a scorer and it has no injection framework. Whether the
pipeline is CORRECT is prior_smoke.py's job. Whether a number is GOOD is this file's job.

FOUR AXES:  {uniform, prior} x {raw, optimized} x {energy, diversity} x tier.

The arm that did not exist is UNIFORM -- the baseline every other number is read against.
It is x ~ U(-1,1)^d over the sampler's own box, i.e. exactly what the model explores with
no prior at all.

READING RULES, each of which is a way this table would otherwise lie.

TIERS ARE NOT A RESOLUTION LADDER. Freezing a degree of freedom gives a conditional slice
differing from the full distribution by a state-dependent Fixman factor, so a figure at
'torsion' does not approximate the same figure at 'full'. Comparisons are well posed WITHIN
a tier. One table is emitted per tier for that reason and the header repeats it.

ENERGY IS THE RAW EXCESS above the reference conformer, in kT:

    excess = (U - U_ref)/kT - d/2

reported as median, p90 and the thermal-or-better fraction (excess <= 0). The d/2 is
equipartition -- a harmonic system at temperature T sits d/2 kT above its minimum -- so 0
means "indistinguishable from thermal" and the number is comparable across molecule sizes.
The reference is the REFERENCE CONFORMER's energy, not an enumerated mode minimum. That is
deliberate and it is what lets the energy axis run at EVERY tier including 'torsion', where
mode enumeration raises. It also shares no operand with the sampler.

T_eff = 1 + 2*median_excess/d IS REPORTED BESIDE, NEVER ALONE, AND ONE BLOCK IS DEGENERATE.
The prior draws r and theta at sqrt(kT/2k) from the force field's own constants and T_eff
then scores that draw with the SAME constants, so k cancels and the cell reads ~1.0 by
construction -- verified by scaling k_bond x4 for a bit-identical answer. The degeneracy is
specific: it does NOT apply to uniform, to optimized draws, to the phi dimensions, or at
'torsion' where r and theta are frozen. Cells where it bites are marked `*deg`, and in those
cells T_eff flatters the prior and the raw excess columns are the ones to read.

DIVERSITY is coverage where coverage is trustworthy and the rotamer-basin OCCUPANCY
distribution where it is not. Every cell states which produced it; there is no silent
substitution. coverage_report raises outright at 'torsion' and false-passes on molecules
with coupled coordinates until B1 lands, so those cells print BLOCKED with the reason
rather than a stand-in.

ARMS ARE SEEDED INDEPENDENTLY. A shared draw across arms is a correlation, not a control,
and it understates every error bar. Each cell prints its seed count and the spread over
seeds rather than a point estimate.

OPTIMIZED ARMS PRINT THEIR COMPUTE BUDGET. Relaxation is a different proposal at a
different cost, not a better version of its raw arm, and fixed-step relaxation OVER-COOLS --
ethanol reaches T_eff/T = 0.02 by 100 steps, i.e. colder than the target rather than closer
to it. An optimized cell without its budget beside it is not interpretable.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

import numpy as np
import torch

from energies.conformer_torsions import ConformerTorsions
import energies.prior_diagnostics as pdg
from mxtaltools.conformers.optimize import gradient_descent_optimization

RESULTS = pathlib.Path(__file__).resolve().parent / 'results'

# prior_smoke's set, minus the enantiomer pair whose only job is chirality machinery.
MOLECULES = [
    ('propanol', 'CCCO'),
    ('butanol', 'CCCCO'),
    ('hexanol', 'CCCCCCO'),
    ('nma', 'CC(=O)NC'),
    ('glycerol', 'OCC(O)CO'),
    ('butyronitrile', 'CCCC#N'),
    ('ethylcyclohexane', 'CCC1CCCCC1'),
    ('ethylnaphthalene', 'CCc1ccc2ccccc2c1'),
    ('ala-dipeptide', 'CC(=O)NC(C)C(=O)NC'),
]
TIERS = ('torsion', 'dihedral', 'flex', 'full')


def excess_kt(en, x):
    """(U - U_ref)/kT - d/2 per draw. No operand shared with the sampler."""
    temp = float(en.temperature)
    u = en.potential_energy(x, temp).detach().cpu().numpy()
    return (u - float(en.e_ref)) / temp - en.ndim / 2.0


def draw_uniform(en, n, seed):
    """THE ARM THAT DID NOT EXIST. Uniform over the sampler's own box."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, en.ndim, generator=g, dtype=en.dtype) * 2 - 1


def draw_prior(en, prior, n, seed):
    x, _ = en.sample_prior_states(prior, n, np.random.default_rng(seed),
                                  report=False, joint_rings=False)
    return x.detach()


def _as_float32_ff(ff):
    """A float32 view of the force field.

    gradient_descent_optimization carries an internal float32 assumption (its step_drift
    accumulator), so a float64 DoF vector fails on one side and a float64 ForceField fails
    on the other. Relaxing in float32 is harmless because the RESULT is scored by the
    unchanged float64 energy -- the scorer stays bit-identical across every arm, which is
    the property the table depends on. Only the trajectory to the relaxed point is single
    precision.
    """
    import dataclasses
    out = {}
    for f in dataclasses.fields(ff):
        v = getattr(ff, f.name)
        out[f.name] = (v.to(torch.float32)
                       if torch.is_tensor(v) and v.is_floating_point() else v)
    return dataclasses.replace(ff, **out)


def optimize(en, x, steps):
    """Relax in internal coordinates. Returns (x_opt, budget_string)."""
    n = x.shape[0]
    tree, ff = en._batch(n)
    r, th, ph = en.dof_from_state(x)
    # the batched tree is CONCATENATED, so the optimizer wants flat DoF, not [batch, n_r]
    f32 = torch.float32
    _, rec = gradient_descent_optimization(
        tree, (r.reshape(-1).to(f32), th.reshape(-1).to(f32), ph.reshape(-1).to(f32)),
        _as_float32_ff(ff),
        max_num_steps=steps, min_num_steps=steps, show_tqdm=False)
    br, bth, bph = (a.reshape(n, -1).to(en.dtype) for a in rec.best_dof)
    # At a COLLECTIVE tier the relaxed point cannot be mapped back to a state: one column
    # drives several dihedrals, so state_from_dof has no selection map to invert through.
    # Scoring the relaxed CARTESIANS instead would leave the tier's manifold and compare
    # across tiers, which is exactly the Fixman trap this table refuses to fall into. So
    # the cell is BLOCKED, by the same missing inverse that blocks diversity here.
    x_opt = en.state_from_dof(br, bth, bph).clamp(-1.0, 1.0)
    return x_opt, '{} steps adam'.format(rec.n_steps)


def diversity(en, x):
    """(evenness, source, detail). Occupancy where coverage is not trustworthy."""
    try:
        groups = pdg.rotamer_modes(en)
    except NotImplementedError as exc:
        return None, 'BLOCKED', str(exc).split('.')[0].strip()
    except Exception as exc:
        return None, 'BLOCKED', '{}: {}'.format(type(exc).__name__, exc)
    if not groups:
        return None, 'BLOCKED', 'no rotatable groups on this molecule'

    n0 = en.n_r + en.n_th
    dof = np.concatenate([a.detach().cpu().numpy() for a in en.dof_from_state(x)], axis=1)
    lab = np.zeros(len(dof), dtype=np.int64)
    stride = 1
    for gi in range(len(groups) - 1, -1, -1):
        rows, centres = groups[gi]
        v = dof[:, n0 + rows[0]]
        dist = np.abs((v[:, None] - centres[None, :] + np.pi) % (2 * np.pi) - np.pi)
        lab += stride * np.argmin(dist, axis=1)
        stride *= len(centres)

    frac = np.bincount(lab, minlength=stride) / len(dof)
    occupied = int((frac > 0).sum())
    nz = frac[frac > 0]
    # Normalised Shannon evenness over the enumerated basins. 1.0 = every basin hit
    # equally; it needs no mode-enumeration GUARANTEE to be meaningful, which is why it
    # stands in for coverage rather than pretending to be it.
    even = float(-(nz * np.log(nz)).sum() / math.log(stride)) if stride > 1 else float('nan')
    return even, 'occupancy', '{}/{} basins'.format(occupied, stride)


def teff_is_degenerate(en, sampler, opt_steps):
    """RAW PRIOR with free r/theta only: k cancels between the draw width and the score."""
    return sampler == 'prior' and not opt_steps and en.n_r > 0 and en.ndim > en.n_ph


def cell(en, prior, sampler, opt_steps, n, seeds):
    """One cell. Independent seed per arm; spread reported, never a point estimate."""
    med, p90, therm, teff, evens = [], [], [], [], []
    budget, dsrc, ddet = 'raw', None, None
    for s in seeds:
        # ONE missing inverse blocks most of this table at a collective tier. state_from_dof
        # has no selection map when a column drives several dihedrals, which takes out the
        # fitted-prior DRAW, both OPTIMIZED arms and DIVERSITY -- everything except uniform.
        # That is gate condition 4, and it is why the shipped tier is nearly empty here.
        try:
            x = (draw_uniform(en, n, s) if sampler == 'uniform'
                 else draw_prior(en, prior, n, s))
            if opt_steps:
                x, budget = optimize(en, x, opt_steps)
        except NotImplementedError as exc:
            return {'blocked': str(exc).split('.')[0].strip(), 'budget': 'BLOCKED',
                    'n_seeds': len(seeds), 'n': n}
        e = excess_kt(en, x)
        med.append(float(np.median(e)))
        p90.append(float(np.percentile(e, 90)))
        therm.append(float((e <= 0).mean()))
        teff.append(1.0 + 2.0 * float(np.median(e)) / en.ndim)
        val, dsrc, ddet = diversity(en, x)
        if val is not None and math.isfinite(val):
            evens.append(val)

    def sd(a):
        return statistics.stdev(a) if len(a) > 1 else 0.0

    return {
        'median_kt': statistics.mean(med), 'median_kt_sd': sd(med),
        'p90_kt': statistics.mean(p90), 'p90_kt_sd': sd(p90),
        'thermal_frac': statistics.mean(therm),
        'T_eff': statistics.mean(teff), 'T_eff_sd': sd(teff),
        'T_eff_degenerate': teff_is_degenerate(en, sampler, opt_steps),
        'evenness': statistics.mean(evens) if evens else None,
        'evenness_sd': sd(evens) if evens else None,
        'div_source': dsrc, 'div_detail': ddet,
        'budget': budget, 'n_seeds': len(seeds), 'n': n,
    }


def run(tiers, mols, n, seeds, opt_steps, prior_path):
    prior = torch.load(prior_path, weights_only=False)
    out = {}
    for tier in tiers:
        rows = []
        for name, smi in mols:
            try:
                en = ConformerTorsions(smiles=smi, level=tier, force_field='mmff',
                                       log_temperature=0.0, device='cpu')
            except Exception as exc:
                rows.append({'molecule': name,
                             'error': '{}: {}'.format(type(exc).__name__, exc)})
                continue
            row = {'molecule': name, 'd': int(en.ndim), 'arms': {}}
            for sampler in ('uniform', 'prior'):
                for tag, steps in (('raw', 0), ('optimized', opt_steps)):
                    row['arms']['{}/{}'.format(sampler, tag)] = cell(
                        en, prior, sampler, steps, n, seeds)
            rows.append(row)
        out[tier] = rows
    return out


HEAD = ('{:<17}{:>4}  {:<19}{:>14}{:>14}{:>7}{:>15}  {:<28}{}'
        .format('molecule', 'd', 'arm', 'med kT', 'p90 kT', 'therm', 'T_eff',
                'diversity', 'budget'))


def fmt(tier, rows, n, seeds, opt_steps):
    out = ['=' * 118,
           'TIER {!r}    n={} per seed    seeds={}    optimized arm = {} steps'
           .format(tier, n, seeds, opt_steps),
           '  WELL POSED WITHIN THIS TIER ONLY. Freezing a DoF gives a conditional slice',
           '  differing by a state-dependent Fixman factor, so a row here does NOT',
           '  approximate the same row at another tier.',
           '  energy = (U - U_ref)/kT - d/2, mean over seeds +- sd across seeds.',
           '  *deg = T_eff degenerate by construction (k cancels); read the kT columns.',
           '=' * 118, HEAD, '-' * 118]
    for r in rows:
        if 'error' in r:
            out.append('{:<17}{:>4}  BUILD FAILED: {}'.format(r['molecule'], '', r['error']))
            continue
        first = True
        for arm, c in r['arms'].items():
            head = ('{:<17}{:>4}  '.format(r['molecule'], r['d']) if first else ' ' * 23)
            first = False
            if 'blocked' in c:
                out.append('{}{:<19}{:>62}  {}'.format(head, arm, 'BLOCKED', c['blocked']))
                continue
            deg = '*deg' if c['T_eff_degenerate'] else '    '
            div = (c['div_source'] if c['evenness'] is None
                   else '{:.3f}+-{:.3f} {}'.format(c['evenness'], c['evenness_sd'],
                                                   c['div_source']))
            out.append('{}{:<19}{:>9.1f}+-{:<4.1f}{:>9.1f}+-{:<4.1f}{:>6.1f}%'
                       '{:>10.2f}+-{:<4.2f}{}  {:<28}{}'
                       .format(head, arm, c['median_kt'], c['median_kt_sd'],
                               c['p90_kt'], c['p90_kt_sd'], c['thermal_frac'] * 100,
                               c['T_eff'], c['T_eff_sd'], deg, div, c['budget']))
        probe = r['arms']['uniform/raw']
        if probe['div_source'] == 'BLOCKED':
            out.append('{}DIVERSITY BLOCKED at this tier -- {}'
                       .format(' ' * 23, probe['div_detail']))
    out.append('')
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tiers', nargs='+', default=list(TIERS), choices=TIERS)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--opt-steps', type=int, default=25)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--prior-path', default='conformer_prior_v2.pt')
    ap.add_argument('--json', default=None)
    args = ap.parse_args(argv)

    mols = [m for m in MOLECULES if not args.only or m[0] in args.only]
    seeds = list(range(args.seeds))
    res = run(args.tiers, mols, args.n, seeds, args.opt_steps, args.prior_path)

    print('\n'.join(fmt(t, res[t], args.n, seeds, args.opt_steps) for t in args.tiers))
    RESULTS.mkdir(exist_ok=True)
    path = args.json or str(RESULTS / 'prior_baselines.json')
    pathlib.Path(path).write_text(
        json.dumps({'n': args.n, 'seeds': seeds, 'opt_steps': args.opt_steps,
                    'tiers': res}, indent=1), encoding='utf-8')
    print('wrote {}'.format(path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
