"""F1 -- the reference table. What a number has to beat before it means anything.

A TABLE GENERATOR, NOT A SECOND HARNESS. It imports the diagnostics and the optimizer and
calls them; it does not reimplement a scorer and it has no injection framework. Whether the
pipeline is CORRECT is prior_smoke.py's job. Whether a number is GOOD is this file's job.

FOUR AXES:  {uniform, prior-rings-on, prior-rings-off} x {raw, optimized}
            x {energy, diversity, rings} x tier.

The arm that did not exist is UNIFORM -- the baseline every other number is read against.
It is x ~ U(-1,1)^d over the sampler's own box, i.e. exactly what the model explores with
no prior at all.

THE PRIOR ARM WAS MEASURING THE DISABLED RING PATH. It called
``sample_prior_states(..., joint_rings=False)``, which draws every ring DoF from an
independent marginal and so violates closure by construction -- while the closure monitor
was gated on the ring-system count and therefore reported 0.000 A on exactly that
configuration. So the table's poor ring results described a path the sampler does not use.
The arm is not renamed in place; it is SPLIT, and both halves say which they are:

    prior-rings-on    joint_rings=True, the real path -- pucker subspace or bank where one
                      resolves, aromatic rings held planar by design, unsupported rings
                      held at a fraction of thermal width, ring-positioning DoF held.
    prior-rings-off   joint_rings=False. A NEGATIVE CONTROL whose job is to make the ring
                      columns falsifiable: if it does not measurably worsen closure on a
                      ring molecule, the ring measurements are not live and nothing else in
                      the ring block should be believed.

Measured at 'full' with conformer_prior_v2.pt, the control moves cyclohexane's closure from
0.086 A (2.2 bond-sigma) to 2.93 A (75 bond-sigma). On an ACYCLIC molecule the two arms are
the same distribution by construction, and the control is reported N/A rather than run
twice; at 'torsion' the draw goes through build_prior_states.draw_states, which has no
joint_rings switch and whose ring DoF are frozen at the reference, so the ring comparison is
INAPPLICABLE there and says so.

RING CLASSES ARE FOUR AND ARE NOT COLLAPSED -- banked (a fitted RingModes subspace or a
RingBank above the row threshold), held-aromatic (planar BY DESIGN; an aromatic ring is
rigid and ``ring_blocks`` refuses to bank it), held-unsupported (saturated, but no bank
resolved for its key -- a gap), and the orthogonal stale-prior flag (a prior predating the
ring-signature fix has keys that cannot resolve, so every ring reads unsupported for a
reason that has nothing to do with the molecule). This runs on conformer_prior_v2.pt by
default and REFUSES a stale prior unless --allow-stale-prior is passed.

RING DENSITY-DEPENDENT NUMBERS ARE UNAVAILABLE, NOT APPROXIMATED. A ring block is drawn
from a mixture over fitted rows or a subspace that is singular in the directions it does not
span, so there is no ring ESS, no D_avoidable and no importance-sampled log Z --
``prior_log_prob`` raises rather than returning a usable-looking number. Energy, closure,
planarity and pucker occupancy need no q(x) and do run. A labelled unavailable cell is the
correct output; substituting the acyclic density or an independent marginal is not.

READING RULES, each of which is a way this table would otherwise lie.

TIERS ARE NOT A RESOLUTION LADDER. Freezing a degree of freedom gives a conditional slice
differing from the full distribution by a state-dependent Fixman factor, so a figure at
'torsion' does not approximate the same figure at 'full'. Comparisons are well posed WITHIN
a tier. One table is emitted per tier for that reason and the header repeats it.

ENERGY IS THE EXCESS ABOVE THE TIER'S OWN MINIMUM, in kT:

    excess = (U - U_tier_min)/kT - d/2

reported as median, p90 and the thermal-or-better fraction (excess <= 0). The d/2 is
equipartition -- a harmonic system at temperature T sits d/2 kT above its minimum -- so 0
means "indistinguishable from thermal" and the number is comparable across molecule sizes.

THE ZERO IS THE TIER-RESTRICTED MINIMUM, NOT THE REFERENCE CONFORMER, and that correction
is the difference between a meaningful column and a misleading one. A tier is not a coarse
view of the full space; it is a DIFFERENT, lower-dimensional object, and the honest
baseline for a generative model trained on that object is the distribution over that
object. The reference conformer generally does NOT lie in the tier's accessible set:
measured at 'torsion', min(U) over the tier sits 2.9 kT (propanol), 4.0 (butanol) and
6.1 (ala-dipeptide) ABOVE e_ref, and no move available at that tier can close the gap. With
e_ref as the zero, thermal-or-better read exactly 0.000 on every arm of every molecule --
not because no draw was thermal, but because the zero was unreachable. Corrected, the best
ala-dipeptide draw moves from +5.40 kT to -0.70, i.e. below thermal.

The offset is REPORTED, not hidden: `zero_vs_e_ref` is the bridge between parameterisations
and is the quantity to look at when relating one tier to another. Within a tier the offset
is common to every arm, so arm-vs-arm comparisons were always sound; it is the absolute
numbers, and anything referenced to thermal, that needed this.

The tier minimum is a MULTI-START LOCAL SEARCH, so it is an upper bound on the true
minimum: the zero could be lower, which would shift a column uniformly without changing any
comparison inside it. Start count and the best/worst spread across starts are printed.

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
import energies.ring_metrics as rmet

RESULTS = pathlib.Path(__file__).resolve().parent / 'results'

# prior_smoke's set, minus the enantiomer pair whose only job is chirality machinery,
# plus the ring set below. Ring molecules are chosen to put each of the four ring classes
# on the table separately -- a set where every ring happened to be banked would report a
# generic pass and could not tell "held by design" from "no bank resolved".
MOLECULES = [
    ('propanol', 'CCCO'),
    ('butanol', 'CCCCO'),
    ('hexanol', 'CCCCCCO'),
    ('nma', 'CC(=O)NC'),
    ('glycerol', 'OCC(O)CO'),
    ('butyronitrile', 'CCCC#N'),
    ('ethylcyclohexane', 'CCC1CCCCC1'),
    ('methyltetrahydropyran', 'CC1CCCCO1'),
    ('ethylbenzene', 'CCc1ccccc1'),
    ('ethylnaphthalene', 'CCc1ccc2ccccc2c1'),
    ('proline', 'OC(=O)C1CCCN1'),
    ('phenyltetrahydropyran', 'C1CCC(CO1)c1ccccc1'),
    ('ala-dipeptide', 'CC(=O)NC(C)C(=O)NC'),
]

# The ring set the ring block is meant to be read on, and what each one is FOR. Named here
# so a test can require the four classes to be represented rather than trusting the list.
RING_MOLECULES = {
    'ethylcyclohexane': 'supported saturated ring -- banked pucker subspace',
    'methyltetrahydropyran': 'substituted heterocycle -- banked, with ring-positioning '
                             'DoF outside the block that must be held',
    'ethylbenzene': 'aromatic ring -- held planar BY DESIGN, never banked',
    'ethylnaphthalene': 'FUSED aromatic -- one ring system, two closure bonds',
    'proline': 'saturated but UNSUPPORTED -- no bank resolves for its key',
    'phenyltetrahydropyran': 'both classes in one molecule -- banked saturated ring and '
                             'a held aromatic ring, reported separately',
}
TIERS = ('torsion', 'dihedral', 'flex', 'full')

# The three prior arms. 'prior-rings-off' is the negative control; see the module docstring.
SAMPLERS = ('uniform', 'prior-rings-on', 'prior-rings-off')


def excess_kt(en, x, zero):
    """(U - zero)/kT - d/2 per draw. No operand shared with the sampler."""
    temp = float(en.temperature)
    u = en.potential_energy(x, temp).detach().cpu().numpy()
    return (u - zero) / temp - en.ndim / 2.0


def descend(en, x0, steps, lr=None, optimizer='rprop'):
    """Adam on the STATE x. Returns (best_x, best_u), the best point SEEN, not the last.

    x IS the tier's coordinate, so a step cannot leave the tier's manifold and there is
    nothing to project back. At a collective tier that is the whole point: one state column
    drives several dihedral rows, and those rows want to move in OPPOSITE directions -- so
    descending per-row pulls the group off the manifold and the projection back through the
    leader column then discards the work, on some draws landing worse than it started.

    Best-seen rather than last means a step that overshoots cannot score worse than the
    start, which is what makes this usable as both a floor search and an arm.

    RPROP BY DEFAULT, and it is not a marginal choice here. Rprop steps on the gradient's
    SIGN with a per-parameter step size that adapts multiplicatively, which suits a start
    that may sit in a 40,000 kT steric clash: the gradient MAGNITUDES there are wildly
    unrepresentative, and Adam's second-moment estimate spends its first steps recovering
    from them. Measured on the same draws, Rprop reaches at 10 steps what Adam reaches at
    25-50 (phenyl-THP: -33.2 vs -8.0 at 10 steps), and reaches the tier minimum at 100
    steps that Adam needs 400 for -- to the same value, four times faster.

    A STATIC DIAGONAL PRECONDITIONER DOES NOTHING ON TOP OF ADAM, which is worth recording
    because it looks like it should. Rescaling the gradient per coordinate by the force
    field's own thermal widths measured bit-identical to plain Adam: Adam already
    normalises per parameter by the gradient's running RMS, so it divides any constant
    diagonal rescaling straight back out. Preconditioning would have to change the geometry
    (non-diagonal), or ride on SGD, to be worth anything.
    """
    if lr is None:
        lr = 0.02 if optimizer == 'rprop' else 0.05
    x = x0.clone().detach().requires_grad_(True)
    opt = (torch.optim.Rprop([x], lr=lr) if optimizer == 'rprop'
           else torch.optim.Adam([x], lr=lr))
    best_u = torch.full((len(x0),), float('inf'), dtype=en.dtype)
    best_x = x0.clone().detach()
    for _ in range(steps):
        opt.zero_grad()
        u = en.potential_energy(x, float(en.temperature), keep_grads=True)
        with torch.no_grad():
            ud = u.detach()
            hit = ud < best_u
            best_u = torch.where(hit, ud, best_u)
            best_x[hit] = x.detach()[hit]
        u.sum().backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(-1.0, 1.0)
    return best_x, best_u


def tier_minimum(en, starts, steps=150, lr=None, optimizer='rprop'):
    """min U over the TIER'S OWN coordinate. The honest zero for this tier.

    Multi-start, so the result is an UPPER bound on the true minimum. Returns
    (best, worst_start, n_starts).

    150 RPROP STEPS, not 400 Adam. Measured on four molecules across the set, Rprop at 100
    steps reaches the same minimum Adam reaches at 400 -- identical to two decimals on
    propanol, ethylcyclohexane and phenyl-THP, and within 0.05 kcal/mol on ala-dipeptide,
    which is the slowest and is why the default sits at 150 rather than 100. This is a
    third of the benchmark's total runtime and it produces the ZERO every other cell is
    measured against, so it is the one descent where a worse answer would shift a whole
    column rather than one arm.
    """
    b = descend(en, starts, steps, lr, optimizer)[1].cpu().numpy()
    return float(b.min()), float(b.max()), int(len(starts))


def draw_uniform(en, n, seed):
    """THE ARM THAT DID NOT EXIST. Uniform over the sampler's own box."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, en.ndim, generator=g, dtype=en.dtype) * 2 - 1


def draw_prior(en, prior, n, seed, joint_rings=True):
    """``(x, stats)``. ``stats`` is None on the collective route, which has no ring switch."""
    rng = np.random.default_rng(seed)
    if en.collective:
        # THE TORSION ROUTE. sample_prior_states ends in state_from_dof, which has no
        # row-wise inverse at a collective tier; draw_states writes the state column
        # directly and is the sampler that refusal message names. It takes NO joint_rings
        # argument -- ring DoF are frozen at the reference there -- which is why the ring
        # comparison is inapplicable at 'torsion' rather than merely unmeasured.
        if not joint_rings:
            raise NotImplementedError(
                'the joint-ring negative control is inapplicable at a collective tier: the '
                'draw goes through build_prior_states.draw_states, which has no ring '
                'switch, and the ring DoF are frozen at the reference')
        import contextlib
        import io
        from build_prior_states import draw_states
        with contextlib.redirect_stdout(io.StringIO()):
            xt, _ = draw_states(en, prior, n, rng)
        return xt.to(en.dtype), None
    x, st = en.sample_prior_states(prior, n, rng, report=False, joint_rings=joint_rings)
    return x.detach(), st


def ring_arm_status(en, sampler):
    """Why an arm cannot run, or None if it can. The four reasons are kept apart.

    A negative control that silently degenerates into a copy of the arm it controls is
    worse than an absent one: both cells then agree, and the agreement reads as evidence.
    """
    if sampler != 'prior-rings-off':
        return None
    if en.collective:
        return ('N/A at a collective tier: draw_states has no joint_rings switch and '
                'ring DoF are frozen')
    if not en.atom_in_ring.any():
        return 'N/A: acyclic molecule, there is no ring block to disable'
    return None


def optimize(en, x, steps, optimizer='rprop'):
    """Relax in the TIER'S OWN coordinate. Returns (x_opt, budget_string).

    ONE route for every tier -- there is no DoF-space variant and no projection. The
    budget string names the OPTIMIZER as well as the step count: changing either changes
    what this arm is, and an optimized cell is only interpretable beside its cost.
    """
    x_opt, _ = descend(en, x, steps, optimizer=optimizer)
    return x_opt, '{} steps {}/x'.format(steps, optimizer)


def diversity(en, x):
    """(evenness, source, detail). Occupancy where coverage is not trustworthy.

    rotamer_modes is CACHED ON THE ENERGY. It is a 721-point scan of the true energy per
    rotatable group and it is a pure function of the molecule -- it does not look at the
    draw -- but it sits inside the per-seed loop, so it was recomputed once per arm per
    seed: 18 identical scans per molecule per tier, measured at 12% of total runtime.
    `en` is constructed fresh for each (molecule, tier), so the cache cannot outlive the
    thing it is keyed on.
    """
    try:
        cached = getattr(en, '_rotamer_modes_cache', None)
        if cached is None:
            cached = pdg.rotamer_modes(en)
            en._rotamer_modes_cache = cached
        groups = cached
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
    if stride < 2:
        # One basin cannot be occupied unevenly. Reporting 1.0 here would be a column that
        # passes because there is nothing to fail, which is worse than reporting nothing.
        return None, 'NO-CONTRAST', '1 enumerated basin'
    even = float(-(nz * np.log(nz)).sum() / math.log(stride))
    return even, 'occupancy', '{}/{} basins'.format(occupied, stride)


def teff_is_degenerate(en, sampler, opt_steps):
    """RAW PRIOR with free r/theta only: k cancels between the draw width and the score."""
    return (sampler.startswith('prior') and not opt_steps
            and en.n_r > 0 and en.ndim > en.n_ph)


def _ring_summary(per_seed):
    """Mean/sd over seeds of the ring numbers, keeping the seed-invariant ones as-is.

    THE POPULATION GUARD IS THE FIRST FIELD. A ring statistic averaged over zero ring
    systems is an absence, not a pass, and ``n_ring_systems == 0`` is what tells them apart
    downstream -- so it is carried even when every other ring field is nan.
    """
    if not per_seed:
        return None
    first = per_seed[0]
    def ms(key):
        v = [r[key] for r in per_seed if r[key] == r[key]]      # drop nan
        return ((statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
                if v else (float('nan'), float('nan')))
    ca, ca_sd = ms('closure_err_a')
    cs, cs_sd = ms('closure_err_sigma')
    out = {k: first[k] for k in
           ('n_ring_systems', 'n_ring_cycles', 'n_closure_bonds', 'stale_prior',
            'banked_modes', 'banked_rows', 'held_aromatic', 'held_unsupported',
            'n_ring_using_bank', 'n_ring_held_fallback', 'n_ring_block_dof',
            'n_ring_extra_dof', 'ring_density')}
    out.update({'closure_err_a': ca, 'closure_err_a_sd': ca_sd,
                'closure_err_sigma': cs, 'closure_err_sigma_sd': cs_sd,
                'n_ring_dof_independent': first['n_ring_dof_independent'],
                'n_seeds': len(per_seed)})
    # per-cycle, averaged over seeds; each cycle keeps its own identity and contract
    def per_cycle(field, keys):
        rows = []
        for i in range(len(first[field])):
            r = {'size': first[field][i]['size'], 'atoms': first[field][i]['atoms']}
            for k in keys:
                v = [q[field][i][k] for q in per_seed if q[field][i][k] == q[field][i][k]]
                r[k] = statistics.mean(v) if v else float('nan')
            rows.append(r)
        return rows
    out['saturated'] = per_cycle('saturated', ('n_basins', 'evenness', 'top_frac',
                                               'median_abs_torsion_deg'))
    out['aromatic'] = per_cycle('aromatic', ('median_abs_torsion_deg',
                                             'p90_abs_torsion_deg'))
    return out


def cell(en, prior, sampler, opt_steps, n, seeds, zero, optimizer='rprop'):
    """One cell. Independent seed per arm; spread reported, never a point estimate."""
    mn, p10, med, p90, therm, teff, evens = [], [], [], [], [], [], []
    budget, dsrc, ddet = 'raw', None, None
    na = ring_arm_status(en, sampler)
    if na is not None:
        return {'blocked': na, 'budget': 'N/A', 'n_seeds': len(seeds), 'n': n,
                'inapplicable': True}
    rings = []
    for s in seeds:
        # ONE missing inverse blocks most of this table at a collective tier. state_from_dof
        # has no selection map when a column drives several dihedrals, which takes out the
        # fitted-prior DRAW, both OPTIMIZED arms and DIVERSITY -- everything except uniform.
        # That is gate condition 4, and it is why the shipped tier is nearly empty here.
        try:
            if sampler == 'uniform':
                x, st = draw_uniform(en, n, s), None
            else:
                x, st = draw_prior(en, prior, n, s,
                                   joint_rings=(sampler != 'prior-rings-off'))
            if opt_steps:
                x, budget = optimize(en, x, opt_steps, optimizer)
        except NotImplementedError as exc:
            return {'blocked': str(exc).split('.')[0].strip(), 'budget': 'BLOCKED',
                    'n_seeds': len(seeds), 'n': n}
        if en.atom_in_ring.any():
            # measured on the x that is SCORED, so an optimized arm reports the closure it
            # actually achieved rather than the closure it was handed
            rings.append(rmet.ring_measurements(en, x, prior, st))
        e = excess_kt(en, x, zero)
        # MIN and P10 are not decoration. The tier floor is -d/2 by construction, so a
        # median of 60 with a min of -0.79 is a spread story, not a catastrophe -- and a
        # table that starts at the median cannot tell those apart.
        mn.append(float(e.min()))
        p10.append(float(np.percentile(e, 10)))
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
        'min_kt': statistics.mean(mn), 'min_kt_sd': sd(mn),
        'p10_kt': statistics.mean(p10), 'p10_kt_sd': sd(p10),
        'median_kt': statistics.mean(med), 'median_kt_sd': sd(med),
        'p90_kt': statistics.mean(p90), 'p90_kt_sd': sd(p90),
        'thermal_frac': statistics.mean(therm),
        'T_eff': statistics.mean(teff), 'T_eff_sd': sd(teff),
        'T_eff_degenerate': teff_is_degenerate(en, sampler, opt_steps),
        'evenness': statistics.mean(evens) if evens else None,
        'evenness_sd': sd(evens) if evens else None,
        'div_source': dsrc, 'div_detail': ddet,
        'budget': budget, 'n_seeds': len(seeds), 'n': n,
        'seeds': list(seeds),
        'rings': _ring_summary(rings),
    }


def load_prior(prior_path, allow_stale=False):
    """Load the prior and REFUSE a stale ring signature unless it was asked for.

    vars(), not getattr. InternalPrior is a dataclass, so a field with a default is also a
    CLASS attribute -- getattr on a prior pickled before the field existed returns the
    current default and reports itself up to date while none of its ring keys resolve.
    Every ring then reads held_unsupported for a reason that has nothing to do with the
    molecule, which is precisely the confusion the four ring classes exist to prevent.
    """
    prior = torch.load(prior_path, weights_only=False)
    ver = vars(prior).get('ring_sig_version', 1)
    if ver < 2:
        msg = ('{} has ring_sig_version {} (pre-fix). NO ring key can resolve, so every '
               'ring would read held_unsupported and the ring block of this table would '
               'be meaningless. Rebuild with build_ring_banks.py, or pass '
               '--allow-stale-prior to measure the stale prior deliberately.'
               .format(prior_path, ver))
        if not allow_stale:
            raise SystemExit('REFUSING: ' + msg)
        print('*' * RULE)
        print('WARNING running on a STALE prior by request -- ' + msg)
        print('*' * RULE)
    return prior, int(ver)


def run(tiers, mols, n, seeds, opt_steps, prior, allow_stale=False, optimizer='rprop'):
    out = {}
    for tier in tiers:
        rows = []
        for name, smi in mols:
            # STDERR, and before the work rather than after. Nothing is printed until every
            # tier has finished -- the tables need the whole result -- so without this a
            # multi-tier run is indistinguishable from a hang for tens of minutes.
            print('  [{}] {} ...'.format(tier, name), file=sys.stderr, flush=True)
            try:
                en = ConformerTorsions(smiles=smi, level=tier, force_field='mmff',
                                       log_temperature=0.0, device='cpu')
            except Exception as exc:
                rows.append({'molecule': name,
                             'error': '{}: {}'.format(type(exc).__name__, exc)})
                continue
            # THE TIER'S OWN ZERO, computed before any arm is scored. Starts: the
            # reference state, the fitted prior, and uniform -- so the search is not
            # seeded only from the thing being measured.
            starts = [torch.zeros(1, en.ndim, dtype=en.dtype),
                      draw_uniform(en, 64, 991)]
            try:
                starts.append(draw_prior(en, prior, 64, 992)[0])
            except NotImplementedError:
                pass
            zmin, zworst, nstart = tier_minimum(en, torch.cat(starts, 0))
            row = {'molecule': name, 'd': int(en.ndim), 'arms': {},
                   'zero': zmin, 'zero_starts': nstart,
                   'zero_spread': zworst - zmin,
                   'has_rings': bool(en.atom_in_ring.any()),
                   'ring_role': RING_MOLECULES.get(name),
                   'zero_vs_e_ref': (zmin - float(en.e_ref)) / float(en.temperature)}
            zero = zmin
            for sampler in SAMPLERS:
                for tag, steps in (('raw', 0), ('optimized', opt_steps)):
                    row['arms']['{}/{}'.format(sampler, tag)] = cell(
                        en, prior, sampler, steps, n, seeds, zero, optimizer)
            rows.append(row)
        out[tier] = rows
    return out


RULE = 168
ARMW = 27                 # 'prior-rings-on/optimized' is 24 -- do not shrink below it
HEAD = ('{:<24}{:>4}  {:<{w}}{:>9}{:>9}{:>16}{:>10}{:>7}{:>14}      {:<26}{}'
        .format('molecule', 'd', 'arm', 'min kT', 'p10 kT', 'med kT', 'p90 kT',
                'therm', 'T_eff', 'diversity', 'budget', w=ARMW))


def fmt(tier, rows, n, seeds, opt_steps):
    out = ['=' * RULE,
           'TIER {!r}    n={} per seed    seeds={}    optimized arm = {} steps'
           .format(tier, n, seeds, opt_steps),
           '  WELL POSED WITHIN THIS TIER ONLY. Freezing a DoF gives a conditional slice',
           '  differing by a state-dependent Fixman factor, so a row here does NOT',
           '  approximate the same row at another tier.',
           '  energy = (U - U_tier_min)/kT - d/2, mean over seeds +- sd across seeds.',
           '  The floor is -d/2 BY CONSTRUCTION: a min column sitting at -d/2 means the',
           '  arm reached this tier own minimum; anything above it is the shortfall.',
           '  *deg = T_eff degenerate by construction (k cancels); read the kT columns.',
           '=' * RULE, HEAD, '-' * RULE]
    spans = []
    for r in rows:
        if 'error' in r:
            out.append('{:<24}{:>4}  BUILD FAILED: {}'.format(r['molecule'], '', r['error']))
            continue
        first = True
        for arm, c in r['arms'].items():
            head = ('{:<24}{:>4}  '.format(r['molecule'], r['d']) if first else ' ' * 30)
            first = False
            if 'blocked' in c:
                out.append('{}{:<{w}}{:>8}  {}'.format(
                    head, arm, 'N/A' if c.get('inapplicable') else 'BLOCKED', c['blocked'],
                    w=ARMW))
                continue
            deg = '*deg' if c['T_eff_degenerate'] else '    '
            div = (c['div_source'] if c['evenness'] is None
                   else '{:.3f}+-{:.3f} {}'.format(c['evenness'], c['evenness_sd'],
                                                   c['div_source']))
            if c['evenness'] is not None:
                spans.append(c['evenness'])
            out.append('{}{:<{w}}{:>9.2f}{:>9.2f}{:>9.1f}+-{:<5.1f}{:>10.1f}{:>6.1f}%'
                       '{:>8.2f}+-{:<4.2f}{}  {:<26}{}'
                       .format(head, arm, c['min_kt'], c['p10_kt'],
                               c['median_kt'], c['median_kt_sd'], c['p90_kt'],
                               c['thermal_frac'] * 100, c['T_eff'], c['T_eff_sd'], deg,
                               div, c['budget'], w=ARMW))
        out.append('{}zero = tier minimum {:.3f} kcal/mol from {} starts '
                   '(spread {:.2f}); it sits {:+.1f} kT from e_ref -- that offset is '
                   'UNREACHABLE at this tier'
                   .format(' ' * 30, r['zero'], r['zero_starts'], r['zero_spread'],
                           r['zero_vs_e_ref']))
        probe = r['arms']['uniform/raw']
        if probe['div_source'] == 'BLOCKED':
            out.append('{}DIVERSITY BLOCKED at this tier -- {}'
                       .format(' ' * 30, probe['div_detail']))
    out.append('-' * RULE)
    out.append(diversity_verdict(tier, rows, spans))
    out.append('')
    return '\n'.join(out)


def diversity_verdict(tier, rows, spans):
    """Say out loud what the diversity column separated, and what it did not.

    A column every arm passes is not evidence that every arm is diverse; it is evidence
    the column has no contrast here. Reporting the RANGE alone is not enough either: a
    range can be statistically resolvable and still mean nothing, because uniform draws
    are even over basins BY CONSTRUCTION. So the ordering is printed too -- if the only
    thing the column resolves is that uniform beats the prior at being uniform, a reader
    can see that for themselves instead of reading it as a quality verdict.
    """
    scored = [r for r in rows if 'error' not in r]
    multi = sum(1 for r in scored
                if r['arms']['uniform/raw'].get('div_source') == 'occupancy')
    if not spans:
        return ('DIVERSITY: no molecule at {!r} produced more than one enumerated basin '
                '-- THE COLUMN CANNOT DISCRIMINATE AT THIS TIER.'.format(tier))
    by_arm = {}
    for r in scored:
        for arm, c in r['arms'].items():
            if c.get('evenness') is not None:
                by_arm.setdefault(arm, []).append(c['evenness'])
    rank = sorted(((sum(v) / len(v), a) for a, v in by_arm.items()), reverse=True)
    lo, hi = min(spans), max(spans)
    out = ['DIVERSITY: {}/{} molecules have >1 enumerated basin; evenness over {} scored '
           'arms spans [{:.3f}, {:.3f}], range {:.3f}.'
           .format(multi, len(scored), len(spans), lo, hi, hi - lo),
           '  by arm: ' + '   '.join('{} {:.3f}'.format(a, m) for m, a in rank)]
    if rank and rank[0][1].startswith('uniform'):
        out.append('  THE TOP ARM IS UNIFORM, which is even over basins BY CONSTRUCTION -- '
                   'this column is NOT ranking sample quality and a high score here is not '
                   'a pass. B1 (relaxed-scan enumeration) is what would give it contrast.')
    elif hi - lo < 0.05:
        out.append('  THE COLUMN DOES NOT DISCRIMINATE HERE -- every arm scores the same to '
                   'within {:.3f}. Do not read it as an arm comparison; B1 (relaxed-scan '
                   'enumeration) is what would give it contrast.'.format(hi - lo))
    return '\n'.join(out)


RING_RULE = 168


def ring_separation(rows):
    """Per ring molecule, how far the negative control moved closure. ``[(name, on, off)]``.

    THE ONE NUMBER THAT MAKES THE REST OF THE RING BLOCK READABLE. Every other ring column
    is a property of a single arm and cannot say whether the measurement is live; this is
    the only one that can, because both arms are drawn through the same scorer and differ
    in exactly one switch.
    """
    out = []
    for r in rows:
        if 'error' in r or not r.get('has_rings'):
            continue
        on = r['arms'].get('prior-rings-on/raw', {})
        off = r['arms'].get('prior-rings-off/raw', {})
        if not on.get('rings') or not off.get('rings'):
            continue
        out.append((r['molecule'], on['rings']['closure_err_sigma'],
                    off['rings']['closure_err_sigma']))
    return out


def fmt_rings(tier, rows):
    """The ring block. Separate table because its rows are a different population.

    Folding rings into the energy table would put a nan closure column on every acyclic
    molecule and invite reading it as a pass.
    """
    ring_rows = [r for r in rows if 'error' not in r and r.get('has_rings')]
    out = ['=' * RING_RULE,
           'RINGS at tier {!r}   -- closure, class accounting, pucker and planarity. '
           'ENERGY columns are in the table above.'.format(tier),
           '  CLASSES ARE NOT COLLAPSED: banked = pucker SAMPLED from a fitted subspace/'
           'bank; held-arom = planar BY DESIGN',
           '  (aromatic rings are rigid and are never banked); held-unsup = saturated but '
           'no bank resolved, a GAP; stale = the',
           '  prior predates the ring-signature fix, in which case NO key resolves and '
           'every ring reads unsupported.',
           '  DENSITY-DEPENDENT RING NUMBERS (ESS, D_avoidable, IS log Z) ARE UNAVAILABLE '
           'BY DERIVATION -- see the module docstring.',
           '  AN OPTIMIZED ARM MAY REPORT WORSE CLOSURE THAN ITS RAW ARM, and that is not '
           'a contradiction: closure is not a state',
           '  coordinate, so descent on U trades ring opening against everything else. It '
           'is a cost of that proposal, printed beside it.',
           '=' * RING_RULE]
    if not ring_rows:
        out += ['  no ring molecule at this tier -- the ring columns have an EMPTY '
                'population and report nothing, which is not a pass.', '']
        return '\n'.join(out)
    out += ['{:<24}{:<20}{:>5}{:>6}{:>10}{:>16}{:>26}{:>10}  {}'
            .format('molecule', 'arm', 'sys', 'cyc', 'closure A', 'closure sigma',
                    'classes b/arom/unsup', 'indep DoF', 'seeds'),
            '-' * RING_RULE]
    for r in ring_rows:
        first = True
        for arm, c in r['arms'].items():
            if arm.startswith('uniform'):
                continue
            head = '{:<24}'.format(r['molecule']) if first else ' ' * 24
            first = False
            if c.get('rings') is None:
                out.append('{}{:<20}{:>73}  {}'.format(
                    head, arm, 'N/A' if c.get('inapplicable') else 'BLOCKED',
                    c.get('blocked', '')))
                continue
            g = c['rings']
            # '--', not 0. The collective route (draw_states) reports no per-DoF ring
            # accounting at all, and printing 0 there would claim it drew none jointly.
            indep = ('--' if g['n_ring_dof_independent'] is None
                     else str(g['n_ring_dof_independent']))
            out.append('{}{:<20}{:>5}{:>6}{:>10.4f}{:>10.2f}+-{:<4.2f}'
                       '{:>13}/{}/{}{:>10}  {} ({})'
                       .format(head, arm, g['n_ring_systems'], g['n_ring_cycles'],
                               g['closure_err_a'], g['closure_err_sigma'],
                               g['closure_err_sigma_sd'],
                               g['n_ring_using_bank'], g['held_aromatic'],
                               g['held_unsupported'],
                               indep, g['n_seeds'], c['budget']))
        # the per-ring contracts, reported against the contract each ring actually has
        on = r['arms'].get('prior-rings-on/raw', {})
        g = on.get('rings')
        if g:
            if g['stale_prior']:
                out.append('{}STALE PRIOR: no ring key can resolve; the class column above '
                           'is not about this molecule'.format(' ' * 24))
            # the sampler's own bar, applied to the arm the table calls correct
            if g['closure_err_sigma'] > 3.0:
                out.append('{}RINGS-ON CLOSURE IS {:.1f} BOND-SIGMA -- above 3, the ring is '
                           'visibly OPEN on the path this table calls correct. The negative '
                           'control still separates, so the measurement is live; this is a '
                           'real weakness of the ring prior on this molecule, not an '
                           'artefact.'.format(' ' * 24, g['closure_err_sigma']))
            for s in g['saturated']:
                # ONE basin is NO-CONTRAST, not evenness 1.0 and not a nan to be squinted
                # at: a single basin cannot be occupied unevenly, so the column has nothing
                # to fail. On a held_unsupported ring that is the EXPECTED reading -- the
                # pucker is rattled, not sampled -- and printing it as a number invites it
                # to be compared against a banked ring's.
                occ = ('NO CONTRAST: 1 basin (pucker is held, not sampled)'
                       if int(s['n_basins']) < 2 else
                       '{} basins occupied, evenness {:.3f}, top basin {:.1%}'
                       .format(int(s['n_basins']), s['evenness'], s['top_frac']))
                out.append('{}saturated {}-ring {}: {}, median |ring torsion| {:.1f} deg'
                           .format(' ' * 24, s['size'], s['atoms'], occ,
                                   s['median_abs_torsion_deg']))
            for a in g['aromatic']:
                flag = '' if a['median_abs_torsion_deg'] < 5.0 else '   <-- NOT PLANAR'
                out.append('{}aromatic {}-ring {}: median |ring torsion| {:.2f} deg, p90 '
                           '{:.2f} -- PLANAR BY DESIGN, not a sampled pucker{}'
                           .format(' ' * 24, a['size'], a['atoms'],
                                   a['median_abs_torsion_deg'],
                                   a['p90_abs_torsion_deg'], flag))
            if r.get('ring_role'):
                out.append('{}role: {}'.format(' ' * 24, r['ring_role']))
            out.append('{}ring density: {}'.format(' ' * 24, g['ring_density']))
        out.append('-' * RING_RULE)
    out.append(ring_verdict(tier, rows))
    out.append('')
    return '\n'.join(out)


def ring_verdict(tier, rows):
    """Say out loud whether the ring columns are LIVE at this tier.

    A ring table where both arms agree is not evidence that rings are handled well; it is
    evidence that the switch did not reach the sampler, and every other ring number on the
    page is then unsupported. So the separation is stated as the verdict rather than left
    for the reader to compute.
    """
    sep = ring_separation(rows)
    if not sep:
        # INAPPLICABLE and NOT DEMONSTRATED are different verdicts. At a collective tier
        # the control cannot exist -- draw_states has no ring switch and the ring DoF are
        # frozen -- and reporting that as a failed control would be manufacturing one.
        ring_rows = [r for r in rows if 'error' not in r and r.get('has_rings')]
        if ring_rows and all(r['arms'].get('prior-rings-off/raw', {}).get('inapplicable')
                             for r in ring_rows):
            return ('RING CONTROL: INAPPLICABLE at {!r} -- the draw goes through '
                    'build_prior_states.draw_states, which has no joint_rings switch, and '
                    'the ring DoF are frozen at the reference. Read the ring columns at a '
                    'selection tier, where the control runs.'.format(tier))
        return ('RING CONTROL: no ring molecule ran BOTH arms at {!r} -- the ring '
                'measurements here are NOT demonstrated live.'.format(tier))
    worst = min((off / max(on, 1e-12), nm, on, off) for nm, on, off in sep)
    lines = ['RING CONTROL (closure, bond-sigma, rings ON -> OFF):  '
             + '   '.join('{} {:.2f}->{:.2f}'.format(nm, on, off) for nm, on, off in sep)]
    if worst[0] < 2.0:
        lines.append('  THE CONTROL DID NOT SEPARATE on {} ({:.2f} vs {:.2f} bond-sigma). '
                     'Both arms are reaching the same sampling path, so NOTHING in this '
                     'ring block is demonstrated live.'.format(worst[1], worst[2], worst[3]))
    else:
        lines.append('  Separated by at least {:.0f}x on every ring molecule, so the '
                     'closure and pucker columns are measuring the switch. Note the OFF '
                     'arm is a BROKEN PROPOSAL, not a worse one: it violates closure by '
                     'construction.'.format(worst[0]))
    return '\n'.join(lines)


def bottoms(res, tiers):
    """The tier-dependent bottom, across tiers, in ONE place.

    How low a molecule can go is a property of the PARAMETERISATION, not of the sampler,
    and it is the one quantity in this report meant to be read across tiers. Only the
    kcal/mol column is directly comparable -- the kT columns are each measured against
    their own tier floor, so a smaller number there does not mean a lower structure.
    """
    names = []
    out = ['=' * RULE,
           'PARAMETERISATION BOTTOM -- the kcal/mol column is comparable ACROSS tiers; '
           'the kT columns are tier-relative and are NOT',
           '=' * RULE,
           '{:<17}{:<10}{:>4}{:>16}{:>12}{:>12}{:>14}  {}'
           .format('molecule', 'tier', 'd', 'tier min kcal', 'vs e_ref kT', 'floor -d/2',
                   'best arm min', 'best arm')]
    for t in tiers:
        for r in res.get(t, []):
            if 'error' not in r and r['molecule'] not in names:
                names.append(r['molecule'])
    for nm in names:
        first = True
        for t in tiers:
            r = next((q for q in res.get(t, []) if q.get('molecule') == nm), None)
            if r is None or 'error' in r:
                continue
            scored = [(a, c) for a, c in r['arms'].items() if 'blocked' not in c]
            if scored:
                arm, c = min(scored, key=lambda kv: kv[1]['min_kt'])
                best, tag = '{:.2f}'.format(c['min_kt']), arm
            else:
                best, tag = '--', 'all blocked'
            out.append('{:<17}{:<10}{:>4}{:>16.3f}{:>12.1f}{:>12.2f}{:>14}  {}'
                       .format(nm if first else '', t, r['d'], r['zero'],
                               r['zero_vs_e_ref'], -r['d'] / 2.0, best, tag))
            first = False
    out.append('')
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tiers', nargs='+', default=list(TIERS), choices=TIERS)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seeds', type=int, default=3)
    # 10 RPROP steps, not 25 Adam: Rprop reaches at 10 what Adam reached at 25-50.
    ap.add_argument('--opt-steps', type=int, default=10)
    ap.add_argument('--optimizer', default='rprop', choices=('rprop', 'adam'))
    ap.add_argument('--only', nargs='*')
    # v2 is the ring-signature-fixed prior. A v1 prior resolves NO ring key, so every ring
    # would read held_unsupported and the ring block would be silently meaningless.
    ap.add_argument('--prior-path', default='conformer_prior_v2.pt')
    ap.add_argument('--allow-stale-prior', action='store_true',
                    help='run anyway on a pre-ring-signature prior, loudly labelled')
    ap.add_argument('--rings-only', action='store_true',
                    help='restrict to the ring set -- the cheap reproducible ring run')
    ap.add_argument('--json', default=None)
    args = ap.parse_args(argv)

    mols = MOLECULES
    if args.rings_only:
        mols = [m for m in mols if m[0] in RING_MOLECULES]
    mols = [m for m in mols if not args.only or m[0] in args.only]
    seeds = list(range(args.seeds))
    prior, ring_sig_version = load_prior(args.prior_path, args.allow_stale_prior)
    res = run(args.tiers, mols, args.n, seeds, args.opt_steps, prior,
              optimizer=args.optimizer)

    for t in args.tiers:
        print(fmt(t, res[t], args.n, seeds, args.opt_steps))
        print(fmt_rings(t, res[t]))
    if len(args.tiers) > 1:
        print(bottoms(res, args.tiers))
    RESULTS.mkdir(exist_ok=True)
    path = args.json or str(RESULTS / 'prior_baselines.json')
    pathlib.Path(path).write_text(
        json.dumps({'n': args.n, 'seeds': seeds, 'opt_steps': args.opt_steps,
                    'prior_path': args.prior_path, 'optimizer': args.optimizer,
                    'ring_sig_version': ring_sig_version,
                    'samplers': list(SAMPLERS),
                    'ring_molecules': RING_MOLECULES,
                    'tiers': res}, indent=1, default=float), encoding='utf-8')
    print('wrote {}'.format(path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
