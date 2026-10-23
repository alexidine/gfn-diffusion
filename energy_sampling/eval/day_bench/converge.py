"""Carry relaxed endpoints on to a force threshold, counting the energy calls each structure costs.

``p1_relax`` gives every structure the same fixed number of steps, so its endpoints sit at different distances from a
stationary point, and two relaxations of one minimum from different starts can end measurably apart. Its optimiser
cannot finish the job: it returns each row's lowest-energy step, and near a minimum the energy (float32, about -75 kJ/mol
per molecule) changes by less than its own resolution, so that choice is noise. Here each endpoint is relaxed further
by FIRE (Bitzek et al. 2006, as ASE implements it), which moves on forces alone and keeps its last state, until its
largest generalised force is at most ``F_TOL`` or ``MAX_STEPS`` steps have run; a structure still above the threshold
then is written as it is and marked unconverged.

Generalised forces (``measures``) are autograd gradients of the target energy per molecule with respect to the
optimiser's own parameters (``energy_grad`` evaluates them as one ``gradient_descent_optimization`` step does):
- molecule force: the centroid gradient taken to Cartesian (``T_fc^-T``), times Z' (kJ/mol per Angstrom);
- molecule rotation: the rotation-vector gradient, times Z' (kJ/mol per radian);
- cell: the cell-length gradient (kJ/mol per molecule per Angstrom) and cell-angle gradient (kJ/mol per molecule per
  radian).
With the energy per molecule, Z' times a molecule's gradient is the gradient of the energy per cell for a P1 cell of
Z' = Z molecules, and the symmetrised force on each molecule for a Z' = 1 space-group crystal. FIRE moves in
length-like coordinates: cell lengths as they are, angles times the mean cell length, fractional centres times their
cell length, rotations times ``ROT_RADIUS``.

Energy calls per structure: ``FIRST_PASS_CALLS`` for ``p1_relax``'s stages (each stage evaluates ``max_num_steps``
times: ``max_num_steps - 1`` steps and a final analysis), one for the force check of the endpoint, and one per FIRE
step.

Two descriptions, as the first pass used them: ``p1`` rebuilds a generator's endpoint as a P1 crystal of Z rigid
molecules (``p1_relax.p1_crystal``); ``sg`` rebuilds a structure relaxed in its space group (reference minima,
experimental forms, random-search endpoints) as a Z' = 1 crystal there (``p1_ingest.ingest`` at the tight tolerances),
falling back to ``p1`` when no symmetric description is found.

usage (licensed CSD Python API):
    python -m eval.day_bench.converge SAMPLE_DIR REFCODE {p1|sg} [FORMS] [energy] [device]

Reads the built rows of SAMPLE_DIR/p1_relax_<energy>.csv and their endpoints in relaxed_<energy>/, plus
relaxed_<energy>/exp_<form>.cif for each comma-separated form present; writes converged_<energy>/ and
converge_<energy>.csv.
"""
import csv
import os
import sys
import time
import warnings

import numpy as np
import torch
from ase.io import read, write

from eval.day_bench.p1_ingest import ingest, make_reference, split_molecules
from eval.day_bench.p1_relax import STAGES, TIGHT, p1_crystal, stages_for, to_atoms
from mxtaltools.crystal_search.crystal_opt_utils import canonicalize_aunit, wrap_centroid
from mxtaltools.dataset_utils.utils import collate_data_list

FIRST_PASS_CALLS = sum(s['max_num_steps'] for s in STAGES)
F_TOL = 0.01  # kJ/mol per Angstrom or per radian, every generalised force
MAX_STEPS = 1000
FIRE = dict(dt=0.01, dt_max=0.1, max_step=0.02, n_min=5, f_inc=1.1, f_dec=0.5, a_start=0.1, f_a=0.99)
ROT_RADIUS = 2.0  # Angstrom an atom moves per radian of molecular rotation, about a small molecule's radius
CHUNK = 50


def prepare(batch):
    """A clone with its molecules in the standard frame, as ``optimize_crystal_parameters`` prepares one, and its
    [n, 6 + 6 * max_z_prime] parameters."""
    b = batch.clone().detach()
    b.orient_molecule(mode='standardized', target_handedness=b.aunit_handedness)
    return b, b.full_cell_parameters().detach().clone()


def energy_grad(b, x, energy='exp6', exp6_kwargs=None):
    """Target energy per molecule [n], its gradient with respect to ``x`` [n, P], and the cells' T_fc [n, 3, 3]."""
    p = x.clone().requires_grad_(True)
    c = b.clone().detach()
    c.set_cell_parameters(p, skip_box_analysis=True)
    c.clean_cell_parameters(mode='hard', canonicalize_orientations=True)
    zp = int(c.max_z_prime)
    c.aunit_centroid = wrap_centroid(p[:, 6:6 + 3 * zp])
    out = c.analyze(computes=[energy], cutoff=10, rdf_cutoff=10, supercell_size=10, return_cluster=False, repulsion=1,
                    surface_padding=0, std_orientation=True, margin=0, elementwise=True, atomwise=False,
                    assign_outputs=False, bins=100, **{f'exp6_{k}': v for k, v in (exp6_kwargs or {}).items()})
    e = out[energy]
    g, = torch.autograd.grad(e.sum(), p)
    return e.detach().double(), g.detach().double(), c.T_fc.detach().double()


def measures(g, T_fc, z_prime):
    """Largest generalised forces per row: 'mol', 'rot', 'len', 'ang' and 'max' of the four (see the module notes)."""
    n, zp = g.shape[0], (g.shape[1] - 6) // 6
    scale = z_prime.double().reshape(n, 1, 1)
    g_cen = g[:, 6:6 + 3 * zp].reshape(n, zp, 3)
    g_rot = g[:, 6 + 3 * zp:6 + 6 * zp].reshape(n, zp, 3)
    f_cart = torch.linalg.solve(T_fc.transpose(1, 2), g_cen.transpose(1, 2)).transpose(1, 2)  # T_fc^-T grad_frac
    real = (torch.arange(zp, device=g.device)[None, :] < z_prime.reshape(n, 1)).double()  # padded slots carry 0
    res = dict(mol=((f_cart * scale).norm(dim=-1) * real).amax(1), rot=((g_rot * scale).norm(dim=-1) * real).amax(1),
               len=g[:, 0:3].abs().amax(1), ang=g[:, 3:6].abs().amax(1))
    res['max'] = torch.stack(list(res.values()), 1).amax(1)
    return res


def fire(batch, device, energy='exp6', exp6_kwargs=None, f_tol=F_TOL, max_steps=MAX_STEPS):
    """FIRE on every row of ``batch`` until each row's largest generalised force is at most ``f_tol`` (a converged row
    stops moving) or ``max_steps`` steps have run. Returns (crystals, measures and energy before, after, FIRE steps
    taken per row)."""
    b, x = prepare(batch.to(device))
    n, zp = x.shape[0], int(b.max_z_prime)
    zpr = b.z_prime.reshape(n).to(device)
    mol_scale = torch.cat([torch.ones(n, 6, device=device, dtype=torch.float64),
                           zpr.double()[:, None].expand(n, 6 * zp)], 1)
    e, g, T = energy_grad(b, x, energy, exp6_kwargs)
    m = measures(g, T, zpr)
    before = {k: v.cpu().clone() for k, v in m.items()} | {'energy': e.cpu().clone()}
    x = x.double()
    v = torch.zeros_like(x)
    dt = torch.full((n,), FIRE['dt'], device=device, dtype=torch.float64)
    a = torch.full((n,), FIRE['a_start'], device=device, dtype=torch.float64)
    n_pos = torch.zeros(n, dtype=torch.long, device=device)
    steps = torch.zeros(n, dtype=torch.long, device=device)
    active = m['max'] > f_tol
    members = b.batch_to_list()
    rows, b_act = torch.arange(n, device=device), b  # the rows evaluated each step, and their batch
    for _ in range(max_steps):
        if not active.any():
            break
        if int(active.sum()) <= len(rows) // 2:  # drop converged rows from the evaluated batch
            rows = torch.nonzero(active).flatten()
            b_act = collate_data_list([members[i] for i in rows.tolist()], max_z_prime=zp).to(device)
        lengths = x[:, :3]
        s = torch.cat([torch.ones(n, 3, device=device, dtype=torch.float64),
                       lengths.mean(1, keepdim=True).expand(n, 3), lengths.repeat(1, zp),
                       torch.full((n, 3 * zp), ROT_RADIUS, device=device, dtype=torch.float64)], 1)
        F = -g * mol_scale / s
        up = (F * v).sum(1) > 0
        f_unit = F / F.norm(dim=1, keepdim=True).clamp_min(1e-12)
        v = torch.where(up[:, None], (1 - a[:, None]) * v + a[:, None] * f_unit * v.norm(dim=1, keepdim=True),
                        torch.zeros_like(v))
        grow = up & (n_pos > FIRE['n_min'])
        dt = torch.where(grow, (dt * FIRE['f_inc']).clamp_max(FIRE['dt_max']), torch.where(up, dt, dt * FIRE['f_dec']))
        a = torch.where(grow, a * FIRE['f_a'], torch.where(up, a, torch.full_like(a, FIRE['a_start'])))
        n_pos = torch.where(up, n_pos + 1, torch.zeros_like(n_pos))
        v = v + dt[:, None] * F
        dy = dt[:, None] * v
        dy = dy * (FIRE['max_step'] / dy.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp_max(1.0)
        dy = dy * active[:, None]
        v = v * active[:, None]
        x = x + dy / s
        steps += active.long()
        e_new, g_new, T_new = energy_grad(b_act, x[rows].float(), energy, exp6_kwargs)
        m_new = measures(g_new, T_new, zpr[rows])
        upd = active[rows]  # rows still in the evaluated batch that converged earlier keep their values
        g[rows[upd]] = g_new[upd]
        e[rows[upd]] = e_new[upd]
        for k in m:
            m[k][rows[upd]] = m_new[k][upd]
        active = active & (m['max'] > f_tol)
    c = b.clone().detach()
    c.set_cell_parameters(x.float(), skip_box_analysis=False)
    c.aunit_centroid = wrap_centroid(c.aunit_centroid)
    canonicalize_aunit(c)
    after = {k: v.cpu() for k, v in m.items()} | {'energy': e.cpu()}
    return c.cpu().batch_to_list(), before, after, steps.cpu().numpy()


def converge(crystals, device, energy='exp6', f_tol=F_TOL, max_steps=MAX_STEPS, log=print):
    """``fire`` over ``crystals`` (one description and Z' per call) in chunks. Returns (crystals, measures and energy
    before, after, FIRE steps per row, energy calls per row)."""
    exp6_kwargs = stages_for(energy)[0].get('exp6_kwargs')
    max_zp = max(int(c.z_prime) for c in crystals)
    out, before, after, steps = [], [], [], []
    chunk = CHUNK if max_zp <= 4 else CHUNK // 2
    for i in range(0, len(crystals), chunk):
        cs, f0, f1, st = fire(collate_data_list(crystals[i:i + chunk], max_z_prime=max_zp), device, energy,
                              exp6_kwargs, f_tol, max_steps)
        out += cs
        before.append(f0)
        after.append(f1)
        steps.append(st)
        log(f'  {min(i + chunk, len(crystals))} of {len(crystals)}: {int((f1["max"] <= f_tol).sum())} of {len(cs)} '
            f'at or below {f_tol}; FIRE steps median {int(np.median(st))}', flush=True)
    cat = lambda parts: {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    steps = np.concatenate(steps)
    return out, cat(before), cat(after), steps, FIRST_PASS_CALLS + 1 + steps


def load_rows(sample_dir, energy, forms):
    rows = [dict(r, kind='sample') for r in csv.DictReader(open(os.path.join(sample_dir, f'p1_relax_{energy}.csv')))
            if r['built'] == 'True']
    for rc in forms:
        if os.path.exists(os.path.join(sample_dir, f'relaxed_{energy}', f'exp_{rc}.cif')):
            rows.append(dict(file=f'exp_{rc}.cif', kind='exp'))
    return rows


def main(sample_dir, refcode, mode, forms=(), energy='exp6', device='cuda'):
    from ccdc import io as ccdc_io
    from eval.day_bench.csd_roundtrip import p1_cell
    torch.set_num_threads(2)
    if str(device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(0.4)
    reader = ccdc_io.EntryReader('CSD')
    atoms_ref, _, _ = p1_cell(refcode, reader)
    ref = make_reference(split_molecules(atoms_ref)[0], smiles=reader.entry(refcode).molecule.heaviest_component.smiles)

    rows = load_rows(sample_dir, energy, forms)
    groups = {}  # (description, Z') -> [(row, crystal)]
    for r in rows:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            atoms = read(os.path.join(sample_dir, f'relaxed_{energy}', r['file']))
        c, desc = None, 'p1'
        if mode == 'sg' or r['kind'] == 'exp':
            res = ingest(atoms, ref, symprecs=TIGHT, identifier=r['file'])
            if res.ok:
                c, desc = res.crystal, f'sg{res.sg}'
        if c is None:
            c, why = p1_crystal(atoms, ref, identifier=r['file'])
            if c is None:
                r['note'] = f'not rebuilt: {why}'
                continue
        r['desc'] = desc
        groups.setdefault(('sg' if desc != 'p1' else 'p1', int(c.z_prime)), []).append((r, c))

    out_dir = os.path.join(sample_dir, f'converged_{energy}')
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for (desc, zp), members in groups.items():
        print(f'{desc}, Z\' = {zp}: {len(members)} structures', flush=True)
        cs, f0, f1, steps, calls = converge([c for _, c in members], device, energy)
        for k, ((r, _), c) in enumerate(zip(members, cs)):
            write(os.path.join(out_dir, r['file']), to_atoms(c), format='cif')
            r.update(z_prime=zp, e_before=float(f0['energy'][k]), e_after=float(f1['energy'][k]),
                     f_before=float(f0['max'][k]), f_after=float(f1['max'][k]), f_mol=float(f1['mol'][k]),
                     f_rot=float(f1['rot'][k]), f_len=float(f1['len'][k]), f_ang=float(f1['ang'][k]),
                     fire_steps=int(steps[k]), calls=int(calls[k]), converged=bool(f1['max'][k] <= F_TOL))
    keys = ['file', 'kind', 'desc', 'z_prime', 'e_before', 'e_after', 'f_before', 'f_after', 'f_mol', 'f_rot',
            'f_len', 'f_ang', 'fire_steps', 'calls', 'converged', 'note']
    with open(os.path.join(sample_dir, f'converge_{energy}.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in keys} for r in rows])
    done = [r for r in rows if 'converged' in r]
    print(f'{len(done)} structures in {time.time() - t0:.0f} s on {device}; {sum(r["converged"] for r in done)} at or '
          f'below {F_TOL} kJ/mol per Angstrom or radian; largest force median '
          f'{np.median([r["f_before"] for r in done]):.3g} -> {np.median([r["f_after"] for r in done]):.3g}; energy '
          f'change median {np.median([r["e_after"] - r["e_before"] for r in done]):+.4f} kJ/mol per molecule; calls '
          f'median {int(np.median([r["calls"] for r in done]))}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         tuple(f for f in sys.argv[4].split(',') if f) if len(sys.argv) > 4 else (),
         sys.argv[5] if len(sys.argv) > 5 else 'exp6', sys.argv[6] if len(sys.argv) > 6 else 'cuda')
