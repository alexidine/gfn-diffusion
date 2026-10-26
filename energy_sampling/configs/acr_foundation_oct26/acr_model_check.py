"""Acridine polymorph check of one energy model (ASE, no MXtalTools): what a foundation model says about the known forms.

    python acr_model_check.py MODEL_KIND MODEL_PATH LABEL [--cif_dir DIR] [--out_dir DIR] [--cifs ACRDIN04,...]
        [--device cuda] [--fmax 0.002]

MODEL_KIND: mace (any .model: the fine-tuned acridine checkpoints, MACE-OFF) or uma (a fairchem checkpoint, task omc).
Per CIF (H-bearing CSD entries; symmetry expanded by ASE):
  1. gas reference: the isolated molecule taken from each crystal, relaxed (BFGS); the lowest over the CIFs is E_gas;
  2. experimental cell, atoms relaxed (fixes the X-ray H positions): E_latt(exp cell) = E/Z - E_gas;
  3. full relaxation, atoms and cell (FrechetCellFilter + BFGS): E_latt(relaxed), V/V_exp;
  4. rigid-molecule isotropic expansion of the relaxed crystal (cell and molecular centroids scaled, molecules rigid):
     energy above the relaxed crystal at cell scales 1.04, 1.10, 1.20, 1.30.
Energies in kJ/mol per molecule. Writes OUT_DIR/LABEL.json and prints a table. MACE models run in float32 and
their energies are read as the sum of node energies (see energy()); UMA uses task omc.
"""
import argparse
import json
import os
import time

import numpy as np

EV = 96.485  # kJ/mol per eV


HERE = os.path.dirname(os.path.abspath(__file__))
ALL_CIFS = ['ACRDIN04', 'ACRDIN05', 'ACRDIN06', 'ACRDIN07', 'ACRDIN08', 'ACRDIN09', 'ACRDIN10', 'ACRDIN11',
            'ACRDIN12', 'ACRDIN13']
# polymorph groups (13 Oct comparison): one form per group
GROUP = {'ACRDIN07': 'A', 'ACRDIN09': 'A', 'ACRDIN10': 'A', 'ACRDIN13': 'A', 'ACRDIN04': 'B', 'ACRDIN11': 'B',
         'ACRDIN05': 'C', 'ACRDIN06': 'D', 'ACRDIN08': 'E', 'ACRDIN12': 'F'}
SCALES = [1.04, 1.10, 1.20, 1.30]
N_ATOMS = 23


def energy(atoms):
    """Potential energy for lattice energies: on a MACE calculator the sum of its node_energy (the atoms' energies without
    the per-atom reference energies E0, which cancel in a lattice energy), else the total. A MACE with DFT atomic
    reference energies (MACE-OFF: about -1.46e6 kJ/mol per acridine molecule) loses up to ~2 kJ/mol per molecule in a
    float32 total; the node sum agreed with float64 to 0.03 kJ/mol on four acridine crystals (2026-10-26)."""
    e = atoms.get_potential_energy()
    r = getattr(atoms.calc, 'results', {})
    return float(np.sum(r['node_energy'])) if 'node_energy' in r else e


def calculator(kind, path, device):
    if kind == 'mace':
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=path, device=device, default_dtype='float32')
    if kind == 'uma':
        import torch
        torch.serialization.add_safe_globals([slice])
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        pu = pretrained_mlip.load_predict_unit(path, device=device)
        return FAIRChemCalculator(pu, task_name='omc')
    raise ValueError(kind)


def _neighbours(atoms):
    from ase.neighborlist import natural_cutoffs, NeighborList
    nl = NeighborList(natural_cutoffs(atoms, mult=1.1), self_interaction=False, bothways=True)
    nl.update(atoms)
    return nl


def molecules(atoms):
    """[n_mol] lists of atom indices, from covalent connectivity under the cell's periodicity."""
    from scipy.sparse.csgraph import connected_components
    n, lab = connected_components(_neighbours(atoms).get_connectivity_matrix(sparse=True))
    groups = [np.nonzero(lab == k)[0] for k in range(n)]
    sizes = sorted(len(g) for g in groups)
    if any(s != N_ATOMS for s in sizes):
        raise ValueError(f'molecule sizes {sizes}, expected all {N_ATOMS}')
    return groups


def unwrapped(atoms, idx):
    """Cartesian positions of one molecule made whole by walking its bonds, each bond taken with the periodic image the
    neighbour list found it through (a molecule can be longer than half a cell axis)."""
    nl = _neighbours(atoms)
    cell = atoms.cell.array
    pos = {int(idx[0]): atoms.positions[idx[0]].copy()}
    todo = [int(idx[0])]
    while todo:
        i = todo.pop()
        nbrs, offs = nl.get_neighbors(i)
        for j, o in zip(nbrs, offs):
            j = int(j)
            if j not in pos:
                pos[j] = atoms.positions[j] + o @ cell - atoms.positions[i] + pos[i]
                todo.append(j)
    if len(pos) != len(idx):
        raise ValueError(f'unwrap reached {len(pos)} of {len(idx)} atoms')
    return np.stack([pos[int(k)] for k in idx])


def relax(atoms, fmax, steps, cell=False):
    from ase.optimize import BFGS
    if cell:
        from ase.filters import FrechetCellFilter
        obj = FrechetCellFilter(atoms)
    else:
        obj = atoms
    opt = BFGS(obj, logfile=None, maxstep=0.1)
    ok = opt.run(fmax=fmax, steps=steps)
    return bool(ok), opt.get_number_of_steps()


def rigid_scaled(atoms, groups, s):
    out = atoms.copy()
    pos = out.positions.copy()
    new_cell = atoms.cell.array * s
    for g in groups:
        p = unwrapped(atoms, g)
        c = p.mean(0)
        pos[g] = p - c + c * s
    out.set_cell(new_cell, scale_atoms=False)
    out.positions = pos
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['mace', 'uma'])
    ap.add_argument('path')
    ap.add_argument('label')
    ap.add_argument('--cif_dir', default='/scratch/mk8347/data/crystal_datasets/acridine')
    ap.add_argument('--out_dir', default=HERE)
    ap.add_argument('--cifs', default=','.join(ALL_CIFS))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--fmax', type=float, default=0.002)
    ap.add_argument('--steps', type=int, default=2000)
    a = ap.parse_args()
    from ase.io import read
    calc = calculator(a.kind, a.path, a.device)
    t0 = time.time()
    rows, gas = {}, []
    for name in a.cifs.split(','):
        xtal = read(os.path.join(a.cif_dir, name + '.cif'))
        groups = molecules(xtal)
        Z = len(groups)
        mol = xtal[groups[0]].copy()
        mol.positions = unwrapped(xtal, groups[0])
        mol.set_cell(None)
        mol.pbc = False
        mol.calc = calc
        ok_g, n_g = relax(mol, a.fmax / 2, a.steps)
        gas.append(energy(mol) * EV)
        rows[name] = dict(Z=Z, V_exp=float(xtal.get_volume()), gas_ok=ok_g)
        x = xtal.copy()
        x.calc = calc
        ok1, n1 = relax(x, a.fmax, a.steps)
        rows[name].update(E_fix=energy(x) * EV / Z, fix_ok=ok1, fix_steps=n1)
        y = x.copy()
        y.calc = calc
        ok2, n2 = relax(y, a.fmax, a.steps, cell=True)
        e_min = energy(y) * EV / Z
        rows[name].update(E_rel=e_min, rel_ok=ok2, rel_steps=n2, V_rel=float(y.get_volume()),
                          cell_rel=[float(v) for v in y.cell.cellpar()])
        g2 = molecules(y)
        rise = []
        for s in SCALES:
            z = rigid_scaled(y, g2, s)
            z.calc = calc
            rise.append(energy(z) * EV / Z - e_min)
        rows[name]['rise'] = rise
        print(f'{name}: Z {Z}, done ({time.time() - t0:.0f} s; relax ok {ok1}/{ok2}, steps {n1}/{n2})', flush=True)
    E_gas = min(gas)
    for name, r in rows.items():
        r['L_fix'] = r['E_fix'] - E_gas
        r['L_rel'] = r['E_rel'] - E_gas
        r['V_ratio'] = r['V_rel'] / r['V_exp']
    e_best = min(r['L_rel'] for r in rows.values())
    out = dict(label=a.label, kind=a.kind, model=os.path.basename(a.path), E_gas=E_gas, gas_spread=max(gas) - E_gas,
               rows=rows)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, f'{a.label}.json'), 'w'), indent=1)
    print(f'\n{a.label} ({os.path.basename(a.path)}): lattice energies in kJ/mol per molecule against the relaxed gas '
          f'molecule (gas minima over the {len(gas)} starts agree to {max(gas) - E_gas:.2f}).')
    print('| CIF | form | Z | lattice, exp cell | lattice, relaxed | above best relaxed | V relaxed / V exp | '
          'rise at 1.04 / 1.10 / 1.20 / 1.30 |')
    print('|---|---|---|---|---|---|---|---|')
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]['L_rel']):
        print(f"| {name} | {GROUP[name]} | {r['Z']} | {r['L_fix']:.1f} | {r['L_rel']:.1f} | {r['L_rel'] - e_best:.1f} | "
              f"{r['V_ratio']:.3f} | {' / '.join(f'{v:.1f}' for v in r['rise'])} |"
              + ('' if r['fix_ok'] and r['rel_ok'] else ' (relax hit the step cap)'))


if __name__ == '__main__':
    main()
