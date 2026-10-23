"""Relax unstructured cells as Z rigid molecules in P1, then name the minimum each one reaches.

A generator's sample is Z independent molecules in P1 that obey a space group only roughly, so it cannot
enter the space-group relaxer directly (``p1_ingest`` symmetrises few raw samples). Here each sample becomes
a P1 crystal with Z' = Z: every molecule is replaced by the rigid reference fitted onto it
(``p1_ingest.fit_reference``), and each copy carries its own pose block (fractional centroid, rotation
vector, handedness), read as MXtalTools defines it by ``reparameterize_unit_cell`` on a one-molecule P1
crystal per copy. The crystals are relaxed with ``MolCrystalOps.optimize_crystal_parameters`` through the
stages given, every row taking every step, and each relaxed unit cell goes back through ``p1_ingest.ingest``
at tight tolerances to find the space group and Z' of the minimum reached.

usage (CPU by default):
    python -m eval.day_bench.p1_relax SAMPLE_DIR REFCODE [N_SAMPLES] [elj|exp6] [OTHER_FORM,OTHER_FORM] [DEVICE]

Relaxed cells go to SAMPLE_DIR/relaxed_<energy>/ as P1 CIFs, beside exp_<refcode>.cif for each experimental form.
"""
import copy
import csv
import glob
import os
import sys
import time
import warnings

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from eval.day_bench.p1_ingest import (Reference, _cell_handedness_fix, _crystal, _lengths_angles, fit_reference,
                                      ingest, make_reference, split_molecules)
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

# the two eLJ stages of mxtaltools configs/crystal_searches/qm9_full_sep29/tasks/0.yaml, every row taking
# every step (convergence_eps 0, as eval/cond_panel/relax.py does)
STAGES = [
    dict(optim_target='elj', enforce_reduced=False, compression_factor=1.0, cutoff=10, init_lr=0.05,
         convergence_eps=0.0, optimizer_func='rprop', anneal_lr=False, grad_norm_clip=0.1, show_tqdm=False,
         max_num_steps=350, target_packing_coeff=None, centroid_boundary='wrap'),
    dict(optim_target='elj', enforce_reduced=False, compression_factor=0.0, cutoff=10, init_lr=0.01,
         convergence_eps=0.0, optimizer_func='rprop', anneal_lr=True, grad_norm_clip=0.1, show_tqdm=False,
         max_num_steps=150, target_packing_coeff=None, centroid_boundary='wrap'),
]
TIGHT = (1e-3, 1e-2, 0.05, 0.1)  # Angstrom: tolerances for naming a relaxed minimum
CHUNK = 25  # crystals relaxed per batch
UNITS = {'elj': 'raw eLJ units per molecule', 'exp6': 'kJ/mol per molecule'}


def stages_for(energy):
    """The two stages with ``energy`` as the target: 'elj', or 'exp6' (FIT exp-6 with RDKit MMFF point charges; the
    key always evaluates the exp-6 pair form, see ``MolCrystalAnalysis.compute_exp6_energy``)."""
    out = [dict(s, optim_target=energy) for s in STAGES]
    if energy == 'exp6':
        for s in out:
            s['exp6_kwargs'] = {'params': 'fit', 'charges': 'mmff'}
    return out


def p1_crystal(atoms: Atoms, ref: Reference, identifier=None):
    """A P1 crystal with Z' = Z rigid copies of the reference, and the largest rebuild error (Angstrom).

    Returns (crystal, residual) or (None, reason).
    """
    try:
        mols = split_molecules(atoms)
        fitted = [fit_reference(m, ref)[0] for m in mols]
    except ValueError as e:
        return None, f'split/fit: {e}'
    zp = len(fitted)
    lattice = np.asarray(atoms.cell, dtype=float)
    frac = np.concatenate(fitted) @ np.linalg.inv(lattice)
    lattice, frac = _cell_handedness_fix(lattice, frac)
    lengths, angles = _lengths_angles(lattice)
    n_at = len(ref.mol.z)
    heavy = ref.mol.z > 1
    frac = frac.reshape(zp, n_at, 3)

    seeds = []
    for k in range(zp):
        c = frac[k][heavy].mean(0)
        f = frac[k] - np.floor(c)
        s = _crystal(ref, 1, lengths, angles, ref.mol.pos, c - np.floor(c), identifier=identifier)
        seeds.append((s, f))
    T_fc = seeds[0][0].T_fc[0].double().numpy()
    for s, f in seeds:
        s.pos = torch.tensor(f @ T_fc.T, dtype=torch.float32)
    batch = collate_data_list([s for s, _ in seeds], max_z_prime=1)
    batch.build_unit_cell()
    cen, ori, hand, _, _ = batch.reparameterize_unit_cell()

    mol = MolData(z=torch.tensor(ref.mol.z, dtype=torch.long), pos=torch.tensor(ref.mol.pos, dtype=torch.float32),
                  x=torch.zeros(len(ref.mol.z), 1), smiles=ref.smiles, do_mol_analysis=True)
    crystal = MolCrystalData(
        molecule=[mol.clone() for _ in range(zp)] if zp > 1 else mol, sg_ind=1,
        cell_lengths=torch.tensor(lengths, dtype=torch.float32),
        cell_angles=torch.tensor(angles, dtype=torch.float32),
        aunit_centroid=cen.reshape(-1).float(), aunit_orientation=ori.reshape(-1).float(),
        aunit_handedness=hand.reshape(-1).float(), identifier=identifier,
        z_prime=torch.ones(1, dtype=torch.long) * zp, max_z_prime=zp, do_box_analysis=True)

    # check: the posed unit cell reproduces the fitted copies
    chk = collate_data_list([copy.deepcopy(crystal)], max_z_prime=zp)
    chk.mol2ucell()
    uc = chk.unit_cell_pos.double().numpy() @ np.linalg.inv(T_fc).T
    want = frac.reshape(-1, 3)
    df = want[:, None] - uc[None]
    df -= np.round(df)
    residual = float(np.linalg.norm(df @ T_fc.T, axis=-1).min(1).max())
    return crystal, residual


def relax(batch, stages, device='cpu', scratch=None):
    """Stages in order; returns the relaxed CPU batch, the target energy per molecule before and after (eLJ is
    a pair sum over all Z' molecules and is divided by Z'; exp6 is per molecule already), and the per-step
    trajectory [steps, rows]."""
    b = batch.to(device)
    key = str(stages[-1]['optim_target'])
    zp = b.z_prime.double().flatten().cpu() if key == 'elj' else torch.ones(b.num_graphs, dtype=torch.float64)
    first, traj = None, []
    for stage in stages:
        out, rec = b.optimize_crystal_parameters(return_record=True, intermediates_path=scratch, **stage)
        assert len(out) == b.num_graphs, 'the optimiser dropped rows'
        e = rec[key].detach().cpu().double().reshape(rec[key].shape[0], -1)
        if first is None:
            first = e[0]
        traj.append(e)
        b = collate_data_list(out, max_z_prime=batch.max_z_prime).to(device)
    traj = torch.cat(traj) / zp
    return b.cpu().detach(), first / zp, traj[-1], traj.numpy()


def to_atoms(crystal) -> Atoms:
    """The relaxed crystal's unit cell as a P1 ``ase.Atoms`` (Cartesian, MXtalTools frame)."""
    one = collate_data_list([crystal], max_z_prime=crystal.max_z_prime)
    one.box_analysis()
    one.mol2ucell()
    T_fc = one.T_fc[0].double().numpy()
    # unit-cell atoms run operator by operator, each in the order of ``z`` (all Z' molecules at Z' > 1);
    # this run only builds P1 crystals (one operator) and Z' = 1 crystals
    return Atoms(numbers=np.tile(one.z.numpy(), int(one.sym_mult[0])),
                 positions=one.unit_cell_pos.double().numpy(), cell=T_fc.T, pbc=True)


def settle_step(traj_col, tol=0.1, window=50):
    """First step after which the energy never falls by more than ``tol`` raw units over ``window`` steps."""
    best = np.minimum.accumulate(traj_col)
    for i in range(len(best) - window):
        if best[i] - best[i + window] <= tol and best[i] - best[-1] <= tol:
            return i
    return len(best)


def relax_experimental(refcodes, ref, stages, reader, out_dir, scratch):
    """Each CSD form fitted with ``ref``, symmetrised, relaxed in its own space group by ``stages`` and written to
    ``out_dir/exp_<refcode>.cif``; returns {refcode: (space group, energy before, energy after)}."""
    from ase.io import write
    from eval.day_bench.csd_roundtrip import p1_cell
    out = {}
    for rc in refcodes:
        atoms, _, _ = p1_cell(rc, reader)
        res = ingest(atoms, ref, identifier=rc)
        if not res.ok:
            print(f'experimental form {rc} not converted: {res.reason}')
            continue
        b, e0, e1, _ = relax(collate_data_list([res.crystal], max_z_prime=1), stages, scratch=scratch)
        write(os.path.join(out_dir, f'exp_{rc}.cif'), to_atoms(b.batch_to_list()[0]), format='cif')
        out[rc] = (res.sg, float(e0[0]), float(e1[0]))
    return out


def main(sample_dir, refcode, n=25, energy='elj', extra_forms=(), device='cpu'):
    from ccdc import io as ccdc_io
    from eval.day_bench.csd_roundtrip import p1_cell
    torch.set_num_threads(2)
    if str(device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(0.4)
    reader = ccdc_io.EntryReader('CSD')
    atoms_ref, _, _ = p1_cell(refcode, reader)
    # the CSD SMILES gives the RDKit charge models their bond orders; a geometry without one is perceived
    ref = make_reference(split_molecules(atoms_ref)[0], smiles=reader.entry(refcode).molecule.heaviest_component.smiles)
    stages = stages_for(energy)
    units = UNITS[energy]

    rows, crystals, names = [], [], []
    # the generator's samples: CIFs, or extended XYZ when the generator labels its molecules (``bb_indices``);
    # every downstream file is named <stem>.cif
    paths = sorted(glob.glob(os.path.join(sample_dir, 'cifs', '*.cif'))
                   + glob.glob(os.path.join(sample_dir, 'cifs', '*.extxyz')))[:n]
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0] + '.cif'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            atoms = read(path)
        crystal, residual = p1_crystal(atoms, ref, identifier=stem)
        if crystal is None:
            rows.append({'file': stem, 'built': False, 'note': residual})
            continue
        crystals.append(crystal)
        names.append(stem)
        rows.append({'file': stem, 'built': True, 'build_residual': residual})
    zps = sorted({int(c.z_prime) for c in crystals})
    assert len(zps) == 1, 'one Z per run'
    print(f'{len(crystals)} of {len(rows)} samples built as P1 crystals with Z\' = {zps[0]}; largest rebuild '
          f'error {max(r["build_residual"] for r in rows if r["built"]):.1e} A')

    out_dir = os.path.join(sample_dir, f'relaxed_{energy}')
    os.makedirs(out_dir, exist_ok=True)
    scratch = os.path.join(sample_dir, 'opt_intermediates.pt')
    t0 = time.time()
    # relaxed CHUNK rows at a time to bound host memory; every row takes every step (convergence_eps 0), so a
    # row's result does not depend on its chunk mates
    chunk = CHUNK * 2 if str(device).startswith('cuda') and zps[0] <= 4 else CHUNK  # GPU batches: 50, or 25 at Z' 8
    parts = [relax(collate_data_list(crystals[i:i + chunk], max_z_prime=zps[0]), stages, device=device, scratch=scratch)
             for i in range(0, len(crystals), chunk)]
    relaxed_list = [c for p in parts for c in p[0].batch_to_list()]
    e0 = torch.cat([p[1] for p in parts])
    e1 = torch.cat([p[2] for p in parts])
    traj = np.concatenate([p[3] for p in parts], axis=1)
    t_relax = time.time() - t0
    # yardsticks: every experimental form, fitted with the same reference, symmetrised and relaxed the same way
    forms = relax_experimental([refcode, *extra_forms], ref, stages, reader, out_dir, scratch)
    ex1 = torch.tensor([forms[refcode][2]])
    print(f'{energy}: relaxed {len(crystals)} P1 crystals in {t_relax:.0f} s on {device} ({len(traj)} steps each)')
    for rc, (sg, a, z) in forms.items():
        print(f'  experimental {rc} relaxed in sg {sg}: {a:.1f} -> {z:.1f} ({units})')

    from ase.io import write
    built = [r for r in rows if r['built']]
    for k, (r, c) in enumerate(zip(built, relaxed_list)):
        cell = to_atoms(c)
        write(os.path.join(out_dir, r['file']), cell, format='cif')
        res = ingest(cell, ref, symprecs=TIGHT, identifier=r['file'])
        r.update(e_start=float(e0[k]), e_end=float(e1[k]), e_vs_exp=float(e1[k] - ex1[0]),
                 settle=settle_step(traj[:, k]), sg=res.sg if res.ok else None,
                 symprec=res.symprec, residual=res.residual_max if res.ok else None,
                 note='' if res.ok else res.reason)
    with open(os.path.join(sample_dir, f'p1_relax_{energy}.csv'), 'w', newline='') as f:
        keys = ['file', 'built', 'build_residual', 'e_start', 'e_end', 'e_vs_exp', 'settle', 'sg', 'symprec',
                'residual', 'note']
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, '') for k in keys} for r in rows])

    named = [r for r in built if r['sg'] is not None]
    sgs = {}
    for r in named:
        sgs[r['sg']] = sgs.get(r['sg'], 0) + 1
    print(f'after relaxation: {len(named)} of {len(built)} are Z\' = 1 structures at tolerance <= {TIGHT[-1]} A; '
          f'space groups {dict(sorted(sgs.items(), key=lambda kv: -kv[1]))}')
    e = np.array([r['e_vs_exp'] for r in built])
    print(f'energy above the relaxed experimental form ({units}): median {np.median(e):.1f}, '
          f'lowest {e.min():.1f}; settle step median {int(np.median([r["settle"] for r in built]))}')
    for r in built:
        print(f"  {r['file']}: e {r['e_start']:.1f} -> {r['e_end']:.1f} ({r['e_vs_exp']:+.1f} vs exp), "
              f"settle {r['settle']}, sg {r['sg']} {r['note'][:60]}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 25,
         sys.argv[4] if len(sys.argv) > 4 else 'elj',
         tuple(f for f in sys.argv[5].split(',') if f) if len(sys.argv) > 5 else (),
         sys.argv[6] if len(sys.argv) > 6 else 'cpu')
