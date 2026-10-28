"""Intramolecular energies and forces for pre-training the intra trunk (pretrain_intra_trunk.py).

    python -u build_intra_energy_set.py --conditions <train conditions.pt> --test-conditions <held-out conditions.pt>
                                        --out <set.pt> [--workers N] [--max-molecules N]

THE SET. For every molecule of the two crystal condition files (read by SMILES): up to
--conformers conformers from RDKit's ETKDGv3, each relaxed under MMFF94, and for each relaxed
conformer the minimum itself and one copy per --noise value with every atom displaced by
Gaussian noise of that width (Angstrom). Each geometry carries MMFF94's energy (kcal/mol) and
force (-dE/dr, kcal/mol/Angstrom), RDKit's own. Nothing is clipped, shifted or scaled here:
references and scales are the fitting script's business and are stored with its checkpoint.

TWO THINGS ARE LEFT OUT, and counted. (1) A geometry whose force disagrees with its own energy:
RDKit's MMFF94 gradient is singular at a few arrangements (a first build held twelve "relaxed
minima" with forces of 1e5 to 1e6 kcal/mol/A at an unremarkable energy), so every force is
checked against a central difference of the energy along its own direction. (2) A relaxed
conformer more than --energy-window above its molecule's lowest: an embedding that relaxed
into a strained trap, not a conformer.

WHY THESE GEOMETRIES. The convention of machine-learned potentials: a model of the total
energy as a sum of per-atom terms, fitted to energies and Cartesian forces on geometries
spread around minima ("rattled" structures). Widths of 0.02 to 0.05 A are thermal for a bond;
0.1 and 0.2 A reach the strained geometries a sampler passes through. Several conformers per
molecule put torsions and non-bonded contacts into the energy differences, which rattling
alone would leave to the stiff terms.

THE SPLIT is the crystal work's: a molecule of --test-conditions is `heldout`, so a trunk
fitted here has not seen the molecules the crystal sampler is tested on.

Atom order is RDKit's (the SMILES with hydrogens added). The trunk reads elements and
positions only, so no order has to match the crystal files'.

Layout of the file (flat tensors, one molecule after another):
    smiles [M] list, heldout [M] bool, z [sum n] and mol_ptr [M + 1] over it,
    geom_mol [G] molecule of each geometry, geom_ptr [G + 1] over pos / force [sum over geometries of n, 3],
    energy [G], noise [G] (0 = a relaxed minimum), conformer [G] index within its molecule.
"""
import argparse
import hashlib
import time
from multiprocessing import Pool

import numpy as np
import torch

MMFF_VARIANT = 'MMFF94'


def _seed(smiles: str) -> int:
    return int(hashlib.sha1(smiles.encode()).hexdigest()[:7], 16)


#: displacement (A, norm over all atoms) of the central difference that checks a force
FD_STEP = 1e-4


def force_is_consistent(ff, x, force, rel=0.05, floor=0.5):
    """Does the energy fall along the force at the rate the force's size says (kcal/mol/A)?"""
    size = float(np.sqrt((force.astype(np.float64) ** 2).sum()))
    if size == 0.0:
        return True
    u = force.astype(np.float64) / size
    up = ff.CalcEnergy((x + FD_STEP * u).ravel().tolist())
    down = ff.CalcEnergy((x - FD_STEP * u).ravel().tolist())
    return abs((down - up) / (2 * FD_STEP) - size) <= rel * size + floor


def label_molecule(job):
    """One molecule -> (smiles, z [n], rows, dropped) or (smiles, reason); a row is
    (conformer, noise, pos [n, 3], energy, force [n, 3]) and `dropped` counts what was left out."""
    smiles, conformers, noise, max_iters, window = job
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, 'unreadable SMILES'
    mol = Chem.AddHs(mol)
    seed = _seed(smiles)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    cids = list(AllChem.EmbedMultipleConfs(mol, conformers, params))
    if not cids:
        # strained cages often defeat the distance-geometry start; random starting coordinates get most of them
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, conformers, params))
    if not cids:
        return smiles, 'no conformer embedded'
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        return smiles, 'no MMFF94 parameters'
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=MMFF_VARIANT)
    rng = np.random.default_rng(seed)
    n = mol.GetNumAtoms()
    rows, floors = [], {}
    dropped = {'unrelaxed conformer': 0, 'inconsistent force': 0, 'strained conformer': 0}
    for k, cid in enumerate(cids):
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        ff.Initialize()
        if ff.Minimize(maxIts=max_iters) != 0:      # not converged: a second, longer attempt, then give the conformer up
            if ff.Minimize(maxIts=4 * max_iters) != 0:
                dropped['unrelaxed conformer'] += 1
                continue
        x0 = np.asarray(ff.Positions(), dtype=np.float64).reshape(n, 3)
        floors[k] = ff.CalcEnergy(x0.ravel().tolist())
        for sigma in (0.0, *noise):
            x = x0 if sigma == 0.0 else x0 + sigma * rng.standard_normal((n, 3))
            flat = x.ravel().tolist()
            energy = ff.CalcEnergy(flat)
            force = -np.asarray(ff.CalcGrad(flat), dtype=np.float64).reshape(n, 3)
            if not (np.isfinite(energy) and np.isfinite(force).all()) or not force_is_consistent(ff, x, force):
                dropped['inconsistent force'] += 1
                continue
            rows.append((k, sigma, x.astype(np.float32), float(energy), force.astype(np.float32)))
    if floors:
        lowest = min(floors.values())
        strained = {k for k, e in floors.items() if e - lowest > window}
        dropped['strained conformer'] = len(strained)
        rows = [r for r in rows if r[0] not in strained]
    if not rows:
        return smiles, 'no conformer relaxed'
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return smiles, z, rows, dropped


def read_smiles(path):
    blob = torch.load(path, weights_only=False, map_location='cpu')
    batch = blob['prior'] if isinstance(blob, dict) and 'prior' in blob else blob
    return sorted(set(batch.smiles))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', default=r'D:/crystal_datasets/conditional/priors/qm9full_conditions.pt')
    ap.add_argument('--test-conditions', default=r'D:/crystal_datasets/conditional/priors/qm9full_test_conditions.pt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--conformers', type=int, default=2, help='conformers embedded per molecule (pruned at 0.5 A RMSD)')
    ap.add_argument('--noise', default='0.02,0.05,0.1,0.2', help='displacement widths (A), one rattled copy each')
    ap.add_argument('--max-iters', type=int, default=500, help='MMFF94 relaxation steps before a second, longer attempt')
    ap.add_argument('--energy-window', type=float, default=50.0,
                    help="a relaxed conformer this far (kcal/mol) above its molecule's lowest is left out")
    ap.add_argument('--max-molecules', type=int, default=0, help='first N molecules of each file only (0 = all)')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(argv)
    noise = tuple(float(v) for v in a.noise.split(','))

    t0 = time.time()
    held = read_smiles(a.test_conditions)
    train = read_smiles(a.conditions)
    both = sorted(set(held) & set(train))
    if both:
        raise SystemExit(f'{len(both)} molecules are in both files, e.g. {both[:3]}')
    if a.max_molecules:
        held, train = held[:a.max_molecules], train[:a.max_molecules]
    print(f"[intra-set] {len(train)} training and {len(held)} held-out molecules read in {time.time() - t0:.0f} s; "
          f"{a.conformers} conformer(s), minimum + noise {noise} A, {MMFF_VARIANT}, {a.workers} workers", flush=True)

    jobs = [(s, a.conformers, noise, a.max_iters, a.energy_window) for s in train + held]
    is_held = {s: True for s in held}
    t0 = time.time()
    with Pool(a.workers) as pool:
        results = pool.map(label_molecule, jobs, chunksize=256)
    failed = [(r[0], r[1]) for r in results if len(r) == 2]
    done = [r[:3] for r in results if len(r) == 4]
    reasons, dropped = {}, {}
    for _, why in failed:
        reasons[why] = reasons.get(why, 0) + 1
    for r in results:
        if len(r) == 4:
            for what, count in r[3].items():
                dropped[what] = dropped.get(what, 0) + count
    print(f"[intra-set] labelled {len(done)} molecules in {time.time() - t0:.0f} s; molecules left out {len(failed)}: "
          f"{reasons}; within the kept molecules, left out: {dropped} (conformers, and single geometries for the "
          f"force check)", flush=True)

    smiles, heldout, z, mol_n, geom_mol, geom_n, pos, force, energy, sigma, conf = [], [], [], [], [], [], [], [], [], [], []
    for m, (smi, zm, rows) in enumerate(done):
        smiles.append(smi)
        heldout.append(bool(is_held.get(smi, False)))
        z.append(zm)
        mol_n.append(len(zm))
        for k, s, x, e, f in rows:
            geom_mol.append(m)
            geom_n.append(len(zm))
            pos.append(x)
            force.append(f)
            energy.append(e)
            sigma.append(s)
            conf.append(k)

    def ptr(counts):
        return torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(counts, dtype=torch.long).cumsum(0)))

    out = {'smiles': smiles, 'heldout': torch.tensor(heldout), 'z': torch.from_numpy(np.concatenate(z)),
           'mol_ptr': ptr(mol_n), 'geom_mol': torch.tensor(geom_mol), 'geom_ptr': ptr(geom_n),
           'pos': torch.from_numpy(np.concatenate(pos)), 'force': torch.from_numpy(np.concatenate(force)),
           'energy': torch.tensor(energy, dtype=torch.float64), 'noise': torch.tensor(sigma, dtype=torch.float32),
           'conformer': torch.tensor(conf), 'left_out': failed, 'dropped': dropped,
           'build': {'conditions': a.conditions, 'test_conditions': a.test_conditions, 'conformers': a.conformers,
                     'noise': noise, 'max_iters': a.max_iters, 'energy_window': a.energy_window,
                     'force_field': MMFF_VARIANT,
                     'units': 'kcal/mol, Angstrom, kcal/mol/Angstrom'}}
    torch.save(out, a.out)
    g = len(energy)
    fmag = out['force'].square().sum(-1).sqrt()
    print(f"[intra-set] {len(smiles)} molecules ({int(out['heldout'].sum())} held out), {g} geometries, "
          f"{out['pos'].shape[0]} atom rows -> {a.out}", flush=True)
    print("[intra-set] noise (A) | geometries | median energy above the molecule's lowest (kcal/mol) | "
          "median largest force on an atom (kcal/mol/A)", flush=True)
    lowest = torch.full((len(smiles),), float('inf'), dtype=torch.float64).scatter_reduce(
        0, out['geom_mol'], out['energy'], reduce='amin')
    excess = out['energy'] - lowest[out['geom_mol']]
    fmax = torch.zeros(g).scatter_reduce(0, torch.repeat_interleave(torch.arange(g), torch.tensor(geom_n)), fmag,
                                         reduce='amax')
    for s in (0.0, *noise):
        sel = out['noise'] == s
        print(f"[intra-set] {s:g} | {int(sel.sum())} | {float(excess[sel].median()):.2f} | "
              f"{float(fmax[sel].median()):.2f}", flush=True)


if __name__ == '__main__':
    main()
