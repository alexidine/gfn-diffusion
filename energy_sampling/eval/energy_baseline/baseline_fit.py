"""How much of a molecule's raw MLIP crystal energy a fixed linear baseline removes.

The question it answers: if the trainer scores the RAW crystal energy per molecule (no gas leg), the per-condition
log Z carries the molecule's whole energy level. A fixed baseline b(c) = sum of per-element (or per-bond-type)
constants, frozen with the model, takes that level out of the learned Z(c) head. This measures what is left for the
head to learn, per molecule, against the spread of the lattice energies themselves.

Crystals: the same 2,000 random CSD Z'=1 organic crystals as eval/exp6_recall/csd_parity.py (same pick and seed),
single points at the experimental geometry.

Per crystal, eV per molecule:
  uma_crys   UMA-s 1.1 'omc' periodic energy / (sym_mult * z_prime)
  uma_gas    UMA-s 1.1 'omc' isolated molecule, mean of the molecule and its mirror image
  mace_crys  MACE-OFF24 (medium) periodic energy / (sym_mult * z_prime)
  mace_gas   MACE-OFF24 isolated molecule
MACE runs in float32 (the MXtalTools data path is float32): its energies sit near -1e4 eV per molecule (isolated-atom
zero), so they carry a few-hundredths-of-a-kJ/mol representation floor; UMA's (references near -5 eV/atom) less.
Composition from z; bond-type counts from the row's SMILES via RDKit with explicit H (bond = sorted element pair +
bond order); a row whose SMILES does not reproduce its element counts gets no bond counts and is left out of the
bond-type fits.

Baselines (each frozen per element or bond type; none reads a search minimum):
  uma_refs   UMA-s 1.1's own omc_energy ElementReferences from the checkpoint (UMA rows only; no fit)
  elem       least squares on element counts, fitted to the GAS energies
  bond       least squares on element + bond-type counts (types in >= MIN_TYPE_ROWS molecules), fitted to the GAS
             energies
Fitted baselines are scored out of fold (K_FOLDS-fold cross-validation), so a residual is never a training residual.

usage: python -m eval.energy_baseline.baseline_fit OUT_DIR [--n 2000] [--seed 0] [--max_atoms 60]
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

from eval.exp6_recall.csd_parity import pick, strip
from eval.exp6_recall.make_configs import SITES
from eval.exp6_recall.mirror_test import cap_gpu

EV_TO_KJMOL = 96.485
T_KJMOL = 2.5  # the trainer's temperature: 1 nat = 2.5 kJ/mol
K_FOLDS = 5
MIN_TYPE_ROWS = 10
SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}


def score(rows, keys, pred, chunk):
    """{key: [n] float64} for analyze(computes=keys); NaN where a chunk fails even one row at a time."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    out = {k: torch.full((len(rows),), float('nan'), dtype=torch.float64) for k in keys}
    kw = dict(cutoff=10, supercell_size=10, std_orientation=True, assign_outputs=False, predictor=pred)

    def go(lo, hi):
        b = collate_data_list(rows[lo:hi]).to('cuda')
        with torch.no_grad():
            res = b.analyze(computes=list(keys), **kw)
        for k in keys:
            out[k][lo:hi] = res[k].reshape(-1).double().cpu()

    for lo in range(0, len(rows), chunk):
        hi = min(lo + chunk, len(rows))
        try:
            go(lo, hi)
        except (RuntimeError, ValueError):
            torch.cuda.empty_cache()
            for i in range(lo, hi):
                try:
                    go(i, i + 1)
                except (RuntimeError, ValueError) as e1:
                    torch.cuda.empty_cache()
                    print(f'{keys} {rows[i].identifier}: {type(e1).__name__}: {str(e1)[:120]}', flush=True)
        if lo % (chunk * 50) == 0:
            print(f'{keys}: {hi}/{len(rows)}', flush=True)
    return out


def mirrored(r):
    m = strip(r)
    m.pos = m.pos * torch.tensor([-1.0, 1.0, 1.0], dtype=m.pos.dtype)
    return m


def bond_counts(r):
    """Counter of bond types ('C1.5C', 'C1H', ...) from the row's SMILES with explicit H, or None when the SMILES does
    not reproduce the row's element counts."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(r.smiles))
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    want = Counter(int(z) for z in r.z.tolist())
    got = Counter(a.GetAtomicNum() for a in mol.GetAtoms())
    if want != got:
        return None
    c = Counter()
    for b in mol.GetBonds():
        a1, a2 = sorted((b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()))
        c[f'{a1}{b.GetBondTypeAsDouble():g}{a2}'] += 1
    return c


def uma_refs():
    ck = torch.load(SITES['local']['uma'], map_location='cpu', weights_only=False)
    from omegaconf import OmegaConf
    for t in ck.tasks_config:
        d = OmegaConf.to_container(t, resolve=False) if not isinstance(t, dict) else t
        if d.get('name') == 'omc_energy':
            return np.array(d['element_references']['element_references']['_args_'][0], dtype=np.float64)
    raise SystemExit('no omc_energy element references in the UMA checkpoint')


def oof_residual(X, y, ok):
    """Out-of-fold residual y - X w (least squares, minimum-norm) over rows `ok`; NaN elsewhere."""
    res = np.full(len(y), np.nan)
    idx = np.flatnonzero(ok)
    folds = np.array_split(np.random.default_rng(0).permutation(idx), K_FOLDS)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        w, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        res[f] = y[f] - X[f] @ w
    return res


def spread(v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, sd=float('nan'), p01=float('nan'), p99=float('nan'))
    q = np.percentile(v, [1, 99])
    return dict(n=int(v.size), sd=float(v.std()), p01=float(q[0]), p99=float(q[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_atoms', type=int, default=60)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, 'baseline_fit.json')):
        raise SystemExit(f'{a.out_dir} already holds a result; use a new directory')
    os.makedirs(a.out_dir, exist_ok=True)
    cap_gpu()
    rows, n_ok = pick(a.n, a.seed, a.max_atoms)
    print(f'{len(rows)} crystals drawn from {n_ok} eligible', flush=True)
    n = len(rows)
    per_mol = np.array([float(r.sym_mult) * float(r.z_prime.reshape(-1)[0]) for r in rows])

    from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    upred = init_uma_crystal_predictor(SITES['local']['uma'], device='cuda', activation_checkpointing=False, tf32=False)
    u = score([strip(r) for r in rows], ('uma_pot', 'uma_gas_pot'), upred, chunk=4)
    u_m = score([mirrored(r) for r in rows], ('uma_gas_pot',), upred, chunk=8)
    del upred
    torch.cuda.empty_cache()
    mpred = load_mace_model(SITES['local']['mace'], device='cuda', dtype=torch.float32)
    m = score([strip(r) for r in rows], ('mace_pot', 'mace_gas_pot'), mpred, chunk=8)
    del mpred

    E = dict(uma_crys=u['uma_pot'].numpy() / per_mol,
             uma_gas=0.5 * (u['uma_gas_pot'].numpy() + u_m['uma_gas_pot'].numpy()),
             mace_crys=m['mace_pot'].numpy() / per_mol,
             mace_gas=m['mace_gas_pot'].numpy())

    zs = sorted({int(z) for r in rows for z in r.z.tolist()})
    Xe = np.array([[int((r.z == z).sum()) for z in zs] for r in rows], dtype=np.float64)
    bcs = [bond_counts(r) for r in rows]
    has_b = np.array([c is not None for c in bcs])
    type_rows = Counter(t for c in bcs if c is not None for t in c)
    types = sorted(t for t, k in type_rows.items() if k >= MIN_TYPE_ROWS)
    Xb = np.array([[(c or {}).get(t, 0) for t in types] for c in bcs], dtype=np.float64)
    Xeb = np.concatenate([Xe, Xb], axis=1)
    refs = uma_refs()
    b_uma = np.array([sum(refs[int(z)] for z in r.z.tolist()) for r in rows])

    out = dict(identifier=[r.identifier for r in rows], num_atoms=[int(r.num_atoms) for r in rows],
               elements=zs, bond_types=types, has_bonds=has_b.tolist(), n_eligible=n_ok,
               uma_refs={SYM.get(z, str(z)): float(refs[z]) for z in zs},
               **{k: v.tolist() for k, v in E.items()})
    table = {}
    for model in ('uma', 'mace'):
        crys, gas = E[f'{model}_crys'], E[f'{model}_gas']
        ok = np.isfinite(crys) & np.isfinite(gas)
        latt = (crys - gas) * EV_TO_KJMOL
        ok &= latt < 0  # a positive lattice energy = a broken structure, as in csd_parity
        rows_ = {'raw crystal energy (no baseline)': crys * EV_TO_KJMOL,
                 'lattice energy (crystal - own gas leg)': latt}
        if model == 'uma':
            rows_['raw minus UMA omc references'] = (crys - b_uma) * EV_TO_KJMOL
        for name, X, sel in (('elem', Xe, ok), ('bond', Xeb, ok & has_b)):
            r_gas = oof_residual(X, gas, sel)  # baseline fitted to gas energies, scored out of fold
            # the crystal residual uses the same out-of-fold baseline: crys - (gas - r_gas)
            rows_[f'raw minus {name}-type baseline (fit on gas)'] = (crys - (gas - r_gas)) * EV_TO_KJMOL
            rows_[f'gas minus {name}-type baseline (fit residual)'] = r_gas * EV_TO_KJMOL
        table[model] = {k: spread(np.where(ok, v, np.nan)) for k, v in rows_.items()}
    out['table'] = table
    with open(os.path.join(a.out_dir, 'baseline_fit.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'per-molecule spread across crystals, kJ/mol (1 nat = {T_KJMOL} kJ/mol); out-of-fold for fitted baselines')
    for model, t in table.items():
        print(f'--- {model}')
        for k, s in t.items():
            print(f'  {k:52s} n {s["n"]:5d}  sd {s["sd"]:10.1f}  p1..p99 {s["p01"]:10.1f} .. {s["p99"]:10.1f}'
                  f'  sd {s["sd"] / T_KJMOL:8.1f} nats')
    print(f'bond types used: {len(types)}; rows with bond counts: {int(has_b.sum())}/{n}')


if __name__ == '__main__':
    main()
