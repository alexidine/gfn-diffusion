"""Is MACE-CSP-25.2 + D3's lattice energy as predictable from the molecular graph as UMA-s's?

The learned log Z(c) needs an energy level that varies smoothly between molecules. A model's gas leg adds a per-molecule
error; if that error is erratic, a graph-based predictor cannot learn it. MACE-CSP-25.2 was trained on crystals only,
so its isolated-molecule energy is out of domain, as UMA-s's 'omc' gas leg is.

Crystals: the 2,000 random CSD Z' = 1 crystals of eval/energy_baseline/baseline_fit.py (same pick and seed, experimental
geometry, single points), restricted to MACE-CSP's elements (H, C, N, O, F). UMA-s 1.1 and MACE-OFF24 energies are read
from baseline_fit.json; MACE-CSP-25.2 large + D3(BJ, PBE) is scored here.

Readout, per model, on the rows every model scores with a negative lattice energy: out-of-fold prediction of the
lattice energy (kJ/mol per molecule) by gradient-boosted trees on Morgan count fingerprints and simple descriptors,
five folds grouped by molecule. R2, RMSE, MAE; RMSE split by whether the molecule carries a CH3.

Criterion, fixed before the run, on the bootstrap 95% interval of RMSE(MACE-CSP) / RMSE(UMA-s) over rows:
    pass          upper end <= 1.10
    fail          lower end >  1.10
    inconclusive  otherwise

usage: python -m eval.energy_baseline.smoothness OUT_DIR [--baseline BASELINE_JSON] [--limit N]
"""
import argparse
import json
import os

import numpy as np
import torch

from eval.energy_baseline.baseline_fit import score
from eval.exp6_recall.csd_parity import pick, strip
from eval.exp6_recall.make_configs import SITES
from eval.exp6_recall.mirror_test import cap_gpu

BASELINE = 'D:/crystal_datasets/energy_baseline_nov19/baseline_fit.json'
EV_TO_KJMOL = 96.485
ELEMENTS = {1, 6, 7, 8, 9}
BLOCK = 200
RATIO_LIMIT = 1.10


def score_mace_csp(rows, out):
    """{crys, gas}: [n] eV per molecule, saved after every block so that a rerun resumes."""
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    path = os.path.join(out, 'mace_csp_scores.npz')
    n = len(rows)
    if os.path.exists(path):
        z = np.load(path)
        crys, gas, done = z['crys'], z['gas'], int(z['done'])
        if len(crys) != n:
            raise SystemExit(f'{path} holds {len(crys)} rows, expected {n}')
    else:
        crys, gas, done = np.full(n, np.nan), np.full(n, np.nan), 0
    if done < n:
        pred = load_mace_model(SITES['local']['mace_csp'], 'cuda', torch.float32, dispersion=dict(xc='pbe', damping='bj'))
    while done < n:
        hi = min(done + BLOCK, n)
        block = rows[done:hi]
        res = score([strip(r) for r in block], ('mace_pot', 'mace_gas_pot'), pred, chunk=2)
        per_mol = np.array([float(r.sym_mult) * float(r.z_prime.reshape(-1)[0]) for r in block])
        crys[done:hi] = res['mace_pot'].numpy() / per_mol
        gas[done:hi] = res['mace_gas_pot'].numpy()
        done = hi
        np.savez(path, crys=crys, gas=gas, done=done)
        print(f'MACE-CSP scored {done}/{n}', flush=True)
    return crys, gas


def graph_features(smiles):
    """[n, d] fingerprint + descriptor matrix, and per row: molecule key (no stereo), has a CH3; None rows are dropped
    by the caller through the mask."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    ch3 = Chem.MolFromSmarts('[CH3]')
    X, keys, has, ok = [], [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            ok.append(False)
            keys.append('')
            has.append(False)
            X.append(np.zeros(1024 + 13))
            continue
        h = Chem.AddHs(m)
        count = [sum(a.GetAtomicNum() == z for a in h.GetAtoms()) for z in (1, 6, 7, 8, 9)]
        d = [Descriptors.MolWt(m), m.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumRings(m),
             rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumRotatableBonds(m),
             rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcTPSA(m)] + count
        X.append(np.concatenate([gen.GetCountFingerprintAsNumPy(m).astype(float), np.array(d, dtype=float)]))
        keys.append(Chem.MolToSmiles(m, isomericSmiles=False))
        has.append(m.HasSubstructMatch(ch3))
        ok.append(True)
    return np.array(X), np.array(keys), np.array(has), np.array(ok)


def oof_predict(X, y, groups):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, groups):
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5,
                                              l2_regularization=1.0, early_stopping=False, random_state=0)
        pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
    return pred


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def report(lat, X, groups, has_ch3, out):
    """lat: {model: [n] kJ/mol}, all finite and negative on the rows passed."""
    res = {}
    err = {}
    for name, y in lat.items():
        p = oof_predict(X, y, groups)
        e = y - p
        err[name] = e
        res[name] = dict(n=len(y), sd_kJ=float(y.std()), r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
                         rmse_kJ=rmse(e), mae_kJ=float(np.abs(e).mean()),
                         rmse_ch3_kJ=rmse(e[has_ch3]) if has_ch3.any() else None,
                         rmse_no_ch3_kJ=rmse(e[~has_ch3]) if (~has_ch3).any() else None,
                         n_ch3=int(has_ch3.sum()))
    rng = np.random.default_rng(0)
    n = len(groups)
    ratios = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        ratios.append(rmse(err['mace_csp'][i]) / rmse(err['uma'][i]))
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    point = rmse(err['mace_csp']) / rmse(err['uma'])
    verdict = 'pass' if hi <= RATIO_LIMIT else ('fail' if lo > RATIO_LIMIT else 'inconclusive')
    summary = dict(models=res, rmse_ratio_mace_csp_over_uma=dict(point=point, ci95=[float(lo), float(hi)]),
                   limit=RATIO_LIMIT, verdict=verdict)
    with open(os.path.join(out, 'smoothness.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    print('\nOut-of-fold prediction of lattice energy from the molecular graph (five folds grouped by molecule), on '
          f'{n} CSD crystals every model scores with a negative lattice energy; kJ/mol per molecule.')
    print(f'{"model":10s} {"sd":>7s} {"R2":>6s} {"RMSE":>7s} {"MAE":>7s} {"RMSE CH3":>9s} {"RMSE no CH3":>12s}')
    for k, v in res.items():
        print(f'{k:10s} {v["sd_kJ"]:7.1f} {v["r2"]:6.3f} {v["rmse_kJ"]:7.2f} {v["mae_kJ"]:7.2f} '
              f'{v["rmse_ch3_kJ"]:9.2f} {v["rmse_no_ch3_kJ"]:12.2f}')
    print(f'CH3 rows: {int(has_ch3.sum())}/{n}')
    print(f'RMSE ratio MACE-CSP / UMA-s: {point:.3f} (95% bootstrap {lo:.3f}..{hi:.3f}); limit {RATIO_LIMIT}: {verdict}')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--baseline', default=BASELINE)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, 'smoothness.json')):
        raise SystemExit(f'{a.out_dir} already holds a result; use a new directory')
    os.makedirs(a.out_dir, exist_ok=True)
    cap_gpu()
    base = json.load(open(a.baseline))
    rows, _ = pick(len(base['identifier']), 0, 60)
    if [r.identifier for r in rows] != base['identifier']:
        raise SystemExit('the picked crystals do not match baseline_fit.json')
    idx = [i for i, r in enumerate(rows) if set(r.z.tolist()) <= ELEMENTS]
    if a.limit:
        idx = idx[:a.limit]
    print(f'{len(idx)} of {len(rows)} crystals contain only H, C, N, O, F', flush=True)
    sub = [rows[i] for i in idx]
    crys, gas = score_mace_csp(sub, a.out_dir)
    pick_b = lambda k: np.array(base[k], dtype=float)[idx]
    lat = dict(mace_csp=(crys - gas) * EV_TO_KJMOL,
               uma=(pick_b('uma_crys') - pick_b('uma_gas')) * EV_TO_KJMOL,
               mace_off=(pick_b('mace_crys') - pick_b('mace_gas')) * EV_TO_KJMOL)
    for k, v in lat.items():
        print(f'{k}: {int(np.isfinite(v).sum())} finite, {int((v >= 0).sum())} non-negative lattice energies', flush=True)
    X, keys, has_ch3, parsed = graph_features([r.smiles for r in sub])
    keep = parsed & np.all([np.isfinite(v) & (v < 0) for v in lat.values()], axis=0)
    np.savez(os.path.join(a.out_dir, 'lattice_energies.npz'), keep=keep, identifier=np.array([r.identifier for r in sub]),
             **lat)
    if int(keep.sum()) < 25:
        print(f'only {int(keep.sum())} usable rows: no report', flush=True)
        return
    X = X[keep]
    X = X[:, X.std(axis=0) > 0]
    report({k: v[keep] for k, v in lat.items()}, X, keys[keep], has_ch3[keep], a.out_dir)


if __name__ == '__main__':
    main()
