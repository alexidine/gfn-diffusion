"""Test molecules for the FD stress sweep and a permutation-aware point-symmetry probe.

``point_symmetry(pos, z)`` searches for the best non-identity proper symmetry (S X = X up to a permutation of like
atoms, S a rotation by more than 1 degree) and the best improper symmetry (mirror, inversion, S_n), as
``artifacts/fd_design_oct15/fd_compare_oct24/rotations/molecule_symmetry_probe.py`` does: candidate rotations (the
principal-axis 2-folds plus random rotations), each refined by alternating optimal assignment of like atoms
(Hungarian on squared distances) and Kabsch fits. The returned RMSD (Angstrom, all atoms, heavy-atom-centred as the
builder centres) is the smallest found; a molecule is called C1 when both are above ``C1_RMSD``.

``pick_c1_molecule`` scans the qm9full test conditions (read only) for rigid (no rotatable bond), chiral (an RDKit
stereocentre), CHNO molecules with 8-9 heavy atoms and returns the first whose geometry passes the probe and whose
heavy atoms are not planar.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

CONDITIONS = 'D:/crystal_datasets/conditional/priors/qm9full_test_conditions.pt'
MIPCAS = 'D:/crystal_datasets/mipcas/MIPCAS_standardized.pt'
#: both symmetry RMSDs above this (Angstrom) and the molecule has no point symmetry for this sweep's purposes
C1_RMSD = 0.25
#: heavy-atom out-of-plane RMS above this (Angstrom): not planar
NONPLANAR_RMS = 0.3


def _centred(pos, z):
    X = np.asarray(pos, dtype=np.float64)
    Z = np.asarray(z)
    return X - X[Z > 1].mean(0), Z


def _matched(X, Y, Z):
    """RMSD of Y against X after the optimal type-preserving assignment; perm[i] = the atom of X that Y_i matches."""
    tot, perm = 0.0, np.arange(len(Z))
    for el in np.unique(Z):
        idx = np.where(Z == el)[0]
        C = ((Y[idx, None, :] - X[None, idx, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(C)
        perm[idx[r]] = idx[c]
        tot += C[r, c].sum()
    return float(np.sqrt(tot / len(Z))), perm


def point_symmetry(pos, z, n_random: int = 3000, refine: int = 6, seed: int = 0) -> dict:
    """Best proper (angle > 1 deg) and improper symmetry RMSDs (Angstrom) and the operations found."""
    X, Z = _centred(pos, z)
    cands = list(Rotation.random(n_random, random_state=seed).as_matrix())
    H = X[Z > 1]
    I = (H ** 2).sum() * np.eye(3) - H.T @ H
    _, evec = np.linalg.eigh(I)
    for ax in evec.T:
        cands.append(Rotation.from_rotvec(np.pi * ax).as_matrix())
    out = {}
    for improper in (False, True):
        M = np.diag([-1.0, 1, 1]) if improper else np.eye(3)
        best = (np.inf, None, None)
        for Rm in cands:
            S = Rm @ M
            rm, perm = _matched(X, X @ S.T, Z)
            for _ in range(refine):  # Kabsch of (X M) onto X[perm]: S = Rk M
                A, B = X @ M.T, X[perm]
                U, _, Vt = np.linalg.svd(A.T @ B)
                d = np.sign(np.linalg.det(Vt.T @ U.T))
                Rk = Vt.T @ np.diag([1, 1, d]) @ U.T
                S = Rk @ M
                rm, perm = _matched(X, X @ S.T, Z)
            if not improper:
                ang = np.degrees(np.arccos(np.clip((np.trace(S) - 1) / 2, -1, 1)))
                if ang < 1.0:
                    continue
            if rm < best[0]:
                best = (rm, S, perm)
        key = 'improper' if improper else 'proper'
        out[f'{key}_rmsd'] = float(best[0])
        out[f'{key}_op'] = None if best[1] is None else np.round(best[1], 4).tolist()
        out[f'{key}_perm_identity'] = None if best[2] is None else bool((best[2] == np.arange(len(Z))).all())
    u, s, vt = np.linalg.svd(H - H.mean(0))
    dev = (H - H.mean(0)) @ vt[2]
    out['heavy_out_of_plane_rms'] = float(np.sqrt((dev ** 2).mean()))
    out['heavy_out_of_plane_max'] = float(np.abs(dev).max())
    out['n_atoms'] = int(len(Z))
    out['n_heavy'] = int((Z > 1).sum())
    out['c1'] = bool(out['proper_rmsd'] > C1_RMSD and out['improper_rmsd'] > C1_RMSD)
    out['planar'] = bool(out['heavy_out_of_plane_rms'] < NONPLANAR_RMS)
    return out


def load_conditions():
    d = torch.load(CONDITIONS, weights_only=False)
    return d['prior']


def candidates(prior, max_n: int = 40):
    """Indices and smiles of rigid chiral CHNO rows with 8-9 heavy atoms (RDKit, no stereo in the smiles)."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    out = []
    for i, s in enumerate(prior.smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        els = {a.GetSymbol() for a in m.GetAtoms()}
        if not els <= {'C', 'N', 'O'} or not (els & {'N', 'O'}):
            continue
        if m.GetNumHeavyAtoms() < 8 or rdMolDescriptors.CalcNumRotatableBonds(m) != 0:
            continue
        cc = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
        if len(cc) < 1:
            continue
        out.append((i, s, m.GetNumHeavyAtoms(), len(cc)))
        if len(out) >= max_n:
            break
    return out


def pick_c1_molecule(prior=None, max_tries: int = 40, prefer=None, log=print):
    """(MolCrystalData row, record dict) of the first candidate that is C1 and not planar (bounded by max_tries);
    a candidate whose smiles is ``prefer`` is tried first."""
    prior = load_conditions() if prior is None else prior
    tried = []
    cands = candidates(prior, 10 ** 9 if prefer else max_tries)
    if prefer:
        cands = [c for c in cands if c[1] == prefer] + [c for c in cands if c[1] != prefer][:max_tries]
    for i, s, nh, ncc in cands:
        mol = prior.get_example(i)
        sym = point_symmetry(mol.pos, mol.z)
        rec = dict(index=i, smiles=s, heavy=nh, rdkit_stereocentres=ncc, **sym)
        tried.append(rec)
        log(f'  candidate {i} {s}: proper {sym["proper_rmsd"]:.3f} A, improper {sym["improper_rmsd"]:.3f} A, '
            f'heavy out-of-plane rms {sym["heavy_out_of_plane_rms"]:.3f} A')
        if sym['c1'] and not sym['planar']:
            return mol, dict(chosen=rec, tried=tried)
    raise RuntimeError(f'no C1 non-planar molecule among {len(tried)} candidates')


def load_mipcas():
    m = torch.load(MIPCAS, weights_only=False)
    return m[0] if isinstance(m, list) else m
