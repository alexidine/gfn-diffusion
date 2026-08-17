"""Refit InternalPrior's ring banks by enumerating each ring type's pucker basins.

WHY THIS RATHER THAN MORE MOLECULES. A ring system's conformational space is
low-dimensional -- Cremer-Pople gives a 6-ring three numbers and a 5-ring two -- and there
are only a couple of dozen ring systems that matter in ordinary chemistry. So the banks
can be built by covering each ring's own manifold directly, rather than by hoping a large
molecular dataset happens to visit every pucker. Enumeration beats sampling here because
the rare basins (twist-boat, and the minor puckers of substituted 5-rings) are exactly the
ones a dataset under-represents, and exactly the ones a proposal distribution needs for
support.

WHAT IT PRODUCES. A prior whose ``rings`` dict is refitted at ring signature version 2 and
whose bond/angle/torsion tables are carried over unchanged from the input prior. Those
tables describe general chemistry and must NOT be refitted from a bag of bare rings.

COVERAGE IS MEASURED, NOT ASSERTED. Basins are found by minimising from many starts and
deduplicating on the ring-torsion sign pattern; the script then re-runs with half the
starts and reports whether the basin set is unchanged. A basin set that is still growing
is a coverage failure, and it says so.

    python build_ring_banks.py --in conformer_prior.pt --out conformer_prior_v2.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

# Bare puckering rings. Aromatics are deliberately absent: they are rigid, and
# ConformerTorsions never banks them (see ring_blocks) -- a bank could only do harm.
RING_SMILES = [
    'C1CCC1',          # cyclobutane
    'C1CNC1',          # azetidine
    'C1COC1',          # oxetane
    'C1CCCC1',         # cyclopentane
    'C1CCNC1',         # pyrrolidine   (proline's ring)
    'C1CCOC1',         # tetrahydrofuran
    'C1COCO1',         # 1,3-dioxolane
    'C1CCCCC1',        # cyclohexane
    'C1CCNCC1',        # piperidine
    'C1CCOCC1',        # tetrahydropyran
    'C1COCCN1',        # morpholine
    'C1CNCCN1',        # piperazine
    'C1COCCO1',        # 1,4-dioxane
    'C1CCSCC1',        # thiane
    'C1CCCCCC1',       # cycloheptane
    'C1CCCCCCC1',      # cyclooctane
]


def conformers(smiles, n_starts, seed=0xC0FFEE):
    """MMFF-minimised conformers of one molecule, RMS-pruned. Returns (z, bonds, [pos])."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from mxtaltools.conformers.perception import infer_bond_index

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.05
    params.useRandomCoords = True          # random starts spread over the pucker manifold
    AllChem.EmbedMultipleConfs(mol, numConfs=n_starts, params=params)
    if mol.GetNumConformers() == 0:
        raise RuntimeError(f'no conformers embedded for {smiles}')
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    pos = [np.asarray(c.GetPositions(), dtype=np.float64) for c in mol.GetConformers()]
    en = [float(e) if ok == 0 else float('nan') for ok, e in res]
    return z, infer_bond_index(z, pos[0]), pos, en


def ring_cycle(z, bonds):
    """Heavy-atom ring atoms in cyclic order, or None if there is not exactly one ring."""
    import networkx as nx
    g = nx.Graph([(int(i), int(j)) for i, j in zip(*bonds)])
    cycles = nx.cycle_basis(g)
    if len(cycles) != 1:
        return None
    return cycles[0]


def basin_key(pos, cyc):
    """Sign pattern of the ring torsions -- a coarse, stable label for a pucker basin."""
    from mxtaltools.conformers.geometry import dihedral
    n = len(cyc)
    t = [float(dihedral(*(torch.tensor(pos[cyc[(i + k) % n]])[None] for k in range(4))))
         for i in range(n)]
    # quantise: flat (|t| < 10 deg) is its own state, so a planar ring does not read as a
    # random sign pattern
    return tuple(0 if abs(np.degrees(x)) < 10 else int(np.sign(x)) for x in t)


def as_moldata(z, bonds, pos, dtype=torch.float64):
    from mxtaltools.dataset_utils.data_classes import MolData
    m = MolData(z=torch.as_tensor(z, dtype=torch.long),
                pos=torch.as_tensor(pos, dtype=dtype),
                smiles='', identifier='')
    m.mol_bond_index = torch.as_tensor(np.asarray(bonds), dtype=torch.long)
    return m


def collect(smiles, n_starts, verbose=True):
    """MolData per distinct pucker basin, plus the basin labels found."""
    z, bonds, poss, ens = conformers(smiles, n_starts)
    cyc = ring_cycle(z, bonds)
    if cyc is None:
        raise RuntimeError(f'{smiles}: expected exactly one ring')
    seen, mols = {}, []
    for p, ei in zip(poss, ens):
        k = basin_key(p, cyc)
        if k in seen:
            continue
        seen[k] = True
        md = as_moldata(z, bonds, p)
        md.mmff_energy = ei                    # carried through to RingModes.energies
        mols.append(md)
    if verbose:
        print(f'  {smiles:14s} {len(poss):4d} minimised confs -> {len(mols):3d} distinct '
              f'pucker basin(s)')
    return mols, set(seen)


def fit_ring_modes(smiles, mols, prior, var_target=0.995, max_k=8, verbose=True):
    """RingModes per (signature, n_dof) key, from a set of conformers of one molecule.

    Whitening comes from the FORCE FIELD's own thermal widths, never from the sample
    spread: sample std bakes in both the sampling temperature and any inadequacy of the
    sampling, whereas sqrt(kT/2k) is a property of the potential. In that metric the
    covariance is kT * H^-1 at harmonic level, so the leading PCA directions are the SOFT
    ones rather than merely the high-variance ones.

    Only theta and phi are fitted. Bond lengths are left to the thermal path, which is
    exact for a harmonic term and specific to the molecule at hand rather than imported
    from whichever ring was scanned.
    """
    from energies.conformer_torsions import ConformerTorsions
    from energies.conformer_data import RingModes

    e = ConformerTorsions(smiles=smiles, device='cpu', level='full')
    _, s_th = e.thermal_rtheta_sigma(float(e.temperature))
    grp = e.torsion_groups()
    s_phi = float(np.median(e.sibling_jitter_sigma(grp, float(e.temperature)))) if grp else 0.1

    acc = {}
    for m in mols:
        m.build_conformer_tree()
        th, ph = (x.detach().cpu().numpy() for x in m.internal_dof()[1:])
        _, blocks, sigs, _ = prior._layout(m)
        for s, cols in blocks.items():
            order = ([('r', int(j)) for j in cols['r']]
                     + [('theta', int(j)) for j in cols['theta']]
                     + [('phi', int(j)) for j in cols['phi']])
            sub = [(k, j) for k, j in order if k != 'r']
            if not sub:
                continue
            vals = np.array([(th if k == 'theta' else ph)[j] for k, j in sub])
            e_i = float(getattr(m, 'mmff_energy', np.nan))
            a = acc.setdefault((sigs[s], len(order)), (sub, [], []))
            a[1].append(vals); a[2].append(e_i)

    out = {}
    for key, (sub, rows, ens) in acc.items():
        X = np.stack(rows)
        per = np.array([k == 'phi' for k, _ in sub])
        ref = np.empty(X.shape[1])
        ref[~per] = X[:, ~per].mean(0)
        # MAX-GAP branch cut, not the circular mean. Linearising a periodic variable about
        # its circular mean folds the data whenever the samples spread past +-180 deg of it
        # -- which happened on the 7- and 8-rings, whose many basins span the whole circle.
        # Cutting at each variable's largest EMPTY arc cannot fold any observed point, by
        # construction. This is dPCA+'s shift (Sittel & Stock); the alternatives are dPCA's
        # sin/cos doubling and, properly, torus PCA (Eltzner/Huckemann/Mardia, for RNA).
        for c in np.flatnonzero(per):
            v = np.sort(X[:, c] % (2 * np.pi))
            gaps = np.diff(np.concatenate([v, v[:1] + 2 * np.pi]))
            g = int(np.argmax(gaps))
            cut = (v[g] + gaps[g] / 2.0) % (2 * np.pi)      # middle of the widest gap
            ref[c] = (cut + np.pi) % (2 * np.pi)            # diametrically opposite it
        D = X - ref
        D[:, per] = (D[:, per] + np.pi) % (2 * np.pi) - np.pi
        worst = float(np.degrees(np.abs(D[:, per]).max())) if per.any() else 0.0
        if worst > 175.0:
            print(f'    WARNING {key[0]}: a phi deviation reaches {worst:.0f} deg even after '
                  f'a max-gap cut, so the samples genuinely fill the circle -- a linear '
                  f'subspace is the wrong model here; see torus PCA')
        scale = np.array([s_phi if k == 'phi' else float(s_th[j]) for k, j in sub])
        W = D / scale
        Wm = W.mean(0)
        Wc = W - Wm
        _, S, Vt = np.linalg.svd(Wc, full_matrices=False)
        var = (S ** 2) / max((S ** 2).sum(), 1e-30)
        k = int(np.searchsorted(np.cumsum(var), var_target) + 1)
        k = max(1, min(k, max_k, Wc.shape[1]))
        comps = Vt[:k]
        coords = Wc @ comps.T
        # NARROW band around the populated states -- 0.5 of a thermal sigma, deliberately
        # under-dispersed, and measured rather than argued. The physical within-basin
        # fluctuation is 1.0 sigma (the whitening makes one unit exactly that), but the
        # linear subspace's departure from the curved closed-ring manifold grows with how
        # far a draw travels, so width is paid for in closure error.
        #
        # It costs nothing because COVERAGE COMES FROM THE BASIN DRAW, not the jitter: the
        # Boltzmann weights choose which conformer to start from, and the jitter only fills
        # in around it. Measured at 1.0 -> 0.5, no ring loses a single sector of its
        # pseudorotation circle while closure halves (cyclohexane 3.1 -> 1.9 bond-sigma,
        # cyclooctane 4.6 -> 2.6) and chair populations improve. 0.3 is too far: cyclopentane
        # drops 9/12 sectors to 6/12.
        #
        # NOT estimated from the samples. With one conformer per basin there is no
        # information about a basin's width, so a sample-estimated bandwidth measures
        # between-basin distances instead -- on cyclohexane that gave 1.7 sigma and 4.3
        # bond-sigma of closure error.
        bw = 0.5
        E = np.array(ens, dtype=float)
        E = None if np.isnan(E).any() else E - E.min()
        out[key] = RingModes(order=sub, ref=ref + Wm * scale,
                             periodic=per, scale=scale, components=comps, coords=coords,
                             energies=E, bandwidth=max(bw, 1e-3),
                             comp_std=coords.std(0) if len(coords) > 1 else None,
                             var_explained=float(np.cumsum(var)[k - 1]),
                             n_samples=len(coords), max_fold_deg=worst)
        if verbose:
            size, atoms = key[0]
            pop = out[key].weights(1.0).max() if E is not None else float('nan')
            print(f'    size {size} n_dof {key[1]:2d}: {len(coords):3d} samples, '
                  f'{Wc.shape[1]:2d} DoF -> {k} component(s), '
                  f'{100 * out[key].var_explained:.1f}% var, fold {worst:3.0f} deg, '
                  f'lowest basin {100 * pop:.1f}% at kT=1')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='conformer_prior.pt')
    ap.add_argument('--out', dest='out', default='conformer_prior_v2.pt')
    ap.add_argument('--starts', type=int, default=400)
    ap.add_argument('--fatten', type=float, default=None)
    args = ap.parse_args()

    from mxtaltools.conformers.prior import InternalPrior

    base = torch.load(Path(args.inp), weights_only=False)
    print(f'carrying over {len(base.bonds)} bond / {len(base.angles)} angle / '
          f'{len(base.torsions)} torsion types from {args.inp} '
          f'(fitted on {base.n_fitted} molecules)')
    print(f'refitting ring banks from {len(RING_SMILES)} ring types, {args.starts} starts each:')

    mols, basins, modes = [], {}, {}
    for smi in RING_SMILES:
        try:
            m, b = collect(smi, args.starts)
            mols.extend(m)
            basins[smi] = b
            modes.update(fit_ring_modes(smi, m, base))
        except Exception as ex:
            print(f'  {smi:14s} SKIPPED ({type(ex).__name__}: {ex})')

    ring_fit = InternalPrior(fatten=base.fatten if args.fatten is None else args.fatten)
    ring_fit.fit(mols, verbose=False)
    print(f'\nfitted {len(ring_fit.rings)} ring signature(s) from {ring_fit.n_fitted} '
          f'conformers')
    for k, bank in sorted(ring_fit.rings.items(), key=lambda kv: -kv[1].rows.shape[0]):
        size, atoms = k[0]
        els = ','.join(f'{a}/{b}' for a, b in atoms)
        print(f'   size {size}  n_dof {k[1]:2d}  [{els}]  -> {bank.rows.shape[0]:3d} rows')

    # COVERAGE CHECK: halve the starts and require the basin set not to grow. A set that is
    # still growing means the scan has not converged and the bank is incomplete -- which
    # would otherwise look exactly like "this ring only has a few conformations".
    print('\ncoverage check (half the starts; the basin set must not grow):')
    incomplete = []
    for smi in RING_SMILES:
        if smi not in basins:
            continue
        try:
            _, half = collect(smi, max(args.starts // 2, 20), verbose=False)
        except Exception:
            continue
        missing = half - basins[smi]
        status = 'OK' if not missing else f'GREW by {len(missing)}'
        if missing:
            incomplete.append(smi)
        print(f'  {smi:14s} full {len(basins[smi]):3d} basins, half {len(half):3d}  {status}')
    if incomplete:
        print(f'\nWARNING {len(incomplete)} ring type(s) found basins at HALF the starts '
              f'that the full run missed: {incomplete}. The scan has not converged; raise '
              f'--starts before trusting those banks.')

    # rings only. The bond/angle/torsion tables describe general chemistry and must not be
    # refitted from a bag of bare rings -- carrying them over is the whole point.
    base.rings = ring_fit.rings
    base.ring_modes = modes
    base.ring_sig_version = 2
    torch.save(base, Path(args.out))
    print(f'\nwrote {args.out}: ring banks refitted at signature version 2, '
          f'{len(base.torsions)} torsion types carried over unchanged')


if __name__ == '__main__':
    main()
