"""Round trip for the P1 converter: CSD entry -> CIF -> ASE P1 cell -> ``p1_ingest.ingest`` -> MXtalTools crystal.

The reference is the cell's own first molecule, so the fit is exact and the residual measures only the
converter: splitting, matching, symmetry detection, the change of setting and the rebuild.

usage (CPU, licensed CSD Python API):
    python -m eval.day_bench.csd_roundtrip [REFCODE ...]
"""
import io
import sys
import time

import torch
from ase.io import read

from eval.day_bench.p1_ingest import ingest, make_reference, split_molecules

# Day-set structures, one or two per space group of the CSD entry (Day ESI Table S3)
DEFAULT = ['HURYUQ', 'UMIMIO', 'BCHXEN01', 'BAPFOP', 'FAFDIA', 'CAXPAT', 'CAPRYL', 'GOLHIB', 'IMAZOL03',
           'ACRDIN04', 'AWUWOE', 'DAZCOE', 'ACNAQU01', 'UZOVUB', 'AZALAO', 'WEXREY', 'MTANNL', 'ROPPUK',
           'CPFLUR', 'XULDUD', 'CAMWUI', 'INDDON', 'FACFAQ', 'BEVCEK', 'TRQUIN', 'CIMNUH', 'CUMJOJ', 'LECZOL',
           'IMAZOL22', 'APYFEB01', 'QQQCIS01']


def p1_cell(refcode, reader):
    entry = reader.entry(refcode)
    cif = entry.crystal.to_string('cif')
    return read(io.StringIO(cif), format='cif'), entry.crystal.spacegroup_symbol, entry.crystal.z_prime


def main(refcodes):
    from ccdc import io as ccdc_io
    torch.set_num_threads(2)
    reader = ccdc_io.EntryReader('CSD')
    print('Round trip of CSD structures through the P1 converter; reference = the cell\'s own first molecule.')
    print(f"{'refcode':10s} {'CSD sg':8s} {'CSD Zp':>6s} {'mols':>4s} {'mirr':>4s} {'fit rmsd (A)':>12s} "
          f"{'found sg':>8s} {'symprec (A)':>11s} {'residual (A)':>12s} {'subgroups':>9s} {'canon':>5s} "
          f"{'time (s)':>8s}  note")
    n_ok = 0
    for rc in refcodes:
        t0 = time.time()
        try:
            atoms, sg_sym, zp = p1_cell(rc, reader)
            ref = make_reference(split_molecules(atoms)[0])
            r = ingest(atoms, ref, identifier=rc)
        except Exception as e:  # reading or reference failures are reported per row
            print(f'{rc:10s} read/reference failed: {type(e).__name__}: {e}')
            continue
        n_ok += int(r.ok)
        sgs = '' if r.sg is None else str(r.sg)
        print(f'{rc:10s} {sg_sym:8s} {zp:6.2f} {r.n_molecules:4d} {r.n_mirrored:4d} {r.fit_rmsd_max:12.2e} '
              f'{sgs:>8s} {"" if r.symprec is None else r.symprec:>11} {r.residual_max:12.2e} '
              f'{r.n_subgroups:9d} {"yes" if r.canonical else "no":>5s} {time.time() - t0:8.2f}  {r.reason}')
    print(f'{n_ok} of {len(refcodes)} converted')


NOISE = (0.0, 0.05, 0.1, 0.2)  # Angstrom, standard deviation per Cartesian component


def noise_sweep(refcodes, seed=0):
    """Gaussian noise on every atom of the P1 cell, fitted with the clean cell's first molecule.

    A generator sample is a noisy, nearly symmetric cell; this asks whether the tolerance ladder still finds
    the clean cell's space group, and how far the symmetrised rebuild sits from the noisy atoms.
    """
    import numpy as np
    from ccdc import io as ccdc_io
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    reader = ccdc_io.EntryReader('CSD')
    print('Noisy P1 cells: per noise level (A), the space group found / symprec used (A) / largest residual (A); '
          '"=" marks the clean cell\'s group.')
    print(f"{'refcode':10s} " + ' '.join(f'{f"noise {s}":>26s}' for s in NOISE))
    hits = {s: 0 for s in NOISE}
    for rc in refcodes:
        atoms, _, _ = p1_cell(rc, reader)
        ref = make_reference(split_molecules(atoms)[0])
        cells, clean_sg = [], None
        for s in NOISE:
            noisy = atoms.copy()
            noisy.positions = atoms.positions + rng.normal(0.0, s, atoms.positions.shape)
            r = ingest(noisy, ref, identifier=rc)
            if s == 0.0:
                clean_sg = r.sg
            same = r.ok and r.sg == clean_sg
            hits[s] += int(same)
            cell = (f'{r.sg}{"=" if same else ""} / {r.symprec} / {r.residual_max:.2f}' if r.ok
                    else r.reason[:24])
            cells.append(f'{cell:>26s}')
        print(f'{rc:10s} ' + ' '.join(cells))
    print('same group as the clean cell: ' + ', '.join(f'noise {s} A: {hits[s]} of {len(refcodes)}' for s in NOISE))


RIGID = ((0.05, 1.0), (0.1, 2.0), (0.2, 5.0), (0.3, 8.0))  # (shift sd per component in A, rotation sd in deg)
JITTER = 0.02  # Angstrom, atomic noise on top of the rigid moves


def rigid_sweep(refcodes, seed=0):
    """Each molecule shifted and rotated as a rigid body about its centroid, plus a small atomic jitter.

    Closer to a generator's errors than independent atom noise: bonds stay intact, poses do not.
    """
    import numpy as np
    from ase import Atoms
    from ccdc import io as ccdc_io
    from scipy.spatial.transform import Rotation
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    reader = ccdc_io.EntryReader('CSD')
    print(f'Rigid-body noise on each molecule plus {JITTER} A atomic jitter: per level (shift sd A, rotation sd '
          'deg), the space group found / symprec used (A) / largest residual (A); "=" marks the clean group.')
    print(f"{'refcode':10s} " + ' '.join(f'{f"{t} A, {r} deg":>24s}' for t, r in RIGID))
    hits = {lv: 0 for lv in RIGID}
    for rc in refcodes:
        atoms, _, _ = p1_cell(rc, reader)
        mols = split_molecules(atoms)
        ref = make_reference(mols[0])
        clean = ingest(atoms, ref, identifier=rc)
        cells = []
        for t, rdeg in RIGID:
            pos = []
            for m in mols:
                c = m.pos.mean(0)
                rot = Rotation.from_rotvec(rng.normal(size=3) / np.sqrt(3) * np.deg2rad(rdeg))
                p = rot.apply(m.pos - c) + c + rng.normal(0.0, t, 3)
                pos.append(p + rng.normal(0.0, JITTER, p.shape))
            noisy = Atoms(numbers=np.concatenate([m.z for m in mols]), positions=np.concatenate(pos),
                          cell=atoms.cell, pbc=True)
            r = ingest(noisy, ref, identifier=rc)
            same = r.ok and r.sg == clean.sg
            hits[(t, rdeg)] += int(same)
            cell = (f'{r.sg}{"=" if same else ""} / {r.symprec} / {r.residual_max:.2f}' if r.ok else r.reason[:22])
            cells.append(f'{cell:>24s}')
        print(f'{rc:10s} ' + ' '.join(cells))
    print('same group as the clean cell: ' + ', '.join(f'{t} A / {r} deg: {hits[(t, r)]} of {len(refcodes)}'
                                                    for t, r in RIGID))


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '--noise':
        noise_sweep(args[1:] or DEFAULT)
    elif args and args[0] == '--rigid':
        rigid_sweep(args[1:] or DEFAULT)
    else:
        main(args or DEFAULT)
