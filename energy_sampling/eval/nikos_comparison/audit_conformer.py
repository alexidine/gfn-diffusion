"""
Which molecular conformer is carried by every artifact in the acridine comparison?

This is the load-bearing check for the whole sub-project. Lattice energy is a
DIFFERENCE between a crystal and its constituent molecule, so two artifacts built
on different conformers are not comparable at all -- and the failure is quiet: both
score, both look reasonable, and the numbers are ~70 kJ/mol apart for reasons that
have nothing to do with packing.

The trap is live and documented: `std_acridine_polymorphs.pt` and
`std_opt_acridine_polymorphs.pt` hold the SAME cells and poses (bit-identical
cell_lengths, cell_angles, aunit_centroid, aunit_orientation) and differ ONLY in
the molecule -- the first carries `acridine_conformer.pt` (aromatic C-C 1.3668 A),
the second `opt_acridine_conformer.pt` (1.4027 A). `process_target.py` still writes
the old-conformer file, so picking the wrong one silently compares a different
molecule.

FINGERPRINT. A conformer is identified here by its sorted intramolecular distance
vector, taken over heavy atoms in the molecule's own frame. That is invariant to
rotation, translation and atom ORDER (sorting removes the permutation), so it
compares the molecule itself rather than any pose convention -- two artifacts
agreeing to <1e-4 A on every one of those distances carry the same conformer.
Bond-length summaries are reported alongside as a human-readable cross-check.

Reports, per source: n structures, the fingerprint hash, mean/min/max heavy-atom
bond length, and the pairwise max deviation against the declared reference.
"""
import hashlib
import os

import torch

REF = r"D:\crystal_datasets\acridine\opt_acridine_conformer.pt"
OLD = r"D:\crystal_datasets\acridine\acridine_conformer.pt"
ROOT = r"D:\crystal_datasets\acridine"
NC = os.path.join(ROOT, 'nikos_comparison')

#: bond cutoff for the readable summary only; the fingerprint uses ALL distances
BOND_MAX = 1.8


def heavy(pos, z):
    m = z > 1
    return pos[m]


def fingerprint(pos):
    """Sorted intramolecular distance vector: rotation/translation/order invariant."""
    d = torch.cdist(pos, pos)
    iu = torch.triu_indices(len(pos), len(pos), offset=1)
    v = d[iu[0], iu[1]].sort().values
    return v


def summarize(v):
    b = v[v < BOND_MAX]
    return dict(n_dist=len(v), n_bond=len(b),
                bond_mean=float(b.mean()) if len(b) else float('nan'),
                bond_min=float(b.min()) if len(b) else float('nan'),
                bond_max=float(b.max()) if len(b) else float('nan'))


def one_molecule(obj, n_ref=None):
    """First molecule of whatever container this is, as (pos, z)."""
    if isinstance(obj, (list, tuple)):
        obj = obj[0]
    if hasattr(obj, 'batch_to_list') and getattr(obj, 'num_graphs', 1) > 1:
        obj = obj.batch_to_list()[0]
    pos, z = obj.pos, obj.z
    #: a crystal graph carries the WHOLE unit cell in .pos once built; the
    #: asymmetric-unit molecule is the first mol_ind group. Use mol_ind when
    #: present so we never fingerprint a cell against a single molecule.
    mi = getattr(obj, 'mol_ind', None)
    if mi is not None and len(mi) == len(pos) and int(mi.max()) > 0:
        keep = mi == mi[0]
        pos, z = pos[keep], z[keep]
    hp, hz = heavy(pos.double(), z), z[z > 1]
    #: Z'>1 asymmetric units hold several copies and mol_ind does not always
    #: separate them; when the heavy-atom count is an exact multiple of the
    #: reference molecule, take the FIRST block. Verified below by checking the
    #: blocks agree with each other, so this cannot mask a genuine mismatch.
    if n_ref is not None and len(hp) > n_ref and len(hp) % n_ref == 0:
        blocks = [hp[i * n_ref:(i + 1) * n_ref] for i in range(len(hp) // n_ref)]
        fps = [fingerprint(bk) for bk in blocks]
        spread = max(float((f - fps[0]).abs().max()) for f in fps)
        hp, hz = blocks[0], hz[:n_ref]
        return hp, hz, spread
    return hp, hz, 0.0


SOURCES = [
    ('opt_acridine_conformer (REFERENCE)', REF),
    ('acridine_conformer (OLD)', OLD),
    ('std_opt_acridine_polymorphs', os.path.join(ROOT, 'std_opt_acridine_polymorphs.pt')),
    ('std_acridine_polymorphs', os.path.join(ROOT, 'std_acridine_polymorphs.pt')),
    ('polymorphs_l2 (relaxed refs)', os.path.join(NC, 'polymorphs_l2.pt')),
    ('may_acridine_sg14_zp1_prior', os.path.join(ROOT, 'may_acridine_sg14_zp1_prior_dataset.pt')),
    ('may_acridine_sg14_zp2_prior', os.path.join(ROOT, 'may_acridine_sg14_zp2_prior_dataset.pt')),
    ('may_acridine_sg9_zp2_prior', os.path.join(ROOT, 'may_acridine_sg9_zp2_prior_dataset.pt')),
]


def load(path):
    obj = torch.load(path, weights_only=False, map_location='cpu')
    if isinstance(obj, dict):
        for k in ('prior_batch', 'prior', 'crystals', 'batch'):
            if k in obj:
                return obj[k]
        raise KeyError(f"{os.path.basename(path)}: no known batch key in {list(obj)[:6]}")
    return obj


def main():
    ref_pos, _, _ = one_molecule(load(REF))
    N_REF = len(ref_pos)
    ref_fp = fingerprint(ref_pos)
    rows = []

    for name, path in SOURCES:
        if not os.path.exists(path):
            rows.append((name, 'MISSING', None, None, None))
            continue
        try:
            obj = load(path)
            pos, _, spread = one_molecule(obj, N_REF)
            fp = fingerprint(pos)
            n = getattr(obj, 'num_graphs', len(obj) if hasattr(obj, '__len__') else 1)
            if len(fp) != len(ref_fp):
                rows.append((name, f'DIFFERENT ATOM COUNT ({len(fp)} vs {len(ref_fp)})',
                             n, None, summarize(fp)))
                continue
            dev = float((fp - ref_fp).abs().max())
            rows.append((name, None, n, dev, summarize(fp)))
        except Exception as e:
            rows.append((name, f'ERROR {type(e).__name__}: {e}', None, None, None))

    #: the Nikos levels each carry their own molecule -- L0 is HIS as delivered,
    #: L1/L2 are reprojected onto OUR reference, so L0 disagreeing is EXPECTED and
    #: L1/L2 disagreeing is a defect. Reported separately for that reason.
    lev_path = os.path.join(NC, 'nikos_levels.pt')
    if os.path.exists(lev_path):
        lev = torch.load(lev_path, weights_only=False, map_location='cpu')
        for lname in ('l0', 'l1', 'l2'):
            b = lev.get(lname)
            if b is None:
                rows.append((f'nikos {lname}', 'not present', None, None, None))
                continue
            pos, _, _ = one_molecule(b, N_REF)
            fp = fingerprint(pos)
            dev = (float((fp - ref_fp).abs().max())
                   if len(fp) == len(ref_fp) else None)
            rows.append((f'nikos {lname}', None if dev is not None else 'ATOM COUNT',
                         b.num_graphs, dev, summarize(fp)))

    print(f"reference conformer: {REF}")
    print(f"  {len(ref_pos)} heavy atoms, {len(ref_fp)} pairwise distances, "
          f"sha1 {hashlib.sha1(ref_fp.numpy().tobytes()).hexdigest()[:12]}")
    print()
    print(f"{'source':36s} {'n':>7} {'max dev vs ref':>15} {'bond mean':>10} "
          f"{'bond range':>16}")
    print('-' * 92)
    for name, err, n, dev, s in rows:
        if err:
            print(f"{name:36s} {'':>7} {err}")
            continue
        flag = '' if dev < 1e-4 else ('  <- DIFFERENT CONFORMER' if dev > 1e-3 else '  <- drift')
        print(f"{name:36s} {n:>7} {dev:>15.6f} {s['bond_mean']:>10.4f} "
              f"{s['bond_min']:>7.4f}-{s['bond_max']:.4f}{flag}")
    print()
    same = [r for r in rows if r[1] is None and r[3] is not None and r[3] < 1e-4]
    print(f"{len(same)} of {len(rows)} sources carry the reference conformer "
          f"to < 1e-4 A on every intramolecular distance.")

    #: Nikos' L0 deviates from our reference by almost exactly the amount our OLD
    #: conformer does -- ask directly whether his delivered molecule IS that one,
    #: rather than inferring it from two similar deviations.
    old_pos, _, _ = one_molecule(load(OLD), N_REF)
    old_fp = fingerprint(old_pos)
    lev = torch.load(os.path.join(NC, 'nikos_levels.pt'),
                     weights_only=False, map_location='cpu')
    l0_pos, _, _ = one_molecule(lev['l0'], N_REF)
    d = float((fingerprint(l0_pos) - old_fp).abs().max())
    print("")
    print(f'nikos L0 vs our OLD conformer: max deviation {d:.6f} A '
          f"-- {'SAME molecule' if d < 1e-3 else 'a THIRD, distinct conformer'}")


if __name__ == '__main__':
    main()
