"""
Characterise low-energy basins by PACKING MOTIF, not just by RDF distance.

The sg14-Z'1 low-energy stratum (mace <= -60) is complete: 1,969 structures in 34
basins at cut 0.10, 99.7% Good-Turing mass coverage. 34 is small enough to describe
one by one, which is the point of having enumerated it.

Acridine is a rigid planar aromatic, so its packing is governed by how molecular
PLANES relate. For each basin representative, take the reference molecule and every
neighbour whose centroid is within `--neighbour-cutoff`, then per neighbour:

    theta   angle between plane normals, folded to [0, 90] deg
            ~0     coplanar or face-to-face stacked
            ~90    edge-to-face
    h       centroid separation ALONG the reference normal (stack height)
    s       centroid separation IN the reference plane (slip)

The motif follows from the joint distribution, not from theta alone -- theta ~ 0
with h ~ 3.5 is a pi-stack, while theta ~ 0 with h ~ 0 is two molecules side by side
in one sheet. Those are completely different packings and an angle-only descriptor
merges them.

    SANDWICH / pi-STACK   theta < 30, h in [3.0, 4.2], s < 3.0
    OFFSET STACK          theta < 30, h in [3.0, 4.2], s >= 3.0
    SHEET (coplanar)      theta < 30, h < 1.5
    HERRINGBONE           theta > 50
    (anything else reported as MIXED, with the numbers, rather than forced)

⚠ VALIDATE BEFORE TRUSTING. The known polymorphs are run first: acridine forms are
documented herringbone/stacked aromatics, so a descriptor that returns nonsense on
ACRDIN04/ACRDIN12 is broken and nothing below it means anything.

    python -m energy_sampling.eval.nikos_comparison.packing_motifs
"""
import argparse
import os
from collections import Counter

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, chunk_by_cluster_cost, harmonize, load_arms, physical)


def plane_normal(pos):
    """Unit normal of the best-fit plane: smallest singular vector of centred pos."""
    x = pos - pos.mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    return v[-1] / v[-1].norm()


def planarity(pos):
    """RMS out-of-plane deviation, so a NON-planar molecule cannot pass silently."""
    x = pos - pos.mean(0, keepdim=True)
    n = plane_normal(pos)
    return float((x @ n).pow(2).mean().sqrt())


def ref_frame(pos):
    """
    Molecular frame from SVD: e1 long in-plane axis, e2 short in-plane, e3 normal.

    Acridine is elongated as well as flat, so the long axis is well defined and
    gives a real in-plane reference. Without it the only thing left to draw is a
    magnitude, and every basin collapses to the same picture.
    """
    x = pos - pos.mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    return v[0], v[1], v[2]


def motif_of(crystal, cutoff=9.0, supercell=3):
    """
    Full per-neighbour geometry IN THE REFERENCE MOLECULE'S FRAME.

    Returns one row per neighbour:
        theta  angle between plane normals, folded to [0,90]
        h      offset along the reference NORMAL      (stack height)
        s      offset magnitude in the reference PLANE (slip)
        d      centroid distance
        u,v,w  the offset resolved on (e1,e2,e3) -- SIGNED, so a projection can be
               drawn rather than a summary statistic
        phi    apparent tilt of the neighbour's long axis in the (e1,e3) projection

    The earlier version returned only (theta,h,s,d) and the atlas then drew from
    the MEDIAN of theta alone. Eleven beta basins share theta=0 and rendered
    identically. Slip and the signed offsets are what separate them.
    """
    b = collate_data_list([crystal.clone()],
                          exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
    b.aunit_handedness = b.aunit_handedness.abs()
    cl = b.mol2cluster(cutoff=cutoff, supercell_size=supercell)
    pos, mol_ind = cl.pos, cl.mol_ind
    n_at = int(crystal.num_atoms) // max(int(crystal.z_prime), 1)

    ref = pos[:n_at]
    e1, e2, e3 = ref_frame(ref)
    rc = ref.mean(0)
    flat = planarity(ref)

    rows = []
    for m in torch.unique(mol_ind).tolist():
        sel = (mol_ind == m).nonzero().flatten()
        if len(sel) != n_at:
            continue
        p = pos[sel]
        dvec = p.mean(0) - rc
        dn = float(dvec.norm())
        if dn < 1e-6 or dn > cutoff:
            continue
        f1, _, f3 = ref_frame(p)
        cos = float(torch.clamp((e3 * f3).sum().abs(), 0, 1))
        theta = float(np.degrees(np.arccos(cos)))
        u, v, w = (float((dvec * e1).sum()), float((dvec * e2).sum()),
                   float((dvec * e3).sum()))
        #: the neighbour's long axis seen in the (e1,e3) plane -- its apparent tilt
        a1, a3 = float((f1 * e1).sum()), float((f1 * e3).sum())
        phi = float(np.degrees(np.arctan2(a3, a1)))
        #: FULL ORIENTATION, so any projection can be drawn downstream without
        #: re-running this. The (long axis, normal) view hides the herringbone
        #: partner -- only 6 of 16 gamma basins keep a >45deg neighbour in a 3 A
        #: slab of it, so ten of them render like beta. The (short axis, normal)
        #: view keeps 14 of 16, and needs f3 resolved on e2/e3 to place the
        #: molecular plane's trace.
        L1 = [float((f1 * e).sum()) for e in (e1, e2, e3)]
        L3 = [float((f3 * e).sum()) for e in (e1, e2, e3)]
        #: FORESHORTENING. `ext` is how much of the long axis survives the (e1,e3)
        #: projection: 1 = lying in the view plane, 0 = pointing straight into it.
        #: Without it every bar draws full length and the picture silently claims
        #: all neighbours are coplanar with the reference.
        ext = float(np.hypot(a1, a3))
        rows.append(dict(theta=theta, h=abs(w), s=float(np.hypot(u, v)), d=dn,
                         u=u, v=v, w=w, phi=phi, ext=ext, L1=L1, L3=L3))
    return rows, flat


def classify(rows):
    """
    Two chemically meaningful axes, not a majority vote over neighbours.

    A packing motif is a property of the whole coordination shell. Majority-voting
    per-neighbour labels put ACRDIN04 -- a textbook pi-stacked herringbone with a
    3.38 A stack -- into "other (82%)", because most of its shell is neither the
    stack nor cleanly edge-to-face.

    AXIS 1  is there a pi-stack?   any neighbour with theta < 25 and 3.1 <= h <= 3.9
    AXIS 2  what do the NON-stacked neighbours do?  median theta over theta > 25

    This reproduces the standard planar-aromatic taxonomy (Desiraju/Gavezzotti):

        stack + edge-to-face      GAMMA / sandwich-herringbone
        stack + shallow           BETA  / stacked layers
        no stack + edge-to-face   HERRINGBONE (T-shaped)
        no stack + shallow        SHEET / coplanar
    """
    if not rows:
        return 'NO NEIGHBOURS', None
    stacks = [r for r in rows if r['theta'] < 25 and 3.1 <= r['h'] <= 3.9]
    rest = [r['theta'] for r in rows if r['theta'] >= 25]
    med_rest = float(np.median(rest)) if rest else 0.0
    edge = med_rest > 45
    if stacks:
        d = min(s2['h'] for s2 in stacks)
        return (f"GAMMA stack {d:.2f}A + herringbone {med_rest:.0f}deg" if edge
                else f"BETA stacked layers {d:.2f}A, tilt {med_rest:.0f}deg"), d
    return (f"HERRINGBONE no stack, {med_rest:.0f}deg" if edge
            else f"SHEET coplanar, {med_rest:.0f}deg"), None


def describe(tag, crystal, extra=''):
    rows, flat = motif_of(crystal)
    if not rows:
        print(f"{tag:26s} NO NEIGHBOURS FOUND -- cluster build failed?")
        return
    th = torch.tensor([r['theta'] for r in rows])
    near = min(rows, key=lambda r: r['d'])
    label, stack_d = classify(rows)
    sd = f"{stack_d:4.2f}" if stack_d else "  --"
    print(f"{tag:26s} flat {flat:5.3f} | {len(rows):3d} nbr | "
          f"th_med {float(th.median()):5.1f} | near {near['d']:4.2f}A | "
          f"stack {sd} | {label}{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='prior_chunks')
    ap.add_argument('--pattern', default='may_acridine_sg14_zp1_*.pt')
    ap.add_argument('--forms', nargs='+', default=['ACRDIN04', 'ACRDIN12'])
    ap.add_argument('--cut', type=float, default=0.10)
    ap.add_argument('--low-energy', type=float, default=-60.0)
    ap.add_argument('--neighbour-cutoff', type=float, default=9.0)
    cli = ap.parse_args()

    #: CONTROL FIRST -- the known forms are documented packings. If these come back
    #: as nonsense the descriptor is broken and the basin table below is worthless.
    poly = torch.load(os.path.join(ROOT, 'std_opt_acridine_polymorphs.pt'),
                      weights_only=False, map_location='cpu').cpu()
    pl, pids = poly.batch_to_list(), list(poly.identifier)
    print("=== CONTROL: known polymorphs (expect planarity ~0, aromatic "
          "stack-h 3.3-3.8 A) ===")
    for n in pids:
        describe(n, pl[pids.index(n)])

    print(f"\n=== LOW-ENERGY BASINS (mace <= {cli.low_energy}, cut {cli.cut}) ===")
    groups = load_arms(os.path.join(ROOT, cli.source), cli.pattern)
    raw = [c for stem in sorted(groups)
           for _, lst in sorted(groups[stem]) for c in lst]
    keep, _ = physical(raw)
    E = torch.tensor([float(c.mace) for c in keep])
    idx = (E <= cli.low_energy).nonzero().flatten().tolist()
    pool = harmonize([keep[i] for i in idx])
    Ep = E[idx]
    print(f"{len(pool):,} structures below {cli.low_energy}")

    r = []
    for lo, hi in chunk_by_cluster_cost(pool, 1_500_000):
        b = collate_data_list([c.clone() for c in pool[lo:hi]],
                              exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        b.aunit_handedness = b.aunit_handedness.abs()
        with torch.no_grad():
            o = b.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
        rr = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
        r.append(rr.cpu())
        del b, o, rr
    r = torch.cat(r)
    bins = torch.linspace(0, 10, r.shape[-1])
    m = len(r)
    D = np.empty((m, m), dtype=np.float32)
    for i in range(m):
        D[i] = compute_rdf_distance(r[i], r, bins).numpy()
    D += D.T
    D *= 0.5
    np.fill_diagonal(D, 0.0)
    lab = fcluster(linkage(squareform(D, checks=False), method='average'),
                   t=cli.cut, criterion='distance')
    occ = Counter(lab)

    #: which basin holds each known form, so the table says where they landed
    form_basin = {}
    for n in cli.forms:
        if n not in pids:
            continue
        fb = collate_data_list([pl[pids.index(n)].clone()],
                               exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        fb.aunit_handedness = fb.aunit_handedness.abs()
        with torch.no_grad():
            fo = fb.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
        fr = fo['rdf'][0] if isinstance(fo['rdf'], (tuple, list)) else fo['rdf']
        d = compute_rdf_distance(fr[0], r, bins)
        form_basin.setdefault(int(lab[int(d.argmin())]), []).append(
            f"{n} (RDF {float(d.min()):.4f})")

    print(f"\n{'basin':>6} {'n':>5} {'minE':>8}  descriptor")
    order = sorted(occ, key=lambda k: -occ[k])
    for bi in order:
        members = (torch.tensor(lab) == bi).nonzero().flatten()
        j = int(members[Ep[members].argmin()])
        note = ('   <<< ' + '; '.join(form_basin[bi])) if bi in form_basin else ''
        describe(f"  {bi:4d} {occ[bi]:5d} {float(Ep[j]):8.2f}", pool[j], note)


if __name__ == '__main__':
    main()
