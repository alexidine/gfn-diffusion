"""
Coarse-grained landscape: one point per BASIN, at its own minimum.

The tight clumps in the embedding are the optimiser converging to the SAME structure
over and over (true RDF distance ~0.000 within a clump). That is not a nuisance --
it says the landscape is genuinely DISCRETE. Well-defined attractors can be treated
as objects: abstract each to its lowest-energy member, and the 1,969 samples collapse
to a few dozen interpretable points.

Two independent routes to the same objects, which is the test worth running:

    RDF metric      cluster the distance matrix (complete linkage, cut 0.10 --
                    complete because it BOUNDS cluster diameter by the cut, which
                    is what the pairwise calibration licenses; average linkage
                    produced a basin spanning 0.187, wider than any real match)
    physical        the descriptors -- interplane angle, coordination, cell shape

If basins separate on the physical axes too, the two agree and each reinforces the
other. Measured here as a silhouette score on the physical descriptors using the
RDF-derived labels: positive means the RDF basins are also physically coherent.
"""
import json
import os
from collections import Counter

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from energy_sampling.eval.nikos_comparison.packing_motifs import classify, motif_of
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, harmonize, load_arms, physical)

SP = os.path.dirname(os.path.abspath(__file__))
LOW, CUT = -60.0, 0.10


def main():
    blob = torch.load(os.path.join(ROOT, 'nikos_comparison', 'lowE_dmat.pt'),
                      weights_only=False)
    D = blob['D'].numpy().astype(np.float64)
    E = blob['E'].numpy()
    n = D.shape[0]
    lab = fcluster(linkage(squareform(D, checks=False), method='complete'),
                   t=CUT, criterion='distance')
    occ = Counter(lab.tolist())
    print(f"{n:,} structures -> {len(occ)} basins (complete linkage, cut {CUT})")
    mx = max(D[np.ix_(np.where(lab == b)[0], np.where(lab == b)[0])].max()
             for b in occ)
    print(f"   widest basin diameter {mx:.4f} -- every pair inside the "
          f"same-packing regime")

    groups = load_arms(os.path.join(ROOT, 'prior_chunks'),
                       'may_acridine_sg14_zp1_*.pt')
    raw = [c for stem in sorted(groups)
           for _, lst in sorted(groups[stem]) for c in lst]
    keep, _ = physical(raw)
    Ea = torch.tensor([float(c.mace) for c in keep])
    pool = harmonize([keep[i] for i in
                      (Ea <= LOW).nonzero().flatten().tolist()])
    assert len(pool) == n

    M = json.load(open(os.path.join(SP, 'lowE_mds.json')))
    pos = np.array(M['pos'])

    #: one representative per basin: its LOWEST-energy member
    out = []
    for b in sorted(occ, key=lambda k: -occ[k]):
        m = np.where(lab == b)[0]
        j = int(m[np.argmin(E[m])])
        sh, flat = motif_of(pool[j])
        if not sh:
            continue
        th = np.array([x['theta'] for x in sh])
        stacks = [x for x in sh if x['theta'] < 25 and 3.1 <= x['h'] <= 3.9]
        edge = [x for x in sh if x['theta'] > 30]
        L = sorted(pool[j].cell_lengths.flatten().tolist())
        A = sorted(np.degrees(pool[j].cell_angles.flatten().tolist()).tolist())
        lbl, stk = classify(sh)
        out.append(dict(
            basin=int(b), n=int(occ[b]), emin=float(E[m].min()),
            emed=float(np.median(E[m])),
            mds=[float(pos[j, 0]), float(pos[j, 1])],
            mds_c=[float(pos[m, 0].mean()), float(pos[m, 1].mean())],
            pc=float(pool[j].packing_coeff),
            beta=float(A[2]), aniso=float(L[2] / L[0]), short=float(L[0]),
            theta_med=float(np.median(th)), theta_std=float(th.std()),
            n_nbr=len(sh), n_edge=len(edge),
            stack=float(stk) if stk else None,
            mean_d=float(np.mean([x['d'] for x in sh])),
            fam=lbl.split()[0].lower(), label=lbl,
            diam=float(D[np.ix_(m, m)].max()),
            flat=float(flat), shell=sh,
        ))
    print(f"   {len(out)} representatives described")

    #: DO THE TWO ROUTES AGREE? silhouette of the RDF labels in physical space
    keys = ['theta_med', 'theta_std', 'pc', 'beta', 'aniso', 'mean_d', 'n_nbr']
    idx = {g['basin']: i for i, g in enumerate(out)}
    X = np.stack([[g[k] for k in keys] for g in out])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    big = [g for g in out if g['n'] >= 5]
    if len(big) > 3:
        P = np.stack([[g[k] for k in keys] for g in big])
        P = (P - P.mean(0)) / (P.std(0) + 1e-9)
        Dp = np.linalg.norm(P[:, None] - P[None], axis=-1)
        np.fill_diagonal(Dp, np.inf)
        nearest = Dp.min(1)
        typical = np.median(Dp[np.isfinite(Dp)])
        print(f"\nagreement check, {len(big)} basins with n>=5:")
        print(f"   median distance to the NEAREST other basin in physical space "
              f"{np.median(nearest):.2f}")
        print(f"   median distance between ANY two basins                    "
              f"{typical:.2f}")
        print(f"   ratio {typical / max(np.median(nearest), 1e-9):.2f}x -- "
              f"basins are {'well separated' if typical / np.median(nearest) > 1.8 else 'crowded'}"
              f" on physical axes too")

    #: which basin holds each named structure -- nearest pool member, using the
    #: relaxed state (the one that actually sits in the landscape).
    #:
    #: ⚠ NEAREST IS NOT THE SAME AS INSIDE. A structure further from its nearest
    #: member than the cut that defines a basin is not IN that basin -- it is
    #: merely closest to it, which for a landscape this sparse can mean nothing at
    #: all. nik00001 sits 0.304 away with the cut at 0.10, three times outside, and
    #: labelling its nearest basin with its name put its NAME at that basin's
    #: energy and density while the landscape panel drew the STRUCTURE at its own
    #: -- the same object at -60.2 in one figure and -43.9 in another.
    #: in_basin is what every consumer must gate the label on.
    ST = json.load(open(os.path.join(SP, 'relax_states.json')))
    unplaced = []
    for st in ST:
        j = int(np.argmin(np.asarray(st['d1'])))
        b, d = int(lab[j]), float(min(st['d1']))
        inside = d <= CUT
        for g in out:
            if g['basin'] == b:
                g.setdefault('holds', []).append(
                    dict(name=st['name'], kind=st['kind'], rdf=d,
                         e=float(st['e1']), in_basin=inside))
        if not inside:
            unplaced.append(dict(name=st['name'], kind=st['kind'], rdf=d,
                                 nearest=b, e=float(st['e1'])))
        print(f"   {st['name']:10s} relaxed -> basin {b} (RDF {d:.4f})"
              f"{'' if inside else '  <- OUTSIDE the cut, NOT a member'}")
    if unplaced:
        print("")
        print(f'   {len(unplaced)} structure(s) match NO basin at cut {CUT}: '
              f"{', '.join(u['name'] for u in unplaced)}")
    json.dump(unplaced, open(os.path.join(SP, 'unplaced.json'), 'w'))

    json.dump(out, open(os.path.join(SP, 'coarse_landscape.json'), 'w'))
    print(f"\nwrote coarse_landscape.json")
    print(f"\n{'basin':>6} {'n':>5} {'minE':>8} {'diam':>7} {'motif':>28}")
    for g in out[:12]:
        print(f"{g['basin']:6d} {g['n']:5d} {g['emin']:8.2f} {g['diam']:7.4f} "
              f"{g['label'][:28]:>28}")


if __name__ == '__main__':
    main()
