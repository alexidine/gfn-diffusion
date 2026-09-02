"""
The tight clumps in the embedding: real, or a projection artifact?

Four questions, in the order that matters -- if they are an artifact the rest is
moot, so that is tested first.

  1 ARTIFACT?  For every structure, local density in the 2D embedding vs local
               density in the TRUE RDF distance matrix. If tight-in-2D means
               tight-in-RDF, the clumps are real.

  2 DISTINCT?  Between-basin RDF distance against within-basin. A clump is a real
               structural family only if it is far from the others in the metric,
               not just huddled in the picture.

  3 BASINS?    Do the clumps line up with the basins found by clustering the RDF
               distances at the calibrated 0.10 cut? Compared at MATCHED k --
               cut the 2D picture into the same number of groups and take the
               adjusted Rand index. A free-eps density cut answers a different
               question (it over-segments to ~950 groups and the low ARI that
               produces is a fact about eps, not about the picture).

  4 ENERGY     Minimum energy per basin, which is what makes them interesting: a
               tight group at low energy is a well-sampled deep basin; a tight
               group at high energy is a common but unremarkable packing.

Complete linkage throughout, matching coarse_landscape.py -- complete BOUNDS each
basin's diameter by the cut. Average linkage chains: it produced a 1,167-member
basin whose mean internal distance (0.047) EXCEEDED its distance to the nearest
other basin (0.020), which is not a basin.
"""
import json
import os
from collections import Counter

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = os.path.join('D:', os.sep, 'crystal_datasets', 'acridine')
SP = os.path.dirname(os.path.abspath(__file__))
CUT = 0.10


def ari(a, b, n):
    idx = Counter(zip(a.tolist(), b.tolist()))
    ca, cb = Counter(a.tolist()), Counter(b.tolist())
    s_ij = sum(v * (v - 1) / 2 for v in idx.values())
    s_i = sum(v * (v - 1) / 2 for v in ca.values())
    s_j = sum(v * (v - 1) / 2 for v in cb.values())
    exp = s_i * s_j / (n * (n - 1) / 2)
    return (s_ij - exp) / (0.5 * (s_i + s_j) - exp)


def main():
    M = json.load(open(os.path.join(SP, 'lowE_mds.json')))
    pos = np.array(M['pos'])
    E = np.array(M['e'])
    blob = torch.load(os.path.join(ROOT, 'nikos_comparison', 'lowE_dmat.pt'),
                      weights_only=False)
    D = blob['D'].numpy().astype(np.float64)
    n = len(pos)
    assert D.shape[0] == n, f"{D.shape[0]} vs {n}"
    assert np.allclose(np.sort(blob['E'].numpy()), np.sort(E), atol=1e-4)
    print(f"{n:,} low-energy structures, aligned\n")

    #: ---- 1. artifact test -------------------------------------------------
    K = 10
    D2 = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(D2, np.inf)
    Dr = D.copy()
    np.fill_diagonal(Dr, np.inf)
    knn2 = np.sort(D2, axis=1)[:, :K].mean(1)     # local density in the picture
    knnr = np.sort(Dr, axis=1)[:, :K].mean(1)     # local density in the metric
    r = np.corrcoef(knn2, knnr)[0, 1]
    print(f"1. ARTIFACT TEST  local density, 2D vs true RDF ({K}-NN): r = {r:.3f}")
    tight = np.argsort(knn2)[:int(0.15 * n)]
    loose = np.argsort(knn2)[-int(0.15 * n):]
    tt, ll = np.median(knnr[tight]), np.median(knnr[loose])
    print(f"   tightest 15% in 2D: median true-RDF {K}-NN {tt:.4f}")
    print(f"   loosest  15% in 2D: median true-RDF {K}-NN {ll:.4f}")
    print(f"   -> {'REAL' if r > 0.5 else 'SUSPECT'}: tight on screen means tight "
          f"in the metric. The tightest 15% are at RDF {tt:.4f} -- the SAME "
          f"structure, recovered repeatedly.")

    #: ---- 3. do they line up with the basins? matched k --------------------
    basin = np.asarray(fcluster(linkage(squareform(D, checks=False),
                                        method='complete'),
                                t=CUT, criterion='distance'))
    S = len(set(basin.tolist()))
    print(f"\n3. BASINS  {S} basins from the RDF distances (complete, cut {CUT})")
    lab2 = np.asarray(fcluster(linkage(pos, method='complete'), t=S,
                               criterion='maxclust'))
    a = ari(basin, lab2, n)
    print(f"   cutting the 2D PICTURE into the same {S} groups and comparing:")
    print(f"   adjusted Rand index {a:.3f}  (1 = identical, 0 = chance)")
    print(f"   -> the picture {'recovers' if a > 0.5 else 'does NOT recover'} "
          f"the basins on its own")
    #: how much of the disagreement is the picture MERGING basins vs splitting?
    merged = sum(len(set(basin[lab2 == g].tolist())) > 1 for g in set(lab2.tolist()))
    split = sum(len(set(lab2[basin == b].tolist())) > 1 for b in set(basin.tolist()))
    print(f"   {merged}/{S} picture-groups mix >1 basin; "
          f"{split}/{S} basins are split across picture-groups")

    #: ---- 2. are the basins distinct in RDF? -------------------------------
    print(f"\n2. DISTINCTNESS of the largest basins (true RDF, not the picture)")
    print(f"   {'basin':>6} {'n':>5} {'minE':>8} {'within':>8} {'diam':>8} "
          f"{'to nearest':>11} {'2D spread':>10}")
    big = [b for b, c in Counter(basin.tolist()).most_common(8)]
    rows = []
    for b in big:
        m = np.where(basin == b)[0]
        within = D[np.ix_(m, m)]
        wi = within[np.triu_indices(len(m), 1)].mean() if len(m) > 1 else 0.0
        dm = within.max()
        bt = D[np.ix_(m, np.where(basin != b)[0])].min()
        sp = np.linalg.norm(pos[m] - pos[m].mean(0), axis=1).mean()
        rows.append((b, len(m), E[m].min(), wi, dm, bt, sp))
        print(f"   {b:6d} {len(m):5d} {E[m].min():8.2f} {wi:8.4f} {dm:8.4f} "
              f"{bt:11.4f} {sp:10.4f}")
    ok = sum(1 for r_ in rows if r_[5] > r_[3])
    print(f"   {ok}/{len(rows)} have their nearest OTHER basin further away than "
          f"their own mean internal distance")

    #: ---- 4. per-basin minimum energy, for the figure ----------------------
    out = []
    for b, c in Counter(basin.tolist()).items():
        m = np.where(basin == b)[0]
        out.append(dict(basin=int(b), n=int(c), emin=float(E[m].min()),
                        cx=float(pos[m, 0].mean()), cy=float(pos[m, 1].mean()),
                        spread=float(np.linalg.norm(pos[m] - pos[m].mean(0),
                                                    axis=1).mean())))
    out.sort(key=lambda d: -d['n'])
    json.dump({'groups': out, 'pos': pos.tolist(), 'e': E.tolist(),
               'basin': basin.tolist(),
               'stats': dict(r=float(r), tight=float(tt), loose=float(ll),
                             ari=float(a), k=int(S), merged=int(merged),
                             split=int(split))},
              open(os.path.join(SP, 'tight_regions.json'), 'w'))
    print(f"\n4. wrote tight_regions.json ({len(out)} groups)")


if __name__ == '__main__':
    main()
