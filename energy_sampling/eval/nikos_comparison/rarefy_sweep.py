"""
Discovery curve vs BOTH sample size and the basin-identity cut.

Two objections to the earlier curves, both fair:

  1. REPRESENTATIVENESS. The low-energy curve is the COMPLETE stratum (all 692
     structures below -59 of 70,271) so it is exactly what it claims. The
     whole-landscape curve was a 4,000 draw -- 5.7% -- so it measured accumulation
     only out to n=4,000 and could not see bending beyond that. Enlarged here.
  2. THE CUT. A confirmed 20/20 COMPACK match sits at RDF 0.044-0.049, so a tight
     cut splits genuine duplicates and manufactures a linear-looking curve. Swept.

The LINKAGE is cached, not the labels: one distance matrix supports every cut, so
the sweep is free after the first run.

Reported per (n, cut): basins found, mass coverage, and the MARGINAL rate at the end
of the curve -- new basins per 100 draws. A curve that is bending has a marginal rate
that falls as n grows AT FIXED CUT.
"""
import math, os, torch
from collections import Counter
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list
from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, chunk_by_cluster_cost, harmonize, load_arms, physical)

N = 10000
CUTS = [0.05, 0.10, 0.15, 0.20, 0.30]
CACHE = os.path.join(ROOT, 'nikos_comparison', f'rarefy_linkage_{N}.pt')


def logC(a, b):
    if b < 0 or b > a:
        return -math.inf
    return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)


def rarefy(occ, n, m):
    tot = 0.0
    for ni in occ:
        lg = logC(n - ni, m) - logC(n, m)
        tot += math.exp(lg) if lg > -700 else 0.0
    return len(occ) - tot


if os.path.exists(CACHE):
    blob = torch.load(CACHE, weights_only=False)
    print(f"reusing cached linkage {CACHE}", flush=True)
else:
    groups = load_arms(os.path.join(ROOT, 'opt_outs'), 'aug21*.pt')
    allc = []
    for stem in sorted(groups):
        if 'seed' in stem:
            continue
        flat, _ = physical([c for _, lst in sorted(groups[stem]) for c in lst])
        allc.extend(flat)
    print(f"{len(allc):,} physical unseeded outputs", flush=True)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(allc), generator=g)[:N].tolist()
    pool = harmonize([allc[i] for i in idx])
    print(f"random sample of {len(pool):,} ({100*len(pool)/len(allc):.1f}% "
          f"of the population); computing RDFs", flush=True)
    r = []
    for lo, hi in chunk_by_cluster_cost(pool, 1_500_000):
        b = collate_data_list([c.clone() for c in pool[lo:hi]],
                              exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        b.aunit_handedness = b.aunit_handedness.abs()
        with torch.no_grad():
            o = b.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
        rr = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
        r.append(rr.cpu()); del b, o, rr
    r = torch.cat(r)
    #: FILL IN PLACE. The previous version built a 12,000-element list of rows,
    #: stacked it, copied to numpy, then symmetrised -- several GB at peak, and the
    #: process was OOM-KILLED with no traceback and a 0 exit code, which reads as
    #: success. Preallocate one float32 array, write rows into it, free the RDFs
    #: before the condensed copy scipy needs.
    import numpy as np
    m = len(r)
    print(f"distance matrix {m:,}x{m:,} (filling in place)", flush=True)
    bins = torch.linspace(0, 10, r.shape[-1])
    D = np.empty((m, m), dtype=np.float32)
    for i in range(m):
        D[i] = compute_rdf_distance(r[i], r, bins).numpy()
        if i % 1000 == 0:
            print(f"   row {i:,}/{m:,}", flush=True)
    del r
    D += D.T; D *= 0.5
    np.fill_diagonal(D, 0.0)
    print("linkage", flush=True)
    Z = linkage(squareform(D, checks=False), method='average')
    del D
    blob = {'Z': Z, 'n': N}
    torch.save(blob, CACHE)
    print(f"cached -> {CACHE}", flush=True)

Z, n = blob['Z'], blob['n']
print(f"\nWHOLE LANDSCAPE, n={n:,} random draws")
print(f"{'cut':>6} {'basins':>8} {'mass cov':>9}   marginal new-per-100 at "
      f"n/8, n/4, n/2, n")
for cut in CUTS:
    lab = fcluster(Z, t=cut, criterion='distance')
    occ = list(Counter(lab).values())
    S = len(occ)
    f1 = sum(1 for v in occ if v == 1)
    marg = []
    pts = [n // 8, n // 4, n // 2, n]
    for m in pts:
        d = max(1, m // 50)
        marg.append((rarefy(occ, n, m) - rarefy(occ, n, m - d)) / d * 100)
    print(f"{cut:6.2f} {S:8,d} {100*(1-f1/n):8.1f}%   "
          + '  '.join(f'{x:6.1f}' for x in marg))
print("\na rate that FALLS across the row = the curve is bending at that cut.")
print("a rate that stays flat = still linear; the basin count is a floor.")
