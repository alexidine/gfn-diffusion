"""
Observed basin-discovery curve, and a HOLD-OUT test of the extrapolation.

The Chao projection ("10x samples -> ~180 basins") is parametric. This measures the
curve directly and then checks whether the projection could have predicted the part
we can already see.

  RAREFACTION, exact (no Monte Carlo needed):
      E[S(m)] = S - sum_i C(n - n_i, m) / C(n, m)
  computed in logs via lgamma.

  HOLD-OUT: fit Chao1 + Chao extrapolation on a random m0 of the captures, project
  to the full n, compare against the OBSERVED S(n). If the projection misses badly
  at a 2-4x reach, it has no business being trusted at 10x.

Cluster labels are cached so cuts and replicates cost nothing.
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

CUT = 0.10
CACHE = os.path.join(ROOT, 'nikos_comparison', 'rarefaction_labels.pt')


def dmat(pool):
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
    bins = torch.linspace(0, 10, r.shape[-1])
    D = torch.stack([compute_rdf_distance(r[i], r, bins)
                     for i in range(len(r))]).numpy()
    D = (D + D.T) / 2; D[range(len(D)), range(len(D))] = 0.0
    return D


if os.path.exists(CACHE):
    blob = torch.load(CACHE, weights_only=False)
    print(f"reusing cached labels {CACHE}")
else:
    groups = load_arms(os.path.join(ROOT, 'opt_outs'), 'aug21*.pt')
    allc = []
    for stem in sorted(groups):
        if 'seed' in stem:
            continue
        flat, _ = physical([c for _, lst in sorted(groups[stem]) for c in lst])
        allc.extend(flat)
    E = torch.tensor([float(c.mace) for c in allc])
    blob = {}
    lo_idx = (E <= -59.0).nonzero().flatten().tolist()
    lo_pool = harmonize([allc[i] for i in lo_idx])
    blob['low'] = fcluster(linkage(squareform(dmat(lo_pool), checks=False),
                                   method='average'), t=CUT, criterion='distance')
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(allc), generator=g)[:4000].tolist()
    pool = harmonize([allc[i] for i in idx])
    blob['all'] = fcluster(linkage(squareform(dmat(pool), checks=False),
                                   method='average'), t=CUT, criterion='distance')
    torch.save(blob, CACHE)
    print(f"cached labels -> {CACHE}")


def logC(a, b):
    if b < 0 or b > a:
        return -math.inf
    return (math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1))


def rarefy(occ, n, m):
    """exact E[S(m)] under sampling m of n captures without replacement"""
    S = len(occ)
    tot = 0.0
    for ni in occ:
        lg = logC(n - ni, m) - logC(n, m)
        tot += math.exp(lg) if lg > -700 else 0.0
    return S - tot


def chao_project(occ_sub, n_sub, m_extra):
    S = len(occ_sub)
    f1 = sum(1 for v in occ_sub if v == 1)
    f2 = sum(1 for v in occ_sub if v == 2)
    chao = S + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
    f0 = max(chao - S, 1e-9)
    if f1 == 0:
        return S, chao
    return S + f0 * (1 - (1 - f1 / (n_sub * f0 + f1)) ** m_extra), chao


for tag, key in (('LOW-ENERGY (mace <= -59)', 'low'),
                 ('WHOLE LANDSCAPE (random 4,000)', 'all')):
    lab = blob[key]
    occ = list(Counter(lab).values())
    n, S = len(lab), len(occ)
    print(f"\n=== {tag}: n={n:,} captures, S={S:,} basins at cut {CUT} ===")
    print(f"{'draws':>8} {'basins E[S(m)]':>15} {'% of S(n)':>10} "
          f"{'new per 100 draws':>18}")
    prev_m, prev_s = 0, 0.0
    for frac in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        m = max(1, int(n * frac))
        s = rarefy(occ, n, m)
        rate = (s - prev_s) / max(m - prev_m, 1) * 100
        print(f"{m:8,d} {s:15.1f} {100*s/S:9.1f}% {rate:18.1f}")
        prev_m, prev_s = m, s

    print(f"  HOLD-OUT: fit on a fraction, project to the full n={n:,}, "
          f"compare to observed S={S:,}")
    g = torch.Generator().manual_seed(1)
    labs = torch.tensor(lab)
    for frac in (0.25, 0.5, 0.75):
        m0 = int(n * frac)
        errs = []
        for rep in range(20):
            perm = torch.randperm(n, generator=g)[:m0]
            sub = list(Counter(labs[perm].tolist()).values())
            proj, _ = chao_project(sub, m0, n - m0)
            errs.append(proj)
        e = torch.tensor(errs)
        print(f"     fit on {frac:.0%} ({m0:,}) -> predicts {float(e.mean()):7.1f} "
              f"+- {float(e.std()):5.1f}   observed {S:,}   "
              f"error {100*(float(e.mean()) - S)/S:+6.1f}%")
