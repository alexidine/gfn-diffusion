"""
Basin richness vs the clustering cut -- because the cut IS the answer here.

A confirmed 20/20 COMPACK match sits at RDF 0.044-0.049, so two independent runs
landing in the SAME basin are ~0.05 apart. Cutting at 0.05 therefore SPLITS genuine
duplicates and inflates the basin count. Sweep instead of asserting: if "almost no
duplicates" survives to 0.10-0.15, it is real; if it evaporates, it was the cut.

Species-richness view, each basin a species, each optimised structure a capture:
  Good-Turing C = 1 - f1/n   fraction of the PROBABILITY MASS covered
  Chao1         = S + f1^2/(2 f2)   lower bound on true richness
  Chao extrapolation for what more sampling buys.

Distance matrix computed ONCE; every cut is then free.
"""
import os, torch
from collections import Counter
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list
from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, chunk_by_cluster_cost, harmonize, load_arms, physical)

CUTS = [0.05, 0.08, 0.10, 0.15, 0.20]
N = 4000

groups = load_arms(os.path.join(ROOT, 'opt_outs'), 'aug21*.pt')
allc = []
for stem in sorted(groups):
    if 'seed' in stem:
        continue
    flat, _ = physical([c for _, lst in sorted(groups[stem]) for c in lst])
    allc.extend(flat)
E_all = torch.tensor([float(c.mace) for c in allc])
print(f"{len(allc):,} physical unseeded outputs", flush=True)


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


def report(tag, pool, D, n_pop):
    Z = linkage(squareform(D, checks=False), method='average')
    n = len(pool)
    print(f"\n=== {tag}  (n={n:,}, drawn from a population of {n_pop:,}) ===")
    print(f"{'cut':>6} {'basins':>8} {'f1':>7} {'f2':>6} {'mass cov':>9} "
          f"{'Chao1':>10} {'largest':>8} {'x10 samples':>12}")
    for cut in CUTS:
        lab = fcluster(Z, t=cut, criterion='distance')
        occ = Counter(lab)
        S = len(occ)
        f1 = sum(1 for v in occ.values() if v == 1)
        f2 = sum(1 for v in occ.values() if v == 2)
        C = 1 - f1 / n
        chao = S + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
        f0 = max(chao - S, 1e-9)
        m = n * 9
        S10 = S + f0 * (1 - (1 - f1 / (n * f0 + f1)) ** m)
        print(f"{cut:6.2f} {S:8,d} {f1:7,d} {f2:6,d} {100*C:8.1f}% "
              f"{chao:10,.0f} {max(occ.values()):8d} {S10:12,.0f}")


g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(allc), generator=g)[:N].tolist()
pool = harmonize([allc[i] for i in idx])
report('RANDOM sample (whole landscape)', pool, dmat(pool), len(allc))

lo_idx = (E_all <= -59.0).nonzero().flatten().tolist()
print(f"\n{len(lo_idx):,} structures at mace <= -59.0", flush=True)
if len(lo_idx) >= 30:
    lo_pool = harmonize([allc[i] for i in lo_idx])
    report('LOW-ENERGY set, mace <= -59.0 (complete, not a sample)',
           lo_pool, dmat(lo_pool), len(lo_idx))
