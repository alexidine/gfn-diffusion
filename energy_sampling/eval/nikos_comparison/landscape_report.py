"""
Deep characterisation of one SG/Z' landscape, from RAW search output.

Answers, in one pass:
  * inventory  -- how much landed, how much of it is physical, where the energies sit
  * basins     -- how many distinct packings, how often each was found
  * coverage   -- Good-Turing mass coverage and Chao1 richness
  * curve      -- observed discovery curve (exact rarefaction) + marginal rate
  * projection -- what more sampling buys, HOLD-OUT VALIDATED before it is quoted
  * targets    -- where the known forms sit in energy and in basin occupancy

⚠ MUST BE RUN ON RAW CHUNKS, NEVER THE THINNED PRIOR. `collate_prior` thins with
`greedy_bottom_up_anchors`, which removes near-duplicates BY CONSTRUCTION. Occupancy
on a thinned set reads as "every basin found once" and fakes complete coverage. For
sg14-Z'1 that is 13,925 thinned against 55,076 raw.

⚠ THE CUT IS NOT A FREE PARAMETER. Take it from `calibrate_basin_metric.py` on THIS
combination. On sg14-Z'2, envwise <0.085 is certainly the same packing, 0.085-0.147
is 29% ambiguous, >0.147 never. Do not carry a cut across combinations unchecked.

⚠ ENVWISE, not atomwise: acridine has a C2 axis, and atomwise fixes the atom
indexing, so packings identical under C2 land far apart and its P(match) curve is
non-monotonic. That is a property of the molecule, not of Z'.

    python -m energy_sampling.eval.nikos_comparison.landscape_report \\
        --pattern 'may_acridine_sg14_zp1_*.pt' --forms ACRDIN04 ACRDIN12
"""
import argparse
import math
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
    ROOT, chunk_by_cluster_cost, harmonize, known_form_rdfs, load_arms, physical)


def rdf_of(lst):
    out = []
    for lo, hi in chunk_by_cluster_cost(lst, 1_500_000):
        b = collate_data_list([c.clone() for c in lst[lo:hi]],
                              exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        b.aunit_handedness = b.aunit_handedness.abs()
        with torch.no_grad():
            o = b.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
        r = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
        out.append(r.cpu())
        del b, o, r
    return torch.cat(out, dim=0)


def dmat(r):
    m = len(r)
    bins = torch.linspace(0, 10, r.shape[-1])
    D = np.empty((m, m), dtype=np.float32)
    for i in range(m):
        D[i] = compute_rdf_distance(r[i], r, bins).numpy()
    D += D.T
    D *= 0.5
    np.fill_diagonal(D, 0.0)
    return D


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


def chao(occ_counts):
    S = len(occ_counts)
    f1 = sum(1 for v in occ_counts if v == 1)
    f2 = sum(1 for v in occ_counts if v == 2)
    est = S + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
    return S, f1, f2, est


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='prior_chunks')
    ap.add_argument('--pattern', default='may_acridine_sg14_zp1_*.pt')
    ap.add_argument('--forms', nargs='+', default=['ACRDIN04', 'ACRDIN12'])
    ap.add_argument('--cut', type=float, default=0.10,
                    help='basin-identity RDF cut; take it from '
                         'calibrate_basin_metric.py ON THIS COMBINATION')
    ap.add_argument('--sample', type=int, default=8000,
                    help='random draw for the O(n^2) basin analysis')
    ap.add_argument('--low-energy', type=float, default=None,
                    help='also analyse the COMPLETE stratum below this energy')
    cli = ap.parse_args()

    groups = load_arms(os.path.join(ROOT, cli.source), cli.pattern)
    raw = [c for stem in sorted(groups)
           for _, lst in sorted(groups[stem]) for c in lst]
    keep, dropped = physical(raw)
    E = torch.tensor([float(c.mace) for c in keep])
    pc = torch.tensor([float(c.packing_coeff) for c in keep])
    q = torch.tensor([0., .01, .25, .5, .75, 1.])
    print(f"=== INVENTORY ===")
    print(f"{len(raw):,} raw -> {len(keep):,} physical ({dropped})")
    print(f"   mace  min/1%/25%/median/75%/max: "
          f"{[round(float(v), 2) for v in torch.quantile(E, q)]}")
    print(f"   pcoef                          : "
          f"{[round(float(v), 3) for v in torch.quantile(pc, q)]}")

    names, form_rdf = known_form_rdfs(cli.forms)
    print(f"\n=== TARGETS: where the known forms sit ===")

    def analyse(tag, pool_idx, complete):
        pool = harmonize([keep[i] for i in pool_idx])
        r = rdf_of(pool)
        D = dmat(r)
        Z = linkage(squareform(D, checks=False), method='average')
        lab = fcluster(Z, t=cli.cut, criterion='distance')
        occ = Counter(lab)
        counts = list(occ.values())
        n = len(pool)
        S, f1, f2, est = chao(counts)
        print(f"\n=== BASINS: {tag} (n={n:,}, {'COMPLETE' if complete else 'sample'},"
              f" cut {cli.cut}) ===")
        print(f"   distinct basins        {S:,}")
        print(f"   singletons / doubletons {f1:,} / {f2:,}")
        print(f"   largest basin           {max(counts)} captures")
        print(f"   Good-Turing mass cover  {100 * (1 - f1 / n):.1f}%")
        print(f"   Chao1 richness          {est:,.0f}  ({est - S:,.0f} unseen)")

        print(f"   discovery curve (new basins per 100 draws):")
        prev_m, prev_s = 0, 0.0
        for frac in (0.05, 0.125, 0.25, 0.5, 1.0):
            m = max(1, int(n * frac))
            sv = rarefy(counts, n, m)
            print(f"      {m:7,d} draws -> {sv:8.1f} basins   "
                  f"marginal {(sv - prev_s) / max(m - prev_m, 1) * 100:6.1f}")
            prev_m, prev_s = m, sv

        #: HOLD-OUT before quoting any projection
        g2 = torch.Generator().manual_seed(1)
        labs = torch.tensor(lab)
        print(f"   projection HOLD-OUT (fit on a fraction, predict the full n):")
        for frac in (0.25, 0.5):
            m0 = int(n * frac)
            preds = []
            for _ in range(10):
                sub = list(Counter(
                    labs[torch.randperm(n, generator=g2)[:m0]].tolist()).values())
                s2, g1, gg2, e2 = chao(sub)
                f0 = max(e2 - s2, 1e-9)
                preds.append(s2 + f0 * (1 - (1 - g1 / (m0 * f0 + g1)) ** (n - m0))
                             if g1 else s2)
            pt = torch.tensor(preds)
            print(f"      fit {frac:.0%} -> {float(pt.mean()):8.1f} +- "
                  f"{float(pt.std()):5.1f}   observed {S:,}   "
                  f"error {100 * (float(pt.mean()) - S) / S:+6.1f}%")

        #: are the known forms in heavily-occupied basins, or rare ones?
        bins = torch.linspace(0, 10, r.shape[-1])
        for k, nm in enumerate(names):
            d = compute_rdf_distance(form_rdf[k], r, bins)
            j = int(d.argmin())
            print(f"   {nm:14s} nearest sample RDF {float(d[j]):.4f}, "
                  f"its basin holds {occ[lab[j]]:4d} captures, "
                  f"E={float(E[pool_idx[j]]):.2f}, "
                  f"{int((d <= cli.cut).sum()):5d} samples within the cut")

    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(keep), generator=g)[:cli.sample].tolist()
    analyse('random sample of the whole landscape', idx, complete=False)

    if cli.low_energy is not None:
        lo = (E <= cli.low_energy).nonzero().flatten().tolist()
        if len(lo) < 30:
            print(f"\nonly {len(lo)} structures below {cli.low_energy}; skipping")
        else:
            if len(lo) > cli.sample:
                print(f"\nlow-energy stratum has {len(lo):,}; sampling "
                      f"{cli.sample:,} -- NO LONGER COMPLETE, coverage will read low")
                lo = [lo[i] for i in
                      torch.randperm(len(lo), generator=g)[:cli.sample].tolist()]
            analyse(f'low-energy stratum, mace <= {cli.low_energy}', lo,
                    complete=len(lo) <= cli.sample)


if __name__ == '__main__':
    main()
