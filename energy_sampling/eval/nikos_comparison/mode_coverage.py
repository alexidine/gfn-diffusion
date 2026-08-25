"""
Did the sampling modes COVER different regions, or repeat each other?

The wave ran four unseeded modes (base, stdnocomp, lightcomp, lowlr) precisely to
widen coverage. They all reach the same energy floor and the same distance to the
known forms, which is suggestive but not a measurement -- two modes can agree on
their extremes and still sample disjoint interiors.

So: pool the lowest-energy `--per-mode` structures from each mode, cluster the pool
on RDF distance alone, and count clusters that only ONE mode reached. That is
coverage stated directly.

  * threshold: `--cut`, default 0.05. Calibrated, not guessed -- on this system
    RDF 0.044-0.049 is a 20/20 COMPACK match and 0.133+ is not
    (see confirm_seeded.py). 0.05 is the tight end of "same structure".
  * clustering is average-linkage agglomerative on the precomputed distance matrix,
    matching `modes.py`. Do NOT use paper1's density clustering: these are
    OPTIMIZER OUTPUTS, so local density measures where the optimizer piles up, not
    free energy.

Reading the result: `unique` clusters are the ONLY evidence a mode earned its
place. A mode with high `n` and zero unique clusters sampled nothing the others
did not, however different its settings look.

    python -m energy_sampling.eval.nikos_comparison.mode_coverage
"""
import argparse
import os
from collections import Counter

import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, chunk_by_cluster_cost, harmonize, load_arms, physical)


def rdf_of(lst, budget=1_500_000):
    out = []
    for lo, hi in chunk_by_cluster_cost(lst, budget):
        b = collate_data_list([c.clone() for c in lst[lo:hi]],
                              exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        with torch.no_grad():
            o = b.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
        r = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
        out.append(r.cpu())
        del b, o, r
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'opt_outs'))
    ap.add_argument('--pattern', default='aug21*.pt')
    ap.add_argument('--per-mode', type=int, default=1200,
                    help='lowest-energy structures taken from each mode')
    ap.add_argument('--cut', type=float, default=0.05)
    cli = ap.parse_args()

    groups = load_arms(cli.out_dir, cli.pattern)
    pool, owner = [], []
    for stem in sorted(groups):
        flat, _ = physical([c for _, lst in sorted(groups[stem]) for c in lst])
        e = torch.tensor([float(c.mace) for c in flat])
        take = e.argsort()[:cli.per_mode].tolist()
        mode = stem.replace('aug21', '').replace('_acridine_sg14_zp2', '')
        print(f"   {mode:12s} {len(flat):7,d} physical -> taking {len(take):,} "
              f"lowest ({float(e[take[0]]):.2f} .. {float(e[take[-1]]):.2f} kJ/mol)")
        pool.extend(flat[i] for i in take)
        owner.extend([mode] * len(take))

    print(f"\npool {len(pool):,}; computing RDFs")
    pool = harmonize(pool)   # modes disagree on rdf_bins / vdw_max
    rdf = rdf_of(pool)
    bins = torch.linspace(0, 10, rdf.shape[-1])

    print("distance matrix")
    D = torch.stack([compute_rdf_distance(rdf[i], rdf, bins)
                     for i in range(len(rdf))]).numpy()
    D = (D + D.T) / 2
    D[range(len(D)), range(len(D))] = 0.0

    #: cache the expensive half. RDFs + a 6000x6000 distance matrix cost ~15
    #: min; re-analysis (thresholds, controls) then costs seconds. Keyed by
    #: pool composition so a changed --per-mode cannot silently reuse it.
    key = f'{len(pool)}_{cli.per_mode}_' + '_'.join(sorted(set(owner)))
    cpath = os.path.join(ROOT, 'nikos_comparison', f'coverage_{key}.pt')

    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    if os.path.exists(cpath):
        blob = torch.load(cpath, weights_only=False)
        D, owner = blob['D'], blob['owner']
        print(f'reusing cached distance matrix {cpath}')
    else:
        torch.save({'D': torch.as_tensor(D), 'owner': owner}, cpath)
        print(f'cached distance matrix -> {cpath}')
    D = torch.as_tensor(D).numpy()

    lab = fcluster(linkage(squareform(D, checks=False), method='average'),
                   t=cli.cut, criterion='distance')
    members = {}
    for l, m in zip(lab, owner):
        members.setdefault(l, Counter())[m] += 1
    print(f"\n{len(members):,} clusters at RDF cut {cli.cut}")

    print(f"\n{'mode':12s} {'in pool':>8} {'clusters':>9} {'UNIQUE':>7} "
          f"{'% of its clusters':>18}")
    for mode in sorted(set(owner)):
        touched = [c for c, cnt in members.items() if mode in cnt]
        uniq = [c for c in touched if len(members[c]) == 1]
        print(f"{mode:12s} {owner.count(mode):8,d} {len(touched):9,d} "
              f"{len(uniq):7,d} {100 * len(uniq) / max(len(touched), 1):17.1f}%")

    shared = sum(1 for c in members.values() if len(c) > 1)
    print(f"\n{shared:,} clusters reached by more than one mode "
          f"({100 * shared / len(members):.1f}%)")

    #: THE CONTROL. A singleton cluster is unique to one mode BY DEFINITION,
    #: so when the pool is sparse relative to the cut, 'unique' counts measure
    #: SPARSITY, not coverage. Under random mode labels a singleton is unique
    #: with probability 1 and a pair with probability 0.2, so a sparse pool
    #: manufactures high uniqueness for free. Compare against relabelled modes:
    #: only the EXCESS over that null is evidence a mode went somewhere else.
    n1 = sum(1 for c in members.values() if sum(c.values()) == 1)
    print()
    print(f'{n1:,} of {len(members):,} clusters are singletons '
          f'({100 * n1 / len(members):.1f}%) -- each unique by definition')

    g = torch.Generator().manual_seed(0)
    obs = {m: sum(1 for c, cnt in members.items()
                  if m in cnt and len(cnt) == 1) for m in set(owner)}
    null = {m: [] for m in set(owner)}
    for _ in range(20):
        perm = [owner[i] for i in
                torch.randperm(len(owner), generator=g).tolist()]
        mem2 = {}
        for l2, m in zip(lab, perm):
            mem2.setdefault(l2, Counter())[m] += 1
        for m in null:
            null[m].append(sum(1 for c, cnt in mem2.items()
                               if m in cnt and len(cnt) == 1))
    print()
    print(f"{'mode':12s} {'unique obs':>11} {'unique null':>16} {'excess':>9}")
    for m in sorted(obs):
        nl = torch.tensor(null[m], dtype=torch.float32)
        mu, sd = float(nl.mean()), float(nl.std())
        z = (obs[m] - mu) / max(sd, 1e-9)
        print(f'{m:12s} {obs[m]:11,d} {mu:11.1f}+-{sd:<4.1f} {obs[m] - mu:+9.1f}  (z={z:+.1f})')
    print()
    print('excess near zero => the modes sampled the SAME region and the raw '
          'unique counts are an artifact of a sparse pool.')

    #: DISPERSION-FREE view. The cluster-uniqueness excess above conflates
    #: 'went somewhere else' with 'produced more scattered output' -- the null
    #: permutes labels, so it cannot reproduce a mode's OWN dispersion, and the
    #: modes differ ~2x in structures-per-cluster. This asks a per-structure
    #: question instead: how far is the nearest structure belonging to ANY
    #: OTHER mode? Scatter within a mode does not affect it.
    Dt = torch.as_tensor(D)
    own = torch.tensor([sorted(set(owner)).index(m) for m in owner])
    print()
    print(f"{'mode':12s} {'median x-mode NN':>17} {'90th pct':>10} "
          f"{'% beyond cut':>13}")
    for k2, m in enumerate(sorted(set(owner))):
        rows = (own == k2).nonzero().flatten()
        sub = Dt[rows][:, (own != k2).nonzero().flatten()]
        nn = sub.min(dim=1).values
        q = torch.quantile(nn, torch.tensor([0.5, 0.9]))
        print(f'{m:12s} {float(q[0]):17.4f} {float(q[1]):10.4f} '
              f'{100 * float((nn > cli.cut).float().mean()):12.1f}%')
    print()
    print('a mode with a LARGER cross-mode nearest-neighbour distance is '
          'genuinely sampling where the others do not.')


if __name__ == '__main__':
    main()
