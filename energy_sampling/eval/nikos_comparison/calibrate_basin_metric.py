"""
What RDF distance actually means, per RDF MODE, against COMPACK.

Everything quantitative in the sg14-Z'2 analysis -- basin counts, coverage,
catchment radii, "the trajectories never came close" -- rests on a threshold
separating "same packing" from "different packing" in RDF distance. That threshold
was set from TWO anecdotes at the tightest possible end (seeded outputs, i.e. the
same structure barely relaxed) and it is WRONG in a known direction:

    ACRDIN04   RDF 0.0685  COMPACK 20/20  RMSD 0.170
    ACRDIN12   RDF 0.1055  COMPACK 20/20  RMSD 0.283
    nik00000   RDF 0.1645  COMPACK 20/20  RMSD 0.477      <- I called this a non-match
    ACRDIN07   RDF 0.1466  COMPACK 10/20  RMSD 0.404      <- I called this a clean miss

So confirmed matches span RDF 0.044-0.165 and RMSD 0.14-0.48. Cutting at 0.05-0.15
splits genuine duplicates and inflates every basin count.

TWO THINGS THIS MEASURES.

  1. P(match) as a function of RDF distance, stratified to 0.30 -- BEYOND the old
     cutoff, because matches may be hiding there (they were).
  2. The same, per RDF MODE. `envwise` merges 2-WL symmetry classes, so packings
     differing in registry can look identical to it; `atomwise` distinguishes every
     atom-index pair; `elementwise` is coarsest. They correlate but not always, and
     the right basin metric is whichever separates matched from unmatched most
     cleanly. This has been `envwise` all along by inheritance, never by test.

The pool deliberately mixes SEEDED outputs (which sit close to the known forms, and
supply the near bins) with a random unseeded sample (which supplies the far bins).

    python -m energy_sampling.eval.nikos_comparison.calibrate_basin_metric
"""
import argparse
import os

import torch

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.compare import compack_confirm
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, chunk_by_cluster_cost, harmonize, load_arms, physical)

#: envwise is the DEFAULT for a reason: acridine has a C2 axis, and atomwise fixes
#: the atom indexing, so packings IDENTICAL UNDER C2 land far apart in atomwise and
#: its P(match) curve is non-monotonic. That is a property of the MOLECULE, not of
#: Z', so it does not go away at Z'=1. Atomwise is worth trying only on highly
#: asymmetric molecules. Kept in the list so the check is repeatable, not assumed.
MODES = ['envwise', 'elementwise', 'atomwise']
BINS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30)]
REFS = ['ACRDIN07', 'ACRDIN06']
WORK = os.path.join(ROOT, 'nikos_comparison', '_calibration')


def rdf_of(lst, mode):
    """RDFs for one mode. Returns None if the mode cannot run on this batch."""
    out = []
    kw = dict(rdf_mode=mode, cutoff=10, rdf_cutoff=10)
    for lo, hi in chunk_by_cluster_cost(lst, 1_500_000):
        b = collate_data_list([c.clone() for c in lst[lo:hi]],
                              exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        b.aunit_handedness = b.aunit_handedness.abs()
        with torch.no_grad():
            o = b.analyze(['rdf'], assign_outputs=False, **kw)
        r = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
        out.append(r.cpu())
        del b, o, r
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--per-bin', type=int, default=10,
                    help='candidates COMPACKed per reference per RDF bin')
    ap.add_argument('--unseeded', type=int, default=2000)
    ap.add_argument('--source', default='opt_outs',
                    help="'opt_outs' (aug21 wave, has seeded arms) or "
                         "'prior_chunks' (the may_ raw searches)")
    ap.add_argument('--pattern', default='aug21*_acridine_sg14_zp2_*.pt')
    ap.add_argument('--refs', nargs='+', default=REFS)
    ap.add_argument('--seeds', default=None,
                    help="glob of seed shards to ADD to the pool, e.g. "
                         "'seeds_sg14_zp1_[0-9]*.pt'. A random draw from the "
                         "landscape cannot populate the near bins when the "
                         "landscape's own closest structure to a form is already "
                         "at RDF 0.10 -- noised copies of the forms can. They are "
                         "UNRELAXED, so this calibrates the RDF-to-COMPACK relation "
                         "at small distance, not the optimiser's behaviour.")
    cli = ap.parse_args()
    refs_names = cli.refs

    g = torch.Generator().manual_seed(0)
    groups = load_arms(os.path.join(ROOT, cli.source), cli.pattern)
    seeded, unseeded = [], []
    for stem in sorted(groups):
        flat, _ = physical([c for _, lst in sorted(groups[stem]) for c in lst])
        (seeded if 'seed' in stem else unseeded).extend(flat)
    unseeded = [unseeded[i] for i in
                torch.randperm(len(unseeded), generator=g)[:cli.unseeded].tolist()]
    if cli.seeds:
        import glob as _glob
        extra = []
        for f in sorted(_glob.glob(os.path.join(ROOT, cli.seeds))):
            lst = torch.load(f, weights_only=False, map_location='cpu')
            extra.extend(lst if isinstance(lst, list) else lst.batch_to_list())
        extra, edrop = physical(extra)
        print(f"   + {len(extra):,} seed structures from {cli.seeds} ({edrop})")
        seeded = seeded + extra
    pool = harmonize(seeded + unseeded)
    print(f"pool {len(pool):,} = {len(seeded):,} seeded + {len(unseeded):,} unseeded"
          f"  (source {cli.source}, refs {refs_names})")

    poly = torch.load(os.path.join(ROOT, 'std_opt_acridine_polymorphs.pt'),
                      weights_only=False, map_location='cpu').cpu()
    pl, pids = poly.batch_to_list(), list(poly.identifier)
    refs = [pl[pids.index(n)] for n in refs_names]

    #: distances in every mode, for the SAME pairs, so the modes are comparable
    dist = {}
    for mode in MODES:
        try:
            r_pool = rdf_of(pool, mode)
            r_ref = rdf_of(refs, mode)
        except Exception as e:
            print(f"   mode {mode}: FAILED ({type(e).__name__}: {e}) -- skipped")
            continue
        bins = torch.linspace(0, 10, r_pool.shape[-1])
        dist[mode] = torch.stack([compute_rdf_distance(r_ref[k], r_pool, bins)
                                  for k in range(len(refs))])
        print(f"   mode {mode}: {r_pool.shape[1]} channels, "
              f"distance range {float(dist[mode].min()):.4f}-"
              f"{float(dist[mode].max()):.4f}")
    if 'envwise' not in dist:
        raise SystemExit("envwise failed; nothing to stratify on")

    #: stratify on envwise (the incumbent), then report ALL modes for those pairs
    picks = []
    for k, name in enumerate(refs_names):
        d = dist['envwise'][k]
        for lo, hi in BINS:
            idx = ((d >= lo) & (d < hi)).nonzero().flatten()
            if not len(idx):
                print(f"   {name} bin [{lo},{hi}): empty")
                continue
            take = idx[torch.randperm(len(idx), generator=g)[:cli.per_bin]]
            for j in take.tolist():
                picks.append((k, j, lo, hi))
    print(f"\n{len(picks)} pairs to COMPACK")

    need = sorted({j for _, j, _, _ in picks})
    remap = {gj: i for i, gj in enumerate(need)}
    sub = collate_data_list([pool[gj].clone() for gj in need])
    q = collate_data_list([c.clone() for c in refs])
    q.aunit_handedness = q.aunit_handedness.abs()
    per_ref = {k: [p for p in picks if p[0] == k] for k in range(len(refs_names))}
    width = max(len(v) for v in per_ref.values())
    nb = torch.zeros((len(refs_names), width), dtype=torch.long)
    for k, v in per_ref.items():
        for i, (_, j, _, _) in enumerate(v):
            nb[k, i] = remap[j]
    rmsds, matched = compack_confirm(q, sub, nb, WORK)

    #: CACHE THE COMPACK RESULTS. They are the expensive half and the analysis
    #: over them (binning by a different mode, moving a threshold) is free --
    #: not caching them meant re-running 84 similarity comparisons to ask a
    #: second question of the same pairs.
    recs = []
    for k, v in per_ref.items():
        for i2, (_, j, blo, bhi) in enumerate(v):
            recs.append(dict(ref=refs_names[k], pool_idx=j, bin_lo=blo,
                             n_matched=int(matched[k, i2]),
                             rmsd=float(rmsds[k, i2]),
                             **{f'd_{m}': float(dist[m][k, j]) for m in dist}))
    torch.save(recs, os.path.join(ROOT, 'nikos_comparison',
                                  'basin_metric_calibration.pt'))
    print(f'cached {len(recs)} COMPACK results')

    #: P(match) binned by EACH mode's own distance, on the same pairs
    for mode in dist:
        vals = torch.tensor([r[f'd_{mode}'] for r in recs])
        okm = torch.tensor([r['n_matched'] for r in recs])
        fail = torch.tensor([(r['n_matched'] == 0 and r['rmsd'] == 0.0)
                             for r in recs])
        edges = torch.quantile(vals, torch.linspace(0, 1, 7))
        print()
        print(f'P(20/20) binned by {mode} distance:')
        print(f"{'range':>18} {'n':>4} {'20/20':>7} {'median matched':>15}")
        for a, b in zip(edges[:-1], edges[1:]):
            m = (vals >= a) & (vals <= b) & (~fail)
            if int(m.sum()) < 3:
                continue
            mm = okm[m].float()
            print(f'[{float(a):.4f},{float(b):.4f}] {int(m.sum()):4d} '
                  f'{100*float((mm >= 20).float().mean()):6.0f}% '
                  f'{float(mm.median()):15.1f}')

    print(f"\nP(COMPACK 20/20) by envwise RDF bin:")
    print(f"{'bin':>14} {'n':>4} {'20/20':>7} {'>=15/20':>9} {'median n_matched':>17} "
          f"{'median rmsd':>12}")
    rows = []
    for lo, hi in BINS:
        m, r = [], []
        for k, v in per_ref.items():
            for i, (_, j, blo, _) in enumerate(v):
                if blo != lo:
                    continue
                mm, rr = int(matched[k, i]), float(rmsds[k, i])
                if mm == 0 and rr == 0.0:      # engine failure, NOT a zero-RMSD hit
                    continue
                m.append(mm); r.append(rr); rows.append((lo, j, mm, rr))
        if not m:
            continue
        mt = torch.tensor(m, dtype=torch.float32)
        rt = torch.tensor(r)
        print(f"[{lo:.2f},{hi:.2f}) {len(m):4d} "
              f"{100*float((mt >= 20).float().mean()):6.0f}% "
              f"{100*float((mt >= 15).float().mean()):8.0f}% "
              f"{float(mt.median()):17.1f} {float(rt.median()):12.3f}")

    #: which mode separates matched from unmatched best?
    print(f"\nseparation by mode (median distance, matched vs not):")
    print(f"{'mode':>12} {'d(20/20)':>10} {'d(<20/20)':>11} {'gap':>8}")
    ok = torch.zeros(len(picks), dtype=torch.bool)
    dvals = {mode: torch.zeros(len(picks)) for mode in dist}
    p_i = 0
    for k, v in per_ref.items():
        for i, (_, j, _, _) in enumerate(v):
            mm, rr = int(matched[k, i]), float(rmsds[k, i])
            ok[p_i] = (mm >= 20)
            for mode in dist:
                dvals[mode][p_i] = dist[mode][k, j]
            p_i += 1
    for mode in dist:
        a, b = dvals[mode][ok], dvals[mode][~ok]
        if not len(a) or not len(b):
            print(f"{mode:>12}   (one class empty)")
            continue
        print(f"{mode:>12} {float(a.median()):10.4f} {float(b.median()):11.4f} "
              f"{float(b.median() - a.median()):8.4f}")
    print("\nlargest gap = the mode whose distance best tracks packing identity.")


if __name__ == '__main__':
    main()
