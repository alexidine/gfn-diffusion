"""
Summarise the sampled states from a search battery wave.

Answers, per sampling mode:
  * how much landed, and how much of what was asked for
  * where the energies sit, against the PREVIOUS battery's minimum
  * whether the known forms for this SG/Z' were found this time

USE THE STORED `mace`, DO NOT RECOMPUTE. The search writes a per-molecule lattice
energy on the same scale as the acr_production priors (~-48 to -63 kJ/mol).
Recomputing the same structure through `MolCrystalData.analyze(['mace'])` instead
returns roughly -11899 -- a constant 11836.127 kJ/mol per molecule apart, the
atomic-E0 sum counted in the crystal leg and omitted from the gas leg. Mixing the
two scales silently produces nonsense, so this reads what the search stored and
never recomputes an energy.

Form-matching is RDF distance only, as a SCREEN. RDF ranks candidates; COMPACK
decides identity, and anything that looks close here should be confirmed with
`landscape_check.py --prior <stem> --raw` before it is believed.

    python -m energy_sampling.eval.nikos_comparison.summarize_search \
        --out-dir D:\crystal_datasets\acridine\opt_outs
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import torch
import tqdm

from torch_geometric.utils import scatter

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.crystal_building.utils import get_cart_translations
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS

ROOT = os.path.join('D:', os.sep, 'crystal_datasets', 'acridine')
#: <stem>_<arm>.pt, arm all digits -- the same shape load_search_chunks requires
ARM_FILE = re.compile(r'^(?P<stem>.+?)_(?P<arm>\d+)\.pt$')
#: save_trajs writes '<run_name>_traj<batch_idx>_<opt_ind>.pt', which ALSO ends in
#: _<digits>.pt and would otherwise be inventoried as an arm named '<run>_trajN'.
#: They hold optimizer traces, not crystals, so they are excluded here.
#: (`load_search_chunks` is safe from this only because '<run>_trajN_0' has a
#: non-digit tail relative to the real stem.)
TRAJ_FILE = re.compile(r'_traj\d+_\d+\.pt$')

#: what the PREVIOUS battery reached, for reference (stored scale, kJ/mol)
PREV_MIN = {(14, 1): -62.81, (14, 2): -62.80, (9, 2): -61.72}

#: known forms per SG/Z', and how far the old search got from each
KNOWN = {(14, 2): ['ACRDIN07', 'ACRDIN06'],
         (14, 1): ['ACRDIN04', 'ACRDIN12'],
         (9, 2): ['ACRDIN05', 'ACRIDIN_VIII'],
         (19, 3): ['ACRDIN08']}
PREV_BEST_RDF = {'ACRDIN07': 0.1428, 'ACRDIN06': 0.1394,
                 'ACRDIN04': 0.0685, 'ACRDIN12': 0.1055,
                 'ACRDIN05': 0.0603, 'ACRIDIN_VIII': 0.0871}


def load_arms(out_dir, pattern):
    """{stem: [(arm, [crystals])]} for everything matching."""
    groups = defaultdict(list)
    n_traj = 0
    for path in sorted(glob.glob(os.path.join(out_dir, pattern))):
        name = os.path.basename(path)
        if TRAJ_FILE.search(name):
            n_traj += 1
            continue
        m = ARM_FILE.match(name)
        if not m:
            continue
        try:
            data = torch.load(path, weights_only=False, map_location='cpu')
        except Exception as e:
            print(f"   unreadable {os.path.basename(path)}: {type(e).__name__}")
            continue
        lst = data if isinstance(data, list) else data.batch_to_list()
        groups[m['stem']].append((int(m['arm']), lst))
    if n_traj:
        print(f"skipped {n_traj} trajectory files (*_traj<n>_<n>.pt)")
    return groups


#: LJ HERE PENALISES HYDROGEN BONDS, so a mildly positive lj is not a verdict --
#: only a very large one is. Set well above any bonded structure. On the sg14-Z'2
#: wave the choice is immaterial: see `physical`.
LJ_BLOWN_UP = 1e4


def physical(lst, lj_max=LJ_BLOWN_UP):
    """
    Drop structures the search produced that are not crystals.

    The criterion of record is collate_prior.py:63, applied when a search wave is
    turned into a prior:

        (angular_factor > 0.1) & (vdw_max < 1.5)

    with angular_factor = cell_volume / prod(cell_lengths). Only the first half is
    reproducible here -- the search does not store vdw_max -- so a very large `lj`
    stands in for the overlap half.

    MEASURED on the sg14-Z'2 wave (84,387 samples), lj is bimodal with a hole
    between the two modes:

        lj < -100          71,927     median stored mace   -43
        -100 .. 1e4             29     median stored mace  +130
        lj > 1e4            12,431     median stored mace  +553, running to +8215

    Nothing sits at mildly positive lj, so no hydrogen-bonded structure is at risk
    here and EVERY threshold from 0 to 1e4 drops the same samples to within 29. The
    cut is set high anyway, because that is the criterion that generalises; on a
    molecule with real donors the band between -100 and 1e4 would be populated and
    a cut at zero WOULD throw away crystals. Re-read this table before reusing the
    threshold on another system.

    angular_factor drops ZERO samples on this wave, so the reproduced half of the
    real criterion is doing none of the work -- do not read a pass here as having
    applied collate_prior's filter.

    This is about the STATISTICS being over crystals, not about memory. An earlier
    version blamed these structures for the `MemoryError` in the RDF pass; that was
    wrong. See `chunk_by_cluster_cost`.
    """
    keep, dropped = [], {'degenerate_cell': 0, 'blown_up': 0, 'nonfinite': 0,
                         'NO_LJ_overlap_check_skipped': 0}
    for c in lst:
        L = c.cell_lengths.flatten()
        vol = float(c.cell_volume)
        #: SEED / unscored structures carry no `lj` -- they were never put through
        #: an energy pass. Count them explicitly instead of crashing OR silently
        #: waving them through: the overlap half of the filter did not run on them,
        #: and a caller reading only `len(keep)` must be able to see that.
        if 'lj' not in c.keys():
            dropped['NO_LJ_overlap_check_skipped'] += 1
            if torch.isfinite(L).all() and vol / max(float(L.prod()), 1e-9) > 0.1:
                keep.append(c)
            continue
        lj = float(c.lj)
        if not (torch.isfinite(torch.tensor(lj)) and torch.isfinite(L).all()):
            dropped['nonfinite'] += 1
            continue
        if vol / max(float(L.prod()), 1e-9) <= 0.1:
            dropped['degenerate_cell'] += 1
            continue
        if lj >= lj_max:
            dropped['blown_up'] += 1
            continue
        keep.append(c)
    return keep, dropped


def harmonize(lst):
    """
    Reduce a mixed pool to the keys EVERY member carries.

    Search outputs are not key-uniform: `rdf_bins` and `vdw_max` appear on some
    arms and not others, and on this wave they vary WITHIN the seeded stem (4 of
    its 6 shards carry vdw_max, the rest do not). `Batch.from_data_list` then dies
    with a bare `KeyError: <name>`, which reads as a corrupt file rather than a
    schema mismatch. Excluding names one at a time is whack-a-mole; take the
    intersection instead.
    """
    if not lst:
        raise ValueError("harmonize() got an empty list -- nothing to intersect. "
                         "The bare set.intersection() failure this replaces read "
                         "as a torch/PyG bug rather than an empty caller.")
    common = set.intersection(*[set(c.keys()) for c in lst])
    out = []
    for c in lst:
        d = c.clone()
        for k in set(d.keys()) - common:
            del d[k]
        out.append(d)
    return out


def stats(lst):
    e = torch.tensor([float(c.mace) for c in lst])
    pc = torch.tensor([float(c.packing_coeff) for c in lst])
    q = torch.tensor([0., .01, .25, .5, 1.])
    return dict(n=len(lst), e=e, pc=pc, eq=torch.quantile(e, q).tolist(),
                pcq=torch.quantile(pc, q).tolist())


def known_form_rdfs(names, device='cpu'):
    """RDFs for the named forms, from the like-for-like (opt conformer) file."""
    poly = torch.load(os.path.join(ROOT, 'std_opt_acridine_polymorphs.pt'),
                      weights_only=False, map_location='cpu').cpu()
    pl, pids = poly.batch_to_list(), list(poly.identifier)
    want = [pids.index(n) for n in names if n in pids]
    sub = collate_data_list([pl[i].clone() for i in want], exclude_keys=['rdf'])
    sub.aunit_handedness = sub.aunit_handedness.abs()
    with torch.no_grad():
        o = sub.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
    r = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
    return [pids[i] for i in want], r


def cluster_atoms(part, cutoff=10.0, supercell_size=10):
    """Atoms each sample's periodic cluster will hold, without instantiating one."""
    b = collate_data_list([c.clone() for c in part],
                          exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
    cc = scatter(b.pos, b.batch, dim_size=b.num_graphs, dim=0, reduce='mean')
    _, cells = get_cart_translations(cc, b.T_fc, b.radius, cutoff, supercell_size)
    return (cells * b.num_atoms * b.sym_mult).long().tolist()


def chunk_by_cluster_cost(lst, budget=1_500_000):
    """
    Chunk boundaries set by cluster ATOMS, not by sample count.

    `analyze(['rdf'])` explodes every crystal into a periodic cluster before the
    radial graph, so a chunk costs sum(cluster atoms), which varies ~10x per sample
    (median 62k, max 243k on the sg14-Z'2 wave). At a flat 200 samples/chunk that
    is ~16M atoms per chunk and `torch_cluster.radius` raised
    `MemoryError: bad allocation` -- on an ordinary chunk, with NO pathological
    structure in it. The worst single sample in the chunk that died was 241k atoms
    against a population 99.9th percentile of 243k.

    So the fix is the chunking, not a filter. Group to a fixed atom budget and every
    chunk costs the same regardless of what cells the search happened to produce.
    """
    costs = []
    for s in range(0, len(lst), 200):        # the sizing pass is bounding-box only
        costs.extend(cluster_atoms(lst[s:s + 200]))
    chunks, start, run = [], 0, 0
    for i, c in enumerate(costs):
        if run and run + c > budget:
            chunks.append((start, i))
            start, run = i, 0
        run += c
    chunks.append((start, len(lst)))
    return chunks


def best_rdf_to_forms(lst, form_rdf, budget=1_500_000, device='cpu'):
    """Smallest RDF distance from any sample to each form. Streams by cost."""
    best = torch.full((len(form_rdf),), float('inf'))
    best_i = [-1] * len(form_rdf)
    bins = None
    chunks = chunk_by_cluster_cost(lst, budget)
    for lo, hi in tqdm.tqdm(chunks, desc='   rdf', leave=False):
        b = collate_data_list([c.clone() for c in lst[lo:hi]],
                              exclude_keys=['rdf', 'fingerprint', 'rdf_bins']).to(device)
        with torch.no_grad():
            o = b.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
        r = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
        r = r.cpu()
        if bins is None:
            bins = torch.linspace(0, 10, r.shape[-1])
        for k in range(len(form_rdf)):
            d = compute_rdf_distance(form_rdf[k], r, bins)
            j = int(d.argmin())
            if float(d[j]) < best[k]:
                best[k] = float(d[j])
                best_i[k] = lo + j
        del b, o, r
    return best, best_i


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'opt_outs'))
    ap.add_argument('--pattern', default='*.pt')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--no-forms', action='store_true',
                    help='skip the RDF form screen (the slow part)')
    cli = ap.parse_args()

    groups = load_arms(cli.out_dir, cli.pattern)
    if not groups:
        raise SystemExit(f"nothing matching {cli.pattern} in {cli.out_dir}")

    print(f"{'stem':40s} {'arms':>4} {'samples':>8} {'min E':>8} {'1%':>8} "
          f"{'median':>8} {'pc med':>7}")
    print('-' * 88)
    summary = {}
    drops = {}
    for stem in sorted(groups):
        arms = sorted(groups[stem])
        raw = [c for _, lst in arms for c in lst]
        flat, dropped = physical(raw)
        drops[stem] = (len(raw), dropped)
        if not flat:
            continue
        st = stats(flat)
        summary[stem] = (arms, flat, st)
        print(f"{stem:40s} {len(arms):4d} {st['n']:8,d} {st['eq'][0]:8.2f} "
              f"{st['eq'][1]:8.2f} {st['eq'][3]:8.2f} {st['pcq'][3]:7.3f}")

    #: the denominator every number above is taken over. Say what left it.
    print(f"\nnon-physical samples removed before all of the above:")
    for stem, (n_raw, d) in sorted(drops.items()):
        tot = sum(d.values())
        print(f"   {stem:40s} {tot:6,d} of {n_raw:7,d} "
              f"({100 * tot / max(n_raw, 1):4.1f}%)  {d}")

    #: per-arm yield, since acr_production lost 14-43% of nominal
    print(f"\nper-arm yield:")
    for stem, (arms, flat, st) in sorted(summary.items()):
        y = sorted(len(physical(l)[0]) for _, l in arms)
        print(f"   {stem:40s} min {y[0]:5d}  median {y[len(y)//2]:5d}  "
              f"max {y[-1]:5d}   ({len(arms)} arms)")

    #: energy, against what the previous battery reached
    print(f"\nlowest energy reached vs the previous battery:")
    for stem, (arms, flat, st) in sorted(summary.items()):
        c0 = flat[0]
        combo = (int(c0.sg_ind), int(c0.z_prime))
        prev = PREV_MIN.get(combo)
        delta = f"{st['eq'][0] - prev:+.2f} vs prev {prev:.2f}" if prev else "no prev"
        print(f"   {stem:40s} {st['eq'][0]:8.2f}   {delta}")

    if cli.no_forms:
        return

    #: THE question -- did any mode land the known forms this time?
    print(f"\nclosest approach to the known forms (RDF screen; COMPACK decides):")
    cache = {}
    for stem, (arms, flat, st) in sorted(summary.items()):
        c0 = flat[0]
        combo = (int(c0.sg_ind), int(c0.z_prime))
        names = KNOWN.get(combo)
        if not names:
            print(f"   {stem}: no known form for sg{combo[0]} Z'={combo[1]}")
            continue
        if combo not in cache:
            cache[combo] = known_form_rdfs(names, cli.device)
        got, form_rdf = cache[combo]
        best, best_i = best_rdf_to_forms(flat, form_rdf, device=cli.device)
        for k, name in enumerate(got):
            prev = PREV_BEST_RDF.get(name)
            mark = ''
            if prev is not None:
                mark = (' CLOSER than before' if float(best[k]) < prev - 1e-4
                        else ' no closer')
            e_at = float(flat[best_i[k]].mace) if best_i[k] >= 0 else float('nan')
            print(f"   {stem:40s} {name:14s} {float(best[k]):.4f}"
                  + (f" (prev {prev:.4f}{mark})" if prev else "")
                  + f"  [that sample: {e_at:.2f} kJ/mol]")

    print("\nRDF is a SCREEN. Confirm anything that looks close with "
          "landscape_check.py --prior <stem> --raw before believing it.")


if __name__ == '__main__':
    main()
