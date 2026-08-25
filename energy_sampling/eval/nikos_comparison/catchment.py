"""
How big is the catchment of a known form, and does the search ever start inside it?

"Random starts never reach ACRDIN07/06" has two very different explanations that
look identical from the endpoints:

  A. The BASIN IS TINY. Near-fatal for the approach -- the landscape then maps where
     rprop-from-random lands, not where the energy minima are, and a generative
     model trained on those outputs inherits the same blind spot.
  B. The INITIALISER NEVER PROPOSES anything nearby. Entirely fixable by reshaping
     the starting distribution.

This measures both.

CATCHMENT (A). The seeded arms are already the experiment: 40 copies at each of six
noise levels (log_noise -3.0 .. -0.5) around each anchor, with anchor and level
recorded in `seeds_sg14_zp2_meta.pt`. Displace, relax, and see what comes back. The
x-axis is the MEASURED latent displacement of each seed from its anchor, not the
nominal noise level -- the mapping from one to the other is not assumed.

REACH (B). Trajectory files store step-0 parameters, i.e. the initialiser's actual
output. Compare where the known forms sit against that distribution.

Return is judged at RDF <= 0.10 -- see RETURN_CUT for why 0.05 is the WRONG cut
here even though that is where a confirmed match sits.

    python -m energy_sampling.eval.nikos_comparison.catchment
"""
import argparse
import glob
import os
import re

import torch
import tqdm

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, chunk_by_cluster_cost, known_form_rdfs, load_arms, physical)

#: NOT 0.05. The optimiser relaxes even the EXACT structure to ~0.046-0.050
#: away from its own reference, so a 0.05 cut sits directly on the mass of the
#: returned population and its fraction becomes threshold noise -- ACRDIN07's
#: unnoised seed lands at 0.0495 and scored 4/40 at the smallest noise level,
#: which is an artifact, not a narrow basin. 0.10 sits in the empty gap between
#: the returned cluster (~0.05) and a non-match (0.13+).
RETURN_CUT = 0.10
FORMS = ['ACRDIN07', 'ACRDIN06']


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


def latent_of(lst):
    out = []
    for c in lst:
        b = collate_data_list([c.clone()],
                              exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        out.append(b.latent_params()[0])
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sg', type=int, default=14)
    ap.add_argument('--zp', type=int, default=2)
    ap.add_argument('--init-files', type=int, default=60,
                    help='trajectory files to sample the initialiser '
                         'from; 0 = all. The MINIMUM distance shrinks '
                         'with sample count, so a subsample gives an '
                         'UPPER bound on the closest approach.')
    cli = ap.parse_args()

    tag = f'sg{cli.sg}_zp{cli.zp}'
    meta = torch.load(os.path.join(ROOT, f'seeds_{tag}_meta.pt'),
                      weights_only=False, map_location='cpu')
    shards = sorted(glob.glob(os.path.join(ROOT, f'seeds_{tag}_[0-9]*.pt')),
                    key=lambda p: int(re.search(r'_(\d+)\.pt$', p).group(1)))
    counts = [len(torch.load(p, weights_only=False, map_location='cpu'))
              for p in shards]
    print(f"{len(meta)} seeds across {len(shards)} shards {counts}")

    #: INDEX ALIGNMENT IS ONLY SAFE WHERE THE ARM RETURNED ITS WHOLE SHARD. Some
    #: arms lose samples, and a silent off-by-N would mislabel every seed's noise
    #: level. Align the arms that match exactly and SAY how many were dropped.
    groups = load_arms(os.path.join(ROOT, 'opt_outs'),
                       f'aug21seed_acridine_{tag}_*.pt')
    stem = f'aug21seed_acridine_{tag}'
    arms = dict(sorted(groups[stem]))
    off, pairs, skipped = 0, [], []
    for k, n in enumerate(counts):
        got = arms.get(k)
        if got is None or len(got) != n:
            skipped.append((k, n, 0 if got is None else len(got)))
        else:
            for j, c in enumerate(got):
                pairs.append((c, meta[off + j]))
        off += n
    print(f"aligned {len(pairs):,} seeds; skipped arms (returned != shard size): "
          f"{skipped if skipped else 'none'}")
    if not pairs:
        raise SystemExit("no arm returned a full shard; cannot align")

    keep, drop = physical([c for c, _ in pairs])
    kept = {id(c) for c in keep}
    pairs = [(c, m) for c, m in pairs if id(c) in kept]
    print(f"{len(pairs):,} physical ({drop})\n")

    names, form_rdf = known_form_rdfs(FORMS)
    anchors = torch.load(os.path.join(ROOT, 'std_opt_acridine_polymorphs.pt'),
                         weights_only=False, map_location='cpu').cpu()
    al, aid = anchors.batch_to_list(), list(anchors.identifier)
    a_lat = {n: latent_of([al[aid.index(n)]])[0] for n in names}

    outs = [c for c, _ in pairs]
    r = rdf_of(outs)
    bins = torch.linspace(0, 10, r.shape[-1])
    d_form = {n: compute_rdf_distance(form_rdf[k], r, bins)
              for k, n in enumerate(names)}
    seed_lat = latent_of([c for c, _ in pairs])

    rows = {}
    for i, (c, m) in enumerate(pairs):
        a, lv = m['anchor'], m['log_noise']
        if a not in d_form:
            continue
        disp = float((seed_lat[i] - a_lat[a]).norm())
        rows.setdefault((a, lv), []).append(
            (float(d_form[a][i]), disp, float(c.mace)))

    print(f"return judged at RDF <= {RETURN_CUT} (a 20/20 COMPACK match is "
          f"0.044-0.049)\n")
    print(f"{'anchor':10s} {'log_noise':>9} {'seed disp':>10} {'n':>4} "
          f"{'returned':>9} {'median RDF':>11} {'best RDF':>9}")
    for a in names:
        for lv in [None, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5]:
            v = rows.get((a, lv))
            if not v:
                continue
            dd = torch.tensor([x[0] for x in v])
            ee = torch.tensor([x[2] for x in v])
            ret = int((dd <= RETURN_CUT).sum())
            print(f"{a:10s} {str(lv):>9} {float(torch.tensor([x[1] for x in v]).median()):10.4f} {len(v):4d} "
                  f"{ret:4d}/{len(v):<4d} {float(dd.median()):11.4f} "
                  f"{float(dd.min()):9.4f}")
        print()

    #: PART B -- does the INITIALISER ever propose anything inside that
    #: catchment? Step 0 of each trajectory IS the initialiser's output. Convert
    #: a sample of them to latent space (the same space the seed displacements
    #: are measured in) and compare the closest approach to the catchment radius.
    init = []
    tf = sorted(glob.glob(os.path.join(ROOT, 'opt_outs',
                                       'aug21*_traj*.pt')))
    tf = [f for f in tf if 'seed' not in os.path.basename(f)]
    if cli.init_files:
        tf = tf[:cli.init_files]
    for f in tqdm.tqdm(tf, desc='inits', leave=False):
        rec = torch.load(f, weights_only=False, map_location='cpu')
        if not (isinstance(rec, dict) and 'params' in rec):
            continue
        base = rec['base_crystal']
        pr = rec['params'][0]              # step 0 = the initialiser's proposal
        b = collate_data_list([base.clone() for _ in range(len(pr))],
                              exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        b.set_cell_parameters(pr, skip_box_analysis=False)
        init.append(b.latent_params())
        del rec, b
    init = torch.cat(init)
    torch.save(init, os.path.join(ROOT, 'nikos_comparison',
                                  f'init_latents_{tag}.pt'))
    print(f'{len(init):,} initialiser proposals sampled from '
          f'{len(tf)} trajectory files')
    print()
    print(f"{'anchor':10s} {'closest init':>13} {'median init':>12}")
    for a in names:
        d = (init - a_lat[a][None, :]).norm(dim=-1)
        print(f'{a:10s} {float(d.min()):13.4f} {float(d.median()):12.4f}')

    #: THE TAIL is what matters, not the minimum. A single closest draw says
    #: nothing about how much mass sits inside the catchment, and the hit rate is
    #: (mass inside) x (return probability there). Report the low quantiles and the
    #: fraction under each displacement where return was actually measured.
    print()
    print('fraction of proposals within a given latent distance of the anchor:')
    print(f"{'anchor':10s} " + ' '.join(f'{r:>9}' for r in
                                        ('<0.35', '<0.70', '<0.90', '<1.20', '<2.00')))
    for a in names:
        d = (init - a_lat[a][None, :]).norm(dim=-1)
        cells = []
        for thr in (0.35, 0.70, 0.90, 1.20, 2.00):
            f = float((d < thr).float().mean())
            cells.append('0' if f == 0 else f'{f:.2e}')
        print(f'{a:10s} ' + ' '.join(f'{c:>9}' for c in cells))
    print()
    print('measured return rate at those displacements: ~50% at 0.35, '
          '7-30% at 0.69-0.92.')
    print('expected hits = n_samples x P(within) x P(return | within).')
    print()
    print('compare CLOSEST INIT against the seed displacement at which return '
          'fails.')
    print('closest init >> that radius -> the initialiser never proposes '
          'anything near the basin (FIXABLE).')
    print('closest init <  that radius -> starts land inside and still do not '
          'converge there (BASIN IS THE PROBLEM).')

    print("a return fraction that stays high to large displacement = a WIDE "
          "catchment,\nand then the miss is the INITIALISER, not the basin.")


if __name__ == '__main__':
    main()
