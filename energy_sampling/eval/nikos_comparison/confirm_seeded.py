"""
Confirm, by COMPACK, what the seeded arms actually produced.

The RDF screen in `summarize_search.py` says the seeded arms reached 0.0489 /
0.0438 of ACRDIN07 / ACRDIN06 while every unseeded mode stayed at 0.13-0.17. RDF
ranks; COMPACK decides, and COMPACK is what gets published. This takes the k
nearest seeded outputs to each known form and runs the 20-molecule packing
comparison against it.

Read the OUTPUT carefully: `compack_confirm` returns 0 matched / 0.0 RMSD when the
similarity engine FAILS, which is a failure to compare and not a perfect hit. A
row of 20/20 is a match; a row of 0 with rmsd exactly 0.0 is a broken comparison.

The __main__ guard is REQUIRED: batch_compack spawns mp.Pool and on Windows each
child re-imports this module.

    python -m energy_sampling.eval.nikos_comparison.confirm_seeded
"""
import argparse
import os

import torch

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.compare import compack_confirm
from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, harmonize, known_form_rdfs, load_arms, physical)

WORK = os.path.join(ROOT, 'nikos_comparison', '_seeded_compack')


def rdf_of(lst, budget=1_500_000):
    from energy_sampling.eval.nikos_comparison.summarize_search import (
        chunk_by_cluster_cost)
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
    ap.add_argument('--stem', default='aug21seed_acridine_sg14_zp2')
    ap.add_argument('--forms', nargs='+', default=['ACRDIN07', 'ACRDIN06'])
    ap.add_argument('-k', type=int, default=5,
                    help='nearest outputs per form to COMPACK')
    cli = ap.parse_args()

    groups = load_arms(cli.out_dir, cli.stem + '_*.pt')
    flat, dropped = physical([c for _, lst in sorted(groups[cli.stem]) for c in lst])
    print(f"{cli.stem}: {len(flat):,} physical outputs ({dropped})")

    names, form_rdf = known_form_rdfs(cli.forms)
    cand_rdf = rdf_of(flat)
    bins = torch.linspace(0, 10, cand_rdf.shape[-1])

    #: the known forms are the COMPACK reference; the candidates are the tests
    poly = torch.load(os.path.join(ROOT, 'std_opt_acridine_polymorphs.pt'),
                      weights_only=False, map_location='cpu').cpu()
    pl, pids = poly.batch_to_list(), list(poly.identifier)
    query = collate_data_list([pl[pids.index(n)].clone() for n in names])
    query.aunit_handedness = query.aunit_handedness.abs()

    nb = torch.stack([compute_rdf_distance(form_rdf[k], cand_rdf, bins).argsort()[:cli.k]
                      for k in range(len(names))])
    d = torch.stack([compute_rdf_distance(form_rdf[k], cand_rdf, bins)[nb[k]]
                     for k in range(len(names))])

    #: compack_confirm indexes a BATCH, so collate only the candidates it needs
    #: and remap the neighbour indices onto that sub-batch
    need = sorted({int(j) for row in nb for j in row})
    remap = {g: i for i, g in enumerate(need)}
    sub = collate_data_list([flat[g].clone() for g in need])
    nb_local = torch.tensor([[remap[int(j)] for j in row] for row in nb])
    rmsds, matched = compack_confirm(query, sub, nb_local, WORK)
    print(f"\n{'form':12s} {'rank':>4} {'rdf':>8} {'matched':>9} {'rmsd':>8} "
          f"{'mace':>8}")
    for k, n in enumerate(names):
        for j in range(cli.k):
            m, r = int(matched[k, j]), float(rmsds[k, j])
            note = '  <- comparison FAILED' if (m == 0 and r == 0.0) else ''
            print(f"{n:12s} {j:4d} {float(d[k, j]):8.4f} {m:6d}/20 {r:8.3f} "
                  f"{float(flat[int(nb[k, j])].mace):8.2f}{note}")


if __name__ == '__main__':
    main()
