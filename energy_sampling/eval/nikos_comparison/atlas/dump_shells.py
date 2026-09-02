"""
Dump the FULL coordination shell of each low-energy basin representative.

The first atlas drew each basin from one number (median herringbone angle), so the
eleven beta basins that share theta=0 rendered identically. The measured geometry is
much richer than that: per neighbour there is a signed offset in the molecular frame
and an apparent long-axis tilt, and coordination number itself runs 10-16.

This writes that whole shell out so the drawing can be a PROJECTION of the real
neighbour set rather than a glyph chosen by category.
"""
import json
import os
from collections import Counter

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.packing_motifs import (
    classify, motif_of)
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, chunk_by_cluster_cost, harmonize, load_arms, physical)

SP = os.path.dirname(os.path.abspath(__file__))
CUT, LOW = 0.10, -60.0
FORMS = ['ACRDIN04', 'ACRDIN12']


def main():
    poly = torch.load(os.path.join(ROOT, 'std_opt_acridine_polymorphs.pt'),
                      weights_only=False, map_location='cpu').cpu()
    pl, pids = poly.batch_to_list(), list(poly.identifier)

    controls = []
    for n in pids:
        rows, flat = motif_of(pl[pids.index(n)])
        label, stack = classify(rows)
        controls.append(dict(name=n, flat=flat, label=label, stack=stack,
                             fam=label.split()[0].lower(), shell=rows))
    print(f"controls: {len(controls)}", flush=True)

    groups = load_arms(os.path.join(ROOT, 'prior_chunks'),
                       'may_acridine_sg14_zp1_*.pt')
    raw = [c for stem in sorted(groups)
           for _, lst in sorted(groups[stem]) for c in lst]
    keep, _ = physical(raw)
    E = torch.tensor([float(c.mace) for c in keep])
    idx = (E <= LOW).nonzero().flatten().tolist()
    pool = harmonize([keep[i] for i in idx])
    Ep = E[idx]
    print(f"{len(pool):,} structures below {LOW}", flush=True)

    r = []
    for lo, hi in chunk_by_cluster_cost(pool, 1_500_000):
        b = collate_data_list([c.clone() for c in pool[lo:hi]],
                              exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        b.aunit_handedness = b.aunit_handedness.abs()
        with torch.no_grad():
            o = b.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
        rr = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
        r.append(rr.cpu())
        del b, o, rr
    r = torch.cat(r)
    bins = torch.linspace(0, 10, r.shape[-1])
    m = len(r)
    D = np.empty((m, m), dtype=np.float32)
    for i in range(m):
        D[i] = compute_rdf_distance(r[i], r, bins).numpy()
    D += D.T
    D *= 0.5
    np.fill_diagonal(D, 0.0)
    lab = fcluster(linkage(squareform(D, checks=False), method='average'),
                   t=CUT, criterion='distance')
    occ = Counter(lab)

    form_basin = {}
    for n in FORMS:
        fb = collate_data_list([pl[pids.index(n)].clone()],
                               exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        fb.aunit_handedness = fb.aunit_handedness.abs()
        with torch.no_grad():
            fo = fb.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
        fr = fo['rdf'][0] if isinstance(fo['rdf'], (tuple, list)) else fo['rdf']
        dd = compute_rdf_distance(fr[0], r, bins)
        form_basin[int(lab[int(dd.argmin())])] = dict(
            name=n, rdf=float(dd.min()))

    basins = []
    for bi in sorted(occ, key=lambda k: -occ[k]):
        members = (torch.tensor(lab) == bi).nonzero().flatten()
        j = int(members[Ep[members].argmin()])
        rows, flat = motif_of(pool[j])
        label, stack = classify(rows)
        basins.append(dict(basin=int(bi), n=int(occ[bi]), e=float(Ep[j]),
                           flat=flat, label=label, stack=stack,
                           fam=label.split()[0].lower(),
                           form=form_basin.get(int(bi)), shell=rows))
        print(f"   basin {bi:3d}  n={occ[bi]:5d}  {len(rows):2d} nbrs  {label}",
              flush=True)

    out = os.path.join(SP, 'shells.json')
    json.dump({'basins': basins, 'controls': controls}, open(out, 'w'))
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
