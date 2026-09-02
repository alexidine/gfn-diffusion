"""
An RDF-distance map with a FIDELITY NUMBER attached.

The current workflow is UMAP on a 10k x 10k RDF distance matrix, read by eye. Two
things are wrong with that, and neither needs packing expertise to fix:

  1. WRONG SUBSET. At the calibrated cut, ~72% of clusters in a random sample are
     singletons -- there is no repeated structure for an embedding to reveal, so the
     picture is a diffuse blob by construction. The LOW-ENERGY stratum has real
     cluster structure (1,969 structures, 42 basins, the largest holding 31%) and is
     26x cheaper to embed than 10k.

  2. NO FIDELITY NUMBER. UMAP's distortion is unquantified, so the map can only be
     vibed. Classical MDS on the same matrix yields Kruskal stress-1 and a distance
     correlation, which say exactly how much to trust it -- typically here "the
     ORDER of similarities survives, the SCALE does not".

This computes the low-energy distance matrix once, caches it, and reports both the
embedding and its honest error.
"""
import json
import os

import numpy as np
import torch

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, chunk_by_cluster_cost, harmonize, load_arms, physical)

SP = os.path.dirname(os.path.abspath(__file__))
LOW = -60.0
CACHE = os.path.join(ROOT, 'nikos_comparison', 'lowE_dmat.pt')


def main():
    if os.path.exists(CACHE):
        blob = torch.load(CACHE, weights_only=False)
        Dm, E = blob['D'].numpy().astype(np.float64), blob['E'].numpy()
        print(f"reusing {CACHE}  ({Dm.shape[0]:,} structures)")
    else:
        groups = load_arms(os.path.join(ROOT, 'prior_chunks'),
                           'may_acridine_sg14_zp1_*.pt')
        raw = [c for stem in sorted(groups)
               for _, lst in sorted(groups[stem]) for c in lst]
        keep, _ = physical(raw)
        Ea = torch.tensor([float(c.mace) for c in keep])
        idx = (Ea <= LOW).nonzero().flatten().tolist()
        pool = harmonize([keep[i] for i in idx])
        E = Ea[idx].numpy()
        print(f"{len(pool):,} structures below {LOW}; computing RDFs", flush=True)
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
        n = len(r)
        print(f"distance matrix {n:,}x{n:,}", flush=True)
        Dm = np.empty((n, n), dtype=np.float32)
        for i in range(n):
            Dm[i] = compute_rdf_distance(r[i], r, bins).numpy()
            if i % 400 == 0:
                print(f"   row {i:,}/{n:,}", flush=True)
        Dm += Dm.T
        Dm *= 0.5
        np.fill_diagonal(Dm, 0.0)
        torch.save({'D': torch.from_numpy(Dm), 'E': torch.from_numpy(E)}, CACHE)
        print(f"cached -> {CACHE}")
        Dm = Dm.astype(np.float64)

    n = Dm.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (Dm ** 2) @ J
    w, V = np.linalg.eigh(B)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]

    iu = np.triu_indices(n, 1)
    dt = Dm[iu]
    print(f"\n{'dims':>5} {'var captured':>13} {'stress-1':>9} {'dist corr':>10}")
    for k in (1, 2, 3, 5, 8):
        pos = V[:, :k] * np.sqrt(np.clip(w[:k], 0, None))
        Dh = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)[iu]
        st = np.sqrt(((dt - Dh) ** 2).sum() / (dt ** 2).sum())
        cr = np.corrcoef(dt, Dh)[0, 1]
        vc = 100 * np.clip(w[:k], 0, None).sum() / np.clip(w, 0, None).sum()
        print(f"{k:5d} {vc:12.1f}% {st:9.3f} {cr:10.3f}")
    print("   stress <0.10 good, 0.10-0.20 fair, >0.20 poor")
    print("   a high distance-correlation with poor stress means the ORDER of")
    print("   similarities is preserved while the SCALE is compressed.")

    pos2 = V[:, :2] * np.sqrt(np.clip(w[:2], 0, None))
    json.dump({'pos': pos2.tolist(), 'e': E.tolist()},
              open(os.path.join(SP, 'lowE_mds.json'), 'w'))
    print(f"\nwrote lowE_mds.json ({n:,} points)")


if __name__ == '__main__':
    main()
