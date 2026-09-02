"""
Which 2D view of this landscape is actually FAITHFUL?

Before drawing a dimension-reduction figure, measure whether 2D can carry the
structure at all. The intrinsic dimension of the low-energy set looked like 5-8, so
a 2D embedding is lossy by construction -- the question is how lossy, and whether an
interpretable pair of axes does as well as an abstract embedding.

Three candidates, each scored:

  PCA on latent parameters      variance explained by the first two components
  classical MDS on RDF distance Kruskal stress-1 + distance correlation
  interpretable descriptor pairs  how much they separate the motif families and
                                  track energy

An embedding that explains 30% of the variance should be labelled as such, not
presented as a map.
"""
import json
import os
from itertools import combinations

import numpy as np
import torch

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, chunk_by_cluster_cost, harmonize, load_arms, physical)

SP = os.path.dirname(os.path.abspath(__file__))
LOW = -60.0


def main():
    D = json.load(open(os.path.join(SP, 'shells.json')))

    groups = load_arms(os.path.join(ROOT, 'prior_chunks'),
                       'may_acridine_sg14_zp1_*.pt')
    raw = [c for stem in sorted(groups)
           for _, lst in sorted(groups[stem]) for c in lst]
    keep, _ = physical(raw)
    E = torch.tensor([float(c.mace) for c in keep])
    pc = torch.tensor([float(c.packing_coeff) for c in keep])
    print(f"{len(keep):,} physical; {int((E <= LOW).sum()):,} below {LOW}")

    idx = (E <= LOW).nonzero().flatten().tolist()
    pool = harmonize([keep[i] for i in idx])
    lat = []
    for c in pool:
        b = collate_data_list([c.clone()],
                              exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        lat.append(b.latent_params()[0].numpy())
    X = np.stack(lat)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    u, s, vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    print(f"\nPCA on {X.shape[1]} latent params, low-energy set:")
    print(f"   PC1 {100*var[0]:.1f}%  PC2 {100*var[1]:.1f}%  "
          f"first two {100*var[:2].sum():.1f}%  first five {100*var[:5].sum():.1f}%")

    #: classical MDS on the 34 basin representatives, scored honestly
    reps = D['basins']
    #: shells.json stored GEOMETRY, not crystals, so recover each basin's
    #: representative from the pool by matching its stored energy
    r = []
    Ep = E[idx]
    chosen = []
    for b in reps:
        j = int((Ep - b['e']).abs().argmin())
        chosen.append(pool[j])
    for lo, hi in chunk_by_cluster_cost(chosen, 1_500_000):
        bb = collate_data_list([c.clone() for c in chosen[lo:hi]],
                               exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        bb.aunit_handedness = bb.aunit_handedness.abs()
        with torch.no_grad():
            o = bb.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
        rr = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
        r.append(rr.cpu())
        del bb, o, rr
    r = torch.cat(r)
    bins = torch.linspace(0, 10, r.shape[-1])
    n = len(r)
    Dm = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        Dm[i] = compute_rdf_distance(r[i], r, bins).numpy()
    Dm = (Dm + Dm.T) / 2
    np.fill_diagonal(Dm, 0.0)

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (Dm ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    pos = V[:, :2] * np.sqrt(np.clip(w[:2], 0, None))
    Dh = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    iu = np.triu_indices(n, 1)
    stress = np.sqrt(((Dm[iu] - Dh[iu]) ** 2).sum() / (Dm[iu] ** 2).sum())
    corr = np.corrcoef(Dm[iu], Dh[iu])[0, 1]
    pos_var = 100 * np.clip(w[:2], 0, None).sum() / np.clip(w, 0, None).sum()
    print(f"\nclassical MDS on the {n} basin RDF distances:")
    print(f"   2D captures {pos_var:.1f}% of the distance variance")
    print(f"   Kruskal stress-1 {stress:.3f}   distance correlation {corr:.3f}")
    print(f"   (stress <0.10 good, 0.10-0.20 fair, >0.20 poor)")

    #: interpretable descriptor pairs -- do any separate the families / track E?
    def slip_of(sh):
        st = [x for x in sh if x['theta'] < 25 and 3.1 <= x['h'] <= 3.9]
        return min((x['s'] for x in st), default=np.nan)

    feats = {}
    feats['slip'] = np.array([slip_of(b['shell']) for b in reps])
    feats['stack'] = np.array([b['stack'] if b['stack'] else np.nan for b in reps])
    feats['theta_med'] = np.array([np.median([x['theta'] for x in b['shell']])
                                   for b in reps])
    feats['theta_hi'] = np.array([np.median([x['theta'] for x in b['shell']
                                             if x['theta'] > 25] or [0]) for b in reps])
    feats['n_nbr'] = np.array([len(b['shell']) for b in reps], dtype=float)
    feats['near'] = np.array([min(x['d'] for x in b['shell']) for b in reps])
    en = np.array([b['e'] for b in reps])
    fam = np.array([1 if b['fam'] == 'gamma' else 0 for b in reps])
    cap = np.array([b['n'] for b in reps], dtype=float)

    print(f"\ndescriptor vs energy (|r|) and family separation (AUC-like):")
    for k, v in feats.items():
        ok = ~np.isnan(v)
        rr = abs(np.corrcoef(v[ok], en[ok])[0, 1]) if ok.sum() > 3 else np.nan
        g, bta = v[ok & (fam == 1)], v[ok & (fam == 0)]
        sep = (abs(g.mean() - bta.mean()) /
               np.sqrt(0.5 * (g.var() + bta.var()) + 1e-9)) if len(g) and len(bta) else np.nan
        print(f"   {k:10s} |r| with E {rr:5.2f}   family separation d' {sep:5.2f}")

    print(f"\nbest descriptor PAIRS for separating gamma from beta:")
    best = []
    for a, b2 in combinations(feats, 2):
        va, vb = feats[a], feats[b2]
        ok = ~np.isnan(va) & ~np.isnan(vb)
        if ok.sum() < 10:
            continue
        P = np.stack([va[ok], vb[ok]], 1)
        P = (P - P.mean(0)) / (P.std(0) + 1e-9)
        f = fam[ok]
        if f.sum() == 0 or f.sum() == len(f):
            continue
        d = np.linalg.norm(P[f == 1].mean(0) - P[f == 0].mean(0))
        best.append((d, a, b2))
    for d, a, b2 in sorted(best, reverse=True)[:5]:
        print(f"   {a:10s} x {b2:10s}  separation {d:.2f}")

    json.dump({'mds': pos.tolist(), 'stress': float(stress), 'corr': float(corr),
               'pos_var': float(pos_var),
               'pca_var': [float(x) for x in var[:6]],
               'feats': {k: [None if np.isnan(x) else float(x) for x in v]
                         for k, v in feats.items()},
               'e': en.tolist(), 'cap': cap.tolist(),
               'fam': [b['fam'] for b in reps],
               'basin': [b['basin'] for b in reps],
               'all_e': E.tolist(), 'all_pc': pc.tolist()},
              open(os.path.join(SP, 'projection.json'), 'w'))
    print(f"\nwrote projection.json")


if __name__ == '__main__':
    main()
