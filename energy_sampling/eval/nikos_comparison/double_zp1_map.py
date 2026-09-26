"""
Double the distinct low-energy minima of a Z'=1 sg14 search map into exact Z'=2 structures (acridine).

Why: a Z'=1 P2_1/c crystal is also a Z'=2 P2_1/c crystal (on its index-2 supercell), with the same energy per
molecule. The Z'=1 search reaches low energies far more often than the Z'=2 one, and on 2026-09-25 none of the six
lowest Z'=1 families had been reached by the Z'=2 random search as the same crystal. Doubling adds them to a Z'=2
map with no relaxation. It does not produce genuinely Z'=2 packings.

What it does:
  1. loads the Z'=1 arms one file at a time (summarize_search.load_arms, physical filter) and keeps end states with
     stored energy <= floor + band_kt * kT;
  2. groups them into families by envwise RDF distance, average linkage cut at --cut (0.10: the acridine review's
     family definition);
  3. takes each family's lowest member and re-describes it as Z'=2 with mxtaltools
     crystal_building.zp_doubling.double_zp1_crystal, which raises unless the doubled unit cell reproduces the
     parent atom for atom;
  4. writes a list of Z'=2 MolCrystalData sorted by energy, each carrying `mace` (the parent's stored energy, kJ/mol
     per molecule), `zp1_family`, `zp1_family_size` and `zp1_source` ('<file>#<index in its physical list>').

    python -m energy_sampling.eval.nikos_comparison.double_zp1_map --out D:/crystal_datasets/acridine/doubled_zp1_sg14.pt
"""
import argparse
import glob
import os
import re
import time

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.crystal_building.zp_doubling import double_zp1_crystal
from mxtaltools.dataset_utils.utils import collate_data_list
from energy_sampling.eval.nikos_comparison.summarize_search import (ROOT, chunk_by_cluster_cost, harmonize, load_arms,
                                                                    physical)

KT = 2.494
RDF_KW = dict(cutoff=10, rdf_cutoff=10, supercell_size=10, bins=100, rdf_mode='envwise', std_orientation=True)
DROP = ['mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']  # keeps `mace`


def low_end_states(root, pattern, e_max):
    """(crystals with stored mace <= e_max, their sources), loading one arm file at a time in NUMERIC arm order.
    The order matters only for ties: stored energies are float32-quantised (~0.0015 kJ/mol per molecule), so a
    family can hold several members at exactly its minimum, and argmin keeps the first loaded."""
    keep, src = [], []
    arm_no = lambda path: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', os.path.basename(path))]
    for path in sorted(glob.glob(os.path.join(root, pattern)), key=arm_no):
        for stem_arms in load_arms(root, os.path.basename(path)).values():
            for arm, lst in stem_arms:
                phys, _ = physical(lst)
                for j, c in enumerate(phys):
                    if float(c.mace) <= e_max:
                        d = c.clone()
                        for k in DROP:
                            if k in d.keys():
                                del d[k]
                        keep.append(d)
                        src.append(f'{os.path.basename(path)}#{j}')
    return keep, src


def rdfs(lst, device):
    out = []
    for lo, hi in chunk_by_cluster_cost(lst):
        b = collate_data_list([c.clone() for c in lst[lo:hi]]).to(device)
        with torch.no_grad():
            o = b.analyze(['rdf'], assign_outputs=False, **RDF_KW)
        r = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
        out.append(r.float().cpu())
    return torch.cat(out)


def rdf_pdist(r, bins, chunk=64):
    """All-pairs envwise RDF distance, the same quantity as mxtaltools compute_rdf_distance (bin width x mean over
    channels active in either structure of the L1 distance between normalised CDFs), vectorised."""
    s = r.sum(-1, keepdim=True)
    cdf = torch.cumsum(r / (s + 1e-10), -1).permute(1, 0, 2).contiguous()      # [channels, n, bins]
    active = (s[..., 0] > 1e-12).float()                                     # [n, channels]
    n_act = active.sum(1)
    bw = float(bins[1] - bins[0])
    out = torch.empty(len(r), len(r))
    for i in range(0, len(r), chunk):
        l1 = torch.cdist(cdf[:, i:i + chunk], cdf, p=1).sum(0)
        inter = active[i:i + chunk] @ active.T
        union = (n_act[i:i + chunk, None] + n_act[None] - inter).clamp_min(1)
        out[i:i + chunk] = bw * l1 / union
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default=os.path.join(ROOT, 'prior_chunks'))
    ap.add_argument('--pattern', default='may_acridine_sg14_zp1_*.pt')
    ap.add_argument('--floor', type=float, default=-62.812, help="energy reference (kJ/mol; the sg14 Z'=2 floor)")
    ap.add_argument('--band_kt', type=float, default=2.0)
    ap.add_argument('--cut', type=float, default=0.10, help='family cut, envwise RDF distance (average linkage)')
    ap.add_argument('--device', default='cpu', help='device for the RDFs')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t0 = time.time()
    e_max = args.floor + args.band_kt * KT
    low, src = low_end_states(args.root, args.pattern, e_max)
    if not low:
        raise SystemExit(f'no end states at or below {e_max:.3f} in {args.root}/{args.pattern}')
    low = harmonize(low)
    energy = np.array([float(c.mace) for c in low])
    print(f"{len(low)} Z'=1 end states <= {e_max:.3f} kJ/mol ({time.time() - t0:.0f}s)", flush=True)

    r = rdfs(low, args.device)
    bins = torch.linspace(0, 10, r.shape[-1])
    dist = rdf_pdist(r, bins).double().numpy()
    probe = np.random.default_rng(0).choice(len(low), size=min(20, len(low)), replace=False)
    ref = torch.stack([compute_rdf_distance(r[i], r[probe], bins).flatten() for i in probe]).numpy()
    assert np.abs(ref - dist[np.ix_(probe, probe)]).max() < 1e-4, 'vectorised RDF distance disagrees with mxtaltools'
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    labels = fcluster(linkage(squareform(dist, checks=False), 'average'), args.cut, 'distance')
    print(f'{len(set(labels.tolist()))} families at cut {args.cut} ({time.time() - t0:.0f}s)', flush=True)

    doubled = []
    for fam in sorted(set(labels.tolist())):
        members = np.nonzero(labels == fam)[0]
        i = int(members[np.argmin(energy[members])])
        c2, _ = double_zp1_crystal(low[i])                     # raises unless atom-for-atom exact
        c2.mace = low[i].mace.clone()
        c2.zp1_family = int(fam)
        c2.zp1_family_size = int(len(members))
        c2.zp1_source = src[i]
        doubled.append(c2)
    doubled.sort(key=lambda c: float(c.mace))
    torch.save(doubled, args.out)

    e = np.array([float(c.mace) for c in doubled])
    print(f"\n{len(doubled)} Z'=2 structures written to {args.out} ({time.time() - t0:.0f}s)")
    print(f'Families by lowest-member energy above {args.floor} kJ/mol (kT {KT}); one doubled structure per family.')
    below = int(((e - args.floor) < 0).sum())
    if below:
        print(f'  below the floor: {below}')
    edges = np.arange(0, args.band_kt + 0.5, 0.5)
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        n = int(((e - args.floor) / KT >= lo_).sum() - ((e - args.floor) / KT >= hi_).sum())
        print(f'  {lo_:.1f}-{hi_:.1f} kT: {n}')


if __name__ == '__main__':
    main()
