"""Recall of UMA-s basins by the exp-6 and eLJ routes, for one molecule of the exp-6 recall test.

Inputs in OUT_DIR (run_pilot.py of mxtaltools configs/crystal_searches/exp6_recall_nov10):
    NAME_uma_search{.pt,_minima.pt}                 R: UMA-s from random starts
    NAME_uma_seeds_from_{exp6,elj}{.pt,_minima.pt}  X, L: UMA-s from the cheap search's lowest minima
    NAME_{exp6,elj}_picks.pt                        the picks, lowest cheap energy first (seed j = row j)
and optionally ANCHOR, a UMA-s relaxation of the experimental structure (stage A's NAME_uma_seeds.pt).

Every polished UMA-s minimum of every arm is pooled and grouped into basins by ``basins.basins`` (symmetry-aware FD
distance < 0.05 A, energies within 0.1 kT). Energies are UMA-s lattice energies (kJ/mol per molecule, the runs'
mirror-mean gas reference) above the lowest basin found by any arm.

usage: python -m eval.exp6_recall.recall OUT_DIR NAME PANEL_DIR SG [--anchor PATH] [--json PATH]
"""
import argparse
import json
import os

import torch

from eval.exp6_recall.basins import basins
from eval.exp6_recall.make_configs import KT

ARMS = {'R': 'uma_search', 'X': 'uma_seeds_from_exp6', 'L': 'uma_seeds_from_elj'}
WINDOWS = (5.0, 10.0)  # kJ/mol above the lowest basin
KS = (10, 30, 100, 300)


def load_arm(out_dir, name, run):
    """(minima rows, every-start rows) of one run, or (None, None) when it is missing."""
    p = os.path.join(out_dir, f'{name}_{run}')
    if not os.path.exists(p + '_minima.pt'):
        return None, None
    return list(torch.load(p + '_minima.pt', weights_only=False)), list(torch.load(p + '.pt', weights_only=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('name')
    ap.add_argument('panel_dir')
    ap.add_argument('sg', type=int)
    ap.add_argument('--anchor', default=None)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    kT = KT['uma']

    # pool every arm's minima: index into `pool`, and per arm a map from its own minimum_id to the pool index
    pool, owner, local = [], [], {}
    starts = {}
    for arm, run in ARMS.items():
        minima, rows = load_arm(a.out_dir, a.name, run)
        if minima is None:
            print(f'{arm} ({run}): missing')
            continue
        local[arm] = {}
        for r in minima:
            local[arm][int(r.minimum_id)] = len(pool)
            pool.append(r)
            owner.append(arm)
        starts[arm] = rows
    if a.anchor:
        anc = torch.load(a.anchor, weights_only=False)[0]
        anchor_ok = int(anc.polish_status) in (1, 5, 6)
        pool.append(anc)
        owner.append('anchor')
    params = torch.stack([r.full_cell_parameters().reshape(-1) for r in pool]).double()
    hand = torch.stack([r.aunit_handedness.reshape(-1)[0] for r in pool])
    f = torch.tensor([float(r.minimum_f) for r in pool], dtype=torch.float64)
    lab, _ = basins(a.sg, os.path.join(a.panel_dir, f'{a.name}_mol.pt'), params, hand, f, df_kT=0.1)
    nb = int(lab.max()) + 1
    E = torch.full((nb,), float('inf'), dtype=torch.float64)
    for i in range(len(pool)):
        E[lab[i]] = min(float(E[lab[i]]), float(f[i]) * kT)
    E = E - E.min()

    # each start's basin, in start order (R: random starts by start_index; X, L: seeds in pick order)
    seq = {}
    unconverged = {}
    for arm, rows in starts.items():
        if arm == 'R':
            keyed = [(int(r.start_index), r) for r in rows if int(r.start_index) >= 0]
        else:
            keyed = [(-int(r.start_index) - 1, r) for r in rows if int(r.start_index) < 0]
        keyed.sort(key=lambda t: t[0])
        seq[arm] = [int(lab[local[arm][int(r.minimum_id)]]) if int(r.minimum_id) >= 0 else -1 for _, r in keyed]
        unconverged[arm] = sum(b < 0 for b in seq[arm])
    anchor_basin = int(lab[-1]) if a.anchor else None

    res = dict(name=a.name, n_minima_pooled=len(pool), n_basins=nb, starts={k: len(v) for k, v in seq.items()},
               unconverged=unconverged, windows={})
    for W in WINDOWS:
        inw = set(torch.nonzero(E <= W).flatten().tolist())
        row = dict(n_basins=len(inw))
        for arm, s in seq.items():
            found = set(s) & inw
            row[arm] = dict(found=len(found), missed=[round(float(E[b]), 2) for b in
                                                      sorted(inw - found, key=lambda b: float(E[b]))],
                            at_K={K: len(set(s[:K]) & inw) for K in KS if K <= len(s)})
        res['windows'][W] = row
    for arm in ('X', 'L'):
        if arm not in seq:
            continue
        s = [b for b in seq[arm] if b >= 0]
        counts = torch.bincount(torch.tensor(s), minlength=nb) if s else torch.zeros(nb)
        inw = E <= WINDOWS[-1]
        res[f'funnel_{arm}'] = dict(picks=len(seq[arm]), distinct_basins=int((counts > 0).sum()),
                                    top_basin_share=float(counts.max() / max(1, len(s))),
                                    outside_window=int(counts[~inw].sum()))
    if a.anchor:
        rank = int((E < E[anchor_basin] - 1e-9).sum()) + 1
        res['anchor'] = dict(converged=anchor_ok, basin=anchor_basin, dE=float(E[anchor_basin]), rank=rank,
                             first_K={arm: (s.index(anchor_basin) + 1 if anchor_basin in s else None)
                                      for arm, s in seq.items()})
    res['basin_energies_lowest20'] = [round(float(e), 3) for e in torch.sort(E).values[:20]]
    report(res)
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(res, fh, indent=1, default=float)


def report(res):
    arms = [k for k in ('R', 'X', 'L') if k in res['starts']]
    names = dict(R='UMA-s random starts', X='exp-6 picks -> UMA-s', L='eLJ picks -> UMA-s')
    print(f"\n{res['name']}: {res['n_minima_pooled']} UMA-s minima pooled into {res['n_basins']} basins; starts per arm "
          + ', '.join(f"{names[k]} {res['starts'][k]} ({res['unconverged'][k]} unconverged)" for k in arms))
    for W, row in res['windows'].items():
        print(f"\nTable. Basins within {W:g} kJ/mol of the lowest UMA-s basin found by any arm ({row['n_basins']} "
              f"basins), found by each arm after its first K relaxations (R: random starts in order; X, L: picks "
              f"lowest cheap energy first) and in all. One seed per arm; basin = symmetry-aware FD < 0.05 A.")
        Ks = sorted({K for k in arms for K in row[k]['at_K']})
        print(f"{'arm':22s} | " + ' | '.join(f'K={K:>4d}' for K in Ks) + f" | {'all':>5s} | missed basins: dE (kJ/mol), lowest first")
        for k in arms:
            r = row[k]
            cells = [f"{r['at_K'][K]:6d}" if K in r['at_K'] else f"{'-':>6s}" for K in Ks]
            print(f"{names[k]:22s} | " + ' | '.join(cells) + f" | {r['found']:5d} | {r['missed'][:10]}")
    for arm in ('X', 'L'):
        if f'funnel_{arm}' in res:
            fu = res[f'funnel_{arm}']
            print(f"funnel {names[arm]}: {fu['picks']} picks -> {fu['distinct_basins']} distinct UMA-s basins; most "
                  f"visited basin takes {100 * fu['top_basin_share']:.0f}% of picks; {fu['outside_window']} picks land "
                  f"above {WINDOWS[-1]:g} kJ/mol")
    if 'anchor' in res:
        an = res['anchor']
        print(f"experimental structure: UMA-s basin {an['basin']} at {an['dE']:.2f} kJ/mol above the lowest (rank "
              f"{an['rank']}); first found at K = " + ', '.join(f"{names[k]} {v}" for k, v in an['first_K'].items()))
    print('lowest basin energies (kJ/mol):', res['basin_energies_lowest20'])


if __name__ == '__main__':
    main()
