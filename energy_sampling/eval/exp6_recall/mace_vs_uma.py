"""Do MACE-OFF24 and UMA-s 1.1 rank a molecule's crystal structures the same way? (owner, 2026-11-18: agreement as a
first-order check before considering MACE-OFF24, which is exactly mirror-invariant and ~3x cheaper per call.)

Agreement, not correctness: neither model is the reference.

Part 1, imidazole (P2_1/c, a group with an improper element, so UMA-s carries none of the error that changes sign
under a mirror): every polished UMA-s minimum of the local pilot (D:/.../pilot_local) and the experimental structure
relaxed under UMA-s, grouped into UMA-s basins (basins.basins); each basin's lowest member within WINDOW of the lowest:
  - MACE-OFF24 single points at those UMA-s minima;
  - MACE-OFF24 relaxations of the N_RELAX lowest (seeds runs), compared with where they started (symmetry-aware FD)
    and ranked again at MACE-OFF24's own minima.
Part 2, the mirror-test molecules (mirror_test.py; chiral groups): MACE-OFF24 single points on the same 60 exp-6
minima, against UMA-s with the mirror average (its sign-changing error removed).

Readouts per set: Spearman rank correlation, the spread (sd) of the difference of relative energies (each model's
energies relative to its own lowest), top-k overlap, and whether both put the same structure lowest.

usage: python -m eval.exp6_recall.mace_vs_uma OUT_DIR [--window 15] [--n_relax 30]
"""
import argparse
import json
import os

import torch

from eval.exp6_recall.basins import SymFD, basins
from eval.exp6_recall.make_configs import KT, SITES, config
from eval.exp6_recall.mirror_test import cap_gpu, clean, paired_minima, run

PANEL = 'D:/crystal_datasets/exp6_recall_nov10/panel'
PILOT = 'D:/crystal_datasets/exp6_recall_nov10/pilot_local'
ANCHOR = 'D:/crystal_datasets/exp6_recall_nov10/stageA_anchors/imidazole_uma_seeds.pt'
MIRROR = 'D:/crystal_datasets/exp6_recall_nov10/mirror_test_nov18'
ld = lambda p: list(torch.load(p, weights_only=False))


def mace_energies(rows, pred):
    from mxtaltools.crystal_search.minima import StageEnergy
    rows = [clean(r) for r in rows]
    for r in rows:
        for k in ('mace_pot', 'mace_gas_pot'):
            if hasattr(r, k):
                delattr(r, k)
    E = StageEnergy(rows, 'mace', KT['mace'], predictor=pred, device='cuda', chunk=8)
    x = torch.stack([r.full_cell_parameters().reshape(-1) for r in rows]).double()
    with torch.no_grad():
        return (E.f(x, list(range(len(rows)))) * KT['mace']).double().cpu()


def agreement(u, m, ks=(1, 5, 10)):
    """u, m: energies (kJ/mol) of the same structures under the two models."""
    u, m = torch.as_tensor(u, dtype=torch.float64), torch.as_tensor(m, dtype=torch.float64)
    ok = torch.isfinite(u) & torch.isfinite(m)
    u, m = u[ok], m[ok]
    rank = lambda v: torch.argsort(torch.argsort(v)).double()
    du, dm = u - u.min(), m - m.min()
    out = dict(n=int(len(u)), spearman=float(torch.corrcoef(torch.stack([rank(u), rank(m)]))[0, 1]),
               rel_diff_sd=float((dm - du).std()), rel_diff_maxabs=float((dm - du - (dm - du).mean()).abs().max()),
               same_lowest=int(torch.argmin(u)) == int(torch.argmin(m)),
               mace_rank_of_uma_lowest=int(rank(m)[int(torch.argmin(u))]) + 1)
    for k in ks:
        if k <= len(u):
            out[f'top{k}_overlap'] = len(set(torch.argsort(u)[:k].tolist()) & set(torch.argsort(m)[:k].tolist()))
    return out


def part1(out, a, pred):
    pool = []
    for run_ in ('uma_search', 'uma_seeds_from_exp6', 'uma_seeds_from_elj'):
        pool += ld(os.path.join(PILOT, f'imidazole_{run_}_minima.pt'))
    pool += ld(ANCHOR)[:1]
    anchor_i = len(pool) - 1
    mol = os.path.join(PANEL, 'imidazole_mol.pt')
    params = torch.stack([r.full_cell_parameters().reshape(-1) for r in pool]).double()
    hand = torch.stack([r.aunit_handedness.reshape(-1)[0] for r in pool])
    f = torch.tensor([float(r.minimum_f) for r in pool], dtype=torch.float64)
    lab, _ = basins(14, mol, params, hand, f, df_kT=0.1)
    rep = {}
    for i in range(len(pool)):
        b = int(lab[i])
        rep[b] = i if b not in rep or f[i] < f[rep[b]] else rep[b]
    e0 = float(f.min())
    keep = sorted([i for i in rep.values() if (float(f[i]) - e0) * KT['uma'] <= a.window], key=lambda i: float(f[i]))
    rows = [pool[i] for i in keep]
    Eu = f[keep] * KT['uma']
    Em = mace_energies(rows, pred)
    res = dict(n_basins=len(keep), window_kJ=a.window, single=agreement(Eu, Em),
               anchor_basin_uma_rank=keep.index(rep[int(lab[anchor_i])]) + 1 if rep[int(lab[anchor_i])] in keep else None)
    # MACE-OFF24 relaxations of the lowest N
    n = min(a.n_relax, len(rows))
    sp = os.path.join(out, 'imidazole_uma_minima_seeds.pt')
    if not os.path.exists(sp):
        torch.save([clean(r) for r in rows[:n]], sp)
    _, c = config('imidazole', 14, 'mace', 'seeds', PANEL, out, 'local', seeds=sp)
    c['run_name'] = 'imidazole_mace_from_uma_minima'
    mins, starts = run(c)
    ends = paired_minima(mins, starts, n)
    ok = [j for j in range(n) if ends[j] is not None]
    both = [rows[j] for j in ok] + [ends[j] for j in ok]
    p2 = torch.stack([r.full_cell_parameters().reshape(-1) for r in both]).double()
    h2 = torch.stack([r.aunit_handedness.reshape(-1)[0] for r in both])
    fd = SymFD(14, mol, p2, h2)
    k = len(ok)
    D = fd.pairs(list(range(k)), [k + i for i in range(k)])
    Emr = torch.tensor([float(ends[j].minimum_f) * KT['mace'] for j in ok], dtype=torch.float64)
    distinct = len({int(ends[j].minimum_id) for j in ok})
    res['relax'] = dict(n=n, converged=k, distinct_mace_minima=distinct, fd_median=float(D.median()),
                        fd_max=float(D.max()), moved_gt_0p05=int((D > 0.05).sum()),
                        agreement=agreement(Eu[ok], Emr),
                        per_seed=[dict(seed=j, E_uma=float(Eu[j]), E_mace_sp=float(Em[j]), E_mace_relaxed=float(Emr[i]),
                                       fd_A=float(D[i]), mace_min=int(ends[j].minimum_id)) for i, j in enumerate(ok)])
    return res


def part2(pred):
    res = {}
    mt = json.load(open(os.path.join(MIRROR, 'mirror_test.json')))
    for name, r in mt.items():
        sg = r['sg']
        minima = ld(os.path.join(MIRROR, f'{name}_sg{sg}_exp6_search_minima.pt'))
        minima.sort(key=lambda q: float(q.minimum_f))
        rows = minima[:r['n_single']]
        Em = mace_energies(rows, pred)
        Ea = torch.tensor(r['single']['Ax'], dtype=torch.float64)
        Er = torch.tensor(r['single']['Rx'], dtype=torch.float64)
        res[name] = dict(sg=sg, vs_uma_averaged=agreement(Ea, Em), vs_uma_raw=agreement(Er, Em),
                         raw_vs_averaged_uma=agreement(Ea, Er), E_mace=Em.tolist())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--window', type=float, default=15.0)
    ap.add_argument('--n_relax', type=int, default=30)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, 'mace_vs_uma.json')):
        raise SystemExit(f'{a.out_dir} already holds a result; use a new directory')
    os.makedirs(a.out_dir, exist_ok=True)
    cap_gpu()
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    pred = load_mace_model(SITES['local']['mace'], device='cuda', dtype=torch.float32)
    out = dict(imidazole=part1(a.out_dir, a, pred), mirror_molecules=part2(pred))
    with open(os.path.join(a.out_dir, 'mace_vs_uma.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    p1 = out['imidazole']
    print('MACEvUMA imidazole single', json.dumps(p1['single']))
    print('MACEvUMA imidazole relax', json.dumps({k: v for k, v in p1['relax'].items() if k != 'per_seed'}))
    for name, r in out['mirror_molecules'].items():
        print('MACEvUMA', name, 'vs averaged', json.dumps(r['vs_uma_averaged']), '| raw vs averaged UMA',
              json.dumps(r['raw_vs_averaged_uma']))


if __name__ == '__main__':
    main()
