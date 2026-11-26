"""Does an exp-6 field keep the MLIP's basins? (docs/design/exp6_calibration_plan.md, comparison diagnostics)

Per panel molecule, after compare.py: every MLIP basin within --window kJ/mol of the lowest basin any arm found
(the pooled reference of compare.report) is represented by its lowest R minimum. That row is relaxed under exp-6
(FIT: arm U; the tuned file: arm T), and the exp-6 minimum reached is relaxed back under the MLIP. The basin is KEPT
when the return lands in the same basin (basins.basins over the pooled minima and the returns together): exp-6 has a
minimum there that the MLIP maps back, so an exp-6 search that reaches it recovers the basin -- a miss is then a
sampling miss. A basin that returns elsewhere is LOST to that field: no exp-6 search, however long, recovers it this
way. For a kept basin, its exp-6 rank (exp-6 minimum rows of that arm's own search lower in exp-6 energy) says whether
a cap of K picks would have taken it.

Outputs in OUT: <name>_rt_seeds.pt, the exp-6 and MLIP seeds runs (<name>_exp6_seeds_rt{U,T},
<name>_mace_csp_seeds_rt{U,T}back), and roundtrip.json: per molecule, every pooled basin (energy above the lowest,
which arms found it, and for the in-window ones the round trip per field).

usage: python -m eval.exp6_calibration.roundtrip OUT TUNED_JSON [--window 10] [--slots 8] [--molecules ...]
"""
import argparse
import json
import os

import numpy as np
import torch

from eval.exp6_calibration import common as C
from eval.exp6_calibration import compare as cmp


def pooled(out, name):
    """compare.report's pool: [(arm, index, row)] of R, U-relax and T-relax MLIP minima."""
    from eval.exp6_recall.mirror_test import paired_minima
    r = cmp.runs(name, None)
    R_min = list(torch.load(os.path.join(out, r['R'] + '_minima.pt'), weights_only=False))
    R_all = list(torch.load(os.path.join(out, r['R'] + '.pt'), weights_only=False))
    byid = {int(x.minimum_id): x for x in R_min}
    arms = {'R': [(int(s.start_index), byid[int(s.minimum_id)]) for s in R_all
                  if 'minimum_id' in s.keys() and int(s.minimum_id) in byid]}
    for a in ('U', 'T'):
        m = list(torch.load(os.path.join(out, r[f'{a}_relax'] + '_minima.pt'), weights_only=False))
        s = list(torch.load(os.path.join(out, r[f'{a}_relax'] + '.pt'), weights_only=False))
        n = len(torch.load(os.path.join(out, f'{name}_{a}_picks.pt'), weights_only=False))
        arms[a] = [(j, x) for j, x in enumerate(paired_minima(m, s, n)) if x is not None]
    return [(a, j, x) for a, lst in arms.items() for j, x in lst]


def labels(sg, mol, rows):
    from eval.exp6_recall.basins import basins
    params = torch.stack([x.full_cell_parameters().reshape(-1) for x in rows]).double()
    hand = torch.stack([x.aunit_handedness.reshape(-1)[0] for x in rows])
    f = torch.tensor([float(x.minimum_f) for x in rows], dtype=torch.float64)
    lab, _ = basins(sg, mol, params, hand, f)
    return np.asarray(lab), f.numpy()


def seeds_run(out, name, energy, tag, seeds, slots, exp6=None):
    run = f'{name}_{energy}_seeds_{tag}'
    if not os.path.exists(os.path.join(out, run + '_minima.pt')):
        cmp.launch(out, name, energy, 0, slots, tag=tag, exp6=exp6, seeds=seeds)
    from eval.exp6_recall.mirror_test import paired_minima
    m = list(torch.load(os.path.join(out, run + '_minima.pt'), weights_only=False))
    s = list(torch.load(os.path.join(out, run + '.pt'), weights_only=False))
    return paired_minima(m, s, len(torch.load(seeds, weights_only=False)))


def molecule(out, name, tuned, window, slots):
    from eval.exp6_recall.make_configs import KT
    from eval.exp6_recall.mirror_test import clean
    from eval.exp6_recall.mlip_basins import E
    rec = json.load(open(os.path.join(out, 'panel.json')))[name]
    mol = os.path.join(out, rec['mol'])
    pool = pooled(out, name)
    lab, f = labels(rec['sg'], mol, [x for _, _, x in pool])
    kT = KT[C.MLIP]
    e = (f - f.min()) * kT
    basins_ = sorted({int(b) for b in lab}, key=lambda b: float(e[lab == b].min()))
    # each in-window basin with an R member, represented by its lowest R member
    reps = {}
    for b in basins_:
        idx = [i for i in np.flatnonzero(lab == b) if pool[i][0] == 'R']
        if idx and float(e[lab == b].min()) <= window:
            reps[b] = min(idx, key=lambda i: f[i])
    seeds = os.path.join(out, f'{name}_rt_seeds.pt')
    if not os.path.exists(seeds):
        torch.save([clean(pool[i][2]) for i in reps.values()], seeds)
    trips = {}
    for arm, exp6 in (('U', None), ('T', tuned)):
        e6 = seeds_run(out, name, 'exp6', f'rt{arm}', seeds, slots, exp6)
        keep = [j for j, x in enumerate(e6) if x is not None]
        back_seeds = os.path.join(out, f'{name}_rt{arm}_back_seeds.pt')
        if not os.path.exists(back_seeds):
            torch.save([clean(e6[j]) for j in keep], back_seeds)
        back = seeds_run(out, name, C.MLIP, f'rt{arm}back', back_seeds, slots)
        own = [E(r, 'exp6') for r in torch.load(os.path.join(out, cmp.runs(name, None)[arm] + '_minima.pt'),
                                                 weights_only=False)]
        trips[arm] = dict(e6=e6, keep=keep, back=back, own=own)
    # joint labels: the pool plus every return
    extra = [(arm, j, x) for arm, t in trips.items() for j, x in zip(t['keep'], t['back']) if x is not None]
    lab2, f2 = labels(rec['sg'], mol, [x for _, _, x in pool] + [x for _, _, x in extra])
    n = len(pool)
    old = {int(b): int(lab2[i]) for b, i in reps.items()}  # pooled basin -> its joint label (via its R rep)
    res = dict(basins=[])
    for b in basins_:
        members = np.flatnonzero(lab == b)
        row = dict(e_kJ=float(e[members].min()), found={a: bool(any(pool[i][0] == a for i in members))
                                                        for a in ('R', 'U', 'T')})
        if b in reps:
            j0 = list(reps).index(b)
            for arm, t in trips.items():
                x6 = t['e6'][j0]
                if x6 is None:
                    row[f'rt_{arm}'] = 'exp6 relax lost'
                    continue
                k = t['keep'].index(j0)
                xb = t['back'][k]
                if xb is None:
                    row[f'rt_{arm}'] = 'mlip relax lost'
                    continue
                ie = n + [i for i, (a, jj, _) in enumerate(extra) if a == arm and jj == j0][0]
                e6v = E(x6, 'exp6')
                row[f'rt_{arm}'] = dict(kept=bool(int(lab2[ie]) == old[b]),
                                        back_e_kJ=float((f2[ie] - f.min()) * kT),
                                        exp6_rank=int(sum(v < e6v - 1e-6 for v in t['own'])),
                                        exp6_above_own_lowest_kJ=float(e6v - min(t['own'])))
        res['basins'].append(row)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('tuned')
    ap.add_argument('--window', type=float, default=10.0)
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--molecules', default=None)
    a = ap.parse_args()
    panel = json.load(open(os.path.join(a.out, 'panel.json')))
    names = a.molecules.split(',') if a.molecules else [k for k, v in panel.items() if v['role'] == 'panel']
    allres = {n: molecule(a.out, n, os.path.abspath(a.tuned), a.window, a.slots) for n in names}
    json.dump(allres, open(os.path.join(a.out, 'roundtrip.json'), 'w'), indent=1)
    print('\nRound trip of each in-window MLIP basin (lowest R member): exp-6 relax, then MLIP relax back.')
    print(f'{"molecule":9s} {"E (kJ/mol)":>10s} {"found":>6s} {"FIT exp-6 (U)":>28s} {"tuned exp-6 (T)":>28s}')
    for n, r in allres.items():
        for b in r['basins']:
            if 'rt_U' not in b:
                continue
            found = ''.join(k for k, v in b['found'].items() if v)

            def show(t):
                if isinstance(t, str):
                    return t
                return f'{"KEPT" if t["kept"] else "lost"} rank {t["exp6_rank"]:2d} +{t["exp6_above_own_lowest_kJ"]:.1f}'
            print(f'{n:9s} {b["e_kJ"]:10.2f} {found:>6s} {show(b["rt_U"]):>28s} {show(b["rt_T"]):>28s}')


if __name__ == '__main__':
    main()
