"""Held-out comparison: is tuned exp-6 + MLIP endpoints a reliable surrogate for an MLIP search?
(docs/design/exp6_calibration_plan.md, step 5)

Per panel molecule (pool 'heldout_scaffold', not in the fit), the same N random starts (same seed) in every arm:
    R      the MLIP search itself (MACE-CSP L + D3), every start relaxed and polished
    T, U   an exp-6 search with the tuned parameter file (T) or the published FIT set (U); its distinct minima within
           --window kJ/mol of its lowest, at most --k, lowest exp-6 energy first, are the picks;
           endpoint 'sp': one MLIP single point per pick (ranking only);
           endpoint 'relax': each pick relaxed and polished by the MLIP (a seeds run)
Reference: every MLIP minimum of R, T-relax and U-relax, grouped into basins (basins.basins: symmetry-aware FD
< 0.05 A, energy within 0.1 kT); energies above the lowest basin any arm found.

Readouts per molecule and arm: in-window basins found (5 and 10 kJ/mol) of those in the union; the same restricted to
the first K picks (K = 8, 16, 32, 64) against R's first K starts; MLIP evaluations spent (search efforts: stage and
polish evaluations; single points count one each) and MLIP evaluations per in-window basin found; for 'sp', whether
the pick that relaxes into the lowest basin is the single-point lowest. Written to compare.json and printed.

usage: python -m eval.exp6_calibration.compare OUT TUNED_JSON {run,report,all} [--n 200] [--k 64] [--window 15]
           [--slots 8] [--molecules panel000,panel001]
OUT must hold panel.json and the conformers (collect.py select). Every search runs in its own process and is reused
when its minima exist, so a rerun resumes.
"""
import argparse
import json
import os

import numpy as np
import torch

from eval.exp6_calibration import common as C

MODULE = 'eval.exp6_calibration.compare'
WINDOWS = (5.0, 10.0)
KS = (8, 16, 32, 64)


def runs(name, tuned):
    return dict(R=f'{name}_{C.MLIP}_search', U=f'{name}_exp6_search', T=f'{name}_exp6_search_tuned',
                U_relax=f'{name}_{C.MLIP}_seeds_U', T_relax=f'{name}_{C.MLIP}_seeds_T')


def search_one(out, name, energy, n, slots, tag, exp6, seeds):
    from eval.exp6_recall.mirror_test import cap_gpu, run
    cap_gpu()
    sg = json.load(open(os.path.join(out, 'panel.json')))[name]['sg']
    mode = 'seeds' if seeds else 'search'
    # every arm at the same slots: the stream draws a condition's starts in blocks of min(slots, N), each block
    # seeded by its index, so equal slots (and N) is what gives R, T and U the same starts
    _, c = C.search_config(name, sg, energy, out, n, exp6=exp6, mode=mode, seeds=seeds, tag=tag, slots=slots,
                           trajectories=not seeds)  # the searches' first states prove the shared starts (report)
    if seeds:
        c['run_name'] = f'{name}_{energy}_seeds_{tag}'
    run(c)


def launch(out, name, energy, n, slots, tag='', exp6=None, seeds=None):
    args = ['search_one', '--name', name, '--energy', energy, '--n', n, '--slots', slots, '--tag', tag]
    if exp6:
        args += ['--exp6', exp6]
    if seeds:
        args += ['--seeds', seeds]
    return C.run_in_subprocess(MODULE, out, args)


def picks(out, name, run, window, k):
    """The exp-6 search's distinct minima within ``window`` of its lowest, lowest first, at most k."""
    from eval.exp6_recall.mlip_basins import window as basin_window
    rec = json.load(open(os.path.join(out, 'panel.json')))[name]
    mins = list(torch.load(os.path.join(out, run + '_minima.pt'), weights_only=False))
    win, _ = basin_window(mins, 'exp6', window, k, rec['sg'], os.path.join(out, rec['mol']))
    return win


def run_molecule(out, name, tuned, n, k, window, slots):
    from eval.exp6_recall.mirror_test import clean
    r = runs(name, tuned)
    have = lambda run: os.path.exists(os.path.join(out, run + '_minima.pt'))  # noqa: E731
    if not have(r['R']):
        launch(out, name, C.MLIP, n, slots)
    for arm, tag, exp6 in (('U', '', None), ('T', 'tuned', tuned)):
        if not have(r[arm]):
            launch(out, name, 'exp6', n, slots, tag=tag, exp6=exp6)
        seeds = os.path.join(out, f'{name}_{arm}_picks.pt')
        if not os.path.exists(seeds) and have(r[arm]):
            torch.save([clean(x) for x in picks(out, name, r[arm], window, k)], seeds)
        if os.path.exists(seeds) and not have(r[f'{arm}_relax']):
            launch(out, name, C.MLIP, 0, slots, tag=arm, seeds=seeds)


def single_points(out, name, arm):
    """MLIP energy (kJ/mol per molecule) of each pick, saved beside the picks."""
    path = os.path.join(out, f'{name}_{arm}_picks_sp.pt')
    if os.path.exists(path):
        return torch.load(path)
    from eval.exp6_recall.make_configs import SITES
    from eval.exp6_recall.mirror_test import cap_gpu
    from mxtaltools.crystal_search import minima as mn
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    cap_gpu()
    rows = torch.load(os.path.join(out, f'{name}_{arm}_picks.pt'), weights_only=False)
    pred = load_mace_model(SITES['local']['mace_csp'], 'cuda', torch.float32, dispersion=dict(xc='pbe', damping='bj'))
    se = mn.StageEnergy(rows, 'mace', 1.0, predictor=pred, device='cuda', chunk=4)
    e = se.f(torch.stack([x.full_cell_parameters().reshape(-1) for x in rows]).float(), list(range(len(rows))))
    torch.save(e, path)
    return e


def evals(out, run):
    p = os.path.join(out, run + '_stream_effort.json')
    if not os.path.exists(p):
        return None
    e = json.load(open(p))
    return int(sum(e['stage_evals']) + e['polish_evals'])


def start_mismatch(out, run_a, run_b):
    """Largest difference (any cell parameter) between the first recorded states of the starts two searches share,
    and how many they share; None when either search has no trajectory."""
    states = []
    for run in (run_a, run_b):
        p = os.path.join(out, run + '_traj.pt')
        if not os.path.exists(p):
            return None
        t = torch.load(p, weights_only=False)
        first = {}
        for i in torch.argsort(t['stage'] * 10 ** 6 + t['step']).tolist():
            first.setdefault(int(t['start_index'][i]), t['params'][i].double())
        states.append(first)
    common = sorted(set(states[0]) & set(states[1]))
    if not common:
        return dict(shared=0, max_diff=None)
    return dict(shared=len(common), max_diff=float(max((states[0][k] - states[1][k]).abs().max() for k in common)))


def report(out, tuned, names):
    from eval.exp6_recall.basins import basins
    from eval.exp6_recall.make_configs import KT
    from eval.exp6_recall.mirror_test import paired_minima
    panel = json.load(open(os.path.join(out, 'panel.json')))
    kT = KT[C.MLIP]
    res = {}
    for name in names:
        r = runs(name, tuned)
        need = [r['R'], r['U_relax'], r['T_relax']]
        if not all(os.path.exists(os.path.join(out, x + '_minima.pt')) for x in need):
            print(f'{name}: incomplete, skipped', flush=True)
            continue
        R_min = list(torch.load(os.path.join(out, r['R'] + '_minima.pt'), weights_only=False))
        R_all = list(torch.load(os.path.join(out, r['R'] + '.pt'), weights_only=False))
        byid = {int(x.minimum_id): x for x in R_min}
        arms = {'R': [(int(s.start_index), byid[int(s.minimum_id)]) for s in R_all
                      if 'minimum_id' in s.keys() and int(s.minimum_id) in byid]}
        for a in ('U', 'T'):
            m = list(torch.load(os.path.join(out, r[f'{a}_relax'] + '_minima.pt'), weights_only=False))
            s = list(torch.load(os.path.join(out, r[f'{a}_relax'] + '.pt'), weights_only=False))
            n_picks = len(torch.load(os.path.join(out, f'{name}_{a}_picks.pt'), weights_only=False))
            arms[a] = [(j, x) for j, x in enumerate(paired_minima(m, s, n_picks)) if x is not None]
        pool = [(a, j, x) for a, lst in arms.items() for j, x in lst]
        params = torch.stack([x.full_cell_parameters().reshape(-1) for _, _, x in pool]).double()
        hand = torch.stack([x.aunit_handedness.reshape(-1)[0] for _, _, x in pool])
        f = torch.tensor([float(x.minimum_f) for _, _, x in pool], dtype=torch.float64)
        lab, _ = basins(panel[name]['sg'], os.path.join(out, panel[name]['mol']), params, hand, f)
        lab = np.asarray(lab)
        e = (f.numpy() - f.numpy().min()) * kT
        b_e = {int(b): float(e[lab == b].min()) for b in set(lab.tolist())}
        mol = dict(n_basins=len(b_e), starts_R_vs_U=start_mismatch(out, r['R'], r['U']),
                   starts_R_vs_T=start_mismatch(out, r['R'], r['T']))
        for w in WINDOWS:
            target = {b for b, v in b_e.items() if v <= w}
            mol[f'in_window_{w:g}'] = len(target)
            for a in ('R', 'U', 'T'):
                sel = [i for i, (aa, _, _) in enumerate(pool) if aa == a]
                found = {int(lab[i]) for i in sel} & target
                mol[f'{a}_found_{w:g}'] = len(found)
                for K in KS:
                    first = {int(lab[i]) for i in sel if pool[i][1] < K} & target
                    mol[f'{a}_found_{w:g}_K{K}'] = len(first)
        cost = dict(R=evals(out, r['R']), U=evals(out, r['U_relax']), T=evals(out, r['T_relax']))
        for a in ('U', 'T'):
            sp = single_points(out, name, a)
            cost[f'{a}_sp'] = int(len(sp))
            low = min(b_e, key=b_e.get)
            hits = [j for (aa, j, _), l_ in zip(pool, lab) if aa == a and int(l_) == low]
            mol[f'{a}_sp_rank_of_lowest_basin_pick'] = (int((sp < sp[min(hits)]).sum()) if hits else None)
        mol['mlip_evals'] = cost
        for a in ('R', 'U', 'T'):
            found = mol[f'{a}_found_10']
            mol[f'{a}_evals_per_basin_10'] = (cost[a] + cost.get(f'{a}_sp', 0)) / found if found and cost[a] else None
        res[name] = mol
    json.dump(res, open(os.path.join(out, 'compare.json'), 'w'), indent=1)
    print('\nIn-window MLIP basins found (union of all arms = reference), per molecule; MLIP evaluations per arm.')
    print(f'{"molecule":10s} {"basins<=10":>10s} {"R":>4s} {"U":>4s} {"T":>4s} {"R evals":>9s} {"U evals":>9s} '
          f'{"T evals":>9s} {"T sp rank":>9s}')
    for n, m in res.items():
        print(f'{n:10s} {m["in_window_10"]:10d} {m["R_found_10"]:4d} {m["U_found_10"]:4d} {m["T_found_10"]:4d} '
              f'{m["mlip_evals"]["R"]:9d} {m["mlip_evals"]["U"]:9d} {m["mlip_evals"]["T"]:9d} '
              f'{str(m["T_sp_rank_of_lowest_basin_pick"]):>9s}')
        for k in ('starts_R_vs_U', 'starts_R_vs_T'):
            s = m[k]
            print(f'  {n} {k}: ' + ('no trajectory' if s is None else
                                    f'{s["shared"]} shared starts, largest first-state difference {s["max_diff"]}'))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('tuned', nargs='?')
    ap.add_argument('step', choices=['run', 'report', 'all', 'search_one'])
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--window', type=float, default=15.0)
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--molecules', default=None)
    ap.add_argument('--name')
    ap.add_argument('--energy')
    ap.add_argument('--tag', default='')
    ap.add_argument('--exp6', default=None)
    ap.add_argument('--seeds', default=None)
    a = ap.parse_args()
    if a.step == 'search_one':
        exp6 = {'params': a.exp6, 'charges': 'mmff'} if a.exp6 else None
        search_one(a.out, a.name, a.energy, a.n, a.slots, a.tag, exp6, a.seeds)
        return
    panel = json.load(open(os.path.join(a.out, 'panel.json')))
    names = a.molecules.split(',') if a.molecules else [k for k, v in panel.items() if v['role'] == 'panel']
    if a.step in ('run', 'all'):
        for name in names:
            run_molecule(a.out, name, os.path.abspath(a.tuned), a.n, a.k, a.window, a.slots)
    if a.step in ('report', 'all'):
        report(a.out, a.tuned, names)


if __name__ == '__main__':
    main()
