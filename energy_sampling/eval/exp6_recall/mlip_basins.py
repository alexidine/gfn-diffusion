"""Do two MLIPs put a molecule's crystal basins in the same places? A symmetric basin-correspondence test.

Models (make_configs energies): 'mace_csp' (MACE-CSP-25.2 large + D3(BJ, PBE)) and 'mace' (MACE-OFF24 medium).
Per molecule (panel and probe set; probes carry CH3 and torsions outside MACE-CSP-25.2's training selection):
  1. a random-start search under each model (same starts: same opt_seed), polished minima;
  2. the experimental anchor relaxed under each model;
  3. cross-relaxation: each model's basins within WINDOW of its lowest (at most MAX_X), relaxed under the other.
Readouts (``readout``), per direction A -> B over A's basins in the window:
  landed   on a basin B's own search found (identity: basins.basins over B's search minima + the ends, FD 0.05 A);
  new      on a B minimum B's search did not find;
  distinct B basins reached (merges = landed + new - distinct); median FD moved; Spearman of A vs B energies;
and per model: the anchor basin's rank among the model's basins and its energy above the lowest; packing coefficient
range of the minima within the window (an untrained region shows up as minima far denser or sparser than the rest).

usage: python -m eval.exp6_recall.mlip_basins OUT_DIR [--molecules imidazole,thymine,...] [--n 1000] [--window 10]
           [--max_x 40] [--energies mace_csp,mace]
Runs that already have minima in OUT_DIR are reused (mirror_test.run), so a rerun resumes.
"""
import argparse
import json
import os

import torch

from eval.exp6_recall.basins import SymFD, basins
from eval.exp6_recall.make_configs import KT, config
from eval.exp6_recall.mirror_test import cap_gpu, clean, paired_minima, run

PANELS = ['D:/crystal_datasets/exp6_recall_nov10/panel', 'D:/crystal_datasets/exp6_recall_nov10/panel_probes_nov18']
DEFAULT = 'imidazole,thymine,p_toluic_acid,acetanilide,paracetamol_I,paracetamol_II'
ld = lambda p: list(torch.load(p, weights_only=False))


def where(name):
    for d in PANELS:
        p = json.load(open(os.path.join(d, 'panel.json')))
        if name in p:
            return d, p[name]
    raise KeyError(name)


def E(r, energy):
    return float(r.minimum_f) * KT[energy]


def window(minima, energy, w, cap, sg, mol):
    """Lowest member of each basin within w kJ/mol of the lowest, at most cap, lowest first."""
    params = torch.stack([r.full_cell_parameters().reshape(-1) for r in minima]).double()
    hand = torch.stack([r.aunit_handedness.reshape(-1)[0] for r in minima])
    f = torch.tensor([float(r.minimum_f) for r in minima], dtype=torch.float64)
    lab, _ = basins(sg, mol, params, hand, f, df_kT=0.1)
    rep = {}
    for i in range(len(minima)):
        b = int(lab[i])
        rep[b] = i if b not in rep or f[i] < f[rep[b]] else rep[b]
    e0 = float(f.min())
    keep = sorted([i for i in rep.values() if (float(f[i]) - e0) * KT[energy] <= w], key=lambda i: float(f[i]))
    return [minima[i] for i in keep[:cap]], len(keep)


def seeds_run(name, sg, energy, rows, tag, panel, out):
    sp = os.path.join(out, f'{name}_{tag}_seeds.pt')
    if not os.path.exists(sp):
        torch.save([clean(r) for r in rows], sp)
    _, c = config(name, sg, energy, 'seeds', panel, out, 'local', seeds=sp)
    c['run_name'] = f'{name}_{energy}_{tag}'
    mins, starts = run(c)
    return paired_minima(mins, starts, len(rows))


def packing(r):
    for k in ('packing_coeff', 'reduced_volume'):
        if hasattr(r, k) and getattr(r, k) is not None:
            return k, float(getattr(r, k).reshape(-1)[0])
    return None, None


def molecule(name, a, out):
    panel, rec = where(name)
    sg, mol = int(rec['sg_searched']), os.path.join(panel, rec['mol'])
    ens = a.energies.split(',')
    res = dict(sg=sg, models={}, cross={})
    search, anchor, win = {}, {}, {}
    for en in ens:
        _, c = config(name, sg, en, 'search', panel, out, 'local', n=a.n)
        search[en], _ = run(c)
        anchor[en] = seeds_run(name, sg, en, ld(os.path.join(panel, rec['anchor'])), 'anchor', panel, out)[0]
        win[en], n_win = window(search[en], en, a.window, a.max_x, sg, mol)
        res['models'][en] = dict(n_minima=len(search[en]), n_basins_in_window=n_win)
    for A in ens:
        for B in ens:
            if A == B:
                continue
            ends = seeds_run(name, sg, B, win[A], f'from_{A}', panel, out)
            ok = [j for j, e in enumerate(ends) if e is not None]
            pool = search[B] + [ends[j] for j in ok] + ([anchor[B]] if anchor[B] is not None else [])
            params = torch.stack([r.full_cell_parameters().reshape(-1) for r in pool]).double()
            hand = torch.stack([r.aunit_handedness.reshape(-1)[0] for r in pool])
            f = torch.tensor([float(r.minimum_f) for r in pool], dtype=torch.float64)
            lab, _ = basins(sg, mol, params, hand, f, df_kT=0.1)
            nS = len(search[B])
            found = set(int(x) for x in lab[:nS])
            end_lab = [int(lab[nS + k]) for k in range(len(ok))]
            # B's basin energies relative to B's own lowest, for the reached basins
            e0B = min(E(r, B) for r in search[B])
            # FD moved: start (A minimum) to end (B minimum)
            both = [win[A][j] for j in ok] + [ends[j] for j in ok]
            p2 = torch.stack([r.full_cell_parameters().reshape(-1) for r in both]).double()
            h2 = torch.stack([r.aunit_handedness.reshape(-1)[0] for r in both])
            D = SymFD(sg, mol, p2, h2).pairs(list(range(len(ok))), [len(ok) + i for i in range(len(ok))])
            eA = torch.tensor([E(win[A][j], A) for j in ok], dtype=torch.float64)
            eB = torch.tensor([E(ends[j], B) for j in ok], dtype=torch.float64)
            rk = lambda v: torch.argsort(torch.argsort(v)).double()
            sp = float(torch.corrcoef(torch.stack([rk(eA), rk(eB)]))[0, 1]) if len(ok) > 2 else float('nan')
            res['cross'][f'{A}->{B}'] = dict(
                n=len(win[A]), converged=len(ok), landed=sum(l in found for l in end_lab),
                new=sum(l not in found for l in end_lab), distinct=len(set(end_lab)),
                fd_median=float(D.median()) if len(ok) else float('nan'), fd_max=float(D.max()) if len(ok) else float('nan'),
                spearman=sp, end_above_B_lowest=[float(x - e0B) for x in eB.tolist()],
                end_below_B_lowest=int(sum(float(x) < e0B - 0.5 for x in eB.tolist())))
    for en in ens:
        m = res['models'][en]
        e0 = min(E(r, en) for r in search[en])
        if anchor[en] is not None:
            pool = search[en] + [anchor[en]]
            params = torch.stack([r.full_cell_parameters().reshape(-1) for r in pool]).double()
            hand = torch.stack([r.aunit_handedness.reshape(-1)[0] for r in pool])
            f = torch.tensor([float(r.minimum_f) for r in pool], dtype=torch.float64)
            lab, _ = basins(sg, mol, params, hand, f, df_kT=0.1)  # labels in order of each basin's lowest E
            m['anchor_basin_rank'] = int(lab[-1]) + 1
            m['anchor_found_by_search'] = bool(int(lab[-1]) in set(int(x) for x in lab[:-1]))
            m['anchor_above_lowest_kJ'] = E(anchor[en], en) - min(e0, E(anchor[en], en))
            m['anchor_below_search_lowest_kJ'] = max(0.0, e0 - E(anchor[en], en))
        k, _ = packing(win[en][0])
        if k:
            v = [packing(r)[1] for r in win[en]]
            m['window_' + k] = [min(v), max(v)]
            m['anchor_' + k] = packing(anchor[en])[1] if anchor[en] is not None else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--molecules', default=DEFAULT)
    ap.add_argument('--energies', default='mace_csp,mace')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--window', type=float, default=10.0)
    ap.add_argument('--max_x', type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cap_gpu()
    path = os.path.join(a.out_dir, 'mlip_basins.json')
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name in a.molecules.split(','):
        out[name] = molecule(name, a, a.out_dir)
        with open(path, 'w') as fh:
            json.dump(out, fh, indent=1)
        r = out[name]
        print('BASINS', name, json.dumps(r['models']), flush=True)
        for k, v in r['cross'].items():
            print('BASINS', name, k, json.dumps({q: w for q, w in v.items() if q != 'end_above_B_lowest'}), flush=True)


if __name__ == '__main__':
    main()
