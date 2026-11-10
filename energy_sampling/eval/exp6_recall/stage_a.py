"""Stage A of the exp-6 recall test: what each energy does to the experimental anchors.

For each panel molecule with an anchor, and each energy (eLJ, exp-6, eSEN-S), from the anchor-seeded runs in
STAGE_DIR (``make_configs`` mode seeds): the energy at the experimental structure and at the polished end, the
polish status, the symmetry-aware FD distance from the anchor to the end (``basins.SymFD``) and the end's closest
intermolecular contact. With RERELAX dirs (eSEN-S runs seeded by the exp-6 or eLJ anchor ends), also whether the
eSEN-S relaxation of the cheap end lands in the eSEN-S basin of the anchor.

usage: python -m eval.exp6_recall.stage_a PANEL_DIR STAGE_DIR [RERELAX_EXP6_DIR RERELAX_ELJ_DIR]
"""
import json
import os
import sys

import torch

from eval.exp6_recall.basins import SymFD
from eval.exp6_recall.make_configs import KT

ENERGIES = ('elj', 'exp6', 'uma')
UNIT = {'elj': 'eLJ', 'exp6': 'kJ/mol', 'uma': 'kJ/mol'}


def energy_at(row, key, predictor=None):
    from mxtaltools.crystal_search.minima import StageEnergy
    E = StageEnergy([row], key, KT[key], predictor=predictor, device='cuda', chunk=4, lj_envelope=1.0)
    x = row.full_cell_parameters().double()
    return float(E.f(x, [0])[0]) * KT[key]


def contact(row):
    from mxtaltools.crystal_search.run_search import closest_intermolecular_contact
    from mxtaltools.dataset_utils.utils import collate_data_list
    return float(closest_intermolecular_contact(collate_data_list([row]).to('cuda'))[0])


def load_end(path):
    if not os.path.exists(path):
        return None
    rows = torch.load(path, weights_only=False)
    return rows[0] if rows else None


def main(panel_dir, stage_dir, rerelax=None):
    from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
    torch.cuda.set_per_process_memory_fraction(0.4, 0)
    pred = init_uma_crystal_predictor('D:/crystal_datasets/esen_s.pt', device='cuda', activation_checkpointing=False,
                                      tf32=False)
    panel = json.load(open(os.path.join(panel_dir, 'panel.json')))
    out = {}
    for name, rec in panel.items():
        if not rec.get('anchor'):
            continue
        a = torch.load(os.path.join(panel_dir, rec['anchor']), weights_only=False)[0]
        mol = os.path.join(panel_dir, rec['mol'])
        res = dict(anchor_contact_A=contact(a))
        ends = {}
        for e in ENERGIES:
            r = load_end(os.path.join(stage_dir, f'{name}_{e}_seeds.pt'))
            if r is None:
                continue
            ends[e] = r
            res[e] = dict(status=int(r.polish_status), E_anchor=energy_at(a, e, pred if e == 'uma' else None),
                          E_end=float(r.minimum_f) * KT[e], contact_A=contact(r))
        for tag, d in (rerelax or {}).items():
            r = load_end(os.path.join(d, f'{name}_uma_seeds.pt'))
            if r is not None:
                ends[f'uma_from_{tag}'] = r
                res[f'uma_from_{tag}'] = dict(status=int(r.polish_status), E_end=float(r.minimum_f) * KT['uma'],
                                             contact_A=contact(r))
        keys = list(ends)
        rows = [a] + [ends[k] for k in keys]
        params = torch.stack([r.full_cell_parameters().reshape(-1) for r in rows]).double()
        hand = torch.stack([r.aunit_handedness.reshape(-1)[0] for r in rows])
        fd = SymFD(rec['sg_searched'], mol, params, hand)
        n = len(rows)
        ia = torch.arange(n).repeat_interleave(n)
        ib = torch.arange(n).repeat(n)
        D = fd.pairs(ia, ib).reshape(n, n)
        res['fd_A'] = {('anchor' if i == 0 else keys[i - 1]): {('anchor' if j == 0 else keys[j - 1]): float(D[i, j])
                                                               for j in range(n)} for i in range(n)}
        res['n_self_symmetries'] = fd.k
        out[name] = res
        print(name, json.dumps(res, default=float))
    with open(os.path.join(stage_dir, 'stage_a.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print_tables(out)


def print_tables(out):
    print('\nTable 1. Each energy relaxing the experimental structure (anchor), rigid molecule, searched space group. '
          'E in kJ/mol per molecule (eLJ in its own units); FD = symmetry-aware FD distance (A) anchor -> end; '
          'status 1 converged, 2 stalled, 4 left the cell box; contact = closest intermolecular pair (A).')
    print(f"{'molecule':13s} | {'energy':6s} | {'status':6s} | {'E(anchor)':>10s} | {'E(end)':>10s} | "
          f"{'dE relax':>9s} | {'FD anchor->end (A)':>18s} | {'contact (A)':>11s}")
    for name, res in out.items():
        for e in ENERGIES:
            if e not in res:
                continue
            r = res[e]
            print(f"{name:13s} | {e:6s} | {r['status']:6d} | {r['E_anchor']:10.2f} | {r['E_end']:10.2f} | "
                  f"{r['E_end'] - r['E_anchor']:9.2f} | {res['fd_A']['anchor'][e]:18.3f} | {r['contact_A']:11.2f}")
    print('\nTable 2. eSEN-S relaxation of each cheap energy\'s anchor end: FD (A) from the eSEN-S anchor end '
          '(< 0.05 = same basin) and its eSEN-S energy above the anchor end (kJ/mol).')
    print(f"{'molecule':13s} | {'seed':10s} | {'status':6s} | {'FD to eSEN-S anchor end (A)':>27s} | "
          f"{'E - E(anchor end)':>17s}")
    for name, res in out.items():
        for tag in ('exp6', 'elj'):
            k = f'uma_from_{tag}'
            if k not in res or 'uma' not in res:
                continue
            print(f"{name:13s} | {tag:10s} | {res[k]['status']:6d} | {res['fd_A']['uma'][k]:27.3f} | "
                  f"{res[k]['E_end'] - res['uma']['E_end']:17.2f}")


if __name__ == '__main__':
    rr = {'exp6': sys.argv[3], 'elj': sys.argv[4]} if len(sys.argv) > 4 else None
    main(sys.argv[1], sys.argv[2], rr)
