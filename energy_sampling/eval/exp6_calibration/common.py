"""Shared pieces of the exp-6 calibration cycle (docs/design/exp6_calibration_plan.md): molecule selection, search
configs, the MLIP, row templates and the intermolecular contact check.

Molecules come from training pool v1 (H, C, N, O, F; no Day benchmark molecule is in it): the fit set from the
'train' split, the comparison panel from 'heldout_scaffold', so no panel molecule shares a scaffold with the fit set.
Each molecule gets one RDKit ETKDG (seed 0) + MMFF94 conformer, held rigid, and one space group (alternating P2_1/c
and P-1 down the seeded draw).
"""
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import torch

POOL = 'D:/crystal_datasets/paper_review_nov02/pool/pool_v1.csv.gz'
SEED = 20261124
SGS = (14, 2)
MAX_HEAVY = 16
MLIP = 'mace_csp'  # make_configs energy: MACE-CSP-25.2 large + D3(BJ, PBE)
EXP6_FIT = {'params': 'fit', 'charges': 'mmff'}  # the published FIT set with MMFF charges: the untuned surrogate
EV_TO_KJMOL = 96.485
MIN_CONTACT_A = 1.2  # shortest intermolecular contact below which a frame is a collapse, not data (H...H >= ~1.8)


def selection(out, n_fit, n_panel):
    """panel.json in ``out``: n_fit 'train' molecules (role 'fit') and n_panel 'heldout_scaffold' molecules (role
    'panel'), each with a conformer file. Seeded; an embedding failure takes the next molecule of the draw."""
    import pandas as pd
    from rdkit import Chem
    from eval.exp6_recall.panel import rdkit_mol
    from eval.exp6_recall.transfer_check import selection_class
    path = os.path.join(out, 'panel.json')
    if os.path.exists(path):
        return json.load(open(path))
    os.makedirs(out, exist_ok=True)
    pool = pd.read_csv(POOL, usecols=['smiles', 'n_heavy', 'mw', 'split'])
    pool = pool[(pool.n_heavy >= 5) & (pool.n_heavy <= MAX_HEAVY)]
    panel = {}
    for role, split, n in (('fit', 'train', n_fit), ('panel', 'heldout_scaffold', n_panel)):
        draw = pool[pool.split == split].sample(min(20 * n, int((pool.split == split).sum())), random_state=SEED)
        k = 0
        for _, row in draw.iterrows():
            if k == n:
                break
            m = Chem.MolFromSmiles(row.smiles)
            if m is None or Chem.GetFormalCharge(m) != 0 or '.' in row.smiles:
                continue
            name = f'{role}{k:03d}'
            try:
                mol = rdkit_mol(name, row.smiles)
            except Exception as e:  # noqa: BLE001  (an embedding failure: next molecule)
                print(f'{row.smiles}: {type(e).__name__}: {e}', flush=True)
                continue
            torch.save([mol], os.path.join(out, f'{name}_mol.pt'))
            inside, ch3, rot = selection_class(m)
            panel[name] = dict(role=role, smiles=row.smiles, n_heavy=int(row.n_heavy), mw=float(row.mw), split=split,
                               sg=SGS[k % len(SGS)], inside_mace_csp_rules=inside, ch3=ch3, n_rot=rot,
                               mol=f'{name}_mol.pt')
            k += 1
        if k < n:
            raise SystemExit(f'{role}: draw exhausted at {k} of {n}')
    json.dump(panel, open(path, 'w'), indent=1)
    return panel


def search_config(name, sg, energy, out, n, seed_offset=0, trajectories=False, exp6=None, mode='search', seeds=None,
                  tag='', slots=None):
    """make_configs.config for one search, every start run (stop_p off); ``exp6`` the exp6 stage options (a
    parameter set by name or JSON path, and charges); ``trajectories`` records every evaluated state."""
    from eval.exp6_recall.make_configs import config
    run, c = config(name, sg, energy, mode, out, out, 'local', n=n, seeds=seeds, tag=tag)
    c['opt_seed'] += seed_offset
    c['stream']['stop_p'] = None
    if slots:
        c['batch_size'] = c['stream']['slots'] = int(slots)
    if trajectories:
        c['stream']['trajectories'] = {'every': 1}
    if energy == 'exp6':
        for st in c['opt']:
            st['exp6_kwargs'] = dict(exp6 or EXP6_FIT)
    return run, c


def run_in_subprocess(module, out, args):
    """One search in its own process, so GPU memory held by an earlier one cannot starve it."""
    return subprocess.run([sys.executable, '-u', '-m', module, out] + [str(a) for a in args]).returncode


def per_molecule(r):
    return float(r.sym_mult) * float(r.z_prime.reshape(-1)[0])


SHIFTS = torch.tensor(list(itertools.product((-2, -1, 0, 1, 2), repeat=3)), dtype=torch.float32)


def shortest_contact(r):
    """Shortest distance (A) between atoms of different molecules of a Z' = 1 crystal row."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    b = collate_data_list([r.clone()])
    b.pose_aunit()
    b.build_unit_cell()
    pos = b.unit_cell_pos.float()
    n_mol = int(round(per_molecule(r)))
    mol = torch.arange(n_mol).repeat_interleave(len(r.z))
    T = b.T_fc[0].float()
    best = float('inf')
    for s in SHIFTS:
        d = torch.cdist(pos, pos + (T @ s).reshape(1, 3))
        if bool((s == 0).all()):
            d[mol[:, None] == mol[None, :]] = float('inf')
        best = min(best, float(d.min()))
    return best


def template(r):
    """A clean copy of a search row to rebuild crystals of its molecule and space group at other parameters."""
    from eval.exp6_recall.mirror_test import clean
    t = clean(r)
    for k in ('minimum_f', 'hessian', 'id_hessian', 'mace', 'exp6', 'elj', 'lj'):
        if k in t.keys():
            delattr(t, k)
    return t


def rows_at(tmpl, params, hand):
    """[n] crystal rows of ``tmpl``'s molecule and group at parameters [n, d] with handedness [n]."""
    out = []
    for p, h in zip(params, hand):
        r = tmpl.clone()
        # box analysis ON: it recomputes T_fc; skipped, the row keeps the template's lattice and builds a wrong crystal
        r.set_cell_parameters(torch.as_tensor(p, dtype=torch.float32).reshape(1, -1), skip_box_analysis=False)
        r.aunit_handedness = torch.full_like(r.aunit_handedness, float(h))
        out.append(r)
    return out


def bands(e, edges=(0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)):
    """Band index of each energy above its minimum (kJ/mol); -1 beyond the last edge."""
    e = np.asarray(e, dtype=float) - np.min(e)
    out = np.searchsorted(np.asarray(edges[1:]), e, side='right')
    out[e > edges[-1]] = -1
    return out
