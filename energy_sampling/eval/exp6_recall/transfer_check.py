"""Do MACE-CSP-25.2's within-molecule crystal energies transfer to pool molecules outside its training selection?

Reference: UMA-m 1.1 ('omc' task), trained on OMC25 (VASP PBE + D3 with zero damping, IVDW = 11). MACE-CSP-25.2 is
PBE without dispersion, so here it gets the same D3: zero damping, 30 A pair cutoff, 20 A coordination cutoff. The
reference is a model, not DFT: agreement is evidence that the energies transfer; a disagreement does not say which
model is wrong.

Molecules: a seeded random draw from training pool v1, 20 outside MACE-CSP-25.2's selection rules and 5 inside as a
control; one RDKit ETKDG (seed 0) + MMFF94 conformer each, held rigid. Per molecule: a 25-start MACE-CSP L + D3(BJ)
search in P2_1/c (the first three molecules ran 50 starts; structures.json records each), and up to 5 distinct minima
from the lowest to about +15 kJ/mol. Single points on those structures, crystal energy per molecule (no gas leg),
compared within the molecule.

Criterion, fixed before the run. With d = E(MACE-CSP) - E(UMA-m) over a molecule's structures and e = d - median(d):
    pass  RMS(e) <= 4 kJ/mol and min(e) >= -10 kJ/mol (no structure far too low under MACE-CSP)
A molecule with fewer than 3 structures is not evaluable and is reported as such.

usage: python -m eval.exp6_recall.transfer_check OUT_DIR {select,search,score,report,all} [--slots 8]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

POOL = 'D:/crystal_datasets/paper_review_nov02/pool/pool_v1.csv.gz'
UMA_M = 'D:/crystal_datasets/uma-m-1p1.pt'
SEED = 20261123
N_OUTSIDE, N_INSIDE = 20, 5
SG = 14
N_STARTS = 25
N_ATTEMPTS = 2
N_STRUCTURES = 5
WINDOW_KJ = 15.0
RMS_LIMIT_KJ, LOW_LIMIT_KJ = 4.0, -10.0
EV_TO_KJMOL = 96.485
D3_ZERO = dict(xc='pbe', damping='zero', cutoff=30.0, cnthr=20.0)
D3_BJ = dict(xc='pbe', damping='bj')


def selection_class(m):
    """CSP-25.2's selection rules, approximated with RDKit: (inside, carries a CH3, RDKit rotatable bonds)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    n = {k: len(m.GetSubstructMatches(Chem.MolFromSmarts(v)))
         for k, v in dict(ch3='[CH3]', oh='[OX2H]', nh2='[NX3H2]', cooh='C(=O)[OX2H]').items()}
    rot, mw = rdMolDescriptors.CalcNumRotatableBonds(m), Descriptors.MolWt(m)
    polar = n['oh'] + n['nh2']
    rigid = rot == 0 and n['ch3'] == 0 and polar == 0
    one_polar = rot == 0 and n['ch3'] == 0 and n['cooh'] == 0 and polar == 1 and mw <= 200
    one_acid = rot <= 1 and n['ch3'] == 0 and n['cooh'] == 1 and n['oh'] == 1 and n['nh2'] == 0 and mw <= 200
    return bool(rigid or one_polar or one_acid), n['ch3'] > 0, int(rot)


def verdict(e_mace, e_uma):
    """Within-molecule comparison of two [n] energy arrays (kJ/mol per molecule); see the module docstring."""
    d = np.asarray(e_mace, dtype=float) - np.asarray(e_uma, dtype=float)
    if len(d) < 3 or not np.isfinite(d).all():
        return dict(evaluable=False, passed=None, rms_kJ=None, lowest_kJ=None)
    e = d - np.median(d)
    rms, low = float(np.sqrt(np.mean(e ** 2))), float(e.min())
    return dict(evaluable=True, passed=bool(rms <= RMS_LIMIT_KJ and low >= LOW_LIMIT_KJ), rms_kJ=rms, lowest_kJ=low)


def pick_structures(energies, n=N_STRUCTURES, span=WINDOW_KJ):
    """Indices into ascending `energies` (kJ/mol): the lowest, then the nearest unused to evenly spaced targets."""
    e = np.asarray(energies, dtype=float)
    if len(e) <= n:
        return list(range(len(e)))
    keep = [0]
    for target in e[0] + span * np.arange(1, n) / (n - 1):
        free = [i for i in range(len(e)) if i not in keep]
        keep.append(min(free, key=lambda i: abs(e[i] - target)))
    return sorted(keep)


def select(out):
    import pandas as pd
    from rdkit import Chem
    from eval.exp6_recall.panel import rdkit_mol
    path = os.path.join(out, 'panel.json')
    if os.path.exists(path):
        return
    os.makedirs(out, exist_ok=True)
    pool = pd.read_csv(POOL, usecols=['smiles', 'n_heavy', 'mw', 'split'])
    draw = pool.sample(5000, random_state=SEED).reset_index(drop=True)
    del pool
    panel, need = {}, {True: N_INSIDE, False: N_OUTSIDE}
    for k, row in draw.iterrows():
        if not any(need.values()):
            break
        m = Chem.MolFromSmiles(row.smiles)
        if m is None or Chem.GetFormalCharge(m) != 0:
            continue
        inside, ch3, rot = selection_class(m)
        if need[inside] == 0:
            continue
        name = f'{"in" if inside else "out"}{(N_INSIDE if inside else N_OUTSIDE) - need[inside]:02d}'
        try:
            mol = rdkit_mol(name, row.smiles)
        except Exception as e:  # an embedding failure takes the next molecule of the draw
            print(f'{row.smiles}: {type(e).__name__}: {e}', flush=True)
            continue
        torch.save([mol], os.path.join(out, f'{name}_mol.pt'))
        panel[name] = dict(smiles=row.smiles, inside=inside, ch3=ch3, n_rot=rot, n_heavy=int(row.n_heavy),
                           mw=float(row.mw), split=row.split, sg_searched=SG, mol=f'{name}_mol.pt', draw_index=int(k))
        need[inside] -= 1
    if any(need.values()):
        raise SystemExit(f'draw exhausted: still need {need}')
    json.dump(panel, open(path, 'w'), indent=1)
    print(f'{len(panel)} molecules selected', flush=True)


def run_name(name, attempt):
    return f'{name}_mace_csp_search' + (f'_retry{attempt}' if attempt else '')


def search_one(out, name, slots, attempt):
    """One molecule's search, in its own process: GPU memory held by an earlier search cannot starve this one."""
    from eval.exp6_recall.make_configs import config
    from eval.exp6_recall.mirror_test import cap_gpu, run
    cap_gpu()
    _, c = config(name, SG, 'mace_csp', 'search', out, out, 'local', n=N_STARTS, tag=f'retry{attempt}' if attempt else '')
    c['opt_seed'] += attempt
    c['batch_size'] = slots
    c['stream']['slots'] = slots
    c['stream']['stop_p'] = None  # always the full N_STARTS
    run(c)


def search(out, slots, finished_only=False):
    """With finished_only, no new search starts: the panel is cut to the molecules whose search already finished."""
    from eval.exp6_recall.mlip_basins import window, E
    panel = json.load(open(os.path.join(out, 'panel.json')))
    path = os.path.join(out, 'structures.pt')
    if os.path.exists(path):
        return
    picked, record = {}, {}
    for name, rec in panel.items():
        done = None
        for attempt in range(N_ATTEMPTS):  # bounded: a molecule that fails every attempt is reported as not evaluable
            f = os.path.join(out, run_name(name, attempt) + '_minima.pt')
            if not os.path.exists(f) and not finished_only:
                subprocess.run([sys.executable, '-u', '-m', 'eval.exp6_recall.transfer_check', out, 'search_one',
                                '--name', name, '--attempt', str(attempt), '--slots', str(slots)])
            if os.path.exists(f):
                done = (attempt, f)
                break
        if done is None and finished_only:
            continue
        if done is None:
            print(f'{name}: search failed in all {N_ATTEMPTS} attempts', flush=True)
            picked[name], record[name] = [], dict(n_minima=0, n_basins=0, energies=[], failed=True)
            continue
        minima = list(torch.load(done[1], weights_only=False))
        effort = json.load(open(os.path.join(out, run_name(name, done[0]) + '_stream_effort.json')))
        if not minima:
            print(f'{name}: no minima', flush=True)
            picked[name], record[name] = [], dict(n_minima=0, n_basins=0, energies=[], n_starts=effort['rows'])
            continue
        win, n_win = window(minima, 'mace_csp', WINDOW_KJ, 10 ** 6, SG, os.path.join(out, rec['mol']))
        e = [E(r, 'mace_csp') for r in win]
        keep = pick_structures(e)
        picked[name] = [win[i] for i in keep]
        record[name] = dict(n_minima=len(minima), n_basins=n_win, energies=[e[i] for i in keep],
                            n_starts=effort['rows'], attempt=done[0])
        print(f'{name}: {effort["rows"]} starts, {len(minima)} minima, {n_win} basins within {WINDOW_KJ} kJ/mol, '
              f'kept {len(keep)}', flush=True)
        torch.save(picked, path + '.partial')
    torch.save(picked, path)
    if os.path.exists(path + '.partial'):
        os.remove(path + '.partial')
    json.dump(record, open(os.path.join(out, 'structures.json'), 'w'), indent=1)


def per_molecule(rows, key, pred, chunk):
    """[n] crystal energy, kJ/mol per molecule."""
    from eval.energy_baseline.baseline_fit import score
    from eval.exp6_recall.csd_parity import strip
    from eval.exp6_recall.mirror_test import clean
    res = score([strip(clean(r)) for r in rows], (key,), pred, chunk)
    per = np.array([float(r.sym_mult) * float(r.z_prime.reshape(-1)[0]) for r in rows])
    return res[key].numpy() / per * EV_TO_KJMOL


def score_all(out):
    from eval.exp6_recall.make_configs import SITES
    from eval.exp6_recall.mirror_test import cap_gpu
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
    path = os.path.join(out, 'scores.json')
    if os.path.exists(path):
        return
    cap_gpu()
    picked = torch.load(os.path.join(out, 'structures.pt'), weights_only=False)
    record = json.load(open(os.path.join(out, 'structures.json')))
    names = [n for n, v in picked.items() if v]
    rows = [r for n in names for r in picked[n]]
    owner = [n for n in names for _ in picked[n]]
    E = {}
    for tag, d3 in (('mace_csp_bj', D3_BJ), ('mace_csp_zero', D3_ZERO)):
        p = load_mace_model(SITES['local']['mace_csp'], 'cuda', torch.float32, dispersion=d3)
        E[tag] = per_molecule(rows, 'mace_pot', p, 1)
        del p
        torch.cuda.empty_cache()
    # the structures were picked at the search's energies: its relative energies must come back through this route
    stored = np.array([x for n in names for x in record[n]['energies']])
    worst = 0.0
    for n in names:
        k = np.array([o == n for o in owner])
        worst = max(worst, float(np.abs((E['mace_csp_bj'][k] - E['mace_csp_bj'][k].mean())
                                        - (stored[k] - stored[k].mean())).max()))
    print(f'search vs single-point relative energies (D3 BJ): worst difference {worst:.3f} kJ/mol', flush=True)
    if worst > 1.0:
        raise SystemExit('single points do not reproduce the search energies')
    p = init_uma_crystal_predictor(UMA_M, device='cuda', activation_checkpointing=True, tf32=False)
    E['uma_m'] = per_molecule(rows, 'uma_pot', p, 1)
    json.dump(dict(owner=owner, route_check_worst_kJ=worst, energies={k: v.tolist() for k, v in E.items()}),
              open(path, 'w'), indent=1)


def uma_cpu(r, pred):
    """One row's UMA-m crystal energy on the CPU, kJ/mol per molecule."""
    from eval.exp6_recall.csd_parity import strip
    from eval.exp6_recall.mirror_test import clean
    from mxtaltools.dataset_utils.utils import collate_data_list
    b = collate_data_list([strip(clean(r))]).to('cpu')
    with torch.no_grad():
        res = b.analyze(computes=['uma_pot'], cutoff=10, supercell_size=10, std_orientation=True, assign_outputs=False,
                        predictor=pred)
    return float(res['uma_pot'].reshape(-1)[0]) / (float(r.sym_mult) * float(r.z_prime.reshape(-1)[0])) * EV_TO_KJMOL


def fill_uma(out):
    """Score on the CPU the UMA-m rows that exceed the GPU share; three rows already scored on the GPU are repeated
    first and must come back within 0.1 kJ/mol. Each row is saved to uma_fill.json, so a rerun resumes."""
    import shutil
    import time
    from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
    path = os.path.join(out, 'scores.json')
    fill_path = os.path.join(out, 'uma_fill.json')
    s = json.load(open(path))
    picked = torch.load(os.path.join(out, 'structures.pt'), weights_only=False)
    rows = [r for n in dict.fromkeys(s['owner']) for r in picked[n]]
    u = np.array(s['energies']['uma_m'], dtype=float)
    missing = np.flatnonzero(~np.isfinite(u))
    control = np.flatnonzero(np.isfinite(u))[:3]
    if not len(missing) or len(control) < 3:
        raise SystemExit(f'{len(missing)} rows to fill, {len(control)} control rows')
    filled = {int(k): v for k, v in json.load(open(fill_path)).items()} if os.path.exists(fill_path) else {}
    p = init_uma_crystal_predictor(UMA_M, device='cpu', activation_checkpointing=True, tf32=False)
    t = time.time()
    worst = max(abs(uma_cpu(rows[i], p) - u[i]) for i in control)
    print(f'repeated UMA-m rows, CPU against GPU: worst difference {worst:.4f} kJ/mol; '
          f'{(time.time() - t) / 3:.0f} s per row', flush=True)
    if not worst <= 0.1:
        raise SystemExit('the CPU route does not reproduce the stored UMA-m energies')
    for i in missing.tolist():
        if i not in filled:
            filled[i] = uma_cpu(rows[i], p)
            json.dump(filled, open(fill_path, 'w'))
            print(f'{s["owner"][i]} row {i}: {len(filled)}/{len(missing)}', flush=True)
    u[missing] = [filled[i] for i in missing.tolist()]
    shutil.copy(path, path + '.before_fill')
    s['energies']['uma_m'] = u.tolist()
    s['uma_fill_control_worst_kJ'] = float(worst)
    s['uma_filled_on_cpu'] = missing.tolist()
    json.dump(s, open(path, 'w'), indent=1)


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def report(out):
    searched = json.load(open(os.path.join(out, 'structures.json')))  # a cut run holds only its finished molecules
    panel = {n: v for n, v in json.load(open(os.path.join(out, 'panel.json'))).items() if n in searched}
    s = json.load(open(os.path.join(out, 'scores.json')))
    owner = np.array(s['owner'])
    E = {k: np.array(v) for k, v in s['energies'].items()}
    res = {}
    for name, rec in panel.items():
        k = owner == name
        v = verdict(E['mace_csp_zero'][k], E['uma_m'][k])
        v_bj = verdict(E['mace_csp_bj'][k], E['uma_m'][k])
        span = float(np.ptp(E['uma_m'][k])) if k.any() else None
        res[name] = dict(smiles=rec['smiles'], inside=rec['inside'], ch3=rec['ch3'], n_rot=rec['n_rot'], n=int(k.sum()),
                         uma_m_span_kJ=span, **v, rms_bj_kJ=v_bj['rms_kJ'], passed_bj=v_bj['passed'])
    groups = {'outside the rules': [v for v in res.values() if not v['inside']],
              '  of which with CH3': [v for v in res.values() if not v['inside'] and v['ch3']],
              '  of which one torsion': [v for v in res.values() if not v['inside'] and v['n_rot'] == 1],
              'inside the rules (control)': [v for v in res.values() if v['inside']]}
    summary = {}
    print('\nMACE-CSP-25.2 L + D3(zero) against UMA-m 1.1 (omc): crystal energies relative within each molecule, at up '
          'to 5 MACE-CSP minima per molecule in P2_1/c. Pass: RMS <= 4 kJ/mol and no structure more than 10 kJ/mol too '
          'low. Interval: exact 95%.')
    print(f'{"molecules":28s} {"evaluable":>9s} {"passed":>7s} {"95% interval (%)":>18s} {"median RMS (kJ/mol)":>20s} '
          f'{"worst RMS (kJ/mol)":>19s}')
    for g, rows in groups.items():
        ev = [v for v in rows if v['evaluable']]
        k = sum(v['passed'] for v in ev)
        lo, hi = clopper_pearson(k, len(ev)) if ev else (float('nan'), float('nan'))
        rms = [v['rms_kJ'] for v in ev]
        summary[g.strip()] = dict(n=len(rows), evaluable=len(ev), passed=k, ci95=[lo, hi],
                                  median_rms_kJ=float(np.median(rms)) if rms else None,
                                  worst_rms_kJ=float(max(rms)) if rms else None)
        print(f'{g:28s} {len(ev):9d} {k:7d} {f"{100 * lo:.0f}-{100 * hi:.0f}":>18s} '
              f'{(np.median(rms) if rms else float("nan")):20.2f} {(max(rms) if rms else float("nan")):19.2f}')
    print('\nper molecule (RMS, lowest error, UMA-m energy span in kJ/mol):')
    for n, v in res.items():
        print(f'{n} n={v["n"]} CH3={int(v["ch3"])} rot={v["n_rot"]} passed={v["passed"]} rms={v["rms_kJ"]} '
              f'low={v["lowest_kJ"]} span={v["uma_m_span_kJ"]} rms_bj={v["rms_bj_kJ"]} {v["smiles"]}')
    json.dump(dict(groups=summary, molecules=res, route_check_worst_kJ=s['route_check_worst_kJ']),
              open(os.path.join(out, 'within_molecule.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('step', choices=['select', 'search', 'search_one', 'score', 'fill_uma', 'report', 'all'])
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--name', default=None, help='search_one: the molecule')
    ap.add_argument('--attempt', type=int, default=0, help='search_one: the retry index (changes the seed)')
    ap.add_argument('--finished_only', action='store_true', help='search: start nothing; cut to the finished molecules')
    a = ap.parse_args()
    if a.step == 'search_one':
        search_one(a.out_dir, a.name, a.slots, a.attempt)
        return
    if a.step == 'fill_uma':
        fill_uma(a.out_dir)
        return
    if a.step in ('select', 'all'):
        select(a.out_dir)
    if a.step in ('search', 'all'):
        search(a.out_dir, a.slots, a.finished_only)
    if a.step in ('score', 'all'):
        score_all(a.out_dir)
    if a.step in ('report', 'all'):
        report(a.out_dir)


if __name__ == '__main__':
    main()
