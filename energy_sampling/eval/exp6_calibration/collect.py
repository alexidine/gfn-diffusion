"""Fitting data for the exp-6 calibration (docs/design/exp6_calibration_plan.md, steps 1-3).

Steps, each resumable (a finished search or labelled molecule is reused):
    select  fit set (pool 'train') and comparison panel (pool 'heldout_scaffold'): common.selection
    exp6    per fit molecule, an exp-6 (FIT + MMFF) search with every evaluated state recorded
    mlip    per --mlip_every-th fit molecule, a MACE-CSP L + D3 search with every evaluated state recorded
    frames  thinned states: per search, steps 0, 1, 2, 4, 8, ... and the last of each stage; then at most
            --per_band states in each energy band above that search's lowest state (0-2-5-10-20-40-80 kJ/mol); plus the
            search's minima within 20 kJ/mol of its lowest (at most 30)
    label   MLIP energy and gradient (kJ/mol per molecule, per cell parameter) at every frame, FIT exp-6 the same,
            and the shortest intermolecular contact

Outputs in OUT: panel.json, conformers, the searches (make_configs run names; exp-6 runs carry trajectories), and
frames.pt: a dict of per-frame lists/tensors (name, sg, source, params [n, 12], hand, e_mlip, g_mlip, e_fit, g_fit,
contact). Energies are relative to nothing (a molecule's MLIP lattice energy carries its constant gas leg; the fit
compares within a molecule only).

usage: python -m eval.exp6_calibration.collect OUT {select,exp6,mlip,frames,label,all} [--n_fit 200] [--n_panel 24]
           [--exp6_starts 200] [--mlip_every 5] [--mlip_starts 25] [--per_band 10] [--slots 8] [--limit N]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from eval.exp6_calibration import common as C

MODULE = 'eval.exp6_calibration.collect'
EDGES = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
MIN_WINDOW_KJ, MIN_CAP = 20.0, 30


def run_name(name, energy):
    return f'{name}_{energy}_search'


def search_one(out, name, energy, n, slots):
    from eval.exp6_recall.mirror_test import cap_gpu, run
    cap_gpu()
    rec = json.load(open(os.path.join(out, 'panel.json')))[name]
    _, c = C.search_config(name, rec['sg'], energy, out, n, trajectories=True, slots=slots)
    run(c)


def searches(out, energy, names, n, slots):
    for k, name in enumerate(names):
        if os.path.exists(os.path.join(out, run_name(name, energy) + '_minima.pt')):
            continue
        t = time.time()
        code = C.run_in_subprocess(MODULE, out, ['search_one', '--name', name, '--energy', energy, '--n', n,
                                                 '--slots', slots])
        print(f'{energy} {name} ({k + 1}/{len(names)}): exit {code}, {time.time() - t:.0f} s', flush=True)


def thin(traj, per_band, rng):
    """Indices into a merged trajectory dict: log-spaced steps, then per energy band at most per_band."""
    step, stage, start = traj['step'].numpy(), traj['stage'].numpy(), traj['start_index'].numpy()
    keep = np.zeros(len(step), dtype=bool)
    last = {}
    for i in range(len(step)):
        k = (int(start[i]), int(stage[i]))
        last[k] = max(last.get(k, -1), int(step[i]))
    for i in range(len(step)):
        s = int(step[i])
        keep[i] = s == 0 or (s & (s - 1)) == 0 or s == last[(int(start[i]), int(stage[i]))]
    idx = np.flatnonzero(keep)
    e = traj['f_kT'].numpy()[idx] * float(traj['kT'])
    ok = np.isfinite(e)
    idx, e = idx[ok], e[ok]
    if not len(idx):
        return idx
    b = C.bands(e, EDGES)
    out = []
    for band in range(len(EDGES) - 1):
        members = idx[b == band]
        if len(members) > per_band:
            members = rng.choice(members, per_band, replace=False)
        out.extend(members.tolist())
    return np.array(sorted(out), dtype=int)


def frames(out, names, energies, per_band):
    """frames.pt without labels: thinned trajectory states and minima of every finished search."""
    from eval.exp6_recall.mlip_basins import E
    rng = np.random.default_rng(C.SEED)
    rec = {k: [] for k in ('name', 'sg', 'source', 'params', 'hand')}
    templates = {}
    panel = json.load(open(os.path.join(out, 'panel.json')))
    for name in names:
        for energy in energies:
            base = os.path.join(out, run_name(name, energy))
            if not os.path.exists(base + '_minima.pt'):
                continue
            mins = list(torch.load(base + '_minima.pt', weights_only=False))
            if name not in templates and mins:
                templates[name] = C.template(mins[0])
            if os.path.exists(base + '_traj.pt'):
                traj = torch.load(base + '_traj.pt', weights_only=False)
                for i in thin(traj, per_band, rng).tolist():
                    rec['name'].append(name)
                    rec['sg'].append(panel[name]['sg'])
                    rec['source'].append(f'{energy}_traj')
                    rec['params'].append(traj['params'][i].double())
                    rec['hand'].append(float(traj['handedness'][i].reshape(-1)[0]))
            fm = [E(r, energy) for r in mins]
            order = np.argsort(fm)
            for i in order[:MIN_CAP]:
                if fm[i] - fm[order[0]] > MIN_WINDOW_KJ:
                    break
                r = mins[int(i)]
                rec['name'].append(name)
                rec['sg'].append(panel[name]['sg'])
                rec['source'].append(f'{energy}_min')
                rec['params'].append(r.full_cell_parameters().reshape(-1).double())
                rec['hand'].append(float(r.aunit_handedness.reshape(-1)[0]))
    rec['params'] = torch.stack(rec['params']) if rec['params'] else torch.zeros(0, 12, dtype=torch.float64)
    rec['hand'] = torch.tensor(rec['hand'])
    torch.save(dict(frames=rec, templates=templates), os.path.join(out, 'frames_unlabelled.pt'))
    src = {s: rec['source'].count(s) for s in set(rec['source'])}
    print(f'{len(rec["name"])} frames from {len(templates)} molecules: {src}', flush=True)


def label(out, chunk=8):
    """MLIP and FIT exp-6 energy (kJ/mol per molecule) and gradient (per cell parameter) at every frame, and the
    shortest intermolecular contact; saved per molecule into label_parts/, merged into frames.pt."""
    from eval.exp6_recall.make_configs import SITES
    from eval.exp6_recall.mirror_test import cap_gpu
    from mxtaltools.crystal_search import minima as mn
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    cap_gpu()
    d = torch.load(os.path.join(out, 'frames_unlabelled.pt'), weights_only=False)
    fr, templates = d['frames'], d['templates']
    parts = os.path.join(out, 'label_parts')
    os.makedirs(parts, exist_ok=True)
    names = list(dict.fromkeys(fr['name']))
    pred = None
    for name in names:
        p = os.path.join(parts, f'{name}.pt')
        if os.path.exists(p):
            continue
        if pred is None:
            pred = load_mace_model(SITES['local']['mace_csp'], 'cuda', torch.float32,
                                   dispersion=dict(xc='pbe', damping='bj'))
        idx = [i for i, n in enumerate(fr['name']) if n == name]
        x = fr['params'][idx].float()
        rows = C.rows_at(templates[name], x, fr['hand'][idx])
        res = {}
        for tag, kw in (('mlip', dict(optim_target='mace', predictor=pred)),
                        ('fit', dict(optim_target='exp6', exp6_kwargs=C.EXP6_FIT))):
            se = mn.StageEnergy(rows, kT=1.0, device='cuda', chunk=chunk, **kw)
            f, g = [], []
            for s in range(0, len(rows), chunk):
                sl = list(range(s, min(s + chunk, len(rows))))
                try:
                    fi, gi = se.fg(x[sl], sl)
                except (RuntimeError, ValueError) as e:  # out of memory or a builder error: one row at a time
                    torch.cuda.empty_cache()
                    fi, gi = torch.full((len(sl),), float('nan'), dtype=torch.float64), \
                        torch.full((len(sl), x.shape[1]), float('nan'), dtype=torch.float64)
                    for j, q in enumerate(sl):
                        try:
                            a, b = se.fg(x[[q]], [q])
                            fi[j], gi[j] = a[0], b[0]
                        except (RuntimeError, ValueError) as e1:
                            torch.cuda.empty_cache()
                            print(f'{name} frame {idx[q]} {tag}: {type(e1).__name__}: {str(e1)[:100]}', flush=True)
                f.append(fi)
                g.append(gi)
            res[f'e_{tag}'], res[f'g_{tag}'] = torch.cat(f), torch.cat(g)
        res['contact'] = torch.tensor([C.shortest_contact(r) for r in rows])
        res['index'] = torch.tensor(idx)
        torch.save(res, p)
        print(f'labelled {name}: {len(idx)} frames, {int(torch.isfinite(res["e_mlip"]).sum())} MLIP-finite, '
              f'{int((res["contact"] < C.MIN_CONTACT_A).sum())} below {C.MIN_CONTACT_A} A', flush=True)
    n = len(fr['name'])
    for k, shape in (('e_mlip', ()), ('g_mlip', (12,)), ('e_fit', ()), ('g_fit', (12,)), ('contact', ())):
        fr[k] = torch.full((n,) + shape, float('nan'), dtype=torch.float64)
    for name in names:
        res = torch.load(os.path.join(parts, f'{name}.pt'), weights_only=False)
        for k in ('e_mlip', 'g_mlip', 'e_fit', 'g_fit', 'contact'):
            fr[k][res['index']] = res[k].double()
    torch.save(dict(frames=fr, templates=templates), os.path.join(out, 'frames.pt'))
    print(f'frames.pt: {n} frames, {int(torch.isfinite(fr["e_mlip"]).sum())} with MLIP labels', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('step', choices=['select', 'exp6', 'mlip', 'frames', 'label', 'all', 'search_one'])
    ap.add_argument('--n_fit', type=int, default=200)
    ap.add_argument('--n_panel', type=int, default=24)
    ap.add_argument('--exp6_starts', type=int, default=200)
    ap.add_argument('--mlip_every', type=int, default=5)
    ap.add_argument('--mlip_starts', type=int, default=25)
    ap.add_argument('--per_band', type=int, default=10)
    ap.add_argument('--slots', type=int, default=8, help='MLIP search slots (exp-6 uses make_configs default)')
    ap.add_argument('--limit', type=int, default=None, help='first N fit molecules only (smoke runs)')
    ap.add_argument('--name')
    ap.add_argument('--energy')
    ap.add_argument('--n', type=int)
    a = ap.parse_args()
    if a.step == 'search_one':
        search_one(a.out, a.name, a.energy, a.n, a.slots if a.energy != 'exp6' else None)
        return
    panel = C.selection(a.out, a.n_fit, a.n_panel)
    fit = [k for k, v in panel.items() if v['role'] == 'fit'][:a.limit]
    mlip = fit[::a.mlip_every]
    if a.step in ('exp6', 'all'):
        searches(a.out, 'exp6', fit, a.exp6_starts, 0)
    if a.step in ('mlip', 'all'):
        searches(a.out, C.MLIP, mlip, a.mlip_starts, a.slots)
    if a.step in ('frames', 'all'):
        frames(a.out, fit, ['exp6', C.MLIP], a.per_band)
    if a.step in ('label', 'all'):
        label(a.out)


if __name__ == '__main__':
    main()
