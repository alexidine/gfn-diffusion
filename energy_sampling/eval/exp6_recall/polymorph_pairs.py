"""Do MLIPs agree on polymorph energy differences? Single points at deposited CSD structures, no relaxation.

For molecules in the GFN training domain (H C N O F only, neutral, one component, MW < 230, at most one RDKit
rotatable bond) with Z' = 1 crystals in two or more space groups (csd_fd_v5), one entry per (molecule, space group),
built from the CSD as the recall panel is (hydrogens normalised, ``eval.day_bench.p1_ingest.ingest``). Per crystal,
the lattice energy per molecule (crystal energy / Z minus the isolated molecule in the same deposited conformer)
under:
  mace_csp  MACE-CSP-25.2 large + D3(BJ, PBE)
  mace      MACE-OFF24 medium
  uma       UMA-s 1.1 (gas leg: mean of the molecule and its mirror image), averaged over the crystal and its mirror
            image in Sohncke groups (raw elsewhere)
The crystal energy alone is not usable at deposited geometries: X-ray conformers (hydrogens especially) carry
intramolecular errors of hundreds of kJ/mol per molecule that differ between entries (measured 2026-11-19, dimethyl
ether ABEWAG vs ABEWAG03: 135 kJ/mol in the isolated-molecule energy); the lattice energy removes them, at the cost of
the isolated-molecule leg, whose model error is roughly common to two forms of one molecule.
Per pair (two space groups of one molecule): dE = E(form B) - E(form A), kJ/mol per molecule. Molecules are labelled
inside or outside MACE-CSP-25.2's training selection (no CH3, OH or NH2, or exactly one OH, NH2 or COOH at MW <= 200
with no CH3).

usage: python -m eval.exp6_recall.polymorph_pairs OUT_DIR select [--max_molecules N]
       python -m eval.exp6_recall.polymorph_pairs OUT_DIR build
       python -m eval.exp6_recall.polymorph_pairs OUT_DIR score
       python -m eval.exp6_recall.polymorph_pairs OUT_DIR relax [--models mace_csp,mace,uma]
"""
import argparse
import io
import itertools
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

FD = 'D:/crystal_datasets/csd_fd_v5/data.pt'
ELEMENTS = {1, 6, 7, 8, 9}
KJ = 96.485
BROKEN = -10.0  # kJ/mol per molecule


def classify(m):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    pat = {k: Chem.MolFromSmarts(v) for k, v in dict(ch3='[CH3]', oh='[OX2H]', nh2='[NX3H2]', cooh='C(=O)[OX2H]').items()}
    c = {k: len(m.GetSubstructMatches(p)) for k, p in pat.items()}
    n_rot, mw = rdMolDescriptors.CalcNumRotatableBonds(m), Descriptors.MolWt(m)
    rigid = n_rot == 0 and c['ch3'] == 0 and c['oh'] == 0 and c['nh2'] == 0
    one = (c['ch3'] == 0 and mw <= 200 and ((n_rot == 0 and c['cooh'] == 0 and c['oh'] + c['nh2'] == 1)
                                          or (n_rot <= 1 and c['cooh'] == 1 and c['oh'] == 1 and c['nh2'] == 0)))
    return dict(n_rot=n_rot, mw=mw, ch3=c['ch3'], inside=bool(rigid or one))


def select(out, max_molecules):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDLogger.DisableLog('rdApp.*')
    d = torch.load(FD, weights_only=False, map_location='cpu')
    smiles, ident, sg, ok = d['columns']['smiles'], d['identifier'], d['sg'].tolist(), (d['status'] == 0).tolist()
    groups = {}
    for i in range(len(ident)):
        if ok[i]:
            groups.setdefault(smiles[i], {}).setdefault(sg[i], []).append(ident[i])
    groups = {s: g for s, g in groups.items() if s and len(g) >= 2}
    keep = {}
    for s, g in groups.items():
        m = Chem.MolFromSmiles(s)
        if m is None or '.' in s or any(a.GetFormalCharge() for a in m.GetAtoms()):
            continue
        mh = Chem.AddHs(m)
        if not {a.GetAtomicNum() for a in mh.GetAtoms()} <= ELEMENTS or 6 not in {a.GetAtomicNum() for a in m.GetAtoms()}:
            continue
        if Descriptors.MolWt(m) >= 230 or rdMolDescriptors.CalcNumRotatableBonds(m) > 1:
            continue
        keep[s] = dict(groups=g, **classify(m))
    names = sorted(keep)
    if max_molecules:
        rng = np.random.default_rng(0)
        names = sorted(rng.choice(names, min(max_molecules, len(names)), replace=False).tolist())
    keep = {s: keep[s] for s in names}
    json.dump(keep, open(os.path.join(out, 'selection.json'), 'w'), indent=1)
    n_in = sum(v['inside'] for v in keep.values())
    print(f'{len(groups)} molecules with Z\'=1 crystals in >= 2 groups; {len(keep)} in the domain '
          f'({n_in} inside the MACE-CSP selection, {len(keep) - n_in} outside)')


def best_entry(reader, refcodes):
    """Lowest-R ordered 3D entry among refcodes, or None."""
    best = None
    for rc in refcodes:
        try:
            e = reader.entry(rc)
            if not e.has_3d_structure or e.has_disorder or e.crystal.z_prime != 1:
                continue
            r = e.r_factor if e.r_factor is not None else 99.0
            if r > 10:
                continue
            if best is None or r < best[1]:
                best = (rc, r)
        except Exception:
            continue
    return best


def build(out):
    from ase.io import read
    from ccdc import io as ccdc_io
    from eval.day_bench.p1_ingest import ingest, make_reference, split_molecules
    from eval.exp6_recall.panel import normalise_h
    sel = json.load(open(os.path.join(out, 'selection.json')))
    reader = ccdc_io.EntryReader('CSD')
    rows, meta = [], []
    for k, (s, v) in enumerate(sel.items()):
        got = []
        for g, refs in sorted(v['groups'].items(), key=lambda kv: int(kv[0])):
            b = best_entry(reader, refs)
            if b is None:
                continue
            try:
                e = reader.entry(b[0])
                atoms = normalise_h(read(io.StringIO(e.crystal.to_string('cif')), format='cif'))
                ref = make_reference(split_molecules(atoms)[0], smiles=s)
                r = ingest(atoms, ref, identifier=b[0], prefer_sg=int(g))
            except Exception as ex:
                print(f'{b[0]}: {type(ex).__name__} {str(ex)[:100]}', flush=True)
                continue
            if r.ok and r.sg == int(g):
                got.append((b[0], int(g), b[1], r.crystal))
        if len(got) >= 2:
            for rc, g, rf, cr in got:
                rows.append(cr)
                meta.append(dict(smiles=s, refcode=rc, sg=g, r_factor=rf, inside=v['inside'], ch3=v['ch3'],
                                 n_rot=v['n_rot'], mw=v['mw']))
        if k % 25 == 0:
            print(f'{k}/{len(sel)} molecules, {len(rows)} crystals', flush=True)
    torch.save(rows, os.path.join(out, 'crystals.pt'))
    json.dump(meta, open(os.path.join(out, 'crystals.json'), 'w'), indent=1)
    print(f'{len(rows)} crystals of {len(set(m["smiles"] for m in meta))} molecules', flush=True)


def score(out):
    """Lattice energies (kJ/mol per molecule) of every crystal under the three models -> scores.json."""
    from eval.exp6_recall.csd_parity import energies, strip
    from eval.exp6_recall.make_configs import SITES, SOHNCKE
    from eval.exp6_recall.mirror_test import cap_gpu
    from mxtaltools.crystal_search.run_search import attach_mlip_parity
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
    cap_gpu()
    rows = [strip(r) for r in torch.load(os.path.join(out, 'crystals.pt'), weights_only=False)]
    meta = json.load(open(os.path.join(out, 'crystals.json')))
    res = {}
    p = load_mace_model(SITES['local']['mace_csp'], 'cuda', torch.float32, dispersion=dict(xc='pbe', damping='bj'))
    res['mace_csp'] = energies([strip(r) for r in rows], 'mace', p, 4).tolist()
    del p
    torch.cuda.empty_cache()
    p = load_mace_model(SITES['local']['mace'], 'cuda', torch.float32)
    res['mace'] = energies([strip(r) for r in rows], 'mace', p, 4).tolist()
    del p
    torch.cuda.empty_cache()
    # UMA-s as csd_parity: mirror-mean gas leg on every row; the periodic energy mirror-averaged in Sohncke groups
    raw = [strip(r) for r in rows]
    cfg = SimpleNamespace(mlip_gas_reference='mirror_mean', mlip_mirror_average=False, opt=[{'optim_target': 'uma'}],
                          uma_predictor_path=SITES['local']['uma'])
    attach_mlip_parity(cfg, raw, 'cuda')
    p = init_uma_crystal_predictor(SITES['local']['uma'], device='cuda', activation_checkpointing=False, tf32=False)
    E = energies(raw, 'uma', p, 2)
    res['uma_raw'] = E.tolist()
    idx = [i for i, m in enumerate(meta) if m['sg'] in SOHNCKE]
    avg = []
    for i in idx:
        q = strip(raw[i])
        q.uma_gas_pot = raw[i].uma_gas_pot
        q.uma_mirror_average = torch.tensor([True])
        avg.append(q)
    if idx:
        E[idx] = energies(avg, 'uma', p, 1)
    res['uma'] = E.tolist()
    json.dump(dict(meta=meta, energies=res), open(os.path.join(out, 'scores.json'), 'w'), indent=1)
    report(out)


def relax(out, models):
    """One local relaxation per crystal per model from the deposited structure (rigid molecule, the search's BFGS
    polish, tol 2e-3 kT): lattice energy at the start and at the end, status, -> relaxed_<model>.json."""
    from eval.exp6_recall.csd_parity import strip
    from eval.exp6_recall.make_configs import KT, SITES, SOHNCKE
    from eval.exp6_recall.mirror_test import cap_gpu
    from mxtaltools.crystal_search.minima import StageEnergy, bfgs_polish
    cap_gpu()
    base = [strip(r) for r in torch.load(os.path.join(out, 'crystals.pt'), weights_only=False)]
    meta = json.load(open(os.path.join(out, 'crystals.json')))
    for model in models:
        path = os.path.join(out, f'relaxed_{model}.json')
        if os.path.exists(path):
            continue
        rows = [strip(r) for r in base]
        if model == 'uma':
            from mxtaltools.crystal_search.run_search import attach_mlip_parity
            from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
            cfg = SimpleNamespace(mlip_gas_reference='mirror_mean', mlip_mirror_average=False,
                                  opt=[{'optim_target': 'uma'}], uma_predictor_path=SITES['local']['uma'])
            attach_mlip_parity(cfg, rows, 'cuda')
            for r, m in zip(rows, meta):
                if m['sg'] in SOHNCKE:
                    r.uma_mirror_average = torch.tensor([True])
            pred, target, chunk = init_uma_crystal_predictor(SITES['local']['uma'], device='cuda',
                                                             activation_checkpointing=False, tf32=False), 'uma', 2
        else:
            from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
            disp = dict(xc='pbe', damping='bj') if model == 'mace_csp' else None
            pred, target, chunk = load_mace_model(SITES['local'][model], 'cuda', torch.float32, dispersion=disp), 'mace', 2
        kT = KT[model]
        E = StageEnergy(rows, target, kT, predictor=pred, device='cuda', chunk=chunk)
        x0 = torch.stack([r.full_cell_parameters().reshape(-1) for r in rows]).double()
        with torch.no_grad():
            f0 = E.f(x0, list(range(len(rows)))).double().cpu()
        res = bfgs_polish(E, x0, max_iter=40, tol=2e-3)
        json.dump(dict(start=(f0 * kT).tolist(), end=(res['f'] * kT).tolist(), status=res['status'].tolist(),
                       x_end=res['x'].tolist()), open(path, 'w'))
        print(f'{model}: relaxed {len(rows)} crystals; status counts',
              {int(k): int(v) for k, v in zip(*torch.unique(res['status'], return_counts=True))}, flush=True)
        del pred, E
        torch.cuda.empty_cache()


def report(out):
    s = json.load(open(os.path.join(out, 'scores.json')))
    meta, E = s['meta'], {k: np.array(v) for k, v in s['energies'].items()}
    by = {}
    for i, m in enumerate(meta):
        by.setdefault(m['smiles'], []).append(i)
    # a crystal with a lattice energy above BROKEN under MACE-OFF24 or UMA-s is not a usable structure
    broken = {i for i in range(len(meta)) if not (E['mace'][i] < BROKEN and E['uma'][i] < BROKEN)}
    pairs = [(a, b) for idx in by.values() for a, b in itertools.combinations(idx, 2)
             if a not in broken and b not in broken]
    print(f'{len(broken)} of {len(meta)} crystals dropped as broken', flush=True)
    dE = {k: np.array([v[b] - v[a] for a, b in pairs]) for k, v in E.items()}
    inside = np.array([meta[a]['inside'] for a, _ in pairs])
    rows = []
    for x, y in (('mace_csp', 'mace'), ('mace_csp', 'uma'), ('mace', 'uma')):
        for lab, sel in (('inside', inside), ('outside', ~inside), ('all', np.ones_like(inside))):
            ok = sel & np.isfinite(dE[x]) & np.isfinite(dE[y])
            a, b = dE[x][ok], dE[y][ok]
            big = ok & (np.abs(dE[x]) > 2) & (np.abs(dE[y]) > 2)
            rows.append(dict(models=f'{x} vs {y}', molecules=lab, pairs=int(ok.sum()),
                             sd_diff=float(np.std(a - b)) if ok.any() else None,
                             mad=float(np.median(np.abs(a - b))) if ok.any() else None,
                             pearson=float(np.corrcoef(a, b)[0, 1]) if ok.sum() > 2 else None,
                             sign_agree_gt2=float(np.mean(np.sign(dE[x][big]) == np.sign(dE[y][big]))) if big.any() else None,
                             n_gt2=int(big.sum())))
    json.dump(rows, open(os.path.join(out, 'report.json'), 'w'), indent=1)
    for r in rows:
        print('PAIRS', json.dumps(r), flush=True)


def report_relaxed(out, models=('mace_csp', 'mace'), r_max=5.0):
    """Pair agreement at each model's own nearby minimum (relax step). Kept: both forms converged (status 1) under
    every model, R <= r_max %, a bound start (lattice energy < BROKEN under MACE-OFF24), and no molecule that can be a
    zwitterion (a basic amine and an acid)."""
    from rdkit import Chem
    zw = Chem.MolFromSmarts('[NX3;H1,H2,H0;!$(N-C=O);!$(N-a);!$(N=*)]'), Chem.MolFromSmarts('C(=O)[OX2H]')
    meta = json.load(open(os.path.join(out, 'crystals.json')))
    R = {m: json.load(open(os.path.join(out, f'relaxed_{m}.json'))) for m in models}
    E = {m: np.array(R[m]['end']) for m in models}
    ok = np.ones(len(meta), bool)
    for m in models:
        ok &= np.array(R[m]['status']) == 1
    ok &= np.array([(x['r_factor'] or 99) <= r_max for x in meta])
    ok &= np.array(R['mace']['start'] if 'mace' in R else R[models[0]]['start']) < BROKEN
    mol = {x['smiles']: Chem.MolFromSmiles(x['smiles']) for x in meta}
    ok &= np.array([not (mol[x['smiles']].HasSubstructMatch(zw[0]) and mol[x['smiles']].HasSubstructMatch(zw[1]))
                    for x in meta])
    by = {}
    for i, x in enumerate(meta):
        if ok[i]:
            by.setdefault(x['smiles'], []).append(i)
    pairs = [(a, b) for idx in by.values() for a, b in itertools.combinations(idx, 2)]
    inside = np.array([meta[a]['inside'] for a, _ in pairs], bool)
    dE = {m: np.array([E[m][b] - E[m][a] for a, b in pairs]) for m in models}
    print(f'{int(ok.sum())} of {len(meta)} crystals kept; {len(pairs)} pairs of {len(by)} molecules '
          f'({int(inside.sum())} pairs inside the MACE-CSP selection)')
    print('median |dE| per model, kJ/mol:', {m: round(float(np.median(np.abs(v))), 2) for m, v in dE.items()})
    rows = []
    for x, y in itertools.combinations(models, 2):
        for lab, sel in (('inside', inside), ('outside', ~inside), ('all', np.ones_like(inside))):
            d = np.abs(dE[x][sel] - dE[y][sel])
            big = (np.abs(dE[x][sel]) > 2) & (np.abs(dE[y][sel]) > 2)
            rows.append(dict(models=f'{x} vs {y}', molecules=lab, pairs=int(sel.sum()),
                             median_abs_diff=float(np.median(d)), within_2=float(np.mean(d < 2)),
                             within_5=float(np.mean(d < 5)),
                             sign_agree_gt2=float(np.mean(np.sign(dE[x][sel][big]) == np.sign(dE[y][sel][big])))
                             if big.any() else None, n_gt2=int(big.sum())))
    json.dump(dict(rows=rows, pairs=[(meta[a]['refcode'], meta[b]['refcode']) for a, b in pairs],
                   dE={m: v.tolist() for m, v in dE.items()}, inside=inside.tolist()),
              open(os.path.join(out, 'report_relaxed.json'), 'w'), indent=1)
    for r in rows:
        print('RELAXED', json.dumps(r), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('step', choices=['select', 'build', 'score', 'relax', 'report', 'report_relaxed'])
    ap.add_argument('--models', default='mace_csp,mace')
    ap.add_argument('--max_molecules', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dict(select=lambda: select(a.out_dir, a.max_molecules), build=lambda: build(a.out_dir),
         score=lambda: score(a.out_dir), relax=lambda: relax(a.out_dir, a.models.split(',')),
         report=lambda: report(a.out_dir),
         report_relaxed=lambda: report_relaxed(a.out_dir, tuple(a.models.split(','))))[a.step]()


if __name__ == '__main__':
    main()
