"""The glued FD distance against COMPACK on Day's FIT+DMA landscapes (docs/design/fd_metric_calibration_plan.md).

Molecules: drawn at random (seeded) among Day's landscapes whose molecule has no proper point symmetry (no C_n,
n = 2..6, about a principal axis maps it onto itself within POINT_TOL), so the FD distance needs no point-group
quotient; planar molecules take the molecular-mirror twin, as compack_glued.py does. Groups: up to GROUPS_PER_MOL of
the molecule's landscape groups, one per crystal family (crystal_fd.charts.cell_family), rarer families first, each
with at least MIN_SET low-energy structures (<= 30 lowest within 10 kJ/mol). In Sohncke groups only structures whose
pose fit is on the +1 sheet are used (an h = -1 fit there is the mirror-image crystal of the reference molecule).

Pairs per (molecule, group) set: PERTURB pairs of the set's lowest structure against rigid perturbations of itself
(centroid by s A, orientation by s / r_g rad, free cell lengths by s / 10 relative, free angles by s / 10 rad; s over
AMPLITUDES) and RANDOM pairs of distinct structures of the set. Every crystal is rebuilt from its fitted pose and
written as a P1 unit-cell CIF (whole molecules, one label per atom), compared by COMPACK on the repaired route
(common.compack_pair_repaired: 20-molecule shells, CCDC defaults; a match is 20/20), and measured by the glued FD
distance (GluedFDMetric, margin 1 A; with the twin when the matched mirror RMSD is < 0.1 A).

Day's CIFs are read only. Output: a new folder under common.OUT_ROOT (summary.md, pairs.pt, cif/, compack caches).

    CUDA_VISIBLE_DEVICES=-1 python -m energy_sampling.eval.fd_analysis.day_calibration --out day_calibration_nov18
"""
import argparse
import csv
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch

from energy_sampling.eval.fd_analysis import common as C

F64 = torch.float64
ROOT = 'D:/crystal_datasets/day1000_fitdma/FIT-DMA_Landscapes'
E_WIN, CAP = 10.0, 30
AMPLITUDES = (0.005, 0.015, 0.04, 0.1, 0.25, 0.6)  # s, A (see the module docstring)
PERTURB, RANDOM = len(AMPLITUDES), 6
MIN_SET = 4
GROUPS_PER_MOL = 3
POINT_TOL = 0.15  # A: a rotation maps the molecule onto itself if every atom lands within this of a like atom
FAMILY_ORDER = ('cubic', 'hexagonal', 'tetragonal', 'orthorhombic', 'triclinic', 'monoclinic')  # rarer first
COV = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
       'P': 1.07, 'B': 0.84, 'Si': 1.11, 'Se': 1.20}
ZNUM = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'Se': 34,
        'I': 53}


# ---------------------------------------------------------------------------- Day CIFs (as fd attack A's oracle)
def read_blocks(path, wanted):
    out, cur, buf = {}, None, []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if line.startswith('data_'):
                if cur is not None and cur in wanted:
                    out[cur] = buf
                cur, buf = line[5:].strip(), []
                if len(out) == len(wanted):
                    break
            elif cur is not None and cur in wanted:
                buf.append(line)
    if cur is not None and cur in wanted:
        out[cur] = buf
    return out


def parse_block(lines):
    cell, atoms, mode = {}, [], None
    for ln in lines:
        t = ln.strip()
        if t.startswith('_cell_length_') or t.startswith('_cell_angle_'):
            k, v = t.split()[:2]
            cell[k] = float(v.split('(')[0])
        elif t.startswith('_atom_site_occupancy') or t.startswith('_atom_site_fract_z'):
            mode = 'atoms'
        elif t.startswith('loop_') or t.startswith('#END'):
            if mode == 'atoms' and atoms:
                mode = None
        elif mode == 'atoms' and t and not t.startswith('_'):
            p = t.split()
            if len(p) >= 5:
                atoms.append((p[0], p[1], float(p[2]), float(p[3]), float(p[4])))
    c = [cell['_cell_length_a'], cell['_cell_length_b'], cell['_cell_length_c'],
         math.radians(cell['_cell_angle_alpha']), math.radians(cell['_cell_angle_beta']),
         math.radians(cell['_cell_angle_gamma'])]
    return dict(cell=np.array(c), labels=[a[0] for a in atoms], elems=[a[1] for a in atoms],
                frac=np.array([a[2:] for a in atoms]))


def cell_matrix(c):
    """Fractional -> Cartesian, a along x, b in xy (MXtalTools' convention)."""
    a, b, cc, al, be, ga = c
    v = math.sqrt(1 - math.cos(al) ** 2 - math.cos(be) ** 2 - math.cos(ga) ** 2
                  + 2 * math.cos(al) * math.cos(be) * math.cos(ga))
    return np.array([[a, b * math.cos(ga), cc * math.cos(be)],
                     [0, b * math.sin(ga), cc * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)],
                     [0, 0, cc * v / math.sin(ga)]])


def unwrap(frac, elems, T):
    """The listed asymmetric unit joined by bonds (27 images): unwrapped fractional coordinates."""
    n = len(frac)
    out, seen, stack = frac.copy(), np.zeros(n, bool), [0]
    seen[0] = True
    imgs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)
    while stack:
        i = stack.pop()
        for j in range(n):
            if seen[j]:
                continue
            d = frac[j] - out[i]
            cand = (d - np.round(d))[None] + imgs
            dist = np.linalg.norm(cand @ T.T, axis=1)
            k = int(np.argmin(dist))
            if dist[k] < COV.get(elems[i], 0.8) + COV.get(elems[j], 0.8) + 0.45:
                out[j] = out[i] + cand[k]
                seen[j] = True
                stack.append(j)
    if not seen.all():
        raise RuntimeError(f'molecule not connected: {int((~seen).sum())} atoms unreached')
    return out


def pose_fit(Y, X, heavy):
    """Y ~ c + R diag(h, 1, 1) X: (c, R proper, h, rmsd)."""
    c = Y[heavy].mean(0)
    U, _, Vt = np.linalg.svd((Y - c).T @ X)
    P = U @ Vt
    h = 1 if np.linalg.det(P) > 0 else -1
    return c, P @ np.diag([h, 1, 1]), h, float(np.sqrt((((Y - c) - X @ P.T) ** 2).sum(1).mean()))


def rot(axis, ang):
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


# ---------------------------------------------------------------------------- molecule
def matched_dev(A, B, elems):
    """Largest distance from each atom of A to its like atom of B under the optimal matching per element."""
    from scipy.optimize import linear_sum_assignment
    worst, perm = 0.0, np.arange(len(A))
    for el in set(elems):
        ids = np.array([i for i, e in enumerate(elems) if e == el])
        D = np.linalg.norm(A[ids][:, None] - B[ids][None], axis=-1)
        r, c = linear_sum_assignment(D ** 2)
        worst = max(worst, float(D[r, c].max()))
        perm[ids[r]] = ids[c]
    return worst, perm


def proper_symmetric(X, elems):
    """True if a rotation by 2 pi / n (n = 2..6) about a principal axis of X maps it onto itself within POINT_TOL."""
    w = np.array([ZNUM.get(e, 6) for e in elems], float)
    Xc = X - (w[:, None] * X).sum(0) / w.sum()
    I = (w[:, None, None] * (np.eye(3)[None] * (Xc ** 2).sum(1)[:, None, None] - Xc[:, :, None] * Xc[:, None, :])).sum(0)
    _, V = np.linalg.eigh(I)
    for k in range(3):
        for n in range(2, 7):
            if matched_dev(Xc @ rot(V[:, k], 2 * math.pi / n).T, Xc, elems)[0] < POINT_TOL:
                return True
    return False


def molecule_info(X, elems):
    """The dict common.molecule_template and metric.twin_canonical read (as crystal_fd.metric.molecule_info)."""
    heavy = np.array([e != 'H' for e in elems])
    _, _, Vt = np.linalg.svd(X[heavy] - X[heavy].mean(0))
    n = Vt[-1]
    sigma = np.eye(3) - 2 * np.outer(n, n)
    dev, perm = matched_dev(X @ sigma.T, X, elems)
    SX = X @ sigma.T
    rms = float(np.sqrt(((SX - X[perm]) ** 2).sum(1).mean()))
    return dict(X=torch.tensor(X, dtype=F64), z=torch.tensor([ZNUM[e] for e in elems]),
                rg=float(np.sqrt((X ** 2).sum(1).mean())), sigma=torch.tensor(sigma, dtype=F64), perm=perm,
                mirror_rmsd_matched=rms, mirror_maxdev_matched=dev)


# ---------------------------------------------------------------------------- crystals
def sym_ops(sg):
    from mxtaltools.constants.space_group_info import SYM_OPS
    M = np.asarray(SYM_OPS[sg], float)
    return M[:, :3, :3], M[:, :3, 3]


def unit_cell(cell, t, R, h, X, sg):
    """[Zmol, n, 3] fractional coordinates of every molecule of the cell, each whole, its centroid in [0, 1)^3."""
    T = cell_matrix(cell)
    asu = np.linalg.solve(T, (T @ t + X @ (R @ np.diag([h, 1, 1])).T).T).T
    mols, keys = [], []
    for W, w in zip(*sym_ops(sg)):
        F = asu @ W.T + w
        F = F - np.floor(F.mean(0))
        c = F.mean(0)
        if not any(np.abs((c - k) - np.round(c - k)).max() < 1e-7 for k in keys):
            mols.append(F)
            keys.append(c)
    return np.stack(mols)


def write_cif(path, cell, mols, elems):
    rows = [f'{e} {e}{k * len(elems) + i + 1} {x:.10f} {y:.10f} {z:.10f}'
            for k, F in enumerate(mols) for i, (e, (x, y, z)) in enumerate(zip(elems, F))]
    a, b, c = cell[:3]
    al, be, ga = np.degrees(cell[3:])
    txt = '\n'.join([f'data_{os.path.basename(path)[:-4]}', f'_cell_length_a {a:.10f}', f'_cell_length_b {b:.10f}',
                     f'_cell_length_c {c:.10f}', f'_cell_angle_alpha {al:.10f}', f'_cell_angle_beta {be:.10f}',
                     f'_cell_angle_gamma {ga:.10f}', "_space_group_name_H-M_alt 'P 1'", '_space_group_IT_number 1',
                     'loop_', '_space_group_symop_operation_xyz', "'x, y, z'", 'loop_', '_atom_site_type_symbol',
                     '_atom_site_label', '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z'] + rows)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')


def perturb(sg, cell, t, R, s, rg, rng):
    """A rigid perturbation of size s (module docstring), keeping the group's metric."""
    from mxtaltools.crystal_fd.charts import cell_family
    fam = cell_family(sg)
    c = cell.copy()
    g = rng.normal(size=6)
    if fam == 'triclinic':
        c[:3] *= 1 + s / 10 * g[:3]
        c[3:] += s / 10 * g[3:]
    elif fam == 'monoclinic':
        c[:3] *= 1 + s / 10 * g[:3]
        c[4] += s / 10 * g[4]
    elif fam == 'orthorhombic':
        c[:3] *= 1 + s / 10 * g[:3]
    elif fam in ('tetragonal', 'hexagonal'):
        c[0] *= 1 + s / 10 * g[0]
        c[1] = c[0]
        c[2] *= 1 + s / 10 * g[2]
    else:
        c[:3] = c[0] * (1 + s / 10 * g[0])
    shift = rng.normal(size=3)
    t2 = t + np.linalg.solve(cell_matrix(c), s * shift / np.linalg.norm(shift))
    return c, t2, rot(rng.normal(size=3), s / rg) @ R


# ---------------------------------------------------------------------------- landscapes
def landscape(ref):
    """(rows within E_WIN of the minimum, at most CAP lowest, as dicts with id, sg, e) of one landscape folder."""
    d = os.path.join(ROOT, ref)
    csvs = [f for f in os.listdir(d) if f.endswith('.csv')]
    rows = {}
    for f in [f for f in csvs if not f.endswith('_all.csv')] or csvs:
        for r in csv.DictReader(open(os.path.join(d, f))):
            rows[r['id']] = r
    if not rows:
        return []
    emin = min(float(r['energy']) for r in rows.values())
    sel = sorted((r for r in rows.values() if float(r['energy']) - emin <= E_WIN), key=lambda r: float(r['energy']))
    return [dict(id=r['id'], sg=int(r['spacegroup']), e=float(r['energy']) - emin) for r in sel[:CAP]]


def fitted_sets(ref):
    """(molecule info, elems, {sg: [structures]}) of one landscape, or None if it is skipped (and why)."""
    from mxtaltools.crystal_fd.charts import cell_family
    from mxtaltools.crystal_fd.io_mxt import is_sohncke
    rows = landscape(ref)
    if not rows:
        return None, 'no csv'
    blocks = {}
    for f in os.listdir(os.path.join(ROOT, ref)):
        if f.endswith('.cif'):
            blocks.update(read_blocks(os.path.join(ROOT, ref, f), {r['id'] for r in rows}))
    rows = [r for r in rows if r['id'] in blocks]
    if not rows:
        return None, 'no blocks'
    b0 = parse_block(blocks[rows[0]['id']])
    T0 = cell_matrix(b0['cell'])
    try:
        Y = unwrap(b0['frac'], b0['elems'], T0) @ T0.T
    except RuntimeError as e:
        return None, str(e)
    elems, labels = b0['elems'], b0['labels']
    heavy = np.array([e != 'H' for e in elems])
    X = Y - Y[heavy].mean(0)
    if proper_symmetric(X, elems):
        return None, 'proper point symmetry'
    sets = defaultdict(list)
    for r in rows:
        b = parse_block(blocks[r['id']])
        if b['labels'] != labels:
            continue
        T = cell_matrix(b['cell'])
        try:
            Yb = unwrap(b['frac'], elems, T) @ T.T
        except RuntimeError:
            continue
        c, R, h, rmsd = pose_fit(Yb, X, heavy)
        if rmsd > 0.01 or (is_sohncke(r['sg']) and h != 1):
            continue
        sets[r['sg']].append(dict(id=r['id'], sg=r['sg'], e=r['e'], cell=b['cell'], t=np.linalg.solve(T, c), R=R, h=h,
                                  family=cell_family(r['sg'])))
    by_fam = defaultdict(list)
    for sg, s in sets.items():
        if len(s) >= MIN_SET:
            by_fam[s[0]['family']].append((len(s), sg))
    chosen = []
    for fam in FAMILY_ORDER:
        if by_fam[fam] and len(chosen) < GROUPS_PER_MOL:
            chosen.append(max(by_fam[fam])[1])
    if not chosen:
        return None, 'no group with enough structures'
    return (molecule_info(X, elems), elems, {sg: sets[sg] for sg in chosen}), 'ok'


# ---------------------------------------------------------------------------- distances
def fd_distances(sg, mi, crystals, pairs):
    """Glued FD distance (with the twin when mi's matched mirror RMSD < 0.1 A) for index pairs into crystals."""
    from mxtaltools.crystal_fd.description import CrystalDescription
    from mxtaltools.crystal_fd.metric import GluedFDMetric, canonical_desc, twin_canonical
    desc = CrystalDescription.from_parts(sg, torch.tensor(np.stack([c['cell'] for c in crystals])),
                                         torch.tensor(np.stack([c['t'] for c in crystals])),
                                         torch.tensor(np.stack([c['R'] for c in crystals])),
                                         torch.tensor([c['h'] for c in crystals]))
    can = canonical_desc(desc)
    gm = GluedFDMetric(sg, mi['rg'])
    P = gm.prepare_glued(can)
    ia = torch.tensor([a for a, _ in pairs])
    ib = torch.tensor([b for _, b in pairs])
    d = gm.distance(P, ia, ib)
    twin = mi['mirror_rmsd_matched'] < 0.1
    if twin:
        Pt = gm.prepare_glued(twin_canonical(sg, can, mi['sigma']))
        d = torch.minimum(d, gm.pairs(P[0], ia, Pt[1], ib))
    return d.numpy(), can['status'], twin


# ---------------------------------------------------------------------------- main
def main():
    C.setup()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_mol', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out = C.new_out_dir(a.out)
    cif_dir = os.path.join(out, 'cif')
    os.makedirs(cif_dir)
    C.log(f'# Day calibration of the glued FD distance against COMPACK ({vars(a)})')
    rng = np.random.default_rng(a.seed)
    refs = sorted(x for x in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, x)))
    rng.shuffle(refs)
    rows, skipped, n_used = [], defaultdict(int), 0
    for ref in refs:
        if n_used >= a.n_mol:
            break
        got, why = fitted_sets(ref)
        if got is None:
            skipped[why.split(':')[0]] += 1
            continue
        mi, elems, sets = got
        tmpl = C.molecule_template(mi)
        cache, cache_path = {}, os.path.join(out, f'compack_{ref}.pt')
        n_used += 1
        X = mi['X'].numpy()
        for sg, S in sets.items():
            base = S[0]
            crystals = list(S)
            pairs, kinds, amps = [], [], []
            for s in AMPLITUDES:
                c, t, R = perturb(sg, base['cell'], base['t'], base['R'], s, mi['rg'], rng)
                crystals.append(dict(id=f'{base["id"]}-p{s:g}', cell=c, t=t, R=R, h=base['h']))
                pairs.append((0, len(crystals) - 1))
                kinds.append('perturbed')
                amps.append(s)
            cand = [(i, j) for i in range(len(S)) for j in range(i + 1, len(S))]
            for k in rng.choice(len(cand), min(RANDOM, len(cand)), replace=False):
                pairs.append(cand[k])
                kinds.append('random')
                amps.append(float('nan'))
            for cr in crystals:
                write_cif(os.path.join(cif_dir, f'{cr["id"]}.cif'), cr['cell'],
                          unit_cell(cr['cell'], cr['t'], cr['R'], cr['h'], X, sg), elems)
            d, status, twin = fd_distances(sg, mi, crystals, pairs)
            res = C.compack([(crystals[i]['id'], crystals[j]['id']) for i, j in pairs], cif_dir, cache, cache_path,
                            tmpl)
            for (i, j), k, s, dk in zip(pairs, kinds, amps, d):
                r = res[C.pair_key(crystals[i]['id'], crystals[j]['id'])]
                rows.append(dict(ref=ref, sg=sg, family=base['family'], kind=k, amp=s, d=float(dk),
                                 status_a=status[i], status_b=status[j], twin=twin, compack=r['status'],
                                 rmsd=r['rmsd'], n_matched=r['n_matched']))
            C.log(f'{ref} sg {sg} ({base["family"]}): {len(S)} structures, {len(pairs)} pairs, twin {twin}')
    torch.save(dict(rows=rows, args=vars(a), skipped=dict(skipped)), os.path.join(out, 'pairs.pt'))
    report(rows, dict(skipped), n_used, out)


def report(rows, skipped, n_used, out):
    from scipy.stats import spearmanr
    from energy_sampling.eval.fd_analysis.compack_glued import auc, logistic
    ok = [r for r in rows if r['compack'] in ('ok', 'failed') and r['status_a'] in ('canonical', 'special_metric')
          and r['status_b'] in ('canonical', 'special_metric')]
    lines = [f'# The glued FD distance against COMPACK on Day landscapes', '',
             f'{n_used} molecules (landscapes skipped, by reason: {skipped}); {len(rows)} pairs, {len(ok)} scored '
             f'(COMPACK ok or failed, both crystals canonicalised); a match is COMPACK 20/20.', '',
             '*Per crystal family: pairs, COMPACK matches, matches among pairs with d < 0.05 A, the logistic d at '
             'P(match) 0.9 / 0.5 / 0.1, the scale glued FD / COMPACK RMSD on matched pairs (P10 / P50 / P90), '
             'Spearman of the two on matched pairs, and AUC (all pairs of the family). The reference: MIPCAS 1.17x and '
             'NEHZOR 1.30x scale, every pair below 0.05 A a match (2026-11-17).*', '',
             '| family | pairs (count) | matches (count) | d < 0.05 A: matched / all (count) | d at P 0.9 / 0.5 / 0.1 (A) '
             '| scale P10 / P50 / P90 (ratio) | Spearman (matched) | AUC |', '|---|---|---|---|---|---|---|---|']
    groups = defaultdict(list)
    for r in ok:
        groups[r['family']].append(r)
    groups['all'] = ok
    for fam in [f for f in FAMILY_ORDER[::-1] if f in groups] + ['all']:
        g = groups[fam]
        d = np.array([r['d'] for r in g])
        y = np.array([1 if (r['compack'] == 'ok' and r['n_matched'] == 20) else 0 for r in g])
        rm = np.array([r['rmsd'] for r in g])
        small = d < 0.05
        m = y == 1
        try:
            p50, p90, p10, _ = logistic(d, y)
            lg = f'{p90:.3f} / {p50:.3f} / {p10:.3f}'
        except Exception:  # noqa: BLE001 - one class only
            lg = 'n/a'
        sc = d[m & (rm > 0)] / rm[m & (rm > 0)]
        scs = ' / '.join(f'{v:.2f}' for v in np.quantile(sc, [0.1, 0.5, 0.9])) if len(sc) else 'n/a'
        sp = spearmanr(d[m], rm[m]).correlation if m.sum() > 2 else float('nan')
        lines.append(f'| {fam} | {len(g)} | {int(m.sum())} | {int(y[small].sum())} / {int(small.sum())} | {lg} | {scs} '
                     f'| {sp:.3f} | {auc(d, y):.3f} |')
    lines += ['', '*Glued FD distance bins (all families): pairs, matches, P(match), median COMPACK RMSD on matches.*', '',
              '| glued FD bin (A) | pairs (count) | matched (count) | P(match) | median COMPACK RMSD, matched (A) |',
              '|---|---|---|---|---|']
    d = np.array([r['d'] for r in ok])
    y = np.array([1 if (r['compack'] == 'ok' and r['n_matched'] == 20) else 0 for r in ok])
    rm = np.array([r['rmsd'] for r in ok])
    for lo, hi in zip([0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 3.0],
                      [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 3.0, np.inf]):
        s = (d >= lo) & (d < hi)
        n, k = int(s.sum()), int(y[s].sum())
        med = float(np.median(rm[s & (y == 1)])) if k else float('nan')
        lines.append(f'| [{lo}, {hi}) | {n} | {k} | {k / n if n else float("nan"):.2f} | {med:.3f} |')
    txt = '\n'.join(lines) + '\n'
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print(txt, flush=True)


if __name__ == '__main__':
    main()
