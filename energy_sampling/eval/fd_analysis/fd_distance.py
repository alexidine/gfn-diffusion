"""FD identity distance between two Z' = 1 crystals of one molecule in one space group, in Angstrom.

The distance, its glued variant and the canonical-form helpers live in ``mxtaltools.crystal_fd.metric`` (moved there
2026-10-30 for the minima library) and are imported here under their old names; this module keeps the campaign
analyses (``face_census``, ``main``). The definition below is the metric's.

Both crystals are first put in canonical form by crystal_fd.canon.canonicalize (reduced cell; the +1 sheet through an
improper element of G; pose folded into ASU(Gamma+) x SO(3)). Then

    d(A, B) = min over g = (W, w) in Gamma+ and n in {-1, 0, 1}^3 of  sqrt( D_cell^2 + D_cent(g, n)^2 + D_rot(g)^2 )

    D_cell^2   = (a_A - a_B)^2 + (b_A - b_B)^2 + (c_A - c_B)^2
                 + (dalpha sqrt(b c))^2 + (dbeta sqrt(a c))^2 + (dgamma sqrt(a b))^2      [A^2; angles in rad,
                 lengths in the scale factors the means of A's and B's]
    D_cent     = | Tbar (u - n) |,  u = t_A - (W t_B + w) wrapped to [-1/2, 1/2)^3,  Tbar = (T_A + T_B) / 2   [A]
    D_rot      = r_g * theta(R_A, Q_W R_B),  theta the geodesic angle (rad) of R_A^T Q_W R_B,
                 Q_W = T_B W T_B^-1 projected on O(3),  r_g the molecule's radius of gyration           [A]

g runs over GroupTable.gamma_plus (crystal_fd.tables.group_table(sg)); r_g is the RMS distance of all atoms from the
heavy-atom centroid (the builder's centre). The cell term compares the two canonical cells directly: the faces of the
triclinic / monoclinic cell domains are NOT glued here (the chart does not record those gluings), so two nearly equal
crystals whose reduced cells lie on opposite sides of a cell-domain face are far apart in d. `face` below measures how
many campaign endpoints sit that close to a face.

Molecular point symmetry is ignored in d (owner decision). For a molecule with an approximate improper self-symmetry
sigma (a near-planar molecule: the mirror through its plane, which maps the atoms onto themselves up to a permutation
of like atoms that preserves the bond graph), d_twin(A, B) = min(d(A, B), d(A, B~)) with B~ the canonical form of
(t_B, P_B sigma) is offered as a flagged variant.

    python -m energy_sampling.eval.fd_analysis.fd_distance --campaign mipcas --out fd_distance_mipcas
"""
import argparse
import os
import time

import numpy as np
import torch

from energy_sampling.eval.fd_analysis import common as C

from mxtaltools.crystal_fd.metric import (F64, FDMetric, GluedFDMetric, _setting_candidates,  # noqa: F401
                                          canonical, canonical_desc, merge_twin, molecule_info, twin_canonical)


def face_census(sg, can, metric, deltas, rng):
    """Axis-aligned cell perturbations of size delta (in D_cell's units) of each canonical row, t and R kept;
    re-canonicalised; a row is near a face at delta if some perturbation's canonical form is further than
    1.5 delta + 1e-4 A from the row (no face crossed gives d <= delta exactly). Returns per delta the counts."""
    from mxtaltools.crystal_fd.description import CrystalDescription
    n = len(can['cell'])
    axes = [0, 1, 2, 3, 4, 5] if sg <= 2 else [0, 1, 2, 4]
    L = can['cell'][:, :3]
    sc = torch.stack([torch.sqrt(L[:, 1] * L[:, 2]), torch.sqrt(L[:, 0] * L[:, 2]), torch.sqrt(L[:, 0] * L[:, 1])], 1)
    A = metric.prepare(can)
    res = {}
    for delta in deltas:
        split_any = torch.zeros(n, dtype=torch.bool)
        split_axis = torch.zeros(len(axes) * 2, dtype=torch.long)
        basis_changed_any = torch.zeros(n, dtype=torch.bool)
        dmax = torch.zeros(n, dtype=F64)
        k = 0
        for ax in axes:
            for sgn in (1.0, -1.0):
                cell = can['cell'].clone()
                if ax < 3:
                    cell[:, ax] += sgn * delta
                else:
                    cell[:, ax] += sgn * delta / sc[:, ax - 3]
                desc = CrystalDescription.from_parts(sg, cell, can['t'], can['R'], torch.ones(n, dtype=torch.long))
                pc = canonical_desc(desc)
                ok = torch.tensor([s in ('canonical', 'special_metric') for s in pc['status']])
                B = metric.prepare(pc)
                d = metric.pairs(A, torch.arange(n), B, torch.arange(n))
                d = torch.where(ok, d, torch.full_like(d, float('nan')))
                sp = ok & (d > 1.5 * delta + 1e-4)
                split_any |= sp
                split_axis[k] = int(sp.sum())
                basis_changed_any |= ok & (pc['N'] != torch.eye(3, dtype=torch.long)).flatten(1).any(1)
                dmax = torch.maximum(dmax, torch.nan_to_num(d, nan=0.0))
                k += 1
        res[delta] = dict(split=int(split_any.sum()), basis_changed=int(basis_changed_any.sum()),
                          per_axis=split_axis.tolist(), split_mask=split_any, dmax_median_split=float(
                dmax[split_any].median()) if split_any.any() else float('nan'))
    return res, axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True, choices=list(C.CAMPAIGNS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--band_kT', type=float, default=2.0)
    ap.add_argument('--n_face', type=int, default=5000)
    ap.add_argument('--n_check', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    C.setup()
    out = C.new_out_dir(a.out)
    rng = np.random.default_rng(a.seed)
    co, cfg, reg, ref = C.load_campaign(a.campaign)
    sg = cfg.sg
    mi = molecule_info(cfg.mol_path)
    lines = [f'# FD identity distance: {a.campaign} (sg {sg}, Z\'=1)', '', __doc__.split('\n\n    python')[0], '',
             f'Molecule ({cfg.mol_path}): {mi["n_atoms"]} atoms; r_g {mi["rg"]:.4f} A (all atoms about the heavy-atom '
             f'centroid, builder frame); heavy-atom plane normal {[round(float(v), 4) for v in mi["normal"]]}; '
             f'mirror through that plane: RMSD {mi["mirror_rmsd"]:.4f} A in identity atom order, '
             f'{mi["mirror_rmsd_matched"]:.4f} A (largest atom deviation {mi["mirror_maxdev_matched"]:.4f} A) with like '
             f'atoms matched by optimal assignment ({mi["perm_moves"]} atoms swapped); heavy-atom out-of-plane max '
             f'{mi["out_of_plane_max"]:.4f} A.', '']
    from mxtaltools.crystal_fd.tables import group_table
    tab = group_table(sg)
    lines += [f'Gamma+ of sg {sg}: {len(tab.gamma_plus)} elements (W distinct: '
              f'{len(torch.unique(tab.gamma_plus_W.reshape(len(tab.gamma_plus), -1), dim=0))}); lattice shifts '
              f'{{-1,0,1}}^3 after wrapping, so {27 * len(tab.gamma_plus)} images per pair.', '']
    st = C.shard_states(a.campaign, co, cfg, reg, ref, a.band_kT)
    del reg
    n_all = len(st['energy'])
    idx = np.sort(rng.choice(n_all, min(a.n_face, n_all), replace=False))
    s = C.subset(st, idx)
    C.log(f'{n_all} shard states within {a.band_kT} kT; {len(idx)} sampled')

    # canonicalisation, timed
    t0 = time.time()
    can = canonical(sg, s['params'], s['hand'])
    t_can = (time.time() - t0) / len(idx)
    stat = {k: can['status'].count(k) for k in set(can['status'])}
    nf = can['near_face']
    lines += [f'## Canonical forms of {len(idx)} endpoints (random, seed {a.seed}, of {n_all} shard states within '
              f'{a.band_kT} kT)', '',
              f'Status: {stat}. near_face at the default tolerance (1e-5, chart units): NEAR_CELL {int(((nf & 1) > 0).sum())}, '
              f'NEAR_CENTROID {int(((nf & 2) > 0).sum())}, NEAR_FOLD_BAND {int(((nf & 4) > 0).sum())}; special metrics '
              f'(n_auto > 1) {int((can["n_auto"] > 1).sum())}. canonicalize: {1e3 * t_can:.2f} ms per row.', '']
    metric = FDMetric(sg, mi['rg'])
    A = metric.prepare(can)

    # check 1: re-descriptions of the same crystal (a Gamma+ element, a lattice shift, sg 1-15 a basis change of the
    # setting group) come back at distance ~0
    from mxtaltools.crystal_fd import ops
    from mxtaltools.crystal_fd.description import CrystalDescription
    nc = min(a.n_check, len(idx))
    sel = torch.as_tensor(rng.choice(len(idx), nc, replace=False))
    base = CrystalDescription.from_parts(sg, can['cell'][sel], can['t'][sel], can['R'][sel], can['h'][sel])
    k = torch.as_tensor(rng.integers(0, metric.m, nc))
    img = ops.apply(base, metric.W[k], metric.w[k] + torch.as_tensor(rng.integers(-2, 3, (nc, 3)), dtype=F64))
    if sg <= 2:
        Ns = []
        for _ in range(nc):
            while True:
                N = rng.integers(-1, 2, (3, 3))
                if round(np.linalg.det(N)) == 1:
                    break
            Ns.append(N)
        img = ops.change_basis(img, torch.as_tensor(np.stack(Ns)), torch.zeros(nc, 3, dtype=F64))
    else:
        from mxtaltools.crystal_search.standardize import keeps_operators
        Ns = []
        for _ in range(nc):
            while True:
                p_, q_, r_, s_ = rng.integers(-1, 2, 4)
                N = np.array([[p_, 0, r_], [0, p_ * s_ - q_ * r_, 0], [q_, 0, s_]])
                if round(np.linalg.det(N)) == 1 and keeps_operators(N, sg):
                    break
            Ns.append(N)
        img = ops.change_basis(img, torch.as_tensor(np.stack(Ns)))
    ci = canonical_desc(img)
    d_self = metric.pairs(A, sel, metric.prepare(ci), torch.arange(nc))
    lines += [f'Check (same crystal re-described: random Gamma+ element, lattice shift in [-2,2]^3, random basis change '
              f'of the setting group; {nc} rows): d max {float(d_self.max()):.2e} A, median {float(d_self.median()):.2e} A; '
              f'statuses {dict((s_, ci["status"].count(s_)) for s_ in set(ci["status"]))}.', '']

    # check 2: d against the literal largest atom deviation for small moves of t and R (same cell)
    from mxtaltools.crystal_fd.checks import same_crystal
    from mxtaltools.crystal_fd.so3 import rotvec_to_matrix
    eps = torch.as_tensor(rng.normal(size=(nc, 3)), dtype=F64)
    eps = 0.05 * eps / eps.norm(dim=1, keepdim=True)
    rv = torch.as_tensor(rng.normal(size=(nc, 3)), dtype=F64)
    rv = (0.05 / mi['rg']) * rv / rv.norm(dim=1, keepdim=True)
    mv = CrystalDescription.from_parts(sg, base.cell, base.t[:, 0] + torch.linalg.solve(base.cell_matrix(), eps[..., None])[..., 0],
                                       rotvec_to_matrix(rv) @ base.R[:, 0], base.h[:, 0])
    dev = same_crystal(base, mv, mi['ref'])
    dm = metric.pairs(A, sel, metric.prepare(canonical_desc(mv)), torch.arange(nc))
    lines += [f'Scale check (t moved 0.05 A and R turned 0.05/r_g rad, random directions, same cell; {nc} rows): d '
              f'median {float(dm.median()):.4f} A (expected ~0.0707 = 0.05 sqrt 2); literal largest atom deviation '
              f'(checks.same_crystal) median {float(dev.median()):.4f} A, max {float(dev.max()):.4f} A.', '']

    # check 3: the molecular-mirror twin
    tw = twin_canonical(sg, can, mi['sigma'])
    At = metric.prepare(tw)
    d_tw = metric.pairs(A, torch.arange(len(idx)), At, torch.arange(len(idx)), parts=True)
    twd = CrystalDescription.from_poses(sg, can['cell'][sel], can['t'][sel], can['R'][sel] @ mi['sigma'])
    dev_tw = same_crystal(base, twd, mi['ref'])
    lines += [f'Mirror twin B~ = (t, R sigma) of every sampled endpoint: literal largest atom deviation from the endpoint '
              f'in identity atom order median {float(dev_tw.median()):.4f} A, max {float(dev_tw.max()):.4f} A ({nc} rows); '
              f'with like atoms matched it is the molecule\'s matched largest deviation, '
              f'{mi["mirror_maxdev_matched"]:.4f} A, for every row (a rigid motion keeps distances). d(endpoint, its own '
              f'twin) median {float(d_tw[0].median()):.3f} A, 10% {float(d_tw[0].quantile(0.1)):.3f} A, rotation part '
              f'median {float(d_tw[1][:, 2].median()):.3f} A. A twin is the same crystal to the matched mirror RMSD '
              f'({mi["mirror_rmsd_matched"]:.4f} A here) when the matching preserves the bond graph.', '']

    # timing of the distance
    ia = torch.as_tensor(rng.integers(0, len(idx), 20000))
    ib = torch.as_tensor(rng.integers(0, len(idx), 20000))
    t0 = time.time()
    metric.pairs(A, ia, A, ib)
    t_pair = (time.time() - t0) / 20000
    t0 = time.time()
    metric.lower_bound(A, ia, A, ib)
    t_lb = (time.time() - t0) / 20000
    lines += [f'Cost: d {1e6 * t_pair:.1f} us per pair ({27 * metric.m} images), pre-screen bound {1e6 * t_lb:.2f} us per '
              f'pair; {C.machine_note()}.', '']

    # face census
    deltas = (0.02, 0.05, 0.1, 0.2, 0.5)
    t0 = time.time()
    fc, axes = face_census(sg, can, metric, deltas, rng)
    names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    lines += ['## Endpoints near a cell-domain face', '',
              f'*Endpoints whose canonical form jumps under an axis-aligned cell perturbation of size delta (each of '
              f'{", ".join(names[x] for x in axes)} moved by +-delta in D_cell units, t and R kept, re-canonicalised): '
              f'the re-canonicalised form lies more than 1.5 delta + 1e-4 A from the endpoint in d, i.e. d would split '
              f'two crystals delta apart. {len(idx)} endpoints within {a.band_kT} kT; the same rows at every delta. '
              f'"basis changed" counts endpoints where some perturbation changed the reduced basis (N != I), split or '
              f'not. ({time.time() - t0:.0f} s)*', '',
              '| delta (A) | endpoints split | share | basis changed | median largest d of split rows (A) | splits per '
              'perturbation (' + ', '.join(f'{s}{names[x]}' for x in axes for s in '+-') + ') |',
              '|---|---|---|---|---|---|']
    for dlt in deltas:
        r = fc[dlt]
        lines.append(f'| {dlt} | {r["split"]} | {r["split"] / len(idx):.4f} | {r["basis_changed"]} | '
                     f'{r["dmax_median_split"]:.3f} | {r["per_axis"]} |')
    lines.append('')
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    torch.save(dict(idx=idx, can={k: v for k, v in can.items() if k != 'status'}, status=can['status'],
                    face={d: {k: v for k, v in r.items()} for d, r in fc.items()}, rg=mi['rg'],
                    mirror_rmsd=mi['mirror_rmsd'], mirror_rmsd_matched=mi['mirror_rmsd_matched'],
                    mirror_maxdev_matched=mi['mirror_maxdev_matched'], perm=mi['perm'], sigma=mi['sigma'], t_can=t_can, t_pair=t_pair, t_lb=t_lb,
                    d_self=d_self, d_twin_self=d_tw[0], scale_d=dm, scale_dev=dev),
               os.path.join(out, 'results.pt'))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
