"""Is UMA-s's mirror (parity) error in a chiral space group a constant offset, or does it reshape the landscape?

In a Sohncke group a crystal x and its mirror image sigma x (every atom inverted through the origin; same group, same
lattice) are different crystals with one true energy. A rotation- but not reflection-equivariant MLIP scores them
differently. For each molecule, in a chiral group:

1. structures: the lowest minima of a short exp-6 search in that group (realistic near-minimum crystals);
2. single points: raw UMA-s 1.1 lattice energies R(x) and R(sigma x) (mirror-mean gas reference, a constant per
   molecule), and the averaged A(x) the runs use in chiral groups; check 2 A(x) - R(x) = R(sigma x), which proves the
   mirror rows are built right;
3. paired relaxations under raw UMA-s: x and sigma x relaxed separately; the mirror of sigma x's minimum is compared
   with x's minimum (symmetry-aware FD distance, basins.SymFD) and the two minimum energies are compared.

An offset-like error has a narrow spread of R(sigma x) - R(x) across structures, ranks structures the same way under
R(x) and R(sigma x), and leaves paired minima at FD ~ 0 with a constant energy difference.

Outputs in OUT_DIR (a new directory): the searches, the seeds runs, mirror_test.json.

usage: python -m eval.exp6_recall.mirror_test OUT_DIR [--molecules pyridone:19,naphthalene:4,acetic_acid:4]
       [--n_single 60] [--n_relax 20] [--n_starts 600]
"""
import argparse
import json
import os

import torch

from eval.exp6_recall.basins import SymFD
from eval.exp6_recall.make_configs import KT, config

PANEL = 'D:/crystal_datasets/exp6_recall_nov10/panel'
UMA = 'D:/crystal_datasets/uma-s-1p1.pt'
FRACTION = 0.4


def cap_gpu():
    orig = torch.cuda.set_per_process_memory_fraction
    torch.cuda.set_per_process_memory_fraction = lambda f, device=None: orig(FRACTION, device=0)
    torch.cuda.set_per_process_memory_fraction(FRACTION)


def run(c):
    from mxtaltools.crystal_search import run_search
    if os.path.exists(os.path.join(c['out_dir'], c['run_name'] + '_minima.pt')):
        print(f"{c['run_name']}: outputs exist; kept", flush=True)
    else:
        run_search.crystal_search(run_search.dict2namespace(c))
    return (list(torch.load(os.path.join(c['out_dir'], c['run_name'] + '_minima.pt'), weights_only=False)),
            list(torch.load(os.path.join(c['out_dir'], c['run_name'] + '.pt'), weights_only=False)))


def clean(r):
    r = r.clone()
    for k in ('uma_pot', 'uma_gas_pot', 'uma_mirror_average', 'unit_cell_pos', 'unit_cell_batch'):
        if hasattr(r, k):
            delattr(r, k)
    return r


def mirrored(rows, sg):
    """sigma x for each row: the description (cell, -t, -P) with P = R diag(h, 1, 1), i.e. every atom inverted
    through the origin. In P2_1 and P2_12_12_1 every translation is 0 or 1/2, so the inverted crystal keeps the
    group's setting."""
    from mxtaltools.crystal_fd.description import CrystalDescription
    p = torch.stack([r.full_cell_parameters().reshape(-1) for r in rows]).double()
    h = torch.stack([r.aunit_handedness.reshape(-1)[0] for r in rows]).reshape(-1, 1)
    d = CrystalDescription.from_crystal_params(sg, p[:, :3], p[:, 3:6], p[:, 6:9], p[:, 9:12], h)
    hh = torch.ones(len(rows), 3, dtype=torch.float64)
    hh[:, 0] = d.h[:, 0].double()
    P = d.R[:, 0] * hh[:, None, :]
    m = CrystalDescription.from_poses(sg, d.cell, torch.remainder(-d.t[:, 0], 1.0), -P)
    lengths, angles, cent, rv, hand = m.crystal_params()
    out = []
    for i, r in enumerate(rows):
        q = clean(r)
        dt = q.full_cell_parameters().dtype
        q.set_cell_parameters(torch.cat([lengths[i], angles[i], cent[i], rv[i]])[None].to(dt), skip_box_analysis=True)
        q.aunit_handedness = torch.full_like(r.aunit_handedness, int(hand[i, 0]))
        out.append(q)
    return out


def energies(rows, cfg, pred, average):
    """Lattice energies (kJ/mol per molecule) of rows under the uma config's parity keys, mirror averaging on/off."""
    from mxtaltools.crystal_search.minima import StageEnergy
    from mxtaltools.crystal_search.run_search import attach_mlip_parity, dict2namespace
    c = dict(cfg, mlip_mirror_average=bool(average))
    rows = [clean(r) for r in rows]
    attach_mlip_parity(dict2namespace(c), rows, 'cuda')
    E = StageEnergy(rows, 'uma', KT['uma'], predictor=pred, device='cuda', chunk=4)
    x = torch.stack([r.full_cell_parameters().reshape(-1) for r in rows]).double()
    with torch.no_grad():
        return (E.f(x, list(range(len(rows)))) * KT['uma']).double().cpu()


def write_seeds(rows, path):
    torch.save([clean(r) for r in rows], path)


def paired_minima(minima, starts, n):
    """Seed j's minimum row (None if it did not converge), j = 0..n-1."""
    byid = {int(r.minimum_id): r for r in minima}
    out = [None] * n
    for s in starts:
        j = -int(s.start_index) - 1
        if 0 <= j < n and int(s.minimum_id) >= 0:
            out[j] = byid.get(int(s.minimum_id))
    return out


def one(name, sg, out, a, pred):
    mol = os.path.join(PANEL, f'{name}_mol.pt')
    # 1. structures: an exp-6 search in sg
    run_name, c = config(name, sg, 'exp6', 'search', PANEL, out, 'local', n=a.n_starts)
    c['run_name'] = f'{name}_sg{sg}_exp6_search'
    minima, _ = run(c)
    minima.sort(key=lambda r: float(r.minimum_f))
    X = minima[:a.n_single]
    MX = mirrored(X, sg)
    # 2. single points
    _, cu = config(name, sg, 'uma', 'seeds', PANEL, out, 'local', seeds='unused')
    Rx = energies(X, cu, pred, False)
    Rm = energies(MX, cu, pred, False)
    Ax = energies(X, cu, pred, True)
    check = float((2 * Ax - Rx - Rm).abs().max())
    d = Rm - Rx
    rank = lambda v: torch.argsort(torch.argsort(v)).double()
    rho = float(torch.corrcoef(torch.stack([rank(Rx), rank(Rm)]))[0, 1])
    res = dict(sg=sg, n_single=len(X), check_max_kJ=check, d_mean=float(d.mean()), d_std=float(d.std()),
               d_min=float(d.min()), d_max=float(d.max()), spread_Rx=float(Rx.max() - Rx.min()),
               spearman_Rx_Rm=rho, lowest_same=int(torch.argmin(Rx)) == int(torch.argmin(Rm)))
    # 3. paired relaxations under raw UMA-s
    n = min(a.n_relax, len(X))
    for tag, rows in (('x', X[:n]), ('mx', MX[:n])):
        sp = os.path.join(out, f'{name}_sg{sg}_{tag}_seeds.pt')
        if not os.path.exists(sp):
            write_seeds(rows, sp)
    ends = {}
    for tag in ('x', 'mx'):
        _, cs = config(name, sg, 'uma', 'seeds', PANEL, out, 'local', seeds=os.path.join(out, f'{name}_sg{sg}_{tag}_seeds.pt'))
        cs['run_name'] = f'{name}_sg{sg}_{tag}_uma_raw'
        cs['mlip_mirror_average'] = False
        mins, starts = run(cs)
        ends[tag] = paired_minima(mins, starts, n)
    pairs = [(i, ends['x'][i], ends['mx'][i]) for i in range(n) if ends['x'][i] is not None and ends['mx'][i] is not None]
    rel = []
    if pairs:
        back = mirrored([p[2] for p in pairs], sg)  # sigma of sigma x's minimum, comparable with x's minimum
        rows = [p[1] for p in pairs] + back
        params = torch.stack([r.full_cell_parameters().reshape(-1) for r in rows]).double()
        hand = torch.stack([r.aunit_handedness.reshape(-1)[0] for r in rows])
        fd = SymFD(sg, mol, params, hand)
        k = len(pairs)
        D = fd.pairs(list(range(k)), [k + i for i in range(k)])
        for (i, mx, mm), dist in zip(pairs, D.tolist()):
            rel.append(dict(seed=i, fd_A=dist, E_x=float(mx.minimum_f) * KT['uma'], E_mx=float(mm.minimum_f) * KT['uma'],
                            dE=(float(mm.minimum_f) - float(mx.minimum_f)) * KT['uma']))
    res['relax'] = rel
    res['n_relax'] = n
    res['n_pairs_converged'] = len(pairs)
    res['single'] = dict(Rx=Rx.tolist(), Rm=Rm.tolist(), Ax=Ax.tolist())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--molecules', default='pyridone:19,naphthalene:4,acetic_acid:4')
    ap.add_argument('--n_single', type=int, default=60)
    ap.add_argument('--n_relax', type=int, default=20)
    ap.add_argument('--n_starts', type=int, default=600)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, 'mirror_test.json')):
        raise SystemExit(f'{a.out_dir} already holds a result; use a new directory')
    os.makedirs(a.out_dir, exist_ok=True)
    cap_gpu()
    from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
    pred = init_uma_crystal_predictor(UMA, device='cuda', activation_checkpointing=False, tf32=False)
    out = {}
    for spec in a.molecules.split(','):
        name, sg = spec.split(':')
        out[name] = one(name, int(sg), a.out_dir, a, pred)
        r = out[name]
        print(f"MIRROR {name} sg {r['sg']}: check {r['check_max_kJ']:.2e} kJ/mol; R(sx)-R(x) mean {r['d_mean']:.2f} "
              f"sd {r['d_std']:.2f} [{r['d_min']:.2f}, {r['d_max']:.2f}] over {r['n_single']} (R(x) spread "
              f"{r['spread_Rx']:.1f}); spearman {r['spearman_Rx_Rm']:.3f}; same lowest {r['lowest_same']}; "
              f"relax pairs {r['n_pairs_converged']}/{r['n_relax']}", flush=True)
        with open(os.path.join(a.out_dir, 'mirror_test.json'), 'w') as fh:
            json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
