"""Two pre-noised prior files built from the same anchors with different child kernels (a kick-scale test), compared
on what must match and what the kernel changes.

  radii    SRC LO HI --expect_log_hi H --width D
                          SRC's K/kernel and log_hi read from its build record; refuses unless log_hi is H; writes
                          log_hi to HI and log_hi - D to LO (the kick-scale test keeps the top radius, narrows the range)
  units    SRC OUT        the held-out descent groups of SRC's anchors: (identifier, space group, Z', basin_group) for
                          every held-out anchor, to OUT; build_noised_prior.py --heldout_units OUT holds out the same
  compare  A B [--window_kT W ...]
           1. the anchor sets: per split, each anchor's key (identifier, space group, Z', basin_group, basin_id) as a
              multiset, PASS when A's equals B's; and the largest crystal-parameter difference of the matched anchors
           2. each child's seed-energy excess over its parent anchor (noised_parent), in kT at the build temperature:
              median and P90, training and held-out children; and the stored kick radii
           3. the rows the trainer keeps at load (noised_prior.select_rows) at each energy window W kT, the packing
              window at its default, train_kinds child: training children, the buffer (training anchors + children),
              held-out children
  Exits 1 when check 1 fails. Files are memory-mapped; the per-row Python lists (identifier) are read whole.

usage, from energy_sampling/:
    python data_processing/prior_kick_compare.py radii SRC.pt LO.txt HI.txt --expect_log_hi -0.375 --width 0.3
    python data_processing/prior_kick_compare.py units SRC.pt OUT.pt
    python data_processing/prior_kick_compare.py compare A.pt B.pt [--window_kT 10] [--density_coeff 10]
"""
import argparse
import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import noised_prior as NP  # noqa: E402

PARAM_KEYS = ('cell_lengths', 'cell_angles', 'aunit_centroid', 'aunit_orientation')


def _load(path):
    f = torch.load(path, weights_only=False, map_location='cpu', mmap=True)
    return f['prior'], f.get(NP.MANIFEST_KEY, {})


def _col(b, key):
    return torch.as_tensor(getattr(b, key)).reshape(-1)


def radii(src, lo_out, hi_out, expect_log_hi, width):
    _, m = _load(src)
    hi = float(m['log_hi'])
    print(f"{src}: K {m.get('k')} {m.get('child_kernel')} kick at 10^U({m.get('log_lo')}, {hi}) "
          f"{m.get('kick_units')}, window_kT target {m.get('window_kT_target')}")
    if abs(hi - expect_log_hi) > 1e-9:
        raise SystemExit(f'{src}: log_hi {hi}, expected {expect_log_hi}')
    lo = round(hi - width, 9)
    for path, v in ((hi_out, hi), (lo_out, lo)):
        with open(path, 'w') as fh:
            print(v, file=fh)
    print(f'log_lo {lo} -> {lo_out}; log_hi {hi} -> {hi_out}')


def units(src, out):
    b, _ = _load(src)
    if getattr(b, 'basin_group', None) is None:
        raise SystemExit(f'{src}: no basin_group column (not a band-set build)')
    rows = torch.nonzero((NP.tag(b, 'noised_kind') == NP.KIND_ANCHOR)
                         & (NP.tag(b, 'noised_split') == NP.SPLIT_HELDOUT)).flatten().tolist()
    ids, sg, zp, grp = b.identifier, _col(b, 'sg_ind'), _col(b, 'z_prime'), _col(b, 'basin_group')
    keys = sorted({(str(ids[i]), int(sg[i]), int(zp[i]), int(grp[i])) for i in rows})
    torch.save({'units': keys, 'source': os.path.abspath(src),
                'note': 'held-out (identifier, space group, Z\', basin_group) of the source\'s anchors'}, out)
    print(f'{out}: {len(keys)} held-out descent groups ({len(rows)} anchors, '
          f'{len({k[0] for k in keys})} molecules) from {src}')


def anchor_keys(b):
    """({split: Counter of anchor keys (identifier, space group, Z', basin_group, basin_id)}, {key: [P] crystal
    parameters of its anchor, or None for a key held more than once})."""
    kind, split = NP.tag(b, 'noised_kind'), NP.tag(b, 'noised_split')
    rows = torch.nonzero(kind == NP.KIND_ANCHOR).flatten()
    params = torch.cat([torch.as_tensor(getattr(b, k)).reshape(int(b.num_graphs), -1)[rows].double()
                        for k in PARAM_KEYS], dim=1).numpy()
    ids = b.identifier
    cols = [_col(b, k)[rows].tolist() for k in ('sg_ind', 'z_prime', 'basin_group', 'basin_id')]
    sp = split[rows].tolist()
    out, par = {NP.SPLIT_TRAIN: Counter(), NP.SPLIT_HELDOUT: Counter()}, {}
    for j, i in enumerate(rows.tolist()):
        key = (str(ids[i]), *(int(c[j]) for c in cols))
        out[sp[j]][key] += 1
        par[key] = None if key in par else params[j]
    return out, par


def seed_energy(b, density_coeff):
    return (_col(b, 'mol_energy').double() + density_coeff * _col(b, 'density_energy').double()
            + _col(b, 'pressure_energy').double())


def child_excess(b, e, T):
    """{'train'/'heldout': [n] excess in kT of each finite child over its parent}, [n_child] radii."""
    kind, split, parent = NP.tag(b, 'noised_kind'), NP.tag(b, 'noised_split'), NP.tag(b, 'noised_parent')
    child = kind == NP.KIND_CHILD
    du = (e - e[parent]) / T
    ok = child & torch.isfinite(du)
    return ({'train': du[ok & (split == NP.SPLIT_TRAIN)], 'heldout': du[ok & (split == NP.SPLIT_HELDOUT)]},
            _col(b, 'noised_radius')[child].double())


def window_counts(b, e, T, w):
    s = NP.settings(SimpleNamespace(noised_prior={'window_kT': w, 'train_kinds': ['child']}))
    kind, split = NP.tag(b, 'noised_kind'), NP.tag(b, 'noised_split')
    masks, _ = NP.select_rows(e, _col(b, 'packing_coeff').double(), kind, split, NP.condition_groups(b), T, s)
    child = kind == NP.KIND_CHILD
    return dict(train_children=int(masks['train'].sum()), buffer=int(masks['buffer'].sum()),
                buffer_anchors=int((masks['buffer'] & ~child).sum()),
                heldout_children=int(masks['heldout'].sum()),
                children_train_split=int((child & (split == NP.SPLIT_TRAIN)).sum()))


def compare(a_path, b_path, windows, density_coeff):
    files = []
    for p in (a_path, b_path):
        b, man = _load(p)
        T = float(man.get('temperature', 2.5))
        files.append(SimpleNamespace(path=p, b=b, man=man, T=T, e=seed_energy(b, density_coeff),
                                     name=os.path.basename(p)))
    print('files:')
    for f in files:
        m = f.man
        print(f'  {f.name}: {int(f.b.num_graphs)} rows; K {m.get("k")} {m.get("child_kernel")} kick at '
              f'10^U({m.get("log_lo")}, {m.get("log_hi")}) {m.get("kick_units")}; T {f.T:g}; '
              f'lj_envelope {(m.get("identity") or {}).get("analyze_kwargs", {}).get("lj_envelope")}; '
              f'forces {"yes" if m.get("forces") else "no"}')

    # 1. anchors
    (ka, pa), (kb, pb) = anchor_keys(files[0].b), anchor_keys(files[1].b)
    ok = True
    print('\n1. anchor sets, per split (key: identifier, space group, Z\', basin_group, basin_id). Counts of anchors.')
    print(f'  {"split":<8} {"A anchors":>10} {"B anchors":>10} {"in both":>10} {"A only":>8} {"B only":>8}')
    for sp, name in ((NP.SPLIT_TRAIN, 'train'), (NP.SPLIT_HELDOUT, 'heldout')):
        both = sum((ka[sp] & kb[sp]).values())
        a_only, b_only = sum((ka[sp] - kb[sp]).values()), sum((kb[sp] - ka[sp]).values())
        ok &= a_only == 0 and b_only == 0
        print(f'  {name:<8} {sum(ka[sp].values()):>10} {sum(kb[sp].values()):>10} {both:>10} {a_only:>8} {b_only:>8}')
    common = [k for k, v in pa.items() if v is not None and pb.get(k) is not None]
    d = np.array([np.abs(pa[k] - pb[k]).max() for k in common]) if common else np.zeros(0)
    if d.size:
        print(f'  crystal parameters (cell lengths A, angles rad, centroid frac, orientation rotvec) of the {d.size} '
              f'anchors whose key is unique in both: largest |A - B| {d.max():.2e}; {int((d > 1e-3).sum())} above 1e-3')
    print(f'  anchor sets {"IDENTICAL" if ok else "DIFFER"}')

    # 2. child excess
    print(f'\n2. child seed-energy excess over its parent anchor (kT at T {files[0].T:g}; seed energy = mol + '
          f'{density_coeff:g} x density + pressure, as stored). Finite children only.')
    print(f'  {"file":<44} {"split":<8} {"children":>10} {"median (kT)":>12} {"P90 (kT)":>10} {"<= 6 kT":>8}')
    for f in files:
        ex, rad = child_excess(f.b, f.e, f.T)
        for name, du in ex.items():
            if du.numel():
                q = torch.quantile(du.float()[torch.randperm(du.numel())[:5_000_000]], torch.tensor([0.5, 0.9]))
                print(f'  {f.name:<44} {name:<8} {du.numel():>10} {float(q[0]):>12.2f} {float(q[1]):>10.2f} '
                      f'{100 * float((du <= 6).double().mean()):>7.1f}%')
        if rad.numel():
            rq = torch.quantile(rad.float()[torch.randperm(rad.numel())[:5_000_000]],
                                torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]))
            print(f'  {f.name:<44} kick radius (A) min/P10/P50/P90/max ' + ' / '.join(f'{float(v):.3g}' for v in rq))

    # 3. load-time window
    print('\n3. rows kept at load (noised_prior.select_rows: packing window at its default, train_kinds child, window '
          'W kT above the condition floor). Counts of rows.')
    print(f'  {"file":<44} {"W (kT)":>6} {"train children":>15} {"of":>10} {"buffer rows":>12} {"(anchors)":>10} '
          f'{"heldout children":>17}')
    for f in files:
        for w in windows:
            c = window_counts(f.b, f.e, f.T, w)
            print(f'  {f.name:<44} {w:>6g} {c["train_children"]:>15} {c["children_train_split"]:>10} '
                  f'{c["buffer"]:>12} {c["buffer_anchors"]:>10} {c["heldout_children"]:>17}')
    return ok


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('radii')
    r.add_argument('src')
    r.add_argument('lo_out')
    r.add_argument('hi_out')
    r.add_argument('--expect_log_hi', type=float, required=True)
    r.add_argument('--width', type=float, required=True)
    u = sub.add_parser('units')
    u.add_argument('src')
    u.add_argument('out')
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    c.add_argument('--window_kT', type=float, nargs='+', default=[10.0])
    c.add_argument('--density_coeff', type=float, default=10.0)
    a = p.parse_args()
    if a.cmd == 'radii':
        radii(a.src, a.lo_out, a.hi_out, a.expect_log_hi, a.width)
    elif a.cmd == 'units':
        units(a.src, a.out)
    else:
        sys.exit(0 if compare(a.a, a.b, a.window_kT, a.density_coeff) else 1)
