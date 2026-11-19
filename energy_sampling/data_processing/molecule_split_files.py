"""The two per-molecule files a conditional MLE run reads beside its pre-noised prior (data_processing/
build_noised_prior.py), built from scored prior files:

  conditions  cfg:molecules_path -- one row per training molecule: its lowest-energy training-split anchor with
              packing inside energies.molecular_crystal.PACKING_COEFF_WINDOW. The run draws its conditions from it.
  heldout     cfg:test_molecules_path -- held-out molecules' crystals: the anchors of a scored held-out file within W
              kT of their molecule's lowest (T 2.5, density coefficient 10), packing in the window; at most
              --per_molecule per molecule (lowest first) and --max_molecules molecules (a seeded draw), so the
              eval_test readouts stay a fixed, affordable row set at large molecule counts. One file per W.
  heldout_groups  cfg:test_molecules_path for a run over several space groups, drawn uniformly over the groups
              (owner 2026-11-20): one scored held-out file per group; the SAME seeded draw of --molecules held-out
              molecules, from those with a crystal in every file's window, and from each file at most --per_molecule
              of each molecule's crystals within W kT of that molecule's own floor in that file (lowest first),
              concatenated in file order. A merged file through `heldout` would take each molecule's floor across the
              groups and so favour whichever group is lowest.

usage, from energy_sampling/:
    python data_processing/molecule_split_files.py conditions TRAIN_PRIOR.pt OUT.pt
    python data_processing/molecule_split_files.py heldout HELDOUT.pt OUT_STEM W [W ...] [--per_molecule N]
        [--max_molecules M] [--seed S]
    python data_processing/molecule_split_files.py heldout_groups OUT.pt W HELDOUT_G1.pt HELDOUT_G2.pt [...]
        --molecules M --per_molecule N [--seed S]
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import noised_prior as NP  # noqa: E402
from energies.molecular_crystal import PACKING_COEFF_WINDOW as PW  # noqa: E402

T_KT, DENSITY_COEFF = 2.5, 10.0


def _energy(b):
    return (b.mol_energy + DENSITY_COEFF * b.density_energy + b.pressure_energy).double().flatten()


def _in_window(b):
    pk = b.packing_coeff.double().flatten()
    return (pk >= PW[0]) & (pk <= PW[1])


def conditions(src, out):
    d = torch.load(src, weights_only=False, map_location='cpu')
    b = d['prior']
    NP.refuse_stale_atom_pointers(b, src)
    e = _energy(b)
    ok = ((NP.tag(b, 'noised_kind') == NP.KIND_ANCHOR) & (NP.tag(b, 'noised_split') == NP.SPLIT_TRAIN)
          & _in_window(b) & torch.isfinite(e))
    ids = np.asarray([str(x) for x in b.identifier])
    rows = torch.nonzero(ok).flatten().numpy()
    # lowest energy per molecule: sort the kept rows by (molecule, energy) and take each molecule's first
    order = np.lexsort((e.numpy()[rows], ids[rows]))
    rows = rows[order]
    first = np.ones(len(rows), dtype=bool)
    first[1:] = ids[rows][1:] != ids[rows][:-1]
    keep = torch.as_tensor(np.sort(rows[first]))
    sub = b.subsample_new_batch(keep)
    torch.save({'prior': sub, 'thermal_scaling_factor': d['thermal_scaling_factor'],
                'note': f'one condition row per training molecule: its lowest training anchor; from {src}'}, out)
    print(f'{out}: {len(keep)} rows; {len(set(ids.tolist()))} molecules in the prior, '
          f'{len(set(ids.tolist())) - len(keep)} without a training anchor in the packing window')


def heldout(src, stem, windows, per_molecule=None, max_molecules=None, seed=0):
    d = torch.load(src, weights_only=False, map_location='cpu')
    b = d['prior']
    NP.refuse_stale_atom_pointers(b, src)
    e = _energy(b)
    inpk = _in_window(b)
    groups = NP.condition_groups(b)
    floor = NP.condition_floor(e, groups, torch.isfinite(e) & inpk)
    du = ((e - floor) / T_KT).numpy()
    ids = np.asarray([str(x) for x in b.identifier])
    mols = np.unique(ids)
    if max_molecules is not None and len(mols) > max_molecules:
        mols = np.sort(np.random.default_rng(seed).choice(mols, size=max_molecules, replace=False))
    chosen = np.isin(ids, mols)
    for w in windows:
        rows = np.nonzero(chosen & inpk.numpy() & np.isfinite(du) & (du <= w))[0]
        order = np.lexsort((du[rows], ids[rows]))
        rows = rows[order]
        if per_molecule is not None:
            rank = np.zeros(len(rows), dtype=np.int64)
            for i in range(1, len(rows)):
                rank[i] = rank[i - 1] + 1 if ids[rows[i]] == ids[rows[i - 1]] else 0
            rows = rows[rank < per_molecule]
        keep = torch.as_tensor(np.sort(rows))
        sub = b.subsample_new_batch(keep)
        out = f'{stem}_w{w:g}.pt'
        torch.save({'prior': sub, 'thermal_scaling_factor': d['thermal_scaling_factor'],
                    'note': f'held-out molecules: anchors within {w:g} kT (T {T_KT}) of each molecule floor, packing '
                            f'in {PW}, at most {per_molecule} per molecule, {len(mols)} molecules; from {src}'}, out)
        counts = np.unique(ids[keep.numpy()], return_counts=True)[1]
        print(f'{out}: {len(keep)} crystals of {len(counts)} molecules; per molecule min/median/max '
              f'{counts.min()}/{np.median(counts):g}/{counts.max()}')



def _eligible(b, w):
    """(identifiers [n], du [n] in kT above each molecule's floor in this file, eligible [n] bool: in the packing
    window and within w kT)."""
    e = _energy(b)
    inpk = _in_window(b)
    floor = NP.condition_floor(e, NP.condition_groups(b), torch.isfinite(e) & inpk)
    du = ((e - floor) / T_KT).numpy()
    return (np.asarray([str(x) for x in b.identifier]), du,
            inpk.numpy() & np.isfinite(du) & (du <= w))


def heldout_groups(srcs, out, w, molecules, per_molecule, seed=0):
    from energies.molecular_crystal import merge_batches
    if os.path.exists(out):
        raise SystemExit(f'{out} exists')
    files, tsf = [], set()
    for src in srcs:
        d = torch.load(src, weights_only=False, map_location='cpu')
        NP.refuse_stale_atom_pointers(d['prior'], src)
        sg = sorted({int(x) for x in torch.as_tensor(d['prior'].sg_ind).reshape(-1).tolist()})
        tsf.add(d.get('thermal_scaling_factor'))
        files.append((src, d['prior'], sg) + _eligible(d['prior'], w))
    if len(tsf) != 1:
        raise SystemExit(f'the files differ in thermal_scaling_factor: {tsf}')
    common = sorted(set.intersection(*[set(ids[ok].tolist()) for _, _, _, ids, _, ok in files]))
    if len(common) < molecules:
        raise SystemExit(f'only {len(common)} molecules have a crystal within {w:g} kT in every file')
    mols = np.sort(np.random.default_rng(seed).choice(np.asarray(common), size=molecules, replace=False))
    parts, report = [], []
    for src, b, sg, ids, du, ok in files:
        rows = np.nonzero(ok & np.isin(ids, mols))[0]
        rows = rows[np.lexsort((du[rows], ids[rows]))]
        rank = np.zeros(len(rows), dtype=np.int64)
        for i in range(1, len(rows)):
            rank[i] = rank[i - 1] + 1 if ids[rows[i]] == ids[rows[i - 1]] else 0
        rows = np.sort(rows[rank < per_molecule])
        sub = b.subsample_new_batch(torch.as_tensor(rows))
        NP.strip_analysis_helpers(sub)
        parts.append(sub)
        counts = np.unique(ids[rows], return_counts=True)[1]
        report.append(f'space group {sg}: {len(rows)} crystals of {len(counts)} molecules, per molecule '
                      f'min/median/max {counts.min()}/{np.median(counts):g}/{counts.max()} ({src})')
    # layout-specific chart rows (the trainer attaches its own) and any column a file lacks are left out, named
    for b in parts:
        for key in ('fd_state', 'fd_log_reference'):
            if key in b.keys():
                delattr(b, key)
    shared = set.intersection(*[set(b.keys()) for b in parts])
    dropped = sorted(set.union(*[set(b.keys()) for b in parts]) - shared)
    for b in parts:
        for key in dropped:
            if key in b.keys():
                delattr(b, key)
    if dropped:
        print(f'columns not in every file, left out: {dropped}')
    merged = merge_batches(parts)
    torch.save({'prior': merged, 'thermal_scaling_factor': tsf.pop(),
                'note': f'held-out molecules over {len(srcs)} space groups, uniform: the same {molecules} molecules '
                        f'(seed {seed}) in every group, at most {per_molecule} crystals each within {w:g} kT (T {T_KT}) '
                        f'of the molecule floor in that group, packing in {PW}; from {list(srcs)}'}, out)
    print(f'{out}: {int(merged.num_graphs)} crystals, {molecules} molecules drawn from {len(common)} common to all '
          f'{len(srcs)} files')
    for r in report:
        print('  ' + r)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('conditions')
    c.add_argument('src')
    c.add_argument('out')
    h = sub.add_parser('heldout')
    h.add_argument('src')
    h.add_argument('stem')
    h.add_argument('windows', nargs='+', type=float)
    h.add_argument('--per_molecule', type=int, default=None)
    h.add_argument('--max_molecules', type=int, default=None)
    h.add_argument('--seed', type=int, default=0)
    g = sub.add_parser('heldout_groups')
    g.add_argument('out')
    g.add_argument('window', type=float)
    g.add_argument('srcs', nargs='+')
    g.add_argument('--molecules', type=int, required=True)
    g.add_argument('--per_molecule', type=int, required=True)
    g.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.cmd == 'conditions':
        conditions(a.src, a.out)
    elif a.cmd == 'heldout':
        heldout(a.src, a.stem, a.windows, a.per_molecule, a.max_molecules, a.seed)
    else:
        heldout_groups(a.srcs, a.out, a.window, a.molecules, a.per_molecule, a.seed)
