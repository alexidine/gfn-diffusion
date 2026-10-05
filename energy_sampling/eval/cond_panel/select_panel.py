"""
Candidate molecules for the per-condition panel, one short list per slot, for the owner to
pick from. Nothing here samples or scores a crystal; it reads the conditions files, the
checkpoint's conditioner and tracker, and RDKit descriptors of each identifier (a SMILES).

SLOTS (proposal section 1). Six training molecules and four held-out:

  small_rigid        train, <= 6 heavy atoms, no rotatable bond, a well-separated inertia frame
  large_flexible     train, 9 heavy atoms, the most rotatable bonds
  planar_aromatic    train, an aromatic ring, heavy atoms within 0.05 A rms of one plane
  frame_ambiguous    train, >= 6 heavy atoms, the smallest relative gap between principal
                     moments: orient_molecule(mode='std') fixes the frame from those axes and
                     their signs, so a near-degenerate pair makes the frame fragile
  isosteric_a/_b     train, a pair sharing one heavy-atom graph once elements and bond orders
                     are erased, differing by >= 2 in N+O count but by <= 2 hydrogens, with
                     every all-atom principal extent within 10%. ELJ has no electrostatics and
                     no hydrogen-bond term, so their landscapes should nearly coincide; the
                     pair is the control that separates "the model reads the energy-relevant
                     shape" from "the model reads the label". Hydrogens are explicit ELJ
                     sites, so a heteroatom swap that sheds hydrogens changes the ELJ shape:
                     the hydrogen and extent bars are what keep the pair a control
  held_out_q1..q4    held-out, nearest-training-molecule distance in the checkpoint's own
                     conditioner space at the 12.5 / 37.5 / 62.5 / 87.5 percentiles of the
                     held-out set: generalisation read against novelty

Every candidate is a fixed point of orient_molecule(mode='std') within 1e-3 A (about 1% of
the qm9c100k rows are not). Within a slot the default is the candidate, among the slot's
best five by its rule, farthest in embedding space from the defaults already chosen.

    cd energy_sampling
    CUDA_VISIBLE_DEVICES=-1 python -m eval.cond_panel.select_panel \
        --checkpoint D:/.../<stem>_step28000.pt \
        --config configs/cond_tb_sep25/ctb25_extreme_l1_cont3.yaml --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Lipinski, rdMolDescriptors

from eval.cond_panel.sampler import load_run

RDLogger.DisableLog('rdApp.*')
FIXED_POINT_TOL = 1e-3  # A, as build_qm9_conditions.py::pin_to_trainer_frame
HELD_OUT_QUANTILES = (0.125, 0.375, 0.625, 0.875)


def generic_graph_key(mol):
    """Heavy-atom graph with every element set to C and every bond single."""
    rw = Chem.RWMol(Chem.RemoveHs(mol))
    for atom in rw.GetAtoms():
        atom.SetAtomicNum(6)
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    return Chem.MolToSmiles(m)


def graph_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))
    return {
        'n_heavy': mol.GetNumHeavyAtoms(),
        'n_atoms_graph': mol_h.GetNumAtoms(),
        'n_rot': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'n_arom_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'n_NO': sum(a.GetAtomicNum() in (7, 8) for a in mol.GetAtoms()),
        'hbd': Lipinski.NumHDonors(mol),
        'hba': Lipinski.NumHAcceptors(mol),
        # atoms sharing a symmetry class with another atom: 0 means atomwise and envwise
        # RDF channels coincide; methyl hydrogens alone make this 2 per methyl
        'n_equiv_atoms': mol_h.GetNumAtoms() - len(set(ranks)),
        # heavy-atom graph automorphisms, identity included: > 1 is where atomwise RDF
        # distance stopped being monotonic in P(same packing) on acridine
        'heavy_automorphisms': len(mol.GetSubstructMatches(mol, uniquify=False, useChirality=False)),
        'generic_key': generic_graph_key(mol),
    }


def shape_descriptors(pos, z):
    heavy = z > 1
    hp, hz = pos[heavy], z[heavy].float()
    mass = torch.where(hz == 6, 12.011, torch.where(hz == 7, 14.007, torch.where(hz == 8, 15.999, 18.998)))
    c = (hp * mass[:, None]).sum(0) / mass.sum()
    r = hp - c
    inertia = (mass[:, None, None] * ((r * r).sum(1)[:, None, None] * torch.eye(3)
                                      - r[:, :, None] * r[:, None, :])).sum(0)
    ev = torch.linalg.eigvalsh(inertia.double()).clamp(min=1e-9)
    gaps = ((ev[1] - ev[0]) / ev[1], (ev[2] - ev[1]) / ev[2])
    gyr = torch.linalg.eigvalsh(((r[:, :, None] * r[:, None, :]).mean(0)).double()).clamp(min=0)
    ra = pos - pos.mean(0)
    g_all = torch.linalg.eigvalsh(((ra[:, :, None] * ra[:, None, :]).mean(0)).double()).clamp(min=0)
    s = g_all.sum()
    return {
        'n_H': int((z == 1).sum()),
        'frame_gap': float(min(gaps)),
        'planarity_rms': float(gyr[0].sqrt()),
        'asphericity': float(1 - 3 * (g_all[0] * g_all[1] + g_all[1] * g_all[2] + g_all[2] * g_all[0]) / s ** 2),
        # all-atom rms extent along each principal axis (A), smallest first
        'extent1': float(g_all[0].sqrt()), 'extent2': float(g_all[1].sqrt()), 'extent3': float(g_all[2].sqrt()),
    }


def fixed_point_moves(batch):
    """Per-molecule max atom displacement under a second orient_molecule(mode='std')."""
    probe = batch.clone()
    probe.orient_molecule(mode='std')
    d = (probe.pos - batch.pos).norm(dim=1)
    return torch.stack([d[batch.ptr[i]:batch.ptr[i + 1]].max() for i in range(batch.num_graphs)])


@torch.no_grad()
def conditioner_coords(run, batch):
    b = batch.to(run.device)
    _, _, cond, _ = run.energy_function.condition_samples(b)
    return run.gfn.get_condition_embedding(cond.to(run.device), b).cpu().double()


def table_rows(run, batch, split, emb16):
    ids = list(batch.identifier)
    moves = fixed_point_moves(batch)
    tr = run.tracker
    out = []
    for i, ident in enumerate(ids):
        g = graph_descriptors(ident)
        if g is None:
            continue
        sl = slice(int(batch.ptr[i]), int(batch.ptr[i + 1]))
        row = {'identifier': ident, 'split': split, 'row': i,
               'n_atoms_file': int(batch.num_atoms[i]), 'fixed_point_move_A': float(moves[i])}
        row.update(g)
        row.update(shape_descriptors(batch.pos[sl].double(), batch.z[sl]))
        cid = run.registry[ident]
        row['condition_id'] = cid
        for key in ('ema_logw', 'fwd_level_ema', 'bwd_level_ema', 'count'):
            v = tr.get(key)
            row[key] = float(v[cid]) if v is not None else float('nan')
        row['level_gap_bwd_minus_fwd'] = row['bwd_level_ema'] - row['fwd_level_ema']
        row['_emb16'] = emb16[i]
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True, help='directory for panel_candidates.csv and all_descriptors.csv')
    ap.add_argument('--per-slot', type=int, default=3)
    args = ap.parse_args()
    assert not torch.cuda.is_available(), 'CPU only: set CUDA_VISIBLE_DEVICES=-1'
    os.makedirs(args.out, exist_ok=True)

    run = load_run(args.checkpoint, args.config, device='cpu')
    e_tr = conditioner_coords(run, run.conditions)
    e_te = conditioner_coords(run, run.test_conditions)
    rows = table_rows(run, run.conditions, 'train', e_tr) + table_rows(run, run.test_conditions, 'held_out', e_te)

    # nearest training molecule in conditioner space; for a training row, the nearest OTHER one.
    # Chunked: one row at a time against every training row is 20 minutes at 130k molecules.
    # A training row's own column is masked BY INDEX: cdist's matrix-product form returns a
    # self-distance of order 1e-3, not 0, so a `d > 0` test keeps the row itself.
    tr_emb = torch.stack([r['_emb16'] for r in rows if r['split'] == 'train']).float()
    all_emb32 = torch.stack([r['_emb16'] for r in rows]).float()
    own = torch.full((len(rows),), -1, dtype=torch.long)  # a training row's column in tr_emb
    own[torch.tensor([r['split'] == 'train' for r in rows])] = torch.arange(tr_emb.shape[0])
    nearest = torch.empty(len(rows))
    for start in range(0, len(rows), 512):
        d = torch.cdist(all_emb32[start:start + 512], tr_emb)
        col = own[start:start + 512]
        has = col >= 0
        d[torch.nonzero(has).flatten(), col[has]] = float('inf')
        nearest[start:start + 512] = d.min(dim=1).values
    for r, v in zip(rows, nearest.tolist()):
        r['nearest_train_dist16'] = v
    # 3-d PCA of the conditioner coordinates, for spreading the defaults
    all_emb = torch.stack([r['_emb16'] for r in rows])
    centred = all_emb - all_emb.mean(0)
    _, _, v = torch.linalg.svd(centred, full_matrices=False)
    pcs = centred @ v[:3].T
    pcs = pcs / pcs.std(0)
    for r, p in zip(rows, pcs):
        r['pc1'], r['pc2'], r['pc3'] = (float(x) for x in p)

    ok = [r for r in rows if r['fixed_point_move_A'] <= FIXED_POINT_TOL]
    train = [r for r in ok if r['split'] == 'train']
    held = [r for r in ok if r['split'] == 'held_out']

    slots = {
        'small_rigid': sorted([r for r in train if r['n_heavy'] <= 6 and r['n_rot'] == 0 and r['frame_gap'] >= 0.10],
                              key=lambda r: (r['n_heavy'], -r['frame_gap'])),
        'large_flexible': sorted([r for r in train if r['n_heavy'] == 9],
                                 key=lambda r: (-r['n_rot'], -r['asphericity'])),
        'planar_aromatic': sorted([r for r in train if r['n_arom_rings'] >= 1 and r['planarity_rms'] <= 0.05],
                                  key=lambda r: (-r['n_heavy'], r['planarity_rms'])),
        'frame_ambiguous': sorted([r for r in train if r['n_heavy'] >= 6], key=lambda r: r['frame_gap']),
    }
    # isosteric pairs: one generic graph, both rigid, N+O count apart by >= 2, hydrogens
    # within 2, every principal extent within 10%; the pair's score prefers the larger
    # polarity contrast, then the closer shape
    by_key = {}
    for r in train:
        if r['n_rot'] == 0:  # both members must be rigid, so only rigid rows are paired
            by_key.setdefault(r['generic_key'], []).append(r)

    def extent_diff(a, b):
        return max(abs(a[k] - b[k]) / max(a[k], b[k], 1e-6) for k in ('extent1', 'extent2', 'extent3'))

    pairs = []
    for members in by_key.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if (abs(a['n_NO'] - b['n_NO']) >= 2 and a['n_rot'] == 0 and b['n_rot'] == 0
                        and abs(a['n_H'] - b['n_H']) <= 2 and extent_diff(a, b) <= 0.10):
                    pairs.append((-abs(a['n_NO'] - b['n_NO']), extent_diff(a, b), a, b))
    pairs.sort(key=lambda t: (t[0], t[1]))
    qs = np.quantile([r['nearest_train_dist16'] for r in held], HELD_OUT_QUANTILES)
    for k, q in enumerate(qs, 1):
        slots[f'held_out_q{k}'] = sorted(held, key=lambda r, q=q: abs(r['nearest_train_dist16'] - q))

    chosen, lines = [], []

    def spread_pick(cands):
        pool = cands[:5]
        if not chosen:
            return pool[0]
        pts = np.array([[c['pc1'], c['pc2'], c['pc3']] for c in chosen])
        return max(pool, key=lambda r: np.linalg.norm(pts - np.array([r['pc1'], r['pc2'], r['pc3']]), axis=1).min())

    picks = []
    for name in ('small_rigid', 'large_flexible', 'planar_aromatic', 'frame_ambiguous'):
        default = spread_pick(slots[name])
        chosen.append(default)
        alts = [r for r in slots[name][:args.per_slot + 1] if r is not default][:args.per_slot - 1]
        picks.append((name, default, alts))
    if pairs:
        _, _, a, b = pairs[0]
        chosen += [a, b]
        alt_pairs = [f"{p[2]['identifier']} / {p[3]['identifier']}" for p in pairs[1:args.per_slot]]
        picks.append(('isosteric_a', a, []))
        picks.append(('isosteric_b', b, alt_pairs))
    for k in range(1, len(HELD_OUT_QUANTILES) + 1):
        name = f'held_out_q{k}'
        pool = [r for r in slots[name] if r not in chosen]
        default = spread_pick(pool)
        chosen.append(default)
        alts = [r for r in pool[:args.per_slot + 1] if r is not default][:args.per_slot - 1]
        picks.append((name, default, alts))

    cols = ['identifier', 'split', 'n_heavy', 'n_H', 'n_rot', 'n_arom_rings', 'n_NO', 'hbd', 'hba',
            'frame_gap', 'planarity_rms', 'asphericity', 'heavy_automorphisms', 'n_equiv_atoms',
            'nearest_train_dist16', 'ema_logw', 'level_gap_bwd_minus_fwd']
    print('\nPanel candidates. Default = the slot rule\'s best five, farthest in conditioner-space '
          'PCA from the defaults already chosen; alternatives follow the rule order. frame_gap is '
          'the smallest relative gap between principal moments of the heavy atoms (dimensionless); '
          'planarity is the heavy atoms\' rms distance from their best plane (A); nearest_train '
          'is the Euclidean distance to the nearest (other) training molecule in the step-'
          f'{run.step} conditioner output; ema_logw and the bwd-fwd level gap are the tracker\'s '
          '(nats, training molecules only).\n')
    print('| slot | identifier (SMILES) | ' + ' | '.join(cols[1:]) + ' | alternatives |')
    print('|' + '---|' * (len(cols) + 2))
    fmt = lambda v: f'{v:.3g}' if isinstance(v, float) else str(v)
    for name, r, alts in picks:
        alt = '; '.join(a if isinstance(a, str) else a['identifier'] for a in alts)
        print(f'| {name} | {r["identifier"]} | ' + ' | '.join(fmt(r[c]) for c in cols[1:]) + f' | {alt} |')

    with open(os.path.join(args.out, 'panel_candidates.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['slot', 'rank'] + cols)
        for name, r, alts in picks:
            w.writerow([name, 'default'] + [r[c] for c in cols])
            for a in alts:
                if not isinstance(a, str):
                    w.writerow([name, 'alternative'] + [a[c] for c in cols])
    keys = [k for k in rows[0] if not k.startswith('_')]
    with open(os.path.join(args.out, 'all_descriptors.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    n_moved = sum(r['fixed_point_move_A'] > FIXED_POINT_TOL for r in rows)
    print(f'\n{len(rows)} molecules described; {n_moved} excluded as not std-frame fixed points '
          f'(> {FIXED_POINT_TOL} A); {len(pairs)} isosteric pairs found.')


if __name__ == '__main__':
    main()
