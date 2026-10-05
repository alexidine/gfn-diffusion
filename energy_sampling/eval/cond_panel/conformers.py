"""
Other conformers of training molecules, given to the model as conditions it never saw.

The crystal route conditions on a molecule's geometry (the Mo3ENet embedding of the stored
conformer, in the trainer's frame) and packs that geometry rigidly. For a flexible training
molecule this module builds two new conditions and draws from each:

  S  the SAME conformer after an MMFF relaxation started from the stored geometry: bond
     lengths and angles move by hundredths of an angstrom, the shape does not. It measures
     how much of a response comes from geometric noise alone.
  A  an ALTERNATE conformer: of the ETKDG conformers within `--window` kcal/mol (MMFF) of the
     lowest, the one farthest in heavy-atom RMSD from S, counting a mirror image as the same
     conformer (P-1 holds both hands). Skipped when nothing is farther than `--min-rmsd`.
  J  the stored geometry with every coordinate JITTERED by Gaussian noise of `--jitter`
     angstrom: the smallest change there is, and the control for the other two.

For each, the heavy-atom RMSD from the stored geometry is recorded twice: after the best
proper rotation (how different the shape is), and as the two sit in the trainer's frame (how
different the model's input is). The condition is the encoder's vectors read in that frame,
so a frame that turns over under a small change of shape changes the condition wholesale.

Only molecules without a potential stereocentre or stereo bond are used, so an embedded
conformer is the same molecule and not a diastereomer. New rows are the stored row with its
coordinates replaced, radius and volume recomputed, pinned to the trainer's frame
(orient_molecule(mode='std')) and embedded by the frozen encoder exactly as
build_qm9_conditions.py does; re-embedding the STORED geometry through the same path must
reproduce the stored embedding, and that gate is checked before anything is drawn. The
encoder's output depends on the batch it is run in, so the gate only holds when the new
geometries are embedded among stored molecules in a batch of the builder's size; how far a
molecule embedded alone lands from its stored embedding is recorded beside every result.

A new conformer has no search minima of its own. Its excess is measured against the stored
conformer's best search minimum, which is a different crystal problem: read it as a level,
not as a distance from that conformer's own optimum.

    cd energy_sampling
    CUDA_VISIBLE_DEVICES=-1 python -m eval.cond_panel.conformers --checkpoint ... --config ... \
        --atlas <dir>/atlas/atlas_step50000.pt --out <dir>/conformers
"""
from __future__ import annotations

import argparse
import os

import networkx as nx
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

from eval.cond_panel.latents import energy_distance, fold
from eval.cond_panel.motifs import COVALENT
from eval.cond_panel.pairs import density_lift, score_under, seed_excess
from eval.cond_panel.sampler import draw, load_run
from eval.cond_panel.select_panel import conditioner_coords

RDLogger.DisableLog('rdApp.*')
DEFAULT_ENCODER = r'D:\crystal_datasets\model_checkpoints\_best_autoencoder_experiments_dev_26-09-13-48-15'


def atom_map(z, pos, mol):
    """For each atom of the stored molecule, its index in the RDKit molecule (same elements,
    same bonds), or None when the stored geometry's bonds do not match the SMILES."""
    cov = np.array([COVALENT[int(a)] for a in z])
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    bonded = (d < 1.3 * (cov[:, None] + cov[None])) & ~np.eye(len(z), dtype=bool)
    gf, gr = nx.Graph(), nx.Graph()
    gf.add_nodes_from((i, {'z': int(a)}) for i, a in enumerate(z))
    gf.add_edges_from(zip(*np.nonzero(np.triu(bonded))))
    gr.add_nodes_from((a.GetIdx(), {'z': a.GetAtomicNum()}) for a in mol.GetAtoms())
    gr.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    if gf.number_of_nodes() != gr.number_of_nodes() or gf.number_of_edges() != gr.number_of_edges():
        return None
    gm = nx.algorithms.isomorphism.GraphMatcher(gf, gr, node_match=lambda a, b: a['z'] == b['z'])
    return np.array([gm.mapping[i] for i in range(len(z))]) if gm.is_isomorphic() else None


def other_conformers(smiles, z, pos, n_confs, window, min_rmsd, seed):
    """(S coordinates, A coordinates, RMSD of S from the stored geometry, RMSD of A from S,
    MMFF energy of A above S in kcal/mol), coordinates in the stored atom order; or the
    reason there is none."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if len(Chem.FindPotentialStereo(mol)) > 0:
        return 'has a potential stereocentre or stereo bond'
    mapping = atom_map(np.asarray(z), np.asarray(pos, dtype=np.float64), mol)
    if mapping is None:
        return 'stored bonds do not match the SMILES'
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return 'no MMFF parameters'
    stored = Chem.Conformer(mol.GetNumAtoms())
    for f, r in enumerate(mapping):
        stored.SetAtomPosition(int(r), [float(x) for x in pos[f]])
    cid_o = mol.AddConformer(stored, assignId=True)
    cid_s = mol.AddConformer(Chem.Conformer(mol.GetConformer(cid_o)), assignId=True)

    def relax(cid):
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        ff.Minimize(maxIts=4000)
        return ff.CalcEnergy()

    e_s = relax(cid_s)
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=seed, clearConfs=False))
    energies = {c: relax(c) for c in cids}
    heavy = Chem.RemoveHs(mol)
    mirror = Chem.Conformer(heavy.GetConformer(cid_s))
    for i in range(heavy.GetNumAtoms()):
        p = mirror.GetAtomPosition(i)
        mirror.SetAtomPosition(i, [-p.x, p.y, p.z])
    cid_m = heavy.AddConformer(mirror, assignId=True)
    rmsd = lambda c, ref: rdMolAlign.GetBestRMS(heavy, heavy, prbId=c, refId=ref)
    e_low = min(list(energies.values()) + [e_s])
    best, best_rmsd = None, min_rmsd
    for c in cids:
        if energies[c] - e_low > window:
            continue
        r = min(rmsd(c, cid_s), rmsd(c, cid_m))
        if r > best_rmsd:
            best, best_rmsd = c, r
    if best is None:
        return f'no conformer within {window} kcal/mol is more than {min_rmsd} A from the stored one'
    in_file_order = lambda cid: np.array([list(mol.GetConformer(cid).GetAtomPosition(int(r))) for r in mapping])
    return (in_file_order(cid_s), in_file_order(best), float(rmsd(cid_o, cid_s)), float(best_rmsd), float(energies[best] - e_s))


def kabsch_rmsd(p, q, reflect=False):
    """RMSD of two point sets after the best proper rotation of one onto the other; with
    `reflect`, after the best proper rotation of its mirror image."""
    p, q = p - p.mean(0), q - q.mean(0)
    if reflect:
        p = p * np.array([-1.0, 1.0, 1.0])
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt((((p @ rot) - q) ** 2).sum(1).mean()))


def frame_rmsds(p, q):
    """How two geometries of one molecule compare, three ways: as they sit in the trainer's
    frame, after the best proper rotation, and after the best rotation or reflection."""
    proper = kabsch_rmsd(p, q)
    return float(np.sqrt(((p - q) ** 2).sum(1).mean())), proper, min(proper, kabsch_rmsd(p, q, reflect=True))


def frame_state(in_frame, proper, either, tol=0.3):
    """'kept' when the two already coincide in the frame as well as any rigid motion could
    make them; else 'mirrored' when only a reflection superimposes them (the standardisation
    returned the other hand), else 'turned' (a rotation of the frame). Arrays in, array out."""
    in_frame, proper, either = (np.asarray(v, dtype=np.float64) for v in (in_frame, proper, either))
    return np.where(in_frame <= either + tol, 'kept', np.where(proper > either + tol, 'mirrored', 'turned'))


def rows_from(batch, row, geometries, standardise=True):
    """Conditions rows for new geometries of one stored molecule: fresh molecules (radius,
    mass and volume computed from the geometry) in the stored row's crystal slot, then
    standardised to the trainer's frame (None when that frame is not a fixed point).
    `standardise=False` leaves the geometries in the frame they are given in."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
    from mxtaltools.dataset_utils.utils import collate_data_list
    sl = slice(int(batch.ptr[row]), int(batch.ptr[row + 1]))
    ident = batch.identifier[row]
    items = []
    for pos in geometries:
        pos = torch.as_tensor(np.asarray(pos), dtype=batch.pos.dtype)
        # the stored radius is the farthest atom, hydrogens included, from the all-atom centroid; a single
        # molecule's mol_analysis would use heavy atoms only, and the radius is the unit of the cell latents
        radius = (pos - pos.mean(0)).norm(dim=1).max()
        mol = MolData(z=batch.z[sl].clone(), pos=pos, x=batch.x[sl].clone(), smiles=ident, identifier=ident,
                      radius=radius, do_mol_analysis=True)
        items.append(MolCrystalData(
            molecule=mol, sg_ind=int(batch.sg_ind[row]), z_prime=1, max_z_prime=1,
            cell_lengths=batch.cell_lengths[row].clone(), cell_angles=batch.cell_angles[row].clone(),
            aunit_centroid=batch.aunit_centroid[row].clone(), aunit_orientation=batch.aunit_orientation[row].clone(),
            aunit_handedness=float(batch.aunit_handedness[row]), do_box_analysis=True))
    out = collate_data_list(items)
    if not standardise:
        return out
    out.orient_molecule(mode='std')
    probe = out.clone()
    probe.orient_molecule(mode='std')
    if float((probe.pos - out.pos).abs().max()) > 1e-3:
        return None
    return out


def molecule_items(batch, rows, positions=None):
    """Bare molecules (atomic numbers and positions) of conditions rows, for the encoder."""
    from mxtaltools.dataset_utils.data_classes import MolData
    out = []
    for k, row in enumerate(rows):
        sl = slice(int(batch.ptr[row]), int(batch.ptr[row + 1]))
        pos = batch.pos[sl] if positions is None else positions[k]
        out.append(MolData(z=batch.z[sl].clone(), pos=torch.as_tensor(np.asarray(pos), dtype=batch.pos.dtype).clone(),
                           identifier=batch.identifier[row], do_mol_analysis=True))
    return out


@torch.no_grad()
def encode(encoder, molecules):
    """The frozen encoder's embedding of each molecule, as build_anchor_conditions.embed takes
    it: recentred on all atoms, one batch."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    chunk = collate_data_list([m.clone() for m in molecules])
    chunk.recenter_molecules(center_on_heavy_atoms=False)
    return encoder.encode(chunk).clone()


def figures(conf_path, atlas_path, fig_dir):
    """The conformer figures; reads the result file and the atlas, loads no model."""
    from eval.cond_panel import figstyle as fs
    C = torch.load(conf_path, weights_only=False)
    A = torch.load(atlas_path, weights_only=False)
    R = C['results']
    F = fs.Figures(fig_dir)
    get = lambda key: np.array([r[key] for r in R], dtype=np.float64)
    nn_dist = float(np.median(A['pair_dist_train']))
    kinds = (('J', 'jittered', fs.THIRD), ('S', 'relaxed, same conformer', fs.TRAIN), ('A', 'alternate conformer', fs.HELD))
    rng = np.random.default_rng(0)

    fig, axes = fs.panels(4, ncols=4, width=3.6, height=3.5)
    header = ['geometry', 'molecules', 'median shape RMSD (A)', 'median RMSD in the trainer frame (A)', 'frame kept (share)',
              'median conditioner distance from the stored condition', 'median excess (kT)', 'median excess minus stored (kT)',
              'median swap penalty (kT)', 'median density lift (nats)']
    rows = [['stored', len(R), 0.0, 0.0, 1.0, 0.0, round(float(np.median(get('excess_O'))), 2), 0.0, 0.0, 0.0],
            ['stored geometry, re-embedded in a batch of 200', len(R), 0.0, 0.0, 1.0, round(float(np.median(get('cond_dist_reembedded'))), 3), '', '', '', ''],
            ['stored geometry, embedded alone', len(R), 0.0, 0.0, 1.0, round(float(np.median(get('cond_dist_alone'))), 3), '', '', '', '']]
    points = [f'stored geometry re-embedded in a batch of 200: median conditioner distance {np.median(get("cond_dist_reembedded")):.2f} from the stored '
              f'condition; embedded alone: {np.median(get("cond_dist_alone")):.2f}; nearest training molecule: {nn_dist:.2f}']
    kept, state = {}, {}
    for k, label, colour in kinds:
        state[k] = frame_state(get(f'frame_rmsd_O{k}'), get(f'aligned_rmsd_O{k}'), get(f'shape_rmsd_O{k}'))
        kept[k] = state[k] == 'kept'
    # 1. how far the condition moves
    ax = axes[0]
    fs.style(ax)
    cols = [('in a batch', get('cond_dist_reembedded'), fs.MUTED), ('alone', get('cond_dist_alone'), fs.MUTED)] + \
           [(label.split(',')[0], get(f'cond_dist_O{k}'), colour) for k, label, colour in kinds]
    for x, (label, v, colour) in enumerate(cols):
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=colour, alpha=0.75, linewidths=0)
        ax.plot([x - 0.28, x + 0.28], [np.median(v)] * 2, color=fs.INK, linewidth=1.5)
    ax.axhline(nn_dist, color=fs.AXIS, linewidth=1)
    ax.text(len(cols) - 0.5, nn_dist, 'nearest training molecule ', ha='right', va='bottom', fontsize=7.5, color=fs.INK2)
    ax.set_xticks(np.arange(len(cols)), ['stored,\nin a batch', 'stored,\nalone', 'jittered', 'relaxed', 'alternate'], fontsize=7.5)
    ax.set_ylabel('conditioner distance from the stored condition')
    # 2. shape change against frame change
    ax = axes[1]
    fs.style(ax, grid='both')
    for k, label, colour in kinds:
        ax.scatter(get(f'shape_rmsd_O{k}'), get(f'frame_rmsd_O{k}'), s=16, color=colour, alpha=0.8, linewidths=0, label=label)
    lim = max(ax.get_xlim()[1], 0.5)
    ax.plot([0, lim], [0, lim], color=fs.AXIS, linewidth=1)
    ax.set_xlabel('heavy-atom RMSD after the best rotation or reflection (A)')
    ax.set_ylabel('heavy-atom RMSD in the trainer\'s frame (A)')
    ax.legend(loc='upper left', fontsize=7.5)
    # 3. condition shift against the RMSD the model sees
    ax = axes[2]
    fs.style(ax, grid='both')
    for k, label, colour in kinds:
        ax.scatter(get(f'frame_rmsd_O{k}'), get(f'cond_dist_O{k}'), s=16, color=colour, alpha=0.8, linewidths=0)
    ax.axhline(nn_dist, color=fs.AXIS, linewidth=1)
    ax.set_xlabel('heavy-atom RMSD in the trainer\'s frame (A)')
    ax.set_ylabel('conditioner distance from the stored condition')
    # 4. sample quality
    ax = axes[3]
    fs.style(ax)
    for x, (k, label, colour) in enumerate((('O', 'stored', fs.MUTED),) + kinds):
        v = get(f'excess_{k}')
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=colour, alpha=0.75, linewidths=0)
        ax.plot([x - 0.28, x + 0.28], [np.median(v)] * 2, color=fs.INK, linewidth=1.5)
    ax.set_xticks(np.arange(4), ['stored', 'jittered', 'relaxed', 'alternate'], fontsize=8)
    ax.set_ylabel('median excess of the draws (kT above the\nstored conformer\'s best search minimum)')
    for k, label, colour in kinds:
        rows.append([label, len(R), round(float(np.median(get(f'shape_rmsd_O{k}'))), 3), round(float(np.median(get(f'frame_rmsd_O{k}'))), 3),
                     round(float(kept[k].mean()), 2), round(float(np.median(get(f'cond_dist_O{k}'))), 2), round(float(np.median(get(f'excess_{k}'))), 2),
                     round(float(np.median(get(f'excess_{k}') - get('excess_O'))), 2), round(float(np.median(get(f'swap_penalty_O{k}'))), 2),
                     round(float(np.median(get(f'lift_O{k}'))), 2)])
        points.append(f'{label}: shape RMSD {rows[-1][2]:.2f} A, in the trainer\'s frame {rows[-1][3]:.2f} A (frame kept for {kept[k].mean():.0%}); '
                      f'conditioner distance {rows[-1][5]:.2f}; excess {rows[-1][7]:+.1f} kT against the stored conformer; swap penalty {rows[-1][8]:.1f} kT; lift {rows[-1][9]:.1f} nats')
    F.save(fig, 'conformers', 'Other geometries of a training molecule as conditions the model never saw',
           f'{len(R)} flexible training molecules without stereocentres, each drawn under four conditions: its stored geometry, that geometry with '
           f'{C["args"]["jitter"]} A of noise per coordinate, the same conformer after an MMFF relaxation, and an alternate conformer (MMFF within '
           f'{C["args"]["window"]} kcal/mol of the lowest, at least {C["args"]["min_rmsd"]} A away). One point per molecule; bars are medians. The first '
           f'panel also shows the stored geometry re-embedded by the encoder in a batch of {C["args"]["embed_batch"]} molecules and alone. The grey '
           f'line is the median distance from a training molecule to its nearest training neighbour. Excess is measured against the STORED '
           f'conformer\'s best search minimum for every geometry.', (header, rows), points)

    # frame kept against frame turned over
    fig, axes = fs.panels(3, ncols=3, width=4.0, height=3.4)
    header, rows, points = ['geometry', 'frame', 'molecules', 'median conditioner distance', 'median excess minus stored (kT)',
                            'median swap penalty (kT)', 'median density lift (nats)'], [], []
    for ax, (key, label) in zip(axes, (('cond_dist_O', 'conditioner distance from the stored condition'),
                                       ('excess_', 'excess minus the stored geometry\'s (kT)'), ('swap_penalty_O', 'swap penalty (kT)'))):
        fs.style(ax)
        x = 0
        ticks = []
        for k, lab, colour in kinds:
            for name, mask in (('kept', kept[k]), ('not kept', ~kept[k])):
                v = get(f'{key}{k}') - (get('excess_O') if key == 'excess_' else 0.0)
                v = v[mask]
                if len(v):
                    ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=colour, alpha=0.8 if name == 'kept' else 0.45, linewidths=0)
                    ax.plot([x - 0.28, x + 0.28], [np.median(v)] * 2, color=fs.INK, linewidth=1.5)
                ticks.append(f'{lab.split(",")[0].split(" ")[0]}\n{name}')
                x += 1
        ax.axhline(0, color=fs.AXIS, linewidth=1)
        ax.set_xticks(np.arange(len(ticks)), ticks, fontsize=7.5)
        ax.set_ylabel(label)
        if key != 'cond_dist_O':
            ax.set_yscale('symlog', linthresh=10.0)
    for k, lab, colour in kinds:
        for name in ('kept', 'turned', 'mirrored'):
            mask = state[k] == name
            if mask.sum():
                rows.append([lab, name, int(mask.sum()), round(float(np.median(get(f'cond_dist_O{k}')[mask])), 2),
                             round(float(np.median((get(f'excess_{k}') - get('excess_O'))[mask])), 2),
                             round(float(np.median(get(f'swap_penalty_O{k}')[mask])), 2), round(float(np.median(get(f'lift_O{k}')[mask])), 2)])
                points.append(f'{lab}, frame {name} ({int(mask.sum())} molecules): conditioner distance {rows[-1][3]:.2f}, excess {rows[-1][4]:+.1f} kT, '
                              f'swap penalty {rows[-1][5]:.1f} kT')
    F.save(fig, 'conformer_frames', 'The same comparison, split by whether the trainer\'s frame survived the change of geometry',
           'A frame counts as kept when the heavy-atom RMSD of the two geometries as they sit in the trainer\'s frame is within 0.3 A of '
           'their RMSD after the best rotation or reflection; otherwise it is mirrored when only a reflection superimposes them (the '
           'standardisation returned the other hand of the conformer) and turned when a rotation does. One point per molecule; bars are medians; the second and '
           'third axes are linear within +-10 and logarithmic beyond.', (header, rows), points)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--atlas', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--figures', default=None, help='figure directory: draw the conformer figures from the result file in --out and stop')
    ap.add_argument('--encoder', default=DEFAULT_ENCODER)
    ap.add_argument('--n-molecules', type=int, default=40)
    ap.add_argument('--min-rot', type=int, default=2, help='rotatable bonds a molecule needs to be tried')
    ap.add_argument('--n-confs', type=int, default=30)
    ap.add_argument('--window', type=float, default=3.0, help='MMFF kcal/mol above the lowest conformer')
    ap.add_argument('--min-rmsd', type=float, default=0.5)
    ap.add_argument('--jitter', type=float, default=0.02, help='angstrom, per coordinate, for the jittered control')
    ap.add_argument('--embed-batch', type=int, default=200, help='molecules per encoder batch, as the conditions were built')
    ap.add_argument('--draws', type=int, default=128)
    ap.add_argument('--lift-n', type=int, default=24)
    ap.add_argument('--lift-k', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    if args.figures:
        import glob
        figures(glob.glob(os.path.join(args.out, 'conformers_step*.pt'))[0], args.atlas, args.figures)
        return
    assert not torch.cuda.is_available(), 'CPU only: set CUDA_VISIBLE_DEVICES=-1'
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    from mxtaltools.common.training_utils import load_molecule_autoencoder

    atlas = torch.load(args.atlas, weights_only=False)
    run = load_run(args.checkpoint, args.config, device='cpu')
    encoder = load_molecule_autoencoder(args.encoder, 'cpu')
    batch = run.conditions
    row_of = {ident: i for i, ident in enumerate(batch.identifier)}
    rng = np.random.default_rng(args.seed)
    # THE ENCODER'S OUTPUT DEPENDS ON ITS BATCH (it carries a batch-level normalisation): a
    # molecule embedded alone lands several percent from the embedding it was given in the
    # builder's batches. New geometries are therefore embedded among stored molecules, in a
    # batch of the builder's size, and the stored geometry re-embedded beside them is the gate.
    filler = molecule_items(batch, rng.choice(batch.num_graphs, args.embed_batch - 4, replace=False).tolist())
    pool = np.flatnonzero(atlas['is_random_train'] & (atlas['descriptors']['n_rot'] >= args.min_rot))
    results, skipped = [], {}
    for m in rng.permutation(pool):
        if len(results) == args.n_molecules:
            break
        ident = atlas['identifier'][m]
        i = row_of[ident]
        sl = slice(int(batch.ptr[i]), int(batch.ptr[i + 1]))
        stored_pos = batch.pos[sl].numpy()
        found = other_conformers(ident, batch.z[sl].numpy(), stored_pos.astype(np.float64), args.n_confs, args.window,
                                 args.min_rmsd, args.seed)
        if isinstance(found, str):
            skipped[found] = skipped.get(found, 0) + 1
            continue
        pos_s, pos_a, rmsd_s, rmsd_a, de_a = found
        # rows: the stored condition, S, A, J, and two re-embeddings of the stored geometry (in a batch, alone)
        pos_j = stored_pos + rng.normal(0.0, args.jitter, stored_pos.shape)
        rows = rows_from(batch, i, [stored_pos, pos_s, pos_a, pos_j, stored_pos, stored_pos])
        if rows is None:
            skipped['frame not a fixed point'] = skipped.get('frame not a fixed point', 0) + 1
            continue
        assert abs(float(rows.radius[0]) - float(batch.radius[i])) < 1e-3, f'{ident}: rebuilt radius differs from the stored one'
        framed = [rows.pos[int(rows.ptr[k]):int(rows.ptr[k + 1])].numpy() for k in range(4)]
        in_batch = encode(encoder, molecule_items(batch, [i] * 4, framed) + filler)[:4]
        alone = encode(encoder, molecule_items(batch, [i], [framed[0]]))[0]
        stored_emb = batch.embedding[i]
        rel = lambda e: float((e - stored_emb).norm() / stored_emb.norm())
        gate = rel(in_batch[0])
        # bound set by the encoder's batch dependence: at most 4.4% over 800 stored molecules re-embedded in batches of 200
        assert gate < 0.06, f'{ident}: the stored geometry re-embedded in a batch is {gate:.3f} (relative) from its stored embedding'
        rows.add_graph_attr(torch.stack([stored_emb, in_batch[1], in_batch[2], in_batch[3], in_batch[0], alone]), 'embedding')
        rows.add_graph_attr(torch.full((rows.num_graphs,), run.registry[ident], dtype=torch.long), 'mol_id')
        cond = conditioner_coords(run, rows).numpy()
        d = {}
        for k, name in enumerate('OSAJ'):
            one = rows.subsample_new_batch(torch.full((args.draws,), k, dtype=torch.long))
            out = draw(run, one, batch_size=args.draws, seed=args.seed + 7 * len(results) + k)
            d[name] = {'terminal': out['terminal_raw'], 'excess': seed_excess(run, out['sample_batch'], out['condition_id']),
                       'latent': fold(out['sample_batch'].latent_params().detach().cpu().numpy().astype(np.float64)),
                       'mol_energy': out['mol_energy'].numpy(), 'packing': out['packing_coeff'].numpy()}
        r = {'identifier': ident, 'atlas_index': int(m), 'n_rot': int(atlas['descriptors']['n_rot'][m]), 'rmsd_stored_to_S': rmsd_s,
             'rmsd_S_to_A': rmsd_a, 'mmff_A_minus_S_kcal': de_a, 'radius': rows.radius[:4].tolist(),
             'enc_rel_reembedded': gate, 'enc_rel_alone': rel(alone),
             'cond_dist_reembedded': float(np.linalg.norm(cond[0] - cond[4])), 'cond_dist_alone': float(np.linalg.norm(cond[0] - cond[5]))}
        heavy = batch.z[sl].numpy() > 1
        for other in 'SAJ':
            a, b = d['O'], d[other]
            k = 'OSAJ'.index(other)
            r[f'frame_rmsd_O{other}'], r[f'aligned_rmsd_O{other}'], r[f'shape_rmsd_O{other}'] = frame_rmsds(
                framed[k][heavy].astype(np.float64), framed[0][heavy].astype(np.float64))
            r[f'cond_dist_O{other}'] = float(np.linalg.norm(cond[0] - cond[k]))
            r[f'enc_rel_O{other}'] = rel(in_batch[k])
            r[f'energy_distance_O{other}'] = energy_distance(a['latent'], b['latent'])
            swaps = (score_under(run, rows.subsample_new_batch(torch.full((args.draws,), k, dtype=torch.long)), a['terminal']),
                     score_under(run, rows.subsample_new_batch(torch.full((args.draws,), 0, dtype=torch.long)), b['terminal']))
            r[f'swap_penalty_O{other}'] = float(0.5 * ((np.median(swaps[0]) - np.median(b['excess'])) + (np.median(swaps[1]) - np.median(a['excess']))))
            two = rows.subsample_new_batch(torch.tensor([0, k]))
            la, lb, _ = density_lift(run, two, torch.cat([a['terminal'][:args.lift_n], b['terminal'][:args.lift_n]]), args.lift_n, args.lift_k)
            r[f'lift_O{other}'] = float(0.5 * (la + lb))
        half = args.draws // 2
        r['energy_distance_floor'] = energy_distance(d['O']['latent'][:half], d['O']['latent'][half:])
        for name in 'OSAJ':
            r[f'excess_{name}'] = float(np.median(d[name]['excess']))
            r[f'mol_energy_{name}'] = float(np.median(d[name]['mol_energy']))
            r[f'packing_{name}'] = float(np.median(d[name]['packing']))
        results.append(r)
        state = {k: str(frame_state(r[f'frame_rmsd_O{k}'], r[f'aligned_rmsd_O{k}'], r[f'shape_rmsd_O{k}'])) for k in 'JSA'}
        print(f'{len(results):3d} {ident:22s} shape rmsd and frame J {r["shape_rmsd_OJ"]:.2f} {state["J"]} S {r["shape_rmsd_OS"]:.2f} {state["S"]} '
              f'A {r["shape_rmsd_OA"]:.2f} {state["A"]} | conditioner: re-embedded {r["cond_dist_reembedded"]:.2f} '
              f'alone {r["cond_dist_alone"]:.2f} J {r["cond_dist_OJ"]:.2f} S {r["cond_dist_OS"]:.2f} A {r["cond_dist_OA"]:.2f} | excess O {r["excess_O"]:.1f} '
              f'J {r["excess_J"]:.1f} S {r["excess_S"]:.1f} A {r["excess_A"]:.1f} | swap J {r["swap_penalty_OJ"]:.1f} S {r["swap_penalty_OS"]:.1f} '
              f'A {r["swap_penalty_OA"]:.1f} | lift J {r["lift_OJ"]:.1f} S {r["lift_OS"]:.1f} A {r["lift_OA"]:.1f}', flush=True)
        torch.save({'results': results, 'skipped': skipped, 'step': run.step, 'args': vars(args)},
                   os.path.join(args.out, f'conformers_step{run.step}.pt'))
    print(f'{len(results)} molecules; skipped {skipped}', flush=True)


if __name__ == '__main__':
    main()
