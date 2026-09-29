"""Per-molecule ring shapes: one molecule's own relaxed ring conformers, measured in its chart.

WHAT IT IS FOR. ``ConformerTorsions.sample_prior_states`` draws a ring block from a type-level
table looked up by (ring signature, block DoF count) -- ``prior.ring_modes`` or ``prior.rings``,
fitted by build_ring_banks.py on bare saturated monocycles -- or, where no table resolves, the
system is fused, bridged or spiro, or the ring is aromatic, holds it at its reference shape plus
jitter. On a 240-molecule QM9 sample 5 of 193 ring systems reached a table
(artifacts/conformer_coverage_2026-09-29/report.md). ``ring_shapes`` measures THIS molecule's own
ring conformers instead, and ``sample_prior_states(..., ring_shapes=...)`` (or
``conformer_prior_draw.draw_member_prior(..., ring_shapes=...)``) draws one of them per block and
per draw, uniformly.

HOW, per molecule:
  1. embed ``n_embed`` conformers of the member's molecule -- its condition's explicit-H parse,
     ``build_conformer_references.condition_stereo``'s ``template``, whose atoms index
     ``member.mol`` -- by RDKit distance geometry with the knowledge terms off
     (``useExpTorsionAnglePrefs``, ``useBasicKnowledge``, ``useSmallRingTorsions`` and
     ``useMacrocycleTorsions`` False) and ``enforceChirality``, seeded by the caller
     (``build_conformer_references.molecule_seed`` is the repository's per-molecule seed);
  2. relax each with RDKit MMFF94, ``maxIters`` 2000 as the member's own reference is;
  3. drop an embedding whose configuration at the condition's specified stereo elements is not
     the condition's (``condition_stereo`` / ``stereo_labels``);
  4. measure each in the member's own tree (``_state_of_positions``, dummy-frame rows against
     their dummies) and read the chart's DoF back off the state (``dof_from_state``), so a shape
     holds what a state of this member's tier can express;
  5. per ring block: drop an embedding whose written theta rows leave the member's box (the draw
     would clip them and the ring would open), check the rest on trial draws, and deduplicate.

Aromatic blocks get no shapes: they stay held planar.

WHICH ROWS A SHAPE CARRIES. A block's ``order`` holds the rows all of whose atoms lie in the ring
system. Where an atom outside the system sits in a ring atom's frame -- the tree's root region, and
the atom a ring the tree enters from outside hangs from -- the rows naming it (``extra``) set part
of the ring's internal geometry too: a pucker written into ``order`` alone does not close the ring.
A shape therefore carries the block's ``order`` and, after it, the extras that change a distance
between two atoms of the system (``internal_extra_rows``: each is perturbed at the reference and
the system's distances compared). The other extras move the whole system rigidly and the draw
holds them, as without shapes. Where the tree enters the system from outside, the shape also
carries the rest of the sibling group of the ring's rotation about its attaching bond
(``ring_block_info['rotation_rows']``): the entry atom's other children, so that all of that
atom's substituents sit as the embedding put them. The draw writes a shape's theta and phi rows,
leaves r on the thermal path, and turns that group's rows by the rotation it draws.

TRIAL DRAWS. A shape passes only if, drawn through ``sample_prior_states`` with every other ring
block at its reference shape and the molecule's other rows as the prior draws them,
``TRIAL_DRAWS`` draws per candidate, every trial draw that took it closes the block's rings to
``CLOSURE_TOL`` and, with the stereo lock on (``stereo_coeff`` > 0), puts no locked element on the
wrong side. The second is the lock's labelling: the lock pins the reference's labelled parity at
every four-coordinate atom, and an embedding may realise a ring whose exocyclic neighbours sit
in the other labelling (two graph-equivalent ring directions, say), which the shape's rows would
carry into the draw. Such an embedding is dropped, not relabelled.

DEDUPLICATION. Per block, greedy in ascending MMFF94 energy of the embedding: an embedding joins
the first kept shape whose phi rows all lie within ``DEDUP_DEG`` of its own (circular difference).
The rows of the rotation's sibling group enter as offsets from its first ring row, since the draw
replaces their common rotation. ``count`` records how many embeddings each shape stands for.
``mmff_energy`` is the RDKit MMFF94 energy of the whole embedding that represents a shape, in that
embedding's own conformation of the rest of the molecule: an ordering, not the energy of a draw.

WHEN EVERY EMBEDDING FAILS. A block with no shape left -- nothing embedded, or every embedding
dropped -- gets an empty ``BlockShapes`` whose ``reason`` says why, and ``sample_prior_states``
draws that block exactly as it does without ring shapes (its type-level table, or the hold). A
list in which every block is empty draws bit for bit as ``ring_shapes=None``.

Not available at a collective tier (``torsion``): the chart there has no row-wise inverse, and
its ring DoF are frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

#: degrees. Two embeddings are one shape when every phi row of the dedup key differs by at most
#: this. MEASURED 2026-09-29, 64 embeddings each of cyclohexane, cyclopentane, proline,
#: trans-decalin, hexylcyclohexane, a 1,4-dioxan-2-one and methylcyclohexane (conformer_mk
#: member): an embedding sat 0 to 9.1 deg from the shape it joined (0.00 for cyclohexane and
#: trans-decalin), and the two closest kept shapes of a molecule were 10.5 to 56 deg apart
#: (cyclopentane's barrier-free pseudorotation 12.6). A same-shape pair split by it only reweights
#: the uniform choice between shapes.
DEDUP_DEG = 10.0
#: Angstrom: the largest closure-bond error a trial draw of an accepted shape may show. MEASURED
#: 2026-09-29 (conformer_mk member, 1000 draws): median closure error 0.02-0.06 A for held and
#: shaped draws of proline, bicyclopropyl, cyclohexylcyclopropane, hexylcyclohexane,
#: trans-decalin and norbornane; 0.65 A for hexylcyclohexane's shapes written into ``order`` alone.
CLOSURE_TOL = 0.25
#: trial draws per candidate embedding, in one sample_prior_states call per block
TRIAL_DRAWS = 16
#: seconds per embedding (RDKit EmbedParameters.timeout)
EMBED_TIMEOUT_S = 10
#: RDKit MMFF94 iterations per embedding, as ConformerTorsions relaxes its reference
MMFF_MAX_ITERS = 2000


@dataclass
class BlockShapes:
    """The distinct shapes of one ring block of one molecule, in its own chart.

    ``rows`` is ``[(kind, j)]``: the block's ``order`` (its first ``n_order`` entries), then the
    extras that set the ring's internal geometry, then the rest of the rotation's sibling group
    (``info['rotation_group']``). ``values [n_shapes, len(rows)]`` holds each
    shape's measured DoF there (r in Angstrom, theta and phi in radians), ascending in
    ``mmff_energy`` (RDKit MMFF94, kcal/mol, of the whole embedding that represents the shape);
    ``count`` is the embeddings each shape stands for. ``reason`` is None when there is at least
    one shape, otherwise 'aromatic', 'no_embedding' or 'none_kept'. ``info`` counts what was
    dropped and why.
    """
    block: int
    rows: list
    n_order: int
    values: np.ndarray
    mmff_energy: np.ndarray
    count: np.ndarray
    reason: Optional[str] = None
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.values))


def internal_extra_rows(member, extra, atoms, step: float = 0.3, tol: float = 1e-6) -> list:
    """The theta and phi rows of ``extra`` that change a distance between two ``atoms``.

    Each row is moved by ``step`` radians from the member's reference in its own copy of the
    molecule and the ring system's interatomic distances are compared with the reference's; a
    row that moves the system only rigidly (the bond into the system and the angle placing its
    first atom; the dihedral placing that atom is not an extra) changes none. r rows are never
    returned: the draw keeps r on the thermal path.
    """
    cand = [(str(k), int(j)) for k, j in extra if k != 'r']
    if not cand:
        return []
    k = int(member.ndim)
    with torch.no_grad():
        r, th, ph = member.dof_from_state(torch.zeros(1, k, dtype=member.dtype,
                                                      device=member.device))
        b = len(cand)
        R, TH, PH = r.repeat(b, 1), th.repeat(b, 1).clone(), ph.repeat(b, 1).clone()
        for i, (kind, j) in enumerate(cand):
            (TH if kind == 'theta' else PH)[i, j] += step
        tree, _ = member._batch(b)
        pos = member._build(tree, R, TH, PH, b).reshape(b, -1, 3)
        tree1, _ = member._batch(1)
        pos0 = member._build(tree1, r, th, ph, 1).reshape(1, -1, 3)
    at = torch.as_tensor(sorted(int(a) for a in atoms), dtype=torch.long, device=pos.device)
    d = torch.cdist(pos[:, at], pos[:, at])
    d0 = torch.cdist(pos0[:, at], pos0[:, at])
    moved = ((d - d0).abs().amax(dim=(1, 2)) > tol).cpu().numpy()
    return [kj for kj, mv in zip(cand, moved) if mv]


def _embed(template, n: int, seed: int, timeout_s: int):
    """``(positions [m, N, 3] RDKit order, mmff_energy [m], n_not_converged)``."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.Mol(template)
    p = AllChem.ETKDGv3()
    # THE KNOWLEDGE TERMS OFF: distance geometry alone, so the embedder does not steer the ring
    # toward the torsions its tables prefer. useMacrocycleTorsions acts on rings of nine or more
    # atoms only and is switched off with the rest.
    p.useExpTorsionAnglePrefs = False
    p.useBasicKnowledge = False
    p.useSmallRingTorsions = False
    p.useMacrocycleTorsions = False
    p.enforceChirality = True
    p.randomSeed = int(seed)
    p.timeout = int(timeout_s)
    p.numThreads = 1
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=int(n), params=p))
    if not cids:
        return np.zeros((0, m.GetNumAtoms(), 3)), np.zeros(0), 0
    res = AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=1, maxIters=MMFF_MAX_ITERS)
    pos = np.stack([m.GetConformer(c).GetPositions() for c in cids])
    e = np.array([res[i][1] for i in range(len(cids))], dtype=np.float64)
    n_nc = int(sum(int(res[i][0]) != 0 for i in range(len(cids))))
    return pos, e, n_nc


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _dedup_key(rows, values, group, rotation) -> np.ndarray:
    """``[m, n_phi]``: a block's phi rows, those of the rotation's sibling group (``group``) as
    offsets from its first ring row (the first of ``rotation``)."""
    phi = [c for c, (k, _) in enumerate(rows) if k == 'phi']
    key = values[:, phi].copy()
    grp = [i for i, c in enumerate(phi) if rows[c][1] in group]
    lead = [i for i, c in enumerate(phi) if rows[c][1] in rotation]
    if grp:
        key[:, grp] = _wrap(key[:, grp] - key[:, [lead[0]]])
    return key


def _dedup(key, energy, tol_rad):
    """Greedy in ascending energy: ``(representatives, count)``, representatives ascending."""
    order = np.argsort(energy, kind='stable')
    reps, count = [], []
    for i in order:
        for r, rep in enumerate(reps):
            if np.all(np.abs(_wrap(key[i] - key[rep])) <= tol_rad):
                count[r] += 1
                break
        else:
            reps.append(int(i))
            count.append(1)
    return np.asarray(reps, dtype=np.int64), np.asarray(count, dtype=np.int64)


def _trial(member, prior, blocks, bk, shp, seed, fixed):
    """Per candidate shape of block ``bk``: True if every trial draw that took it closes the
    block's rings to CLOSURE_TOL and, with the lock on, puts no locked element on the wrong
    side. A candidate no trial draw took is not passed. Every other block draws from
    ``fixed[i]`` (its reference shape, or None where aromatic), so a failure is this block's."""
    n_c = len(shp)
    n = TRIAL_DRAWS * n_c
    trial = list(fixed)
    trial[bk] = shp
    x, st = member.sample_prior_states(prior, n, np.random.default_rng(seed), report=False,
                                       ring_shapes=trial)
    pick = st['ring_shapes'][bk]['pick']
    open_, wrong = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    with torch.no_grad():
        pos = member.build_positions(x).reshape(n, -1, 3)
        atoms = set(shp.info['atoms'])
        br = np.asarray(member.spec.broken_bond_index).reshape(-1, 2)
        sel = [i for i, (a, b) in enumerate(br) if int(a) in atoms and int(b) in atoms]
        if sel:
            _, ff = member._batch(1)
            r0 = ff.closure_r0.detach().cpu().numpy()[sel]
            e = torch.as_tensor(br[sel], dtype=torch.long, device=pos.device)
            cl = (pos[:, e[:, 0]] - pos[:, e[:, 1]]).norm(dim=-1).cpu().numpy()
            open_ = np.abs(cl - r0).max(axis=1) > CLOSURE_TOL
        if member.stereo_coeff > 0.0 and member.stereo.n:
            _, _, sign, _ = member.stereo.tensors(pos.device, pos.dtype)
            wrong = (sign * member.stereo.values(pos) <= 0.0).any(dim=1).cpu().numpy()
    per = lambda m: np.bincount(pick[m], minlength=n_c) > 0
    seen, opened, locked = per(np.ones(n, dtype=bool)), per(open_), per(wrong)
    shp.info['n_trial_open'] = int(opened.sum())
    shp.info['n_trial_locked_out'] = int(locked.sum())
    shp.info['n_trial_unseen'] = int((~seen).sum())
    return seen & ~opened & ~locked


def ring_shapes(member, prior, n_embed: int = 64, seed: int = 0,
                dedup_deg: float = DEDUP_DEG) -> List[BlockShapes]:
    """This molecule's distinct ring shapes per ring block, aligned with ``member.ring_blocks``.

    ``member`` is a ``ConformerTorsions`` (one molecule's own chart, below ``torsion``); ``prior``
    the fitted InternalPrior the member draws with (``ring_blocks`` and the trial draws read it);
    ``seed`` the RDKit embedding seed and the trial draws' generator seed. Returns one
    ``BlockShapes`` per block, empty (``reason`` set) where the block has none. See the module
    docstring for the steps and what each drop means.
    """
    if member.collective:
        raise ValueError(f'{member.smiles}: ring shapes need a row-wise chart, and level '
                         f'{member.level!r} has collective columns (ring DoF are frozen there)')
    import build_conformer_references as bcr
    from mxtaltools.conformers.builder import collate

    blocks = member.ring_blocks(prior)
    binfo = [dict(r) for r in member.ring_block_info]
    groups = member.torsion_groups()
    out: List[Optional[BlockShapes]] = [None] * len(blocks)
    todo = []
    for bk, ((order, _, extra), bi) in enumerate(zip(blocks, binfo)):
        order = [(str(k), int(j)) for k, j in order]
        rows = order + internal_extra_rows(member, extra, bi['atoms'])
        n_int = len(rows) - len(order)
        # the rotation's whole sibling group: the entry atom's other children as well, so the
        # draw places all of that atom's substituents as the embedding did
        rot = set(int(j) for j in bi['rotation_rows'])
        group = sorted({int(j) for g in groups if set(g) & rot for j in g})
        rows += [('phi', j) for j in group if ('phi', j) not in rows]
        info = dict(atoms=list(bi['atoms']), rotation_rows=sorted(rot), rotation_group=group,
                    n_extra=len(extra), n_internal_extra=n_int)
        out[bk] = BlockShapes(bk, rows, len(order), np.zeros((0, len(rows))), np.zeros(0),
                              np.zeros(0, np.int64), info=info)
        if bi['aromatic']:
            out[bk].reason = 'aromatic'
        else:
            todo.append(bk)
    if not todo:
        return out

    pin = bcr.condition_stereo(member)
    pos, energy, n_nc = _embed(pin['template'], n_embed, seed, EMBED_TIMEOUT_S)
    mol_info = dict(n_embed=int(n_embed), seed=int(seed), n_embedded=int(len(pos)),
                    n_mmff_not_converged=n_nc, n_other_stereo=0)
    keep = np.ones(len(pos), dtype=bool)
    xs = torch.zeros(0, int(member.ndim), dtype=member.dtype)
    if len(pos):
        tree = collate([member.spec], device=member.device)
        xs = torch.cat([bcr._state_of_positions(member, tree, rd) for rd in pos])
        if bcr._pinned(pin):
            labels = bcr.stereo_labels(member, xs, pin)
            keep = np.array([lab == pin['target'] for lab in labels], dtype=bool)
            mol_info['n_other_stereo'] = int((~keep).sum())
    with torch.no_grad():
        r, th, ph = member.dof_from_state(xs)
        ref = torch.cat(member.dof_from_state(torch.zeros(1, int(member.ndim), dtype=member.dtype,
                                                          device=member.device)), dim=1)
    dof = torch.cat([r, th, ph], dim=1).cpu().numpy()
    ref = ref.cpu().numpy()
    # state column of each driven DoF row, for the box check on the written theta rows
    col_of = {int(g): c for c, g in enumerate(member._sel_rows.cpu().numpy())}
    x_np = xs.detach().cpu().numpy()
    # the trial draws hold every block but the one under test at its reference shape
    fixed = [None if b.reason == 'aromatic' else
             BlockShapes(b.block, b.rows, b.n_order,
                         ref[:, [member._global_row(k, j) for k, j in b.rows]], np.zeros(1),
                         np.ones(1, np.int64), info=b.info)
             for b in out]

    for bk in todo:
        shp = out[bk]
        shp.info.update(mol_info)
        if not len(pos):
            shp.reason = 'no_embedding'
            continue
        g = [member._global_row(k, j) for k, j in shp.rows]
        vals = dof[:, g]
        ok = keep.copy()
        th_cols = [col_of[gr] for (k, _), gr in zip(shp.rows, g)
                   if k == 'theta' and gr in col_of and member._free_block[col_of[gr]] == 1]
        out_box = (np.abs(x_np[:, th_cols]) > 1.0).any(axis=1) if th_cols else np.zeros(len(ok), bool)
        shp.info['n_outside_box'] = int((ok & out_box).sum())
        ok &= ~out_box
        cand = np.flatnonzero(ok)
        shp.info['n_failed_trial'] = 0
        if len(cand):
            trial = BlockShapes(bk, shp.rows, shp.n_order, vals[cand], energy[cand],
                                np.ones(len(cand), np.int64), info=shp.info)
            passed = _trial(member, prior, blocks, bk, trial, seed, fixed)
            shp.info['n_failed_trial'] = int((~passed).sum())
            cand = cand[passed]
        shp.info['n_candidates'] = int(len(cand))
        if not len(cand):
            shp.reason = 'none_kept'
            continue
        key = _dedup_key(shp.rows, vals[cand], set(shp.info['rotation_group']),
                         shp.info['rotation_rows'])
        reps, count = _dedup(key, energy[cand], np.radians(dedup_deg))
        shp.values = vals[cand[reps]]
        shp.mmff_energy = energy[cand[reps]]
        shp.count = count
        shp.info['dedup_deg'] = float(dedup_deg)
    return out
