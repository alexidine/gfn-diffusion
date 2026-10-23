"""Offline QM9 CONFORMER DATABASE: the distinct low-energy relaxed conformers of every condition
of the conformer set builder's universe, one resumable file per shard of a CPU array job.

    python build_conformer_database.py build --source configs/conformer_db_sep29/qm9_index.tsv.gz \\
        --prior conformer_prior_v2.pt --out-dir OUT --shard 0 --n-shards 400
    python build_conformer_database.py summarize OUT

A ROW is one distinct relaxed conformer of one CONDITION: relaxed in the member's own chart with
the run's force field, and within ``--window-kt`` kT (default 10) of that condition's lowest
found conformer, kT being the member's temperature (1 kcal/mol at configs/conformer_mk.yaml).
Only the distinct relaxed shapes are stored; the spread about each comes later, from the run's
anchor noising.

THE UNIVERSE AND ITS CONDITIONS ARE THE SET BUILDER'S, by calling it rather than restating it,
so a later rung's conditions file finds its rows by identifier: ``build_conformer_set.py``'s
``read_source``, ``pool_rows`` (``--pool-need``, default 0: no encoder pool is excluded and the
universe is all of QM9, the set builder's default since the owner's 2026-09-29 decision),
``plan_split`` (index_min = the pool's end, 0 without one; BOTH sides, held-out keys marked) and
``build_molecule`` (the pick plus its mirror when chiral, each verified by ``build_member``),
under the set builder's own
argument defaults for the split salt, the held-out permille, the stereo salt and the per-molecule
cap (``set_builder_defaults``), with the member kwargs ``energy_kwargs_from_config`` reads off
``--config`` (the committed configs/conformer_mk.yaml).

SHARDS. The universe is every key of both sides in one order, ``(split hash, key)``, in which
each side keeps its own walk order; shard k of N takes ``universe[k::N]`` (``ASSIGNMENT_RULE``),
and ``--max-molecules M`` keeps its first M (the pilot).

PER CONDITION (``search_condition``), the measured recipe of
artifacts/conformer_coverage_2026-09-29/coverage.py with the prior's ring handling replaced:
  1. ``energies/ring_shapes.py::ring_shapes(member, prior, seed=molecule_seed(identifier))``;
  2. batches of ``--batch`` draws, ``draw_member_prior(..., ring_shapes=shapes)``, one numpy
     generator per (condition seed, batch index);
  3. ``relax``: ``prior_baselines.descend`` for ``--steps``, then the starts whose best energy
     still fell by more than ``--fall-rate`` kcal/mol per step over the last ``--fall-window``
     steps of that descent are continued from their best state, ``--steps`` more, at most
     ``--rounds`` times;
  4. ``screen``: each start excluded under the first reason of ``EXCLUSION_REASONS`` that
     applies -- a non-finite energy, still falling after the last round, a non-periodic state
     column on the box wall, a locked element of the stereo lock on the wrong side of its
     reference sign, or (RDKit 3D perception) another configuration than the condition's at an
     element its SMILES pins -- and every kept start scored in the currency the run expects on
     a stored row, raw potential at T = 1 (``conformer_data.bake_energies``);
  5. ``BasinSet``: heavy-atom symmetric RMSD < ``RMSD_TOL`` AND |dE| < ``DE_TOL``, greedy in
     ascending energy within a batch, energy-gated (RMSD only against representatives within
     ``DE_TOL``), no reflection (mirror images are distinct basins); a basin keeps its
     lowest-energy member as representative and counts its hits;
  6. STOP (``stop_after``) once at least ``--min-batches`` batches have run and the last
     ``--stop-streak`` CONSECUTIVE batches each founded no basin within the window of the
     current best; at most ``--max-batches`` (the termination bound). The reason is recorded.

THE FILE (``shard_KKKK_of_NNNN.pt``, ``torch.save``): ``header`` (provenance), ``header_hash``,
``run_hash``, ``assigned`` keys, ``keys`` (one record per processed key, each with its
conditions), ``complete``. Positions are float64 in the member's PLACEMENT order (``spec.z``,
the atom order of ``build_positions`` and of a conditions file's ``z``/``pos``), in the frame
``build_positions`` builds them; ``perm`` maps a placement slot to its RDKit atom. The reference
``ref_pos`` is the member's, also placement order, in RDKit's embedding frame.
``states_of_rows`` measures stored positions back into a member's chart. Written atomically
(tmp, then rename) every ``--checkpoint-every`` keys and at the end.

RESUME. A rerun skips the keys the shard file already holds, and REFUSES a file whose header
hash differs: the arguments that decide a row, the source, prior and config sha256, the set
builder's defaults, the recipe constants, the RDKit/torch/numpy versions and the sha256 of every
file in ``CODE_FILES`` plus MXtalTools' conformers package. ``--out-dir``, ``--threads``,
``--checkpoint-every`` and ``--max-molecules`` change no row and are not compared; the git
revisions are recorded, not compared (a changed HEAD is reported).

BOUNDED. Per condition at most ``--max-batches`` x ``--batch`` starts, each relaxed for at most
``(1 + --rounds) x --steps`` steps, one RDKit perception each at most, and RMSD comparisons only
against representatives within ``DE_TOL``; per key the set builder's walk is bounded (its
docstring). A shard is a finite list of keys.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FORMAT = 'conformer_database/1'

#: Angstrom: heavy-atom symmetric best RMSD (rdMolAlign.GetBestRMS, no reflection) below which
#: two minima within DE_TOL are one basin -- the coverage harness's value
RMSD_TOL = 0.25
#: kcal/mol: the energy gate of the basin rule
DE_TOL = 0.5
#: a non-periodic state column at |x| >= 1 - WALL_TOL is on the box wall
WALL_TOL = 1e-9
#: the exclusion reasons, in the order a start is charged to the FIRST that applies
EXCLUSION_REASONS = ('nonfinite', 'still_falling', 'box_wall', 'lock_wrong_side',
                     'stereo_mismatch')
#: refusals of a key beyond build_conformer_set.ALL_CODES
KEY_CODES = {'builder_error': 'build_molecule raised instead of returning a refusal; the '
                              'message is kept'}
#: refusals of a built condition
CONDITION_CODES = {
    'stereo_pin': "build_conformer_references.condition_stereo raised on the member",
    'ring_shapes': 'energies/ring_shapes.py::ring_shapes raised',
    'prior_draw': 'energies/conformer_prior_draw.py::draw_member_prior raised',
    'search': 'the relaxation, the screen or the clustering raised',
    'no_valid_minimum': 'every start of every batch was excluded, so the condition has no basin',
    'rescore': 'a kept row, its stored positions measured back into the member, re-scores more '
               'than RESCORE_TOL from its stored energy (or the measurement raised)',
}
STRATA = ('acyclic', 'single_ring', 'fused')
ASSIGNMENT_RULE = ('universe = build_conformer_set.plan_split keys of both sides (index_min = '
                   'the end of the excluded encoder pool, 0 without one), sorted by (split hash, '
                   'key), so each side keeps its '
                   'own walk order; shard k of N takes universe[k::N]; --max-molecules M keeps '
                   'the first M of those')
#: files whose code decides a row, relative to energy_sampling/; every .py of MXtalTools'
#: mxtaltools/conformers/ is added (``code_identity``)
CODE_FILES = ('build_conformer_database.py', 'build_conformer_set.py',
              'build_conformer_conditions.py', 'build_conformer_references.py',
              'energies/conformer_torsions.py', 'energies/conformer_data.py',
              'energies/conformer_carrier.py', 'energies/ring_shapes.py',
              'energies/conformer_prior_draw.py', 'energies/prior_baselines.py',
              'energies/prior_diagnostics.py', 'energies/ring_metrics.py',
              'energies/stereo_lock.py', 'energies/invertible_centres.py',
              'energies/dof_features.py', 'models/encoder_probe.py')
#: arguments that change no row (not compared on resume), and the ones recorded elsewhere in
#: the header (the shard under 'shard', the three files by sha256)
NON_DEFINING_ARGS = ('cmd', 'out_dir', 'threads', 'checkpoint_every', 'max_molecules',
                     'source', 'prior', 'config', 'shard', 'n_shards')
#: kcal/mol: the largest gap allowed between a kept row's stored energy and the re-score of its
#: stored positions measured back into the member (``states_of_rows``); a larger one refuses
#: the condition ('rescore')
RESCORE_TOL = 1e-6


# ------------------------------------------------------------------ identity and provenance


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _content_sha256(path) -> Optional[str]:
    """sha256 of a gzipped file's CONTENT (a .gz header carries a timestamp); None otherwise."""
    import gzip

    if Path(path).suffix != '.gz':
        return None
    h = hashlib.sha256()
    with gzip.open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def set_builder_defaults() -> dict:
    """The set builder's own argument defaults: split salt, held-out permille, stereo salt and
    the per-molecule condition cap. Read off ``build_conformer_set.parse_args``, not restated."""
    import build_conformer_set as bcs

    d = bcs.parse_args(['--out-dir', '.', '--n-train', 'all', '--n-heldout', 'all'])
    return {'split_salt': d.split_salt, 'heldout_permille': int(d.heldout_permille),
            'stereo_salt': d.stereo_salt, 'cap': bcs._n(d.max_stereoisomers_per_molecule)}


def code_identity() -> dict:
    """``{file: sha256}`` over ``CODE_FILES`` and MXtalTools' conformers package (None where a
    file is absent)."""
    import mxtaltools.conformers as mc

    out = {}
    for rel in CODE_FILES:
        p = HERE / rel
        out[rel] = _sha256(p) if p.is_file() else None
    pkg = Path(mc.__file__).resolve().parent
    for p in sorted(pkg.glob('*.py')):
        out[f'mxtaltools/conformers/{p.name}'] = _sha256(p)
    return out


def git_revisions() -> dict:
    """Both repositories' HEAD and dirty flag, None where git is unavailable. Recorded only."""
    import build_conformer_set as bcs
    import mxtaltools

    out = {}
    for name, repo in (('gfn', HERE), ('mxtaltools', Path(mxtaltools.__file__).resolve().parent)):
        head = bcs._git(repo, 'rev-parse', 'HEAD')
        status = bcs._git(repo, 'status', '--porcelain', '--untracked-files=no')
        out[name] = {'head': head.strip() if head else None,
                     'dirty': None if status is None else bool(status.strip())}
    return out


def hashed_view(header: dict, with_shard: bool = True) -> dict:
    """The part of a header the resume check compares: no paths, no git, no clock. Without the
    shard index it is the identity of the RUN, which every shard of one run shares."""
    shard = {k: header['shard'][k] for k in ('n_shards', 'rule')}
    if with_shard:
        shard['k'] = header['shard']['k']
    return {'format': header['format'], 'shard': shard, 'args': header['args'],
            'source': {k: header['source'][k] for k in ('sha256', 'n_rows')},
            'prior': {k: header['prior'][k] for k in ('bytes', 'sha256')},
            'config': {k: header['config'][k] for k in ('sha256', 'energy_kwargs')},
            'set_builder': header['set_builder'], 'recipe': header['recipe'],
            'code': header['code'], 'versions': header['versions']}


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f'{prefix}{k}.'))
        else:
            out[f'{prefix}{k}'] = v
    return out


def header_differences(old: dict, new: dict) -> List[str]:
    a, b = _flat(hashed_view(old)), _flat(hashed_view(new))
    return [f'{k}: {a.get(k, "<absent>")!r} -> {b.get(k, "<absent>")!r}'
            for k in sorted(set(a) | set(b)) if a.get(k, '<absent>') != b.get(k, '<absent>')]


def member_signature(identifier: str, member) -> str:
    """``ConformerModeller._member_signature``: the run's condition-set digest of this member.

    Imported, not restated. The import pulls in the trainer (about 13 s, once per process)."""
    from conformer_modeller import ConformerModeller

    return ConformerModeller._member_signature(identifier, member)


# ------------------------------------------------------------------ the universe


def stratum_of(smiles: str) -> str:
    """'acyclic' (no ring), 'single_ring' (every ring system is one cycle) or 'fused' (some ring
    system holds two or more cycles: fused, bridged or spiro), by union-find over ring bonds."""
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetRingInfo().NumRings() == 0:
        return 'acyclic'
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ring_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds() if b.IsInRing()]
    for a, b in ring_bonds:
        parent[find(a)] = find(b)
    edges, nodes = {}, {}
    for a, b in ring_bonds:
        r = find(a)
        edges[r] = edges.get(r, 0) + 1
    for a in parent:
        r = find(a)
        nodes[r] = nodes.get(r, 0) + 1
    return 'fused' if any(edges[r] - nodes[r] + 1 > 1 for r in edges) else 'single_ring'


def plan_universe(rows, pool_need: int, defaults: dict):
    """``(universe entries, info)``: both sides of ``plan_split``, in (split hash, key) order."""
    import build_conformer_set as bcs

    pool_idx, pool_end, pool_rule = bcs.pool_rows(rows, int(pool_need))
    sides, set_rej, n_pool_keys = bcs.plan_split(
        rows, index_min=pool_end, index_max=None, pool_idx=pool_idx,
        salt=defaults['split_salt'], permille=defaults['heldout_permille'])
    entries = sorted(sides['train'] + sides['heldout'], key=lambda e: (e.h, e.key))
    refused = {}
    for r in set_rej:
        refused[r['reason_code']] = refused.get(r['reason_code'], 0) + 1
    info = {'pool_need': int(pool_need), 'pool_end': int(pool_end), 'pool_selection': pool_rule,
            'n_pool_keys': int(n_pool_keys),
            'keys': {s: len(sides[s]) for s in ('train', 'heldout')},
            'refused_before_build': dict(sorted(refused.items()))}
    return entries, info


# ------------------------------------------------------------------ one condition


class _Traced:
    """``member`` as ``descend`` sees it, recording the energies of each call (one per step)."""

    def __init__(self, member):
        self._member = member
        self.trace = []

    def __getattr__(self, name):
        return getattr(self._member, name)

    def potential_energy(self, x, temperature, keep_grads=False, return_positions=False):
        out = self._member.potential_energy(x, temperature, keep_grads=keep_grads,
                                            return_positions=return_positions)
        u = out[0] if return_positions else out
        self.trace.append(u.detach().cpu().double().numpy().copy())
        return out


def still_falling(trace, window: int, rate: float) -> np.ndarray:
    """Per start: the best energy seen fell by more than ``rate * window`` over the last
    ``window`` steps of ``trace`` ``[steps, B]`` (kcal/mol; non-finite reads as +inf)."""
    t = np.asarray(trace, dtype=np.float64)
    if t.shape[0] <= int(window):
        raise ValueError(f'a trace of {t.shape[0]} steps has no {window}-step tail')
    best = np.minimum.accumulate(np.where(np.isfinite(t), t, np.inf), axis=0)
    with np.errstate(invalid='ignore'):
        return (best[-1 - int(window)] - best[-1]) > float(rate) * int(window)


def relax(member, x0, *, steps: int, rounds: int, window: int, rate: float):
    """``(states, still_falling, rounds_taken)``: ``descend`` for ``steps``, then up to ``rounds``
    continuations of ``steps`` from the best state of each start still falling. Bounded by
    ``rounds``."""
    from energies.prior_baselines import descend

    traced = _Traced(member)
    with torch.enable_grad():
        bx, _ = descend(traced, x0, int(steps))
    bx = bx.detach().clone()
    fall = still_falling(traced.trace, window, rate)
    taken = np.zeros(len(bx), dtype=np.int64)
    for _ in range(int(rounds)):
        act = np.flatnonzero(fall)
        if not len(act):
            break
        ai = torch.as_tensor(act, dtype=torch.long)
        traced = _Traced(member)
        with torch.enable_grad():
            bxa, _ = descend(traced, bx[ai], int(steps))
        bx[ai] = bxa.detach()
        fall[act] = still_falling(traced.trace, window, rate)
        taken[act] += 1
    return bx, fall, taken


def lock_wrong_side(member, x) -> np.ndarray:
    """Per state: some element of the member's stereo lock has ``s * v <= 0``, the wrong side of
    its reference sign (``energies/stereo_lock.py``). All False with the lock off."""
    n = len(x)
    if not (float(member.stereo_coeff) > 0.0 and member.stereo.n):
        return np.zeros(n, dtype=bool)
    with torch.no_grad():
        pos = member.build_positions(x).reshape(n, -1, 3)
        _, _, sign, _ = member.stereo.tensors(pos.device, pos.dtype)
        return (sign * member.stereo.values(pos) <= 0.0).any(dim=1).cpu().numpy()


def lock_active(member, x) -> np.ndarray:
    """Per state: the lock potential is above zero (some element inside its band)."""
    n = len(x)
    if not (float(member.stereo_coeff) > 0.0 and member.stereo.n):
        return np.zeros(n, dtype=bool)
    with torch.no_grad():
        pos = member.build_positions(x).reshape(n, -1, 3)
        return (member.stereo.lock_energy(pos, member.stereo_coeff) > 0.0).cpu().numpy()


def screen(member, pin, x, fall):
    """``(energies, reason)`` per relaxed state: raw potential at T = 1 (``bake_energies``) and
    the first of ``EXCLUSION_REASONS`` that applies, '' for a kept state."""
    import build_conformer_references as bcr
    from energies.conformer_data import bake_energies

    n = len(x)
    with torch.no_grad():
        e = bake_energies(member, x).detach().cpu().double().numpy()
    per = np.asarray(member.periodic_dims, dtype=bool)
    xa = x.detach().abs().cpu().numpy()
    wall = ((xa[:, ~per] >= 1.0 - WALL_TOL).any(axis=1) if (~per).any()
            else np.zeros(n, dtype=bool))
    reason = np.full(n, '', dtype=object)
    for name, hit in (('nonfinite', ~np.isfinite(e)), ('still_falling', np.asarray(fall)),
                      ('box_wall', wall), ('lock_wrong_side', lock_wrong_side(member, x))):
        reason[(reason == '') & hit] = name
    if bcr._pinned(pin):
        cand = np.flatnonzero(reason == '')
        if len(cand):
            labels = bcr.stereo_labels(member, x[torch.as_tensor(cand, dtype=torch.long)], pin)
            bad = np.array([tuple(lab) != tuple(pin['target']) for lab in labels], dtype=bool)
            reason[cand[bad]] = 'stereo_mismatch'
    return e, reason


class BasinSet:
    """Greedy energy-gated basins over one condition's relaxed minima, grown batch by batch.

    A minimum joins the representative nearest in heavy-atom symmetric RMSD
    (``rdMolAlign.GetBestRMS``, no reflection) among those within ``de_tol`` of its energy,
    when that RMSD is below ``rmsd_tol``; otherwise it founds a basin. One batch is taken in
    ascending energy. A representative is its basin's lowest-energy member: a lower member
    replaces it. ``pos`` holds placement-order positions, ``energy`` kcal/mol, ``hits`` the
    minima each basin collected.
    """

    def __init__(self, member, rmsd_tol: float = RMSD_TOL, de_tol: float = DE_TOL):
        from rdkit import Chem

        m = Chem.Mol(member.mol)
        m.RemoveAllConformers()
        self._heavy = np.array([a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1],
                               dtype=np.int64)
        try:
            h = Chem.RemoveAllHs(m)
        except Exception:                                  # noqa: BLE001 - unsanitisable copy
            h = Chem.RemoveAllHs(m, sanitize=False)
        if ([a.GetAtomicNum() for a in h.GetAtoms()]
                != [m.GetAtomWithIdx(int(i)).GetAtomicNum() for i in self._heavy]):
            raise RuntimeError(f'{member.smiles}: RemoveAllHs reordered the heavy atoms')
        self.member, self._template, self._ref = member, h, Chem.Mol(h)
        self._perm = np.asarray(member.spec.perm)
        self.rmsd_tol, self.de_tol = float(rmsd_tol), float(de_tol)
        self.pos, self.energy, self.hits, self._cid = [], [], [], []
        self.n_rms = 0

    def __len__(self):
        return len(self.energy)

    def _conformer(self, rd_pos):
        from rdkit import Chem

        c = Chem.Conformer(len(self._heavy))
        c.SetPositions(np.ascontiguousarray(rd_pos[self._heavy], dtype=np.float64))
        return c

    def add(self, states, energies, window: float) -> int:
        """Merge one batch of minima. Returns how many basins it founded that lie within
        ``window`` kcal/mol of the lowest representative after the merge."""
        from rdkit import Chem
        from rdkit.Chem import rdMolAlign

        if len(states) == 0:
            return 0
        with torch.no_grad():
            place = self.member.build_positions(states).detach().cpu().double().numpy()
        place = place.reshape(len(states), -1, 3)
        rd = np.empty_like(place)
        rd[:, self._perm] = place
        probe = Chem.Mol(self._template)
        pid = [probe.AddConformer(self._conformer(p), assignId=True) for p in rd]
        founded = []
        for i in np.argsort(np.asarray(energies), kind='stable'):
            e = float(energies[i])
            best, best_rms = -1, np.inf
            for j, ej in enumerate(self.energy):
                if abs(ej - e) < self.de_tol:
                    rms = rdMolAlign.GetBestRMS(probe, self._ref, int(pid[i]), int(self._cid[j]))
                    self.n_rms += 1
                    if rms < best_rms:
                        best, best_rms = j, rms
            if best >= 0 and best_rms < self.rmsd_tol:
                self.hits[best] += 1
                if e < self.energy[best]:
                    self.energy[best], self.pos[best] = e, place[i].copy()
                    self._ref.GetConformer(int(self._cid[best])).SetPositions(
                        np.ascontiguousarray(rd[i][self._heavy], dtype=np.float64))
            else:
                self._cid.append(self._ref.AddConformer(self._conformer(rd[i]), assignId=True))
                self.pos.append(place[i].copy())
                self.energy.append(e)
                self.hits.append(1)
                founded.append(len(self.energy) - 1)
        lo = min(self.energy)
        return sum(1 for j in founded if self.energy[j] <= lo + float(window))

    def rows(self, window: float) -> dict:
        """The basins within ``window`` kcal/mol of the lowest, ascending in energy."""
        n_atoms = int(self.member.spec.n_atoms)
        if not self.energy:
            return {'pos': np.zeros((0, n_atoms, 3)), 'energy': np.zeros(0),
                    'hits': np.zeros(0, dtype=np.int64)}
        e = np.asarray(self.energy, dtype=np.float64)
        keep = [int(j) for j in np.argsort(e, kind='stable') if e[j] <= e.min() + float(window)]
        return {'pos': np.stack([self.pos[j] for j in keep]).astype(np.float64),
                'energy': e[keep], 'hits': np.asarray(self.hits, dtype=np.int64)[keep]}


def states_of_rows(member, pos) -> torch.Tensor:
    """``[B, k]`` states of stored placement-order positions ``[B, N, 3]`` in ``member``'s own
    chart: ``build_conformer_references._state_of_positions`` (MXtalTools ``measure`` under the
    member's dummy frames, then ``state_from_dof``), the inverse of ``build_positions``."""
    import build_conformer_references as bcr
    from mxtaltools.conformers.builder import collate

    tree = collate([member.spec], device=member.device)
    perm = np.asarray(member.spec.perm)
    out = []
    for p in np.asarray(pos, dtype=np.float64):
        rd = np.empty_like(p)
        rd[perm] = p
        out.append(bcr._state_of_positions(member, tree, rd))
    return (torch.cat(out) if out
            else torch.zeros(0, int(member.ndim), dtype=member.dtype, device=member.device))


def stop_after(new_in_window, min_batches: int, stop_streak: int) -> bool:
    """Whether the search stops after the batches whose new in-window basin counts are
    ``new_in_window``: at least ``min_batches`` of them, the last ``stop_streak`` all zero."""
    n, k = len(new_in_window), int(stop_streak)
    return n >= int(min_batches) and n >= k and all(v == 0 for v in new_in_window[n - k:])


def search_condition(member, prior, shapes, pin, seed: int, args) -> dict:
    """The batches, relaxation, screen and basins of one condition. See the module docstring."""
    from energies.conformer_prior_draw import draw_member_prior

    window = float(args.window_kt) * float(member.temperature)
    basins = BasinSet(member)
    excluded = {r: 0 for r in EXCLUSION_REASONS}
    counts = dict(starts=0, valid=0, continued=0, new_in_window=[])
    secs = dict(draw=0.0, relax=0.0, screen=0.0, cluster=0.0)
    stop, stage = 'cap', 'prior_draw'
    try:
        for b in range(int(args.max_batches)):
            stage = 'prior_draw'
            t = time.perf_counter()
            x, _, _ = draw_member_prior(member, int(args.batch),
                                        np.random.default_rng([int(seed), b]), relax_steps=0,
                                        prior=prior, report=False, ring_shapes=shapes)
            secs['draw'] += time.perf_counter() - t
            stage = 'search'
            t = time.perf_counter()
            bx, fall, taken = relax(member, x, steps=args.steps, rounds=args.rounds,
                                    window=args.fall_window, rate=args.fall_rate)
            secs['relax'] += time.perf_counter() - t
            t = time.perf_counter()
            e, reason = screen(member, pin, bx, fall)
            secs['screen'] += time.perf_counter() - t
            for r in EXCLUSION_REASONS:
                excluded[r] += int((reason == r).sum())
            ok = np.flatnonzero(reason == '')
            counts['starts'] += len(bx)
            counts['valid'] += len(ok)
            counts['continued'] += int((taken > 0).sum())
            t = time.perf_counter()
            n_new = basins.add(bx[torch.as_tensor(ok, dtype=torch.long)], e[ok], window)
            secs['cluster'] += time.perf_counter() - t
            counts['new_in_window'].append(int(n_new))
            if stop_after(counts['new_in_window'], args.min_batches, args.stop_streak):
                stop = 'converged'
                break
    except MemoryError:
        raise
    except Exception as exc:                                   # noqa: BLE001 - recorded
        return dict(refusal={'code': stage, 'message': f'{type(exc).__name__}: {exc}'},
                    basins=basins, window=window, excluded=excluded, counts=counts,
                    seconds=secs, stop=None)
    return dict(refusal=None, basins=basins, window=window, excluded=excluded, counts=counts,
                seconds=secs, stop=stop)


def process_condition(mb, built, prior, args) -> dict:
    """One condition's record: its identity, member arrays, refusal or basins, counts, times."""
    import build_conformer_references as bcr
    from energies.ring_shapes import ring_shapes

    t0 = time.perf_counter()
    en, ident = mb.energy, mb.identifier
    seed = bcr.molecule_seed(ident)
    mirror = built.mirror_of.get(ident)
    n_atoms = int(en.spec.n_atoms)
    rec = {'identifier': ident, 'isomer_rank': int(built.isomer_rank[ident]),
           'mirror_of': (mirror if mirror is not None and mirror != ident
                         and mirror in built.isomer_rank else ''),
           'achiral': mirror == ident, 'seed': int(seed), 'n_atoms': n_atoms,
           'k': int(en.ndim), 'z': np.asarray(en.spec.z, dtype=np.int64).copy(),
           'perm': np.asarray(en.spec.perm, dtype=np.int64).copy(),
           'ref_pos': en.ref_pos.detach().cpu().double().numpy().copy(),
           'signature': member_signature(ident, en), 'kT': float(en.temperature),
           # the stereo lock's build-time thermal check on this reference (build_member): its
           # result, or why it was skipped; None with the lock off
           'thermal_check': mb.thermal_check,
           'window_kcal': float(args.window_kt) * float(en.temperature), 'refusal': None,
           'stereo_pin': None, 'ring_shapes': [], 'stop': None, 'e_min': None,
           'basins': {'pos': np.zeros((0, n_atoms, 3)), 'energy': np.zeros(0),
                      'hits': np.zeros(0, dtype=np.int64)},
           'counts': {'starts': 0, 'excluded': {r: 0 for r in EXCLUSION_REASONS}, 'valid': 0,
                      'continued': 0, 'basins_total': 0, 'basins_window': 0, 'batches': 0,
                      'new_in_window': [], 'rmsd_calls': 0, 'window_lock_active': 0,
                      'rescore_gap': None},
           'seconds': {'ring_shapes': 0.0, 'draw': 0.0, 'relax': 0.0, 'screen': 0.0,
                       'cluster': 0.0, 'total': 0.0}}

    def done():
        rec['seconds']['total'] = time.perf_counter() - t0
        return rec

    try:
        pin = bcr.condition_stereo(en)
    except Exception as exc:                                   # noqa: BLE001 - recorded
        rec['refusal'] = {'code': 'stereo_pin', 'message': f'{type(exc).__name__}: {exc}'}
        return done()
    rec['stereo_pin'] = pin['pin']
    t = time.perf_counter()
    try:
        shapes = ring_shapes(en, prior, seed=seed)
    except Exception as exc:                                   # noqa: BLE001 - recorded
        rec['seconds']['ring_shapes'] = time.perf_counter() - t
        rec['refusal'] = {'code': 'ring_shapes', 'message': f'{type(exc).__name__}: {exc}'}
        return done()
    rec['seconds']['ring_shapes'] = time.perf_counter() - t
    rec['ring_shapes'] = [{'block': int(s.block), 'reason': s.reason, 'n_shapes': len(s)}
                          for s in shapes]

    out = search_condition(en, prior, shapes, pin, seed, args)
    basins = out['basins']
    c = rec['counts']
    c.update(starts=out['counts']['starts'], valid=out['counts']['valid'],
             continued=out['counts']['continued'], excluded=out['excluded'],
             new_in_window=out['counts']['new_in_window'],
             batches=len(out['counts']['new_in_window']), rmsd_calls=int(basins.n_rms),
             basins_total=len(basins))
    for k, v in out['seconds'].items():
        rec['seconds'][k] = float(v)
    rec['stop'] = out['stop']
    if out['refusal'] is not None:
        rec['refusal'] = out['refusal']
        return done()
    rows = basins.rows(out['window'])
    if not len(rows['energy']):
        rec['refusal'] = {'code': 'no_valid_minimum',
                          'message': f"0 of {c['starts']} starts kept ({c['excluded']})"}
        return done()
    # THE STORED ROW IS WHAT IT SAYS: its positions, measured back into this member's chart,
    # re-score to its stored energy
    try:
        from energies.conformer_data import bake_energies
        xs = states_of_rows(en, rows['pos'])
        with torch.no_grad():
            e2 = bake_energies(en, xs).detach().cpu().double().numpy()
        gap = float(np.max(np.abs(e2 - rows['energy'])))
    except Exception as exc:                                   # noqa: BLE001 - recorded
        rec['refusal'] = {'code': 'rescore', 'message': f'{type(exc).__name__}: {exc}'}
        return done()
    c['rescore_gap'] = gap
    if not gap <= RESCORE_TOL:
        rec['refusal'] = {'code': 'rescore',
                          'message': f're-score gap {gap:.3g} kcal/mol > {RESCORE_TOL:g}'}
        return done()
    rec['basins'] = rows
    rec['e_min'] = float(rows['energy'][0])
    c['basins_window'] = int(len(rows['energy']))
    c['window_lock_active'] = int(lock_active(en, xs).sum())
    return done()


def process_key(entry, ctx) -> dict:
    """One key's record: ``build_molecule``'s conditions, or its refusal."""
    import build_conformer_set as bcs

    t0 = time.perf_counter()
    rec = {'key': entry.key, 'dataset_index': int(entry.index), 'side': entry.side,
           'hash': f'{entry.h:016x}', 'source_smiles': entry.smiles,
           'stratum': stratum_of(entry.key), 'n_isomers': None, 'refusal': None,
           'rejections': [], 'conditions': [], 'seconds': {'build': 0.0, 'total': 0.0}}
    try:
        built, rej = bcs.build_molecule(entry, ctx['energy_kw'], bundle=None, cap=ctx['cap'],
                                        stereo_salt=ctx['stereo_salt'])
    except MemoryError:
        raise
    except Exception as exc:                                   # noqa: BLE001 - recorded
        built, rej = None, []
        rec['refusal'] = {'code': 'builder_error', 'message': f'{type(exc).__name__}: {exc}'}
    rec['seconds']['build'] = time.perf_counter() - t0
    rec['rejections'] = [{k: r[k] for k in ('level', 'identifier', 'reason_code', 'message')}
                         for r in rej]
    if built is None:
        if rec['refusal'] is None:
            mol = [r for r in rej if r['level'] == 'molecule']
            rec['refusal'] = ({'code': mol[-1]['reason_code'], 'message': mol[-1]['message']}
                              if mol else {'code': 'builder_error',
                                           'message': 'no conditions and no molecule-level '
                                                      'refusal'})
    else:
        rec['n_isomers'] = int(built.n_isomers)
        for mb in built.members:
            rec['conditions'].append(process_condition(mb, built, ctx['prior'], ctx['args']))
    rec['seconds']['total'] = time.perf_counter() - t0
    return rec


# ------------------------------------------------------------------ build


def shard_name(k: int, n: int) -> str:
    return f'shard_{int(k):04d}_of_{int(n):04d}.pt'


def _atomic_save(obj, path: Path):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def build_header(args, energy_kw, ec, defaults, universe_info, n_rows, n_assigned) -> dict:
    """The shard file's provenance. ``hashed_view`` names the part a resume compares."""
    import rdkit

    import build_conformer_set as bcs
    from conformer_modeller import ConformerModeller
    from energies import ring_shapes as rs

    defining = {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(vars(args).items())
                if k not in NON_DEFINING_ARGS}
    cfg_git = bcs._provenance_git(Path(args.config))
    return {
        'format': FORMAT,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'argv': list(sys.argv),
        'shard': {'k': int(args.shard), 'n_shards': int(args.n_shards), 'rule': ASSIGNMENT_RULE},
        'assignment': {'assigned': int(n_assigned), 'max_molecules': args.max_molecules},
        'args': defining,
        'source': {'path': str(Path(args.source).resolve()), 'sha256': _sha256(args.source),
                   'content_sha256': _content_sha256(args.source), 'n_rows': int(n_rows)},
        'prior': {'path': str(Path(args.prior).resolve()), 'bytes': Path(args.prior).stat().st_size,
                  'sha256': _sha256(args.prior)},
        'config': {'path': str(Path(args.config).resolve()), 'sha256': _sha256(args.config),
                   'matches_head': cfg_git.get('config_matches_head'),
                   'energy_kwargs': dict(sorted(energy_kw.items()))},
        'set_builder': {**defaults, **{k: universe_info[k] for k in
                                       ('pool_need', 'pool_end', 'pool_selection',
                                        'n_pool_keys')}},
        'universe': {k: universe_info[k] for k in ('keys', 'refused_before_build')},
        'recipe': {
            'rmsd_tol_A': RMSD_TOL, 'de_tol_kcal': DE_TOL, 'wall_tol': WALL_TOL,
            'exclusion_order': list(EXCLUSION_REASONS),
            'currency': 'raw potential at T = 1, kcal/mol (energies/conformer_data.py::'
                        'bake_energies)',
            'window': '--window-kt x member.temperature (kT, kcal/mol)',
            'atom_order': 'placement order (member.spec.z); perm[i] = RDKit atom of slot i',
            'lock_rule': 'excluded when some stereo-lock element has s * v <= 0',
            'draw': 'draw_member_prior(relax_steps=0, ring_shapes=ring_shapes(member, prior, '
                    'seed=molecule_seed(identifier))), numpy default_rng([seed, batch])',
            'ring_shapes': {'n_embed': inspect.signature(rs.ring_shapes).parameters[
                                'n_embed'].default,
                            'dedup_deg': rs.DEDUP_DEG, 'closure_tol': rs.CLOSURE_TOL,
                            'trial_draws': rs.TRIAL_DRAWS,
                            'embed_timeout_s': rs.EMBED_TIMEOUT_S},
            'signature': ConformerModeller._MEMBER_SIGNATURE,
            'internal_prior_path_in_config': ec.get('internal_prior_path'),
        },
        'code': code_identity(),
        'git': git_revisions(),
        'versions': {'rdkit': rdkit.__version__, 'torch': torch.__version__,
                     'numpy': np.__version__, 'python': platform.python_version()},
    }


def _blob(header, assigned, done, complete):
    return {'format': FORMAT, 'header': header,
            'header_hash': _digest(hashed_view(header)),
            'run_hash': _digest(hashed_view(header, with_shard=False)),
            'assigned': list(assigned), 'keys': done, 'complete': bool(complete),
            'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}


def cmd_build(args) -> dict:
    from rdkit import RDLogger

    import build_conformer_set as bcs
    from build_conformer_conditions import energy_kwargs_from_config
    from energies.prior_baselines import load_prior

    RDLogger.DisableLog('rdApp.*')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(args.threads))
    t_start = time.time()
    if not 0 <= int(args.shard) < int(args.n_shards):
        raise SystemExit(f'--shard {args.shard} is not in [0, --n-shards {args.n_shards})')
    if not 1 <= int(args.min_batches) <= int(args.max_batches):
        raise SystemExit('need 1 <= --min-batches <= --max-batches')
    if int(args.stop_streak) < 1:
        raise SystemExit('--stop-streak must be >= 1')
    if int(args.steps) <= int(args.fall_window):
        raise SystemExit('--steps must exceed --fall-window: the falling test reads the last '
                         '--fall-window steps of a descent')
    for flag, p in (('--source', args.source), ('--prior', args.prior), ('--config', args.config)):
        if not Path(p).is_file():
            raise SystemExit(f'{flag} {p} does not exist')

    energy_kw, ec = energy_kwargs_from_config(args.config)
    # as build_conformer_set.main: the production target when the config names none
    energy_kw.setdefault('level', 'full')
    energy_kw.setdefault('force_field', 'mmff')
    defaults = set_builder_defaults()
    prior, _ = load_prior(str(args.prior))

    rows = bcs.read_source(args.source)
    universe, uinfo = plan_universe(rows, args.pool_need, defaults)
    assigned = universe[int(args.shard)::int(args.n_shards)]
    if args.max_molecules is not None:
        assigned = assigned[:int(args.max_molecules)]
    header = build_header(args, energy_kw, ec, defaults, uinfo, len(rows), len(assigned))
    print(f"universe {sum(uinfo['keys'].values())} keys ({uinfo['keys']}); shard "
          f"{args.shard}/{args.n_shards}: {len(assigned)} keys; level {energy_kw.get('level')}; "
          f"config matches HEAD: {header['config']['matches_head']}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / shard_name(args.shard, args.n_shards)
    done: Dict[str, dict] = {}
    if path.exists():
        old = torch.load(path, weights_only=False, map_location='cpu')
        if old.get('format') != FORMAT or old.get('header_hash') != _digest(hashed_view(header)):
            diff = (header_differences(old['header'], header) if old.get('format') == FORMAT
                    else [f'format {old.get("format")!r}'])
            raise SystemExit(f'{path} was built under another header; refusing to resume. '
                             f'Remove it or choose another --out-dir. Differences:\n  '
                             + '\n  '.join(diff[:20]))
        if old['header'].get('git') != header['git']:
            print(f"note: git revisions differ from the file's ({old['header'].get('git')} -> "
                  f"{header['git']}); the compared code files are unchanged", flush=True)
        done = dict(old['keys'])
        header = dict(header, created_utc=old['header'].get('created_utc'))
    want = {e.key for e in assigned}
    todo = [e for e in assigned if e.key not in done]
    print(f'{path}: {len(done)} key(s) resumed, {len(todo)} to process', flush=True)
    ctx = {'energy_kw': energy_kw, 'cap': defaults['cap'], 'stereo_salt': defaults['stereo_salt'],
           'prior': prior, 'args': args}
    since = 0
    for i, e in enumerate(todo):
        rec = process_key(e, ctx)
        done[e.key] = rec
        conds = rec['conditions']
        n_rows = sum(len(c['basins']['energy']) for c in conds)
        why = rec['refusal']['code'] if rec['refusal'] else ','.join(
            (c['refusal']['code'] if c['refusal'] else c['stop']) for c in conds)
        print(f"  [{i + 1}/{len(todo)}] {e.key} ({e.index}, {e.side}, {rec['stratum']}): "
              f"{len(conds)} condition(s), {n_rows} row(s), {why}; "
              f"{rec['seconds']['total']:.1f} s", flush=True)
        since += 1
        if since >= int(args.checkpoint_every):
            _atomic_save(_blob(header, [a.key for a in assigned], done, False), path)
            since = 0
    complete = want <= set(done)
    _atomic_save(_blob(header, [a.key for a in assigned], done, complete), path)
    print(f'wrote {path}: {len(done)} key(s), complete {complete}, '
          f'{time.time() - t_start:.0f} s', flush=True)
    return {'path': path, 'header': header, 'keys': done, 'complete': complete}


# ------------------------------------------------------------------ summarize


def _q(v, p):
    return float(np.percentile(v, p)) if len(v) else float('nan')


def summarize(out_dir, *, mirror_de_tol: float = 0.05, mirror_count_rel: float = 0.5,
              mirror_count_abs: int = 3, log=print) -> dict:
    """Counts per stratum, refusals by code, the cap fraction, timing and a projection for the
    full universe, and the mirror-pair check, over every shard file of ``out_dir``."""
    files = sorted(Path(out_dir).glob('shard_*_of_*.pt'))
    if not files:
        raise SystemExit(f'{out_dir}: no shard files')
    # ONE SHARD AT A TIME, kept slim: a full run's shard files hold every basin's positions
    # (2.9 GB on disk for the 400-shard QM9 run), and loading them all at once was killed on a
    # login node. Only the fields read below are kept.
    heavy = ('basins', 'ref_pos', 'z', 'perm', 'ring_shapes', 'stereo_pin')

    def _slim(blob):
        for r in blob['keys'].values():
            r['conditions'] = [{k: v for k, v in c.items() if k not in heavy}
                               for c in r['conditions']]
        return blob

    blobs = []
    for f in files:
        blobs.append(_slim(torch.load(f, weights_only=False, map_location='cpu')))
    runs = {b['run_hash'] for b in blobs}
    if len(runs) != 1:
        raise SystemExit(f'{out_dir} mixes {len(runs)} runs (different headers); summarize one')
    h0 = blobs[0]['header']
    n_shards = int(h0['shard']['n_shards'])
    present = sorted(int(b['header']['shard']['k']) for b in blobs)
    keys = [r for b in blobs for r in b['keys'].values()]
    conds = [(r, c) for r in keys for c in r['conditions']]
    level = h0['config']['energy_kwargs'].get('level')

    def per(stratum):
        ks = [r for r in keys if stratum in (None, r['stratum'])]
        cs = [c for r in ks for c in r['conditions']]
        searched = [c for c in cs if c['stop'] is not None]
        kept = [c for c in cs if c['refusal'] is None]
        nb = np.array([c['counts']['basins_window'] for c in kept], dtype=float)
        nt = np.array([c['counts']['basins_total'] for c in kept], dtype=float)
        return {'keys': len(ks), 'keys_refused': sum(1 for r in ks if r['refusal']),
                'conditions': len(cs), 'conditions_refused': len(cs) - len(kept),
                'rows': int(nb.sum()), 'rows_per_condition_mean': float(nb.mean()) if len(nb)
                else float('nan'), 'rows_per_condition_median': _q(nb, 50),
                'rows_per_condition_max': float(nb.max()) if len(nb) else float('nan'),
                'basins_total_mean': float(nt.mean()) if len(nt) else float('nan'),
                'batches_mean': float(np.mean([c['counts']['batches'] for c in searched]))
                if searched else float('nan'),
                'stopped_at_cap_frac': (sum(1 for c in searched if c['stop'] == 'cap')
                                        / len(searched)) if searched else float('nan')}

    strata = {s: per(s) for s in STRATA}
    strata['all'] = per(None)
    key_codes, cond_codes = {}, {}
    for r in keys:
        if r['refusal']:
            k = (r['stratum'], r['refusal']['code'])
            key_codes[k] = key_codes.get(k, 0) + 1
    for r, c in conds:
        if c['refusal']:
            k = (r['stratum'], c['refusal']['code'])
            cond_codes[k] = cond_codes.get(k, 0) + 1
    starts = sum(c['counts']['starts'] for _, c in conds)
    excl = {x: sum(c['counts']['excluded'][x] for _, c in conds) for x in EXCLUSION_REASONS}

    secs = {'build': sum(r['seconds']['build'] for r in keys)}
    for st in ('ring_shapes', 'draw', 'relax', 'screen', 'cluster'):
        secs[st] = sum(c['seconds'][st] for _, c in conds)
    total = sum(r['seconds']['total'] for r in keys)
    n_universe = int(sum(h0['universe']['keys'].values()))
    per_key = total / len(keys) if keys else float('nan')
    proj = {'universe_keys': n_universe, 'keys_done': len(keys),
            'cpu_hours': per_key * n_universe / 3600.0,
            'shard_hours': per_key * n_universe / 3600.0 / n_shards,
            'conditions': len(conds) / len(keys) * n_universe if keys else float('nan'),
            'rows': strata['all']['rows'] / len(keys) * n_universe if keys else float('nan')}

    by_ident = {c['identifier']: (r, c) for r, c in conds}
    pairs, outliers = [], []
    for r, c in conds:
        m = c['mirror_of']
        if not m or m not in by_ident or c['identifier'] > m:
            continue
        _, d = by_ident[m]
        if c['e_min'] is None or d['e_min'] is None:
            continue
        de = abs(c['e_min'] - d['e_min'])
        na, nb_ = c['counts']['basins_window'], d['counts']['basins_window']
        count_bad = (abs(na - nb_) >= int(mirror_count_abs)
                     and abs(na - nb_) > float(mirror_count_rel) * max(na, nb_))
        row = {'a': c['identifier'], 'b': m, 'dataset_index': r['dataset_index'],
               'stratum': r['stratum'], 'de_min': de, 'n_a': na, 'n_b': nb_,
               'outlier': bool(de > float(mirror_de_tol) or count_bad)}
        pairs.append(row)
        if row['outlier']:
            outliers.append(row)
    outliers.sort(key=lambda p: -p['de_min'])

    out = {'files': len(files), 'n_shards': n_shards, 'shards_present': len(present),
           'shards_complete': sum(1 for b in blobs if b['complete']), 'level': level,
           'strata': strata, 'key_refusals': {f'{s}/{c}': n for (s, c), n in sorted(key_codes.items())},
           'condition_refusals': {f'{s}/{c}': n for (s, c), n in sorted(cond_codes.items())},
           'starts': starts, 'excluded': excl, 'seconds': secs, 'seconds_total': total,
           'projection': proj, 'mirror_pairs': len(pairs), 'mirror_outliers': outliers,
           'mirror_rule': {'de_tol_kcal': mirror_de_tol, 'count_rel': mirror_count_rel,
                           'count_abs': mirror_count_abs}}

    L = log
    L(f"{out_dir}: {len(files)} shard file(s) of {n_shards} ({out['shards_complete']} complete), "
      f"level {level!r}, {len(keys)} key(s), {len(conds)} condition(s).")
    L('')
    L(f'Table 1. Per stratum, over the processed keys of this run (level {level!r}; a key is a '
      'constitution, a condition a stereoisomer). Rows are basins within the window of each '
      "condition's lowest; the cap fraction is over conditions that searched.")
    L('| stratum | keys | keys refused | conditions | conditions refused | rows | '
      'rows per condition mean / median / max | basins found per condition (mean) | '
      'batches per condition (mean) | stopped at the batch cap (fraction) |')
    L('|---|---|---|---|---|---|---|---|---|---|')
    for s in (*STRATA, 'all'):
        v = strata[s]
        L(f"| {s} | {v['keys']} | {v['keys_refused']} | {v['conditions']} | "
          f"{v['conditions_refused']} | {v['rows']} | {v['rows_per_condition_mean']:.1f} / "
          f"{v['rows_per_condition_median']:.0f} / {v['rows_per_condition_max']:.0f} | "
          f"{v['basins_total_mean']:.1f} | {v['batches_mean']:.2f} | "
          f"{v['stopped_at_cap_frac']:.3f} |")
    L('')
    L('Table 2. Refusals by stratum and code (keys: build_conformer_set codes and '
      'builder_error; conditions: CONDITION_CODES).')
    L('| level | stratum/code | count |')
    L('|---|---|---|')
    for name, d in (('key', out['key_refusals']), ('condition', out['condition_refusals'])):
        for k, n in d.items():
            L(f'| {name} | {k} | {n} |')
    if not out['key_refusals'] and not out['condition_refusals']:
        L('| - | none | 0 |')
    L('')
    L(f'Table 3. Relaxed starts excluded, by the first reason that applied, over {starts} '
      'starts.')
    L('| reason | starts | fraction of starts |')
    L('|---|---|---|')
    for x in EXCLUSION_REASONS:
        L(f'| {x} | {excl[x]} | {excl[x] / starts if starts else float("nan"):.4f} |')
    L('')
    L(f'Table 4. Wall time by stage, summed over the {len(keys)} processed keys (one thread '
      'each), and the projection to the whole universe at the same mean cost per key. Keys '
      'are assigned in split-hash order, so a shard prefix is an unordered sample; a pilot of '
      'a few keys per shard carries sampling error.')
    L('| stage | seconds (sum) | seconds per key (mean) |')
    L('|---|---|---|')
    for st, v in secs.items():
        L(f'| {st} | {v:.1f} | {v / len(keys) if keys else float("nan"):.2f} |')
    L(f'| total | {total:.1f} | {per_key:.2f} |')
    L(f"Projection over {n_universe} universe keys: {proj['cpu_hours']:.1f} CPU-hours, "
      f"{proj['shard_hours']:.2f} h per shard over {n_shards} shard(s), "
      f"{proj['conditions']:.0f} conditions, {proj['rows']:.0f} rows.")
    L('')
    L(f'Table 5. Mirror pairs (pick and mirror built as conditions, both with basins): '
      f'{len(pairs)} pair(s), {len(outliers)} outlier(s). An outlier has |dE_min| > '
      f'{mirror_de_tol} kcal/mol, or window basin counts differing by >= {mirror_count_abs} '
      f'and by more than {mirror_count_rel:.0%} of the larger. The two members are independent '
      'embeddings of one physics, so both readings agree in expectation.')
    L('| pick | mirror | dataset index | stratum | abs dE_min (kcal/mol) | rows pick | '
      'rows mirror |')
    L('|---|---|---|---|---|---|---|')
    for p in outliers[:20]:
        L(f"| {p['a']} | {p['b']} | {p['dataset_index']} | {p['stratum']} | {p['de_min']:.4f} | "
          f"{p['n_a']} | {p['n_b']} |")
    if not outliers:
        L('| - | - | - | - | none | - | - |')
    return out


# ------------------------------------------------------------------ read (consumers)


#: the fields of a condition record ``read_conditions`` keeps; the rest (counts, seconds, ring
#: shapes, the stereo pin) describe the search, not the rows
READ_FIELDS = ('identifier', 'isomer_rank', 'mirror_of', 'n_atoms', 'k', 'z', 'perm', 'ref_pos',
               'signature', 'kT', 'window_kcal', 'refusal', 'e_min', 'basins')
#: fields a reader may ask for that an older record lacks (read as None)
OPTIONAL_FIELDS = ('thermal_check',)
#: what ``read_references`` keeps of a condition record: the member's reference and identity
REFERENCE_FIELDS = ('identifier', 'z', 'perm', 'ref_pos', 'signature', 'thermal_check')


def database_identity(blobs_or_headers) -> dict:
    """The identity a consumer records for a whole database: its run hash (every shard of one
    run shares it; ``hashed_view`` without the shard) and the sha256 over its shards' sorted
    header hashes, which names the exact set of shard files."""
    hashes = sorted(h for h in blobs_or_headers)
    return {'header_hashes_sha256': hashlib.sha256('\n'.join(hashes).encode()).hexdigest(),
            'n_header_hashes': len(hashes)}


def first_header(db_dir) -> dict:
    """The header of the database's first shard file, for checks made before a full read."""
    files = sorted(Path(db_dir).glob('shard_*_of_*.pt'))
    if not files:
        raise SystemExit(f'{db_dir}: no database shard files')
    b = torch.load(files[0], weights_only=False, map_location='cpu')
    if b.get('format') != FORMAT:
        raise SystemExit(f'{files[0]}: format {b.get("format")!r}, this reader takes {FORMAT!r}')
    return {'path': str(Path(db_dir).resolve()), 'energy_kwargs': b['header']['config'][
        'energy_kwargs'], 'set_builder': b['header']['set_builder']}


def read_conditions(db_dir, identifiers=None, *, log=print, fields=READ_FIELDS,
                    refusals: Optional[dict] = None):
    """``(info, {identifier: condition record})`` for a finished database.

    One shard at a time; only the conditions named in ``identifiers`` are kept (all when None),
    each with ``fields`` (default ``READ_FIELDS``; one of ``OPTIONAL_FIELDS`` an older record
    lacks reads as None) plus its key's ``key`` and ``side``. REFUSES a directory whose
    shards are not one complete run: no shard file, a shard count other than the header's
    ``n_shards``, shards of different runs (``run_hash``), an incomplete shard, another format,
    or one identifier in two records. ``info`` carries the path, format, run hash, the digest
    over the shard header hashes (``database_identity``), the member kwargs the run was built
    under and the set builder's defaults it planned its universe with. A ``refusals`` dict,
    when passed, is filled with ``{identifier: (key, reason_code, message)}`` for every
    isomer-level rejection of the keys read (``identifiers`` filters these too).
    """
    db_dir = Path(db_dir)
    files = sorted(db_dir.glob('shard_*_of_*.pt'))
    if not files:
        raise SystemExit(f'{db_dir}: no database shard files')
    want = None if identifiers is None else set(identifiers)
    out: Dict[str, dict] = {}
    runs, hashes, incomplete, h0 = set(), [], [], None
    t0 = time.time()
    for f in files:
        b = torch.load(f, weights_only=False, map_location='cpu')
        if b.get('format') != FORMAT:
            raise SystemExit(f'{f}: format {b.get("format")!r}, this reader takes {FORMAT!r}')
        runs.add(b['run_hash'])
        hashes.append(b['header_hash'])
        if not b['complete']:
            incomplete.append(f.name)
        if h0 is None:
            h0 = b['header']
        for r in b['keys'].values():
            if refusals is not None:
                for j in r['rejections']:
                    ident = j['identifier']
                    if j['level'] == 'isomer' and ident and (want is None or ident in want):
                        refusals[ident] = (r['key'], j['reason_code'], j['message'])
            for c in r['conditions']:
                ident = c['identifier']
                if want is not None and ident not in want:
                    continue
                if ident in out:
                    raise SystemExit(f'{db_dir}: identifier {ident!r} has two records')
                rec = {k: (c.get(k) if k in OPTIONAL_FIELDS else c[k]) for k in fields}
                rec.update(key=r['key'], side=r['side'])
                out[ident] = rec
        del b
    n_shards = int(h0['shard']['n_shards'])
    problems = []
    if len(runs) != 1:
        problems.append(f'{len(runs)} runs (different headers) in one directory')
    if len(files) != n_shards:
        problems.append(f'{len(files)} shard files of {n_shards}')
    if incomplete:
        problems.append(f'{len(incomplete)} incomplete shard(s), e.g. {incomplete[0]}')
    if problems:
        raise SystemExit(f'{db_dir} is not one finished database: ' + '; '.join(problems))
    info = {'path': str(db_dir.resolve()), 'format': FORMAT, 'run_hash': runs.pop(),
            **database_identity(hashes), 'n_shards': n_shards,
            'created_utc': h0.get('created_utc'), 'git': h0.get('git'),
            'energy_kwargs': h0['config']['energy_kwargs'],
            'set_builder': h0['set_builder'], 'source_sha256': h0['source']['sha256'],
            'window_kt': h0['args'].get('window_kt')}
    log(f'database {db_dir}: {len(files)} shard(s), {len(out)} condition record(s) kept, '
        f'{time.time() - t0:.0f} s')
    return info, out


def read_references(db_dir, *, log=print):
    """``(info, {key: {identifier: StoredReference | StoredRefusal}})``: every condition's
    stored reference, and every isomer the database's walk refused.

    What ``build_conformer_set.py --database`` builds its members from, so a set built in any
    environment has the database's charts rather than its own RDKit's embeddings, and refuses
    the isomers the database's walk refused (with the database's code) rather than retrying
    them under its own RDKit. Read with ``read_conditions``' checks (one finished run),
    keeping ``REFERENCE_FIELDS`` only. Only member-level codes (``REASON_CODES``) are kept as
    refusals; the set-level ones are decided by the walk itself.

    ``thermal_check`` on each reference is the database's record of the stereo lock's
    thermal check on that geometry: the record's own result where it carries one, and
    otherwise, with the lock on in the database's member kwargs, the fact that every
    condition record of a ``conformer_database/1`` is a member ``build_member`` returned, which
    refuses one whose check fires (the check predates the database builder). None with the
    lock off, where no check runs.
    """
    from build_conformer_conditions import REASON_CODES, StoredReference, StoredRefusal

    refused: Dict[str, tuple] = {}
    info, recs = read_conditions(db_dir, None, log=log, fields=REFERENCE_FIELDS,
                                 refusals=refused)
    ekw = info['energy_kwargs']
    lock = float(ekw.get('stereo_coeff', 0) or 0)
    tag = f"database {info['run_hash'][:12]}"
    implied = (f"{tag}: built by build_member, which refuses a member whose thermal check "
               f"fires (stereo_coeff {lock:g}, seed {int(ekw.get('seed', 0) or 0)})")
    out: Dict[str, dict] = {}
    for ident, r in recs.items():
        tc = r['thermal_check']
        if lock <= 0:
            why = None
        elif tc is None:
            why = implied
        elif tc.get('status') == 'passed':
            why = (f"{tag}: passed when the database built this member ({tc['n']} samples, "
                   f"seed {tc['seed']}, {tc['steps']} steps)")
        else:
            why = str(tc.get('reason')) if tc.get('reason') else None
        out.setdefault(r['key'], {})[ident] = StoredReference(
            pos=np.asarray(r['ref_pos'], dtype=np.float64), perm=np.asarray(r['perm']),
            z=np.asarray(r['z']), signature=r['signature'], thermal_check=why, source=tag)
    for ident, (key, code, message) in refused.items():
        if code in REASON_CODES and ident not in out.get(key, {}):
            out.setdefault(key, {})[ident] = StoredRefusal(code, message, tag)
    return info, out


def refuse_other_member_kwargs(info: dict, energy_kw: dict, what: str):
    """SystemExit when the database's members were built under other member-defining
    ConformerTorsions arguments than ``energy_kw`` (``build_conformer_references.
    defining_energy``, resolved against the signature defaults)."""
    import build_conformer_references as bcr

    have = bcr.defining_energy(bcr.member_kwargs(info['energy_kwargs']))
    want = bcr.defining_energy(bcr.member_kwargs(energy_kw))
    diff = [f'{k}: database {have.get(k, "<absent>")!r}, {what} {want.get(k, "<absent>")!r}'
            for k in sorted(set(have) | set(want))
            if have.get(k, '<absent>') != want.get(k, '<absent>')]
    if diff:
        raise SystemExit(f"{info['path']} was built under other member arguments than the "
                         f'{what}:\n  ' + '\n  '.join(diff))


#: why a consumer takes no rows from the database for a condition
MATCH_CODES = {
    'db_absent': 'the database holds no record of this identifier',
    'db_refused': 'the database refused the condition (its code in the message)',
    'db_signature': "the database member's signature (ConformerModeller._member_signature: "
                    'SMILES, block codes, placement z) or placement z differs from this member',
    'db_outside_box': 'a stored row, measured into this member, lies outside the state box on a '
                      'non-periodic column',
    'db_rescore': 'a stored row, measured into this member, re-scores more than the tolerance '
                  'from its stored energy',
}


def match_rows(member, identifier: str, rec: Optional[dict], cap: Optional[int], tol: float):
    """``(states [m, k] float64, stored energies [m], info)`` of one condition's lowest ``cap``
    database rows, measured into ``member``'s own chart (``states_of_rows``), or
    ``(None, None, info)`` with ``info['code']`` one of ``MATCH_CODES``.

    Matched by identifier and member signature. Every taken row is re-scored through
    ``member`` (``bake_energies``, raw potential at T = 1) and the condition is refused when the
    worst gap to the stored energy exceeds ``tol`` kcal/mol, or when a row falls outside the
    box on a non-periodic column. ``info`` records the gap and the reference-geometry gap.
    """
    from energies.conformer_data import bake_energies

    info = {'identifier': identifier, 'code': None, 'message': ''}
    if rec is None:
        info.update(code='db_absent', message='no record')
        return None, None, info
    if rec['refusal'] is not None:
        info.update(code='db_refused',
                    message=f"{rec['refusal']['code']}: {rec['refusal']['message']}"[:300])
        return None, None, info
    sig = member_signature(identifier, member)
    z = np.asarray(member.spec.z, dtype=np.int64)
    if sig != rec['signature'] or not np.array_equal(z, np.asarray(rec['z'])):
        info.update(code='db_signature', message=f"member {sig}, database {rec['signature']}")
        return None, None, info
    ref = member.ref_pos.detach().cpu().double().numpy()
    info['ref_pos_gap'] = float(np.abs(ref - np.asarray(rec['ref_pos'])).max())
    n = len(rec['basins']['energy']) if cap is None else min(int(cap),
                                                              len(rec['basins']['energy']))
    pos = np.asarray(rec['basins']['pos'][:n], dtype=np.float64)
    stored = np.asarray(rec['basins']['energy'][:n], dtype=np.float64)
    x = states_of_rows(member, pos)
    per = np.asarray(member.periodic_dims, dtype=bool)
    if (~per).any() and bool((x[:, torch.as_tensor(~per)].abs() > 1.0).any()):
        info.update(code='db_outside_box',
                    message=f'max |x| {float(x[:, torch.as_tensor(~per)].abs().max()):.6g}')
        return None, None, info
    with torch.no_grad():
        e = bake_energies(member, x).detach().cpu().double().numpy()
    gap = float(np.max(np.abs(e - stored))) if n else 0.0
    info['rescore_gap'] = gap
    if not gap <= float(tol):
        info.update(code='db_rescore', message=f're-score gap {gap:.3g} kcal/mol > {tol:g}')
        return None, None, info
    info.update(rows=int(n), rows_available=int(len(rec['basins']['energy'])),
                e_min=float(stored[0]) if n else None)
    return x, stored, info


# ------------------------------------------------------------------ CLI


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='build (or resume) one shard')
    b.add_argument('--out-dir', type=Path, required=True)
    b.add_argument('--source', type=Path, required=True,
                   help='the QM9 index: .pt, or .csv/.tsv (optionally .gz) with '
                        'dataset_index,smiles')
    b.add_argument('--prior', type=Path, required=True,
                   help='the fitted InternalPrior (conformer_prior_v2.pt)')
    b.add_argument('--pool-need', type=int, default=0,
                   help="leading distinct SMILES excluded as the frozen encoder's pool; 0 "
                        '(default, as build_conformer_set.py) excludes none')
    b.add_argument('--config', type=Path,
                   default=HERE / 'configs' / 'conformer_mk.yaml',
                   help='the run config whose energy_config gives the member kwargs')
    b.add_argument('--shard', type=int, default=0)
    b.add_argument('--n-shards', type=int, default=1)
    b.add_argument('--batch', type=int, default=32, help='prior draws per batch')
    b.add_argument('--steps', type=int, default=150, help='descent steps per round')
    b.add_argument('--rounds', type=int, default=3,
                   help='continuation rounds for starts still falling')
    b.add_argument('--fall-window', type=int, default=20, help='steps the falling test reads')
    b.add_argument('--fall-rate', type=float, default=0.01, help='kcal/mol per step')
    b.add_argument('--min-batches', type=int, default=2)
    b.add_argument('--max-batches', type=int, default=24)
    b.add_argument('--stop-streak', type=int, default=2,
                   help='consecutive batches founding no in-window basin before the stop')
    b.add_argument('--window-kt', type=float, default=10.0)
    b.add_argument('--max-molecules', type=int, default=None,
                   help='process only the first M keys of this shard (pilot)')
    b.add_argument('--checkpoint-every', type=int, default=4, help='keys between writes')
    b.add_argument('--threads', type=int, default=1)
    s = sub.add_parser('summarize', help='tables over an output directory')
    s.add_argument('out_dir', type=Path)
    s.add_argument('--mirror-de-tol', type=float, default=0.05, help='kcal/mol')
    s.add_argument('--mirror-count-rel', type=float, default=0.5)
    s.add_argument('--mirror-count-abs', type=int, default=3)
    s.add_argument('--json', type=Path, default=None, help='also write the summary here')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.cmd == 'build':
        return cmd_build(args)
    out = summarize(args.out_dir, mirror_de_tol=args.mirror_de_tol,
                    mirror_count_rel=args.mirror_count_rel,
                    mirror_count_abs=args.mirror_count_abs)
    if args.json is not None:
        with open(args.json, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(out, f, indent=1, default=str)
    return out


if __name__ == '__main__':
    main()
