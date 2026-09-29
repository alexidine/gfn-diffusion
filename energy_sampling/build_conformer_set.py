"""Build a conformer CONDITION SET: QM9 molecules, split train /
held-out by constitution, stereoisomers enumerated as separate conditions, one carrier
layout over both files, every refusal recorded, and a manifest that pins it all.

    python build_conformer_set.py --out-dir D:/conformer_sets/r1 --n-train 50 --n-heldout 10

writes into ``--out-dir``:

    conditions_train.pt     cfg:molecules_path       (carrier form, embeddings baked)
    conditions_heldout.pt   cfg:test_molecules_path  (same K, same block widths)
    prior_train.pt          cfg:prior_path, only with --prior-rows-per-condition > 0
    split.tsv               every universe key: its side and its place in the walk order
    molecules.tsv           one row per kept CONDITION (stereoisomer)
    rejections.tsv          one row per refused molecule and per refused stereoisomer
    manifest.json           provenance, layout, counts per reason code, sha256 per artifact

THE UNIVERSE. By default (``--pool-need 0``, owner decision 2026-09-29: the frozen encoder is
a black box whose train/test overlap is accepted) every QM9 row, deduplicated by key and
split as below; ``pool_rows`` returns an empty pool and the manifest records ``need`` 0.
With ``--pool-need N > 0`` (or ``--pool-from-encoder``, which derives N) the universe is the
file indices at or past the end of the frozen encoder's training pool, with every
constitution the pool contains removed. That pool is reproduced, not assumed: the encoder
checkpoint is matched to the results file that trained it (``encoder_pool``), its ``need``
is recomputed from that file's metadata, and ``encoder_probe.load_qm9``'s selection -- the
first ``need`` distinct raw SMILES in file order -- is replayed on the source.
build_conformer_database.py plans its universe through the same two functions and the same
default, so a set built here finds its rows there.

THE KEY is the canonical stereo-free SMILES (``constitution_key``, which is
``encoder_probe.parent_skeleton`` with explicit hydrogens dropped first). It dedupes QM9's 87
Kekule-variant duplicates, decides the side of the split, and seeds every per-molecule
choice, so all stereoisomers of one molecule share a side, and a molecule's side and
isomers never change as the ladder grows.

THE SPLIT AND THE LADDER. A key is held out when ``blake2b(salt, key) mod 1000 < permille``,
and each side is walked in hash order. A rung is the first ``n`` molecules of that walk that
yield at least one condition, so rung n is a subset of rung n' > n on both sides -- the walk
is deterministic because every step of it is (fixed ETKDG seed, hash-ordered isomers).

STEREO. Each stereoisomer is its own condition, identified by a stereo-tagged SMILES that
parses back to it; the encoder is fed that SMILES, so its CIP parity channel is live. "One
stereoisomer" is ``build_conformer_conditions.stereo_identity`` (fixed-H InChI and CIP
labels), not canonical SMILES equality, which on some cages names one isomer twice or
rewrites one into another. Per molecule the isomers are ordered by a salted hash and walked:
each built isomer is verified against its own embedded reference
(``build_conformer_conditions.build_member``), and a chiral pick is followed by its mirror
image, until ``--max-stereoisomers-per-molecule`` conditions (default 2: a pick plus its
mirror when chiral). The mirror pair has equal log Z in expectation -- the energy is achiral
-- which is a correctness check at `full`, where no exact log Z exists. Only in expectation:
the two references are independent ETKDG embeddings, not reflections of one another.

REFUSALS. Every per-molecule refusal is a code from ``REASON_CODES`` (per member) or
``SET_REASON_CODES`` (set level). Linear groups are not refused here: the builder admits
whatever ``ConformerTorsions`` and ``CarrierLayout`` admit, and records what they refuse
under their codes. Nitriles and alkynes are admitted: a linear bend is a transverse (u, v)
pair, placed in the carrier's theta region, and an alkyne's collinear frames take a Z-matrix
dummy reference (energies/conformer_carrier.py, energies/conformer_torsions.py).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from build_conformer_conditions import (REASON_CODES, Member, MemberRefused, build_member,
                                        energy_kwargs_from_config, smiles_identity,
                                        stereoisomer_classes)

QM9_PATH = 'D:/crystal_datasets/qm9_dataset.pt'
FORMAT = 'conformer_set_v1'
ARTIFACT_NAMES = ('conditions_train.pt', 'conditions_heldout.pt', 'prior_train.pt',
                  'split.tsv', 'molecules.tsv', 'rejections.tsv')
DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs' / 'conformer_mk.yaml'

#: set-level codes, beside the per-member REASON_CODES
SET_REASON_CODES = {
    'unparseable': 'RDKit cannot parse the source SMILES',
    'duplicate_key': 'the constitution already appears at a lower dataset index',
    'encoder_pool_key': "the constitution is in the frozen encoder's training pool",
    'stereo_enumeration': 'stereoisomer enumeration raised, or exceeded its bound',
    'duplicate_stereoisomer': 'this SMILES names the same stereoisomer (stereo_identity) as '
                              'the enumerated name given in the message; not built twice',
    'no_buildable_stereoisomer': 'no stereoisomer built; its isomers failed with different '
                                 'codes (see the isomer rows)',
    'heldout_wider_than_train': 'a held-out molecule with more columns in some block than '
                                'the train maxima, so no train-layout column could hold it',
    'heldout_moves_train_layout': 'a held-out molecule within the train maxima whose presence '
                                  'in the shared layout moves a training member off the '
                                  'columns the run rebuilds from the training file alone',
}
ALL_CODES = {**REASON_CODES, **SET_REASON_CODES}
#: ``pool_rows``'s rule at need 0: the universe is every row (owner decision 2026-09-29)
NO_POOL_RULE = 'no pool (need 0): the encoder pool is not excluded; every row is in the universe'

#: refusals decided by the bond graph alone under force_field mmff (atom count, MMFF typing,
#: MMFF-typed linearity), so every stereoisomer fails them identically and the walk stops at
#: the first instead of re-embedding up to 1024 isomers to learn the same thing.
#: A code that depends on the SPANNING TREE is not one: the tree follows atom numbering, and
#: the canonical SMILES of two stereoisomers number atoms differently.
#:   * `closure_encoding` -- QM9 132157 fails it under one stereo salt and builds two isomers
#:     under another;
#:   * `incomplete_chart_full` -- since the transverse pair and the dummy frame, what is
#:     still held is a collinear frame no dummy can carry (mxtaltools `dummy_frame_refs`: b
#:     must be c's parent and not the root; an unchained dummy's anchor angle must not itself
#:     be linear) and a linear angle framed by one: properties of the tree
#:     (ConformerTorsions.__init__). Before them every alkyne was held whatever its tree,
#:     which is what made this code constitution-level.
CONSTITUTION_CODES = frozenset({'lt4_atoms', 'mmff_typing', 'no_free_dof', 'chart_defect'})


def _hash64(text: str, person: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8,
                                          person=person).digest(), 'big')


def split_hash(key: str, salt: str) -> int:
    return _hash64(f'{salt}\0{key}', b'cset_split')


def constitution_key(smiles: str) -> str:
    """The split key: ``encoder_probe.parent_skeleton`` after explicit hydrogens are dropped.

    On raw QM9 (no explicit H) this IS ``parent_skeleton``. The difference is an identifier:
    an imine keeps the ``[H]`` that defines its E/Z, and ``parent_skeleton`` strips the
    stereo but not the atom, so ``[H]/N=C/C`` would key as ``[H]N=CC`` against its source's
    ``CC=N`` -- a stereoisomer on a different key from its own molecule, which is exactly
    the straddle the split exists to prevent. Every built isomer is checked against its
    molecule's key with this function.
    """
    from rdkit import Chem

    from models.encoder_probe import parent_skeleton
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f'unparseable SMILES: {smiles}')
    Chem.RemoveStereochemistry(m)
    return parent_skeleton(Chem.MolToSmiles(Chem.RemoveHs(m)))


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ source and pool


def read_source(path) -> List[Tuple[int, str]]:
    """``[(dataset_index, smiles)]`` in FILE ORDER.

    A ``.pt`` is the QM9 MolData list itself, indexed by position. A ``.csv`` / ``.tsv``
    needs ``dataset_index`` and ``smiles`` columns -- a slice of QM9 written with its own
    indices, which is what the tests and a quick rebuild use instead of the 360 MB file.
    Either may be gzipped (``.csv.gz`` / ``.tsv.gz``, read through ``gzip``), as the full
    index table configs/conformer_db_sep29/qm9_index.tsv.gz is.
    """
    import gzip

    path = Path(path)
    if path.suffix == '.pt':
        data = torch.load(path, weights_only=False, map_location='cpu')
        return [(i, str(getattr(d, 'smiles'))) for i, d in enumerate(data)]
    gz = path.suffix == '.gz'
    delim = '\t' if (Path(path.stem).suffix if gz else path.suffix) == '.tsv' else ','
    with (gzip.open if gz else open)(path, 'rt', encoding='utf-8', newline='') as f:
        rows = [(int(r['dataset_index']), r['smiles'].strip())
                for r in csv.DictReader(f, delimiter=delim)]
    return rows


def encoder_pool(ckpt_path) -> Tuple[int, dict]:
    """``(need, info)``: how many leading QM9 SMILES the encoder's battery loaded.

    THE CHECKPOINT DOES NOT RECORD ITS TRAINING SET, so it is reconstructed. The checkpoint
    stores (arm, seed, n_train, best_step, best_heldout); exactly one row of one results file
    beside it (models/results/*.json) carries the same five, and that file's `sizes` and
    `meta.n_test` fix what ``encoder_probe.main`` loaded:

        need = int((max(sizes) + n_test) * 1.1) + 64            (encoder_probe.main)

    through ``load_qm9_stereo(need)`` -> ``load_qm9(need)``, the first ``need`` distinct raw
    SMILES of qm9_dataset.pt in file order. The train and encoder-held-out molecules are both
    drawn from those. Refuses when no row, or more than one, matches: an unidentified
    encoder has no known pool.
    """
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ident = {k: ck.get(k) for k in ('arm', 'seed', 'n_train', 'best_step', 'best_heldout')}
    results = Path(ckpt_path).resolve().parent.parent
    hits = []
    for f in sorted(results.glob('*.json')):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                blob = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(blob, dict):
            continue
        for row in blob.get('rows') or []:
            if isinstance(row, dict) and all(row.get(k) == v for k, v in ident.items()):
                hits.append((f, blob))
    if len(hits) != 1:
        raise SystemExit(
            f'{ckpt_path}: {len(hits)} results rows in {results} match the checkpoint '
            f'{ident} ({[str(f.name) for f, _ in hits]}); the encoder pool cannot be '
            f'reconstructed. Pass --pool-need explicitly if you know it')
    f, blob = hits[0]
    sizes, n_test = blob.get('sizes'), (blob.get('meta') or {}).get('n_test')
    if not sizes or n_test is None:
        raise SystemExit(f'{f}: no sizes / meta.n_test; cannot recompute the encoder pool')
    need = int((max(sizes) + int(n_test)) * 1.1) + 64
    return need, {'results_file': str(f), 'sizes': sizes, 'n_test': int(n_test),
                  'meta': blob.get('meta'), 'checkpoint_row': ident,
                  'need_rule': 'int((max(sizes) + n_test) * 1.1) + 64 (encoder_probe.main)'}


def pool_rows(rows: Sequence[Tuple[int, str]], need: int) -> Tuple[set, int, str]:
    """``(pool indices, end, rule)`` -- ``load_qm9``'s selection, replayed on ``rows``.

    Exact when the source starts at file index 0 (the QM9 .pt, or a full index table): the
    first ``need`` DISTINCT raw SMILES, so a duplicate inside the range pushes the end past
    ``need``. A slice that does not start at 0 cannot replay it, and falls back to the
    index rule ``[0, need)`` -- recorded as such in the manifest. ``need`` 0 is NO POOL (the
    default since 2026-09-29): an empty pool ending at index 0, whatever the source.
    """
    if int(need) < 0:
        raise SystemExit(f'pool need {need} is negative')
    if int(need) == 0:
        return set(), 0, NO_POOL_RULE
    if rows and rows[0][0] == 0 and all(i == j for j, (i, _) in enumerate(rows[:need])):
        seen, idx, end = set(), set(), 0
        for i, smi in rows:
            if len(seen) >= need:
                break
            if smi not in seen:
                seen.add(smi)
                idx.add(i)
            end = i + 1
        if len(seen) < need:
            raise SystemExit(f'the source holds {len(seen)} distinct SMILES, fewer than the '
                             f'encoder pool\'s {need}; it is not the file the encoder read')
        return idx, end, 'replayed: first `need` distinct raw SMILES in file order'
    return set(range(need)), need, 'index rule [0, need): the source does not start at 0'


# ------------------------------------------------------------------ split


@dataclass
class Entry:
    key: str
    index: int
    smiles: str                   # the source SMILES of the representative row
    side: str                     # 'train' | 'heldout'
    h: int                        # split hash; the walk order within a side


def plan_split(rows, *, index_min: int, index_max: Optional[int], pool_idx: set,
               salt: str, permille: int):
    """``(entries by side, rejections, n pool keys)``: the universe, deduplicated and split.

    A PURE FUNCTION OF THE SET OF ROWS, not of their order: each key's representative is its
    lowest dataset index, and each side is sorted by (hash, key). So a reordered source
    produces a byte-identical split table.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    rejections = []
    pool_keys = set()
    in_range = []
    for i, smi in rows:
        if i in pool_idx:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                pool_keys.add(constitution_key(smi))
            continue
        if i < index_min or (index_max is not None and i >= index_max):
            continue
        in_range.append((i, smi))
    by_key: Dict[str, Tuple[int, str]] = {}
    for i, smi in sorted(in_range):
        if Chem.MolFromSmiles(smi) is None:
            rejections.append(_rej('molecule', '', i, '', '', 'unparseable', smi))
            continue
        key = constitution_key(smi)
        if key in pool_keys:
            rejections.append(_rej('molecule', key, i, '', '', 'encoder_pool_key',
                                   f'{smi}: constitution {key} is in the encoder pool'))
            continue
        if key in by_key:
            rejections.append(_rej('molecule', key, i, '', '', 'duplicate_key',
                                   f'{smi}: same constitution as dataset index '
                                   f'{by_key[key][0]}'))
            continue
        by_key[key] = (i, smi)
    sides = {'train': [], 'heldout': []}
    for key, (i, smi) in by_key.items():
        h = split_hash(key, salt)
        side = 'heldout' if (h % 1000) < int(permille) else 'train'
        sides[side].append(Entry(key, i, smi, side, h))
    for side in sides:
        sides[side].sort(key=lambda e: (e.h, e.key))
    return sides, rejections, len(pool_keys)


def _rej(level, key, index, side, ident, code, message):
    if code not in ALL_CODES:
        raise KeyError(f'unknown reason code {code!r}')
    return {'level': level, 'key': key, 'dataset_index': index, 'split': side,
            'identifier': ident, 'reason_code': code, 'message': ' '.join(str(message).split())}


# ------------------------------------------------------------------ one molecule


@dataclass
class Built:
    entry: Entry
    members: List[Member] = field(default_factory=list)
    isomer_rank: Dict[str, int] = field(default_factory=dict)   # 0 = pick, 1 = its mirror, ...
    n_isomers: int = 0
    mirror_of: Dict[str, str] = field(default_factory=dict)     # name -> its mirror's name


def stereo_plan(key: str):
    """``(names, mirror_of, rejections-as-tuples)`` for one constitution.

    ``names`` are the enumerated stereoisomers (``stereoisomer_classes``), one per
    ``stereo_identity``. ``mirror_of[name]`` is the enumerated name whose identity is that of
    ``name`` with every tetrahedral tag inverted -- looked up by IDENTITY, since the writer's
    canonical form of the inverted string can name another isomer on a cage; absent when the
    mirror is not an enumerated name, and equal to ``name`` when it is achiral. Strings that
    name the same isomer as an enumerated name come back as ``(identifier, code, message)``
    for the isomer rows.
    """
    from models.encoder_probe import mirror_smiles

    classes = stereoisomer_classes(key)
    names = [n for n, _, _ in classes]
    by_identity = {idt: n for n, idt, _ in classes}
    mirror_of = {}
    for n in names:
        m = by_identity.get(smiles_identity(mirror_smiles(n)))
        if m is not None:
            mirror_of[n] = m
    rows = []
    for n, _, others in classes:
        for o in others:
            rows.append((o, 'duplicate_stereoisomer', f'names the same stereoisomer as {n}'))
    return names, mirror_of, rows


def build_molecule(entry: Entry, energy_kw: dict, *, bundle, cap: Optional[int],
                   stereo_salt: str, check: bool = True):
    """``(Built or None, rejections)`` -- the molecule's conditions, or why there are none.

    TERMINATION: the walk is over the enumerated list (``stereoisomer_classes`` refuses more
    than 1024 assignments) plus at most one mirror per pick, and each identifier is
    attempted at most once.
    """
    rej = []
    try:
        isos, mirror_of, extra = stereo_plan(entry.key)
    except Exception as exc:                                   # noqa: BLE001 - recorded
        rej.append(_rej('molecule', entry.key, entry.index, entry.side, '',
                        'stereo_enumeration', f'{type(exc).__name__}: {exc}'))
        return None, rej
    for ident, code, msg in extra:
        rej.append(_rej('isomer', entry.key, entry.index, entry.side, ident, code, msg))
    order = sorted(isos, key=lambda s: (_hash64(f'{stereo_salt}\0{entry.key}\0{s}',
                                                b'cset_stereo'), s))
    out = Built(entry, n_isomers=len(isos), mirror_of=mirror_of)
    tried, codes = set(), []

    def attempt(ident) -> bool:
        tried.add(ident)
        try:
            mb = build_member(ident, ident, energy_kw, bundle=bundle, carrier=True,
                              check=check)
        except MemberRefused as exc:
            codes.append(exc.code)
            rej.append(_rej('isomer', entry.key, entry.index, entry.side, ident,
                            exc.code, exc.message))
            return False
        if constitution_key(ident) != entry.key:
            raise RuntimeError(f'{ident} keys as {constitution_key(ident)}, its molecule as '
                               f'{entry.key}: a stereoisomer would sit on another split key')
        out.isomer_rank[ident] = len(out.members)
        out.members.append(mb)
        return True

    for ident in order:
        if cap is not None and len(out.members) >= cap:
            break
        if ident in tried:
            continue
        ok = attempt(ident)
        # a constitution-level refusal met BEFORE any isomer has built ends the walk; after
        # one has built it cannot occur (the code would have refused that isomer too)
        if not ok and not out.members and codes[-1] in CONSTITUTION_CODES:
            break
        if ok and (cap is None or len(out.members) < cap):
            mirror = mirror_of.get(ident)
            if mirror is not None and mirror != ident and mirror not in tried:
                attempt(mirror)
    if out.members:
        return out, rej
    # THE MOLECULE'S REASON: a constitution-level code wins (no isomer could pass it, so it
    # is the blocker even when impossible isomers failed to embed first), then a code every
    # isomer shared, then an embedding limitation (an isomer shown to exist failed only to
    # embed, so the molecule is lost to ETKDG, not to physics), else the mixed case -- its
    # per-isomer codes are on the isomer rows
    const = [c for c in codes if c in CONSTITUTION_CODES]
    code = (const[0] if const else codes[0] if len(set(codes)) == 1
            else 'embed_failed_realisable' if 'embed_failed_realisable' in codes
            else 'no_buildable_stereoisomer')
    rej.append(_rej('molecule', entry.key, entry.index, entry.side, '', code,
                    f'{len(tried)} of {len(isos)} stereoisomer(s) attempted, none built'
                    f' (codes {sorted(set(codes))})'))
    return None, rej


def walk_side(entries: Sequence[Entry], n: Optional[int], energy_kw, *, bundle, cap,
              stereo_salt, check=True, label=''):
    """The first ``n`` molecules of ``entries`` (hash order) that yield a condition.

    ``n=None`` walks the whole side. Bounded by ``len(entries)``.
    """
    kept, rej, attempted = [], [], 0
    t0 = time.time()
    for e in entries:
        if n is not None and len(kept) >= n:
            break
        attempted += 1
        b, r = build_molecule(e, energy_kw, bundle=bundle, cap=cap,
                              stereo_salt=stereo_salt, check=check)
        rej.extend(r)
        if b is not None:
            kept.append(b)
        if attempted % 50 == 0:
            print(f'  {label}: {attempted} attempted, {len(kept)} kept, '
                  f'{time.time() - t0:.0f} s', flush=True)
    return kept, rej, attempted


def train_placement_change(train_layout, layout, train_idents) -> str:
    """``''`` when ``layout`` places every training member exactly as ``train_layout`` does.

    ``train_layout`` is ``CarrierLayout`` over the training members alone -- what the run's
    ``MultiConformerTorsions`` rebuilds from ``conditions_train.pt`` -- and ``layout`` one
    over more members. Same ``K``, the same region code per carrier column (``free_block``:
    what the run's periodic mask and box wall read) and the same carrier columns, in member
    order, for every training member (``cols``: what the file's reconstruction map points
    at). ``is_identity`` and ``offsets`` may differ; neither is written into a row. Otherwise
    the first differences, as text.
    """
    out = []
    if int(layout.K) != int(train_layout.K):
        out.append(f'K {train_layout.K} -> {layout.K}')
    elif not np.array_equal(np.asarray(layout.free_block), np.asarray(train_layout.free_block)):
        out.append(f'region codes {np.asarray(train_layout.free_block).tolist()} -> '
                   f'{np.asarray(layout.free_block).tolist()}')
    moved = [i for i in train_idents if i not in layout.cols
             or not np.array_equal(layout.cols[i], train_layout.cols[i])]
    if moved:
        out.append(f'{len(moved)} training member(s) moved, e.g. {moved[0]}: '
                   f'{np.asarray(train_layout.cols[moved[0]]).tolist()} -> '
                   f'{np.asarray(layout.cols[moved[0]]).tolist() if moved[0] in layout.cols else None}')
    return '; '.join(out)


def admit_heldout(train_members: Dict[str, Member], held: Sequence[Built]):
    """``(admitted, rejections, train block widths)``: drop held-out molecules the training
    layout cannot hold.

    ONE LAYOUT SPANS BOTH FILES, and the run builds its layout from the TRAINING file alone
    (``ConformerModeller._condition_set_molecules`` reads molecules_path, and
    ``MultiConformerTorsions`` calls ``CarrierLayout`` on those members). So a held-out
    molecule is admitted only when the shared layout places every training member where the
    training-only layout does (``train_placement_change``), and is dropped and named
    otherwise:

      * ``heldout_wider_than_train`` -- more columns in some region than every training
        member. It would either widen the layout past what the run builds, or own a column
        no training row ever populates, scored only off-distribution by the flat backward
        net;
      * ``heldout_moves_train_layout`` -- within the widths, but its presence moves a
        training member. ``CarrierLayout`` keeps the IDENTITY when every member has the same
        ``_free_block``, e.g. a training set of one nitrile's stereoisomers, whose transverse
        v sits among its phi columns; a held-out molecule with other codes makes the shared
        layout a region-ordered carrier, which moves that v into the theta region. The
        run's identity layout wraps the column the file's map now reads as a phi, and walls
        the one it reads as v, with no error at run time.

    Drop-after-walk rather than skip-during-walk keeps held-out rungs NESTED: a larger
    training rung only widens the maxima, and can only keep the training identity or end
    it, and a carrier places by region rank, so nothing within the maxima moves it -- a
    larger rung only ever re-admits.

    Both rules ask ``CarrierLayout`` itself (a one-member layout's widths are that member's
    region counts), so a layout that changes how it places needs no change here. A
    molecule's stereoisomers share their region counts, so its first member decides the
    width rule; the placement rule is asked with all of its members.
    """
    from energies.conformer_carrier import CarrierLayout

    tcodes = {i: m.energy for i, m in train_members.items()}
    train_layout = CarrierLayout(tcodes)
    width = train_layout.block_width
    admitted, rej = [], []
    for b in held:
        own = CarrierLayout({b.members[0].identifier: b.members[0].energy}).block_width
        if len(own) != len(width) or any(o > w for o, w in zip(own, width)):
            rej.append(_rej('molecule', b.entry.key, b.entry.index, 'heldout', '',
                            'heldout_wider_than_train',
                            f'block counts {list(own)} exceed the train maxima {list(width)}'))
            continue
        moved = train_placement_change(
            train_layout, CarrierLayout({**tcodes, **{m.identifier: m.energy
                                                      for m in b.members}}), tcodes)
        if moved:
            rej.append(_rej('molecule', b.entry.key, b.entry.index, 'heldout', '',
                            'heldout_moves_train_layout', moved))
            continue
        admitted.append(b)
    return admitted, rej, list(width)


# ------------------------------------------------------------------ files


def write_split_table(path, sides):
    """``split.tsv``: every key of the universe, its side, and its walk position (hash)."""
    _write_tsv(path, ['key', 'dataset_index', 'side', 'hash'],
               [{'key': e.key, 'dataset_index': e.index, 'side': s, 'hash': f'{e.h:016x}'}
                for s in ('train', 'heldout') for e in sides[s]])


def rebuilt_layout(batch) -> Tuple[int, Dict[str, np.ndarray]]:
    """``(K, {identifier: the carrier columns its row owns, ascending})`` off a written batch.

    From ``n_torsions`` and each graph's ``state_mask`` alone -- the fields the run's energy
    checks rows against -- so it is a reading of what the FILE says, not of the layout
    object that wrote it. It is the column SET: which member column sits in which of those
    columns is the reconstruction map ``ctree_{r,th,ph}_col``, and is not in member order
    once a transverse column sits in the theta region (``verify_conditions_file``).
    """
    ks = torch.as_tensor(batch.n_torsions).reshape(-1)
    if not bool((ks == ks[0]).all()):
        raise ValueError(f'mixed n_torsions {sorted(set(ks.tolist()))} in a carrier file')
    sm = torch.as_tensor(batch.state_mask).reshape(len(ks), -1).bool()
    return int(ks[0]), {ident: np.flatnonzero(sm[i].numpy())
                        for i, ident in enumerate(list(batch.identifier))}


#: the per-atom fields that name a STATE COLUMN, the ones ``carrier_pad_condition`` remaps
_COLUMN_MAP_FIELDS = ('ctree_r_col', 'ctree_th_col', 'ctree_ph_col')


def verify_conditions_file(path, layout, members: List[dict], own: Dict[str, object], *,
                           covers_all: bool):
    """Refuse a written conditions file that, re-read from disk, is not what its layout says.

    ``layout`` is the ``CarrierLayout`` the file must be read through, asked directly; no
    column map is re-derived here. ``main`` passes, for the training file, the layout over
    the training members alone -- what the run's ``MultiConformerTorsions`` rebuilds from
    it -- and for the held-out file the shared layout that wrote both.
    ``members`` are the manifest rows, ``own`` each member's member-width condition graph
    (``Member.condition``). Checked, per written row:

      * ``K`` and the identifier list and its order, and each member's k and pad count,
        against ``layout.K`` and the manifest rows;
      * ``state_mask`` against ``layout.valid(ident)``, the columns the member owns -- the
        comparison ``MultiConformerTorsions._resolve_rows`` makes at run time;
      * the reconstruction map ``ctree_{r,th,ph}_col`` against the member's own map sent
        through ``layout.cols[ident]``. This one sees ORDER, which the mask cannot: a
        transverse u or v sits in the theta region, so a member's carrier columns are not in
        member order, and two members can share every region width (a pure permutation,
        K = k, no pads), where a member-order file owns exactly the right columns and reads
        them in the wrong order.

    The train file must also cover every carrier column: block widths are maxima over the
    training members, so each column belongs to at least one of them, and a column no
    training row populates would be scored only off-distribution.
    """
    from energies.conformer_data import require_conformer_fields

    blob = torch.load(path, weights_only=False, map_location='cpu')
    batch = blob['prior']
    require_conformer_fields(batch)
    K, cols = rebuilt_layout(batch)
    problems = []
    if K != layout.K:
        problems.append(f'K {K} != layout K {layout.K}')
    want = [m['identifier'] for m in members]
    if list(cols) != want:
        problems.append(f'identifiers differ from the manifest ({len(cols)} vs {len(want)})')
    row_of = {ident: i for i, ident in enumerate(list(batch.identifier))}
    ptr = torch.as_tensor(batch.ptr).reshape(-1)
    for m in members:
        ident = m['identifier']
        c = cols.get(ident)
        if c is None:
            continue
        if len(c) != m['k'] or K - len(c) != m['n_pad']:
            problems.append(f'{ident}: k {len(c)} / pads {K - len(c)} against '
                            f'manifest {m["k"]} / {m["n_pad"]}')
            continue
        if ident not in layout.cols:
            problems.append(f'{ident}: not a member of the layout')
            continue
        if K != layout.K:
            continue                       # reported above; the column checks need one width
        if not np.array_equal(np.isin(np.arange(K), c), layout.valid(ident)):
            problems.append(f'{ident}: carrier columns differ from the layout (owns '
                            f'{c.tolist()}, layout {np.flatnonzero(layout.valid(ident)).tolist()})')
            continue
        at = slice(int(ptr[row_of[ident]]), int(ptr[row_of[ident] + 1]))
        to_carrier = torch.as_tensor(layout.cols[ident], dtype=torch.long)
        for name in _COLUMN_MAP_FIELDS:
            mine = torch.as_tensor(getattr(own[ident], name)).reshape(-1).long()
            expect = torch.where(mine >= 0, to_carrier[mine.clamp_min(0)], mine)
            got = torch.as_tensor(getattr(batch, name)).reshape(-1)[at].long()
            if got.shape != expect.shape or not torch.equal(got, expect):
                problems.append(f'{ident}: {name} is not its member map through the layout '
                                f'(the reconstruction map reads the carrier columns in '
                                f'another order)')
                break
    if covers_all and cols:
        union = np.zeros(K, dtype=bool)
        for c in cols.values():
            union[c] = True
        if not union.all():
            problems.append(f'carrier columns {np.flatnonzero(~union).tolist()} are owned by '
                            f'no member')
    if problems:
        raise SystemExit(f'{path}: the written file disagrees with the manifest, refusing '
                         f'it:\n  ' + '\n  '.join(problems[:20]))


def _write_then_verify(write, final: Path, verify):
    """Write to ``final.tmp``, verify the re-read, then rename -- or delete and raise."""
    tmp = final.with_name(final.name + '.tmp')
    write(tmp)
    try:
        verify(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, final)


def _cast_floats(batch, dtype):
    """Every float tensor on ``batch`` to ``dtype``, in place (the modeller's idiom)."""
    for key, val in list(batch._store.items()):
        if torch.is_tensor(val) and val.is_floating_point() and val.dtype != dtype:
            batch[key] = val.to(dtype)
    return batch


def build_prior(padded: Dict[str, object], members: Dict[str, Member], layout, n: int, *,
                prior, relax_steps: int, seed: int, dtype):
    """``(batch, stats)``: ``n`` prior rows per training condition, in carrier form.

    EQUAL ROWS PER CONDITION, so the buffer represents the set rather than whichever
    molecule drew first. Each state is rounded to the STORAGE dtype before it is scored, so
    the stored energy is the energy of the stored state, not of a neighbour that no longer
    exists after the cast. One collate over every row, not an append loop.
    """
    from energies.conformer_data import bake_energies, collate_conditions, wrap_state
    from energies.conformer_prior_draw import draw_member_prior, member_prior_seed

    rows, stats = [], {}
    periodic = [bool(b == 2) for b in layout.free_block]
    for ident, pm in padded.items():
        en = members[ident].energy
        rng = np.random.default_rng(member_prior_seed(ident, seed))
        x, _, _ = draw_member_prior(en, n, rng, relax_steps=relax_steps, prior=prior)
        x = wrap_state(x, en.periodic_dims).to(dtype).to(en.dtype)
        with torch.no_grad():
            e = bake_energies(en, x)
        xc = wrap_state(layout.to_carrier(ident, x), periodic)
        for i in range(n):
            row = pm.__copy__()
            row.torsion_state = xc[i:i + 1]
            row.conformer_energy = e[i:i + 1]
            row.identifier = ident
            rows.append(row)
        stats[ident] = {'energy_median': float(e.median()),
                        'energy_p90': float(torch.quantile(e, 0.9)),
                        'relax_steps': int(relax_steps)}
    batch = collate_conditions(rows, require_state=True)
    return _cast_floats(batch, dtype), stats


def verify_prior_file(path, layout, members: Dict[str, Member], n: int):
    """Refuse a prior file whose width, pads, row counts or stored energies are wrong.

    Width K; pad columns exactly 0; exactly ``n`` rows per condition; and EVERY row
    re-scored through its member from the STORED state, the result cast to the storage
    dtype and required BIT-EQUAL to the stored energy. Bit-equal is attainable because
    ``build_prior`` scores the storage-rounded state in the same n-row batch this re-scores,
    and it is what makes the rounding observable. Scoring the unrounded state instead
    shifts the energy by a median 2e-7 to 2e-6 kcal/mol, below a spot check's tolerance,
    yet changes the stored float32 value of 424 of 1,280 prior rows (5 molecules at `full`,
    mmff, 256 draws each; the largest shift 0.06 kcal/mol, on a clashing draw).
    """
    from energies.conformer_data import bake_energies

    b = torch.load(path, weights_only=False, map_location='cpu')['equalized_prior']
    x = torch.as_tensor(b.torsion_state)
    e = torch.as_tensor(b.conformer_energy).reshape(-1)
    rows_of: Dict[str, List[int]] = {}
    for j, s in enumerate(b.identifier):
        rows_of.setdefault(s, []).append(j)
    problems = []
    if set(rows_of) != set(members):
        problems.append(f'identifiers {sorted(set(rows_of) ^ set(members))[:5]} are in only '
                        f'one of the prior file and the training conditions')
    if x.shape[1] != layout.K:
        problems.append(f'torsion_state width {x.shape[1]} != K {layout.K}')
    sm = torch.as_tensor(b.state_mask).bool()
    if sm.shape != x.shape or bool((x[~sm] != 0).any()):
        problems.append('non-zero pad columns (or a state_mask of another shape)')
    for ident, en in ((i, m.energy) for i, m in members.items()):
        rows = rows_of.get(ident, [])
        if len(rows) != n:
            problems.append(f'{ident}: {len(rows)} rows, expected {n}')
            continue
        if x.shape[1] != layout.K:
            continue
        xs = layout.from_carrier(ident, x[rows].to(en.dtype))
        with torch.no_grad():
            e2 = bake_energies(en, xs).to(e.dtype)
        bad = torch.nonzero(e[rows] != e2).reshape(-1)
        if len(bad):
            j = int(bad[0])
            problems.append(f'{ident}: {len(bad)} of {n} stored energies differ from the '
                            f're-score of the stored state, e.g. row {rows[j]}: '
                            f'{float(e[rows[j]])!r} against {float(e2[j])!r}')
    if problems:
        raise SystemExit(f'{path}: prior file refused:\n  ' + '\n  '.join(problems[:20]))


def _write_tsv(path, header, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            f.write('\t'.join('' if r.get(h) is None else str(r[h]) for h in header) + '\n')


def _git(repo: Path, *args, timeout=30) -> Optional[str]:
    """A git query, bounded: ``None`` when git is absent, fails, or exceeds ``timeout`` s."""
    try:
        r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                           timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.decode('utf-8', 'replace') if r.returncode == 0 else None


def _provenance_git(config: Path) -> dict:
    here = Path(__file__).resolve().parent
    head = _git(here, 'rev-parse', 'HEAD')
    status = _git(here, 'status', '--porcelain', '--untracked-files=no')
    out = {'head': head.strip() if head else None,
           'dirty': None if status is None else bool(status.strip())}
    top = _git(here, 'rev-parse', '--show-toplevel')
    if top:
        try:
            rel = config.resolve().relative_to(Path(top.strip()).resolve()).as_posix()
            blob = subprocess.run(['git', '-C', top.strip(), 'show', f'HEAD:{rel}'],
                                  capture_output=True, timeout=30)
            # line endings normalised: a CRLF checkout (core.autocrlf) of an unchanged
            # file differs from its blob byte for byte
            norm = lambda b: b.replace(b'\r\n', b'\n')
            out['config_matches_head'] = (blob.returncode == 0 and norm(blob.stdout)
                                          == norm(config.read_bytes()))
        except (ValueError, OSError, subprocess.TimeoutExpired):
            out['config_matches_head'] = None
    return out


# ------------------------------------------------------------------ main


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--source', type=Path, default=Path(QM9_PATH),
                    help='the QM9 MolData .pt, or a .csv/.tsv with dataset_index,smiles')
    ap.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                    help='run config whose energy_config supplies the member kwargs, '
                         'filtered by the ConformerTorsions signature as the run filters them')
    ap.add_argument('--n-train', required=True,
                    help="molecules on the train side, or 'all'")
    ap.add_argument('--n-heldout', required=True,
                    help="molecules on the held-out side (before the width drop), or 'all'")
    ap.add_argument('--heldout-permille', type=int, default=100)
    ap.add_argument('--split-salt', default='conformer_set_v1')
    ap.add_argument('--index-min', type=int, default=None,
                    help='first dataset index of the universe; default and minimum: the end '
                         "of the excluded encoder pool, 0 when none is excluded")
    ap.add_argument('--index-max', type=int, default=None, help='exclusive')
    ap.add_argument('--max-stereoisomers-per-molecule', default='2',
                    help="conditions per molecule, or 'all'. A chiral pick brings its mirror")
    ap.add_argument('--stereo-salt', default='')
    ap.add_argument('--encoder-ckpt', type=Path, default=None,
                    help='default models/encoder_cache.DEFAULT_CKPT')
    ap.add_argument('--no-encoder', action='store_true',
                    help='no embeddings (test and debugging builds)')
    ap.add_argument('--pool-need', type=int, default=0,
                    help="leading distinct QM9 SMILES excluded as the encoder's training pool. "
                         '0 (default, owner decision 2026-09-29) excludes none: the universe '
                         "is all of QM9. N > 0 excludes the first N (pool_rows) and every "
                         'constitution among them')
    ap.add_argument('--pool-from-encoder', action='store_true',
                    help="exclude the encoder's pool, its `need` derived from the encoder's "
                         'results file (encoder_pool); a --pool-need above 0 must then agree')
    ap.add_argument('--prior-rows-per-condition', type=int, default=0,
                    help='write prior_train.pt with this many rows per training condition')
    ap.add_argument('--internal-prior', type=Path, default=None,
                    help="default: the config's energy_config.internal_prior_path")
    ap.add_argument('--prior-relax-steps', type=int, default=None,
                    help="default: the config's energy_config.prior_relax_steps, else 0")
    ap.add_argument('--prior-seed', type=int, default=0)
    ap.add_argument('--prior-dtype', choices=('float32', 'float64'), default='float32')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--no-check', action='store_true',
                    help='skip the graph-vs-energy geometry checks (don\'t)')
    ap.add_argument('--force', action='store_true',
                    help='overwrite an --out-dir that already holds a manifest')
    return ap.parse_args(argv)


def _n(v) -> Optional[int]:
    return None if str(v).lower() == 'all' else int(v)


def main(argv=None):
    args = parse_args(argv)
    from rdkit import RDLogger, rdBase
    RDLogger.DisableLog('rdApp.*')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    t_start = time.time()

    out = args.out_dir
    if (out / 'manifest.json').exists() and not args.force:
        raise SystemExit(f'{out} already holds a manifest; pass --force to overwrite it')
    out.mkdir(parents=True, exist_ok=True)
    # A REBUILD CLEARS EVERY ARTIFACT THIS BUILDER WRITES, the manifest first. Otherwise a
    # rebuild without --prior-rows-per-condition leaves the previous prior_train.pt beside a
    # manifest that does not list it, and an interrupted rebuild leaves the old manifest
    # describing new files. Only these names are touched.
    for name in ('manifest.json', *ARTIFACT_NAMES):
        for p in (out / name, out / f'{name}.tmp'):
            p.unlink(missing_ok=True)

    energy_kw, ec = energy_kwargs_from_config(args.config)
    energy_kw.setdefault('level', 'full')
    energy_kw.setdefault('force_field', 'mmff')
    print(f'energy kwargs from {args.config}: {energy_kw}')
    cap = _n(args.max_stereoisomers_per_molecule)
    if cap is not None and cap < 1:
        raise SystemExit('--max-stereoisomers-per-molecule must be >= 1 or all')

    # ---- encoder, and (only when asked) the pool it was trained on
    bundle, enc_info, pool_info = None, None, {}
    need = int(args.pool_need)
    if need < 0:
        raise SystemExit('--pool-need must be >= 0')
    from models import encoder_cache
    ckpt = args.encoder_ckpt or Path(encoder_cache.DEFAULT_CKPT)
    if args.pool_from_encoder:
        need_meta, pool_info = encoder_pool(ckpt)
        if need and need != need_meta:
            raise SystemExit(f'--pool-need {need} disagrees with the encoder metadata '
                             f'({need_meta}, {pool_info["results_file"]})')
        need = need_meta
    if not args.no_encoder:
        bundle = encoder_cache.load_encoder(str(ckpt), device='cpu')
        enc_info = {k: bundle[k] for k in ('arm', 'hidden', 'layers', 'k', 'attention',
                                           'ckpt_path', 'sha256')}
        enc_info['embedding_dim'] = 2 * int(bundle['hidden'])
        print(f"encoder {bundle['arm']} @ {bundle['sha256'][:12]}")
    print(f'encoder pool: need {need}'
          + (f" from {pool_info['results_file']}" if pool_info else
             ' (none excluded)' if need == 0 else ' (given)'))

    rows = read_source(args.source)
    pool_idx, pool_end, pool_rule = pool_rows(rows, need)
    index_min = pool_end if args.index_min is None else int(args.index_min)
    if index_min < pool_end:
        raise SystemExit(f'--index-min {index_min} reaches into the encoder pool, which '
                         f'ends at dataset index {pool_end}')
    sides, set_rej, n_pool_keys = plan_split(
        rows, index_min=index_min, index_max=args.index_max, pool_idx=pool_idx,
        salt=args.split_salt, permille=args.heldout_permille)
    n_universe_rows = sum(1 for i, _ in rows if i >= index_min
                          and (args.index_max is None or i < args.index_max))
    print(f'universe: {n_universe_rows} rows at dataset index >= {index_min}; '
          f'{len(sides["train"])} train / {len(sides["heldout"])} held-out keys; '
          f'{len(set_rej)} rows refused before any build')
    write_split_table(out / 'split.tsv', sides)

    # ---- build
    check = not args.no_check
    walk = dict(bundle=bundle, cap=cap, stereo_salt=args.stereo_salt, check=check)
    train, rej_t, att_t = walk_side(sides['train'], _n(args.n_train), energy_kw,
                                    label='train', **walk)
    held, rej_h, att_h = walk_side(sides['heldout'], _n(args.n_heldout), energy_kw,
                                   label='heldout', **walk)
    rejections = set_rej + rej_t + rej_h
    if not train:
        raise SystemExit('no training molecule built; nothing to write')

    from energies.conformer_carrier import (TRANSVERSE, CarrierLayout, carrier_pad_condition,
                                            check_carrier_convention)
    tmembers = {mb.identifier: mb for b in train for mb in b.members}
    admitted, rej_w, width = admit_heldout(tmembers, held)
    rejections += rej_w
    hmembers = {mb.identifier: mb for b in admitted for mb in b.members}
    clash = set(tmembers) & set(hmembers)
    if clash or ({b.entry.key for b in train} & {b.entry.key for b in admitted}):
        raise SystemExit(f'a key or identifier is on both sides of the split: '
                         f'{sorted(clash)[:5]}')
    # TWO LAYOUTS, one placement. `layout`, over both sides, writes both files; `run_layout`,
    # over the training members alone, is the one the run rebuilds from conditions_train.pt
    # (MultiConformerTorsions -> CarrierLayout). admit_heldout admitted only held-out
    # molecules that leave the training placement unchanged; checked again over the whole
    # admitted set, and the training file and the prior are verified against `run_layout`.
    run_layout = CarrierLayout({i: m.energy for i, m in tmembers.items()})
    layout = CarrierLayout({**{i: m.energy for i, m in tmembers.items()},
                            **{i: m.energy for i, m in hmembers.items()}})
    if list(layout.block_width) != list(width):
        raise SystemExit(f'held-out members widened the layout {width} -> '
                         f'{layout.block_width} past the width check')
    moved = train_placement_change(run_layout, layout, tmembers)
    if moved:
        raise SystemExit(f'the admitted held-out members move the training placement past '
                         f'admit_heldout ({moved}); the run would read the training file '
                         f'through another layout')
    print(layout.describe().splitlines()[0])

    # ---- pad, check, collate (one R over both files, so the two collate alike)
    allm = {**tmembers, **hmembers}
    R = max((m.atoms.shape[1] for m in allm.values() if m.atoms is not None), default=1)
    padded = {}
    for ident, m in allm.items():
        pm = carrier_pad_condition(m.condition, layout, ident, m.energy,
                                   atoms=m.atoms, mask=m.mask,
                                   R=R if m.atoms is not None else None)
        if check:
            try:
                check_carrier_convention(pm, layout, ident, m.energy)
            except AssertionError as exc:
                # not a molecule to drop: build_member passed this member through the same
                # check in a one-member layout, so the SHARED layout is what is wrong
                raise SystemExit(f'{ident} passes the carrier check alone and fails it in the '
                                 f'shared layout, a CarrierLayout defect: {exc}') from None
        padded[ident] = pm

    rank_of = {}
    for side, blist in (('train', train), ('heldout', admitted)):
        for rank, b in enumerate(blist):
            for mb in b.members:
                rank_of[mb.identifier] = (side, rank, b)
    from energies.conformer_data import bake_energies
    from models.graph_encodings import graph_from_smiles

    def member_row(ident):
        side, rank, b = rank_of[ident]
        m = allm[ident]
        own = CarrierLayout({ident: m.energy}).block_width
        mirror = b.mirror_of.get(ident, ident)
        # the member's own reference (state 0), raw potential at T = 1: the two halves of a
        # mirror pair are independent ETKDG embeddings, not reflections, and this is where
        # their offset shows
        with torch.no_grad():
            e_ref = float(bake_energies(m.energy, torch.zeros(1, int(m.energy.data_ndim),
                                                              dtype=m.energy.dtype)))
        # ENCODER GROUP: the lowest isomer rank whose frozen embedding is bit-identical to this
        # one. The encoder reads bond order but not E/Z, and CIP R/S but not r/s or ring
        # cis/trans, so such isomers share an embedding and anything that reads only the
        # embedding cannot tell them apart. Recorded, not refused: each is still its own
        # condition (the per-coordinate features are where E/Z has to be carried).
        grp = [j for j, o in enumerate(b.members)
               if getattr(o.condition, 'embedding', None) is not None
               and getattr(m.condition, 'embedding', None) is not None
               and torch.equal(o.condition.embedding, m.condition.embedding)]
        return {'identifier': ident, 'key': b.entry.key, 'dataset_index': b.entry.index,
                'split': side, 'molecule_rank': rank, 'n_stereoisomers': b.n_isomers,
                'isomer_rank': b.isomer_rank[ident],
                'mirror': mirror if mirror != ident and mirror in b.isomer_rank else '',
                'n_atoms': int(m.energy.spec.n_atoms), 'k': int(layout.k(ident)),
                'n_pad': int(layout.K - layout.k(ident)),
                # REGION widths r/theta/phi, the ones admit_heldout compares: a transverse u or
                # v counts in theta, and n_transverse says how many of the theta are u or v
                'blocks': '/'.join(str(int(v)) for v in own),
                'n_transverse': int((layout.kind(ident) == TRANSVERSE).sum()),
                'reference_energy': f'{e_ref:.6f}',
                'encoder_parity_atoms': int(np.count_nonzero(graph_from_smiles(ident)[2])),
                'encoder_group': min(grp) if grp else ''}

    mrows = {'train': [member_row(i) for i in tmembers],
             'heldout': [member_row(i) for i in hmembers]}
    own_maps = {i: m.condition for i, m in allm.items()}      # member-width, before padding

    from energies.conformer_data import collate_conditions, save_condition_file
    files = {}
    for side, fname in (('train', 'conditions_train.pt'), ('heldout', 'conditions_heldout.pt')):
        idents = [r['identifier'] for r in mrows[side]]
        if not idents:
            print(f'no {side} conditions; {fname} not written')
            continue
        batch = collate_conditions([padded[i] for i in idents])
        # the training file against the layout the RUN rebuilds from it, not the shared one
        # that wrote it: the two agree by the check above, and this re-asks it of the file
        _write_then_verify(lambda p, b=batch: save_condition_file(b, p), out / fname,
                           lambda p, s=side: verify_conditions_file(
                               p, run_layout if s == 'train' else layout, mrows[s], own_maps,
                               covers_all=(s == 'train')))
        files[fname] = out / fname
        print(f'wrote {out / fname}: {len(idents)} conditions, K = {layout.K}')

    # ---- optional prior, training conditions only
    prior_info = None
    internal_prior = args.internal_prior or ec.get('internal_prior_path')
    if args.prior_rows_per_condition > 0:
        n = int(args.prior_rows_per_condition)
        if n < 2:
            raise SystemExit('--prior-rows-per-condition must be >= 2 (attach_states)')
        if internal_prior is None or not Path(internal_prior).exists():
            raise SystemExit(f'a prior file needs the fitted InternalPrior; got '
                             f'{internal_prior}')
        prior = torch.load(internal_prior, weights_only=False)
        relax = (args.prior_relax_steps if args.prior_relax_steps is not None
                 else int(ec.get('prior_relax_steps', 0) or 0))
        dtype = getattr(torch, args.prior_dtype)
        tpadded = {i: padded[i] for i in tmembers}
        pbatch, pstats = build_prior(tpadded, tmembers, run_layout, n, prior=prior,
                                     relax_steps=relax, seed=args.prior_seed, dtype=dtype)
        from energies.conformer_data import save_prior_file
        _write_then_verify(
            lambda p: save_prior_file(pbatch, p, source='InternalPrior',
                                      n_per_molecule=n, format=FORMAT),
            out / 'prior_train.pt',
            lambda p: verify_prior_file(p, run_layout, tmembers, n))
        files['prior_train.pt'] = out / 'prior_train.pt'
        prior_info = {'path': str(Path(internal_prior).resolve()),
                      'sha256': _sha256(internal_prior), 'rows_per_condition': n,
                      'relax_steps': relax, 'seed': args.prior_seed, 'dtype': args.prior_dtype,
                      'per_condition': pstats}
        print(f'wrote {out / "prior_train.pt"}: {n} rows x {len(tmembers)} conditions')

    # ---- tables and manifest
    header = ['identifier', 'key', 'dataset_index', 'split', 'molecule_rank',
              'n_stereoisomers', 'isomer_rank', 'mirror', 'n_atoms', 'k', 'n_pad', 'blocks',
              'n_transverse', 'reference_energy', 'encoder_parity_atoms', 'encoder_group']
    _write_tsv(out / 'molecules.tsv', header, mrows['train'] + mrows['heldout'])
    _write_tsv(out / 'rejections.tsv', ['level', 'key', 'dataset_index', 'split',
                                        'identifier', 'reason_code', 'message'], rejections)
    for name in ('split.tsv', 'molecules.tsv', 'rejections.tsv'):
        files[name] = out / name

    mol_rej = [r for r in rejections if r['level'] == 'molecule']
    kept_mols = len(train) + len(admitted)
    n_attempted = att_t + att_h
    manifest = {
        'format': FORMAT,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'argv': sys.argv if argv is None else ['build_conformer_set.py', *argv],
        'git': _provenance_git(args.config),
        'versions': {'rdkit': rdBase.rdkitVersion, 'torch': torch.__version__,
                     'numpy': np.__version__, 'python': platform.python_version()},
        'source': {'path': str(Path(args.source).resolve()), 'sha256': _sha256(args.source),
                   'n_rows': len(rows), 'index_min': index_min, 'index_max': args.index_max,
                   'universe_rows': n_universe_rows},
        'encoder': enc_info,
        'encoder_pool': {'need': need, 'end': pool_end, 'selection': pool_rule,
                         'n_pool_keys': n_pool_keys, **pool_info},
        'energy': {'config': str(Path(args.config).resolve()),
                   'config_sha256': _sha256(args.config), 'kwargs': energy_kw},
        'split': {'key': 'build_conformer_set.py::constitution_key', 'salt': args.split_salt,
                  'heldout_permille': args.heldout_permille,
                  'keys': {s: len(sides[s]) for s in sides},
                  'requested': {'train': args.n_train, 'heldout': args.n_heldout},
                  'attempted': {'train': att_t, 'heldout': att_h}},
        'seeds': {'etkdg': energy_kw.get('seed', 0), 'split_salt': args.split_salt,
                  'stereo_salt': args.stereo_salt,
                  'prior_seed': args.prior_seed if prior_info else None},
        'rungs': {s: {'molecules': len(bl), 'conditions': len(mrows[s])}
                  for s, bl in (('train', train), ('heldout', admitted))},
        'stereo': {'policy': 'each stereoisomer a condition, identified by its stereo-tagged '
                             'canonical SMILES; every element specified; reference '
                             're-perceived from 3D',
                   'max_per_molecule': args.max_stereoisomers_per_molecule,
                   'salt': args.stereo_salt, 'perception': 'legacy (pinned)',
                   'identity': 'build_conformer_conditions.py::stereo_identity (fixed-H '
                               'InChI and CIP labels), never canonical SMILES equality',
                   'tetrahedral_N': 'not a stereo element (working assumption)',
                   'mirror_pairs': sorted({tuple(sorted((r['identifier'], r['mirror'])))
                                           for r in mrows['train'] + mrows['heldout']
                                           if r['mirror']}),
                   'mirror_pairs_note': 'the two members of a pair are independent seeded '
                                        'ETKDG embeddings of their own SMILES, not '
                                        'reflections of one reference: reference geometry, '
                                        'reference energy (molecules.tsv reference_energy) '
                                        'and box centre differ. Their log Z is equal in '
                                        'expectation (the energy is achiral), not by '
                                        'construction; compare within estimator noise'},
        'layout': {'K': int(layout.K), 'block_width': [int(v) for v in layout.block_width],
                   'block_width_note': 'region widths r | theta | phi; a transverse u or v '
                                       'sits in the theta region (molecules.tsv '
                                       'n_transverse)',
                   # None on the identity layout, which places nothing
                   'offsets': (None if layout.offsets is None
                               else [int(v) for v in layout.offsets]),
                   'is_identity': bool(layout.is_identity),
                   # the layout the run rebuilds from the training file; it places the
                   # training rows as the shared one does, but can be the identity where the
                   # shared one is not (held-out codes differ, training in region order)
                   'train_only_is_identity': bool(run_layout.is_identity)},
        'members': {s: [{k: r[k] for k in ('identifier', 'key', 'dataset_index', 'k',
                                          'n_pad')} for r in mrows[s]] for s in mrows},
        'prior': prior_info,
        'reasons': {'molecule': _count(r['reason_code'] for r in mol_rej),
                    'isomer': _count(r['reason_code'] for r in rejections
                                     if r['level'] == 'isomer'),
                    'codes': ALL_CODES},
        'accounting': {'universe_rows': n_universe_rows,
                       'refused_before_build': len(set_rej),
                       'attempted': n_attempted, 'kept_molecules': kept_mols,
                       'refused_molecules': len(mol_rej),
                       'not_attempted': n_universe_rows - len(set_rej) - n_attempted},
        'artifacts': {name: {'sha256': _sha256(p), 'bytes': p.stat().st_size}
                      for name, p in files.items()},
        'seconds': round(time.time() - t_start, 1),
    }
    acc = manifest['accounting']
    if acc['kept_molecules'] + acc['refused_molecules'] != acc['refused_before_build'] \
            + acc['attempted']:
        raise SystemExit(f'molecule accounting does not close: {acc}')
    with open(out / 'manifest.json', 'w', encoding='utf-8', newline='\n') as f:
        json.dump(manifest, f, indent=1)
    print(f"kept {kept_mols} molecules ({manifest['rungs']}), refused "
          f"{len(mol_rej)}: {manifest['reasons']['molecule']}")
    print(f'wrote {out / "manifest.json"} in {manifest["seconds"]} s')
    return manifest


if __name__ == '__main__':
    main()
