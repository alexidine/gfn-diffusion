"""Every member's single-copy force field, packed so a MIXED-molecule batch scores in one pass.

WHY. `MultiConformerTorsions` used to score a mixed batch by looping in Python over the
molecules present, one member call per molecule. That costs a fixed 5-7 ms per molecule on
top of the work itself, so at B=1024 the loop took ~2 s at 50 molecules and ~7 s at 500 on
CUDA, against 28 ms for one molecule. The per-molecule part of the energy is not the
arithmetic -- it is only the PARAMETERS.

WHAT MAKES ONE PASS EXACT. `ForceField` is a per-ROW parameter table: every term is a
gather over `*_index` plus an `index_add` over `*_batch` into `n_mols` bins
(mxtaltools/conformers/energy.py::intramolecular_energy). Its only scalars are run-global
(`SCALARS` below), identical across members. So concatenating each member's single-copy
table, with its atom indices shifted to where that row's atoms sit in the batch and its
batch index set to the row, IS the force field of the mixed batch -- not an approximation
of it. `ff_from_mmff` cannot build that table itself: it refuses a mixed tree and queries
RDKit per pair, so it runs once per molecule, here, and never per step.

WHAT IS STORED. Per term, the members' index rows (placement-slot numbering, as each
member's `ff_single` holds them) and values concatenated, with a `ptr` saying which rows
belong to which library entry; plus each entry's atom count, placement-order `z`, per-atom
TRANSVERSE and DUMMY-FRAME flags and per-atom STEREO-LOCK code and quad (same order and `z_ptr`
as `z`) and per-molecule chart constant `log_chart_jacobian`. `gather` is a ragged gather over those arrays -- a fixed number of
tensor calls whatever the batch size or the molecule count.

A later package stores this table in the conditions file; for now it is built at energy
init from the live members (`ForceFieldLibrary.from_members`).
"""
from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from mxtaltools.conformers.energy import ForceField

#: term -> (index field, index width, value fields, batch field, OPTIONAL). An optional term
#: is one whose ForceField default is None -- the "this force field does not carry the term"
#: sentinel -- so a batch in which NO row carries it is emitted as None, exactly what a
#: member without it holds. A required term is emitted as a (possibly empty) tensor.
TERMS = {
    'bond': ('bond_index', 2, ('r0', 'k_bond'), 'bond_batch', False),
    'angle': ('angle_index', 3, ('theta0', 'k_angle', 'angle_linear'), 'angle_batch', False),
    'pair': ('pair_index', 2, ('sigma', 'epsilon', 'vdw_rstar', 'ele_qq', 'ele_scale'),
             'pair_batch', False),
    # unused by intramolecular_energy (only closure_error reads it), carried so the gathered
    # force field is the same object a member's would be
    'closure': ('closure_index', 2, ('closure_r0',), 'closure_batch', False),
    'torsion': ('torsion_index', 4, ('tors_v', 'tors_n', 'tors_gamma'), 'torsion_batch', True),
    'stretch_bend': ('sb_index', 3, ('sb_k_ijk', 'sb_k_kji', 'sb_r0_ij', 'sb_r0_kj',
                                     'sb_theta0'), 'sb_batch', True),
    'oop': ('oop_index', 4, ('oop_k',), 'oop_batch', True),
}
#: run-global constants of the force field: must be identical across members, since the
#: gathered table can carry only one of each
SCALARS = ('lj_k_factor', 'bond_cs', 'angle_cb', 'ele_delta', 'ele_dielectric',
           'vdw_softcore_frac')


def _covered_fields():
    out = {'n_mols', *SCALARS}
    for ifield, _, vfields, bfield, _ in TERMS.values():
        out.update((ifield, bfield, *vfields))
    return out


class ForceFieldLibrary:
    """Single-copy force fields of M molecules, packed; `gather` assembles a mixed batch's."""

    def __init__(self, ffs: Sequence[ForceField], n_atoms: Sequence[int],
                 z: Sequence[Sequence[int]], log_chart: Sequence[float],
                 device='cpu', dtype: Optional[torch.dtype] = None,
                 transverse: Optional[Sequence[Sequence[bool]]] = None,
                 dummy_frame: Optional[Sequence[Sequence[bool]]] = None,
                 stereo_code: Optional[Sequence[Sequence[int]]] = None,
                 stereo_nbr: Optional[Sequence[np.ndarray]] = None):
        ffs = list(ffs)
        if not ffs:
            raise ValueError('ForceFieldLibrary needs at least one force field')
        if not (len(ffs) == len(n_atoms) == len(z) == len(log_chart)):
            raise ValueError(f'{len(ffs)} force fields, {len(n_atoms)} atom counts, {len(z)} z '
                             f'arrays and {len(log_chart)} chart constants')
        # A ForceField field this library does not pack would be DROPPED from every gathered
        # table -- a missing term, i.e. a plausible wrong energy. Refused by name instead, so
        # an MXtalTools change to ForceField stops here rather than in a result.
        unknown = sorted({f.name for f in dataclass_fields(ForceField)} - _covered_fields())
        if unknown:
            raise NotImplementedError(
                f'ForceField carries field(s) {unknown} that ForceFieldLibrary does not pack; '
                f'extend energies/ff_library.py::TERMS before scoring through it')
        dtype = torch.get_default_dtype() if dtype is None else dtype
        self.device = torch.device(device)
        self.dtype = dtype
        self.M = len(ffs)

        self.n_atoms = torch.as_tensor(np.asarray(n_atoms, dtype=np.int64), device=self.device)
        zs = [np.asarray(zi, dtype=np.int64).reshape(-1) for zi in z]
        for i, (zi, na) in enumerate(zip(zs, n_atoms)):
            if len(zi) != int(na):
                raise ValueError(f'entry {i}: {len(zi)} atomic numbers for {na} atoms')
        self.z = torch.as_tensor(np.concatenate(zs), device=self.device)
        self.z_ptr = torch.as_tensor(np.concatenate([[0], np.cumsum(n_atoms)]),
                                     dtype=torch.long, device=self.device)
        # PER ATOM, POSITIONAL, like z: which atom's (theta, phi) slots hold a transverse
        # (u, v) pair (conformer_data.transverse_atom_flags). The energy compares a batch's
        # `ctree_transverse` against this atom by atom, so a file whose flags sit on other
        # atoms -- same count, different chart -- is refused rather than built as a bend on
        # the wrong atom. None means no member carries one.
        # The DUMMY-FRAME flag (conformer_data.dummy_frame_atom_flags) is stored the same way
        # and for the same reason: it changes what an atom's phi slot means, so a file whose
        # flags sit on other atoms is a different chart with the same shapes.
        per_atom = {}
        for name, given in (('transverse', transverse), ('dummy_frame', dummy_frame)):
            fl = ([np.zeros(int(na), dtype=bool) for na in n_atoms] if given is None
                  else [np.asarray(t, dtype=bool).reshape(-1) for t in given])
            if len(fl) != len(zs):
                raise ValueError(f'{len(fl)} {name} flag arrays for {len(zs)} entries')
            for i, (ti, na) in enumerate(zip(fl, n_atoms)):
                if len(ti) != int(na):
                    raise ValueError(f'entry {i}: {len(ti)} {name} flags for {na} atoms')
            per_atom[name] = torch.as_tensor(np.concatenate(fl), dtype=torch.bool,
                                             device=self.device)
        self.transverse = per_atom['transverse']
        self.dummy_frame = per_atom['dummy_frame']
        # PER ATOM, POSITIONAL, like z: the STEREO LOCK's element keyed on each atom, as
        # kind * sign (0 none, +-1 tetrahedral, +-2 double bond; energies/stereo_lock.py), and
        # its four indicator atoms as deltas. Stereoisomers share z, atom count and every
        # chart flag, so this is the only per-atom field that sees a mol_id bound to the
        # wrong stereoisomer -- which would score its rows against the other isomer's lock.
        # None means no member carries a lock element (all zeros).
        codes = ([np.zeros(int(na), dtype=np.int64) for na in n_atoms] if stereo_code is None
                 else [np.asarray(c, dtype=np.int64).reshape(-1) for c in stereo_code])
        nbrs = ([np.zeros((int(na), 4), dtype=np.int64) for na in n_atoms] if stereo_nbr is None
                else [np.asarray(q, dtype=np.int64).reshape(-1, 4) for q in stereo_nbr])
        if len(codes) != len(zs) or len(nbrs) != len(zs):
            raise ValueError(f'{len(codes)} stereo code and {len(nbrs)} stereo quad arrays for '
                             f'{len(zs)} entries')
        for i, (c, q, na) in enumerate(zip(codes, nbrs, n_atoms)):
            if len(c) != int(na) or len(q) != int(na):
                raise ValueError(f'entry {i}: {len(c)} stereo codes / {len(q)} quads for '
                                 f'{na} atoms')
        self.stereo_code = torch.as_tensor(np.concatenate(codes), device=self.device)
        self.stereo_nbr = torch.as_tensor(np.concatenate(nbrs), device=self.device)
        self.log_chart = torch.as_tensor(np.asarray(log_chart, dtype=np.float64),
                                         dtype=dtype, device=self.device)

        self.scalars = {}
        for s in SCALARS:
            vals = {repr(getattr(f, s)) for f in ffs}
            if len(vals) != 1:
                raise ValueError(f'{s} differs across members ({sorted(vals)}); one gathered '
                                 f'force field can carry only one value')
            self.scalars[s] = getattr(ffs[0], s)

        self.idx: Dict[str, torch.Tensor] = {}
        self.ptr: Dict[str, torch.Tensor] = {}
        self.val: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        for term, (ifield, width, vfields, bfield, _) in TERMS.items():
            parts, counts = [], []
            for i, f in enumerate(ffs):
                a = getattr(f, ifield)
                a = (torch.zeros((0, width), dtype=torch.long) if a is None
                     else a.detach().cpu().long().reshape(-1, width))
                self._check_single_copy(i, f, term, a, getattr(f, bfield), int(n_atoms[i]))
                parts.append(a)
                counts.append(int(a.shape[0]))
            self.idx[term] = torch.cat(parts).to(self.device)
            self.ptr[term] = torch.as_tensor(np.concatenate([[0], np.cumsum(counts)]),
                                             dtype=torch.long, device=self.device)
            self.val[term] = {v: self._pack_values(ffs, counts, term, v) for v in vfields}

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_members(cls, members: Mapping[str, object], device=None,
                     dtype: Optional[torch.dtype] = None) -> 'ForceFieldLibrary':
        """One entry per member, in the mapping's order, from each member's `ff_single`.

        `z` is the member's PLACEMENT-order atomic numbers (`spec.z`), the order a condition
        graph stores its atoms in, so it can be compared atom for atom against a batch; the
        transverse flags are the same function `condition_from_energy` writes them with.
        """
        from energies.conformer_data import dummy_frame_atom_flags, transverse_atom_flags
        from energies.stereo_lock import atom_codes

        ms = list(members.values())
        first = ms[0]
        stereo = [atom_codes(m.stereo, int(m.spec.n_atoms)) for m in ms]
        return cls([m.ff_single for m in ms],
                   [int(m.spec.n_atoms) for m in ms],
                   [np.asarray(m.spec.z) for m in ms],
                   [float(m.log_chart_jacobian) for m in ms],
                   device=first.device if device is None else device,
                   dtype=first.dtype if dtype is None else dtype,
                   transverse=[transverse_atom_flags(m).numpy() for m in ms],
                   dummy_frame=[dummy_frame_atom_flags(m).numpy() for m in ms],
                   stereo_code=[c for c, _ in stereo], stereo_nbr=[q for _, q in stereo])

    @staticmethod
    def _check_single_copy(i, f, term, idx, batch, n_atoms):
        """A packed entry must be ONE molecule's table in its own slot numbering.

        A tiled table, or one whose indices point outside the molecule, would gather rows
        that address another row's atoms once offset -- finite and wrong -- so both are
        refused here, once, rather than trusted on every call.
        """
        if int(f.n_mols) != 1:
            raise ValueError(f'entry {i}: the library takes SINGLE-copy force fields '
                             f'(ff_single), got n_mols = {f.n_mols}')
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= n_atoms):
            raise ValueError(f'entry {i}: {term} indices span [{int(idx.min())}, '
                             f'{int(idx.max())}] for a {n_atoms}-atom molecule')
        if batch is not None and batch.numel() and bool((batch != 0).any()):
            raise ValueError(f'entry {i}: {term} batch index is not all zero on a '
                             f'single-copy force field')

    def _pack_values(self, ffs, counts, term, v) -> Optional[torch.Tensor]:
        """Concatenated values of field `v`, or None when the field is absent BY DESIGN.

        Presence is decided over the members that carry rows of this term: they must agree,
        or one gathered table would mix "no such term" with "this term's value" across rows.
        A member with no rows contributes nothing whether it holds None or an empty tensor --
        `ff_from_mmff` writes None for an absent torsion block and empty tensors for an
        absent pair block, and both mean the same thing here.
        """
        held = [getattr(f, v) for f in ffs]
        with_rows = [h is not None for h, c in zip(held, counts) if c]
        if with_rows and not all(with_rows) and any(with_rows):
            raise ValueError(f'{term}.{v} is present on some members and None on others that '
                             f'carry {term} rows; a gathered table cannot mix the two')
        present = [h for h in held if h is not None]
        if not present or (with_rows and not with_rows[0]):
            return None
        is_bool = present[0].dtype == torch.bool
        cat = torch.cat([h.detach().cpu().reshape(-1) for h, c in zip(held, counts) if c]
                        or [present[0].detach().cpu().reshape(-1)[:0]])
        if int(cat.numel()) != sum(counts):
            raise ValueError(f'{term}.{v}: {cat.numel()} values for {sum(counts)} rows')
        return cat.to(device=self.device, dtype=torch.bool if is_bool else self.dtype)

    # ------------------------------------------------------------------ gather

    @staticmethod
    def _ragged(ptr: torch.Tensor, lib_ids: torch.Tensor):
        """`(rows, owner)`: the library rows of every batch row's entry, in batch order.

        ``counts = ptr_diff[lib_ids]``; ``owner = repeat_interleave(arange(B), counts)``; each
        row's slice starts at ``ptr[lib_ids][owner]`` and runs over its local offset.
        """
        counts = (ptr[1:] - ptr[:-1]).index_select(0, lib_ids)
        owner = torch.repeat_interleave(
            torch.arange(lib_ids.numel(), dtype=torch.long, device=lib_ids.device), counts)
        first = torch.cumsum(counts, 0) - counts
        local = (torch.arange(owner.numel(), dtype=torch.long, device=lib_ids.device)
                 - first.index_select(0, owner))
        rows = ptr[:-1].index_select(0, lib_ids).index_select(0, owner) + local
        return rows, owner

    def gather(self, lib_ids: torch.Tensor, atom_off: torch.Tensor) -> ForceField:
        """The ForceField of a batch whose row b is library entry ``lib_ids[b]``.

        ``atom_off[b]`` is where row b's atoms start in the batch (``ptr[:-1]``). Row b's
        index rows are its entry's, shifted by ``atom_off[b]``; its batch index is b. The
        caller is responsible for the batch's atoms actually being that entry's, in
        placement order -- `MultiConformerTorsions` checks `n_atoms` and `z` before calling.
        """
        lib_ids = lib_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        atom_off = atom_off.to(device=self.device, dtype=torch.long).reshape(-1)
        kw = {}
        for term, (ifield, _, vfields, bfield, optional) in TERMS.items():
            rows, owner = self._ragged(self.ptr[term], lib_ids)
            if optional and rows.numel() == 0:
                # absent from the WHOLE batch: the member-side sentinel, not an empty table
                kw[ifield] = kw[bfield] = None
                kw.update({v: None for v in vfields})
                continue
            kw[ifield] = (self.idx[term].index_select(0, rows)
                          + atom_off.index_select(0, owner).unsqueeze(-1))
            kw[bfield] = owner
            for v in vfields:
                src = self.val[term][v]
                kw[v] = None if src is None else src.index_select(0, rows)
        kw.update(self.scalars)
        return ForceField(n_mols=int(lib_ids.numel()), **kw)

    def describe(self) -> str:
        rows = ', '.join(f'{t} {int(p[-1])}' for t, p in self.ptr.items())
        return (f'   FORCE-FIELD LIBRARY: {self.M} molecule(s), {int(self.z.numel())} atoms '
                f'({int(self.transverse.sum())} transverse, {int(self.dummy_frame.sum())} '
                f'dummy-frame, {int((self.stereo_code != 0).sum())} stereo-lock keys); '
                f'rows {rows}')
