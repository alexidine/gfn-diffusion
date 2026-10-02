"""One energy interface over MANY molecules, dispatched per row.

WHY. `ConformerTorsions` is a single molecule: `energy` reads `self._M`, `self.r0`,
`self.log_jacobian_const` and the force-field terms, all built for one chart. So a batch
drawn from a 45-molecule condition set was scored against ONE molecule's chart -- every row's
reward computed as though it were a different molecule. The shapes agree whenever k agrees,
so this does not raise; it silently returns numbers.

The constants alone make that fatal. At `level='torsion'` the reward reduces to
``-U/T + log_jacobian_const + log_chart_jacobian``, and both terms are per-molecule:

    CCC(=O)CC   log_jacobian_const 4.508   log_chart_jacobian 2.289
    CCCCO                          3.994                      2.289
    OCc1cnco1                      3.472                      1.145

so a mixed batch scored against one member is wrong by a different constant for every row --
which is exactly a per-condition log Z error, the thing conditional training is trying to
learn.

ONE PASS, NOT A LOOP OVER MOLECULES. Everything per-molecule about the energy is either on
the condition graph or a parameter table. The tree, the state -> (r, theta, phi) map and the
BAT volume element are read graph-natively off the batch (`conformer_data.batch_tree`,
`state_to_dof`), whatever mix of molecules it holds; the force field is gathered from a
packed per-molecule library (energies/ff_library.py); the chart constant is a per-molecule
lookup. So a mixed batch costs one build and one force-field evaluation, independent of how
many molecules it holds. The per-member loop this replaced cost a fixed 5-7 ms per molecule
present -- ~2 s per 1024-row step at 50 molecules, ~7 s at 500 -- and survives only as the
test oracle (`_energy_per_member`).

ONE STATE WIDTH FOR THE WHOLE SET. The GFN's state dimension is fixed when it is built. A set
whose members share their block layout uses the reference chart's state unchanged. A MIXED-k
set uses the width-K CARRIER (energies/conformer_carrier.py): each member's columns are placed
in fixed r | theta | phi blocks (a linear bend's transverse u and v in the theta block), pads
are pinned to 0, and the condition graph's reconstruction map reads each row's own columns --
with a positive check that the pads really are 0, that the batch's `state_mask` matches the
layout, and that each row's atoms (and which of them carry a linear bend) are the molecule its
`mol_id` names, rather than a removed assertion.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from energies.conformer_data import stereo_coeff_mismatch
from energies.conformer_torsions import ConformerTorsions


class MultiConformerTorsions(ConformerTorsions):
    """`ConformerTorsions` for a molecule SET, scoring each row with its own chart.

    Subclasses rather than wraps so that the ~15-member energy protocol train.py's Modeller
    talks to -- `data_ndim`, `periodic_dims`, `dtype`, `sample_prior_states`, `describe`, the
    reward clip, the timing counters -- is inherited rather than re-forwarded by hand. The
    first molecule is the REFERENCE member and supplies every property that must agree across
    the set; the rest are held as members and used only for scoring.
    """

    def __init__(self, smiles: Sequence[str], identifiers: Optional[Sequence[str]] = None,
                 reference_positions: Optional[Dict[str, object]] = None, **kw):
        """``reference_positions`` maps an identifier to its STORED reference conformer, in
        the RDKit atom order ``ConformerTorsions(reference_positions=...)`` takes; a member
        whose identifier it names is built from it (no embedding), the rest are embedded.
        """
        refs = dict(reference_positions or {})
        smiles = list(smiles)
        if not smiles:
            raise ValueError('MultiConformerTorsions needs at least one molecule')
        idents = list(identifiers) if identifiers is not None else list(smiles)
        if len(idents) != len(smiles):
            raise ValueError(f'{len(smiles)} smiles against {len(idents)} identifiers')

        # BEFORE super().__init__, because the parent's constructor CALLS self.energy() to
        # cache `e_ref` -- so the dispatch guard runs while this object is still half-built.
        # An empty table there means "not a set yet", and the guard reads <= 1 rather than
        # == 1 so that call takes the parent's own path.
        self._members: Dict[str, ConformerTorsions] = {}
        self._by_mol_id: Dict[int, str] = {}
        self._member_smiles: Dict[str, str] = {}
        self._carrier = None
        super().__init__(smiles=smiles[0], reference_positions=refs.get(idents[0]), **kw)
        for smi, ident in zip(smiles, idents):
            if ident in self._members:
                continue
            member = (self if smi == smiles[0] and ident == idents[0]
                      else ConformerTorsions(smiles=smi, reference_positions=refs.get(ident),
                                             **kw))
            self._members[ident] = member
            self._member_smiles[ident] = smi

        # ONE LAYOUT FOR THE WHOLE SET. Members whose state columns fall in the same blocks
        # share the reference chart's layout exactly (identity carrier) and nothing below
        # changes. Otherwise the state becomes the width-K CARRIER (energies/conformer_carrier):
        # this object stops being a chart and becomes a dispatcher, so the reference member is
        # rebuilt as its own instance and every chart method on `self` is refused.
        from energies.conformer_carrier import CarrierLayout
        layout = CarrierLayout(self._members)
        if not layout.is_identity:
            ref_ident = idents[0]
            self._members[ref_ident] = ConformerTorsions(
                smiles=smiles[0], reference_positions=refs.get(ref_ident), **kw)
            self._carrier = layout
            self.data_ndim = layout.K
            self._free_block = layout.free_block
            self._lin_free_idx = torch.as_tensor(np.flatnonzero(layout.free_block != 2),
                                                 dtype=torch.long, device=self.device)
        self._build_library()

    def _build_library(self) -> None:
        """The per-molecule tables the one-pass energy reads, in `_members` order.

        `_lib` packs every member's force field, atom count, placement-order `z`, per-atom
        transverse flag and chart constant; `_valid_of_lib[i]` is entry i's carrier columns
        (all True on the identity layout, which has no pads). `_lib_of_mol_id` stays None
        until `bind_identifier_registry` supplies the mol_id numbering.
        """
        from energies.ff_library import ForceFieldLibrary

        # THE WALL CONSTANTS ARE READ OFF THE DISPATCHER, once for the whole batch: the box and
        # the transverse disc (`_energy_one_pass`) use `bounding_coeff` and `rho_wall`, the
        # clip `energy_clip`, and the stereo lock `stereo_coeff`. Every member is built from the
        # same kwargs, so they agree today; checked because one member that did not would be
        # scored against a wall that is not its own -- a plausible number, and the per-member
        # oracle would disagree silently.
        # `double_bond_box_deg` joins them: it decides which of a member's columns are walled
        # and their scale, and a member built under another box would sit in a layout, and
        # under a chart constant, that are not the set's.
        for name in ('bounding_coeff', 'rho_wall', 'energy_clip', 'stereo_coeff',
                     'double_bond_box_deg'):
            want = getattr(self, name)
            off = [i for i, m in self._members.items() if getattr(m, name) != want]
            if off:
                raise ValueError(
                    f'{name} differs between the set and {len(off)} member(s) (e.g. '
                    f'{off[0]!r}: {getattr(self._members[off[0]], name)!r} against {want!r}); '
                    f'the one-pass energy applies one value to every row')
        self._lib_idents: List[str] = list(self._members)
        self._lib_index: Dict[str, int] = {k: i for i, k in enumerate(self._lib_idents)}
        self._lib = ForceFieldLibrary.from_members(self._members, device=self.device,
                                                   dtype=self.dtype)
        if self._carrier is None:
            valid = np.ones((len(self._lib_idents), int(self.data_ndim)), dtype=bool)
        else:
            valid = np.stack([self._carrier.valid(i) for i in self._lib_idents])
        self._valid_of_lib = torch.as_tensor(valid, device=self.device)
        # `torsion` and `dihedral` freeze r and theta, so log J is a per-molecule CONSTANT
        # there and the prebuilt reward needs no geometry at all; `flex` and `full` have none
        consts = [m.log_jacobian_const for m in self._members.values()]
        if all(c is None for c in consts):
            self._log_jac_const_of_lib = None
        elif any(c is None for c in consts):
            raise ValueError('members disagree on whether log J is constant; they were built '
                             'at different levels')
        else:
            self._log_jac_const_of_lib = torch.as_tensor(
                np.asarray(consts, dtype=np.float64), device=self.device)
        self._lib_of_mol_id: Optional[torch.Tensor] = None
        self._clip_floor_of_lib: Optional[torch.Tensor] = None
        self._reference_potential_of_lib: Optional[torch.Tensor] = None

    def install_clip_floor(self) -> torch.Tensor:
        """`energy_clip_origin` 'reference' on the set: every member takes its own floor
        (`ConformerTorsions.install_clip_floor`) and the one-pass energy reads them per row
        from `_clip_floor_of_lib`, in `_members` order. Returns that table."""
        floors = []
        for m in self._members.values():
            floors.append(ConformerTorsions.install_clip_floor(m))
            m.release_batch_cache()
        self._clip_floor_of_lib = torch.as_tensor(np.asarray(floors, dtype=np.float64),
                                                  dtype=self.dtype, device=self.device)
        self._reference_potential_of_lib = self._clip_floor_of_lib.detach().double().cpu()
        self.energy_clip_origin = 'reference'
        return self._clip_floor_of_lib

    def reference_potentials(self) -> torch.Tensor:
        """Every member's `reference_potential`, float64 on the CPU, in `_members` order.
        Computed once."""
        if getattr(self, '_reference_potential_of_lib', None) is None:
            vals = []
            for m in self._members.values():
                vals.append(ConformerTorsions.reference_potential(m))
                m.release_batch_cache()
            self._reference_potential_of_lib = torch.as_tensor(
                np.asarray(vals, dtype=np.float64))
        return self._reference_potential_of_lib

    @property
    def is_carrier(self) -> bool:
        """True when the state is the width-K carrier rather than the reference chart."""
        return self._carrier is not None

    @property
    def carrier(self):
        return self._carrier
    # ------------------------------------------------------------------ dispatch

    def bind_identifier_registry(self, registry) -> None:
        """Accept init_identifiers' `{identifier: mol_id}` so buffered rows can be dispatched.

        A batch straight off a conditions file carries `identifier`; one that has been through
        a buffer carries only `mol_id`, because buffers keep tensors and `identifier` is a list.
        Binding the registry lets both resolve to the same chart instead of the second silently
        falling back to the reference molecule.

        Also builds `_lib_of_mol_id`, the mol_id -> library-index table the one-pass energy
        resolves rows through: a tensor lookup, where the identifier route is a Python walk
        over the rows. -1 marks a mol_id this energy holds no member for.
        """
        reg = {k: int(v) for k, v in dict(registry).items()}
        if any(v < 0 for v in reg.values()):
            raise ValueError('identifier registry holds a negative mol_id')
        self._by_mol_id = {v: k for k, v in reg.items() if k in self._members}
        table = torch.full((max(reg.values(), default=-1) + 1,), -1, dtype=torch.long)
        for k, v in reg.items():
            if k in self._lib_index:
                table[v] = self._lib_index[k]
        self._lib_of_mol_id = table.to(self.device)

    #: NOT `n_molecules` -- that name is already the energy protocol's, set by
    #: `set_n_molecules` from init_identifiers' mol_id registry and used as the condition_id
    #: radix. Shadowing it with a property broke the parent's own constructor.
    @property
    def n_charts(self) -> int:
        return len(self._members)

    @property
    def reference_identifier(self) -> str:
        """The identifier of the member every single-molecule path implicitly draws from.

        `self.smiles` is the reference member's SMILES, and on this route that is NOT a key
        of anything: the buffers, the mol_id registry and the per-molecule energy table are
        all keyed by IDENTIFIER, which a conditions file is free to make distinct from the
        SMILES (and must, when the same molecule appears more than once).
        """
        return next(iter(self._members))

    @property
    def distinct_smiles(self) -> int:
        """How many genuinely different molecules the set holds.

        1 when a set is one molecule repeated -- the case where a reference-only draw is
        representative of the whole set rather than a sample of one member of it.
        """
        return len(set(self._member_smiles.values()))

    def _row_identifiers(self, mol_batch, n: int) -> List[str]:
        """One identifier per ROW, in batch order.

        `identifier` is the key `init_identifiers` mints `mol_id` from, so it is the same
        notion of molecular identity the condition vector and the buffers already use. A
        batch without it cannot be dispatched -- and must not be silently scored against the
        reference member, which is the bug this class exists to remove.
        """
        idents = getattr(mol_batch, 'identifier', None)
        if idents is None:
            # `identifier` is a python LIST, and the buffers keep tensors -- so a row that has
            # been through a buffer arrives with `mol_id` (a tensor, minted by
            # init_identifiers) and nothing else. Same identity, different carrier.
            mid = getattr(mol_batch, 'mol_id', None)
            if mid is not None and self._by_mol_id:
                out = []
                for j in mid.reshape(-1).tolist():
                    if j not in self._by_mol_id:
                        raise RuntimeError(
                            f'mol_id {j} is not in the identifier registry this energy was '
                            f'bound to ({len(self._by_mol_id)} molecules)')
                    out.append(self._by_mol_id[j])
                if len(out) != n:
                    raise RuntimeError(f'{len(out)} mol_ids for {n} rows')
                return out
            raise RuntimeError(
                'MultiConformerTorsions needs `identifier` on the batch to know which chart '
                'each row belongs to. Scoring it against the reference molecule instead is '
                'exactly the silent error this class exists to prevent.')
        if isinstance(idents, str):
            idents = [idents]
        idents = list(idents)
        if len(idents) != n:
            raise RuntimeError(
                f'{len(idents)} identifiers for {n} rows; the batch and the state disagree '
                f'about how many samples there are')
        return idents

    def _lib_ids(self, mol_batch, n: int) -> torch.Tensor:
        """Per-row LIBRARY index, ``[n]`` long on the batch's device; -1 = mol_id not held.

        `mol_id` first, through the tensor table `bind_identifier_registry` built: it is what
        survives a buffer, and a lookup costs one gather where the identifier route walks the
        rows in Python. `identifier` is the fallback -- a batch straight off a conditions
        file, or an energy whose registry is not bound yet. Neither present is refused, as
        `_row_identifiers` refuses it: scoring against the reference member instead is the
        silent error this class exists to remove.
        """
        dev = mol_batch.z.device
        mid = getattr(mol_batch, 'mol_id', None)
        if mid is not None and self._lib_of_mol_id is not None:
            mid = mid.reshape(-1).long()
            if int(mid.numel()) != n:
                raise RuntimeError(f'{mid.numel()} mol_ids for {n} rows')
            table = self._lib_of_mol_id.to(dev)
            if int(table.numel()) == 0:
                return torch.full((n,), -1, dtype=torch.long, device=dev)
            oob = (mid < 0) | (mid >= table.numel())
            got = table.index_select(0, mid.clamp(0, int(table.numel()) - 1))
            return torch.where(oob, torch.full_like(got, -1), got)
        idents = self._row_identifiers(mol_batch, n)
        unknown = sorted({k for k in idents if k not in self._lib_index})
        if unknown:
            raise RuntimeError(
                f'batch carries {len(unknown)} molecule(s) this energy was not built for, '
                f'e.g. {unknown[0]!r}. The energy set and the condition set must be built '
                f'from the same molecule list.')
        return torch.as_tensor([self._lib_index[k] for k in idents], dtype=torch.long,
                               device=dev)

    def _resolve_rows(self, mol_batch, n: int, x: Optional[torch.Tensor] = None,
                      geometry: bool = True):
        """``(lib_ids, ptr)`` for a batch, after POSITIVE checks that each row is its molecule.

        Every check is a tensor reduction, and they are read back together -- ONE host sync
        per call whatever the batch size or molecule count; the per-check detail is computed
        only on the failure path, to name the offending row. A failure raises, because each
        one otherwise returns a plausible energy for the wrong molecule:

          * a mol_id this energy holds no member for;
          * a row whose recorded STEREO COEFFICIENT (``ctree_stereo_coeff``, written by
            `condition_from_energy` from the energy the graph was built with) differs from
            this energy's. A baked row carries clip(U + P) at that coefficient and is never
            re-scored, and a condition graph's per-coordinate features mark the rows the prior
            holds only when the lock is on, so a file built unlocked and read by a locked run
            (or the reverse) trains on another target with no other error. A batch without the
            field reads as built unlocked;
          * a row whose atom count, placement-order ``z`` or per-atom TRANSVERSE or
            DUMMY-FRAME flag (``ctree_transverse``, ``ctree_dummy_frame``) differs from its
            library entry's. The z part is the only
            check that sees a mol_id registered to the WRONG member between two molecules with
            the same block counts -- constitutional isomers such as CCCO and CC(C)O share
            every carrier column, so the pad and mask checks pass, and that member's force
            field would be applied silently to the other's geometry. The flag part is the one
            that sees a STALE file: the flags decide which theta/phi slots the build reads as
            a (u, v) bend, so a file whose flags sit on other atoms -- even the same NUMBER
            of them -- builds a wrong geometry from right-shaped tensors. Compared atom by
            atom, not by count, in the same reduction as z;
          * with the stereo lock on, a row whose per-atom lock element (``ctree_stereo_kind``
            times ``ctree_stereo_sign``, and ``ctree_stereo_nbr``) differs from its library
            entry's. Stereoisomers share z, atom count and every chart flag, so this is the
            only check that sees a mol_id bound to the WRONG STEREOISOMER; the lock read off
            the graph would then pin one isomer on the row of the other's condition;
          * a nonzero PAD column in ``x`` (pads are pinned to 0 along the whole trajectory, so
            the row was not produced in its member's layout);
          * a ``state_mask`` that is not the row's member's.

        With ``geometry`` the batch's float fields must also be in this energy's dtype: the
        reconstruction reads them, and a float64 file under a float32 run would build in the
        wrong precision or fail deep inside the force field.
        """
        if not mol_batch.is_batch:
            raise RuntimeError('the multi-molecule energy scores a collated BATCH of condition '
                               'graphs; got a single graph')
        if int(mol_batch.num_graphs) != n:
            raise RuntimeError(
                f'{mol_batch.num_graphs} graphs in the batch against {n} rows of state; the '
                f'batch and the state disagree about how many samples there are')
        dev = mol_batch.z.device
        if x is not None and x.device != dev:
            raise RuntimeError(
                f'state on {x.device} but mol_batch on {dev}: the one-pass energy reads the '
                f'tree and the DoF map off the batch, so both must sit on one device')
        if geometry:
            fdt = mol_batch.ctree_r0.dtype
            if fdt != self.dtype:
                raise RuntimeError(
                    f'batch conformer fields are {fdt} but this energy runs in {self.dtype}; '
                    f'cast the batch to the run dtype (ConformerModeller._as_run_dtype) '
                    f'before scoring it')

        lib_ids = self._lib_ids(mol_batch, n)
        bad_id = lib_ids < 0
        safe = lib_ids.clamp_min(0)
        ptr, graph = mol_batch.ptr, mol_batch.batch
        bad_coeff = stereo_coeff_mismatch(mol_batch, self.stereo_coeff).to(dev)

        # atom count, placement-order z and the transverse flag, at atom level so one gather
        # covers the batch. A batch without `ctree_transverse` reads as flag-free, which is
        # what a file written before the transverse chart meant -- and is refused against any
        # member that has one.
        n_atoms = self._lib.n_atoms.to(dev)
        n_lib = n_atoms.index_select(0, safe)
        bad_n = (ptr[1:] - ptr[:-1]) != n_lib
        slot = torch.arange(int(graph.numel()), device=dev) - ptr[:-1].index_select(0, graph)
        n_own = n_lib.index_select(0, graph)
        z_at = (self._lib.z_ptr.to(dev).index_select(0, safe.index_select(0, graph))
                + torch.minimum(slot, n_own - 1))
        bad_z_atom = (slot >= n_own) | (self._lib.z.to(dev).index_select(0, z_at)
                                        != mol_batch.z.reshape(-1).long())
        tvf = getattr(mol_batch, 'ctree_transverse', None)
        tvf = (torch.zeros_like(bad_z_atom) if tvf is None
               else tvf.reshape(-1).to(device=dev, dtype=torch.bool))
        bad_tv_atom = self._lib.transverse.to(dev).index_select(0, z_at) != tvf
        # the DUMMY-FRAME flag, compared the same way and in the same reduction: it decides
        # which phi slots are read against a dummy atom, so flags on other atoms build a
        # wrong geometry from right-shaped tensors exactly as misplaced transverse flags do
        dmf = getattr(mol_batch, 'ctree_dummy_frame', None)
        dmf = (torch.zeros_like(bad_z_atom) if dmf is None
               else dmf.reshape(-1).to(device=dev, dtype=torch.bool))
        bad_tv_atom = bad_tv_atom | (self._lib.dummy_frame.to(dev).index_select(0, z_at) != dmf)
        # THE STEREO LOCK, compared only when it is ON: off, it adds no term, so a row carrying
        # another stereoisomer's table is scored exactly as its own would be. A batch without
        # the fields reads as lock-free and is refused against any member that has an element.
        bad_st_atom = torch.zeros_like(bad_z_atom)
        if self.stereo_coeff > 0.0:
            sk = getattr(mol_batch, 'ctree_stereo_kind', None)
            ss = getattr(mol_batch, 'ctree_stereo_sign', None)
            sq = getattr(mol_batch, 'ctree_stereo_nbr', None)
            code = (torch.zeros_like(z_at) if sk is None or ss is None
                    else (sk.reshape(-1) * ss.reshape(-1)).to(device=dev, dtype=torch.long))
            quad = (torch.zeros(int(z_at.numel()), 4, dtype=torch.long, device=dev) if sq is None
                    else sq.reshape(-1, 4).to(device=dev, dtype=torch.long))
            bad_st_atom = ((self._lib.stereo_code.to(dev).index_select(0, z_at) != code)
                           | (self._lib.stereo_nbr.to(dev).index_select(0, z_at)
                              != quad).any(-1))
        bad_z = torch.zeros(n, dtype=torch.long, device=dev).index_add(
            0, graph, (bad_z_atom | bad_tv_atom | bad_st_atom).long()) > 0

        valid_rows = self._valid_of_lib.to(dev).index_select(0, safe)
        K = int(valid_rows.shape[1])
        bad_pad = torch.zeros(n, dtype=torch.bool, device=dev)
        if x is not None and self._carrier is not None:
            # masked_fill keeps a NaN or inf in a pad visible to the != 0 test
            bad_pad = (x.detach().masked_fill(valid_rows, 0) != 0).any(-1)
        bad_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        mask = getattr(mol_batch, 'state_mask', None)
        if mask is None and self._carrier is not None:
            # REQUIRED ON A CARRIER, because a carrier is no longer always WIDER than its
            # members. A transverse v moves from the phi region to the theta one, so two
            # same-size nitriles whose bends sit on different rows (OCCCC#N and OCC(C)C#N)
            # share every region width and the carrier is a pure PERMUTATION: K = k, no pads.
            # A member-order file then passes the pad check and every width, and its
            # `ctree_*_col` read the carrier state in the wrong order. `state_mask` is what
            # `carrier_pad_condition` writes; its absence is the marker.
            raise RuntimeError(
                'the batch carries no state_mask, but this energy is a CARRIER: its condition '
                'graphs were not re-expressed in the carrier layout (carrier_pad_condition), '
                'so their reconstruction map indexes member columns, not carrier ones')
        if mask is not None:
            if int(mask.numel()) != n * K:
                raise RuntimeError(
                    f'state_mask has {mask.numel()} entries for {n} rows of width {K}; the '
                    f'conditions file was built against a different layout')
            bad_mask = (mask.reshape(n, K).bool() != valid_rows).any(-1)

        flags = torch.stack([bad_id.any(), bad_coeff.any(), (bad_n | bad_z).any(),
                             bad_pad.any(), bad_mask.any()]).tolist()
        if any(flags):
            self._raise_row_check(flags, mol_batch, lib_ids, bad_id, bad_n | bad_z, bad_pad,
                                  x, valid_rows,
                                  bad_z_or_count_atom=bad_z_atom | bad_n.index_select(0, graph),
                                  bad_stereo_atom=bad_st_atom, bad_coeff=bad_coeff)
        return lib_ids, ptr

    def _raise_row_check(self, flags, mol_batch, lib_ids, bad_id, bad_atoms, bad_pad, x,
                         valid_rows, bad_z_or_count_atom=None, bad_stereo_atom=None,
                         bad_coeff=None):
        """The failure path of `_resolve_rows`: name the first offending row, then raise.

        ``bad_z_or_count_atom`` is the per-atom mismatch WITHOUT the transverse flags, so a row
        flagged in `bad_atoms` but clean here is a flag mismatch alone and is named as one;
        ``bad_stereo_atom`` separates a stereo-lock mismatch from a chart-flag one the same way.
        """
        first = lambda m: int(torch.nonzero(m)[0])
        mid = getattr(mol_batch, 'mol_id', None)
        mid_of = lambda i: ('' if mid is None else f' (mol_id {int(mid.reshape(-1)[i])})')
        if flags[0]:
            i = first(bad_id)
            raise RuntimeError(
                f'row {i}{mid_of(i)}: this mol_id is not in the identifier registry this '
                f'energy was bound to, or names no member of it ({self.n_charts} molecules); '
                f'the energy set and the condition set must be built from the same list')
        if flags[1]:
            i = first(bad_coeff)
            rec = getattr(mol_batch, 'ctree_stereo_coeff', None)
            got = 0.0 if rec is None else float(rec.reshape(-1)[i])
            raise RuntimeError(
                f'row {i}{mid_of(i)} was built under stereo_coeff {got:g} '
                f'(ctree_stereo_coeff{"" if rec is not None else ": absent, read as 0"}), but '
                f'this energy locks at {self.stereo_coeff:g}. Its baked energy and its '
                f'per-coordinate features belong to the other target, and a baked row is never '
                f're-scored. Rebuild the conditions / prior file with '
                f'build_conformer_conditions.py --stereo-coeff {self.stereo_coeff:g}.')
        if flags[2]:
            i = first(bad_atoms)
            ident = self._lib_idents[int(lib_ids[i])]
            in_row = mol_batch.batch == i
            if (bad_z_or_count_atom is not None and bad_stereo_atom is not None
                    and not bool(bad_z_or_count_atom[in_row].any())
                    and bool(bad_stereo_atom[in_row].any())):
                raise RuntimeError(
                    f'row {i}{mid_of(i)} resolves to {ident!r} and carries its atoms, but its '
                    f'stereo lock (ctree_stereo_kind * ctree_stereo_sign, ctree_stereo_nbr) '
                    f'differs from that member\'s on atom(s) '
                    f'{torch.nonzero(bad_stereo_atom[in_row]).flatten().tolist()}: the row is '
                    f'another STEREOISOMER of the same molecule, or its conditions file was '
                    f'built without the lock or against another reference. Scoring it would pin '
                    f'the other isomer on this condition. Check the mol_id registry, or rebuild '
                    f'the conditions file.')
            if (bad_z_or_count_atom is not None
                    and not bool(bad_z_or_count_atom[in_row].any())):
                # count and z agree, so it is the flags alone: the right molecule, charted
                # differently from the member this run built
                raise RuntimeError(
                    f'row {i}{mid_of(i)} resolves to {ident!r} and carries its atoms, but its '
                    f'transverse flags (ctree_transverse) or dummy-frame flags '
                    f'(ctree_dummy_frame) sit on different atoms from that member\'s chart: '
                    f'the conditions file was built against another chart (a stale file). '
                    f'Building it would read a (u, v) bend or a dummy-frame dihedral out of '
                    f'the wrong atoms\' slots. Rebuild the conditions file.')
            raise RuntimeError(
                f'row {i}{mid_of(i)} resolves to {ident!r}, but its atoms (count or '
                f'placement-order z) are not that molecule\'s: the mol_id registry, the '
                f'conditions file and this energy\'s member set disagree. Scoring it would '
                f'apply {ident!r}\'s force field to another molecule\'s geometry.')
        if flags[3]:
            i = first(bad_pad)
            ident = self._lib_idents[int(lib_ids[i])]
            worst = float(x.detach()[i].masked_fill(valid_rows[i], 0).abs().max())
            raise RuntimeError(
                f'{ident}: carrier row {i} carries nonzero PAD columns (max |x| {worst:.3g}). '
                f'Pads are pinned to 0 along the whole trajectory, so this row was not '
                f'produced in {ident}\'s layout; scoring it would read another chart\'s '
                f'coordinates.')
        raise RuntimeError(
            'the batch\'s state_mask disagrees with this energy\'s carrier layout -- the '
            'conditions file was built against a different member set')

    def member_groups(self, mol_batch) -> List[Tuple[str, torch.Tensor]]:
        """``[(identifier, row indices)]`` per molecule present, in library order.

        For PER-MEMBER work that genuinely needs one chart at a time -- per-molecule
        evaluation, the oracle below -- not for scoring, which is one pass. Built by a stable
        sort on the resolved library index, with the same identity checks the energy runs
        (the pad check needs a state and is the energy's).
        """
        n = int(mol_batch.num_graphs)
        lib_ids, _ = self._resolve_rows(mol_batch, n, geometry=False)
        order = torch.argsort(lib_ids, stable=True)
        uniq, counts = torch.unique_consecutive(lib_ids.index_select(0, order),
                                                return_counts=True)
        return [(self._lib_idents[int(u)], rows)
                for u, rows in zip(uniq.tolist(), torch.split(order, counts.tolist()))]

    # ------------------------------------------------------------------ energy

    def energy(self, x, mol_batch=None, log_temperature=None, return_exp: bool = False,
               keep_grads: bool = False, internal_oom_recovery=None):
        """E/T per sample, each row through ITS OWN chart. See `ConformerTorsions.energy`."""
        if self._carrier is not None and mol_batch is None:
            raise RuntimeError(
                'a carrier-state energy needs mol_batch to know which member owns each row; '
                'without it there is no chart to score against')
        if self.n_charts <= 1 or mol_batch is None:
            # the single-molecule case is the parent's, byte for byte -- no library, no
            # gather, and no behaviour to diverge
            return super().energy(x, mol_batch, log_temperature, return_exp,
                                  keep_grads=keep_grads,
                                  internal_oom_recovery=internal_oom_recovery)
        return self._energy_one_pass(x, mol_batch, log_temperature, return_exp, keep_grads)

    def _row_log_temperature(self, log_temperature, n: int, device) -> torch.Tensor:
        if log_temperature is None:
            log_temperature = torch.tensor(self.log_temperature)
        log_T = torch.as_tensor(log_temperature, dtype=self.dtype, device=device).flatten()
        if log_T.numel() == 1:
            return log_T.expand(n)
        if int(log_T.numel()) != n:
            raise RuntimeError(f'{log_T.numel()} log-temperatures for {n} rows')
        return log_T

    def _energy_one_pass(self, x, mol_batch, log_temperature, return_exp: bool,
                         keep_grads: bool):
        """The whole mixed batch in one build and one force-field evaluation.

        Term for term the member's `energy` (`potential_energy` + `jacobian_energy` - T *
        `log_chart_jacobian`, divided by T), in the same association, with each piece read
        per row instead of from one chart:

          * the tree and (r, theta, phi) from the condition graph (`batch_tree`,
            `state_to_dof`) -- on a carrier its reconstruction map already points at each
            row's own columns, so no slice back to member width is needed;
          * U from the force field GATHERED for these rows (`ForceFieldLibrary.gather`), plus
            the stereo lock read off the rows' own ``ctree_stereo_*``
            (`stereo_lock.batch_lock_energy`), both before the clip;
          * the box wall over `_lin_free_idx`, the carrier's non-phi columns: pads are
            exactly 0 (checked), so relu adds exactly 0 there and the wall is each member's.
            A bounded double-bond dihedral sits in the theta region, so it is walled here
            exactly as the member's `bounding_energy` walls it, and its scale is the graph's
            per-atom ``ctree_ph_scale``;
          * the transverse DISC wall in rho, per ATOM off ``ctree_transverse`` (`_disc_wall`),
            added to the box before the T pre-multiplication exactly as the member's
            `bounding_energy` adds it -- so a pad, which owns no atom, cannot reach it;
          * log J_BAT by `log_jacobian`, an index_add over the rows' own bond/angle entries,
            with the transverse rows measured as log sinc(rho) under the same mask the build
            used (the member's `_log_jac`);
          * log|dq/dx| as the per-molecule CONSTANT from the library -- not the graph-derived
            sum of log|scale|, which overcounts at `torsion`, where one column drives
            several dihedral rows.

        With `return_exp` the baked potential is ``clip(U + lock) + wall`` at T = 1 from this
        same pass --
        the member's `potential_energy(x, 1)`, without a second evaluation.
        """
        from mxtaltools.conformers.builder import build, log_jacobian
        from mxtaltools.conformers.energy import intramolecular_energy

        from energies.conformer_data import (batch_tree, dummy_frame_mask, state_to_dof,
                                             transverse_mask)

        n = int(x.shape[0])
        lib_ids, ptr = self._resolve_rows(mol_batch, n, x)
        temperature = 10 ** self._row_log_temperature(log_temperature, n, x.device)

        grad_ctx = torch.enable_grad() if keep_grads else torch.no_grad()
        with grad_ctx:
            xs = x.to(self.dtype)
            tree = batch_tree(mol_batch)
            r, th, ph = state_to_dof(mol_batch, xs)
            # None on a batch with no linear centre, which keeps it on exactly the code path
            # it was on before transverse rows were admitted
            tv = transverse_mask(mol_batch)
            # None likewise on a batch with no dummy-frame row. Not passed to log_jacobian: a
            # dummy moves the frame phi is measured in, not the volume element
            pos = build(tree, r, th, ph, transverse=tv,
                        dummy_frame=dummy_frame_mask(mol_batch))
            e = intramolecular_energy(tree, pos, self._lib.gather(lib_ids, ptr[:-1]))
            if self.stereo_coeff > 0.0:
                # THE STEREO LOCK, graph-natively off ctree_stereo_* -- the sign read off each
                # graph's own reference, which `_resolve_rows` has just compared atom by atom
                # with the member's table. Inside the clip, as the member adds it
                # (ConformerTorsions.potential_energy), so the baked value carries it too.
                from energies.stereo_lock import batch_lock_energy
                e = e + batch_lock_energy(mol_batch, pos, self.stereo_coeff)
            if self.energy_clip is not None:
                # the force field (and the lock) only, before the wall --
                # ConformerTorsions.potential_energy
                from mxtaltools.common.utils import log_rescale_positive
                if self.energy_clip_origin == 'absolute':
                    cutoff = self.energy_clip
                else:
                    # each row's own member's floor (install_clip_floor)
                    cutoff = self._clip_floor_of_lib.index_select(0, lib_ids) + self.energy_clip
                e = log_rescale_positive(e, cutoff)
            baked = e
            if self._lin_free_idx.numel():
                xl = xs.index_select(-1, self._lin_free_idx)
                wall = self.bounding_coeff * (torch.relu(xl - 1.0) ** 2
                                              + torch.relu(-(xl + 1.0)) ** 2).sum(-1)
                if tv is not None:
                    wall = wall + self._disc_wall(mol_batch, th, ph, n)
                baked = e + wall
                e = e + wall * temperature
            e = e + (-temperature * log_jacobian(tree, r, th, None if tv is None else ph,
                                                 transverse=tv))
            e = e - temperature * self._lib.log_chart.index_select(0, lib_ids)
        # BEFORE the division, as the parent stores it: the crystal convention for
        # `gfn_energy`, which the eval publishes as 'Mean Sample Energy'
        gfn_e = e
        e = e / temperature
        if not return_exp:
            return e

        from energies.conformer_data import set_batch_states
        return e, set_batch_states(mol_batch, x.detach(), baked.detach(),
                                   gfn_energy=gfn_e.detach(), periodic=self.periodic_dims)

    def _disc_wall(self, mol_batch, th, ph, n: int) -> torch.Tensor:
        """``bounding_coeff * sum relu(rho - rho_wall)^2`` per ROW, ``[n]``, graph-natively.

        The disc term of the member's `bounding_energy`, which a box over the columns cannot
        express: the box is a SQUARE in (x_u, x_v) and the transverse domain a disc, so
        without this term the one-pass energy would score a different target from every
        member (see `ConformerTorsions.bounding_energy` for the measured escape route).

        Formed from the reconstruction this pass already made, not from state columns. On a
        transverse atom `state_to_dof` returns u in the atom's theta slot (unclamped: the clamp
        exempts it) and v in its phi slot, each ``ref + scale * x`` -- the same affine map, the
        same reference and the same signed scale `_transverse_rho2` applies -- so rho is the
        member's number. PER ATOM, off ``ctree_transverse``, which `_resolve_rows` has just
        compared atom by atom with the member's chart; a pad owns no atom and cannot reach it.
        """
        from energies.conformer_data import dof_rank

        flag = mol_batch.ctree_transverse.reshape(-1).bool()
        rank = dof_rank(mol_batch)
        # `th` is aligned with the rank >= 2 atoms and `ph` with the rank >= 3 atoms, both in
        # atom order, so the per-atom flag restricted to each selects the same atoms in the
        # same order: u and v pair by position.
        u = th[flag[rank >= 2]]
        v = ph[flag[rank >= 3]]
        if u.shape != v.shape:
            # a flagged frame seed (rank 2) has no phi row to carry v. build() refuses that
            # too, but a shape of 1 would BROADCAST here rather than fail, pairing one v with
            # every u
            raise RuntimeError(
                f'{int(u.numel())} transverse u components against {int(v.numel())} v: a '
                f'flagged atom owns no torsion row, so its bend has no second component')
        rho = torch.sqrt((u * u + v * v).clamp_min(1e-24))
        disc = torch.zeros(n, dtype=th.dtype, device=th.device).index_add(
            0, mol_batch.batch[flag], torch.relu(rho - self.rho_wall) ** 2)
        return self.bounding_coeff * disc

    def _energy_per_member(self, x, mol_batch, log_temperature=None,
                           keep_grads: bool = False) -> torch.Tensor:
        """The per-member LOOP the one-pass energy replaced, kept as its REFERENCE ORACLE.

        Each molecule's rows through that member's own `energy`, in its own chart and at its
        own width, reassembled into batch order by a gather (differentiable, so the
        `keep_grads` path can be compared too). Not a scoring path: it pays the fixed
        per-molecule cost the one-pass energy exists to remove, and it does not repeat the
        pad check. Tests and the timing comparison call it; nothing in training does.
        """
        n = int(x.shape[0])
        log_T = self._row_log_temperature(log_temperature, n, x.device)
        es, idxs = [], []
        for ident, rows in self.member_groups(mol_batch):
            rows = rows.to(x.device)
            xi = x.index_select(0, rows)
            if self._carrier is not None:
                xi = self._carrier.from_carrier(ident, xi)
            es.append(self._members[ident].energy(xi, None, log_T.index_select(0, rows),
                                                  keep_grads=keep_grads))
            idxs.append(rows)
        taken = torch.cat(idxs)
        inverse = torch.empty(n, dtype=torch.long, device=taken.device)
        inverse[taken] = torch.arange(n, dtype=torch.long, device=taken.device)
        return torch.cat(es)[inverse]

    # ------------------------------------------------------------------ prebuilt rewards

    def prebuilt_sample_to_reward(self, mols, temperature, raw_latents=None):
        """log reward from a baked `conformer_energy`, with EACH ROW'S OWN measure terms.

        The parent's arithmetic, ``-(U / T) + log J + log|dq/dx|``, with both measure terms
        read per row instead of from the reference member. Those terms span ~1 nat across
        ordinary QM9 molecules at `torsion` and far more at `full`, so using one member's for
        the whole batch is a per-condition log Z error of that size -- silent, and pointed
        straight at the quantity the conditional route is trying to learn.

        `flex` and `full`: log J moves with the sample, so it is recomputed graph-natively
        from the stored state, in one pass over the mixed batch -- this used to be a loop
        over the molecules present, on EVERY backward draw. `torsion` and `dihedral`: r and
        theta are frozen, log J is each member's constant, and no geometry is read.
        """
        if raw_latents is not None:
            # the trainer's replay re-score passes raw_latents on every route; only the crystal
            # energy scores a bounding term from them, and a conformer state has none
            raise ValueError('the conformer energy has no raw latents; got a non-None raw_latents')
        e = getattr(mols, 'conformer_energy', None)
        if e is None:
            raise AttributeError(
                'prebuilt_sample_to_reward needs a `conformer_energy` graph attribute; the '
                'prior/replay prep must attach it (see build_conformer_conditions.py)')
        e = e.flatten()
        n = int(e.shape[0])
        if self.n_charts <= 1:
            return super().prebuilt_sample_to_reward(mols, temperature)
        t = torch.as_tensor(temperature, dtype=e.dtype, device=e.device).flatten()
        chart = self._lib.log_chart.to(device=e.device, dtype=e.dtype)
        if self._log_jac_const_of_lib is not None:
            lib_ids, _ = self._resolve_rows(mols, n, geometry=False)
            lj = self._log_jac_const_of_lib.to(device=e.device, dtype=e.dtype)
            return -(e / t) + lj.index_select(0, lib_ids) + chart.index_select(0, lib_ids)

        from mxtaltools.conformers.builder import log_jacobian

        from energies.conformer_data import (batch_states, batch_tree, state_to_dof,
                                             transverse_mask)
        state = batch_states(mols)
        lib_ids, _ = self._resolve_rows(mols, n, state)
        r, th, ph = state_to_dof(mols, state)
        # the transverse rows' measure is log sinc(rho), which reads BOTH components -- the
        # same mask and the same phi the energy's own log J takes, or the prebuilt reward and
        # energy() disagree on every row carrying a linear bend
        tv = transverse_mask(mols)
        log_j = log_jacobian(batch_tree(mols), r, th, None if tv is None else ph,
                             transverse=tv).to(e.dtype)
        return -(e / t) + log_j + chart.index_select(0, lib_ids)

    # ------------------------------------------------------------------ reporting

    def describe(self) -> str:
        if self._carrier is not None:
            # the parent's describe reads THIS object's chart, which on a carrier is a
            # dispatcher's layout, not a molecule -- so describe each member instead
            return '\n'.join([f'MOLECULE SET: {self.n_charts} charts on a CARRIER state'] +
                             [m.describe() for m in self._members.values()] +
                             [self._carrier.describe()])
        head = super().describe()
        return (f'{head}\n   MOLECULE SET: {self.n_charts} charts, k = {self.data_ndim}, '
                f'each row scored through its own; reference member '
                f'{self._member_smiles[next(iter(self._members))]!r}')


# ---------------------------------------------------------------- carrier guards
#
# On a carrier `self` is a DISPATCHER: `data_ndim`, `_free_block` and `_lin_free_idx` describe
# the width-K layout, while `_M`, `spec`, `r0` and the force field are still the reference
# member's. Every inherited method below reads the latter, so on a carrier it would compute the
# reference molecule's answer for a state that is not in its chart -- a shape error at best,
# and at worst (K equal to the reference's k) a plausible wrong number. Refused by name; the
# per-member version is `self._members[ident].<method>` on `carrier.from_carrier(ident, x)`.
_CHART_METHODS = (
    'dof_from_state', 'build_positions', 'bounding_energy', '_transverse_rho2',
    'transverse_crossings', 'state_from_dof', 'prior_dof_types', 'torsion_groups',
    'improper_phi_rows', 'held_phi_rows', 'improper_phi_sigma', 'sibling_jitter_sigma',
    'ring_blocks',
    'ring_frame_groups', 'prior_log_prob', 'thermal_rtheta_sigma', 'sample_prior_states',
    'potential_energy', 'jacobian_energy', 'brute_force_log_z', 'sample', '_batch',
    '_log_jac', '_tiled_transverse', '_build', '_tiled_dummy', 'dummy_frame_crossings',
    'torsion_frame_atoms',
)


def _guard(name):
    parent = getattr(ConformerTorsions, name)

    def method(self, *args, **kwargs):
        if getattr(self, '_carrier', None) is not None:
            raise NotImplementedError(
                f'MultiConformerTorsions.{name} on a CARRIER state: this object is a '
                f'dispatcher over {self.n_charts} charts, not a chart. Call it on the member, '
                f'self._members[ident].{name}, with carrier.from_carrier(ident, x).')
        return parent(self, *args, **kwargs)

    method.__name__ = name
    method.__doc__ = parent.__doc__
    return method


for _name in _CHART_METHODS:
    setattr(MultiConformerTorsions, _name, _guard(_name))
del _name
