"""The CARRIER state: one fixed-width layout that holds every molecule of a mixed-k set.

WHY. A conformer state is ``cat[r, theta, phi]`` over the columns that survive at the run's
`level`, so its width k = 3N-6 (at `full`) differs per molecule, and so does which columns
wrap. The GFN fixes both at construction. Rather than make every trajectory, loss, buffer and
metric ragged, each molecule's state is PLACED into a shared width-K carrier:

    carrier = [ r block (R_max) | theta block (T_max) | phi block (P_max) ]

A molecule's j-th state column goes to ``offset[region(j)] + rank of j within its region``,
and every other column is PAD. Two properties follow by construction and are what make the
carrier cheap:

  * the periodic mask is COLUMN-CONSTANT -- every carrier column in the phi block wraps, for
    every molecule -- so the GFN's single `angular_mask` is still the truth;
  * when every member has the same per-column block codes the layout is the IDENTITY
    (K = k, column j goes to j), so a same-layout set is byte-identical to the pre-carrier
    route.

A TRANSVERSE COLUMN (block code 3, one component of a linear bend's (u, v) pair) is placed in
the THETA region, both u and v. It is non-periodic, walled by the same box and scaled by the
same `delta_theta_max` as theta, so the region's three shared properties -- no wrap, the box
wall, the scale -- all hold for it, and the phi region keeps only columns that wrap. A fourth
block would buy nothing the theta region lacks and costs width (K 79 against 75 on the
3000-molecule QM9 census). What the region code LOSES is which columns are u and v: that is
kept per member in `kind` (the member's own code, 3 included), because the region code is
right for periodicity, walls and the checkpoint's `block_width` stamp but wrong for anything
that reports or conditions on what a column IS. The pair's own properties -- the disc wall in
rho and the log sinc measure -- are per-atom and read off the condition graph
(`ctree_transverse`), never off a carrier column.

A BOUNDED DOUBLE-BOND DIHEDRAL (block code 4, ``ConformerTorsions(double_bond_box_deg=W)``:
the proper dihedral about a double bond the stereo lock holds) is placed in the THETA region
by the same rule and for the same two properties: it does not wrap and it takes the box wall.
Its SCALE is its own (W degrees, not `delta_theta_max`), which costs nothing here: no reader
takes a scale from the region, each row's scale travels with its atom on the condition graph
(``ctree_ph_scale``) and the chart constant per member in the library. Its row is still a
dihedral, read from the atom's phi slot; only its COLUMN moved. `kind` keeps the code.

A SIBLING OFFSET (block code 5, ``ConformerTorsions(sibling_offset_box_deg=W)``: a follower's
dihedral minus its group's leader's) is placed in the THETA region likewise: it does not wrap
and it takes the box wall, and its scale travels on the graph. The LEADER's column, which
turns every row of its group, keeps code 2 and stays in the phi region, so that region still
holds only columns that wrap. A follower's dihedral reads two carrier columns, its own in the
theta region and its leader's in the phi region; the graph names both (``ctree_ph_col``,
``ctree_ph_lead_col``) and `carrier_pad_condition` remaps both.

PADS ARE NOT COORDINATES. They are pinned to exactly 0 along the whole trajectory
(``ConformerGFN._pin_dead``), excluded from every log-prob sum (``state_mask``), never
tokenised by the policy, and the energy REFUSES a row whose pads are not exactly 0 -- a
positive check that each row is being read through its own chart, not a mask that can be
dropped somewhere and read as a real coordinate at value 0.

See docs/design/ragged_multi_molecule_state.md.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import torch

#: block codes, as ``ConformerTorsions._free_block`` spells them
BLOCKS = (0, 1, 2)           # r, theta, phi -- the carrier's REGIONS
TRANSVERSE = 3
BOUNDED_DIHEDRAL = 4
SIBLING_OFFSET = 5
#: member block code -> the carrier region its column is placed in (module docstring)
REGION = {0: 0, 1: 1, 2: 2, TRANSVERSE: 1, BOUNDED_DIHEDRAL: 1, SIBLING_OFFSET: 1}
#: labels for a member block code, for reporting; -1 is a pad
KIND_NAMES = {0: 'r', 1: 'theta', 2: 'phi', TRANSVERSE: 'transverse',
              BOUNDED_DIHEDRAL: 'double_bond', SIBLING_OFFSET: 'sibling_offset'}


class CarrierLayout:
    """Where each member's state columns live in the shared width-K carrier."""

    def __init__(self, members: Mapping[str, object]):
        if not members:
            raise ValueError('CarrierLayout needs at least one member')
        kinds, blocks = {}, {}
        for ident, en in members.items():
            fb = np.asarray(en._free_block, dtype=np.int64).reshape(-1)
            unknown = sorted(set(fb.tolist()) - set(REGION))
            if unknown:
                # a code with no region would be placed nowhere, or -- read as an index --
                # into some other block's columns
                raise ValueError(f'{ident}: block code(s) {unknown} have no carrier region')
            kinds[ident] = fb
            blocks[ident] = np.asarray([REGION[int(b)] for b in fb], dtype=np.int64)

        # THE IDENTITY is decided on the members' OWN codes, before any placement: when every
        # member has the same `_free_block`, column j means the same thing for every member,
        # so it goes to j. Two consequences, both deliberate. A set of one nitrile's
        # stereoisomers (identical codes, v in the phi part of the state) stays the identity
        # instead of becoming a permuted carrier for no reason. And a set whose members
        # DISAGREE on a column's kind is never the identity: the identity route keeps no
        # layout, so the dispatcher's own chart methods -- `bounding_energy`, the eval
        # statistics, the column labels -- read the reference member's `_free_block` for
        # EVERY row, and would read another member's theta as a bend, or its bend as a theta.
        # With no transverse column this is exactly the old test (every column in its block).
        first = next(iter(kinds.values()))
        self.is_identity = all(len(fb) == len(first) and (fb == first).all()
                               for fb in kinds.values())
        if self.is_identity:
            self.K = int(len(first))
            #: per carrier column, its REGION code (0 r, 1 theta -- transverse, bounded
            #: double-bond dihedrals and sibling offsets included -- 2 phi): what periodicity, the box wall and the `block_width` stamp read. In
            #: column order, which on the identity is the members' own order
            self.free_block = next(iter(blocks.values()))
            self.block_width = [int((self.free_block == b).sum()) for b in BLOCKS]
            self.offsets = None
            self.cols: Dict[str, np.ndarray] = {ident: np.arange(self.K, dtype=np.int64)
                                                for ident in kinds}
        else:
            self.block_width = [max(int((fb == b).sum()) for fb in blocks.values())
                                for b in BLOCKS]
            self.offsets = [0, self.block_width[0], self.block_width[0] + self.block_width[1]]
            self.K = int(sum(self.block_width))
            self.free_block = np.repeat(np.asarray(BLOCKS), self.block_width)
            self.cols = {}
            for ident, fb in blocks.items():
                seen = [0, 0, 0]
                cols = np.empty(len(fb), dtype=np.int64)
                for j, b in enumerate(fb):
                    cols[j] = self.offsets[int(b)] + seen[int(b)]
                    seen[int(b)] += 1
                self.cols[ident] = cols
        #: per member, ``[K]``: the member's OWN block code at each of its carrier columns
        #: (3 on u and v), -1 on its pads. PER MEMBER because the same theta-region column can
        #: be one member's theta and another's u; a single per-column code cannot say both.
        self.kinds: Dict[str, np.ndarray] = {}
        for ident, fb in kinds.items():
            kd = np.full(self.K, -1, dtype=np.int64)
            kd[self.cols[ident]] = fb
            self.kinds[ident] = kd

    # ------------------------------------------------------------------ per member

    def k(self, ident: str) -> int:
        return int(len(self.cols[ident]))

    def valid(self, ident: str) -> np.ndarray:
        """``[K]`` bool, True on the carrier columns this member owns."""
        v = np.zeros(self.K, dtype=bool)
        v[self.cols[ident]] = True
        return v

    def pad_cols(self, ident: str) -> np.ndarray:
        return np.flatnonzero(~self.valid(ident))

    def kind(self, ident: str) -> np.ndarray:
        """``[K]`` long: the member's own block code per carrier column (3 = u or v), -1 pad."""
        return self.kinds[ident]

    def column_kinds(self) -> list:
        """Per carrier column, the sorted member codes that occupy it (pads excluded).

        A theta-region column is ``(1,)``, ``(3,)`` or ``(1, 3)`` -- theta for every member
        owning it, a bend component for every one, or theta for some and a bend for others.
        What a pooled per-column reading over a mixed batch is a reading OF.
        """
        tab = np.stack(list(self.kinds.values()))
        return [tuple(sorted(set(c[c >= 0].tolist()))) for c in tab.T]

    def column_label(self, j: int) -> str:
        """``'theta'``, ``'transverse'`` or ``'theta|transverse'`` -- `column_kinds` as text."""
        return '|'.join(KIND_NAMES[int(c)] for c in self.column_kinds()[int(j)]) or 'pad'

    def col_map(self, ident: str) -> np.ndarray:
        """``[K]`` long: the member column at each carrier column, -1 on a pad."""
        m = np.full(self.K, -1, dtype=np.int64)
        m[self.cols[ident]] = np.arange(len(self.cols[ident]))
        return m

    def to_carrier(self, ident: str, x: torch.Tensor) -> torch.Tensor:
        """``[n, k_member] -> [n, K]``, pads exactly 0."""
        x = torch.as_tensor(x)
        if x.shape[-1] != self.k(ident):
            raise ValueError(f'{ident}: state has {x.shape[-1]} columns, member has '
                             f'{self.k(ident)}')
        out = x.new_zeros(*x.shape[:-1], self.K)
        idx = torch.as_tensor(self.cols[ident], device=x.device)
        return out.index_copy(-1, idx, x)

    def from_carrier(self, ident: str, x: torch.Tensor) -> torch.Tensor:
        """``[n, K] -> [n, k_member]``. Differentiable (a gather)."""
        idx = torch.as_tensor(self.cols[ident], device=x.device)
        return x.index_select(-1, idx)

    def describe(self) -> str:
        w = self.block_width
        lines = [f'   CARRIER K = {self.K}  (r {w[0]} | theta {w[1]} | phi {w[2]}; '
                 f'transverse u/v, bounded double-bond dihedrals and sibling offsets sit in '
                 f'the theta region)'
                 + ('  -- identity, every member has the same per-column block codes'
                    if self.is_identity else '')]
        for ident, c in self.cols.items():
            n_tv = int((self.kinds[ident] == TRANSVERSE).sum())
            n_db = int((self.kinds[ident] == BOUNDED_DIHEDRAL).sum())
            n_so = int((self.kinds[ident] == SIBLING_OFFSET).sum())
            lines.append(f'      {ident}: k = {len(c)}, {self.K - len(c)} pad column(s)'
                         + (f', {n_tv} transverse' if n_tv else '')
                         + (f', {n_db} bounded double-bond dihedral(s)' if n_db else '')
                         + (f', {n_so} sibling offset(s)' if n_so else ''))
        return '\n'.join(lines)


#: the per-atom condition-graph fields that name a STATE COLUMN, which `carrier_pad_condition`
#: rewrites into carrier columns. The last is present only on a graph built under
#: `sibling_offset_box_deg` (energies.conformer_data.SIBLING_LEAD_FIELDS).
OPTIONAL_COLUMN_FIELDS = ('ctree_ph_lead_col',)
CARRIER_COLUMN_FIELDS = ('ctree_r_col', 'ctree_th_col', 'ctree_ph_col') + OPTIONAL_COLUMN_FIELDS


def _remap(col: torch.Tensor, cols: np.ndarray) -> torch.Tensor:
    """Member column indices -> carrier column indices, keeping the -1 sentinel."""
    col = torch.as_tensor(col).clone()
    hit = col >= 0
    if bool(hit.any()):
        table = torch.as_tensor(cols, dtype=col.dtype)
        col[hit] = table[col[hit]]
    return col


def carrier_pad_condition(mol, layout: CarrierLayout, ident: str, member,
                          atoms: Optional[np.ndarray] = None,
                          mask: Optional[np.ndarray] = None,
                          R: Optional[int] = None):
    """A member's condition graph, re-expressed in the carrier layout. Returns a COPY.

    Rewrites exactly the fields that name a STATE COLUMN:

      * ``ctree_r_col`` / ``ctree_th_col`` / ``ctree_ph_col`` -- the graph-native
        reconstruction map, so ``state_to_dof`` on a mixed batch reads each row's own
        columns out of the carrier (-1 stays -1) -- and ``ctree_ph_lead_col``, a sibling
        offset row's second column, on a graph that carries it (`CARRIER_COLUMN_FIELDS`);
      * ``n_torsions`` -> K, so the batch is uniform-width and collates;
      * ``state_mask`` ``[1, K]`` -- the per-row validity mask;
      * ``dof_static`` ``[1, K * F]`` -- the handcrafted per-column features, pads 0;
      * ``dof_atoms`` / ``dof_mask`` (when ``atoms``/``mask`` are given) padded to K columns
        and ``R`` rows per column, pad columns fully masked.

    ``ctree_state_col`` is NOT remapped: it indexes ROTATABLE AXES (``energy.mask``), which
    coincide with state columns only at `torsion` -- where the phi block is the whole state
    and the carrier map is the identity on it.
    """
    from energies.dof_features import state_features
    cols = layout.cols[ident]
    if len(cols) != int(member.data_ndim):
        raise ValueError(f'{ident}: layout has {len(cols)} columns, member has '
                         f'{member.data_ndim}')
    out = mol.__copy__()
    for name in CARRIER_COLUMN_FIELDS:
        if name in OPTIONAL_COLUMN_FIELDS and getattr(mol, name, None) is None:
            continue
        setattr(out, name, _remap(getattr(mol, name), cols))
    out.n_torsions = torch.tensor([layout.K], dtype=torch.long)
    out.state_mask = torch.as_tensor(layout.valid(ident)).reshape(1, -1)

    f = np.asarray(state_features(member, None), dtype=np.float64)
    fs = np.zeros((layout.K, f.shape[1]), dtype=np.float64)
    fs[cols] = f
    out.dof_static = torch.as_tensor(fs, dtype=torch.get_default_dtype()).reshape(1, -1)

    if atoms is not None:
        atoms, mask = np.asarray(atoms), np.asarray(mask, dtype=bool)
        R = int(R if R is not None else atoms.shape[1])
        if atoms.shape[1] > R:
            raise ValueError(f'{ident}: {atoms.shape[1]} rows per column exceeds R = {R}')
        pa = np.zeros((layout.K, R, atoms.shape[2]), dtype=np.int64)
        pm = np.zeros((layout.K, R), dtype=bool)
        pa[cols, :atoms.shape[1]] = atoms
        pm[cols, :mask.shape[1]] = mask
        out.dof_atoms = torch.as_tensor(pa).reshape(1, -1)
        out.dof_mask = torch.as_tensor(pm).reshape(1, -1)
    return out


def check_carrier_convention(condition, layout: CarrierLayout, ident: str, member,
                             n: int = 64, tol: float = 1e-9, seed: int = 0) -> float:
    """``check_state_convention`` for a carrier-padded condition.

    Draws member-width states, places them in the carrier, and asserts the carrier graph
    reconstructs the SAME geometry the member energy builds from the member-width states.
    A wrong column map produces plausible geometry, so this is checked rather than trusted.
    """
    from mxtaltools.dataset_utils.utils import collate_data_list

    from energies.conformer_data import states_to_positions
    g = torch.Generator().manual_seed(seed)
    xs = torch.rand((n, layout.k(ident)), generator=g, dtype=member.dtype) * 2 - 1
    full = collate_data_list([condition] * (2 * n))
    batch = full.subsample_new_batch(np.arange(n))
    pos_graph = states_to_positions(batch, layout.to_carrier(ident, xs))
    pos_energy = member.build_positions(xs)
    err = (pos_graph - pos_energy).abs().max().item()
    if not err < tol:
        raise AssertionError(f'{ident}: carrier graph disagrees with the member chart by '
                             f'{err:.3g} A (tol {tol:g}); the column map is wrong')
    return err
