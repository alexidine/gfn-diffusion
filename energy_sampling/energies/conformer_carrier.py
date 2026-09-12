"""The CARRIER state: one fixed-width layout that holds every molecule of a mixed-k set.

WHY. A conformer state is ``cat[r, theta, phi]`` over the columns that survive at the run's
`level`, so its width k = 3N-6 (at `full`) differs per molecule, and so does which columns
wrap. The GFN fixes both at construction. Rather than make every trajectory, loss, buffer and
metric ragged, each molecule's state is PLACED into a shared width-K carrier:

    carrier = [ r block (R_max) | theta block (T_max) | phi block (P_max) ]

A molecule's j-th state column goes to ``offset[block(j)] + rank of j within its block``, and
every other column is PAD. Two properties follow by construction and are what make the
carrier cheap:

  * the periodic mask is COLUMN-CONSTANT -- every carrier column in the phi block wraps, for
    every molecule -- so the GFN's single `angular_mask` is still the truth;
  * when every member has the same block counts the layout is the IDENTITY (K = k, column j
    goes to j), so a same-k set is byte-identical to the pre-carrier route.

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
BLOCKS = (0, 1, 2)           # r, theta, phi
TRANSVERSE = 3


class CarrierLayout:
    """Where each member's state columns live in the shared width-K carrier."""

    def __init__(self, members: Mapping[str, object]):
        if not members:
            raise ValueError('CarrierLayout needs at least one member')
        blocks = {}
        for ident, en in members.items():
            fb = np.asarray(en._free_block).reshape(-1)
            if (fb == TRANSVERSE).any():
                # a transverse column is walled like r/theta but is a (u, v) pair with its own
                # measure; it has no block of its own in the carrier yet
                raise NotImplementedError(
                    f'{ident}: carries {int((fb == TRANSVERSE).sum())} transverse column(s); '
                    f'the carrier layout has no transverse block yet')
            blocks[ident] = fb
        self.block_width = [max(int((fb == b).sum()) for fb in blocks.values())
                            for b in BLOCKS]
        self.offsets = [0, self.block_width[0], self.block_width[0] + self.block_width[1]]
        self.K = int(sum(self.block_width))
        self.free_block = np.repeat(np.asarray(BLOCKS), self.block_width)

        self.cols: Dict[str, np.ndarray] = {}
        for ident, fb in blocks.items():
            seen = [0, 0, 0]
            cols = np.empty(len(fb), dtype=np.int64)
            for j, b in enumerate(fb):
                cols[j] = self.offsets[int(b)] + seen[int(b)]
                seen[int(b)] += 1
            self.cols[ident] = cols
        self.is_identity = all(len(c) == self.K and (c == np.arange(self.K)).all()
                               for c in self.cols.values())

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
        lines = [f'   CARRIER K = {self.K}  (r {w[0]} | theta {w[1]} | phi {w[2]})'
                 + ('  -- identity, every member has the same block counts'
                    if self.is_identity else '')]
        for ident, c in self.cols.items():
            lines.append(f'      {ident}: k = {len(c)}, {self.K - len(c)} pad column(s)')
        return '\n'.join(lines)


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
        columns out of the carrier (-1 stays -1);
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
    for name in ('ctree_r_col', 'ctree_th_col', 'ctree_ph_col'):
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
