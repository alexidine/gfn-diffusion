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

WHAT MAKES THIS CHEAP. `ConformerTorsions.energy` never reads `mol_batch`; it computes from
`x` and its own chart. So dispatch is a grouping, not a rewrite: split the rows by molecule,
call each member on its own rows, and gather the results back into batch order.

ONE STATE WIDTH FOR THE WHOLE SET. The GFN's state dimension is fixed when it is built. A set
whose members share their block layout uses the reference chart's state unchanged. A MIXED-k
set uses the width-K CARRIER (energies/conformer_carrier.py): each member's columns are placed
in fixed r | theta | phi blocks, pads are pinned to 0, and each row is sliced back to its own
member's columns before scoring -- with a positive check that the pads really are 0 and that
the batch's `state_mask` matches the layout, rather than a removed assertion.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

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
                 **kw):
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
        super().__init__(smiles=smiles[0], **kw)
        for smi, ident in zip(smiles, idents):
            if ident in self._members:
                continue
            member = (self if smi == smiles[0] and ident == idents[0]
                      else ConformerTorsions(smiles=smi, **kw))
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
            self._members[ref_ident] = ConformerTorsions(smiles=smiles[0], **kw)
            self._carrier = layout
            self.data_ndim = layout.K
            self._free_block = layout.free_block
            self._lin_free_idx = torch.as_tensor(np.flatnonzero(layout.free_block != 2),
                                                 dtype=torch.long, device=self.device)

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
        """
        self._by_mol_id = {int(v): k for k, v in dict(registry).items()
                           if k in self._members}

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

    def _groups(self, mol_batch, n: int, device
                ) -> List[Tuple[str, ConformerTorsions, torch.Tensor]]:
        """`(identifier, member, row indices)` per molecule present, first-appearance order."""
        idents = self._row_identifiers(mol_batch, n)
        order: Dict[str, List[int]] = {}
        for i, ident in enumerate(idents):
            order.setdefault(ident, []).append(i)
        unknown = [k for k in order if k not in self._members]
        if unknown:
            raise RuntimeError(
                f'batch carries {len(unknown)} molecule(s) this energy was not built for, '
                f'e.g. {unknown[0]!r}. The energy set and the condition set must be built '
                f'from the same molecule list.')
        return [(k, self._members[k], torch.as_tensor(v, dtype=torch.long, device=device))
                for k, v in order.items()]

    def _member_rows(self, ident: str, x: torch.Tensor, mol_batch,
                     idx: torch.Tensor) -> torch.Tensor:
        """Rows `idx` of a carrier state, read through member `ident`'s own columns.

        The identity layout returns the rows unchanged. On a real carrier, two POSITIVE
        checks before the slice, because a wrong one returns a plausible energy:
          * every PAD column is exactly 0 -- pads are pinned, so a nonzero pad means the row
            was produced for a different member's layout;
          * when the batch carries `state_mask`, each row's mask IS this member's.
        """
        xi = x.index_select(0, idx)
        if self._carrier is None:
            return xi
        lay = self._carrier
        pads = torch.as_tensor(lay.pad_cols(ident), dtype=torch.long, device=x.device)
        if pads.numel() and bool((xi.index_select(1, pads) != 0).any()):
            worst = float(xi.index_select(1, pads).abs().max())
            raise RuntimeError(
                f'{ident}: carrier rows carry nonzero PAD columns (max |x| {worst:.3g}). Pads '
                f'are pinned to 0 along the whole trajectory, so this row was not produced in '
                f'{ident}\'s layout; scoring it would read another chart\'s coordinates.')
        mask = getattr(mol_batch, 'state_mask', None) if mol_batch is not None else None
        if mask is not None:
            want = torch.as_tensor(lay.valid(ident), device=mask.device)
            got = mask.reshape(-1, lay.K).index_select(0, idx.to(mask.device)).bool()
            if not bool((got == want).all()):
                raise RuntimeError(
                    f'{ident}: the batch\'s state_mask disagrees with this energy\'s carrier '
                    f'layout -- the conditions file was built against a different member set')
        return lay.from_carrier(ident, xi)

    @staticmethod
    def _regroup(parts: Sequence[torch.Tensor], index: Sequence[torch.Tensor],
                 n: int) -> torch.Tensor:
        """Concatenated per-group results -> batch order, DIFFERENTIABLY.

        A gather, not an in-place scatter into an empty tensor: `keep_grads=True` is a real
        call path (the pathwise-gradient forward branch), and writing into a fresh tensor
        would sever it silently -- the values would be right and no gradient would flow.
        """
        taken = torch.cat(list(index))
        inverse = torch.empty(n, dtype=torch.long, device=taken.device)
        inverse[taken] = torch.arange(n, dtype=torch.long, device=taken.device)
        return torch.cat(list(parts))[inverse]

    # ------------------------------------------------------------------ energy

    def energy(self, x, mol_batch=None, log_temperature=None, return_exp: bool = False,
               keep_grads: bool = False, internal_oom_recovery=None):
        """E/T per sample, each row through ITS OWN chart. See `ConformerTorsions.energy`."""
        n = int(x.shape[0])
        if self._carrier is not None and mol_batch is None:
            raise RuntimeError(
                'a carrier-state energy needs mol_batch to know which member owns each row; '
                'without it there is no chart to score against')
        if self.n_charts <= 1 or mol_batch is None:
            # the single-molecule case is the parent's, byte for byte -- no grouping, no
            # gather, and no behaviour to diverge
            return super().energy(x, mol_batch, log_temperature, return_exp,
                                  keep_grads=keep_grads,
                                  internal_oom_recovery=internal_oom_recovery)

        groups = self._groups(mol_batch, n, x.device)
        if log_temperature is None:
            log_temperature = torch.tensor(self.log_temperature)
        log_T = torch.as_tensor(log_temperature, dtype=self.dtype, device=self.device).flatten()
        if log_T.numel() == 1:
            log_T = log_T.expand(n)

        es, bakes, idxs = [], [], []
        one = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        for ident, member, idx in groups:
            xi = self._member_rows(ident, x, mol_batch, idx)
            es.append(member.energy(xi, None, log_T[idx], return_exp=False,
                                    keep_grads=keep_grads))
            idxs.append(idx)
            if return_exp:
                with torch.no_grad():
                    bakes.append(member.potential_energy(xi.detach(), one))
        e = self._regroup(es, idxs, n)
        if not return_exp:
            return e

        # `set_batch_states` only writes attributes, so it runs ONCE over the whole batch --
        # the per-molecule part is the scoring above. `gfn_energy` is the pre-division value
        # the crystal route stores and the eval publishes as 'Mean Sample Energy', recovered
        # here rather than re-derived so the two routes keep meaning the same thing.
        from energies.conformer_data import set_batch_states
        baked = self._regroup(bakes, idxs, n)
        temperature = (10 ** log_T).to(e.dtype)
        return e, set_batch_states(mol_batch, x.detach(), baked,
                                   gfn_energy=(e * temperature).detach(),
                                   periodic=self.periodic_dims)

    # ------------------------------------------------------------------ prebuilt rewards

    def prebuilt_sample_to_reward(self, mols, temperature):
        """log reward from a baked `conformer_energy`, with EACH ROW'S OWN measure terms.

        At `level='torsion'` this is the parent's arithmetic with two per-molecule constants
        gathered per row instead of taken from the reference member. Those constants span
        ~1 nat across ordinary QM9 molecules, so using one member's for the whole batch is a
        per-condition log Z error of that size -- silent, and pointed straight at the quantity
        the conditional route is trying to learn.
        """
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
        if self.log_jacobian_const is None:
            # STATE-DEPENDENT log J -- `flex` and `full`, where r and theta are free so the
            # BAT term prod r^2 sin(theta) moves with the sample and cannot be a per-molecule
            # constant. This used to raise, which made `full` unreachable on the ENTIRE
            # conditional route: the conditional arm is always a MultiConformerTorsions, and
            # the anchor-buffer seed calls this before the first training step.
            #
            # The dispatch is the same grouping `energy()` uses, and each member is a whole
            # ConformerTorsions with its own `_batch` tree cache -- so the "per-member cache
            # this class does not keep" was already there, one level down. Rebuilding per
            # member rather than per row keeps it one `build` call per distinct molecule.
            from energies.conformer_data import batch_states
            state = torch.as_tensor(batch_states(mols), dtype=self.dtype,
                                    device=self.device)
            if int(state.shape[0]) != n:
                raise RuntimeError(
                    f'{state.shape[0]} baked states against {n} baked energies; the prebuilt '
                    f'rows disagree with themselves')
            parts, idxs = [], []
            for ident, member, idx in self._groups(mols, n, state.device):
                xi = self._member_rows(ident, state, mols, idx)
                r, th, ph = member.dof_from_state(xi)
                tree, _ = member._batch(int(xi.shape[0]))
                # the CHART term is per member too: it is sum(log scale) over that molecule's
                # own free columns, so it differs whenever the free-column sets differ. Adding
                # the reference member's to every row is the same per-condition log Z error
                # this class exists to remove, one level further in.
                parts.append(member._log_jac(tree, r, th, ph, int(xi.shape[0])).flatten()
                             + float(member.log_chart_jacobian))
                idxs.append(idx)
            log_j = self._regroup(parts, idxs, n).to(e.device)
            return -(e / t) + log_j

        idents = self._row_identifiers(mols, n)
        lj = torch.tensor([float(self._members[i].log_jacobian_const) for i in idents],
                          dtype=e.dtype, device=e.device)
        ch = torch.tensor([float(self._members[i].log_chart_jacobian) for i in idents],
                          dtype=e.dtype, device=e.device)
        return -(e / t) + lj + ch

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
    'improper_phi_rows', 'improper_phi_sigma', 'sibling_jitter_sigma', 'ring_blocks',
    'ring_frame_groups', 'prior_log_prob', 'thermal_rtheta_sigma', 'sample_prior_states',
    'potential_energy', 'jacobian_energy', 'brute_force_log_z', 'sample', '_batch',
    '_log_jac', '_tiled_transverse',
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
