"""
Single-molecule conformational energy over rotatable-bond torsions.

The unconditional v0.1 target: one fixed molecular graph, local geometry frozen at a
reference conformer, state = one angle per rotatable bond. Fixed dimension, so the
existing ``GFN`` works unchanged -- no variable-dimension machinery needed yet.

Two properties make this a clean first target rather than a compromise:

*The energy is pure sterics.* A rigid rotation about bond (u, v) preserves every bond
length and every bond angle -- the angles at the axis atoms are invariant under rotation
about that axis, and everything within each rotating fragment moves as a body. So the
bonded terms are exactly constant and only nonbonded distances change. That is the LJ
target, not an approximation of it.

*The Jacobian is constant.* With r and theta frozen, ``prod r^2 sin(theta)`` does not
depend on the sampled coordinates, so it contributes an additive constant to log Z and
drops out of the TB residual entirely. Nothing to get wrong.

Energy convention matches ``MolecularCrystal``: ``energy()`` returns E/T, so
``log_reward = -E/T``.

**State units are [-1, 1], not radians.** The GFN's angular latents live on [-1, 1]
representing (-pi, pi] -- see ``_wrap_ang`` in models/gfn.py, which wraps via
``wrap_to_pi(x * pi) / pi``. Feeding radians instead makes the sampler's full-circle
coverage land as +/-1 rad (~57 deg) of the torsion space, with a spurious wrap
discontinuity there. Everything public here takes and returns [-1, 1]; the conversion
to radians happens once, in ``build_positions``.
"""

from typing import Optional

import networkx as nx
import numpy as np
import torch

from energies.base_set import BaseSet
from energies.conformer_data import RingModes


class ConformerTorsions(BaseSet):
    # Free-DoF levels, as freeze sets over InternalParams.CLASSES = ("r","theta","phi").
    # These are NOT a ladder of approximations: freezing a DoF at a constant gives
    # p_full(free | frozen = c0), a conditional slice, which differs from the
    # rigid-constraint ensemble by a state-dependent Fixman factor. `full` is the target;
    # the rest are each some related distribution, useful for staging and regression.
    # See docs/design/internal_dof_ladder.md section 2.
    LEVELS = ("torsion", "dihedral", "flex", "full")
    # matches topology.spec_from_graph's own default, so a flag measured here means the
    # same thing as one measured there
    LINEAR_TOL_DEG = 175.0

    def __init__(self,
                 smiles: str = "CCCCO",
                 device: str = "cpu",
                 log_temperature: float = 0.0,
                 epsilon: float = 0.1,
                 min_separation: int = 3,
                 scale_14: float = 0.5,
                 lj_k_factor: float = 2.5,
                 include_trivial_rotations: bool = False,
                 mmff_reference: bool = True,
                 seed: int = 0,
                 # dtype FOLLOWS torch's default when not given, rather than being
                 # pinned. Measured against float64 on the same DoF draw, float32's
                 # relative error is ~5e-7 on the potential, ~6e-7 on log J and ~1e-6 A on
                 # the closure bond -- pure roundoff, with nothing compounding through the
                 # NeRF chain, and four orders below the closure errors this code reports.
                 # So float32 is the right default for throughput, and `build` folds the
                 # batch into the atom dimension for exactly the huge-batch GPU case where
                 # consumer fp64 runs at a fraction of fp32.
                 #
                 # FOLLOWING THE DEFAULT RATHER THAN PINNING float32 IS THE POINT.
                 # train_conformer.run() sets the global default to float64 deliberately;
                 # a hard-coded float32 here does not fail against that, it SILENTLY
                 # DOWNCASTS -- returning float32 rewards to a float64 policy, which is a
                 # precision change nothing would report. Honouring the default lets each
                 # caller state its own intent, and the exactness gates pass dtype outright.
                 dtype=None,
                 temperature_conditioning: bool = False,
                 log_temperature_range=(-1.0, 1.0),
                 lj_coeff: float = 1.0,
                 *,
                 level: str,
                 delta_r_max: float = 0.30,
                 delta_theta_max: float = 0.50,
                 bounding_coeff: float = 10.0,
                 r_floor: float = 0.50,
                 theta_floor: float = 1.0e-3,
                 ring_jitter_scale: float = 0.1,
                 ring_min_bank_rows: int = 2,
                 force_field: str = 'reference',
                 ring_mode_fill: float = 0.0,
                 ring_pop_temper: float = 1.0,
                 ):
        """
        `level` is keyword-only and has NO default, and there is deliberately no
        ``**kwargs``. Both are load-bearing: a swallowed level is a config that says
        `full` on a run that is `torsion`, with a loss curve that looks fine either way,
        and `**kwargs` is exactly the mechanism that swallows it (it is also what would
        make a `chirality_coeff` passed through energy_config never become an attribute,
        so `set_energy_coeffs`' hasattr guard silently skips the ramp).
        """
        super().__init__()
        if level not in self.LEVELS:
            raise ValueError(f"level must be one of {self.LEVELS}, got {level!r}")
        self.level = level
        # Modeller energy-protocol fields -- see the protocol section below. n_sg/n_zp are
        # 1 (not 0) so the mixed-radix condition_id arithmetic stays valid and collapses
        # to mol_id; condition_library_size is re-set by init_identifiers().
        self.energy_function = 'conformer_torsions'
        self.temperature_conditioning = bool(temperature_conditioning)
        self.log_temperature_range = tuple(log_temperature_range)
        self.lj_coeff = float(lj_coeff)
        self.n_sg, self.n_zp, self.n_molecules = 1, 1, 1
        self.condition_library_size = 1
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from mxtaltools.conformers.builder import collate, measure
        from mxtaltools.conformers.energy import ff_from_reference
        from mxtaltools.conformers.perception import infer_bond_index
        from mxtaltools.conformers.topology import spec_from_graph

        self.device = torch.device(device)
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        self.smiles = smiles
        self.log_temperature = log_temperature

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"could not embed {smiles}")
        if mmff_reference:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        self.mol = mol

        z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
        ref_pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
        bonds = infer_bond_index(z, ref_pos)
        self.spec = spec_from_graph(z, bonds, ref_pos, use_geometry=False)
        # full bond graph in PLACEMENT-SLOT numbering. The tree in `spec` is a spanning
        # tree, so it cannot supply atom degree -- which the handcrafted prior needs as
        # its hybridisation proxy when typing a torsion.
        slot = np.empty(len(z), dtype=np.int64)
        slot[self.spec.perm] = np.arange(len(z))
        self.bond_index_slot = slot[np.asarray(bonds)]
        self.atom_keys = np.stack([np.asarray(self.spec.z), np.bincount(
            self.bond_index_slot.reshape(-1), minlength=len(z))], axis=1)

        # reference internal coordinates, in placement order
        tree1 = collate([self.spec], device=self.device)
        pos1 = torch.tensor(ref_pos[self.spec.perm], dtype=dtype, device=self.device)
        r0, th0, ph0 = measure(tree1, pos1)
        self.r0, self.th0, self.ph0 = r0, th0, ph0
        self.ref_pos = pos1
        # NOTE force_field is read here, before it is stored below -- keep the local name
        self._ff_choice = force_field
        self.ff_single = self._make_ff(tree1, pos1, epsilon, min_separation, scale_14,
                                       lj_k_factor)

        # ---- linearity flags, MEASURED rather than defaulted -------------------------
        # spec_from_graph is called with use_geometry=False above -- required, since a
        # geometry-steered tree is not reproducible at load -- and _linear_mask returns
        # all-False when pos is None. So spec.angle_is_linear has been all-False for every
        # molecule ever run here, and gets written into condition files where it reads as
        # a measurement rather than as an absence. The TREE must stay geometry-free; the
        # FLAGS need not, so they are measured here off the reference conformer, against
        # the tree that was built without it.
        pos_np = np.asarray(ref_pos)[self.spec.perm]

        def _linear(triples):
            triples = np.asarray(triples).reshape(-1, 3)
            if len(triples) == 0:
                return np.zeros(0, dtype=bool)
            u = pos_np[triples[:, 0]] - pos_np[triples[:, 1]]
            v = pos_np[triples[:, 2]] - pos_np[triples[:, 1]]
            ang = np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), (u * v).sum(-1))
            return ang > np.deg2rad(self.LINEAR_TOL_DEG)

        def _typed_linear(triples):
            """Linearity from MMFF's TYPED equilibrium angle -- a function of the GRAPH.

            The measured route reads the reference conformer, so in principle the same
            molecule can get a different chart from a different embedding seed, and `d`
            with it. MMFF types its angles from the graph, so theta0 >= 179.99 is a
            property of the molecule alone and the hazard cannot arise.

            The hazard is LATENT, not observed: measured across nitriles, alkynes, an
            azide, an allene and diphenylacetylene, every linear centre sits at 179.4-180
            deg and every other angle below 121, so the nearest approach to the 175 deg
            threshold is 4.4 deg and no chart varied over six seeds. The two routes agree
            exactly on all ten molecules tested -- which is what makes this swap free.
            test_conformer_levels.py pins that agreement, so a future divergence surfaces
            there rather than as a silently different `d`.

            Indexed by ATOM IDENTITY, never by row: ff.angle_index is the GRAPH angle list
            and is longer than the tree's, so a positional lookup reads the wrong constant.
            """
            triples = np.asarray(triples).reshape(-1, 3)
            if len(triples) == 0:
                return np.zeros(0, dtype=bool)
            ai_ff = self.ff_single.angle_index.detach().cpu().numpy()
            th0_ff = self.ff_single.theta0.detach().cpu().numpy()
            amap = {(int(j), frozenset((int(i), int(k)))): th0_ff[m]
                    for m, (i, j, k) in enumerate(ai_ff)}
            return np.array([np.degrees(amap.get(
                (int(r[1]), frozenset((int(r[0]), int(r[2])))), 0.0)) >= 179.99
                for r in triples])

        # 'mmff' types from the graph, so the chart can be graph-determined. The
        # 'reference' force field measures theta0 off the embedded conformer itself, so
        # there is no graph-determined constant to appeal to and the measured route stands.
        self.linearity_source = 'mmff_typed' if self._ff_choice == 'mmff' else 'measured'
        if self.linearity_source == 'mmff_typed':
            self.angle_is_linear = _typed_linear(self.spec.angle_index)
            self.torsion_frame_is_linear = _typed_linear(
                np.asarray(self.spec.torsion_index)[:, :3])
        else:
            self.angle_is_linear = _linear(self.spec.angle_index)
            self.torsion_frame_is_linear = _linear(
                np.asarray(self.spec.torsion_index)[:, :3])
        self.linearity_verified = True

        # ring membership in PLACEMENT-SLOT numbering, for the prior draw. InternalPrior
        # samples ring systems JOINTLY (a whole observed DoF block) because closure is a
        # hard constraint that a product of marginals is guaranteed to violate; the
        # per-DoF draw below cannot do that, so it has to know when it is being asked to.
        in_ring_orig = np.array([a.IsInRing() for a in mol.GetAtoms()], dtype=bool)
        self.atom_in_ring = np.zeros(len(z), dtype=bool)
        self.atom_in_ring[slot] = in_ring_orig
        arom_orig = np.array([a.GetIsAromatic() for a in mol.GetAtoms()], dtype=bool)
        self.atom_is_aromatic = np.zeros(len(z), dtype=bool)
        self.atom_is_aromatic[slot] = arom_orig

        # ---- the free-DoF mask over the concatenated [r | theta | phi] vector ---------
        self.rotatable, self.mask = self._find_rotatable(bonds, z, include_trivial_rotations)
        mask_np = self.mask.detach().cpu().numpy()
        self.rotatable_cols = (np.argmax(mask_np, axis=0).astype(np.int64)
                               if mask_np.shape[1] else np.zeros(0, dtype=np.int64))

        n_at = self.spec.n_atoms
        self.n_r, self.n_th, self.n_ph = n_at - 1, n_at - 2, n_at - 3
        n_dof = self.n_r + self.n_th + self.n_ph          # == 3N - 6 == spec.n_dof
        assert n_dof == self.spec.n_dof, (n_dof, self.spec.n_dof)

        block = np.concatenate([np.zeros(self.n_r, dtype=np.int64),
                                np.ones(self.n_th, dtype=np.int64),
                                np.full(self.n_ph, 2, dtype=np.int64)])

        # The state -> DoF map is a LINEAR MAP, not an index subset, because `torsion` is
        # a set of COLLECTIVE coordinates: rotating about one bond shifts every dihedral
        # whose central bond is that one, generally several (describe() prints the count).
        # _find_rotatable's mask column is therefore not one-hot, and treating it as an
        # index was wrong -- it drove only the first of each bond's dihedrals and left the
        # rest at their reference. The bitwise gate caught it; the fix is to carry the map
        # itself. The other levels are the degenerate case where each column is a scaled
        # selection, and they go through the identical formula.
        if level == "torsion":
            if len(self.rotatable) == 0:
                raise ValueError(f"{smiles} has no rotatable bonds; pick a flexible molecule")
            m_full = np.zeros((n_dof, mask_np.shape[1]))
            m_full[self.n_r + self.n_th:, :] = mask_np
            col_block = np.full(mask_np.shape[1], 2, dtype=np.int64)
        else:
            sel = {"dihedral": np.arange(self.n_r + self.n_th, n_dof),
                   "flex": np.arange(self.n_r, n_dof),
                   "full": np.arange(n_dof)}[level]
            m_full = np.zeros((n_dof, len(sel)))
            m_full[sel, np.arange(len(sel))] = 1.0
            col_block = block[sel]

        # A DoF sitting on a parameterisation singularity is HELD, not driven: log sin
        # theta diverges as theta -> pi and the dependent dihedral frame is undefined
        # there. Physically these are stiff and constant anyway (alkynes, nitriles,
        # azides). Zeroing the ROW (not dropping the column) is what makes this uniform
        # across levels; a column left driving nothing is then dropped below.
        singular = np.zeros(n_dof, dtype=bool)
        singular[self.n_r + np.flatnonzero(self.angle_is_linear)] = True
        singular[self.n_r + self.n_th + np.flatnonzero(self.torsion_frame_is_linear)] = True
        m_full[singular, :] = 0.0

        keep = m_full.any(axis=0)
        m_full, col_block = m_full[:, keep], col_block[keep]
        self.data_ndim = int(keep.sum())
        if self.data_ndim == 0:
            raise ValueError(f"{smiles} at level {level!r} has no free degrees of freedom")

        self._free_block = col_block                       # per STATE COLUMN: 0=r 1=th 2=phi
        self.free_mask = m_full.any(axis=1)                # per DoF ROW: is it driven

        # A COLLECTIVE column drives more than one DoF row (a `torsion` column rotates a
        # whole bond). The state -> DoF map is then not invertible row-wise, so
        # state_from_dof -- and therefore the InternalPrior draw -- is unavailable and says
        # so. Every other level is a selection, where each column owns exactly one row.
        col_nnz = (m_full != 0).sum(axis=0)
        self.collective = bool((col_nnz > 1).any())
        self._sel_rows = (None if self.collective else
                          torch.as_tensor(np.argmax(m_full, axis=0), dtype=torch.long,
                                          device=self.device))
        self._driven_idx = torch.as_tensor(np.flatnonzero(self.free_mask),
                                           dtype=torch.long, device=self.device)
        self._M = torch.as_tensor(m_full[self.free_mask], dtype=dtype, device=self.device)

        scale = np.where(col_block == 0, float(delta_r_max),
                         np.where(col_block == 1, float(delta_theta_max), np.pi))
        self._ref_dof = torch.cat([r0, th0, ph0]).to(dtype)
        self._free_scale = torch.as_tensor(scale, dtype=dtype, device=self.device)
        # indexes the STATE, not the DoF vector: the box wall applies to the non-periodic
        # blocks only. Empty at `torsion` and `dihedral`, which is what keeps those levels
        # bitwise identical to the pre-ladder code.
        self._lin_free_idx = torch.as_tensor(np.flatnonzero(col_block != 2),
                                             dtype=torch.long, device=self.device)

        self.delta_r_max, self.delta_theta_max = float(delta_r_max), float(delta_theta_max)
        self.bounding_coeff = float(bounding_coeff)
        self.r_floor, self.theta_floor = float(r_floor), float(theta_floor)
        self.ring_jitter_scale = float(ring_jitter_scale)
        self.ring_min_bank_rows = int(ring_min_bank_rows)
        if force_field not in ('reference', 'mmff'):
            raise ValueError(f"force_field must be 'reference' or 'mmff', got {force_field!r}")
        self.force_field = force_field
        self.ring_mode_fill = float(ring_mode_fill)
        self.ring_pop_temper = float(ring_pop_temper)

        # The wall's box must sit strictly inside the physical domain, or the clamp binds
        # permanently and the sampler is exploring a geometry the reward cannot see.
        free_r = self.free_mask[:self.n_r]
        free_th = self.free_mask[self.n_r:self.n_r + self.n_th]
        if free_r.any():
            worst = float(r0[free_r].min().item()) - self.delta_r_max
            if worst <= self.r_floor:
                raise ValueError(
                    f"delta_r_max={self.delta_r_max} puts the box floor at {worst:.3f} A, "
                    f"at or below r_floor={self.r_floor}; the clamp would bind inside the wall")
        if free_th.any():
            lo = float(th0[free_th].min().item()) - self.delta_theta_max
            hi = float(th0[free_th].max().item()) + self.delta_theta_max
            if lo <= self.theta_floor or hi >= np.pi - self.theta_floor:
                raise ValueError(
                    f"delta_theta_max={self.delta_theta_max} puts the theta box at "
                    f"[{lo:.3f}, {hi:.3f}] rad, outside (0, pi) with margin "
                    f"{self.theta_floor}; the clamp would bind inside the wall")

        self._ff_cache, self._tree_cache = {}, {}
        self._ff_kwargs = dict(epsilon=epsilon, min_separation=min_separation,
                               scale_14=scale_14, lj_k_factor=lj_k_factor)

        # log J is constant in x exactly when no r/theta column is free -- `torsion` and
        # `dihedral`. Recorded, NOT used to short-circuit jacobian_energy: one code path
        # means the constant cannot silently drift from the computed value. Its uses are
        # (a) adding the measure back to a baked potential, and (b) reporting, since it is
        # the offset by which step 2 moved every stored log Z on those levels.
        from mxtaltools.conformers.builder import log_jacobian as _log_jac
        _probe = torch.zeros(1, self.data_ndim, dtype=dtype, device=self.device)
        _tree, _ = self._batch(1)
        _pr, _pth, _ = self.dof_from_state(_probe)
        self.log_jacobian_const = (
            float(_log_jac(_tree, _pr.reshape(-1), _pth.reshape(-1)).item())
            if self._lin_free_idx.numel() == 0 else None)

        # THE CHART VOLUME ELEMENT, log|dq/dx|. The sampler proposes x on [-1, 1]^d, but
        # the Boltzmann density lives on the internal coordinates q -- and dof_from_state
        # writes q_free = ref + scale * x, so the two measures differ by prod_j scale_j.
        #
        # CONSTANT IN x, WHICH IS WHY IT WAS INVISIBLE. A constant shifts log Z and cancels
        # out of every TB residual, so no unconditional result depends on it and no gate
        # would have fired. It stops being harmless the moment log Z is compared ACROSS
        # molecules: the free-column counts differ, so the constant does too -- measured at
        # 'full' it spans 9.0 nats over eight molecules (propanol -9.87 to ethylcyclohexane
        # -18.90), and 3.4 nats at 'torsion'. Without it log Z(c) is not a physical
        # quantity and cross-condition comparison is meaningless.
        self.log_chart_jacobian = float(
            torch.log(self._free_scale).sum().item())

        self.e_ref = self.energy(torch.zeros(1, self.data_ndim, dtype=dtype,
                                             device=self.device),
                                 None, torch.tensor(log_temperature)).item()

    # ------------------------------------------------------------------ topology

    def _find_rotatable(self, bonds, z, include_trivial: bool):
        """Tree bonds whose rotation moves a genuine fragment.

        A rotatable bond must be a *bridge* -- a ring bond cannot be rotated
        independently, since the ring constrains it. The moving side must also contain a
        heavy atom, or the "rotation" is a terminal hydrogen wobble.
        """
        spec = self.spec
        n = spec.n_atoms
        g = nx.Graph([(int(i), int(j)) for i, j in zip(*bonds)])
        g.add_nodes_from(range(n))
        # bridges in original atom numbering -> placement-slot numbering
        slot = np.empty(n, dtype=np.int64)
        slot[spec.perm] = np.arange(n)
        bridges = {tuple(sorted((int(slot[a]), int(slot[b])))) for a, b in nx.bridges(g)}

        parent = np.full(n, -1, dtype=np.int64)
        parent[spec.bond_index[:, 1]] = spec.bond_index[:, 0]
        z_slot = np.asarray(spec.z)

        # descendants of each slot, placement order is topological
        desc = [[k] for k in range(n)]
        for k in range(n - 1, 0, -1):
            desc[parent[k]].extend(desc[k])

        ti = spec.torsion_index
        rotatable, columns = [], []
        for v in range(1, n):
            u = parent[v]
            if u < 0 or tuple(sorted((int(u), int(v)))) not in bridges:
                continue
            col = (ti[:, 1] == u) & (ti[:, 2] == v)
            if not col.any():
                continue
            moving = [a for a in desc[v] if a != v]
            if not moving:
                continue
            heavy = [a for a in moving if z_slot[a] > 1]
            if not heavy:
                continue  # terminal hydrogens only
            if not include_trivial and len(heavy) == 1 and len(moving) <= 3:
                continue  # methyl / amine / hydroxyl spin
            rotatable.append((int(u), int(v)))
            columns.append(col)

        mask = (torch.tensor(np.stack(columns, axis=1), dtype=self.dtype,
                             device=self.device)
                if columns else torch.zeros((len(ti), 0), dtype=self.dtype,
                                            device=self.device))
        return rotatable, mask

    def describe(self) -> str:
        from rdkit import Chem
        sym = Chem.GetPeriodicTable()
        z = np.asarray(self.spec.z)
        name = lambda i: f"{sym.GetElementSymbol(int(z[i]))}{i}"
        n_free = [int((self._free_block == b).sum()) for b in (0, 1, 2)]
        lines = [f"{self.smiles}: {self.spec.n_atoms} atoms, {self.spec.n_dof} internal DoF",
                 f"   level {self.level!r}: {self.data_ndim} free "
                 f"(r {n_free[0]}/{self.n_r}, theta {n_free[1]}/{self.n_th}, "
                 f"phi {n_free[2]}/{self.n_ph})",
                 f"   linearity flags MEASURED: {int(self.angle_is_linear.sum())} linear "
                 f"angle(s), {int(self.torsion_frame_is_linear.sum())} ill-conditioned "
                 f"frame(s), all held"]
        if n_free[0] or n_free[1]:
            lines.append(f"   box: r +/-{self.delta_r_max} A, theta "
                         f"+/-{self.delta_theta_max} rad, wall {self.bounding_coeff}, "
                         f"clamp r>={self.r_floor} theta in "
                         f"({self.theta_floor}, pi-{self.theta_floor})")
        for j, (u, v) in enumerate(self.rotatable):
            moved = int(self.mask[:, j].sum())
            lines.append(f"   rotatable {j}: about {name(u)}-{name(v)} "
                         f"({moved} dihedral(s) shifted)")
        return "\n".join(lines)

    # -------------------------------------------------------------------- energy

    def _make_ff(self, tree, ref_pos, epsilon, min_separation, scale_14, lj_k_factor):
        """The force field, by the `force_field` switch.

        'reference' -- ff_from_reference: r0/theta0 MEASURED off the embedded conformer, so
        the bonded terms are exactly zero there. It carries NO TORSION TERM AT ALL, which
        makes every rotamer distribution nearly uniform (propanol's target entropy is 7.601
        of a maximum 7.625) and leaves amide omega degenerate between cis and trans. Its
        parameters also depend on the embedding seed, which is fatal conditionally.

        'mmff' -- ff_from_mmff: RDKit's MMFF94 typing, graph-determined, with a real
        3-term torsion. Full organic coverage, so peptides and aromatics work; ff_from_graph
        raises on both. Note the reference conformer STOPS being the energy minimum, since
        r0/theta0 are now typed rather than measured.
        """
        from mxtaltools.conformers.energy import ff_from_reference, ff_from_mmff
        if self._ff_choice == 'mmff':
            return ff_from_mmff(tree, self.mol, self.spec.perm, dtype=self.dtype,
                                min_separation=min_separation, lj_k_factor=lj_k_factor)
        return ff_from_reference(tree, ref_pos, epsilon=epsilon,
                                 min_separation=min_separation,
                                 scale_14=scale_14, lj_k_factor=lj_k_factor)

    def _batch(self, batch_size: int):
        """Cached collated tree and force field for a given batch size."""
        if batch_size not in self._tree_cache:
            from mxtaltools.conformers.builder import collate

            tree = collate([self.spec] * batch_size, device=self.device)
            ref = self.ref_pos.repeat(batch_size, 1)
            self._tree_cache[batch_size] = tree
            self._ff_cache[batch_size] = self._make_ff(tree, ref, **self._ff_kwargs)
        return self._tree_cache[batch_size], self._ff_cache[batch_size]

    def dof_from_state(self, x: torch.Tensor):
        """State ``[B, d]`` on [-1, 1] -> ``(r, theta, phi)``, each ``[B, n_block]``.

        One scatter, whatever the level: the free columns are written as
        ``reference + scale * x`` into a copy of the reference DoF vector and everything
        else holds. At `torsion` this reproduces the old ``phi_ref + mask @ (pi * x)``
        bitwise -- the old matmul ran over a 0/1 selection column, so every dropped term
        was an exact zero and ``v + 0.0 == v``.

        r and theta are CLAMPED to the physical domain here, and that is the domain
        guarantee -- not the box wall. The wall is a preference; the clamp is what makes
        the reward finite for every latent in R^d. Without it: log_jacobian's ``2 log r``
        is NaN at r <= 0 and ``log sin theta`` is NaN or -inf outside (0, pi), and `build`
        is non-injective off-domain (``d(r, 2pi-theta, phi+pi) == d(r, theta, phi)``) so
        an excursion double-covers rather than merely being re-weighted -- which `measure`
        cannot even detect, since bond_angle returns [0, pi] always. MolecularCrystal does
        the same thing for the same reason (crystal_ops.py:301 clamps before the map).
        phi is not clamped: it wraps, which is a total map.
        """
        b = x.shape[0]
        dof = self._ref_dof.unsqueeze(0).repeat(b, 1)
        # (x * scale) @ M.T reproduces the old `(pi * x) @ mask.T` bitwise: M is 0/1, so
        # scaling before the matmul is the identical rounding and every dropped term is an
        # exact zero. index_add rather than a full-width add so rows no column drives are
        # not touched at all.
        dof = dof.index_add(1, self._driven_idx,
                            (x.to(self.dtype) * self._free_scale) @ self._M.T)

        r = dof[:, :self.n_r]
        th = dof[:, self.n_r:self.n_r + self.n_th]
        ph = dof[:, self.n_r + self.n_th:]
        if self._lin_free_idx.numel():
            # only pay for the clamp where a linear block is actually free; at `torsion`
            # and `dihedral` r and th are the frozen reference and cannot leave the domain
            r = r.clamp_min(self.r_floor)
            th = th.clamp(self.theta_floor, np.pi - self.theta_floor)
        return r, th, ph

    def build_positions(self, x: torch.Tensor) -> torch.Tensor:
        """State ``[B, d]`` in **[-1, 1]** -> Cartesian positions ``[B * N, 3]``.

        The per-block affine map (delta from a reference, fixed per-block scale) has a
        constant Jacobian, so it shifts log Z by a constant -- but the constant has counts
        affine in N, so it is a PER-MOLECULE offset and must be carried wherever log Z(c)
        is compared across molecules.
        """
        from mxtaltools.conformers.builder import build

        r, th, ph = self.dof_from_state(x)
        tree, _ = self._batch(x.shape[0])
        return build(tree, r.reshape(-1), th.reshape(-1), ph.reshape(-1))

    def bounding_energy(self, x: torch.Tensor, temperature) -> torch.Tensor:
        """Box wall on the NON-PERIODIC state blocks, pre-multiplied by temperature.

        ``energy()`` divides the total by T, so pre-multiplying keeps this
        domain-validity constraint equally stiff across sampling temperatures -- the same
        compensation MolecularCrystal.generator_energy applies to its own bounding term
        (molecular_crystal.py:446), and the same one the Jacobian needs
        (compute_jacobian, molecular_crystal.py:552).

        Zero-width when no linear block is free, and the caller skips the add entirely in
        that case, which is what keeps `torsion` and `dihedral` bitwise unchanged.
        """
        xl = x.to(self.dtype).index_select(-1, self._lin_free_idx)
        v = torch.relu(xl - 1.0) ** 2 + torch.relu(-(xl + 1.0)) ** 2
        return self.bounding_coeff * v.sum(-1) * temperature

    # -------------------------------------------------------------- prior draw

    def state_from_dof(self, r, th, ph) -> torch.Tensor:
        """Inverse of dof_from_state: ``(r, theta, phi)`` -> state ``[B, d]`` on [-1, 1].

        SELECTION levels only. At a collective level (`torsion`) a column drives several
        DoF rows and the map is not invertible row-wise; build_prior_states.draw_states is
        the torsion-specific path and it works by looking up one representative dihedral
        per bond rather than by inverting anything.
        """
        if self.collective:
            raise NotImplementedError(
                f"state_from_dof needs a selection map, but level {self.level!r} has "
                f"collective columns (one bond rotation drives several dihedrals). Use "
                f"build_prior_states.draw_states for the torsion route.")
        dof = torch.cat([r, th, ph], dim=-1).to(self.dtype)
        sel = self._sel_rows
        x = ((dof.index_select(1, sel) - self._ref_dof.index_select(0, sel).unsqueeze(0))
             / self._free_scale)
        # phi columns are deltas on a circle: wrap, so a draw near the seam comes back as
        # a small latent instead of a large one the wall would then fight
        is_phi = torch.as_tensor(self._free_block == 2, dtype=torch.bool, device=x.device)
        return torch.where(is_phi, (x + 1.0) % 2.0 - 1.0, x)

    def prior_dof_types(self, prior):
        """``(kind, histogram_or_None, key, is_ring)`` per DoF ROW, in SPEC numbering.

        Keys are built with InternalPrior's OWN key functions, so they match the fitted
        tables exactly -- but indexed through this class's `spec`, never through
        mxtaltools' ``tree_*`` fields. Those are a different encoding of the same tree
        (see energies/conformer_data.py's module docstring) with a different index
        convention, and mixing the two numberings would scramble the columns silently.
        This is the same reason build_prior_states.torsion_histograms does its own lookup
        rather than calling ``prior.sample(mol, ...)``.
        """
        keys = self.atom_keys
        bi = np.asarray(self.spec.bond_index)
        ai = np.asarray(self.spec.angle_index)
        ti = np.asarray(self.spec.torsion_index)
        ring = lambda idx: bool(self.atom_in_ring[list(idx)].all())

        out = []
        for j in range(self.n_r):
            k = prior.bond_key(keys[bi[j, 0]], keys[bi[j, 1]])
            out.append(('r', prior.bonds.get(k), k, ring(bi[j])))
        for j in range(self.n_th):
            # spec.angle_index is (b, c, n) with the APEX in the middle, which is the
            # position angle_key expects
            k = prior.angle_key(keys[ai[j, 0]], keys[ai[j, 1]], keys[ai[j, 2]])
            out.append(('theta', prior.angles.get(k), k, ring(ai[j])))
        for j in range(self.n_ph):
            k = prior.torsion_key(keys[ti[j, 1]], keys[ti[j, 2]])   # central bond only
            out.append(('phi', prior.torsions.get(k), k, ring(ti[j])))
        return out

    def torsion_groups(self):
        """phi DoF rows grouped by CENTRAL BOND, improper rows excluded, leader first.

        A group is the set of atoms placed onto one parent -- every dihedral whose
        DIFFERENCES from the others fix a bond angle at that parent. An H-C-H angle is a
        difference of two of them, and it is one of the graph angles the force field
        scores but the tree does not expose as a coordinate. So they have to be drawn
        JOINTLY. Drawn independently, even from perfect marginals, a substantial fraction
        of sibling pairs land on the same rotamer mode and put two substituents in the
        same place.

        OF THE TWO FACTS IN THAT SUMMARY LINE, ONLY THE SECOND IS LOAD-BEARING.

        The mechanism is that every member takes the leader's angular displacement, which
        is a rigid rotation of the set only when the members share a reference axis. On a
        SPANNING TREE that is automatic: every atom has exactly one parent, so for a proper
        row the reference `b` is a function of `c`, and keying on the parent atom therefore
        yields the IDENTICAL partition. The key is a free choice here, and a test written
        to detect the "wrong" key cannot fire. Do not add one.

        WHAT IS NOT A FREE CHOICE is excluding the improper rows (see improper_phi_rows).
        They are the only rows that share a parent with different references -- that is
        what an improper IS -- so with them in the group the two keyings genuinely differ
        and the displacement lands about mismatched axes, destroying the angle at the
        shared parent. An earlier version of this docstring credited the KEY for that
        damage on the strength of a pinned measurement; the improper rows were the whole
        cause, and the pinned number outlived the claim it supported. Quantities belong in
        the harness output or in findings, not here -- see prior_smoke's
        improper_rows_ungrouped and group_rigid_angle, which re-measure this on every run.
        """
        from collections import defaultdict
        ti = np.asarray(self.spec.torsion_index)
        imp = set(self.improper_phi_rows())
        g = defaultdict(list)
        for j in range(self.n_ph):
            if j in imp:
                continue
            g[(int(ti[j, 1]), int(ti[j, 2]))].append(j)
        return [sorted(rows) for rows in g.values()]

    def improper_phi_rows(self):
        """phi rows that are LOCAL GEOMETRY, not rotatable torsions.

        A tree dihedral for atom `n` placed on parent `c` with references `b`, `a` is a
        genuine torsion about the c-b bond only when `a` lies one bond FURTHER OUT, i.e.
        bonded to `b`. When `a` is instead bonded to `c`, the dihedral is measured between
        two substituents of the same parent and IS the angle between them -- an improper.
        Drawing it from a pooled rotamer histogram destroys that angle outright.

        Ethanol is the clean example: row 1 places O4 on C0 referenced to (C3, H1) with
        H1 a neighbour of C0, so phi(row 1) is precisely the O4-C0-H1 angle. Sampled from
        a histogram it lands at a median 14.5 degrees against theta0 = 108.6, and that one
        angle carried 251 of the 252 kcal/mol of angle strain in the whole molecule.

        Always a small set -- the first one or two rows, where the tree is still building
        its initial frame -- which is exactly why this survived: every genuinely rotatable
        bond in the molecule behaves correctly.
        """
        from collections import defaultdict
        ti = np.asarray(self.spec.torsion_index)
        nb = defaultdict(set)
        for u, v in np.asarray(self.spec.bond_index):
            nb[int(u)].add(int(v))
            nb[int(v)].add(int(u))
        return [j for j in range(self.n_ph) if int(ti[j, 0]) in nb[int(ti[j, 2])]]

    def improper_phi_sigma(self, temperature: float):
        """Thermal width for the improper phi rows, in radians.

        An improper dihedral IS an angle at the parent, so its Boltzmann width is that
        angle's own sqrt(kT/2k). The controlled angle is a redundant graph angle rather
        than a tree angle, so this uses the median tree-angle width as the stand-in --
        near-tetrahedral centres put the dihedral-to-angle Jacobian at order one, and the
        measurement that matters (the angle energy of a draw) is checked directly.
        """
        _, s_th = self.thermal_rtheta_sigma(temperature)
        return float(np.median(s_th)) if len(s_th) else 0.1

    def sibling_jitter_sigma(self, groups, temperature: float):
        """Per-group jitter width for the sibling offsets, in radians.

        ``sigma = sqrt(kT / 2k)`` of the REDUNDANT angle the group's members determine --
        i.e. the thermal width of the very quantity independent draws destroy. Taken from
        the force field's own constant, so it is automatically tighter at a stiff centre
        than a soft one and scales with temperature; nothing here is a tuned number.

        The jitter has to be nonzero. Locking the offsets rigidly gives a prior with
        measure-zero support in those dimensions, and support is the one property TB
        actually needs from a prior -- it is the same reason InternalPrior fattens its
        marginals toward uniform.
        """
        _, ff1 = self._batch(1)
        ai = ff1.angle_index.detach().cpu().numpy()
        ka = ff1.k_angle.detach().cpu().numpy()
        ti = np.asarray(self.spec.torsion_index)
        out = []
        for rows in groups:
            c = int(ti[rows[0], 2])
            placed = {int(ti[j, 3]) for j in rows}
            k = [ka[i] for i in range(len(ai))
                 if int(ai[i, 1]) == c and int(ai[i, 0]) in placed and int(ai[i, 2]) in placed]
            k_med = float(np.median(k)) if k else 50.0
            out.append(float(np.sqrt(max(temperature, 1e-12) / (2.0 * k_med))))
        return out

    def ring_blocks(self, prior):
        """Ring-system DoF blocks in SPEC numbering, paired with their fitted bank.

        Returns ``[(order, bank)]`` where ``order`` is ``[(kind, row), ...]`` in exactly
        the sequence InternalPrior fitted the block in (all r, then all theta, then all
        phi), and ``bank`` is the RingBank or None.

        Ring systems are the second place a product of marginals cannot work: closure is a
        hard constraint, so independently-drawn ring DoF violate it by construction --
        prior.py says so outright. Unlike the sibling case there is no structural fix, so
        this defers to InternalPrior's own joint bank.

        The ASSERTION is the load-bearing part. The bank was fitted in mxtaltools'
        ``tree_*`` encoding and is consumed here in ``spec`` numbering. Those are different
        encodings of the same tree (conformer_data's module docstring) and they agree
        today -- verified on cyclohexane, toluene, naphthalene, proline and ibuprofen --
        but a silent divergence would permute the block rather than fail, and the symptom
        would look like "rings just sample badly".
        """
        from energies.conformer_data import condition_from_energy
        m = condition_from_energy(self, partial_charges=False)
        m.build_conformer_tree()
        for lbl, a, b in (('bond', np.asarray(self.spec.bond_index), m.tree_bond_index),
                          ('angle', np.asarray(self.spec.angle_index), m.tree_angle_index),
                          ('torsion', np.asarray(self.spec.torsion_index), m.tree_torsion_index)):
            b = b.detach().cpu().numpy().T
            if a.shape != b.shape or not (a == b).all():
                raise RuntimeError(
                    f"the spec tree and mxtaltools' tree_* encoding disagree on the {lbl} "
                    f"index for {self.smiles}. RingBank blocks are fitted in the latter and "
                    f"consumed in the former, so they cannot be mapped -- refusing rather "
                    f"than permuting the block silently.")
        # A prior fitted before the ring signature was fixed has keys that simply do not
        # resolve, and every ring then falls through to the hold -- indistinguishable from
        # "this ring was never fitted". Say which it is.
        # vars(), NOT getattr. InternalPrior is a dataclass, so a field with a default is
        # also a CLASS attribute -- getattr on a prior pickled before the field existed
        # returns the current default and reports itself up to date. Only the instance
        # __dict__ distinguishes "fitted under v2" from "predates the field".
        self.ring_sig_stale = vars(prior).get('ring_sig_version', 1) < 2
        sysid, _ = prior.ring_systems(m)
        _, blocks, sigs, _ = prior._layout(m)
        bi, ai, ti = (np.asarray(self.spec.bond_index), np.asarray(self.spec.angle_index),
                      np.asarray(self.spec.torsion_index))
        # WHY THIS IS RECORDED RATHER THAN RE-DERIVED. Four different things end in
        # `bank = None` below -- aromatic by design, no key resolved, a bank too thin, and
        # a stale prior whose keys cannot resolve at all -- and the tuple return cannot
        # tell them apart. A reader that re-derives aromaticity and the lookup to label
        # them is a second copy of this branch, free to drift from it; energies/
        # ring_metrics.py reads this instead. See that module for what each class means.
        self.ring_block_info = []
        out = []
        for s, cols in blocks.items():
            order = ([('r', int(j)) for j in cols['r']]
                     + [('theta', int(j)) for j in cols['theta']]
                     + [('phi', int(j)) for j in cols['phi']])
            in_sys_pre = {int(a) for a in range(len(sysid)) if int(sysid[a]) == int(s)}
            aromatic = bool(self.atom_is_aromatic[list(in_sys_pre)].all())
            # a fitted MODE SUBSPACE wins over the discrete bank: the rows were isolated
            # islands with near-zero mass between basins, and the saddles live between them
            bank = getattr(prior, 'ring_modes', {}).get((sigs[s], len(order)))
            if bank is None:
                bank = prior.rings.get((sigs[s], len(order)))
            if aromatic:
                # An aromatic ring is RIGID: there is no pucker to sample, so a bank buys
                # nothing and can only do harm. It did: under signature version 1 the key
                # carried element but not degree, so benzene and cyclohexane shared a bank
                # and benzene drew chairs -- median |ring torsion| 47 deg, 75% of draws
                # past 20 deg, with ff_from_reference unable to object (no torsion term,
                # and bond angles stay near 120 deg through a pucker). Holding it planar
                # near the reference is correct by construction and needs no fit.
                bank = None
            elif isinstance(bank, RingModes):
                pass                                    # subspaces carry their own width
            elif bank is not None and (bank.rows.shape[1] != len(order)
                                       or bank.rows.shape[0] < self.ring_min_bank_rows):
                # a 1-row bank is a single observation on replay -- measured on
                # naphthalene under signature v1, where it was 30x WORSE than holding the
                # ring. The default is 2, not higher: with a v2 signature and a
                # purpose-built bank (build_ring_banks.py) a small bank is COMPLETE rather
                # than thin -- pyrrolidine genuinely has two envelope basins. The real
                # protection against a contaminated bank is the signature, not this count.
                bank = None
            # DoF that PLACE a ring atom, which is a superset of the block: _layout's
            # owner() requires every atom of a DoF to be in the ring, so a ring-positioning
            # dihedral whose reference atom sits outside is excluded -- and then moves
            # freely and breaks closure. Proline's closure error was 1.5 A at ANY jitter
            # scale for exactly this reason. These extras are held, never banked, since
            # the bank was fitted against the narrower set.
            in_sys = {int(a) for a in range(len(sysid)) if int(sysid[a]) == int(s)}
            placing = ([('r', j) for j in range(self.n_r) if int(bi[j, 1]) in in_sys]
                       + [('theta', j) for j in range(self.n_th) if int(ai[j, 2]) in in_sys]
                       + [('phi', j) for j in range(self.n_ph) if int(ti[j, 3]) in in_sys])
            extra = [kj for kj in placing if kj not in set(order)]
            self.ring_block_info.append({
                'system': int(s), 'aromatic': aromatic, 'key': (sigs[s], len(order)),
                'ring_class': ('held_aromatic' if aromatic else
                               'banked_modes' if isinstance(bank, RingModes) else
                               'banked_rows' if bank is not None else 'held_unsupported'),
                'n_block_dof': len(order), 'n_extra_dof': len(extra),
                'stale_prior': bool(self.ring_sig_stale),
            })
            out.append((order, bank, extra))
        return out

    def ring_frame_groups(self, ring_rows):
        """Torsion groups about a RING BOND that hold no ring-placed member of their own.

        THE GAP THE MIXED-GROUP RULE LEAVES. Groups are keyed on the central bond, and a
        group containing a ring-placed row lets that row lead: the substituents take its
        displacement, which is right because they share its axis. But the LAST ring atom in
        placement order has both of its ring neighbours already placed, so the group about
        its incoming ring bond contains only substituents -- no ring member, the rule does
        not fire, and the group falls through to "draw the leader from a rotamer histogram".
        That histogram is keyed on the central bond type, and the central bond here is a
        RING bond, which does not rotate. Measured on cyclohexane, the leader landed 81 deg
        from the reference while every correctly-mixed group sat within 5 deg, and the two
        redundant angles at that carbon carried most of the molecule's angle strain.

        The fix is not a new fitted object and not a correction after building: the ring
        block has already placed every ring atom, so the frame's own dihedral to the OTHER
        ring neighbour is determined, and the substituents hang off it at the fixed offset
        they have in the reference conformer. Same move as the mixed-group rule, with the
        leader taken from the ring geometry instead of from a member of the group.

        Returns ``[(rows, a, b, c, p, gi)]`` -- the group's phi rows in SPEC numbering, its
        torsion frame ``(a, b, c)``, the ring neighbour ``p`` of ``c`` that is not ``b``,
        and the group index into ``torsion_groups()``.
        """
        ti = np.asarray(self.spec.torsion_index)
        inr = self.atom_in_ring
        nbr = {}
        for u, v in np.asarray(self.spec.graph_bond_index):
            nbr.setdefault(int(u), set()).add(int(v))
            nbr.setdefault(int(v), set()).add(int(u))
        out = []
        for gi, rows in enumerate(self.torsion_groups()):
            if any(self.n_r + self.n_th + j in ring_rows for j in rows):
                continue
            a, b, c = (int(ti[rows[0], 0]), int(ti[rows[0], 1]), int(ti[rows[0], 2]))
            if not (inr[b] and inr[c]):
                continue                       # not a ring bond: the default rule is right
            p = [q for q in nbr.get(c, ()) if inr[q] and q != b]
            if not p:
                continue
            out.append((list(rows), a, b, c, int(p[0]), gi))
        return out

    def prior_log_prob(self, prior, dof: np.ndarray, joint_torsions: bool = True,
                       thermal_rtheta: bool = True) -> np.ndarray:
        """``log q(dof)`` for exactly what ``sample_prior_states`` draws. ``[n]``.

        WHY THIS HAS TO EXIST SEPARATELY. InternalPrior ships a matched sample/log_prob
        pair, but ``sample_prior_states`` is NOT InternalPrior's sampler -- it adds joint
        sibling draws, thermal r/theta and ring subspaces on top, in SPEC numbering rather
        than mxtaltools' ``tree_*`` numbering. Without a density that mirrors those extras
        there is no importance weight, and so no ESS: the number people quote for the
        upgraded prior would silently be the density of a DIFFERENT distribution.

        ACYCLIC ONLY. Ring blocks draw from a bank or a pucker subspace whose density is a
        mixture over fitted rows, and the subspace is lower-dimensional than the block it
        fills, so the block's density is singular in the held directions. That is a real
        derivation, not an oversight -- it raises rather than returning a number that
        would look usable.

        The BOX CLAMP in sample_prior_states is not represented here either: it puts
        finite mass exactly on the wall, which no continuous density can express. Callers
        must check ``stats['clip_frac']`` is ~0 before treating these weights as valid.
        """
        from mxtaltools.conformers.prior import R_RANGE, THETA_RANGE, PHI_RANGE
        spans = {'r': R_RANGE, 'theta': THETA_RANGE, 'phi': PHI_RANGE}

        if any(True for _ in self.ring_blocks(prior)):
            raise NotImplementedError(
                'prior_log_prob covers acyclic molecules only: a ring block is drawn from '
                'a bank or pucker subspace whose density is a mixture, and is singular in '
                'the directions the subspace does not span. Restrict the ESS measurement '
                'to acyclic molecules, or derive the ring block density first.')

        dof = np.atleast_2d(np.asarray(dof, dtype=np.float64))
        n = dof.shape[0]
        total = np.zeros(n)
        types = self.prior_dof_types(prior)
        n_phi0 = self.n_r + self.n_th

        ph0 = self.ph0.detach().cpu().numpy()
        r0 = self.r0.detach().cpu().numpy()
        th0 = self.th0.detach().cpu().numpy()
        s_r, s_th = self.thermal_rtheta_sigma(float(self.temperature))
        groups = self.torsion_groups()
        g_sigma = self.sibling_jitter_sigma(groups, float(self.temperature))

        def gauss(x, mu, s):
            return -0.5 * ((x - mu) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)

        def wrapped_gauss(x, mu, s):
            """phi lives on a circle; for sigma near pi the images matter."""
            acc = np.zeros_like(x)
            for k in (-1, 0, 1):
                acc = acc + np.exp(gauss(x + 2 * np.pi * k, mu, s))
            return np.log(np.clip(acc, 1e-300, None))

        def marginal(row, kind, hist):
            if hist is None:
                lo, hi = spans[kind]
                return np.full(n, -np.log(hi - lo))
            return np.asarray(hist.log_prob(dof[:, row], prior.fatten))

        # ---- r / theta ----
        if thermal_rtheta:
            for j in range(self.n_r):
                total += gauss(dof[:, j], r0[j], s_r[j])
            for j in range(self.n_th):
                total += gauss(dof[:, self.n_r + j], th0[j], s_th[j])

        # ---- rows the sampler leaves on their own marginal ----
        for row, (kind, hist, key, is_ring) in enumerate(types):
            if (joint_torsions and row >= n_phi0) or (thermal_rtheta and row < n_phi0):
                continue
            total += marginal(row, kind, hist)

        # ---- phi, mirroring the leader/follower structure exactly ----
        if joint_torsions:
            s_imp = self.improper_phi_sigma(float(self.temperature))
            for j in self.improper_phi_rows():
                total += wrapped_gauss(dof[:, n_phi0 + j], ph0[j], s_imp)
            for gi, rows_j in enumerate(groups):
                grows = [n_phi0 + j for j in rows_j]
                lead_i = 0
                _, hist, _, _ = types[grows[lead_i]]
                total += marginal(grows[lead_i], 'phi', hist)
                disp = ((dof[:, grows[lead_i]] - ph0[rows_j[lead_i]] + np.pi)
                        % (2 * np.pi) - np.pi)
                for i, gr in enumerate(grows):
                    if i == lead_i:
                        continue
                    total += wrapped_gauss(dof[:, gr], ph0[rows_j[i]] + disp, g_sigma[gi])
        return total

    def _global_row(self, kind: str, j: int) -> int:
        return {'r': j, 'theta': self.n_r + j, 'phi': self.n_r + self.n_th + j}[kind]

    def thermal_rtheta_sigma(self, temperature: float):
        """``sqrt(kT / 2k)`` per tree bond and tree angle, from the FF's own constants.

        For a HARMONIC term the exact local Boltzmann marginal is a Gaussian of this
        width about the term's minimum -- which the force field states outright, so there
        is nothing to fit. InternalPrior's r/theta histograms are pooled across chemical
        environments and are therefore much broader than any individual bond's thermal
        spread: on Ala5 they cost ~180 kcal/mol of bond strain against a thermal ~26.

        This does NOT replace the prior for phi. There the force field (ff_from_reference)
        has no torsion term at all, so the rotamer distribution comes entirely from
        sterics and the empirical histogram is the only thing that knows about it.
        """
        _, ff1 = self._batch(1)
        bi_ff = ff1.bond_index.detach().cpu().numpy()
        ai_ff = ff1.angle_index.detach().cpu().numpy()
        kb, ka = ff1.k_bond.detach().cpu().numpy(), ff1.k_angle.detach().cpu().numpy()
        bmap = {frozenset((int(a), int(b))): kb[i] for i, (a, b) in enumerate(bi_ff)}
        amap = {(int(j), frozenset((int(i), int(k)))): ka[m]
                for m, (i, j, k) in enumerate(ai_ff)}
        bi, ai = np.asarray(self.spec.bond_index), np.asarray(self.spec.angle_index)
        kt = max(float(temperature), 1e-12)
        s_r = np.array([np.sqrt(kt / (2.0 * bmap.get(frozenset((int(r[0]), int(r[1]))), 300.0)))
                        for r in bi])
        s_th = np.array([np.sqrt(kt / (2.0 * amap.get((int(r[1]), frozenset((int(r[0]), int(r[2])))), 50.0)))
                         for r in ai])
        return s_r, s_th

    def sample_prior_states(self, prior, n: int, rng, report: bool = True,
                            joint_torsions: bool = True, thermal_rtheta: bool = True,
                            joint_rings: bool = True):
        """``[n, d]`` states drawn from a fitted InternalPrior. Returns ``(x, stats)``.

        Per-DoF marginals for the acyclic part, joint draws where a product of marginals
        cannot work: sibling torsions about a shared bond, and RING SYSTEMS.

        ``joint_rings`` DEFAULTS TO TRUE and is the real ring path -- each ring block is
        drawn from its fitted pucker subspace or discrete bank, aromatic rings are held
        planar by design, an unsupported ring is held at a fraction of thermal width, and
        the ring-positioning DoF outside the block are held either way. See ``ring_blocks``
        for the four classes and energies/ring_metrics.py for how they are reported.

        ``joint_rings=False`` IS A NEGATIVE CONTROL, NOT A SAMPLING MODE. Every ring DoF
        then gets an independent marginal, which violates closure by construction: measured
        on cyclohexane at 'full', closure error goes from 0.086 A (2.2 bond-sigma) to
        2.93 A (75 bond-sigma) and the median potential rises by two orders of magnitude.
        The draws remain valid support, which is all TB strictly needs, but as a proposal
        they are broken -- so a benchmark quoting this path is measuring the disabled path,
        not the prior. ``stats['closure_err']`` is measured on BOTH, deliberately.
        """
        from mxtaltools.conformers.prior import R_RANGE, THETA_RANGE, PHI_RANGE
        spans = {'r': R_RANGE, 'theta': THETA_RANGE, 'phi': PHI_RANGE}

        types = self.prior_dof_types(prior)
        dof = np.empty((n, len(types)))
        stats = {'n_uniform': {'r': 0, 'theta': 0, 'phi': 0},
                 'n_ring_marginal': 0, 'n_dof': len(types), 'joint_torsions': joint_torsions}
        n_phi0 = self.n_r + self.n_th

        def draw(row, kind, hist):
            if hist is None:
                lo, hi = spans[kind]
                stats['n_uniform'][kind] += 1
                return rng.uniform(lo, hi, n)
            return hist.sample(n, prior.fatten, rng)

        ph0 = self.ph0.detach().cpu().numpy()
        r0 = self.r0.detach().cpu().numpy()
        th0 = self.th0.detach().cpu().numpy()
        s_r, s_th = self.thermal_rtheta_sigma(float(self.temperature))
        groups = self.torsion_groups()
        g_sigma = self.sibling_jitter_sigma(groups, float(self.temperature))
        phi_sig = float(np.median(g_sigma)) if g_sigma else 0.1

        # ---- ring systems FIRST: closure is a hard constraint, so their DoF are joint,
        # and substituents hanging off a ring atom then lock to what the ring chose ----
        ring_rows = set()
        stats.update(n_rings=0, n_ring_banked=0, n_ring_thermal=0, n_ring_extra_held=0,
                     n_ring_remapped=0)
        if joint_rings:
            ref = {'r': r0, 'theta': th0, 'phi': ph0}
            sig = {'r': s_r, 'theta': s_th}
            sc = self.ring_jitter_scale

            def hold(kj):
                """Rattle one ring DoF about its reference at a FRACTION of thermal width.

                Full thermal width does not work: closure is nonlinear, so independent
                per-DoF perturbations accumulate around the loop with a lever arm. Measured
                on cyclohexane and naphthalene, closure error is linear in this scale, and
                0.1 puts it at 0.025-0.043 A -- at or under a bond's own thermal width of
                0.041 A. Larger and the ring is visibly open; this is the price of having
                no bank, and it means pucker is rattled rather than sampled.
                """
                kind, j = kj
                gr = self._global_row(kind, j)
                s = (sig[kind][j] if kind in sig else phi_sig) * sc
                dof[:, gr] = ref[kind][j] + rng.normal(0.0, s, n)
                return gr

            for order, bank, extra in self.ring_blocks(prior):
                rows = [self._global_row(k, j) for k, j in order]
                stats['n_rings'] += 1
                if isinstance(bank, RingModes):
                    # subspace draw: theta/phi from the pucker manifold, r from the
                    # thermal path, everything else in the block held
                    stats['ring_fill'] = self.ring_mode_fill
                    # THE BANK'S ROW INDICES BELONG TO THE MOLECULE IT WAS FITTED ON.
                    # ``bank.order`` is [(kind, row)] in the SPEC numbering of the bare
                    # ring that build_ring_banks scanned. The lookup key is
                    # (signature, n_dof), which identifies the ring TYPE and says nothing
                    # about row numbering -- and the tree numbers a ring's DoF differently
                    # depending on what else is attached. Writing the bank's columns into
                    # its own stored rows therefore PERMUTES the block whenever the two
                    # molecules disagree, which is exactly the silent-permutation failure
                    # ring_blocks' tree assertion is written to prevent within a molecule.
                    #
                    # Measured on phenyl-tetrahydropyran: its block is theta 1,5,8,13 /
                    # phi 4,7,12 while the bank carries theta 2,5,8,11 / phi 4,7,10, so two
                    # bank columns landed on DoF placing atoms outside the ring. The ring
                    # read chair-like on 48% of draws instead of 99.6% -- half the draws
                    # were twist-boats with the phenyl clashing, worth ~40,000 kT of LJ, and
                    # it looked like "rings just sample badly" rather than a mapping error.
                    #
                    # The correspondence is POSITIONAL: both sequences are _layout's block
                    # order with the r rows removed, so column i means the i-th non-r DoF of
                    # THIS block. The kinds must line up or the two blocks are not the same
                    # object and there is nothing to map -- refuse rather than permute.
                    own = [kj for kj in order if kj[0] != 'r']
                    if [k for k, _ in own] != [k for k, _ in bank.order]:
                        raise RuntimeError(
                            f"ring bank for {self.smiles} has kind sequence "
                            f"{[k for k, _ in bank.order]} but this molecule's block is "
                            f"{[k for k, _ in own]}; the key matched but the blocks are "
                            f"not the same object -- refusing rather than permuting it")
                    stats['n_ring_remapped'] += int(list(own) != list(bank.order))
                    dev = np.asarray(bank.sample(n, rng, fill=self.ring_mode_fill,
                                                 temperature=float(self.temperature),
                                                 temper=self.ring_pop_temper))
                    for col, (kind, j) in enumerate(own):
                        gr = self._global_row(kind, j)
                        v = bank.ref[col] + dev[:, col]
                        dof[:, gr] = ((v + np.pi) % (2 * np.pi) - np.pi) if kind == 'phi' else v
                        ring_rows.add(gr)
                    for kj in order:
                        gr = self._global_row(*kj)
                        if gr not in ring_rows:
                            ring_rows.add(hold(kj))
                    stats['n_ring_banked'] += 1
                elif bank is not None:
                    drawn = np.asarray(bank.sample(n, rng))
                    for col, ((kind, j), gr) in enumerate(zip(order, rows)):
                        if kind == 'r':
                            # bank the PUCKER, not the bond lengths. A bank row carries
                            # whatever molecule was fitted, so taking its r would import
                            # another molecule's bonds and override the thermal path --
                            # which is the exact local Boltzmann marginal for a harmonic
                            # term, and specific to THIS molecule. Pucker lives in the
                            # torsions and angles.
                            hold((kind, j))
                        else:
                            dof[:, gr] = drawn[:, col]
                        ring_rows.add(gr)
                    stats['n_ring_banked'] += 1
                else:
                    for kj in order:
                        ring_rows.add(hold(kj))
                    stats['n_ring_thermal'] += 1
                # ring-POSITIONING DoF outside the block are held either way: banked or
                # not, letting them float re-opens the ring (see ring_blocks)
                for kj in extra:
                    ring_rows.add(hold(kj))
                stats['n_ring_extra_held'] += len(extra)

        # ---- r / theta ----
        if thermal_rtheta:
            for j in range(self.n_r):
                if j not in ring_rows:
                    dof[:, j] = rng.normal(r0[j], s_r[j], n)
            for j in range(self.n_th):
                if self.n_r + j not in ring_rows:
                    dof[:, self.n_r + j] = rng.normal(th0[j], s_th[j], n)
            stats['rtheta_sigma_deg'] = (float(np.degrees(s_th.mean())), float(s_r.mean()))
        for row, (kind, hist, key, is_ring) in enumerate(types):
            stats['n_ring_marginal'] += int(is_ring and row not in ring_rows)
            if row in ring_rows:
                continue
            if (joint_torsions and row >= n_phi0) or (thermal_rtheta and row < n_phi0):
                continue
            dof[:, row] = draw(row, kind, hist)

        # ---- phi ----
        if joint_torsions:
            # improper rows FIRST: they are angles at the parent, not rotations, so they
            # rattle thermally about the reference instead of taking a rotamer histogram
            imp = [j for j in self.improper_phi_rows() if n_phi0 + j not in ring_rows]
            s_imp = self.improper_phi_sigma(float(self.temperature))
            stats['n_improper'] = len(imp)
            for j in imp:
                dof[:, n_phi0 + j] = ph0[j] + rng.normal(0.0, s_imp, n)
            stats['n_groups'] = len(groups)
            stats['sigma_deg'] = ((float(np.degrees(min(g_sigma))),
                                   float(np.degrees(max(g_sigma)))) if g_sigma else (0.0, 0.0))
            for gi, rows_j in enumerate(groups):
                grows = [n_phi0 + j for j in rows_j]
                in_ring = [i for i, gr in enumerate(grows) if gr in ring_rows]
                if in_ring and len(in_ring) == len(grows):
                    continue                       # wholly intra-ring: the bank owns it
                if in_ring:
                    # a mixed group is a ring bond carrying substituents. The ring member
                    # is already placed, so it leads and the substituents follow it -- an
                    # H on a ring carbon sits at a fixed offset from the ring's own dihedral
                    lead_i = in_ring[0]
                else:
                    lead_i = 0
                    _, hist, _, _ = types[grows[lead_i]]
                    dof[:, grows[lead_i]] = draw(grows[lead_i], 'phi', hist)
                disp = (dof[:, grows[lead_i]] - ph0[rows_j[lead_i]] + np.pi) % (2 * np.pi) - np.pi
                for i, gr in enumerate(grows):
                    if i == lead_i or gr in ring_rows:
                        continue
                    dof[:, gr] = ph0[rows_j[i]] + disp + rng.normal(0.0, g_sigma[gi], n)

            # ---- substituents on a ring atom whose group has no ring member ----
            # See ring_frame_groups. TWO-PASS, and the second pass is exact rather than
            # iterative: the frame (a, b, c) and the reference neighbour p are all RING
            # atoms, placed by the ring block, which sits upstream of every row corrected
            # here in the tree order. So the provisional build below fixes their positions
            # no matter what these rows currently hold, and one measurement is enough.
            frame_groups = self.ring_frame_groups(ring_rows) if joint_rings else []
            stats['n_ring_frame_groups'] = len(frame_groups)
            if frame_groups:
                tt = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)
                prov = self.build_positions(
                    self.state_from_dof(tt(dof[:, :self.n_r]),
                                        tt(dof[:, self.n_r:n_phi0]),
                                        tt(dof[:, n_phi0:])).clamp(-1.0, 1.0)
                ).reshape(n, -1, 3)
                ref = self.ref_pos.reshape(1, -1, 3)
                from mxtaltools.conformers.geometry import dihedral
                for rows_j, a, b, c, p, gi in frame_groups:
                    ind = dihedral(prov[:, a], prov[:, b], prov[:, c],
                                   prov[:, p]).detach().cpu().numpy()
                    ind0 = float(dihedral(ref[:, a], ref[:, b], ref[:, c],
                                          ref[:, p]).detach().cpu().numpy()[0])
                    for j in rows_j:
                        off = (ph0[j] - ind0 + np.pi) % (2 * np.pi) - np.pi
                        dof[:, n_phi0 + j] = ind + off + rng.normal(0.0, g_sigma[gi], n)
        else:
            # independent marginals for phi too: the pre-fix behaviour, kept so the A/B
            # is runnable and the gate below can require the difference
            for row in range(n_phi0, len(types)):
                if row in ring_rows:
                    continue
                kind, hist, _, _ = types[row]
                dof[:, row] = draw(row, kind, hist)

        t = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)
        x = self.state_from_dof(t(dof[:, :self.n_r]),
                                t(dof[:, self.n_r:self.n_r + self.n_th]),
                                t(dof[:, self.n_r + self.n_th:]))
        # a physical draw can land outside the box the sampler explores. Clip -- a
        # clipped row sits exactly ON the wall, where the wall is zero -- and report the
        # rate, because a high one means the box is too narrow for the prior and that is
        # information, not noise.
        outside = (x.abs() > 1.0)
        stats['clip_frac'] = {
            'r': float(outside[:, self._free_block == 0].to(self.dtype).mean()) if (self._free_block == 0).any() else 0.0,
            'theta': float(outside[:, self._free_block == 1].to(self.dtype).mean()) if (self._free_block == 1).any() else 0.0,
        }
        x = x.clamp(-1.0, 1.0)

        # CLOSURE MONITOR. Ring closure is the one constraint the state cannot express --
        # the closure bond is not a tree DoF, it is whatever the ring's internals imply --
        # so it has to be measured on the draw rather than assumed. Reported against a
        # bond's own thermal width, since that is the scale at which it stops mattering.
        # GATED ON THE MOLECULE, NOT ON joint_rings. It used to be gated on
        # ``stats['n_rings']``, which is zero whenever joint ring sampling is OFF -- so the
        # one configuration whose closure is catastrophic reported closure_err 0.000 and
        # read as perfect. Measured on cyclohexane at 'full': 0.086 A with rings on, 2.93 A
        # (75 bond-sigma) with them off, both previously indistinguishable at 0.000. A
        # diagnostic that goes quiet exactly where the thing it monitors fails is worse
        # than none, and it is what let the reference table benchmark the disabled path.
        # nan, not 0.0, when there is no closure bond: an acyclic molecule has no closure
        # error to report and 0.0 is a passing measurement of nothing.
        stats['closure_err'] = float('nan')
        stats['closure_sigma'] = float('nan')
        stats['n_closure_bonds'] = 0
        from mxtaltools.conformers.builder import closure_length
        tree, ff = self._batch(n)
        if ff.closure_index.numel():
            cl = closure_length(tree, self.build_positions(x))
            err = (cl - ff.closure_r0).abs().reshape(n, -1).max(1).values
            stats['closure_err'] = float(err.median())
            s_rc, _ = self.thermal_rtheta_sigma(float(self.temperature))
            stats['closure_sigma'] = stats['closure_err'] / max(float(np.mean(s_rc)), 1e-12)
            stats['n_closure_bonds'] = int(ff.closure_index.numel() // 2)
        stats['joint_rings'] = bool(joint_rings)

        if report:
            u = stats['n_uniform']
            print(f"InternalPrior draw: {n} states, {len(types)} DoF "
                  f"(uniform fallback r {u['r']}/{self.n_r}, theta {u['theta']}/{self.n_th}, "
                  f"phi {u['phi']}/{self.n_ph})")
            if joint_torsions:
                lo, hi = stats['sigma_deg']
                print(f"  phi drawn JOINTLY: {stats['n_groups']} bond groups, one leader "
                      f"each, siblings at the leader's displacement + N(0, sigma) with "
                      f"sigma {lo:.1f}-{hi:.1f} deg from the FF's own k_angle")
            else:
                print(f"  phi drawn INDEPENDENTLY per DoF (pre-fix behaviour)")
            print(f"  clipped to box: r {stats['clip_frac']['r']:.1%}, "
                  f"theta {stats['clip_frac']['theta']:.1%}")
            if stats['n_rings']:
                print(f"  rings: {stats['n_rings']} system(s) -- {stats['n_ring_banked']} "
                      f"from a fitted RingBank (joint, samples pucker), "
                      f"{stats['n_ring_thermal']} held at thermal jitter about the "
                      f"reference (closure preserved, pucker NOT sampled; aromatic rings "
                      f"take this path by design, being rigid)")
            elif stats['n_closure_bonds'] and not joint_rings:
                print(f"  rings: joint ring sampling is OFF -- {stats['n_closure_bonds']} "
                      f"closure bond(s) are being violated by construction. This is the "
                      f"NEGATIVE CONTROL, not a sampling mode.")
            if stats['n_closure_bonds']:
                print(f"  closure error {stats['closure_err']:.3f} A = "
                      f"{stats['closure_sigma']:.1f} bond-sigma"
                      + ("  <-- ABOVE 3 sigma, the ring is visibly open"
                         if stats['closure_sigma'] > 3 else ""))
            if stats['n_rings']:
                if getattr(self, 'ring_sig_stale', False):
                    print(f"  WARNING this prior predates the ring-signature fix "
                          f"(ring_sig_version < 2), so NO ring key can resolve and every "
                          f"ring above is held rather than banked. Refit to recover pucker "
                          f"sampling on saturated rings.")
            if stats['n_ring_marginal']:
                print(f"  WARNING {stats['n_ring_marginal']} ring DoF got independent "
                      f"MARGINALS. A product of marginals violates ring closure by "
                      f"construction -- valid support, poor proposal.")
        # the RAW dof, before state_from_dof and before the box clamp. prior_log_prob must
        # score what was actually drawn: scoring the clamped state would evaluate the
        # density at a point the sampler never proposed.
        stats['dof'] = dof
        return x, stats

    def potential_energy(self, x: torch.Tensor, temperature, keep_grads: bool = False,
                         return_positions: bool = False):
        """Bonded + LJ + box wall. NO change of measure, and NOT divided by T.

        Split out from energy() because `bake_energies` must store THIS: the baked field
        is divided by the sampling temperature when it is read back, and a change of
        measure divided by T is not a change of measure.
        """
        from mxtaltools.conformers.energy import intramolecular_energy

        grad_ctx = torch.enable_grad() if keep_grads else torch.no_grad()
        with grad_ctx:
            # _batch FIRST, and use what it RETURNS. It is a getter that mutates
            # (populating _tree_cache/_ff_cache as a side effect) and the old ordering
            # read the dict directly, relying on build_positions having already triggered
            # the fill -- so reordering the two lines gave a stale tree or a KeyError.
            tree, ff = self._batch(x.shape[0])
            pos = self.build_positions(x)
            e = intramolecular_energy(tree, pos, ff)
            if self._lin_free_idx.numel():
                # skipped, not added-as-zero, so the geometry path stays bitwise
                e = e + self.bounding_energy(x, temperature)
        return (e, pos) if return_positions else e

    def jacobian_energy(self, x: torch.Tensor, temperature) -> torch.Tensor:
        """``-T * log J``: the CHANGE OF MEASURE, in the potential's own units.

        ``log_jacobian`` is the BAT volume element -- ``prod r^2 sin(theta)``, relating
        the internal-coordinate measure to the full 3N Cartesian measure with the 6
        external DoF integrated at Haar x Lebesgue. It is NOT the determinant of `build`,
        which is SE(3)-reduced and square; the two differ by the orbit volume
        ``log(r_1^2 * r_2 * sin theta_2)``, so a test that compares them fails on correct
        code. See docs/design/internal_dof_ladder.md section 4.

        PRE-MULTIPLIED BY T, because energy() divides the total by T afterwards. Without
        it the term lands as ``J^(1/T)`` rather than ``J`` -- invisible at the default
        T=1, which is why the gate runs at two temperatures. This is the same compensation
        MolecularCrystal.compute_jacobian applies, and its comment names it the same way.

        ALWAYS ON, never gated on the freeze set: with r and theta frozen this is constant
        in x, but it is NOT constant in c, and log_jacobian's own docstring says it "must
        be added back if partition functions are compared across molecules".
        """
        from mxtaltools.conformers.builder import log_jacobian

        tree, _ = self._batch(x.shape[0])
        r, th, _ = self.dof_from_state(x)
        return -temperature * log_jacobian(tree, r.reshape(-1), th.reshape(-1))

    def energy(self, x, mol_batch=None, log_temperature=None,
               return_exp: bool = False, keep_grads: bool = False,
               internal_oom_recovery=None):
        """E/T per sample, ``[B]``. ``log_reward = -energy``.

        Carries BOTH changes of measure, so ``exp(-energy)`` is proportional to the
        Cartesian Boltzmann density read through the chart the SAMPLER actually proposes
        in -- the latent box, not the internal coordinates:

            log_reward = -U/T + log J_BAT + log|dq/dx|

        ``log J_BAT`` relates internal coordinates to Cartesian; ``log|dq/dx|`` relates the
        latent box to the internal coordinates and is a constant. Both are needed: the
        first alone gives a density on q, and the sampler does not propose q.

        Note this SHIFTS log Z relative to the pre-step-2 code by ``log J``, which is a
        constant at `torsion` and `dihedral` and state-dependent above them. Stored
        reference values from before the shift are not comparable.

        ``internal_oom_recovery`` is ACCEPTED AND IGNORED, deliberately. It selects the
        crystal energy's adaptive sub-batching path, which exists because an MLIP scan over
        a whole prior dataset can exhaust the card mid-call. This force field is a handful
        of fused kernels over a fixed-size state block with no such path to select, so
        there is nothing to switch on. It is in the signature because ``BaseSet.log_reward``
        forwards it unconditionally and the anchor scans pass it explicitly; raising here
        would make those callers crystal-only for no reason.
        """
        if log_temperature is None:
            log_temperature = torch.tensor(self.log_temperature)
        temperature = 10 ** torch.as_tensor(log_temperature, dtype=self.dtype,
                                            device=self.device)

        grad_ctx = torch.enable_grad() if keep_grads else torch.no_grad()
        with grad_ctx:
            e, pos = self.potential_energy(x, temperature, keep_grads=keep_grads,
                                           return_positions=True)
            e = e + self.jacobian_energy(x, temperature)
            # PRE-MULTIPLIED BY T for the same reason the BAT term is: a change of measure
            # must contribute the same amount to log_reward at every temperature, and
            # energy() divides the total by T below.
            e = e - temperature * self.log_chart_jacobian
        # BEFORE the division: this is what the crystal route stores as `gfn_energy`
        # (molecular_crystal.energy attaches it, then returns energy / temperature), and
        # what the eval publishes as 'Mean Sample Energy'. Keeping the same convention is
        # what makes that metric mean the same thing on both routes.
        gfn_e = e
        e = e / temperature
        if not return_exp:
            return e

        # THE SECOND RETURN IS A GRAPH BATCH, NOT POSITIONS. Every consumer of
        # return_exp=True -- fwd_eval_sampling, get_loss_reward, replay admission, the
        # anchor screen -- treats it as the crystal route does: a batch it can append,
        # index row-wise, read a state off and hand to a buffer. It used to return `pos`,
        # which no conformer caller ever consumed because the stripped training loop had
        # none of those paths.
        if mol_batch is None:
            raise ValueError(
                'energy(return_exp=True) returns the scored BATCH, so it needs a mol_batch '
                'to write onto; pass one or use return_exp=False')

        # `conformer_energy` is baked at T = 1 in bake_energies' convention, NOT from the
        # `e` computed above. Two reasons, and both would corrupt buffers silently rather
        # than fail: `e` is E/T with both measure terms folded in, and the read side
        # divides by T and re-adds the measure itself. A row admitted from here has to be
        # the same currency as a row from the prior dataset, or the buffers just mix them.
        with torch.no_grad():
            one = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            baked = self.potential_energy(x.detach(), one)

        from energies.conformer_data import set_batch_states

        return e, set_batch_states(mol_batch, x.detach(), baked,
                                   gfn_energy=gfn_e.detach(),
                                   periodic=self.periodic_dims)

    # ------------------------------------------------- Modeller energy protocol
    #
    # train.py's Modeller talks to its energy through a 15-member interface. Implementing
    # it here is what lets ConformerModeller subclass Modeller and inherit the protocol
    # controller, the buffer managers, the LR controller with its tripwires, checkpointing,
    # OOM handling, replay and z-calibration, rather than reimplementing them.
    #
    # `crystal` appears 17 times in train.py against 501 for `condition` -- the loop is
    # coupled to being CONDITIONAL, not to crystals, and a conformer conditioned on a
    # molecular graph is the same shape of problem. So this is an adapter, not a port.

    is_crystal = False

    @property
    def periodic_dims(self):
        """Which state dims live on a circle: the phi block, and only it.

        The base GFN infers this from `is_crystal`, which conflates "not a crystal" with
        "not periodic" and hands a non-crystal state ZERO wrapped dims -- silently, since
        that branch just writes `[False] * dim`. For a torsion state that is not a
        degraded layout but an unnormalizable target: the reward is exactly 2-periodic in
        every phi dim, so with no wrap the integral diverges and no log Z exists.

        Pass this to GFN(angular_mask=...). It was declared here from the start and read
        by NOTHING for the whole life of the conformer route -- one usage in the repo, its
        own definition -- while train.py:1502 kept inferring from is_crystal.
        """
        return [bool(b == 2) for b in self._free_block]

    @property
    def temperature(self):
        """The fixed sampling temperature, in the energy's own units (kcal/mol).

        Only meaningful when temperature_conditioning is off -- with it on, temperature is
        per-sample and rides the condition vector instead.
        """
        return 10.0 ** float(self.log_temperature)

    def set_n_molecules(self, n_molecules: int):
        """Called by init_identifiers() once the mol_id registry exists."""
        self.n_molecules = int(n_molecules)
        self.condition_library_size = self.n_molecules * self.n_sg * self.n_zp

    def condition_samples(self, mol_batch, temperature=None, sg_inds=None,
                          z_primes=None, repeats: int = 1):
        """Conformer analogue of MolecularCrystal.condition_samples.

        Returns ``(mol_batch, log_T_tensor, condition, condition_id)`` and attaches
        ``conditions`` / ``condition_id`` to the batch, matching the crystal contract so
        every caller in train.py works unchanged.

        ``sg_inds`` and ``z_primes`` are accepted and ignored -- a conformer has neither.
        They stay in the signature because callers pass them positionally by keyword and
        it costs nothing to tolerate; with n_sg = n_zp = 1 the mixed-radix condition_id
        collapses to mol_id exactly.

        Conditions are sampled per GROUP of ``repeats`` and broadcast, so all K rollouts
        for one molecule share a condition -- the same invariant the crystal path relies
        on for its exact-MLE and consistency objectives.
        """
        n = mol_batch.num_graphs
        n_groups = n // max(repeats, 1)
        dev = mol_batch.device

        if self.temperature_conditioning:
            if temperature is not None:
                log_T = torch.log10(torch.as_tensor(temperature, device=dev)).flatten()
            else:
                lo, hi = self.log_temperature_range
                u = torch.rand(n_groups, device=dev)
                log_T = (lo + u * (hi - lo)).repeat_interleave(max(repeats, 1))
            condition = log_T.reshape(-1, 1).float()
        else:
            log_T = torch.full((n,), float(self.log_temperature), device=dev)
            # matches the crystal's no-conditioning branch: a single zero column, so the
            # conditioner sees a well-shaped tensor rather than an empty one
            condition = torch.zeros((n, 1), device=dev)

        mol_id = getattr(mol_batch, 'mol_id', None)
        mol_id = (torch.zeros(n, dtype=torch.long, device=dev) if mol_id is None
                  else mol_id.to(dev))
        condition_id = mol_id * (self.n_sg * self.n_zp)   # n_sg = n_zp = 1

        mol_batch.conditions = condition.detach()
        mol_batch.condition_id = condition_id
        return mol_batch, log_T.flatten(), condition, condition_id

    def prebuilt_sample_to_reward(self, mols, temperature):
        """log reward for samples whose energy is already attached to the graphs.

        The crystal version re-scores from stored energy terms; the conformer equivalent
        needs a ``conformer_energy`` graph attribute, written by whatever prepared the
        buffer. Raising rather than silently rescoring is deliberate: a silent recompute
        here would hide a prep bug behind plausible numbers.
        """
        e = getattr(mols, 'conformer_energy', None)
        if e is None:
            raise AttributeError(
                "prebuilt_sample_to_reward needs a `conformer_energy` graph attribute; "
                "the prior/replay prep must attach it (see build_prior_states.py)")
        # `conformer_energy` stores the POTENTIAL only (bake_energies), because the baked
        # value is divided by the sampling T here and a change of measure divided by T is
        # not a change of measure. So the measure has to be added back, in log-reward
        # units, AFTER the division.
        t = torch.as_tensor(temperature, dtype=e.dtype, device=e.device).flatten()
        if self.log_jacobian_const is None:
            # STATE-DEPENDENT MEASURE: log J varies per row above `dihedral`, so a baked
            # scalar alone is not a reward. It is still reconstructible, because the graph
            # carries the state that produced it -- which is what unblocks `flex` and
            # `full` for every path that reads a prebuilt reward (backward training draws,
            # the prior buffer, the anchor seed). Recomputed rather than baked for the same
            # reason the potential is baked: a measure divided by the sampling temperature
            # is not a measure, so it cannot be folded into the stored scalar.
            from energies.conformer_data import batch_states
            from mxtaltools.conformers.builder import log_jacobian
            state = torch.as_tensor(batch_states(mols), dtype=self.dtype,
                                    device=self.device)
            r, th, _ = self.dof_from_state(state)
            tree, _ = self._batch(state.shape[0])
            log_j = log_jacobian(tree, r.reshape(-1), th.reshape(-1)).to(e.device)
            return -(e.flatten() / t) + log_j.flatten() + self.log_chart_jacobian
        # BOTH measure terms, or this path disagrees with energy() by a constant that is
        # different for every molecule. The chart term is as temperature-independent as the
        # BAT term and is added after the division for the same reason.
        return -(e.flatten() / t) + self.log_jacobian_const + self.log_chart_jacobian

    def batched_analyze_crystal_batch(self, *args, **kwargs):
        raise NotImplementedError(
            "batched_analyze_crystal_batch is crystal-only; both call sites in train.py "
            "are behind `if energy_function.is_crystal`, so reaching this is a bug")

    # ---------------------------------------------------------------- validation

    def brute_force_log_z(self, grid: int = 64, chunk: int = 4096) -> float:
        """Exact log Z by quadrature over the torus. Only sane for k <= 3.

        The point of the v0.1 target: for a few torsions the partition function is a
        low-dimensional periodic integral, so there is a ground truth to check the
        sampler against rather than a plausibility argument.
        """
        k = self.data_ndim
        if grid ** k > 5e7:
            raise ValueError(f"{grid}^{k} grid points is too many; lower `grid` or use "
                             f"a molecule with fewer rotatable bonds")
        # integrate over the STATE space [-1, 1]^k, matching what the sampler explores
        axis = torch.linspace(-1.0, 1.0, grid + 1, dtype=self.dtype,
                              device=self.device)[:-1]
        pts = torch.cartesian_prod(*([axis] * k)).reshape(-1, k)
        cell = (2.0 / grid) ** k

        acc = []
        for i in range(0, len(pts), chunk):
            acc.append((-self.energy(pts[i:i + chunk])).double())
        log_terms = torch.cat(acc)
        return (torch.logsumexp(log_terms, 0) + np.log(cell)).item()

    def sample(self, batch_size):
        raise NotImplementedError(
            "no closed-form sampler; use brute_force_log_z for ground truth at small k, "
            "or build_conformer_buffer.py for a mode-covering reference set")
