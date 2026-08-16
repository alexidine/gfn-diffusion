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
                 dtype=torch.float64,
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
        self.dtype = dtype
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
        self.ff_single = ff_from_reference(
            tree1, pos1, epsilon=epsilon, min_separation=min_separation,
            scale_14=scale_14, lj_k_factor=lj_k_factor)

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

        self.angle_is_linear = _linear(self.spec.angle_index)
        self.torsion_frame_is_linear = _linear(np.asarray(self.spec.torsion_index)[:, :3])
        self.linearity_verified = True

        # ring membership in PLACEMENT-SLOT numbering, for the prior draw. InternalPrior
        # samples ring systems JOINTLY (a whole observed DoF block) because closure is a
        # hard constraint that a product of marginals is guaranteed to violate; the
        # per-DoF draw below cannot do that, so it has to know when it is being asked to.
        in_ring_orig = np.array([a.IsInRing() for a in mol.GetAtoms()], dtype=bool)
        self.atom_in_ring = np.zeros(len(z), dtype=bool)
        self.atom_in_ring[slot] = in_ring_orig

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

    def _batch(self, batch_size: int):
        """Cached collated tree and force field for a given batch size."""
        if batch_size not in self._tree_cache:
            from mxtaltools.conformers.builder import collate
            from mxtaltools.conformers.energy import ff_from_reference

            tree = collate([self.spec] * batch_size, device=self.device)
            ref = self.ref_pos.repeat(batch_size, 1)
            self._tree_cache[batch_size] = tree
            self._ff_cache[batch_size] = ff_from_reference(tree, ref, **self._ff_kwargs)
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
        """phi DoF rows grouped by central bond, leader first.

        A group is one bond's substituent set -- every dihedral that turns when that bond
        turns. Their DIFFERENCES are what fix the local geometry: an H-C-H angle is a
        difference of two of them, and it is one of the 40 graph angles the force field
        scores but the tree does not expose as a coordinate. So they have to be drawn
        JOINTLY. Drawn independently, even from perfect marginals, roughly a third of
        sibling pairs land on the same rotamer mode and put two substituents in the same
        place -- measured on Ala5 at a median sibling-difference error of 91 degrees.
        """
        from collections import defaultdict
        ti = np.asarray(self.spec.torsion_index)
        g = defaultdict(list)
        for j in range(self.n_ph):
            g[(int(ti[j, 1]), int(ti[j, 2]))].append(j)
        return [sorted(rows) for rows in g.values()]

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
                            joint_torsions: bool = True, thermal_rtheta: bool = True):
        """``[n, d]`` states drawn from a fitted InternalPrior. Returns ``(x, stats)``.

        Per-DoF marginals, which is the prior's acyclic case. Ring systems are the one
        place this is NOT what InternalPrior would do -- it draws a whole observed DoF
        block per ring system, because closure is a hard constraint that a product of
        marginals is guaranteed to violate. Ring DoF here therefore get marginals and the
        count is REPORTED rather than silently absorbed: the draws are still valid support
        (all TB needs), just much worse proposals.
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

        if thermal_rtheta:
            s_r, s_th = self.thermal_rtheta_sigma(float(self.temperature))
            r0 = self.r0.detach().cpu().numpy(); th0 = self.th0.detach().cpu().numpy()
            for j in range(self.n_r):
                dof[:, j] = rng.normal(r0[j], s_r[j], n)
            for j in range(self.n_th):
                dof[:, self.n_r + j] = rng.normal(th0[j], s_th[j], n)
            stats['rtheta_sigma_deg'] = (float(np.degrees(s_th.mean())), float(s_r.mean()))
        for row, (kind, hist, key, is_ring) in enumerate(types):
            stats['n_ring_marginal'] += int(is_ring)
            # phi is handled below when drawing jointly; r/theta above when thermal
            if (joint_torsions and row >= n_phi0) or (thermal_rtheta and row < n_phi0):
                continue
            dof[:, row] = draw(row, kind, hist)

        if joint_torsions:
            groups = self.torsion_groups()
            sigmas = self.sibling_jitter_sigma(groups, float(self.temperature))
            ph0 = self.ph0.detach().cpu().numpy()
            stats['n_groups'] = len(groups)
            stats['sigma_deg'] = (float(np.degrees(min(sigmas))),
                                  float(np.degrees(max(sigmas)))) if sigmas else (0.0, 0.0)
            for gi, rows in enumerate(groups):
                lead = rows[0]
                _, hist, _, _ = types[n_phi0 + lead]
                lead_val = draw(n_phi0 + lead, 'phi', hist)
                dof[:, n_phi0 + lead] = lead_val
                # the group turns together: every member takes the LEADER'S displacement
                # from its own reference, which is what preserves the reference offsets
                disp = (lead_val - ph0[lead] + np.pi) % (2 * np.pi) - np.pi
                for r in rows[1:]:
                    dof[:, n_phi0 + r] = ph0[r] + disp + rng.normal(0.0, sigmas[gi], n)
        else:
            # independent marginals for phi too: the pre-fix behaviour, kept so the A/B
            # is runnable and the gate below can require the difference
            for row in range(n_phi0, len(types)):
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
            if stats['n_ring_marginal']:
                print(f"  WARNING {stats['n_ring_marginal']}/{len(types)} DoF are in ring "
                      f"systems and got MARGINALS, not InternalPrior's joint ring block. "
                      f"A product of marginals violates ring closure by construction, so "
                      f"these draws are valid support but poor proposals.")
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
               return_exp: bool = False, keep_grads: bool = False):
        """E/T per sample, ``[B]``. ``log_reward = -energy``.

        Carries the change of measure, so ``exp(-energy)`` is proportional to the
        Cartesian Boltzmann density read through the internal-coordinate chart:

            log_reward = -U/T + log J

        Note this SHIFTS log Z relative to the pre-step-2 code by ``log J``, which is a
        constant at `torsion` and `dihedral` and state-dependent above them. Stored
        reference values from before the shift are not comparable.
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
        e = e / temperature
        return (e, pos) if return_exp else e

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
        if self.log_jacobian_const is None:
            raise NotImplementedError(
                f"the change of measure is state-dependent at level {self.level!r}, so a "
                f"reward cannot be reconstructed from a baked scalar. Recomputing log J "
                f"needs each row's own state, which this signature does not receive -- "
                f"see docs/design/internal_dof_ladder.md section 6, where "
                f"prebuilt_sample_to_reward is rewritten to read `torsion_state` off the "
                f"graph. Until then this path is torsion/dihedral only.")
        t = torch.as_tensor(temperature, dtype=e.dtype, device=e.device).flatten()
        return -(e.flatten() / t) + self.log_jacobian_const

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
