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
                 **kwargs):
        super().__init__()
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

        self.rotatable, self.mask = self._find_rotatable(bonds, z, include_trivial_rotations)
        self.data_ndim = len(self.rotatable)
        if self.data_ndim == 0:
            raise ValueError(f"{smiles} has no rotatable bonds; pick a flexible molecule")

        self._ff_cache, self._tree_cache = {}, {}
        self._ff_kwargs = dict(epsilon=epsilon, min_separation=min_separation,
                               scale_14=scale_14, lj_k_factor=lj_k_factor)
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
        lines = [f"{self.smiles}: {self.spec.n_atoms} atoms, {self.spec.n_dof} internal DoF, "
                 f"{self.data_ndim} rotatable torsions sampled"]
        for j, (u, v) in enumerate(self.rotatable):
            moved = int(self.mask[:, j].sum())
            lines.append(f"   torsion {j}: rotate about {name(u)}-{name(v)} "
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

    def build_positions(self, x: torch.Tensor) -> torch.Tensor:
        """Torsion deltas ``[B, k]`` in **[-1, 1]** -> Cartesian positions ``[B * N, 3]``.

        ``phi = phi_ref + mask @ (pi * x)`` is a pure translation on the torus, so this
        map has constant Jacobian on top of the (constant) frozen-geometry one. The
        pi factor is a fixed scale, so it too only shifts log Z by k*log(pi).
        """
        from mxtaltools.conformers.builder import build

        x = x.to(self.dtype) * np.pi
        b = x.shape[0]
        tree, _ = self._batch(b)
        phi = (self.ph0.unsqueeze(0) + x @ self.mask.T).reshape(-1)
        return build(tree, self.r0.repeat(b), self.th0.repeat(b), phi)

    def energy(self, x, mol_batch=None, log_temperature=None,
               return_exp: bool = False, keep_grads: bool = False):
        """E/T per sample, ``[B]``. ``log_reward = -energy``."""
        from mxtaltools.conformers.energy import intramolecular_energy

        if log_temperature is None:
            log_temperature = torch.tensor(self.log_temperature)
        temperature = 10 ** torch.as_tensor(log_temperature, dtype=self.dtype,
                                            device=self.device)

        grad_ctx = torch.enable_grad() if keep_grads else torch.no_grad()
        with grad_ctx:
            pos = self.build_positions(x)
            _, ff = self._batch(x.shape[0])
            e = intramolecular_energy(self._tree_cache[x.shape[0]], pos, ff)
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
        """Which state dims live on a circle: for torsions, all of them.

        The base GFN infers this from `is_crystal`, which conflates "not a crystal" with
        "not periodic" and hands a torsion state ZERO wrapped dims. Declaring the layout
        here (rather than having the model guess from the energy's type) is what makes
        that inference unnecessary.
        """
        return [True] * self.data_ndim

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
        t = torch.as_tensor(temperature, dtype=e.dtype, device=e.device).flatten()
        return -(e.flatten() / t)

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
