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


class ChartRefused(ValueError):
    """A molecule this chart refuses at a tier, with a machine-readable ``code``.

    A ValueError, so every existing ``except ValueError`` and ``pytest.raises(ValueError)``
    keeps working; the code is what a conditions builder records, so a rejection list can be
    counted by cause without parsing prose:

      * ``incomplete_chart`` -- rows held at a linear centre the chart could not cover;
      * ``wholly_linear`` -- every atom on one line: 3N-5 internal DoF, so no 3N-6 chart
        exists and the shortfall is the molecule's, not the chart's;
      * ``cumulated`` -- a collinear frame through a cumulated sp centre (allene, cumulene),
        held deliberately: freeing it frees an end-to-end twist the force field does not
        restrain.

    and, only when the stereo lock is on (``stereo_coeff > 0``; energies/stereo_lock.py):

      * ``stereo_unspecified`` -- the SMILES leaves a lockable stereo element unassigned, so
        the isomer the lock would pin is whatever the embedding happened to realise;
      * ``stereo_unsupported`` -- a stereo element the lock does not enforce is tagged (a
        tetrahedral N, an allene or atropisomer axis);
      * ``stereo_verify_failed`` -- the reference embedding realised a different isomer
        than the SMILES names;
      * ``stereo_lock_in_band`` -- an element's best indicator is within ``MIN_MARGIN`` of
        zero at the reference, so the wrong configuration would pay too little to be locked
        out (build_conformer_conditions.py also records it when the lock fires on a thermal
        sample of the correct isomer, `stereo_lock.thermal_check`);
      * ``stereo_torsion_double_bond`` -- at ``torsion`` a rotatable column turns a locked
        double bond, which the torsion prior draws into both E and Z.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ConformerTorsions(BaseSet):
    # Free-DoF levels, as freeze sets over InternalParams.CLASSES = ("r","theta","phi").
    # These are NOT a ladder of approximations: freezing a DoF at a constant gives
    # p_full(free | frozen = c0), a conditional slice, which differs from the
    # rigid-constraint ensemble by a state-dependent Fixman factor. `full` is the target;
    # the rest are each some related distribution, useful for staging and regression.
    # See docs/design/internal_dof_ladder.md section 2.
    LEVELS = ("torsion", "dihedral", "flex", "full")
    # THE TIERS THAT CARRY A LINEAR CENTRE THROUGH: the transverse bend, the Z-matrix dummy
    # frame and the sp-root rule. They move angle rows (theta), so a tier that holds every
    # theta cannot use them; `torsion` and `dihedral` keep the pre-transverse chart byte for
    # byte, which is what keeps the helper tiers' stored anchors meaning what they meant.
    CHART_TIERS = ("flex", "full")
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
                 #: Pre-encoded molecular identity, appended to the condition vector. Same
                 #: name and same contract as MolecularCrystal's, so `get_conditioning_dim`
                 #: and the config invariants already handle it. The embedding is baked by
                 #: models/encoder_cache.py from a FROZEN encoder, so there is no per-step
                 #: cost and no gradient into it -- see that module on why the encoder can
                 #: be frozen and why it must be pre-encoded rather than run in the loop.
                 embedding_conditioning: bool = False,
                 embedding_conditioning_dim: Optional[int] = None,
                 log_temperature_range=(-1.0, 1.0),
                 lj_coeff: float = 1.0,
                 *,
                 level: str,
                 delta_r_max: float = 0.30,
                 delta_theta_max: float = 0.50,
                 bounding_coeff: float = 10.0,
                 r_floor: float = 0.50,
                 theta_floor: float = 1.0e-3,
                 # radius at which the TRANSVERSE disc wall engages, in radians of rho. The
                 # chart is injective only on rho < pi; this is a soft preference well inside
                 # it, exactly zero below rho_wall, and measured to be inert at the current
                 # operating point (max rho over a 400-epoch run: 0.682). See bounding_energy.
                 rho_wall: float = 1.0,
                 # opt in to a chart that is INCOMPLETE at level 'full'. Off by default: a
                 # globally nonlinear molecule has 3N-6 internal degrees of freedom whether
                 # or not a centre is locally linear, so a shortfall at 'full' means this
                 # chart froze real coordinates and is sampling a different distribution.
                 # Partial coverage is fine when it is EXPLICIT -- which is what setting this
                 # makes it -- and not fine when it is silent.
                 allow_constrained: bool = False,
                 ring_jitter_scale: float = 0.1,
                 ring_min_bank_rows: int = 2,
                 force_field: str = 'reference',
                 ring_mode_fill: float = 0.0,
                 ring_pop_temper: float = 1.0,
                 energy_clip: float = None,
                 # THE STEREO LOCK's stiffness, kcal/mol per unit squared indicator
                 # (energies/stereo_lock.py). 0 = off, the default, so no existing target
                 # changes: the table is still built (condition graphs and the per-coordinate
                 # features carry it), but no term is added and nothing is refused.
                 stereo_coeff: float = 0.0,
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
        # SOFT REWARD CLIP. Absolute kcal/mol; None = off, which is the default so no
        # existing run changes. NOT the crystal's `reward_range`: that derives its cutoff
        # from max(dataset_rewards), and on this route the prior dataset is SAMPLED AND
        # SCORED before the tier minimum is known, so an energy-distribution-relative
        # cutoff would be set after 50,000 rows are already baked. An absolute number is
        # order-independent and cannot acquire that bug.
        if energy_clip is not None:
            energy_clip = float(energy_clip)
            if not np.isfinite(energy_clip):
                raise ValueError(f'energy_clip must be finite, got {energy_clip!r}')
        self.energy_clip = energy_clip
        # Modeller energy-protocol fields -- see the protocol section below. n_sg/n_zp are
        # 1 (not 0) so the mixed-radix condition_id arithmetic stays valid and collapses
        # to mol_id; condition_library_size is re-set by init_identifiers().
        self.energy_function = 'conformer_torsions'
        self.temperature_conditioning = bool(temperature_conditioning)
        self.embedding_conditioning = bool(embedding_conditioning)
        if self.embedding_conditioning and embedding_conditioning_dim is None:
            raise ValueError(
                "embedding_conditioning requires embedding_conditioning_dim (the width of "
                "the `embedding` on each conditions-file entry -- 2 * the encoder hidden "
                "size, e.g. 256 for the shipped 128-wide encoder, because the pooled "
                "readout is [softmax-weighted sum || unnormalised sum])")
        self.embedding_conditioning_dim = embedding_conditioning_dim
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
        # NEVER ROOT ON AN sp CARBON, at the tiers that carry linear centres. Rooted there,
        # the seed angle between the root's two neighbours is linear and nothing downstream
        # can recover it (topology.choose_root). Gated to CHART_TIERS because the rule changes
        # the tree of exactly the molecules whose default root is an sp carbon, at every tier
        # it runs at -- and at torsion/dihedral those charts must not move. It changes nothing
        # else: a molecule whose default root is not an sp carbon gets the identical tree, and
        # one whose root it does move is checked below to have been held without it.
        self.avoid_sp_root = level in self.CHART_TIERS
        self.spec = spec_from_graph(z, bonds, ref_pos, use_geometry=False,
                                    avoid_sp_root=self.avoid_sp_root)
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
        _is_linear = _typed_linear if self.linearity_source == 'mmff_typed' else _linear

        # THE ROOT MOVED ONLY WHERE IT HAD TO. `choose_root` reads the GRAPH (a carbon with two
        # neighbours) because MolData's tree builder has no typing; this chart's own linearity
        # predicate is the authority on whether that atom is a linear centre. If it is not,
        # the default tree would have built -- and moving its root would change a chart that
        # works today -- so the disagreement is refused rather than absorbed. Measured over all
        # 133,726 QM9 molecules the two agree exactly; this makes that a checked property.
        if self.spec.root_moved_from >= 0:
            _r = int(slot[self.spec.root_moved_from])
            _nb = sorted({int(v) for u, v in self.bond_index_slot.T if int(u) == _r}
                         | {int(u) for u, v in self.bond_index_slot.T if int(v) == _r})
            if len(_nb) != 2 or not bool(_is_linear(np.array([[_nb[0], _r, _nb[1]]]))[0]):
                raise RuntimeError(
                    f"{smiles}: the sp-root rule moved the root off atom {_r} (a carbon with "
                    f"two neighbours), but this chart's {self.linearity_source} linearity does "
                    f"not call that atom a linear centre -- so the default tree was not "
                    f"singular there and the move would change a chart that builds today. "
                    f"The graph rule and the typing disagree on this molecule.")

        # CUMULATED sp CENTRES (allene, cumulene: a carbon with two double bonds), in slot
        # numbering. A collinear frame through one is NOT given a dummy reference below: for an
        # allene the dummy row IS the end-to-end twist, which the held chart pins at the
        # reference and MMFF94 does not restrain (flat to 0.007 kcal/mol under a rigid 90 deg
        # twist of penta-2,3-diene), so freeing it would interconvert axially chiral
        # stereoisomers the condition is supposed to fix. QM9 has none.
        from rdkit.Chem import BondType
        self.cumulated_centres = np.array(sorted(
            int(slot[a.GetIdx()]) for a in mol.GetAtoms()
            if a.GetAtomicNum() == 6 and a.GetDegree() == 2
            and all(b.GetBondType() == BondType.DOUBLE for b in a.GetBonds())),
            dtype=np.int64)

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

        # ---- THE STEREO LOCK's table, built whether or not the lock is on --------------
        # Off (stereo_coeff 0) it adds no term and refuses nothing; the table still exists so
        # a condition graph and the per-coordinate features are the same object either way.
        self.stereo_coeff = stereo_coeff             # validated by the property's setter
        self._init_stereo(smiles, slot, pos_np, level)
        self.rotatable_cols = (np.argmax(mask_np, axis=0).astype(np.int64)
                               if mask_np.shape[1] else np.zeros(0, dtype=np.int64))

        n_at = self.spec.n_atoms
        self.n_r, self.n_th, self.n_ph = n_at - 1, n_at - 2, n_at - 3
        n_dof = self.n_r + self.n_th + self.n_ph          # == 3N - 6 == spec.n_dof
        assert n_dof == self.spec.n_dof, (n_dof, self.spec.n_dof)

        # ---- DUMMY-FRAME rows ---------------------------------------------------------
        # A torsion row whose frame a-b-c is COLLINEAR (b an sp centre, a and c on its axis)
        # has no plane to measure phi in, so today it is held. The Z-matrix remedy is a DUMMY
        # ATOM X on b at 90 degrees to the axis, and phi measured as X-b-c-d: the row stays
        # driven and `full` keeps 3N-6. See the note above mxtaltools builder.DummyFrame for
        # the construction and why log J is unchanged.
        #
        # WHICH ROWS: this chart's own collinear-frame flags, at CHART_TIERS only (the
        # transverse chart's tiers, so torsion and dihedral stay byte-identical), minus
        #   * frames through a CUMULATED centre (see `cumulated_centres`), and
        #   * rows X cannot be built for (builder.dummy_frame_refs: b must be c's parent and
        #     not the root) or whose azimuth anchor is itself a linear angle -- an unchained
        #     X's anchor angle is a tree angle at b's parent, and if THAT is linear the anchor
        #     is on the axis. Dropping a row can unchain its children, so this is iterated to
        #     a fixed point: each pass removes at least one row or stops, so it ends within
        #     n_ph + 1 passes.
        # Rows that remain collinear and unflagged are held exactly as before.
        _ti = np.asarray(self.spec.torsion_index)
        self.cumulated_frames = (np.asarray(self.torsion_frame_is_linear, dtype=bool)
                                 & np.isin(_ti[:, 1], self.cumulated_centres))
        dummy = np.zeros(self.n_ph, dtype=bool)
        if level in self.CHART_TIERS:
            from mxtaltools.conformers.builder import dummy_frame_refs
            dummy = np.asarray(self.torsion_frame_is_linear, dtype=bool) & ~self.cumulated_frames
            for _ in range(self.n_ph + 1):
                _refs, _ok = dummy_frame_refs(tree1, torch.as_tensor(dummy), strict=False)
                _anc = _refs.anchor.cpu().numpy()[_ti[:, 3]]
                _unch = dummy & ~_refs.chained.cpu().numpy()[_ti[:, 3]] & (_anc >= 0)
                # the anchor angle's row is the one placing its LATER arm: b's own row when
                # the anchor is b's angle reference, the anchor's when b is the root's first
                # child -- in placement numbering angle row j places slot j + 2
                _own = np.maximum(_ti[:, 1], _anc) - 2
                _anc_lin = np.zeros_like(dummy)
                _anc_lin[_unch] = np.asarray(self.angle_is_linear, dtype=bool)[_own[_unch]]
                keep = dummy & _ok.cpu().numpy() & ~_anc_lin
                if (keep == dummy).all():
                    break
                dummy = keep
        #: per TORSION ROW: this row's phi (or transverse v) is measured against the Z-matrix
        #: dummy X instead of its collinear real reference atom
        self.dummy_frame_rows = dummy
        #: per torsion row: collinear and NOT carried by a dummy, hence held
        self.held_frame_rows = np.asarray(self.torsion_frame_is_linear, dtype=bool) & ~dummy
        if dummy.any():
            # RE-MEASURED, because the measure above ran before the flags existed: on a
            # dummy row it read the dihedral against the collinear real atom, i.e. noise
            # multiplied by a vanishing frame. `ph0` stays POLAR (theta0/phi0 are what the
            # prior histograms read); only its reference atom changes. Nothing else moves --
            # r0 and th0 do not depend on the torsion frame.
            _, _, ph0 = measure(tree1, pos1, dummy_frame=torch.as_tensor(dummy))
            self.ph0 = ph0

        # ---- TRANSVERSE linear-bending rows ------------------------------------------
        # A linear bend is a POLE OF THE (theta, phi) CHART, not a rigid constraint. Below,
        # such a row is re-expressed as the transverse pair (u, v) = rho (cos phi, sin phi)
        # with rho = pi - theta, which is regular at the pole and carries the SAME two
        # coordinates -- so the row stays driven instead of being held, and `full` keeps its
        # 3N-6. See mxtaltools/conformers/geometry.py:place_nerf_transverse.
        #
        # THREE CONDITIONS, and the last two are why this does not fix every linear centre:
        #   1. the bend is linear                              (angle_is_linear)
        #   2. the atom HAS a torsion row to carry v           (frame seeds do not)
        #   3. its placement frame a-b-c is NOT itself collinear, or is carried by a dummy
        # Failing 2 means the missing component is the sixth EXTERNAL DoF, under a frame
        # convention that stops fixing a frame at all once atoms 0-1-2 are collinear -- which
        # the sp-root rule removes wherever the molecule is not wholly linear. Failing 3 means
        # the normal defining phi is arbitrary, so there is no frame to bend in; the dummy
        # frame above is the smooth frame construction that supplies one. Rows failing either
        # are still HELD, and `describe()` reports them separately.
        ang_atom = np.asarray(self.spec.angle_index)[:, 2]
        tor_atom = np.asarray(self.spec.torsion_index)[:, 3]
        slot_of = {int(a): i for i, a in enumerate(tor_atom)}
        partner = np.array([slot_of.get(int(a), -1) for a in ang_atom], dtype=np.int64)
        has_partner = partner >= 0
        frame_bad = np.zeros(self.n_th, dtype=bool)
        frame_bad[has_partner] = self.held_frame_rows[partner[has_partner]]
        #: per ANGLE ROW: this row's (theta, phi) pair is carried as (u, v)
        self.transverse_angles = self.angle_is_linear & has_partner & ~frame_bad
        #: per angle row: the torsion row holding its v, -1 where there is none
        self.transverse_partner = partner
        self.uncovered_linear_angles = int(
            (self.angle_is_linear & ~self.transverse_angles).sum())

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

        # A TRANSVERSE PAIR IS ONE 2-D COORDINATE AND CANNOT BE HALF-FREE. `dihedral` drives
        # the phi rows only, so it would free v and leave u frozen -- half a bend, and a
        # measure term reading a component the state cannot move. `torsion` is worse: its
        # col_block is overwritten to all-2 above, so a driven transverse column would be
        # declared PERIODIC, and u and v do not wrap.
        #
        # Both tiers therefore keep the PREVIOUS treatment -- the linear row stays held and
        # is reported as constrained -- rather than being refused. The transverse chart is a
        # `flex` / `full` feature, which is where the target is; narrowing here rather than
        # raising keeps every tier below it byte-identical to the pre-transverse code.
        if level == 'torsion':
            self.transverse_angles = np.zeros_like(self.transverse_angles)
        else:
            _in_sel = np.zeros(n_dof, dtype=bool)
            _in_sel[sel] = True
            for _j in np.flatnonzero(self.transverse_angles):
                if not (_in_sel[self.n_r + _j]
                        and _in_sel[self.n_r + self.n_th + partner[_j]]):
                    self.transverse_angles[_j] = False
        self.uncovered_linear_angles = int(
            (self.angle_is_linear & ~self.transverse_angles).sum())

        # BLOCK 3 = transverse. A distinct code rather than reusing 1 or 2, because these
        # columns differ from both: unlike phi they do NOT wrap (so `periodic_dims`, which
        # reads `block == 2`, correctly excludes them and `state_from_dof` does not fold
        # them onto a circle), and unlike theta they must NOT be clamped to (0, pi) -- they
        # live on a disc of radius pi and the pole is an interior point, not a boundary.
        _tv_rows = np.flatnonzero(self.transverse_angles)
        if _tv_rows.size:
            block[self.n_r + _tv_rows] = 3
            block[self.n_r + self.n_th + partner[_tv_rows]] = 3
            col_block = block[sel]

        # A DoF sitting on a parameterisation singularity is HELD, not driven: log sin
        # theta diverges as theta -> pi and the dependent dihedral frame is undefined
        # there. Zeroing the ROW (not dropping the column) is what makes this uniform
        # across levels; a column left driving nothing is then dropped below.
        #
        # ⚠ THIS IS A CONSTRAINED APPROXIMATION, NOT `full`, AND IT IS NOT BECAUSE THE
        # MOTION IS ABSENT. An earlier version of this comment said these bends are
        # "physically stiff and constant anyway"; that is wrong on its own terms -- measured
        # over alkynes and nitriles they cost 0.62-1.43 kcal/mol at +/-10 deg against 2.2-2.4
        # for the ordinary angles this chart keeps free, i.e. they are among the SOFTEST
        # angular perturbations tested, not the stiffest.
        #
        # A linear EQUILIBRIUM geometry is not a rigid constraint. The divergence is a
        # SINGULARITY OF THE (theta, phi) CHART, exactly as the polar coordinate (rho, phi)
        # is singular at rho = 0 while displacement in either transverse direction stays
        # perfectly physical. The standard remedy in the internal-coordinate literature is a
        # pair of TRANSVERSE LINEAR-BENDING coordinates rather than deletion:
        #
        #     rho = pi - theta,  u = rho cos(phi),  v = rho sin(phi)
        #     sin(theta) dtheta dphi = (sin(rho) / rho) du dv        -> finite, smooth at 0
        #
        # so the divergent `log sin theta` becomes a regular term, the coordinate count is
        # unchanged (2 -> 2), and `data_ndim` returns to 3N-6. Freezing instead DEFINES A
        # DIFFERENT DISTRIBUTION -- legitimate as an approximation, but it is not the `full`
        # target and must not be reported as one.
        #
        # Until that chart lands, the reduction is RECORDED rather than silent: see
        # `self.constrained_rows` and the CONSTRAINED line in `describe()`.
        # A TRANSVERSE ROW IS NOT SINGULAR AND IS NOT HELD -- that is the whole point of the
        # pair. Its partner phi row carries v and is free for the same reason; it cannot be
        # in `held_frame_rows`, because condition 3 above excluded exactly those. Nor is a
        # DUMMY-FRAME row: its frame is X-b-c, which is regular, so only the collinear rows
        # no dummy carries are held.
        singular = np.zeros(n_dof, dtype=bool)
        singular[self.n_r + np.flatnonzero(self.angle_is_linear
                                           & ~self.transverse_angles)] = True
        singular[self.n_r + self.n_th + np.flatnonzero(self.held_frame_rows)] = True
        m_full[singular, :] = 0.0

        keep = m_full.any(axis=0)
        #: rows held at their reference because the (theta, phi) chart is singular there, and
        #: the state columns lost with them. A GLOBALLY NONLINEAR molecule has 3N-6 internal
        #: degrees of freedom whether or not one of its centres is locally linear -- the
        #: 3N-5 count belongs to a WHOLLY linear molecule -- so a shortfall here is a property
        #: of this chart, never of the molecule.
        self.constrained_rows = int(singular.sum())
        self.constrained_columns = int((~keep).sum())
        # REFUSED IN THE ENERGY, not only in build_conformer_conditions.py. The builder guard
        # covered exactly one of the eight paths that construct a chart; a config naming a
        # constrained molecule at 'full' trained silently, and its run summary, its checkpoint
        # and its conditions file all said 'full'. Raising here closes every path at once and
        # demotes the builder's check to a redundant early skip.
        #
        # THREE CAUSES, each with its own code (ChartRefused), because they mean different
        # things: a WHOLLY LINEAR molecule has 3N-5 internal DoF, so no 3N-6 chart exists and
        # the shortfall is the molecule's; a CUMULATED centre is held on purpose; anything
        # else is this chart's limitation.
        if level == 'full' and self.constrained_rows and not allow_constrained:
            n_lin = 3 * self.spec.n_atoms - 6
            deg = np.bincount(self.bond_index_slot.reshape(-1), minlength=n_at)
            centres = np.flatnonzero(deg >= 2)
            wholly_linear = bool(len(centres)) and bool((deg[centres] == 2).all())
            if wholly_linear:
                _nbr = {int(i): sorted({int(v) for u, v in self.bond_index_slot.T if u == i}
                                       | {int(u) for u, v in self.bond_index_slot.T if v == i})
                        for i in centres}
                wholly_linear = bool(_is_linear(np.array(
                    [[_nbr[i][0], i, _nbr[i][1]] for i in _nbr]).reshape(-1, 3)).all())
            if wholly_linear:
                raise ChartRefused(
                    'wholly_linear',
                    f"{smiles} is WHOLLY LINEAR: every atom lies on one axis, so it has 3N-5 "
                    f"internal degrees of freedom, not 3N-6 = {n_lin}. No atom tree charts "
                    f"it at level 'full' -- a property of the molecule, not a chart "
                    f"limitation; the missing coordinate is the rotation about its own axis, "
                    f"which is external. Pass allow_constrained=True to study the held chart "
                    f"deliberately.")
            n_cum = int(self.cumulated_frames.sum())
            if n_cum:
                raise ChartRefused(
                    'cumulated',
                    f"{smiles} has {n_cum} collinear frame(s) through a CUMULATED sp centre "
                    f"(allene or cumulene), held deliberately: the dummy frame would free the "
                    f"end-to-end twist, which MMFF94 does not restrain, and axially chiral "
                    f"isomers would interconvert. It does not have a complete chart at level "
                    f"'full' until a twist term exists. Pass allow_constrained=True to study "
                    f"the held chart deliberately.")
            raise ChartRefused(
                'incomplete_chart',
                f"{smiles} does not have a complete chart at level 'full': "
                f"{self.constrained_rows} row(s) held and {self.constrained_columns} "
                f"column(s) dropped, giving d = {int(keep.sum())} against 3N-6 = {n_lin}. "
                f"{self.uncovered_linear_angles} linear angle(s) are not covered by the "
                f"transverse pair (frame seed, or collinear reference frame) and "
                f"{int(self.held_frame_rows.sum())} collinear frame(s) could not take a dummy "
                f"reference -- a chart limitation, not a rigid molecule. Pass "
                f"allow_constrained=True to study it deliberately; the shortfall is then "
                f"recorded on the run.")
        m_full, col_block = m_full[:, keep], col_block[keep]
        self.data_ndim = int(keep.sum())
        if self.data_ndim == 0:
            raise ValueError(
                f"{smiles} at level {level!r} has no free degrees of freedom. "
                f"{self.constrained_rows} row(s) were held because the (theta, phi) chart is "
                f"singular at a linear centre -- which is a chart limitation, not a rigid "
                f"molecule. See section 3.2 of docs/design/conformer_parameterisation.md.")

        #: per STATE COLUMN: 0=r 1=th 2=phi 3=TRANSVERSE (a (u, v) component of a linear
        #: bend -- non-periodic like r/theta, unclamped like neither; see the block-3 note)
        self._free_block = col_block
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

        scale = np.select(
            [col_block == 0, col_block == 1, col_block == 3],
            [float(delta_r_max), float(delta_theta_max), float(delta_theta_max)],
            default=np.pi)

        # THE REFERENCE IN CHART UNITS. A transverse row's stored reference is (u0, v0), not
        # (theta0, phi0), or `dof_from_state` would write a bend displacement onto an angle.
        # `self.th0`/`self.ph0` stay POLAR: they are what the prior histograms and the
        # reporting paths are keyed on, and converting them in place would silently change
        # the meaning of every one of those consumers.
        th_ref, ph_ref = th0, ph0
        if bool(self.transverse_angles.any()):
            from mxtaltools.conformers.geometry import transverse_from_polar
            tj = torch.as_tensor(_tv_rows, dtype=torch.long, device=th0.device)
            tm = torch.as_tensor(partner[_tv_rows], dtype=torch.long, device=th0.device)
            u0, v0 = transverse_from_polar(th0.index_select(0, tj), ph0.index_select(0, tm))
            th_ref = th0.clone().index_copy(0, tj, u0)
            ph_ref = ph0.clone().index_copy(0, tm, v0)
            # The chart is valid on the open disc rho < pi. The reachable set is
            # |u - u0| <= delta_theta_max and likewise v, so this is a CHECK on the
            # configured scale rather than an assumption about it.
            reach = (u0.abs() + float(delta_theta_max)) ** 2 + \
                    (v0.abs() + float(delta_theta_max)) ** 2
            if float(reach.max()) >= np.pi ** 2:
                raise ValueError(
                    f"{smiles}: delta_theta_max {delta_theta_max} lets a transverse bend "
                    f"reach rho >= pi, where the (u, v) chart stops being injective and its "
                    f"measure turns negative. Reduce it, or exclude this molecule.")
            # TIGHTER FOR A BEND THAT ANCHORS A DUMMY FRAME: the frame angle X-b-c is
            # pi/2 +- rho_c, so the dummy chart goes collinear at rho_c = pi/2 -- inside the
            # disc, long before the (u, v) chart's own boundary at pi.
            _anchors = np.isin(ang_atom[_tv_rows], _ti[dummy, 2])
            if _anchors.any() and float(reach[torch.as_tensor(_anchors)].max()) >= (np.pi / 2) ** 2:
                raise ValueError(
                    f"{smiles}: delta_theta_max {delta_theta_max} lets a transverse bend that "
                    f"anchors a dummy frame reach rho >= pi/2, where that frame goes collinear "
                    f"(its angle is pi/2 +- rho). Reduce it, or exclude this molecule.")

        # THE DUMMY FRAMES' CONDITIONING, BOUNDED OVER THE BOX rather than sampled. Every frame
        # a dummy row touches has an angle the box confines: X_d's own frame angle
        # angle(X_d, b, c) = pi/2 +- rho_c; a chained X's azimuth frame angle(X_c, q, b) =
        # pi/2 +- rho_b; an unchained X's is a tree angle theta0 +- delta_theta_max. The sine of
        # that angle scales the frame normal's cross product, so its minimum over the box is a
        # certificate that no frame goes collinear anywhere a state in [-1, 1]^d can put it.
        # The design this replaced took X's azimuth from c's OWN frame, whose sibling-pick
        # angle is set by free dihedrals and was measured 0.42 deg from collinear in the box.
        self.dummy_frame_min_sin = None
        if dummy.any():
            _rho_max = {}
            if bool(self.transverse_angles.any()):
                for _k, _j in enumerate(_tv_rows):
                    _rho_max[int(ang_atom[_j])] = float(reach[_k].sqrt())
            _th0 = th0.detach().cpu().numpy()
            _rho_of = lambda a: _rho_max.get(int(a), float(np.pi - _th0[int(a) - 2]))
            _ch = _refs.chained.cpu().numpy()[_ti[:, 3]]
            _anc = _refs.anchor.cpu().numpy()[_ti[:, 3]]
            _sins = []
            for _j in np.flatnonzero(dummy):
                _b, _c = int(_ti[_j, 1]), int(_ti[_j, 2])
                _sins.append(np.cos(_rho_of(_c)))
                if _ch[_j]:
                    _sins.append(np.cos(_rho_of(_b)))
                else:
                    _t0 = _th0[max(_b, int(_anc[_j])) - 2]
                    _sins.append(min(np.sin(_t0 - float(delta_theta_max)),
                                     np.sin(_t0 + float(delta_theta_max))))
            #: smallest sine of any dummy-row frame angle over the whole state box; > 0 is
            #: the regularity certificate (see the comment above)
            self.dummy_frame_min_sin = float(min(_sins))
        self._ref_dof = torch.cat([r0, th_ref, ph_ref]).to(dtype)
        self._free_scale = torch.as_tensor(scale, dtype=dtype, device=self.device)
        # indexes the STATE, not the DoF vector: the box wall applies to the non-periodic
        # blocks only. Empty at `torsion` and `dihedral`, which is what keeps those levels
        # bitwise identical to the pre-ladder code.
        self._lin_free_idx = torch.as_tensor(np.flatnonzero(col_block != 2),
                                             dtype=torch.long, device=self.device)
        #: per ANGLE ROW, for `build` / `log_jacobian` / the theta clamp. None when no row
        #: is transverse, which keeps every molecule without a linear centre on exactly the
        #: code path it was on before.
        self._transverse_t = (
            torch.as_tensor(self.transverse_angles, dtype=torch.bool, device=self.device)
            if bool(self.transverse_angles.any()) else None)
        #: STATE COLUMNS of each transverse pair, u and v aligned pairwise. Derived from the
        #: same `_M` the map uses rather than from the block code, because block 3 alone does
        #: not say WHICH v belongs to WHICH u -- and a wall or a crossing count computed on
        #: mismatched halves would be a plausible number for the wrong quantity.
        _u_cols, _v_cols = [], []
        if self._transverse_t is not None:
            _sel = np.argmax(m_full, axis=0)               # state column -> its DoF row
            _row_of_col = {int(r): int(c) for c, r in enumerate(_sel)
                           if m_full[int(r), int(c)] != 0}
            for _j in _tv_rows:
                _ur = self.n_r + int(_j)
                _vr = self.n_r + self.n_th + int(partner[int(_j)])
                if _ur in _row_of_col and _vr in _row_of_col:
                    _u_cols.append(_row_of_col[_ur])
                    _v_cols.append(_row_of_col[_vr])
        self._tv_u_cols = torch.as_tensor(_u_cols, dtype=torch.long, device=self.device)
        self._tv_v_cols = torch.as_tensor(_v_cols, dtype=torch.long, device=self.device)
        # the affine map's reference and SIGNED scale for those columns, so rho can be formed
        # in chart radians straight from a state. Signed because `_M` is a signed selection:
        # a chart that drove a row negatively would otherwise get a wall on |x| with the
        # wrong sense, silently.
        _uref = [float(th_ref[int(_j)]) for _j in _tv_rows[:len(_u_cols)]]
        _vref = [float(ph_ref[int(partner[int(_j)])]) for _j in _tv_rows[:len(_u_cols)]]
        _usc = [float(m_full[self.n_r + int(_j), _c] * scale[_c])
                for _j, _c in zip(_tv_rows[:len(_u_cols)], _u_cols)]
        _vsc = [float(m_full[self.n_r + self.n_th + int(partner[int(_j)]), _c] * scale[_c])
                for _j, _c in zip(_tv_rows[:len(_u_cols)], _v_cols)]
        _t = lambda a: torch.as_tensor(a, dtype=dtype, device=self.device)
        self._tv_u_ref, self._tv_v_ref = _t(_uref), _t(_vref)
        self._tv_u_scale, self._tv_v_scale = _t(_usc), _t(_vsc)
        #: radius past which the transverse wall engages. Below pi with a wide margin: the
        #: measured maximum over a 400-epoch run is 0.682, so this is inert today.
        self.rho_wall = float(rho_wall)
        # every surviving transverse row must own BOTH state columns -- guaranteed by the
        # narrowing above, asserted because a wall or a crossing count formed from half a
        # pair would be a plausible number for the wrong quantity rather than an error.
        assert len(_u_cols) == int(self.transverse_angles.sum()), (
            f"{smiles} at {level!r}: {len(_u_cols)} complete pairs of "
            f"{int(self.transverse_angles.sum())} flagged")
        #: per transverse PAIR (the `_tv_u_cols` order): this bend anchors a dummy frame, so
        #: its chart is regular only on rho < pi/2 -- see `dummy_frame_crossings`
        self._tv_dummy_anchor = torch.as_tensor(
            np.isin(ang_atom[_tv_rows[:len(_u_cols)]], _ti[dummy, 2]), dtype=torch.bool,
            device=self.device)
        # THE DISC WALL MUST ENGAGE BEFORE THE DUMMY FRAME DEGENERATES. The box keeps rho well
        # inside pi/2 (checked above), but the wall is what prices a state OUTSIDE the box, and
        # past pi/2 a dummy frame is collinear: a wall that only starts there guards nothing.
        if dummy.any() and not self.rho_wall < np.pi / 2:
            raise ValueError(
                f"{smiles}: rho_wall {self.rho_wall} is not below pi/2, where the dummy frames "
                f"this molecule carries go collinear; the disc wall must engage first")

        #: per TORSION ROW, for `build` / `measure`. None when no row carries a dummy frame,
        #: which keeps every molecule without one on exactly the code path it was on before.
        self._dummy_t = (torch.as_tensor(dummy, dtype=torch.bool, device=self.device)
                         if dummy.any() else None)
        #: per torsion row, the atom in the first slot of the row's FRAME: `torsion_index[:, 0]`
        #: except on a dummy row, where it is the real atom whose side X points to (the anchor
        #: at the start of a dummy chain) -- the collinear atom in `torsion_index` fixes nothing
        self.torsion_frame_first = np.asarray(_ti[:, 0], dtype=np.int64).copy()
        if dummy.any():
            from mxtaltools.conformers.builder import dummy_frame_anchor_atoms
            _real = dummy_frame_anchor_atoms(tree1, self._dummy_t).cpu().numpy()
            self.torsion_frame_first[dummy] = _real[_ti[dummy, 3]]

        #: set when the caller opted in to an incomplete chart, so downstream reporting can
        #: mark the result rather than letting `level: full` speak for it
        self.allow_constrained = bool(allow_constrained)
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
        # A TRANSVERSE ROW IS NOT A THETA ROW and this guard does not apply to it: its slot
        # holds u, whose domain is a disc, not the interval (0, pi). Checking it here would
        # reject every molecule the transverse pair exists to support -- theta0 ~ pi by
        # definition at a linear centre, so theta0 + delta always breaches pi. The disc is
        # checked separately, on (u0, v0), where the constructor builds the reference.
        free_th = free_th & ~self.transverse_angles
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

        # SEEDED WITH THE BATCH-1 ENTRY ALREADY BUILT ABOVE. `_batch(1)` would collate
        # [spec] into a tree identical to tree1 and hand `_make_ff` the same positions and
        # kwargs -- so for 'mmff' it re-ran ff_from_mmff, every RDKit parameter query
        # again, only to rebuild ff_single for the log J probe and e_ref just below. One
        # typing per construction; the eviction rule treats this entry like any other.
        self._ff_cache, self._tree_cache = {1: self.ff_single}, {1: tree1}
        self._ff_kwargs = dict(epsilon=epsilon, min_separation=min_separation,
                               scale_14=scale_14, lj_k_factor=lj_k_factor)

        # log J is constant in x exactly when no r/theta column is free -- `torsion` and
        # `dihedral`. Recorded, NOT used to short-circuit jacobian_energy: one code path
        # means the constant cannot silently drift from the computed value. Its uses are
        # (a) adding the measure back to a baked potential, and (b) reporting, since it is
        # the offset by which step 2 moved every stored log Z on those levels.
        _probe = torch.zeros(1, self.data_ndim, dtype=dtype, device=self.device)
        _tree, _ = self._batch(1)
        _pr, _pth, _pph = self.dof_from_state(_probe)
        self.log_jacobian_const = (
            float(self._log_jac(_tree, _pr, _pth, _pph, 1).item())
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

    # ------------------------------------------------------------------ stereo lock

    @property
    def stereo_coeff(self) -> float:
        """The stereo lock's stiffness, kcal/mol; 0 = off. See energies/stereo_lock.py."""
        return self._stereo_coeff

    @stereo_coeff.setter
    def stereo_coeff(self, value):
        """Set ONCE, at construction; afterwards only the value already in force is accepted.

        train.set_energy_coeffs sets any numeric energy_config key a stage's coeff_schedule or
        anneal names, every step, so a ramp of this key reaches here -- and is refused, for two
        reasons. A BAKED row (prior, anchor, replay) stores clip(U + P) at the coefficient in
        force when it was scored, and prebuilt_sample_to_reward never recomputes it, so after
        any change the stored rows and the live ones score different locks, silently; every
        condition graph records the coefficient it was built under (``ctree_stereo_coeff``) and
        `MultiConformerTorsions._resolve_rows` refuses a row whose record differs. And whether
        the lock is on decides things fixed at construction -- the refusals of an unassigned or
        unverifiable stereoisomer, and the prior's held double-bond rows -- so a ramp from 0
        would lock whatever isomer the embedding happened to realise, unrefused and unrecorded.
        """
        value = float(value)
        if not (np.isfinite(value) and value >= 0.0):
            raise ValueError(f'stereo_coeff must be finite and >= 0, got {value!r}')
        old = getattr(self, '_stereo_coeff', None)
        if old is not None and value != old:
            raise ValueError(
                f'stereo_coeff {old:g} -> {value:g} after construction. It is fixed for the '
                f'life of the energy: baked rows (prior, anchor, replay) carry the lock at '
                f'{old:g} and are never re-scored, and the refusals and held prior rows were '
                f'decided at construction. Set it in energy_config and name it in no stage\'s '
                f'coeff_schedule or anneal_coeffs.')
        self._stereo_coeff = value

    def _init_stereo(self, smiles: str, slot: np.ndarray, pos_slot: np.ndarray, level: str):
        """Build `self.stereo` (energies/stereo_lock.StereoTable) and, if locking, refuse.

        THE ISOMER IS PINNED ON THE CONDITION SIDE. The tagged SMILES says which stereoisomer
        the condition is; the lock enforces it; so with the lock on, a SMILES that leaves a
        lockable element unassigned is refused rather than locked to whatever the ETKDG
        embedding realised -- a single seeded embedding, which a different RDKit build can
        resolve the other way under the same SMILES and the same problem hash. With every
        element tagged, the SMILES in energy_config (or a condition's identifier) names the
        locked isomer, which is how it enters the problem identity.

        The signs are then read off the reference, not off the tags, so the reference must BE
        the tagged isomer: `realised_isomer` re-perceives it from 3D and it must equal the
        input's canonical isomeric SMILES.
        """
        from energies import stereo_lock as sl

        #: the INPUT molecule's potential stereo elements (stereo_lock.tagged_elements), in
        #: `self.mol`'s atom indexing -- the condition side of the stereo contract
        self.stereo_elements = sl.tagged_elements(smiles)
        dbl = [tuple(int(slot[a]) for a in e['atoms']) for e in self.stereo_elements
               if e['kind'] == sl.DOUBLE_BOND]
        centres = [int(slot[e['atoms'][0]]) for e in self.stereo_elements
                   if e['kind'] == sl.TETRAHEDRAL and e['specified'] and e['degree'] == 4]
        self.stereo = sl.build_table(pos_slot, self.bond_index_slot, dbl, stereocentres=centres)
        #: the isomer the reference conformer REALISES (re-perceived from 3D), and the input's
        self.stereo_isomer = sl.realised_isomer(self.mol)
        self.stereo_input = sl.canonical_isomeric(smiles)
        if self.stereo_coeff <= 0.0:
            return

        odd = [e for e in self.stereo_elements if e['kind'] == 0]
        inv = [e for e in self.stereo_elements
               if e['kind'] == sl.TETRAHEDRAL and e['degree'] != 4 and e['specified']]
        if odd or inv:
            what = ([f"{e['type']} at atom {e['atoms']}" for e in odd]
                    + [f"tetrahedral {e['element']}{e['atoms'][0]} (3-coordinate)" for e in inv])
            raise ChartRefused(
                'stereo_unsupported',
                f"{smiles}: stereo element(s) {what} are present, but the stereo lock does not "
                f"enforce them -- a three-coordinate centre (an amine N) is left to invert "
                f"inside one condition (RDKit does not recover N stereo from 3D, so it could "
                f"not be verified), and axial stereo has no indicator. Strip those tags, or run "
                f"with stereo_coeff 0.")
        isomers = sl.consistent_isomers(smiles)
        if len(isomers) != 1:
            loose = [(e['type'], e['atoms']) for e in self.stereo_elements
                     if not e['specified'] and not (e['kind'] == sl.TETRAHEDRAL
                                                    and e['degree'] != 4)]
            raise ChartRefused(
                'stereo_unspecified',
                f"{smiles}: the tags are consistent with {len(isomers)} stereoisomers (e.g. "
                f"{isomers[:3]}); unassigned elements {loose}. With stereo_coeff > 0 the lock "
                f"pins one isomer, and the SMILES must say which: an untagged element would be "
                f"locked to whatever the ETKDG embedding realised ({self.stereo_isomer!r} "
                f"this time), which the problem identity does not record. Pass a fully tagged "
                f"stereoisomer.")
        if self.stereo_isomer != self.stereo_input:
            raise ChartRefused(
                'stereo_verify_failed',
                f"{smiles}: the reference embedding realises {self.stereo_isomer!r}, not the "
                f"requested {self.stereo_input!r}. The lock reads its signs off the "
                f"reference, so it would pin the wrong isomer.")
        thin = np.flatnonzero(self.stereo.margin < sl.MIN_MARGIN)
        if thin.size:
            raise ChartRefused(
                'stereo_lock_in_band',
                f"{smiles}: {thin.size} stereo element(s) have best |indicator| below "
                f"{sl.MIN_MARGIN} at the reference (atoms {self.stereo.key[thin].tolist()}, "
                f"|v| {np.round(self.stereo.margin[thin], 4).tolist()}): the reference sits "
                f"near the stereo boundary, where the wrong configuration's mirror point pays "
                f"too little to be locked out (stereo_lock.MIN_MARGIN).")
        if level == 'torsion':
            # THE TORSION TIER'S ROTATABLE SET HAS NO BOND-ORDER TEST (_find_rotatable), so an
            # acyclic C=C or C=N is a column there and its prior (build_prior_states.
            # draw_states) draws it per column, into both E and Z. Freezing it instead would
            # change the torsion tier's column count, a helper-tier target change that is the
            # owner's to make; refused until then.
            locked = self.stereo.bonds()
            hit = [(int(u), int(v)) for u, v in self.rotatable
                   if frozenset((int(u), int(v))) in locked]
            if hit:
                raise ChartRefused(
                    'stereo_torsion_double_bond',
                    f"{smiles}: at level 'torsion' the rotatable column(s) about {hit} turn a "
                    f"locked double bond. The torsion tier admits acyclic double bonds as "
                    f"rotatable (no bond-order test) and its prior draws both E and Z, so the "
                    f"lock would reject a fixed share of every prior draw. Run it unlocked "
                    f"(stereo_coeff 0), or at dihedral/flex/full.")

    def held_phi_rows(self):
        """phi rows the prior HOLDS at their reference: impropers, plus locked double bonds.

        `improper_phi_rows` always. With the lock on, also every row whose central bond is a
        LOCKED double bond: the sibling-group draw takes a group's leader from a rotamer
        histogram keyed on the central bond, which puts a double bond into the other E/Z
        basin on a large share of draws, and the lock rejects every one of them. Held rows
        take the improper path instead -- a thermal rattle about the reference -- at every
        site improper rows are special (`torsion_groups`, `sample_prior_states`,
        `prior_log_prob`), so the draw and its density stay matched.

        Identical to `improper_phi_rows` when the lock is off, so no prior draw changes.
        """
        rows = set(self.improper_phi_rows())
        if self.stereo_coeff > 0.0 and self.stereo.n:
            locked = self.stereo.bonds()
            ti = np.asarray(self.spec.torsion_index)
            rows |= {j for j in range(self.n_ph)
                     if frozenset((int(ti[j, 1]), int(ti[j, 2]))) in locked}
        return sorted(rows)

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
        n_free = [int((self._free_block == b).sum()) for b in (0, 1, 2, 3)]
        lines = [f"{self.smiles}: {self.spec.n_atoms} atoms, {self.spec.n_dof} internal DoF",
                 f"   level {self.level!r}: {self.data_ndim} free "
                 f"(r {n_free[0]}/{self.n_r}, theta {n_free[1]}/{self.n_th}, "
                 f"phi {n_free[2]}/{self.n_ph}, transverse {n_free[3]})",
                 f"   linearity flags {self.linearity_source.upper()}: "
                 f"{int(self.angle_is_linear.sum())} linear "
                 f"angle(s), {int(self.torsion_frame_is_linear.sum())} ill-conditioned "
                 f"frame(s)"]
        n_tv = int(self.transverse_angles.sum())
        if n_tv:
            lines.append(
                f"   TRANSVERSE: {n_tv} linear bend(s) carried as (u, v) = rho(cos phi, "
                f"sin phi), regular at the pole; their measure is log sinc(rho), not "
                f"log sin(theta). {self.uncovered_linear_angles} linear angle(s) NOT "
                f"covered (frame seed or collinear reference frame) and still held.")
        n_dm = int(np.asarray(getattr(self, 'dummy_frame_rows', ())).sum())
        if n_dm:
            lines.append(
                f"   DUMMY FRAME: {n_dm} dihedral(s) whose frame runs along a linear axis are "
                f"measured against a Z-matrix dummy atom (90 deg to the axis) instead of the "
                f"collinear real atom; min frame-angle sine over the box "
                f"{self.dummy_frame_min_sin:.3f}. "
                f"{int(np.asarray(self.held_frame_rows).sum())} collinear frame(s) held.")
        if getattr(self.spec, 'root_moved_from', -1) >= 0:
            lines.append(f"   ROOT moved off the sp carbon the default rule picks (input atom "
                         f"{self.spec.root_moved_from}); see topology.choose_root")
        # SAY SO WHEN THE TIER IS NOT WHAT IT CLAIMS. A globally nonlinear molecule has 3N-6
        # internal degrees of freedom regardless of a locally linear centre, so at 'full' any
        # shortfall is this chart's, and reporting `full` without saying so would present a
        # constrained approximation as the complete one.
        if getattr(self, 'constrained_rows', 0):
            expected = 3 * self.spec.n_atoms - 6
            lines.append(
                f"   CONSTRAINED, not {self.level!r}: {self.constrained_rows} row(s) held and "
                f"{self.constrained_columns} column(s) dropped at linear centre(s), where the "
                f"(theta, phi) chart is singular -- NOT because the bend is absent")
            if self.level == 'full':
                lines.append(
                    f"      d = {self.data_ndim} against 3N-6 = {expected}; the molecule is "
                    f"globally nonlinear, so the shortfall is the chart's. Transverse "
                    f"linear-bending coordinates would restore it (design note 3.2).")
        st = getattr(self, 'stereo', None)
        if st is not None:
            n_t = int((st.kind == 1).sum())
            n_b = int((st.kind == 2).sum())
            state = (f'ON, stereo_coeff {self.stereo_coeff:g} kcal/mol'
                     if self.stereo_coeff > 0 else 'OFF (stereo_coeff 0): table built, no term')
            lines.append(
                f"   STEREO LOCK {state}: {n_t} tetrahedral centre(s) ({int(st.stereocentre.sum())} "
                f"stereocentre(s), the rest labelled parity), {n_b} double bond(s); reference "
                f"realises {self.stereo_isomer!r}")
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

    _batch_cache_slack = 1.5

    def _batch(self, batch_size: int):
        """Cached collated tree and force field for a given batch size.

        BOUNDED, because the cache lives on the device and is keyed on batch size while
        the batch sizer WALKS batch sizes. One entry costs ~28 MB per 1000 samples for a
        26-atom molecule, so an unbounded cache parks a gigabyte of dead tree on the card
        after a single ladder walk to 20000 -- which competes with the live batch, trips
        the OOM cycle, moves the batch size, and caches another entry.

        The budget is in SAMPLES, not entries. Capping the entry count instead retains
        the most recent few, and on a monotonically growing ladder those are the LARGEST
        few -- measured worse than no cap at all (+1557 MB against +1046 MB over a
        seven-rung walk to 20000). Here the entry just requested is always kept, since it
        is about to be used, and least-recently-used entries are dropped until the total
        is within ``_batch_cache_slack`` of it. So the overhead is a fixed fraction of the
        one entry that is genuinely needed, whatever size the sizer settles on.

        A miss is cheap enough to be worth eating: collate is tiled, so a rebuild is
        ~0.4 s at 20000 rather than the ~11 s it was before that path existed.
        """
        if batch_size in self._tree_cache:
            self._tree_cache[batch_size] = self._tree_cache.pop(batch_size)   # touch: MRU
            self._ff_cache[batch_size] = self._ff_cache.pop(batch_size)
            return self._tree_cache[batch_size], self._ff_cache[batch_size]

        from mxtaltools.conformers.builder import collate

        tree = collate([self.spec] * batch_size, device=self.device)
        ref = self.ref_pos.repeat(batch_size, 1)
        self._tree_cache[batch_size] = tree
        self._ff_cache[batch_size] = self._make_ff(tree, ref, **self._ff_kwargs)
        budget = self._batch_cache_slack * batch_size
        while len(self._tree_cache) > 1 and sum(self._tree_cache) > budget:
            stale = next(iter(self._tree_cache))          # dicts iterate in insertion order
            self._tree_cache.pop(stale, None)
            self._ff_cache.pop(stale, None)
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
            th_c = th.clamp(self.theta_floor, np.pi - self.theta_floor)
            # A TRANSVERSE ROW IS NOT CLAMPED. Its slot holds u, not theta: the domain is a
            # disc on which the pole is an INTERIOR point, so clamping to [floor, pi-floor]
            # would fence off the linear geometry all over again -- and would do it by
            # pinning a positive-probability region onto a boundary, which is the failure
            # this chart was adopted to avoid. The disc is enforced at construction instead,
            # by the reach check on delta_theta_max.
            th = th_c if self._transverse_t is None else torch.where(self._transverse_t,
                                                                     th, th_c)
        return r, th, ph

    def build_positions(self, x: torch.Tensor) -> torch.Tensor:
        """State ``[B, d]`` in **[-1, 1]** -> Cartesian positions ``[B * N, 3]``.

        The per-block affine map (delta from a reference, fixed per-block scale) has a
        constant Jacobian, so it shifts log Z by a constant -- but the constant has counts
        affine in N, so it is a PER-MOLECULE offset and must be carried wherever log Z(c)
        is compared across molecules.
        """
        r, th, ph = self.dof_from_state(x)
        tree, _ = self._batch(x.shape[0])
        return self._build(tree, r, th, ph, x.shape[0])

    def _build(self, tree, r, th, ph, b: int) -> torch.Tensor:
        """`builder.build` with BOTH chart masks: transverse bends and dummy frames.

        The counterpart of `_log_jac`, and routed through one helper for the same reason:
        each mask changes what a slot MEANS, so a call site that builds without one returns
        a finite, right-shaped geometry for a different state -- a silent error. Every
        builder.build on this molecule's chart goes through here (build_positions, the prior
        smoke harness), so there is no way to build it with a mask left behind.
        """
        from mxtaltools.conformers.builder import build
        return build(tree, r.reshape(-1), th.reshape(-1), ph.reshape(-1),
                     transverse=self._tiled_transverse(b), dummy_frame=self._tiled_dummy(b))

    def _tiled_transverse(self, b: int):
        """The per-angle-row transverse mask, tiled over a b-replica batch tree.

        `_batch(b)` collates b copies of one spec molecule-major, so the batched
        `angle_index` is this molecule's rows repeated b times and a plain tile lines up.
        None passes straight through, which is what keeps a molecule with no linear centre
        byte-identical to the pre-transverse code path.
        """
        return None if self._transverse_t is None else self._transverse_t.repeat(b)

    def _tiled_dummy(self, b: int):
        """The per-torsion-row dummy-frame mask, tiled like `_tiled_transverse`; None if none."""
        return None if self._dummy_t is None else self._dummy_t.repeat(b)

    def torsion_frame_atoms(self) -> np.ndarray:
        """``[n_ph, 4]`` the atoms that FIX each dihedral row, in spec numbering.

        `spec.torsion_index` with its first column replaced, on a dummy-frame row, by the
        real atom the dummy X points toward (`torsion_frame_first`). Anything that names a
        row's frame -- the per-coordinate features, the atom frames a correlator reads --
        takes it from here: on a dummy row the collinear atom `torsion_index` names carries
        no direction at all.
        """
        out = np.asarray(self.spec.torsion_index, dtype=np.int64).copy()
        out[:, 0] = self.torsion_frame_first
        return out

    def _log_jac(self, tree, r, th, ph, b: int):
        """`builder.log_jacobian` with the transverse rows measured as ``log sinc(rho)``.

        Routed through one helper because the mask has to reach EVERY call site. A site that
        built with it and measured without it would return a correct geometry under a wrong
        density -- finite, right-shaped, and wrong -- so there is deliberately no way to
        compute this Jacobian here without the mask coming along.
        """
        from mxtaltools.conformers.builder import log_jacobian
        tv = self._tiled_transverse(b)
        return log_jacobian(tree, r.reshape(-1), th.reshape(-1),
                            None if tv is None else ph.reshape(-1), transverse=tv)

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
        w = self.bounding_coeff * v.sum(-1)

        # THE TRANSVERSE DISC, walled in RHO rather than per column. The box above is a
        # SQUARE in (x_u, x_v) and does not know the transverse domain is a disc: at
        # |x_u| = |x_v| = 1, where it is exactly zero, rho is already 0.71, and the diagonal
        # route out is measured 42 kcal/mol CHEAPER than the axial one. So the cheapest
        # escape is the direction nothing was checking.
        #
        # A PREFERENCE, NOT A CLAMP -- it pins nothing and it is exactly zero inside
        # rho_wall. The number is measured, not chosen: over 947,022 transverse rows of a
        # 400-epoch run the largest rho was 0.682, equal to the seed prior's own maximum, and
        # the physical 99.99th percentile is 0.53. At rho_wall = 1.0 this term is therefore
        # identically zero on every state either run has ever produced, and it only bites in
        # a regime nothing has reached.
        #
        # WHY IT IS NEEDED AT ALL, given log sinc already vanishes at rho = pi: that barrier
        # is LOGARITHMIC and therefore weak -- 8.1 nats one milliradian from the boundary,
        # against a box wall worth 279 kcal/mol at the same point. The measure marks the
        # boundary; it does not hold it. Past rho = pi the chart is an orientation-reversing
        # double cover and log_sinc is -inf, so the reward is zero and the TB residual
        # infinite. See docs/design/conformer_parameterisation.md 3.2.
        if self._transverse_t is not None:
            rho2 = self._transverse_rho2(x)
            rho = torch.sqrt(rho2.clamp_min(1e-24))
            w = w + self.bounding_coeff * (torch.relu(rho - self.rho_wall) ** 2).sum(-1)
        return w * temperature

    def _transverse_rho2(self, x: torch.Tensor) -> torch.Tensor:
        """``u^2 + v^2`` per transverse PAIR, ``[B, n_pairs]``. Empty when there are none.

        IN RADIANS OF THE CHART, not in state units: ``u = u0 + scale * x``, the same affine
        map `dof_from_state` applies. A first version of this read the raw state columns as
        though they were u and v, which made the wall fire on the box coordinate instead of
        the bend and made the crossing counter report a crossing at rho = 2.0 -- inside the
        disc, whose boundary is pi. The two differ by a factor of 1/delta_theta_max, so the
        error was a plausible number for the wrong quantity in both consumers at once.

        Computed from the state rather than by calling `dof_from_state`, so the wall and the
        counter can run on the proposal itself before any reconstruction.
        """
        xt = x.to(self.dtype)
        u = self._tv_u_ref + self._tv_u_scale * xt.index_select(-1, self._tv_u_cols)
        v = self._tv_v_ref + self._tv_v_scale * xt.index_select(-1, self._tv_v_cols)
        return u * u + v * v

    def transverse_crossings(self, x: torch.Tensor) -> int:
        """How many transverse pairs in this batch left the chart's disc (``rho >= pi``).

        NAMED AND COUNTED rather than left to surface as a NaN. Past the disc the chart is
        not injective, ``log sinc`` is ``-inf``, the log reward is ``-inf`` and one row takes
        the whole batch's TB loss and every gradient to infinity -- with nothing in the reward
        path checking finiteness, so the operator would see a dead run and no attribution.
        Extrapolated per-row probability is 1e-15 (pessimistic) to 1e-127 (Gaussian fit) at
        the current operating point, and the measured maximum over 947,022 rows is 0.682
        against a boundary of pi -- so this is expected to read zero forever, and that is
        exactly why it has to be reported rather than assumed.
        """
        if self._transverse_t is None:
            return 0
        with torch.no_grad():
            return int((self._transverse_rho2(x) >= np.pi ** 2).sum())

    def dummy_frame_crossings(self, x: torch.Tensor) -> int:
        """How many bends that ANCHOR A DUMMY FRAME reached ``rho >= pi/2`` in this batch.

        The dummy chart's own singular set, and one `transverse_crossings` cannot see: a
        dummy row's frame angle is ``pi/2 +- rho_c``, collinear at rho_c = pi/2, which is
        inside the (u, v) disc. The box keeps rho under pi/2 by a construction-time check and
        the disc wall engages before it (`rho_wall < pi/2`, asserted), so this should read
        zero.

        A PROBE, NOT A MONITOR: as of 2026-09-26 no trainer, modeller or eval path calls
        this or `transverse_crossings`, so a run that crossed would not say so. What guards a
        run today is the construction-time check and the disc wall; a run that must SHOW
        zero needs these wired into its eval logging first.
        """
        if self._transverse_t is None or not bool(self._tv_dummy_anchor.any()):
            return 0
        with torch.no_grad():
            rho2 = self._transverse_rho2(x)[..., self._tv_dummy_anchor]
            return int((rho2 >= (np.pi / 2) ** 2).sum())

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
        # THE INPUT IS POLAR, THE STATE IS NOT. Every caller of this function -- the prior
        # histograms, the ring-frame correction, prior_diagnostics -- produces (theta, phi),
        # because that is what an angle distribution is fitted in. A transverse row's state
        # slot holds (u, v), so the pair is converted HERE rather than at each call site.
        # Skipping it would land a theta near pi in a u slot: a bend of ~3.1 rad instead of
        # ~0, off the disc entirely, and silent because the shapes are identical.
        #
        # This makes `state_from_dof` the inverse of (dof_from_state THEN chart -> polar),
        # not of `dof_from_state` alone. They agree on every non-transverse row, which is
        # every row of every molecule without a linear centre.
        if self._transverse_t is not None:
            from mxtaltools.conformers.geometry import transverse_from_polar
            tj = torch.as_tensor(np.flatnonzero(self.transverse_angles),
                                 dtype=torch.long, device=th.device)
            tm = torch.as_tensor(self.transverse_partner[self.transverse_angles],
                                 dtype=torch.long, device=ph.device)
            u, v = transverse_from_polar(th.index_select(-1, tj), ph.index_select(-1, tm))
            th = th.index_copy(-1, tj, u)
            ph = ph.index_copy(-1, tm, v)

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
        """phi DoF rows grouped by CENTRAL BOND, HELD rows excluded, leader first.

        Held rows are `held_phi_rows`: the improper rows, plus -- with the stereo lock on --
        the rows about a locked double bond. Everything below about impropers is about the
        first set; the second is excluded so its rows are held at the reference rather than
        led from a rotamer histogram into the other E/Z basin.

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
        # HELD rows, which are the improper rows unless the stereo lock is on (held_phi_rows)
        imp = set(self.held_phi_rows())
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

        Returns ``[(order, bank, extra)]`` where ``order`` is ``[(kind, row), ...]`` in
        exactly the sequence InternalPrior fitted the block in (all r, then all theta, then
        all phi), ``bank`` is the RingBank, RingModes or None, and ``extra`` the rows that
        place a ring atom but name an atom outside the system, less the dihedral that places
        the system's entry atom (see below). Each block's record on ``ring_block_info``
        carries its atoms, ``rotation_rows``, the extra phi rows of its rotation about the
        bond attaching it to the tree's parent side, and ``entry_rows``, that dihedral.

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
        # the SAME root rule this chart was built with, or the two trees differ on every
        # molecule whose default root is an sp carbon and the assertion below fires on them
        m.build_conformer_tree(avoid_sp_root=self.avoid_sp_root)
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
        # the bond list ring_systems itself read, so it is in `sysid`'s numbering by
        # construction; deduplicated so a list carrying both directions cannot double-count
        mol_bonds = {tuple(sorted((int(a), int(b))))
                     for a, b in m.mol_bond_index.detach().cpu().numpy().T}
        # WHY THIS IS RECORDED RATHER THAN RE-DERIVED. Five different things end in
        # `bank = None` below -- aromatic by design, no key resolved, a bank too thin, a
        # polycyclic system refused a topology-blind key, and a stale prior whose keys
        # cannot resolve at all -- and the tuple return cannot tell them apart. A reader
        # that re-derives aromaticity and the lookup to label them is a second copy of this
        # branch, free to drift from it; energies/ring_metrics.py reads this instead. See
        # that module for what each class means.
        self.ring_block_info = []
        # tree parent of every slot, for the attaching-bond rows below (-1 at the root)
        parent = np.full(self.spec.n_atoms, -1, dtype=np.int64)
        parent[bi[:, 1]] = bi[:, 0]
        held_rows = set(self.held_phi_rows())
        out = []
        for s, cols in blocks.items():
            order = ([('r', int(j)) for j in cols['r']]
                     + [('theta', int(j)) for j in cols['theta']]
                     + [('phi', int(j)) for j in cols['phi']])
            in_sys_pre = {int(a) for a in range(len(sysid)) if int(sysid[a]) == int(s)}
            aromatic = bool(self.atom_is_aromatic[list(in_sys_pre)].all())
            # CYCLOMATIC NUMBER of the system -- its count of independent rings, E - V + 1
            # over its own bonds (a system is one connected component by construction, and
            # any bond joining two of its atoms is a ring bond, or it would lie on no cycle
            # and they would not share a system). 1 is a monocycle; 2+ is fused, bridged
            # or spiro. Below 1 the bond list and `sysid` disagree on numbering, and the
            # gate below would then misread every ring -- so that raises.
            n_cycles = sum(1 for a, b in mol_bonds
                           if a in in_sys_pre and b in in_sys_pre) - len(in_sys_pre) + 1
            if n_cycles < 1:
                raise RuntimeError(
                    f"{self.smiles}: ring system {int(s)} has {len(in_sys_pre)} atoms but "
                    f"cyclomatic number {n_cycles}; the bond list and the ring-system ids are "
                    f"not in the same numbering, so the polycyclic gate cannot be evaluated")
            # a fitted MODE SUBSPACE wins over the discrete bank: the rows were isolated
            # islands with near-zero mass between basins, and the saddles live between them
            bank = getattr(prior, 'ring_modes', {}).get((sigs[s], len(order)))
            if bank is None:
                bank = prior.rings.get((sigs[s], len(order)))
            bank_refused = None
            if aromatic:
                # An aromatic ring is RIGID: there is no pucker to sample, so a bank buys
                # nothing and can only do harm. It did: under signature version 1 the key
                # carried element but not degree, so benzene and cyclohexane shared a bank
                # and benzene drew chairs -- median |ring torsion| 47 deg, 75% of draws
                # past 20 deg, with ff_from_reference unable to object (no torsion term,
                # and bond angles stay near 120 deg through a pucker). Holding it planar
                # near the reference is correct by construction and needs no fit.
                bank = None
            elif n_cycles > 1 and bank is not None:
                # A POLYCYCLIC SYSTEM IS HELD, NOT BANKED. The key is (atom count, sorted
                # (element, degree) multiset, block DoF count) and carries NO TOPOLOGY, while
                # every bank in conformer_prior_v2.pt was fitted on a bare MONOCYCLE
                # (build_ring_banks.RING_SMILES). So a fused, bridged or spiro system with
                # the same count and atom types resolves a monocycle's bank: norbornane
                # takes cycloheptane's, CC12CN1C1C(O)C21 piperidine's, spiro[2.4]heptane
                # cycloheptane's. RingModes' kind-sequence check does not catch it -- the
                # sequences agree -- and the draw then writes a seven-ring pucker into a
                # bicycle. Measured at 'full'/mmff, 400 draws: median prior energy 69,359
                # and 20,526 kcal/mol banked against 40.8 and 70.9 held, closure error
                # 1.1-1.8 A against 0.03-0.04. Holding is the known-good fallback.
                #
                # This also refuses a bank fitted on a GENUINE polycycle, deliberately:
                # the key cannot tell decalin from spiro[4.5]decane either, so no bank a
                # polycycle resolves can be trusted until the signature carries ring
                # topology (a ring_sig_version 3 change in MXtalTools' prior.ring_systems).
                # Reported through the existing 'held_unsupported' class, with the reason on
                # `bank_refused`, so energies/ring_metrics.py's four classes stand as they are.
                bank, bank_refused = None, 'polycyclic'
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
            # the bank was fitted against the narrower set -- all but the rotation rows
            # below, and the rows a per-molecule ring shape writes (energies/ring_shapes.py).
            in_sys = {int(a) for a in range(len(sysid)) if int(sysid[a]) == int(s)}
            placing = ([('r', j) for j in range(self.n_r) if int(bi[j, 1]) in in_sys]
                       + [('theta', j) for j in range(self.n_th) if int(ai[j, 2]) in in_sys]
                       + [('phi', j) for j in range(self.n_ph) if int(ti[j, 3]) in in_sys])
            extra = [kj for kj in placing if kj not in set(order)]
            # THE DIHEDRAL THAT PLACES THE SYSTEM'S ENTRY ATOM is not an extra. Where the tree
            # enters the system from outside, the phi row placing the entry atom turns about
            # a bond one further out, (a, b): a change moves the entry atom's whole subtree,
            # the system included, rigidly about a-b and no distance inside it, so it is drawn
            # as an ordinary member of its sibling group -- a rotor, a substituent following
            # another ring's own rows, or a child in another ring's rotation. Held, it froze
            # that rotor at its reference and, where the bond belongs to another ring, kept
            # this system's entry atom at its reference dihedral whatever shape that ring took.
            entry = [j for k, j in extra if k == 'phi' and int(ti[j, 2]) not in in_sys]
            extra = [kj for kj in extra if not (kj[0] == 'phi' and kj[1] in entry)]
            # THE RING'S ROTATION ABOUT THE BOND THAT ATTACHES IT. Where the tree enters the
            # system from outside, at atom c from its parent b, the phi rows whose central
            # bond is (b, c) place c's children on the frame (a, b, c): a displacement common
            # to that bond's sibling group turns c's whole subtree -- the ring and everything
            # placed on frames holding it -- rigidly about b-c, and moves no distance inside
            # the ring. sample_prior_states draws that displacement like any other rotor
            # instead of holding it; the ring rows keep their offsets from one another,
            # which is what closes the ring. A held phi row (an improper, or a locked
            # double bond) is not in a sibling group and is left out.
            rotation = [j for k, j in extra if k == 'phi'
                        and int(ti[j, 2]) in in_sys and int(ti[j, 1]) not in in_sys
                        and int(parent[int(ti[j, 2])]) == int(ti[j, 1])
                        and j not in held_rows]
            self.ring_block_info.append({
                'system': int(s), 'aromatic': aromatic, 'key': (sigs[s], len(order)),
                'ring_class': ('held_aromatic' if aromatic else
                               'banked_modes' if isinstance(bank, RingModes) else
                               'banked_rows' if bank is not None else 'held_unsupported'),
                'n_block_dof': len(order), 'n_extra_dof': len(extra),
                'stale_prior': bool(self.ring_sig_stale),
                # orthogonal to ring_class, like stale_prior: 'held_unsupported' with
                # bank_refused 'polycyclic' means a key DID resolve and was refused for
                # topology, not that no bank exists for this ring
                'n_cycles': int(n_cycles), 'bank_refused': bank_refused,
                # the system's atoms (slot numbering), the phi rows of its rotation about the
                # attaching bond, and the phi row placing its entry atom (both empty when the
                # tree's root is in the system)
                'atoms': sorted(in_sys), 'rotation_rows': rotation, 'entry_rows': entry,
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

        NO TRANSVERSE ROWS. This returns a density over the POLAR dof, while the state those
        dof map to is in (u, v) on a transverse row -- so the two differ by the chart
        Jacobian ``|d(theta, phi)/d(u, v)| = 1 / rho`` per such row, and an importance weight
        formed from them would be wrong by that factor. Refused rather than approximated: the
        whole point of this function is that an unmatched density gives an ESS for a
        distribution nobody sampled. Fitting the bend in (u, v) directly -- a 2-D
        distribution on a disc, not a 1-D angle histogram -- is the real fix and is a
        modelling question, not plumbing.

        ACYCLIC ONLY. Ring blocks draw from a bank or a pucker subspace whose density is a
        mixture over fitted rows, and the subspace is lower-dimensional than the block it
        fills, so the block's density is singular in the held directions. That is a real
        derivation, not an oversight -- it raises rather than returning a number that
        would look usable.

        The BOX CLAMP in sample_prior_states is not represented here either: it puts
        finite mass exactly on the wall, which no continuous density can express. Callers
        must check ``stats['clip_frac']`` is ~0 before treating these weights as valid.

        FREE INVERTIBLE CENTRES ARE A MIXTURE. With ``joint_torsions`` the draw reflects each
        centre of energies/invertible_centres.py's `invertible_centres`, planar ones included,
        on an independent half of its draws, so that centre's rows are scored as
        ``(q_c(x) + q_c(R_c x)) / 2``; every other row keeps the term it had before, bitwise.
        At a planar centre the two components nearly coincide and the sum is exact all the
        same. A FRAME centre, or a SIBLING pivot that is not its group's leader, raises
        NotImplementedError below; the table makes either only at a ring atom (a FRAME needs a
        ring bond to its ring-closure neighbour, a pivot other than the leader is a ring
        child), and a molecule with a ring is refused above first, so no acyclic molecule
        reaches that refusal.
        """
        from mxtaltools.conformers.prior import R_RANGE, THETA_RANGE, PHI_RANGE
        spans = {'r': R_RANGE, 'theta': THETA_RANGE, 'phi': PHI_RANGE}

        if self._transverse_t is not None:
            raise NotImplementedError(
                f'prior_log_prob is a density over the POLAR dof, but {self.smiles} has '
                f'{int(self.transverse_angles.sum())} transverse bend row(s) whose state is '
                f'(u, v). The two differ by |d(theta, phi)/d(u, v)| = 1/rho per row, so an '
                f'importance weight built from them would be wrong by that factor -- which '
                f'is exactly the "ESS for a distribution nobody sampled" this function was '
                f'written to prevent. Fit the bend as a 2-D distribution on the disc first.')

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
            from energies.invertible_centres import (ROOT, SIBLING, invertible_centres,
                                                     reflect_phi)
            s_imp = self.improper_phi_sigma(float(self.temperature))
            # the rows of a FREE INVERTIBLE CENTRE are scored at the end, as one mixture;
            # `comp` keeps each one's own component (mean per draw, width) until then
            inv = invertible_centres(self)
            mixed = {j for c in inv for j in c.rows}
            comp = {}
            # the HELD rows, as sample_prior_states draws them (held_phi_rows)
            for j in self.held_phi_rows():
                if j in mixed:
                    comp[j] = (np.full(n, ph0[j]), s_imp)
                    continue
                total += wrapped_gauss(dof[:, n_phi0 + j], ph0[j], s_imp)
            leaders = set()
            for gi, rows_j in enumerate(groups):
                grows = [n_phi0 + j for j in rows_j]
                lead_i = 0
                leaders.add(rows_j[lead_i])
                _, hist, _, _ = types[grows[lead_i]]
                total += marginal(grows[lead_i], 'phi', hist)
                disp = ((dof[:, grows[lead_i]] - ph0[rows_j[lead_i]] + np.pi)
                        % (2 * np.pi) - np.pi)
                for i, gr in enumerate(grows):
                    if i == lead_i:
                        continue
                    if rows_j[i] in mixed:
                        comp[rows_j[i]] = (ph0[rows_j[i]] + disp, g_sigma[gi])
                        continue
                    total += wrapped_gauss(dof[:, gr], ph0[rows_j[i]] + disp, g_sigma[gi])

            # FREE INVERTIBLE CENTRES. sample_prior_states flips each one (R_c, a ROOT or SIBLING
            # flip of energies/invertible_centres.py: a FRAME flip needs a ring, refused above)
            # on an independent half of its draws. R_c is an involution with |det| 1 that moves
            # only the centre's own rows, and no other row's density reads them -- a pivot is its
            # group's leader, whose value R_c keeps and every follower's mean is built from --
            # so the density of those rows is (q_c(x) + q_c(R_c x)) / 2, each component the
            # product of the rows' own wrapped Gaussians. Differences are wrapped onto the
            # circle before the images are summed: a reflected sibling is not wrapped.
            def circ_gauss(d, s):
                d = (d + np.pi) % (2 * np.pi) - np.pi
                acc = np.zeros_like(d)
                for k in (-1, 0, 1):
                    acc = acc + np.exp(gauss(d + 2 * np.pi * k, 0.0, s))
                return np.log(np.clip(acc, 1e-300, None))

            ph = dof[:, n_phi0:]
            for c in inv:
                if (c.kind not in (ROOT, SIBLING) or (c.kind == SIBLING and c.pivot not in leaders)
                        or set(c.rows) - set(comp)):
                    raise NotImplementedError(
                        f'{self.smiles}: the invertible centre {c.name} ({c.kind}) flips about '
                        f'a ring frame or a pivot that is not its group\'s drawn leader, or '
                        f'moves a row this density does not draw as a held row or a follower; '
                        f'the mixture below assumes neither (energies/invertible_centres.py)')
                mirror = reflect_phi(ph.copy(), c)
                own = sum(circ_gauss(ph[:, j] - comp[j][0], comp[j][1]) for j in c.rows)
                mir = sum(circ_gauss(mirror[:, j] - comp[j][0], comp[j][1]) for j in c.rows)
                total += np.logaddexp(own, mir) - np.log(2.0)
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
                            joint_rings: bool = True, ring_shapes=None):
        """``[n, d]`` states drawn from a fitted InternalPrior. Returns ``(x, stats)``.

        Per-DoF marginals for the acyclic part, joint draws where a product of marginals
        cannot work: sibling torsions about a shared bond, and RING SYSTEMS.

        ``joint_rings`` DEFAULTS TO TRUE and is the real ring path -- each ring block is
        drawn from its fitted pucker subspace or discrete bank, aromatic rings are held
        planar by design, an unsupported ring is held at a fraction of thermal width, and
        the ring-positioning DoF outside the block are held, except the ring's rotation
        about the bond attaching it (below); the dihedral placing a system's entry atom is
        not among them and is drawn with its sibling group like any other row (see
        ``ring_blocks``). See ``ring_blocks`` for the four classes and energies/ring_metrics.py
        for how they are reported.

        ``ring_shapes`` (None, the default, draws as above) is energies/ring_shapes.py's
        per-block list for THIS molecule, aligned with ``ring_blocks``. A block with stored
        shapes takes one of them per draw, uniformly and independently per block and per
        draw: the shape's theta and phi rows plus jitter at ``ring_jitter_scale`` times the
        thermal width, as the hold jitters, and r held about the reference as the bank path
        holds it. Its other rows, and every block without shapes, are drawn as above.
        ``stats['ring_shapes']`` gives, per block, the shapes available, how many draws
        took each, and each draw's shape (``pick``, -1 without shapes).

        THE RING'S ROTATION ABOUT ITS ATTACHING BOND. With ``joint_torsions``, where the tree
        enters a ring system from outside, the phi rows about the entering bond
        (``ring_block_info``'s ``rotation_rows``) are not held: their sibling group takes one
        displacement per draw, its leader drawn from the central bond's marginal like any
        other rotor, and the ring rows keep their offsets from one another, so the ring is
        turned rigidly and stays closed. ``stats['n_ring_rotations']`` counts those groups.

        ``joint_rings=False`` IS A NEGATIVE CONTROL, NOT A SAMPLING MODE. Every ring DoF
        then gets an independent marginal, which violates closure by construction: measured
        on cyclohexane at 'full', closure error goes from 0.086 A (2.2 bond-sigma) to
        2.93 A (75 bond-sigma) and the median potential rises by two orders of magnitude.
        The draws remain valid support, which is all TB strictly needs, but as a proposal
        they are broken -- so a benchmark quoting this path is measuring the disabled path,
        not the prior. ``stats['closure_err']`` is measured on BOTH, deliberately.

        BOTH SIDES OF A FREE INVERTIBLE CENTRE. With ``joint_torsions`` each centre
        energies/invertible_centres.py's `invertible_centres` names -- three-coordinate, left
        free by the stereo lock, with a substituent offset whose sign can be negated as an
        exact inversion without turning the centre's ring system, planar or not -- has that
        offset negated on an independent half of the draws, after every other draw. At a
        planar centre (an sp2 C, an aromatic ring atom) the flip moves a drawn row by twice its
        distance from the plane. A free four-coordinate centre (``stereo_coeff`` 0) is not
        flipped. ``stats['invertible_centres']`` names them, ``stats['reflected']`` ``[n, m]``
        marks the flipped draws and ``stats['reflected_frac']`` gives each centre's share. A
        molecule without one (CH4, ethanol) draws exactly as before, bitwise, generator state
        included; one whose only such centres are planar (H2CO, benzene) draws differently bit
        for bit, and in distribution only by its reference's own departure from the plane
        (energies/invertible_centres.py, BITWISE WHERE NOTHING IS FLIPPED).
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
        # phi row -> [n] value its ring's rotation about the attaching bond displaces: the
        # reference, or the drawn shape's value. Drawn with the sibling groups below.
        # `rot_ring` holds the rows that place a ring atom (ring_block_info's rotation_rows).
        rot_base, rot_ring = {}, set()
        stats.update(n_rings=0, n_ring_banked=0, n_ring_thermal=0, n_ring_extra_held=0,
                     n_ring_remapped=0, n_ring_shaped=0, n_ring_rotation_rows=0,
                     n_ring_rotations=0, ring_shapes=[])
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

            blocks = self.ring_blocks(prior)
            if ring_shapes is not None and len(ring_shapes) != len(blocks):
                raise ValueError(
                    f'{self.smiles}: ring_shapes has {len(ring_shapes)} entries but ring_blocks '
                    f'gives {len(blocks)} blocks; the shapes were built for another chart')
            for bk, ((order, bank, extra), binfo) in enumerate(zip(blocks,
                                                                  self.ring_block_info)):
                rows = [self._global_row(k, j) for k, j in order]
                stats['n_rings'] += 1
                # the rotation is drawn with the sibling groups, which only joint_torsions has
                rotation = set(binfo['rotation_rows']) if joint_torsions else set()
                rot_ring |= rotation
                stats['n_ring_rotation_rows'] += len(rotation)
                shp = None if ring_shapes is None else ring_shapes[bk]
                n_avail = 0 if shp is None else int(len(shp.values))
                # per block: shapes available, draws per shape, and each draw's shape (-1: none)
                rec = {'block': bk, 'available': n_avail, 'drawn': np.zeros(n_avail, np.int64),
                       'pick': np.full(n, -1, np.int64)}
                if n_avail:
                    # PER-MOLECULE RING SHAPES (energies/ring_shapes.py): this molecule's own
                    # relaxed ring conformers, measured in this chart. A shape is written into
                    # every row it carries -- the block, and the extras that set the ring's
                    # internal geometry, without which a ring the tree enters from outside is
                    # not determined by its block -- except r, which stays with the thermal
                    # path as the bank's does, and the rows of the rotation's sibling group,
                    # whose values are the base the drawn rotation displaces.
                    srows = [(str(k), int(j)) for k, j in shp.rows]
                    rgroup = {('phi', j) for g in groups
                              if set(g) & set(binfo['rotation_rows']) for j in g}
                    own = set(order) | set(extra) | rgroup
                    miss = (set(order) | rgroup) - set(srows)
                    if not set(srows) <= own or miss:
                        raise ValueError(
                            f'{self.smiles}: ring block {bk}\'s shapes carry rows '
                            f'{sorted(set(srows) - own)} outside the block and lack '
                            f'{sorted(miss)}; they were measured in another chart')
                    pick = rng.integers(n_avail, size=n)
                    rec['drawn'], rec['pick'] = np.bincount(pick, minlength=n_avail), pick
                    vals = np.asarray(shp.values, dtype=np.float64)[pick]
                    for col, (kind, j) in enumerate(srows):
                        if kind == 'r':
                            continue
                        if rotation and (kind, j) in rgroup:
                            rot_base[j] = vals[:, col]
                            continue
                        gr = self._global_row(kind, j)
                        s = (sig[kind][j] if kind in sig else phi_sig) * sc
                        v = vals[:, col] + rng.normal(0.0, s, n)
                        dof[:, gr] = ((v + np.pi) % (2 * np.pi) - np.pi) if kind == 'phi' else v
                        ring_rows.add(gr)
                    for kj in order:
                        if self._global_row(*kj) not in ring_rows:
                            ring_rows.add(hold(kj))
                    stats['n_ring_shaped'] += 1
                elif isinstance(bank, RingModes):
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
                # not, letting them float re-opens the ring (see ring_blocks). Not the
                # rotation rows, which the sibling groups draw, nor a row a shape wrote.
                for kj in extra:
                    if kj[0] == 'phi' and kj[1] in rotation:
                        rot_base.setdefault(kj[1], np.full(n, ph0[kj[1]]))
                        continue
                    if self._global_row(*kj) not in ring_rows:
                        ring_rows.add(hold(kj))
                        stats['n_ring_extra_held'] += 1
                stats['ring_shapes'].append(rec)

        # ---- r / theta ----
        if thermal_rtheta:
            for j in range(self.n_r):
                if j not in ring_rows:
                    dof[:, j] = rng.normal(r0[j], s_r[j], n)
            for j in range(self.n_th):
                if self.n_r + j not in ring_rows:
                    dof[:, self.n_r + j] = rng.normal(th0[j], s_th[j], n)
            stats['rtheta_sigma_deg'] = (float(np.degrees(s_th.mean())), float(s_r.mean()))
        # the rotation rows are drawn jointly with their sibling group, not from a marginal
        rot_rows = {n_phi0 + j for j in rot_base}
        for row, (kind, hist, key, is_ring) in enumerate(types):
            stats['n_ring_marginal'] += int(is_ring and row not in ring_rows
                                            and row not in rot_rows)
            if row in ring_rows:
                continue
            if (joint_torsions and row >= n_phi0) or (thermal_rtheta and row < n_phi0):
                continue
            dof[:, row] = draw(row, kind, hist)

        # ---- phi ----
        # DUMMY-FRAME ROWS draw from their central-bond marginal like any other row, but
        # InternalPrior.fit measured it with no dummy (mol.internal_dof on the default tree), so
        # on a bond to a linear centre it was fitted against a collinear frame and its shape
        # carries no signal. MEASURED BENIGN, 2026-09-26, six alkynes at 'full', mmff,
        # conformer_prior_v2.pt, 2000 draws each: replacing every dummy group's fitted twist by
        # a uniform RIGID rotation of the group (a transverse dummy row: its bend direction by a
        # uniform angle) moved the median potential above the reference by <= 0.06 kcal/mol and
        # its 95th percentile by <= 0.8; a rigid twist of a dummy group about its axis spans
        # 3e-5 to 0.86 kcal/mol of MMFF94. The twist across an alkyne is nearly free, so a flat
        # marginal is close to right. A fit through the chart's masks is what would give it
        # signal; see docs/wiki/conformer-force-field-and-prior.md.
        if joint_torsions:
            # improper rows FIRST: they are angles at the parent, not rotations, so they
            # rattle thermally about the reference instead of taking a rotamer histogram. With
            # the stereo lock on, rows about a locked double bond join them (held_phi_rows).
            imp = [j for j in self.held_phi_rows() if n_phi0 + j not in ring_rows]
            s_imp = self.improper_phi_sigma(float(self.temperature))
            _true_imp = set(self.improper_phi_rows())
            stats['n_improper'] = sum(1 for j in imp if j in _true_imp)
            stats['n_held_bond'] = len(imp) - stats['n_improper']
            for j in imp:
                dof[:, n_phi0 + j] = ph0[j] + rng.normal(0.0, s_imp, n)
            stats['n_groups'] = len(groups)
            stats['sigma_deg'] = ((float(np.degrees(min(g_sigma))),
                                   float(np.degrees(max(g_sigma)))) if g_sigma else (0.0, 0.0))
            s_rot = phi_sig * self.ring_jitter_scale      # the hold's own phi jitter
            for gi, rows_j in enumerate(groups):
                grows = [n_phi0 + j for j in rows_j]
                if any(j in rot_ring for j in rows_j):
                    # A RING'S ROTATION ABOUT THE BOND ATTACHING IT (ring_blocks). The leader,
                    # the group's first row as for any rotor, is drawn from the central bond's
                    # marginal and every row takes its displacement. The ring rows keep their
                    # offsets from one another -- the reference's, or the drawn shape's -- at
                    # the hold's jitter, so the ring turns rigidly and stays closed. The
                    # entry atom's other children, another system's entry atom among them, keep
                    # their offsets from the ring rows -- the reference's, or the shape's, which
                    # carries this group whole -- at the sibling jitter.
                    base = [rot_base[j] if j in rot_base else ph0[j] for j in rows_j]
                    _, hist, _, _ = types[grows[0]]
                    dof[:, grows[0]] = draw(grows[0], 'phi', hist)
                    disp = (dof[:, grows[0]] - base[0] + np.pi) % (2 * np.pi) - np.pi
                    for i, (j, gr) in enumerate(zip(rows_j, grows)):
                        if i == 0:
                            continue
                        jit = rng.normal(0.0, s_rot if j in rot_ring else g_sigma[gi], n)
                        dof[:, gr] = base[i] + disp + jit
                    stats['n_ring_rotations'] += 1
                    continue
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
            # the rotation rows place ring atoms too: their groups are the rotation's, above
            frame_groups = self.ring_frame_groups(ring_rows | rot_rows) if joint_rings else []
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

        # ---- FREE INVERTIBLE CENTRES: each on either side, with probability 1/2 ----
        # The joint draw above holds every improper row and every substituent offset about the
        # reference, sign included, so it proposes one side of each non-planar centre. Where
        # the lock leaves a three-coordinate centre free the target holds both, and each one
        # energies/invertible_centres.py qualifies, planar or not, has that offset negated on
        # its own half of the draws; the eval's parity metric reads the same table and requires
        # both sides of the non-planar ones. The coins come from `rng` AFTER every other draw,
        # and a molecule with no such centre makes no call on it: its draw, and the generator's
        # state after it, are bitwise what they were without this block. Only periodic phi rows
        # move, so the box clamp below cannot bind on a flip, and every other atom moves
        # rigidly, so no stereo element changes its indicator but a double bond the centre is an
        # atom of, which the flip turns by twice the centre's distance from its plane
        # (energies/invertible_centres.py, QUALIFIED). prior_log_prob scores the two-component
        # mixture this makes.
        stats['invertible_centres'] = []
        stats['reflected'] = np.zeros((n, 0), dtype=bool)
        if joint_torsions:
            from energies.invertible_centres import (FRAME, frame_dihedral,
                                                     invertible_centres, reflect_phi)
            inv = invertible_centres(self)
            if inv:
                flip = rng.random((n, len(inv))) < 0.5
                pos = None
                if any(c.kind == FRAME for c in inv):
                    # phi_ring of each draw, from its positions before any flip: no flip moves
                    # the ring atoms it is measured on (energies/invertible_centres.py)
                    tt = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)
                    pos = self.build_positions(
                        self.state_from_dof(tt(dof[:, :self.n_r]), tt(dof[:, self.n_r:n_phi0]),
                                            tt(dof[:, n_phi0:])).clamp(-1.0, 1.0)
                    ).reshape(n, -1, 3)
                for k, c in enumerate(inv):
                    reflect_phi(dof[:, n_phi0:], c, flip[:, k],
                                ring=frame_dihedral(pos, c) if c.kind == FRAME else None)
                stats['invertible_centres'] = [c.name for c in inv]
                stats['reflected'] = flip
        stats['reflected_frac'] = [float(f) for f in stats['reflected'].mean(0)] if n else []

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
            # transverse columns are non-periodic and walled like r/theta, so a draw can
            # leave their box too. Reported separately rather than folded into 'theta': the
            # box means a different thing there (a disc radius, not an angle range), and a
            # clip rate that mixed the two would not say which box was too narrow.
            'transverse': float(outside[:, self._free_block == 3].to(self.dtype).mean()) if (self._free_block == 3).any() else 0.0,
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
            if stats['invertible_centres']:
                print("  free invertible centres reflected (share of draws): " + ', '.join(
                    f'{nm} {fr:.1%}' for nm, fr in zip(stats['invertible_centres'],
                                                      stats['reflected_frac'])))
            print(f"  clipped to box: r {stats['clip_frac']['r']:.1%}, "
                  f"theta {stats['clip_frac']['theta']:.1%}")
            if stats['n_rings']:
                print(f"  rings: {stats['n_rings']} system(s) -- {stats['n_ring_shaped']} "
                      f"from this molecule's own ring shapes (shapes available per block: "
                      f"{[r['available'] for r in stats['ring_shapes']]}), "
                      f"{stats['n_ring_banked']} "
                      f"from a fitted RingBank (joint, samples pucker), "
                      f"{stats['n_ring_thermal']} held at thermal jitter about the "
                      f"reference (closure preserved, pucker NOT sampled; aromatic rings "
                      f"take this path by design, being rigid); {stats['n_ring_rotations']} "
                      f"ring(s) turned about the bond attaching them")
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
        """Bonded + LJ (+ stereo lock), clipped, + box wall. NO change of measure, NOT / T.

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
            if self.stereo_coeff > 0.0 and self.stereo.n:
                # THE STEREO LOCK, a potential in kcal/mol, INSIDE the clip below: the soft
                # clip is the identity wherever U + P is under the cap, which covers the
                # locked basins and their edges, and above it the owner's cap then bounds the
                # total rather than being defeated by a term added after it. NOT multiplied by
                # T: it is divided by T with U, so the baked value (T = 1, divided on the read
                # side) and a live row agree at every temperature. Skipped, not added as zero,
                # when off, so the unlocked path stays bitwise.
                e = e + self.stereo.lock_energy(pos.reshape(x.shape[0], -1, 3),
                                                self.stereo_coeff)
            if self.energy_clip is not None:
                # Above the cutoff, U -> cutoff + log1p(U - cutoff): monotone, smooth,
                # IDENTITY BELOW IT, so nothing a physical conformer reaches is deformed.
                # A clash at 10,000 kcal/mol becomes ~109 at cutoff 100, which is the whole
                # point -- log Z's TB fixed point is a MEAN of log w, so an unbounded left
                # tail on log_reward drags it without limit.
                #
                # THE FORCE FIELD (AND THE STEREO LOCK) ONLY, and BEFORE the wall. The lock is
                # part of the target, not a domain guarantee. The box wall is the domain
                # guarantee, not part of the potential being tempered; compressing it would
                # make a far off-domain excursion CHEAPER than the clamp intends, and the
                # sampler would be free to leave the domain. The measure terms (BAT volume
                # element, log_chart_jacobian) are added later in energy(), so they are
                # untouched here for the same reason.
                from mxtaltools.common.utils import log_rescale_positive
                e = log_rescale_positive(e, self.energy_clip)
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
        tree, _ = self._batch(x.shape[0])
        r, th, ph = self.dof_from_state(x)
        return -temperature * self._log_jac(tree, r, th, ph, x.shape[0])

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

        The condition carries log-temperature (when `temperature_conditioning`) and the
        pre-encoded molecular embedding (when `embedding_conditioning`), in that order. With
        neither, it is a single zero column.

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

        # A SET NEEDS mol_id. Without it condition_id falls back to zeros, which is right
        # for the single-molecule route (library size 1, one condition) and silently wrong
        # on a set: every row lands on condition 0, so the tracker, the per-condition log Z
        # and prior-row expiry all book it under the first molecule. Checked FIRST, before
        # the temperature draw consumes the RNG, so a refused call changes nothing.
        mol_id = getattr(mol_batch, 'mol_id', None)
        if mol_id is None and int(self.condition_library_size) > 1:
            raise RuntimeError(
                f"condition_samples got a batch with no `mol_id` on a "
                f"{self.condition_library_size}-condition set. condition_id would default "
                f"every row to condition 0 -- the first molecule -- and nothing downstream "
                f"could tell. Attach mol_id from the identifier registry (init_identifiers) "
                f"before conditioning.")

        conds = []
        if self.temperature_conditioning:
            if temperature is not None:
                log_T = torch.log10(torch.as_tensor(temperature, device=dev)).flatten()
            else:
                lo, hi = self.log_temperature_range
                u = torch.rand(n_groups, device=dev)
                log_T = (lo + u * (hi - lo)).repeat_interleave(max(repeats, 1))
            conds.append(log_T.reshape(-1, 1).float())
        else:
            log_T = torch.full((n,), float(self.log_temperature), device=dev)

        # MOLECULAR IDENTITY. Until this existed the condition vector was log-temperature or
        # a zeros column, and `mol_id` reached only `condition_id`, which the policy never
        # reads -- so the policy was MOLECULE-BLIND BY CONSTRUCTION and no encoder, however
        # good, could have changed that. The embedding is baked offline by
        # models/encoder_cache.py onto each conditions-file entry; same attribute name and
        # same width contract as MolecularCrystal's, so `train.get_conditioning_dim` and the
        # config invariants need no conformer-specific case.
        if self.embedding_conditioning:
            emb = getattr(mol_batch, 'embedding', None)
            if emb is None:
                raise RuntimeError(
                    "embedding_conditioning is enabled but this batch has no `embedding` -- "
                    "the conditions file was not built with one. Bake it with: "
                    "python build_conformer_conditions.py --smiles ... --encoder-ckpt "
                    "<ckpt> --out conformer_conditions.pt")
            emb = emb.to(dev).reshape(mol_batch.num_graphs, -1)
            if emb.shape[-1] != self.embedding_conditioning_dim:
                raise RuntimeError(
                    f"embedding width {emb.shape[-1]} != embedding_conditioning_dim "
                    f"{self.embedding_conditioning_dim}; the conditioner was built for a "
                    f"different encoder than the one that baked this file")
            conds.append(emb.float())

        if not conds:
            # the crystal's no-conditioning branch: a single zero column, so the conditioner
            # sees a well-shaped tensor rather than an empty one
            condition = torch.zeros((n, 1), device=dev)
        else:
            condition = torch.cat(conds, dim=-1)

        mol_id = (torch.zeros(n, dtype=torch.long, device=dev) if mol_id is None
                  else mol_id.to(dev))
        condition_id = mol_id * (self.n_sg * self.n_zp)   # n_sg = n_zp = 1

        mol_batch.conditions = condition.detach()
        mol_batch.condition_id = condition_id
        return mol_batch, log_T.flatten(), condition, condition_id

    def prebuilt_sample_to_reward(self, mols, temperature, raw_latents=None):
        """log reward for samples whose energy is already attached to the graphs.

        The crystal version re-scores from stored energy terms; the conformer equivalent
        needs a ``conformer_energy`` graph attribute, written by whatever prepared the
        buffer. Raising rather than silently rescoring is deliberate: a silent recompute
        here would hide a prep bug behind plausible numbers.
        """
        if raw_latents is not None:
            # the trainer's replay re-score passes raw_latents on every route; only the crystal
            # energy scores a bounding term from them, and a conformer state has none
            raise ValueError('the conformer energy has no raw latents; got a non-None raw_latents')
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
            state = torch.as_tensor(batch_states(mols), dtype=self.dtype,
                                    device=self.device)
            r, th, ph = self.dof_from_state(state)
            tree, _ = self._batch(state.shape[0])
            log_j = self._log_jac(tree, r, th, ph, state.shape[0]).to(e.device)
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
