import copy
import gc
from argparse import Namespace
from time import sleep
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from energy_sampling.energies.base_set import BaseSet
from mxtaltools.common.geometry_utils import lat2sph_rotvec
from mxtaltools.common.utils import log_rescale_positive, is_cuda_oom
from mxtaltools.constants.space_group_feature_tensor import SG_FEATURE_TENSOR
from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import COMPUTES_REQUIRE_CLUSTER
from mxtaltools.dataset_utils.data_classes import MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model, drain_mace_phase_timing
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor


def density_penalty(packing_coeff, turnover = 1):
    """
    draw crystals into the physically reasonable region
    :param packing_coeff:
    :return:
    """
    cp = packing_coeff.clamp_min(1e-6)          # numerical guard only, NOT a gradient cap
    u = F.relu(np.log(0.55) - torch.log(cp))    # >0 when too loose; 0 in the physical region
    # quadratic onset, linear tail: C1 at u == turnover
    loose = torch.where(u <= turnover, u ** 2,
                        turnover ** 2 + 2 * turnover * (u - turnover))
    return loose + F.relu(cp - 0.95) ** 2

def soften_high(energy, turnover_pot, coeff, clip: Optional[float] = None):
    # soften the repulsion
    softened_energy = energy.clone()
    high_bools = softened_energy > turnover_pot
    delta = softened_energy[high_bools] - turnover_pot
    softened_energy[high_bools] = turnover_pot + delta ** coeff
    if clip is not None:
        softened_energy = softened_energy.clip(max=clip)

    return softened_energy


class MolecularCrystal(BaseSet):
    def __init__(self, device,
                 energy_function: str,
                 max_temperature: float = 10,
                 min_temperature: float = 0.01,
                 density_coeff: float = 0,
                 temperature: float = 1.0,
                 temperature_conditioning: bool = False,
                 lj_coeff: float = 1.0,
                 sg_conditioning: bool = False,
                 zp_conditioning: bool = False,
                 vector_conditioning: bool = False,
                 vector_conditioning_dim: Optional[int] = None,
                 embedding_conditioning: bool = False,
                 embedding_conditioning_dim: Optional[int] = None,
                 space_groups: Optional[list] = [2],
                 bounding_coeff: float = 1.0,
                 reduction_coeff: float = 1.0,
                 max_z_prime: Optional[int] = None,  # defaults to max(z_primes); see below
                 z_primes: Tuple[int] = (1,),
                 mlip_path: Optional[str] = None,
                 reward_range: float = None,
                 lj_rescale: float = None,
                 pressure: float = 1,  # in atm
                 log_temperature_range: list = None,
                 analyze_kwargs: Optional[dict] = None,  # extra kwargs passed through to crystal_batch.analyze()
                 internal_oom_recovery: bool = True,  # if False, skip the adaptive sub-batching/OOM catch-and-shrink loop in batched_analyze_crystal_batch and analyze the whole batch in one call, letting any OOM propagate to the caller
                 host_gas_phase_reference: bool = True,  # uma/mace only: compute the isolated-molecule leg ONCE per molecule and carry it, instead of recomputing it every energy call. See attach_gas_phase_reference
                 ):

        super(MolecularCrystal, self).__init__()
        self.device = device
        # max_z_prime is the STATE-LAYOUT width (6 box params + 6 per z'), so it is not a
        # free knob -- it must cover the largest z' this problem can present. It used to
        # default to 1 while train.py's init_energy_function passed only `z_primes`, which
        # is inert at z'=1 (both are 1) and silently wrong above it: prod0810's two Z'=2
        # arms built a 12-dim energy while _build_gfn_config sized the policy from
        # max(z_primes)=2, and died at startup in GFN.get_periodic_dimensions. Derive it
        # from z_primes so the two can't drift again; an explicit value is still honoured
        # (a zp-conditioned run may want headroom) but may not be SMALLER than the z' it
        # will be asked to represent.
        derived_max_zp = max(int(zp) for zp in z_primes)
        if max_z_prime is None:
            max_z_prime = derived_max_zp
        elif max_z_prime < derived_max_zp:
            raise ValueError(f"max_z_prime={max_z_prime} cannot represent z_primes={list(z_primes)}; "
                             f"the state layout needs at least {derived_max_zp}")
        self.data_ndim = 6 + 6 * max_z_prime
        self.energy_function = energy_function
        self.SG_FEATURE_TENSOR = SG_FEATURE_TENSOR.clone()  # store space group information

        self.density_coeff = density_coeff
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.temperature_conditioning = temperature_conditioning
        self.lj_coeff = lj_coeff
        self.bounding_coeff = bounding_coeff
        self.reduction_coeff = reduction_coeff
        self.sg_conditioning = sg_conditioning
        self.space_groups = space_groups
        self.max_z_prime = max_z_prime
        self.z_primes = z_primes
        self.zp_conditioning = zp_conditioning
        self.vector_conditioning = vector_conditioning
        if self.vector_conditioning:
            assert vector_conditioning_dim is not None, \
                "vector_conditioning requires vector_conditioning_dim to be set (the dimensionality of the `c` " \
                "vectors carried by molecules_path/prior_path entries -- e.g. mxtaltools' cond_dim for " \
                "latent_multiharmonic, or data_ndim for latent_harmonic)"
        self.vector_conditioning_dim = vector_conditioning_dim
        # Pre-embedded molecule conditioning: each molecules_path/prior_path entry carries
        # an `embedding` baked by a FROZEN encoder (build_qm9_conditions.py), appended to
        # the condition tensor and read by the same scalarMLP conditioner vector_conditioning
        # uses (conditions_type stays 'vector'). Deliberately separate from
        # vector_conditioning: `c` is a TARGET parameter that also goes into analyze() as the
        # energy's condition, whereas an embedding is only ever an INPUT to the policy -- the
        # energy of a crystal does not depend on how the molecule was encoded. Also distinct
        # from molecule_conditioning, which runs a GNN over the graph inside the training
        # loop; here the encoder ran once, offline, so there is no per-step cost and no
        # gradient into it.
        self.embedding_conditioning = embedding_conditioning
        if self.embedding_conditioning:
            assert embedding_conditioning_dim is not None, \
                "embedding_conditioning requires embedding_conditioning_dim (the FLATTENED " \
                "width of the `embedding` on molecules_path/prior_path entries -- 3 * the " \
                "encoder bottleneck, e.g. 192 for Mo3ENet's 64)"
        self.embedding_conditioning_dim = embedding_conditioning_dim
        self.reward_range = reward_range
        self.lj_rescale = lj_rescale
        self.pressure = pressure
        self.log_temperature_range = log_temperature_range
        self.internal_oom_recovery = internal_oom_recovery
        if isinstance(analyze_kwargs, Namespace):  # yaml configs nest dicts as Namespaces
            analyze_kwargs = vars(analyze_kwargs)
        self.analyze_kwargs = analyze_kwargs or {}
        self.host_gas_phase_reference = bool(host_gas_phase_reference)
        # mol_id -> isolated-molecule MLIP energy, filled lazily on first sight of
        # each molecule. Deliberately NOT checkpointed: it is a pure function of the
        # molecule set and the MLIP weights, so a resume refills it in one call
        # rather than risking a stale value riding across a model or dataset change.
        self._gas_pot_cache = {}
        if self.energy_function == 'uma':
            self.mlip_path = mlip_path
            self.predictor = init_uma_crystal_predictor(mlip_path, device=self.device)
        elif self.energy_function == 'mace':
            self.mlip_path = mlip_path
            self.predictor = load_mace_model(self.mlip_path, self.device, torch.float32)
        else:
            self.predictor = None

        self.temperature = temperature  # for static temperature work
        self.energy_clip = None
        self.reward_clip = None

        # TWO independent facts, previously conflated into one:
        #
        #   is_crystal     -- the state IS a crystal parameterization, so the crystal
        #                     latent layout applies: periodic angle dims, the jacobian
        #                     correction, and (D33) dead-row resolution.
        #   latent_energy  -- the reward is a cheap analytic function OF THE LATENT, so
        #                     there is no packing coefficient, no pressure term and no
        #                     physical mol_energy to read.
        #
        # `latent_gaussian` is the combination that did not previously exist: a real
        # crystal parameterization scored by an analytic gaussian. That is what lets a
        # toy measure the dead-row machinery against a CLOSED-FORM log Z, which no
        # physical energy can provide. latent_harmonic/latent_multiharmonic keep
        # is_crystal False exactly as before, so every existing toy config is untouched.
        self.latent_energy = self.energy_function in ['latent_harmonic',
                                                      'latent_multiharmonic',
                                                      'latent_gaussian']
        self.is_crystal = not self.energy_function in ['latent_harmonic', 'latent_multiharmonic']  # not a toy model

        self.batch = collate_data_list([MolCrystalData(max_z_prime=max_z_prime)], max_z_prime=max_z_prime)

        self.sg_cache = {}
        for sg in range(1, 230):
            self.sg_cache[sg] = np.stack(SYM_OPS[int(sg)])

        self.computes = ['reduction_en']
        self.computes.append(self.energy_function)
        self.computes_require_cluster = any(COMPUTES_REQUIRE_CLUSTER.get(k, False) for k in self.computes)

        self._init_condition_library()

    def _init_condition_library(self):
        """
        Lookup tables for the per-sample, immutable `condition_id` computed
        in condition_samples(): a mixed-radix combination of molecule index
        (mol_id -- a dense registry built from crystal_batch.identifier,
        spanning both mol_dataset and prior_dataset, in train.py's
        init_identifiers()) and dense-local SG/Z' index. condition_id keys
        ConditionLogZTracker only -- it never touches the (potentially
        large) condition vector itself, so its cost doesn't depend on
        conditions_dim.

        n_molecules defaults to 1 here (library sized as if there's a single
        molecule) and is corrected via set_n_molecules() once the identifier
        registry is built in train.py -- MolecularCrystal is constructed
        before that, so it can't know the true count upfront.
        """
        self.n_sg = len(self.space_groups)
        self.n_zp = len(self.z_primes)
        self.n_molecules = 1

        max_sg = max(self.space_groups)
        sg_lookup = torch.full((max_sg + 1,), -1, dtype=torch.long)
        sg_lookup[torch.tensor(self.space_groups, dtype=torch.long)] = torch.arange(self.n_sg)
        self.sg_to_local = sg_lookup

        max_zp = max(self.z_primes)
        zp_lookup = torch.full((max_zp + 1,), -1, dtype=torch.long)
        zp_lookup[torch.tensor(self.z_primes, dtype=torch.long)] = torch.arange(self.n_zp)
        self.zp_to_local = zp_lookup

        # upfront library size for ConditionLogZTracker preallocation
        self.condition_library_size = self.n_molecules * self.n_sg * self.n_zp

    def set_n_molecules(self, n_molecules: int):
        """
        Called once from train.py's init_identifiers(), after the
        identifier -> mol_id registry (spanning mol_dataset and
        prior_dataset) is built, to correct condition_id's library sizing.
        """
        self.n_molecules = n_molecules
        self.condition_library_size = self.n_molecules * self.n_sg * self.n_zp

    def set_reward_clip(self, dataset_rewards):
        """
        We want to restrain the range of allowable rewards, by log-clipping the log reward below a certain threshold.
        NOTE this would have to be re-done dynamically if the conditioning evolves
        :param dataset_rewards:
        :return:
        """
        max_reward = max(dataset_rewards)
        reward_range = self.reward_range
        min_allowed_reward = max_reward - reward_range
        self.energy_clip = float(
            - min_allowed_reward * self.temperature)  # convert the minimum allowed reward to a clip on the energy
        self.reward_clip = min_allowed_reward

    def instantiate_crystals(self, x, mol_batch):
        if self.computes_require_cluster:
            crystal_batch = self.init_blank_crystal_batch(mol_batch)
        else:
            crystal_batch = mol_batch.clone()
        crystal_batch.latent_to_cell_params(x)
        if self.max_z_prime > 1:
            # Aunit labelling is a gauge: relabelling the z' units describes the same
            # crystal, and the reward is invariant to it (measured on real Z'=2 acridine:
            # median |d elj| 0.003 on a -1200 scale). Pin the gauge here anyway so the
            # physics, and every crystal that leaves this function for a buffer, is
            # ordering-independent by construction rather than only to within numerics.
            # The Z'>1 ordering penalty is unaffected -- it reads the raw latents.
            crystal_batch.canonicalize_zp_aunits()
        return crystal_batch

    def drain_energy_timing(self) -> dict:
        """BaseSet's counters, plus the mace call's INTERNAL split when this run is
        on the mace route.

        The split is what decides whether vectorising batch_to_mace_atomicdata is
        worth doing: `energy/mace_host_frac` is the share a vectorised builder could
        address, and one minus it is the Amdahl ceiling. Measured on the machine that
        actually runs the job -- the same question could not be answered locally,
        where MACE inference takes the box down (2026-08-14, twice).

        Drained unconditionally so the phase counters cannot accumulate across a
        window in which nothing else was logged; returns {} on stages with no energy
        calls, exactly as BaseSet does."""
        base = super().drain_energy_timing()
        phases = drain_mace_phase_timing()
        if base and phases:
            base.update(phases)
        return base

    def attach_gas_phase_reference(self, crystal_batch):
        """
        Host the MLIP gas-phase reference instead of recomputing it every call.

        WHAT IT IS. A lattice energy is (crystal - isolated molecule); mxtaltools'
        compute_lattice_uma/_mace evaluate BOTH legs per call. The second leg is an
        isolated molecule in a P1 cell at pbc=False, so for the rigid molecules this
        sampler uses its value depends only on MOLECULE IDENTITY -- not on the cell,
        not on the centroid, and (UMA being rotation-invariant) not on the sample's
        orientation either. Recomputing it per call buys nothing and costs a whole
        extra MLIP forward plus its host-side batch construction: roughly half the
        energy call.

        HOW IT SKIPS. compute_lattice_uma is already written as
        `if not hasattr(self, 'uma_gas_pot')`, so simply carrying the attribute on
        the batch bypasses the leg -- no change is needed in mxtaltools.

        HOW THE VALUE IS PRODUCED. By calling mxtaltools' own
        compute_lattice_gas_phase_* on one representative crystal per molecule,
        rather than reimplementing it. That leg has real structure worth not
        duplicating: a P1 reset, box_analysis, force_rebuild, and for Z'>1 a
        per-conformer split followed by a scatter-MEAN back onto the parent graph.

        WHAT CHANGES NUMERICALLY. Every sample of a molecule now gets ONE reference
        value instead of that molecule's value at its own orientation. The
        difference is UMA's rotation-invariance error, multiplied by 96.485 into
        kJ/mol -- i.e. this REMOVES a per-sample noise term from the reward rather
        than adding one. `gas_reference_audit` measures that term; run it before
        trusting the assumption on a new molecule set.

        Keyed on mol_id, which train.py's init_identifiers mints per identifier
        string. No mol_id (toys, unkeyed batches) -> silently do nothing and let the
        original per-call path run.
        """
        if self.energy_function not in ('uma', 'mace'):
            return
        if not getattr(self, 'host_gas_phase_reference', True):
            return
        mol_id = getattr(crystal_batch, 'mol_id', None)
        if mol_id is None:
            return

        key = f'{self.energy_function}_gas_pot'
        cache = self._gas_pot_cache
        ids = mol_id.tolist()
        missing = [i for i in dict.fromkeys(ids) if i not in cache]
        if missing:
            # one representative crystal per unseen molecule -- FIRST occurrence, so
            # the reference is a real sample of that molecule rather than a synthetic
            # pose the rest of the pipeline never produces
            first = {}
            for row, mid in enumerate(ids):
                if mid in missing and mid not in first:
                    first[mid] = row
            rows = torch.tensor([first[m] for m in missing], dtype=torch.long,
                                device=crystal_batch.device)
            sub = crystal_batch.subsample_new_batch(rows)
            fn = (sub.compute_lattice_gas_phase_uma if self.energy_function == 'uma'
                  else sub.compute_lattice_gas_phase_mace)
            vals = fn(self.predictor).detach()
            for slot, mid in enumerate(missing):
                cache[mid] = float(vals[slot])
            print(f"gas reference: cached {len(missing)} molecule(s) "
                  f"({len(cache)} total) -- skipping the gas leg from here on")

        ref = torch.tensor([cache[i] for i in ids], dtype=torch.float32,
                           device=crystal_batch.device)
        crystal_batch.add_graph_attr(ref, key)

    @torch.no_grad()
    def gas_reference_audit(self, crystal_batch, n: int = 8):
        """
        Measure what hosting the gas reference actually costs in accuracy: recompute
        the leg for `n` live samples and compare against the cached value, in kJ/mol.

        This is the number that decides whether hosting is a pure speedup or a
        correctness fix. The gas energy is subtracted and then scaled by 96.485, so a
        spread here is noise that was riding in every reward -- against a lattice
        energy of order 100 kJ/mol, a few tenths is real. Returns {} when hosting is
        off or unkeyed.
        """
        if self.energy_function not in ('uma', 'mace'):
            return {}
        key = f'{self.energy_function}_gas_pot'
        cached = getattr(crystal_batch, key, None)
        if cached is None:
            return {}
        rows = torch.arange(min(n, crystal_batch.num_graphs), device=crystal_batch.device)
        sub = crystal_batch.subsample_new_batch(rows)
        delattr(sub, key)  # force the real leg to run
        fn = (sub.compute_lattice_gas_phase_uma if self.energy_function == 'uma'
              else sub.compute_lattice_gas_phase_mace)
        fresh = fn(self.predictor).detach()
        drift = (fresh - cached[rows]) * 96.485
        return {'gas_ref/drift_mean_kj': float(drift.mean()),
                'gas_ref/drift_absmax_kj': float(drift.abs().max()),
                'gas_ref/drift_std_kj': float(drift.std()) if drift.numel() > 1 else 0.0}

    def analyze_crystal_batch(self, x, mol_batch, temperature, return_batch=False,
                              keep_grads: bool = False):  # x is gfn_outputs
        crystal_batch = self.instantiate_crystals(x, mol_batch)
        self.attach_gas_phase_reference(crystal_batch)

        analyze_kwargs = dict(cutoff=10,
                              supercell_size=10,
                              std_orientation=False,
                              predictor=self.predictor)
        analyze_kwargs.update(self.analyze_kwargs)

        if self.vector_conditioning:
            # per-sample condition override: today's static self.analyze_kwargs['c']/['width']
            # (a single fixed vector from YAML) isn't valid once distinct molecules_path entries
            # carry their own condition vectors, so replace them with the per-sample values
            # carried on the batch itself (set on each entry when the toy dataset was built,
            # riding through condition_samples/instantiate_crystals like any other attribute).
            # mxtaltools' latent_multiharmonic_en/_latent_field_params already broadcasts a
            # [B, cond_dim] `c` correctly (einsum 'kdc,...c->...kd').
            if not hasattr(crystal_batch, 'c'):
                raise RuntimeError(
                    "vector_conditioning is enabled but crystal_batch has no `c` attribute -- "
                    "instantiate_crystals likely dropped it (e.g. via init_blank_crystal_batch's "
                    "set_mol_attrs), or molecules_path/prior_path entries weren't built with `c` set "
                    "(see data_processing/generate_toy_prior.py)."
                )
            analyze_kwargs['c'] = crystal_batch.c.to(x.device)
            if hasattr(crystal_batch, 'width'):
                analyze_kwargs['width'] = crystal_batch.width.to(x.device)

        with torch.set_grad_enabled(keep_grads):
            out = crystal_batch.analyze(self.computes, **analyze_kwargs)

        for key in out.keys():
            crystal_batch.add_graph_attr(out[key], key)

        crystal_energy, ens_dict = self.generator_energy(crystal_batch, temperature, raw_latents=x)

        crystal_batch.add_graph_attr(crystal_energy, 'gfn_energy')

        if torch.any(torch.isinf(crystal_energy)) or torch.any(torch.isnan(crystal_energy)):
            crystal_energy[torch.isinf(crystal_energy)] = 0  # just patch it for now
            crystal_energy[torch.isnan(crystal_energy)] = 0

        for key in ens_dict.keys():
            setattr(crystal_batch, key, ens_dict[key].cpu().detach())

        if return_batch:
            return crystal_energy, clean_batch(crystal_batch.cpu().detach())
        else:
            return crystal_energy, None

    def generator_energy(self, crystal_batch, temperature, raw_latents: Optional[torch.tensor] = None):
        ens_dict = {}

        # BEFORE latent_params() below, which calls canonicalize_zp_aunits() IN PLACE:
        # the Z'>1 ordering term punishes the POLICY for emitting a non-canonical aunit
        # order, so it has to be read off the emitted state. Reading it after the batch
        # had been canonicalized is what made it identically zero (it was unreachable
        # anyway while max_z_prime was pinned at 1).
        zp_ordering_energy = None
        if self.max_z_prime > 1:
            zp_ordering_energy = self.compute_zp_order_penalty(crystal_batch, raw_latents)

        latents = crystal_batch.latent_params()
        if raw_latents is not None:
            upper_violation = F.relu(raw_latents - 1)
            lower_violation = F.relu(-(raw_latents + 1))
            # quadratic term gives a gentle, zero-slope onset right at the boundary; quartic
            # term steepens the wall for larger excursions without sharpening that onset
            bounding_energy = (upper_violation ** 2# + upper_violation ** 4
                               + lower_violation ** 2# + lower_violation ** 4
                               ).sum(
                dim=-1)  # discourage exploration beyond clip range
        else:
            bounding_energy = torch.zeros_like(latents[:, 0])

        if zp_ordering_energy is not None:
            bounding_energy = bounding_energy + zp_ordering_energy

        # energy() divides the total energy by temperature before use; pre-multiply here so
        # this domain-validity constraint stays equally stiff across sampling temperatures,
        # matching the same compensation already applied to jacobian_energy below
        bounding_energy = bounding_energy * temperature

        # `and not self.latent_energy`: a latent-scored crystal (latent_gaussian) has a
        # real crystal parameterization but no packing coefficient, pressure term or
        # physical mol_energy to read, so it must not enter this block. Without the
        # second clause it would walk straight into density_penalty(packing_coeff).
        if self.is_crystal and not self.latent_energy:
            density_energy = density_penalty(crystal_batch.packing_coeff)
            mol_energy = getattr(crystal_batch, self.energy_function)
            if self.energy_function not in ['uma', 'mace']:
                mol_energy = mol_energy / crystal_batch.z_prime

            if self.energy_function in ['lj', 'qlj', 'elj'] and self.lj_rescale is not None:
                # rescale functions with LJ-type minima to uma statistics
                mol_energy = self.lj_rescale * mol_energy

            reduction_energy = F.relu(crystal_batch.reduction_en)  # punish positive energies
            # same temperature compensation as bounding_energy above -- this is a validity
            # constraint, not part of the target Boltzmann distribution, so it shouldn't soften at high T
            reduction_energy = reduction_energy * temperature

            atm_conv = 101325  # conversion from atmospheres to Pa
            PV_en_conv = 6.022 * 10 ** -10  # conversion to energy in Pa*A^3 to kJ/mol
            pressure_energy = self.pressure * PV_en_conv * atm_conv * crystal_batch.cell_volume / crystal_batch.sym_mult / crystal_batch.z_prime

            ens_dict['reduction_energy'] = reduction_energy
            ens_dict['mol_energy'] = mol_energy
            ens_dict['density_energy'] = density_energy
            ens_dict['bounding_energy'] = bounding_energy
            ens_dict['pressure_energy'] = pressure_energy
        else:
            # reduction_energy is structurally zero for a latent-scored problem, which is
            # what keeps the analytic log Z exact. It matters: on P-1 the reduced region
            # is a thin, oddly-shaped set in box-latent space, and no zero-reduction ball
            # wide enough for a gaussian was found in 4000 draws (best: 0 at the centre,
            # 0.105 at the edge of a +-0.15 ball). Leaving the penalty on would contaminate
            # the target by ~1 nat at 1.5 sigma. Being structural rather than a config
            # knob means a config cannot switch it back on by accident.
            reduction_energy = torch.zeros_like(bounding_energy)
            # still log the bounding term -- for a latent-scored problem it is the ONLY
            # active penalty, so dropping it from ens_dict would leave the run with no
            # visibility into the one thing constraining the box
            ens_dict['bounding_energy'] = bounding_energy

        if self.energy_function == 'latent_harmonic':
            crystal_energy = getattr(crystal_batch, 'latent_harmonic')

        elif self.energy_function == 'latent_multiharmonic':
            crystal_energy = getattr(crystal_batch, 'latent_multiharmonic')

        elif self.energy_function == 'latent_gaussian':
            # Same compute as latent_harmonic (crystal_analysis.py reuses the function),
            # but on a batch whose is_crystal is True -- so dead rows, periodic angle dims
            # and the jacobian correction are all live. Set `c` to the canonical value
            # (0.0) on dead rows in analyze_kwargs and the energy becomes live-dims-only
            # for free: their contribution is exactly ((0 - 0)/w)^2 = 0.
            crystal_energy = getattr(crystal_batch, 'latent_gaussian')

        elif self.energy_function in ['lj', 'qlj', 'elj', 'silu', 'uma', 'mace']:
            crystal_energy = self.lj_coeff * mol_energy + self.density_coeff * density_energy + pressure_energy

        else:
            assert False, f'{self.energy_function} not implemented'

        # `or self.latent_energy`: the jacobian is a CHANGE OF MEASURE from the box-latent
        # parameterization to the physical one (cartesian aunit position, Haar rotation), so
        # that a target defined by a PHYSICAL energy is sampled correctly in physical space.
        # A latent-space analytic target is defined IN the latent space, so there is no
        # measure to correct and the correction is not merely unnecessary but wrong: the
        # target would become gaussian * |J|, which has no closed form in box coordinates.
        # It would also silently break the dead-row test itself -- rows 3/4/5 are cell
        # angles, which enter cell_volume, so with the jacobian on those dims are no longer
        # flat even when the energy ignores them, and the rows-live arm's fictitious volume
        # stops being log 2 per dim.
        if not self.is_crystal or self.latent_energy:
            jacobian_energy = torch.zeros_like(bounding_energy)
        else:
            jacobian_energy, jacobian_components = self.compute_jacobian(crystal_batch, temperature)
            ens_dict.update(jacobian_components)

        if self.energy_clip is not None:

            total_energy = (log_rescale_positive(crystal_energy, self.energy_clip) +
                            bounding_energy * self.bounding_coeff +
                            reduction_energy * self.reduction_coeff)
            return (
                log_rescale_positive(total_energy, self.energy_clip + 0.1 * np.abs(self.energy_clip)) + jacobian_energy,
                # to prevent total saturation by the energy function, add a buffer over the clip
                ens_dict)
        else:
            total_energy = (crystal_energy +
                            bounding_energy * self.bounding_coeff +
                            reduction_energy * self.reduction_coeff +
                            jacobian_energy)
            return total_energy, ens_dict

    def compute_jacobian(self, crystal_batch, temperature):
        """jacobian correction for aunit positions and orientation angles only"""
        latent_rotvecs = crystal_batch.latent_params()[:, -3 * crystal_batch.max_z_prime:]
        sph_rotvec = lat2sph_rotvec(latent_rotvecs, crystal_batch.max_z_prime)
        sph = sph_rotvec.view(crystal_batch.num_graphs, crystal_batch.max_z_prime, 3)
        theta = sph[..., 0]  # polar angle
        r = sph[..., 2]  # rotation magnitude
        eps = 1e-8
        # these come from composing the transforms of spherical -> cartesian ball and then to uniform rotation;
        # sum over z', because each dim gets its own correction
        rot_r_energy = - temperature * 2 * torch.log(torch.sin(r / 2).clamp_min(eps)).sum(dim=-1)
        rot_theta_energy = - temperature * torch.log(torch.sin(theta).clamp_min(eps)).sum(dim=-1)
        frac_jacobian_energy = - crystal_batch.z_prime * temperature * torch.log(
            # this comes from the cartesian -> fractional transform, with a factor for each independent object transformed
            crystal_batch.cell_volume / crystal_batch.sym_mult)
        jacobian_energy = rot_r_energy + rot_theta_energy + frac_jacobian_energy
        # split out for logging: the rotational terms diverge (log) at r -> 0 and
        # theta -> 0/pi, so their batch MAX is the diagnostic (train.py's
        # log_thermo_properties logs it); the frac term is the -T*z'*log V measure
        # tilt toward larger cells. jacob_july24 measured healthy-window peaks at
        # 0.8-3.5 nats -- the clamp caps a graze at ~37, seen only during excursions
        components = {
            'jacobian_energy': jacobian_energy,
            'rot_r_jacobian_energy': rot_r_energy,
            'rot_theta_jacobian_energy': rot_theta_energy,
            'frac_jacobian_energy': frac_jacobian_energy,
        }
        return jacobian_energy, components

    def compute_zp_order_penalty(self, crystal_batch, raw_latents: Optional[torch.tensor] = None):
        """
        Penalize the policy for emitting asymmetric units out of the canonical order
        (closest -> furthest from the origin). A domain-validity term on the raw latent,
        exactly like the [-1, 1] range term it is added to.

        Reads `raw_latents` rather than crystal_batch.aunit_centroid so the penalty does
        not depend on whether anything has canonicalized the batch: mxtaltools'
        `latent_params()` calls `canonicalize_zp_aunits()` in place, and the crystal
        builder may canonicalize too, either of which silently zeroes a batch-read
        version of this term. Datasets are stored canonical by construction, so the
        raw_latents=None fallback (prebuilt_sample_to_reward) reads the batch and
        legitimately scores ~0 there.

        The comparison is done on the CELL-fractional centroid, not on the latent
        directly: that is the quantity canonicalize_aunit_order sorts on, and the two
        orderings differ because the aunit box is anisotropic (SG14 auv = [1, .25, 1]),
        so ranking by |latent| would punish states that are already canonical.
        """
        n, k = crystal_batch.num_graphs, self.max_z_prime
        if raw_latents is not None:
            # same [-1,1] -> [0,1] -> *auv chain as inv_latent_transform's centroid leg
            if not hasattr(crystal_batch, 'asym_unit_lut'):
                crystal_batch.build_asym_unit_tensor()
            auvs = crystal_batch.asym_unit_lut[crystal_batch.sg_ind].to(raw_latents.device)
            lat_centroids = raw_latents[:, 6:6 + 3 * k].reshape(n, k, 3)
            per_aunit_centroids = (lat_centroids / 2 + 0.5) * auvs.unsqueeze(1)
        else:
            per_aunit_centroids = crystal_batch.aunit_centroid.reshape(n, k, 3)

        idx = torch.arange(k, device=crystal_batch.device)[None, ...]
        mask = (idx >= (crystal_batch.z_prime[..., None]))[..., None].expand(-1, -1, 3)
        # fill unused z' slots with 1 so they sort last -- OUT OF PLACE: reshape of a
        # contiguous (n, 3*max_z_prime) returns a VIEW, so writing through it would mutate
        # crystal_batch.aunit_centroid itself and sever the autograd path the bounding
        # energy rides on. Same idiom as mxtaltools' canonicalize_aunit_order.
        per_aunit_centroids = torch.where(mask, torch.ones_like(per_aunit_centroids), per_aunit_centroids)
        origin_dists = per_aunit_centroids.norm(dim=2)
        overlaps = -origin_dists.diff(dim=1)
        return F.relu(overlaps).mean(dim=-1) ** 2

    def crystal_multiharmonic_en(self, crystal_batch, latents):
        if not hasattr(self, 'modes'):
            self.modes = torch.tensor(generate_modes(10, self.dim, 4.0, 3.0), device=self.device)
            self.crystal_modes = crystal_batch.latent_transform.inverse(self.modes,
                                                                        crystal_batch.sg_ind[:10],
                                                                        crystal_batch.radius[:10])
        diffs = latents[:, None, :] - self.modes[None, :, :]
        sqdist = (diffs ** 2).sum(dim=-1)  # (B, K)
        exponent = -0.5 * sqdist  # (B, K)
        crystal_energy = -torch.logsumexp(exponent, dim=1)  # (B,)
        return crystal_energy

    def latent_multiharmonic_en(self, crystal_batch, latents):
        if not hasattr(self, 'modes'):
            self.modes = torch.tensor(generate_modes(10, self.dim, 4.0, 3.0), device=self.device)
            self.crystal_modes = crystal_batch.latent_transform.inverse(self.modes,
                                                                        crystal_batch.sg_ind[:10],
                                                                        crystal_batch.radius[:10])
        diffs = latents[:, None, :] - self.modes[None, :, :]
        sqdist = (diffs ** 2).sum(dim=-1)  # (B, K)
        exponent = -0.5 * sqdist  # (B, K)
        crystal_energy = -torch.logsumexp(exponent, dim=1)  # (B,)
        """
                    #Partition function
                    
                    D = self.modes.shape[1]
                    det_term = (2 * np.pi * self.temperature) ** (D / 2)
                    weights = torch.ones(self.modes.shape[0], device=self.modes.device) / self.modes.shape[0]
                    Z = det_term * torch.sum(weights).item()
                    log_Z = np.log(Z)
                    """
        return crystal_energy

    def crystal_harmonic_en(self, crystal_batch):
        # a trivial energy function, for testing
        cell_params = crystal_batch.zp1_cell_parameters()
        if not hasattr(self, 'modes'):
            self.modes = -torch.ones((1, self.dim), device=self.device)
            self.crystal_modes = crystal_batch.latent_transform.inverse(self.modes,
                                                                        crystal_batch.sg_ind[:1],
                                                                        crystal_batch.radius[:1])
        crystal_energy = 0.5 * (cell_params - self.crystal_modes[0]).pow(2).sum(dim=1)
        # analytic Z = (2pi*T)^(d/2)
        return crystal_energy

    @torch.no_grad()
    def prebuilt_sample_to_reward(self, crystals, temperature):
        """
        For pre-built, pre-scored crystal, generate the approriate reward for this point in training.
        :param temperature: per-sample torch float tensor containing temperature for each sample to be rewarded
        :param crystals:
        :return:
        """
        if isinstance(crystals, list):
            crystal_batch = collate_data_list(crystals)
        else:
            crystal_batch = crystals

        energy, _ = self.generator_energy(crystal_batch, temperature)

        if torch.is_tensor(temperature):
            sample_temperature = temperature.to(crystal_batch.device)
        elif isinstance(temperature, float) or isinstance(temperature, int):
            sample_temperature = temperature * torch.ones_like(energy, device=crystal_batch.device)
        else:
            assert False

        return -energy / sample_temperature

    def energy(self,
               x,
               mol_batch,
               log_temperature: torch.tensor,
               return_exp: bool = False,
               keep_grads: bool = False,
               internal_oom_recovery: Optional[bool] = None):
        """
        Energy is not really bounded. Or necessarily well scaled.
        We do exponential rescaling later with a temperature. For higher temperature,
        potential is less sharply peaked.
        :param mol_batch:
        :param temperature:
        :param x:
        :param internal_oom_recovery: per-call override of self.internal_oom_recovery -- pass
            True to force the adaptive sub-batching/OOM catch-and-shrink path for this call
            regardless of the instance default (e.g. a one-off pass over a huge prior dataset
            at init, where a slow, self-healing pass is preferable to a hard crash).
        :return:
        """
        temperature = 10 ** log_temperature
        if return_exp:
            energy, crystal_batch = self.batched_analyze_crystal_batch(x, mol_batch, temperature,
                                                                       return_batch=return_exp,
                                                                       keep_grads=keep_grads,
                                                                       internal_oom_recovery=internal_oom_recovery)
            return energy / temperature, crystal_batch
        else:
            energy = self.batched_analyze_crystal_batch(x, mol_batch, temperature,
                                                        return_batch=return_exp,
                                                        keep_grads=keep_grads,
                                                        internal_oom_recovery=internal_oom_recovery)
            return energy / temperature

    def batched_analyze_crystal_batch(self, x, mol_batch, temperature,
                                      return_batch=False, keep_grads: bool = False,
                                      internal_oom_recovery: Optional[bool] = None):
        use_recovery = self.internal_oom_recovery if internal_oom_recovery is None else internal_oom_recovery
        if not use_recovery:
            # analyze the whole batch in a single call: no adaptive sub-batching, no OOM
            # catch-and-shrink here -- any OOM raises straight through to the caller, so
            # a global batch size reduction (e.g. train.py's handle_train_epoch_error) can
            # handle it instead of this class quietly shrinking its own internal chunk size.
            outs = self.analyze_crystal_batch(x, mol_batch, temperature,
                                              return_batch=return_batch, keep_grads=keep_grads)
            energy = outs[0] if keep_grads else outs[0].detach()
            if return_batch:
                return energy, outs[1]
            else:
                return energy

        if not hasattr(self, 'batch_size'):
            if self.energy_function in ['uma', 'mace']:
                self.batch_size = 1000
            else:
                self.batch_size = 10000
        cursor = 0
        n_samples = len(x)
        energies = torch.zeros(len(x), dtype=torch.float32, device='cpu')
        already_oomed = False
        samples_batch = None
        while cursor < n_samples:  # todo get a single unified interface for all the places we do this
            try:
                inds = np.arange(cursor, min(n_samples, cursor + self.batch_size))
                mol_batch_i = mol_batch.subsample_new_batch(inds)

                outs = self.analyze_crystal_batch(x[inds], mol_batch_i, temperature[inds],
                                                  return_batch=return_batch, keep_grads=keep_grads)

                if not keep_grads:
                    energies[inds] = outs[0].cpu().detach()
                else:
                    energies[inds] = outs[0].cpu()

                if return_batch:
                    batch_i = outs[1].detach().cpu()

                    if samples_batch is None:
                        samples_batch = batch_i
                    else:
                        samples_batch = samples_batch.append_batch(batch_i)

                cursor += len(inds)
                if (self.batch_size <= 100000) and (self.batch_size < n_samples) and not already_oomed:
                    self.batch_size += max(int(self.batch_size * 0.01), 1)

            except (RuntimeError, ValueError) as e:
                if is_cuda_oom(e):
                    if self.batch_size == 1:
                        assert False, "Cascading OOM failure in molecule energy evaluation"
                    self.batch_size = max(int(self.batch_size * 0.65), 1)
                    # print(f"OOM in energy evaluation: dropping batch size to {self.batch_size}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    already_oomed = True
                    sleep(0.1)
                else:
                    raise e

        del outs
        # RESTORED 2026-08-14. These were commented out; on the mace/uma route that
        # left the largest allocation the process ever makes -- the init pass over the
        # whole prior dataset, chunked at supercell_size 10 -- holding its blocks in
        # the caching allocator for the rest of the run. They are free blocks, not a
        # leak, but `cuda_memory_fraction` is a HARD cap (set_per_process_memory_fraction
        # in train.py), so cache the run cannot reuse still counts against it, and the
        # blocks are shaped like supercell neighbour lists that a T-step MLP rollout
        # never asks for.
        #
        # THIS IS NOT THE PER-STEP HOT PATH, which is why it can afford to sync. The
        # `not use_recovery` branch at the top of this function returns EARLY, and
        # every physical config sets energy_config.internal_oom_recovery: false -- so
        # training-time energy calls never reach this line. What does reach it is the
        # init pass (internal_oom_recovery=True, explicitly) and the other one-off
        # whole-dataset scans, i.e. exactly the calls whose footprint outlives them.
        gc.collect()
        if torch.cuda.is_available():        # synchronize() RAISES with no device
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if return_batch:
            return energies.to(x.device), samples_batch
        else:
            return energies.to(x.device)

    def init_blank_crystal_batch(self,
                                 mol_batch):
        device = mol_batch.device
        n = mol_batch.num_graphs

        if self.sg_conditioning:
            sgs = mol_batch.sg_ind
        else:
            sgs = torch.full(
                (n,),
                int(self.space_groups[0]),
                dtype=torch.long,
                device=device,
            )

        crystal_batch = self.batch.clone().to(self.device)

        ones3 = torch.ones((n, 3), device=device)
        zeros1 = torch.zeros(n, device=device)
        eye3 = torch.eye(3, device=device).expand(n, 3, 3).clone()
        ones1 = torch.ones(n, device=device)
        trues1 = torch.ones(n, dtype=torch.bool, device=device)

        z_ones3 = torch.ones((n, 3 * self.max_z_prime), device=device)
        z_ones1 = torch.ones((n, self.max_z_prime), device=device)
        setattr(crystal_batch, '_num_graphs', mol_batch.num_graphs)
        # `device` is a computed property (self.z.device) on crystal_batch, but z isn't
        # set yet, so temporarily force it via setattr so set_mol_attrs can use self.device.
        # setattr on this class writes straight into the underlying PyG store rather than
        # the property, so it must be removed again once z is populated below -- otherwise
        # it leaks a raw torch.device into to_dict()/clone() and breaks any code that later
        # treats this batch as a plain mol_batch (e.g. crystal batches recycled as anchors).
        setattr(crystal_batch, 'device', mol_batch.device)
        crystal_batch.set_mol_attrs(mol_batch.clone())
        delattr(crystal_batch, 'device')

        blank_batch_properties = {
            'aunit_handedness': z_ones1,
            'nonstandard_symmetry': ~trues1,
            'cell_lengths': ones3,
            'cell_angles': ones3,
            'aunit_centroid': z_ones3,
            'aunit_orientation': z_ones3,
            'lj': zeros1,
            'scaled_lj': zeros1,
            'reduction_en': zeros1,
            'T_fc': eye3,
            'T_cf': eye3,
            'cell_volume': zeros1,
            'packing_coeff': zeros1,
            'density': zeros1,
            'z_prime': ones1,
            'is_well_defined': trues1,
        }

        crystal_batch.set_mol_attrs(mol_batch.clone())
        slice_dict = torch.arange(0, crystal_batch.num_graphs + 1, 1, device='cpu')
        inc_dict = torch.zeros(crystal_batch.num_graphs, dtype=torch.long, device='cpu')
        for key in blank_batch_properties:
            crystal_batch.add_graph_attr(blank_batch_properties[key], key, slice_dict, inc_dict)

        crystal_batch.reset_sg_info(sgs)
        crystal_batch.add_graph_attr(mol_batch.z_prime, 'z_prime', slice_dict, inc_dict)

        return crystal_batch

    def sample(self,
               batch_size,
               reasonable_only: bool = False,
               target_packing_coeff: Optional[float] = None
               ):
        """
        Return random crystal sample
        note this is NOT weighted by energy
        """
        assert False, "not implemented"
        # with torch.no_grad():
        #     crystal_batch = self.init_blank_crystal_batch(batch_size)
        #     if not reasonable_only:
        #         crystal_batch.sample_random_reduced_crystal_parameters(target_packing_coeff=target_packing_coeff)
        #
        #     else:  # higher quality crystals, but expensive
        #         crystal_batch.sample_reasonable_random_parameters(
        #             tolerance=3,
        #             max_attempts=50,
        #             target_packing_coeff=target_packing_coeff,
        #             sample_niggli=True
        #         )
        #
        #     return crystal_batch.zp1_std_cell_parameters()

    def condition_samples(self,
                          mol_batch,
                          temperature: torch.tensor = None,
                          sg_inds: torch.tensor = None,
                          z_primes: torch.tensor = None,
                          repeats: int = 1,
                          ):
        """
        mol_batch is assumed to already be tiled into `repeats`-sized groups of
        identical molecules (x-repeated-K-times layout, as produced by
        CrystalBuffer.loader(..., repeats=repeats)). Conditions are sampled once
        per group and broadcast across the group via repeat_interleave, so every
        trajectory drawn for a given molecule shares the same condition.

        Also computes and attaches, as plain attributes on `mol_batch`
        (matching the existing z_prime/sg_ind convention rather than
        add_graph_attr), two immutable per-sample fields:
          - `conditions`: the realized float condition vector (detached)
          - `condition_id`: a mixed-radix combination of mol_id (identifier
            -> dense integer registry spanning mol_dataset AND prior_dataset,
            built once in train.py's init_identifiers()) and dense-local
            SG/Z' index, used only to key ConditionLogZTracker
        Both ride along automatically into any buffer this batch is later
        added to, since CrystalBuffer/AnchorBuffer.add carries the whole
        resident Batch through generically -- no buffer.py changes needed.

        When vector_conditioning is on, toy energy functions read their
        condition vector directly off the batch too: molecules_path entries
        carry a `c` attribute (and optionally `width`), which gets appended
        into the condition tensor here and read again in
        analyze_crystal_batch. This is deliberately independent of
        molecule_conditioning: that flag routes the condition through
        VectorMoleculeGraphModel (a GNN over the actual molecule graph, for
        physical runs where "the condition is the graph itself");
        vector_conditioning instead keeps conditions_type='vector' (a plain
        scalarMLP over the condition tensor -- see models/gfn.py's
        init_conditioner) since toy `c` vectors aren't tied to any real
        molecular graph and running the GNN over blank/dummy scaffolding
        would be wasted machinery.
        """
        num_groups = mol_batch.num_graphs // repeats

        conds = []  # feedback of zp information is broken
        if self.temperature_conditioning:
            """
            sample temp range, or a fixed temp, or an override temp
            """
            if temperature is not None:
                log_T_tensor = torch.log10(temperature)
            else:
                # Uniform samples in [0, 1]
                u = torch.rand(num_groups, dtype=torch.float32)
                # Transform to [log_low, log_high]
                log_T_tensor = self.log_temperature_range[0] + u * (
                            self.log_temperature_range[1] - self.log_temperature_range[0])
                log_T_tensor = log_T_tensor.repeat_interleave(repeats)

            log_T_tensor = log_T_tensor.to(mol_batch.device)
            conds.append(log_T_tensor)
        else:
            log_T_tensor = torch.log10(
                torch.ones((mol_batch.num_graphs, 1), device=mol_batch.device) * self.temperature)

        if sg_inds is not None:
            sg_to_sample = sg_inds.clone()
        else:
            sg_to_sample = torch.tensor(
                np.random.choice(self.space_groups, num_groups, replace=True)
            ).repeat_interleave(repeats).to(mol_batch.device)

        if z_primes is not None:
            zp_to_sample = z_primes.clone()
        else:
            zp_to_sample = torch.tensor(
                np.random.choice(self.z_primes, num_groups, replace=True)
            ).repeat_interleave(repeats).to(mol_batch.device)

        if self.sg_conditioning:
            conds.append(torch.stack([self.SG_FEATURE_TENSOR[sg]
                                      for sg in sg_to_sample]).to(mol_batch.device)
                         )

        if self.zp_conditioning:
            conds.append(zp_to_sample.clone()[:, None].float())

        if self.vector_conditioning:
            if not hasattr(mol_batch, 'c'):
                raise RuntimeError(
                    "vector_conditioning is enabled but mol_batch has no `c` attribute -- "
                    "molecules_path/prior_path entries weren't built with `c` set (see "
                    "data_processing/generate_toy_prior.py)."
                )
            conds.append(mol_batch.c.to(mol_batch.device))

        if self.embedding_conditioning:
            if not hasattr(mol_batch, 'embedding') or mol_batch.embedding is None:
                raise RuntimeError(
                    "embedding_conditioning is enabled but mol_batch has no `embedding` "
                    "attribute -- molecules_path/prior_path entries weren't built with one "
                    "(see build_qm9_conditions.py).")
            # stored [n_graphs, 3, bottleneck] (equivariant, kept in that shape so a future
            # augmentation can rotate it with the molecule); the conditioner takes a flat
            # vector, and flattening is well defined ONLY because the molecule frame is
            # pinned -- build_qm9_conditions.py makes the file a fixed point of the
            # trainer's own orient_molecule(mode='std') for exactly this reason
            emb = mol_batch.embedding.to(mol_batch.device)
            emb = emb.reshape(mol_batch.num_graphs, -1)
            if emb.shape[-1] != self.embedding_conditioning_dim:
                raise RuntimeError(
                    f"embedding width {emb.shape[-1]} != embedding_conditioning_dim "
                    f"{self.embedding_conditioning_dim}; the conditioner was built for a "
                    f"different encoder than the one that baked this file")
            conds.append(emb)

        mol_batch.z_prime = zp_to_sample
        mol_batch.reset_sg_info(sg_to_sample)

        conds = [c if c.dim() > 1 else c[:, None] for c in conds]
        if len(conds) == 0:
            condition = torch.zeros((mol_batch.num_graphs, 1), device=mol_batch.device)
        else:
            condition = torch.cat(conds, dim=-1)

        mol_id = getattr(mol_batch, 'mol_id', None)
        if mol_id is None:
            # a batch whose dataset had no .identifier field to register (e.g. an
            # older/synthetic prior file that predates init_identifiers()) has no
            # molecule identity of its own -- collapse onto molecule index 0
            # rather than erroring, so condition_id there only resolves SG/Z'.
            mol_id = torch.zeros(mol_batch.num_graphs, dtype=torch.long, device=mol_batch.device)
        else:
            mol_id = mol_id.to(mol_batch.device)

        sg_local = self.sg_to_local.to(mol_batch.device)[sg_to_sample]
        zp_local = self.zp_to_local.to(mol_batch.device)[zp_to_sample]
        condition_id = (mol_id * (self.n_sg * self.n_zp) +
                        sg_local * self.n_zp +
                        zp_local)

        mol_batch.conditions = condition.detach()
        mol_batch.condition_id = condition_id

        # sg_to_sample / zp_to_sample used to ride along here and were unpacked but never
        # read at any of the 11 call sites -- they are already on the batch as
        # `sg_ind` / `z_prime`, which is where callers that want them get them. Dropping
        # them also removes the last crystal-shaped members of this signature, so a
        # non-crystal energy_function can implement it without padding.
        return (mol_batch,
                log_T_tensor.flatten(),
                condition,
                condition_id,
                )


def generate_modes(K=20, D=12, rho=4.0, delta=3.0, seed=42):
    np.random.seed(seed)
    mus = []

    def is_well_separated(new_mu, mus, delta):
        if len(mus) == 0:
            return True
        dists = np.linalg.norm(np.array(mus) - new_mu, axis=1)
        return np.all(dists >= delta)

    while len(mus) < K:
        mu = np.random.randn(D)
        mu = rho * mu / np.linalg.norm(mu)
        if is_well_separated(mu, mus, delta):
            mus.append(mu)

    return np.stack(mus)  # shape (K, D)


def clean_batch(batch):
    # Detach all tensors and move them to CPU in-place
    keys = set()
    if hasattr(batch, 'keys'):
        keys.update(batch.keys())  # standard PyG data attributes

    # Also grab any extra custom tensor attributes (e.g. ellipsoid_overlap)
    for k in batch.__dict__:
        val = getattr(batch, k)
        if torch.is_tensor(val) or (isinstance(val, list) and all(torch.is_tensor(v) for v in val)):
            keys.add(k)

    for key in keys:
        try:
            val = getattr(batch, key)
            if torch.is_tensor(val):
                setattr(batch, key, val.detach().cpu())
            elif isinstance(val, list) and all(torch.is_tensor(v) for v in val):
                setattr(batch, key, [v.detach().cpu() for v in val])
        except Exception:
            continue  # ignore protected or bad attrs
    del batch.asym_unit_dict
    return batch
