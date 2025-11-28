import copy
import gc
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mxtaltools.common.utils import log_rescale_positive
from mxtaltools.constants.space_group_feature_tensor import SG_FEATURE_TENSOR
from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.dataset_utils.data_classes import MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
from .base_set import BaseSet


def density_penalty(packing_coeff):
    """
    draw crystals into the physically reasonable region
    :param packing_coeff:
    :return:
    """
    cp = packing_coeff.clip(min=0.1, max=2)  # clip here for safety - this loss term can explode
    return F.relu(-(torch.log(cp) - np.log(0.55))) ** 2 + F.relu(cp - 0.95) ** 2


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
                 temperature_scaling_factor: float = 1,
                 temperature: float = 1.0,
                 temperature_conditioning: bool = False,
                 lj_coeff: float = 1.0,
                 molecule_conditioning: bool = False,
                 sg_conditioning: bool = False,
                 zp_conditioning: bool = False,
                 space_groups: Optional[list] = [2],
                 bounding_coeff: float = 1.0,
                 niggli_coeff: float = 1.0,
                 max_z_prime: int = 1,
                 z_primes: Tuple[int] = (1,),
                 uma_path: Optional[str] = None,
                 reward_range: float = None,
                 lj_rescale: list = None,
                 ):

        super(MolecularCrystal, self).__init__()
        self.device = device
        self.data_ndim = 6 + 6 * max_z_prime
        self.energy_function = energy_function
        self.SG_FEATURE_TENSOR = SG_FEATURE_TENSOR.clone()  # store space group information

        self.density_coeff = density_coeff
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.temperature_scaling_factor = temperature_scaling_factor
        self.temperature_conditioning = temperature_conditioning
        self.lj_coeff = lj_coeff
        self.bounding_coeff = bounding_coeff
        self.niggli_coeff = niggli_coeff
        self.molecule_conditioning = molecule_conditioning
        self.sg_conditioning = sg_conditioning
        self.space_groups = space_groups
        self.max_z_prime = max_z_prime
        self.z_primes = z_primes
        self.zp_conditioning = zp_conditioning
        self.reward_range = reward_range
        self.lj_rescale = lj_rescale
        if self.energy_function == 'uma':
            self.uma_predictor = init_uma_crystal_predictor(uma_path, device=self.device)

        self.temperature = temperature  # for static temperature work
        self.energy_clip = None

        self.batch = collate_data_list([MolCrystalData(max_z_prime=max_z_prime)], max_z_prime=max_z_prime)

        self.sg_cache = {}
        for sg in range(1, 230):
            self.sg_cache[sg] = np.stack(SYM_OPS[int(sg)])

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
        self.energy_clip = - min_allowed_reward * self.temperature  # convert the minimum allowed reward to a clip on the energy

    def instantiate_crystals(self, x, mol_batch):
        crystal_batch = self.init_blank_crystal_batch(mol_batch)
        crystal_batch.latent_to_cell_params(x)
        return crystal_batch

    def analyze_crystal_batch(self, x, mol_batch, return_batch=False):  # x is gfn_outputs
        crystal_batch = self.instantiate_crystals(x, mol_batch)

        if self.energy_function not in ['lj', 'qlj','elj', 'silu', 'uma']:  # todo simplify all these energies
            lj_energy = torch.zeros(crystal_batch.num_graphs, device=self.device)
            qlj_energy = torch.zeros_like(lj_energy)
            elj_energy = torch.zeros_like(lj_energy)
            normed_lj_energy = torch.zeros_like(lj_energy)
            silu_energy = torch.zeros_like(lj_energy)
            niggli_overlap = torch.zeros_like(lj_energy)
            uma_energy = torch.zeros_like(lj_energy)
        else:
            cutoff = 10

            out = crystal_batch.analyze(['lj', 'qlj', 'elj','silu', 'niggli'],
                                        cutoff=cutoff,
                                        supercell_size=5,
                                        std_orientation=False)
            lj_energy = out['lj']
            qlj_energy = out['qlj']
            elj_energy = out['elj']
            normed_lj_energy = log_rescale_positive(lj_energy, 0)
            silu_energy = out['silu']
            niggli_overlap = out['niggli']
            if self.energy_function == 'uma':
                # clear memory
                del out
                torch.cuda.empty_cache()
                gc.collect()

                with torch.no_grad():
                    uma_energy = crystal_batch.compute_crystal_uma(
                        predictor=self.uma_predictor,
                        std_orientation=False) * 96.485  # output in kJ/mol (of unit cells)
            else:
                uma_energy = torch.zeros_like(lj_energy)

        # todo this is getting silly
        crystal_batch.add_graph_attr(silu_energy, 'silu_pot')
        crystal_batch.add_graph_attr(lj_energy, 'lj_pot')
        crystal_batch.add_graph_attr(qlj_energy, 'qlj_pot')
        crystal_batch.add_graph_attr(elj_energy, 'elj_pot')
        crystal_batch.add_graph_attr(niggli_overlap, 'niggli_overlap')
        crystal_batch.add_graph_attr(normed_lj_energy, 'scaled_lj_pot')
        crystal_batch.add_graph_attr(uma_energy, 'uma_pot')

        crystal_energy, ens_dict = self.generator_energy(crystal_batch, raw_latents=x)

        crystal_batch.add_graph_attr(crystal_energy, 'gfn_energy')

        if torch.any(torch.isinf(crystal_energy)) or torch.any(torch.isnan(crystal_energy)):
            crystal_energy[torch.isinf(crystal_energy)] = 0  # just patch it for now
            crystal_energy[torch.isnan(crystal_energy)] = 0

        for key in ens_dict.keys():
            setattr(crystal_batch, key, ens_dict[key].cpu().detach())

        if return_batch:
            return crystal_energy, clean_batch(crystal_batch.cpu().detach())
        else:
            return crystal_energy

    def generator_energy(self, crystal_batch, raw_latents=None):
        ens_dict = {}

        latents = crystal_batch.latent_params()
        if raw_latents is not None:
            bounding_energy = (F.relu(raw_latents - 1) ** 2 + F.relu(-(raw_latents + 1)) ** 2).sum(
                dim=-1)  # discourage exploration beyond clip range
        else:
            bounding_energy = torch.zeros_like(latents[:, 0])

        if self.max_z_prime > 1:
            bounding_energy = self.compute_zp_order_penalty(bounding_energy, crystal_batch)

        if self.energy_function in ['lj', 'qlj', 'elj','silu', 'uma']:
            density_energy = density_penalty(crystal_batch.packing_coeff)
            if self.energy_function == 'lj':
                mol_energy = crystal_batch.lj_pot  # / crystal_batch.num_atoms
            elif self.energy_function == 'qlj':
                mol_energy = crystal_batch.qlj_pot  # / crystal_batch.num_atoms
            elif self.energy_function == 'elj':
                mol_energy = crystal_batch.elj_pot
            elif self.energy_function == 'silu':
                mol_energy = crystal_batch.silu_pot  # / crystal_batch.num_atoms
            elif self.energy_function == 'uma':
                # gas_pot =  crystal_batch.uma_gas_pot
                gas_pot = -9587.2559
                mol_energy = (
                            crystal_batch.uma_pot / crystal_batch.sym_mult - gas_pot)  # the raw lattice energdy # / crystal_batch.num_atoms  # todo un-hardcode this when we fix it in the training set
            else:
                assert False

            if self.energy_function in ['lj','qlj','elj'] and self.lj_rescale is not None:
                # rescale functions with LJ-type minima to uma statistics
                lj_mean, lj_std, uma_mean, uma_std = self.lj_rescale
                mol_energy = (mol_energy - lj_mean)/lj_std * uma_std + uma_mean

            niggli_energy = F.relu(-crystal_batch.niggli_overlap) ** 2  # punish negative overlaps

            ens_dict['niggli_energy'] = niggli_energy
            ens_dict['mol_energy'] = mol_energy
            ens_dict['density_energy'] = density_energy
            ens_dict['bounding_energy'] = bounding_energy
        else:
            niggli_energy = torch.zeros_like(bounding_energy)

        if self.energy_function == 'latent_harmonic':
            crystal_energy = self.latent_harmonic_en(crystal_batch, latents)

        elif self.energy_function == 'crystal_harmonic':
            crystal_energy = self.crystal_harmonic_en(crystal_batch)

        elif self.energy_function == 'latent_multiharmonic':
            crystal_energy = self.latent_multiharmonic_en(crystal_batch, latents)

        elif self.energy_function == 'crystal_multiharmonic':
            crystal_energy = self.crystal_multiharmonic_en(crystal_batch, latents)

        elif self.energy_function in ['lj', 'qlj','elj', 'silu', 'uma']:
            crystal_energy = self.lj_coeff * mol_energy + self.density_coeff * density_energy

        else:
            assert False, f'{self.energy_function} not implemented'

        total_energy = crystal_energy + bounding_energy * self.bounding_coeff + niggli_energy * self.niggli_coeff
        if self.energy_clip is not None:
            return log_rescale_positive(total_energy, self.energy_clip), ens_dict
        else:
            return total_energy, ens_dict

    def compute_zp_order_penalty(self, bounding_energy, crystal_batch):
        # penalize the model for placing asymmetric units out of the canonical order (closest -> furthest from origin)
        per_aunit_centroids = crystal_batch.aunit_centroid.reshape(crystal_batch.num_graphs,
                                                                   crystal_batch.max_z_prime, 3)
        idx = torch.arange(crystal_batch.max_z_prime, device=crystal_batch.device)[None, ...]
        mask = (idx >= (crystal_batch.z_prime[..., None]))[..., None].expand(-1, -1, 3)
        per_aunit_centroids[mask] = 1  # this will put lower Z' options always at the end
        origin_dists = per_aunit_centroids.norm(dim=2)
        overlaps = -origin_dists.diff(dim=1)
        zp_ordering_energy = F.relu(overlaps).mean(dim=-1) ** 2
        bounding_energy = bounding_energy + zp_ordering_energy
        return bounding_energy

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

    def latent_harmonic_en(self, crystal_batch, latents):
        # a trivial energy function, for testing
        if not hasattr(self, 'modes'):
            self.modes = -torch.ones((1, self.dim), device=self.device)
            self.crystal_modes = crystal_batch.latent_transform.inverse(self.modes,
                                                                        crystal_batch.sg_ind[:1],
                                                                        crystal_batch.radius[:1])
        crystal_energy = 0.5 * (latents - self.modes[0]).pow(2).sum(dim=1)
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

        energy, _ = self.generator_energy(crystal_batch)

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
               return_exp: bool = False):
        """
        Energy is not really bounded. Or necessarily well scaled.
        We do exponential rescaling later with a temperature. For higher temperature,
        potential is less sharply peaked.
        :param mol_batch:
        :param temperature:
        :param x:
        :return:
        """
        energy, crystal_batch = self.analyze_crystal_batch(x, mol_batch, return_batch=True)
        temperature = 10 ** log_temperature
        sample_temperature = temperature

        if return_exp:
            return energy / sample_temperature, crystal_batch
        else:
            return energy / sample_temperature

    def init_blank_crystal_batch(self, mol_batch):  # todo no possible way this is the most efficient way to do this
        if self.sg_conditioning:
            sgs = mol_batch.sg_ind
        else:
            sgs = [self.space_groups[0] for _ in range(mol_batch.num_graphs)]

        crystal_batch = self.batch.clone()
        ones3 = torch.ones((mol_batch.num_graphs, 3), device='cpu')
        zeros1 = torch.zeros((mol_batch.num_graphs), device='cpu')
        eye3 = torch.eye(3, device='cpu').repeat(mol_batch.num_graphs, 1, 1)
        ones1 = torch.ones(mol_batch.num_graphs, device='cpu')
        trues1 = torch.zeros(mol_batch.num_graphs, dtype=torch.bool, device='cpu').fill_(True)
        zones3 = torch.ones((mol_batch.num_graphs, 3 * self.max_z_prime), device='cpu')
        zones1 = torch.ones((mol_batch.num_graphs, self.max_z_prime), device='cpu')
        blank_batch_properties = {
            'aunit_handedness': zones1,
            'nonstandard_symmetry': ~trues1,
            'cell_lengths': ones3,
            'cell_angles': ones3,
            'aunit_centroid': zones3,
            'aunit_orientation': zones3,
            'silu_pot': zeros1,
            'lj_pot': zeros1,
            'qlj_pot': zeros1,
            'elj_pot': zeros1,
            'scaled_lj_pot': zeros1,
            'niggli_overlap': zeros1,
            'T_fc': eye3,
            'T_cf': eye3,
            'cell_volume': zeros1,
            'packing_coeff': zeros1,
            'density': zeros1,
            'z_prime': ones1,
            'is_well_defined': trues1,
        }
        setattr(crystal_batch, '_num_graphs', mol_batch.num_graphs)
        setattr(crystal_batch, 'device', mol_batch.device)
        crystal_batch.set_mol_attrs(mol_batch.clone())
        slice_dict = torch.arange(0, crystal_batch.num_graphs + 1, 1, device='cpu')
        inc_dict = torch.zeros(crystal_batch.num_graphs, dtype=torch.long, device='cpu')
        for key in blank_batch_properties:
            crystal_batch.add_graph_attr(blank_batch_properties[key], key, slice_dict, inc_dict)

        crystal_batch.reset_sg_info(sgs)
        crystal_batch.add_graph_attr(mol_batch.z_prime, 'z_prime', slice_dict, inc_dict)

        crystal_batch = crystal_batch.to(self.device)

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
        with torch.no_grad():
            crystal_batch = self.init_blank_crystal_batch(batch_size)
            if not reasonable_only:
                crystal_batch.sample_random_reduced_crystal_parameters(target_packing_coeff=target_packing_coeff)

            else:  # higher quality crystals, but expensive
                crystal_batch.sample_reasonable_random_parameters(
                    tolerance=3,
                    max_attempts=50,
                    target_packing_coeff=target_packing_coeff,
                    sample_niggli=True
                )

            return crystal_batch.zp1_std_cell_parameters()

    def get_conditioning_tensor(self,
                                mol_batch,
                                temperature: torch.tensor = None,
                                sg_inds: torch.tensor = None,
                                z_primes: torch.tensor = None,
                                ):

        conds = []  # feedback of zp information is broken
        if self.temperature_conditioning:
            """
            sample temp range, or a fixed temp, or an override temp
            """
            if temperature is None:  # sample randomly in log space
                rands = torch.rand(mol_batch.num_graphs, device=mol_batch.device, dtype=torch.float32)

                log_min = torch.log10(torch.tensor(self.min_temperature, dtype=torch.float32, device=mol_batch.device))
                log_max = torch.log10(torch.tensor(self.max_temperature, dtype=torch.float32, device=mol_batch.device))

                log_temps = log_min + (log_max - log_min) * rands ** self.temperature_scaling_factor
                log_T_tensor = log_temps[:, None]
            else:
                log_T_tensor = torch.log10(temperature[:, None])

            conds.append(log_T_tensor)
        else:
            log_T_tensor = torch.log10(
                torch.ones((mol_batch.num_graphs, 1), device=mol_batch.device) * self.temperature)

        if self.molecule_conditioning:
            mol_embedding = mol_batch.embedding.flatten(1, 2)
            conds.append(mol_embedding)

        if sg_inds is not None:
            sg_to_sample = sg_inds.clone()
        else:
            sg_to_sample = torch.tensor(np.random.choice(self.space_groups, mol_batch.num_graphs, replace=True)).to(
                mol_batch.device)

        # if z_primes is not None:
        #     zp_to_sample = z_primes.clone()
        # else:  # can't sample z prime here because of issues in the formatting of the cyrstaldata
        #
        #     # zp_to_sample = torch.tensor(np.random.choice(self.z_primes, mol_batch.num_graphs, replace=True)).to(
        #     #     mol_batch.device)

        if self.sg_conditioning:
            conds.append(torch.stack([self.SG_FEATURE_TENSOR[sg]
                                      for sg in sg_to_sample]).to(mol_batch.device)
                         )

        if self.zp_conditioning:
            conds.append(z_primes.clone()[:, None].float())

        # todo add z prime information to conditioning tensor

        return (log_T_tensor.flatten(),
                sg_to_sample,
                torch.cat(conds, dim=1) if len(conds) > 0 else torch.zeros_like(log_T_tensor),
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
    del batch.asym_unit_dict, batch.latent_transform
    return batch
