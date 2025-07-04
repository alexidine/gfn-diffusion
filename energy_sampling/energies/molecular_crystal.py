import copy
from typing import Optional

import numpy as np
import torch

from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

import torch.nn.functional as F

from .base_set import BaseSet


class MolecularCrystal(BaseSet):
    def __init__(self, device,
                 energy_function: str,
                 dim: int = 12,
                 space_group: int = 2,
                 max_temperature: float = 10,
                 min_temperature: float = 0.01,
                 turnover_pot: float = 20.0,
                 density_coeff: float = 0,
                 temperature_scaling_factor: float = 1,
                 temperature: float = 1.0,
                 temperature_conditioning: bool = False,
                 energy_clip: float = 100,
                 ):
        super(MolecularCrystal, self).__init__()
        self.device = device
        self.data_ndim = dim
        self.space_group = space_group
        self.energy_function = energy_function
        self.energy_clip = energy_clip

        self.ellipsoid_scale = 1
        self.density_coeff = density_coeff
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.temperature_scaling_factor = temperature_scaling_factor
        self.temperature_conditioning = temperature_conditioning
        self.turnover_pot = turnover_pot  # energy above which to soften intermolecular repulsion

        self.temperature = temperature  # for static temperature work

    def instantiate_crystals(self, x, mol_batch):
        crystal_batch = self.init_blank_crystal_batch(mol_batch)
        crystal_batch.gen_basis_to_cell_params(x)
        crystal_batch.box_analysis()
        return crystal_batch

    def analyze_crystal_batch(self, x, mol_batch, return_batch=False):  # x is gfn_outputs
        crystal_batch = self.instantiate_crystals(x, mol_batch)
        if self.energy_function not in ['ellipsoid_overlap',
                                        'silu_energy']:  # no need to actually build the crystal, this is much faster
            cluster_batch = crystal_batch
            lj_energy = torch.zeros(crystal_batch.num_graphs, device=self.device)
            silu_energy = torch.zeros_like(lj_energy)
        else:
            cluster_batch = crystal_batch.mol2cluster(cutoff=6,
                                                      supercell_size=10,
                                                      align_to_standardized_orientation=True)

            cluster_batch.construct_radial_graph(cutoff=6)
            #lj_energy, normed_lj_energy = cluster_batch.compute_LJ_energy()
            silu_energy = cluster_batch.compute_silu_energy()  # softened short-range LJ-type energy

        if self.energy_function == 'ellipsoid_overlap':
            if not hasattr(self, 'ellipsoid_model'):
                cluster_batch.load_ellipsoid_model()
                self.ellipsoid_model = copy.deepcopy(cluster_batch.ellipsoid_model)
                self.ellipsoid_model = self.ellipsoid_model.to(self.device)
                self.ellipsoid_model.eval()
            # simplified ellipsoid energy testing
            _, _, _, _, _, _, normed_ellipsoid_overlap \
                = cluster_batch.compute_ellipsoidal_overlap(
                semi_axis_scale=self.ellipsoid_scale,
                model=self.ellipsoid_model,
                return_details=True)

            cluster_batch.ellipsoid_overlap = normed_ellipsoid_overlap.flatten()
        else:
            cluster_batch.ellipsoid_overlap = torch.zeros_like(silu_energy)

        cluster_batch.silu_pot = silu_energy
        cluster_batch.lj_pot = silu_energy  #lj_energy
        crystal_energy = self.generator_energy(cluster_batch)
        cluster_batch.gfn_energy = crystal_energy
        if return_batch:
            return crystal_energy, cluster_batch
        else:
            return crystal_energy

    def generator_energy(self, cluster_batch):
        if cluster_batch.device != self.device:
            cluster_batch = cluster_batch.to(self.device)

        if self.energy_function == 'latent_harmonic':
            # a trivial energy function, for testing
            latents = cluster_batch.cell_params_to_gen_basis()
            if not hasattr(self, 'modes'):
                self.modes = -torch.ones((1, 12), device=self.device)
                self.crystal_modes = cluster_batch.latent_transform.inverse(self.modes,
                                                                            cluster_batch.sg_ind[:1],
                                                                            cluster_batch.radius[:1])
            crystal_energy = 0.5 * (latents - self.modes[0]).pow(2).sum(dim=1) / self.temperature
            # analytic Z = (2pi*T)^(d/2)
        elif self.energy_function == 'crystal_harmonic':
            # a trivial energy function, for testing
            cell_params = cluster_batch.cell_parameters()
            if not hasattr(self, 'modes'):
                self.modes = -torch.ones((1, 12), device=self.device)
                self.crystal_modes = cluster_batch.latent_transform.inverse(self.modes,
                                                                            cluster_batch.sg_ind[:1],
                                                                            cluster_batch.radius[:1])
            crystal_energy = 0.5 * (cell_params - self.crystal_modes[0]).pow(2).sum(dim=1) / self.temperature
            # analytic Z = (2pi*T)^(d/2)

        elif self.energy_function == 'latent_multiharmonic':
            latents = cluster_batch.cell_params_to_gen_basis()
            if not hasattr(self, 'modes'):
                self.modes = torch.tensor(generate_modes(10, 12, 4.0, 3.0), device=self.device)
                self.crystal_modes = cluster_batch.latent_transform.inverse(self.modes,
                                                                            cluster_batch.sg_ind[:10],
                                                                            cluster_batch.radius[:10])

            diffs = latents[:, None, :] - self.modes[None, :, :]
            sqdist = (diffs ** 2).sum(dim=-1)  # (B, K)
            exponent = -0.5 * sqdist / self.temperature  # (B, K)
            crystal_energy = -torch.logsumexp(exponent, dim=1)  # (B,)
            """
            #Partition function
            
            D = self.modes.shape[1]
            det_term = (2 * np.pi * self.temperature) ** (D / 2)
            weights = torch.ones(self.modes.shape[0], device=self.modes.device) / self.modes.shape[0]
            Z = det_term * torch.sum(weights).item()
            log_Z = np.log(Z)
            """

        elif self.energy_function == 'crystal_multiharmonic':
            latents = cluster_batch.cell_params_to_gen_basis()
            if not hasattr(self, 'modes'):
                self.modes = torch.tensor(generate_modes(10, 12, 4.0, 3.0), device=self.device)
                self.crystal_modes = cluster_batch.latent_transform.inverse(self.modes,
                                                                            cluster_batch.sg_ind[:10],
                                                                            cluster_batch.radius[:10])

            diffs = latents[:, None, :] - self.modes[None, :, :]
            sqdist = (diffs ** 2).sum(dim=-1)  # (B, K)
            exponent = -0.5 * sqdist / self.temperature  # (B, K)
            crystal_energy = -torch.logsumexp(exponent, dim=1)  # (B,)

        elif self.energy_function == 'ellipsoid_overlap':
            intermolecular_energy = cluster_batch.ellipsoid_overlap.detach().clone().contiguous()
            density_energy = F.relu(-(cluster_batch.packing_coeff.detach().clone().contiguous() - 0.9)) ** 2
            crystal_energy = intermolecular_energy + self.density_coeff * density_energy

        elif self.energy_function == 'silu_energy':
            density_energy = F.relu(-(cluster_batch.packing_coeff - 0.9)) ** 2
            intermolecular_energy = self.soften_LJ_energy(cluster_batch.silu_pot) / cluster_batch.num_atoms
            crystal_energy = intermolecular_energy + self.density_coeff * density_energy

        else:
            assert False, f'{self.energy_function} not implemented'

        return crystal_energy.clip(min=-self.energy_clip, max=self.energy_clip)

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

        with torch.no_grad():
            energy = self.generator_energy(crystal_batch)

        if torch.is_tensor(temperature):
            sample_temperature = temperature.to(self.device)
        elif isinstance(temperature, float) or isinstance(temperature, int):
            sample_temperature = temperature * torch.ones_like(energy, device=self.device)
        else:
            assert False

        return (-energy / sample_temperature).detach()

    def energy(self, x, mol_batch, log_temperature: torch.tensor, return_exp: bool = False):
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

    def soften_LJ_energy(self, lj_energy):
        # soften the repulsion
        softened_energy = lj_energy.clone()
        high_bools = softened_energy > self.turnover_pot
        softened_energy[high_bools] = self.turnover_pot + torch.log(
            softened_energy[high_bools] + 1 - self.turnover_pot)
        softened_energy = softened_energy.clip(max=50)

        return softened_energy

    def init_blank_crystal_batch(self, mol_batch):  # todo no possible way this is the most efficient way to do this

        ones3 = torch.ones(3, device=self.device)
        zeros1 = torch.zeros(1, device=self.device)

        if self.energy_function == 'ellipsoid_overlap':
            overlap_tensor = torch.zeros(1, device=self.device)
        else:
            overlap_tensor = None

        crystal_batch = collate_data_list([MolCrystalData(
            molecule=mol_batch[ind].clone(),  # must be cloned
            sg_ind=self.space_group,
            aunit_handedness=torch.ones(1),
            cell_lengths=torch.ones(3, device=self.device),
            # if we don't put dummies in here, later ops to_data_list fail
            # but if we do put dummies in here, it does box analysis one-by-one which is super slow
            cell_angles=ones3,
            aunit_centroid=ones3,
            aunit_orientation=ones3,
            skip_box_analysis=True,
            silu_pot=zeros1,
            packing_coeff=zeros1,
            lj_pot=zeros1,
            scaled_lj_pot=zeros1,
            es_pot=zeros1,
            ellipsoid_overlap=overlap_tensor,
        ) for ind in range(len(mol_batch))]).to(self.device)

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

            return crystal_batch.standardize_cell_parameters()

    def get_conditioning_tensor(self,
                                mol_batch,
                                temperature: torch.tensor = None,
                                ):

        if self.temperature_conditioning:
            if temperature is None:  # sample randomly in log space
                rands = torch.rand(mol_batch.num_graphs, device=mol_batch.device, dtype=torch.float32)

                log_min = torch.log10(torch.tensor(self.min_temperature, dtype=torch.float32, device=mol_batch.device))
                log_max = torch.log10(torch.tensor(self.max_temperature, dtype=torch.float32, device=mol_batch.device))

                log_temps = log_min + (log_max - log_min) * rands ** self.temperature_scaling_factor
                return log_temps[:, None]
            else:
                return torch.log10(temperature[:, None])
        else:
            return torch.log10(torch.ones((mol_batch.num_graphs, 1), device=mol_batch.device) * self.temperature)


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
