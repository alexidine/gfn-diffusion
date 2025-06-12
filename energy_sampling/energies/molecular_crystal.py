import copy
from typing import Optional

import torch

from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

import torch.nn.functional as F

from .base_set import BaseSet


class MolecularCrystal(BaseSet):
    def __init__(self, device,
                 dim: int = 12,
                 space_group: int = 2,
                 max_temperature: float = 10,
                 min_temperature: float = 0.01,
                 turnover_pot: float = 0.01,
                 density_coeff: float = 0,
                 temperature_scaling_factor: float = 1,
                 temperature_conditioning: bool = False
                 ):
        super(MolecularCrystal, self).__init__()
        self.device = device
        self.data_ndim = dim
        self.space_group = space_group

        self.ellipsoid_scale = 1
        self.density_coeff = density_coeff
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.temperature_scaling_factor = temperature_scaling_factor
        self.temperature_conditioning = temperature_conditioning
        self.turnover_pot = turnover_pot  # energy above which to soften intermolecular repulsion

    def instantiate_crystals(self, x, mol_batch):
        crystal_batch = self.init_blank_crystal_batch(mol_batch)
        crystal_batch.gen_basis_to_cell_params(x, clip_min_length=0.5)  # don't allow micro cells
        #crystal_batch.cell_lengths = crystal_batch.cell_lengths + 3  # TODO add this bias directly to the policy model
        crystal_batch.box_analysis()
        return crystal_batch

    def analyze_crystal_batch(self, x, mol_batch, return_batch=False):  # x is gfn_outputs
        crystal_batch = self.instantiate_crystals(x, mol_batch)
        cluster_batch = crystal_batch.mol2cluster(cutoff=6,
                                                  supercell_size=10,
                                                  align_to_standardized_orientation=True)

        cluster_batch.construct_radial_graph(cutoff=6)
        cluster_batch.compute_LJ_energy()
        silu_energy = cluster_batch.compute_silu_energy()  # softened short-range LJ-type energy

        # if not hasattr(self, 'ellipsoid_model'):
        #     cluster_batch.load_ellipsoid_model()
        #     self.ellipsoid_model = copy.deepcopy(cluster_batch.ellipsoid_model)
        #     self.ellipsoid_model = self.ellipsoid_model.to(self.device)
        #     self.ellipsoid_model.eval()
        # # simplified ellipsoid energy testing
        # _, _, _, _, _, _, normed_ellipsoid_overlap \
        #     = cluster_batch.compute_ellipsoidal_overlap(
        #     semi_axis_scale=self.ellipsoid_scale,
        #     model=self.ellipsoid_model,
        #     return_details=True)

        #cluster_batch.ellipsoid_overlap = torch.zeros_like(silu_energy) #normed_ellipsoid_overlap.flatten()
        cluster_batch.silu_pot = silu_energy
        crystal_energy = self.generator_energy(cluster_batch)
        cluster_batch.gfn_energy = crystal_energy
        if return_batch:
            return crystal_energy, cluster_batch
        else:
            return crystal_energy

    def generator_energy(self, cluster_batch):
        density_energy = F.relu(-(cluster_batch.packing_coeff - 1)) ** 2
        intermolecular_energy = self.soften_LJ_energy(cluster_batch.silu_pot / cluster_batch.num_atoms)
        #intermolecular_energy = cluster_batch.ellipsoid_overlap
        crystal_energy = intermolecular_energy + self.density_coeff * density_energy

        return crystal_energy

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

        energy = self.generator_energy(crystal_batch)

        if torch.is_tensor(temperature):
            sample_temperature = temperature
        elif isinstance(temperature, float) or isinstance(temperature, int):
            sample_temperature = temperature * torch.ones_like(energy)
        else:
            assert False

        return -energy / sample_temperature

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
        softened_energy[high_bools] = self.turnover_pot + torch.log10(
            softened_energy[high_bools] + 1 - self.turnover_pot)
        softened_energy = softened_energy.clip(max=50)

        return softened_energy

    def init_blank_crystal_batch(self, mol_batch):  # todo no possible way this is the most efficient way to do this
        crystal_batch = collate_data_list([MolCrystalData(
            molecule=mol_batch[ind].clone(),
            sg_ind=self.space_group,
            aunit_handedness=torch.ones(1),
            cell_lengths=torch.ones(3, device=self.device),
            # if we don't put dummies in here, later ops to_data_list fail
            # but if we do put dummies in here, it does box analysis one-by-one which is super slow
            cell_angles=torch.ones(3, device=self.device),
            aunit_centroid=torch.ones(3, device=self.device),
            aunit_orientation=torch.ones(3, device=self.device),
            skip_box_analysis=True,
            silu_pot=torch.zeros(1, device=self.device),
            packing_coeff=torch.zeros(1, device=self.device),
            lj_pot=torch.zeros(1, device=self.device),
            scaled_lj_pot=torch.zeros(1, device=self.device),
            es_pot=torch.zeros(1, device=self.device),
            #ellipsoid_overlap=torch.zeros(1, device=self.device)
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
        """Todo add autoencoder conditioning"""
        if self.temperature_conditioning:
            if temperature is None:  # sample randomly in log space
                rands = torch.rand(mol_batch.num_graphs, device=mol_batch.device, dtype=torch.float32)

                log_min = torch.log10(torch.tensor(self.min_temperature, dtype=torch.float32, device=self.device))
                log_max = torch.log10(torch.tensor(self.max_temperature, dtype=torch.float32, device=self.device))

                log_temps = log_min + (log_max - log_min) * rands ** self.temperature_scaling_factor
                return log_temps[:, None]
            else:
                return torch.log10(temperature[:, None])
        else:
            return None
