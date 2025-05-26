from typing import Optional

import torch

from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.crystal_search.standalone_crystal_opt import sample_about_crystal
from mxtaltools.common.utils import softplus_shift
import torch.nn.functional as F

from .base_set import BaseSet


class MolecularCrystal(BaseSet):
    def __init__(self, device,
                 dim: int = 12,
                 test_molecule: str = 'UREA',
                 space_group: int = 2,
                 temperature: float = 10,
                 turnover_pot: float = 0.01,
                 ):
        super(MolecularCrystal, self).__init__()
        self.device = device
        self.data_ndim = dim
        self.space_group = space_group

        self.test_molecule = test_molecule
        self.initialize_test_molecule(test_molecule)
        self.temperature = temperature
        self.turnover_pot = turnover_pot  # energy above which to soften intermolecular repulsion

    def initialize_test_molecule(self, test_molecule):
        # UREA from molview - default if not specified
        if test_molecule == 'UREA':
            self.atom_coords = torch.tensor([
                [-1.3042, - 0.0008, 0.0001],
                [0.6903, - 1.1479, 0.0001],
                [0.6888, 1.1489, 0.0001],
                [- 0.0749, - 0.0001, - 0.0003],
            ], dtype=torch.float32, device=self.device)
            self.atom_coords -= self.atom_coords.mean(0)
            self.atom_types = torch.tensor([8, 7, 7, 6], dtype=torch.long, device=self.device)

        # NICOTANIMIDE from molview
        elif test_molecule == 'NICOTANIMIDE':
            self.atom_coords = torch.tensor([
                [-2.3940, 1.1116, -0.0088],
                [1.7614, -1.2284, -0.0034],
                [-2.4052, -1.1814, 0.0027],
                [-0.2969, 0.0397, 0.0024],
                [0.4261, 1.2273, 0.0039],
                [0.4117, -1.1510, -0.0013],
                [1.8161, 1.1886, 0.0018],
                [-1.7494, 0.0472, 0.0045],
                [2.4302, -0.0535, -0.0018]
            ], dtype=torch.float32, device=self.device)
            self.atom_coords -= self.atom_coords.mean(dim=0)
            self.atom_types = torch.tensor([8, 7, 7, 6, 6, 6, 6, 6, 6], dtype=torch.long, device=self.device)

        self.mol = MolData(
            z=self.atom_types,
            pos=self.atom_coords,
            x=self.atom_types,
            skip_mol_analysis=False,
        )

    def instantiate_crystals(self, x):
        crystal_batch = self.init_blank_crystal_batch(len(x))
        crystal_batch.gen_basis_to_cell_params(x)

        return crystal_batch

    def analyze_crystal_batch(self, x, return_batch=False):  # x is gfn_outputs
        crystal_batch = self.instantiate_crystals(x)
        cluster_batch = crystal_batch.mol2cluster(cutoff=6,
                                                  supercell_size=10,
                                                  align_to_standardized_orientation=False)
        cluster_batch.construct_radial_graph(cutoff=6)
        cluster_batch.compute_LJ_energy()
        silu_energy = cluster_batch.compute_silu_energy()
        cluster_batch.silu_pot = silu_energy / cluster_batch.num_atoms
        crystal_energy = self.generator_energy(crystal_batch,
                                               silu_energy,
                                               crystal_batch.num_atoms)

        if return_batch:
            return crystal_energy, cluster_batch
        else:
            return crystal_energy

    def generator_energy(self, cluster_batch, silu_pot, num_atoms):
        # aunit_lengths = cluster_batch.scale_lengths_to_aunit()
        # box_loss = F.relu(-(aunit_lengths - 3)).sum(1) + F.relu(
        #     aunit_lengths - (3 * 2 * cluster_batch.radius[:, None])).sum(1)
        # crystal_energy = silu_energy / num_atoms / self.temperature + box_loss

        # soften the repulsion
        crystal_energy = silu_pot.clone()
        high_bools = crystal_energy > self.turnover_pot
        crystal_energy[high_bools] = self.turnover_pot + torch.log10(crystal_energy[high_bools] + 1 - self.turnover_pot)
        crystal_energy = crystal_energy.clip(max=50)

        return crystal_energy

    def local_opt(self, x,
                  max_num_steps,
                  samples_per_opt):
        """
        Do a local optimization of the crystal parameters
        :param x:
        :return:
        """
        crystal_batch = self.instantiate_crystals(x)
        optimization_record = crystal_batch.optimize_crystal_parameters(
            mol_orientation=None,
            enforce_niggli=True,
            optim_target='silu',
            cutoff=6,
            compression_factor=1,
            max_num_steps=max_num_steps,
            lr=1e-3,
        )
        samples_out = optimization_record[-1]
        if samples_per_opt > 0:
            nearby_samples = sample_about_crystal(samples_out,
                                                  noise_level=0.05,  # empirically gets us an LJ std about 3
                                                  num_samples=samples_per_opt,
                                                  cutoff=6,
                                                  do_silu_pot=True,
                                                  enforce_niggli=True)
            for ss in nearby_samples:
                samples_out.extend(ss)

        samples = torch.cat([elem.cell_params_to_gen_basis() for elem in samples_out])
        silu_energies = torch.tensor([elem.silu_pot for elem in samples_out])
        packing_coeffs = torch.tensor([elem.packing_coeff for elem in samples_out])
        num_atoms = torch.tensor([elem.num_atoms for elem in samples_out])
        energy_out = torch.tensor(
            [-self.generator_energy(sample_batch, silu_energies, num_atoms) for sample_batch in samples_out])

        return samples, energy_out

    def energy(self, x):
        """
        Energy is not really bounded. Or necessarily well scaled.
        We do exponential rescaling later with a temperature. For higher temperature,
        potential is less sharply peaked.
        :param x:
        :return:
        """
        return self.analyze_crystal_batch(x)

    def init_blank_crystal_batch(self, batch_size):
        return collate_data_list([MolCrystalData(
            molecule=self.mol.clone(),
            sg_ind=self.space_group,
            aunit_handedness=torch.ones(1),
            cell_lengths=torch.ones(3, device=self.device),
            # if we don't put dummies in here, later ops to_data_list fail
            cell_angles=torch.ones(3, device=self.device),
            aunit_centroid=torch.ones(3, device=self.device),
            aunit_orientation=torch.ones(3, device=self.device),
        ) for _ in range(batch_size)]).to(self.device)

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
