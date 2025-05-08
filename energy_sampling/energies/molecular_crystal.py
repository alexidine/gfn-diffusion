import torch
import torch.nn.functional as F

from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

from .base_set import BaseSet


class MolecularCrystal(BaseSet):
    def __init__(self, device,
                 dim: int = 12,
                 test_molecule: str = 'UREA',
                 space_group: int = 2,
                 temperature: float = 10):
        super(MolecularCrystal, self).__init__()
        self.device = device
        self.data_ndim = dim
        self.space_group = space_group

        self.test_molecule = test_molecule
        self.initialize_test_molecule(test_molecule)
        self.temperature = temperature

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

    def analyze_crystal_batch(self, x, return_batch=False):  # x is gfn_outputs
        crystal_batch = self.init_blank_crystal_batch(len(x))
        raw_cell_params = crystal_batch.destandardize_cell_parameters(x)
        crystal_batch.set_cell_parameters(raw_cell_params,
                                          skip_box_analysis=True)
        crystal_batch.clean_cell_parameters(mode='soft',
                                            length_pad=1.5,
                                            canonicalize_orientations=False,
                                            constrain_z=True,
                                            enforce_niggli=True)
        cluster_batch = crystal_batch.mol2cluster(cutoff=6,
                                                  supercell_size=10,
                                                  align_to_standardized_orientation=False)
        cluster_batch.construct_radial_graph(cutoff=6)
        cluster_batch.compute_LJ_energy()
        silu_energy = cluster_batch.compute_silu_energy() / cluster_batch.num_atoms
        cluster_batch.silu_pot = silu_energy
        packing_loss = 100*F.relu(-(cluster_batch.packing_coeff - 0.5))**2  # apply a squared penalty for packing coeffs less than 0.5
        crystal_energy = silu_energy + packing_loss

        if return_batch:
            return crystal_energy, cluster_batch
        else:
            return crystal_energy

    def energy(self, x):
        """
        Energy is not really bounded. Or necessarily well scaled.
        We do exponential rescaling later with a temperature. For higher temperature,
        potential is less sharply peaked.
        :param x:
        :return:
        """
        return self.analyze_crystal_batch(x)/self.temperature

    def init_blank_crystal_batch(self, batch_size):
        return collate_data_list([MolCrystalData(
            molecule=self.mol.clone(),
            sg_ind=self.space_group,
            aunit_handedness=torch.ones(1),
        ) for _ in range(batch_size)]).to(self.device)

    def sample(self, batch_size):
        """
        Return random crystal sample
        note this is NOT weighted by energy
        """
        crystal_batch = self.init_blank_crystal_batch(batch_size)
        crystal_batch.sample_random_reduced_crystal_parameters(cleaning_mode='hard')
        # higher quality crystals but very expensive
        # crystal_batch.sample_reasonable_random_parameters(
        #     tolerance=3,
        #     max_attempts=50
        # )

        return crystal_batch.standardize_cell_parameters().cpu().detach()
