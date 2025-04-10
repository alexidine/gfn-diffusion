import torch

from mxtaltools.dataset_utils.data_classes import MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list

from .base_set import BaseSet


class MolecularCrystal(BaseSet):
    def __init__(self, device, dim=12, test_molecule='UREA'):
        super(MolecularCrystal, self).__init__()
        self.device = device
        self.data_ndim = dim

        self.test_molecule = test_molecule
        self.initialize_test_molecule(test_molecule)


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
            self.mol_mass = 60.06

        # NICOTANIMIDE from molview
        elif test_molecule == 'NICOTANIMIDE':
            self.atom_coords = torch.tensor([
                [-2.3940, 1.1116, -0.0088],
                [1.7614,   -1.2284,   -0.0034],
                [-2.4052,   -1.1814,    0.0027],
                [-0.2969,    0.0397,    0.0024],
                [0.4261,   1.2273,    0.0039],
                [0.4117,   -1.1510,   -0.0013],
                [1.8161,    1.1886,    0.0018],
                [-1.7494,    0.0472,    0.0045],
                [2.4302,   -0.0535,   -0.0018]
            ], dtype=torch.float32, device=self.device)
            self.atom_coords -= self.atom_coords.mean(dim=0)
            self.atom_types = torch.tensor([8, 7, 7, 6, 6, 6, 6, 6, 6], dtype=torch.long, device=self.device)
            self.mol_mass = 122.12

    def prep_crystal_data(self,
                          sample,
                          space_group,
                          cell_lengths,
                          cell_angles,
                          aunit_centroid,
                          aunit_orientation,
                          aunit_handedness):
        crystal_list = [
            MolCrystalData(
                molecule=sample,
                sg_ind=space_group,
                cell_lengths=torch.ones(3),
                cell_angles=torch.ones(3) * torch.pi / 2,
                aunit_centroid=torch.ones(3) * 0.5,
                aunit_orientation=torch.ones(3),
                aunit_handedness=int(aunit_handedness[ind]),
                identifier=sample.smiles,
            )
            for ind in range(len(samples))
        ]
        crystal_batch = collate_data_list(crystal_list)

        return crystal_batch

    # def score_crystal_data(self):
    #     prep_crystal_data = self.prep_crystal_data(
    #         molecule=sample,
    #         sg_ind=space_group,
    #         cell_lengths=torch.ones(3),
    #         cell_angles=torch.ones(3) * torch.pi / 2,
    #         aunit_centroid=torch.ones(3) * 0.5,
    #         aunit_orientation=torch.ones(3),
    #         aunit_handedness=int(aunit_handedness[ind]),
    #         identifier=sample.smiles)
    #
    #     pass

    def score_crystal_data(self, x): # x is gfn_outputs

        sample = self.sample(batch_size)
        crystal_list = [
            MolCrystalData(
                molecule=sample,
                sg_ind=space_group,
                cell_lengths=torch.ones(3),
                cell_angles=torch.ones(3) * torch.pi / 2,
                aunit_centroid=torch.ones(3) * 0.5,
                aunit_orientation=torch.ones(3),
                aunit_handedness=int(aunit_handedness[ind]),
                identifier=sample.smiles,
            )
            for ind in range(len(samples))
        ]
        crystal_batch = collate_data_list(crystal_list)

        raw_cell_params = crystal_batch.destandardize_cell_parameters(x)
        crystal_batch.set_cell_parameters(raw_cell_params)
        crystal_batch.clean_cell_parameters()

        lj_pot, es_pot, scaled_lj_pot = crystal_batch.build_and_analyze()

        return scaled_lj_pot

    def energy(self, x):
        # return score_crystal_data(...)
        pass

    def sample(self, batch_size):
        # return MolData(...)
        pass