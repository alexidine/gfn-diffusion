import copy
from typing import Optional

import numpy as np
import torch

from mxtaltools.constants.space_group_feature_tensor import SG_FEATURE_TENSOR
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

import torch.nn.functional as F

from .base_set import BaseSet


class MolecularCrystal(BaseSet):
    def __init__(self, device,
                 energy_function: str,
                 dim: int = 12,
                 max_temperature: float = 10,
                 min_temperature: float = 0.01,
                 lj_turnover_pot: float = 10.0,
                 density_coeff: float = 0,
                 temperature_scaling_factor: float = 1,
                 temperature: float = 1.0,
                 temperature_conditioning: bool = False,
                 energy_clip: float = 100,
                 ellipsoid_scale: float = 1.0,
                 core_coeff: float = 1.0,
                 lj_coeff: float = 1.0,
                 lj_repulsion: float = 1.0,
                 molecule_conditioning: bool = False,
                 sg_conditioning: bool = False,
                 space_groups: Optional[list] = [2],
                 bounding_coeff: float = 1.0,
                 niggli_coeff: float = 1.0,
                 ):

        super(MolecularCrystal, self).__init__()
        self.device = device
        self.data_ndim = dim
        self.energy_function = energy_function
        self.energy_clip = energy_clip
        self.SG_FEATURE_TENSOR = SG_FEATURE_TENSOR.clone()  # store space group information

        self.ellipsoid_scale = ellipsoid_scale
        self.density_coeff = density_coeff
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.temperature_scaling_factor = temperature_scaling_factor
        self.temperature_conditioning = temperature_conditioning
        self.lj_turnover_pot = lj_turnover_pot  # energy above which to soften intermolecular repulsion
        self.lj_repulsion = lj_repulsion  #  for values < 1, shifts and softens the silu attraction
        self.core_coeff = core_coeff
        self.lj_coeff = lj_coeff
        self.bounding_coeff = bounding_coeff
        self.niggli_coeff = niggli_coeff
        self.molecule_conditioning = molecule_conditioning
        self.sg_conditioning=sg_conditioning
        self.space_groups = space_groups

        self.temperature = temperature  # for static temperature work

    def instantiate_crystals(self, x, mol_batch):
        crystal_batch = self.init_blank_crystal_batch(mol_batch)
        crystal_batch.gen_basis_to_cell_params(x)
        crystal_batch.box_analysis()
        return crystal_batch

    def analyze_crystal_batch(self, x, mol_batch, return_batch=False):  # x is gfn_outputs
        crystal_batch = self.instantiate_crystals(x, mol_batch)
        if self.energy_function not in ['ellipsoid_overlap',
                                        'silu_energy',
                                        'combo']:  # no need to actually build the crystal, this is much faster
            cluster_batch = crystal_batch
            lj_energy = torch.zeros(crystal_batch.num_graphs, device=self.device)
            silu_energy = torch.zeros_like(lj_energy)
        else:
            # for crystals at realistic densities, supercell_size=2 is sufficient. Very dense crystals will not be accurate, but they get punished later by the density energy term
            cluster_batch = crystal_batch.mol2cluster(cutoff=6,
                                                      supercell_size=2,
                                                      align_to_standardized_orientation=True)

            cluster_batch.construct_radial_graph(cutoff=6,
                                                 max_num_neighbors=100)
            #lj_energy, normed_lj_energy = cluster_batch.compute_LJ_energy()
            silu_energy = cluster_batch.compute_silu_energy(repulsion=self.lj_repulsion)

        if self.energy_function in ['ellipsoid_overlap', 'combo']:
            if not hasattr(self, 'ellipsoid_model'):
                cluster_batch.load_ellipsoid_model()
                self.ellipsoid_model = copy.deepcopy(cluster_batch.ellipsoid_model)
                self.ellipsoid_model = self.ellipsoid_model.to(self.device)
                self.ellipsoid_model.eval()
            # simplified ellipsoid energy testing
            _, _, _, _, _, _, normed_ellipsoid_overlap \
                = cluster_batch.compute_ellipsoidal_overlap(
                surface_padding=self.ellipsoid_scale,
                model=self.ellipsoid_model,
                return_details=True)
            ellipsoid_overlap = normed_ellipsoid_overlap.flatten()
        else:
            ellipsoid_overlap = torch.zeros_like(silu_energy)

        cluster_batch.silu_pot = silu_energy
        cluster_batch.lj_pot = silu_energy
        cluster_batch.ellipsoid_overlap = ellipsoid_overlap
        cluster_batch.niggli_overlap = self.compute_niggli_overlap(cluster_batch.cell_parameters())

        crystal_energy = self.generator_energy(cluster_batch)

        cluster_batch.gfn_energy = crystal_energy

        crystal_batch.gfn_energy = crystal_energy.cpu().detach()
        crystal_batch.silu_pot = silu_energy.cpu().detach()
        crystal_batch.lj_pot = silu_energy.cpu().detach()
        crystal_batch.ellipsoid_overlap = ellipsoid_overlap.cpu().detach()
        crystal_batch.niggli_overlap = cluster_batch.ellipsoid_overlap.cpu().detach()

        if return_batch:
            return crystal_energy, clean_batch(crystal_batch)
        else:
            return crystal_energy

    def compute_niggli_overlap(self, cell_parameters):
        """
        Compute the overlap g4 + g5 + g6, which must be >=0 for valid niggli cells
        :param cell_parameters:
        :return:
        """

        a, b, c, al, be, ga = cell_parameters[:, :6].split(1, dim=1)
        ab = a * b
        ac = a * c
        bc = b * c

        al_cos = torch.cos(al)
        be_cos = torch.cos(be)
        ga_cos = torch.cos(ga)

        return (ab * ga_cos + ac * be_cos + bc * al_cos).flatten()

    def generator_energy(self, cluster_batch):
        if cluster_batch.device != self.device:
            cluster_batch = cluster_batch.to(self.device)

        latents = cluster_batch.cell_params_to_gen_basis()
        if self.energy_function == 'latent_harmonic':
            # a trivial energy function, for testing
            if not hasattr(self, 'modes'):
                self.modes = -torch.ones((1, 12), device=self.device)
                self.crystal_modes = cluster_batch.latent_transform.inverse(self.modes,
                                                                            cluster_batch.sg_ind[:1],
                                                                            cluster_batch.radius[:1])
            crystal_energy = 0.5 * (latents - self.modes[0]).pow(2).sum(dim=1)
            # analytic Z = (2pi*T)^(d/2)
        elif self.energy_function == 'crystal_harmonic':
            # a trivial energy function, for testing
            cell_params = cluster_batch.cell_parameters()
            if not hasattr(self, 'modes'):
                self.modes = -torch.ones((1, 12), device=self.device)
                self.crystal_modes = cluster_batch.latent_transform.inverse(self.modes,
                                                                            cluster_batch.sg_ind[:1],
                                                                            cluster_batch.radius[:1])
            crystal_energy = 0.5 * (cell_params - self.crystal_modes[0]).pow(2).sum(dim=1)
            # analytic Z = (2pi*T)^(d/2)

        elif self.energy_function == 'latent_multiharmonic':
            if not hasattr(self, 'modes'):
                self.modes = torch.tensor(generate_modes(10, 12, 4.0, 3.0), device=self.device)
                self.crystal_modes = cluster_batch.latent_transform.inverse(self.modes,
                                                                            cluster_batch.sg_ind[:10],
                                                                            cluster_batch.radius[:10])

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

        elif self.energy_function == 'crystal_multiharmonic':
            if not hasattr(self, 'modes'):
                self.modes = torch.tensor(generate_modes(10, 12, 4.0, 3.0), device=self.device)
                self.crystal_modes = cluster_batch.latent_transform.inverse(self.modes,
                                                                            cluster_batch.sg_ind[:10],
                                                                            cluster_batch.radius[:10])

            diffs = latents[:, None, :] - self.modes[None, :, :]
            sqdist = (diffs ** 2).sum(dim=-1)  # (B, K)
            exponent = -0.5 * sqdist  # (B, K)
            crystal_energy = -torch.logsumexp(exponent, dim=1)  # (B,)

        elif self.energy_function == 'ellipsoid_overlap':
            density_energy = self.density_penalty(cluster_batch.packing_coeff)
            core_energy = self.core_energy_penalty(cluster_batch.ellipsoid_overlap)
            crystal_energy = self.core_coeff * core_energy + self.density_coeff * density_energy

        elif self.energy_function == 'silu_energy':
            density_energy = self.density_penalty(cluster_batch.packing_coeff)
            lj_energy = self.soften_LJ_energy(cluster_batch.silu_pot) / cluster_batch.num_atoms
            crystal_energy = self.lj_coeff * lj_energy + self.density_coeff * density_energy

        elif self.energy_function == 'combo':
            density_energy = self.density_penalty(cluster_batch.packing_coeff)
            lj_energy = self.soften_LJ_energy(cluster_batch.silu_pot) / cluster_batch.num_atoms
            core_energy = self.core_energy_penalty(cluster_batch.ellipsoid_overlap)
            crystal_energy = self.lj_coeff * lj_energy + self.core_coeff * core_energy + self.density_coeff * density_energy

        else:
            assert False, f'{self.energy_function} not implemented'

        if self.energy_function in ['ellipsoid_overlap', 'silu_energy', 'combo']:
            niggli_energy = F.relu(-cluster_batch.niggli_overlap)**2  # punish negative overlaps
        else:
            niggli_energy = torch.zeros_like(crystal_energy)

        bounding_energy = (F.relu(latents - 6)**2 + F.relu(-(latents + 6))**2).sum(dim=-1)  # discourage exploration beyond clip range
        total_energy = crystal_energy + bounding_energy * self.bounding_coeff + niggli_energy* self.niggli_coeff
        return self.soft_clip(total_energy,
                              self.energy_clip)  # softly bound from above  #crystal_energy.clip(min=-self.energy_clip, max=self.energy_clip)

    def core_energy_penalty(self, ellipsoid_overlap):
        return ellipsoid_overlap ** 2 + ellipsoid_overlap

    def density_penalty(self, packing_coeff):
        """
        draw crystals into the physically reasonable region
        :param packing_coeff:
        :return:
        """
        return F.relu(-(torch.log(packing_coeff) - np.log(0.5))) ** 2 + F.relu(packing_coeff - 0.9) ** 2

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

    def soften_LJ_energy(self, lj_energy, clip: Optional[float] = None):
        # soften the repulsion
        softened_energy = lj_energy.clone()
        high_bools = softened_energy > self.lj_turnover_pot
        # softened_energy[high_bools] = self.lj_turnover_pot + torch.log(softened_energy[high_bools] + 1 - self.lj_turnover_pot)
        delta = softened_energy[high_bools] - self.lj_turnover_pot + 1
        softened_energy[high_bools] = self.lj_turnover_pot + delta ** 0.9
        if clip is not None:
            softened_energy = softened_energy.clip(max=clip)

        return softened_energy

    def soft_clip(self, y, clip_value):
        new_y = y.clone()
        #delta = new_y[y>clip_value] - clip_value + 1
        new_y[y > clip_value] = clip_value + torch.log(y[y > clip_value] + 1 - clip_value)
        #new_y[y>clip_value] = clip_value + delta ** (0.5)
        return new_y

    def init_blank_crystal_batch(self, mol_batch):  # todo no possible way this is the most efficient way to do this

        ones3 = torch.ones(3, device=self.device)
        zeros1 = torch.zeros(1, device=self.device)

        if self.energy_function in ['ellipsoid_overlap', 'combo']:
            overlap_tensor = torch.zeros(1, device=self.device)
        else:
            overlap_tensor = None

        if self.sg_conditioning:
            sgs = mol_batch.sg_ind
        else:
            sgs = [self.space_groups[0] for _ in range(mol_batch.num_graphs)]

        crystal_batch = collate_data_list([MolCrystalData(
            molecule=mol_batch[ind].clone(),  # must be cloned
            sg_ind=sgs[ind],
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
            niggli_overlap=zeros1,
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
                                sg_inds: torch.tensor = None,
                                ):

        conds = []
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
            log_T_tensor = torch.log10(torch.ones((mol_batch.num_graphs, 1), device=mol_batch.device) * self.temperature)

        if self.molecule_conditioning:
            mol_embedding = mol_batch.embedding.flatten(1, 2)
            conds.append(mol_embedding)

        if sg_inds is not None:
            sg_to_sample = sg_inds.clone()
        else:
            sg_to_sample = torch.tensor(np.random.choice(self.space_groups, mol_batch.num_graphs, replace=True)).to(
                mol_batch.device)

        if self.sg_conditioning:
            conds.append(self.SG_FEATURE_TENSOR[sg_to_sample])

        return (log_T_tensor.flatten(), sg_to_sample,
                torch.cat(conds, dim=1) if len(conds) > 0 else torch.zeros_like(log_T_tensor))

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

    return batch
