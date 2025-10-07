from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.loader import DataLoader

from mxtaltools.crystal_building.crystal_latent_transforms import compute_niggli_overlap
from mxtaltools.dataset_utils.utils import collate_data_list
from utils import compute_sample_overlap, iter_forever


class CrystalReplayBuffer:
    def __init__(self, buffer_size,
                 device,
                 energy_function,
                 batch_size,
                 beta=1.0,
                 rank_weight=1e-2,
                 prioritized=None,
                 keep_initial_samples: bool = False,
                 diversity_coeff: float = 0.0,
                 ):
        self.buffer_size = buffer_size
        self.prioritized = prioritized
        self.device = device
        self.batch_size = batch_size
        self.dataset = None
        self.buffer_idx = 0
        self.buffer_full = False
        self.energy_function = energy_function
        self.beta = beta
        self.rank_weight = rank_weight
        self.beta = beta
        self.keep_initial_samples = keep_initial_samples  # never delete originally loaded dataset
        self.rewards_list = None
        self.x = None
        self.diversity_check_size = 1000
        self.original_dataset_inds = None
        self.diversity_coeff = diversity_coeff

    def add(self,
            data_list,
            diversity_cutoff: float = 1.0):
        with torch.no_grad():
            if self.dataset is None:
                self.init_fresh_dataset(data_list)
                self.init_loader()

            else:
                assert False, "NOTE: loader not currently set up for on-the-fly dataset addition!"
                new_data_batch = collate_data_list(data_list)

                # do not take samples with bad overlaps
                # these could be 'flipped' to their valid cell form and added, but I don't have the transform
                # for the molecule positions, only for the cell (angle = pi-angle)
                a, b, c = new_data_batch.cell_lengths.split(1, dim=1)
                al, be, ga = new_data_batch.cell_angles.split(1, dim=1)
                _, _, _, _, _, _, overlap = compute_niggli_overlap(a, b, c, al, be, ga)
                bad_inds = torch.argwhere(overlap.flatten() < 0).flatten().tolist()

                # also, hard filter samples which are above or below our density cutoffs
                bad_inds.extend(torch.argwhere(0.55 < new_data_batch.packing_coeff).flatten().tolist())
                bad_inds.extend(torch.argwhere(new_data_batch.packing_coeff > 0.95).flatten().tolist())

                # also strictly reject unbound states
                bad_inds.extend(torch.argwhere(new_data_batch.lj_pot > 0))

                bad_inds = list(set(bad_inds))
                data_list = [elem for ind, elem in enumerate(data_list) if ind not in bad_inds]

                if len(data_list) > 0:
                    new_data_batch = collate_data_list(data_list)

                    new_x_tensor = new_data_batch.latent_params().to(self.device)
                    scores = self.energy_function.prebuilt_sample_to_reward(new_data_batch,
                                                                            temperature=torch.ones(len(new_data_batch))
                                                                            ).cpu().detach().numpy()
                    scores_list = list(scores)

                    ref_x_tensor = torch.stack(self.x_list).to(self.device)
                    min_buffer_dist = torch.cdist(ref_x_tensor, new_x_tensor).amin(0)
                    new_x_tensor = new_x_tensor.cpu()
                    sg_list = new_data_batch.sg_ind.cpu()

                    far_enough = (min_buffer_dist >= diversity_cutoff).cpu().detach().numpy()
                    existing_rewards = np.array(self.rewards_list)
                    rewards_cutoff = np.quantile(existing_rewards, 0.2)
                    good_enough = scores >= rewards_cutoff
                    new_x_inds_to_keep = np.argwhere(far_enough * good_enough).flatten().tolist()
                    data_list_to_add = [data_list[ind] for ind in new_x_inds_to_keep]

                    if len(data_list_to_add) > 0:
                        self.dataset.extend(list(data_list_to_add))
                        self.x_list.extend([new_x_tensor[ind] for ind in new_x_inds_to_keep])
                        self.rewards_list.extend([scores_list[ind] for ind in new_x_inds_to_keep])
                        self.sg_list.extend([sg_list[ind] for ind in new_x_inds_to_keep])

            if len(self) > self.buffer_size:  # pare down buffer
                self.truncate_buffer()

            assert len(self.dataset) == len(self.x_list) == len(self.rewards_list)

    def init_fresh_dataset(self, data_list):
        self.dataset = list(data_list)  # I think this is memory safe and faster #copy.deepcopy(data_list)
        dataset_batch = collate_data_list(self.dataset)
        x_tensor = dataset_batch.latent_params()
        scores = self.energy_function.prebuilt_sample_to_reward(dataset_batch, temperature=torch.ones(len(self)))
        self.x_list = [x_tensor[i] for i in range(x_tensor.shape[0])]
        self.rewards_list = list(scores.flatten().cpu().detach().numpy())
        self.original_dataset_inds = list(np.arange(len(self.dataset)))
        self.sg_list = list(dataset_batch.sg_ind.cpu())

    def truncate_buffer(self, override_buffer_size=None):
        if override_buffer_size is not None:
            self.buffer_size = override_buffer_size

        inds_to_keep = self.sample_indices(self.buffer_size, replace=False, diversity_coeff=self.diversity_coeff)
        if self.keep_initial_samples:
            inds_to_keep = list(set(list(inds_to_keep) + self.original_dataset_inds))[:self.buffer_size]
        else:
            inds_to_keep = list(set(inds_to_keep))
        self.dataset = [self.dataset[ind] for ind in inds_to_keep]
        self.rewards_list = [self.rewards_list[ind] for ind in inds_to_keep]
        self.x_list = [self.x_list[ind] for ind in inds_to_keep]
        self.sg_list = [self.sg_list[ind] for ind in inds_to_keep]

    def __len__(self):
        if self.dataset is None:
            return 0
        else:
            return len(self.dataset)

    def sample_indices(self, batch_size,
                       replace: bool,
                       diversity_coeff: float,
                       override_method: Optional[str] = None):
        inds = np.random.choice(len(self),
                                size=batch_size,
                                replace=replace,
                                p=self.get_sampler_weights(diversity_coeff=diversity_coeff,
                                                           override_method=override_method,
                                                           ))
        return inds

    @torch.no_grad()
    def get_sampler_weights(self,
                            diversity_coeff,
                            eps: float = 1e-6,
                            override_method: Optional[str] = None):
        if override_method is not None:
            method = override_method
        else:
            method = self.prioritized

        if method is not None:
            scores = np.array(self.rewards_list)
            if diversity_coeff > 0:
                x_tensor = torch.stack(self.x_list).to(self.device)
                if len(x_tensor) > 1000:
                    subsample_inds = np.random.choice(len(self), 1000, replace=False)
                    scores -= diversity_coeff * ((compute_sample_overlap(x_tensor[subsample_inds].float(),
                                                                         x_tensor.float(),
                                                                         ga=0.01,
                                                                         agg='sum')).cpu().detach().numpy() - 1)  # subtract self contribution
                else:
                    scores -= diversity_coeff * ((compute_sample_overlap(x_tensor.float(),
                                                                         ga=0.01,
                                                                         agg='sum')).cpu().detach().numpy() - 1)  # subtract self contribution

        if method == 'rank':
            ranks = np.argsort(np.argsort(-1 * scores))
            weights_i = 1.0 / (self.rank_weight * len(scores) + ranks)
        elif method == 'boltzmann':
            logits = self.beta * scores
            logits -= np.max(logits)  # subtract max for stability
            weights_i = np.nan_to_num(np.exp(logits)) + eps  # all samples need nonzero probability
        else:  # uniform weights
            weights_i = np.ones(len(self.x_list))

        return weights_i / np.sum(weights_i)  # enforce explicit normalization

    @torch.no_grad()
    def sample(self,
               override_batch: Optional[int] = None,
               return_preload: Optional[bool] = False,
               override_sampler: Optional[str] = None,
               randomize_orientations: Optional[bool] = False,
               ):

        if override_batch is not None:
            self.batch_size = override_batch

        # manual dataloader
        if return_preload:
            rand_inds = self.original_dataset_inds
        else:
            if self.batch_size > len(self):
                rand_inds = np.arange(len(self))
                missing_len = self.batch_size - len(self)
                rand_inds = np.concatenate(
                    [rand_inds, self.sample_indices(missing_len, replace=True, diversity_coeff=self.diversity_coeff,
                                                    override_method=override_sampler)])
            else:
                rand_inds = self.sample_indices(self.batch_size, replace=False, diversity_coeff=self.diversity_coeff,
                                                override_method=override_sampler)

        sample_batch = collate_data_list([self.dataset[ind] for ind in rand_inds])

        if randomize_orientations:
            # this is a form of sample augmentation, where we rotate the molecule and its applied orientation
            # in order to construct the identical crystal, but with a distinct conditioning vector & sample
            # also rotate the embedding vector which is passed to conditioning
            random_rotations = torch.tensor(
                Rotation.random(num=sample_batch.num_graphs).as_matrix(),
                device=sample_batch.device, dtype=torch.float32)
            sample_batch.orient_molecule(mode='std')
            sample_batch.orient_molecule(mode='random',  # important that the rotation is applied *from* the standard
                                         include_inversion=False,
                                         correct_orientation=True,
                                         override_random_rotations=random_rotations)
            sample_batch.embedding = sample_batch.rotate_embedding(random_rotations)
            """
            sample_batch.orient_molecule(mode='std')
            sample_batch.orient_molecule(mode='random',  # important that the rotation is applied *from* the standard
                                         include_inversion=False,
                                         correct_orientation=True,
                                         override_random_rotations=random_rotations)
            aa = sample_batch.analyze(['lj'], std_orientation=False, cutoff=10)
            print(((aa['lj']-sample_batch.lj_pot).abs()/sample_batch.lj_pot.abs()).mean())
            # test to make sure this is working
            # important that we standardize before applying the orientation adjustment!!!
            """

        T_tensor, sg_inds, condition, z_primes = self.energy_function.get_conditioning_tensor(sample_batch,
                                                                                              sg_inds=sample_batch.sg_ind)
        sample_batch.sg_ind = sg_inds
        sample_batch.z_prime = z_primes
        temperature = 10 ** T_tensor  # first dimension is the log temperature
        reward = self.energy_function.prebuilt_sample_to_reward(
            sample_batch, temperature)  # recompute reward in case parameters have changed

        return sample_batch.latent_params(), reward, sample_batch, condition

    def init_loader(self):
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=0,  # os.cpu_count() - 2,  # use all but two available CPUs
            persistent_workers=False,  # True,
            drop_last=True,
            pin_memory=True,
            # prefetch_factor=4,
        )
        self._loader_iter = iter_forever(self.loader)

    def adjust_batch_size(self, new_batch_size: int):
        self.loader.batch_sampler.batch_size = new_batch_size
        self._loader_iter = iter_forever(self.loader)

    @torch.no_grad()
    def recompute_silu_pot(self, batch_size, lj_repulsion, device):
        """when updating the silu repulsive term,
        we have to rebuild and re-analyze the full dataset"""
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            drop_last=False
        )
        silus = []

        for crystal_batch in loader:
            crystal_batch = crystal_batch.to(device)
            crystal_batch.box_analysis()
            cluster_batch = crystal_batch.mol2cluster(cutoff=6,
                                                      supercell_size=10,
                                                      std_orientation=True)

            cluster_batch.construct_radial_graph(cutoff=6)

            _ = cluster_batch.compute_LJ_energy()
            silu_energy = cluster_batch.compute_silu_energy(
                repulsion=lj_repulsion,
            )
            silus.extend(silu_energy.cpu())

        silus = torch.tensor(silus)
        for ind, elem in enumerate(self.dataset):
            elem.silu_pot = torch.ones(1) * silus[ind]

        scores = self.energy_function.prebuilt_sample_to_reward(self.dataset, temperature=torch.ones(len(self)))
        self.rewards_list = list(scores.flatten().cpu().detach().numpy())

    def sample_mol_unconditional_prior(self, sg_inds, noise: Optional[float] = None):
        """
        sample from the buffer, unconditional on molecules, conditional on space groups
        then optionally noise
        :param sg_inds:
        :return:
        """
        samples = torch.zeros((len(sg_inds), 12), dtype=torch.float32)
        sgs_to_sample = torch.unique(sg_inds).tolist()
        sg_buffer = torch.tensor(self.sg_list)
        x_tensor = torch.stack(self.x_list).to(self.device)

        for sg in sgs_to_sample:
            sample_mask = sg_inds == sg
            mask = (sg_buffer == sg)
            relevant = x_tensor[mask]

            n = sample_mask.sum()
            rand_idx = torch.randint(0, relevant.size(0), (n,), device=self.device)
            samples[sample_mask] = relevant[rand_idx]

        if noise is not None:
            samples += torch.randn_like(samples) * noise

        return samples.clip(min=-6, max=6)


def collate_fn(data_list):
    return collate_data_list(data_list, exclude_unit_cell=True)
