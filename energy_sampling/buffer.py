from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.loader import DataLoader

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
                 max_z_prime: int = 1,
                 buffer_dist_cutoff: float = 0.25,
                 noised_buffer_length: int = 100000,
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
        self.max_z_prime = max_z_prime
        self.buffer_dist_cutoff = buffer_dist_cutoff
        self.staging_buffer = []
        self.noised_buffer_length = noised_buffer_length
        self.noised_rewards = torch.zeros(self.noised_buffer_length, dtype=torch.float32, device='cpu')
        self.noised_samples = torch.zeros((self.noised_buffer_length, 6+6*max_z_prime), dtype=torch.float32, device='cpu')
        self.noised_ptr = 0
        self.noised_size = 0

    @torch.no_grad()
    def add_to_staging(self, data_list=None, data_batch=None):
        if len(self.staging_buffer) < len(
                self):  # don't stage a crazy number of samples - downstream cost becomes too high
            if data_list is None and data_batch is not None:
                data_list = data_batch.batch_to_list()

            self.staging_buffer.extend(data_list)

    @torch.no_grad()
    def add(self,
            data_list=None):
        if self.dataset is None:
            self.init_fresh_dataset(data_list)
        else:
            self.add_samples_to_dataset(data_list)

        if len(self) > self.buffer_size:  # pare down buffer
            self.truncate_buffer()

        assert len(self.dataset) == len(self.x_list) == len(self.rewards_list)

        # self.init_loader()  # never used

    def incorporate_staging_buffer(self):
        if len(self.staging_buffer) > 0:  # will fail if staging buffer is empty
            self.add_samples_to_dataset(self.staging_buffer, skip_staging=True)
            self.staging_buffer = []

            if len(self) > self.buffer_size:  # pare down buffer
                self.truncate_buffer()

            assert len(self.dataset) == len(self.x_list) == len(self.rewards_list)

        # self.init_loader()  # never used

    def add_samples_to_dataset(self, data_list, skip_staging: bool = False):
        # batch samples
        if not skip_staging:
            if len(self.staging_buffer) > 0:  # include staged samples
                data_list.extend(self.staging_buffer)
                self.staging_buffer = []

        data_batch = collate_data_list(data_list, max_z_prime=self.max_z_prime)
        new_latents = data_batch.latent_params()
        new_sgs = data_batch.sg_ind
        # get new samples rewards
        scores = self.energy_function.prebuilt_sample_to_reward(
            data_batch,
            temperature=torch.ones(data_batch.num_graphs) * self.energy_function.temperature)

        # enforce reasonable standards for consideration in the buffer
        # score_cut = np.quantile(self.rewards_list, 0.5)
        score_cut = max(self.reward_clip, np.amin(self.rewards_list))  # the lowest reward in our dynamical range
        packing_coeffs = data_batch.packing_coeff.cpu().detach().numpy()
        good_inds = [ind for ind in range(len(data_list)) if
                     (data_list[ind].reduction_en <= 1e-3) and (scores[ind] > score_cut) and (
                             packing_coeffs[ind] > 0.55) and (packing_coeffs[ind] < 0.95)]
        # add anything reasonable
        if len(good_inds) > 0:
            data_to_add = [data_list[ind] for ind in good_inds]
            self.dataset.extend(data_to_add)
            good_scores = scores[torch.tensor(good_inds, dtype=torch.long)]
            self.x_list.extend([new_latents[i] for i in good_inds])
            self.rewards_list.extend(good_scores.flatten().cpu().detach().numpy())
            self.sg_list.extend([new_sgs[i] for i in good_inds])

    def init_fresh_dataset(self, data_list):
        self.dataset = list(data_list)  # I think this is memory safe and faster #copy.deepcopy(data_list)
        for elem in self.dataset:  # have to do this now because collation is a mess
            del elem.fingerprint, elem.smiles, elem.mol_ind, elem.identifier, (
                elem.aunit_batch), elem.skip_box_analysis, elem.cocrystal, elem.symmetry_operators

        dataset_batch = collate_data_list(self.dataset, max_z_prime=self.max_z_prime)
        x_tensor = dataset_batch.latent_params()
        rewards = self.energy_function.prebuilt_sample_to_reward(
            dataset_batch,
            temperature=torch.ones(len(self)) * self.energy_function.temperature)

        if self.energy_function.reward_range is not None:
            # reward scaling is temperature dependent
            self.energy_function.set_reward_clip(rewards)
            self.reward_clip = -self.energy_function.energy_clip / self.energy_function.temperature
            # recompute with new clip
            rewards = self.energy_function.prebuilt_sample_to_reward(
                dataset_batch,
                temperature=torch.ones(len(self)) * self.energy_function.temperature)

        self.x_list = [x_tensor[i] for i in range(x_tensor.shape[0])]
        self.rewards_list = list(rewards.flatten().cpu().detach().numpy())
        self.original_dataset_inds = list(np.arange(len(self.dataset)))
        self.sg_list = list(dataset_batch.sg_ind.cpu())

    def truncate_buffer(self):
        """
        1 - keep initial states
        2 - bottom-up energy greedy selection
        3 - clustering
        :return:
        """
        # get descriptors
        device = self.device
        x_tensor = torch.stack(self.x_list).to(device)
        e_tensor = -torch.nan_to_num(torch.tensor(self.rewards_list, device=device)) * self.energy_function.temperature

        # define cutoffs
        d_cut = self.buffer_dist_cutoff
        e_cut = self.energy_function.energy_clip

        if self.keep_initial_samples:
            max_new_samples = self.buffer_size - len(self.original_dataset_inds)
        else:
            max_new_samples = self.buffer_size

        inds_to_keep = self.bottom_up_cluster(x_tensor, e_tensor, d_cut, e_cut, max_new_samples)

        if self.keep_initial_samples:
            orig_dataset_ind_tensor = torch.tensor(self.original_dataset_inds, device=self.device, dtype=torch.long)
            keep_inds_tensor = torch.as_tensor(inds_to_keep)
            combined = torch.unique(torch.cat([keep_inds_tensor,
                                               orig_dataset_ind_tensor]))
            # Protect originals by giving them artificially low energy ranks
            energies = e_tensor[combined].clone()
            mask_orig = torch.isin(combined, orig_dataset_ind_tensor)
            energies[mask_orig] -= 1e6  # will always be kept
            sorted_combined = combined[torch.argsort(energies)]
            inds_to_keep = sorted_combined[:max(len(self.original_dataset_inds), self.buffer_size)]
        else:
            # Sort by (modified) energy and truncate
            if len(inds_to_keep) > self.buffer_size:
                sorted_by_e = inds_to_keep[torch.argsort(e_tensor[inds_to_keep])]
                inds_to_keep = sorted_by_e[:self.buffer_size]

        inds_to_keep = inds_to_keep.tolist()  # for convenience

        # Fill with random samples if needed
        n_keep = len(inds_to_keep)
        if n_keep < self.buffer_size:
            # Candidates = all indices not already kept
            all_inds = set(range(len(self.x_list)))
            remaining = list(all_inds - set(inds_to_keep))

            n_fill = self.buffer_size - n_keep
            if n_fill > 0 and len(remaining) > 0:
                fill_inds = np.random.choice(remaining, size=min(n_fill, len(remaining)), replace=False)
                inds_to_keep.extend(fill_inds.tolist())

        self.dataset = [self.dataset[ind] for ind in inds_to_keep]
        self.rewards_list = [self.rewards_list[ind] for ind in inds_to_keep]
        self.x_list = [self.x_list[ind] for ind in inds_to_keep]
        self.sg_list = [self.sg_list[ind] for ind in inds_to_keep]

    def __len__(self):
        if self.dataset is None:
            return 0
        else:
            return len(self.dataset)

    @torch.no_grad()
    def bottom_up_cluster(self, xx, e, d_cut, e_cut, max_new_samples: int):

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = self.device

        # Sort by energy ascending
        sort_inds = torch.argsort(e.to(device))
        xx_sorted = xx.to(device)[sort_inds]
        e_sorted = e.to(device)[sort_inds]
        mask = e_sorted < e_cut

        xx_sorted_cuda = xx_sorted.to(device)
        blocked = torch.zeros(len(xx_sorted), dtype=torch.bool, device=device)
        keep = torch.zeros(len(xx_sorted), dtype=bool, device=device)
        d_cut_squared = d_cut * d_cut
        for i in range(len(xx_sorted)):
            if not mask[i]:
                break

            if blocked[i]:
                continue

            keep[i] = True
            if torch.sum(keep) == max_new_samples:
                break

            drow = ((xx_sorted_cuda - xx_sorted_cuda[i, None, :]) ** 2).sum(-1)  # faster, skips sqrt
            nearby = drow < d_cut_squared
            blocked |= nearby

        keep_inds = sort_inds[keep]

        return keep_inds.cpu()

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
               standardize_orientations: Optional[bool] = False,
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

        sample_batch = collate_data_list([self.dataset[ind] for ind in rand_inds],
                                         max_z_prime=self.max_z_prime, exclude_keys=['symmetry_operators'])

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

        # if standardize_orientations:
        #     assert not randomize_orientations
        #     sample_batch.orient_molecule(mode='std',
        #                                  correct_orientation=True)

        T_tensor, sg_inds, condition = self.energy_function.get_conditioning_tensor(
            sample_batch, sg_inds=sample_batch.sg_ind, z_primes=sample_batch.z_prime)

        sample_batch.reset_sg_info(sg_inds)
        temperature = 10 ** T_tensor  # first dimension is the log temperature
        reward = self.energy_function.prebuilt_sample_to_reward(
            sample_batch, temperature)  # recompute reward in case parameters have changed

        latents = sample_batch.latent_params()
        return latents, reward, sample_batch, condition

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

    def sample_mol_unconditional_prior(self, sg_inds, noise: Optional[float] = None):
        """
        sample from the buffer, unconditional on molecules, conditional on space groups
        then optionally noise
        :param sg_inds:
        :return:
        """
        assert False, ("Unconditional prior sampling needs to be rewritten, as "
                       "the latent space is no longer strictly std normal")
        # samples = torch.zeros((len(sg_inds), 12), dtype=torch.float32)
        # sgs_to_sample = torch.unique(sg_inds).tolist()
        # sg_buffer = torch.tensor(self.sg_list)
        # x_tensor = torch.stack(self.x_list).to(self.device)
        #
        # for sg in sgs_to_sample:
        #     sample_mask = sg_inds == sg
        #     mask = (sg_buffer == sg)
        #     relevant = x_tensor[mask]
        #
        #     n = sample_mask.sum()
        #     rand_idx = torch.randint(0, relevant.size(0), (n,), device=self.device)
        #     samples[sample_mask] = relevant[rand_idx]
        #
        # if noise is not None:
        #     samples += torch.randn_like(samples) * noise
        #
        # return samples.clip(min=-6, max=6)

    def add_to_noised(self, rewards, samples):
        rewards = rewards.detach().to(self.device)
        samples = samples.detach().to(self.device)

        B = rewards.shape[0]
        ptr = self.noised_ptr
        max_size = self.noised_buffer_length

        end = ptr + B

        # Case 1: no wraparound
        if end <= max_size:
            self.noised_rewards[ptr:end] = rewards
            self.noised_samples[ptr:end] = samples
        else:
            # Case 2: wraparound
            first = max_size - ptr
            self.noised_rewards[ptr:] = rewards[:first]
            self.noised_samples[ptr:] = samples[:first]

            second = end % max_size
            self.noised_rewards[:second] = rewards[first:]
            self.noised_samples[:second] = samples[first:]

        # Update pointer & size
        self.noised_ptr = end % max_size
        self.noised_size = min(self.noised_size + B, max_size)

    def sample_from_noised(self, num_samples):
        if self.noised_size == 0:
            raise RuntimeError("No noised samples available in buffer.")

        idx = torch.randint(self.noised_size, (num_samples,), device=self.device)

        rewards = self.noised_rewards[idx]
        samples = self.noised_samples[idx]

        return rewards, samples


def collate_fn(data_list):
    return collate_data_list(data_list, exclude_unit_cell=True)
