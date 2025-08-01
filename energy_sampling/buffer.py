from typing import Optional

import torch
import numpy as np

from mxtaltools.crystal_building.crystal_latent_transforms import compute_niggli_overlap
from mxtaltools.dataset_utils.utils import collate_data_list

from utils import compute_sample_overlap


class CrystalReplayBuffer:
    def __init__(self, buffer_size,
                 device,
                 energy_function,
                 batch_size,
                 beta=1.0,
                 rank_weight=1e-2,
                 prioritized=None,
                 keep_initial_samples: bool = False,
                 gpu_available: bool = False,
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
        self.gpu_available = gpu_available
        self.diversity_coeff = diversity_coeff

    def add(self,
            data_list,
            diversity_cutoff: float = 1.0):
        with torch.no_grad():
            if self.dataset is None:
                self.init_fresh_dataset(data_list)

            else:
                new_data_batch = collate_data_list(data_list)

                # do not take samples with bad overlaps
                # these could be 'flipped' to their valid cell form and added, but I don't have the transform
                # for the molecule positions, only for the cell (angle = pi-angle)
                a, b, c = new_data_batch.cell_lengths.split(1, dim=1)
                al, be, ga = new_data_batch.cell_angles.split(1, dim=1)
                _, _, _, _, _, _, overlap = compute_niggli_overlap(a, b, c, al, be, ga)
                bad_inds = torch.argwhere(overlap.flatten() < 0).flatten().tolist()
                data_list = [elem for ind, elem in enumerate(data_list) if ind not in bad_inds]
                if len(data_list) > 0:
                    new_data_batch = collate_data_list(data_list)

                    new_x_tensor = new_data_batch.cell_params_to_gen_basis().to('cuda' if self.gpu_available else 'cpu')
                    scores = self.energy_function.prebuilt_sample_to_reward(new_data_batch,
                                                                            temperature=torch.ones(len(new_data_batch))
                                                                            ).cpu().detach().numpy()
                    scores_list = list(scores)

                    ref_x_tensor = torch.stack(self.x_list).to('cuda' if self.gpu_available else 'cpu')
                    min_buffer_dist = torch.cdist(ref_x_tensor, new_x_tensor).amin(0)
                    new_x_tensor = new_x_tensor.cpu()

                    far_enough = (min_buffer_dist >= diversity_cutoff).cpu().detach().numpy()
                    existing_rewards = np.array(self.rewards_list)
                    rewards_cutoff = np.amin(existing_rewards) - np.ptp(existing_rewards) * 0.1
                    good_enough = scores >= rewards_cutoff
                    new_x_inds_to_keep = np.argwhere(far_enough * good_enough).flatten().tolist()
                    data_list_to_add = [data_list[ind] for ind in new_x_inds_to_keep]

                    if len(data_list_to_add) > 0:
                        self.dataset.extend(list(data_list_to_add))
                        self.x_list.extend([new_x_tensor[ind] for ind in new_x_inds_to_keep])
                        self.rewards_list.extend([scores_list[ind] for ind in new_x_inds_to_keep])

            if len(self) > self.buffer_size:  # pare down buffer
                self.truncate_buffer()

            assert len(self.dataset) == len(self.x_list) == len(self.rewards_list)

    def init_fresh_dataset(self, data_list):
        self.dataset = list(data_list)  # I think this is memory safe and faster #copy.deepcopy(data_list)
        dataset_batch = collate_data_list(self.dataset)
        x_tensor = dataset_batch.cell_params_to_gen_basis()
        scores = self.energy_function.prebuilt_sample_to_reward(dataset_batch, temperature=torch.ones(len(self)))
        self.x_list = [x_tensor[i] for i in range(x_tensor.shape[0])]
        self.rewards_list = list(scores.flatten().cpu().detach().numpy())
        self.original_dataset_inds = list(np.arange(len(self.dataset)))

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

    def __len__(self):
        if self.dataset is None:
            return 0
        else:
            return len(self.dataset)

    def sample_indices(self, batch_size, replace: bool, diversity_coeff: float):
        inds = np.random.choice(len(self),
                                size=batch_size,
                                replace=replace,
                                p=self.get_sampler_weights(diversity_coeff=diversity_coeff,
                                                           ))
        return inds

    def get_sampler_weights(self,
                            diversity_coeff,
                            eps: float = 1e-6):
        scores = np.array(self.rewards_list)
        if diversity_coeff > 0:
            x_tensor = torch.stack(self.x_list).to('cuda' if self.gpu_available else 'cpu')
            if len(x_tensor) > 5000:
                subsample_inds = np.random.choice(len(self), 5000, replace=False)
                scores -= diversity_coeff * ((compute_sample_overlap(x_tensor[subsample_inds].half(),
                                                                     x_tensor.half(),
                                                                     agg='sum')).cpu().detach().numpy() - 1)  # subtract self contribution
            else:
                scores -= diversity_coeff * ((compute_sample_overlap(x_tensor.half(),
                                                                     agg='sum')).cpu().detach().numpy() - 1)  # subtract self contribution

        if self.prioritized == 'rank':
            ranks = np.argsort(np.argsort(-1 * scores))
            weights_i = 1.0 / (self.rank_weight * len(scores) + ranks)
        elif self.prioritized == 'boltzmann':
            logits = scores / self.beta
            logits -= np.max(logits)  # subtract max for stability
            weights_i = np.nan_to_num(np.exp(logits)) + eps  # all samples need nonzero probability
        else:  # uniform weights
            weights_i = np.ones(len(scores))

        return weights_i / np.sum(weights_i)  # enforce explicit normalization

    @torch.no_grad()
    def sample(self,
               temperature: Optional[torch.tensor] = None,
               return_conditioning: Optional[bool] = False,
               override_batch: Optional[int] = None,
               return_preload: Optional[bool] = False):

        assert return_conditioning or (
                temperature is not None), "Must provide temperature or generate it here with return_conditioning=True"

        if override_batch is not None:
            batch_size = override_batch
        else:
            batch_size = self.batch_size

        # manual dataloader
        if return_preload:
            rand_inds = self.original_dataset_inds
        else:
            if batch_size > len(self):
                rand_inds = np.arange(len(self))
                missing_len = batch_size - len(self)
                rand_inds = np.concatenate(
                    [rand_inds, self.sample_indices(missing_len, replace=True, diversity_coeff=self.diversity_coeff)])
            else:
                rand_inds = self.sample_indices(batch_size, replace=False, diversity_coeff=self.diversity_coeff)

        sample = collate_data_list([self.dataset[ind] for ind in rand_inds])

        T_tensor, sg_inds, condition = self.energy_function.get_conditioning_tensor(sample)
        sample.sg_ind = sg_inds
        temperature = 10 ** T_tensor  # first dimension is the log temperature
        with torch.no_grad():
            reward = self.energy_function.prebuilt_sample_to_reward(
                sample, temperature)  # recompute reward in case parameters have changed

        if return_conditioning:
            return sample.cell_params_to_gen_basis(), reward, sample, condition
        else:
            return sample.cell_params_to_gen_basis(), reward, sample
