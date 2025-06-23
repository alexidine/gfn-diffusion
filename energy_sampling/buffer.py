import copy
import gc
from typing import Optional

import torch
import numpy as np
from mxtaltools.dataset_utils.utils import collate_data_list


class CrystalReplayBuffer():
    def __init__(self, buffer_size,
                 device,
                 energy_function,
                 batch_size,
                 beta=1.0,
                 rank_weight=1e-2,
                 prioritized=None,
                 keep_initial_samples: bool = True):
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
        self.scores_np_list = None
        self.x = None
        self.diversity_check_size = 1000
        self.original_dataset_inds = None

    def add(self, data_list, filter_diversity: bool = True, diversity_cutoff: float = 1.0):
        with torch.no_grad():
            if self.dataset is None:
                self.dataset = list(data_list)  # I think this is memory safe and faster #copy.deepcopy(data_list)
                self.x = collate_data_list(self.dataset).cell_params_to_gen_basis()
                self.scores_np_list = list(
                    self.energy_function.prebuilt_sample_to_reward(self.dataset, temperature=torch.ones(
                        len(self))).detach().cpu().view(-1).numpy())
                self.original_dataset_inds = list(np.arange(len(self.dataset)))
            else:
                if filter_diversity:
                    new_x = collate_data_list(data_list).cell_params_to_gen_basis()
                    rands = np.random.choice(len(self.dataset), self.diversity_check_size, replace=False)
                    new_x_dists = torch.cdist(self.x[rands], new_x)
                    new_x_inds_to_keep = new_x_dists.amin(dim=0) >= diversity_cutoff
                    data_list = [data_list[ind] for ind in new_x_inds_to_keep]

                self.dataset.extend(list(data_list))
                self.x = torch.cat([self.x, new_x[new_x_inds_to_keep]], dim=0)
                self.scores_np_list.extend(
                    list(self.energy_function.prebuilt_sample_to_reward(
                        data_list,
                        temperature=torch.ones(len(data_list))).detach().cpu().view(-1).numpy())
                )

            if len(self) > self.buffer_size:  # pare down buffer
                inds_to_keep = self.sample_indices(self.buffer_size, replace=False)
                if self.keep_initial_samples:
                    inds_to_keep = list(set(list(inds_to_keep) + self.original_dataset_inds))[:self.buffer_size]
                else:
                    inds_to_keep = list(set(inds_to_keep))

                self.dataset = [self.dataset[ind] for ind in inds_to_keep]
                self.scores_np_list = [self.scores_np_list[ind] for ind in inds_to_keep]

        gc.collect()

    def __len__(self):
        if self.dataset is None:
            return 0
        else:
            return len(self.dataset)

    def sample_indices(self, batch_size, replace: bool):
        inds = np.random.choice(len(self),
                                size=batch_size,
                                replace=replace,
                                p=self.get_sampler_weights())
        return inds

    def get_sampler_weights(self, eps: float=1e-6):
        scores = np.array(self.scores_np_list)
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

    def sample(self,
               temperature: Optional[torch.tensor] = None,
               return_conditioning: Optional[bool] = False,
               override_batch: Optional[int] = None):

        assert return_conditioning or (
                temperature is not None), "Must provide temperature or generate it here with return_conditioning=True"

        if override_batch is not None:
            batch_size = override_batch
        else:
            batch_size = self.batch_size

        # manual dataloader
        if batch_size > len(self):
            rand_inds = self.sample_indices(batch_size, replace=True)
        else:
            rand_inds = self.sample_indices(batch_size, replace=False)

        sample = collate_data_list([self.dataset[ind] for ind in rand_inds])

        condition = self.energy_function.get_conditioning_tensor(sample)
        temperature = 10 ** condition[:, 0]  # first dimension is the log temperature
        with torch.no_grad():
            reward = self.energy_function.prebuilt_sample_to_reward(
                sample, temperature)  # recompute reward in case parameters have changed

        if return_conditioning:
            return sample.cell_params_to_gen_basis(), reward, sample, condition
        else:
            return sample.cell_params_to_gen_basis(), reward, sample
