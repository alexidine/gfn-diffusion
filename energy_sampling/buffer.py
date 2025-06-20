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
                 prioritized=None):
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

    def add(self, data_list):
        with torch.no_grad():
            if self.dataset is None:
                self.dataset = copy.deepcopy(data_list)
            else:
                self.dataset.extend(copy.deepcopy(data_list))

            if not hasattr(self, 'scores_np_list'):
                self.scores_np_list = list(
                    self.energy_function.prebuilt_sample_to_reward(self.dataset, temperature=torch.ones(
                        len(self))).detach().cpu().view(-1).numpy())
            else:
                self.scores_np_list.extend(
                    list(self.energy_function.prebuilt_sample_to_reward(
                    data_list,
                    temperature=torch.ones(len(data_list))).detach().cpu().view(-1).numpy())
                    )

            if len(self) > self.buffer_size:
                if hasattr(self, 'sampler'):
                    inds_to_keep = self.get_sample_indices(self.buffer_size)
                else:
                    inds_to_keep = np.arange(len(self) - self.buffer_size, len(self))

                self.dataset = [self.dataset[ind] for ind in inds_to_keep]
                self.scores_np_list = [self.scores_np_list[ind] for ind in inds_to_keep]

        gc.collect()

    def __len__(self):
        if self.dataset is None:
            return 0
        else:
            return len(self.dataset)

    def get_sample_indices(self, batch_size):  # todo add pruning / sampling according to diversity. Expensive to repeat though.
        inds = np.random.choice(len(self),
                                size=batch_size,
                                replace=True,
                                p=self.get_sampler_weights())
        return inds

    def get_sampler_weights(self):
        scores = np.array(self.scores_np_list)
        if self.prioritized == 'rank':
            ranks = np.argsort(np.argsort(-1 * scores))
            weights_i = 1.0 / (self.rank_weight * len(scores) + ranks)
        elif self.prioritized == 'boltzmann':
            logits = scores / self.beta
            logits -= np.max(logits)  # subtract max for stability
            weights_i = np.nan_to_num(np.exp(logits))
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
        rand_inds = self.get_sample_indices(batch_size)
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
