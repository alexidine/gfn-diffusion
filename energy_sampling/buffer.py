from typing import Optional

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from mxtaltools.dataset_utils.utils import collate_data_list


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, sample):
        super(SampleDataset, self).__init__()
        self.sample_list = sample

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        return sample

    def update(self, sample):
        self.sample_list = torch.cat([self.sample_list, sample], dim=0)

    def deque(self, length):
        self.sample_list = self.sample_list[length:]

    def get_seq(self):
        return self.sample_list

    def __len__(self):
        return len(self.sample_list)

    def collate(data_list):
        return torch.stack(data_list)


class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, rewards):
        super(RewardDataset, self).__init__()
        self.rewards = rewards
        self.raw_tsrs = self.rewards

    def __getitem__(self, idx):
        return self.rewards[idx]
        #return  self.score_list[idx]

    def update(self, rewards):
        new_rewards = rewards

        self.raw_tsrs = torch.cat([self.rewards, new_rewards], dim=0)
        self.rewards = self.raw_tsrs

    def deque(self, length):
        self.raw_tsrs = self.raw_tsrs[length:]
        self.rewards = self.raw_tsrs

    def get_tsrs(self):
        return self.rewards

    def __len__(self):
        return self.rewards.size(0)

    def collate(data_list):
        return torch.stack(data_list)


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return [dataset[idx] for dataset in self.datasets]

    def collate(data_list):
        return [dataset.collate(data_list) for dataset, data_list in zip(self.datasets, zip(*data_list))]


def collate(data_list):
    sample, rewards = zip(*data_list)

    sample_data = SampleDataset.collate(sample)
    reward_data = RewardDataset.collate(rewards)

    return sample_data, reward_data


class ReplayBuffer():
    def __init__(self, buffer_size,
                 device,
                 log_reward,
                 batch_size,
                 data_ndim=2,
                 beta=1.0,
                 rank_weight=1e-2,
                 prioritized=None):
        self.buffer_size = buffer_size
        self.prioritized = prioritized
        self.device = device
        self.data_ndim = data_ndim
        self.batch_size = batch_size
        self.reward_dataset = None
        self.raw_reward_dataset = None
        self.buffer_idx = 0
        self.buffer_full = False
        self.log_reward = log_reward
        self.beta = beta
        self.rank_weight = rank_weight
        self.beta = beta

    def add(self, samples, log_r, raw_reward):
        if self.reward_dataset is None:
            self.reward_dataset = RewardDataset(log_r.detach())
            self.sample_dataset = SampleDataset(samples.detach())
            self.raw_reward_dataset = RewardDataset(raw_reward.detach())  # store raw values for easier rescaling
            self.sample_dataset.update(samples.detach())
            self.reward_dataset.update(log_r.detach())
            self.raw_reward_dataset.update(raw_reward.detach())
        else:
            self.sample_dataset.update(samples.detach())
            self.reward_dataset.update(log_r.detach())
            self.raw_reward_dataset.update(raw_reward.detach())

        if self.reward_dataset.__len__() > self.buffer_size:
            self.reward_dataset.deque(self.reward_dataset.__len__() - self.buffer_size)
            self.sample_dataset.deque(self.sample_dataset.__len__() - self.buffer_size)
            self.raw_reward_dataset.deque(self.raw_reward_dataset.__len__() - self.buffer_size)

        if self.prioritized == 'rank':
            self.scores_np = self.reward_dataset.get_tsrs().detach().cpu().view(-1).numpy()
            ranks = np.argsort(np.argsort(-1 * self.scores_np))
            weights = 1.0 / (1e-2 * len(self.scores_np) + ranks)
            self.dataset = ZipDataset(self.sample_dataset, self.reward_dataset)
            self.sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=len(self.scores_np), replacement=True
            )

            self.loader = torch.utils.data.DataLoader(
                self.dataset,
                sampler=self.sampler,
                batch_size=self.batch_size,
                collate_fn=collate,
                #drop_last=True
                drop_last=False
            )
        else:
            weights = 1.0
            self.dataset = ZipDataset(self.sample_dataset, self.reward_dataset)
            self.sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=len(self.scores_np), replacement=True
            )

            self.loader = torch.utils.data.DataLoader(
                self.dataset,
                sampler=self.sampler,
                batch_size=self.batch_size,
                collate_fn=collate,
                drop_last=True
            )
        # check if we have any additional samples before updating the buffer and the scorer!

    def __len__(self):
        if self.reward_dataset is None:
            return 0
        else:
            return len(self.reward_dataset.rewards)

    def sample(self):

        try:
            sample, reward = next(self.data_iter)
        except:
            self.data_iter = iter(self.loader)
            sample, reward = next(self.data_iter)

        return sample.detach(), reward.detach()


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

    def add(self, dataset):
        if self.dataset is None:
            self.dataset = dataset
        else:
            self.dataset.extend(dataset)

        if len(self.dataset) > self.buffer_size:  # todo consider sorting here by rank / diversity
            self.dataset = self.dataset[-self.buffer_size:]
            self.scores_np = self.dataset[-self.buffer_size:]

        if not hasattr(self, 'scores_np'):
            self.scores_np = self.energy_function.prebuilt_sample_to_reward(self.dataset, temperature=torch.ones(
                len(self.dataset))).detach().cpu().view(-1).numpy()
        else:
            self.scores_np = np.concatenate([
                self.scores_np,
                self.energy_function.prebuilt_sample_to_reward(
                    dataset,
                    temperature=torch.ones(len(dataset))).detach().cpu().view(-1).numpy()
                ]
            )

        if self.prioritized == 'rank':  # todo add a diversity-type sampler
            ranks = np.argsort(np.argsort(-1 * self.scores_np))
            weights = 1.0 / (1e-2 * len(self.scores_np) + ranks)
            self.sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=len(self.scores_np), replacement=True
            )

            self.loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=self.sampler,
                num_workers=0,
                pin_memory=True,
                drop_last=False)
        else:
            weights = 1.0
            self.sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=len(self.scores_np), replacement=True
            )

            self.loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=self.sampler,
                num_workers=0,
                pin_memory=True,
                drop_last=False)

    def __len__(self):
        if self.dataset is None:
            return 0
        else:
            return len(self.dataset)

    def sample(self,
               temperature: Optional[torch.tensor] = None,
               return_conditioning: Optional[bool] = False,
               override_batch: Optional[int] = None):

        assert return_conditioning or (temperature is not None), "Must provide temperature or generate it here"

        if override_batch is not None and override_batch != self.loader.batch_size:  # manual resampling if we want a custom batch size
            if override_batch >= len(self.dataset):
                rand_inds = np.random.randint(len(self.dataset), override_batch)
            else:
                rand_inds = np.random.choice(len(self.dataset), override_batch, replace=False)

            sample = collate_data_list([self.loader.dataset[ind] for ind in rand_inds])
        else:
            sample = next(iter(self.loader))

        condition = self.energy_function.get_conditioning_tensor(sample)
        temperature = 10 ** condition[:, 0]  # first dimension is the log temperature
        reward = self.energy_function.prebuilt_sample_to_reward(sample, temperature)  # recompute reward in case parameters have changed

        if return_conditioning:
            return sample.cell_params_to_gen_basis(), reward, sample, condition
        else:
            return sample.cell_params_to_gen_basis(), reward, sample
