import abc
import torch
import numpy as np
from torch.utils.data import Dataset


def nll_unit_gaussian(data, sigma=1.0):
    data = data.view(data.shape[0], -1)
    loss = 0.5 * np.log(2 * np.pi) + np.log(sigma) + 0.5 * data * data / (sigma ** 2)
    return torch.sum(torch.flatten(loss, start_dim=1), -1)


class BaseSet(abc.ABC, Dataset):
    def __init__(self, len_data=-2333):
        self.num_sample = len_data
        self.data = None
        self.data_ndim = None
        self._gt_ksd = None

    def gt_logz(self):
        raise NotImplementedError

    @abc.abstractmethod
    def energy(self, x, mol_batch, log_temperature, return_exp: bool=False):
        return

    @property
    def ndim(self):
        return self.data_ndim

    def sample(self, batch_size):
        del batch_size
        raise NotImplementedError

    def log_reward(self, x, mol_batch, log_temperature, return_exp: bool = False):
        if return_exp:
            energy, sample = self.energy(x, mol_batch, log_temperature, return_exp)
            return -energy, sample
        else:
            return -self.energy(x, mol_batch, log_temperature, return_exp)

