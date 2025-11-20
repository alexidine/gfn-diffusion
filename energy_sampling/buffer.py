from typing import Optional

import hdbscan
import numpy as np
import torch
from poetry.console.commands import self
from scipy.spatial.transform import Rotation
from torch_geometric.loader import DataLoader
from torch_scatter import scatter

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
                 max_z_prime: int = 1,
                 buffer_dist_cutoff: float = 0.25,
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

    @torch.no_grad()
    def add(self,
            data_list=None,
            data_batch=None,
            max_z_prime: int = 1):
        if self.dataset is None:
            self.init_fresh_dataset(data_list, max_z_prime)
        else:
            self.add_samples_to_dataset(data_batch, data_list, max_z_prime)

        if len(self) > self.buffer_size:  # pare down buffer
            self.truncate_buffer()

        assert len(self.dataset) == len(self.x_list) == len(self.rewards_list)

        self.init_loader()

    def add_samples_to_dataset(self, data_batch, data_list, max_z_prime):
        # batch samples
        if data_list is None and data_batch is not None:
            data_list = data_batch.batch_to_list()
        elif data_list is not None and data_batch is None:
            data_batch = collate_data_list(data_list, max_z_prime=self.max_z_prime)
        # get rewards
        scores = self.energy_function.prebuilt_sample_to_reward(data_batch,
                                                                temperature=torch.ones(data_batch.num_graphs))
        # enforce reasonable standards for consideration in the buffer
        score_cut = np.quantile(self.rewards_list, 0.5)
        good_inds = [ind for ind in range(len(data_list)) if
                     ((data_list[ind].lj_pot < 0) and (data_list[ind].niggli_overlap >= 0) and (
                                 scores[ind] > score_cut))
                     ]
        # add anything reasonable
        if len(good_inds) > 0:
            data_to_add = [data_list[ind] for ind in good_inds]

            self.dataset.extend(data_to_add)
            dataset_batch = collate_data_list(self.dataset, max_z_prime=max_z_prime)
            x_tensor = dataset_batch.latent_params()
            good_scores = scores[torch.tensor(good_inds, dtype=torch.long)]
            self.x_list = [x_tensor[i] for i in range(x_tensor.shape[0])]
            self.rewards_list.extend(good_scores.flatten().cpu().detach().numpy())
            self.sg_list = list(dataset_batch.sg_ind.cpu())

    def init_fresh_dataset(self, data_list, max_z_prime):
        self.dataset = list(data_list)  # I think this is memory safe and faster #copy.deepcopy(data_list)
        for elem in self.dataset:  # have to do this now because collation is a mess
            del elem.fingerprint, elem.smiles, elem.mol_ind, elem.identifier, (
                elem.aunit_batch), elem.skip_box_analysis, elem.cocrystal, elem.symmetry_operators

        dataset_batch = collate_data_list(self.dataset, max_z_prime=max_z_prime)
        x_tensor = dataset_batch.latent_params()
        rewards = self.energy_function.prebuilt_sample_to_reward(dataset_batch, temperature=torch.ones(len(self)))
        if self.energy_function.reward_range is not None:
            self.energy_function.set_reward_clip(rewards)

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
        e_tensor = -torch.nan_to_num(torch.tensor(self.rewards_list, device=device))

        # define cutoffs
        d_cut = self.buffer_dist_cutoff
        if self.original_dataset_inds is not None:
            e_cut = np.quantile(e_tensor[self.original_dataset_inds], 0.25)
        else:
            e_cut = np.quantile(e_tensor, 0.25)

        inds_to_keep = self.bottom_up_cluster(x_tensor, e_tensor, d_cut, e_cut)

        if self.keep_initial_samples:
            orig_dataset_ind_tensor = torch.tensor(self.original_dataset_inds, device=self.device, dtype=torch.long)
            combined = torch.unique(torch.cat([torch.tensor(inds_to_keep),
                                               orig_dataset_ind_tensor]))
            # Protect originals by giving them artificially low energy ranks
            energies = e_tensor[combined].clone()
            mask_orig = torch.isin(combined, orig_dataset_ind_tensor)
            energies[mask_orig] -= 1e6  # will always be kept
            sorted_combined = combined[torch.argsort(energies)]
            inds_to_keep = sorted_combined[:self.buffer_size]
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

    def bottom_up_cluster(self, xx, e, d_cut, e_cut):
        # Sort by energy ascending
        sort_inds = torch.argsort(e)
        xx_sorted = xx[sort_inds]
        e_sorted = e[sort_inds]

        mask = e_sorted < e_cut

        # Compute full distance matrix once (O(n^2) but fast on GPU)
        dmat = torch.cdist(xx_sorted, xx_sorted)

        keep = torch.zeros(len(xx_sorted), dtype=bool, device=xx.device)
        for i in range(len(xx_sorted)):
            if not mask[i]:
                break
            # check if this point is farther than d_cut from all previously kept points
            too_close = (dmat[i, keep] < d_cut).any()
            if not too_close:
                keep[i] = True

        keep_inds = sort_inds[keep]

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.1)
        labels = clusterer.fit_predict(xx[keep_inds].cpu().numpy())
        minima_inds = []
        for lbl in torch.unique(torch.tensor(labels), sorted=True):
            mask = labels == lbl.item()
            if mask.sum() == 0: continue
            idx = torch.argmin(e[keep_inds][mask])
            minima_inds.append(keep_inds[mask][idx])

        noisy_inds = keep_inds[labels==-1]

        inds_to_keep = torch.cat([
            torch.tensor(minima_inds[1:]), noisy_inds
        ])

        return inds_to_keep

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

        if hasattr(sample_batch,'latent_transform'):
            del sample_batch.latent_transform

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


def collate_fn(data_list):
    return collate_data_list(data_list, exclude_unit_cell=True)
