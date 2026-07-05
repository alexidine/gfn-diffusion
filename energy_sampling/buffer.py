from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from mxtaltools.dataset_utils.utils import collate_data_list
from utils import compute_sample_overlap, iter_forever, stdz


def robust_range(x, k=8):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma_robust = 1.4826 * mad  # consistent with Gaussian σ

    lower = med - k * sigma_robust
    upper = med + k * sigma_robust
    return lower, upper


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
                 noised_max_steps: int = 50,
                 kT_range: float = 6.0,
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
        self.kT_range = kT_range
        self.diversity_check_size = 1000
        self.original_dataset_inds = None
        self.diversity_coeff = diversity_coeff
        self.max_z_prime = max_z_prime
        self.buffer_dist_cutoff = buffer_dist_cutoff
        self.staging_buffer = []
        self.noised_buffer_length = noised_buffer_length
        self.noised_rewards = []  # torch.zeros(self.noised_buffer_length, dtype=torch.float32, device='cpu')
        self.noised_samples = []  # torch.zeros((self.noised_buffer_length, 6 + 6 * max_z_prime), dtype=torch.float32, device='cpu')
        self.noised_losses = []  # torch.zeros(self.noised_buffer_length, dtype=torch.float32, device='cpu')
        self.noised_select_counts = []
        self.noised_max_steps = noised_max_steps

    @torch.no_grad()
    def add_to_staging(self, importance_weight, data_list=None, data_batch=None):
        if len(self.staging_buffer) < len(
                self):  # don't stage a crazy number of samples - downstream cost becomes too high
            if data_list is None and data_batch is not None:
                data_list = data_batch.cpu().detach().batch_to_list()

            self.staging_buffer.extend([elem for ind, elem in enumerate(data_list) if
                                        importance_weight[ind] > 0])  # keep any plausibly underweighted states

    @torch.no_grad()
    def add_init(self,
                 data_list):
        self.init_fresh_dataset(data_list)

        assert len(self.dataset) == len(self.x_list) == len(self.rewards_list)

    def incorporate_staging_buffer(self):
        if len(self.staging_buffer) > 0:  # will fail if staging buffer is empty
            self.add_samples_to_dataset(self.staging_buffer, skip_staging=False)
            assert len(self.dataset) == len(self.x_list) == len(self.rewards_list)

    def add_samples_to_dataset(self, data_list, skip_staging: bool = False):
        # batch samples
        assert False, "This needs to be rewritten / checked for index issues"  # todo
        if not skip_staging:
            if len(self.staging_buffer) > 0:  # include staged samples
                data_list.extend(self.staging_buffer)
                self.staging_buffer = []

        data_batch = collate_data_list(data_list, max_z_prime=self.max_z_prime)
        new_latents = data_batch.latent_params()
        new_sgs = data_batch.sg_ind
        # get new samples rewards
        rewards = self.energy_function.prebuilt_sample_to_reward(
            data_batch,
            temperature=torch.ones(data_batch.num_graphs) * self.energy_function.temperature)

        # enforce standards for consideration in the buffer
        score_cut = np.amax(self.rewards_list) - 2  # don't keep things more than 2*kT worse than the best sample
        # the lowest reward in our dynamical range
        packing_coeffs = data_batch.packing_coeff.cpu().detach().numpy()
        good_inds = [ind for ind in range(len(data_list)) if
                     (data_list[ind].reduction_en <= 1e-3) and (rewards[ind] > score_cut) and (
                             packing_coeffs[ind] > 0.55) and (packing_coeffs[ind] < 0.95)]

        data_list = [data_batch[ind] for ind in good_inds]
        data_batch = collate_data_list(data_list, max_z_prime=self.max_z_prime)

        # add anything reasonable
        if len(good_inds) > 0:
            data_to_add = [data_list[ind] for ind in good_inds]
            self.dataset.extend(data_to_add)
            good_scores = rewards[torch.tensor(good_inds, dtype=torch.long)]
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

        l, u = robust_range(x=rewards, k=4)
        good_inds = rewards > l
        self.dataset = [self.dataset[ind] for ind in torch.argwhere(good_inds).flatten()]
        dataset_batch = collate_data_list(self.dataset, max_z_prime=self.max_z_prime)
        x_tensor = dataset_batch.latent_params()
        rewards = self.energy_function.prebuilt_sample_to_reward(
            dataset_batch,
            temperature=torch.ones(len(self)) * self.energy_function.temperature)

        if self.energy_function.reward_range is not None:
            # reward scaling is temperature dependent
            self.energy_function.set_reward_clip(rewards)
            self.reward_clip = -self.energy_function.energy_clip / self.energy_function.temperature
            self.energy_clip = self.energy_function.energy_clip
            # recompute with new clip
            rewards = self.energy_function.prebuilt_sample_to_reward(
                dataset_batch,
                temperature=torch.ones(len(self)) * self.energy_function.temperature)

        self.x_list = [x_tensor[i] for i in range(x_tensor.shape[0])]
        self.rewards_list = list(rewards.flatten().cpu().detach().numpy())
        self.original_dataset_inds = list(np.arange(len(self.dataset)))
        self.sg_list = list(dataset_batch.sg_ind.cpu())

    def truncate_buffer(self, importance_weight):
        """
        1 - keep initial states
        2 - bottom-up energy greedy selection
        3 - clustering
        :return:
        """
        new_inds = self.find_new_permanent_candidates(
            dist_cutoff=0.05,
            energy_window=1.0
        )

        for ind in new_inds:
            self.original_dataset_inds.append(ind)

        inds_to_keep = torch.argsort(importance_weight, descending=True)

        if self.keep_initial_samples:
            orig_dataset_ind_tensor = torch.tensor(self.original_dataset_inds, device=self.device, dtype=torch.long)
            combined = torch.unique(torch.cat([orig_dataset_ind_tensor, inds_to_keep]))[:self.buffer_size]

        inds_to_keep = combined.tolist()  # for convenience
        self.dataset = [self.dataset[ind] for ind in inds_to_keep]
        self.rewards_list = [self.rewards_list[ind] for ind in inds_to_keep]
        self.x_list = [self.x_list[ind] for ind in inds_to_keep]
        self.sg_list = [self.sg_list[ind] for ind in inds_to_keep]

    def find_new_permanent_candidates(self, dist_cutoff=0.05, energy_window=1.0):
        """
        Identify high-reward, geometrically distinct samples to promote
        to the permanent buffer.

        Returns
        -------
        List[int]
            Indices of samples suitable for promotion.
        """

        rewards = np.array(self.rewards_list)
        best_reward = np.amax(rewards)

        # candidates: within 1 kT of best, but NOT already original
        candidate_inds = [
            i for i in range(len(rewards))
            if (i not in self.original_dataset_inds)
               and (rewards[i] >= best_reward - energy_window)
        ]

        if len(candidate_inds) == 0:
            return []

        # Latents
        x_all = torch.stack(self.x_list).to(self.device)
        x_candidates = x_all[candidate_inds]

        # Reference set: existing protected samples
        ref_inds = self.original_dataset_inds
        if len(ref_inds) == 0:
            # If nothing is protected yet, accept all candidates
            return candidate_inds

        x_ref = x_all[ref_inds]

        # Compute pairwise distances (candidate → protected)
        # shape: [N_candidates, N_ref]
        dists = torch.cdist(x_candidates, x_ref)

        # Minimum distance to protected set
        min_dists = dists.min(dim=1).values

        # Accept only sufficiently distinct samples
        accepted = [
            candidate_inds[i]
            for i in range(len(candidate_inds))
            if min_dists[i].item() > dist_cutoff
        ]

        return accepted

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
                            eps: float = 1e-3,
                            override_method: Optional[str] = None):
        if override_method is not None:
            method = override_method
        else:
            method = self.prioritized

        if method is not None:
            if method in ['rank', 'boltzmann']:
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

        else:
            weights_i = np.ones(len(self.x_list))

        return weights_i / np.sum(weights_i)  # enforce explicit normalization

    @torch.no_grad()
    def sample(self,
               override_batch: Optional[int] = None,
               return_preload: Optional[bool] = False,
               override_sampler: Optional[str] = None,
               randomize_orientations: Optional[bool] = False,
               return_sample_inds: Optional[bool] = False,
               override_sample_inds: Optional[torch.Tensor] = None,
               ):

        if override_batch is not None:
            self.batch_size = override_batch

        # manual dataloader
        if return_preload:
            rand_inds = self.original_dataset_inds
        elif override_sample_inds is not None:
            rand_inds = override_sample_inds
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
                                         max_z_prime=self.max_z_prime,
                                         exclude_keys=['symmetry_operators']
                                         )

        if randomize_orientations:
            assert False, "Orientation work currently deprecated"
            # # this is a form of sample augmentation, where we rotate the molecule and its applied orientation
            # # in order to construct the identical crystal, but with a distinct conditioning vector & sample
            # # also rotate the embedding vector which is passed to conditioning
            # random_rotations = torch.tensor(
            #     Rotation.random(num=sample_batch.num_graphs).as_matrix(),
            #     device=sample_batch.device, dtype=torch.float32)
            # sample_batch.orient_molecule(mode='std')
            # sample_batch.orient_molecule(mode='random',  # important that the rotation is applied *from* the standard
            #                              include_inversion=False,
            #                              correct_orientation=True,
            #                              override_random_rotations=random_rotations)
            # sample_batch.embedding = sample_batch.rotate_embedding(random_rotations)

        # if standardize_orientations:
        #     assert not randomize_orientations
        #     sample_batch.orient_molecule(mode='std',
        #                                  correct_orientation=True)

        T_tensor, sg_inds, condition = self.energy_function.condition_samples(
            sample_batch, sg_inds=sample_batch.sg_ind, z_primes=sample_batch.z_prime)

        sample_batch.reset_sg_info(sg_inds)
        temperature = 10 ** T_tensor  # first dimension is the log temperature
        reward = self.energy_function.prebuilt_sample_to_reward(
            sample_batch, temperature)  # recompute reward in case parameters have changed

        latents = sample_batch.latent_params()

        if return_sample_inds:
            return latents, reward, sample_batch, condition, rand_inds
        else:
            return latents, reward, sample_batch, condition

    def init_loader(self):  # todo this is almost never used, currently
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

    def adjust_batch_size(self, new_batch_size: int):  # todo almost never used
        self.loader.batch_sampler.batch_size = new_batch_size
        self._loader_iter = iter_forever(self.loader)

    def add_to_noised(self, rewards, samples, losses, override_size: bool = False):
        rewards = rewards.detach().to(self.device)
        samples = samples.detach().to(self.device)
        losses = losses.detach().to(self.device)
        B = rewards.shape[0]

        for i in range(B):
            # If buffer is full, remove oldest entry (FIFO)
            if not override_size:
                if len(self.noised_rewards) >= self.noised_buffer_length:
                    self.noised_rewards.pop(0)
                    self.noised_samples.pop(0)
                    self.noised_losses.pop(0)
                    self.noised_select_counts.pop(0)

            self.noised_rewards.append(rewards[i])
            self.noised_samples.append(samples[i])
            self.noised_losses.append(losses[i])
            self.noised_select_counts.append(0)

    @torch.no_grad()
    def update_noised_losses(self, losses, indices, beta: float = 0.9):
        """Update losses for specific indices with EMA."""
        losses = losses.detach().cpu()
        for i, idx in enumerate(indices):
            old_loss = self.noised_losses[idx]
            if old_loss == 0:
                self.noised_losses[idx] = losses[i]
            else:
                self.noised_losses[idx] = beta * old_loss + (1.0 - beta) * losses[i]

    def sample_from_noised(self, num_samples):
        buffer_size = len(self.noised_rewards)

        if num_samples <= buffer_size:
            indices = np.random.choice(range(buffer_size), num_samples, replace=False)
        else:
            indices = np.random.choice(range(buffer_size), num_samples, replace=True)

        rewards = torch.stack([self.noised_rewards[i] for i in indices])
        samples = torch.stack([self.noised_samples[i] for i in indices])

        # Increment selection counts
        for idx in indices:
            self.noised_select_counts[idx] += 1

        return rewards, samples, indices

    def purge_noised_buffer(self):
        steps_cutoff = self.noised_max_steps
        loss_cutoff = np.mean(self.noised_losses)

        losses = np.array(self.noised_losses)
        steps = np.array(self.noised_select_counts)

        purge_list = np.argwhere(
            (losses < loss_cutoff) * (steps > steps_cutoff)
        ).flatten()

        self.purge_noised_by_index(purge_list)

    def purge_noised_by_index(self, indices_to_remove):
        """Remove samples by their current indices. Indices should be sorted in descending order."""
        # Sort in descending order to avoid index shifting issues
        for idx in sorted(indices_to_remove, reverse=True):
            if 0 <= idx < len(self.noised_rewards):
                self.noised_rewards.pop(idx)
                self.noised_samples.pop(idx)
                self.noised_losses.pop(idx)
                self.noised_select_counts.pop(idx)

    @torch.no_grad()
    def replace_initial_with_local_optima(self,
                                          opt_latents: torch.Tensor,
                                          opt_rewards: torch.Tensor
                                          ):
        """
        Replace protected initial samples with their local optima
        when clearly better and geometrically equivalent.
        """

        assert self.original_dataset_inds is not None
        assert len(self.original_dataset_inds) == len(opt_latents)

        for k, buf_idx in enumerate(self.original_dataset_inds):
            x_new = opt_latents[k]
            r_new = opt_rewards[k]

            # Replace latent + reward
            self.x_list[buf_idx] = x_new.detach().cpu()
            self.rewards_list[buf_idx] = float(r_new)

            self.dataset[buf_idx].latent_to_cell_params(x_new[None, :])
        for elem in self.dataset:
            del elem.asym_unit_dict


def collate_fn(data_list):
    return collate_data_list(data_list, exclude_unit_cell=True)


class CrystalBuffer:
    """
    Prior dataset with per-sample bookkeeping.

    Holds a resident PyG Batch plus precomputed tensors x (latents) and optional
    scalars y. Tracks an EMA loss and a selection count per sample.

    Avoids batch -> data_list -> batch round trips during sampling, add, and purge.
    Requires the Batch class to implement:
        - subsample_new_batch(idx)
        - append_batch(other)
    """

    def __init__(
            self,
            data,
            device,
            max_z_prime: int = 1,
            x_fn=None,
            y_fn=None,
            traj: Optional[torch.Tensor] = None,
    ):
        self.device = device
        self.max_z_prime = max_z_prime
        self.x_fn = x_fn
        self.y_fn = y_fn

        self.batch = self._as_batch(data).to(device)
        self.x, self.y = self._compute_xy(self.batch)

        n = len(self)
        self.ema_loss = torch.full((n,), float("nan"), dtype=torch.float32)
        self.select_counts = torch.zeros(n, dtype=torch.long)

        if traj is not None:
            assert traj.shape[0] == n, \
                f"traj has {traj.shape[0]} entries, expected {n} to match dataset size"
            traj = traj.detach().to(device).contiguous()
        self.traj = traj

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _as_batch(self, data):
        """
        Accept either a list of Data objects or an existing Batch.
        Collates only when absolutely necessary.
        """
        if isinstance(data, list):
            return collate_data_list(data, max_z_prime=self.max_z_prime)

        # Assume already a compatible PyG batch.
        # You may want a stricter isinstance check here if you have a known Batch class.
        if data.max_z_prime != self.max_z_prime:
            data.max_z_prime = self.max_z_prime
            data.aunit_handedness = data.aunit_handedness[:, :self.max_z_prime]
            data.aunit_orientation = data.aunit_orientation[:, :3 * self.max_z_prime]
            data.aunit_centroid = data.aunit_centroid[:, :3 * self.max_z_prime]
            data.z_prime = data.z_prime.clip(max=self.max_z_prime)

        data.box_analysis()
        return data

    def _compute_xy(self, batch):
        """
        Compute cached x/y tensors directly from a resident batch.
        """
        if self.x_fn is None:
            x = batch.latent_params()
        elif callable(self.x_fn):
            x = self.x_fn(batch)
        else:
            x = batch[self.x_fn]

        x = x.detach().to(self.device).contiguous()

        if self.y_fn is None:
            y = None
        elif callable(self.y_fn):
            y = self.y_fn(batch).detach().to(self.device).contiguous()
        else:
            y = batch[self.y_fn].detach().to(self.device).contiguous()

        return x, y

    def __len__(self):
        return self.batch.num_graphs

    def _sample_indices(
            self,
            batch_size: int,
            replace: Optional[bool] = None,
            repeats: int = 1,
            p: Optional[np.ndarray] = None,
            beta: float = 0.0,  # fraction drawn uniformly
    ):
        n = len(self)

        if n == 0:
            raise ValueError("Cannot sample from an empty SimpleDataset.")

        if p is not None and beta > 0.0:
            n_uniform = max(1, int(batch_size * beta))
            n_weighted = batch_size - n_uniform

            weighted_inds = np.random.choice(n, size=n_weighted, replace=True, p=p)
            uniform_inds = np.random.choice(n, size=n_uniform, replace=n_uniform > n)
            inds = np.concatenate([weighted_inds, uniform_inds])
        else:
            if replace is None:
                replace = batch_size > n
            inds = np.random.choice(n, size=batch_size, replace=replace, p=p)

        if repeats > 1:
            inds = np.repeat(inds, repeats)

        return inds

    def _bump_counts(self, inds):
        """
        Count by occurrence so repeats / replacement duplicates each register.
        """
        bc = np.bincount(np.asarray(inds), minlength=len(self))
        self.select_counts += torch.as_tensor(bc, dtype=torch.long)

    @staticmethod
    def _drop_keys(batch, exclude_keys):
        """
        Optional post-subsample key removal, matching old sample_graphs behavior.

        This is cheap relative to rebuilding from a data_list.
        """
        if exclude_keys is None:
            return batch

        for key in exclude_keys:
            if key in batch._store:
                del batch[key]

        return batch

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def sample_tensors(
            self,
            batch_size: int,
            replace: Optional[bool] = None,
            repeats: int = 1,
            weighted: bool = False,
            temperature: Optional[float] = None,
            beta: Optional[float] = None,
            return_traj: bool = False,
    ):
        p = self._loss_weights(temperature) if weighted else None
        inds = self._sample_indices(batch_size, replace=replace, repeats=repeats, p=p, beta=beta)
        self._bump_counts(inds)

        t_inds = torch.as_tensor(inds, device=self.device, dtype=torch.long)

        x = self.x[t_inds]
        y = self.y[t_inds] if self.y is not None else None
        traj = self.traj[t_inds] if (return_traj and self.traj is not None) else None

        # numpy inds returned for update_losses
        return x, y, traj, inds

    @torch.no_grad()
    def sample_graphs(
            self,
            batch_size: int,
            replace: Optional[bool] = None,
            repeats: int = 1,
            exclude_keys=("symmetry_operators", "smiles", "identifier"),
            orient: bool = True,
            weighted: bool = False,
            temperature: Optional[float] = None,
            beta: Optional[float] = None,
            return_traj: bool = False,
    ):
        p = self._loss_weights(temperature) if weighted else None
        inds = self._sample_indices(batch_size, replace=replace, repeats=repeats, p=p, beta=beta)
        self._bump_counts(inds)

        # No data_list round trip.
        graphs = self.batch.subsample_new_batch(inds)
        graphs = self._drop_keys(graphs, exclude_keys)

        if orient:
            graphs.orient_molecule(mode="std")

        if return_traj and self.traj is not None:
            t_inds = torch.as_tensor(inds, device=self.device, dtype=torch.long)
            traj = self.traj[t_inds]
        else:
            traj = None

        return graphs, inds, traj

    def loader(
            self,
            batch_size: int,
            mode: str = "tensors",
            repeats: int = 1,
            return_inds: bool = False,
            weighted: bool = False,
            temperature: Optional[float] = None,
            beta: Optional[float] = None,
            return_traj: bool = False,
    ):
        """
        Infinite random-batch generator. Use next() on it.

        return_traj appends the sampled [batch, traj_length, dim] trajectory
        tensor to the yielded tuple (after y for "tensors" mode, after the
        graphs batch for "graphs" mode, and before inds if return_inds is
        also set).
        """
        assert mode in ("tensors", "graphs")

        while True:
            if mode == "tensors":
                x, y, traj, inds = self.sample_tensors(batch_size,
                                                       repeats=repeats, weighted=weighted, temperature=temperature,
                                                       beta=beta, return_traj=return_traj)
                result = (x, y)
                if return_traj:
                    result = result + (traj,)
                if return_inds:
                    result = result + (inds,)
                yield result

            else:
                graphs, inds, traj = self.sample_graphs(batch_size,
                                                        repeats=repeats, weighted=weighted, temperature=temperature,
                                                        beta=beta, return_traj=return_traj)
                result = (graphs,)
                if return_traj:
                    result = result + (traj,)
                if return_inds:
                    result = result + (inds,)
                yield result[0] if len(result) == 1 else result

    # ---------------------------------------------------------------------
    # Tracking
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def update_losses(
            self,
            losses,
            indices,
            beta: float = 0.9,
    ):
        losses = torch.as_tensor(losses, dtype=self.ema_loss.dtype).detach().cpu().flatten()
        indices = torch.as_tensor(indices, dtype=torch.long)

        if len(losses) != len(indices):
            raise ValueError(
                f"losses and indices must have same length, got "
                f"{len(losses)} and {len(indices)}."
            )

        old = self.ema_loss[indices]
        nan_mask = torch.isnan(old)

        updated = torch.where(nan_mask, losses, beta * old + (1.0 - beta) * losses)

        # handle duplicates: last write wins (same as sequential loop)
        self.ema_loss[indices] = updated

    # ---------------------------------------------------------------------
    # Mutation
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def add(self, data, traj: Optional[torch.Tensor] = None):
        """
        Append new graphs.

        Accepts either a list[Data] or an already-collated Batch. No data_list
        round trip if a Batch is provided.

        traj, if given, is a [k, traj_length, dim] tensor of per-entry
        trajectories aligned with the k new graphs being added.
        """
        if isinstance(data, list) and len(data) == 0:
            return

        new_batch = self._as_batch(data).to(self.device)

        if new_batch.num_graphs == 0:
            return

        new_x, new_y = self._compute_xy(new_batch)

        self.batch = self.batch.append_batch(new_batch)

        self.x = torch.cat([self.x, new_x], dim=0)

        if self.y is not None:
            if new_y is None:
                raise ValueError("Existing dataset has y, but added batch produced y=None.")
            self.y = torch.cat([self.y, new_y], dim=0)

        if self.traj is not None:
            if traj is None:
                raise ValueError("Existing dataset has traj, but added batch produced traj=None.")
            assert traj.shape[0] == new_batch.num_graphs, \
                f"traj has {traj.shape[0]} entries, expected {new_batch.num_graphs} to match added batch size"
            self.traj = torch.cat([self.traj, traj.detach().to(self.device)], dim=0)

        k = new_batch.num_graphs
        self.ema_loss = torch.cat(
            [
                self.ema_loss,
                torch.full((k,), float("nan"), dtype=self.ema_loss.dtype),
            ],
            dim=0,
        )
        self.select_counts = torch.cat(
            [
                self.select_counts,
                torch.zeros(k, dtype=torch.long),
            ],
            dim=0,
        )

    @torch.no_grad()
    def purge_by_index(self, indices_to_remove):
        """
        Remove samples by current index.

        Uses batch.subsample_new_batch on the keep indices, avoiding any
        batch -> list -> batch rebuild.
        """
        n = len(self)
        if n == 0:
            return

        drop = np.zeros(n, dtype=bool)
        indices_to_remove = np.asarray(indices_to_remove, dtype=int)

        if indices_to_remove.size == 0:
            return

        if indices_to_remove.min() < 0 or indices_to_remove.max() >= n:
            raise IndexError(
                f"indices_to_remove out of bounds for dataset of length {n}."
            )

        drop[indices_to_remove] = True
        keep = ~drop

        if keep.all():
            return

        keep_idx = np.flatnonzero(keep)

        self.batch = self.batch.subsample_new_batch(keep_idx)

        keep_t = torch.as_tensor(keep_idx, device=self.device, dtype=torch.long)
        self.x = self.x[keep_t].contiguous()
        if self.y is not None:
            self.y = self.y[keep_t].contiguous()
        if self.traj is not None:
            self.traj = self.traj[keep_t].contiguous()

        keep_cpu = torch.as_tensor(keep_idx, dtype=torch.long)
        self.ema_loss = self.ema_loss[keep_cpu]
        self.select_counts = self.select_counts[keep_cpu]

    @torch.no_grad()
    def purge(
            self,
            max_count: int,
            loss_cutoff: Optional[float] = None,
    ):
        """
        Drop well-sampled, below-cutoff samples.

        Purges where:
            count > max_count
            ema_loss < loss_cutoff

        Uninitialized NaN losses are never purged.
        """
        losses = self.ema_loss
        counts = self.select_counts

        valid = ~torch.isnan(losses)
        if valid.sum() == 0:
            return

        if loss_cutoff is None:
            loss_cutoff = torch.nanmean(losses).item()

        mask = valid & (losses < loss_cutoff) & (counts > max_count)
        purge_list = torch.nonzero(mask, as_tuple=False).flatten().tolist()

        self.purge_by_index(purge_list)

    @torch.no_grad()
    def purge_lowest(
            self,
            num_to_purge: int,
            quantile: float = 0.25,
            loss_floor: float = 1.0,
            min_visits: int = 3,
            temperature: float = 1.0,
            loss_min: float = 1.0,
    ):
        """
        Purge samples with low loss.

        Forced purge:
            valid loss
            loss < loss_min

        Additional stochastic purge:
            visited >= min_visits
            loss is initialized
            loss <= min(loss_floor, quantile cutoff)

        Stochastic samples are chosen without replacement with probability
        increasing as loss decreases.
        """
        assert loss_min <= loss_floor

        elig_idx, losses, valid = self.get_elig_drop_count(
            loss_floor,
            min_visits,
            quantile,
        )

        # Hard purge everything below loss_min
        forced_idx = torch.where(valid & (losses < loss_min))[0]

        # Avoid choosing forced samples again stochastically
        if forced_idx.numel() > 0 and elig_idx.numel() > 0:
            forced_mask = torch.zeros_like(valid, dtype=torch.bool)
            forced_mask[forced_idx] = True
            elig_idx = elig_idx[~forced_mask[elig_idx]]

        sampled_choice = np.array([], dtype=np.int64)

        remaining = max(num_to_purge - forced_idx.numel(), 0)
        k = min(remaining, elig_idx.numel())

        if k > 0:
            elig_losses = losses[elig_idx]
            logits = -elig_losses / max(temperature, 1e-8)
            logits = logits - logits.max()

            p = torch.softmax(logits, dim=0).double().cpu().numpy()
            p /= p.sum()

            sampled_choice = np.random.choice(
                elig_idx.cpu().numpy(),
                size=k,
                replace=False,
                p=p,
            )
            k = min(num_to_purge, elig_idx.numel())

            elig_losses = losses[elig_idx]
            logits = -elig_losses / max(temperature, 1e-8)
            logits = logits - logits.max()

            p = torch.softmax(logits, dim=0).double().cpu().numpy()
            p /= p.sum()

            sampled_choice = np.random.choice(
                elig_idx.cpu().numpy(),
                size=k,
                replace=False,
                p=p,
            )

        forced_choice = forced_idx.cpu().numpy()

        if forced_choice.size == 0 and sampled_choice.size == 0:
            return

        choice = np.concatenate([forced_choice, sampled_choice])

        # continue with your existing purge logic using `choice`
        self.purge_by_index(choice.tolist())

    def get_elig_drop_count(self, loss_floor, min_visits, quantile):
        losses = self.ema_loss
        valid = (~torch.isnan(losses)) & (self.select_counts >= min_visits)
        if valid.sum() == 0:
            quantile = 0
        else:
            quantile = torch.quantile(losses[valid], quantile).item()
        cutoff = min(loss_floor, quantile)
        eligible = valid & (losses <= cutoff)
        elig_idx = torch.nonzero(eligible, as_tuple=False).flatten()
        return elig_idx, losses, valid

    def _loss_weights(
            self,
            temperature: float = 1.0,
            nan_quantile: float = 0.90,
            epsilon: float = 1e-8,
    ) -> np.ndarray:
        """
        Convert ema_loss to a sampling distribution.

        Higher loss → higher probability.  Unvisited (NaN) samples are
        assigned the ``nan_quantile``-th observed loss so they are treated
        as moderately hard and visited promptly rather than starved.

        Parameters
        ----------
        temperature:
            Softmax temperature τ.  Large τ → nearly uniform; small τ →
            concentrates on the highest-loss sample.
        nan_quantile:
            Quantile of observed losses used to fill NaN entries.
            0.9 keeps unvisited samples competitive without dominating.
        """
        losses = self.ema_loss.clone().float()
        valid = ~torch.isnan(losses)

        if valid.any():
            nan_fill = torch.quantile(losses[valid], nan_quantile).item()
        else:
            # Nothing visited yet — assign high probability.
            return np.ones(len(self), dtype=np.float64) / len(self)

        losses[~valid] = nan_fill

        loss_range = losses.max() - losses.min()
        normed = (losses - losses.min()) / (loss_range + 1e-8)  # [0, 1]
        logits = normed / max(temperature, 1e-8)
        logits -= logits.max()
        p = torch.softmax(logits, dim=0).double().cpu().numpy()
        p = np.clip(p, epsilon, None)  # floor before renorm
        p /= p.sum()
        return p

'''
import plotly.graph_objects as go
go.Figure(go.Histogram(x=np.log(p), nbinsx=100)).show()
'''
