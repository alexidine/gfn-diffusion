import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from mxtaltools.dataset_utils.utils import collate_data_list
from utils import compute_sample_overlap, iter_forever, stdz

# Space-group lookup tables (indexed 0..230) that data_classes builds lazily the
# first time a batch runs a latent transform. They are deterministic globals, not
# per-graph data, but they get attached to whichever batch happens to trigger the
# build. That leaves two batches disagreeing on whether the attr exists (or on its
# shape, if built under an older codebase covering fewer space groups), so a merge
# via Batch.append_batch(validate=True) raises. Strip them before any merge so they
# rebuild fresh and consistent. See train.py init_prior_dataset / this file's
# checkpoint load and anchor-seed paths for the same fix at other side-load points.
_LAZY_SG_CACHES = ('asym_unit_lut', 'asym_unit_dict', 'sym_mult_lut')


def strip_lazy_sg_caches(batch):
    """Drop lazily-built space-group caches so batches merge without mismatch."""
    for attr in _LAZY_SG_CACHES:
        if hasattr(batch, attr):
            delattr(batch, attr)
    return batch


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
            init_loss: Optional[torch.Tensor] = None,
            exclude_keys: Optional[tuple] = None,
    ):
        self.device = device
        self.max_z_prime = max_z_prime
        self.x_fn = x_fn
        self.y_fn = y_fn
        # keys stripped from STORAGE at admission (draws already drop them):
        # churned buffers don't need string/list attrs, and python-list keys
        # force a per-subsample idx.tolist() -- a device sync on GPU-resident
        # buffers -- plus per-element copying on every rebuild
        self.exclude_keys = tuple(exclude_keys) if exclude_keys else ()

        self.batch = self._as_batch(data).to(device)
        self._drop_keys(self.batch, self.exclude_keys)
        self._orient_stored_batch(self.batch)
        self.x, self.y = self._compute_xy(self.batch)

        n = len(self)
        if init_loss is None:
            self.ema_loss = torch.full((n,), float("nan"), dtype=torch.float32)
        else:
            self.ema_loss = torch.as_tensor(init_loss, dtype=torch.float32).detach().cpu().flatten()
            if self.ema_loss.numel() == 1:
                self.ema_loss = self.ema_loss.expand(n).clone()
            assert self.ema_loss.shape[0] == n, \
                f"init_loss has {self.ema_loss.shape[0]} entries, expected {n} to match dataset size"
        self.select_counts = torch.zeros(n, dtype=torch.long)

        # per-sample rolling estimates of the log importance weight
        # logw = log_r + log_pb - log_pf under the current policy.
        self.ema_logw = torch.full((n,), float("nan"), dtype=torch.float32)
        self.ema_logw_sq = torch.full((n,), float("nan"), dtype=torch.float32)
        self.ema_log_z_emp = torch.full((n,), float("nan"), dtype=torch.float32)

        if traj is not None:
            assert traj.shape[0] == n, \
                f"traj has {traj.shape[0]} entries, expected {n} to match dataset size"
            traj = traj.detach().to(device).contiguous()
        self.traj = traj

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------

    def state_dict(self):
        # NB copy.copy first: PyG's Data.cpu()/.to() MUTATE IN PLACE (apply
        # rewrites the store and returns self), so a bare self.batch.cpu()
        # here silently demoted the entire live GPU-resident buffer to CPU at
        # every save_buffers -- i.e. at the first eval -- which is exactly the
        # bug that made buffer_device: cuda appear to change nothing. The
        # shallow copy gives .cpu() its own store to rewrite; tensor-level
        # .cpu() is non-mutating, so the live tensors are untouched.
        return {
            'batch': copy.copy(self.batch).cpu(),
            'max_z_prime': self.max_z_prime,
            'x_fn': self.x_fn,
            'y_fn': self.y_fn,
            'exclude_keys': getattr(self, 'exclude_keys', ()),
            'x': self.x.cpu(),
            'y': self.y.cpu() if self.y is not None else None,
            'ema_loss': self.ema_loss.cpu(),
            'select_counts': self.select_counts.cpu(),
            'ema_logw': self.ema_logw.cpu(),
            'ema_logw_sq': self.ema_logw_sq.cpu(),
            'ema_log_z_emp': self.ema_log_z_emp.cpu(),
            'traj': self.traj.cpu() if self.traj is not None else None,
        }

    @classmethod
    def from_state_dict(cls, state, device):
        obj = cls.__new__(cls)
        obj.device = device
        obj.max_z_prime = state['max_z_prime']
        obj.x_fn = state['x_fn']
        obj.y_fn = state['y_fn']
        obj.exclude_keys = tuple(state.get('exclude_keys', ()) or ())
        obj.batch = state['batch'].to(device)
        # A checkpoint may have been pickled with stale/differently-shaped
        # versions of these caches (e.g. from an older codebase covering fewer
        # space groups). Strip them so they rebuild fresh and consistent with
        # every other batch in this run. See strip_lazy_sg_caches for details.
        strip_lazy_sg_caches(obj.batch)
        # Older checkpoints stored unoriented graphs (orientation used to
        # happen per-draw in sample_graphs).
        cls._orient_stored_batch(obj.batch)
        obj.x = state['x'].to(device)
        obj.y = state['y'].to(device) if state['y'] is not None else None
        # These are CPU-resident bookkeeping tensors, but torch.load's map_location
        # remaps every tensor in the pickle (even nested ones), so force them back
        # to cpu here regardless of what device the checkpoint was loaded onto.
        obj.ema_loss = state['ema_loss'].cpu()
        obj.select_counts = state['select_counts'].cpu()
        obj.ema_logw = state['ema_logw'].cpu()
        obj.ema_logw_sq = state['ema_logw_sq'].cpu()
        obj.ema_log_z_emp = state['ema_log_z_emp'].cpu()
        obj.traj = state['traj'].to(device) if state['traj'] is not None else None
        return obj

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

    @staticmethod
    def _orient_stored_batch(batch):
        """
        Std-orient molecules once at admission, so sample_graphs draws skip the
        per-draw recenter + principal-axes work (rows are drawn far more often
        than they are admitted, and std orientation is idempotent).
        """
        if batch.num_graphs > 0:
            batch.orient_molecule(mode="std")

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

    def _sample_condition_blocked_indices(self, batch_size: int, m: int):
        """
        Condition-blocked draw: sample conditions, then up to m DISTINCT rows
        (different terminals) within each, until batch_size rows are
        collected. Exists because condition-grouped VarGrad's cross-terminal
        signal needs multiple distinct terminals per condition per batch,
        which independent row draws only provide via birthday collisions
        (~B^2/2N of rows at 10k conditions). `repeats` tiling (same-terminal
        rollouts -- the TBC/MLE axis) applies downstream exactly as for the
        plain path: this changes only WHICH distinct rows are drawn, so all
        layout conventions (terminal-major tiles, update_losses inds) hold.
        Conditions with a single row carry no cross-terminal information and
        are skipped; if eligible conditions can't fill the budget, the
        remainder falls back to uniform draws, degrading gracefully to the
        old behavior on thin or singleton-heavy buffers. Row choice within a
        condition is uniform without replacement -- deliberately no energy
        stratification: trust the buffer not to be degenerate.
        """
        n = len(self)
        cid = np.asarray(self.batch.condition_id.detach().cpu().flatten())
        order = np.argsort(cid, kind="stable")
        sorted_cid = cid[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_cid[1:] != sorted_cid[:-1]])
        counts = np.diff(np.r_[boundaries, sorted_cid.size])

        eligible = np.flatnonzero(counts >= 2)
        budget = min(batch_size, n)
        chosen = []
        n_chosen = 0
        for g in np.random.permutation(eligible):
            take = min(m, int(counts[g]), budget - n_chosen)
            if take <= 0:
                break
            rows = order[boundaries[g]:boundaries[g] + counts[g]]
            chosen.append(np.random.choice(rows, size=take, replace=False))
            n_chosen += take
        inds = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)

        if inds.size < batch_size:  # thin/singleton-heavy buffer: top off uniformly
            extra = np.random.choice(n, size=batch_size - inds.size, replace=batch_size > n)
            inds = np.concatenate([inds, extra])
        return inds.astype(np.int64)

    def _sample_indices(
            self,
            batch_size: int,
            replace: Optional[bool] = None,
            repeats: int = 1,
            p: Optional[np.ndarray] = None,
            beta: float = 0.0,  # fraction drawn uniformly
            condition_block_m: int = 0,
    ):
        n = len(self)

        if n == 0:
            raise ValueError("Cannot sample from an empty SimpleDataset.")

        if condition_block_m >= 2:
            # blocked draws bypass the weighted/p machinery (bwd training draws
            # pass weighted=False anyway) -- see _sample_condition_blocked_indices
            inds = self._sample_condition_blocked_indices(batch_size, condition_block_m)
            if repeats > 1:
                inds = np.repeat(inds, repeats)
            return inds

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
            p: Optional[np.ndarray] = None,
    ):
        # p, if given, overrides the built-in loss-weighted distribution entirely
        # (e.g. an externally-computed per-condition z_gap weighting) -- weighted/
        # temperature are ignored in that case.
        if p is None:
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
            weighted: bool = False,
            temperature: Optional[float] = None,
            beta: Optional[float] = None,
            return_traj: bool = False,
            p: Optional[np.ndarray] = None,
            condition_block_m: int = 0,
    ):
        # p, if given, overrides the built-in loss-weighted distribution entirely
        # (e.g. an externally-computed per-condition z_gap weighting) -- weighted/
        # temperature are ignored in that case.
        if p is None:
            p = self._loss_weights(temperature) if weighted else None
        inds = self._sample_indices(batch_size, replace=replace, repeats=repeats, p=p, beta=beta,
                                    condition_block_m=condition_block_m)
        self._bump_counts(inds)

        # No data_list round trip. Storage is std-oriented at admission
        # (_orient_stored_batch), so draws need no per-draw orientation.
        graphs = self.batch.subsample_new_batch(inds)
        graphs = self._drop_keys(graphs, exclude_keys)

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
            p: Optional[np.ndarray] = None,
            condition_block_m: int = 0,
    ):
        """
        Infinite random-batch generator. Use next() on it.

        return_traj appends the sampled [batch, traj_length, dim] trajectory
        tensor to the yielded tuple (after y for "tensors" mode, after the
        graphs batch for "graphs" mode, and before inds if return_inds is
        also set).

        p, if given, is used directly as the per-row sampling distribution
        (bypassing the built-in loss-weighted one from weighted/temperature) --
        e.g. an externally-computed per-condition z_gap weighting over this
        buffer's rows. beta (fraction drawn uniformly instead) still applies
        on top of it.
        """
        assert mode in ("tensors", "graphs")

        while True:
            if mode == "tensors":
                x, y, traj, inds = self.sample_tensors(batch_size,
                                                       repeats=repeats, weighted=weighted, temperature=temperature,
                                                       beta=beta, return_traj=return_traj, p=p)
                result = (x, y)
                if return_traj:
                    result = result + (traj,)
                if return_inds:
                    result = result + (inds,)
                yield result

            else:
                graphs, inds, traj = self.sample_graphs(batch_size,
                                                        repeats=repeats, weighted=weighted, temperature=temperature,
                                                        beta=beta, return_traj=return_traj, p=p,
                                                        condition_block_m=condition_block_m)
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

    @torch.no_grad()
    def update_logw_stats(
            self,
            logw,
            indices,
            beta: float = 0.9,
    ):
        """
        Update per-sample rolling estimates of the log importance weight

            logw = log_r + log_pb - log_pf

        under the (approximately) current policy. Maintains, via EMA:

            ema_logw       ~ E[logw]           (Jensen / lower-bound log Z estimate)
            ema_logw_sq    ~ E[logw ** 2]       (used to derive logw_std)
            ema_log_z_emp  ~ log E[exp(logw)]   (empirical / upper log Z estimate)

        ema_log_z_emp is updated in log-space via logaddexp so it stays an
        EMA of exp(logw) without overflowing. z_gap and logw_std are exposed
        as derived properties rather than stored directly, so they always
        stay consistent with the underlying EMAs.
        """
        logw = torch.as_tensor(logw, dtype=self.ema_logw.dtype).detach().cpu().flatten()
        indices = torch.as_tensor(indices, dtype=torch.long)

        if len(logw) != len(indices):
            raise ValueError(
                f"logw and indices must have same length, got "
                f"{len(logw)} and {len(indices)}."
            )

        old_mean = self.ema_logw[indices]
        old_sq = self.ema_logw_sq[indices]
        old_log_z = self.ema_log_z_emp[indices]

        nan_mask = torch.isnan(old_mean)

        new_mean = torch.where(nan_mask, logw, beta * old_mean + (1.0 - beta) * logw)
        new_sq = torch.where(nan_mask, logw ** 2, beta * old_sq + (1.0 - beta) * logw ** 2)

        log_beta = math.log(beta)
        log_1m_beta = math.log(1.0 - beta)
        new_log_z = torch.where(
            nan_mask,
            logw,
            torch.logaddexp(log_beta + old_log_z, log_1m_beta + logw),
        )

        # handle duplicates: last write wins (same as sequential loop)
        self.ema_logw[indices] = new_mean
        self.ema_logw_sq[indices] = new_sq
        self.ema_log_z_emp[indices] = new_log_z

    @property
    def z_jensen(self):
        """Per-sample rolling Jensen (lower-bound) log Z estimate: E[logw]."""
        return self.ema_logw

    @property
    def z_emp(self):
        """Per-sample rolling empirical log Z estimate: log E[exp(logw)]."""
        return self.ema_log_z_emp

    @property
    def z_gap(self):
        """Per-sample rolling gap z_emp - z_jensen (>= 0 by Jensen's inequality)."""
        return self.ema_log_z_emp - self.ema_logw

    @property
    def logw_std(self):
        """Per-sample rolling std of logw, from EMA[logw] and EMA[logw ** 2]."""
        return torch.sqrt(torch.clamp(self.ema_logw_sq - self.ema_logw ** 2, min=0.0))

    # ---------------------------------------------------------------------
    # Mutation
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def add(self, data, traj: Optional[torch.Tensor] = None, init_loss: Optional[torch.Tensor] = None):
        """
        Append new graphs.

        Accepts either a list[Data] or an already-collated Batch. No data_list
        round trip if a Batch is provided.

        traj, if given, is a [k, traj_length, dim] tensor of per-entry
        trajectories aligned with the k new graphs being added.

        init_loss, if given, seeds ema_loss for the k new entries instead of
        leaving them NaN. Accepts a scalar or a [k] tensor.
        """
        if isinstance(data, list) and len(data) == 0:
            return

        new_batch = self._as_batch(data).to(self.device)

        if new_batch.num_graphs == 0:
            return

        self._drop_keys(new_batch, getattr(self, 'exclude_keys', ()))
        self._orient_stored_batch(new_batch)
        new_x, new_y = self._compute_xy(new_batch)

        # _compute_xy's latent transform lazily builds the space-group caches on
        # new_batch; strip both sides so append_batch never sees one batch with
        # the attr and the other without. They rebuild on demand.
        strip_lazy_sg_caches(self.batch)
        strip_lazy_sg_caches(new_batch)

        # validate=False: the shared-metadata equality checks are torch.equal
        # on device tensors -- each one is a stream sync, priced at whatever
        # the GPU has queued (the buffer-churn spike mechanism). This admission
        # path runs every train step on homogeneous same-pipeline batches;
        # shared metadata here is uniform by construction (_as_batch enforces
        # max_z_prime, lazy SG caches are stripped above).
        self.batch = self.batch.append_batch(new_batch, validate=False)

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
        if init_loss is None:
            new_ema_loss = torch.full((k,), float("nan"), dtype=self.ema_loss.dtype)
        else:
            new_ema_loss = torch.as_tensor(init_loss, dtype=self.ema_loss.dtype).detach().cpu().flatten()
            if new_ema_loss.numel() == 1:
                new_ema_loss = new_ema_loss.expand(k).clone()
            assert new_ema_loss.shape[0] == k, \
                f"init_loss has {new_ema_loss.shape[0]} entries, expected {k} to match added batch size"

        self.ema_loss = torch.cat(
            [
                self.ema_loss,
                new_ema_loss,
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

        new_nan = torch.full((k,), float("nan"), dtype=torch.float32)
        self.ema_logw = torch.cat([self.ema_logw, new_nan], dim=0)
        self.ema_logw_sq = torch.cat([self.ema_logw_sq, new_nan.clone()], dim=0)
        self.ema_log_z_emp = torch.cat([self.ema_log_z_emp, new_nan.clone()], dim=0)

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
        self.ema_logw = self.ema_logw[keep_cpu]
        self.ema_logw_sq = self.ema_logw_sq[keep_cpu]
        self.ema_log_z_emp = self.ema_log_z_emp[keep_cpu]

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


class ConditionLogZTracker:
    """
    Persistent per-condition EMA of the empirical log Z, decoupled from any
    buffer: keyed by an immutable integer `condition_id` (a deterministic
    mixed-radix combination of whichever discrete conditioning axes are
    active -- see energy_function.condition_samples), never by buffer row
    index and never by hashing the (potentially large, float) condition
    vector itself.

    Storage is a flat preallocated tensor per stat, sized to the full
    condition library (`library_size`), not a Python dict -- so cost is
    O(unique ids touched per update/lookup call), independent of both the
    library size and the condition vector's dimensionality (only a handful
    of scalars are ever stored per id).

    Also tracks a per-condition running minimum energy (`best_energy`),
    independent of the logw/log_Z EMA machinery above -- an exact order
    statistic (monotone, no beta/decay/variance concerns), updated via a
    plain scatter-min rather than any EMA math. See update_best_energy().

    EMA math (logw, logw_sq, log_z_emp via logaddexp) is structurally like
    CrystalBuffer.update_logw_stats, applied to the group-mean/group-logsumexp
    of whichever samples in a given update() call share a condition_id --
    duplicates within one call (e.g. repeats-tiled trajectories) are folded
    into a single observation for that step rather than applied sequentially.
    Unlike CrystalBuffer's per-sample version, the mixing weight here is NOT
    the fixed (1 - beta): CrystalBuffer's updates are inherently one sample
    at a time, so a fixed weight is fine, but a single ConditionLogZTracker
    update() call can fold together anywhere from 1 to hundreds of samples
    for the same condition_id (depending on how that condition happened to
    get sampled this step). A fixed weight would let a 1-sample step swing
    the estimate exactly as much as a 500-sample step. Instead, `half_life_steps`
    decays a per-condition `effective_count` (a time-discounted running
    sample size, keyed off elapsed training steps -- see update()'s
    docstring for why), and each step's actual mixing weight is its share of
    (decayed old effective_count + this step's evidence) -- see update()'s
    docstring.
    """

    def __init__(self, library_size: int, min_visits: int = 20, half_life_steps: float = 7.0,
                 trim_frac: float = 0.1, max_batch_weight: float = 200.0,
                 discovery_half_life_steps: float = 200.0):
        self.library_size = library_size
        self.min_visits = min_visits
        self.half_life_steps = half_life_steps
        self.trim_frac = trim_frac
        self.max_batch_weight = max_batch_weight
        self.discovery_half_life_steps = discovery_half_life_steps
        self.ema_logw = torch.full((library_size,), float("nan"), dtype=torch.float32)
        self.ema_logw_sq = torch.full((library_size,), float("nan"), dtype=torch.float32)
        self.ema_log_z_emp = torch.full((library_size,), float("nan"), dtype=torch.float32)
        self.count = torch.zeros((library_size,), dtype=torch.long)
        self.effective_count = torch.zeros((library_size,), dtype=torch.float32)
        # -1 sentinel ("never updated") for conditions not yet visited -- see
        # update()'s docstring for why decay is keyed off elapsed training
        # steps rather than elapsed update() calls.
        self.last_update_step = torch.full((library_size,), -1, dtype=torch.long)
        self.best_energy = torch.full((library_size,), float("inf"), dtype=torch.float32)
        # per-condition EMA of |logw - log_Z_learned| -- how far the network's
        # OWN Z prediction currently is from the fresh, per-sample importance
        # weight, live (not smoothed through ema_logw). See update_z_residual/
        # rms_z_lag.
        self.z_resid_ema = torch.full((library_size,), float("nan"), dtype=torch.float32)
        # SIGNED batch-mean residual EMA, sharing z_resid's evidence/decay state:
        # the location/calibration component of the z error, decoupled from
        # spread -- see update_z_residual/rms_z_bias
        self.z_bias_ema = torch.full((library_size,), float("nan"), dtype=torch.float32)
        self.z_resid_effective_count = torch.zeros((library_size,), dtype=torch.float32)
        self.z_resid_last_step = torch.full((library_size,), -1, dtype=torch.long)
        # discovery-rate telemetry over best_energy: update_best_energy()
        # accumulates strict per-condition minimum improvements (count/depth)
        # and first visits into the _window_* scalars; pop_discovery_stats()
        # drains them into time-decayed per-step rate EMAs. Pure monitoring --
        # nothing here feeds back into training. See pop_discovery_stats.
        self.minima_improved_total = 0
        self.minima_depth_total = 0.0
        self._window_improved = 0
        self._window_depth = 0.0
        self._window_first_visits = 0
        self.discovery_rate_ema = 0.0
        self.discovery_depth_rate_ema = 0.0
        self.discovery_last_step = -1

    def __len__(self):
        return self.library_size

    @torch.no_grad()
    def update(self, condition_id, logw, step: int,
               trim_frac: Optional[float] = None, max_batch_weight: Optional[float] = None,
               half_life_steps: Optional[float] = None):
        """
        Effective-sample-size-weighted EMA update. `step` is the caller's
        current global training step (self.step_ind); it drives the decay
        of a per-condition `effective_count`:

            decay             = 0.5 ** (1 / half_life_steps)
            delta_t           = step - last_update_step[condition]   (>= 0)
            decayed_eff_count = old_eff_count * decay ** delta_t
            new_eff_count     = decayed_eff_count + evidence_count
            w_new             = evidence_count / new_eff_count

        Decay is keyed off elapsed *training steps* since that condition's
        own last visit, not elapsed *update() calls*. The two are not the
        same thing: call() cadence is wildly uneven across conditions (a
        library of hundreds of conditions might see any given one only once
        every dozens of steps, and phase/loss-coefficient gating means not
        every train step even calls update() for every mode), so decaying
        by a flat factor "per call" gives old evidence a forgetting rate
        that implicitly depends on how often the condition happened to be
        revisited -- not a stable, portable notion of "how stale is this."
        Decaying by elapsed steps instead gives half_life_steps a fixed,
        literal meaning (the number of *training steps* for old evidence's
        weight to halve) that doesn't need retuning as visit frequency
        varies across conditions or across differently-sized condition
        libraries/configs.

        w_new is this step's actual mixing weight into the running
        estimate -- it's 1.0 on a condition's first-ever visit (old_eff_count
        == 0, so the running estimate is just set to this step's value) and
        shrinks as old_eff_count grows relative to evidence_count, so a step
        with many samples for a condition properly outweighs a step with
        few, instead of both getting the same fixed decay. half_life_steps
        still controls the forgetting rate of old evidence (effective_count
        itself decays with elapsed time), it just no longer doubles as the
        per-call weight.

        evidence_count is NOT simply counts_this_step (the raw number of
        samples this call saw for a condition) -- it's that, further capped
        by (a) the batch's effective sample size (ESS = (sum w)^2 / sum w^2
        on the shifted importance weights w = exp(logw - max), the standard
        IS diagnostic for "how much real information is in this batch") and
        (b) a hard ceiling `max_batch_weight`. Two failure modes this guards
        against, both observed in practice:

        - A batch whose weights are internally degenerate (dominated by one
          or two samples, e.g. a badly-calibrated off-policy trajectory)
          has ESS << counts_this_step regardless of how large the batch is,
          so it can't buy outsized trust just by being big.
        - Even a perfectly well-behaved large batch (ESS ~= counts_this_step)
          -- e.g. a pooled eval batch orders of magnitude larger than a
          train-step batch -- is capped at max_batch_weight, so no single
          update() call can claim more trust than a few hundred samples'
          worth, however large it actually was. Without this, a huge but
          otherwise-clean batch could still nearly overwrite hundreds of
          steps of accumulated training-time evidence in one call, purely
          because of its size (empirically: a persistent "sawblade" pattern
          coinciding with eval cycles).

        Note evidence_count only affects *how much this call is trusted*,
        not the *value* being mixed in -- self.count (lifetime observation
        count, used for the min_visits gate) still accumulates the true,
        uncapped counts_this_step.

        mean_logw (the arithmetic mean fed into the Jensen/lower-bound
        estimate ema_logw) is a trimmed mean within each condition group --
        drop the top/bottom `trim_frac` of that group's logw values by rank
        before averaging. This protects the *value* of a batch's mean from
        a minority of outliers (e.g. unbounded-below energy blowups from
        clashing geometries); it's a separate, complementary concern from
        evidence_count above, which protects the *trust* given to the
        resulting value regardless of how it was computed. Small train-step
        batches rarely have enough samples per group to trim anything, so
        this is essentially a no-op there and only bites on large batches.
        group_log_mean_exp (the empirical estimate, via logsumexp -- see
        below) is left untrimmed, deliberately: it's already outlier-robust
        in the opposite direction (dominated by the best samples, via
        exp()), and keeping it untrimmed preserves its role as an
        independent cross-check -- a large persistent gap between it and
        ema_logw (z_gap) is a signal the Jensen estimate is still being
        distorted, and trimming both would mute that signal.
        """
        trim_frac = self.trim_frac if trim_frac is None else trim_frac
        max_batch_weight = self.max_batch_weight if max_batch_weight is None else max_batch_weight
        half_life_steps = self.half_life_steps if half_life_steps is None else half_life_steps
        decay_per_step = 0.5 ** (1.0 / half_life_steps)
        condition_id = torch.as_tensor(condition_id, dtype=torch.long).detach().cpu().flatten()
        logw = torch.as_tensor(logw, dtype=torch.float32).detach().cpu().flatten()

        if condition_id.numel() == 0:
            return

        if condition_id.shape[0] != logw.shape[0]:
            raise ValueError(
                f"condition_id and logw must have same length, got "
                f"{condition_id.shape[0]} and {logw.shape[0]}."
            )

        n = logw.shape[0]
        unique_ids, inverse = torch.unique(condition_id, return_inverse=True)
        k = unique_ids.shape[0]

        counts_this_step = torch.zeros(k, dtype=torch.float32).scatter_add_(
            0, inverse, torch.ones_like(logw))

        # trimmed mean/mean-of-squares per condition group: sort samples by
        # (group, value) via a value-sort followed by a stable group-sort,
        # then rank each sample within its group to mask off the top/bottom
        # trim_k on each side before averaging. trim_k is floor(trim_frac *
        # count), capped so at least one sample always survives.
        value_order = torch.argsort(logw)
        group_order = torch.argsort(inverse[value_order], stable=True)
        sorted_idx = value_order[group_order]
        sorted_group = inverse[sorted_idx]
        sorted_logw = logw[sorted_idx]

        group_start = torch.zeros(k, dtype=torch.long)
        group_start[1:] = torch.cumsum(counts_this_step, dim=0)[:-1].long()
        rank_in_group = torch.arange(n) - group_start[sorted_group]

        counts_long = counts_this_step.long()
        trim_k = torch.clamp((trim_frac * counts_this_step).floor().long(),
                             max=(counts_long - 1) // 2)
        kept_counts = counts_this_step - 2 * trim_k.float()
        keep_mask = (rank_in_group >= trim_k[sorted_group]) & \
                    (rank_in_group < counts_long[sorted_group] - trim_k[sorted_group])

        trimmed_logw = torch.where(keep_mask, sorted_logw, torch.zeros_like(sorted_logw))
        mean_logw = torch.zeros(k, dtype=torch.float32).scatter_add_(
            0, sorted_group, trimmed_logw) / kept_counts
        mean_logw_sq = torch.zeros(k, dtype=torch.float32).scatter_add_(
            0, sorted_group, torch.where(keep_mask, sorted_logw ** 2, torch.zeros_like(sorted_logw))) / kept_counts

        # group logsumexp(logw) - log(count), computed manually for numerical
        # stability (no built-in scatter-logsumexp reduction) -- over the
        # FULL untrimmed batch, deliberately (see docstring above). The same
        # shifted exponentials also give us this batch's per-group effective
        # sample size (ESS), used below to cap evidence_count.
        group_max = torch.full((k,), float("-inf"), dtype=torch.float32).scatter_reduce_(
            0, inverse, logw, reduce="amax", include_self=True)
        shifted = (logw - group_max[inverse]).exp()
        sum_exp = torch.zeros(k, dtype=torch.float32).scatter_add_(0, inverse, shifted)
        sum_exp_sq = torch.zeros(k, dtype=torch.float32).scatter_add_(0, inverse, shifted ** 2)
        group_log_mean_exp = group_max + sum_exp.log() - counts_this_step.log()
        ess = sum_exp ** 2 / sum_exp_sq  # in [1, counts_this_step], scale-invariant

        evidence_count = torch.clamp(ess, max=max_batch_weight)

        old_mean = self.ema_logw[unique_ids]
        old_sq = self.ema_logw_sq[unique_ids]
        old_log_z = self.ema_log_z_emp[unique_ids]
        old_eff_count = self.effective_count[unique_ids]
        old_last_step = self.last_update_step[unique_ids]

        nan_mask = torch.isnan(old_mean)

        # delta_t clamped to >= 0 as a defensive guard (step should be
        # monotonically non-decreasing across calls); for a condition's
        # first-ever visit old_last_step is the -1 sentinel, so delta_t is
        # large and decayed_eff_count underflows toward 0 harmlessly -- that
        # path is bypassed by nan_mask below regardless.
        delta_t = torch.clamp((step - old_last_step).float(), min=0.0)
        decayed_eff_count = old_eff_count * decay_per_step ** delta_t

        # evidence_count > 0 for every id in unique_ids (they only appear here
        # because they were observed this step, and ESS >= 1 for any nonempty
        # group), so new_eff_count > 0 always -- no div-by-zero. w_new == 1.0
        # exactly on a condition's first visit.
        new_eff_count = decayed_eff_count + evidence_count
        w_new = evidence_count / new_eff_count

        new_mean = torch.where(nan_mask, mean_logw, (1.0 - w_new) * old_mean + w_new * mean_logw)
        new_sq = torch.where(nan_mask, mean_logw_sq, (1.0 - w_new) * old_sq + w_new * mean_logw_sq)

        log_w_new = torch.log(w_new)
        log_1m_w_new = torch.log1p(-w_new)  # log1p for precision as w_new -> 0 (well-established conditions)
        new_log_z = torch.where(
            nan_mask,
            group_log_mean_exp,
            torch.logaddexp(log_1m_w_new + old_log_z, log_w_new + group_log_mean_exp),
        )

        self.ema_logw[unique_ids] = new_mean
        self.ema_logw_sq[unique_ids] = new_sq
        self.ema_log_z_emp[unique_ids] = new_log_z
        self.effective_count[unique_ids] = new_eff_count
        self.count[unique_ids] += counts_this_step.long()
        self.last_update_step[unique_ids] = int(step)

    @torch.no_grad()
    def update_z_residual(self, condition_id, logw, log_Z_learned, step: int,
                          half_life_steps: Optional[float] = None):
        """
        Per-condition EMA of |logw - log_Z_learned|: how far the network's
        own Z prediction currently is for this condition, measured directly
        against this call's fresh per-sample logw -- not against ema_logw,
        which is itself smoothed and therefore a step removed from "right
        now". This is what rms_z_lag() reduces over the library to give the
        controller a richer, per-condition-aware miscalibration signal than
        a single global batch-mean (quick_tb_stats' jensen_z_err) can: a
        mean lets a majority of well-calibrated conditions dilute away a
        badly-miscalibrated minority, which is exactly the failure mode this
        whole tracker exists to guard against elsewhere (see update()'s
        trim_frac docstring) -- no reason the detection signal should be
        vulnerable to the same thing the estimator itself was hardened
        against.

        Same time-based decay and evidence-capping as update() (elapsed
        training steps since this condition's own last z-residual update,
        capped effective evidence per call), for the same reasons -- kept
        as separate state (z_resid_*) so this monitoring signal's own decay
        dynamics can't interact with the primary logw/log_z_emp EMA math.
        """
        half_life_steps = self.half_life_steps if half_life_steps is None else half_life_steps
        decay_per_step = 0.5 ** (1.0 / half_life_steps)

        condition_id = torch.as_tensor(condition_id, dtype=torch.long).detach().cpu().flatten()
        logw = torch.as_tensor(logw, dtype=torch.float32).detach().cpu().flatten()
        log_Z_learned = torch.as_tensor(log_Z_learned, dtype=torch.float32).detach().cpu().flatten()

        if condition_id.numel() == 0:
            return

        # two views of the same residual stream, EMA'd side by side with shared
        # evidence/decay state: the ABS mean (z_resid_ema, behind rms_z_lag) is
        # floored at the per-condition MAD of log w no matter how well Z is
        # placed -- a spread metric. The SIGNED batch mean (z_bias_ema, behind
        # rms_z_bias) averages the spread away BEFORE the abs, so it goes to ~0
        # exactly when Z(c) sits on the stream's mean(log w) -- the pure
        # location/calibration component. zerr >> z_bias means the 'error' is
        # spread (only policy work shrinks it); z_bias large means Z genuinely
        # lags (Z training shrinks it). The two together disambiguate what a
        # high zerr alone cannot.
        resid_signed = logw - log_Z_learned
        residual = resid_signed.abs()

        unique_ids, inverse = torch.unique(condition_id, return_inverse=True)
        k = unique_ids.shape[0]
        counts_this_step = torch.zeros(k, dtype=torch.float32).scatter_add_(
            0, inverse, torch.ones_like(residual))
        mean_resid = torch.zeros(k, dtype=torch.float32).scatter_add_(
            0, inverse, residual) / counts_this_step
        mean_bias = torch.zeros(k, dtype=torch.float32).scatter_add_(
            0, inverse, resid_signed) / counts_this_step

        old_mean = self.z_resid_ema[unique_ids]
        old_bias = self.z_bias_ema[unique_ids]
        old_eff_count = self.z_resid_effective_count[unique_ids]
        old_last_step = self.z_resid_last_step[unique_ids]
        nan_mask = torch.isnan(old_mean)
        # separate NaN mask: checkpoints predating z_bias_ema restore it as NaN
        # while z_resid_ema is warm, so the two can disagree after a reload
        bias_nan_mask = torch.isnan(old_bias)

        delta_t = torch.clamp((step - old_last_step).float(), min=0.0)
        decayed_eff_count = old_eff_count * decay_per_step ** delta_t
        evidence_count = torch.clamp(counts_this_step, max=self.max_batch_weight)
        new_eff_count = decayed_eff_count + evidence_count
        w_new = evidence_count / new_eff_count

        new_mean = torch.where(nan_mask, mean_resid, (1.0 - w_new) * old_mean + w_new * mean_resid)
        new_bias = torch.where(bias_nan_mask, mean_bias, (1.0 - w_new) * old_bias + w_new * mean_bias)

        self.z_resid_ema[unique_ids] = new_mean
        self.z_bias_ema[unique_ids] = new_bias
        self.z_resid_effective_count[unique_ids] = new_eff_count
        self.z_resid_last_step[unique_ids] = int(step)

    @torch.no_grad()
    def rms_z_lag(self, min_effective_count: Optional[float] = None):
        """
        RMS of z_resid_ema over conditions with enough evidence to trust
        (effective count >= min_effective_count, defaulting to min_visits).
        RMS rather than mean deliberately: it doesn't let a majority of
        well-calibrated conditions dilute away a badly-miscalibrated
        minority the way an arithmetic mean would (see update_z_residual's
        docstring). Returns 0.0 (not NaN) when no condition currently has
        enough evidence, so this is always safe to compare directly against
        a threshold.
        """
        min_effective_count = self.min_visits if min_effective_count is None else min_effective_count
        mask = (~torch.isnan(self.z_resid_ema)) & (self.z_resid_effective_count >= min_effective_count)
        if not mask.any():
            return 0.0
        return torch.sqrt((self.z_resid_ema[mask] ** 2).mean()).item()

    @torch.no_grad()
    def rms_z_bias(self, min_effective_count: Optional[float] = None):
        """
        RMS over trusted conditions of the SIGNED z-residual EMA (z_bias_ema)
        -- the location/calibration component of the z error, decoupled from
        spread. rms_z_lag EMAs |logw - Z| per sample, which is floored at the
        per-condition MAD of log w no matter how well Z is placed (a spread
        metric wearing a calibration metric's name); the signed batch mean
        averages the spread away BEFORE any abs, so this goes to ~0 exactly
        when Z(c) sits on the stream's mean(log w). Read the pair together:
        zerr >> z_bias means the 'error' is spread (only policy work shrinks
        it); z_bias large means Z genuinely lags (Z training shrinks it).
        Same trust mask/state as rms_z_lag; returns 0.0 when nothing is
        trusted (same fire-on-large semantics as rms_z_lag).
        """
        min_effective_count = self.min_visits if min_effective_count is None else min_effective_count
        mask = (~torch.isnan(self.z_bias_ema)) & (self.z_resid_effective_count >= min_effective_count)
        if not mask.any():
            return 0.0
        return torch.sqrt((self.z_bias_ema[mask] ** 2).mean()).item()

    @torch.no_grad()
    def rms_logw_std(self, min_visits: Optional[int] = None):
        """
        RMS over trusted conditions (count >= min_visits, non-nan) of the
        per-condition std of logw implied by the tracker's running moments:
        sqrt(clamp(ema_logw_sq - ema_logw^2, 0)). The phase-2
        (variance-conditioning) exit signal: 'the policy is self-consistent
        enough for Z learning to get traction', measured as tightness of the
        per-condition log-importance-weight distribution -- NOT distance to
        the true target. RMS over conditions for the same anti-dilution
        reason as rms_z_lag. Mildly underestimates tails (both moments come
        from trim_frac-trimmed group means), fine for a gate.

        Returns +inf (not 0) when no condition has enough evidence: unlike
        rms_z_lag, whose callers act on LARGE values (so an ignorant 0 is
        the safe default), this gates a phase transition that fires on SMALL
        values -- a gate that must wait for proof of tightness should never
        pass on ignorance.
        """
        min_visits = self.min_visits if min_visits is None else min_visits
        var = self.ema_logw_sq - self.ema_logw ** 2
        mask = (~torch.isnan(var)) & (self.count >= min_visits)
        if not mask.any():
            return float('inf')
        return torch.sqrt(var[mask].clamp(min=0.0).mean()).item()

    @torch.no_grad()
    def lookup(self, condition_id):
        """
        Returns (log_z_target, mask) where mask is True for entries that
        have been visited at least `min_visits` times. log_z_target is
        self.ema_logw (the Jensen lower-bound / EMA of mean log importance
        weight), NOT self.ema_log_z_emp (EMA of logsumexp(logw) - log(count)).

        These estimate different things, and which one is safe to put here
        (i.e. to actually feed gradients) is a separate question from which
        one is the statistically "correct" estimator of log Z:

        ema_log_z_emp IS the (nearly) unbiased one -- the importance-sampling
        identity E_pi[exp(logw)] = Z holds exactly for any policy pi with
        full support, so mean(exp(logw)) is exactly unbiased for Z, and
        ema_log_z_emp = log(mean(exp(logw))) is consistent for log Z with
        only a small, n-shrinking finite-sample bias. ema_logw is an
        unbiased estimator of a *different*, structurally lower quantity,
        E_pi[logw] <= log Z (Jensen), with a permanent gap (see z_gap below)
        that doesn't vanish with more samples, only at the TB fixed point.

        But being unbiased for Z requires giving large importance weights
        their full, unclipped influence -- which is exactly what makes
        ema_log_z_emp dangerous as a live gradient target: logsumexp has no
        1/n dilution the way a plain mean does, so one badly-calibrated
        off-policy sample (e.g. log_pf collapsing under an under-warmed
        proposal) can send it to billions of nats in a single update, and
        because it was feeding the TB residual, that corrupted value then
        pushed log_pf/log_pb in a fixed direction every step, which can
        actively degrade calibration network-wide and produce more of the
        same mismatched-trajectory events -- a closed loop that measurably
        broke a live run. ema_logw's arithmetic mean dilutes any single
        sample by 1/n and is additionally rank-trimmed in update() above, so
        it can't blow up the same way; it's the "wrong" (structurally
        biased) number asymptotically, but the safe one to close a gradient
        loop around. ema_log_z_emp is still fully tracked and logged
        (ten_step_reporting, log_condition_log_z_stats) as a diagnostic --
        its stability there is a genuinely useful convergence signal (see
        z_gap) -- it's just not what gets fed back into training. log_z_target
        is 0 (not NaN) wherever mask is False, so it's always safe to use
        directly in a torch.where without a separate NaN-guard.
        """
        condition_id = torch.as_tensor(condition_id, dtype=torch.long).detach().cpu().flatten()
        log_z = self.ema_logw[condition_id]
        mask = (self.count[condition_id] >= self.min_visits) & (~torch.isnan(log_z))
        log_z = torch.nan_to_num(log_z, nan=0.0)
        return log_z, mask

    @property
    def z_gap(self):
        """Per-condition rolling gap ema_log_z_emp - ema_logw (>= 0 by Jensen's inequality)."""
        return self.ema_log_z_emp - self.ema_logw

    @torch.no_grad()
    def lookup_z_gap(self, condition_id):
        """
        Returns (gap, mask) where gap is the per-condition Jensen/logmeanexp
        divergence (z_emp - z_jensen) and mask is True for entries visited at
        least once. Unlike lookup()'s min_visits gate (which guards a value
        that gets fed back into training), any single visit already gives an
        exact-enough gap to prioritize sampling attention toward, so no
        warm-up threshold is applied here. gap is 0 (not NaN) wherever mask
        is False, so it's always safe to use directly without a NaN-guard.
        """
        condition_id = torch.as_tensor(condition_id, dtype=torch.long).detach().cpu().flatten()
        gap = self.z_gap[condition_id]
        mask = self.count[condition_id] >= 1
        gap = torch.nan_to_num(gap, nan=0.0)
        return gap, mask

    @torch.no_grad()
    def update_best_energy(self, condition_id, energy):
        """
        Per-condition running minimum energy. Unlike update()'s EMA math,
        this needs no torch.unique/inverse pass: torch.scatter_reduce_
        already folds together duplicate condition_id entries within one
        call natively, so a single scatter-min directly against the
        persistent tensor is sufficient. best_energy is initialized to
        +inf, so a condition's first-ever visit is handled by the reduce
        itself (min(inf, x) == x) with no special-casing.

        A torch.unique pass IS spent on the discovery telemetry: comparing
        each touched condition's best_energy before/after the scatter gives
        the number of strict minimum improvements this call, their summed
        depth (old - new, only defined where the old minimum was finite),
        and first visits (inf -> finite). These accumulate into the
        _window_* scalars for pop_discovery_stats to drain.
        """
        condition_id = torch.as_tensor(condition_id, dtype=torch.long).detach().cpu().flatten()
        energy = torch.as_tensor(energy, dtype=torch.float32).detach().cpu().flatten()

        if condition_id.numel() == 0:
            return

        if condition_id.shape[0] != energy.shape[0]:
            raise ValueError(
                f"condition_id and energy must have same length, got "
                f"{condition_id.shape[0]} and {energy.shape[0]}."
            )

        unique_ids = torch.unique(condition_id)
        old_best = self.best_energy[unique_ids]

        self.best_energy.scatter_reduce_(0, condition_id, energy, reduce="amin", include_self=True)

        new_best = self.best_energy[unique_ids]
        had_min = torch.isfinite(old_best)
        improved = had_min & (new_best < old_best)
        n_improved = int(improved.sum().item())
        depth = float((old_best[improved] - new_best[improved]).sum().item())
        self._window_improved += n_improved
        self._window_depth += depth
        self._window_first_visits += int((~had_min & torch.isfinite(new_best)).sum().item())
        self.minima_improved_total += n_improved
        self.minima_depth_total += depth

    @torch.no_grad()
    def pop_discovery_stats(self, step: int, half_life_steps: Optional[float] = None):
        """
        Drain the discovery-rate accumulators fed by update_best_energy()
        and fold this window's per-step rates into their EMAs -- the
        "minima discovery velocity" readout for the anchor buffer /
        conditional controller: near 0 when per-condition minima have gone
        static, rising quickly when fresh conditional minima are being
        found. `step` is the caller's current global training step
        (Modeller.step_ind); rates are in events (or energy units of
        reduction per visited condition) per TRAINING STEP, so they're
        comparable across eval cadences and window lengths.

        EMA mixing uses the same elapsed-training-steps convention as
        update(): the window's mean rate (count / elapsed) enters with
        weight 1 - 0.5 ** (elapsed / half_life), so a long window carries
        proportionally more weight and quiet windows decay the EMA toward
        0 on the same clock regardless of how often this is called.
        Intended to be drained from exactly one call site per period (the
        eval-cycle logger) -- each call resets the window.

        The count rate is scale-free (an improvement is an improvement in
        any condition's own energy scale). All depth outputs are INTENSIVE:
        the raw summed energy reduction is divided by the number of visited
        conditions (finite best_energy) at drain time, so "mean deepening
        per tracked condition" (energy units) is comparable across runs
        with different condition-library sizes. The internal accumulators
        stay raw sums (checkpoint layout unchanged); normalization happens
        only here. A single deep drop -- real or a scoring outlier -- can
        still dominate a window. Read the pair together: count rate for
        churn, depth rate for magnitude.
        """
        half_life = self.discovery_half_life_steps if half_life_steps is None else half_life_steps
        improved = self._window_improved
        n_visited = max(int(torch.isfinite(self.best_energy).sum().item()), 1)
        depth = self._window_depth / n_visited
        first_visits = self._window_first_visits
        self._window_improved = 0
        self._window_depth = 0.0
        self._window_first_visits = 0

        # fresh runs start the clock at step 0; from_state_dict seeds
        # discovery_last_step to the resume step for reloaded runs (and to
        # current_step for checkpoints predating this telemetry)
        last = self.discovery_last_step if self.discovery_last_step >= 0 else 0
        elapsed = max(int(step) - last, 1)
        self.discovery_last_step = int(step)

        alpha = 1.0 - 0.5 ** (elapsed / half_life)
        self.discovery_rate_ema += alpha * (improved / elapsed - self.discovery_rate_ema)
        self.discovery_depth_rate_ema += alpha * (depth / elapsed - self.discovery_depth_rate_ema)

        return {
            'improved': improved,
            'first_visits': first_visits,
            'depth': depth,
            'rate_ema': self.discovery_rate_ema,
            'depth_rate_ema': self.discovery_depth_rate_ema,
            'improved_total': self.minima_improved_total,
            'depth_total': self.minima_depth_total / n_visited,
        }

    @torch.no_grad()
    def lookup_best_energy(self, condition_id):
        """
        Returns (best_energy, mask) where mask is True for entries that
        have been visited at least once -- no min_visits threshold here,
        since a single observed minimum is already an exact record, not a
        noisy estimate needing warm-up. best_energy is 0 (not inf) wherever
        mask is False, so it's always safe to use directly without a
        separate inf-guard.
        """
        condition_id = torch.as_tensor(condition_id, dtype=torch.long).detach().cpu().flatten()
        best = self.best_energy[condition_id]
        mask = torch.isfinite(best)
        best = torch.where(mask, best, torch.zeros_like(best))
        return best, mask

    def state_dict(self):
        return {
            "library_size": self.library_size,
            "min_visits": self.min_visits,
            "half_life_steps": self.half_life_steps,
            "trim_frac": self.trim_frac,
            "max_batch_weight": self.max_batch_weight,
            "ema_logw": self.ema_logw.cpu(),
            "ema_logw_sq": self.ema_logw_sq.cpu(),
            "ema_log_z_emp": self.ema_log_z_emp.cpu(),
            "count": self.count.cpu(),
            "effective_count": self.effective_count.cpu(),
            "last_update_step": self.last_update_step.cpu(),
            "best_energy": self.best_energy.cpu(),
            "z_resid_ema": self.z_resid_ema.cpu(),
            "z_bias_ema": self.z_bias_ema.cpu(),
            "z_resid_effective_count": self.z_resid_effective_count.cpu(),
            "z_resid_last_step": self.z_resid_last_step.cpu(),
            "discovery_half_life_steps": self.discovery_half_life_steps,
            "minima_improved_total": self.minima_improved_total,
            "minima_depth_total": self.minima_depth_total,
            "window_improved": self._window_improved,
            "window_depth": self._window_depth,
            "window_first_visits": self._window_first_visits,
            "discovery_rate_ema": self.discovery_rate_ema,
            "discovery_depth_rate_ema": self.discovery_depth_rate_ema,
            "discovery_last_step": self.discovery_last_step,
        }

    @classmethod
    def from_state_dict(cls, state, current_step: int = 0):
        """
        current_step is only consulted as a fallback for checkpoints saved
        before last_update_step existed (see below) -- it should be the step
        training is resuming at (e.g. Modeller.step_ind after restoring it),
        not a step recorded inside `state` itself.
        """
        obj = cls.__new__(cls)
        obj.library_size = state["library_size"]
        obj.min_visits = state["min_visits"]
        # older checkpoints predate half_life_steps and stored a per-call
        # `beta` instead -- there's no exact conversion (that depends on how
        # often update() happened to be called, which is exactly what this
        # change moves away from), so just fall back to the new default
        # rather than reverse-engineer a number from the old semantics.
        obj.half_life_steps = state.get("half_life_steps", 7.0)
        obj.trim_frac = state.get("trim_frac", 0.1)
        obj.max_batch_weight = state.get("max_batch_weight", 200.0)
        obj.ema_logw = state["ema_logw"].cpu()
        obj.ema_logw_sq = state["ema_logw_sq"].cpu()
        obj.ema_log_z_emp = state["ema_log_z_emp"].cpu()
        obj.count = state["count"].cpu()
        # older checkpoints predate effective_count -- fall back to the lifetime
        # count (a mild over-estimate of "true" decayed effective count, but a
        # far better prior than 0, which would make the very next update() call
        # after reload treat every already-warmed-up condition as brand new)
        obj.effective_count = state.get(
            "effective_count", obj.count.float()).cpu()
        # older checkpoints predate last_update_step -- default to current_step
        # (not the -1 "never visited" sentinel) so the first post-reload
        # update() call sees delta_t=0 for already-warmed-up conditions,
        # rather than a huge fabricated gap that would decay effective_count
        # to ~0 and silently discard everything just restored above.
        obj.last_update_step = state.get(
            "last_update_step",
            torch.full_like(obj.count, int(current_step))).cpu()
        # older checkpoints predate best_energy -- +inf ("never visited") is the
        # only honest fallback; there's no lifetime stat to reconstruct it from
        obj.best_energy = state.get(
            "best_energy", torch.full_like(obj.count, float("inf"), dtype=torch.float32)).cpu()
        # older checkpoints predate the z-residual monitor entirely -- NaN/0/current_step
        # (never-updated sentinels) are the only honest fallback, same reasoning as above
        obj.z_resid_ema = state.get(
            "z_resid_ema", torch.full_like(obj.count, float("nan"), dtype=torch.float32)).cpu()
        # older checkpoints predate the signed-bias track -- NaN sentinel; the
        # bias_nan_mask in update_z_residual warm-starts it from the first
        # post-reload batch even where z_resid_ema is already warm
        obj.z_bias_ema = state.get(
            "z_bias_ema", torch.full_like(obj.count, float("nan"), dtype=torch.float32)).cpu()
        obj.z_resid_effective_count = state.get(
            "z_resid_effective_count", torch.zeros_like(obj.count, dtype=torch.float32)).cpu()
        obj.z_resid_last_step = state.get(
            "z_resid_last_step", torch.full_like(obj.count, int(current_step))).cpu()
        # older checkpoints predate the discovery telemetry -- zeroed
        # accumulators/EMAs are the honest fallback (rates rebuild within a
        # half-life), and discovery_last_step falls back to current_step so
        # the first post-reload pop doesn't see a huge fabricated window
        obj.discovery_half_life_steps = state.get("discovery_half_life_steps", 200.0)
        obj.minima_improved_total = state.get("minima_improved_total", 0)
        obj.minima_depth_total = state.get("minima_depth_total", 0.0)
        obj._window_improved = state.get("window_improved", 0)
        obj._window_depth = state.get("window_depth", 0.0)
        obj._window_first_visits = state.get("window_first_visits", 0)
        obj.discovery_rate_ema = state.get("discovery_rate_ema", 0.0)
        obj.discovery_depth_rate_ema = state.get("discovery_depth_rate_ema", 0.0)
        obj.discovery_last_step = state.get("discovery_last_step", int(current_step))
        return obj


def _per_condition_min(ids: torch.Tensor, values: torch.Tensor, query_ids: torch.Tensor) -> torch.Tensor:
    """
    Per-condition minimum of `values` grouped by `ids`, evaluated at each of
    `query_ids`; +inf wherever a query id has no representative in `ids`.

    Shared by AnchorBuffer.admit's per-condition admission gate and thin's
    per-condition max_size protection, so a single dominant condition's
    energy scale can never be used (via a buffer-wide scalar) to gate or
    evict anchors belonging to some other condition.
    """
    out = torch.full((query_ids.numel(),), float('inf'), dtype=torch.float32)
    if ids.numel() == 0:
        return out
    uniq_ids, inverse = torch.unique(ids, return_inverse=True)
    mins = torch.full((uniq_ids.numel(),), float('inf'), dtype=torch.float32)
    mins.scatter_reduce_(0, inverse, values, reduce='amin', include_self=True)
    pos = torch.searchsorted(uniq_ids, query_ids).clamp(max=uniq_ids.numel() - 1)
    found = uniq_ids[pos] == query_ids
    out[found] = mins[pos[found]]
    return out


def _per_condition_max(ids: torch.Tensor, values: torch.Tensor, query_ids: torch.Tensor) -> torch.Tensor:
    """
    Per-condition maximum of `values` grouped by `ids`, evaluated at each of
    `query_ids`; -inf wherever a query id has no representative in `ids`.

    Mirrors _per_condition_min but for reward (to be maximized rather than
    minimized) -- used to anchor train.py's reward-ramp kwargs to each
    condition's own best achievable reward instead of the anchor buffer's
    global max, since conditions can sit on very different reward scales
    (e.g. under temperature_conditioning=True).
    """
    out = torch.full((query_ids.numel(),), float('-inf'), dtype=torch.float32)
    if ids.numel() == 0:
        return out
    uniq_ids, inverse = torch.unique(ids, return_inverse=True)
    maxs = torch.full((uniq_ids.numel(),), float('-inf'), dtype=torch.float32)
    maxs.scatter_reduce_(0, inverse, values, reduce='amax', include_self=True)
    pos = torch.searchsorted(uniq_ids, query_ids).clamp(max=uniq_ids.numel() - 1)
    found = uniq_ids[pos] == query_ids
    out[found] = maxs[pos[found]]
    return out


def bottom_up_cluster(xx, e, d_cut, e_cut, max_new_samples: int, device):
    """
    Greedily keep the lowest-e point in each d_cut neighborhood: sort by e
    ascending, accept a point if it isn't already blocked by an accepted
    neighbor, then block everything within d_cut of it. Standalone (not a
    buffer method) so it can be reused without depending on the legacy
    CrystalReplayBuffer this pattern originated in.
    """
    sort_inds = torch.argsort(e.to(device))
    xx_sorted = xx.to(device)[sort_inds]
    e_sorted = e.to(device)[sort_inds]
    mask = e_sorted < e_cut

    blocked = torch.zeros(len(xx_sorted), dtype=torch.bool, device=device)
    keep = torch.zeros(len(xx_sorted), dtype=torch.bool, device=device)
    d_cut_squared = d_cut * d_cut
    for i in range(len(xx_sorted)):
        if not mask[i]:
            break

        if blocked[i]:
            continue

        keep[i] = True
        if torch.sum(keep) == max_new_samples:
            break

        drow = ((xx_sorted - xx_sorted[i, None, :]) ** 2).sum(-1)  # faster, skips sqrt
        nearby = drow < d_cut_squared
        blocked |= nearby

    keep_inds = sort_inds[keep]

    return keep_inds.cpu()


class AnchorBuffer(CrystalBuffer):
    """
    Permanent archive of surprising, high-quality samples: states the
    forward policy assigns low probability despite good Boltzmann weight
    (see train.py's screen_and_admit_anchors). Grows only through admit();
    nothing is evicted by ordinary training churn -- only thin() or an
    explicit purge_by_index() call can shrink it.

    ema_loss/update_losses/_loss_weights (inherited from CrystalBuffer) are
    repurposed here as the *replay priority* signal rather than a training
    loss: train.py's top_up_prior_from_anchors routes each noised child's
    freshly-measured surprise back to its parent anchor via update_losses,
    and then samples anchors for replay weighted by that EMA (with a random
    floor via sample_graphs(weighted=True, beta=...)) -- a well-learned
    anchor draws little replay; one the policy is drifting off draws more,
    automatically. purge()/purge_lowest() are still meaningless here (no
    "training loss" is ever computed against an anchor) and should not be
    called on an AnchorBuffer.

    Reward is stored explicitly per entry (self.reward) rather than derived
    on demand from the resident batch, because temperature isn't persisted
    per-graph -- condition_samples returns it as a separate tensor rather
    than writing it back onto the batch, so re-deriving reward later would
    require re-sampling temperature, which is wrong under
    temperature_conditioning=True. Energy (self.energy) is stored the same
    way and for the same reason, and is the quantity all of this class's own
    accounting (thin's energy-window/max_size trim) is actually done in:
    reward = -energy / temperature folds temperature into the number, so
    comparing raw reward across anchors sampled under different conditions
    (e.g. temperature_conditioning=True) silently conflates real energy
    differences with temperature differences. Energy has no such
    dependence, so it's the physically meaningful quantity to rank/gate on;
    reward is kept only for logging and for the reward-scale ramps
    elsewhere in train.py that were already written in reward space.

    original_surprise (self.original_surprise) is stored per entry at
    admission time and never updated afterward -- it's the one non-adaptive
    quantity on an anchor, used only by thin()'s hard-cap backstop (evict
    lowest original_surprise first on overflow). Every other quantity here
    is adaptive (ema_loss/replay-priority tracks current policy drift,
    energy tracks Emin(c) as it falls), so keeping eviction keyed to a
    frozen quantity keeps that one destructive decision outside the
    feedback loop -- see the design doc for why.

    Per-anchor condition_id/conditions need no separate storage --
    condition_samples already writes them onto the resident batch
    (mol_batch.condition_id/mol_batch.conditions), and CrystalBuffer.add/
    subsample_new_batch carry the whole batch through generically, so
    self.batch.condition_id (exposed via the condition_id property below)
    always stays aligned with self.energy/self.reward/self.x with no
    bookkeeping of its own. This is what admit()/thin() key their
    per-condition logic on: without it, admission and space-pressure
    trimming both compare across the whole buffer, so a single condition
    whose achievable energy happens to be lowest can gate out or evict every
    other condition's anchors even though each condition has its own
    distinct target.

    Novelty admission is gated entirely by surprise (see
    screen_and_admit_anchors), not by latent distance -- admit()'s
    dup_cutoff (formerly dist_cutoff) is kept only as a cheap literal-
    duplicate catch among the rare confirmed-surprising candidates handed
    to it, restricted to same-condition pairs, same as before.
    """

    def __init__(
            self,
            data,
            device,
            reward,
            energy,
            original_surprise=None,
            max_z_prime: int = 1,
            x_fn=None,
            exclude_keys: Optional[tuple] = None,
    ):
        super().__init__(data, device, max_z_prime=max_z_prime, x_fn=x_fn, y_fn=None,
                         exclude_keys=exclude_keys)
        n = len(self)
        self.reward = torch.as_tensor(reward, dtype=torch.float32).detach().cpu().flatten()
        assert self.reward.shape[0] == n, \
            f"reward has {self.reward.shape[0]} entries, expected {n} to match dataset size"
        self.energy = torch.as_tensor(energy, dtype=torch.float32).detach().cpu().flatten()
        assert self.energy.shape[0] == n, \
            f"energy has {self.energy.shape[0]} entries, expected {n} to match dataset size"
        if original_surprise is None:
            # bootstrap/legacy path (e.g. seeding from a curated dataset with no
            # rollout-based surprise measurement) -- NaN rather than 0 or -inf so
            # it never dominates a thin() hard-cap eviction ranking silently.
            self.original_surprise = torch.full((n,), float('nan'), dtype=torch.float32)
        else:
            self.original_surprise = torch.as_tensor(
                original_surprise, dtype=torch.float32).detach().cpu().flatten()
            assert self.original_surprise.shape[0] == n, \
                f"original_surprise has {self.original_surprise.shape[0]} entries, expected {n} to match dataset size"
        # previous eval-window snapshot of per-condition mean stored energy,
        # consumed/refreshed by pop_mean_energy_improvement()
        self._prev_cond_ids = None
        self._prev_cond_mean_energy = None

    @property
    def condition_id(self):
        """Per-anchor condition_id, read straight off the resident batch -- see class docstring."""
        return self.batch.condition_id.detach().cpu().flatten()

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------

    def state_dict(self):
        state = super().state_dict()
        state['reward'] = self.reward.cpu()
        state['energy'] = self.energy.cpu()
        state['original_surprise'] = self.original_surprise.cpu()
        if self._prev_cond_ids is not None:
            state['prev_cond_ids'] = self._prev_cond_ids.cpu()
            state['prev_cond_mean_energy'] = self._prev_cond_mean_energy.cpu()
        return state

    @classmethod
    def from_state_dict(cls, state, device):
        obj = super(AnchorBuffer, cls).from_state_dict(state, device)
        obj.reward = state['reward'].cpu()
        # older checkpoints predate storing energy explicitly -- fall back to
        # -reward (exact at temperature == 1; otherwise an approximation
        # until the buffer churns in fresh, correctly energy-scored
        # admissions)
        obj.energy = state.get('energy', -state['reward']).cpu()
        # older checkpoints predate original_surprise -- NaN ("unmeasured") is
        # the only honest fallback; see __init__'s legacy-path comment
        obj.original_surprise = state.get(
            'original_surprise', torch.full_like(obj.energy, float('nan'))).cpu()
        # older checkpoints predate the mean-energy-improvement snapshot --
        # None just means the first post-reload window reports nothing
        obj._prev_cond_ids = state.get('prev_cond_ids', None)
        obj._prev_cond_mean_energy = state.get('prev_cond_mean_energy', None)
        return obj

    # ---------------------------------------------------------------------
    # Monitoring
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def pop_mean_energy_improvement(self):
        """
        Windowed "the average anchor sample got this much better" readout,
        in energy units. Each call snapshots the current per-condition mean
        stored energy and returns the mean drop since the previous call's
        snapshot (prev - current, positive = the population deepened),
        averaged with equal weight over the conditions present in BOTH
        snapshots. The condition decomposition matters: conditions sit on
        very different absolute energy scales, so a raw whole-buffer mean
        delta would mostly measure cross-condition composition churn (which
        conditions happened to gain anchors), not actual deepening. For the
        same reason, conditions appearing or vanishing between snapshots
        are composition changes rather than improvement and are excluded --
        n_conditions reports how many were actually compared. Within a
        shared condition, freshly admitted anchors DO count: new low-energy
        admissions pulling that condition's mean down is exactly the
        population improving.

        Returns (improvement, n_conditions); (None, 0) on the first call,
        on an empty buffer (the stored snapshot is kept so a transient
        empty read doesn't erase the baseline), or when no condition is
        shared. Intended to be drained from exactly one call site per eval
        period (the buffer-stats logger), same convention as
        ConditionLogZTracker.pop_discovery_stats; the snapshot persists
        through state_dict so a resume doesn't fabricate a giant first
        window.
        """
        if len(self) == 0:
            return None, 0
        ids = self.condition_id
        uniq, inverse = torch.unique(ids, return_inverse=True)
        sums = torch.zeros(uniq.shape[0], dtype=torch.float64).scatter_add_(
            0, inverse, self.energy.double())
        counts = torch.bincount(inverse, minlength=uniq.shape[0]).double()
        means = (sums / counts).float()

        prev_ids, prev_means = self._prev_cond_ids, self._prev_cond_mean_energy
        self._prev_cond_ids, self._prev_cond_mean_energy = uniq, means

        if prev_ids is None or prev_ids.numel() == 0:
            return None, 0
        # checkpointed snapshots live on cpu while the buffer tensors may be on
        # device, so align before comparing (first call after a resume)
        prev_ids = prev_ids.to(uniq.device)
        prev_means = prev_means.to(means.device)
        # torch.unique output is sorted, so shared conditions match by searchsorted
        pos = torch.searchsorted(prev_ids, uniq).clamp(max=prev_ids.numel() - 1)
        shared = prev_ids[pos] == uniq
        n_shared = int(shared.sum().item())
        if n_shared == 0:
            return None, 0
        improvement = float((prev_means[pos[shared]] - means[shared]).mean().item())
        return improvement, n_shared

    # ---------------------------------------------------------------------
    # Mutation
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def add(self, data, reward, energy, original_surprise=None, traj=None, init_loss=None):
        reward = torch.as_tensor(reward, dtype=torch.float32).detach().cpu().flatten()
        energy = torch.as_tensor(energy, dtype=torch.float32).detach().cpu().flatten()
        n_before = len(self)
        super().add(data, traj=traj, init_loss=init_loss)
        k = len(self) - n_before
        assert reward.shape[0] == k, \
            f"reward has {reward.shape[0]} entries, expected {k} to match added batch size"
        assert energy.shape[0] == k, \
            f"energy has {energy.shape[0]} entries, expected {k} to match added batch size"
        self.reward = torch.cat([self.reward, reward], dim=0)
        self.energy = torch.cat([self.energy, energy], dim=0)
        if original_surprise is None:
            new_surprise = torch.full((k,), float('nan'), dtype=torch.float32)
        else:
            new_surprise = torch.as_tensor(
                original_surprise, dtype=torch.float32).detach().cpu().flatten()
            assert new_surprise.shape[0] == k, \
                f"original_surprise has {new_surprise.shape[0]} entries, expected {k} to match added batch size"
        self.original_surprise = torch.cat([self.original_surprise, new_surprise], dim=0)

    @torch.no_grad()
    def purge_by_index(self, indices_to_remove):
        n = len(self)
        if n == 0:
            return
        indices_to_remove = np.asarray(indices_to_remove, dtype=int)
        if indices_to_remove.size == 0:
            return

        drop = np.zeros(n, dtype=bool)
        drop[indices_to_remove] = True
        keep_cpu = torch.as_tensor(np.flatnonzero(~drop), dtype=torch.long)

        super().purge_by_index(indices_to_remove)
        self.reward = self.reward[keep_cpu]
        self.energy = self.energy[keep_cpu]
        self.original_surprise = self.original_surprise[keep_cpu]

    @torch.no_grad()
    def admit(self, candidate_batch, reward, energy, dup_cutoff, admit_range: Optional[float] = None,
              original_surprise=None):
        """
        candidate_batch/reward/energy are expected to already be the
        confirmed-surprising set (see train.py's screen_and_admit_anchors) --
        surprise, not distance or energy proximity, is the novelty gate.
        admit_range is kept only for the legacy/bootstrap fallback path (see
        train.py's manage_anchor_buffer bootstrap branch); pass None to admit
        every candidate handed in, which is the normal case now.

        Greedily admit survivors -- processed best-energy-first -- against a
        reference set seeded with the current anchors and grown as
        candidates are accepted. A survivor within dup_cutoff of the nearest
        same-condition reference point only displaces it if strictly lower
        energy; otherwise it's dropped as a near-duplicate. This is a cheap
        literal-duplicate catch, not a novelty judgment -- it only ever runs
        on the rare confirmed set handed in here, restricted to reference
        entries sharing the candidate's condition_id so two anchors that
        happen to sit close together in latent space but arose under
        different conditions are never compared against each other -- they're
        distinct targets, not duplicates. This handles both intra-batch
        duplicates and duplicates against existing anchors in one pass, and
        never evicts an existing anchor except by something strictly
        lower-energy appearing within dup_cutoff of it under the same
        condition -- so an isolated, merely-adequate anchor is never
        displaced just for being rare.

        original_surprise, if given, is a per-candidate tensor aligned with
        reward/energy, threaded through to whichever candidates end up
        admitted or replacing an existing slot -- see AnchorBuffer's
        docstring for why it's stored frozen rather than updated later.

        Returns the number of anchors admitted or replaced.
        """
        reward = torch.as_tensor(reward, dtype=torch.float32).detach().cpu().flatten()
        energy = torch.as_tensor(energy, dtype=torch.float32).detach().cpu().flatten()
        if reward.numel() == 0:
            return 0
        if original_surprise is not None:
            original_surprise = torch.as_tensor(
                original_surprise, dtype=torch.float32).detach().cpu().flatten()

        cand_condition_id = candidate_batch.condition_id.detach().cpu().flatten()

        if admit_range is not None:
            cond_best = _per_condition_min(self.condition_id, self.energy, cand_condition_id)
            keep = torch.nonzero(energy < cond_best + admit_range, as_tuple=False).flatten()
        else:
            keep = torch.arange(energy.numel())
        if keep.numel() == 0:
            return 0

        candidate_batch = candidate_batch.subsample_new_batch(keep)
        reward = reward[keep]
        energy = energy[keep]
        cand_condition_id = cand_condition_id[keep]
        if original_surprise is not None:
            original_surprise = original_surprise[keep]

        if self.x_fn is None:
            cand_x = candidate_batch.latent_params()
        elif callable(self.x_fn):
            cand_x = self.x_fn(candidate_batch)
        else:
            cand_x = candidate_batch[self.x_fn]
        cand_x = cand_x.detach().to(self.device).contiguous()

        order = torch.argsort(energy)  # ascending -- lowest energy (best) first

        ref_x = self.x.clone()
        ref_reward = self.reward.clone()
        ref_energy = self.energy.clone()
        ref_condition_id = self.condition_id.clone()
        n_existing = ref_x.shape[0]

        replace_map = {}  # existing-anchor slot idx -> winning local candidate idx
        new_slot_owner = []  # local candidate idx currently occupying each newly created slot

        for local_idx in order.tolist():  # todo check speed here
            x_i = cand_x[local_idx:local_idx + 1]
            r_i = reward[local_idx]
            e_i = energy[local_idx]
            cid_i = cand_condition_id[local_idx]

            same_cond = torch.nonzero(ref_condition_id == cid_i, as_tuple=False).flatten()
            if same_cond.numel() > 0:
                d = torch.cdist(x_i, ref_x[same_cond]).flatten()
                nn_val, nn_local = d.min(0)
                nn_val = nn_val.item()
                nn_pos = int(same_cond[nn_local].item())
            else:
                nn_val = float('inf')
                nn_pos = -1

            if nn_val <= dup_cutoff:
                if e_i.item() < ref_energy[nn_pos].item():
                    if nn_pos < n_existing:
                        replace_map[nn_pos] = local_idx
                    else:
                        new_slot_owner[nn_pos - n_existing] = local_idx
                    ref_x[nn_pos] = x_i[0]
                    ref_reward[nn_pos] = r_i
                    ref_energy[nn_pos] = e_i
                # else: candidate is a worse duplicate -- dropped
            else:
                new_slot_owner.append(local_idx)
                ref_x = torch.cat([ref_x, x_i], dim=0)
                ref_reward = torch.cat([ref_reward, r_i[None]], dim=0)
                ref_energy = torch.cat([ref_energy, e_i[None]], dim=0)
                ref_condition_id = torch.cat([ref_condition_id, cid_i[None]], dim=0)

        n_admitted = len(replace_map) + len(new_slot_owner)
        if n_admitted == 0:
            return 0

        if replace_map:
            self.purge_by_index(list(replace_map.keys()))

        add_local = list(replace_map.values()) + new_slot_owner
        add_inds = torch.tensor(add_local, dtype=torch.long)
        self.add(candidate_batch.subsample_new_batch(add_inds),
                 reward=reward[add_inds], energy=energy[add_inds],
                 original_surprise=original_surprise[add_inds] if original_surprise is not None else None)

        return n_admitted

    @torch.no_grad()
    def thin(self, per_condition_min_energy, energy_window: Optional[float] = None,
             max_size: Optional[int] = None):
        """
        Energy-window trim, run per condition_id group: drop anchors whose
        energy - per_condition_min_energy[cid] exceeds energy_window (deliberately
        wide -- an anchor irrelevant at low T may carry real weight at high T,
        so this should stay generous). per_condition_min_energy is supplied by
        the caller (train.py's condition_log_z.best_energy, i.e. Emin(c)) rather
        than recomputed here, so buffer.py stays free of any condition_log_z
        coupling -- pass a [library_size] tensor indexable by condition_id, or a
        dict/Tensor-like keyed the same way; energy_window=None skips this pass.

        This replaces the old distance-based dedup pass: novelty admission is
        now gated by surprise (see AnchorBuffer.admit/screen_and_admit_anchors),
        so a periodic re-clustering correction is no longer needed -- an anchor
        near an existing one will simply fail re-confirmation once the policy
        has generalized locally, and Emin(c) is monotonically non-increasing, so
        this trim is a one-way, always-safe operation: an anchor that falls out
        can never re-qualify.

        Hard-cap backstop: if the buffer is still over max_size afterward,
        each condition's own
        current-best (lowest-energy) anchor is protected first, then the
        remaining pool is trimmed by lowest original_surprise first (NaN --
        i.e. unmeasured, legacy-seeded anchors -- sorts last, so freshly
        confirmed anchors are the ones actually exposed to this backstop). This
        is the one non-adaptive, explicitly irreversible eviction path -- see
        AnchorBuffer's docstring. If this fires regularly, energy_window is too
        wide; that's the diagnostic, not a reason for a smarter eviction rule.
        """
        n = len(self)
        if n == 0:
            return

        if energy_window is not None:
            condition_id = self.condition_id
            cond_min = torch.as_tensor(per_condition_min_energy)[condition_id].to(self.energy.device)
            drop = torch.nonzero(
                self.energy - cond_min > energy_window, as_tuple=False).flatten()
            if drop.numel() > 0:
                self.purge_by_index(drop.numpy())

        if max_size is not None and len(self) > max_size:
            condition_id = self.condition_id
            energy = self.energy
            surprise = self.original_surprise
            protected = {
                int(rows[torch.argmin(energy[rows])].item())
                for cid in torch.unique(condition_id)
                for rows in [torch.nonzero(condition_id == cid, as_tuple=False).flatten()]
            }

            excess = len(self) - max_size
            candidates = [i for i in range(len(self)) if i not in protected]
            if excess > 0 and candidates:
                cand_idx = torch.tensor(candidates, dtype=torch.long)
                order = torch.argsort(torch.nan_to_num(surprise[cand_idx], nan=float('inf')))  # lowest surprise first
                drop_idx = cand_idx[order[:min(excess, cand_idx.numel())]]
                self.purge_by_index(drop_idx.numpy())


'''
import plotly.graph_objects as go
go.Figure(go.Histogram(x=np.log(p), nbinsx=100)).show()
'''
