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
            birth_step: int = 0,
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
        # global training step at which each row was admitted (CPU bookkeeping,
        # like ema_loss) -- lets callers enforce a residence-time ceiling.
        # Callers that don't care leave the default 0.
        self.birth_step = torch.full((n,), int(birth_step), dtype=torch.long)
        # admission-time snapshot of the ema_loss seed. ema_loss evolves via
        # update_losses; birth_loss never does -- the pair gives death-vs-birth
        # residual deltas at eviction (replay TTL-cohort telemetry).
        self.birth_loss = self.ema_loss.clone()

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
            'birth_step': self.birth_step.cpu(),
            'birth_loss': self.birth_loss.cpu(),
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
        obj.birth_step = state.get(
            'birth_step', torch.zeros_like(obj.select_counts)).cpu()
        obj.birth_loss = state.get(
            'birth_loss', torch.full_like(obj.ema_loss, float('nan'))).cpu()
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
                # A supplied `p` is a DESIGN MEASURE, and importance-sampling
                # correctness assumes iid draws from it -- so it must be drawn
                # WITH replacement. Without-replacement also fails outright when
                # p has zeros: the one-sided prioritised draw (B5) zeroes every
                # delta_plus <= 0 row, and once the eligible pool falls below
                # batch_size numpy raises "Fewer non-zero entries in p than
                # size". That killed the kappa=0 arm at step 119 on 2026-08-07.
                replace = True if p is not None else batch_size > n
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
    def add(self, data, traj: Optional[torch.Tensor] = None, init_loss: Optional[torch.Tensor] = None,
            birth_step: int = 0):
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

        # Stage every allocation before committing any of it: train.py's OOM
        # handler catches CUDA OOM mid-train-step and keeps going, so a partial
        # commit here (batch appended, side arrays not) leaves a corrupted
        # buffer that only detonates on a later draw.

        # validate=False: the shared-metadata equality checks are torch.equal
        # on device tensors -- each one is a stream sync, priced at whatever
        # the GPU has queued (the buffer-churn spike mechanism). This admission
        # path runs every train step on homogeneous same-pipeline batches;
        # shared metadata here is uniform by construction (_as_batch enforces
        # max_z_prime, lazy SG caches are stripped above).
        new_resident = self.batch.append_batch(new_batch, validate=False)

        new_x_full = torch.cat([self.x, new_x], dim=0)

        new_y_full = None
        if self.y is not None:
            if new_y is None:
                raise ValueError("Existing dataset has y, but added batch produced y=None.")
            new_y_full = torch.cat([self.y, new_y], dim=0)

        new_traj_full = None
        if self.traj is not None:
            if traj is None:
                raise ValueError("Existing dataset has traj, but added batch produced traj=None.")
            assert traj.shape[0] == new_batch.num_graphs, \
                f"traj has {traj.shape[0]} entries, expected {new_batch.num_graphs} to match added batch size"
            new_traj_full = torch.cat([self.traj, traj.detach().to(self.device)], dim=0)

        k = new_batch.num_graphs
        if init_loss is None:
            new_ema_loss = torch.full((k,), float("nan"), dtype=self.ema_loss.dtype)
        else:
            new_ema_loss = torch.as_tensor(init_loss, dtype=self.ema_loss.dtype).detach().cpu().flatten()
            if new_ema_loss.numel() == 1:
                new_ema_loss = new_ema_loss.expand(k).clone()
            assert new_ema_loss.shape[0] == k, \
                f"init_loss has {new_ema_loss.shape[0]} entries, expected {k} to match added batch size"

        new_ema_loss_full = torch.cat([self.ema_loss, new_ema_loss], dim=0)
        new_select_counts_full = torch.cat([self.select_counts, torch.zeros(k, dtype=torch.long)], dim=0)
        new_birth_step_full = torch.cat(
            [self.birth_step, torch.full((k,), int(birth_step), dtype=torch.long)], dim=0)
        new_birth_loss_full = torch.cat([self.birth_loss, new_ema_loss.clone()], dim=0)

        new_nan = torch.full((k,), float("nan"), dtype=torch.float32)
        new_ema_logw_full = torch.cat([self.ema_logw, new_nan], dim=0)
        new_ema_logw_sq_full = torch.cat([self.ema_logw_sq, new_nan.clone()], dim=0)
        new_ema_log_z_emp_full = torch.cat([self.ema_log_z_emp, new_nan.clone()], dim=0)

        self.batch = new_resident
        self.x = new_x_full
        if new_y_full is not None:
            self.y = new_y_full
        if new_traj_full is not None:
            self.traj = new_traj_full
        self.ema_loss = new_ema_loss_full
        self.select_counts = new_select_counts_full
        self.birth_step = new_birth_step_full
        self.birth_loss = new_birth_loss_full
        self.ema_logw = new_ema_logw_full
        self.ema_logw_sq = new_ema_logw_sq_full
        self.ema_log_z_emp = new_ema_log_z_emp_full

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

        # Stage every allocation before committing any of it: train.py's OOM
        # handler catches CUDA OOM mid-train-step and keeps going, so a partial
        # commit here (batch shrunk, side arrays not) leaves a corrupted buffer
        # that only detonates on a later draw.
        new_resident = self.batch.subsample_new_batch(keep_idx)

        keep_t = torch.as_tensor(keep_idx, device=self.device, dtype=torch.long)
        new_x = self.x[keep_t].contiguous()
        new_y = self.y[keep_t].contiguous() if self.y is not None else None
        new_traj = self.traj[keep_t].contiguous() if self.traj is not None else None

        keep_cpu = torch.as_tensor(keep_idx, dtype=torch.long)
        new_ema_loss = self.ema_loss[keep_cpu]
        new_select_counts = self.select_counts[keep_cpu]
        new_birth_step = self.birth_step[keep_cpu]
        new_birth_loss = self.birth_loss[keep_cpu]
        new_ema_logw = self.ema_logw[keep_cpu]
        new_ema_logw_sq = self.ema_logw_sq[keep_cpu]
        new_ema_log_z_emp = self.ema_log_z_emp[keep_cpu]

        self.batch = new_resident
        self.x = new_x
        if new_y is not None:
            self.y = new_y
        if new_traj is not None:
            self.traj = new_traj
        self.ema_loss = new_ema_loss
        self.select_counts = new_select_counts
        self.birth_step = new_birth_step
        self.birth_loss = new_birth_loss
        self.ema_logw = new_ema_logw
        self.ema_logw_sq = new_ema_logw_sq
        self.ema_log_z_emp = new_ema_log_z_emp

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

            # NB k is min(REMAINING, eligible) -- remaining already nets out the
            # forced drops above. A duplicate of this block used to follow,
            # recomputing k as min(num_to_purge, eligible) (ignoring the forced
            # count) and overwriting sampled_choice, so any forced purge made
            # the call evict forced + num_to_purge rows instead of num_to_purge.
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

    def absorption_stats(self):
        """
        Memorisation sensor (docs/to_do_rebuild.md B7d). Compares each resident
        row's CURRENT residual against the residual it was ADMITTED with:

            ratio    = mean(ema_loss) / mean(birth_loss)     in (0, 1]
            absorbed = 1 - ratio
            lambda_tau = -ln(ratio)

        `ratio` is the whole sensor. 1.0 = the buffer is a pure delay line, its
        composition is the intake distribution, nothing has been fitted. Falling
        toward 0 = resident rows have been corrected *at those exact
        trajectories* while the intake distribution has not moved, which is
        memorisation by definition.

        WHY THIS NEEDS NO CALIBRATION. Under exponential relaxation at rate
        lambda and exponential residence with mean tau, ratio ~ exp(-lambda*tau),
        so the B7a boundary lambda*tau = 1 -- rows corrected faster than they
        are evicted -- lands at ratio = 1/e = 0.368. That is a DERIVED setpoint,
        not a measured band, so it transfers across problem, T and buffer size.

        WHY THERE IS NO SURVIVORSHIP BIAS. birth_loss is only available for rows
        still resident, so it is the intake distribution *of survivors*. Under
        the uniform-random hazard (B7b) survivors are an unbiased sample of
        admits, so that equals the intake distribution. This property is a
        direct consequence of making the purge uniform and would NOT hold under
        the old floor/stalled eviction.

        Undrawn rows have ema_loss == birth_loss exactly (update_losses only
        touches drawn rows) and so contribute ratio 1. That is correct: a row
        nothing has trained on cannot have been memorised, and a buffer churning
        fast enough that most rows are never drawn genuinely is not memorising.
        """
        e = self.ema_loss.double()
        b = self.birth_loss.double()
        m = torch.isfinite(e) & torch.isfinite(b) & (b > 0)
        n = int(m.sum())
        if n < 8:
            return {}
        em, bm = float(e[m].mean()), float(b[m].mean())
        if bm <= 0:
            return {}
        ratio = max(em / bm, 1e-9)
        return {
            'replay/ema_loss_mean': em,
            'replay/birth_loss_mean': bm,
            'replay/resid_vs_intake': ratio,          # the servo's sensor
            'replay/absorbed_frac': 1.0 - ratio,
            'replay/lambda_tau': -math.log(ratio),    # 1.0 is the B7a boundary
            'replay/absorption_n': float(n),
        }

    def prioritised_weights(
            self,
            log_z: float,
            kappa: float = 1.0,
            nan_quantile: float = 0.90,
            floor_frac: float = 0.25,
    ):
        """
        Prioritised-IS sampling distribution over rows (docs/to_do_rebuild.md
        B5/B5b). Returns (p, w_of_row) where p is the draw distribution and
        w_of_row is the per-row importance weight that UNDOES it, so that

            E_p[w * f]  ==  E_uniform[f]

        i.e. the estimator is unbiased for the uniform-buffer average at every
        kappa. Only the VARIANCE changes with kappa -- that is the whole point
        of the kappa ladder, and it is why any difference a ladder measures is
        estimator variance and nothing else.

        THE RESIDUAL IS RECONSTRUCTED, NOT STORED. delta = log_Z - log_w, and
        ema_logw already carries a per-row EMA of log_w, so

            delta_i = log_z - ema_logw[i]

        is available signed with no new per-row field. `ema_loss` cannot serve
        here: it stores |resid|, and the sign is exactly what the one-sided
        priority needs.

        ONE-SIDED BY DESIGN: delta_plus = max(delta, 0). A row the policy has
        moved off has a fallen log_pf and therefore a strongly NEGATIVE delta,
        so it takes priority ~0 automatically -- which is both the intended
        replay/backward split (B2) and, incidentally, most of what the drift
        term was introduced to do (B8).

        Unvisited rows (ema_logw NaN) get the nan_quantile of the observed
        delta_plus, matching _loss_weights' policy: moderately hard, visited
        promptly, never starved.

        floor_frac is the RELATIVE floor on delta_plus, as a fraction of its
        median among eligible rows. It bounds the weight range, and it is the
        knob that decides whether this estimator is usable at all.

        MEASURED 2026-08-07 against r2_wiring's live buffer (kappa=1, 1000-row
        draws) -- the shipped 0.01 was far too permissive:

            floor_frac   ESS/n_drawn   max(w)/mean(w)
                  0.01         0.11              73
                  0.15         0.50             5.3
                  0.25         0.63             3.3
                  0.50         0.80             1.9

        At 0.01 the live run reported is_ess_frac 0.02-0.06: a 1000-row batch
        was doing the work of ~20-60 rows, because a row just barely above zero
        residual draws with p ~ 0 and therefore carries w ~ 1/p. 0.25 is the
        knee -- it restores the ESS the synthetic test predicted (0.65) while
        keeping most of the prioritisation. Watch replay/is_ess_frac; if it
        falls below ~0.3 this is the first knob to move.

        floor_frac keeps a small uniform component so that a row at
        delta_plus == 0 is not permanently unreachable (its delta can go
        positive again as the policy moves) and so w stays bounded.
        """
        n = len(self)
        if n == 0:
            return np.ones(0, dtype=np.float64), np.ones(0, dtype=np.float64)

        logw = self.ema_logw.clone().float()
        valid = ~torch.isnan(logw)
        if not bool(valid.any()):
            p = np.ones(n, dtype=np.float64) / n
            return p, np.ones(n, dtype=np.float64)

        delta = float(log_z) - logw
        dplus = torch.clamp(delta, min=0.0)
        dplus[~valid] = torch.quantile(dplus[valid], nan_quantile)

        # ELIGIBILITY: delta_plus == 0 rows are excluded from the draw outright
        # (p = 0), not floored into it. They contribute zero force by
        # construction -- delta_plus IS the priority -- so the estimator's
        # target is the uniform mean over the POSITIVE half, which is exactly
        # what the replay branch is for (B2). Including them via a uniform floor
        # is what made max(w) blow up to 1/floor_frac: a row drawn at
        # probability ~0 carries weight ~inf and single-handedly owns a
        # self-normalised batch.
        elig = dplus > 0
        n_elig = int(elig.sum())
        if n_elig == 0:
            p = np.ones(n, dtype=np.float64) / n
            return p, np.ones(n, dtype=np.float64)

        # RELATIVE FLOOR on the surviving positive values, so the weight's
        # dynamic range is bounded by (median/floor)^kappa rather than by the
        # smallest residual that happens to be in the buffer. w ~ 1/delta_plus^
        # kappa is unbounded as delta_plus -> 0+, and a near-zero-but-positive
        # row is exactly as uninformative as a zero one.
        med = float(torch.median(dplus[elig]))
        floor_abs = max(floor_frac * med, 1e-12)
        d = torch.where(elig, torch.clamp(dplus, min=floor_abs),
                        torch.zeros_like(dplus))

        raw = torch.pow(d.double(), float(kappa))
        raw[~elig] = 0.0
        s = float(raw.sum())
        if not np.isfinite(s) or s <= 0.0:
            p = np.ones(n, dtype=np.float64) / n
            return p, np.ones(n, dtype=np.float64)
        p = (raw / s).cpu().numpy()

        # Importance weight per ROW: uniform target over the ELIGIBLE rows,
        # divided by the draw density. Self-normalisation over the drawn batch
        # happens at the call site, so only the ratio matters. Ineligible rows
        # get w = 0; they are never drawn, so it is never read.
        w = np.zeros(n, dtype=np.float64)
        pe = p[elig.cpu().numpy()]
        w[elig.cpu().numpy()] = (1.0 / n_elig) / pe
        return p, w


class ConformerBuffer(CrystalBuffer):
    """CrystalBuffer over conformer graphs (``MolData`` + internal-coordinate tree).

    Everything the buffer actually does -- row-wise draws, EMA loss/logw bookkeeping,
    admission, purge, TTL, persistence -- is graph-agnostic. Only three hooks reach into
    the crystal data model, and they are the three overridden here:

      ``_as_batch``        touches ``max_z_prime`` / ``aunit_*`` and calls
                           ``box_analysis()``. A conformer has no asymmetric unit and no
                           cell; ``max_z_prime`` is not merely unused, it raises
                           (``MXtalBase.__getattr__`` defers to the PyG store).
      ``_orient_stored_batch``  std-orients molecules so per-draw principal-axis work is
                           paid once. Pointless here: the state is internal coordinates,
                           and geometry is rebuilt in the tree's own canonical frame, so
                           the stored orientation is never read.
      ``_compute_xy``      derives the GFN state via ``latent_params()`` (cell params).
                           The conformer state is the stored ``torsion_state``.

    Subclassing rather than generalising CrystalBuffer keeps every crystal run bit-identical
    -- the same call made for ``ConformerModeller(Modeller)``. Rolling the split back into
    the base class is a separate decision, once there is a second non-crystal consumer to
    tell which parts are genuinely shared.
    """

    def _as_batch(self, data):
        """Collate if needed; no cell/Z' normalisation to do."""
        from energies.conformer_data import require_conformer_fields

        if isinstance(data, list):
            batch = collate_data_list(data)
        else:
            batch = data
        return require_conformer_fields(batch)

    @staticmethod
    def _orient_stored_batch(batch):
        """No-op: the stored orientation of a conformer graph is never read."""
        return batch

    def _compute_xy(self, batch):
        """x = the torsion state; y as configured.

        A conditions-only batch carries no ``torsion_state`` (a condition is a molecule,
        not a sample), and ``mol_dataset`` is exactly that -- so x falls back to the
        reference conformer, state 0, rather than raising. That is the honest value: the
        reference conformer IS the zero of this parameterisation. A prior/replay buffer
        without a state, by contrast, would be a real prep bug -- but it is
        ``prebuilt_sample_to_reward`` that catches it, on the energy it cannot fake.
        """
        from energies.conformer_data import batch_states, state_dim

        if callable(self.x_fn):
            x = self.x_fn(batch)
        elif 'torsion_state' in batch._store:
            x = batch_states(batch)
        else:
            x = torch.zeros((batch.num_graphs, state_dim(batch)))

        x = x.detach().to(self.device).contiguous()

        if self.y_fn is None:
            y = None
        elif callable(self.y_fn):
            y = self.y_fn(batch).detach().to(self.device).contiguous()
        else:
            y = batch[self.y_fn].detach().to(self.device).contiguous()

        return x, y


def _upper_tail(quantile: float) -> float:
    """conditional_worst_quantile -> the torch.quantile position for a
    'larger is worse' metric. The config value is the FRACTION OF CONDITIONS
    ALLOWED BEYOND THE BAR (0.5 = median, 0.05 = 95% must clear), so every
    worst-case reducer here converts it the same way and callers pass the raw
    config value. Clamped, since a quantile outside [0, 1] is a hard error in
    torch and a silent nonsense bar here."""
    return min(max(1.0 - float(quantile), 0.0), 1.0)


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
    the estimate exactly as much as a 500-sample step. Instead, `half_life_visits`
    decays a per-condition `effective_count` (a discounted running sample
    size, keyed off that condition's own visit count -- see update()'s
    docstring for why), and each step's actual mixing weight is its share of
    (decayed old effective_count + this step's evidence) -- see update()'s
    docstring.
    """

    def __init__(self, library_size: int, min_visits: int = 20, half_life_visits: float = 7.0,
                 trim_frac: float = 0.1, max_batch_weight: float = 200.0,
                 discovery_half_life_steps: float = 200.0, clip_beta: float = 10.0):
        self.library_size = library_size
        self.min_visits = min_visits
        self.half_life_visits = half_life_visits
        self.trim_frac = trim_frac
        self.max_batch_weight = max_batch_weight
        self.discovery_half_life_steps = discovery_half_life_steps
        # FIXED reference Huber beta for the z_grad stream. Taken once from the
        # base fwd_loss_coeffs rather than the live stage's beta on purpose: a
        # per-stage beta would silently rescale the ruler at every transition,
        # so the same reading would mean different things in different stages.
        self.clip_beta = float(clip_beta)
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
        # per-condition EMA of the CLIPPED signed residual mean(clamp(logw -
        # log_Z_learned, +-clip_beta)) -- the per-condition dL/dZ ruler, the
        # persistent analog of quick_tb_stats' pooled 'tb_resid_clipped'. Clip,
        # group-mean, THEN abs at read time (worst_z_grad), so within-condition
        # spread averages out and this is purely the LEVEL error Z training can
        # fix. Bounded by clip_beta, so one degenerate off-policy log_pf cannot
        # inflate it. See update_z_residual/rms_z_grad.
        self.z_grad_ema = torch.full((library_size,), float("nan"), dtype=torch.float32)
        # UNCLIPPED signed batch-mean residual EMA, sharing z_grad's
        # evidence/decay state. Kept separate from z_grad_ema because the exact
        # decomposition E[r^2] = mean(r)^2 + Var(log w) behind tb_err/
        # lookup_fit_error needs the true (unwinsorized) level term -- see
        # update_z_residual/rms_z_bias/cond_tb_err
        self.z_bias_ema = torch.full((library_size,), float("nan"), dtype=torch.float32)
        self.z_resid_effective_count = torch.zeros((library_size,), dtype=torch.float32)
        self.z_resid_last_step = torch.full((library_size,), -1, dtype=torch.long)
        # monotonic (NEVER decayed) count of update_z_residual calls per
        # condition -- the trust mask for rms_tb_err/rms_z_grad/rms_z_bias,
        # same role and reasoning as fwd_level_visits/bwd_level_visits below.
        self.z_resid_visits = torch.zeros((library_size,), dtype=torch.long)
        # per-mode level EMAs (the z_match delta gate): per-condition mean log w
        # kept as two SEPARATE streams -- 'bwd' (backward rollouts from the local
        # buffer: the buffer-implied level J_B) and 'fwd' (on-policy forward
        # rollouts: the on-policy level J_F). ema_logw above blends whatever
        # feeds it (do_update gating); the level-matching gap
        # delta(c) = J_B(c) - J_F(c) needs the two sides unblended, so these
        # are fed from their own loss paths regardless of do_update (see
        # update_and_lookup_condition_log_z's mode_level_stream) and are never
        # fed back into training -- pure measurement. delta_stats() reduces them.
        self.fwd_level_ema = torch.full((library_size,), float("nan"), dtype=torch.float32)
        self.bwd_level_ema = torch.full((library_size,), float("nan"), dtype=torch.float32)
        self.fwd_level_effective_count = torch.zeros((library_size,), dtype=torch.float32)
        self.bwd_level_effective_count = torch.zeros((library_size,), dtype=torch.float32)
        self.fwd_level_last_step = torch.full((library_size,), -1, dtype=torch.long)
        self.bwd_level_last_step = torch.full((library_size,), -1, dtype=torch.long)
        # monotonic per-stream visit counts (NEVER decayed): how many
        # update_mode_level calls a condition has ever received on each stream.
        # The delta gate's TRUST mask reads these, NOT the decayed
        # *_level_effective_count above -- kept as a separate signal from the
        # EMA's own smoothing weight on purpose (see z_resid_visits below for
        # why "have I visited this enough to trust it" and "how should new
        # evidence be weighted" are different questions even though effective_count
        # decay is now keyed to own-visits too and no longer collapses on a large,
        # sparsely-revisited library -- cw02, library 10000, is what motivated
        # splitting them). Freshness (step - *_level_last_step) is still
        # recorded above for an optional staleness filter but is deliberately
        # not gated on.
        self.fwd_level_visits = torch.zeros((library_size,), dtype=torch.long)
        self.bwd_level_visits = torch.zeros((library_size,), dtype=torch.long)
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
               half_life_visits: Optional[float] = None):
        """
        Effective-sample-size-weighted EMA update. `step` is the caller's
        current global training step (self.step_ind); it is recorded into
        `last_update_step` for freshness telemetry only. It drives the decay
        of a per-condition `effective_count`:

            decay             = 0.5 ** (1 / half_life_visits)
            decayed_eff_count = old_eff_count * decay
            new_eff_count     = decayed_eff_count + evidence_count
            w_new             = evidence_count / new_eff_count

        Decay is applied exactly once per update() call for a condition --
        keyed to that condition's own visit count, not elapsed training
        steps. This used to be step-keyed (a fixed decay per elapsed
        training step since the condition's own last visit), on the theory
        that "how stale is old evidence" should track how much the policy
        has moved rather than how often any given condition gets sampled.
        That reasoning holds for a small/dense condition library where
        revisits are frequent relative to the half-life, but breaks down at
        the scale this tracker actually runs at: with thousands of
        conditions, a condition's revisit interval is routinely hundreds of
        steps, so a 7-step half-life decayed old evidence to ~0 on every
        single revisit -- not smoothing, just silently replacing the
        estimate with the latest batch's raw value every time, and starving
        every downstream min-visits gate reading effective_count in the
        process (the cw02 / library-10000 failure that motivated the
        separate monotonic *_visits trust counters elsewhere in this class).
        Keying decay to own-visit count instead makes half_life_visits mean
        the same thing (a memory window of a few visits) regardless of how
        sparsely or densely a condition happens to get revisited or how
        large the library is -- at the cost of no longer discounting
        evidence for wall-clock policy drift between visits; last_update_step
        is still recorded for anyone who wants that as a separate staleness
        filter, but nothing here gates on it.

        w_new is this step's actual mixing weight into the running
        estimate -- it's 1.0 on a condition's first-ever visit (old_eff_count
        == 0, so the running estimate is just set to this step's value) and
        shrinks as old_eff_count grows relative to evidence_count, so a step
        with many samples for a condition properly outweighs a step with
        few, instead of both getting the same fixed decay. half_life_visits
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
        half_life_visits = self.half_life_visits if half_life_visits is None else half_life_visits
        decay_per_visit = 0.5 ** (1.0 / half_life_visits)
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

        nan_mask = torch.isnan(old_mean)

        # one decay step per own visit, regardless of elapsed training steps
        # since the last one -- see docstring. old_eff_count is 0 on a
        # condition's first-ever visit, so decayed_eff_count is trivially 0
        # there too (nan_mask handles that path regardless).
        decayed_eff_count = old_eff_count * decay_per_visit

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
                          half_life_visits: Optional[float] = None):
        """
        Per-condition EMAs of the TB residual logw - log_Z_learned: how far
        the network's own Z prediction currently is for this condition,
        measured directly against this call's fresh per-sample logw -- not
        against ema_logw, which is itself smoothed and therefore a step
        removed from "right now". These are what rms_z_grad()/worst_z_grad()
        reduce over the library to give the controller a per-condition-aware
        miscalibration signal that a single global batch-mean
        (quick_tb_stats' jensen_z_err) cannot: a mean lets a majority of
        well-calibrated conditions dilute away a badly-miscalibrated
        minority, which is exactly the failure mode this whole tracker exists
        to guard against elsewhere (see update()'s trim_frac docstring) -- no
        reason the detection signal should be vulnerable to the same thing
        the estimator itself was hardened against.

        Same own-visit decay and evidence-capping as update() (see its
        docstring), for the same reasons -- kept as separate state
        (z_resid_*) so this monitoring signal's own decay dynamics can't
        interact with the primary logw/log_z_emp EMA math. z_resid_visits is
        the monotonic (never-decayed) counterpart that rms_tb_err/rms_z_grad/
        rms_z_bias trust-gate on, same split as effective_count vs. *_visits
        elsewhere in this class.
        """
        half_life_visits = self.half_life_visits if half_life_visits is None else half_life_visits
        decay_per_visit = 0.5 ** (1.0 / half_life_visits)

        condition_id = torch.as_tensor(condition_id, dtype=torch.long).detach().cpu().flatten()
        logw = torch.as_tensor(logw, dtype=torch.float32).detach().cpu().flatten()
        log_Z_learned = torch.as_tensor(log_Z_learned, dtype=torch.float32).detach().cpu().flatten()

        if condition_id.numel() == 0:
            return

        # two LEVEL views of the same residual stream, EMA'd side by side with
        # shared evidence/decay state. Both take the group mean BEFORE any abs,
        # so within-condition spread averages out of each and neither is floored
        # at the condition's MAD of log w the way an abs-then-mean (E|r|) stream
        # is -- spread is carried separately by ema_logw_sq, and mixing the two
        # into one number is what made the old MAE stream impossible to act on.
        #   z_grad_ema  CLIPPED (+-clip_beta): the dL/dZ ruler. Bounded, so a fat
        #               tail can't inflate it and a lagging Z shows as sign.
        #   z_bias_ema  UNCLIPPED: the true level term, so that
        #               z_bias^2 + Var(log w) is EXACTLY the mean squared
        #               residual (cond_tb_err / lookup_fit_error). Winsorizing
        #               here would break that identity.
        # tb_err >> |z_bias| means the error is spread (only policy work shrinks
        # it); |z_bias| large means Z genuinely lags (Z training shrinks it).
        resid_signed = logw - log_Z_learned
        resid_clipped = resid_signed.clamp(-self.clip_beta, self.clip_beta)

        unique_ids, inverse = torch.unique(condition_id, return_inverse=True)
        k = unique_ids.shape[0]
        counts_this_step = torch.zeros(k, dtype=torch.float32).scatter_add_(
            0, inverse, torch.ones_like(resid_signed))
        mean_resid = torch.zeros(k, dtype=torch.float32).scatter_add_(
            0, inverse, resid_clipped) / counts_this_step
        mean_bias = torch.zeros(k, dtype=torch.float32).scatter_add_(
            0, inverse, resid_signed) / counts_this_step

        old_mean = self.z_grad_ema[unique_ids]
        old_bias = self.z_bias_ema[unique_ids]
        old_eff_count = self.z_resid_effective_count[unique_ids]
        nan_mask = torch.isnan(old_mean)
        bias_nan_mask = torch.isnan(old_bias)

        decayed_eff_count = old_eff_count * decay_per_visit
        evidence_count = torch.clamp(counts_this_step, max=self.max_batch_weight)
        new_eff_count = decayed_eff_count + evidence_count
        w_new = evidence_count / new_eff_count

        new_mean = torch.where(nan_mask, mean_resid, (1.0 - w_new) * old_mean + w_new * mean_resid)
        new_bias = torch.where(bias_nan_mask, mean_bias, (1.0 - w_new) * old_bias + w_new * mean_bias)

        self.z_grad_ema[unique_ids] = new_mean
        self.z_bias_ema[unique_ids] = new_bias
        self.z_resid_effective_count[unique_ids] = new_eff_count
        self.z_resid_last_step[unique_ids] = int(step)
        self.z_resid_visits[unique_ids] += 1

    def _group_trimmed_mean(self, inverse, values, k, trim_frac: Optional[float] = None):
        """
        Per-group trimmed mean over k groups indexed by `inverse` (same
        scheme as update(): drop the top/bottom trim_frac of each group by
        rank, keep at least one sample). Returns (means, counts).
        """
        trim_frac = self.trim_frac if trim_frac is None else trim_frac
        n = values.shape[0]
        counts = torch.zeros(k, dtype=torch.float32).scatter_add_(
            0, inverse, torch.ones_like(values))
        value_order = torch.argsort(values)
        group_order = torch.argsort(inverse[value_order], stable=True)
        sorted_idx = value_order[group_order]
        sorted_group = inverse[sorted_idx]
        sorted_vals = values[sorted_idx]
        group_start = torch.zeros(k, dtype=torch.long)
        group_start[1:] = torch.cumsum(counts, dim=0)[:-1].long()
        rank_in_group = torch.arange(n) - group_start[sorted_group]
        counts_long = counts.long()
        trim_k = torch.clamp((trim_frac * counts).floor().long(),
                             max=(counts_long - 1) // 2)
        kept_counts = counts - 2 * trim_k.float()
        keep_mask = (rank_in_group >= trim_k[sorted_group]) & \
                    (rank_in_group < counts_long[sorted_group] - trim_k[sorted_group])
        means = torch.zeros(k, dtype=torch.float32).scatter_add_(
            0, sorted_group,
            torch.where(keep_mask, sorted_vals, torch.zeros_like(sorted_vals))) / kept_counts
        return means, counts

    @torch.no_grad()
    def update_mode_level(self, mode: str, condition_id, logw, step: int,
                          half_life_visits: Optional[float] = None):
        """
        Feed one mode's per-condition level stream (fwd_level_ema /
        bwd_level_ema, see __init__): an own-visit-decayed, evidence-capped
        EMA of the per-condition trimmed mean of log w, same decay
        convention as update()/update_z_residual() (decayed once per own
        visit on THIS stream, not by elapsed training steps -- see update()'s
        docstring). Trimming for the same reason as update()'s ema_logw: the
        delta gate reads levels, and one catastrophic-energy outlier in a
        batch mean would otherwise swing a whole condition's level reading
        for a half-life.
        """
        if mode == 'fwd':
            level_ema = self.fwd_level_ema
            eff_count = self.fwd_level_effective_count
            last_step_t = self.fwd_level_last_step
            visits = self.fwd_level_visits
        elif mode == 'bwd':
            level_ema = self.bwd_level_ema
            eff_count = self.bwd_level_effective_count
            last_step_t = self.bwd_level_last_step
            visits = self.bwd_level_visits
        else:
            raise ValueError(f"unknown mode-level stream '{mode}' (expected 'fwd' or 'bwd')")

        half_life_visits = self.half_life_visits if half_life_visits is None else half_life_visits
        decay_per_visit = 0.5 ** (1.0 / half_life_visits)
        condition_id = torch.as_tensor(condition_id, dtype=torch.long).detach().cpu().flatten()
        logw = torch.as_tensor(logw, dtype=torch.float32).detach().cpu().flatten()

        if condition_id.numel() == 0:
            return
        if condition_id.shape[0] != logw.shape[0]:
            raise ValueError(
                f"condition_id and logw must have same length, got "
                f"{condition_id.shape[0]} and {logw.shape[0]}.")

        unique_ids, inverse = torch.unique(condition_id, return_inverse=True)
        mean_logw, counts = self._group_trimmed_mean(inverse, logw, unique_ids.shape[0])

        old_mean = level_ema[unique_ids]
        old_eff_count = eff_count[unique_ids]
        nan_mask = torch.isnan(old_mean)

        decayed_eff_count = old_eff_count * decay_per_visit
        evidence_count = torch.clamp(counts, max=self.max_batch_weight)
        new_eff_count = decayed_eff_count + evidence_count
        w_new = evidence_count / new_eff_count

        level_ema[unique_ids] = torch.where(
            nan_mask, mean_logw, (1.0 - w_new) * old_mean + w_new * mean_logw)
        eff_count[unique_ids] = new_eff_count
        last_step_t[unique_ids] = int(step)
        visits[unique_ids] += 1   # monotonic: one increment per step this condition appears on this stream

    @torch.no_grad()
    def delta_stats(self, quantile: float = 0.0, min_visits: Optional[float] = None,
                    min_trusted_frac: float = 0.5):
        """
        The z_match level-matching gap: per-condition
        delta(c) = bwd_level_ema - fwd_level_ema (buffer-implied minus
        on-policy level; equals minus the mean backward TB residual at
        Z-stationarity), reduced over conditions CHARACTERIZED on BOTH streams
        -- monotonic visit count >= min_visits, NOT the decayed
        effective_count (which is unreachable for a large library and pinned
        'worst' to +inf; see fwd_level_visits in __init__).

        'worst' is the (1 - `quantile`) UPPER-tail quantile of |delta(c)|, so
        `quantile` carries the same meaning it has on tb_err_worst
        (conditional_worst_quantile): the fraction of worst-case conditions
        allowed to sit beyond the bar. Both read the high tail, since for the
        gap as for the residual, larger is worse. quantile=0.0
        recovers the strict max (old behaviour). +inf on the same
        never-pass-on-ignorance convention as rms_logw_std (the exit gate fires
        on SMALL values), held until at least `min_trusted_frac` of the library
        is characterized on both streams, so a fresh/resumed run cannot exit
        z_match before the level EMAs have saturated. 'mean' is the trusted mean
        of |delta(c)|, 'n' the trusted-condition count.
        """
        min_visits = self.min_visits if min_visits is None else min_visits
        mask = (~torch.isnan(self.fwd_level_ema)) & (~torch.isnan(self.bwd_level_ema)) \
               & (self.fwd_level_visits >= min_visits) \
               & (self.bwd_level_visits >= min_visits)
        n = int(mask.sum().item())
        if n == 0 or n < min_trusted_frac * self.library_size:
            return {'worst': float('inf'), 'mean': float('inf'), 'n': n}
        gap = (self.bwd_level_ema[mask] - self.fwd_level_ema[mask]).abs()
        worst = torch.quantile(gap, _upper_tail(quantile)).item()
        return {'worst': worst, 'mean': gap.mean().item(), 'n': n}

    @torch.no_grad()
    def pooled_levels(self, min_effective_count: Optional[float] = None):
        """
        Dashboard companions to delta_stats(): the trusted-masked mean of each
        per-mode level stream, so the two sides of the level-matching gap are
        visible on the SAME monotonic-visits trust mask the delta gate reads.
        The logged bwd/jensen_z / fwd/jensen_z pair are metric_tracker EMAs
        with a much longer horizon -- during an 80-step z_match walkdown they
        lag the true levels by nats (fxr4h4zy: apparent 4-nat gap at exit vs
        0.1 real). NaN (not inf) when a stream has no trusted condition:
        display-only, never gated on.
        """
        min_effective_count = self.min_visits if min_effective_count is None else min_effective_count
        out = {}
        for name, ema, visits in (('fwd', self.fwd_level_ema, self.fwd_level_visits),
                                   ('bwd', self.bwd_level_ema, self.bwd_level_visits)):
            mask = (~torch.isnan(ema)) & (visits >= min_effective_count)
            out[name] = ema[mask].mean().item() if mask.any() else float('nan')
        return out

    @torch.no_grad()
    def lookup_delta(self, condition_id, min_visits: Optional[float] = None):
        """
        Row-aligned SIGNED per-condition level gap delta(c) = J_B(c) - J_F(c),
        the same quantity delta_stats() reduces over the library -- but
        un-abs'd, un-reduced, and per row, for use as a detached coefficient in
        a loss term (see get_gfn_backward_loss's level_gap).

        Returns (delta, mask), same contract as lookup(): delta is 0 (not NaN)
        wherever a stream is uncharacterized, and the caller applies the mask.
        The mask requires BOTH streams past min_visits on the monotonic visit
        count (same trust rule as delta_stats), so a condition seen on only one
        stream carries no gap gradient.
        """
        min_visits = self.min_visits if min_visits is None else min_visits
        condition_id = torch.as_tensor(condition_id, dtype=torch.long).detach().cpu().flatten()
        fwd = self.fwd_level_ema[condition_id]
        bwd = self.bwd_level_ema[condition_id]
        mask = (~torch.isnan(fwd)) & (~torch.isnan(bwd)) \
               & (self.fwd_level_visits[condition_id] >= min_visits) \
               & (self.bwd_level_visits[condition_id] >= min_visits)
        delta = torch.nan_to_num(bwd - fwd, nan=0.0)
        return delta, mask

    @property
    def cond_tb_err(self):
        """Per-condition RMS TB residual, in NATS: sqrt(z_bias_ema^2 +
        Var(log w)). The residual decomposes EXACTLY into a LEVEL term
        (z_bias_ema, the signed mean) and a SPREAD term (within-condition
        Var(log w)), and the mean square is their sum -- so this is the whole
        quality of fit, and subtracting the level part recovers what only
        policy training can fix. The persistent (cross-visit) analog of
        quick_tb_stats' per-batch cond_tb_err. NaN where a condition has
        neither component yet."""
        return (self.z_bias_ema ** 2 + self.logw_var).sqrt()

    @torch.no_grad()
    def rms_tb_err(self, min_visits: Optional[float] = None):
        """
        RMS of cond_tb_err over conditions with enough evidence to trust
        (monotonic z_resid_visits >= min_visits, NOT the decayed
        z_resid_effective_count -- same reasoning as worst_tb_err/delta_stats:
        gating on the decayed count risks starving this on a large, sparsely
        revisited library). RMS rather than mean deliberately: it doesn't let
        a majority of well-fit conditions dilute away a badly-fit minority
        the way an arithmetic mean would (see update_z_residual's
        docstring). Returns 0.0 (not NaN) when no condition currently has
        enough evidence, so this is always safe to compare directly against
        a threshold.
        """
        min_visits = self.min_visits if min_visits is None else min_visits
        err = self.cond_tb_err
        mask = (~torch.isnan(err)) & (self.z_resid_visits >= min_visits)
        if not mask.any():
            return 0.0
        return torch.sqrt((err[mask] ** 2).mean()).item()

    @torch.no_grad()
    def rms_z_grad(self, min_visits: Optional[float] = None):
        """
        RMS over trusted conditions of the CLIPPED signed residual EMA
        (z_grad_ema) -- the per-condition dL/dZ ruler, level-only. Read the
        pair together: rms_tb_err >> rms_z_grad means the error is spread
        (only policy work shrinks it); a large rms_z_grad means Z genuinely
        lags (Z training shrinks it). Same trust mask/state as rms_tb_err;
        returns 0.0 when nothing is trusted (same fire-on-large semantics).
        """
        min_visits = self.min_visits if min_visits is None else min_visits
        mask = (~torch.isnan(self.z_grad_ema)) & (self.z_resid_visits >= min_visits)
        if not mask.any():
            return 0.0
        return torch.sqrt((self.z_grad_ema[mask] ** 2).mean()).item()

    @torch.no_grad()
    def rms_z_bias(self, min_visits: Optional[float] = None):
        """RMS over trusted conditions of the UNCLIPPED signed residual EMA --
        rms_z_grad without the Winsorization, so it still moves once a condition
        is further off than clip_beta. Diagnostic companion to the ruler, never
        a control variable (unbounded: one degenerate off-policy log_pf can
        dominate it, which is exactly what clipping the ruler protects against).
        Same trust mask/state and 0.0-when-cold convention as rms_z_grad."""
        min_visits = self.min_visits if min_visits is None else min_visits
        mask = (~torch.isnan(self.z_bias_ema)) & (self.z_resid_visits >= min_visits)
        if not mask.any():
            return 0.0
        return torch.sqrt((self.z_bias_ema[mask] ** 2).mean()).item()

    @torch.no_grad()
    def worst_tb_err(self, quantile: float = 0.5):
        """
        Worst-case CONDITION quality of fit: the upper-tail quantile of
        cond_tb_err (RMS TB residual in NATS) across all EVER-VISITED
        conditions.

        `quantile` is the fraction of conditions allowed to sit beyond the bar
        (0.5 = median condition, 0.05 = 95% must clear), i.e. the same
        conditional_worst_quantile the config documents and delta_stats takes;
        the (1 - quantile) upper-tail conversion happens HERE. Passing the
        already-converted value would silently read the BEST condition for any
        q < 0.5 -- which is what the predecessor of this method did, harmless
        only because the config happened to sit at the self-inverse 0.5.

        The persistent analog of quick_tb_stats' per-batch 'tb_err_worst', and
        the EMA-safe replacement for the retired fwd/r2_worst:
          * r2 is a RATIO (1 - ss_resid/ss_total); a ratio does not EMA
            (E[A/B] != E[A]/E[B]) and its denominator (within-condition
            Var(log_pf)) collapses for tight conditions, so the per-batch value
            was unbounded-below noise no smoothing could recover. Every term
            here is a per-sample mean already accumulated across visits, so it
            EMAs exactly.
          * It is in nats, so a bar means something physical and transfers
            across problems -- unlike a bounded saturating ratio.

        Masked on the MONOTONIC count (>= 1), NOT effective_count: even
        though effective_count now decays per own-visit rather than per
        elapsed step (so it no longer collapses to 0 between revisits on a
        large library -- the cw02 / delta-gate failure this mask split was
        originally built to route around), the monotonic count remains the
        simpler and more direct "has this condition ever been characterized"
        question, decoupled from the EMA's own smoothing weight. Returns 0.0
        when nothing has
        ever been visited, matching the fire-on-large convention above.
        """
        err = self.cond_tb_err
        mask = (~torch.isnan(err)) & (self.count >= 1)
        if not mask.any():
            return 0.0
        return torch.quantile(err[mask], _upper_tail(quantile)).item()

    @torch.no_grad()
    def worst_z_grad(self, quantile: float = 0.5):
        """Upper-tail quantile of |z_grad_ema| (clipped signed mean residual,
        level-only) -- the per-condition dL/dZ ruler, in nats. Use as the Z-only
        gate in z_match/terminal where the frozen policy can move level but not
        spread (see the weighted_condition_sampling rationale). Same `quantile`
        convention, monotonic-count mask and no-freshness-gate as
        worst_tb_err."""
        mask = (~torch.isnan(self.z_grad_ema)) & (self.count >= 1)
        if not mask.any():
            return 0.0
        return torch.quantile(self.z_grad_ema[mask].abs(), _upper_tail(quantile)).item()

    @torch.no_grad()
    def worst_z_bias(self, quantile: float = 0.5):
        """
        Upper-tail quantile of |z_bias_ema| -- the UNCLIPPED per-condition level
        error, in nats. The tail companion to rms_z_bias, and the only reading
        that sees the population that actually does damage.

        Why this exists separately from worst_z_grad: that one reads the
        CLIPPED stream (winsorized at clip_beta), so it saturates and stops
        distinguishing exactly where it matters. A condition sitting 30 nats
        off and one sitting 10 nats off read identically there. But a
        badly-mis-levelled condition forces the policy to lower log_pf across
        its own samples, and since P_F is normalized the only way to comply is
        to spread mass off-support -- so the policy inflates, its samples get
        worse, log_r falls, and the residual grows. The Huber caps that push's
        MAGNITUDE but not its SIGN, and Adam normalizes a persistent
        sign-consistent gradient to a full step, so the shove repeats every
        step indefinitely without ever tripping a grad-norm wire.

        Read as a PAIR with rms_z_bias: rms is bulk-weighted (a handful of
        extreme conditions barely move it), this is the tail. 'Narrow and
        light-tailed' is the health condition; both readings are needed to
        state it. Same quantile convention and mask as worst_z_grad.
        """
        mask = (~torch.isnan(self.z_bias_ema)) & (self.count >= 1)
        if not mask.any():
            return 0.0
        return torch.quantile(self.z_bias_ema[mask].abs(), _upper_tail(quantile)).item()

    @torch.no_grad()
    def var_z_bias(self, min_visits: Optional[float] = None):
        """
        VARIANCE across trusted conditions of the unclipped per-condition level
        error -- the dispersion the z_var loss term penalizes, reported so the
        loss has a matching diagnostic.

        Variance, not RMS, deliberately: a UNIFORM offset in z_bias is
        harmless (it is a global Z shift, and TB's own gradient owns the
        global level). What hurts is conditions disagreeing about where the
        level is, because a single shared policy cannot satisfy them at once.
        """
        min_visits = self.min_visits if min_visits is None else min_visits
        mask = (~torch.isnan(self.z_bias_ema)) & (self.z_resid_visits >= min_visits)
        if mask.sum() < 2:
            return 0.0
        return self.z_bias_ema[mask].var(unbiased=False).item()

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
        reason as rms_tb_err. Mildly underestimates tails (both moments come
        from trim_frac-trimmed group means), fine for a gate.

        Masks on the MONOTONIC count (>= 1), NOT a min_visits freshness threshold:
        the EMA already discounts stale evidence via its own decay, so on a large
        library a condition with a wide revisit interval still carries a usable
        running moment and must not be gated out for "not enough recent visits" --
        that freshness gate is what emptied the mask on big libraries (cw02). A
        condition seen once has weak but real evidence, ranked accordingly.

        Returns +inf only on a genuine cold start (NOTHING ever visited): callers
        act on SMALL values (a tightness gate that must not pass on ignorance), and
        the old logw_std_rms_ceiling configs still depend on that default. The
        min_visits arg is retained for signature compatibility but no longer gates.
        """
        var = self.ema_logw_sq - self.ema_logw ** 2
        mask = (~torch.isnan(var)) & (self.count >= 1)
        if not mask.any():
            return float('inf')
        return torch.sqrt(var[mask].clamp(min=0.0).mean()).item()

    @torch.no_grad()
    @torch.no_grad()
    def calibration_targets(self, condition_id, step: int,
                            min_visits: Optional[int] = None,
                            freshness_half_life_steps: float = 300.0,
                            se2_floor: float = 0.25):
        """
        (targets, weights) for the interspersed z-calibration step (train.py
        z_calibration_tick). Targets are ema_logw -- the same estimator
        lookup() serves, for the same reason (see lookup()'s docstring on why
        ema_log_z_emp must never feed gradients). Weights are a trust score:

            w_c = valid_c * freshness_c / (SE^2_c + se2_floor)

        - valid: non-nan and count >= min_visits (lookup()'s own gate)
        - freshness = 0.5 ** (steps_since_last_update / freshness_half_life_steps).
          update()'s own EMA decay is keyed to own-visits, deliberately NOT
          wall-clock; calibration is exactly the consumer that DOES want
          wall-clock staleness discounted, because it fires between visits --
          so the discount lives here, on the read, not in the tracker state.
        - SE^2 = (ema_logw_sq - ema_logw^2) / effective_count: the sampling
          noise of the MEAN being regressed to, so a heavily-sampled condition
          outweighs a thinly-sampled one at equal spread. se2_floor caps any
          single condition's weight (the EMA lags even when perfectly sampled).

        Invalid conditions come back with weight 0.0 AND target 0.0 (never
        nan), so callers can use both tensors directly without masking.
        """
        min_visits = self.min_visits if min_visits is None else min_visits
        ids = torch.as_tensor(condition_id, dtype=torch.long).detach().cpu().flatten()
        tgt = self.ema_logw[ids]
        valid = (~torch.isnan(tgt)) & (self.count[ids] >= min_visits)
        age = (float(step) - self.last_update_step[ids].float()).clamp(min=0.0)
        fresh = torch.pow(0.5, age / max(freshness_half_life_steps, 1.0))
        var = (self.ema_logw_sq[ids] - tgt ** 2).clamp(min=0.0)
        se2 = var / self.effective_count[ids].clamp(min=1.0)
        w = fresh / (se2 + se2_floor)
        zeros = torch.zeros_like(tgt)
        return torch.where(valid, tgt, zeros), torch.where(valid, w, zeros)

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

    @property
    def logw_var(self):
        """Per-condition within-condition Var(log w) = E[logw^2] - E[logw]^2.
        Clamped at 0 -- EMA noise can push the difference slightly negative."""
        return (self.ema_logw_sq - self.ema_logw ** 2).clamp(min=0.0)

    @torch.no_grad()
    def lookup_fit_error(self, condition_id):
        """
        Returns (err, mask) with err = z_bias_ema^2 + Var(log w): the per-condition
        MEAN SQUARED TB residual. The residual log_Z_learned - log w decomposes
        exactly into a LEVEL term (z_bias_ema, the signed mean residual) and a
        SPREAD term (within-condition Var(log w)), and the mean square is the sum
        of the two. Both matter and neither alone is "the fit": bias^2 is what a
        Z-only forward branch can fix, Var(log w) is what the policy can fix.

        This is cond_tb_err SQUARED (the property above), kept unrooted because a
        sampling weight only needs a monotone priority and the square keeps the
        two components additive. Uses z_bias_ema, NOT the clipped z_grad_ema: the
        Winsorized level would break the exact bias^2 + Var identity and would
        cap the priority of exactly the worst-fit conditions this is meant to
        steer toward.

        Deliberately NOT normalized into a per-condition r2. That needs the
        within-condition variance of the target in the denominator, which is noisy
        at the ~10-20 samples per condition an eval batch affords and degenerate
        for tight conditions (denominator ~ 0). Callers make it scale-free with a
        quantile clip instead, which is robust to both.

        Masked on the MONOTONIC count (>= 1), never effective_count: same
        decoupling as worst_tb_err -- the monotonic count is the simpler,
        more direct "has this condition ever been characterized" question,
        independent of the EMA's own smoothing weight. A condition is never
        declared dead here. err is 0 where mask is False;
        callers fill those with a neutral value rather than starving them.
        """
        condition_id = torch.as_tensor(condition_id, dtype=torch.long).detach().cpu().flatten()
        bias = self.z_bias_ema[condition_id]
        var = self.logw_var[condition_id]
        err = torch.nan_to_num(bias, nan=0.0) ** 2 + torch.nan_to_num(var, nan=0.0)
        # z_bias_ema is fed by update_z_residual and ema_logw_sq by update(), which
        # need not both have run for a given condition -- one live component is
        # enough to rank on, both missing is not
        valid = (~torch.isnan(bias)) | (~torch.isnan(var))
        mask = (self.count[condition_id] >= 1) & valid
        return err, mask

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
            "half_life_visits": self.half_life_visits,
            "trim_frac": self.trim_frac,
            "max_batch_weight": self.max_batch_weight,
            "clip_beta": self.clip_beta,
            "ema_logw": self.ema_logw.cpu(),
            "ema_logw_sq": self.ema_logw_sq.cpu(),
            "ema_log_z_emp": self.ema_log_z_emp.cpu(),
            "count": self.count.cpu(),
            "effective_count": self.effective_count.cpu(),
            "last_update_step": self.last_update_step.cpu(),
            "best_energy": self.best_energy.cpu(),
            "z_grad_ema": self.z_grad_ema.cpu(),
            "z_bias_ema": self.z_bias_ema.cpu(),
            "z_resid_effective_count": self.z_resid_effective_count.cpu(),
            "z_resid_last_step": self.z_resid_last_step.cpu(),
            "z_resid_visits": self.z_resid_visits.cpu(),
            "fwd_level_ema": self.fwd_level_ema.cpu(),
            "bwd_level_ema": self.bwd_level_ema.cpu(),
            "fwd_level_effective_count": self.fwd_level_effective_count.cpu(),
            "bwd_level_effective_count": self.bwd_level_effective_count.cpu(),
            "fwd_level_last_step": self.fwd_level_last_step.cpu(),
            "bwd_level_last_step": self.bwd_level_last_step.cpu(),
            "fwd_level_visits": self.fwd_level_visits.cpu(),
            "bwd_level_visits": self.bwd_level_visits.cpu(),
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
        # older checkpoints predate half_life_visits (formerly half_life_steps,
        # a step-elapsed clock rather than an own-visit one) -- no exact
        # conversion between the two decay bases, so just fall back to the
        # default rather than reverse-engineer a number from the old semantics.
        obj.half_life_visits = state.get("half_life_visits", 7.0)
        obj.trim_frac = state.get("trim_frac", 0.1)
        obj.max_batch_weight = state.get("max_batch_weight", 200.0)
        # restored from the checkpoint rather than re-read from the live config
        # on purpose: z_grad_ema below was accumulated under THIS beta, and
        # swapping the ruler mid-stream would silently rescale an EMA whose
        # history can't be rescaled with it (same fixed-reference reasoning as
        # __init__'s "never a stage override")
        obj.clip_beta = float(state.get("clip_beta", 10.0))
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
        # last_update_step is freshness telemetry only (decay no longer keys
        # off it) -- default to current_step rather than the -1 "never
        # visited" sentinel purely so a freshly reloaded run doesn't report
        # every already-warmed-up condition as infinitely stale.
        obj.last_update_step = state.get(
            "last_update_step",
            torch.full_like(obj.count, int(current_step))).cpu()
        # older checkpoints predate best_energy -- +inf ("never visited") is the
        # only honest fallback; there's no lifetime stat to reconstruct it from
        obj.best_energy = state.get(
            "best_energy", torch.full_like(obj.count, float("inf"), dtype=torch.float32)).cpu()
        # NaN/0/current_step (never-updated sentinels) are the only honest
        # fallback for a stream a checkpoint doesn't carry, same reasoning as above:
        # update_z_residual's nan masks warm it from the first post-reload batch
        obj.z_grad_ema = state.get(
            "z_grad_ema", torch.full_like(obj.count, float("nan"), dtype=torch.float32)).cpu()
        obj.z_bias_ema = state.get(
            "z_bias_ema", torch.full_like(obj.count, float("nan"), dtype=torch.float32)).cpu()
        obj.z_resid_effective_count = state.get(
            "z_resid_effective_count", torch.zeros_like(obj.count, dtype=torch.float32)).cpu()
        obj.z_resid_last_step = state.get(
            "z_resid_last_step", torch.full_like(obj.count, int(current_step))).cpu()
        # older checkpoints predate this monotonic counter -- zero is the
        # honest fallback (same reasoning as fwd/bwd_level_visits below): the
        # RMS gates re-earn trust as z_resid_visits reaccumulates post-reload.
        obj.z_resid_visits = state.get(
            "z_resid_visits", torch.zeros_like(obj.count)).cpu()
        # older checkpoints predate the per-mode level streams (z_match delta
        # gate) -- NaN/0/current_step never-updated sentinels, same reasoning
        # as the z-residual monitor above
        obj.fwd_level_ema = state.get(
            "fwd_level_ema", torch.full_like(obj.count, float("nan"), dtype=torch.float32)).cpu()
        obj.bwd_level_ema = state.get(
            "bwd_level_ema", torch.full_like(obj.count, float("nan"), dtype=torch.float32)).cpu()
        obj.fwd_level_effective_count = state.get(
            "fwd_level_effective_count", torch.zeros_like(obj.count, dtype=torch.float32)).cpu()
        obj.bwd_level_effective_count = state.get(
            "bwd_level_effective_count", torch.zeros_like(obj.count, dtype=torch.float32)).cpu()
        obj.fwd_level_last_step = state.get(
            "fwd_level_last_step", torch.full_like(obj.count, int(current_step))).cpu()
        obj.bwd_level_last_step = state.get(
            "bwd_level_last_step", torch.full_like(obj.count, int(current_step))).cpu()
        # older checkpoints predate the monotonic level-visit counters -- fall
        # back to zeros: the delta gate then re-earns trust as the level EMAs
        # re-saturate post-reload (the count climbs one per per-stream visit),
        # which is exactly the intended resume behaviour, never a fabricated
        # over-credit from the decayed effective_count
        obj.fwd_level_visits = state.get(
            "fwd_level_visits", torch.zeros_like(obj.count)).cpu()
        obj.bwd_level_visits = state.get(
            "bwd_level_visits", torch.zeros_like(obj.count)).cpu()
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
    def best_per_condition_indices(self):
        """
        For every condition_id present in the buffer, the row index of its
        lowest-energy entry. Deterministic (one row per distinct
        condition_id, ties broken to the lowest row index), unlike
        sample_graphs' priority-weighted random draw. Returns
        (unique_condition_ids, row_idx), both empty if the buffer is empty.
        """
        condition_id = self.condition_id
        energy = self.energy
        n = condition_id.numel()
        if n == 0:
            return condition_id.new_empty(0), torch.empty(0, dtype=torch.long)

        uniq_ids, inverse = torch.unique(condition_id, return_inverse=True)
        k = uniq_ids.numel()
        group_min = torch.full((k,), float('inf'), dtype=torch.float32)
        group_min.scatter_reduce_(0, inverse, energy, reduce='amin', include_self=True)
        is_min = energy == group_min[inverse]
        row_idx = torch.full((k,), -1, dtype=torch.long)
        all_rows = torch.arange(n, dtype=torch.long)
        row_idx.scatter_reduce_(0, inverse[is_min], all_rows[is_min], reduce='amin', include_self=False)
        return uniq_ids, row_idx

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
