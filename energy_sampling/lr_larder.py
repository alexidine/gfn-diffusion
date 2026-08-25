"""
The larder -- a rolling, per-branch ring of replay-scoreable live batches, and
the scorer that re-evaluates them under the current parameters.

WHY IT EXISTS. `ray` measures alpha* by re-scoring ONE FIXED batch at several
points along the step the optimizer just took. That needs data carrying STORED
trajectories: a draw that re-samples a path puts trajectory noise into every
contrast, so the pairing that makes the sensor work is silently broken and the
confidence interval reports high confidence in nothing. Until now the only such
source was the replay buffer, which is why the sensor was coherent ONLY in a
fused stage training replay TB -- and why phase 1 has never been measured. That
was ray's real gap: coverage, not correctness.

Trajectories already leave every live step as `loss_dict['flow_states']`, so
recording them costs a read. This module is that record and its scorer, salvaged
from the frozen-training race that `docs/design/lr_handoff_2026-08-21.md` section
4 retired -- the harvest is the piece of it that carries forward.

TWO PROPERTIES THE DRAW MUST HAVE, neither of which the replay draw it replaces
had:

  * NO RNG. `RayCalibration.measure` used to draw n_sub sub-batches from the
    replay buffer, and those draws consume RNG that nothing restores -- so a
    calibration whose reading the controller then discarded still shifted every
    subsequent training step (findings.md F-039). `take` is deterministic: it
    returns the most recent eligible records, in order. Nothing is sampled, so
    the whole calibration is RNG-neutral and a discarded reading costs compute
    and nothing else.

  * HELD OUT FROM THE STEP BEING RATED. The step's own batches are exactly the
    data its gradient descends, so scoring the ray on them is biased high and
    would license too-large steps. `before_step` excludes them -- and under
    gradient accumulation that is the whole accumulation window, not just the
    last host iteration.

HOST MEMORY, not device. A full larder is `depth` batches PER BRANCH, and on a
fused crystal stage that is three rings of trajectories and crystal batches. Left
on the card they would compete for exactly the VRAM the batch sizer is trying to
fill.
"""

from __future__ import annotations

import copy
from collections import deque, namedtuple

import torch

from gflownet_losses import get_gfn_backward_loss

#: One replay-scoreable training batch, recorded as a live step went past.
#:
#: `traj` is the draw's stored path where there is one (replay), and otherwise
#: the path the step ACTUALLY sampled, which only ever surfaces as
#: `loss_dict['flow_states']` -- which is precisely why harvesting has to happen
#: inside the branch step functions rather than around them.
#:
#: `step`, `scramble_tiles` and `sample_weights` are carried because the score
#: has to be the loss the branch ACTUALLY trained, not a near neighbour of it:
#: `step` is what makes the held-out rule enforceable, `scramble_tiles` changes
#: the value on a conditional prior stage, and `sample_weights` is the replay
#: branch's self-normalised IS reweighting -- scoring its unweighted mean would
#: rate a different objective from the one the step descended.
Harvested = namedtuple(
    'Harvested',
    'branch step condition condition_id log_r mol_batch traj repeats '
    'scramble_tiles sample_weights')


def to_host(v):
    """Detach and park on the host WITHOUT touching the live object.

    `copy.copy` first is not optional: PyG's `Data.cpu()`/`.to()` MUTATE IN
    PLACE (apply rewrites the store and returns self), so a bare
    `mol_batch.cpu()` would drag the batch the live step is still training on
    off the device. A shallow copy gives `.cpu()` its own store to rewrite, and
    tensor-level `.cpu()` is non-mutating, so the live tensors are untouched.
    This is buffer.py's idiom, for the same reason.
    """
    if v is None:
        return None
    if torch.is_tensor(v):
        return v.detach().to('cpu', copy=True)
    if hasattr(v, 'cpu'):                      # a PyG Data/Batch
        return copy.copy(v).cpu()
    return v


def to_device(v, device):
    if v is None:
        return None
    if torch.is_tensor(v):
        return v.to(device)
    if hasattr(v, 'to'):
        return copy.copy(v).to(device)         # same in-place hazard, back
    return v


class Larder:
    """A rolling, per-branch ring of harvested batches.

    ALWAYS ON while a stage declares a sensor that reads it: a calibration falls
    due on a fixed clock and has to find a full ring already there. The cost is
    the tee plus `depth` batches per branch in host memory.
    """

    def __init__(self, depth: int = 32):
        self.depth = max(1, int(depth))
        self.rings: dict[str, deque] = {}
        self.n_seen = 0

    def record(self, rec: Harvested) -> None:
        ring = self.rings.get(rec.branch)
        if ring is None:
            ring = self.rings[rec.branch] = deque(maxlen=self.depth)
        ring.append(rec)
        self.n_seen += 1

    def branches(self) -> tuple:
        return tuple(k for k, v in self.rings.items() if v)

    def count(self, branch: str) -> int:
        return len(self.rings.get(branch, ()))

    def clear(self) -> None:
        """Drop the harvest.

        Called at a stage transition: those batches were drawn under the
        OUTGOING stage's branches and loss mixture, so scoring a ray on them
        rates an objective the run has already left.
        """
        self.rings.clear()

    @staticmethod
    def _bytes(v) -> int:
        """Host bytes held by one harvested field.

        Approximate by design -- it walks tensors and PyG stores and counts
        `element_size * nelement`, which is the storage that actually dominates.
        Approximate is enough: the question this answers is "is the larder tens
        of megabytes or tens of gigabytes", and nothing in between changes a
        decision.
        """
        if v is None:
            return 0
        if torch.is_tensor(v):
            return v.element_size() * v.nelement()
        # a PyG Data/Batch: sum its tensor stores
        store = getattr(v, '_store', None) or getattr(v, '__dict__', {})
        items = store.items() if hasattr(store, 'items') else []
        return sum(t.element_size() * t.nelement()
                   for _k, t in items if torch.is_tensor(t))

    def nbytes(self) -> int:
        """Host bytes the whole larder is holding, right now.

        MEASURED RATHER THAN ASSUMED, and that distinction cost a run. `depth`
        was set to `4 * n_sub` on the reasoning that headroom above n_sub "only
        buys older data" -- true, but it says nothing about the PRICE, and the
        price is per branch. Phase 1 has one branch; a fused crystal stage has
        three, each holding full trajectories and PyG graphs for a batch of
        1000. `nbytes()` stays so the cost is never again a thing
        anyone has to reason about.
        """
        return sum(sum(self._bytes(f) for f in rec)
                   for ring in self.rings.values() for rec in ring)

    def eligible(self, branch: str, before_step: int) -> list:
        """This branch's records that did NOT feed the step being rated."""
        return [r for r in self.rings.get(branch, ()) if r.step < before_step]

    def have(self, branches, n: int, before_step: int) -> bool:
        return bool(branches) and all(
            len(self.eligible(b, before_step)) >= n for b in branches)

    def take(self, branches, n: int, before_step: int):
        """`n` sub-batches, each one record per branch, or None if short.

        DETERMINISTIC, newest last: the most recent `n` eligible records of each
        branch, index-aligned across branches. Nothing is sampled, so no RNG is
        consumed (see the module docstring), and no two sub-batches share a
        record -- the replicates are disjoint rather than overlapping draws from
        one buffer, which is what the paired t across them assumes.

        Newest rather than a spread over the ring because the freshest records
        sit closest to the distribution the step was taken on; a record from
        `depth` steps ago describes an older sampler.
        """
        branches = tuple(branches)
        if not self.have(branches, n, before_step):
            return None
        per = {b: self.eligible(b, before_step)[-n:] for b in branches}
        return [{b: per[b][i] for b in branches} for i in range(n)]


class BranchRefused(Exception):
    """This branch cannot be scored on replayed trajectories, and saying so is
    the point: a composite missing an active branch is the optimum for a
    direction nobody took."""


class LarderScorer:
    """Re-score harvested batches under the CURRENT parameters.

    Stored trajectory and stored reward, so `get_gfn_backward_loss` routes to
    `get_traj_replay`: no rollout and no energy call. `update_log_z=False` and
    `mode_level_stream=None` are the gates that keep this out of the run's
    trackers -- the same contract the replay-drawn `_probe_loss` maintained.
    """

    #: Coefficient keys `get_gfn_backward_loss` reads WITHOUT a getattr guard,
    #: so an absent one raises AttributeError instead of meaning "off". Every
    #: branch is replayed through that evaluator (there is no
    #: `get_gfn_forward_loss(trajectories=...)`), and the FWD bank legitimately
    #: has no `mle`/`pf_boost` because the fwd branch does not train them. For a
    #: replayed score that means coefficient ZERO, which is what we fill.
    #: `coeff_matrix` is deliberately NOT here: it is structural rather than a
    #: coefficient and is only read when subtb > 0, so filling it would mask a
    #: real misconfiguration instead of expressing an absent term.
    REQUIRED_COEFFS = ('db', 'mle', 'pf_boost', 'subtb', 'tb', 'traj_grads',
                       'vg_lb', 'vg_lme')

    #: Z-SIDECAR TERMS, ZEROED IN EVERY BANK BEFORE SCORING. Owner decision
    #: 2026-08-22: the Z sidecar is excluded from ALL learning-rate control.
    #:
    #: It is the coherent choice rather than a convenience. `ray` rays POLICY
    #: parameters only (decision D26b) -- the flow head is LR-pinned separately
    #: and held at its post-step value throughout, so it contributes an
    #: identical constant to every evaluation. These terms exist to TRAIN that
    #: head, i.e. they are the loss of the one thing the sensor deliberately
    #: does not measure. Scoring them would put a quantity the ray holds fixed
    #: into the objective the ray is differencing.
    #:
    #: It also removes the only structural refusal there was. `var_conditioning`
    #: ships `emp_z: 1.0` on its forward bank, and the backward evaluator
    #: ASSERTS against emp_z under vg_by_condition (and reaches an undefined
    #: `log_Z_emp` without it) -- so before this the whole stage was
    #: unmeasurable. Zeroed, its VarGrad terms replay normally.
    #:
    #: `reward_grads` and `traj_grads` are NOT here and are not zeroed: they
    #: only decide which paths carry gradient, and the ray runs under no_grad,
    #: so they cannot change a scored value.
    Z_SIDECAR_TERMS = ('emp_z', 'emp_z_persistent', 'z_level')

    def __init__(self, modeller, verbose: bool = True):
        self.m = modeller
        self.verbose = bool(verbose)
        self._padded = {}
        self._announced = set()

    def _raw_bank(self, branch: str):
        a = self.m.args
        return {'fwd': a.fwd_loss_coeffs, 'bwd': a.bwd_loss_coeffs,
                'replay': a.replay_loss_coeffs}[branch]

    def refusal(self, branch: str):
        """Why this branch cannot be replay-scored, or None.

        NOTHING, CURRENTLY, and that is the point. It used to refuse a forward
        bank carrying a Z-sidecar term; those are zeroed now (see
        `Z_SIDECAR_TERMS`), so no stage is structurally unmeasurable. The hook
        stays because it is the one place such a rule would live, and
        `Modeller._probe_refusal` still asks before spending a parameter clone.
        """
        return None

    def bank(self, branch: str):
        """The branch's coefficient bank, adjusted for the replay evaluator.

        TWO ADJUSTMENTS, both announced once per branch rather than applied
        silently -- a zero that nobody chose is exactly the kind of default that
        later reads as a measurement:

          PADDED. `get_gfn_backward_loss` reads `mle`/`pf_boost`/... without a
          getattr guard, and the fwd bank legitimately has neither. Found the
          hard way: the first fused-stage score died with `'Namespace' object
          has no attribute 'pf_boost'`.

          Z SIDECAR ZEROED. Owner decision: the Z sidecar is excluded from all
          LR control. See `Z_SIDECAR_TERMS`.

        The LIVE bank is never mutated -- both are applied to a copy.
        """
        bank = self._raw_bank(branch)
        missing = [k for k in self.REQUIRED_COEFFS if not hasattr(bank, k)]
        zeroed = [k for k in self.Z_SIDECAR_TERMS
                  if abs(float(getattr(bank, k, 0.0) or 0.0)) > 0]
        if not (missing or zeroed):
            return bank
        cached = self._padded.get(branch)
        if cached is None:
            cached = copy.copy(bank)
            for k in missing + zeroed:
                setattr(cached, k, 0.0)
            self._padded[branch] = cached
            if self.verbose and branch not in self._announced:
                self._announced.add(branch)
                bits = []
                if missing:
                    bits.append(f"padded with {missing} = 0 (terms this branch "
                                f"does not train)")
                if zeroed:
                    bits.append(f"Z sidecar {zeroed} zeroed (excluded from LR "
                                f"control: the ray holds the flow head fixed)")
                print(f"larder: {branch} bank " + '; '.join(bits))
        return cached

    @staticmethod
    def _check_condition_grouping(bank, rec):
        """A condition-grouped bank replayed WITHOUT condition_id silently
        computes a different loss. Refuse instead.

        `get_gfn_backward_loss` gates the condition-grouped VarGrad on
        `vg_by_condition and condition_id is not None and (vg_lb > 0 or
        vg_lme > 0)`. Drop the ids and that `is not None` fails, so control falls
        through to the LEGACY repeats-grouped branch -- which for same-terminal
        tiles is TBC in disguise. A different objective, computed with no error
        and no warning, and the ray would then be rating a loss the stage does
        not train.

        The larder stores the WHOLE batch, so the ids do come back unchanged and
        the groups are identical to the live step's by construction -- verified.
        This guard is for the case where that stops being true.
        """
        if not (float(getattr(bank, 'vg_by_condition', 0) or 0) > 0.5):
            return
        if not (float(getattr(bank, 'vg_lb', 0) or 0) > 0
                or float(getattr(bank, 'vg_lme', 0) or 0) > 0):
            return
        if rec.condition_id is None:
            raise BranchRefused(
                f"branch {rec.branch!r} trains condition-grouped VarGrad "
                f"(vg_by_condition) but its harvested record carries no "
                f"condition_id. Replaying it would fall through to the LEGACY "
                f"repeats-grouped VarGrad -- a different objective, silently.")

    @torch.no_grad()
    def score(self, rec: Harvested, discretizer, resample: bool = False) -> float:
        """One harvested batch, one number, at the current parameters.

        `resample=False` -- the default, and the ONLY thing the controller ever
        actuates on -- REPLAYS the stored path: passing `trajectories=` routes
        `get_gfn_backward_loss` to `get_traj_replay`, which treats the path as
        data and re-scores it under the current parameters.

        `resample=True` passes `trajectories=None`, so the same terminal states
        are scored under a path sampled FRESH from the current P_B --
        `get_traj_bwd`. That is what a `bwd` branch whose live draw sets
        `traj=None` (`bwd_sampling_mode: dataset` or `prior`) actually trains
        on, and it is not the same objective. Along the ray the stored path was
        sampled from P_B at theta_before, so as alpha grows it goes
        off-distribution for the P_B being scored: the replayed loss can rise
        where the trained one falls. DIAGNOSTIC ONLY -- see `RayCalibration`.

        Neither costs an energy call: `log_r` is stored either way. The fresh
        pass is backward-sampling passes only.
        """
        dev = self.m.device
        traj = to_device(rec.traj, dev)
        bank = self.bank(rec.branch)
        self._check_condition_grouping(bank, rec)
        loss, _ = get_gfn_backward_loss(
            bank,
            traj[:, -1] if traj.dim() == 3 else traj,
            self.m.gfn_model,
            to_device(rec.log_r, dev),
            discretizer,
            to_device(rec.mol_batch, dev),
            condition=to_device(rec.condition, dev),
            repeats=rec.repeats,
            report_losses=False,
            trajectories=None if resample else traj,
            condition_log_z=self.m.condition_log_z,
            condition_id=to_device(rec.condition_id, dev),
            tb_z_source=self.m.tb_z_source(rec.branch),
            update_log_z=False,
            step=self.m.step_ind,
            scramble_condition_tiles=int(rec.scramble_tiles or 0),
            mode_level_stream=None,
            sample_weights=to_device(rec.sample_weights, dev))
        return float(loss.detach())
