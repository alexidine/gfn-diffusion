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

    #: The mirror case, and the one padding must NOT paper over: terms the FWD
    #: loss trains for which the backward evaluator has no counterpart at all.
    #: `z_level` and the condition-grouped `emp_z` branch change the loss VALUE,
    #: so a replayed fwd score would be a different objective (worse: the
    #: backward evaluator ASSERTS against emp_z under vg_by_condition, and
    #: reaches an undefined `log_Z_emp` without it). `traj_grads` and
    #: `reward_grads` change only which paths carry gradient, which a no_grad
    #: ray never uses -- they are refused anyway, because a bank carrying them
    #: is a bank whose forward branch is not the one this evaluator computes,
    #: and the cost of finding that out later is a wrong rate.
    #: All four are 0 in the canonical fwd bank; this asserts rather than
    #: assumes (handoff 2026-08-21 section 6A).
    FWD_ONLY_TERMS = ('z_level', 'emp_z', 'reward_grads', 'traj_grads')

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

        Asked BEFORE a calibration arms, so a structurally unscoreable stage
        refuses its period instead of cloning every policy parameter and then
        discovering the same thing n_sub scores later.
        """
        if branch != 'fwd':
            return None
        bank = self._raw_bank(branch)
        live = [k for k in self.FWD_ONLY_TERMS
                if abs(float(getattr(bank, k, 0.0) or 0.0)) > 0]
        if not live:
            return None
        return 'fwd_bank_' + '_'.join(live)

    def bank(self, branch: str):
        """The branch's coefficient bank, padded for the replay evaluator.

        Found the hard way: the first fused-stage score died with
        `'Namespace' object has no attribute 'pf_boost'`. Padding is announced
        once per branch rather than applied silently -- a zero that nobody chose
        is exactly the kind of default that later reads as a measurement.
        """
        refused = self.refusal(branch)
        if refused is not None:
            raise BranchRefused(refused)
        bank = self._raw_bank(branch)
        missing = [k for k in self.REQUIRED_COEFFS if not hasattr(bank, k)]
        if not missing:
            return bank
        cached = self._padded.get(branch)
        if cached is None:
            cached = copy.copy(bank)
            for k in missing:
                setattr(cached, k, 0.0)
            self._padded[branch] = cached
            if self.verbose and branch not in self._announced:
                self._announced.add(branch)
                print(f"larder: {branch} bank padded with {missing} = 0 for the "
                      f"replay evaluator (terms this branch does not train)")
        return cached

    @torch.no_grad()
    def score(self, rec: Harvested, discretizer) -> float:
        """One harvested batch, one number, at the current parameters."""
        dev = self.m.device
        traj = to_device(rec.traj, dev)
        loss, _ = get_gfn_backward_loss(
            self.bank(rec.branch),
            traj[:, -1] if traj.dim() == 3 else traj,
            self.m.gfn_model,
            to_device(rec.log_r, dev),
            discretizer,
            to_device(rec.mol_batch, dev),
            condition=to_device(rec.condition, dev),
            repeats=rec.repeats,
            report_losses=False,
            trajectories=traj,
            condition_log_z=self.m.condition_log_z,
            condition_id=to_device(rec.condition_id, dev),
            tb_z_source=self.m.tb_z_source(rec.branch),
            update_log_z=False,
            step=self.m.step_ind,
            scramble_condition_tiles=int(rec.scramble_tiles or 0),
            mode_level_stream=None,
            sample_weights=to_device(rec.sample_weights, dev))
        return float(loss.detach())
