"""
kNN density energy over a FROZEN set of prior draws.

    E(x) = d_live * log r_k(x)

where r_k(x) is the distance from x to the k-th nearest reference point under a
minimum-image metric (period 2 on the [-1, 1] latent box) on the wrapped dims.

WHY THE CONSTANTS ARE MISSING. The Loftsgaarden-Quesenberry estimator is

    log p(x) = log k - log N - log V_d - d * log r_k(x)

and E = -log p keeps only the last term. Everything dropped is constant in x,
and VarGrad reads only the SPREAD of log-weights WITHIN a condition group,
where an additive constant cancels exactly.

*** DO NOT USE THIS AS A NULL-TEST TARGET AT d=12. ***

The argument above is sound and insufficient, and the gap was measured, not
reasoned about. Dropping the constants is fine; the surviving term is not
-log p + const at the shipped dimensionality. Regressing E(x) on a known
closed-form -log p at SG2/Z'=1 (d_live=12, k=32) returns a SLOPE of 0.72 for a
concentrated prior and 1.37 for a broad one -- never 1, and the sign of the
distortion flips with the prior's width. The same code returns slope ~1.0 at
d=2 and d=3, so this is the estimator, not the implementation.

The cause is that the kNN ball is not local here. At d=12 with N=2e4..4e5 the
k-th neighbour sits at roughly 2.2 sigma in a cloud whose own typical radius is
sqrt(12) ~ 3.5 sigma, so r_k spans a sizeable fraction of the distribution and
d*log r_k reads a heavily smoothed density. Smoothing compresses log-density
contrast, which is a MULTIPLICATIVE error on exactly the quantity VarGrad
reads. It does not wash out with data: the rate is N^(-1/6), and a 20x larger
reference moved the slope by 0.07.

Consequence: exp(-E) is approximately p^slope, so a policy sitting exactly at
the prior still sees a systematic gradient -- measured at 0.44-0.56 nats^2 of
VarGrad loss where the design predicts 0. Every number stays finite and
plausible, and the drift reads as a sampler failure.

Any density model proposed for that null test must be run through
energies/density_calibration.py first; a slope near 1 is the acceptance
criterion, and this estimator does not meet it at d=12.

TAILS. Off the reference cloud r_k grows and E rises only as d*log r, i.e.
logarithmically. That is far heavier-tailed than a gaussian fit, so this does
not cage the policy inside the prior's support the way -log of a tight GMM
would. The cost is the other side of the same coin: the target is diffuse far
from the data and pins the policy only weakly out there. The box is held by
generator_energy's bounding_energy, not by this term.

GRADIENTS. log r_k is piecewise-smooth with kinks where the identity of the
k-th neighbour changes. Irrelevant under the shipped `reward_grads: 0`, which
detaches the terminal state before the energy is called; it is a real
consideration only if pathwise reward gradients are ever switched on.

This module deliberately does NOT wrap its own no_grad around energy() -- the
caller (get_loss_reward) owns that decision for the shared energy path.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import torch

#: Latent dims live on [-1, 1], so a wrapped dim has period 2.
LATENT_PERIOD = 2.0

#: Cap on the element count of the [B, chunk, n_wrap] intermediate, which is the
#: only large temporary. The reference chunk size is derived from this per call
#: so batch size and n_wrap cannot conspire into an OOM.
DEFAULT_MAX_PAIR_ELEMS = 64_000_000


def reference_digest(reference: torch.Tensor) -> str:
    """sha256 over the reference coordinates, as float32 on cpu.

    Identity for the reference set, so a run can be attributed to the exact
    draw it was scored against and a silently-swapped file is fatal rather
    than merely different. Snapshot bases have drifted unnoticed here before.
    """
    ref = reference.detach().to('cpu', torch.float32).contiguous()
    return hashlib.sha256(ref.numpy().tobytes()).hexdigest()


class PriorKNN:
    """Frozen kNN density over prior draws, in gauge-fixed latent space.

    The reference set must be built from `latent_params(gauge_fix_free_axes=True)`
    -- the same call generator_energy scores -- or the two live in different
    gauges and every distance is wrong in a way nothing would flag.
    """

    def __init__(self,
                 reference: torch.Tensor,
                 wrap_mask: Sequence[bool],
                 dead_rows: Sequence[int] = (),
                 k: int = 32,
                 period: float = LATENT_PERIOD,
                 min_radius: float = 1e-4,
                 max_pair_elems: int = DEFAULT_MAX_PAIR_ELEMS,
                 provenance: Optional[dict] = None,
                 device=None):
        reference = torch.as_tensor(reference, dtype=torch.float32)
        if reference.ndim != 2:
            raise ValueError(f"reference must be [N, D], got shape {tuple(reference.shape)}")
        n, d = reference.shape

        wrap = torch.as_tensor(list(wrap_mask), dtype=torch.bool)
        if wrap.numel() != d:
            raise ValueError(
                f"wrap_mask has {wrap.numel()} entries but the reference is {d}-dimensional; "
                f"the mask must be the policy's ang_mask, one flag per state dim")

        dead = tuple(sorted(set(int(r) for r in (dead_rows or ()))))
        if any(r < 0 or r >= d for r in dead):
            raise ValueError(f"dead_rows must index [0, {d}), got {dead}")

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k >= n:
            raise ValueError(
                f"k={k} needs a reference of more than k points, got N={n}; "
                f"a reference this thin cannot estimate a density at all")

        self.data_ndim = d
        self.n_reference = n
        self.k = int(k)
        self.period = float(period)
        self.min_radius = float(min_radius)
        self.max_pair_elems = int(max_pair_elems)
        self.dead_rows = dead
        self.wrap_mask = wrap
        self.provenance = dict(provenance or {})
        self.sha256 = reference_digest(reference)

        # Dead dims are pinned constants held out of the diffusion, so they carry no
        # information about density. Excluding them is not merely an optimisation: if
        # the reference was built with even a slightly different pinned value, keeping
        # them would add a constant to every SQUARED distance, which is not a constant
        # in log r and would therefore distort the shape -- the one thing that matters.
        live = [i for i in range(d) if i not in dead]
        if not live:
            raise ValueError("every dim is dead; there is nothing to estimate a density over")
        self.live_idx = torch.tensor(live, dtype=torch.long)
        self.d_live = len(live)

        live_wrap = wrap[self.live_idx]
        self._wrap_idx = self.live_idx[live_wrap]
        self._lin_idx = self.live_idx[~live_wrap]
        self._n_wrap = int(self._wrap_idx.numel())
        self._n_lin = int(self._lin_idx.numel())

        self._reference = reference
        self._ref_wrap = reference[:, self._wrap_idx].contiguous()
        self._ref_lin = reference[:, self._lin_idx].contiguous()
        self._ref_lin_sq = self._ref_lin.pow(2).sum(-1)

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------ io

    @classmethod
    def load(cls, path: str, device=None, k: Optional[int] = None,
             min_radius: Optional[float] = None) -> "PriorKNN":
        """Load a reference file written by build_prior_knn_reference.py.

        `k` and `min_radius` may be overridden here: they are smoothing knobs on a
        fixed reference, so sweeping them costs nothing and does not invalidate
        the draw. The coordinates themselves are checked against the stored digest.
        """
        blob = torch.load(path, map_location='cpu', weights_only=False)
        for key in ('reference', 'wrap_mask', 'k', 'sha256'):
            if key not in blob:
                raise ValueError(
                    f"{path} is missing '{key}'; it was not written by "
                    f"build_prior_knn_reference.py, or predates its current format")

        found = reference_digest(blob['reference'])
        if found != blob['sha256']:
            raise ValueError(
                f"reference digest mismatch in {path}: stored {blob['sha256'][:16]}, "
                f"recomputed {found[:16]}. The coordinates changed after the file was "
                f"written -- do not train against it.")

        obj = cls(reference=blob['reference'],
                  wrap_mask=blob['wrap_mask'],
                  dead_rows=blob.get('dead_rows', ()),
                  k=int(k if k is not None else blob['k']),
                  period=float(blob.get('period', LATENT_PERIOD)),
                  min_radius=float(min_radius if min_radius is not None
                                   else blob.get('min_radius', 1e-4)),
                  provenance=blob.get('provenance', {}),
                  device=device)
        return obj

    def to(self, device) -> "PriorKNN":
        self._reference = self._reference.to(device)
        self._ref_wrap = self._ref_wrap.to(device)
        self._ref_lin = self._ref_lin.to(device)
        self._ref_lin_sq = self._ref_lin_sq.to(device)
        self.live_idx = self.live_idx.to(device)
        self._wrap_idx = self._wrap_idx.to(device)
        self._lin_idx = self._lin_idx.to(device)
        self.wrap_mask = self.wrap_mask.to(device)
        return self

    # ----------------------------------------------------------- validation

    def verify_against_policy(self, ang_mask, dead_rows: Sequence[int] = ()) -> None:
        """Raise unless the policy's geometry matches the reference's.

        The wrap mask and dead rows are properties of (space group, max_z_prime,
        periodic_centroids), and the reference set was built under one particular
        resolution of them. A config that changes the space group, flips
        periodic_centroids, or moves to a different Z' produces a policy whose
        latent means something else -- and the resulting distances would be
        quietly, uniformly wrong. Nothing downstream would notice.
        """
        theirs = torch.as_tensor(ang_mask, dtype=torch.bool).cpu()
        ours = self.wrap_mask.cpu()
        if theirs.numel() != ours.numel() or bool((theirs != ours).any()):
            raise ValueError(
                f"prior_knn reference was built for wrapped dims "
                f"{sorted(torch.nonzero(ours).flatten().tolist())} but the policy wraps "
                f"{sorted(torch.nonzero(theirs).flatten().tolist())}; the reference does "
                f"not describe this problem's latent space")

        their_dead = tuple(sorted(set(int(r) for r in (dead_rows or ()))))
        if their_dead != self.dead_rows:
            raise ValueError(
                f"prior_knn reference was built with dead rows {self.dead_rows} but the "
                f"policy pins {their_dead}; the reference does not describe this "
                f"problem's latent space")

    # --------------------------------------------------------------- energy

    def _kth_sq_dist(self, x: torch.Tensor) -> torch.Tensor:
        """Squared distance to the k-th nearest reference point, [B]."""
        b = x.shape[0]
        x_wrap = x[:, self._wrap_idx] if self._n_wrap else None
        x_lin = x[:, self._lin_idx] if self._n_lin else None
        if x_lin is not None:
            x_lin_sq = x_lin.pow(2).sum(-1, keepdim=True)

        per_row = max(1, self._n_wrap)
        chunk = max(1, int(self.max_pair_elems // max(1, b * per_row)))

        best = None
        for start in range(0, self.n_reference, chunk):
            stop = min(start + chunk, self.n_reference)

            if x_lin is not None:
                # ||a-b||^2 by expansion: avoids ever materialising [B, C, n_lin].
                sq = (x_lin_sq
                      + self._ref_lin_sq[start:stop].unsqueeze(0)
                      - 2.0 * (x_lin @ self._ref_lin[start:stop].T))
            else:
                sq = x.new_zeros((b, stop - start))

            if x_wrap is not None:
                d = x_wrap.unsqueeze(1) - self._ref_wrap[start:stop].unsqueeze(0)
                # minimum image: the same convention as a periodic cell, on a box of
                # width `period` centred at 0.
                d = d - self.period * torch.round(d / self.period)
                sq = sq + d.pow(2).sum(-1)

            # Expansion can go slightly negative on coincident points.
            sq = sq.clamp_min(0.0)

            best = sq if best is None else torch.cat([best, sq], dim=1)
            if best.shape[1] > self.k:
                best = best.topk(self.k, dim=1, largest=False).values

        return best[:, -1] if best.shape[1] == self.k else best.max(dim=1).values

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """E(x) = d_live * log r_k(x), shape [B].

        `x` must be gauge-fixed latents in the SAME layout as the reference
        (i.e. `crystal_batch.latent_params(gauge_fix_free_axes=True)`), not the
        raw policy output.
        """
        if x.ndim != 2 or x.shape[1] != self.data_ndim:
            raise ValueError(
                f"expected latents [B, {self.data_ndim}], got {tuple(x.shape)}")

        sq_k = self._kth_sq_dist(x)
        # A query coinciding with a reference point sends log r to -inf and the
        # reward to +inf. Floor the radius rather than the energy, so the floor is
        # a length in latent units and reads as one.
        sq_k = sq_k.clamp_min(self.min_radius ** 2)
        return 0.5 * float(self.d_live) * torch.log(sq_k)

    # ----------------------------------------------------------------- misc

    def describe(self) -> str:
        wrapped = sorted(torch.nonzero(self.wrap_mask.cpu()).flatten().tolist())
        return (f"PriorKNN(N={self.n_reference}, d={self.data_ndim}, d_live={self.d_live}, "
                f"k={self.k}, wrapped={wrapped}, dead={self.dead_rows or '()'}, "
                f"sha={self.sha256[:12]})")
