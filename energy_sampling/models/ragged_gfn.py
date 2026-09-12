"""GFN specialisation for a RAGGED multi-molecule conformer batch.

PARALLEL TO THE DENSE ROUTE, not a replacement (owner decision 2026-09-10: crystal stays
dense, molecules get a subclassed/parallel ragged version; merging them is a problem for
another day). See ``docs/design/ragged_multi_molecule_state.md`` for the sequence this
belongs to.

STATE OF THIS FILE: step 2 of that sequence, PARTIAL. What is implemented and tested is the
policy-head split for a ragged layout. The trajectory loop, the backward policy and the
ragged energy are NOT here yet -- constructing this class does not give you a runnable
rollout, and nothing selects it from a config. It is written as a separate file so the
pieces can land and be tested one at a time instead of as one unreviewable change.

WHY THE SPLIT IS THE FIRST PIECE. ``GFN.split_params`` slices the policy output into
CONTIGUOUS BLOCKS of width ``dim`` -- ``[mean(dim), logvar(dim)]`` -- which is the assumption
that makes a per-token model emit "the same numbers in the wrong order" (``set_policy.py``
carries ``_to_blocks`` solely to undo it). A ragged per-token policy emits
``[sum_k, out_per_token]``, where mean and logvar are COLUMNS, so the split is a slice on the
last axis and the reordering step disappears. Everything downstream of the split -- the
variance shaping, the clip, the baseline addition -- is shared with the dense path verbatim,
which is what makes the two provably the same function.
"""
from __future__ import annotations

import torch

from models.conformer_gfn import ConformerGFN


class RaggedConformerGFN(ConformerGFN):
    """``ConformerGFN`` whose policy output is ``[sum_k, out_per_token]``.

    Only the layout differs. The variance shaping below is copied from
    ``GFN.split_params``' non-DPLR branch rather than re-derived, and
    ``tests/models/test_ragged_gfn.py`` asserts the two agree numerically.
    """

    def split_params(self, tensor, log_var_base):
        """``[sum_k, 2] -> (mean[sum_k], logvar[sum_k], None, None)``.

        DPLR is refused rather than reshaped. ``ConformerGFN.predict_next_state`` already
        refuses ``dplr_rank > 0`` with a set policy because the low-rank factor would be
        silently transposed; the same reasoning applies here and the same refusal is
        repeated rather than assumed, because this method can be reached by a caller that
        never went through that one.
        """
        if self.dplr_rank > 0:
            raise NotImplementedError(
                f'RaggedConformerGFN.split_params with dplr_rank {self.dplr_rank}: the '
                f'low-rank factor has no per-token column layout, so it would be silently '
                f'transposed. Set dplr_rank: 0 for a ragged run.')
        if tensor.ndim != 2 or tensor.shape[-1] != 2:
            raise ValueError(
                f'ragged policy output must be [sum_k, 2] (mean, logvar as COLUMNS), got '
                f'{tuple(tensor.shape)}. A dense [B, 2*dim] block layout reaching here means '
                f'the flat policy is installed on a ragged GFN.')
        mean, logvar_i = tensor[:, 0], tensor[:, 1]

        # --- identical to GFN.split_params from here on -------------------------------
        if not self.learned_variance:
            logvar = torch.zeros_like(logvar_i)
        elif self.log_var_range == -1:
            logvar = logvar_i
        else:
            logvar = torch.tanh(logvar_i / self.log_var_range) * self.log_var_range
        logvar = (logvar + log_var_base).clip(min=-self.var_clip, max=self.var_clip)
        return mean, logvar, None, None
