"""A reporting step on a branch whose TB coefficient is zero must not raise.

`log_z_target` is published from the reporting block of both branch losses, and the flag it
is gated on (`use_persistent_z`) is bound only inside `if loss_coeffs.tb > 0`. Every phase-1
MLE step is a backward call with tb 0 and report_losses on, so an unguarded read raised
UnboundLocalError there -- on the crystal and conformer routes alike. Driven through the real
backward loss on a tiny untrained GFN, the same way tests/losses/test_stored_force.py drives it.
"""
from types import SimpleNamespace

import pytest
import torch

from energy_sampling.gflownet_losses import get_gfn_backward_loss
from energy_sampling.models.gfn import GFN
from energy_sampling.utils import uniform_discretizer, get_gfn_init_state

DIM, B, TRAJ = 5, 8, 4

_COEFFS = dict(beta=10.0, freeze_policy=0.0, freeze_z=1.0, vg_by_condition=0.0, vg_lb=0.0,
               vg_lme=0.0, level_gap=0.0, pf_boost=0.0, db=0.0, subtb=0.0, emp_z=0.0,
               emp_z_persistent=0.0, tbc=0.0, loss_clip=1.0e9, traj_grads=0.0,
               resample_last_k=0, stored_force_mode='implied', stored_force_k=0)


def _disc():
    return lambda bsz: uniform_discretizer(bsz, TRAJ)


def _terminals(gfn):
    torch.manual_seed(1)
    init = get_gfn_init_state(B, DIM, 'cpu')
    states, _, _, _ = gfn.get_traj_fwd(init, _disc(), None, False, None, detach_traj=True)
    return states[:, -1].detach()


@pytest.mark.parametrize('tb,mle', [(0.0, 1.0), (1.0, 0.0)])
def test_backward_reporting_step_runs_with_and_without_tb(tb, mle):
    torch.manual_seed(0)
    gfn = GFN(dim=DIM, s_emb_dim=16, conditions_dim=0, harmonics_dim=8, t_dim=8,
              device='cpu', angular_mask=[True] * DIM)
    coeffs = SimpleNamespace(**_COEFFS, tb=tb, mle=mle)
    log_r = -torch.arange(B, dtype=torch.float32) / B
    out = get_gfn_backward_loss(coeffs, _terminals(gfn), gfn, log_r, _disc(), SimpleNamespace(),
                                repeats=1, report_losses=True, live_stash=None)
    loss_dict = out[-1] if isinstance(out, tuple) else None
    assert loss_dict is not None and 'log_z_target' not in loss_dict
