"""The Z fill runs INSIDE the fused step, between the forward rollout and the
backward/replay losses, whenever the forward loss carries no gradient.

Left at the host loop's post-step site, the fill pinned log Z to the root of
the PRE-update rollout after an optimizer step on Z had already been taken
against the stale level: one policy step of staleness on the pin and one
discarded Z update on every rollout step, for no reason. With a live forward
loss log_Z is already in fwd's graph and the post-step site keeps the fill.
"""
from types import SimpleNamespace

import pytest
import torch

from energy_sampling.protocol import Stage
from energy_sampling.train import Modeller


class _Stop(Exception):
    """Raised by the bwd stub: everything after the bwd loss is out of scope."""


def _modeller(fwd_frac, calls):
    m = Modeller.__new__(Modeller)
    stage = Stage({'name': 'equilibration', 'train_mode': 'fused',
                   'bwd_sampling_mode': 'prior', 'fwd_rollout_every': 20}, 0)
    m.protocol = SimpleNamespace(stage=stage,
                                 mode_boostable=lambda k: False,
                                 mode_dormant=lambda k: False)
    m.args = SimpleNamespace(
        controller=SimpleNamespace(deactivate_threshold=0.01, refresh_every=10),
        fwd_loss_coeffs=SimpleNamespace(pooled_source='fwd', pooled_vg=0.0))
    m.gfn_model = SimpleNamespace(_live_fwd=None, _live_bwd=None, _live_replay=None)
    m.step_ind = 0
    m._cadence_anchor = 0
    m._cadence_anchor_stage = 'equilibration'
    m._rollout_trigger_fires = lambda *_a, **_k: False
    m.fwd_frac, m.bwd_frac, m.replay_frac = fwd_frac, 1.0 - fwd_frac, 0.0
    m.mode_repeats = lambda mode: 1
    m._z_fill_head_is_fillable = lambda: True

    def fwd_train_step(*_a, **_k):
        calls.append('fwd')
        loss = torch.zeros((), requires_grad=True) * 1.0
        d = {'log_pb': torch.zeros(4), 'log_r': torch.zeros(4), 'log_pf': torch.zeros(4)}
        return loss, None, d

    def z_level_fill(*_a, **_k):
        calls.append('fill')
        m._z_fill_logw = None  # single-use, as the real one

    def bwd_train_step(*_a, **_k):
        calls.append('bwd')
        raise _Stop

    m.fwd_train_step = fwd_train_step
    m.z_level_fill = z_level_fill
    m.bwd_train_step = bwd_train_step
    return m


def test_fill_precedes_the_backward_loss_when_fwd_is_detached():
    """Production shape: fwd_frac 0 under a cadence. The pin must land before
    the residuals that will be trained against it are built."""
    calls = []
    m = _modeller(fwd_frac=0.0, calls=calls)
    with pytest.raises(_Stop):
        Modeller.fused_train_step(m, discretizer=None, report_losses=False)
    assert calls == ['fwd', 'fill', 'bwd'], calls
    assert m._z_fill_logw is None, 'the stash must be consumed inside the step'


def test_fill_is_deferred_when_the_forward_loss_is_live():
    """With fwd carrying weight, log_Z sits in the forward graph; the fill stays
    at the post-step site and the stash survives the step for it."""
    calls = []
    m = _modeller(fwd_frac=0.5, calls=calls)
    with pytest.raises(_Stop):
        Modeller.fused_train_step(m, discretizer=None, report_losses=False)
    assert calls == ['fwd', 'bwd'], calls
    assert m._z_fill_logw is not None, 'the stash must survive for the post-step fill'
