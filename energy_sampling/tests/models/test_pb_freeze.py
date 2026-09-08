"""P_B freeze: the snapshot really fixes P_B's FUNCTION, survives a
checkpoint round trip, and lifts cleanly.

The cheap freeze (requires_grad off on backward_policy) leaves P_B moving
through the shared s_model/t_model trunk; the full freeze evaluates P_B on a
snapshot of all three. These tests perturb the live trunk after freezing and
check P_B's log-prob on fixed paths does not move -- and that it DOES move
without the freeze, so the test can tell the two apart.
"""
import os
import sys

import pytest
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(_here)),
          os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_here))), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from models.gfn import GFN  # noqa: E402
from checkpointing import Checkpointer  # noqa: E402
from utils import uniform_discretizer  # noqa: E402

DIM, T, B = 4, 5, 6


def make_gfn(seed=0):
    torch.manual_seed(seed)
    return GFN(dim=DIM, s_emb_dim=16, harmonics_dim=8, t_dim=8,
               t_hidden_dim=16, s_hidden_dim=16, s_layers=2,
               policy_hidden_dim=16, policy_layers=2,
               flow_hidden_dim=16, flow_layers=2,
               cond_hidden_dim=16, cond_layers=2,
               conditions_dim=0, condition_embedding_dim=8,
               conditions_type='vector', conditional=False,
               learn_pb=True, learned_variance=True,
               device=torch.device('cpu'),
               do_periodic_angles=False, hold_dead_latent_rows=False)


def paths(seed=1):
    torch.manual_seed(seed)
    return torch.randn(B, T + 1, DIM) * 0.3


@torch.no_grad()
def logpb(g, traj):
    _, _, lpb, _ = g.get_traj_replay(traj, lambda b: uniform_discretizer(b, T), condition=None, mol_batch=None)
    return lpb.sum(-1)


@torch.no_grad()
def perturb_trunk(g, scale=0.5):
    """Move the shared trunk the way P_F's training would."""
    for mod in (g.s_model, g.t_model):
        for p in mod.parameters():
            p.add_(scale * torch.randn_like(p))


def test_full_freeze_isolates_pb_from_trunk_drift():
    g, traj = make_gfn(), paths()
    before = logpb(g, traj)

    # control: without a freeze, moving the trunk moves P_B (the head-only leak)
    g_ctrl = make_gfn()
    for p in g_ctrl.backward_policy.parameters():
        p.requires_grad_(False)
    perturb_trunk(g_ctrl)
    assert not torch.allclose(logpb(g_ctrl, traj), before, atol=1e-4), \
        "perturbing the trunk should move P_B when only the head is frozen -- else the test is blind"

    g.freeze_backward_policy()
    assert g.pb_frozen
    perturb_trunk(g)
    assert torch.allclose(logpb(g, traj), before, atol=1e-6), \
        "full freeze: P_B must not move when the live trunk moves"

    # no gradient reaches anything through P_B
    g.train()
    _, lpf, lpb, _ = g.get_traj_replay(traj, lambda b: uniform_discretizer(b, T), condition=None, mol_batch=None)
    assert not lpb.requires_grad
    assert lpf.requires_grad


def test_unfreeze_returns_to_live_function():
    g, traj = make_gfn(), paths()
    g.freeze_backward_policy()
    perturb_trunk(g)
    g.unfreeze_backward_policy()
    assert not g.pb_frozen
    live = make_gfn()
    live.load_state_dict(g.state_dict())
    assert torch.allclose(logpb(g, traj), logpb(live, traj), atol=1e-6)


def test_snapshot_state_round_trip_restores_the_same_pb():
    g, traj = make_gfn(), paths()
    g.freeze_backward_policy()
    frozen_ref = logpb(g, traj)
    state = g.pb_snapshot_state()
    assert state is not None and all(v.device.type == 'cpu' for v in state.values())

    # a "resumed leg": same live weights but a drifted trunk, snapshot restored from state
    g2 = make_gfn()
    g2.load_state_dict(g.state_dict())
    perturb_trunk(g2)
    g2.freeze_backward_policy(source_state=state)
    assert torch.allclose(logpb(g2, traj), frozen_ref, atol=1e-6), \
        "restoring the checkpoint's snapshot must give the ORIGINAL P_B, not a re-snapshot of the drifted trunk"

    # the wrong thing -- re-snapshotting the drifted trunk -- gives a different P_B
    g3 = make_gfn()
    g3.load_state_dict(g.state_dict())
    perturb_trunk(g3)
    g3.freeze_backward_policy()
    assert not torch.allclose(logpb(g3, traj), frozen_ref, atol=1e-4)

    assert make_gfn().pb_snapshot_state() is None


class _Stub:
    """The slice of Modeller the checkpointer's restore hook touches."""

    def __init__(self, g, ema):
        self.gfn_model, self.ema_model, self.calls = g, ema, []

    def set_pb_freeze(self, mode, source_state=None):
        self.calls.append((mode, source_state is not None))
        if mode is None:
            for m in (self.gfn_model, self.ema_model):
                m.unfreeze_backward_policy()
            return
        fr = self.gfn_model.freeze_backward_policy(source_state=source_state)
        self.ema_model.install_pb_snapshot(fr)


def test_checkpointer_restore_hook_restores_and_lifts():
    g, ema, traj = make_gfn(), make_gfn(), paths()
    g.freeze_backward_policy()
    ref = logpb(g, traj)
    ck = {'pb_frozen': g.pb_snapshot_state()}

    holder = type('H', (), {})()
    holder.modeller = _Stub(make_gfn(), make_gfn())
    holder.modeller.gfn_model.load_state_dict(g.state_dict())
    perturb_trunk(holder.modeller.gfn_model)
    Checkpointer._restore_pb_snapshot(holder, ck)
    assert holder.modeller.calls == [('full', True)]
    assert holder.modeller.gfn_model.pb_frozen and holder.modeller.ema_model.pb_frozen
    assert holder.modeller.gfn_model._pb_frozen is holder.modeller.ema_model._pb_frozen, \
        "train and EMA models must share ONE snapshot object"
    assert torch.allclose(logpb(holder.modeller.gfn_model, traj), ref, atol=1e-6)

    # a trainable checkpoint (None) LIFTS a freeze -- a rewind must not keep a later snapshot
    Checkpointer._restore_pb_snapshot(holder, {'pb_frozen': None})
    assert holder.modeller.calls[-1] == (None, False)
    assert not holder.modeller.gfn_model.pb_frozen

    # pre-snapshot checkpoints (no key): no-op
    n = len(holder.modeller.calls)
    Checkpointer._restore_pb_snapshot(holder, {})
    assert len(holder.modeller.calls) == n


def test_model_state_dict_does_not_carry_the_snapshot():
    g = make_gfn()
    keys_before = set(g.state_dict().keys())
    g.freeze_backward_policy()
    assert set(g.state_dict().keys()) == keys_before, \
        "the snapshot must stay out of the model state_dict (checkpoint compatibility)"
    assert 'backward_policy' not in [n for n, _ in g.named_children() if n == '_pb_frozen']
