"""Gates for the raw-state passthrough into ``GFN.predict_next_state``.

The set policy cannot consume ``s_emb``: ``StateEncoding`` has already mixed the
coordinates, so the per-token structure is gone. The passthrough hands it the raw state
instead, gated on the policy declaring ``wants_raw_state``.

``models/gfn.py`` is SHARED with crystal, so the first test here is the one that matters
most -- it pins that the flat path cannot be perturbed by the new argument at all. Every
test is written so that breaking the property FAILS rather than degrades:

  1. A flat PolicyModel ignores `state` entirely -- proven by feeding it a wildly wrong
     one and requiring a bitwise-identical head.
  2. A policy declaring `wants_raw_state` receives the RAW state, not the embedding. The
     separator is the width: s_emb_dim differs from dim, so a mix-up cannot pass.
  3. Declaring `wants_raw_state` and calling without a state RAISES rather than silently
     falling back to s_emb.
  4. `wants_raw_state` with dplr_rank > 0 REFUSES, because the low-rank block orders
     disagree and u_raw would be silently transposed.
  5. A real SetPolicy rolls a trajectory end to end, and fwd/replay agree bitwise.

    python -m pytest -q tests/models/test_policy_raw_state_passthrough.py
"""
import os
import sys

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for _root in (os.path.dirname(_here),                                   # gfn_diffusion
              os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import pytest
import torch
import torch.nn as nn

from energy_sampling.models.gfn import GFN
from energy_sampling.models.set_policy import SetPolicy
from energy_sampling.utils import uniform_discretizer, get_gfn_init_state

DEVICE = 'cpu'
DIM = 7
S_EMB = 32          # deliberately != DIM, so swapping state for s_emb cannot pass silently
T_DIM = 8
TRAJ = 5


def _gfn(dplr_rank=0, **kw):
    torch.manual_seed(0)
    return GFN(dim=DIM, s_emb_dim=S_EMB, conditions_dim=0, harmonics_dim=8, t_dim=T_DIM,
               device=DEVICE, angular_mask=[True] * DIM, dplr_rank=dplr_rank, **kw)


def _emb(gfn, state):
    """The two encoded arguments predict_next_state is normally called with."""
    return (gfn.s_model(gfn.expand_state_for_policy(state), None),
            gfn.t_model(torch.full((state.shape[0],), 0.3)))


class _Recorder(nn.Module):
    """Stands in for a set policy: declares the flag, records what it was handed."""

    wants_raw_state = True

    def __init__(self, dim, out_per_token=2):
        super().__init__()
        self.seen = None
        self.width = out_per_token * dim

    def forward(self, x, t_emb):
        self.seen = x.detach().clone()
        return torch.zeros(x.shape[0], self.width)


# --------------------------------------------------------------------------- 1

def test_flat_policy_is_untouched_by_the_new_argument():
    """The crystal regression guard. `state` must be inert on the flat path.

    Not 'the shapes still work' -- a wildly wrong state is fed and the head is required
    to be BITWISE identical, so a policy that started reading the argument would fail.
    """
    gfn = _gfn()
    gfn.eval()
    torch.manual_seed(1)
    state = torch.randn(4, DIM) * 0.2
    s_emb, t_emb = _emb(gfn, state)
    junk = torch.randn(4, DIM) * 100.0

    with torch.no_grad():
        two_arg = gfn.predict_next_state(s_emb, t_emb)
        with_state = gfn.predict_next_state(s_emb, t_emb, state)
        with_junk = gfn.predict_next_state(s_emb, t_emb, junk)

    assert torch.equal(two_arg, with_state), \
        'passing the raw state changed a flat-policy head; the argument is not inert'
    assert torch.equal(two_arg, with_junk), \
        'a nonsense raw state changed a flat-policy head; the flat path is reading it'


# --------------------------------------------------------------------------- 2

def test_raw_state_policy_receives_the_state_not_the_embedding():
    gfn = _gfn()
    gfn.forward_policy = _Recorder(DIM)
    torch.manual_seed(2)
    state = torch.randn(3, DIM) * 0.2
    s_emb, t_emb = _emb(gfn, state)
    assert s_emb.shape[-1] != state.shape[-1], \
        'the separator is gone: s_emb and state are the same width, so this test is blind'

    with torch.no_grad():
        gfn.predict_next_state(s_emb, t_emb, state)

    seen = gfn.forward_policy.seen
    assert seen is not None, 'the policy was never called'
    assert seen.shape == state.shape, \
        f'policy received width {seen.shape[-1]}, expected the raw state width {DIM}'
    assert torch.equal(seen, state), 'policy received something other than the raw state'


# --------------------------------------------------------------------------- 3

def test_raw_state_policy_without_a_state_raises():
    """Silently falling back to s_emb would train a wrong-but-plausible policy."""
    gfn = _gfn()
    gfn.forward_policy = _Recorder(DIM)
    torch.manual_seed(3)
    state = torch.randn(2, DIM) * 0.2
    s_emb, t_emb = _emb(gfn, state)

    with pytest.raises(ValueError, match='wants_raw_state'):
        gfn.predict_next_state(s_emb, t_emb)
    assert gfn.forward_policy.seen is None, 'the policy ran despite the missing state'


# --------------------------------------------------------------------------- 4

def test_raw_state_policy_refuses_dplr():
    """Block orders disagree: rank-major out, dim-major in. Refuse, do not transpose."""
    gfn = _gfn(dplr_rank=6)
    gfn.forward_policy = _Recorder(DIM)
    torch.manual_seed(4)
    state = torch.randn(2, DIM) * 0.2
    s_emb, t_emb = _emb(gfn, state)

    with pytest.raises(NotImplementedError, match='dplr_rank'):
        gfn.predict_next_state(s_emb, t_emb, state)

    # and the same GFN is fine with the flat policy, so the refusal is about the pairing
    flat = _gfn(dplr_rank=6)
    with torch.no_grad():
        head = flat.predict_next_state(*_emb(flat, state))
    assert torch.isfinite(head).all()


# --------------------------------------------------------------------------- 5

def test_set_policy_rolls_a_trajectory_and_replay_agrees():
    """End to end on the real module, with the strongest available check.

    fwd/replay bitwise agreement is what catches a head that is merely shaped right: the
    replay rescores the stored trajectory through the same policy, so any state-dependent
    inconsistency in the passthrough shows up as a mismatch.
    """
    gfn = _gfn()
    torch.manual_seed(5)
    static = torch.randn(DIM, 6)                    # [dim, F]; any static identity will do
    gfn.forward_policy = SetPolicy(static, [True] * DIM, T_DIM,
                                   hidden_dim=16, layers=2, out_per_token=2)
    gfn.eval()

    init = get_gfn_init_state(4, DIM, DEVICE)
    disc = lambda b: uniform_discretizer(b, TRAJ)
    with torch.no_grad():
        s, pf, pb, _ = gfn.get_traj_fwd(init, disc, None, False, None, detach_traj=True)
        _, rpf, rpb, _ = gfn.get_traj_replay(s, disc, False, None)

    assert torch.isfinite(s).all() and torch.isfinite(pf).all() and torch.isfinite(pb).all()
    assert s.shape[-1] == DIM
    assert torch.equal(pf, rpf) and torch.equal(pb, rpb), \
        'fwd and replay disagree through the set policy'
