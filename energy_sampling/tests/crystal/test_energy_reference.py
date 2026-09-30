"""
The Z side of energy_config.energy_reference: the tracker's all-condition level and
the untrusted-condition fallback it feeds (condition_log_z.untrusted_z), the per-
trajectory Z the TB residual uses, and the config rule that ties the two keys.

Nothing is built here -- the tracker is fed tensors directly.

    pytest tests/crystal/test_energy_reference.py
"""
import math
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.fast   # torch imported, but nothing is built

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling.buffer import ConditionLogZTracker  # noqa: E402
from energy_sampling.config_invariants import energy_reference_is_consistent  # noqa: E402
from energy_sampling.gflownet_losses import get_tb_loss, tb_z_per_traj, untrusted_z_for  # noqa: E402


def _tracker(mode='global', half_life=50.0, n=8):
    return ConditionLogZTracker(library_size=n, min_visits=20, trim_frac=0.0,
                                untrusted_z=mode, global_half_life_updates=half_life)


# --- the all-condition level -------------------------------------------------

def test_fallback_is_none_under_head_and_before_the_first_update():
    t = _tracker('head')
    t.update(torch.tensor([0, 1]), torch.tensor([1.0, 3.0]), step=0)
    assert t.untrusted_fallback() is None
    g = _tracker('global')
    assert g.untrusted_fallback() is None
    g.update(torch.tensor([0, 1]), torch.tensor([1.0, 3.0]), step=0)
    assert g.untrusted_fallback() == pytest.approx(2.0)


def test_level_weights_conditions_equally_not_rows():
    # ten rows of condition 0 at 0, one row of condition 1 at 10: the pooled ROW
    # mean is ~0.9, the mean of the two conditions' means is 5
    g = _tracker('global')
    cid = torch.tensor([0] * 10 + [1])
    logw = torch.tensor([0.0] * 10 + [10.0])
    g.update(cid, logw, step=0)
    assert g.untrusted_fallback() == pytest.approx(5.0)


def test_level_half_life_counts_update_calls():
    h = 50.0
    g = _tracker('global', half_life=h)
    for s in range(3000):                  # effective count saturates at 1/(1 - decay)
        g.update(torch.tensor([0]), torch.tensor([0.0]), step=s)
    for s in range(int(h)):
        g.update(torch.tensor([0]), torch.tensor([10.0]), step=3000 + s)
    assert g.untrusted_fallback() == pytest.approx(5.0, abs=0.05)


def test_level_survives_a_state_round_trip_and_old_states_default_to_head():
    g = _tracker('global', half_life=7.0)
    g.update(torch.tensor([0, 3]), torch.tensor([2.0, 4.0]), step=5)
    back = ConditionLogZTracker.from_state_dict(g.state_dict(), current_step=5)
    assert back.untrusted_z == 'global'
    assert back.global_half_life_updates == 7.0
    assert back.global_logw == pytest.approx(g.global_logw)
    assert back.global_effective_count == pytest.approx(g.global_effective_count)

    old = g.state_dict()
    for k in ('untrusted_z', 'global_half_life_updates', 'global_logw', 'global_effective_count'):
        old.pop(k)
    legacy = ConditionLogZTracker.from_state_dict(old, current_step=5)
    assert legacy.untrusted_z == 'head'
    assert math.isnan(legacy.global_logw)
    assert legacy.untrusted_fallback() is None


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match='untrusted_z'):
        ConditionLogZTracker(library_size=4, untrusted_z='learned')


# --- the Z each TB residual uses ------------------------------------------------

def test_no_mask_is_the_learned_head():
    learned = torch.randn(4, requires_grad=True)
    assert tb_z_per_traj(learned) is learned


def test_mask_without_fallback_keeps_the_live_head_on_untrusted_rows():
    learned = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = torch.tensor([10.0, 20.0, 30.0])
    mask = torch.tensor([True, False, True])
    z = tb_z_per_traj(learned, target, mask)
    assert z.tolist() == [10.0, 2.0, 30.0]
    z.sum().backward()
    assert learned.grad.tolist() == [0.0, 1.0, 0.0]


def test_global_fallback_takes_the_head_out_of_the_residual():
    learned = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = torch.tensor([10.0, 20.0, 30.0])
    mask = torch.tensor([True, False, False])
    z = tb_z_per_traj(learned, target, mask, untrusted_log_Z=7.5)
    assert z.tolist() == [10.0, 7.5, 7.5]
    assert not z.requires_grad            # neither branch carries the head's gradient


def test_tb_loss_with_fallback_equals_tb_against_that_constant():
    log_pf, log_pb, log_r = torch.randn(5), torch.randn(5), torch.randn(5)
    learned = torch.randn(5)
    mask = torch.zeros(5, dtype=torch.bool)
    got = get_tb_loss(learned, log_pb, log_pf, log_r, beta=80.0,
                      log_Z_target=torch.zeros(5), target_mask=mask, untrusted_log_Z=3.0)
    want = get_tb_loss(torch.full((5,), 3.0), log_pb, log_pf, log_r, beta=80.0)
    torch.testing.assert_close(got, want)


def test_untrusted_z_for_only_under_a_persistent_target():
    g = _tracker('global')
    g.update(torch.tensor([0]), torch.tensor([4.0]), step=0)
    assert untrusted_z_for(g, use_persistent_z=True) == pytest.approx(4.0)
    assert untrusted_z_for(g, use_persistent_z=False) is None
    assert untrusted_z_for(None, use_persistent_z=True) is None


# --- the config rule -------------------------------------------------------------

def _rule(cfg):
    return [v.detail for v in energy_reference_is_consistent(cfg)]


def test_rule_passes_the_defaults_and_absence():
    assert _rule({}) == []
    assert _rule({'energy_config': {'energy_reference': None},
                  'condition_log_z': {'untrusted_z': 'head'}}) == []
    assert _rule({'energy_config': {'energy_reference': 'seed_min'},
                  'condition_log_z': {'untrusted_z': 'global'}}) == []


def test_rule_refuses_global_without_the_reference():
    out = _rule({'condition_log_z': {'untrusted_z': 'global'}})
    assert len(out) == 1 and 'needs energy_config.energy_reference' in out[0]


def test_rule_refuses_the_reference_with_temperature_conditioning():
    out = _rule({'energy_config': {'energy_reference': 'seed_min'}, 'temperature_conditioning': True})
    assert len(out) == 1 and 'temperature_conditioning' in out[0]


def test_rule_refuses_unknown_values():
    out = _rule({'energy_config': {'energy_reference': 'seed_mean'},
                 'condition_log_z': {'untrusted_z': 'learned'}})
    assert len(out) == 2
