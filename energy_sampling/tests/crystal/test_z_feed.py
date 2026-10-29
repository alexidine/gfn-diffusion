"""condition_log_z.z_feed through Modeller._offpolicy_feeds_log_z (the backward and replay branches' update_log_z) and
Modeller._eval_feed_log_z (the eval's forward draws), on a Modeller built without __init__ and a real
ConditionLogZTracker. What each test would catch:

  * backward/replay rows writing ema_logw under 'forward', or no longer writing it under 'mixed' (the conformer
    route's code default);
  * the eval writing ema_logw outside a forward-only, backward-branch-only stage under update_log_z, which mixes the
    eval stream into a train stream (the eval-cycle sawblade fwd_eval_sampling documents);
  * a non-finite eval row entering the level;
  * the eval feed keeping the tracker's own long half-life, so the level averages the whole MLE stage instead of
    following the current policy;
  * config_invariants' copy of the modes and defaults drifting from buffer.py's.

CPU, synthetic log weights, no model or energy.

    python -m pytest -q tests/crystal/test_z_feed.py
"""
from types import SimpleNamespace

import pytest
import torch

from energy_sampling import buffer
from energy_sampling import config_invariants as ci
from energy_sampling.buffer import ConditionLogZTracker
from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast


def _modeller(z_feed=None, train_mode='bwd', update_log_z=True, half_life=2.0):
    m = Modeller.__new__(Modeller)
    flags = {'update_log_z': update_log_z}
    m.protocol = SimpleNamespace(flag=lambda name: flags.get(name, False),
                                 stage=SimpleNamespace(train_mode=train_mode))
    if z_feed is not None:
        m.z_feed = z_feed
    m.eval_half_life_visits = half_life
    m.condition_log_z = ConditionLogZTracker(library_size=3)
    m.step_ind = 0
    return m


@pytest.mark.parametrize('z_feed,flag,expected', [('mixed', True, True), ('mixed', False, False),
                                                  ('forward', True, False), ('forward', False, False),
                                                  (None, True, True)])
def test_backward_and_replay_write_the_level_only_under_mixed_and_the_flag(z_feed, flag, expected):
    assert _modeller(z_feed, update_log_z=flag)._offpolicy_feeds_log_z() is expected


@pytest.mark.parametrize('z_feed,train_mode,flag,expected', [('forward', 'bwd', True, True),
                                                             ('forward', 'fused', True, False),
                                                             ('forward', 'bwd', False, False),
                                                             ('mixed', 'bwd', True, False),
                                                             (None, 'bwd', True, False)])
def test_the_eval_writes_the_level_only_in_a_backward_only_stage_under_forward_and_the_flag(
        z_feed, train_mode, flag, expected):
    m = _modeller(z_feed, train_mode=train_mode, update_log_z=flag)
    wrote = m._eval_feed_log_z(torch.tensor([0, 0, 1]), torch.tensor([3.0, 3.0, -2.0]))
    assert wrote is expected
    ema = m.condition_log_z.ema_logw
    if expected:
        assert ema[0].item() == pytest.approx(3.0) and ema[1].item() == pytest.approx(-2.0)
        assert torch.isnan(ema[2])
    else:
        assert torch.isnan(ema).all()


def test_a_nonfinite_eval_row_is_left_out():
    m = _modeller('forward')
    assert m._eval_feed_log_z(torch.tensor([0, 0, 0, 1]),
                              torch.tensor([4.0, float('nan'), float('-inf'), float('inf')]))
    assert m.condition_log_z.ema_logw[0].item() == pytest.approx(4.0)
    assert torch.isnan(m.condition_log_z.ema_logw[1])
    assert int(m.condition_log_z.count[0]) == 1
    assert not _modeller('forward')._eval_feed_log_z(torch.tensor([0]), torch.tensor([float('nan')]))


def test_the_eval_level_follows_the_policy_at_its_own_half_life():
    """Two evals of equal evidence at levels 0 then 10: the second carries weight 1 / (1 + 2^(-1/h)) at half-life h
    calls. At the tracker's own 200 the level would sit near the mean of the two."""
    ids = torch.zeros(5, dtype=torch.long)
    for h in (2.0, 1.0):
        m = _modeller('forward', half_life=h)
        m._eval_feed_log_z(ids, torch.zeros(5))
        m._eval_feed_log_z(ids, torch.full((5,), 10.0))
        assert m.condition_log_z.ema_logw[0].item() == pytest.approx(10.0 / (1.0 + 0.5 ** (1.0 / h)), rel=1e-5)
    assert 10.0 / (1.0 + 0.5 ** (1.0 / buffer.DEFAULT_HALF_LIFE_VISITS)) < 5.01


def test_the_rule_copies_match_the_trainer_constants():
    assert ci.Z_FEED_MODES == buffer.Z_FEED_MODES
    assert ci.Z_FEED_DEFAULTS == dict(z_feed='mixed', eval_half_life_visits=buffer.DEFAULT_EVAL_HALF_LIFE_VISITS)
