"""
Stage.read_modes must name the hot-LR sensor's branch.

WHAT THE CLAIM IS. The force-refresh rollout is what keeps a low-fraction or
untrained branch's rolling stats fresh, and read_modes is its allow-list: a mode
nobody reads skips the rollout entirely and is simply never written. The hot-LR
sensor advances its window ONLY on fresh writes, so a sensor pointed at a branch
outside read_modes never fills its window -- it reports NO_READING for the whole
stage and never claims to be blind, which is the failure mode the sensor exists
to remove.

WHY THIS TEST EXISTS AT ALL. read_modes already carried a clause naming the
PLATEAU LR sensor's channels, for exactly this reason. `plateau` then left
LR_SENSOR_KINDS, and the clause went dead: a gate on a retired key can never
fire, so it read as "this is handled" while handling nothing. This test is what
stops the replacement clause dying the same way.

The stage spec is REAL -- conditional_vargrad's var_conditioning, transcribed
from configs/mk_dev.yaml -- and it is the right one because its balance rules
read {fwd, bwd} only. On mk_dev today all four declared channels already sit in
read_modes through those rules, so the clause is a no-op on the live configs;
this stage is where the coincidence runs out.

Mutation check (re-introduces the bug and requires a FAILURE):
  - drop the hot_lr_sensor clause from read_modes -> test_a_replay_sensor_keeps
    _replay_fresh fails, and its companion shows the pass was not incidental.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from protocol import Stage


#: conditional_vargrad / var_conditioning, verbatim from configs/mk_dev.yaml.
#: Its balance metrics are fwd/* and bwd/* only, so `replay` is dormant here.
VAR_CONDITIONING = {
    'name': 'var_conditioning',
    'train_mode': 'fused',
    'bwd_sampling_mode': 'prior',
    'balance': {
        'kind': 'proportional',
        'pinned': {'replay': 0.0},
        'metrics': {'fwd': 'fwd/logw_std_within', 'bwd': 'bwd/logw_std_within'},
        'drive': 'relative',
        'targets': {'fwd': 1.0, 'bwd': 1.0},
        'default_boost': {'fwd': 0.5, 'bwd': 0.5},
        'floor': 0.1,
        'alpha': 0.01,
    },
}


def stage(**patch):
    return Stage({**VAR_CONDITIONING, **patch}, 0)


def test_replay_is_dormant_here_without_a_sensor():
    """The premise. If this ever goes green-by-default the test below passes for
    a reason that has nothing to do with the sensor."""
    assert 'replay' not in stage().read_modes


def test_a_replay_sensor_keeps_replay_fresh():
    """The clause under test: declaring the sensor is what pulls the branch in."""
    s = stage(hot_lr_sensor={'channel': 'replay/mle', 'form': 'absolute',
                             'rows': 11, 'above': 3.0})
    assert 'replay' in s.read_modes, (
        'a hot_lr_sensor on replay/* left replay outside read_modes, so the '
        'branch skips its force-refresh rollout, the channel is never written, '
        'and the sensor sits at NO_READING for the entire stage')


def test_the_sensor_does_not_disturb_the_other_modes():
    """It ADDS a branch; it must not narrow the set the balance rules need."""
    before = stage().read_modes
    after = stage(hot_lr_sensor={'channel': 'replay/mle', 'form': 'absolute',
                                 'rows': 11, 'above': 3.0}).read_modes
    assert before <= after and after - before == {'replay'}


@pytest.mark.parametrize('channel', ['fwd/vg_lb', 'bwd/mle'])
def test_an_already_read_branch_is_idempotent(channel):
    """The live mk_dev case: the channel is already covered, and naming it twice
    must not change the answer."""
    s = stage(hot_lr_sensor={'channel': channel, 'rows': 11, 'above': 3.0})
    assert s.read_modes == stage().read_modes | {channel.partition('/')[0]}


def test_a_non_branch_channel_cannot_be_declared_at_all():
    """read_modes only ever sees a branch channel, because the PARSER is the
    stricter of the two: a channel outside MODES is refused at load time rather
    than silently dropped here. That ordering is what makes the clause above a
    total guarantee -- every channel that parses is a channel read_modes keeps
    fresh."""
    with pytest.raises(ValueError, match='<mode>/<channel>'):
        stage(hot_lr_sensor={'channel': 'lr_ctrl/scale', 'form': 'absolute',
                             'rows': 11, 'above': 3.0})


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
