"""freeze_pb / unfreeze_pb stage actions: parsed by the real Stage parser,
dispatched to Modeller.set_pb_freeze with the right mode."""
import os
import sys

import pytest

_here = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(_here)),
          os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_here))), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from protocol import Stage, StageProtocol  # noqa: E402

EQ = {'name': 'equilibration', 'train_mode': 'fused', 'bwd_sampling_mode': 'prior'}


@pytest.mark.parametrize('action, expected', [
    ('freeze_pb', ('freeze_pb', '')),
    ('freeze_pb:full', ('freeze_pb', 'full')),
    ('freeze_pb:head', ('freeze_pb', 'head')),
    ('unfreeze_pb', ('unfreeze_pb', '')),
])
def test_actions_parse(action, expected):
    st = Stage({**EQ, 'on_enter': [action]}, 1)
    assert st.on_enter == [expected]


@pytest.mark.parametrize('action', ['freeze_pb:bogus', 'freeze_pb:1', 'unfreeze_pb:full'])
def test_bad_arguments_fail_at_load(action):
    with pytest.raises(ValueError):
        Stage({**EQ, 'on_exit': [action]}, 1)


class _M:
    def __init__(self):
        self.calls = []

    def set_pb_freeze(self, mode, source_state=None):
        self.calls.append(mode)


def test_dispatch_reaches_set_pb_freeze():
    proto = StageProtocol.__new__(StageProtocol)
    proto.m = _M()
    proto._run_action('freeze_pb', '', None)
    proto._run_action('freeze_pb', 'head', None)
    proto._run_action('unfreeze_pb', '', None)
    assert proto.m.calls == ['full', 'head', None]
