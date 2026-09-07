"""balance.kind: gated_ramp -- one sensor, two motions, hard rails
(docs/design/rarer_rollouts.md, the v1 replay/bwd controller).

The stage is the REAL shipped equilibration stage (configs/prod_sep02, active
protocol) parsed by the real protocol.Stage, with the controller patched in; the
tick is the real StageProtocol method bound onto a stub carrying only what it
reads. A hand-built stage would test the parser against my own idea of the
config's shape.

THE LOAD-BEARING TEST is test_active_modes_knows_the_kind. Stage.active_modes and
read_modes ENUMERATE balance kinds and fall through to self.balance['rules'] for
any other -- a KeyError on the first fused step, which is exactly how 'ratio'
first shipped (the comment in active_modes records it). A new kind that is not
in those tuples parses fine, loads fine, and dies at step 1.
"""
import copy
import pathlib
from types import MethodType, SimpleNamespace

import pytest
import yaml

from energy_sampling import config_invariants
from energy_sampling.protocol import Stage, StageProtocol

HERE = pathlib.Path(__file__).resolve().parent
SHIPPED = HERE.parent.parent / 'configs' / 'prod_sep02' / 'p02_mip_lr1.yaml'

BOUNDS = {'bwd': [0.25, 0.9], 'replay': [0.1, 0.75]}
BALANCE = {'kind': 'gated_ramp', 'ramp': 'replay', 'guard': 'bwd',
           'pinned': {'fwd': 0.0}, 'metric': 'bwd/under_coverage_rise150',
           'bar': 1.0, 'up': 0.0017, 'down': 0.043, 'bounds': BOUNDS}


def _spec(**patch):
    cfg = yaml.safe_load(SHIPPED.read_text(encoding='utf-8'))
    for i, st in enumerate(config_invariants.active_stages(cfg)):
        if st.get('name') == 'equilibration':
            st = copy.deepcopy(st)
            st['fwd_rollout_every'] = 10
            st['flags']['z_calibration'] = False
            st['fracs'] = {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}
            st.pop('min_fracs', None)
            st['balance'] = copy.deepcopy(BALANCE)
            st.update(patch)
            return st, i
    raise AssertionError('no equilibration stage in the active protocol')


def _stage(**patch):
    spec, i = _spec(**patch)
    return Stage(spec, i)


class _Tracker:
    def __init__(self, **values):
        self.values = values

    def get(self, direction, name, default=None):
        return self.values.get(f'{direction}/{name}', default)


def _proto(stg, tracker, share=None):
    m = SimpleNamespace(fwd_frac=0.0, bwd_frac=0.5, replay_frac=0.5, step_ind=0,
                        metric_tracker=tracker)
    p = SimpleNamespace(m=m, stage=stg, ctrl={'gates': {}, 'gr_share': share, 'gr_fired': 0.0})
    p._resolve = MethodType(StageProtocol._resolve, p)
    p._gated_ramp_tick = MethodType(StageProtocol._gated_ramp_tick, p)
    return p


# ---------------------------------------------------------------- parsing

def test_parses_the_shipped_stage_with_the_controller():
    stg = _stage()
    b = stg.balance
    assert b['kind'] == 'gated_ramp' and b['ramp'] == 'replay' and b['guard'] == 'bwd'
    assert b['bar'] == 1.0 and b['up'] == 0.0017 and b['down'] == 0.043
    assert b['pinned'] == {'fwd': 0.0}
    assert set(b['metrics']) == {'bwd', 'replay'}, 'the split pair, for pinned/bounds parsing'


@pytest.mark.parametrize('patch, match', [
    ({'ramp': 'bwd'}, 'distinct'),
    ({'bar': 0.0}, 'strictly positive'),
    ({'bar': -1.0}, 'strictly positive'),
    ({'up': 0.0}, r'\(0, 1\]'),
    ({'down': 2.0}, r'\(0, 1\]'),
    ({'gain': 0.5}, 'unknown keys'),
    ({'metric': 'under_coverage'}, 'dir/name'),
])
def test_refuses_malformed_controllers(patch, match):
    spec, i = _spec()
    spec['balance'].update(patch)
    with pytest.raises(ValueError, match=match):
        Stage(spec, i)


def test_refuses_a_pin_that_disagrees_with_fracs():
    spec, i = _spec()
    spec['fracs']['fwd'] = 0.05           # pinned says 0.0
    with pytest.raises(ValueError, match='pinned'):
        Stage(spec, i)


# ------------------------------------------------------- active / read modes

def test_active_modes_knows_the_kind():
    """Must not raise, and must name BOTH split modes plus the pinned one.
    Against a build where 'gated_ramp' is missing from active_modes' kind tuple
    this raises KeyError('rules') -- the failure the fused step would hit."""
    stg = _stage()
    modes = stg.active_modes                   # a property; raises KeyError('rules') on the old tuple
    assert {'bwd', 'replay', 'fwd'} <= set(modes)
    # read_modes maps the balance's metric names back to the branches that
    # produce them; bwd produces the sensor, so it must never be dormant --
    # a dormant branch skips even its force-refresh and the sensor goes stale
    assert 'bwd' in stg.read_modes


# --------------------------------------------------------------------- tick

def test_holds_still_while_the_sensor_is_unwritten():
    p = _proto(_stage(), _Tracker())
    p._gated_ramp_tick(p.stage.balance)
    assert (p.m.replay_frac, p.m.bwd_frac) == (0.5, 0.5)
    assert p.ctrl['gr_share'] == 0.5 and p.ctrl['gr_fired'] == 0.0


def test_ramps_up_by_up_when_the_guard_is_quiet():
    p = _proto(_stage(), _Tracker(**{'bwd/under_coverage_rise150': 0.2}))
    p._gated_ramp_tick(p.stage.balance)
    assert p.m.replay_frac == pytest.approx(0.5 + 0.0017)
    assert p.m.bwd_frac == pytest.approx(0.5 - 0.0017)
    assert p.ctrl['gr_fired'] == 0.0


def test_drops_by_down_and_reports_fired_when_the_guard_trips():
    p = _proto(_stage(), _Tracker(**{'bwd/under_coverage_rise150': 2.5}))
    p._gated_ramp_tick(p.stage.balance)
    assert p.m.replay_frac == pytest.approx(0.5 - 0.043)
    assert p.m.bwd_frac == pytest.approx(0.5 + 0.043)
    assert p.ctrl['gr_fired'] == 1.0


def test_bar_is_a_strict_threshold():
    """Exactly AT the bar is 'not rising': the deadband is the bar itself."""
    p = _proto(_stage(), _Tracker(**{'bwd/under_coverage_rise150': 1.0}))
    p._gated_ramp_tick(p.stage.balance)
    assert p.ctrl['gr_fired'] == 0.0 and p.m.replay_frac > 0.5


def test_rails_hold_at_both_ends():
    up = _proto(_stage(), _Tracker(**{'bwd/under_coverage_rise150': 0.0}), share=0.749)
    up._gated_ramp_tick(up.stage.balance)
    assert up.m.replay_frac == pytest.approx(0.75), 'replay cap 0.75 (== 1 - bwd floor 0.25)'
    assert up.m.bwd_frac == pytest.approx(0.25)
    down = _proto(_stage(), _Tracker(**{'bwd/under_coverage_rise150': 9.0}), share=0.12)
    down._gated_ramp_tick(down.stage.balance)
    assert down.m.replay_frac == pytest.approx(0.10), 'replay floor 0.1 (== 1 - bwd cap 0.9)'
    assert down.m.bwd_frac == pytest.approx(0.90)


def test_pinned_mode_is_reasserted_every_tick():
    p = _proto(_stage(), _Tracker())
    p.m.fwd_frac = 0.3                    # something drifted it
    p._gated_ramp_tick(p.stage.balance)
    assert p.m.fwd_frac == 0.0


def test_the_split_pair_is_conserved():
    p = _proto(_stage(), _Tracker(**{'bwd/under_coverage_rise150': 3.0}))
    for _ in range(25):
        p._gated_ramp_tick(p.stage.balance)
    assert p.m.replay_frac + p.m.bwd_frac == pytest.approx(1.0)
    assert p.m.replay_frac == pytest.approx(0.10), '25 ticks x 0.043 from 0.5 is past the floor; must rail, not overshoot'
