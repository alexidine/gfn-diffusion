"""fwd_rollout_every: the stage key behind rarer rollouts (docs/design/rarer_rollouts.md).

The tests that matter are the REFUSALS. Accepting a good config proves nothing
about a guard; a guard is a claim about what is NOT allowed through, and it has to
be shown to fire. Both refusal tests fail against a build where the key lands on
the Namespace silently (the pre-change behaviour of an unknown stage key).

The stage under test is taken from the ACTIVE protocol via
config_invariants.active_stages -- the shipped config carries several protocols
each with an 'equilibration' stage, and the invariant only reads the live one, so
mutating the first stage found by name would test nothing.
"""
import copy
import pathlib

import pytest
import yaml

from energy_sampling import config_invariants
from energy_sampling.protocol import Stage

HERE = pathlib.Path(__file__).resolve().parent
SHIPPED = HERE.parent.parent / 'configs' / 'prod_sep02' / 'p02_mip_lr1.yaml'
RULE = 'fwd_rollout_cadence_is_well_formed'


def _cfg():
    return yaml.safe_load(SHIPPED.read_text(encoding='utf-8'))


def _equilibration(cfg):
    """(stage dict, index) from the ACTIVE protocol; the dict is cfg's own object,
    so mutating it mutates cfg."""
    for i, st in enumerate(config_invariants.active_stages(cfg)):
        if isinstance(st, dict) and st.get('name') == 'equilibration':
            return st, i
    raise AssertionError('active protocol has no equilibration stage')


def _violations(cfg):
    return [v for v in config_invariants.check(cfg) if v.rule == RULE]


# ------------------------------------------------------------------ Stage

def test_stage_accepts_the_key_and_defaults_to_zero():
    st, i = _equilibration(_cfg())
    assert Stage(copy.deepcopy(st), i).fwd_rollout_every == 0, \
        'absent key must mean "every step", today\'s behaviour'
    st = copy.deepcopy(st)
    st['fwd_rollout_every'] = 10
    st['flags']['z_calibration'] = False
    assert Stage(st, i).fwd_rollout_every == 10


def test_stage_refuses_negative():
    st, i = _equilibration(_cfg())
    st = copy.deepcopy(st)
    st['fwd_rollout_every'] = -1
    with pytest.raises(ValueError, match='fwd_rollout_every'):
        Stage(st, i)


def test_stage_refuses_cadence_with_z_calibration_on():
    """The load-bearing refusal. The shipped stage carries z_calibration: true, so
    adding the cadence key WITHOUT turning the servo off must die at load, not
    quietly call the energy function on every skipped step."""
    st, i = _equilibration(_cfg())
    st = copy.deepcopy(st)
    st['fwd_rollout_every'] = 10
    assert st['flags'].get('z_calibration') is True, 'precondition: shipped stage has the servo on'
    with pytest.raises(ValueError, match='z_calibration'):
        Stage(st, i)


# -------------------------------------------------------- config_invariants

def test_invariant_is_silent_on_the_shipped_config():
    assert _violations(_cfg()) == [], 'no cadence key => nothing to say'


def test_invariant_fires_on_servo_left_on():
    cfg = _cfg()
    st, _ = _equilibration(cfg)
    st['fwd_rollout_every'] = 10           # servo still true on the shipped stage
    cfg['z_calibration']['fill_threshold'] = 0.5
    v = _violations(cfg)
    assert v and any('z_calibration' in x.detail for x in v)


def test_invariant_fires_on_fill_disabled():
    """fill_threshold 0 is 'the fill is off by configuration'. With the servo off
    too, NOTHING pins log Z at a rollout -- the exact failure the design forbids."""
    cfg = _cfg()
    st, _ = _equilibration(cfg)
    st['fwd_rollout_every'] = 10
    st['flags']['z_calibration'] = False
    cfg['z_calibration']['fill_threshold'] = 0
    v = _violations(cfg)
    assert v and any('fill_threshold' in x.detail for x in v)


def test_invariant_is_silent_on_a_well_formed_cadence():
    cfg = _cfg()
    st, _ = _equilibration(cfg)
    st['fwd_rollout_every'] = 10
    st['flags']['z_calibration'] = False
    cfg['z_calibration']['fill_threshold'] = 0.5
    assert _violations(cfg) == []


# -------------------------------------------- fwd_rollout_drift_max (item A)

def test_stage_accepts_the_drift_key_and_defaults_to_off():
    st, i = _equilibration(_cfg())
    assert Stage(copy.deepcopy(st), i).fwd_rollout_drift_max == 0.0
    st = copy.deepcopy(st)
    st['fwd_rollout_every'] = 10
    st['flags']['z_calibration'] = False
    st['fwd_rollout_drift_max'] = 2.5
    assert Stage(st, i).fwd_rollout_drift_max == 2.5


@pytest.mark.parametrize('bar', [-1.0, 3.0])
def test_stage_refuses_a_negative_bar_or_a_bar_without_a_cadence(bar):
    """A bar with no cadence is dead config: the trigger only adds rollouts to
    steps a cadence skipped."""
    st, i = _equilibration(_cfg())
    st = copy.deepcopy(st)
    st['flags']['z_calibration'] = False
    if bar < 0:
        st['fwd_rollout_every'] = 10
    st['fwd_rollout_drift_max'] = bar
    with pytest.raises(ValueError, match='fwd_rollout_drift_max'):
        Stage(st, i)


def test_invariant_fires_on_a_drift_bar_without_a_cadence():
    cfg = _cfg()
    st, _ = _equilibration(cfg)
    st['fwd_rollout_drift_max'] = 2.5
    v = _violations(cfg)
    assert v and any('fwd_rollout_drift_max' in x.detail for x in v)


def test_invariant_is_silent_on_an_armed_trigger_under_a_cadence():
    cfg = _cfg()
    st, _ = _equilibration(cfg)
    st['fwd_rollout_every'] = 10
    st['flags']['z_calibration'] = False
    st['fwd_rollout_drift_max'] = 2.5
    cfg['z_calibration']['fill_threshold'] = 0.5
    assert _violations(cfg) == []
