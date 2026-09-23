"""stage.coeff_schedule: a step-clock ramp of energy_config coefficients (protocol.Stage._parse_coeff_schedule,
StageProtocol.energy_coeffs).

The ramp is anchored at the stage's FIRST energy_coeffs() evaluation -- the anchor is stamped in stage_ctrl,
which rides the checkpoint, so a requeued leg continues where it left off -- and progresses as
p = min(1, elapsed / steps): geometric multiplies from the base config value to target, linear adds. It holds
at target. It works on any balance kind (the lexicographic anneal_coeffs is a different, event-driven ramp),
and naming one coefficient in both is refused at parse. The engine is a real StageProtocol over a one-stage
protocol with no balance; only the metric tracker is a dict.
"""
import copy
import math
from types import SimpleNamespace

import pytest

from energy_sampling.protocol import Stage, StageProtocol, fresh_stage_ctrl

GEO = {'bounding_coeff': {'target': 1000.0, 'steps': 200}}
LIN = {'reduction_coeff': {'target': 110.0, 'steps': 100, 'kind': 'linear'}}


def _spec(schedule, balance=None):
    node = {'name': 's', 'train_mode': 'fused', 'fracs': {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}}
    if schedule is not None:
        node['coeff_schedule'] = copy.deepcopy(schedule)
    if balance is not None:
        node['balance'] = copy.deepcopy(balance)
    return node


def _stage(schedule, balance=None):
    return Stage(_spec(schedule, balance), 0)


class _Tracker:
    def __init__(self):
        self.values = {}

    def get(self, direction, name, default=None):
        return self.values.get(f'{direction}/{name}', default)


def _engine(schedule, step=0, ctrl=None, balance=None):
    args = SimpleNamespace(
        protocol='p', protocols=SimpleNamespace(p=SimpleNamespace(stages=[_spec(schedule, balance)])),
        controller=SimpleNamespace(anneal_patience=5, beta=0.025, min_mode_frac=0.001, decay_rate=0.95),
        energy_config=SimpleNamespace(bounding_coeff=10.0, reduction_coeff=10.0, lambda_mix=1.0, name='elj'))
    m = SimpleNamespace(args=args, stage='s', stage_ctrl=ctrl if ctrl is not None else fresh_stage_ctrl(),
                        metric_tracker=_Tracker(), step_ind=step,
                        fwd_frac=0.0, bwd_frac=0.5, replay_frac=0.5)
    return StageProtocol(m), m


# ---------------------------------------------------------------- parsing

def test_absent_or_empty_means_no_schedule():
    assert _stage(None).coeff_schedule == {}
    assert _stage({}).coeff_schedule == {}


def test_parses_with_geometric_default():
    st = _stage(GEO)
    assert st.coeff_schedule == {'bounding_coeff': {'target': 1000.0, 'steps': 200, 'kind': 'geometric'}}
    assert _stage(LIN).coeff_schedule['reduction_coeff']['kind'] == 'linear'


@pytest.mark.parametrize('bad, match', [
    ({'bounding_coeff': {'target': 100.0}}, "needs 'target' and 'steps'"),
    ({'bounding_coeff': {'target': 100.0, 'steps': 10, 'rate': 0.9}}, 'unknown keys'),
    ({'bounding_coeff': {'target': 100.0, 'steps': 0}}, 'steps must be an integer >= 1'),
    ({'bounding_coeff': {'target': 100.0, 'steps': 2.5}}, 'steps must be an integer >= 1'),
    ({'bounding_coeff': {'target': '100', 'steps': 10}}, 'target'),
    ({'bounding_coeff': {'target': 100.0, 'steps': 10, 'kind': 'cosine'}}, "kind must be 'geometric' or 'linear'"),
    ({'bounding_coeff': {'target': 0.0, 'steps': 10}}, 'geometric ramp needs target > 0'),
    ('bounding_coeff', 'must be a mapping'),
])
def test_malformed_schedules_are_refused_at_parse(bad, match):
    with pytest.raises(ValueError, match=match):
        _stage(bad)


def test_a_coefficient_in_both_ramps_is_refused():
    lexi = {'kind': 'lexicographic', 'default_boost': {'bwd': 1.0}, 'rules': [],
            'anneal_coeffs': {'bounding_coeff': {'target': 50.0, 'rate': 0.5}}}
    with pytest.raises(ValueError, match='named in both coeff_schedule and balance.anneal_coeffs'):
        _stage(GEO, lexi)
    # a different coefficient in each is fine
    assert _stage(LIN, lexi).coeff_schedule['reduction_coeff']['target'] == 110.0


# ---------------------------------------------------------------- values

def test_geometric_ramp_from_the_base_value_to_target_then_holds():
    p, m = _engine(GEO, step=100)
    assert p.energy_coeffs() == {'bounding_coeff': pytest.approx(10.0)}   # anchored at the first evaluation
    assert p.ctrl['coeff_sched_entry'] == 100
    m.step_ind = 200                                                        # halfway: the geometric mean
    assert p.energy_coeffs()['bounding_coeff'] == pytest.approx(math.sqrt(10.0 * 1000.0))
    m.step_ind = 300
    assert p.energy_coeffs()['bounding_coeff'] == pytest.approx(1000.0)
    m.step_ind = 5000                                                       # holds at target
    assert p.energy_coeffs()['bounding_coeff'] == pytest.approx(1000.0)


def test_linear_ramp():
    p, m = _engine(LIN, step=0)
    assert p.energy_coeffs()['reduction_coeff'] == pytest.approx(10.0)
    m.step_ind = 50
    assert p.energy_coeffs()['reduction_coeff'] == pytest.approx(60.0)
    m.step_ind = 100
    assert p.energy_coeffs()['reduction_coeff'] == pytest.approx(110.0)


def test_a_requeued_leg_continues_the_ramp_from_the_stamped_anchor():
    p, m = _engine(GEO, step=1000)
    p.energy_coeffs()                                     # stamps entry = 1000
    m.step_ind = 1100
    v_leg1 = p.energy_coeffs()['bounding_coeff']
    # leg 2: a fresh engine handed the checkpointed stage_ctrl at the same step reads the same value
    p2, m2 = _engine(GEO, step=1100, ctrl=copy.deepcopy(m.stage_ctrl))
    assert p2.energy_coeffs()['bounding_coeff'] == pytest.approx(v_leg1)
    assert p2.ctrl['coeff_sched_entry'] == 1000
    # a checkpoint from before the key existed (no slot at all) anchors at the resume step
    old_ctrl = fresh_stage_ctrl(); old_ctrl.pop('coeff_sched_entry')
    p3, m3 = _engine(GEO, step=7000, ctrl=old_ctrl)
    assert p3.energy_coeffs()['bounding_coeff'] == pytest.approx(10.0)
    assert p3.ctrl['coeff_sched_entry'] == 7000


def test_a_name_that_is_not_a_numeric_energy_key_is_refused_at_first_evaluation():
    p, m = _engine({'wall_height': {'target': 5.0, 'steps': 10}})
    with pytest.raises(ValueError, match='not a numeric energy_config key'):
        p.energy_coeffs()


def test_the_schedule_does_not_need_a_lexicographic_balance():
    gated = {'kind': 'gated_ramp', 'ramp': 'replay', 'guard': 'bwd', 'pinned': {'fwd': 0.0},
             'metric': 'bwd/under_coverage_rise150', 'bar': 0.0, 'ratchet_metric': 'bwd/under_coverage',
             'ratchet_tol': 0.5, 'ratchet_release_tol': 0.25, 'up': 0.004, 'down': 0.006,
             'bounds': {'bwd': [0.5, 0.5], 'replay': [0.5, 0.5]}}
    p, m = _engine(GEO, step=0, balance=gated)
    p.energy_coeffs()                                     # anchors the clock at step 0
    m.step_ind = 100
    assert p.energy_coeffs()['bounding_coeff'] == pytest.approx(math.sqrt(10.0 * 1000.0))


def test_report_carries_the_progress():
    p, m = _engine(GEO, step=0)
    p.energy_coeffs()
    m.step_ind = 50
    assert p.report()['protocol/coeff_sched_p_bounding_coeff'] == pytest.approx(0.25)
