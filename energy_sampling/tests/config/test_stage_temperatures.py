"""A stage's `temperatures` (multi-temperature TB on the store route): its parse (protocol.Stage._parse_temperatures),
where it is read (config_invariants.stage_temperatures_are_read) and the store-route refusals of a stage that sets it
(config_invariants.stage_temperatures_problems, through row_store_problems). Pure config, no torch model.

    python -m pytest -q tests/config/test_stage_temperatures.py
"""
import copy
import pathlib

import pytest
import yaml

from energy_sampling import config_invariants as ci
from energy_sampling.protocol import Stage

pytestmark = pytest.mark.fast
BASE = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'pool4_nov24' / 'tb_m4_visits.yaml'
TEMPS = [2.5, 5.0, 10.0, 25.0]


@pytest.fixture(scope='module')
def base():
    return yaml.safe_load(BASE.read_text(encoding='utf-8'))


def _stage(cfg, name):
    return next(s for s in cfg['protocols'][cfg['protocol']]['stages'] if s['name'] == name)


def multi_t(base, temps=TEMPS):
    """tb_m4_visits (a store-route TB arm: visits, fwd repeats 2, emp_z 1) with temperature conditioning and the TB
    stage drawing over `temps`."""
    cfg = copy.deepcopy(base)
    cfg['temperature_conditioning'] = True
    cfg['noised_prior']['kick_kappa_per_atom'] = 14.9
    cfg['energy_config']['log_temperature_range'] = [-3.0, 1.5]
    _stage(cfg, 'tb')['temperatures'] = list(temps)
    return cfg


def messages(cfg):
    return [v.detail for v in ci.errors(cfg)]


def test_a_store_route_tb_arm_with_temperatures_passes(base):
    assert messages(base) == []
    assert messages(multi_t(base)) == []
    assert ci.store_terms_on(multi_t(base)) and not ci.store_terms_on(base)


def test_single_temperature_tb_under_temperature_conditioning_passes(base):
    cfg = multi_t(base)
    del _stage(cfg, 'tb')['temperatures']
    assert messages(cfg) == [] and not ci.store_terms_on(cfg)


@pytest.mark.parametrize('bad', [[], 'x', 2.5, [0.0], [-1.0], [float('nan')], [float('inf')], [True], [2.5, None]])
def test_the_parse_refuses_anything_but_a_non_empty_list_of_positive_finite_kt(bad):
    with pytest.raises(ValueError, match='temperatures must be a non-empty list of finite kT values > 0'):
        Stage({'name': 'tb', 'temperatures': bad}, 0)


def test_the_parse_keeps_the_list_and_absent_is_none():
    assert Stage({'name': 'tb', 'temperatures': [25, 10, 10]}, 0).temperatures == (25.0, 10.0, 10.0)
    assert Stage({'name': 'tb'}, 0).temperatures is None
    assert Stage({'name': 'tb', 'temperatures': None}, 0).temperatures is None


def test_temperatures_without_temperature_conditioning_or_the_store_is_a_dead_key(base):
    cfg = copy.deepcopy(base)
    _stage(cfg, 'tb')['temperatures'] = TEMPS
    assert any("temperatures needs temperature_conditioning true and buffers.store enabled" in m
               for m in messages(cfg))
    cfg = multi_t(base)
    cfg['buffers']['store']['enabled'] = False
    assert any("temperatures needs temperature_conditioning true" in m for m in messages(cfg))


def test_temperatures_on_a_stage_that_reads_no_reward_is_refused(base):
    cfg = multi_t(base)
    _stage(cfg, 'train_prior')['temperatures'] = [2.5]
    assert any("stage 'train_prior': temperatures on a stage that reads no reward" in m for m in messages(cfg))


def test_a_temperature_outside_the_scaler_range_is_refused(base):
    msgs = messages(multi_t(base, [2.5, 100.0]))
    assert any("temperature 100 kT (log10 2.000) is outside energy_config.log_temperature_range" in m for m in msgs)


def test_duty_is_refused_for_the_whole_run(base):
    cfg = multi_t(base)
    cfg['buffers']['store']['replay_weight'] = 'duty'
    assert any("buffers.store.replay_weight 'duty' with stage(s) ['tb'] setting temperatures" in m
               for m in messages(cfg))


@pytest.mark.parametrize('mode,source', [('bwd', 'persistent'), ('replay', 'persistent'), ('fwd', 'batch_root')])
def test_a_tracker_or_batch_root_z_source_is_refused(base, mode, source):
    cfg = multi_t(base)
    block = _stage(cfg, 'tb')['loss_coeffs'][mode]
    block['tb_z_source'] = source
    block['tb'] = 1.0
    assert any(f"{mode} tb with tb_z_source {source!r} in a stage setting temperatures" in m for m in messages(cfg))


def test_the_base_block_z_source_is_read_when_the_stage_sets_none(base):
    cfg = multi_t(base)
    del _stage(cfg, 'tb')['loss_coeffs']['bwd']['tb_z_source']
    cfg['bwd_loss_coeffs']['tb_z_source'] = 'persistent'
    assert any("bwd tb with tb_z_source 'persistent'" in m for m in ci.stage_temperatures_problems(
        cfg, ci.row_store_settings(cfg)))


def test_emp_z_persistent_is_refused(base):
    cfg = multi_t(base)
    _stage(cfg, 'tb')['loss_coeffs']['fwd']['emp_z_persistent'] = 1.0
    assert any("fwd emp_z_persistent > 0 in a stage setting temperatures" in m for m in messages(cfg))


@pytest.mark.parametrize('mode', ['bwd', 'replay'])
def test_level_gap_is_refused(base, mode):
    """The level_gap coefficient is the tracker's J_B(c) - J_F(c), measured on run-temperature rows only."""
    cfg = multi_t(base)
    _stage(cfg, 'tb')['loss_coeffs'][mode]['level_gap'] = 0.5
    assert any(f"{mode} level_gap > 0 in a stage setting temperatures" in m for m in messages(cfg))
    del _stage(cfg, 'tb')['temperatures']
    assert not [m for m in messages(cfg) if 'level_gap' in m]       # legal on the store route at one temperature


@pytest.mark.parametrize('metric', ['gates/delta_worst', 'gates/delta_mean'])
@pytest.mark.parametrize('where', ['exit', 'balance'])
def test_a_tracker_gate_in_a_stage_leaving_out_the_run_temperature_is_refused(base, metric, where):
    """No row at energy_config.temperature feeds the tracker's level streams, so the gate never moves in that stage."""
    def gated(temps):
        cfg = multi_t(base, temps)
        st = _stage(cfg, 'tb')
        if where == 'exit':
            st['exit'] = [{'metric': metric, 'below': 1.0, 'patience': 3}]
        else:
            st['balance'] = {'kind': 'lexicographic', 'rules': [{'metric': metric, 'above': 1.0}]}
        return ci.stage_temperatures_problems(cfg, ci.row_store_settings(cfg))
    kind = 'exit' if where == 'exit' else 'balance rule'
    want = f"stage 'tb': {kind} on {metric} in a stage whose temperatures [25.0, 10.0] leave out "            f"energy_config.temperature 2.5"
    assert any(want in m for m in gated([25.0, 10.0]))
    assert not [m for m in gated([25.0, 2.5]) if metric in m]          # the run's temperature feeds it
    assert not [m for m in gated([25.0, 2.5 * (1 + 1e-9)]) if metric in m]


def test_an_exit_on_another_metric_leaving_out_the_run_temperature_passes(base):
    cfg = multi_t(base, [25.0])
    _stage(cfg, 'tb')['exit'] = [{'metric': 'eval/kl', 'below': 1.0}]
    assert ci.stage_temperatures_problems(cfg, ci.row_store_settings(cfg)) == []


def test_emp_z_needs_two_rows_per_condition_and_temperature(base):
    cfg = multi_t(base)
    _stage(cfg, 'tb')['loss_coeffs']['fwd']['repeats'] = 1.0
    assert any("emp_z groups rows by (condition, temperature) and needs >= 2 rows per group" in m
               for m in messages(cfg))


def test_z_calibration_is_refused(base):
    cfg = multi_t(base)
    _stage(cfg, 'tb').setdefault('flags', {})['z_calibration'] = True
    msgs = ci.stage_temperatures_problems(cfg, ci.row_store_settings(cfg))
    assert any("flags.z_calibration in a stage setting temperatures" in m for m in msgs)


def test_the_harvest_is_refused(base):
    cfg = multi_t(base)
    cfg['harvest']['enabled'] = True
    assert any("harvest.enabled with stage(s) ['tb'] setting temperatures" in m for m in messages(cfg))


def test_no_temperature_refusal_fires_on_a_config_without_temperatures(base):
    """duty, the harvest, a tracker z source, level_gap, a tracker gate and z_calibration are judged as before on a run
    without temperatures."""
    cfg = copy.deepcopy(base)
    cfg['temperature_conditioning'] = True
    cfg['noised_prior']['kick_kappa_per_atom'] = 14.9
    cfg['energy_config']['log_temperature_range'] = [-3.0, 1.5]
    cfg['buffers']['store']['replay_weight'] = 'duty'
    _stage(cfg, 'tb')['loss_coeffs']['bwd']['tb_z_source'] = 'persistent'
    _stage(cfg, 'tb')['loss_coeffs']['fwd']['repeats'] = 1.0
    _stage(cfg, 'tb')['loss_coeffs']['bwd']['level_gap'] = 0.5
    _stage(cfg, 'tb')['exit'] = [{'metric': 'gates/delta_worst', 'below': 1.0}]
    assert ci.stage_temperatures_problems(cfg, ci.row_store_settings(cfg)) == []
    assert not [m for m in messages(cfg) if 'temperatures' in m]
