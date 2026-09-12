"""The replay-seat keys on the RESOLVED config: config_invariants.replay_seat_problems,
its rule (replay_seat_is_well_formed), the fill waiver for sidecar stages, and the
trainer's refusal in Modeller.set_loss_coeffs.

config_invariants only REPORTS at load (utils._report_config_invariants), so the
raise in set_loss_coeffs is the refusal; the rule is its audit-path twin and the
two share one implementation over getters.
"""
import copy
import pathlib
from types import MethodType, SimpleNamespace

import pytest
import yaml

from energy_sampling import config_invariants as ci

MK_DEV = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'
RULE = 'replay_seat_is_well_formed'
BOOST = {'replay': 0.5, 'bwd': 0.5}


def _mk_dev():
    return yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))


def _s7_cfg(prioritise=False, max_size=150000):
    """mk_dev on the conditional route, var_conditioning in the replay-seat shape
    with the route's global values."""
    cfg = _mk_dev()
    cfg['protocol'] = 'conditional_vargrad'
    st = cfg['protocols']['conditional_vargrad']['stages'][1]
    st.update(fracs={'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}, fwd_rollout_every=20,
              z_pin_rollout_every=0, fwd_z_sidecar=True, replay_warmup_rows=20000)
    st['loss_coeffs']['fwd'] = {'tb': 0.0, 'vg_lb': 0.0, 'vg_by_condition': 1.0,
                                'emp_z': 1.0, 'freeze_policy': 1.0, 'repeats': 2.0,
                                'pooled_vg': 1.0, 'pooled_source': 'replay',
                                'tb_z_source': 'persistent'}
    st['balance']['default_boost'] = dict(BOOST)
    for r in st['balance']['rules']:
        r['boost'] = dict(BOOST)
    cfg['buffers']['replay_buffer']['prioritise']['enabled'] = prioritise
    cfg['buffers']['replay_buffer']['max_size'] = max_size
    return cfg


def _stage(cfg):
    return cfg['protocols']['conditional_vargrad']['stages'][1]


def _msgs(cfg, rule=RULE):
    return [v.detail for v in ci.check(cfg) if v.rule == rule]


def test_the_shipped_config_is_clean():
    assert _msgs(_mk_dev()) == []


def test_the_intended_shape_is_clean():
    assert _msgs(_s7_cfg()) == []


def test_prioritised_replay_is_refused_under_the_replay_source():
    """The pooled term takes replay rows unweighted: a prioritised draw would
    hand it rows from p with no IS correction."""
    assert any('prioritise' in m for m in _msgs(_s7_cfg(prioritise=True)))


def test_the_source_with_the_term_off_in_the_base_is_a_dead_key():
    cfg = _s7_cfg()
    del _stage(cfg)['loss_coeffs']['fwd']['pooled_vg']        # base holds 0
    assert any('dead key' in m for m in _msgs(cfg))


def test_a_bad_base_source_is_refused_on_every_stage():
    cfg = _mk_dev()
    cfg['fwd_loss_coeffs']['pooled_source'] = 'bwd'
    assert len(_msgs(cfg)) == len(ci.active_stages(cfg))


def test_the_sidecar_needs_forward_groups_from_the_resolved_block():
    cfg = _s7_cfg()
    del _stage(cfg)['loss_coeffs']['fwd']['repeats']           # base repeats 1
    assert any('repeats=1' in m for m in _msgs(cfg))
    cfg = _s7_cfg()
    _stage(cfg)['loss_coeffs']['fwd']['vg_by_condition'] = 0.0  # no grouped estimate
    assert any('NO grouped estimate' in m for m in _msgs(cfg))


def test_a_global_frac_floor_at_the_threshold_is_refused():
    """The global floor is the FALLBACK: it binds only where the stage names no
    min_fracs.fwd. mk_dev's stage names 0, which shields it."""
    cfg = _s7_cfg()
    cfg['controller']['min_mode_frac'] = 0.02
    assert _stage(cfg)['min_fracs'] == {'fwd': 0.0}
    assert not any('frac floor' in m for m in _msgs(cfg))
    del _stage(cfg)['min_fracs']
    assert any('frac floor' in m for m in _msgs(cfg))


def test_a_warmup_the_buffer_cannot_hold_is_refused():
    assert any('max_size' in m for m in _msgs(_s7_cfg(max_size=20000)))


def test_an_inherited_replay_source_without_the_aligned_draw_is_refused():
    """The Stage parse sees only what a stage declares; a base 'replay' reaches
    var_conditioning through the resolved block, and without condition_draw
    nothing aligns the pair's conditions."""
    cfg = _s7_cfg()
    del _stage(cfg)['loss_coeffs']['fwd']['pooled_source']
    del _stage(cfg)['condition_draw']
    cfg['fwd_loss_coeffs']['pooled_source'] = 'replay'
    assert any("'var_conditioning'" in m and 'without condition_draw' in m
               for m in _msgs(cfg))
    with pytest.raises(ValueError, match='without condition_draw'):
        _modeller(cfg).set_loss_coeffs()
    assert not any('without condition_draw' in m for m in _msgs(_s7_cfg()))


# ------------------------------------------------------------ the fill waiver

def _fill_msgs(cfg):
    return [m for m in _msgs(cfg, 'fwd_rollout_cadence_is_well_formed')
            if 'fill_threshold' in m]


def test_a_sidecar_stage_waives_the_fill():
    cfg = _s7_cfg()
    cfg['z_calibration']['fill_threshold'] = 0
    assert _fill_msgs(cfg) == []


def test_a_cadenced_stage_without_the_sidecar_still_needs_the_fill():
    cfg = _s7_cfg()
    cfg['z_calibration']['fill_threshold'] = 0
    _stage(cfg)['fwd_z_sidecar'] = False
    assert _fill_msgs(cfg)


def test_the_waiver_does_not_waive_the_servo_refusal():
    cfg = _s7_cfg()
    _stage(cfg)['flags']['z_calibration'] = True
    assert any('z_calibration' in m for m in
               _msgs(cfg, 'fwd_rollout_cadence_is_well_formed'))


# ------------------------------------------------------------ the trainer's raise

def _modeller(cfg):
    from energy_sampling.protocol import StageProtocol
    from energy_sampling.train import Modeller
    from energy_sampling.utils import dict2namespace
    m = SimpleNamespace(args=dict2namespace(copy.deepcopy(cfg)), gfn_model=SimpleNamespace(),
                        stage=None, _warn_if_z_untrained=lambda: None)
    m.protocol = StageProtocol(m)
    m.set_loss_coeffs = MethodType(Modeller.set_loss_coeffs, m)
    return m


def test_set_loss_coeffs_raises_at_step_0_on_a_later_stage():
    """The live stage is train_prior; the bad one is the stage after it. The run
    must die before its first step, not hours in at the transition."""
    m = _modeller(_s7_cfg(prioritise=True))
    with pytest.raises(ValueError, match='prioritise'):
        m.set_loss_coeffs()


def test_set_loss_coeffs_is_silent_on_the_intended_shape_and_the_shipped_route():
    _modeller(_s7_cfg()).set_loss_coeffs()
    _modeller(_mk_dev()).set_loss_coeffs()
