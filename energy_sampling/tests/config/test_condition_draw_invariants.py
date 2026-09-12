"""The two condition draws on the RESOLVED config: config_invariants.
condition_draw_problems / rollout_condition_draw_problems, their rule
(condition_draw_is_well_formed), vargrad_needs_groups' third bwd group source, and
the trainer's refusal in Modeller.set_loss_coeffs.

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
RULE = 'condition_draw_is_well_formed'
BOOST = {'replay': 0.5, 'bwd': 0.5}
CD = {'conditions': 0, 'replay_rows': 2, 'prior_rows': 2, 'pick': 'uniform'}


def _mk_dev():
    return yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))


def _s7_cfg(prioritise=False, draw=True):
    """mk_dev on the conditional route, var_conditioning in the replay-seat shape
    with the aligned draw on and the route's global values."""
    cfg = _mk_dev()
    cfg['protocol'] = 'conditional_vargrad'
    st = cfg['protocols']['conditional_vargrad']['stages'][1]
    st.update(fracs={'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}, fwd_rollout_every=20,
              z_pin_rollout_every=0, fwd_z_sidecar=True, replay_warmup_rows=20000)
    st['loss_coeffs']['fwd'] = {'tb': 0.0, 'vg_lb': 0.0, 'vg_by_condition': 1.0,
                                'emp_z': 1.0, 'freeze_policy': 1.0, 'repeats': 2.0,
                                'pooled_vg': 1.0, 'pooled_source': 'replay',
                                'tb_z_source': 'persistent'}
    st['loss_coeffs']['bwd'] = {'tb': 0.0, 'vg_lb': 0.0, 'condition_block_m': 0,
                                'repeats': 1.0, 'level_gap': 0.0, 'tb_z_source': 'persistent'}
    st['loss_coeffs']['replay'] = {'tb': 0.0, 'vg_by_condition': 1.0, 'condition_block_m': 0,
                                   'tb_z_source': 'persistent'}
    st['flags']['weighted_bwd_sampling'] = False
    st['flags']['weighted_condition_sampling'] = False
    st['balance']['default_boost'] = dict(BOOST)
    for r in st['balance']['rules']:
        r['boost'] = dict(BOOST)
    if draw:
        st['condition_draw'] = dict(CD)
    else:
        st.pop('condition_draw', None)    # mk_dev's stage ships the block
        # pooled_source 'replay' is refused without it; the fwd source keeps the
        # pooled term on (so bwd still needs groups) without that refusal
        st['loss_coeffs']['fwd']['pooled_source'] = 'fwd'
    cfg['buffers']['replay_buffer']['prioritise']['enabled'] = prioritise
    cfg['buffers']['replay_buffer']['max_size'] = 150000
    cfg['condition_log_z']['rollout_condition_draw'] = 'cycle'
    return cfg


def _stage(cfg):
    return cfg['protocols']['conditional_vargrad']['stages'][1]


def _msgs(cfg, rule=RULE):
    return [v.detail for v in ci.check(cfg) if v.rule == rule]


# ------------------------------------------------------------------ clean shapes

def test_the_shipped_config_is_clean():
    assert _msgs(_mk_dev()) == []


def test_the_intended_shape_is_clean():
    cfg = _s7_cfg()
    assert _msgs(cfg) == []
    assert _msgs(cfg, 'vargrad_needs_groups') == []


# ------------------------------------------------------------ condition_draw

def test_the_prioritised_replay_draw_is_refused():
    assert any('condition_draw with buffers.replay_buffer.prioritise' in m
               for m in _msgs(_s7_cfg(prioritise=True)))


def test_a_block_size_inherited_from_the_base_is_refused():
    """mk_dev's base bwd condition_block_m is 2: a stage that only omits the key
    runs the blocked draw's setting beside the aligned one."""
    cfg = _s7_cfg()
    del _stage(cfg)['loss_coeffs']['bwd']['condition_block_m']
    assert any('effective bwd condition_block_m' in m for m in _msgs(cfg))
    cfg = _s7_cfg()
    del _stage(cfg)['loss_coeffs']['replay']['condition_block_m']
    cfg['replay_loss_coeffs']['condition_block_m'] = 2
    assert any('effective replay condition_block_m' in m for m in _msgs(cfg))


def test_more_than_one_space_group_is_refused():
    cfg = _s7_cfg()
    cfg['space_groups'] = [2, 14]
    assert any('space-group' in m for m in _msgs(cfg))


def test_the_block_is_a_bwd_group_source():
    """bwd repeats 1 and condition_block_m 0: the aligned draw's prior_rows is
    what gives the backward branch its groups."""
    assert _msgs(_s7_cfg(), 'vargrad_needs_groups') == []
    assert any('bwd VarGrad' in m for m in _msgs(_s7_cfg(draw=False), 'vargrad_needs_groups'))


# ------------------------------------------------------------ rollout_condition_draw

def test_an_unknown_rollout_draw_is_refused():
    cfg = _mk_dev()
    cfg['condition_log_z']['rollout_condition_draw'] = 'stratified'
    assert any('rollout_condition_draw' in m for m in _msgs(cfg))


def test_a_non_iid_rollout_draw_beside_the_weighted_flag_is_refused():
    cfg = _s7_cfg()
    _stage(cfg)['flags']['weighted_condition_sampling'] = True
    assert any('weighted_condition_sampling' in m for m in _msgs(cfg))
    cfg['condition_log_z']['rollout_condition_draw'] = 'iid'
    assert _msgs(cfg) == []


@pytest.mark.parametrize('mode,power,ok', [
    ('under_drawn', 2.0, True), ('under_drawn', 0.0, False), ('under_drawn', -1.0, False),
    ('under_drawn', 'x', False), ('under_drawn', float('inf'), False),
    ('iid', 1.0, True), ('iid', 2.0, False), ('cycle', 0.5, False)])
def test_the_under_drawn_power(mode, power, ok):
    cfg = _mk_dev()
    cfg['condition_log_z']['rollout_condition_draw'] = mode
    cfg['condition_log_z']['rollout_under_drawn_power'] = power
    assert (_msgs(cfg) == []) is ok, _msgs(cfg)


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
    """The live stage is train_prior; the bad one is the stage after it."""
    cfg = _s7_cfg()
    del _stage(cfg)['loss_coeffs']['bwd']['condition_block_m']
    with pytest.raises(ValueError, match='condition_block_m'):
        _modeller(cfg).set_loss_coeffs()


def test_set_loss_coeffs_raises_on_a_bad_rollout_draw():
    cfg = _mk_dev()
    cfg['condition_log_z']['rollout_condition_draw'] = 'stratified'
    with pytest.raises(ValueError, match='rollout_condition_draw'):
        _modeller(cfg).set_loss_coeffs()


def test_set_loss_coeffs_is_silent_on_the_intended_shape_and_the_shipped_route():
    _modeller(_s7_cfg()).set_loss_coeffs()
    _modeller(_mk_dev()).set_loss_coeffs()
