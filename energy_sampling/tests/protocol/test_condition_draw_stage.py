"""Stage parse of the aligned per-condition draw block, `condition_draw`
(protocol.Stage._parse_condition_draw).

The parse is the only check that runs on EVERY protocol in the library, so what a
stage declares is refused here; the prioritised replay draw, an inherited
condition_block_m and the space-group count are judged on the resolved config by
config_invariants.condition_draw_problems (tests/config/test_condition_draw_invariants.py).
"""
import copy
import pathlib

import pytest
import yaml

from energy_sampling.protocol import Stage

MK_DEV = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'
BOOST = {'replay': 0.5, 'bwd': 0.5}
CD = {'conditions': 0, 'replay_rows': 2, 'prior_rows': 2, 'pick': 'uniform'}


def _s7(**kw):
    """The intended conditional stage: replay in fwd's seat, aligned draw on."""
    spec = {'name': 'var_conditioning', 'train_mode': 'fused', 'bwd_sampling_mode': 'prior',
            'fracs': {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}, 'deactivate_threshold': 0.01,
            'fwd_rollout_every': 20, 'fwd_z_sidecar': True, 'replay_warmup_rows': 20000,
            'condition_draw': dict(CD),
            'loss_coeffs': {'fwd': {'tb': 0.0, 'vg_by_condition': 1.0, 'emp_z': 1.0,
                                    'freeze_policy': 1.0, 'repeats': 2.0, 'pooled_vg': 1.0,
                                    'pooled_source': 'replay'},
                            'bwd': {'condition_block_m': 0, 'repeats': 1.0},
                            'replay': {'tb': 0.0, 'vg_by_condition': 1.0,
                                       'condition_block_m': 0}},
            'balance': {'kind': 'lexicographic', 'default_boost': dict(BOOST),
                        'rules': [{'metric': 'pooled/pooled_vg', 'relative': 'best',
                                   'margin': 1.25, 'drift': 0.01, 'if_missing': 'violated',
                                   'boost': dict(BOOST)}]}}
    spec.update(kw)
    return spec


def _cd(**kw):
    spec = _s7()
    spec['condition_draw'].update(kw)
    for k, v in list(kw.items()):
        if v is None:
            del spec['condition_draw'][k]
    return spec


def _refused(spec, match):
    with pytest.raises(ValueError, match=match):
        Stage(spec, 1)


# ------------------------------------------------------------------ defaults

def test_absent_or_null_is_off():
    assert Stage({'name': 'x', 'train_mode': 'fused'}, 0).condition_draw is None
    assert Stage({'name': 'x', 'train_mode': 'fused', 'condition_draw': None}, 0) \
        .condition_draw is None


def test_every_shipped_stage_parses_and_only_var_conditioning_draws_aligned():
    cfg = yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))
    for pname, entry in cfg['protocols'].items():
        for i, s in enumerate(entry['stages']):
            st = Stage(s, i)
            want = CD if (pname, st.name) == ('conditional_vargrad', 'var_conditioning') else None
            assert st.condition_draw == want, (pname, st.name)


def test_the_intended_shape_parses():
    assert Stage(_s7(), 1).condition_draw == CD


def test_pick_defaults_to_uniform_and_accepts_weighted():
    assert Stage(_cd(pick=None), 1).condition_draw['pick'] == 'uniform'
    assert Stage(_cd(pick='weighted'), 1).condition_draw['pick'] == 'weighted'


# ------------------------------------------------------------ the block itself

def test_a_partial_block_is_refused_not_read_as_off():
    _refused(_cd(conditions=None), 'missing')
    _refused(_s7(condition_draw={}), 'missing')


def test_a_non_mapping_is_refused():
    _refused(_s7(condition_draw=2), 'mapping')


def test_an_unknown_key_is_refused():
    _refused(_cd(block_m=2), 'unknown keys')


@pytest.mark.parametrize('key', ['replay_rows', 'prior_rows'])
@pytest.mark.parametrize('bad', [1, 0, 2.0, True, '2'])
def test_rows_must_be_an_integer_of_at_least_two(key, bad):
    """One row per condition is a singleton group: no grouped signal at all."""
    _refused(_cd(**{key: bad}), key)


def test_conditions_must_be_a_non_negative_integer():
    _refused(_cd(conditions=-1), 'conditions')
    _refused(_cd(conditions=1.5), 'conditions')
    assert Stage(_cd(conditions=300), 1).condition_draw['conditions'] == 300


def test_an_unknown_pick_is_refused():
    _refused(_cd(pick='stratified'), 'pick')


# ------------------------------------------------------------ the stage around it

def test_it_needs_a_fused_stage_drawing_bwd_from_the_prior_buffer():
    base = {'name': 'x', 'fracs': {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5},
            'condition_draw': dict(CD)}
    Stage(dict(base, train_mode='fused'), 0)
    _refused(dict(base, train_mode='bwd'), "train_mode 'fused'")
    _refused(dict(base, train_mode='fused', bwd_sampling_mode='dataset'),
             "bwd_sampling_mode 'prior'")


def test_it_is_refused_where_replay_can_never_carry_weight():
    spec = {'name': 'x', 'train_mode': 'fused',
            'fracs': {'fwd': 0.5, 'bwd': 0.5, 'replay': 0.0}, 'condition_draw': dict(CD)}
    _refused(spec, 'replay branch can never carry weight')


def test_the_loss_weighted_bwd_draw_is_refused_beside_it():
    spec = _s7(flags={'weighted_bwd_sampling': True})
    _refused(spec, 'weighted_bwd_sampling')
    Stage(_s7(flags={'weighted_bwd_sampling': False}), 1)


@pytest.mark.parametrize('mode', ['bwd', 'replay'])
def test_a_stage_block_size_on_either_branch_is_refused(mode):
    spec = _s7()
    spec['loss_coeffs'][mode]['condition_block_m'] = 2
    _refused(spec, f'{mode}.condition_block_m')
    spec['loss_coeffs'][mode]['condition_block_m'] = 'two'
    _refused(spec, f'{mode}.condition_block_m')
    spec['loss_coeffs'][mode]['condition_block_m'] = 0.0
    Stage(spec, 1)
