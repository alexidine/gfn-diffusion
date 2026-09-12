"""Stage parse of the three replay-seat keys: loss_coeffs.fwd.pooled_source,
fwd_z_sidecar and replay_warmup_rows (protocol.Stage._parse_replay_seat).

The parse is the only check that runs on EVERY protocol in the library, so what
a stage declares is refused here; what depends on the base blocks or global keys
is config_invariants.replay_seat_problems (tests/config/test_replay_seat_invariants.py).
"""
import copy
import pathlib

import pytest
import yaml

from energy_sampling.protocol import Stage

MK_DEV = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'
BOOST = {'replay': 0.5, 'bwd': 0.5}


def _s7(**kw):
    """The intended conditional stage: replay in fwd's seat."""
    spec = {'name': 'var_conditioning', 'train_mode': 'fused', 'bwd_sampling_mode': 'prior',
            'fracs': {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}, 'deactivate_threshold': 0.01,
            'fwd_rollout_every': 20, 'fwd_z_sidecar': True, 'replay_warmup_rows': 20000,
            'condition_draw': {'conditions': 0, 'replay_rows': 2, 'prior_rows': 2},
            'loss_coeffs': {'fwd': {'tb': 0.0, 'vg_by_condition': 1.0, 'emp_z': 1.0,
                                    'freeze_policy': 1.0, 'repeats': 2.0, 'pooled_vg': 1.0,
                                    'pooled_source': 'replay'}},
            'balance': {'kind': 'lexicographic', 'default_boost': dict(BOOST),
                        'rules': [{'metric': 'pooled/pooled_vg', 'relative': 'best',
                                   'margin': 1.25, 'drift': 0.01, 'if_missing': 'violated',
                                   'boost': dict(BOOST)}]}}
    spec.update(kw)
    return spec


def _fwd(spec, **kw):
    spec = copy.deepcopy(spec)
    spec['loss_coeffs']['fwd'].update(kw)
    for k, v in list(kw.items()):
        if v is None:
            del spec['loss_coeffs']['fwd'][k]
    return spec


def _mk_dev_stage(protocol, name):
    cfg = yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))
    for i, s in enumerate(cfg['protocols'][protocol]['stages']):
        if s['name'] == name:
            return copy.deepcopy(s), i
    raise AssertionError(name)


# ------------------------------------------------------------------ defaults

def test_absent_keys_are_off():
    st = Stage({'name': 'x', 'train_mode': 'fused'}, 0)
    assert st.fwd_z_sidecar is False and st.replay_warmup_rows == 0


def test_every_shipped_stage_parses_and_only_var_conditioning_turns_the_keys_on():
    cfg = yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))
    for pname, entry in cfg['protocols'].items():
        for i, s in enumerate(entry['stages']):
            st = Stage(s, i)
            if (pname, st.name) == ('conditional_vargrad', 'var_conditioning'):
                assert st.fwd_z_sidecar is True and st.replay_warmup_rows == 20000
                assert st.loss_coeffs['fwd']['pooled_source'] == 'replay'
            else:
                assert st.fwd_z_sidecar is False and st.replay_warmup_rows == 0


def test_the_intended_shape_parses():
    st = Stage(_s7(), 1)
    assert st.fwd_z_sidecar and st.replay_warmup_rows == 20000
    assert st.replay_trains and not st.balance_can_raise('fwd', 0.01)


# ------------------------------------------------------------ pooled_source

def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError, match='pooled_source'):
        Stage(_fwd(_s7(), pooled_source='bwd'), 1)


@pytest.mark.parametrize('mode', ['bwd', 'replay'])
def test_the_source_on_a_backward_block_is_refused(mode):
    spec = _s7()
    spec['loss_coeffs'][mode] = {'pooled_source': 'replay'}
    with pytest.raises(ValueError, match='fwd block'):
        Stage(spec, 1)


def _fwd_bwd_only(spec):
    """var_conditioning's shape before the replay seat: every boost fwd/bwd and
    replay entering at 0, so replay never leaves the floor."""
    spec['fracs'] = {'fwd': 0.5, 'bwd': 0.5, 'replay': 0.0}
    spec['balance']['default_boost'] = {'fwd': 0.5, 'bwd': 0.5}
    for r in spec['balance']['rules']:
        r['boost'] = {'fwd': 0.5, 'bwd': 0.5}
    return spec


def test_replay_source_where_replay_cannot_train_is_refused():
    spec = _fwd_bwd_only(_s7(fwd_z_sidecar=False, replay_warmup_rows=0))
    assert spec['loss_coeffs']['fwd']['pooled_source'] == 'replay'
    with pytest.raises(ValueError, match='never carry weight'):
        Stage(spec, 1)


def test_replay_source_with_the_term_declared_off_is_a_dead_key():
    with pytest.raises(ValueError, match='dead key'):
        Stage(_fwd(_s7(), pooled_vg=0.0), 1)


def test_the_default_source_is_accepted_anywhere():
    spec, i = _mk_dev_stage('conditional_vargrad', 'var_conditioning')
    spec['loss_coeffs']['fwd']['pooled_source'] = 'fwd'
    assert Stage(spec, i)


def test_replay_source_without_the_aligned_draw_is_refused():
    """bwd is aligned to the forward batch only under the fwd source, and replay
    runs after bwd: without condition_draw the pooled pair shares a condition
    only by chance collision. Absent and null are both off."""
    spec = _s7()
    del spec['condition_draw']
    with pytest.raises(ValueError, match='without condition_draw'):
        Stage(spec, 1)
    with pytest.raises(ValueError, match='without condition_draw'):
        Stage(_s7(condition_draw=None), 1)


def test_the_fwd_source_needs_no_aligned_draw():
    spec = _fwd(_s7(fwd_z_sidecar=False, replay_warmup_rows=0), pooled_source='fwd')
    del spec['condition_draw']
    assert Stage(spec, 1).condition_draw is None


# ------------------------------------------------------------ fwd_z_sidecar

@pytest.mark.parametrize('change,match', [
    (dict(fwd_z_sidecar='yes'), 'true or false'),
    (dict(fwd_rollout_every=0, replay_warmup_rows=0), 'fwd_rollout_every > 0'),
    (dict(deactivate_threshold=None), 'declare deactivate_threshold'),
    (dict(fracs=None), 'declare fracs'),
    (dict(fracs={'fwd': 0.05, 'bwd': 0.5, 'replay': 0.45}), 'entry fwd share'),
    (dict(min_fracs={'fwd': 0.02}), 'min_fracs.fwd'),
])
def test_stage_shape_refusals(change, match):
    spec = _s7()
    for k, v in change.items():
        if v is None:
            spec.pop(k)
        else:
            spec[k] = v
    with pytest.raises(ValueError, match=match):
        Stage(spec, 1)


@pytest.mark.parametrize('change,match', [
    (dict(freeze_policy=None), 'freeze_policy'),
    (dict(freeze_policy=0.0), 'freeze_policy'),
    (dict(emp_z=None), 'emp_z'),
    (dict(emp_z=0.0), 'emp_z'),
    (dict(repeats=1.0), 'repeats'),
])
def test_forward_coefficient_refusals(change, match):
    with pytest.raises(ValueError, match=match):
        Stage(_fwd(_s7(), **change), 1)


def test_a_balance_that_boosts_fwd_is_refused():
    spec = _s7()
    spec['balance']['default_boost'] = {'fwd': 0.1, 'bwd': 0.45, 'replay': 0.45}
    with pytest.raises(ValueError, match='raise fwd'):
        Stage(spec, 1)


def test_a_split_that_names_fwd_is_refused():
    spec = _s7()
    spec['balance'] = {'kind': 'proportional', 'metrics': {'fwd': 'fwd/a', 'replay': 'replay/b'},
                       'pinned': {'bwd': 0.5}, 'drive': 'relative',
                       'targets': {'fwd': 1.0, 'replay': 1.0},
                       'default_boost': {'fwd': 0.5, 'replay': 0.5}, 'floor': 0.1,
                       'alpha': 0.01}
    with pytest.raises(ValueError, match='raise fwd'):
        Stage(spec, 1)


def test_a_pin_at_zero_is_not_a_way_to_rise():
    """active_modes counts pinned: {fwd: 0.0} by its key; the sidecar check reads
    it by value."""
    spec, i = _mk_dev_stage('unconditional_tb', 'equilibration')
    st = Stage(spec, i)
    assert 'fwd' in st.active_modes
    assert not st.balance_can_raise('fwd', 0.01)
    assert st.balance_can_raise('replay', 0.01)


# ------------------------------------------------------------ replay_warmup_rows

@pytest.mark.parametrize('rows', [-1, 2.0e4, True, '20000'])
def test_warmup_must_be_a_non_negative_integer(rows):
    with pytest.raises(ValueError, match='replay_warmup_rows'):
        Stage(_s7(replay_warmup_rows=rows), 1)


def test_warmup_needs_a_cadence():
    with pytest.raises(ValueError, match='fwd_rollout_every > 0'):
        Stage(_s7(fwd_z_sidecar=False, fwd_rollout_every=0), 1)


def test_warmup_on_a_stage_that_never_trains_replay_is_refused():
    spec = _fwd_bwd_only(_s7(fwd_z_sidecar=False, replay_warmup_rows=100))
    spec['loss_coeffs']['fwd'].pop('pooled_source')    # isolate the warm-up's own check
    with pytest.raises(ValueError, match='replay_warmup_rows=100 on a stage'):
        Stage(spec, 1)


def test_replay_trains_reads_the_pin_by_value():
    spec = _s7(fwd_z_sidecar=False, replay_warmup_rows=0)
    spec['balance'] = {'kind': 'proportional', 'metrics': {'fwd': 'fwd/a', 'bwd': 'bwd/b'},
                       'pinned': {'replay': 0.0}, 'drive': 'relative',
                       'targets': {'fwd': 1.0, 'bwd': 1.0},
                       'default_boost': {'fwd': 0.5, 'bwd': 0.5}, 'floor': 0.1, 'alpha': 0.01}
    spec['fracs'] = {'fwd': 0.5, 'bwd': 0.5, 'replay': 0.0}
    spec['loss_coeffs']['fwd'].pop('pooled_source')
    spec.pop('condition_draw')     # refused where replay cannot train
    st = Stage(spec, 1)
    assert 'replay' in st.active_modes and not st.replay_trains
