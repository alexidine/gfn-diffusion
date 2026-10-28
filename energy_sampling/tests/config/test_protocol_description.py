"""
config_snapshot.describe_stage / describe_protocol: the prose summary of a
protocol, printed under `--check` for every declared protocol and by
StageProtocol.advance at stage entry (StageProtocol.describe_live).

Run: python -m pytest tests/config/test_protocol_description.py -q
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import config_snapshot as cs
import protocol as P

HERE = Path(__file__).resolve().parents[2]   # tests/<area>/x.py -> energy_sampling/
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'


@pytest.fixture(scope='module')
def canonical():
    snap, issues, args = cs._contract(str(CANONICAL))
    assert args is not None, snap.get('load_error')
    return snap, issues, args


def _protocols(args):
    return cs._to_plain(args.protocols)


def test_every_mk_dev_protocol_names_its_exits_and_branches(canonical):
    """Each declared protocol yields one paragraph per stage. A stage with an exit
    names its exit metric and successor; the terminal stage says it has none; the
    fused stage names every branch."""
    _, _, args = canonical
    protocols = _protocols(args)
    assert len(protocols) >= 2
    for name, proto in protocols.items():
        specs = proto['stages']
        text = cs.describe_protocol(args, name)
        paras = text.split('\n')
        assert len(paras) == len(specs), (name, text)
        for i, (spec, para) in enumerate(zip(specs, paras)):
            assert f"'{spec['name']}'" in para
            if spec.get('exit'):
                for term in spec['exit']:
                    assert term['metric'] in para, para
                assert f"advances to '{specs[i + 1]['name']}'" in para, para
            else:
                assert 'has no exit' in para, para
            if spec.get('train_mode', 'fused') == 'fused':
                for branch in ('fwd', 'bwd', 'replay'):
                    assert f'{branch} ' in para, (branch, para)
            else:
                assert 'bwd alone trains' in para, para


def test_inactive_protocol_is_described_on_its_own_overlay(canonical):
    """The inactive protocol's coefficients come from the trainer's overlay with
    it selected, not from the active protocol's same-named stage."""
    _, _, args = canonical
    inactive = [n for n in _protocols(args) if n != args.protocol]
    assert inactive
    for name in inactive:
        assert 'unavailable' not in cs.describe_protocol(args, name)
    assert 'conditional_vargrad' in _protocols(args)
    text = cs.describe_protocol(args, 'conditional_vargrad')
    assert 'pooled VarGrad term' in text
    assert 'weight-1 sidecar' in text


def test_unphrased_keys_land_verbatim_in_the_tail(canonical):
    _, _, args = canonical
    proto = _protocols(args)[args.protocol]
    spec = proto['stages'][0]
    assert spec.get('hot_lr_sensor')
    para = cs.describe_protocol(args, args.protocol).split('\n')[0]
    assert ' also: ' in para
    assert 'hot_lr_sensor={' in para.split(' also: ', 1)[1]


def _bare(**over):
    spec = {'name': 's1', 'train_mode': 'fused', 'fracs': {'fwd': 0.2, 'bwd': 0.8, 'replay': 0.0}}
    spec.update(over)
    return spec


def test_max_steps_and_stop_phrase_the_end():
    spec = _bare(max_steps=500, on_exit=['snapshot:done', 'stop'],
                 exit=[{'metric': 'gates/progress_done', 'above': 0.5, 'patience': 2}])
    text = cs.describe_stage(spec, 0, None, coeffs={m: {} for m in P.MODES})
    assert 'after 500 of its own steps' in text
    assert 'gates/progress_done > 0.5 for 2' in text
    assert 'runs snapshot:done and ends the run' in text


def test_coeffs_argument_sets_the_trained_parameters():
    coeffs = {'fwd': {'tb': 1.0, 'freeze_policy': 1.0}, 'bwd': {'tb': 1.0, 'freeze_z': 1.0},
              'replay': {}}
    text = cs.describe_stage(_bare(), 1, None, coeffs=coeffs, successor='s2', n_stages=2)
    assert text.startswith("Stage 2/2 's1' (fused)")
    assert 'fwd trains log Z only' in text
    assert 'bwd trains the policy only' in text
    assert 'replay carries no loss weight' in text
    assert 'on entry' not in text.lower() or 'no entry action' in text


def test_global_deactivate_threshold_decides_a_branch_without_a_stage_override():
    """No stage deactivate_threshold: controller.deactivate_threshold decides, as in
    train.py's fused step. fwd's entry share is 0.2."""
    coeffs = {m: {'tb': 1.0} for m in P.MODES}
    low = SimpleNamespace(controller=SimpleNamespace(deactivate_threshold=0.1))
    high = SimpleNamespace(controller=SimpleNamespace(deactivate_threshold=0.3))
    assert 'fwd trains' in cs.describe_stage(_bare(), 0, low, coeffs=coeffs)
    assert 'fwd carries no loss weight' in cs.describe_stage(_bare(), 0, high, coeffs=coeffs)


def test_first_stage_on_enter_is_said_not_to_fire():
    text = cs.describe_stage(_bare(on_enter=['freeze_pb']), 0, None,
                             coeffs={m: {} for m in P.MODES})
    assert 'never fires' in text


def _engine(spec2):
    args = SimpleNamespace(
        protocol='p',
        protocols=SimpleNamespace(p=SimpleNamespace(stages=[_bare(), spec2])),
        fwd_loss_coeffs=SimpleNamespace(tb=1.0, freeze_policy=1.0),
        bwd_loss_coeffs=SimpleNamespace(tb=1.0, freeze_z=1.0),
        replay_loss_coeffs=SimpleNamespace(tb=1.0, mle=0.0, freeze_z=1.0))
    m = SimpleNamespace(args=args, stage=spec2['name'])
    return P.StageProtocol(m)


def test_describe_live_uses_the_live_overlay():
    spec2 = {'name': 's2', 'train_mode': 'fused',
             'fracs': {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5},
             'loss_coeffs': {'replay': {'tb': 0.0, 'mle': 2.0}}}
    text = _engine(spec2).describe_live()
    assert text.startswith("Stage 2/2 's2' (fused)"), text
    assert 'replay trains the policy only on stored forward rollouts in the replay buffer with mle 2' in text
    assert 'fwd carries no loss weight' in text


def test_advance_prints_the_incoming_stage_description(capsys):
    """StageProtocol.advance prints describe_live after the 'engaged' line; the
    transition harness is the one tests/protocol/test_weights_loaded_skip.py keeps."""
    import importlib.util
    path = HERE / 'tests' / 'protocol' / 'test_weights_loaded_skip.py'
    spec = importlib.util.spec_from_file_location('_wl_skip_harness', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p, m, _ = mod.engine([mod.PHASE1, mod.PHASE2], weights_only_loaded=True)
    for mode in P.MODES:
        setattr(m.args, f'{mode}_loss_coeffs', SimpleNamespace(tb=1.0))
    p.begin()
    lines = capsys.readouterr().out.splitlines()
    engaged = [i for i, ln in enumerate(lines) if "stage 'tb_conditioning' engaged" in ln]
    assert engaged, lines
    desc = lines[engaged[0] + 1]
    assert desc.startswith("protocol: Stage 2/2 'tb_conditioning' (fused)"), desc
    assert 'on entry it runs freeze_pb' in desc.lower(), desc
    assert 'FAILED' not in desc


def test_describe_live_reports_failure_instead_of_raising():
    sp = _engine({'name': 's2', 'train_mode': 'fused'})
    sp.m.args.protocols = None
    sp._stages = [P.Stage(_bare(), 0), P.Stage({'name': 's2'}, 1)]
    assert sp.describe_live().startswith('stage description FAILED')


def test_check_prints_every_declared_protocol(canonical, monkeypatch, capsys):
    monkeypatch.setattr(cs, '_contract', lambda path: canonical)
    monkeypatch.setattr(sys, 'argv', ['config_snapshot', str(CANONICAL), '--check'])
    assert cs._main() == 0
    out = capsys.readouterr().out
    _, _, args = canonical
    for name in _protocols(args):
        assert f"protocol '{name}'" in out
    assert f"protocol '{args.protocol}' (active)" in out
    assert 'DESCRIPTION FAILED' not in out
