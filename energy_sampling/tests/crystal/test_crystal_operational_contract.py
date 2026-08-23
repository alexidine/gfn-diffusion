"""Fast proof for the canonical GFN crystal launch surface.

Run from ``energy_sampling/``:

    python -m pytest -q test_crystal_operational_contract.py

This deliberately stops before model construction.  It proves that the fused
``mk_dev.yaml`` operational surface loads, every declared protocol parses, the
selected unconditional route passes the shared cross-field invariants and
resolves through the real ``StageProtocol`` path, and the trainer accepts the
documented ``--config`` invocation without touching GPU, data, checkpoints, or
W&B.
"""

import sys
from pathlib import Path

import pytest

import config_snapshot
import utils

HERE = Path(__file__).resolve().parent
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'


def test_fused_canonical_config_is_a_complete_operational_contract():
    snap, issues = config_snapshot.contract(str(CANONICAL))
    assert issues == [], '\n'.join(issues)

    cfg = snap['config']
    assert cfg['_active_protocol'] == 'unconditional_tb'
    assert {'unconditional_tb', 'conditional_vargrad'} <= set(cfg['protocols'])
    assert [stage['name'] for stage in snap['stages']] == [
        'train_prior', 'equilibration']


def test_documented_launch_uses_the_named_config_contract(capsys):
    args = utils.get_train_args(['--config', str(CANONICAL)])
    capsys.readouterr()  # derived-value summary is run provenance, not test output
    assert args.protocol == 'unconditional_tb'
    assert args.run_name == 'mk_dev'


def test_ambiguous_bare_config_path_is_rejected():
    with pytest.raises(SystemExit):
        utils.get_train_args([str(CANONICAL)])


def test_check_command_reports_the_resolved_route(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, 'argv', ['config_snapshot', str(CANONICAL), '--check'])
    assert config_snapshot._main() == 0
    out = capsys.readouterr().out
    assert 'contract ok' in out
    assert 'active protocol: unconditional_tb' in out
    assert 'declared protocols: conditional_vargrad, unconditional_tb' in out
    assert 'active stages: train_prior -> equilibration' in out
    assert 'runtime services touched: none' in out
