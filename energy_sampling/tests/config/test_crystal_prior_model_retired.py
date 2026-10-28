"""The frozen prior MODEL is retired off the conformer route.

config_invariants.crystal_prior_model_is_retired refuses each spelling that would ask for
one: a non-null prior_model_name, a prior buffer whose source is not 'anchors' (absent =
the code default 'prior_model'), `skip_if: prior_loaded`, and a `snapshot_prior` action.
mk_dev, which carries none of them, is silent; the conformer route keeps its own rules.

    python -m pytest -q tests/config/test_crystal_prior_model_retired.py
"""
import copy
from pathlib import Path

import pytest
import yaml

from config_invariants import ERROR, check

pytestmark = pytest.mark.fast

RULE = 'crystal_prior_model_is_retired'
MK_DEV = Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'


@pytest.fixture(scope='module')
def mk_dev():
    return yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))


def _fired(cfg):
    return [v for v in check(cfg) if v.rule == RULE]


def _stage(cfg, protocol, name):
    return next(s for s in cfg['protocols'][protocol]['stages'] if s['name'] == name)


def test_mk_dev_is_silent(mk_dev):
    assert 'prior_model_name' not in mk_dev
    assert _fired(mk_dev) == []


def test_a_named_prior_model_is_an_error(mk_dev):
    cfg = copy.deepcopy(mk_dev)
    cfg['prior_model_name'] = 'x_prior.pt'
    (v,) = _fired(cfg)
    assert v.severity == ERROR and 'prior_model_name' in v.detail
    cfg['prior_model_name'] = None
    assert _fired(cfg) == []


@pytest.mark.parametrize('source', ['prior_model', None])
def test_a_prior_buffer_source_other_than_anchors_is_an_error(mk_dev, source):
    cfg = copy.deepcopy(mk_dev)
    if source is None:
        del cfg['buffers']['prior_buffer']['source']      # absent = code default 'prior_model'
    else:
        cfg['buffers']['prior_buffer']['source'] = source
    (v,) = _fired(cfg)
    assert v.severity == ERROR and 'prior_buffer.source' in v.detail


def test_skip_if_prior_loaded_is_an_error(mk_dev):
    cfg = copy.deepcopy(mk_dev)
    _stage(cfg, cfg['protocol'], 'train_prior')['skip_if'] = 'prior_loaded'
    (v,) = _fired(cfg)
    assert v.severity == ERROR and 'prior_loaded' in v.detail


@pytest.mark.parametrize('where', ['on_exit', 'on_enter'])
def test_snapshot_prior_is_an_error(mk_dev, where):
    cfg = copy.deepcopy(mk_dev)
    _stage(cfg, cfg['protocol'], 'train_prior').setdefault(where, []).append('snapshot_prior')
    (v,) = _fired(cfg)
    assert v.severity == ERROR and 'snapshot_prior' in v.detail


def test_inactive_protocols_are_not_judged(mk_dev):
    """Only the selected protocol's stages are read, like every stage rule."""
    cfg = copy.deepcopy(mk_dev)
    other = next(p for p in cfg['protocols'] if p != cfg['protocol'])
    _stage(cfg, other, 'train_prior')['skip_if'] = 'prior_loaded'
    assert _fired(cfg) == []


def test_the_conformer_route_is_left_to_its_own_rules(mk_dev):
    cfg = copy.deepcopy(mk_dev)
    cfg['energy_function'] = 'conformer_torsions'
    cfg['prior_model_name'] = 'x_prior.pt'
    cfg['buffers']['prior_buffer']['source'] = 'prior_model'
    assert _fired(cfg) == []
