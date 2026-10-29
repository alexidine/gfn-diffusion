"""condition_log_z.z_feed / eval_half_life_visits on the raw config: config_invariants.z_feed_is_well_formed, which
Modeller.init_condition_log_z raises on. What each test would catch:

  * mk_dev carrying a value the rule refuses, or not carrying the keys at all;
  * an absent key judged as anything but the code default;
  * a malformed value passing.

The rule's copy of the modes and defaults is pinned to buffer.py's in tests/crystal/test_z_feed.py (torch).
"""
import copy
import pathlib

import pytest
import yaml

from energy_sampling import config_invariants as ci

pytestmark = pytest.mark.fast

MK_DEV = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'
RULE = 'z_feed_is_well_formed'


@pytest.fixture(scope='module')
def mk_dev():
    return yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))


def _msgs(cfg):
    return [v.detail for v in ci.check(cfg) if v.rule == RULE]


def test_mk_dev_takes_the_forward_feed_and_is_clean(mk_dev):
    assert mk_dev['condition_log_z']['z_feed'] == 'forward'
    assert 'eval_half_life_visits' in mk_dev['condition_log_z']
    assert _msgs(mk_dev) == []


def test_absent_keys_are_the_code_defaults(mk_dev):
    cfg = copy.deepcopy(mk_dev)
    del cfg['condition_log_z']['z_feed'], cfg['condition_log_z']['eval_half_life_visits']
    assert _msgs(cfg) == []


@pytest.mark.parametrize('key,value', [('z_feed', 'fwd'), ('z_feed', None), ('z_feed', True),
                                       ('eval_half_life_visits', 0), ('eval_half_life_visits', -1.0),
                                       ('eval_half_life_visits', float('nan')), ('eval_half_life_visits', float('inf')),
                                       ('eval_half_life_visits', True), ('eval_half_life_visits', '2')])
def test_malformed_values_are_refused(mk_dev, key, value):
    cfg = copy.deepcopy(mk_dev)
    cfg['condition_log_z'][key] = value
    assert len(_msgs(cfg)) == 1
