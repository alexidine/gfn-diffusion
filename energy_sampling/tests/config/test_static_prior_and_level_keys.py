"""Three refusals and one switch added with the phase-2 level battery (2026-11-01).

- buffers.prior_buffer.static makes Modeller._prior_is_static hold on a prior that is not a
  pre-noised build, so churn, top-up, expiry and rebuild skip; a static buffer must be
  seeded whole, from the prior dataset.
- tb_z_source: an unknown value used to train as 'learned' silently; the stage parse, the
  config rule and Modeller.tb_z_source now refuse it.
- fwd_loss_coeffs.pooled_center: only on the fwd block, only from POOLED_CENTERS, and a
  dead key when the pooled term is off.
"""
import copy
import pathlib
from types import MethodType, SimpleNamespace

import pytest
import yaml

import config_invariants as ci
from protocol import POOLED_CENTERS, TB_Z_SOURCES, Stage
from train import Modeller

MK_DEV = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'
FUSED = {'name': 'p2', 'train_mode': 'fused', 'bwd_sampling_mode': 'prior'}


def _modeller(static, prenoised=False):
    m = SimpleNamespace(args=SimpleNamespace(buffers=SimpleNamespace(prior_buffer=SimpleNamespace(static=static))),
                        prior_is_prenoised=prenoised)
    for name in ('_prior_is_static', '_prior_static_flag'):
        setattr(m, name, MethodType(getattr(Modeller, name), m))
    return m


@pytest.mark.parametrize('static,prenoised,expected', [
    (False, False, False), (True, False, True), (False, True, True), (True, True, True)])
def test_the_static_flag_or_a_prenoised_prior_holds_the_buffer(static, prenoised, expected, capsys):
    m = _modeller(static, prenoised)
    assert m._prior_is_static('manage_prior_buffer') is expected
    m._prior_is_static('manage_prior_buffer')
    assert capsys.readouterr().out.count('skipped') == (1 if expected else 0), 'printed once per call site'


def _seed_modeller(seed_source, max_size, rows):
    pb = SimpleNamespace(static=True, seed_source=seed_source, max_size=max_size)
    m = SimpleNamespace(args=SimpleNamespace(buffers=SimpleNamespace(prior_buffer=pb)),
                        prior_dataset=list(range(rows)), prior_is_prenoised=False)
    m._prior_static_flag = MethodType(Modeller._prior_static_flag, m)
    m._prior_dataset_seed_batches = lambda limit: iter(())
    m.init_prior_buffer_seed = MethodType(Modeller.init_prior_buffer_seed, m)
    return m


@pytest.mark.parametrize('seed_source,max_size,rows', [('generated', 10, 5), ('prior_dataset', 4, 5)])
def test_a_static_buffer_that_would_not_hold_every_row_is_refused(seed_source, max_size, rows):
    with pytest.raises(ValueError, match='static'):
        _seed_modeller(seed_source, max_size, rows).init_prior_buffer_seed()


def test_unknown_tb_z_source_is_refused_everywhere():
    assert 'pooled' not in TB_Z_SOURCES
    with pytest.raises(ValueError, match='tb_z_source'):
        Stage({**FUSED, 'loss_coeffs': {'bwd': {'tb_z_source': 'pooled'}}}, 0)
    m = SimpleNamespace(args=SimpleNamespace(bwd_loss_coeffs=SimpleNamespace(tb_z_source='pooled')))
    with pytest.raises(ValueError, match='tb_z_source'):
        Modeller.tb_z_source(m, 'bwd')
    cfg = {'bwd_loss_coeffs': {'tb_z_source': 'persistant'},
           'protocols': {'x': {'stages': [{**FUSED, 'loss_coeffs': {'replay': {'tb_z_source': 'pooled'}}}]}}}
    assert len(ci.tb_z_source_is_known(cfg)) == 2


def test_pooled_center_lives_on_fwd_and_takes_a_known_value():
    assert POOLED_CENTERS == ('mean', 'head', 'head_stored', 'mean_detached', 'fwd_mean')
    with pytest.raises(ValueError, match='pooled_center'):
        Stage({**FUSED, 'loss_coeffs': {'bwd': {'pooled_center': 'head'}}}, 0)
    with pytest.raises(ValueError, match='pooled_center'):
        Stage({**FUSED, 'loss_coeffs': {'fwd': {'pooled_center': 'median'}}}, 0)
    Stage({**FUSED, 'loss_coeffs': {'fwd': {'pooled_center': 'head_stored'}}}, 0)


def test_mk_dev_states_the_new_keys_and_validates():
    cfg = yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))
    assert cfg['buffers']['prior_buffer']['static'] is True
    assert cfg['fwd_loss_coeffs']['pooled_center'] == 'mean'
    assert ci.static_prior_buffer_is_seeded(cfg) == []
    assert ci.tb_z_source_is_known(cfg) == []
    bad = copy.deepcopy(cfg)
    bad['buffers']['prior_buffer']['seed_source'] = 'generated'
    assert [v.rule for v in ci.static_prior_buffer_is_seeded(bad)] == ['static_prior_buffer_is_seeded']


def test_the_buffer_seeds_from_the_scored_rows_when_a_prenoised_prior_has_them():
    """_prior_buffer_seed_source: a pre-noised prior's scored training rows (prior_buffer_source) when the load built
    them, else prior_dataset; the seed draw reads it."""
    import types as _t
    src, ds = _t.SimpleNamespace(batch='scored'), _t.SimpleNamespace(batch='mle')
    m = SimpleNamespace(prior_dataset=ds, prior_buffer_source=src)
    pick = MethodType(Modeller._prior_buffer_seed_source, m)
    assert pick() is src
    m.prior_buffer_source = None
    assert pick() is ds
    del m.prior_buffer_source
    assert pick() is ds
