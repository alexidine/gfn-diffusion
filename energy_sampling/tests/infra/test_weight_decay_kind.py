"""weight_decay_kind through the real Modeller.init_schedulers_optimizers (stub modeller of test_nf_checkpoint):

  l2 (the default, also when the key is absent)  torch Adam everywhere, weight_decay on every group incl. the Z head
  decoupled                                       torch AdamW on the policy, the Z head (flow_model) at weight decay 0;
                                                  one step with zero gradient shrinks a policy weight by exactly
                                                  lr x weight_decay and leaves the Z head unchanged
and the config rule config_invariants.weight_decay_is_well_formed.

    python -m pytest -q tests/infra/test_weight_decay_kind.py
"""
import pytest
import torch

from energy_sampling import config_invariants as ci
from energy_sampling.tests.infra.test_nf_checkpoint import _base_config, _modeller
from models.nf_policy import build_policy

pytestmark = pytest.mark.fast
POLICY = ('fwd', 'bwd', 'replay', 'fused')


def _built(tmp_path, **wd):
    m = _modeller(tmp_path, conditional=True)
    for k, v in wd.items():
        setattr(m.args, k, v)
    m.gfn_config = m._build_gfn_config()
    torch.manual_seed(0)
    m.gfn_model = build_policy(m.gfn_config)
    m.ema_model = m.gfn_model
    m.init_schedulers_optimizers()
    return m


def _flow_ids(m):
    return {id(p) for p in m.gfn_model.flow_model.parameters()}


def test_absent_kind_is_l2_adam_everywhere(tmp_path):
    m = _built(tmp_path, use_weight_decay=True, weight_decay=0.05)
    flow = _flow_ids(m)
    for name, opt in m.optimizers.items():
        if name not in POLICY + ('flow',):
            continue
        assert type(opt) is torch.optim.Adam, name
        for g in opt.param_groups:
            assert g['weight_decay'] == 0.05, (name, any(id(p) in flow for p in g['params']))


def test_decoupled_is_adamw_on_policy_and_spares_the_z_head(tmp_path):
    m = _built(tmp_path, use_weight_decay=True, weight_decay=0.05, weight_decay_kind='decoupled')
    flow = _flow_ids(m)
    for name in POLICY:
        assert type(m.optimizers[name]) is torch.optim.AdamW, name
        for g in m.optimizers[name].param_groups:
            is_flow = any(id(p) in flow for p in g['params'])
            assert g['weight_decay'] == (0.0 if is_flow else 0.05), (name, is_flow)
    assert type(m.optimizers['flow']) is torch.optim.Adam
    assert all(g['weight_decay'] == 0.0 for g in m.optimizers['flow'].param_groups)


def test_decoupled_step_shrinks_policy_weights_by_lr_times_decay(tmp_path):
    m = _built(tmp_path, use_weight_decay=True, weight_decay=0.05, weight_decay_kind='decoupled')
    opt = m.optimizers['fused']
    flow = _flow_ids(m)
    params = [p for g in opt.param_groups for p in g['params']]
    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.zeros_like(p)
    opt.step()
    lr = {id(p): g['lr'] for g in opt.param_groups for p in g['params']}
    for p, b in zip(params, before):
        want = b if id(p) in flow else b * (1 - lr[id(p)] * 0.05)
        torch.testing.assert_close(p.detach(), want, rtol=0, atol=1e-7)
    assert any(id(p) not in flow and b.abs().sum() > 0 for p, b in zip(params, before))


def test_unknown_kind_raises(tmp_path):
    with pytest.raises(ValueError, match='weight_decay_kind'):
        _built(tmp_path, use_weight_decay=True, weight_decay=0.05, weight_decay_kind='adamw')


@pytest.mark.parametrize('cfg, bad', [
    ({}, False),
    ({'use_weight_decay': True, 'weight_decay': 0.1, 'weight_decay_kind': 'decoupled'}, False),
    ({'use_weight_decay': True, 'weight_decay': 0.1, 'weight_decay_kind': 'l2'}, False),
    ({'use_weight_decay': False, 'weight_decay': 0.1, 'weight_decay_kind': 'decoupled'}, True),
    ({'use_weight_decay': True, 'weight_decay': 0.1, 'weight_decay_kind': 'adamw'}, True),
    ({'use_weight_decay': True, 'weight_decay': None}, True),
])
def test_config_rule(cfg, bad):
    assert bool(ci.weight_decay_is_well_formed(cfg)) is bad
