"""
model.sde.T is the one trajectory length on the crystal route (2026-10-28).

Training and evaluation integrate the same uniform grid of `model.T` steps
(utils.trajectory_T, utils.get_discretizer); `eval_T` and the `integrator` block
are refused off the conformer route (config_invariants.trajectory_length_problems,
raised by train.py::Modeller.init_gfn), and the conformer configs keep `integrator.T`.
"""

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import config_invariants as ci
import gpu_guard
import utils

HERE = Path(__file__).resolve().parents[2]   # tests/<area>/x.py -> energy_sampling/
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'


@pytest.fixture(scope='module')
def raw():
    """mk_dev as written: the sectioned model block."""
    return yaml.safe_load(CANONICAL.read_text(encoding='utf-8'))


def load(tmp_path, cfg, name='cfg.yaml'):
    """The trainer's load path (utils.get_train_args without the CLI)."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return utils.resolve_derived_config(utils.preflight_config(utils.dict2namespace(utils.load_yaml(path))))


def expected_clip(T, W=512):
    return utils._CLIP_ANCHOR * utils._grad_median(T) / utils._GRAD_MEDIAN[utils._SCALING_T_REF] * math.sqrt(W / 512)


def test_mk_dev_carries_one_trajectory_length(raw):
    assert 'eval_T' not in raw and 'integrator' not in raw
    assert raw['model']['sde']['T'] == 10
    assert ci.trajectory_length_problems(ci.canonical_model_section(raw)) == []


def test_auto_lr_and_clip_resolve_from_model_T(tmp_path, raw):
    """resolve_derived_config reads T from model.sde.T: the `auto` LRs are seeded and recorded as
    managed, and gradient_norm_clip `auto` follows T."""
    assert raw['gradient_norm_clip'] == 'auto'
    auto = [k for k in utils._LR_KEYS if utils._is_auto(raw.get(k))]
    assert auto, 'mk_dev writes no `auto` learning rate: this test no longer covers their resolution'
    for T in (10, 25):
        cfg = copy.deepcopy(raw)
        cfg['model']['sde']['T'] = T
        args = load(tmp_path, cfg, f'T{T}.yaml')
        assert utils.trajectory_T(args) == T
        assert args.gradient_norm_clip == pytest.approx(expected_clip(T))
        assert set(args.lr_servo_managed) == set(auto)
        assert all(isinstance(getattr(args, k), float) for k in auto)


def _init_gfn(args):
    import train
    train.Modeller.init_gfn(SimpleNamespace(args=args))


def test_init_gfn_refuses_eval_T_and_integrator_on_the_crystal_route(tmp_path, raw):
    """The config still loads (old battery files stay comparable through config_snapshot); the trainer
    refuses it at model construction."""
    for key, value, match in (('eval_T', 10, 'eval_T is retired'),
                              ('integrator', {'T': 10}, 'integrator block is retired')):
        cfg = copy.deepcopy(raw)
        cfg[key] = value
        args = load(tmp_path, cfg, f'{key}.yaml')
        with pytest.raises(ValueError, match=match):
            _init_gfn(args)


def test_init_gfn_refuses_an_sde_policy_without_model_T(tmp_path, raw):
    cfg = copy.deepcopy(raw)
    del cfg['model']['sde']['T']
    cfg['gradient_norm_clip'] = 1.0
    args = load(tmp_path, cfg)
    with pytest.raises(ValueError, match='model.sde.T is unset'):
        _init_gfn(args)


def test_an_old_crystal_config_still_resolves_from_integrator_T(tmp_path, raw):
    """A battery file written before the move (eval_T + integrator, no model.T) loads and derives its clip
    from integrator.T, so config_snapshot can still compare it."""
    cfg = copy.deepcopy(raw)
    del cfg['model']['sde']['T']
    cfg['eval_T'], cfg['integrator'] = 25, {'T': 25}
    args = load(tmp_path, cfg)
    assert utils.trajectory_T(args) == 25
    assert args.gradient_norm_clip == pytest.approx(expected_clip(25))


def test_an_nf_policy_needs_no_T(tmp_path, raw):
    """model.kind nf: no T, the learning rates still resolve, and the discretizer the trainer hands the
    one-step policy is built without raising (it raises only if called)."""
    cfg = copy.deepcopy(raw)
    cfg['model']['kind'] = 'nf'
    del cfg['model']['sde']['T']
    cfg['gradient_norm_clip'] = 1.0
    args = load(tmp_path, cfg)
    assert utils.trajectory_T(args) is None
    assert all(isinstance(getattr(args, k), float) for k in args.lr_servo_managed)
    disc = utils.get_discretizer(utils.trajectory_T(args))
    with pytest.raises(ValueError, match='no trajectory length'):
        disc(2)


def test_auto_clip_without_T_raises(tmp_path, raw):
    cfg = copy.deepcopy(raw)
    cfg['model']['kind'] = 'nf'
    del cfg['model']['sde']['T']
    with pytest.raises(ValueError, match='gradient_norm_clip `auto`'):
        load(tmp_path, cfg)


def test_the_conformer_configs_keep_integrator_T():
    """conformer_dev (train_conformer.py, no energy_function) and conformer_mk (conformer_torsions) set
    integrator.T; trajectory_T reads it and the crystal rule abstains."""
    for name in ('conformer_dev.yaml', 'conformer_mk.yaml'):
        cfg = ci.canonical_model_section(yaml.safe_load((HERE / 'configs' / name).read_text(encoding='utf-8')))
        assert 'integrator' in cfg, name
        assert ci.trajectory_length_problems(cfg) == [], name
        assert utils.trajectory_T(cfg) == cfg['integrator']['T'], name
        assert utils.trajectory_T(utils.dict2namespace(copy.deepcopy(cfg))) == cfg['integrator']['T'], name


def test_get_discretizer_takes_a_count_or_a_holder():
    for T in (4, SimpleNamespace(T=4)):
        grid = utils.get_discretizer(T)(3)
        assert tuple(grid.shape) == (3, 5)
        assert float(grid[0, 0]) == 0.0 and float(grid[0, -1]) == 1.0


def test_the_vram_signature_is_unchanged_by_the_move():
    """gpu_guard.config_signature keys recorded peaks on T: model.T and the old integrator.T sign alike,
    so peaks measured before the move are still found."""
    base = {'energy_function': 'elj', 'batch_size': 100, 'z_primes': [1]}
    new = dict(base, model={'s_emb_dim': 8, 'dplr_rank': 0, 'T': 10})
    old = dict(base, model={'s_emb_dim': 8, 'dplr_rank': 0}, integrator={'T': 10})
    assert gpu_guard.config_signature(new) == gpu_guard.config_signature(old)


def test_gfn_config_leaves_T_out():
    """GFN takes no T: _build_latent_gfn_config passes the model block without it."""
    import train
    model = SimpleNamespace(T=10, s_emb_dim=8, policy_hidden_dim=16)
    stub = SimpleNamespace(
        _state_features_dim=lambda: 0, get_conditioning_dim=lambda: 0,
        energy_function=SimpleNamespace(data_ndim=12, is_crystal=True),
        _resolve_periodic_centroid_axes=lambda: None, _resolve_dead_latent_rows=lambda: None,
        _policy_entry=lambda: {}, device='cpu',
        args=SimpleNamespace(molecule_conditioning=False, temperature_conditioning=False, sg_conditioning=False,
                             zp_conditioning=False, z_primes=[1], model=model))
    cfg = train.Modeller._build_latent_gfn_config(stub)
    assert 'T' not in cfg
    assert cfg['s_emb_dim'] == 8 and cfg['policy_hidden_dim'] == 16
    assert model.T == 10, 'the config namespace itself is not modified'
