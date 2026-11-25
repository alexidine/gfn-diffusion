"""buffers.store on the raw config: config_invariants.row_store_is_well_formed (row_store_problems), which
Modeller.init_row_store raises on. Every refusal is a mutation of mk_dev put on the NF route with the store on; the
unmutated shape must be clean, so each test fails if its refusal stops firing or starts firing on the clean shape."""
import copy
import pathlib

import pytest
import yaml

from energy_sampling import config_invariants as ci

pytestmark = pytest.mark.fast

MK_DEV = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'
RULE = 'row_store_is_well_formed'


@pytest.fixture(scope='module')
def mk_dev():
    return yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))


def _on(mk_dev):
    """mk_dev under model.kind nf with the store on: the shape the smoke run trains."""
    cfg = copy.deepcopy(mk_dev)
    cfg['model']['kind'] = 'nf'
    cfg['gradient_norm_clip'] = 10.0
    block = {k: v for k, v in cfg['buffers']['store'].items() if k not in ci.ROW_STORE_RETIRED_KEYS}
    cfg['buffers']['store'] = dict(block, enabled=True, ram_gb=16)
    cfg['buffers']['anchor_buffer'].update(frozen=True, seed_floors_only=True)
    cfg['noised_prior'].update(live_kernel='physical', train_kinds=['anchor'])
    cfg['energy_config']['crystal_coordinates'] = 'fd'
    cfg['grow_batch_size'] = False
    stages = cfg['protocols'][cfg['protocol']]['stages']
    for st in stages:
        st['max_steps'] = 100
        for hook in ('on_enter', 'on_exit'):
            st[hook] = [a for a in st.get(hook) or [] if not ci.row_store_refuses_action(a)]
    cfg['epochs'] = 100 * len(stages) + ci.ROW_STORE_STAGE_MARGIN * len(stages)
    return cfg


def _stage(cfg, i=1):
    return cfg['protocols'][cfg['protocol']]['stages'][i]


def _msgs(cfg, severity=ci.ERROR):
    return [v.detail for v in ci.check(cfg) if v.rule == RULE and v.severity == severity]


#: the store keys whose mk_dev value is not the code default
MK_DEV_OVERRIDES = {'uniform_frac': 0.5}


def test_mk_dev_carries_every_store_key_with_the_code_default(mk_dev):
    assert set(mk_dev['buffers']['store']) == set(ci.ROW_STORE_DEFAULTS)
    assert mk_dev['buffers']['store'] == {**ci.ROW_STORE_DEFAULTS, **MK_DEV_OVERRIDES}
    assert ci.ROW_STORE_DEFAULTS['is_exponent'] == 1.0


def test_shipped_config_is_clean(mk_dev):
    assert _msgs(mk_dev) == [] and _msgs(mk_dev, ci.BASELINE) == []


def test_store_absent_or_off_judges_nothing(mk_dev):
    cfg = copy.deepcopy(mk_dev)
    del cfg['buffers']['store']
    cfg['temperature_conditioning'] = True       # would be refused with the store on
    assert _msgs(cfg) == []
    assert ci.row_store_settings(cfg) is None
    cfg['buffers']['store'] = {'enabled': False, 'kappa': -1}
    assert _msgs(cfg) == []


@pytest.mark.parametrize('block', [{'enabled': False, 'not_a_key': 1}, {'not_a_key': 1},
                                   {'enabled': 'true', 'not_a_key': 1}, {'enabled': 1, 'not_a_key': 1},
                                   {'enabled': None, 'not_a_key': 1}])
def test_a_block_not_enabled_is_not_inspected(mk_dev, block):
    cfg = copy.deepcopy(mk_dev)
    cfg['buffers']['store'] = block
    assert ci.row_store_settings(cfg) is None
    assert ci.row_store_problems(cfg) == []
    assert _msgs(cfg) == []


def test_a_namespace_block_not_enabled_is_not_inspected():
    from types import SimpleNamespace
    ns = SimpleNamespace(buffers=SimpleNamespace(store=SimpleNamespace(enabled=False, not_a_key=1)))
    assert ci.row_store_settings(ns) is None


def test_an_enabled_block_still_refuses_an_unknown_key():
    with pytest.raises(ValueError, match='unknown keys'):
        ci.row_store_settings({'buffers': {'store': {'enabled': True, 'not_a_key': 1}}})


def test_new_store_keys_and_their_defaults(mk_dev):
    new = {'ram_gb': None, 'ram_frac': 0.5, 'mle_draw': 'anchor', 'replay_weight': 'uniform'}
    assert {k: ci.ROW_STORE_DEFAULTS[k] for k in new} == new
    assert {k: mk_dev['buffers']['store'][k] for k in new} == new
    assert mk_dev['buffers']['store']['enabled'] is False


@pytest.mark.parametrize('ram_gb,ndim,frac,rows', [(224, 12, 0.5, 595342001), (16, 12, 0.5, 42524428),
                                                   (224, 12, 1.0, 1190684002), (10 ** 6, 12, 0.5, 2 ** 31 - 1)])
def test_store_row_limit(ram_gb, ndim, frac, rows):
    assert ci.store_row_limit(ram_gb, ndim, frac) == rows


def test_store_row_limit_default_share_and_dimension():
    assert ci.store_row_limit(224, 12) == ci.store_row_limit(224, 12, ci.ROW_STORE_DEFAULTS['ram_frac'])
    assert ci.store_row_limit(224, 24) < ci.store_row_limit(224, 12)


def test_store_row_limit_counts_the_store_allocation():
    """The per-slot bytes store_row_limit charges for the columns and index are what CrystalStore allocates."""
    from energy_sampling import crystal_store as cs
    ndim, slots, n_cond = 12, 1000, 3
    st = cs.CrystalStore(ndim, n_cond, slots, 5)
    columns = sum(getattr(st, name).numel() * getattr(st, name).element_size() for name, _ in cs._COLUMNS)
    assert columns == slots * (4 * ndim + 43)
    pool_slack = cs.MIN_POOL_CAP * cs.N_KINDS * n_cond * st.index.arena.itemsize
    assert st.index.where.nbytes + st.index.arena.nbytes - pool_slack + st._free.nbytes == 16 * slots
    assert cs.MAX_ROWS_LIMIT == 2 ** 31 - 1


def test_store_row_limit_needs_no_torch():
    import ast
    tree = ast.parse(pathlib.Path(ci.__file__).read_text(encoding='utf-8'))
    imported = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or '').split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert 'torch' not in imported


def test_store_on_nf_is_clean(mk_dev):
    assert _msgs(_on(mk_dev)) == []


def test_store_on_sde_is_refused(mk_dev):
    cfg = _on(mk_dev)
    cfg['model']['kind'] = 'sde'
    assert any('model.kind nf' in m for m in _msgs(cfg))


def test_unknown_key_is_refused(mk_dev):
    cfg = _on(mk_dev)
    cfg['buffers']['store']['kapa'] = 1.0
    assert any('unknown keys' in m for m in _msgs(cfg))


def test_absent_key_reads_the_code_default(mk_dev):
    cfg = _on(mk_dev)
    del cfg['buffers']['store']['ram_frac']
    assert ci.row_store_settings(cfg)['ram_frac'] == ci.ROW_STORE_DEFAULTS['ram_frac']
    assert _msgs(cfg) == []


GLOBAL_REFUSALS = [
    ('energy_function', lambda c: c.update(energy_function='conformer_torsions'), 'conformer'),
    ('temperature_conditioning', lambda c: c.update(temperature_conditioning=True), 'temperature_conditioning'),
    ('prior_flow_path', lambda c: c['energy_config'].update(prior_flow_path='x.pt'), 'prior_flow_path'),
    ('prioritise', lambda c: c['buffers']['replay_buffer']['prioritise'].update(enabled=True), 'prioritise'),
    ('val_frac', lambda c: c['buffers']['replay_buffer'].update(val_frac=0.1), 'val_frac'),
    ('anchor frozen', lambda c: c['buffers']['anchor_buffer'].update(frozen=False), 'frozen'),
    ('train_kinds', lambda c: c['noised_prior'].update(train_kinds=['anchor', 'child']), 'train_kinds'),
    ('live_kernel', lambda c: c['noised_prior'].update(live_kernel=None), 'live_kernel'),
    ('fd chart', lambda c: c['energy_config'].update(crystal_coordinates='latent'), 'crystal_coordinates'),
    ('seed_floors_only', lambda c: c['buffers']['anchor_buffer'].update(seed_floors_only=False), 'seed_floors_only'),
    ('ram_gb', lambda c: c['buffers']['store'].update(ram_gb=None), 'ram_gb'),
    ('retired key', lambda c: c['buffers']['store'].update(kappa=1.0), 'previous row store'),
    ('rollout_group_draw', lambda c: c['condition_log_z'].update(rollout_group_draw='level'), 'rollout_group_draw'),
    ('admit_from_eval', lambda c: c['buffers']['replay_buffer'].update(admit_from_eval=False), 'admit_from_eval'),
    ('replay_weight', lambda c: c['buffers']['store'].update(replay_weight='not_a_rule'), 'REPLAY_WEIGHTS'),
    ('short epochs', lambda c: c.update(epochs=c['epochs'] - 1), 'epochs'),
    ('under_drawn', lambda c: c['condition_log_z'].update(rollout_condition_draw='under_drawn'), 'under_drawn'),
]


@pytest.mark.parametrize('name,mutate,needle', GLOBAL_REFUSALS, ids=[g[0] for g in GLOBAL_REFUSALS])
def test_global_refusals(mk_dev, name, mutate, needle):
    cfg = _on(mk_dev)
    mutate(cfg)
    assert any(needle in m for m in _msgs(cfg)), _msgs(cfg)


def _coeffs(mode, **kw):
    def f(st):
        st.setdefault('loss_coeffs', {}).setdefault(mode, {}).update(kw)
    return f


STAGE_REFUSALS = [
    ('condition_draw', lambda s: s.update(condition_draw={'conditions': 0, 'replay_rows': 2, 'prior_rows': 2}),
     'condition_draw'),
    ('replay_warmup_rows', lambda s: s.update(replay_warmup_rows=1000), 'replay_warmup_rows'),
    ('replay mle', _coeffs('replay', mle=1.0), 'replay mle'),
    ('pooled_vg', _coeffs('fwd', pooled_vg=1.0), 'pooled_vg'),
    ('bwd vg_lb', _coeffs('bwd', vg_lb=1.0), 'bwd vg_lb'),
    ('replay vg_lme', _coeffs('replay', vg_lme=1.0), 'replay vg_lme'),
    ('weighted_bwd_sampling', lambda s: s['flags'].update(weighted_bwd_sampling=True), 'weighted_bwd_sampling'),
    ('coeff_schedule', lambda s: s.update(coeff_schedule={'bounding_coeff': {'target': 1.0, 'steps': 10,
                                                                             'kind': 'linear'}}), 'coeff_schedule'),
    ('anneal_coeffs', lambda s: s.setdefault('balance', {}).update(anneal_coeffs={'bounding_coeff': {'target': 1.0}}),
     'anneal_coeffs'),
    ('drift_std_max', lambda s: s['fwd_rollout_triggers'].update(drift_std_max=1.0), 'drift_std_max'),
    ('ess_min', lambda s: s['fwd_rollout_triggers'].update(ess_min=0.5), 'ess_min'),
    ('val_gap_max', lambda s: s['fwd_rollout_triggers'].update(val_gap_max=1.0), 'val_gap_max'),
    ('lr_sensor ray', lambda s: s.update(lr_sensor={'kind': 'ray'}), 'ray'),
    ('z_pin', lambda s: s.update(z_pin_rollout_every=5), 'z_pin_rollout_every'),
    ('prior action', lambda s: s.update(on_enter=['reseed_prior_from_dataset']), 'reseed_prior_from_dataset'),
    ('bare bootstrap_z', lambda s: s.update(on_enter=['bootstrap_z']), 'bootstrap_z:offset'),
    ('bootstrap_z rollout', lambda s: s.update(on_enter=['bootstrap_z:rollout:4000']), 'bootstrap_z:offset'),
    ('bootstrap_z conditioner', lambda s: s.update(on_exit=['bootstrap_z:train_conditioner']), 'bootstrap_z:offset'),
    ('no max_steps', lambda s: s.pop('max_steps'), 'max_steps'),
]


@pytest.mark.parametrize('name,mutate,needle', STAGE_REFUSALS, ids=[g[0] for g in STAGE_REFUSALS])
def test_stage_refusals(mk_dev, name, mutate, needle):
    cfg = _on(mk_dev)
    mutate(_stage(cfg))
    assert any(needle in m for m in _msgs(cfg)), _msgs(cfg)


def test_z_calibration_replay_mode_is_refused_only_where_a_stage_runs_it(mk_dev):
    cfg = _on(mk_dev)
    cfg['z_calibration']['mode'] = 'replay'
    assert not any('z_calibration.mode replay' in m for m in _msgs(cfg))     # no stage runs the servo
    _stage(cfg)['flags']['z_calibration'] = True
    assert any('z_calibration.mode replay' in m for m in _msgs(cfg)), _msgs(cfg)
    cfg['z_calibration']['mode'] = 'rollout'
    assert not any('z_calibration.mode' in m for m in _msgs(cfg))


def test_occupancy_trigger_is_allowed(mk_dev):
    cfg = _on(mk_dev)
    _stage(cfg)['fwd_rollout_triggers']['occupancy_min_batches'] = 4.0
    assert _msgs(cfg) == []


MALFORMED = [('ram_gb', 0), ('ram_gb', -1.0), ('ram_gb', 'lots'), ('ram_gb', True), ('ram_frac', 0.0), ('ram_frac', 1.5),
             ('ram_frac', None), ('mle_draw', 'row'), ('mle_draw', None)]


@pytest.mark.parametrize('key,value', MALFORMED, ids=[f'{k}={v}' for k, v in MALFORMED])
def test_malformed_values_are_refused(mk_dev, key, value):
    cfg = _on(mk_dev)
    cfg['buffers']['store'][key] = value
    assert any(f'buffers.store.{key}' in m for m in _msgs(cfg)), _msgs(cfg)


def test_replay_weight_names_are_the_registry():
    from energy_sampling import crystal_store as cs
    assert ci.replay_weight_rules() == tuple(cs.REPLAY_WEIGHTS)


def test_cond_noise_is_refused_only_beside_a_tb_branch(mk_dev):
    cfg = _on(mk_dev)
    cfg['model'].setdefault('nf', {})['cond_noise'] = 0.1
    for st in cfg['protocols'][cfg['protocol']]['stages']:
        st.setdefault('loss_coeffs', {}).setdefault('bwd', {})['tb'] = 0.0
        st['loss_coeffs'].setdefault('replay', {})['tb'] = 0.0
    assert not any('cond_noise' in m for m in _msgs(cfg))
    _stage(cfg)['loss_coeffs']['replay']['tb'] = 1.0
    assert any('cond_noise' in m for m in _msgs(cfg)), _msgs(cfg)


def test_settings_read_the_trainer_namespace(mk_dev):
    from types import SimpleNamespace
    ns = SimpleNamespace(buffers=SimpleNamespace(store=SimpleNamespace(enabled=True, bar=2.0)))
    s = ci.row_store_settings(ns)
    assert s['bar'] == 2.0 and s['kappa'] == ci.ROW_STORE_DEFAULTS['kappa']


@pytest.mark.parametrize('action', ['bootstrap_z:offset', 'bootstrap_z:offset:500'])
def test_the_offset_bootstrap_is_allowed(mk_dev, action):
    cfg = _on(mk_dev)
    _stage(cfg).update(on_enter=[action])
    assert not any('bootstrap_z' in m for m in _msgs(cfg)), _msgs(cfg)


def test_batch_growth_is_allowed_on_the_store(mk_dev):
    cfg = _on(mk_dev)
    cfg['grow_batch_size'] = True
    assert not [m for sev, m in ci.row_store_problems(cfg) if sev == ci.ERROR and 'grow_batch_size' in m]
