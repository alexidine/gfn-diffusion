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
    cfg['buffers']['store']['enabled'] = True
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
    del cfg['buffers']['store']['kappa']
    assert ci.row_store_settings(cfg)['kappa'] == ci.ROW_STORE_DEFAULTS['kappa']
    assert _msgs(cfg) == []


GLOBAL_REFUSALS = [
    ('energy_function', lambda c: c.update(energy_function='conformer_torsions'), 'conformer'),
    ('temperature_conditioning', lambda c: c.update(temperature_conditioning=True), 'temperature_conditioning'),
    ('prior_flow_path', lambda c: c['energy_config'].update(prior_flow_path='x.pt'), 'prior_flow_path'),
    ('atom_tokens', lambda c: c['model'].setdefault('nf', {}).update(atom_tokens=True), 'atom_tokens'),
    ('prioritise', lambda c: c['buffers']['replay_buffer']['prioritise'].update(enabled=True), 'prioritise'),
    ('val_frac', lambda c: c['buffers']['replay_buffer'].update(val_frac=0.1), 'val_frac'),
    ('anchor frozen', lambda c: c['buffers']['anchor_buffer'].update(frozen=False), 'frozen'),
    ('harvested', lambda c: c['noised_prior'].update(train_kinds=['child', 'harvested']), 'harvested'),
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


MALFORMED = [('kappa', -0.1), ('kappa', 'x'), ('floor_frac', 1.5), ('uniform_frac', -0.1), ('bar', -1.0),
             ('eps', 0.0), ('cap_per_condition', -1), ('cap_per_condition', 2.5), ('max_rows', -5),
             ('records_kept', 0), ('records_kept', True), ('refresh_chunk', 0), ('admit_reward_min', 'low'),
             ('kappa', float('nan')), ('is_exponent', 1.5), ('is_exponent', -0.5), ('is_exponent', float('nan')),
             ('is_exponent', None), ('is_exponent', 'half'), ('is_exponent', True)]


@pytest.mark.parametrize('key,value', MALFORMED, ids=[f'{k}={v}' for k, v in MALFORMED])
def test_malformed_values_are_refused(mk_dev, key, value):
    cfg = _on(mk_dev)
    cfg['buffers']['store'][key] = value
    assert any(f'buffers.store.{key}' in m for m in _msgs(cfg)), _msgs(cfg)


@pytest.mark.parametrize('value', [0, 0.0, 0.5, 1, 1.0])
def test_is_exponent_accepts_the_closed_unit_interval(mk_dev, value):
    cfg = _on(mk_dev)
    cfg['buffers']['store']['is_exponent'] = value
    assert _msgs(cfg) == []


def test_records_kept_null_is_a_baseline_not_an_error(mk_dev):
    cfg = _on(mk_dev)
    cfg['buffers']['store']['records_kept'] = None
    assert _msgs(cfg) == []
    assert any('unbounded' in m for m in _msgs(cfg, ci.BASELINE))


def test_settings_read_the_trainer_namespace(mk_dev):
    from types import SimpleNamespace
    ns = SimpleNamespace(buffers=SimpleNamespace(store=SimpleNamespace(enabled=True, bar=2.0)))
    s = ci.row_store_settings(ns)
    assert s['bar'] == 2.0 and s['kappa'] == ci.ROW_STORE_DEFAULTS['kappa']
