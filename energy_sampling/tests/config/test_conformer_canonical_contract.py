"""configs/conformer_mk.yaml is the conformer route's operational surface: it passes the
config_snapshot contract with each protocol selected, carries every key mk_dev carries bar a
named crystal-only list, carries every ConformerTorsions parameter explicitly, sets the
conditional protocol the way the battery generator requires of every arm, and sizes its
buffers for conformer rows on the local card.

The route table in the file (the comment above `protocol:`) is applied here as ROUTE_TABLE, so
the table the file prints and the switch this test proves are one list. If a switch stops
being sufficient, the unconditional contract fails; if the table stops being necessary, the
bare switch below stops failing and says so.
"""
import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

import config_invariants
import config_snapshot

ES = Path(__file__).resolve().parents[2]
CANONICAL = ES / 'configs' / 'conformer_mk.yaml'
MK_DEV = ES / 'configs' / 'mk_dev.yaml'

_spec = importlib.util.spec_from_file_location('conformer_cond_make',
                                               ES / 'configs' / 'conformer_cond' / 'make.py')
make = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make)

#: the file's route table, conformer_conditional_tb -> conformer_unconditional
ROUTE_TABLE = {
    'protocol': 'conformer_unconditional',
    'molecules_path': None,
    'energy_config.smiles': 'CCCO',
    'embedding_conditioning': False,
    'model.policy_kind': 'flat',
    'lr_flow': 0.1,
    'z_calibration.fill_threshold': 0.5,
    'z_calibration.fill_from_eval': 'fill',
}
#: constructor parameters ConformerModeller.init_energy_function writes from the top level
#: (or leaves to torch's default dtype), or from the conditions file (each member's stored
#: reference), so energy_config does not carry them
_INJECTED = {'device', 'dtype', 'temperature_conditioning', 'embedding_conditioning',
             'embedding_conditioning_dim', 'reference_positions'}


@pytest.fixture(scope='module')
def raw():
    return yaml.safe_load(CANONICAL.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def mk_dev():
    return yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))


def _set(cfg, dotted, value):
    node = cfg
    *head, last = dotted.split('.')
    for k in head:
        node = node[k]
    node[last] = value


def _contract(tmp_path, cfg, name):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return config_snapshot.contract(str(p))


def _stage(cfg, protocol, name):
    return next(s for s in cfg['protocols'][protocol]['stages'] if s['name'] == name)


def test_the_canonical_file_passes_the_contract():
    snap, issues = config_snapshot.contract(str(CANONICAL))
    assert issues == []
    assert snap['config']['_active_protocol'] == 'conformer_conditional_tb'
    assert [s['name'] for s in snap['stages']] == ['train_prior', 'tb_conditioning']


def test_the_canonical_file_is_clean_under_every_rule(raw):
    assert config_invariants.check(copy.deepcopy(raw)) == []


def test_the_route_table_switches_to_the_unconditional_protocol(tmp_path, raw):
    cfg = copy.deepcopy(raw)
    for k, v in ROUTE_TABLE.items():
        _set(cfg, k, v)
    snap, issues = _contract(tmp_path, cfg, 'uncond.yaml')
    assert issues == []
    assert snap['config']['_active_protocol'] == 'conformer_unconditional'
    assert [s['name'] for s in snap['stages']] == ['train_prior', 'equilibration']


def test_the_protocol_word_alone_is_not_a_switch(tmp_path, raw):
    """The table is load-bearing: selecting the unconditional protocol on the conditional
    globals fails the contract (its rollout cadence needs the z fill armed, and the
    conditional globals refuse its `learned` Z source)."""
    cfg = copy.deepcopy(raw)
    cfg['protocol'] = 'conformer_unconditional'
    _snap, issues = _contract(tmp_path, cfg, 'bare.yaml')
    assert issues, 'the bare protocol switch passed; the route table has become unnecessary'


def test_every_mk_dev_key_is_carried_or_named_crystal_only(raw, mk_dev):
    assert make.missing_mk_dev_keys(raw, mk_dev) == []
    # and the crystal-only list does not rot: each entry is an mk_dev key this file omits
    mk = make.flat({k: v for k, v in mk_dev.items() if k != 'protocols'})
    mine = make.flat({k: v for k, v in raw.items() if k != 'protocols'})
    stale = sorted(k for k in make.CRYSTAL_ONLY_KEYS if k not in mk or k in mine)
    assert stale == [], f'CRYSTAL_ONLY_KEYS entries that are not an omitted mk_dev key: {stale}'


def test_every_energy_parameter_is_explicit(raw):
    params, non_energy, _defaults = make.energy_contract(
        (ES / 'energies' / 'conformer_torsions.py').read_text(encoding='utf-8'),
        (ES / 'conformer_modeller.py').read_text(encoding='utf-8'))
    ec = set(raw['energy_config'])
    crystal_non_energy = {k.split('.', 1)[1] for k in make.CRYSTAL_ONLY_KEYS}
    assert sorted((params | non_energy) - ec - _INJECTED - crystal_non_energy) == []
    assert sorted(ec - params - non_energy - make.TOLERATED_ENERGY_KEYS) == []


def test_the_conditional_protocol_carries_the_baseline_settings(raw):
    """make.check_protocol is what every generated arm is held to; the canonical file is
    held to the same function, plus the settings an arm overrides."""
    make.check_protocol(copy.deepcopy(raw), 'canonical')
    tp = _stage(raw, 'conformer_conditional_tb', 'train_prior')
    tb = _stage(raw, 'conformer_conditional_tb', 'tb_conditioning')
    assert tp['max_steps'] > raw['progress_gate']['min_history']
    assert tp['flags']['scramble_conditions'] is False
    assert tb['flags']['update_log_z'] is True                 # the mixed feed is the default
    assert tb['on_enter'] == ['rebuild_prior_by_churn', 'set_lr_flow:1.0e-4', 'freeze_pb']
    assert raw['embedding_conditioning_dim'] == 256
    assert raw['xcond_eval']['enabled'] is False


@pytest.mark.parametrize('dotted, value, named', [
    ('fwd.emp_z_persistent', 0.0, 'emp_z_persistent'),
    ('fwd.reward_grads', 0.0, 'reward_grads'),
    ('bwd.freeze_z', 0.0, 'freeze_z'),
    ('bwd.beta', 10.0, 'beta'),
])
def test_the_protocol_check_refuses_a_seat_off_the_baseline(raw, dotted, value, named):
    cfg = copy.deepcopy(raw)
    _set(_stage(cfg, 'conformer_conditional_tb', 'tb_conditioning')['loss_coeffs'], dotted, value)
    with pytest.raises(SystemExit, match=named):
        make.check_protocol(cfg, 'mutant')


def test_the_canonical_buffers_fit_the_local_card(raw):
    """The budget is the footprint of the store form the config runs: compact rows and the
    conditions table with molecules_path set, full graph rows without it."""
    assert make.stores_compact(raw)
    make.refuse_over_vram(copy.deepcopy(raw), 'canonical', make.LOCAL_CARD_BYTES)
    budget = make.LOCAL_CARD_BYTES * raw['cuda_memory_fraction'] * make.BUFFER_SHARE
    # compact rows: 250,000 prior rows are a few hundred MB, where full rows would not fit
    over = copy.deepcopy(raw)
    over['buffers']['prior_buffer']['max_size'] = 250_000
    make.refuse_over_vram(over, 'compact', make.LOCAL_CARD_BYTES)
    full = copy.deepcopy(over)
    full['molecules_path'] = None
    assert make.row_bytes(full)['prior'] == make.BYTES_PER_ROW_MAX
    with pytest.raises(SystemExit, match='full graph rows.*buffer budget'):
        make.refuse_over_vram(full, 'full', make.LOCAL_CARD_BYTES)
    # what binds a compact run: the conditions table, and replay rows with their trajectories
    n_over = int(budget // make.TABLE_BYTES_PER_CONDITION_MAX) + 1
    with pytest.raises(SystemExit, match='compact rows.*conditions table.*buffer budget'):
        make.refuse_over_vram(copy.deepcopy(raw), 'table', make.LOCAL_CARD_BYTES, n_over)
    replay = copy.deepcopy(raw)
    replay['buffers']['replay_buffer']['max_size'] = int(budget // make.row_bytes(raw)['replay']) + 1
    with pytest.raises(SystemExit, match='buffer budget'):
        make.refuse_over_vram(replay, 'replay', make.LOCAL_CARD_BYTES)


def test_row_bytes_follow_the_store_form(raw):
    K, T = make.K_MAX, int(raw['integrator']['T'])
    per = make.row_bytes(raw)
    assert per['prior_sample'] == 4 * K + 4
    assert per['prior'] == per['anchor'] == 2 * (4 * K + 4) + 4
    assert per['replay'] == per['prior'] + 4 * K * (T + 1 + make.FORCE_LEGS) + 12 * make.ATOMS_MAX
    noised = dict(copy.deepcopy(raw), prior_dataset_noise='thermal')
    assert make.row_bytes(noised)['prior_sample'] == 2 * (4 * K + 4)
    assert make.row_bytes(raw, K=66)['prior'] == 2 * (4 * 66 + 4) + 4
    full = dict(copy.deepcopy(raw), molecules_path=None)
    assert set(make.row_bytes(full).values()) == {make.BYTES_PER_ROW_MAX}
    # the anchor buffer is budgeted at its resolved capacity: max_size, or the seed if larger
    assert raw['buffers']['anchor_buffer']['seed_source'] == 'prior_dataset'
    sample, cap = raw['energy_config']['prior_sample_size'], raw['buffers']['anchor_buffer']['max_size']
    assert make.store_rows(raw)['anchor'] == make.anchor_capacity(raw) == max(sample, cap)
    assert make.anchor_capacity(raw, seed_rows=cap + 7) == cap + 7
    assert make.anchor_capacity(raw, seed_rows=1) == cap
    lazy = copy.deepcopy(raw)
    lazy['buffers']['anchor_buffer']['seed_source'] = 'generated'
    assert make.anchor_capacity(lazy, seed_rows=10 ** 9) == cap
    # a sidecar holds the three buffers, a compact row with its host columns
    one = dict(copy.deepcopy(raw), archive_period=0)
    rows = make.store_rows(raw)
    assert make.sidecar_disk_bytes(one) == sum(
        rows[k] * (per[k] + make.COMPACT_HOST_BYTES_PER_ROW) for k in ('prior', 'anchor', 'replay'))


def test_replay_is_dormant_in_phase_two(raw):
    """At frac 0, replay must also be UNREAD, or the fused step force-refreshes it every
    controller.refresh_every steps. A stage without a balance block reads every mode."""
    from protocol import Stage
    spec = _stage(raw, 'conformer_conditional_tb', 'tb_conditioning')
    assert Stage(copy.deepcopy(spec), 1).read_modes == {'fwd', 'bwd'}
    bare = {k: v for k, v in copy.deepcopy(spec).items() if k != 'balance'}
    assert 'replay' in Stage(bare, 1).read_modes
