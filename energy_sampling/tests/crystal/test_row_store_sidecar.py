"""The row store through the buffer sidecar (checkpointing.Checkpointer.buffer_state / restore_buffers) and
Modeller.init_row_store, on a Modeller built without __init__. What each test would catch:

  * a sidecar that does not carry the store, or carries it without the run's stamp;
  * restore_buffers building the store itself (it must defer to init_row_store, which runs after the registry);
  * a matching stamp reseeding, or a mismatched one loading rows scored under another prior, chart or reward;
  * init_row_store accepting a configuration the config rule refuses, or a prior that is not pre-noised.

CPU, synthetic rows, no energy computed.

    python -m pytest -q tests/crystal/test_row_store_sidecar.py
"""
import copy
import pathlib
from types import SimpleNamespace

import pytest
import torch
import yaml

from energy_sampling.checkpointing import Checkpointer
from energy_sampling.config_invariants import canonical_model_section
from energy_sampling import row_store
from energy_sampling.row_store import RowStore, TAG_CHILD
from energy_sampling.train import Modeller
from energy_sampling.utils import dict2namespace

pytestmark = pytest.mark.fast

MK_DEV = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'
SETTINGS = dict(bar=1.0, eps=0.1, cap_per_condition=100, max_rows=1000, records_kept=2, kappa=1.0,
                floor_frac=0.25, uniform_frac=0.1, admit_reward_min=None)


def _args(**store):
    cfg = yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))
    cfg['model']['kind'] = 'nf'
    cfg['gradient_norm_clip'] = 10.0
    cfg['buffers']['store'].update(enabled=True, **store)
    return dict2namespace(canonical_model_section(cfg))


def _store():
    st = RowStore(2, 1, 1, 'cpu', **SETTINGS)
    st.set_conditions(torch.zeros(6, dtype=torch.long), torch.zeros(6, 1))
    st.seed(torch.zeros(6, dtype=torch.long), torch.arange(12.).reshape(6, 2), torch.arange(6.), TAG_CHILD,
            torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool), energy=-torch.arange(6.))
    return st


def _modeller(stamp, **store):
    m = Modeller.__new__(Modeller)
    m.args = _args(**store)
    m.args.buffer_device = 'cpu'
    m.prior_is_prenoised = True
    m.problem_def, m.problem_hash, m.step_ind = {'p': 1}, 'abc', 7
    m._store_stamp = lambda: stamp
    m.seeded, m.refreshed = 0, 0

    def seed(settings):
        m.seeded += 1
        m.seed_settings = settings
        m.row_store = _store()
    m._store_seed = seed
    m._store_refresh = lambda: setattr(m, 'refreshed', m.refreshed + 1)
    return m


def test_sidecar_carries_the_store_and_its_stamp():
    m = _modeller({'prior': 'a'})
    m.row_store = _store()
    state = Checkpointer(m).buffer_state()
    assert state['row_store']['stamp'] == {'prior': 'a'} and state['row_store']['n_mle'] == 4


def test_no_store_writes_none():
    m = _modeller({'prior': 'a'})
    assert m.row_store is None and Checkpointer(m).buffer_state()['row_store'] is None


def test_restore_defers_to_init_row_store_and_a_matching_stamp_loads():
    src = _modeller({'prior': 'a'})
    src.row_store = _store()
    src.row_store.version = 3
    state = Checkpointer(src).buffer_state()
    dst = _modeller({'prior': 'a'})
    Checkpointer(dst).restore_buffers(copy.deepcopy(state), 'x_buffers.pt')
    assert dst.row_store is None and dst._row_store_state is not None
    dst.init_row_store()
    assert dst.seeded == 0 and dst.refreshed == 1 and dst._row_store_state is None
    assert torch.equal(dst.row_store.x, src.row_store.x) and dst.row_store.version == 3


def test_a_mismatched_stamp_reseeds(capsys):
    src = _modeller({'prior': 'a'})
    src.row_store = _store()
    src.row_store.version = 3
    dst = _modeller({'prior': 'b'})
    Checkpointer(dst).restore_buffers(Checkpointer(src).buffer_state(), 'x_buffers.pt')
    dst.init_row_store()
    assert dst.seeded == 1 and dst.refreshed == 1 and dst.row_store.version == 0
    assert 'ignored' in capsys.readouterr().out


def test_settings_come_from_the_config_not_the_sidecar():
    src = _modeller({'prior': 'a'})
    src.row_store = _store()
    dst = _modeller({'prior': 'a'}, bar=3.5, is_exponent=0.5)
    Checkpointer(dst).restore_buffers(Checkpointer(src).buffer_state(), 'x_buffers.pt')
    dst.init_row_store()
    assert dst.row_store.bar == 3.5 and dst.row_store.is_exponent == 0.5      # the config's, not the sidecar store's 1.0


def test_seeding_is_handed_every_store_setting_from_the_config():
    m = _modeller({'prior': 'a'}, is_exponent=0.25)
    m.init_row_store()
    assert m.seeded == 1 and m.seed_settings['is_exponent'] == 0.25
    assert set(m.seed_settings) == set(row_store._SETTINGS)


def test_store_off_builds_nothing():
    m = _modeller({'prior': 'a'})
    m.args.buffers.store.enabled = False
    m.init_row_store()
    assert m.row_store is None and m.seeded == 0


def test_refused_configuration_raises_at_init():
    m = _modeller({'prior': 'a'})
    m.args.temperature_conditioning = True
    with pytest.raises(ValueError, match='temperature_conditioning'):
        m.init_row_store()


def test_a_prior_that_is_not_prenoised_raises():
    m = _modeller({'prior': 'a'})
    m.prior_is_prenoised = False
    with pytest.raises(ValueError, match='pre-noised'):
        m.init_row_store()


def test_the_stamp_carries_the_store_format_so_an_older_sidecar_reseeds(monkeypatch, capsys):
    """Modeller._store_stamp itself (no stub): it names row_store.FORMAT_VERSION, so a sidecar written by an older
    format (whose stamp has no such entry, or another value) is ignored and the store seeded afresh rather than handed
    to RowStore.from_state_dict."""
    from argparse import Namespace
    from energy_sampling import train as train_mod
    from energy_sampling.row_store import FORMAT_VERSION
    monkeypatch.setattr(train_mod.noised_prior, 'identity', lambda args, ef: {'id': 1})
    monkeypatch.setattr(train_mod.noised_prior, 'settings', lambda args: {'window_kT': 3.0})
    stub = SimpleNamespace(
        args=Namespace(energy_config=Namespace(temperature=2.5, lj_coeff=0.36), prior_path='nowhere.pt'),
        energy_function=SimpleNamespace(condition_library_size=4, data_ndim=12),
        identifier_registry={}, prior_data_record={})
    stamp = Modeller._store_stamp(stub)
    assert stamp['store_format'] == FORMAT_VERSION
    old = {k: v for k, v in stamp.items() if k != 'store_format'}
    src = _modeller(old)
    src.row_store = _store()
    state = Checkpointer(src).buffer_state()
    state['row_store']['format_version'] = FORMAT_VERSION - 1
    dst = _modeller(stamp)
    Checkpointer(dst).restore_buffers(state, 'x_buffers.pt')
    dst.init_row_store()
    assert dst.seeded == 1 and 'ignored' in capsys.readouterr().out
