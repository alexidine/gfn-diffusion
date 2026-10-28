"""Only ConformerModeller keeps an EMA model; for every other modeller `ema_model` IS `gfn_model`.

Pinned here:
  * mk_dev carries no ema_decay and config_invariants.ema_decay_is_conformer_only is silent
    on it (the rule's mutation tests are in tests/config/test_config_invariants.py);
  * Modeller.init_gfn raises on a non-null ema_decay before building anything, unless the
    modeller is a ConformerModeller (utils.keeps_ema_copy);
  * Modeller.update_ema_model is a no-op on an aliased model even with a decay set (an EMA
    step of a module onto itself rescales its weights);
  * Checkpointer.save writes no `model_eval` for any other modeller, and load_full /
    load_weights_only alias ema_model to gfn_model, ignoring a `model_eval` an older
    checkpoint stored;
  * a ConformerModeller keeps its EMA copy and its `model_eval`.

The conformer route's EMA behaviour itself is pinned by tests/conformer/test_set_policy_resume.py.

    python -m pytest -q tests/infra/test_no_ema_off_conformer_route.py
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from config_invariants import check
from energy_sampling.checkpointing import MODELLER_STATE_DEFAULTS, Checkpointer
from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast

RULE = 'ema_decay_is_conformer_only'
MK_DEV = Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'


def _fired(cfg):
    return [v for v in check(cfg) if v.rule == RULE]


@pytest.fixture(scope='module')
def mk_dev():
    return yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))


# ------------------------------------------------------------------ the config rule

def test_mk_dev_carries_no_ema_decay_and_is_silent(mk_dev):
    assert 'ema_decay' not in mk_dev
    assert _fired(mk_dev) == []


# ------------------------------------------------------------------ the trainer

def test_init_gfn_refuses_a_decay_off_the_conformer_route():
    m = object.__new__(Modeller)
    m.args = SimpleNamespace(ema_decay=0.99, checkpoint_name=None)
    with pytest.raises(ValueError, match='ema_decay'):
        m.init_gfn()


def test_update_ema_model_leaves_an_aliased_model_alone():
    m = object.__new__(Modeller)
    m.gfn_model = torch.nn.Linear(3, 1)
    m.ema_model = m.gfn_model
    m.args = SimpleNamespace(ema_decay=0.5)
    before = copy.deepcopy(m.gfn_model.state_dict())
    m.update_ema_model()
    assert m.ema_model is m.gfn_model
    assert all(torch.equal(before[k], v) for k, v in m.gfn_model.state_dict().items())


def test_policies_is_one_module_when_aliased_and_two_when_a_copy_is_kept():
    m = object.__new__(Modeller)
    m.gfn_model = torch.nn.Linear(3, 1)
    m.ema_model = m.gfn_model
    assert m._policies() == (m.gfn_model,)
    m.ema_model = copy.deepcopy(m.gfn_model)
    assert m._policies() == (m.gfn_model, m.ema_model)


# ------------------------------------------------------------------ the checkpoint

class _Policy(torch.nn.Module):
    pb_frozen = False

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def pb_snapshot_state(self):
        return None


def _modeller(ckdir, conformer=False):
    """A stub carrying what Checkpointer.save / load_full / load_weights_only read; with
    `conformer`, an instance of the real ConformerModeller class, nothing constructed."""
    if conformer:
        from conformer_modeller import ConformerModeller
        m = ConformerModeller.__new__(ConformerModeller)
    else:
        m = SimpleNamespace()
    m.args = SimpleNamespace(
        checkpoints_dir=str(ckdir), archive_period=0, archive_buffers=False,
        checkpoint_read_only=False, grow_batch_size=False, batch_size=8, max_batch_size=8,
        override_learning_rates=False, model=SimpleNamespace(), buffers=SimpleNamespace())
    m.run_name, m.problem_slug = 'no_ema', 'probe-abc123'
    m.problem_def, m.problem_hash = {'energy_function': 'probe'}, 'abc123'
    m.device = 'cpu'
    if not conformer:
        m.buffer_device = 'cpu'     # a property on Modeller
    for k, v in MODELLER_STATE_DEFAULTS.items():
        setattr(m, k, copy.deepcopy(v))
    m.metric_tracker = SimpleNamespace(state_dict=dict, load_state_dict=lambda s: None)
    m.grad_guard = SimpleNamespace(state_dict=lambda: None, load_state_dict=lambda s: None)
    m.gfn_from_config = lambda cfg: _Policy()
    m.gfn_config = {}
    m.gfn_model = _Policy()
    m.ema_model = m.gfn_model
    m.optimizers = {}
    m.init_schedulers_optimizers = lambda: None
    m.checkpointer = Checkpointer(m)
    return m


def _stored(m, tag='running'):
    return torch.load(m.checkpointer.path_for(tag), map_location='cpu', weights_only=False)


def test_save_writes_no_model_eval_for_a_crystal_modeller(tmp_path):
    m = _modeller(tmp_path)
    m.checkpointer.save('running')
    assert 'model_eval' not in _stored(m)


@pytest.mark.parametrize('loader', ['load_full', 'load_weights_only'])
def test_load_aliases_and_ignores_an_older_checkpoints_model_eval(tmp_path, loader):
    a = _modeller(tmp_path)
    with torch.no_grad():
        a.gfn_model.w.fill_(1.0)
    a.checkpointer.save('running')
    path = a.checkpointer.path_for('running')
    ck = _stored(a)
    ck['model_eval'] = {'w': torch.full((1,), 7.0)}      # what a pre-2026-10-28 checkpoint carried
    torch.save(ck, path)

    b = _modeller(tmp_path)
    getattr(b.checkpointer, loader)(path)
    assert b.ema_model is b.gfn_model
    assert float(b.gfn_model.w) == 1.0
    assert b.gfn_model.training


def test_the_conformer_route_keeps_its_copy_and_model_eval(tmp_path):
    a = _modeller(tmp_path, conformer=True)
    a.ema_model = copy.deepcopy(a.gfn_model)
    with torch.no_grad():
        a.gfn_model.w.fill_(1.0)
        a.ema_model.w.fill_(2.0)
    a.checkpointer.save('running')
    assert float(_stored(a)['model_eval']['w']) == 2.0

    b = _modeller(tmp_path, conformer=True)
    b.checkpointer.load_full(b.checkpointer.path_for('running'))
    assert b.ema_model is not b.gfn_model
    assert (float(b.gfn_model.w), float(b.ema_model.w)) == (1.0, 2.0)
    assert b.gfn_model.training and not b.ema_model.training
