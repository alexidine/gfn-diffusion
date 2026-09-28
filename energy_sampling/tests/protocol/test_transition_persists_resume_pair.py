"""An eval-time stage transition leaves a resume pair -- model file + buffer sidecar -- that describes the
stage it entered, at the transition step and at every step until the next eval.

train.py writes the rolling buffer sidecar BEFORE evaluation(), which is where StageProtocol.maybe_advance
fires a transition, and writes 'running' and the step archive AFTER it. on_enter's buffer surgery
(rebuild_prior_by_churn here) therefore landed after the sidecar write, so 'running' and step<N> carried the
NEW stage (P_B frozen) beside the PRE-on_enter buffers until the next eval, and on_enter does not re-fire on
resume (conformer G0 gate, 2026-09-27: step100.pt in tb_conditioning with P_B frozen, step100_buffers.pt
holding the 2502 phase-1 prior rows where the rebuild had just made 1562).

Driven through the REAL StageProtocol.maybe_advance/advance and the REAL Checkpointer save / save_buffers /
archive / load_full, in train.py's order (pinned by the first test), on a stub modeller carrying only what
those paths read. Every buffer state carries a label naming who produced it and at which step.

    python -m pytest -q tests/protocol/test_transition_persists_resume_pair.py
"""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from energy_sampling.checkpointing import MODELLER_STATE_DEFAULTS, Checkpointer
from protocol import StageProtocol
from utils import MetricTracker

# torch is imported for a one-parameter module; nothing is built and nothing reads the data drive
pytestmark = pytest.mark.fast

TRAIN_PY = Path(__file__).resolve().parents[2] / 'train.py'

PHASE1 = {'name': 'train_prior', 'train_mode': 'bwd', 'bwd_sampling_mode': 'dataset',
          'exit': [{'metric': 'eval/done', 'above': 0.5, 'patience': 1}]}
PHASE2 = {'name': 'tb_conditioning', 'train_mode': 'fused', 'bwd_sampling_mode': 'prior',
          'on_enter': ['freeze_pb', 'rebuild_prior_by_churn']}
EXIT_NOW = {'done': 1.0}          # the eval metrics that satisfy PHASE1's exit
STAY = {}                         # an eval that does not


class _Policy(torch.nn.Module):
    """A policy reduced to one weight and the P_B-freeze mark the checkpoint carries."""
    pb_frozen = None

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def pb_snapshot_state(self):
        return {'mode': self.pb_frozen} if self.pb_frozen else None


class _Rows:
    """A buffer reduced to the one thing the sidecar round trip must preserve: which rows it holds."""

    def __init__(self, label):
        self.label = label

    def state_dict(self):
        return {'label': self.label}

    @classmethod
    def from_state_dict(cls, state, device=None):
        return cls(state['label'])


def _modeller(ckdir, stages=(PHASE1, PHASE2), read_only=False, **attrs):
    m = SimpleNamespace()
    m.args = SimpleNamespace(
        protocol='p', protocols=SimpleNamespace(p=SimpleNamespace(stages=list(stages))),
        checkpoints_dir=str(ckdir), archive_period=50, archive_buffers=True,
        checkpoint_read_only=read_only, grow_batch_size=False, batch_size=8, max_batch_size=8,
        override_learning_rates=False, model=SimpleNamespace(), buffers=SimpleNamespace())
    m.run_name, m.problem_slug = 'g0_p1', 'probe-abc123'
    m.problem_def, m.problem_hash = {'energy_function': 'probe'}, 'abc123'
    m.device = m.buffer_device = 'cpu'
    m.buffer_cls = _Rows
    for k, v in MODELLER_STATE_DEFAULTS.items():
        setattr(m, k, copy.deepcopy(v))
    m._runaway_last_cut = m._runaway_unresponsive_stage = m._accum_floor_warned_stage = None
    m.metric_tracker = MetricTracker(period=25.0)
    m.grad_guard = SimpleNamespace(state_dict=lambda: None, load_state_dict=lambda s: None,
                                   refresh=lambda reason=None: None)
    m.gfn_from_config = lambda cfg: _Policy()
    m.gfn_config = {}
    m.gfn_model, m.ema_model = _Policy(), _Policy()
    m.optimizers = {}
    m.init_schedulers_optimizers = lambda: None
    m.set_loss_coeffs = lambda: None
    m.lr_controller = SimpleNamespace(on_stage_change=lambda: 0)
    m.set_pb_freeze = lambda mode, source_state=None: setattr(m.gfn_model, 'pb_frozen', mode)
    # the on_enter surgery under test: the prior buffer is replaced by rows built at this step
    m.rebuild_prior_by_churn = lambda n=None: setattr(m, 'prior_buffer', _Rows(f'rebuilt@{m.step_ind}'))
    m.prior_buffer = _Rows('unset')
    m.checkpointer = Checkpointer(m)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m, StageProtocol(m)


def _train_to(m, step):
    """Stand-in for the training between checkpoint writes: the clock moves and the buffer rows turn over."""
    m.step_ind = step
    m.prior_buffer = _Rows(f'{m.stage}@{step}')


def _eval_step(m, p, eval_metrics):
    """train.py at an eval step: the rolling sidecar, then evaluation() (maybe_advance), then -- on the
    50-step grid -- 'running' and the step archive."""
    m.checkpointer.save_buffers()
    p.maybe_advance(eval_metrics)
    if m.step_ind % 50 == 0:
        m.checkpointer.save('running')
        m.checkpointer.archive(m.step_ind)


def _grid_step(m):
    """train.py at a 50-grid step with no eval: 'running' and the archive, no sidecar."""
    m.checkpointer.save('running')
    m.checkpointer.archive(m.step_ind)


def _resume(ckdir, path):
    """A new process loading `path` through load_full, as init_gfn does on both reload branches:
    (stage, step, P_B freeze, prior-buffer label) of what that leg would train on."""
    r, _ = _modeller(ckdir)
    r.checkpointer.load_full(path)
    return r.stage, r.step_ind, r.gfn_model.pb_frozen, r.prior_buffer.label


def _phase1_to_100(ckdir):
    """Phase 1 up to its eval at 100, which does not exit: running/step100 and the sidecar at 100."""
    m, p = _modeller(ckdir)
    p.begin()
    _train_to(m, 50)
    _eval_step(m, p, STAY)
    _train_to(m, 100)
    _eval_step(m, p, STAY)
    return m, p


def test_the_harness_follows_train_py_order():
    """The premise: in Modeller.train the rolling sidecar is written before evaluation(), and 'running'
    and the archive after it. If this order changes, the sequence the other tests drive is not the run's."""
    src = TRAIN_PY.read_text(encoding='utf-8')
    body = src[src.index('    def train(self):'):]
    at = [body.index(s) for s in ('self.checkpointer.save_buffers()', 'self.evaluation(',
                                  "self.checkpointer.save('running')",
                                  'self.checkpointer.archive(self.step_ind)')]
    assert at == sorted(at)


def test_on_grid_transition_pairs_running_and_its_archive_with_the_post_on_enter_buffers(tmp_path):
    """The G0 sequence: the exit fires at the eval of step 150, a multiple of 50 and of archive_period.
    Fails on the pre-fix code: both pairs read ('tb_conditioning', 150, 'full', 'train_prior@150')."""
    m, p = _phase1_to_100(tmp_path)
    _train_to(m, 150)
    _eval_step(m, p, EXIT_NOW)
    assert m.stage == 'tb_conditioning' and m.prior_buffer.label == 'rebuilt@150'

    want = ('tb_conditioning', 150, 'full', 'rebuilt@150')
    assert _resume(tmp_path, m.checkpointer.path_for('running')) == want
    assert _resume(tmp_path, m.checkpointer.path_for('step150')) == want
    # the archive before the transition is untouched: phase 1 with its own rows
    assert _resume(tmp_path, m.checkpointer.path_for('step100')) == \
        ('train_prior', 100, None, 'train_prior@100')


def test_off_grid_transition_leaves_no_window_that_mixes_the_stages(tmp_path):
    """A transition pulled forward by request_eval lands off the 50-step grid (130), where train.py writes
    neither 'running' nor an archive. Right after it, and at the next grid step (150, no eval), a resume
    must get the new stage on the rebuilt rows. Fails on the pre-fix code: at 130 'running' is still
    phase 1 at 100, and at 150 'running'/step150 read ('tb_conditioning', 150, 'full', 'train_prior@130')."""
    m, p = _phase1_to_100(tmp_path)
    _train_to(m, 130)
    _eval_step(m, p, EXIT_NOW)
    assert m.step_ind % 50 != 0 and m.stage == 'tb_conditioning'
    assert _resume(tmp_path, m.checkpointer.path_for('running')) == \
        ('tb_conditioning', 130, 'full', 'rebuilt@130')

    _train_to(m, 150)
    _grid_step(m)
    # the sidecar lags the weights to the last eval, as at any grid step -- but it is the NEW stage's
    want = ('tb_conditioning', 150, 'full', 'rebuilt@130')
    assert _resume(tmp_path, m.checkpointer.path_for('running')) == want
    assert _resume(tmp_path, m.checkpointer.path_for('step150')) == want


def test_an_eval_that_does_not_transition_writes_what_it_always_did(tmp_path):
    """Unchanged behaviour away from a transition: an off-grid eval writes the sidecar only."""
    m, p = _phase1_to_100(tmp_path)
    running_before = os.path.getmtime(m.checkpointer.path_for('running'))
    _train_to(m, 130)
    _eval_step(m, p, STAY)
    assert m.stage == 'train_prior'
    assert os.path.getmtime(m.checkpointer.path_for('running')) == running_before
    assert _resume(tmp_path, m.checkpointer.path_for('running')) == \
        ('train_prior', 100, None, 'train_prior@130')


def test_a_stop_persists_nothing_extra(tmp_path):
    """A stop enters no stage: 'running' stays where the grid left it and train.py writes 'final'."""
    last = {**PHASE1, 'on_exit': ['stop']}
    m, p = _modeller(tmp_path, stages=(last,))
    p.begin()
    _train_to(m, 100)
    _eval_step(m, p, STAY)
    _train_to(m, 130)
    _eval_step(m, p, EXIT_NOW)
    assert m._stop_requested
    assert _resume(tmp_path, m.checkpointer.path_for('running'))[:2] == ('train_prior', 100)


def test_the_step_zero_skip_chain_persists_nothing(tmp_path):
    """begin()'s skip chain runs on_enter at step 0, before anything of the run is on disk; its first
    eval writes the sidecar before its first 'running'. Only 'stage_start' is written there, as before."""
    skippable = {**PHASE1, 'skip_if': 'weights_loaded'}
    m, p = _modeller(tmp_path, stages=(skippable, PHASE2), weights_only_loaded=True)
    p.begin()
    assert m.stage == 'tb_conditioning' and m.prior_buffer.label == 'rebuilt@0'
    assert sorted(os.listdir(tmp_path)) == [Path(m.checkpointer.path_for('stage_start')).name]


def test_read_only_still_writes_nothing(tmp_path):
    m, p = _modeller(tmp_path, read_only=True)
    p.begin()
    _train_to(m, 130)
    _eval_step(m, p, EXIT_NOW)
    assert m.stage == 'tb_conditioning'
    assert os.listdir(tmp_path) == []
