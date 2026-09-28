"""Checkpointer.archive keeps the step archive a resumed leg re-runs; every other step archives as before.

A full load restores step_ind and train.py's loop re-runs that step. At a step that is a multiple of
archive_period the rerun used to re-link `running` -> step<N> and the rolling sidecar -> step<N>_buffers,
replacing the archive the leg resumed from. The rolling sidecar at that point holds whatever the resume
restored, i.e. a LATER eval's rows, so the archive became leg 2's weights paired with leg 1's later buffers,
and a CK_STEP restart then loaded that pair (G0 gate, 2026-09-27: g0_resume step50_buffers.pt held leg 1's
step-75 prior rows after leg 2).

Driven through the REAL Checkpointer.save / save_buffers / load_full / archive, in train.py's order at a step
that is a multiple of 50 -- eval (save_buffers) -> save('running') -> archive(step) -- on a stub modeller
carrying only what those paths read. Each leg writes a distinct marker into the weights and the prior
buffer, so every file on disk says which leg and step produced it.

    python -m pytest -q tests/infra/test_archive_resume_step.py
"""
import copy
import os

import pytest
import torch
from types import SimpleNamespace

from energy_sampling.checkpointing import MODELLER_STATE_DEFAULTS, Checkpointer

# torch is imported for a one-parameter module; nothing is built and nothing reads the data drive
pytestmark = pytest.mark.fast

PERIOD = 50


class _Policy(torch.nn.Module):
    """A policy reduced to one weight, whose value marks the leg that trained it."""
    pb_frozen = False

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def pb_snapshot_state(self):
        return None


class _Rows:
    """A buffer reduced to the one thing the sidecar round trip must preserve: which rows it holds."""

    def __init__(self, label):
        self.label = label

    def state_dict(self):
        return {'label': self.label}

    @classmethod
    def from_state_dict(cls, state, device=None):
        return cls(state['label'])


def _modeller(ckdir, archive_buffers=True, archive_period=PERIOD):
    m = SimpleNamespace()
    m.args = SimpleNamespace(
        checkpoints_dir=str(ckdir), archive_period=archive_period, archive_buffers=archive_buffers,
        checkpoint_read_only=False, grow_batch_size=False, batch_size=8, max_batch_size=8,
        override_learning_rates=False, model=SimpleNamespace(), buffers=SimpleNamespace())
    m.run_name, m.problem_slug = 'g0_resume', 'probe-abc123'
    m.problem_def, m.problem_hash = {'energy_function': 'probe'}, 'abc123'
    m.device = m.buffer_device = 'cpu'
    m.buffer_cls = _Rows
    for k, v in MODELLER_STATE_DEFAULTS.items():
        setattr(m, k, copy.deepcopy(v))
    m.metric_tracker = SimpleNamespace(state_dict=dict, load_state_dict=lambda s: None)
    m.grad_guard = SimpleNamespace(state_dict=lambda: None, load_state_dict=lambda s: None)
    m.gfn_from_config = lambda cfg: _Policy()
    m.gfn_config = {}
    m.gfn_model, m.ema_model = _Policy(), _Policy()
    m.optimizers = {}
    m.init_schedulers_optimizers = lambda: None
    m.prior_buffer = _Rows('unset')
    m.checkpointer = Checkpointer(m)
    return m


def _train_to(m, step, label, weight):
    """Stand-in for the training between saves: the step clock, the weights and the buffer rows move."""
    m.step_ind = step
    with torch.no_grad():
        m.gfn_model.w.fill_(weight)
    m.prior_buffer = _Rows(label)


def _end_of_step(m, eval_here=True):
    """train.py at a step_ind % 50 == 0: eval writes the rolling sidecar, then running, then the archive."""
    if eval_here:
        m.checkpointer.save_buffers()
    m.checkpointer.save('running')
    return m.checkpointer.archive(m.step_ind)


def _archive(m, step):
    """(step_ind, weight) of the step<N> model file, and the prior-buffer label of its frozen sidecar."""
    ck = m.checkpointer
    model = torch.load(ck.path_for(f'step{step}'), map_location='cpu', weights_only=False)
    side = ck.buffers_path(f'step{step}')
    label = (torch.load(side, map_location='cpu', weights_only=False)['prior_buffer']['label']
             if os.path.exists(side) else None)
    return model['modeller_state']['step_ind'], float(model['model_train']['w']), label


def _leg1_killed_after_75(ckdir, **kw):
    """Leg 1 archives step 50, evaluates at 75 (rolling sidecar <- its step-75 rows), is killed at 80."""
    m = _modeller(ckdir, **kw)
    _train_to(m, 50, 'leg1@50', 1.0)
    _end_of_step(m)
    _train_to(m, 75, 'leg1@75', 1.5)
    m.checkpointer.save_buffers()
    return m


def _resume(ckdir, path, **kw):
    """A new process: load_full of `path`, exactly as train.py's init_gfn does on either reload branch."""
    m = _modeller(ckdir, **kw)
    m.checkpointer.load_full(path)
    return m


def test_resumed_leg_keeps_the_archive_of_the_step_it_reruns(tmp_path, capsys):
    """The G0 sequence. Fails on the pre-fix archive(): step50 then reads (50, 2.0, 'leg1@75')."""
    leg1 = _leg1_killed_after_75(tmp_path)
    assert _archive(leg1, 50) == (50, 1.0, 'leg1@50')

    leg2 = _resume(tmp_path, leg1.checkpointer.path_for('running'))
    assert leg2.step_ind == 50
    assert leg2.prior_buffer.label == 'leg1@75'           # the resume restored the rolling (later) sidecar
    _train_to(leg2, 50, 'leg1@75', 2.0)                   # the loop re-runs step 50 on those rows
    capsys.readouterr()
    assert _end_of_step(leg2) == 'step50'

    assert _archive(leg2, 50) == (50, 1.0, 'leg1@50')
    assert leg2.checkpointer.resume_step == 50
    out = capsys.readouterr().out
    assert 'step50 is the step this leg resumed at' in out and 'kept, not overwritten' in out
    # running is still the rolling file: it holds the rerun
    assert float(torch.load(leg2.checkpointer.path_for('running'), weights_only=False)
                 ['model_train']['w']) == 2.0


def test_ck_step_restart_loads_the_original_pair_and_keeps_it(tmp_path):
    """The CK_STEP=50 leg after a resume leg: it loads leg 1's step-50 pair, and its own rerun keeps it."""
    leg1 = _leg1_killed_after_75(tmp_path)
    leg2 = _resume(tmp_path, leg1.checkpointer.path_for('running'))
    _train_to(leg2, 50, 'leg2@50', 2.0)
    _end_of_step(leg2)

    ck = _resume(tmp_path, leg1.checkpointer.path_for('step50'))
    assert float(ck.gfn_model.w.detach()) == 1.0 and ck.prior_buffer.label == 'leg1@50'
    _train_to(ck, 50, 'ck@50', 3.0)
    _end_of_step(ck)
    assert _archive(ck, 50) == (50, 1.0, 'leg1@50')


def test_later_archives_are_written_and_a_fork_replaces_the_abandoned_branch(tmp_path):
    """Unchanged behaviour: an archive step other than the resume step links running and the rolling
    sidecar, including over an archive a now-abandoned leg wrote at that step."""
    leg1 = _leg1_killed_after_75(tmp_path)
    _train_to(leg1, 100, 'leg1@100', 1.75)                # leg 1 ran on to 100 before it was abandoned
    _end_of_step(leg1)
    assert _archive(leg1, 100) == (100, 1.75, 'leg1@100')

    fork = _resume(tmp_path, leg1.checkpointer.path_for('step50'))
    _train_to(fork, 50, 'fork@50', 3.0)
    _end_of_step(fork)
    _train_to(fork, 100, 'fork@100', 3.5)
    assert _end_of_step(fork) == 'step100'
    assert _archive(fork, 100) == (100, 3.5, 'fork@100')
    assert _archive(fork, 50) == (50, 1.0, 'leg1@50')


def test_fresh_leg_archives_every_period_step(tmp_path):
    m = _modeller(tmp_path)
    assert m.checkpointer.resume_step is None
    _train_to(m, 50, 'a@50', 1.0)
    assert _end_of_step(m) == 'step50'
    _train_to(m, 75, 'a@75', 1.5)
    assert m.checkpointer.archive(75) is None             # not a multiple of archive_period
    _train_to(m, 100, 'a@100', 2.0)
    assert _end_of_step(m) == 'step100'
    assert _archive(m, 50) == (50, 1.0, 'a@50')
    assert _archive(m, 100) == (100, 2.0, 'a@100')


def test_resume_at_an_archive_step_with_no_archive_on_disk_writes_it(tmp_path):
    """Leg 1 had archiving off; leg 2 turns it on and resumes at 50. Nothing to keep, so step50 is written."""
    leg1 = _modeller(tmp_path, archive_period=0)
    _train_to(leg1, 50, 'leg1@50', 1.0)
    _end_of_step(leg1)
    assert not os.path.exists(leg1.checkpointer.path_for('step50'))

    leg2 = _resume(tmp_path, leg1.checkpointer.path_for('running'))
    _train_to(leg2, 50, 'leg2@50', 2.0)
    assert _end_of_step(leg2) == 'step50'
    assert _archive(leg2, 50) == (50, 2.0, 'leg2@50')


def test_a_kept_model_archive_is_not_given_a_sidecar_from_the_resumed_leg(tmp_path):
    """Leg 1 archived without buffers; leg 2 archives with them. The rerun's rolling sidecar would pair
    leg 2's rows with leg 1's weights, so no sidecar is written beside the kept model."""
    leg1 = _modeller(tmp_path, archive_buffers=False)
    _train_to(leg1, 50, 'leg1@50', 1.0)
    _end_of_step(leg1)

    leg2 = _resume(tmp_path, leg1.checkpointer.path_for('running'))
    _train_to(leg2, 50, 'leg2@50', 2.0)
    _end_of_step(leg2)
    assert _archive(leg2, 50) == (50, 1.0, None)


def test_kept_archive_is_ranked_newest_by_mtime_as_the_old_relink_was(tmp_path):
    """The sbatch newest-archive seeds (`ls -t ..._step[0-9]*.pt | head -1`) order by mtime."""
    leg1 = _leg1_killed_after_75(tmp_path)
    old = 1_000_000_000
    for p in (leg1.checkpointer.path_for('step50'), leg1.checkpointer.buffers_path('step50')):
        os.utime(p, (old, old))

    leg2 = _resume(tmp_path, leg1.checkpointer.path_for('running'))
    _train_to(leg2, 50, 'leg2@50', 2.0)
    _end_of_step(leg2)
    for p in (leg2.checkpointer.path_for('step50'), leg2.checkpointer.buffers_path('step50')):
        assert os.path.getmtime(p) > old
    assert _archive(leg2, 50) == (50, 1.0, 'leg1@50')


def test_read_only_still_writes_nothing(tmp_path):
    leg1 = _leg1_killed_after_75(tmp_path)
    before = sorted(os.listdir(tmp_path))
    leg2 = _resume(tmp_path, leg1.checkpointer.path_for('running'))
    leg2.args.checkpoint_read_only = True
    _train_to(leg2, 100, 'leg2@100', 2.0)
    assert _end_of_step(leg2) is None
    assert sorted(os.listdir(tmp_path)) == before
