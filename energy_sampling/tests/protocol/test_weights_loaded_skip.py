"""Phase-2 entry on a route with no prior model, and a bounded stage exit.

`skip_if: weights_loaded`. The crystal phase-2 arm enters its conditional stage at step 0
through `skip_if: prior_loaded`: prior_model_name loads a frozen GFN, begin() sees
`prior_model`, skips the MLE stage, and advancing fires the next stage's on_enter
(freeze_pb, rebuild_prior_by_churn, set_lr_flow). The conformer route can never hold a
`prior_model` -- ConformerModeller.init_prior_dataset never builds one -- so the same arm
there silently re-ran phase 1 from the loaded weights. `weights_loaded` holds iff this
process loaded checkpoint_name with load_weights_only (Checkpointer.load_weights_only sets
`weights_only_loaded`). Crystal configs never name it, and `prior_loaded` is unchanged.

`max_steps`. Stage exits are AND-lists with no step metric, so "exit on the gate OR after N
steps" could not be written; and progress_gate reads 0 below its min_history, so an exit on
`gates/progress_done` alone was an unbounded wait. max_steps is the bound: after N of the
stage's own steps the next eval advances it, saying so. Absent, nothing changes.

    python -m pytest -q tests/protocol/test_weights_loaded_skip.py
"""
from types import SimpleNamespace

import pytest

from protocol import SKIP_CONDITIONS, Stage, StageProtocol, fresh_stage_ctrl
from utils import MetricTracker

pytestmark = pytest.mark.fast

TICK = 10

PHASE1 = {'name': 'train_prior', 'train_mode': 'bwd', 'bwd_sampling_mode': 'dataset',
          'skip_if': 'weights_loaded',
          'exit': [{'metric': 'gates/progress_done', 'above': 0.5, 'patience': 1}]}
PHASE2 = {'name': 'tb_conditioning', 'train_mode': 'fused', 'bwd_sampling_mode': 'prior',
          'on_enter': ['freeze_pb', 'rebuild_prior_by_churn', 'set_lr_flow:0.0001']}


def engine(stages, **attrs):
    """A StageProtocol over `stages` with every transition side effect RECORDED, so a test
    can see which on_enter actions fired and in what order."""
    calls = []
    args = SimpleNamespace(protocol='p', grow_batch_size=False, lr_flow=1e-2,
                           protocols=SimpleNamespace(p=SimpleNamespace(stages=stages)))
    m = SimpleNamespace(
        args=args, stage=None, stage_ctrl=fresh_stage_ctrl(),
        metric_tracker=MetricTracker(period=25.0), step_ind=0, optimizers={},
        combo_loss_record=[], batch_sizer=None,
        batch_size_oom_ceiling=None, batch_size_oom_ceiling_at=None,
        batch_size_oom_min=None, _runaway_last_cut=None, _runaway_unresponsive_stage=None,
        _accum_floor_warned_stage=None, batch_size_last_grow=0,
        fwd_frac=0.0, bwd_frac=1.0, replay_frac=0.0,
        init_schedulers_optimizers=lambda: calls.append('init_optimizers'),
        set_loss_coeffs=lambda: None,
        set_pb_freeze=lambda mode, source_state=None: calls.append(f'freeze_pb:{mode}'),
        rebuild_prior_by_churn=lambda n=None: calls.append('rebuild_prior_by_churn'),
        lr_controller=SimpleNamespace(on_stage_change=lambda: 0),
        grad_guard=SimpleNamespace(refresh=lambda reason=None: None),
        checkpointer=SimpleNamespace(save=lambda tag: calls.append(f'save:{tag}'),
                                     save_buffers=lambda: calls.append('save_buffers')))
    for k, v in attrs.items():
        setattr(m, k, v)
    return StageProtocol(m), m, calls


# ------------------------------------------------------------------ weights_loaded


def test_the_condition_is_declared():
    assert SKIP_CONDITIONS == ('prior_loaded', 'weights_loaded')
    assert Stage(PHASE1, 0).skip_if == 'weights_loaded'
    with pytest.raises(ValueError, match='skip_if'):
        Stage({**PHASE1, 'skip_if': 'weights_restored'}, 0)


def test_a_weights_only_load_enters_phase_2_and_fires_its_on_enter(capsys):
    p, m, calls = engine([PHASE1, PHASE2], weights_only_loaded=True)
    p.begin()
    assert m.stage == 'tb_conditioning'
    assert calls.index('freeze_pb:full') < calls.index('rebuild_prior_by_churn')
    assert m.args.lr_flow == pytest.approx(1e-4)
    out = capsys.readouterr().out
    assert "weights loaded (checkpoint_name, load_weights_only) -- skipping stage " \
           "'train_prior'" in out
    assert 'lr_flow -> 0.0001' in out


def test_without_the_load_the_run_stays_in_phase_1():
    """The separator: nothing else about the modeller changed."""
    p, m, calls = engine([PHASE1, PHASE2])
    p.begin()
    assert m.stage == 'train_prior'
    assert calls == []


def test_a_full_load_is_not_a_weights_only_load():
    """A full resume restores its own stage; begin() leaves a resumed run where it is."""
    p, m, calls = engine([PHASE1, PHASE2], weights_only_loaded=True)
    m.stage, m.step_ind = 'train_prior', 12000
    p.begin()
    assert m.stage == 'train_prior' and calls == []


def test_prior_loaded_is_unchanged():
    crystal = {**PHASE1, 'skip_if': 'prior_loaded'}
    p, m, _ = engine([crystal, PHASE2], weights_only_loaded=True)
    p.begin()
    assert m.stage == 'train_prior', 'a weights-only load must not trip prior_loaded'
    p, m, _ = engine([crystal, PHASE2], prior_model=object())
    p.begin()
    assert m.stage == 'tb_conditioning'


# ------------------------------------------------------------------ max_steps


@pytest.mark.parametrize('bad', [0, -5, 2.5, True, '100'])
def test_max_steps_must_be_a_positive_int(bad):
    with pytest.raises(ValueError, match='max_steps'):
        Stage({**PHASE1, 'max_steps': bad}, 0)


def test_absent_max_steps_is_none_and_writes_nothing():
    """Inert when absent: no clock stamped, no metric, no pulled eval."""
    p, m, _ = engine([{k: v for k, v in PHASE1.items() if k != 'skip_if'}, PHASE2])
    p.begin()
    assert p.stage.max_steps is None
    for _ in range(50):
        m.step_ind += TICK
        p.tick()
    assert 'max_steps_entry' not in m.stage_ctrl
    assert not m.stage_ctrl['request_eval']
    assert 'protocol/max_steps_frac' not in p.report()


def test_a_cap_on_the_last_stage_needs_stop():
    p, _, _ = engine([PHASE1, {**PHASE2, 'max_steps': 100}])
    with pytest.raises(ValueError, match='LAST stage'):
        p.stages
    p, _, _ = engine([PHASE1, {**PHASE2, 'max_steps': 100, 'on_exit': ['stop']}])
    assert p.stages[-1].max_steps == 100


def test_the_cap_forces_the_exit_the_gate_never_gives(capsys):
    """progress_gate publishes nothing below its min_history, so the exit term never gets
    a measurement: without the cap this stage would never end. With it, the tick at the cap
    pulls the eval forward and that eval advances, saying the exit was FORCED."""
    capped = {k: v for k, v in PHASE1.items() if k != 'skip_if'}
    capped['max_steps'] = 100
    p, m, calls = engine([capped, PHASE2])
    p.begin()
    assert m.stage_ctrl['max_steps_entry'] == 0

    while m.step_ind < 90:
        m.step_ind += TICK
        p.tick()
        assert not m.stage_ctrl['request_eval']
        assert not p.maybe_advance({}), f'advanced early at step {m.step_ind}'
    m.step_ind += TICK
    p.tick()
    assert m.stage_ctrl['request_eval'], 'the cap must pull the eval forward'
    assert p.report()['protocol/max_steps_frac'] == pytest.approx(1.0)
    assert p.maybe_advance({})
    assert m.stage == 'tb_conditioning' and 'freeze_pb:full' in calls
    assert 'FORCED EXIT at max_steps=100 (100 steps in stage)' in capsys.readouterr().out


def test_the_gate_still_exits_first_when_it_fires():
    capped = {k: v for k, v in PHASE1.items() if k != 'skip_if'}
    capped['max_steps'] = 10_000
    p, m, _ = engine([capped, PHASE2])
    p.begin()
    m.step_ind += TICK
    p.publish_gate('progress_done', 1.0)
    p.tick()
    assert p.maybe_advance({}) and m.stage == 'tb_conditioning'


def test_the_clock_is_stamped_at_entry_and_rides_the_stage_state():
    """Stamped when the stage is ENTERED, so the cap counts the stage's own steps; and it
    lives in stage_ctrl, which the checkpoint carries, so a requeued leg continues it."""
    capped2 = {**PHASE2, 'max_steps': 50, 'on_exit': ['stop']}
    p, m, _ = engine([{k: v for k, v in PHASE1.items() if k != 'skip_if'}, capped2])
    p.begin()
    m.step_ind = 700
    p.advance(None)
    assert m.stage_ctrl['max_steps_entry'] == 700

    # a requeued leg: a fresh engine over the SAME restored stage state
    restored = dict(m.stage_ctrl)
    q, n, _ = engine([{k: v for k, v in PHASE1.items() if k != 'skip_if'}, capped2])
    n.stage, n.stage_ctrl, n.step_ind = 'tb_conditioning', restored, 740
    q.begin()
    assert n.stage_ctrl['max_steps_entry'] == 700, 'a resume must not restart the clock'
    n.step_ind = 750
    q.tick()
    assert n.stage_ctrl['request_eval']
    assert q.maybe_advance({})
    assert n._stop_requested, 'the last stage ends the run at its cap'


def test_a_restored_stage_without_a_stamp_starts_the_clock_and_says_so(capsys):
    capped = {k: v for k, v in PHASE1.items() if k != 'skip_if'}
    capped['max_steps'] = 30
    p, m, _ = engine([capped, PHASE2])
    m.stage, m.step_ind = 'train_prior', 5000          # resumed, cap added since
    p.begin()
    m.step_ind += TICK
    p.tick()
    assert m.stage_ctrl['max_steps_entry'] == 5010
    assert "max_steps=30 clock starts at step 5010" in capsys.readouterr().out
    m.step_ind += 30
    p.tick()
    assert p.maybe_advance({}) and m.stage == 'tb_conditioning'
