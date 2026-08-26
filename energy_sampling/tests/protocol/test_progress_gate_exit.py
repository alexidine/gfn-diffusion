"""
The progress gate must reach the exit trigger THROUGH publish_gate.

WHAT WAS WRONG (found live, toy_wk2_aug24 / sl1ebn35, 2026-08-24). progress_metrics
computed the verdict and wrote 'gates/progress_done' into the WANDB METRICS DICT
only. The exit trigger resolves 'gates/*' against protocol.ctrl['gates'] and
advances its streak only on a fresh ctrl['gate_written'] stamp -- both of which
only publish_gate writes. So the wandb panel showed done=1 for 1500+ steps while
the trigger's streak held at 0 (hold-not-reset semantics on a missing stamp) and
train_prior could never exit. The exact welded-shut silent non-exit the exit
redesign existed to remove, reintroduced through the stamp channel.

Three tests: the protocol seam advances when published; the bug shape (verdict
computed, never published) holds the stage FOREVER, so a reintroduction fails
loudly here; and the real Modeller.progress_metrics body publishes.

`pytest tests/protocol/test_progress_gate_exit.py -q`
"""

from types import SimpleNamespace

import numpy as np
import pytest

from protocol import StageProtocol, fresh_stage_ctrl
from utils import MetricTracker

pytestmark = pytest.mark.fast

TICK = 10

PROGRESS_EXIT = [{'metric': 'gates/progress_done', 'above': 0.5, 'patience': 1}]


def engine(exit_block):
    stages = [{'name': 's0', 'train_mode': 'bwd', 'bwd_sampling_mode': 'dataset',
               'exit': exit_block},
              {'name': 's1', 'train_mode': 'bwd', 'bwd_sampling_mode': 'dataset'}]
    args = SimpleNamespace(protocol='p', grow_batch_size=False,
                           protocols=SimpleNamespace(p=SimpleNamespace(stages=stages)))
    m = SimpleNamespace(
        args=args, stage='s0', stage_ctrl=fresh_stage_ctrl(),
        metric_tracker=MetricTracker(period=25.0), step_ind=0,
        combo_loss_record=[], batch_sizer=None,
        batch_size_oom_ceiling=None, batch_size_oom_ceiling_at=None,
        batch_size_oom_min=None, _runaway_last_cut=None, _runaway_unresponsive_stage=None,
        _accum_floor_warned_stage=None, batch_size_last_grow=0,
        fwd_frac=0.0, bwd_frac=1.0, replay_frac=0.0,
        init_schedulers_optimizers=lambda: None, set_loss_coeffs=lambda: None,
        lr_controller=SimpleNamespace(on_stage_change=lambda: 0),
        grad_guard=SimpleNamespace(refresh=lambda reason=None: None),
        checkpointer=SimpleNamespace(save=lambda tag: None))
    return StageProtocol(m), m


def test_published_progress_done_advances_the_stage():
    """The intended path: verdict published at eval cadence -> streak reaches
    patience 1 -> trigger arms -> maybe_advance transitions."""
    p, m = engine(PROGRESS_EXIT)
    m.step_ind = 2000
    p.publish_gate('progress_done', 1.0)   # what progress_metrics now does
    p.tick()
    assert m.stage_ctrl['exit'].get(0, 0) >= 1
    assert p.maybe_advance({})
    assert p.stage.name == 's1'


def test_an_unpublished_verdict_holds_the_stage_forever():
    """THE BUG SHAPE. The verdict lives only in a metrics dict the protocol
    never sees: no value in ctrl['gates'], no stamp in ctrl['gate_written'].
    However long the run ticks, the streak must hold at 0 and the stage must
    not advance -- if someone reroutes the publish back to metrics-dict-only,
    this test is the one that stays green while the run silently stalls, so it
    asserts the STALL rather than the fix."""
    p, m = engine(PROGRESS_EXIT)
    wandb_metrics = {}
    for _ in range(50):
        m.step_ind += TICK
        wandb_metrics['gates/progress_done'] = 1.0   # computed, never published
        p.tick()
    assert m.stage_ctrl['exit'].get(0, 0) == 0
    assert not p.maybe_advance({})
    assert p.stage.name == 's0'


def test_modeller_progress_metrics_publishes_the_verdict():
    """The real seam: Modeller.progress_metrics' body must hand its verdict to
    protocol.publish_gate, not only to the metrics dict. Runs the actual method
    on a minimal fake; a canned-early verdict (below min_history) is enough --
    the publish must happen for 0 verdicts too, since a 0 resets the trigger
    streak the way a fresh mle_flat=0 does."""
    from train import Modeller

    published = []
    latents = np.random.RandomState(0).rand(64, 4)
    fake = SimpleNamespace(
        args=SimpleNamespace(progress_gate=None),
        prior_dataset=SimpleNamespace(batch=object(), y=np.random.rand(64)),
        energy_function=SimpleNamespace(periodic_dims=None, temperature=1.0),
        step_ind=50,
        protocol=SimpleNamespace(
            publish_gate=lambda name, value: published.append((name, float(value)))),
        _batch_latents=lambda b: latents,
        _buffer_y_fn=lambda: 'y',
    )
    sample_batch = SimpleNamespace(y=np.random.rand(64))
    metrics = {}
    Modeller.progress_metrics(fake, metrics, sample_batch)
    assert 'gates/progress_done' in metrics, (
        'progress_metrics did not compute a verdict at all -- the fake no longer '
        'reaches the gate; fix the fake before trusting the publish assertion')
    assert ('progress_done', metrics['gates/progress_done']) in published, (
        'progress_metrics computed gates/progress_done but never handed it to '
        'protocol.publish_gate -- the exit trigger cannot see it and the stage '
        'can never exit (toy_wk2_aug24/sl1ebn35)')


# ----------------------------------------------- on_exit 'stop' (owner 2026-08-26)
# Phase-1 probe fans must yield the GPU the moment their phase is done: the
# stage's exit fires, its snapshots land, and the run ENDS instead of advancing.

def engine_stop(exit_block, on_exit):
    stages = [{'name': 's0', 'train_mode': 'bwd', 'bwd_sampling_mode': 'dataset',
               'exit': exit_block, 'on_exit': on_exit}]
    args = SimpleNamespace(protocol='p', grow_batch_size=False,
                           protocols=SimpleNamespace(p=SimpleNamespace(stages=stages)))
    saved = []
    m = SimpleNamespace(
        args=args, stage='s0', stage_ctrl=fresh_stage_ctrl(),
        metric_tracker=MetricTracker(period=25.0), step_ind=0,
        combo_loss_record=[], batch_sizer=None,
        batch_size_oom_ceiling=None, batch_size_oom_ceiling_at=None,
        batch_size_oom_min=None, _runaway_last_cut=None, _runaway_unresponsive_stage=None,
        _accum_floor_warned_stage=None, batch_size_last_grow=0,
        fwd_frac=0.0, bwd_frac=1.0, replay_frac=0.0,
        init_schedulers_optimizers=lambda: None, set_loss_coeffs=lambda: None,
        lr_controller=SimpleNamespace(on_stage_change=lambda: 0),
        grad_guard=SimpleNamespace(refresh=lambda reason=None: None),
        checkpointer=SimpleNamespace(save=lambda tag, with_buffers=False: saved.append(tag)))
    return StageProtocol(m), m, saved


def test_a_single_stage_protocol_with_stop_is_legal_and_ends_the_run():
    """Before 'stop', a final stage whose exit fired was an IndexError waiting
    at the successor lookup; with it, the exit runs its snapshots and requests
    the end of the run without advancing."""
    p, m, saved = engine_stop(PROGRESS_EXIT,
                              ['snapshot:phase1_exit', 'stop'])
    m.step_ind = 2000
    p.publish_gate('progress_done', 1.0)
    p.tick()
    assert p.maybe_advance({})
    assert m._stop_requested is True
    assert p.stage.name == 's0', 'stop must not advance the stage'
    assert 'phase1_exit' in saved, (
        'the phase snapshot must land BEFORE the stop is honored -- a stop '
        'that skips it ends the run with nothing to hand to phase 2')


def test_stop_is_a_valid_action_name():
    from protocol import ACTIONS
    assert 'stop' in ACTIONS
