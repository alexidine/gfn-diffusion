"""
Exit-trigger streak semantics: `patience` counts MEASUREMENTS, not checks.

WHAT WAS WRONG. Every value source an exit term can read persists its last
value -- the metric tracker is an EMA dict, `gates/*` is a plain dict, and both
survive indefinitely after the last write. `_exit_tick` runs every 10 steps and
used to advance a streak on whatever `_resolve` returned, so a term read faster
than its metric is written counted ONE sample as N. Measured on the prod0810
phase-1 block before the fix:

    bwd/tbc written once at step 100   -> streak 20 after 20 quiet ticks
    gates/mle_flat published once      -> cleared `patience: 5` three ticks later

`eval/*` terms had the mirror-image defect: skipped by `_exit_tick` entirely, so
`protocol/exit_streak_eval_wass_debiased` was a STRUCTURAL zero at every tick of
every run -- which docs/design/next_battery.md 1.3 read as "this gate never
fires" -- while `_exit_satisfied` tested them once against the fresh eval
metrics with `patience` accepted by the parser and then silently discarded.

THE DIRECTION MATTERS, and it is the opposite of what 1.3 recorded. Patience
over-counted; it did not reset. A fix that reset the streak on a quiet tick
would make patience > 1 genuinely unreachable for any metric slower than the
tick -- it would implement the bug that was reported. Hence the three-way rule
tested here: fresh pass advances, fresh fail resets, no fresh write HOLDS.

`pytest test_protocol_exit_streaks.py -q`
"""

from types import SimpleNamespace

import pytest

from protocol import StageProtocol, fresh_stage_ctrl
from utils import MetricTracker

# torch arrives transitively through utils, but nothing here builds a model,
# runs a rollout or touches the data drive -- the whole file is dict arithmetic.
pytestmark = pytest.mark.fast

TICK = 10  # train.py: `if self.step_ind % 10 == 0: ... self.protocol.tick()`

# The literal prod0810 / mk_dev phase-1 exit block.
PHASE1_EXIT = [
    {'metric': 'gates/mle_flat', 'above': 0.5, 'patience': 5},
    {'metric': 'eval/wass_debiased', 'below': 0.015},
    {'metric': 'bwd/tbc', 'below': 2.0},
]


def engine(exit_block, second_stage=True):
    """A StageProtocol over a two-stage protocol, with a real MetricTracker.

    The tracker is the real one deliberately: `written_at` is the seam the fix
    depends on, and a fake that stamped on demand would pass these tests whether
    or not the tracker stamps anything.

    Everything a TRANSITION touches (optimizer rebuild, LR re-warm, grad-guard
    recalibration, checkpoint) is stubbed -- those are the transition's business
    and are exercised elsewhere; what matters here is whether it fires."""
    stages = [{'name': 's0', 'train_mode': 'bwd', 'bwd_sampling_mode': 'dataset',
               'exit': exit_block}]
    if second_stage:
        stages.append({'name': 's1', 'train_mode': 'bwd', 'bwd_sampling_mode': 'dataset'})
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


def run_ticks(p, m, n, *, tbc=None, mle=None):
    """`n` ticks at the real 10-step cadence, writing the named metrics on every
    tick (as a live bwd stage does) or on none (as a stalled sensor does)."""
    for _ in range(n):
        m.step_ind += TICK
        if tbc is not None:
            m.metric_tracker.update('bwd', {'tbc': tbc}, m.step_ind)
        if mle is not None:
            p.publish_gate('mle_flat', mle)
        p.tick()


# ---------------------------------------------------------------------------
# The bug: one sample counted many times
# ---------------------------------------------------------------------------

def test_a_stale_tracker_value_does_not_advance_the_streak():
    """ONE write, then twenty quiet ticks. Before the fix this reached 20 and
    would have cleared any patience; the value is still readable the whole time,
    which is exactly why `_resolve` alone cannot judge it."""
    p, m = engine(PHASE1_EXIT)
    m.step_ind = 100
    m.metric_tracker.update('bwd', {'tbc': 0.8}, 100)
    run_ticks(p, m, 20)

    assert p.ctrl['exit'].get(2, 0) == 1, 'one write is one measurement'
    assert m.metric_tracker.get('bwd', 'tbc') == pytest.approx(0.8), \
        'the value must still be READABLE -- staleness is invisible to _resolve'


def test_a_stale_gate_publish_does_not_advance_the_streak():
    """gates/* is the same dict-persistence trap. A gate that stops publishing
    leaves its last verdict behind, and a latched 'flat' verdict clearing a
    patience of 5 on its own is the swallowed-diagnostic shape."""
    p, m = engine(PHASE1_EXIT)
    p.publish_gate('mle_flat', 1.0)
    run_ticks(p, m, 30)

    assert p.ctrl['exit'].get(0, 0) == 1
    assert not p.ctrl['exit_armed'], 'patience 5 must not be cleared by one publish'


def test_patience_counts_writes_at_the_metrics_own_cadence():
    """A metric written every tick reaches patience 5 in 5 ticks -- the fix must
    not slow down the ordinary case, which is what makes the arming edge and the
    pulled-forward eval work at all."""
    p, m = engine(PHASE1_EXIT)
    run_ticks(p, m, 5, tbc=0.8, mle=1.0)

    assert p.ctrl['exit'][0] == 5
    assert p.ctrl['exit'][2] == 5
    assert p.ctrl['exit_armed'] and p.ctrl['request_eval'], \
        'all tick terms at patience must still pull the eval forward'


# ---------------------------------------------------------------------------
# Hold, not reset -- the failure the fix must not introduce
# ---------------------------------------------------------------------------

def test_a_quiet_tick_holds_the_streak_rather_than_resetting_it():
    """THE REPORTED BUG, ASSERTED ABSENT. Resetting on a tick with no fresh
    write is the obvious fix and it is wrong: it makes patience > 1 unreachable
    for every metric slower than the 10-step tick. Three clean writes, then a
    long silence, must leave the streak at 3 -- not 0."""
    p, m = engine(PHASE1_EXIT)
    run_ticks(p, m, 3, tbc=0.8)
    assert p.ctrl['exit'][2] == 3

    run_ticks(p, m, 50)  # sensor silent for 500 steps
    assert p.ctrl['exit'][2] == 3, 'silence must not destroy earned measurements'


def test_a_fresh_failing_read_does_reset_the_streak():
    """Holding applies to ABSENCE, not to failure. A metric that is written and
    misses its bar resets, or the streak stops meaning 'consecutive'."""
    p, m = engine(PHASE1_EXIT)
    run_ticks(p, m, 4, tbc=0.8)
    assert p.ctrl['exit'][2] == 4

    run_ticks(p, m, 1, tbc=9.0)  # written, and over the bar
    assert p.ctrl['exit'][2] == 0


def test_a_never_written_metric_never_satisfies_its_term():
    """A term whose metric no branch produces holds at zero rather than passing
    by default. `_resolve` returns None here, and 'no reading' must not read as
    'reading is fine'."""
    p, m = engine([{'metric': 'replay/never_written', 'below': 1.0, 'patience': 1}])
    run_ticks(p, m, 20, tbc=0.8, mle=1.0)

    assert p.ctrl['exit'].get(0, 0) == 0
    assert not p._exit_satisfied({})


# ---------------------------------------------------------------------------
# eval/* terms: patience honoured, streak real
# ---------------------------------------------------------------------------

def test_patience_on_an_eval_term_is_honoured():
    """It used to be accepted by _parse_exit and then discarded, so `patience: 5`
    on an eval metric fired on the FIRST clean eval."""
    p, m = engine([{'metric': 'eval/wass_debiased', 'below': 0.015, 'patience': 3}])

    for i in range(2):
        m.step_ind += 250
        assert not p.maybe_advance({'wass_debiased': 0.01245}), \
            f'exited after {i + 1} clean eval(s), patience is 3'

    m.step_ind += 250
    assert p.maybe_advance({'wass_debiased': 0.01245})
    assert m.stage == 's1'


def test_an_eval_term_streak_is_logged_rather_than_pinned_at_zero():
    """THE MISREAD THAT STARTED THIS. `protocol/exit_streak_eval_wass_debiased`
    was a structural zero -- eval terms were skipped by the tick loop, so the
    series said 'never passes' when it meant 'never judged here'. The streak is
    now real, and `exit_age_*` reports how long since the term last had a value
    to judge, which is what separates the two readings."""
    p, m = engine([{'metric': 'eval/wass_debiased', 'below': 0.015, 'patience': 5}])
    m.step_ind = 500
    p.maybe_advance({'wass_debiased': 0.01245})

    logged = p.report()
    assert logged['protocol/exit_streak_eval_wass_debiased'] == 1
    assert logged['protocol/exit_age_eval_wass_debiased'] == 0.0

    m.step_ind = 900  # 400 steps on with no eval
    assert p.report()['protocol/exit_age_eval_wass_debiased'] == 400.0


def test_an_eval_metric_missing_from_one_eval_holds_the_streak():
    """Not every eval computes every metric. An absent key is a
    non-measurement, and the hold rule applies to it exactly as it does to a
    quiet tick -- silently resetting here would be the reported bug, arriving
    through the eval path instead."""
    p, m = engine([{'metric': 'eval/wass_debiased', 'below': 0.015, 'patience': 3}])
    m.step_ind = 250
    p.maybe_advance({'wass_debiased': 0.01245})
    assert p.ctrl['exit'][0] == 1

    m.step_ind = 500
    p.maybe_advance({})                      # wass not computed this eval
    assert p.ctrl['exit'][0] == 1, 'an absent metric must hold, not reset'

    m.step_ind = 750
    p.maybe_advance({'wass_debiased': 0.02})  # present, over the bar
    assert p.ctrl['exit'][0] == 0, 'a fresh failing read must reset'


# ---------------------------------------------------------------------------
# The whole trigger, end to end
# ---------------------------------------------------------------------------

def test_the_prod0810_phase1_block_still_exits_when_all_three_are_met():
    """THE POSITIVE PATH. Every test above tightens a condition, and a fix that
    tightened them into a stage that can never advance would satisfy all of them
    while being worse than the bug. The real three-term block, driven at the
    real cadences, must still fire."""
    p, m = engine(PHASE1_EXIT)
    run_ticks(p, m, 5, tbc=0.8, mle=1.0)
    assert p.ctrl['request_eval'], 'tick terms armed -> eval pulled forward'

    assert p.maybe_advance({'wass_debiased': 0.01245})
    assert m.stage == 's1'
    assert not p.ctrl['request_eval'], 'the pulled-forward request is consumed'


def test_the_stage_does_not_exit_while_one_term_is_unmet():
    """The AND-list is an AND-list: wass over its bar holds the transition even
    with both tick terms long since at patience."""
    p, m = engine(PHASE1_EXIT)
    run_ticks(p, m, 20, tbc=0.8, mle=1.0)

    assert not p.maybe_advance({'wass_debiased': 0.02})
    assert m.stage == 's0'


def test_a_transition_resets_the_freshness_bookkeeping():
    """`exit_seen` rides in stage_ctrl, which is replaced wholesale at a
    transition. A stamp surviving the boundary would let the incoming stage's
    first tick judge the outgoing stage's last measurement."""
    p, m = engine(PHASE1_EXIT)
    run_ticks(p, m, 5, tbc=0.8, mle=1.0)
    assert p.ctrl['exit_seen']

    p.maybe_advance({'wass_debiased': 0.01245})
    assert m.stage_ctrl['exit_seen'] == {}
    assert m.stage_ctrl['exit'] == {}
