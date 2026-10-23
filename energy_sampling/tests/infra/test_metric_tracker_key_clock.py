"""
Every MetricTracker key decays on its OWN clock, so the 'replay' direction's two
10-step writers run at dt = 10 whatever their phase, before and after a resume.

The writers: the replay step's payload (absorption, policy drift, held-out gap),
on step_ind % 10 inside replay_train_step, and the rolling stats, written after
it by record_fused_substep_losses on replay_step_count % 10. Under the old
per-direction clock their phase set the smoothing. A parent with
replay_step_count = step_ind (mod 10) wrote both in one step, payload first, so
rolling keys landed at dt = 1 and the payload at dt = 10. A resume re-runs the
saved step and advances every *_step_count once more. p20_p20_mip_n5's
step240000 archive has replay_step_count % 10 = 0, and p23_p23_mip_ft, resumed
from it, has 1 at every archive. That moves the rolling write to
step_ind = 9 (mod 10): rolling dt = 9, payload dt = 1.

The dt a write used is read back off the EMA itself: with prev and v known,
alpha = (new - prev) / (v - prev) and dt = -period * log(1 - alpha).

    pytest tests/infra/test_metric_tracker_key_clock.py
"""
import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from energy_sampling.utils import MetricTracker

PERIOD = 100          # train.py: MetricTracker(period=100)
CADENCE = 10          # both replay writers' gate


def _write(tracker, log, direction, name, step):
    """One write, recording the dt the EMA actually applied. Values alternate
    0 / 100 so v never equals prev and alpha is always recoverable."""
    key = (direction, name)
    n = len(log.setdefault(key, []))
    v = 100.0 * (n % 2)
    prev = tracker.get(direction, name)
    tracker.update(direction, {name: v}, step)
    new = tracker.get(direction, name)
    if prev is None:
        dt = None
    else:
        alpha = (new - prev) / (v - prev)
        dt = -PERIOD * math.log(1.0 - alpha)
    log[key].append((step, dt))


def _steady(log, key, skip):
    return [round(dt, 6) for _, dt in log[key][skip:]]


@pytest.mark.parametrize('offset', range(CADENCE))
@pytest.mark.parametrize('b_first', [False, True])
def test_two_writers_on_one_direction_each_decay_on_their_own_cadence(offset, b_first):
    t, log = MetricTracker(period=PERIOD), {}
    for step in range(0, 400):
        writes = []
        if step % CADENCE == 0:
            writes.append('a')
        if step % CADENCE == offset:
            writes.append('b')
        for w in (reversed(writes) if b_first else writes):
            _write(t, log, 'replay', w, step)
    for key in log:
        assert _steady(log, key, skip=1) == [float(CADENCE)] * (len(log[key]) - 1), key


def _trainer(tracker, step_ind, replay_step_count, log):
    """The two real gates around a stub modeller. The rolling side is
    train.py's own record_fused_substep_losses; _update_rolling is reduced to
    its last line (one tracker.update on the replay direction)."""
    from energy_sampling.train import Modeller

    m = SimpleNamespace(metric_tracker=tracker, step_ind=step_ind,
                        replay_step_count=replay_step_count)
    m._per_step_probe = lambda loss_dict, sub_type: None
    m._update_rolling = lambda loss_dict, sub_loss, sub_type: _write(
        m.metric_tracker, log, sub_type, 'tb_err', m.step_ind)
    m.record = lambda sub_losses: Modeller.record_fused_substep_losses(m, sub_losses)
    return m


def _run(m, steps, log):
    for m.step_ind in steps:
        # replay_train_step: the payload, on step_ind % 10, inside the step
        if m.step_ind % CADENCE == 0:
            _write(m.metric_tracker, log, 'replay', 'val_gap_nats', m.step_ind)
        # train_step's tail: the rolling stats, on replay_step_count % 10
        m.record({'replay': (None, {}, True)})


def test_replay_rolling_and_payload_intervals_survive_a_resume():
    save_step = 500   # a 'running' save: step_ind % 50 == 0, end of the step

    parent_log = {}
    parent = _trainer(MetricTracker(period=PERIOD), 0, 0, parent_log)
    _run(parent, range(1, save_step + 1), parent_log)
    ck = pickle.loads(pickle.dumps({
        'step_ind': parent.step_ind,
        'replay_step_count': parent.replay_step_count,
        'metrics': parent.metric_tracker.state_dict()}))

    # the resume path: tracker from the checkpoint, counters restored, and the
    # loop restarting AT the restored step_ind (train.py: trange(init_step, ...))
    leg_log = {}
    tracker = MetricTracker(period=PERIOD)
    tracker.load_state_dict(ck['metrics'])
    leg = _trainer(tracker, ck['step_ind'], ck['replay_step_count'], leg_log)
    _run(leg, range(ck['step_ind'], 2 * save_step), leg_log)

    rolling, payload = ('replay', 'tb_err'), ('replay', 'val_gap_nats')

    # the scenario reproduces the observed phases, or the test proves nothing
    assert {s % CADENCE for s, _ in parent_log[rolling]} == {0}
    assert {s % CADENCE for s, _ in parent_log[payload]} == {0}
    assert {s % CADENCE for s, _ in leg_log[rolling]} == {CADENCE - 1}
    assert {s % CADENCE for s, _ in leg_log[payload]} == {0}

    ten = float(CADENCE)
    for key in (rolling, payload):
        before = _steady(parent_log, key, skip=1)
        after = _steady(leg_log, key, skip=1)
        assert before == [ten] * len(before), (key, sorted(set(before)))
        assert after == [ten] * len(after), (key, sorted(set(after)))
        # the first write after the resume realigns the phase within one window
        first = leg_log[key][0][1]
        assert 1.0 - 1e-9 <= first <= ten + 1e-9, (key, first)


def test_per_direction_checkpoint_layout_loads_onto_every_key():
    t = MetricTracker(period=PERIOD)
    t.load_state_dict({'period': PERIOD, 'last_it': {'replay': 100},
                       'values': {'replay': {'tb_err': 0.0, 'val_gap_nats': 0.0}},
                       'best': {}})
    t.update('replay', {'tb_err': 1.0, 'val_gap_nats': 1.0}, 110)
    alpha = 1.0 - np.exp(-10 / PERIOD)
    assert t.get('replay', 'tb_err') == pytest.approx(alpha)
    assert t.get('replay', 'val_gap_nats') == pytest.approx(alpha)
    assert t.state_dict()['last_it'] == {'replay': {'tb_err': 110, 'val_gap_nats': 110}}


def test_a_skipped_non_finite_write_does_not_advance_the_clock():
    t = MetricTracker(period=PERIOD)
    t.update('replay', {'x': 0.0}, 0)
    t.update('replay', {'x': float('nan')}, 10)
    t.update('replay', {'x': 1.0}, 20)
    assert t.get('replay', 'x') == pytest.approx(1.0 - np.exp(-20 / PERIOD))
