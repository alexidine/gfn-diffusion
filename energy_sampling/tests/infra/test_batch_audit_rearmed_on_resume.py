"""A pending S2 occupancy audit is re-armed on resume, not fired.

The sizer holds a grown rung with `audit_at = now + gpu_util_policy_window_s` (train.select_batch_size), a wall-clock
deadline, and the audit compares the occupancy LIVED over that window with the base rung's calibration. The whole
`batch_sizer` dict rides in the checkpoint. A requeued leg starts hours after the stored deadline with an empty
occupancy record, so the audit used to fire on the first step, read start-up idle, fail, and stand the batch down to
the base rung for the rest of the stage (`stood_down` is never re-probed). Seen on mlepl_neh_lr2, 2026-10-06:
"a full policy window at 1600 reads 0.7%" -> 1000, then 2.4 h at 51% occupancy.

`Checkpointer.reconcile_batch_size` (called by set_state_dict on every full load) moves a pending deadline to one
policy window after the resume.
"""
import types

import pytest

pytest.importorskip("torch")

from energy_sampling.checkpointing import Checkpointer  # noqa: E402

NOW = 1_000_000.0


def _modeller(sizer, batch=1600, grow=True, window=7200):
    args = types.SimpleNamespace(batch_size=1000, max_batch_size=4000, grow_batch_size=grow,
                                 gpu_util_policy_window_s=window)
    return types.SimpleNamespace(args=args, batch_size=batch, batch_sizer=sizer, batch_size_last_grow=0,
                                 batch_size_cooldown_until=-1, _now=lambda: NOW)


def _hold(audit_at, reason='target_met'):
    return dict(phase='hold', reason=reason, selected=1600, audit_at=audit_at,
                table=[dict(batch=1000, util=53.3), dict(batch=1600, util=66.8)])


def test_a_stale_deadline_moves_one_policy_window_past_the_resume():
    m = _modeller(_hold(audit_at=NOW - 15 * 3600))          # the leg was requeued 15 h after the checkpoint
    Checkpointer(m).reconcile_batch_size()
    assert m.batch_sizer['audit_at'] == NOW + 7200
    assert m.batch_size == 1600 and m.batch_sizer['reason'] == 'target_met', 'the selection itself is untouched'
    assert m._now() < m.batch_sizer['audit_at'], 'so the audit cannot fire on the first step of the leg'


def test_a_deadline_still_in_the_future_is_also_moved():
    """A quick requeue leaves the stored deadline ahead of the clock, but the window it closes would still be mostly
    the previous process, which this one has no record of."""
    m = _modeller(_hold(audit_at=NOW + 600), window=3600)
    Checkpointer(m).reconcile_batch_size()
    assert m.batch_sizer['audit_at'] == NOW + 3600


@pytest.mark.parametrize('sizer', [None, _hold(audit_at=None), _hold(audit_at=None, reason='stood_down'),
                                   dict(phase='calibrating', reason=None, selected=1000, table=[], audit_at=None)])
def test_nothing_is_armed_that_was_not_pending(sizer):
    before = None if sizer is None else dict(sizer)
    m = _modeller(sizer)
    Checkpointer(m).reconcile_batch_size()
    assert m.batch_sizer == before


def test_growth_off_still_drops_the_conclusion():
    m = _modeller(_hold(audit_at=NOW - 10), grow=False)
    Checkpointer(m).reconcile_batch_size()
    assert m.batch_sizer is None and m.batch_size == 1000
