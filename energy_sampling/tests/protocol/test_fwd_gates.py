"""Modeller._fwd_gates -- the one place that decides whether a fused step runs
a forward ROLLOUT (fwd_ran) and whether the forward loss carries any WEIGHT
(fwd_active).

The two were separate expressions in two branches; the risk in unifying them is
that the off-cadence branch stops reproducing today's table. So the table is
pinned outright, both branches, including the force-refresh case where a branch
runs only to keep its rolling stats fresh and contributes zero weight.

The off-cadence drift trigger (stage.fwd_rollout_drift_max) ships at 0. Its
tests are the REFUSALS -- off by default, and the minimum two-step gap -- since
the gap is what stops a chronically high drift turning the cadence into
rollouts at 0 and 1 (mod N), a deterministic cadence change the key does not
claim to make.
"""
from types import MethodType, SimpleNamespace

import pytest

from energy_sampling.train import Modeller

THRESH = 0.01


def _m(every=0, fwd_frac=0.0, drift_max=0.0, drift_std=None, dormant=False):
    m = SimpleNamespace(
        step_ind=0, fwd_frac=fwd_frac, _replay_drift_std=drift_std,
        protocol=SimpleNamespace(
            stage=SimpleNamespace(fwd_rollout_every=every,
                                  fwd_rollout_drift_max=drift_max),
            mode_dormant=lambda mode: dormant))
    m._drift_trigger_fires = MethodType(Modeller._drift_trigger_fires, m)
    m._fwd_gates = MethodType(Modeller._fwd_gates, m)
    return m


# ------------------------------------------------- no cadence: today's table

@pytest.mark.parametrize('fwd_frac,force_refresh,dormant,expected', [
    (0.5, False, False, (True, True)),      # trains
    (0.5, True, False, (True, True)),
    (0.0, False, False, (False, False)),    # below threshold, no refresh due
    (0.0, True, False, (True, False)),      # force-refresh only: runs, weight 0
    (0.0, True, True, (False, False)),      # dormant branch skips its refresh
])
def test_the_uncadenced_table_is_unchanged(fwd_frac, force_refresh, dormant, expected):
    m = _m(every=0, fwd_frac=fwd_frac, dormant=dormant)
    assert m._fwd_gates(THRESH, force_refresh) == expected


# ---------------------------------------------------------------- cadence

def test_the_cadence_runs_on_its_multiples_and_never_trains():
    m = _m(every=7, fwd_frac=0.0)
    ran = []
    for step in range(70):
        m.step_ind = step
        r, active = m._fwd_gates(THRESH, force_refresh=(step % 10 == 0))
        assert active is False, 'a cadenced stage pins fwd_frac at 0'
        if r:
            ran.append(step)
    assert ran == list(range(0, 70, 7))


def test_the_force_refresh_cannot_add_a_rollout_under_a_cadence():
    """At refresh_every 10 it would call the energy function every 10 steps
    regardless of the cadence, silently."""
    m = _m(every=7)
    m.step_ind = 3
    assert m._fwd_gates(THRESH, force_refresh=True)[0] is False


def test_a_cadenced_rollout_would_train_if_the_stage_gave_fwd_weight():
    """fwd_active is DERIVED from fwd_ran, so a stage that later unpins fwd
    trains on its rollout steps rather than needing a second gate. Inert today:
    every cadenced stage ships fracs.fwd 0."""
    m = _m(every=7, fwd_frac=0.5)
    m.step_ind = 14
    assert m._fwd_gates(THRESH, False) == (True, True)
    m.step_ind = 15
    assert m._fwd_gates(THRESH, False) == (False, False)


# ------------------------------------------------------------- drift trigger

def test_the_trigger_is_off_by_default_even_at_pathological_drift():
    m = _m(every=7, drift_max=0.0, drift_std=99.0)
    ran = [s for s in range(70) if (setattr(m, 'step_ind', s) or m._fwd_gates(THRESH, False)[0])]
    assert ran == list(range(0, 70, 7))
    assert getattr(m, '_drift_trigger_count', 0) == 0


def test_the_trigger_holds_before_any_drift_has_been_measured():
    """_replay_drift_std is unset on step 1 of a fresh run and after a resume;
    a bare attribute read would abort the run, since the training loop's
    handler catches only RuntimeError/ValueError."""
    m = SimpleNamespace(
        step_ind=1, fwd_frac=0.0,
        protocol=SimpleNamespace(
            stage=SimpleNamespace(fwd_rollout_every=7, fwd_rollout_drift_max=0.3),
            mode_dormant=lambda mode: False))
    m._drift_trigger_fires = MethodType(Modeller._drift_trigger_fires, m)
    m._fwd_gates = MethodType(Modeller._fwd_gates, m)
    assert m._fwd_gates(THRESH, False) == (False, False)


def test_the_trigger_respects_the_two_step_gap():
    """The drift reading is taken BEFORE that step's admission, so without the
    gap a chronically high drift fires at step 1 of every window."""
    m = _m(every=7, drift_max=0.3, drift_std=5.0)
    ran, triggered, last = [], [], None
    for step in range(28):
        m.step_ind = step
        before = getattr(m, '_drift_trigger_count', 0)
        if m._fwd_gates(THRESH, False)[0]:
            if getattr(m, '_drift_trigger_count', 0) > before:
                assert last is not None and step - last >= 2, (step, last)
                triggered.append(step)
            ran.append(step)
            last = step
    assert set(range(0, 28, 7)) <= set(ran), 'the cadence itself is never blocked'
    assert triggered and m._drift_trigger_count == len(triggered)


def test_the_trigger_holds_on_a_drift_under_the_bar():
    m = _m(every=7, drift_max=5.0, drift_std=4.9)
    m.step_ind = 3
    assert m._fwd_gates(THRESH, False)[0] is False


def test_a_non_finite_reading_never_fires():
    m = _m(every=7, drift_max=0.3, drift_std=float('nan'))
    m.step_ind = 3
    assert m._fwd_gates(THRESH, False)[0] is False
