"""A Z-pin rollout pins log Z without feeding the replay buffer.

`fwd_ran` gates the rollout, the z_fill stash and replay admission TOGETHER, so
fwd_rollout_every moves Z-pin frequency, fresh-data rate and buffer reuse as one
knob -- and no fan over it can say which of the three drove a result. z_pin_rollout_every
splits the Z pin off: same forward cadence, more pins, and nothing extra in the
buffer.
"""
import pytest

from energy_sampling.protocol import Stage

BASE = {'name': 'equilibration', 'train_mode': 'fused', 'bwd_sampling_mode': 'prior'}


def _stage(**kw):
    return Stage({**BASE, **kw}, 0)


def test_off_by_default():
    """Everything before this ran without the key, and an absent key must mean
    the CODE DEFAULT, not a surprise extra energy call."""
    assert _stage().z_pin_rollout_every == 0
    assert _stage(fwd_rollout_every=20).z_pin_rollout_every == 0


def test_the_intended_setting_parses():
    st = _stage(fwd_rollout_every=20, z_pin_rollout_every=10)
    assert st.fwd_rollout_every == 20 and st.z_pin_rollout_every == 10


def test_it_refuses_a_cadence_with_no_interval_to_pin_inside():
    """With the forward branch on every step there is no skipped step for a z-pin
    to land on, so the key would read as configured and do nothing."""
    with pytest.raises(ValueError, match='needs fwd_rollout_every > 0'):
        _stage(z_pin_rollout_every=10)


@pytest.mark.parametrize('z', [20, 40])
def test_it_refuses_a_cadence_that_can_never_fire_off_cadence(z):
    """A z-pin only fires on steps the ordinary cadence skipped. At >= the forward
    cadence it fires on nothing -- a knob that reads as set and never runs."""
    with pytest.raises(ValueError, match='STRICTLY LESS'):
        _stage(fwd_rollout_every=20, z_pin_rollout_every=z)


def test_negative_is_refused():
    with pytest.raises(ValueError, match='must be >= 0'):
        _stage(fwd_rollout_every=20, z_pin_rollout_every=-1)


def test_the_firing_pattern_is_the_one_the_experiment_wants():
    """N=20 with a z-pin at 10 must give: ordinary rollouts at 0,20,40 and z-pins
    at 10,30,50 -- one extra pin exactly halfway between rollouts, and never a
    z-pin on a step that already rolled out."""
    every, z_every = 20, 10
    ordinary = [s for s in range(60) if s % every == 0]
    z_pins = [s for s in range(60)
              if s % every != 0 and s % z_every == 0]
    assert ordinary == [0, 20, 40]
    assert z_pins == [10, 30, 50]
    assert not set(ordinary) & set(z_pins), 'a z-pin must never double a rollout'


# ---------------------------------------------------------------------------
# The gate itself. The parse tests above pin the CONFIG; these pin the BEHAVIOUR,
# which is what the experiment rests on.
# ---------------------------------------------------------------------------

from energy_sampling.train import Modeller


def _gates(step, every=20, z_every=10, fwd_frac=0.05):
    m = Modeller.__new__(Modeller)
    m.protocol = type('P', (), {'stage': _stage(fwd_rollout_every=every,
                                                z_pin_rollout_every=z_every)})()
    m.step_ind = step
    m._cadence_anchor = 0
    m._cadence_anchor_stage = 'equilibration'
    m.fwd_frac = fwd_frac
    m._rollout_trigger_fires = lambda *_a, **_k: False
    return Modeller._fwd_gates(m, deactivate_threshold=0.01, force_refresh=False)


@pytest.mark.parametrize('step', [0, 20, 40])
def test_an_ordinary_rollout_is_not_flagged_as_a_z_pin(step):
    ran, active, z_pin = _gates(step)
    assert ran and active and not z_pin


@pytest.mark.parametrize('step', [10, 30, 50])
def test_a_z_pin_runs_the_rollout_but_carries_no_weight(step):
    """It exists for the z_fill stash. Any gradient weight would make it an
    ordinary forward step at a different cadence, which is a different experiment."""
    ran, active, z_pin = _gates(step)
    assert ran, 'the rollout must actually happen -- the stash comes from it'
    assert z_pin
    assert not active, 'a z-pin must contribute no gradient'


@pytest.mark.parametrize('step', [1, 5, 11, 19, 21])
def test_nothing_fires_off_both_cadences(step):
    ran, active, z_pin = _gates(step)
    assert not ran and not z_pin


def test_the_key_is_inert_when_unset():
    ran, active, z_pin = _gates(10, z_every=0)
    assert not ran and not z_pin, 'without the key, step 10 is an ordinary skip'


def test_z_pins_are_counted_as_energy_calls_AND_separately():
    """A z-pin IS an MLIP call. Under-reporting it in the rollout tally would make
    the knob look free; not reporting it separately would make its cost
    unattributable."""
    m = Modeller.__new__(Modeller)
    m.protocol = type('P', (), {'stage': _stage(fwd_rollout_every=20,
                                                z_pin_rollout_every=10)})()
    m._cadence_anchor, m._cadence_anchor_stage = 0, 'equilibration'
    m.fwd_frac = 0.05
    m._rollout_trigger_fires = lambda *_a, **_k: False
    for step in range(60):
        m.step_ind = step
        Modeller._fwd_gates(m, deactivate_threshold=0.01, force_refresh=False)
    assert m._rollout_count == 6, f'3 ordinary + 3 z-pin, got {m._rollout_count}'
    assert m._z_pin_count == 3, f'z-pins must be attributable, got {m._z_pin_count}'
