"""
The OOM ceiling must EXPIRE, and a pin below the batch floor must not deadlock.

WHAT KILLED THREE RUNS. prod0810's acridine/mace arms (indices 4-6) were cancelled
by the scheduler for low GPU utilization while still in `train_prior` -- a stage that
makes no MLIP calls at all (the reward is a stored attribute on the prior rows), so
occupancy is a pure function of batch size. The batch never grew. One OOM at the BASE
batch of 1000 was enough:

    OOM at 1000  ->  ceiling = 1000, batch = 500      (handle_train_epoch_error)
    walk         ->  500 -> 825                        (825 < 1000, allowed)
    walk         ->  825 -> 1361 REFUSED               (1361 >= ceiling) -> PIN
    knee recheck ->  drop target = max(floor 1000, 500) = 1000 >= 825 -> return

...and that last line returned WITHOUT re-arming batch_size_pinned_at, so the recheck
branch was re-entered every step and could never fire again. The ceiling was a
permanent conclusion, the pin sat below _batch_floor() where the recheck could not
reach it, and the arm ran the rest of the stage at 0.825x its configured batch.

Both halves are tested here, and both are tested by REPRODUCING THE FAILURE, not by
asserting that the fixed code does what it does: each test drives the real
increment_batch_size / handle_train_epoch_error through the exact prod0810 numbers and
requires the batch to come back UP. Reverting either fix must turn these red -- see
the module docstring in bench/README.md on what a blind batch test looks like.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.fake_modeller import (FakeModeller, FakeStage, attach_real_batch_sizer,
                                 make_args)

attach_real_batch_sizer()

# prod0810/4.yaml (acridine_sg14_zp1_mace), the batch controller's view of it
PROD0810 = dict(batch_size=1000, max_batch_size=50000, grow_batch_size=True,
                batch_growth_factor=1.65, batch_growth_interval=50,
                batch_growth_slow_interval=300, auto_batch_throughput_opt=True,
                batch_growth_min_throughput_gain=0.05, max_step_seconds=300,
                batch_knee_recheck_steps=2000, oom_batch_shrink_factor=0.5,
                oom_cooldown_steps=200, batch_oom_ceiling_retest_steps=1000)

#: DERIVED, never hardcoded below. This mirrors a shipping config, and the shipping
#: value has already moved once (2000 -> 1000). A test that hardcodes the window
#: silently stops testing the boundary it was written for the next time the config is
#: retuned -- it keeps passing while measuring nothing.
RETEST = PROD0810['batch_oom_ceiling_retest_steps']


# Two step-cost models, because "did the batch recover" has a different right answer
# under each and a test that only knows one of them proves very little.
#
#   FLAT      step time strictly proportional to batch => samples/sec constant. The
#             knee has no reason of its own to climb, so the configured batch_size is
#             the correct resting place and anything the batch does here is the
#             ceiling logic doing it, not a throughput gradient rescuing the run.
#             This is close to what train_prior actually is on this route: no energy
#             call, step dominated by host-side batch collation.
#   HEADROOM  a fixed per-step overhead plus linear cost => samples/sec rises with
#             batch and saturates. This is the case the acridine arms NEEDED to
#             reach, and the only one that can show a climb back through the ceiling.
FLAT = lambda b: b / 1000.0
HEADROOM = lambda b: 0.05 + b / 20000.0          # knee near batch ~7400


def _drive(m, steps, cost=FLAT):
    """Run the real controller for `steps` steps against a step-cost model."""
    for _ in range(steps):
        m.step_ind += 1
        m._recent_step_times.append(cost(m.batch_size))
        m._recent_step_work.append(m.batch_size)
        m.increment_batch_size()
    return m.batch_size


def _fresh():
    m = FakeModeller(make_args(**PROD0810),
                     optimizers={}, stage=FakeStage(name='train_prior',
                                                    train_mode='bwd'))
    m.batch_size = 1000
    return m


def test_single_oom_does_not_cost_the_stage():
    """The prod0810 chain end to end on the FLAT surface: one OOM at the base batch
    must not leave the stage running below its configured batch_size. Before the fix
    this parked at 825 forever."""
    m = _fresh()
    _drive(m, 60)                       # measure a rung at the base batch

    m.handle_train_epoch_error(RuntimeError('CUDA out of memory.'), 'bwd')
    assert m.batch_size == 500, m.batch_size
    assert m.batch_size_oom_ceiling == 1000, m.batch_size_oom_ceiling

    # THE CEILING MUST BIND FIRST, or "recovery" proves nothing. Sampled across the
    # window rather than asserted at one fixed step count, so retuning
    # batch_oom_ceiling_retest_steps cannot turn this into a test that passes without
    # ever observing the state it names.
    pinned_at_some_point = False
    for _ in range(max(1, RETEST // 50)):
        _drive(m, 50)
        if m.batch_size_oom_ceiling is not None and m.batch_size <= 825:
            pinned_at_some_point = True
    assert pinned_at_some_point, (
        'the ceiling never bound, so the recovery below demonstrates nothing')

    # ...but the ceiling EXPIRES, and the walk re-probes past it
    recovered = _drive(m, 8000)
    assert m.batch_size_oom_ceiling is None, 'ceiling never expired'
    assert recovered >= 1000, (
        f'batch stuck at {recovered} after the ceiling expired -- below the '
        f'configured batch_size of 1000. This is the prod0810 acridine/mace '
        f'failure: a stage judged by GPU occupancy running under-sized for its life')


def test_recovery_climbs_back_through_the_ceiling_when_there_is_headroom():
    """The case the acridine arms needed. With real throughput headroom above the
    OOM'd size, an expired ceiling must let the walk climb well past it -- recovering
    to exactly batch_size would still leave the card mostly idle."""
    m = _fresh()
    _drive(m, 60, cost=HEADROOM)
    m.handle_train_epoch_error(RuntimeError('CUDA out of memory.'), 'bwd')
    assert m.batch_size_oom_ceiling == 1000

    pinned = _drive(m, max(600, RETEST - 200), cost=HEADROOM)
    assert pinned <= 825, f'ceiling should bind first, got {pinned}'

    recovered = _drive(m, 20000, cost=HEADROOM)
    assert recovered > 2000, (
        f'batch only reached {recovered} against a knee near 7400 -- the expired '
        f'ceiling is still throttling the walk')


def test_ceiling_stands_while_ooms_keep_happening():
    """Expiry must not become a licence to re-OOM on a cadence. A ceiling that keeps
    being re-confirmed keeps standing, because every OOM restarts the clock."""
    m = _fresh()
    _drive(m, 60)
    for _ in range(6):
        m.handle_train_epoch_error(RuntimeError('CUDA out of memory.'), 'bwd')
        _drive(m, RETEST - 200)         # always short of the retest window
        assert m.batch_size_oom_ceiling is not None, (
            'ceiling expired while OOMs were still being observed')
    assert m.batch_size <= 1000


def test_pin_below_the_floor_rearms_instead_of_spinning():
    """The second half: a pin BELOW _batch_floor() must re-arm the walk upward. The
    old code returned early, leaving batch_size_pinned_at stale, so the recheck
    branch was re-entered every step and never fired again."""
    m = _fresh()
    m.batch_size = 825                  # where the prod0810 walk parked
    m.batch_size_saturated_stage = 'train_prior'
    m.batch_size_pinned_at = 0
    m.batch_size_oom_ceiling = None     # isolate the deadlock from the expiry
    m.step_ind = 0
    assert m._batch_floor() == 1000, 'test assumes the pin sits below the floor'

    _drive(m, 2100)                     # past batch_knee_recheck_steps
    assert m.batch_size_saturated_stage is None, (
        'still pinned after the recheck window -- the walk never re-armed')
    assert m.batch_size_pinned_at > 0, (
        'batch_size_pinned_at was left stale, so the recheck can never fire again')
    assert _drive(m, 2000) > 825, 'batch never recovered from a sub-floor pin'


def test_restored_ceiling_serves_its_window_from_the_resume():
    """A resume brings back the ceiling; the CLOCK may not come with it. Measured
    against an absent clock read as step 0, a run resuming at step 20000 expires its
    ceiling on the first post-resume step and walks straight back into the OOM it was
    checkpointed to remember. `None` must mean unstamped, never step 0.

    This is bench/old/test_A1's scenario, kept here because the expiry is what makes
    it reachable again."""
    m = _fresh()
    m.step_ind = 20000                  # a long-running run, resumed
    m.batch_size = 500
    m.batch_size_oom_ceiling = 1000
    m.batch_size_oom_ceiling_at = None  # what a restore leaves behind
    m.batch_size_ever_oomed = True

    _drive(m, RETEST - 400, cost=HEADROOM)   # well inside the retest window
    assert m.batch_size_oom_ceiling == 1000, (
        'restored ceiling expired immediately -- the clock was read as step 0')
    assert m.batch_size <= 1000, 'resumed run climbed into the restored OOM ceiling'

    _drive(m, 3000, cost=HEADROOM)      # ...and past it, it expires normally
    assert m.batch_size_oom_ceiling is None


def test_growth_is_unchanged_when_nothing_ooms():
    """Negative control: on a clean run the new code must be inert. A test that only
    proves the escape hatch opens says nothing about whether it opens too often."""
    m = _fresh()
    _drive(m, 4000)
    assert m.batch_size_oom_ceiling is None
    assert not m.batch_size_ever_oomed
    # flat throughput => the knee pins, and the floor holds it at the configured batch
    assert m.batch_size >= 1000, m.batch_size


def test_ooms_and_expiries_are_counted_for_the_history_stream():
    """The controller's account of itself must reach wandb HISTORY, not just stdout.

    A run killed hard -- scancel, node loss, the scheduler cancelling on occupancy --
    is left in state 'crashed', and wandb uploads NO console log for those. That is
    exactly the run whose postmortem needs to know how many times it OOM'd and at what
    size, so the counters behind `batch/oom_events` and `batch/ceiling_expiries` are
    load-bearing diagnostics, not decoration. Counted cumulatively because the reporter
    fires every ten steps and an event landing between two reports must still show as a
    step change rather than being missed."""
    m = _fresh()
    _drive(m, 60)
    assert getattr(m, 'batch_oom_events', 0) == 0, 'counted an OOM that never happened'

    m.handle_train_epoch_error(RuntimeError('CUDA out of memory.'), 'bwd')
    assert getattr(m, 'batch_oom_events', 0) == 1, (
        'an OOM did not reach batch_oom_events -- with no console log on a crashed '
        'run, this event would leave no trace anywhere')
    assert m.batch_size_oom_ceiling == 1000

    # ...and the expiry that follows is its own event, separately counted: "the ceiling
    # was cleared and re-probed" and "the probe failed" are different conclusions and a
    # postmortem has to be able to tell them apart.
    assert getattr(m, 'batch_ceiling_expiries', 0) == 0
    _drive(m, 3000)
    assert m.batch_size_oom_ceiling is None, 'test assumes the ceiling expired'
    assert getattr(m, 'batch_ceiling_expiries', 0) == 1, (
        'the ceiling expired without incrementing batch_ceiling_expiries')

    # eval OOMs count too -- they withhold the CEILING, not the record of the failure
    m.handle_train_epoch_error(RuntimeError('CUDA out of memory.'), 'eval_fwd')
    assert getattr(m, 'batch_oom_events', 0) == 2, 'an eval OOM went unrecorded'


if __name__ == '__main__':
    # keeps going after a failure ON PURPOSE: these are mutation-tested, and
    # "which tests does reverting fix X turn red" is unanswerable if the runner
    # stops at the first one.
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except AssertionError as e:
                failures.append(name)
                print(f'FAIL {name}: {e}')
    print(f'\n{len(failures)} failed' if failures else '\nall passed')
    sys.exit(1 if failures else 0)
