"""
The OOM ceiling must EXPIRE, and a cut batch must find its way back to the base.

WHAT KILLED THREE RUNS. prod0810's acridine/mace arms (indices 4-6) were cancelled
by the scheduler for low GPU utilization while still in `train_prior` -- a stage that
makes no MLIP calls at all (the reward is a stored attribute on the prior rows), so
occupancy is a pure function of batch size. One OOM at the BASE batch of 1000 was
enough: the ceiling latched at 1000, the batch parked below the configured base, and
the arm ran the rest of the stage under-sized on a stage judged by occupancy.

Under the state-8 replacement (train.select_batch_size) there is no growth walk, so
the same failure has a NEW way to happen by omission: an OOM cuts the batch to 500,
and with nothing regrowing, expiry of the ceiling would change nothing unless the
controller explicitly RESTORES the base. These tests drive the real
select_batch_size / handle_train_epoch_error through the prod0810 numbers and
require the batch to come back to the configured base -- reverting either the expiry
or the restore rule must turn them red.

Two deliberate behaviour changes from the old controller are pinned here rather than
papered over:

  * recovery goes to the BASE, never past it. The old walk climbed toward a
    samples/sec knee; the objective decision (phase6_batch_sizer.md section 0.0)
    makes the throughput optimum the constant base batch, so "recovered" now means
    "back at batch_size", not "climbing".
  * while a ceiling stands AT OR BELOW the base, the controller holds the cut size
    -- the base itself is the size that OOM'd, and re-approaching it before the
    ceiling expires would be growing blind.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.fake_modeller import (FakeModeller, FakeStage, attach_real_batch_sizer,
                                 make_args)

attach_real_batch_sizer()

# prod0810/4.yaml (acridine_sg14_zp1_mace), the batch controller's view of it.
# The retired walk keys (auto_batch_throughput_opt & co.) are gone from make_args,
# which is itself part of the point: the config surface no longer carries them.
PROD0810 = dict(batch_size=1000, max_batch_size=50000, grow_batch_size=True,
                batch_growth_factor=1.65, batch_growth_interval=50,
                max_step_seconds=300, oom_batch_shrink_factor=0.5,
                oom_cooldown_steps=200, batch_oom_ceiling_retest_steps=1000,
                # prod0810 predates the occupancy ladder: no target existed.
                # Pinned explicitly since the canonical default became 60
                # (2026-08-19) -- this file's premise is the no-target regime.
                batch_util_target=0)

#: DERIVED, never hardcoded below. This mirrors a shipping config, and the shipping
#: value has already moved once (2000 -> 1000). A test that hardcodes the window
#: silently stops testing the boundary it was written for the next time the config is
#: retuned -- it keeps passing while measuring nothing.
RETEST = PROD0810['batch_oom_ceiling_retest_steps']

# step cost strictly proportional to batch. The cost model no longer changes what
# recovery means -- the controller reads no throughput gradient -- but the deques
# still need plausible timings for the runaway guard's median.
FLAT = lambda b: b / 1000.0


def _drive(m, steps, cost=FLAT):
    """Run the real controller for `steps` steps against a step-cost model."""
    for _ in range(steps):
        m.step_ind += 1
        m._recent_step_times.append(cost(m.batch_size))
        m._recent_step_work.append(m.batch_size)
        m.select_batch_size()
    return m.batch_size


def _fresh():
    m = FakeModeller(make_args(**PROD0810),
                     optimizers={}, stage=FakeStage(name='train_prior',
                                                    train_mode='bwd'))
    m.batch_size = 1000
    return m


def test_single_oom_does_not_cost_the_stage():
    """The prod0810 chain end to end: one OOM at the base batch must not leave the
    stage running below its configured batch_size forever. The old controller parked
    at 825 for the stage's life; the new one must hold the cut size only while the
    ceiling stands, then restore the base when it expires."""
    m = _fresh()
    _drive(m, 60)

    m.handle_train_epoch_error(RuntimeError('CUDA out of memory.'), 'bwd')
    assert m.batch_size == 500, m.batch_size
    assert m.batch_size_oom_ceiling == 1000, m.batch_size_oom_ceiling

    # THE CEILING MUST BIND FIRST, or "recovery" proves nothing. The base (1000) IS
    # the size that OOM'd, so while the ceiling stands the controller must hold the
    # cut size rather than re-approach it. Sampled across the window rather than
    # asserted at one fixed step count.
    held_below = False
    for _ in range(max(1, RETEST // 50)):
        _drive(m, 50)
        if m.batch_size_oom_ceiling is not None and m.batch_size < 1000:
            held_below = True
    assert held_below, (
        'the ceiling never bound, so the recovery below demonstrates nothing')

    # ...but the ceiling EXPIRES, and the restore rule brings the base back
    recovered = _drive(m, 3000)
    assert m.batch_size_oom_ceiling is None, 'ceiling never expired'
    assert recovered == 1000, (
        f'batch at {recovered} after the ceiling expired -- the configured base is '
        f'1000. Below it is the prod0810 acridine/mace failure (a stage judged by '
        f'GPU occupancy running under-sized for its life); above it is growth no '
        f'objective asked for.')


def test_recovery_stops_at_the_base_not_a_knee():
    """THE DELIBERATE INVERSION of the old headroom test, which required the walk to
    climb well past the expired ceiling toward a samples/sec knee. Under the decided
    objective (steps/sec at effective batch = the accum target) the throughput
    optimum is the CONSTANT base, so recovery must stop exactly there -- climbing
    past it is the retired objective reasserting itself."""
    m = _fresh()
    _drive(m, 60)
    m.handle_train_epoch_error(RuntimeError('CUDA out of memory.'), 'bwd')
    assert m.batch_size_oom_ceiling == 1000

    recovered = _drive(m, RETEST + 3000)
    assert recovered == 1000, (
        f'recovered to {recovered}, not the configured base 1000 -- with no '
        f'batch_util_target set, nothing may move the batch off the base')


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
    assert m.batch_size < 1000


def test_sub_base_batch_with_no_ceiling_is_restored():
    """A batch parked below the base with NO ceiling standing (the shape a resume or
    a stale cut can leave behind) must be restored to the base promptly -- there is
    no evidence against the base, so running under it serves nothing."""
    m = _fresh()
    m.batch_size = 825                  # where the prod0810 walk parked
    m.batch_sizer = None
    m.batch_size_oom_ceiling = None
    m.step_ind = 0
    assert m._batch_floor() == 1000, 'test assumes the batch sits below the base'

    _drive(m, 5)
    assert m.batch_size == 1000, (
        f'batch still {m.batch_size} with no ceiling standing -- the restore rule '
        f'never fired')


def test_ceiling_above_the_base_does_not_block_the_restore():
    """The restore must distinguish 'the base OOMd' from 'something above the base
    OOMd'. A ceiling at 1200 is no evidence against running at 1000."""
    m = _fresh()
    m.batch_size = 600
    m.batch_sizer = None
    m.batch_size_oom_ceiling = 1200
    m.batch_size_oom_ceiling_at = None
    m.step_ind = 0
    _drive(m, 5)
    assert m.batch_size == 1000, (
        f'batch held at {m.batch_size} under a ceiling of 1200 -- the base 1000 is '
        f'inside the domain and should be restored')


def test_restored_ceiling_serves_its_window_from_the_resume():
    """A resume brings back the ceiling; the CLOCK may not come with it. Measured
    against an absent clock read as step 0, a run resuming at step 20000 expires its
    ceiling on the first post-resume step and walks straight back into the OOM it was
    checkpointed to remember. `None` must mean unstamped, never step 0."""
    m = _fresh()
    m.step_ind = 20000                  # a long-running run, resumed
    m.batch_size = 500
    m.batch_size_oom_ceiling = 1000
    m.batch_size_oom_ceiling_at = None  # what a restore leaves behind

    _drive(m, RETEST - 400)             # well inside the retest window
    assert m.batch_size_oom_ceiling == 1000, (
        'restored ceiling expired immediately -- the clock was read as step 0')
    assert m.batch_size < 1000, 'resumed run climbed into the restored OOM ceiling'

    _drive(m, 3000)                     # ...and past it, it expires normally
    assert m.batch_size_oom_ceiling is None
    assert m.batch_size == 1000


def test_holds_the_base_when_nothing_ooms():
    """Negative control: on a clean run with no batch_util_target the controller is
    a HOLD -- the batch never leaves the configured base, in either direction."""
    m = _fresh()
    _drive(m, 4000)
    assert m.batch_size_oom_ceiling is None
    assert m.batch_size == 1000, m.batch_size
    assert m.batch_sizer is not None and m.batch_sizer['reason'] == 'no_target'


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


def test_repeated_eval_ooms_shrink_the_eval_draw_and_terminate():
    """THE TEST THE FIRST ATTEMPT AT THIS NEEDED AND DID NOT HAVE.

    Every eval retry loop draws at eval_draw_size() and retries on OOM, so the
    ONLY thing that makes a retry smaller is the handler. A first version of this
    change stopped cutting on eval OOMs without giving eval its own size; the
    loops then spun forever at an unchanged draw -- 100% GPU, no further logging,
    eight hours burned on p4_mace_mle before it was noticed.

    So the property under test is TERMINATION, not just "the train batch is
    untouched": the draw must strictly decrease, and once it can no longer
    decrease the handler must RAISE rather than let the caller loop again."""
    m = _fresh()
    _drive(m, 60)
    train_batch = m.batch_size

    draws, guard = [m.eval_draw_size()], 0
    while guard < 200:
        guard += 1
        try:
            m.handle_train_epoch_error(RuntimeError('CUDA out of memory.'), 'eval_bwd')
        except RuntimeError as e:
            assert 'spin here forever' in str(e), f'unexpected raise: {e}'
            break
        d = m.eval_draw_size()
        assert d < draws[-1], (
            f'eval draw did not shrink: {draws[-1]} -> {d}; the retry loop would '
            f'reissue the same allocation forever')
        draws.append(d)
    else:
        raise AssertionError('handler never raised; the eval loop cannot terminate')

    assert draws[-1] == 1, draws
    assert m.batch_size == train_batch, (
        f'the train batch moved {train_batch} -> {m.batch_size} on eval OOMs')


def test_an_eval_oom_does_not_cut_the_training_batch():
    """AN EVAL OVERFLOW IS EVIDENCE ABOUT A DIFFERENT ALLOCATION.

    On the MLIP routes `train_prior` makes no energy call at all, so MLE can hold
    a large batch, while eval scores eval_num_samples through the MLIP every
    eval_period steps. Cutting the TRAIN batch when an EVAL pass overflows made
    the two compete: measured on p4_mace_mle (2026-08-21) the ladder climbed
    25 -> 400, an eval overflowed, the train batch was cut to 156, and the cycle
    repeated for 11 OOM events in two hours.

    The ceiling was already withheld for eval; this pins the CUT as well. Both
    directions are asserted, because a handler that never cut anything would pass
    the eval half on its own."""
    m = _fresh()
    _drive(m, 60)
    base = m.batch_size

    # eval: counted, but the training batch and the sizer's work are untouched
    m.handle_train_epoch_error(RuntimeError('CUDA out of memory.'), 'eval_fwd')
    assert getattr(m, 'batch_oom_events', 0) == 1, 'the eval OOM went unrecorded'
    assert m.batch_size == base, (
        f'an eval OOM cut the train batch {base} -> {m.batch_size}')
    assert m.batch_size_oom_ceiling is None, 'an eval OOM installed a ceiling'

    # train: the cut still applies, or this test would pass on a handler that
    # simply stopped cutting
    m.handle_train_epoch_error(RuntimeError('CUDA out of memory.'), 'bwd')
    assert m.batch_size < base, (
        f'a TRAIN OOM failed to cut the batch (still {m.batch_size})')
    assert m.batch_size_oom_ceiling == base, m.batch_size_oom_ceiling


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


def test_set_max_batch_size_clamps_the_live_batch():
    """THE ACTION IS A TRANSITION GUARD, so it must move the batch, not just the
    ceiling.

    Per-stage batch caps exist because what a step costs is a property of the
    stage: on the MLIP routes train_prior makes no energy call and can hold
    thousands, while the stage after it scores every step through the MLIP and
    cannot. A cap that lowered the ceiling but left the live batch above it would
    guard nothing -- the first step of the new stage would run at the old size,
    which is precisely the allocation the cap was lowered to prevent.

    Both directions, because a clamp that always assigned `v` would pass the
    lowering half on its own while silently shrinking batches on the way UP."""
    from protocol import StageProtocol

    m = _fresh()
    _drive(m, 60)

    # `stage` is a read-only property that resolves through the modeller, and
    # this action reads it only to name itself in a log line -- so stub the
    # property rather than build a whole protocol, which would drag in the
    # config tree for no added coverage.
    class _Named:
        name = 'equilibration'

    class _Proto(StageProtocol):
        stage = _Named()

    proto = object.__new__(_Proto)
    proto.m = m

    # LOWERING clamps the live batch and re-arms the ladder
    m.batch_size = 8000
    m.batch_sizer = {'reason': 'target_met'}
    _Proto._run_action(proto, 'set_max_batch_size', '250', {})
    assert m.args.max_batch_size == 250, m.args.max_batch_size
    assert m.batch_size == 250, (
        f'the ceiling moved but the live batch stayed at {m.batch_size} -- the '
        f'next step would run at the size the cap exists to prevent')
    assert m.batch_sizer is None, 'ladder conclusions from the old cap survived'

    # RAISING leaves the batch alone; the ladder climbs on its own terms
    m.batch_size = 120
    _Proto._run_action(proto, 'set_max_batch_size', '20000', {})
    assert m.args.max_batch_size == 20000, m.args.max_batch_size
    assert m.batch_size == 120, (
        f'raising the cap moved the batch to {m.batch_size}')


def test_every_eval_retry_loop_sizes_itself_from_the_eval_draw():
    """SOURCE-LEVEL, because the runtime tests cannot see this one.

    An eval loop that catches OOM and `continue`s has exactly one way to make
    progress: draw smaller next time. It draws smaller only if it sizes itself
    from eval_draw_size(). A loop that sizes from self.batch_size instead
    reissues an identical allocation forever -- and the train batch is now held
    deliberately steady, so it will never shrink underneath it.

    That is not hypothetical. p4_mace_mle asked for 45 GiB at eval draws of 8, 5,
    3 and 1 alike, because fwd_eval_sampling read self.batch_size while the
    handler shrank a number nothing consulted. The two runtime tests in this file
    both passed throughout: they check the HANDLER, and the defect was in the
    CALLER.

    Checked by inspection rather than execution because driving a real eval needs
    a model, a prior and a GPU -- and the property is a one-line syntactic fact.
    """
    import inspect
    import train

    def _safe_src(fn):
        try:
            return inspect.getsource(fn)
        except (OSError, TypeError):
            return ''

    # DERIVED FROM THE CALL SITES, not hardcoded: any function that recovers an
    # OOM under a NON-TRAIN step_type is subject to this, so a hand-written list
    # would miss the next one added. The step_type literal is what separates an
    # eval retry from a train one -- the handler itself gates on TRAIN_MODES.
    import re
    from protocol import TRAIN_MODES
    call = re.compile(r"handle_train_epoch_error\(\s*\w+\s*,\s*'([^']+)'")
    names = sorted({
        m for m in dir(train.Modeller)
        if not m.startswith('__') and callable(getattr(train.Modeller, m, None))
        and any(st not in TRAIN_MODES
                for st in call.findall(_safe_src(getattr(train.Modeller, m))))})
    assert len(names) >= 3, f'expected at least 3 eval retry loops, found {names}'
    for name in names:
        fn = getattr(train.Modeller, name)
        src = inspect.getsource(fn)
        assert 'handle_train_epoch_error' in src, (
            f'{name} no longer retries on OOM -- if the recovery moved, this '
            f'test is pinning the wrong function')
        assert 'eval_draw_size()' in src, (
            f'{name} retries on OOM but never calls eval_draw_size(); its retry '
            f'cannot get smaller and the loop will spin')
        # the draw itself must not come from the train batch
        for line in src.splitlines():
            if 'self.batch_size' not in line or line.strip().startswith('#'):
                continue
            raise AssertionError(
                f'{name} sizes from self.batch_size: {line.strip()!r} -- the '
                f'train batch is held steady, so this retry never shrinks')
