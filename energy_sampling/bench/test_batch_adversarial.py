"""
REGRESSION tests for the auto batch-size controller (train.Modeller.increment_batch_size).

Every test here started life as an adversarial REPRO -- written 2026-08-13 to fail
against the then-shipping code, one per defect found by attacking the 2026-08-11/12
rewrite. The defects are fixed; the assertions are now inverted to demand the correct
behaviour, and each docstring keeps the original mechanism so a reintroduction is
recognisable rather than merely red.

All eight were INTERACTIONS between mechanisms that were individually correct
(occupancy floor, runaway guard, pin/recheck, OOM ceiling, grad-accum guard) or gaps
in what survived a checkpoint. That is the thing to keep testing: the five-way
combination, not the five parts.

The occupancy floor is GONE (see increment_batch_size), so A-2/A-3/A-8 no longer have
a subject in the form they were written. They are kept, re-aimed at what the deletion
is supposed to have bought: a sensor that reads on slow steps, a runaway guard that is
always reachable, and a cooldown nothing walks through.

Naming: A-n, to keep them distinct from the F-nnn findings in test_batch_sizer.py.
"""

from copy import deepcopy

import numpy as np
import pytest

from bench.fake_modeller import (FakeModeller, FakeStage, attach_real_batch_sizer,
                                 make_args)
from bench.harness import BenchRun

from energy_sampling.checkpointing import MODELLER_STATE_DEFAULTS


OOM = "CUDA out of memory. Tried to allocate 2.00 GiB (synthetic eval OOM)"


def _oom_steps(run):
    """Steps whose train_step raised -- harness.step leaves loss=None there."""
    return [h['step'] for h in run.history if h['loss'] is None]


def _med_step_time(run, n=20):
    return float(np.median([h['step_time'] for h in run.history[-n:]]))


# =============================================================================
# A-1  RESUME with a checkpointed OOM ceiling
# =============================================================================

def _as_a_freshly_constructed_modeller(m):
    """
    Strip the attributes a REAL Modeller does not have at construction.

    train.py's init_train_constants sets EXACTLY the MODELLER_STATE_DEFAULTS keys and
    nothing else, so anything outside that dict springs into existence only when
    protocol.advance / handle_train_epoch_error / increment_batch_size first writes
    it. Guarded on the real dict, so the helper tracks what the trainer actually does
    instead of asserting a snapshot of it.
    """
    for k in ('batch_size_saturated_stage', 'batch_size_pinned_at', '_rung_throughput'):
        if k not in MODELLER_STATE_DEFAULTS and hasattr(m, k):
            delattr(m, k)


def _restore_checkpoint(m, state):
    """Checkpointer.load_state_dict + reconcile_batch_size's growth-on clamp."""
    for k, default in MODELLER_STATE_DEFAULTS.items():
        setattr(m, k, state[k] if k in state else deepcopy(default))
    m.batch_size = min(m.batch_size, int(m.args.max_batch_size))


def test_A1_resume_with_a_stored_oom_ceiling_survives_and_keeps_its_pin():
    """
    MECHANISM (fixed). `batch_size_oom_ceiling` was checkpointed but
    `batch_size_saturated_stage` was not, and the growth walk reads the latter as a
    BARE attribute:

        ceiling = getattr(self, 'batch_size_oom_ceiling', None)
        if ceiling is not None and target >= ceiling:
            if self.batch_size_saturated_stage != stage_name:      # <-- bare

    In a live process that is safe by accident: every writer of the ceiling also
    writes the pin. A RESUME broke the pairing -- ceiling restored from disk, pin
    undefined -- and since the call site is OUTSIDE the train loop's try/except, the
    resumed job died of AttributeError ~2 growth intervals in. ADDING the ceiling to
    the checkpoint is what armed it.

    Both keys are in MODELLER_STATE_DEFAULTS now, which fixes the crash AND stops a
    resumed run re-climbing a knee it already paid to find.
    """
    for k in ('batch_size_saturated_stage', 'batch_size_pinned_at'):
        assert k in MODELLER_STATE_DEFAULTS, (
            f'{k} must travel with batch_size_oom_ceiling or a resume crashes')

    run = BenchRun(
        game='equilibration',
        gpu_kwargs=dict(t_fixed=2.0, sps_max=5000.0),
        args_overrides=dict(batch_size=1000, max_batch_size=50000,
                            grow_batch_size=True, auto_batch_throughput_opt=True,
                            batch_growth_interval=50, batch_growth_slow_interval=60,
                            max_step_seconds=0),
    )
    m = run.m
    # what the checkpointer wrote after an in-stage OOM at batch 1000, with the knee
    # already pinned -- the state that used to be unrepresentable on disk
    state = {k: getattr(m, k, deepcopy(v)) for k, v in MODELLER_STATE_DEFAULTS.items()}
    state.update(step_ind=20000, batch_size=500, batch_size_oom_ceiling=1000,
                 batch_size_ever_oomed=True, batch_size_last_grow=19000,
                 batch_size_cooldown_until=-1,
                 batch_size_saturated_stage='equilibration',
                 batch_size_pinned_at=19500)

    _as_a_freshly_constructed_modeller(m)      # a NEW process, before the restore
    _restore_checkpoint(m, state)

    r = run.run(600)                            # must not raise
    assert r.m.batch_size <= 1000, 'resumed run climbed into the restored OOM ceiling'


# =============================================================================
# A-2  the occupancy sensor must read on SLOW steps
# =============================================================================

def _util_run(t_fixed, steps=400, **over):
    kw = dict(batch_size=1000, max_batch_size=1000, grow_batch_size=True,
              auto_batch_throughput_opt=True, gpu_util_window_s=900,
              gpu_util_sample_period_s=60, max_step_seconds=0)
    kw.update(over)
    return BenchRun(game='equilibration',
                    gpu_kwargs=dict(t_fixed=t_fixed, sps_max=5000.0, host_frac=0.8),
                    args_overrides=kw).run(steps)


def test_A2_occupancy_window_fills_at_any_step_time():
    """
    MECHANISM (fixed). `_gpu_util` was appended once per ten_step_reporting, i.e.
    every 10 steps, and `_gpu_util_mean` refuses to average fewer than 5 samples in
    the window. 5 samples 10 steps apart span 40 step-times, so a 900 s window only
    ever filled when

        step_time <= gpu_util_window_s / 40 = 22.5 s

    prod0810's uma/mace arms run 181-262 s steps, so consecutive samples were ~2000 s
    apart, at most one landed in the window, and `_gpu_util_mean` returned None on
    every call. The occupancy floor that read it was therefore structurally inert on
    exactly the arms it was written for -- and SILENTLY: a missing NVML sensor warns
    loudly, a window that cannot fill printed nothing and simply left gpu/util_recent
    absent from wandb, which reads as 'not logged' rather than 'not working'.

    The floor is gone, but the METRIC still matters -- it is the number the scheduler
    cancels on. Sampling is now wall-clock (gpu_util_sample_period_s), so the window
    fills at 30 s/step as readily as at 2 s/step.
    """
    slow = _util_run(t_fixed=30.0)
    fast = _util_run(t_fixed=2.0)
    for label, r in (('slow', slow), ('fast', fast)):
        assert r.m._gpu_util_mean(900) is not None, (
            f'{label} arm: 900 s occupancy window never filled -- the metric is '
            f'absent on exactly the runs that need it')
    # the sampler is rate-limited, not free-running: at 2 s/step, 400 steps is 800 s
    # of simulated time, so a 60 s period allows ~14 samples, not 400
    assert len(fast.m._gpu_util) < 30, (
        f'sample period not enforced ({len(fast.m._gpu_util)} readings)')


def test_A2b_missing_sensor_is_loud_and_harmless():
    """No NVML and no nvidia-smi: training must continue, the metric must be absent,
    and the run must have said so once. A silent inert sensor is the failure mode
    this whole area exists to prevent."""
    run = BenchRun(
        game='equilibration',
        gpu_kwargs=dict(t_fixed=2.0, sps_max=5000.0, host_frac=0.6),
        args_overrides=dict(batch_size=1000, max_batch_size=50000,
                            grow_batch_size=True, auto_batch_throughput_opt=True,
                            max_step_seconds=0),
    )
    run.m._gpu_util.clear()
    run.m._gpu_util_off = True             # as if both sources raised
    r = run.run(2000)
    assert r.m._gpu_util_mean(900) is None
    assert r.m.batch_size >= 1000, 'a missing sensor must not stop the throughput gate'


# =============================================================================
# A-3  the runaway guard must always be reachable, and must know when to stop
# =============================================================================

def _slow_clock_run(steps=500, **over):
    """t_fixed 10 s with 90% host-serial time: utilization tops out ~40% however big
    the batch gets. This is the clock the occupancy floor used to climb against."""
    kw = dict(batch_size=1000, max_batch_size=5000, grow_batch_size=True,
              auto_batch_throughput_opt=True, batch_growth_interval=50,
              gpu_util_window_s=900, max_step_seconds=5,
              fused_grad_accum_min_samples=0)
    kw.update(over)
    return BenchRun(game='equilibration',
                    gpu_kwargs=dict(t_fixed=10.0, sps_max=1000.0, host_frac=0.9),
                    args_overrides=kw).run(steps)


def test_A3_runaway_guard_is_never_locked_out(capsys):
    """
    MECHANISM (fixed by deletion). The occupancy branch ended in an UNCONDITIONAL
    `return` that also covered its own 'nowhere left to go' arm. So once utilization
    was under the floor and the batch was stuck at max_batch_size, increment_batch_size
    returned there on EVERY subsequent step and `max_step_seconds` was never evaluated
    again for the rest of the stage.

    The design claimed the guard was 'checked after the floor so it cannot pre-empt
    it'. It was checked after the floor only when the floor was SATISFIED. When the
    floor was losing -- the measured MLIP case -- the guard was dead, and dead in the
    state where step time had just been grown 5x. Observed: the guard cut 1000 -> 409
    for cause, the floor then grew 409 -> ... -> 5000, and 15 s steps ran against a
    5 s ceiling for the rest of the stage.

    With no occupancy rule there is nothing above the guard but a cooldown check, so
    on this clock the batch must move DOWN, not up.
    """
    r = _slow_clock_run()
    out = capsys.readouterr().out
    assert 'cutting' in out, f'runaway guard never fired:\n{out[-600:]}'
    assert r.m.batch_size < 1000, (
        f'batch grew to {r.m.batch_size} on a clock whose steps are 2-3x the ceiling')


def test_A3b_runaway_guard_stands_down_instead_of_ratcheting_to_one(capsys):
    """
    MECHANISM (fixed). The proportional cut is bounded below by `max(1, ...)` and
    nothing else -- `_batch_floor()` deliberately does not apply, since a hard
    wall-clock ceiling outranks a batch-size preference. The only real floor was the
    grad-accum guard, and that is gated on `train_mode == 'fused'`.

    So non-fused stages (bwd / MLE warm start) had NO floor. When the step's FIXED
    cost alone exceeds max_step_seconds -- an MLIP hiccup, a z_cal transition at
    ~12x, a slow host -- the cut ratcheted 1000 -> 538 -> 290 -> ... -> 1 and then
    trained at batch 1 without ever raising an error, one torch.compile recompile per
    rung.

    The guard now measures its own last cut: if the batch fell materially and step
    time did not, the overrun is not batch-driven and it stands down for the stage.
    Shipping values here: mk_dev max_step_seconds 60 with a 100 s fixed clock.
    """
    r = BenchRun(
        game='equilibration',
        stage=FakeStage(name='train_prior', train_mode='bwd'),
        gpu_kwargs=dict(t_fixed=100.0, sps_max=5000.0),
        args_overrides=dict(batch_size=1000, max_batch_size=50000,
                            grow_batch_size=True, auto_batch_throughput_opt=True,
                            max_step_seconds=60, fused_grad_accum_min_samples=1000,
                            oom_cooldown_steps=200),
    ).run(3000)
    out = capsys.readouterr().out
    assert r.m.batch_size >= 100, (
        f'guard ratcheted to {r.m.batch_size} against a FIXED per-step cost')
    assert 'NOT batch-driven' in out, (
        f'guard cut without ever noticing the cuts did nothing:\n{out[-600:]}')
    assert out.count('NOT batch-driven') == 1, 'stand-down must be said once per stage'


def test_A3c_runaway_guard_still_cuts_when_cutting_actually_works():
    """The control for A-3b: on a clock where step time IS proportional to batch, the
    guard must still do its job. A stand-down rule that fires on the responsive case
    would silently disable the only protection against 181 s steps."""
    r = BenchRun(
        game='equilibration',
        stage=FakeStage(name='train_prior', train_mode='bwd'),
        gpu_kwargs=dict(t_fixed=0.0, sps_max=20.0),   # ~50 s at batch 1000, ~5 s at 100
        args_overrides=dict(batch_size=1000, max_batch_size=50000,
                            grow_batch_size=True, auto_batch_throughput_opt=True,
                            max_step_seconds=10, fused_grad_accum_min_samples=0,
                            oom_cooldown_steps=50),
    ).run(2000)
    assert r.m.batch_size < 1000, 'guard failed to cut a genuinely batch-driven overrun'
    assert _med_step_time(r) <= 10 * 1.5, (
        f'median step {_med_step_time(r):.1f}s still far over the 10 s ceiling')


# =============================================================================
# A-4 / A-5  the throughput pin must never JUMP
# =============================================================================

def test_A4_pin_does_not_rebuild_the_oom_sawtooth():
    """
    MECHANISM (fixed). The pin was

        self.batch_size = max(self._batch_floor(), prev_batch)

    `_batch_floor()` is `args.batch_size`. It consulted neither
    `batch_size_oom_ceiling` nor the rungs just measured, so whenever the configured
    batch was at or above the ceiling -- the NORMAL case, since protocol.advance
    re-enters every stage at exactly `args.batch_size` and the ceiling is whatever
    that stage could not fit -- the pin set the batch to a size already recorded as
    OOMing, then marked it saturated so nothing re-examined it.

    Next step OOMs, handle_train_epoch_error re-cuts and clears the pin, the walk
    re-climbs, the gate fails again, the pin jumps back to the ceiling: the permanent
    sawtooth the ceiling was added to kill, rebuilt out of the fix for it.

    Planted with a kernel-regime step at 2400 so the gate fails at a rung whose next
    target is still BELOW the ceiling -- the only way to reach the pin at all, since
    the ceiling check pre-empts it otherwise.
    """
    r = BenchRun(
        game='equilibration',
        gpu_kwargs=dict(t_fixed=2.0, sps_max=5000.0, oom_at=4000,
                        regimes=[(2400, 0.62)]),
        args_overrides=dict(batch_size=4000, max_batch_size=50000,
                            grow_batch_size=True, auto_batch_throughput_opt=True,
                            batch_growth_factor=1.15, batch_growth_interval=60,
                            batch_growth_slow_interval=60, oom_cooldown_steps=60,
                            batch_growth_min_throughput_gain=0.05,
                            max_step_seconds=0, fused_grad_accum_min_samples=0),
    ).run(2500)
    ooms = _oom_steps(r)
    assert r.m.batch_size_oom_ceiling == 4000, r.m.batch_size_oom_ceiling
    late = [s for s in ooms if s > 400]      # OOMs AFTER the ceiling was known
    assert not late, (
        f'walked back into a known OOM ceiling at steps {late} -- the sawtooth is back')
    assert r.m.batch_size < 4000, f'sitting at the ceiling ({r.m.batch_size})'


def test_A5_pin_never_exceeds_the_batch_it_measured():
    """
    The same line, isolated -- no clock tuning, one call to the real controller.

    Standing at 495 with a measured rung at 300, a KNOWN OOM ceiling of 1000 and
    `args.batch_size` 1000, the old pin set the batch to 1000: above both rungs it had
    ever timed, and exactly the size it recorded as unfittable. It then pinned that,
    and the periodic recheck could not move it either -- `dropped = max(floor, B/f)`
    is >= B whenever the floor binds -- so the batch was welded to the OOM ceiling
    until a stage transition.

    A pin is a decision to STOP CLIMBING. It may step back to the previous rung; it
    may never raise the batch.
    """
    attach_real_batch_sizer(FakeModeller)
    args = make_args(batch_size=1000, max_batch_size=50000, grow_batch_size=True,
                     auto_batch_throughput_opt=True, batch_growth_factor=1.65,
                     batch_growth_interval=50, batch_growth_min_throughput_gain=0.05,
                     max_step_seconds=0, fused_grad_accum_min_samples=0)
    m = FakeModeller(args, optimizers={},
                     stage=FakeStage(name='equilibration', train_mode='fused'))
    m.step_ind, m.batch_size_last_grow = 5000, 4000
    m.batch_size = 495                      # two rungs down from a max_step_seconds cut
    m.batch_size_oom_ceiling = 1000         # ...and 1000 is known to OOM
    m.batch_size_ever_oomed = True
    m._rung_throughput = (300, 1000.0)      # last rung measured 1000 samples/s
    for _ in range(20):                     # this rung measures 495 samples/s: no gain
        m._recent_step_times.append(1.0)
        m._recent_step_work.append(495)

    m.increment_batch_size()

    assert m.batch_size <= 495, (
        f'pin RAISED the batch to {m.batch_size} -- above every rung it timed')
    assert m.batch_size < m.batch_size_oom_ceiling, 'pinned at a known-OOM size'
    assert m.batch_size_saturated_stage == 'equilibration', 'should still be pinned'


# =============================================================================
# A-6  an EVAL OOM must not cap the TRAIN batch for the stage
# =============================================================================

def test_A6_eval_oom_does_not_install_a_training_ceiling():
    """
    MECHANISM (fixed). handle_train_epoch_error is the shared recovery path for eval
    too (eval_bwd, eval_fwd, anchor_refresh), and the rewrite made it record
    `batch_size_oom_ceiling = self.batch_size` -- the TRAIN batch, whatever the eval
    loop was actually allocating for.

    Eval has a different memory profile: eval_num_samples per pass, eval_T integration
    steps, the EMA model, no gradients. Before the ceiling existed an eval OOM was
    self-limiting (the eval loops cut and retry). Afterwards one transient eval OOM
    halved the train batch AND installed a stage-lifetime cap on it that only a stage
    transition cleared.

    The cut still applies to eval OOMs -- that is the shared recovery policy. Only the
    ceiling is withheld, gated on protocol.TRAIN_MODES so a new eval call site cannot
    opt itself in by accident.
    """
    run = BenchRun(
        game='equilibration',
        gpu_kwargs=dict(t_fixed=2.0, sps_max=5000.0),
        args_overrides=dict(batch_size=1000, max_batch_size=50000,
                            grow_batch_size=True, auto_batch_throughput_opt=True,
                            batch_growth_interval=50, oom_cooldown_steps=50,
                            batch_growth_min_throughput_gain=0.05,
                            max_step_seconds=0),
    ).run(400)
    grown = run.m.batch_size
    assert grown > 1000, 'run never grew -- nothing to cap'

    run.m.handle_train_epoch_error(RuntimeError(OOM), 'eval_fwd')
    assert run.m.batch_size_oom_ceiling is None, (
        'an EVAL OOM installed a permanent ceiling on the TRAIN batch')
    assert run.m.batch_size == grown // 2, 'shared cut policy should still apply'

    run.run(3000)
    assert run.m.batch_size >= grown, (
        f'train batch never recovered past {grown} (got {run.m.batch_size})')


def test_A6b_train_oom_still_installs_the_ceiling():
    """The control for A-6: withholding the ceiling on eval must not withhold it on
    the path it was built for. Both real train_modes are checked, because gating on a
    list of EVAL names instead would have silently excluded 'bwd'."""
    for mode in ('fused', 'bwd'):
        run = BenchRun(
            game='equilibration',
            stage=FakeStage(name='equilibration', train_mode=mode),
            gpu_kwargs=dict(t_fixed=2.0, sps_max=5000.0),
            args_overrides=dict(batch_size=1000, max_batch_size=50000,
                                grow_batch_size=True, auto_batch_throughput_opt=True,
                                max_step_seconds=0),
        )
        # past step 0: handle_train_epoch_error returns early there (an OOM on the
        # very first step is a config problem, not something to adapt to)
        run.m.step_ind = 100
        run.m.handle_train_epoch_error(RuntimeError(OOM), mode)
        assert run.m.batch_size_oom_ceiling == 1000, (
            f'train_mode {mode!r}: OOM ceiling not recorded -- grow-blind again')


# =============================================================================
# A-7  the accum guard must not shout every step
# =============================================================================

def test_A7_accum_floor_message_is_said_once_per_stage(capsys):
    """
    MECHANISM (fixed). With `fused_grad_accum_min_samples >= batch_size` in a fused
    stage the guard raises the proposed cut back to the current batch, detects
    `shrunk >= self.batch_size`, prints, and returns -- with no cooldown, because no
    cut happened. increment_batch_size runs every step, so the message repeated every
    step for as long as the condition held.

    mk_dev ships batch_size 1000 and fused_grad_accum_min_samples 1000, so at the base
    batch of a fused stage max_step_seconds is structurally incapable of cutting and
    the guard's entire contribution was one 4-line message per step -- ~10k copies on
    a 7-day run at 60 s/step.

    Freezing the controller here is CORRECT (growing a step already over the ceiling
    makes it worse), so that is asserted too, not just the quiet.
    """
    r = BenchRun(
        game='equilibration',
        gpu_kwargs=dict(t_fixed=100.0, sps_max=5000.0),
        args_overrides=dict(batch_size=1000, max_batch_size=50000,
                            grow_batch_size=True, auto_batch_throughput_opt=True,
                            max_step_seconds=60, fused_grad_accum_min_samples=1000),
    ).run(300)
    out = capsys.readouterr().out
    n = out.count('already the smallest size')
    assert n == 1, f'expected exactly one notice per stage, got {n}'
    assert r.m.batch_size == 1000, 'controller should be frozen, not growing'


# =============================================================================
# A-8  the OOM cooldown must actually be a dwell
# =============================================================================

def test_A8_nothing_grows_through_the_oom_cooldown():
    """
    MECHANISM (fixed by deletion). The occupancy branch's own clock was
    `batch_size_last_grow` only; it never consulted `batch_size_cooldown_until`, the
    AIMD cooldown that handle_train_epoch_error installs and that both the runaway
    guard and the growth walk respect. handle_train_epoch_error does not touch
    `batch_size_last_grow` either, so that clock was typically already expired at the
    moment of the OOM.

    Net: with the floor on -- every cluster arm -- `oom_cooldown_steps` was inert and
    the batch regrew toward the freshly discovered ceiling within one growth interval,
    with no dwell at the reduced size. The OOM ceiling stopped it in the bench, where
    the ceiling is exact; it is not exact in reality (fragmentation and buffer growth
    move it), which is what the cooldown was for.
    """
    r = BenchRun(
        game='equilibration',
        gpu_kwargs=dict(t_fixed=2.0, sps_max=5000.0, host_frac=0.9, oom_at=4000),
        args_overrides=dict(batch_size=1000, max_batch_size=50000,
                            grow_batch_size=True, auto_batch_throughput_opt=True,
                            batch_growth_interval=50, oom_cooldown_steps=2000,
                            gpu_util_window_s=900, max_step_seconds=0),
    ).run(400)
    ooms = _oom_steps(r)
    assert ooms, "never OOM'd -- nothing to cool down from"
    first = ooms[0]
    after = {h['step']: h['batch'] for h in r.history if h['step'] > first}
    post_cut = after[first + 1]
    early = sorted(s for s, b in after.items() if b > post_cut and s < first + 2000)
    assert not early, (
        f'batch grew at step {early[0]} -- {early[0] - first} steps into a '
        f'2000-step cooldown from the OOM at {first}')
