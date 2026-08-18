"""
THE WARMUP RAMP IS DETERMINISTIC; THE SENSORS DECIDE ONLY WHEN IT ENDS.

WHAT THE BUG WAS. `on_hypergradient` was exempted from the warmup hold so that
`_maybe_freeze_envelope` could see `peak_scale` fall and stop the ramp early.
That coupled the DETECTOR to the ACTUATOR: the freeze needed the sensor to be
moving `peak_scale` during the ramp, so the sensor moved it in both directions --
and during a ramp the readings are structurally positive, because the envelope is
holding the rate ~lr_warmup_ratio below the operating point and a cosine measured
through that suppression says "too cold" whatever the operating point is.

Measured 2026-08-17 on the bench (Adam cell, the production operand): with the
exemption, `peak_scale` reached ~1,968 of its 2,000 bound INSIDE warmup; without
it, 1.0 flat. At beta 0.1 the bound is ~76 steps away. The ramp stopped being
deterministic, which is the one property it exists to provide.

THE FIX. Neither sensor actuates during warmup. Each stops the ramp on the
cadence it can support:

  hyper  speaks every step, so it averages `warmup_freeze_cos_window` of them and
         freezes when the mean ERROR reaches the setpoint -- "ramp until cos
         reaches the target, then stop". This is not a rare safety catch: cos at
         the operating point sits near zero (measured -0.043..+0.052 across the
         hyperslope arms), so a healthy ramp is EXPECTED to end this way.
  ray    gets two readings in a 1000-step ramp at period 500, so it cannot
         average and freezes on the FIRST alpha* below alpha_target. Licensed by
         the asymmetry the controller already documents: freezing early costs
         some warmup, not the operating point.

WHAT THESE TESTS PIN, and why each earns its place:

  * the ramp is deterministic -- cos pinned at +1 for a whole warmup leaves
    peak_scale at exactly 1.0. This is the regression;
  * a sustained too-hot verdict still freezes, so the exemption's PURPOSE
    survives the fix. Deleting the climb while also deleting the freeze would
    pass a naive "peak_scale didn't move" assertion;
  * NOISE AROUND A COLD MEAN DOES NOT FREEZE, while cos ARRIVING at the setpoint
    DOES -- the two are deliberately a pair, because the difference between them
    is the whole rule. The first kills the obvious wrong fix (gate only the up
    branch), which rectifies noise into downward drift and stops a ramp that is
    still cold. The second pins the stated intent, and is the reason the freeze
    must not be described as a rare safety catch: near the operating point cos
    straddles zero, so a healthy ramp is SUPPOSED to end this way;
  * the sensor channel stays live through the ramp -- measured every step, only
    the actuation withheld -- so a held reading is not a dark sensor.

Drives the REAL `LRController` through the real call order against a stub
modeller; `controller.py` imports only `math`, so this is a fast-lane test.
"""

import pytest

from controller import LRController

pytestmark = pytest.mark.fast

BETA = 0.05
WARMUP = 200
SPAN = 25


class _Bag:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _controller(kind='hyper', warmup_steps=WARMUP, freeze=True,
                window=SPAN, alpha_target=4.0):
    adaptive = _Bag(warmup_steps=warmup_steps, seed_lr=1.25e-4,
                    bounds=(0.01, 2000.0), divergence_loss_abs=1.0e9,
                    divergence_grad_abs=1.0e9, divergence_cut=0.5,
                    envelope_freeze=freeze, restart_after=None,
                    control_flow_lr=False, warmup_freeze_cos_window=window,
                    calibration=_Bag(alpha_target=alpha_target, eta_up=0.25,
                                     eta_down=0.5))
    args = _Bag(adaptive_lr=adaptive, lr_warmup_ratio=10, min_lr=1.0e-6,
                lr_policy=1.25e-4, lr_back=1.25e-4, lr_replay=1.25e-4,
                lr_fused=1.25e-4, lr_flow=0.1, lr_servo_managed=['lr_fused'])
    sensor = {'kind': kind} if kind != 'hyper' else {'kind': 'hyper', 'beta': BETA}
    modeller = _Bag(
        args=args, step_ind=0, phase=0, lr_ctrl=None,
        optimizers={'fused': _Bag(param_groups=[{'lr': 1.25e-4}, {'lr': 0.1}]),
                    'fwd': _Bag(param_groups=[{'lr': 1.25e-4}])},
        protocol=_Bag(stage=_Bag(lr_sensor=sensor)))
    return modeller, LRController(modeller)


def _fire(modeller, ctrl, cosines):
    for cos in cosines:
        modeller.step_ind += 1
        ctrl.on_hypergradient(cos, BETA)


def _past_warmup(modeller, ctrl):
    """Advance to after the ramp.

    STATE MUST BE MATERIALISED FIRST. `_fresh_state` stamps `stage_start_step`
    from `modeller.step_ind` on the first `_state()` call, and `_elapsed` is
    measured from that -- so setting step_ind before the controller has a state
    yields elapsed 0 and silently tests the warmup path instead."""
    ctrl._state()
    modeller.step_ind = WARMUP + 1


def _frozen(ctrl):
    return ctrl._state().get('envelope_frozen_at')


def _peak(ctrl):
    return ctrl._state()['peak_scale']


# --------------------------------------------------------------- hyper: the ramp

def test_the_ramp_is_deterministic_under_a_pinned_cold_reading():
    """THE REGRESSION. cos = +1 is what a ramp manufactures -- the rate is held
    ~10x low on purpose. peak_scale must not move at all."""
    modeller, ctrl = _controller()
    _fire(modeller, ctrl, [1.0] * (WARMUP - 1))
    assert _peak(ctrl) == 1.0
    assert _frozen(ctrl) is None, 'a cold ramp must run to its end, not freeze'


def test_a_sustained_too_hot_verdict_still_freezes_the_ramp():
    """The exemption's PURPOSE. If this stops working the fix has silently
    disabled the early stop, which is what the exemption was added for."""
    modeller, ctrl = _controller()
    # EXACTLY enough to freeze, and no further. The freeze fires ON the SPAN-th
    # reading and that reading returns without actuating, so peak is untouched --
    # which is the claim. Firing PAST the freeze would no longer test it: since
    # 2026-08-17 the warmup hold is keyed on whether the envelope is still MOVING
    # (LRController._ramping), not on the step budget, so readings after a freeze
    # actuate normally. This used to fire SPAN + 5 and the extra five cut the peak
    # to 0.8187 = exp(beta_down * -0.4 * 5), which is correct behaviour rather
    # than a broken latch.
    _fire(modeller, ctrl, [-0.4] * SPAN)
    assert _frozen(ctrl) is not None
    assert _peak(ctrl) == 1.0, 'freezing is a latch on the ENVELOPE, not a rate cut'


def test_noise_around_a_COLD_mean_does_not_freeze_the_ramp():
    """THE CONTROL THAT KILLS THE OBVIOUS WRONG FIX. Gating only the up branch
    rectifies noise into downward drift and ends a ramp that is still cold.

    The noise here is 100% of the signal -- mean +0.3 swinging to 0.0 and +0.6,
    harsher than the measured live swing of -0.2 .. +0.4 around a positive mean.
    A rule that survives this is reading the MEAN, not the excursions."""
    modeller, ctrl = _controller()
    noise = [0.0, 0.6] * (WARMUP // 2 - 1)
    _fire(modeller, ctrl, noise)
    assert _frozen(ctrl) is None, (
        f'froze while still cold after {len(noise)} readings (mean +0.3) -- a '
        f'ramp that stops here never reaches the operating point')


def test_the_ramp_ENDS_when_cos_reaches_zero_even_though_that_is_noisy():
    """THE STATED INTENT, and deliberately the mirror of the test above: "ramp
    until cos reaches 0, then stop".

    cos at the operating point sits near zero -- measured -0.043 .. +0.052 across
    the hyperslope arms -- so the average straddles zero there and the freeze is
    EXPECTED to fire. That is the rule working, not a false positive: a rate
    already at the setpoint has nothing left to ramp toward. This is why the
    interlock must not be described as a rare safety catch."""
    modeller, ctrl = _controller()
    _fire(modeller, ctrl, [0.05, -0.05] * (SPAN + 5))
    assert _frozen(ctrl) is not None, (
        'a ramp that has arrived must stop; otherwise the envelope keeps '
        'climbing past the point the sensor says it reached')


def test_one_early_reading_cannot_end_the_ramp():
    """A full window must accumulate first. Without this a single negative
    reading freezes within a few steps and the ramp never happens -- the failure
    the original high-water rule was built against."""
    modeller, ctrl = _controller()
    _fire(modeller, ctrl, [-1.0] * (SPAN - 1))
    assert _frozen(ctrl) is None
    _fire(modeller, ctrl, [-1.0])
    assert _frozen(ctrl) is not None, 'must fire once the window is full'


def test_the_setpoint_moves_the_exit_point():
    """`cos_target` shifts where the ramp ends, since the rule reads the ERROR.
    A cos of +0.10 is 'arrived' against a target of 0.15 and 'still cold'
    against 0.0 -- so this fails if the rule reads the raw cosine."""
    modeller, ctrl = _controller()
    for _ in range(SPAN + 5):
        modeller.step_ind += 1
        ctrl.on_hypergradient(0.10, BETA, None, 0.15)
    assert _frozen(ctrl) is not None

    modeller, ctrl = _controller()
    for _ in range(SPAN + 5):
        modeller.step_ind += 1
        ctrl.on_hypergradient(0.10, BETA, None, 0.0)
    assert _frozen(ctrl) is None


def test_actuation_resumes_after_warmup():
    """The hold is warmup-only. Past it the sensor owns peak_scale again, in
    both directions -- otherwise this fix would delete the controller."""
    modeller, ctrl = _controller()
    _past_warmup(modeller, ctrl)
    _fire(modeller, ctrl, [0.5])
    assert _peak(ctrl) > 1.0
    _fire(modeller, ctrl, [-0.9] * 5)
    assert _peak(ctrl) < 1.0


def test_the_sensor_channel_stays_live_through_the_ramp():
    """MEASURED EVERY STEP, ACTUATION WITHHELD. A held reading must still reach
    the report, or a ramp is indistinguishable from a dead sensor -- the exact
    defect test_hyper_sensor_channel.py exists for."""
    modeller, ctrl = _controller()
    _fire(modeller, ctrl, [0.4] * 10)
    ctrl.step()
    report = ctrl.report()
    assert report['lr_ctrl/hyper_cos'] == pytest.approx(0.4)
    assert report['lr_ctrl/hyper_n'] == 10.0
    assert report['lr_ctrl/hyper_applied'] == pytest.approx(1.0), (
        'nothing was actuated, so the applied multiplier must be exactly 1')


def test_freeze_off_is_still_honoured():
    """`envelope_freeze: false` documents FREEZE OFF (state 7 replaced the
    per-sensor fall threshold with this boolean). A new trigger path must not
    resurrect the freeze."""
    modeller, ctrl = _controller(freeze=False)
    _fire(modeller, ctrl, [-1.0] * (SPAN + 5))
    assert _frozen(ctrl) is None


# ----------------------------------------------------------------------- ray

def _calibrate(modeller, ctrl, alpha, status='bracketed'):
    ctrl.on_calibration({'status': status, 'alpha_star': alpha})


def test_ray_no_longer_refuses_during_warmup():
    """The probe has to ARM to see anything, and train.py gates arming on this."""
    modeller, ctrl = _controller(kind='ray')
    assert ctrl._elapsed(ctrl._state()) < WARMUP, 'precondition: inside warmup'
    assert ctrl.calibration_refusal() is None


def test_ray_freezes_on_its_first_downward_reading():
    """alpha* below alpha_target says the rate is already hotter than we steer
    to. One reading, no averaging -- there are only two in a whole ramp."""
    modeller, ctrl = _controller(kind='ray')
    _calibrate(modeller, ctrl, alpha=1.0)          # target 4.0
    assert _frozen(ctrl) is not None
    assert _peak(ctrl) == 1.0, 'freeze-only during warmup: the rate must not move'


def test_ray_does_not_freeze_on_an_upward_reading():
    """alpha* above target means room to climb; the ramp is doing that already."""
    modeller, ctrl = _controller(kind='ray')
    _calibrate(modeller, ctrl, alpha=40.0)
    assert _frozen(ctrl) is None
    assert _peak(ctrl) == 1.0


@pytest.mark.parametrize('status', ['unresolved', 'inconsistent'])
def test_ray_does_not_freeze_on_a_reading_that_did_not_resolve(status):
    """'A calibration that cannot see the answer must not guess it' -- that rule
    governs the freeze exactly as it governs the actuator."""
    modeller, ctrl = _controller(kind='ray')
    _calibrate(modeller, ctrl, alpha=1.0, status=status)
    assert _frozen(ctrl) is None


def test_ray_actuates_normally_after_warmup():
    """REGRESSION GUARD. The freeze-only rule is warmup-scoped; past it the
    calibration must move peak_scale as it always did."""
    modeller, ctrl = _controller(kind='ray')
    _past_warmup(modeller, ctrl)
    _calibrate(modeller, ctrl, alpha=1.0)
    assert _peak(ctrl) < 1.0, 'alpha* below target must cut the peak after warmup'
