"""
The hypergradient sensor's REPORT CHANNEL: what `lr_ctrl/hyper_*` means.

WHAT THE BUG WAS. `_hyper_last` held the most recent reading and `_emit`
republished it on every reporting period, forever. `on_hypergradient` is reached
from `train.py::step_loss` only AFTER the non-finite-gradient guard, which
returns early -- so a run whose gradients go non-finite stops firing the sensor
while the channel goes on publishing the last live value.

Measured 2026-08-17 on the qm9 conditional route, run `real_qm9cond_liveservo`:
from step 902 the fused gradient was non-finite on every optimizer step, and the
run then logged cos = -0.267178 for 327 consecutive rows -- one unique value, sd
exactly 0.0 -- with `hypergrads` frozen at 379 and `peak_scale` frozen at 2.0929.
A DEAD sensor was indistinguishable from a working one, and the mean of that
channel over the run (-0.209) was an average over a repeated constant, of the
opposite sign to the readings that had actually moved `peak_scale` (+0.037).

THE SECOND DEFECT, fixed here too: the channel carried the LAST firing while
`peak_scale` integrates EVERY firing. With `fused_grad_accum_min_samples` above
`batch_size` the optimizer steps once per several `step_ind` -- 1000 over 500 on
that route -- so ~5 firings landed per reported row and the published cos was one
sample of five. Sensor and actuator described different steps.

WHAT THESE TESTS PIN:

  * a period with NO firing publishes no cos/applied/status at all -- absent, not
    stale. This is the fix, and it is the one assertion that fails if `report`
    stops draining;
  * `hypergrads` is STILL published in that period, so a flat counter beside an
    absent cos reads as "the sensor stopped" and not as "the stage never had
    one" -- the distinction the counter was added for in the first place;
  * cos is the period MEAN and applied is the period's TOTAL multiplier, so the
    two channels describe the same steps as `peak_scale`;
  * the channel recovers when the sensor does, so the fix is not "publish
    nothing", which would pass a naive absence assertion while deleting the
    diagnostic.

These drive the REAL `LRController` through the REAL call order
(`on_hypergradient` ... `step` ... `report`), against a stub modeller. A local
re-implementation of the accumulator would pass its own tests while the shipped
one regressed. `controller.py` imports only `math`, so this is a fast-lane test.
"""

import math

import pytest

from controller import LRController

pytestmark = pytest.mark.fast

BETA = 0.05


class _Bag:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _controller(warmup_steps=0):
    """A controller on a stub modeller, wired like a fused hyper stage."""
    adaptive = _Bag(warmup_steps=warmup_steps, seed_lr=1.25e-4,
                    bounds=(0.01, 2000.0), divergence_loss_abs=1.0e9,
                    divergence_grad_abs=1.0e9, divergence_cut=0.5,
                    envelope_freeze=True, restart_after=None,
                    control_flow_lr=False)
    args = _Bag(adaptive_lr=adaptive, lr_warmup_ratio=10, min_lr=1.0e-6,
                lr_policy=1.25e-4, lr_back=1.25e-4, lr_replay=1.25e-4,
                lr_fused=1.25e-4, lr_flow=0.1, lr_servo_managed=['lr_fused'])
    modeller = _Bag(
        args=args, step_ind=0, phase=0, lr_ctrl=None,
        optimizers={'fused': _Bag(param_groups=[{'lr': 1.25e-4}, {'lr': 0.1}]),
                    'fwd': _Bag(param_groups=[{'lr': 1.25e-4}])},
        protocol=_Bag(stage=_Bag(lr_sensor={'kind': 'hyper', 'beta': BETA})))
    return modeller, LRController(modeller)


def _fire(modeller, ctrl, *cosines):
    """One optimizer step per reading, two step_ind apart -- the cadence grad
    accumulation actually produces on this route."""
    for cos in cosines:
        modeller.step_ind += 2
        ctrl.on_hypergradient(cos, BETA)


def _period(modeller, ctrl):
    """One reporting period, in train.py's order: step_lr_schedule() then
    ten_step_reporting(). Anything else would eat readings -- see report()."""
    ctrl.step()
    return ctrl.report()


# ------------------------------------------------------- the reading is absent

def test_silent_period_publishes_no_reading():
    """THE BUG. Three periods with no firing must publish no cos at all, rather
    than repeating the last live one 327 times as liveservo did."""
    modeller, ctrl = _controller()
    _fire(modeller, ctrl, 0.2, -0.1, 0.3)
    live = _period(modeller, ctrl)
    assert 'lr_ctrl/hyper_cos' in live

    for _ in range(3):
        modeller.step_ind += 10
        silent = _period(modeller, ctrl)
        assert 'lr_ctrl/hyper_cos' not in silent
        assert 'lr_ctrl/hyper_applied' not in silent
        assert 'lr_ctrl/hyper_status' not in silent
        assert 'lr_ctrl/hyper_n' not in silent


def test_silent_period_still_publishes_the_counter():
    """...but `hypergrads` stays, flat. Absent cos plus a flat counter is
    "stopped"; absent cos plus an absent counter is "never configured", and the
    two need different responses."""
    modeller, ctrl = _controller()
    _fire(modeller, ctrl, 0.2, -0.1, 0.3)
    _period(modeller, ctrl)

    modeller.step_ind += 10
    silent = _period(modeller, ctrl)
    assert silent['lr_ctrl/hypergrads'] == 3.0


def test_never_fired_publishes_no_hyper_block():
    """A stage whose sensor never fires publishes nothing, counter included."""
    modeller, ctrl = _controller()
    modeller.step_ind = 50
    report = _period(modeller, ctrl)
    assert not [k for k in report if k.startswith('lr_ctrl/hyper')]


def test_channel_recovers_when_the_sensor_does():
    """The fix must not be "publish nothing" -- a naive absence assertion would
    pass with the sensor deleted."""
    modeller, ctrl = _controller()
    _fire(modeller, ctrl, 0.2)
    _period(modeller, ctrl)
    modeller.step_ind += 10
    _period(modeller, ctrl)

    _fire(modeller, ctrl, -0.4)
    back = _period(modeller, ctrl)
    assert back['lr_ctrl/hyper_cos'] == pytest.approx(-0.4)


# --------------------------------- the reading describes the whole period

def test_cos_is_the_period_mean_not_the_last_firing():
    """peak_scale integrates every firing; so must the channel. The cosines here
    are chosen so the mean and the last value DIFFER -- with a degenerate set
    this assertion passes against the old last-firing behaviour."""
    modeller, ctrl = _controller()
    cosines = (0.2, -0.1, 0.3, 0.0, 0.15)
    _fire(modeller, ctrl, *cosines)
    report = _period(modeller, ctrl)

    expected = sum(cosines) / len(cosines)
    assert expected != cosines[-1], 'degenerate fixture: mean equals last'
    assert report['lr_ctrl/hyper_cos'] == pytest.approx(expected)
    assert report['lr_ctrl/hyper_n'] == len(cosines)


def test_applied_is_the_total_multiplier_over_the_period():
    """Accumulated in LOG space, because the actuator is multiplicative: a period
    that halved then doubled has moved by 1.0, not by 2.5.

    THE EXPECTED VALUE CARRIES THE ASYMMETRIC GAIN, and must: since 2026-08-17 a
    negative error moves the peak with `beta_down`, which defaults to
    hyper_down_gain (2.0) times beta. Writing this as exp(BETA * sum(cosines))
    would be asserting the old symmetric rule, and it is exactly the assertion
    that failed when the asymmetry landed.
    """
    modeller, ctrl = _controller()
    cosines = (0.2, -0.1, 0.3, 0.0, 0.15)
    _fire(modeller, ctrl, *cosines)
    report = _period(modeller, ctrl)

    down_gain = 2.0    # controller default for hyper_down_gain
    expected_log = sum((BETA if c > 0 else BETA * down_gain) * c for c in cosines)
    assert report['lr_ctrl/hyper_applied'] == pytest.approx(math.exp(expected_log))
    assert expected_log != pytest.approx(BETA * sum(cosines)), \
        'fixture is degenerate: it cannot tell the asymmetric rule from the old one'


def test_applied_tracks_peak_scale_over_the_period():
    """The two channels must not be able to disagree: applied is exactly the
    factor peak_scale moved by, which is what makes the pair readable."""
    modeller, ctrl = _controller()
    before = float(ctrl._state()['peak_scale'])
    _fire(modeller, ctrl, 0.2, -0.1, 0.3, 0.0, 0.15)
    report = _period(modeller, ctrl)

    after = float(ctrl._state()['peak_scale'])
    assert report['lr_ctrl/hyper_applied'] == pytest.approx(after / before)


def test_nonfinite_is_counted_but_kept_out_of_the_mean():
    """A non-finite reading moves nothing, so averaging it in would report a
    period as colder than the readings that actually acted."""
    modeller, ctrl = _controller()
    _fire(modeller, ctrl, float('nan'), 0.2)
    report = _period(modeller, ctrl)

    assert report['lr_ctrl/hyper_cos'] == pytest.approx(0.2)
    assert report['lr_ctrl/hyper_n'] == 2
    assert report['lr_ctrl/hyper_status'] == float(
        LRController._HYPER_STATUS['clean'])
