"""
LRController v8 tests. No train.py import, no GPU -- about a second.

These pin the ACTUATOR, not just the sensor. The module's own docstring records
that "a controller that is holding and one that is satisfied are otherwise
indistinguishable from peak_scale alone, which is the failure this module has
logged three separate times." Every test below therefore asserts on the LR that
actually reached an optimizer, not on internal state.
"""

import math

import pytest
import torch

from bench.fake_modeller import FakeModeller, make_args
from bench.old.harness import BenchRun
from energy_sampling.controller import LRController


def _modeller(**overrides):
    args = make_args(**overrides)
    p = torch.nn.Parameter(torch.zeros(4))
    q = torch.nn.Parameter(torch.zeros(4))
    opts = {
        'fwd': torch.optim.SGD([p], lr=args.lr_policy),
        'flow': torch.optim.SGD([q], lr=args.lr_flow),
        'fused': torch.optim.SGD([{'params': [p]}, {'params': [q]}], lr=args.lr_fused),
    }
    m = FakeModeller(args, opts)
    m.ray_cal = None
    m.lr_controller = LRController(m)
    return m


def _reading(alpha, status='bracketed'):
    return {'status': status, 'alpha_star': float(alpha), 'lo': None, 'hi': None}


# ------------------------------------------------------------------- envelope

def test_envelope_ramps_from_one_over_warmup_ratio_to_one():
    m = _modeller(**{'adaptive_lr.warmup_steps': 1000})
    m.step_ind = 0
    m.lr_controller.step()
    assert m.lr_of('fwd') == pytest.approx(m.args.lr_policy / m.args.lr_warmup_ratio, rel=1e-6)

    m.step_ind = 1000
    m.lr_controller.step()
    assert m.lr_of('fwd') == pytest.approx(m.args.lr_policy, rel=1e-6)


def test_min_lr_floors_the_product():
    m = _modeller(**{'adaptive_lr.warmup_steps': 1000, 'min_lr': 5e-3})
    m.step_ind = 0
    m.lr_controller.step()
    assert m.lr_of('fwd') == pytest.approx(5e-3)


# ------------------------------------------------------------------- actuator

def test_flow_group_is_pinned_not_scheduled():
    """
    The Z head is LR-pinned: neither the envelope nor peak_scale may move it,
    because alpha* is measured over policy params only. Both the standalone
    'flow' optimizer and the TRAILING group of 'fused' take this branch.
    """
    m = _modeller(**{'adaptive_lr.warmup_steps': 1000})
    m.step_ind = 0
    m.lr_controller.step()
    assert m.lr_of('flow') == pytest.approx(m.args.lr_flow)
    assert m.optimizers['fused'].param_groups[-1]['lr'] == pytest.approx(m.args.lr_flow)
    # ...while the leading fused group is scheduled
    assert m.optimizers['fused'].param_groups[0]['lr'] < m.args.lr_fused

    m.step_ind = 2000
    m.lr_controller.on_calibration(_reading(64.0))
    assert m.lr_ctrl['peak_scale'] > 1.0
    assert m.lr_of('flow') == pytest.approx(m.args.lr_flow), 'peak_scale leaked into the flow head'
    assert m.optimizers['fused'].param_groups[-1]['lr'] == pytest.approx(m.args.lr_flow)


def test_control_flow_lr_grants_the_envelope_but_not_peak_scale():
    """
    PIN THE REAL SEMANTICS, which differ from the config comment.

    mk_dev.yaml says of control_flow_lr: "true = the flow/Z groups also take
    envelope and peak_scale". They take the envelope. They do NOT take
    peak_scale, because _apply_lrs gates peak on `base_key in managed`, and
    lr_servo_managed holds only the keys written `auto`. lr_flow is an explicit
    float (0.1) in every shipping config, so it is never managed and peak_scale
    can never reach it -- control_flow_lr on its own is a warmup-envelope switch.

    Getting peak_scale onto the flow head requires writing lr_flow: auto, which
    also changes what it starts from.
    """
    m = _modeller(**{'adaptive_lr.warmup_steps': 1000, 'adaptive_lr.control_flow_lr': True})
    m.step_ind = 0
    m.lr_controller.step()
    assert m.lr_of('flow') == pytest.approx(m.args.lr_flow / m.args.lr_warmup_ratio), \
        'control_flow_lr must at least grant the envelope'

    m.step_ind = 2000
    m.lr_controller.on_calibration(_reading(64.0))
    assert m.lr_ctrl['peak_scale'] > 1.0
    assert m.lr_of('flow') == pytest.approx(m.args.lr_flow), \
        'peak_scale reached an unmanaged key'

    # ...and it DOES reach it once lr_flow is declared managed
    m2 = _modeller(**{'adaptive_lr.warmup_steps': 0, 'adaptive_lr.control_flow_lr': True,
                      'lr_servo_managed': ('lr_policy', 'lr_flow')})
    m2.step_ind = 100
    m2.lr_controller.step()          # materialise state here...
    m2.step_ind = 101                # ...and clear the one-step ramp (see below)
    m2.lr_controller.on_calibration(_reading(64.0))
    assert m2.lr_of('flow') > m2.args.lr_flow


def test_warmup_steps_zero_still_costs_one_suppressed_step():
    """
    `warmup_steps: 0` does not disable the envelope. _envelope takes
    max(1, warmup_steps), and elapsed is 0 on the evaluation that CREATES the
    state, so the first tick still comes out at 1/lr_warmup_ratio. It releases
    on the next step.

    Harmless in a real run, but it silently costs one step at a tenth of the
    intended rate, and it is enough to make a test that sets warmup_steps: 0 and
    reads the LR immediately measure the wrong number.
    """
    m = _modeller(**{'adaptive_lr.warmup_steps': 0})
    m.step_ind = 100
    m.lr_controller.step()
    assert m.lr_of('fwd') == pytest.approx(m.args.lr_policy / m.args.lr_warmup_ratio)
    m.step_ind = 101
    m.lr_controller.step()
    assert m.lr_of('fwd') == pytest.approx(m.args.lr_policy)


def test_unmanaged_keys_are_a_control_arm():
    """
    lr_servo_managed is what resolve_derived_config records for keys written
    `auto`. An empty set must leave the controller reading and logging while
    actuating NOTHING -- that is a deliberate experimental arm, so it has to keep
    working.
    """
    m = _modeller(**{'adaptive_lr.warmup_steps': 0, 'lr_servo_managed': ()})
    m.step_ind = 100
    m.lr_controller.step()
    before = m.lr_of('fwd')
    m.lr_controller.on_calibration(_reading(64.0))
    assert m.lr_ctrl['peak_scale'] > 1.0, 'peak_scale should still move (it is logged)'
    assert m.lr_of('fwd') == pytest.approx(before), 'unmanaged key must not be actuated'


# ---------------------------------------------------------------- calibration

def test_warmup_holds_every_sensor():
    """
    The envelope is deliberately below 1 during warmup, so alpha* rates a
    SUPPRESSED step. Acting on it would inflate peak_scale by exactly the warmup
    factor and hand that back as a real LR the moment the envelope releases.
    """
    m = _modeller(**{'adaptive_lr.warmup_steps': 1000})
    m.step_ind = 0
    m.lr_controller.step()                      # creates state, stage_start_step = 0
    m.step_ind = 500
    m.lr_controller.on_calibration(_reading(64.0))
    assert m.lr_ctrl['peak_scale'] == 1.0
    # 'warmup_ramp' since the freeze-only reversal: the reading is LOOKED at
    # (it can freeze the ramp) but still actuates nothing, which is what the
    # peak_scale assertion above pins
    assert m.lr_controller._last['status'] == 'warmup_ramp'

    m.lr_controller.on_plateau(True, 0.5)
    assert m.lr_ctrl['peak_scale'] == 1.0
    assert m.lr_controller._plateau_last['status'] == 'warmup'

    assert m.lr_controller.in_warmup() is True
    m.step_ind = 1001
    assert m.lr_controller.in_warmup() is False


def test_warmup_is_measured_from_state_creation_not_from_step_zero():
    """
    _elapsed is step_ind - stage_start_step, and a FRESH state stamps
    stage_start_step with whatever step_ind is when it is first built. So a v7
    state discarded at step 8000 (or any other cause of a rebuild mid-run) buys
    a full warmup re-ramp from there, not an immediate release.

    Pinned because it is the difference between "the servo is held for 1000 more
    steps" and "the servo is live", and nothing in the logs distinguishes them
    except lr_ctrl/warmup.
    """
    m = _modeller(**{'adaptive_lr.warmup_steps': 1000})
    m.step_ind = 8000
    m.lr_ctrl = {'ver': 7, 'peak_scale': 55.0}    # stale -> discarded, rebuilt here
    m.lr_controller.step()
    assert m.lr_ctrl['stage_start_step'] == 8000
    assert m.lr_controller.in_warmup() is True
    m.step_ind = 8999
    assert m.lr_controller.in_warmup() is True
    m.step_ind = 9000
    assert m.lr_controller.in_warmup() is False


@pytest.mark.parametrize('status', ['unresolved', 'inconsistent'])
def test_unresolved_and_inconsistent_produce_no_move(status):
    """A calibration that cannot see the answer must not guess it."""
    m = _modeller(**{'adaptive_lr.warmup_steps': 0})
    m.step_ind = 100
    m.lr_controller.on_calibration({'status': status, 'alpha_star': float('nan')})
    assert m.lr_ctrl['peak_scale'] == 1.0
    assert m.lr_controller._last['applied'] == 0.0


def test_update_is_asymmetric():
    """
    peak_scale <- peak_scale * (alpha/target)^eta, with eta_up < eta_down on
    principle: raising is licensed only by a one-step measurement that cannot
    see multi-step effects, while lowering is the safe direction.
    """
    up = _modeller(**{'adaptive_lr.warmup_steps': 0})
    up.step_ind = 100
    up.lr_controller.on_calibration(_reading(16.0))       # 4x target
    assert up.lr_ctrl['peak_scale'] == pytest.approx(4.0 ** 0.25)

    down = _modeller(**{'adaptive_lr.warmup_steps': 0})
    down.step_ind = 100
    down.lr_controller.on_calibration(_reading(1.0))      # 1/4 target
    assert down.lr_ctrl['peak_scale'] == pytest.approx(0.25 ** 0.5)

    # one calibration can halve the rate but cannot double it
    assert (1.0 / down.lr_ctrl['peak_scale']) > up.lr_ctrl['peak_scale']


def test_peak_scale_respects_bounds():
    m = _modeller(**{'adaptive_lr.warmup_steps': 0, 'adaptive_lr.bounds': (0.5, 1.5)})
    m.step_ind = 100
    for _ in range(50):
        m.lr_controller.on_calibration(_reading(1e6))
    assert m.lr_ctrl['peak_scale'] == pytest.approx(1.5)
    for _ in range(50):
        m.lr_controller.on_calibration(_reading(1e-6))
    assert m.lr_ctrl['peak_scale'] == pytest.approx(0.5)


# ----------------------------------------------------------------- divergence

def test_divergence_bar_below_1e5_is_refused():
    """Anything that fires on ordinary training is a graduated cut tier, deleted in v8."""
    with pytest.raises(ValueError, match='below 1e5'):
        _modeller(**{'adaptive_lr.divergence_loss_abs': 1.0e3})


@pytest.mark.parametrize('loss,grad', [(float('nan'), 1.0), (float('inf'), 1.0),
                                       (1.0, float('nan')), (1e12, 1.0), (1.0, 1e12)])
def test_check_spike_catches_explosion(loss, grad):
    m = _modeller()
    assert m.lr_controller.check_spike('fused', loss, grad) == 'diverged'


def test_check_spike_ignores_ordinary_training():
    m = _modeller()
    assert m.lr_controller.check_spike('fused', 5000.0, 900.0) is None


def test_relative_bar_is_ARMED_by_default_and_matches_the_shipping_value():
    """The bench transcribes the shipping config, so the rule is live here too
    (test_fidelity pins that they agree). A steady loss must not trip it, and a
    hundredfold excursion must -- while staying far below the absolute bar."""
    m = _modeller()
    assert float(m.args.adaptive_lr.divergence_loss_rel) == 100.0
    for _ in range(60):
        assert m.lr_controller.check_spike('fused', 1.0, 1.0) is None
    assert m.lr_controller.check_spike('fused', 50.0, 1.0) is None
    assert m.lr_controller.check_spike('fused', 500.0, 1.0) == 'diverged'


def test_relative_bar_can_be_switched_off():
    """null/0 = absolute bars only, for a route that legitimately swings."""
    m = _modeller(**{'adaptive_lr.divergence_loss_rel': None})
    for _ in range(60):
        assert m.lr_controller.check_spike('fused', 1.0, 1.0) is None
    assert m.lr_controller.check_spike('fused', 1.0e6, 1.0) is None


def test_relative_bar_convicts_a_hundredfold_excursion():
    """THE POINT OF THE RULE: a route whose loss lives at O(1) can go up 100x --
    destroying the run -- while staying six orders of magnitude below the
    absolute 1e9 bar."""
    m = _modeller(**{'adaptive_lr.divergence_loss_rel': 100.0})
    for _ in range(60):                       # settle a minimum, past _REL_MIN_OBS
        assert m.lr_controller.check_spike('bwd', 2.0, 1.0) is None
    assert m.lr_controller.check_spike('bwd', 150.0, 1.0) is None, 'under 100x'
    assert m.lr_controller.check_spike('bwd', 250.0, 1.0) == 'diverged'


def test_relative_bar_is_inert_until_it_has_seen_enough():
    """A bar armed on its first reading would let the SECOND observation convict
    the run -- the reference has to be a minimum, not a sample."""
    m = _modeller(**{'adaptive_lr.divergence_loss_rel': 100.0})
    assert m.lr_controller.check_spike('bwd', 1.0, 1.0) is None
    assert m.lr_controller.check_spike('bwd', 1.0e5, 1.0) is None, 'not yet armed'


def test_relative_bar_does_not_convict_a_new_LOW():
    """A genuinely better loss must never trip the rule it is about to become
    the reference for."""
    m = _modeller(**{'adaptive_lr.divergence_loss_rel': 100.0})
    for _ in range(60):
        m.lr_controller.check_spike('bwd', 5.0, 1.0)
    assert m.lr_controller.check_spike('bwd', 0.01, 1.0) is None


def test_relative_bar_is_floored_so_a_near_zero_loss_cannot_arm_a_hair_trigger():
    """Without the floor, a loss touching ~0 makes the bar ~0 and ordinary noise
    reads as divergence."""
    m = _modeller(**{'adaptive_lr.divergence_loss_rel': 100.0})
    for _ in range(60):
        m.lr_controller.check_spike('bwd', 1.0e-12, 1.0)
    # floor 1e-3 * 100 = 0.1, so ordinary small values stay clean
    assert m.lr_controller.check_spike('bwd', 0.05, 1.0) is None


def test_relative_bar_is_per_channel():
    """fwd and bwd differ in scale; a shared minimum would be the smaller of the
    two and would convict the other."""
    m = _modeller(**{'adaptive_lr.divergence_loss_rel': 100.0})
    for _ in range(60):
        m.lr_controller.check_spike('bwd', 0.01, 1.0)
        m.lr_controller.check_spike('fwd', 100.0, 1.0)
    assert m.lr_controller.check_spike('fwd', 150.0, 1.0) is None


def test_relative_bar_reference_is_cleared_at_a_stage_transition():
    """Loss SCALE differs by orders of magnitude between stages, so a minimum
    carried across would convict the next stage on its first ordinary reading."""
    m = _modeller(**{'adaptive_lr.divergence_loss_rel': 100.0})
    for _ in range(60):
        m.lr_controller.check_spike('bwd', 0.01, 1.0)
    m.lr_controller.rearm_warmup()
    for _ in range(60):
        assert m.lr_controller.check_spike('bwd', 50.0, 1.0) is None


def test_divergence_cuts_and_records_a_permanent_ceiling():
    """
    The ceiling is INSTANCE state, never in lr_ctrl: the rewind that follows a
    divergence restores lr_ctrl from a healthy checkpoint and would otherwise
    erase the evidence.
    """
    m = _modeller(**{'adaptive_lr.warmup_steps': 0})
    m.step_ind = 100
    for _ in range(6):
        m.lr_controller.on_calibration(_reading(1024.0))
    hot = m.lr_ctrl['peak_scale']
    assert hot > 2.0

    m.lr_controller.on_divergence()
    assert m.lr_ctrl['peak_scale'] == pytest.approx(hot * 0.5)
    ceiling = m.lr_controller._current_ceiling()
    assert ceiling == pytest.approx(hot * 0.5)

    # simulate the rewind: lr_ctrl restored from a healthy checkpoint
    m.lr_ctrl['peak_scale'] = hot * 4
    m.lr_controller.step()
    assert m.lr_ctrl['peak_scale'] == pytest.approx(ceiling), \
        'the ceiling must survive a state restore, or the rewind erases the evidence'

    # and the servo may never climb back through it
    for _ in range(10):
        m.lr_controller.on_calibration(_reading(4096.0))
    assert m.lr_ctrl['peak_scale'] <= ceiling * (1 + 1e-9)


# ---------------------------------------------------------------- transitions

def test_rearm_warmup_resets_peak_and_forgets_the_ceiling():
    """
    Each stage re-discovers its own peak: the optimizers were rebuilt onto a
    surface with different curvature, so the previous stage's verdict does not
    apply and its ceiling describes a surface that no longer exists.
    """
    m = _modeller(**{'adaptive_lr.warmup_steps': 500})
    m.step_ind = 1000
    m.lr_controller.on_calibration(_reading(1024.0))
    m.lr_controller.on_divergence()
    assert m.lr_controller._current_ceiling() is not None
    before = m.lr_of('fwd')

    m.step_ind = 4000
    warmup = m.lr_controller.rearm_warmup()
    assert warmup == 500
    assert m.lr_ctrl['peak_scale'] == 1.0
    assert m.lr_controller._current_ceiling() is None
    assert m.lr_controller.in_warmup() is True

    # THE RAMP STARTS AT THE OUTGOING RATE (2026-08-20), not at a fixed fraction
    # of seed. This assertion used to read
    #     lr == lr_policy / lr_warmup_ratio
    # which anchors the restart to `seed_lr` -- and a stage that has run for
    # thousands of steps has usually moved a long way from seed, so that rule
    # could RAISE the rate at a transition rather than lower it. Measured on
    # mmnxotsr: train_prior exited at peak 0.0113 x envelope 0.1194 = 1.7e-7 and
    # phase 1 -> 2 landed as a bare 81x step in one optimizer step.
    #
    # So assert the CONTRACT rearm_warmup documents rather than a value:
    # peak_scale going to 1.0 changes no learning rate on the transition step,
    # because the ramp absorbs the reset (here peak 0.5 x env 0.1 becomes
    # peak 1.0 x env 0.05). A hardcoded number would simply re-rot the next time
    # the ramp rule moves.
    assert m.lr_of('fwd') == pytest.approx(before, rel=1e-9), \
        'the reset was not absorbed by the ramp -- the transition moved the LR'
    assert m.lr_ctrl['envelope'] == pytest.approx(0.05, rel=1e-9)


def test_stale_state_is_discarded_never_reinterpreted():
    """v7 accumulated peak_scale by a different rule; reusing it would let a
    deleted state machine steer a controller that no longer has one."""
    m = _modeller(**{'adaptive_lr.warmup_steps': 0})
    m.lr_ctrl = {'ver': 7, 'peak_scale': 55.0, 'disc_state': 'ramp', 'envelope': 1.0}
    m.step_ind = 100
    m.lr_controller.step()
    assert m.lr_ctrl['ver'] == 8
    assert m.lr_ctrl['peak_scale'] == 1.0
    assert 'disc_state' not in m.lr_ctrl


def test_floor_does_not_trigger_a_warm_restart():
    """
    INVERTED 2026-08-14, and the inversion is the whole point of the test.

    This asserted `floor must trigger a warm restart`, on the reasoning in its old
    docstring: the plateau rule is a pure ratchet, so without a restart an over-cut
    run walks to the floor and stays there.

    That is right about the ratchet and wrong about the remedy. Restarting BECAUSE
    peak_scale reached its floor multiplies the LR by 1/floor in a single step --
    100x at the 0.01 bound this test uses -- and that detonated five of six
    qm9anchor_aug14 arms from a healthy state. It also fired regardless of which
    sensor moved peak_scale, so a hypergradient run correctly tracking a descending
    optimum was reset by plateau-rule machinery it never used.

    peak_scale sitting at the floor means THE FLOOR IS TOO HIGH; the fix is to lower
    the bound, not to fire a 100x step change. The floor trigger is therefore gone
    from LRController (see its docstring) and the timed one is the only restart --
    test_timed_warm_restart covers that, and it still passes, so this inversion
    removes a trigger rather than the feature.
    """
    m = _modeller(**{'adaptive_lr.warmup_steps': 0, 'adaptive_lr.bounds': (0.01, 2000.0)})
    m.step_ind = 100
    for _ in range(40):
        m.lr_controller.on_plateau(True, 0.5)
    assert m.lr_ctrl['peak_scale'] == pytest.approx(0.01)
    m.lr_controller.step()
    assert m.lr_ctrl['peak_scale'] == pytest.approx(0.01), (
        'the floor must NOT trigger a warm restart -- at these bounds that is a '
        '100x LR step in one go, which detonated five of six qm9anchor_aug14 arms')


def test_timed_warm_restart():
    m = _modeller(**{'adaptive_lr.warmup_steps': 0, 'adaptive_lr.restart_after': 300})
    m.step_ind = 100
    m.lr_controller.on_plateau(True, 0.5)
    assert m.lr_ctrl['peak_scale'] == pytest.approx(0.5)
    m.step_ind = 500
    m.lr_controller.step()
    assert m.lr_ctrl['peak_scale'] == 1.0


# ------------------------------------------------- closed loop on known ground

def test_servo_cuts_from_a_hot_start():
    """
    Closed loop against a surface whose alpha* is known analytically. Started
    HOT (alpha* below target), the servo must cut -- the direction that matters,
    since undertraining is recovered and damage is not.
    """
    run = BenchRun(
        game='mle', need_batch_sizer=False,
        game_kwargs=dict(dim=16, cond=8.0, noise=0.02, lr=2.0e-1, init_scale=1.0, seed=3),
        args_overrides={'adaptive_lr.warmup_steps': 50, 'adaptive_lr.ray_calibration.period': 50,
                        'lr_policy': 2.0e-1, 'min_lr': 1e-9},
        probe_batch=2048,
    ).run(600)

    assert run.summary()['n_resolved'] >= 8, run.status_counts()
    assert run.summary()['final_peak'] < 0.5, 'a hot start must be cut, not raised'
    assert not run.summary()['diverged']


def test_saturated_sensor_raises_open_loop_at_a_fixed_rate():
    """
    THE RAMP. When the true alpha* is above the grid, every reading comes back
    `above_range` pinned at the largest testable alpha (32 for the shipping grid
    [0..64], since testing alpha needs the loss at 2*alpha). The controller then
    applies a CONSTANT multiplier

        (32 / alpha_target) ^ eta_up = (32/4) ^ 0.25 = 1.682

    every calibration period, carrying no information about how far above the
    grid the truth actually is. That is open loop: 1.68x per period, forever,
    until either a reading lands inside the grid or the divergence bar fires.

    This is pinned rather than fixed because it IS the current design -- a
    reading outside the grid is treated as a bound and never extrapolated. The
    cost of the policy is measured in bench/experiments.py (saturation_ramp).
    """
    m = _modeller(**{'adaptive_lr.warmup_steps': 0})
    m.step_ind = 100
    m.lr_controller.step()               # materialise state
    m.step_ind = 101
    expected = (32.0 / 4.0) ** 0.25
    scales = []
    for _ in range(8):
        before = m.lr_ctrl['peak_scale']
        m.lr_controller.on_calibration(_reading(32.0, status='above_range'))
        scales.append(m.lr_ctrl['peak_scale'] / before)

    assert all(s == pytest.approx(expected) for s in scales), scales
    assert m.lr_ctrl['peak_scale'] == pytest.approx(expected ** 8)
    assert expected ** 8 > 60, 'eight saturated periods is already a >60x ramp'


def test_only_the_divergence_bar_stops_a_saturated_ramp():
    """
    Closed loop from a cold start on an ill-conditioned surface: the step
    direction migrates into the soft subspace as the stiff modes converge, so
    the probe reads ever-lower curvature and stays saturated while the LR climbs
    past 2/lambda_max. Nothing in the calibration path stops it; the absolute
    divergence bar does, and records a ceiling.

    See bench/experiments.py::probe_blindness for the measurement of the
    mechanism. This test only asserts the outcome, so it will start failing the
    moment the ramp acquires any other brake -- which is the point.
    """
    run = BenchRun(
        game='mle', need_batch_sizer=False,
        game_kwargs=dict(dim=32, cond=300.0, noise=0.01, lr=5e-5, init_scale=3.0, seed=3),
        args_overrides={'adaptive_lr.warmup_steps': 50, 'adaptive_lr.ray_calibration.period': 50,
                        'lr_policy': 5e-5, 'min_lr': 1e-10},
        probe_batch=2048,
    ).run(700, stop_on_divergence=False)

    counts = run.status_counts()
    assert counts.get('above_range', 0) > 0.7 * sum(counts.values()), counts
    assert run.divergences > 0, 'the ramp ran to the bar in the reference setup'
    assert run.m.lr_controller._current_ceiling() is not None, 'a ceiling must be recorded'

    lam_max = float(run.game.H.max())
    peak_lr = max(h['lr'] for h in run.history)
    assert peak_lr * lam_max > 2.0, (
        'the reference setup is meant to cross the SGD stability limit; '
        f'reached lr*lambda_max = {peak_lr * lam_max:.2f}')
