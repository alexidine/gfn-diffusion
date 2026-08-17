"""
The pre-draw gate on the ray probe (findings.md F-039).

WHAT THE BUG WAS. `RayCalibration.measure` drew `n_sub` sub-batches from the
replay buffer and scored them at every alpha, and only afterwards handed the
reading to `LRController.on_calibration`, which could refuse it outright. The
draws consume RNG that nothing restores, so a calibration that changed no
learning rate still shifted every subsequent training step. Measured on a
600-step tier-C pair: bit-identical to step 500, divergent from step 501 -- the
first probe -- with every LR bit-identical and `cal_applied` at 0.0.

WHAT THESE TESTS PIN, and why each is here rather than left to the tier-C run:

  * the probe takes NO draws while the reading would be refused -- the fix;
  * it still draws once the refusal lifts -- so the fix is not "never probe",
    which would pass a naive no-draw assertion while deleting the sensor;
  * a refusal consumes its period exactly as a completed calibration does -- so
    the first APPLIED calibration lands on the step it always did. This is the
    "behaviour when applied is unchanged" requirement, and it is the part most
    easily got wrong: leaving the latch pending makes the probe fire on the
    first step after warmup instead of the next period boundary.

These drive the REAL `Modeller._ray_probe_armed` and the REAL `LRController`,
not a local re-implementation of the call order. A copy of the gate would pass
its own tests while the shipped one regressed.
"""

import pytest
import torch

from bench.fake_modeller import FakeModeller, FakeStage, make_args
from energy_sampling.controller import LRController
from energy_sampling.ray_calibration import RayCalibration

WARMUP = 1000
PERIOD = 500
N_SUB = 8


def _modeller(sensor_kind='ray', warmup_steps=WARMUP):
    args = make_args(**{'adaptive_lr.warmup_steps': warmup_steps})
    p = torch.nn.Parameter(torch.zeros(4))
    q = torch.nn.Parameter(torch.zeros(4))
    opts = {
        'fwd': torch.optim.SGD([p], lr=args.lr_policy),
        'flow': torch.optim.SGD([q], lr=args.lr_flow),
        'fused': torch.optim.SGD([{'params': [p]}, {'params': [q]}], lr=args.lr_fused),
    }
    stage = FakeStage(lr_sensor=({'kind': sensor_kind} if sensor_kind else None))
    m = FakeModeller(args, opts, stage=stage)
    m.ray_cal = RayCalibration([p], n_sub=N_SUB, period=PERIOD, enabled=True)
    m.lr_controller = LRController(m)
    return m, p


def _drive(m, p, last_step, gated=True):
    """Walk the training loop the way train.py does: decide, step, measure.

    `gated=False` reproduces the pre-fix call order -- arm unconditionally --
    so the tests can show the same assertions failing with the bug back in."""
    import train

    out = {'n': 0, 'steps': [], 'probe_steps': [], 'applied_steps': []}

    def draw_fn():
        out['n'] += 1
        out['steps'].append(int(m.step_ind))
        return torch.zeros(2)

    def loss_fn(_batch):
        return float(p.detach().sum())

    for step in range(1, last_step + 1):
        m.step_ind = step
        refusal = m.lr_controller.calibration_refusal()
        if gated:
            armed = train.Modeller._ray_probe_armed(m)
        else:
            sensor = m.protocol.stage.lr_sensor
            armed = (sensor is not None and sensor['kind'] == 'ray'
                     and m.ray_cal.arm(step))
        with torch.no_grad():           # the optimizer step the probe rates
            p.add_(0.01)
        if armed:
            out['probe_steps'].append(step)
            reading = m.ray_cal.measure(draw_fn, loss_fn)
            # What the controller would DO with it. `refusal is None` is the
            # same predicate on_calibration applies, read at the same step.
            if reading is not None and refusal is None:
                out['applied_steps'].append(step)
    return out


# ------------------------------------------------------------- the predicate --

def test_refusal_is_warmup_inside_the_envelope_and_none_outside():
    m, _ = _modeller()
    m.step_ind = 0
    assert m.lr_controller.calibration_refusal() == 'warmup'
    m.step_ind = WARMUP - 1
    assert m.lr_controller.calibration_refusal() == 'warmup'
    m.step_ind = WARMUP
    assert m.lr_controller.calibration_refusal() is None


def test_on_calibration_uses_the_same_predicate_so_the_two_cannot_drift():
    """The gate that skips the probe and the gate that refuses the reading are
    one function. Two copies would drift, and the drift would be silent: the
    probe would skip while the controller still acted, or the reverse."""
    m, _ = _modeller()
    reading = {'status': 'bracketed', 'alpha_star': 8.0}

    m.step_ind = 10
    m.lr_controller.on_calibration(reading)
    assert m.lr_controller._last['status'] == 'warmup'
    assert m.lr_controller._last['applied'] == 0.0

    m.step_ind = WARMUP + 10
    assert m.lr_controller.calibration_refusal() is None
    m.lr_controller.on_calibration(reading)
    assert m.lr_controller._last['status'] == 'bracketed'
    assert m.lr_controller._last['applied'] > 0.0, 'the applied path must still apply'


# ------------------------------------------------------------------ the fix --

def test_probe_takes_no_draws_while_the_reading_would_be_refused():
    """THE FIX. Every draw inside warmup is RNG the run cannot get back."""
    m, p = _modeller()
    draws = _drive(m, p, last_step=WARMUP - 1)
    assert draws['n'] == 0, (
        f'probe drew {draws["n"]} sub-batches at steps {draws["steps"]} whose '
        f'reading the controller would have discarded')


def test_the_bug_reintroduced_does_draw_so_the_test_can_fail():
    """A test that cannot fail is not evidence. With the pre-fix call order the
    same window draws a full probe's worth."""
    m, p = _modeller()
    draws = _drive(m, p, last_step=WARMUP - 1, gated=False)
    assert draws['n'] == N_SUB
    assert draws['steps'] == [PERIOD] * N_SUB, (
        'the pre-fix probe should fire at the first period boundary inside warmup')


def test_probe_still_draws_once_the_refusal_lifts():
    """The fix must not be 'never probe'. A gate that always refused would pass
    the no-draw assertion above and quietly delete the sensor.

    Note WHERE the first draw lands: not at the step warmup ends, but at the
    next period boundary. The refusal consumed the period containing that
    boundary, which is precisely what the old code did too -- it probed there,
    completed, and had the reading thrown away. Asserting a draw at
    `WARMUP + 1` is the wrong expectation and this test used to carry it."""
    m, p = _modeller()
    draws = _drive(m, p, last_step=WARMUP + PERIOD)
    assert draws['n'] == N_SUB
    assert all(s >= WARMUP for s in draws['steps'])
    assert draws['steps'][0] % PERIOD == 0


# ------------------------------------------- the applied path is unchanged ---

def test_a_refusal_consumes_its_period_exactly_as_a_calibration_does():
    """`due` latches from the moment a calibration falls due until one
    completes. If a refusal left the latch pending, the probe would fire on the
    FIRST step after warmup rather than at the next period boundary -- a change
    to the applied path, which is what this fix is not allowed to touch."""
    gated, _ = _modeller()
    plain, _ = _modeller()

    # Same walk, one refusing and one completing, must leave the same latch.
    for step in range(1, WARMUP):
        gated.step_ind = step
        if gated.lr_controller.calibration_refusal() is not None:
            gated.ray_cal.refuse('warmup', step)
        plain.ray_cal.due(step)          # first-sight bookkeeping only
        if step == PERIOD:
            plain.ray_cal._last_done = step // PERIOD   # as a completed measure would

    assert gated.ray_cal._last_done == plain.ray_cal._last_done
    # And neither is due again until the next boundary.
    assert gated.ray_cal.due(WARMUP - 1) is False
    assert gated.ray_cal.due(2 * PERIOD) is True


def test_applied_calibrations_land_on_exactly_the_same_steps_as_before_the_fix():
    """THE REQUIREMENT, stated directly: the fix removes draws on the refused
    paths ONLY, so every calibration that is actually acted on must happen at
    the same step it did before.

    Proven by running both call orders over the same 2600-step walk and
    comparing the applied-step lists element for element -- not by reasoning
    about the latch. The gated run probes strictly fewer times (that IS the
    fix), and the two agree exactly on the subset that matters."""
    g, pg = _modeller()
    u, pu = _modeller()
    gated = _drive(g, pg, last_step=2600, gated=True)
    buggy = _drive(u, pu, last_step=2600, gated=False)

    assert gated['applied_steps'], 'no calibration was ever applied; vacuous'
    assert gated['applied_steps'] == buggy['applied_steps'], (
        f"applied at {gated['applied_steps']} with the gate, "
        f"{buggy['applied_steps']} without -- the fix moved the applied path")
    assert len(gated['probe_steps']) < len(buggy['probe_steps']), (
        'the gated run must probe strictly fewer times, or nothing was saved')
    assert gated['n'] < buggy['n']


def test_first_applied_calibration_lands_on_a_period_boundary():
    m, p = _modeller()
    draws = _drive(m, p, last_step=WARMUP + PERIOD)
    assert draws['steps'], 'the probe never fired'
    first = draws['steps'][0]
    assert first % PERIOD == 0, f'first probe at {first}, not a period boundary'
    assert first >= WARMUP


# ------------------------------------------------------------- visibility ----

def test_a_refused_probe_is_counted_and_reported():
    """A probe that is deliberately silent and one that was never wired up must
    not look alike from the logs. With the gate in place there is no measurement
    at all inside warmup, so every `raycal/*` measurement key is absent, and
    `raycal/refused` is the only thing distinguishing the two."""
    m, p = _modeller()
    _drive(m, p, last_step=WARMUP - 1)
    assert m.ray_cal.n_refused >= 1
    rep = m.ray_cal.report()
    assert rep.get('raycal/refused') == float(m.ray_cal.n_refused)
    assert rep.get('raycal/refused_reason') == float(
        RayCalibration._REFUSAL['warmup'])
    assert 'raycal/alpha_star' not in rep, 'nothing was measured, so nothing may be reported'


def test_a_non_ray_stage_never_reaches_the_gate():
    """Unchanged behaviour: the sensor kind is still the outer switch."""
    m, p = _modeller(sensor_kind='hyper')
    draws = _drive(m, p, last_step=WARMUP + PERIOD)
    assert draws['n'] == 0
    assert m.ray_cal.n_refused == 0, (
        'a stage that never asked for the probe must not be counted as refusing it')
