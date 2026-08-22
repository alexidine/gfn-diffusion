"""
The pre-draw gate on the ray probe, and what is left of it after the freeze-only
warmup reversal.

HISTORY, in two moves:

  * F-039 (findings.md): `RayCalibration.measure` drew `n_sub` replay sub-batches
    whose RNG nothing restores, so a calibration whose reading the controller
    then refused still shifted every subsequent training step. The fix was a
    pre-draw predicate, `LRController.calibration_refusal`, consulted by
    `Modeller._ray_probe_armed` BEFORE any draw, with warmup as the one
    refusable case.

  * THE REVERSAL (controller.py, `calibration_refusal`): warmup is no longer a
    refusal. The envelope ramp needs something watching it -- a resolved
    alpha* below target is the "stop climbing" verdict the warmup freeze
    consumes -- so the probe now DRAWS during warmup and only ACTUATION is
    withheld (`on_calibration` reports 'warmup_ramp', applied 0.0, and may
    freeze the ramp). The RNG cost F-039 measured is accepted, and the
    controller comment records that runs are not comparable across the change.

WHAT THESE TESTS PIN NOW:

  * the predicate currently refuses nothing, and the two consumers of it (the
    probe gate and `on_calibration`) therefore agree -- one function, no drift;
  * during the ramp a resolved reading is looked at but not acted on;
  * the refusal MACHINERY (`RayCalibration.refuse`) still consumes a period and
    still reports, so a future refusable case slots back in without moving the
    applied path;
  * probes land on period boundaries, warmup included.

These drive the REAL `Modeller._ray_probe_armed` and the REAL `LRController`,
not a local re-implementation of the call order.
"""

import pytest
import torch

from bench.fake_modeller import (FakeModeller, FakeStage,
                                 attach_real_probe_gate, make_args)
from energy_sampling.controller import LRController
from energy_sampling.lr_larder import Harvested
from energy_sampling.ray_calibration import RayCalibration

WARMUP = 1000
PERIOD = 500
N_SUB = 8


def _stock(larder, branches, step):
    """Backfill the larder so the gate's readiness check is satisfied.

    The gate needs n_sub records per active branch drawn on steps BEFORE the one
    being rated; only `.step` is read here, so the payload is a placeholder. The
    tests that care what is IN a record are in test_lr_larder.py.
    """
    for b in branches:
        for k in range(N_SUB):
            larder.record(Harvested(branch=b, step=step - N_SUB + k,
                                    condition=None, condition_id=None, log_r=None,
                                    mol_batch=None, traj=None, repeats=1,
                                    scramble_tiles=0, sample_weights=None))


def _modeller(sensor_kind='ray', warmup_steps=WARMUP):
    attach_real_probe_gate()
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


import train  # noqa: E402  -- 11 s; this module drives the REAL gate


def _drive(m, p, last_step, gated=True):
    """Walk the training loop the way train.py does: decide, step, measure.

    `gated=False` reproduces the pre-F-039 call order -- arm unconditionally --
    which, with nothing currently refusable, must now be indistinguishable."""
    out = {'n': 0, 'steps': [], 'probe_steps': [], 'applied_steps': []}

    def draw_fn():
        out['n'] += 1
        out['steps'].append(int(m.step_ind))
        return torch.zeros(2)

    def loss_fn(_batch):
        return float(p.detach().sum())

    for step in range(1, last_step + 1):
        m.step_ind = step
        # what train_step does at the top of every accumulation cycle: mark
        # where the pending step's own training data starts, then harvest into
        # the larder as the branches run
        m._probe_exclude_from = step
        _stock(m.larder, m._probe_weights, step)
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
            if reading is not None and refusal is None:
                out['applied_steps'].append(step)
    return out


# ------------------------------------------------------------- the predicate --

def test_refusal_is_currently_never():
    """The reversal, pinned: warmup no longer refuses, and nothing else is
    decidable before the draw. If a refusable case is ever reintroduced this
    fails first, which is the desired tripwire -- the probe-side and
    controller-side consumers below both assume the predicate's answer."""
    m, _ = _modeller()
    for step in (0, WARMUP - 1, WARMUP, WARMUP + PERIOD):
        m.step_ind = step
        assert m.lr_controller.calibration_refusal() is None


def test_on_calibration_looks_but_does_not_actuate_during_the_ramp():
    """Freeze-only warmup: a resolved reading inside the ramp is consumed as
    'warmup_ramp' with nothing applied.

    Past the ramp the applied path must still APPLY -- but under the pooled
    estimator (handoff 6B) that takes a settled stage and `pool_min_readings`
    consistent readings, not one. Both halves are asserted, so this cannot pass
    on a controller whose actuation is simply dead.
    """
    m, _ = _modeller()
    reading = {'status': 'bracketed', 'alpha_star': 8.0, 'lo': 4.0, 'hi': 16.0}

    m.step_ind = 10
    m.lr_controller.on_calibration(reading)
    assert m.lr_controller._last['status'] == 'warmup_ramp'
    assert m.lr_controller._last['applied'] == 0.0

    # inside the stage's opening transient the reading is LOOKED AT and logged,
    # and deliberately not pooled: it describes the transition, not the stage
    m.step_ind = WARMUP + 10
    m.lr_controller.tick()
    assert m.lr_controller.calibration_refusal() is None
    m.lr_controller.on_calibration(reading)
    assert m.lr_controller._last['status'] == 'unsettled'
    assert m.lr_controller._last['applied'] == 0.0

    # settled, and enough consistent evidence: alpha* 8 against target 4 asks
    # for a 2x raise, and the estimator moves the whole way there
    before = m.lr_ctrl['peak_scale']
    for i in range(m.lr_controller.pool.min_readings):
        m.step_ind = WARMUP + 1000 + 500 * i
        m.lr_controller.tick()
        m.lr_controller.on_calibration(reading)
    assert m.lr_controller._last['status'] == 'bracketed'
    assert m.lr_controller._last['applied'] > 0.0, 'the applied path must still apply'
    assert m.lr_ctrl['peak_scale'] == pytest.approx(2.0 * before, rel=1e-6)


def test_the_retired_servo_is_still_reachable_and_is_not_the_default():
    """`mode: servo` restores the per-reading rule, so the replacement can be
    compared against it rather than argued about. One reading, one move -- which
    is the property that made it track noise."""
    m, _ = _modeller()
    assert m.lr_controller._cal_mode() == 'pooled'
    m.args.adaptive_lr.calibration.mode = 'servo'
    st = m.lr_controller._state()          # stamps the warmup clock at step 0
    m.step_ind = WARMUP + 10               # ...so the ramp is behind us here
    before = st['peak_scale']
    m.lr_controller.on_calibration({'status': 'bracketed', 'alpha_star': 8.0})
    assert st['peak_scale'] == pytest.approx(before * 2.0 ** 0.25, rel=1e-6),         'one reading, one move -- the property that made the servo track noise'

    m.args.adaptive_lr.calibration.mode = 'nonsense'
    with pytest.raises(ValueError, match='calibration.mode'):
        m.lr_controller._cal_mode()


# ------------------------------------------------------- draws during warmup --

def test_probe_draws_during_warmup():
    """The reversal's observable half: the ramp is watched, so the first period
    boundary inside warmup takes a full probe's worth of draws. (This is the
    exact assertion F-039's fix inverted; the inversion back is deliberate and
    the RNG cost is accepted -- see controller.calibration_refusal.)"""
    m, p = _modeller()
    draws = _drive(m, p, last_step=WARMUP - 1)
    assert draws['n'] == N_SUB
    assert draws['steps'] == [PERIOD] * N_SUB, (
        'the probe should fire at the first period boundary inside warmup')


def test_gated_and_ungated_call_orders_agree_while_nothing_is_refusable():
    """With the predicate returning None everywhere, the F-039 gate must be
    TRANSPARENT: same probe steps, same draw count, same applied steps as the
    ungated order. A difference here means a refusal path re-opened without
    this file noticing."""
    g, pg = _modeller()
    u, pu = _modeller()
    gated = _drive(g, pg, last_step=2600, gated=True)
    ungated = _drive(u, pu, last_step=2600, gated=False)

    assert gated['applied_steps'], 'no calibration was ever applied; vacuous'
    assert gated['probe_steps'] == ungated['probe_steps']
    assert gated['applied_steps'] == ungated['applied_steps']
    assert gated['n'] == ungated['n']


def test_first_probe_lands_on_a_period_boundary():
    m, p = _modeller()
    draws = _drive(m, p, last_step=WARMUP + PERIOD)
    assert draws['steps'], 'the probe never fired'
    assert all(s % PERIOD == 0 for s in draws['probe_steps']), (
        f'probes at {draws["probe_steps"]}, expected period boundaries only')


# ------------------------------------------- the refusal machinery survives ---

def test_a_refusal_still_consumes_its_period():
    """`refuse` is currently unreachable from the shipping predicate, but it is
    the slot a future refusable case plugs into, and its contract -- consume the
    period exactly as a completed calibration would, so the applied path does
    not move -- must hold when it comes back."""
    r = RayCalibration([torch.nn.Parameter(torch.zeros(4))],
                       n_sub=N_SUB, period=PERIOD, enabled=True)
    assert r.due(PERIOD) is False, 'first sight only records the baseline'
    assert r.due(2 * PERIOD) is True
    assert r.refuse('warmup', 2 * PERIOD) is True
    assert r._last_done == 2
    assert r.due(3 * PERIOD - 1) is False
    assert r.due(3 * PERIOD) is True


def test_a_refused_probe_is_counted_and_reported():
    """A probe that is deliberately silent and one that was never wired up must
    not look alike from the logs: `raycal/refused` is the discriminator, and a
    refuse-only history reports no measurement keys."""
    r = RayCalibration([torch.nn.Parameter(torch.zeros(4))],
                       n_sub=N_SUB, period=PERIOD, enabled=True)
    r.due(PERIOD)                      # first sight: baseline only
    r.due(2 * PERIOD)
    r.refuse('warmup', 2 * PERIOD)
    assert r.n_refused == 1
    rep = r.report()
    assert rep.get('raycal/refused') == 1.0
    assert rep.get('raycal/refused_reason') == float(RayCalibration._REFUSAL['warmup'])
    assert 'raycal/alpha_star' not in rep, 'nothing was measured, so nothing may be reported'


def test_a_non_ray_stage_never_reaches_the_gate():
    """Unchanged behaviour: the sensor kind is still the outer switch."""
    m, p = _modeller(sensor_kind='hyper')
    draws = _drive(m, p, last_step=WARMUP + PERIOD)
    assert draws['n'] == 0
    assert m.ray_cal.n_refused == 0, (
        'a stage that never asked for the probe must not be counted as refusing it')


# ------------------------------------- the larder-backed gate (handoff 6A) ----

def test_a_filling_larder_defers_without_consuming_the_period():
    """A stage entry is exactly when the harvest is thinnest -- the transition
    clears it -- so a short larder must NOT burn the calibration. It defers,
    the latch stands, and the reading lands as soon as the ring is deep enough.

    The predecessor design lost its most important reading this way: a
    cold-start escape race was silently dropped because the flag was consumed on
    the attempt rather than on a completed measurement.
    """
    m, p = _modeller()
    m.ray_cal.due(PERIOD)                       # first sight records the baseline
    m.step_ind = 2 * PERIOD
    m._probe_exclude_from = m.step_ind
    _stock(m.larder, m._probe_weights, m.step_ind)
    m.larder.rings['bwd'].pop()                 # one short on one branch

    assert train.Modeller._ray_probe_armed(m) is False
    assert m.ray_cal.n_refused == 0, 'a filling larder is not a refusal'
    assert m.ray_cal.n_deferred == 1
    assert m.ray_cal.skip_reason == 'larder_filling'

    # ...and the very next step, once the ring is full, it fires
    m.step_ind = 2 * PERIOD + 1
    m._probe_exclude_from = m.step_ind
    _stock(m.larder, m._probe_weights, m.step_ind)
    assert train.Modeller._ray_probe_armed(m) is True


def test_an_unscoreable_active_branch_refuses_the_period():
    """Structural, so it consumes the period rather than deferring: a fwd bank
    carrying a term the replay evaluator has no counterpart for does not become
    scoreable by waiting. `var_conditioning` ships `emp_z: 1.0`, so this is the
    live case."""
    m, p = _modeller()
    m.ray_cal.due(PERIOD)                       # first sight records the baseline
    m.args.fwd_loss_coeffs.emp_z = 1.0
    m.step_ind = 2 * PERIOD
    m._probe_exclude_from = m.step_ind
    _stock(m.larder, m._probe_weights, m.step_ind)

    assert train.Modeller._ray_probe_armed(m) is False
    assert m.ray_cal.n_refused == 1
    assert m.ray_cal.refuse_reason == 'branch_refused'
    assert m.ray_cal.report()['raycal/refused_reason'] == float(
        RayCalibration._REFUSAL['branch_refused'])
    # the period is consumed, exactly as a completed calibration consumes it
    assert m.ray_cal.due(2 * PERIOD + 1) is False


def test_the_composite_is_the_weighted_sum_the_step_descended():
    """`_probe_loss` returns each branch's own number AND the frac-weighted sum,
    and the sum uses the weights the step actually formed -- not the configured
    fracs. The real method runs here; only the per-record scorer is stubbed, so
    what is under test is the weighting, not the loss."""
    m, _ = _modeller()
    m._probe_weights = {'fwd': 0.25, 'bwd': 0.25, 'replay': 0.5}
    values = {'fwd': 10.0, 'bwd': 20.0, 'replay': 4.0}

    class _Scorer:
        def score(self, rec, _discretizer):
            return values[rec.branch]

    m.larder_scorer = _Scorer()
    drawn = {b: Harvested(branch=b, step=0, condition=None, condition_id=None,
                          log_r=None, mol_batch=None, traj=None, repeats=1,
                          scramble_tiles=0, sample_weights=None)
             for b in values}

    out = train.Modeller._probe_loss(m, drawn, discretizer=None)
    assert out['fwd'] == 10.0 and out['bwd'] == 20.0 and out['replay'] == 4.0
    assert out['composite'] == pytest.approx(0.25 * 10 + 0.25 * 20 + 0.5 * 4)
