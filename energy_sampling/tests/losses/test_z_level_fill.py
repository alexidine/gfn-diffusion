"""`z_level_fill` must re-level log_Z only where the evidence licenses it.

The fill exists because the sidecar servo is magnitude-blind: the Huber clips
dL/dZ at beta and Adam normalises what survives, so log_Z crawls at ~lr_flow
nats/step no matter how far off it is. The fill reads the fixed point straight
off the forward batch's logw instead -- free, because the equilibration stage
already runs fwd as a Z-only branch.

That power is why the gates matter more than the actuator. These tests pin BOTH
directions: that a real gap fires, and that each guard independently refuses.
A test suite that only proved firing would pass a fill wired to no gate at all.
"""
import types

import pytest
import torch

from models.architectures import LearnableScalar
from train import Modeller

BETA = 10.0


def _stub(logw, current_z=0.0, step=1000, conditional=False, full_flow=False,
          freeze_policy=1.0, **fill_cfg):
    cfg = types.SimpleNamespace(
        **{'fill_threshold': 20.0, 'fill_se': 5.0, 'fill_cooldown_steps': 200, **fill_cfg})
    flow, ema_flow = LearnableScalar(current_z), LearnableScalar(current_z)
    tracker = types.SimpleNamespace(
        clip_beta=BETA,
        z_bias_ema=torch.tensor([4.0, float('nan')]),
        z_grad_ema=torch.tensor([4.0, float('nan')]))
    # NB _z_cal_report is deliberately ABSENT. The fill runs before
    # z_calibration_tick, which is what otherwise creates it, so on the first
    # fused step of a run it does not exist -- a stub that pre-creates it cannot
    # see a report path that assumes it does.
    m = types.SimpleNamespace(
        _z_fill_logw=(None if logw is None else torch.as_tensor(logw, dtype=torch.float64)),
        step_ind=step,
        args=types.SimpleNamespace(
            z_calibration=cfg,
            fwd_loss_coeffs=types.SimpleNamespace(beta=BETA, freeze_policy=freeze_policy)),
        gfn_model=types.SimpleNamespace(flow_model=flow, conditional=conditional,
                                        full_flow=full_flow),
        ema_model=types.SimpleNamespace(flow_model=ema_flow),
        optimizers={},
        condition_log_z=tracker,
    )
    return m


def _far_batch(centre=-500.0, n=2000, seed=0):
    """A well-resolved batch a long way from log_Z = 0."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, dtype=torch.float64) * 3.0 + centre


def _z(m):
    return float(m.gfn_model.flow_model.scalar.detach())


# ------------------------------------------------------------------- it fires

def test_a_large_well_resolved_gap_fills():
    m = _stub(_far_batch(), current_z=0.0)
    Modeller.z_level_fill(m)
    assert m._z_cal_report.get('z_fill/fired') == 1
    assert _z(m) == pytest.approx(-500.0, abs=1.0)


def test_the_ema_model_is_filled_too():
    """Backward sampling runs off the EMA copy. Filling only the live scalar
    leaves it on the pre-fill level, and the EMA rule then drags the live value
    back toward it."""
    m = _stub(_far_batch(), current_z=0.0)
    Modeller.z_level_fill(m)
    assert float(m.ema_model.flow_model.scalar.detach()) == pytest.approx(_z(m))


def test_adam_moments_for_the_flow_scalar_are_dropped():
    """Stale moments describe a pre-fill gradient. Left in place they push the
    head straight back off the level just set."""
    m = _stub(_far_batch(), current_z=0.0)
    p = next(m.gfn_model.flow_model.parameters())
    keep = torch.nn.Parameter(torch.zeros(1))
    for name in ('fused', 'flow'):
        m.optimizers[name] = types.SimpleNamespace(
            state={p: {'exp_avg': torch.ones(())}, keep: {'exp_avg': torch.ones(())}})
    Modeller.z_level_fill(m)
    for name in ('fused', 'flow'):
        assert p not in m.optimizers[name].state
        assert keep in m.optimizers[name].state, 'cleared more than the flow head'


def test_tracker_level_emas_shift_by_the_gap():
    """Stored residual is logw - log_Z_learned, so raising log_Z by `gap` must
    LOWER every stored level reading by the same amount. Getting this sign wrong
    doubles the apparent error instead of clearing it."""
    m = _stub(_far_batch(), current_z=0.0)
    Modeller.z_level_fill(m)
    gap = m._z_cal_report['z_fill/gap']
    assert gap < 0
    assert float(m.condition_log_z.z_bias_ema[0]) == pytest.approx(4.0 - gap)
    # winsorized companion stays inside its own clip
    assert abs(float(m.condition_log_z.z_grad_ema[0])) <= BETA
    assert torch.isnan(m.condition_log_z.z_bias_ema[1]), 'unvisited slot was written'


# ----------------------------------------------------------- and it refuses to

def test_a_gap_under_fill_threshold_does_not_fire():
    """THE knob. A 12-nat gap is inside the servo's competence and must be left
    to it."""
    m = _stub(_far_batch(centre=-12.0), current_z=0.0)
    Modeller.z_level_fill(m)
    assert 'z_fill/fired' not in m._z_cal_report
    assert _z(m) == 0.0
    # and the same batch DOES fire once the bar is lowered under the gap --
    # otherwise this test would pass on a fill that never fires at all
    m2 = _stub(_far_batch(centre=-12.0), current_z=0.0, fill_threshold=5.0)
    Modeller.z_level_fill(m2)
    assert m2._z_cal_report.get('z_fill/fired') == 1


def test_an_unresolved_batch_is_blocked_by_the_se_gate():
    """Every row saturated: the loss is flat in z over a wide interval, the root
    is not identified, and se is +inf. The gap is enormous, so ONLY the se gate
    can refuse this one."""
    logw = torch.tensor([-1000.0] * 64 + [1000.0] * 64, dtype=torch.float64)
    m = _stub(logw, current_z=0.0)
    Modeller.z_level_fill(m)
    assert m._z_cal_report.get('z_fill/blocked_by_se') == 1
    assert 'z_fill/fired' not in m._z_cal_report
    assert _z(m) == 0.0


def test_cooldown_blocks_a_second_fill():
    m = _stub(_far_batch(), current_z=0.0, step=1000)
    Modeller.z_level_fill(m)
    assert m._z_fill_last_step == 1000

    m._z_fill_logw = _far_batch(centre=-900.0)
    m.step_ind = 1100                       # inside the 200-step cooldown
    Modeller.z_level_fill(m)
    assert m._z_cal_report['z_fill/fired'] == 1, 'fired again inside the cooldown'

    m._z_fill_logw = _far_batch(centre=-900.0)
    m.step_ind = 1300                       # past it
    Modeller.z_level_fill(m)
    assert m._z_cal_report['z_fill/fired'] == 2


def test_zero_threshold_disables_the_fill():
    m = _stub(_far_batch(), current_z=0.0, fill_threshold=0.0)
    Modeller.z_level_fill(m)
    assert _z(m) == 0.0
    assert 'z_fill/gap' not in getattr(m, '_z_cal_report', {}), \
        'solved the root on a disabled fill'


def test_the_batch_is_consumed_so_one_batch_cannot_fill_twice():
    """Two calls without an intervening step must not both act -- the stash is
    single-use, so a fill can never compound off one measurement."""
    m = _stub(_far_batch(), current_z=0.0, fill_cooldown_steps=0)
    Modeller.z_level_fill(m)
    Modeller.z_level_fill(m)
    assert m._z_cal_report['z_fill/fired'] == 1


def test_a_non_finite_batch_is_reported_not_filled():
    logw = _far_batch()
    logw[3] = float('nan')
    m = _stub(logw, current_z=0.0)
    Modeller.z_level_fill(m)
    assert m._z_cal_report.get('z_fill/bad_batch') == 1
    assert _z(m) == 0.0


# ----------------------------------------------------- and it is never armed at all

def _stash(**kw):
    m = _stub(None, **kw)
    n = 128
    loss_dict = {'log_pb': torch.zeros(n), 'log_r': torch.full((n,), -500.0),
                 'log_pf': torch.zeros(n)}
    Modeller._stash_z_fill_logw(m, loss_dict)
    return m._z_fill_logw


def test_stash_arms_on_the_unconditional_z_only_forward_branch():
    """The equilibration configuration: fwd is freeze_policy on a scalar head."""
    got = _stash()
    assert got is not None and got.shape == (128,)
    assert float(got[0]) == pytest.approx(-500.0)


def test_stash_refuses_a_training_policy():
    """Without freeze_policy the level and the policy move together, so the
    fixed point this batch reports is still moving and is not a target."""
    assert _stash(freeze_policy=0.0) is None


def test_stash_refuses_conditional_and_full_flow_heads():
    """Neither has a single scalar to fill -- the level is a field."""
    assert _stash(conditional=True) is None
    assert _stash(full_flow=True) is None


def test_z_calibration_reports_what_it_costs():
    """A SUBSYSTEM ALLOWED TO TRIPLE A TRAIN STEP MUST REPORT ITS SECONDS.

    The interspersed z-calibration loop may run max_steps_per_step rollouts per
    TRAIN step -- 100 canonically -- and each is a full T-step policy rollout, so
    it can outweigh the MLIP call several times over. It logged steps, p, sensor,
    fresh, early_out, rollout_loss and rollout_errors, and no timing at all.

    On p4_mace_mle's phase 2 that left energy/frac_of_step reading 0.045 against
    a 7.4 s step with nothing to compare it to: z_cal was running 2-7 rollouts a
    step and its cost had to be INFERRED from rollout-count arithmetic.

    Source-level because driving the real loop needs a model, a prior and a GPU,
    while the property is structural -- the timer must bracket the DISPATCH, and
    the fraction must use the denominator energy/frac_of_step uses, or the two
    cannot be read against each other."""
    import inspect
    import train

    src = inspect.getsource(train.Modeller.z_calibration_tick)
    assert "rep['z_cal/seconds']" in src, (
        'the z-calibration loop does not accumulate seconds; its cost is '
        'unmeasurable and has to be inferred from rollout-count arithmetic')

    # the timer must open IMMEDIATELY before the mode dispatch, not somewhere
    # else in the method -- an index comparison would be satisfied by a docstring
    lines = [l.strip() for l in src.splitlines()]
    d = next(i for i, l in enumerate(lines) if l.startswith("if mode == 'rollout'"))
    assert any(l == '_t0 = time()' for l in lines[max(0, d - 3):d]), (
        'the timer does not open just before the rollout dispatch, so it is not '
        'timing the rollout')

    rep = inspect.getsource(train.Modeller.ten_step_reporting)
    assert 'z_cal/frac_of_step' in rep, 'the fraction is never emitted'
    assert "self._throughput['seconds']" in rep, (
        'z_cal/frac_of_step uses a different denominator from '
        'energy/frac_of_step, so the two cannot be read against each other')
