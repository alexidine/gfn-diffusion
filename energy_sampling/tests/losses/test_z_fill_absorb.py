"""z_level_fill in 'absorb' mode (a 1-D Kalman filter on log Z), report-only
measurements, and the eval-source prefix.

The snap mode's gates are pinned by tests/losses/test_z_level_fill.py; this file
pins what replaces them: every finite-se measurement moves Z by exactly
K = P_pred / (P_pred + se^2) of its gap, the first one is taken whole, a
report-only call moves nothing and leaves the belief untouched, and the
absorber's state survives a checkpoint field-for-field.
"""
import math
import types

import pytest
import torch

from models.architectures import LearnableScalar
from train import Modeller
from checkpointing import MODELLER_STATE_DEFAULTS

BETA = 10.0


def _stub(logw, current_z=0.0, step=1000, **fill_cfg):
    cfg = types.SimpleNamespace(
        **{'fill_threshold': 0.5, 'fill_se': 3.0, 'fill_cooldown_steps': 0,
           'fill_mode': 'absorb', 'fill_process_var': 0.01, 'fill_moment_reset': 0.5,
           **fill_cfg})
    flow, ema_flow = LearnableScalar(current_z), LearnableScalar(current_z)
    tracker = types.SimpleNamespace(
        clip_beta=BETA, z_bias_ema=torch.tensor([4.0]), z_grad_ema=torch.tensor([4.0]))
    m = types.SimpleNamespace(
        _z_fill_logw=(None if logw is None else torch.as_tensor(logw, dtype=torch.float64)),
        step_ind=step,
        args=types.SimpleNamespace(
            z_calibration=cfg,
            fwd_loss_coeffs=types.SimpleNamespace(beta=BETA, freeze_policy=1.0)),
        gfn_model=types.SimpleNamespace(flow_model=flow, conditional=False, full_flow=False),
        ema_model=types.SimpleNamespace(flow_model=ema_flow),
        optimizers={},
        condition_log_z=tracker,
    )
    m.z_level_fill = types.MethodType(Modeller.z_level_fill, m)
    return m


def _batch(centre, n=2000, spread=3.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g, dtype=torch.float64) * spread + centre


def _z(m):
    return float(m.gfn_model.flow_model.scalar.detach())


def test_first_measurement_is_taken_whole_and_sets_P_to_its_variance():
    m = _stub(_batch(-50.0))
    m.z_level_fill()
    rep = m._z_cal_report
    assert rep['z_fill/K'] == pytest.approx(1.0)
    assert _z(m) == pytest.approx(rep['z_fill/root'])
    assert rep['z_fill/P'] == pytest.approx(rep['z_fill/se'] ** 2)
    assert m._z_fill_last_applied == 1000
    assert float(m.ema_model.flow_model.scalar.detach()) == pytest.approx(_z(m))


def test_precise_measurement_nearly_snaps_and_noisy_one_barely_moves():
    # same gap, two batches: 20000 rows (se ~ 0.02) vs 10 rows (se ~ 1),
    # against a belief of P = 0.01 one step old (P_pred = 0.02)
    for n, lo, hi in ((20000, 0.9, 1.0), (10, 0.0, 0.1)):
        m = _stub(_batch(-50.0, n=n))
        m._z_fill_P = 0.01
        m._z_fill_last_applied = 999
        m.z_level_fill()
        K = m._z_cal_report['z_fill/K']
        assert lo <= K <= hi, (n, K)
        assert _z(m) == pytest.approx(K * m._z_cal_report['z_fill/gap'])


def test_belief_widens_with_steps_since_the_last_measurement():
    Ks = []
    for elapsed in (1, 1000):
        m = _stub(_batch(-50.0, n=40), step=1000)
        m._z_fill_P = 0.05
        m._z_fill_last_applied = 1000 - elapsed
        m.z_level_fill()
        Ks.append(m._z_cal_report['z_fill/K'])
    assert Ks[1] > Ks[0], Ks


def test_kalman_update_is_exact():
    m = _stub(_batch(-50.0, n=400), step=1000)
    P, q, last = 0.3, 0.01, 950
    m._z_fill_P, m._z_fill_last_applied = P, last
    m.z_level_fill()
    rep = m._z_cal_report
    se = rep['z_fill/se']
    P_pred = P + q * (1000 - last)
    K = P_pred / (P_pred + se * se)
    assert rep['z_fill/K'] == pytest.approx(K)
    assert rep['z_fill/dz'] == pytest.approx(K * rep['z_fill/gap'])
    assert rep['z_fill/P'] == pytest.approx(P_pred * se * se / (P_pred + se * se))
    assert _z(m) == pytest.approx(rep['z_fill/dz'])
    # the tracker's level readings shift by the applied move, not the raw gap
    assert float(m.condition_log_z.z_bias_ema[0]) == pytest.approx(4.0 - rep['z_fill/dz'])


def test_all_saturated_batch_is_blocked_not_absorbed():
    # half the rows 1000 nats below any candidate root, half 1000 above: every
    # row is clipped wherever the search lands, so the root is unresolved (se inf)
    m = _stub(torch.tensor([-1000.0] * 64 + [1000.0] * 64, dtype=torch.float64))
    m.z_level_fill()
    assert m._z_cal_report.get('z_fill/blocked_by_se') == 1
    assert 'z_fill/fired' not in m._z_cal_report
    assert _z(m) == 0.0


def test_small_absorbed_move_keeps_adam_moments_large_one_drops_them():
    class _Opt:
        def __init__(self):
            self.state = {}
    for gap, kept in ((0.2, True), (40.0, False)):
        m = _stub(_batch(gap, n=20000, spread=0.5))
        opt = _Opt(); p = m.gfn_model.flow_model.scalar; opt.state[p] = 'moments'
        m.optimizers = {'fused': opt}
        m.z_level_fill()
        assert (p in opt.state) is kept, gap


def test_report_only_measurement_moves_nothing():
    m = _stub(None)
    m._z_fill_P, m._z_fill_last_applied = 0.3, 900
    m.z_level_fill(logw=_batch(-50.0), source='eval', apply=False)
    rep = m._z_cal_report
    assert 'z_fill/eval_gap' in rep and 'z_fill/eval_se' in rep and rep['z_fill/eval_n'] == 2000
    assert 'z_fill/gap' not in rep
    assert _z(m) == 0.0
    assert m._z_fill_P == 0.3 and m._z_fill_last_applied == 900
    assert not hasattr(m, '_z_fill_last_step')


def test_explicit_logw_leaves_the_training_stash_alone():
    m = _stub(_batch(-50.0))
    stash = m._z_fill_logw
    m.z_level_fill(logw=_batch(-20.0), source='eval', apply=True)
    assert m._z_fill_logw is stash
    assert m._z_cal_report['z_fill/eval_gap'] == pytest.approx(m._z_cal_report['z_fill/root'])


def test_snap_mode_is_unchanged_by_the_absorber_fields():
    m = _stub(_batch(-50.0), fill_mode='snap', fill_threshold=20.0, fill_se=5.0)
    m.z_level_fill()
    assert m._z_cal_report['z_fill/fired'] == 1
    assert 'z_fill/K' not in m._z_cal_report
    assert _z(m) == pytest.approx(m._z_cal_report['z_fill/root'])


def test_unknown_fill_mode_is_refused():
    m = _stub(_batch(-50.0), fill_mode='ema')
    with pytest.raises(ValueError):
        m.z_level_fill()


def test_absorber_state_is_checkpointed():
    assert '_z_fill_P' in MODELLER_STATE_DEFAULTS and '_z_fill_last_applied' in MODELLER_STATE_DEFAULTS
    from lr_bracket_probe import TrainerSnapshot
    assert '_z_fill_P' in TrainerSnapshot.FIELDS and '_z_fill_last_applied' in TrainerSnapshot.FIELDS
