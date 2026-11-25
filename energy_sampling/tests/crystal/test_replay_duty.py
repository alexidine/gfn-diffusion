"""The duty replay rule (crystal_store.duty; train.py Modeller._store_replay_rows; config_invariants.row_store_problems), on
a real CrystalStore. CPU, no energy computed.

    python -m pytest -q tests/crystal/test_replay_duty.py
"""
import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

import config_invariants as ci
from energy_sampling import crystal_store as CS
from energy_sampling.tests.crystal import store_stub as S

pytestmark = pytest.mark.fast

CONFIGS = Path(__file__).resolve().parents[2] / 'configs'


def admitted(cid, log_r, log_pf, log_z=3.0):
    """A stub Modeller whose rollout rows have the given columns; returns (modeller, their slots)."""
    m = S.build()
    fwd, batch = S.rollout(cid, log_r=log_r, log_pf=log_pf)
    fwd['log_Z_learned'] = torch.full((len(cid),), float(log_z))
    return m, m._store_admit(fwd, batch, CS.ORIGIN_ROLLOUT)


def test_resid_is_minus_r_on_an_admitted_row():
    log_r, log_pf, log_z = [-1.0, -8.0], [-5.0, -2.0], 3.0
    m, rows = admitted([0, 0], log_r, log_pf, log_z)
    r = torch.tensor(log_r) - torch.tensor(log_pf) - log_z
    assert torch.allclose(m.row_store.resid[rows], -r)


def test_a_row_still_asking_q_up_keeps_full_weight_however_far_q_fell():
    m, rows = admitted([0], [10.0], [-2.0])         # r = 10 + 2 - 3 > 0
    st = m.row_store
    st.log_q[rows] = -50.0
    st.resid[rows] = -1.0
    assert CS.duty(st, rows, 0).tolist() == [1.0]


@pytest.mark.parametrize('drop, want', [(0.0, 1.0), (0.5, math.exp(-0.5)), (2.0, math.exp(-2.0)),
                                        (math.log(20), 0.05), (30.0, 0.05), (-4.0, 1.0)])
def test_a_row_at_or_above_its_level_rests_by_how_far_q_fell_since_birth(drop, want):
    m, rows = admitted([0], [-8.0], [-2.0])         # r = -8 + 2 - 3 < 0
    st = m.row_store
    st.log_q[rows] = st.log_q_birth[rows] - drop
    st.resid[rows] = 1.0
    assert float(CS.duty(st, rows, 0)) == pytest.approx(want, rel=1e-5)


def test_r_exactly_zero_is_not_the_full_weight_branch():
    m, rows = admitted([0], [-8.0], [-2.0])
    st = m.row_store
    st.log_q[rows] = st.log_q_birth[rows] - 2.0
    st.resid[rows] = 0.0
    assert float(CS.duty(st, rows, 0)) == pytest.approx(math.exp(-2.0), rel=1e-5)


def test_a_rewound_row_has_unknown_r_and_full_weight():
    m, rows = admitted([0], [-8.0], [-2.0])
    st = m.row_store
    st.touch(rows, st.epoch, torch.tensor([-9.0]), torch.tensor([4.0]), 50)
    assert float(CS.duty(st, rows, 50)) < 1.0
    assert st.invalidate_q_after(10) == 1 and bool(torch.isnan(st.resid[rows]).all())
    assert CS.duty(st, rows, 50).tolist() == [1.0]


def test_the_rule_reads_the_columns_a_training_draw_rewrites():
    m, rows = admitted([0], [-8.0], [-2.0])
    st = m.row_store
    assert float(CS.duty(st, rows, 0)) == 1.0       # at birth q has not moved
    st.touch(rows, st.epoch, st.log_q_birth[rows] - 1.0, torch.tensor([1.0]), 9)
    assert float(CS.duty(st, rows, 9)) == pytest.approx(math.exp(-1.0), rel=1e-5)
    assert float(st.log_q_birth[rows]) == -2.0


def test_the_draw_follows_the_rule_within_a_condition():
    m, rows = admitted([0, 0], [-8.0, -8.0], [-2.0, -2.0])
    st = m.row_store
    st.touch(rows[:1], st.epoch, torch.tensor([-30.0]), torch.tensor([1.0]), 9)     # at the floor
    got, _ = st.draw_weighted(CS.ONPOLICY, 20000, CS.duty, 9)
    share = float((got == rows[0]).float().mean())
    assert share == pytest.approx(0.05 / 1.05, abs=0.006)


def test_the_replay_loss_takes_no_importance_weight_under_duty():
    m, rows = admitted([0, 0, 1], [-8.0, -8.0, -8.0], [-2.0, -2.0, -2.0])
    st = m.row_store
    st.touch(rows[:1], st.epoch, torch.tensor([-30.0]), torch.tensor([1.0]), 9)
    m._store_cfg['replay_weight'] = 'duty'
    drawn, w = m._store_replay_rows(32)
    assert w is None and drawn.numel() == 32
    m._store_draw('replay', 1)
    assert m._replay_is_w is None


def test_the_replay_draw_reads_one_row_per_slot_however_many_rows_the_condition_holds(monkeypatch):
    k = 60
    m, rows = admitted([0] * k, [-8.0] * k, [-2.0] * k)         # at birth every weight is 1
    read = []

    def counted(store, r, step):
        read.append(len(r))
        return CS.duty(store, r, step)

    monkeypatch.setitem(CS.REPLAY_WEIGHTS, 'duty', counted)
    m._store_cfg['replay_weight'] = 'duty'
    drawn, w = m._store_replay_rows(32)
    assert w is None and drawn.numel() == 32 and sum(read) == 32


def test_the_accepted_draw_has_the_scan_draws_law():
    m, rows = admitted([0] * 6 + [1] * 2, [-8.0] * 8, [-2.0] * 8)
    st = m.row_store
    drop = torch.tensor([0.0, 0.5, 1.0, 2.0, 30.0, 3.0, 0.0, 30.0])
    st.touch(rows, st.epoch, st.log_q_birth[rows] - drop, torch.ones(8), 9)
    B = 200000
    scan, _ = st.draw_weighted(CS.ONPOLICY, B, CS.duty, 9)
    got = st.draw_accepted(CS.ONPOLICY, B, CS.duty, 9)
    for row in rows.tolist():
        assert float((got == row).float().mean()) == pytest.approx(float((scan == row).float().mean()), abs=0.005)


def test_the_accepted_draw_is_bounded_and_scans_what_stays_open():
    m, rows = admitted([0, 0, 0], [-8.0] * 3, [-2.0] * 3)
    calls = []

    def never(store, r, step):
        calls.append(len(r))
        return torch.zeros(len(r))

    got = m.row_store.draw_accepted(CS.ONPOLICY, 300, never, 0)
    assert len(calls) == CS.ACCEPT_ROUNDS + 1 and calls[-1] == 3        # the last call is the scan of the one condition
    assert set(got.tolist()) == set(rows.tolist())                      # all-zero weights: uniform, as draw_weighted


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -0.1, 1.5])
def test_the_accepted_draw_refuses_a_weight_outside_zero_to_one(bad):
    m, _ = admitted([0, 0], [-8.0] * 2, [-2.0] * 2)
    with pytest.raises(CS.CrystalStoreError, match='weight'):
        m.row_store.draw_accepted(CS.ONPOLICY, 8, lambda s, r, step: torch.full((len(r),), bad), 0)


def test_only_duty_and_visits_skip_the_correction():
    assert CS.UNCORRECTED_RULES == ('duty', 'visits') and set(CS.UNCORRECTED_RULES) <= set(CS.REPLAY_WEIGHTS)


def test_visits_weighs_a_row_by_one_over_one_plus_its_draws_down_to_the_floor():
    m, rows = admitted([0, 0, 0], [-8.0] * 3, [-2.0] * 3)
    st = m.row_store
    st.touch(rows[1:2].repeat(3), st.epoch, torch.full((3,), -2.0), torch.zeros(3), 9)
    st.touch(rows[2:3].repeat(100), st.epoch, torch.full((100,), -2.0), torch.zeros(100), 9)
    w = CS.visits(st, rows, 9)
    assert torch.allclose(w, torch.tensor([1.0, 0.25, CS.VISITS_FLOOR]))


def test_the_visits_draw_follows_the_rule_within_a_condition():
    m, rows = admitted([0, 0], [-8.0, -8.0], [-2.0, -2.0])
    st = m.row_store
    st.touch(rows[1:2].repeat(3), st.epoch, torch.full((3,), -2.0), torch.zeros(3), 9)     # w = 1 and 1/4
    m._store_cfg['replay_weight'] = 'visits'
    drawn, w = m._store_replay_rows(4000)
    assert w is None
    share = float((drawn == rows[1]).double().mean())
    assert share == pytest.approx(0.25 / 1.25, abs=0.025)


def _store_on(cfg):
    cfg = copy.deepcopy(cfg)
    cfg['buffers']['store']['replay_weight'] = 'duty'
    return cfg


@pytest.mark.parametrize('key', ['rotate_frames', 'mirror_frames'])
def test_duty_with_random_frames_is_a_config_error(key):
    cfg = _store_on(yaml.safe_load((CONFIGS / 'unified_nov23' / 'unified.yaml').read_text(encoding='utf-8')))
    assert not [m for sev, m in ci.row_store_problems(ci.canonical_model_section(cfg)) if sev == ci.ERROR]
    cfg['model']['nf'][key] = True
    msgs = [m for sev, m in ci.row_store_problems(ci.canonical_model_section(cfg)) if sev == ci.ERROR]
    assert len(msgs) == 1 and key in msgs[0] and 'duty' in msgs[0]


@pytest.mark.parametrize('key', ['rotate_frames', 'mirror_frames'])
def test_visits_with_random_frames_is_not_refused(key):
    cfg = yaml.safe_load((CONFIGS / 'unified_nov23' / 'unified.yaml').read_text(encoding='utf-8'))
    cfg['buffers']['store']['replay_weight'] = 'visits'
    cfg['model']['nf'][key] = True
    assert not [m for sev, m in ci.row_store_problems(ci.canonical_model_section(cfg)) if sev == ci.ERROR]


def test_uniform_with_random_frames_is_not_refused_by_this_rule():
    cfg = yaml.safe_load((CONFIGS / 'unified_nov23' / 'unified.yaml').read_text(encoding='utf-8'))
    cfg['buffers']['store']['replay_weight'] = 'uniform'
    cfg['model']['nf']['rotate_frames'] = True
    assert not [m for sev, m in ci.row_store_problems(ci.canonical_model_section(cfg)) if 'duty' in m]


def test_the_unified_config_uses_duty_and_mk_dev_stays_uniform():
    uni = yaml.safe_load((CONFIGS / 'unified_nov23' / 'unified.yaml').read_text(encoding='utf-8'))
    mk = yaml.safe_load((CONFIGS / 'mk_dev.yaml').read_text(encoding='utf-8'))
    assert uni['buffers']['store']['replay_weight'] == 'duty'
    assert mk['buffers']['store']['replay_weight'] == 'uniform'
