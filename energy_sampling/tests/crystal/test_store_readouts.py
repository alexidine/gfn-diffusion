"""Store buffer readouts (train.py Modeller._tally_store_draw, ._store_draw_readouts, ._store_population_readouts), on a
real CrystalStore. CPU, no energy computed.

    python -m pytest -q tests/crystal/test_store_readouts.py
"""
import math

import numpy as np
import pytest
import torch

from energy_sampling import crystal_store as CS
from energy_sampling.tests.crystal import store_stub as S

pytestmark = pytest.mark.fast


def with_rollouts(cid, log_r=-1.0, log_pf=-2.0, step=5):
    m = S.build()
    m.step_ind = step
    fwd, batch = S.rollout(cid, log_r=[log_r] * len(cid), log_pf=[log_pf] * len(cid))
    return m, m._store_admit(fwd, batch, CS.ORIGIN_ROLLOUT)


def kick_children(m, cid, step=5, energy=1.0):
    st = m.row_store
    anchors = torch.nonzero(st.kind[:st.high_water] == CS.ANCHOR).flatten()
    parent = torch.stack([anchors[st.cid[anchors] == c][0] for c in cid])
    n = len(cid)
    return st.add(torch.full((n, S.DIM), 0.1), torch.tensor(cid), CS.CHILD, CS.ORIGIN_KICK, torch.full((n,), -1.0),
                  torch.full((n,), float(energy)), step, parent=parent)


def test_the_bwd_draw_tally_is_the_share_of_kicked_children_it_drew():
    m = S.build()
    kick_children(m, [0, 0, 1])
    rows = torch.tensor([int(r) for r in torch.nonzero(m.row_store.kind == CS.CHILD).flatten()])
    m._tally_store_draw('bwd', rows)
    kicked = int((m.row_store.origin[rows] == CS.ORIGIN_KICK).sum())
    assert kicked == 3
    assert m._store_draw_readouts()['store/bwd_draw_kicked_frac'] == pytest.approx(3 / rows.numel())


def test_the_replay_tally_reads_age_and_prior_draws_then_clears():
    m, rows = with_rollouts([0, 1], step=5)
    st = m.row_store
    st.touch(rows[:1], st.epoch, torch.tensor([-2.0]), torch.tensor([0.0]), 6)    # one earlier draw of row 0
    m.step_ind = 25
    m._tally_store_draw('replay', torch.tensor([int(rows[0]), int(rows[0]), int(rows[1])]))
    out = m._store_draw_readouts()
    assert out['store/replay_draw_age_mean'] == pytest.approx(20.0)
    assert out['store/replay_draw_reuse_mean'] == pytest.approx(2 / 3)
    assert math.isnan(out['store/bwd_draw_kicked_frac'])
    again = m._store_draw_readouts()
    assert all(math.isnan(v) for v in again.values())


def test_the_training_draw_feeds_the_tally_and_tiles_nothing_twice():
    m, rows = with_rollouts([0, 1, 2, 3])
    m._store_draw('replay', 2)
    m._store_draw('bwd', 2)
    t = m._store_draw_tally
    assert t['replay_n'] == m.batch_size and t['bwd_n'] == m.batch_size


def test_population_readouts_cover_reuse_and_rows_per_condition():
    m, rows = with_rollouts([0, 0, 1])
    st = m.row_store
    st.touch(rows[:1].repeat(3), st.epoch, torch.full((3,), -2.0), torch.zeros(3), 6)
    out = m._store_population_readouts()
    assert out['store/onpolicy_draws_mean'] == pytest.approx(1.0)
    assert out['store/onpolicy_drawn_frac'] == pytest.approx(1 / 3)
    # drawable on-policy rows of the four seeded conditions: 2, 1, 0, 0; replay draws from the two that hold one
    assert out['store/onpolicy_cond_frac'] == pytest.approx(0.5)
    assert out['store/onpolicy_per_cond_median'] == pytest.approx(1.5)
    assert 'store/replay_w_mean' not in out             # the uniform rule has no weights


def test_population_readouts_report_the_duty_weights_of_drawable_on_policy_rows():
    m, rows = with_rollouts([0, 0, 1, 1], log_r=-8.0, log_pf=-2.0)
    st = m.row_store
    m._store_cfg['replay_weight'] = 'duty'
    # row 0 asks q up (resid < 0): w = 1; row 1 rests, q fell 30 nats: floor; rows 2 and 3 rest, q unchanged: w = 1
    st.touch(rows[:2], st.epoch, torch.tensor([-2.0, -32.0]), torch.tensor([-1.0, 1.0]), 9)
    st.touch(rows[2:], st.epoch, st.log_q_birth[rows[2:]].clone(), torch.tensor([1.0, 1.0]), 9)
    w = CS.duty(st, rows, 9)
    out = m._store_population_readouts()
    assert out['store/replay_w_mean'] == pytest.approx(float(w.double().mean()))
    assert out['store/replay_w_one_frac'] == pytest.approx(0.75)
    assert out['store/replay_w_floor_frac'] == pytest.approx(0.25)


def test_undrawn_on_policy_rows_are_left_out_of_the_population():
    m, rows = with_rollouts([0])
    st = m.row_store
    fwd, batch = S.rollout([1], log_r=[-1.0], log_pf=[float('nan')])     # a non-finite log q is stored UNDRAWN
    m._store_admit(fwd, batch, CS.ORIGIN_ROLLOUT)
    out = m._store_population_readouts()
    assert out['store/onpolicy_drawn_frac'] == pytest.approx(0.0)
    assert out['store/onpolicy_cond_frac'] == pytest.approx(0.25)
    assert out['store/onpolicy_per_cond_median'] == pytest.approx(1.0)
    assert int(st.n_drawable(CS.ONPOLICY)) == 1


def test_a_store_with_no_on_policy_rows_reports_no_reuse():
    m = S.build()
    out = m._store_population_readouts()
    assert 'store/onpolicy_draws_mean' not in out and 'store/onpolicy_per_cond_median' not in out
    assert out['store/onpolicy_cond_frac'] == 0.0


def test_a_row_born_after_the_current_step_reads_age_zero():
    m, rows = with_rollouts([0], step=50)
    m.step_ind = 30                                     # a rewind to step 30 keeps the row born at 50
    m._tally_store_draw('replay', rows)
    assert m._store_draw_readouts()['store/replay_draw_age_mean'] == 0.0


def test_reuse_reads_the_draws_column_before_the_training_draw_writes_it():
    m, rows = with_rollouts([0])                        # one drawable on-policy row: every replay slot draws it
    R, B = 2, m.batch_size
    *_, inds, _, _, _, _ = m._store_draw('replay', R)
    assert m._store_draw_readouts()['store/replay_draw_reuse_mean'] == 0.0
    n = inds[0].numel()
    m._store_touch(inds, {'log_pf': torch.full((n,), -2.0), 'resid': torch.zeros(n)})
    m._store_draw('replay', R)
    assert m._store_draw_readouts()['store/replay_draw_reuse_mean'] == pytest.approx(R * B)


def test_rounding_below_the_seed_minimum_is_not_an_improvement():
    m = S.build()
    st, seed = m.row_store, m._store_seed_best
    c = int(torch.nonzero(seed.abs() > 1).flatten()[0])
    st.best_energy[c] = seed[c] - 1e-7 * seed[c].abs()
    assert m._store_population_readouts()['store/conditions_improved'] == 0
    st.best_energy[c] = seed[c] - 0.5
    assert m._store_population_readouts()['store/conditions_improved'] == 1


def test_an_improvement_survives_a_resume_into_a_freshly_seeded_store():
    m = S.build()
    kick_children(m, [1], energy=float(m._store_seed_best[1]) - 1.0)     # below condition 1's seed minimum
    assert m._store_population_readouts()['store/conditions_improved'] == 1
    state = m.row_store.state_dict(compact=True)
    fresh = S.build()                                   # init_row_store: seed, take the seed minimum, then restore
    fresh.row_store.apply_state(state)
    assert fresh._store_population_readouts()['store/conditions_improved'] == 1


@pytest.mark.parametrize('grow, want_b', [(False, 125), (True, 4000)])
def test_headroom_takes_the_batch_ceiling_when_the_sizer_can_grow_the_batch(grow, want_b):
    from pathlib import Path
    import yaml
    from utils import dict2namespace
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / 'configs' / 'unified_nov23' / 'unified.yaml')
                         .read_text(encoding='utf-8'))
    cfg.update(grow_batch_size=grow, max_batch_size=4000, batch_size=125, eval_num_samples=64)
    m = S.build()
    m.args = dict2namespace(cfg)
    m.batch_size = 125
    reps = 2                                            # the unified tb stage's fwd repeats
    assert m._store_headroom() == 2 * want_b * reps
