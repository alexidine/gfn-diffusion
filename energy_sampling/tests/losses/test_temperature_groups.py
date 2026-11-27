"""Multi-temperature TB in the losses (gflownet_losses.py): the (condition, temperature) group key
(condition_temperature_groups) that emp_z, the condition-grouped VarGrad, z_level and the group stats take in a stage
setting `temperatures`, and the tracker feed mask (`rows` of update_and_lookup_condition_log_z and
update_condition_best_energy). CPU, no model.

    python -m pytest -q tests/losses/test_temperature_groups.py
"""
import copy
import math

import pytest
import torch

from buffer import ConditionLogZTracker
from gflownet_losses import (condition_group_stats, condition_grouped_empirical_z, condition_temperature_groups,
                             update_and_lookup_condition_log_z, update_condition_best_energy, z_level_loss)

pytestmark = pytest.mark.fast
LT_RUN, LT_HOT = math.log10(2.5), math.log10(25.0)


def _rows(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, generator=g), torch.randn(n, generator=g) * 3 - 10, torch.randn(n, generator=g) * 2)


def test_two_tiles_of_one_condition_at_two_temperatures_are_two_emp_z_groups():
    cid = torch.tensor([3, 3, 3, 3])
    log_T = torch.tensor([LT_RUN, LT_RUN, LT_HOT, LT_HOT], dtype=torch.float32)
    log_pb = torch.zeros(4)
    log_pf = torch.zeros(4)
    log_r = torch.tensor([-1.0, -3.0, -10.0, -20.0])
    group = condition_temperature_groups(cid, log_T)
    assert group.tolist() == [0, 0, 1, 1]
    target, mask, _ = condition_grouped_empirical_z(log_pb, log_pf, log_r, group)
    assert target.tolist() == [-2.0, -2.0, -15.0, -15.0] and mask.all()
    pooled, _, _ = condition_grouped_empirical_z(log_pb, log_pf, log_r, cid)
    assert pooled.tolist() == [-8.5] * 4                      # by condition alone the temperatures pool


def test_groups_order_by_condition_then_temperature():
    cid = torch.tensor([5, 1, 5, 1, 5])
    log_T = torch.tensor([LT_HOT, LT_RUN, LT_RUN, LT_HOT, LT_HOT], dtype=torch.float32)
    assert condition_temperature_groups(cid, log_T).tolist() == [3, 0, 2, 1, 3]


def test_one_temperature_groups_exactly_as_condition_id():
    log_pb, log_pf, log_r = _rows(40)
    cid = torch.randint(0, 7, (40,), generator=torch.Generator().manual_seed(4)) * 3 + 2
    log_T = torch.full((40,), LT_RUN, dtype=torch.float32)
    group = condition_temperature_groups(cid, log_T)
    assert torch.equal(group, torch.unique(cid, return_inverse=True)[1])
    for lme in (False, True):
        a, b = condition_grouped_empirical_z(log_pb, log_pf, log_r, cid, lme=lme), \
            condition_grouped_empirical_z(log_pb, log_pf, log_r, group, lme=lme)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    z = torch.randn(40)
    assert torch.equal(z_level_loss(log_pf, log_pb, log_r, z, cid), z_level_loss(log_pf, log_pb, log_r, z, group))
    sa, sb = condition_group_stats(cid), condition_group_stats(group)
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _tracker():
    t = ConditionLogZTracker(library_size=6, min_visits=1)
    return t


def _state(t):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in vars(t).items()}


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        x, y = a[k], b[k]
        if torch.is_tensor(x):
            assert torch.equal(torch.nan_to_num(x.double(), 1e30), torch.nan_to_num(y.double(), 1e30)), k
        else:
            assert repr(x) == repr(y), k


def _feed(t, cid, log_r, log_pb, log_pf, log_z, rows):
    update_and_lookup_condition_log_z(t, cid, log_r, log_pb, log_pf, 'cpu', step=7, log_Z_learned=log_z,
                                      mode_level_stream='fwd', rows=rows)
    update_condition_best_energy(t, cid, log_r, torch.full_like(log_r, LT_RUN), rows=rows)


def test_rows_off_the_run_temperature_leave_the_tracker_as_the_on_rows_alone_would():
    log_pb, log_pf, log_r = _rows(12)
    log_z = torch.randn(12)
    cid = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
    on = torch.tensor([True] * 6 + [False] * 6)
    masked, alone = _tracker(), _tracker()
    _feed(masked, cid, log_r, log_pb, log_pf, log_z, on)
    _feed(alone, cid[on], log_r[on], log_pb[on], log_pf[on], log_z[on], None)
    _same(_state(masked), _state(alone))
    target, mask = update_and_lookup_condition_log_z(masked, cid, log_r, log_pb, log_pf, 'cpu', step=8,
                                                     do_update=False)
    assert target.shape == (12,) and mask.shape == (12,)       # the lookup is of every row


def test_an_all_false_mask_writes_nothing():
    log_pb, log_pf, log_r = _rows(6)
    t = _tracker()
    _feed(t, torch.tensor([0, 1, 2, 0, 1, 2]), log_r, log_pb, log_pf, torch.randn(6), None)
    before = _state(t)
    _feed(t, torch.tensor([3, 4, 5, 3, 4, 5]), log_r - 50, log_pb, log_pf, torch.randn(6), torch.zeros(6, dtype=torch.bool))
    _same(before, _state(t))


def test_no_mask_feeds_every_row_as_before():
    log_pb, log_pf, log_r = _rows(6)
    cid = torch.tensor([0, 1, 2, 0, 1, 2])
    a, b = _tracker(), _tracker()
    _feed(a, cid, log_r, log_pb, log_pf, torch.zeros(6), None)
    _feed(b, cid, log_r, log_pb, log_pf, torch.zeros(6), torch.ones(6, dtype=torch.bool))
    _same(_state(a), _state(b))
