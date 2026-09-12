"""
The forward loss feeds condition_log_z's physical (anchor-currency) minimum.

The forward rollout is the one update_best_energy site outside train.py
(gflownet_losses.update_condition_best_energy, reached from every forward loss).
It used to pass the mixture alone, so on a run carrying a prior_flow the armed
tracker raised at the first forward step. It now reads E_anchor off the scored
batch through the reader the trainer hands it (Modeller._anchor_energy_phys).

    pytest tests/losses/test_fwd_best_energy_phys.py
"""
import math
import types
from types import SimpleNamespace

import pytest
import torch

from buffer import ConditionLogZTracker
from gflownet_losses import update_condition_best_energy
from train import Modeller


def reader(flow):
    m = SimpleNamespace(energy_function=SimpleNamespace(
        prior_flow=object() if flow else None, bounding_coeff=3.0))
    return types.MethodType(Modeller._anchor_energy_phys, m)


def armed(flow):
    t = ConditionLogZTracker(library_size=4)
    t.requires_phys_energy = flow
    return t


CID = torch.tensor([0, 1])
LOG_R = torch.tensor([-2.0, -4.0])
LOG_T = torch.full((2,), math.log10(2.0))       # T = 2: mixture = 4, 8
BATCH = SimpleNamespace(physical_energy=torch.tensor([10.0, 20.0]),
                        bounding_energy=torch.tensor([0.0, 1.0]))


def test_a_flow_run_writes_e_anchor_into_the_physical_minimum():
    t = armed(True)
    update_condition_best_energy(t, CID, LOG_R, LOG_T, crystal_batch=BATCH,
                                 anchor_energy_fn=reader(True))
    assert not t.phys_is_alias
    assert t.best_energy_phys[:2].tolist() == [10.0, 23.0]
    assert t.best_energy[:2].tolist() == [4.0, 8.0]


def test_a_flow_run_without_the_scored_batch_raises():
    with pytest.raises(ValueError, match='scored batch'):
        update_condition_best_energy(armed(True), CID, LOG_R, LOG_T, crystal_batch=None,
                                     anchor_energy_fn=reader(True))


def test_a_flow_run_without_the_reader_is_refused_by_the_tracker():
    with pytest.raises(ValueError, match='energy_phys'):
        update_condition_best_energy(armed(True), CID, LOG_R, LOG_T, crystal_batch=BATCH)


def test_a_lambda_free_run_keeps_the_alias():
    t = armed(False)
    update_condition_best_energy(t, CID, LOG_R, LOG_T, crystal_batch=BATCH,
                                 anchor_energy_fn=reader(False))
    assert t.phys_is_alias
    assert torch.equal(t.best_energy, t.best_energy_phys)


def test_the_forward_loss_forwards_the_batch_and_the_reader():
    import ast
    import inspect
    import gflownet_losses
    tree = ast.parse(inspect.getsource(gflownet_losses.get_gfn_forward_loss))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, 'id', None) == 'update_condition_best_energy']
    assert len(calls) == 1
    assert {'crystal_batch', 'anchor_energy_fn'} <= {k.arg for k in calls[0].keywords}
