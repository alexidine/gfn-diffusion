"""bootstrap_z:offset -- the stage-entry level seed for a TB stage entered from a weights-only load.

The action shifts log Z(c) by ONE constant, the same for every condition, so that over one forward rollout the
winsorized-Huber root (knee fwd_loss_coeffs.beta) of log w - log Z(c) is zero (train.py::Modeller.bootstrap_z_offset, through
models/nf_policy.py::NFPolicy.shift_level). These pin the shift's exactness, the estimator and its clipping of a deep tail, the refusal on a
policy without shift_level, and that the shift reaches every policy copy. The run it was built for, and the live
check that the level lands on the forward level at step 0, are in the method's docstring and the change's report.
"""
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import train  # noqa: E402
from train import Modeller  # noqa: E402
from models.nf_policy import NFPolicy  # noqa: E402
from gflownet_losses import winsorized_z_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_policy import flow_policy  # noqa: E402

pytestmark = pytest.mark.fast


# -- NFPolicy.shift_level ---------------------------------------------------------------------------------------------

def test_shift_level_moves_every_condition_by_exactly_delta():
    pol = flow_policy(conditional=True)
    with torch.no_grad():   # give the head a per-condition shape, so the test is not of a constant
        torch.nn.init.normal_(pol.sampler.flow.level.out[-1].weight)
    c = torch.randn(32, 4)
    before = pol.sampler.level(c).detach().clone()
    pol.shift_level(7.25)
    after = pol.sampler.level(c).detach()
    assert torch.allclose(after - before, torch.full_like(before, 7.25), atol=1e-5)


def test_shift_level_on_the_unconditional_scalar():
    pol = flow_policy(conditional=False)
    before = float(pol.flow_model())
    pol.shift_level(-3.0)
    assert float(pol.flow_model()) == pytest.approx(before - 3.0)


# -- Modeller.bootstrap_z_offset ----------------------------------------------------------------------------------------

class _Policy:
    def __init__(self):
        self.shifts = []

    def shift_level(self, delta):
        self.shifts.append(delta)


def _modeller(monkeypatch, logw, head, policies=None, beta=10.0, n_default=None):
    monkeypatch.setattr(train, 'get_discretizer', lambda T: None)
    monkeypatch.setattr(train, 'trajectory_T', lambda args: 10)
    logw = torch.as_tensor(logw, dtype=torch.float64)
    head = torch.as_tensor(head, dtype=torch.float64)
    calls = []

    def fwd_eval_sampling(model, disc, override_num_samples=None, side_effects=True):
        calls.append((override_num_samples, side_effects))
        n = logw.numel()
        # log w = log_r + sum log_pb - sum log_pf: put it all in log_r
        return {'log_r': logw.clone(), 'log_pbs': torch.zeros(n, 3), 'log_pfs': torch.zeros(n, 3),
                'log_flow': head.reshape(-1, 1).clone()}, None

    pols = policies if policies is not None else (_Policy(),)
    m = SimpleNamespace(args=SimpleNamespace(eval_num_samples=n_default, fwd_loss_coeffs=SimpleNamespace(beta=beta)),
                        batch_size=64, gfn_model=pols[0],
                        fwd_eval_sampling=fwd_eval_sampling, _policies=lambda: pols)
    m.bootstrap_z_offset = MethodType(Modeller.bootstrap_z_offset, m)
    return m, pols, calls


def test_shift_lands_the_head_on_the_forward_level(monkeypatch):
    """The head opened at 0 below a forward level of 21 with a spread well inside the knee: the shift is the gap,
    the Huber root of the residuals, which there is the mean to within its noise."""
    g = torch.Generator().manual_seed(0)
    logw = 21.0 + 3.0 * torch.randn(1000, generator=g, dtype=torch.float64)
    m, (p,), calls = _modeller(monkeypatch, logw, torch.zeros(1000))
    m.bootstrap_z_offset(1000)
    root = float(winsorized_z_root(logw, 10.0)[0])
    assert p.shifts == [pytest.approx(root)]
    assert abs(root - float(logw.mean())) < 0.05
    assert calls == [(1000, False)]   # a Z measurement: no tracker, anchor or buffer writes


def test_the_residual_is_against_the_head_so_a_partly_fitted_head_moves_by_the_remainder(monkeypatch):
    logw = torch.tensor([20.0, 22.0, 24.0, 26.0], dtype=torch.float64)
    head = torch.tensor([19.0, 21.0, 23.0, 25.0], dtype=torch.float64)   # already within 1 nat, per row
    m, (p,), _ = _modeller(monkeypatch, logw, head)
    m.bootstrap_z_offset(4)
    assert p.shifts == [pytest.approx(1.0)]


def test_a_deep_tail_counts_only_to_the_knee(monkeypatch):
    """1% of rows 10,000 nats deep would drag the mean ~100 nats; under the knee each pulls only beta, so the root
    solves 990 (z - 21) + 10 * beta = 0: where the emp_z sidecar would bring the head to rest."""
    logw = torch.full((1000,), 21.0, dtype=torch.float64)
    logw[:10] = -1.0e4
    m, (p,), _ = _modeller(monkeypatch, logw, torch.zeros(1000), beta=10.0)
    m.bootstrap_z_offset(1000)
    assert p.shifts == [pytest.approx(21.0 - 100.0 / 990.0, abs=1e-6)]


def test_the_knee_follows_fwd_loss_coeffs_beta(monkeypatch):
    logw = torch.full((1000,), 21.0, dtype=torch.float64)
    logw[:10] = -1.0e4
    m, (p,), _ = _modeller(monkeypatch, logw, torch.zeros(1000), beta=40.0)
    m.bootstrap_z_offset(1000)
    assert p.shifts == [pytest.approx(21.0 - 400.0 / 990.0, abs=1e-6)]


def test_non_finite_rows_are_excluded(monkeypatch):
    logw = torch.tensor([10.0, float('nan'), 12.0, float('-inf')], dtype=torch.float64)
    m, (p,), _ = _modeller(monkeypatch, logw, torch.zeros(4))
    m.bootstrap_z_offset(4)
    assert p.shifts == [pytest.approx(11.0)]


def test_fewer_than_two_finite_rows_leave_the_level_alone(monkeypatch):
    logw = torch.tensor([10.0, float('nan')], dtype=torch.float64)
    m, (p,), _ = _modeller(monkeypatch, logw, torch.zeros(2))
    m.bootstrap_z_offset(2)
    assert p.shifts == []


def test_every_policy_copy_is_shifted(monkeypatch):
    pols = (_Policy(), _Policy())
    m, _, _ = _modeller(monkeypatch, torch.full((8,), 5.0), torch.zeros(8), policies=pols)
    m.bootstrap_z_offset(8)
    assert [q.shifts for q in pols] == [[pytest.approx(5.0)], [pytest.approx(5.0)]]


def test_draw_size_defaults_to_eval_num_samples_then_the_batch(monkeypatch):
    m, _, calls = _modeller(monkeypatch, torch.ones(4), torch.zeros(4), n_default=2500)
    m.bootstrap_z_offset()
    m.args.eval_num_samples = None
    m.bootstrap_z_offset()
    assert [c[0] for c in calls] == [2500, 64]


def test_refuses_a_policy_without_shift_level(monkeypatch):
    m, _, calls = _modeller(monkeypatch, torch.ones(4), torch.zeros(4), policies=(SimpleNamespace(),))
    with pytest.raises(NotImplementedError, match='shift_level'):
        m.bootstrap_z_offset(4)
    assert calls == []   # refused before spending the rollout


def test_nf_policy_offers_shift_level():
    assert callable(getattr(NFPolicy, 'shift_level', None))
