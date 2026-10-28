"""The cross-condition matrix (eval/xcond.py) under a policy with an exact density: eval/xcond.py::
cross_condition_delta_exact scores each cell once with NFPolicy.log_density and returns the matrix and replicates
cross_condition_delta's k-rollout estimate reduces to under log P_B = 0; train.py::Modeller.log_xcond_metrics takes
that path when the policy has log_density and ignores xcond_eval.k (null accepted), and refuses k null under the
SDE policy.

    python -m pytest -q tests/models/test_nf_xcond.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.xcond import cross_condition_delta, cross_condition_delta_exact, xcond_metrics  # noqa: E402
from models.gfn import GFN  # noqa: E402
from test_nf_policy import flow_policy, gfn_config  # noqa: E402
from train import Modeller  # noqa: E402

pytestmark = pytest.mark.fast
C, PER, COND_DIM = 3, 2, 4


class _Batch:
    """A stand-in graph batch: rows carrying a condition vector and a condition_id (vector conditioning reads no
    graph, so subsample_new_batch only has to carry the rows)."""

    def __init__(self, conditions, condition_id):
        self.conditions, self.condition_id = conditions, condition_id
        self.num_graphs = int(conditions.shape[0])

    def subsample_new_batch(self, idx):
        idx = idx.cpu()
        return _Batch(self.conditions[idx], self.condition_id[idx])

    def to(self, device):
        return self


def _cols():
    torch.manual_seed(1)
    return _Batch(torch.randn(C, COND_DIM), torch.arange(C))


def test_exact_matrix_is_the_k_rollout_estimate_at_log_pb_zero():
    pol = flow_policy(True, dead=(9,))
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():   # a fresh flow reads no condition; perturbed, its columns differ
        for p in pol.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g))
    cols = _cols()
    tile = torch.arange(C).repeat_interleave(PER)
    states, *_ = pol.get_traj_fwd(torch.zeros(C * PER, 12), None, 0, cols.conditions[tile], None)
    x = states[:, -1]
    ref = cross_condition_delta(pol, cols, cols.conditions, x, None, k=2, chunk_rows=5)
    got = cross_condition_delta_exact(pol, cols, cols.conditions, x, chunk_rows=4)
    assert got[0].shape == (C * PER, C) and got[0].dtype == torch.float64
    assert float((got[0] - got[0].mean(1, keepdim=True)).abs().max()) > 1.0   # the columns are not one number
    # equal up to float32 rounding: the flow's log q of one row moves by ~2e-5 relative with the batch it is scored in
    # (chunk_rows 1 vs 18 on this matrix), and the two paths batch the rows differently
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-4), (a - b).abs().max()
    assert torch.equal(got[1], got[0]) and torch.equal(got[2], got[0])
    m_ref, m_got = xcond_metrics(*ref), xcond_metrics(*got)
    assert set(m_got) == set(m_ref) and m_got['xcond/noise_sd'] == 0.0
    for key in ('xcond/interaction_frac', 'xcond/diag_lift_nats', 'xcond/accuracy', 'xcond/logpost_true'):
        assert abs(m_got[key] - m_ref[key]) < 1e-4, key


def _modeller(policy, k, step_ind=0, every_n_evals=1):
    xc = SimpleNamespace(enabled=True, every_n_evals=every_n_evals, conditions=C, per_condition=PER, k=k,
                         chunk_rows=7)
    args = SimpleNamespace(xcond_eval=xc, eval_period=250, vector_conditioning=True,
                           conditional_worst_quantile=0.9)
    return SimpleNamespace(args=args, step_ind=step_ind, device='cpu', gfn_model=policy,
                           _batch_latents=lambda b: torch.zeros(b.num_graphs, 12))


@pytest.mark.parametrize('step_ind, every_n_evals, runs', [
    (2500, 10, True), (250, 10, False), (50, 10, False), (2500, 0, False), (2500, None, False)])
def test_the_matrix_runs_on_every_n_evals_eval_periods(step_ind, every_n_evals, runs):
    """xcond_eval.every_n_evals counts eval periods on the step grid (train.py::eval_cadence_fires);
    an off-cadence evaluation returns no xcond/ key at all."""
    torch.manual_seed(2)
    out = Modeller.log_xcond_metrics(_modeller(flow_policy(True), None, step_ind, every_n_evals),
                                     lambda b: None, _cols())
    assert (out.get('xcond/skipped') == 0.0) if runs else out == {}


def test_k_null_is_accepted_under_the_flow_policy():
    pol = flow_policy(True)
    torch.manual_seed(2)
    out = Modeller.log_xcond_metrics(_modeller(pol, None), lambda b: None, _cols())
    assert out['xcond/skipped'] == 0.0 and out['xcond/noise_sd'] == 0.0
    assert out['xcond/n_conditions'] == C and out['xcond/n_per_condition'] == PER
    # the published set is xcond_metrics' own, plus the skip flag
    probe = torch.randn(C * PER, C, dtype=torch.float64)
    assert set(out) == set(xcond_metrics(probe, probe, probe)) | {'xcond/skipped'}


def test_k_null_is_refused_under_the_sde_policy():
    torch.manual_seed(0)
    sde = GFN(**gfn_config(True))
    assert not hasattr(sde, 'log_density')
    with pytest.raises(ValueError, match='xcond_eval.k is null'):
        Modeller.log_xcond_metrics(_modeller(sde, None), lambda b: None, _cols())
