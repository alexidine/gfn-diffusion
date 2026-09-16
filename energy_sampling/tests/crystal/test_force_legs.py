"""The per-leg terminal force on the crystal route (replay buffer force_legs).

The claim the lambda anneal needs: a row's force legs recorded at ONE lambda,
re-mixed by gflownet_losses.remix_force_legs at ANOTHER, equal the force a
fresh reward call would give at that other lambda -- so a stored row's path
gradient follows every lambda move without an energy call. Checked against
autograd of log_reward on an energy built at the target lambda, at lambda = 0
(flow only), an interior lambda, and 1 (physical only), with the bounding term
live (rows placed outside the box). And on a flow-free energy the legs are
(0, phys, bound) and remix to the old single-column force.

    python -m pytest -q tests/crystal/test_force_legs.py
"""
import importlib.util
import math
import os
import sys

import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling.buffer import N_FORCE_LEGS  # noqa: E402
from energy_sampling.gflownet_losses import remix_force_legs, terminal_force_legs  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'lambda_mix_helpers', os.path.join(os.path.dirname(__file__), 'test_lambda_mix.py'))
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)

N, T = 10, 2.5


@pytest.fixture(scope='module')
def flow_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp('legs') / 'flow.pt')
    _h.fitted_flow(path)
    return path


def _rows(seed=3):
    x = _h.in_box(N, seed=seed)
    x[:3, 0] = 1.3      # outside the box on a free axis: bounding is live on these rows
    x[3, 1] = -1.2
    return x


def _direct_force(ef, x):
    """autograd of the actual log_reward at ef's lambda -- the reference."""
    xl = x.detach().clone().requires_grad_(True)
    log_r = ef.log_reward(xl, _h.mol_batch(N), torch.full((N,), math.log10(T)), False, keep_grads=True)
    return torch.autograd.grad(log_r.sum(), xl)[0].detach()


def test_legs_recorded_at_one_lambda_remix_to_the_force_at_another(flow_path):
    x = _rows()
    ef_rec = _h.energy_fn(flow_path, lam=0.3, temperature=T)
    legs, stats = terminal_force_legs(torch.full((N,), math.log10(T)), ef_rec, _h.mol_batch(N), x)
    assert legs.shape == (N, N_FORCE_LEGS, 12)
    assert float(stats['force/nonfinite_frac']) == 0.0
    assert legs[:, 2][:4].abs().sum() > 0 and legs[:, 2][4:].abs().sum() == 0, \
        'bounding leg must be live outside the box and zero inside'
    for lam in (0.0, 0.3, 0.7, 1.0):
        ef = _h.energy_fn(flow_path, lam=lam, temperature=T)
        ref = _direct_force(ef, x)
        got = remix_force_legs(legs, lam)
        assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), \
            (lam, (got - ref).abs().max().item())
    # the flow leg is a log-density gradient: temperature-free
    ef_hot = _h.energy_fn(flow_path, lam=0.3, temperature=6.9)
    legs_hot, _ = terminal_force_legs(torch.full((N,), math.log10(6.9)), ef_hot, _h.mol_batch(N), x)
    assert torch.allclose(legs_hot[:, 0], legs[:, 0], atol=1e-4)
    assert not torch.allclose(legs_hot[:, 1], legs[:, 1], atol=1e-2), 'the physical leg must temper'


def test_a_flow_free_energy_stores_zero_flow_and_remixes_to_the_old_force():
    x = _rows(seed=4)
    ef = _h.energy_fn(None, lam=1.0, temperature=T)
    legs, _ = terminal_force_legs(torch.full((N,), math.log10(T)), ef, _h.mol_batch(N), x)
    assert torch.equal(legs[:, 0], torch.zeros(N, 12))
    ref = _direct_force(ef, x)
    assert torch.allclose(remix_force_legs(legs, 1.0), ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(legs[:, 1] + legs[:, 2], ref, atol=1e-4, rtol=1e-4)


def test_capture_is_dropped_on_exit_and_does_not_nest(flow_path):
    ef = _h.energy_fn(flow_path, lam=0.5, temperature=T)
    assert ef._reward_leg_capture is None
    with ef.capture_reward_legs() as got:
        with pytest.raises(RuntimeError, match='nest'):
            with ef.capture_reward_legs():
                pass
        ef.log_reward(_rows(), _h.mol_batch(N), torch.full((N,), math.log10(T)))
    assert len(got) == 1 and got[0].shape == (N, N_FORCE_LEGS)
    assert ef._reward_leg_capture is None, 'a graph parked on the energy function outlives the call'
    # a call outside the block captures nothing and costs nothing
    ef.log_reward(_rows(), _h.mol_batch(N), torch.full((N,), math.log10(T)))
    assert ef._reward_leg_capture is None
