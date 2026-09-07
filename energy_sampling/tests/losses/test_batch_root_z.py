"""batch_root_z: the level the forward TB loss uses under tb_z_source
'batch_root' -- the batch's own winsorized Huber root, detached, broadcast to
the learned head's shape; the learned scalar only when the root is undefined.

The point of the mode is level-blindness: the loss must not change when the
learned log Z moves, because that motion is the fill's job, not the policy's.
"""
import pytest
import torch

from gflownet_losses import batch_root_z, get_tb_loss, winsorized_z_root

BETA = 10.0


def _batch(n=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    log_pf = torch.randn(n, generator=g, dtype=torch.float64) * 2.0
    log_pb = torch.randn(n, generator=g, dtype=torch.float64) * 2.0 - 3.0
    log_r = torch.randn(n, generator=g, dtype=torch.float64) * 3.0 - 40.0
    return log_pf, log_pb, log_r


def test_root_is_the_winsorized_root_broadcast_and_detached():
    log_pf, log_pb, log_r = _batch()
    log_Z = torch.full((500,), 7.0, dtype=torch.float64, requires_grad=True)
    z = batch_root_z(log_Z, log_pb, log_pf, log_r, BETA)
    root, _, _ = winsorized_z_root(log_pb + log_r - log_pf, BETA)
    assert z.shape == log_Z.shape
    assert torch.allclose(z, torch.full_like(z, root))
    assert not z.requires_grad


def test_loss_is_blind_to_the_learned_level():
    log_pf, log_pb, log_r = _batch()
    losses = []
    for level in (-100.0, 0.0, 100.0):
        log_Z = torch.full((500,), level, dtype=torch.float64)
        z = batch_root_z(log_Z, log_pb, log_pf, log_r, BETA)
        losses.append(float(get_tb_loss(z, log_pb, log_pf, log_r, beta=BETA).mean()))
    assert losses[0] == pytest.approx(losses[1]) and losses[1] == pytest.approx(losses[2])


def test_winsorized_mean_residual_is_zero_at_the_root():
    log_pf, log_pb, log_r = _batch()
    log_Z = torch.zeros(500, dtype=torch.float64)
    z = batch_root_z(log_Z, log_pb, log_pf, log_r, BETA)
    resid = (log_pf + z - log_pb - log_r).clamp(-BETA, BETA)
    assert float(resid.mean()) == pytest.approx(0.0, abs=1e-6)


def test_falls_back_to_the_learned_scalar_when_every_row_is_saturated():
    # a batch with no spread 500 nats from anywhere the root search could land:
    # winsorized_z_root cannot resolve it and the learned level stands in
    n = 50
    log_pf = torch.zeros(n, dtype=torch.float64)
    log_pb = torch.zeros(n, dtype=torch.float64)
    log_r = torch.full((n,), -500.0, dtype=torch.float64)
    log_Z = torch.full((n,), 3.0, dtype=torch.float64)
    try:
        winsorized_z_root(log_pb + log_r - log_pf, BETA)
        pytest.skip("this batch resolves a root; the fallback is not exercised")
    except ValueError:
        pass
    assert batch_root_z(log_Z, log_pb, log_pf, log_r, BETA) is log_Z
