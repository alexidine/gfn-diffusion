"""nf canonical_frame (models/nf_policy.py NF_PENDING_DEFAULTS): the split encoder computes a proper frame F from the atoms
it is shown (models/spline_flow.py::inertial_frame; 'learned' adds weighted displacement vectors from the invariant
trunk), its point stream reads the atoms in that frame, and the flow models the orientation rows re-expressed for it
(R -> R F; ConditionalLatentFlow.set_row_frame). So:
  * an 'inertial' policy is a frame-free policy (same weights) shown each molecule in its inertial frame, scoring the
    row re-expressed for that frame, plus the map's log-Jacobian;
  * with rotate_frames on, log q no longer depends on the presented frame (inertial and learned), so the frame
    augmentation is exact;
  * a draw re-scores to its own log q (training mode with frames, and eval mode);
  * the learned frame is proper and equivariant, F(Q y) = Q F(y);
  * a non-finite gradient into the learned frame is zeroed, and the setting is refused where it cannot hold.

    python -m pytest -q tests/models/test_nf_canonical_frame.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_typed_rows import _typed_policy, _mols  # noqa: E402
from energies.fd_coordinates import rotate_orientation  # noqa: E402
from models.nf_policy import NF_PENDING_DEFAULTS  # noqa: E402
from models.spline_flow import inertial_frame, quat_to_rotmat, _nan_safe  # noqa: E402

pytestmark = pytest.mark.fast
N = 4


def _inputs(pol, seed=2):
    mb = _mols(N, seed)
    cond = torch.randn(N, 4, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    x = torch.rand(N, pol.dim, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 1.6 - 0.8
    return mb, cond, x


def _quats(n, seed, dtype=torch.float64):
    q = torch.nn.functional.normalize(torch.randn(n, 4, generator=torch.Generator().manual_seed(seed), dtype=dtype), dim=1)
    return torch.where(q[:, :1] < 0, -q, q)


def test_the_setting_defaults_off():
    assert NF_PENDING_DEFAULTS['canonical_frame'] == 'none'


def test_inertial_is_the_frame_free_policy_shown_the_molecule_in_its_inertial_frame():
    none = _typed_policy(2, scale=0.05, atom_tokens='split')
    inert = _typed_policy(2, scale=0.05, atom_tokens='split', canonical_frame='inertial')
    inert.load_state_dict(none.state_dict())
    mb, cond, x = _inputs(none)
    centre, F, _ = inertial_frame(mb.pos.double(), mb.z > 1, mb.batch, N)
    canon = mb.__class__(**vars(mb))
    canon.pos = torch.einsum('nij,ni->nj', F[mb.batch], mb.pos.double() - centre[mb.batch]).to(mb.pos.dtype)
    xc, lj = rotate_orientation(x, F.transpose(1, 2), rows=inert.orientation_rows)
    with torch.no_grad():
        a = inert.log_density(x, cond, mb)
        b = none.log_density(xc, cond, canon) + lj
    assert torch.allclose(a, b, atol=1e-6), float((a - b).abs().max())


@pytest.mark.parametrize('kind', ('inertial', 'learned'))
@pytest.mark.parametrize('mirror', (False, True))
def test_log_q_does_not_depend_on_the_presented_frame(kind, mirror):
    pol = _typed_policy(2, scale=0.05, atom_tokens='split', rotate_frames=True, mirror_frames=mirror,
                        canonical_frame=kind)
    mb, cond, x = _inputs(pol)
    with torch.no_grad():
        ref = pol.log_density(x, cond, mb)
        for k in range(8):
            q = _quats(N, 10 + k)
            if mirror:   # proper frames only: a parity -1 frame is a different (enantiomeric) problem
                q = torch.cat([q, torch.ones(N, 1, dtype=q.dtype)], 1)
            got = pol.log_density(x, cond, mb, frames=q)
            assert torch.allclose(got, ref, atol=1e-6), (k, float((got - ref).abs().max()))


@pytest.mark.parametrize('kind', ('inertial', 'learned'))
@pytest.mark.parametrize('training', (False, True))
def test_a_draw_rescores_to_its_own_log_q(kind, training):
    pol = _typed_policy(2, scale=0.05, atom_tokens='split', rotate_frames=True, canonical_frame=kind)
    mb, cond, _ = _inputs(pol)
    pol.train(training)
    torch.manual_seed(5)
    with torch.no_grad():
        states, log_pf, _, _ = pol.get_traj_fwd(torch.zeros(N, pol.dim, dtype=torch.float64), None, 0.0, cond, mb)
        pol.eval()
        again = pol.log_density(states[:, -1], cond, mb)
    assert torch.allclose(log_pf[:, 0], again, atol=1e-5), float((log_pf[:, 0] - again).abs().max())


def test_the_learned_frame_is_proper_and_equivariant():
    pol = _typed_policy(2, scale=0.05, atom_tokens='split', canonical_frame='learned')
    enc = pol.sampler.flow.atom_encoder
    mb = _mols(N, 2)
    z = torch.zeros(N, 8, dtype=torch.long)
    pos = torch.zeros(N, 8, 3, dtype=torch.float64)
    counts = torch.bincount(mb.batch, minlength=N)
    for i in range(N):
        sel = mb.batch == i
        z[i, :int(counts[i])] = mb.z[sel]
        pos[i, :int(counts[i])] = mb.pos[sel].double()
    with torch.no_grad():
        F = enc(z, pos, counts, return_frame=True)[4]
        Q = quat_to_rotmat(_quats(N, 7))
        FQ = enc(z, pos @ Q.transpose(1, 2), counts, return_frame=True)[4]
    assert torch.allclose(torch.linalg.det(F), torch.ones(N, dtype=F.dtype), atol=1e-9)
    assert torch.allclose(FQ, Q @ F, atol=1e-8), float((FQ - Q @ F).abs().max())
    assert float((F - inertial_frame(mb.pos.double(), mb.z > 1, mb.batch, N)[1]).abs().max()) > 1e-4   # learned moved


def test_a_training_step_through_the_learned_frame_has_a_finite_gradient():
    pol = _typed_policy(2, scale=0.05, atom_tokens='split', rotate_frames=True, canonical_frame='learned').train()
    mb, cond, x = _inputs(pol)
    _, log_pf, _, _ = pol.get_traj_bwd(x, None, cond, mb)
    (-log_pf.sum()).backward()
    head = pol.sampler.flow.atom_encoder.frame_head
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert any(float(p.grad.abs().sum()) > 0 for p in head.parameters())


def test_a_non_finite_gradient_into_the_frame_is_zeroed():
    t = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    torch.sqrt(_nan_safe(t * 1.0)).sum().backward()      # d sqrt / dt = inf at 0
    assert torch.equal(t.grad, torch.zeros(3, dtype=torch.float64))


def test_the_setting_is_refused_where_it_cannot_hold():
    with pytest.raises(ValueError):
        _typed_policy(2, scale=0.0, atom_tokens=True, canonical_frame='inertial')        # not the split encoder
    with pytest.raises(ValueError):
        _typed_policy(2, scale=0.0, atom_tokens='split', canonical_frame='sideways')
    with pytest.raises(NotImplementedError):
        _typed_policy(7, scale=0.0, atom_tokens='split', canonical_frame='inertial')     # a K-cut group
