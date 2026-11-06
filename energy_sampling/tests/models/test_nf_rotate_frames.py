"""nf rotate_frames (models/nf_policy.py NF_PENDING_DEFAULTS): every training-mode step presents each row's molecule in
a fresh random frame R0; the encoder reads the atoms rotated by R0 and the flow models the FD orientation rows
re-expressed for R0 (energies/fd_coordinates.py::rotate_orientation), while the states and log q stay standard-frame.
So:
  * eval mode presents the standard frame: the backward score is log_density's;
  * a training-mode backward score is the flow's log q of the re-expressed row under the frame's context, plus the
    map's log-Jacobian;
  * a training-mode forward draw returns a standard-frame state whose log q is that same backward score under the
    draw's own frame (sample -> standard -> re-expressed is the identity);
  * a frame in the condition equals the standard frame with the molecule's atoms rotated by it;
  * rotate_frames are refused on the K-cut groups (6-9) or without an atom encoder.

    python -m pytest -q tests/models/test_nf_rotate_frames.py
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_typed_rows import _typed_policy, _mols  # noqa: E402
from models.spline_flow import quat_to_rotmat  # noqa: E402
from mxtaltools.common.geometry_utils import apply_rotation_to_batch  # noqa: E402

pytestmark = pytest.mark.fast
N = 4   # the fixture (test_nf_typed_rows._mols) builds at most four molecules


def _setup(mode='split'):
    pol = _typed_policy(2, scale=0.05, atom_tokens=mode, rotate_frames=True).float()
    mb = _mols(N, 2)
    cond = torch.randn(N, 4, generator=torch.Generator().manual_seed(2))
    x = torch.rand(N, pol.dim, generator=torch.Generator().manual_seed(3)) * 1.6 - 0.8
    return pol, mb, cond, x


def test_eval_mode_scores_the_standard_frame():
    pol, mb, cond, x = _setup()
    pol.eval()
    with torch.no_grad():
        _, log_pf, _, _ = pol.get_traj_bwd(x, None, cond, mb)
        ref = pol.log_density(x, cond, mb)
    assert torch.allclose(log_pf[:, 0], ref, atol=1e-5)


def test_a_training_step_scores_the_reexpressed_row_plus_the_jacobian():
    pol, mb, cond, x = _setup()
    pol.train()
    torch.manual_seed(11)
    with torch.no_grad():
        states, log_pf, _, _ = pol.get_traj_bwd(x, None, cond, mb)
    torch.manual_seed(11)
    q = pol._frames(N)
    with torch.no_grad():
        c = pol._context(cond, mb, N, frames=q)
        xm, lj = pol._present(pol._wrap_ang(x), q)
        manual = pol.sampler.log_prob(xm, c) + lj
        c0 = pol._context(cond, mb, N)
        standard = pol.sampler.log_prob(pol._wrap_ang(x), c0)
    assert torch.allclose(states[:, -1], pol._wrap_ang(x))           # the state stays standard-frame
    assert torch.allclose(log_pf[:, 0], manual, atol=1e-4)
    assert float((log_pf[:, 0] - standard).abs().max()) > 1e-3       # the frame is used


def test_a_training_draw_is_standard_and_its_log_q_is_the_framed_score():
    pol, mb, cond, _ = _setup()
    pol.train()
    torch.manual_seed(12)
    with torch.no_grad():
        states, log_pf, _, _ = pol.get_traj_fwd(torch.zeros(N, pol.dim), None, 0.0, cond, mb)
    torch.manual_seed(12)
    q = pol._frames(N)
    xs = states[:, -1]
    with torch.no_grad():
        c = pol._context(cond, mb, N, frames=q)
        xm, lj = pol._present(xs, q)
        rescored = pol.sampler.log_prob(xm, c) + lj
    assert torch.allclose(log_pf[:, 0], rescored, atol=1e-3), float((log_pf[:, 0] - rescored).abs().max())


def test_a_frame_in_the_condition_is_the_molecule_rotated():
    pol, mb, cond, _ = _setup()
    g = torch.Generator().manual_seed(5)
    q = torch.randn(N, 4, generator=g, dtype=torch.float64)
    q = q / q.norm(dim=1, keepdim=True)
    with torch.no_grad():
        v_q = pol.sampler.flow.context_input(pol._context(cond, mb, N, frames=q))[0][0]
        rot = copy.copy(mb)
        rot.pos = apply_rotation_to_batch(mb.pos.float(), quat_to_rotmat(q.float()), mb.batch).to(mb.pos.dtype)
        v_rot = pol.sampler.flow.context_input(pol._context(cond, rot, N))[0][0]
        v_std = pol.sampler.flow.context_input(pol._context(cond, mb, N))[0][0]
    assert torch.allclose(v_q, v_rot, atol=1e-4), float((v_q - v_rot).abs().max())
    assert float((v_q - v_std).abs().max()) > 1e-3


def test_rotate_frames_are_refused_where_they_do_not_hold():
    with pytest.raises(NotImplementedError, match='K-cut'):
        _typed_policy(7, scale=0.0, atom_tokens="split", rotate_frames=True)   # a K-cut group
    with pytest.raises(ValueError, match='rotate_frames'):
        _typed_policy(2, scale=0.0, atom_tokens='scalars', rotate_frames=True)


def test_log_density_in_a_frame_is_the_framed_backward_score():
    pol, mb, cond, x = _setup()
    pol.train()
    torch.manual_seed(13)
    with torch.no_grad():
        _, log_pf, _, _ = pol.get_traj_bwd(x, None, cond, mb)
    torch.manual_seed(13)
    q = pol._frames(N)
    pol.eval()
    framed = pol.log_density(x, cond, mb, frames=q)
    assert torch.allclose(framed, log_pf[:, 0], atol=1e-4)
    assert torch.allclose(pol.log_density(x, cond, mb, frames=torch.tensor([[1.0, 0, 0, 0]]).expand(N, 4)),
                          pol.log_density(x, cond, mb), atol=1e-4)   # the identity frame is the standard one
