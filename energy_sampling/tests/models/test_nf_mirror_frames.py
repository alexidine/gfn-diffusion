"""nf mirror_frames (models/nf_policy.py NF_PENDING_DEFAULTS): a training-mode frame is improper with probability 1/2.
The molecule is reflected, X -> diag(-1, 1, 1) X, before its frame rotation, the encoder reads the reflected atoms
(the parity rides in the condition after the quaternion), and the flow models the row's mirror image
(energies/fd_coordinates.py::mirror_state, then the rotation); states and log q stay standard-frame. So:
  * a parity -1 frame in the condition is the standard frame with the molecule's atoms reflected through x -> -x;
  * a training-mode backward score is the flow's log q of the mirrored (and rotated) row plus the maps'
    log-Jacobians, and a forward draw maps back to a standard-frame state whose log q is that score;
  * the identity frame of either parity column value +1 is the standard frame;
  * a group without a measured mirror map is refused.

    python -m pytest -q tests/models/test_nf_mirror_frames.py
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_typed_rows import _typed_policy, _mols  # noqa: E402
from energies.fd_coordinates import mirror_state  # noqa: E402

pytestmark = pytest.mark.fast
N = 4   # the fixture (test_nf_typed_rows._mols) builds at most four molecules


def _setup(sg=2, rotate=False):
    pol = _typed_policy(sg, scale=0.05, atom_tokens='split', mirror_frames=True, rotate_frames=rotate).float()
    mb = _mols(N, 2)
    cond = torch.randn(N, 4, generator=torch.Generator().manual_seed(2))
    x = torch.rand(N, pol.dim, generator=torch.Generator().manual_seed(3)) * 1.6 - 0.8
    return pol, mb, cond, x


def _mirror_frames(rotate=False, seed=5):
    g = torch.Generator().manual_seed(seed)
    if rotate:
        q = torch.randn(N, 4, generator=g, dtype=torch.float64)
        q = q / q.norm(dim=1, keepdim=True)
    else:
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).expand(N, 4)
    parity = torch.tensor([-1.0, 1.0, -1.0, -1.0], dtype=torch.float64)[:, None]
    return torch.cat([q, parity], 1)


def test_a_parity_minus_one_frame_is_the_molecule_reflected():
    pol, mb, cond, _ = _setup()
    q = _mirror_frames()
    q_all = q.clone()
    q_all[:, 4] = -1.0
    with torch.no_grad():
        v_q = pol.sampler.flow.context_input(pol._context(cond, mb, N, frames=q_all))[0][0]
        ref = copy.copy(mb)
        ref.pos = mb.pos * torch.tensor([-1.0, 1.0, 1.0], dtype=mb.pos.dtype)
        v_ref = pol.sampler.flow.context_input(pol._context(cond, ref, N))[0][0]
        v_std = pol.sampler.flow.context_input(pol._context(cond, mb, N))[0][0]
    assert torch.allclose(v_q, v_ref, atol=1e-4), float((v_q - v_ref).abs().max())
    assert float((v_q - v_std).abs().max()) > 1e-3   # the encoder sees the reflection


@pytest.mark.parametrize('sg', (2, 14))
def test_a_mirrored_training_step_scores_the_mirror_image_plus_the_jacobian(sg):
    pol, mb, cond, x = _setup(sg)
    q = _mirror_frames()
    pol.train()
    pol._frames = lambda n: q
    with torch.no_grad():
        states, log_pf, _, _ = pol.get_traj_bwd(x, None, cond, mb)
        c = pol._context(cond, mb, N, frames=q)
        xw = pol._wrap_ang(x)
        m = q[:, 4] < 0
        xm, lj = xw.clone(), torch.zeros(N)
        part, lj_m = mirror_state(xw[m].double(), torch.full((int(m.sum()),), sg), pol.mirror_maps,
                                  pol.orientation_rows)
        xm[m], lj[m] = part.float(), lj_m.float()
        manual = pol.sampler.log_prob(pol._wrap_ang(xm), c) + lj
    assert torch.allclose(states[:, -1], xw)              # the state stays standard-frame
    assert torch.allclose(log_pf[:, 0], manual, atol=1e-4), float((log_pf[:, 0] - manual).abs().max())


@pytest.mark.parametrize('rotate', (False, True))
def test_a_mirrored_training_draw_is_standard_and_its_log_q_is_the_framed_score(rotate):
    pol, mb, cond, _ = _setup(2, rotate)
    q = _mirror_frames(rotate)
    pol.train()
    pol._frames = lambda n: q
    with torch.no_grad():
        states, log_pf, _, _ = pol.get_traj_fwd(torch.zeros(N, pol.dim), None, 0.0, cond, mb)
        xs = states[:, -1]
        c = pol._context(cond, mb, N, frames=q)
        xm, lj = pol._present(xs, q)
        rescored = pol.sampler.log_prob(xm, c) + lj
        back, lj_back = pol._present(xm, q, inverse=True)
    assert torch.allclose(log_pf[:, 0], rescored, atol=1e-3), float((log_pf[:, 0] - rescored).abs().max())
    assert float(pol._wrap_ang(back - xs).abs().max()) < 1e-4   # present then unpresent is the identity
    assert torch.allclose(lj + lj_back, torch.zeros(N), atol=1e-4)


def test_the_identity_frame_is_the_standard_frame():
    pol, mb, cond, x = _setup(2, rotate=True)
    ident = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]]).expand(N, 5)
    assert torch.allclose(pol.log_density(x, cond, mb, frames=ident), pol.log_density(x, cond, mb), atol=1e-4)
    with pytest.raises(ValueError, match='frames of 4 columns'):
        pol.log_density(x, cond, mb, frames=ident[:, :4])


def test_mirror_frames_are_refused_without_a_map():
    with pytest.raises(NotImplementedError, match='no mirror map'):
        _typed_policy(4, scale=0.0, atom_tokens='split', mirror_frames=True)
    with pytest.raises(ValueError, match='mirror_frames'):
        _typed_policy(2, scale=0.0, atom_tokens='scalars', mirror_frames=True)


def test_a_mixed_group_state_mirrors_each_row_by_its_own_group():
    """On an FDSlotLayout state (P-1 and P2_1/c rows in one batch) a mirrored row is its own group's mirror image,
    computed on that group's chart rows and packed back into its slots."""
    from energies.fd_coordinates import frame_rows, mirror_rows
    from test_nf_mixed_groups import SGS, _mols as _mixed_mols, _policy as _mixed_policy, _states
    pol, layout = _mixed_policy(mirror_frames=True)
    pol = pol.float()
    x = _states(layout, SGS, 7).float()
    mb = _mixed_mols(SGS)
    cond = torch.randn(len(SGS), 4, generator=torch.Generator().manual_seed(0))
    q = torch.tensor([[1.0, 0, 0, 0, -1.0]], dtype=torch.float64).expand(len(SGS), 5)
    with torch.no_grad():
        pol._context(cond, mb, len(SGS), frames=q)
        got, _ = pol._present(pol._wrap_ang(x), q)
    for i, sg in enumerate(SGS.tolist()):
        slots = torch.as_tensor(layout.slots[sg])
        chart = pol._wrap_ang(x)[i:i + 1, slots].double()
        want, _ = mirror_state(chart, torch.tensor([sg]), mirror_rows([sg]), frame_rows([sg]))
        d = (got[i, slots].double() - want[0]).abs()
        d = torch.minimum(d, 2 - d)   # circle rows: s = -1 and 1 are one point
        assert float(d.max()) < 1e-5, (sg, float(d.max()))
