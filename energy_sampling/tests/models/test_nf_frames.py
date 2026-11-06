"""nf frames (models/nf_policy.py NF_PENDING_DEFAULTS): the condition carries the frame each row's molecule is presented
in, mol_batch.presented_frame, and the encoder reads the atoms rotated into it. A batch without the attribute reads
the standard frame (every draw but the MLE draw under train.py frames, and every readout). So:
  * no presented_frame gives the context of an explicit identity frame;
  * a row presented in frame q gets the context of the same row in the standard frame with its atoms rotated by q
    (and not the standard frame's own context: a negative control);
  * frames are refused where no encoder reads atoms.

    python -m pytest -q tests/models/test_nf_frames.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_typed_rows import _typed_policy, _mols  # noqa: E402
from models.spline_flow import quat_to_rotmat  # noqa: E402
from mxtaltools.common.geometry_utils import apply_rotation_to_batch  # noqa: E402

pytestmark = pytest.mark.fast


def _ctx(pol, mb, cond, n):
    with torch.no_grad():
        c = pol._context(cond, mb, n).double()
        out, _ = pol.sampler.flow.context_input(c)
    vec = out[0] if isinstance(out, tuple) else out
    return c, vec


@pytest.mark.parametrize('mode', ('split', 'split_pooled'))
def test_a_presented_frame_rotates_the_atoms_the_encoder_reads(mode):
    pol = _typed_policy(14, scale=0.05, atom_tokens=mode, frames=True)
    n = 4
    mb = _mols(n, 2)
    cond = torch.randn(n, 4, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    g = torch.Generator().manual_seed(5)
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    q = q / q.norm(dim=1, keepdim=True)
    c_std, v_std = _ctx(pol, mb, cond, n)
    assert torch.equal(c_std[:, 1:5], torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64).expand(n, 4))
    mb.presented_frame = q.float()
    c_q, v_q = _ctx(pol, mb, cond, n)
    assert torch.allclose(c_q[:, 1:5], q, atol=1e-6)
    del mb.presented_frame
    import copy
    rot = copy.copy(mb)   # the fixture is a namespace of tensors: a shallow copy with its own pos
    rot.pos = apply_rotation_to_batch(mb.pos.float(), quat_to_rotmat(q.float()), mb.batch).to(mb.pos.dtype)
    _, v_rot = _ctx(pol, rot, cond, n)
    assert torch.allclose(v_q, v_rot, atol=1e-4), float((v_q - v_rot).abs().max())
    assert float((v_q - v_std).abs().max()) > 1e-3                  # the frame is read


def test_frames_need_an_encoder_that_reads_atoms():
    with pytest.raises(ValueError, match='frames'):
        _typed_policy(14, scale=0.0, atom_tokens='scalars', frames=True)
