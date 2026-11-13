"""A weights-only warm start may switch the NF frame flags (models/nf_policy.py::FRAME_ONLY_KEYS: rotate_frames,
mirror_frames) on or off: they change what a training step presents (the molecule in a random frame) and no
parameter's shape. Only the condition scaler (c_mu, c_sd) spans the frame columns, and
checkpointing.py::Checkpointer._reframed_state re-lays it. So:
  * every parameter loads unchanged;
  * the condition columns keep their scaling (an appended or shifted scaler would move every condition column);
  * in eval mode (the standard frame) the warm-started policy's density is the checkpoint policy's, row for row.

    python -m pytest -q tests/models/test_nf_frames_warm_start.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_typed_rows import _typed_policy, _mols  # noqa: E402
from checkpointing import Checkpointer  # noqa: E402

pytestmark = pytest.mark.fast
N = 4
BOTH = dict(rotate_frames=True, mirror_frames=True)


def _policy(frames, scaler_seed=None):
    pol = _typed_policy(2, scale=0.05, atom_tokens='split', **frames)
    if scaler_seed is not None:   # a non-trivial condition scaler, so a misplaced entry shows in the density
        f, g = pol.sampler.flow, torch.Generator().manual_seed(scaler_seed)
        with torch.no_grad():
            f.c_mu.copy_(torch.randn(f.c_mu.shape, generator=g, dtype=f.c_mu.dtype))
            f.c_sd.copy_(torch.rand(f.c_sd.shape, generator=g, dtype=f.c_sd.dtype) + 0.5)
    return pol


def _log_q(pol, x, cond, mb):
    with torch.no_grad():
        return pol.sampler.log_prob(pol._wrap_ang(x), pol._context(cond, mb, N))


@pytest.mark.parametrize('old, new', [({}, BOTH), ({}, dict(rotate_frames=True)), (BOTH, {}),
                                      (dict(rotate_frames=True), BOTH)])
def test_a_warm_start_across_frame_flags_keeps_the_standard_frame_density(old, new):
    src, dst = _policy(old, scaler_seed=7), _policy(new)
    with pytest.raises(RuntimeError):   # the scaler's length differs: a plain load refuses
        dst.load_state_dict(src.state_dict())
    dst.load_state_dict(Checkpointer._reframed_state(dst, src.state_dict(), 'test'))   # strict
    assert [k for k, _ in src.named_parameters()] == [k for k, _ in dst.named_parameters()]
    for (k, a), (_, b) in zip(src.named_parameters(), dst.named_parameters()):
        assert torch.equal(a, b), k
    fs, fd = src.sampler.flow, dst.sampler.flow
    assert torch.equal(fs.c_mu[src._lead_columns():], fd.c_mu[dst._lead_columns():])
    assert torch.equal(fs.c_sd[src._lead_columns():], fd.c_sd[dst._lead_columns():])
    mb = _mols(N, 2)
    cond = torch.randn(N, 4, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    x = torch.rand(N, src.dim, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 1.6 - 0.8
    a, b = _log_q(src, x, cond, mb), _log_q(dst, x, cond, mb)
    assert torch.allclose(a, b, atol=1e-10, rtol=0), float((a - b).abs().max())


def test_a_state_already_laid_for_the_model_passes_through():
    pol = _policy(BOTH, scaler_seed=7)
    state = pol.state_dict()
    assert Checkpointer._reframed_state(pol, state, 'test') is state
