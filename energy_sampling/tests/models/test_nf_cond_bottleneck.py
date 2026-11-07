"""nf cond_bottleneck / cond_noise (models/nf_policy.py NF_PENDING_DEFAULTS): each pooled molecule vector is projected
to cond_bottleneck dims and bounded by tanh, and N(0, cond_noise^2) is added per row in training mode only. So:
  * the context and the level head read bottleneck-wide vectors;
  * eval mode is deterministic; training mode draws fresh noise on every call;
  * absent settings build the model they did before (no bottleneck layers);
  * noise without a bottleneck, and either without an atom encoder, are refused.

    python -m pytest -q tests/models/test_nf_cond_bottleneck.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_typed_rows import _typed_policy, _mols  # noqa: E402

pytestmark = pytest.mark.fast
N = 4


def _ctx(pol, seed=0):
    mb = _mols(N, 2)
    cond = torch.randn(N, 4, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    torch.manual_seed(seed)
    with torch.no_grad():
        out, s = pol.sampler.flow.context_input(pol._context(cond, mb, N).double())
    return (out[0] if isinstance(out, tuple) else out), s


@pytest.mark.parametrize('mode', ('split', 'split_pooled', True))
def test_the_bottleneck_squeezes_the_pooled_vectors_and_noises_them_in_training_only(mode):
    pol = _typed_policy(14, scale=0.05, atom_tokens=mode, cond_bottleneck=16, cond_noise=0.5)
    f = pol.sampler.flow
    assert f.bn_s.out_features == 16 and (mode is True or f.bn_p.out_features == 16)
    pol.eval()
    v1, s1 = _ctx(pol, 0)
    v2, s2 = _ctx(pol, 1)
    assert torch.equal(v1, v2) and torch.equal(s1, s2)                       # no noise in eval
    assert float(s1[:, :16].abs().max()) <= 1.0                              # tanh-bounded
    pol.train()
    t1, ts1 = _ctx(pol, 0)
    t2, ts2 = _ctx(pol, 1)
    assert float((ts1[:, :16] - ts2[:, :16]).abs().max()) > 0.05             # fresh noise per call
    assert float((t1 - t2).abs().max()) > 0


def test_absent_settings_build_no_bottleneck():
    pol = _typed_policy(14, scale=0.0, atom_tokens='split')
    assert not hasattr(pol.sampler.flow, 'bn_s') and pol.sampler.flow.bottleneck == 0


def test_noise_needs_a_bottleneck_and_both_need_an_encoder():
    with pytest.raises(ValueError, match='needs cond_bottleneck'):
        _typed_policy(14, scale=0.0, atom_tokens='split', cond_noise=0.5)
    with pytest.raises(ValueError, match='cond_bottleneck'):
        _typed_policy(14, scale=0.0, atom_tokens='scalars', cond_bottleneck=8)
