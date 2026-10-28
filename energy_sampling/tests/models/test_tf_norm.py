"""
CPU tests for --tf_norm, the normalisation of the flow's causal-transformer conditioner (energies/ar_transformer.py:
LayerNorm, RMSNorm or none in every pre-norm slot and the final one).

WHAT THIS FILE IS FOR. Each norm must leave the conditioner a drop-in: (1) the modules are the ones asked for, (2) a
row's spline parameters still see only the rows before it in the order (the triangular Jacobian), and (3) the
sampler's cached decode inverts what the full forward pass computes, so q's draws and its log-density belong to one
flow. A norm the cached decode did not route through would break (3) without an error.

    python -m pytest -q tests/models/test_tf_norm.py
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here)):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

import inn_modeller as im  # noqa: E402
from energies.ar_transformer import NORMS, CausalTransformerConditioner  # noqa: E402

WRAP = [False] * 7 + [True, True, False, True, True]
ORDER = [8, 11, 10, 9, 7, 6, 5, 4, 3, 2, 1, 0]
KIND = {'layer': nn.LayerNorm, 'rms': nn.RMSNorm, 'none': nn.Identity}


def _live(m):
    """the zero-initialised head (an identity spline) given weights, so the outputs depend on the input"""
    torch.manual_seed(5)
    with torch.no_grad():
        for prm in m.parameters():
            if float(prm.abs().max()) == 0.0:
                prm.normal_(0, 0.1)
    return m


@pytest.mark.parametrize('norm', NORMS)
def test_the_norm_modules_are_the_ones_asked_for(norm):
    m = CausalTransformerConditioner(ORDER, list(range(12)), 8, width=32, layers=2, heads=4, norm=norm)
    mods = [m.ln] + [layer.ln1 for layer in m.layers] + [layer.ln2 for layer in m.layers]
    assert all(type(x) is KIND[norm] for x in mods)


@pytest.mark.parametrize('norm', NORMS)
def test_the_order_stays_causal(norm):
    m = _live(CausalTransformerConditioner(ORDER, list(range(12)), 8, width=32, layers=2, heads=4, norm=norm)).double()
    torch.manual_seed(3)
    x = torch.rand(4, 12, dtype=torch.float64) * 2 - 1
    with torch.no_grad():
        base = m(x).view(4, 12, -1)
        for j, row in enumerate(ORDER):
            x2 = x.clone()
            x2[:, row] += 0.3
            out = m(x2).view(4, 12, -1)
            for k, other in enumerate(ORDER):
                if k <= j:
                    assert torch.allclose(out[:, other], base[:, other], atol=1e-12), (norm, row, other)
            later = ORDER[j + 1:]
            if later:   # negative control: some later row does see it
                assert any(not torch.allclose(out[:, r], base[:, r], atol=1e-9) for r in later), (norm, row)


@pytest.mark.parametrize('norm', NORMS)
def test_cached_decode_inverts_the_forward_pass(norm):
    torch.manual_seed(0)
    f = _live(im.LatentFlow(12, WRAP, [], 1, 8, (32, 32), 5.0, 0, transformer=(32, 2, 4, norm),
                            block_orders=['.'.join(map(str, ORDER))])).double()
    with torch.no_grad():
        x, z0 = f.sample(256, return_base=True)    # the cached decode, one conditioner step per row
        z, _ = f.to_base(x)                        # the full forward pass
    assert float((z - z0).abs().max()) < 1e-8, norm
