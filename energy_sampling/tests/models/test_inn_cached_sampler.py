"""
CPU tests for the cached sampler of the exact-density flow (inn_modeller.invert_block_cached,
energies/ar_transformer.py::CausalTransformerConditioner.decode).

WHAT THIS FILE IS FOR. LatentFlow.sample inverts a transformer block one row per step through a key/value cache instead
of running the conditioner d times. A wrong cache (a token attending to the wrong slots, a feature column of the wrong
row, a shift not undone) still returns a finite x whose log q is finite, so nothing downstream fails. The check here is
equality with the d-pass inverse (prior_flow.invert_block) on the same base points, in double precision where the two
should agree to rounding, on flows whose parameters are far from the identity, for unbounded and bounded rows. The
NEGATIVE CONTROL perturbs one row's slot embedding after the cached draw and requires the d-pass inverse to move,
so equality cannot come from a conditioner that ignores its input. Equality is tight on 99.9% of rows; a row
whose spline is nearly flat at its draw amplifies rounding and is held to a looser bound.

    python test_inn_cached_sampler.py
"""
import os
import sys

import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for p in (_here, os.path.dirname(_here)):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

import inn_modeller as im  # noqa: E402
from energies.prior_flow import invert_block  # noqa: E402

WRAP = [False] * 7 + [True, True, False, True, True]
ORDER = ['8.11.10.9.7.6.5.4.3.2.1.0']


def _flow(bounded):
    torch.manual_seed(0)
    f = im.LatentFlow(12, WRAP, [], 1, 32, (256,) * 2, 5.0, 0, transformer=(64, 2, 4), block_orders=ORDER,
                      bounded_rows=[i for i in range(12) if not WRAP[i]] if bounded else [])
    f.set_bf16(False)
    f.double()
    m = f.flow.blocks[0].made
    with torch.no_grad():   # move the conditioner well away from the identity it starts at
        m.head.weight.normal_(0, 0.3)
        m.head.bias.normal_(0, 1.0)
        m.dim_emb.normal_(0, 0.5)
        m.start.normal_(0, 0.5)
    for blk in f.flow.blocks:
        blk.min_height = torch.full_like(blk.bound, 1e-7)
        blk.tied = 2.0
    return f


def _base(f, n=2000):
    g = torch.Generator().manual_seed(1)
    u = 2 * torch.rand(n, f.flow.d, generator=g, dtype=torch.float64) - 1
    return torch.where(f.flow.unit_base, u, torch.randn(n, f.flow.d, generator=g, dtype=torch.float64))


def test_cached_equals_dpass_unbounded_and_bounded():
    for bounded in (False, True):
        f = _flow(bounded)
        blk = f.flow.blocks[0]
        z = _base(f)
        with torch.no_grad():
            a = invert_block(blk, z, f.flow.d)
            b = im.invert_block_cached(blk, z)
            back, _ = blk(b)
        diff = (a - b).abs().flatten()
        # rounding, amplified only where a spline is nearly flat: tight on 99.9% of rows, loose on the rest
        assert float(torch.quantile(diff, 0.999)) < 1e-10, bounded
        assert float(diff.max()) < 1e-5, bounded
        assert float((back - z).abs().max()) < 1e-9, bounded


def test_sample_uses_the_cache_and_inverts():
    f = _flow(True)
    torch.manual_seed(2)
    with torch.no_grad():
        x, z0 = f.sample(500, return_base=True)
        zb, _ = f.to_base(x)
    assert float((zb - z0).abs().max()) < 1e-9


def test_negative_control_conditioner_input_matters():
    f = _flow(True)
    blk = f.flow.blocks[0]
    z = _base(f, 500)
    with torch.no_grad():
        b = im.invert_block_cached(blk, z)
        # a random direction: a constant added to every feature would vanish in the LayerNorms
        blk.made.slot_emb[5].add_(torch.randn(blk.made.slot_emb.shape[1], generator=torch.Generator().manual_seed(3),
                                              dtype=blk.made.slot_emb.dtype))
        a = invert_block(blk, z, f.flow.d)
    assert float((a - b).abs().max()) > 1e-3


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
