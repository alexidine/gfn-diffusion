"""nf atom_tokens 'scalars': the molecule route with no atom encoder, the bottom rungs of a conditioning ladder. The
context and the level head read only the molecule scalars (mol_scalars) and the condition columns; prefix rows still
read the molecule's radius. A draw round-trips and log q is finite; two molecules with the same scalars and radius
get the same density whatever their atoms.

    python -m pytest -q tests/models/test_nf_scalars_route.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_typed_rows import _typed_policy, _mols  # noqa: E402

pytestmark = pytest.mark.fast


@pytest.mark.parametrize('scalars', (True, False))
def test_the_scalars_route_has_no_encoder_and_round_trips(scalars):
    pol = _typed_policy(14, scale=0.05, atom_tokens='scalars', mol_scalars=scalars, row_tokens='prefix')
    f = pol.sampler.flow
    assert not hasattr(f, 'atom_encoder') and not f.cross
    mb = _mols(4, 2)
    mb.mol_volume = torch.rand(mb.num_graphs, generator=torch.Generator().manual_seed(1)) * 100 + 80
    cond = torch.randn(4, 4, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    c = pol._context(cond, mb, 4).double()
    with torch.no_grad():
        x, z0 = pol.sampler.sample(4, c, return_base=True)
        z, _ = pol.sampler.to_base(x, c)
        lq = pol.sampler.log_prob(x, c)
    assert torch.allclose(z, z0, atol=1e-8)
    assert torch.isfinite(lq).all()


def test_the_scalars_route_is_smaller_than_the_atom_route():
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(_typed_policy(14, scale=0.0, atom_tokens='scalars', mol_scalars=True)) < n(
        _typed_policy(14, scale=0.0, atom_tokens='split_pooled'))
