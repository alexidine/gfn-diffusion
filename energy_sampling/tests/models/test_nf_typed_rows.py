"""Typed FD rows of the AR flow policy (models/fd_row_describer.py; NFPolicy nf typed_rows; design
docs/design/conditional_ar_architecture.md §5.3 and §5.6-5.7).

Contracts:
- the describer's features for row i read rows < i only, for every chart family the Day groups use (triclinic,
  monoclinic incl. the K-cut sg 9, orthorhombic, tetragonal with a sheared centroid basis, rhombohedral), checked by
  changing rows >= i and asking for bit-identical features ("no peeking"; needs no gradient, so a non-differentiable
  feature cannot hide a leak);
- the physical values it reports are the chart's own: cell and orientation coordinates as charts.from_unit decodes
  them (lengths over the molecule radius), centroid coordinates as the polytope chart maps them (t, not u);
- the whole conditioner with random (non-zero) weights: row j's spline parameters do not move when rows >= j move;
- the policy draws and scores one density (round trip through the cached decode), and eval/nf_fit.py's per-row terms
  sum to it;
- the per-row features are read (they move log q), and typed rows are refused where they would not be causal."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.fd_row_describer import FD_TYPES, K_EXTRA, FDRowDescriber  # noqa: E402
from models.nf_policy import NFPolicy  # noqa: E402
from test_nf_policy import SMALL, gfn_config  # noqa: E402

pytestmark = pytest.mark.fast
DAY_FAMILIES = (1, 2, 9, 14, 19, 61, 88, 148)
TF = dict(conditioner='transformer', tf_width=32, tf_layers=2, tf_heads=2, tf_fp32=True, atom_node_dim=16,
          atom_convs=2, atom_tokens='split', typed_rows=True)


def _rows(d, b, seed):
    return torch.rand(b, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * 2 - 1


@pytest.mark.parametrize('sg', DAY_FAMILIES)
def test_describer_row_i_reads_only_the_rows_before_it(sg):
    desc = FDRowDescriber(sg)
    s = _rows(desc.dim, 64, sg)
    radius = torch.rand(64, dtype=torch.float64, generator=torch.Generator().manual_seed(7)) * 3 + 2
    out = desc(s, radius)
    assert out.shape == (64, desc.dim, K_EXTRA) and torch.isfinite(out).all()
    for i in range(desc.dim):
        s2 = s.clone()
        s2[:, i:] = _rows(desc.dim - i, 64, 100 + i)
        assert torch.equal(desc(s2, radius)[:, i], out[:, i]), (sg, desc.names[i])


@pytest.mark.parametrize('sg', (2, 14, 88, 9))
def test_describer_values_are_the_charts_physical_coordinates(sg):
    from energies.fd_coordinates import FDCoordinates
    from mxtaltools.crystal_fd import charts
    desc = FDRowDescriber(sg)
    b = 32
    s = _rows(desc.dim, b, sg) * 0.99
    radius = torch.full((b,), 3.7, dtype=torch.float64)
    out = desc(s, radius)
    chart = FDCoordinates(sg, radius).chart
    y, _ = charts.from_unit(chart, (s + 1) / 2)
    want = y.clone()
    for off, blk in zip(chart.offsets, chart.blocks):
        if isinstance(blk, charts.PolytopeARChart) and blk.ndim:
            t, _ = blk.map(y[:, off:off + blk.ndim])
            want[:, off:off + blk.ndim] = blk.ar_coordinates(t)
    want = torch.where(desc.is_length, want - math.log(3.7), want)
    got = out[:, 1:, 2]                     # row i + 1 carries row i's value
    assert torch.allclose(got, want[:, :-1], atol=1e-10), (got - want[:, :-1]).abs().max()


def _mols(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    counts = torch.tensor([3, 5, 4, 6][:n])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return SimpleNamespace(z=torch.randint(1, 9, (int(counts.sum()),), generator=g),
                           pos=torch.randn(int(counts.sum()), 3, generator=g) * 1.5,
                           batch=torch.repeat_interleave(torch.arange(n), counts), ptr=ptr, num_graphs=n,
                           radius=torch.rand(n, generator=g) * 2 + 2.5)


def _typed_policy(sg, scale=0.3, **flow):
    desc = FDRowDescriber(sg)
    cfg = dict(gfn_config(True, cond_dim=4), dim=desc.dim, angular_mask=list(desc.wrap_mask),
               periodic_centroid_axes=None, dead_latent_rows=None, conditions_type='molecule')
    torch.manual_seed(0)
    pol = NFPolicy(cfg, {**SMALL, **TF, **flow}, seed=0, fd_space_group=sg).double()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():   # random, non-zero weights everywhere (zero-initialised heads would hide a leak)
        for p in pol.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return pol.eval()


def _inputs(pol, n=4, seed=0):
    mb = _mols(n, seed)
    cond = torch.randn(n, 4, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    c = pol._context(cond, mb, n).double()
    return mb, cond, c


@pytest.mark.parametrize('sg', (14, 9, 88))
def test_typed_conditioner_row_j_does_not_see_rows_from_j_on(sg):
    pol = _typed_policy(sg)
    f = pol.sampler.flow
    blk = f.blocks[0]
    _, _, c = _inputs(pol)
    with torch.no_grad():
        ctx = f.context_input(c)[0]
        x = _rows(f.d, 4, 11)
        x = torch.where(f.is_wrap, x, x * 0.98)
        shift, sp = blk.spline_params(x, ctx, f.row_extra(x, c))
        for j in range(f.d):
            x2 = x.clone()
            x2[:, j:] = _rows(f.d - j, 4, 50 + j) * 0.98
            s2, p2 = blk.spline_params(x2, ctx, f.row_extra(x2, c))
            assert torch.equal(s2[:, j], shift[:, j]) and torch.equal(p2[:, j], sp[:, j]), (sg, j)


@pytest.mark.parametrize('mode', ['split', 'split_pooled'])
def test_typed_draws_round_trip_and_the_row_terms_sum_to_log_q(mode):
    from eval.nf_fit import block_context, rows_given_ctx
    pol = _typed_policy(14, scale=0.05, atom_tokens=mode)
    mb, cond, c = _inputs(pol, n=4, seed=2)
    with torch.no_grad():
        x, z0 = pol.sampler.sample(4, c, return_base=True)
        z, _ = pol.sampler.to_base(x, c)
        assert torch.allclose(z, z0, atol=1e-8), (z - z0).abs().max()
        exact = pol.sampler.log_prob(x, c)
        ctx, extra = block_context(pol, cond, mb, 4, x)
        terms = rows_given_ctx(pol.sampler, x, ctx, extra)
        assert torch.allclose(terms.sum(-1), exact, atol=1e-8)


def test_the_per_row_features_move_log_q_and_the_type_tables_are_shared():
    pol = _typed_policy(14, scale=0.1)
    _, _, c = _inputs(pol, n=4, seed=4)
    made = pol.sampler.flow.blocks[0].made
    assert made.type_slot_emb.shape[0] == len(FD_TYPES) and made.row_types.tolist() == FDRowDescriber(14).types
    x = _rows(pol.sampler.flow.d, 4, 9) * 0.9
    with torch.no_grad():
        a = pol.sampler.log_prob(x, c)
        made.extra.weight.zero_()
        made.extra.bias.zero_()
        b = pol.sampler.log_prob(x, c)
    assert not torch.allclose(a, b, atol=1e-6)


def test_typed_rows_are_refused_off_the_fd_chart_and_off_the_conditional_route():
    desc = FDRowDescriber(14)
    cfg = dict(gfn_config(True, cond_dim=4), dim=desc.dim, angular_mask=list(desc.wrap_mask),
               periodic_centroid_axes=None, dead_latent_rows=None, conditions_type='molecule')
    with pytest.raises(ValueError, match='crystal_coordinates fd'):
        NFPolicy(cfg, {**SMALL, **TF}, seed=0)
    with pytest.raises(NotImplementedError, match='one block'):
        NFPolicy(cfg, {**SMALL, **TF, 'n_blocks': 2}, seed=0, fd_space_group=14)
    with pytest.raises(ValueError, match="chart has 10 rows"):
        NFPolicy(dict(cfg, dim=12, angular_mask=[False] * 12), {**SMALL, **TF}, seed=0, fd_space_group=14)
