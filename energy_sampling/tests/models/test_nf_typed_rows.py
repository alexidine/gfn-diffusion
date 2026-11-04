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
    assert out.shape == (64, desc.dim, K_EXTRA) and desc.k == K_EXTRA and torch.isfinite(out).all()
    for i in range(desc.dim):
        s2 = s.clone()
        s2[:, i:] = _rows(desc.dim - i, 64, 100 + i)
        assert torch.equal(desc(s2, radius)[:, i], out[:, i]), (sg, desc.names[i])


@pytest.mark.parametrize('sg', (2, 14, 88, 9, 19))
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
        if isinstance(blk, charts.CellChart):   # lengths and cosines of the cell the chart decodes
            cell, _ = blk.map(y[:, off:off + blk.ndim])
            L, cos = torch.log(cell[:, :3]), torch.cos(cell[:, 3:])
            phys = {'triclinic': [L[:, 2], L[:, 1], L[:, 0], cos[:, 1], cos[:, 2], cos[:, 0]],
                    'monoclinic': [L[:, 2], L[:, 0], L[:, 1], cos[:, 1]]}.get(blk.family)
            if phys is not None:
                want[:, off:off + blk.ndim] = torch.stack(phys, 1)
    want = torch.where(desc.is_length, want - math.log(3.7), want)
    got = out[:, 1:, 2]                     # row i + 1 carries row i's value
    assert torch.allclose(got, want[:, :-1], atol=1e-10), (got - want[:, :-1]).abs().max()
    lo, hi = out[:, :-1, 0], out[:, :-1, 1]   # and each value lies in its own row's physical interval
    assert bool(((got >= lo - 1e-9) & (got <= hi + 1e-9)).all())


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


# -- the unconditional route (one molecule; the prefix-feature testbed) ------------------------------------------------

def _unconditional_typed(sg, scale=0.3):
    desc = FDRowDescriber(sg)
    cfg = dict(gfn_config(False), dim=desc.dim, angular_mask=list(desc.wrap_mask), periodic_centroid_axes=None,
               dead_latent_rows=None)
    flow = {**SMALL, 'conditioner': 'transformer', 'tf_width': 32, 'tf_layers': 2, 'tf_heads': 2, 'tf_fp32': True,
            'typed_rows': True}
    torch.manual_seed(0)
    pol = NFPolicy(cfg, flow, seed=0, fd_space_group=sg).double()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in pol.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return pol.eval()


def test_unconditional_typed_rows_keep_the_charts_order_even_on_twelve_rows():
    """sg 2's chart has 12 rows, which would otherwise resolve block_orders 'auto' to the latent's Z1_ORDER."""
    pol = _unconditional_typed(2)
    assert pol.sampler.flow.orders == [list(range(12))]
    assert pol.sampler.flow.blocks[0].made.order.tolist() == list(range(12))


@pytest.mark.parametrize('sg', (14, 2))
def test_unconditional_typed_draws_round_trip_and_do_not_peek(sg):
    pol = _unconditional_typed(sg)
    n = 6
    mb = SimpleNamespace(radius=torch.full((n,), 3.3))
    pol._context(None, mb, n)
    f = pol.sampler.flow
    with torch.no_grad():
        x, z0 = pol.sampler.sample(n, return_base=True)
        z, _ = pol.sampler.to_base(x)
        assert torch.allclose(z, z0, atol=1e-8), (z - z0).abs().max()
        lp = pol.sampler.log_prob(x)
        assert torch.isfinite(lp).all()
        blk = f.blocks[0]
        made = blk.made
        xs = _rows(f.d, n, 21) * 0.98
        p0 = made(blk._feats(xs), None, pol.sampler.row_extra(xs))
        p0 = p0.view(n, f.d, -1)
        for j in range(f.d):
            x2 = xs.clone()
            x2[:, j:] = _rows(f.d - j, n, 70 + j) * 0.98
            p2 = made(blk._feats(x2), None, pol.sampler.row_extra(x2)).view(n, f.d, -1)
            assert torch.equal(p2[:, j], p0[:, j]), j


def test_unconditional_typed_rows_hold_one_molecule_radius():
    pol = _unconditional_typed(14)
    pol._context(None, SimpleNamespace(radius=torch.full((4,), 3.3)), 4)
    pol._context(None, None, 5)                      # a call without the batch reuses the molecule's radius
    assert pol.sampler.row_radius.shape == (5,)
    with pytest.raises(ValueError, match='differ'):
        pol._context(None, SimpleNamespace(radius=torch.tensor([3.3, 4.0])), 2)


# -- prefix physics features (nf prefix_features) ---------------------------------------------------------------------

@pytest.mark.parametrize('sg', DAY_FAMILIES)
def test_prefix_features_read_only_completed_blocks(sg):
    from models.fd_row_describer import CELL_FEATS, POS_FEATS
    desc = FDRowDescriber(sg, prefix=True)
    assert desc.k == K_EXTRA + len(CELL_FEATS) + len(POS_FEATS)
    s = _rows(desc.dim, 32, sg) * 0.98
    radius = torch.full((32,), 3.1, dtype=torch.float64)
    vol = torch.full((32,), 140.0, dtype=torch.float64)
    out = desc(s, radius, vol)
    assert torch.isfinite(out).all()
    for i in range(desc.dim):
        s2 = s.clone()
        s2[:, i:] = _rows(desc.dim - i, 32, 300 + i) * 0.98
        assert torch.equal(desc(s2, radius, vol)[:, i], out[:, i]), (sg, desc.names[i])
    nc = len(CELL_FEATS)
    cell_done = out[:, :, K_EXTRA + nc - 1]
    pos_done = out[:, :, -1]
    rows = torch.arange(desc.dim)
    assert torch.equal(cell_done, (rows >= desc.cell_end).to(out.dtype).expand(32, -1))
    assert torch.equal(pos_done, (rows >= desc.pos_end).to(out.dtype).expand(32, -1))


def test_prefix_cell_and_image_features_match_a_direct_computation():
    """sg 14 (P2_1/c): log(V_cell / (Z V_mol)) from the cell parameters the chart decodes; the inversion class's
    image distance (one operation, so its soft minimum is exact) against a brute-force minimum over 125 cells."""
    from energies.fd_coordinates import FDCoordinates
    from mxtaltools.crystal_fd.ops import cell_matrices
    from models.fd_row_describer import CELL_FEATS
    desc = FDRowDescriber(14, prefix=True)
    b = 16
    s = _rows(desc.dim, b, 5) * 0.95
    radius = torch.full((b,), 3.0, dtype=torch.float64)
    vol = torch.full((b,), 150.0, dtype=torch.float64)
    out = desc(s, radius, vol)
    d, _ = FDCoordinates(14, 3.0).to_description(s)
    T = cell_matrices(d.cell)
    v_cell = torch.linalg.det(T).abs()
    last = out[:, -1]
    assert torch.allclose(last[:, K_EXTRA], torch.log(v_cell / (4 * 150.0)), atol=1e-9)
    t = d.t[:, 0]
    grid = torch.stack(torch.meshgrid(*(torch.arange(-2, 3, dtype=torch.float64),) * 3, indexing='ij'), -1)
    disp = (-t - t)[:, None] + grid.reshape(-1, 3)[None]
    brute = torch.einsum('bij,bkj->bki', T, disp).norm(dim=-1).amin(-1) / 3.0
    inv = last[:, K_EXTRA + len(CELL_FEATS) + 2]
    assert torch.allclose(inv, brute, atol=1e-9), (inv - brute).abs().max()


def test_p1_has_no_images_and_says_so():
    from models.fd_row_describer import CELL_FEATS
    desc = FDRowDescriber(1, prefix=True)
    out = desc(_rows(desc.dim, 4, 1), torch.full((4,), 3.0), torch.full((4,), 100.0))
    pos = out[:, -1, K_EXTRA + len(CELL_FEATS):]
    assert torch.equal(pos[:, :8], torch.zeros_like(pos[:, :8])) and torch.equal(pos[:, 8], torch.ones(4, dtype=pos.dtype))


def test_prefix_policy_round_trips_and_does_not_peek():
    pol = _typed_policy(14, prefix_features=True)
    f = pol.sampler.flow
    mb = _mols(4, 6)
    mb.mol_volume = torch.rand(4, generator=torch.Generator().manual_seed(1)) * 100 + 80
    cond = torch.randn(4, 4, dtype=torch.float64)
    c = pol._context(cond, mb, 4).double()
    with torch.no_grad():
        x, z0 = pol.sampler.sample(4, c, return_base=True)
        z, _ = pol.sampler.to_base(x, c)
        assert torch.allclose(z, z0, atol=1e-8)
        ctx = f.context_input(c)[0]
        blk = f.blocks[0]
        xs = _rows(f.d, 4, 31) * 0.97
        sh, sp = blk.spline_params(xs, ctx, f.row_extra(xs, c))
        for j in range(f.d):
            x2 = xs.clone()
            x2[:, j:] = _rows(f.d - j, 4, 90 + j) * 0.97
            s2, p2 = blk.spline_params(x2, ctx, f.row_extra(x2, c))
            assert torch.equal(p2[:, j], sp[:, j]), j
    with pytest.raises(ValueError, match='mol_volume'):
        pol._context(cond, _mols(4, 6), 4)
