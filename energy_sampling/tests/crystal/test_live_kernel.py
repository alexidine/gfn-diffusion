"""noised_prior.live_kernel through noised_prior.settings and Modeller._live_kernel_draw, on a Modeller built without
__init__ and a hand-written kernel table. What each test would catch:

  * an unknown kernel, or a non-positive / non-finite tau or width cap, passing settings;
  * a draw leaving [-1, 1] on a bounded row, or a wrapped row not folded onto its period;
  * a dead row moved;
  * a row drawn with another row's kernel (the table is indexed by the loader's row indices).

CPU, no model or energy.

    python -m pytest -q tests/crystal/test_live_kernel.py
"""
import math
from types import SimpleNamespace

import pytest
import torch

from energy_sampling import noised_prior
from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast


def _args(**block):
    return SimpleNamespace(noised_prior=SimpleNamespace(**block))


@pytest.mark.parametrize('block', [dict(live_kernel='gauss'), dict(live_tau=0.0), dict(live_tau=math.nan),
                                   dict(live_width_cap=-1.0), dict(live_width_cap=True), dict(live_tau='1')])
def test_malformed_live_settings_are_refused(block):
    with pytest.raises(ValueError, match='live_'):
        noised_prior.settings(_args(**block))


def test_absent_live_keys_are_off():
    s = noised_prior.settings(_args())
    assert s['live_kernel'] is None and s['live_tau'] == 1.0


def _modeller(L, wrap, live):
    m = Modeller.__new__(Modeller)
    m.gfn_model = SimpleNamespace(live_idx=torch.tensor(live), ang_mask=torch.tensor(wrap))
    m._live_L = L
    return m


def test_draws_stay_in_the_box_wrap_folds_and_dead_rows_hold():
    torch.manual_seed(0)
    # rows 0..2 live (row 1 wrapped), row 3 dead; widths far past the box so every fold is exercised
    L = 5.0 * torch.eye(3).expand(2, 3, 3).clone()
    m = _modeller(L, wrap=[False, True, False, False], live=[0, 1, 2])
    x = torch.tensor([[0.9, 0.9, -0.9, 0.42]]).repeat(4000, 1)
    y = m._live_kernel_draw(x, torch.zeros(4000, dtype=torch.long))
    assert bool((y[:, :3].abs() <= 1.0).all())
    assert torch.equal(y[:, 3], x[:, 3])
    # the reflected bounded rows and the folded wrapped row each spread over the whole box
    for j in range(3):
        assert float(y[:, j].min()) < -0.9 and float(y[:, j].max()) > 0.9


def test_each_row_is_drawn_with_its_own_kernel():
    torch.manual_seed(0)
    L = torch.stack([torch.zeros(2, 2), 1e-2 * torch.eye(2)])
    m = _modeller(L, wrap=[False, False], live=[0, 1])
    x = torch.zeros(1000, 2)
    inds = torch.arange(1000) % 2
    y = m._live_kernel_draw(x, inds)
    assert torch.equal(y[inds == 0], x[inds == 0])
    assert 0.005 < float(y[inds == 1].std()) < 0.02


# ---------------------------------------------------------------- live_kernel 'basin' (docs/design/basin_set.md)

#: one real MIPCAS eLJ minimum (sg 2, Z' = 1, 2026-10-29 search): identity-description cell parameters, handedness +1
P_MIPCAS = [3.42406, 8.815858, 8.946602, 1.665358, 1.578231, 1.610735, 0.250891, 0.32719, 0.261217,
            1.072443, 1.292957, 1.133738]
RADIUS = 3.369637966156006


@pytest.mark.parametrize('block', [dict(live_caps=[1.0, 0.2, 0.1]), dict(live_caps=[1.0, 0.2, 0.1, -1.0]),
                                   dict(live_redraws=-1), dict(live_redraws=1.5), dict(live_bank=-1),
                                   dict(live_fresh_frac=1.5), dict(live_fresh_frac=True)])
def test_malformed_basin_settings_are_refused(block):
    with pytest.raises(ValueError, match='live_'):
        noised_prior.settings(_args(**block))


def _basin_modeller(n=3, H_scale=1e4, caps=(1.0, 0.2, 0.1, 0.5), bank=0, fresh=0.1, redraws=4, chart='fd'):
    pytest.importorskip('mxtaltools.crystal_fd')
    from energy_sampling.energies.fd_coordinates import FDCoordinates
    fc = FDCoordinates(2, RADIUS)
    p = torch.tensor([P_MIPCAS] * n, dtype=torch.float64)
    s0 = fc.from_params(p, torch.ones(n, dtype=torch.float64))
    assert bool(s0.ok.all())
    m = Modeller.__new__(Modeller)
    m._noised_prior_settings = noised_prior.settings(_args(
        live_kernel='basin', live_tau=1.0, live_caps=list(caps), live_redraws=redraws, live_bank=bank,
        live_fresh_frac=fresh))
    batch = SimpleNamespace(basin_p=p, basin_hand=torch.ones(n), basin_H=H_scale * torch.eye(12).expand(n, 12, 12),
                            basin_live=torch.ones(n, 12, dtype=torch.bool), sg_ind=torch.full((n,), 2),
                            radius=torch.full((n,), RADIUS), fd_state=s0.s.float(),
                            num_graphs=n, cell_lengths=torch.zeros(n, 3))
    m.prior_dataset = SimpleNamespace(batch=batch)
    m.energy_function = SimpleNamespace(temperature=2.5, crystal_coordinates=chart, is_crystal=True,
                                        fd_coordinates_rows=lambda sg, r: FDCoordinates(int(sg), float(r[0])))
    return m, s0.s.float()


def test_basin_draws_land_on_chart_points_around_their_basin():
    torch.manual_seed(0)
    m, s0 = _basin_modeller(H_scale=1e4)
    L = m._basin_kernel_table()
    assert L.shape == (3, 12, 12)
    x = m._basin_fresh(torch.zeros(200, dtype=torch.long))
    assert bool((x.abs() <= 1).all()) and m._basin['misses'] == 0
    assert float((x - s0[0]).abs().max()) > 0          # the draws move
    # physically near a stiff basin: chart rows can jump at the orientation chart's singular points, so compare the
    # decoded cell lengths (reduced, so the same for every description of a crystal) with the basin's
    from energy_sampling.energies.fd_coordinates import FDCoordinates
    lengths = FDCoordinates(2, RADIUS).to_params(x.double())[0][:, :3]
    assert float((lengths - torch.tensor(P_MIPCAS[:3])).abs().max(1).values.median()) < 0.1


def test_a_draw_with_no_chart_point_is_redrawn_a_bounded_number_of_times_then_counted():
    torch.manual_seed(0)
    # a soft basin with 100 A caps: most draws leave the length box; the loop must still end
    m, s0 = _basin_modeller(H_scale=1e-6, caps=(100.0, 0.2, 0.1, 0.5), redraws=2)
    m._basin_kernel_table()
    x = m._basin_fresh(torch.zeros(50, dtype=torch.long))
    assert m._basin['misses'] > 0 and m._basin['draws'] == 50
    kept = (x == s0[0]).all(1)
    assert int(kept.sum()) == m._basin['misses']        # a missed row keeps its stored chart point


def test_the_bank_serves_draws_and_takes_back_the_fresh_ones():
    torch.manual_seed(0)
    m, _ = _basin_modeller(bank=8, fresh=0.5)
    m._live_L = m._basin_kernel_table()
    bank0 = m._basin['bank'].clone()
    assert bank0.shape == (3, 8, 12) and m._basin['draws'] == 24
    out = m._basin_kernel_draw(torch.zeros(10, 12), torch.arange(10) % 3, None)
    assert out.shape == (10, 12)
    changed = (m._basin['bank'] != bank0).any(-1)
    assert 1 <= int(changed.sum()) <= 5                 # at most ceil(0.5 x 10) fresh draws written back
    assert m._basin['draws'] == 24 + 5


def test_basin_draws_on_the_latent_chart_decode_to_crystals_near_their_basin():
    """latent_rows_from_params: canonicalise, one trainer-box image drawn uniformly, unclipped latent inside the
    live clamp; the trainer's own decode of those latents gives back cells near the basin's."""
    torch.manual_seed(0)
    from mxtaltools.crystal_fd.charts import LatentV1Chart
    m, _ = _basin_modeller(chart='latent')
    m.prior_dataset.batch.latent_params = lambda gauge_fix_free_axes=True: torch.zeros(3, 12)
    m._basin_kernel_table()
    x = m._basin_fresh(torch.zeros(100, dtype=torch.long))
    assert m._basin['misses'] == 0
    chart = LatentV1Chart(2, RADIUS)
    lo, hi = chart.clamp_bounds()
    assert bool(((x >= lo - 1e-6) & (x <= hi + 1e-6)).all())
    desc, clamped = chart.describe(x)
    assert not bool(clamped.any())
    lengths = torch.sort(desc.crystal_params()[0], dim=1).values
    assert float((lengths - torch.sort(torch.tensor(P_MIPCAS[:3]))[0]).abs().max(1).values.median()) < 0.1


def test_latent_rows_refuse_a_cell_outside_the_length_box():
    from energy_sampling.energies.molecular_crystal import latent_rows_from_params
    far = torch.tensor([P_MIPCAS], dtype=torch.float64)
    far[0, 2] = 45.0                                   # a 45 A axis: no trainer latent holds it
    x, ok = latent_rows_from_params(torch.cat([torch.tensor([P_MIPCAS], dtype=torch.float64), far]),
                                    torch.ones(2), 2, torch.full((2,), RADIUS))
    assert ok.tolist() == [True, False] and bool((x[1] == 0).all())


def _pd_hessians(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, 12, 12, generator=g, dtype=torch.float64)
    return M @ M.transpose(1, 2) * 50 + 5 * torch.eye(12, dtype=torch.float64)


def test_the_hessian_shape_is_the_thermal_ellipsoid_where_no_mode_is_held():
    H = 1e2 * _pd_hessians()
    L = noised_prior.basin_factors(H, torch.ones(4, 12, dtype=torch.bool), (1.0, 0.2, 0.1, 0.5), 2.5, 1.0, 'hessian')
    assert torch.allclose(L @ L.transpose(1, 2), 2.5 * torch.linalg.inv(H), rtol=1e-8, atol=1e-12)


def test_the_iso_shape_spends_the_hessian_shapes_expected_excess_evenly():
    """'iso' = sigma I in cap-scaled coordinates, sigma chosen so the quadratic model's expected excess
    1/2 tr(L^T H L) / kT equals the 'hessian' shape's, held modes included; sigma <= 1 (its cap)."""
    H, caps, kT = _pd_hessians(), (1.0, 0.2, 0.1, 0.5), 2.5
    live = torch.ones(4, 12, dtype=torch.bool)
    Lh = noised_prior.basin_factors(H, live, caps, kT, 1.0, 'hessian')
    Li = noised_prior.basin_factors(H, live, caps, kT, 1.0, 'iso')
    excess = lambda L: 0.5 * torch.einsum('nji,njk,nkl->nil', L, H / kT, L).diagonal(dim1=1, dim2=2).sum(1)
    assert torch.allclose(excess(Lh), excess(Li), rtol=1e-9)
    D = torch.tensor(caps, dtype=torch.float64).repeat_interleave(3)
    scaled = Li / D[None, :, None]
    sigma = scaled[:, 0, 0]
    assert torch.allclose(scaled, sigma[:, None, None] * torch.eye(12, dtype=torch.float64), atol=1e-12)
    assert bool((sigma > 0).all() & (sigma <= 1).all())
    # an indefinite Hessian (a saddle): the match is on |Hs|, the quadratic model both shapes spend tau / 2 nats in
    D_ = torch.diag(D)
    Hs = D_ @ H[0] @ D_ / kT
    lam, V = torch.linalg.eigh(Hs)
    lam[:2] = -torch.tensor([50.0, 0.2], dtype=torch.float64)
    Hi = (torch.linalg.inv(D_) @ (V * lam) @ V.T @ torch.linalg.inv(D_) * kT)[None]
    absH = (torch.linalg.inv(D_) @ (V * lam.abs()) @ V.T @ torch.linalg.inv(D_))[None]   # |H| / kT
    ex = lambda L: 0.5 * torch.einsum('nji,njk,nkl->nil', L, absH, L).diagonal(dim1=1, dim2=2).sum(1)
    Lh1 = noised_prior.basin_factors(Hi, live[:1], caps, kT, 1.0, 'hessian')
    Li1 = noised_prior.basin_factors(Hi, live[:1], caps, kT, 1.0, 'iso')
    assert torch.allclose(ex(Lh1), ex(Li1), rtol=1e-9)
    dead = live.clone()
    dead[:, 3] = False
    Ld = noised_prior.basin_factors(H, dead, caps, kT, 1.0, 'iso')
    assert bool((Ld[:, 3] == 0).all() & (Ld[:, :, 3] == 0).all())


def test_live_shape_is_refused_unless_the_basin_kernel_reads_it():
    with pytest.raises(ValueError, match='live_shape must be one of'):
        noised_prior.settings(_args(live_kernel='basin', live_shape='round'))
    with pytest.raises(ValueError, match="live_kernel 'basin' only"):
        noised_prior.settings(_args(live_shape='iso'))
    assert noised_prior.settings(_args(live_kernel='basin', live_shape='iso'))['live_shape'] == 'iso'


def test_iso_basin_draws_land_on_chart_points_too():
    torch.manual_seed(0)
    m, s0 = _basin_modeller(H_scale=1e4)
    m._noised_prior_settings['live_shape'] = 'iso'
    m._basin_kernel_table()
    x = m._basin_fresh(torch.zeros(100, dtype=torch.long))
    assert m._basin['misses'] == 0 and bool((x.abs() <= 1).all())
    assert float((x - s0[0]).abs().max()) > 0


def test_basin_draw_is_reproducible_under_a_generator_and_drops_nothing_silently():
    pytest.importorskip('mxtaltools.crystal_fd')
    from energy_sampling.energies.fd_coordinates import FDCoordinates
    p = torch.tensor([P_MIPCAS] * 2, dtype=torch.float64)
    L = noised_prior.basin_factors(1e4 * torch.eye(12).expand(2, 12, 12), torch.ones(2, 12, dtype=torch.bool),
                                   (1.0, 0.2, 0.1, 0.5), 2.5, 1.0)
    args = (p, torch.ones(2), L, torch.full((2,), 2), torch.full((2,), RADIUS, dtype=torch.float64),
            torch.tensor([0, 1, 1, 0]))
    fdc = lambda sg, r: FDCoordinates(int(sg), float(r[0]))
    for chart in ('fd', 'latent'):
        a, ok_a = noised_prior.basin_draw(*args, chart, fd_coordinates=fdc, generator=torch.Generator().manual_seed(3))
        b, ok_b = noised_prior.basin_draw(*args, chart, fd_coordinates=fdc, generator=torch.Generator().manual_seed(3))
        assert torch.equal(a, b) and torch.equal(ok_a, ok_b) and bool(ok_a.all())
    with pytest.raises(ValueError, match="'fd' or 'latent'"):
        noised_prior.basin_draw(*args, 'asu')


def test_the_fresh_gap_is_published_only_against_draws_of_the_readout_kernel():
    m = Modeller.__new__(Modeller)
    assert m._fresh_gap_source() is None                              # not a pre-noised prior
    m._noised_prior_settings = noised_prior.settings(_args(live_kernel='basin'))
    assert m._fresh_gap_source() == 'eval_bwd/mle_bank'
    for block in (dict(live_kernel='basin', live_shape='iso'), dict(live_kernel='basin', live_tau=0.5),
                  dict(live_kernel='basin', live_bank=0), dict(live_kernel='hessian')):
        m._noised_prior_settings = noised_prior.settings(_args(**block))
        assert m._fresh_gap_source() is None, block
    m._noised_prior_settings = noised_prior.settings(_args(train_kinds=['child']))
    caps = list(noised_prior.DEFAULTS['live_caps'])
    m._noised_prior_manifest = dict(child_kernel='basin', basin_kernel=dict(shape='hessian', tau=1.0, caps=caps))
    assert m._fresh_gap_source() == 'eval_bwd/mle'
    for man in (dict(child_kernel='iso'), dict(child_kernel='basin', basin_kernel=dict(shape='iso', tau=1.0, caps=caps)),
                dict(child_kernel='basin', basin_kernel=dict(shape='hessian', tau=1.0, caps=[2.0] + caps[1:]))):
        m._noised_prior_manifest = man
        assert m._fresh_gap_source() is None, man
    m._noised_prior_manifest = dict(child_kernel='basin', basin_kernel=dict(shape='hessian', tau=1.0, caps=caps))
    m._noised_prior_settings = noised_prior.settings(_args(train_kinds=['child', 'anchor']))
    assert m._fresh_gap_source() is None
