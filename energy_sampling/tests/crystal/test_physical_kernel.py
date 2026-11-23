"""noised_prior.live_kernel 'physical': physical_kick.py (metric_kick's step, certified canonicalisation, the full-route
fallback) and its trainer wiring on a Modeller built without __init__. What each test would catch:

  * a certified row that differs from canon.canonicalize, on and near the chart's faces as well as in the interior;
  * a draw outside the box, or one that does not round-trip through the production chart;
  * a kick centred on a child, or on another row's anchor;
  * a redraw loop with no bound;
  * the kernel running under a stage that reads log R, or beside buffers.store, which never calls it;
  * the global torch thread count left changed;
  * a step or a decode that differs from metric_kick's or the production chart's arithmetic;
  * tiles of one terminal holding different kicks, which terminal_mle would read as one terminal;
  * the key set on a route that would silently not draw it (a prior that is not a build, the INN trainer);
  * the fallback taking every row with nothing said.

CPU only, no model or energy.

    python -m pytest -q tests/crystal/test_physical_kernel.py
"""
import copy
import os
import pathlib
from types import MethodType, SimpleNamespace

import mxtaltools
import numpy as np
import pytest
import torch
import yaml

from energies.fd_coordinates import FDCoordinates, FDSlotLayout
from energy_sampling import config_invariants as ci, noised_prior, physical_kick as PK
from energy_sampling.train import Modeller
from mxtaltools.crystal_fd import charts, domains

pytestmark = pytest.mark.fast

MK_DEV = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'

GROUPS = (2, 14, 19, 61)
F64 = torch.float64


@pytest.fixture(scope='module')
def layout():
    return FDSlotLayout(GROUPS)


@pytest.fixture(scope='module')
def kits(layout):
    return PK.group_kits(layout)


def _anchors(k, n, seed, radius=4.0):
    """(unit rows [n, d] of a cell near close packing for a molecule of this radius, radii [n], r_g [n])."""
    gen = torch.Generator().manual_seed(seed)
    u = 0.05 + 0.9 * torch.rand(n, k.d, generator=gen, dtype=F64)
    u[:, :min(3, k.nc)] = 0.15 + 0.45 * torch.rand(n, min(3, k.nc), generator=gen, dtype=F64)
    return u, torch.full((n,), radius, dtype=F64), torch.full((n,), 0.55 * radius, dtype=F64)


def _on_and_near_faces(k, u):
    """Copies of the rows u with one coordinate on, or within 1e-12 to 1e-3 of, each of its two faces."""
    gaps = (0.0, 1e-12, 1e-9, 1e-7, 1e-5, 1e-3)
    out = []
    for j in range(k.d):
        for gap in gaps:
            for face in (gap, 1 - gap):
                v = u.clone()
                v[:, j] = face
                out.append(v)
    return torch.cat(out)


def _unit_gap(k, a, b):
    """[n] largest coordinate difference of unit rows, circle coordinates taken on their circle."""
    d = (a - b).abs()
    wrap = torch.tensor(FDCoordinates(k.g, 4.0).wrap_mask)
    d = torch.where(wrap, torch.minimum(d, 1 - d), d)
    return d.amax(1)


def _check_against_full_route(k, cell, t, R, radius):
    """Every certified row equals the full canonicaliser's chart row, and every certified rejection is one there.
    Returns (certified rows, rejected rows, rows left to the full route)."""
    u, state = PK.certified_chart(k, cell, t, R, radius)
    uf, okf = PK.full_route(k, cell, t, R, radius)
    yes, no = state == 1, state == 0
    assert bool(okf[yes].all()), 'a certified row has no chart point on the full route'
    assert not bool(okf[no].any()), 'a certified rejection has a chart point on the full route'
    if bool(yes.any()):
        assert float(_unit_gap(k, u[yes], uf[yes]).max()) < 1e-7
    return int(yes.sum()), int(no.sum()), int((state < 0).sum())


@pytest.mark.parametrize('g', GROUPS)
def test_certified_kicks_equal_the_full_canonicaliser(kits, g):
    k = kits[g]
    u, radius, rg = _anchors(k, 1500, seed=g)
    cell, t, R = PK.decode(k, u, radius)
    st = PK.draw_step(k, len(u), -2.0, -0.375, torch.Generator().manual_seed(g + 1))
    c2, t2, R2 = PK.apply_step(st, cell, t, R, rg, PK.t_fc(cell))
    yes, no, rest = _check_against_full_route(k, c2, t2, R2, radius)
    assert yes > 0.9 * len(u), (yes, no, rest)     # the comparison above is not over an empty set


@pytest.mark.parametrize('g', GROUPS)
@pytest.mark.parametrize('step', [0.0, 1e-9, 1e-3])
def test_certified_rows_on_and_near_faces_equal_the_full_canonicaliser(kits, g, step):
    k = kits[g]
    u, radius, rg = _anchors(k, 6, seed=g + 10)
    v = _on_and_near_faces(k, u)
    r, gy = radius[0].repeat(len(v)), rg[0].repeat(len(v))
    cell, t, R = PK.decode(k, v, r)
    d = PK.draw_step(k, len(v), 0.0, 0.0, torch.Generator().manual_seed(g + 2)) * step
    c2, t2, R2 = PK.apply_step(d, cell, t, R, gy, PK.t_fc(cell))
    yes, no, rest = _check_against_full_route(k, c2, t2, R2, r)
    assert yes > 0 and rest > 0, (yes, no, rest)   # some rows certified, and the rows on the walls left to the full route


def _mixed(layout, kits, n_per, seed=0):
    states, sg, radius, rg = [], [], [], []
    for i, g in enumerate(GROUPS):
        u, r, gy = _anchors(kits[g], n_per, seed=seed + i)
        states.append(layout.pack(g, (2 * u - 1).float()))
        sg.append(torch.full((n_per,), g))
        radius.append(r.float())
        rg.append(gy.float())
    return torch.cat(states), torch.cat(sg), torch.cat(radius), torch.cat(rg)


def test_a_four_group_draw_stays_in_the_box_on_its_slots_and_round_trips(layout, kits):
    torch.manual_seed(3)
    centres, sg, radius, rg = _mixed(layout, kits, 200)
    x, counts = PK.kick(kits, centres, sg, radius, rg, -2.0, -0.375, 5)
    assert x.dtype == torch.float32 and x.shape == centres.shape and bool(torch.isfinite(x).all())
    assert float(x.abs().max()) <= 1 and counts['rows'] == len(x) and counts['kept'] == 0
    assert bool((x[~layout.live_mask(sg)] == 0).all())                   # pad slots hold 0
    moved = (x - centres).abs().amax(1)
    assert bool((moved > 0).all())
    for g in GROUPS:
        at = sg == g
        fc = FDCoordinates(g, radius[at].to(F64))
        s = layout.unpack(g, x[at]).to(F64)
        params, hand, _ = fc.to_params(s)
        back = fc.from_params(params, hand)
        assert bool(back.ok.all())
        gap = _unit_gap(kits[g], (back.s + 1) / 2, (s + 1) / 2)
        assert float(gap.max()) < 1e-4, g


def test_the_radius_law_is_the_requested_one(kits):
    k = kits[2]
    st = PK.draw_step(k, 20000, -2.0, -0.375, torch.Generator().manual_seed(4))
    rho = torch.log10(st.norm(dim=1))
    assert float(rho.min()) >= -2.0 - 1e-9 and float(rho.max()) <= -0.375 + 1e-9
    assert abs(float(rho.mean()) + 1.1875) < 0.02
    dead = PK.draw_step(kits[61], 100, -2.0, -0.375, torch.Generator().manual_seed(5))
    assert bool((dead[:, 3:6] == 0).all())                               # an orthorhombic cell's angles do not move


@pytest.mark.parametrize('redraws', [0, 4])
def test_the_redraw_bound_ends_when_no_draw_can_succeed(kits, monkeypatch, redraws):
    k = kits[14]
    u, radius, rg = _anchors(k, 32, seed=7)
    s = 2 * u - 1
    calls = []

    def none(kit, cell, t, R, r):
        calls.append(len(cell))
        return torch.full((len(cell), kit.d), float('nan'), dtype=F64), torch.full((len(cell),), -1, dtype=torch.int8)

    monkeypatch.setattr(PK, 'certified_chart', none)
    monkeypatch.setattr(PK, 'full_route', lambda kit, c, t, R, r: (torch.full((len(c), kit.d), float('nan'), dtype=F64),
                                                                   torch.zeros(len(c), dtype=torch.bool)))
    out, counts = PK.kick_group(k, s, radius, rg, -2.0, -0.375, 1 + redraws)
    assert calls == [32] * (1 + redraws)
    assert torch.equal(out, s)                                           # every row keeps its anchor
    assert counts == dict(rows=32, full_route=32, redrawn=32, kept=32)


def test_the_thread_count_is_restored_after_a_draw_and_after_a_failure(layout, kits, monkeypatch):
    centres, sg, radius, rg = _mixed(layout, kits, 8)
    was = torch.get_num_threads()
    try:
        torch.set_num_threads(3)
        seen = []
        real = PK.kick_group
        monkeypatch.setattr(PK, 'kick_group', lambda *a, **k: (seen.append(torch.get_num_threads()), real(*a, **k))[1])
        PK.kick(kits, centres, sg, radius, rg, -2.0, -0.375, 5)
        assert set(seen) == {1} and torch.get_num_threads() == 3

        def boom(*a, **k):
            raise RuntimeError('inside the kick')
        monkeypatch.setattr(PK, 'kick_group', boom)
        with pytest.raises(RuntimeError, match='inside the kick'):
            PK.kick(kits, centres, sg, radius, rg, -2.0, -0.375, 5)
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(was)


def test_a_group_without_a_table_is_refused_by_name():
    with pytest.raises(ValueError, match='space group 33 has no basis table'):
        PK.group_kit(33, torch.arange(9))


def test_a_row_of_a_group_outside_the_run_is_refused_by_name(layout, kits):
    centres, sg, radius, rg = _mixed(layout, kits, 2)
    sg = sg.clone()
    sg[0] = 33
    with pytest.raises(ValueError, match='space group 33 has no table in this run'):
        PK.kick(kits, centres, sg, radius, rg, -2.0, -0.375, 5)


def test_the_carried_tables_are_unimodular_and_the_fold_margin_is_the_librarys():
    for g, rows in PK.BASIS_TABLES.items():
        if rows is None:
            continue
        N = torch.tensor(rows, dtype=F64).reshape(-1, 3, 3)
        assert torch.equal(N[0], torch.eye(3, dtype=F64)), g
        assert bool((torch.linalg.det(N).abs().round() == 1).all()), g
        assert len({tuple(r) for r in rows}) == len(rows), g
    assert PK._PAD == domains._PAD


def test_radius_of_gyration_rows_is_noised_prior_s():
    gen = torch.Generator().manual_seed(1)
    counts = torch.tensor([3, 5, 2, 7])
    pos = torch.randn(int(counts.sum()), 3, generator=gen)
    z = torch.randint(1, 9, (int(counts.sum()),), generator=gen)
    z[torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])] = 6       # one heavy atom per molecule at least
    batch = SimpleNamespace(pos=pos, z=z, batch=torch.arange(4).repeat_interleave(counts), num_graphs=4,
                            ptr=torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]))
    every = noised_prior.radius_of_gyration(batch).float()
    assert torch.allclose(PK.radius_of_gyration_rows(batch, torch.arange(4), chunk=3), every)
    assert torch.allclose(PK.radius_of_gyration_rows(batch, torch.tensor([3, 1]))[1], every[1])


# -- the trainer ---------------------------------------------------------------------------------------------------

def _prior(layout, kits):
    """Two anchors per group, each with two children stored far from it in the chart, and one harvested row."""
    states, sg, kinds, parents = [], [], [], []
    for i, g in enumerate(GROUPS):
        u, _, _ = _anchors(kits[g], 6, seed=20 + i)
        at = len(sg)
        states.append(layout.pack(g, (2 * u - 1).float()))
        sg += [g] * 6
        kinds += [noised_prior.KIND_ANCHOR] * 2 + [noised_prior.KIND_CHILD] * 4
        parents += [at, at + 1, at, at, at + 1, at + 1]
    n = len(sg)
    z = torch.full((3 * n,), 6)
    gen = torch.Generator().manual_seed(9)
    return SimpleNamespace(fd_state=torch.cat(states), sg_ind=torch.tensor(sg), radius=torch.full((n,), 4.0),
                           noised_parent=torch.tensor(parents), noised_kind=torch.tensor(kinds),
                           pos=torch.randn(3 * n, 3, generator=gen), z=z, batch=torch.arange(n).repeat_interleave(3),
                           ptr=torch.arange(n + 1) * 3, num_graphs=n), torch.tensor(kinds)


def _trainer(layout, prior, redraws=4, log_range=(-3.0, -3.0), free=True):
    m = Modeller.__new__(Modeller)
    m.energy_function = SimpleNamespace(fd_layout=layout)
    m._noised_prior_settings = {**noised_prior.DEFAULTS, 'live_kernel': 'physical', 'live_log_range': log_range,
                                'live_redraws': redraws}
    m._bwd_draw_is_reward_free = lambda: free
    m.prior_dataset = SimpleNamespace(batch=prior)
    return m


def _load(m, prior, kinds, fd_ok=None, keep=None, manifest=None):
    n = len(kinds)
    fd_ok = torch.ones(n, dtype=torch.bool) if fd_ok is None else fd_ok
    keep = kinds == noised_prior.KIND_CHILD if keep is None else keep
    out = Modeller._physical_kernel_parents(m, prior, manifest or {}, kinds, fd_ok, keep)
    prior_rows = torch.nonzero(out).flatten()
    m.prior_dataset.batch = SimpleNamespace(fd_state=prior.fd_state[prior_rows], num_graphs=len(prior_rows))
    return out, prior_rows


def test_every_draw_is_centred_on_the_parent_anchor_of_a_child(layout, kits, monkeypatch):
    prior, kinds = _prior(layout, kits)
    m = _trainer(layout, prior)
    keep, rows = _load(m, prior, kinds)
    seen = {}
    real = PK.kick

    def spy(k, centres, *a, **kw):
        seen['centres'] = centres.clone()
        return real(k, centres, *a, **kw)
    monkeypatch.setattr(PK, 'kick', spy)
    inds = torch.arange(len(rows)).numpy()
    latents = prior.fd_state[rows].to(F64)
    x = Modeller._physical_kernel_draw(m, latents, inds)
    assert torch.equal(seen['centres'], prior.fd_state[prior.noised_parent[rows]])
    assert not torch.equal(seen['centres'], prior.fd_state[rows])       # the children are stored elsewhere
    assert x.dtype == latents.dtype and x.device == latents.device
    near = (x.float() - prior.fd_state[prior.noised_parent[rows]]).abs().amax(1)
    far = (x.float() - prior.fd_state[rows]).abs().amax(1)
    assert bool((near < far).all())


def test_a_row_whose_parent_has_no_chart_point_is_left_out_counted_and_printed(layout, kits, capsys):
    prior, kinds = _prior(layout, kits)
    fd_ok = torch.ones(len(kinds), dtype=torch.bool)
    fd_ok[0] = False                                                     # the first anchor: two children in the rows
    m = _trainer(layout, prior)
    keep, rows = _load(m, prior, kinds, fd_ok=fd_ok)
    assert m._physical['no_parent'] == 2 and len(rows) == int((kinds == noised_prior.KIND_CHILD).sum()) - 2
    assert not bool((prior.noised_parent[rows] == 0).any())
    assert '2 kept rows' in capsys.readouterr().out
    assert torch.equal(m._physical['centre'][m._physical['parent'].long()], prior.fd_state[prior.noised_parent[rows]])


def test_no_usable_parent_at_all_is_refused(layout, kits):
    prior, kinds = _prior(layout, kits)
    with pytest.raises(ValueError, match='no kept row has a parent anchor'):
        _load(_trainer(layout, prior), prior, kinds, fd_ok=torch.zeros(len(kinds), dtype=torch.bool))


def test_a_harvested_row_or_a_foreign_parent_is_refused(layout, kits):
    prior, kinds = _prior(layout, kits)
    prior.noised_parent = prior.noised_parent.clone()
    prior.noised_parent[3] = -1
    with pytest.raises(ValueError, match='not an anchor'):
        _load(_trainer(layout, prior), prior, kinds)
    prior, kinds = _prior(layout, kits)
    prior.noised_parent = prior.noised_parent.clone()
    prior.noised_parent[3] = 6                                           # an anchor of the next space group
    with pytest.raises(ValueError, match='another space group'):
        _load(_trainer(layout, prior), prior, kinds)


def test_the_radius_law_defaults_to_the_build_and_needs_one(layout, kits):
    prior, kinds = _prior(layout, kits)
    m = _trainer(layout, prior, log_range=None)
    _load(m, prior, kinds, manifest=dict(child_kernel='metric', log_lo=-2.0, log_hi=-0.375))
    assert (m._physical['lo'], m._physical['hi']) == (-2.0, -0.375)
    with pytest.raises(ValueError, match='live_log_range'):
        _load(_trainer(layout, prior, log_range=None), prior, kinds, manifest=dict(child_kernel='iso'))


def test_the_reward_free_guard_raises(layout, kits):
    prior, kinds = _prior(layout, kits)
    m = _trainer(layout, prior, free=False)
    _, rows = _load(m, prior, kinds)
    with pytest.raises(ValueError, match='reward-free'):
        Modeller._physical_kernel_draw(m, prior.fd_state[rows], torch.arange(len(rows)).numpy())


def test_the_live_draw_dispatches_and_checks_the_table_alignment(layout, kits):
    prior, kinds = _prior(layout, kits)
    m = _trainer(layout, prior)
    _, rows = _load(m, prior, kinds)
    latents = prior.fd_state[rows]
    x = Modeller._live_kernel_draw(m, latents, torch.arange(len(rows)).numpy())
    assert m._physical['rows'] == len(rows) and not torch.equal(x, latents)
    n = Modeller.__new__(Modeller)
    n.__dict__.update(m.__dict__)
    n._live_L = None
    n.prior_dataset = SimpleNamespace(batch=SimpleNamespace(fd_state=latents.flip(0), num_graphs=len(rows)))
    with pytest.raises(ValueError, match='not aligned'):
        Modeller._live_kernel_draw(n, latents, torch.arange(len(rows)).numpy())


# -- settings and refusals -----------------------------------------------------------------------------------------

def _settings(**block):
    return noised_prior.settings(SimpleNamespace(noised_prior=block))


def test_settings_accept_the_kernel_with_and_without_a_range():
    assert _settings(live_kernel='physical')['live_log_range'] is None
    assert list(_settings(live_kernel='physical', live_log_range=[-2.0, -0.5])['live_log_range']) == [-2.0, -0.5]


@pytest.mark.parametrize('block, message', [
    (dict(live_kernel='metric', live_log_range=[-2.0, -0.5]), "'chart' or 'physical' only"),
    (dict(live_kernel='physical', live_log_range=[-0.5, -2.0]), 'lo <= hi'),
    (dict(live_kernel='physical', live_log_range=[-2.0]), 'live_log_range'),
    (dict(live_kernel='physical', live_shape='iso'), "'basin' only"),
    (dict(live_kernel='physical', live_redraws=-1), 'live_redraws'),
])
def test_settings_refuse_the_wrong_combinations(block, message):
    with pytest.raises(ValueError, match=message):
        _settings(**block)


@pytest.mark.parametrize('kernel, refused', [('physical', True), ('chart', False), (None, False)])
def test_the_row_store_refuses_this_kernel_and_only_this_one(kernel, refused):
    from config_invariants import ERROR, row_store_problems
    cfg = {'buffers': {'store': {'enabled': True}}, 'noised_prior': {'live_kernel': kernel}}
    hit = [m for sev, m in row_store_problems(cfg) if sev == ERROR and 'live_kernel' in m]
    assert bool(hit) is refused
    cfg['buffers']['store']['enabled'] = False
    assert not row_store_problems(cfg)


def test_the_protocol_card_names_the_kernel():
    import protocol_card
    args = SimpleNamespace(noised_prior={'live_kernel': 'physical', 'train_kinds': ['child'], 'window_kT': 6.0},
                           prior_path='prior.pt')
    _, text = protocol_card._rows('bwd', SimpleNamespace(bwd_sampling_mode='dataset'), args, {})
    assert "physical kick of the row's parent anchor" in text


# -- the law: the kernel's own arithmetic against the production code it restates -----------------------------------

def _csd_rows(g, per=8):
    """Real Z' = 1 rows of space group g (mxtaltools mini_new_csd.pt) as one batch."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    data = torch.load(os.path.join(os.path.dirname(os.path.dirname(mxtaltools.__file__)), 'mini_datasets', 'mini_new_csd.pt'),
                      weights_only=False)
    rows = [e for e in data if int(e.sg_ind) == g and int(e.z_prime) == 1 and bool(e.is_well_defined)][:per]
    assert rows, g
    return collate_data_list(rows)


@pytest.mark.parametrize('g', GROUPS)
def test_decode_is_the_production_chart(kits, g):
    k = kits[g]
    u, _, _ = _anchors(k, 600, seed=g + 30)
    radius = 3.0 + 2.0 * torch.rand(600, generator=torch.Generator().manual_seed(g), dtype=F64)
    fc = FDCoordinates(g, radius)
    desc, _ = fc.to_description(2 * u - 1)
    cell, t, R = PK.decode(k, u, radius)
    assert float((cell - desc.cell).abs().max()) < 1e-9
    gap = t - desc.t[:, 0]
    assert float((gap - gap.round()).abs().max()) < 1e-9                 # the description wraps its centres into [0, 1)
    assert float((R - desc.R[:, 0]).abs().max()) < 1e-9
    v = charts.to_unit(fc.chart, fc.chart.inverse(desc)[0])
    assert float((PK.cell_unit(k, cell, radius) - v[:, :k.nc]).abs().max()) < 1e-9


@pytest.mark.parametrize('g', GROUPS)
def test_the_step_is_metric_kick_with_the_same_draws(kits, g):
    """draw_step and apply_step against noised_prior.metric_kick(chart='fd') on a real batch, one seed for both."""
    b, k = _csd_rows(g), kits[g]
    n = int(b.num_graphs)
    p0 = b.full_cell_parameters().detach().to(F64)
    rg = noised_prior.radius_of_gyration(b)
    st = PK.draw_step(k, n, -1.0, -0.375, torch.Generator().manual_seed(g))
    c2, t2, R2 = PK.apply_step(st, p0[:, :6], p0[:, 6:9], noised_prior._rodrigues(p0[:, 9:12]), rg, PK.t_fc(p0[:, :6]))
    noised_prior.metric_kick(b, -1.0, -0.375, generator=torch.Generator().manual_seed(g), chart='fd')
    p1 = b.full_cell_parameters().detach().to(F64)
    assert float((c2 - p1[:, :6]).abs().max()) < 2e-5                    # the batch stores float32
    assert float((t2 - p1[:, 6:9]).abs().max()) < 2e-5
    assert float((R2 - noised_prior._rodrigues(p1[:, 9:12])).abs().max()) < 1e-4
    assert float(st.norm(dim=1).min()) > 0.09                            # the comparison is over a real step


def test_the_radius_is_rounded_as_the_production_chart_rounds_it(layout, kits, monkeypatch):
    centres, sg, _, rg = _mixed(layout, kits, 16)
    radius = (3.0 + 2.0 * torch.rand(len(sg), generator=torch.Generator().manual_seed(6))).float()
    seen = []
    real = PK.kick_group
    monkeypatch.setattr(PK, 'kick_group', lambda kit, s, r, *a, **kw: (seen.append(r.clone()), real(kit, s, r, *a, **kw))[1])
    PK.kick(kits, centres, sg, radius, rg, -2.0, -0.375, 5)
    want = torch.round(radius.to(F64) * 1e6) / 1e6
    assert not torch.equal(want, radius.to(F64))                         # the rounding is not a no-op on these radii
    assert torch.equal(torch.cat(seen).sort()[0], want.sort()[0])


# -- the trainer: tiles, draw counts, load wiring, refusals ---------------------------------------------------------

def test_every_tile_of_a_terminal_holds_the_same_kick(layout, kits, monkeypatch):
    prior, kinds = _prior(layout, kits)
    m = _trainer(layout, prior)
    _, rows = _load(m, prior, kinds)
    B, K = len(rows), 3
    sizes = []
    real = PK.kick
    monkeypatch.setattr(PK, 'kick', lambda kit, c, *a, **kw: (sizes.append(len(c)), real(kit, c, *a, **kw))[1])
    x = Modeller._live_kernel_draw(m, prior.fd_state[rows].repeat_interleave(K, 0),
                                   torch.arange(B).repeat_interleave(K).numpy(), repeats=K)
    assert sizes == [B] and x.shape[0] == B * K and m._physical['rows'] == B
    tiles = x.reshape(B, K, -1)
    assert torch.equal(tiles[:, 0], tiles[:, 1]) and torch.equal(tiles[:, 0], tiles[:, 2])
    assert len(torch.unique(tiles[:, 0], dim=0)) == B                    # the terminals themselves are different points
    with pytest.raises(ValueError, match='tiles per terminal'):
        Modeller._live_kernel_draw(m, prior.fd_state[rows], torch.arange(B).numpy(), repeats=K)


def test_draw_bwd_sample_hands_its_repeats_to_the_kernel():
    seen = {}

    class Stop(Exception):
        pass

    def stop(latents, inds, mol_batch, repeats=1):
        seen['repeats'] = repeats
        raise Stop
    rows = SimpleNamespace(loader=lambda **kw: iter([(SimpleNamespace(), [0, 0])]))
    m = SimpleNamespace(row_store=None, bwd_sampling_mode='dataset', prior_dataset=rows, batch_size=1, device='cpu',
                        _batch_latents=lambda b: torch.zeros(2, 3), _noised_prior_settings=dict(live_kernel='physical'),
                        _live_kernel_draw=stop)
    with pytest.raises(Stop):
        Modeller.draw_bwd_sample(m, 2)
    assert seen['repeats'] == 2


@pytest.mark.parametrize('redraws', [0, 4])
def test_the_draw_count_reaching_the_kick_is_one_plus_the_redraws(layout, kits, monkeypatch, redraws):
    prior, kinds = _prior(layout, kits)
    m = _trainer(layout, prior, redraws=redraws)
    _, rows = _load(m, prior, kinds)
    tries = []
    real = PK.kick

    def spy(kit, c, s, r, g, lo, hi, n, *a, **kw):
        tries.append(n)
        return real(kit, c, s, r, g, lo, hi, n, *a, **kw)
    monkeypatch.setattr(PK, 'kick', spy)
    x = Modeller._physical_kernel_draw(m, prior.fd_state[rows], torch.arange(len(rows)).numpy())
    assert tries == [1 + redraws]
    assert bool((x != prior.fd_state[prior.noised_parent[rows]]).any(1).all())      # every row was kicked


def test_the_parent_table_holds_each_anchor_s_radius_of_gyration(layout, kits):
    prior, kinds = _prior(layout, kits)
    m = _trainer(layout, prior)
    _, rows = _load(m, prior, kinds)
    parents = torch.unique(prior.noised_parent[rows])
    want = noised_prior.radius_of_gyration(prior).float()[parents]
    assert torch.allclose(m._physical['rg'], want)
    assert not torch.allclose(m._physical['rg'], m._physical['radius'])  # r_g is not the molecule radius


def test_a_parent_that_is_a_child_is_refused(layout, kits):
    prior, kinds = _prior(layout, kits)
    prior.noised_parent = prior.noised_parent.clone()
    prior.noised_parent[3] = 2                                           # row 2 is a child of the same group
    with pytest.raises(ValueError, match='not an anchor'):
        _load(_trainer(layout, prior), prior, kinds)


def test_a_parent_of_another_molecule_radius_is_refused(layout, kits):
    prior, kinds = _prior(layout, kits)
    prior.radius = prior.radius.clone()
    prior.radius[0] = 4.1                                                # the anchor of children 2 and 3
    with pytest.raises(ValueError, match='molecule radius'):
        _load(_trainer(layout, prior), prior, kinds)


class _Rows(SimpleNamespace):
    def to(self, device):
        return self


def _load_setup(layout, kits, monkeypatch, train=None):
    """A Modeller on the fixture prior with the steps of _load_noised_prior around the selection stubbed."""
    prior, kinds = _prior(layout, kits)
    n = len(kinds)
    prior.packing_coeff, prior.noised_split = torch.full((n,), 0.7), torch.zeros(n, dtype=torch.long)
    prior.subsample_new_batch = lambda idx: _Rows(fd_state=prior.fd_state[idx.cpu()], num_graphs=len(idx))
    train = kinds == noised_prior.KIND_CHILD if train is None else train
    masks = dict(train=train, heldout=torch.zeros(n, dtype=torch.bool), anchors=kinds == noised_prior.KIND_ANCHOR,
                 buffer=kinds == noised_prior.KIND_CHILD)
    for name, fn in dict(identity_mismatch=lambda *a: [], identity=lambda *a: {},
                         check_rows=lambda *a, **k: np.arange(4), compare_terms=lambda *a: (0.0, 0.0, {}),
                         condition_groups=lambda p: torch.zeros(n, dtype=torch.long),
                         window_energy=lambda *a, **k: torch.zeros(n, dtype=F64),
                         select_rows=lambda *a, **k: (masks, None)).items():
        monkeypatch.setattr(noised_prior, name, fn)
    m = Modeller.__new__(Modeller)
    m.device = 'cpu'
    m.energy_function = SimpleNamespace(fd_layout=layout, crystal_coordinates='fd')
    m.args = SimpleNamespace(energy_config=SimpleNamespace(temperature=2.5), prior_path='p.pt', seed=0,
                             energy_function='elj', noised_prior=dict(live_kernel='physical', live_log_range=[-3.0, -3.0]))
    m._fd_chart_points_cached = lambda p, path: (torch.ones(n, dtype=torch.bool), None, None)
    m._attach_fd_reference = lambda p, points: points[0]
    m._score_prior_rows = lambda b, announce=False: (None, None)
    m._stored_reward_terms = lambda p: (torch.zeros(n, dtype=F64), torch.zeros(n, dtype=F64))
    return m, prior, kinds, dict(format_version=noised_prior.FORMAT_VERSION, identity={})


def test_the_load_builds_the_parent_table_and_drops_rows_without_a_centre(layout, kits, monkeypatch):
    m, prior, kinds, manifest = _load_setup(layout, kits, monkeypatch)
    fd_ok = torch.ones(len(kinds), dtype=torch.bool)
    fd_ok[0] = False                                                     # an anchor with two children in the rows
    m._fd_chart_points_cached = lambda p, path: (fd_ok, None, None)
    energy, batch = Modeller._load_noised_prior(m, prior, manifest)
    children = int((kinds == noised_prior.KIND_CHILD).sum())
    assert m._physical['no_parent'] == 2 and m._physical['n_rows'] == children - 2
    assert len(energy) == batch.num_graphs == children - 2


def test_an_empty_selection_is_reported_as_the_selection(layout, kits, monkeypatch):
    m, prior, kinds, manifest = _load_setup(layout, kits, monkeypatch, train=torch.zeros(24, dtype=torch.bool))
    with pytest.raises(ValueError, match='no row survives the load-time selection'):
        Modeller._load_noised_prior(m, prior, manifest)


def test_a_prior_that_is_not_a_build_is_refused():
    args = SimpleNamespace(energy_config=SimpleNamespace(temperature=2.5), prior_path='p.pt',
                           noised_prior=dict(live_kernel='physical'))
    m = SimpleNamespace(args=args, NOISED_PRIOR_RECORD=Modeller.NOISED_PRIOR_RECORD)
    with pytest.raises(ValueError, match='needs a pre-noised prior'):
        Modeller._noised_prior_record(m, None)
    args.noised_prior = dict(live_kernel='chart')                        # the other kernels keep their behaviour
    assert Modeller._noised_prior_record(m, None) == {'data/prior_prenoised': 0}


def test_the_inn_route_refuses_the_kernel():
    from inn_modeller import InnModeller
    args = SimpleNamespace(noised_prior=dict(live_kernel='physical'))
    with pytest.raises(ValueError, match='never calls'):
        InnModeller.check_buffer_options(SimpleNamespace(inn=SimpleNamespace(), args=args))


def test_a_run_gone_to_the_full_route_is_told_once(layout, kits, monkeypatch, capsys):
    prior, kinds = _prior(layout, kits)
    m = _trainer(layout, prior)
    _, rows = _load(m, prior, kinds)
    capsys.readouterr()
    monkeypatch.setattr(PK, 'FULL_ROUTE_WARN_ROWS', 2 * len(rows))
    draw = lambda: Modeller._physical_kernel_draw(m, prior.fd_state[rows], torch.arange(len(rows)).numpy())
    draw()
    assert 'WARNING' not in capsys.readouterr().out                      # healthy, and below the row count
    m._physical.update(rows=0, full_route=0)
    monkeypatch.setattr(PK, 'certified_chart', lambda kit, c, t, R, r: (
        torch.full((len(c), kit.d), float('nan'), dtype=F64), torch.full((len(c),), -1, dtype=torch.int8)))
    draw()
    assert 'WARNING' not in capsys.readouterr().out                      # below the row count
    draw()
    assert capsys.readouterr().out.count('took the full canonicaliser') == 1
    draw()
    assert 'WARNING' not in capsys.readouterr().out


def test_settings_the_kernel_does_not_read_are_printed_at_load(layout, kits, capsys):
    prior, kinds = _prior(layout, kits)
    m = _trainer(layout, prior)
    m._noised_prior_settings = {**noised_prior.DEFAULTS, 'live_kernel': 'physical', 'live_log_range': (-3.0, -3.0),
                                'live_tau': 7.0, 'live_bank': 1024, 'live_caps': [1.0, 0.2, 0.1, 0.5]}
    _load(m, prior, kinds)
    assert "not read by this kernel: {'live_tau': 7.0, 'live_bank': 1024}" in capsys.readouterr().out


def _physical_mk_dev():
    cfg = yaml.safe_load(open(MK_DEV, encoding='utf-8'))
    cfg['noised_prior']['live_kernel'] = 'physical'
    return cfg


def _rule_hits(cfg):
    return [v.detail for v in ci.check(cfg) if v.rule == 'physical_kernel_is_reward_free']


@pytest.mark.parametrize('term, branch', [('tb', 'bwd'), ('subtb', 'bwd'), (ci.CROSS_BRANCH_VARGRAD_COEFF, 'fwd')])
def test_a_later_dataset_stage_that_reads_log_r_is_refused_at_load(term, branch):
    """The rule on the raw config, and the trainer's raise at step 0 on a stage it has not reached."""
    from energy_sampling.protocol import StageProtocol
    from energy_sampling.utils import dict2namespace
    cfg = _physical_mk_dev()
    assert not _rule_hits(cfg)
    stages = cfg[ci.PROTOCOL_LIBRARY][cfg[ci.PROTOCOL_SELECTOR]]['stages']
    later = copy.deepcopy(stages[0])
    assert later['bwd_sampling_mode'] == 'dataset'
    later['name'] = 'later_dataset'
    later.setdefault('loss_coeffs', {}).setdefault(branch, {})[term] = 1.0
    stages.insert(1, later)
    hit = _rule_hits(cfg)
    assert len(hit) == 1 and 'later_dataset' in hit[0] and term in hit[0]
    m = SimpleNamespace(args=dict2namespace(copy.deepcopy(cfg)), gfn_model=SimpleNamespace(), stage=None,
                        _warn_if_z_untrained=lambda: None)
    m.protocol = StageProtocol(m)
    with pytest.raises(ValueError, match='later_dataset'):
        MethodType(Modeller.set_loss_coeffs, m)()


def test_the_rule_ignores_a_reward_reading_stage_of_another_source_or_kernel():
    cfg = _physical_mk_dev()
    stages = cfg[ci.PROTOCOL_LIBRARY][cfg[ci.PROTOCOL_SELECTOR]]['stages']
    later = copy.deepcopy(stages[0])
    later.update(name='later_prior', bwd_sampling_mode='prior')
    later.setdefault('loss_coeffs', {}).setdefault('bwd', {})['tb'] = 1.0
    stages.insert(1, later)
    assert not _rule_hits(cfg)
    later['bwd_sampling_mode'] = 'dataset'
    cfg['noised_prior']['live_kernel'] = 'chart'
    assert not _rule_hits(cfg)


def test_the_miner_lists_the_canonicaliser_s_choices_and_they_make_an_accepted_table(layout):
    from data_processing.mine_basis_tables import IDENTITY, literal, mine
    bare = PK.group_kit(14, layout.slots[14], table=(IDENTITY,))
    u, radius, rg = _anchors(bare, 300, seed=41)
    entries = mine(bare, 2 * u - 1, radius, rg, 4000, -2.0, -0.375, seed=3)
    table = [n for n, _ in entries]
    assert table[0] == IDENTITY and len(set(table)) == len(table) > 1
    assert sum(c for _, c in entries) > 3900
    assert [c for _, c in entries[1:]] == sorted((c for _, c in entries[1:]), reverse=True)
    N = torch.tensor(table, dtype=F64).reshape(-1, 3, 3)
    assert bool((torch.linalg.det(N).abs().round() == 1).all())
    assert eval(literal(14, entries).strip().split(': ', 1)[1].rstrip(',')) == tuple(table)
    k = PK.group_kit(14, layout.slots[14], table=table)
    cell, t, R = PK.decode(k, u, radius)
    st = PK.draw_step(k, len(u), -2.0, -0.375, torch.Generator().manual_seed(4))
    c2, t2, R2 = PK.apply_step(st, cell, t, R, rg, PK.t_fc(cell))
    yes, no, rest = _check_against_full_route(k, c2, t2, R2, radius)
    assert yes > 0.9 * len(u), (yes, no, rest)
