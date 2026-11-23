"""noised_prior.live_kernel 'physical': physical_kick.py (metric_kick's step, certified canonicalisation, the full-route
fallback) and its trainer wiring on a Modeller built without __init__. What each test would catch:

  * a certified row that differs from canon.canonicalize, on and near the chart's faces as well as in the interior;
  * a draw outside the box, or one that does not round-trip through the production chart;
  * a kick centred on a child, or on another row's anchor;
  * a redraw loop with no bound;
  * the kernel running under a stage that reads log R, or beside buffers.store, which never calls it;
  * the global torch thread count left changed.

CPU only, no model or energy.

    python -m pytest -q tests/crystal/test_physical_kernel.py
"""
from types import SimpleNamespace

import pytest
import torch

from energies.fd_coordinates import FDCoordinates, FDSlotLayout
from energy_sampling import noised_prior, physical_kick as PK
from energy_sampling.train import Modeller
from mxtaltools.crystal_fd import domains

pytestmark = pytest.mark.fast

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
    m._noised_prior_settings = dict(live_kernel='physical', live_log_range=log_range, live_redraws=redraws)
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
