"""The temperature-first MLE kick (cfg:noised_prior.live_temperature_range): each MLE row draws its temperature T first,
log-uniform over the range, then a Gaussian step of variance T / kappa per live coordinate, and is labelled T
(physical_kick.draw_step / kick with sigma, train.py Modeller._kick_fresh, _kick_anchors, _kick_kappa,
_draw_kick_temperatures, _store_draw; noised_prior.settings; config_invariants.row_store_problems). What each test would
catch:

  * a step whose length law is not sigma^2 chi^2_d, or whose direction is not uniform, or that moves a dead coordinate;
  * the length-first path changed (its generator draws and arithmetic);
  * sigma routed to the wrong rows of a mixed-group batch, or a non-positive sigma accepted;
  * a slot whose temperature changes through its redraws, or a kappa read from the wrong condition's template;
  * a label other than the drawn T, or the global generators advanced by the draw;
  * the key accepted where it is dead or outside the temperature column's fitted span.

CPU only, no model or energy.

    python -m pytest -q tests/crystal/test_kick_temperature_first.py
"""
import math
from types import SimpleNamespace

import pytest
import torch

from energy_sampling import config_invariants as ci, noised_prior, physical_kick as PK
from energy_sampling.tests.crystal import store_stub as S

pytestmark = pytest.mark.fast
F64 = torch.float64
LIVE10 = SimpleNamespace(live=torch.tensor([1.0] * 10 + [0.0] * 2, dtype=F64))


# -- physical_kick ---------------------------------------------------------------------------------------------------

def test_a_gaussian_step_has_length_squared_sigma_squared_chi_squared_and_a_uniform_direction():
    n, sigma = 200000, 0.3
    st = PK.draw_step(LIVE10, n, -2.0, -0.375, torch.Generator().manual_seed(1), sigma=torch.full((n,), sigma))
    r2 = (st ** 2).sum(1) / sigma ** 2                       # chi^2 with d = 10: mean 10, variance 20
    assert abs(float(r2.mean()) - 10.0) < 0.05 and abs(float(r2.var()) - 20.0) < 0.4
    assert bool((st[:, 10:] == 0).all())
    u = st[:, :10] / st[:, :10].norm(dim=1, keepdim=True)
    assert float(u.mean(0).abs().max()) < 0.01


def test_the_length_first_step_is_unchanged():
    n, g1, g2 = 64, torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    got = PK.draw_step(LIVE10, n, -2.0, -0.375, g1)
    y = torch.randn(n, 12, dtype=F64, generator=g2) * LIVE10.live
    rho = 10 ** (-2.0 + 1.625 * torch.rand(n, dtype=F64, generator=g2))
    assert torch.equal(got, rho[:, None] * (y / y.norm(dim=1, keepdim=True)))


def test_kick_routes_each_rows_sigma_to_its_group_and_refuses_a_bad_one(monkeypatch):
    seen = {}

    def kick_group(k, s, radius, rg, lo, hi, tries, generator=None, rho_out=None, sigma=None):
        seen[k.g] = sigma.clone()
        return s.clone(), dict(rows=len(s), full_route=0, redrawn=0, kept=0)
    monkeypatch.setattr(PK, 'kick_group', kick_group)
    kits = {g: SimpleNamespace(g=g, slots=torch.arange(3)) for g in (2, 14)}
    sg = torch.tensor([2, 14, 2, 14, 14])
    sigma = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=F64)
    PK.kick(kits, torch.zeros(5, 3), sg, torch.ones(5), torch.ones(5), -2.0, -1.0, 2, sigma=sigma)
    assert torch.equal(seen[2], sigma[[0, 2]]) and torch.equal(seen[14], sigma[[1, 3, 4]])
    for bad in (torch.tensor([0.1, 0.0, 0.3, 0.4, 0.5]), torch.tensor([0.1, 0.2])):
        with pytest.raises(ValueError, match='sigma'):
            PK.kick(kits, torch.zeros(5, 3), sg, torch.ones(5), torch.ones(5), -2.0, -1.0, 2, sigma=bad)


# -- the trainer -----------------------------------------------------------------------------------------------------

def _tfirst(monkeypatch, keep=None, atoms=(3.0, 5.0, 7.0, 9.0), rng=(2.5, 10.0)):
    """A store-stub Modeller with temperature-first kicks, and the list of (rows, sigma) the kick was handed."""
    m = S.build()
    m.energy_function.temperature_conditioning = True
    m._noised_prior_settings.update(kick_kappa_per_atom=15.0, live_temperature_range=list(rng))
    m._physical['kits'] = {g: SimpleNamespace(live=LIVE10.live) for g in S.GROUPS}
    m._template_atoms = torch.tensor(atoms, dtype=F64)
    calls = []

    def kick(kits, centres, sg, radius, rg, lo, hi, tries, generator=None, rho_out=None, sigma=None):
        x = (centres + 0.01).float()
        kept = torch.zeros(len(x), dtype=torch.bool) if keep is None else keep(centres)
        x[kept] = centres[kept].float()
        if rho_out is not None:
            step = torch.full((len(x),), 0.05, dtype=F64) if sigma is None else sigma
            rho_out[:] = torch.where(kept, torch.zeros_like(step), step)
        calls.append(dict(centres=centres.clone(), sigma=None if sigma is None else sigma.clone()))
        return x, dict(rows=len(x), full_route=0, redrawn=0, kept=int(kept.sum()))
    monkeypatch.setattr(S.physical_kick, 'kick', kick)
    return m, calls


def test_each_rows_sigma_is_root_t_over_its_conditions_kappa(monkeypatch):
    m, calls = _tfirst(monkeypatch)
    st = m.row_store
    rows = torch.nonzero(st.kind[:st.high_water] == S.CS.ANCHOR).flatten()
    T = torch.linspace(2.5, 10.0, rows.numel(), dtype=F64)
    got_rows, _, rho, temp = m._kick_fresh(rows, with_rho=True, temperature=T)
    kappa = 15.0 * m._template_atoms[st.tmpl_of[st.cid[rows].long()].long()]
    assert torch.allclose(calls[0]['sigma'], torch.sqrt(T / kappa))
    assert torch.equal(got_rows, rows) and torch.equal(temp, T)


def test_a_redrawn_slot_keeps_its_temperature_and_takes_the_new_anchors_kappa(monkeypatch):
    m0 = S.build()
    st0 = m0.row_store
    a0 = st0.x[0].clone()
    m, calls = _tfirst(monkeypatch, keep=lambda c: (c == a0).all(1))   # anchor 0 never leaves its point
    st = m.row_store
    anchors = torch.nonzero(st.kind[:st.high_water] == S.CS.ANCHOR).flatten()
    other = int(anchors[torch.nonzero(st.cid[anchors] != st.cid[0]).flatten()[0]])
    monkeypatch.setattr(st, 'draw_anchors', lambda n, by='anchor': torch.full((n,), other))
    T = torch.tensor([3.0, 7.0], dtype=F64)
    rows, x, rho, temp = m._kick_fresh(torch.tensor([0, other]), with_rho=True, temperature=T)
    assert rows.tolist() == [other, other] and torch.equal(temp, T)
    kappa_other = 15.0 * float(m._template_atoms[int(st.tmpl_of[int(st.cid[other])])])
    assert torch.allclose(calls[1]['sigma'], torch.tensor([math.sqrt(3.0 / kappa_other)], dtype=F64))


def test_a_slot_dropped_after_its_redraws_drops_its_temperature(monkeypatch):
    m0 = S.build()
    a0 = m0.row_store.x[0].clone()
    m, _ = _tfirst(monkeypatch, keep=lambda c: (c == a0).all(1))
    st = m.row_store
    other = int(torch.nonzero(st.kind[:st.high_water] == S.CS.ANCHOR).flatten()[1])
    monkeypatch.setattr(st, 'draw_anchors', lambda n, by='anchor': torch.zeros(n, dtype=torch.long))
    rows, _, temp = m._kick_fresh(torch.tensor([0, other]), temperature=torch.tensor([3.0, 7.0], dtype=F64))
    assert rows.tolist() == [other] and temp.tolist() == [7.0] and m._physical['mle_kept_dropped'] == 1


def test_the_kick_temperatures_are_log_uniform_from_the_kicks_own_generator(monkeypatch):
    m, _ = _tfirst(monkeypatch)
    state = torch.get_rng_state()
    t = m._draw_kick_temperatures(100000)
    assert torch.equal(torch.get_rng_state(), state)
    assert float(t.min()) >= 2.5 and float(t.max()) <= 10.0
    lt = torch.log(t)
    assert abs(float(lt.mean()) - 0.5 * (math.log(2.5) + math.log(10.0))) < 0.01
    assert abs(float(lt.var()) - (math.log(4.0) ** 2) / 12) < 0.01


def test_the_mle_draw_labels_each_row_with_its_drawn_temperature(monkeypatch):
    m, calls = _tfirst(monkeypatch)
    ef, seen = m.energy_function, {}
    orig = ef.condition_samples

    def condition_samples(mol, temperature=None, **kw):
        seen['T'] = None if temperature is None else temperature.clone()
        return orig(mol, temperature=temperature, **kw)
    ef.condition_samples = condition_samples
    _, cid, inds, x, log_r, mol, traj = m.draw_bwd_sample(2)
    sigma = calls[0]['sigma']
    rows_kappa = sigma.numel()
    assert inds is None and log_r is None and seen['T'].numel() == 2 * rows_kappa
    T = seen['T'][0::2].double()
    assert torch.equal(seen['T'][0::2], seen['T'][1::2])
    assert float(T.min()) >= 2.5 - 1e-6 and float(T.max()) <= 10.0 + 1e-6
    st = m.row_store
    kappa = (T / sigma ** 2)                                   # sigma^2 = T / kappa
    assert torch.allclose(kappa, 15.0 * m._template_atoms[st.tmpl_of[cid[0::2]].long()], rtol=1e-5)


def test_without_the_range_the_label_is_the_effective_temperature(monkeypatch):
    m, calls = _tfirst(monkeypatch)
    m._noised_prior_settings['live_temperature_range'] = None
    assert m._kick_temperature_range() is None
    m.draw_bwd_sample(1)
    assert calls[0]['sigma'] is None


# -- settings and config ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('bad', [[0.0, 10.0], [10.0, 2.5], [2.5], 'hot', [2.5, float('inf')], [True, 10.0]])
def test_a_malformed_range_is_refused(bad):
    with pytest.raises(ValueError, match='live_temperature_range'):
        noised_prior.settings(SimpleNamespace(noised_prior={'live_temperature_range': bad}))


def test_a_well_formed_range_passes():
    assert noised_prior.settings(SimpleNamespace(noised_prior={'live_temperature_range': [2.5, 10.0]}))[
        'live_temperature_range'] == [2.5, 10.0]


def _problems(monkeypatch, t_rng, tcond=True, kappa=15.0, rng=(math.log10(2.5), 1.0)):
    store = dict(ci.ROW_STORE_DEFAULTS, enabled=True)
    cfg = {'buffers': {'store': store}, 'temperature_conditioning': tcond,
           'noised_prior': {'kick_kappa_per_atom': kappa, 'live_temperature_range': t_rng},
           'energy_config': {'log_temperature_range': list(rng)}, 'bwd_loss_coeffs': {'mle': 1.0, 'tb': 0.0}}
    stage = {'name': 'train_prior', 'fracs': {'fwd': 0.0, 'bwd': 1.0, 'replay': 0.0}, 'max_steps': 10,
             'loss_coeffs': {'bwd': {'mle': 1.0, 'tb': 0.0}}}
    monkeypatch.setattr(ci, 'active_stages', lambda c: [stage])
    return [m for sev, m in ci.row_store_problems(cfg) if sev == ci.ERROR and 'live_temperature_range' in m]


def test_the_range_inside_the_temperature_column_passes(monkeypatch):
    assert _problems(monkeypatch, [2.5, 10.0]) == []


@pytest.mark.parametrize('t_rng', [[2.0, 10.0], [2.5, 12.0]])
def test_a_range_outside_the_temperature_column_is_refused(monkeypatch, t_rng):
    assert any('outside' in m for m in _problems(monkeypatch, t_rng))


@pytest.mark.parametrize('tcond, kappa', [(False, None), (True, None)])
def test_the_range_without_labelled_kicks_is_a_dead_key(monkeypatch, tcond, kappa):
    assert any('dead key' in m for m in _problems(monkeypatch, [2.5, 10.0], tcond=tcond, kappa=kappa))
