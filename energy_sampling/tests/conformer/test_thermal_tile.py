"""The THERMAL anchor tile (energies/thermal_tile.py) and its two conformer-route consumers.

Load-bearing claims, each asserted directly because a failure of any returns plausible states:
  * the per-column widths are the force field's own: capped exactly at
    `thermal_rtheta_sigma` (and the prior's held / sibling widths), and for a bond length the
    curvature width agrees with it -- the factor-of-2 convention of sqrt(kT / 2k);
  * a CARRIER batch of mixed members takes each row's OWN member widths, and its pads stay
    exactly 0;
  * `tile: 'iso'` is bitwise the draw it was before the thermal tile existed;
  * at c = 1 the excess energy over a minimum averages k/2 kT (equipartition);
  * `prior_dataset_noise: thermal` re-bakes the rows it moves and hands the anchor seed the
    unnoised rows.
"""
import types

import numpy as np
import pytest
import torch

import energies.thermal_tile as tt
from conformer_modeller import ConformerModeller
from energies.conformer_carrier import carrier_pad_condition
from energies.conformer_data import (batch_states, collate_conditions, condition_from_energy,
                                     wrap_state)
from energies.conformer_torsions import ConformerTorsions
from energies.dof_features import free_dof_atom_index
from energies.multi_conformer import MultiConformerTorsions

KW = dict(device='cpu', level='full', force_field='mmff')
#: an alcohol, a nitrile (a transverse (u, v) pair) and a ring
SMIS = ['CCO', 'CC#N', 'C1CC1']


@pytest.fixture(scope='module', autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture(scope='module')
def multi(float64):
    en = MultiConformerTorsions(SMIS, identifiers=SMIS, **KW)
    assert en.is_carrier
    return en


def _carrier_batch(multi, per_member: int, mol_id: bool = True):
    """Rows of every member INTERLEAVED, at each member's reference state."""
    lay = multi.carrier
    conds = {}
    for ident, m in multi._members.items():
        c = condition_from_energy(m, identifier=ident)
        a, msk = free_dof_atom_index(m)
        conds[ident] = carrier_pad_condition(c, lay, ident, m, atoms=a, mask=msk, R=1)
    idents = [SMIS[i % len(SMIS)] for i in range(per_member * len(SMIS))]
    batch = collate_conditions([conds[i].__copy__() for i in idents])
    if mol_id:
        batch.add_graph_attr(torch.as_tensor([SMIS.index(i) for i in idents]), 'mol_id')
    multi.bind_identifier_registry({s: i for i, s in enumerate(SMIS)})
    X = torch.zeros(len(idents), lay.K)
    return batch, X, idents


# ------------------------------------------------------------------ widths

def _kinds(member):
    """Per member column: 'r', 'theta', 'uv', 'held', 'leader' or 'follower'."""
    sel = member._sel_rows.numpy()
    n_r, n_th = member.n_r, member.n_th
    held = set(member.held_phi_rows())
    leaders = {g[0] for g in member.torsion_groups()}
    out = []
    for c, r in enumerate(sel):
        if member._free_block[c] == 3:
            out.append('uv')
        elif r < n_r:
            out.append('r')
        elif r < n_r + n_th:
            out.append('theta')
        elif (r - n_r - n_th) in held:
            out.append('held')
        elif (r - n_r - n_th) in leaders:
            out.append('leader')
        else:
            out.append('follower')
    return np.array(out)


@pytest.mark.parametrize('smi', SMIS)
def test_the_cap_is_exactly_the_force_fields_own_width(monkeypatch, float64, smi):
    """With every curvature soft, each column takes its OWN-TERM width -- exactly
    `thermal_rtheta_sigma` for r and theta, the linear angle's theta width for u and v,
    `improper_phi_sigma` for held rows -- and every rotation the flat-rotor width."""
    m = ConformerTorsions(smiles=smi, **KW)
    monkeypatch.setattr(tt, 'curvatures', lambda member, d, s: np.zeros(d.shape[0]))
    sigma, rot = tt.member_widths(m, 1.0)
    scale = m._free_scale.numpy()
    s_r, s_th = m.thermal_rtheta_sigma(1.0)
    sel = m._sel_rows.numpy()
    kinds = _kinds(m)
    q = sigma * scale
    for c, kind in enumerate(kinds):
        row = int(sel[c])
        if kind == 'r':
            assert q[c] == pytest.approx(s_r[row], rel=1e-12)
        elif kind == 'theta':
            assert q[c] == pytest.approx(s_th[row - m.n_r], rel=1e-12)
        elif kind == 'held':
            assert q[c] == pytest.approx(m.improper_phi_sigma(1.0), rel=1e-12)
        elif kind == 'leader':
            assert q[c] == 0.0
        elif kind == 'uv':
            tv = np.flatnonzero(m.transverse_angles)
            assert q[c] == pytest.approx(s_th[tv[0]], rel=1e-12)
    # one rotation per group, each at the flat-rotor width, in radians
    assert rot.shape[0] == (kinds == 'leader').sum()
    for g in range(rot.shape[0]):
        nz = np.flatnonzero(rot[g])
        assert np.allclose(rot[g, nz] * scale[nz], tt.PHI_SIGMA_MAX)
    if smi == 'CC#N':
        assert (kinds == 'uv').sum() == 2


@pytest.mark.parametrize('smi', SMIS)
def test_curvature_widths_agree_with_the_bond_constant_and_never_exceed_the_cap(float64, smi):
    """The FACTOR CONVENTION: a bond-length column's curvature width is sqrt(kT / 2k) of its
    own term to within the small non-bonded contribution, so a factor-of-2 slip in either
    shows. Every other column is at most its own-term width, and every width is finite."""
    m = ConformerTorsions(smiles=smi, **KW)
    sigma, rot = tt.member_widths(m, 1.0)
    q = sigma * m._free_scale.numpy()
    s_r, s_th = m.thermal_rtheta_sigma(1.0)
    sel = m._sel_rows.numpy()
    kinds = _kinds(m)
    assert np.isfinite(q).all() and np.isfinite(rot).all()
    for c in np.flatnonzero(kinds == 'r'):
        assert 0.85 * s_r[sel[c]] <= q[c] <= s_r[sel[c]] * (1 + 1e-12)
    for c in np.flatnonzero(kinds == 'theta'):
        assert 0 < q[c] <= s_th[sel[c] - m.n_r] * (1 + 1e-12)
    assert (q[kinds != 'leader'] > 0).all()


def test_a_collective_chart_is_refused(float64):
    m = ConformerTorsions(smiles='CCCO', device='cpu', level='torsion', force_field='mmff')
    with pytest.raises(NotImplementedError, match='SELECTION'):
        tt.member_widths(m, 1.0)


# ------------------------------------------------------------------ carrier

def test_a_mixed_carrier_batch_takes_each_rows_own_widths_and_pads_stay_zero(multi):
    """Empirical per-column variance of each member's rows against THAT member's
    sigma^2 + sum_g rot^2, over 3000 rows per member; pads exactly 0 on every row."""
    torch.manual_seed(0)
    batch, X, idents = _carrier_batch(multi, 3000)
    tile = tt.ThermalTile(multi)
    d = tile.draw(batch, X, torch.ones(X.shape[0]))
    pads = ~batch.state_mask.reshape(X.shape).bool()
    assert pads.any() and torch.equal(d[pads], torch.zeros_like(d[pads]))
    ids = np.array(idents)
    want = {}
    for ident in SMIS:
        sigma, rot = tile.widths(ident)
        want[ident] = (sigma ** 2 + (rot ** 2).sum(0)).numpy()
        got = d[torch.as_tensor(ids == ident)].var(0).numpy()
        live = multi.carrier.valid(ident)
        np.testing.assert_allclose(got[live], want[ident][live], rtol=0.12)
    # the members genuinely differ on a shared column, so a swapped lookup would fail above
    shared = multi.carrier.valid('CCO') & multi.carrier.valid('CC#N')
    assert (np.abs(want['CCO'][shared] - want['CC#N'][shared])
            > 0.5 * want['CCO'][shared]).any()


# ------------------------------------------------------------------ the modeller seam

def _stub(multi, tile, **extra):
    m = ConformerModeller.__new__(ConformerModeller)
    ab = types.SimpleNamespace(tile=tile, thermal_noise_log_range=[0.0, 0.0],
                               seed_source='prior_dataset', seed_relax_steps=0)
    m.args = types.SimpleNamespace(buffers=types.SimpleNamespace(anchor_buffer=ab), **extra)
    m.energy_function = multi
    m.device = torch.device('cpu')
    return m


def _old_iso(multi, batch, noise_log_range):
    """`ConformerModeller._noise_and_condition`'s draw as it stood before the thermal tile."""
    log_min, log_max = float(noise_log_range[0]), float(noise_log_range[1])
    state = batch_states(batch)
    direction = torch.randn_like(state)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u = torch.rand(state.shape[0], device=state.device)
    magnitude = 10 ** (log_min + (log_max - log_min) * u)
    noised = state + direction * magnitude[:, None]
    lin = multi._lin_free_idx.to(noised.device)
    if lin.numel():
        noised[:, lin] = noised[:, lin].clip(min=-1, max=1)
    smask = batch.state_mask.reshape(noised.shape).bool()
    noised = torch.where(smask, noised, torch.zeros_like(noised))
    return wrap_state(noised, multi.periodic_dims)


def test_iso_is_bitwise_unchanged(multi):
    from energies.conformer_data import set_batch_states
    batch, X, _ = _carrier_batch(multi, 20)
    X = torch.rand_like(X) * 0.2 - 0.1
    X = X.masked_fill(~batch.state_mask.reshape(X.shape).bool(), 0.0)
    set_batch_states(batch, X, periodic=multi.periodic_dims)
    torch.manual_seed(7)
    want = _old_iso(multi, batch.clone(), [-1.0, 0.0])
    torch.manual_seed(7)
    got, *_ = ConformerModeller._noise_and_condition(_stub(multi, 'iso'), batch.clone(),
                                                     [-1.0, 0.0])
    assert torch.equal(batch_states(got), want)


def test_thermal_through_the_seam_clips_wraps_and_zeroes_pads(multi):
    torch.manual_seed(1)
    batch, X, _ = _carrier_batch(multi, 200)
    from energies.conformer_data import set_batch_states
    set_batch_states(batch, X, periodic=multi.periodic_dims)
    m = _stub(multi, 'thermal')
    m.args.buffers.anchor_buffer.thermal_noise_log_range = [1.0, 1.0]     # c = 10: forces wraps
    got, *_ = ConformerModeller._noise_and_condition(m, batch, [-1.0, 0.0])
    x = batch_states(got)
    pads = ~got.state_mask.reshape(x.shape).bool()
    assert torch.equal(x[pads], torch.zeros_like(x[pads]))
    lin = multi._lin_free_idx
    assert float(x[:, lin].abs().max()) <= 1.0
    per = torch.as_tensor(multi.periodic_dims)
    assert float(x[:, per].abs().max()) <= 1.0 and float(x[:, per].abs().max()) > 0.5
    assert not torch.equal(x, X)


def test_an_unknown_tile_is_refused(multi):
    batch, X, _ = _carrier_batch(multi, 2)
    from energies.conformer_data import set_batch_states
    set_batch_states(batch, X, periodic=multi.periodic_dims)
    with pytest.raises(ValueError, match="'iso' or 'thermal'"):
        ConformerModeller._noise_and_condition(_stub(multi, 'thermall'), batch, [-1.0, 0.0])


# ------------------------------------------------------------------ equipartition

@pytest.mark.parametrize('smi', ['CCCO', 'C1CCOC1'])
def test_excess_at_c1_is_k_over_2(float64, smi):
    """4000 draws at c = 1 about the member's minimum (its MMFF reference, relaxed further by
    descent): mean excess / (k/2) within 0.1 of 1, and the median within 15% of the
    Gamma(k/2) median. Measured 1.00 / 0.99 when written."""
    from scipy import stats

    from energies.prior_baselines import descend

    torch.manual_seed(0)
    en = ConformerTorsions(smiles=smi, **KW)
    one = torch.tensor(1.0)
    x0, _ = descend(en, torch.zeros(1, en.data_ndim), 200)
    x0 = x0.detach()
    tile = tt.ThermalTile(en)
    n = 4000
    x = x0 + tile.draw(None, x0.repeat(n, 1), torch.ones(n))
    x[:, en._lin_free_idx] = x[:, en._lin_free_idx].clip(-1, 1)
    x = wrap_state(x, en.periodic_dims)
    excess = (en.potential_energy(x, one) - en.potential_energy(x0, one)).numpy()
    k = en.data_ndim
    assert abs(excess.mean() / (k / 2) - 1.0) < 0.1
    assert abs(np.median(excess) / stats.gamma(a=k / 2).median() - 1.0) < 0.15


# ------------------------------------------------------------------ prior dataset intake

def _baked_batch(multi):
    from energies.conformer_data import set_batch_states
    batch, X, idents = _carrier_batch(multi, 30, mol_id=False)
    assert list(batch.identifier) == idents
    e = ConformerModeller._bake_rows(_stub(multi, 'thermal'), batch, X)
    return set_batch_states(batch, X, energies=e, periodic=multi.periodic_dims)


def test_prior_dataset_noise_none_returns_the_rows_untouched(multi):
    batch = _baked_batch(multi)
    m = _stub(multi, 'thermal', prior_dataset_noise='none')
    assert ConformerModeller._maybe_noise_prior_rows(m, batch) is batch
    assert not hasattr(m, '_prior_dataset_raw')


def test_prior_dataset_noise_thermal_rebakes_and_keeps_the_raw_rows(multi):
    torch.manual_seed(2)
    batch = _baked_batch(multi)
    raw_x = batch_states(batch).clone()
    m = _stub(multi, 'thermal', prior_dataset_noise='thermal')
    out = ConformerModeller._maybe_noise_prior_rows(m, batch)
    assert m._prior_dataset_raw is batch and torch.equal(batch_states(batch), raw_x)
    x = batch_states(out)
    assert not torch.equal(x, raw_x)
    pads = ~out.state_mask.reshape(x.shape).bool()
    assert torch.equal(x[pads], torch.zeros_like(x[pads]))
    # each row re-baked at its own member's potential (T = 1)
    lay = multi.carrier
    one = torch.tensor(1.0)
    for i, ident in enumerate(out.identifier):
        want = multi._members[ident].potential_energy(lay.from_carrier(ident, x[i:i + 1]), one)
        assert float(out.conformer_energy[i]) == pytest.approx(float(want), abs=1e-8)


def test_the_anchor_seed_reads_the_unnoised_rows(multi, monkeypatch):
    import train

    seen = {}
    monkeypatch.setattr(train.Modeller, 'init_anchor_buffer_seed',
                        lambda self: seen.setdefault('batch', self.prior_dataset.batch))
    m = _stub(multi, 'thermal', prior_dataset_noise='thermal')
    raw, noised = object(), types.SimpleNamespace(batch=object())
    m._prior_dataset_raw = raw
    m.prior_dataset = noised
    ConformerModeller.init_anchor_buffer_seed(m)
    assert seen['batch'] is raw
    assert m.prior_dataset is noised and not hasattr(m, '_prior_dataset_raw')


def test_an_unknown_prior_dataset_noise_is_refused(multi):
    m = _stub(multi, 'thermal', prior_dataset_noise='hot')
    with pytest.raises(ValueError, match='prior_dataset_noise'):
        ConformerModeller._maybe_noise_prior_rows(m, object())
