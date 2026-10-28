"""`double_bond_box_deg`: the dihedral about a LOCKED double bond as a bounded column.

With the stereo lock on, the prior holds the proper dihedral rows about a locked double bond
at the reference and the lock prices the other isomer, but the state column driving such a
row was periodic with a +/-180 degree box. ``ConformerTorsions(double_bond_box_deg=W)`` makes
it a bounded, non-periodic column: reference +/- W degrees at x = +/-1, the box wall outside,
block code 4, placed in the carrier's theta region. None, the default, changes nothing.

Each claim is checked against the default chart of the same molecule or against the member's
own energy, at level `full` (and `dihedral` where the tier matters), force field `mmff`,
`stereo_coeff` 300. The database test reads one shard of the QM9 conformer database and is
skipped where that file is absent (GFN_CONFORMER_DB_SHARD names another).
"""
import copy
import inspect
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from energies.conformer_carrier import (BOUNDED_DIHEDRAL, CarrierLayout, carrier_pad_condition,
                                        check_carrier_convention)
from energies.conformer_data import (batch_states, check_state_convention,
                                     condition_from_energy, wrap_state)
from energies.conformer_torsions import ConformerTorsions, rdkit_order_reference
from test_multi_energy_vectorized import _default_dtype, _Set

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

HERE = Path(__file__).resolve().parents[2]
W = 60.0
W_RAD = float(np.deg2rad(W))
LOCK = dict(device='cpu', force_field='mmff', stereo_coeff=300.0)
ENONE = 'C/C=C/C(C)=O'                 # acyclic, no linear centre: the prior density covers it
DB_SMIS = ['C/C=C/C(C)=O', 'CC/C=C\\CO', 'C/C(=N/O)C1CC1', 'C#C/C(=N/C)N(C)C=O']
MIXED = ['C/C=C/C(C)=O', 'CCCO', 'CC/C=C\\CO', 'C1CCOC1', 'C/C(=N/O)C1CC1', 'CO',
         'C#C/C(=N/C)N(C)C=O']
SIZES = [6, 5, 4, 3, 4, 1, 5]
SHARD = Path(os.environ.get('GFN_CONFORMER_DB_SHARD',
                            'D:/crystal_datasets/qm9_db_sep29/shard_0000_of_0400.pt'))
SHARD_IDENT = '[H]/N=C\\Nc1[nH]nnc1O'


def _member(smi=ENONE, box=W, level='full', **kw):
    with _default_dtype(torch.float64):
        return ConformerTorsions(smiles=smi, level=level, double_bond_box_deg=box,
                                 **{**LOCK, **kw})


def _plain(smi=ENONE, level='full', **kw):
    """The same member with the argument OMITTED, not passed as None."""
    with _default_dtype(torch.float64):
        return ConformerTorsions(smiles=smi, level=level, **{**LOCK, **kw})


def _lock_rows(en):
    """The rows `held_phi_rows` adds beyond the impropers: proper, about a locked bond."""
    return sorted(set(en.held_phi_rows()) - set(en.improper_phi_rows()))


def _cols_of_rows(en, rows):
    sel = en._sel_rows.numpy()
    n0 = en.n_r + en.n_th
    return np.array([int(np.flatnonzero(sel == n0 + j)[0]) for j in rows], dtype=np.int64)


def _states(en, n=32, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, en.data_ndim, generator=g, dtype=torch.float64) * 2 - 1) * scale


@pytest.fixture(scope='module')
def prior():
    p = Path(os.environ.get('GFN_CONFORMER_PRIOR', HERE / 'conformer_prior_v2.pt'))
    if not p.exists():
        pytest.skip(f'fitted InternalPrior not found at {p}')
    return torch.load(p, weights_only=False)


@pytest.fixture(scope='module')
def boxed():
    return _member()


@pytest.fixture(scope='module')
def plain():
    return _plain()


# ------------------------------------------------------------------ None is the old chart

@pytest.mark.parametrize('level', ['full', 'dihedral'])
def test_none_is_the_chart_without_the_argument(level, prior):
    a, b = _plain(level=level), _member(box=None, level=level)
    assert b.double_bond_box_deg is None and not b.double_bond_rows.any()
    assert np.array_equal(a._free_block, b._free_block) and not (b._free_block == 4).any()
    assert a.periodic_dims == b.periodic_dims
    assert torch.equal(a._free_scale, b._free_scale)
    assert a.log_chart_jacobian == b.log_chart_jacobian
    assert a.log_jacobian_const == b.log_jacobian_const and a.e_ref == b.e_ref
    assert b._rtheta_free == bool(b._lin_free_idx.numel())
    x = _states(a, scale=1.2)
    logT = torch.linspace(-0.3, 0.3, len(x), dtype=torch.float64)
    assert torch.equal(a.energy(x, None, logT), b.energy(x, None, logT))
    xa, sa = a.sample_prior_states(prior, 64, np.random.default_rng(0), report=False)
    xb, sb = b.sample_prior_states(prior, 64, np.random.default_rng(0), report=False)
    assert torch.equal(xa, xb) and np.array_equal(sa['dof'], sb['dof'])
    assert sorted(sb['clip_frac']) == ['r', 'theta', 'transverse']
    # the columns the box would bound are ordinary phi columns: periodic, one full turn
    cols = _cols_of_rows(b, _lock_rows(b))
    assert len(cols) == 2
    assert all(b.periodic_dims[c] for c in cols)
    assert torch.equal(b._free_scale[cols], torch.full((2,), np.pi, dtype=torch.float64))


def test_a_molecule_without_a_locked_double_bond_is_unchanged():
    a, b = _plain('CCCO'), _member('CCCO')
    assert b.double_bond_box_deg == W and not (b._free_block == 4).any()
    assert np.array_equal(a._free_block, b._free_block)
    assert a.log_chart_jacobian == b.log_chart_jacobian
    x = _states(a, scale=1.2)
    assert torch.equal(a.energy(x), b.energy(x))
    assert 'DOUBLE-BOND BOX: 0 column(s)' in b.describe()


# ------------------------------------------------------------------ refusals

@pytest.mark.parametrize('bad', [0.0, -30.0, 90.0, 120.0, float('nan'), float('inf')])
def test_a_box_outside_zero_to_ninety_is_refused(bad):
    with pytest.raises(ValueError, match=r'must lie in \(0, 90\)'):
        _member(box=bad)


@pytest.mark.parametrize('smi', [ENONE, 'CCCO'])
def test_the_box_needs_the_lock(smi):
    """Refused with the lock off for every molecule, with or without a double bond."""
    with pytest.raises(ValueError, match='with stereo_coeff 0'):
        _member(smi, stereo_coeff=0.0)


def test_the_argument_reaches_members_through_the_signature_filter():
    import build_conformer_references as bcr
    assert 'double_bond_box_deg' in inspect.signature(ConformerTorsions.__init__).parameters
    kw = bcr.member_kwargs({'level': 'full', 'double_bond_box_deg': W, 'temperature': 1.0})
    assert kw == {'level': 'full', 'double_bond_box_deg': W}


# ------------------------------------------------------------------ the column

@pytest.mark.parametrize('smi', DB_SMIS)
def test_the_column_is_bounded_and_not_periodic(smi):
    a, b = _plain(smi), _member(smi)
    rows = _lock_rows(b)
    cols = _cols_of_rows(b, rows)
    assert len(cols) >= 1
    assert np.array_equal(np.flatnonzero(b._free_block == 4), np.sort(cols))
    assert np.array_equal(np.flatnonzero(b.double_bond_rows), rows)
    # every other column keeps its code, and an improper row about the bond stays phi
    other = np.setdiff1d(np.arange(b.data_ndim), cols)
    assert np.array_equal(a._free_block[other], b._free_block[other])
    assert all(a._free_block[c] == 2 for c in cols)
    per = np.asarray(b.periodic_dims)
    assert not per[cols].any() and np.array_equal(per[other], np.asarray(a.periodic_dims)[other])
    assert set(cols.tolist()) <= set(b._lin_free_idx.tolist())
    assert torch.allclose(b._free_scale[cols], torch.full((len(cols),), W_RAD,
                                                          dtype=torch.float64), rtol=0, atol=1e-15)
    assert torch.equal(b._free_scale[other], a._free_scale[other])
    # log|dq/dx| moves by log(W / 180) per bounded column, and by nothing else
    assert b.log_chart_jacobian == pytest.approx(
        a.log_chart_jacobian + len(cols) * np.log(W / 180.0), abs=1e-12)
    assert f'DOUBLE-BOND BOX: {len(cols)} column(s)' in b.describe()
    assert f'+/-{W:g} deg' in b.describe()


def test_the_edges_are_sixty_degrees_from_the_reference(boxed):
    from mxtaltools.conformers.geometry import dihedral
    rows = _lock_rows(boxed)
    cols = _cols_of_rows(boxed, rows)
    ti = np.asarray(boxed.spec.torsion_index)
    n0 = boxed.n_r + boxed.n_th
    wrap = lambda d: (d + np.pi) % (2 * np.pi) - np.pi
    for j, c in zip(rows, cols):
        x = torch.zeros(3, boxed.data_ndim, dtype=torch.float64)
        x[0, c], x[2, c] = -1.0, 1.0
        _, _, ph = boxed.dof_from_state(x)
        ref = float(boxed._ref_dof[n0 + j])
        want = torch.tensor([ref - W_RAD, ref, ref + W_RAD], dtype=torch.float64)
        assert torch.allclose(ph[:, j], want, atol=1e-14)
        # and on the built geometry: the dihedral itself turns by 60 degrees each way
        pos = boxed.build_positions(x).reshape(3, -1, 3)
        got = dihedral(*(pos[:, int(a)] for a in ti[j])).numpy()
        assert np.degrees(wrap(got - got[1])) == pytest.approx([-W, 0.0, W], abs=1e-8)


def test_the_wall_is_zero_inside_and_positive_outside(boxed, plain):
    cols = _cols_of_rows(boxed, _lock_rows(boxed))
    one = torch.tensor(1.0, dtype=torch.float64)
    x = _states(boxed, n=64)                       # every column inside [-1, 1]
    x[0, cols], x[1, cols] = 1.0, -1.0             # the edges are inside
    assert torch.equal(boxed.bounding_energy(x, one), torch.zeros(64, dtype=torch.float64))
    for v in (1.25, -1.25):
        y = torch.zeros(1, boxed.data_ndim, dtype=torch.float64)
        y[0, cols[0]] = v
        assert float(boxed.bounding_energy(y, one)) == pytest.approx(
            boxed.bounding_coeff * 0.25 ** 2, abs=1e-12)
        # ...where the default chart, which wraps this column, has none
        assert float(plain.bounding_energy(y, one)) == 0.0
    # inside the box the two charts are one potential at one geometry; the energies differ
    # by the chart constant alone
    xp = x.clone()
    xp[:, cols] = x[:, cols] * (W / 180.0)
    assert torch.allclose(boxed.potential_energy(x, one), plain.potential_energy(xp, one),
                          rtol=1e-12, atol=1e-9)
    gap = boxed.energy(x) - plain.energy(xp)
    assert torch.allclose(gap, torch.full_like(gap, plain.log_chart_jacobian
                                               - boxed.log_chart_jacobian), atol=1e-9)


@pytest.mark.parametrize('smi', DB_SMIS[:3])
def test_the_state_round_trips(smi):
    en = _member(smi)
    cols = _cols_of_rows(en, _lock_rows(en))
    per = torch.as_tensor(en.periodic_dims)
    x = _states(en, n=64, seed=3)
    x[:8, cols] = torch.linspace(-1.4, 1.4, 8, dtype=torch.float64)[:, None]   # past the box
    r, th, ph = en.dof_from_state(x)
    back = en.state_from_dof(r, th, ph)
    assert torch.allclose(back, x, atol=1e-12)
    # a measured dihedral arrives in (-pi, pi]: the same state whichever branch it is on
    shifted = en.state_from_dof(r, th, (ph + np.pi) % (2 * np.pi) - np.pi)
    assert torch.allclose(shifted, x, atol=1e-12)
    # not folded: a bounded column past its box stays past it, a phi column wraps
    assert float(back[:, cols].abs().max()) == pytest.approx(1.4, abs=1e-12)
    assert torch.equal(wrap_state(back, en.periodic_dims)[:, ~per], back[:, ~per])


def test_dihedral_tier_walls_the_column_and_keeps_the_frozen_blocks_frozen():
    a, b = _plain(level='dihedral'), _member(level='dihedral')
    cols = np.flatnonzero(b._free_block == 4)
    assert len(cols) == 2 and b._lin_free_idx.tolist() == cols.tolist()
    # r and theta are still the frozen reference: no clamp, log J still a constant
    assert not b._rtheta_free
    assert b.log_jacobian_const == a.log_jacobian_const is not None
    c = condition_from_energy(b)
    assert not bool(c.ctree_clamp) and check_state_convention(c, b) < 1e-9
    one = torch.tensor(1.0, dtype=torch.float64)
    y = torch.zeros(1, b.data_ndim, dtype=torch.float64)
    y[0, cols[0]] = 1.5
    assert float(b.bounding_energy(y, one)) == pytest.approx(b.bounding_coeff * 0.25)


# ------------------------------------------------------------------ graph and carrier

@pytest.fixture(scope='module')
def mixed():
    return _Set(MIXED, SIZES, torch.float64, stereo_coeff=300.0, double_bond_box_deg=W)


def test_the_graph_carries_the_scale(boxed):
    c = condition_from_energy(boxed)
    rows = _lock_rows(boxed)
    scale = c.ctree_ph_scale[3:]                   # one phi row per atom of rank >= 3
    assert torch.allclose(scale[rows], torch.full((len(rows),), W_RAD, dtype=torch.float64))
    rest = np.setdiff1d(np.arange(boxed.n_ph), rows)
    assert torch.equal(scale[rest], torch.full((len(rest),), np.pi, dtype=torch.float64))
    assert check_state_convention(c, boxed) < 1e-9


def test_the_carrier_places_the_column_in_the_theta_region(mixed):
    multi, lay = mixed.multi, mixed.lay
    assert multi.is_carrier and not lay.is_identity
    assert all(m.double_bond_box_deg == W for m in multi._members.values())
    per = np.asarray(multi.periodic_dims)
    assert np.array_equal(per, lay.free_block == 2)
    n_db = 0
    for ident, m in multi._members.items():
        mine = np.flatnonzero(m._free_block == 4)
        carried = lay.cols[ident][mine]
        n_db += len(mine)
        assert (lay.free_block[carried] == 1).all() and not per[carried].any()
        assert (lay.kind(ident)[carried] == BOUNDED_DIHEDRAL).all()
        assert set(carried.tolist()) <= set(multi._lin_free_idx.tolist())
        c = carrier_pad_condition(condition_from_energy(m, identifier=ident), lay, ident, m)
        assert check_carrier_convention(c, lay, ident, m) < 1e-9
    assert n_db == 6                               # 2 + 2 + 1 + 1 over the four E/Z members
    # the theta region is wider, and the phi region narrower, than the unboxed layout's
    with _default_dtype(torch.float64):
        old = CarrierLayout({i: _plain(i) for i in MIXED})
    assert lay.block_width[0] == old.block_width[0]
    assert lay.block_width[1] > old.block_width[1] and lay.block_width[2] < old.block_width[2]
    assert 'bounded double-bond dihedral' in lay.describe()


def test_the_one_pass_energy_is_each_members_own(mixed):
    multi = mixed.multi
    lin = multi._lin_free_idx
    assert bool((mixed.X.index_select(1, lin).abs() > 1).any()), 'the wall must be live'
    got = multi.energy(mixed.X, mixed.batch, mixed.logT)
    want = mixed.oracle()
    assert torch.allclose(got, want, rtol=1e-12, atol=1e-9), float((got - want).abs().max())
    assert torch.equal(multi._energy_per_member(mixed.X, mixed.batch, mixed.logT), want)
    # a state on a bounded column past its box pays the wall in the one-pass energy too
    ident = MIXED[0]
    row = int(np.flatnonzero(mixed.assign == 0)[0])
    m = multi._members[ident]
    col = int(mixed.lay.cols[ident][np.flatnonzero(m._free_block == 4)[0]])
    X = torch.zeros_like(mixed.X)
    X2 = X.clone()
    X2[row, col] = 1.5
    zero = torch.zeros(len(X), dtype=torch.float64)
    e0 = multi.energy(X, mixed.batch, zero, return_exp=True)[1].conformer_energy.flatten()
    e1 = multi.energy(X2, mixed.batch, zero, return_exp=True)[1].conformer_energy.flatten()
    x_m = torch.zeros(2, m.data_ndim, dtype=torch.float64)
    x_m[1, np.flatnonzero(m._free_block == 4)[0]] = 1.5
    one = torch.tensor(1.0, dtype=torch.float64)
    u = m.potential_energy(x_m, one)
    assert float(e1[row] - e0[row]) == pytest.approx(float(u[1] - u[0]), abs=1e-9)
    assert float(m.bounding_energy(x_m, one)[1]) == pytest.approx(m.bounding_coeff * 0.25)


def test_pads_stay_zero_and_the_bounded_column_is_not_wrapped(mixed):
    multi = mixed.multi
    _, out = multi.energy(mixed.X, mixed.batch, mixed.logT, return_exp=True)
    stored = batch_states(out).reshape(mixed.X.shape)
    pads = mixed.pad_mask
    assert pads.any() and torch.equal(stored[pads], torch.zeros_like(stored[pads]))
    per = torch.as_tensor(multi.periodic_dims)
    # what the write path stores: phi wrapped, every other column -- bounded ones included --
    # exactly as proposed, even past the box
    assert torch.equal(stored[:, ~per], mixed.X[:, ~per])
    assert bool((stored[:, per].abs() <= 1.0).all())
    past = 0
    for j, ident in enumerate(MIXED):
        m = multi._members[ident]
        cols = mixed.lay.cols[ident][np.flatnonzero(m._free_block == 4)]
        rows = np.flatnonzero(mixed.assign == j)
        if len(cols):
            past += int((stored[rows][:, cols].abs() > 1.0).sum())
    assert past > 0, 'some bounded column must sit past its box for the claim to bite'
    assert torch.allclose(out.conformer_energy.flatten().double(), mixed.potential_oracle(),
                          rtol=1e-12, atol=1e-9)


# ------------------------------------------------------------------ prior, tile, features

def test_prior_draws_land_inside_the_box_at_the_held_width(boxed, plain, prior):
    n = 6000
    x, stats = boxed.sample_prior_states(prior, n, np.random.default_rng(5), report=False)
    cols = _cols_of_rows(boxed, _lock_rows(boxed))
    assert stats['n_held_bond'] == len(cols) == 2
    assert stats['clip_frac']['double_bond'] == 0.0
    assert float(x[:, cols].abs().max()) < 1.0
    s_imp = boxed.improper_phi_sigma(float(boxed.temperature))
    sd = x[:, cols].std(0).numpy()
    assert sd == pytest.approx(s_imp / W_RAD, rel=0.05)
    # the same draws, column for column, as the default chart's -- its phi column rescaled
    xp, sp = plain.sample_prior_states(prior, n, np.random.default_rng(5), report=False)
    assert np.array_equal(stats['dof'], sp['dof'])
    other = np.setdiff1d(np.arange(boxed.data_ndim), cols)
    assert torch.equal(x[:, other], xp[:, other])
    assert torch.allclose(x[:, cols], xp[:, cols] * (180.0 / W), atol=1e-12)


def test_prior_log_prob_is_the_draw_density_under_the_new_scale(boxed, plain, prior):
    """Change of variables, checked against the draws themselves.

    `prior_log_prob` is a density over the DoF (radians). On the bounded column x = (phi -
    phi0) / s with s the column's scale, so the state density is s q(phi0 + s x). The row's
    marginal q is read off `prior_log_prob` by moving that row alone and normalising over the
    circle; its histogram prediction must match the drawn states with s = W, and must NOT
    match with s = pi, the scale the column had before.
    """
    n = 20000
    x, stats = boxed.sample_prior_states(prior, n, np.random.default_rng(11), report=False)
    rows = _lock_rows(boxed)
    cols = _cols_of_rows(boxed, rows)
    n0 = boxed.n_r + boxed.n_th
    # the density is over the DoF and does not read the chart's scale
    assert np.array_equal(boxed.prior_log_prob(prior, stats['dof'][:64]),
                          plain.prior_log_prob(prior, stats['dof'][:64]))
    base = stats['dof'][:1].copy()
    for j, c in zip(rows, cols):
        ref = float(boxed.ph0[j])
        grid = np.linspace(-np.pi, np.pi, 7201)[:-1]
        dof = np.repeat(base, len(grid), axis=0)
        dof[:, n0 + j] = ref + grid
        lp = boxed.prior_log_prob(prior, dof)
        q = np.exp(lp - lp.max())
        q /= q.sum() * (grid[1] - grid[0])                     # the row's marginal, per radian
        edges = np.linspace(-0.4, 0.4, 17)
        hist = np.histogram(x[:, c].numpy(), bins=edges)[0] / n

        def predicted(scale):
            lo, hi = edges[:-1] * scale, edges[1:] * scale     # the bins, in radians
            cdf = np.concatenate([[0.0], np.cumsum(q) * (grid[1] - grid[0])])
            at = lambda v: np.interp(v, np.concatenate([grid, [np.pi]]), cdf)
            return at(hi) - at(lo)

        want = predicted(float(boxed._free_scale[c]))
        assert float(boxed._free_scale[c]) == pytest.approx(W_RAD)
        noise = np.sqrt(np.maximum(want, 1e-6) / n)
        assert np.abs(hist - want).max() < 5 * noise.max() + 2e-3, (hist, want)
        assert hist.sum() > 0.99 and want.sum() == pytest.approx(hist.sum(), abs=0.01)
        assert np.abs(hist - predicted(np.pi)).max() > 0.1     # the old scale is refuted


def test_thermal_tile_width_follows_the_scale(boxed, plain):
    from energies.thermal_tile import member_widths
    cols = _cols_of_rows(boxed, _lock_rows(boxed))
    sb, rb = member_widths(boxed, float(boxed.temperature))
    sp, rp = member_widths(plain, float(plain.temperature))
    other = np.setdiff1d(np.arange(boxed.data_ndim), cols)
    assert sb[cols] == pytest.approx(sp[cols] * (180.0 / W), rel=1e-6)
    assert sb[other] == pytest.approx(sp[other], rel=1e-9, abs=1e-12)
    assert rb == pytest.approx(rp, rel=1e-9, abs=1e-12)
    # capped at the held-row width the prior draws at, in the column's own units
    s_imp = boxed.improper_phi_sigma(float(boxed.temperature))
    assert (sb[cols] <= s_imp / W_RAD + 1e-12).all() and (sb[cols] > 0).all()


def test_the_held_bond_feature_marks_exactly_the_bounded_columns(boxed):
    from energies.dof_features import state_feature_names, state_features
    f = state_features(boxed)
    held = f[:, state_feature_names().index('is_held_bond')]
    assert np.array_equal(held == 1.0, boxed._free_block == 4)


def test_descend_clamps_the_bounded_column(boxed):
    from energies.prior_baselines import descend
    cols = _cols_of_rows(boxed, _lock_rows(boxed))
    x0 = torch.zeros(4, boxed.data_ndim, dtype=torch.float64)
    x0[:, cols[0]] = torch.tensor([0.999, -0.999, 0.5, 0.0], dtype=torch.float64)
    bx, _ = descend(boxed, x0, steps=6, lr=0.2)
    assert float(bx[:, cols].abs().max()) <= 1.0


# ------------------------------------------------------------------ stamps and the database

def test_an_absent_key_is_none_in_every_stamp():
    import build_conformer_database as db
    import build_conformer_references as bcr
    old = {'level': 'full', 'force_field': 'mmff', 'stereo_coeff': 300.0}
    assert bcr.defining_energy(bcr.member_kwargs(old)) == \
        bcr.defining_energy(bcr.member_kwargs({**old, 'double_bond_box_deg': None}))
    assert bcr.defining_energy(bcr.member_kwargs(old))['double_bond_box_deg'] is None
    # a database stores geometries, which the box does not change: built without the box
    # it serves a consumer with it, and still refuses any other member argument
    info = {'path': 'db', 'energy_kwargs': old}
    db.refuse_other_member_kwargs(info, {**old, 'double_bond_box_deg': W}, 'test')
    with pytest.raises(SystemExit, match='delta_r_max'):
        db.refuse_other_member_kwargs(info, {**old, 'delta_r_max': 0.2}, 'test')


def test_a_reference_table_stamped_before_the_argument_still_loads(tmp_path):
    """`load_references`: a stamp without the key was built under the default, None. A run
    WITH the box is another chart for the table's stored states and is refused."""
    import build_conformer_references as bcr
    kw = {'level': 'full', 'force_field': 'mmff', 'stereo_coeff': 300.0}
    cond = tmp_path / 'conditions.pt'
    cond.write_bytes(b'conditions')
    stamp_energy = bcr.defining_energy(kw)
    del stamp_energy['double_bond_box_deg']              # as a table written before it existed
    table = tmp_path / 'refs.pt'
    torch.save({'stamp': {'format': bcr.FORMAT, 'energy': stamp_energy,
                          'conditions': {'path': str(cond), 'sha256': bcr.file_sha256(cond),
                                         'identifiers': []}},
                'entries': {}, 'failures': {}}, table)
    bcr.load_references(table, conditions_path=cond, energy_kwargs=kw)
    bcr.load_references(table, conditions_path=cond,
                        energy_kwargs={**kw, 'double_bond_box_deg': None})
    with pytest.raises(bcr.StaleReferencesError, match='double_bond_box_deg'):
        bcr.load_references(table, conditions_path=cond,
                            energy_kwargs={**kw, 'double_bond_box_deg': W})


def test_a_stored_database_row_measures_into_the_boxed_chart():
    """A real shard: rows stored by a build WITHOUT the box, matched by a member WITH it."""
    import build_conformer_database as db
    from conformer_modeller import ConformerModeller
    if not SHARD.exists():
        pytest.skip(f'conformer database shard not found at {SHARD}')
    blob = torch.load(SHARD, weights_only=False, map_location='cpu')
    ekw = dict(blob['header']['config']['energy_kwargs'])
    assert 'double_bond_box_deg' not in ekw and float(ekw['stereo_coeff']) > 0
    rec = next(c for r in blob['keys'].values() for c in r['conditions']
               if c['identifier'] == SHARD_IDENT)
    rec = {k: (rec.get(k) if k in db.OPTIONAL_FIELDS else rec[k]) for k in db.READ_FIELDS}
    members = {}
    with _default_dtype(torch.float64):
        for box in (None, W):
            pos, _ = rdkit_order_reference(SHARD_IDENT, rec['ref_pos'], level=ekw['level'],
                                           perm=rec['perm'], z=rec['z'])
            members[box] = ConformerTorsions(smiles=SHARD_IDENT, reference_positions=pos,
                                             double_bond_box_deg=box, **ekw)
    en, old = members[W], members[None]
    cols = np.flatnonzero(en._free_block == 4)
    assert len(cols) >= 1
    cap = 8
    x, stored, info = db.match_rows(en, SHARD_IDENT, rec, cap, 1e-6)
    assert info['code'] is None, info
    assert info['rescore_gap'] <= 1e-6 and info['ref_pos_gap'] == 0.0
    assert len(stored) == info['rows'] == min(cap, len(rec['basins']['energy'])) > 1
    assert float(x[:, cols].abs().max()) <= 1.0
    # the same rows in the chart they were built in: every column equal but the bounded one,
    # which holds the same angle in units of W instead of 180 degrees
    x0, stored0, info0 = db.match_rows(old, SHARD_IDENT, rec, cap, 1e-6)
    assert info0['code'] is None and np.array_equal(stored, stored0)
    other = np.setdiff1d(np.arange(en.data_ndim), cols)
    assert torch.equal(x[:, other], x0[:, other])
    assert torch.allclose(x[:, cols], x0[:, cols] * (180.0 / W), atol=1e-12)
    assert float(x[:, cols].abs().max()) > 0.05, 'the rows must move the bounded column'
    # the database signature does not read the box; the run's condition-set digest does
    assert db.member_signature(SHARD_IDENT, en) == rec['signature']
    assert ConformerModeller._member_signature(SHARD_IDENT, en) != rec['signature']
    assert ConformerModeller._member_signature(SHARD_IDENT, old) == rec['signature']
    # a row past the box is refused, by name
    wide = copy.deepcopy(rec)
    turned = old.build_positions(torch.as_tensor(
        np.where(np.arange(old.data_ndim) == cols[0], 80.0 / 180.0, 0.0)[None],
        dtype=torch.float64)).reshape(1, -1, 3).numpy()
    wide['basins'] = {**rec['basins'], 'pos': turned, 'energy': np.zeros(1)}
    assert db.match_rows(en, SHARD_IDENT, wide, 1, 1e9)[2]['code'] == 'db_outside_box'
