"""`sibling_offset_box_deg`: a sibling group as ONE rotation plus bounded OFFSETS.

A sibling group is the dihedral rows that place one atom's children on one frame
(``ConformerTorsions.torsion_groups``). In the default chart each row has its own periodic
column, so the group's rigid rotation about the bond (soft) and the spacing between two
members (stiff) are mixtures of the same columns. ``ConformerTorsions(sibling_offset_box_deg=W)``
re-expresses each ELIGIBLE group (``sibling_offset_census``): the leader's column, still
periodic, drives every row of the group, and each follower's column is its dihedral minus the
leader's, as a displacement from the reference: bounded to +/- W degrees, non-periodic, block
code 5, placed in the carrier's theta region. None, the default, changes nothing.

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

from energies.conformer_carrier import (SIBLING_OFFSET, CarrierLayout, carrier_pad_condition,
                                        check_carrier_convention)
from energies.conformer_data import (SIBLING_LEAD_FIELDS, batch_states, check_state_convention,
                                     condition_from_energy, wrap_state)
from energies.conformer_torsions import ConformerTorsions, rdkit_order_reference
from test_multi_energy_vectorized import _default_dtype, _Set

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

HERE = Path(__file__).resolve().parents[2]
W = 60.0
W_RAD = float(np.deg2rad(W))
LOCK = dict(device='cpu', force_field='mmff', stereo_coeff=300.0)
ACYC = 'CCCO'                         # acyclic, no free invertible centre: two CH2/CH3 groups
AMINE = 'CC(C)N'                      # two methyl groups, and an amine N the prior flips
# molecules with a converted group, between them: an amine and an amide N (kept periodic),
# a ring (its groups kept), a locked double bond (its rows held), a stereocentre
SMIS = [ACYC, AMINE, 'CC(=O)N(C)C', 'C[C@H]1CCCO1', 'C/C=C/C(C)=O', 'CC[C@H](C)O', 'CCCCC1CC1']
MIXED = ['CCCO', 'CC(C)N', 'CO', 'C[C@H]1CCCO1', 'C/C=C/C(C)=O', 'CC#CC', 'CC(=O)N(C)C']
SIZES = [6, 5, 2, 4, 4, 3, 5]
SHARD = Path(os.environ.get('GFN_CONFORMER_DB_SHARD',
                            'D:/crystal_datasets/qm9_db_sep29/shard_0000_of_0400.pt'))
SHARD_IDENT = 'CC[C@@H](C(=O)[O-])[C@@H](C)[NH3+]'


def _member(smi=ACYC, box=W, level='full', **kw):
    with _default_dtype(torch.float64):
        return ConformerTorsions(smiles=smi, level=level, sibling_offset_box_deg=box,
                                 **{**LOCK, **kw})


def _plain(smi=ACYC, level='full', **kw):
    """The same member with the argument OMITTED, not passed as None."""
    with _default_dtype(torch.float64):
        return ConformerTorsions(smiles=smi, level=level, **{**LOCK, **kw})


def _cols_of_rows(en, rows):
    """The OWN state column of each torsion row."""
    sel = en._sel_rows.numpy()
    n0 = en.n_r + en.n_th
    return np.array([int(np.flatnonzero(sel == n0 + j)[0]) for j in rows], dtype=np.int64)


def _offset_cols(en):
    return np.flatnonzero(en._free_block == 5)


def _states(en, n=32, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, en.data_ndim, generator=g, dtype=torch.float64) * 2 - 1) * scale


def _to_new(old, new, x):
    """A state of the default chart, as the state of the offset chart at the same geometry."""
    return new.state_from_dof(*old.dof_from_state(x))


def _dihedrals(en, x):
    """``[B, n_ph]`` the dihedral of every torsion row on the BUILT positions."""
    from mxtaltools.conformers.geometry import dihedral
    pos = en.build_positions(x).reshape(len(x), -1, 3)
    ti = np.asarray(en.spec.torsion_index)
    return torch.stack([dihedral(*(pos[:, int(a)] for a in row)) for row in ti], dim=1)


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


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

@pytest.mark.parametrize('smi,level', [(ACYC, 'full'), (ACYC, 'dihedral'), (AMINE, 'full'),
                                       ('C[C@H]1CCCO1', 'full')])
def test_none_is_the_chart_without_the_argument(smi, level, prior):
    a, b = _plain(smi, level=level), _member(smi, box=None, level=level)
    assert b.sibling_offset_box_deg is None and not b.sibling_offset_rows.any()
    assert b.sibling_offset_groups == [] and int(b._so_cols.numel()) == 0
    assert any(len(g['rows']) > 1 for g in b.sibling_offset_census()), 'needs a sibling group'
    assert np.array_equal(a._free_block, b._free_block) and not (b._free_block == 5).any()
    assert a.periodic_dims == b.periodic_dims
    assert torch.equal(a._M, b._M) and torch.equal(a._sel_rows, b._sel_rows)
    assert torch.equal(a._free_scale, b._free_scale)
    assert torch.equal(a._lin_free_idx, b._lin_free_idx)
    assert a.log_chart_jacobian == b.log_chart_jacobian
    assert a.log_jacobian_const == b.log_jacobian_const and a.e_ref == b.e_ref
    assert not a.collective and not b.collective
    x = _states(a, scale=1.2)
    logT = torch.linspace(-0.3, 0.3, len(x), dtype=torch.float64)
    assert torch.equal(a.energy(x, None, logT), b.energy(x, None, logT))
    one = torch.tensor(1.0, dtype=torch.float64)
    assert torch.equal(a.potential_energy(x, one), b.potential_energy(x, one))
    assert torch.equal(a.build_positions(x), b.build_positions(x))
    xa, sa = a.sample_prior_states(prior, 64, np.random.default_rng(0), report=False)
    xb, sb = b.sample_prior_states(prior, 64, np.random.default_rng(0), report=False)
    assert torch.equal(xa, xb) and np.array_equal(sa['dof'], sb['dof'])
    assert sorted(sb['clip_frac']) == ['r', 'theta', 'transverse']
    # the condition graph: the same fields, bit for bit, and no second-column pair
    ca, cb = condition_from_energy(a), condition_from_energy(b)
    assert sorted(ca._store.keys()) == sorted(cb._store.keys())
    assert not any(f in cb._store.keys() for f in SIBLING_LEAD_FIELDS)
    for key in ca._store.keys():
        va, vb = ca[key], cb[key]
        assert (torch.equal(va, vb) if torch.is_tensor(va) else va == vb), key
    # the thermal tile and the per-column features read the chart, and agree too
    from energies.dof_features import free_dof_atom_index, state_features
    from energies.thermal_tile import member_widths
    for u, v in zip(member_widths(a, float(a.temperature)),
                    member_widths(b, float(b.temperature))):
        assert np.array_equal(u, v)
    assert np.array_equal(state_features(a), state_features(b))
    for u, v in zip(free_dof_atom_index(a), free_dof_atom_index(b)):
        assert np.array_equal(u, v)


def test_a_molecule_without_an_eligible_group_is_unchanged():
    """2-butyne's one sibling group of several rows is a methyl on the alkyne axis, measured
    against a dummy frame: not eligible, so the chart is the default's."""
    a, b = _plain('CC#CC'), _member('CC#CC')
    assert b.sibling_offset_box_deg == W and not (b._free_block == 5).any()
    assert [g['reason'] for g in b.sibling_offset_census() if len(g['rows']) > 1] == \
        ['linear_frame']
    assert np.array_equal(a._free_block, b._free_block) and torch.equal(a._M, b._M)
    assert a.log_chart_jacobian == b.log_chart_jacobian
    x = _states(a, scale=1.2)
    assert torch.equal(a.energy(x), b.energy(x))
    assert 'SIBLING OFFSETS: 0 of 1 sibling group(s)' in b.describe()
    # its graph still carries the (empty) second-column pair, so a set's graphs collate
    c = condition_from_energy(b)
    assert bool((c.ctree_ph_lead_col == -1).all()) and bool((c.ctree_ph_lead_scale == 0).all())


# ------------------------------------------------------------------ refusals

@pytest.mark.parametrize('bad', [0.0, -30.0, 180.0, 240.0, float('nan'), float('inf')])
def test_a_box_outside_zero_to_one_eighty_is_refused(bad):
    with pytest.raises(ValueError, match=r'must lie in \(0, 180\)'):
        _member(box=bad)


@pytest.mark.parametrize('smi', [ACYC, 'CC#CC'])
def test_the_box_needs_the_lock(smi):
    """Refused with the lock off for every molecule, with or without an eligible group."""
    with pytest.raises(ValueError, match='with stereo_coeff 0'):
        _member(smi, stereo_coeff=0.0)


def test_the_argument_reaches_members_through_the_signature_filter():
    import build_conformer_references as bcr
    assert 'sibling_offset_box_deg' in inspect.signature(ConformerTorsions.__init__).parameters
    kw = bcr.member_kwargs({'level': 'full', 'sibling_offset_box_deg': W, 'temperature': 1.0})
    assert kw == {'level': 'full', 'sibling_offset_box_deg': W}


def test_the_canonical_config_declares_it_off():
    import yaml
    with open(HERE / 'configs' / 'conformer_mk.yaml', encoding='utf-8') as f:
        ec = yaml.safe_load(f)['energy_config']
    assert 'sibling_offset_box_deg' in ec and ec['sibling_offset_box_deg'] is None


# ------------------------------------------------------------------ which groups

@pytest.mark.parametrize('smi', SMIS)
def test_the_eligibility_rule(smi, prior):
    """Converted: two or more rows, the centre off every ring and named by the lock as a
    four-neighbour centre, no transverse, dummy-frame or collinear-held row. Nothing else."""
    from energies.invertible_centres import LOCKED, centre_table, invertible_centres
    from energies.stereo_lock import TETRAHEDRAL
    a, b = _plain(smi), _member(smi)
    n0 = b.n_r + b.n_th
    ti = np.asarray(b.spec.torsion_index)
    named = {int(k) for k, kd in zip(b.stereo.key, b.stereo.kind) if int(kd) == TETRAHEDRAL}
    nbr = {}
    for u, v in np.asarray(b.bond_index_slot).reshape(2, -1).T:
        nbr.setdefault(int(u), set()).add(int(v))
        nbr.setdefault(int(v), set()).add(int(u))
    cen = b.sibling_offset_census()
    assert [g['rows'] for g in cen] == b.torsion_groups() == a.torsion_groups()
    assert cen == a.sibling_offset_census(), 'the census does not read the option'
    conv = [g['rows'] for g in cen if not g['reason']]
    assert conv == b.sibling_offset_groups and len(conv) >= 1
    for g in cen:
        c = g['centre']
        assert all(int(ti[j, 2]) == c for j in g['rows'])
        if not g['reason']:
            assert len(g['rows']) >= 2 and c in named and len(nbr[c]) == 4
            assert not b.atom_in_ring[c]
            assert not b.dummy_frame_rows[g['rows']].any()
    # the rows: followers are block 5 and read two columns, everything else is as it was
    followers = sorted(j for g in conv for j in g[1:])
    leaders = [g[0] for g in conv]
    assert np.array_equal(np.flatnonzero(b.sibling_offset_rows), followers)
    off = _cols_of_rows(b, followers)
    assert np.array_equal(np.sort(off), _offset_cols(b))
    other = np.setdiff1d(np.arange(b.data_ndim), off)
    assert np.array_equal(a._free_block[other], b._free_block[other])
    assert (a._free_block[off] == 2).all() and np.array_equal(a._sel_rows, b._sel_rows)
    m_new, m_old = b._M.numpy(), a._M.numpy()
    extra = np.argwhere(m_new != m_old)
    driven = b._driven_idx.numpy()
    lead_col = dict(zip(leaders, _cols_of_rows(b, leaders)))
    want = sorted((int(np.flatnonzero(driven == n0 + f)[0]), int(lead_col[g[0]]))
                  for g in conv for f in g[1:])
    assert sorted(map(tuple, extra.tolist())) == want and (m_new[tuple(extra.T)] == 1).all()
    # HELD rows (impropers, a locked double bond's rows) are in no group and keep their code
    held = b.held_phi_rows()
    assert not set(held) & {j for g in conv for j in g}
    assert np.array_equal(a._free_block[_cols_of_rows(a, held)],
                          b._free_block[_cols_of_rows(b, held)])
    # RING rows: the blocks, the rows placing a ring atom, a ring's rotation rows and the
    # substituents hung off a ring frame are all left alone
    touched = {j for g in conv for j in g}
    ring = set()
    blocks = b.ring_blocks(prior)
    for (order, _, extra_rows), info in zip(blocks, b.ring_block_info):
        ring |= {j for k, j in order if k == 'phi'} | {j for k, j in extra_rows if k == 'phi'}
        ring |= set(info['rotation_rows'])
    ring |= {j for rows, *_ in b.ring_frame_groups({n0 + j for j in ring}) for j in rows}
    for g in cen:                              # and every group AT a ring atom, whole
        if b.atom_in_ring[g['centre']]:
            ring |= set(g['rows'])
            assert g['reason'] in ('ring_atom', 'single_row')
    assert not ring & touched
    # INVERTIBLE centres: every centre the prior flips, and every centre the lock leaves
    # free, moves and pivots on periodic columns only
    for c in centre_table(b):
        if c.lock != LOCKED:
            rows = set(c.rows) | ({c.pivot} if c.pivot is not None else set())
            assert not rows & touched, c
    inv = invertible_centres(b)
    assert [(c.name, c.rows) for c in inv] == [(c.name, c.rows) for c in invertible_centres(a)]
    for c in inv:
        assert all(b.periodic_dims[k] for k in _cols_of_rows(b, list(c.rows)))


def test_groups_the_rule_leaves_alone_by_name():
    reasons = lambda smi: sorted(g['reason'] for g in _member(smi).sibling_offset_census()
                                 if len(g['rows']) > 1)
    assert reasons(AMINE) == ['', '', 'centre_not_locked']              # the NH2
    assert reasons('CC(=O)N(C)C') == ['', '', '', 'centre_not_locked']  # the amide N
    assert reasons('C[C@H]1CCCO1') == ['', 'ring_atom', 'ring_atom', 'ring_atom']
    assert reasons('CC#CC') == ['linear_frame']                         # a methyl on an alkyne
    with _default_dtype(torch.float64):
        tors = ConformerTorsions(smiles='CCCCO', level='torsion', **LOCK)
    assert {g['reason'] for g in tors.sibling_offset_census()
            if len(g['rows']) > 1} == {'torsion_tier'}


def test_a_ring_entry_row_on_a_chain_carbon_is_an_ordinary_follower(prior):
    """The row placing a ring's entry atom on a chain CH2 is drawn with its sibling group like
    any other row (`ring_blocks`); the group is converted, and the ring stays closed."""
    b = _member('CCCCC1CC1')
    b.ring_blocks(prior)
    entry = {j for info in b.ring_block_info for j in info['entry_rows']}
    (g,) = [g for g in b.sibling_offset_groups if set(g) & entry]
    ti = np.asarray(b.spec.torsion_index)
    assert not b.atom_in_ring[int(ti[g[0], 2])] and b.atom_in_ring[[int(ti[j, 3]) for j in g]].any()
    x, stats = b.sample_prior_states(prior, 1000, np.random.default_rng(0), report=False)
    assert stats['clip_frac']['sibling_offset'] == 0.0 and stats['closure_err'] < 0.05
    xa, sa = _plain('CCCCC1CC1').sample_prior_states(prior, 1000, np.random.default_rng(0),
                                                     report=False)
    # the default chart's draws to roundoff, not bit for bit: a ring-frame group's rows are
    # set from a dihedral measured on provisional positions, which each chart builds from its
    # own state
    assert np.abs(stats['dof'] - sa['dof']).max() < 1e-9
    assert stats['closure_err'] == pytest.approx(sa['closure_err'], abs=1e-9)


def test_the_torsion_tier_is_untouched():
    with _default_dtype(torch.float64):
        a = ConformerTorsions(smiles='CCCCO', level='torsion', **LOCK)
        b = ConformerTorsions(smiles='CCCCO', level='torsion', sibling_offset_box_deg=W, **LOCK)
    assert b.collective and not b.sibling_offset_groups
    assert torch.equal(a._M, b._M) and np.array_equal(a._free_block, b._free_block)
    x = _states(a)
    assert torch.equal(a.energy(x), b.energy(x))


# ------------------------------------------------------------------ the columns

@pytest.mark.parametrize('smi', SMIS)
def test_offset_columns_are_bounded_and_the_chart_constant_moves_by_their_scale(smi):
    a, b = _plain(smi), _member(smi)
    off = _offset_cols(b)
    other = np.setdiff1d(np.arange(b.data_ndim), off)
    per = np.asarray(b.periodic_dims)
    assert not per[off].any() and np.array_equal(per[other], np.asarray(a.periodic_dims)[other])
    assert set(off.tolist()) <= set(b._lin_free_idx.tolist())
    assert torch.allclose(b._free_scale[off], torch.full((len(off),), W_RAD,
                                                         dtype=torch.float64), rtol=0, atol=1e-15)
    assert torch.equal(b._free_scale[other], a._free_scale[other])
    # the leaders keep a full-circle periodic column
    lead = _cols_of_rows(b, [g[0] for g in b.sibling_offset_groups])
    assert per[lead].all() and torch.equal(b._free_scale[lead],
                                           torch.full((len(lead),), np.pi, dtype=torch.float64))
    # the coordinate count is the default chart's, and log|dq/dx| moves by log(W / 180) per
    # offset column and by nothing else: the map between the two charts is unit-triangular
    assert b.data_ndim == a.data_ndim == 3 * b.spec.n_atoms - 6
    assert b.log_chart_jacobian == pytest.approx(
        a.log_chart_jacobian + len(off) * np.log(W / 180.0), abs=1e-12)
    assert not b.collective and b._rtheta_free == a._rtheta_free
    assert f'{len(b.sibling_offset_groups)} of ' in b.describe()
    assert f'{len(off)} offset column(s)' in b.describe() and f'+/-{W:g} deg' in b.describe()


@pytest.mark.parametrize('smi', SMIS)
def test_the_two_charts_are_one_geometry_and_one_potential(smi):
    """Random states of the OLD chart, read into the new one: the same positions, the same
    force field + lock, and `energy` apart by the difference of the two chart constants."""
    a, b = _plain(smi), _member(smi)
    off = _offset_cols(b)
    one = torch.tensor(1.0, dtype=torch.float64)
    # (1) uniform over the old chart's box: followers anywhere on the circle, so the new
    # chart's offsets run far past their box and its wall is live there
    x = _states(a, n=48, seed=1)
    y = _to_new(a, b, x)
    assert float(y[:, off].abs().max()) > 1.5
    assert torch.allclose(a.build_positions(x), b.build_positions(y), rtol=0, atol=1e-10)
    ua = a.potential_energy(x, one) - a.bounding_energy(x, one)
    ub = b.potential_energy(y, one) - b.bounding_energy(y, one)
    assert torch.allclose(ua, ub, rtol=1e-10, atol=1e-8)
    assert torch.allclose(a.jacobian_energy(x, one), b.jacobian_energy(y, one), atol=1e-10)
    # (2) followers within the offset box (uniform over the NEW chart's box, read back into
    # the old chart): no wall on either side beyond the shared r/theta one
    y = _states(b, n=48, seed=2)
    x = a.state_from_dof(*b.dof_from_state(y))
    assert torch.allclose(_to_new(a, b, x), y, atol=1e-12)
    assert torch.allclose(a.build_positions(x), b.build_positions(y), rtol=0, atol=1e-10)
    assert torch.allclose(a.potential_energy(x, one), b.potential_energy(y, one),
                          rtol=1e-10, atol=1e-8)
    for logT in (0.0, 0.3, -0.4):
        t = torch.full((len(y),), logT, dtype=torch.float64)
        ea = a.energy(x, None, t)
        gap = b.energy(y, None, t) - ea
        want = a.log_chart_jacobian - b.log_chart_jacobian
        # a difference of two energies of box-uniform states (clashes): roundoff scales with
        # their size
        tol = 1e-11 * float(1.0 + ea.abs().max())
        assert float((gap - want).abs().max()) < tol, (float((gap - want).abs().max()), tol)
        assert want == pytest.approx(len(off) * np.log(180.0 / W), abs=1e-12)


@pytest.mark.parametrize('smi', SMIS[:4])
def test_the_state_round_trips_across_the_seam(smi):
    en = _member(smi)
    off = _offset_cols(en)
    per = torch.as_tensor(en.periodic_dims)
    n0 = en.n_r + en.n_th
    x = _states(en, n=96, seed=3)
    x[:8, off] = torch.linspace(-1.4, 1.4, 8, dtype=torch.float64)[:, None]   # past the box
    # put each follower's dihedral within a few degrees of the +/-pi seam, on either side,
    # by turning its group's leader: the offset then straddles the branch cut
    ref = en._ref_dof.numpy()
    k = 8
    for g in en.sibling_offset_groups:
        lc = int(_cols_of_rows(en, [g[0]])[0])
        for f in g[1:]:
            fc = int(_cols_of_rows(en, [f])[0])
            for eps in (-0.02, 0.02):
                # ref_f + pi * x_lead + W * x_f = pi + eps, and its mirror at -pi
                x[k, fc] = 0.3
                x[k, lc] = _wrap(np.pi + eps - ref[n0 + f] - W_RAD * 0.3) / np.pi
                x[k + 1, fc] = -0.3
                x[k + 1, lc] = _wrap(-np.pi + eps - ref[n0 + f] + W_RAD * 0.3) / np.pi
                k += 2
    assert k <= len(x)
    r, th, ph = en.dof_from_state(x)
    seam = np.abs(np.abs(_wrap(ph.numpy()[8:k])) - np.pi) < 0.03
    assert seam.any(axis=1).all(), 'every constructed row has a dihedral at the seam'
    back = en.state_from_dof(r, th, ph)
    assert torch.allclose(back, x, atol=1e-12)
    # a MEASURED dihedral arrives in (-pi, pi]: the same state whichever branch it is on
    shifted = en.state_from_dof(r, th, (ph + np.pi) % (2 * np.pi) - np.pi)
    assert torch.allclose(shifted, x, atol=1e-12)
    # and measured off the built positions, as the builders do it
    from mxtaltools.conformers.builder import collate, measure
    tree = collate([en.spec] * len(x), device='cpu')
    rm, thm, phm = measure(tree, en.build_positions(x), dummy_frame=en._tiled_dummy(len(x)))
    inside = (x[:, en._lin_free_idx].abs() <= 1.0).all(1)        # r, theta not clamped
    meas = en.state_from_dof(rm.reshape(len(x), -1), thm.reshape(len(x), -1),
                             phm.reshape(len(x), -1))
    assert int(inside.sum()) > 40
    assert torch.allclose(meas[inside], x[inside], atol=1e-8)
    # not folded: an offset past its box stays past it, a phi column wraps
    assert float(back[:, off].abs().max()) == pytest.approx(1.4, abs=1e-12)
    assert torch.equal(wrap_state(back, en.periodic_dims)[:, ~per], back[:, ~per])


@pytest.mark.parametrize('smi', SMIS)
def test_a_leader_turns_its_group_rigidly_and_an_offset_moves_one_member(smi):
    en = _member(smi)
    x = _states(en, n=6, seed=4, scale=0.5)
    base = _dihedrals(en, x)
    pos0 = en.build_positions(x).reshape(len(x), -1, 3)
    ti = np.asarray(en.spec.torsion_index)
    d = 0.37
    for g in en.sibling_offset_groups:
        lc = int(_cols_of_rows(en, [g[0]])[0])
        y = x.clone()
        y[:, lc] += d
        moved = _dihedrals(en, y)
        # every row of the group turns by the same angle, pi * d ...
        turn = _wrap((moved - base)[:, g].numpy())
        assert np.abs(turn - _wrap(np.pi * d)).max() < 1e-9
        # ... so the spacings between its members are unchanged, and so is every dihedral
        # outside the group
        gap = lambda ph: _wrap((ph[:, g[1:]] - ph[:, g[:1]]).numpy())
        assert np.abs(_wrap(gap(moved) - gap(base))).max() < 1e-9
        rest = np.setdiff1d(np.arange(en.n_ph), g)
        assert np.abs(_wrap((moved - base)[:, rest].numpy())).max() < 1e-9
        # rigid: the children, and the centre, keep every distance between them
        kids = [int(ti[j, 3]) for j in g] + [int(ti[g[0], 2]), int(ti[g[0], 1])]
        pos1 = en.build_positions(y).reshape(len(x), -1, 3)
        dist = lambda p: torch.cdist(p[:, kids], p[:, kids])
        assert torch.allclose(dist(pos0), dist(pos1), atol=1e-9)
        assert float((pos1[:, kids[:len(g)]] - pos0[:, kids[:len(g)]]).norm(dim=-1).min()) > 1e-2
        for f in g[1:]:
            fc = int(_cols_of_rows(en, [f])[0])
            z = x.clone()
            z[:, fc] += d
            one = _dihedrals(en, z)
            delta = _wrap((one - base).numpy())
            assert np.abs(delta[:, f] - W_RAD * d).max() < 1e-9      # W degrees per unit
            assert np.abs(np.delete(delta, f, axis=1)).max() < 1e-9  # and nothing else


def test_the_wall_is_zero_inside_and_positive_outside(boxed, plain):
    off = _offset_cols(boxed)
    one = torch.tensor(1.0, dtype=torch.float64)
    x = _states(boxed, n=64)                       # every column inside [-1, 1]
    x[0, off], x[1, off] = 1.0, -1.0               # the edges are inside
    assert torch.equal(boxed.bounding_energy(x, one), torch.zeros(64, dtype=torch.float64))
    for v in (1.25, -1.25):
        y = torch.zeros(1, boxed.data_ndim, dtype=torch.float64)
        y[0, off[0]] = v
        assert float(boxed.bounding_energy(y, one)) == pytest.approx(
            boxed.bounding_coeff * 0.25 ** 2, abs=1e-12)
        # ...where the default chart, which wraps this column, has none
        assert float(plain.bounding_energy(y, one)) == 0.0
    # a leader past +/-1 is not walled: it wraps
    lead = _cols_of_rows(boxed, [g[0] for g in boxed.sibling_offset_groups])
    y = torch.zeros(1, boxed.data_ndim, dtype=torch.float64)
    y[0, lead] = 1.5
    assert float(boxed.bounding_energy(y, one)) == 0.0
    assert not any(boxed.periodic_dims[c] for c in off)
    wrapped = wrap_state(torch.full((1, boxed.data_ndim), 1.3, dtype=torch.float64),
                         boxed.periodic_dims)
    assert torch.equal(wrapped[0, off], torch.full((len(off),), 1.3, dtype=torch.float64))
    assert torch.allclose(wrapped[0, lead], torch.full((len(lead),), -0.7, dtype=torch.float64))


def test_dihedral_tier_walls_the_offsets_and_keeps_the_frozen_blocks_frozen():
    a, b = _plain(level='dihedral'), _member(level='dihedral')
    off = _offset_cols(b)
    assert len(off) == 4 and b._lin_free_idx.tolist() == off.tolist()
    # r and theta are still the frozen reference: no clamp, log J still a constant
    assert not b._rtheta_free
    assert b.log_jacobian_const == a.log_jacobian_const is not None
    c = condition_from_energy(b)
    assert not bool(c.ctree_clamp) and check_state_convention(c, b) < 1e-9
    one = torch.tensor(1.0, dtype=torch.float64)
    y = torch.zeros(1, b.data_ndim, dtype=torch.float64)
    y[0, off[0]] = 1.5
    assert float(b.bounding_energy(y, one)) == pytest.approx(b.bounding_coeff * 0.25)
    x = _states(a, n=16, seed=6)
    assert torch.allclose(a.build_positions(x), b.build_positions(_to_new(a, b, x)), atol=1e-10)


# ------------------------------------------------------------------ graph and carrier

@pytest.fixture(scope='module')
def mixed():
    return _Set(MIXED, SIZES, torch.float64, stereo_coeff=300.0, sibling_offset_box_deg=W)


@pytest.mark.parametrize('smi', SMIS)
def test_the_graph_carries_both_columns_of_a_follower(smi):
    b = _member(smi)
    c = condition_from_energy(b)
    n0 = b.n_r + b.n_th
    rank3 = np.flatnonzero(np.minimum(np.asarray(b.spec.round_id), 3) >= 3)
    own, lead = c.ctree_ph_col[rank3].numpy(), c.ctree_ph_lead_col[rank3].numpy()
    scale, lscale = c.ctree_ph_scale[rank3].numpy(), c.ctree_ph_lead_scale[rank3].numpy()
    followers = np.flatnonzero(b.sibling_offset_rows)
    assert np.array_equal(np.flatnonzero(lead >= 0), followers)
    assert np.array_equal(own, _cols_of_rows(b, range(b.n_ph)))      # the own column, as ever
    assert np.array_equal(lead[followers],
                          _cols_of_rows(b, b.sibling_leader_row[followers]))
    assert scale[followers] == pytest.approx(W_RAD) and lscale[followers] == pytest.approx(np.pi)
    rest = np.setdiff1d(np.arange(b.n_ph), followers)
    assert (lead[rest] == -1).all() and (lscale[rest] == 0).all()
    assert scale[rest] == pytest.approx(np.pi)
    assert check_state_convention(c, b) < 1e-9
    from energies.conformer_data import sibling_offset_atom_flags
    assert np.array_equal(np.flatnonzero(sibling_offset_atom_flags(b).numpy()),
                          rank3[followers])


def test_the_carrier_places_offsets_in_the_theta_region_and_leaders_in_phi(mixed):
    multi, lay = mixed.multi, mixed.lay
    assert multi.is_carrier and not lay.is_identity
    assert all(m.sibling_offset_box_deg == W for m in multi._members.values())
    per = np.asarray(multi.periodic_dims)
    assert np.array_equal(per, lay.free_block == 2)
    n_off = 0
    for ident, m in multi._members.items():
        mine = _offset_cols(m)
        carried = lay.cols[ident][mine]
        n_off += len(mine)
        assert (lay.free_block[carried] == 1).all() and not per[carried].any()
        assert (lay.kind(ident)[carried] == SIBLING_OFFSET).all()
        assert set(carried.tolist()) <= set(multi._lin_free_idx.tolist())
        lead = lay.cols[ident][_cols_of_rows(m, [g[0] for g in m.sibling_offset_groups])]
        assert (lay.free_block[lead] == 2).all() and per[lead].all()
        c = carrier_pad_condition(condition_from_energy(m, identifier=ident), lay, ident, m)
        assert check_carrier_convention(c, lay, ident, m) < 1e-9
        # both of a follower's columns were remapped into the carrier
        hit = c.ctree_ph_lead_col >= 0
        assert int(hit.sum()) == len(mine)
        assert (lay.free_block[c.ctree_ph_lead_col[hit].numpy()] == 2).all()
        assert (lay.free_block[c.ctree_ph_col[hit].numpy()] == 1).all()
    assert n_off == 4 + 4 + 0 + 2 + 4 + 0 + 6
    # the theta region is wider, and the phi region narrower, than the default layout's
    with _default_dtype(torch.float64):
        old = CarrierLayout({i: _plain(i) for i in MIXED})
    assert lay.block_width[0] == old.block_width[0]
    assert lay.block_width[1] > old.block_width[1] and lay.block_width[2] < old.block_width[2]
    assert 'sibling offset' in lay.describe()


def test_the_one_pass_energy_is_each_members_own(mixed):
    multi = mixed.multi
    lin = multi._lin_free_idx
    assert bool((mixed.X.index_select(1, lin).abs() > 1).any()), 'the wall must be live'
    got = multi.energy(mixed.X, mixed.batch, mixed.logT)
    want = mixed.oracle()
    assert torch.allclose(got, want, rtol=1e-12, atol=1e-9), float((got - want).abs().max())
    assert torch.equal(multi._energy_per_member(mixed.X, mixed.batch, mixed.logT), want)
    # gradients too: the leader column reaches every row of its group through the graph
    X = mixed.X.clone().requires_grad_(True)
    g1, = torch.autograd.grad(multi.energy(X, mixed.batch, mixed.logT, keep_grads=True).sum(), X)
    g2, = torch.autograd.grad(mixed.oracle(X, keep_grads=True).sum(), X)
    assert torch.allclose(g1, g2, rtol=1e-9, atol=1e-7), float((g1 - g2).abs().max())
    # an offset past its box pays the wall in the one-pass energy too
    ident = MIXED[0]
    row = int(np.flatnonzero(mixed.assign == 0)[0])
    m = multi._members[ident]
    col = int(mixed.lay.cols[ident][_offset_cols(m)[0]])
    X = torch.zeros_like(mixed.X)
    X2 = X.clone()
    X2[row, col] = 1.5
    zero = torch.zeros(len(X), dtype=torch.float64)
    e0 = multi.energy(X, mixed.batch, zero, return_exp=True)[1].conformer_energy.flatten()
    e1 = multi.energy(X2, mixed.batch, zero, return_exp=True)[1].conformer_energy.flatten()
    x_m = torch.zeros(2, m.data_ndim, dtype=torch.float64)
    x_m[1, _offset_cols(m)[0]] = 1.5
    one = torch.tensor(1.0, dtype=torch.float64)
    u = m.potential_energy(x_m, one)
    assert float(e1[row] - e0[row]) == pytest.approx(float(u[1] - u[0]), abs=1e-9)
    assert float(m.bounding_energy(x_m, one)[1]) == pytest.approx(m.bounding_coeff * 0.25)


def test_pads_stay_zero_and_offsets_are_not_wrapped(mixed):
    multi = mixed.multi
    _, out = multi.energy(mixed.X, mixed.batch, mixed.logT, return_exp=True)
    stored = batch_states(out).reshape(mixed.X.shape)
    pads = mixed.pad_mask
    assert pads.any() and torch.equal(stored[pads], torch.zeros_like(stored[pads]))
    per = torch.as_tensor(multi.periodic_dims)
    assert torch.equal(stored[:, ~per], mixed.X[:, ~per])
    assert bool((stored[:, per].abs() <= 1.0).all())
    past = 0
    for j, ident in enumerate(MIXED):
        m = multi._members[ident]
        cols = mixed.lay.cols[ident][_offset_cols(m)]
        rows = np.flatnonzero(mixed.assign == j)
        if len(cols):
            past += int((stored[rows][:, cols].abs() > 1.0).sum())
    assert past > 0, 'some offset must sit past its box for the claim to bite'
    assert torch.allclose(out.conformer_energy.flatten().double(), mixed.potential_oracle(),
                          rtol=1e-12, atol=1e-9)
    # the prebuilt reward, read back off the stored rows at T = 1, is the live one there
    zero = torch.zeros(len(mixed.X), dtype=torch.float64)
    e1, out1 = multi.energy(mixed.X, mixed.batch, zero, return_exp=True)
    lr = multi.prebuilt_sample_to_reward(out1, torch.ones(len(mixed.X), dtype=torch.float64))
    assert torch.allclose(lr, -e1, rtol=1e-10, atol=1e-8)


def test_a_graph_of_the_other_chart_is_refused(mixed):
    """A conditions file built without the option, read by a set built with it (and the
    reverse): the same atoms and widths in an identity layout, another reconstruction."""
    # a set of one molecule scores through the parent; two constitutional isomers with the
    # same block codes are the smallest set that takes the one-pass path on an identity layout
    two = ['CCCO', 'CC(C)O']
    a = _Set(two, [3, 3], torch.float64, stereo_coeff=300.0, sibling_offset_box_deg=W)
    b = _Set(two, [3, 3], torch.float64, stereo_coeff=300.0)
    assert a.multi.carrier is None and b.multi.carrier is None
    assert torch.allclose(a.multi.energy(a.X, a.batch, a.logT), a.oracle(), atol=1e-9)
    with pytest.raises(RuntimeError, match='sibling-offset rows'):
        a.multi.energy(a.X, b.batch, a.logT)
    with pytest.raises(RuntimeError, match='sibling-offset rows'):
        b.multi.energy(b.X, a.batch, b.logT)


def test_a_set_refuses_members_built_under_another_box():
    with _default_dtype(torch.float64):
        from energies.multi_conformer import MultiConformerTorsions
        multi = MultiConformerTorsions(['CCCO', 'CC(C)O'], level='full',
                                       sibling_offset_box_deg=W, **LOCK)
        multi._members['CC(C)O'] = _member('CC(C)O', box=45.0)
        with pytest.raises(ValueError, match='sibling_offset_box_deg differs'):
            multi._build_library()


# ------------------------------------------------------------------ prior, tile, features

def test_prior_draws_land_inside_the_box_and_are_the_default_draws(boxed, plain, prior):
    n = 6000
    x, stats = boxed.sample_prior_states(prior, n, np.random.default_rng(5), report=False)
    off = _offset_cols(boxed)
    assert stats['clip_frac']['sibling_offset'] == 0.0
    assert float(x[:, off].abs().max()) < 1.0
    # each offset at its group's sibling jitter, in units of W
    groups = boxed.torsion_groups()
    g_sigma = boxed.sibling_jitter_sigma(groups, float(boxed.temperature))
    for g in boxed.sibling_offset_groups:
        s = g_sigma[groups.index(g)]
        sd = x[:, _cols_of_rows(boxed, g[1:])].std(0).numpy()
        assert sd == pytest.approx(s / W_RAD, rel=0.05)
    # the same draws as the default chart's, read into this one: same generator stream,
    # same dof, the same state on every column that is not an offset
    xp, sp = plain.sample_prior_states(prior, n, np.random.default_rng(5), report=False)
    assert np.array_equal(stats['dof'], sp['dof'])
    other = np.setdiff1d(np.arange(boxed.data_ndim), off)
    assert torch.equal(x[:, other], xp[:, other])
    assert torch.allclose(x, _to_new(plain, boxed, xp), atol=1e-12)
    # the default chart spreads the same follower over the whole circle
    assert float(xp[:, off].std(0).min()) > 0.3 and float(x[:, off].std(0).max()) < 0.15


def test_a_flipped_centre_keeps_its_periodic_followers(prior):
    """The amine N of CC(C)N: the prior draws both pyramids, an offset of about +120 or -120
    degrees, on the periodic column it always had; a box would have cut one."""
    from energies.invertible_centres import invertible_centres
    b = _member(AMINE)
    (c,) = invertible_centres(b)
    col = int(_cols_of_rows(b, list(c.rows))[0])
    piv = int(_cols_of_rows(b, [c.pivot])[0])
    assert b.periodic_dims[col] and b.periodic_dims[piv]
    x, stats = b.sample_prior_states(prior, 4000, np.random.default_rng(2), report=False)
    assert 0.4 < stats['reflected_frac'][0] < 0.6
    off = _wrap(np.pi * (x[:, col] - x[:, piv]).numpy())
    n0 = b.n_r + b.n_th
    ref_off = _wrap(float(b._ref_dof[n0 + c.rows[0]] - b._ref_dof[n0 + c.pivot]))
    d = _wrap(off + ref_off)                      # the drawn offset itself
    flipped = stats['reflected'][:, 0]
    assert np.abs(_wrap(d[~flipped] - ref_off)).max() < 0.6
    assert np.abs(_wrap(d[flipped] + ref_off)).max() < 0.6
    assert abs(ref_off) > 1.5


def test_prior_log_prob_is_the_draw_density_under_the_new_chart(boxed, plain, prior):
    """Change of variables, checked against the draws themselves.

    `prior_log_prob` is a density over the DoF (radians), whatever the chart. The map from the
    default columns to (leader, offsets) has unit determinant, so on an offset column
    x = (phi_f - phi_leader - reference offset) / W the state density is W q(...), read off
    `prior_log_prob` by moving that follower's row alone and normalising over the circle. Its
    histogram prediction must match the drawn states with the scale W, and must not with pi.
    """
    n = 20000
    x, stats = boxed.sample_prior_states(prior, n, np.random.default_rng(11), report=False)
    n0 = boxed.n_r + boxed.n_th
    # the density is over the DoF and does not read the chart
    assert np.array_equal(boxed.prior_log_prob(prior, stats['dof'][:64]),
                          plain.prior_log_prob(prior, stats['dof'][:64]))
    base = stats['dof'][:1].copy()
    for g in boxed.sibling_offset_groups:
        for f in g[1:]:
            c = int(_cols_of_rows(boxed, [f])[0])
            # the follower's value at offset 0, on this base draw
            zero = (float(boxed.ph0[f]) + base[0, n0 + g[0]] - float(boxed.ph0[g[0]]))
            grid = np.linspace(-np.pi, np.pi, 7201)[:-1]
            dof = np.repeat(base, len(grid), axis=0)
            dof[:, n0 + f] = zero + grid
            lp = boxed.prior_log_prob(prior, dof)
            q = np.exp(lp - lp.max())
            q /= q.sum() * (grid[1] - grid[0])                 # the offset's density, per radian
            edges = np.linspace(-0.4, 0.4, 17)
            hist = np.histogram(x[:, c].numpy(), bins=edges)[0] / n

            def predicted(scale):
                lo, hi = edges[:-1] * scale, edges[1:] * scale
                cdf = np.concatenate([[0.0], np.cumsum(q) * (grid[1] - grid[0])])
                at = lambda v: np.interp(v, np.concatenate([grid, [np.pi]]), cdf)
                return at(hi) - at(lo)

            want = predicted(float(boxed._free_scale[c]))
            assert float(boxed._free_scale[c]) == pytest.approx(W_RAD)
            noise = np.sqrt(np.maximum(want, 1e-6) / n)
            assert np.abs(hist - want).max() < 5 * noise.max() + 2e-3, (hist, want)
            assert hist.sum() > 0.99 and want.sum() == pytest.approx(hist.sum(), abs=0.01)
            assert np.abs(hist - predicted(np.pi)).max() > 0.1  # the old scale is refuted
    # and the importance weight is the default chart's, draw for draw: the state density is
    # q(dof) |dq/dx|, and log|dq/dx| is the constant the reward carries too
    lw_new = (-boxed.energy(x[:256]).numpy()
              - (boxed.prior_log_prob(prior, stats['dof'][:256]) + boxed.log_chart_jacobian))
    xp = plain.state_from_dof(*boxed.dof_from_state(x[:256]))
    lw_old = (-plain.energy(xp).numpy()
              - (plain.prior_log_prob(prior, stats['dof'][:256]) + plain.log_chart_jacobian))
    assert np.allclose(lw_new, lw_old, atol=1e-8)


@pytest.mark.parametrize('smi', [ACYC, AMINE, 'C[C@H]1CCCO1'])
def test_thermal_tile_widths_are_the_default_charts(smi):
    """The same directions and the same widths in radians: a follower's own axis moves it
    alone in both charts, and the group's rotation is the leader column in this one."""
    from energies.thermal_tile import member_widths
    a, b = _plain(smi), _member(smi)
    sa, ra = member_widths(a, float(a.temperature))
    sb, rb = member_widths(b, float(b.temperature))
    sc_a, sc_b = a._free_scale.numpy(), b._free_scale.numpy()
    off = _offset_cols(b)
    other = np.setdiff1d(np.arange(b.data_ndim), off)
    assert sb * sc_b == pytest.approx(sa * sc_a, rel=1e-6, abs=1e-12)      # radians / Angstrom
    assert sb[off] == pytest.approx(sa[off] * (180.0 / W), rel=1e-6)
    assert sb[other] == pytest.approx(sa[other], rel=1e-9, abs=1e-12)
    assert (sb[off] > 0).all()
    # the rotations: one row per sibling group in both; the same angle per unit draw, which
    # the default chart writes on every column of the group and this one on the leader's
    assert ra.shape == rb.shape
    groups = b.torsion_groups()
    conv = b.sibling_offset_groups
    for gi, g in enumerate(groups):
        cols = _cols_of_rows(b, g)
        w_a = ra[gi, cols] * sc_a[cols]                 # radians on each member, default chart
        assert w_a == pytest.approx(w_a[0], rel=1e-12) and w_a[0] > 0
        if g in conv:
            assert rb[gi, cols[0]] * np.pi == pytest.approx(w_a[0], rel=1e-6)
            assert (rb[gi, cols[1:]] == 0).all()
            # ... which turns every row of the group by that angle through the chart's map
            turned = (torch.as_tensor(rb[gi] * sc_b) @ b._M.T).numpy()
            rows = np.searchsorted(b._driven_idx.numpy(), b.n_r + b.n_th + np.asarray(g))
            assert turned[rows] == pytest.approx(w_a, rel=1e-6)
        else:
            assert rb[gi] == pytest.approx(ra[gi], rel=1e-9, abs=1e-12)
        assert np.count_nonzero(np.delete(rb[gi], cols)) == 0


def test_a_leader_column_carries_its_groups_rows_in_the_features(boxed, plain):
    from energies.dof_features import (dof_features, free_dof_atom_index, state_feature_names,
                                       state_features)
    f, f0 = state_features(boxed), state_features(plain)
    n_rows = f[:, state_feature_names().index('n_driven_rows')]
    lead = _cols_of_rows(boxed, [g[0] for g in boxed.sibling_offset_groups])
    rest = np.setdiff1d(np.arange(boxed.data_ndim), lead)
    assert n_rows[lead].tolist() == [len(g) for g in boxed.sibling_offset_groups]
    assert (n_rows[rest] == 1).all() and np.array_equal(f[rest], f0[rest])
    rows = dof_features(boxed)
    n0 = boxed.n_r + boxed.n_th
    for c, g in zip(lead, boxed.sibling_offset_groups):
        assert f[c, :-1] == pytest.approx(rows[[n0 + j for j in g]].mean(0))
    # the atom frames: R is the largest group, slot 0 of a leader is its own dihedral (what
    # `shared_atom_relation` reads), and its other slots are its followers' frames
    atoms, mask = free_dof_atom_index(boxed)
    atoms0, mask0 = free_dof_atom_index(plain)
    assert atoms0.shape[1] == 1 and atoms.shape[1] == max(len(g) for g in boxed.sibling_offset_groups)
    assert np.array_equal(atoms[:, 0], atoms0[:, 0]) and mask[:, 0].all()
    assert mask[rest, 1:].sum() == 0
    tf = boxed.torsion_frame_atoms()
    for c, g in zip(lead, boxed.sibling_offset_groups):
        assert np.array_equal(atoms[c, :len(g)], tf[g]) and mask[c, :len(g)].all()
    from models.ragged_set_policy import shared_atom_relation
    rel = shared_atom_relation(torch.as_tensor(atoms)[None])
    rel0 = shared_atom_relation(torch.as_tensor(atoms0)[None])
    assert torch.equal(rel, rel0)


def test_descend_clamps_an_offset_and_wraps_its_leader(boxed):
    from energies.prior_baselines import descend
    off = _offset_cols(boxed)
    x0 = torch.zeros(4, boxed.data_ndim, dtype=torch.float64)
    x0[:, off[0]] = torch.tensor([0.999, -0.999, 0.5, 0.0], dtype=torch.float64)
    bx, _ = descend(boxed, x0, steps=6, lr=0.2)
    assert float(bx[:, off].abs().max()) <= 1.0


def test_the_eval_class_tables_name_the_offsets(boxed, plain):
    from energies.conformer_eval_metrics import _dof_class_columns
    cols = _dof_class_columns(boxed)
    assert np.array_equal(cols['sibling_offset'], _offset_cols(boxed))
    assert len(cols['phi']) + len(cols['sibling_offset']) == len(_dof_class_columns(plain)['phi'])
    assert 'sibling_offset' not in _dof_class_columns(plain)


# ------------------------------------------------------------------ stamps and the database

def test_an_absent_key_is_none_in_every_stamp():
    import build_conformer_database as db
    import build_conformer_references as bcr
    old = {'level': 'full', 'force_field': 'mmff', 'stereo_coeff': 300.0}
    assert bcr.defining_energy(bcr.member_kwargs(old)) == \
        bcr.defining_energy(bcr.member_kwargs({**old, 'sibling_offset_box_deg': None}))
    assert bcr.defining_energy(bcr.member_kwargs(old))['sibling_offset_box_deg'] is None
    # a database stores geometries, which the option does not change: built without it, it
    # serves a consumer with it, and still refuses any other member argument
    assert 'sibling_offset_box_deg' in db.CHART_ONLY_KWARGS
    info = {'path': 'db', 'energy_kwargs': old}
    db.refuse_other_member_kwargs(info, {**old, 'sibling_offset_box_deg': W}, 'test')
    with pytest.raises(SystemExit, match='delta_r_max'):
        db.refuse_other_member_kwargs(info, {**old, 'delta_r_max': 0.2}, 'test')


def test_a_reference_table_stamped_before_the_argument_still_loads(tmp_path):
    """`load_references`: a stamp without the key was built under the default, None. A run
    WITH the option is another chart for the table's stored states and is refused."""
    import build_conformer_references as bcr
    kw = {'level': 'full', 'force_field': 'mmff', 'stereo_coeff': 300.0}
    cond = tmp_path / 'conditions.pt'
    cond.write_bytes(b'conditions')
    stamp_energy = bcr.defining_energy(kw)
    del stamp_energy['sibling_offset_box_deg']           # as a table written before it existed
    table = tmp_path / 'refs.pt'
    torch.save({'stamp': {'format': bcr.FORMAT, 'energy': stamp_energy,
                          'conditions': {'path': str(cond), 'sha256': bcr.file_sha256(cond),
                                         'identifiers': []}},
                'entries': {}, 'failures': {}}, table)
    bcr.load_references(table, conditions_path=cond, energy_kwargs=kw)
    bcr.load_references(table, conditions_path=cond,
                        energy_kwargs={**kw, 'sibling_offset_box_deg': None})
    with pytest.raises(bcr.StaleReferencesError, match='sibling_offset_box_deg'):
        bcr.load_references(table, conditions_path=cond,
                            energy_kwargs={**kw, 'sibling_offset_box_deg': W})


def test_a_stored_database_row_measures_into_the_offset_chart():
    """A real shard: rows stored by a build WITHOUT the option, matched by a member WITH it."""
    import build_conformer_database as db
    from conformer_modeller import ConformerModeller
    if not SHARD.exists():
        pytest.skip(f'conformer database shard not found at {SHARD}')
    blob = torch.load(SHARD, weights_only=False, map_location='cpu')
    ekw = dict(blob['header']['config']['energy_kwargs'])
    assert 'sibling_offset_box_deg' not in ekw and float(ekw['stereo_coeff']) > 0
    rec = next(c for r in blob['keys'].values() for c in r['conditions']
               if c['identifier'] == SHARD_IDENT)
    rec = {k: (rec.get(k) if k in db.OPTIONAL_FIELDS else rec[k]) for k in db.READ_FIELDS}
    members = {}
    with _default_dtype(torch.float64):
        for box in (None, W):
            pos, _ = rdkit_order_reference(SHARD_IDENT, rec['ref_pos'], level=ekw['level'],
                                           perm=rec['perm'], z=rec['z'])
            members[box] = ConformerTorsions(smiles=SHARD_IDENT, reference_positions=pos,
                                             sibling_offset_box_deg=box, **ekw)
    en, old = members[W], members[None]
    off = _offset_cols(en)
    assert len(en.sibling_offset_groups) >= 3 and len(off) >= 6
    cap = 8
    x, stored, info = db.match_rows(en, SHARD_IDENT, rec, cap, 1e-6)
    assert info['code'] is None, info
    assert info['rescore_gap'] <= 1e-6 and info['ref_pos_gap'] == 0.0
    assert len(stored) == info['rows'] == min(cap, len(rec['basins']['energy'])) > 1
    assert float(x[:, off].abs().max()) <= 1.0
    # the same rows in the chart they were built in: every column equal but the offsets,
    # which hold follower minus leader in units of W; the leaders hold what they held
    x0, stored0, info0 = db.match_rows(old, SHARD_IDENT, rec, cap, 1e-6)
    assert info0['code'] is None and np.array_equal(stored, stored0)
    other = np.setdiff1d(np.arange(en.data_ndim), off)
    assert torch.equal(x[:, other], x0[:, other])
    assert torch.allclose(x, _to_new(old, en, x0), atol=1e-12)
    assert torch.allclose(en.build_positions(x), old.build_positions(x0), atol=1e-9)
    assert float(x[:, off].abs().max()) > 0.05, 'the rows must move an offset'
    # the leaders of these rows sit all round the circle; the offsets within the box
    lead = _cols_of_rows(en, [g[0] for g in en.sibling_offset_groups])
    assert float(x[:, lead].abs().max()) > 0.3
    # the database signature does not read the option; the run's condition-set digest does
    assert db.member_signature(SHARD_IDENT, en) == rec['signature']
    assert ConformerModeller._member_signature(SHARD_IDENT, en) != rec['signature']
    assert ConformerModeller._member_signature(SHARD_IDENT, old) == rec['signature']
    # a row past the box is refused, by name
    wide = copy.deepcopy(rec)
    x_bad = torch.zeros(1, old.data_ndim, dtype=torch.float64)
    x_bad[0, off[0]] = 80.0 / 180.0                    # one follower alone, 80 degrees
    turned = old.build_positions(x_bad).reshape(1, -1, 3).numpy()
    wide['basins'] = {**rec['basins'], 'pos': turned, 'energy': np.zeros(1)}
    assert db.match_rows(en, SHARD_IDENT, wide, 1, 1e9)[2]['code'] == 'db_outside_box'
    assert db.match_rows(old, SHARD_IDENT, wide, 1, 1e9)[2]['code'] is None
