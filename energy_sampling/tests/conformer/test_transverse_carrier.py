"""Nitriles on the carrier: a linear bend's (u, v) pair scored in one pass, exactly as its member.

A nitrile's linear bend at `full` is carried as the transverse pair (u, v) -- block code 3 --
rather than held (tests/conformer/test_transverse_chart_contract.py). The carrier places u and v
in its THETA region and the one-pass multi-molecule energy scores them graph-natively: the
build and log J under the per-atom `ctree_transverse` mask, and the DISC wall in rho that the
member's `bounding_energy` adds on top of the box. Each claim is asserted against every
MEMBER's own `energy` on its own rows at its own width (the `_Set` oracle of
test_multi_energy_vectorized.py), because a transverse slot read as an angle, or a disc wall
skipped, returns a plausible number rather than an error.

The set mixes nitriles (one bend: propionitrile, 3-hydroxypropionitrile; two: malononitrile)
with molecules that have none, at level `full`, force field `mmff`. States are drawn in
[-1.2, 1.2] (box wall live) and then every bend is pushed to rho in (1.0, 1.4), past
`rho_wall` = 1.0, so the disc wall is live on every nitrile row -- inside the box the disc term
is exactly zero and a missing one would pass.
"""
import numpy as np
import pytest
import torch

from test_multi_energy_vectorized import _Set, _close, _clip, _default_dtype

from energies.conformer_carrier import (TRANSVERSE, CarrierLayout, carrier_pad_condition,
                                        check_carrier_convention)
from energies.conformer_data import (collate_conditions, condition_from_energy,
                                     transverse_atom_flags)

NITRILES = ['CCC#N', 'OCCC#N', 'N#CCC#N']
SMIS = NITRILES + ['CCO', 'CCCO', 'C']
SIZES = [9, 7, 5, 6, 3, 2]                    # B = 32, unequal
KW = dict(device='cpu', level='full', force_field='mmff')


def _to_disc_edge(s, X, seed=0, lo=1.0, hi=1.4):
    """``X`` with every transverse pair of every row moved to rho in (lo, hi), chart radians.

    Written through each member's own affine map (reference and signed scale), so rho here is
    the member's `_transverse_rho2`, and the other columns are left where they were.
    """
    rng = np.random.default_rng(seed)
    X = X.clone()
    for i, j in enumerate(s.assign):
        ident = s.smis[j]
        m = s.multi._members[ident]
        if m._transverse_t is None:
            continue
        cols = s.lay.cols[ident] if s.lay is not None else np.arange(m.data_ndim)
        for p in range(int(m._tv_u_cols.numel())):
            rho, a = rng.uniform(lo, hi), rng.uniform(0, 2 * np.pi)
            X[i, int(cols[int(m._tv_u_cols[p])])] = \
                (rho * np.cos(a) - float(m._tv_u_ref[p])) / float(m._tv_u_scale[p])
            X[i, int(cols[int(m._tv_v_cols[p])])] = \
                (rho * np.sin(a) - float(m._tv_v_ref[p])) / float(m._tv_v_scale[p])
    return X


@pytest.fixture(scope='module')
def tv64():
    s = _Set(SMIS, SIZES, torch.float64)
    s.X_edge = _to_disc_edge(s, s.X)
    return s


# ------------------------------------------------------------------ the layout

def test_the_set_reaches_the_transverse_paths(tv64):
    """Premises, so a later edit to the fixture cannot quietly make every test below easy."""
    s, lay = tv64, tv64.lay
    assert s.multi.is_carrier and not lay.is_identity
    for ident in NITRILES:
        m = s.multi._members[ident]
        assert m._transverse_t is not None and int(m.uncovered_linear_angles) == 0
    assert sum(int(s.multi._members[i]._tv_u_cols.numel()) for i in NITRILES) == 4
    # the DISC wall is live on every nitrile row at the edge states: the member's own wall
    # exceeds its box-only wall (rho_wall at infinity) there, and not at the uniform states
    for ident in NITRILES:
        m = s.multi._members[ident]
        rows = torch.as_tensor(np.flatnonzero(s.assign == SMIS.index(ident)))
        xm = lay.from_carrier(ident, s.X_edge[rows])
        full = m.bounding_energy(xm, 1.0)
        held, m.rho_wall = m.rho_wall, float('inf')
        try:
            box = m.bounding_energy(xm, 1.0)
        finally:
            m.rho_wall = held
        assert bool((full - box > 1e-3).all()), ident
    # and the box wall is live somewhere too
    assert bool((s.X.index_select(1, s.multi._lin_free_idx).abs() > 1).any())


def test_u_and_v_sit_in_the_theta_region_and_keep_their_kind(tv64):
    """No fourth block: u and v go to the theta region, so the phi region holds only columns
    that wrap (periodic mask column-constant) and the stamp stays three widths. The member's
    own code survives per column in `kind`."""
    s, lay = tv64, tv64.lay
    assert lay.free_block.tolist() == ([0] * lay.block_width[0] + [1] * lay.block_width[1]
                                       + [2] * lay.block_width[2])
    assert s.multi.periodic_dims == [b == 2 for b in lay.free_block.tolist()]
    for ident in SMIS:
        m = s.multi._members[ident]
        fb = np.asarray(m._free_block)
        # every width is the per-region maximum, u and v counted as theta
        region = np.where(fb == TRANSVERSE, 1, fb)
        for b in (0, 1, 2):
            assert int((region == b).sum()) <= lay.block_width[b]
        kd = lay.kind(ident)
        assert (kd[lay.cols[ident]] == fb).all()
        assert (kd[lay.pad_cols(ident)] == -1).all()
        tv_cols = lay.cols[ident][np.flatnonzero(fb == TRANSVERSE)]
        assert (lay.free_block[tv_cols] == 1).all()
        assert sorted(tv_cols.tolist()) == sorted(
            lay.cols[ident][torch.cat([m._tv_u_cols, m._tv_v_cols]).numpy()].tolist())
    assert lay.block_width[1] == max(int(np.isin(np.asarray(s.multi._members[i]._free_block),
                                                 (1, TRANSVERSE)).sum()) for i in SMIS)
    labels = {lay.column_label(j) for j in range(lay.K)}
    assert 'theta|transverse' in labels and 'phi' in labels


def test_the_library_flags_are_the_files_flags(tv64):
    """The positional check compares the batch against the library; both must be written by
    the one function, atom for atom."""
    lib = tv64.multi._lib
    for ident, m in tv64.multi._members.items():
        i = tv64.multi._lib_index[ident]
        a, b = int(lib.z_ptr[i]), int(lib.z_ptr[i + 1])
        got = lib.transverse[a:b]
        assert torch.equal(got, condition_from_energy(m, identifier=ident).ctree_transverse)
        # and independently of that function: the atom each flagged angle row places
        want = torch.zeros(m.spec.n_atoms, dtype=torch.bool)
        want[np.asarray(m.spec.angle_index)[np.asarray(m.transverse_angles), 2]] = True
        assert torch.equal(got, want)
        assert int(got.sum()) == int(np.asarray(m.transverse_angles).sum())


def test_check_carrier_convention_passes_for_every_member(tv64):
    """The carrier graph rebuilds each member's own geometry, bends included."""
    for ident, m in tv64.multi._members.items():
        cc = carrier_pad_condition(condition_from_energy(m, identifier=ident), tv64.lay,
                                   ident, m)
        assert check_carrier_convention(cc, tv64.lay, ident, m, tol=1e-9) < 1e-9


# ------------------------------------------------------------------ values and gradients

@pytest.mark.parametrize('edge', [False, True], ids=['uniform', 'disc_edge'])
@pytest.mark.parametrize('clip', [None, 30.0])
def test_energy_matches_the_member_oracle(tv64, edge, clip):
    X = tv64.X_edge if edge else tv64.X
    with _clip(tv64.multi, clip):
        got = tv64.multi.energy(X, tv64.batch, tv64.logT)
        want = tv64.oracle(X)
    assert _close(got, want), float((got - want).abs().max())


@pytest.mark.parametrize('edge', [False, True], ids=['uniform', 'disc_edge'])
def test_gradient_matches_the_member_oracle(tv64, edge):
    """The pathwise force through the bend, the disc wall and log sinc; pads carry none."""
    X = tv64.X_edge if edge else tv64.X
    xa = X.clone().requires_grad_(True)
    g, = torch.autograd.grad(tv64.multi.energy(xa, tv64.batch, tv64.logT,
                                               keep_grads=True).sum(), xa)
    xb = X.clone().requires_grad_(True)
    go, = torch.autograd.grad(tv64.oracle(xb, keep_grads=True).sum(), xb)
    assert float((g - go).abs().max()) <= 1e-9 * float(go.abs().max())
    pads = tv64.pad_mask
    assert pads.any() and torch.equal(g[pads], torch.zeros_like(g[pads]))


def test_return_exp_bakes_the_members_potential_disc_included(tv64):
    """`conformer_energy` is the member's `potential_energy` at T = 1 -- box AND disc."""
    e, out = tv64.multi.energy(tv64.X_edge, tv64.batch.clone(), tv64.logT, return_exp=True)
    assert torch.equal(e, tv64.multi.energy(tv64.X_edge, tv64.batch, tv64.logT))
    want = torch.zeros(tv64.X.shape[0], dtype=torch.float64)
    one = torch.tensor(1.0, dtype=torch.float64)
    for j, ident in enumerate(SMIS):
        rows = torch.as_tensor(np.flatnonzero(tv64.assign == j))
        m = tv64.multi._members[ident]
        want[rows] = m.potential_energy(tv64.lay.from_carrier(ident, tv64.X_edge[rows]), one)
    assert _close(out.conformer_energy, want)


def test_prebuilt_reward_measures_the_bend_as_the_member_does(tv64):
    """The prebuilt reward re-adds log J from the stored state; on a bend that is log sinc(rho),
    which reads v. At T = 1 it is exactly -energy of each member."""
    zero = torch.zeros_like(tv64.logT)
    _, baked = tv64.multi.energy(tv64.X_edge, tv64.batch.clone(), zero, return_exp=True)
    want = -tv64.oracle(tv64.X_edge, logT=zero)
    assert _close(tv64.multi.prebuilt_sample_to_reward(baked, 1.0), want)


def test_float32_matches_the_member_oracle():
    """PRODUCTION DTYPE (conformer_modeller.py pins float32): the same comparison at relative
    1e-5 -- two summation orders of the same float32 arithmetic."""
    s = _Set(SMIS, SIZES, torch.float32)
    X = _to_disc_edge(s, s.X)
    got = s.multi.energy(X, s.batch, s.logT)
    want = s.oracle(X)
    assert got.dtype == torch.float32
    rel = ((got - want.to(got.dtype)).abs() / want.abs().clamp_min(1.0).to(got.dtype)).max()
    assert float(rel) < 1e-5, float(rel)


# ------------------------------------------------------------------ eval labels

@pytest.mark.parametrize('ident', ['CCC#N', 'OCCC#N', 'N#CCC#N', 'CCO'])
def test_each_column_is_labelled_with_its_own_atoms_element(tv64, ident):
    """`dof_elem/*` groups columns by the element of the atom that owns them. A bend's u and v
    leave the theta and phi classes, so assigning the j-th column of a class to the j-th row
    of that class's table shifts every later column onto its neighbour's atom. Checked
    against the per-column atom frames `free_dof_atom_index` reads off the same chart."""
    import energies.conformer_eval_metrics as cm
    from energies.dof_features import free_dof_atom_index

    m = tv64.multi._members[ident]
    z = np.asarray(m.spec.z)
    atoms, _ = free_dof_atom_index(m)
    frame = atoms[:, 0]                           # a selection tier: one row per column
    fb = np.asarray(m._free_block)
    want = np.select([fb == 0, fb == 1, fb == 2, fb == TRANSVERSE],
                     [np.maximum(z[frame[:, 0]], z[frame[:, 1]]), z[frame[:, 1]],
                      z[frame[:, 1]], z[frame[:, 2]]])  # a bend's vertex is its frame's c
    assert (cm._central_elements(m) == want).all()

    stats = cm.dof_element_stats(m, tv64.lay.from_carrier(ident, tv64.X))
    n_tv = int((fb == TRANSVERSE).sum())
    assert stats.get('dof_elem/transverse_C_n', 0) == n_tv


# ------------------------------------------------------------------ refusals

def _move_flag(batch, row):
    """Move `row`'s transverse flag onto another of its atoms that owns a torsion row: the
    same COUNT of flags, on a different atom -- a file charted differently from the member."""
    b2 = batch.clone()
    flag = b2.ctree_transverse.clone()
    atoms = torch.nonzero(b2.batch == row).flatten()
    have = atoms[flag[atoms]]
    free = atoms[~flag[atoms] & (b2.ctree_round[atoms] >= 3)]
    assert have.numel() >= 1 and free.numel() >= 1
    flag[have[0]] = False
    flag[free[0]] = True
    assert int(flag.sum()) == int(b2.ctree_transverse.sum())
    b2.ctree_transverse = flag
    return b2


def test_a_stale_file_with_moved_flags_is_refused(tv64):
    """Same molecule, same atoms, same number of bends -- on other atoms. A count check passes
    this; the positional check refuses it on every path that resolves rows."""
    row = int(np.flatnonzero(tv64.assign == SMIS.index('OCCC#N'))[0])
    b2 = _move_flag(tv64.batch, row)
    with pytest.raises(RuntimeError, match=f'row {row} .*transverse flags'):
        tv64.multi.energy(tv64.X, b2, tv64.logT)
    with pytest.raises(RuntimeError, match='transverse flags'):
        tv64.multi.member_groups(b2)
    zero = torch.zeros_like(tv64.logT)
    _, baked = tv64.multi.energy(tv64.X, tv64.batch.clone(), zero, return_exp=True)
    with pytest.raises(RuntimeError, match='transverse flags'):
        tv64.multi.prebuilt_sample_to_reward(_move_flag(baked, row), 1.0)


def test_a_batch_without_flags_is_refused_against_a_member_with_them(tv64):
    """A file written before the transverse chart reads as flag-free, which no nitrile is."""
    b2 = tv64.batch.clone()
    del b2.ctree_transverse
    with pytest.raises(RuntimeError, match='transverse flags'):
        tv64.multi.member_groups(b2)


# ------------------------------------------------------------------ layouts with no pads

def test_a_pure_permutation_carrier_scores_exactly_and_requires_state_mask():
    """Two same-size nitriles whose bends sit on different rows share every region width: the
    carrier has NO pads (K = k) but is not the identity. A member-order file then passes the
    pad check and every width, so a carrier batch without `state_mask` is refused."""
    s = _Set(['OCCCC#N', 'OCC(C)C#N'], [4, 5], torch.float64)
    m1, m2 = (s.multi._members[i] for i in s.smis)
    assert s.multi.is_carrier and s.lay.K == m1.data_ndim == m2.data_ndim
    assert not s.lay.pad_cols(s.smis[0]).size and not s.lay.pad_cols(s.smis[1]).size
    assert _close(s.multi.energy(s.X, s.batch, s.logT), s.oracle())

    with _default_dtype(torch.float64):
        rows = [condition_from_energy(s.multi._members[s.smis[j]], identifier=s.smis[j])
                for j in s.assign]
    for c, j in zip(rows, s.assign):
        c.mol_id = torch.tensor([s.registry[s.smis[j]]])
    with pytest.raises(RuntimeError, match='no state_mask'):
        s.multi.energy(s.X, collate_conditions(rows), s.logT)


def test_a_same_layout_nitrile_set_is_the_identity_and_scores_exactly():
    """Members with identical per-column codes, bend included (OCCC#N and CC(O)C#N; also a
    nitrile's stereoisomers), keep the identity layout: no permutation, no pads."""
    s = _Set(['OCCC#N', 'CC(O)C#N'], [4, 5], torch.float64)
    assert not s.multi.is_carrier
    lay = CarrierLayout(s.multi._members)
    assert lay.is_identity and all((c == np.arange(lay.K)).all() for c in lay.cols.values())
    X = _to_disc_edge(s, s.X)
    assert _close(s.multi.energy(X, s.batch, s.logT), s.oracle(X))


def test_a_single_nitrile_condition_carries_the_same_flags_as_before():
    """The single-molecule route writes `ctree_transverse` through the shared helper; the
    values are the scatter of `transverse_angles` onto the atoms their angle rows place."""
    from energies.conformer_torsions import ConformerTorsions
    with _default_dtype(torch.float64):
        m = ConformerTorsions(smiles='CCC#N', **KW)
    c = condition_from_energy(m)
    rank = np.minimum(np.asarray(m.spec.round_id), 3)
    want = torch.zeros(m.spec.n_atoms, dtype=torch.bool)
    want[torch.as_tensor(rank >= 2)] = torch.as_tensor(np.asarray(m.transverse_angles))
    assert torch.equal(c.ctree_transverse, want) and bool(want.any())
    assert torch.equal(transverse_atom_flags(m), want)
