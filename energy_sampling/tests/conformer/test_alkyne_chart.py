"""Alkynes at `full`: the sp-root rule and the Z-matrix dummy frame, end to end.

An alkyne was refused at `full` for two reasons the transverse pair alone could not fix: a
tree rooted on an sp carbon (atoms 0-1-2 collinear, so the frame convention fixes no plane) and
dihedrals whose frame runs through an sp centre (a-b-c collinear, so no plane to measure phi
in). `ConformerTorsions` now roots off the sp carbon and measures those dihedrals against a
Z-matrix dummy atom on the axis (mxtaltools builder.DummyFrame), at `flex` and `full` only.

What this file pins, each against the thing that would silently go wrong:

  * the chart is COMPLETE (d = 3N-6, nothing held), x = 0 rebuilds the reference, and the
    graph path reconstructs the energy's geometry exactly, single-molecule and on a carrier;
  * molecules WITHOUT a collinear frame are on the pre-change code path at every tier (no
    dummy mask, default tree), and an sp-root alkyne's helper tiers keep the default tree;
  * the one-pass multi-molecule energy scores alkyne rows exactly as each member does, and a
    file whose dummy flags sit on other atoms is refused;
  * the per-coordinate frames and features name the atom a dummy points to, not the
    collinear one;
  * the regularity certificate bounds the frames the builder ACTUALLY constructs (read from
    inside it), with both of its terms; a dummy whose anchor is on the axis is not taken;
  * the dummy chart's own singular set (rho = pi/2 on an anchoring bend) is refused at
    construction and walled; `dummy_frame_crossings` counts it when called.

Byte-identity against the PRE-change code is not a unit test -- there is no pre-change code to
call. It was measured when this change was made, on 310 molecules that build at `full` and 80
refused alkynes, at all four tiers; what this file pins is the mechanism that makes it hold
(no dummy mask and the default tree wherever there is no collinear frame, the default tree at
the helper tiers everywhere).
"""
import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from test_multi_energy_vectorized import _Set, _close, _default_dtype
from test_transverse_carrier import _to_disc_edge

from energies.conformer_carrier import (CarrierLayout, carrier_pad_condition,
                                        check_carrier_convention)
from energies.conformer_data import (check_state_convention, condition_from_energy,
                                     dummy_frame_atom_flags)
from energies.conformer_torsions import ChartRefused, ConformerTorsions

KW = dict(device='cpu', force_field='mmff')
#: sp root by default (propyne, but-2-yne, the alcohol), chains of dummies (the tetrayne, and
#: the diyne, whose on-axis atom is c's own torsion reference), a dummy next to a ring (the
#: epoxide), and a branch point carrying two alkynes -- the worst in-box frame of the design
#: this replaced
ALKYNES = ['CC#C', 'CC#CC', 'CC#CCO', 'CCC#CC#CC#C', 'CC#CC#CC', 'C#CC1CO1', 'CC(=O)C(C#C)C#C']
CONTROLS = ['CCCCO', 'CCCO', 'CC(C)O', 'CCC(C)O', 'c1ccccc1', 'Oc1ccccc1', 'CCC#N', 'N#CCCO',
            'CC(N)=O', 'C1CCOC1']
LEVELS = ('torsion', 'dihedral', 'flex', 'full')


@pytest.fixture(autouse=True)
def _float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _en(smiles, level='full', **kw):
    return ConformerTorsions(smiles=smiles, level=level, **{**KW, **kw})


# ------------------------------------------------------------------ completeness

@pytest.mark.parametrize('smiles', ALKYNES)
def test_an_alkyne_is_complete_and_both_builders_agree(smiles):
    en = _en(smiles)
    assert en.constrained_rows == 0 and en.uncovered_linear_angles == 0
    assert en.data_ndim == 3 * en.spec.n_atoms - 6
    assert int(np.asarray(en.dummy_frame_rows).sum()) == int(
        np.asarray(en.torsion_frame_is_linear).sum()) > 0
    assert check_state_convention(condition_from_energy(en), en) == 0.0
    # and on a genuine carrier: a partner with a different layout forces padding
    partner = _en('CCCCO')
    lay = CarrierLayout({smiles: en, 'partner': partner})
    assert not lay.is_identity
    cc = carrier_pad_condition(condition_from_energy(en, identifier=smiles), lay, smiles, en)
    assert check_carrier_convention(cc, lay, smiles, en) == 0.0


@pytest.mark.parametrize('smiles', ALKYNES)
def test_the_zero_state_rebuilds_the_reference(smiles):
    """x = 0 IS the reference conformer, which e_ref, the anchors and `state_from_dof` all
    assume. It holds only because `ph0` is RE-MEASURED against the dummies: the first measure
    runs before the flags exist and reads a dummy row's phi against the collinear atom.
    Without the re-measure the density is unchanged (phi spans the circle) but x = 0 lands
    0.33 A (CC#CC) to 0.42 A (CC#CCO) off the reference, and every other test here passes."""
    en = _en(smiles)
    p = en.build_positions(torch.zeros(1, en.data_ndim))
    err = (torch.cdist(p, p) - torch.cdist(en.ref_pos, en.ref_pos)).abs().max()
    assert float(err) < 1e-10, f'{smiles}: x = 0 is {float(err):.3g} A off the reference'


def _frame_sines(monkeypatch, en, x):
    """The sines of every dummy-frame angle `build` CONSTRUCTED for states ``x``: X's azimuth
    frame angle(P, q, b) and the placement frame angle(X, b, c).

    Recorded from inside mxtaltools' builder (what `_dummy_points` hands `dummy_reference`),
    so they are the frames the energy was computed in -- not a recomputation from the anchor
    rule, which is what `dummy_frame_min_sin` is and so cannot be checked against itself.
    """
    import mxtaltools.conformers.builder as B
    calls = []
    real_points, real_ref = B._dummy_points, B.dummy_reference

    def ref(pp, pq, pb):
        out = real_ref(pp, pq, pb)
        calls[-1].update(p=pp, q=pq, b=pb, x=out)
        return out

    def points(tree, refs, pos, xs, s):
        calls.append(dict(c=tree.ref_c[s]))
        return real_points(tree, refs, pos, xs, s)

    monkeypatch.setattr(B, 'dummy_reference', ref)
    monkeypatch.setattr(B, '_dummy_points', points)
    try:
        pos = en.build_positions(x)
    finally:
        monkeypatch.setattr(B, 'dummy_reference', real_ref)
        monkeypatch.setattr(B, '_dummy_points', real_points)
    assert calls and all('x' in k for k in calls), 'build never constructed a dummy'

    def sine(u, w):
        return torch.linalg.cross(u, w, dim=-1).norm(dim=-1) / (u.norm(dim=-1) * w.norm(dim=-1))
    return torch.cat([sine(k['p'] - k['q'], k['b'] - k['q']) for k in calls]
                     + [sine(k['x'] - k['b'], pos[k['c']] - k['b']) for k in calls])


def _box_draws(en, n, seed):
    """``n`` uniform states and ``n`` random box CORNERS, where the theta and (u, v) terms of
    the certificate are attained."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, en.data_ndim, generator=g) * 2 - 1
    corners = torch.where(torch.rand(n, en.data_ndim, generator=g) < 0.5, -1.0, 1.0)
    return torch.cat([x, corners])


@pytest.mark.parametrize('smiles', ALKYNES)
def test_the_dummy_frames_carry_a_regularity_certificate(smiles, monkeypatch):
    """`dummy_frame_min_sin` bounds the sine of every frame angle the builder actually
    constructs, over uniform states and box corners; and nothing crosses rho = pi/2."""
    en = _en(smiles)
    assert en.dummy_frame_min_sin is not None and en.dummy_frame_min_sin > 0.2
    x = _box_draws(en, 128, 0)
    got = float(_frame_sines(monkeypatch, en, x).min())
    assert got >= en.dummy_frame_min_sin - 1e-9, (got, en.dummy_frame_min_sin)
    assert en.dummy_frame_crossings(x) == 0
    assert bool(torch.isfinite(en.energy(x)).all())


def test_the_certificate_carries_the_bend_term(monkeypatch):
    """The own-frame term cos(rho_c) is in the bound, not only the tree-angle term.

    Propyne has one dummy row, unchained, so its certificate is min(cos rho_c, sin of the
    anchor's tree angle over its box). At delta_theta_max = 1.0 the bend term binds (rho_c
    reaches ~1.41 rad, cos 0.156, against a tree term of ~0.21), so a certificate that
    dropped it would claim a bound the builder's frames can break."""
    en = _en('CC#C', delta_theta_max=1.0)
    k = en._tv_dummy_anchor
    assert bool(k.any())
    corners = torch.zeros(4, en.data_ndim)
    for i, (su, sv) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        corners[i, en._tv_u_cols] = float(su)
        corners[i, en._tv_v_cols] = float(sv)
    rho = float(en._transverse_rho2(corners)[:, k].sqrt().max())
    assert en.dummy_frame_min_sin <= np.cos(rho) + 1e-12, (en.dummy_frame_min_sin, np.cos(rho))
    got = float(_frame_sines(monkeypatch, en, _box_draws(en, 128, 1)).min())
    assert got >= en.dummy_frame_min_sin - 1e-9


# ------------------------------------------------------------------ what must not move

@pytest.mark.parametrize('smiles', CONTROLS)
@pytest.mark.parametrize('level', LEVELS)
def test_a_molecule_without_a_collinear_frame_is_on_the_old_path(smiles, level):
    """No dummy mask, the default root, and `_build` is builder.build with the old arguments."""
    from mxtaltools.conformers.builder import build
    try:
        en = _en(smiles, level)
    except ValueError as exc:
        pytest.skip(f'{smiles} does not build at {level}: {exc}')
    assert en._dummy_t is None and not np.asarray(en.dummy_frame_rows).any()
    assert en.spec.root_moved_from == -1
    assert np.array_equal(en.torsion_frame_atoms(), np.asarray(en.spec.torsion_index))
    x = torch.rand(4, en.data_ndim, generator=torch.Generator().manual_seed(1)) * 2 - 1
    r, th, ph = en.dof_from_state(x)
    tree, _ = en._batch(4)
    old = build(tree, r.reshape(-1), th.reshape(-1), ph.reshape(-1),
                transverse=en._tiled_transverse(4))
    assert torch.equal(en.build_positions(x), old)


@pytest.mark.parametrize('level,k', [('torsion', 2), ('dihedral', 8)])
def test_an_sp_root_alkyne_keeps_its_default_tree_at_the_helper_tiers(level, k):
    """The root rule changes the tree of an sp-root molecule at every tier it runs at, so it
    runs at `flex` and `full` only: the helper tiers' stored anchors keep their meaning.
    COCC#CCCC#C builds at both helper tiers; run ungated, the rule took its k from 2 to 3 at
    `torsion` and from 8 to 11 at `dihedral` (the design critique's measurement)."""
    low = _en('COCC#CCCC#C', level)
    assert low.spec.root_moved_from == -1 and low._dummy_t is None
    assert low.data_ndim == k
    full = _en('COCC#CCCC#C', 'full')
    assert full.spec.root_moved_from >= 0
    assert int(full.spec.perm[0]) != int(low.spec.perm[0])


def test_a_root_move_the_typing_does_not_back_is_refused(monkeypatch):
    """The graph rule (carbon, two neighbours) is checked against this chart's own linearity;
    a disagreement would change a chart that builds, so it raises instead of proceeding."""
    import mxtaltools.conformers.topology as T
    real = T.spec_from_graph

    def lying(*a, **k):
        spec = real(*a, **k)
        spec.root_moved_from = 1              # propanol's middle carbon: four neighbours
        return spec
    monkeypatch.setattr(T, 'spec_from_graph', lying)
    with pytest.raises(RuntimeError, match='graph rule and the typing disagree'):
        _en('CCCO', 'full')


# ------------------------------------------------------------------ refusals by cause

def test_a_wholly_linear_molecule_is_refused_with_its_own_code():
    with pytest.raises(ChartRefused) as exc:
        _en('C#C')
    assert exc.value.code == 'wholly_linear' and '3N-5' in str(exc.value)


def test_a_cumulated_centre_is_held_and_refused_with_its_own_code():
    """An allene's dummy row IS its end-to-end twist, which MMFF94 does not restrain."""
    with pytest.raises(ChartRefused) as exc:
        _en('CC=C=CC')
    assert exc.value.code == 'cumulated'
    flex = _en('CC=C=CC', 'flex')
    assert not np.asarray(flex.dummy_frame_rows).any()
    assert int(np.asarray(flex.cumulated_frames).sum()) == int(flex.held_frame_rows.sum()) > 0


# ------------------------------------------------------------------ the dummy's own singularity

def test_a_disc_wall_past_half_pi_is_refused_when_dummies_exist():
    with pytest.raises(ValueError, match='rho_wall'):
        _en('CC#CC', rho_wall=1.6)
    _en('CCC#N', rho_wall=1.6)                  # a nitrile anchors no dummy: accepted


def test_a_box_that_lets_an_anchoring_bend_reach_half_pi_is_refused():
    """At the default delta_theta_max = 0.5 the reach is ~0.71 rad and this check is inert;
    from ~1.11 it is the only guard against an in-box collinear dummy frame. 1.15 puts the
    corner at rho ~1.63 > pi/2, still inside the (u, v) chart's own limit of pi and the
    theta boxes' (0, pi) -- so the nitrile, whose bend anchors no dummy, is accepted."""
    with pytest.raises(ValueError, match='anchors a dummy frame'):
        _en('CC#C', delta_theta_max=1.15)
    _en('CCC#N', delta_theta_max=1.15)


def test_a_dummy_whose_anchor_lies_on_the_axis_is_not_taken():
    """Butadiyne at `flex`: the terminal H's collinear frame passes the tree rule, but its X's
    azimuth anchor is the other seed, on the axis (the anchor angle is the linear seed
    angle), so X would be built from a collinear frame. The fixed-point pass drops it and the
    molecule, wholly linear, has nothing free at `flex`. Taking the row instead constructs a
    'free' coordinate whose frame is noise (measured: d = 2 instead of a refusal)."""
    with pytest.raises(ValueError, match='no free degrees of freedom'):
        _en('C#CC#C', 'flex')


def test_ring_blocks_compares_against_a_tree_built_with_the_same_root_rule():
    """`ring_blocks` asserts the spec tree equals mxtaltools' tree_* encoding before it maps a
    ring bank. For a molecule whose root the sp rule moved, the comparison tree must be built
    with the same rule or the assertion fires -- on every such molecule, and only once a prior
    is loaded. The stub prior stops the call right after the assertion."""
    class _Reached(Exception):
        pass

    class _Prior:
        def ring_systems(self, m):
            raise _Reached

    en = _en('CC#CC')
    assert en.spec.root_moved_from >= 0
    with pytest.raises(_Reached):
        en.ring_blocks(_Prior())


def test_the_half_pi_set_is_counted_where_transverse_crossings_cannot_see_it():
    en = _en('CC#CC')
    anchor = int(torch.nonzero(en._tv_dummy_anchor)[0])
    x = torch.zeros(1, en.data_ndim)
    for rho, want in ((1.0, 0), (1.6, 1)):
        x[0, en._tv_u_cols[anchor]] = (rho - float(en._tv_u_ref[anchor])) / float(
            en._tv_u_scale[anchor])
        x[0, en._tv_v_cols[anchor]] = -float(en._tv_v_ref[anchor]) / float(
            en._tv_v_scale[anchor])
        assert en.dummy_frame_crossings(x) == want
        assert en.transverse_crossings(x) == 0


# ------------------------------------------------------------------ the frames a policy reads

@pytest.mark.parametrize('smiles', ['CC#CC', 'CCC#CC#CC#C'])
def test_a_dummy_rows_frame_names_the_atom_the_dummy_points_to(smiles):
    """`torsion_index[:, 0]` on a dummy row is the collinear atom, which fixes nothing. The
    features and the atom frames must name the real anchor, which sits OFF the axis."""
    from energies.dof_features import dof_features, free_dof_atom_index
    en = _en(smiles)
    dm = np.asarray(en.dummy_frame_rows)
    ti = np.asarray(en.spec.torsion_index)
    tf = en.torsion_frame_atoms()
    assert (tf[dm, 0] != ti[dm, 0]).all() and (tf[~dm] == ti[~dm]).all()
    pos = en.ref_pos.numpy()
    for a, b, c in tf[dm, :3]:
        u, w = pos[a] - pos[b], pos[c] - pos[b]
        ang = np.degrees(np.arccos(np.clip(u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1, 1)))
        assert 10.0 < ang < 170.0, (a, b, c, ang)
    atoms, mask = free_dof_atom_index(en)
    rows = en._driven_idx.numpy()[en._M.numpy().argmax(0)]
    phi_row = rows - en.n_r - en.n_th
    for j, pr in enumerate(phi_row):
        if 0 <= pr < en.n_ph and dm[pr]:
            assert atoms[j, 0, 0] == tf[pr, 0]

    # the handcrafted features: slot a0 of a dummy row -- its phi row, and the u row of a
    # transverse pair whose v it carries -- describes the atom the dummy points to
    from energies.dof_features import _elem_onehot, atom_parity, feature_names
    f = dof_features(en)
    assert f.shape[0] == en.spec.n_dof
    a0 = [i for i, nm in enumerate(feature_names()) if nm.startswith('a0_')]
    z, keys, par = np.asarray(en.spec.z), en.atom_keys, atom_parity(en)

    def block(a):
        return np.concatenate([_elem_onehot(int(z[a])),
                               [float(keys[a, 1]), float(en.atom_in_ring[a]),
                                float(en.atom_is_aromatic[a]), float(par[a]), 1.0]])
    tv, part = np.asarray(en.transverse_angles), np.asarray(en.transverse_partner)
    checked = [(en.n_r + en.n_th + j, j) for j in np.flatnonzero(dm)]
    checked += [(en.n_r + i, int(part[i])) for i in np.flatnonzero(tv) if dm[part[i]]]
    for row, j in checked:
        assert np.array_equal(f[row, a0], block(tf[j, 0])), (row, j)
    # and the check discriminates: on some dummy row the collinear atom reads differently
    assert any(not np.array_equal(block(tf[j, 0]), block(ti[j, 0])) for _, j in checked)


# ------------------------------------------------------------------ the multi-molecule energy

SET = ['CC#CCO', 'CCC#CC#CC#C', 'CCC#N', 'CCO', 'CCCO']
SIZES = [7, 6, 5, 4, 3]


@pytest.fixture(scope='module')
def alk64():
    s = _Set(SET, SIZES, torch.float64)
    s.X_edge = _to_disc_edge(s, s.X)
    return s


def test_the_set_is_a_carrier_with_dummy_rows(alk64):
    assert alk64.multi.is_carrier
    lib = alk64.multi._lib
    for ident, m in alk64.multi._members.items():
        i = alk64.multi._lib_index[ident]
        got = lib.dummy_frame[int(lib.z_ptr[i]):int(lib.z_ptr[i + 1])]
        assert torch.equal(got, condition_from_energy(m, identifier=ident).ctree_dummy_frame)
        assert torch.equal(got, dummy_frame_atom_flags(m))
    assert int(lib.dummy_frame.sum()) > 0


@pytest.mark.parametrize('edge', [False, True], ids=['uniform', 'disc_edge'])
def test_energy_and_gradient_match_the_member_oracle(alk64, edge):
    X = alk64.X_edge if edge else alk64.X
    assert _close(alk64.multi.energy(X, alk64.batch, alk64.logT), alk64.oracle(X))
    xa = X.clone().requires_grad_(True)
    g, = torch.autograd.grad(alk64.multi.energy(xa, alk64.batch, alk64.logT,
                                                keep_grads=True).sum(), xa)
    xb = X.clone().requires_grad_(True)
    go, = torch.autograd.grad(alk64.oracle(xb, keep_grads=True).sum(), xb)
    assert float((g - go).abs().max()) <= 1e-9 * float(go.abs().max())


def test_prebuilt_reward_is_minus_energy_at_unit_temperature(alk64):
    zero = torch.zeros_like(alk64.logT)
    _, baked = alk64.multi.energy(alk64.X_edge, alk64.batch.clone(), zero, return_exp=True)
    assert _close(alk64.multi.prebuilt_sample_to_reward(baked, 1.0),
                  -alk64.oracle(alk64.X_edge, logT=zero))


def test_float32_matches_the_member_oracle():
    """The production dtype: the dummy is built in float32 on the one-pass path."""
    s = _Set(SET, SIZES, torch.float32)
    X = _to_disc_edge(s, s.X)
    got, want = s.multi.energy(X, s.batch, s.logT), s.oracle(X)
    rel = ((got - want.to(got.dtype)).abs() / want.abs().clamp_min(1.0).to(got.dtype)).max()
    assert float(rel) < 1e-5, float(rel)


def test_a_file_with_moved_dummy_flags_is_refused(alk64):
    """Same molecule, same atoms, same NUMBER of dummy rows, on other atoms: refused."""
    row = int(np.flatnonzero(alk64.assign == SET.index('CC#CCO'))[0])
    b2 = alk64.batch.clone()
    flag = b2.ctree_dummy_frame.clone()
    atoms = torch.nonzero(b2.batch == row).flatten()
    have = atoms[flag[atoms]]
    free = atoms[~flag[atoms] & (b2.ctree_round[atoms] >= 3)]
    flag[have[0]], flag[free[0]] = False, True
    b2.ctree_dummy_frame = flag
    with pytest.raises(RuntimeError, match=f'row {row} .*dummy-frame flags'):
        alk64.multi.energy(alk64.X, b2, alk64.logT)
    b3 = alk64.batch.clone()
    del b3.ctree_dummy_frame
    with pytest.raises(RuntimeError, match='dummy-frame flags'):
        alk64.multi.member_groups(b3)


# ------------------------------------------------------------------ every build site

def test_prior_smoke_parses_and_builds_only_through_the_chart_helper():
    """prior_smoke.py called builder.build without the chart's masks at four sites -- a
    different geometry from the one en.energy builds for every nitrile and alkyne -- and did
    not parse (an `elif` after `except`). Every build there now goes through en._build."""
    src = (Path(__file__).resolve().parents[2] / 'energies' / 'prior_smoke.py').read_text(
        encoding='utf-8')
    tree = ast.parse(src)
    bare = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'build']
    assert bare == [], f'bare builder.build calls at lines {bare}'
