"""The contract around the transverse chart: refusal, features, and the disc.

Three properties that were all UNTESTED when the chart landed, each for a different reason:

  * THE REFUSAL had exactly one enforcement point (a guard in build_conformer_conditions.py)
    and no test at all -- deleting those four lines broke nothing in the suite. It is the
    invariant that makes partial coverage safe to ship, so it was the half with no gate.
  * THE FEATURES were wrong on exactly the rows the chart rescues, and wrong in a way no
    existing assertion could see: the geometry and the density read `energy._ref_dof`, while
    the policy read `th0` and got pi for a coordinate whose reference is ~0. Nothing fails;
    the policy is simply trained against a misplaced origin.
  * THE DISC is bounded by a wall that did not know it was a disc. The chart is injective only
    on rho < pi; past it the map is an orientation-reversing double cover, log sinc is -inf,
    and one row takes the batch's TB loss to infinity.

`CC#N` and `CCC#N` are the COVERED molecules (their linear bend becomes a transverse pair);
`CC#C` and `CC#CCO` are the UNCOVERED contrast (frame seed and collinear reference frames), and
they must stay refused. Keeping both in every test is what stops "it works" from meaning "it
works on the case I chose".
"""
import numpy as np
import pytest
import torch

from energies import dof_features as DF
from energies.conformer_data import condition_from_energy
from energies.conformer_torsions import ConformerTorsions

COVERED = ['CC#N', 'CCC#N']
UNCOVERED = ['CC#C', 'CC#CCO']
PLAIN = ['CCCO', 'CCCCO']


@pytest.fixture(autouse=True)
def _float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _en(smiles, level='full', **kw):
    kw.setdefault('force_field', 'mmff')
    return ConformerTorsions(smiles=smiles, device='cpu', level=level, **kw)


def _state_at_rho(en, rho, n=1):
    """A state whose first transverse pair sits at radius ``rho``, exactly.

    v is driven to exactly zero rather than left at its reference: v0 is small but NOT zero
    (~1e-6 rad of embedding residue), and rho = hypot(u, v), so leaving it alone makes rho
    differ from the requested u by that residue -- enough to fail a 1e-9 comparison and look
    like a units bug when it is arithmetic.
    """
    x = torch.zeros(n, en.data_ndim, dtype=torch.float64)
    x[:, en._tv_u_cols[0]] = (rho - float(en._tv_u_ref[0])) / float(en._tv_u_scale[0])
    x[:, en._tv_v_cols[0]] = (0.0 - float(en._tv_v_ref[0])) / float(en._tv_v_scale[0])
    return x


def _pair(en):
    """``(u_row, v_row, angle_row)`` for the first transverse pair, in DoF numbering."""
    j = int(np.flatnonzero(np.asarray(en.transverse_angles, dtype=bool))[0])
    p = int(np.asarray(en.transverse_partner)[j])
    return en.n_r + j, en.n_r + en.n_th + p, j


# ------------------------------------------------------------------ the refusal

@pytest.mark.parametrize('smiles', UNCOVERED)
def test_an_incomplete_full_chart_is_refused_by_the_energy(smiles):
    """Not just by the conditions builder -- that covered one of eight construction paths.

    A config naming one of these at `full` used to train silently, and its run summary, its
    checkpoint and its conditions file all said 'full'.
    """
    with pytest.raises(ValueError, match='complete chart at level'):
        _en(smiles, 'full')


@pytest.mark.parametrize('smiles', UNCOVERED)
def test_the_refusal_names_the_shortfall_and_the_opt_out(smiles):
    """An error that says only "refused" makes the next person delete the guard."""
    with pytest.raises(ValueError) as exc:
        _en(smiles, 'full')
    msg = str(exc.value)
    assert '3N-6' in msg and 'allow_constrained' in msg
    assert 'not covered by the transverse pair' in msg
    # and it must not blame the molecule
    assert 'chart limitation, not a rigid molecule' in msg


@pytest.mark.parametrize('smiles', UNCOVERED)
def test_partial_coverage_is_allowed_when_explicit(smiles):
    """Deliberate study of a constrained chart stays possible, and is recorded."""
    en = _en(smiles, 'full', allow_constrained=True)
    assert en.allow_constrained is True
    assert en.constrained_rows > 0
    assert en.data_ndim < 3 * en.spec.n_atoms - 6
    assert 'CONSTRAINED' in en.describe()


@pytest.mark.parametrize('smiles', COVERED)
def test_a_covered_molecule_is_not_refused_and_is_complete(smiles):
    en = _en(smiles, 'full')
    assert en.constrained_rows == 0
    assert en.uncovered_linear_angles == 0
    assert en.data_ndim == 3 * en.spec.n_atoms - 6
    assert 'CONSTRAINED' not in en.describe()
    assert 'TRANSVERSE' in en.describe()


@pytest.mark.parametrize('smiles', UNCOVERED + COVERED)
def test_the_lower_tiers_are_unchanged_by_the_chart(smiles):
    """`torsion` and `dihedral` cannot drive a whole pair, so they keep the old treatment.

    Refusing them instead would have been a behaviour change nobody asked for; the transverse
    pair is a `flex` / `full` feature because that is where the target is.
    """
    try:
        en = _en(smiles, 'dihedral', allow_constrained=True)
    except ValueError as exc:
        # 'CC#C' has no free dihedral once its linear frames are excluded. That predates the
        # transverse chart and is a different statement; skip rather than assert on it.
        pytest.skip(f'{smiles} does not build at dihedral: {exc}')
    assert int(np.asarray(en.transverse_angles).sum()) == 0
    assert not (np.asarray(en._free_block) == 3).any()


# ------------------------------------------------------------------ the features

@pytest.mark.parametrize('smiles', COVERED)
def test_a_transverse_row_is_not_labelled_an_angle_or_a_dihedral(smiles):
    en = _en(smiles)
    f = DF.dof_features(en)
    ix = {n: i for i, n in enumerate(DF.feature_names())}
    u, v, _ = _pair(en)
    for r, own in ((u, 'kind_bend_u'), (v, 'kind_bend_v')):
        assert f[r, ix[own]] == 1.0
        assert f[r, ix['kind_theta']] == 0.0 and f[r, ix['kind_phi']] == 0.0
        assert f[r, ix['kind_r']] == 0.0
    # u and v are NOT interchangeable -- they lie along different frame vectors
    assert f[u, ix['kind_bend_u']] != f[v, ix['kind_bend_u']]
    assert (f[:, :5].sum(1) == 1).all(), 'every row must carry exactly one kind'


@pytest.mark.parametrize('smiles', COVERED)
def test_the_policy_gets_the_chart_reference_not_the_polar_one(smiles):
    """The bug this file exists for: ref_theta ~ pi on a coordinate whose reference is ~0.

    Geometry and density were always right (both read `_ref_dof`); only the policy was
    misinformed, so nothing failed and no existing assertion could see it.
    """
    en = _en(smiles)
    f = DF.dof_features(en)
    ix = {n: i for i, n in enumerate(DF.feature_names())}
    ref = en._ref_dof.detach().cpu().numpy()
    th0 = en.th0.detach().cpu().numpy()
    u, v, j = _pair(en)
    for r in (u, v):
        assert f[r, ix['ref_bend']] == pytest.approx(float(ref[r]), abs=1e-12)
        assert f[r, ix['ref_theta']] == 0.0
        assert f[r, ix['ref_r']] == 0.0
    # the counterfactual: the polar reference really is ~pi, i.e. the old feature was not
    # merely imprecise but off by the whole range of the coordinate
    assert abs(float(th0[j])) > 3.0
    assert abs(float(ref[u])) < 0.01


@pytest.mark.parametrize('smiles', COVERED)
def test_both_components_carry_the_bend_stiffness_not_a_torsion_jitter(smiles):
    """v used to take the phi branch's sigma -- a rotation jitter handed over for a bend."""
    en = _en(smiles)
    f = DF.dof_features(en)
    ix = {n: i for i, n in enumerate(DF.feature_names())}
    _, s_th = en.thermal_rtheta_sigma(float(en.temperature))
    u, v, j = _pair(en)
    want = float(np.log(s_th[j]))
    for r in (u, v):
        assert f[r, ix['log_thermal_sigma']] == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize('smiles', COVERED)
def test_a_bend_is_not_a_bond_rotation(smiles):
    """is_improper / is_group_member / is_rotatable are rotation-about-a-bond semantics."""
    en = _en(smiles)
    f = DF.dof_features(en)
    ix = {n: i for i, n in enumerate(DF.feature_names())}
    u, v, _ = _pair(en)
    for r in (u, v):
        for flag in ('is_improper', 'is_group_member', 'is_rotatable'):
            assert f[r, ix[flag]] == 0.0, f'{flag} set on a transverse row'


@pytest.mark.parametrize('smiles', COVERED)
def test_the_pair_shares_the_four_atom_placement_frame(smiles):
    """u's direction is fixed by atom a, which the 3-atom angle frame omits entirely.

    Both transverse directions are built from ``pb - pa`` (geometry.place_nerf_transverse), so
    the angle frame under-determines u. Sharing the torsion frame also makes the two rows of a
    pair identical in every atom column and different only in the kind bit and the reference,
    which is exactly the truth about them.
    """
    en = _en(smiles)
    f = DF.dof_features(en)
    names = DF.feature_names()
    atom_cols = [i for i, n in enumerate(names) if n.startswith('a')]
    u, v, _ = _pair(en)
    assert np.allclose(f[u, atom_cols], f[v, atom_cols])
    assert f[u, names.index('a3_present')] == 1.0, 'the u row lost its fourth frame atom'

    atoms, mask = DF.free_dof_atom_index(en)
    cols = np.flatnonzero(np.asarray(en._free_block) == 3)
    frames = atoms.reshape(atoms.shape[0], -1)[cols] if atoms.shape[1] == 1 else atoms[cols]
    for fr in np.asarray(frames).reshape(len(cols), -1):
        assert len(set(int(a) for a in fr)) == 4, (
            f'a transverse column still pads a repeated atom: {fr.tolist()}')


@pytest.mark.parametrize('smiles', PLAIN)
def test_a_molecule_with_no_linear_centre_is_untouched(smiles):
    """The change must be additive: three new columns, all zero, nothing else moves."""
    en = _en(smiles)
    f = DF.dof_features(en)
    ix = {n: i for i, n in enumerate(DF.feature_names())}
    assert f[:, ix['kind_bend_u']].sum() == 0.0
    assert f[:, ix['kind_bend_v']].sum() == 0.0
    assert f[:, ix['ref_bend']].sum() == 0.0
    assert (f[:, :5].sum(1) == 1).all()


def test_the_tier_flag_did_not_shift_when_a_column_was_appended():
    """The silent one. `is_free_at_tier` used to be positioned by counting back from the end.

    Appending `ref_bend` would have moved it one slot with every shape still correct and every
    value still finite -- so it is pinned against `free_mask`, by name, rather than by index.
    """
    en = _en('CCC#N')
    f = DF.dof_features(en)
    ix = {n: i for i, n in enumerate(DF.feature_names())}
    assert np.array_equal(f[:, ix['is_free_at_tier']].astype(bool),
                          np.asarray(en.free_mask, dtype=bool))


# ------------------------------------------------------------------ the disc

@pytest.mark.parametrize('smiles', COVERED)
def test_the_radial_wall_is_zero_everywhere_the_sampler_has_ever_been(smiles):
    """A preference, not a clamp: identically zero inside rho_wall.

    rho_wall = 1.0 rad against a measured maximum of 0.682 over 947,022 transverse rows of a
    400-epoch run (itself equal to the seed prior's own maximum), and a physical 99.99th
    percentile of 0.53. The term exists for a regime nothing has reached.
    """
    en = _en(smiles)
    for rho in (0.0, 0.3, 0.682, 1.0):
        x = _state_at_rho(en, rho)
        assert en.bounding_coeff * max(0.0, rho - en.rho_wall) ** 2 == 0.0
        assert float(torch.sqrt(en._transverse_rho2(x))[0, 0]) == pytest.approx(rho, abs=1e-9)


@pytest.mark.parametrize('smiles', COVERED)
def test_rho_is_measured_in_chart_radians_not_state_units(smiles):
    """The two differ by 1/delta_theta_max, which is a plausible number for the wrong thing.

    A first implementation read the raw state columns as u and v, which put the wall on the
    box coordinate and made the crossing counter fire at rho = 2.0 -- inside a disc whose
    boundary is pi.
    """
    en = _en(smiles)
    x = _state_at_rho(en, 0.0, n=3)
    x[:, en._tv_u_cols[0]] = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    rho = torch.sqrt(en._transverse_rho2(x))[:, 0]
    _, th, _ = en.dof_from_state(x)
    u = th[:, int(np.flatnonzero(np.asarray(en.transverse_angles))[0])]
    # rho is |u| in CHART radians; the raw state column is u / delta_theta_max
    assert torch.allclose(rho, u.abs(), atol=1e-9)
    # u = u0 + delta_theta_max * x, so the reference belongs in the expected value; the
    # point of the check is the SCALE, and state units would read 2.0 rather than ~1.0
    want = abs(float(en._tv_u_ref[0]) + 2.0 * en.delta_theta_max)
    assert float(rho[2]) == pytest.approx(want, abs=1e-9)
    assert float(rho[2]) < 2.0, 'state units leaked into rho'


@pytest.mark.parametrize('smiles', COVERED)
def test_leaving_the_disc_is_counted_and_named(smiles):
    """Past rho = pi the reward is -inf and the TB loss infinite, with no finiteness guard.

    The probability is remote (extrapolated 1e-15 per row at the current operating point, and
    the wall now makes it remoter) but the consequence is a dead run with no attribution. A
    counter is what turns that into an attributable event.
    """
    en = _en(smiles)
    inside = torch.zeros(4, en.data_ndim, dtype=torch.float64)
    assert en.transverse_crossings(inside) == 0

    out = torch.cat([_state_at_rho(en, 3.5), _state_at_rho(en, 4.0)])
    assert en.transverse_crossings(out) == 2
    # and the reward really is the thing being guarded against
    assert bool(torch.isinf(en.energy(out)).all())


@pytest.mark.parametrize('smiles', PLAIN)
def test_a_molecule_with_no_pair_has_no_wall_and_no_counter(smiles):
    en = _en(smiles)
    x = torch.rand(4, en.data_ndim, dtype=torch.float64) * 2 - 1
    assert en._transverse_t is None
    assert en.transverse_crossings(x) == 0
    assert float(en.bounding_energy(x, 1.0).sum()) == 0.0


# ------------------------------------------------------------------ multi-molecule full

def test_the_multi_molecule_energy_rewards_full_like_the_single_one():
    """`flex`/`full` on the CONDITIONAL route, where log J moves with the sample.

    `MultiConformerTorsions.prebuilt_sample_to_reward` used to raise above `dihedral`, which
    made `full` unreachable on the entire conditional route -- the conditional arm is always a
    multi-molecule energy, and the anchor-buffer seed calls this before the first training
    step. The invariant: a set whose members are all the SAME molecule must reward exactly as
    the single-molecule energy does. Anything else is a dispatch error, and a dispatch error
    here is a per-condition log Z error, which is the quantity conditional training is trying
    to learn.
    """
    from energies.conformer_data import attach_states, bake_energies, condition_from_energy
    from energies.multi_conformer import MultiConformerTorsions

    idents = [f'p{i}' for i in range(4)]
    single = _en('CCCO')
    multi = MultiConformerTorsions(smiles=['CCCO'] * 4, identifiers=idents, device='cpu',
                                   level='full', force_field='mmff')
    assert multi.n_charts == 4
    assert multi.log_jacobian_const is None, 'full must have a STATE-DEPENDENT Jacobian'

    x = torch.rand(8, single.data_ndim, dtype=torch.float64) * 2 - 1
    e = bake_energies(single, x)
    base = condition_from_energy(single, identifier=idents[0])
    batch = None
    for i, ident in enumerate(idents):        # >= 2 states per attach_states call
        m = base.clone()
        m.identifier = ident
        part = attach_states(m, x[2 * i:2 * i + 2], e[2 * i:2 * i + 2], identifier=ident,
                             periodic=single.periodic_dims)
        batch = part if batch is None else batch.append_batch(part)

    got = multi.prebuilt_sample_to_reward(batch, 1.0)
    want = single.prebuilt_sample_to_reward(batch, 1.0)
    assert torch.isfinite(got).all()
    assert float((got - want).abs().max()) < 1e-9
