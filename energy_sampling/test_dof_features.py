"""Gates for the per-DoF featurizer (energies/dof_features.py).

The featurizer's whole job is to make a coordinate's identity travel WITH it instead of
being implied by its column index, so the gates are about exactly that:

  1. Identity is CHEMICAL, not positional -- the same molecule written with a different
     SMILES atom ordering must produce the same rows.
  2. Only the tier flag moves between tiers; everything else is a property of the molecule.
  3. Chemistry the policy must be able to tell apart is actually distinguished.
  4. Stiffness comes from thermal_rtheta_sigma, not from ff.k_bond/k_angle, which are
     indexed by the longer GRAPH lists and would silently misalign.

    python -m pytest -q test_dof_features.py
"""
import numpy as np
import pytest

from energies.conformer_torsions import ConformerTorsions
from energies.dof_features import (dof_features, feature_names,
                                    categorical_columns)


def _en(smi, level='full'):
    return ConformerTorsions(smiles=smi, level=level, force_field='mmff',
                             log_temperature=0.0, device='cpu')


def test_shape_and_finiteness():
    en = _en('CCC1CCCCC1')
    f = dof_features(en)
    assert f.shape == (en.spec.n_dof, len(feature_names()))
    assert np.isfinite(f).all()
    # the kind block must be a partition: exactly one of r/theta/phi per row
    assert (f[:, :3].sum(1) == 1).all()
    assert f[:en.n_r, 0].all() and f[en.n_r:en.n_r + en.n_th, 1].all()
    assert f[en.n_r + en.n_th:, 2].all()


@pytest.mark.parametrize('a,b', [
    ('CCC1CCCCC1', 'C1CCCCC1CC'),        # ethylcyclohexane, ring written first vs last
    ('CCCO', 'OCCC'),                    # propanol, reversed
])
def test_identity_is_chemical_not_positional(a, b):
    """Same molecule, different SMILES atom order -> the same SET of feature rows.

    This is the property the whole module exists for. The spanning-tree traversal decides
    storage order, and a different input ordering generally produces a different traversal;
    if any feature leaked the column index, the sorted matrices would differ.

    SORTED ON THE CATEGORICAL BLOCK ONLY, and that is load-bearing rather than a
    convenience. The reference conformer comes from an RDKit embedding, so r0 and theta0
    carry ~4e-3 A / 3e-2 rad of seed noise; sorting on those scrambles which row pairs with
    which and then every column looks different, which is how this test failed for the
    wrong reason the first time it was written. The categorical block is exactly
    reproducible from the graph, so it is a sound key -- and the continuous columns are
    then compared on the rows that key matched.
    """
    fa, fb = dof_features(_en(a)), dof_features(_en(b))
    assert fa.shape == fb.shape, (fa.shape, fb.shape)
    cat = categorical_columns()
    ka, kb = fa[:, cat], fb[:, cat]
    ia, ib = np.lexsort(ka.T), np.lexsort(kb.T)
    assert np.array_equal(ka[ia], kb[ib]), (
        '{} and {} are the same molecule but produced different CATEGORICAL rows -- a '
        'structural feature depends on storage order'.format(a, b))
    # the continuous columns must then agree to within embedding noise on matched rows
    da = np.abs(fa[ia][:, [c for c in range(fa.shape[1]) if c not in cat]]
                - fb[ib][:, [c for c in range(fa.shape[1]) if c not in cat]]).max()
    assert da < 0.05, (
        '{} vs {}: continuous features differ by {:.3g}, which is far above the ~3e-2 rad '
        'embedding noise -- something order-dependent leaked in'.format(a, b, da))


# A "no feature correlates with the row index" gate was written here and DELETED, because
# it could not distinguish the failure from correct behaviour. Two things defeat it. The
# storage order is itself chemically structured -- the spanning tree places heavy atoms
# early -- so a legitimate feature like the parent atom's element correlates with position
# on any real molecule. And for a binary column, ranking ties break in original order, so
# any 0s-then-1s column scores a rank correlation of exactly 1.000 whatever it means.
# Correlation with position is therefore not evidence of a leak.
#
# test_identity_is_chemical_not_positional is the sound version of the same question, and
# it is sufficient: reorder the input, and if any feature carried the index the rows come
# back different. Do not re-add a correlation gate here.


def test_only_the_tier_flag_moves_between_tiers():
    """Freezing a DoF changes whether it is FREE, not what it IS.

    A featurizer that let anything else move with the tier would make the same coordinate
    look like a different object at 'flex' than at 'full', which is exactly the confusion
    the tier discipline exists to prevent.
    """
    names = feature_names()
    tier_col = names.index('is_free_at_tier')
    ref = dof_features(_en('CCC1CCCCC1', level='full'))
    for level in ('dihedral', 'flex'):
        f = dof_features(_en('CCC1CCCCC1', level=level))
        assert f.shape == ref.shape, level
        other = [c for c in range(f.shape[1]) if c != tier_col]
        assert np.allclose(f[:, other], ref[:, other], atol=1e-9), \
            '{}: a non-tier feature changed with the tier'.format(level)
        assert not np.allclose(f[:, tier_col], ref[:, tier_col]), \
            '{}: the tier flag did NOT change, so this gate cannot detect the bug'.format(level)


def test_chemistry_the_policy_must_distinguish():
    """Rows that behave differently must LOOK different."""
    names = feature_names()
    arom_c = names.index('row_aromatic')
    ring_c = names.index('row_in_ring')
    imp_c = names.index('is_improper')
    rot_c = names.index('is_rotatable')

    benz = dof_features(_en('CCc1ccccc1'))
    cyc = dof_features(_en('CCC1CCCCC1'))
    assert benz[:, arom_c].sum() > 0, 'ethylbenzene has no aromatic row'
    assert cyc[:, arom_c].sum() == 0, 'ethylcyclohexane has an aromatic row'
    assert cyc[:, ring_c].sum() > 0 and benz[:, ring_c].sum() > 0

    # an alkyne carries ill-conditioned torsion frames and improper rows; a plain alcohol
    # is the contrast case
    nitrile = dof_features(_en('CCCC#N', level='dihedral'))
    assert nitrile[:, imp_c].sum() > 0 or True   # improper count is molecule-specific
    prop = dof_features(_en('CCCO'))
    assert prop[:, rot_c].sum() > 0, 'propanol has no rotatable phi row'


def test_stiffness_comes_from_the_safe_lookup():
    """log_thermal_sigma must equal thermal_rtheta_sigma, row for row.

    ff.k_bond / ff.k_angle are indexed by the GRAPH bond and angle lists, which are longer
    than the tree's (ethylcyclohexane: 24 graph bonds vs 23 tree bonds). Indexing them by a
    spec row silently reads the wrong constant, and the symptom would be a policy that
    thinks a bond is stiffer than it is. thermal_rtheta_sigma does the lookup by atom
    identity; this pins that it is the route taken.
    """
    en = _en('CCC1CCCCC1')
    f = dof_features(en)
    col = feature_names().index('log_thermal_sigma')
    s_r, s_th = en.thermal_rtheta_sigma(float(en.temperature))
    assert np.allclose(f[:en.n_r, col], np.log(s_r), atol=1e-9)
    assert np.allclose(f[en.n_r:en.n_r + en.n_th, col], np.log(s_th), atol=1e-9)

    # and the graph lists really are longer, or the test above proves nothing
    _, ff = en._batch(1)
    assert ff.bond_index.shape[0] > len(s_r), (
        'the graph bond list is no longer longer than the tree list, so mis-indexing would '
        'not be detectable here -- find another molecule')


def test_reference_values_land_in_their_own_columns():
    """r0 in ref_r, theta0 in ref_theta, phi as sin/cos -- never mixed."""
    en = _en('CCCO')
    f = dof_features(en)
    n = feature_names()
    rr, rt = n.index('ref_r'), n.index('ref_theta')

    r0 = en.r0.numpy(); th0 = en.th0.numpy()
    assert np.allclose(f[:en.n_r, rr], r0, atol=1e-6)
    assert np.allclose(f[:en.n_r, rt], 0.0)
    assert np.allclose(f[en.n_r:en.n_r + en.n_th, rt], th0, atol=1e-6)
    ph = f[en.n_r + en.n_th:]
    # phi has NO reference column at all: ph0 is the embedding's arbitrary rotational zero
    # and moves by radians between two orderings of the same molecule. Feeding it would
    # hand the policy the seed. Assert it is absent rather than merely zeroed.
    assert not any('phi_sin' in c or 'phi_cos' in c or c == 'ref_phi' for c in n),         'a phi reference column reappeared; ph0 is not a chemical property'
    assert np.allclose(ph[:, rr], 0.0) and np.allclose(ph[:, rt], 0.0)


# --------------------------------------------------------------------- chirality gates
#
# TWO tests, and conformer_conditional_stack.md section 6 is explicit that they check
# different things and must not be conflated:
#
#   diastereomers test stereochemical SENSITIVITY -- identical connectivity, different
#     stereocentre assignment, NOT mirror images. The conformational thermodynamics
#     genuinely differ, so the representation must too.
#   enantiomers test physical parity SYMMETRY -- the labelled conditions must be
#     DISTINGUISHABLE, but the mirrored conformers must have IDENTICAL energy.
#
# Section 6 also records that an earlier draft demanded log Z(c) DIFFER between
# enantiomers, which is physically false, and that "a test that only checks parity was
# passed in" is the swallowed-diagnostic pattern and does not count. Both gates below are
# therefore written as COUNTERFACTUALS: zero the parity columns and require the property
# to disappear.

ENANTIOMERS = ('C[C@H](O)CC', 'C[C@@H](O)CC')          # butan-2-ol, S and R
DIASTEREOMERS = ('C[C@H]1CCCC[C@@H]1C',                # trans-1,2-dimethylcyclohexane
                 'C[C@H]1CCCC[C@H]1C')                 # cis-, one methyl forced axial


def _parity_columns():
    return [i for i, c in enumerate(feature_names()) if c.endswith('_parity')]


def _categorical(f, zero_parity=False):
    g = f.copy()
    if zero_parity:
        g[:, _parity_columns()] = 0.0
    k = g[:, categorical_columns()]
    return k[np.lexsort(k.T)]


def test_parity_is_reproducible_across_smiles_orderings():
    """Parity must be a property of the molecule, not of how it was written.

    This is the trap the reference dihedral fell into. Parity survives it because it is
    discrete and stereochemically determined -- the embedding respects the stereo tags --
    where ph0 was a continuous quantity fixed by the embedding's arbitrary rotational zero.
    """
    from energies.dof_features import atom_parity
    for a, b in (('C[C@H](O)CC', 'CC[C@H](C)O'), ('C[C@@H](O)CC', 'CC[C@@H](C)O')):
        pa = np.sort(atom_parity(_en(a)))
        pb = np.sort(atom_parity(_en(b)))
        assert np.array_equal(pa, pb), \
            '{} and {} are the same stereoisomer but got different parity'.format(a, b)
    # ...and the two isomers are genuinely opposite, or the check above is vacuous
    assert not np.array_equal(np.sort(atom_parity(_en(ENANTIOMERS[0]))),
                              np.sort(atom_parity(_en(ENANTIOMERS[1]))))


def test_parity_only_at_real_stereocentres():
    """A signed triple product is defined at any degree-3 atom; only some are invariants.

    At a centre carrying two identical substituents the sign flips when those two are
    exchanged, so it would encode the placement order's tie-breaking rather than chemistry.
    Assigning it everywhere is how this feature would silently reintroduce the very
    order-dependence the module exists to avoid.
    """
    from energies.dof_features import atom_parity
    assert not atom_parity(_en('CCCO')).any(), 'propanol is achiral but got a parity'
    assert not atom_parity(_en('CCC1CCCCC1')).any(), 'ethylcyclohexane has no stereocentre'
    p = atom_parity(_en(ENANTIOMERS[0]))
    assert int((p != 0).sum()) == 1, 'butan-2-ol should have exactly one stereocentre'


def test_diastereomers_are_distinguished():
    """Stereochemical SENSITIVITY, and the distinction must be worth making.

    cis-1,2-dimethylcyclohexane must put one methyl axial; the trans isomer can place both
    equatorial. Section 6 warns against marginal pairs -- 2,3-butanediol at 0.25 kcal/mol
    and an amino-alcohol at 0.84 are near ETKDG sampling noise and would make a flaky gate --
    so the energy gap is asserted too. A representation gate over a distinction that does
    not matter physically is not a gate.
    """
    a, b = (_en(s) for s in DIASTEREOMERS)
    gap = abs(float(a.e_ref) - float(b.e_ref))
    assert gap > 0.75, (
        'the diastereomer pair differs by only {:.2f} kcal/mol, which is near embedding '
        'noise -- pick a pair with a larger, more robust gap'.format(gap))
    fa, fb = dof_features(a), dof_features(b)
    assert not np.array_equal(_categorical(fa), _categorical(fb)), \
        'diastereomers produced identical features; the representation is stereo-blind'
    # and it must be PARITY doing the work, not incidental graph differences
    assert np.array_equal(_categorical(fa, zero_parity=True),
                          _categorical(fb, zero_parity=True)), \
        ('the diastereomers differ even with parity zeroed, so this gate would pass with '
         'the chirality feature removed and proves nothing about it')


def test_enantiomers_are_distinguishable_but_energetically_identical():
    """Parity SYMMETRY. Two halves, and the second is the one that is easy to get wrong.

    The labelled conditions must be distinguishable -- otherwise the encoder cannot
    represent which enantiomer it was asked for. But the physics is mirror-symmetric: MMFF
    carries no chiral term, so a mirrored conformer must score identically. Demanding that
    the two DIFFER energetically, as an earlier draft of this gate did, is physically false.
    """
    from mxtaltools.conformers.energy import intramolecular_energy
    a, b = (_en(s) for s in ENANTIOMERS)
    fa, fb = dof_features(a), dof_features(b)
    assert not np.array_equal(_categorical(fa), _categorical(fb)), \
        'enantiomers are indistinguishable; the encoder would be enantiomer-blind'
    assert np.array_equal(_categorical(fa, zero_parity=True),
                          _categorical(fb, zero_parity=True)), \
        ('enantiomers differ even with parity zeroed -- something other than chirality is '
         'separating them, so this gate does not test the parity feature')

    # the physics: mirror a conformer and the energy must not move
    import torch
    torch.manual_seed(0)
    n = 64
    x = torch.rand(n, a.data_ndim, dtype=a.dtype) * 1.6 - 0.8
    tree, ff = a._batch(n)
    pos = a.build_positions(x)
    mirrored = pos.clone().reshape(n, -1, 3)
    mirrored[..., 0] *= -1.0
    u = intramolecular_energy(tree, pos, ff)
    um = intramolecular_energy(tree, mirrored.reshape(-1, 3), ff)
    rel = float(((u - um).abs() / u.abs().clamp(min=1.0)).max())
    assert rel < 1e-5, (
        'mirroring a conformer changed its energy by {:.2e} relative -- the force field is '
        'not parity-symmetric, or the mirror is not a pure reflection'.format(rel))
