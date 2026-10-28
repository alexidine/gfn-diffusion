"""`lock_stereo_nitrogen`: a tagged stereo nitrogen as a locked stereo element.

A STEREO NITROGEN (energies/stereo_lock.py) is an N with three single bonds to non-hydrogen
atoms that RDKit reports as a tetrahedral element on the parsed SMILES: an N in a
three-membered ring or a bridgehead N. `ConformerTorsions(lock_stereo_nitrogen=True)` makes
the configuration of one the SMILES tags part of the condition's identity and locks it;
False, the default, changes nothing.

Each claim is checked at level `full`, force field `mmff`, `stereo_coeff` 300, on
1,2-dimethylaziridine (`AZ_*`: a real invertomer pair), on the same ring carrying an ordinary
dimethylamino group (`MIX_*`), on a fused aziridine and a caged bridgehead N (one geometric
side), and on molecules the option must not touch (an aziridine N-H, an acyclic amine, an
amide, a molecule with no nitrogen). The database tests read one shard of the QM9 conformer
database and are skipped where that file is absent (GFN_CONFORMER_DB_SHARD names another).
"""
import inspect
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from energies import stereo_lock as sl
from energies.conformer_data import bake_energies, condition_from_energy
from energies.conformer_torsions import ChartRefused, ConformerTorsions
from energies.invertible_centres import (LOCKED, centre_table, invertible_centres,
                                         reference_side)
from test_multi_energy_vectorized import _default_dtype, _Set

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

HERE = Path(__file__).resolve().parents[2]
LOCK = dict(device='cpu', level='full', force_field='mmff', stereo_coeff=300.0)
# 1,2-dimethylaziridine: one carbon configuration, its two nitrogen configurations, and the
# SMILES the option-off route names the pair by
AZ_A, AZ_B, AZ_OPEN = 'C[C@@H]1C[N@]1C', 'C[C@@H]1C[N@@]1C', 'C[C@@H]1CN1C'
# the same ring with an ordinary amine on a side chain
MIX_A, MIX_B, MIX_OPEN = ('CN(C)C[C@@H]1C[N@]1C', 'CN(C)C[C@@H]1C[N@@]1C',
                          'CN(C)C[C@@H]1CN1C')
FUSED = 'C1C[C@@H]2C[N@]2C1'             # 1-azabicyclo[3.1.0]hexane: the N at the ring fusion
FUSED_BAD = 'C1C[C@@H]2C[N@@]2C1'        # its inside-out twin, which does not embed
CAGE = 'N#C[C@@H]1C[C@H]2C[N@]1C2'       # 1-azabicyclo[2.1.1]hexane: a bridgehead N
SYM_CAGE = 'CCCOC12CN(C1)C2'             # a bridgehead N no enumeration distinguishes
# molecules without a stereo nitrogen the SMILES could tag
PLAIN = ['CCN(C)C', 'C[C@H]1CN1', 'CC(=O)N1C[C@@H]1C', 'CC(=O)N(C)C', 'C[C@H](O)CC=O', 'CCCO',
         SYM_CAGE]
SHARD = Path(os.environ.get('GFN_CONFORMER_DB_SHARD',
                            'D:/crystal_datasets/qm9_db_sep29/shard_0000_of_0400.pt'))


def _member(smi, on=True, **kw):
    with _default_dtype(torch.float64):
        return ConformerTorsions(smiles=smi, lock_stereo_nitrogen=on, **{**LOCK, **kw})


def _omitted(smi, **kw):
    """The member with the argument OMITTED, not passed as False."""
    with _default_dtype(torch.float64):
        return ConformerTorsions(smiles=smi, **{**LOCK, **kw})


def _slot(en):
    s = np.empty(int(en.spec.n_atoms), dtype=np.int64)
    s[np.asarray(en.spec.perm)] = np.arange(int(en.spec.n_atoms))
    return s


def _n_element(en):
    """(element index, slot) of the one locked stereo nitrogen of `en`."""
    (a,) = en.stereo_nitrogen_atoms
    s = int(_slot(en)[a])
    return en.stereo.element_of(s), s


def _states(en, n=16, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, en.data_ndim, generator=g, dtype=torch.float64) * 2 - 1) * scale


def _rd_positions(en):
    return np.asarray(en.mol.GetConformer().GetPositions(), dtype=np.float64)


def _through_plane(pos, centre, nbrs, scale):
    """`pos` with atom `centre` moved along the normal of its three neighbours' plane so that
    its height above the plane is `scale` times what it was (-1: its mirror image)."""
    p = np.array(pos, dtype=np.float64)
    a, b, c = (p[k] for k in nbrs)
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n)
    h = float(np.dot(p[centre] - a, n))
    p[centre] = p[centre] + (scale - 1.0) * h * n
    return p


@pytest.fixture(scope='module')
def prior():
    p = Path(os.environ.get('GFN_CONFORMER_PRIOR', HERE / 'conformer_prior_v2.pt'))
    if not p.exists():
        pytest.skip(f'fitted InternalPrior not found at {p}')
    return torch.load(p, weights_only=False)


@pytest.fixture(scope='module')
def az():
    return _member(AZ_A), _member(AZ_B)


# ------------------------------------------------------------------ the rule

@pytest.mark.parametrize('smi,want', [
    ('CN1CC1C', [1]),                  # N-methyl aziridine with distinct ring carbons
    ('C1CC2CN2C1', [4]),               # fused aziridine
    ('C1CN2CCC1CC2', [2]),             # quinuclidine: a bridgehead N (never stereogenic)
    ('CN1CC1', []),                    # symmetry: the two ring carbons are one
    ('N1CC1C', []),                    # an N-H: no SMILES can tag it
    ('CC(=O)N1CC1C', []),              # amide-like aziridine N
    ('CCN(C)C', []), ('CN1CCCC1C', []), ('C1CC2CCN2C1', []),   # ordinary and ring-fusion amines
])
def test_which_nitrogens_are_stereo_nitrogens(smi, want):
    from rdkit import Chem
    assert sl.stereo_nitrogens(Chem.MolFromSmiles(smi)) == want
    flagged = [e['atoms'][0] for e in sl.tagged_elements(smi) if e['stereo_nitrogen']]
    assert flagged == want


def test_an_nh_tag_does_not_survive_the_parse():
    """Why an aziridine N-H is outside the rule: RDKit drops the tag, so no identifier could
    carry its configuration."""
    from rdkit import Chem
    for s in ('C[C@H]1C[N@H]1', 'C[C@H]1C[N@@H]1'):
        assert Chem.MolToSmiles(Chem.MolFromSmiles(s)) == 'C[C@H]1CN1'


@pytest.mark.parametrize('smi', ['C[C@H](O)CC=O', 'C[C@@H](N)C(=O)O', 'O[C@H]1CC[C@@H](O)CC1',
                                 'C[C@@H]1C[N@]1C', 'C[C@H](F)[C@@H](C)O'])
def test_the_tag_convention_is_rdkits_own(smi):
    """`nitrogen_tag_sign` against RDKit's 3D perception where it has one (carbon): the tag
    it writes on a centre is the sign of the chiral volume of the centre's first three
    neighbours in bond order. This is the map a nitrogen's tag is verified through."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3()
    p.randomSeed = 0
    assert AllChem.EmbedMolecule(m, p) == 0
    with sl.pinned_perception():
        Chem.AssignAtomChiralTagsFromStructure(m, replaceExistingTags=True)
    pos = m.GetConformer().GetPositions()
    seen = 0
    for a in m.GetAtoms():
        want = sl.nitrogen_tag_sign(a)
        if want == 0:
            continue
        nb = sl.bond_order_neighbours(a)
        u, v, w = (pos[k] - pos[a.GetIdx()] for k in nb[:3])
        assert np.sign(np.dot(u, np.cross(v, w))) == want, (smi, a.GetIdx())
        seen += 1
    assert seen >= 1


def test_quad_sign_follows_the_permutation_parity():
    assert sl.quad_sign_of_tag(1, [2, 5, 9]) == 1
    assert sl.quad_sign_of_tag(1, [5, 2, 9]) == -1
    assert sl.quad_sign_of_tag(1, [5, 9, 2]) == 1
    assert sl.quad_sign_of_tag(-1, [9, 5, 2]) == 1
    with pytest.raises(ValueError):
        sl.quad_sign_of_tag(1, [1, 1, 2])


@pytest.mark.parametrize('smi', ['C[C@H](O)CC=O', 'C/C=C/C(C)=O', 'O[C@H]1CC[C@@H](O)CC1',
                                 '[H]/N=C/C', 'C[C@H]1CN1'])
def test_the_nitrogen_aware_perception_is_rdkits_where_there_is_no_stereo_nitrogen(smi):
    """`perceive_with_nitrogens` re-runs RDKit's three perception steps; without a nitrogen to
    add it must write the isomer RDKit's own call writes."""
    from rdkit import Chem
    en = _omitted(smi)
    with sl.pinned_perception():
        h = sl.perceive_with_nitrogens(en.mol)
        got = Chem.MolToSmiles(sl._strip_invertible(h))
    assert got == sl.realised_isomer(en.mol) == en.stereo_isomer


# ------------------------------------------------------------------ off is the old member

@pytest.mark.parametrize('smi', PLAIN + [AZ_OPEN, MIX_OPEN])
def test_off_is_the_member_without_the_argument(smi, prior):
    a, b = _omitted(smi), _member(smi, on=False)
    assert a.lock_stereo_nitrogen is False and a.stereo_nitrogen_atoms == ()
    for name in ('kind', 'key', 'quad', 'sign', 'margin', 'lo', 'stereocentre'):
        assert np.array_equal(getattr(a.stereo, name), getattr(b.stereo, name)), name
    # no element on a three-neighbour atom
    deg = np.bincount(np.asarray(a.bond_index_slot).reshape(-1), minlength=a.spec.n_atoms)
    tet = a.stereo.key[a.stereo.kind == sl.TETRAHEDRAL]
    assert (deg[tet] == 4).all()
    assert a.stereo_isomer == b.stereo_isomer and a.stereo_input == b.stereo_input
    x = _states(a)
    one = torch.tensor(1.0, dtype=torch.float64)
    assert torch.equal(a.potential_energy(x, one), b.potential_energy(x, one))
    xa, sa = a.sample_prior_states(prior, 64, np.random.default_rng(3), report=False)
    xb, sb = b.sample_prior_states(prior, 64, np.random.default_rng(3), report=False)
    assert torch.equal(xa, xb) and sa['invertible_centres'] == sb['invertible_centres']


@pytest.mark.parametrize('smi', PLAIN)
def test_a_smiles_without_an_n_tag_is_the_same_member_under_the_option(smi, prior):
    """The claim a database read across the two values rests on
    (build_conformer_database.IDENTITY_KWARGS)."""
    import build_conformer_conditions as bcc
    from energies.dof_features import atom_parity
    a, b = _omitted(smi), _member(smi, on=True)
    assert b.stereo_nitrogen_atoms == ()
    for name in ('kind', 'key', 'quad', 'sign', 'margin', 'lo', 'stereocentre'):
        assert np.array_equal(getattr(a.stereo, name), getattr(b.stereo, name)), name
    assert a.stereo_isomer == b.stereo_isomer and a.stereo_input == b.stereo_input
    assert torch.equal(a.ref_pos, b.ref_pos)
    x = _states(a)
    one = torch.tensor(1.0, dtype=torch.float64)
    assert torch.equal(a.potential_energy(x, one), b.potential_energy(x, one))
    xa, sa = a.sample_prior_states(prior, 64, np.random.default_rng(3), report=False)
    xb, sb = b.sample_prior_states(prior, 64, np.random.default_rng(3), report=False)
    assert torch.equal(xa, xb) and sa['invertible_centres'] == sb['invertible_centres']
    assert np.array_equal(atom_parity(a), atom_parity(b))
    ga, gb = condition_from_energy(a, identifier=smi), condition_from_energy(b, identifier=smi)
    for f in sl.STEREO_FIELDS:
        assert torch.equal(getattr(ga, f), getattr(gb, f)), f
    # and the builder's identity and enumeration of it
    assert bcc.smiles_identity(smi) == bcc.smiles_identity(smi, True)
    assert bcc.stereoisomer_classes(smi) == bcc.stereoisomer_classes(smi, lock_nitrogen=True)


@pytest.mark.parametrize('smi', [AZ_A, MIX_B, FUSED])
def test_off_refuses_a_tagged_nitrogen_as_before(smi):
    import build_conformer_conditions as bcc
    with pytest.raises(ChartRefused) as exc:
        _omitted(smi)
    assert exc.value.code == 'stereo_unsupported'
    # and with the lock off the builder's own refusal stands
    with _default_dtype(torch.float64):
        with pytest.raises(bcc.MemberRefused) as exc:
            bcc.build_member(smi, smi, {'level': 'full', 'force_field': 'mmff'}, carrier=True)
    assert exc.value.code == 'stereo_n_tagged'


def test_the_option_needs_the_lock_and_a_boolean():
    for smi in (AZ_A, 'CCCO'):
        with pytest.raises(ValueError, match='lock_stereo_nitrogen with stereo_coeff 0'):
            _member(smi, stereo_coeff=0.0)
    with pytest.raises(ValueError, match='must be true or false'):
        _member('CCCO', on='yes')


def test_the_argument_reaches_members_and_the_canonical_config_declares_it_off():
    import yaml

    import build_conformer_references as bcr
    assert 'lock_stereo_nitrogen' in inspect.signature(ConformerTorsions.__init__).parameters
    kw = bcr.member_kwargs({'level': 'full', 'lock_stereo_nitrogen': True, 'temperature': 1.0})
    assert kw == {'level': 'full', 'lock_stereo_nitrogen': True}
    with open(HERE / 'configs' / 'conformer_mk.yaml', encoding='utf-8') as f:
        ec = yaml.safe_load(f)['energy_config']
    assert ec['lock_stereo_nitrogen'] is False


# ------------------------------------------------------------------ the lock element

def test_a_tagged_aziridine_nitrogen_gets_one_element_on_its_three_neighbours(az):
    for en in az:
        e, s = _n_element(en)
        off = _omitted(AZ_OPEN)
        assert en.stereo.n == off.stereo.n + 1
        assert int(en.stereo.kind[e]) == sl.TETRAHEDRAL and bool(en.stereo.stereocentre[e])
        nb = sorted({int(v) for u, v in np.asarray(en.bond_index_slot).T if int(u) == s}
                    | {int(u) for u, v in np.asarray(en.bond_index_slot).T if int(v) == s})
        assert len(nb) == 3 and en.stereo.quad[e].tolist() == nb + [s]
        assert float(en.stereo.margin[e]) > 0.6 and float(en.stereo.lo[e]) == sl.LO_MAX[1]
        assert 'STEREO NITROGENS LOCKED' in en.describe()


def test_the_lock_is_zero_at_the_reference_and_large_at_the_inverted_nitrogen(az):
    for en in az:
        e, s = _n_element(en)
        ref = en.ref_pos.detach().double().reshape(1, -1, 3)
        assert float(en.stereo.lock_energy(ref, en.stereo_coeff)) == 0.0
        inv = torch.as_tensor(_through_plane(ref[0].numpy(), s, en.stereo.quad[e][:3], -1.0))
        v0, v1 = en.stereo.values(ref)[0, e], en.stereo.values(inv[None])[0, e]
        assert float(v1) == pytest.approx(-float(v0), abs=1e-9)
        m = float(en.stereo.margin[e])
        want = en.stereo_coeff * (float(en.stereo.lo[e]) + m) ** 2
        assert float(en.stereo.lock_energy(inv[None], en.stereo_coeff)) == pytest.approx(want)
        assert want > 150.0
        # every other element is untouched by moving the nitrogen alone? No: the ring carbons'
        # quads may hold it. The claim is only the nitrogen's own term, measured above.


def test_the_other_invertomers_reference_is_on_the_wrong_side_of_this_chart(az):
    """Through the chart and the potential, not the table alone: B's reference conformer,
    measured into A's state, pays the lock and is what the database screens out."""
    import build_conformer_database as db
    import build_conformer_references as bcr
    a, b = az
    assert np.array_equal(a.spec.perm, b.spec.perm) and np.array_equal(a.spec.z, b.spec.z)
    for own, other in ((a, b), (b, a)):
        x = db.states_of_rows(own, other.ref_pos.detach().double().numpy()[None])
        x0 = torch.zeros_like(x)
        assert bool(db.lock_wrong_side(own, x)[0]) and not bool(db.lock_wrong_side(own, x0)[0])
        e = bake_energies(own, torch.cat([x0, x]))
        u_other = float(bake_energies(other, torch.zeros_like(x))[0])
        # the same geometry under the two locks: only the lock differs, and it is the
        # nitrogen's term (the carbon elements keep their signs: one carbon configuration)
        gap = float(e[1]) - u_other
        assert gap > 100.0, gap
        pin = bcr.condition_stereo(own)
        (n_atom,) = own.stereo_nitrogen_atoms
        assert n_atom in pin['atoms'] and pin['pin'] == 'full'
        assert bcr.stereo_labels(own, x0, pin)[0] == pin['target']
        assert bcr.stereo_labels(own, x, pin)[0] != pin['target']
        assert bcr.stereo_signatures(own, x)[0] == other.stereo_input
        _, reason = db.screen(own, pin, torch.cat([x0, x]), np.zeros(2, dtype=bool))
        assert list(reason) == ['', 'lock_wrong_side']


def test_the_graph_fields_and_the_parity_feature_carry_the_nitrogen(az):
    from energies.dof_features import atom_parity
    par = []
    for en in az:
        e, s = _n_element(en)
        g = condition_from_energy(en, identifier=en.smiles)
        assert int(g.ctree_stereo_kind[s]) == sl.TETRAHEDRAL
        assert int(g.ctree_stereo_sign[s]) == int(en.stereo.sign[e])
        assert (g.ctree_stereo_nbr[s].numpy() + s).tolist() == en.stereo.quad[e].tolist()
        p = atom_parity(en)
        assert p[s] == float(en.stereo.sign[e]) != 0.0
        par.append(p[s])
    assert par[0] == -par[1]
    off = _omitted(AZ_OPEN)
    assert atom_parity(off)[int(_slot(off)[3])] == 0.0


# ------------------------------------------------------------------ identity and verification

def test_the_two_configurations_are_two_conditions_each_verified():
    import build_conformer_conditions as bcc
    import build_conformer_set as bcs
    names, mirror_of, extra = bcs.stereo_plan('CC1CN1C', True)
    assert sorted(names) == sorted(['C[C@@H]1C[N@@]1C', 'C[C@@H]1C[N@]1C', 'C[C@H]1C[N@@]1C',
                                    'C[C@H]1C[N@]1C']) and not extra
    idents = {n: bcc.smiles_identity(n, True) for n in names}
    assert len(set(idents.values())) == 4
    # the mirror image inverts the nitrogen with the carbon: an enantiomer, not the invertomer
    assert mirror_of['C[C@@H]1C[N@]1C'] == 'C[C@H]1C[N@@]1C'
    assert mirror_of['C[C@@H]1C[N@@]1C'] == 'C[C@H]1C[N@]1C'
    # option off: the old two names, the nitrogen stripped
    assert bcs.stereo_plan('CC1CN1C')[0] == ['C[C@@H]1CN1C', 'C[C@H]1CN1C']
    assert bcc.smiles_identity(AZ_A) == bcc.smiles_identity(AZ_B)
    kw = {'level': 'full', 'force_field': 'mmff', 'stereo_coeff': 300.0,
          'lock_stereo_nitrogen': True}
    from rdkit import Chem
    with _default_dtype(torch.float64):
        for n in names:
            mb = bcc.build_member(n, n, kw, carrier=True)
            en = mb.energy
            assert en.stereo_isomer == en.stereo_input == n
            assert mb.thermal_check['status'] == 'passed'
            # the reference realises the tag, read independently of the table: the chiral
            # volume of the N's neighbours in bond order, on the member's own conformer
            (a,) = en.stereo_nitrogen_atoms
            atom = Chem.MolFromSmiles(n).GetAtomWithIdx(a)
            pos = _rd_positions(en)
            u, v, w = (pos[k] - pos[a] for k in sl.bond_order_neighbours(atom))
            assert np.sign(np.dot(u, np.cross(v, w))) == sl.nitrogen_tag_sign(atom)


def test_the_set_builders_walk_enumerates_the_invertomers_and_keeps_its_cap():
    """`build_conformer_set.build_molecule`, which the database builder also walks: the option
    reaches the enumeration through the energy kwargs; uncapped every invertomer is a
    condition, and at the default cap of two the molecule keeps a pick and its mirror image,
    which is its enantiomer and not its other invertomer."""
    import build_conformer_conditions as bcc
    import build_conformer_set as bcs
    from models.encoder_probe import mirror_smiles
    kw = {'level': 'full', 'force_field': 'mmff', 'stereo_coeff': 300.0}
    on = {**kw, 'lock_stereo_nitrogen': True}
    entry = bcs.Entry(key='CC1CN1C', index=0, smiles='CC1CN1C', side='train', h=0)
    four = sorted(['C[C@@H]1C[N@@]1C', 'C[C@@H]1C[N@]1C', 'C[C@H]1C[N@@]1C', 'C[C@H]1C[N@]1C'])
    with _default_dtype(torch.float64):
        built, rej = bcs.build_molecule(entry, on, bundle=None, cap=None, stereo_salt='')
        assert built.n_isomers == 4 and not rej
        assert sorted(m.identifier for m in built.members) == four
        assert all(len(m.energy.stereo_nitrogen_atoms) == 1 for m in built.members)
        capped, _ = bcs.build_molecule(entry, on, bundle=None, cap=2, stereo_salt='')
        a, b = [m.identifier for m in capped.members]
        assert capped.mirror_of[a] == b
        assert bcc.smiles_identity(mirror_smiles(a), True) == bcc.smiles_identity(b, True)
        # the option off, and omitted: the two names with the nitrogen stripped
        for off in (kw, {**kw, 'lock_stereo_nitrogen': False}):
            old, _ = bcs.build_molecule(entry, off, bundle=None, cap=None, stereo_salt='')
            assert sorted(m.identifier for m in old.members) == ['C[C@@H]1CN1C', 'C[C@H]1CN1C']
            assert all(m.energy.stereo_nitrogen_atoms == () for m in old.members)


@pytest.mark.parametrize('smi', [AZ_OPEN, MIX_OPEN, 'C1C[C@@H]2CN2C1'])
def test_an_untagged_stereogenic_nitrogen_is_unspecified_like_a_carbon(smi):
    import build_conformer_conditions as bcc
    with pytest.raises(ChartRefused) as exc:
        _member(smi)
    assert exc.value.code == 'stereo_unspecified' and 'N' not in exc.value.code
    assert len(sl.consistent_isomers(smi, lock_nitrogen=True)) == 2
    assert len(sl.consistent_isomers(smi)) == 1
    assert len(bcc.unspecified_stereo(smi, True)) == 2 and bcc.unspecified_stereo(smi) == []
    kw = {'level': 'full', 'force_field': 'mmff', 'stereo_coeff': 300.0,
          'lock_stereo_nitrogen': True}
    with pytest.raises(bcc.MemberRefused) as exc:
        bcc.build_member(smi, smi, kw, carrier=True)
    assert exc.value.code == 'stereo_unspecified'
    # the carbon case it is treated as
    with pytest.raises(ChartRefused) as exc:
        _member('CC(O)CC=O')
    assert exc.value.code == 'stereo_unspecified'


def test_a_nitrogen_only_one_side_of_which_exists_builds_on_that_side_only():
    import build_conformer_conditions as bcc
    import build_conformer_set as bcs
    kw = {'level': 'full', 'force_field': 'mmff', 'stereo_coeff': 300.0,
          'lock_stereo_nitrogen': True}
    names = bcs.stereo_plan('C1CC2CN2C1', True)[0]
    assert len(names) == 4 and FUSED in names and FUSED_BAD in names
    built, codes = [], []
    with _default_dtype(torch.float64):
        for n in names:
            try:
                built.append(bcc.build_member(n, n, kw, carrier=True).identifier)
            except bcc.MemberRefused as exc:
                codes.append(exc.code)
    assert len(built) == 2 and FUSED in built and codes == ['embed_failed'] * 2
    en = _member(CAGE)
    assert len(en.stereo_nitrogen_atoms) == 1 and en.stereo_isomer == en.stereo_input
    # a bridgehead N no enumeration distinguishes stays untagged, unlocked and one condition
    assert bcs.stereo_plan(SYM_CAGE, True)[0] == [SYM_CAGE]
    assert _member(SYM_CAGE).stereo_nitrogen_atoms == ()


def test_a_reference_on_the_other_side_is_refused_and_a_flat_one_is_not_locked_onto_noise():
    """The verification, on stored references: the member's own reference with the nitrogen
    pushed through the plane of its neighbours (the other invertomer's geometry: refused as
    another isomer) and pushed nearly into it (refused as in-band, not built with the
    nitrogen free)."""
    en = _member(AZ_A)
    (a,) = en.stereo_nitrogen_atoms
    nb = sl.bond_order_neighbours(en.mol.GetAtomWithIdx(a))
    pos = _rd_positions(en)
    same = _member(AZ_A, reference_positions=pos)
    assert np.array_equal(same.stereo.sign, en.stereo.sign)
    with pytest.raises(ChartRefused) as exc:
        _member(AZ_A, reference_positions=_through_plane(pos, a, nb, -1.0))
    assert exc.value.code == 'stereo_verify_failed' and 'OTHER configuration' in str(exc.value)
    for scale in (0.3, 0.05, -0.05, -0.3):
        with pytest.raises(ChartRefused) as exc:
            _member(AZ_A, reference_positions=_through_plane(pos, a, nb, scale))
        assert exc.value.code == 'stereo_lock_in_band', scale
        assert 'nearly planar' in str(exc.value)
    # with the option off the flat reference of the untagged SMILES builds: the N is free
    off = _omitted(AZ_OPEN)
    (n_off,) = [i for i in range(off.mol.GetNumAtoms())
                if off.mol.GetAtomWithIdx(i).GetAtomicNum() == 7]
    flat = _through_plane(_rd_positions(off), n_off,
                          sl.bond_order_neighbours(off.mol.GetAtomWithIdx(n_off)), 0.05)
    assert _omitted(AZ_OPEN, reference_positions=flat).stereo.n == off.stereo.n


# ------------------------------------------------------------------ the prior

def test_the_prior_holds_a_locked_nitrogen_and_still_flips_an_amine(prior):
    on = _member(MIX_A)
    off = _omitted(MIX_OPEN)
    s_on, s_off = _slot(on), _slot(off)
    ring_n, amine_n = int(s_on[on.stereo_nitrogen_atoms[0]]), int(s_on[1])
    tab = {c.slot: c for c in centre_table(on)}
    assert tab[ring_n].lock == LOCKED and tab[ring_n].n_bonded == 3
    assert [c.slot for c in invertible_centres(on)] == [amine_n]
    # with the option off the same two nitrogens are both free and both flipped
    assert sorted(c.slot for c in invertible_centres(off)) == sorted(
        int(s_off[i]) for i in (1, 6))
    n = 2000
    x, stats = on.sample_prior_states(prior, n, np.random.default_rng(7), report=False)
    assert stats['invertible_centres'] == [tab[amine_n].name]
    assert 0.45 < stats['reflected_frac'][0] < 0.55
    pos = on.build_positions(x).reshape(n, -1, 3).detach()
    assert bool(reference_side(pos, tab[ring_n]).all())
    amine_side = reference_side(pos, tab[amine_n])
    assert 0.4 < float(amine_side.mean()) < 0.6
    # and the lock never fires on the nitrogen's element over the prior's draws
    e, _ = on.stereo.element_of(ring_n), None
    _, _, sign, lo = on.stereo.tensors('cpu', torch.float64)
    sv = sign[e] * on.stereo.values(pos.double())[:, e]
    assert float(sv.min()) > float(lo[e])
    # option off, untagged: the ring nitrogen is drawn on both sides
    xo, so = off.sample_prior_states(prior, n, np.random.default_rng(7), report=False)
    po = off.build_positions(xo).reshape(n, -1, 3).detach()
    ring_off = {c.slot: c for c in centre_table(off)}[int(s_off[6])]
    assert 0.4 < float(reference_side(po, ring_off).mean()) < 0.6


def test_a_locked_nitrogens_sibling_group_is_not_converted():
    """`sibling_offset_box_deg` with the option on: the nitrogen is named by the lock but its
    group keeps periodic followers (a ring atom, and a three-neighbour centre)."""
    # N-(2-hydroxyethyl)-2-methylaziridine: the tree enters the nitrogen from a ring carbon
    # and places two children on it, so its sibling group has two rows
    en = _member('C[C@H]1C[N@]1CCO', sibling_offset_box_deg=60.0)
    s = int(_slot(en)[en.stereo_nitrogen_atoms[0]])
    at_n = [g for g in en.sibling_offset_census() if g['centre'] == s]
    assert [len(g['rows']) for g in at_n] == [2]
    assert at_n[0]['reason'] == 'ring_atom'
    ti = np.asarray(en.spec.torsion_index)
    for rows in en.sibling_offset_groups:
        assert int(ti[rows[0], 2]) != s
    assert en.sibling_offset_groups, 'the molecule has eligible carbon groups'
    centre_table(en)                               # the free-centre guard does not fire
    # the three-neighbour rule on its own, with the ring rule taken away: every stereo
    # nitrogen RDKit reports is a ring atom, so only this reaches it
    en._ring_atoms = lambda: set()
    try:
        (g,) = [g for g in en.sibling_offset_census() if g['centre'] == s]
        assert g['reason'] == 'centre_not_locked'
    finally:
        del en._ring_atoms


# ------------------------------------------------------------------ the set

def test_the_one_pass_set_energy_is_each_members_own():
    import build_conformer_database as db
    smis = [AZ_A, AZ_B, MIX_A, FUSED, 'CCN(C)C', 'CCCO']
    st = _Set(smis, [6, 6, 5, 4, 3, 3], torch.float64, stereo_coeff=300.0,
              lock_stereo_nitrogen=True)
    multi = st.multi
    assert multi.lock_stereo_nitrogen is True
    # two rows of AZ_A put on the OTHER invertomer's reference, so the nitrogen's term is live
    a, b = multi._members[AZ_A], multi._members[AZ_B]
    x_inv = db.states_of_rows(a, b.ref_pos.detach().double().numpy()[None])
    rows = np.flatnonzero(st.assign == 0)[:2]
    X = st.X.clone()
    for r in rows:
        X[r] = st.lay.to_carrier(AZ_A, x_inv)[0] if st.lay is not None else x_inv[0]
    got = multi.energy(X, st.batch, st.logT)
    want = st.oracle(X)
    assert torch.allclose(got, want, rtol=1e-12, atol=1e-9), float((got - want).abs().max())
    assert torch.equal(multi._energy_per_member(X, st.batch, st.logT), want)
    # the nitrogen's term is live in that comparison: on those rows the member's own lock is
    # the big term, and the one-pass potential carries it (the same rows at the reference
    # differ from them by it and the force field's change, which the member gives too)
    own = float(a.stereo.lock_energy(a.build_positions(x_inv).reshape(1, -1, 3),
                                     a.stereo_coeff))
    assert own > 100.0
    zero = torch.zeros(len(X), dtype=torch.float64)
    X0 = X.clone()
    X0[rows] = 0.0
    e_inv = multi.energy(X, st.batch, zero, return_exp=True)[1].conformer_energy.flatten()
    e_ref = multi.energy(X0, st.batch, zero, return_exp=True)[1].conformer_energy.flatten()
    one = torch.tensor(1.0, dtype=torch.float64)
    u = a.potential_energy(torch.cat([x_inv, torch.zeros_like(x_inv)]), one)
    for r in rows:
        assert float(e_inv[r] - e_ref[r]) == pytest.approx(float(u[0] - u[1]), abs=1e-8)
    assert float(u[0] - u[1]) > 100.0
    # a member built under the other value is refused by the library
    multi._members['CCCO'].lock_stereo_nitrogen = False
    try:
        with pytest.raises(ValueError, match='lock_stereo_nitrogen differs'):
            multi._build_library()
    finally:
        multi._members['CCCO'].lock_stereo_nitrogen = True


def test_a_set_built_without_the_option_fails_loudly_under_it():
    """An old set names 1,2-dimethylaziridine by its carbon alone. A run with the option on
    refuses that member at construction; one that names an invertomer runs."""
    from energies.multi_conformer import MultiConformerTorsions
    kw = dict(LOCK, lock_stereo_nitrogen=True)
    with _default_dtype(torch.float64):
        with pytest.raises(ChartRefused) as exc:
            MultiConformerTorsions(['CCCO', AZ_OPEN], identifiers=['CCCO', AZ_OPEN], **kw)
        assert exc.value.code == 'stereo_unspecified'
        MultiConformerTorsions(['CCCO', AZ_A], identifiers=['CCCO', AZ_A], **kw)
        # and a set built WITH it, in a run without: the tagged member is refused
        with pytest.raises(ChartRefused) as exc:
            MultiConformerTorsions(['CCCO', AZ_A], identifiers=['CCCO', AZ_A], **LOCK)
        assert exc.value.code == 'stereo_unsupported'


# ------------------------------------------------------------------ stamps and the database

def test_an_absent_key_is_false_in_every_stamp():
    import build_conformer_database as db
    import build_conformer_references as bcr
    old = {'level': 'full', 'force_field': 'mmff', 'stereo_coeff': 300.0}
    assert bcr.defining_energy(bcr.member_kwargs(old)) == \
        bcr.defining_energy(bcr.member_kwargs({**old, 'lock_stereo_nitrogen': False}))
    assert bcr.defining_energy(bcr.member_kwargs(old))['lock_stereo_nitrogen'] is False
    # NOT chart-only: it is in no list that says a store does not depend on it ...
    assert 'lock_stereo_nitrogen' not in db.CHART_ONLY_KWARGS
    assert 'lock_stereo_nitrogen' not in bcr.NON_DEFINING_KWARGS
    # ... a database is read BY IDENTIFIER across the two values, and says so
    assert db.IDENTITY_KWARGS == ('lock_stereo_nitrogen',)
    info = {'path': 'db', 'energy_kwargs': old}
    on = {**old, 'lock_stereo_nitrogen': True}
    db.refuse_other_member_kwargs(info, on, 'test')
    assert db.other_identity_kwargs(info, on) == {
        'lock_stereo_nitrogen': {'database': False, 'consumer': True}}
    assert db.other_identity_kwargs(info, old) == {}
    with pytest.raises(SystemExit, match='delta_r_max'):
        db.refuse_other_member_kwargs(info, {**on, 'delta_r_max': 0.2}, 'test')


def test_a_reference_table_stamped_before_the_argument_loads_off_and_is_refused_on(tmp_path):
    import build_conformer_references as bcr
    kw = {'level': 'full', 'force_field': 'mmff', 'stereo_coeff': 300.0}
    cond = tmp_path / 'conditions.pt'
    cond.write_bytes(b'conditions')
    stamp_energy = bcr.defining_energy(kw)
    del stamp_energy['lock_stereo_nitrogen']             # as a table written before it existed
    table = tmp_path / 'refs.pt'
    torch.save({'stamp': {'format': bcr.FORMAT, 'energy': stamp_energy,
                          'conditions': {'path': str(cond), 'sha256': bcr.file_sha256(cond),
                                         'identifiers': []}},
                'entries': {}, 'failures': {}}, table)
    bcr.load_references(table, conditions_path=cond, energy_kwargs=kw)
    bcr.load_references(table, conditions_path=cond,
                        energy_kwargs={**kw, 'lock_stereo_nitrogen': False})
    with pytest.raises(bcr.StaleReferencesError, match='lock_stereo_nitrogen'):
        bcr.load_references(table, conditions_path=cond,
                            energy_kwargs={**kw, 'lock_stereo_nitrogen': True})


def test_a_database_built_without_the_option_under_it():
    """A real shard, built without the option. Under the option: a record whose SMILES has no
    stereogenic stereo nitrogen matches exactly as before; a record of a molecule that has one
    is never asked for (its identifier is refused, and the names the walk asks for are not in
    the database)."""
    import build_conformer_database as db
    import build_conformer_set as bcs
    from energies.conformer_torsions import rdkit_order_reference
    if not SHARD.exists():
        pytest.skip(f'conformer database shard not found at {SHARD}')
    blob = torch.load(SHARD, weights_only=False, map_location='cpu')
    ekw = dict(blob['header']['config']['energy_kwargs'])
    assert 'lock_stereo_nitrogen' not in ekw and float(ekw['stereo_coeff']) > 0
    recs = [(r['key'], c) for r in blob['keys'].values() for c in r['conditions']
            if c['refusal'] is None and len(c['basins']['energy']) > 1]
    idents = {c['identifier'] for _, c in recs}

    def open_nitrogen(smi):
        return len(sl.consistent_isomers(smi, lock_nitrogen=True)) > 1

    plain = next(c for _, c in recs if not open_nitrogen(c['identifier']))
    key, hit = next((k, c) for k, c in recs if open_nitrogen(c['identifier']))
    with _default_dtype(torch.float64):
        # (1) unaffected record: the same rows, bit for bit, under either value
        out = {}
        for on in (False, True):
            ident = plain['identifier']
            pos, _ = rdkit_order_reference(ident, plain['ref_pos'], level=ekw['level'],
                                           perm=plain['perm'], z=plain['z'])
            en = ConformerTorsions(smiles=ident, reference_positions=pos,
                                   lock_stereo_nitrogen=on, **ekw)
            rec = {k: (plain.get(k) if k in db.OPTIONAL_FIELDS else plain[k])
                   for k in db.READ_FIELDS}
            x, stored, info = db.match_rows(en, ident, rec, 8, 1e-6)
            assert info['code'] is None, info
            out[on] = (x, stored)
        assert torch.equal(out[False][0], out[True][0])
        assert np.array_equal(out[False][1], out[True][1])
        # (2) affected record: its identifier does not build under the option ...
        ident = hit['identifier']
        pos, _ = rdkit_order_reference(ident, hit['ref_pos'], level=ekw['level'],
                                       perm=hit['perm'], z=hit['z'])
        ConformerTorsions(smiles=ident, reference_positions=pos, **ekw)
        with pytest.raises(ChartRefused) as exc:
            ConformerTorsions(smiles=ident, reference_positions=pos, lock_stereo_nitrogen=True,
                              **ekw)
        assert exc.value.code == 'stereo_unspecified'
    # ... and the walk under the option asks for other names, none of them in the database
    names = bcs.stereo_plan(key, True)[0]
    assert names and not (set(names) & idents)
    assert set(bcs.stereo_plan(key)[0]) & idents
    assert db.match_rows(None, names[0], None, 8, 1e-6)[2]['code'] == 'db_absent'
