"""Key and filter contract of the SMILES training pool (training_pool_census.judge, training_pool_build.expand).

Owner decisions of 2026-11-03: mirror images share a key; diastereomers, E/Z isomers and tautomers do not; the
standard-InChIKey skeleton (ik14) still joins tautomers for the Day leak check; formal charges only as local
dipoles, separated zwitterions only from the CSD; isotope labels merge into the parent; unassigned stereo is
enumerated.
"""
import pytest

from data_processing import training_pool_build as build
from data_processing import training_pool_census as census

pytestmark = pytest.mark.fast   # RDKit only, no data drive


@pytest.fixture(autouse=True)
def _no_separated(monkeypatch):
    monkeypatch.setattr(census, 'ALLOW_SEPARATED', False)


def key(s):
    r, why = census.judge(s)
    assert r is not None, (s, why)
    return r['key'], r['ik14']


def test_mirror_images_share_a_key():
    assert key('C[C@H](N)C(=O)O')[0] == key('C[C@@H](N)C(=O)O')[0]


def test_diastereomers_and_ez_isomers_are_separate():
    assert key('C[C@H](O)[C@H](C)O')[0] != key('C[C@H](O)[C@@H](C)O')[0]
    assert key('C/C=C/C')[0] != key('C/C=C\\C')[0]


def test_tautomers_separate_but_share_the_leak_skeleton():
    (k1, s1), (k2, s2) = key('Oc1ccccn1'), key('O=c1cccc[nH]1')
    assert k1 != k2 and s1 == s2


def test_charges_only_as_local_dipoles_except_csd(monkeypatch):
    assert census.judge('C[N+](=O)[O-]')[0] is not None
    assert census.judge('[NH3+]CC(=O)[O-]')[1] == 'separated_zwitterion'
    monkeypatch.setattr(census, 'ALLOW_SEPARATED', True)
    assert census.judge('[NH3+]CC(=O)[O-]')[0] is not None


@pytest.mark.parametrize('s, why', [('CCCCO', 'rotatable_gt1'), ('CCl', 'elements'),
                                    ('c1ccc2ccccc2c1-c1ccc2ccccc2c1', 'mw'), ('C[NH3+]', 'charge_radical')])
def test_filters_reject(s, why):
    assert census.judge(s) == (None, why)


def test_expand_strips_isotopes_into_the_parent():
    rows, cnt = build.expand(('[2H]c1ccccc1', 'pubchem'))
    assert [r['key'] for r in rows] == [key('c1ccccc1')[0]] and cnt['isotope_stripped'] == 1


def test_expand_enumerates_unassigned_stereo_and_merges_mirrors():
    assert len(build.expand(('CC=CC', 'pubchem'))[0]) == 2
    assert len({r['key'] for r in build.expand(('CC(O)C(C)O', 'pubchem'))[0]}) == 2   # meso + racemic pair
    assert len({r['key'] for r in build.expand(('CC(N)C(=O)O', 'csd'))[0]}) == 1
