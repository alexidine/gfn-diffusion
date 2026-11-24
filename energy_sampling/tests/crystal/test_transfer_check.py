"""The pre-set pass rule and structure pick of eval.exp6_recall.transfer_check."""
import numpy as np
import pytest

pytest.importorskip('rdkit')
from rdkit import Chem  # noqa: E402

from eval.exp6_recall.transfer_check import pick_structures, selection_class, verdict  # noqa: E402


def test_a_constant_offset_passes():
    assert verdict([1.0, 2.0, 3.0, 4.0], [101.0, 102.0, 103.0, 104.0])['passed'] is True


def test_rms_above_the_limit_fails():
    assert verdict([0.0, 5.0, -5.0, 5.0, -5.0], np.zeros(5))['passed'] is False


def test_one_structure_far_too_low_fails_though_the_rms_passes():
    e = np.zeros(10)
    e[0] = -11.0  # RMS 3.5 kJ/mol
    v = verdict(e, np.zeros(10))
    assert v['rms_kJ'] <= 4.0 and v['passed'] is False


def test_fewer_than_three_structures_is_not_evaluable():
    assert verdict([1.0, 2.0], [1.0, 2.0])['evaluable'] is False


def test_pick_keeps_the_lowest_and_distinct_rows():
    e = np.arange(0, 20, 0.5)
    k = pick_structures(e)
    assert k[0] == 0 and len(set(k)) == 5 and e[k[-1]] == pytest.approx(15.0)


@pytest.mark.parametrize('smiles, inside, ch3', [
    ('c1ccccc1', True, False), ('Oc1ccccc1', True, False), ('OC(=O)c1ccccc1', True, False),
    ('Cc1ccccc1', False, True), ('CCCC', False, True), ('c1ccc(cc1)-c1ccccc1', False, False)])
def test_selection_rules(smiles, inside, ch3):
    got = selection_class(Chem.MolFromSmiles(smiles))
    assert got[0] is inside and got[1] is ch3
