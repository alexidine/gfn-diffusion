"""train.py's Modeller.check_held_out_molecules, run from Modeller.init_identifiers: a held-out molecule
(test_molecules_path) that is also in molecules_path, the prior, or the pre-noised prior's held-out anchor groups or
clean anchors raises, naming both files and the count; one molecule under two identifiers is caught by canonical
SMILES when both batches carry .smiles; a prior whose molecule set differs from the condition set is printed, not
refused.

Nothing is built: the methods run on a namespace holding stand-in batches.

    python -m pytest -q tests/crystal/test_held_out_leak.py
"""
from types import SimpleNamespace

import pytest

from train import Modeller


def _batch(ids, smiles=None):
    return SimpleNamespace(identifier=list(ids), smiles=smiles, device='cpu', add_graph_attr=lambda v, name: None)


def _modeller(mol, prior, test=None, heldout=None, anchors=None):
    m = SimpleNamespace(args=SimpleNamespace(molecules_path='train_conds.pt', prior_path='train_prior.pt',
                                             test_molecules_path='test_conds.pt'),
                        mol_dataset=SimpleNamespace(batch=mol), prior_dataset=SimpleNamespace(batch=prior),
                        test_mol_dataset=None if test is None else SimpleNamespace(batch=test),
                        prior_heldout_batch=heldout, prior_anchor_batch=anchors,
                        energy_function=SimpleNamespace(set_n_molecules=lambda n: None))
    m._molecule_keys = Modeller._molecule_keys
    m.check_held_out_molecules = lambda: Modeller.check_held_out_molecules(m)
    return m


def test_disjoint_test_set_passes_and_builds_the_registry():
    m = _modeller(_batch('ab'), _batch('ab'), test=_batch('cd'))
    Modeller.init_identifiers(m)
    assert m.identifier_registry == {'a': 0, 'b': 1, 'c': 2, 'd': 3}


@pytest.mark.parametrize('where', ['mol', 'prior', 'heldout', 'anchors'])
def test_a_shared_identifier_raises_naming_both_files_and_the_count(where):
    batches = dict(mol=_batch('ab'), prior=_batch('ab'), heldout=None, anchors=None)
    batches[where] = _batch('abc')
    m = _modeller(test=_batch('cd'), **batches)
    with pytest.raises(ValueError) as e:
        Modeller.init_identifiers(m)
    msg = str(e.value)
    assert 'test_conds.pt' in msg
    assert ('train_conds.pt' if where == 'mol' else 'train_prior.pt') in msg
    assert "1 by identifier (e.g. ['c'])" in msg


def test_one_molecule_under_two_identifiers_is_caught_by_canonical_smiles():
    # benzene written aromatic and Kekule under different identifiers
    m = _modeller(_batch(['mol_a', 'mol_b'], smiles=['c1ccccc1', 'CCO']), _batch(['mol_a', 'mol_b']),
                  test=_batch(['mol_z'], smiles=['C1=CC=CC=C1']))
    with pytest.raises(ValueError, match=r"molecules_path \(train_conds.pt\): 0 by identifier .*, 1 by canonical "
                                         r"SMILES \(e.g. \['c1ccccc1'\]\)"):
        Modeller.check_held_out_molecules(m)


def test_smiles_compared_only_when_both_sides_carry_them():
    m = _modeller(_batch(['mol_a'], smiles=['c1ccccc1']), _batch(['mol_a']), test=_batch(['mol_z']))
    Modeller.check_held_out_molecules(m)


def test_no_test_file_is_not_checked():
    Modeller.check_held_out_molecules(_modeller(_batch('ab'), _batch('ab')))


def test_prior_molecule_set_differing_from_the_conditions_prints(capsys):
    Modeller.check_held_out_molecules(_modeller(_batch('ab'), _batch('bc')))
    out = capsys.readouterr().out
    assert 'prior and condition molecule sets differ' in out
    assert "1 of 2 prior_path (train_prior.pt) molecules are not in molecules_path (train_conds.pt) (e.g. ['c'])" in out
    Modeller.check_held_out_molecules(_modeller(_batch('ab'), _batch('ba')))
    assert capsys.readouterr().out == ''
