"""train.py's Modeller.publish_data_set_sizes: the data/ keys that train.py and inn_modeller.py write to the wandb run
config once the data sets are loaded. The molecule counts are DISTINCT identifiers per file, not rows; a missing test
file counts 0; a trainer's own keys ride along; the dict reaches wandb.config only when a run is open.

Nothing is built: the method is called on a namespace holding stand-in data sets.

    python -m pytest -q tests/crystal/test_data_set_sizes.py
"""
from types import SimpleNamespace

import wandb

from train import Modeller


def _ds(ids):
    return SimpleNamespace(batch=SimpleNamespace(identifier=list(ids), num_graphs=len(ids)))


def _modeller(test=True):
    return SimpleNamespace(mol_dataset=_ds('abc'), prior_dataset=_ds('aabbcd'),
                           test_mol_dataset=_ds('ef') if test else None,
                           identifier_registry={k: i for i, k in enumerate('abcdef')},
                           energy_function=SimpleNamespace(condition_library_size=12))


def test_counts_are_distinct_molecules_per_file():
    sizes = Modeller.publish_data_set_sizes(_modeller())
    assert sizes == {'data/train_molecules': 3, 'data/test_molecules': 2, 'data/prior_rows': 6,
                     'data/prior_molecules': 4, 'data/molecules': 6, 'data/condition_library_size': 12}


def test_no_test_file_counts_zero_and_extra_keys_ride_along():
    sizes = Modeller.publish_data_set_sizes(_modeller(test=False), **{'data/train_rows': 5, 'data/val_rows': 1})
    assert sizes['data/test_molecules'] == 0
    assert sizes['data/train_rows'] == 5 and sizes['data/val_rows'] == 1


def test_written_to_the_run_config_only_when_a_run_is_open(monkeypatch):
    got = []
    monkeypatch.setattr(wandb, 'config', SimpleNamespace(update=lambda d, allow_val_change=False: got.append(d)))
    monkeypatch.setattr(wandb, 'run', None)
    Modeller.publish_data_set_sizes(_modeller())
    assert got == []
    monkeypatch.setattr(wandb, 'run', object())
    sizes = Modeller.publish_data_set_sizes(_modeller())
    assert got == [sizes]
