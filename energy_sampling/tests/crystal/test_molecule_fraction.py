"""inn_modeller.py --molecule_fraction: nested molecule subsets of the conditions file and the prior, by the fixed draw
data_processing/molecule_subset.py::molecule_u, applied through Modeller.subset_training_molecules at load."""
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_processing.molecule_subset import keep_molecules, molecule_u  # noqa: E402

pytestmark = pytest.mark.fast


class _Rows:
    """The two things keep_molecules reads off a crystal batch: identifier and subsample_new_batch."""

    def __init__(self, identifier):
        self.identifier = list(identifier)
        self.num_graphs = len(self.identifier)

    def subsample_new_batch(self, idx):
        return _Rows([self.identifier[i] for i in idx])


def test_molecule_u_is_a_fixed_draw_in_the_unit_interval():
    # pinned: the subsets of every build and run depend on these exact values (build_rc_prior.py uses the same draw)
    assert molecule_u('mol_a') == pytest.approx(0.43388624636742484, abs=0)
    assert molecule_u('QM9_000123') == pytest.approx(0.44316331117843827, abs=0)
    us = [molecule_u(f'm{i}') for i in range(2000)]
    assert all(0.0 <= u < 1.0 for u in us)
    assert 0.08 < sum(u < 0.1 for u in us) / len(us) < 0.12


def test_whole_molecules_are_kept_and_fractions_nest():
    ids = [f'm{i // 3}' for i in range(3 * 400)]                    # 400 molecules, 3 rows each
    rows = _Rows(ids)
    small, n_small = keep_molecules(rows, 0.1)
    big, n_big = keep_molecules(rows, 0.5)
    for sub, f in ((small, 0.1), (big, 0.5)):
        kept = set(sub.identifier)
        assert all(molecule_u(m) < f for m in kept)
        assert all(sub.identifier.count(m) == 3 for m in kept)      # a molecule comes with all its rows
        assert kept == {m for m in set(ids) if molecule_u(m) < f}
    assert set(small.identifier) <= set(big.identifier)
    assert (n_small, n_big) == (len(set(small.identifier)), len(set(big.identifier)))


def test_two_files_keep_the_same_molecules():
    conditions = _Rows([f'm{i}' for i in range(300)])
    prior = _Rows([f'm{i % 300}' for i in range(2100)])
    c, _ = keep_molecules(conditions, 0.2)
    p, _ = keep_molecules(prior, 0.2)
    assert set(c.identifier) == set(p.identifier)


def test_fraction_one_is_the_batch_itself_and_bad_fractions_raise():
    rows = _Rows(['a', 'b'])
    assert keep_molecules(rows, 1.0)[0] is rows
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match='must lie in'):
            keep_molecules(rows, bad)
    with pytest.raises(ValueError, match='keeps no row'):
        keep_molecules(_Rows(['CC(=O)O']), 0.5)                     # molecule_u 0.8375


def test_the_trainers_route_both_files_through_the_hook():
    from inn_modeller import InnModeller
    from train import Modeller
    assert Modeller.subset_training_molecules(None, 'b', 'prior_path') == 'b'
    for fn in (Modeller.init_prior_dataset, Modeller.init_mol_dataset):
        assert 'self.subset_training_molecules(' in inspect.getsource(fn)
    m = object.__new__(InnModeller)
    m.inn = SimpleNamespace(molecule_fraction=0.5)
    rows = _Rows([f'm{i}' for i in range(200)])
    sub = m.subset_training_molecules(rows, 'prior_path')
    assert set(sub.identifier) == {f'm{i}' for i in range(200) if molecule_u(f'm{i}') < 0.5}
    m.inn.molecule_fraction = 1.0
    assert m.subset_training_molecules(rows, 'prior_path') is rows
