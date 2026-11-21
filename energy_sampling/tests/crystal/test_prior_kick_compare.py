"""The kick-scale rebuild's two tools (2026-11-20): build_noised_prior --heldout_units and prior_kick_compare.py.

What each test would catch:
  * a rebuild that holds out other descent groups than the file its units came from (units -> heldout_from_units);
  * a compare that calls two files with the same anchors different, or misses an anchor moved between splits;
  * a child excess read against another row than its parent, or a load-time count that is not select_rows'.

    python -m pytest -q tests/crystal/test_prior_kick_compare.py
"""
from types import SimpleNamespace

import pytest
import torch

import noised_prior as NP
from data_processing import prior_kick_compare as PKC
from data_processing.build_noised_prior import heldout_from_units

pytestmark = pytest.mark.fast


class Rows(SimpleNamespace):
    def keys(self):
        return list(vars(self))


def _file(split_of_anchor, child_e, seed=0):
    """Two molecules x three anchors (descent groups 0, 0, 1), each anchor with two children of energy anchor +
    child_e (run units). split_of_anchor: [6] 0/1."""
    g = torch.Generator().manual_seed(seed)
    n_a = 6
    ids = ['m0'] * 3 + ['m1'] * 3
    e_a = torch.tensor([0.0, 1.0, 2.0, 0.5, 1.5, 2.5], dtype=torch.float64)
    parent = torch.cat([torch.arange(n_a), torch.arange(n_a).repeat_interleave(2)])
    kind = torch.cat([torch.zeros(n_a), torch.ones(2 * n_a)]).long()
    split = torch.as_tensor(split_of_anchor).long()[parent]
    mol = torch.cat([e_a, e_a.repeat_interleave(2) + child_e])
    n = len(parent)
    return Rows(identifier=[ids[int(p)] for p in parent], sg_ind=torch.full((n,), 2), z_prime=torch.ones(n, dtype=torch.long),
                basin_group=torch.tensor([0, 0, 1, 0, 0, 1])[parent], basin_id=torch.tensor([0, 1, 2, 0, 1, 2])[parent],
                noised_kind=kind, noised_split=split, noised_parent=parent,
                noised_radius=torch.where(kind == 1, torch.rand(n, generator=g), torch.zeros(n)),
                mol_energy=mol, density_energy=torch.zeros(n), pressure_energy=torch.zeros(n),
                packing_coeff=torch.full((n,), 0.7), cell_lengths=torch.rand(n, 3, generator=g),
                cell_angles=torch.rand(n, 3, generator=g), aunit_centroid=torch.rand(n, 3, generator=g),
                aunit_orientation=torch.rand(n, 3, generator=g), num_graphs=n)


def test_units_written_from_a_file_hold_out_the_same_anchors_in_a_rebuild(tmp_path, monkeypatch):
    src = _file([0, 0, 1, 0, 1, 1], child_e=1.0)
    monkeypatch.setattr(PKC, '_load', lambda p: (src, {}))
    out = tmp_path / 'units.pt'
    PKC.units('src.pt', str(out))
    assert sorted(torch.load(out, weights_only=False)['units']) == [('m0', 2, 1, 1), ('m1', 2, 1, 0), ('m1', 2, 1, 1)]
    anchors = Rows(**{k: (v[:6] if torch.is_tensor(v) else v[:6]) for k, v in vars(src).items() if k != 'num_graphs'})
    held = heldout_from_units(anchors, str(out))
    # a descent group goes whole: m1's group 0 holds anchors 3 and 4, so anchor 3 (held in training in src) goes too
    assert held.tolist() == [False, False, True, True, True, True]


def test_compare_passes_on_the_same_anchors_and_reads_children_against_their_parents(tmp_path, monkeypatch, capsys):
    a = _file([0, 0, 1, 0, 0, 1], child_e=2.5, seed=1)    # excess 1 kT at T 2.5
    b = _file([0, 0, 1, 0, 0, 1], child_e=30.0, seed=2)   # excess 12 kT: outside a 10 kT window
    monkeypatch.setattr(PKC, '_load', lambda p: ({'a': a, 'b': b}[p], {'temperature': 2.5}))
    assert PKC.compare('a', 'b', [10.0], 10.0)
    text = capsys.readouterr().out
    assert 'anchor sets IDENTICAL' in text
    ex_a, _ = PKC.child_excess(a, PKC.seed_energy(a, 10.0), 2.5)
    ex_b, _ = PKC.child_excess(b, PKC.seed_energy(b, 10.0), 2.5)
    torch.testing.assert_close(ex_a['train'], torch.full((8,), 1.0, dtype=torch.float64))
    torch.testing.assert_close(ex_b['heldout'], torch.full((4,), 12.0, dtype=torch.float64))
    # select_rows' floor is each condition's lowest training row: a's children all sit inside 10 kT of it, b's none
    ca, cb = PKC.window_counts(a, PKC.seed_energy(a, 10.0), 2.5, 10.0), PKC.window_counts(b, PKC.seed_energy(b, 10.0),
                                                                                          2.5, 10.0)
    assert (ca['train_children'], ca['children_train_split'], ca['buffer_anchors']) == (8, 8, 4)
    assert (cb['train_children'], cb['buffer']) == (0, 4)


def test_compare_fails_when_an_anchor_changes_split(monkeypatch, capsys):
    a = _file([0, 0, 1, 0, 0, 1], child_e=1.0)
    b = _file([0, 0, 0, 0, 0, 1], child_e=1.0)
    monkeypatch.setattr(PKC, '_load', lambda p: ({'a': a, 'b': b}[p], {'temperature': 2.5}))
    assert not PKC.compare('a', 'b', [10.0], 10.0)
    assert 'anchor sets DIFFER' in capsys.readouterr().out
