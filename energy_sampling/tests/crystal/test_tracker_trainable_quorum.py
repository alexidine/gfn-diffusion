"""ConditionLogZTracker.delta_stats' quorum against the TRAINABLE conditions.

delta_stats returns +inf until min_trusted_frac of the conditions are characterised
on both level streams. The denominator was the whole library, held-out test
conditions included, which training never visits -- on a library mostly held out
the gate could never open. set_trainable (applied by Modeller._set_tracker_trainable
from the molecules both mol_dataset and prior_dataset carry) makes the quorum characterised
trainable conditions over the trainable count; unset keeps the whole library.

Nothing is built: the trainer method runs on a namespace with stand-in data sets.

    python -m pytest -q tests/crystal/test_tracker_trainable_quorum.py
"""
import math
from types import SimpleNamespace

import pytest
import torch

from buffer import ConditionLogZTracker
from energies.molecular_crystal import MolecularCrystal
from train import Modeller

MIN_VISITS = 3


def _characterise(tracker, cids, gap=0.5):
    """Feed both level streams of each condition past min_visits, bwd - fwd = gap."""
    cid = torch.as_tensor(cids, dtype=torch.long)
    for step in range(MIN_VISITS):
        tracker.update_mode_level('fwd', cid, torch.zeros(len(cid)), step)
        tracker.update_mode_level('bwd', cid, torch.full((len(cid),), gap), step)


def _tracker():
    return ConditionLogZTracker(library_size=10, min_visits=MIN_VISITS)


def test_quorum_reached_on_two_of_four_trainable():
    t = _tracker()
    t.set_trainable([0, 1, 2, 3])
    _characterise(t, [0])
    assert math.isinf(t.delta_stats()['worst'])
    _characterise(t, [1])
    d = t.delta_stats()
    assert d['n'] == 2 and d['worst'] == pytest.approx(0.5)


def test_unset_mask_keeps_the_whole_library_denominator():
    t = _tracker()
    _characterise(t, [0, 1])
    assert math.isinf(t.delta_stats()['worst'])     # 2 < 0.5 * 10
    _characterise(t, [2, 3, 4])
    assert t.delta_stats()['worst'] == pytest.approx(0.5)


def test_characterised_held_out_conditions_do_not_count_toward_the_quorum():
    t = _tracker()
    t.set_trainable([0, 1, 2, 3])
    _characterise(t, [0, 5, 6, 7])
    d = t.delta_stats()
    assert math.isinf(d['worst']) and d['n'] == 4


def test_set_trainable_refuses_empty_and_out_of_range_and_clears_on_none():
    t = _tracker()
    with pytest.raises(ValueError):
        t.set_trainable([])
    with pytest.raises(ValueError):
        t.set_trainable([0, 10])
    t.set_trainable([1])
    t.set_trainable(None)
    assert t.trainable is None


def test_mask_is_not_checkpointed():
    t = _tracker()
    t.set_trainable([0, 1])
    assert ConditionLogZTracker.from_state_dict(t.state_dict()).trainable is None


def _crystal_ef(n_mol, n_sg, n_zp):
    ef = MolecularCrystal.__new__(MolecularCrystal)
    ef.n_sg, ef.n_zp = n_sg, n_zp
    ef.condition_library_size = n_mol * n_sg * n_zp
    return ef


def _modeller(tracker, ef, mol_ids, prior_ids):
    def ds(ids):
        return SimpleNamespace(batch=SimpleNamespace(mol_id=torch.tensor(ids, dtype=torch.long)))
    return SimpleNamespace(energy_function=ef, condition_log_z=tracker,
                           mol_dataset=ds(mol_ids), prior_dataset=ds(prior_ids))


def test_trainer_marks_each_training_molecules_whole_sg_zp_block():
    # 5 molecules x 2 SG; molecules 0, 1 train on both streams, 2..4 are held out
    t = ConditionLogZTracker(library_size=10, min_visits=MIN_VISITS)
    Modeller._set_tracker_trainable(_modeller(t, _crystal_ef(5, 2, 1), [0, 1, 0], [1, 0]))
    assert t.trainable.nonzero().flatten().tolist() == [0, 1, 2, 3]


def test_trainer_leaves_out_a_molecule_only_one_stream_draws():
    # molecule 1 is only in the prior (bwd), molecule 2 only in mol_dataset (fwd):
    # neither can be characterised on both level streams, so neither is trainable
    t = ConditionLogZTracker(library_size=10, min_visits=MIN_VISITS)
    Modeller._set_tracker_trainable(_modeller(t, _crystal_ef(5, 2, 1), [0, 2], [0, 1]))
    assert t.trainable.nonzero().flatten().tolist() == [0, 1]


def test_trainer_leaves_mask_unset_when_the_sources_share_no_molecule():
    t = ConditionLogZTracker(library_size=10, min_visits=MIN_VISITS)
    Modeller._set_tracker_trainable(_modeller(t, _crystal_ef(5, 2, 1), [0], [1]))
    assert t.trainable is None


def test_trainer_leaves_mask_unset_on_a_library_mismatch_or_non_crystal_route():
    t = ConditionLogZTracker(library_size=8, min_visits=MIN_VISITS)
    Modeller._set_tracker_trainable(_modeller(t, _crystal_ef(5, 2, 1), [0], [1]))
    assert t.trainable is None
    t = ConditionLogZTracker(library_size=10, min_visits=MIN_VISITS)
    ef = SimpleNamespace(n_sg=2, n_zp=1, condition_library_size=10)
    Modeller._set_tracker_trainable(_modeller(t, ef, [0], [1]))
    assert t.trainable is None
