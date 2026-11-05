"""train.py::generalisation_gaps: each density-readout gap is a readout minus its like-for-like reference.

The held-out-molecule gap reads the training anchors where a pre-noised prior has them: molecules_path may hold only
one crystal per molecule (its lowest), which scores better than a molecule's full minimum set, so the old reference
overstated the gap against a held-out file of anchors.

    python -m pytest -q tests/crystal/test_generalisation_gaps.py
"""
import pytest

from energy_sampling.train import generalisation_gaps

pytestmark = pytest.mark.fast


def test_the_held_out_molecule_gap_reads_the_training_anchors_when_present():
    out = {'eval_test/mle': -0.5, 'eval_bwd/mle_anchors': -0.9, 'eval_bwd/mle_mol_crystals': -1.3}
    assert generalisation_gaps(out)['eval_test/mle_gap'] == pytest.approx(0.4)


def test_without_anchors_it_reads_the_molecules_path_crystals():
    out = {'eval_test/mle': -0.5, 'eval_bwd/mle_mol_crystals': -1.3}
    assert generalisation_gaps(out)['eval_test/mle_gap'] == pytest.approx(0.8)


def test_the_other_gaps_and_a_missing_side():
    out = {'eval_bwd/mle_heldout_rows': -0.7, 'eval_bwd/mle': -0.9, 'eval_bwd/mle_kernel': -1.0,
           'eval_bwd/mle_bank': -1.2}
    g = generalisation_gaps(out, trained='eval_bwd/mle_bank')
    assert g == pytest.approx({'eval_bwd/mle_heldout_gap': 0.2, 'eval_bwd/mle_fresh_gap': 0.2})
