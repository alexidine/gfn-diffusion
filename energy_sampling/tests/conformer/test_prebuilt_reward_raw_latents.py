"""The trainer's replay re-score calls ``prebuilt_sample_to_reward(mols, T, raw_latents=raw)``
on every route (train.py, the replay draw); raw is None on the conformer route. Before
2026-09-30 the conformer energies had no such parameter, so every replay draw on a conformer
run raised TypeError -- conformer_mk's later stage runs replay at 0.1."""
import inspect

import pytest

from energies.conformer_torsions import ConformerTorsions
from energies.multi_conformer import MultiConformerTorsions


@pytest.mark.fast
@pytest.mark.parametrize('cls', [ConformerTorsions, MultiConformerTorsions])
def test_the_conformer_reward_takes_raw_latents_defaulting_to_none(cls):
    p = inspect.signature(cls.prebuilt_sample_to_reward).parameters
    assert 'raw_latents' in p and p['raw_latents'].default is None


@pytest.mark.fast
def test_a_non_none_raw_latents_is_refused():
    class _Mols:
        conformer_energy = None
    with pytest.raises(ValueError, match='raw latents'):
        ConformerTorsions.prebuilt_sample_to_reward(object.__new__(ConformerTorsions), _Mols(), 1.0,
                                                    raw_latents=object())
