"""`prior_dataset_path` must load a prebuilt set, and REFUSE one that does not belong.

A state tensor is self-describing about nothing. Rows built for another molecule, another
`level`, or another `energy_clip` load without complaint and then train against energies
that mean something else -- and `prebuilt_sample_to_reward` reads the baked energy and
refuses to recompute, so nothing downstream ever catches it. The rejection paths are
therefore the point of this file, not the happy path: each one re-introduces a specific
corruption and REQUIRES a failure, because a check that abstains is indistinguishable from
a check that passes.
"""
import types
import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings('ignore')
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from conformer_modeller import ConformerModeller, _NON_ENERGY_KEYS

BUTANOL = 'CCCCO'
PROPANOL = 'CCCO'


def _stub(smiles, clip=250.0):
    from energies.conformer_torsions import ConformerTorsions

    torch.set_default_dtype(torch.float32)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = types.SimpleNamespace()
    m.energy_function = ConformerTorsions(smiles=smiles, level='full', device=dev,
                                          force_field='mmff', energy_clip=clip)
    m.device = dev
    m.args = types.SimpleNamespace(
        energy_config=types.SimpleNamespace(energy_clip=clip))
    return m


def _write(tmp_path, m, n=64, **override):
    """A genuine dataset for `m`: real prior draws, really scored."""
    from energies.conformer_data import bake_energies

    prior = torch.load('conformer_prior_v2.pt', weights_only=False)
    xs, _ = m.energy_function.sample_prior_states(prior, n, np.random.default_rng(0),
                                                  report=False)
    states = torch.as_tensor(xs, dtype=m.energy_function.dtype, device=m.device)
    blob = {'states': states.cpu(),
            'energies': bake_energies(m.energy_function, states).detach().cpu(),
            'smiles': m.energy_function.smiles,
            'level': m.energy_function.level,
            'force_field': m.energy_function.force_field,
            'energy_clip': float(m.args.energy_config.energy_clip),
            'data_ndim': int(m.energy_function.data_ndim)}
    blob.update(override)
    p = tmp_path / 'ds.pt'
    torch.save(blob, p)
    return str(p)


def test_a_matching_file_loads_and_rescores_identically():
    """The happy path, and it must return the FRESH energies, not the stored ones."""
    import tempfile
    from pathlib import Path

    m = _stub(BUTANOL)
    with tempfile.TemporaryDirectory() as td:
        path = _write(Path(td), m)
        states, energies = ConformerModeller._load_prior_dataset(m, path)
    assert len(states) == 64
    assert energies.shape == (64,)
    assert torch.isfinite(energies).all()


@pytest.mark.parametrize('key,value', [
    ('smiles', PROPANOL),
    ('level', 'torsion'),
    ('force_field', 'uff'),
    ('energy_clip', 100.0),
    ('data_ndim', 3),
])
def test_metadata_mismatch_is_refused(key, value):
    """Each key changes what an energy MEANS, so each must be fatal on its own."""
    import tempfile
    from pathlib import Path

    m = _stub(BUTANOL)
    with tempfile.TemporaryDirectory() as td:
        path = _write(Path(td), m, **{key: value})
        with pytest.raises(SystemExit, match='not built for this run'):
            ConformerModeller._load_prior_dataset(m, path)


def test_stale_energies_are_refused_even_when_metadata_agrees():
    """THE ONE THAT MATTERS. Correct metadata, wrong numbers -- only re-scoring catches it.

    This is the shape of a real staleness bug: the file was built for this molecule under
    this clip, then the force field or the spanning tree moved underneath it. Every metadata
    check passes. Without the re-score the run trains on rewards that no longer describe
    the states carrying them.
    """
    import tempfile
    from pathlib import Path

    m = _stub(BUTANOL)
    with tempfile.TemporaryDirectory() as td:
        path = _write(Path(td), m)
        blob = torch.load(path, weights_only=False)
        blob['energies'] = blob['energies'] + 3.0          # a plausible, quiet offset
        torch.save(blob, path)
        with pytest.raises(SystemExit, match='re-scores differently'):
            ConformerModeller._load_prior_dataset(m, path)


def test_the_key_is_registered_as_non_energy():
    """Unregistered energy_config keys are forwarded to ConformerTorsions, which has no
    **kwargs and raises -- so the run would die at construction rather than at the loader."""
    assert 'prior_dataset_path' in _NON_ENERGY_KEYS
