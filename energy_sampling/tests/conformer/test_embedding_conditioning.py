"""Molecular identity reaching the policy's condition vector.

Until 2026-09-08 `condition_samples` built the condition from log-temperature or a single
zeros column, and `mol_id` reached only `condition_id`, which the policy never reads -- so the
conformer policy was MOLECULE-BLIND BY CONSTRUCTION. No encoder, however good, could have
changed that. The first test here is the one that would have caught it: two different
molecules must not produce the same condition.
"""
import os

import pytest
import torch

from energies.conformer_torsions import ConformerTorsions

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'models', 'results', 'encoder_ckpt',
    'mp+attn+spd_n20000_s0.pt')
NEEDS_CKPT = pytest.mark.skipif(not os.path.exists(CKPT),
                                reason='pretrained encoder checkpoint not present')

#: two molecules with the SAME k, because a conditions file is one k
PAIR = ['CCCCO', 'CCCCN']
WIDTH = 256                                   # 2 * hidden, the augmented pooled readout


@pytest.fixture(scope='module')
def batch(tmp_path_factory):
    """A two-molecule conditions batch with frozen embeddings baked on."""
    from energies.conformer_data import collate_conditions, condition_from_energy
    from models import encoder_cache
    bundle = encoder_cache.load_encoder(CKPT, device='cpu')
    mols = []
    for smi in PAIR:
        en = ConformerTorsions(smiles=smi, device='cpu', level='torsion')
        mol = condition_from_energy(en, identifier=smi)
        h, g, _ = encoder_cache.embed(bundle, smi, perm=en.spec.perm, z_tree=en.spec.z)
        mol.embedding = g[None, :].to(torch.get_default_dtype())
        mol.atom_embedding = h.to(torch.get_default_dtype())
        mols.append(mol)
    return collate_conditions(mols)


def _energy(**kw):
    return ConformerTorsions(smiles=PAIR[0], device='cpu', level='torsion', **kw)


@NEEDS_CKPT
def test_without_embedding_the_policy_is_molecule_blind(batch):
    """The pre-fix behaviour, pinned so a regression is loud rather than silent.

    This is not a bug being tested -- it is the DEFAULT, and it is correct for an
    unconditional run. It is here so that "the condition does not distinguish molecules" is a
    stated property of that mode rather than something rediscovered later.
    """
    _, _, cond, _ = _energy().condition_samples(batch.clone())
    assert cond.shape == (2, 1)
    assert torch.allclose(cond[0], cond[1])


@NEEDS_CKPT
def test_embedding_conditioning_distinguishes_molecules(batch):
    """The fix: two different molecules must produce different conditions."""
    en = _energy(embedding_conditioning=True, embedding_conditioning_dim=WIDTH)
    _, _, cond, _ = en.condition_samples(batch.clone())
    assert cond.shape == (2, WIDTH)
    assert not torch.allclose(cond[0], cond[1])
    assert float((cond[0] - cond[1]).abs().max()) > 1e-2


@NEEDS_CKPT
def test_temperature_and_embedding_compose_in_order(batch):
    """log-temperature first, then the embedding -- the order `get_conditioning_dim` assumes."""
    en = _energy(temperature_conditioning=True,
                 embedding_conditioning=True, embedding_conditioning_dim=WIDTH)
    _, log_T, cond, _ = en.condition_samples(batch.clone())
    assert cond.shape == (2, 1 + WIDTH)
    assert torch.allclose(cond[:, 0], log_T.float())
    assert torch.allclose(cond[:, 1:], batch.embedding.float())


@NEEDS_CKPT
def test_width_mismatch_is_refused(batch):
    """A conditioner built for a different encoder must fail loudly, not broadcast."""
    en = _energy(embedding_conditioning=True, embedding_conditioning_dim=WIDTH // 2)
    with pytest.raises(RuntimeError, match='embedding width'):
        en.condition_samples(batch.clone())


@NEEDS_CKPT
def test_missing_embedding_names_the_builder(batch):
    """The error has to say how to produce the missing file, not just that it is missing."""
    en = _energy(embedding_conditioning=True, embedding_conditioning_dim=WIDTH)
    plain = batch.clone()
    plain.embedding = None
    with pytest.raises(RuntimeError, match='build_conformer_conditions'):
        en.condition_samples(plain)


def test_dim_is_required_when_conditioning_is_on():
    """Without a width there is nothing to check the file against, so refuse at construction."""
    with pytest.raises(ValueError, match='embedding_conditioning_dim'):
        _energy(embedding_conditioning=True)


@NEEDS_CKPT
def test_atom_embedding_rides_along_per_atom(batch):
    """Per-atom rows are for the n-body heads; they must be per-ATOM, not per-graph."""
    assert batch.atom_embedding.shape[0] == batch.z.shape[0]
    assert batch.atom_embedding.shape[1] * 2 == batch.embedding.shape[1]
