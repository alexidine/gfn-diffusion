"""The pre-encoded molecular embedding cache.

Every test here guards a failure that is SILENT -- no shape error, no exception, plausible
numbers -- which is why they exist at all. The atom-order test is the important one: the
encoder and the conformer path genuinely disagree on atom order, so a cache built without
`spec.perm` would condition every atom on another atom's embedding and nothing downstream
would notice.
"""
import numpy as np
import pytest
import torch

import models.encoder_cache as ec

pytestmark = pytest.mark.skipif(
    not __import__('os').path.exists(ec.DEFAULT_CKPT),
    reason='pretrained encoder checkpoint not present')

R, S_ = 'C[C@H](O)CC=O', 'C[C@@H](O)CC=O'


@pytest.fixture(scope='module')
def cache(tmp_path_factory):
    out = tmp_path_factory.mktemp('emb') / 'cache.pt'
    ec.build_cache(['CCCCO', R, S_], ec.DEFAULT_CKPT, str(out), device='cpu')
    return str(out)


@pytest.fixture(scope='module')
def bundle():
    return ec.load_encoder(ec.DEFAULT_CKPT, 'cpu')


def _spec(smiles):
    from energies.conformer_torsions import ConformerTorsions
    return ConformerTorsions(smiles=smiles, device='cpu', level='torsion').spec


def test_embeddings_are_in_tree_order(cache):
    """The cached row order must be the TORSION TREE's, not the encoder's.

    These two orderings really do differ -- `graph_from_smiles` appends every hydrogen while
    the tree interleaves them -- so this asserts the permutation was applied, not merely that
    the shapes line up.
    """
    blob = ec.load_cache(cache, ec.DEFAULT_CKPT)
    spec = _spec('CCCCO')
    h, g = ec.lookup(blob, 'CCCCO', z_tree=spec.z)
    assert h.shape[0] == len(spec.z)
    assert np.array_equal(blob['entries']['CCCCO']['z_tree'].numpy(),
                          np.asarray(spec.z).astype(int))
    # and it is NOT the encoder's order, or the permutation was a no-op and proves nothing
    from models.graph_encodings import graph_from_smiles
    z_enc = np.asarray(graph_from_smiles('CCCCO')[0]).astype(int)
    assert not np.array_equal(z_enc, np.asarray(spec.z).astype(int)), \
        'butanol orders now agree; this test can no longer detect a missing permutation'


def test_enantiomers_get_different_embeddings(cache):
    """Mirror images share an adjacency, so only the parity channel can separate them.

    If this fails the encoder is enantiomer-blind through the cache and every stereochemical
    claim downstream is void -- the same tripwire role `cip_code` plays in the probe battery.
    """
    blob = ec.load_cache(cache, ec.DEFAULT_CKPT)
    hR, gR = ec.lookup(blob, R)
    hS, gS = ec.lookup(blob, S_)
    assert float((hR - hS).abs().max()) > 1e-3
    assert float((gR - gS).abs().max()) > 1e-3


def test_scrambled_permutation_is_refused(bundle):
    """The guard must fire on a wrong permutation, not silently store it."""
    spec = _spec('CCCCO')
    bad = np.asarray(spec.perm).astype(int)[::-1].copy()
    with pytest.raises(ValueError, match='ATOM ORDER MISMATCH'):
        ec.embed(bundle, 'CCCCO', perm=bad, z_tree=spec.z)


def test_wrong_length_permutation_is_refused(bundle):
    spec = _spec('CCCCO')
    with pytest.raises(ValueError, match='spec.perm has'):
        ec.embed(bundle, 'CCCCO', perm=np.arange(3), z_tree=spec.z)


def test_cache_refuses_a_different_encoder(cache, tmp_path):
    """A cache is only meaningful against the encoder that produced it.

    Same discipline the replay buffers grew after an energy-currency mismatch silently
    restored rows in the wrong units.
    """
    other = tmp_path / 'other.pt'
    ck = torch.load(ec.DEFAULT_CKPT, map_location='cpu', weights_only=False)
    ck['state_dict'] = {k: v + 0.0 for k, v in ck['state_dict'].items()}
    ck['_salt'] = 1                       # changes the bytes, hence the fingerprint
    torch.save(ck, other)
    with pytest.raises(ValueError, match='DIFFERENT encoder'):
        ec.load_cache(cache, str(other))


def test_lookup_is_canonical_and_reports_a_miss(cache):
    """Keys are canonical SMILES, so a differently-written input finds the same entry."""
    blob = ec.load_cache(cache, ec.DEFAULT_CKPT)
    h1, _ = ec.lookup(blob, 'CCCCO')
    h2, _ = ec.lookup(blob, 'OCCCC')                    # same molecule, written backwards
    assert torch.equal(h1, h2)
    with pytest.raises(KeyError):
        ec.lookup(blob, 'c1ccccc1')


def test_batch_mismatch_is_caught_at_lookup(cache):
    """Passing a z that does not match the cached row must raise, not return the wrong rows."""
    blob = ec.load_cache(cache, ec.DEFAULT_CKPT)
    spec = _spec('CCCCO')
    wrong = np.asarray(spec.z).astype(int)[::-1].copy()
    with pytest.raises(ValueError, match='cached atom order'):
        ec.lookup(blob, 'CCCCO', z_tree=wrong)
